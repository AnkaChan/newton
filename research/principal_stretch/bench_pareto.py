# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Accuracy-vs-wall-clock Pareto: SolverVBD budget grid vs the learned pipeline.

This is the Phase B claim's measurement instrument (plan B0.5).  Both sides
replay the *same* recorded force schedules from the validation set and are
scored against the recorded reference trajectory (generated at 10 substeps x
10 VBD iterations):

- **VBD side:** re-simulate each trajectory from the same rest start at a
  (substeps x iterations) budget grid.  At the reference budget this replays
  the generator and its error is ~0 by construction (determinism sanity
  anchor); smaller budgets show real degradation.
- **Learned side:** autoregressive rollout of a trained checkpoint at several
  decoder iteration counts, seeded with the first two reference frames.

Per config: mean ms/frame (CUDA-synchronised, warmup excluded) and
mean/final-frame vertex error over the scored frames.  Results go to a JSON
for plotting.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import warp as wp

from newton.solvers import SolverVBD

from . import torch_solver as ts
from .predictor import build_stretch_predictor, checkpoint_predictor_config, load_stretch_predictor_state
from .rollout import vert_to_tet_pin_flag
from .run_forward import build_model
from .torch_solver import compute_S_from_x, inertial_predictor


def validate_legacy_net_checkpoint(predictor_config: dict) -> None:
    """Reject full-gradient checkpoints from this legacy trajectory benchmark."""
    if predictor_config.get("kind") != "graph-transformer":
        return
    graph_config = predictor_config.get("graph_transformer", {})
    if int(graph_config.get("architecture_version", 0)) >= 3:
        raise ValueError(
            "bench_pareto evaluates the legacy right-stretch/local-global pipeline and cannot score "
            "architecture-v3 full-gradient checkpoints; use the common-objective solver benchmark"
        )


def vbd_curve(data, trajs, grid, dims, frame_dt, device="cuda:0"):
    wp.init()
    results = []
    for substeps, iterations in grid:
        errs_mean, errs_final, ms = [], [], []
        for traj in trajs:
            s = int(data["traj_start"][traj])
            e = int(data["traj_start"][traj + 1]) if traj + 1 < data["traj_start"].size else data["x"].shape[0]
            n_frames = e - s

            model, _builder = build_model(dim_x=dims[0], dim_y=dims[1], dim_z=dims[2])
            solver = SolverVBD(
                model=model,
                iterations=iterations,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )
            state_0, state_1 = model.state(), model.state()
            control, contacts = model.control(), model.contacts()
            sim_dt = frame_dt / substeps

            frame_errs, frame_ms = [], []
            for f in range(n_frames):
                f_ext_frame = data["f_ext"][s + f]
                wp.synchronize()
                t0 = time.perf_counter()
                for _ in range(substeps):
                    state_0.clear_forces()
                    state_0.particle_f.assign(f_ext_frame)
                    model.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0
                wp.synchronize()
                frame_ms.append((time.perf_counter() - t0) * 1e3)
                x = state_0.particle_q.numpy()
                frame_errs.append(np.linalg.norm(x - data["x"][s + f], axis=-1).mean())
            # Score the same frames the net is scored on (2..end); skip frame 0-1 warmup timing.
            errs_mean.append(np.mean(frame_errs[2:]))
            errs_final.append(frame_errs[-1])
            ms.append(np.mean(frame_ms[2:]))
        results.append(
            {
                "method": "vbd",
                "config": f"s{substeps}xi{iterations}",
                "substeps": substeps,
                "iterations": iterations,
                "ms_per_frame": float(np.mean(ms)),
                "err_mean": float(np.mean(errs_mean)),
                "err_final": float(np.mean(errs_final)),
            }
        )
        print(
            f"vbd s{substeps:2d} x i{iterations:2d}: {results[-1]['ms_per_frame']:8.2f} ms/frame  "
            f"err mean={results[-1]['err_mean']:.4e}  final={results[-1]['err_final']:.4e}",
            flush=True,
        )
    return results


def net_curve(data, trajs, ckpt_path, iters_list, frame_dt, device="cuda:0"):
    dev = torch.device(device)
    dtype = torch.float32
    rest_q = data["rest_q"]
    tets_np = data["tet_indices"]
    state = ts.build_solver(rest_q, tets_np, data["tet_poses"], data["pinned_indices"], device=dev, dtype=torch.float64)
    mu_t = torch.as_tensor(data["mu_per_tet"], dtype=dtype, device=dev)
    lam_t = torch.as_tensor(data["lam_per_tet"], dtype=dtype, device=dev)
    pin_flag = torch.as_tensor(vert_to_tet_pin_flag(data["pinned_indices"], tets_np), dtype=dtype, device=dev)
    pinned_targets = torch.as_tensor(rest_q[data["pinned_indices"]], dtype=torch.float64, device=dev)
    gravity = torch.as_tensor(data["gravity"], dtype=torch.float64, device=dev)

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    predictor_config = checkpoint_predictor_config(ckpt)
    validate_legacy_net_checkpoint(predictor_config)
    predictor = build_stretch_predictor(
        predictor_config["kind"],
        rest_q,
        tets_np,
        dev,
        dtype,
        residual=bool(predictor_config.get("residual", False)),
        graph_config=predictor_config.get("graph_transformer"),
    )
    load_stretch_predictor_state(predictor, ckpt)
    predictor.eval()
    ckpt_args = ckpt.get("args", {})
    warm = ckpt_args.get("warm", "prev")
    blocks = int(ckpt_args.get("blocks", 1))

    results = []
    for n_iters in iters_list:
        errs_mean, errs_final, ms = [], [], []
        with torch.no_grad():
            for traj in trajs:
                s = int(data["traj_start"][traj])
                e = int(data["traj_start"][traj + 1]) if traj + 1 < data["traj_start"].size else data["x"].shape[0]
                x_prev = torch.as_tensor(data["x"][s], dtype=torch.float64, device=dev)
                x_t = torch.as_tensor(data["x"][s + 1], dtype=torch.float64, device=dev)
                S_prev = compute_S_from_x(state, x_prev)
                S_t = compute_S_from_x(state, x_t)
                frame_errs, frame_ms = [], []
                for step in range(e - s - 2):
                    f_ext = torch.as_tensor(data["f_ext"][s + 1 + step], dtype=torch.float64, device=dev)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    x0 = inertial_predictor(state, x_t, x_prev, pinned_targets) if warm == "inertial" else x_t
                    x_next = x0
                    S_cur = S_t
                    for block in range(blocks):
                        S_star = predictor(
                            state,
                            x_t,
                            x_prev,
                            f_ext,
                            gravity,
                            mu_t,
                            lam_t,
                            pin_flag,
                            S_cur,
                            S_prev,
                        )
                        x_next = ts.solve(
                            state,
                            S_star.double(),
                            pinned_targets,
                            x_init=x_next,
                            n_iters=max(1, n_iters // blocks),
                        )
                        if block + 1 < blocks:
                            S_cur = compute_S_from_x(state, x_next)
                    S_new = compute_S_from_x(state, x_next)
                    torch.cuda.synchronize()
                    frame_ms.append((time.perf_counter() - t0) * 1e3)
                    x_gt = data["x"][s + 2 + step]
                    frame_errs.append(float(np.linalg.norm(x_next.cpu().numpy() - x_gt, axis=-1).mean()))
                    S_prev, S_t = S_t, S_new
                    x_prev, x_t = x_t, x_next
                errs_mean.append(np.mean(frame_errs))
                errs_final.append(frame_errs[-1])
                ms.append(np.mean(frame_ms[1:]))  # drop first frame (allocator warmup)
        results.append(
            {
                "method": "net",
                "predictor": predictor.kind,
                "config": f"{predictor.kind}-it{n_iters}",
                "decoder_iters": n_iters,
                "ms_per_frame": float(np.mean(ms)),
                "err_mean": float(np.mean(errs_mean)),
                "err_final": float(np.mean(errs_final)),
            }
        )
        print(
            f"net it{n_iters:3d}:       {results[-1]['ms_per_frame']:8.2f} ms/frame  "
            f"err mean={results[-1]['err_mean']:.4e}  final={results[-1]['err_final']:.4e}",
            flush=True,
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="val npz (x, f_ext, topology)")
    p.add_argument("--ckpt", default=None, help="trained checkpoint; omit for VBD-only pass")
    p.add_argument("--dims", type=int, nargs=3, default=(24, 12, 12))
    p.add_argument("--trajs", type=int, nargs="*", default=[0, 1, 5, 10, 15])
    p.add_argument("--frame-dt", type=float, default=1.0 / 60.0)
    p.add_argument("--net-iters", type=int, nargs="*", default=[4, 6, 10, 20])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = np.load(args.data)
    grid = [(1, 2), (1, 5), (1, 10), (2, 5), (2, 10), (5, 10), (10, 10)]

    results = vbd_curve(data, args.trajs, grid, tuple(args.dims), args.frame_dt)
    if args.ckpt:
        results += net_curve(data, args.trajs, args.ckpt, args.net_iters, args.frame_dt)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
