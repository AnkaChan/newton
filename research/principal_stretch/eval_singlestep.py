# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced single-step eval: feed GT (x_{t-1}, x_t) and predict x_{t+1}."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .predictor import build_stretch_predictor, checkpoint_predictor_config, load_stretch_predictor_state
from .rollout import vert_to_tet_pin_flag
from .torch_solver import compute_S_from_x, inertial_predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--solver-iters", type=int, default=10)
    p.add_argument("--out", help="optional NPZ with per-sample and per-trajectory errors")
    args = p.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.float32

    d = np.load(args.data)
    rest_q = d["rest_q"]
    tets_np = d["tet_indices"]
    poses_np = d["tet_poses"]
    pinned_np = d["pinned_indices"]
    mu_np = d["mu_per_tet"]
    lam_np = d["lam_per_tet"]
    x_all = d["x"]
    f_ext_all = d["f_ext"]
    traj_start = d["traj_start"]
    gravity_np = d["gravity"]
    n_total = x_all.shape[0]
    n_traj = traj_start.size

    state = ts.build_solver(rest_q, tets_np, poses_np, pinned_np, device=device, dtype=torch.float64)
    mu_t = torch.as_tensor(mu_np, dtype=dtype, device=device)
    lam_t = torch.as_tensor(lam_np, dtype=dtype, device=device)
    pin_flag = torch.as_tensor(vert_to_tet_pin_flag(pinned_np, tets_np), dtype=dtype, device=device)
    pinned_targets = torch.as_tensor(rest_q[pinned_np], dtype=torch.float64, device=device)
    gravity = torch.as_tensor(gravity_np, dtype=torch.float64, device=device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor_config = checkpoint_predictor_config(ckpt)
    predictor = build_stretch_predictor(
        predictor_config["kind"],
        rest_q,
        tets_np,
        device,
        dtype,
        residual=bool(predictor_config.get("residual", False)),
        graph_config=predictor_config.get("graph_transformer"),
    )
    load_stretch_predictor_state(predictor, ckpt)
    predictor.eval()
    ckpt_args = ckpt.get("args", {})
    warm = ckpt_args.get("warm", "prev")
    blocks = int(ckpt_args.get("blocks", 1))
    print(f"ckpt config: predictor={predictor.kind} warm={warm} blocks={blocks} solver_iters={args.solver_iters}")

    errs = []
    sample_trajectory = []
    sample_frame = []
    with torch.no_grad():
        for traj in range(n_traj):
            s = int(traj_start[traj])
            e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
            for t in range(s + 1, e - 1):
                x_prev = torch.as_tensor(x_all[t - 1], dtype=torch.float64, device=device)
                x_t = torch.as_tensor(x_all[t], dtype=torch.float64, device=device)
                x_target = torch.as_tensor(x_all[t + 1], dtype=torch.float64, device=device)
                f_ext = torch.as_tensor(f_ext_all[t], dtype=torch.float64, device=device)
                S_previous = compute_S_from_x(state, x_prev)
                x0 = inertial_predictor(state, x_t, x_prev, pinned_targets) if warm == "inertial" else x_t
                iters_per_block = max(1, args.solver_iters // blocks)
                x_next = x0
                S_cur = compute_S_from_x(state, x_t)
                for _b in range(blocks):
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
                        S_previous,
                    )
                    x_next = ts.solve(state, S_star.double(), pinned_targets, x_init=x_next, n_iters=iters_per_block)
                    if _b + 1 < blocks:
                        S_cur = compute_S_from_x(state, x_next)
                e_per_v = (x_next - x_target).norm(dim=-1)
                errs.append(e_per_v.cpu().numpy())
                sample_trajectory.append(traj)
                sample_frame.append(t - s)

    error = np.stack(errs)
    E = error.reshape(-1)
    print(f"Teacher-forced single-step eval on {n_traj} trajs, {len(errs)} samples")
    print(f"  per-vertex mean = {E.mean():.4e} m")
    print(f"  per-vertex 95%  = {np.quantile(E, 0.95):.4e} m")
    print(f"  per-vertex max  = {E.max():.4e} m")
    if args.out:
        trajectory_index = np.asarray(sample_trajectory, dtype=np.int64)
        trajectory_mean = np.asarray([error[trajectory_index == traj].mean() for traj in range(n_traj)])
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            error=error,
            trajectory_index=trajectory_index,
            frame_index=np.asarray(sample_frame, dtype=np.int64),
            trajectory_mean=trajectory_mean,
            solver_iters=args.solver_iters,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
