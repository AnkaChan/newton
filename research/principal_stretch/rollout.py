# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive rollout: given (x_0, x_1) and a force schedule, run the
trained network + differentiable decoder for N steps. Compare to FEM."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .predictor import (
    build_stretch_predictor,
    checkpoint_predictor_config,
    decode_predictor_step,
    load_stretch_predictor_state,
    predictor_architecture_version,
    predictor_decoder_work,
    resolve_solver_iterations,
    validate_static_pin_trajectory,
)
from .torch_solver import compute_S_from_x, inertial_predictor


def vert_to_tet_pin_flag(pinned, tets):
    pin_set = {int(v) for v in pinned}
    flag = np.zeros(tets.shape[0], dtype=np.float32)
    for t in range(tets.shape[0]):
        for k in range(4):
            if int(tets[t, k]) in pin_set:
                flag[t] = 1.0
                break
    return flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="val or train npz with trajectories")
    parser.add_argument("--traj", type=int, default=0)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument(
        "--solver-iters",
        type=int,
        default=None,
        help="legacy local-global sweeps (default 10); v3 requires its single projection",
    )
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.float32

    data = np.load(args.data)
    rest_q = data["rest_q"]
    tets_np = data["tet_indices"]
    poses_np = data["tet_poses"]
    pinned_np = data["pinned_indices"]
    mu_np = data["mu_per_tet"]
    lam_np = data["lam_per_tet"]
    x_all = data["x"]
    f_ext_all = data["f_ext"]
    traj_start = data["traj_start"]
    gravity_np = data["gravity"]

    n_traj = traj_start.size
    s = int(traj_start[args.traj])
    e = int(traj_start[args.traj + 1]) if args.traj + 1 < n_traj else x_all.shape[0]
    traj_len = e - s
    n_steps = min(args.steps, traj_len - 2)

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
    if predictor_architecture_version(predictor) == 3:
        validate_static_pin_trajectory(rest_q, pinned_np, x_all)
    # Inference must match the training configuration.
    ckpt_args = ckpt.get("args", {})
    warm = ckpt_args.get("warm", "prev")
    blocks = int(ckpt_args.get("blocks", 1))
    solver_iterations = resolve_solver_iterations(predictor, args.solver_iters)
    decoder_work = predictor_decoder_work(predictor, solver_iterations, blocks)
    full_gradient_decoder = decoder_work["target"] == "full-deformation-gradient"
    if predictor_architecture_version(predictor) == 3:
        saved_work = ckpt.get("decoder_work")
        if saved_work != decoder_work:
            raise ValueError(
                "architecture-v3 checkpoint decoder_work is missing or inconsistent; "
                f"saved={saved_work!r}, expected={decoder_work!r}"
            )
    print(
        f"ckpt config: predictor={predictor.kind} warm={warm} blocks={blocks} "
        f"solver_iters={solver_iterations} decoder_work={decoder_work}"
    )

    # Seed: GT first 2 frames
    x_prev = torch.as_tensor(x_all[s], dtype=torch.float64, device=device)
    x_t = torch.as_tensor(x_all[s + 1], dtype=torch.float64, device=device)
    if full_gradient_decoder:
        S_prev = None
        S_t = None
    else:
        S_prev = compute_S_from_x(state, x_prev)
        S_t = compute_S_from_x(state, x_t)

    x_pred = [x_prev.cpu().numpy(), x_t.cpu().numpy()]
    x_gt = [x_all[s], x_all[s + 1]]

    with torch.no_grad():
        for step in range(n_steps):
            i_t = s + 1 + step
            f_ext = torch.as_tensor(f_ext_all[i_t], dtype=torch.float64, device=device)
            if full_gradient_decoder:
                x0 = None
            elif warm == "inertial":
                x0 = inertial_predictor(state, x_t, x_prev, pinned_targets)
            else:
                x0 = x_t
            x_next = decode_predictor_step(
                predictor,
                state,
                x_t,
                x_prev,
                f_ext,
                gravity,
                mu_t,
                lam_t,
                pin_flag,
                S_t,
                S_prev,
                pinned_targets,
                x_init=x0,
                solver_iterations=solver_iterations,
                blocks=blocks,
            )

            x_pred.append(x_next.cpu().numpy())
            x_gt.append(x_all[s + 2 + step] if (s + 2 + step) < e else x_all[e - 1])

            if not full_gradient_decoder:
                S_prev = S_t
                S_t = compute_S_from_x(state, x_next)
            x_prev = x_t
            x_t = x_next

    x_pred = np.stack(x_pred)
    x_gt = np.stack(x_gt)
    err = np.linalg.norm(x_pred - x_gt, axis=-1)  # (F, V)
    print(f"rollout {x_pred.shape[0]} frames, traj {args.traj}")
    for f in range(0, x_pred.shape[0], max(1, x_pred.shape[0] // 6)):
        print(f"  frame {f:3d}  mean_err={err[f].mean():.4e}  max_err={err[f].max():.4e}")
    generated_err = err[2:]
    print(f"  generated mean={generated_err.mean():.4e}  max={generated_err.max():.4e}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save in record_side_by_side-compatible format (x_gt, x_rec).
    np.savez_compressed(
        out,
        x_rec=x_pred,
        x_gt=x_gt,
        err=err,
        traj=args.traj,
        solver_iters=solver_iterations,
        decoder=np.asarray(decoder_work["decoder"]),
        global_triangular_solves=np.asarray(decoder_work["global_triangular_solves"], dtype=np.int64),
        local_polar_sweeps=np.asarray(decoder_work["local_polar_sweeps"], dtype=np.int64),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
