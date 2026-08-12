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
from .hier_model import HierStretchNet
from .hierarchy import build_hierarchy
from .model import StretchNet, build_face_adjacency, build_features
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
    parser.add_argument("--solver-iters", type=int, default=10)
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
    face_adj = torch.as_tensor(build_face_adjacency(tets_np), dtype=torch.int64, device=device)
    mu_t = torch.as_tensor(mu_np, dtype=dtype, device=device)
    lam_t = torch.as_tensor(lam_np, dtype=dtype, device=device)
    pin_flag = torch.as_tensor(vert_to_tet_pin_flag(pinned_np, tets_np), dtype=dtype, device=device)
    pinned_targets = torch.as_tensor(rest_q[pinned_np], dtype=torch.float64, device=device)
    gravity32 = torch.as_tensor(gravity_np, dtype=dtype, device=device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Inference must match the training configuration.
    ckpt_args = ckpt.get("args", {})
    residual = bool(ckpt_args.get("residual", False))
    warm = ckpt_args.get("warm", "prev")
    blocks = int(ckpt_args.get("blocks", 1))
    hier = bool(ckpt_args.get("hier", False))
    if hier:
        # The hier net recomputes its pooled state from (x_t, x_prev) once per
        # step; the block loop below only refreshes S_cur / feat between blocks.
        assert blocks == 1, "hier checkpoints with blocks>1 are not supported by this harness"
        # Same article -> deterministic same hierarchy as at training time; the
        # constructor args live in the checkpoint's hier_config, not the state_dict.
        cfg = ckpt["hier_config"]
        hierarchy = build_hierarchy(tets_np, rest_q, n_levels=cfg["n_levels"], target=cfg["target"])
        net = HierStretchNet(
            hierarchy,
            hidden=cfg["hidden"],
            mp_rounds=cfg["mp_rounds"],
            delta_fine=cfg["delta_fine"],
            delta_coarse=cfg["delta_coarse"],
        ).to(device)
    else:
        net = StretchNet().to(device=device, dtype=dtype)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"ckpt config: residual={residual} warm={warm} blocks={blocks}" + (" hier=True" if hier else ""))

    # Seed: GT first 2 frames
    x_prev = torch.as_tensor(x_all[s], dtype=torch.float64, device=device)
    x_t = torch.as_tensor(x_all[s + 1], dtype=torch.float64, device=device)
    S_prev = compute_S_from_x(state, x_prev)
    S_t = compute_S_from_x(state, x_t)

    x_pred = [x_prev.cpu().numpy(), x_t.cpu().numpy()]
    x_gt = [x_all[s], x_all[s + 1]]

    with torch.no_grad():
        for step in range(n_steps):
            i_t = s + 1 + step
            f_ext = torch.as_tensor(f_ext_all[i_t], dtype=torch.float64, device=device)
            x0 = inertial_predictor(state, x_t, x_prev, pinned_targets) if warm == "inertial" else x_t
            iters_per_block = max(1, args.solver_iters // blocks)
            x_next = x0
            S_cur = S_t
            S_prev_f = S_prev.to(dtype=dtype)
            for _b in range(blocks):
                S_cur_f = S_cur.to(dtype=dtype)
                feat = build_features(
                    S_cur_f,
                    S_prev_f,
                    gravity32,
                    f_ext.to(dtype=dtype),
                    mu_t,
                    lam_t,
                    pin_flag,
                    state.tets,
                    face_adj,
                )
                if hier:
                    S_star = net(state, x_t, x_prev, f_ext, feat, S_cur).double()
                else:
                    S_star = net(feat, S_base=S_cur_f if residual else None).double()
                x_next = ts.solve(state, S_star, pinned_targets, x_init=x_next, n_iters=iters_per_block)
                if _b + 1 < blocks:
                    S_cur = compute_S_from_x(state, x_next)

            x_pred.append(x_next.cpu().numpy())
            x_gt.append(x_all[s + 2 + step] if (s + 2 + step) < e else x_all[e - 1])

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
    print(f"  overall mean={err.mean():.4e}  max={err.max():.4e}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save in record_side_by_side-compatible format (x_gt, x_rec).
    np.savez_compressed(out, x_rec=x_pred, x_gt=x_gt, err=err, traj=args.traj)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
