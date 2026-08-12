# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: torch decoder vs Warp LocalGlobalRecover on forward_run.npz."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import warp as wp

from . import torch_solver as ts
from .recover_local_global import LocalGlobalRecover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")
        + "/Newton/tasks/PrincipalStrecchSolver/data/forward_run.npz",
    )
    parser.add_argument("--frame", type=int, default=60)
    parser.add_argument("--iters", type=int, default=80)
    args = parser.parse_args()

    wp.init()
    data = np.load(args.data)
    rest_q = data["rest_q"]
    tets = data["tet_indices"]
    poses = data["tet_poses"]
    pinned = data["pinned_indices"]
    x_gt = data["x"][args.frame]
    S_gt = data["S"][args.frame]

    # Warp reference
    rec = LocalGlobalRecover(rest_q, tets, poses, pinned, device="cuda:0")
    res = rec.solve(S_target=S_gt, pinned_targets=x_gt[pinned], x_init=None, max_iters=args.iters, tol=0.0)
    x_warp = res.x

    # Torch port
    dev = torch.device("cuda:0")
    state = ts.build_solver(rest_q, tets, poses, pinned, device=dev, dtype=torch.float64)
    S_t = torch.as_tensor(S_gt, dtype=torch.float64, device=dev)
    pin_t = torch.as_tensor(x_gt[pinned], dtype=torch.float64, device=dev)
    x_torch = ts.solve(state, S_t, pin_t, x_init=None, n_iters=args.iters).cpu().numpy()

    err = np.linalg.norm(x_torch.astype(np.float32) - x_warp, axis=1)
    err_gt = np.linalg.norm(x_torch.astype(np.float32) - x_gt, axis=1)

    print(f"frame {args.frame}, {args.iters} iters")
    print(f"  ||x_torch - x_warp||  mean={err.mean():.3e}  max={err.max():.3e}")
    print(f"  ||x_torch - x_GT||    mean={err_gt.mean():.3e}  max={err_gt.max():.3e}")

    # Also test gradient flow
    print("\nautograd check:")
    S_req = torch.as_tensor(S_gt, dtype=torch.float64, device=dev).clone().requires_grad_(True)
    x_out = ts.solve(state, S_req, pin_t, x_init=None, n_iters=6)
    loss = x_out.pow(2).sum()
    loss.backward()
    print(f"  loss={loss.item():.3e}   grad_S norm={S_req.grad.norm().item():.3e}")

    return 0 if err.max() < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
