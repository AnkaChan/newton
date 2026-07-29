# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: how many local-global iterations does the decoder actually need?

Phase 1 reported ~1e-6 m recovery, but at **500** iterations.  Phase 2 unrolls
**6** iterations during training and runs **10** at eval.  This script feeds the
decoder the exact stretches of the GT next frame (so the true minimum has
energy exactly zero and sits at the GT positions) and sweeps the iteration
count, warm-started at ``x_t``.  Whatever error remains is pure local-global
truncation and is a hard floor under every Phase 2 number.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .torch_solver import compute_S_from_x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    d = np.load(args.data)
    rest_q = d["rest_q"]
    tets_np = d["tet_indices"]
    poses_np = d["tet_poses"]
    pinned_np = d["pinned_indices"]
    x_all = d["x"]
    traj_start = d["traj_start"]
    n_total = x_all.shape[0]
    n_traj = traj_start.size

    state = ts.build_solver(rest_q, tets_np, poses_np, pinned_np, device=device, dtype=torch.float64)
    pinned_targets = torch.as_tensor(rest_q[pinned_np], dtype=torch.float64, device=device)

    cand = []
    for traj in range(n_traj):
        s = int(traj_start[traj])
        e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
        cand += list(range(s + 1, e - 1))
    rng = np.random.default_rng(0)
    picks = rng.choice(len(cand), size=min(args.samples, len(cand)), replace=False)

    iters_list = [1, 2, 4, 6, 10, 20, 50, 100, 200, 500, 1000]
    warm_modes = ["x_t", "inertial"]
    res = {(w, k): [] for w in warm_modes for k in iters_list}

    with torch.no_grad():
        for pi in picks:
            t = cand[int(pi)]
            x_prev = torch.as_tensor(x_all[t - 1], dtype=torch.float64, device=device)
            x_t = torch.as_tensor(x_all[t], dtype=torch.float64, device=device)
            x_gt = torch.as_tensor(x_all[t + 1], dtype=torch.float64, device=device)
            S_exact = compute_S_from_x(state, x_gt)
            for w in warm_modes:
                x0 = x_t if w == "x_t" else (2.0 * x_t - x_prev)
                for k in iters_list:
                    xd = ts.solve(state, S_exact, pinned_targets, x_init=x0, n_iters=k)
                    res[(w, k)].append((xd - x_gt).norm(dim=-1).mean().item())

    print(f"\nDecoder convergence with EXACT target stretches ({len(picks)} windows)")
    print("mean vertex error vs GT x_{t+1} [m]; the true minimiser is x_gt with zero energy\n")
    print(f"{'iters':>6} | {'warm=x_t':>12} | {'warm=inertial':>14} | {'rate/iter':>10}")
    print("-" * 52)
    prev = None
    for k in iters_list:
        a = np.mean(res[("x_t", k)])
        b = np.mean(res[("inertial", k)])
        rate = ""
        if prev is not None and prev[1] > 0:
            n = k - prev[0]
            rate = f"{(a / prev[1]) ** (1.0 / n):.4f}"
        print(f"{k:6d} | {a:12.4e} | {b:14.4e} | {rate:>10}")
        prev = (k, a)

    a6 = np.mean(res[("x_t", 6)])
    a500 = np.mean(res[("x_t", 500)])
    print(f"\ntraining used 6 iters  -> {a6:.3e} m")
    print(f"eval used 10 iters     -> {np.mean(res[('x_t', 10)]):.3e} m")
    print(f"Phase 1 used 500 iters -> {a500:.3e} m")
    print(f"ratio (6 iters / 500 iters) = {a6 / max(a500, 1e-30):.1f}x")


if __name__ == "__main__":
    sys.exit(main())
