# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: trivial baselines the Phase 2 numbers were never compared against.

Single-step (teacher-forced):
  * do-nothing            x_{t+1} = x_t
  * constant velocity     x_{t+1} = 2 x_t - x_{t-1}      (zero-cost, no learning)
  * exact backward Euler  argmin of the training objective
  * trained network       for reference

Rollout (autoregressive, same trajectories as the reported MP4s):
  * exact backward-Euler integrator at the loss's dt, stepped 18 times.
    This is the best any model trained on this objective could possibly do,
    so it separates "objective is wrong" from "network is bad".
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .diag_floors import minimise_potential
from .model import StretchNet, build_face_adjacency, build_features
from .rollout import vert_to_tet_pin_flag
from .torch_solver import compute_S_from_x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--rollout-trajs", type=int, nargs="*", default=[0, 1, 5, 10, 15])
    p.add_argument("--rollout-steps", type=int, default=18)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    d = np.load(args.data)
    rest_q = d["rest_q"]
    tets_np = d["tet_indices"]
    poses_np = d["tet_poses"]
    pinned_np = d["pinned_indices"]
    mass_np = d["particle_mass"]
    mu_np = d["mu_per_tet"]
    lam_np = d["lam_per_tet"]
    x_all = d["x"]
    f_ext_all = d["f_ext"]
    traj_start = d["traj_start"]
    gravity_np = d["gravity"]
    n_total = x_all.shape[0]
    n_traj = traj_start.size

    state = ts.build_solver(rest_q, tets_np, poses_np, pinned_np, device=device, dtype=torch.float64)
    mass = torch.as_tensor(mass_np, dtype=torch.float64, device=device)
    mu = torch.as_tensor(mu_np, dtype=torch.float64, device=device)
    lam = torch.as_tensor(lam_np, dtype=torch.float64, device=device)
    volume = state.w.double()
    gravity = torch.as_tensor(gravity_np, dtype=torch.float64, device=device)
    pinned_targets = torch.as_tensor(rest_q[pinned_np], dtype=torch.float64, device=device)

    net = None
    if args.ckpt:
        face_adj = torch.as_tensor(build_face_adjacency(tets_np), dtype=torch.int64, device=device)
        pin_flag = torch.as_tensor(vert_to_tet_pin_flag(pinned_np, tets_np), dtype=torch.float32, device=device)
        mu32 = mu.float()
        lam32 = lam.float()
        g32 = gravity.float()
        net = StretchNet().to(device=device, dtype=torch.float32)
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        net.eval()

    # ---------- single-step baselines ----------
    cand = []
    for traj in range(n_traj):
        s = int(traj_start[traj])
        e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
        cand += list(range(s + 1, e - 1))
    rng = np.random.default_rng(1)
    picks = rng.choice(len(cand), size=min(args.samples, len(cand)), replace=False)

    err = {k: [] for k in ("none", "cv", "net")}
    with torch.no_grad():
        for pi in picks:
            t = cand[int(pi)]
            x_prev = torch.as_tensor(x_all[t - 1], dtype=torch.float64, device=device)
            x_t = torch.as_tensor(x_all[t], dtype=torch.float64, device=device)
            x_gt = torch.as_tensor(x_all[t + 1], dtype=torch.float64, device=device)
            err["none"].append((x_t - x_gt).norm(dim=-1).mean().item())
            cv = 2.0 * x_t - x_prev
            cv[state.pinned] = pinned_targets
            err["cv"].append((cv - x_gt).norm(dim=-1).mean().item())
            if net is not None:
                f_ext = torch.as_tensor(f_ext_all[t], dtype=torch.float64, device=device)
                S_t = compute_S_from_x(state, x_t).float()
                S_p = compute_S_from_x(state, x_prev).float()
                feat = build_features(S_t, S_p, g32, f_ext.float(), mu32, lam32, pin_flag, state.tets, face_adj)
                xn = ts.solve(state, net(feat).double(), pinned_targets, x_init=x_t, n_iters=10)
                err["net"].append((xn - x_gt).norm(dim=-1).mean().item())

    print(f"\n=== single-step, teacher-forced, {len(picks)} windows ===")
    for k, label in (
        ("none", "do nothing  x_{t+1}=x_t"),
        ("cv", "const. velocity 2x_t-x_{t-1}"),
        ("net", "trained network + decoder"),
    ):
        if err[k]:
            a = np.array(err[k])
            print(f"  {label:32s} mean={a.mean():.4e}  median={np.median(a):.4e}  95%={np.quantile(a, 0.95):.4e}")

    # ---------- rollout: exact BE integrator ----------
    print(
        f"\n=== {args.rollout_steps}-step autoregressive rollout of the EXACT "
        f"backward-Euler integrator at dt={args.dt:.5f} ==="
    )
    print("(this is the ceiling for anything trained on this objective)\n")
    all_final, all_mean = [], []
    for traj in args.rollout_trajs:
        if traj >= n_traj:
            continue
        s = int(traj_start[traj])
        e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
        n_steps = min(args.rollout_steps, (e - s) - 2)
        x_prev = torch.as_tensor(x_all[s], dtype=torch.float64, device=device)
        x_t = torch.as_tensor(x_all[s + 1], dtype=torch.float64, device=device)
        errs = []
        for k in range(n_steps):
            f_ext = torch.as_tensor(f_ext_all[s + 1 + k], dtype=torch.float64, device=device)
            x_next, _ = minimise_potential(
                state, x_t, x_prev, f_ext, mass, gravity, mu, lam, volume, pinned_targets, args.dt
            )
            x_gt = torch.as_tensor(x_all[s + 2 + k], dtype=torch.float64, device=device)
            errs.append((x_next - x_gt).norm(dim=-1).mean().item())
            x_prev, x_t = x_t, x_next
        errs = np.array(errs)
        all_final.append(errs[-1])
        all_mean.append(errs.mean())
        print(f"  traj {traj:3d}: step-1 err={errs[0]:.4e}  mean={errs.mean():.4e}  final={errs[-1]:.4e}")
    if all_final:
        print(f"\n  BE integrator   : mean={np.mean(all_mean):.4e}  final-frame={np.mean(all_final):.4e}")
        print("  (compare: best trained model mean=6.15e-02, final-frame=1.37e-01)")


if __name__ == "__main__":
    sys.exit(main())
