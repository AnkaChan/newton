# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Self-supervised training: predict S* such that the decoded x_{t+1} minimises
the incremental physics potential."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

from . import torch_solver as ts
from .model import StretchNet, build_face_adjacency, build_features
from .potentials import incremental_potential
from .torch_solver import compute_S_from_x


def vert_to_tet_pin_flag(pinned: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pin_set = set(int(v) for v in pinned)
    flag = np.zeros(tets.shape[0], dtype=np.float32)
    for t in range(tets.shape[0]):
        for k in range(4):
            if int(tets[t, k]) in pin_set:
                flag[t] = 1.0
                break
    return flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--solver-iters", type=int, default=6)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-rollout", type=int, default=4,
                        help="curriculum target: rollout steps per training sample at end of training")
    parser.add_argument("--curriculum-frac", type=float, default=0.5,
                        help="fraction of training over which to ramp rollout 1 -> max-rollout")
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="optional starting checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32  # fp32 for the network; solver upcasts to fp64 internally

    print(f"loading {args.train}")
    data = np.load(args.train)
    rest_q = data["rest_q"]
    tets_np = data["tet_indices"]
    poses_np = data["tet_poses"]
    pinned_np = data["pinned_indices"]
    mass_np = data["particle_mass"]
    mu_np = data["mu_per_tet"]
    lam_np = data["lam_per_tet"]
    x_all = data["x"].astype(np.float32)        # (N, V, 3)
    f_ext_all = data["f_ext"].astype(np.float32)  # (N, V, 3)
    S_all = data["S"].astype(np.float32)        # (N, T, 3, 3)
    traj_start = data["traj_start"]
    fpt = int(data["frames_per_traj"])
    n_total = x_all.shape[0]
    n_traj = traj_start.size

    print(f"train: {n_total} frames, {n_traj} trajectories")

    # For curriculum: build (start_prev, start_t, traj_end_exclusive) windows.
    # At each step we'll pick a rollout length K and use indices i_prev..i_t+K.
    K_max = args.max_rollout
    windows = []  # list of (i_prev, i_t, room) where room = max K possible
    for traj in range(n_traj):
        s = int(traj_start[traj])
        if traj + 1 < n_traj:
            e = int(traj_start[traj + 1])
        else:
            e = n_total
        length = e - s
        if length < 3:
            continue
        # i_prev = s+t-1, i_t = s+t, last reachable = e-1, so K_max_room = e-1-(s+t).
        for t_off in range(1, length - 1):
            i_prev = s + t_off - 1
            i_t = s + t_off
            room = (e - 1) - i_t  # rollout length we can ground-truth (1..K_max)
            if room < 1:
                continue
            windows.append((i_prev, i_t, min(room, K_max)))
    windows = np.array(windows, dtype=np.int64)
    print(f"  {len(windows)} training windows (with up to K={K_max} rollout)")

    # Build solver state and face adjacency.
    solver = ts.build_solver(rest_q, tets_np, poses_np, pinned_np, device=device, dtype=torch.float64)
    face_adj = torch.as_tensor(build_face_adjacency(tets_np), dtype=torch.int64, device=device)

    # Per-tet pin flag (network feature) — boolean tet-has-pinned-vertex.
    pin_flag_tet = torch.as_tensor(vert_to_tet_pin_flag(pinned_np, tets_np), dtype=dtype, device=device)

    # Pre-load tensors.
    mass_t = torch.as_tensor(mass_np, dtype=torch.float64, device=device)
    mu_t = torch.as_tensor(mu_np, dtype=dtype, device=device)
    lam_t = torch.as_tensor(lam_np, dtype=dtype, device=device)
    mu_t64 = mu_t.double()
    lam_t64 = lam_t.double()
    # solver.w stores 1/(6 det(Dm_inv)) = V_rest by construction.
    volume = solver.w.double()
    pinned_t = torch.as_tensor(pinned_np, dtype=torch.int64, device=device)
    pinned_targets_t = torch.as_tensor(rest_q[pinned_np], dtype=torch.float64, device=device)
    gravity_np = data["gravity"]
    gravity64 = torch.as_tensor(gravity_np, dtype=torch.float64, device=device)
    gravity32 = gravity64.to(dtype)

    net = StretchNet().to(device=device, dtype=dtype)
    if args.init_ckpt is not None:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        print(f"loaded init weights from {args.init_ckpt}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)

    # Move all data to GPU once (small enough).
    x_gpu = torch.as_tensor(x_all, dtype=torch.float64, device=device)
    f_ext_gpu = torch.as_tensor(f_ext_all, dtype=torch.float64, device=device)
    S_gpu = torch.as_tensor(S_all, dtype=torch.float64, device=device)

    rng = np.random.default_rng(0)

    log = []
    t0 = time.time()
    curriculum_end = max(1, int(args.steps * args.curriculum_frac))
    for step in range(args.steps):
        # Curriculum schedule for rollout length.
        if step < curriculum_end:
            K_target = 1 + int((args.max_rollout - 1) * step / curriculum_end)
        else:
            K_target = args.max_rollout

        idx = rng.choice(len(windows), size=args.batch, replace=False)
        batch = windows[idx]
        loss_accum = 0.0
        last_pos_err = 0.0
        opt.zero_grad()

        for (i_prev, i_t, room) in batch:
            K = int(min(K_target, room))

            x_prev = x_gpu[i_prev]
            x_t = x_gpu[i_t]
            S_prev = S_gpu[i_prev]
            S_now = S_gpu[i_t]

            L_total_sample = torch.zeros((), dtype=torch.float64, device=device)
            x_pred_acc = None
            for k in range(K):
                i_force = i_t + k  # use force at the current "now" step
                f_ext = f_ext_gpu[i_force]

                S_now_f = S_now.to(dtype=dtype)
                S_prev_f = S_prev.to(dtype=dtype)
                f_ext_f = f_ext.to(dtype=dtype)
                feat = build_features(
                    S_now_f, S_prev_f, gravity32, f_ext_f,
                    mu_t, lam_t, pin_flag_tet, solver.tets, face_adj,
                )
                S_star = net(feat)
                x_next = ts.solve(solver, S_star.double(), pinned_targets_t,
                                  x_init=x_t, n_iters=args.solver_iters)

                losses = incremental_potential(
                    x_next=x_next, x_t=x_t, x_prev=x_prev,
                    mass=mass_t, gravity=gravity64, f_ext=f_ext,
                    tets=solver.tets, J=solver.J,
                    mu=mu_t64, lam=lam_t64, volume=volume,
                    dt=args.dt,
                    pin_idx=pinned_t, pin_target=pinned_targets_t,
                )
                L_total_sample = L_total_sample + losses["total"]

                # Advance state for next rollout step.
                S_prev = S_now
                S_now = compute_S_from_x(solver, x_next)
                x_prev = x_t
                x_t = x_next
                x_pred_acc = x_next

            ((L_total_sample / args.batch) / max(K, 1)).backward()
            loss_accum += float(L_total_sample.item()) / max(K, 1)

            with torch.no_grad():
                x_target_final = x_gpu[int(i_t) + K]
                last_pos_err = (x_pred_acc - x_target_final).norm(dim=-1).mean().item()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            mean_L = loss_accum / args.batch
            print(f"step {step:5d}  K={K_target}  L={mean_L:+.4e}  pos_err={last_pos_err:.4e}  {elapsed:.1f}s")
            log.append({"step": step, "K": K_target, "loss": mean_L, "pos_err": last_pos_err})

    print(f"training done in {time.time()-t0:.1f}s")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "log": log, "args": vars(args)}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
