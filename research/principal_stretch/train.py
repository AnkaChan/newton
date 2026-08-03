# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Train the stretch predictor through the batched differentiable decoder.

Supersedes the original per-sample-loop trainer and folds in the repairs from
the 2026-07-28 takeover review:

- **Batched decoder** — one ``solve()`` call per rollout step for the whole
  batch (the mesh, and therefore the Cholesky factor, is shared).
- **Inertial warm start** (``--warm inertial``) — decode from ``2 x_t - x_prev``
  instead of ``x_t``.  Local-global converges at ~0.98/iter, so the warm start
  dominates the output; measured 10x on the decoder floor.
- **Residual parameterisation** (``--residual``) — ``S* = S_t + delta`` instead
  of ``S* = I + delta``, so the zero-initialised network starts at "stretch
  unchanged" rather than "rest shape".
- **Position supervision** (``--loss pos``) — mass-weighted squared error
  against the reference trajectory.  The self-supervised incremental potential
  (``--loss phys``) is retained as an ablation; note it is one backward-Euler
  step at ``--dt`` while the reference data is 10 substeps at ``dt/10``, which
  caps its fidelity (review §2.3).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

from . import torch_solver as ts
from .model import StretchNet, build_face_adjacency, build_features
from .potentials import incremental_potential_batched
from .torch_solver import compute_S_from_x, inertial_predictor


def vert_to_tet_pin_flag(pinned: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pin_set = {int(v) for v in pinned}
    flag = np.zeros(tets.shape[0], dtype=np.float32)
    for t in range(tets.shape[0]):
        if any(int(tets[t, k]) in pin_set for k in range(4)):
            flag[t] = 1.0
    return flag


def build_windows(traj_start: np.ndarray, n_total: int, k_max: int) -> np.ndarray:
    """(i_prev, i_t, room) triples; room = GT frames available after i_t."""
    windows = []
    n_traj = traj_start.size
    for traj in range(n_traj):
        s = int(traj_start[traj])
        e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
        for t_off in range(1, (e - s) - 1):
            i_t = s + t_off
            room = (e - 1) - i_t
            if room >= 1:
                windows.append((i_t - 1, i_t, min(room, k_max)))
    return np.array(windows, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-rollout", type=int, default=4)
    parser.add_argument("--curriculum-frac", type=float, default=0.5)
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument("--loss", choices=("pos", "phys"), default="pos")
    parser.add_argument("--residual", action="store_true", help="predict S* = S_t + delta instead of S* = I + delta")
    parser.add_argument(
        "--warm", choices=("inertial", "prev"), default="inertial", help="decoder warm start: inertial predictor or x_t"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="MGN-style input noise: perturb (x_prev, x_t) at window start "
        "so the model learns to contract off-manifold states",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="alternating network<->decoder blocks per step; decoder iterations are split evenly across blocks",
    )
    parser.add_argument(
        "--phys-weight",
        type=float,
        default=0.0,
        help="with --loss pos: add this weight of the incremental potential as an off-manifold regulariser",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32  # network dtype; decoder runs fp64

    data = np.load(args.train)
    rest_q = data["rest_q"]
    tets_np = data["tet_indices"]
    x_all = data["x"].astype(np.float64)
    n_total = x_all.shape[0]

    solver = ts.build_solver(
        rest_q, tets_np, data["tet_poses"], data["pinned_indices"], device=device, dtype=torch.float64
    )
    face_adj = torch.as_tensor(build_face_adjacency(tets_np), dtype=torch.int64, device=device)
    pin_flag = torch.as_tensor(vert_to_tet_pin_flag(data["pinned_indices"], tets_np), dtype=dtype, device=device)
    mass = torch.as_tensor(data["particle_mass"], dtype=torch.float64, device=device)
    mu32 = torch.as_tensor(data["mu_per_tet"], dtype=dtype, device=device)
    lam32 = torch.as_tensor(data["lam_per_tet"], dtype=dtype, device=device)
    mu64, lam64 = mu32.double(), lam32.double()
    volume = solver.w.double()
    gravity64 = torch.as_tensor(data["gravity"], dtype=torch.float64, device=device)
    gravity32 = gravity64.to(dtype)
    pinned_targets = torch.as_tensor(rest_q[data["pinned_indices"]], dtype=torch.float64, device=device)

    windows = build_windows(data["traj_start"], n_total, args.max_rollout)
    print(f"{len(windows)} training windows, K_max={args.max_rollout}, batch={args.batch}")

    x_gpu = torch.as_tensor(x_all, dtype=torch.float64, device=device)
    f_ext_gpu = torch.as_tensor(data["f_ext"], dtype=torch.float64, device=device)
    S_gpu = torch.as_tensor(data["S"], dtype=torch.float64, device=device)

    net = StretchNet().to(device=device, dtype=dtype)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        print(f"loaded init weights from {args.init_ckpt}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)

    rng = np.random.default_rng(args.seed)
    pin_b = pinned_targets[None].expand(args.batch, -1, -1)
    curriculum_end = max(1, int(args.steps * args.curriculum_frac))

    log = []
    t0 = time.time()
    for step in range(args.steps):
        if step < curriculum_end:
            k_target = 1 + int((args.max_rollout - 1) * step / curriculum_end)
        else:
            k_target = args.max_rollout

        # Prefer windows with enough room for the full target rollout.
        ok = np.where(windows[:, 2] >= k_target)[0]
        if ok.size >= args.batch:
            idx = rng.choice(ok, size=args.batch, replace=False)
            k_roll = k_target
        else:
            idx = rng.choice(len(windows), size=args.batch, replace=False)
            k_roll = int(windows[idx, 2].min())
        b = windows[idx]
        i_t0 = torch.as_tensor(b[:, 1], dtype=torch.int64, device=device)

        x_prev = x_gpu[b[:, 0]]  # (B, V, 3)
        x_t = x_gpu[b[:, 1]]
        if args.noise_std > 0.0:
            # Perturb the input state (targets stay clean) so training visits the
            # off-manifold states rollout inevitably produces.
            x_prev = x_prev + args.noise_std * torch.randn_like(x_prev)
            x_t = x_t + args.noise_std * torch.randn_like(x_t)
            x_prev = x_prev.clone()
            x_t = x_t.clone()
            x_prev[:, solver.pinned] = pinned_targets
            x_t[:, solver.pinned] = pinned_targets
            S_prev = compute_S_from_x(solver, x_prev)
            S_now = compute_S_from_x(solver, x_t)
        else:
            S_prev = S_gpu[b[:, 0]]
            S_now = S_gpu[b[:, 1]]

        opt.zero_grad()
        loss_total = torch.zeros((), dtype=torch.float64, device=device)
        x_pred = None
        for k in range(k_roll):
            f_ext = f_ext_gpu[i_t0 + k]  # (B, V, 3)

            if args.warm == "inertial":
                x0 = inertial_predictor(solver, x_t, x_prev, pin_b)
            else:
                x0 = x_t
            # Alternating network <-> decoder blocks (PoissonNet-style).  Each
            # block's global solve propagates the previous block's local
            # prediction across the whole mesh, so B blocks give the *network*
            # B global hops of receptive field at matched total decoder cost.
            S_prev_f = S_prev.to(dtype)
            iters_per_block = max(1, args.solver_iters // args.blocks)
            x_next = x0
            S_cur = S_now
            for _b in range(args.blocks):
                S_cur_f = S_cur.to(dtype)
                feat = build_features(
                    S_cur_f, S_prev_f, gravity32, f_ext.to(dtype), mu32, lam32, pin_flag, solver.tets, face_adj
                )
                S_star = net(feat, S_base=S_cur_f if args.residual else None)
                x_next = ts.solve(solver, S_star.double(), pin_b, x_init=x_next, n_iters=iters_per_block)
                if _b + 1 < args.blocks:
                    S_cur = compute_S_from_x(solver, x_next)

            if args.loss == "pos":
                diff = x_next - x_gpu[i_t0 + k + 1]
                loss_total = loss_total + (mass[None, :, None] * diff * diff).sum()
                if args.phys_weight > 0.0:
                    loss_total = loss_total + args.phys_weight * incremental_potential_batched(
                        x_next=x_next,
                        x_t=x_t,
                        x_prev=x_prev,
                        mass=mass,
                        gravity=gravity64,
                        f_ext=f_ext,
                        tets=solver.tets,
                        J=solver.J,
                        mu=mu64,
                        lam=lam64,
                        volume=volume,
                        dt=args.dt,
                    )
            else:
                loss_total = loss_total + incremental_potential_batched(
                    x_next=x_next,
                    x_t=x_t,
                    x_prev=x_prev,
                    mass=mass,
                    gravity=gravity64,
                    f_ext=f_ext,
                    tets=solver.tets,
                    J=solver.J,
                    mu=mu64,
                    lam=lam64,
                    volume=volume,
                    dt=args.dt,
                )

            S_prev = S_now
            S_now = compute_S_from_x(solver, x_next)
            x_prev = x_t
            x_t = x_next
            x_pred = x_next

        (loss_total / (args.batch * k_roll)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                pos_err = (x_pred - x_gpu[i_t0 + k_roll]).norm(dim=-1).mean().item()
            mean_loss = loss_total.item() / (args.batch * k_roll)
            elapsed = time.time() - t0
            print(f"step {step:5d}  K={k_roll}  L={mean_loss:+.4e}  pos_err={pos_err:.4e}  {elapsed:.1f}s", flush=True)
            log.append({"step": step, "K": k_roll, "loss": mean_loss, "pos_err": pos_err})

    print(f"training done in {time.time() - t0:.1f}s")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "log": log, "args": vars(args)}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
