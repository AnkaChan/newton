# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: decompose the Phase-2 single-step error into its floors.

Three questions:

1. **Objective floor.** The training loss is *one* exact backward-Euler step
   at ``dt = 1/60``.  The ground-truth data is *ten* VBD substeps at
   ``dt = 1/600``.  How far apart are those two answers?  Anything below this
   distance is unreachable no matter how good the network is.
   -> minimise the incremental potential directly over ``x`` and compare to GT.

2. **Decoder floor.** Feed the decoder the *exact* stretches of the GT next
   frame and run it with the training iteration count (6) and the eval count
   (10), warm-started at ``x_t``.  How much error does decoder truncation add?

3. **Where the trained network sits.** Compare the potential attained by the
   network's decoded ``x`` against the potential at the exact minimiser and at
   the GT frame.  If ``L(x_net) ~ L(x_min) < L(x_gt)`` the network is doing
   what it was asked and the objective is the binding constraint.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .model import StretchNet, build_face_adjacency, build_features
from .potentials import incremental_potential
from .rollout import vert_to_tet_pin_flag
from .torch_solver import compute_S_from_x


def minimise_potential(state, x_t, x_prev, f_ext, mass, gravity, mu, lam, volume, pinned_targets, dt, iters=400):
    """Exact single backward-Euler step: argmin_x of the incremental potential."""
    free = state.free
    x = x_t.clone()
    x[state.pinned] = pinned_targets
    xf = x[free].clone().requires_grad_(True)

    def closure_x():
        full = x.clone()
        full[free] = xf
        return full

    opt = torch.optim.LBFGS(
        [xf],
        max_iter=iters,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        L = incremental_potential(
            x_next=closure_x(),
            x_t=x_t,
            x_prev=x_prev,
            mass=mass,
            gravity=gravity,
            f_ext=f_ext,
            tets=state.tets,
            J=state.J,
            mu=mu,
            lam=lam,
            volume=volume,
            dt=dt,
            pin_idx=None,
            pin_target=None,
        )["total"]
        L.backward()
        return L

    for _ in range(4):
        opt.step(closure)

    with torch.no_grad():
        out = closure_x().detach()
    # residual force norm at the solution
    xf2 = out[free].clone().requires_grad_(True)
    full = out.clone()
    full = full.index_copy(0, free, xf2)
    L = incremental_potential(
        x_next=full,
        x_t=x_t,
        x_prev=x_prev,
        mass=mass,
        gravity=gravity,
        f_ext=f_ext,
        tets=state.tets,
        J=state.J,
        mu=mu,
        lam=lam,
        volume=volume,
        dt=dt,
        pin_idx=None,
        pin_target=None,
    )["total"]
    g = torch.autograd.grad(L, xf2)[0]
    return out, g.norm().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--samples", type=int, default=40)
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

    rng = np.random.default_rng(0)
    cand = []
    for traj in range(n_traj):
        s = int(traj_start[traj])
        e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
        cand += list(range(s + 1, e - 1))
    picks = rng.choice(len(cand), size=min(args.samples, len(cand)), replace=False)

    rec = {k: [] for k in ("be", "dec6", "dec10", "net6", "net10", "L_gt", "L_min", "L_net", "gradnorm", "gt_step")}
    for pi in picks:
        t = cand[int(pi)]
        x_prev = torch.as_tensor(x_all[t - 1], dtype=torch.float64, device=device)
        x_t = torch.as_tensor(x_all[t], dtype=torch.float64, device=device)
        x_gt = torch.as_tensor(x_all[t + 1], dtype=torch.float64, device=device)
        f_ext = torch.as_tensor(f_ext_all[t], dtype=torch.float64, device=device)

        kw = {
            "x_t": x_t,
            "x_prev": x_prev,
            "mass": mass,
            "gravity": gravity,
            "f_ext": f_ext,
            "tets": state.tets,
            "J": state.J,
            "mu": mu,
            "lam": lam,
            "volume": volume,
            "dt": args.dt,
            "pin_idx": None,
            "pin_target": None,
        }

        # 1. objective floor
        x_be, gn = minimise_potential(
            state, x_t, x_prev, f_ext, mass, gravity, mu, lam, volume, pinned_targets, args.dt
        )
        rec["be"].append((x_be - x_gt).norm(dim=-1).mean().item())
        rec["gradnorm"].append(gn)
        rec["gt_step"].append((x_gt - x_t).norm(dim=-1).mean().item())

        # 2. decoder floor with exact GT stretches
        S_exact = compute_S_from_x(state, x_gt)
        for k, tag in ((6, "dec6"), (10, "dec10")):
            xd = ts.solve(state, S_exact, pinned_targets, x_init=x_t, n_iters=k)
            rec[tag].append((xd - x_gt).norm(dim=-1).mean().item())

        # 3. potentials
        with torch.no_grad():
            rec["L_gt"].append(incremental_potential(x_next=x_gt, **kw)["total"].item())
            rec["L_min"].append(incremental_potential(x_next=x_be, **kw)["total"].item())

        if net is not None:
            with torch.no_grad():
                S_t = compute_S_from_x(state, x_t).float()
                S_p = compute_S_from_x(state, x_prev).float()
                feat = build_features(S_t, S_p, g32, f_ext.float(), mu32, lam32, pin_flag, state.tets, face_adj)
                S_star = net(feat).double()
                for k, tag in ((6, "net6"), (10, "net10")):
                    xn = ts.solve(state, S_star, pinned_targets, x_init=x_t, n_iters=k)
                    rec[tag].append((xn - x_gt).norm(dim=-1).mean().item())
                    if k == 10:
                        rec["L_net"].append(incremental_potential(x_next=xn, **kw)["total"].item())

    def st(k):
        a = np.array(rec[k])
        return f"mean={a.mean():.4e}  median={np.median(a):.4e}  max={a.max():.4e}" if a.size else "n/a"

    print(f"\n=== {len(picks)} random single-step windows from {args.data} ===")
    print(f"dt used by the loss        : {args.dt:.6f} s")
    print(f"GT per-step motion         : {st('gt_step')}")
    print(f"LBFGS residual |dL/dx|     : {st('gradnorm')}  (N; total weight ~{(mass.sum() * 9.81).item():.0f} N)")
    print("\n-- position error vs GT x_{t+1} [m] --")
    print(f"  exact BE minimiser        : {st('be')}   <-- OBJECTIVE FLOOR")
    print(f"  decoder(GT stretch, 6 it) : {st('dec6')}   <-- decoder floor @train iters")
    print(f"  decoder(GT stretch, 10 it): {st('dec10')}  <-- decoder floor @eval iters")
    if net is not None:
        print(f"  network (6 iters)         : {st('net6')}")
        print(f"  network (10 iters)        : {st('net10')}")
    print("\n-- incremental potential [J] (lower = better per the training objective) --")
    print(f"  at GT x_{{t+1}}             : {st('L_gt')}")
    print(f"  at exact minimiser        : {st('L_min')}")
    if net is not None:
        print(f"  at network x              : {st('L_net')}")
        Lg = np.array(rec["L_gt"])
        Lm = np.array(rec["L_min"])
        Ln = np.array(rec["L_net"])
        print(f"\n  L(GT) - L(min)   = {np.mean(Lg - Lm):+.4e} J   (GT is NOT the objective's optimum by this much)")
        print(f"  L(net) - L(min)  = {np.mean(Ln - Lm):+.4e} J   (how far the network is from the objective's optimum)")
        print(f"  fraction of the gap the network closed: {1.0 - np.mean(Ln - Lm) / max(np.mean(Lg - Lm), 1e-30):.3f}")


if __name__ == "__main__":
    sys.exit(main())
