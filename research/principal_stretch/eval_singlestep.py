# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced single-step eval: feed GT (x_{t-1}, x_t) and predict x_{t+1}."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .hier_model import HierStretchNet
from .hierarchy import build_hierarchy
from .model import StretchNet, build_face_adjacency, build_features
from .rollout import vert_to_tet_pin_flag
from .torch_solver import compute_S_from_x, inertial_predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--solver-iters", type=int, default=10)
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
    face_adj = torch.as_tensor(build_face_adjacency(tets_np), dtype=torch.int64, device=device)
    mu_t = torch.as_tensor(mu_np, dtype=dtype, device=device)
    lam_t = torch.as_tensor(lam_np, dtype=dtype, device=device)
    pin_flag = torch.as_tensor(vert_to_tet_pin_flag(pinned_np, tets_np), dtype=dtype, device=device)
    pinned_targets = torch.as_tensor(rest_q[pinned_np], dtype=torch.float64, device=device)
    gravity32 = torch.as_tensor(gravity_np, dtype=dtype, device=device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    residual = bool(ckpt_args.get("residual", False))
    warm = ckpt_args.get("warm", "prev")
    blocks = int(ckpt_args.get("blocks", 1))
    hier = bool(ckpt_args.get("hier", False))
    if hier:
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

    errs = []
    with torch.no_grad():
        for traj in range(n_traj):
            s = int(traj_start[traj])
            e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
            for t in range(s + 1, e - 1):
                x_prev = torch.as_tensor(x_all[t - 1], dtype=torch.float64, device=device)
                x_t = torch.as_tensor(x_all[t], dtype=torch.float64, device=device)
                x_target = torch.as_tensor(x_all[t + 1], dtype=torch.float64, device=device)
                f_ext = torch.as_tensor(f_ext_all[t], dtype=torch.float64, device=device)
                S_prev_f = compute_S_from_x(state, x_prev).to(dtype)
                x0 = inertial_predictor(state, x_t, x_prev, pinned_targets) if warm == "inertial" else x_t
                iters_per_block = max(1, args.solver_iters // blocks)
                x_next = x0
                S_cur = compute_S_from_x(state, x_t)
                for _b in range(blocks):
                    S_cur_f = S_cur.to(dtype)
                    feat = build_features(
                        S_cur_f, S_prev_f, gravity32, f_ext.to(dtype), mu_t, lam_t, pin_flag, state.tets, face_adj
                    )
                    if hier:
                        S_star = net(state, x_t, x_prev, f_ext, feat, S_cur).double()
                    else:
                        S_star = net(feat, S_base=S_cur_f if residual else None).double()
                    x_next = ts.solve(state, S_star, pinned_targets, x_init=x_next, n_iters=iters_per_block)
                    if _b + 1 < blocks:
                        S_cur = compute_S_from_x(state, x_next)
                e_per_v = (x_next - x_target).norm(dim=-1)
                errs.append(e_per_v.cpu().numpy())

    E = np.concatenate(errs)
    print(f"Teacher-forced single-step eval on {n_traj} trajs, {len(errs)} samples")
    print(f"  per-vertex mean = {E.mean():.4e} m")
    print(f"  per-vertex 95%  = {np.quantile(E, 0.95):.4e} m")
    print(f"  per-vertex max  = {E.max():.4e} m")


if __name__ == "__main__":
    sys.exit(main())
