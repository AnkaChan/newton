# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""kNN conditional-variance test: the information floor of the per-tet features.

Hypothesis under test (notes/01): the 28-dim per-tet feature vector does not
determine the next-step stretch — far-field state the tet cannot see changes
the correct answer.  If true, samples with near-identical features carry
different targets, and NO per-tet predictor (any capacity, any training) can
beat that conditional spread.

Estimator: kNN regression across trajectories.
  pool   = all (frame, tet) samples from the training set
  query  = all (frame, tet) samples from the validation set (disjoint trajs)
  predict S_gt(t+dt) as the mean target of the k nearest pool samples in
  z-scored feature space.

Readout, in S-space and decoded to positions (solver_iters=10, inertial warm,
matching eval_singlestep):
  persistence   S* = S_t                 zero-information baseline
  net           trained checkpoint       what the MLP actually achieves
  kNN global    k in {1, 5, 20}          ~ floor of any per-tet function
  kNN same-tet  k over same tet id only  adds positional identity (distance
                                         to the pinned end etc.); if clearly
                                         lower, position features would help
  oracle S_gt   decode ground truth S    decoder-only floor

If net ~= kNN floor  -> the features are the bottleneck (notes/01 confirmed).
If net >> kNN floor  -> capacity/training slack remains; hierarchy premature.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .model import StretchNet, build_face_adjacency, build_features, sym_to_vec, vec_to_sym
from .rollout import vert_to_tet_pin_flag
from .torch_solver import compute_S_from_x, inertial_predictor


def sample_indices(traj_start: np.ndarray, n_total: int) -> np.ndarray:
    """Frame indices t with a valid (t-1, t, t+1) triplet inside one trajectory."""
    n_traj = traj_start.size
    idx = []
    for traj in range(n_traj):
        s = int(traj_start[traj])
        e = int(traj_start[traj + 1]) if traj + 1 < n_traj else n_total
        idx.extend(range(s + 1, e - 1))
    return np.asarray(idx, dtype=np.int64)


def load_split(path: str, state, gravity32, mu_t, lam_t, pin_flag, face_adj, device, dtype, chunk=256):
    """Returns per-frame tensors: feat (N,T,28), S_t, S_target (N,T,3,3), x triplets.

    Frame-chunked so the 4k article (17k tets x 4k frames) fits in GPU memory.
    """
    d = np.load(path)
    x_all = torch.as_tensor(d["x"], dtype=torch.float64, device=device)
    f_ext_all = torch.as_tensor(d["f_ext"], dtype=torch.float64, device=device)
    t_idx = torch.as_tensor(sample_indices(d["traj_start"], d["x"].shape[0]), device=device)

    S_all = torch.cat([compute_S_from_x(state, x_all[i : i + chunk]) for i in range(0, x_all.shape[0], chunk)])
    S_t = S_all[t_idx]
    S_prev = S_all[t_idx - 1]
    S_target = S_all[t_idx + 1]
    feat = torch.cat(
        [
            build_features(
                S_t[i : i + chunk].to(dtype),
                S_prev[i : i + chunk].to(dtype),
                gravity32,
                f_ext_all[t_idx[i : i + chunk]].to(dtype),
                mu_t,
                lam_t,
                pin_flag,
                state.tets,
                face_adj,
            )
            for i in range(0, t_idx.shape[0], chunk)
        ]
    )
    return {
        "feat": feat,  # (N, T, 28)
        "S_t": S_t,
        "S_target": S_target,
        "x_prev": x_all[t_idx - 1],
        "x_t": x_all[t_idx],
        "x_target": x_all[t_idx + 1],
    }


def knn_predict(pool_feat, pool_tgt, query_feat, ks, chunk=512):
    """Chunked exact kNN regression. pool_feat (P,28) z-scored, pool_tgt (P,6).

    Returns {k: (Q,6)} predictions and the 1-NN distances (Q,).
    """
    kmax = max(ks)
    preds = {k: [] for k in ks}
    d1 = []
    for i in range(0, query_feat.shape[0], chunk):
        q = query_feat[i : i + chunk]
        dist = torch.cdist(q, pool_feat)  # (q, P)
        dd, jj = dist.topk(kmax, dim=1, largest=False)
        tgt = pool_tgt[jj]  # (q, kmax, 6)
        for k in ks:
            preds[k].append(tgt[:, :k].mean(dim=1))
        d1.append(dd[:, 0])
    return {k: torch.cat(v) for k, v in preds.items()}, torch.cat(d1)


def knn_predict_same_tet(pool_feat_t, pool_tgt_t, query_feat_t, ks, tet_chunk=32):
    """Per-tet kNN. pool_feat_t (T,P,28), pool_tgt_t (T,P,6), query (T,Q,28).

    Returns {k: (T,Q,6)}.
    """
    kmax = max(ks)
    T = pool_feat_t.shape[0]
    preds = {k: [] for k in ks}
    for i in range(0, T, tet_chunk):
        dist = torch.cdist(query_feat_t[i : i + tet_chunk], pool_feat_t[i : i + tet_chunk])  # (c,Q,P)
        _, jj = dist.topk(kmax, dim=2, largest=False)  # (c,Q,kmax)
        tgt = torch.gather(
            pool_tgt_t[i : i + tet_chunk, None].expand(-1, jj.shape[1], -1, -1),
            2,
            jj[..., None].expand(-1, -1, -1, 6),
        )  # (c,Q,kmax,6)
        for k in ks:
            preds[k].append(tgt[:, :, :k].mean(dim=2))
    return {k: torch.cat(v) for k, v in preds.items()}


def s_err(S_pred, S_gt):
    """Mean/median Frobenius error over all (frame, tet) samples."""
    e = (S_pred - S_gt).flatten(0, -3).norm(dim=(-2, -1))
    return e.mean().item(), e.median().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-train", required=True)
    p.add_argument("--data-val", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--solver-iters", type=int, default=10)
    p.add_argument("--pool-max", type=int, default=4_000_000, help="subsample the pool beyond this many samples")
    p.add_argument("--query-frames-max", type=int, default=0, help="0 = all val frames")
    p.add_argument("--no-same-tet", action="store_true", help="skip the per-tet-pool kNN (memory-heavy at 4k)")
    args = p.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.float32
    d = np.load(args.data_train)
    rest_q, tets_np = d["rest_q"], d["tet_indices"]
    state = ts.build_solver(rest_q, tets_np, d["tet_poses"], d["pinned_indices"], device=device, dtype=torch.float64)
    face_adj = torch.as_tensor(build_face_adjacency(tets_np), dtype=torch.int64, device=device)
    mu_t = torch.as_tensor(d["mu_per_tet"], dtype=dtype, device=device)
    lam_t = torch.as_tensor(d["lam_per_tet"], dtype=dtype, device=device)
    pin_flag = torch.as_tensor(vert_to_tet_pin_flag(d["pinned_indices"], tets_np), dtype=dtype, device=device)
    pinned_targets = torch.as_tensor(rest_q[d["pinned_indices"]], dtype=torch.float64, device=device)
    gravity32 = torch.as_tensor(d["gravity"], dtype=dtype, device=device)
    n_tets = tets_np.shape[0]

    print(f"building pool from {args.data_train} ...")
    pool = load_split(args.data_train, state, gravity32, mu_t, lam_t, pin_flag, face_adj, device, dtype)
    print(f"building queries from {args.data_val} ...")
    q = load_split(args.data_val, state, gravity32, mu_t, lam_t, pin_flag, face_adj, device, dtype)
    if args.query_frames_max and q["feat"].shape[0] > args.query_frames_max:
        q = {k: v[: args.query_frames_max] for k, v in q.items()}
    n_qf = q["feat"].shape[0]

    # ---- z-scored flat pool ------------------------------------------------
    pool_feat = pool["feat"].reshape(-1, pool["feat"].shape[-1])
    pool_tgt = sym_to_vec(pool["S_target"]).reshape(-1, 6).to(dtype)
    mean = pool_feat.mean(dim=0)
    std = pool_feat.std(dim=0).clamp(min=1e-8)
    pool_z = (pool_feat - mean) / std
    if pool_z.shape[0] > args.pool_max:
        keep = torch.randperm(pool_z.shape[0], device=device)[: args.pool_max]
        pool_z, pool_tgt = pool_z[keep], pool_tgt[keep]
    query_z = ((q["feat"] - mean) / std).reshape(-1, q["feat"].shape[-1])
    print(f"pool {pool_z.shape[0]} samples, query {query_z.shape[0]} samples ({n_qf} frames x {n_tets} tets)")

    S_gt = q["S_target"]

    # ---- kNN, global pool --------------------------------------------------
    preds, d1 = knn_predict(pool_z, pool_tgt, query_z, args.ks)
    knn_S = {k: vec_to_sym(v).reshape(n_qf, n_tets, 3, 3) for k, v in preds.items()}
    # Feature-match quality: 1-NN distance per z-scored dim (should be << 1).
    d1n = d1 / np.sqrt(pool_z.shape[1])
    print(f"1-NN z-dist/sqrt(dim): median {d1n.median():.3f}  p90 {d1n.quantile(0.9):.3f}")

    # ---- kNN, same-tet pool ------------------------------------------------
    knn_st_S = {}
    if not args.no_same_tet:
        pf_t = ((pool["feat"] - mean) / std).transpose(0, 1).contiguous()  # (T, Np, 28)
        pt_t = sym_to_vec(pool["S_target"]).to(dtype).transpose(0, 1).contiguous()  # (T, Np, 6)
        qf_t = query_z.reshape(n_qf, n_tets, -1).transpose(0, 1).contiguous()
        preds_st = knn_predict_same_tet(pf_t, pt_t, qf_t, args.ks)
        knn_st_S = {k: vec_to_sym(v.transpose(0, 1)).reshape(n_qf, n_tets, 3, 3) for k, v in preds_st.items()}

    # ---- trained net -------------------------------------------------------
    net_S = None
    if args.ckpt:
        net = StretchNet().to(device=device, dtype=dtype)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        residual = bool(ckpt.get("args", {}).get("residual", False))
        with torch.no_grad():
            net_S = net(q["feat"], S_base=q["S_t"].to(dtype) if residual else None)
        print(f"ckpt {args.ckpt} (residual={residual})")

    # ---- S-space table -----------------------------------------------------
    rows = [("persistence S*=S_t", q["S_t"].to(dtype))]
    if net_S is not None:
        rows.append(("net", net_S))
    for k in args.ks:
        rows.append((f"kNN global k={k}", knn_S[k]))
    for k in sorted(knn_st_S):
        rows.append((f"kNN same-tet k={k}", knn_st_S[k]))
    print("\n=== S-space error ||S_pred - S_gt(t+dt)||_F per tet ===")
    for name, S_pred in rows:
        m, med = s_err(S_pred.to(dtype), S_gt.to(dtype))
        print(f"  {name:22s} mean {m:.4e}  median {med:.4e}")

    # ---- decoded position error (matches eval_singlestep protocol) ---------
    print(f"\n=== decoded single-step position error (iters={args.solver_iters}, inertial warm) ===")
    x0 = inertial_predictor(state, q["x_t"], q["x_prev"], pinned_targets)
    dec_rows = [("oracle S_gt (decoder floor)", S_gt), *rows]
    with torch.no_grad():
        for name, S_pred in dec_rows:
            x_next = ts.solve(state, S_pred.double(), pinned_targets, x_init=x0, n_iters=args.solver_iters)
            e = (x_next - q["x_target"]).norm(dim=-1)
            print(
                f"  {name:28s} per-vertex mean {e.mean().item():.4e} m   95% {e.flatten().quantile(0.95).item():.4e} m"
            )


if __name__ == "__main__":
    sys.exit(main())
