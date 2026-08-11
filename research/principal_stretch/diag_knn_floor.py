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

Feature arms (``--feature-arm``, audit 0a per A7): the kNN machinery is
identical across arms — only the per-tet feature vector changes.
  base      the 28-dim ``build_features`` vector (default; today's behavior)
  edge      + the tet-level F6 edge features averaged over the tet's valid
            face-adjacent edges (7 extra dims -> 35)
  ancestor  + the 31-dim ``HierStretchNet`` coarse feature vectors of the
            tet's level-1..3 ancestor-chain clusters, oracle-computed from
            the ground-truth state (93 extra dims -> 128)
Pool and query features are built identically (both from GT states): this is
an oracle audit of feature *information*, not of any trained model.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .hier_model import HierStretchNet
from .hierarchy import build_hierarchy
from .model import StretchNet, build_face_adjacency, build_features, sym_to_vec, vec_to_sym
from .polar import polar_rotation
from .rollout import vert_to_tet_pin_flag
from .spd_log import so3_log_axial
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


def tet_edge_features(state, face_adj: torch.Tensor, edge_l0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean F6 edge features over each tet's valid face-adjacent edges.

    Per directed edge (receiver ``a`` = the tet, sender ``b`` = a face
    neighbor), with ``c`` the current tet centroids (mean of the 4 vertices),
    ``l0`` the rest centroid distance and ``R`` the polar rotation of the
    per-tet deformation gradient:

        e_ab = [ R_a^T (c_b - c_a) / l0,  |c_b - c_a| / l0 - 1,
                 axial(LogSO3(R_a^T R_b)) ]                          (7 dims)

    — the tet-level analogue of ``HierStretchNet._edge_features``, averaged
    over the (up to 4) valid edges.  Padded slots contribute zero.

    GT data contains transiently inverted tets (the task-3 finding that
    motivated ``spd_floor``); their polar rotation flips, so the relative
    rotation to a neighbor approaches pi, where the SO(3) log degenerates
    (``so3_log_axial`` raises beyond 3 rad).  Such edges — measured 2 in
    1.3M toy edge-frames, all incident to an inverted tet, healthy edges
    peak at 1.24 rad — are masked out of the average like padding.

    Args:
        state: solver state (``tets`` and ``J`` are used).
        face_adj: (T, 4) int64 face adjacency, -1 padded.
        edge_l0: (T, 4) rest centroid distances, 1.0 on padded slots.
        x: (B, V, 3) current positions.

    Returns:
        (B, T, 7) in the dtype of ``x``.
    """
    valid = face_adj >= 0  # (T, 4)
    idx = face_adj.clamp(min=0)
    x_tet = x[:, state.tets]  # (B, T, 4, 3)
    F = torch.einsum("tac,btad->btdc", state.J, x_tet)
    R = polar_rotation(F)  # (B, T, 3, 3)
    c = x_tet.mean(dim=2)  # (B, T, 3)
    d = c[:, idx] - c[:, :, None]  # (B, T, 4, 3)
    l0 = edge_l0[None].to(x.dtype)  # (1, T, 4)
    rel = torch.einsum("btji,btkj->btki", R, d) / l0[..., None]  # R_a^T (c_b - c_a) / l0
    stretch = d.norm(dim=-1) / l0 - 1.0  # (B, T, 4)
    R_rel = torch.einsum("btji,btkjl->btkil", R, R[:, idx])  # R_a^T R_b
    tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, T, 4)
    ok = valid[None] & (torch.clamp(0.5 * (tr - 1.0), -1.0, 1.0) > math.cos(2.9))  # theta < 2.9 rad
    eye = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R_rel)
    R_rel = torch.where(ok[..., None, None], R_rel, eye)
    e = torch.cat([rel, stretch[..., None], so3_log_axial(R_rel)], dim=-1)  # (B, T, 4, 7)
    e = e * ok[..., None].to(e.dtype)
    n_ok = ok.sum(dim=2).to(e.dtype).clamp(min=1.0)  # (B, T)
    return e.sum(dim=2) / n_ok[..., None]


def load_split(
    path: str,
    state,
    gravity32,
    mu_t,
    lam_t,
    pin_flag,
    face_adj,
    device,
    dtype,
    chunk=256,
    arm="base",
    edge_l0=None,
    hnet=None,
    chains=None,
):
    """Returns per-frame tensors: feat (N,T,D), S_t, S_target (N,T,3,3), x triplets.

    D is 28 (arm ``base``), 35 (arm ``edge``: + mean F6 tet-edge features) or
    128 (arm ``ancestor``: + the 31-dim coarse features of the tet's
    level-1..3 ancestor clusters, oracle-computed from the GT state).  The
    ancestor arm stages ``feat`` on the CPU: the full 4k pool at 128 dims is
    ~32 GB fp32, which fits in host RAM but not on the GPU next to the S
    tensors.

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

    feat_device = torch.device("cpu") if arm == "ancestor" else device
    # cusolver's batched eigh (inside level_features' sym_log/spd_floor)
    # rejects more than ~31.6k 3x3 fp64 matrices per call (task-3 finding),
    # so its frame batches are sub-chunked; one full frame is always legal.
    fchunk = max(1, 16384 // state.tets.shape[0])
    feat_parts = []
    for i in range(0, t_idx.shape[0], chunk):
        f = build_features(
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
        if arm != "base":
            ti = t_idx[i : i + chunk]
            x_c = x_all[ti]
            parts = [f, tet_edge_features(state, face_adj, edge_l0, x_c).to(dtype)]
            if arm == "ancestor":
                x_p, f_c, S_c = x_all[ti - 1], f_ext_all[ti], S_t[i : i + chunk]
                ancestor_parts = []
                for j in range(0, x_c.shape[0], fchunk):
                    level_feats = hnet.level_features(
                        state,
                        x_c[j : j + fchunk],
                        x_p[j : j + fchunk],
                        f_c[j : j + fchunk],
                        f[j : j + fchunk],
                        S_c[j : j + fchunk],
                    )  # list over levels of (b, N_l, 31) fp32
                    ancestor_parts.append(
                        torch.cat([lf[:, ch] for lf, ch in zip(level_feats, chains, strict=True)], dim=-1)
                    )
                parts.append(torch.cat(ancestor_parts).to(dtype))
            f = torch.cat(parts, dim=-1)
        feat_parts.append(f.to(feat_device))
    return {
        "feat": torch.cat(feat_parts),  # (N, T, D); CPU for the ancestor arm
        "S_t": S_t,
        "S_target": S_target,
        "x_prev": x_all[t_idx - 1],
        "x_t": x_all[t_idx],
        "x_target": x_all[t_idx + 1],
    }


def knn_predict(pool_feat, pool_tgt, query_feat, ks, chunk=512):
    """Chunked exact kNN regression. pool_feat (P,D) z-scored, pool_tgt (P,6).

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
    """Per-tet kNN. pool_feat_t (T,P,D), pool_tgt_t (T,P,6), query (T,Q,D).

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
    p.add_argument(
        "--feature-arm",
        choices=("base", "edge", "ancestor"),
        default="base",
        help="per-tet feature set: base = the 28-dim build_features vector; "
        "edge = + mean F6 tet-edge features (35 dims); ancestor = + the 31-dim "
        "coarse features of the level-1..3 ancestor-chain clusters, "
        "oracle-computed from the GT state (128 dims)",
    )
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

    # ---- feature-arm setup (topology-only, shared by pool and queries) -----
    edge_l0 = hnet = chains = None
    if args.feature_arm != "base":
        c0 = state.rest_q[state.tets].mean(dim=1)  # (T, 3) rest tet centroids
        l0 = (c0[face_adj.clamp(min=0)] - c0[:, None]).norm(dim=-1)
        edge_l0 = torch.where(face_adj >= 0, l0.clamp(min=1e-12), torch.ones_like(l0))
    if args.feature_arm == "ancestor":
        print("building 3-level hierarchy ...")
        hierarchy = build_hierarchy(tets_np, rest_q)
        # Only the (parameter-free) feature builder and static buffers of the
        # model are used — level_features touches no learned weights.
        hnet = HierStretchNet(hierarchy).to(device)
        chains, chain = [], None
        for lev in hierarchy.levels:
            assign = torch.as_tensor(lev.assign, dtype=torch.int64, device=device)
            chain = assign if chain is None else assign[chain]
            chains.append(chain)  # (T,) ancestor id of each tet at this level
        print(f"cluster counts per level: {[len(lev.vol) for lev in hierarchy.levels]}")

    arm_kwargs = {"arm": args.feature_arm, "edge_l0": edge_l0, "hnet": hnet, "chains": chains}
    print(f"building pool from {args.data_train} ...")
    pool = load_split(args.data_train, state, gravity32, mu_t, lam_t, pin_flag, face_adj, device, dtype, **arm_kwargs)
    print(f"building queries from {args.data_val} ...")
    q = load_split(args.data_val, state, gravity32, mu_t, lam_t, pin_flag, face_adj, device, dtype, **arm_kwargs)
    if args.query_frames_max and q["feat"].shape[0] > args.query_frames_max:
        q = {k: v[: args.query_frames_max] for k, v in q.items()}
    n_qf = q["feat"].shape[0]
    print(f"feature arm: {args.feature_arm} ({pool['feat'].shape[-1]} dims)")

    # ---- z-scored flat pool ------------------------------------------------
    # The ancestor arm stages `feat` on the CPU: full-pool z-stats and the
    # z-copy happen there, and only the subsampled pool moves to the GPU.
    # For the other arms every .to() below is a no-op.
    pool_feat = pool["feat"].reshape(-1, pool["feat"].shape[-1])
    pool_tgt = sym_to_vec(pool["S_target"]).reshape(-1, 6).to(dtype)
    mean = pool_feat.mean(dim=0)
    std = pool_feat.std(dim=0).clamp(min=1e-8)
    pool_z = (pool_feat - mean) / std
    if pool_z.shape[0] > args.pool_max:
        keep = torch.randperm(pool_z.shape[0], device=device)[: args.pool_max]
        pool_z, pool_tgt = pool_z[keep.to(pool_z.device)], pool_tgt[keep]
    pool_z, mean, std = pool_z.to(device), mean.to(device), std.to(device)
    query_z = ((q["feat"].to(device) - mean) / std).reshape(-1, q["feat"].shape[-1])
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
        pf_t = ((pool["feat"].to(device) - mean) / std).transpose(0, 1).contiguous()  # (T, Np, D)
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
            # Checkpoints are 28-dim StretchNets; the base features are always
            # the first 28 dims of any arm.
            net_S = net(q["feat"][..., :28].to(device), S_base=q["S_t"].to(dtype) if residual else None)
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
