# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Composition oracle audit (A6): linear vs log-space telescoping reconstruction.

Decides — before any training — whether log-space (Hencky) residual
composition through the tet hierarchy is sound (notes/00 D7.11).  For each
validation frame the ground-truth next-step stretch field is decomposed into
a telescope of per-level residuals and reconstructed at increasing truncation
depths under two composition rules:

  log rule     H_gt = log(S_gt) is telescoped; S_hat = exp(sum of components)
  linear rule  the same telescope on (S_gt - I);  S_hat = I + sum

Telescope (levels L..1, per-tet fields throughout):

  acc = 0
  for l = L, ..., 1:
      target_l = prolong_to_tets(pool_to_level(X, l))   # best level-l view
      r_l = target_l - acc                               # what level l adds
      acc += r_l                                         # acc == target_l
  r_0 = X - acc                                          # level-0 residual

Depth "coarsest-only" keeps r_L; "+level2" adds r_2; ...; "full" adds r_0 and
is exact by construction for BOTH rules — its role is a consistency check
(full-depth decode must match the oracle decode to < 1e-6 m, asserted).  The
informative rows are the truncated depths: they show how much each rule loses
at each scale.

Score (F14): per-vertex mean |decode(S_hat) - x_gt| with the standard decode
protocol (10 local-global iterations, inertial warm start), plus the oracle
row decode(S_gt).  Gate (task 3): log rule <= linear rule at every depth.

Non-SPD frames: the GT data contains a small tail of inverted tets (S_gt with
a negative eigenvalue — det F < 0, so the det-+1 polar flips one axis), where
log(S) is undefined.  Frames containing any such tet are excluded from the
audit (count reported); every remaining row is scored on the identical frame
set.  This is itself a finding: a log-space pipeline needs an SPD floor or
exclusion for these samples.

Run (GPU):
  uv run python -m research.principal_stretch.diag_composition \
      --data data/val.npz    --out artifacts_fix/composition_audit_toy.json
  uv run python -m research.principal_stretch.diag_composition \
      --data data/val_4k.npz --out artifacts_fix/composition_audit_4k.json \
      --max-frames 180
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from . import torch_solver as ts
from .diag_knn_floor import sample_indices
from .hierarchy import build_hierarchy, pool_mean, prolong
from .spd_log import sym_exp, sym_log
from .torch_solver import compute_S_from_x, inertial_predictor


def hierarchy_to_torch(hier, device) -> list[dict[str, torch.Tensor]]:
    """Per-level torch tensors: assign, CHILD-level volumes, PoU rows (fp64)."""
    levels = []
    child_vol = torch.as_tensor(hier.tet_vol, dtype=torch.float64, device=device)
    for lev in hier.levels:
        levels.append(
            {
                "assign": torch.as_tensor(lev.assign, dtype=torch.int64, device=device),
                "child_vol": child_vol,
                "pou_idx": torch.as_tensor(lev.pou_idx, dtype=torch.int64, device=device),
                "pou_w": torch.as_tensor(lev.pou_w, dtype=torch.float64, device=device),
            }
        )
        child_vol = torch.as_tensor(lev.vol, dtype=torch.float64, device=device)
    return levels


def telescope(x: torch.Tensor, levels: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
    """Coarse-to-fine residual components [r_L, ..., r_1, r_0], all per-tet fields.

    ``x`` is (T, ..., 3, 3) with tets on dim 0 (frames as a trailing batch —
    this keeps the pooling operators' node-dimension inference unambiguous).
    Partial sums of the returned components reconstruct ``x`` at increasing
    depth; the full sum equals ``x`` exactly.
    """
    acc = torch.zeros_like(x)
    components = []
    for level in range(len(levels), 0, -1):
        y = x
        for lev in levels[:level]:  # pool up through levels 1..level
            y = pool_mean(y, lev["assign"], lev["child_vol"])
        for lev in reversed(levels[:level]):  # chain PoU back down to tets
            y = prolong(y, lev["pou_idx"], lev["pou_w"])
        r = y - acc
        components.append(r)
        acc = acc + r  # acc == best level-`level` view of x
    components.append(x - acc)  # r_0: exact-reconstruction remainder
    return components


def depth_names(n_levels: int) -> list[str]:
    names = ["coarsest-only"]
    names += [f"+level{lev}" for lev in range(n_levels - 1, 0, -1)]
    names.append("full")
    return names


def decode_error(state, S_hat, pinned_targets, x0, x_gt, n_iters, chunk):
    """F14: decode S_hat with the standard protocol, score against recorded x_gt."""
    xs = []
    for i in range(0, S_hat.shape[0], chunk):
        xs.append(ts.solve(state, S_hat[i : i + chunk], pinned_targets, x_init=x0[i : i + chunk], n_iters=n_iters))
    x_next = torch.cat(xs)
    e = (x_next - x_gt).norm(dim=-1)
    return x_next, e.mean().item(), e.flatten().quantile(0.95).item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all valid val frames")
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--target", type=int, default=8, help="aggregation cluster size per level")
    p.add_argument("--solver-iters", type=int, default=10)
    p.add_argument("--decode-chunk", type=int, default=256, help="frames per batched decode")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data)
    rest_q, tets_np = d["rest_q"], d["tet_indices"]
    state = ts.build_solver(rest_q, tets_np, d["tet_poses"], d["pinned_indices"], device=device, dtype=torch.float64)
    pinned_targets = torch.as_tensor(rest_q[d["pinned_indices"]], dtype=torch.float64, device=device)

    x_all = torch.as_tensor(d["x"], dtype=torch.float64, device=device)
    t_idx = sample_indices(d["traj_start"], x_all.shape[0])
    if args.max_frames and t_idx.shape[0] > args.max_frames:
        t_idx = t_idx[: args.max_frames]
    t_idx = torch.as_tensor(t_idx, device=device)
    x_prev, x_t, x_gt = x_all[t_idx - 1], x_all[t_idx], x_all[t_idx + 1]
    n_frames, n_tets = t_idx.shape[0], tets_np.shape[0]

    hier = build_hierarchy(tets_np, rest_q, n_levels=args.levels, target=args.target)
    levels = hierarchy_to_torch(hier, device)
    cluster_counts = [lev.vol.shape[0] for lev in hier.levels]
    print(f"{args.data}: {n_frames} frames x {n_tets} tets, cluster counts {cluster_counts}")

    with torch.no_grad():
        # S_gt(t+1) per frame, chunked; then tets-first fp64 fields for the audit.
        S_gt = torch.cat([compute_S_from_x(state, x_gt[i : i + 256]) for i in range(0, n_frames, 256)])

        # log(S) is undefined on inverted tets (negative S eigenvalue); drop
        # frames containing any, so both rules and the oracle share one set.
        flat = S_gt.reshape(-1, 3, 3)
        lam_min = torch.cat(
            [torch.linalg.eigvalsh(flat[i : i + 16384]).min(dim=-1).values for i in range(0, flat.shape[0], 16384)]
        ).reshape(n_frames, n_tets)
        spd_frame = (lam_min > 0.0).all(dim=1)
        n_excluded = int((~spd_frame).sum().item())
        if n_excluded:
            print(
                f"excluding {n_excluded}/{n_frames} frames with non-SPD S_gt(t+1) "
                f"({int((lam_min <= 0.0).sum().item())} inverted (frame, tet) samples; "
                f"min eigenvalue {lam_min.min().item():.3f})"
            )
            S_gt, x_prev, x_t, x_gt = S_gt[spd_frame], x_prev[spd_frame], x_t[spd_frame], x_gt[spd_frame]
            n_frames = S_gt.shape[0]

        # sym_log / sym_exp chunk their eigh calls internally (cusolver's
        # batched-size limit) and fall back to CPU LAPACK on convergence
        # failures, so one flat call over all (frame, tet) samples is fine.
        eye = torch.eye(3, dtype=torch.float64, device=device)
        fields = {
            "log": sym_log(S_gt).transpose(0, 1),  # H_gt, (T, B, 3, 3)
            "linear": (S_gt - eye).transpose(0, 1),
        }
        names = depth_names(args.levels)
        recon = {}  # (rule, depth) -> (B, T, 3, 3)
        for rule, field in fields.items():
            acc = torch.zeros_like(field)
            for name, comp in zip(names, telescope(field, levels), strict=True):
                acc = acc + comp
                s_hat = sym_exp(acc) if rule == "log" else eye + acc
                recon[rule, name] = s_hat.transpose(0, 1).contiguous()

        # F14 decode for every row, plus the oracle.
        x0 = inertial_predictor(state, x_t, x_prev, pinned_targets)
        x_oracle, oracle_mean, oracle_p95 = decode_error(
            state, S_gt, pinned_targets, x0, x_gt, args.solver_iters, args.decode_chunk
        )
        results = {}
        consistency = {}
        for (rule, name), s_hat in recon.items():
            x_hat, e_mean, e_p95 = decode_error(
                state, s_hat, pinned_targets, x0, x_gt, args.solver_iters, args.decode_chunk
            )
            s_frob = (s_hat - S_gt).norm(dim=(-2, -1)).mean().item()
            results[rule, name] = {"mean_m": e_mean, "p95_m": e_p95, "s_frob_mean": s_frob}
            if name == "full":
                consistency[rule] = (x_hat - x_oracle).norm(dim=-1).max().item()

    # Consistency check: full depth is exact by construction for both rules,
    # so its decode must match the oracle decode to numerical tolerance.
    for rule, dev_m in consistency.items():
        print(f"consistency [{rule}]: max |decode(full) - decode(oracle)| = {dev_m:.3e} m")
        assert dev_m < 1e-6, f"{rule} full-depth decode deviates {dev_m:.3e} m from oracle (tol 1e-6)"

    print(f"\n=== F14 decoded per-vertex mean error [m] (iters={args.solver_iters}, inertial warm) ===")
    print(f"  {'depth':16s} {'linear mean':>12s} {'log mean':>12s} {'linear p95':>12s} {'log p95':>12s}")
    for name in names:
        lin, log = results["linear", name], results["log", name]
        print(f"  {name:16s} {lin['mean_m']:12.4e} {log['mean_m']:12.4e} {lin['p95_m']:12.4e} {log['p95_m']:12.4e}")
    print(f"  {'oracle S_gt':16s} {oracle_mean:12.4e} {'':12s} {oracle_p95:12.4e}   (decoder floor)")

    print("\n=== supplementary: mean ||S_hat - S_gt||_F per tet ===")
    for name in names:
        lin, log = results["linear", name], results["log", name]
        print(f"  {name:16s} linear {lin['s_frob_mean']:.4e}   log {log['s_frob_mean']:.4e}")

    tol = 1e-9  # absorb float noise where both rules are exact (full depth)
    gate_log_le_linear = all(
        results["log", name]["mean_m"] <= results["linear", name]["mean_m"] + tol for name in names
    )
    gate_full_2x = results["log", "full"]["mean_m"] <= 2.0 * oracle_mean
    print(f"\ngate: log <= linear at every depth: {'PASS' if gate_log_le_linear else 'FAIL'}")
    print(f"gate: full-depth log within 2x of oracle: {'PASS' if gate_full_2x else 'FAIL'}")

    out = {
        "data": args.data,
        "n_frames_scored": n_frames,
        "n_frames_excluded_non_spd": n_excluded,
        "n_tets": n_tets,
        "levels": args.levels,
        "target": args.target,
        "cluster_counts": cluster_counts,
        "solver_iters": args.solver_iters,
        "oracle": {"mean_m": oracle_mean, "p95_m": oracle_p95},
        "rows": [{"depth": name, "rule": rule, **results[rule, name]} for name in names for rule in ("linear", "log")],
        "consistency_max_dev_m": consistency,
        "gates": {"log_le_linear_all_depths": gate_log_le_linear, "full_log_within_2x_oracle": gate_full_2x},
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
