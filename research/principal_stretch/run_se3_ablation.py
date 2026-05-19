# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""SE(3) ambiguity ablation.

Run recovery with NO pinned vertices and only a tiny Tikhonov regularization
to make L invertible. The minimum-energy x is found in some arbitrary SE(3)
frame (centred at origin, rotation undetermined). Raw vertex error vs the
recorded GT will be large; Procrustes-aligned error should still be tiny —
demonstrating that the stretch field determines the deformed shape exactly
up to a global rigid transform.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

from .recover_local_global import LocalGlobalRecover
from .run_recovery import _procrustes_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--frames", type=str, default="0,30,60,90,119")
    parser.add_argument("--tikhonov", type=float, default=1e-6, help="regularisation to make unpinned L invertible")
    args = parser.parse_args()

    wp.init()
    data = np.load(args.data)
    rest_q = data["rest_q"]
    tet_indices = data["tet_indices"]
    tet_poses = data["tet_poses"]
    x_gt = data["x"]
    S_gt = data["S"]

    # No anchors — pure stretch-driven recovery.
    rec = LocalGlobalRecover(
        rest_q,
        tet_indices,
        tet_poses,
        pinned_indices=np.array([], dtype=np.int64),
        device="cuda:0",
        tikhonov=args.tikhonov,
    )
    frame_idx = [int(s) for s in args.frames.split(",")]

    print(f"V={rest_q.shape[0]} T={tet_indices.shape[0]}   pinned=NONE   tikhonov={args.tikhonov}")
    print(f"{'frame':>5}  {'iters':>5}  {'S_err':>10}  {'vert_raw':>10}  {'vert_aligned':>12}")
    empty = np.zeros((0, 3), dtype=np.float32)
    for f in frame_idx:
        res = rec.solve(
            S_target=S_gt[f],
            pinned_targets=empty,
            x_init=None,
            max_iters=args.max_iters,
            tol=args.tol,
        )
        vert_raw = float(np.linalg.norm(res.x - x_gt[f], axis=1).mean())
        vert_aligned = _procrustes_rmse(res.x, x_gt[f])
        print(f"{f:>5d}  {res.iters:>5d}  {res.stretch_err:>10.3e}  {vert_raw:>10.3e}  {vert_aligned:>12.3e}")


if __name__ == "__main__":
    sys.exit(main())
