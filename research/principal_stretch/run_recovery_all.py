# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run recovery on every recorded frame, warm-started, save x_recovered."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import warp as wp

from .recover_local_global import LocalGlobalRecover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    wp.init()
    data = np.load(args.data)
    rest_q = data["rest_q"]
    tet_indices = data["tet_indices"]
    tet_poses = data["tet_poses"]
    pinned = data["pinned_indices"]
    x_gt = data["x"]
    S_gt = data["S"]
    n_frames = x_gt.shape[0]

    rec = LocalGlobalRecover(rest_q, tet_indices, tet_poses, pinned, device=args.device)

    x_rec = np.zeros_like(x_gt)
    iters_log = np.zeros(n_frames, dtype=np.int32)
    serr_log = np.zeros(n_frames, dtype=np.float64)
    verr_log = np.zeros(n_frames, dtype=np.float64)

    x_prev = None
    t0 = time.time()
    for f in range(n_frames):
        res = rec.solve(
            S_target=S_gt[f],
            pinned_targets=x_gt[f, pinned],
            x_init=x_prev,
            max_iters=args.max_iters,
            tol=args.tol,
        )
        x_rec[f] = res.x
        iters_log[f] = res.iters
        serr_log[f] = res.stretch_err
        verr_log[f] = float(np.linalg.norm(res.x - x_gt[f], axis=1).mean())
        x_prev = res.x
        if f % 10 == 0:
            print(f"  frame {f:3d}/{n_frames}  S_err={res.stretch_err:.3e}  vert_err={verr_log[f]:.3e}")

    dt = time.time() - t0
    print(f"recovery done in {dt:.1f}s for {n_frames} frames")
    print(f"mean vert_err={verr_log.mean():.3e}  max vert_err={verr_log.max():.3e}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rest_q=rest_q,
        tet_indices=tet_indices,
        tet_poses=tet_poses,
        pinned_indices=pinned,
        x_gt=x_gt,
        x_rec=x_rec,
        iters=iters_log,
        stretch_err=serr_log,
        vert_err=verr_log,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
