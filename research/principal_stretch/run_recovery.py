# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run local-global stretch recovery against a recorded forward run."""

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
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--frames", type=str, default="all", help="'all' or comma-list of frame indices")
    parser.add_argument("--warm-start", action="store_true", help="warm-start each frame from previous recovery")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    wp.init()
    data = np.load(args.data)
    rest_q = data["rest_q"]
    tet_indices = data["tet_indices"]
    tet_poses = data["tet_poses"]
    pinned = data["pinned_indices"]
    x_gt = data["x"]  # (n_frames, V, 3)
    S_gt = data["S"]  # (n_frames, T, 3, 3)
    n_frames = x_gt.shape[0]

    if args.frames == "all":
        frame_idx = np.arange(n_frames)
    else:
        frame_idx = np.array([int(s) for s in args.frames.split(",")])

    rec = LocalGlobalRecover(
        rest_q=rest_q,
        tet_indices=tet_indices,
        tet_poses=tet_poses,
        pinned_indices=pinned,
        device=args.device,
    )

    print(f"loaded {args.data}: V={rest_q.shape[0]} T={tet_indices.shape[0]} P={pinned.size} frames={n_frames}")

    metrics = []
    x_prev = None
    for f in frame_idx:
        S_target = S_gt[f]
        pinned_targets = x_gt[f, pinned]
        x_init = x_prev if args.warm_start else None
        t0 = time.time()
        res = rec.solve(
            S_target=S_target,
            pinned_targets=pinned_targets,
            x_init=x_init,
            max_iters=args.max_iters,
            tol=args.tol,
        )
        dt = time.time() - t0

        # Diagnostics
        vert_err = float(np.linalg.norm(res.x - x_gt[f], axis=1).mean())
        # Procrustes-aligned error (rigid-best-fit of res.x to x_gt[f]).
        aligned_err = _procrustes_rmse(res.x, x_gt[f])
        m = {
            "frame": int(f),
            "iters": res.iters,
            "converged": res.converged,
            "stretch_err": res.stretch_err,
            "vert_err_mean": vert_err,
            "vert_err_aligned": aligned_err,
            "time_s": dt,
        }
        metrics.append(m)
        print(
            f"  frame {int(f):3d}  iters={res.iters:3d}  S_err={res.stretch_err:.3e}  "
            f"vert_err={vert_err:.3e}  aligned={aligned_err:.3e}  ({dt * 1e3:.0f} ms)"
        )
        x_prev = res.x

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            frame=np.array([m["frame"] for m in metrics], dtype=np.int32),
            iters=np.array([m["iters"] for m in metrics], dtype=np.int32),
            converged=np.array([m["converged"] for m in metrics], dtype=bool),
            stretch_err=np.array([m["stretch_err"] for m in metrics], dtype=np.float64),
            vert_err_mean=np.array([m["vert_err_mean"] for m in metrics], dtype=np.float64),
            vert_err_aligned=np.array([m["vert_err_aligned"] for m in metrics], dtype=np.float64),
        )
        print(f"wrote {args.out}")

    vmean = np.mean([m["vert_err_mean"] for m in metrics])
    smean = np.mean([m["stretch_err"] for m in metrics])
    print(f"\nSummary: mean vert_err={vmean:.3e}  mean stretch_err={smean:.3e}")


def _procrustes_rmse(A: np.ndarray, B: np.ndarray) -> float:
    """Rigid-Procrustes-aligned RMSE between A and B."""
    a = A - A.mean(0)
    b = B - B.mean(0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    aligned = a @ R.T + B.mean(0)
    return float(np.linalg.norm(aligned - B, axis=1).mean())


if __name__ == "__main__":
    sys.exit(main())
