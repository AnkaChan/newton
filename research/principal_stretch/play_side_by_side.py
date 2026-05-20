# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render original (GT) and recovered animations side-by-side to USD.

Builds a single Newton model with two identical soft grids, replays the
recorded ground-truth FEM positions in the left grid and the recovered
positions in the right grid (offset along +X), and dumps to a single .usda
that can be opened in Blender / usdview.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerUSD


def build_two_grids(dim_x, dim_y, dim_z, cell, x_offset):
    builder = newton.ModelBuilder()
    common = {
        "rot": wp.quat_identity(),
        "vel": wp.vec3(0.0, 0.0, 0.0),
        "dim_x": dim_x,
        "dim_y": dim_y,
        "dim_z": dim_z,
        "cell_x": cell,
        "cell_y": cell,
        "cell_z": cell,
        "density": 1.0e3,
        "k_mu": 1.0e5,
        "k_lambda": 1.0e5,
        "k_damp": 1e-3,
        "fix_left": True,
    }
    # GT (left)
    builder.add_soft_grid(pos=wp.vec3(0.0, 1.0, 0.0), **common)
    n_left = len(builder.particle_q)
    # Recovered (right) — same topology, just shifted in X
    builder.add_soft_grid(pos=wp.vec3(x_offset, 1.0, 0.0), **common)
    builder.color()
    model = builder.finalize()
    return model, n_left


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", type=str, required=True, help="path to recovery.npz from run_recovery_all")
    parser.add_argument("--out", type=str, required=True, help="output .usda path")
    parser.add_argument("--dim-x", type=int, default=8)
    parser.add_argument("--dim-y", type=int, default=4)
    parser.add_argument("--dim-z", type=int, default=4)
    parser.add_argument("--cell", type=float, default=0.1)
    parser.add_argument("--gap", type=float, default=0.4, help="x gap between the two grids (m)")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    wp.init()
    data = np.load(args.recovery)
    x_gt = data["x_gt"].astype(np.float32)  # (F, V, 3)
    x_rec = data["x_rec"].astype(np.float32)  # (F, V, 3)
    n_frames, n_verts, _ = x_gt.shape

    # X-offset for the recovered grid: grid width + gap.
    grid_width = args.dim_x * args.cell
    x_offset = grid_width + args.gap
    offset_vec = np.array([x_offset, 0.0, 0.0], dtype=np.float32)

    model, n_left = build_two_grids(args.dim_x, args.dim_y, args.dim_z, args.cell, x_offset)
    if n_left != n_verts:
        raise RuntimeError(f"grid vertex count mismatch: model={n_left}, data={n_verts}")
    n_total = n_left * 2
    print(f"model has {n_total} particles ({n_left} per grid x 2), {n_frames} frames")

    state = model.state()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ViewerUSD with up_axis="Y" since Newton's default scene is Y-up.
    viewer = ViewerUSD(output_path=str(out_path), fps=args.fps, up_axis="Z", num_frames=n_frames)
    viewer.set_model(model)

    sim_time = 0.0
    dt = 1.0 / args.fps
    for f in range(n_frames):
        # Concatenate: left = GT, right = recovered + offset
        q = np.concatenate([x_gt[f], x_rec[f] + offset_vec], axis=0).astype(np.float32)
        state.particle_q.assign(q)
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()
        sim_time += dt
        if f % 20 == 0:
            print(f"  frame {f:3d}/{n_frames}")
    viewer.close()
    print(f"wrote {out_path}")
    print(f"open in Blender/usdview. Left = ground truth, right = recovered (offset {x_offset:.2f} m).")


if __name__ == "__main__":
    sys.exit(main())
