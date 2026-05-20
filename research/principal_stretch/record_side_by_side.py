# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record an MP4 of GT vs recovered animations side-by-side.

Uses the AI-Logs/Newton/tools/newton_capture skill for headless GL rendering.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs"), "Newton/tools"))
from newton_capture import Capture
from newton_capture._video import VideoWriter

from .play_side_by_side import build_two_grids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dim-x", type=int, default=8)
    parser.add_argument("--dim-y", type=int, default=4)
    parser.add_argument("--dim-z", type=int, default=4)
    parser.add_argument("--cell", type=float, default=0.1)
    parser.add_argument("--gap", type=float, default=0.4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    wp.init()
    data = np.load(args.recovery)
    x_gt = data["x_gt"].astype(np.float32)
    x_rec = data["x_rec"].astype(np.float32)
    n_frames, n_verts, _ = x_gt.shape

    grid_width = args.dim_x * args.cell
    x_offset = grid_width + args.gap
    offset_vec = np.array([x_offset, 0.0, 0.0], dtype=np.float32)

    model, n_left = build_two_grids(args.dim_x, args.dim_y, args.dim_z, args.cell, x_offset)
    assert n_left == n_verts, f"{n_left} vs {n_verts}"
    state = model.state()

    # Newton is Z-up with gravity along -Z. Grids hang from +x=0 face and swing
    # in -Z. For a true side view we look along -Y at the XZ plane.
    grid_center = np.array(
        [x_offset / 2 + grid_width / 2, args.dim_y * args.cell / 2, 1.0 - args.dim_z * args.cell / 2]
    )
    cam_target = tuple(float(v) for v in grid_center)
    # Position camera a few metres along +Y (in front of the swing plane).
    cam_pos = (cam_target[0], cam_target[1] + 3.5, cam_target[2] + 0.2)

    out_dir = pathlib.Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with Capture(
        out_dir=str(out_dir),
        width=args.width,
        height=args.height,
        camera_pos=cam_pos,
        camera_target=cam_target,
    ) as cap:
        viewer = cap._get_viewer(model)
        cap._apply_camera(viewer)
        with VideoWriter(args.out, fps=args.fps) as writer:
            for f in range(n_frames):
                q = np.concatenate([x_gt[f], x_rec[f] + offset_vec], axis=0).astype(np.float32)
                state.particle_q.assign(q)
                viewer.begin_frame(0.0)
                viewer.log_state(state)
                viewer.end_frame()
                writer.write_frame(viewer.get_frame().numpy())
                if f % 20 == 0:
                    print(f"  frame {f:3d}/{n_frames}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
