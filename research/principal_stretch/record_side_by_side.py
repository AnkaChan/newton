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


def _load_font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _label_frame(frame: np.ndarray, width: int, gt_x_norm: float, rec_x_norm: float) -> np.ndarray:
    """Draw 'Ground truth' over the left grid and 'Recovered' over the right grid.

    gt_x_norm / rec_x_norm are the projected X centres of each grid in [0, 1].
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _load_font(36)
    h = frame.shape[0]
    y = int(h * 0.08)
    for text, x_norm, color in (
        ("Ground truth", gt_x_norm, (255, 220, 80)),
        ("Recovered", rec_x_norm, (120, 220, 255)),
    ):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = int(x_norm * width) - tw // 2
        # Shadow for legibility against any background.
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=color)
    return np.asarray(img)


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
    # Aim at a point ~30 cm below the rest pose so the swing arc is centred
    # in the frame, and place the camera at the same low Z (level / slight
    # upward tilt) along +Y.
    swing_z = 1.0 - 0.4
    cam_target = (
        x_offset / 2 + grid_width / 2,
        args.dim_y * args.cell / 2,
        swing_z,
    )
    cam_pos = (cam_target[0], cam_target[1] + 3.5, swing_z - 0.1)

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
        # Approximate horizontal screen positions of each grid's centre, used
        # only for label placement. The camera is centred on the gap; the GT
        # grid is at smaller X and the recovered at larger X.
        gt_x_norm = 0.27
        rec_x_norm = 0.73
        with VideoWriter(args.out, fps=args.fps) as writer:
            for f in range(n_frames):
                q = np.concatenate([x_gt[f], x_rec[f] + offset_vec], axis=0).astype(np.float32)
                state.particle_q.assign(q)
                viewer.begin_frame(0.0)
                viewer.log_state(state)
                viewer.end_frame()
                frame = viewer.get_frame().numpy()
                frame = _label_frame(frame, args.width, gt_x_norm, rec_x_norm)
                writer.write_frame(frame)
                if f % 20 == 0:
                    print(f"  frame {f:3d}/{n_frames}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
