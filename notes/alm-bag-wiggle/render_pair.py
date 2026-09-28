# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render matched saved trajectories with one fixed camera and baked labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp
from run_case import ROOT, Bag

sys.path.insert(0, str(Path(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")) / "Newton/tools"))
from newton_capture import Capture
from newton_capture._video import VideoWriter


def render(args):
    from PIL import Image, ImageDraw, ImageFont

    wp.init()
    wp.config.quiet = True
    sim = Bag(args.stiffness, False)
    stem = f"ke{int(args.stiffness)}"
    results = [json.loads((args.output / f"{stem}_{mode}.json").read_text()) for mode in ("off", "on")]
    trajectories = [np.load(args.output / f"{stem}_{mode}.npz") for mode in ("off", "on")]
    assert results[0]["model_sha256"] == results[1]["model_sha256"]
    np.testing.assert_array_equal(trajectories[0]["particle_q"][0], trajectories[1]["particle_q"][0])
    frames = min(args.frames or results[0]["frames"], results[0]["frames"])
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 29)
    font = ImageFont.truetype(font_path, 21)
    small_font = ImageFont.truetype(font_path, 18)
    pins = wp.empty(len(sim.pins), dtype=wp.vec3, device=sim.model.device)
    pin_colors = wp.full(len(sim.pins), wp.vec3(1.0, 0.65, 0.08), dtype=wp.vec3, device=sim.model.device)
    with Capture(
        out_dir=str(args.output),
        width=960,
        height=720,
        camera_pos=(0.50, -0.84, 0.46),
        camera_target=(0.0, 0.0, 0.14),
        camera_fov=42.0,
    ) as cap:
        viewer = cap._get_viewer(sim.model)
        viewer.show_ui = False
        viewer.renderer.draw_wireframe = True
        cap._apply_camera(viewer)
        path = args.output / f"{stem}_comparison.mp4"
        with VideoWriter(str(path), fps=60) as writer:
            for frame in range(1, frames + 1):
                panels = []
                for index, (trajectory, result) in enumerate(zip(trajectories, results, strict=True)):
                    sim.state_0.particle_q.assign(trajectory["particle_q"][frame])
                    sim.state_0.body_q.assign(trajectory["body_q"][frame])
                    pins.assign(trajectory["particle_q"][frame, sim.info["top_global_indices"]])
                    viewer.begin_frame(frame / 60.0)
                    viewer.log_state(sim.state_0)
                    viewer.log_points("pinned_rim", pins, radii=0.0023, colors=pin_colors)
                    viewer.end_frame()
                    panel = Image.fromarray(viewer.get_frame().numpy())
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, 960, 103), fill=(18, 25, 36))
                    title = "ALM OFF" if index == 0 else "ALM ON  |  bending only"
                    color = (241, 169, 101) if index == 0 else (93, 213, 212)
                    draw.text((22, 12), title, font=title_font, fill=color)
                    draw.text(
                        (22, 54),
                        f"tri_ke = {args.stiffness:.0e}   tri_ka = {0.2 * args.stiffness:.0e}   edge_ke = 200",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (22, 81),
                        "10 substeps x 10 iterations | triangle membrane unchanged",
                        font=small_font,
                        fill=(187, 199, 215),
                    )
                    row = result["rows"][frame]
                    draw.rectangle((0, 626, 960, 720), fill=(18, 25, 36))
                    phase = "settling" if frame <= 60 else "wiggling"
                    draw.text(
                        (22, 635),
                        f"t = {frame / 60:.2f} s  |  {phase}  |  stretch RMS = {100 * row['stretch_score']:.3f}%",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (22, 665),
                        f"bend RMS = {row['bend_score']:.4f} rad  |  fixed camera, synchronized states",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (22, 694),
                        "Reconstructed May fixture; same current solver branch in both panels",
                        font=small_font,
                        fill=(187, 199, 215),
                    )
                    panels.append(np.asarray(panel))
                combined = np.concatenate(panels, axis=1)
                writer.write_frame(combined)
                if frame in {1, 60, 180, 360}:
                    Image.fromarray(combined).save(args.output / f"{stem}_frame{frame:03d}.png")
                if frame % 60 == 0:
                    print(f"render {stem}: {frame}/{frames}", flush=True)
        print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiffness", type=float, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--frames", type=int)
    render(parser.parse_args())
