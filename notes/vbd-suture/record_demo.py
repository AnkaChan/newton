"""Record the VBD suturing example with a fixed camera (headless, Xvfb) and write stills and metrics.

Usage (from the worktree root, GPU 0):
    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python notes/vbd-suture/record_demo.py --output <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, help="Output directory for the video, stills and metrics.")
parser.add_argument("--frames", type=int, default=600)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--camera-pos", type=float, nargs=3, default=(0.0, -0.16, 0.085))
parser.add_argument("--camera-target", type=float, nargs=3, default=(0.0, 0.0, 0.015))
parser.add_argument("--camera-fov", type=float, default=28.0)
parser.add_argument("--stills", type=int, nargs="*", default=(0, 60, 150, 300, 420, 599))
parser.add_argument("--tools", default="/home/horde/Code/AI-Docs/AI-Logs/Newton/tools")
args = parser.parse_args()
sys.path.insert(0, args.tools)

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.config.log_level = wp.LOG_WARNING

import newton  # noqa: E402
import newton.viewer  # noqa: E402
from newton.examples.vbd.example_vbd_suture import Example  # noqa: E402
from newton_capture import Capture  # noqa: E402
from newton_capture._video import VideoWriter  # noqa: E402

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
example_args = Example.create_parser().parse_args(["--viewer", "null", "--num-frames", str(args.frames)])
example = Example(newton.viewer.ViewerNull(), example_args)
example._init_faces()

metrics = []
t0 = time.time()
with Capture(str(out), width=args.width, height=args.height, camera_pos=tuple(args.camera_pos), camera_target=tuple(args.camera_target), camera_fov=args.camera_fov) as cap:
    viewer = cap._get_viewer(example.model)
    cap._apply_camera(viewer)
    with VideoWriter(str(out / "vbd_suture.mp4"), fps=example.params["fps"]) as writer:
        for frame in range(args.frames):
            viewer.begin_frame(example.sim_time)
            viewer.log_state(example.state_0)
            viewer.end_frame()
            image = viewer.get_frame().numpy()
            writer.write_frame(image)
            if frame in args.stills:
                from PIL import Image

                Image.fromarray(image).save(out / f"frame_{frame:04d}.png")
            example.step()
            example.test_post_step()
            if frame % 10 == 0:
                tunnels = int(example.solver._suture_tunnel_count.numpy()[0])
                metrics.append(
                    {
                        "frame": frame,
                        "time": example.sim_time,
                        "gap_mm": example.gap_width() * 1e3,
                        "needle_state": int(example.solver.suture_needle_state()[0]),
                        "tunnels": tunnels,
                        "tunnel_states": example.solver._suture_tunnel_state.numpy()[:tunnels].tolist(),
                        "points": [len(example.solver.suture_tunnel_points(0, t)[0]) for t in range(tunnels)],
                        "max_residual_mm": example.max_residual * 1e3,
                        "max_penetration_mm": example.max_penetration * 1e3,
                    }
                )
                print(json.dumps(metrics[-1]))
        video = writer.close()
elapsed = time.time() - t0
summary = {"frames": args.frames, "seconds": elapsed, "video": str(video), "final": metrics[-1] if metrics else None}
try:
    example.test_final()
    summary["test_final"] = "pass"
except AssertionError as exc:
    summary["test_final"] = f"fail: {exc}"
(out / "metrics.json").write_text(json.dumps({"summary": summary, "metrics": metrics}, indent=1))
print(json.dumps(summary, indent=1))
