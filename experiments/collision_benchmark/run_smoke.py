# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record one VBD smoke scene per process, using its current state buffer."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.collision_benchmark._capture import Capture  # noqa: E402


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--family", choices=["soft", "cable", "manipulation"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--substeps", type=int, help="Explicit scene substep override")
    parser.add_argument("--iterations", type=int, help="Explicit scene iteration override")
    parser.add_argument("--slow", type=float, default=1.0, help="Playback slowdown, simulation unchanged")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", f"/tmp/collision-benchmark-warp-{args.family}")
    wp.init()
    module = importlib.import_module(
        f"experiments.collision_benchmark.{args.family}_scenes"
        if args.family != "manipulation"
        else "experiments.collision_benchmark.manipulation"
    )
    start = time.monotonic()
    overrides = {}
    if args.substeps is not None:
        if not 1 <= args.substeps <= 10:
            parser.error("--substeps must be between 1 and 10")
        overrides["substeps"] = args.substeps
    if args.iterations is not None:
        if args.iterations < 1:
            parser.error("--iterations must be positive")
        overrides["iterations"] = args.iterations
    scene = module.build_scene(args.scene, device=args.device, **overrides)
    for key, value in overrides.items():
        if getattr(scene, key, None) != value:
            raise ValueError(f"Scene did not apply requested {key}={value}")
    duration = args.duration or scene.duration
    steps = round(duration / scene.frame_dt)
    fps = 1.0 / scene.frame_dt / args.slow
    stats = {
        "scene": args.scene,
        "title": scene.title,
        "description": scene.description,
        "asset_ids": scene.asset_ids,
        "solver": "VBD",
        "device": args.device,
        "frame_dt_s": scene.frame_dt,
        "duration_s": duration,
        "playback_slowdown": args.slow,
        "smoke_only": True,
        "requested_overrides": overrides,
        "finite_all_frames": True,
        "samples": [],
    }
    frame_samples = set(np.linspace(0, steps, 12).round().astype(int).tolist())
    pictures = []
    writer = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default(size=19)
        small_font = ImageFont.load_default(size=15)

    def observe(frame):
        state = scene.state
        entry = {"time": float(scene.time), "phase": scene.phase}
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
            arr = getattr(state, name, None)
            if arr is None or arr.size == 0:
                continue
            values = arr.numpy()
            finite = bool(np.isfinite(values).all())
            stats["finite_all_frames"] &= finite
            entry[name + "_finite"] = finite
            if name == "particle_q":
                entry["particle_min"] = values.min(axis=0).tolist()
                entry["particle_max"] = values.max(axis=0).tolist()
                entry["particle_mean"] = values.mean(axis=0).tolist()
            elif name == "particle_qd":
                entry["particle_speed_max"] = float(np.linalg.norm(values, axis=1).max())
        stats["samples"].append(entry)
        if not stats["finite_all_frames"]:
            raise RuntimeError(f"Non-finite state at {scene.time:.4f} s")

    try:
        with Capture(
            str(out),
            width=args.width,
            height=args.height,
            camera_pos=scene.camera_pos,
            camera_target=scene.camera_target,
            camera_fov=getattr(scene, "camera_fov", 45.0),
        ) as cap:
            viewer = None if args.no_video else cap._get_viewer(getattr(scene, "visual_model", scene.model))
            if viewer is not None:
                cap._apply_camera(viewer)
                writer = imageio_ffmpeg.write_frames(
                    str(out / "video.mp4"),
                    (args.width, args.height),
                    fps=fps,
                    codec="libx264",
                    pix_fmt_in="rgb24",
                    pix_fmt_out="yuv420p",
                    quality=7,
                    macro_block_size=1,
                    output_params=["-movflags", "+faststart"],
                )
                writer.send(None)
            for frame in range(steps + 1):
                observe(frame)
                if viewer is not None:
                    if hasattr(scene, "render"):
                        scene.render(viewer)
                        pixels = viewer.get_frame().numpy()
                    else:
                        pixels = cap._render_frame(viewer, scene.model, scene.state)
                    img = Image.fromarray(pixels).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((0, 0, args.width, 66), fill=(15, 23, 35))
                    draw.text((16, 8), scene.title.replace("·", "|"), font=font, fill="white")
                    draw.text(
                        (16, 37),
                        f"VBD smoke  |  {scene.time:.2f} s  |  {scene.phase}  |  {args.slow:g}x slower playback",
                        font=small_font,
                        fill=(191, 216, 238),
                    )
                    writer.send(np.ascontiguousarray(img))
                    if frame in frame_samples:
                        img.save(out / f"frame-{frame:04d}.jpg", quality=91)
                        thumb = img.copy()
                        thumb.thumbnail((480, 360))
                        pictures.append(thumb)
                    if frame == 0:
                        img.save(out / "poster.jpg", quality=93)
                if frame < steps:
                    scene.step()
                if frame % max(1, round(1.0 / scene.frame_dt)) == 0:
                    print(f"{args.scene}: t={scene.time:.2f}s phase={scene.phase}", flush=True)
        stats["completed"] = True
    except Exception as exc:
        stats["completed"] = False
        stats["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if writer is not None:
            writer.close()
        try:
            stats["scene_metrics"] = scene.metrics()
        except Exception as exc:
            stats["metrics_error"] = str(exc)
        stats["wall_time_s"] = time.monotonic() - start
        (out / "metrics.json").write_text(json.dumps(json_safe(stats), indent=2) + "\n")
        if pictures:
            sheet = Image.new("RGB", (480 * 3, 360 * math.ceil(len(pictures) / 3)), "#101923")
            for i, img in enumerate(pictures):
                sheet.paste(img, ((i % 3) * 480, (i // 3) * 360))
            sheet.save(out / "contact-sheet.jpg", quality=91)
    print(json.dumps(json_safe({k: v for k, v in stats.items() if k != "samples"}), indent=2), flush=True)


if __name__ == "__main__":
    main()
