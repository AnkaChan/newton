# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

import newton.viewer
from newton.examples.vbd import example_vbd_bag_franka_pickup as bag_example


def _sci(value: float) -> str:
    text = f"{value:.2e}"
    mantissa, exp = text.split("e")
    return f"{float(mantissa):g}e{int(exp)}"


def _case_slug(ka: float, ke: float) -> str:
    return f"ka_{_sci(ka).replace('+', '')}_ke_{_sci(ke).replace('+', '')}".replace("-", "m")


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    pad = 10
    bbox = draw.textbbox((0, 0), label)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.rectangle((pad - 4, pad - 4, pad + w + 8, pad + h + 8), fill=(0, 0, 0, 150))
    draw.text((pad, pad), label, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _run_case(
    *,
    output_dir: Path,
    ka: float,
    ke: float,
    frames: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
) -> dict[str, object]:
    params = dict(bag_example.PARAMS)
    params["cloth_tri_ka"] = float(ka)
    params["cloth_tri_ke"] = float(ke)
    params["settle_frames"] = int(frames)
    bag_example.PARAMS = params

    slug = _case_slug(ka, ke)
    video_path = output_dir / f"{slug}.mp4"
    label = f"ka={_sci(ka)}  ke={_sci(ke)}"
    print(f"[sweep] Running {label} -> {video_path}", flush=True)

    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True, vsync=False)
    if hasattr(viewer, "show_ui"):
        viewer.show_ui = False

    args = SimpleNamespace(seed=seed)
    example = bag_example.Example(viewer, args)

    try:
        with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index in range(frames):
                example.step()
                example.render()
                frame = viewer.get_frame().numpy()
                writer.append_data(_annotate(frame, label))
                if (frame_index + 1) % max(1, frames // 10) == 0:
                    print(f"[sweep]   {slug}: {frame_index + 1}/{frames} frames", flush=True)
    finally:
        viewer.close()

    return {
        "cloth_tri_ka": float(ka),
        "cloth_tri_ke": float(ke),
        "video": str(video_path),
        "label": label,
    }


def _merge_grid(cases: list[dict[str, object]], output_path: Path, *, fps: int, frames: int) -> None:
    if len(cases) != 6:
        raise ValueError(f"Expected exactly 6 cases for a 2x3 grid, got {len(cases)}")

    readers = [imageio.get_reader(case["video"]) for case in cases]
    try:
        first = readers[0].get_data(0)
        h, w = first.shape[:2]
        with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index in range(frames):
                grid = np.zeros((2 * h, 3 * w, 3), dtype=np.uint8)
                for case_index, reader in enumerate(readers):
                    row = case_index // 3
                    col = case_index % 3
                    frame = reader.get_data(frame_index)
                    grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame[:, :, :3]
                writer.append_data(grid)
                if (frame_index + 1) % max(1, frames // 10) == 0:
                    print(f"[sweep]   grid: {frame_index + 1}/{frames} frames", flush=True)
    finally:
        for reader in readers:
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a VBD bag stiffness sweep and merge it into a 2x3 video.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vbd_bag_franka_stiffness_sweep"))
    parser.add_argument("--frames", type=int, default=bag_example.PARAMS["settle_frames"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=bag_example.PARAMS["fps"])
    parser.add_argument("--seed", type=int, default=bag_example.PARAMS["seed"])
    parser.add_argument("--ka-min", type=float, default=1.0e4)
    parser.add_argument("--ka-max", type=float, default=1.0e6)
    parser.add_argument("--ka-count", type=int, default=6)
    parser.add_argument("--ke-scales", type=float, nargs="+", default=[10.0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ka", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ke", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--merge-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ka_values = np.logspace(np.log10(args.ka_min), np.log10(args.ka_max), args.ka_count)

    if args.worker:
        if args.ka is None or args.ke is None:
            raise ValueError("--worker requires --ka and --ke")
        _run_case(
            output_dir=args.output_dir,
            ka=float(args.ka),
            ke=float(args.ke),
            frames=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
        )
        return

    cases: list[dict[str, object]] = []
    for ke_scale in args.ke_scales:
        for ka in ka_values:
            ke = float(ke_scale) * float(ka)
            cases.append(
                {
                    "cloth_tri_ka": float(ka),
                    "cloth_tri_ke": ke,
                    "ke_scale": float(ke_scale),
                    "video": str(args.output_dir / f"{_case_slug(float(ka), ke)}.mp4"),
                    "label": f"ka={_sci(float(ka))}  ke={_sci(ke)}",
                }
            )

    if not args.merge_only:
        script_path = Path(__file__).resolve()
        for case in cases:
            cmd = [
                sys.executable,
                str(script_path),
                "--worker",
                "--output-dir",
                str(args.output_dir),
                "--frames",
                str(args.frames),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--fps",
                str(args.fps),
                "--seed",
                str(args.seed),
                "--ka",
                str(case["cloth_tri_ka"]),
                "--ke",
                str(case["cloth_tri_ke"]),
            ]
            print(f"[sweep] Launching worker: {case['label']}", flush=True)
            subprocess.run(cmd, check=True)

    grid_path = args.output_dir / "vbd_bag_stiffness_sweep_2x3.mp4"
    print(f"[sweep] Merging 2x3 grid -> {grid_path}", flush=True)
    _merge_grid(cases, grid_path, fps=args.fps, frames=args.frames)

    manifest = {
        "frames": args.frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "grid_video": str(grid_path),
        "cases": cases,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[sweep] Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
