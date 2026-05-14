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


def _case_slug(edge_ke: float) -> str:
    return f"edge_ke_{_sci(edge_ke).replace('+', '')}".replace("-", "m")


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    pad = 10
    bbox = draw.multiline_textbbox((0, 0), label, spacing=4)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.rectangle((pad - 4, pad - 4, pad + w + 8, pad + h + 8), fill=(0, 0, 0, 150))
    draw.multiline_text((pad, pad), label, fill=(255, 255, 255, 255), spacing=4)
    return np.asarray(image)


def _run_case(
    *,
    output_dir: Path,
    edge_ke: float,
    frames: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
) -> dict[str, object]:
    params = dict(bag_example.PARAMS)
    params["cloth_edge_ke"] = float(edge_ke)
    params["settle_frames"] = int(frames)
    bag_example.PARAMS = params

    slug = _case_slug(edge_ke)
    video_path = output_dir / f"{slug}.mp4"
    edge_kd = float(params["cloth_edge_kd"])
    tri_ke = float(params["cloth_tri_ke"])
    tri_ka = float(params["cloth_tri_ka"])
    label = f"edge_ke={_sci(edge_ke)}  edge_kd={_sci(edge_kd)}\ntri_ke={_sci(tri_ke)}  tri_ka={_sci(tri_ka)}"
    print(f"[edge-sweep] Running {label} -> {video_path}", flush=True)

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
                    print(f"[edge-sweep]   {slug}: {frame_index + 1}/{frames} frames", flush=True)
    finally:
        viewer.close()

    return {
        "cloth_edge_ke": float(edge_ke),
        "cloth_edge_kd": edge_kd,
        "cloth_tri_ke": float(tri_ke),
        "cloth_tri_ka": float(tri_ka),
        "video": str(video_path),
        "label": label,
    }


def _merge_row(cases: list[dict[str, object]], output_path: Path, *, fps: int, frames: int) -> None:
    if len(cases) != 3:
        raise ValueError(f"Expected exactly 3 cases for a 1x3 row, got {len(cases)}")

    readers = [imageio.get_reader(case["video"]) for case in cases]
    try:
        first = readers[0].get_data(0)
        h, w = first.shape[:2]
        with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index in range(frames):
                row = np.zeros((h, 3 * w, 3), dtype=np.uint8)
                for case_index, reader in enumerate(readers):
                    frame = reader.get_data(frame_index)
                    row[:, case_index * w : (case_index + 1) * w] = frame[:, :, :3]
                writer.append_data(row)
                if (frame_index + 1) % max(1, frames // 10) == 0:
                    print(f"[edge-sweep]   row: {frame_index + 1}/{frames} frames", flush=True)
    finally:
        for reader in readers:
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a 1x3 VBD bag Franka edge stiffness sweep.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vbd_bag_franka_edge_ke_sweep"))
    parser.add_argument("--frames", type=int, default=bag_example.PARAMS["settle_frames"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=bag_example.PARAMS["fps"])
    parser.add_argument("--seed", type=int, default=bag_example.PARAMS["seed"])
    parser.add_argument("--edge-ke-min", type=float, default=20.0)
    parser.add_argument("--edge-ke-max", type=float, default=2000.0)
    parser.add_argument("--edge-ke-count", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--edge-ke", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--merge-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_ke_values = np.logspace(np.log10(args.edge_ke_min), np.log10(args.edge_ke_max), args.edge_ke_count)

    if args.worker:
        if args.edge_ke is None:
            raise ValueError("--worker requires --edge-ke")
        _run_case(
            output_dir=args.output_dir,
            edge_ke=float(args.edge_ke),
            frames=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
        )
        return

    cases: list[dict[str, object]] = []
    for edge_ke_value in edge_ke_values:
        edge_ke = float(edge_ke_value)
        edge_kd = float(bag_example.PARAMS["cloth_edge_kd"])
        tri_ke = float(bag_example.PARAMS["cloth_tri_ke"])
        tri_ka = float(bag_example.PARAMS["cloth_tri_ka"])
        cases.append(
            {
                "cloth_edge_ke": edge_ke,
                "cloth_edge_kd": edge_kd,
                "cloth_tri_ke": tri_ke,
                "cloth_tri_ka": tri_ka,
                "video": str(args.output_dir / f"{_case_slug(edge_ke)}.mp4"),
                "label": (
                    f"edge_ke={_sci(edge_ke)}  edge_kd={_sci(edge_kd)}\ntri_ke={_sci(tri_ke)}  tri_ka={_sci(tri_ka)}"
                ),
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
                "--edge-ke",
                str(case["cloth_edge_ke"]),
            ]
            print(f"[edge-sweep] Launching worker: {case['label']}", flush=True)
            subprocess.run(cmd, check=True)

    row_path = args.output_dir / "vbd_bag_franka_edge_ke_sweep_1x3.mp4"
    print(f"[edge-sweep] Merging 1x3 row -> {row_path}", flush=True)
    _merge_row(cases, row_path, fps=args.fps, frames=args.frames)

    manifest = {
        "frames": args.frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "row_video": str(row_path),
        "cases": cases,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[edge-sweep] Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
