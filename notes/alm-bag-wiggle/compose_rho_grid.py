# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rearrange existing six-panel movies into a simultaneous material-by-floor grid."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from render_rho_sweep import build_viewer

ROOT = Path(__file__).resolve().parent
STIFFNESSES = (1000, 100000, 10000000)
COLUMNS = ("ALM off", "rho floor 9k", "rho floor 1k", "rho floor 0.1k", "rho floor 0.01k", "inertia only (floor 0)")
COLORS = ((241, 169, 101), (93, 213, 212), (132, 221, 163), (237, 206, 117), (241, 138, 161), (169, 149, 245))
WIDTH, HEIGHT, FPS, FRAMES = 3840, 1680, 60, 360
LEFT, TOP, CELL_WIDTH, ROW_HEIGHT, CELL_HEIGHT = 240, 144, 600, 512, 476


def make_grid(frames, frame_number, fonts):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), (18, 25, 36))
    draw = ImageDraw.Draw(image)
    draw.text((24, 12), "Three material rows x six rho-floor settings", font=fonts["title"], fill="white")
    draw.text(
        (24, 61),
        "rho = max(rho_inertia, m*k) | bend_ke = 200 | k_eff/k order: triangle stretch / triangle area / bending",
        font=fonts["legend"],
        fill=(187, 199, 215),
    )
    phase = "settling" if frame_number <= 60 else "wiggling"
    draw.text((3370, 17), f"t = {frame_number / FPS:.2f} s | {phase}", font=fonts["column"], fill="white")
    for column, (title, color) in enumerate(zip(COLUMNS, COLORS, strict=True)):
        x = LEFT + column * CELL_WIDTH
        draw.text((x + CELL_WIDTH / 2, 100), title, anchor="mt", font=fonts["column"], fill=color)
    for row, (stiffness, pixels) in enumerate(zip(STIFFNESSES, frames, strict=True)):
        y = TOP + row * ROW_HEIGHT
        draw.rectangle((0, y, WIDTH - 1, y + ROW_HEIGHT - 1), outline=(65, 80, 100), width=3)
        draw.rectangle((0, y, LEFT - 1, y + ROW_HEIGHT - 1), fill=(26, 39, 58))
        draw.text((22, y + 116), "tri_ke", font=fonts["row_key"], fill=(187, 199, 215))
        draw.text((20, y + 164), f"1e{int(np.log10(stiffness))}", font=fonts["row_value"], fill="white")
        draw.text((22, y + 266), "tri_ka", font=fonts["row_key"], fill=(187, 199, 215))
        draw.text((22, y + 313), f"2e{int(np.log10(stiffness)) - 1}", font=fonts["area"], fill="white")
        source = Image.fromarray(pixels)
        for column in range(6):
            source_x = (column % 3) * 640
            source_y = 64 + (column // 3) * 508
            panel = source.crop((source_x, source_y, source_x + 640, source_y + 508))
            panel = panel.resize((CELL_WIDTH, CELL_HEIGHT), Image.Resampling.LANCZOS)
            image.paste(panel, (LEFT + column * CELL_WIDTH, y + (ROW_HEIGHT - CELL_HEIGHT) // 2))
    return image


def validate(path):
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count == FRAMES and abs(seconds - 6.0) < 0.04, (count, seconds)
    reader = imageio_ffmpeg.read_frames(str(path), output_params=["-frames:v", "1"])
    try:
        metadata = next(reader)
        assert metadata["size"] == (WIDTH, HEIGHT) and metadata["fps"] == FPS, metadata
        pixels = np.frombuffer(next(reader), dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    finally:
        reader.close()
    for row in range(3):
        for column in range(6):
            x = LEFT + column * CELL_WIDTH
            y = TOP + row * ROW_HEIGHT + 60
            assert pixels[y : y + 350, x : x + CELL_WIDTH].std() > 2.0, (row, column)
    result = {
        "file": path.name,
        "frames": count,
        "duration_s": seconds,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "codec": "H.264",
        "crf": 18,
        "bytes": path.stat().st_size,
        "rows_tri_ke": STIFFNESSES,
        "columns": COLUMNS,
        "source_frame_mapping": "Output frame n uses frame n from each source movie, without interpolation.",
    }
    path.with_name("grid_video_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main(args):
    from PIL import ImageFont

    fonts = {
        name: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        for name, size in (
            ("title", 40),
            ("legend", 27),
            ("column", 32),
            ("row_key", 32),
            ("row_value", 76),
            ("area", 48),
        )
    }
    readers = []
    writer = None
    try:
        for stiffness in STIFFNESSES:
            params = ["-threads", "2"]
            if args.preview:
                params += ["-ss", str((args.preview_frame - 1) / FPS)]
            reader = imageio_ffmpeg.read_frames(str(args.output / f"ke{stiffness}_rho_sweep.mp4"), input_params=params)
            metadata = next(reader)
            assert metadata["size"] == (1920, 1080) and metadata["fps"] == FPS, metadata
            readers.append(reader)
        if args.preview:
            numbers = [args.preview_frame]
        else:
            numbers = range(1, FRAMES + 1)
            writer = imageio.get_writer(
                args.output / "rho_floor_sweep_grid.mp4",
                fps=FPS,
                quality=None,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                output_params=["-crf", "18", "-preset", "medium", "-threads", "4", "-movflags", "+faststart"],
            )
        for number in numbers:
            frames = [np.frombuffer(next(reader), dtype=np.uint8).reshape(1080, 1920, 3) for reader in readers]
            grid = make_grid(frames, number, fonts)
            if writer is not None:
                writer.append_data(np.asarray(grid))
            if args.preview or number in (1, 180, 360):
                grid.save(args.output / f"rho_floor_sweep_grid_frame{number:03d}.png")
            if args.preview or number % 60 == 0:
                print(f"grid frame {number}/{FRAMES}", flush=True)
    finally:
        for reader in readers:
            reader.close()
        if writer is not None:
            writer.close()
    if not args.preview:
        validate(args.output / "rho_floor_sweep_grid.mp4")
        build_viewer(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results-rho-video-sweep")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-frame", type=int, default=180)
    main(parser.parse_args())
