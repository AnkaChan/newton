# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render saved inertia-scale trajectories as one simultaneous three-by-five grid."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import warp as wp
from render_rho_sweep import CAMERA, FPS, Bag, Capture, formatted, load_case, visible_fraction

ROOT = Path(__file__).resolve().parent
STIFFNESSES = (1000, 100000, 10000000)
MODES = ("off", "inertia1", "inertia10", "inertia100", "inertia1000")
LABELS = ("ALM off", "rho = 1 x inertia", "rho = 10 x inertia", "rho = 100 x inertia", "rho = 1000 x inertia")
COLORS = ((241, 169, 101), (93, 213, 212), (132, 221, 163), (237, 206, 117), (169, 149, 245))
WIDTH, HEIGHT = 3240, 1680
LEFT, TOP, CELL_WIDTH, ROW_HEIGHT, CELL_HEIGHT = 240, 144, 600, 512, 476
BACKGROUND, MUTED = (18, 25, 36), (187, 199, 215)
MOVIE = "inertia_scale_sweep_grid.mp4"


def panel_image(scene, result, data, frame, fonts):
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (640, 508), BACKGROUND)
    panel.paste(scene, (0, 44))
    draw = ImageDraw.Draw(panel)
    mode = result["mode"]
    column = MODES.index(mode)
    draw.text((16, 8), LABELS[column], font=fonts["panel"], fill=COLORS[column])
    row = result["rows"][frame]
    if not math.isclose(row["time_s"], frame / FPS, abs_tol=1e-6):
        raise ValueError(f"Unsynchronized frame {frame}: {mode}")
    stretch = row.get("stretch_score")
    stretch_text = formatted(None if stretch is None else 100.0 * stretch, "%")
    lines = (
        f"Shape: stretch RMS {stretch_text} | bend {formatted(row.get('bend_score'))} rad",
        f"Original force residual RMS: {formatted(row.get('original_rms_N'))} N",
    )
    for y, line in zip((435, 459), lines, strict=True):
        draw.text((16, y), line, font=fonts["metric"], fill="white")
    if mode == "off":
        ratios = "k_eff/k: 1 / 1 / 1 (original material)"
    else:
        values = [
            row.get(family, {}).get("effective_over_k", {}).get("median")
            for family in ("tri_stretch", "tri_area", "bend")
        ]
        ratios = "Median k_eff/k: " + " / ".join(formatted(value) for value in values)
    draw.text((16, 483), ratios, font=fonts["small"], fill=MUTED)
    if not bool(data["valid"][frame]) or row.get("status") == "held_after_failure":
        draw.rectangle((2, 2, 637, 505), outline=(255, 95, 103), width=5)
        draw.rectangle((16, 183, 624, 286), fill=(99, 25, 35))
        failure_frame = result.get("failure_frame", frame)
        draw.text((30, 196), f"FAILED AT {failure_frame / FPS:.2f} s", font=fonts["panel"], fill="white")
        draw.text((30, 231), "Last finite pose held; no further simulation", font=fonts["metric"], fill="white")
        reason = str(result.get("failure_reason") or result.get("reason") or "See case JSON for details")
        while draw.textlength(reason, font=fonts["small"]) > 580:
            reason = reason[:-4] + "..."
        draw.text((30, 259), reason, font=fonts["small"], fill=(255, 206, 210))
    else:
        warnings = []
        fraction = visible_fraction(data["particle_q"][frame])
        if fraction < 1.0:
            warnings.append(f"Fixed camera clips {100.0 * (1.0 - fraction):.1f}% of cloth vertices")
        bodies = data["body_q"][frame, :, :3]
        outside = round(len(bodies) * (1.0 - visible_fraction(bodies)))
        if outside:
            warnings.append(f"Contents outside view: {outside}/{len(bodies)} (body centers)")
        for index, warning in enumerate(warnings):
            y = 55 + 29 * index
            draw.rectangle((12, y, 628, y + 29), fill=(107, 69, 9))
            draw.text((24, y + 4), warning, font=fonts["metric"], fill="white")
    return panel.resize((CELL_WIDTH, CELL_HEIGHT), Image.Resampling.LANCZOS)


def grid_background(frame, fonts):
    from PIL import Image, ImageDraw

    grid = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(grid)
    draw.text((24, 12), "Inertia-scaled ALM: three materials x five settings", font=fonts["title"], fill="white")
    draw.text(
        (24, 61),
        "rho = c * rho_inertia | no material-stiffness floor | bend_ke = 200 | k_eff/k: stretch / area / bending",
        font=fonts["legend"],
        fill=MUTED,
    )
    phase = "settling" if frame <= 60 else "wiggling"
    draw.text((2765, 17), f"t = {frame / FPS:.2f} s | {phase}", font=fonts["column"], fill="white")
    for column, label in enumerate(LABELS):
        draw.text(
            (LEFT + (column + 0.5) * CELL_WIDTH, 100), label, anchor="mt", font=fonts["column"], fill=COLORS[column]
        )
    for row, stiffness in enumerate(STIFFNESSES):
        y = TOP + row * ROW_HEIGHT
        draw.rectangle((0, y, WIDTH - 1, y + ROW_HEIGHT - 1), outline=(65, 80, 100), width=3)
        draw.rectangle((0, y, LEFT - 1, y + ROW_HEIGHT - 1), fill=(26, 39, 58))
        exponent = int(np.log10(stiffness))
        draw.text((22, y + 116), "tri_ke", font=fonts["row_key"], fill=MUTED)
        draw.text((20, y + 164), f"1e{exponent}", font=fonts["row_value"], fill="white")
        draw.text((22, y + 266), "tri_ka", font=fonts["row_key"], fill=MUTED)
        draw.text((22, y + 313), f"2e{exponent - 1}", font=fonts["area"], fill="white")
    return grid


def render(args):
    from PIL import Image, ImageFont

    wp.init()
    wp.config.quiet = True
    loaded = {
        (stiffness, mode): load_case(args.output, stiffness, mode, args.frames)
        for stiffness in STIFFNESSES
        for mode in MODES
    }
    sim = Bag(STIFFNESSES[0], False)
    reference = loaded[STIFFNESSES[0], MODES[0]][1]
    for _result, data in loaded.values():
        np.testing.assert_array_equal(data["particle_q"][0], reference["particle_q"][0])
        np.testing.assert_array_equal(data["body_q"][0], reference["body_q"][0])
        assert data["particle_q"].shape[1:] == (sim.model.particle_count, 3)
    font_sizes = {
        "title": 40,
        "legend": 27,
        "column": 32,
        "row_key": 32,
        "row_value": 76,
        "area": 48,
        "panel": 24,
        "metric": 19,
        "small": 17,
    }
    fonts = {
        name: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        for name, size in font_sizes.items()
    }
    pins = wp.empty(len(sim.pins), dtype=wp.vec3, device=sim.model.device)
    colors = wp.full(len(sim.pins), wp.vec3(1.0, 0.65, 0.08), dtype=wp.vec3, device=sim.model.device)
    frames = sorted(set(args.preview_frames)) if args.preview else range(1, args.frames + 1)
    writer = None
    with Capture(out_dir=str(args.output), width=640, height=384, **CAMERA) as cap:
        viewer = cap._get_viewer(sim.model)
        viewer.show_ui = False
        viewer.renderer.draw_wireframe = True
        cap._apply_camera(viewer)
        if not args.preview:
            writer = imageio.get_writer(
                args.output / MOVIE,
                fps=FPS,
                quality=None,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                output_params=["-crf", "18", "-preset", "medium", "-threads", "4", "-movflags", "+faststart"],
            )
        try:
            for frame in frames:
                grid = grid_background(frame, fonts)
                for row, stiffness in enumerate(STIFFNESSES):
                    for column, mode in enumerate(MODES):
                        result, data = loaded[stiffness, mode]
                        sim.state_0.particle_q.assign(data["particle_q"][frame])
                        sim.state_0.body_q.assign(data["body_q"][frame])
                        pins.assign(data["particle_q"][frame, sim.info["top_global_indices"]])
                        viewer.begin_frame(frame / FPS)
                        viewer.log_state(sim.state_0)
                        viewer.log_points("pinned_rim", pins, radii=0.0023, colors=colors)
                        viewer.end_frame()
                        panel = panel_image(Image.fromarray(viewer.get_frame().numpy()), result, data, frame, fonts)
                        grid.paste(panel, (LEFT + column * CELL_WIDTH, TOP + row * ROW_HEIGHT + 18))
                if writer is not None:
                    writer.append_data(np.asarray(grid))
                if args.preview or frame in (1, 60, 180, 360):
                    grid.save(args.output / f"inertia_scale_sweep_grid_frame{frame:03d}.png")
                if args.preview or frame % 60 == 0:
                    print(f"inertia grid frame {frame}/{args.frames}", flush=True)
        finally:
            if writer is not None:
                writer.close()
    if not args.preview:
        validate(args.output, args.frames)
        build_viewer(args.output)


def validate(output, expected_frames):
    path = output / MOVIE
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count == expected_frames and abs(seconds - expected_frames / FPS) < 0.04, (count, seconds)
    samples = []
    for time in (0.0, seconds / 2.0, seconds - 0.1):
        reader = imageio_ffmpeg.read_frames(
            str(path), input_params=["-ss", str(time)], output_params=["-frames:v", "1"]
        )
        try:
            metadata = next(reader)
            assert metadata["size"] == (WIDTH, HEIGHT) and metadata["fps"] == FPS, metadata
            pixels = np.frombuffer(next(reader), dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        finally:
            reader.close()
        deviations = []
        for row in range(3):
            for column in range(5):
                x, y = LEFT + column * CELL_WIDTH, TOP + row * ROW_HEIGHT + 60
                scene = pixels[y : y + 350, x : x + CELL_WIDTH]
                deviations.append(float(scene.std()))
                assert scene.std() > 2.0 and scene.mean() > 2.0, (time, row, column)
        samples.append({"time_s": time, "scene_pixel_std": deviations})
    result = {
        "file": MOVIE,
        "frames": count,
        "duration_s": seconds,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "codec": "H.264",
        "crf": 18,
        "bytes": path.stat().st_size,
        "rows_tri_ke": STIFFNESSES,
        "columns": LABELS,
        "camera": CAMERA,
        "samples": samples,
        "source_frame_mapping": "Output frame n renders saved snapshot n from every independent trajectory, without interpolation.",
    }
    (output / "video_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def build_viewer(output):
    rows = []
    for stiffness in STIFFNESSES:
        links = " | ".join(
            f'<a href="ke{stiffness}_{mode}.json">{html.escape(label)}</a>'
            for mode, label in zip(MODES, LABELS, strict=True)
        )
        rows.append(f"<p><b>tri_ke = {stiffness:.0e}:</b> {links}</p>")
    validations = " | ".join(
        f'<a href="{name}">{title}</a>'
        for name, title in (
            ("validation.json", "Trajectory validation"),
            ("force-validation.json", "Force diagnostic validation"),
            ("video_validation.json", "Video validation"),
        )
        if (output / name).exists()
    )
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bag trajectories: inertia-scaled ALM</title><style>
:root{color-scheme:dark}body{margin:0;background:#101722;color:#edf2f7;font:17px/1.55 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px 24px 60px}h1{font-size:36px;line-height:1.15}p{max-width:1300px}a{color:#8edcdd}video{width:100%;aspect-ratio:27/14;background:#000;display:block}small,.muted{color:#b9c7d7}code{background:#233247;padding:2px 5px;border-radius:4px}details{font-size:15px}
</style></head><body><main><h1>Inertia-scaled ALM: three materials x five settings</h1>
<p>All 15 trajectories appear simultaneously at synchronized times. Rows use triangle stretch stiffness <code>tri_ke = 1e3, 1e5, 1e7</code>. Columns show ALM off, then <code>rho = c * rho_inertia</code> for <code>c = 1, 10, 100, 1000</code>. There is no material-stiffness floor.</p>
<video controls preload="metadata" poster="inertia_scale_sweep_grid_frame180.png" src="inertia_scale_sweep_grid.mp4"></video>
<p><a href="inertia_scale_sweep_grid.mp4">Open or download the six-second grid at full resolution</a>. Use fullscreen to inspect the per-panel labels.</p>
<p><b>Matched settings:</b> 360 frames at 60 fps; 10 substeps per frame; 10 VBD iterations per substep; tri_ka = 0.2 tri_ke; bending stiffness = 200. The first second settles the bag; the next five seconds move its pinned rim. All cases share the fixed 30-degree camera, initial scene, rim motion and 16-times-baseline self-contact storage. Each case evolves independently and retains its own multiplier history.</p>
<p><b>Read the labels separately:</b> stretch and bend RMS measure physical deformation. The original material force residual measures particle force balance in newtons after the final particle iteration of the final substep, before velocity finalization. It includes original elastic forces, material damping, fresh native contacts and inertia. It excludes rigid-body residuals and DAT constraint reactions, so it is not a full coupled-system KKT residual. This is a matched-settings trajectory comparison, not an equal-state convergence benchmark.</p>
<p>Actual median <code>k_eff/k</code> values are shown separately for triangle stretch, triangle area and bending, in that order. ALM off uses the original material stiffness. Amber labels identify cloth clipping or contents outside the fixed view. Freely falling rigid contents do not stop the simulation. Failed cases remain visible with their last finite pose held and a red failure label.</p>
<details><summary>Per-frame measurements and validation</summary>__ROWS__<p>__VALIDATIONS__</p></details>
<p class="muted">This scene reconstructs the documented May pinned-bag fixture, with three rigid contents and no ground or gripper; the original untracked fixture was unavailable. Saved particle and rigid-body snapshots are rendered directly, with no trajectory interpolation. This is a visual comparison, not a runtime benchmark.</p>
<p><a href="../render_inertia_sweep.py">Renderer and viewer source</a> | <a href="../results-rho-video-sweep/index.html">Previous material-floor sweep</a></p><small>Local files only. No external scripts, fonts, telemetry or network requests.</small></main></body></html>"""
    (output / "index.html").write_text(
        page.replace("__ROWS__", "\n".join(rows)).replace("__VALIDATIONS__", validations)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results-inertia-video-sweep")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-frames", nargs="+", type=int, default=[180])
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.html_only:
        build_viewer(args.output)
    else:
        render(args)
