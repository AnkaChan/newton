# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001 -- Mathematical symbols and punctuation in HTML.

"""Render six synchronized rho-floor trajectories and build an offline video viewer."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import warp as wp
from run_case import ROOT, Bag

sys.path.insert(0, str(Path(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")) / "Newton/tools"))
from newton_capture import Capture

MODES = ("off", "floor9", "floor1", "floor0p1", "floor0p01", "inertia")
FLOORS = {"off": None, "floor9": 9.0, "floor1": 1.0, "floor0p1": 0.1, "floor0p01": 0.01, "inertia": 0.0}
COLORS = ((241, 169, 101), (93, 213, 212), (132, 221, 163), (237, 206, 117), (241, 138, 161), (169, 149, 245))
STIFFNESSES = (1000, 100000, 10000000)
CAMERA = {"camera_pos": (0.50, -0.84, 0.46), "camera_target": (0.0, 0.0, 0.14), "camera_fov": 30.0}
FPS = 60
WIDTH, HEIGHT = 1920, 1080
PANEL_WIDTH, PANEL_HEIGHT = 640, 508
SCENE_HEIGHT, PANEL_HEADER, HEADER = 384, 44, 64
BACKGROUND = (18, 25, 36)
MUTED = (187, 199, 215)


def label(mode):
    if mode == "off":
        return "ALM OFF"
    suffix = " (inertia only)" if mode == "inertia" else ""
    return f"ALM | m_floor = {FLOORS[mode]:g}{suffix}"


def formatted(value, suffix=""):
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.3g}{suffix}"


def visible_fraction(points):
    camera = np.array(CAMERA["camera_pos"])
    forward = np.array(CAMERA["camera_target"]) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - camera
    depth = relative @ forward
    half_height = depth * np.tan(np.radians(CAMERA["camera_fov"]) / 2.0)
    inside = (depth > 0.0) & (np.abs(relative @ up) < half_height)
    inside &= np.abs(relative @ right) < half_height * PANEL_WIDTH / SCENE_HEIGHT
    return float(inside.mean())


def load_case(output, stiffness, mode, frames):
    stem = f"ke{int(stiffness)}_{mode}"
    result = json.loads((output / f"{stem}.json").read_text())
    with np.load(output / f"{stem}.npz") as archive:
        data = {name: archive[name] for name in ("particle_q", "body_q", "frame", "valid")}
    np.testing.assert_array_equal(data["frame"][: frames + 1], np.arange(frames + 1))
    if len(data["frame"]) < frames + 1 or len(result["rows"]) < frames + 1:
        raise ValueError(f"Incomplete trajectory: {stem}")
    if not np.isfinite(data["particle_q"]).all() or not np.isfinite(data["body_q"]).all():
        raise ValueError(f"Trajectory must hold its last finite pose after failure: {stem}")
    result["mode"] = mode
    return result, data


def draw_panel(scene, result, data, frame, fonts, preview=False):
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), BACKGROUND)
    panel.paste(scene, (0, PANEL_HEADER))
    draw = ImageDraw.Draw(panel)
    mode = result["mode"]
    color = COLORS[MODES.index(mode)]
    title = label(mode) + (" | pilot" if preview else "")
    draw.text((16, 8), title, font=fonts["panel"], fill=color)
    row = result["rows"][frame]
    if not math.isclose(row["time_s"], frame / FPS, abs_tol=1e-6):
        raise ValueError(f"Unsynchronized frame {frame}: {mode}")
    stretch = row.get("stretch_score")
    stretch_text = formatted(None if stretch is None else 100.0 * stretch, "%")
    draw.text(
        (16, 435),
        f"Shape: stretch RMS {stretch_text} | bend {formatted(row.get('bend_score'))} rad",
        font=fonts["metric"],
        fill="white",
    )
    draw.text(
        (16, 459),
        f"Original force residual RMS: {formatted(row.get('original_rms_N'))} N",
        font=fonts["metric"],
        fill="white",
    )
    if mode == "off":
        ratio_text = "k_eff/k: 1 / 1 / 1 (original material)"
    else:
        ratios = [
            row.get(family, {}).get("effective_over_k", {}).get("median")
            for family in ("tri_stretch", "tri_area", "bend")
        ]
        ratio_text = "Median k_eff/k: " + " / ".join(formatted(ratio) for ratio in ratios)
    draw.text((16, 483), ratio_text, font=fonts["small"], fill=MUTED)
    failed = not bool(data["valid"][frame]) or row.get("status") == "held_after_failure"
    if failed:
        draw.rectangle((2, 2, PANEL_WIDTH - 3, PANEL_HEIGHT - 3), outline=(255, 95, 103), width=5)
        draw.rectangle((16, 183, PANEL_WIDTH - 16, 286), fill=(99, 25, 35))
        failure_frame = result.get("failure_frame", frame)
        draw.text((30, 196), f"FAILED AT {failure_frame / FPS:.2f} s", font=fonts["panel"], fill="white")
        draw.text((30, 231), "Last finite pose held; no further simulation", font=fonts["metric"], fill="white")
        reason = result.get("failure_reason") or result.get("reason") or "See case JSON for details"
        reason = str(reason)
        while draw.textlength(reason, font=fonts["small"]) > PANEL_WIDTH - 60:
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
            draw.rectangle((12, y, PANEL_WIDTH - 12, y + 29), fill=(107, 69, 9))
            draw.text((24, y + 4), warning, font=fonts["metric"], fill="white")
    draw.line((PANEL_WIDTH - 1, 0, PANEL_WIDTH - 1, PANEL_HEIGHT), fill=(53, 64, 80), width=2)
    draw.line((0, PANEL_HEIGHT - 1, PANEL_WIDTH, PANEL_HEIGHT - 1), fill=(53, 64, 80), width=2)
    return panel


def render(args):
    from PIL import Image, ImageDraw, ImageFont

    wp.init()
    wp.config.quiet = True
    sim = Bag(args.stiffness, False)
    source_modes = args.preview_modes if args.preview else MODES
    loaded = {mode: load_case(args.output, args.stiffness, mode, args.frames) for mode in source_modes}
    reference_result, reference_data = loaded[source_modes[0]]
    for result, data in loaded.values():
        np.testing.assert_array_equal(data["particle_q"][0], reference_data["particle_q"][0])
        np.testing.assert_array_equal(data["body_q"][0], reference_data["body_q"][0])
        if "model_sha256" in reference_result:
            assert result["model_sha256"] == reference_result["model_sha256"]
        assert data["particle_q"].shape[1:] == (sim.model.particle_count, 3)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fonts = {
        name: ImageFont.truetype(font_path, size)
        for name, size in (("title", 27), ("panel", 24), ("metric", 19), ("small", 17))
    }
    pins = wp.empty(len(sim.pins), dtype=wp.vec3, device=sim.model.device)
    colors = wp.full(len(sim.pins), wp.vec3(1.0, 0.65, 0.08), dtype=wp.vec3, device=sim.model.device)
    stem = f"ke{int(args.stiffness)}"
    movie = args.output / f"{stem}_rho_sweep.mp4"
    rendered_frames = (
        sorted({min(frame, args.frames) for frame in args.preview_frames})
        if args.preview
        else range(1, args.frames + 1)
    )
    writer = None
    with Capture(out_dir=str(args.output), width=PANEL_WIDTH, height=SCENE_HEIGHT, **CAMERA) as cap:
        viewer = cap._get_viewer(sim.model)
        viewer.show_ui = False
        viewer.renderer.draw_wireframe = True
        cap._apply_camera(viewer)
        if not args.preview:
            writer = imageio.get_writer(
                movie,
                fps=FPS,
                quality=None,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
            )
        try:
            for frame in rendered_frames:
                combined = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
                draw = ImageDraw.Draw(combined)
                draw.text(
                    (18, 7),
                    f"rho floor sweep | tri_ke = {args.stiffness:.0e} | tri_ka = {0.2 * args.stiffness:.0e} | bend_ke = 200",
                    font=fonts["title"],
                    fill="white",
                )
                phase = "settling" if frame <= 60 else "wiggling"
                draw.text((1540, 9), f"t = {frame / FPS:.2f} s | {phase}", font=fonts["panel"], fill="white")
                subtitle = "10 substeps x 10 iterations | fixed camera | k_eff/k order: triangle stretch / triangle area / bending"
                if args.preview:
                    subtitle = (
                        "LAYOUT PREVIEW: available pilot trajectories repeated; this is not the six-method comparison"
                    )
                draw.text((18, 40), subtitle, font=fonts["small"], fill=MUTED)
                for index in range(6):
                    mode = source_modes[index % len(source_modes)] if args.preview else MODES[index]
                    result, data = loaded[mode]
                    sim.state_0.particle_q.assign(data["particle_q"][frame])
                    sim.state_0.body_q.assign(data["body_q"][frame])
                    pins.assign(data["particle_q"][frame, sim.info["top_global_indices"]])
                    viewer.begin_frame(frame / FPS)
                    viewer.log_state(sim.state_0)
                    viewer.log_points("pinned_rim", pins, radii=0.0023, colors=colors)
                    viewer.end_frame()
                    scene = Image.fromarray(viewer.get_frame().numpy())
                    panel = draw_panel(scene, result, data, frame, fonts, preview=args.preview)
                    combined.paste(panel, ((index % 3) * PANEL_WIDTH, HEADER + (index // 3) * PANEL_HEIGHT))
                if writer is not None:
                    writer.append_data(np.asarray(combined))
                if args.preview or frame in (1, 60, 180, 360):
                    kind = "preview" if args.preview else "rho_sweep"
                    combined.save(args.output / f"{stem}_{kind}_frame{frame:03d}.png")
                if args.preview or frame % 60 == 0:
                    print(f"render {stem}: {frame}/{args.frames}", flush=True)
        finally:
            if writer is not None:
                writer.close()
    if not args.preview:
        validation = validate_video(movie, args.frames)
        (args.output / f"{stem}_video_validation.json").write_text(json.dumps(validation, indent=2) + "\n")


def validate_video(path, expected_frames):
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count == expected_frames, (path, count, expected_frames)
    assert abs(seconds - expected_frames / FPS) < 0.04, (path, seconds)
    samples = []
    for timestamp in (0.0, max(0.0, seconds / 2.0), max(0.0, seconds - 0.1)):
        reader = imageio_ffmpeg.read_frames(
            str(path), input_params=["-ss", str(timestamp)], output_params=["-frames:v", "1"]
        )
        try:
            metadata = next(reader)
            assert metadata["size"] == (WIDTH, HEIGHT), metadata
            assert abs(metadata["fps"] - FPS) < 1e-6, metadata
            pixels = np.frombuffer(next(reader), dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        finally:
            reader.close()
        deviations = []
        for index in range(6):
            x = (index % 3) * PANEL_WIDTH
            y = HEADER + (index // 3) * PANEL_HEIGHT + PANEL_HEADER
            scene = pixels[y : y + SCENE_HEIGHT, x : x + PANEL_WIDTH]
            deviations.append(float(scene.std()))
            assert scene.std() > 2.0 and scene.mean() > 2.0, (path, timestamp, index, "empty GL view")
        samples.append({"time_s": timestamp, "scene_pixel_std": deviations})
    result = {
        "file": path.name,
        "frames": count,
        "duration_s": seconds,
        "fps": FPS,
        "codec": "H.264",
        "crf": 18,
        "width": WIDTH,
        "height": HEIGHT,
        "bytes": path.stat().st_size,
        "samples": samples,
    }
    print(json.dumps(result), flush=True)
    return result


def compose(output):
    movies = [output / f"ke{stiffness}_rho_sweep.mp4" for stiffness in STIFFNESSES]
    validations = [validate_video(movie, 360) for movie in movies]
    combined = output / "rho_floor_sweep_all_materials.mp4"
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        directory = Path(temporary)
        listing = directory / "movies.txt"
        listing.write_text("".join(f"file '{movie.resolve()}'\n" for movie in movies))
        metadata = directory / "chapters.txt"
        chapters = [";FFMETADATA1\n"]
        for index, stiffness in enumerate(STIFFNESSES):
            chapters.append(
                f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={6000 * index}\nEND={6000 * (index + 1)}\ntitle=tri_ke {stiffness:.0e}\n"
            )
        metadata.write_text("".join(chapters))
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-i",
                str(metadata),
                "-map",
                "0:v:0",
                "-map_metadata",
                "1",
                "-map_chapters",
                "1",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                str(combined),
            ],
            check=True,
        )
    validations.append(validate_video(combined, 1080))
    (output / "video_validation.json").write_text(
        json.dumps({"videos": validations, "camera": CAMERA}, indent=2) + "\n"
    )
    build_viewer(output)


def build_viewer(output):
    cases = []
    for stiffness in STIFFNESSES:
        stem = f"ke{stiffness}"
        links = " | ".join(f'<a href="{stem}_{mode}.json">{html.escape(label(mode))}</a>' for mode in MODES)
        cases.append(
            f"<section><h2>tri_ke = {stiffness:.0e}</h2><p>tri_ka = {0.2 * stiffness:.0e}; bending stiffness = 200.</p>"
            f'<video controls preload="metadata" poster="{stem}_rho_sweep_frame180.png" src="{stem}_rho_sweep.mp4"></video>'
            f'<p><a href="{stem}_rho_sweep.mp4">Open or download this six-second video</a></p><details><summary>Per-frame measurements</summary><p>{links}</p></details></section>'
        )
    validation_links = " | ".join(
        f'<a href="{name}">{title}</a>'
        for name, title in (
            ("validation.json", "Trajectory validation"),
            ("native-control-validation.json", "Native control validation"),
            ("force-validation.json", "Force diagnostic validation"),
            ("grid_video_validation.json", "Simultaneous grid validation"),
        )
        if (output / name).exists()
    )
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bag trajectories: ALM rho-floor sweep</title><style>
:root{color-scheme:dark}body{margin:0;background:#101722;color:#edf2f7;font:17px/1.55 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:28px 24px 60px}h1{font-size:36px;line-height:1.15}h2{font-size:25px}p{max-width:1120px}a{color:#8edcdd}section{margin:30px 0;padding:22px;background:#182232;border:1px solid #344253;border-radius:12px}video{width:100%;aspect-ratio:16/9;background:#000;display:block}small,.muted{color:#b9c7d7}code{background:#233247;padding:2px 5px;border-radius:4px}nav{display:flex;flex-wrap:wrap;gap:18px}details{font-size:14px}
</style></head><body><main><h1>ALM rho-floor sweep: three material stiffnesses</h1>
<p>The main video shows all 18 trajectories at synchronized times: three material stiffness rows and six rho-floor setting columns. All panels share the same fixed camera and prescribed rim motion. The columns are ALM off, 9k, 1k, 0.1k, 0.01k, and inertia only (floor 0).</p>
<nav><a href="rho_floor_sweep_grid.mp4">Simultaneous 3 x 6 grid (6 s)</a><a href="rho_floor_sweep_all_materials.mp4">Sequential chapters (18 s)</a><a href="ke1000_rho_sweep.mp4">tri_ke 1e3</a><a href="ke100000_rho_sweep.mp4">tri_ke 1e5</a><a href="ke10000000_rho_sweep.mp4">tri_ke 1e7</a></nav>
<section><h2>Three stiffness rows x six rho-floor columns</h2><video style="aspect-ratio:16/7" controls preload="metadata" poster="rho_floor_sweep_grid_frame180.png" src="rho_floor_sweep_grid.mp4"></video><p class="muted">Rows: tri_ke = 1e3, 1e5, 1e7. Columns: off, rho floor 9k, 1k, 0.1k, 0.01k, and inertia only. All 18 panels advance together through the same six-second schedule. Open the full-resolution video or use fullscreen to inspect the labels.</p></section>
<details><summary>Secondary version: sequential chapters (18 seconds)</summary><video controls preload="metadata" poster="ke1000_rho_sweep_frame180.png" src="rho_floor_sweep_all_materials.mp4"></video><p class="muted">Chapters: 0–6 s at 1e3; 6–12 s at 1e5; 12–18 s at 1e7.</p></details>
<p><b>Settings:</b> 360 frames at 60 fps; 10 substeps per frame; 10 VBD iterations per substep; tri_ka = 0.2 tri_ke; bending stiffness = 200. The first second settles the bag; the next five seconds move its pinned rim. Every panel uses the same self-contact storage, preallocated at 16 times the baseline capacity to avoid storage overflow; material and contact parameters are unchanged.</p>
<p><b>Read the labels separately:</b> stretch and bend RMS describe physical deformation. The original material force residual measures particle force balance in newtons after the final particle iteration of the final substep, before velocity finalization. It includes original elastic forces, material damping, fresh native contacts and inertia. It excludes rigid-body residuals and DAT constraint reactions, so it is not a full coupled-system KKT residual. Deformation is not a convergence residual.</p>
<p>The last panel line reports the actual median <code>k_eff/k</code> separately for triangle stretch, triangle area and bending, in that order. The floor is <code>rho = max(rho_inertia, m_floor k)</code>; inertia can keep the actual ratio above the nominal floor value. ALM off retains the original material stiffness. Cloth clipping and contents outside the fixed view receive amber labels; freely falling contents do not stop the simulation. Failed trajectories remain present, with their last finite pose held and a red failure label.</p>
__CASES__
<p class="muted">The scene is the documented reconstruction of the May pinned-bag fixture, with three rigid contents and no ground or gripper. The original untracked fixture was unavailable. Each method starts from the same initial scene and retains its own history through an independently evolving trajectory. This compares matched settings; it is not a convergence benchmark at equal states. These runs override only the experiment's triangle and bending metric floors; they do not change the production solver default. This is a visual comparison, not a runtime benchmark.</p>
<p><a href="../render_rho_sweep.py">Renderer and viewer source</a> | <a href="video_validation.json">Video validation</a> | <a href="../results-floor-sweep/index.html">Earlier matched-state residual experiment</a></p><small>Local files only. No external scripts, fonts, telemetry or network requests.</small></main></body></html>"""
    page = page.replace("__CASES__", "\n".join(cases) + f"<p>{validation_links}</p>")
    (output / "index.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results-rho-video-sweep")
    parser.add_argument("--stiffness", type=float)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-modes", nargs="+", choices=MODES, default=["off", "floor9"])
    parser.add_argument("--preview-frames", nargs="+", type=int, default=[1, 180])
    parser.add_argument("--compose", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compose:
        compose(args.output)
    elif args.stiffness is None:
        parser.error("--stiffness is required for rendering")
    else:
        render(args)
