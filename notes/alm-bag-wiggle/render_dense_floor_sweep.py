# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render synchronized rho rows with three material columns, or transpose the saved inertia movie."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent
STIFFNESSES = (1000, 100000, 10000000)
FLOORS = (90, 60, 30, 20, 10, 9, 6, 3, 2, 1, 0.6, 0.3, 0.1)
MODES = ("off", *(f"floor{floor:g}".replace(".", "p") for floor in FLOORS))
INERTIA_MODES = ("off", "inertia1", "inertia10", "inertia100", "inertia1000")
WIDTH, FPS, FRAMES = 1440, 60, 360
LEFT, TOP, CELL_WIDTH, ROW_HEIGHT, CELL_HEIGHT = 156, 112, 428, 346, 340
BACKGROUND, MUTED = (18, 25, 36), (187, 199, 215)
CAMERA = {"camera_pos": (0.50, -0.84, 0.46), "camera_target": (0.0, 0.0, 0.14), "camera_fov": 30.0}


def fonts():
    from PIL import ImageFont

    return {
        name: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        for name, size in (
            ("title", 28),
            ("legend", 18),
            ("column", 25),
            ("row", 22),
            ("value", 35),
            ("metric", 13),
            ("small", 12),
            ("failure", 18),
        )
    }


def labels(inertia):
    return (
        ["ALM off", *(f"rho = {scale} x inertia" for scale in (1, 10, 100, 1000))]
        if inertia
        else ["ALM off", *(f"rho floor = {floor:g} k" for floor in FLOORS)]
    )


def background(frame, inertia, text_fonts):
    from PIL import Image, ImageDraw

    row_labels = labels(inertia)
    grid = Image.new("RGB", (WIDTH, TOP + len(row_labels) * ROW_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(grid)
    title = "Inertia scale" if inertia else "Material-stiffness floor"
    draw.text((16, 9), f"{title}: rho rows x material columns", font=text_fonts["title"], fill="white")
    draw.text((1190, 15), f"t = {frame / FPS:.2f} s", font=text_fonts["column"], fill="white")
    formula = "rho = c * rho_inertia; no material floor" if inertia else "rho = max(rho_inertia, f*k)"
    draw.text(
        (16, 46),
        f"{formula} | bend_ke = 200 | 10 substeps x 10 iterations | fixed 30-degree camera",
        font=text_fonts["legend"],
        fill=MUTED,
    )
    for column, stiffness in enumerate(STIFFNESSES):
        x = LEFT + (column + 0.5) * CELL_WIDTH
        draw.text(
            (x, 76),
            f"tri_ke = {stiffness:.0e} | tri_ka = {0.2 * stiffness:.0e}",
            anchor="mt",
            font=text_fonts["legend"],
            fill="white",
        )
    for row, label in enumerate(row_labels):
        y = TOP + row * ROW_HEIGHT
        draw.rectangle((0, y, WIDTH - 1, y + ROW_HEIGHT - 1), outline=(65, 80, 100), width=2)
        draw.rectangle((0, y, LEFT - 1, y + ROW_HEIGHT - 1), fill=(26, 39, 58))
        if row == 0:
            draw.text((14, y + 150), label, font=text_fonts["row"], fill=(241, 169, 101))
        else:
            value = (1, 10, 100, 1000)[row - 1] if inertia else FLOORS[row - 1]
            draw.text((14, y + 100), "rho =" if inertia else "rho floor", font=text_fonts["row"], fill=MUTED)
            draw.text(
                (14, y + 142), f"{value:g}" + ("" if inertia else " k"), font=text_fonts["value"], fill=(93, 213, 212)
            )
            if inertia:
                draw.text((14, y + 196), "x inertia", font=text_fonts["row"], fill=MUTED)
            else:
                draw.text((14, y + 196), "scalar k_eff/k", font=text_fonts["small"], fill=MUTED)
                draw.text((14, y + 215), f"min ~ {value / (1.0 + value):.3g}", font=text_fonts["legend"], fill=MUTED)
    return grid


def validate(output, inertia):
    stem = "inertia_scale_sweep_transposed" if inertia else "dense_floor_sweep_grid"
    path = output / f"{stem}.mp4"
    height = TOP + len(labels(inertia)) * ROW_HEIGHT
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count == FRAMES and abs(seconds - FRAMES / FPS) < 0.04, (count, seconds)
    samples = []
    for time in (0.0, 3.0, 5.9):
        reader = imageio_ffmpeg.read_frames(
            str(path), input_params=["-ss", str(time)], output_params=["-frames:v", "1"]
        )
        try:
            metadata = next(reader)
            assert metadata["size"] == (WIDTH, height) and metadata["fps"] == FPS, metadata
            pixels = np.frombuffer(next(reader), dtype=np.uint8).reshape(height, WIDTH, 3)
        finally:
            reader.close()
        deviations = []
        for row in range(len(labels(inertia))):
            for column in range(3):
                x, y = LEFT + column * CELL_WIDTH, TOP + row * ROW_HEIGHT + 35
                scene = pixels[y : y + 230, x : x + CELL_WIDTH]
                deviations.append(float(scene.std()))
                assert scene.std() > 2.0 and scene.mean() > 2.0, (time, row, column)
        samples.append({"time_s": time, "scene_pixel_std": deviations})
    result = {
        "file": path.name,
        "frames": count,
        "duration_s": seconds,
        "fps": FPS,
        "width": WIDTH,
        "height": height,
        "codec": "H.264",
        "crf": 18,
        "bytes": path.stat().st_size,
        "rows": labels(inertia),
        "columns_tri_ke": STIFFNESSES,
        "camera": CAMERA,
        "samples": samples,
        "frame_mapping": "Every output frame uses the matching saved frame from every independent trajectory, without interpolation or rotation.",
    }
    (output / ("transpose_video_validation.json" if inertia else "video_validation.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


def build_viewer(output, inertia):
    stem = "inertia_scale_sweep_transposed" if inertia else "dense_floor_sweep_grid"
    row_labels = labels(inertia)
    modes = INERTIA_MODES if inertia else MODES
    height = TOP + len(row_labels) * ROW_HEIGHT
    options = "".join(f'<option value="{row}">{html.escape(label)}</option>' for row, label in enumerate(row_labels))
    measurements = "".join(
        f"<p><b>{html.escape(label)}:</b> "
        + " | ".join(f'<a href="ke{stiffness}_{mode}.json">tri_ke {stiffness:.0e}</a>' for stiffness in STIFFNESSES)
        + "</p>"
        for mode, label in zip(modes, row_labels, strict=True)
    )
    validation_name = "transpose_video_validation.json" if inertia else "video_validation.json"
    force_validation = "force-validation.json" if inertia else "../results-inertia-video-sweep/force-validation.json"
    validations = " | ".join(
        f'<a href="{name}">{title}</a>'
        for name, title in (
            ("validation.json", "Trajectory validation"),
            (force_validation, "Force diagnostic validation (same unchanged diagnostic)"),
            (validation_name, "Video validation"),
        )
        if (output / name).exists()
    )
    formula = (
        "rho = c * rho_inertia, with no material-stiffness floor"
        if inertia
        else "rho = max(rho_inertia, f*k), where each listed floor is a multiplier f of the elastic row's material stiffness k"
    )
    floor_note = (
        ""
        if inertia
        else "<p>For each active scalar row, floor f implies k_eff/k &gt;= f/(1+f). This bounds the scalar curvature coefficient; the full particle Hessian also includes geometric curvature and other terms. The actual median ratios remain visible in every cell.</p><p>Each setting is a single run with a fresh ALM-off baseline from the same frozen sources. GPU and contact ordering can produce rerun differences, so tiny differences between nearby floors do not establish statistical improvements.</p>"
    )
    previous = (
        '<a href="index.html">Original inertia grid and measurements</a>'
        if inertia
        else '<a href="../results-inertia-video-sweep/index-transposed.html">Inertia-scale comparison</a>'
    )
    material_legend = "" if inertia else "<span>tri_ke: left 1e3 | middle 1e5 | right 1e7</span>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rho rows and material columns</title><style>
:root{{color-scheme:dark}}body{{margin:0;background:#101722;color:#edf2f7;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1488px;margin:auto;padding:24px}}h1{{font-size:30px}}a{{color:#8edcdd}}p{{max-width:1250px}}.toolbar{{position:sticky;top:0;z-index:2;background:#182232;padding:12px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}}button,select,input{{font:inherit}}.viewport{{overflow:auto;border:1px solid #344253}}video{{display:block;width:{WIDTH}px;height:{height}px;max-width:none;background:#000}}small,.muted{{color:#b9c7d7}}details{{font-size:14px}}
</style></head><body><main><h1>One synchronized video: rho settings down, materials across</h1><p>Each row compares three upright material cases side by side: <b>tri_ke = 1e3, 1e5, 1e7</b>, with tri_ka = 0.2 tri_ke. Rows use {html.escape(formula)}. Bending stiffness is 200 in every case.</p>
<div class="toolbar"><button id="play">Play</button><button id="restart">Restart</button><label>Time <input id="seek" type="range" min="0" max="6" value="0" step="0.0166667"></label><output id="time">0.00 s</output><label>Scroll to row <select id="row">{options}</select></label>{material_legend}<a href="{stem}.mp4">Open/download MP4</a></div>
<div class="viewport"><video id="video" muted playsinline preload="metadata" poster="{stem}_frame180.png" src="{stem}.mp4"></video></div>
<p>The complete tall grid remains in one video, with every row synchronized. The selector scrolls to a row without hiding or replacing any row. The video is displayed at its native pixel size; scroll vertically to inspect lower settings and horizontally on narrow windows. The original movies remain available.</p>
<p><b>Settings:</b> 360 frames at 60 fps, 10 substeps per frame, 10 VBD iterations per substep. The first second settles the bag; the next five seconds move its pinned rim. All cases share the same camera, initial scene, prescribed motion and 16-times-baseline self-contact storage, and evolve independently with retained multiplier history.</p>
<p><b>Metrics:</b> stretch and bend describe physical deformation. Original force residual RMS is particle force balance after the final particle iteration of the final substep, before velocity finalization; it includes original elastic forces, damping, fresh native contacts and inertia, and excludes rigid-body residuals and DAT reactions. It is not a full-system KKT residual or an equal-state convergence benchmark. Median k_eff/k is shown for stretch, area and bending, in that order. Amber labels identify clipping or escaped contents; failed cases keep their last finite pose with an explicit red marker.</p>
{floor_note}
<details><summary>Measurements and validation</summary>{measurements}<p>{validations}</p></details><p class="muted">The documented May bag fixture was reconstructed with three rigid contents and no ground or gripper because its original untracked file was unavailable. No trajectory interpolation or scene rotation is applied.</p><p>{previous} | <a href="../render_dense_floor_sweep.py">Renderer source</a></p><small>Local files only; no external resources.</small></main><script>
const video=document.querySelector('#video'), play=document.querySelector('#play'), seek=document.querySelector('#seek'), time=document.querySelector('#time');
play.onclick=()=>video.paused?video.play():video.pause(); document.querySelector('#restart').onclick=()=>{{video.currentTime=0;video.play();}};
video.onplay=()=>play.textContent='Pause'; video.onpause=()=>play.textContent='Play'; video.ontimeupdate=()=>{{seek.value=video.currentTime;time.textContent=video.currentTime.toFixed(2)+' s';}};seek.oninput=()=>video.currentTime=Number(seek.value);
document.querySelector('#row').onchange=e=>window.scrollTo({{top:video.getBoundingClientRect().top+window.scrollY+{TOP}+Number(e.target.value)*{ROW_HEIGHT}-document.querySelector('.toolbar').getBoundingClientRect().height-8,behavior:'smooth'}});
</script></body></html>"""
    (output / ("index-transposed.html" if inertia else "index.html")).write_text(page)


def writer(path):
    return imageio.get_writer(
        path,
        fps=FPS,
        quality=None,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "18", "-preset", "medium", "-threads", "4", "-movflags", "+faststart"],
    )


def transpose(args):
    from PIL import Image

    output = args.output or ROOT / "results-inertia-video-sweep"
    stem = "inertia_scale_sweep_transposed"
    params = ["-threads", "2"]
    if args.preview:
        params += ["-ss", str((args.preview_frame - 1) / FPS)]
    reader = imageio_ffmpeg.read_frames(str(output / "inertia_scale_sweep_grid.mp4"), input_params=params)
    text_fonts, video_writer = fonts(), None
    try:
        metadata = next(reader)
        assert metadata["size"] == (3240, 1680) and metadata["fps"] == FPS, metadata
        if not args.preview:
            video_writer = writer(output / f"{stem}.mp4")
        for frame in [args.preview_frame] if args.preview else range(1, FRAMES + 1):
            pixels = np.frombuffer(next(reader), dtype=np.uint8).reshape(1680, 3240, 3)
            source, grid = Image.fromarray(pixels), background(frame, True, text_fonts)
            for row in range(5):
                for column in range(3):
                    x, y = 240 + row * 600, 144 + column * 512 + 18
                    panel = source.crop((x, y, x + 600, y + 476)).resize(
                        (CELL_WIDTH, CELL_HEIGHT), Image.Resampling.LANCZOS
                    )
                    grid.paste(panel, (LEFT + column * CELL_WIDTH, TOP + row * ROW_HEIGHT + 3))
            if video_writer is not None:
                video_writer.append_data(np.asarray(grid))
            if args.preview or frame in (1, 180, 360):
                grid.save(output / f"{stem}_frame{frame:03d}.png")
            if args.preview or frame % 60 == 0:
                print(f"transpose frame {frame}/{FRAMES}", flush=True)
    finally:
        reader.close()
        if video_writer is not None:
            video_writer.close()
    if not args.preview:
        validate(output, True)
        build_viewer(output, True)


def dense_panel(scene, result, data, frame, text_fonts):
    from PIL import Image, ImageDraw
    from render_rho_sweep import (  # noqa: PLC0415 -- Only dense rendering imports Newton helpers.
        formatted,
        visible_fraction,
    )

    panel = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), BACKGROUND)
    panel.paste(scene.resize((CELL_WIDTH, 257), Image.Resampling.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(panel)
    row = result["rows"][frame]
    assert math.isclose(row["time_s"], frame / FPS, abs_tol=1e-6)
    stretch = row.get("stretch_score")
    values = [
        row.get(family, {}).get("effective_over_k", {}).get("median") for family in ("tri_stretch", "tri_area", "bend")
    ]
    metric_lines = (
        f"Stretch RMS: {formatted(None if stretch is None else stretch * 100.0, '%')} | bend: {formatted(row.get('bend_score'))} rad",
        f"Original force residual RMS: {formatted(row.get('original_rms_N'))} N",
        "Median k_eff/k: stretch / area / bending",
        "1 / 1 / 1 (original material)"
        if result["mode"] == "off"
        else " / ".join(formatted(value) for value in values),
    )
    for index, line in enumerate(metric_lines):
        draw.text((8, 263 + index * 18), line, font=text_fonts["metric"], fill="white" if index < 2 else MUTED)
    if not bool(data["valid"][frame]) or row.get("status") == "held_after_failure":
        draw.rectangle((1, 1, CELL_WIDTH - 2, CELL_HEIGHT - 2), outline=(255, 95, 103), width=3)
        draw.rectangle((8, 90, CELL_WIDTH - 8, 176), fill=(99, 25, 35))
        draw.text(
            (16, 99),
            f"FAILED AT {result.get('failure_frame', frame) / FPS:.2f} s",
            font=text_fonts["failure"],
            fill="white",
        )
        draw.text((16, 128), "Last finite pose held; no further simulation", font=text_fonts["metric"], fill="white")
        reason = str(result.get("failure_reason") or "See case JSON for details")
        while draw.textlength(reason, font=text_fonts["small"]) > CELL_WIDTH - 32:
            reason = reason[:-4] + "..."
        draw.text((16, 151), reason, font=text_fonts["small"], fill=(255, 206, 210))
    else:
        warnings = []
        fraction = visible_fraction(data["particle_q"][frame])
        if fraction < 1.0:
            warnings.append(f"Cloth outside fixed view: {100.0 * (1.0 - fraction):.1f}% of vertices")
        bodies = data["body_q"][frame, :, :3]
        outside = round(len(bodies) * (1.0 - visible_fraction(bodies)))
        if outside:
            warnings.append(f"Contents outside view: {outside}/{len(bodies)} (centers)")
        for index, warning in enumerate(warnings):
            y = 8 + index * 21
            draw.rectangle((6, y, CELL_WIDTH - 6, y + 21), fill=(107, 69, 9))
            draw.text((12, y + 2), warning, font=text_fonts["small"], fill="white")
    return panel


def render_dense(args):
    import warp as wp  # noqa: PLC0415 -- Keep the transpose workflow CPU-only.
    from PIL import Image
    from render_rho_sweep import Bag, Capture, load_case  # noqa: PLC0415 -- Dense rendering requires Newton and GL.

    output = args.output or ROOT / "results-dense-floor-video-sweep"
    wp.init()
    loaded = {
        (stiffness, mode): load_case(output, stiffness, mode, FRAMES) for mode in MODES for stiffness in STIFFNESSES
    }
    sim = Bag(STIFFNESSES[0], False)
    reference = loaded[STIFFNESSES[0], MODES[0]][1]
    for _result, data in loaded.values():
        np.testing.assert_array_equal(data["particle_q"][0], reference["particle_q"][0])
        np.testing.assert_array_equal(data["body_q"][0], reference["body_q"][0])
    pins = wp.empty(len(sim.pins), dtype=wp.vec3, device=sim.model.device)
    colors = wp.full(len(sim.pins), wp.vec3(1.0, 0.65, 0.08), dtype=wp.vec3, device=sim.model.device)
    text_fonts, video_writer = fonts(), None
    with Capture(out_dir=str(output), width=480, height=288, **CAMERA) as cap:
        viewer = cap._get_viewer(sim.model)
        viewer.show_ui = False
        viewer.renderer.draw_wireframe = True
        cap._apply_camera(viewer)
        if not args.preview:
            video_writer = writer(output / "dense_floor_sweep_grid.mp4")
        try:
            for frame in [args.preview_frame] if args.preview else range(1, FRAMES + 1):
                grid = background(frame, False, text_fonts)
                for row, mode in enumerate(MODES):
                    for column, stiffness in enumerate(STIFFNESSES):
                        result, data = loaded[stiffness, mode]
                        sim.state_0.particle_q.assign(data["particle_q"][frame])
                        sim.state_0.body_q.assign(data["body_q"][frame])
                        pins.assign(data["particle_q"][frame, sim.info["top_global_indices"]])
                        viewer.begin_frame(frame / FPS)
                        viewer.log_state(sim.state_0)
                        viewer.log_points("pinned_rim", pins, radii=0.0023, colors=colors)
                        viewer.end_frame()
                        panel = dense_panel(
                            Image.fromarray(viewer.get_frame().numpy()), result, data, frame, text_fonts
                        )
                        grid.paste(panel, (LEFT + column * CELL_WIDTH, TOP + row * ROW_HEIGHT + 3))
                if video_writer is not None:
                    video_writer.append_data(np.asarray(grid))
                if args.preview or frame in (1, 180, 360):
                    grid.save(output / f"dense_floor_sweep_grid_frame{frame:03d}.png")
                if args.preview or frame % 60 == 0:
                    print(f"dense grid frame {frame}/{FRAMES}", flush=True)
        finally:
            if video_writer is not None:
                video_writer.close()
    if not args.preview:
        validate(output, False)
        build_viewer(output, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transpose-inertia", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-frame", type=int, default=180)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    if args.html_only:
        build_viewer(
            args.output
            or ROOT / ("results-inertia-video-sweep" if args.transpose_inertia else "results-dense-floor-video-sweep"),
            args.transpose_inertia,
        )
    elif args.transpose_inertia:
        transpose(args)
    else:
        render_dense(args)
