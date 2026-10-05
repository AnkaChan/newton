# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transpose the saved dense grid into a separate wide movie without rerunning physics."""

import argparse
import hashlib
import json

import imageio_ffmpeg
import numpy as np
from render_dense_floor_sweep import (
    BACKGROUND,
    CELL_HEIGHT,
    CELL_WIDTH,
    FLOORS,
    FPS,
    FRAMES,
    LEFT,
    MUTED,
    ROOT,
    ROW_HEIGHT,
    STIFFNESSES,
    TOP,
    fonts,
    writer,
)

OUTPUT = ROOT / "results-dense-floor-video-sweep"
STEM = "dense_floor_sweep_wide"
WIDTH, HEIGHT = LEFT + (len(FLOORS) + 1) * CELL_WIDTH, TOP + len(STIFFNESSES) * ROW_HEIGHT


def background(frame, text_fonts, *, floors=FLOORS):
    from PIL import Image, ImageDraw

    width = LEFT + (len(floors) + 1) * CELL_WIDTH
    grid = Image.new("RGB", (width, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(grid)
    draw.text((16, 9), "Material stiffness down | rho floors across", font=text_fonts["title"], fill="white")
    draw.text(
        (16, 47),
        "rho = max(rho_inertia, f*k) | bend_ke = 200 | 10 substeps x 10 iterations | same saved trajectories",
        font=text_fonts["legend"],
        fill=MUTED,
    )
    draw.text((width - 250, 15), f"t = {frame / FPS:.2f} s", font=text_fonts["column"], fill="white")
    for column, floor in enumerate((None, *floors)):
        label = "Vanilla VBD (ALM off)" if floor is None else f"rho floor = {floor:g} k"
        draw.text(
            (LEFT + (column + 0.5) * CELL_WIDTH, 78),
            label,
            anchor="mt",
            font=text_fonts["row"],
            fill=(241, 169, 101) if floor is None else (93, 213, 212),
        )
    for row, stiffness in enumerate(STIFFNESSES):
        y = TOP + row * ROW_HEIGHT
        draw.rectangle((0, y, width - 1, y + ROW_HEIGHT - 1), outline=(65, 80, 100), width=2)
        draw.rectangle((0, y, LEFT - 1, y + ROW_HEIGHT - 1), fill=(26, 39, 58))
        for offset, label in (
            (75, "tri_ke"),
            (115, f"{stiffness:.0e}"),
            (192, "tri_ka"),
            (232, f"{0.2 * stiffness:.0e}"),
        ):
            draw.text((14, y + offset), label, font=text_fonts["row"], fill="white")
    return grid


def build_viewer(*, floors=FLOORS, stem=STEM, filename="index-wide.html", validation="wide_video_validation.json"):
    width = LEFT + (len(floors) + 1) * CELL_WIDTH
    column_count = len(floors) + 1
    options = '<option value="0">Vanilla VBD</option>' + "".join(
        f'<option value="{column}">{floor:g} k</option>' for column, floor in enumerate(floors, 1)
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dense ALM floor sweep: wide layout</title><style>
:root{{color-scheme:dark}}body{{margin:0;background:#101722;color:#edf2f7;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1800px;margin:auto;padding:24px}}a{{color:#8edcdd}}button,select,input{{font:inherit}}
.toolbar{{position:sticky;top:0;z-index:2;background:#182232;padding:12px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}}
.viewport{{overflow:auto;border:1px solid #344253}}video{{display:block;width:100%;height:auto;max-width:{width}px}}
.viewport.native video{{width:{width}px;max-width:none}}
</style></head><body><main><h1>Three stiffness rows, {column_count} columns</h1>
<p>{column_count * len(STIFFNESSES)} selected trajectories from the same experiment. Rows: tri_ke = 1e3, 1e5, 1e7, with tri_ka = 0.2 tri_ke.
Columns: <b>vanilla VBD (ALM off)</b>, then ALM floor multipliers {", ".join(f"{floor:g}" for floor in floors)}.
The bags and labels remain upright. No physics was rerun; the original video is preserved.</p>
<div class="toolbar"><button id="play">Play</button><button id="restart">Restart</button>
<label>Time <input id="seek" type="range" min="0" max="6" value="0" step="0.0166667"></label><output id="time">0.00 s</output>
<button id="size">Enlarge panels</button>
<label>Scroll to floor <select id="column">{options}</select></label>
<span>Stiffness: top 1e3 | middle 1e5 | bottom 1e7</span><a href="{stem}.mp4">Open/download MP4</a></div>
<div class="viewport" id="viewport"><video id="video" muted playsinline preload="metadata" poster="{stem}_frame180.png" src="{stem}.mp4"></video></div>
<p>All columns fit the viewer by default. Enlarge the panels to inspect the labels and scroll horizontally; all columns stay synchronized. The floor is rho = max(rho_inertia, f*k).
The video retains the deformation, original force residual, and scalar effective-stiffness labels from the original.</p>
<p><a href="index.html">Original tall video, measurements, and experiment limitations</a> |
<a href="index-wide.html">Original fourteen-column video</a> |
<a href="{validation}">Video validation</a> | <a href="../transpose_dense_floor_video.py">Recomposition source</a></p>
</main><script>
const video=document.querySelector('#video'),play=document.querySelector('#play'),seek=document.querySelector('#seek');
play.onclick=()=>video.paused?video.play():video.pause();video.onplay=()=>play.textContent='Pause';video.onpause=()=>play.textContent='Play';
document.querySelector('#restart').onclick=()=>{{video.currentTime=0;video.play();}};
video.ontimeupdate=()=>{{seek.value=video.currentTime;document.querySelector('#time').textContent=video.currentTime.toFixed(2)+' s';}};
seek.oninput=()=>video.currentTime=Number(seek.value);
const viewport=document.querySelector('#viewport'),size=document.querySelector('#size');
size.onclick=()=>{{viewport.classList.toggle('native');size.textContent=viewport.classList.contains('native')?'Fit all columns':'Enlarge panels';}};
document.querySelector('#column').onchange=e=>{{viewport.classList.add('native');size.textContent='Fit all columns';viewport.scrollTo({{left:{LEFT}+Number(e.target.value)*{CELL_WIDTH},behavior:'smooth'}});}};
</script></body></html>
"""
    (OUTPUT / filename).write_text(page)


def main(*, six_columns=False):
    from PIL import Image

    floors = (90, 9, 1, 0.6, 0.1) if six_columns else FLOORS
    column_count = len(floors) + 1
    source_rows = [0, *(FLOORS.index(floor) + 1 for floor in floors)]
    width = LEFT + column_count * CELL_WIDTH
    stem = "dense_floor_sweep_six_columns" if six_columns else STEM
    filename = "index-six-columns.html" if six_columns else "index-wide.html"
    validation = "six_columns_video_validation.json" if six_columns else "wide_video_validation.json"
    original = OUTPUT / "dense_floor_sweep_grid.mp4"
    preserved = {
        name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
        for name in (original.name, "index.html", "dense_floor_sweep_wide.mp4", "index-wide.html")
        if (OUTPUT / name).exists()
    }
    for stiffness in STIFFNESSES:
        baseline = json.loads((OUTPUT / f"ke{stiffness}_off.json").read_text())
        assert baseline["mode"] == "off" and baseline["alm"] is False
    target = OUTPUT / f"{stem}.mp4"
    if target.exists():
        raise FileExistsError(target)
    reader = imageio_ffmpeg.read_frames(str(original), input_params=["-threads", "4"])
    text_fonts = fonts()
    try:
        metadata = next(reader)
        assert metadata["size"] == (1440, 4956) and metadata["fps"] == FPS
        with writer(target) as output:
            for frame in range(1, FRAMES + 1):
                source = Image.fromarray(np.frombuffer(next(reader), np.uint8).reshape(4956, 1440, 3))
                grid = background(frame, text_fonts, floors=floors)
                for row in range(len(STIFFNESSES)):
                    for column, source_row in enumerate(source_rows):
                        x, y = LEFT + row * CELL_WIDTH, TOP + source_row * ROW_HEIGHT + 3
                        panel = source.crop((x, y, x + CELL_WIDTH, y + CELL_HEIGHT))
                        grid.paste(panel, (LEFT + column * CELL_WIDTH, TOP + row * ROW_HEIGHT + 3))
                output.append_data(np.asarray(grid))
                if frame in (1, 180, 360):
                    grid.save(OUTPUT / f"{stem}_frame{frame:03d}.png")
                if frame % 60 == 0:
                    print(f"Wide grid frame {frame}/{FRAMES}", flush=True)
    finally:
        reader.close()
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(target))
    assert count == FRAMES and abs(duration - FRAMES / FPS) < 0.01
    for name, digest in preserved.items():
        assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == digest
    samples = []
    for time in (0, 3, 5.9):
        decoded = imageio_ffmpeg.read_frames(
            str(target), input_params=["-ss", str(time)], output_params=["-frames:v", "1"]
        )
        try:
            info = next(decoded)
            assert info["size"] == (width, HEIGHT) and info["fps"] == FPS
            pixels = np.frombuffer(next(decoded), np.uint8).reshape(HEIGHT, width, 3)
            values = []
            for row in range(3):
                for column in range(column_count):
                    x, y = LEFT + column * CELL_WIDTH, TOP + row * ROW_HEIGHT + 35
                    values.append(float(pixels[y : y + 220, x : x + CELL_WIDTH].std()))
            assert min(values) > 2
            samples.append({"time_s": time, "scene_pixel_std": values})
        finally:
            decoded.close()
    result = {
        "file": target.name,
        "frames": count,
        "duration_s": duration,
        "fps": FPS,
        "width": width,
        "height": HEIGHT,
        "rows_tri_ke": list(STIFFNESSES),
        "columns_floor": [None, *floors],
        "source_cell_rows": source_rows,
        "vanilla_vbd_verified": True,
        "source_video": original.name,
        "preserved_original_sha256": preserved,
        "bytes": target.stat().st_size,
        "frame_mapping": "Decoded source frame n is cropped into cells and transposed without scaling, time resampling, or scene rotation.",
        "samples": samples,
    }
    (OUTPUT / validation).write_text(json.dumps(result, indent=2) + "\n")
    build_viewer(floors=floors, stem=stem, filename=filename, validation=validation)
    print(f"Saved {target} ({width}x{HEIGHT}, {count} frames)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--six-columns", action="store_true", help="Select vanilla VBD and five representative floors")
    args = parser.parse_args()
    main(six_columns=args.six_columns)
