import glob
import json
import os.path
import re

# --- video_maker.py ---
from pathlib import Path

import cv2
import numpy as np
from wedge_sims import *
import shutil

def _numeric_key(p: Path):
    """Sort by frame number if present; fallback to name."""
    m = re.search(r"(\d+)(?=\.[a-zA-Z]+$)", p.name)
    return (int(m.group(1)) if m else float("inf"), p.name)


def _find_image_dirs(root: Path):
    """Yield subdirs (including root) that contain PNGs."""
    yielded = set()
    # check root
    if list(root.glob("*.png")):
        yielded.add(root.resolve())
        yield root
    # check subfolders
    for d in root.rglob("*"):
        if d.is_dir() and d.resolve() not in yielded:
            if list(d.glob("*.png")):
                yielded.add(d.resolve())
                yield d


def _draw_banner(img, lines, margin=16, pad=12, alpha=0.55):
    """Draw a semi-transparent black banner with white text.
    - First line uses 2x font scale of others
    - Extra spacing between lines (like a blank line)
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    base_scale = max(1.0, min(w, h) / 800.0)  # auto scale with resolution
    scales = [(base_scale * 2.0 if i == 0 else base_scale) for i in range(len(lines))]
    thicknesses = [max(1, int(2 * s)) for s in scales]

    # Measure each line with its own scale/thickness
    sizes = [cv2.getTextSize(line, font, scales[i], thicknesses[i])[0] for i, line in enumerate(lines)]

    # Extra gap between lines to mimic a blank line
    line_gap = int(pad * 1.6)

    text_h = sum(sz[1] for sz in sizes) + pad * 2 + (len(lines) - 1) * line_gap
    text_w = max(sz[0] for sz in sizes) + pad * 2

    x0, y0 = margin, margin
    x1, y1 = min(w - margin, x0 + text_w), min(h - margin, y0 + text_h)

    # Background rectangle (semi-transparent)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw lines with per-line font scale
    y = y0 + pad
    for i, line in enumerate(lines):
        sz = sizes[i]
        # baseline is y + current line height
        y += sz[1]
        cv2.putText(img, line, (x0 + pad, y), font, scales[i], (255, 255, 255), thicknesses[i], cv2.LINE_AA)
        # add extra gap after each line except the last
        if i < len(lines) - 1:
            y += line_gap


def _safe_read(cap, last_frame):
    ok, frame = cap.read()
    if ok:
        return frame, frame
    return last_frame, last_frame  # hold last frame if ended


def combine_videos_grid_2x2(
    paths,
    out_path,
    fps=None,
    size=None,  # (w, h) for each cell; None -> use first valid video
    labels=None,
    font_scale=0.6,
    write_final_hold_frame=True,  # 写一帧“最终保持画面”，然后停止
):
    # 规范化路径并保持长度≤4（不足补None）
    paths = list(paths)
    while len(paths) < 4:
        paths.append(None)

    # 打开视频
    caps, metas = [], []
    for p in paths:
        if p and Path(p).exists():
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                cap = None
        else:
            cap = None

        caps.append(cap)
        if cap:
            w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f0 = cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            w0, h0, f0 = 640, 360, 30.0  # 占位元信息
        metas.append((w0, h0, f0))

    # 选择 FPS 和单格尺寸
    if fps is None:
        fps = next((m[2] for m in metas if m[2] > 0), 30.0)
    if size is None:
        size = next(((m[0] // 2, m[1] // 2) for m in metas if m[0] > 0 and m[1] > 0), (640, 360))
    cell_w, cell_h = size

    out_w, out_h = cell_w * 2, cell_h * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (out_w, out_h))

    def black_cell():
        return np.zeros((cell_h, cell_w, 3), np.uint8)

    # 状态：记录每路的“最后一帧”和“是否已结束”
    last_frames = [None, None, None, None]
    ended = [False, False, False, False]

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    first_frame_written = False

    while True:
        any_new_frame = False
        cells = []

        for i, cap in enumerate(caps):
            frame = None
            if cap and not ended[i]:
                ok, f = cap.read()
                if ok:
                    frame = f
                    last_frames[i] = f
                    any_new_frame = True  # 本轮至少有一路获得新帧
                else:
                    # 到结尾
                    ended[i] = True

            # 没新帧/已结束 → 用最后一帧占位，否则黑帧
            if frame is None:
                frame = last_frames[i] if last_frames[i] is not None else black_cell()
            else:
                frame = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

            # 贴角标
            if labels and i < len(labels) and labels[i]:
                cell = frame
                txt = str(labels[i])
                (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                pad = 6
                overlay = cell.copy()
                cv2.rectangle(overlay, (6, 6), (6 + tw + 2 * pad, 6 + th + 2 * pad), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, cell, 0.5, 0, cell)
                cv2.putText(
                    cell, txt, (6 + pad, 6 + th + pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
                )
                frame = cell

            cells.append(frame if frame.shape[:2] == (cell_h, cell_w) else cv2.resize(frame, (cell_w, cell_h)))

        # 组 2x2
        top = np.hstack((cells[0], cells[1]))
        bot = np.hstack((cells[2], cells[3]))
        grid = np.vstack((top, bot))

        if any_new_frame:
            writer.write(grid)
            first_frame_written = True
        else:
            # 所有输入都没新帧了：可选写一帧“最后画面”，然后退出
            if write_final_hold_frame and not first_frame_written:
                # 确保至少有1帧（全黑也行）
                writer.write(grid)
            break

    writer.release()
    for cap in caps:
        if cap:
            cap.release()

    print(f"✅ 2x2 grid video created: {out_path}")


def process_batch(batch):
    """Put your processing code here for a batch of up to 4 files."""
    print("Processing batch:", batch)
    # Example: call your 2x2 combiner here
    # combine_videos_grid_2x2(paths=batch, out_path=f"grid_{i}.mp4")


def run_in_batches_of_4(files):
    files = [str(Path(f)) for f in files if f]  # normalize paths
    for i in range(0, len(files), 4):
        batch = files[i : i + 4]  # up to 4 items
        # pad to exactly 4 if needed (use None for empty slots)
        while len(batch) < 4:
            batch.append(None)
        process_batch(batch)


def make_videos_for_sweep(
    rendering_folder: str | Path, cfg: dict, fps: int | None = None, out_dir: str | Path | None = None, banner_lines=None,
):
    """
    Build one video per image-containing folder under `rendering_folder`.
    Names videos like: <foldername>_sweep_<N>.mp4 (in the same folder or out_dir if provided).
    Overlays 'Sweep <sweep_number> — <text>' and a second line '<foldername>' on each frame.
    """
    rendering_folder = Path(rendering_folder)
    out_dir = Path(out_dir) if out_dir else None

    sweep_num = cfg.get("wedge_number", "NA")
    text = cfg.get("text", "")
    fps = fps or cfg.get("frame_rate", 60)

    for img_dir in _find_image_dirs(rendering_folder):
        images = sorted(img_dir.glob("*.png"), key=_numeric_key)
        if not images:
            continue

        # Read first frame to get size
        first = cv2.imread(str(images[0]))
        if first is None:
            print(f"⚠️ Skipping {img_dir} (cannot read first image)")
            continue
        h, w = first.shape[:2]

        # Writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely compatible
        video_name = "rendering.mp4"
        video_path = (out_dir / video_name) if out_dir else (img_dir / video_name)
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            print(f"❌ Failed to open writer for {video_path}")
            continue

        if banner_lines is None:
            banner_lines = [f"{sweep_num}", text]

        for idx, im_path in enumerate(images):
            frame = cv2.imread(str(im_path))
            if frame is None:
                print(f"⚠️ Cannot read {im_path}, skipping frame")
                continue
            _draw_banner(frame, banner_lines)
            writer.write(frame)

        writer.release()
        print(f"🎬 Wrote video: {video_path}")

    return video_path


def process_batch(batch):
    """Put your processing code here for a batch of up to 4 files."""
    print("Processing batch:", batch)
    # Example: call your 2x2 combiner here


def run_in_batches_of_4(files, base_dir):
    files = [str(Path(f)) for f in files if f]  # normalize paths
    for i in range(0, len(files), 4):
        batch = files[i : i + 4]  # up to 4 items
        # pad to exactly 4 if needed (use None for empty slots)
        while len(batch) < 4:
            batch.append(None)
        combine_videos_grid_2x2(paths=batch, out_path=os.path.join(base_dir, f"grid_{i}.mp4"))


def render_usd(sweep_folder, render_temp_dir, rendering_folder, src_filename_base="", dst_filename=None, sim_project_name="full_20251021_to_sim_tdSimClothB_01_physics_sim_v.usd"):
    if dst_filename is None:
        dst_filename = "20251021_to_sim_tdSimClothB_01_physics_sim_v.usd"

    src_file = glob.glob(os.path.join(sweep_folder, "*.usd"))[0]
    if Path(src_file).exists():
        dst_file = os.path.join(render_temp_dir, dst_filename)  # Keep folder name in output
        # If you want the SAME name for all (no prefix), use: dst_file = render_dir / dst_filename

        shutil.copy2(src_file, dst_file)
        print(f"Copied: {src_file} -> {dst_file}")
    else:
        print(f"❌ File not found in {sweep_folder}")

    cfg = json.load(open(os.path.join(sweep_folder, "config.json")))

    os.makedirs(rendering_folder, exist_ok=True)
    cmd = [
        r"C:\isaac-sim\python.bat",
        r"D:\Code\Graphics\newton\newton\_src\utils\render_py.py",
        os.path.join(render_temp_dir, sim_project_name),
        "-o",
        rendering_folder,
        "-n",
        str(cfg["frames"]),
        # "100",
        "-f",
        r"60",
        "-t",
        str(cfg["initial_time"]),
        "-y",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    # Base paths
    base_dir = Path(r"D:\Data\GTC2025DC_Demo\B1021")
    render_dir = base_dir / "rendering"

    # Pattern to match sweep folders
    n = 16

    src_filename_base = "20251021_to_sim_tdSimClothB_01_physics_sweep_"
    dst_filename = "20251021_to_sim_tdSimClothB_01_physics_sim_v.usd"

    # Ensure rendering directory exists
    render_dir.mkdir(exist_ok=True)
    videos = []
    for sweep_idx in range(n):
        sweep_folder = os.path.join(base_dir, "sweep_" + str(sweep_idx).zfill(3))
        cfg = json.load(open(os.path.join(sweep_folder, "config.json")))
        rendering_folder = os.path.join(sweep_folder, r"rendering")
        render_usd(sweep_folder, rendering_folder, cfg)

        videos.append(
            make_videos_for_sweep(
                rendering_folder,
                cfg,
                60,
            )
        )

    run_in_batches_of_4(videos, base_dir)

    print("\n✅ Done!")
