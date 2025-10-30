import json

import cv2
import numpy as np
from pathlib import Path

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from pathlib import Path
import glob
import re
from typing import List, Optional, Tuple, Union
from os.path import join
from wedge_rendering import *

# ---------- 工具：按文件名中的数字排序 ----------
def _numeric_key(p: Path):
    m = re.search(r'(\d+)(?=\.[a-zA-Z]+$)', p.name)
    return (int(m.group(1)) if m else float('inf'), p.name)

def _collect_images(path_like: Union[str, Path]) -> List[Path]:
    """path_like 可是文件夹或通配符（eg: '.../camA/*.png'）"""
    p = Path(path_like) if path_like is not None else None
    if not p:
        return []
    if p.is_dir():
        imgs = sorted(p.glob("*.png"), key=_numeric_key)
    else:
        imgs = sorted((Path(x) for x in glob.glob(str(p))), key=_numeric_key)
    return [x for x in imgs if x.suffix.lower() == ".png"]

# ---------- 多行角标（首行更大） ----------
def _draw_multiline_label(
    img,
    label: Union[str, List[str], None],
    top_left: Tuple[int, int] = (10, 10),
    base_scale: float = 0.7,             # 第二行及以后字号
    first_line_scale_factor: float = 2.0,# 第一行 = base_scale * factor
    font = cv2.FONT_HERSHEY_SIMPLEX,
    color = (255, 255, 255),
    thickness: int = 1,
    pad: int = 6,
    gap_factor: float = 0.6,
    bg_alpha: float = 0.5
):
    if not label:
        return
    lines = label.splitlines() if isinstance(label, str) else [str(x) for x in label]
    if not lines:
        return

    x0, y0 = top_left
    scales = [base_scale * first_line_scale_factor] + [base_scale] * (len(lines) - 1)

    sizes, line_heights = [], []
    for i, line in enumerate(lines):
        s = scales[i]
        t = max(1, int(thickness * max(s, 1)))
        (tw, th), _ = cv2.getTextSize(line, font, s, t)
        sizes.append((tw, th))
        line_heights.append(th)

    gaps = [int(h * gap_factor) for h in line_heights[:-1]] + [0]
    text_w = max((tw for tw, _ in sizes), default=0)
    text_h = sum(line_heights) + sum(gaps)

    x1 = x0 + text_w + pad * 2
    y1 = y0 + text_h + pad * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    y = y0 + pad
    for i, line in enumerate(lines):
        s = scales[i]
        t = max(1, int(thickness * max(s, 1)))
        y += line_heights[i]
        cv2.putText(img, line, (x0 + pad, y), font, s, color, t, cv2.LINE_AA)
        y += gaps[i]

# ---------- 中心裁剪 ----------
def _center_crop(img: np.ndarray, ratio_wh: Tuple[float, float]) -> np.ndarray:
    """按 (rw, rh) 在中心裁剪，rw/rh ∈ (0,1]，返回裁剪区域。"""
    h, w = img.shape[:2]
    rw = max(0.0, min(1.0, float(ratio_wh[0])))
    rh = max(0.0, min(1.0, float(ratio_wh[1])))
    cw = max(1, int(round(w * rw)))
    ch = max(1, int(round(h * rh)))
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)
    return img[y0:y0 + ch, x0:x0 + cw]

# ---------- 自动选择“裁剪后”的单格尺寸 ----------
def _choose_cell_size_after_crop(img_lists: List[List[Path]], ratio_wh: Tuple[float, float], mode: str) -> Tuple[int, int]:
    """
    mode: 'first' | 'min' | 'max'
    基于每路首帧 -> 中心裁剪 -> 得到候选尺寸，然后选取策略。
    """
    sizes = []
    for imgs in img_lists:
        if imgs:
            im = cv2.imread(str(imgs[0]))
            if im is not None:
                cropped = _center_crop(im, ratio_wh)
                sizes.append((cropped.shape[1], cropped.shape[0]))
    if not sizes:
        return (640, 360)
    if mode == 'min':
        return (min(w for w, h in sizes), min(h for w, h in sizes))
    if mode == 'max':
        return (max(w for w, h in sizes), max(h for w, h in sizes))
    return sizes[0]  # 'first'

# ---------- 主函数：1×3，中心裁剪拼接 ----------
def image_folders_to_grid_1x3_center_crop(
    folders: List[Optional[Union[str, Path]]],   # 最多 3 个；可文件夹或 '*.png'
    out_path: Union[str, Path],
    fps: float = 60.0,                           # PNG 序列需指定输出帧率
    crop_ratio: Optional[float] = None,          # 单值比例（同时作用于宽高），如 0.6=保留中间60%
    crop_ratio_wh: Optional[Tuple[float, float]] = None,  # 单独给 (rw, rh)
    cell_size: Optional[Tuple[int, int]] = None, # None=基于裁剪后自动；否则 (w,h)
    cell_size_mode: str = 'first',               # 'first' | 'min' | 'max'
    labels: Optional[List[Optional[Union[str, List[str]]]]] = None,
    label_base_scale: float = 1.2,
    label_first_line_factor: float = 2.5,
    write_final_hold_frame: bool = False
):
    # 裁剪比例解析
    if crop_ratio_wh is not None:
        rw, rh = crop_ratio_wh
    else:
        if crop_ratio:
            rw, rh = crop_ratio
        else:
            r = 0.6 if crop_ratio is None else float(crop_ratio)  # 默认保留中间60%
            rw, rh = r, r
    ratio_wh = (rw, rh)

    # 补齐 3 路
    folders = list(folders)
    while len(folders) < 3:
        folders.append(None)

    # 收集每路 PNG 列表
    img_lists: List[List[Path]] = []
    for f in folders:
        imgs = _collect_images(f) if f else []
        img_lists.append(imgs)

    # 单格尺寸（按“裁剪后”的尺寸来推断）
    if cell_size is None:
        cell_size = _choose_cell_size_after_crop(img_lists, ratio_wh, cell_size_mode)
    cell_w, cell_h = cell_size

    out_w, out_h = cell_w * 3, cell_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (out_w, out_h))

    def black_cell():
        return np.zeros((cell_h, cell_w, 3), np.uint8)

    # 主循环：用最长序列为上限，其余用最后一帧保持；全部完成即可停止
    idxs = [0, 0, 0]
    last_frames = [None, None, None]
    first_written = False

    # 以三路中“最大帧数”为循环上限
    min_len = min(len(lst) for lst in img_lists) if any(img_lists) else 0

    for step in range(min_len if min_len > 0 else 1):
        any_new = False
        cells = []

        for i in range(3):
            imgs = img_lists[i]
            frame = None
            if step < len(imgs):
                im = cv2.imread(str(imgs[step]))
                if im is not None:
                    cropped = _center_crop(im, ratio_wh)
                    frame = cv2.resize(cropped, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                    last_frames[i] = frame
                    any_new = True

            if frame is None:
                frame = last_frames[i] if last_frames[i] is not None else black_cell()

            # 多行角标（第一行更大）
            if labels and i < len(labels) and labels[i]:
                _draw_multiline_label(
                    frame,
                    labels[i],
                    top_left=(10, 10),
                    base_scale=label_base_scale,
                    first_line_scale_factor=label_first_line_factor,
                    thickness=1,
                    pad=6,
                    gap_factor=0.6,
                    bg_alpha=0.5,
                )

            # 尺寸兜底
            if frame.shape[0] != cell_h or frame.shape[1] != cell_w:
                frame = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

            cells.append(frame)

        grid = np.hstack((cells[0], cells[1], cells[2]))

        if any_new:
            writer.write(grid)
            first_written = True
        else:
            if write_final_hold_frame and not first_written:
                writer.write(grid)
            break

    writer.release()
    print(f"✅ 1x3 grid (center-crop) saved: {out_path} | crop=({rw:.2f}, {rh:.2f}) | cell {cell_w}x{cell_h} | out {out_w}x{out_h}")

def get_desc(cfg):
    return "bending: " + str(cfg["cloth_cfg"]["bending_ke"]) + " | damping: " + str(cfg["cloth_cfg"]["bending_kd"])

if __name__ == '__main__':

    # render_usd(r"D:\Data\GTC2025DC_Demo\B1021_2\sweep_air_drag_000",
    #            r"D:\Data\GTC2025DC_Demo\B1021_2\rendering2",
    #            r"D:\Data\GTC2025DC_Demo\B1021_2\rendering2\rendering",
    #            sim_project_name="full_20251024_to_sim_tdSimClothB_01.usd",
    #            dst_filename="20251024_to_sim_tdSimClothB_01_physics_sim_v.usd", )

    best_6 = [
        # sweep 14
        r"D:\Data\GTC2025DC_Demo\B1021_2\sweep_air_drag_000",
        r"D:\Data\GTC2025DC_Demo\B1024\bending_damping\sweep_bending_damping_000",
        r"D:\Data\GTC2025DC_Demo\B1024\bending_damping\sweep_bending_damping_002",
        r"D:\Data\GTC2025DC_Demo\B1024\new_bending_ke\sweep_new_sim_000",
        r"D:\Data\GTC2025DC_Demo\B1024\new_bending_ke\sweep_new_sim_001",
        r"D:\Data\GTC2025DC_Demo\B1024\new_bending_ke\sweep_new_sim_002",
    ]

    names = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F"
    ]

    out_path = r'D:\Data\GTC2025DC_Demo\for_picking\Best6_01'

    labels = []

    for i, in_path in enumerate(best_6):
        cfg = json.load(open(join(in_path, "config.json")))
        labels.append([names[i], get_desc(cfg)])

    image_folders_to_grid_1x3_center_crop(
        [join(f, "rendering") for f in best_6[:3]],
        join(out_path, "rendering_A_C.mp4"),
        crop_ratio=(0.4, 0.8),
        labels=labels
    )

    image_folders_to_grid_1x3_center_crop(
        [join(f, "rendering") for f in best_6[3:]],
        join(out_path, "rendering_D_G.mp4"),
        crop_ratio=(0.4, 0.8),
        labels=labels[3:]
    )