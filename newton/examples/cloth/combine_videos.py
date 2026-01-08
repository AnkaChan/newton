#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np


def _video_info(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, frame_count, width, height


def _set_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def _draw_caption(
    img: np.ndarray,
    text: str,
    font_scale: float,
    font_thickness: int,
    margin_px: int = 20,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = margin_px, margin_px + text_h

    # background box
    box_w = text_w + 2 * margin_px // 3
    box_h = text_h + 2 * margin_px // 3
    cv2.rectangle(
        img,
        (x - margin_px // 3, y - text_h - margin_px // 3),
        (x - margin_px // 3 + box_w, y + box_h // 3),
        (0, 0, 0),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # outline
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
    # foreground
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


def combine_side_by_side(
    video_a: str,
    video_b: str,
    output_path: str,
    start_a: int = 0,
    start_b: int = 0,
    max_frames: int | None = None,
    caption_a: str | None = None,
    caption_b: str | None = None,
    font_scale: float = 1.0,
    font_thickness: int = 2,
):
    cap_a, fps_a, count_a, w_a, h_a = _video_info(video_a)
    cap_b, fps_b, count_b, w_b, h_b = _video_info(video_b)

    if start_a >= count_a or start_b >= count_b:
        raise ValueError("Start frame is beyond video length.")

    _set_frame(cap_a, start_a)
    _set_frame(cap_b, start_b)

    # Choose fps conservatively to keep both in sync
    fps = min(fps_a, fps_b) if fps_a and fps_b else fps_a or fps_b or 30.0

    # Compute resizing to match heights
    target_h = min(h_a, h_b)
    scale_a = target_h / h_a if h_a else 1.0
    scale_b = target_h / h_b if h_b else 1.0
    target_w_a = int(round(w_a * scale_a))
    target_w_b = int(round(w_b * scale_b))

    writer = None
    frames_written = 0
    max_available = min(count_a - start_a, count_b - start_b)
    if max_frames is not None:
        max_available = min(max_available, max_frames)

    while frames_written < max_available:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            break

        if frame_a.shape[0] != target_h:
            frame_a = cv2.resize(frame_a, (target_w_a, target_h), interpolation=cv2.INTER_AREA)
        if frame_b.shape[0] != target_h:
            frame_b = cv2.resize(frame_b, (target_w_b, target_h), interpolation=cv2.INTER_AREA)

        if caption_a:
            _draw_caption(frame_a, caption_a, font_scale, font_thickness)
        if caption_b:
            _draw_caption(frame_b, caption_b, font_scale, font_thickness)

        combined = np.hstack([frame_a, frame_b])

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (combined.shape[1], combined.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

        writer.write(combined)
        frames_written += 1

    cap_a.release()
    cap_b.release()
    if writer is not None:
        writer.release()


def main():
    # Hard-coded configuration. Edit these paths/captions as needed.
    # video_a = r"D:\Data\VBD_cloth_Results\Divide_and_truncate\Twist_and_release/isometric.mp4"
    # video_b = r"D:\Data\VBD_cloth_Results\Divide_and_truncate\Twist_and_release/planar.mp4"
    # output_path = r"D:\Data\VBD_cloth_Results\Divide_and_truncate\Twist_and_release\combined.mp4"

    video_a = r"D:\Data\DAT_Sim\cloth_twist_release\truncation_1_iter_10_20260105_150357\video.mp4"
    video_b = r"D:\Data\DAT_Sim\cloth_twist_release\truncation_0_iter_10_20260105_154924\video.mp4"
    output_path = r"D:\Data\DAT_Sim\cloth_twist_release\combined_truncation_comparison.mp4"

    caption_a = "Truncation Mode 1"
    caption_b = "Truncation Mode 0"

    # Optional start frames
    start_a = 0
    start_b = 0

    # Optional limit on frames to write (None = run until either video ends)
    max_frames = None

    combine_side_by_side(
        video_a=video_a,
        video_b=video_b,
        output_path=output_path,
        start_a=start_a,
        start_b=start_b,
        max_frames=max_frames,
        caption_a=caption_a,
        caption_b=caption_b,
        font_scale=2.0,
        font_thickness=3,
    )


if __name__ == "__main__":
    main()
