# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MP4 video encoding from numpy frames via imageio-ffmpeg."""

from __future__ import annotations

import os

import numpy as np


class VideoWriter:
    """Write numpy RGB frames to an MP4 file.

    Uses imageio with the ffmpeg backend. Falls back to saving individual
    PNG frames if ffmpeg is unavailable.
    """

    def __init__(self, path: str, fps: int = 60, quality: int = 8):
        self.path = path
        self.fps = fps
        self.quality = quality
        self._writer = None
        self._frame_count = 0
        self._fallback_dir: str | None = None

    def open(self):
        try:
            import imageio  # noqa: PLC0415

            self._writer = imageio.get_writer(
                self.path,
                fps=self.fps,
                quality=self.quality,
                codec="libx264",
                pixelformat="yuv420p",
            )
        except Exception:
            # Fall back to PNG frame sequence
            self._fallback_dir = self.path.rsplit(".", 1)[0] + "_frames"
            os.makedirs(self._fallback_dir, exist_ok=True)
            print(f"[newton_capture] ffmpeg unavailable, saving frames to {self._fallback_dir}/")

    def write_frame(self, frame: np.ndarray):
        """Append an RGB uint8 frame (H, W, 3)."""
        if self._writer is not None:
            self._writer.append_data(frame)
        elif self._fallback_dir is not None:
            from PIL import Image

            img = Image.fromarray(frame)
            img.save(os.path.join(self._fallback_dir, f"frame_{self._frame_count:06d}.png"))
        self._frame_count += 1

    def close(self) -> str:
        """Finalize the video. Returns the output path."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            return self.path
        if self._fallback_dir is not None:
            return self._fallback_dir
        return self.path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
