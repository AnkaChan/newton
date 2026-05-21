# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import imageio.v2 as imageio

_CAPTURE_SENTINEL = object()
_LOG_PREFIX = "[replay_capture]"


def configure_capture(
    owner: Any,
    *,
    save_mp4: str | None = None,
    capture_replay: bool = False,
    capture_frames: int = 300,
    capture_fps: int = 60,
    capture_dir: str = "outputs/replay_capture",
    capture_format: str = "mp4",
    capture_background_writes: bool = True,
    capture_max_pending_frames: int = 1,
) -> None:
    """Populate common capture state for a bag example instance."""
    owner.save_mp4 = save_mp4
    owner.capture_replay = bool(capture_replay)
    owner.capture_frames = int(capture_frames)
    owner.capture_fps = int(capture_fps)
    owner.capture_format = str(capture_format)
    owner.capture_count = 0
    owner.capture_done = False
    owner.capture_video_path = None
    owner.capture_dir = None
    owner._video_process = None
    owner._capture_frame_queue = None
    owner._capture_writer_thread = None
    owner._capture_write_error = None
    owner._capture_background_writes = bool(capture_background_writes)
    owner._capture_max_pending_frames = max(1, int(capture_max_pending_frames))
    owner._last_captured_frame_key = None

    if owner.capture_replay and owner.capture_frames > 0:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(capture_dir)
        owner.capture_dir = base_dir / f"run_{run_tag}"
        owner.capture_dir.mkdir(parents=True, exist_ok=True)
        _log(owner, f"Capture directory: {owner.capture_dir.resolve()}")
        _log(
            owner,
            f"  PNG frames will be written here; replay.{owner.capture_format} "
            "is stitched at the end.",
        )


def add_capture_arguments(
    parser,
    *,
    replay_help: str,
    capture_frames_default: int = 300,
    capture_fps_default: int = 60,
    include_save_mp4: bool = True,
) -> None:
    """Add the shared MP4 and replay-capture CLI flags.

    ``capture_fps_default`` lets each example pick a default that matches
    its physics-step rate (so the captured video plays at real time
    rather than at e.g. 2x speed). Users can still override with
    ``--capture-fps`` on the command line.
    """
    if include_save_mp4:
        parser.add_argument(
            "--save-mp4",
            type=str,
            default=None,
            help="Save simulation to MP4 file",
        )
    parser.add_argument(
        "--capture-replay",
        action="store_true",
        help=replay_help,
    )
    parser.add_argument(
        "--capture-frames",
        type=int,
        default=capture_frames_default,
        help="Number of frames to capture when replay capture is enabled",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=capture_fps_default,
        help="Output replay video FPS (defaults to the example's physics step rate)",
    )
    parser.add_argument(
        "--capture-dir",
        type=str,
        default="outputs/replay_capture",
        help="Directory to store captured frames and replay video",
    )
    parser.add_argument(
        "--capture-format",
        type=str,
        default="mp4",
        choices=["mp4", "gif"],
        help="Preferred replay output format",
    )


def get_viewer_frame(viewer: Any, *, render_ui: bool = False):
    """Read a frame from the viewer across both supported call signatures."""
    try:
        return viewer.get_frame(render_ui=render_ui)
    except TypeError:
        return viewer.get_frame()


def init_video_capture(owner: Any) -> None:
    """Start ffmpeg-based frame capture for ``--save-mp4`` if available."""
    if not hasattr(owner.viewer, "get_frame"):
        _warn(owner, "viewer lacks get_frame(); skipping MP4")
        return

    try:
        width = owner.viewer.renderer._screen_width
        height = owner.viewer.renderer._screen_height
    except AttributeError:
        _warn(owner, "cannot determine screen size; skipping MP4")
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(owner.fps),
        "-i",
        "pipe:0",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        owner.save_mp4,
    ]
    try:
        owner._video_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        _warn(owner, "ffmpeg not found; skipping MP4")


def write_video_frame(owner: Any) -> None:
    """Push the current viewer frame into the ffmpeg pipe when enabled."""
    if owner._video_process is None or not hasattr(owner.viewer, "get_frame"):
        return

    frame = get_viewer_frame(owner.viewer)
    owner._video_process.stdin.write(frame.numpy().tobytes())


def capture_replay_frame(
    owner: Any,
    *,
    frame_key: Any | None = None,
    target_frame_count: int | None = None,
    close_viewer: bool = True,
) -> None:
    """Write one replay frame and finalize when the quota is reached."""
    if not owner.capture_replay or owner.capture_done:
        return
    if owner.capture_dir is None:
        return
    target_count = _target_frame_count(owner, target_frame_count)
    if owner.capture_count >= target_count:
        finalize_replay_video(owner)
        owner.capture_done = True
        if close_viewer:
            _close_viewer(owner)
        return
    last_frame_key = getattr(owner, "_last_captured_frame_key", None)
    if frame_key is not None and frame_key == last_frame_key:
        return
    if not hasattr(owner.viewer, "get_frame"):
        return

    frame_wp = get_viewer_frame(owner.viewer, render_ui=False)
    frame_np = frame_wp.numpy()
    out_path = owner.capture_dir / f"frame_{owner.capture_count:05d}.png"
    _queue_capture_frame(owner, out_path, frame_np)
    owner.capture_count += 1
    owner._last_captured_frame_key = frame_key

    if owner.capture_count % 20 == 0:
        print(
            f"{_LOG_PREFIX} saved "
            f"{owner.capture_count}/{target_count} frames"
        )

    if owner.capture_count >= target_count:
        finalize_replay_video(owner)
        owner.capture_done = True
        if close_viewer:
            _close_viewer(owner)


def should_hold_simulation_for_capture(owner: Any) -> bool:
    """Return True while pending capture writes should throttle simulation."""
    if (
        not getattr(owner, "capture_replay", False)
        or getattr(owner, "capture_done", False)
    ):
        return False
    queue = getattr(owner, "_capture_frame_queue", None)
    if queue is None:
        return False
    max_pending = int(getattr(owner, "_capture_max_pending_frames", 2))
    pending_writes = int(getattr(queue, "unfinished_tasks", queue.qsize()))
    return pending_writes >= max_pending


def finalize_replay_video(owner: Any) -> None:
    """Assemble the captured PNG sequence into an MP4 or GIF."""
    if owner.capture_dir is None:
        return
    _stop_capture_writer(owner)

    png_files = sorted(owner.capture_dir.glob("frame_*.png"))
    if len(png_files) == 0:
        return

    try:
        if owner.capture_format == "gif":
            video_path = owner.capture_dir / "replay.gif"
            with imageio.get_writer(
                video_path,
                mode="I",
                duration=1.0 / max(owner.capture_fps, 1),
            ) as writer:
                for path in png_files:
                    writer.append_data(imageio.imread(path))
        else:
            video_path = owner.capture_dir / "replay.mp4"
            with imageio.get_writer(
                video_path,
                fps=max(owner.capture_fps, 1),
                codec="libx264",
            ) as writer:
                for path in png_files:
                    writer.append_data(imageio.imread(path))
        owner.capture_video_path = video_path
        print(f"{_LOG_PREFIX} wrote video: {video_path}")
    except Exception as exc:
        fallback = owner.capture_dir / "replay.gif"
        with imageio.get_writer(
            fallback,
            mode="I",
            duration=1.0 / max(owner.capture_fps, 1),
        ) as writer:
            for path in png_files:
                writer.append_data(imageio.imread(path))
        owner.capture_video_path = fallback
        print(
            f"{_LOG_PREFIX} mp4 failed ({exc});"
            f" wrote gif: {fallback}"
        )


def finalize_capture(owner: Any) -> None:
    """Flush any pending replay output and shut down ffmpeg cleanly."""
    _stop_capture_writer(owner)
    if (
        getattr(owner, "capture_replay", False)
        and getattr(owner, "capture_count", 0) > 0
        and getattr(owner, "capture_video_path", None) is None
    ):
        finalize_replay_video(owner)

    video_process = getattr(owner, "_video_process", None)
    if video_process is None:
        return

    stdin = getattr(video_process, "stdin", None)
    if stdin is not None and not stdin.closed:
        stdin.close()
    video_process.wait()


def trim_replay_capture(
    owner: Any,
    frame_count: int,
    *,
    target_frame_count: int | None = None,
) -> None:
    """Discard captured replay frames beyond an accepted replay prefix."""
    keep_count = max(0, int(frame_count))
    _stop_capture_writer(owner)

    if owner.capture_dir is None:
        owner.capture_count = min(owner.capture_count, keep_count)
        owner._last_captured_frame_key = (
            owner.capture_count - 1 if owner.capture_count > 0 else None
        )
        return

    if owner.capture_count <= keep_count:
        # Rollback can accept frames before render writes their PNGs.
        # Do not mark those future frame keys as already captured.
        owner._last_captured_frame_key = (
            owner.capture_count - 1 if owner.capture_count > 0 else None
        )
        return

    for frame_index in range(keep_count, owner.capture_count):
        frame_path = owner.capture_dir / f"frame_{frame_index:05d}.png"
        frame_path.unlink(missing_ok=True)

    owner.capture_count = keep_count
    owner.capture_done = False
    owner.capture_video_path = None
    owner._last_captured_frame_key = (
        keep_count - 1 if keep_count > 0 else None
    )
    target_count = _target_frame_count(owner, target_frame_count)
    print(
        f"{_LOG_PREFIX} trimmed to {keep_count}/"
        f"{target_count} accepted frames"
    )


def _target_frame_count(owner: Any, target_frame_count: int | None) -> int:
    if target_frame_count is None:
        target_frame_count = getattr(owner, "capture_frames", 0)
    return max(0, int(target_frame_count))


def _log(owner: Any, message: str) -> None:
    print(f"{_LOG_PREFIX} {message}")


def _warn(owner: Any, message: str) -> None:
    _log(owner, f"Warning: {message}")


def _queue_capture_frame(owner: Any, out_path: Path, frame_np: Any) -> None:
    _raise_capture_write_error(owner)
    if not getattr(owner, "_capture_background_writes", True):
        imageio.imwrite(out_path, frame_np)
        return

    queue = getattr(owner, "_capture_frame_queue", None)
    if queue is None:
        queue = Queue()
        owner._capture_frame_queue = queue
        owner._capture_writer_thread = Thread(
            target=_capture_writer_main,
            args=(owner, queue),
            name="bag-replay-capture-writer",
            daemon=True,
        )
        owner._capture_writer_thread.start()
    queue.put((out_path, frame_np.copy()))


def _capture_writer_main(owner: Any, queue: Queue) -> None:
    while True:
        item = queue.get()
        try:
            if item is _CAPTURE_SENTINEL:
                return
            out_path, frame_np = item
            imageio.imwrite(out_path, frame_np)
        except Exception as exc:
            owner._capture_write_error = exc
        finally:
            queue.task_done()


def _stop_capture_writer(owner: Any) -> None:
    queue = getattr(owner, "_capture_frame_queue", None)
    thread = getattr(owner, "_capture_writer_thread", None)
    if queue is None or thread is None:
        _raise_capture_write_error(owner)
        return

    queue.join()
    queue.put(_CAPTURE_SENTINEL)
    thread.join()
    owner._capture_frame_queue = None
    owner._capture_writer_thread = None
    _raise_capture_write_error(owner)


def _raise_capture_write_error(owner: Any) -> None:
    error = getattr(owner, "_capture_write_error", None)
    if error is not None:
        owner._capture_write_error = None
        raise RuntimeError("Replay frame capture failed.") from error


def _close_viewer(owner: Any) -> None:
    if hasattr(owner, "_stop_viewer"):
        owner._stop_viewer(close=True)
        return
    if hasattr(owner.viewer, "close"):
        owner.viewer.close()
