# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Non-invasive screenshot and video capture for Newton physics simulations.

Usage::

    from experiments.collision_benchmark._capture import Capture

    with Capture(out_dir="path/to/output") as cap:
        cap.screenshot(model, state, "frame.png")
        cap.record(model, state, step_fn=sim.step, steps=300, filename="sim.mp4")
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ._deps import ensure_imageio_ffmpeg
from ._display import VirtualDisplay, needs_virtual_display
from ._video import VideoWriter

if TYPE_CHECKING:
    import newton

__all__ = ["Capture"]


def _look_at_angles(pos: tuple, target: tuple) -> tuple[float, float]:
    """Compute (yaw, pitch) in degrees for Z-up coordinate system."""
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    dz = target[2] - pos[2]
    dist_xy = np.sqrt(dx * dx + dy * dy)
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = np.degrees(np.arctan2(dz, dist_xy)) if dist_xy > 1e-8 else (90.0 if dz > 0 else -90.0)
    return float(yaw), float(pitch)


def _create_viewer_gl(viewer_module, width: int, height: int, shading_style: str):
    """Create a headless ViewerGL, tolerating older Newton versions without shading_style."""
    import inspect  # noqa: PLC0415

    sig = inspect.signature(viewer_module.ViewerGL.__init__)
    kwargs = {"width": width, "height": height, "headless": True}
    if "shading_style" in sig.parameters:
        kwargs["shading_style"] = shading_style
    return viewer_module.ViewerGL(**kwargs)


class Capture:
    """Context manager for capturing screenshots and video from Newton simulations.

    Handles Xvfb display setup, ViewerGL creation, frame capture, and cleanup.

    Args:
        out_dir: Directory for output files.
        width: Frame width in pixels.
        height: Frame height in pixels.
        camera_pos: Default camera position (x, y, z).
        camera_target: Default look-at point (x, y, z).
        camera_fov: Default field of view in degrees.
        shading_style: ViewerGL shading style ("classic" or "studio").
    """

    def __init__(
        self,
        out_dir: str,
        width: int = 1280,
        height: int = 720,
        camera_pos: tuple[float, float, float] | None = None,
        camera_target: tuple[float, float, float] | None = None,
        camera_fov: float | None = None,
        shading_style: str = "studio",
    ):
        self.out_dir = out_dir
        self.width = width
        self.height = height
        self.camera_pos = camera_pos
        self.camera_target = camera_target
        self.camera_fov = camera_fov
        self.shading_style = shading_style

        self._display: VirtualDisplay | None = None
        self._viewer = None  # ViewerGL, created lazily
        self._current_model = None
        self._ffmpeg_ok = False

    def __enter__(self):
        os.makedirs(self.out_dir, exist_ok=True)

        # Check the optional ffmpeg backend
        self._ffmpeg_ok = ensure_imageio_ffmpeg()
        if not self._ffmpeg_ok:
            print("[newton_capture] Warning: imageio-ffmpeg not available, video will fall back to PNG frames")

        # Start virtual display if needed
        if needs_virtual_display():
            self._display = VirtualDisplay(width=self.width, height=self.height)
            self._display.start()
            print(f"[newton_capture] Started Xvfb on {self._display.display}")

        return self

    def __exit__(self, *exc):
        self._close_viewer()
        if self._display is not None:
            self._display.stop()
            self._display = None

    def _get_viewer(self, model: newton.Model) -> object:
        """Get or create a ViewerGL for the given model."""
        import newton.viewer  # noqa: PLC0415

        if self._viewer is not None and self._current_model is model:
            return self._viewer

        self._close_viewer()
        self._viewer = _create_viewer_gl(newton.viewer, self.width, self.height, self.shading_style)
        self._viewer.set_model(model)
        self._current_model = model
        return self._viewer

    def _close_viewer(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
            self._current_model = None

    def _apply_camera(
        self,
        viewer,
        camera_pos: tuple | None = None,
        camera_target: tuple | None = None,
        camera_fov: float | None = None,
    ):
        """Set camera parameters on the viewer."""
        from pyglet.math import Vec3 as PyVec3

        pos = camera_pos or self.camera_pos
        target = camera_target or self.camera_target
        fov = camera_fov or self.camera_fov

        if pos is not None:
            viewer.camera.pos = PyVec3(*pos)
        if target is not None and pos is not None:
            yaw, pitch = _look_at_angles(pos, target)
            viewer.camera.yaw = yaw
            viewer.camera.pitch = pitch
        if fov is not None:
            viewer.camera.fov = fov

    def _render_frame(self, viewer, model: newton.Model, state: newton.State) -> np.ndarray:
        """Render one frame and return as numpy RGB array (H, W, 3)."""
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        frame_wp = viewer.get_frame()
        return frame_wp.numpy()

    def screenshot(
        self,
        model: newton.Model,
        state: newton.State,
        filename: str = "screenshot.png",
        camera_pos: tuple | None = None,
        camera_target: tuple | None = None,
        camera_fov: float | None = None,
    ) -> str:
        """Capture a single frame as a PNG image.

        Args:
            model: Newton model.
            state: Current simulation state.
            filename: Output filename (relative to out_dir if not absolute).
            camera_pos: Override camera position.
            camera_target: Override look-at point.
            camera_fov: Override field of view.

        Returns:
            Absolute path to the saved image.
        """
        from PIL import Image

        viewer = self._get_viewer(model)
        self._apply_camera(viewer, camera_pos, camera_target, camera_fov)
        frame = self._render_frame(viewer, model, state)

        path = filename if os.path.isabs(filename) else os.path.join(self.out_dir, filename)
        Image.fromarray(frame).save(path)
        print(f"[newton_capture] Screenshot saved: {path}")
        return path

    def record(
        self,
        model: newton.Model,
        state: newton.State,
        step_fn: Callable,
        steps: int = 300,
        fps: int = 60,
        filename: str = "recording.mp4",
        camera_pos: tuple | None = None,
        camera_target: tuple | None = None,
        camera_fov: float | None = None,
    ) -> str:
        """Record a video by stepping the simulation and capturing each frame.

        Args:
            model: Newton model.
            state: Current simulation state (will be modified by step_fn).
            step_fn: Callable that advances the simulation by one frame.
                Called as ``step_fn()`` each frame.
            steps: Number of frames to record.
            fps: Video frame rate.
            filename: Output filename (relative to out_dir if not absolute).
            camera_pos: Override camera position.
            camera_target: Override look-at point.
            camera_fov: Override field of view.

        Returns:
            Absolute path to the saved video (or frame directory on fallback).
        """
        viewer = self._get_viewer(model)
        self._apply_camera(viewer, camera_pos, camera_target, camera_fov)

        path = filename if os.path.isabs(filename) else os.path.join(self.out_dir, filename)

        with VideoWriter(path, fps=fps) as writer:
            for i in range(steps):
                frame = self._render_frame(viewer, model, state)
                writer.write_frame(frame)
                step_fn()
                if (i + 1) % 100 == 0:
                    print(f"[newton_capture] Recorded frame {i + 1}/{steps}")
            result_path = writer.close()

        print(f"[newton_capture] Video saved: {result_path} ({steps} frames @ {fps} fps)")
        return result_path

    def capture_example(
        self,
        example_name: str,
        screenshot: bool = True,
        video: bool = False,
        duration: float = 5.0,
        fps: int = 60,
        camera_pos: tuple | None = None,
        camera_target: tuple | None = None,
        camera_fov: float | None = None,
    ) -> dict[str, str]:
        """Run a named Newton example and capture its output.

        Args:
            example_name: Name of the example (e.g., "basic_pendulum").
            screenshot: Capture a screenshot after initial setup.
            video: Record a video of the simulation.
            duration: Video duration in seconds.
            fps: Video frame rate.
            camera_pos: Override camera position.
            camera_target: Override look-at point.
            camera_fov: Override field of view.

        Returns:
            Dict mapping capture type ("screenshot", "video") to file paths.
        """
        import importlib  # noqa: PLC0415
        import sys  # noqa: PLC0415

        import newton.examples  # noqa: PLC0415

        # Find and load the example module
        examples = newton.examples.get_examples()
        if example_name not in examples:
            available = ", ".join(sorted(examples.keys()))
            raise ValueError(f"Unknown example '{example_name}'. Available: {available}")

        module_name = examples[example_name]
        # Temporarily override sys.argv so the example's parser works
        old_argv = sys.argv
        sys.argv = [module_name, "--headless", "--num-frames", str(int(duration * fps))]

        try:
            mod = importlib.import_module(module_name)
            ExampleClass = mod.Example

            # Create a headless ViewerGL for the example
            import newton.viewer  # noqa: PLC0415

            viewer = _create_viewer_gl(newton.viewer, self.width, self.height, self.shading_style)

            args = newton.examples.default_args()
            example = ExampleClass(viewer, args)

            # Apply camera
            self._apply_camera(viewer, camera_pos, camera_target, camera_fov)

            results = {}

            # Run a few warmup steps so the scene is initialized
            for _ in range(10):
                example.step()
            example.render()

            if screenshot:
                frame = viewer.get_frame().numpy()
                from PIL import Image

                path = os.path.join(self.out_dir, f"{example_name}.png")
                Image.fromarray(frame).save(path)
                results["screenshot"] = path
                print(f"[newton_capture] Example screenshot: {path}")

            if video:
                path = os.path.join(self.out_dir, f"{example_name}.mp4")
                total_frames = int(duration * fps)

                with VideoWriter(path, fps=fps) as writer:
                    for i in range(total_frames):
                        example.step()
                        example.render()
                        frame = viewer.get_frame().numpy()
                        writer.write_frame(frame)
                        if (i + 1) % 100 == 0:
                            print(f"[newton_capture] Recorded frame {i + 1}/{total_frames}")
                    result_path = writer.close()

                results["video"] = result_path
                print(f"[newton_capture] Example video: {result_path}")

            viewer.close()
            return results

        finally:
            sys.argv = old_argv
