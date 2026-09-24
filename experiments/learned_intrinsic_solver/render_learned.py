# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render one saved learned-solver physical trajectory with Newton ViewerGL.

Run this module separately for each seed: a second ViewerGL in one process can
render black frames under the headless capture environment. Rendering reads the
saved positions and never advances or changes the physical simulation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

__all__ = ["render_trajectory"]

_FACES = (
    (0, 1, 3, 2),  # -x
    (4, 6, 7, 5),  # +x
    (0, 4, 5, 1),  # -y
    (2, 3, 7, 6),  # +y
    (0, 2, 6, 4),  # -z
    (1, 5, 7, 3),  # +z
)


def _boundary_geometry(cell_corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return outward triangles and exposed voxel-grid edges from hex corners."""
    cells = np.asarray(cell_corners)
    if cells.ndim != 2 or cells.shape[1] != 8 or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("cell_corner_indices must be integer [cell, 8]")
    faces = {}
    for cell in cells:
        for local in _FACES:
            quad = tuple(int(cell[index]) for index in local)
            key = tuple(sorted(quad))
            if key in faces:
                faces[key] = None
            else:
                faces[key] = quad
    quads = [quad for quad in faces.values() if quad is not None]
    if not quads:
        raise ValueError("cell topology has no exposed faces")
    triangles = np.asarray(
        [(quad[0], quad[1], quad[2]) for quad in quads] + [(quad[0], quad[2], quad[3]) for quad in quads],
        dtype=np.int32,
    )
    edges = np.asarray(
        sorted({tuple(sorted((quad[corner], quad[(corner + 1) % 4]))) for quad in quads for corner in range(4)}),
        dtype=np.int32,
    )
    return triangles, edges


def _camera_for_trajectory(positions: np.ndarray, *, fov: float = 45.0) -> dict:
    """Choose one perspective that contains all saved valid trajectory frames."""
    values = np.asarray(positions)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("positions must be finite [frame, corner, 3]")
    lower = values.min(axis=(0, 1)).astype(np.float64)
    upper = values.max(axis=(0, 1)).astype(np.float64)
    target = (lower + upper) / 2
    radius = float(np.linalg.norm(upper - lower) / 2)
    distance = max(0.5, radius / math.sin(math.radians(fov) / 2) * 1.2)
    direction = np.array((1.3, -1.6, 0.9), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    camera = target + distance * direction
    return {
        "position": camera.tolist(),
        "target": target.tolist(),
        "distance": distance,
        "fov_degrees": fov,
        "bounds_min": lower.tolist(),
        "bounds_max": upper.tolist(),
    }


def _load_trajectory(path: Path):
    with np.load(path) as data:
        required = {"positions", "times", "rest_positions", "fixed_indices", "cell_corner_indices"}
        if not required <= set(data.files):
            raise ValueError(f"trajectory is missing {sorted(required - set(data.files))}")
        positions = np.asarray(data["positions"], dtype=np.float32)
        times = np.asarray(data["times"], dtype=np.float64)
        rest = np.asarray(data["rest_positions"], dtype=np.float32)
        fixed = np.asarray(data["fixed_indices"], dtype=np.int64)
        cells = np.asarray(data["cell_corner_indices"], dtype=np.int64)
    if positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[-1] != 3:
        raise ValueError("positions must be nonempty [frame, corner, 3]")
    if rest.shape != positions.shape[1:] or times.shape != (len(positions),):
        raise ValueError("rest positions and times must match trajectory frames")
    if not np.isfinite(positions).all() or not np.isfinite(rest).all() or not np.isfinite(times).all():
        raise ValueError("trajectory positions and times must be finite")
    if abs(times[0]) > 1e-9 or np.any(np.diff(times) <= 0):
        raise ValueError("times must start at zero and strictly increase")
    if fixed.ndim != 1 or len(fixed) < 1 or len(np.unique(fixed)) != len(fixed):
        raise ValueError("fixed_indices must be a nonempty unique vector")
    if np.any(fixed < 0) or np.any(fixed >= len(rest)) or np.any(cells < 0) or np.any(cells >= len(rest)):
        raise ValueError("trajectory topology contains invalid corner indices")
    if not np.allclose(positions[:, fixed], positions[0, fixed], rtol=0, atol=1e-5):
        raise ValueError("saved trajectory moves prescribed pins")
    return positions, times, rest, fixed, cells


def _surface_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    corners = positions[triangles]
    faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], faces)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals


def _annotate(frame: np.ndarray, time_seconds: float, *, final: bool):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 25)
    except OSError:
        font = ImageFont.load_default()
    label = f"Physical time {time_seconds:.3f} s" + ("  ·  last valid state" if final else "")
    text_bounds = draw.textbbox((32, 24), label, font=font)
    draw.rounded_rectangle((20, 18, text_bounds[2] + 14, text_bounds[3] + 8), radius=8, fill=(13, 29, 39))
    draw.text((32, 24), label, font=font, fill=(244, 249, 251))
    return image


def render_trajectory(trajectory: Path, output: Path, *, fps: int = 30, width: int = 1280, height: int = 720) -> dict:
    """Render one saved trajectory into MP4 and first/last PNG frames.

    The source trajectory contains the last valid state if simulation failed;
    this function writes only those frames and records their actual times.
    """
    import warp as wp  # noqa: PLC0415 - Optional rendering boundary.

    import newton  # noqa: PLC0415 - Optional rendering boundary.

    trajectory, output = Path(trajectory), Path(output)
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"render output already contains files: {output}")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")
    positions, times, rest, fixed, cells = _load_trajectory(trajectory)
    triangles, edges = _boundary_geometry(cells)
    camera = _camera_for_trajectory(positions)
    metadata_path = trajectory.with_name("report.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    capture_tools = Path(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")) / "Newton" / "tools"
    sys.path.insert(0, str(capture_tools))
    from newton_capture import Capture  # noqa: PLC0415 - External capture helper.
    from newton_capture._video import VideoWriter  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    model = newton.ModelBuilder().finalize(device="cpu")
    point_array = wp.array(rest, dtype=wp.vec3, device="cpu")
    triangle_array = wp.array(triangles.ravel(), dtype=wp.int32, device="cpu")
    edge_starts = wp.empty(len(edges), dtype=wp.vec3, device="cpu")
    edge_ends = wp.empty_like(edge_starts)
    pin_points = wp.array(rest[fixed], dtype=wp.vec3, device="cpu")
    pin_colors = wp.full(len(fixed), wp.vec3(0.98, 0.42, 0.12), dtype=wp.vec3, device="cpu")
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[fixed] = True
    pin_edges = edges[fixed_mask[edges].all(axis=1)]
    pin_edge_starts = wp.empty(len(pin_edges), dtype=wp.vec3, device="cpu")
    pin_edge_ends = wp.empty_like(pin_edge_starts)
    surface_scale = max(float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0))), 1e-3)
    edge_offset = 0.0004 * surface_scale
    pin_offset = 0.012 * surface_scale
    pin_direction = rest[fixed].astype(np.float64) - rest.mean(axis=0)
    pin_direction /= np.maximum(np.linalg.norm(pin_direction, axis=1, keepdims=True), 1e-12)
    video = output / "simulation.mp4"
    frame_std_min = math.inf
    with (
        wp.ScopedDevice("cpu"),
        Capture(
            out_dir=str(output),
            width=width,
            height=height,
            camera_pos=tuple(camera["position"]),
            camera_target=tuple(camera["target"]),
            camera_fov=camera["fov_degrees"],
        ) as capture,
    ):
        viewer = capture._get_viewer(model)
        viewer.show_particles = False
        viewer.show_ui = False
        viewer.renderer.draw_wireframe = False
        viewer.renderer.line_width = 0.9

        def render_frame(index):
            nonlocal frame_std_min
            current = positions[index]
            point_array.assign(current)
            normals = _surface_normals(current, triangles)
            visible_lines = current + edge_offset * normals
            edge_starts.assign(visible_lines[edges[:, 0]])
            edge_ends.assign(visible_lines[edges[:, 1]])
            pin_points.assign(current[fixed] + pin_offset * pin_direction)
            if len(pin_edges):
                lowered = current.copy()
                lowered[fixed, 2] -= pin_offset
                pin_edge_starts.assign(lowered[pin_edges[:, 0]])
                pin_edge_ends.assign(lowered[pin_edges[:, 1]])
            capture._apply_camera(viewer)
            viewer.begin_frame(float(times[index]))
            viewer.log_mesh("learned_surface", point_array, triangle_array, color=(0.27, 0.69, 0.75), dynamic=True)
            viewer.log_lines("surface_voxel_grid", edge_starts, edge_ends, (0.035, 0.15, 0.19))
            if len(pin_edges):
                viewer.log_lines("clamped_grid", pin_edge_starts, pin_edge_ends, (0.98, 0.42, 0.12))
            viewer.log_points("fixed_corners", pin_points, radii=0.008 * surface_scale, colors=pin_colors)
            viewer.end_frame()
            pixels = viewer.get_frame().numpy()
            frame_std_min = min(frame_std_min, float(pixels.std()))
            if frame_std_min < 3:
                raise RuntimeError(f"black or uniform ViewerGL frame at index {index}")
            return _annotate(
                pixels, float(times[index]), final=index == len(times) - 1 and metadata.get("status") != "complete"
            )

        initial_image = render_frame(0)
        initial_image.save(output / "initial.png")
        video_indices = range(1, len(times)) if len(times) > 1 else range(1)
        with VideoWriter(str(video), fps=fps) as writer:
            for index in video_indices:
                image = render_frame(index) if index else initial_image
                if index == len(times) - 1:
                    image.save(output / "final.png")
                writer.write_frame(np.asarray(image))
                if index % 30 == 0 or index == len(times) - 1:
                    print(f"rendered frame={index}/{len(times) - 1} physical_time={times[index]:.3f}s", flush=True)
    if not video.is_file():
        raise RuntimeError("MP4 encoding did not produce a video; check fallback PNG frames")
    result = {
        "status": "complete",
        "video": video.name,
        "initial_image": "initial.png",
        "final_image": "final.png",
        "rendered_frame_count": max(len(times) - 1, 1),
        "fps": fps,
        "first_physical_time_seconds": float(times[0]),
        "last_physical_time_seconds": float(times[-1]),
        "simulation_status": metadata.get("status"),
        "frame_pixel_std_min": frame_std_min,
        "camera": camera,
        "surface_triangle_count": len(triangles),
        "surface_grid_edge_count": len(edges),
    }
    temporary = output / "render.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "render.json")
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    result = render_trajectory(args.trajectory, args.output, fps=args.fps, width=args.width, height=args.height)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
