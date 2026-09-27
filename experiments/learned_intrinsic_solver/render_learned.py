# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render one saved learned-solver physical trajectory with Newton ViewerGL.

Run this module separately for each seed: a second ViewerGL in one process can
render black frames under the headless capture environment. Rendering reads the
saved positions and never advances or changes the physical simulation.

Trajectories written by ``simulate_mixed`` carry their static contact partners
(``notes/contact-design-20260927.md``, section 3.2). The ground plane, when
present, is drawn as a flat quad at its sampled height spanning at least
:data:`_PLANE_MIN_HALF_EXTENT` around the beam, and every static contact point
as a sphere of its lateral radius with a tick along its normal. The scene
is viewed with +y up, opposite to gravity, so the floor lies under the beam.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

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

_CONTACT_KEYS = (
    "contact_plane_present",
    "contact_plane_point",
    "contact_plane_normal",
    "contact_point_positions",
    "contact_point_normals",
    "contact_point_radii",
)
"""Optional ``trajectory.npz`` entries written by ``simulate_mixed`` for the contact partners."""

_PLANE_MIN_HALF_EXTENT = 1.5
"""Half edge length [m] of the smallest drawn ground quad, so it spans at least 3 m x 3 m."""

_PLANE_MARGIN = 1.25
"""Factor by which the quad exceeds the beam's footprint when that is larger than the minimum."""

_CAMERA_DIRECTION = (1.6, 0.9, 0.9)
"""Unnormalized camera offset from the framed center: the +x side, above the floor, toward the free end."""

_UNIT_NORMAL_TOLERANCE = 1e-3

_PLANE_COLOR = (0.60, 0.62, 0.65)
_PLANE_OPACITY = 0.75
"""Slight translucency so static points sampled below the floor stay faintly visible."""
_POINT_COLOR = (0.93, 0.35, 0.60)
_POINT_NORMAL_COLOR = (0.55, 0.12, 0.32)


class _ContactGeometry(NamedTuple):
    """Static contact partners saved next to a trajectory as float32 arrays [m]."""

    plane_present: bool
    plane_point: np.ndarray
    plane_normal: np.ndarray
    point_positions: np.ndarray
    point_normals: np.ndarray
    point_radii: np.ndarray

    @property
    def point_count(self) -> int:
        return int(len(self.point_positions))

    @classmethod
    def empty(cls) -> _ContactGeometry:
        return cls(
            False,
            np.zeros(3, dtype=np.float32),
            np.array((0.0, 1.0, 0.0), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
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


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("normal must be a finite nonzero vector")
    return values / norm


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the unit normal and two orthonormal tangents ``u, v`` with ``u x v = n``."""
    n = _unit(normal)
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(n)))] = 1.0
    u = _unit(np.cross(n, axis))
    v = np.cross(n, u)
    return n, u, v


def _plane_quad(
    plane_point: np.ndarray, plane_normal: np.ndarray, *, center: np.ndarray, half_extent: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a square quad in the plane, centered under ``center``, whose triangles face along the normal.

    Args:
        plane_point: A point on the plane [m], shape [3].
        plane_normal: Plane normal, shape [3]; normalized here.
        center: Point whose projection onto the plane becomes the quad center [m].
        half_extent: Half edge length of the quad [m].

    Returns:
        Vertices, float32 [4, 3], and triangles, int32 [2, 3], wound so the
        geometric normal of both triangles equals the plane normal.
    """
    if not math.isfinite(half_extent) or half_extent <= 0:
        raise ValueError("half_extent must be finite and positive")
    n, u, v = _plane_basis(plane_normal)
    point = np.asarray(plane_point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    origin = center - np.dot(center - point, n) * n
    vertices = np.stack(
        [
            origin - half_extent * u - half_extent * v,
            origin + half_extent * u - half_extent * v,
            origin + half_extent * u + half_extent * v,
            origin - half_extent * u + half_extent * v,
        ]
    ).astype(np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, triangles


def _plane_geometry(positions: np.ndarray, contact: _ContactGeometry) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the ground quad under all saved frames and its half extent [m].

    The quad is centered under the trajectory's bounding-box center and spans
    at least :data:`_PLANE_MIN_HALF_EXTENT` on each side, more when the frames
    reach farther across the plane.
    """
    values = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n, u, v = _plane_basis(contact.plane_normal)
    center = (values.min(axis=0) + values.max(axis=0)) / 2
    offsets = values - center
    reach = max(float(np.abs(offsets @ u).max()), float(np.abs(offsets @ v).max()))
    half_extent = max(_PLANE_MIN_HALF_EXTENT, _PLANE_MARGIN * reach)
    vertices, triangles = _plane_quad(contact.plane_point, n, center=center, half_extent=half_extent)
    return vertices, triangles, half_extent


def _contact_framing_points(positions: np.ndarray, contact: _ContactGeometry) -> np.ndarray:
    """Return extra points the camera must contain: the floor under the beam and every static sphere."""
    values = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    extra = []
    if contact.plane_present:
        lower, upper = values.min(axis=0), values.max(axis=0)
        corners = np.array(
            [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
        )
        n = _unit(contact.plane_normal)
        point = np.asarray(contact.plane_point, dtype=np.float64)
        extra.append(corners - ((corners - point) @ n)[:, None] * n)
    if contact.point_count:
        centers = np.asarray(contact.point_positions, dtype=np.float64)
        radii = np.asarray(contact.point_radii, dtype=np.float64)[:, None]
        extra.extend((centers - radii, centers + radii))
    return np.concatenate(extra) if extra else np.zeros((0, 3))


def _camera_for_trajectory(positions: np.ndarray, *, fov: float = 45.0, extra_points: np.ndarray | None = None) -> dict:
    """Choose one +y-up perspective that contains all saved valid trajectory frames.

    Args:
        positions: Saved frames [m], shape [frame, corner, 3].
        fov: Vertical field of view in degrees.
        extra_points: Further points [m], shape [K, 3], that the view must also
            contain, for example the floor region under the beam.
    """
    values = np.asarray(positions)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("positions must be finite [frame, corner, 3]")
    points = values.reshape(-1, 3).astype(np.float64)
    if extra_points is not None:
        extra = np.asarray(extra_points, dtype=np.float64)
        if extra.ndim != 2 or extra.shape[1] != 3 or not np.isfinite(extra).all():
            raise ValueError("extra_points must be finite [K, 3]")
        if len(extra):
            points = np.concatenate((points, extra))
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    target = (lower + upper) / 2
    radius = float(np.linalg.norm(upper - lower) / 2)
    distance = max(0.5, radius / math.sin(math.radians(fov) / 2) * 1.2)
    direction = _unit(np.array(_CAMERA_DIRECTION, dtype=np.float64))
    camera = target + distance * direction
    return {
        "position": camera.tolist(),
        "target": target.tolist(),
        "distance": distance,
        "fov_degrees": fov,
        "up_axis": "Y",
        "bounds_min": lower.tolist(),
        "bounds_max": upper.tolist(),
    }


def _look_at(position, target) -> tuple[float, float]:
    """Return the ViewerGL camera yaw and pitch [degrees] looking from ``position`` to ``target`` with +y up."""
    dx, dy, dz = (np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)).tolist()
    horizontal = math.hypot(dx, dz)
    yaw = math.degrees(math.atan2(dz, dx))
    pitch = math.degrees(math.atan2(dy, horizontal)) if horizontal > 1e-12 else (90.0 if dy > 0 else -90.0)
    return yaw, pitch


def _apply_camera(viewer, camera: dict) -> None:
    from pyglet.math import Vec3

    viewer.camera.pos = Vec3(*camera["position"])
    viewer.camera.yaw, viewer.camera.pitch = _look_at(camera["position"], camera["target"])
    viewer.camera.fov = camera["fov_degrees"]


def _check_unit_normals(normals: np.ndarray, name: str) -> None:
    if len(normals) and not np.allclose(np.linalg.norm(normals, axis=-1), 1.0, rtol=0, atol=_UNIT_NORMAL_TOLERANCE):
        raise ValueError(f"{name} must contain unit vectors")


def _load_contact(data) -> _ContactGeometry:
    """Read the optional contact partner arrays; trajectories without them are contact-free."""
    present = [key for key in _CONTACT_KEYS if key in data.files]
    if not present:
        return _ContactGeometry.empty()
    if len(present) != len(_CONTACT_KEYS):
        raise ValueError(f"trajectory is missing contact arrays {sorted(set(_CONTACT_KEYS) - set(present))}")
    flag = np.asarray(data["contact_plane_present"])
    if flag.size != 1 or flag.dtype != np.bool_:
        raise ValueError("contact_plane_present must be a single boolean")
    plane_present = bool(flag.reshape(-1)[0])
    plane_point = np.asarray(data["contact_plane_point"], dtype=np.float32)
    plane_normal = np.asarray(data["contact_plane_normal"], dtype=np.float32)
    positions = np.asarray(data["contact_point_positions"], dtype=np.float32)
    normals = np.asarray(data["contact_point_normals"], dtype=np.float32)
    radii = np.asarray(data["contact_point_radii"], dtype=np.float32)
    if plane_point.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError("contact_plane_point and contact_plane_normal must have shape [3]")
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or normals.shape != positions.shape
        or radii.shape != positions.shape[:1]
    ):
        raise ValueError("contact point arrays must be [N, 3], [N, 3] and [N]")
    for name, value in (
        ("contact_plane_point", plane_point),
        ("contact_plane_normal", plane_normal),
        ("contact_point_positions", positions),
        ("contact_point_normals", normals),
        ("contact_point_radii", radii),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if plane_present:
        _check_unit_normals(plane_normal[None], "contact_plane_normal")
    _check_unit_normals(normals, "contact_point_normals")
    if np.any(radii <= 0):
        raise ValueError("contact_point_radii must be positive")
    return _ContactGeometry(plane_present, plane_point, plane_normal, positions, normals, radii)


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
        contact = _load_contact(data)
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
    return positions, times, rest, fixed, cells, contact


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
    Static contact partners saved with the trajectory are drawn in every frame.
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
    positions, times, rest, fixed, cells, contact = _load_trajectory(trajectory)
    triangles, edges = _boundary_geometry(cells)
    plane_half_extent = None
    if contact.plane_present:
        plane_vertices, plane_triangles, plane_half_extent = _plane_geometry(positions, contact)
    camera = _camera_for_trajectory(positions, extra_points=_contact_framing_points(positions, contact))
    metadata_path = trajectory.with_name("report.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    capture_tools = Path(os.environ.get("AI_LOGS", "/home/horde/Code/AI-Docs/AI-Logs")) / "Newton" / "tools"
    sys.path.insert(0, str(capture_tools))
    from newton_capture import Capture  # noqa: PLC0415 - External capture helper.
    from newton_capture._video import VideoWriter  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    # Gravity acts along -y in the learned solver, so the viewer's up axis is +y.
    model = newton.ModelBuilder(up_axis=newton.Axis.Y).finalize(device="cpu")
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
    if contact.plane_present:
        plane_points = wp.array(plane_vertices, dtype=wp.vec3, device="cpu")
        plane_indices = wp.array(plane_triangles.ravel(), dtype=wp.int32, device="cpu")
        plane_normals = wp.array(
            np.repeat(_unit(contact.plane_normal)[None].astype(np.float32), 4, axis=0), dtype=wp.vec3, device="cpu"
        )
    if contact.point_count:
        static_points = wp.array(contact.point_positions, dtype=wp.vec3, device="cpu")
        static_radii = wp.array(contact.point_radii, dtype=wp.float32, device="cpu")
        # ViewerGL point colors must be a Warp array; a plain RGB tuple is rejected.
        static_colors = wp.full(contact.point_count, wp.vec3(*_POINT_COLOR), dtype=wp.vec3, device="cpu")
        # Normal ticks start on the sphere surface and reach 1.5 r_p beyond it so they are not hidden inside.
        tick_offsets = contact.point_radii[:, None] * contact.point_normals
        tick_starts = wp.array(contact.point_positions + tick_offsets, dtype=wp.vec3, device="cpu")
        tick_ends = wp.array(contact.point_positions + 2.5 * tick_offsets, dtype=wp.vec3, device="cpu")
    video = output / "simulation.mp4"
    frame_std_min = math.inf
    with (
        wp.ScopedDevice("cpu"),
        Capture(out_dir=str(output), width=width, height=height) as capture,
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
            _apply_camera(viewer, camera)
            viewer.begin_frame(float(times[index]))
            if contact.plane_present:
                viewer.log_mesh(
                    "contact_plane",
                    plane_points,
                    plane_indices,
                    normals=plane_normals,
                    color=_PLANE_COLOR,
                    roughness=0.95,
                    opacity=_PLANE_OPACITY,
                )
            if contact.point_count:
                viewer.log_points("contact_points", static_points, radii=static_radii, colors=static_colors)
                viewer.log_lines("contact_point_normals", tick_starts, tick_ends, _POINT_NORMAL_COLOR)
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
        "contact": {
            "plane_present": contact.plane_present,
            "plane_height": float(contact.plane_point[1]) if contact.plane_present else None,
            "plane_half_extent": plane_half_extent,
            "point_count": contact.point_count,
        },
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
