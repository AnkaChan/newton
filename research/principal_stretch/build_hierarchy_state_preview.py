# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build a small static preview of hierarchy-aware random tet-mesh states.

The preview is intentionally an offline, low-complexity artifact: legacy ASCII
VTK meshes are parsed without an extra mesh dependency, Newton's existing
topological hierarchy is rebuilt, and Matplotlib writes static PNGs through its
headless Agg backend.  The generated configurations are initial states, not a
simulation, rollout, or claim about dynamics.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import io
import json
import math
import os
import pathlib
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .hierarchy import Hierarchy, build_hierarchy
from .hierarchy_random_state import (
    HierarchyRandomState,
    HierarchyRandomStateConfig,
    generate_hierarchy_random_state,
)

ASSET_BASENAMES = ("torus", "thin_sheet", "cube", "ditto", "bunny_small")
DEFAULT_ASSET_DIR = pathlib.Path("/home/horde/Code/Data/vtk")
DEFAULT_MAX_POINTS = 10_000
DEFAULT_MAX_TETS = 10_000
DEFAULT_BASE_SEED = 20_260_819

_MAX_RENDER_FACES = 30_000
_MAX_RENDER_EDGES = 25_000
_MAX_RENDER_NODES = 10_000
_MAX_RENDER_ARROWS = 72
_CAMERA_ELEVATION_DEGREES = 24.0
_CAMERA_AZIMUTH_DEGREES = -58.0
_NPY_VERSION = (2, 0)
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_STATE_FIGURE_HEADLINE = "Generated initial state \N{EM DASH} not a simulation"
_INITIAL_STATE_NOTICE = "These are procedurally generated initial states, not dynamics or simulated trajectories."


@dataclasses.dataclass(frozen=True, eq=False)
class LegacyVTKTetMesh:
    """Validated geometry from one legacy ASCII VTK unstructured grid."""

    rest_positions: np.ndarray
    """Referenced rest positions in undeclared asset units, shape ``(point_count, 3)``."""
    tet_indices: np.ndarray
    """Tetrahedron vertex indices, shape ``(tet_count, 4)``."""
    source_point_count: int
    """Point count declared by the source VTK file before compaction."""
    dropped_unused_point_count: int
    """Number of source points omitted because no tetrahedron references them."""
    source_sha256: str
    """SHA-256 of the exact VTK source bytes."""

    def __post_init__(self) -> None:
        positions = np.array(self.rest_positions, dtype=np.float64, order="C", copy=True)
        tets = np.array(self.tet_indices, dtype=np.int64, order="C", copy=True)
        if type(self.source_point_count) is not int or type(self.dropped_unused_point_count) is not int:
            raise TypeError("source point counts must be built-in integers")
        if self.dropped_unused_point_count < 0:
            raise ValueError("dropped_unused_point_count must be nonnegative")
        if self.source_point_count != positions.shape[0] + self.dropped_unused_point_count:
            raise ValueError("source point counts do not match the compacted position array")
        positions.setflags(write=False)
        tets.setflags(write=False)
        object.__setattr__(self, "rest_positions", positions)
        object.__setattr__(self, "tet_indices", tets)


class _TokenReader:
    """Small strict token reader for the geometry portion of legacy VTK."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def take(self, description: str) -> str:
        if self.position >= len(self.tokens):
            raise ValueError(f"truncated VTK file while reading {description}")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def keyword(self, expected: str) -> None:
        token = self.take(expected)
        if token.upper() != expected:
            raise ValueError(f"expected {expected} in VTK geometry, got {token!r}")

    def integer(self, description: str) -> int:
        token = self.take(description)
        try:
            return int(token, 10)
        except ValueError as exc:
            raise ValueError(f"invalid integer for {description}: {token!r}") from exc

    def real(self, description: str) -> float:
        token = self.take(description)
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"invalid real value for {description}: {token!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite real value for {description}: {token!r}")
        return value

    def remaining(self) -> list[str]:
        return self.tokens[self.position :]


def default_asset_paths(asset_dir: str | pathlib.Path | None = None) -> tuple[pathlib.Path, ...]:
    """Return the ordered five-asset pilot inventory.

    Args:
        asset_dir: Explicit directory override.  When omitted,
            ``PSS_VTK_ASSET_DIR`` is honored before the repository-independent
            default ``/home/horde/Code/Data/vtk``.

    Returns:
        Paths for torus, thin_sheet, cube, ditto, and bunny_small, in that
        order.
    """
    if asset_dir is None:
        root = pathlib.Path(os.environ.get("PSS_VTK_ASSET_DIR", DEFAULT_ASSET_DIR))
    else:
        root = pathlib.Path(asset_dir)
    return tuple(root / f"{name}.vtk" for name in ASSET_BASENAMES)


def _positive_cap(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_legacy_vtk_tet_mesh(
    path: str | pathlib.Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    max_tets: int = DEFAULT_MAX_TETS,
) -> LegacyVTKTetMesh:
    """Parse a legacy ASCII VTK tetrahedral unstructured grid.

    Only geometry is consumed.  Optional point/cell attribute sections after
    ``CELL_TYPES`` are ignored.  Binary input, non-tetrahedral cells, invalid
    indices, degenerate tetrahedra, malformed declarations, and meshes over
    the explicit caps are rejected before hierarchy construction.

    Args:
        path: Legacy ``.vtk`` source file.
        max_points: Maximum accepted declared point count.
        max_tets: Maximum accepted declared cell count.

    Returns:
        A validated immutable tetrahedral mesh.
    """
    max_points = _positive_cap(max_points, "max_points")
    max_tets = _positive_cap(max_tets, "max_tets")
    source_path = pathlib.Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_bytes = source_path.read_bytes()
    try:
        text = source_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source_path} is not a legacy ASCII VTK file") from exc
    lines = text.splitlines()
    if len(lines) < 4 or not lines[0].strip().lower().startswith("# vtk datafile version"):
        raise ValueError("missing legacy VTK version header")
    if lines[2].strip().upper() != "ASCII":
        raise ValueError("only legacy ASCII VTK files are supported")
    dataset_fields = lines[3].split()
    if len(dataset_fields) != 2 or [field.upper() for field in dataset_fields] != ["DATASET", "UNSTRUCTURED_GRID"]:
        raise ValueError("VTK DATASET must be UNSTRUCTURED_GRID")

    reader = _TokenReader(" ".join(lines[4:]).split())
    reader.keyword("POINTS")
    point_count = reader.integer("POINTS count")
    if point_count < 1:
        raise ValueError("POINTS count must be positive")
    if point_count > max_points:
        raise ValueError(f"POINTS count {point_count} exceeds cap {max_points}")
    point_scalar_type = reader.take("POINTS scalar type").lower()
    if point_scalar_type not in {
        "char",
        "double",
        "float",
        "int",
        "long",
        "short",
        "unsigned_char",
        "unsigned_int",
        "unsigned_long",
        "unsigned_short",
    }:
        raise ValueError(f"unsupported VTK POINTS scalar type {point_scalar_type!r}")
    positions = np.fromiter(
        (reader.real(f"POINTS coordinate {index}") for index in range(3 * point_count)),
        dtype=np.float64,
        count=3 * point_count,
    ).reshape(point_count, 3)

    reader.keyword("CELLS")
    tet_count = reader.integer("CELLS count")
    cell_word_count = reader.integer("CELLS integer count")
    if tet_count < 1:
        raise ValueError("CELLS count must be positive")
    if tet_count > max_tets:
        raise ValueError(f"CELLS count {tet_count} exceeds cap {max_tets}")
    expected_word_count = 5 * tet_count
    if cell_word_count < tet_count:
        raise ValueError("CELLS integer count is smaller than the number of cell-size declarations")
    tets = np.empty((tet_count, 4), dtype=np.int64)
    for cell_index in range(tet_count):
        arity = reader.integer(f"CELLS cell {cell_index} arity")
        if arity != 4:
            raise ValueError(f"non-tetrahedral cell {cell_index}: expected 4 vertices, got {arity}")
        for local_vertex in range(4):
            vertex_index = reader.integer(f"CELLS cell {cell_index} vertex {local_vertex}")
            if not 0 <= vertex_index < point_count:
                raise ValueError("tetrahedron vertex index is outside the POINTS range")
            tets[cell_index, local_vertex] = vertex_index
    if cell_word_count != expected_word_count:
        raise ValueError(f"tetrahedral CELLS integer count must be {expected_word_count}, got {cell_word_count}")

    reader.keyword("CELL_TYPES")
    cell_type_count = reader.integer("CELL_TYPES count")
    if cell_type_count != tet_count:
        raise ValueError(f"CELL_TYPES count {cell_type_count} does not match CELLS count {tet_count}")
    for cell_index in range(tet_count):
        cell_type = reader.integer(f"CELL_TYPES value {cell_index}")
        if cell_type != 10:
            raise ValueError(f"non-tetrahedral VTK cell type {cell_type} at cell {cell_index}; expected 10")
    remaining = reader.remaining()
    if remaining and remaining[0].upper() not in {"CELL_DATA", "FIELD", "METADATA", "POINT_DATA"}:
        raise ValueError(f"unexpected token after CELL_TYPES: {remaining[0]!r}")

    if np.any(np.diff(np.sort(tets, axis=1), axis=1) == 0):
        raise ValueError("tetrahedron contains a repeated vertex index")
    corners = positions[tets]
    edges = corners[:, 1:] - corners[:, :1]
    determinants = np.linalg.det(edges)
    roundoff = 64.0 * np.finfo(np.float64).eps * np.prod(np.linalg.norm(edges, axis=2), axis=1)
    degenerate = np.abs(determinants) <= roundoff
    if degenerate.any():
        first = int(np.flatnonzero(degenerate)[0])
        raise ValueError(f"degenerate tetrahedron at cell {first}")
    referenced_point_ids = np.unique(tets)
    source_to_compact = np.full(point_count, -1, dtype=np.int64)
    source_to_compact[referenced_point_ids] = np.arange(referenced_point_ids.size, dtype=np.int64)
    compact_tets = source_to_compact[tets]
    compact_positions = positions[referenced_point_ids]
    return LegacyVTKTetMesh(
        rest_positions=compact_positions,
        tet_indices=compact_tets,
        source_point_count=point_count,
        dropped_unused_point_count=point_count - int(referenced_point_ids.size),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def _load_matplotlib():
    """Load Matplotlib only when image generation is requested."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: PLC0415

    return matplotlib, plt, Line3DCollection, Poly3DCollection


def _even_sample(count: int, cap: int) -> np.ndarray:
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    return (np.arange(cap, dtype=np.int64) * count // cap).astype(np.int64)


def _boundary_faces(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_faces = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64)
    faces = tets[:, local_faces].reshape(-1, 3)
    canonical = np.sort(faces, axis=1)
    _, first, counts = np.unique(canonical, axis=0, return_index=True, return_counts=True)
    boundary_first = first[counts == 1]
    return faces[boundary_first], boundary_first // 4


def _graph_edges(adj: np.ndarray) -> np.ndarray:
    rows, slots = np.nonzero(adj >= 0)
    columns = adj[rows, slots]
    keep = rows < columns
    return np.stack((rows[keep], columns[keep]), axis=1)


def _fine_ancestor_assignments(hierarchy: Hierarchy) -> list[np.ndarray]:
    assignments = [np.arange(hierarchy.tet_c0.shape[0], dtype=np.int64)]
    current = assignments[0]
    for level in hierarchy.levels:
        current = level.assign[current]
        assignments.append(current)
    return assignments


def _level_geometry(hierarchy: Hierarchy) -> list[tuple[np.ndarray, np.ndarray]]:
    geometry = [(hierarchy.tet_c0, hierarchy.tet_adj)]
    geometry.extend((level.c0, level.adj) for level in hierarchy.levels)
    return geometry


def _common_bounds(*positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.concatenate(positions, axis=0)
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    center = 0.5 * (lower + upper)
    span = upper - lower
    largest = max(float(span.max()), 1.0e-12)
    padded_span = np.maximum(span, largest * 0.08) * 1.08
    return center - 0.5 * padded_span, center + 0.5 * padded_span, padded_span


def _set_camera(ax, bounds: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    lower, upper, span = bounds
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.set_box_aspect(tuple(float(value) for value in span))
    ax.view_init(elev=_CAMERA_ELEVATION_DEGREES, azim=_CAMERA_AZIMUTH_DEGREES)
    ax.set_axis_off()


def _cluster_colors(matplotlib, cluster_ids: np.ndarray, *, alpha: float) -> np.ndarray:
    phase = np.mod(np.asarray(cluster_ids, dtype=np.float64) * 0.6180339887498949, 1.0)
    colors = matplotlib.colormaps["turbo"](phase)
    colors[:, 3] = alpha
    return colors


def _depth_shaded_face_colors(
    matplotlib,
    triangles: np.ndarray,
    colors: np.ndarray | str,
    *,
    alpha: float | None,
) -> np.ndarray:
    """Apply deterministic normal- and view-depth shading to triangle colors."""
    if isinstance(colors, np.ndarray):
        face_colors = np.array(colors, dtype=np.float64, copy=True)
    else:
        face_colors = np.broadcast_to(matplotlib.colors.to_rgba(colors), (triangles.shape[0], 4)).copy()
    if face_colors.shape != (triangles.shape[0], 4):
        raise ValueError("surface colors must provide one RGBA row per rendered face")
    if alpha is not None:
        face_colors[:, 3] = alpha

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(
        normals,
        normal_lengths[:, None],
        out=np.zeros_like(normals),
        where=normal_lengths[:, None] > 0.0,
    )
    elevation = math.radians(_CAMERA_ELEVATION_DEGREES)
    azimuth = math.radians(_CAMERA_AZIMUTH_DEGREES)
    view_direction = np.array(
        [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)]
    )
    light_direction = view_direction + np.array([0.15, -0.10, 0.75])
    light_direction /= np.linalg.norm(light_direction)
    diffuse = np.abs(normals @ light_direction)
    depths = triangles.mean(axis=1) @ view_direction
    depth_range = float(np.ptp(depths))
    if depth_range > 0.0:
        depth = (depths - float(depths.min())) / depth_range
    else:
        depth = np.full(depths.shape, 0.5)
    brightness = 0.52 + 0.32 * diffuse + 0.16 * depth
    face_colors[:, :3] = np.clip(face_colors[:, :3] * brightness[:, None], 0.0, 1.0)
    return face_colors


def _add_surface(
    ax,
    matplotlib,
    Poly3DCollection,
    positions: np.ndarray,
    surface_faces: np.ndarray,
    *,
    colors: np.ndarray | str,
    alpha: float | None = None,
    edgecolor: str = "#182230",
    edge_alpha: float = 0.30,
    linewidth: float = 0.16,
) -> None:
    face_sample = _even_sample(surface_faces.shape[0], _MAX_RENDER_FACES)
    triangles = positions[surface_faces[face_sample]]
    sampled_colors = colors[face_sample] if isinstance(colors, np.ndarray) else colors
    face_colors = _depth_shaded_face_colors(matplotlib, triangles, sampled_colors, alpha=alpha)
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors=matplotlib.colors.to_rgba(edgecolor, edge_alpha),
        linewidths=linewidth,
        rasterized=True,
    )
    ax.add_collection3d(collection)


def _save_figure(fig, plt, destination: pathlib.Path) -> None:
    fig.savefig(
        destination,
        dpi=130,
        facecolor="white",
        metadata={"Software": "Newton hierarchy state preview"},
    )
    plt.close(fig)


def _render_hierarchy_image(
    destination: pathlib.Path,
    mesh: LegacyVTKTetMesh,
    hierarchy: Hierarchy,
) -> dict[str, object]:
    matplotlib, plt, Line3DCollection, Poly3DCollection = _load_matplotlib()
    surface_faces, surface_owners = _boundary_faces(mesh.tet_indices)
    fine_assignments = _fine_ancestor_assignments(hierarchy)
    levels = _level_geometry(hierarchy)
    bounds = _common_bounds(mesh.rest_positions)
    figure = plt.figure(figsize=(4.1 * len(levels), 4.2))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.78, wspace=0.02)
    rendered_levels: list[dict[str, int]] = []
    for level_index, ((centroids, adjacency), fine_assignment) in enumerate(zip(levels, fine_assignments, strict=True)):
        ax = figure.add_subplot(1, len(levels), level_index + 1, projection="3d")
        surface_cluster = fine_assignment[surface_owners]
        _add_surface(
            ax,
            matplotlib,
            Poly3DCollection,
            mesh.rest_positions,
            surface_faces,
            colors=_cluster_colors(matplotlib, surface_cluster, alpha=0.50),
        )
        edges = _graph_edges(adjacency)
        edge_sample = _even_sample(edges.shape[0], _MAX_RENDER_EDGES)
        if edge_sample.size:
            segments = centroids[edges[edge_sample]]
            ax.add_collection3d(
                Line3DCollection(segments, colors="#25364a", linewidths=0.55, alpha=0.62, rasterized=True)
            )
        node_sample = _even_sample(centroids.shape[0], _MAX_RENDER_NODES)
        node_colors = _cluster_colors(matplotlib, np.arange(centroids.shape[0]), alpha=0.92)
        ax.scatter(
            centroids[node_sample, 0],
            centroids[node_sample, 1],
            centroids[node_sample, 2],
            c=node_colors[node_sample],
            depthshade=False,
            s=4.0,
            alpha=0.82,
            rasterized=True,
        )
        ax.set_title(f"Level {level_index}\n{centroids.shape[0]:,} physical centroids", fontsize=10)
        _set_camera(ax, bounds)
        rendered_levels.append(
            {
                "level": level_index,
                "node_count": int(centroids.shape[0]),
                "edge_count": int(edges.shape[0]),
                "rendered_node_count": int(node_sample.size),
                "rendered_edge_count": int(edge_sample.size),
            }
        )
    figure.suptitle("Hierarchy: physical-centroid graphs + fine-tet ancestor colors", fontsize=11, y=0.98)
    _save_figure(figure, plt, destination)
    return {
        "surface_face_count": int(surface_faces.shape[0]),
        "rendered_surface_face_count": int(min(surface_faces.shape[0], _MAX_RENDER_FACES)),
        "levels": rendered_levels,
    }


def _state_arrays(
    mesh: LegacyVTKTetMesh,
    state: HierarchyRandomState,
) -> dict[str, np.ndarray]:
    """Extract the durable minimum state vocabulary from the core result."""
    return {
        "rest_positions": np.asarray(state.rest_positions, dtype=np.float64),
        "deformed_positions": np.asarray(state.deformed_positions, dtype=np.float64),
        "velocities": np.asarray(state.velocities, dtype=np.float64),
        "tet_indices": np.asarray(mesh.tet_indices, dtype=np.int64),
    }


def _state_metrics(
    state: HierarchyRandomState,
    arrays: Mapping[str, np.ndarray],
    config: HierarchyRandomStateConfig,
) -> dict[str, float]:
    characteristic_length = float(state.characteristic_length)
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("generated state characteristic_length must be finite and positive")
    displacement = arrays["deformed_positions"] - arrays["rest_positions"]
    max_displacement = float(np.linalg.norm(displacement, axis=1).max())
    max_speed = float(np.linalg.norm(arrays["velocities"], axis=1).max())
    return {
        "minimum_determinant": float(state.minimum_determinant),
        "minimum_singular_value": float(state.minimum_singular_value),
        "minimum_directional_stretch_safety_threshold": config.minimum_singular_value,
        "characteristic_length": characteristic_length,
        "deformation_scale": float(state.deformation_scale),
        "core_max_displacement_fraction": float(state.max_displacement_fraction),
        "max_centered_displacement_fraction": float(state.max_centered_displacement_fraction),
        "core_max_velocity_fraction_per_second": float(state.max_velocity_fraction_per_second),
        "max_displacement_asset_units": max_displacement,
        "max_displacement_over_characteristic_length": max_displacement / characteristic_length,
        "max_speed_asset_units_per_s": max_speed,
        "max_speed_over_characteristic_length_per_s": max_speed / characteristic_length,
    }


def _sparse_vectors(vectors: np.ndarray) -> np.ndarray:
    magnitudes = np.linalg.norm(vectors, axis=1)
    if not magnitudes.size:
        return np.empty(0, dtype=np.int64)
    threshold = max(float(magnitudes.max()) * 1.0e-10, np.finfo(np.float64).tiny)
    nonzero = np.flatnonzero(magnitudes > threshold)
    return nonzero[_even_sample(nonzero.size, _MAX_RENDER_ARROWS)]


def _scaled_velocity(velocities: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = float(np.linalg.norm(velocities, axis=1).max(initial=0.0))
    if maximum == 0.0:
        return np.zeros_like(velocities), 0.0
    extent = float(np.ptp(positions, axis=0).max())
    scale = 0.18 * max(extent, 1.0e-12) / maximum
    return velocities * scale, scale


def _quiver(ax, origins: np.ndarray, vectors: np.ndarray, indices: np.ndarray, color: str) -> None:
    if not indices.size:
        return
    p = origins[indices]
    v = vectors[indices]
    ax.quiver(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        v[:, 0],
        v[:, 1],
        v[:, 2],
        color=color,
        linewidth=1.05,
        arrow_length_ratio=0.24,
        normalize=False,
    )


def _render_state_image(
    destination: pathlib.Path,
    mesh: LegacyVTKTetMesh,
    state: HierarchyRandomState,
    *,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    matplotlib, plt, _, Poly3DCollection = _load_matplotlib()
    arrays = _state_arrays(mesh, state)
    rest = arrays["rest_positions"]
    deformed = arrays["deformed_positions"]
    velocities = arrays["velocities"]
    displacement = deformed - rest
    displacement_magnitude = np.linalg.norm(displacement, axis=1)
    velocity_display, velocity_display_scale = _scaled_velocity(velocities, deformed)
    displacement_indices = _sparse_vectors(displacement)
    velocity_indices = _sparse_vectors(velocities)
    bounds_inputs = [rest, deformed]
    if velocity_indices.size:
        bounds_inputs.append(deformed[velocity_indices] + velocity_display[velocity_indices])
    bounds = _common_bounds(*bounds_inputs)
    surface_faces, _ = _boundary_faces(mesh.tet_indices)
    figure = plt.figure(figsize=(16.0, 4.4))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.10, top=0.76, wspace=0.02)
    axes = [figure.add_subplot(1, 4, index + 1, projection="3d") for index in range(4)]

    _add_surface(axes[0], matplotlib, Poly3DCollection, rest, surface_faces, colors="#4f81bd", alpha=0.88)
    axes[0].set_title("Original (before)")
    heatmap_maximum = max(float(displacement_magnitude.max(initial=0.0)), np.finfo(np.float64).eps)
    heatmap_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=heatmap_maximum)
    heatmap = matplotlib.colormaps["magma"]
    face_displacement = displacement_magnitude[surface_faces].mean(axis=1)
    _add_surface(
        axes[1],
        matplotlib,
        Poly3DCollection,
        deformed,
        surface_faces,
        colors=heatmap(heatmap_norm(face_displacement)),
        alpha=0.94,
        edge_alpha=0.40,
        linewidth=0.20,
    )
    axes[1].set_title(
        "Deformed (after): exact geometry\ncolor = exact distance moved",
        fontsize=9,
    )
    colorbar_axis = figure.add_axes((0.555, 0.070, 0.14, 0.018))
    colorbar = matplotlib.colorbar.ColorbarBase(
        colorbar_axis, cmap=heatmap, norm=heatmap_norm, orientation="horizontal"
    )
    colorbar.set_label("distance moved (asset units)", fontsize=7, labelpad=1)
    colorbar.ax.tick_params(labelsize=6, length=2, pad=1)

    _add_surface(
        axes[2],
        matplotlib,
        Poly3DCollection,
        rest,
        surface_faces,
        colors="#a7a7a7",
        alpha=0.34,
        edge_alpha=0.22,
    )
    _add_surface(
        axes[2],
        matplotlib,
        Poly3DCollection,
        deformed,
        surface_faces,
        colors=heatmap(heatmap_norm(face_displacement)),
        alpha=0.52,
        edge_alpha=0.28,
    )
    _quiver(axes[2], rest, displacement, displacement_indices, "#d00000")
    axes[2].set_title(
        "Gray original + exact deformed heatmap\n(movement arrows at actual scale)",
        fontsize=8.5,
    )

    _add_surface(
        axes[3],
        matplotlib,
        Poly3DCollection,
        deformed,
        surface_faces,
        colors="#77aadd",
        alpha=0.84,
    )
    _quiver(axes[3], deformed, velocity_display, velocity_indices, "#006d2c")
    axes[3].set_title(
        "Exact deformed + independent velocity\n"
        "velocity arrows uniformly rescaled for visibility\n"
        "(directions and relative lengths retained)",
        fontsize=8,
    )
    for ax in axes:
        _set_camera(ax, bounds)
    figure.suptitle(
        f"{_STATE_FIGURE_HEADLINE}\n"
        f"asset size {metrics['characteristic_length']:.4g} · "
        f"minimum volume ratio {metrics['minimum_determinant']:.4g} · "
        f"minimum directional stretch {metrics['minimum_singular_value']:.4g}",
        fontsize=9,
        y=0.98,
    )
    _save_figure(figure, plt, destination)
    return {
        "displacement_arrow_count": int(displacement_indices.size),
        "velocity_arrow_count": int(velocity_indices.size),
        "velocity_display_scale_factor": velocity_display_scale,
        "surface_style": "per_face_depth_shading_with_visible_edges",
        "after_surface_color": "displacement_magnitude_heatmap",
        "after_geometry": "exact_deformed_positions",
        "headline": _STATE_FIGURE_HEADLINE,
        "after_heatmap_label": "distance moved (asset units)",
        "after_heatmap_max_displacement_asset_units": metrics["max_displacement_asset_units"],
    }


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("state arrays must not use object dtype")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError("state arrays must contain only finite values")
    dtype = array.dtype if array.dtype.byteorder == "|" else array.dtype.newbyteorder("<")
    return np.array(array, dtype=dtype, order="C", copy=True)


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, _canonical_array(value), version=_NPY_VERSION, allow_pickle=False)
    return stream.getvalue()


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": digest.hexdigest(),
    }


def _write_deterministic_npz(destination: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> str:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name in sorted(arrays):
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                        raise ValueError(f"noncanonical state array name {name!r}")
                    info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, _npy_bytes(arrays[name]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _file_sha256(destination)


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: pathlib.Path, kind: str) -> dict[str, object]:
    return {
        "path": path.name,
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, pathlib.Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _asset_seeds(base_seed: int, asset_name: str, source_sha256: str) -> tuple[int, int]:
    prefix = f"{base_seed}:{asset_name}:{source_sha256}"

    def derive(role: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{prefix}:{role}".encode("ascii")).digest()[:4], "little")

    return derive("deformation"), derive("velocity")


def _asset_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    if not slug:
        raise ValueError(f"asset name {name!r} does not yield a safe output filename")
    return slug


def _render_index(asset_records: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for asset in asset_records:
        name = html.escape(str(asset["name"]))
        outputs = asset["outputs"]
        hierarchy_ref = html.escape(str(outputs["hierarchy_png"]["path"]), quote=True)
        state_ref = html.escape(str(outputs["state_png"]["path"]), quote=True)
        npz_ref = html.escape(str(outputs["state_npz"]["path"]), quote=True)
        level_count = len(asset["hierarchy"]["levels"])
        metrics = asset["metrics"]
        dropped_point_count = int(asset["dropped_unused_point_count"])
        source_point_label = "point" if dropped_point_count == 1 else "points"
        maximum_movement_percent = 100.0 * float(metrics["max_displacement_over_characteristic_length"])
        maximum_speed_percent = 100.0 * float(metrics["max_speed_over_characteristic_length_per_s"])
        centered_movement_percent = 100.0 * float(metrics["max_centered_displacement_fraction"])
        retained_deformation_percent = 100.0 * float(metrics["deformation_scale"])
        if retained_deformation_percent < 100.0:
            deformation_explanation = (
                f"The sampled deformation was reduced to {retained_deformation_percent:.3g}% to keep tetrahedra valid."
            )
        else:
            deformation_explanation = (
                "The sampled deformation stayed at 100% strength. A reported 50% means the sampled deformation "
                "was reduced to 50% to keep tetrahedra valid."
            )
        sections.append(
            f"""    <section>
      <h2>{name}</h2>
      <p>{asset["point_count"]:,} active vertices ({dropped_point_count:,} unused source {source_point_label} dropped), {asset["tet_count"]:,} tetrahedra, {level_count} actual hierarchy levels.</p>
      <p>Asset size: {metrics["characteristic_length"]:.5g} asset units. <strong>Maximum movement:</strong> {metrics["max_displacement_asset_units"]:.5g} asset units ({maximum_movement_percent:.2f}% of asset size). <strong>Maximum speed:</strong> {metrics["max_speed_asset_units_per_s"]:.5g} asset units/s ({maximum_speed_percent:.2f}% of asset size per second).</p>
      <p><strong>Validity:</strong> the smallest local volume ratio is {metrics["minimum_determinant"]:.5g}. The smallest directional stretch is {metrics["minimum_singular_value"]:.5g} (safety threshold {metrics["minimum_directional_stretch_safety_threshold"]:.2g}). {deformation_explanation}</p>
      <details><summary>Reproducibility details</summary><p>Deformation seed {asset["deformation_seed"]}; velocity seed {asset["velocity_seed"]}. Maximum movement after removing overall mean translation: {centered_movement_percent:.2f}% of asset size. Exact arrays, hashes, and technical fields are in <a href="manifest.json">manifest.json</a>.</p></details>
      <div class="figures">
        <figure><p class="scroll-hint">Swipe horizontally to see all hierarchy levels &rarr;</p><div class="figure-scroll"><img class="hierarchy-strip" src="{hierarchy_ref}" alt="{name} hierarchy levels"></div><figcaption>Physical-centroid graph at every actual level; the surface is colored by each fine tetrahedron's ancestor cluster.</figcaption></figure>
        <figure><p class="scroll-hint">Swipe horizontally to see all four state panels &rarr;</p><div class="figure-scroll"><img class="state-strip" src="{state_ref}" alt="{name} generated initial state"></div><figcaption>Original and deformed views use identical camera and bounds, deterministic per-face diffuse/depth shading, and visible mesh edges. The deformed panel uses exact generated positions and a zero-anchored perceptual distance-moved heatmap. The third panel overlays a gray original ghost, exact deformed heatmap, and actual-scale movement arrows. Independent velocity: velocity arrows uniformly rescaled for visibility; directions and relative lengths retained.</figcaption></figure>
      </div>
      <p><a href="{npz_ref}">Download deterministic generated state (.npz)</a></p>
    </section>"""
        )
    sections_html = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hierarchy-aware initial-state preview</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font: 16px/1.45 system-ui, sans-serif; margin: auto; max-width: 1500px; padding: 2rem; color: #172033; }}
    .notice {{ background: #fff7d6; border-left: 5px solid #d49b00; padding: .8rem 1rem; }}
    .figures {{ display: grid; gap: 1.5rem; grid-template-columns: minmax(0, 1fr); }}
    .figure-scroll {{ max-width: 100%; overflow-x: auto; overscroll-behavior-inline: contain; }}
    .scroll-hint {{ display: none; margin: 0 0 .35rem; color: #526075; font-size: .9rem; }}
    figure {{ margin: 0; }} img {{ display: block; max-width: 100%; height: auto; }} figcaption {{ margin-top: .45rem; }}
    section {{ border-top: 1px solid #ccd3df; margin-top: 2rem; padding-top: 1rem; }}
    @media (max-width: 640px) {{
      body {{ padding: 1rem; }}
      .scroll-hint {{ display: block; }}
      .hierarchy-strip {{ width: auto; max-width: none; }}
      .state-strip {{ width: auto; max-width: none; }}
    }}
  </style>
</head>
<body>
  <h1>Hierarchy-aware generated initial states</h1>
  <p class="notice">{html.escape(_INITIAL_STATE_NOTICE)}</p>
  <p>Asset size means the diagonal of the original mesh bounding box. Percentages below use that common scale so differently sized assets can be compared.</p>
  <p>This static preview uses no JavaScript, video, or time integration. See <a href="manifest.json">manifest.json</a> for seeds, hashes, shapes, and dtypes.</p>
{sections_html}
</body>
</html>
"""


def build_preview(
    output_dir: str | pathlib.Path,
    *,
    asset_paths: Sequence[str | pathlib.Path] | None = None,
    asset_dir: str | pathlib.Path | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    n_levels: int = 5,
    cluster_size: int = 8,
    max_points: int = DEFAULT_MAX_POINTS,
    max_tets: int = DEFAULT_MAX_TETS,
    state_config: HierarchyRandomStateConfig | None = None,
) -> dict[str, object]:
    """Build all preview images, state archives, HTML, and the manifest.

    Args:
        output_dir: Destination directory for the complete static artifact.
        asset_paths: Explicit ordered VTK inputs, primarily for hermetic tests.
        asset_dir: Directory holding the exact five pilot basenames.  Mutually
            exclusive with ``asset_paths``.
        base_seed: Deterministic seed namespace for per-asset seeds.
        n_levels: Maximum number of hierarchy coarsening levels.
        cluster_size: Aggregation target passed to :func:`build_hierarchy`.
        max_points: Parser point-count cap.
        max_tets: Parser tetrahedron-count cap.
        state_config: Random-state generation configuration.

    Returns:
        The same JSON-compatible manifest written to ``manifest.json``.
    """
    if asset_paths is not None and asset_dir is not None:
        raise ValueError("asset_paths and asset_dir are mutually exclusive")
    if type(base_seed) is not int or not 0 <= base_seed < 2**32:
        raise ValueError("base_seed must be an integer in [0, 2**32)")
    n_levels = _positive_cap(n_levels, "n_levels")
    cluster_size = _positive_cap(cluster_size, "cluster_size")
    max_points = _positive_cap(max_points, "max_points")
    max_tets = _positive_cap(max_tets, "max_tets")
    sources = (
        tuple(pathlib.Path(path) for path in asset_paths) if asset_paths is not None else default_asset_paths(asset_dir)
    )
    if not sources:
        raise ValueError("at least one VTK asset is required")
    slugs = [_asset_slug(path.stem) for path in sources]
    if len(set(slugs)) != len(slugs):
        raise ValueError("asset basenames must map to unique output filenames")
    destination = pathlib.Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = HierarchyRandomStateConfig() if state_config is None else state_config
    if type(config) is not HierarchyRandomStateConfig:
        raise ValueError("state_config must be a HierarchyRandomStateConfig")

    asset_records: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for source, slug in zip(sources, slugs, strict=True):
        mesh = load_legacy_vtk_tet_mesh(source, max_points=max_points, max_tets=max_tets)
        hierarchy = build_hierarchy(
            mesh.tet_indices,
            mesh.rest_positions,
            n_levels=n_levels,
            target=cluster_size,
        )
        deformation_seed, velocity_seed = _asset_seeds(base_seed, source.stem, mesh.source_sha256)
        state = generate_hierarchy_random_state(
            mesh.rest_positions,
            mesh.tet_indices,
            hierarchy,
            deformation_seed=deformation_seed,
            velocity_seed=velocity_seed,
            config=config,
        )
        state_arrays = _state_arrays(mesh, state)
        metrics = _state_metrics(state, state_arrays, config)
        hierarchy_path = destination / f"{slug}_hierarchy.png"
        state_path = destination / f"{slug}_state.png"
        npz_path = destination / f"{slug}_state.npz"
        hierarchy_record = _render_hierarchy_image(hierarchy_path, mesh, hierarchy)
        state_figure_record = _render_state_image(
            state_path,
            mesh,
            state,
            metrics=metrics,
        )
        state_npz_sha256 = _write_deterministic_npz(npz_path, state_arrays)
        outputs = {
            "hierarchy_png": _file_record(hierarchy_path, "hierarchy_png"),
            "state_png": _file_record(state_path, "state_png"),
            "state_npz": {
                **_file_record(npz_path, "generated_initial_state_npz"),
                "sha256": state_npz_sha256,
                "arrays": {name: _array_record(value) for name, value in sorted(state_arrays.items())},
                "allow_pickle": False,
                "npy_version": list(_NPY_VERSION),
                "zip_timestamp": list(_ZIP_TIMESTAMP),
            },
        }
        artifacts.extend(outputs.values())
        asset_records.append(
            {
                "name": source.stem,
                "source_file": source.name,
                "source_sha256": mesh.source_sha256,
                "deformation_seed": deformation_seed,
                "velocity_seed": velocity_seed,
                "point_count": int(mesh.rest_positions.shape[0]),
                "source_point_count": mesh.source_point_count,
                "dropped_unused_point_count": mesh.dropped_unused_point_count,
                "tet_count": int(mesh.tet_indices.shape[0]),
                "state_kind": "generated_initial_state",
                "is_dynamics": False,
                "metrics": metrics,
                "hierarchy": hierarchy_record,
                "state_figure": state_figure_record,
                "outputs": outputs,
            }
        )

    index_path = destination / "index.html"
    index_path.write_text(_render_index(asset_records), encoding="utf-8", newline="\n")
    index_record = _file_record(index_path, "html_index")
    artifacts.append(index_record)
    manifest: dict[str, object] = {
        "schema": "newton-hierarchy-state-preview-v1",
        "notice": _INITIAL_STATE_NOTICE,
        "is_dynamics": False,
        "asset_basenames": [path.stem for path in sources],
        "base_seed": base_seed,
        "hierarchy_config": {"n_levels": n_levels, "cluster_size": cluster_size},
        "state_config": _json_value(config),
        "caps": {"max_points": max_points, "max_tets": max_tets},
        "assets": asset_records,
        "artifact_inventory": sorted(artifacts, key=lambda record: str(record["path"])),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path, help="destination for the static preview")
    parser.add_argument(
        "--asset-dir",
        type=pathlib.Path,
        help="directory holding the five .vtk assets (or set PSS_VTK_ASSET_DIR)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="deterministic base seed")
    parser.add_argument("--n-levels", type=int, default=5, help="maximum coarsening levels")
    parser.add_argument("--cluster-size", type=int, default=8, help="hierarchy aggregation target")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="per-asset vertex cap")
    parser.add_argument("--max-tets", type=int, default=DEFAULT_MAX_TETS, help="per-asset tetrahedron cap")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    build_preview(
        args.output_dir,
        asset_dir=args.asset_dir,
        base_seed=args.seed,
        n_levels=args.n_levels,
        cluster_size=args.cluster_size,
        max_points=args.max_points,
        max_tets=args.max_tets,
    )
    print(f"Wrote static hierarchy-state preview to {args.output_dir}")
    print(_INITIAL_STATE_NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
