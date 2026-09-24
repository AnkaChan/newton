# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental compatible random displacements on an automatic grid hierarchy.

The canonical rest geometry is unchanged. Every level spans its exact bounds,
uses an independent seeded stream, and fixes the entire material z-min plane.
Finite orientation checks do not certify global injectivity or no self-contact.
"""

from dataclasses import dataclass

import numpy as np

from .data import VoxelGridData

__all__ = [
    "DeformationLevel",
    "MultiscaleSample",
    "build_hierarchy",
    "generate_multiscale",
    "interpolate_control_grid",
    "screen_geometry",
]


@dataclass(frozen=True)
class DeformationLevel:
    """Experimental control-grid level; dimensions count points, not cells."""

    name: str
    """Display name of this level."""
    control_counts: tuple[int, int, int]
    """Control point counts including both bounds along every axis."""
    amplitude_m: float
    """Uniform per-component displacement half-width [m], before backtracking."""


@dataclass(frozen=True)
class MultiscaleSample:
    """Experimental deterministic sample with auditable level contributions."""

    positions: np.ndarray
    """Current shared corner positions [m], shape [corner_count, 3]."""
    levels: tuple[DeformationLevel, ...]
    """Coarse-to-fine hierarchy and requested amplitudes."""
    controls: tuple[np.ndarray, ...]
    """Unscaled random control vectors [m], each shape [nx, ny, nz, 3]."""
    level_displacements: np.ndarray
    """Unscaled interpolated fields [m], shape [level_count, corner_count, 3]."""
    effective_scale: float
    """Common factor applied to every level after orientation screening."""
    backtracking_steps: int
    """Number of deterministic halvings of the requested amplitudes."""
    screen: dict[str, float]
    """Combined geometry's sampled dimensionless volume-ratio minima."""
    level_screens: tuple[dict[str, float], ...]
    """Equivalent screening for each level alone at the same effective scale."""


def build_hierarchy(
    rest: VoxelGridData, *, strength: float = 0.5, max_levels: int = 3, coarse_max_intervals: int = 4
) -> tuple[DeformationLevel, ...]:
    """Derive an experimental hierarchy automatically from the canonical grid.

    Start at the real voxel spacing and double the target spacing until no axis
    has more than ``coarse_max_intervals`` control intervals. Every grid spans
    the exact rest bounds; ceil(extent/spacing) intervals need not divide the
    voxel counts. If capped, retain evenly spaced hierarchy indices including
    the finest and coarsest. One requested level retains the finest.

    Args:
        rest: Canonical cuboid including dimensions and rest voxel size [m].
        strength: Total per-component amplitude budget as a fraction of the
            shortest rest side. Split among levels with weights spacing**1.5.
        max_levels: Positive maximum number of retained levels.
        coarse_max_intervals: Positive stopping threshold on the longest axis.

    Returns:
        Coarse-to-fine control grids. Requested amplitudes sum to
        strength * shortest rest side, independent of the number of levels.
    """
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    if not isinstance(max_levels, int) or max_levels < 1:
        raise ValueError("max_levels must be a positive integer")
    if not isinstance(coarse_max_intervals, int) or coarse_max_intervals < 1:
        raise ValueError("coarse_max_intervals must be a positive integer")
    counts = np.asarray(rest.cell_counts)
    extent = counts * rest.cell_size
    candidates = [(counts.copy(), rest.cell_size)]
    factor = 1
    while candidates[-1][0].max() > coarse_max_intervals:
        factor *= 2
        intervals = (counts + factor - 1) // factor
        candidates.append((intervals, factor * rest.cell_size))
    take = np.unique(np.linspace(0, len(candidates) - 1, min(max_levels, len(candidates))).round().astype(int))
    chosen = [candidates[index] for index in reversed(take)]
    weights = np.array([spacing for _, spacing in chosen]) ** 1.5
    amplitudes = strength * extent.min() * weights / weights.sum()
    if len(chosen) == 1:
        names = ["fine"]
    elif len(chosen) == 2:
        names = ["coarse", "fine"]
    elif len(chosen) == 3:
        names = ["coarse", "middle", "fine"]
    else:
        names = ["coarse", *[f"middle-{index}" for index in range(1, len(chosen) - 1)], "fine"]
    return tuple(
        DeformationLevel(name, tuple(int(n) + 1 for n in intervals), float(amplitude))
        for name, (intervals, _), amplitude in zip(names, chosen, amplitudes, strict=True)
    )


def interpolate_control_grid(
    controls: np.ndarray, positions: np.ndarray, *, origin: np.ndarray, extent: np.ndarray
) -> np.ndarray:
    """Trilinearly interpolate experimental control vectors in rest coordinates.

    Args:
        controls: Control displacement vectors [m], shape [nx, ny, nz, 3].
        positions: Query rest positions [m], shape [point_count, 3].
        origin: Minimum rest corner [m], shape [3].
        extent: Positive physical side lengths [m], shape [3].

    Returns:
        Interpolated displacement vectors [m], shape [point_count, 3].
    """
    controls = np.asarray(controls, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    if controls.ndim != 4 or controls.shape[-1] != 3 or min(controls.shape[:3]) < 2:
        raise ValueError("controls must have shape [nx, ny, nz, 3] with at least two points per axis")
    if positions.ndim != 2 or positions.shape[1] != 3 or origin.shape != (3,) or extent.shape != (3,):
        raise ValueError("positions, origin or extent have incorrect shapes")
    if not all(np.isfinite(array).all() for array in (controls, positions, origin, extent)) or (extent <= 0).any():
        raise ValueError("inputs must be finite and extents positive")
    normalized = (positions - origin) / extent
    if (normalized < -1e-12).any() or (normalized > 1 + 1e-12).any():
        raise ValueError("query positions must lie inside the rest bounds")
    intervals = np.array(controls.shape[:3]) - 1
    coordinate = np.clip(normalized, 0, 1) * intervals
    lower = np.minimum(np.floor(coordinate).astype(int), intervals - 1)
    fraction = coordinate - lower
    result = np.zeros_like(positions)
    for bits in np.ndindex(2, 2, 2):
        offset = np.array(bits)
        weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
        index = lower + offset
        result += weight[:, None] * controls[index[:, 0], index[:, 1], index[:, 2]]
    return result


class _GeometryScreen:
    """Cache rest shape derivatives and Newton's alternating five-tet topology."""

    def __init__(self, rest: VoxelGridData):
        self.rest = rest
        bits = np.indices((2, 2, 2)).reshape(3, -1).T
        gauss = (1 + np.array([-1, 1]) / np.sqrt(3)) / 2
        points = np.concatenate((bits, np.array(list(np.ndindex(2, 2, 2))), [[0.5, 0.5, 0.5]])).astype(float)
        points[8:16] = gauss[points[8:16].astype(int)]
        self.derivatives = np.empty((len(points), 8, 3))
        for axis in range(3):
            other = [index for index in range(3) if index != axis]
            weights = np.where(bits[None, :, other], points[:, None, other], 1 - points[:, None, other])
            self.derivatives[:, :, axis] = (2 * bits[:, axis] - 1) * weights.prod(axis=-1) / rest.cell_size
        # ModelBuilder.add_soft_grid uses v0..v7 in this material-corner order.
        local = np.array([0, 4, 5, 1, 2, 6, 7, 3])
        patterns = local[
            np.array(
                [
                    [[1, 2, 5, 0], [3, 0, 7, 2], [4, 7, 0, 5], [6, 5, 2, 7], [5, 2, 7, 0]],
                    [[0, 1, 4, 3], [2, 3, 6, 1], [5, 4, 1, 6], [7, 6, 3, 4], [4, 1, 6, 3]],
                ]
            )
        ]
        parity = np.indices(rest.cell_counts).reshape(3, -1).sum(axis=0) % 2
        self.tets = rest.cell_corner_indices[np.arange(len(parity))[:, None, None], patterns[parity]]
        self.rest_tet_determinants = self._tet_determinants(rest.corner_rest_positions)

    def _tet_determinants(self, positions: np.ndarray) -> np.ndarray:
        corners = positions[self.tets]
        return np.linalg.det((corners[:, :, 1:] - corners[:, :, :1]).transpose(0, 1, 3, 2))

    def check(self, positions: np.ndarray) -> dict[str, float]:
        if positions.shape != self.rest.corner_rest_positions.shape or not np.isfinite(positions).all():
            raise ValueError("positions must be finite with the same shape as the rest corners")
        corners = positions[self.rest.cell_corner_indices]
        relative = corners - corners[:, :1]
        gradients = np.einsum("nki,qkj->nqij", relative, self.derivatives)
        return {
            "min_tet_volume_ratio": float(np.min(self._tet_determinants(positions) / self.rest_tet_determinants)),
            "min_sampled_jacobian": float(np.linalg.det(gradients).min()),
        }


def screen_geometry(rest: VoxelGridData, positions: np.ndarray) -> dict[str, float]:
    """Screen experimental geometry beyond cell centers.

    Args:
        rest: Canonical cubic grid and unchanged rest positions [m].
        positions: Current shared positions [m], shape [corner_count, 3].

    Returns:
        Minimum Newton five-tet signed volume ratio and minimum trilinear
        Jacobian determinant at all eight corners, eight two-point Gauss
        locations and the cell center. This does not establish continuous
        injectivity or exclude self-intersections of distant cells.
    """
    return _GeometryScreen(rest).check(np.asarray(positions, dtype=np.float64))


def generate_multiscale(
    rest: VoxelGridData,
    *,
    seed: int,
    strength: float = 0.5,
    max_levels: int = 3,
    levels: tuple[DeformationLevel, ...] | None = None,
    min_volume_ratio: float = 0.2,
) -> MultiscaleSample:
    """Generate an experimental compatible random shape with a fixed z-min end.

    Args:
        rest: Unmodified canonical rest grid.
        seed: Nonnegative integer seed; each level gets a separate PCG64 stream.
        strength: Amplitude budget used by the automatic hierarchy.
        max_levels: Maximum automatic hierarchy depth, including both endpoints.
        levels: Optional explicit hierarchy override for experiments.
        min_volume_ratio: Positive sampled orientation margin, below one.

    Returns:
        Compatible shared positions, controls and per-level fields. Sample every
        control-vector component uniformly in [-amplitude_m, amplitude_m], then
        zero the entire z-min control plane at each level. Halve a common scale
        until the combined shape AND each contribution alone meet the margin.
        Neither resampling nor changes to canonical rest geometry are used.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(min_volume_ratio) or not 0 < min_volume_ratio < 1:
        raise ValueError("min_volume_ratio must lie strictly between zero and one")
    levels = build_hierarchy(rest, strength=strength, max_levels=max_levels) if levels is None else tuple(levels)
    if not levels or len({level.name for level in levels}) != len(levels):
        raise ValueError("levels must be nonempty and have unique names")
    origin = rest.corner_rest_positions.min(axis=0)
    extent = np.array(rest.cell_counts) * rest.cell_size
    controls, fields = [], []
    for index, level in enumerate(levels):
        if len(level.control_counts) != 3 or any(not isinstance(n, int) or n < 2 for n in level.control_counts):
            raise ValueError("control counts must be three integers of at least two")
        if not np.isfinite(level.amplitude_m) or level.amplitude_m < 0:
            raise ValueError("level amplitudes must be finite and nonnegative")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(index,))))
        vectors = rng.uniform(-level.amplitude_m, level.amplitude_m, (*level.control_counts, 3))
        vectors[:, :, 0] = 0
        controls.append(vectors)
        fields.append(interpolate_control_grid(vectors, rest.corner_rest_positions, origin=origin, extent=extent))
    fields = np.stack(fields)
    total = fields.sum(axis=0)
    checker = _GeometryScreen(rest)
    scale = 1.0
    for backtracking_steps in range(25):
        positions = rest.corner_rest_positions + scale * total
        combined = checker.check(positions)
        individual = tuple(checker.check(rest.corner_rest_positions + scale * field) for field in fields)
        if min(value for screen in (combined, *individual) for value in screen.values()) >= min_volume_ratio:
            return MultiscaleSample(
                positions, levels, tuple(controls), fields, scale, backtracking_steps, combined, individual
            )
        scale *= 0.5
    raise ValueError("Could not produce a valid shape within 24 deterministic amplitude halvings")
