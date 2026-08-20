# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Deterministic hierarchy-driven random states for research previews."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .hierarchy import Hierarchy


@dataclasses.dataclass(frozen=True)
class HierarchyRandomStateConfig:
    """Scale-normalized bounds for hierarchy-driven random states.

    Fractions are relative to the referenced-vertex bounding-box diagonal.
    Rates are per second.
    Every field deliberately requires its exact built-in Python type so a
    serialized pilot configuration cannot silently change numeric semantics.
    """

    translation_fraction: float = 0.08
    rotation_radians: float = 0.20
    log_stretch: float = 0.10
    max_displacement_fraction: float = 0.18
    velocity_fraction_per_second: float = 0.25
    angular_velocity_radians_per_second: float = 0.75
    log_stretch_rate_per_second: float = 0.35
    max_velocity_fraction_per_second: float = 0.50
    level_decay: float = 0.70
    minimum_singular_value: float = 0.35
    validity_scale_decay: float = 0.50
    max_rescale_steps: int = 16

    def __post_init__(self) -> None:
        float_fields = (
            "translation_fraction",
            "rotation_radians",
            "log_stretch",
            "max_displacement_fraction",
            "velocity_fraction_per_second",
            "angular_velocity_radians_per_second",
            "log_stretch_rate_per_second",
            "max_velocity_fraction_per_second",
            "level_decay",
            "minimum_singular_value",
            "validity_scale_decay",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a built-in float")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        nonnegative_fields = (
            "translation_fraction",
            "rotation_radians",
            "log_stretch",
            "max_displacement_fraction",
            "velocity_fraction_per_second",
            "angular_velocity_radians_per_second",
            "log_stretch_rate_per_second",
            "max_velocity_fraction_per_second",
        )
        for name in nonnegative_fields:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 < self.level_decay <= 1.0:
            raise ValueError("level_decay must be in (0, 1]")
        if not 0.0 < self.minimum_singular_value <= 1.0:
            raise ValueError("minimum_singular_value must be in (0, 1]")
        if not 0.0 < self.validity_scale_decay < 1.0:
            raise ValueError("validity_scale_decay must be in (0, 1)")
        if type(self.max_rescale_steps) is not int:
            raise TypeError("max_rescale_steps must be a built-in int")
        if self.max_rescale_steps < 0:
            raise ValueError("max_rescale_steps must be nonnegative")


_DEFAULT_CONFIG = HierarchyRandomStateConfig()


@dataclasses.dataclass(frozen=True)
class HierarchyRandomState:
    """One validated random initial state and its scale-free diagnostics."""

    rest_positions: np.ndarray
    """Rest vertex positions [m], shape [vertex_count, 3]."""
    deformed_positions: np.ndarray
    """Deformed vertex positions [m], shape [vertex_count, 3]."""
    displacements: np.ndarray
    """Position displacement from rest [m], shape [vertex_count, 3]."""
    velocities: np.ndarray
    """Vertex velocities [m/s], shape [vertex_count, 3]."""
    deformation_seed: int
    velocity_seed: int
    characteristic_length: float
    """Referenced-vertex rest bounding-box diagonal [m]."""
    max_displacement_fraction: float
    max_centered_displacement_fraction: float
    """Maximum displacement after removing referenced-vertex mean translation, divided by characteristic length."""
    max_velocity_fraction_per_second: float
    minimum_determinant: float
    minimum_singular_value: float
    deformation_level_scales: tuple[float, ...]
    """Accepted per-level deformation multipliers, ordered fine to root."""
    deformation_scale: float
    """Final uniform cap and validity multiplier applied after per-level acceptance."""


def _require_seed(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_mesh(
    rest_positions: np.ndarray, tet_indices: np.ndarray, hierarchy: Hierarchy
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rest = np.asarray(rest_positions)
    tets = np.asarray(tet_indices)
    if rest.ndim != 2 or rest.shape[1:] != (3,) or rest.shape[0] == 0:
        raise ValueError("rest_positions must have shape [vertex_count, 3] with at least one vertex")
    if not np.issubdtype(rest.dtype, np.number) or np.issubdtype(rest.dtype, np.complexfloating):
        raise TypeError("rest_positions must be a real numeric array")
    rest = np.asarray(rest, dtype=np.float64)
    if not np.isfinite(rest).all():
        raise ValueError("rest_positions must be finite")
    if tets.ndim != 2 or tets.shape[1:] != (4,) or tets.shape[0] == 0:
        raise ValueError("tet_indices must have shape [tet_count, 4] with at least one tetrahedron")
    if not np.issubdtype(tets.dtype, np.integer) or np.issubdtype(tets.dtype, np.bool_):
        raise TypeError("tet_indices must contain integers")
    tets = np.asarray(tets, dtype=np.int64)
    if np.any(tets < 0) or np.any(tets >= rest.shape[0]):
        raise ValueError("tet_indices contain an out-of-range vertex index")
    if np.any(np.sort(tets, axis=1)[:, 1:] == np.sort(tets, axis=1)[:, :-1]):
        raise ValueError("each tetrahedron must reference four distinct vertices")
    if type(hierarchy) is not Hierarchy:
        raise TypeError("hierarchy must be exactly Hierarchy")

    corners = rest[tets]
    rest_matrices = np.stack(
        (
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 0],
            corners[:, 3] - corners[:, 0],
        ),
        axis=-1,
    )
    rest_determinants = np.linalg.det(rest_matrices)
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    squared_edge_lengths = np.stack(
        [np.sum((corners[:, a] - corners[:, b]) ** 2, axis=1) for a, b in edge_pairs], axis=1
    )
    local_length = np.sqrt(np.max(squared_edge_lengths, axis=1))
    referenced_positions = rest[np.unique(tets)]
    characteristic_length = float(np.linalg.norm(np.ptp(referenced_positions, axis=0)))
    determinant_floor = 64.0 * np.finfo(np.float64).eps * local_length**3
    if (
        not math.isfinite(characteristic_length)
        or characteristic_length <= 0.0
        or not np.isfinite(rest_determinants).all()
        or np.any(np.abs(rest_determinants) <= determinant_floor)
    ):
        raise ValueError("rest mesh must have finite positive edge scale and nondegenerate tetrahedra")

    tet_volumes = np.asarray(hierarchy.tet_vol)
    tet_centroids = np.asarray(hierarchy.tet_c0)
    expected_volumes = np.abs(rest_determinants) / 6.0
    expected_centroids = corners.mean(axis=1)
    if tet_volumes.shape != (tets.shape[0],) or tet_centroids.shape != (tets.shape[0], 3):
        raise ValueError("hierarchy tet arrays do not match tet_indices")
    if not np.isfinite(tet_volumes).all() or np.any(tet_volumes <= 0.0) or not np.isfinite(tet_centroids).all():
        raise ValueError("hierarchy tet volumes and centroids must be finite and positive")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if not np.allclose(tet_volumes, expected_volumes, rtol=tolerance, atol=tolerance * local_length**3):
        raise ValueError("hierarchy tet volumes do not match the rest mesh")
    if not np.allclose(tet_centroids, expected_centroids, rtol=tolerance, atol=tolerance * characteristic_length):
        raise ValueError("hierarchy tet centroids do not match the rest mesh")
    return rest, tets, corners, rest_matrices, characteristic_length


def _validate_pins(pinned_indices: np.ndarray | None, vertex_count: int) -> np.ndarray:
    if pinned_indices is None:
        return np.empty(0, dtype=np.int64)
    pins = np.asarray(pinned_indices)
    if pins.ndim != 1:
        raise ValueError("pinned_indices must be one-dimensional")
    if not np.issubdtype(pins.dtype, np.integer) or np.issubdtype(pins.dtype, np.bool_):
        raise TypeError("pinned_indices must contain integers")
    pins = np.asarray(pins, dtype=np.int64)
    if np.any(pins < 0) or np.any(pins >= vertex_count):
        raise ValueError("pinned_indices contain an out-of-range vertex index")
    return np.unique(pins)


def _hierarchy_levels(hierarchy: Hierarchy, tet_count: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    child_count = tet_count
    for level_index, level in enumerate(hierarchy.levels):
        assign = np.asarray(level.assign)
        centroids = np.asarray(level.c0)
        volumes = np.asarray(level.vol)
        pou_indices = np.asarray(level.pou_idx)
        pou_weights = np.asarray(level.pou_w)
        if assign.shape != (child_count,) or not np.issubdtype(assign.dtype, np.integer):
            raise ValueError(f"hierarchy level {level_index} assignment has an invalid shape or dtype")
        if centroids.ndim != 2 or centroids.shape[1:] != (3,) or centroids.shape[0] == 0:
            raise ValueError(f"hierarchy level {level_index} centroids have an invalid shape")
        if volumes.shape != (centroids.shape[0],):
            raise ValueError(f"hierarchy level {level_index} volumes have an invalid shape")
        if not np.isfinite(centroids).all() or not np.isfinite(volumes).all() or np.any(volumes <= 0.0):
            raise ValueError(f"hierarchy level {level_index} geometry must be finite and positive")
        assign = np.asarray(assign, dtype=np.int64)
        if np.any(assign < 0) or np.any(assign >= centroids.shape[0]):
            raise ValueError(f"hierarchy level {level_index} assignment contains an invalid node")
        if (
            pou_indices.ndim != 2
            or pou_indices.shape[0] != child_count
            or pou_indices.shape[1] == 0
            or pou_weights.shape != pou_indices.shape
            or not np.issubdtype(pou_indices.dtype, np.integer)
        ):
            raise ValueError(f"hierarchy level {level_index} prolongation arrays have invalid shapes or dtypes")
        pou_indices = np.asarray(pou_indices, dtype=np.int64)
        pou_weights = np.asarray(pou_weights, dtype=np.float64)
        valid = pou_indices >= 0
        if (
            np.any(pou_indices[valid] >= centroids.shape[0])
            or not np.isfinite(pou_weights).all()
            or np.any(pou_weights < 0.0)
            or np.any(pou_weights[~valid] != 0.0)
            or not np.allclose(pou_weights.sum(axis=1), 1.0, rtol=0.0, atol=64.0 * np.finfo(np.float64).eps)
        ):
            raise ValueError(f"hierarchy level {level_index} prolongation weights are invalid")
        levels.append((np.asarray(centroids, dtype=np.float64), pou_indices, pou_weights))
        child_count = centroids.shape[0]
    return levels


def _prolong_affine_to_tets(
    linear: np.ndarray,
    center_values: np.ndarray,
    tet_centroids: np.ndarray,
    hierarchy_levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    result_linear = linear
    result_center_values = center_values
    for level_index in range(source_level, -1, -1):
        parent_centroids, pou_indices, pou_weights = hierarchy_levels[level_index]
        child_centroids = tet_centroids if level_index == 0 else hierarchy_levels[level_index - 1][0]
        safe_indices = np.clip(pou_indices, 0, None)
        gathered_linear = result_linear[safe_indices]
        parent_to_child = child_centroids[:, None, :] - parent_centroids[safe_indices]
        gathered_center_values = result_center_values[safe_indices] + np.einsum(
            "npij,npj->npi", gathered_linear, parent_to_child
        )
        result_linear = (gathered_linear * pou_weights[:, :, None, None]).sum(axis=1)
        result_center_values = (gathered_center_values * pou_weights[:, :, None]).sum(axis=1)
    return result_linear, result_center_values


def _sample_vectors(rng: np.random.Generator, count: int, maximum_norm: float) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    norms = np.linalg.norm(vectors, axis=1)
    directions = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] > 0.0)
    radii = rng.random(count) ** (1.0 / 3.0)
    return directions * (maximum_norm * radii[:, None])


def _sample_symmetric_matrices(rng: np.random.Generator, count: int, maximum_spectral_norm: float) -> np.ndarray:
    raw = rng.normal(size=(count, 3, 3))
    symmetric = 0.5 * (raw + np.swapaxes(raw, 1, 2))
    spectral_norms = np.max(np.abs(np.linalg.eigvalsh(symmetric)), axis=1)
    directions = np.divide(
        symmetric,
        spectral_norms[:, None, None],
        out=np.zeros_like(symmetric),
        where=spectral_norms[:, None, None] > 0.0,
    )
    radii = rng.random(count)
    return directions * (maximum_spectral_norm * radii[:, None, None])


def _skew_matrices(vectors: np.ndarray) -> np.ndarray:
    skew = np.zeros((vectors.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def _rotation_matrices(axis_angles: np.ndarray) -> np.ndarray:
    count = axis_angles.shape[0]
    angles = np.linalg.norm(axis_angles, axis=1)
    axes = np.divide(axis_angles, angles[:, None], out=np.zeros_like(axis_angles), where=angles[:, None] > 0.0)
    skew = _skew_matrices(axes)
    identity = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    return identity + np.sin(angles)[:, None, None] * skew + (1.0 - np.cos(angles))[:, None, None] * (skew @ skew)


def _symmetric_matrix_exponential(matrices: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    with np.errstate(over="ignore", invalid="ignore"):
        exponentials = np.exp(eigenvalues)
    return np.einsum("nij,nj,nkj->nik", eigenvectors, exponentials, eigenvectors)


def _volume_average_to_vertices(
    tet_values: np.ndarray, tet_indices: np.ndarray, tet_volumes: np.ndarray, vertex_count: int
) -> np.ndarray:
    numerator = np.zeros((vertex_count, 3), dtype=np.float64)
    denominator = np.zeros(vertex_count, dtype=np.float64)
    for corner in range(4):
        indices = tet_indices[:, corner]
        np.add.at(numerator, indices, tet_volumes[:, None] * tet_values[:, corner])
        np.add.at(denominator, indices, tet_volumes)
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator[:, None], out=result, where=denominator[:, None] > 0.0)
    return result


def _level_weights(scale_count: int, decay: float) -> np.ndarray:
    # Coarse fields carry the largest share; normalization prevents hierarchy
    # depth from increasing the aggregate component bounds.
    weights = decay ** np.arange(scale_count - 1, -1, -1, dtype=np.float64)
    return weights / weights.sum()


def _sample_deformation(
    rng: np.random.Generator,
    rest_corners: np.ndarray,
    tet_indices: np.ndarray,
    tet_volumes: np.ndarray,
    tet_centroids: np.ndarray,
    hierarchy_levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    weights: np.ndarray,
    characteristic_length: float,
    config: HierarchyRandomStateConfig,
    vertex_count: int,
) -> np.ndarray:
    level_displacements = []
    scale_centroids = [tet_centroids, *(level[0] for level in hierarchy_levels)]
    for scale_index, (centroids, weight) in enumerate(zip(scale_centroids, weights, strict=True)):
        node_count = centroids.shape[0]
        translations = _sample_vectors(rng, node_count, config.translation_fraction * characteristic_length * weight)
        rotations = _rotation_matrices(_sample_vectors(rng, node_count, config.rotation_radians * weight))
        log_stretches = _sample_symmetric_matrices(rng, node_count, config.log_stretch * weight)
        transforms = rotations @ _symmetric_matrix_exponential(log_stretches)
        linear = transforms - np.eye(3)[None, :, :]
        if scale_index > 0:
            tet_linear, tet_center_displacement = _prolong_affine_to_tets(
                linear, translations, tet_centroids, hierarchy_levels, scale_index - 1
            )
        else:
            tet_linear = linear
            tet_center_displacement = translations
        relative = rest_corners - tet_centroids[:, None, :]
        tet_displacement = tet_center_displacement[:, None, :] + np.einsum("tij,tkj->tki", tet_linear, relative)
        level_displacements.append(
            _volume_average_to_vertices(tet_displacement, tet_indices, tet_volumes, vertex_count)
        )
    return np.stack(level_displacements, axis=0)


def _sample_velocity(
    rng: np.random.Generator,
    rest_corners: np.ndarray,
    tet_indices: np.ndarray,
    tet_volumes: np.ndarray,
    tet_centroids: np.ndarray,
    hierarchy_levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    weights: np.ndarray,
    characteristic_length: float,
    config: HierarchyRandomStateConfig,
    vertex_count: int,
) -> np.ndarray:
    velocity = np.zeros((vertex_count, 3), dtype=np.float64)
    scale_centroids = [tet_centroids, *(level[0] for level in hierarchy_levels)]
    for scale_index, (centroids, weight) in enumerate(zip(scale_centroids, weights, strict=True)):
        node_count = centroids.shape[0]
        translations = _sample_vectors(
            rng, node_count, config.velocity_fraction_per_second * characteristic_length * weight
        )
        angular = _sample_vectors(rng, node_count, config.angular_velocity_radians_per_second * weight)
        stretch_rates = _sample_symmetric_matrices(rng, node_count, config.log_stretch_rate_per_second * weight)
        linear = _skew_matrices(angular) + stretch_rates
        if scale_index > 0:
            tet_linear, tet_center_velocity = _prolong_affine_to_tets(
                linear, translations, tet_centroids, hierarchy_levels, scale_index - 1
            )
        else:
            tet_linear = linear
            tet_center_velocity = translations
        relative = rest_corners - tet_centroids[:, None, :]
        tet_velocity = tet_center_velocity[:, None, :] + np.einsum("tij,tkj->tki", tet_linear, relative)
        velocity += _volume_average_to_vertices(tet_velocity, tet_indices, tet_volumes, vertex_count)
    return velocity


def _cap_vector_field(values: np.ndarray, maximum_norm: float) -> tuple[np.ndarray, float]:
    observed = float(np.max(np.linalg.norm(values, axis=1), initial=0.0))
    if not math.isfinite(observed):
        raise RuntimeError("sampled hierarchy field is nonfinite")
    if observed == 0.0:
        return values.copy(), 1.0
    scale = min(1.0, maximum_norm / observed)
    return values * scale, scale


def _deformation_metrics(
    rest_matrices: np.ndarray, deformed_positions: np.ndarray, tet_indices: np.ndarray
) -> tuple[float, float] | None:
    if not np.isfinite(deformed_positions).all():
        return None
    corners = deformed_positions[tet_indices]
    deformed_matrices = np.stack(
        (
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 0],
            corners[:, 3] - corners[:, 0],
        ),
        axis=-1,
    )
    try:
        deformation_gradients = deformed_matrices @ np.linalg.inv(rest_matrices)
        determinants = np.linalg.det(deformation_gradients)
        singular_values = np.linalg.svd(deformation_gradients, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(determinants).all() or not np.isfinite(singular_values).all():
        return None
    return float(determinants.min()), float(singular_values.min())


def _read_only_float64(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def generate_hierarchy_random_state(
    rest_positions: np.ndarray,
    tet_indices: np.ndarray,
    hierarchy: Hierarchy,
    *,
    deformation_seed: int,
    velocity_seed: int,
    config: HierarchyRandomStateConfig = _DEFAULT_CONFIG,
    pinned_indices: np.ndarray | None = None,
) -> HierarchyRandomState:
    """Generate one bounded deformation and an independently seeded velocity.

    The fine tet level and every coarsened hierarchy level independently sample
    node translation, rotation, and symmetric log-stretch fields. Coarse affine
    coefficients and values at node centroids are recursively
    partition-of-unity prolonged to tetrahedra; all normalized scale
    contributions are rest-volume averaged onto vertices. Deformation levels
    are accepted from root to fine with deterministic per-level safety backoff,
    then summed and globally capped. Velocity sums all levels using independent
    translation, angular, and symmetric stretch-rate fields evaluated on rest
    geometry. Vertices not referenced by any tetrahedron remain at rest with
    zero velocity.

    Args:
        rest_positions: Rest vertex positions [m], shape [vertex_count, 3].
        tet_indices: Tetrahedron vertex indices, shape [tet_count, 4].
        hierarchy: Existing hierarchy built for this exact rest mesh.
        deformation_seed: Nonnegative seed for position deformation only.
        velocity_seed: Nonnegative seed for velocity only.
        config: Exact random-state configuration.
        pinned_indices: Optional pinned vertex indices. Pinned output positions
            equal rest exactly and pinned velocities are exactly zero.

    Returns:
        Frozen random state whose NumPy arrays are read-only.

    Raises:
        RuntimeError: If no deformation satisfying the configured quality
            threshold is found within the deterministic backoff budget.
    """
    if type(config) is not HierarchyRandomStateConfig:
        raise TypeError("config must be exactly HierarchyRandomStateConfig")
    _require_seed("deformation_seed", deformation_seed)
    _require_seed("velocity_seed", velocity_seed)
    rest, tets, rest_corners, rest_matrices, characteristic_length = _validate_mesh(
        rest_positions, tet_indices, hierarchy
    )
    pins = _validate_pins(pinned_indices, rest.shape[0])
    hierarchy_levels = _hierarchy_levels(hierarchy, tets.shape[0])
    weights = _level_weights(1 + len(hierarchy_levels), config.level_decay)
    tet_volumes = np.asarray(hierarchy.tet_vol, dtype=np.float64)
    tet_centroids = np.asarray(hierarchy.tet_c0, dtype=np.float64)

    deformation_rng = np.random.default_rng(deformation_seed)
    sampled_level_displacements = _sample_deformation(
        deformation_rng,
        rest_corners,
        tets,
        tet_volumes,
        tet_centroids,
        hierarchy_levels,
        weights,
        characteristic_length,
        config,
        rest.shape[0],
    )
    sampled_level_displacements[:, pins] = 0.0

    displacement = np.zeros_like(rest)
    deformation_level_scales = [1.0] * sampled_level_displacements.shape[0]
    for level_index in range(sampled_level_displacements.shape[0] - 1, -1, -1):
        level_scale = 1.0
        for _ in range(config.max_rescale_steps + 1):
            candidate_displacement = displacement + level_scale * sampled_level_displacements[level_index]
            candidate_displacement[pins] = 0.0
            candidate_deformed = rest + candidate_displacement
            candidate_deformed[pins] = rest[pins]
            metrics = _deformation_metrics(rest_matrices, candidate_deformed, tets)
            if metrics is not None:
                minimum_determinant, minimum_singular_value = metrics
                if minimum_determinant > 0.0 and minimum_singular_value >= config.minimum_singular_value:
                    displacement = candidate_displacement
                    displacement[pins] = 0.0
                    deformation_level_scales[level_index] = float(level_scale)
                    break
            level_scale *= config.validity_scale_decay
        else:
            raise RuntimeError(
                f"unable to generate a valid deformation: level {level_index} could not be accepted "
                "within max_rescale_steps "
                f"(required minimum_singular_value={config.minimum_singular_value})"
            )

    capped_displacement, cap_scale = _cap_vector_field(
        displacement, config.max_displacement_fraction * characteristic_length
    )
    accepted: tuple[np.ndarray, float, float, float] | None = None
    validity_scale = 1.0
    for _ in range(config.max_rescale_steps + 1):
        candidate_displacement = capped_displacement * validity_scale
        candidate_displacement[pins] = 0.0
        candidate_deformed = rest + candidate_displacement
        candidate_deformed[pins] = rest[pins]
        metrics = _deformation_metrics(rest_matrices, candidate_deformed, tets)
        if metrics is not None:
            minimum_determinant, minimum_singular_value = metrics
            if minimum_determinant > 0.0 and minimum_singular_value >= config.minimum_singular_value:
                accepted = (candidate_deformed, minimum_determinant, minimum_singular_value, validity_scale)
                break
        validity_scale *= config.validity_scale_decay
    if accepted is None:
        raise RuntimeError(
            "unable to generate a valid globally capped deformation within max_rescale_steps "
            f"(required minimum_singular_value={config.minimum_singular_value})"
        )
    deformed, minimum_determinant, minimum_singular_value, validity_scale = accepted
    displacement = deformed - rest
    displacement[pins] = 0.0

    velocity_rng = np.random.default_rng(velocity_seed)
    sampled_velocity = _sample_velocity(
        velocity_rng,
        rest_corners,
        tets,
        tet_volumes,
        tet_centroids,
        hierarchy_levels,
        weights,
        characteristic_length,
        config,
        rest.shape[0],
    )
    sampled_velocity[pins] = 0.0
    velocity, _ = _cap_vector_field(sampled_velocity, config.max_velocity_fraction_per_second * characteristic_length)
    velocity[pins] = 0.0
    if not np.isfinite(velocity).all():
        raise RuntimeError("generated velocity is nonfinite")

    maximum_displacement = float(np.max(np.linalg.norm(displacement, axis=1), initial=0.0))
    referenced_vertices = np.unique(tets)
    mean_displacement = displacement[referenced_vertices].mean(axis=0)
    maximum_centered_displacement = float(
        np.max(np.linalg.norm(displacement[referenced_vertices] - mean_displacement, axis=1), initial=0.0)
    )
    maximum_velocity = float(np.max(np.linalg.norm(velocity, axis=1), initial=0.0))
    return HierarchyRandomState(
        rest_positions=_read_only_float64(rest),
        deformed_positions=_read_only_float64(deformed),
        displacements=_read_only_float64(displacement),
        velocities=_read_only_float64(velocity),
        deformation_seed=deformation_seed,
        velocity_seed=velocity_seed,
        characteristic_length=characteristic_length,
        max_displacement_fraction=maximum_displacement / characteristic_length,
        max_centered_displacement_fraction=maximum_centered_displacement / characteristic_length,
        max_velocity_fraction_per_second=maximum_velocity / characteristic_length,
        minimum_determinant=minimum_determinant,
        minimum_singular_value=minimum_singular_value,
        deformation_level_scales=tuple(deformation_level_scales),
        deformation_scale=float(cap_scale * validity_scale),
    )
