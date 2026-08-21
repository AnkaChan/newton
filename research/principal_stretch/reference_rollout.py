# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Deterministic free-body reference rollouts for hierarchy-random states.

This module turns the existing procedural ``(x_0, v_0)`` previews into
contact-free, undamped physical sequences.  SolverVBD produces candidates in
its native float32 arithmetic.  Durable states are lossless float64 promotions
of those exact float32 values so the existing common-objective evaluator and
v5 data path see one unambiguous numerical state.

The registered pilot is deliberately narrow: unit bounding-box diagonal,
zero gravity and load, no pins, three fresh-restart SolverVBD budgets, and an
independent stable-Neo-Hookean acceptance gate.  A failed gate remains useful
diagnostic evidence but is never written as a training shard.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import re
import shutil
import struct
import tempfile
import types
import zipfile
from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .build_hierarchy_state_preview import (
    DEFAULT_BASE_SEED,
    DEFAULT_MAX_POINTS,
    DEFAULT_MAX_TETS,
    _sample_seeds,
    default_asset_paths,
    load_legacy_vtk_tet_mesh,
)
from .hierarchy import Hierarchy, build_hierarchy
from .hierarchy_random_state import HierarchyRandomStateConfig, generate_hierarchy_random_state
from .solver_benchmark import (
    CommonStateMetrics,
    TetBenchmarkScene,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
    scene_from_model,
)

_SCHEMA = "pss-free-body-reference-shard-v1"
_INDEX_SCHEMA = "pss-free-body-reference-index-v1"
_REQUESTED_DT_SECONDS = 1.0 / 300.0
_EXECUTION_DT_SECONDS = float(np.float32(_REQUESTED_DT_SECONDS))
_EXECUTION_DT_FLOAT32_BITS = "0x3b5a740e"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_NPY_VERSION = (2, 0)
_ACTIVE_FLAG = int(newton.ParticleFlags.ACTIVE)


def _readonly_array(value, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    # ``np.ascontiguousarray`` promotes a scalar from shape ``()`` to ``(1,)``.
    # Preserve the producer schema's scalar dt and seeds while still owning a
    # C-order, little-endian copy for every non-scalar payload.
    return np.array(array, dtype=canonical_dtype, order="C", copy=True)


def _array_digest(value: np.ndarray) -> str:
    """Hash an array together with its canonical dtype and shape."""
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    # ``tobytes`` also handles arrays with a zero-sized dimension, which is
    # required for the explicit free-body pin inventories.
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = _canonical_array(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _array_digest(array),
    }


def _canonical_json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return types.MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _verified_self_hash(value: Mapping[str, object], hash_field: str, name: str) -> Mapping[str, object]:
    record = _thaw_json(value)
    if not isinstance(record, dict):
        raise TypeError(f"{name} must be a JSON object")
    declared = record.pop(hash_field, None)
    _require_sha256(declared, f"{name}.{hash_field}")
    if _canonical_json_digest(record) != declared:
        raise ValueError(f"{name} self-hash does not match {hash_field}")
    record[hash_field] = declared
    return _freeze_json(record)


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _float32_bits(value: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', np.float32(value)))[0]:08x}"


def _require_sha256(value: str, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_seed(value: int, name: str) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32)")
    return value


def _identifier(value: str, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _asset_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    if not slug:
        raise ValueError(f"asset id {value!r} does not yield a safe path component")
    return slug


def _logical_source_name(path: str | pathlib.Path) -> str:
    """Return the path-independent logical name recorded in canonical evidence."""
    source_name = pathlib.Path(path).name
    return _identifier(source_name, "source name")


@dataclasses.dataclass(frozen=True)
class FreeBodyReferenceProtocol:
    """Registered physical and numerical policy for the first rollout pilot."""

    requested_dt_seconds: float = _REQUESTED_DT_SECONDS
    normalized_characteristic_length_m: float = 1.0
    density_kg_m3: float = 1000.0
    shear_modulus_pa: float = 1.0e4
    linear_lame_lambda_pa: float = 1.0e5
    tet_damping: float = 0.0
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    iteration_budgets: tuple[int, ...] = (20, 50, 100)
    rollout_steps: int = 8
    maximum_relative_residual: float = 2.0e-2
    maximum_residual_ratio: float = 1.0e-1
    tile_solve: bool = True

    def __post_init__(self) -> None:
        if type(self.requested_dt_seconds) is not float or self.requested_dt_seconds != _REQUESTED_DT_SECONDS:
            raise ValueError("requested_dt_seconds must be exactly 1/300 second for the registered pilot")
        for name in (
            "normalized_characteristic_length_m",
            "density_kg_m3",
            "shear_modulus_pa",
            "linear_lame_lambda_pa",
            "maximum_relative_residual",
            "maximum_residual_ratio",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite built-in float")
        if type(self.tet_damping) is not float or self.tet_damping != 0.0:
            raise ValueError("tet_damping must be exactly zero for the common objective")
        if (
            type(self.gravity_m_s2) is not tuple
            or len(self.gravity_m_s2) != 3
            or any(type(value) is not float for value in self.gravity_m_s2)
            or self.gravity_m_s2 != (0.0, 0.0, 0.0)
        ):
            raise ValueError("the first elastic+velocity pilot requires exactly zero gravity")
        if type(self.iteration_budgets) is not tuple or not self.iteration_budgets:
            raise ValueError("iteration_budgets must be a non-empty tuple")
        if any(type(value) is not int or value < 1 for value in self.iteration_budgets):
            raise ValueError("iteration_budgets must contain positive built-in integers")
        if any(left >= right for left, right in zip(self.iteration_budgets, self.iteration_budgets[1:], strict=False)):
            raise ValueError("iteration_budgets must be strictly increasing")
        if type(self.rollout_steps) is not int or self.rollout_steps < 1:
            raise ValueError("rollout_steps must be a positive built-in integer")
        if type(self.tile_solve) is not bool:
            raise ValueError("tile_solve must be a bool")
        if self.execution_dt_float32_bits != _EXECUTION_DT_FLOAT32_BITS:
            raise RuntimeError("registered float32 timestep bits changed unexpectedly")

    @property
    def execution_dt_seconds(self) -> float:
        """Exact promoted float32 timestep consumed by SolverVBD."""
        return float(np.float32(self.requested_dt_seconds))

    @property
    def execution_dt_float32_bits(self) -> str:
        """IEEE-754 binary32 identity of :attr:`execution_dt_seconds`."""
        return _float32_bits(self.execution_dt_seconds)

    @property
    def vbd_stored_lambda_pa(self) -> float:
        """Stable-NH lambda stored by this Newton revision [Pa]."""
        return self.shear_modulus_pa + self.linear_lame_lambda_pa

    def as_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible registered policy."""
        return {
            "contract": "pss-free-body-reference-protocol-v1",
            "requested_dt_seconds": self.requested_dt_seconds,
            "requested_dt_expression": "1/300",
            "execution_dt_seconds": self.execution_dt_seconds,
            "execution_dt_float32_bits": self.execution_dt_float32_bits,
            "producer_numeric_policy": "SolverVBD-float32-output-losslessly-promoted-to-float64",
            "normalized_characteristic_length_m": self.normalized_characteristic_length_m,
            "normalization_center": "referenced-rest-vertex-arithmetic-mean",
            "density_kg_m3": self.density_kg_m3,
            "shear_modulus_pa": self.shear_modulus_pa,
            "linear_lame_lambda_pa": self.linear_lame_lambda_pa,
            "vbd_stored_lambda_pa": self.vbd_stored_lambda_pa,
            "lambda_convention": "vbd_stored_lambda=mu+linear_lame_lambda",
            "tet_damping": self.tet_damping,
            "gravity_m_s2": list(self.gravity_m_s2),
            "external_force": "exactly-zero",
            "boundary_condition": "free-body-no-pins",
            "translation_gauge_pin": False,
            "contact": False,
            "self_contact": False,
            "iteration_budgets": list(self.iteration_budgets),
            "selected_iterations": self.iteration_budgets[-1],
            "budget_restart_policy": "fresh-identical-input-state",
            "rollout_steps": self.rollout_steps,
            "maximum_relative_residual": self.maximum_relative_residual,
            "maximum_residual_ratio": self.maximum_residual_ratio,
            "acceptance_gate": (
                "finite-and-uninverted; min-singular>0; objective<=iterate-zero; "
                "relative-residual<iterate-zero-and<=registered-absolute-and-ratio-thresholds; "
                "exact-VBD-velocity-commit"
            ),
            "tile_solve_requested": self.tile_solve,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class TetOrientationRepair:
    """Positive rest orientation and the exact tetrahedra that were repaired."""

    tet_indices: np.ndarray
    repaired_tet_indices: tuple[int, ...]
    minimum_absolute_determinant: float

    def __post_init__(self) -> None:
        tets = _readonly_array(self.tet_indices, np.int64, "tet_indices")
        if tets.ndim != 2 or tets.shape[1:] != (4,):
            raise ValueError("tet_indices must have shape (T, 4)")
        object.__setattr__(self, "tet_indices", tets)
        object.__setattr__(self, "repaired_tet_indices", tuple(int(value) for value in self.repaired_tet_indices))
        if not math.isfinite(self.minimum_absolute_determinant) or self.minimum_absolute_determinant <= 0.0:
            raise ValueError("minimum_absolute_determinant must be finite and positive")


def repair_tet_orientation(rest_positions: np.ndarray, tet_indices: np.ndarray) -> TetOrientationRepair:
    """Swap two local corners of every negatively oriented rest tetrahedron.

    Degenerate elements are rejected rather than repaired.  Vertex numbering,
    tetrahedron ordering, and positively oriented rows remain unchanged.
    """
    rest_input = np.asarray(rest_positions)
    tets_input = np.asarray(tet_indices)
    if rest_input.ndim != 2 or rest_input.shape[1:] != (3,) or rest_input.shape[0] == 0:
        raise ValueError("rest_positions must have shape (V, 3) with at least one vertex")
    if not np.issubdtype(rest_input.dtype, np.number) or np.issubdtype(rest_input.dtype, np.complexfloating):
        raise TypeError("rest_positions must be a real numeric array")
    if tets_input.ndim != 2 or tets_input.shape[1:] != (4,) or tets_input.shape[0] == 0:
        raise ValueError("tet_indices must have shape (T, 4) with at least one tetrahedron")
    if not np.issubdtype(tets_input.dtype, np.integer) or np.issubdtype(tets_input.dtype, np.bool_):
        raise TypeError("tet_indices must contain integers")
    rest = np.asarray(rest_input, dtype=np.float64)
    tets = np.array(tets_input, dtype=np.int64, order="C", copy=True)
    if not np.isfinite(rest).all():
        raise ValueError("rest_positions must be finite")
    if np.any(tets < 0) or np.any(tets >= rest.shape[0]):
        raise ValueError("tet_indices contain an out-of-range vertex")
    if np.any(np.diff(np.sort(tets, axis=1), axis=1) == 0):
        raise ValueError("tetrahedron contains a repeated vertex")

    corners = rest[tets]
    edges = corners[:, 1:] - corners[:, :1]
    determinants = np.linalg.det(np.swapaxes(edges, 1, 2))
    edge_scale = np.prod(np.linalg.norm(edges, axis=2), axis=1)
    floor = 64.0 * np.finfo(np.float64).eps * edge_scale
    degenerate = ~np.isfinite(determinants) | (np.abs(determinants) <= floor)
    if np.any(degenerate):
        first = int(np.flatnonzero(degenerate)[0])
        raise ValueError(f"degenerate tetrahedron at index {first}")

    repaired = np.flatnonzero(determinants < 0.0)
    if repaired.size:
        old_corner_two = tets[repaired, 2].copy()
        tets[repaired, 2] = tets[repaired, 3]
        tets[repaired, 3] = old_corner_two
    repaired_corners = rest[tets]
    repaired_edges = repaired_corners[:, 1:] - repaired_corners[:, :1]
    repaired_determinants = np.linalg.det(np.swapaxes(repaired_edges, 1, 2))
    if np.any(repaired_determinants <= 0.0) or not np.isfinite(repaired_determinants).all():
        raise RuntimeError("orientation repair did not produce positive rest tetrahedra")
    return TetOrientationRepair(
        tet_indices=tets,
        repaired_tet_indices=tuple(int(value) for value in repaired),
        minimum_absolute_determinant=float(np.abs(determinants).min()),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class NormalizedInitialState:
    """One float32-staged random initial state in canonical metre units."""

    rest_q: np.ndarray
    x_initial: np.ndarray
    velocity_initial: np.ndarray
    tet_indices: np.ndarray
    source_center: np.ndarray
    source_characteristic_length: float
    normalized_characteristic_length_m: float
    orientation_repaired_count: int

    def __post_init__(self) -> None:
        rest = _readonly_array(self.rest_q, np.float32, "rest_q")
        positions = _readonly_array(self.x_initial, np.float32, "x_initial")
        velocities = _readonly_array(self.velocity_initial, np.float32, "velocity_initial")
        tets = _readonly_array(self.tet_indices, np.int32, "tet_indices")
        center = _readonly_array(self.source_center, np.float64, "source_center")
        if rest.ndim != 2 or rest.shape[1:] != (3,) or positions.shape != rest.shape or velocities.shape != rest.shape:
            raise ValueError("rest_q, x_initial, and velocity_initial must share shape (V, 3)")
        if tets.ndim != 2 or tets.shape[1:] != (4,):
            raise ValueError("tet_indices must have shape (T, 4)")
        if center.shape != (3,):
            raise ValueError("source_center must have shape (3,)")
        if not math.isfinite(self.source_characteristic_length) or self.source_characteristic_length <= 0.0:
            raise ValueError("source_characteristic_length must be finite and positive")
        if not math.isfinite(self.normalized_characteristic_length_m) or self.normalized_characteristic_length_m <= 0.0:
            raise ValueError("normalized_characteristic_length_m must be finite and positive")
        if type(self.orientation_repaired_count) is not int or self.orientation_repaired_count < 0:
            raise ValueError("orientation_repaired_count must be a nonnegative built-in integer")
        object.__setattr__(self, "rest_q", rest)
        object.__setattr__(self, "x_initial", positions)
        object.__setattr__(self, "velocity_initial", velocities)
        object.__setattr__(self, "tet_indices", tets)
        object.__setattr__(self, "source_center", center)


def normalize_initial_state(
    rest_positions: np.ndarray,
    deformed_positions: np.ndarray,
    velocities: np.ndarray,
    tet_indices: np.ndarray,
    *,
    normalized_characteristic_length_m: float = 1.0,
) -> NormalizedInitialState:
    """Center and scale one hierarchy-random state, then stage it in float32."""
    if (
        type(normalized_characteristic_length_m) is not float
        or not math.isfinite(normalized_characteristic_length_m)
        or normalized_characteristic_length_m <= 0.0
    ):
        raise ValueError("normalized_characteristic_length_m must be a positive finite built-in float")
    rest = np.asarray(rest_positions, dtype=np.float64)
    deformed = np.asarray(deformed_positions, dtype=np.float64)
    velocity = np.asarray(velocities, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1:] != (3,) or rest.shape[0] == 0:
        raise ValueError("rest_positions must have shape (V, 3) with at least one vertex")
    if deformed.shape != rest.shape or velocity.shape != rest.shape:
        raise ValueError("deformed_positions and velocities must match rest_positions")
    if not np.isfinite(rest).all() or not np.isfinite(deformed).all() or not np.isfinite(velocity).all():
        raise ValueError("initial-state arrays must be finite")
    orientation = repair_tet_orientation(rest, tet_indices)
    referenced = np.unique(orientation.tet_indices)
    center = rest[referenced].mean(axis=0)
    source_length = float(np.linalg.norm(np.ptp(rest[referenced], axis=0)))
    if not math.isfinite(source_length) or source_length <= 0.0:
        raise ValueError("referenced rest vertices must have a positive bounding-box diagonal")
    scale = normalized_characteristic_length_m / source_length
    rest_normalized = (rest - center) * scale
    deformed_normalized = (deformed - center) * scale
    velocity_normalized = velocity * scale

    # Builder geometry is float32, so fail before physics if staging collapses
    # an otherwise valid element.
    staged_orientation = repair_tet_orientation(
        np.asarray(rest_normalized, dtype=np.float32),
        orientation.tet_indices,
    )
    if staged_orientation.repaired_tet_indices:
        raise RuntimeError("float32 staging unexpectedly changed repaired tet orientation")
    return NormalizedInitialState(
        rest_q=rest_normalized,
        x_initial=deformed_normalized,
        velocity_initial=velocity_normalized,
        tet_indices=staged_orientation.tet_indices,
        source_center=center,
        source_characteristic_length=source_length,
        normalized_characteristic_length_m=normalized_characteristic_length_m,
        orientation_repaired_count=len(orientation.repaired_tet_indices),
    )


def _hierarchy_array_mapping(hierarchy: Hierarchy | None) -> Mapping[str, np.ndarray]:
    if hierarchy is None:
        return {}
    if type(hierarchy) is not Hierarchy:
        raise TypeError("hierarchy must be exactly Hierarchy")
    arrays: dict[str, np.ndarray] = {
        "hierarchy_tet_adj": _readonly_array(hierarchy.tet_adj, np.int32, "hierarchy_tet_adj"),
        "hierarchy_tet_c0": _readonly_array(hierarchy.tet_c0, np.float64, "hierarchy_tet_c0"),
        "hierarchy_tet_vol": _readonly_array(hierarchy.tet_vol, np.float64, "hierarchy_tet_vol"),
    }
    for level_index, level in enumerate(hierarchy.levels):
        prefix = f"hierarchy_level_{level_index}"
        arrays[f"{prefix}_adj"] = _readonly_array(level.adj, np.int32, f"{prefix}_adj")
        arrays[f"{prefix}_assign"] = _readonly_array(level.assign, np.int32, f"{prefix}_assign")
        arrays[f"{prefix}_c0"] = _readonly_array(level.c0, np.float64, f"{prefix}_c0")
        arrays[f"{prefix}_pou_idx"] = _readonly_array(level.pou_idx, np.int32, f"{prefix}_pou_idx")
        arrays[f"{prefix}_pou_w"] = _readonly_array(level.pou_w, np.float64, f"{prefix}_pou_w")
        arrays[f"{prefix}_vol"] = _readonly_array(level.vol, np.float64, f"{prefix}_vol")
    return arrays


@dataclasses.dataclass(frozen=True, eq=False)
class FreeBodyReferenceScene:
    """Reusable Newton model plus the immutable first-step scene contract."""

    model: newton.Model
    template: TetBenchmarkScene
    initial_state: NormalizedInitialState
    protocol: FreeBodyReferenceProtocol
    asset_id: str
    source: str
    source_sha256: str
    deformation_seed: int
    velocity_seed: int
    hierarchy_arrays: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.asset_id, "asset_id")
        _identifier(self.source, "source")
        if self.source != _logical_source_name(self.source):
            raise ValueError("source must be a logical basename, never a machine-local path")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_seed(self.deformation_seed, "deformation_seed")
        _require_seed(self.velocity_seed, "velocity_seed")
        if type(self.protocol) is not FreeBodyReferenceProtocol:
            raise TypeError("protocol must be exactly FreeBodyReferenceProtocol")
        if type(self.initial_state) is not NormalizedInitialState:
            raise TypeError("initial_state must be exactly NormalizedInitialState")
        copied = {
            name: _readonly_array(value, np.asarray(value).dtype, name)
            for name, value in sorted(dict(self.hierarchy_arrays).items())
        }
        object.__setattr__(self, "hierarchy_arrays", types.MappingProxyType(copied))

    def static_arrays(self) -> dict[str, np.ndarray]:
        """Return the exact static model arrays shared by every transition."""
        scene = self.template
        arrays = {
            "boundary_triangles": np.asarray(scene.tri_indices, dtype=np.int32),
            "color_group_offsets": np.asarray(scene.color_group_offsets, dtype=np.int32),
            "color_group_particles": np.asarray(scene.color_group_particles, dtype=np.int32),
            "mass": np.asarray(scene.mass, dtype=np.float32),
            "particle_flags": np.asarray(scene.particle_flags, dtype=np.int32),
            "particle_inv_mass": np.asarray(scene.particle_inv_mass, dtype=np.float32),
            "rest_q": np.asarray(scene.rest_q, dtype=np.float64),
            "tet_indices": np.asarray(scene.tet_indices, dtype=np.int32),
            "tet_materials": np.asarray(scene.tet_materials, dtype=np.float32),
            "tet_poses": np.asarray(scene.tet_poses, dtype=np.float64),
        }
        arrays.update(self.hierarchy_arrays)
        return arrays

    def physical_identities(self) -> dict[str, object]:
        """Return stable identities for topology, operators, material, and protocol."""
        arrays = self.static_arrays()
        topology_names = ("boundary_triangles", "rest_q", "tet_indices")
        material_names = ("tet_materials",)
        operator_names = tuple(sorted(name for name in arrays if name not in material_names))

        def inventory(names: Sequence[str]) -> dict[str, object]:
            return {name: _array_record(arrays[name]) for name in names}

        topology_inventory = inventory(topology_names)
        topology_sha256 = _canonical_json_digest(topology_inventory)
        material_payload = {
            "arrays": inventory(material_names),
            "density_kg_m3": self.protocol.density_kg_m3,
            "linear_lame_lambda_pa": self.protocol.linear_lame_lambda_pa,
            "shear_modulus_pa": self.protocol.shear_modulus_pa,
            "tet_damping": self.protocol.tet_damping,
            "vbd_stored_lambda_pa": self.protocol.vbd_stored_lambda_pa,
        }
        operator_payload = {
            "arrays": inventory(operator_names),
            "topology_sha256": topology_sha256,
        }
        return {
            "contract": "pss-free-body-physical-identities-v1",
            "material_arrays": list(material_names),
            "material_sha256": _canonical_json_digest(material_payload),
            "operator_arrays": list(operator_names),
            "operator_sha256": _canonical_json_digest(operator_payload),
            "protocol_sha256": _canonical_json_digest(self.protocol.as_dict()),
            "topology_arrays": list(topology_names),
            "topology_sha256": topology_sha256,
        }


def build_free_body_scene(
    initial_state: NormalizedInitialState,
    *,
    protocol: FreeBodyReferenceProtocol,
    device: str,
    asset_id: str,
    source: str,
    source_sha256: str,
    deformation_seed: int,
    velocity_seed: int,
    hierarchy: Hierarchy | None = None,
) -> FreeBodyReferenceScene:
    """Build one pin-free, contact-free stable-NH Newton model."""
    if type(initial_state) is not NormalizedInitialState:
        raise TypeError("initial_state must be exactly NormalizedInitialState")
    if type(protocol) is not FreeBodyReferenceProtocol:
        raise TypeError("protocol must be exactly FreeBodyReferenceProtocol")
    if not device:
        raise ValueError("device must not be empty")
    _identifier(asset_id, "asset_id")
    _identifier(source, "source")
    _require_sha256(source_sha256, "source_sha256")
    _require_seed(deformation_seed, "deformation_seed")
    _require_seed(velocity_seed, "velocity_seed")
    if initial_state.normalized_characteristic_length_m != protocol.normalized_characteristic_length_m:
        raise ValueError("initial-state scale does not match the reference protocol")

    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=initial_state.rest_q,
        indices=initial_state.tet_indices.reshape(-1),
        density=protocol.density_kg_m3,
        k_mu=protocol.shear_modulus_pa,
        k_lambda=protocol.vbd_stored_lambda_pa,
        k_damp=protocol.tet_damping,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        add_surface_mesh_edges=False,
        particle_radius=0.0,
    )
    if len(builder.tet_indices) != initial_state.tet_indices.shape[0]:
        raise RuntimeError("Newton builder did not retain every positively oriented tetrahedron")
    if any(mass <= 0.0 for mass in builder.particle_mass):
        raise RuntimeError("free-body model contains a non-positive particle mass")
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity(protocol.gravity_m_s2)
    force = np.zeros_like(initial_state.x_initial, dtype=np.float32)
    template = scene_from_model(
        model,
        name=f"{asset_id}-free-body-reference",
        source=source,
        dt=protocol.execution_dt_seconds,
        x_current=initial_state.x_initial,
        velocity=initial_state.velocity_initial,
        external_force=force,
        metadata={
            "protocol": protocol.as_dict(),
            "source_sha256": source_sha256,
            "deformation_seed": deformation_seed,
            "velocity_seed": velocity_seed,
            "source_characteristic_length": initial_state.source_characteristic_length,
            "orientation_repaired_count": initial_state.orientation_repaired_count,
        },
    )
    if template.pinned_indices.size or template.pin_targets.shape != (0, 3):
        raise RuntimeError("free-body reference construction introduced physical pins")
    if np.any(template.mass <= 0.0) or np.any((template.particle_flags & _ACTIVE_FLAG) == 0):
        raise RuntimeError("free-body reference requires positive mass and ACTIVE flags at every vertex")
    return FreeBodyReferenceScene(
        model=model,
        template=template,
        initial_state=initial_state,
        protocol=protocol,
        asset_id=asset_id,
        source=source,
        source_sha256=source_sha256,
        deformation_seed=deformation_seed,
        velocity_seed=velocity_seed,
        hierarchy_arrays=_hierarchy_array_mapping(hierarchy),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class _CandidateState:
    """One exact SolverVBD float32 output promoted losslessly to float64."""

    positions: np.ndarray
    velocities: np.ndarray
    effective_tile_solve: bool
    position_float32_sha256: str = dataclasses.field(init=False)
    velocity_float32_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        positions32 = np.array(self.positions, dtype=np.float32, order="C", copy=True)
        velocities32 = np.array(self.velocities, dtype=np.float32, order="C", copy=True)
        if positions32.ndim != 2 or positions32.shape[1:] != (3,) or velocities32.shape != positions32.shape:
            raise ValueError("candidate positions and velocities must share shape (V, 3)")
        if not np.isfinite(positions32).all() or not np.isfinite(velocities32).all():
            raise ValueError("candidate state must be finite in SolverVBD float32")
        if type(self.effective_tile_solve) is not bool:
            raise ValueError("effective_tile_solve must be a bool")
        object.__setattr__(self, "positions", _readonly_array(positions32, np.float64, "candidate positions"))
        object.__setattr__(self, "velocities", _readonly_array(velocities32, np.float64, "candidate velocities"))
        object.__setattr__(self, "position_float32_sha256", _array_digest(positions32))
        object.__setattr__(self, "velocity_float32_sha256", _array_digest(velocities32))


class _ReusableVBDRunner:
    """One model/solver allocation reused across fresh candidate restarts."""

    def __init__(self, scene: FreeBodyReferenceScene) -> None:
        self.scene = scene
        self.solver = SolverVBD(
            model=scene.model,
            iterations=scene.protocol.iteration_budgets[-1],
            particle_enable_self_contact=False,
            particle_enable_tile_solve=scene.protocol.tile_solve,
        )
        self.state_in = scene.model.state()
        self.state_out = scene.model.state()
        self.control = scene.model.control()

    def solve(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        external_force: np.ndarray,
        iterations: int,
    ) -> _CandidateState:
        """Restart the same implicit objective and run exactly ``iterations`` sweeps."""
        if iterations not in self.scene.protocol.iteration_budgets:
            raise ValueError("iterations is not registered by the reference protocol")
        positions64 = np.asarray(positions, dtype=np.float64)
        velocities64 = np.asarray(velocities, dtype=np.float64)
        force32 = np.asarray(external_force, dtype=np.float32)
        positions32 = np.asarray(positions64, dtype=np.float32)
        velocities32 = np.asarray(velocities64, dtype=np.float32)
        if not np.array_equal(positions64, positions32.astype(np.float64)):
            raise ValueError("input positions are not a lossless promotion of staged float32")
        if not np.array_equal(velocities64, velocities32.astype(np.float64)):
            raise ValueError("input velocities are not a lossless promotion of staged float32")
        if force32.shape != positions32.shape or not np.isfinite(force32).all():
            raise ValueError("external_force must be finite with shape (V, 3)")

        # SolverVBD mutates state_in.particle_q during its sweeps.  Restoring all
        # three arrays here is what makes every budget a genuine fresh restart.
        self.state_in.particle_q.assign(positions32)
        self.state_in.particle_qd.assign(velocities32)
        self.state_in.clear_forces()
        self.state_in.particle_f.assign(force32)
        self.solver.iterations = iterations
        self.solver.step(
            self.state_in,
            self.state_out,
            self.control,
            None,
            self.scene.protocol.execution_dt_seconds,
        )
        output_positions = self.state_out.particle_q.numpy()
        output_velocities = self.state_out.particle_qd.numpy()
        return _CandidateState(
            positions=output_positions,
            velocities=output_velocities,
            effective_tile_solve=bool(self.scene.protocol.tile_solve and self.scene.model.device.is_cuda),
        )


def _metrics_record(metrics: CommonStateMetrics) -> dict[str, object]:
    return dataclasses.asdict(metrics)


@dataclasses.dataclass(frozen=True)
class ReferenceCandidateEvidence:
    """Independent common-objective evidence for one fresh VBD budget."""

    iterations: int
    effective_tile_solve: bool
    metrics: CommonStateMetrics
    position_float32_sha256: str
    velocity_float32_sha256: str
    velocity_float64_sha256: str
    displacement_from_previous_budget_m: float | None
    relative_residual_over_iterate_zero: float

    def as_dict(self) -> dict[str, object]:
        """Return deterministic evidence without wall-clock timings."""
        return {
            "iterations": self.iterations,
            "fresh_restart": True,
            "effective_tile_solve": self.effective_tile_solve,
            "position_float32_sha256": self.position_float32_sha256,
            "position_float64_sha256": self.metrics.position_sha256,
            "velocity_float32_sha256": self.velocity_float32_sha256,
            "velocity_float64_sha256": self.velocity_float64_sha256,
            "displacement_from_previous_budget_m": self.displacement_from_previous_budget_m,
            "relative_residual_over_iterate_zero": self.relative_residual_over_iterate_zero,
            "metrics": _metrics_record(self.metrics),
        }


@dataclasses.dataclass(frozen=True)
class ReferenceStepEvidence:
    """Immutable acceptance evidence for one physical transition."""

    step_id: int
    input_position_sha256: str
    input_velocity_sha256: str
    inertial_target_sha256: str
    dynamic_scene_manifest: Mapping[str, object]
    objective_manifest: Mapping[str, object]
    dynamic_scene_sha256: str
    objective_instance_sha256: str
    iterate_zero_metrics: CommonStateMetrics
    candidates: tuple[ReferenceCandidateEvidence, ...]
    selected_iterations: int
    output_position_sha256: str
    output_velocity_sha256: str
    exact_velocity_commit: bool
    reference_accepted: bool
    reference_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a nonnegative built-in integer")
        for name in (
            "input_position_sha256",
            "input_velocity_sha256",
            "inertial_target_sha256",
            "dynamic_scene_sha256",
            "objective_instance_sha256",
            "output_position_sha256",
            "output_velocity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        scene_manifest = _verified_self_hash(
            self.dynamic_scene_manifest,
            "scene_sha256",
            "dynamic_scene_manifest",
        )
        objective_manifest = _verified_self_hash(
            self.objective_manifest,
            "objective_instance_sha256",
            "objective_manifest",
        )
        if scene_manifest["scene_sha256"] != self.dynamic_scene_sha256:
            raise ValueError("dynamic scene identity does not match its manifest")
        if objective_manifest["objective_instance_sha256"] != self.objective_instance_sha256:
            raise ValueError("objective identity does not match its manifest")
        if objective_manifest.get("scene_sha256") != self.dynamic_scene_sha256:
            raise ValueError("objective identity does not bind the dynamic scene")
        candidates = tuple(self.candidates)
        failures = tuple(str(value) for value in self.reference_failures)
        if not candidates:
            raise ValueError("step evidence must contain at least one reference candidate")
        if self.selected_iterations != candidates[-1].iterations:
            raise ValueError("selected_iterations must name the final committed candidate")
        if self.reference_accepted != (not failures):
            raise ValueError("reference_accepted must be exactly the absence of gate failures")
        object.__setattr__(self, "dynamic_scene_manifest", scene_manifest)
        object.__setattr__(self, "objective_manifest", objective_manifest)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "reference_failures", failures)

    def as_dict(self) -> dict[str, object]:
        """Return the complete deterministic step record."""
        return {
            "step_id": self.step_id,
            "input_position_sha256": self.input_position_sha256,
            "input_velocity_sha256": self.input_velocity_sha256,
            "inertial_target_sha256": self.inertial_target_sha256,
            "dynamic_scene_manifest": _thaw_json(self.dynamic_scene_manifest),
            "objective_manifest": _thaw_json(self.objective_manifest),
            "dynamic_scene_sha256": self.dynamic_scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "iterate_zero_metrics": _metrics_record(self.iterate_zero_metrics),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "selected_iterations": self.selected_iterations,
            "output_position_sha256": self.output_position_sha256,
            "output_velocity_sha256": self.output_velocity_sha256,
            "exact_velocity_commit": self.exact_velocity_commit,
            "reference_accepted": self.reference_accepted,
            "reference_failures": list(self.reference_failures),
        }


def _finite_metrics(metrics: CommonStateMetrics) -> bool:
    values = (
        metrics.objective,
        metrics.inertia,
        metrics.elastic,
        metrics.gradient_norm,
        metrics.relative_residual,
        metrics.determinant_min,
        metrics.determinant_max,
        metrics.inverted_tet_fraction,
        metrics.minimum_singular_value,
        metrics.max_pin_error_m,
    )
    return all(math.isfinite(value) for value in values)


def _reference_failures(
    iterate_zero: CommonStateMetrics,
    selected: CommonStateMetrics,
    *,
    protocol: FreeBodyReferenceProtocol,
    exact_velocity_commit: bool,
) -> tuple[str, ...]:
    failures: list[str] = []
    if not _finite_metrics(iterate_zero) or not _finite_metrics(selected):
        failures.append("independent common-objective metrics are nonfinite")
    if selected.determinant_min <= 0.0 or selected.inverted_tet_fraction != 0.0:
        failures.append("selected candidate contains an inverted tetrahedron")
    if selected.minimum_singular_value <= 0.0:
        failures.append("selected candidate has non-positive minimum singular value")
    if selected.max_pin_error_m != 0.0:
        failures.append("free-body candidate unexpectedly reports a pin error")
    if selected.objective > iterate_zero.objective:
        failures.append("selected candidate objective exceeds iterate-zero objective")
    if selected.relative_residual >= iterate_zero.relative_residual:
        failures.append("selected candidate relative residual did not strictly decrease")
    if selected.relative_residual > protocol.maximum_relative_residual:
        failures.append(
            f"selected candidate relative residual {selected.relative_residual:.6g} exceeds registered "
            f"threshold {protocol.maximum_relative_residual:.6g}"
        )
    if iterate_zero.relative_residual <= 0.0:
        failures.append("iterate-zero relative residual is non-positive and cannot define a reduction ratio")
    elif selected.relative_residual / iterate_zero.relative_residual > protocol.maximum_residual_ratio:
        failures.append(
            f"selected-to-iterate-zero residual ratio "
            f"{selected.relative_residual / iterate_zero.relative_residual:.6g} exceeds registered "
            f"threshold {protocol.maximum_residual_ratio:.6g}"
        )
    if not exact_velocity_commit:
        failures.append("selected candidate velocity differs from SolverVBD's exact float32 commit formula")
    return tuple(failures)


@dataclasses.dataclass(frozen=True, eq=False)
class ReferenceRollout:
    """One candidate or fully accepted free-body sequence."""

    scene: FreeBodyReferenceScene
    q: np.ndarray
    qd: np.ndarray
    inertial_target: np.ndarray
    external_force: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray
    steps: tuple[ReferenceStepEvidence, ...]

    def __post_init__(self) -> None:
        q = _readonly_array(self.q, np.float64, "q")
        qd = _readonly_array(self.qd, np.float64, "qd")
        inertial = _readonly_array(self.inertial_target, np.float64, "inertial_target")
        force = _readonly_array(self.external_force, np.float32, "external_force")
        pins = _readonly_array(self.pinned_indices, np.int32, "pinned_indices")
        targets = _readonly_array(self.pin_targets, np.float64, "pin_targets")
        steps = tuple(self.steps)
        if q.ndim != 3 or q.shape[2:] != (3,) or qd.shape != q.shape:
            raise ValueError("q and qd must share shape (S + 1, V, 3)")
        if inertial.shape != (q.shape[0] - 1, q.shape[1], 3):
            raise ValueError("inertial_target must have shape (S, V, 3)")
        if force.shape != (q.shape[0] - 1, q.shape[1], 3):
            raise ValueError("external_force must have shape (S, V, 3)")
        if pins.ndim != 1 or targets.shape != (q.shape[0] - 1, pins.size, 3):
            raise ValueError("pin arrays do not match rollout length")
        if pins.size:
            raise ValueError("the registered free-body rollout must not contain pins")
        if len(steps) != q.shape[0] - 1:
            raise ValueError("step evidence count does not match q")
        for name, values in (("q", q), ("qd", qd), ("inertial_target", inertial)):
            staged = values.astype(np.float32).astype(np.float64)
            if not np.array_equal(values, staged):
                raise ValueError(f"{name} is not a lossless promotion of producer float32")
        if not np.array_equal(q[0], self.scene.template.x_current):
            raise ValueError("q[0] does not match the immutable initial scene")
        if not np.array_equal(qd[0], self.scene.template.velocity):
            raise ValueError("qd[0] does not match the immutable initial scene")
        if np.any(force) or pins.size or targets.size:
            raise ValueError("registered free-body rollout requires exactly zero load and no pins")
        for step_id, step in enumerate(steps):
            if type(step) is not ReferenceStepEvidence or step.step_id != step_id:
                raise ValueError("step evidence must be contiguous and strongly typed")
            array_records = step.dynamic_scene_manifest.get("arrays")
            if not isinstance(array_records, Mapping):
                raise ValueError("dynamic scene manifest is missing its array inventory")
            source_arrays = {
                "external_force": force[step_id],
                "pin_targets": targets[step_id],
                "pinned_indices": pins,
                "vbd_inertial_target": inertial[step_id],
                "velocity": qd[step_id],
                "x_current": q[step_id],
            }
            expected_hashes: dict[str, str] = {}
            for name, source_array in source_arrays.items():
                record = array_records.get(name)
                if not isinstance(record, Mapping) or not isinstance(record.get("dtype"), str):
                    raise ValueError(f"dynamic scene manifest is missing {name} at step {step_id}")
                canonical_source = np.asarray(source_array, dtype=np.dtype(record["dtype"]))
                expected_hash = _array_digest(canonical_source)
                expected_hashes[name] = expected_hash
                if record.get("sha256") != expected_hash:
                    raise ValueError(f"dynamic objective identity does not bind {name} at step {step_id}")
            if step.input_position_sha256 != expected_hashes["x_current"]:
                raise ValueError(f"input position hash does not bind q[{step_id}]")
            if step.input_velocity_sha256 != expected_hashes["velocity"]:
                raise ValueError(f"input velocity hash does not bind qd[{step_id}]")
            if step.inertial_target_sha256 != expected_hashes["vbd_inertial_target"]:
                raise ValueError(f"inertial target hash does not bind step {step_id}")
            output_position_hash = _array_digest(q[step_id + 1])
            output_velocity_hash = _array_digest(qd[step_id + 1])
            if step.output_position_sha256 != output_position_hash:
                raise ValueError(f"output position hash does not bind q[{step_id + 1}]")
            if step.output_velocity_sha256 != output_velocity_hash:
                raise ValueError(f"output velocity hash does not bind qd[{step_id + 1}]")
            selected = step.candidates[-1]
            if selected.metrics.position_sha256 != output_position_hash:
                raise ValueError("selected common-objective metrics do not bind the committed position")
            if selected.velocity_float64_sha256 != output_velocity_hash:
                raise ValueError("selected candidate evidence does not bind the committed velocity")
            expected_velocity = np.asarray(
                (q[step_id + 1].astype(np.float32) - q[step_id].astype(np.float32))
                / np.float32(self.scene.protocol.execution_dt_seconds),
                dtype=np.float32,
            ).astype(np.float64)
            if step.exact_velocity_commit != np.array_equal(qd[step_id + 1], expected_velocity):
                raise ValueError("exact_velocity_commit does not match the stored SolverVBD output")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", qd)
        object.__setattr__(self, "inertial_target", inertial)
        object.__setattr__(self, "external_force", force)
        object.__setattr__(self, "pinned_indices", pins)
        object.__setattr__(self, "pin_targets", targets)
        object.__setattr__(self, "steps", steps)

    @property
    def reference_accepted(self) -> bool:
        """Whether the complete requested sequence passed every gate."""
        return len(self.steps) == self.scene.protocol.rollout_steps and all(
            step.reference_accepted for step in self.steps
        )

    def sequence_arrays(self) -> dict[str, np.ndarray]:
        """Return the exact arrays consumed by the training shard loader."""
        step_count = len(self.steps)
        return {
            "deformation_seed": np.asarray(self.scene.deformation_seed, dtype=np.int64),
            "dt": np.asarray(self.scene.protocol.execution_dt_seconds, dtype=np.float32),
            "external_force": self.external_force,
            "gravity": np.asarray(self.scene.protocol.gravity_m_s2, dtype=np.float32),
            "inertial_target": self.inertial_target,
            "pin_targets": self.pin_targets,
            "pinned_indices": self.pinned_indices,
            "q": self.q,
            "qd": self.qd,
            "step_ids": np.arange(step_count, dtype=np.int64),
            "velocity_seed": np.asarray(self.scene.velocity_seed, dtype=np.int64),
        }


def run_reference_rollout(scene: FreeBodyReferenceScene) -> ReferenceRollout:
    """Run fresh-restart VBD budgets and independently gate each committed step."""
    if type(scene) is not FreeBodyReferenceScene:
        raise TypeError("scene must be exactly FreeBodyReferenceScene")
    protocol = scene.protocol
    runtime = _ReusableVBDRunner(scene)
    q_current = np.asarray(scene.template.x_current, dtype=np.float64)
    qd_current = np.asarray(scene.template.velocity, dtype=np.float64)
    q_history = [q_current.copy()]
    qd_history = [qd_current.copy()]
    inertial_history: list[np.ndarray] = []
    force_history: list[np.ndarray] = []
    evidence: list[ReferenceStepEvidence] = []
    external_force = np.zeros_like(q_current, dtype=np.float32)

    for step_id in range(protocol.rollout_steps):
        dynamic_scene = dataclasses.replace(
            scene.template,
            x_current=q_current,
            velocity=qd_current,
            external_force=external_force,
            pin_targets=np.empty((0, 3), dtype=np.float64),
            metadata=_thaw_json(scene.template.metadata),
        )
        problem = build_common_problem(dynamic_scene)
        dynamic_scene_manifest = dynamic_scene.manifest()
        objective_manifest = common_objective_manifest(dynamic_scene, problem)
        dynamic_scene_sha256 = str(dynamic_scene_manifest["scene_sha256"])
        objective_instance_sha256 = str(objective_manifest["objective_instance_sha256"])
        inertial_target = np.asarray(dynamic_scene.vbd_inertial_target, dtype=np.float64)
        iterate_zero_metrics = evaluate_common_state(problem, inertial_target)
        previous_positions: np.ndarray | None = None
        candidate_records: list[ReferenceCandidateEvidence] = []
        selected_state: _CandidateState | None = None

        for iterations in protocol.iteration_budgets:
            candidate = runtime.solve(q_current, qd_current, external_force, iterations)
            metrics = evaluate_common_state(problem, candidate.positions)
            if metrics.position_sha256 != _array_digest(candidate.positions):
                raise RuntimeError("common evaluator position hash does not bind the VBD candidate")
            displacement = None
            if previous_positions is not None:
                displacement = float(np.sqrt(np.mean(np.sum((candidate.positions - previous_positions) ** 2, axis=1))))
            candidate_records.append(
                ReferenceCandidateEvidence(
                    iterations=iterations,
                    effective_tile_solve=candidate.effective_tile_solve,
                    metrics=metrics,
                    position_float32_sha256=candidate.position_float32_sha256,
                    velocity_float32_sha256=candidate.velocity_float32_sha256,
                    velocity_float64_sha256=_array_digest(candidate.velocities),
                    displacement_from_previous_budget_m=displacement,
                    relative_residual_over_iterate_zero=(
                        metrics.relative_residual / iterate_zero_metrics.relative_residual
                        if iterate_zero_metrics.relative_residual > 0.0
                        else math.inf
                    ),
                )
            )
            previous_positions = candidate.positions
            selected_state = candidate

        if selected_state is None:
            raise RuntimeError("reference protocol did not produce a candidate")
        q_current32 = np.asarray(q_current, dtype=np.float32)
        q_next32 = np.asarray(selected_state.positions, dtype=np.float32)
        expected_velocity32 = np.asarray(
            (q_next32 - q_current32) / np.float32(protocol.execution_dt_seconds),
            dtype=np.float32,
        )
        actual_velocity32 = np.asarray(selected_state.velocities, dtype=np.float32)
        exact_velocity_commit = bool(np.array_equal(actual_velocity32, expected_velocity32))
        selected_metrics = candidate_records[-1].metrics
        failures = _reference_failures(
            iterate_zero_metrics,
            selected_metrics,
            protocol=protocol,
            exact_velocity_commit=exact_velocity_commit,
        )
        step = ReferenceStepEvidence(
            step_id=step_id,
            input_position_sha256=_array_digest(q_current),
            input_velocity_sha256=_array_digest(qd_current),
            inertial_target_sha256=_array_digest(inertial_target),
            dynamic_scene_manifest=dynamic_scene_manifest,
            objective_manifest=objective_manifest,
            dynamic_scene_sha256=dynamic_scene_sha256,
            objective_instance_sha256=objective_instance_sha256,
            iterate_zero_metrics=iterate_zero_metrics,
            candidates=tuple(candidate_records),
            selected_iterations=protocol.iteration_budgets[-1],
            output_position_sha256=_array_digest(selected_state.positions),
            output_velocity_sha256=_array_digest(selected_state.velocities),
            exact_velocity_commit=exact_velocity_commit,
            reference_accepted=not failures,
            reference_failures=failures,
        )
        evidence.append(step)
        inertial_history.append(inertial_target.copy())
        force_history.append(external_force.copy())
        q_current = np.asarray(selected_state.positions, dtype=np.float64)
        qd_current = np.asarray(selected_state.velocities, dtype=np.float64)
        q_history.append(q_current.copy())
        qd_history.append(qd_current.copy())
        if failures:
            break

    step_count = len(evidence)
    return ReferenceRollout(
        scene=scene,
        q=np.stack(q_history, axis=0),
        qd=np.stack(qd_history, axis=0),
        inertial_target=np.stack(inertial_history, axis=0),
        external_force=np.stack(force_history, axis=0),
        pinned_indices=np.empty((0,), dtype=np.int32),
        pin_targets=np.empty((step_count, 0, 3), dtype=np.float64),
        steps=tuple(evidence),
    )


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, _canonical_array(value), version=_NPY_VERSION, allow_pickle=False)
    return stream.getvalue()


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"noncanonical shard array name {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return stream.getvalue()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_atomic_exact(path: pathlib.Path, contents: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == contents:
            return
        raise FileExistsError(f"refusing to replace non-identical shard output {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@dataclasses.dataclass(frozen=True)
class ReferenceShardFiles:
    """Paths written for one accepted sequence and its shared static data."""

    static_npz: pathlib.Path
    sequence_npz: pathlib.Path
    evidence_json: pathlib.Path
    manifest_json: pathlib.Path


def write_reference_rollout_shard(
    destination: str | pathlib.Path,
    rollout: ReferenceRollout,
    *,
    sequence_id: str,
) -> ReferenceShardFiles:
    """Write one accepted sequence, failing before any output for a weak label."""
    if not isinstance(rollout, ReferenceRollout) or not rollout.reference_accepted:
        raise ValueError("rollout is not accepted and must not be written as training data")
    sequence = _asset_slug(_identifier(sequence_id, "sequence_id"))
    output_dir = pathlib.Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    static_path = output_dir / "static.npz"
    sequence_path = output_dir / f"{sequence}.npz"
    evidence_path = output_dir / f"{sequence}.evidence.json"
    manifest_path = output_dir / f"{sequence}.manifest.json"

    static_arrays = rollout.scene.static_arrays()
    sequence_arrays = rollout.sequence_arrays()
    static_contents = _npz_bytes(static_arrays)
    sequence_contents = _npz_bytes(sequence_arrays)
    identities = rollout.scene.physical_identities()
    initial_scene = rollout.scene.template.manifest()
    evidence = {
        "schema": "pss-free-body-reference-evidence-v1",
        "asset_id": rollout.scene.asset_id,
        "sequence_id": sequence_id,
        "protocol": rollout.scene.protocol.as_dict(),
        "steps": [step.as_dict() for step in rollout.steps],
    }
    evidence_contents = _json_bytes(evidence)

    _write_atomic_exact(static_path, static_contents)
    _write_atomic_exact(sequence_path, sequence_contents)
    _write_atomic_exact(evidence_path, evidence_contents)
    files = {
        "static_npz": {
            "path": static_path.name,
            "bytes": len(static_contents),
            "sha256": hashlib.sha256(static_contents).hexdigest(),
            "arrays": {name: _array_record(value) for name, value in sorted(static_arrays.items())},
        },
        "sequence_npz": {
            "path": sequence_path.name,
            "bytes": len(sequence_contents),
            "sha256": hashlib.sha256(sequence_contents).hexdigest(),
            "arrays": {name: _array_record(value) for name, value in sorted(sequence_arrays.items())},
        },
        "evidence_json": {
            "path": evidence_path.name,
            "bytes": len(evidence_contents),
            "sha256": hashlib.sha256(evidence_contents).hexdigest(),
        },
    }
    manifest = {
        "schema": _SCHEMA,
        "asset_id": rollout.scene.asset_id,
        "source": rollout.scene.source,
        "source_sha256": rollout.scene.source_sha256,
        "sequence_id": sequence_id,
        "deformation_seed": rollout.scene.deformation_seed,
        "velocity_seed": rollout.scene.velocity_seed,
        "reference_accepted": True,
        "step_count": len(rollout.steps),
        "protocol": rollout.scene.protocol.as_dict(),
        "identities": identities,
        "initial_scene": initial_scene,
        "initial_scene_sha256": initial_scene["scene_sha256"],
        "normalization": {
            "source_center": rollout.scene.initial_state.source_center.tolist(),
            "source_characteristic_length": rollout.scene.initial_state.source_characteristic_length,
            "normalized_characteristic_length_m": (rollout.scene.initial_state.normalized_characteristic_length_m),
            "orientation_repaired_count": rollout.scene.initial_state.orientation_repaired_count,
        },
        "files": files,
        "inventory_sha256": _canonical_json_digest({"files": files, "identities": identities}),
    }
    _write_atomic_exact(manifest_path, _json_bytes(manifest))
    return ReferenceShardFiles(
        static_npz=static_path,
        sequence_npz=sequence_path,
        evidence_json=evidence_path,
        manifest_json=manifest_path,
    )


def _validate_output_directory(path: pathlib.Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing reference shard directory {path}")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def generate_reference_shard(
    output_dir: str | pathlib.Path,
    *,
    asset_paths: Sequence[str | pathlib.Path] | None = None,
    asset_dir: str | pathlib.Path | None = None,
    protocol: FreeBodyReferenceProtocol | None = None,
    device: str = "cuda",
    base_seed: int = DEFAULT_BASE_SEED,
    samples_per_asset: int = 3,
    n_levels: int = 5,
    cluster_size: int = 8,
    max_points: int = DEFAULT_MAX_POINTS,
    max_tets: int = DEFAULT_MAX_TETS,
) -> pathlib.Path:
    """Generate an atomically published multi-asset reference shard index.

    The defaults regenerate the exact five assets and three role-separated
    seed pairs used by the existing hierarchy-random gallery.  The caller is
    responsible for claiming a GPU before selecting a CUDA device.
    """
    if asset_paths is not None and asset_dir is not None:
        raise ValueError("asset_paths and asset_dir are mutually exclusive")
    if type(samples_per_asset) is not int or samples_per_asset < 1:
        raise ValueError("samples_per_asset must be a positive built-in integer")
    for name, value in (
        ("n_levels", n_levels),
        ("cluster_size", cluster_size),
        ("max_points", max_points),
        ("max_tets", max_tets),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive built-in integer")
    _require_seed(base_seed, "base_seed")
    active_protocol = FreeBodyReferenceProtocol() if protocol is None else protocol
    if type(active_protocol) is not FreeBodyReferenceProtocol:
        raise TypeError("protocol must be exactly FreeBodyReferenceProtocol")
    sources = (
        tuple(pathlib.Path(path) for path in asset_paths) if asset_paths is not None else default_asset_paths(asset_dir)
    )
    if not sources:
        raise ValueError("at least one asset path is required")
    slugs = tuple(_asset_slug(path.stem) for path in sources)
    if len(set(slugs)) != len(slugs):
        raise ValueError("asset basenames must yield unique path components")

    destination = pathlib.Path(output_dir).resolve()
    _validate_output_directory(destination)
    temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        asset_records: list[dict[str, object]] = []
        for source_path, slug in zip(sources, slugs, strict=True):
            logical_source = _logical_source_name(source_path)
            mesh = load_legacy_vtk_tet_mesh(source_path, max_points=max_points, max_tets=max_tets)
            oriented = repair_tet_orientation(mesh.rest_positions, mesh.tet_indices)
            source_hierarchy = build_hierarchy(
                oriented.tet_indices,
                mesh.rest_positions,
                n_levels=n_levels,
                target=cluster_size,
            )
            sequence_records: list[dict[str, object]] = []
            asset_identities: dict[str, object] | None = None
            asset_output = temporary / slug
            for sample_index in range(samples_per_asset):
                deformation_seed, velocity_seed = _sample_seeds(
                    base_seed,
                    source_path.stem,
                    mesh.source_sha256,
                    sample_index,
                )
                generated = generate_hierarchy_random_state(
                    mesh.rest_positions,
                    oriented.tet_indices,
                    source_hierarchy,
                    deformation_seed=deformation_seed,
                    velocity_seed=velocity_seed,
                    config=HierarchyRandomStateConfig(),
                )
                initial = normalize_initial_state(
                    mesh.rest_positions,
                    generated.deformed_positions,
                    generated.velocities,
                    mesh.tet_indices,
                    normalized_characteristic_length_m=active_protocol.normalized_characteristic_length_m,
                )
                normalized_hierarchy = build_hierarchy(
                    initial.tet_indices,
                    initial.rest_q.astype(np.float64),
                    n_levels=n_levels,
                    target=cluster_size,
                )
                reference_scene = build_free_body_scene(
                    initial,
                    protocol=active_protocol,
                    device=device,
                    asset_id=source_path.stem,
                    source=logical_source,
                    source_sha256=mesh.source_sha256,
                    deformation_seed=deformation_seed,
                    velocity_seed=velocity_seed,
                    hierarchy=normalized_hierarchy,
                )
                rollout = run_reference_rollout(reference_scene)
                sequence_id = f"sample-{sample_index:03d}"
                files = write_reference_rollout_shard(asset_output, rollout, sequence_id=sequence_id)
                shard_manifest = json.loads(files.manifest_json.read_text())
                sequence_identities = shard_manifest["identities"]
                if asset_identities is None:
                    asset_identities = sequence_identities
                elif asset_identities != sequence_identities:
                    raise RuntimeError("static physical identities changed between sequences of one asset")
                sequence_records.append(
                    {
                        "sequence_id": sequence_id,
                        "deformation_seed": deformation_seed,
                        "velocity_seed": velocity_seed,
                        "manifest": {
                            "path": str(files.manifest_json.relative_to(temporary)),
                            "sha256": _file_sha256(files.manifest_json),
                        },
                        "sequence_npz": {
                            "path": str(files.sequence_npz.relative_to(temporary)),
                            "sha256": _file_sha256(files.sequence_npz),
                        },
                    }
                )
            static_path = asset_output / "static.npz"
            asset_records.append(
                {
                    "asset_id": source_path.stem,
                    "source": logical_source,
                    "source_sha256": mesh.source_sha256,
                    "vertex_count": int(mesh.rest_positions.shape[0]),
                    "tet_count": int(mesh.tet_indices.shape[0]),
                    "static_npz": {
                        "path": str(static_path.relative_to(temporary)),
                        "sha256": _file_sha256(static_path),
                    },
                    "identities": asset_identities,
                    "sequences": sequence_records,
                }
            )
        index = {
            "schema": _INDEX_SCHEMA,
            "protocol": active_protocol.as_dict(),
            "protocol_sha256": _canonical_json_digest(active_protocol.as_dict()),
            "base_seed": base_seed,
            "samples_per_asset": samples_per_asset,
            "hierarchy_config": {"n_levels": n_levels, "cluster_size": cluster_size},
            "asset_count": len(asset_records),
            "accepted_sequence_count": sum(len(record["sequences"]) for record in asset_records),
            "assets": asset_records,
        }
        _write_atomic_exact(temporary / "index.json", _json_bytes(index))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "index.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--asset-dir", type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the registered five-by-three pilot shard."""
    args = _parse_args(argv)
    index_path = generate_reference_shard(
        args.output,
        asset_dir=args.asset_dir,
        device=args.device,
    )
    print(f"Wrote accepted free-body reference shard index to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
