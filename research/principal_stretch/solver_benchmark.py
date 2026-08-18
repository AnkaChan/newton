# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Common-objective benchmark plumbing for principal-stretch research.

This module deliberately starts with the smallest comparison that can be made
honestly: one contact-free, undamped implicit-Euler tetrahedral step.  It builds
the exact same scene for SolverVBD and the dense CPU Newton reference, restarts
every VBD iteration budget from the same input state, and scores every result
with one independent float64 objective and residual evaluator.

The code is research infrastructure rather than a public Newton API.  Larger
PR #2901 and Gaia scenes can use the same :class:`TetBenchmarkScene` contract
after the tiny stationary-point agreement gate passes.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import numbers
import pathlib
import platform
import statistics
import subprocess
import sys
import time
import types
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverVBD

from .newton_baseline import (
    NewtonConfig,
    NewtonProblem,
    NewtonResidualPolishConfig,
    NewtonResidualPolishResult,
    NewtonResult,
    build_newton_problem,
    solve_newton,
    solve_newton_residual_polish,
)
from .potentials import incremental_potential_stable_neo_hookean
from .sparse_newton_reference import (
    SPARSE_FACTOR_PIVOT_RELATIVE_MARGIN,
    SPARSE_FACTOR_RELATION_RELATIVE_LIMIT,
    SPARSE_FACTOR_UNIT_DIAGONAL_LIMIT,
    SPARSE_FACTORIZATION_RELATIVE_RESIDUAL_LIMIT,
    SPARSE_LINEAR_RESIDUAL_LIMIT,
    SPARSE_MAX_REFINEMENT_STEPS,
    SparseNewtonResult,
    solve_sparse_newton,
)
from .torch_solver import compute_F

_SCHEMA_VERSION = 2


def _readonly_array(value, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _readonly_vbd_float(value, name: str) -> np.ndarray:
    """Canonicalize a VBD-consumed float through fp32, then promote."""
    float32 = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(float32).all():
        raise ValueError(f"{name} must remain finite when represented in SolverVBD float32")
    return _readonly_array(float32, np.float64, name)


def _vbd_inertial_target(
    x_current: np.ndarray,
    velocity: np.ndarray,
    gravity: np.ndarray,
    external_force: np.ndarray,
    inverse_mass: np.ndarray,
    pinned_indices: np.ndarray,
    pin_targets: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reproduce ``SolverVBD``'s float32 forward-step operation order."""
    positions = np.asarray(x_current, dtype=np.float32).copy()
    positions[pinned_indices] = np.asarray(pin_targets, dtype=np.float32)
    velocity32 = np.asarray(velocity, dtype=np.float32)
    gravity32 = np.asarray(gravity, dtype=np.float32)
    force32 = np.asarray(external_force, dtype=np.float32)
    inverse_mass32 = np.asarray(inverse_mass, dtype=np.float32)
    dt32 = np.float32(dt)

    force_acceleration = (force32 * inverse_mass32[:, None]).astype(np.float32)
    acceleration = (gravity32[None, :] + force_acceleration).astype(np.float32)
    velocity_new = (velocity32 + (acceleration * dt32).astype(np.float32)).astype(np.float32)
    target = (positions + (velocity_new * dt32).astype(np.float32)).astype(np.float32)
    target[pinned_indices] = positions[pinned_indices]
    return _readonly_array(target, np.float64, "vbd_inertial_target")


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return types.MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_array(array: np.ndarray) -> np.ndarray:
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=canonical_dtype)


def _array_digest(array: np.ndarray) -> str:
    """Hash an array together with its dtype and shape."""
    contiguous = _canonical_array(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> str | None:
    repository = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty_digest() -> str | None:
    """Hash tracked changes and untracked source paths, or return ``None``."""
    repository = pathlib.Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--porcelain", "-z"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if status.returncode != 0 or not status.stdout:
        return None
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    digest = hashlib.sha256()
    digest.update(status.stdout)
    digest.update(diff.stdout)
    if untracked.returncode == 0:
        for relative_bytes in sorted(item for item in untracked.stdout.split(b"\0") if item):
            relative = relative_bytes.decode("utf-8", errors="surrogateescape")
            path = repository / relative
            digest.update(relative_bytes)
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, eq=False)
class TetBenchmarkScene:
    """Immutable data contract for one common-objective tet substep.

    Arrays are copied into canonical CPU NumPy dtypes and made read-only.  A
    scene contains no contacts, surface energies, or damping.  Kinematic
    particles are identified by the union of zero mass and a missing
    :attr:`newton.ParticleFlags.ACTIVE` bit, exactly matching SolverVBD.
    """

    name: str
    source: str
    rest_q: np.ndarray
    tet_indices: np.ndarray
    tet_poses: np.ndarray
    mass: np.ndarray
    particle_inv_mass: np.ndarray
    tet_materials: np.ndarray
    tri_indices: np.ndarray
    tri_poses: np.ndarray
    tri_materials: np.ndarray
    tri_areas: np.ndarray
    particle_flags: np.ndarray
    color_group_offsets: np.ndarray
    color_group_particles: np.ndarray
    x_current: np.ndarray
    velocity: np.ndarray
    gravity: np.ndarray
    external_force: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray
    dt: float
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)
    vbd_inertial_target: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        arrays = {
            "rest_q": _readonly_vbd_float(self.rest_q, "rest_q"),
            "tet_indices": _readonly_array(self.tet_indices, np.int64, "tet_indices"),
            "tet_poses": _readonly_vbd_float(self.tet_poses, "tet_poses"),
            "mass": _readonly_vbd_float(self.mass, "mass"),
            "particle_inv_mass": _readonly_vbd_float(self.particle_inv_mass, "particle_inv_mass"),
            "tet_materials": _readonly_vbd_float(self.tet_materials, "tet_materials"),
            "tri_indices": _readonly_array(self.tri_indices, np.int64, "tri_indices"),
            "tri_poses": _readonly_vbd_float(self.tri_poses, "tri_poses"),
            "tri_materials": _readonly_vbd_float(self.tri_materials, "tri_materials"),
            "tri_areas": _readonly_vbd_float(self.tri_areas, "tri_areas"),
            "particle_flags": _readonly_array(self.particle_flags, np.int32, "particle_flags"),
            "color_group_offsets": _readonly_array(self.color_group_offsets, np.int64, "color_group_offsets"),
            "color_group_particles": _readonly_array(self.color_group_particles, np.int64, "color_group_particles"),
            "x_current": _readonly_vbd_float(self.x_current, "x_current"),
            "velocity": _readonly_vbd_float(self.velocity, "velocity"),
            "gravity": _readonly_vbd_float(self.gravity, "gravity"),
            "external_force": _readonly_vbd_float(self.external_force, "external_force"),
            "pinned_indices": _readonly_array(self.pinned_indices, np.int64, "pinned_indices"),
            "pin_targets": _readonly_vbd_float(self.pin_targets, "pin_targets"),
        }
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

        if not self.name:
            raise ValueError("scene name must not be empty")
        if not self.source:
            raise ValueError("scene source must not be empty")
        canonical_dt = float(np.float32(self.dt))
        object.__setattr__(self, "dt", canonical_dt)
        if not math.isfinite(canonical_dt) or canonical_dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {self.dt}")

        n_vertices = self.rest_q.shape[0]
        n_tets = self.tet_indices.shape[0]
        n_triangles = self.tri_indices.shape[0]
        expected_shapes = {
            "rest_q": (n_vertices, 3),
            "tet_indices": (n_tets, 4),
            "tet_poses": (n_tets, 3, 3),
            "mass": (n_vertices,),
            "particle_inv_mass": (n_vertices,),
            "tet_materials": (n_tets, 3),
            "tri_indices": (n_triangles, 3),
            "tri_poses": (n_triangles, 2, 2),
            "tri_materials": (n_triangles, 5),
            "tri_areas": (n_triangles,),
            "particle_flags": (n_vertices,),
            "color_group_particles": (n_vertices,),
            "x_current": (n_vertices, 3),
            "velocity": (n_vertices, 3),
            "gravity": (3,),
            "external_force": (n_vertices, 3),
            "pin_targets": (self.pinned_indices.size, 3),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        if n_vertices == 0 or n_tets == 0:
            raise ValueError("scene must contain at least one vertex and tetrahedron")
        if self.tet_indices.min() < 0 or self.tet_indices.max() >= n_vertices:
            raise ValueError("tet_indices contains an out-of-range vertex")
        if n_triangles and (self.tri_indices.min() < 0 or self.tri_indices.max() >= n_vertices):
            raise ValueError("tri_indices contains an out-of-range vertex")
        if np.any(self.mass < 0.0):
            raise ValueError("mass must be non-negative")
        expected_inverse_mass = np.zeros(n_vertices, dtype=np.float32)
        positive_mass = self.mass.astype(np.float32) > 0.0
        expected_inverse_mass[positive_mass] = (np.float32(1.0) / self.mass.astype(np.float32)[positive_mass]).astype(
            np.float32
        )
        if not np.array_equal(self.particle_inv_mass, expected_inverse_mass.astype(np.float64)):
            raise ValueError("particle_inv_mass must exactly match SolverVBD's float32 reciprocal mass")
        if np.any(self.tet_materials < 0.0):
            raise ValueError("tet material coefficients must be non-negative")
        active_material = np.any(self.tet_materials[:, :2] > 0.0, axis=1)
        if np.any(active_material & (self.tet_materials[:, 1] <= 0.0)):
            raise ValueError("lambda must be positive on active tetrahedra")
        if np.any(self.tri_materials != 0.0):
            raise ValueError("benchmark boundary triangles must have exactly zero material coefficients")
        if n_triangles and (np.any(self.tri_areas <= 0.0) or np.any(np.linalg.det(self.tri_poses) <= 0.0)):
            raise ValueError("triangle rest geometry must have positive area and orientation")

        if self.color_group_offsets.ndim != 1 or self.color_group_offsets.size < 2:
            raise ValueError("color_group_offsets must have shape (C + 1,) for at least one color")
        if self.color_group_offsets[0] != 0 or self.color_group_offsets[-1] != n_vertices:
            raise ValueError("color_group_offsets must cover every color_group_particles entry")
        if np.any(np.diff(self.color_group_offsets) <= 0):
            raise ValueError("every stored VBD color group must be non-empty")
        if not np.array_equal(np.sort(self.color_group_particles), np.arange(n_vertices)):
            raise ValueError("stored VBD color groups must contain every vertex exactly once")

        if self.pinned_indices.ndim != 1:
            raise ValueError("pinned_indices must be one-dimensional")
        if self.pinned_indices.size:
            if self.pinned_indices.min() < 0 or self.pinned_indices.max() >= n_vertices:
                raise ValueError("pinned_indices contains an out-of-range vertex")
            if not np.array_equal(self.pinned_indices, np.unique(self.pinned_indices)):
                raise ValueError("pinned_indices must be sorted and unique")

        active_flag = int(newton.ParticleFlags.ACTIVE)
        expected_pins = np.where((self.mass == 0.0) | ((self.particle_flags & active_flag) == 0))[0]
        if not np.array_equal(self.pinned_indices, expected_pins):
            raise ValueError("pinned_indices must match SolverVBD's zero-mass/inactive particles")
        free = np.setdiff1d(np.arange(n_vertices), self.pinned_indices, assume_unique=True)
        if free.size == 0 or np.any(self.mass[free] <= 0.0):
            raise ValueError("scene must contain a positive-mass free vertex")

        object.__setattr__(
            self,
            "vbd_inertial_target",
            _vbd_inertial_target(
                self.x_current,
                self.velocity,
                self.gravity,
                self.external_force,
                self.particle_inv_mass,
                self.pinned_indices,
                self.pin_targets,
                self.dt,
            ),
        )

        determinant = np.linalg.det(self.tet_poses)
        if np.any(determinant <= 0.0):
            raise ValueError("tet_poses must be finite, positively oriented inverse rest matrices")

        try:
            metadata_copy = json.loads(json.dumps(dict(self.metadata), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON values") from exc
        object.__setattr__(self, "metadata", _freeze_json(metadata_copy))

    @property
    def n_vertices(self) -> int:
        return int(self.rest_q.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tet_indices.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.tri_indices.shape[0])

    @property
    def free_indices(self) -> np.ndarray:
        mask = np.ones(self.n_vertices, dtype=bool)
        mask[self.pinned_indices] = False
        return np.where(mask)[0]

    def manifest(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible scene manifest."""
        arrays = {}
        for name in (
            "rest_q",
            "tet_indices",
            "tet_poses",
            "mass",
            "particle_inv_mass",
            "tet_materials",
            "tri_indices",
            "tri_poses",
            "tri_materials",
            "tri_areas",
            "particle_flags",
            "color_group_offsets",
            "color_group_particles",
            "x_current",
            "velocity",
            "gravity",
            "external_force",
            "pinned_indices",
            "pin_targets",
            "vbd_inertial_target",
        ):
            array = getattr(self, name)
            arrays[name] = {
                "dtype": array.dtype.name,
                "shape": list(array.shape),
                "sha256": _array_digest(array),
            }
        payload: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "name": self.name,
            "source": self.source,
            "dt_seconds": self.dt,
            "n_vertices": self.n_vertices,
            "n_tets": self.n_tets,
            "n_triangles": self.n_triangles,
            "n_pinned": int(self.pinned_indices.size),
            "metadata": _thaw_json(self.metadata),
            "arrays": arrays,
        }
        payload["scene_sha256"] = _canonical_digest(payload)
        return payload


def scene_from_model(
    model: newton.Model,
    *,
    name: str,
    source: str,
    dt: float,
    x_current: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    external_force: np.ndarray | None = None,
    pin_targets: np.ndarray | None = None,
    metadata: Mapping[str, object] | None = None,
) -> TetBenchmarkScene:
    """Snapshot one finalized public Newton model into the benchmark contract.

    The supported model contains particles, tetrahedral elasticity, and
    zero-energy boundary triangles only. Contact, damping, springs, bending
    edges, bodies, and surface energy are intentionally excluded from the
    shared scalar objective.
    """
    if model.particle_count == 0 or model.tet_count == 0:
        raise ValueError("benchmark model must contain particles and tetrahedra")
    if model.edge_count or model.spring_count or model.body_count or model.shape_count:
        raise ValueError("benchmark model must not contain edges, springs, bodies, or shapes")
    if not model.particle_color_groups:
        raise ValueError("benchmark model must be colored before finalization")

    rest_q = model.particle_q.numpy().astype(np.float64)
    mass = model.particle_mass.numpy().astype(np.float64)
    particle_flags = model.particle_flags.numpy().astype(np.int32)
    active_flag = int(newton.ParticleFlags.ACTIVE)
    pinned = np.where((mass == 0.0) | ((particle_flags & active_flag) == 0))[0].astype(np.int64)
    positions = rest_q if x_current is None else np.asarray(x_current, dtype=np.float64)
    velocities = model.particle_qd.numpy().astype(np.float64) if velocity is None else velocity
    forces = np.zeros_like(rest_q) if external_force is None else external_force
    targets = positions[pinned] if pin_targets is None else pin_targets

    if model.tri_count:
        tri_indices = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
        tri_poses = model.tri_poses.numpy().reshape(-1, 2, 2).astype(np.float64)
        tri_materials = model.tri_materials.numpy().reshape(-1, 5).astype(np.float64)
        tri_areas = model.tri_areas.numpy().astype(np.float64)
    else:
        tri_indices = np.empty((0, 3), dtype=np.int64)
        tri_poses = np.empty((0, 2, 2), dtype=np.float64)
        tri_materials = np.empty((0, 5), dtype=np.float64)
        tri_areas = np.empty((0,), dtype=np.float64)

    color_groups = [group.numpy().astype(np.int64) for group in model.particle_color_groups]
    color_group_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum([group.size for group in color_groups], dtype=np.int64))
    )
    scene_metadata = {
        "newton_revision": _git_revision(),
        "dirty_tree_sha256": _git_dirty_digest(),
    }
    if metadata is not None:
        overlap = scene_metadata.keys() & metadata.keys()
        if overlap:
            raise ValueError(f"metadata must not override reserved keys: {sorted(overlap)}")
        scene_metadata.update(metadata)
    return TetBenchmarkScene(
        name=name,
        source=source,
        rest_q=rest_q,
        tet_indices=model.tet_indices.numpy().reshape(-1, 4).astype(np.int64),
        tet_poses=model.tet_poses.numpy().reshape(-1, 3, 3).astype(np.float64),
        mass=mass,
        particle_inv_mass=model.particle_inv_mass.numpy().astype(np.float64),
        tet_materials=model.tet_materials.numpy().reshape(-1, 3).astype(np.float64),
        tri_indices=tri_indices,
        tri_poses=tri_poses,
        tri_materials=tri_materials,
        tri_areas=tri_areas,
        particle_flags=particle_flags,
        color_group_offsets=color_group_offsets,
        color_group_particles=np.concatenate(color_groups),
        x_current=positions,
        velocity=velocities,
        gravity=model.gravity.numpy().reshape(-1, 3)[0],
        external_force=forces,
        pinned_indices=pinned,
        pin_targets=targets,
        dt=dt,
        metadata=scene_metadata,
    )


def build_structured_cantilever_scene(
    *,
    dimensions: Sequence[int] = (1, 1, 1),
    cell_size: Sequence[float] = (1.0, 1.0, 1.0),
    density: float = 1.0,
    mu: float = 20.0,
    lam: float = 40.0,
    dt: float = 1.0 / 16.0,
    gravity: Sequence[float] = (0.0, 0.0, -2.0),
    total_tip_force: Sequence[float] = (4.0, -3.0, -6.0),
    initial_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    name: str | None = None,
) -> TetBenchmarkScene:
    """Build a deterministic fixed-left structured tet benchmark.

    The input is first finalized as a Newton model on CPU.  Consequently the
    manifest records the exact float32 masses, inverse rest poses, and material
    values consumed by SolverVBD, promoted losslessly to float64 for the common
    evaluator.  ``total_tip_force`` is a total resultant distributed uniformly
    over the free vertices on the right face; it does not grow with refinement.
    """
    dimensions_tuple = tuple(dimensions)
    if len(dimensions_tuple) != 3 or any(
        not isinstance(value, numbers.Integral) or isinstance(value, bool) for value in dimensions_tuple
    ):
        raise ValueError("dimensions must contain three positive integers")
    dims = tuple(int(value) for value in dimensions_tuple)
    cells = tuple(float(value) for value in cell_size)
    if any(value <= 0 for value in dims):
        raise ValueError("dimensions must contain three positive integers")
    if len(cells) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in cells):
        raise ValueError("cell_size must contain three finite positive values")
    for scalar_name, scalar in (("density", density), ("mu", mu), ("lam", lam)):
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{scalar_name} must be finite and positive")

    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dims[0],
        dim_y=dims[1],
        dim_z=dims[2],
        cell_x=cells[0],
        cell_y=cells[1],
        cell_z=cells[2],
        density=density,
        k_mu=mu,
        k_lambda=lam,
        k_damp=0.0,
        fix_left=True,
        add_surface_mesh_edges=False,
    )
    builder.color()
    model = builder.finalize(device="cpu")

    rest_q = model.particle_q.numpy().astype(np.float64)
    tet_indices = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)
    tet_poses = model.tet_poses.numpy().reshape(-1, 3, 3).astype(np.float64)
    mass = model.particle_mass.numpy().astype(np.float64)
    particle_inv_mass = model.particle_inv_mass.numpy().astype(np.float64)
    tet_materials = model.tet_materials.numpy().reshape(-1, 3).astype(np.float64)
    tri_indices = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
    tri_poses = model.tri_poses.numpy().reshape(-1, 2, 2).astype(np.float64)
    tri_materials = model.tri_materials.numpy().reshape(-1, 5).astype(np.float64)
    tri_areas = model.tri_areas.numpy().astype(np.float64)
    particle_flags = model.particle_flags.numpy().astype(np.int32)
    color_groups = [group.numpy().astype(np.int64) for group in model.particle_color_groups]
    color_group_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum([group.size for group in color_groups], dtype=np.int64))
    )
    color_group_particles = np.concatenate(color_groups)
    active_flag = int(newton.ParticleFlags.ACTIVE)
    pinned = np.where((mass == 0.0) | ((particle_flags & active_flag) == 0))[0].astype(np.int64)

    velocity_vector = np.asarray(initial_velocity, dtype=np.float64)
    force_vector = np.asarray(total_tip_force, dtype=np.float64)
    gravity_vector = np.asarray(gravity, dtype=np.float64)
    if velocity_vector.shape != (3,) or force_vector.shape != (3,) or gravity_vector.shape != (3,):
        raise ValueError("gravity, total_tip_force, and initial_velocity must have shape (3,)")
    if not np.isfinite(velocity_vector).all() or not np.isfinite(force_vector).all():
        raise ValueError("load and velocity vectors must be finite")

    velocity = np.broadcast_to(velocity_vector, rest_q.shape).copy()
    velocity[pinned] = 0.0
    external_force = np.zeros_like(rest_q)
    free_mask = np.ones(rest_q.shape[0], dtype=bool)
    free_mask[pinned] = False
    tip_mask = free_mask & np.isclose(rest_q[:, 0], rest_q[:, 0].max())
    if not np.any(tip_mask):
        raise RuntimeError("structured scene has no free right-face vertices")
    external_force[tip_mask] = force_vector / int(np.count_nonzero(tip_mask))

    scene_name = name or f"structured-cantilever-{dims[0]}x{dims[1]}x{dims[2]}"
    return TetBenchmarkScene(
        name=scene_name,
        source="newton.ModelBuilder.add_soft_grid",
        rest_q=rest_q,
        tet_indices=tet_indices,
        tet_poses=tet_poses,
        mass=mass,
        particle_inv_mass=particle_inv_mass,
        tet_materials=tet_materials,
        tri_indices=tri_indices,
        tri_poses=tri_poses,
        tri_materials=tri_materials,
        tri_areas=tri_areas,
        particle_flags=particle_flags,
        color_group_offsets=color_group_offsets,
        color_group_particles=color_group_particles,
        x_current=rest_q,
        velocity=velocity,
        gravity=gravity_vector,
        external_force=external_force,
        pinned_indices=pinned,
        pin_targets=rest_q[pinned],
        dt=dt,
        metadata={
            "dimensions_cells": list(dims),
            "cell_size_m": list(cells),
            "density_kg_m3": density,
            "mu_pa": mu,
            "lambda_pa_direct_vbd_semantics": lam,
            "lambda_linearized_pa": lam - mu,
            "coefficient_convention": "a727e58c-stored-k_mu-and-k_lambda-consumed-directly",
            "damping": 0.0,
            "contact": False,
            "vbd_model_contents": "tet elasticity plus preserved zero-stiffness boundary triangles",
            "tip_force_semantics": "total resultant uniformly distributed over free right-face vertices",
            "requested_total_tip_force_N": force_vector.tolist(),
            "realized_total_tip_force_N": external_force.astype(np.float32).astype(np.float64).sum(axis=0).tolist(),
            "newton_revision": _git_revision(),
            "dirty_tree_sha256": _git_dirty_digest(),
        },
    )


def build_common_problem(scene: TetBenchmarkScene) -> NewtonProblem:
    """Build the shared float64 objective for a benchmark scene."""
    if np.any(scene.tet_materials[:, 2] != 0.0):
        raise ValueError("the common objective currently requires exactly zero tet damping")
    return build_newton_problem(
        scene.rest_q,
        scene.tet_indices,
        scene.tet_poses,
        scene.mass,
        scene.tet_materials[:, 0],
        scene.tet_materials[:, 1],
        scene.dt,
        x_current=scene.x_current,
        velocity=scene.velocity,
        gravity=scene.gravity,
        external_force=scene.external_force,
        pinned_indices=scene.pinned_indices,
        pin_targets=scene.pin_targets,
        inertial_target=scene.vbd_inertial_target,
    )


@dataclasses.dataclass(frozen=True)
class CommonStateMetrics:
    """Independent common-objective measurements for one candidate state."""

    objective: float
    inertia: float
    elastic: float
    gradient_norm: float
    relative_residual: float
    determinant_min: float
    determinant_max: float
    inverted_tet_fraction: float
    minimum_singular_value: float
    free_rms_error_m: float | None
    mass_weighted_rms_error_m: float | None
    max_pin_error_m: float
    position_sha256: str

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def evaluate_common_state(
    problem: NewtonProblem,
    positions: np.ndarray | torch.Tensor,
    *,
    reference_positions: np.ndarray | torch.Tensor | None = None,
) -> CommonStateMetrics:
    """Score a state without using any solver's internal residual.

    Errors are computed over free vertices, avoiding the pinned-zero dilution
    that affected the earlier rollout report. Pin violations are never
    projected away: canonical Dirichlet values must match bit-for-bit before
    the constrained free-gradient is evaluated.
    """
    if isinstance(positions, np.ndarray):
        x = torch.from_numpy(np.array(positions, dtype=np.float64, copy=True))
    else:
        x = torch.as_tensor(positions, dtype=torch.float64, device="cpu").detach().clone()
    if x.shape != problem.rest_q.shape:
        raise ValueError(f"positions must have shape {tuple(problem.rest_q.shape)}, got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("positions must be finite")
    if problem.pinned.numel():
        pin_error = torch.linalg.vector_norm(x[problem.pinned] - problem.pin_targets, dim=1)
        max_pin_error = float(pin_error.max())
    else:
        max_pin_error = 0.0
    if max_pin_error != 0.0:
        raise ValueError(f"candidate violates Dirichlet targets by {max_pin_error:.3e} m")

    z = problem.free_from_positions(x).requires_grad_(True)
    constrained_x = problem.positions_from_free(z)
    components = incremental_potential_stable_neo_hookean(
        constrained_x,
        problem.inertial_target,
        problem.mass,
        problem.tets,
        problem.J,
        problem.mu,
        problem.lam,
        problem.volume,
        problem.dt,
    )
    (gradient,) = torch.autograd.grad(components["total"], z)
    gradient_norm = float(torch.linalg.vector_norm(gradient))

    deformation_gradient = compute_F(constrained_x, problem.tets, problem.J)
    determinants = torch.linalg.det(deformation_gradient)
    singular_values = torch.linalg.svdvals(deformation_gradient)

    free_rms_error = None
    mass_weighted_rms_error = None
    if reference_positions is not None:
        if isinstance(reference_positions, np.ndarray):
            reference = torch.from_numpy(np.array(reference_positions, dtype=np.float64, copy=True))
        else:
            reference = torch.as_tensor(reference_positions, dtype=torch.float64, device="cpu").detach()
        if reference.shape != x.shape or not torch.isfinite(reference).all():
            raise ValueError("reference_positions must be a finite array with the candidate shape")
        difference_sq = ((constrained_x[problem.free] - reference[problem.free]) ** 2).sum(dim=1)
        free_rms_error = float(torch.sqrt(difference_sq.mean()).detach())
        free_mass = problem.mass[problem.free]
        mass_weighted_rms_error = float(torch.sqrt((free_mass * difference_sq).sum() / free_mass.sum()).detach())

    x_numpy = x.detach().numpy()
    return CommonStateMetrics(
        objective=float(components["total"].detach()),
        inertia=float(components["inertia"].detach()),
        elastic=float(components["elastic"].detach()),
        gradient_norm=gradient_norm,
        relative_residual=gradient_norm / problem.residual_scale,
        determinant_min=float(determinants.min().detach()),
        determinant_max=float(determinants.max().detach()),
        inverted_tet_fraction=float((determinants <= 0.0).to(torch.float64).mean().detach()),
        minimum_singular_value=float(singular_values.min().detach()),
        free_rms_error_m=free_rms_error,
        mass_weighted_rms_error_m=mass_weighted_rms_error,
        max_pin_error_m=max_pin_error,
        position_sha256=_array_digest(x_numpy),
    )


def _assert_model_matches_scene(model: newton.Model, scene: TetBenchmarkScene) -> None:
    checks = [
        ("rest positions", model.particle_q.numpy(), scene.rest_q),
        ("masses", model.particle_mass.numpy(), scene.mass),
        ("inverse masses", model.particle_inv_mass.numpy(), scene.particle_inv_mass),
        ("inverse rest poses", model.tet_poses.numpy().reshape(-1, 3, 3), scene.tet_poses),
        ("tet materials", model.tet_materials.numpy().reshape(-1, 3), scene.tet_materials),
    ]
    if scene.n_triangles:
        checks.extend(
            (
                ("triangle inverse rest poses", model.tri_poses.numpy().reshape(-1, 2, 2), scene.tri_poses),
                ("triangle materials", model.tri_materials.numpy().reshape(-1, 5), scene.tri_materials),
                ("triangle areas", model.tri_areas.numpy(), scene.tri_areas),
            )
        )
    for name, actual, expected in checks:
        if not np.array_equal(actual.astype(np.float64), expected):
            error = float(np.max(np.abs(actual - expected)))
            raise RuntimeError(f"rebuilt VBD model changed {name} (max error {error:.3e})")
    if not np.array_equal(model.tet_indices.numpy().reshape(-1, 4), scene.tet_indices):
        raise RuntimeError("rebuilt VBD model changed tet topology or ordering")
    actual_triangles = (
        np.empty((0, 3), dtype=np.int64) if model.tri_indices is None else model.tri_indices.numpy().reshape(-1, 3)
    )
    if not np.array_equal(actual_triangles, scene.tri_indices):
        raise RuntimeError("rebuilt VBD model changed boundary-triangle topology or ordering")
    if not np.array_equal(model.particle_flags.numpy(), scene.particle_flags):
        raise RuntimeError("rebuilt VBD model changed particle flags")


def _build_vbd_model(scene: TetBenchmarkScene, device: str):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particles(
        pos=[wp.vec3(*position) for position in scene.rest_q],
        vel=[wp.vec3(*velocity) for velocity in scene.velocity],
        mass=scene.mass.tolist(),
        flags=scene.particle_flags.tolist(),
    )
    for tet, material in zip(scene.tet_indices, scene.tet_materials, strict=True):
        volume = builder.add_tetrahedron(
            int(tet[0]),
            int(tet[1]),
            int(tet[2]),
            int(tet[3]),
            float(material[0]),
            float(material[1]),
            float(material[2]),
        )
        if volume <= 0.0:
            raise RuntimeError("scene reconstruction produced an inverted rest tetrahedron")
    for triangle, material in zip(scene.tri_indices, scene.tri_materials, strict=True):
        area = builder.add_triangle(
            int(triangle[0]),
            int(triangle[1]),
            int(triangle[2]),
            float(material[0]),
            float(material[1]),
            float(material[2]),
            float(material[3]),
            float(material[4]),
        )
        if area <= 0.0:
            raise RuntimeError("scene reconstruction produced a degenerate boundary triangle")
    groups = [
        scene.color_group_particles[scene.color_group_offsets[index] : scene.color_group_offsets[index + 1]].astype(
            np.int32
        )
        for index in range(scene.color_group_offsets.size - 1)
    ]
    builder.set_coloring(groups)
    model = builder.finalize(device=device)
    model.set_gravity(scene.gravity)
    _assert_model_matches_scene(model, scene)
    return model


def _make_vbd_state(model: newton.Model, scene: TetBenchmarkScene):
    state_in = model.state()
    state_out = model.state()
    positions = scene.x_current.copy()
    positions[scene.pinned_indices] = scene.pin_targets
    state_in.clear_forces()
    state_in.particle_q.assign(wp.array(positions.astype(np.float32), dtype=wp.vec3, device=model.device))
    state_in.particle_qd.assign(wp.array(scene.velocity.astype(np.float32), dtype=wp.vec3, device=model.device))
    state_in.particle_f.assign(wp.array(scene.external_force.astype(np.float32), dtype=wp.vec3, device=model.device))
    return state_in, state_out, model.control()


@dataclasses.dataclass(frozen=True, eq=False)
class VBDRunResult:
    """One fixed-objective VBD solve with synchronized repeat timings."""

    positions: np.ndarray
    velocities: np.ndarray
    iterations: int
    requested_tile_solve: bool
    effective_tile_solve: bool
    color_group_count: int
    device: str
    setup_seconds: float
    warmup_seconds: float
    repeat_seconds: tuple[float, ...]
    transfer_seconds: tuple[float, ...]
    scene_sha256: str
    objective_instance_sha256: str
    physical_state_sha256: str
    iterate_zero_sha256: str
    result_state_sha256: str
    run_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _readonly_array(self.positions, np.float64, "positions"))
        object.__setattr__(self, "velocities", _readonly_array(self.velocities, np.float64, "velocities"))

    @property
    def median_solve_seconds(self) -> float:
        return statistics.median(self.repeat_seconds)


def _vbd_run_digest(result: VBDRunResult) -> str:
    payload = {
        "positions_sha256": _array_digest(result.positions),
        "velocities_sha256": _array_digest(result.velocities),
        "iterations": result.iterations,
        "requested_tile_solve": result.requested_tile_solve,
        "effective_tile_solve": result.effective_tile_solve,
        "color_group_count": result.color_group_count,
        "device": result.device,
        "setup_seconds": result.setup_seconds,
        "warmup_seconds": result.warmup_seconds,
        "repeat_seconds": list(result.repeat_seconds),
        "transfer_seconds": list(result.transfer_seconds),
        "scene_sha256": result.scene_sha256,
        "objective_instance_sha256": result.objective_instance_sha256,
        "physical_state_sha256": result.physical_state_sha256,
        "iterate_zero_sha256": result.iterate_zero_sha256,
        "result_state_sha256": result.result_state_sha256,
    }
    return _canonical_digest(payload)


def run_vbd(
    scene: TetBenchmarkScene,
    iterations: int,
    *,
    device: str = "cpu",
    tile_solve: bool = False,
    warmup: bool = True,
    repeats: int = 5,
) -> VBDRunResult:
    """Run ``SolverVBD(iterations=K)`` from the exact manifest state.

    Every timing repeat gets fresh input/output states.  This function never
    calls ``step(iterations=1)`` repeatedly to form a curve, because that would
    advance physical time and change the implicit objective at every call.
    """
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    requested_device = wp.get_device(device)
    if tile_solve and requested_device.is_cuda and scene.n_triangles == 0:
        raise ValueError("CUDA tile solve requires preserved boundary triangles in this benchmark adapter")

    common_problem = build_common_problem(scene)
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(common_objective_manifest(scene, common_problem)["objective_instance_sha256"])
    iterate_zero = common_problem.inertial_target.index_copy(
        0, common_problem.pinned, common_problem.pin_targets
    ).numpy()

    with wp.ScopedTimer("vbd-setup", print=False, synchronize=True) as setup_timer:
        model = _build_vbd_model(scene, str(requested_device))
        solver = SolverVBD(
            model=model,
            iterations=iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=tile_solve,
        )
    setup_seconds = setup_timer.elapsed * 1.0e-3

    warmup_seconds = 0.0
    warmup_positions = None
    warmup_velocities = None
    if warmup:
        state_in, state_out, control = _make_vbd_state(model, scene)
        with wp.ScopedTimer("vbd-warmup", print=False, synchronize=True) as timer:
            solver.step(state_in, state_out, control, None, scene.dt)
        warmup_seconds = timer.elapsed * 1.0e-3
        warmup_positions = state_out.particle_q.numpy().astype(np.float64)
        warmup_velocities = state_out.particle_qd.numpy().astype(np.float64)

    repeat_seconds: list[float] = []
    transfer_seconds: list[float] = []
    reference_positions = None
    reference_velocities = None
    for _ in range(repeats):
        state_in, state_out, control = _make_vbd_state(model, scene)
        with wp.ScopedTimer("vbd-solve", print=False, synchronize=True) as timer:
            solver.step(state_in, state_out, control, None, scene.dt)
        repeat_seconds.append(timer.elapsed * 1.0e-3)
        transfer_start = time.perf_counter()
        positions = state_out.particle_q.numpy().astype(np.float64)
        velocities = state_out.particle_qd.numpy().astype(np.float64)
        transfer_seconds.append(time.perf_counter() - transfer_start)
        if reference_positions is None:
            reference_positions = positions
            reference_velocities = velocities
        else:
            np.testing.assert_array_equal(positions, reference_positions)
            np.testing.assert_array_equal(velocities, reference_velocities)
    if warmup_positions is not None:
        np.testing.assert_array_equal(reference_positions, warmup_positions)
        np.testing.assert_array_equal(reference_velocities, warmup_velocities)

    result = VBDRunResult(
        positions=reference_positions,
        velocities=reference_velocities,
        iterations=iterations,
        requested_tile_solve=tile_solve,
        effective_tile_solve=bool(tile_solve and model.device.is_cuda),
        color_group_count=len(model.particle_color_groups),
        device=str(model.device),
        setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        repeat_seconds=tuple(repeat_seconds),
        transfer_seconds=tuple(transfer_seconds),
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        physical_state_sha256=_array_digest(scene.x_current),
        iterate_zero_sha256=_array_digest(iterate_zero),
        result_state_sha256=_array_digest(reference_positions),
        run_sha256="",
    )
    return dataclasses.replace(result, run_sha256=_vbd_run_digest(result))


@dataclasses.dataclass(frozen=True, eq=False)
class NewtonRunResult:
    """Repeated Newton solve plus independent reference-acceptance evidence."""

    result: NewtonResult
    config: NewtonConfig
    warmup_seconds: float
    repeat_seconds: tuple[float, ...]
    scene_sha256: str
    objective_instance_sha256: str
    physical_state_sha256: str
    iterate_zero_sha256: str
    result_state_sha256: str
    verification_displacement_relative: float
    verification_converged: bool
    verification_reason: str
    alternate_start_displacement_relative: float
    alternate_start_converged: bool
    alternate_start_reason: str
    alternate_start_gradient_norm: float
    alternate_start_relative_residual: float
    reference_accepted: bool
    reference_failures: tuple[str, ...]
    run_sha256: str

    def __post_init__(self) -> None:
        safe_result = dataclasses.replace(self.result, x=self.result.x.detach().clone())
        object.__setattr__(self, "result", safe_result)

    @property
    def median_solve_seconds(self) -> float:
        return statistics.median(self.repeat_seconds)


def _newton_run_digest(run: NewtonRunResult) -> str:
    result_scalars = {
        field.name: getattr(run.result, field.name)
        for field in dataclasses.fields(run.result)
        if field.name not in ("x", "trace")
    }
    result_scalars["trace"] = [dataclasses.asdict(item) for item in run.result.trace]
    payload = {
        "result_state_sha256": _array_digest(run.result.x.detach().numpy()),
        "result": result_scalars,
        "config": dataclasses.asdict(run.config),
        "warmup_seconds": run.warmup_seconds,
        "repeat_seconds": list(run.repeat_seconds),
        "scene_sha256": run.scene_sha256,
        "objective_instance_sha256": run.objective_instance_sha256,
        "physical_state_sha256": run.physical_state_sha256,
        "iterate_zero_sha256": run.iterate_zero_sha256,
        "bound_result_state_sha256": run.result_state_sha256,
        "verification_displacement_relative": run.verification_displacement_relative,
        "verification_converged": run.verification_converged,
        "verification_reason": run.verification_reason,
        "alternate_start_displacement_relative": run.alternate_start_displacement_relative,
        "alternate_start_converged": run.alternate_start_converged,
        "alternate_start_reason": run.alternate_start_reason,
        "alternate_start_gradient_norm": run.alternate_start_gradient_norm,
        "alternate_start_relative_residual": run.alternate_start_relative_residual,
        "reference_accepted": run.reference_accepted,
        "reference_failures": list(run.reference_failures),
    }
    return _canonical_digest(payload)


def run_newton(
    scene: TetBenchmarkScene,
    problem: NewtonProblem | None = None,
    *,
    config: NewtonConfig | None = None,
    warmup: bool = True,
    repeats: int = 5,
) -> NewtonRunResult:
    """Run warm repeated Newton solves from VBD's exact iterate zero.

    The candidate is independently checked by a verification solve and by a
    second solve from ``x_current``. Reference acceptance is separate from the
    solver's native termination flag.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    common_problem = build_common_problem(scene) if problem is None else problem
    expected_objective = common_objective_manifest(scene, build_common_problem(scene))
    actual_objective = common_objective_manifest(scene, common_problem)
    if actual_objective["objective_instance_sha256"] != expected_objective["objective_instance_sha256"]:
        raise ValueError("Newton problem does not match the supplied scene")

    cfg = config or NewtonConfig(
        max_iterations=50,
        gradient_absolute_tolerance=1.0e-10,
        gradient_relative_tolerance=1.0e-10,
        step_relative_tolerance=1.0e-14,
    )
    cfg.validate()
    iterate_zero = common_problem.inertial_target.index_copy(0, common_problem.pinned, common_problem.pin_targets)

    warmup_seconds = 0.0
    if warmup:
        warmup_result = solve_newton(common_problem, iterate_zero, cfg)
        warmup_seconds = warmup_result.total_seconds

    representative = None
    repeat_seconds: list[float] = []
    for _ in range(repeats):
        current = solve_newton(common_problem, iterate_zero, cfg)
        repeat_seconds.append(current.total_seconds)
        if representative is None:
            representative = current
        else:
            torch.testing.assert_close(current.x, representative.x, rtol=0.0, atol=0.0)
            if current.reason != representative.reason:
                raise RuntimeError("repeated Newton solves returned inconsistent termination reasons")

    verification = solve_newton(common_problem, representative.x, cfg)
    alternate = solve_newton(common_problem, scene.x_current, cfg)
    free_count = int(common_problem.free.numel())
    bbox_diagonal = float(np.linalg.norm(scene.rest_q.max(axis=0) - scene.rest_q.min(axis=0)))
    displacement_scale = max(math.sqrt(free_count) * bbox_diagonal, 1.0e-30)
    verification_displacement = float(torch.linalg.vector_norm(verification.x - representative.x))
    alternate_displacement = float(torch.linalg.vector_norm(alternate.x - representative.x))
    verification_relative = verification_displacement / displacement_scale
    alternate_relative = alternate_displacement / displacement_scale
    metrics = evaluate_common_state(common_problem, representative.x)
    alternate_metrics = evaluate_common_state(common_problem, alternate.x)

    failures = []
    residual_limit = max(1.0e-10, 1.0e-10 * common_problem.residual_scale)
    if not representative.converged:
        failures.append(f"native termination: {representative.reason}")
    if metrics.gradient_norm > residual_limit:
        failures.append(f"independent gradient {metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N")
    if not verification.converged:
        failures.append(f"verification termination: {verification.reason}")
    if verification_relative > 1.0e-12:
        failures.append(f"verification displacement {verification_relative:.3e} exceeds 1e-12")
    if alternate_relative > 1.0e-9:
        failures.append(f"alternate-start displacement {alternate_relative:.3e} exceeds 1e-9")
    if alternate_metrics.gradient_norm > residual_limit:
        failures.append(
            f"alternate-start gradient {alternate_metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N"
        )
    if metrics.inverted_tet_fraction != 0.0:
        failures.append("reference contains inverted tetrahedra")

    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(actual_objective["objective_instance_sha256"])
    run = NewtonRunResult(
        result=representative,
        config=cfg,
        warmup_seconds=warmup_seconds,
        repeat_seconds=tuple(repeat_seconds),
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        physical_state_sha256=_array_digest(scene.x_current),
        iterate_zero_sha256=_array_digest(iterate_zero.detach().numpy()),
        result_state_sha256=_array_digest(representative.x.detach().numpy()),
        verification_displacement_relative=verification_relative,
        verification_converged=verification.converged,
        verification_reason=verification.reason,
        alternate_start_displacement_relative=alternate_relative,
        alternate_start_converged=alternate.converged,
        alternate_start_reason=alternate.reason,
        alternate_start_gradient_norm=alternate_metrics.gradient_norm,
        alternate_start_relative_residual=alternate_metrics.relative_residual,
        reference_accepted=not failures,
        reference_failures=tuple(failures),
        run_sha256="",
    )
    return dataclasses.replace(run, run_sha256=_newton_run_digest(run))


@dataclasses.dataclass(frozen=True, eq=False)
class SparseNewtonRunResult:
    """Repeated exact sparse Newton solve with authenticated CPU evidence."""

    result: SparseNewtonResult
    verification_result: SparseNewtonResult = dataclasses.field(repr=False, compare=False)
    alternate_start_result: SparseNewtonResult = dataclasses.field(repr=False, compare=False)
    config: NewtonConfig
    warmup_seconds: float
    repeat_seconds: tuple[float, ...]
    repeat_deterministic_sha256: tuple[str, ...]
    scene_sha256: str
    objective_instance_sha256: str
    physical_state_sha256: str
    iterate_zero_sha256: str
    result_state_sha256: str
    verification_displacement_relative: float
    verification_converged: bool
    verification_reason: str
    alternate_start_displacement_relative: float
    alternate_start_converged: bool
    alternate_start_reason: str
    alternate_start_gradient_norm: float
    alternate_start_relative_residual: float
    reference_accepted: bool
    reference_failures: tuple[str, ...]
    run_sha256: str

    def __post_init__(self) -> None:
        self.config.validate()
        for name in ("result", "verification_result", "alternate_start_result"):
            result = getattr(self, name)
            if type(result) is not SparseNewtonResult:
                raise TypeError(f"{name} must be an exact SparseNewtonResult")
            object.__setattr__(self, name, dataclasses.replace(result, positions=result.positions))
        object.__setattr__(self, "repeat_seconds", tuple(self.repeat_seconds))
        object.__setattr__(self, "repeat_deterministic_sha256", tuple(self.repeat_deterministic_sha256))
        object.__setattr__(self, "reference_failures", tuple(self.reference_failures))
        if not math.isfinite(self.warmup_seconds) or self.warmup_seconds < 0.0:
            raise ValueError("sparse Newton warmup time must be finite and non-negative")
        if not self.repeat_seconds or any(not math.isfinite(value) or value < 0.0 for value in self.repeat_seconds):
            raise ValueError("sparse Newton repeat times must be finite and non-negative")
        if len(self.repeat_deterministic_sha256) != len(self.repeat_seconds):
            raise ValueError("sparse Newton deterministic repeat evidence has the wrong length")
        expected_repeat_sha256 = _canonical_digest(self.result.deterministic_record())
        for value in self.repeat_deterministic_sha256:
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("sparse Newton deterministic repeat identity must be a SHA-256")
            if value != expected_repeat_sha256:
                raise ValueError("sparse Newton repeats changed deterministic evidence")
        evidence_scalars = (
            self.verification_displacement_relative,
            self.alternate_start_displacement_relative,
            self.alternate_start_gradient_norm,
            self.alternate_start_relative_residual,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in evidence_scalars):
            raise ValueError("sparse Newton verification evidence must be finite and non-negative")
        if self.reference_accepted != (not self.reference_failures):
            raise ValueError("sparse Newton acceptance must agree with its failures")
        if (
            self.verification_converged != self.verification_result.converged
            or self.verification_reason != self.verification_result.reason
        ):
            raise ValueError("sparse Newton verification summary changed its result")
        if (
            self.alternate_start_converged != self.alternate_start_result.converged
            or self.alternate_start_reason != self.alternate_start_result.reason
        ):
            raise ValueError("sparse Newton alternate-start summary changed its result")
        for name in (
            "scene_sha256",
            "objective_instance_sha256",
            "physical_state_sha256",
            "iterate_zero_sha256",
            "result_state_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.result_state_sha256 != _array_digest(self.result.positions):
            raise ValueError("sparse Newton result-state hash is invalid")
        if self.run_sha256 and self.run_sha256 != _sparse_newton_run_digest(self):
            raise ValueError("sparse Newton run digest is invalid")

    @property
    def median_solve_seconds(self) -> float:
        """Median sparse solve time [s], excluding warmup and problem setup."""
        return statistics.median(self.repeat_seconds)


def _sparse_newton_run_digest(run: SparseNewtonRunResult) -> str:
    payload = {
        "result": run.result.deterministic_record(),
        "result_timing": run.result.timing_record(),
        "verification_result": run.verification_result.deterministic_record(),
        "verification_result_timing": run.verification_result.timing_record(),
        "alternate_start_result": run.alternate_start_result.deterministic_record(),
        "alternate_start_result_timing": run.alternate_start_result.timing_record(),
        "config": dataclasses.asdict(run.config),
        "warmup_seconds": run.warmup_seconds,
        "repeat_seconds": list(run.repeat_seconds),
        "repeat_deterministic_sha256": list(run.repeat_deterministic_sha256),
        "scene_sha256": run.scene_sha256,
        "objective_instance_sha256": run.objective_instance_sha256,
        "physical_state_sha256": run.physical_state_sha256,
        "iterate_zero_sha256": run.iterate_zero_sha256,
        "bound_result_state_sha256": run.result_state_sha256,
        "verification_displacement_relative": run.verification_displacement_relative,
        "verification_converged": run.verification_converged,
        "verification_reason": run.verification_reason,
        "alternate_start_displacement_relative": run.alternate_start_displacement_relative,
        "alternate_start_converged": run.alternate_start_converged,
        "alternate_start_reason": run.alternate_start_reason,
        "alternate_start_gradient_norm": run.alternate_start_gradient_norm,
        "alternate_start_relative_residual": run.alternate_start_relative_residual,
        "reference_accepted": run.reference_accepted,
        "reference_failures": list(run.reference_failures),
    }
    return _canonical_digest(payload)


def _sparse_newton_trace_failures(
    result: SparseNewtonResult,
    config: NewtonConfig,
    role: str,
) -> list[str]:
    """Authenticate sparse heuristic, factor certificate, solve, and Armijo evidence."""
    failures: list[str] = []
    if result.residual_scale <= 0.0:
        failures.append(f"{role} residual scale is invalid")
    for item_index, item in enumerate(result.trace):
        if not item.has_accepted_outgoing_step:
            continue
        required = (
            item.minimum_eigenvalue,
            item.eigenpair_residual,
            item.diagonal_scale,
            item.ritz_regularization,
            item.regularization,
            item.last_attempted_regularization,
            item.factor_nnz,
            item.factor_l_unit_diagonal_error,
            item.factor_minimum_diagonal,
            item.factor_maximum_diagonal_magnitude,
            item.factor_minimum_diagonal_relative,
            item.factor_relation_relative_residual,
            item.factorization_relative_residual,
            item.linear_relative_residual,
            item.directional_derivative,
        )
        if any(value is None for value in required):
            failures.append(f"{role} iteration {item.iteration} lacks linear-solve evidence")
            continue
        if item.eigenpair_residual < 0.0 or item.diagonal_scale < 1.0 or item.regularization < 0.0:
            failures.append(f"{role} iteration {item.iteration} has invalid Ritz heuristic evidence")
        expected_ritz_regularization = max(
            0.0,
            config.minimum_eigenvalue_relative * item.diagonal_scale - item.minimum_eigenvalue,
        )
        shift_roundoff = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(expected_ritz_regularization))
        if abs(item.ritz_regularization - expected_ritz_regularization) > shift_roundoff:
            failures.append(f"{role} iteration {item.iteration} changed its Ritz shift heuristic")
        if item.regularization + shift_roundoff < item.ritz_regularization:
            failures.append(f"{role} iteration {item.iteration} used less than its Ritz heuristic shift")
        if item.last_attempted_regularization != item.regularization:
            failures.append(f"{role} iteration {item.iteration} changed its last attempted regularization")
        if item.gershgorin_rescue_used:
            if item.gershgorin_lower_bound is None or item.gershgorin_rescue_regularization is None:
                failures.append(f"{role} iteration {item.iteration} lacks Gershgorin rescue evidence")
            else:
                expected_rescue = max(
                    0.0,
                    config.minimum_eigenvalue_relative * item.diagonal_scale - item.gershgorin_lower_bound,
                )
                rescue_roundoff = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(expected_rescue))
                if abs(item.gershgorin_rescue_regularization - expected_rescue) > rescue_roundoff:
                    failures.append(f"{role} iteration {item.iteration} changed its Gershgorin rescue seed")
                if item.regularization + rescue_roundoff < item.gershgorin_rescue_regularization:
                    failures.append(f"{role} iteration {item.iteration} did not attempt its Gershgorin rescue")
        elif item.gershgorin_lower_bound is not None or item.gershgorin_rescue_regularization is not None:
            failures.append(f"{role} iteration {item.iteration} has unrequested Gershgorin evidence")
        if (
            item.factor_nnz <= 0
            or item.factorization_attempts < 1
            or item.factor_certificate_attempts < 1
            or item.factor_certificate_attempts > item.factorization_attempts
            or item.linear_solve_attempts < 1
            or item.linear_solve_attempts > item.factor_certificate_attempts
        ):
            failures.append(f"{role} iteration {item.iteration} has invalid factorization evidence")
        pivot_scale = max(item.diagonal_scale, item.factor_maximum_diagonal_magnitude, 1.0)
        expected_pivot_relative = item.factor_minimum_diagonal / pivot_scale
        pivot_roundoff = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(expected_pivot_relative))
        if (
            item.factor_certificate_passed is not True
            or item.factor_permutations_match is not True
            or item.factor_l_unit_diagonal_error < 0.0
            or item.factor_l_unit_diagonal_error > SPARSE_FACTOR_UNIT_DIAGONAL_LIMIT
            or item.factor_maximum_diagonal_magnitude < 0.0
            or item.factor_minimum_diagonal <= 0.0
            or abs(item.factor_minimum_diagonal_relative - expected_pivot_relative) > pivot_roundoff
            or item.factor_minimum_diagonal_relative <= SPARSE_FACTOR_PIVOT_RELATIVE_MARGIN
            or item.factor_relation_relative_residual < 0.0
            or item.factor_relation_relative_residual > SPARSE_FACTOR_RELATION_RELATIVE_LIMIT
            or item.factorization_relative_residual < 0.0
            or item.factorization_relative_residual > SPARSE_FACTORIZATION_RELATIVE_RESIDUAL_LIMIT
        ):
            failures.append(f"{role} iteration {item.iteration} failed its symmetric factor certificate")
        if (
            item.linear_relative_residual < 0.0
            or item.linear_relative_residual > SPARSE_LINEAR_RESIDUAL_LIMIT
            or item.linear_refinement_steps > SPARSE_MAX_REFINEMENT_STEPS
        ):
            failures.append(f"{role} iteration {item.iteration} failed the linear residual gate")

        expected_step = 1.0
        expected_trials = None
        for trial in range(1, config.max_line_search_steps + 1):
            if item.accepted_step_size == expected_step:
                expected_trials = trial
                break
            expected_step *= config.backtrack
        if expected_trials is None or item.line_search_trials != expected_trials:
            failures.append(f"{role} iteration {item.iteration} has invalid backtracking evidence")
        if item_index + 1 >= len(result.trace):
            failures.append(f"{role} iteration {item.iteration} lacks its accepted endpoint")
            continue
        next_objective = result.trace[item_index + 1].objective
        armijo_limit = item.objective + config.armijo * item.accepted_step_size * item.directional_derivative
        armijo_roundoff = (
            16.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(next_objective),
                abs(armijo_limit),
            )
        )
        if next_objective > armijo_limit + armijo_roundoff:
            failures.append(f"{role} iteration {item.iteration} violates Armijo")
    return failures


def run_sparse_newton(
    scene: TetBenchmarkScene,
    problem: NewtonProblem | None = None,
    *,
    config: NewtonConfig | None = None,
    warmup: bool = True,
    repeats: int = 5,
) -> SparseNewtonRunResult:
    """Run and independently authenticate repeated exact sparse CPU Newton.

    SciPy is imported lazily by :func:`solve_sparse_newton`. The adapter uses
    the same iterate zero, convergence thresholds, alternate start, and
    independent common-objective checks as :func:`run_newton`.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    common_problem = build_common_problem(scene) if problem is None else problem
    expected_objective = common_objective_manifest(scene, build_common_problem(scene))
    actual_objective = common_objective_manifest(scene, common_problem)
    if actual_objective["objective_instance_sha256"] != expected_objective["objective_instance_sha256"]:
        raise ValueError("sparse Newton problem does not match the supplied scene")

    cfg = config or NewtonConfig(
        max_iterations=50,
        gradient_absolute_tolerance=1.0e-10,
        gradient_relative_tolerance=1.0e-10,
        step_relative_tolerance=1.0e-14,
    )
    cfg.validate()
    iterate_zero = (
        common_problem.inertial_target.index_copy(0, common_problem.pinned, common_problem.pin_targets).detach().numpy()
    )

    warmup_seconds = 0.0
    if warmup:
        warmup_result = solve_sparse_newton(common_problem, iterate_zero, cfg)
        warmup_seconds = warmup_result.total_seconds

    representative = None
    representative_record = None
    repeat_seconds: list[float] = []
    repeat_deterministic_sha256: list[str] = []
    for _ in range(repeats):
        current = solve_sparse_newton(common_problem, iterate_zero, cfg)
        repeat_seconds.append(current.total_seconds)
        current_record = current.deterministic_record()
        repeat_deterministic_sha256.append(_canonical_digest(current_record))
        if representative is None:
            representative = current
            representative_record = current_record
        else:
            np.testing.assert_array_equal(current.positions, representative.positions)
            if current_record != representative_record:
                raise RuntimeError("repeated sparse Newton solves returned inconsistent deterministic evidence")

    verification = solve_sparse_newton(common_problem, representative.positions, cfg)
    alternate = solve_sparse_newton(common_problem, scene.x_current, cfg)
    free_count = int(common_problem.free.numel())
    bbox_diagonal = float(np.linalg.norm(scene.rest_q.max(axis=0) - scene.rest_q.min(axis=0)))
    displacement_scale = max(math.sqrt(free_count) * bbox_diagonal, 1.0e-30)
    verification_relative = (
        float(np.linalg.norm(verification.positions - representative.positions)) / displacement_scale
    )
    alternate_relative = float(np.linalg.norm(alternate.positions - representative.positions)) / displacement_scale
    metrics = evaluate_common_state(common_problem, representative.positions)
    verification_metrics = evaluate_common_state(common_problem, verification.positions)
    alternate_metrics = evaluate_common_state(common_problem, alternate.positions)

    failures = []
    residual_limit = max(1.0e-10, 1.0e-10 * common_problem.residual_scale)
    for role, result in (
        ("native", representative),
        ("verification", verification),
        ("alternate-start", alternate),
    ):
        failures.extend(_sparse_newton_trace_failures(result, cfg, role))
        if result.residual_scale != common_problem.residual_scale:
            failures.append(f"{role} residual scale changed")
    if not representative.converged:
        failures.append(f"native termination: {representative.reason}")
    if metrics.gradient_norm > residual_limit:
        failures.append(f"independent gradient {metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N")
    if not verification.converged:
        failures.append(f"verification termination: {verification.reason}")
    if verification_metrics.gradient_norm > residual_limit:
        failures.append(
            f"verification gradient {verification_metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N"
        )
    if verification_relative > 1.0e-12:
        failures.append(f"verification displacement {verification_relative:.3e} exceeds 1e-12")
    if not alternate.converged:
        failures.append(f"alternate-start termination: {alternate.reason}")
    if alternate_relative > 1.0e-9:
        failures.append(f"alternate-start displacement {alternate_relative:.3e} exceeds 1e-9")
    if alternate_metrics.gradient_norm > residual_limit:
        failures.append(
            f"alternate-start gradient {alternate_metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N"
        )
    for role, state_metrics in (
        ("reference", metrics),
        ("verification", verification_metrics),
        ("alternate-start", alternate_metrics),
    ):
        if state_metrics.inverted_tet_fraction != 0.0:
            failures.append(f"{role} contains inverted tetrahedra")
        if state_metrics.max_pin_error_m != 0.0:
            failures.append(f"{role} violates exact pins")

    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(actual_objective["objective_instance_sha256"])
    run = SparseNewtonRunResult(
        result=representative,
        verification_result=verification,
        alternate_start_result=alternate,
        config=cfg,
        warmup_seconds=warmup_seconds,
        repeat_seconds=tuple(repeat_seconds),
        repeat_deterministic_sha256=tuple(repeat_deterministic_sha256),
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        physical_state_sha256=_array_digest(scene.x_current),
        iterate_zero_sha256=_array_digest(iterate_zero),
        result_state_sha256=_array_digest(representative.positions),
        verification_displacement_relative=verification_relative,
        verification_converged=verification.converged,
        verification_reason=verification.reason,
        alternate_start_displacement_relative=alternate_relative,
        alternate_start_converged=alternate.converged,
        alternate_start_reason=alternate.reason,
        alternate_start_gradient_norm=alternate_metrics.gradient_norm,
        alternate_start_relative_residual=alternate_metrics.relative_residual,
        reference_accepted=not failures,
        reference_failures=tuple(failures),
        run_sha256="",
    )
    return dataclasses.replace(run, run_sha256=_sparse_newton_run_digest(run))


def _newton_result_deterministic_record(result: NewtonResult) -> dict[str, object]:
    """Return one ordinary Newton result without timing fields."""
    return {
        "converged": result.converged,
        "reason": result.reason,
        "accepted_iterations": result.accepted_iterations,
        "final_objective": result.final_objective,
        "final_gradient_norm": result.final_gradient_norm,
        "final_relative_residual": result.final_relative_residual,
        "work": {
            "objective_evaluations": result.objective_evaluations,
            "gradient_evaluations": result.gradient_evaluations,
            "hessian_evaluations": result.hessian_evaluations,
            "eigenvalue_evaluations": result.eigenvalue_evaluations,
            "factorization_attempts": result.factorization_attempts,
            "line_search_trials": result.line_search_trials,
        },
        "trace": [
            {
                "iteration": item.iteration,
                "objective": item.objective,
                "gradient_norm": item.gradient_norm,
                "relative_residual": item.relative_residual,
                "accepted_step_norm": item.accepted_step_norm,
                "accepted_step_size": item.accepted_step_size,
                "regularization": item.regularization,
            }
            for item in result.trace
        ],
    }


def _newton_result_timing_record(result: NewtonResult) -> dict[str, object]:
    """Return one ordinary Newton result's timing-only evidence."""
    return {
        "total_seconds": result.total_seconds,
        "objective_gradient_seconds": result.objective_gradient_seconds,
        "hessian_seconds": result.hessian_seconds,
        "linear_solve_seconds": result.linear_solve_seconds,
        "line_search_seconds": result.line_search_seconds,
        "trace_elapsed_seconds": [item.elapsed_seconds for item in result.trace],
    }


def _residual_recovery_source_record(run: NewtonRunResult) -> dict[str, object]:
    """Return deterministic zero-step-run evidence without timing fields."""
    return {
        "config": dataclasses.asdict(run.config),
        "scene_sha256": run.scene_sha256,
        "objective_instance_sha256": run.objective_instance_sha256,
        "physical_state_sha256": run.physical_state_sha256,
        "iterate_zero_sha256": run.iterate_zero_sha256,
        "result_state_sha256": run.result_state_sha256,
        "reference_accepted": run.reference_accepted,
        "reference_failures": list(run.reference_failures),
        "verification_displacement_relative": run.verification_displacement_relative,
        "verification_converged": run.verification_converged,
        "verification_reason": run.verification_reason,
        "alternate_start_displacement_relative": run.alternate_start_displacement_relative,
        "alternate_start_converged": run.alternate_start_converged,
        "alternate_start_reason": run.alternate_start_reason,
        "alternate_start_gradient_norm": run.alternate_start_gradient_norm,
        "alternate_start_relative_residual": run.alternate_start_relative_residual,
        "result": _newton_result_deterministic_record(run.result),
    }


def _common_metrics_are_finite(metrics: CommonStateMetrics) -> bool:
    return all(
        math.isfinite(value)
        for value in (
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
    )


def _validate_residual_polish_globalization(
    result: NewtonResidualPolishResult,
    config: NewtonResidualPolishConfig,
    role: str,
) -> None:
    """Authenticate one converged trace against its bound globalization config."""
    if not result.converged:
        return
    accepted_trial_count = 0
    for item, next_item in zip(result.trace, result.trace[1:], strict=False):
        step_size = 1.0
        accepted_trial_index = None
        for trial_index in range(config.max_line_search_steps):
            if item.accepted_step_size == step_size:
                accepted_trial_index = trial_index
                break
            step_size *= config.backtrack
        if accepted_trial_index is None:
            raise ValueError(f"{role} residual-polish step size is not in the iterative backtrack sequence")
        accepted_trial_count += accepted_trial_index + 1
        merit_limit = item.residual_merit + config.armijo * item.accepted_step_size * item.merit_directional_derivative
        if next_item.residual_merit > merit_limit:
            raise ValueError(f"{role} residual-polish accepted step violates Armijo")
    if result.line_search_trials != accepted_trial_count:
        raise ValueError(f"{role} residual-polish line-search trials do not match its accepted backtracks")


@dataclasses.dataclass(frozen=True, eq=False)
class NewtonResidualReferenceRecovery:
    """Three-start strict-reference recovery with authenticated evidence.

    Standalone records bind their source run and internal evidence. Selecting
    one as a committed reference additionally requires independent scene and
    problem authentication, as performed by ``pr_scene_history``.
    """

    source_run: NewtonRunResult = dataclasses.field(repr=False, compare=False)
    polish_config: NewtonResidualPolishConfig
    scene_sha256: str
    objective_instance_sha256: str
    rejected_metrics: CommonStateMetrics
    canonical: NewtonResidualPolishResult
    verification: NewtonResidualPolishResult
    alternate: NewtonResidualPolishResult
    canonical_metrics: CommonStateMetrics | None
    verification_metrics: CommonStateMetrics | None
    alternate_metrics: CommonStateMetrics | None
    displacement_scale: float
    rejected_state_sha256: str = dataclasses.field(init=False)
    verification_displacement_relative: float = dataclasses.field(init=False)
    alternate_start_displacement_relative: float = dataclasses.field(init=False)
    reference_accepted: bool = dataclasses.field(init=False)
    reference_failures: tuple[str, ...] = dataclasses.field(init=False)
    recovery_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        run = self.source_run
        run.config.validate()
        self.polish_config.validate()
        if _newton_run_digest(run) != run.run_sha256:
            raise ValueError("residual recovery source Newton run digest is invalid")
        if run.reference_accepted:
            raise ValueError("residual recovery requires a rejected Newton run")
        if run.result.converged or run.result.reason not in ("stalled", "line_search"):
            raise ValueError("residual recovery requires a stalled or line_search native Newton result")
        if run.config.step_relative_tolerance != 0.0:
            raise ValueError("residual recovery requires the zero-step-tolerance Newton retry")
        expected_polish_config = NewtonResidualPolishConfig.from_newton_config(run.config)
        if self.polish_config != expected_polish_config:
            raise ValueError("residual recovery polish config does not match its source Newton run")
        if self.scene_sha256 != run.scene_sha256:
            raise ValueError("residual recovery scene identity does not match its source Newton run")
        if self.objective_instance_sha256 != run.objective_instance_sha256:
            raise ValueError("residual recovery objective identity does not match its source Newton run")
        for name in ("scene_sha256", "objective_instance_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not math.isfinite(self.displacement_scale) or self.displacement_scale <= 0.0:
            raise ValueError("displacement_scale must be finite and positive")
        verification_relative = (
            float(torch.linalg.vector_norm(self.verification.x - self.canonical.x)) / self.displacement_scale
        )
        alternate_relative = (
            float(torch.linalg.vector_norm(self.alternate.x - self.canonical.x)) / self.displacement_scale
        )
        object.__setattr__(self, "verification_displacement_relative", verification_relative)
        object.__setattr__(self, "alternate_start_displacement_relative", alternate_relative)

        rejected_state_sha256 = _array_digest(run.result.x.detach().numpy())
        if run.result_state_sha256 != rejected_state_sha256:
            raise ValueError("residual recovery source result-state hash is invalid")
        if self.rejected_metrics.position_sha256 != rejected_state_sha256:
            raise ValueError("residual recovery rejected metrics belong to different positions")
        if not _common_metrics_are_finite(self.rejected_metrics):
            raise ValueError("residual recovery rejected metrics must be finite")
        if self.rejected_metrics.max_pin_error_m != 0.0:
            raise ValueError("residual recovery rejected state violates exact pins")
        if self.rejected_metrics.inverted_tet_fraction != 0.0:
            raise ValueError("residual recovery refuses an inverted rejected state")
        object.__setattr__(self, "rejected_state_sha256", rejected_state_sha256)

        for role in ("canonical", "verification", "alternate"):
            result = getattr(self, role)
            metrics = getattr(self, f"{role}_metrics")
            _validate_residual_polish_globalization(result, self.polish_config, role)
            if result.residual_scale != self.canonical.residual_scale:
                raise ValueError("residual recovery polish attempts changed the residual scale")
            if result.gradient_limit != self.canonical.gradient_limit:
                raise ValueError("residual recovery polish attempts changed the gradient limit")
            if metrics is not None:
                if not _common_metrics_are_finite(metrics):
                    raise ValueError(f"{role} residual recovery metrics must be finite")
                if metrics.position_sha256 != _array_digest(result.x.detach().numpy()):
                    raise ValueError(f"{role} residual recovery metrics belong to different positions")

        failures = self._selection_failures()
        object.__setattr__(self, "reference_failures", failures)
        object.__setattr__(self, "reference_accepted", not failures)
        object.__setattr__(self, "recovery_sha256", _canonical_digest(self._deterministic_payload()))

    @property
    def source_newton_config(self) -> NewtonConfig:
        """The authenticated zero-step Newton configuration."""
        return self.source_run.config

    @property
    def native_reason(self) -> str:
        """The authenticated zero-step Newton termination reason."""
        return self.source_run.result.reason

    def _selection_failures(self) -> tuple[str, ...]:
        residual_limit = max(1.0e-10, 1.0e-10 * self.canonical.residual_scale)
        failures = []
        if not self.canonical.converged:
            failures.append(f"canonical residual-polish termination: {self.canonical.reason}")
        canonical_first_eigenvalue = (
            None if not self.canonical.trace else self.canonical.trace[0].hessian_minimum_eigenvalue
        )
        if canonical_first_eigenvalue is None or not math.isfinite(canonical_first_eigenvalue):
            failures.append("canonical residual polish did not establish a finite SPD first Hessian")
        elif canonical_first_eigenvalue <= 0.0:
            failures.append("canonical residual polish first Hessian is not positive definite")
        failures.extend(self._metrics_failures("canonical", self.canonical_metrics, residual_limit))

        if not self.verification.converged:
            failures.append(f"verification residual-polish termination: {self.verification.reason}")
        if self.verification_displacement_relative > 1.0e-12:
            failures.append(f"verification displacement {self.verification_displacement_relative:.3e} exceeds 1e-12")
        failures.extend(self._metrics_failures("verification", self.verification_metrics, residual_limit))

        if not self.alternate.converged:
            failures.append(f"alternate residual-polish termination: {self.alternate.reason}")
        if self.alternate_start_displacement_relative > 1.0e-9:
            failures.append(
                f"alternate-start displacement {self.alternate_start_displacement_relative:.3e} exceeds 1e-9"
            )
        failures.extend(self._metrics_failures("alternate", self.alternate_metrics, residual_limit))
        return tuple(failures)

    @staticmethod
    def _metrics_failures(
        role: str,
        metrics: CommonStateMetrics | None,
        residual_limit: float,
    ) -> list[str]:
        if metrics is None:
            return [f"{role} residual polish common evaluation failed"]
        failures = []
        label = "alternate-start" if role == "alternate" else role
        if metrics.gradient_norm > residual_limit:
            failures.append(f"{label} gradient {metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N")
        if metrics.inverted_tet_fraction != 0.0:
            failures.append(f"{role} residual polish contains inverted tetrahedra")
        if metrics.max_pin_error_m != 0.0:
            failures.append(f"{role} residual polish violates exact pins")
        return failures

    @staticmethod
    def _attempt_record(
        result: NewtonResidualPolishResult,
        metrics: CommonStateMetrics | None,
    ) -> dict[str, object]:
        return {
            "position_sha256": _array_digest(result.x.detach().numpy()),
            "result": result.deterministic_record(),
            "metrics": None if metrics is None else metrics.as_dict(),
        }

    def _deterministic_payload(self) -> dict[str, object]:
        source_run_record = _residual_recovery_source_record(self.source_run)
        return {
            "method": "strict-reference-residual-newton-three-start-v1",
            "source_newton_config": dataclasses.asdict(self.source_newton_config),
            "source_run": source_run_record,
            "source_run_deterministic_sha256": _canonical_digest(source_run_record),
            "polish_config": self.polish_config.deterministic_record(),
            "native_reason": self.native_reason,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "rejected_state_sha256": self.rejected_state_sha256,
            "rejected_metrics": self.rejected_metrics.as_dict(),
            "attempts": {
                "canonical": self._attempt_record(self.canonical, self.canonical_metrics),
                "verification": self._attempt_record(self.verification, self.verification_metrics),
                "alternate": self._attempt_record(self.alternate, self.alternate_metrics),
            },
            "displacement_scale": self.displacement_scale,
            "verification_displacement_relative": self.verification_displacement_relative,
            "alternate_start_displacement_relative": self.alternate_start_displacement_relative,
            "reference_accepted": self.reference_accepted,
            "reference_failures": list(self.reference_failures),
        }

    def deterministic_record(self) -> dict[str, object]:
        """Return the content-addressed recovery record without timings."""
        payload = self._deterministic_payload()
        payload["recovery_sha256"] = self.recovery_sha256
        return payload

    def timing_record(self) -> dict[str, object]:
        """Return all three complete timing records outside the content hash."""
        return {
            "method": "strict-reference-residual-newton-three-start-v1",
            "validated_source_run_sha256": self.source_run.run_sha256,
            "attempts": {
                "canonical": self.canonical.timing_record(),
                "verification": self.verification.timing_record(),
                "alternate": self.alternate.timing_record(),
            },
        }


def _finite_common_metrics(
    problem: NewtonProblem,
    positions: np.ndarray | torch.Tensor,
    role: str,
) -> tuple[CommonStateMetrics | None, tuple[str, ...]]:
    """Evaluate one recovery state and turn anomalies into closed failures."""
    try:
        metrics = evaluate_common_state(problem, positions)
    except (RuntimeError, ValueError) as exc:
        return None, (f"{role} common evaluation raised {type(exc).__name__}: {exc}",)
    scalar_values = (
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
    if not all(math.isfinite(value) for value in scalar_values):
        return None, (f"{role} common evaluation produced nonfinite metrics",)
    return metrics, ()


@dataclasses.dataclass(frozen=True, eq=False)
class NewtonAlternateResidualVerification:
    """Alternate-start-only residual verification of a valid representative.

    The authenticated zero-step run must already have an accepted
    representative and verification endpoint under every ordinary gate, with
    exactly one remaining failure: the independent alternate-start gradient.
    Standalone records still require scene/problem authentication before a
    history may select them.
    """

    source_run: NewtonRunResult = dataclasses.field(repr=False, compare=False)
    polish_config: NewtonResidualPolishConfig
    scene_sha256: str
    objective_instance_sha256: str
    representative_metrics: CommonStateMetrics
    source_alternate: NewtonResult = dataclasses.field(repr=False, compare=False)
    source_alternate_metrics: CommonStateMetrics
    alternate: NewtonResidualPolishResult
    repeat: NewtonResidualPolishResult
    alternate_metrics: CommonStateMetrics | None
    repeat_metrics: CommonStateMetrics | None
    displacement_scale: float
    representative_state_sha256: str = dataclasses.field(init=False)
    source_alternate_displacement_relative: float = dataclasses.field(init=False)
    alternate_displacement_relative: float = dataclasses.field(init=False)
    repeat_displacement_relative: float = dataclasses.field(init=False)
    reference_accepted: bool = dataclasses.field(init=False)
    reference_failures: tuple[str, ...] = dataclasses.field(init=False)
    verification_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        run = self.source_run
        run.config.validate()
        self.polish_config.validate()
        if _newton_run_digest(run) != run.run_sha256:
            raise ValueError("alternate residual verification source Newton run digest is invalid")
        if run.reference_accepted:
            raise ValueError("alternate residual verification requires a rejected Newton run")
        if not run.result.converged or run.result.reason != "gradient":
            raise ValueError("alternate residual verification requires a converged representative")
        if not run.verification_converged or run.verification_reason != "gradient":
            raise ValueError("alternate residual verification requires a converged ordinary verification")
        if run.verification_displacement_relative > 1.0e-12:
            raise ValueError("alternate residual verification requires the ordinary verification displacement gate")
        if run.alternate_start_converged or run.alternate_start_reason not in (
            "stalled",
            "line_search",
            "max_iterations",
        ):
            raise ValueError(
                "alternate residual verification requires a stalled, line_search, or max_iterations alternate solve"
            )
        if len(run.reference_failures) != 1 or not run.reference_failures[0].startswith("alternate-start gradient "):
            raise ValueError("alternate residual verification requires only an alternate-start gradient failure")
        if run.config.step_relative_tolerance != 0.0:
            raise ValueError("alternate residual verification requires the zero-step-tolerance Newton retry")
        expected_polish_config = NewtonResidualPolishConfig.from_newton_config(run.config)
        if self.polish_config != expected_polish_config:
            raise ValueError("alternate residual verification polish config does not match its source Newton run")
        if self.scene_sha256 != run.scene_sha256:
            raise ValueError("alternate residual verification scene identity does not match its source Newton run")
        if self.objective_instance_sha256 != run.objective_instance_sha256:
            raise ValueError("alternate residual verification objective identity does not match its source Newton run")
        if not math.isfinite(self.displacement_scale) or self.displacement_scale <= 0.0:
            raise ValueError("displacement_scale must be finite and positive")

        representative_sha256 = _array_digest(run.result.x.detach().numpy())
        if run.result_state_sha256 != representative_sha256:
            raise ValueError("alternate residual verification source result-state hash is invalid")
        if self.representative_metrics.position_sha256 != representative_sha256:
            raise ValueError("alternate residual verification representative metrics belong to different positions")
        if not _common_metrics_are_finite(self.representative_metrics):
            raise ValueError("alternate residual verification representative metrics must be finite")
        object.__setattr__(self, "representative_state_sha256", representative_sha256)

        if self.source_alternate.converged != run.alternate_start_converged:
            raise ValueError("ordinary alternate convergence does not reproduce the source Newton run")
        if self.source_alternate.reason != run.alternate_start_reason:
            raise ValueError("ordinary alternate reason does not reproduce the source Newton run")
        if not _common_metrics_are_finite(self.source_alternate_metrics):
            raise ValueError("ordinary alternate metrics must be finite")
        source_alternate_sha256 = _array_digest(self.source_alternate.x.detach().numpy())
        if self.source_alternate_metrics.position_sha256 != source_alternate_sha256:
            raise ValueError("ordinary alternate metrics belong to different positions")
        if self.source_alternate_metrics.gradient_norm != run.alternate_start_gradient_norm:
            raise ValueError("ordinary alternate gradient does not reproduce the source Newton run")
        if self.source_alternate_metrics.relative_residual != run.alternate_start_relative_residual:
            raise ValueError("ordinary alternate residual does not reproduce the source Newton run")
        if not self.source_alternate.trace:
            raise ValueError("ordinary alternate result must contain an evaluated trace")
        if len(self.source_alternate.trace) != self.source_alternate.accepted_iterations + 1:
            raise ValueError("ordinary alternate trace does not match its accepted-update count")
        if tuple(item.iteration for item in self.source_alternate.trace) != tuple(
            range(self.source_alternate.accepted_iterations + 1)
        ):
            raise ValueError("ordinary alternate trace iterations are not contiguous")
        if (
            self.source_alternate.objective_evaluations
            != self.source_alternate.gradient_evaluations + self.source_alternate.line_search_trials
            or self.source_alternate.gradient_evaluations != len(self.source_alternate.trace)
        ):
            raise ValueError("ordinary alternate objective/gradient work accounting is invalid")
        expected_hessian_evaluations = self.source_alternate.accepted_iterations + int(
            self.source_alternate.reason == "line_search"
        )
        if (
            self.source_alternate.hessian_evaluations != expected_hessian_evaluations
            or self.source_alternate.eigenvalue_evaluations != expected_hessian_evaluations
            or self.source_alternate.factorization_attempts < expected_hessian_evaluations
            or self.source_alternate.line_search_trials < self.source_alternate.accepted_iterations
        ):
            raise ValueError("ordinary alternate derivative/line-search work accounting is invalid")
        if (
            self.source_alternate.reason == "line_search"
            and self.source_alternate.line_search_trials
            < self.source_alternate.accepted_iterations + run.config.max_line_search_steps
        ):
            raise ValueError("ordinary alternate line-search exhaustion accounting is invalid")
        if (
            self.source_alternate.reason == "max_iterations"
            and self.source_alternate.accepted_iterations != run.config.max_iterations
        ):
            raise ValueError("ordinary alternate iteration-limit accounting is invalid")
        if self.source_alternate.final_objective != self.source_alternate_metrics.objective:
            raise ValueError("ordinary alternate final objective does not match independent metrics")
        if self.source_alternate.final_gradient_norm != self.source_alternate_metrics.gradient_norm:
            raise ValueError("ordinary alternate final gradient does not match independent metrics")
        if self.source_alternate.final_relative_residual != self.source_alternate_metrics.relative_residual:
            raise ValueError("ordinary alternate final residual does not match independent metrics")

        for role in ("alternate", "repeat"):
            result = getattr(self, role)
            metrics = getattr(self, f"{role}_metrics")
            _validate_residual_polish_globalization(result, self.polish_config, role)
            if result.residual_scale != self.alternate.residual_scale:
                raise ValueError("alternate residual verification attempts changed the residual scale")
            if result.gradient_limit != self.alternate.gradient_limit:
                raise ValueError("alternate residual verification attempts changed the gradient limit")
            if metrics is not None:
                if not _common_metrics_are_finite(metrics):
                    raise ValueError(f"{role} residual verification metrics must be finite")
                if metrics.position_sha256 != _array_digest(result.x.detach().numpy()):
                    raise ValueError(f"{role} residual verification metrics belong to different positions")

        source_alternate_relative = (
            float(torch.linalg.vector_norm(self.source_alternate.x - run.result.x)) / self.displacement_scale
        )
        source_relative_roundoff = (
            8.0
            * torch.finfo(torch.float64).eps
            * max(
                1.0,
                abs(source_alternate_relative),
                abs(run.alternate_start_displacement_relative),
            )
        )
        if abs(source_alternate_relative - run.alternate_start_displacement_relative) > source_relative_roundoff:
            raise ValueError("ordinary alternate displacement does not reproduce the source Newton run")
        alternate_relative = float(torch.linalg.vector_norm(self.alternate.x - run.result.x)) / self.displacement_scale
        repeat_relative = float(torch.linalg.vector_norm(self.repeat.x - self.alternate.x)) / self.displacement_scale
        object.__setattr__(self, "source_alternate_displacement_relative", source_alternate_relative)
        object.__setattr__(self, "alternate_displacement_relative", alternate_relative)
        object.__setattr__(self, "repeat_displacement_relative", repeat_relative)

        failures = self._selection_failures()
        object.__setattr__(self, "reference_failures", failures)
        object.__setattr__(self, "reference_accepted", not failures)
        object.__setattr__(self, "verification_sha256", _canonical_digest(self._deterministic_payload()))

    def _selection_failures(self) -> tuple[str, ...]:
        residual_limit = max(1.0e-10, 1.0e-10 * self.alternate.residual_scale)
        failures = []
        if self.representative_metrics.gradient_norm > residual_limit:
            failures.append(
                "representative gradient "
                f"{self.representative_metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N"
            )
        if self.representative_metrics.inverted_tet_fraction != 0.0:
            failures.append("representative contains inverted tetrahedra")
        if self.representative_metrics.max_pin_error_m != 0.0:
            failures.append("representative violates exact pins")

        if not self.alternate.converged:
            failures.append(f"alternate residual-polish termination: {self.alternate.reason}")
        alternate_first_eigenvalue = (
            None if not self.alternate.trace else self.alternate.trace[0].hessian_minimum_eigenvalue
        )
        if alternate_first_eigenvalue is None or not math.isfinite(alternate_first_eigenvalue):
            failures.append("alternate residual polish did not establish a finite SPD first Hessian")
        elif alternate_first_eigenvalue <= 0.0:
            failures.append("alternate residual polish first Hessian is not positive definite")
        failures.extend(self._metrics_failures("alternate", self.alternate_metrics, residual_limit))
        if self.alternate_displacement_relative > 1.0e-9:
            failures.append(f"alternate-start displacement {self.alternate_displacement_relative:.3e} exceeds 1e-9")

        if not self.repeat.converged:
            failures.append(f"alternate repeat termination: {self.repeat.reason}")
        failures.extend(self._metrics_failures("repeat", self.repeat_metrics, residual_limit))
        if self.repeat_displacement_relative > 1.0e-12:
            failures.append(f"alternate repeat displacement {self.repeat_displacement_relative:.3e} exceeds 1e-12")
        return tuple(failures)

    @staticmethod
    def _metrics_failures(
        role: str,
        metrics: CommonStateMetrics | None,
        residual_limit: float,
    ) -> list[str]:
        if metrics is None:
            return [f"{role} residual verification common evaluation failed"]
        failures = []
        if metrics.gradient_norm > residual_limit:
            failures.append(f"{role} gradient {metrics.gradient_norm:.3e} N exceeds {residual_limit:.3e} N")
        if metrics.inverted_tet_fraction != 0.0:
            failures.append(f"{role} residual verification contains inverted tetrahedra")
        if metrics.max_pin_error_m != 0.0:
            failures.append(f"{role} residual verification violates exact pins")
        return failures

    @staticmethod
    def _attempt_record(
        result: NewtonResidualPolishResult,
        metrics: CommonStateMetrics | None,
    ) -> dict[str, object]:
        return {
            "position_sha256": _array_digest(result.x.detach().numpy()),
            "result": result.deterministic_record(),
            "metrics": None if metrics is None else metrics.as_dict(),
        }

    def _deterministic_payload(self) -> dict[str, object]:
        source_run_record = _residual_recovery_source_record(self.source_run)
        return {
            "method": "alternate-start-only-residual-verification-v1",
            "source_newton_config": dataclasses.asdict(self.source_run.config),
            "source_run": source_run_record,
            "source_run_deterministic_sha256": _canonical_digest(source_run_record),
            "polish_config": self.polish_config.deterministic_record(),
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "representative_state_sha256": self.representative_state_sha256,
            "representative_metrics": self.representative_metrics.as_dict(),
            "ordinary_alternate": {
                "position_sha256": _array_digest(self.source_alternate.x.detach().numpy()),
                "displacement_relative": self.source_alternate_displacement_relative,
                "result": _newton_result_deterministic_record(self.source_alternate),
                "metrics": self.source_alternate_metrics.as_dict(),
            },
            "attempts": {
                "alternate": self._attempt_record(self.alternate, self.alternate_metrics),
                "repeat": self._attempt_record(self.repeat, self.repeat_metrics),
            },
            "displacement_scale": self.displacement_scale,
            "alternate_displacement_relative": self.alternate_displacement_relative,
            "repeat_displacement_relative": self.repeat_displacement_relative,
            "selected_state": "zero-step-newton-representative",
            "reference_accepted": self.reference_accepted,
            "reference_failures": list(self.reference_failures),
        }

    def deterministic_record(self) -> dict[str, object]:
        """Return content-addressed alternate-verification evidence."""
        payload = self._deterministic_payload()
        payload["verification_sha256"] = self.verification_sha256
        return payload

    def timing_record(self) -> dict[str, object]:
        """Return ordinary-alternate and both polish timing records."""
        return {
            "method": "alternate-start-only-residual-verification-v1",
            "validated_source_run_sha256": self.source_run.run_sha256,
            "ordinary_alternate": _newton_result_timing_record(self.source_alternate),
            "attempts": {
                "alternate": self.alternate.timing_record(),
                "repeat": self.repeat.timing_record(),
            },
        }


def recover_newton_reference_with_residual_polish(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    rejected_run: NewtonRunResult,
) -> NewtonResidualReferenceRecovery:
    """Try a bounded three-start stationary-root recovery for one reference.

    Eligibility is strict: the problem must be the exact scene objective, the
    rejected state must be finite, exactly pinned, and uninverted, and the
    native zero-step-tolerance retry must have stopped as ``stalled`` or
    ``line_search``. All
    existing common-gradient, displacement, pin, and inversion gates remain
    unchanged. The rejected state, reason, configuration, and identities are
    derived only from the validated run object; callers cannot relabel them.
    """
    rejected_run.config.validate()
    if _newton_run_digest(rejected_run) != rejected_run.run_sha256:
        raise ValueError("residual recovery source Newton run digest is invalid")
    if rejected_run.reference_accepted:
        raise ValueError("residual recovery requires a rejected Newton run")
    if rejected_run.result.converged or rejected_run.result.reason not in ("stalled", "line_search"):
        raise ValueError("residual recovery requires a stalled or line_search native Newton result")
    if rejected_run.config.step_relative_tolerance != 0.0:
        raise ValueError("residual recovery requires the zero-step-tolerance Newton retry")

    expected_objective = common_objective_manifest(scene, build_common_problem(scene))
    actual_objective = common_objective_manifest(scene, problem)
    if actual_objective["objective_instance_sha256"] != expected_objective["objective_instance_sha256"]:
        raise ValueError("residual-recovery problem does not match the supplied scene")
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(actual_objective["objective_instance_sha256"])
    iterate_zero = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets)
    expected_identities = {
        "scene_sha256": scene_sha256,
        "objective_instance_sha256": objective_instance_sha256,
        "physical_state_sha256": _array_digest(scene.x_current),
        "iterate_zero_sha256": _array_digest(iterate_zero.detach().numpy()),
    }
    for name, expected in expected_identities.items():
        if getattr(rejected_run, name) != expected:
            raise ValueError(f"residual recovery source Newton run changed {name}")

    rejected = rejected_run.result.x.detach().numpy().copy()
    if rejected.shape != tuple(problem.rest_q.shape) or not np.isfinite(rejected).all():
        raise ValueError("rejected Newton result must be finite and have the problem shape")
    if rejected_run.result_state_sha256 != _array_digest(rejected):
        raise ValueError("residual recovery source result-state hash is invalid")
    rejected_metrics, rejected_evaluation_failures = _finite_common_metrics(
        problem, rejected, "rejected zero-step-tolerance Newton result"
    )
    if rejected_evaluation_failures:
        raise ValueError(rejected_evaluation_failures[0])
    assert rejected_metrics is not None
    if rejected_metrics.inverted_tet_fraction != 0.0:
        raise ValueError("residual recovery refuses an inverted rejected state")

    polish_config = NewtonResidualPolishConfig.from_newton_config(rejected_run.config)
    canonical = solve_newton_residual_polish(problem, rejected, polish_config)
    verification = solve_newton_residual_polish(problem, canonical.x, polish_config)
    alternate = solve_newton_residual_polish(problem, scene.x_current, polish_config)

    canonical_metrics, _ = _finite_common_metrics(problem, canonical.x, "canonical residual polish")
    verification_metrics, _ = _finite_common_metrics(problem, verification.x, "verification residual polish")
    alternate_metrics, _ = _finite_common_metrics(problem, alternate.x, "alternate residual polish")

    free_count = int(problem.free.numel())
    bbox_diagonal = float(np.linalg.norm(scene.rest_q.max(axis=0) - scene.rest_q.min(axis=0)))
    displacement_scale = max(math.sqrt(free_count) * bbox_diagonal, 1.0e-30)
    return NewtonResidualReferenceRecovery(
        source_run=rejected_run,
        polish_config=polish_config,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        rejected_metrics=rejected_metrics,
        canonical=canonical,
        verification=verification,
        alternate=alternate,
        canonical_metrics=canonical_metrics,
        verification_metrics=verification_metrics,
        alternate_metrics=alternate_metrics,
        displacement_scale=displacement_scale,
    )


def verify_newton_alternate_start_with_residual_polish(
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    rejected_run: NewtonRunResult,
) -> NewtonAlternateResidualVerification:
    """Verify only a failed alternate start for an otherwise valid reference."""
    rejected_run.config.validate()
    if _newton_run_digest(rejected_run) != rejected_run.run_sha256:
        raise ValueError("alternate residual verification source Newton run digest is invalid")
    if rejected_run.reference_accepted:
        raise ValueError("alternate residual verification requires a rejected Newton run")
    if not rejected_run.result.converged or rejected_run.result.reason != "gradient":
        raise ValueError("alternate residual verification requires a converged representative")
    if rejected_run.config.step_relative_tolerance != 0.0:
        raise ValueError("alternate residual verification requires the zero-step-tolerance Newton retry")

    expected_objective = common_objective_manifest(scene, build_common_problem(scene))
    actual_objective = common_objective_manifest(scene, problem)
    if actual_objective["objective_instance_sha256"] != expected_objective["objective_instance_sha256"]:
        raise ValueError("alternate residual-verification problem does not match the supplied scene")
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(actual_objective["objective_instance_sha256"])
    iterate_zero = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets)
    expected_identities = {
        "scene_sha256": scene_sha256,
        "objective_instance_sha256": objective_instance_sha256,
        "physical_state_sha256": _array_digest(scene.x_current),
        "iterate_zero_sha256": _array_digest(iterate_zero.detach().numpy()),
    }
    for name, expected in expected_identities.items():
        if getattr(rejected_run, name) != expected:
            raise ValueError(f"alternate residual verification source Newton run changed {name}")
    representative = rejected_run.result.x
    if rejected_run.result_state_sha256 != _array_digest(representative.detach().numpy()):
        raise ValueError("alternate residual verification source result-state hash is invalid")
    representative_metrics = evaluate_common_state(problem, representative)

    polish_config = NewtonResidualPolishConfig.from_newton_config(rejected_run.config)
    source_alternate = solve_newton(problem, scene.x_current, rejected_run.config)
    source_alternate_metrics = evaluate_common_state(problem, source_alternate.x)
    alternate = solve_newton_residual_polish(problem, scene.x_current, polish_config)
    repeat = solve_newton_residual_polish(problem, alternate.x, polish_config)
    alternate_metrics, _ = _finite_common_metrics(problem, alternate.x, "alternate residual verification")
    repeat_metrics, _ = _finite_common_metrics(problem, repeat.x, "alternate residual verification repeat")
    free_count = int(problem.free.numel())
    bbox_diagonal = float(np.linalg.norm(scene.rest_q.max(axis=0) - scene.rest_q.min(axis=0)))
    displacement_scale = max(math.sqrt(free_count) * bbox_diagonal, 1.0e-30)
    return NewtonAlternateResidualVerification(
        source_run=rejected_run,
        polish_config=polish_config,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        representative_metrics=representative_metrics,
        source_alternate=source_alternate,
        source_alternate_metrics=source_alternate_metrics,
        alternate=alternate,
        repeat=repeat,
        alternate_metrics=alternate_metrics,
        repeat_metrics=repeat_metrics,
        displacement_scale=displacement_scale,
    )


def _newton_record(run: NewtonRunResult, metrics: CommonStateMetrics) -> dict[str, object]:
    result = run.result
    config = dataclasses.asdict(run.config)
    return {
        "method": "dense-cpu-newton-float64",
        "run_sha256": run.run_sha256,
        "config": config,
        "config_sha256": _canonical_digest(config),
        "converged": result.converged,
        "reason": result.reason,
        "reference_accepted": run.reference_accepted,
        "reference_failures": list(run.reference_failures),
        "verification_displacement_relative": run.verification_displacement_relative,
        "verification_converged": run.verification_converged,
        "verification_reason": run.verification_reason,
        "alternate_start_displacement_relative": run.alternate_start_displacement_relative,
        "alternate_start_converged": run.alternate_start_converged,
        "alternate_start_reason": run.alternate_start_reason,
        "alternate_start_gradient_norm": run.alternate_start_gradient_norm,
        "alternate_start_relative_residual": run.alternate_start_relative_residual,
        "scene_sha256": run.scene_sha256,
        "objective_instance_sha256": run.objective_instance_sha256,
        "physical_state_sha256": run.physical_state_sha256,
        "iterate_zero_sha256": run.iterate_zero_sha256,
        "result_state_sha256": run.result_state_sha256,
        "accepted_iterations": result.accepted_iterations,
        "timing_seconds": {
            "problem_setup": result.problem_setup_seconds,
            "residual_scale_setup": result.residual_scale_setup_seconds,
            "untimed_warmup_solve": run.warmup_seconds,
            "solve_repeats": list(run.repeat_seconds),
            "solve_median": run.median_solve_seconds,
            "problem_setup_plus_untimed_warmup": result.problem_setup_seconds + run.warmup_seconds,
            "steady_state_problem_setup_plus_solve_median": (result.problem_setup_seconds + run.median_solve_seconds),
            "representative_repeat_phase_breakdown": {
                "objective_gradient": result.objective_gradient_seconds,
                "hessian": result.hessian_seconds,
                "linear_solve": result.linear_solve_seconds,
                "line_search": result.line_search_seconds,
            },
        },
        "work": {
            "objective_evaluations": result.objective_evaluations,
            "gradient_evaluations": result.gradient_evaluations,
            "hessian_evaluations": result.hessian_evaluations,
            "eigenvalue_evaluations": result.eigenvalue_evaluations,
            "factorization_attempts": result.factorization_attempts,
            "line_search_trials": result.line_search_trials,
        },
        "metrics": metrics.as_dict(),
        "trace": [dataclasses.asdict(item) for item in result.trace],
    }


def common_objective_manifest(scene: TetBenchmarkScene, problem: NewtonProblem) -> dict[str, object]:
    """Describe and hash the exact scalar objective used for all methods."""
    derived_arrays = {
        "rest_q": problem.rest_q.detach().numpy(),
        "tet_indices": problem.tets.detach().numpy(),
        "shape_gradients_J": problem.J.detach().numpy(),
        "rest_volume": problem.volume.detach().numpy(),
        "mass": problem.mass.detach().numpy(),
        "mu_stored": problem.mu.detach().numpy(),
        "lambda_stored": problem.lam.detach().numpy(),
        "inertial_target": problem.inertial_target.detach().numpy(),
        "free_indices": problem.free.detach().numpy(),
        "pinned_indices": problem.pinned.detach().numpy(),
        "pin_targets": problem.pin_targets.detach().numpy(),
    }
    payload: dict[str, object] = {
        "contract": "pss-common-stable-nh-implicit-euler-v1",
        "scene_sha256": scene.manifest()["scene_sha256"],
        "integrator": "one implicit-Euler substep",
        "elastic_model": "a727e58c SolverVBD stable Neo-Hookean",
        "native_evaluator_dtype": "float64",
        "coefficient_convention": scene.metadata.get("coefficient_convention"),
        "lambda_floor": 1.0e-6,
        "retain_rest_energy_constant": True,
        "damping": 0.0,
        "contact": False,
        "dt_seconds": problem.dt,
        "residual_scale_newtons": problem.residual_scale,
        "residual_contract": "free-gradient L2 divided by max(inertial-target free-gradient L2, 1 N)",
        "derived_arrays": {
            name: {
                "dtype": array.dtype.name,
                "shape": list(array.shape),
                "sha256": _array_digest(array),
            }
            for name, array in derived_arrays.items()
        },
    }
    payload["objective_instance_sha256"] = _canonical_digest(payload)
    return payload


def _vbd_record(result: VBDRunResult, metrics: CommonStateMetrics) -> dict[str, object]:
    config = {
        "iterations": result.iterations,
        "requested_tile_solve": result.requested_tile_solve,
        "effective_tile_solve": result.effective_tile_solve,
        "device": result.device,
        "color_group_count": result.color_group_count,
    }
    return {
        "method": "solver-vbd",
        "run_sha256": result.run_sha256,
        "config": config,
        "config_sha256": _canonical_digest(config),
        "scene_sha256": result.scene_sha256,
        "objective_instance_sha256": result.objective_instance_sha256,
        "physical_state_sha256": result.physical_state_sha256,
        "iterate_zero_sha256": result.iterate_zero_sha256,
        "result_state_sha256": result.result_state_sha256,
        "iterations": result.iterations,
        "requested_tile_solve": result.requested_tile_solve,
        "effective_tile_solve": result.effective_tile_solve,
        "color_group_count": result.color_group_count,
        "device": result.device,
        "timing_seconds": {
            "setup": result.setup_seconds,
            "untimed_warmup_solve": result.warmup_seconds,
            "solve_repeats": list(result.repeat_seconds),
            "solve_median": result.median_solve_seconds,
            "setup_plus_untimed_warmup": result.setup_seconds + result.warmup_seconds,
            "transfer_repeats": list(result.transfer_seconds),
        },
        "work": {
            "sweeps": result.iterations,
            "color_passes": result.iterations * result.color_group_count,
        },
        "metrics": metrics.as_dict(),
    }


def write_benchmark_bundle(
    output_json: pathlib.Path,
    scene: TetBenchmarkScene,
    problem: NewtonProblem,
    newton_run: NewtonRunResult,
    vbd_results: Sequence[VBDRunResult],
) -> None:
    """Write self-checking JSON metadata and raw NPZ states."""
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    expected_problem = build_common_problem(scene)
    expected_objective_sha256 = str(common_objective_manifest(scene, expected_problem)["objective_instance_sha256"])
    actual_objective_sha256 = str(common_objective_manifest(scene, problem)["objective_instance_sha256"])
    if actual_objective_sha256 != expected_objective_sha256:
        raise ValueError("problem does not match the supplied scene/objective")
    if newton_run.scene_sha256 != scene_sha256 or newton_run.objective_instance_sha256 != expected_objective_sha256:
        raise ValueError("Newton result does not belong to the supplied scene/objective")
    expected_physical_state_sha256 = _array_digest(scene.x_current)
    expected_iterate_zero_sha256 = _array_digest(problem.inertial_target.detach().numpy())
    if newton_run.physical_state_sha256 != expected_physical_state_sha256:
        raise ValueError("Newton result physical input state does not match the supplied scene")
    if newton_run.iterate_zero_sha256 != expected_iterate_zero_sha256:
        raise ValueError("Newton result iterate zero does not match the common VBD inertial target")
    if newton_run.result_state_sha256 != _array_digest(newton_run.result.x.detach().numpy()):
        raise ValueError("Newton result state was modified after the bound run")
    if newton_run.run_sha256 != _newton_run_digest(newton_run):
        raise ValueError("Newton execution/configuration record was modified after the bound run")
    if not newton_run.reference_accepted:
        raise ValueError(f"Newton candidate failed reference gates: {newton_run.reference_failures}")
    for result in vbd_results:
        if result.scene_sha256 != scene_sha256 or result.objective_instance_sha256 != expected_objective_sha256:
            raise ValueError("VBD result does not belong to the supplied scene/objective")
        if result.physical_state_sha256 != expected_physical_state_sha256:
            raise ValueError("VBD result physical input state does not match the supplied scene")
        if result.iterate_zero_sha256 != expected_iterate_zero_sha256:
            raise ValueError("VBD result iterate zero does not match the common VBD inertial target")
        if result.result_state_sha256 != _array_digest(result.positions):
            raise ValueError("VBD result state was modified after the bound run")
        if result.run_sha256 != _vbd_run_digest(result):
            raise ValueError("VBD execution/configuration record was modified after the bound run")
    budgets = [result.iterations for result in vbd_results]
    if len(set(budgets)) != len(budgets):
        raise ValueError("VBD iteration budgets must be unique in one bundle")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    newton_result = newton_run.result
    evaluation_start = time.perf_counter()
    reference_metrics = evaluate_common_state(problem, newton_result.x, reference_positions=newton_result.x)
    reference_evaluation_seconds = time.perf_counter() - evaluation_start
    vbd_records = []
    arrays: dict[str, np.ndarray] = {
        "scene_rest_q": scene.rest_q,
        "scene_tet_indices": scene.tet_indices,
        "scene_tet_poses": scene.tet_poses,
        "scene_mass": scene.mass,
        "scene_particle_inv_mass": scene.particle_inv_mass,
        "scene_tet_materials": scene.tet_materials,
        "scene_tri_indices": scene.tri_indices,
        "scene_tri_poses": scene.tri_poses,
        "scene_tri_materials": scene.tri_materials,
        "scene_tri_areas": scene.tri_areas,
        "scene_particle_flags": scene.particle_flags,
        "scene_color_group_offsets": scene.color_group_offsets,
        "scene_color_group_particles": scene.color_group_particles,
        "scene_x_current": scene.x_current,
        "scene_velocity": scene.velocity,
        "scene_gravity": scene.gravity,
        "scene_external_force": scene.external_force,
        "scene_pinned_indices": scene.pinned_indices,
        "scene_pin_targets": scene.pin_targets,
        "scene_vbd_inertial_target": scene.vbd_inertial_target,
        "newton_positions": newton_result.x.detach().numpy(),
        "objective_J": problem.J.detach().numpy(),
        "objective_rest_volume": problem.volume.detach().numpy(),
        "objective_inertial_target": problem.inertial_target.detach().numpy(),
        "objective_free_indices": problem.free.detach().numpy(),
    }
    for result in vbd_results:
        evaluation_start = time.perf_counter()
        metrics = evaluate_common_state(problem, result.positions, reference_positions=newton_result.x)
        evaluation_seconds = time.perf_counter() - evaluation_start
        if metrics.position_sha256 != _array_digest(result.positions):
            raise RuntimeError("VBD common evaluator did not hash the stored raw state")
        record = _vbd_record(result, metrics)
        record["timing_seconds"]["common_evaluation"] = evaluation_seconds
        vbd_records.append(record)
        arrays[f"vbd_positions_k{result.iterations}"] = result.positions
        arrays[f"vbd_velocities_k{result.iterations}"] = result.velocities

    raw_npz = output_json.with_suffix(".npz")
    np.savez_compressed(raw_npz, **arrays)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "scene": scene.manifest(),
        "objective": common_objective_manifest(scene, problem),
        "environment": {
            "newton_revision": _git_revision(),
            "dirty_tree_sha256": _git_dirty_digest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "warp": wp.__version__,
        },
        "raw_npz": {
            "path": raw_npz.name,
            "sha256": hashlib.sha256(raw_npz.read_bytes()).hexdigest(),
        },
        "newton_reference": _newton_record(newton_run, reference_metrics),
        "vbd": vbd_records,
    }
    payload["newton_reference"]["timing_seconds"]["common_evaluation"] = reference_evaluation_seconds
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/pss-common-objective.json"))
    parser.add_argument("--dims", nargs=3, type=int, default=(1, 1, 1), metavar=("X", "Y", "Z"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vbd-iterations", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tile-solve", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    wp.init()
    scene = build_structured_cantilever_scene(dimensions=args.dims)
    problem = build_common_problem(scene)
    newton_run = run_newton(scene, problem, warmup=True, repeats=args.repeats)
    newton_result = newton_run.result
    if not newton_run.reference_accepted:
        print(
            f"Newton reference gates failed: {newton_run.reference_failures}; "
            f"relative residual {newton_result.final_relative_residual:.3e}",
            file=sys.stderr,
        )
        return 2

    results = [
        run_vbd(
            scene,
            iterations,
            device=args.device,
            tile_solve=args.tile_solve,
            warmup=True,
            repeats=args.repeats,
        )
        for iterations in args.vbd_iterations
    ]
    write_benchmark_bundle(args.out, scene, problem, newton_run, results)

    print(f"scene: {scene.name} ({scene.n_vertices} vertices, {scene.n_tets} tets)")
    print(
        f"Newton: {newton_result.accepted_iterations} updates, "
        f"residual={newton_result.final_relative_residual:.3e}, "
        f"median={newton_run.median_solve_seconds * 1.0e3:.3f} ms"
    )
    for result in results:
        metrics = evaluate_common_state(problem, result.positions, reference_positions=newton_result.x)
        print(
            f"VBD K={result.iterations:4d}: residual={metrics.relative_residual:.3e}, "
            f"free RMS={metrics.free_rms_error_m * 1.0e3:.6f} mm, "
            f"median={result.median_solve_seconds * 1.0e3:.3f} ms"
        )
    print(f"wrote {args.out} and {args.out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
