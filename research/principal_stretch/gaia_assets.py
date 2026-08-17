# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audited Gaia tetrahedral assets for common-objective solver scenes.

The large meshes remain in their upstream Gaia checkout.  This module binds a
scene to an upstream revision and file digest, parses the explicit ``Vertex``
and ``Tet`` records, applies a declared source-unit-to-metre conversion, and
uses only public Newton builders to create a :class:`TetBenchmarkScene`.

The default boundary protocol fixes a slab at the minimum of a declared axis
and distributes a declared *total* load over the opposite slab.  It is a
reproducible benchmark protocol, not a claim about the asset's original units
or intended physical experiment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import types
from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp

import newton

from .solver_benchmark import TetBenchmarkScene, scene_from_model

GAIA_REPOSITORY_URL = "https://github.com/AnkaChan/Gaia"
GAIA_SOURCE_REVISION = "c229692045465a76233f9fba9197fb22bbfb3694"


@dataclasses.dataclass(frozen=True)
class GaiaAssetSpec:
    """Pinned provenance for one Gaia asset."""

    relative_path: str
    sha256: str
    role: str


GAIA_ASSETS: Mapping[str, GaiaAssetSpec] = types.MappingProxyType(
    {
        "bunny_small": GaiaAssetSpec(
            "Data/mesh_models/t/bunny_small.t",
            "5052f098fd0eba9efa20c6dbb4a8915f50df09948a4b9d438a44976e86f9b746",
            "primary irregular validation asset",
        ),
        "Armadilo_lowres": GaiaAssetSpec(
            "Data/mesh_models/t/Armadilo_lowres.t",
            "6226e096aa61f27ec4de582fcf82d834bf2647bbfcbaefb0ba9c320d99809644",
            "primary irregular validation asset",
        ),
        "spaghetti": GaiaAssetSpec(
            "Data/mesh_models/t/spaghetti.t",
            "f056f9c677396e57d9c9ef6a654782c4711b5f82b46454b6c74d5508eeea0d9c",
            "high-resolution sliver smoke asset",
        ),
    }
)


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    dtype = array.dtype if array.dtype.byteorder == "|" else array.dtype.newbyteorder("<")
    array = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(_array_sha256(arrays[name]).encode("ascii"))
    return digest.hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed_six_volumes(vertices: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = vertices[tets[:, 1:]] - vertices[tets[:, :1]]
    determinants = np.linalg.det(np.swapaxes(edges, 1, 2))
    roundoff = 64.0 * np.finfo(np.float64).eps * np.prod(np.linalg.norm(edges, axis=2), axis=1)
    return determinants, roundoff


@dataclasses.dataclass(frozen=True, eq=False)
class GaiaTetMesh:
    """Validated, SI-scaled contents of one Gaia ``.t`` file."""

    vertices_m: np.ndarray
    tet_indices: np.ndarray
    source_vertex_ids: np.ndarray
    source_tet_ids: np.ndarray
    signed_six_volumes_m3: np.ndarray
    repaired_source_tet_ids: np.ndarray
    source_file_sha256: str
    unit_scale_m_per_source_unit: float
    original_inverted_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices_m", _readonly(self.vertices_m, np.float64))
        object.__setattr__(self, "tet_indices", _readonly(self.tet_indices, np.int64))
        object.__setattr__(self, "source_vertex_ids", _readonly(self.source_vertex_ids, np.int64))
        object.__setattr__(self, "source_tet_ids", _readonly(self.source_tet_ids, np.int64))
        object.__setattr__(self, "signed_six_volumes_m3", _readonly(self.signed_six_volumes_m3, np.float64))
        object.__setattr__(self, "repaired_source_tet_ids", _readonly(self.repaired_source_tet_ids, np.int64))

    @property
    def orientation_repaired_count(self) -> int:
        return int(self.repaired_source_tet_ids.size)

    @property
    def topology_sha256(self) -> str:
        return _payload_sha256(
            {},
            {
                "source_vertex_ids": self.source_vertex_ids,
                "source_tet_ids": self.source_tet_ids,
                "tet_indices": self.tet_indices,
            },
        )


def load_gaia_tet_mesh(
    path: str | pathlib.Path,
    *,
    unit_scale_m_per_source_unit: float,
    repair_orientation: bool = True,
    expected_file_sha256: str | None = None,
) -> GaiaTetMesh:
    """Parse and validate an explicit-record Gaia tetrahedral mesh.

    Vertex and tetrahedron record IDs may be sparse or out of order; they are
    sorted and mapped to dense Newton indices.  Inverted tetrahedra are either
    preserved or repaired deterministically by swapping local vertices 1 and
    2.  Degenerate tetrahedra and unused vertices are rejected.

    Args:
        path: Gaia ``.t`` file.
        unit_scale_m_per_source_unit: Metres per source coordinate unit.
        repair_orientation: Whether to repair negative signed volumes.
        expected_file_sha256: Optional fail-closed source-file digest.

    Returns:
        The validated, immutable mesh.
    """
    source_path = pathlib.Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not math.isfinite(unit_scale_m_per_source_unit) or unit_scale_m_per_source_unit <= 0.0:
        raise ValueError("unit_scale_m_per_source_unit must be finite and positive")
    if not isinstance(repair_orientation, bool):
        raise ValueError("repair_orientation must be a bool")

    file_sha256 = _file_sha256(source_path)
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256.lower():
        raise ValueError(
            f"source file SHA-256 mismatch for {source_path.name}: "
            f"expected {expected_file_sha256.lower()}, got {file_sha256}"
        )

    vertices: dict[int, tuple[float, float, float]] = {}
    tets: dict[int, tuple[int, int, int, int]] = {}
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source_path} is not a UTF-8 Gaia .t file") from exc
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        kind = fields[0]
        expected_fields = 5 if kind == "Vertex" else 6 if kind == "Tet" else None
        if expected_fields is None:
            raise ValueError(f"line {line_number}: expected a Vertex or Tet record, got {kind!r}")
        if len(fields) != expected_fields:
            raise ValueError(f"line {line_number}: {kind} record must contain {expected_fields} fields")
        try:
            record_id = int(fields[1], 10)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid {kind} record ID") from exc
        if record_id < 0 or record_id > np.iinfo(np.int64).max:
            raise ValueError(f"line {line_number}: {kind} record ID is outside int64 range")
        records = vertices if kind == "Vertex" else tets
        if record_id in records:
            raise ValueError(f"line {line_number}: duplicate {kind} record ID {record_id}")
        try:
            if kind == "Vertex":
                coordinates = tuple(float(field) for field in fields[2:5])
                if not np.isfinite(coordinates).all():
                    raise ValueError
                vertices[record_id] = coordinates
            else:
                indices = tuple(int(field, 10) for field in fields[2:6])
                if any(index < 0 or index > np.iinfo(np.int64).max for index in indices):
                    raise ValueError
                if len(set(indices)) != 4:
                    raise ValueError
                tets[record_id] = indices
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid {kind} values") from exc

    if not vertices or not tets:
        raise ValueError("Gaia .t file must contain at least one Vertex and Tet record")
    source_vertex_ids = np.array(sorted(vertices), dtype=np.int64)
    source_tet_ids = np.array(sorted(tets), dtype=np.int64)
    dense_index = {int(source_id): dense for dense, source_id in enumerate(source_vertex_ids)}
    vertices_m = np.array([vertices[int(source_id)] for source_id in source_vertex_ids], dtype=np.float64)
    vertices_m *= float(unit_scale_m_per_source_unit)
    if not np.isfinite(vertices_m).all():
        raise ValueError("scaled vertex coordinates must be finite")

    dense_tets = np.empty((source_tet_ids.size, 4), dtype=np.int64)
    seen_topology: set[tuple[int, int, int, int]] = set()
    for dense_tet, source_tet_id in enumerate(source_tet_ids):
        source_indices = tets[int(source_tet_id)]
        missing = [index for index in source_indices if index not in dense_index]
        if missing:
            raise ValueError(f"Tet record {source_tet_id} references missing Vertex ID {missing[0]}")
        mapped = tuple(dense_index[index] for index in source_indices)
        topology_key = tuple(sorted(mapped))
        if topology_key in seen_topology:
            raise ValueError(f"Tet record {source_tet_id} duplicates an earlier tetrahedron")
        seen_topology.add(topology_key)
        dense_tets[dense_tet] = mapped
    used = np.unique(dense_tets)
    if not np.array_equal(used, np.arange(source_vertex_ids.size)):
        raise ValueError("Gaia .t file contains vertices that are not referenced by any tetrahedron")

    original_volumes, roundoff = _signed_six_volumes(vertices_m, dense_tets)
    degenerate = np.abs(original_volumes) <= roundoff
    if np.any(degenerate):
        source_tet_id = int(source_tet_ids[np.flatnonzero(degenerate)[0]])
        raise ValueError(f"Tet record {source_tet_id} has zero or numerically degenerate rest volume")
    inverted = original_volumes < 0.0
    repaired_ids = np.empty(0, dtype=np.int64)
    if repair_orientation and np.any(inverted):
        repaired_ids = source_tet_ids[inverted].copy()
        first = dense_tets[inverted, 1].copy()
        dense_tets[inverted, 1] = dense_tets[inverted, 2]
        dense_tets[inverted, 2] = first
    signed_volumes, repaired_roundoff = _signed_six_volumes(vertices_m, dense_tets)
    if repair_orientation and np.any(signed_volumes <= repaired_roundoff):
        raise RuntimeError("deterministic orientation repair did not produce positive tetrahedra")

    return GaiaTetMesh(
        vertices_m=vertices_m,
        tet_indices=dense_tets,
        source_vertex_ids=source_vertex_ids,
        source_tet_ids=source_tet_ids,
        signed_six_volumes_m3=signed_volumes,
        repaired_source_tet_ids=repaired_ids,
        source_file_sha256=file_sha256,
        unit_scale_m_per_source_unit=float(unit_scale_m_per_source_unit),
        original_inverted_count=int(np.count_nonzero(inverted)),
    )


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def build_gaia_tet_scene(
    path: str | pathlib.Path,
    *,
    name: str,
    source_revision: str,
    source_relative_path: str,
    unit_scale_m_per_source_unit: float,
    expected_file_sha256: str | None = None,
    source_repository_url: str = GAIA_REPOSITORY_URL,
    source_license_spdx: str = "Apache-2.0",
    density: float = 1000.0,
    mu: float = 5.0e4,
    public_lambda: float = 5.0e4,
    support_axis: int = 1,
    boundary_fraction: float = 0.02,
    gravity: Sequence[float] = (0.0, -9.81, 0.0),
    total_tip_force: Sequence[float] = (10.0, 0.0, 0.0),
    dt: float = 1.0 / 360.0,
) -> TetBenchmarkScene:
    """Build a static-support, loaded Gaia common-objective scene.

    ``public_lambda`` is converted to the current stable-Neo-Hookean storage
    convention, ``stored_lambda = public_lambda + mu``.  Damping, contact,
    surface elasticity, and bending are all disabled.
    """
    if not name or not source_revision or not source_repository_url or not source_license_spdx:
        raise ValueError("name and source provenance strings must not be empty")
    relative = pathlib.PurePosixPath(source_relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("source_relative_path must be a non-traversing repository-relative path")
    if pathlib.Path(path).name != relative.name:
        raise ValueError("source_relative_path basename must match the input file")
    if isinstance(support_axis, bool) or not isinstance(support_axis, int) or support_axis not in range(3):
        raise ValueError("support_axis must be 0, 1, or 2")
    if not math.isfinite(boundary_fraction) or not 0.0 <= boundary_fraction < 0.5:
        raise ValueError("boundary_fraction must lie in [0, 0.5)")
    for scalar_name, scalar in (("density", density), ("mu", mu), ("public_lambda", public_lambda), ("dt", dt)):
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{scalar_name} must be finite and positive")
    gravity_vector = _vector3(gravity, "gravity")
    force_vector = _vector3(total_tip_force, "total_tip_force")

    mesh = load_gaia_tet_mesh(
        path,
        unit_scale_m_per_source_unit=unit_scale_m_per_source_unit,
        repair_orientation=True,
        expected_file_sha256=expected_file_sha256,
    )
    coordinates = mesh.vertices_m[:, support_axis]
    minimum = float(coordinates.min())
    maximum = float(coordinates.max())
    extent = maximum - minimum
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("the support axis must have positive extent")
    support = coordinates <= minimum + boundary_fraction * extent
    tip = (~support) & (coordinates >= maximum - boundary_fraction * extent)
    support_indices = np.flatnonzero(support).astype(np.int64)
    tip_indices = np.flatnonzero(tip).astype(np.int64)
    if support_indices.size == 0 or tip_indices.size == 0:
        raise ValueError("boundary protocol must select non-empty disjoint support and tip slabs")

    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=mesh.vertices_m,
        indices=mesh.tet_indices.reshape(-1),
        density=density,
        k_mu=mu,
        k_lambda=mu + public_lambda,
        k_damp=0.0,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        add_surface_mesh_edges=False,
        particle_radius=0.0,
    )
    active = int(newton.ParticleFlags.ACTIVE)
    for index in support_indices:
        builder.particle_flags[int(index)] = int(builder.particle_flags[int(index)]) & ~active
    builder.color()
    model = builder.finalize(device="cpu")
    model.set_gravity(gravity_vector)
    if model.particle_count != mesh.vertices_m.shape[0] or model.tet_count != mesh.tet_indices.shape[0]:
        raise ValueError("Gaia mesh lost vertices or tetrahedra in Newton's float32 public builder")

    rest_q = model.particle_q.numpy().astype(np.float64)
    external_force = np.zeros_like(rest_q)
    per_tip_force = np.asarray(force_vector / tip_indices.size, dtype=np.float32)
    external_force[tip_indices] = per_tip_force
    tet_indices = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)
    tri_indices = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
    tet_materials = model.tet_materials.numpy().reshape(-1, 3).astype(np.float64)
    mass = model.particle_mass.numpy().astype(np.float64)
    particle_flags = model.particle_flags.numpy().astype(np.int32)
    topology_sha256 = _payload_sha256({}, {"tet_indices": tet_indices, "tri_indices": tri_indices})
    material_sha256 = _payload_sha256(
        {
            "density_kg_m3": float(density),
            "mu_public_pa": float(mu),
            "lambda_public_pa": float(public_lambda),
            "lambda_stored_pa": float(mu + public_lambda),
            "tet_damping": 0.0,
        },
        {"tet_materials": tet_materials, "particle_mass": mass},
    )
    boundary_sha256 = _payload_sha256(
        {
            "protocol": "minimum-axis support slab plus opposite-axis distributed total tip load",
            "support_axis": support_axis,
            "boundary_fraction": float(boundary_fraction),
            "dt_seconds": float(np.float32(dt)),
        },
        {
            "support_indices": support_indices,
            "tip_indices": tip_indices,
            "particle_flags": particle_flags,
            "pin_targets": rest_q[support_indices],
            "gravity": np.asarray(gravity_vector, dtype=np.float32),
            "external_force": np.asarray(external_force, dtype=np.float32),
        },
    )
    return scene_from_model(
        model,
        name=name,
        source=f"{source_repository_url}@{source_revision}:{relative.as_posix()}",
        dt=dt,
        external_force=external_force,
        metadata={
            "asset_family": "Gaia tetrahedral mesh",
            "source_repository_url": source_repository_url,
            "source_revision": source_revision,
            "source_relative_path": relative.as_posix(),
            "source_file_sha256": mesh.source_file_sha256,
            "source_license_spdx": source_license_spdx,
            "unit_scale_m_per_source_unit": mesh.unit_scale_m_per_source_unit,
            "source_units_claimed": False,
            "n_source_vertices": int(mesh.vertices_m.shape[0]),
            "n_source_tets": int(mesh.tet_indices.shape[0]),
            "original_inverted_tet_count": mesh.original_inverted_count,
            "orientation_repair_rule": "swap local vertices 1 and 2 for every negative signed rest volume",
            "orientation_repaired_tet_count": mesh.orientation_repaired_count,
            "orientation_repaired_source_tet_ids_sha256": _array_sha256(mesh.repaired_source_tet_ids),
            "minimum_signed_six_volume_m3": float(mesh.signed_six_volumes_m3.min()),
            "source_topology_sha256": mesh.topology_sha256,
            "geometry_sha256": _array_sha256(rest_q),
            "topology_sha256": topology_sha256,
            "material_sha256": material_sha256,
            "boundary_sha256": boundary_sha256,
            "density_kg_m3": float(density),
            "mu_public_pa": float(mu),
            "lambda_public_pa": float(public_lambda),
            "lambda_stored_pa": float(mu + public_lambda),
            "coefficient_convention": "stored-lambda-equals-public-lambda-plus-mu",
            "support_axis": support_axis,
            "boundary_fraction": float(boundary_fraction),
            "support_vertex_count": int(support_indices.size),
            "tip_vertex_count": int(tip_indices.size),
            "gravity_m_s2": gravity_vector.tolist(),
            "declared_total_tip_force_n": force_vector.tolist(),
            "actual_float32_total_tip_force_n": np.asarray(external_force, dtype=np.float32).sum(axis=0).tolist(),
            "common_objective_adaptations": [
                "set tetrahedral and surface damping to zero",
                "set boundary triangle energies to zero",
                "omit contact and collision shapes",
                "use one implicit step at manifest dt",
            ],
        },
    )


def build_registered_gaia_scene(
    asset_name: str,
    asset_root: str | pathlib.Path,
    *,
    unit_scale_m_per_source_unit: float,
    **scene_kwargs: object,
) -> TetBenchmarkScene:
    """Build one digest-pinned Gaia asset without vendoring it into Newton."""
    try:
        spec = GAIA_ASSETS[asset_name]
    except KeyError as exc:
        raise ValueError(f"unknown Gaia asset {asset_name!r}; expected one of {tuple(GAIA_ASSETS)}") from exc
    return build_gaia_tet_scene(
        pathlib.Path(asset_root) / pathlib.PurePosixPath(spec.relative_path),
        name=f"gaia-{asset_name}-static-support-common-step",
        source_revision=GAIA_SOURCE_REVISION,
        source_relative_path=spec.relative_path,
        unit_scale_m_per_source_unit=unit_scale_m_per_source_unit,
        expected_file_sha256=spec.sha256,
        **scene_kwargs,
    )
