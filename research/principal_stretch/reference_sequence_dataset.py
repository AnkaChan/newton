# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validated sequence-shard input for principal-stretch reference rollouts.

The adapter stops at immutable NumPy payloads.  It does not construct a
``V5TrainingSample`` and does not choose a model or optimizer.  Its job is to
preserve asset/split/sequence/step identities, authenticate the durable NPZ
files, and replay one deterministic cross-asset sampling order.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import re
import struct
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from urllib.parse import quote

import numpy as np
import torch

from .torch_solver import (
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    operator_geometry_sha256,
)
from .train_pr_history_v5 import canonical_training_tensor_sha256
from .v5_dataset import DatasetRole, canonical_topology_sha256, verify_file_sha256

REFERENCE_SEQUENCE_INDEX_CONTRACT = "pss-reference-sequence-index-v1"
REFERENCE_SEQUENCE_SAMPLING_CONTRACT = "pss-reference-asset-sequence-step-pcg64-cycle-v1"
REFERENCE_SEQUENCE_SAMPLING_GENERATOR = "numpy.random.Generator(numpy.random.PCG64(seed))"
REFERENCE_SEQUENCE_SAMPLING_ORDER = (
    "lexicographically sorted inputs; independent PCG64 shuffled cycles: asset_id -> sequence_id -> step_id"
)
REFERENCE_REQUESTED_DT_SECONDS = 1.0 / 300.0
REFERENCE_EXECUTION_DT_SECONDS = np.float32(REFERENCE_REQUESTED_DT_SECONDS)
REFERENCE_EXECUTION_DT_FLOAT32_BITS = (
    f"0x{struct.unpack('<I', struct.pack('<f', REFERENCE_EXECUTION_DT_SECONDS))[0]:08x}"
)

_STATIC_REQUIRED_ARRAY_NAMES = frozenset(
    (
        "rest_q",
        "tet_indices",
        "tet_poses",
        "mass",
        "particle_inv_mass",
        "particle_flags",
        "tet_materials",
        "boundary_triangles",
        "color_group_offsets",
        "color_group_particles",
        "hierarchy_tet_adj",
        "hierarchy_tet_c0",
        "hierarchy_tet_vol",
    )
)
_HIERARCHY_LEVEL_SUFFIXES = frozenset(("adj", "assign", "c0", "pou_idx", "pou_w", "vol"))
_HIERARCHY_LEVEL_PATTERN = re.compile(r"hierarchy_level_([0-9]+)_([a-z0-9_]+)\Z")
_SEQUENCE_ARRAY_NAMES = frozenset(
    (
        "q",
        "qd",
        "inertial_target",
        "external_force",
        "dt",
        "gravity",
        "pinned_indices",
        "pin_targets",
        "deformation_seed",
        "velocity_seed",
        "step_ids",
    )
)
_SEQUENCE_RECORD_KEYS = frozenset(
    (
        "role",
        "asset_id",
        "asset_source_sha256",
        "sequence_id",
        "topology_sha256",
        "operator_sha256",
        "material_sha256",
        "protocol_sha256",
        "producer_manifest_json",
        "producer_manifest_json_sha256",
        "step_ids",
        "reference_state_float64_sha256",
    )
)
_PRODUCER_MANIFEST_SCHEMA = "pss-free-body-reference-shard-v1"
_PRODUCER_EVIDENCE_SCHEMA = "pss-free-body-reference-evidence-v1"
_PRODUCER_IDENTITY_CONTRACT = "pss-free-body-physical-identities-v1"
_PRODUCER_MANIFEST_KEYS = frozenset(
    (
        "schema",
        "asset_id",
        "source",
        "source_sha256",
        "sequence_id",
        "deformation_seed",
        "velocity_seed",
        "reference_accepted",
        "step_count",
        "protocol",
        "identities",
        "initial_scene",
        "initial_scene_sha256",
        "normalization",
        "files",
        "inventory_sha256",
    )
)
_PRODUCER_IDENTITY_KEYS = frozenset(
    (
        "contract",
        "material_arrays",
        "material_sha256",
        "operator_arrays",
        "operator_sha256",
        "protocol_sha256",
        "topology_arrays",
        "topology_sha256",
    )
)


def reference_sequence_index_header() -> dict[str, object]:
    """Return the exact versioned header required by the split index."""
    return {
        "schema_version": 1,
        "contract": REFERENCE_SEQUENCE_INDEX_CONTRACT,
        "sampling_contract": REFERENCE_SEQUENCE_SAMPLING_CONTRACT,
        "sampling_generator": REFERENCE_SEQUENCE_SAMPLING_GENERATOR,
        "sampling_order": REFERENCE_SEQUENCE_SAMPLING_ORDER,
        "requested_dt_seconds": REFERENCE_REQUESTED_DT_SECONDS,
        "execution_dt_seconds": float(REFERENCE_EXECUTION_DT_SECONDS),
        "execution_dt_float32_bits": REFERENCE_EXECUTION_DT_FLOAT32_BITS,
    }


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _dataset_role(value: DatasetRole | str) -> DatasetRole:
    try:
        return DatasetRole(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"role must be exactly one of {tuple(role.value for role in DatasetRole)}") from exc


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(_thaw_json(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"reference sequence index contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_object(path: pathlib.Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=_json_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read canonical reference sequence index {path}") from exc
    if type(payload) is not dict:
        raise ValueError("reference sequence index root must be a JSON object")
    return payload


def _relative_artifact_path(root: pathlib.Path, value: object, name: str) -> tuple[str, pathlib.Path]:
    relative = _identifier(value, name)
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or relative != pure.as_posix()
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in relative
    ):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    resolved = (root / pathlib.Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} resolves outside the index directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} does not name an existing file")
    return relative, resolved


def _require_array(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype,
    shape: tuple[int | None, ...],
    finite: bool = False,
) -> None:
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype.name}")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def canonical_reference_state_float64_sha256(positions: np.ndarray) -> str:
    """Hash one finite float64 ``(V, 3)`` label with the v5 tensor contract."""
    canonical = np.ascontiguousarray(np.asarray(positions))
    _require_array(
        canonical,
        name="reference positions",
        dtype=np.dtype(np.float64),
        shape=(None, 3),
        finite=True,
    )
    if canonical.shape[0] == 0:
        raise ValueError("reference positions must contain at least one vertex")
    owned = np.array(canonical, dtype=np.float64, order="C", copy=True)
    return canonical_training_tensor_sha256(torch.from_numpy(owned))


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    owned = np.array(value, order="C", copy=True)
    owned.setflags(write=False)
    return owned


@dataclasses.dataclass(frozen=True, order=True)
class ReferenceTransitionKey:
    """Stable identity of one ``q[k], qd[k] -> q[k+1]`` transition."""

    asset_id: str
    sequence_id: str
    step_id: int

    def __post_init__(self) -> None:
        _identifier(self.asset_id, "transition asset_id")
        _identifier(self.sequence_id, "transition sequence_id")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("transition step_id must be a non-negative integer")


@dataclasses.dataclass(frozen=True)
class _ProducerArtifact:
    path: pathlib.Path
    sha256: str
    byte_count: int
    array_records: Mapping[str, object] | None


@dataclasses.dataclass(frozen=True)
class _ProducerShard:
    manifest_path: pathlib.Path
    manifest_sha256: str
    static: _ProducerArtifact
    sequence: _ProducerArtifact
    evidence: _ProducerArtifact
    protocol: Mapping[str, object]
    evidence_steps: tuple[Mapping[str, object], ...]
    deformation_seed: int
    velocity_seed: int


@dataclasses.dataclass(frozen=True)
class ReferenceSequenceProvenanceAnchor:
    """Public immutable view of authenticated sequence producer identities.

    The four state hashes use the producer array-digest algorithm over dtype,
    shape, and C-order bytes.  Calling :meth:`ReferenceSequenceDataset.provenance_anchor`
    does not materialize either NPZ shard.
    """

    dataset_index_uri: str
    dataset_index_sha256: str
    asset_id: str
    asset_source_sha256: str
    sequence_id: str
    producer_manifest_uri: str
    producer_manifest_sha256: str
    static_bundle_uri: str
    static_bundle_sha256: str
    sequence_bundle_uri: str
    sequence_bundle_sha256: str
    evidence_uri: str
    evidence_sha256: str
    protocol_sha256: str
    initial_position_sha256: str
    initial_velocity_field_sha256: str
    final_position_sha256: str
    final_velocity_field_sha256: str
    deformation_seed: int
    velocity_seed: int
    source_transition_count: int
    requested_dt_seconds: float
    dt_seconds: float
    execution_dt_float32_bits: str

    def __post_init__(self) -> None:
        for name in ("asset_id", "sequence_id"):
            _identifier(getattr(self, name), f"provenance anchor {name}")
        for name in (
            "dataset_index_sha256",
            "asset_source_sha256",
            "producer_manifest_sha256",
            "static_bundle_sha256",
            "sequence_bundle_sha256",
            "evidence_sha256",
            "protocol_sha256",
            "initial_position_sha256",
            "initial_velocity_field_sha256",
            "final_position_sha256",
            "final_velocity_field_sha256",
        ):
            _sha256(getattr(self, name), f"provenance anchor {name}")
        for name in (
            "dataset_index_uri",
            "producer_manifest_uri",
            "static_bundle_uri",
            "sequence_bundle_uri",
            "evidence_uri",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.startswith("artifact://reference-sequence/"):
                raise ValueError(f"provenance anchor {name} must be a logical reference-sequence artifact URI")
        for name in ("deformation_seed", "velocity_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"provenance anchor {name} must be an integer in [0, 2**32)")
        if type(self.source_transition_count) is not int or self.source_transition_count < 1:
            raise ValueError("provenance anchor source_transition_count must be positive")
        if (
            self.requested_dt_seconds != REFERENCE_REQUESTED_DT_SECONDS
            or self.dt_seconds != float(REFERENCE_EXECUTION_DT_SECONDS)
            or self.execution_dt_float32_bits != REFERENCE_EXECUTION_DT_FLOAT32_BITS
        ):
            raise ValueError("provenance anchor timestep differs from the registered sequence contract")


@dataclasses.dataclass(frozen=True)
class ReferenceSequenceRecord:
    """Authenticated index record for one sequence shard."""

    role: DatasetRole
    asset_id: str
    asset_source_sha256: str
    sequence_id: str
    topology_sha256: str
    operator_sha256: str
    material_sha256: str
    protocol_sha256: str
    producer_manifest_json: str
    producer_manifest_json_sha256: str
    step_ids: tuple[int, ...]
    reference_state_float64_sha256: tuple[str, ...]
    _producer: _ProducerShard

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _dataset_role(self.role))
        _identifier(self.asset_id, "sequence asset_id")
        _identifier(self.sequence_id, "sequence sequence_id")
        _identifier(self.producer_manifest_json, "sequence producer_manifest_json")
        for name in (
            "asset_source_sha256",
            "topology_sha256",
            "operator_sha256",
            "material_sha256",
            "protocol_sha256",
            "producer_manifest_json_sha256",
        ):
            _sha256(getattr(self, name), f"sequence {name}")
        if self.step_ids != tuple(range(len(self.step_ids))) or not self.step_ids:
            raise ValueError("sequence step_ids must be the non-empty contiguous range 0..S-1")
        if len(self.reference_state_float64_sha256) != len(self.step_ids):
            raise ValueError("sequence reference-state hashes must align one-to-one with step_ids")
        for step_id, digest in zip(self.step_ids, self.reference_state_float64_sha256, strict=True):
            _sha256(digest, f"sequence step {step_id} reference_state_float64_sha256")
        if type(self._producer) is not _ProducerShard:
            raise ValueError("sequence record must contain one validated producer shard")

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON fields from the split index."""
        return {
            "role": self.role.value,
            "asset_id": self.asset_id,
            "asset_source_sha256": self.asset_source_sha256,
            "sequence_id": self.sequence_id,
            "topology_sha256": self.topology_sha256,
            "operator_sha256": self.operator_sha256,
            "material_sha256": self.material_sha256,
            "protocol_sha256": self.protocol_sha256,
            "producer_manifest_json": self.producer_manifest_json,
            "producer_manifest_json_sha256": self.producer_manifest_json_sha256,
            "step_ids": list(self.step_ids),
            "reference_state_float64_sha256": list(self.reference_state_float64_sha256),
        }


@dataclasses.dataclass(frozen=True)
class ReferenceStaticData:
    """Validated static mesh arrays shared by one or more sequences."""

    topology_sha256: str
    v5_topology_sha256: str
    operator_sha256: str
    v5_operator_geometry_sha256: str
    material_sha256: str
    static_npz_sha256: str
    rest_q: np.ndarray
    tet_indices: np.ndarray
    tet_poses: np.ndarray
    mass: np.ndarray
    particle_inv_mass: np.ndarray
    particle_flags: np.ndarray
    tet_materials: np.ndarray
    boundary_triangles: np.ndarray
    color_group_offsets: np.ndarray
    color_group_particles: np.ndarray
    hierarchy_arrays: Mapping[str, np.ndarray]
    v5_source_rest_q: np.ndarray
    v5_source_tet_indices: np.ndarray
    v5_source_tet_poses: np.ndarray


@dataclasses.dataclass(frozen=True)
class ReferenceTransition:
    """One validated trainer-facing transition without trainer construction."""

    key: ReferenceTransitionKey
    role: DatasetRole
    asset_source_sha256: str
    topology_sha256: str
    operator_sha256: str
    material_sha256: str
    protocol_sha256: str
    sequence_npz_sha256: str
    reference_state_float64_sha256: str
    execution_dt_seconds: np.float32
    deformation_seed: int
    velocity_seed: int
    static: ReferenceStaticData
    x_current: np.ndarray
    velocity: np.ndarray
    x_previous: np.ndarray
    inertial_target: np.ndarray
    external_force: np.ndarray
    gravity: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray
    reference_positions: np.ndarray


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _producer_array_record(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    dtype = array.dtype if array.dtype.byteorder == "|" else array.dtype.newbyteorder("<")
    canonical = np.array(array, dtype=dtype, order="C", copy=True)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(json.dumps(canonical.shape, separators=(",", ":")).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "nbytes": int(canonical.nbytes),
        "sha256": digest.hexdigest(),
    }


def _array_inventory(value: object, name: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    result: dict[str, object] = {}
    for array_name, record in value.items():
        _identifier(array_name, f"{name} array name")
        if type(record) is not dict or set(record) != {"dtype", "shape", "nbytes", "sha256"}:
            raise ValueError(f"{name}.{array_name} must contain dtype/shape/nbytes/sha256 exactly")
        dtype = record["dtype"]
        shape = record["shape"]
        nbytes = record["nbytes"]
        if (
            type(dtype) is not str
            or type(shape) is not list
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise ValueError(f"{name}.{array_name} has invalid dtype or shape metadata")
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError(f"{name}.{array_name}.nbytes must be non-negative")
        _sha256(record["sha256"], f"{name}.{array_name}.sha256")
        result[array_name] = _freeze_json(record)
    return MappingProxyType(result)


def _producer_artifact(
    root: pathlib.Path,
    payload: object,
    *,
    name: str,
    arrays: bool,
) -> _ProducerArtifact:
    expected_keys = {"path", "bytes", "sha256", "arrays"} if arrays else {"path", "bytes", "sha256"}
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError(f"producer {name} record keys must be exactly {tuple(sorted(expected_keys))}")
    _relative, path = _relative_artifact_path(root, payload["path"], f"producer {name} path")
    byte_count = payload["bytes"]
    if type(byte_count) is not int or byte_count < 0 or path.stat().st_size != byte_count:
        raise ValueError(f"producer {name} byte count disagrees with its file")
    digest = _sha256(payload["sha256"], f"producer {name} sha256")
    inventory = _array_inventory(payload["arrays"], f"producer {name} arrays") if arrays else None
    return _ProducerArtifact(path=path, sha256=digest, byte_count=byte_count, array_records=inventory)


def _verify_self_hash(value: object, hash_field: str, name: str) -> str:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    record = dict(value)
    declared = _sha256(record.pop(hash_field, None), f"{name}.{hash_field}")
    if _canonical_digest(record) != declared:
        raise ValueError(f"{name} self-hash does not match {hash_field}")
    return declared


def _verify_producer_protocol(protocol: object) -> tuple[Mapping[str, object], str]:
    if type(protocol) is not dict:
        raise ValueError("producer protocol must be a JSON object")
    required = {
        "contract": "pss-free-body-reference-protocol-v1",
        "requested_dt_seconds": REFERENCE_REQUESTED_DT_SECONDS,
        "execution_dt_seconds": float(REFERENCE_EXECUTION_DT_SECONDS),
        "execution_dt_float32_bits": REFERENCE_EXECUTION_DT_FLOAT32_BITS,
        "boundary_condition": "free-body-no-pins",
        "translation_gauge_pin": False,
        "contact": False,
        "self_contact": False,
        "external_force": "exactly-zero",
        "tet_damping": 0.0,
    }
    for name, expected in required.items():
        if name not in protocol or not _exact_json_equal(protocol[name], expected):
            raise ValueError(f"producer protocol {name} does not match the registered free-body contract")
    gravity = protocol.get("gravity_m_s2")
    if gravity != [0.0, 0.0, 0.0]:
        raise ValueError("producer protocol gravity must be exactly zero")
    budgets = protocol.get("iteration_budgets")
    if (
        type(budgets) is not list
        or not budgets
        or any(type(value) is not int or value < 1 for value in budgets)
        or any(left >= right for left, right in itertools.pairwise(budgets))
        or protocol.get("selected_iterations") != budgets[-1]
    ):
        raise ValueError("producer protocol iteration budgets are not canonical")
    for name in ("maximum_relative_residual", "maximum_residual_ratio"):
        value = protocol.get(name)
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"producer protocol {name} must be positive and finite")
    frozen = _freeze_json(protocol)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("internal producer protocol freezing failed")
    return frozen, _canonical_digest(protocol)


def _verify_producer_identities(
    identities: object,
    *,
    protocol: Mapping[str, object],
    protocol_sha256: str,
    static_arrays: Mapping[str, object],
) -> None:
    if type(identities) is not dict or set(identities) != _PRODUCER_IDENTITY_KEYS:
        raise ValueError(f"producer identity keys must be exactly {tuple(sorted(_PRODUCER_IDENTITY_KEYS))}")
    if identities["contract"] != _PRODUCER_IDENTITY_CONTRACT:
        raise ValueError("producer physical identity contract is not registered")
    topology_names = ("boundary_triangles", "rest_q", "tet_indices")
    material_names = ("tet_materials",)
    operator_names = tuple(sorted(name for name in static_arrays if name not in material_names))
    expected_name_lists = {
        "topology_arrays": list(topology_names),
        "material_arrays": list(material_names),
        "operator_arrays": list(operator_names),
    }
    for name, expected in expected_name_lists.items():
        if identities[name] != expected:
            raise ValueError(f"producer {name} does not close over the registered static inventory")
    topology_payload = {name: static_arrays[name] for name in topology_names}
    material_payload = {
        "arrays": {name: static_arrays[name] for name in material_names},
        "density_kg_m3": protocol["density_kg_m3"],
        "linear_lame_lambda_pa": protocol["linear_lame_lambda_pa"],
        "shear_modulus_pa": protocol["shear_modulus_pa"],
        "tet_damping": protocol["tet_damping"],
        "vbd_stored_lambda_pa": protocol["vbd_stored_lambda_pa"],
    }
    topology_sha256 = _canonical_digest(topology_payload)
    operator_payload = {
        "arrays": {name: static_arrays[name] for name in operator_names},
        "topology_sha256": topology_sha256,
    }
    expected_hashes = {
        "topology_sha256": topology_sha256,
        "material_sha256": _canonical_digest(material_payload),
        "operator_sha256": _canonical_digest(operator_payload),
        "protocol_sha256": protocol_sha256,
    }
    for name, expected in expected_hashes.items():
        _sha256(identities[name], f"producer identities.{name}")
        if identities[name] != expected:
            raise ValueError(f"producer {name} does not match its canonical payload")


def _verify_producer_evidence(
    artifact: _ProducerArtifact,
    *,
    asset_id: str,
    sequence_id: str,
    protocol: Mapping[str, object],
    step_ids: tuple[int, ...],
) -> tuple[Mapping[str, object], ...]:
    verify_file_sha256(artifact.path, artifact.sha256)
    evidence = _read_json_object(artifact.path)
    if set(evidence) != {"schema", "asset_id", "sequence_id", "protocol", "steps"}:
        raise ValueError("producer evidence root has an unexpected inventory")
    if (
        evidence["schema"] != _PRODUCER_EVIDENCE_SCHEMA
        or evidence["asset_id"] != asset_id
        or evidence["sequence_id"] != sequence_id
        or not _exact_json_equal(evidence["protocol"], _thaw_json(protocol))
    ):
        raise ValueError("producer evidence identity disagrees with its shard manifest")
    steps = evidence["steps"]
    if type(steps) is not list or len(steps) != len(step_ids):
        raise ValueError("producer evidence steps disagree with the split index")
    result: list[Mapping[str, object]] = []
    for expected_step_id, step in zip(step_ids, steps, strict=True):
        if type(step) is not dict or step.get("step_id") != expected_step_id:
            raise ValueError("producer evidence step_ids must be contiguous and ordered")
        if step.get("reference_accepted") is not True or step.get("reference_failures") != []:
            raise ValueError("training split must not contain rejected reference evidence")
        if step.get("exact_velocity_commit") is not True:
            raise ValueError("producer evidence does not attest the exact SolverVBD velocity commit")
        candidates = step.get("candidates")
        budgets = protocol["iteration_budgets"]
        if (
            type(candidates) is not list
            or [candidate.get("iterations") for candidate in candidates if type(candidate) is dict] != list(budgets)
            or any(
                type(candidate) is not dict or candidate.get("fresh_restart") is not True for candidate in candidates
            )
        ):
            raise ValueError("producer evidence candidate budgets do not match the fresh-restart protocol")
        selected = candidates[-1]
        if type(selected) is not dict or step.get("selected_iterations") != selected.get("iterations"):
            raise ValueError("producer evidence selected budget is not the final candidate")
        metrics = selected.get("metrics")
        if type(metrics) is not dict or metrics.get("position_sha256") != step.get("output_position_sha256"):
            raise ValueError("producer selected metrics do not bind the output position")
        if selected.get("position_float64_sha256") != step.get("output_position_sha256"):
            raise ValueError("producer selected candidate does not bind the output position")
        if selected.get("velocity_float64_sha256") != step.get("output_velocity_sha256"):
            raise ValueError("producer selected candidate does not bind the output velocity")
        iterate_zero = step.get("iterate_zero_metrics")
        metric_names = (
            "objective",
            "inertia",
            "elastic",
            "gradient_norm",
            "relative_residual",
            "determinant_min",
            "determinant_max",
            "inverted_tet_fraction",
            "minimum_singular_value",
            "max_pin_error_m",
        )
        if type(iterate_zero) is not dict or any(
            isinstance(record.get(name), bool)
            or not isinstance(record.get(name), (int, float))
            or not math.isfinite(float(record[name]))
            for record in (iterate_zero, metrics)
            for name in metric_names
        ):
            raise ValueError("producer acceptance metrics must be finite numeric values")
        selected_residual = float(metrics["relative_residual"])
        zero_residual = float(iterate_zero["relative_residual"])
        if (
            float(metrics["determinant_min"]) <= 0.0
            or float(metrics["inverted_tet_fraction"]) != 0.0
            or float(metrics["minimum_singular_value"]) <= 0.0
            or float(metrics["max_pin_error_m"]) != 0.0
            or float(metrics["objective"]) > float(iterate_zero["objective"])
            or zero_residual <= 0.0
            or selected_residual >= zero_residual
            or selected_residual > float(protocol["maximum_relative_residual"])
            or selected_residual / zero_residual > float(protocol["maximum_residual_ratio"])
            or selected.get("relative_residual_over_iterate_zero") != selected_residual / zero_residual
        ):
            raise ValueError("producer evidence does not independently satisfy the registered acceptance gate")
        dynamic_scene = step.get("dynamic_scene_manifest")
        objective = step.get("objective_manifest")
        scene_sha256 = _verify_self_hash(dynamic_scene, "scene_sha256", "dynamic_scene_manifest")
        objective_sha256 = _verify_self_hash(objective, "objective_instance_sha256", "objective_manifest")
        if (
            step.get("dynamic_scene_sha256") != scene_sha256
            or step.get("objective_instance_sha256") != objective_sha256
        ):
            raise ValueError("producer step does not bind its scene/objective self-hashes")
        if objective.get("scene_sha256") != scene_sha256:
            raise ValueError("producer objective does not bind its dynamic scene")
        for name in (
            "input_position_sha256",
            "input_velocity_sha256",
            "inertial_target_sha256",
            "output_position_sha256",
            "output_velocity_sha256",
        ):
            _sha256(step.get(name), f"producer evidence step {expected_step_id} {name}")
        frozen = _freeze_json(step)
        if not isinstance(frozen, Mapping):
            raise RuntimeError("internal producer evidence freezing failed")
        result.append(frozen)
    return tuple(result)


def _load_producer_shard(
    manifest_path: pathlib.Path,
    manifest_sha256: str,
    *,
    asset_id: str,
    asset_source_sha256: str,
    sequence_id: str,
    topology_sha256: str,
    operator_sha256: str,
    material_sha256: str,
    protocol_sha256: str,
    step_ids: tuple[int, ...],
) -> _ProducerShard:
    verify_file_sha256(manifest_path, manifest_sha256)
    manifest = _read_json_object(manifest_path)
    if set(manifest) != _PRODUCER_MANIFEST_KEYS or manifest.get("schema") != _PRODUCER_MANIFEST_SCHEMA:
        raise ValueError("producer shard manifest has an unregistered root inventory")
    if (
        manifest["asset_id"] != asset_id
        or manifest["source_sha256"] != asset_source_sha256
        or manifest["sequence_id"] != sequence_id
        or manifest["reference_accepted"] is not True
        or manifest["step_count"] != len(step_ids)
    ):
        raise ValueError("producer shard identity disagrees with the training split index")
    _identifier(manifest["source"], "producer source")
    _sha256(manifest["source_sha256"], "producer source_sha256")
    for name in ("deformation_seed", "velocity_seed"):
        if type(manifest[name]) is not int or not 0 <= manifest[name] < 2**32:
            raise ValueError(f"producer {name} must be an integer in [0, 2**32)")
    protocol, actual_protocol_sha256 = _verify_producer_protocol(manifest["protocol"])
    files = manifest["files"]
    if type(files) is not dict or set(files) != {"static_npz", "sequence_npz", "evidence_json"}:
        raise ValueError("producer shard files inventory is not closed")
    static = _producer_artifact(manifest_path.parent, files["static_npz"], name="static_npz", arrays=True)
    sequence = _producer_artifact(manifest_path.parent, files["sequence_npz"], name="sequence_npz", arrays=True)
    evidence = _producer_artifact(manifest_path.parent, files["evidence_json"], name="evidence_json", arrays=False)
    _sha256(manifest["inventory_sha256"], "producer inventory_sha256")
    if _canonical_digest({"files": files, "identities": manifest["identities"]}) != manifest["inventory_sha256"]:
        raise ValueError("producer shard inventory_sha256 mismatch")
    if static.array_records is None:
        raise RuntimeError("producer static artifact lost its array inventory")
    _verify_producer_identities(
        manifest["identities"],
        protocol=protocol,
        protocol_sha256=actual_protocol_sha256,
        static_arrays=static.array_records,
    )
    identities = manifest["identities"]
    expected_identities = {
        "topology_sha256": topology_sha256,
        "operator_sha256": operator_sha256,
        "material_sha256": material_sha256,
        "protocol_sha256": protocol_sha256,
    }
    if any(identities[name] != expected for name, expected in expected_identities.items()):
        raise ValueError("producer physical identities disagree with the training split index")
    initial_scene_sha256 = _verify_self_hash(manifest["initial_scene"], "scene_sha256", "initial_scene")
    if initial_scene_sha256 != manifest["initial_scene_sha256"]:
        raise ValueError("producer initial_scene_sha256 mismatch")
    evidence_steps = _verify_producer_evidence(
        evidence,
        asset_id=asset_id,
        sequence_id=sequence_id,
        protocol=protocol,
        step_ids=step_ids,
    )
    return _ProducerShard(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        static=static,
        sequence=sequence,
        evidence=evidence,
        protocol=protocol,
        evidence_steps=evidence_steps,
        deformation_seed=manifest["deformation_seed"],
        velocity_seed=manifest["velocity_seed"],
    )


def _sequence_record(
    payload: object,
    *,
    role: DatasetRole,
    root: pathlib.Path,
) -> ReferenceSequenceRecord:
    if type(payload) is not dict or set(payload) != _SEQUENCE_RECORD_KEYS:
        raise ValueError(f"sequence record keys must be exactly {tuple(sorted(_SEQUENCE_RECORD_KEYS))}")
    if payload["role"] != role.value:
        raise ValueError("sequence record role must exactly match its split role")
    producer_manifest_json, producer_manifest_path = _relative_artifact_path(
        root,
        payload["producer_manifest_json"],
        "producer_manifest_json",
    )
    producer_manifest_sha256 = _sha256(
        payload["producer_manifest_json_sha256"],
        "producer_manifest_json_sha256",
    )
    step_ids_value = payload["step_ids"]
    hash_values = payload["reference_state_float64_sha256"]
    if type(step_ids_value) is not list or any(type(step_id) is not int for step_id in step_ids_value):
        raise ValueError("sequence step_ids must be a JSON integer list")
    if type(hash_values) is not list:
        raise ValueError("sequence reference_state_float64_sha256 must be a JSON list")
    step_ids = tuple(step_ids_value)
    producer = _load_producer_shard(
        producer_manifest_path,
        producer_manifest_sha256,
        asset_id=payload["asset_id"],
        asset_source_sha256=payload["asset_source_sha256"],
        sequence_id=payload["sequence_id"],
        topology_sha256=payload["topology_sha256"],
        operator_sha256=payload["operator_sha256"],
        material_sha256=payload["material_sha256"],
        protocol_sha256=payload["protocol_sha256"],
        step_ids=step_ids,
    )
    return ReferenceSequenceRecord(
        role=role,
        asset_id=payload["asset_id"],
        asset_source_sha256=payload["asset_source_sha256"],
        sequence_id=payload["sequence_id"],
        topology_sha256=payload["topology_sha256"],
        operator_sha256=payload["operator_sha256"],
        material_sha256=payload["material_sha256"],
        protocol_sha256=payload["protocol_sha256"],
        producer_manifest_json=producer_manifest_json,
        producer_manifest_json_sha256=producer_manifest_sha256,
        step_ids=step_ids,
        reference_state_float64_sha256=tuple(hash_values),
        _producer=producer,
    )


def _load_npz(
    artifact: _ProducerArtifact,
    expected_names: frozenset[str] | None,
) -> dict[str, np.ndarray]:
    path = artifact.path
    if artifact.array_records is None:
        raise RuntimeError("NPZ artifact is missing its producer array inventory")
    verify_file_sha256(path, artifact.sha256)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError(f"{path.name} contains duplicate array names")
            inventory_names = set(artifact.array_records)
            if set(archive.files) != inventory_names:
                raise ValueError(f"{path.name} arrays disagree with the producer manifest")
            if expected_names is not None and inventory_names != expected_names:
                raise ValueError(f"{path.name} arrays must be exactly {tuple(sorted(expected_names))}")
            arrays = {name: np.array(archive[name], order="C", copy=True) for name in sorted(archive.files)}
            for name, array in arrays.items():
                if not _exact_json_equal(_producer_array_record(array), _thaw_json(artifact.array_records[name])):
                    raise ValueError(f"{path.name} array {name!r} disagrees with its producer manifest record")
            return arrays
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and (
            "arrays must be exactly" in str(exc)
            or "duplicate array names" in str(exc)
            or "producer manifest" in str(exc)
        ):
            raise
        raise ValueError(f"could not load authenticated NPZ artifact {path}") from exc


def _hierarchy_level_indices(array_names: set[str]) -> tuple[int, ...]:
    if not _STATIC_REQUIRED_ARRAY_NAMES.issubset(array_names):
        missing = tuple(sorted(_STATIC_REQUIRED_ARRAY_NAMES - array_names))
        raise ValueError(f"static NPZ is missing required arrays {missing}")
    level_suffixes: dict[int, set[str]] = {}
    for name in array_names - _STATIC_REQUIRED_ARRAY_NAMES:
        match = _HIERARCHY_LEVEL_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"static NPZ contains unregistered array {name!r}")
        level_index = int(match.group(1))
        if str(level_index) != match.group(1):
            raise ValueError("hierarchy level array names must use canonical decimal indices")
        level_suffixes.setdefault(level_index, set()).add(match.group(2))
    level_indices = tuple(sorted(level_suffixes))
    if not level_indices or level_indices != tuple(range(len(level_indices))):
        raise ValueError("static NPZ hierarchy levels must be a non-empty contiguous range from zero")
    if any(suffixes != _HIERARCHY_LEVEL_SUFFIXES for suffixes in level_suffixes.values()):
        raise ValueError(f"every hierarchy level must contain exactly {tuple(sorted(_HIERARCHY_LEVEL_SUFFIXES))}")
    return level_indices


def _validated_hierarchy_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    tet_count: int,
) -> Mapping[str, np.ndarray]:
    level_indices = _hierarchy_level_indices(set(arrays))
    tet_adj = arrays["hierarchy_tet_adj"]
    tet_c0 = arrays["hierarchy_tet_c0"]
    tet_vol = arrays["hierarchy_tet_vol"]
    _require_array(tet_adj, name="hierarchy_tet_adj", dtype=np.dtype(np.int32), shape=(tet_count, 4))
    _require_array(
        tet_c0,
        name="hierarchy_tet_c0",
        dtype=np.dtype(np.float64),
        shape=(tet_count, 3),
        finite=True,
    )
    _require_array(
        tet_vol,
        name="hierarchy_tet_vol",
        dtype=np.dtype(np.float64),
        shape=(tet_count,),
        finite=True,
    )
    if np.any(tet_adj < -1) or np.any(tet_adj >= tet_count) or np.any(tet_vol <= 0.0):
        raise ValueError("tet hierarchy adjacency or volumes are invalid")

    child_count = tet_count
    for level_index in level_indices:
        prefix = f"hierarchy_level_{level_index}"
        assign = arrays[f"{prefix}_assign"]
        _require_array(assign, name=f"{prefix}_assign", dtype=np.dtype(np.int32), shape=(child_count,))
        if np.any(assign < 0):
            raise ValueError(f"{prefix}_assign contains a negative cluster")
        cluster_count = int(assign.max(initial=-1)) + 1
        if cluster_count < 1 or not np.array_equal(np.unique(assign), np.arange(cluster_count, dtype=np.int32)):
            raise ValueError(f"{prefix}_assign cluster ids must be contiguous from zero")
        adj = arrays[f"{prefix}_adj"]
        c0 = arrays[f"{prefix}_c0"]
        volume = arrays[f"{prefix}_vol"]
        pou_idx = arrays[f"{prefix}_pou_idx"]
        pou_w = arrays[f"{prefix}_pou_w"]
        _require_array(adj, name=f"{prefix}_adj", dtype=np.dtype(np.int32), shape=(cluster_count, None))
        _require_array(
            c0,
            name=f"{prefix}_c0",
            dtype=np.dtype(np.float64),
            shape=(cluster_count, 3),
            finite=True,
        )
        _require_array(
            volume,
            name=f"{prefix}_vol",
            dtype=np.dtype(np.float64),
            shape=(cluster_count,),
            finite=True,
        )
        _require_array(
            pou_idx,
            name=f"{prefix}_pou_idx",
            dtype=np.dtype(np.int32),
            shape=(child_count, None),
        )
        _require_array(
            pou_w,
            name=f"{prefix}_pou_w",
            dtype=np.dtype(np.float64),
            shape=pou_idx.shape,
            finite=True,
        )
        if (
            np.any(adj < -1)
            or np.any(adj >= cluster_count)
            or np.any(pou_idx < -1)
            or np.any(pou_idx >= cluster_count)
            or np.any(volume <= 0.0)
            or np.any(pou_w < 0.0)
            or not np.all(pou_w[pou_idx < 0] == 0.0)
            or not np.allclose(pou_w.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"{prefix} contains invalid adjacency, volume, or partition-of-unity data")
        child_count = cluster_count
    hierarchy_names = sorted(name for name in arrays if name.startswith("hierarchy_"))
    return MappingProxyType({name: _readonly_copy(arrays[name]) for name in hierarchy_names})


def _load_static(record: ReferenceSequenceRecord) -> ReferenceStaticData:
    arrays = _load_npz(record._producer.static, None)
    _hierarchy_level_indices(set(arrays))
    rest_q = arrays["rest_q"]
    tets = arrays["tet_indices"]
    tet_poses = arrays["tet_poses"]
    mass = arrays["mass"]
    particle_inv_mass = arrays["particle_inv_mass"]
    particle_flags = arrays["particle_flags"]
    materials = arrays["tet_materials"]
    boundary = arrays["boundary_triangles"]
    color_offsets = arrays["color_group_offsets"]
    color_particles = arrays["color_group_particles"]
    _require_array(rest_q, name="rest_q", dtype=np.dtype(np.float64), shape=(None, 3), finite=True)
    _require_array(tets, name="tet_indices", dtype=np.dtype(np.int32), shape=(None, 4))
    _require_array(tet_poses, name="tet_poses", dtype=np.dtype(np.float64), shape=(tets.shape[0], 3, 3), finite=True)
    _require_array(mass, name="mass", dtype=np.dtype(np.float32), shape=(rest_q.shape[0],), finite=True)
    _require_array(
        particle_inv_mass,
        name="particle_inv_mass",
        dtype=np.dtype(np.float32),
        shape=(rest_q.shape[0],),
        finite=True,
    )
    _require_array(particle_flags, name="particle_flags", dtype=np.dtype(np.int32), shape=(rest_q.shape[0],))
    _require_array(
        materials,
        name="tet_materials",
        dtype=np.dtype(np.float32),
        shape=(tets.shape[0], 3),
        finite=True,
    )
    _require_array(boundary, name="boundary_triangles", dtype=np.dtype(np.int32), shape=(None, 3))
    _require_array(color_offsets, name="color_group_offsets", dtype=np.dtype(np.int32), shape=(None,))
    _require_array(color_particles, name="color_group_particles", dtype=np.dtype(np.int32), shape=(rest_q.shape[0],))
    if rest_q.shape[0] == 0 or tets.shape[0] == 0:
        raise ValueError("static shard must contain vertices and tetrahedra")
    if np.any(tets < 0) or np.any(tets >= rest_q.shape[0]):
        raise ValueError("tet_indices contain an out-of-range vertex")
    if np.any(boundary < 0) or np.any(boundary >= rest_q.shape[0]):
        raise ValueError("boundary_triangles contain an out-of-range vertex")
    if (
        np.any(mass <= 0.0)
        or np.any(particle_inv_mass <= 0.0)
        or np.any((particle_flags & 1) == 0)
        or np.any(materials[:, :2] <= 0.0)
        or np.any(materials[:, 2] != 0.0)
        or color_offsets.size < 2
        or color_offsets[0] != 0
        or color_offsets[-1] != color_particles.size
        or np.any(color_offsets[1:] < color_offsets[:-1])
        or not np.array_equal(np.sort(color_particles), np.arange(rest_q.shape[0], dtype=np.int32))
    ):
        raise ValueError("free-body particle activity, inverse mass, or color groups are invalid")
    hierarchy = _validated_hierarchy_arrays(arrays, tet_count=tets.shape[0])
    _require_lossless_float32_promotion(rest_q, "rest_q")
    _require_lossless_float32_promotion(tet_poses, "tet_poses")
    v5_source_rest_q = rest_q.astype(np.float32)
    v5_source_tets = tets.astype(np.int64)
    v5_source_tet_poses = tet_poses.astype(np.float32)
    actual_topology = canonical_topology_sha256(v5_source_rest_q, v5_source_tets)
    v5_operator_geometry = operator_geometry_sha256(
        v5_source_rest_q,
        v5_source_tets,
        v5_source_tet_poses,
        policy=OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    )
    return ReferenceStaticData(
        topology_sha256=record.topology_sha256,
        v5_topology_sha256=actual_topology,
        operator_sha256=record.operator_sha256,
        v5_operator_geometry_sha256=v5_operator_geometry,
        material_sha256=record.material_sha256,
        static_npz_sha256=record._producer.static.sha256,
        rest_q=_readonly_copy(rest_q),
        tet_indices=_readonly_copy(tets),
        tet_poses=_readonly_copy(tet_poses),
        mass=_readonly_copy(mass),
        particle_inv_mass=_readonly_copy(particle_inv_mass),
        particle_flags=_readonly_copy(particle_flags),
        tet_materials=_readonly_copy(materials),
        boundary_triangles=_readonly_copy(boundary),
        color_group_offsets=_readonly_copy(color_offsets),
        color_group_particles=_readonly_copy(color_particles),
        hierarchy_arrays=hierarchy,
        v5_source_rest_q=_readonly_copy(v5_source_rest_q),
        v5_source_tet_indices=_readonly_copy(v5_source_tets),
        v5_source_tet_poses=_readonly_copy(v5_source_tet_poses),
    )


def _require_lossless_float32_promotion(array: np.ndarray, name: str) -> None:
    promoted = array.astype(np.float32).astype(np.float64)
    if not np.array_equal(array.view(np.uint64), promoted.view(np.uint64)):
        raise ValueError(f"{name} must be a lossless float64 promotion of float32 producer values")


def _load_sequence(record: ReferenceSequenceRecord, vertex_count: int) -> dict[str, np.ndarray]:
    arrays = _load_npz(record._producer.sequence, _SEQUENCE_ARRAY_NAMES)
    step_count = len(record.step_ids)
    q = arrays["q"]
    qd = arrays["qd"]
    inertial_target = arrays["inertial_target"]
    external_force = arrays["external_force"]
    dt = arrays["dt"]
    gravity = arrays["gravity"]
    pinned = arrays["pinned_indices"]
    pin_targets = arrays["pin_targets"]
    step_ids = arrays["step_ids"]
    _require_array(q, name="q", dtype=np.dtype(np.float64), shape=(step_count + 1, vertex_count, 3), finite=True)
    _require_array(qd, name="qd", dtype=np.dtype(np.float64), shape=(step_count + 1, vertex_count, 3), finite=True)
    _require_array(
        inertial_target,
        name="inertial_target",
        dtype=np.dtype(np.float64),
        shape=(step_count, vertex_count, 3),
        finite=True,
    )
    _require_array(
        external_force,
        name="external_force",
        dtype=np.dtype(np.float32),
        shape=(step_count, vertex_count, 3),
        finite=True,
    )
    _require_array(dt, name="dt", dtype=np.dtype(np.float32), shape=(), finite=True)
    _require_array(gravity, name="gravity", dtype=np.dtype(np.float32), shape=(3,), finite=True)
    _require_array(pinned, name="pinned_indices", dtype=np.dtype(np.int32), shape=(None,))
    _require_array(
        pin_targets,
        name="pin_targets",
        dtype=np.dtype(np.float64),
        shape=(step_count, pinned.shape[0], 3),
        finite=True,
    )
    _require_array(step_ids, name="step_ids", dtype=np.dtype(np.int64), shape=(step_count,))
    for name in ("deformation_seed", "velocity_seed"):
        _require_array(arrays[name], name=name, dtype=np.dtype(np.int64), shape=())
        if int(arrays[name]) < 0:
            raise ValueError(f"{name} must be non-negative")
    if (
        int(arrays["deformation_seed"]) != record._producer.deformation_seed
        or int(arrays["velocity_seed"]) != record._producer.velocity_seed
    ):
        raise ValueError("sequence NPZ seeds disagree with the producer manifest")
    if dt.tobytes() != np.asarray(REFERENCE_EXECUTION_DT_SECONDS, dtype=np.float32).tobytes():
        raise ValueError("dt must equal the reference execution dt float32(1/300) exactly")
    if tuple(int(value) for value in step_ids) != record.step_ids:
        raise ValueError("sequence NPZ step_ids disagree with the split index")
    if pinned.size and (np.any(pinned < 0) or np.any(pinned >= vertex_count)):
        raise ValueError("pinned_indices contain an out-of-range vertex")
    if pinned.size and (not np.all(pinned[1:] > pinned[:-1])):
        raise ValueError("pinned_indices must be unique and strictly increasing")
    if pinned.size or pin_targets.size or np.any(external_force) or np.any(gravity):
        raise ValueError("registered free-body sequence must have zero load/gravity and no pins")
    for name, array in (("q", q), ("qd", qd), ("inertial_target", inertial_target)):
        _require_lossless_float32_promotion(array, name)

    q_float32 = q.astype(np.float32)
    qd_float32 = qd.astype(np.float32)
    expected_inertial_target = np.add(
        q_float32[:-1],
        np.multiply(dt, qd_float32[:-1], dtype=np.float32),
        dtype=np.float32,
    )
    actual_inertial_target = inertial_target.astype(np.float32)
    if not np.array_equal(actual_inertial_target.view(np.uint32), expected_inertial_target.view(np.uint32)):
        raise ValueError(
            "inertial_target does not match the exact zero-load SolverVBD float32 position/velocity contract"
        )

    expected_qd = np.divide(
        np.subtract(q_float32[1:], q_float32[:-1], dtype=np.float32),
        dt,
        dtype=np.float32,
    )
    actual_qd = qd_float32[1:]
    if not np.array_equal(actual_qd.view(np.uint32), expected_qd.view(np.uint32)):
        raise ValueError("qd[1:] does not match the exact SolverVBD float32 position-update contract")

    for step_id, evidence in zip(record.step_ids, record._producer.evidence_steps, strict=True):
        expected_hashes = {
            "input_position_sha256": _producer_array_record(q[step_id])["sha256"],
            "input_velocity_sha256": _producer_array_record(qd[step_id])["sha256"],
            "inertial_target_sha256": _producer_array_record(inertial_target[step_id])["sha256"],
            "output_position_sha256": _producer_array_record(q[step_id + 1])["sha256"],
            "output_velocity_sha256": _producer_array_record(qd[step_id + 1])["sha256"],
        }
        if any(evidence[name] != expected for name, expected in expected_hashes.items()):
            raise ValueError(f"producer evidence hashes do not bind sequence arrays at step {step_id}")
        dynamic_scene = evidence["dynamic_scene_manifest"]
        array_records = dynamic_scene.get("arrays") if isinstance(dynamic_scene, Mapping) else None
        if not isinstance(array_records, Mapping):
            raise ValueError("producer dynamic scene is missing its array inventory")
        dynamic_hashes = {
            "external_force": _producer_array_record(external_force[step_id])["sha256"],
            "pin_targets": _producer_array_record(pin_targets[step_id])["sha256"],
            "pinned_indices": _producer_array_record(pinned)["sha256"],
            "vbd_inertial_target": _producer_array_record(inertial_target[step_id])["sha256"],
            "velocity": _producer_array_record(qd[step_id])["sha256"],
            "x_current": _producer_array_record(q[step_id])["sha256"],
        }
        for name, expected in dynamic_hashes.items():
            array_record = array_records.get(name)
            if not isinstance(array_record, Mapping) or array_record.get("sha256") != expected:
                raise ValueError(f"producer dynamic scene does not bind {name} at step {step_id}")

    for step_id, expected in zip(record.step_ids, record.reference_state_float64_sha256, strict=True):
        actual = canonical_reference_state_float64_sha256(q[step_id + 1])
        if actual != expected:
            raise ValueError(
                f"float64 reference-state SHA-256 mismatch for sequence {record.sequence_id!r} step {step_id}"
            )
    return arrays


class _ShuffledCycle:
    def __init__(self, values: Sequence[object], rng: np.random.Generator):
        self._values = tuple(values)
        if not self._values:
            raise ValueError("sampling cycle must not be empty")
        self._rng = rng
        self._order: tuple[int, ...] = ()
        self._offset = 0

    def next(self) -> object:
        if self._offset == len(self._order):
            self._order = tuple(int(index) for index in self._rng.permutation(len(self._values)))
            self._offset = 0
        value = self._values[self._order[self._offset]]
        self._offset += 1
        return value


@dataclasses.dataclass(frozen=True)
class ReferenceSequenceDataset:
    """Lazy authenticated view of one asset-disjoint sequence split index."""

    index_path: pathlib.Path
    index_sha256: str
    _records: tuple[ReferenceSequenceRecord, ...]

    @classmethod
    def load(cls, index_path: str | pathlib.Path) -> ReferenceSequenceDataset:
        """Load and validate index metadata without materializing NPZ shards."""
        path = pathlib.Path(index_path).resolve()
        payload = _read_json_object(path)
        header = reference_sequence_index_header()
        expected_root_keys = set(header) | {"splits"}
        if set(payload) != expected_root_keys:
            raise ValueError(f"reference sequence index keys must be exactly {tuple(sorted(expected_root_keys))}")
        for name, expected in header.items():
            if not _exact_json_equal(payload[name], expected):
                raise ValueError(f"reference sequence index {name} does not match the registered contract")
        splits = payload["splits"]
        expected_roles = {role.value for role in DatasetRole}
        if type(splits) is not dict or set(splits) != expected_roles:
            raise ValueError(f"split roles must be exactly {tuple(role.value for role in DatasetRole)}")

        records: list[ReferenceSequenceRecord] = []
        root = path.parent.resolve()
        for role in DatasetRole:
            role_payload = splits[role.value]
            if type(role_payload) is not list:
                raise ValueError(f"split role {role.value!r} must contain a JSON list")
            records.extend(_sequence_record(value, role=role, root=root) for value in role_payload)
        if not records:
            raise ValueError("reference sequence index must contain at least one sequence")
        records.sort(key=lambda record: (tuple(DatasetRole).index(record.role), record.asset_id, record.sequence_id))
        _validate_index_identities(records)
        canonical_payload = dict(header)
        canonical_payload["splits"] = {
            role.value: [record.as_dict() for record in records if record.role is role] for role in DatasetRole
        }
        return cls(
            index_path=path,
            index_sha256=_canonical_digest(canonical_payload),
            _records=tuple(records),
        )

    def records(self, role: DatasetRole | str) -> tuple[ReferenceSequenceRecord, ...]:
        """Return lexicographically ordered records for one exact role."""
        canonical_role = _dataset_role(role)
        return tuple(record for record in self._records if record.role is canonical_role)

    def provenance_anchor(self, record: ReferenceSequenceRecord) -> ReferenceSequenceProvenanceAnchor:
        """Return relocation-stable producer identities for one owned record."""
        if type(record) is not ReferenceSequenceRecord or not any(record is value for value in self._records):
            raise ValueError("record must be one exact record owned by this dataset")
        root = self.index_path.parent

        def artifact_uri(path: pathlib.Path) -> str:
            relative = path.relative_to(root).as_posix()
            encoded = quote(relative, safe="/-._~")
            return f"artifact://reference-sequence/{self.index_sha256}/{encoded}"

        first_step = record._producer.evidence_steps[0]
        final_step = record._producer.evidence_steps[-1]
        return ReferenceSequenceProvenanceAnchor(
            dataset_index_uri=f"artifact://reference-sequence/{self.index_sha256}/index.json",
            dataset_index_sha256=self.index_sha256,
            asset_id=record.asset_id,
            asset_source_sha256=record.asset_source_sha256,
            sequence_id=record.sequence_id,
            producer_manifest_uri=artifact_uri(record._producer.manifest_path),
            producer_manifest_sha256=record._producer.manifest_sha256,
            static_bundle_uri=artifact_uri(record._producer.static.path),
            static_bundle_sha256=record._producer.static.sha256,
            sequence_bundle_uri=artifact_uri(record._producer.sequence.path),
            sequence_bundle_sha256=record._producer.sequence.sha256,
            evidence_uri=artifact_uri(record._producer.evidence.path),
            evidence_sha256=record._producer.evidence.sha256,
            protocol_sha256=record.protocol_sha256,
            initial_position_sha256=str(first_step["input_position_sha256"]),
            initial_velocity_field_sha256=str(first_step["input_velocity_sha256"]),
            final_position_sha256=str(final_step["output_position_sha256"]),
            final_velocity_field_sha256=str(final_step["output_velocity_sha256"]),
            deformation_seed=record._producer.deformation_seed,
            velocity_seed=record._producer.velocity_seed,
            source_transition_count=len(record.step_ids),
            requested_dt_seconds=REFERENCE_REQUESTED_DT_SECONDS,
            dt_seconds=float(REFERENCE_EXECUTION_DT_SECONDS),
            execution_dt_float32_bits=REFERENCE_EXECUTION_DT_FLOAT32_BITS,
        )

    def sample_keys(
        self,
        role: DatasetRole | str,
        *,
        count: int,
        seed: int,
    ) -> tuple[ReferenceTransitionKey, ...]:
        """Replay the stateless asset/sequence/step PCG64 shuffled cycles."""
        canonical_role = _dataset_role(role)
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        records = self.records(canonical_role)
        if not records:
            raise ValueError(f"cannot sample the empty {canonical_role.value} role")
        records_by_asset = {
            asset_id: tuple(record for record in records if record.asset_id == asset_id)
            for asset_id in sorted({record.asset_id for record in records})
        }
        rng = np.random.Generator(np.random.PCG64(seed))
        asset_cycle = _ShuffledCycle(tuple(records_by_asset), rng)
        sequence_cycles = {
            asset_id: _ShuffledCycle(asset_records, rng) for asset_id, asset_records in records_by_asset.items()
        }
        step_cycles = {
            (record.asset_id, record.sequence_id): _ShuffledCycle(record.step_ids, rng) for record in records
        }
        result: list[ReferenceTransitionKey] = []
        for _ in range(count):
            asset_id = asset_cycle.next()
            if type(asset_id) is not str:
                raise RuntimeError("internal asset sampling cycle is malformed")
            record = sequence_cycles[asset_id].next()
            if type(record) is not ReferenceSequenceRecord:
                raise RuntimeError("internal sequence sampling cycle is malformed")
            step_id = step_cycles[(record.asset_id, record.sequence_id)].next()
            if type(step_id) is not int:
                raise RuntimeError("internal step sampling cycle is malformed")
            result.append(
                ReferenceTransitionKey(
                    asset_id=record.asset_id,
                    sequence_id=record.sequence_id,
                    step_id=step_id,
                )
            )
        return tuple(result)

    def transition(self, key: ReferenceTransitionKey) -> ReferenceTransition:
        """Authenticate and materialize one indexed transition."""
        if type(key) is not ReferenceTransitionKey:
            raise TypeError("key must be a canonical ReferenceTransitionKey")
        matching = tuple(
            record
            for record in self._records
            if (record.asset_id, record.sequence_id) == (key.asset_id, key.sequence_id)
        )
        if not matching:
            raise KeyError(f"unknown asset/sequence pair {(key.asset_id, key.sequence_id)!r}")
        record = matching[0]
        if key.step_id not in record.step_ids:
            raise KeyError(f"unknown step_id {key.step_id} for sequence {key.sequence_id!r}")
        static = _load_static(record)
        sequence = _load_sequence(record, static.rest_q.shape[0])
        step_id = key.step_id
        q_current_float32 = sequence["q"][step_id].astype(np.float32)
        velocity_float32 = sequence["qd"][step_id].astype(np.float32)
        x_previous = np.subtract(
            q_current_float32,
            np.multiply(sequence["dt"], velocity_float32, dtype=np.float32),
            dtype=np.float32,
        ).astype(np.float64)
        return ReferenceTransition(
            key=key,
            role=record.role,
            asset_source_sha256=record.asset_source_sha256,
            topology_sha256=record.topology_sha256,
            operator_sha256=record.operator_sha256,
            material_sha256=record.material_sha256,
            protocol_sha256=record.protocol_sha256,
            sequence_npz_sha256=record._producer.sequence.sha256,
            reference_state_float64_sha256=record.reference_state_float64_sha256[step_id],
            execution_dt_seconds=REFERENCE_EXECUTION_DT_SECONDS,
            deformation_seed=int(sequence["deformation_seed"]),
            velocity_seed=int(sequence["velocity_seed"]),
            static=static,
            x_current=_readonly_copy(sequence["q"][step_id]),
            velocity=_readonly_copy(sequence["qd"][step_id]),
            x_previous=_readonly_copy(x_previous),
            inertial_target=_readonly_copy(sequence["inertial_target"][step_id]),
            external_force=_readonly_copy(sequence["external_force"][step_id]),
            gravity=_readonly_copy(sequence["gravity"]),
            pinned_indices=_readonly_copy(sequence["pinned_indices"]),
            pin_targets=_readonly_copy(sequence["pin_targets"][step_id]),
            reference_positions=_readonly_copy(sequence["q"][step_id + 1]),
        )


def _validate_index_identities(records: Sequence[ReferenceSequenceRecord]) -> None:
    seen_sequences: set[tuple[str, str]] = set()
    seen_sequence_paths: set[pathlib.Path] = set()
    roles_by_asset_id: dict[str, set[DatasetRole]] = {}
    roles_by_source: dict[str, set[DatasetRole]] = {}
    sources_by_asset_id: dict[str, set[str]] = {}
    static_identity_by_file: dict[pathlib.Path, tuple[str, str, str]] = {}
    for record in records:
        sequence_key = (record.asset_id, record.sequence_id)
        if sequence_key in seen_sequences:
            raise ValueError(f"duplicate asset/sequence pair {sequence_key!r}")
        seen_sequences.add(sequence_key)
        if record._producer.sequence.path in seen_sequence_paths:
            raise ValueError("one sequence NPZ path must not alias multiple sequence records")
        seen_sequence_paths.add(record._producer.sequence.path)
        roles_by_asset_id.setdefault(record.asset_id, set()).add(record.role)
        roles_by_source.setdefault(record.asset_source_sha256, set()).add(record.role)
        sources_by_asset_id.setdefault(record.asset_id, set()).add(record.asset_source_sha256)
        static_identity = (record._producer.static.sha256, record.topology_sha256, record.material_sha256)
        previous_static_identity = static_identity_by_file.setdefault(record._producer.static.path, static_identity)
        if previous_static_identity != static_identity:
            raise ValueError("one static NPZ path has conflicting file/topology/material identities")
    if any(len(roles) != 1 for roles in roles_by_asset_id.values()):
        raise ValueError("asset_id appears in multiple roles")
    if any(len(roles) != 1 for roles in roles_by_source.values()):
        raise ValueError("asset_source_sha256 appears in multiple roles")
    if any(len(sources) != 1 for sources in sources_by_asset_id.values()):
        raise ValueError("one asset_id has conflicting asset_source_sha256 identities")


__all__ = [
    "REFERENCE_EXECUTION_DT_FLOAT32_BITS",
    "REFERENCE_EXECUTION_DT_SECONDS",
    "REFERENCE_REQUESTED_DT_SECONDS",
    "REFERENCE_SEQUENCE_INDEX_CONTRACT",
    "REFERENCE_SEQUENCE_SAMPLING_CONTRACT",
    "REFERENCE_SEQUENCE_SAMPLING_GENERATOR",
    "REFERENCE_SEQUENCE_SAMPLING_ORDER",
    "ReferenceSequenceDataset",
    "ReferenceSequenceProvenanceAnchor",
    "ReferenceSequenceRecord",
    "ReferenceStaticData",
    "ReferenceTransition",
    "ReferenceTransitionKey",
    "canonical_reference_state_float64_sha256",
    "reference_sequence_index_header",
]
