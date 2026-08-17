# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Immutable dataset, split, access, and sampling contracts for PSS v5.

This module intentionally contains no model or trainer dependency. It binds
numeric payload identities to whole trajectories before v5 training code can
open those payloads and represents the policy that confirmation data remains
sealed from training and model selection. Its functional access ledger is
branch-local audit evidence, not global access control; enforcing one
canonical ledger head requires an external append-only anchor.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import itertools
import json
import math
import pathlib
import struct
from collections.abc import Callable, Sequence
from urllib.parse import urlsplit

import numpy as np

from .torch_solver import static_mesh_sha256

_NUMERIC_COMPONENTS = (
    "observed_f",
    "input_f",
    "reference_f",
    "observed_state",
    "input_state",
    "reference_state",
)
_OBJECTIVE_COMPONENTS = ("physical_step", "common_objective")
_PAYLOAD_COMPONENTS = (*_NUMERIC_COMPONENTS, *_OBJECTIVE_COMPONENTS)
_SAMPLE_CONTRACT = "pss-v5-dataset-sample-v2"
_STATIC_LAYOUT_CONTRACT = "pss-v5-dataset-static-layout-v2"
_PROVENANCE_CONTRACT = "pss-v5-dataset-trajectory-provenance-v1"
_TRAJECTORY_CONTRACT = "pss-v5-dataset-trajectory-v2"
_SPLIT_CONTRACT = "pss-v5-dataset-split-v2"
_ACCESS_CONTRACT = "pss-v5-dataset-access-ledger-v1"
_PAYLOAD_SELECTION_CONTRACT = "pss-v5-dataset-payload-selection-v1"
_SAMPLING_CONTRACT = "pss-v5-static-layout-homogeneous-trajectory-first-sampling-v2"
_OBJECTIVE_ROUTING = "per-sample-unbatched-physical-objective-v1"
_COMPLETE_TRAJECTORY_SELECTION = "complete-contiguous-trajectory-v1"
_AUTHENTICATED_SUBRANGE_SELECTION = "authenticated-contiguous-subrange-v1"
_TRAJECTORY_SELECTION_CONTRACTS = (
    _COMPLETE_TRAJECTORY_SELECTION,
    _AUTHENTICATED_SUBRANGE_SELECTION,
)


class DatasetRole(str, enum.Enum):
    """A mutually exclusive role assigned to a complete trajectory."""

    TRAIN = "train"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"
    CONSUMED_REGRESSION = "consumed_regression"


class DataAccessPurpose(str, enum.Enum):
    """Declared reason for reading dataset metadata or payloads."""

    TRAINING = "training"
    MODEL_SELECTION = "model_selection"
    CONFIRMATION_EVALUATION = "confirmation_evaluation"
    REGRESSION_EVALUATION = "regression_evaluation"
    AUDIT = "audit"


class DataAccessScope(str, enum.Enum):
    """Whether an access exposes only provenance or sample payloads."""

    METADATA = "metadata"
    PAYLOAD = "payload"


_ROLE_ORDER = tuple(DatasetRole)
_PAYLOAD_ROLES_BY_PURPOSE = {
    DataAccessPurpose.TRAINING: frozenset((DatasetRole.TRAIN,)),
    DataAccessPurpose.MODEL_SELECTION: frozenset((DatasetRole.TRAIN, DatasetRole.VALIDATION)),
    DataAccessPurpose.CONFIRMATION_EVALUATION: frozenset((DatasetRole.CONFIRMATION,)),
    DataAccessPurpose.REGRESSION_EVALUATION: frozenset((DatasetRole.CONSUMED_REGRESSION,)),
    DataAccessPurpose.AUDIT: frozenset(),
}


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_uri(value: object, name: str) -> str:
    uri = _identifier(value, name)
    if any(character.isspace() for character in uri):
        raise ValueError(f"{name} must not contain whitespace")
    parsed = urlsplit(uri)
    if not parsed.scheme or (not parsed.netloc and not parsed.path):
        raise ValueError(f"{name} must be an absolute canonical URI")
    if not uri.startswith(f"{parsed.scheme}:"):
        raise ValueError(f"{name} URI scheme must use canonical lowercase spelling")
    return uri


def _positive_float64(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite float64 value")
    canonical = float(value)
    if not math.isfinite(canonical) or canonical <= 0.0:
        raise ValueError(f"{name} must be a positive finite float64 value")
    return canonical


def _finite_float64(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float64 value")
    canonical = float(value)
    if not math.isfinite(canonical):
        raise ValueError(f"{name} must be a finite float64 value")
    return canonical


def _float64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def verify_file_sha256(
    path: str | pathlib.Path,
    expected_sha256: str,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Stream and verify one durable artifact without loading it in memory.

    Args:
        path: Local materialization of the durable artifact.
        expected_sha256: Lowercase SHA-256 bound by its provenance record.
        chunk_size: Positive number of bytes read per streaming update.

    Returns:
        The verified lowercase SHA-256 digest.
    """
    expected = _sha256(expected_sha256, "expected file sha256")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"durable artifact SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def canonical_topology_sha256(rest_q: np.ndarray, tet_indices: np.ndarray) -> str:
    """Return the runtime-canonical ordered tetrahedral topology identity.

    Dataset producers and trusted loaders must use the same exact algorithm as
    compatibility projection: float64 rest positions plus ordered int64 tet
    corners under :func:`torch_solver.static_mesh_sha256`.
    """
    return static_mesh_sha256(rest_q, tet_indices)


def verify_trajectory_topology(
    trajectory: TrajectoryRecord,
    rest_q: np.ndarray,
    tet_indices: np.ndarray,
) -> str:
    """Recompute and verify a trajectory's materialized static topology."""
    if type(trajectory) is not TrajectoryRecord:
        raise TypeError("trajectory must be a canonical TrajectoryRecord")
    actual = canonical_topology_sha256(rest_q, tet_indices)
    if actual != trajectory.topology_sha256:
        raise ValueError("materialized rest positions/connectivity differ from the trajectory topology")
    return actual


def _dataset_role(value: DatasetRole | str) -> DatasetRole:
    try:
        return DatasetRole(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"role must be one of {tuple(role.value for role in DatasetRole)}") from exc


def _access_purpose(value: DataAccessPurpose | str) -> DataAccessPurpose:
    try:
        return DataAccessPurpose(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"purpose must be one of {tuple(purpose.value for purpose in DataAccessPurpose)}") from exc


def _access_scope(value: DataAccessScope | str) -> DataAccessScope:
    try:
        return DataAccessScope(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scope must be one of {tuple(scope.value for scope in DataAccessScope)}") from exc


@dataclasses.dataclass(frozen=True)
class TrajectoryProvenance:
    """Durable generation and physical provenance for one trajectory."""

    generation_spec_sha256: str
    history_manifest_sha256: str
    root_checkpoint_sha256: str
    final_checkpoint_sha256: str
    artifact_bundle_uri: str
    artifact_bundle_sha256: str
    artifact_source_uri: str
    artifact_source_sha256: str
    static_bundle_sha256: str
    density_kg_m3: float
    initial_velocity_m_s: tuple[float, float, float]
    pin_schedule_sha256: str
    event_inventory_sha256: str
    coordinate_start_sha256: str
    coordinate_stop_sha256: str
    coordinate_range_sha256: str
    dt_seconds: float
    generation_seed: int
    density_float64_bits: str = dataclasses.field(init=False)
    initial_velocity_float64_bits: tuple[str, str, str] = dataclasses.field(init=False)
    initial_velocity_sha256: str = dataclasses.field(init=False)
    dt_float64_bits: str = dataclasses.field(init=False)
    provenance_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "generation_spec_sha256",
            "history_manifest_sha256",
            "root_checkpoint_sha256",
            "final_checkpoint_sha256",
            "artifact_bundle_sha256",
            "artifact_source_sha256",
            "static_bundle_sha256",
            "pin_schedule_sha256",
            "event_inventory_sha256",
            "coordinate_start_sha256",
            "coordinate_stop_sha256",
            "coordinate_range_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("artifact_bundle_uri", "artifact_source_uri"):
            _canonical_uri(getattr(self, name), name)

        density = _positive_float64(self.density_kg_m3, "density_kg_m3")
        velocity_values = tuple(self.initial_velocity_m_s)
        if len(velocity_values) != 3:
            raise ValueError("initial_velocity_m_s must contain exactly three components")
        velocity = tuple(
            _finite_float64(value, f"initial_velocity_m_s[{index}]") for index, value in enumerate(velocity_values)
        )
        dt_seconds = _positive_float64(self.dt_seconds, "dt_seconds")
        if type(self.generation_seed) is not int or self.generation_seed < 0:
            raise ValueError("generation_seed must be a non-negative integer")

        density_bits = _float64_bits(density)
        velocity_bits = tuple(_float64_bits(value) for value in velocity)
        dt_bits = _float64_bits(dt_seconds)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "initial_velocity_m_s", velocity)
        object.__setattr__(self, "dt_seconds", dt_seconds)
        object.__setattr__(self, "density_float64_bits", density_bits)
        object.__setattr__(self, "initial_velocity_float64_bits", velocity_bits)
        object.__setattr__(
            self,
            "initial_velocity_sha256",
            _canonical_digest({"float64_bits": list(velocity_bits)}),
        )
        object.__setattr__(self, "dt_float64_bits", dt_bits)
        object.__setattr__(self, "provenance_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _PROVENANCE_CONTRACT,
            "generation_spec_sha256": self.generation_spec_sha256,
            "history_manifest_sha256": self.history_manifest_sha256,
            "root_checkpoint_sha256": self.root_checkpoint_sha256,
            "final_checkpoint_sha256": self.final_checkpoint_sha256,
            "artifact_bundle": {
                "uri": self.artifact_bundle_uri,
                "sha256": self.artifact_bundle_sha256,
            },
            "artifact_source": {
                "uri": self.artifact_source_uri,
                "sha256": self.artifact_source_sha256,
            },
            "static_bundle_sha256": self.static_bundle_sha256,
            "density_kg_m3": self.density_kg_m3,
            "density_float64_bits": self.density_float64_bits,
            "initial_velocity_m_s": list(self.initial_velocity_m_s),
            "initial_velocity_float64_bits": list(self.initial_velocity_float64_bits),
            "initial_velocity_sha256": self.initial_velocity_sha256,
            "pin_schedule_sha256": self.pin_schedule_sha256,
            "event_inventory_sha256": self.event_inventory_sha256,
            "coordinate_range": {
                "start_sha256": self.coordinate_start_sha256,
                "stop_sha256": self.coordinate_stop_sha256,
                "range_sha256": self.coordinate_range_sha256,
            },
            "dt_seconds": self.dt_seconds,
            "dt_float64_bits": self.dt_float64_bits,
            "generation_seed": self.generation_seed,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible provenance record."""
        payload = self._payload()
        payload["provenance_sha256"] = self.provenance_sha256
        return payload

    def verify_artifact_bundle(self, path: str | pathlib.Path, *, chunk_size: int = 1024 * 1024) -> str:
        """Verify a local materialization of the durable artifact bundle."""
        return verify_file_sha256(path, self.artifact_bundle_sha256, chunk_size=chunk_size)

    def verify_artifact_source(self, path: str | pathlib.Path, *, chunk_size: int = 1024 * 1024) -> str:
        """Verify a local materialization of the artifact-source record."""
        return verify_file_sha256(path, self.artifact_source_sha256, chunk_size=chunk_size)


@dataclasses.dataclass(frozen=True)
class NumericContentIdentity:
    """Logical identifier and byte-level hash for one numeric payload."""

    identifier: str
    sha256: str
    identity_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.identifier, "numeric content identifier")
        _sha256(self.sha256, "numeric content sha256")
        object.__setattr__(self, "identity_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {"identifier": self.identifier, "sha256": self.sha256}

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible identity."""
        payload = self._payload()
        payload["identity_sha256"] = self.identity_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class TrajectorySampleRecord:
    """Immutable identities for all numeric inputs and objectives of one sample.

    ``topology_sha256`` must be produced by
    :func:`canonical_topology_sha256`. ``physical_step_sha256`` and
    ``common_objective_sha256`` are declarations made by the dataset producer.
    A trusted loader must reconstruct the materialized topology and both
    runtime contexts, then compare all three canonical digests before use.
    """

    sample_id: str
    ordinal: int
    topology_sha256: str
    operator_geometry_sha256: str
    material_sha256: str
    pin_signature_sha256: str
    dt_seconds: float
    physical_step_sha256: str
    common_objective_sha256: str
    observed_f: NumericContentIdentity
    input_f: NumericContentIdentity
    reference_f: NumericContentIdentity
    observed_state: NumericContentIdentity
    input_state: NumericContentIdentity
    reference_state: NumericContentIdentity
    dt_float64_bits: str = dataclasses.field(init=False)
    static_layout_sha256: str = dataclasses.field(init=False)
    sample_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.sample_id, "sample_id")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("sample ordinal must be a non-negative integer")
        for name in (
            "topology_sha256",
            "operator_geometry_sha256",
            "material_sha256",
            "pin_signature_sha256",
            "physical_step_sha256",
            "common_objective_sha256",
        ):
            _sha256(getattr(self, name), f"sample {name}")
        dt_seconds = _positive_float64(self.dt_seconds, "sample dt_seconds")
        object.__setattr__(self, "dt_seconds", dt_seconds)
        object.__setattr__(self, "dt_float64_bits", _float64_bits(dt_seconds))
        object.__setattr__(
            self,
            "static_layout_sha256",
            _canonical_digest(
                {
                    "contract": _STATIC_LAYOUT_CONTRACT,
                    "topology_sha256": self.topology_sha256,
                    "operator_geometry_sha256": self.operator_geometry_sha256,
                    "material_sha256": self.material_sha256,
                    "pin_signature_sha256": self.pin_signature_sha256,
                    "dt_float64_bits": self.dt_float64_bits,
                }
            ),
        )
        for name, identity in self.numeric_content:
            if type(identity) is not NumericContentIdentity:
                raise ValueError(f"{name} must be a canonical NumericContentIdentity")
            if identity.identity_sha256 != _canonical_digest(identity._payload()):
                raise ValueError(f"{name} identity changed after authentication")
        object.__setattr__(self, "sample_sha256", _canonical_digest(self._payload()))

    @property
    def numeric_content(self) -> tuple[tuple[str, NumericContentIdentity], ...]:
        """Numeric identities in canonical component order."""
        return tuple((name, getattr(self, name)) for name in _NUMERIC_COMPONENTS)

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _SAMPLE_CONTRACT,
            "sample_id": self.sample_id,
            "ordinal": self.ordinal,
            "topology_sha256": self.topology_sha256,
            "operator_geometry_sha256": self.operator_geometry_sha256,
            "material_sha256": self.material_sha256,
            "pin_signature_sha256": self.pin_signature_sha256,
            "dt_seconds": self.dt_seconds,
            "dt_float64_bits": self.dt_float64_bits,
            "static_layout_sha256": self.static_layout_sha256,
            "physical_step_sha256": self.physical_step_sha256,
            "common_objective_sha256": self.common_objective_sha256,
            "numeric_content": {name: identity.as_dict() for name, identity in self.numeric_content},
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible sample record."""
        payload = self._payload()
        payload["sample_sha256"] = self.sample_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class TrajectoryRecord:
    """Canonical metadata and numeric identities for one complete trajectory.

    The topology field is the exact ordered rest-mesh identity returned by
    :func:`canonical_topology_sha256`; trusted loaders verify it with
    :func:`verify_trajectory_topology` before constructing runtime state.
    """

    trajectory_id: str
    scene_family: str
    load_program_id: str
    load_program_sha256: str
    source_chain_sha256: str
    topology_sha256: str
    operator_geometry_sha256: str
    material_sha256: str
    provenance: TrajectoryProvenance
    source_transition_count: int
    samples: tuple[TrajectorySampleRecord, ...]
    selection_contract: str = _COMPLETE_TRAJECTORY_SELECTION
    selection_provenance_sha256: str | None = None
    trajectory_id_sha256: str = dataclasses.field(init=False)
    trajectory_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "scene_family", "load_program_id"):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "trajectory_id_sha256", hashlib.sha256(self.trajectory_id.encode("utf-8")).hexdigest())
        for name in (
            "load_program_sha256",
            "source_chain_sha256",
            "topology_sha256",
            "operator_geometry_sha256",
            "material_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.provenance) is not TrajectoryProvenance:
            raise ValueError("provenance must be a canonical TrajectoryProvenance")
        if self.provenance.provenance_sha256 != _canonical_digest(self.provenance._payload()):
            raise ValueError("trajectory provenance changed after authentication")
        if type(self.source_transition_count) is not int or self.source_transition_count < 1:
            raise ValueError("source_transition_count must be a positive integer")
        if self.selection_contract not in _TRAJECTORY_SELECTION_CONTRACTS:
            raise ValueError(f"selection_contract must be one of {_TRAJECTORY_SELECTION_CONTRACTS}")

        samples = tuple(self.samples)
        if not samples:
            raise ValueError("a trajectory must contain at least one sample")
        if any(type(sample) is not TrajectorySampleRecord for sample in samples):
            raise ValueError("trajectory samples must be canonical TrajectorySampleRecord values")
        samples = tuple(sorted(samples, key=lambda sample: (sample.ordinal, sample.sample_id)))
        sample_ids = [sample.sample_id for sample in samples]
        ordinals = [sample.ordinal for sample in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_id values must be unique within a trajectory")
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("sample ordinals must be unique within a trajectory")
        if any(next_ordinal != ordinal + 1 for ordinal, next_ordinal in itertools.pairwise(ordinals)):
            raise ValueError("trajectory sample ordinals must form one contiguous selection")
        if ordinals[-1] >= self.source_transition_count:
            raise ValueError("trajectory sample ordinal exceeds source_transition_count")
        if self.selection_contract == _COMPLETE_TRAJECTORY_SELECTION:
            if ordinals[0] != 0 or len(ordinals) != self.source_transition_count:
                raise ValueError("complete trajectory selection must contain every source transition")
            if self.selection_provenance_sha256 is not None:
                raise ValueError("complete trajectory selection must not use subrange provenance")
        else:
            _sha256(self.selection_provenance_sha256, "selection_provenance_sha256")
        for sample in samples:
            if sample.topology_sha256 != self.topology_sha256:
                raise ValueError("sample topology disagrees with its trajectory")
            if sample.operator_geometry_sha256 != self.operator_geometry_sha256:
                raise ValueError("sample operator geometry disagrees with its trajectory")
            if sample.material_sha256 != self.material_sha256:
                raise ValueError("sample material disagrees with its trajectory")
            if sample.dt_float64_bits != self.provenance.dt_float64_bits:
                raise ValueError("sample time step disagrees with trajectory provenance")
            if sample.sample_sha256 != _canonical_digest(sample._payload()):
                raise ValueError(f"sample {sample.sample_id!r} changed after authentication")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "trajectory_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _TRAJECTORY_CONTRACT,
            "trajectory_id": self.trajectory_id,
            "trajectory_id_sha256": self.trajectory_id_sha256,
            "scene_family": self.scene_family,
            "load_program_id": self.load_program_id,
            "load_program_sha256": self.load_program_sha256,
            "source_chain_sha256": self.source_chain_sha256,
            "topology_sha256": self.topology_sha256,
            "operator_geometry_sha256": self.operator_geometry_sha256,
            "material_sha256": self.material_sha256,
            "provenance": self.provenance.as_dict(),
            "source_transition_count": self.source_transition_count,
            "selection_contract": self.selection_contract,
            "selection_provenance_sha256": self.selection_provenance_sha256,
            "samples": [sample.as_dict() for sample in self.samples],
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible trajectory record."""
        payload = self._payload()
        payload["trajectory_sha256"] = self.trajectory_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class SplitRoleOverlap:
    """Topology, source-operator, and material overlap for two split roles."""

    first: DatasetRole
    second: DatasetRole
    topology_sha256: tuple[str, ...]
    operator_geometry_sha256: tuple[str, ...]
    material_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", _dataset_role(self.first))
        object.__setattr__(self, "second", _dataset_role(self.second))
        if _ROLE_ORDER.index(self.first) >= _ROLE_ORDER.index(self.second):
            raise ValueError("split overlap roles must use canonical order")
        for name in ("topology_sha256", "operator_geometry_sha256", "material_sha256"):
            values = tuple(sorted(getattr(self, name)))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} overlap values must be unique")
            for value in values:
                _sha256(value, f"overlap {name}")
            object.__setattr__(self, name, values)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible overlap record."""
        return {
            "roles": [self.first.value, self.second.value],
            "topology_sha256": list(self.topology_sha256),
            "operator_geometry_sha256": list(self.operator_geometry_sha256),
            "material_sha256": list(self.material_sha256),
        }


@dataclasses.dataclass(frozen=True)
class SplitOverlapReport:
    """Non-fatal topology, source-operator, and material split overlap."""

    role_pairs: tuple[SplitRoleOverlap, ...]

    def __post_init__(self) -> None:
        role_pairs = tuple(self.role_pairs)
        if any(type(pair) is not SplitRoleOverlap for pair in role_pairs):
            raise ValueError("role_pairs must contain canonical SplitRoleOverlap values")
        role_pairs = tuple(
            sorted(
                role_pairs,
                key=lambda pair: (_ROLE_ORDER.index(pair.first), _ROLE_ORDER.index(pair.second)),
            )
        )
        keys = [(pair.first, pair.second) for pair in role_pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("split overlap role pairs must be unique")
        object.__setattr__(self, "role_pairs", role_pairs)

    @property
    def has_topology_overlap(self) -> bool:
        """Whether any role pair shares a topology."""
        return any(pair.topology_sha256 for pair in self.role_pairs)

    @property
    def has_material_overlap(self) -> bool:
        """Whether any role pair shares a material."""
        return any(pair.material_sha256 for pair in self.role_pairs)

    @property
    def has_operator_geometry_overlap(self) -> bool:
        """Whether any role pair shares exact authenticated source geometry."""
        return any(pair.operator_geometry_sha256 for pair in self.role_pairs)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible overlap report."""
        return {
            "role_pairs": [pair.as_dict() for pair in self.role_pairs],
            "has_topology_overlap": self.has_topology_overlap,
            "has_operator_geometry_overlap": self.has_operator_geometry_overlap,
            "has_material_overlap": self.has_material_overlap,
        }


def _cross_role_values(
    records_by_role: dict[DatasetRole, tuple[TrajectoryRecord, ...]],
    values: Callable[[TrajectoryRecord], Sequence[str]],
) -> dict[str, tuple[DatasetRole, ...]]:
    roles_by_value: dict[str, set[DatasetRole]] = {}
    for role in _ROLE_ORDER:
        for record in records_by_role[role]:
            record_values = values(record)
            for value in set(record_values):
                roles_by_value.setdefault(value, set()).add(role)
    return {
        value: tuple(role for role in _ROLE_ORDER if role in roles)
        for value, roles in sorted(roles_by_value.items())
        if len(roles) > 1
    }


def _format_role_collision(collisions: dict[str, tuple[DatasetRole, ...]]) -> str:
    value, roles = next(iter(collisions.items()))
    return f"{value} ({', '.join(role.value for role in roles)})"


@dataclasses.dataclass(frozen=True)
class SplitManifest:
    """Canonical whole-trajectory v5 split with fail-closed leakage checks."""

    train: tuple[TrajectoryRecord, ...]
    validation: tuple[TrajectoryRecord, ...]
    confirmation: tuple[TrajectoryRecord, ...]
    consumed_regression: tuple[TrajectoryRecord, ...] = ()
    reject_topology_overlap: bool = False
    reject_material_overlap: bool = False
    schema_version: int = 2
    overlap_report: SplitOverlapReport = dataclasses.field(init=False)
    manifest_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("v5 dataset split schema_version must be exactly 2")
        if type(self.reject_topology_overlap) is not bool or type(self.reject_material_overlap) is not bool:
            raise ValueError("overlap rejection settings must be bool values")

        canonical_roles: dict[DatasetRole, tuple[TrajectoryRecord, ...]] = {}
        for role in _ROLE_ORDER:
            records = tuple(getattr(self, role.value))
            if any(type(record) is not TrajectoryRecord for record in records):
                raise ValueError(f"{role.value} must contain canonical TrajectoryRecord values")
            records = tuple(sorted(records, key=lambda record: record.trajectory_id))
            identifiers = [record.trajectory_id for record in records]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"trajectory_id values must be unique within {role.value}")
            for record in records:
                if record.trajectory_sha256 != _canonical_digest(record._payload()):
                    raise ValueError(f"trajectory {record.trajectory_id!r} changed after authentication")
            canonical_roles[role] = records
            object.__setattr__(self, role.value, records)

        trajectory_overlap = _cross_role_values(canonical_roles, lambda record: (record.trajectory_id,))
        if trajectory_overlap:
            raise ValueError(f"trajectory overlap across roles: {_format_role_collision(trajectory_overlap)}")
        trajectory_source_overlap = _cross_role_values(
            canonical_roles,
            lambda record: (record.source_chain_sha256, record.provenance.provenance_sha256),
        )
        if trajectory_source_overlap:
            raise ValueError(
                f"trajectory source overlap across roles: {_format_role_collision(trajectory_source_overlap)}"
            )
        load_overlap = _cross_role_values(
            canonical_roles,
            lambda record: (record.load_program_id, record.load_program_sha256),
        )
        if load_overlap:
            raise ValueError(f"load-program overlap across roles: {_format_role_collision(load_overlap)}")

        def numeric_identifiers(record: TrajectoryRecord) -> tuple[str, ...]:
            return tuple(identity.identifier for sample in record.samples for _name, identity in sample.numeric_content)

        def payload_hashes(record: TrajectoryRecord) -> tuple[str, ...]:
            numeric = tuple(identity.sha256 for sample in record.samples for _name, identity in sample.numeric_content)
            objective = tuple(
                digest
                for sample in record.samples
                for digest in (sample.physical_step_sha256, sample.common_objective_sha256)
            )
            return (*numeric, *objective)

        numeric_identifier_overlap = _cross_role_values(canonical_roles, numeric_identifiers)
        if numeric_identifier_overlap:
            raise ValueError(
                f"numeric content identifier overlap across roles: {_format_role_collision(numeric_identifier_overlap)}"
            )
        payload_hash_overlap = _cross_role_values(canonical_roles, payload_hashes)
        if payload_hash_overlap:
            raise ValueError(
                f"sample payload SHA-256 overlap across roles: {_format_role_collision(payload_hash_overlap)}"
            )

        role_pairs: list[SplitRoleOverlap] = []
        for first, second in itertools.combinations(_ROLE_ORDER, 2):
            first_topologies = {record.topology_sha256 for record in canonical_roles[first]}
            second_topologies = {record.topology_sha256 for record in canonical_roles[second]}
            first_operators = {record.operator_geometry_sha256 for record in canonical_roles[first]}
            second_operators = {record.operator_geometry_sha256 for record in canonical_roles[second]}
            first_materials = {record.material_sha256 for record in canonical_roles[first]}
            second_materials = {record.material_sha256 for record in canonical_roles[second]}
            topology_overlap = tuple(sorted(first_topologies & second_topologies))
            operator_overlap = tuple(sorted(first_operators & second_operators))
            material_overlap = tuple(sorted(first_materials & second_materials))
            if topology_overlap or operator_overlap or material_overlap:
                role_pairs.append(
                    SplitRoleOverlap(
                        first=first,
                        second=second,
                        topology_sha256=topology_overlap,
                        operator_geometry_sha256=operator_overlap,
                        material_sha256=material_overlap,
                    )
                )
        overlap_report = SplitOverlapReport(tuple(role_pairs))
        if self.reject_topology_overlap and overlap_report.has_topology_overlap:
            raise ValueError("topology overlap across roles is forbidden by this split")
        if self.reject_material_overlap and overlap_report.has_material_overlap:
            raise ValueError("material overlap across roles is forbidden by this split")
        object.__setattr__(self, "overlap_report", overlap_report)
        object.__setattr__(self, "manifest_sha256", _canonical_digest(self._payload()))

    def records(self, role: DatasetRole | str) -> tuple[TrajectoryRecord, ...]:
        """Return the canonical records assigned to ``role``."""
        canonical_role = _dataset_role(role)
        return getattr(self, canonical_role.value)

    def role_for_trajectory(self, trajectory_id: str) -> DatasetRole:
        """Resolve the unique role for a trajectory identifier."""
        _identifier(trajectory_id, "trajectory_id")
        for role in _ROLE_ORDER:
            if any(record.trajectory_id == trajectory_id for record in self.records(role)):
                return role
        raise ValueError(f"trajectory {trajectory_id!r} does not belong to this split")

    def trajectory(self, trajectory_id: str) -> TrajectoryRecord:
        """Return one trajectory record after resolving its split role."""
        role = self.role_for_trajectory(trajectory_id)
        return next(record for record in self.records(role) if record.trajectory_id == trajectory_id)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": _SPLIT_CONTRACT,
            "roles": {role.value: [record.as_dict() for record in self.records(role)] for role in _ROLE_ORDER},
            "overlap_policy": {
                "reject_topology_overlap": self.reject_topology_overlap,
                "reject_material_overlap": self.reject_material_overlap,
            },
            "overlap_report": self.overlap_report.as_dict(),
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible split manifest."""
        payload = self._payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload


def _verify_manifest(manifest: SplitManifest) -> None:
    if type(manifest) is not SplitManifest:
        raise ValueError("manifest must be a canonical SplitManifest")
    if manifest.manifest_sha256 != _canonical_digest(manifest._payload()):
        raise ValueError("split manifest changed after authentication")


def _payload_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(sorted(values))
    if len(set(names)) != len(names):
        raise ValueError("payload_names must be unique")
    unknown = tuple(name for name in names if name not in _PAYLOAD_COMPONENTS)
    if unknown:
        raise ValueError(f"payload_names contain unknown payload components: {unknown}")
    return names


def _payload_identity_digest(trajectory: TrajectoryRecord, payload_names: Sequence[str]) -> str:
    names = _payload_names(payload_names)

    def component_identity(sample: TrajectorySampleRecord, name: str) -> dict[str, object]:
        if name in _NUMERIC_COMPONENTS:
            identity = getattr(sample, name)
            if type(identity) is not NumericContentIdentity:
                raise ValueError(f"{name} must be a canonical NumericContentIdentity")
            return identity.as_dict()
        return {"sha256": getattr(sample, f"{name}_sha256")}

    bindings = [
        {
            "sample_id": sample.sample_id,
            "sample_sha256": sample.sample_sha256,
            "component": name,
            "identity": component_identity(sample, name),
        }
        for sample in trajectory.samples
        for name in names
    ]
    return _canonical_digest(
        {
            "contract": _PAYLOAD_SELECTION_CONTRACT,
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_sha256": trajectory.trajectory_sha256,
            "payload_names": list(names),
            "bindings": bindings,
        }
    )


@dataclasses.dataclass(frozen=True)
class DataAccessRecord:
    """One immutable, hash-chained metadata or payload access."""

    sequence: int
    trajectory_id: str
    role: DatasetRole
    purpose: DataAccessPurpose
    scope: DataAccessScope
    payload_names: tuple[str, ...]
    payload_identity_sha256: str | None
    previous_sha256: str
    access_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("access sequence must be a non-negative integer")
        _identifier(self.trajectory_id, "trajectory_id")
        object.__setattr__(self, "role", _dataset_role(self.role))
        object.__setattr__(self, "purpose", _access_purpose(self.purpose))
        object.__setattr__(self, "scope", _access_scope(self.scope))
        payload_names = _payload_names(self.payload_names)
        if self.scope is DataAccessScope.METADATA:
            if payload_names or self.payload_identity_sha256 is not None:
                raise ValueError("metadata access must not bind sample payloads")
        else:
            if not payload_names:
                raise ValueError("payload access must name every opened payload")
            _sha256(self.payload_identity_sha256, "payload_identity_sha256")
        _sha256(self.previous_sha256, "previous access sha256")
        object.__setattr__(self, "payload_names", payload_names)
        object.__setattr__(self, "access_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "trajectory_id": self.trajectory_id,
            "role": self.role.value,
            "purpose": self.purpose.value,
            "scope": self.scope.value,
            "payload_names": list(self.payload_names),
            "payload_identity_sha256": self.payload_identity_sha256,
            "previous_sha256": self.previous_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible access record."""
        payload = self._payload()
        payload["access_sha256"] = self.access_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class DataAccessLedger:
    """Functional, branch-local audit ledger for the frozen split policy.

    The hash chain proves ordering and policy compliance for this ledger
    branch. It is not a global access-control mechanism: an independent
    process can start another branch from the same manifest unless an external
    append-only service anchors the canonical head.
    """

    manifest: SplitManifest
    accesses: tuple[DataAccessRecord, ...] = ()
    confirmation_payload_released: bool = dataclasses.field(init=False)
    ledger_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _verify_manifest(self.manifest)
        accesses = tuple(self.accesses)
        previous_sha256 = self.manifest.manifest_sha256
        confirmation_payload_released = False
        for sequence, access in enumerate(accesses):
            if type(access) is not DataAccessRecord:
                raise ValueError("accesses must contain canonical DataAccessRecord values")
            if access.sequence != sequence:
                raise ValueError("access sequence is not contiguous")
            if access.previous_sha256 != previous_sha256:
                raise ValueError("access hash chain is disconnected")
            if access.access_sha256 != _canonical_digest(access._payload()):
                raise ValueError("access record changed after authentication")
            trajectory = self.manifest.trajectory(access.trajectory_id)
            expected_role = self.manifest.role_for_trajectory(access.trajectory_id)
            if access.role is not expected_role:
                raise ValueError("access role disagrees with the frozen split")
            if access.scope is DataAccessScope.PAYLOAD:
                expected_payload_identity = _payload_identity_digest(trajectory, access.payload_names)
                if access.payload_identity_sha256 != expected_payload_identity:
                    raise ValueError("access payload identity disagrees with the frozen trajectory")
                if confirmation_payload_released and access.purpose in (
                    DataAccessPurpose.TRAINING,
                    DataAccessPurpose.MODEL_SELECTION,
                ):
                    raise ValueError(
                        "training or model-selection payload access cannot resume after confirmation release "
                        "on this ledger branch"
                    )
            self._validate_payload_policy(access)
            if access.scope is DataAccessScope.PAYLOAD and access.role is DatasetRole.CONFIRMATION:
                confirmation_payload_released = True
            previous_sha256 = access.access_sha256
        object.__setattr__(self, "accesses", accesses)
        object.__setattr__(self, "confirmation_payload_released", confirmation_payload_released)
        object.__setattr__(self, "ledger_sha256", _canonical_digest(self._payload()))

    @staticmethod
    def _validate_payload_policy(access: DataAccessRecord) -> None:
        if access.scope is DataAccessScope.METADATA:
            return
        allowed_roles = _PAYLOAD_ROLES_BY_PURPOSE[access.purpose]
        if access.role not in allowed_roles:
            if access.role is DatasetRole.CONFIRMATION and access.purpose in (
                DataAccessPurpose.TRAINING,
                DataAccessPurpose.MODEL_SELECTION,
            ):
                raise ValueError("confirmation payload access is forbidden during training or model selection")
            if access.role is DatasetRole.CONSUMED_REGRESSION and access.purpose in (
                DataAccessPurpose.TRAINING,
                DataAccessPurpose.MODEL_SELECTION,
            ):
                raise ValueError("consumed-regression payload access is forbidden during training or model selection")
            raise ValueError(f"{access.purpose.value} may not access {access.role.value} payloads")

    def record_access(
        self,
        trajectory_id: str,
        *,
        purpose: DataAccessPurpose | str,
        scope: DataAccessScope | str,
        payload_names: Sequence[str] = (),
    ) -> DataAccessLedger:
        """Return a new ledger with one policy-checked access appended.

        Args:
            trajectory_id: Frozen trajectory identifier being accessed.
            purpose: Declared training, selection, confirmation, or audit use.
            scope: Metadata-only or numeric-payload access.
            payload_names: Exact numeric or objective payload names opened by
                the access.
        """
        role = self.manifest.role_for_trajectory(trajectory_id)
        trajectory = self.manifest.trajectory(trajectory_id)
        canonical_scope = _access_scope(scope)
        canonical_payload_names = _payload_names(payload_names)
        payload_identity_sha256 = (
            _payload_identity_digest(trajectory, canonical_payload_names)
            if canonical_scope is DataAccessScope.PAYLOAD
            else None
        )
        previous_sha256 = self.accesses[-1].access_sha256 if self.accesses else self.manifest.manifest_sha256
        access = DataAccessRecord(
            sequence=len(self.accesses),
            trajectory_id=trajectory_id,
            role=role,
            purpose=_access_purpose(purpose),
            scope=canonical_scope,
            payload_names=canonical_payload_names,
            payload_identity_sha256=payload_identity_sha256,
            previous_sha256=previous_sha256,
        )
        return DataAccessLedger(self.manifest, (*self.accesses, access))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _ACCESS_CONTRACT,
            "manifest_sha256": self.manifest.manifest_sha256,
            "accesses": [access.as_dict() for access in self.accesses],
            "confirmation_payload_released": self.confirmation_payload_released,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible access ledger."""
        payload = self._payload()
        payload["ledger_sha256"] = self.ledger_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class SamplingReference:
    """One scheduled sample selected through its trajectory and static layout."""

    trajectory_id: str
    trajectory_sha256: str
    topology_sha256: str
    operator_geometry_sha256: str
    pin_signature_sha256: str
    static_layout_sha256: str
    sample_id: str
    sample_sha256: str
    physical_step_sha256: str
    common_objective_sha256: str
    ordinal: int

    def __post_init__(self) -> None:
        _identifier(self.trajectory_id, "scheduled trajectory_id")
        _identifier(self.sample_id, "scheduled sample_id")
        for name in (
            "trajectory_sha256",
            "topology_sha256",
            "operator_geometry_sha256",
            "pin_signature_sha256",
            "static_layout_sha256",
            "sample_sha256",
            "physical_step_sha256",
            "common_objective_sha256",
        ):
            _sha256(getattr(self, name), f"scheduled {name}")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("scheduled ordinal must be a non-negative integer")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible scheduled reference."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SamplingBatch:
    """One topology/operator/material/pin/timestep homogeneous batch.

    Static-layout compatibility does not mean the samples share a physical
    objective. Inertial targets, external loads, and history remain per-sample
    inputs and require per-sample evaluation with the current unbatched
    objective implementation.
    """

    topology_sha256: str
    operator_geometry_sha256: str
    pin_signature_sha256: str
    static_layout_sha256: str
    samples: tuple[SamplingReference, ...]

    def __post_init__(self) -> None:
        for name in (
            "topology_sha256",
            "operator_geometry_sha256",
            "pin_signature_sha256",
            "static_layout_sha256",
        ):
            _sha256(getattr(self, name), f"batch {name}")
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("sampling batch must not be empty")
        if any(type(sample) is not SamplingReference for sample in samples):
            raise ValueError("sampling batch must contain canonical SamplingReference values")
        if any(sample.topology_sha256 != self.topology_sha256 for sample in samples):
            raise ValueError("sampling batch mixes topologies")
        if any(sample.operator_geometry_sha256 != self.operator_geometry_sha256 for sample in samples):
            raise ValueError("sampling batch mixes operator geometries")
        if any(sample.pin_signature_sha256 != self.pin_signature_sha256 for sample in samples):
            raise ValueError("sampling batch mixes pin signatures")
        if any(sample.static_layout_sha256 != self.static_layout_sha256 for sample in samples):
            raise ValueError("sampling batch mixes static layouts")
        object.__setattr__(self, "samples", samples)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible sampling batch."""
        return {
            "topology_sha256": self.topology_sha256,
            "operator_geometry_sha256": self.operator_geometry_sha256,
            "pin_signature_sha256": self.pin_signature_sha256,
            "static_layout_sha256": self.static_layout_sha256,
            "physical_objective_routing": _OBJECTIVE_ROUTING,
            "samples": [sample.as_dict() for sample in self.samples],
        }


@dataclasses.dataclass(frozen=True)
class SamplingSchedule:
    """Immutable deterministic static-layout-homogeneous sample stream.

    A batch is homogeneous only in topology, source operator, material, pin
    signature, and timestep. The schedule explicitly requires physical objectives to be
    reconstructed and evaluated per sample; it does not authenticate a shared
    inertial target, external load, or history context.
    """

    manifest: SplitManifest
    role: DatasetRole
    seed: int
    steps: int
    batch_size: int
    batches: tuple[SamplingBatch, ...]
    manifest_sha256: str = dataclasses.field(init=False)
    trajectory_count: int = dataclasses.field(init=False)
    trajectory_epoch_count: int = dataclasses.field(init=False)
    trajectory_exposure: tuple[tuple[str, int], ...] = dataclasses.field(init=False)
    static_layout_exposure: tuple[tuple[str, str, int], ...] = dataclasses.field(init=False)
    sample_exposure: tuple[tuple[str, str, int], ...] = dataclasses.field(init=False)
    schedule_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _verify_manifest(self.manifest)
        object.__setattr__(self, "manifest_sha256", self.manifest.manifest_sha256)
        object.__setattr__(self, "role", _dataset_role(self.role))
        role_records = self.manifest.records(self.role)
        if not role_records:
            raise ValueError(f"cannot authenticate a schedule for the empty {self.role.value} role")
        object.__setattr__(self, "trajectory_count", len(role_records))
        records_by_id = {record.trajectory_id: record for record in role_records}
        samples_by_trajectory = {
            record.trajectory_id: {sample.sample_id: sample for sample in record.samples} for record in role_records
        }

        for name in ("seed", "steps", "batch_size"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        batches = tuple(self.batches)
        if len(batches) != self.steps:
            raise ValueError("sampling batch count must equal steps")
        if any(type(batch) is not SamplingBatch for batch in batches):
            raise ValueError("batches must contain canonical SamplingBatch values")
        if any(len(batch.samples) != self.batch_size for batch in batches):
            raise ValueError("every sampling batch must have batch_size samples")
        for batch in batches:
            for reference in batch.samples:
                if reference.topology_sha256 != batch.topology_sha256:
                    raise ValueError("scheduled reference disagrees with its batch topology")
                if reference.operator_geometry_sha256 != batch.operator_geometry_sha256:
                    raise ValueError("scheduled reference disagrees with its batch operator geometry")
                if reference.pin_signature_sha256 != batch.pin_signature_sha256:
                    raise ValueError("scheduled reference disagrees with its batch pin signature")
                if reference.static_layout_sha256 != batch.static_layout_sha256:
                    raise ValueError("scheduled reference disagrees with its batch static layout")
                trajectory = records_by_id.get(reference.trajectory_id)
                if trajectory is None:
                    raise ValueError("scheduled trajectory does not belong to the declared manifest role")
                if reference.trajectory_sha256 != trajectory.trajectory_sha256:
                    raise ValueError("scheduled trajectory hash disagrees with the split manifest")
                if reference.topology_sha256 != trajectory.topology_sha256:
                    raise ValueError("scheduled topology hash disagrees with the split manifest")
                if reference.operator_geometry_sha256 != trajectory.operator_geometry_sha256:
                    raise ValueError("scheduled operator-geometry hash disagrees with the split manifest")
                sample = samples_by_trajectory[trajectory.trajectory_id].get(reference.sample_id)
                if sample is None:
                    raise ValueError("scheduled sample does not belong to its trajectory")
                if reference.pin_signature_sha256 != sample.pin_signature_sha256:
                    raise ValueError("scheduled pin signature disagrees with the split manifest")
                if reference.static_layout_sha256 != sample.static_layout_sha256:
                    raise ValueError("scheduled static layout disagrees with the split manifest")
                if reference.physical_step_sha256 != sample.physical_step_sha256:
                    raise ValueError("scheduled physical-step identity disagrees with the split manifest")
                if reference.common_objective_sha256 != sample.common_objective_sha256:
                    raise ValueError("scheduled common-objective identity disagrees with the split manifest")
                if reference.sample_sha256 != sample.sample_sha256 or reference.ordinal != sample.ordinal:
                    raise ValueError("scheduled sample identity disagrees with the split manifest")
        if self.steps % self.trajectory_count != 0:
            raise ValueError("sampling steps must contain complete trajectory epochs")
        object.__setattr__(self, "batches", batches)
        object.__setattr__(self, "trajectory_epoch_count", self.steps // self.trajectory_count)
        layouts_by_trajectory = {
            record.trajectory_id: tuple(sorted({sample.static_layout_sha256 for sample in record.samples}))
            for record in role_records
        }
        if any(self.trajectory_epoch_count % len(layouts) != 0 for layouts in layouts_by_trajectory.values()):
            raise ValueError("sampling steps must contain complete static-layout cycles for every trajectory")
        expected_batches = _build_sampling_batches(
            role_records,
            steps=self.steps,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        if batches != expected_batches:
            raise ValueError("sampling batches do not match deterministic PCG64 replay")

        trajectory_counts: dict[str, int] = {}
        layout_counts: dict[tuple[str, str], int] = {}
        sample_counts: dict[tuple[str, str], int] = {}
        for batch in batches:
            for sample in batch.samples:
                trajectory_counts[sample.trajectory_id] = trajectory_counts.get(sample.trajectory_id, 0) + 1
                layout_key = (sample.trajectory_id, sample.static_layout_sha256)
                layout_counts[layout_key] = layout_counts.get(layout_key, 0) + 1
                sample_key = (sample.trajectory_id, sample.sample_id)
                sample_counts[sample_key] = sample_counts.get(sample_key, 0) + 1
        object.__setattr__(self, "trajectory_exposure", tuple(sorted(trajectory_counts.items())))
        expected_exposure = self.trajectory_epoch_count * self.batch_size
        if len(trajectory_counts) != self.trajectory_count or any(
            count != expected_exposure for count in trajectory_counts.values()
        ):
            raise ValueError("sampling schedule is not exactly balanced across trajectories")
        expected_layout_exposure = {
            (trajectory_id, layout_sha256): self.trajectory_epoch_count // len(layouts) * self.batch_size
            for trajectory_id, layouts in layouts_by_trajectory.items()
            for layout_sha256 in layouts
        }
        if layout_counts != expected_layout_exposure:
            raise ValueError("sampling schedule is not exactly balanced across static layouts")
        object.__setattr__(
            self,
            "static_layout_exposure",
            tuple(
                (trajectory_id, layout_sha256, count)
                for (trajectory_id, layout_sha256), count in sorted(layout_counts.items())
            ),
        )
        object.__setattr__(
            self,
            "sample_exposure",
            tuple(
                (trajectory_id, sample_id, count) for (trajectory_id, sample_id), count in sorted(sample_counts.items())
            ),
        )
        object.__setattr__(self, "schedule_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _SAMPLING_CONTRACT,
            "generator": "numpy.random.Generator(PCG64)",
            "selection_order": "trajectory -> static_layout(topology,operator,pins,material,dt) -> sample",
            "physical_objective_routing": _OBJECTIVE_ROUTING,
            "manifest_sha256": self.manifest_sha256,
            "role": self.role.value,
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "trajectory_count": self.trajectory_count,
            "trajectory_epoch_count": self.trajectory_epoch_count,
            "batches": [batch.as_dict() for batch in self.batches],
            "trajectory_exposure": dict(self.trajectory_exposure),
            "static_layout_exposure": [
                {"trajectory_id": trajectory_id, "static_layout_sha256": layout_sha256, "count": count}
                for trajectory_id, layout_sha256, count in self.static_layout_exposure
            ],
            "sample_exposure": [
                {"trajectory_id": trajectory_id, "sample_id": sample_id, "count": count}
                for trajectory_id, sample_id, count in self.sample_exposure
            ],
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON-compatible sampling schedule."""
        payload = self._payload()
        payload["schedule_sha256"] = self.schedule_sha256
        return payload


class _ShuffledCycle:
    def __init__(self, values: Sequence[object], rng: np.random.Generator):
        self._values = tuple(values)
        if not self._values:
            raise ValueError("balanced sampling cycle must not be empty")
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


def _build_sampling_batches(
    records: tuple[TrajectoryRecord, ...],
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[SamplingBatch, ...]:
    """Replay the canonical PCG64 trajectory/layout/sample hierarchy."""
    rng = np.random.Generator(np.random.PCG64(seed))
    trajectory_cycle = _ShuffledCycle(records, rng)
    layout_cycles: dict[str, _ShuffledCycle] = {}
    samples_by_layout: dict[tuple[str, str], tuple[TrajectorySampleRecord, ...]] = {}
    sample_cycles: dict[tuple[str, str], _ShuffledCycle] = {}
    for record in records:
        layout_hashes = tuple(sorted({sample.static_layout_sha256 for sample in record.samples}))
        layout_cycles[record.trajectory_id] = _ShuffledCycle(layout_hashes, rng)
        for layout_sha256 in layout_hashes:
            layout_samples = tuple(sample for sample in record.samples if sample.static_layout_sha256 == layout_sha256)
            key = (record.trajectory_id, layout_sha256)
            samples_by_layout[key] = layout_samples
            sample_cycles[key] = _ShuffledCycle(layout_samples, rng)

    batches: list[SamplingBatch] = []
    for _step in range(steps):
        trajectory = trajectory_cycle.next()
        if type(trajectory) is not TrajectoryRecord:
            raise RuntimeError("internal trajectory schedule is malformed")
        static_layout_sha256 = layout_cycles[trajectory.trajectory_id].next()
        if type(static_layout_sha256) is not str:
            raise RuntimeError("internal static-layout schedule is malformed")
        layout_key = (trajectory.trajectory_id, static_layout_sha256)
        layout_samples = samples_by_layout[layout_key]
        representative = layout_samples[0]
        references: list[SamplingReference] = []
        for _sample_index in range(batch_size):
            sample = sample_cycles[layout_key].next()
            if type(sample) is not TrajectorySampleRecord:
                raise RuntimeError("internal sample schedule is malformed")
            references.append(
                SamplingReference(
                    trajectory_id=trajectory.trajectory_id,
                    trajectory_sha256=trajectory.trajectory_sha256,
                    topology_sha256=trajectory.topology_sha256,
                    operator_geometry_sha256=trajectory.operator_geometry_sha256,
                    pin_signature_sha256=sample.pin_signature_sha256,
                    static_layout_sha256=sample.static_layout_sha256,
                    sample_id=sample.sample_id,
                    sample_sha256=sample.sample_sha256,
                    physical_step_sha256=sample.physical_step_sha256,
                    common_objective_sha256=sample.common_objective_sha256,
                    ordinal=sample.ordinal,
                )
            )
        batches.append(
            SamplingBatch(
                topology_sha256=trajectory.topology_sha256,
                operator_geometry_sha256=trajectory.operator_geometry_sha256,
                pin_signature_sha256=representative.pin_signature_sha256,
                static_layout_sha256=static_layout_sha256,
                samples=tuple(references),
            )
        )
    return tuple(batches)


def build_sampling_schedule(
    manifest: SplitManifest,
    *,
    role: DatasetRole | str = DatasetRole.TRAIN,
    steps: int,
    batch_size: int,
    seed: int,
) -> SamplingSchedule:
    """Build a deterministic, trajectory-balanced, static-layout schedule.

    Each complete outer cycle selects every trajectory exactly once. For each
    selected trajectory, one of its static layouts is chosen from a balanced
    shuffled cycle, then the entire batch is drawn only from samples in that
    layout. Every trajectory therefore receives exactly ``batch_size``
    selections per outer cycle regardless of its recorded length, while no
    batch mixes topology, source operator, pin set, material, or timestep.

    Static-layout homogeneity is not physical-objective compatibility.
    Inertial targets, external loads, and history can differ between samples
    in the same batch. Callers must evaluate those objectives per sample with
    the current unbatched objective implementation (or provide a genuinely
    batched objective implementation).

    Args:
        manifest: Frozen whole-trajectory split manifest.
        role: Dataset role to schedule.
        steps: Number of batches; must be a multiple of the trajectory count.
        batch_size: Number of samples per static-layout-homogeneous batch.
        seed: Non-negative PCG64 seed.
    """
    _verify_manifest(manifest)
    canonical_role = _dataset_role(role)
    for name, value in (("steps", steps), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    records = manifest.records(canonical_role)
    if not records:
        raise ValueError(f"cannot sample the empty {canonical_role.value} role")
    if steps % len(records) != 0:
        raise ValueError("steps must be divisible by the trajectory count for exact balance")
    trajectory_epoch_count = steps // len(records)
    for record in records:
        layout_count = len({sample.static_layout_sha256 for sample in record.samples})
        if trajectory_epoch_count % layout_count != 0:
            raise ValueError("steps must complete every trajectory's static-layout cycle for exact balance")

    return SamplingSchedule(
        manifest=manifest,
        role=canonical_role,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        batches=_build_sampling_batches(records, steps=steps, batch_size=batch_size, seed=seed),
    )
