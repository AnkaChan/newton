# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Portable-volume dataset records for principal-stretch research.

This module is a successor identity domain, not a reinterpretation of the v5
dataset.  Every serialized record has a new contract and digest domain and
binds the host-authenticated operator-volume policy and SHA-256.  The access
ledger is branch-local audit evidence; it is not global access control.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import struct
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import torch

from .torch_solver import (
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
    OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT,
)
from .v5_dataset import DataAccessPurpose, DataAccessScope, DatasetRole

PORTABLE_DATASET_SCHEMA_VERSION = 1

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

_NUMERIC_IDENTITY_CONTRACT = "pss-portable-volume-numeric-content-v1"
_REFERENCE_SOURCE_TRANSITION_CONTRACT = "pss-portable-volume-reference-source-transition-preimage-v1"
_REFERENCE_NUMERIC_IDENTIFIER_PREFIX = "reference-sequence-portable-volume-v1"
_REFERENCE_SEQUENCE_PROVENANCE_CONTRACT = "pss-portable-volume-reference-sequence-provenance-v1"
_SAMPLE_CONTRACT = "pss-portable-volume-dataset-sample-v1"
_STATIC_LAYOUT_CONTRACT = "pss-portable-volume-dataset-static-layout-v1"
_COMMON_OBJECTIVE_BINDING_CONTRACT = "pss-portable-volume-common-objective-binding-v1"
_TRAJECTORY_CONTRACT = "pss-portable-volume-dataset-trajectory-v1"
_SPLIT_CONTRACT = "pss-portable-volume-dataset-split-v1"
_ACCESS_RECORD_CONTRACT = "pss-portable-volume-dataset-access-record-v1"
_ACCESS_LEDGER_CONTRACT = "pss-portable-volume-dataset-access-ledger-v1"
_PAYLOAD_SELECTION_CONTRACT = "pss-portable-volume-dataset-payload-selection-v1"
_SAMPLING_REFERENCE_CONTRACT = "pss-portable-volume-dataset-sampling-reference-v1"
_SAMPLING_BATCH_CONTRACT = "pss-portable-volume-dataset-sampling-batch-v1"
_SAMPLING_CONTRACT = "pss-portable-volume-dataset-sampling-v1"
_TRAINING_TENSOR_CONTRACT = b"pss-portable-volume-training-tensor-v1\0"
_REFERENCE_SEQUENCE_ARRAY_DIGEST_CONTRACT = "numpy-little-endian-dtype-shape-c-order-sha256-v1"
_PHYSICAL_INTEGRATION_POLICIES = (
    "algebraic-float64-position-history-loads-v1",
    "solver-vbd-staged-float32-v1",
)
_OBJECTIVE_ROUTING = "per-sample-unbatched-portable-volume-bound-objective-v1"
_SAMPLING_GENERATOR = "numpy.random.Generator(PCG64)"
_SAMPLING_ORDER = "trajectory -> portable_static_layout -> sample"
_LEDGER_CLAIM_SCOPE = "branch-local-evidence-not-global-access-control"

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


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(first, second) for first, second in zip(left, right, strict=True)
        )
    return left == right


def _strict_mapping(value: object, expected: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} keys must be exactly {tuple(sorted(expected))}")
    return value


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_json_object(value: str | bytes, name: str) -> dict[str, object]:
    if type(value) is bytes:
        try:
            value = value.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError(f"{name} must be UTF-8 JSON") from exc
    if type(value) is not str:
        raise TypeError(f"{name} must be str or bytes")
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if type(payload) is not dict:
        raise ValueError(f"{name} root must be a JSON object")
    return payload


def _canonical_json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_exact_round_trip(value: object, expected: object, name: str) -> None:
    if not _exact_json_equal(value, expected):
        raise ValueError(f"{name} is not the exact canonical portable dataset record")


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
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float64 value")
    return result


def _float64_bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def _dataset_role(value: DatasetRole | str) -> DatasetRole:
    try:
        return DatasetRole(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"role must be one of {tuple(role.value for role in DatasetRole)}") from exc


def _access_purpose(value: DataAccessPurpose | str) -> DataAccessPurpose:
    try:
        return DataAccessPurpose(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"purpose must be one of {tuple(item.value for item in DataAccessPurpose)}") from exc


def _access_scope(value: DataAccessScope | str) -> DataAccessScope:
    try:
        return DataAccessScope(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scope must be one of {tuple(item.value for item in DataAccessScope)}") from exc


def _validate_physical_integration_identity(policy: object, evidence_sha256: object, subject: str) -> None:
    if type(policy) is not str or policy not in _PHYSICAL_INTEGRATION_POLICIES:
        raise ValueError(f"{subject} physical_integration_policy is not registered canonical text")
    if policy == _PHYSICAL_INTEGRATION_POLICIES[0]:
        if evidence_sha256 is not None:
            raise ValueError(f"{subject} algebraic integration must not name source evidence")
    else:
        _sha256(evidence_sha256, f"{subject} source_integration_evidence_sha256")


def canonical_portable_training_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one tensor in a device-independent successor digest domain."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("authenticated training values must be torch.Tensor instances")
    if tensor.layout != torch.strided:
        raise ValueError("authenticated training tensors must have strided layout")
    value = tensor.detach().contiguous()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("authenticated training tensors must be finite")
    try:
        array = value.cpu().numpy()
    except TypeError as exc:
        raise ValueError("authenticated training tensor dtype is not supported by the portable contract") from exc
    canonical_dtype = np.dtype(array.dtype).newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    raw = canonical.tobytes(order="C")
    digest = hashlib.sha256(_TRAINING_TENSOR_CONTRACT)
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class PortableNumericContentIdentity:
    """Logical identifier and portable byte-level hash for one tensor."""

    identifier: str
    sha256: str
    identity_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.identifier, "numeric content identifier")
        _sha256(self.sha256, "numeric content sha256")
        object.__setattr__(self, "identity_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _NUMERIC_IDENTITY_CONTRACT,
            "identifier": self.identifier,
            "sha256": self.sha256,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["identity_sha256"] = self.identity_sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> PortableNumericContentIdentity:
        """Strictly reconstruct one portable numeric identity."""
        payload = _strict_mapping(
            value,
            {"schema_version", "contract", "identifier", "sha256", "identity_sha256"},
            "portable numeric identity",
        )
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _NUMERIC_IDENTITY_CONTRACT,
        ):
            raise ValueError("portable numeric identity has an unregistered schema identity")
        result = cls(identifier=payload["identifier"], sha256=payload["sha256"])
        _require_exact_round_trip(result.as_dict(), payload, "portable numeric identity")
        return result


@dataclasses.dataclass(frozen=True)
class PortableReferenceSourceTransitionIdentity:
    """Sealed preimage for one authenticated reference transition."""

    reference_sequence_index_sha256: str
    asset_id: str
    asset_source_sha256: str
    sequence_id: str
    step_id: int
    static_npz_sha256: str
    sequence_npz_sha256: str
    protocol_sha256: str
    producer_topology_sha256: str
    producer_operator_sha256: str
    producer_material_sha256: str
    accepted_reference_state_sha256: str
    portable_topology_sha256: str
    operator_geometry_policy: str
    operator_geometry_sha256: str
    operator_volume_policy: str
    operator_volume_sha256: str
    portable_material_sha256: str
    portable_pin_signature_sha256: str
    source_transition_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        for name in ("asset_id", "sequence_id"):
            _identifier(getattr(self, name), f"source transition {name}")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("source transition step_id must be a non-negative integer")
        if self.operator_geometry_policy != OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME:
            raise ValueError("source transition must use the registered portable operator geometry policy")
        if self.operator_volume_policy != OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT:
            raise ValueError("source transition must use the registered portable operator volume policy")
        for name in (
            "reference_sequence_index_sha256",
            "asset_source_sha256",
            "static_npz_sha256",
            "sequence_npz_sha256",
            "protocol_sha256",
            "producer_topology_sha256",
            "producer_operator_sha256",
            "producer_material_sha256",
            "accepted_reference_state_sha256",
            "portable_topology_sha256",
            "operator_geometry_sha256",
            "operator_volume_sha256",
            "portable_material_sha256",
            "portable_pin_signature_sha256",
        ):
            _sha256(getattr(self, name), f"source transition {name}")
        object.__setattr__(self, "source_transition_sha256", _canonical_digest(self._payload()))

    @property
    def producer_static_identity(self) -> tuple[str, str, str]:
        """Return producer topology, operator, and material identities."""
        return (
            self.producer_topology_sha256,
            self.producer_operator_sha256,
            self.producer_material_sha256,
        )

    @property
    def portable_static_identity(self) -> tuple[str, str, str, str, str, str, str]:
        """Return the complete portable runtime-static identity."""
        return (
            self.portable_topology_sha256,
            self.operator_geometry_policy,
            self.operator_geometry_sha256,
            self.operator_volume_policy,
            self.operator_volume_sha256,
            self.portable_material_sha256,
            self.portable_pin_signature_sha256,
        )

    def numeric_identifier(self, component: str) -> str:
        """Return the only registered numeric identifier for one component."""
        if component not in _NUMERIC_COMPONENTS:
            raise ValueError(f"numeric component must be one of {_NUMERIC_COMPONENTS}")
        return (
            f"{_REFERENCE_NUMERIC_IDENTIFIER_PREFIX}:{self.reference_sequence_index_sha256}:"
            f"{self.source_transition_sha256}:{component}"
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _REFERENCE_SOURCE_TRANSITION_CONTRACT,
            "reference_sequence_index_sha256": self.reference_sequence_index_sha256,
            "transition_key": {
                "asset_id": self.asset_id,
                "sequence_id": self.sequence_id,
                "step_id": self.step_id,
            },
            "source_artifacts": {
                "asset_source_sha256": self.asset_source_sha256,
                "static_npz_sha256": self.static_npz_sha256,
                "sequence_npz_sha256": self.sequence_npz_sha256,
                "protocol_sha256": self.protocol_sha256,
                "accepted_reference_state_sha256": self.accepted_reference_state_sha256,
            },
            "producer_static": {
                "topology_sha256": self.producer_topology_sha256,
                "operator_sha256": self.producer_operator_sha256,
                "material_sha256": self.producer_material_sha256,
            },
            "portable_static": {
                "topology_sha256": self.portable_topology_sha256,
                "operator_geometry": {
                    "policy": self.operator_geometry_policy,
                    "sha256": self.operator_geometry_sha256,
                },
                "operator_volume": {
                    "policy": self.operator_volume_policy,
                    "sha256": self.operator_volume_sha256,
                },
                "material_sha256": self.portable_material_sha256,
                "pin_signature_sha256": self.portable_pin_signature_sha256,
            },
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["source_transition_sha256"] = self.source_transition_sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> PortableReferenceSourceTransitionIdentity:
        """Strictly reconstruct one reference source-transition preimage."""
        payload = _strict_mapping(
            value,
            {
                "schema_version",
                "contract",
                "reference_sequence_index_sha256",
                "transition_key",
                "source_artifacts",
                "producer_static",
                "portable_static",
                "source_transition_sha256",
            },
            "portable reference source transition",
        )
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _REFERENCE_SOURCE_TRANSITION_CONTRACT,
        ):
            raise ValueError("portable reference source transition has an unregistered schema identity")
        key = _strict_mapping(payload["transition_key"], {"asset_id", "sequence_id", "step_id"}, "transition_key")
        source = _strict_mapping(
            payload["source_artifacts"],
            {
                "asset_source_sha256",
                "static_npz_sha256",
                "sequence_npz_sha256",
                "protocol_sha256",
                "accepted_reference_state_sha256",
            },
            "source_artifacts",
        )
        producer = _strict_mapping(
            payload["producer_static"],
            {"topology_sha256", "operator_sha256", "material_sha256"},
            "producer_static",
        )
        portable = _strict_mapping(
            payload["portable_static"],
            {"topology_sha256", "operator_geometry", "operator_volume", "material_sha256", "pin_signature_sha256"},
            "portable_static",
        )
        geometry = _strict_mapping(portable["operator_geometry"], {"policy", "sha256"}, "operator_geometry")
        volume = _strict_mapping(portable["operator_volume"], {"policy", "sha256"}, "operator_volume")
        result = cls(
            reference_sequence_index_sha256=payload["reference_sequence_index_sha256"],
            asset_id=key["asset_id"],
            asset_source_sha256=source["asset_source_sha256"],
            sequence_id=key["sequence_id"],
            step_id=key["step_id"],
            static_npz_sha256=source["static_npz_sha256"],
            sequence_npz_sha256=source["sequence_npz_sha256"],
            protocol_sha256=source["protocol_sha256"],
            producer_topology_sha256=producer["topology_sha256"],
            producer_operator_sha256=producer["operator_sha256"],
            producer_material_sha256=producer["material_sha256"],
            accepted_reference_state_sha256=source["accepted_reference_state_sha256"],
            portable_topology_sha256=portable["topology_sha256"],
            operator_geometry_policy=geometry["policy"],
            operator_geometry_sha256=geometry["sha256"],
            operator_volume_policy=volume["policy"],
            operator_volume_sha256=volume["sha256"],
            portable_material_sha256=portable["material_sha256"],
            portable_pin_signature_sha256=portable["pin_signature_sha256"],
        )
        _require_exact_round_trip(result.as_dict(), payload, "portable reference source transition")
        return result


@dataclasses.dataclass(frozen=True)
class PortableReferenceSequenceProvenance:
    """Relocation-stable provenance for one authenticated sequence shard."""

    dataset_index_uri: str
    dataset_index_sha256: str
    asset_id: str
    asset_source_sha256: str
    producer_topology_sha256: str
    producer_operator_sha256: str
    producer_material_sha256: str
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
    accepted_reference_state_sha256: tuple[str, ...]
    deformation_seed: int
    velocity_seed: int
    source_transition_count: int
    requested_dt_seconds: float
    dt_seconds: float
    execution_dt_float32_bits: str
    dt_float64_bits: str = dataclasses.field(init=False)
    provenance_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "dataset_index_uri",
            "producer_manifest_uri",
            "static_bundle_uri",
            "sequence_bundle_uri",
            "evidence_uri",
        ):
            _canonical_uri(getattr(self, name), name)
        for name in ("asset_id", "sequence_id"):
            _identifier(getattr(self, name), name)
        for name in (
            "dataset_index_sha256",
            "asset_source_sha256",
            "producer_topology_sha256",
            "producer_operator_sha256",
            "producer_material_sha256",
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
            _sha256(getattr(self, name), name)
        if type(self.source_transition_count) is not int or self.source_transition_count < 1:
            raise ValueError("source_transition_count must be a positive integer")
        if isinstance(self.accepted_reference_state_sha256, (str, bytes)):
            raise ValueError("accepted_reference_state_sha256 must be a sequence of digests")
        accepted = tuple(self.accepted_reference_state_sha256)
        if len(accepted) != self.source_transition_count:
            raise ValueError("accepted reference-state identities must align with every source transition")
        for step_id, digest in enumerate(accepted):
            _sha256(digest, f"accepted reference state {step_id} sha256")
        object.__setattr__(self, "accepted_reference_state_sha256", accepted)
        for name in ("deformation_seed", "velocity_seed"):
            seed = getattr(self, name)
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        requested_dt = _positive_float64(self.requested_dt_seconds, "requested_dt_seconds")
        dt_seconds = _positive_float64(self.dt_seconds, "dt_seconds")
        expected_execution_dt = float(struct.unpack("<f", struct.pack("<f", requested_dt))[0])
        if dt_seconds != expected_execution_dt:
            raise ValueError("dt_seconds must be the exact float32 execution of requested_dt_seconds")
        expected_float32_bits = f"0x{struct.unpack('<I', struct.pack('<f', dt_seconds))[0]:08x}"
        if self.execution_dt_float32_bits != expected_float32_bits:
            raise ValueError("execution_dt_float32_bits disagrees with dt_seconds")
        object.__setattr__(self, "requested_dt_seconds", requested_dt)
        object.__setattr__(self, "dt_seconds", dt_seconds)
        object.__setattr__(self, "dt_float64_bits", _float64_bits(dt_seconds))
        object.__setattr__(self, "provenance_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _REFERENCE_SEQUENCE_PROVENANCE_CONTRACT,
            "dataset_index": {"uri": self.dataset_index_uri, "sha256": self.dataset_index_sha256},
            "asset_id": self.asset_id,
            "asset_source_sha256": self.asset_source_sha256,
            "producer_static": {
                "topology_sha256": self.producer_topology_sha256,
                "operator_sha256": self.producer_operator_sha256,
                "material_sha256": self.producer_material_sha256,
            },
            "sequence_id": self.sequence_id,
            "producer_manifest": {
                "uri": self.producer_manifest_uri,
                "sha256": self.producer_manifest_sha256,
            },
            "artifacts": {
                "static_bundle": {"uri": self.static_bundle_uri, "sha256": self.static_bundle_sha256},
                "sequence_bundle": {"uri": self.sequence_bundle_uri, "sha256": self.sequence_bundle_sha256},
                "evidence": {"uri": self.evidence_uri, "sha256": self.evidence_sha256},
            },
            "protocol_sha256": self.protocol_sha256,
            "state_anchors": {
                "digest_contract": _REFERENCE_SEQUENCE_ARRAY_DIGEST_CONTRACT,
                "initial": {
                    "position_sha256": self.initial_position_sha256,
                    "velocity_field_sha256": self.initial_velocity_field_sha256,
                },
                "final": {
                    "position_sha256": self.final_position_sha256,
                    "velocity_field_sha256": self.final_velocity_field_sha256,
                },
                "accepted_reference_state_sha256": list(self.accepted_reference_state_sha256),
            },
            "deformation_seed": self.deformation_seed,
            "velocity_seed": self.velocity_seed,
            "source_transition_count": self.source_transition_count,
            "requested_dt_seconds": self.requested_dt_seconds,
            "dt_seconds": self.dt_seconds,
            "execution_dt_float32_bits": self.execution_dt_float32_bits,
            "dt_float64_bits": self.dt_float64_bits,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["provenance_sha256"] = self.provenance_sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> PortableReferenceSequenceProvenance:
        """Strictly reconstruct sequence provenance."""
        root_keys = {
            "schema_version",
            "contract",
            "dataset_index",
            "asset_id",
            "asset_source_sha256",
            "producer_static",
            "sequence_id",
            "producer_manifest",
            "artifacts",
            "protocol_sha256",
            "state_anchors",
            "deformation_seed",
            "velocity_seed",
            "source_transition_count",
            "requested_dt_seconds",
            "dt_seconds",
            "execution_dt_float32_bits",
            "dt_float64_bits",
            "provenance_sha256",
        }
        payload = _strict_mapping(value, root_keys, "portable reference-sequence provenance")
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _REFERENCE_SEQUENCE_PROVENANCE_CONTRACT,
        ):
            raise ValueError("portable reference-sequence provenance has an unregistered schema identity")
        dataset_index = _strict_mapping(payload["dataset_index"], {"uri", "sha256"}, "dataset_index")
        producer_manifest = _strict_mapping(payload["producer_manifest"], {"uri", "sha256"}, "producer_manifest")
        artifacts = _strict_mapping(
            payload["artifacts"], {"static_bundle", "sequence_bundle", "evidence"}, "provenance artifacts"
        )
        static_bundle = _strict_mapping(artifacts["static_bundle"], {"uri", "sha256"}, "static_bundle")
        sequence_bundle = _strict_mapping(artifacts["sequence_bundle"], {"uri", "sha256"}, "sequence_bundle")
        evidence = _strict_mapping(artifacts["evidence"], {"uri", "sha256"}, "evidence")
        state_anchors = _strict_mapping(
            payload["state_anchors"],
            {"digest_contract", "initial", "final", "accepted_reference_state_sha256"},
            "state_anchors",
        )
        if state_anchors["digest_contract"] != _REFERENCE_SEQUENCE_ARRAY_DIGEST_CONTRACT:
            raise ValueError("state-anchor digest contract is not registered")
        initial = _strict_mapping(
            state_anchors["initial"], {"position_sha256", "velocity_field_sha256"}, "initial state anchor"
        )
        final = _strict_mapping(
            state_anchors["final"], {"position_sha256", "velocity_field_sha256"}, "final state anchor"
        )
        accepted = state_anchors["accepted_reference_state_sha256"]
        if type(accepted) is not list:
            raise ValueError("accepted_reference_state_sha256 must be a JSON list")
        producer = _strict_mapping(
            payload["producer_static"],
            {"topology_sha256", "operator_sha256", "material_sha256"},
            "producer_static",
        )
        result = cls(
            dataset_index_uri=dataset_index["uri"],
            dataset_index_sha256=dataset_index["sha256"],
            asset_id=payload["asset_id"],
            asset_source_sha256=payload["asset_source_sha256"],
            producer_topology_sha256=producer["topology_sha256"],
            producer_operator_sha256=producer["operator_sha256"],
            producer_material_sha256=producer["material_sha256"],
            sequence_id=payload["sequence_id"],
            producer_manifest_uri=producer_manifest["uri"],
            producer_manifest_sha256=producer_manifest["sha256"],
            static_bundle_uri=static_bundle["uri"],
            static_bundle_sha256=static_bundle["sha256"],
            sequence_bundle_uri=sequence_bundle["uri"],
            sequence_bundle_sha256=sequence_bundle["sha256"],
            evidence_uri=evidence["uri"],
            evidence_sha256=evidence["sha256"],
            protocol_sha256=payload["protocol_sha256"],
            initial_position_sha256=initial["position_sha256"],
            initial_velocity_field_sha256=initial["velocity_field_sha256"],
            final_position_sha256=final["position_sha256"],
            final_velocity_field_sha256=final["velocity_field_sha256"],
            accepted_reference_state_sha256=tuple(accepted),
            deformation_seed=payload["deformation_seed"],
            velocity_seed=payload["velocity_seed"],
            source_transition_count=payload["source_transition_count"],
            requested_dt_seconds=payload["requested_dt_seconds"],
            dt_seconds=payload["dt_seconds"],
            execution_dt_float32_bits=payload["execution_dt_float32_bits"],
        )
        _require_exact_round_trip(result.as_dict(), payload, "portable reference-sequence provenance")
        return result


@dataclasses.dataclass(frozen=True)
class PortableDatasetSampleRecord:
    """One immutable portable-volume sample identity."""

    sample_id: str
    ordinal: int
    topology_sha256: str
    operator_geometry_policy: str
    operator_geometry_sha256: str
    operator_volume_policy: str
    operator_volume_sha256: str
    material_sha256: str
    pin_signature_sha256: str
    dt_seconds: float
    physical_step_sha256: str
    physical_integration_policy: str
    source_integration_evidence_sha256: str | None
    source_transition: PortableReferenceSourceTransitionIdentity | None
    common_objective_sha256: str
    observed_f: PortableNumericContentIdentity
    input_f: PortableNumericContentIdentity
    reference_f: PortableNumericContentIdentity
    observed_state: PortableNumericContentIdentity
    input_state: PortableNumericContentIdentity
    reference_state: PortableNumericContentIdentity
    dt_float64_bits: str = dataclasses.field(init=False)
    static_layout_sha256: str = dataclasses.field(init=False)
    sample_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.sample_id, "sample_id")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("sample ordinal must be a non-negative integer")
        if self.operator_geometry_policy != OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME:
            raise ValueError("sample must use the registered portable operator geometry policy")
        if self.operator_volume_policy != OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT:
            raise ValueError("sample must use the registered portable operator volume policy")
        for name in (
            "topology_sha256",
            "operator_geometry_sha256",
            "operator_volume_sha256",
            "material_sha256",
            "pin_signature_sha256",
            "physical_step_sha256",
            "common_objective_sha256",
        ):
            _sha256(getattr(self, name), f"sample {name}")
        _validate_physical_integration_identity(
            self.physical_integration_policy,
            self.source_integration_evidence_sha256,
            "sample",
        )
        if self.physical_integration_policy == _PHYSICAL_INTEGRATION_POLICIES[0]:
            if self.source_transition is not None:
                raise ValueError("algebraic sample must not carry a reference source transition")
        else:
            if type(self.source_transition) is not PortableReferenceSourceTransitionIdentity:
                raise ValueError("SolverVBD sample must carry a canonical reference source transition")
            if self.source_transition.source_transition_sha256 != _canonical_digest(self.source_transition._payload()):
                raise ValueError("sample reference source transition changed after authentication")
            if self.source_transition.step_id != self.ordinal:
                raise ValueError("sample ordinal differs from its reference source transition")
            source_static = self.source_transition.portable_static_identity
            sample_static = (
                self.topology_sha256,
                self.operator_geometry_policy,
                self.operator_geometry_sha256,
                self.operator_volume_policy,
                self.operator_volume_sha256,
                self.material_sha256,
                self.pin_signature_sha256,
            )
            if source_static != sample_static:
                raise ValueError("sample portable static identity differs from its reference source transition")
        dt_seconds = _positive_float64(self.dt_seconds, "sample dt_seconds")
        object.__setattr__(self, "dt_seconds", dt_seconds)
        object.__setattr__(self, "dt_float64_bits", _float64_bits(dt_seconds))
        object.__setattr__(
            self,
            "static_layout_sha256",
            _canonical_digest(
                {
                    "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
                    "contract": _STATIC_LAYOUT_CONTRACT,
                    "topology_sha256": self.topology_sha256,
                    "operator_geometry_policy": self.operator_geometry_policy,
                    "operator_geometry_sha256": self.operator_geometry_sha256,
                    "operator_volume_policy": self.operator_volume_policy,
                    "operator_volume_sha256": self.operator_volume_sha256,
                    "material_sha256": self.material_sha256,
                    "pin_signature_sha256": self.pin_signature_sha256,
                    "dt_float64_bits": self.dt_float64_bits,
                }
            ),
        )
        for name, identity in self.numeric_content:
            if type(identity) is not PortableNumericContentIdentity:
                raise ValueError(f"{name} must be a canonical PortableNumericContentIdentity")
            if identity.identity_sha256 != _canonical_digest(identity._payload()):
                raise ValueError(f"{name} identity changed after authentication")
            if self.source_transition is not None and identity.identifier != self.source_transition.numeric_identifier(
                name
            ):
                raise ValueError(f"{name} identifier differs from the reference source transition")
        object.__setattr__(self, "sample_sha256", _canonical_digest(self._payload()))

    @property
    def numeric_content(self) -> tuple[tuple[str, PortableNumericContentIdentity], ...]:
        """Return tensor identities in canonical component order."""
        return tuple((name, getattr(self, name)) for name in _NUMERIC_COMPONENTS)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _SAMPLE_CONTRACT,
            "sample_id": self.sample_id,
            "ordinal": self.ordinal,
            "topology_sha256": self.topology_sha256,
            "operator_geometry": {
                "policy": self.operator_geometry_policy,
                "sha256": self.operator_geometry_sha256,
            },
            "operator_volume": {
                "policy": self.operator_volume_policy,
                "sha256": self.operator_volume_sha256,
            },
            "material_sha256": self.material_sha256,
            "pin_signature_sha256": self.pin_signature_sha256,
            "dt_seconds": self.dt_seconds,
            "dt_float64_bits": self.dt_float64_bits,
            "static_layout_sha256": self.static_layout_sha256,
            "physical_step": {
                "sha256": self.physical_step_sha256,
                "integration_policy": self.physical_integration_policy,
                "source_integration_evidence_sha256": self.source_integration_evidence_sha256,
                "source_transition_sha256": (
                    None if self.source_transition is None else self.source_transition.source_transition_sha256
                ),
            },
            "source_transition": None if self.source_transition is None else self.source_transition.as_dict(),
            "common_objective": {
                "contract": _COMMON_OBJECTIVE_BINDING_CONTRACT,
                "sha256": self.common_objective_sha256,
                "operator_geometry_sha256": self.operator_geometry_sha256,
                "operator_volume_policy": self.operator_volume_policy,
                "operator_volume_sha256": self.operator_volume_sha256,
            },
            "numeric_content": {name: identity.as_dict() for name, identity in self.numeric_content},
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["sample_sha256"] = self.sample_sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> PortableDatasetSampleRecord:
        """Strictly reconstruct one portable sample."""
        keys = {
            "schema_version",
            "contract",
            "sample_id",
            "ordinal",
            "topology_sha256",
            "operator_geometry",
            "operator_volume",
            "material_sha256",
            "pin_signature_sha256",
            "dt_seconds",
            "dt_float64_bits",
            "static_layout_sha256",
            "physical_step",
            "source_transition",
            "common_objective",
            "numeric_content",
            "sample_sha256",
        }
        payload = _strict_mapping(value, keys, "portable sample")
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _SAMPLE_CONTRACT,
        ):
            raise ValueError("portable sample has an unregistered schema identity")
        geometry = _strict_mapping(payload["operator_geometry"], {"policy", "sha256"}, "operator_geometry")
        volume = _strict_mapping(payload["operator_volume"], {"policy", "sha256"}, "operator_volume")
        physical = _strict_mapping(
            payload["physical_step"],
            {
                "sha256",
                "integration_policy",
                "source_integration_evidence_sha256",
                "source_transition_sha256",
            },
            "physical_step",
        )
        source_transition = (
            None
            if payload["source_transition"] is None
            else PortableReferenceSourceTransitionIdentity.from_dict(payload["source_transition"])
        )
        expected_source_transition_sha256 = (
            None if source_transition is None else source_transition.source_transition_sha256
        )
        if physical["source_transition_sha256"] != expected_source_transition_sha256:
            raise ValueError("portable physical step differs from its reference source transition")
        objective = _strict_mapping(
            payload["common_objective"],
            {
                "contract",
                "sha256",
                "operator_geometry_sha256",
                "operator_volume_policy",
                "operator_volume_sha256",
            },
            "common_objective",
        )
        if objective["contract"] != _COMMON_OBJECTIVE_BINDING_CONTRACT:
            raise ValueError("portable sample common-objective binding contract is not registered")
        if (
            objective["operator_geometry_sha256"] != geometry["sha256"]
            or objective["operator_volume_policy"] != volume["policy"]
            or objective["operator_volume_sha256"] != volume["sha256"]
        ):
            raise ValueError("portable sample common-objective operator binding is inconsistent")
        numeric = _strict_mapping(payload["numeric_content"], set(_NUMERIC_COMPONENTS), "numeric_content")
        result = cls(
            sample_id=payload["sample_id"],
            ordinal=payload["ordinal"],
            topology_sha256=payload["topology_sha256"],
            operator_geometry_policy=geometry["policy"],
            operator_geometry_sha256=geometry["sha256"],
            operator_volume_policy=volume["policy"],
            operator_volume_sha256=volume["sha256"],
            material_sha256=payload["material_sha256"],
            pin_signature_sha256=payload["pin_signature_sha256"],
            dt_seconds=payload["dt_seconds"],
            physical_step_sha256=physical["sha256"],
            physical_integration_policy=physical["integration_policy"],
            source_integration_evidence_sha256=physical["source_integration_evidence_sha256"],
            source_transition=source_transition,
            common_objective_sha256=objective["sha256"],
            **{name: PortableNumericContentIdentity.from_dict(numeric[name]) for name in _NUMERIC_COMPONENTS},
        )
        _require_exact_round_trip(result.as_dict(), payload, "portable sample")
        return result


@dataclasses.dataclass(frozen=True)
class PortableDatasetTrajectoryRecord:
    """One complete sequence under the portable-volume identity domain."""

    trajectory_id: str
    scene_family: str
    load_program_id: str
    load_program_sha256: str
    source_chain_sha256: str
    topology_sha256: str
    operator_geometry_policy: str
    operator_geometry_sha256: str
    operator_volume_policy: str
    operator_volume_sha256: str
    material_sha256: str
    provenance: PortableReferenceSequenceProvenance
    source_transition_count: int
    samples: tuple[PortableDatasetSampleRecord, ...]
    trajectory_id_sha256: str = dataclasses.field(init=False)
    trajectory_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "scene_family", "load_program_id"):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "trajectory_id_sha256", hashlib.sha256(self.trajectory_id.encode("utf-8")).hexdigest())
        if self.operator_geometry_policy != OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME:
            raise ValueError("trajectory must use the registered portable operator geometry policy")
        if self.operator_volume_policy != OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT:
            raise ValueError("trajectory must use the registered portable operator volume policy")
        for name in (
            "load_program_sha256",
            "source_chain_sha256",
            "topology_sha256",
            "operator_geometry_sha256",
            "operator_volume_sha256",
            "material_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.provenance) is not PortableReferenceSequenceProvenance:
            raise ValueError("provenance must be canonical portable reference-sequence provenance")
        if self.provenance.provenance_sha256 != _canonical_digest(self.provenance._payload()):
            raise ValueError("trajectory provenance changed after authentication")
        if type(self.source_transition_count) is not int or self.source_transition_count < 1:
            raise ValueError("source_transition_count must be a positive integer")
        samples = tuple(self.samples)
        if any(type(sample) is not PortableDatasetSampleRecord for sample in samples):
            raise ValueError("trajectory samples must be canonical PortableDatasetSampleRecord values")
        samples = tuple(sorted(samples, key=lambda sample: (sample.ordinal, sample.sample_id)))
        if len(samples) != self.source_transition_count:
            raise ValueError("portable trajectory must contain every source transition")
        if tuple(sample.ordinal for sample in samples) != tuple(range(self.source_transition_count)):
            raise ValueError("portable trajectory sample ordinals must be the complete contiguous range")
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("sample_id values must be unique within a trajectory")
        if self.provenance.source_transition_count != self.source_transition_count:
            raise ValueError("trajectory transition count differs from sequence provenance")
        for sample in samples:
            source = sample.source_transition
            if type(source) is not PortableReferenceSourceTransitionIdentity:
                raise ValueError("portable reference trajectory sample lacks a canonical source transition")
            expected_source = (
                self.provenance.dataset_index_sha256,
                self.provenance.asset_id,
                self.provenance.asset_source_sha256,
                self.provenance.sequence_id,
                sample.ordinal,
                self.provenance.static_bundle_sha256,
                self.provenance.sequence_bundle_sha256,
                self.provenance.protocol_sha256,
                self.provenance.producer_topology_sha256,
                self.provenance.producer_operator_sha256,
                self.provenance.producer_material_sha256,
                self.provenance.accepted_reference_state_sha256[sample.ordinal],
            )
            observed_source = (
                source.reference_sequence_index_sha256,
                source.asset_id,
                source.asset_source_sha256,
                source.sequence_id,
                source.step_id,
                source.static_npz_sha256,
                source.sequence_npz_sha256,
                source.protocol_sha256,
                source.producer_topology_sha256,
                source.producer_operator_sha256,
                source.producer_material_sha256,
                source.accepted_reference_state_sha256,
            )
            if observed_source != expected_source:
                raise ValueError("sample source transition differs from reference-sequence provenance")
            expected_static = (
                self.topology_sha256,
                self.operator_geometry_policy,
                self.operator_geometry_sha256,
                self.operator_volume_policy,
                self.operator_volume_sha256,
                self.material_sha256,
            )
            observed_static = (
                sample.topology_sha256,
                sample.operator_geometry_policy,
                sample.operator_geometry_sha256,
                sample.operator_volume_policy,
                sample.operator_volume_sha256,
                sample.material_sha256,
            )
            if observed_static != expected_static:
                raise ValueError("sample portable static identity disagrees with its trajectory")
            if sample.dt_float64_bits != self.provenance.dt_float64_bits:
                raise ValueError("sample time step disagrees with trajectory provenance")
            if sample.sample_sha256 != _canonical_digest(sample._payload()):
                raise ValueError("trajectory sample changed after authentication")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "trajectory_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _TRAJECTORY_CONTRACT,
            "trajectory_id": self.trajectory_id,
            "trajectory_id_sha256": self.trajectory_id_sha256,
            "scene_family": self.scene_family,
            "load_program_id": self.load_program_id,
            "load_program_sha256": self.load_program_sha256,
            "source_chain_sha256": self.source_chain_sha256,
            "topology_sha256": self.topology_sha256,
            "operator_geometry": {
                "policy": self.operator_geometry_policy,
                "sha256": self.operator_geometry_sha256,
            },
            "operator_volume": {
                "policy": self.operator_volume_policy,
                "sha256": self.operator_volume_sha256,
            },
            "material_sha256": self.material_sha256,
            "provenance": self.provenance.as_dict(),
            "source_transition_count": self.source_transition_count,
            "samples": [sample.as_dict() for sample in self.samples],
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["trajectory_sha256"] = self.trajectory_sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> PortableDatasetTrajectoryRecord:
        """Strictly reconstruct one complete portable trajectory."""
        keys = {
            "schema_version",
            "contract",
            "trajectory_id",
            "trajectory_id_sha256",
            "scene_family",
            "load_program_id",
            "load_program_sha256",
            "source_chain_sha256",
            "topology_sha256",
            "operator_geometry",
            "operator_volume",
            "material_sha256",
            "provenance",
            "source_transition_count",
            "samples",
            "trajectory_sha256",
        }
        payload = _strict_mapping(value, keys, "portable trajectory")
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _TRAJECTORY_CONTRACT,
        ):
            raise ValueError("portable trajectory has an unregistered schema identity")
        geometry = _strict_mapping(payload["operator_geometry"], {"policy", "sha256"}, "operator_geometry")
        volume = _strict_mapping(payload["operator_volume"], {"policy", "sha256"}, "operator_volume")
        samples = payload["samples"]
        if type(samples) is not list:
            raise ValueError("portable trajectory samples must be a JSON list")
        result = cls(
            trajectory_id=payload["trajectory_id"],
            scene_family=payload["scene_family"],
            load_program_id=payload["load_program_id"],
            load_program_sha256=payload["load_program_sha256"],
            source_chain_sha256=payload["source_chain_sha256"],
            topology_sha256=payload["topology_sha256"],
            operator_geometry_policy=geometry["policy"],
            operator_geometry_sha256=geometry["sha256"],
            operator_volume_policy=volume["policy"],
            operator_volume_sha256=volume["sha256"],
            material_sha256=payload["material_sha256"],
            provenance=PortableReferenceSequenceProvenance.from_dict(payload["provenance"]),
            source_transition_count=payload["source_transition_count"],
            samples=tuple(PortableDatasetSampleRecord.from_dict(sample) for sample in samples),
        )
        _require_exact_round_trip(result.as_dict(), payload, "portable trajectory")
        return result


def _cross_role_values(
    records: Mapping[DatasetRole, tuple[PortableDatasetTrajectoryRecord, ...]],
    values,
) -> dict[str, tuple[DatasetRole, ...]]:
    roles_by_value: dict[str, set[DatasetRole]] = {}
    for role, role_records in records.items():
        for record in role_records:
            for value in values(record):
                roles_by_value.setdefault(value, set()).add(role)
    return {
        value: tuple(role for role in _ROLE_ORDER if role in roles)
        for value, roles in roles_by_value.items()
        if len(roles) > 1
    }


def _format_role_collision(collisions: dict[str, tuple[DatasetRole, ...]]) -> str:
    value, roles = next(iter(collisions.items()))
    return f"{value} ({', '.join(role.value for role in roles)})"


@dataclasses.dataclass(frozen=True)
class PortableDatasetSplitManifest:
    """Whole-trajectory portable split with fail-closed leakage checks."""

    train: tuple[PortableDatasetTrajectoryRecord, ...]
    validation: tuple[PortableDatasetTrajectoryRecord, ...]
    confirmation: tuple[PortableDatasetTrajectoryRecord, ...]
    consumed_regression: tuple[PortableDatasetTrajectoryRecord, ...] = ()
    manifest_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        canonical: dict[DatasetRole, tuple[PortableDatasetTrajectoryRecord, ...]] = {}
        for role in _ROLE_ORDER:
            records = tuple(getattr(self, role.value))
            if any(type(record) is not PortableDatasetTrajectoryRecord for record in records):
                raise ValueError(f"{role.value} must contain canonical portable trajectories")
            records = tuple(sorted(records, key=lambda record: record.trajectory_id))
            if len({record.trajectory_id for record in records}) != len(records):
                raise ValueError(f"trajectory_id values must be unique within {role.value}")
            if any(record.trajectory_sha256 != _canonical_digest(record._payload()) for record in records):
                raise ValueError("split manifest contains a changed portable trajectory")
            canonical[role] = records
            object.__setattr__(self, role.value, records)

        source_by_asset_id: dict[str, str] = {}
        for records in canonical.values():
            for record in records:
                asset_id = record.provenance.asset_id
                asset_source_sha256 = record.provenance.asset_source_sha256
                previous = source_by_asset_id.setdefault(asset_id, asset_source_sha256)
                if previous != asset_source_sha256:
                    raise ValueError(f"asset_id {asset_id!r} maps to conflicting asset source SHA-256 values")

        checks = (
            ("trajectory overlap", lambda record: (record.trajectory_id,)),
            ("reference asset_id overlap", lambda record: (record.provenance.asset_id,)),
            (
                "reference asset source overlap",
                lambda record: (record.provenance.asset_source_sha256,),
            ),
            (
                "trajectory source overlap",
                lambda record: (record.source_chain_sha256, record.provenance.provenance_sha256),
            ),
            ("load-program overlap", lambda record: (record.load_program_id, record.load_program_sha256)),
            (
                "numeric content identifier overlap",
                lambda record: tuple(
                    identity.identifier for sample in record.samples for _name, identity in sample.numeric_content
                ),
            ),
            (
                "sample payload SHA-256 overlap",
                lambda record: tuple(
                    itertools.chain.from_iterable(
                        (
                            *(identity.sha256 for _name, identity in sample.numeric_content),
                            sample.physical_step_sha256,
                            sample.common_objective_sha256,
                        )
                        for sample in record.samples
                    )
                ),
            ),
        )
        for label, getter in checks:
            collisions = _cross_role_values(canonical, getter)
            if collisions:
                raise ValueError(f"{label} across roles: {_format_role_collision(collisions)}")
        object.__setattr__(self, "manifest_sha256", _canonical_digest(self._payload()))

    def records(self, role: DatasetRole | str) -> tuple[PortableDatasetTrajectoryRecord, ...]:
        """Return records assigned to one role."""
        return getattr(self, _dataset_role(role).value)

    def role_for_trajectory(self, trajectory_id: str) -> DatasetRole:
        """Resolve a trajectory's unique role."""
        _identifier(trajectory_id, "trajectory_id")
        for role in _ROLE_ORDER:
            if any(record.trajectory_id == trajectory_id for record in self.records(role)):
                return role
        raise ValueError(f"trajectory {trajectory_id!r} does not belong to this split")

    def trajectory(self, trajectory_id: str) -> PortableDatasetTrajectoryRecord:
        """Return one trajectory by identifier."""
        role = self.role_for_trajectory(trajectory_id)
        return next(record for record in self.records(role) if record.trajectory_id == trajectory_id)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _SPLIT_CONTRACT,
            "roles": {role.value: [record.as_dict() for record in self.records(role)] for role in _ROLE_ORDER},
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload

    def to_json(self) -> str:
        """Return canonical JSON text."""
        return _canonical_json_text(self.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> PortableDatasetSplitManifest:
        """Strictly reconstruct one portable split manifest."""
        payload = _strict_mapping(
            value,
            {"schema_version", "contract", "roles", "manifest_sha256"},
            "portable split manifest",
        )
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _SPLIT_CONTRACT,
        ):
            raise ValueError("portable split manifest has an unregistered schema identity")
        roles = _strict_mapping(payload["roles"], {role.value for role in _ROLE_ORDER}, "portable split roles")
        parsed: dict[str, tuple[PortableDatasetTrajectoryRecord, ...]] = {}
        for role in _ROLE_ORDER:
            records = roles[role.value]
            if type(records) is not list:
                raise ValueError(f"portable split role {role.value} must be a JSON list")
            parsed[role.value] = tuple(PortableDatasetTrajectoryRecord.from_dict(record) for record in records)
        result = cls(**parsed)
        _require_exact_round_trip(result.as_dict(), payload, "portable split manifest")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> PortableDatasetSplitManifest:
        """Strictly deserialize one portable split manifest."""
        return cls.from_dict(_read_json_object(value, "portable split manifest"))


def _verify_manifest(manifest: PortableDatasetSplitManifest) -> None:
    if type(manifest) is not PortableDatasetSplitManifest:
        raise ValueError("manifest must be a canonical PortableDatasetSplitManifest")
    if manifest.manifest_sha256 != _canonical_digest(manifest._payload()):
        raise ValueError("portable split manifest changed after authentication")


def _payload_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("payload_names must be a sequence of component names")
    names = tuple(sorted(values))
    if len(set(names)) != len(names):
        raise ValueError("payload_names must be unique")
    unknown = tuple(name for name in names if name not in _PAYLOAD_COMPONENTS)
    if unknown:
        raise ValueError(f"payload_names contain unknown payload components: {unknown}")
    return names


def _payload_identity_digest(
    trajectory: PortableDatasetTrajectoryRecord,
    payload_names: Sequence[str],
) -> str:
    names = _payload_names(payload_names)

    def component(sample: PortableDatasetSampleRecord, name: str) -> dict[str, object]:
        if name in _NUMERIC_COMPONENTS:
            return getattr(sample, name).as_dict()
        if name == "physical_step":
            return {
                "sha256": sample.physical_step_sha256,
                "integration_policy": sample.physical_integration_policy,
                "source_integration_evidence_sha256": sample.source_integration_evidence_sha256,
                "source_transition_sha256": (
                    None if sample.source_transition is None else sample.source_transition.source_transition_sha256
                ),
            }
        return {
            "sha256": sample.common_objective_sha256,
            "operator_geometry_sha256": sample.operator_geometry_sha256,
            "operator_volume_policy": sample.operator_volume_policy,
            "operator_volume_sha256": sample.operator_volume_sha256,
        }

    return _canonical_digest(
        {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _PAYLOAD_SELECTION_CONTRACT,
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_sha256": trajectory.trajectory_sha256,
            "payload_names": list(names),
            "bindings": [
                {
                    "sample_id": sample.sample_id,
                    "sample_sha256": sample.sample_sha256,
                    "component": name,
                    "identity": component(sample, name),
                }
                for sample in trajectory.samples
                for name in names
            ],
        }
    )


@dataclasses.dataclass(frozen=True)
class PortableDatasetAccessRecord:
    """One immutable portable dataset access event."""

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
        names = _payload_names(self.payload_names)
        if self.scope is DataAccessScope.METADATA:
            if names or self.payload_identity_sha256 is not None:
                raise ValueError("metadata access must not bind sample payloads")
        else:
            if not names:
                raise ValueError("payload access must name every opened payload")
            _sha256(self.payload_identity_sha256, "payload_identity_sha256")
        _sha256(self.previous_sha256, "previous access sha256")
        object.__setattr__(self, "payload_names", names)
        object.__setattr__(self, "access_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _ACCESS_RECORD_CONTRACT,
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
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["access_sha256"] = self.access_sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> PortableDatasetAccessRecord:
        """Strictly reconstruct one access event."""
        keys = {
            "schema_version",
            "contract",
            "sequence",
            "trajectory_id",
            "role",
            "purpose",
            "scope",
            "payload_names",
            "payload_identity_sha256",
            "previous_sha256",
            "access_sha256",
        }
        payload = _strict_mapping(value, keys, "portable access record")
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _ACCESS_RECORD_CONTRACT,
        ):
            raise ValueError("portable access record has an unregistered schema identity")
        names = payload["payload_names"]
        if type(names) is not list:
            raise ValueError("portable access payload_names must be a JSON list")
        result = cls(
            sequence=payload["sequence"],
            trajectory_id=payload["trajectory_id"],
            role=payload["role"],
            purpose=payload["purpose"],
            scope=payload["scope"],
            payload_names=tuple(names),
            payload_identity_sha256=payload["payload_identity_sha256"],
            previous_sha256=payload["previous_sha256"],
        )
        _require_exact_round_trip(result.as_dict(), payload, "portable access record")
        return result


@dataclasses.dataclass(frozen=True)
class PortableDatasetAccessLedger:
    """Functional branch-local evidence for portable dataset access."""

    manifest: PortableDatasetSplitManifest
    accesses: tuple[PortableDatasetAccessRecord, ...] = ()
    confirmation_payload_released: bool = dataclasses.field(init=False)
    ledger_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _verify_manifest(self.manifest)
        accesses = tuple(self.accesses)
        previous = self.manifest.manifest_sha256
        confirmation_released = False
        for sequence, access in enumerate(accesses):
            if type(access) is not PortableDatasetAccessRecord:
                raise ValueError("accesses must contain canonical portable access records")
            if access.sequence != sequence or access.previous_sha256 != previous:
                raise ValueError("portable access hash chain is disconnected")
            if access.access_sha256 != _canonical_digest(access._payload()):
                raise ValueError("portable access record changed after authentication")
            trajectory = self.manifest.trajectory(access.trajectory_id)
            if access.role is not self.manifest.role_for_trajectory(access.trajectory_id):
                raise ValueError("access role disagrees with the frozen portable split")
            if access.scope is DataAccessScope.PAYLOAD:
                if access.payload_identity_sha256 != _payload_identity_digest(trajectory, access.payload_names):
                    raise ValueError("access payload identity disagrees with the portable trajectory")
                if confirmation_released and access.purpose in (
                    DataAccessPurpose.TRAINING,
                    DataAccessPurpose.MODEL_SELECTION,
                ):
                    raise ValueError(
                        "training or model-selection payload access cannot resume after confirmation release "
                        "on this ledger branch"
                    )
            self._validate_payload_policy(access)
            if access.scope is DataAccessScope.PAYLOAD and access.role is DatasetRole.CONFIRMATION:
                confirmation_released = True
            previous = access.access_sha256
        object.__setattr__(self, "accesses", accesses)
        object.__setattr__(self, "confirmation_payload_released", confirmation_released)
        object.__setattr__(self, "ledger_sha256", _canonical_digest(self._payload()))

    @staticmethod
    def _validate_payload_policy(access: PortableDatasetAccessRecord) -> None:
        if access.scope is DataAccessScope.METADATA:
            return
        if access.role not in _PAYLOAD_ROLES_BY_PURPOSE[access.purpose]:
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
    ) -> PortableDatasetAccessLedger:
        """Return a new ledger with one policy-checked event appended."""
        role = self.manifest.role_for_trajectory(trajectory_id)
        trajectory = self.manifest.trajectory(trajectory_id)
        canonical_scope = _access_scope(scope)
        names = _payload_names(payload_names)
        payload_sha256 = (
            _payload_identity_digest(trajectory, names) if canonical_scope is DataAccessScope.PAYLOAD else None
        )
        previous = self.accesses[-1].access_sha256 if self.accesses else self.manifest.manifest_sha256
        access = PortableDatasetAccessRecord(
            sequence=len(self.accesses),
            trajectory_id=trajectory_id,
            role=role,
            purpose=_access_purpose(purpose),
            scope=canonical_scope,
            payload_names=names,
            payload_identity_sha256=payload_sha256,
            previous_sha256=previous,
        )
        return PortableDatasetAccessLedger(self.manifest, (*self.accesses, access))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _ACCESS_LEDGER_CONTRACT,
            "claim_scope": _LEDGER_CLAIM_SCOPE,
            "manifest_sha256": self.manifest.manifest_sha256,
            "accesses": [access.as_dict() for access in self.accesses],
            "confirmation_payload_released": self.confirmation_payload_released,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["ledger_sha256"] = self.ledger_sha256
        return payload

    def to_json(self) -> str:
        """Return canonical ledger JSON text."""
        return _canonical_json_text(self.as_dict())

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        manifest: PortableDatasetSplitManifest,
    ) -> PortableDatasetAccessLedger:
        """Strictly reconstruct a ledger against its external manifest."""
        keys = {
            "schema_version",
            "contract",
            "claim_scope",
            "manifest_sha256",
            "accesses",
            "confirmation_payload_released",
            "ledger_sha256",
        }
        payload = _strict_mapping(value, keys, "portable access ledger")
        if (
            payload["schema_version"],
            payload["contract"],
            payload["claim_scope"],
            payload["manifest_sha256"],
        ) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _ACCESS_LEDGER_CONTRACT,
            _LEDGER_CLAIM_SCOPE,
            manifest.manifest_sha256,
        ):
            raise ValueError("portable access ledger identity differs from its manifest or registered contract")
        accesses = payload["accesses"]
        if type(accesses) is not list:
            raise ValueError("portable access ledger accesses must be a JSON list")
        result = cls(manifest, tuple(PortableDatasetAccessRecord.from_dict(access) for access in accesses))
        _require_exact_round_trip(result.as_dict(), payload, "portable access ledger")
        return result

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        *,
        manifest: PortableDatasetSplitManifest,
    ) -> PortableDatasetAccessLedger:
        """Strictly deserialize a ledger against its external manifest."""
        return cls.from_dict(_read_json_object(value, "portable access ledger"), manifest=manifest)


@dataclasses.dataclass(frozen=True)
class PortableDatasetSamplingReference:
    """One scheduled portable sample identity."""

    trajectory_id: str
    trajectory_sha256: str
    topology_sha256: str
    operator_geometry_policy: str
    operator_geometry_sha256: str
    operator_volume_policy: str
    operator_volume_sha256: str
    material_sha256: str
    pin_signature_sha256: str
    dt_float64_bits: str
    static_layout_sha256: str
    sample_id: str
    sample_sha256: str
    physical_step_sha256: str
    physical_integration_policy: str
    source_integration_evidence_sha256: str | None
    common_objective_sha256: str
    ordinal: int

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "sample_id"):
            _identifier(getattr(self, name), name)
        if self.operator_geometry_policy != OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME:
            raise ValueError("scheduled sample must use the portable operator geometry policy")
        if self.operator_volume_policy != OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT:
            raise ValueError("scheduled sample must use the portable operator volume policy")
        for name in (
            "trajectory_sha256",
            "topology_sha256",
            "operator_geometry_sha256",
            "operator_volume_sha256",
            "material_sha256",
            "pin_signature_sha256",
            "static_layout_sha256",
            "sample_sha256",
            "physical_step_sha256",
            "common_objective_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.dt_float64_bits) is not str or not self.dt_float64_bits.startswith("0x"):
            raise ValueError("dt_float64_bits must be canonical hexadecimal text")
        _validate_physical_integration_identity(
            self.physical_integration_policy,
            self.source_integration_evidence_sha256,
            "scheduled sample",
        )
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("scheduled sample ordinal must be non-negative")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON object."""
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _SAMPLING_REFERENCE_CONTRACT,
            "trajectory_id": self.trajectory_id,
            "trajectory_sha256": self.trajectory_sha256,
            "topology_sha256": self.topology_sha256,
            "operator_geometry_policy": self.operator_geometry_policy,
            "operator_geometry_sha256": self.operator_geometry_sha256,
            "operator_volume_policy": self.operator_volume_policy,
            "operator_volume_sha256": self.operator_volume_sha256,
            "material_sha256": self.material_sha256,
            "pin_signature_sha256": self.pin_signature_sha256,
            "dt_float64_bits": self.dt_float64_bits,
            "static_layout_sha256": self.static_layout_sha256,
            "sample_id": self.sample_id,
            "sample_sha256": self.sample_sha256,
            "physical_step_sha256": self.physical_step_sha256,
            "physical_integration_policy": self.physical_integration_policy,
            "source_integration_evidence_sha256": self.source_integration_evidence_sha256,
            "common_objective_sha256": self.common_objective_sha256,
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, value: object) -> PortableDatasetSamplingReference:
        """Strictly reconstruct a scheduled reference."""
        keys = set(cls.__dataclass_fields__) | {"schema_version", "contract"}
        payload = _strict_mapping(value, keys, "portable sampling reference")
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _SAMPLING_REFERENCE_CONTRACT,
        ):
            raise ValueError("portable sampling reference has an unregistered schema identity")
        result = cls(**{name: payload[name] for name in cls.__dataclass_fields__})
        _require_exact_round_trip(result.as_dict(), payload, "portable sampling reference")
        return result


@dataclasses.dataclass(frozen=True)
class PortableDatasetSamplingBatch:
    """One portable-static-layout-homogeneous batch."""

    topology_sha256: str
    operator_geometry_policy: str
    operator_geometry_sha256: str
    operator_volume_policy: str
    operator_volume_sha256: str
    material_sha256: str
    pin_signature_sha256: str
    dt_float64_bits: str
    static_layout_sha256: str
    samples: tuple[PortableDatasetSamplingReference, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples or any(type(sample) is not PortableDatasetSamplingReference for sample in samples):
            raise ValueError("portable sampling batch must contain canonical references")
        fields = (
            "topology_sha256",
            "operator_geometry_policy",
            "operator_geometry_sha256",
            "operator_volume_policy",
            "operator_volume_sha256",
            "material_sha256",
            "pin_signature_sha256",
            "dt_float64_bits",
            "static_layout_sha256",
        )
        if any(any(getattr(sample, name) != getattr(self, name) for name in fields) for sample in samples):
            raise ValueError("portable sampling batch mixes static layouts")
        object.__setattr__(self, "samples", samples)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON object."""
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _SAMPLING_BATCH_CONTRACT,
            "topology_sha256": self.topology_sha256,
            "operator_geometry_policy": self.operator_geometry_policy,
            "operator_geometry_sha256": self.operator_geometry_sha256,
            "operator_volume_policy": self.operator_volume_policy,
            "operator_volume_sha256": self.operator_volume_sha256,
            "material_sha256": self.material_sha256,
            "pin_signature_sha256": self.pin_signature_sha256,
            "dt_float64_bits": self.dt_float64_bits,
            "static_layout_sha256": self.static_layout_sha256,
            "physical_objective_routing": _OBJECTIVE_ROUTING,
            "samples": [sample.as_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, value: object) -> PortableDatasetSamplingBatch:
        """Strictly reconstruct one sampling batch."""
        fields = set(cls.__dataclass_fields__)
        payload = _strict_mapping(
            value,
            fields | {"schema_version", "contract", "physical_objective_routing"},
            "portable sampling batch",
        )
        if (
            payload["schema_version"],
            payload["contract"],
            payload["physical_objective_routing"],
        ) != (PORTABLE_DATASET_SCHEMA_VERSION, _SAMPLING_BATCH_CONTRACT, _OBJECTIVE_ROUTING):
            raise ValueError("portable sampling batch has an unregistered schema identity")
        samples = payload["samples"]
        if type(samples) is not list:
            raise ValueError("portable sampling batch samples must be a JSON list")
        kwargs = {name: payload[name] for name in fields if name != "samples"}
        result = cls(**kwargs, samples=tuple(PortableDatasetSamplingReference.from_dict(item) for item in samples))
        _require_exact_round_trip(result.as_dict(), payload, "portable sampling batch")
        return result


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


def _sampling_reference(
    trajectory: PortableDatasetTrajectoryRecord,
    sample: PortableDatasetSampleRecord,
) -> PortableDatasetSamplingReference:
    return PortableDatasetSamplingReference(
        trajectory_id=trajectory.trajectory_id,
        trajectory_sha256=trajectory.trajectory_sha256,
        topology_sha256=sample.topology_sha256,
        operator_geometry_policy=sample.operator_geometry_policy,
        operator_geometry_sha256=sample.operator_geometry_sha256,
        operator_volume_policy=sample.operator_volume_policy,
        operator_volume_sha256=sample.operator_volume_sha256,
        material_sha256=sample.material_sha256,
        pin_signature_sha256=sample.pin_signature_sha256,
        dt_float64_bits=sample.dt_float64_bits,
        static_layout_sha256=sample.static_layout_sha256,
        sample_id=sample.sample_id,
        sample_sha256=sample.sample_sha256,
        physical_step_sha256=sample.physical_step_sha256,
        physical_integration_policy=sample.physical_integration_policy,
        source_integration_evidence_sha256=sample.source_integration_evidence_sha256,
        common_objective_sha256=sample.common_objective_sha256,
        ordinal=sample.ordinal,
    )


def _build_sampling_batches(
    records: tuple[PortableDatasetTrajectoryRecord, ...],
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[PortableDatasetSamplingBatch, ...]:
    rng = np.random.Generator(np.random.PCG64(seed))
    trajectory_cycle = _ShuffledCycle(records, rng)
    layout_cycles: dict[str, _ShuffledCycle] = {}
    sample_cycles: dict[tuple[str, str], _ShuffledCycle] = {}
    samples_by_layout: dict[tuple[str, str], tuple[PortableDatasetSampleRecord, ...]] = {}
    for record in records:
        layouts = tuple(sorted({sample.static_layout_sha256 for sample in record.samples}))
        layout_cycles[record.trajectory_id] = _ShuffledCycle(layouts, rng)
        for layout in layouts:
            values = tuple(sample for sample in record.samples if sample.static_layout_sha256 == layout)
            samples_by_layout[(record.trajectory_id, layout)] = values
            sample_cycles[(record.trajectory_id, layout)] = _ShuffledCycle(values, rng)
    result: list[PortableDatasetSamplingBatch] = []
    for _ in range(steps):
        trajectory = trajectory_cycle.next()
        if type(trajectory) is not PortableDatasetTrajectoryRecord:
            raise RuntimeError("internal portable trajectory cycle is malformed")
        layout = layout_cycles[trajectory.trajectory_id].next()
        if type(layout) is not str:
            raise RuntimeError("internal portable layout cycle is malformed")
        key = (trajectory.trajectory_id, layout)
        representative = samples_by_layout[key][0]
        references = []
        for _sample_index in range(batch_size):
            sample = sample_cycles[key].next()
            if type(sample) is not PortableDatasetSampleRecord:
                raise RuntimeError("internal portable sample cycle is malformed")
            references.append(_sampling_reference(trajectory, sample))
        result.append(
            PortableDatasetSamplingBatch(
                topology_sha256=representative.topology_sha256,
                operator_geometry_policy=representative.operator_geometry_policy,
                operator_geometry_sha256=representative.operator_geometry_sha256,
                operator_volume_policy=representative.operator_volume_policy,
                operator_volume_sha256=representative.operator_volume_sha256,
                material_sha256=representative.material_sha256,
                pin_signature_sha256=representative.pin_signature_sha256,
                dt_float64_bits=representative.dt_float64_bits,
                static_layout_sha256=representative.static_layout_sha256,
                samples=tuple(references),
            )
        )
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class PortableDatasetSamplingSchedule:
    """Deterministic portable-volume sample stream."""

    manifest: PortableDatasetSplitManifest
    role: DatasetRole
    seed: int
    steps: int
    batch_size: int
    batches: tuple[PortableDatasetSamplingBatch, ...]
    manifest_sha256: str = dataclasses.field(init=False)
    trajectory_count: int = dataclasses.field(init=False)
    trajectory_epoch_count: int = dataclasses.field(init=False)
    schedule_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _verify_manifest(self.manifest)
        object.__setattr__(self, "manifest_sha256", self.manifest.manifest_sha256)
        role = _dataset_role(self.role)
        object.__setattr__(self, "role", role)
        records = self.manifest.records(role)
        if not records:
            raise ValueError(f"cannot authenticate a schedule for the empty {role.value} role")
        for name in ("seed", "steps", "batch_size"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.steps % len(records) != 0:
            raise ValueError("sampling steps must contain complete trajectory epochs")
        epoch_count = self.steps // len(records)
        for record in records:
            layout_count = len({sample.static_layout_sha256 for sample in record.samples})
            if epoch_count % layout_count != 0:
                raise ValueError("sampling steps must complete every portable static-layout cycle")
        batches = tuple(self.batches)
        if len(batches) != self.steps or any(type(batch) is not PortableDatasetSamplingBatch for batch in batches):
            raise ValueError("portable sampling batches do not match steps")
        if any(len(batch.samples) != self.batch_size for batch in batches):
            raise ValueError("every portable sampling batch must have batch_size samples")
        expected = _build_sampling_batches(records, steps=self.steps, batch_size=self.batch_size, seed=self.seed)
        if batches != expected:
            raise ValueError("portable sampling batches do not match deterministic PCG64 replay")
        object.__setattr__(self, "batches", batches)
        object.__setattr__(self, "trajectory_count", len(records))
        object.__setattr__(self, "trajectory_epoch_count", epoch_count)
        object.__setattr__(self, "schedule_sha256", _canonical_digest(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _SAMPLING_CONTRACT,
            "generator": _SAMPLING_GENERATOR,
            "selection_order": _SAMPLING_ORDER,
            "physical_objective_routing": _OBJECTIVE_ROUTING,
            "manifest_sha256": self.manifest_sha256,
            "role": self.role.value,
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "trajectory_count": self.trajectory_count,
            "trajectory_epoch_count": self.trajectory_epoch_count,
            "batches": [batch.as_dict() for batch in self.batches],
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["schedule_sha256"] = self.schedule_sha256
        return payload

    def to_json(self) -> str:
        """Return canonical schedule JSON text."""
        return _canonical_json_text(self.as_dict())

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        manifest: PortableDatasetSplitManifest,
    ) -> PortableDatasetSamplingSchedule:
        """Strictly reconstruct a schedule against its external manifest."""
        keys = {
            "schema_version",
            "contract",
            "generator",
            "selection_order",
            "physical_objective_routing",
            "manifest_sha256",
            "role",
            "seed",
            "steps",
            "batch_size",
            "trajectory_count",
            "trajectory_epoch_count",
            "batches",
            "schedule_sha256",
        }
        payload = _strict_mapping(value, keys, "portable sampling schedule")
        if (
            payload["schema_version"],
            payload["contract"],
            payload["generator"],
            payload["selection_order"],
            payload["physical_objective_routing"],
            payload["manifest_sha256"],
        ) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _SAMPLING_CONTRACT,
            _SAMPLING_GENERATOR,
            _SAMPLING_ORDER,
            _OBJECTIVE_ROUTING,
            manifest.manifest_sha256,
        ):
            raise ValueError("portable sampling schedule identity differs from its manifest or contract")
        batches = payload["batches"]
        if type(batches) is not list:
            raise ValueError("portable sampling batches must be a JSON list")
        result = cls(
            manifest=manifest,
            role=payload["role"],
            seed=payload["seed"],
            steps=payload["steps"],
            batch_size=payload["batch_size"],
            batches=tuple(PortableDatasetSamplingBatch.from_dict(batch) for batch in batches),
        )
        _require_exact_round_trip(result.as_dict(), payload, "portable sampling schedule")
        return result

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        *,
        manifest: PortableDatasetSplitManifest,
    ) -> PortableDatasetSamplingSchedule:
        """Strictly deserialize a schedule against its external manifest."""
        return cls.from_dict(_read_json_object(value, "portable sampling schedule"), manifest=manifest)


def build_portable_sampling_schedule(
    manifest: PortableDatasetSplitManifest,
    *,
    role: DatasetRole | str = DatasetRole.TRAIN,
    steps: int,
    batch_size: int,
    seed: int,
) -> PortableDatasetSamplingSchedule:
    """Build a deterministic trajectory/layout-balanced portable schedule."""
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
    return PortableDatasetSamplingSchedule(
        manifest=manifest,
        role=canonical_role,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        batches=_build_sampling_batches(records, steps=steps, batch_size=batch_size, seed=seed),
    )


__all__ = [
    "PORTABLE_DATASET_SCHEMA_VERSION",
    "PortableDatasetAccessLedger",
    "PortableDatasetAccessRecord",
    "PortableDatasetSampleRecord",
    "PortableDatasetSamplingBatch",
    "PortableDatasetSamplingReference",
    "PortableDatasetSamplingSchedule",
    "PortableDatasetSplitManifest",
    "PortableDatasetTrajectoryRecord",
    "PortableNumericContentIdentity",
    "PortableReferenceSequenceProvenance",
    "PortableReferenceSourceTransitionIdentity",
    "build_portable_sampling_schedule",
    "canonical_portable_training_tensor_sha256",
]
