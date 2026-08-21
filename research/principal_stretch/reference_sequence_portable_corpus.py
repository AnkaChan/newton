# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Portable-volume corpus metadata for authenticated reference sequences.

The producer entry point is explicitly preparation-only. Its receipt covers
complete output metadata under an exact source-index inventory; it is not
proof that payloads were opened. Training consumers derive a TRAIN/VALIDATION
view without opening sequence arrays and authenticate a serialized view
against a corpus digest obtained out of band. A view self-hash proves only
consistency, not origin. Runtime payloads remain lazy and are loaded one
transition at a time by the bridge.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .portable_dataset import (
    PORTABLE_DATASET_SCHEMA_VERSION,
    PortableDatasetSplitManifest,
    PortableDatasetTrajectoryRecord,
    PortableReferenceSequenceProvenance,
)
from .reference_sequence_dataset import (
    ReferenceSequenceDataset,
    ReferenceSequenceRecord,
    ReferenceTransitionKey,
    reference_sequence_index_header,
)
from .reference_sequence_v5_bridge import (
    ReferencePortableAssetIdentities,
    ReferenceSequencePortableDatasetBridge,
)
from .v5_checkpoint import canonical_json_sha256
from .v5_dataset import DatasetRole

_CORPUS_CONTRACT = "pss-reference-sequence-portable-volume-corpus-v2"
_PREPARATION_RECEIPT_CONTRACT = "pss-reference-sequence-portable-volume-producer-preparation-v2"
_PREPARATION_CLAIM_SCOPE = "complete-requested-role-metadata-not-proof-of-payload-opens-or-access-control"
_SOURCE_SEQUENCE_INVENTORY_CONTRACT = "pss-reference-sequence-portable-volume-source-sequence-inventory-v1"
_SOURCE_ROLE_INVENTORY_CONTRACT = "pss-reference-sequence-portable-volume-source-role-inventory-v1"
_CONSUMER_VIEW_CONTRACT = "pss-reference-sequence-portable-volume-consumer-view-v2"
_CONSUMER_VIEW_CLAIM_SCOPE = "self-consistency-only-external-corpus-root-required-for-origin"
_SOURCE_CHAIN_CONTRACT = "pss-reference-sequence-portable-volume-source-chain-v1"
_LOAD_PROGRAM_CONTRACT = "pss-reference-sequence-portable-volume-dynamics-program-v1"

_ROLE_ORDER = tuple(DatasetRole)
_CONSUMER_ROLES = frozenset((DatasetRole.TRAIN, DatasetRole.VALIDATION))


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


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


def _read_json(value: str | bytes, name: str) -> dict[str, object]:
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


def _canonical_roles(values: Sequence[DatasetRole | str], *, consumer: bool = False) -> tuple[DatasetRole, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("roles must be a sequence")
    try:
        roles = tuple(DatasetRole(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("roles contain an unregistered dataset role") from exc
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("roles must be non-empty and unique")
    canonical = tuple(role for role in _ROLE_ORDER if role in roles)
    if consumer and any(role not in _CONSUMER_ROLES for role in canonical):
        raise ValueError("consumer view may contain only train and validation roles")
    return canonical


def _trajectory_id(asset_id: str, sequence_id: str) -> str:
    return f"reference-sequence:{asset_id}:{sequence_id}"


def _load_program_payload(provenance: PortableReferenceSequenceProvenance) -> dict[str, object]:
    return {
        "contract": _LOAD_PROGRAM_CONTRACT,
        "asset_source_sha256": provenance.asset_source_sha256,
        "protocol_sha256": provenance.protocol_sha256,
        "initial_position_sha256": provenance.initial_position_sha256,
        "initial_velocity_field_sha256": provenance.initial_velocity_field_sha256,
        "deformation_seed": provenance.deformation_seed,
        "velocity_seed": provenance.velocity_seed,
        "source_transition_count": provenance.source_transition_count,
    }


def _source_chain_payload(provenance: PortableReferenceSequenceProvenance) -> dict[str, object]:
    return {
        "contract": _SOURCE_CHAIN_CONTRACT,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "producer_manifest_sha256": provenance.producer_manifest_sha256,
        "static_bundle_sha256": provenance.static_bundle_sha256,
        "sequence_bundle_sha256": provenance.sequence_bundle_sha256,
        "evidence_sha256": provenance.evidence_sha256,
    }


@dataclasses.dataclass(frozen=True, order=True)
class _SourceSequenceInventory:
    """Authenticated source-index identity for one sequence."""

    source_index_sha256: str
    role: DatasetRole
    asset_id: str
    asset_source_sha256: str
    sequence_id: str
    producer_topology_sha256: str
    producer_operator_sha256: str
    producer_material_sha256: str
    protocol_sha256: str
    producer_manifest_json: str
    producer_manifest_json_sha256: str
    accepted_reference_state_sha256: tuple[str, ...]
    source_transition_count: int
    inventory_sha256: str = dataclasses.field(init=False, compare=False)

    def __post_init__(self) -> None:
        _sha256(self.source_index_sha256, "source sequence index sha256")
        try:
            role = DatasetRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError("source sequence role is not registered") from exc
        object.__setattr__(self, "role", role)
        for name in ("asset_id", "sequence_id", "producer_manifest_json"):
            _identifier(getattr(self, name), f"source sequence {name}")
        for name in (
            "asset_source_sha256",
            "producer_topology_sha256",
            "producer_operator_sha256",
            "producer_material_sha256",
            "protocol_sha256",
            "producer_manifest_json_sha256",
        ):
            _sha256(getattr(self, name), f"source sequence {name}")
        if type(self.source_transition_count) is not int or self.source_transition_count < 1:
            raise ValueError("source sequence transition count must be positive")
        if isinstance(self.accepted_reference_state_sha256, (str, bytes)):
            raise ValueError("source sequence accepted-reference identities must be a sequence")
        accepted = tuple(self.accepted_reference_state_sha256)
        if len(accepted) != self.source_transition_count:
            raise ValueError("source sequence accepted-reference identities must cover every transition")
        for step_id, digest in enumerate(accepted):
            _sha256(digest, f"source sequence accepted reference {step_id} sha256")
        object.__setattr__(self, "accepted_reference_state_sha256", accepted)
        object.__setattr__(self, "inventory_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _SOURCE_SEQUENCE_INVENTORY_CONTRACT,
            "source_index_sha256": self.source_index_sha256,
            "role": self.role.value,
            "asset_id": self.asset_id,
            "asset_source_sha256": self.asset_source_sha256,
            "sequence_id": self.sequence_id,
            "producer_static": {
                "topology_sha256": self.producer_topology_sha256,
                "operator_sha256": self.producer_operator_sha256,
                "material_sha256": self.producer_material_sha256,
            },
            "protocol_sha256": self.protocol_sha256,
            "producer_manifest": {
                "json": self.producer_manifest_json,
                "json_sha256": self.producer_manifest_json_sha256,
            },
            "accepted_reference_state_sha256": list(self.accepted_reference_state_sha256),
            "source_transition_count": self.source_transition_count,
        }

    def as_dict(self) -> dict[str, object]:
        """Return one strict source-sequence inventory record."""
        payload = self._payload()
        payload["inventory_sha256"] = self.inventory_sha256
        return payload

    def index_record(self) -> dict[str, object]:
        """Return the exact source-index record preimage."""
        return {
            "role": self.role.value,
            "asset_id": self.asset_id,
            "asset_source_sha256": self.asset_source_sha256,
            "sequence_id": self.sequence_id,
            "topology_sha256": self.producer_topology_sha256,
            "operator_sha256": self.producer_operator_sha256,
            "material_sha256": self.producer_material_sha256,
            "protocol_sha256": self.protocol_sha256,
            "producer_manifest_json": self.producer_manifest_json,
            "producer_manifest_json_sha256": self.producer_manifest_json_sha256,
            "step_ids": list(range(self.source_transition_count)),
            "reference_state_float64_sha256": list(self.accepted_reference_state_sha256),
        }

    @property
    def materialized_identity(self) -> tuple[object, ...]:
        """Return fields independently preserved by portable trajectories."""
        return (
            self.role.value,
            self.asset_id,
            self.asset_source_sha256,
            self.sequence_id,
            self.producer_topology_sha256,
            self.producer_operator_sha256,
            self.producer_material_sha256,
            self.protocol_sha256,
            self.producer_manifest_json_sha256,
            self.accepted_reference_state_sha256,
            self.source_transition_count,
        )

    @classmethod
    def from_dict(cls, value: object) -> _SourceSequenceInventory:
        """Strictly reconstruct one source-sequence inventory record."""
        payload = _strict_mapping(
            value,
            {
                "schema_version",
                "contract",
                "source_index_sha256",
                "role",
                "asset_id",
                "asset_source_sha256",
                "sequence_id",
                "producer_static",
                "protocol_sha256",
                "producer_manifest",
                "accepted_reference_state_sha256",
                "source_transition_count",
                "inventory_sha256",
            },
            "source sequence inventory",
        )
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _SOURCE_SEQUENCE_INVENTORY_CONTRACT,
        ):
            raise ValueError("source sequence inventory has an unregistered schema identity")
        producer = _strict_mapping(
            payload["producer_static"],
            {"topology_sha256", "operator_sha256", "material_sha256"},
            "source sequence producer_static",
        )
        producer_manifest = _strict_mapping(
            payload["producer_manifest"],
            {"json", "json_sha256"},
            "source sequence producer manifest",
        )
        accepted = payload["accepted_reference_state_sha256"]
        if type(accepted) is not list:
            raise ValueError("source sequence accepted-reference identities must be a JSON list")
        result = cls(
            source_index_sha256=payload["source_index_sha256"],
            role=payload["role"],
            asset_id=payload["asset_id"],
            asset_source_sha256=payload["asset_source_sha256"],
            sequence_id=payload["sequence_id"],
            producer_topology_sha256=producer["topology_sha256"],
            producer_operator_sha256=producer["operator_sha256"],
            producer_material_sha256=producer["material_sha256"],
            protocol_sha256=payload["protocol_sha256"],
            producer_manifest_json=producer_manifest["json"],
            producer_manifest_json_sha256=producer_manifest["json_sha256"],
            accepted_reference_state_sha256=tuple(accepted),
            source_transition_count=payload["source_transition_count"],
        )
        if not _exact_json_equal(result.as_dict(), payload):
            raise ValueError("source sequence inventory is not the exact canonical record")
        return result


@dataclasses.dataclass(frozen=True)
class _SourceRoleInventory:
    """Complete authenticated source-index inventory for one role."""

    source_index_sha256: str
    role: DatasetRole
    sequences: tuple[_SourceSequenceInventory, ...]
    trajectory_count: int = dataclasses.field(init=False)
    source_transition_count: int = dataclasses.field(init=False)
    inventory_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_index_sha256, "source role index sha256")
        try:
            role = DatasetRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError("source inventory role is not registered") from exc
        object.__setattr__(self, "role", role)
        sequences = tuple(self.sequences)
        if any(type(sequence) is not _SourceSequenceInventory for sequence in sequences):
            raise ValueError("source role inventory contains a non-canonical sequence")
        sequences = tuple(sorted(sequences, key=lambda item: (item.asset_id, item.sequence_id)))
        if len({(item.asset_id, item.sequence_id) for item in sequences}) != len(sequences):
            raise ValueError("source role inventory sequence identities must be unique")
        for sequence in sequences:
            if sequence.source_index_sha256 != self.source_index_sha256 or sequence.role is not role:
                raise ValueError("source sequence inventory differs from its role inventory")
            if sequence.inventory_sha256 != canonical_json_sha256(sequence._payload()):
                raise ValueError("source sequence inventory changed after authentication")
        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "trajectory_count", len(sequences))
        object.__setattr__(
            self,
            "source_transition_count",
            sum(sequence.source_transition_count for sequence in sequences),
        )
        object.__setattr__(self, "inventory_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _SOURCE_ROLE_INVENTORY_CONTRACT,
            "source_index_sha256": self.source_index_sha256,
            "role": self.role.value,
            "trajectory_count": self.trajectory_count,
            "source_transition_count": self.source_transition_count,
            "sequences": [sequence.as_dict() for sequence in self.sequences],
        }

    def as_dict(self) -> dict[str, object]:
        """Return one strict source-role inventory record."""
        payload = self._payload()
        payload["inventory_sha256"] = self.inventory_sha256
        return payload

    @classmethod
    def from_dict(cls, value: object) -> _SourceRoleInventory:
        """Strictly reconstruct one source-role inventory record."""
        payload = _strict_mapping(
            value,
            {
                "schema_version",
                "contract",
                "source_index_sha256",
                "role",
                "trajectory_count",
                "source_transition_count",
                "sequences",
                "inventory_sha256",
            },
            "source role inventory",
        )
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _SOURCE_ROLE_INVENTORY_CONTRACT,
        ):
            raise ValueError("source role inventory has an unregistered schema identity")
        sequences = payload["sequences"]
        if type(sequences) is not list:
            raise ValueError("source role inventory sequences must be a JSON list")
        result = cls(
            source_index_sha256=payload["source_index_sha256"],
            role=payload["role"],
            sequences=tuple(_SourceSequenceInventory.from_dict(item) for item in sequences),
        )
        if not _exact_json_equal(result.as_dict(), payload):
            raise ValueError("source role inventory is not the exact canonical record")
        return result


def _source_sequence_from_record(
    source_index_sha256: str,
    record: ReferenceSequenceRecord,
) -> _SourceSequenceInventory:
    return _SourceSequenceInventory(
        source_index_sha256=source_index_sha256,
        role=record.role,
        asset_id=record.asset_id,
        asset_source_sha256=record.asset_source_sha256,
        sequence_id=record.sequence_id,
        producer_topology_sha256=record.topology_sha256,
        producer_operator_sha256=record.operator_sha256,
        producer_material_sha256=record.material_sha256,
        protocol_sha256=record.protocol_sha256,
        producer_manifest_json=record.producer_manifest_json,
        producer_manifest_json_sha256=record.producer_manifest_json_sha256,
        accepted_reference_state_sha256=record.reference_state_float64_sha256,
        source_transition_count=len(record.step_ids),
    )


def _materialized_sequence_identity(
    role: DatasetRole,
    trajectory: PortableDatasetTrajectoryRecord,
) -> tuple[object, ...]:
    provenance = trajectory.provenance
    return (
        role.value,
        provenance.asset_id,
        provenance.asset_source_sha256,
        provenance.sequence_id,
        provenance.producer_topology_sha256,
        provenance.producer_operator_sha256,
        provenance.producer_material_sha256,
        provenance.protocol_sha256,
        provenance.producer_manifest_sha256,
        provenance.accepted_reference_state_sha256,
        trajectory.source_transition_count,
    )


def _source_role_from_records(
    dataset: ReferenceSequenceDataset,
    role: DatasetRole,
) -> _SourceRoleInventory:
    return _SourceRoleInventory(
        source_index_sha256=dataset.index_sha256,
        role=role,
        sequences=tuple(_source_sequence_from_record(dataset.index_sha256, record) for record in dataset.records(role)),
    )


def _materialized_role_identity(
    role: DatasetRole,
    trajectories: Sequence[PortableDatasetTrajectoryRecord],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (_materialized_sequence_identity(role, trajectory) for trajectory in trajectories),
            key=lambda item: (item[1], item[3]),
        )
    )


@dataclasses.dataclass(frozen=True)
class ReferenceSequencePortableConsumerView:
    """Metadata-only TRAIN/VALIDATION projection of one sealed corpus.

    ``view_sha256`` proves only internal consistency. Origin authentication
    requires an out-of-band trusted corpus SHA-256 and exact projection
    validation against that corpus.
    """

    source_corpus_sha256: str
    source_manifest_sha256: str
    roles: tuple[DatasetRole, ...]
    records_by_role: Mapping[DatasetRole, tuple[PortableDatasetTrajectoryRecord, ...]]
    transition_keys_by_sample: Mapping[tuple[str, str], ReferenceTransitionKey]
    view_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_corpus_sha256, "source_corpus_sha256")
        _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        roles = _canonical_roles(self.roles, consumer=True)
        records_source = dict(self.records_by_role)
        if set(records_source) != set(roles):
            raise ValueError("consumer-view record roles differ from selected roles")
        canonical_records: dict[DatasetRole, tuple[PortableDatasetTrajectoryRecord, ...]] = {}
        for role in roles:
            records = tuple(records_source[role])
            if any(type(record) is not PortableDatasetTrajectoryRecord for record in records):
                raise ValueError("consumer view contains a non-canonical portable trajectory")
            records = tuple(sorted(records, key=lambda record: record.trajectory_id))
            canonical_records[role] = records
        PortableDatasetSplitManifest(
            train=canonical_records.get(DatasetRole.TRAIN, ()),
            validation=canonical_records.get(DatasetRole.VALIDATION, ()),
            confirmation=(),
            consumed_regression=(),
        )
        bindings = dict(self.transition_keys_by_sample)
        if any(type(value) is not ReferenceTransitionKey for value in bindings.values()):
            raise ValueError("consumer-view transition values must be canonical keys")
        expected_bindings: dict[tuple[str, str], ReferenceTransitionKey] = {}
        for records in canonical_records.values():
            for record in records:
                for sample in record.samples:
                    source = sample.source_transition
                    if source is None:
                        raise ValueError("consumer-view sample lacks its sealed source transition")
                    lookup = (record.trajectory_id, sample.sample_id)
                    if lookup in expected_bindings:
                        raise ValueError("consumer-view sample lookup identities must be unique")
                    expected_bindings[lookup] = ReferenceTransitionKey(
                        asset_id=source.asset_id,
                        sequence_id=source.sequence_id,
                        step_id=source.step_id,
                    )
        if bindings != expected_bindings:
            raise ValueError("consumer-view transition-key bindings differ from selected metadata")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "records_by_role", MappingProxyType(canonical_records))
        object.__setattr__(self, "transition_keys_by_sample", MappingProxyType(dict(sorted(bindings.items()))))
        object.__setattr__(self, "view_sha256", canonical_json_sha256(self._payload()))

    def validate_authenticated_projection(
        self,
        source_corpus: ReferenceSequencePortableCorpus,
        *,
        trusted_source_corpus_sha256: str,
    ) -> None:
        """Validate exact inclusion under an out-of-band authenticated root.

        Args:
            source_corpus: Canonical metadata corpus authenticated by the
                caller.
            trusted_source_corpus_sha256: Corpus digest obtained independently
                of this view and its serialized source-corpus field.
        """
        trusted = _sha256(trusted_source_corpus_sha256, "trusted source corpus sha256")
        if type(source_corpus) is not ReferenceSequencePortableCorpus:
            raise TypeError("source_corpus must be a canonical ReferenceSequencePortableCorpus")
        if source_corpus.corpus_sha256 != canonical_json_sha256(source_corpus._payload()):
            raise ValueError("authenticated source corpus changed after authentication")
        if source_corpus.corpus_sha256 != trusted:
            raise ValueError("source corpus differs from the externally authenticated root")
        if (
            self.source_corpus_sha256 != trusted
            or self.source_manifest_sha256 != source_corpus.split_manifest.manifest_sha256
        ):
            raise ValueError("consumer view source roots differ from the authenticated corpus")
        if any(role not in source_corpus.prepared_roles for role in self.roles):
            raise ValueError("consumer view selects a role absent from authenticated producer preparation")
        expected_records = {role: source_corpus.split_manifest.records(role) for role in self.roles}
        if dict(self.records_by_role) != expected_records:
            raise ValueError("consumer-view records are not an exact authenticated corpus projection")
        lookups = {
            (record.trajectory_id, sample.sample_id)
            for records in expected_records.values()
            for record in records
            for sample in record.samples
        }
        expected_bindings = {lookup: source_corpus.transition_keys_by_sample[lookup] for lookup in sorted(lookups)}
        if dict(self.transition_keys_by_sample) != expected_bindings:
            raise ValueError("consumer-view bindings are not an exact authenticated corpus projection")

    def records(self, role: DatasetRole | str) -> tuple[PortableDatasetTrajectoryRecord, ...]:
        """Return metadata for one selected role."""
        canonical = DatasetRole(role)
        if canonical not in self.roles:
            raise ValueError(f"role {canonical.value!r} is not exposed by this consumer view")
        return self.records_by_role[canonical]

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _CONSUMER_VIEW_CONTRACT,
            "claim_scope": _CONSUMER_VIEW_CLAIM_SCOPE,
            "source_corpus_sha256": self.source_corpus_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "roles": [role.value for role in self.roles],
            "records": {role.value: [record.as_dict() for record in self.records_by_role[role]] for role in self.roles},
            "transition_keys": [
                {
                    "trajectory_id": trajectory_id,
                    "sample_id": sample_id,
                    "asset_id": key.asset_id,
                    "sequence_id": key.sequence_id,
                    "step_id": key.step_id,
                }
                for (trajectory_id, sample_id), key in self.transition_keys_by_sample.items()
            ],
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["view_sha256"] = self.view_sha256
        return payload

    def to_json(self) -> str:
        """Return canonical consumer-view JSON text."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        authenticated_source_corpus: ReferenceSequencePortableCorpus,
        trusted_source_corpus_sha256: str,
    ) -> ReferenceSequencePortableConsumerView:
        """Strictly reconstruct and origin-authenticate one consumer view."""
        payload = _strict_mapping(
            value,
            {
                "schema_version",
                "contract",
                "claim_scope",
                "source_corpus_sha256",
                "source_manifest_sha256",
                "roles",
                "records",
                "transition_keys",
                "view_sha256",
            },
            "portable consumer view",
        )
        if (payload["schema_version"], payload["contract"], payload["claim_scope"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _CONSUMER_VIEW_CONTRACT,
            _CONSUMER_VIEW_CLAIM_SCOPE,
        ):
            raise ValueError("portable consumer view has an unregistered schema identity")
        role_values = payload["roles"]
        if type(role_values) is not list:
            raise ValueError("portable consumer-view roles must be a JSON list")
        roles = _canonical_roles(role_values, consumer=True)
        record_payloads = _strict_mapping(
            payload["records"], {role.value for role in roles}, "portable consumer-view records"
        )
        records_by_role: dict[DatasetRole, tuple[PortableDatasetTrajectoryRecord, ...]] = {}
        for role in roles:
            role_records = record_payloads[role.value]
            if type(role_records) is not list:
                raise ValueError("portable consumer-view role records must be a JSON list")
            records_by_role[role] = tuple(PortableDatasetTrajectoryRecord.from_dict(record) for record in role_records)
        key_payloads = payload["transition_keys"]
        if type(key_payloads) is not list:
            raise ValueError("portable consumer-view transition_keys must be a JSON list")
        bindings: dict[tuple[str, str], ReferenceTransitionKey] = {}
        for item in key_payloads:
            key_record = _strict_mapping(
                item,
                {"trajectory_id", "sample_id", "asset_id", "sequence_id", "step_id"},
                "portable consumer-view transition key",
            )
            lookup = (key_record["trajectory_id"], key_record["sample_id"])
            if lookup in bindings:
                raise ValueError("portable consumer view contains duplicate transition lookup keys")
            bindings[lookup] = ReferenceTransitionKey(
                asset_id=key_record["asset_id"],
                sequence_id=key_record["sequence_id"],
                step_id=key_record["step_id"],
            )
        result = cls(
            source_corpus_sha256=payload["source_corpus_sha256"],
            source_manifest_sha256=payload["source_manifest_sha256"],
            roles=roles,
            records_by_role=records_by_role,
            transition_keys_by_sample=bindings,
        )
        if not _exact_json_equal(result.as_dict(), payload):
            raise ValueError("portable consumer view is not the exact canonical record")
        result.validate_authenticated_projection(
            authenticated_source_corpus,
            trusted_source_corpus_sha256=trusted_source_corpus_sha256,
        )
        return result

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        *,
        authenticated_source_corpus: ReferenceSequencePortableCorpus,
        trusted_source_corpus_sha256: str,
    ) -> ReferenceSequencePortableConsumerView:
        """Strictly deserialize and origin-authenticate one consumer view."""
        return cls.from_dict(
            _read_json(value, "portable consumer view"),
            authenticated_source_corpus=authenticated_source_corpus,
            trusted_source_corpus_sha256=trusted_source_corpus_sha256,
        )


@dataclasses.dataclass(frozen=True)
class ReferenceSequencePortableCorpus:
    """Sealed portable metadata and exact lazy transition lookup."""

    split_manifest: PortableDatasetSplitManifest
    transition_keys_by_sample: Mapping[tuple[str, str], ReferenceTransitionKey]
    source_index_sha256: str
    source_role_inventories: Mapping[DatasetRole, _SourceRoleInventory]
    prepared_roles: tuple[DatasetRole, ...]
    corpus_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.split_manifest) is not PortableDatasetSplitManifest:
            raise TypeError("split_manifest must be a canonical PortableDatasetSplitManifest")
        if PortableDatasetSplitManifest.from_dict(self.split_manifest.as_dict()) != self.split_manifest:
            raise ValueError("portable split manifest failed self-authentication")
        _sha256(self.source_index_sha256, "source_index_sha256")
        roles = _canonical_roles(self.prepared_roles)
        inventory_source = dict(self.source_role_inventories)
        if set(inventory_source) != set(_ROLE_ORDER):
            raise ValueError("source role inventories must cover every dataset role")
        inventories: dict[DatasetRole, _SourceRoleInventory] = {}
        for role in _ROLE_ORDER:
            inventory = inventory_source[role]
            if type(inventory) is not _SourceRoleInventory:
                raise ValueError("source role inventory must be a canonical inventory record")
            if (
                inventory.role is not role
                or inventory.source_index_sha256 != self.source_index_sha256
                or inventory.inventory_sha256 != canonical_json_sha256(inventory._payload())
            ):
                raise ValueError("source role inventory differs from the corpus source index or role")
            materialized_identity = _materialized_role_identity(
                role,
                self.split_manifest.records(role),
            )
            source_identity = tuple(sequence.materialized_identity for sequence in inventory.sequences)
            if role in roles:
                if materialized_identity != source_identity:
                    raise ValueError(
                        "producer preparation materialization does not cover the complete source role inventory"
                    )
            elif materialized_identity:
                raise ValueError("portable manifest contains an unrequested producer preparation role")
            inventories[role] = inventory
        source_index_preimage = reference_sequence_index_header()
        source_index_preimage["splits"] = {
            role.value: [sequence.index_record() for sequence in inventories[role].sequences] for role in _ROLE_ORDER
        }
        if canonical_json_sha256(source_index_preimage) != self.source_index_sha256:
            raise ValueError("source role inventories do not reconstruct the authenticated source index")

        expected: dict[tuple[str, str], ReferenceTransitionKey] = {}
        for role in _ROLE_ORDER:
            for trajectory in self.split_manifest.records(role):
                provenance = trajectory.provenance
                if provenance.dataset_index_sha256 != self.source_index_sha256:
                    raise ValueError("portable trajectory provenance binds a different sequence index")
                if (
                    trajectory.trajectory_id != _trajectory_id(provenance.asset_id, provenance.sequence_id)
                    or trajectory.scene_family != f"reference-sequence:{provenance.asset_id}"
                    or trajectory.load_program_id != trajectory.trajectory_id
                    or trajectory.load_program_sha256 != canonical_json_sha256(_load_program_payload(provenance))
                    or trajectory.source_chain_sha256 != canonical_json_sha256(_source_chain_payload(provenance))
                ):
                    raise ValueError("portable trajectory metadata differs from its reference-sequence provenance")
                for sample in trajectory.samples:
                    source = sample.source_transition
                    if source is None:
                        raise ValueError("portable corpus sample lacks its sealed source transition")
                    expected[(trajectory.trajectory_id, sample.sample_id)] = ReferenceTransitionKey(
                        asset_id=source.asset_id,
                        sequence_id=source.sequence_id,
                        step_id=source.step_id,
                    )
        bindings = dict(self.transition_keys_by_sample)
        if bindings != expected:
            raise ValueError("transition-key bindings differ from the complete portable split manifest")
        if any(type(value) is not ReferenceTransitionKey for value in bindings.values()):
            raise ValueError("transition-key values must be canonical ReferenceTransitionKey values")
        object.__setattr__(self, "source_role_inventories", MappingProxyType(inventories))
        object.__setattr__(self, "prepared_roles", roles)
        object.__setattr__(self, "transition_keys_by_sample", MappingProxyType(dict(sorted(bindings.items()))))
        object.__setattr__(self, "corpus_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _CORPUS_CONTRACT,
            "source_index_sha256": self.source_index_sha256,
            "source_role_inventories": {
                role.value: self.source_role_inventories[role].as_dict() for role in _ROLE_ORDER
            },
            "producer_preparation": {
                "contract": _PREPARATION_RECEIPT_CONTRACT,
                "claim_scope": _PREPARATION_CLAIM_SCOPE,
                "prepared_roles": [role.value for role in self.prepared_roles],
            },
            "split_manifest": self.split_manifest.as_dict(),
            "transition_keys": [
                {
                    "trajectory_id": trajectory_id,
                    "sample_id": sample_id,
                    "asset_id": key.asset_id,
                    "sequence_id": key.sequence_id,
                    "step_id": key.step_id,
                }
                for (trajectory_id, sample_id), key in self.transition_keys_by_sample.items()
            ],
        }

    def as_dict(self) -> dict[str, object]:
        """Return a self-checking JSON object."""
        payload = self._payload()
        payload["corpus_sha256"] = self.corpus_sha256
        return payload

    def to_json(self) -> str:
        """Return canonical corpus JSON text."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, value: object) -> ReferenceSequencePortableCorpus:
        """Strictly reconstruct sealed portable corpus metadata."""
        payload = _strict_mapping(
            value,
            {
                "schema_version",
                "contract",
                "source_index_sha256",
                "source_role_inventories",
                "producer_preparation",
                "split_manifest",
                "transition_keys",
                "corpus_sha256",
            },
            "portable corpus",
        )
        if (payload["schema_version"], payload["contract"]) != (
            PORTABLE_DATASET_SCHEMA_VERSION,
            _CORPUS_CONTRACT,
        ):
            raise ValueError("portable corpus has an unregistered schema identity")
        receipt = _strict_mapping(
            payload["producer_preparation"],
            {"contract", "claim_scope", "prepared_roles"},
            "producer preparation receipt",
        )
        if (receipt["contract"], receipt["claim_scope"]) != (
            _PREPARATION_RECEIPT_CONTRACT,
            _PREPARATION_CLAIM_SCOPE,
        ):
            raise ValueError("portable corpus preparation receipt is not registered")
        if type(receipt["prepared_roles"]) is not list:
            raise ValueError("producer preparation roles must be a JSON list")
        inventory_payloads = _strict_mapping(
            payload["source_role_inventories"],
            {role.value for role in _ROLE_ORDER},
            "source role inventories",
        )
        inventories = {role: _SourceRoleInventory.from_dict(inventory_payloads[role.value]) for role in _ROLE_ORDER}
        key_payloads = payload["transition_keys"]
        if type(key_payloads) is not list:
            raise ValueError("portable corpus transition_keys must be a JSON list")
        bindings: dict[tuple[str, str], ReferenceTransitionKey] = {}
        for item in key_payloads:
            key_record = _strict_mapping(
                item,
                {"trajectory_id", "sample_id", "asset_id", "sequence_id", "step_id"},
                "portable corpus transition key",
            )
            lookup = (key_record["trajectory_id"], key_record["sample_id"])
            if lookup in bindings:
                raise ValueError("portable corpus contains duplicate transition lookup keys")
            bindings[lookup] = ReferenceTransitionKey(
                asset_id=key_record["asset_id"],
                sequence_id=key_record["sequence_id"],
                step_id=key_record["step_id"],
            )
        result = cls(
            split_manifest=PortableDatasetSplitManifest.from_dict(payload["split_manifest"]),
            transition_keys_by_sample=bindings,
            source_index_sha256=payload["source_index_sha256"],
            source_role_inventories=inventories,
            prepared_roles=tuple(receipt["prepared_roles"]),
        )
        if not _exact_json_equal(result.as_dict(), payload):
            raise ValueError("portable corpus is not the exact canonical record")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> ReferenceSequencePortableCorpus:
        """Strictly deserialize sealed corpus metadata without loading arrays."""
        return cls.from_dict(_read_json(value, "portable corpus"))

    def consumer_view(
        self,
        roles: Sequence[DatasetRole | str] = (DatasetRole.TRAIN, DatasetRole.VALIDATION),
    ) -> ReferenceSequencePortableConsumerView:
        """Build a metadata-only TRAIN/VALIDATION view.

        This method does not own a dataset or bridge and therefore cannot load
        a static or dynamic array. The returned self-hash proves consistency,
        not origin. After crossing a serialization boundary, callers must use
        :meth:`ReferenceSequencePortableConsumerView.from_json` with a corpus
        and trusted corpus SHA-256 obtained outside the view. Payload access
        still requires a separate branch-local ledger and lazy bridge call.
        """
        selected = _canonical_roles(roles, consumer=True)
        if any(role not in self.prepared_roles for role in selected):
            raise ValueError("consumer view may select only prepared corpus roles")
        records_by_role = {role: self.split_manifest.records(role) for role in selected}
        lookups = {
            (record.trajectory_id, sample.sample_id)
            for records in records_by_role.values()
            for record in records
            for sample in record.samples
        }
        bindings = {lookup: self.transition_keys_by_sample[lookup] for lookup in sorted(lookups)}
        result = ReferenceSequencePortableConsumerView(
            source_corpus_sha256=self.corpus_sha256,
            source_manifest_sha256=self.split_manifest.manifest_sha256,
            roles=selected,
            records_by_role=records_by_role,
            transition_keys_by_sample=bindings,
        )
        result.validate_authenticated_projection(self, trusted_source_corpus_sha256=self.corpus_sha256)
        return result


def _build_trajectory(
    dataset: ReferenceSequenceDataset,
    bridge: ReferenceSequencePortableDatasetBridge,
    record: ReferenceSequenceRecord,
    transition_bindings: dict[tuple[str, str], ReferenceTransitionKey],
) -> PortableDatasetTrajectoryRecord:
    provenance = PortableReferenceSequenceProvenance(
        **dataclasses.asdict(dataset.provenance_anchor(record)),
        producer_topology_sha256=record.topology_sha256,
        producer_operator_sha256=record.operator_sha256,
        producer_material_sha256=record.material_sha256,
        accepted_reference_state_sha256=record.reference_state_float64_sha256,
    )
    trajectory_id = _trajectory_id(record.asset_id, record.sequence_id)
    sample_records = []
    static_identity: tuple[str, str, str, str, str, str] | None = None
    materialized_asset_identity: ReferencePortableAssetIdentities | None = None
    sequence_source_identity: tuple[str, ...] | None = None
    for step_id in record.step_ids:
        transition_key = ReferenceTransitionKey(record.asset_id, record.sequence_id, step_id)
        materialized = bridge.materialize(transition_key)
        sample = materialized.sample_record
        if (
            materialized.transition_key != transition_key
            or materialized.key != (trajectory_id, f"step-{step_id:08d}")
            or sample.ordinal != step_id
        ):
            raise ValueError("portable bridge identity differs from the sequence record")
        source = materialized.source_transition
        expected_source = (
            dataset.index_sha256,
            record.asset_id,
            record.asset_source_sha256,
            record.sequence_id,
            step_id,
            provenance.static_bundle_sha256,
            provenance.sequence_bundle_sha256,
            record.protocol_sha256,
            record.topology_sha256,
            record.operator_sha256,
            record.material_sha256,
            record.reference_state_float64_sha256[step_id],
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
            raise ValueError("materialized source transition differs from authenticated sequence metadata")
        expected_asset_identity = ReferencePortableAssetIdentities(
            asset_id=source.asset_id,
            reference_sequence_index_sha256=source.reference_sequence_index_sha256,
            asset_source_sha256=source.asset_source_sha256,
            static_npz_sha256=source.static_npz_sha256,
            producer_topology_sha256=source.producer_topology_sha256,
            producer_operator_sha256=source.producer_operator_sha256,
            producer_material_sha256=source.producer_material_sha256,
            portable_topology_sha256=source.portable_topology_sha256,
            operator_geometry_policy=source.operator_geometry_policy,
            operator_geometry_sha256=source.operator_geometry_sha256,
            operator_volume_policy=source.operator_volume_policy,
            operator_volume_sha256=source.operator_volume_sha256,
            portable_material_sha256=source.portable_material_sha256,
            portable_pin_signature_sha256=source.portable_pin_signature_sha256,
        )
        if materialized.identities != expected_asset_identity:
            raise ValueError("materialized asset identities differ from the authenticated source transition")
        if materialized_asset_identity is None:
            materialized_asset_identity = materialized.identities
        elif materialized.identities != materialized_asset_identity:
            raise ValueError("one sequence resolved to non-constant materialized asset identities")
        current_sequence_source = (
            source.reference_sequence_index_sha256,
            source.asset_id,
            source.asset_source_sha256,
            source.sequence_id,
            source.static_npz_sha256,
            source.sequence_npz_sha256,
            source.protocol_sha256,
            source.producer_topology_sha256,
            source.producer_operator_sha256,
            source.producer_material_sha256,
        )
        if sequence_source_identity is None:
            sequence_source_identity = current_sequence_source
        elif current_sequence_source != sequence_source_identity:
            raise ValueError("one sequence resolved to non-constant source identities")
        materialized.validate_immutable()
        current_static = (
            sample.topology_sha256,
            sample.operator_geometry_policy,
            sample.operator_geometry_sha256,
            sample.operator_volume_policy,
            sample.operator_volume_sha256,
            sample.material_sha256,
        )
        if static_identity is None:
            static_identity = current_static
        elif current_static != static_identity:
            raise ValueError("one sequence resolved to conflicting portable static identities")
        lookup = (trajectory_id, sample.sample_id)
        if lookup in transition_bindings:
            raise ValueError("duplicate portable bridge sample identity")
        transition_bindings[lookup] = transition_key
        sample_records.append(sample)
        del materialized
    if static_identity is None:
        raise RuntimeError("authenticated sequence unexpectedly contained no transitions")
    topology, geometry_policy, geometry_sha256, volume_policy, volume_sha256, material_sha256 = static_identity
    return PortableDatasetTrajectoryRecord(
        trajectory_id=trajectory_id,
        scene_family=f"reference-sequence:{record.asset_id}",
        load_program_id=trajectory_id,
        load_program_sha256=canonical_json_sha256(_load_program_payload(provenance)),
        source_chain_sha256=canonical_json_sha256(_source_chain_payload(provenance)),
        topology_sha256=topology,
        operator_geometry_policy=geometry_policy,
        operator_geometry_sha256=geometry_sha256,
        operator_volume_policy=volume_policy,
        operator_volume_sha256=volume_sha256,
        material_sha256=material_sha256,
        provenance=provenance,
        source_transition_count=provenance.source_transition_count,
        samples=tuple(sample_records),
    )


def materialize_reference_sequence_portable_corpus(
    dataset: ReferenceSequenceDataset,
    bridge: ReferenceSequencePortableDatasetBridge,
    *,
    roles: Sequence[DatasetRole | str] = tuple(DatasetRole),
) -> ReferenceSequencePortableCorpus:
    """Producer-only materialization of requested portable metadata roles.

    This preparation API opens every transition in ``roles``. Its receipt
    proves that output metadata completely covers each requested role under
    the stored source-index inventory; it is not independent proof that a
    payload was opened and is not global access control. Training consumers
    should deserialize the sealed result with
    :meth:`ReferenceSequencePortableCorpus.from_json`, call
    :meth:`ReferenceSequencePortableCorpus.consumer_view`, and materialize
    only scheduled TRAIN/VALIDATION transitions on demand.
    """
    if type(dataset) is not ReferenceSequenceDataset:
        raise TypeError("dataset must be a canonical ReferenceSequenceDataset")
    if type(bridge) is not ReferenceSequencePortableDatasetBridge:
        raise TypeError("bridge must be a canonical ReferenceSequencePortableDatasetBridge")
    if bridge.dataset is not dataset:
        raise ValueError("bridge must own the exact dataset")
    selected = _canonical_roles(roles)
    source_role_inventories = {role: _source_role_from_records(dataset, role) for role in _ROLE_ORDER}
    bindings: dict[tuple[str, str], ReferenceTransitionKey] = {}
    records_by_role: dict[DatasetRole, tuple[PortableDatasetTrajectoryRecord, ...]] = {}
    for role in _ROLE_ORDER:
        records_by_role[role] = (
            tuple(_build_trajectory(dataset, bridge, record, bindings) for record in dataset.records(role))
            if role in selected
            else ()
        )
    manifest = PortableDatasetSplitManifest(
        train=records_by_role[DatasetRole.TRAIN],
        validation=records_by_role[DatasetRole.VALIDATION],
        confirmation=records_by_role[DatasetRole.CONFIRMATION],
        consumed_regression=records_by_role[DatasetRole.CONSUMED_REGRESSION],
    )
    return ReferenceSequencePortableCorpus(
        split_manifest=manifest,
        transition_keys_by_sample=bindings,
        source_index_sha256=dataset.index_sha256,
        source_role_inventories=source_role_inventories,
        prepared_roles=selected,
    )


__all__ = [
    "ReferenceSequencePortableConsumerView",
    "ReferenceSequencePortableCorpus",
    "materialize_reference_sequence_portable_corpus",
]
