# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Portable-volume corpus metadata for authenticated reference sequences.

The producer entry point is explicitly preparation-only: it materializes the
requested roles and records those roles in its receipt. Training consumers
must load the resulting authenticated JSON metadata and derive a
TRAIN/VALIDATION view; that path opens no sequence arrays. Runtime payloads
remain lazy and are loaded one transition at a time by the bridge.
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
)
from .reference_sequence_v5_bridge import ReferenceSequencePortableDatasetBridge
from .v5_checkpoint import canonical_json_sha256
from .v5_dataset import DatasetRole

_CORPUS_CONTRACT = "pss-reference-sequence-portable-volume-corpus-v1"
_MATERIALIZATION_RECEIPT_CONTRACT = "pss-reference-sequence-portable-volume-producer-materialization-v1"
_MATERIALIZATION_CLAIM_SCOPE = "producer-preparation-roles-materialized-not-consumer-access-control"
_CONSUMER_VIEW_CONTRACT = "pss-reference-sequence-portable-volume-consumer-view-v1"
_CONSUMER_VIEW_CLAIM_SCOPE = "metadata-only-train-validation-view-no-payload-access-proof"
_SOURCE_CHAIN_CONTRACT = "pss-reference-sequence-portable-volume-source-chain-v1"
_LOAD_PROGRAM_CONTRACT = "pss-reference-sequence-portable-volume-dynamics-program-v1"

_ROLE_ORDER = tuple(DatasetRole)
_CONSUMER_ROLES = frozenset((DatasetRole.TRAIN, DatasetRole.VALIDATION))


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
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


@dataclasses.dataclass(frozen=True)
class ReferenceSequencePortableConsumerView:
    """Metadata-only TRAIN/VALIDATION projection of one sealed corpus."""

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
        selected_trajectory_ids: set[str] = set()
        for role in roles:
            records = tuple(records_source[role])
            if any(type(record) is not PortableDatasetTrajectoryRecord for record in records):
                raise ValueError("consumer view contains a non-canonical portable trajectory")
            records = tuple(sorted(records, key=lambda record: record.trajectory_id))
            selected_trajectory_ids.update(record.trajectory_id for record in records)
            canonical_records[role] = records
        bindings = dict(self.transition_keys_by_sample)
        expected_keys = {
            (record.trajectory_id, sample.sample_id)
            for records in canonical_records.values()
            for record in records
            for sample in record.samples
        }
        if set(bindings) != expected_keys:
            raise ValueError("consumer-view transition-key bindings differ from selected metadata")
        if any(type(value) is not ReferenceTransitionKey for value in bindings.values()):
            raise ValueError("consumer-view transition values must be canonical keys")
        if any(trajectory_id not in selected_trajectory_ids for trajectory_id, _sample_id in bindings):
            raise ValueError("consumer-view transition key names an excluded trajectory")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "records_by_role", MappingProxyType(canonical_records))
        object.__setattr__(self, "transition_keys_by_sample", MappingProxyType(dict(sorted(bindings.items()))))
        object.__setattr__(self, "view_sha256", canonical_json_sha256(self._payload()))

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
    def from_dict(cls, value: object) -> ReferenceSequencePortableConsumerView:
        """Strictly reconstruct one metadata-only consumer view."""
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
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> ReferenceSequencePortableConsumerView:
        """Strictly deserialize one metadata-only consumer view."""
        return cls.from_dict(_read_json(value, "portable consumer view"))


@dataclasses.dataclass(frozen=True)
class ReferenceSequencePortableCorpus:
    """Sealed portable metadata and exact lazy transition lookup."""

    split_manifest: PortableDatasetSplitManifest
    transition_keys_by_sample: Mapping[tuple[str, str], ReferenceTransitionKey]
    source_index_sha256: str
    materialized_roles: tuple[DatasetRole, ...]
    corpus_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.split_manifest) is not PortableDatasetSplitManifest:
            raise TypeError("split_manifest must be a canonical PortableDatasetSplitManifest")
        if PortableDatasetSplitManifest.from_dict(self.split_manifest.as_dict()) != self.split_manifest:
            raise ValueError("portable split manifest failed self-authentication")
        _sha256(self.source_index_sha256, "source_index_sha256")
        roles = _canonical_roles(self.materialized_roles)
        roles_with_records = {role for role in _ROLE_ORDER if self.split_manifest.records(role)}
        if not roles_with_records.issubset(roles):
            raise ValueError("producer materialization receipt omits a role present in the portable manifest")

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
                    expected[(trajectory.trajectory_id, sample.sample_id)] = ReferenceTransitionKey(
                        asset_id=provenance.asset_id,
                        sequence_id=provenance.sequence_id,
                        step_id=sample.ordinal,
                    )
        bindings = dict(self.transition_keys_by_sample)
        if bindings != expected:
            raise ValueError("transition-key bindings differ from the complete portable split manifest")
        if any(type(value) is not ReferenceTransitionKey for value in bindings.values()):
            raise ValueError("transition-key values must be canonical ReferenceTransitionKey values")
        object.__setattr__(self, "materialized_roles", roles)
        object.__setattr__(self, "transition_keys_by_sample", MappingProxyType(dict(sorted(bindings.items()))))
        object.__setattr__(self, "corpus_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": PORTABLE_DATASET_SCHEMA_VERSION,
            "contract": _CORPUS_CONTRACT,
            "source_index_sha256": self.source_index_sha256,
            "producer_materialization": {
                "contract": _MATERIALIZATION_RECEIPT_CONTRACT,
                "claim_scope": _MATERIALIZATION_CLAIM_SCOPE,
                "roles": [role.value for role in self.materialized_roles],
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
                "producer_materialization",
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
            payload["producer_materialization"], {"contract", "claim_scope", "roles"}, "materialization receipt"
        )
        if (receipt["contract"], receipt["claim_scope"]) != (
            _MATERIALIZATION_RECEIPT_CONTRACT,
            _MATERIALIZATION_CLAIM_SCOPE,
        ):
            raise ValueError("portable corpus materialization receipt is not registered")
        if type(receipt["roles"]) is not list:
            raise ValueError("materialization receipt roles must be a JSON list")
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
            materialized_roles=tuple(receipt["roles"]),
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
        a static or dynamic array. Its claim is limited to metadata selection;
        payload access still requires a separate branch-local ledger and lazy
        bridge call.
        """
        selected = _canonical_roles(roles, consumer=True)
        records_by_role = {role: self.split_manifest.records(role) for role in selected}
        selected_trajectories = {record.trajectory_id for records in records_by_role.values() for record in records}
        bindings = {
            key: transition
            for key, transition in self.transition_keys_by_sample.items()
            if key[0] in selected_trajectories
        }
        return ReferenceSequencePortableConsumerView(
            source_corpus_sha256=self.corpus_sha256,
            source_manifest_sha256=self.split_manifest.manifest_sha256,
            roles=selected,
            records_by_role=records_by_role,
            transition_keys_by_sample=bindings,
        )


def _build_trajectory(
    dataset: ReferenceSequenceDataset,
    bridge: ReferenceSequencePortableDatasetBridge,
    record: ReferenceSequenceRecord,
    transition_bindings: dict[tuple[str, str], ReferenceTransitionKey],
) -> PortableDatasetTrajectoryRecord:
    provenance = PortableReferenceSequenceProvenance(**dataclasses.asdict(dataset.provenance_anchor(record)))
    trajectory_id = _trajectory_id(record.asset_id, record.sequence_id)
    sample_records = []
    static_identity: tuple[str, str, str, str, str, str] | None = None
    for step_id in record.step_ids:
        transition_key = ReferenceTransitionKey(record.asset_id, record.sequence_id, step_id)
        materialized = bridge.materialize(transition_key)
        sample = materialized.sample_record
        if materialized.key[0] != trajectory_id or sample.ordinal != step_id:
            raise ValueError("portable bridge identity differs from the sequence record")
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

    This preparation API opens every transition in ``roles``. Its returned
    receipt states exactly which roles it materialized, but is not proof of a
    global embargo. Training consumers should deserialize the sealed result
    with :meth:`ReferenceSequencePortableCorpus.from_json`, call
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
        materialized_roles=selected,
    )


__all__ = [
    "ReferenceSequencePortableConsumerView",
    "ReferenceSequencePortableCorpus",
    "materialize_reference_sequence_portable_corpus",
]
