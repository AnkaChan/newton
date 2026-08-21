# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build canonical v5 split metadata from authenticated reference sequences.

This is a dataset-preparation boundary, not a training payload cache.  It
materializes one transition at a time through :class:`ReferenceSequenceV5Bridge`,
retains only immutable sample records and transition-key bindings, and drops
each ``V5TrainingSample`` before advancing to the next transition.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from .reference_sequence_dataset import ReferenceSequenceDataset, ReferenceSequenceRecord, ReferenceTransitionKey
from .reference_sequence_v5_bridge import ReferenceSequenceV5Bridge
from .v5_checkpoint import canonical_json_sha256
from .v5_dataset import (
    DatasetRole,
    ReferenceSequenceProvenance,
    SplitManifest,
    TrajectoryRecord,
    TrajectorySampleRecord,
    _verify_manifest,
)

_CORPUS_CONTRACT = "pss-reference-sequence-v5-corpus-v1"
_SOURCE_CHAIN_CONTRACT = "pss-reference-sequence-source-chain-v1"
_LOAD_PROGRAM_CONTRACT = "pss-reference-sequence-dynamics-program-v1"


@dataclasses.dataclass(frozen=True)
class ReferenceSequenceV5Corpus:
    """Frozen v5 manifest and lazy bridge lookup for one sequence index."""

    split_manifest: SplitManifest
    transition_keys_by_sample: Mapping[tuple[str, str], ReferenceTransitionKey]
    source_index_sha256: str
    corpus_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.split_manifest) is not SplitManifest:
            raise TypeError("split_manifest must be a canonical SplitManifest")
        _verify_manifest(self.split_manifest)
        if (
            type(self.source_index_sha256) is not str
            or len(self.source_index_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_index_sha256)
        ):
            raise ValueError("source_index_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.transition_keys_by_sample, Mapping):
            raise TypeError("transition_keys_by_sample must be a mapping")

        bindings = dict(self.transition_keys_by_sample)
        expected: dict[tuple[str, str], ReferenceTransitionKey] = {}
        for role in DatasetRole:
            for trajectory in self.split_manifest.records(role):
                provenance = trajectory.provenance
                if type(provenance) is not ReferenceSequenceProvenance:
                    raise ValueError("reference-sequence corpus trajectories require sequence-native provenance")
                if provenance.dataset_index_sha256 != self.source_index_sha256:
                    raise ValueError("trajectory provenance binds a different source index")
                for sample in trajectory.samples:
                    expected[(trajectory.trajectory_id, sample.sample_id)] = ReferenceTransitionKey(
                        asset_id=provenance.asset_id,
                        sequence_id=provenance.sequence_id,
                        step_id=sample.ordinal,
                    )
        if bindings != expected:
            raise ValueError("transition-key bindings differ from the complete split manifest")
        if any(type(key) is not tuple or len(key) != 2 for key in bindings):
            raise ValueError("transition-key lookup keys must be exact (trajectory_id, sample_id) tuples")
        if any(type(value) is not ReferenceTransitionKey for value in bindings.values()):
            raise ValueError("transition-key lookup values must be canonical ReferenceTransitionKey values")

        ordered = dict(sorted(bindings.items()))
        object.__setattr__(self, "transition_keys_by_sample", MappingProxyType(ordered))
        object.__setattr__(self, "corpus_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _CORPUS_CONTRACT,
            "source_index_sha256": self.source_index_sha256,
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
        """Return canonical JSON-compatible corpus metadata."""
        payload = self._payload()
        payload["corpus_sha256"] = self.corpus_sha256
        return payload


def _trajectory_id(asset_id: str, sequence_id: str) -> str:
    return f"reference-sequence:{asset_id}:{sequence_id}"


def _load_program_payload(provenance: ReferenceSequenceProvenance) -> dict[str, object]:
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


def _source_chain_payload(provenance: ReferenceSequenceProvenance) -> dict[str, object]:
    return {
        "contract": _SOURCE_CHAIN_CONTRACT,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "producer_manifest_sha256": provenance.producer_manifest_sha256,
        "static_bundle_sha256": provenance.static_bundle_sha256,
        "sequence_bundle_sha256": provenance.sequence_bundle_sha256,
        "evidence_sha256": provenance.evidence_sha256,
    }


def _build_trajectory(
    dataset: ReferenceSequenceDataset,
    bridge: ReferenceSequenceV5Bridge,
    record: ReferenceSequenceRecord,
    transition_bindings: dict[tuple[str, str], ReferenceTransitionKey],
) -> TrajectoryRecord:
    anchor = dataset.provenance_anchor(record)
    provenance = ReferenceSequenceProvenance(**dataclasses.asdict(anchor))
    trajectory_id = _trajectory_id(record.asset_id, record.sequence_id)
    sample_records: list[TrajectorySampleRecord] = []
    topology_sha256: str | None = None
    operator_geometry_sha256: str | None = None
    material_sha256: str | None = None
    for step_id in record.step_ids:
        transition_key = ReferenceTransitionKey(record.asset_id, record.sequence_id, step_id)
        materialized = bridge.materialize(transition_key)
        sample = materialized.training_sample.sample_record
        if materialized.training_sample.trajectory_id != trajectory_id:
            raise ValueError("bridge trajectory identity differs from the sequence record")
        if sample.ordinal != step_id:
            raise ValueError("bridge sample ordinal differs from the sequence transition")
        if topology_sha256 is None:
            topology_sha256 = sample.topology_sha256
            operator_geometry_sha256 = sample.operator_geometry_sha256
            material_sha256 = sample.material_sha256
        elif (
            sample.topology_sha256 != topology_sha256
            or sample.operator_geometry_sha256 != operator_geometry_sha256
            or sample.material_sha256 != material_sha256
        ):
            raise ValueError("one sequence resolved to conflicting v5 static identities")
        lookup_key = (trajectory_id, sample.sample_id)
        if lookup_key in transition_bindings:
            raise ValueError("duplicate bridge sample identity")
        transition_bindings[lookup_key] = transition_key
        sample_records.append(sample)
        del materialized

    if topology_sha256 is None or operator_geometry_sha256 is None or material_sha256 is None:
        raise RuntimeError("authenticated sequence unexpectedly contained no transitions")
    load_program = _load_program_payload(provenance)
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        scene_family=f"reference-sequence:{record.asset_id}",
        load_program_id=trajectory_id,
        load_program_sha256=canonical_json_sha256(load_program),
        source_chain_sha256=canonical_json_sha256(_source_chain_payload(provenance)),
        topology_sha256=topology_sha256,
        operator_geometry_sha256=operator_geometry_sha256,
        material_sha256=material_sha256,
        provenance=provenance,
        source_transition_count=provenance.source_transition_count,
        samples=tuple(sample_records),
    )


def build_reference_sequence_v5_corpus(
    dataset: ReferenceSequenceDataset,
    bridge: ReferenceSequenceV5Bridge,
) -> ReferenceSequenceV5Corpus:
    """Build complete v5 metadata while retaining no dynamic training samples."""
    if type(dataset) is not ReferenceSequenceDataset:
        raise TypeError("dataset must be a canonical ReferenceSequenceDataset")
    if type(bridge) is not ReferenceSequenceV5Bridge:
        raise TypeError("bridge must be a canonical ReferenceSequenceV5Bridge")
    if bridge.dataset is not dataset:
        raise ValueError("bridge must own the exact dataset")

    transition_bindings: dict[tuple[str, str], ReferenceTransitionKey] = {}
    records_by_role: dict[DatasetRole, tuple[TrajectoryRecord, ...]] = {}
    for role in DatasetRole:
        records_by_role[role] = tuple(
            _build_trajectory(dataset, bridge, record, transition_bindings) for record in dataset.records(role)
        )
    manifest = SplitManifest(
        train=records_by_role[DatasetRole.TRAIN],
        validation=records_by_role[DatasetRole.VALIDATION],
        confirmation=records_by_role[DatasetRole.CONFIRMATION],
        consumed_regression=records_by_role[DatasetRole.CONSUMED_REGRESSION],
    )
    return ReferenceSequenceV5Corpus(
        split_manifest=manifest,
        transition_keys_by_sample=transition_bindings,
        source_index_sha256=dataset.index_sha256,
    )


__all__ = ["ReferenceSequenceV5Corpus", "build_reference_sequence_v5_corpus"]
