# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the successor portable-volume dataset contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import unittest

import numpy as np

from .. import train_pr_history_v5 as trainer_contract
from .. import v5_checkpoint as checkpoint_contract
from ..portable_dataset import (
    PORTABLE_DATASET_SCHEMA_VERSION,
    PortableDatasetAccessLedger,
    PortableDatasetSampleRecord,
    PortableDatasetSplitManifest,
    PortableDatasetTrajectoryRecord,
    PortableNumericContentIdentity,
    PortableReferenceSequenceProvenance,
    PortableReferenceSourceTransitionIdentity,
    build_portable_sampling_schedule,
)
from ..torch_solver import (
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
    OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT,
)
from ..v5_dataset import DataAccessPurpose, DataAccessScope

_COMPONENTS = (
    "observed_f",
    "input_f",
    "reference_f",
    "observed_state",
    "input_state",
    "reference_state",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _provenance(namespace: str) -> PortableReferenceSequenceProvenance:
    requested_dt = 1.0 / 300.0
    execution_dt = float(np.float32(requested_dt))
    return PortableReferenceSequenceProvenance(
        dataset_index_uri=f"artifact://reference-sequence/{namespace}/index.json",
        dataset_index_sha256=_digest(f"dataset-index:{namespace}"),
        asset_id=f"asset-{namespace}",
        asset_source_sha256=_digest(f"asset-source:{namespace}"),
        producer_topology_sha256=_digest(f"producer-topology:{namespace}"),
        producer_operator_sha256=_digest(f"producer-operator:{namespace}"),
        producer_material_sha256=_digest(f"producer-material:{namespace}"),
        sequence_id=f"sequence-{namespace}",
        producer_manifest_uri=f"artifact://reference-sequence/{namespace}/manifest.json",
        producer_manifest_sha256=_digest(f"producer-manifest:{namespace}"),
        static_bundle_uri=f"artifact://reference-sequence/{namespace}/static.npz",
        static_bundle_sha256=_digest(f"static-bundle:{namespace}"),
        sequence_bundle_uri=f"artifact://reference-sequence/{namespace}/sequence.npz",
        sequence_bundle_sha256=_digest(f"sequence-bundle:{namespace}"),
        evidence_uri=f"artifact://reference-sequence/{namespace}/evidence.json",
        evidence_sha256=_digest(f"evidence:{namespace}"),
        protocol_sha256=_digest(f"protocol:{namespace}"),
        initial_position_sha256=_digest(f"initial-position:{namespace}"),
        initial_velocity_field_sha256=_digest(f"initial-velocity:{namespace}"),
        final_position_sha256=_digest(f"final-position:{namespace}"),
        final_velocity_field_sha256=_digest(f"final-velocity:{namespace}"),
        accepted_reference_state_sha256=tuple(
            _digest(f"accepted-reference:{namespace}:{ordinal}") for ordinal in range(2)
        ),
        deformation_seed=101,
        velocity_seed=211,
        source_transition_count=2,
        requested_dt_seconds=requested_dt,
        dt_seconds=execution_dt,
        execution_dt_float32_bits="0x3b5a740e",
    )


def _sample(
    namespace: str,
    ordinal: int,
    *,
    topology_sha256: str | None = None,
    operator_geometry_sha256: str | None = None,
    operator_volume_sha256: str | None = None,
    material_sha256: str | None = None,
    common_objective_sha256: str | None = None,
) -> PortableDatasetSampleRecord:
    topology = topology_sha256 or _digest(f"topology:{namespace}")
    geometry = operator_geometry_sha256 or _digest(f"operator:{namespace}")
    volume = operator_volume_sha256 or _digest(f"volume:{namespace}")
    material = material_sha256 or _digest(f"material:{namespace}")
    pin_signature = _digest(f"pins:{namespace}")
    source_transition = PortableReferenceSourceTransitionIdentity(
        reference_sequence_index_sha256=_digest(f"dataset-index:{namespace}"),
        asset_id=f"asset-{namespace}",
        asset_source_sha256=_digest(f"asset-source:{namespace}"),
        sequence_id=f"sequence-{namespace}",
        step_id=ordinal,
        static_npz_sha256=_digest(f"static-bundle:{namespace}"),
        sequence_npz_sha256=_digest(f"sequence-bundle:{namespace}"),
        protocol_sha256=_digest(f"protocol:{namespace}"),
        producer_topology_sha256=_digest(f"producer-topology:{namespace}"),
        producer_operator_sha256=_digest(f"producer-operator:{namespace}"),
        producer_material_sha256=_digest(f"producer-material:{namespace}"),
        accepted_reference_state_sha256=_digest(f"accepted-reference:{namespace}:{ordinal}"),
        portable_topology_sha256=topology,
        operator_geometry_policy=OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
        operator_geometry_sha256=geometry,
        operator_volume_policy=OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT,
        operator_volume_sha256=volume,
        portable_material_sha256=material,
        portable_pin_signature_sha256=pin_signature,
    )
    identities = {
        name: PortableNumericContentIdentity(
            identifier=source_transition.numeric_identifier(name),
            sha256=_digest(f"bytes:{namespace}:{ordinal}:{name}"),
        )
        for name in _COMPONENTS
    }
    return PortableDatasetSampleRecord(
        sample_id=f"{namespace}:sample:{ordinal}",
        ordinal=ordinal,
        topology_sha256=topology,
        operator_geometry_policy=OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
        operator_geometry_sha256=geometry,
        operator_volume_policy=OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT,
        operator_volume_sha256=volume,
        material_sha256=material,
        pin_signature_sha256=pin_signature,
        dt_seconds=float(np.float32(1.0 / 300.0)),
        physical_step_sha256=_digest(f"physical:{namespace}:{ordinal}"),
        physical_integration_policy="solver-vbd-staged-float32-v1",
        source_integration_evidence_sha256=_digest(f"evidence:{namespace}:{ordinal}"),
        source_transition=source_transition,
        common_objective_sha256=common_objective_sha256 or _digest(f"objective:{namespace}:{ordinal}"),
        **identities,
    )


def _trajectory(namespace: str) -> PortableDatasetTrajectoryRecord:
    provenance = _provenance(namespace)
    topology = _digest(f"topology:{namespace}")
    geometry = _digest(f"operator:{namespace}")
    volume = _digest(f"volume:{namespace}")
    material = _digest(f"material:{namespace}")
    samples = tuple(
        _sample(
            namespace,
            ordinal,
            topology_sha256=topology,
            operator_geometry_sha256=geometry,
            operator_volume_sha256=volume,
            material_sha256=material,
        )
        for ordinal in range(2)
    )
    return PortableDatasetTrajectoryRecord(
        trajectory_id=f"trajectory:{namespace}",
        scene_family=f"scene:{namespace}",
        load_program_id=f"load:{namespace}",
        load_program_sha256=_digest(f"load:{namespace}"),
        source_chain_sha256=_digest(f"chain:{namespace}"),
        topology_sha256=topology,
        operator_geometry_policy=OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
        operator_geometry_sha256=geometry,
        operator_volume_policy=OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT,
        operator_volume_sha256=volume,
        material_sha256=material,
        provenance=provenance,
        source_transition_count=2,
        samples=samples,
    )


def _rebind_sample_source(
    sample: PortableDatasetSampleRecord,
    *,
    source_changes: dict[str, object],
    sample_changes: dict[str, object] | None = None,
) -> PortableDatasetSampleRecord:
    source = dataclasses.replace(sample.source_transition, **source_changes)
    numeric = {
        name: dataclasses.replace(identity, identifier=source.numeric_identifier(name))
        for name, identity in sample.numeric_content
    }
    return dataclasses.replace(
        sample,
        source_transition=source,
        **numeric,
        **({} if sample_changes is None else sample_changes),
    )


def _relabel_trajectory_asset(
    trajectory: PortableDatasetTrajectoryRecord,
    *,
    asset_id: str | None = None,
    asset_source_sha256: str | None = None,
) -> PortableDatasetTrajectoryRecord:
    provenance = dataclasses.replace(
        trajectory.provenance,
        asset_id=asset_id or trajectory.provenance.asset_id,
        asset_source_sha256=asset_source_sha256 or trajectory.provenance.asset_source_sha256,
    )
    samples = [
        _rebind_sample_source(
            sample,
            source_changes={
                "asset_id": provenance.asset_id,
                "asset_source_sha256": provenance.asset_source_sha256,
            },
        )
        for sample in trajectory.samples
    ]
    return dataclasses.replace(trajectory, provenance=provenance, samples=tuple(samples))


class TestPortableDatasetRecords(unittest.TestCase):
    def test_records_have_fresh_contracts_and_strict_round_trip(self) -> None:
        sample = _sample("roundtrip", 0)
        trajectory = _trajectory("roundtrip")
        manifest = PortableDatasetSplitManifest(
            train=(trajectory,),
            validation=(),
            confirmation=(),
        )

        self.assertEqual(PORTABLE_DATASET_SCHEMA_VERSION, 1)
        self.assertEqual(sample.as_dict()["contract"], "pss-portable-volume-dataset-sample-v1")
        self.assertEqual(trajectory.as_dict()["contract"], "pss-portable-volume-dataset-trajectory-v1")
        self.assertEqual(manifest.as_dict()["contract"], "pss-portable-volume-dataset-split-v1")
        self.assertEqual(PortableDatasetSampleRecord.from_dict(sample.as_dict()), sample)
        self.assertEqual(
            PortableReferenceSourceTransitionIdentity.from_dict(sample.source_transition.as_dict()),
            sample.source_transition,
        )
        self.assertEqual(PortableDatasetTrajectoryRecord.from_dict(trajectory.as_dict()), trajectory)
        self.assertEqual(PortableDatasetSplitManifest.from_dict(manifest.as_dict()), manifest)
        self.assertEqual(PortableDatasetSplitManifest.from_json(manifest.to_json()), manifest)
        self.assertEqual(sample.dt_float64_bits, "0x3f6b4e81c0000000")
        self.assertEqual(trajectory.provenance.execution_dt_float32_bits, "0x3b5a740e")

        for payload in (sample.as_dict(), trajectory.as_dict(), manifest.as_dict()):
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            self.assertNotIn("pss-v5-dataset", encoded)

    def test_volume_and_objective_identity_are_transitive(self) -> None:
        sample = _sample("transitive", 0)
        changed_volume_sha256 = _digest("changed-volume")
        changed_volume = _rebind_sample_source(
            sample,
            source_changes={"operator_volume_sha256": changed_volume_sha256},
            sample_changes={"operator_volume_sha256": changed_volume_sha256},
        )
        changed_objective = dataclasses.replace(sample, common_objective_sha256=_digest("changed-objective"))
        self.assertNotEqual(changed_volume.static_layout_sha256, sample.static_layout_sha256)
        self.assertNotEqual(changed_volume.sample_sha256, sample.sample_sha256)
        self.assertNotEqual(changed_objective.sample_sha256, sample.sample_sha256)

        trajectory = _trajectory("transitive")
        changed_sample = dataclasses.replace(
            trajectory.samples[0],
            common_objective_sha256=_digest("trajectory-objective-change"),
        )
        changed_trajectory = dataclasses.replace(
            trajectory,
            samples=(changed_sample, trajectory.samples[1]),
        )
        self.assertNotEqual(changed_trajectory.trajectory_sha256, trajectory.trajectory_sha256)
        self.assertNotEqual(
            PortableDatasetSplitManifest(train=(changed_trajectory,), validation=(), confirmation=()).manifest_sha256,
            PortableDatasetSplitManifest(train=(trajectory,), validation=(), confirmation=()).manifest_sha256,
        )

    def test_closed_json_inventory_duplicate_keys_and_nonfinite_values_fail(self) -> None:
        manifest = PortableDatasetSplitManifest(
            train=(_trajectory("closed"),),
            validation=(),
            confirmation=(),
        )
        payload = manifest.as_dict()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            PortableDatasetSplitManifest.from_dict(payload)

        duplicate = '{"schema_version":1,"schema_version":1}'
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            PortableDatasetSplitManifest.from_json(duplicate)

        nonfinite = manifest.to_json().replace('"schema_version":1', '"schema_version":NaN')
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            PortableDatasetSplitManifest.from_json(nonfinite)

        wrong_bits = manifest.as_dict()
        wrong_bits["roles"]["train"][0]["provenance"]["dt_float64_bits"] = "0x0000000000000000"
        with self.assertRaisesRegex(ValueError, "exact canonical portable dataset"):
            PortableDatasetSplitManifest.from_dict(wrong_bits)

        nested_extra = manifest.as_dict()
        nested_extra["roles"]["train"][0]["samples"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            PortableDatasetSplitManifest.from_dict(nested_extra)

        source_extra = manifest.as_dict()
        source_extra["roles"]["train"][0]["samples"][0]["source_transition"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            PortableDatasetSplitManifest.from_dict(source_extra)

        source_preimage_tamper = manifest.as_dict()
        source_preimage_tamper["roles"]["train"][0]["samples"][0]["source_transition"]["source_artifacts"][
            "static_npz_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact canonical portable dataset"):
            PortableDatasetSplitManifest.from_dict(source_preimage_tamper)

    def test_unregistered_operator_policy_and_float_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "portable operator geometry policy"):
            dataclasses.replace(_sample("policy", 0), operator_geometry_policy="source-tet-poses-promoted")
        with self.assertRaisesRegex(ValueError, "portable operator volume policy"):
            dataclasses.replace(_sample("policy", 0), operator_volume_policy="invented")
        for dt in (0.0, math.inf, math.nan):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "dt_seconds"):
                dataclasses.replace(_sample("dt", 0), dt_seconds=dt)


class TestPortableDatasetSplitAndAccess(unittest.TestCase):
    def setUp(self) -> None:
        self.train = _trajectory("train")
        self.validation = _trajectory("validation")
        self.confirmation = _trajectory("confirmation")
        self.manifest = PortableDatasetSplitManifest(
            train=(self.train,),
            validation=(self.validation,),
            confirmation=(self.confirmation,),
        )

    def test_cross_role_source_and_payload_aliases_fail_closed(self) -> None:
        aliased_source = dataclasses.replace(
            _trajectory("source-alias"),
            source_chain_sha256=self.train.source_chain_sha256,
        )
        with self.assertRaisesRegex(ValueError, "trajectory source overlap"):
            PortableDatasetSplitManifest(
                train=(self.train,),
                validation=(aliased_source,),
                confirmation=(),
            )

        collision = dataclasses.replace(
            _trajectory("payload-alias").samples[0],
            common_objective_sha256=self.train.samples[0].common_objective_sha256,
        )
        payload_alias = dataclasses.replace(
            _trajectory("payload-alias"),
            samples=(collision, _trajectory("payload-alias").samples[1]),
        )
        with self.assertRaisesRegex(ValueError, "sample payload SHA-256 overlap"):
            PortableDatasetSplitManifest(
                train=(self.train,),
                validation=(payload_alias,),
                confirmation=(),
            )

    def test_asset_id_and_asset_source_leakage_are_independent(self) -> None:
        cases = {
            "same-id-same-source": (
                _relabel_trajectory_asset(
                    self.validation,
                    asset_id=self.train.provenance.asset_id,
                    asset_source_sha256=self.train.provenance.asset_source_sha256,
                ),
                "reference asset_id overlap",
            ),
            "same-id-different-source": (
                _relabel_trajectory_asset(
                    self.validation,
                    asset_id=self.train.provenance.asset_id,
                ),
                "asset_id.*conflicting asset source",
            ),
            "different-id-same-source": (
                _relabel_trajectory_asset(
                    self.validation,
                    asset_source_sha256=self.train.provenance.asset_source_sha256,
                ),
                "reference asset source overlap",
            ),
        }
        for label, (validation, error) in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, error):
                PortableDatasetSplitManifest(
                    train=(self.train,),
                    validation=(validation,),
                    confirmation=(),
                )

    def test_one_asset_id_cannot_map_to_conflicting_sources_within_role(self) -> None:
        conflicting = _relabel_trajectory_asset(
            _trajectory("same-role-conflict"),
            asset_id=self.train.provenance.asset_id,
        )
        with self.assertRaisesRegex(ValueError, "asset_id.*source"):
            PortableDatasetSplitManifest(
                train=(self.train, conflicting),
                validation=(),
                confirmation=(),
            )

    def test_one_asset_id_may_share_one_source_within_role(self) -> None:
        same_asset = _relabel_trajectory_asset(
            _trajectory("same-role-sequence"),
            asset_id=self.train.provenance.asset_id,
            asset_source_sha256=self.train.provenance.asset_source_sha256,
        )
        manifest = PortableDatasetSplitManifest(
            train=(self.train, same_asset),
            validation=(),
            confirmation=(),
        )
        self.assertEqual(len(manifest.train), 2)

    def test_confirmation_ledger_is_branch_local_evidence_and_fail_closed(self) -> None:
        ledger = PortableDatasetAccessLedger(self.manifest)
        with self.assertRaisesRegex(ValueError, "confirmation payload access is forbidden"):
            ledger.record_access(
                self.confirmation.trajectory_id,
                purpose=DataAccessPurpose.TRAINING,
                scope=DataAccessScope.PAYLOAD,
                payload_names=("reference_state",),
            )
        metadata = ledger.record_access(
            self.confirmation.trajectory_id,
            purpose=DataAccessPurpose.TRAINING,
            scope=DataAccessScope.METADATA,
        )
        released = metadata.record_access(
            self.confirmation.trajectory_id,
            purpose=DataAccessPurpose.CONFIRMATION_EVALUATION,
            scope=DataAccessScope.PAYLOAD,
            payload_names=("common_objective", "reference_state"),
        )
        self.assertTrue(released.confirmation_payload_released)
        with self.assertRaisesRegex(ValueError, "cannot resume after confirmation release"):
            released.record_access(
                self.train.trajectory_id,
                purpose=DataAccessPurpose.TRAINING,
                scope=DataAccessScope.PAYLOAD,
                payload_names=("reference_state",),
            )
        self.assertEqual(
            PortableDatasetAccessLedger.from_dict(released.as_dict(), manifest=self.manifest),
            released,
        )
        self.assertEqual(
            PortableDatasetAccessLedger.from_json(released.to_json(), manifest=self.manifest),
            released,
        )
        self.assertEqual(
            released.as_dict()["claim_scope"],
            "branch-local-evidence-not-global-access-control",
        )

    def test_portable_schedule_is_deterministic_and_binds_volume(self) -> None:
        first = build_portable_sampling_schedule(
            self.manifest,
            steps=1,
            batch_size=3,
            seed=20260821,
        )
        repeated = build_portable_sampling_schedule(
            self.manifest,
            steps=1,
            batch_size=3,
            seed=20260821,
        )
        self.assertEqual(first, repeated)
        self.assertEqual(first.as_dict()["contract"], "pss-portable-volume-dataset-sampling-v1")
        batch = first.batches[0]
        self.assertEqual(batch.operator_volume_sha256, self.train.operator_volume_sha256)
        self.assertTrue(all(sample.operator_volume_sha256 == batch.operator_volume_sha256 for sample in batch.samples))
        self.assertEqual(type(first).from_dict(first.as_dict(), manifest=self.manifest), first)
        self.assertEqual(type(first).from_json(first.to_json(), manifest=self.manifest), first)

    def test_v5_checkpoint_and_trainer_consumers_reject_successor_contracts(self) -> None:
        schedule = build_portable_sampling_schedule(
            self.manifest,
            steps=1,
            batch_size=1,
            seed=7,
        )
        with self.assertRaisesRegex(ValueError, "canonical SplitManifest"):
            checkpoint_contract._verify_split_manifest(self.manifest)
        with self.assertRaisesRegex(ValueError, "canonical SamplingSchedule"):
            checkpoint_contract._verify_sampling_schedule(
                schedule,
                manifest_sha256=self.manifest.manifest_sha256,
                expected_steps=schedule.steps,
            )
        with self.assertRaisesRegex(TypeError, "canonical SamplingSchedule"):
            trainer_contract._canonical_schedule(schedule, None)


if __name__ == "__main__":
    unittest.main()
