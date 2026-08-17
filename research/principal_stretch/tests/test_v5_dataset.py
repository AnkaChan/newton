# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Focused synthetic tests for the v5 dataset foundation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..v5_dataset import (
    DataAccessLedger,
    DataAccessPurpose,
    DataAccessScope,
    DatasetRole,
    NumericContentIdentity,
    SplitManifest,
    TrajectoryProvenance,
    TrajectoryRecord,
    TrajectorySampleRecord,
    build_sampling_schedule,
    canonical_topology_sha256,
    verify_file_sha256,
    verify_trajectory_topology,
)

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


def _identity(label: str) -> NumericContentIdentity:
    return NumericContentIdentity(identifier=label, sha256=_digest(f"bytes:{label}"))


def _provenance(
    namespace: str,
    *,
    dt_seconds: float = 1.0 / 300.0,
    density_kg_m3: float = 1000.0,
    initial_velocity_m_s: tuple[float, float, float] = (0.125, -0.25, 0.5),
    generation_seed: int = 17,
    artifact_bundle_sha256: str | None = None,
    artifact_source_sha256: str | None = None,
) -> TrajectoryProvenance:
    return TrajectoryProvenance(
        generation_spec_sha256=_digest(f"generation-spec:{namespace}"),
        history_manifest_sha256=_digest(f"history-manifest:{namespace}"),
        root_checkpoint_sha256=_digest(f"root-checkpoint:{namespace}"),
        final_checkpoint_sha256=_digest(f"final-checkpoint:{namespace}"),
        artifact_bundle_uri=f"artifact://dataset/{namespace}.npz",
        artifact_bundle_sha256=artifact_bundle_sha256 or _digest(f"artifact-bundle:{namespace}"),
        artifact_source_uri=f"source://history/{namespace}.json",
        artifact_source_sha256=artifact_source_sha256 or _digest(f"artifact-source:{namespace}"),
        static_bundle_sha256=_digest(f"static-bundle:{namespace}"),
        density_kg_m3=density_kg_m3,
        initial_velocity_m_s=initial_velocity_m_s,
        pin_schedule_sha256=_digest(f"pin-schedule:{namespace}"),
        event_inventory_sha256=_digest(f"event-inventory:{namespace}"),
        coordinate_start_sha256=_digest(f"coordinate-start:{namespace}"),
        coordinate_stop_sha256=_digest(f"coordinate-stop:{namespace}"),
        coordinate_range_sha256=_digest(f"coordinate-range:{namespace}"),
        dt_seconds=dt_seconds,
        generation_seed=generation_seed,
    )


def _sample(
    namespace: str,
    ordinal: int,
    *,
    topology_sha256: str | None = None,
    operator_geometry_sha256: str | None = None,
    material_sha256: str | None = None,
    pin_signature_sha256: str | None = None,
    physical_step_sha256: str | None = None,
    common_objective_sha256: str | None = None,
    dt_seconds: float = 1.0 / 300.0,
    **overrides: NumericContentIdentity,
) -> TrajectorySampleRecord:
    identities = {component: _identity(f"{namespace}:{ordinal}:{component}") for component in _COMPONENTS}
    identities.update(overrides)
    return TrajectorySampleRecord(
        sample_id=f"{namespace}:sample:{ordinal}",
        ordinal=ordinal,
        topology_sha256=topology_sha256 or _digest(f"topology:{namespace}"),
        operator_geometry_sha256=operator_geometry_sha256 or _digest(f"operator:{namespace}"),
        material_sha256=material_sha256 or _digest(f"material:{namespace}"),
        pin_signature_sha256=pin_signature_sha256 or _digest(f"pins:{namespace}:default"),
        dt_seconds=dt_seconds,
        physical_step_sha256=physical_step_sha256 or _digest(f"physical-step:{namespace}:{ordinal}"),
        common_objective_sha256=common_objective_sha256 or _digest(f"common-objective:{namespace}:{ordinal}"),
        **identities,
    )


def _trajectory(
    trajectory_id: str,
    *,
    sample_count: int = 2,
    load_program_id: str | None = None,
    topology: str | None = None,
    operator_geometry: str | None = None,
    material: str | None = None,
    samples: tuple[TrajectorySampleRecord, ...] | None = None,
    provenance: TrajectoryProvenance | None = None,
) -> TrajectoryRecord:
    trajectory_provenance = provenance or _provenance(trajectory_id)
    trajectory_topology = topology or _digest(f"topology:{trajectory_id}")
    trajectory_operator = operator_geometry or _digest(f"operator:{trajectory_id}")
    trajectory_material = material or _digest(f"material:{trajectory_id}")
    trajectory_samples = samples
    if trajectory_samples is None:
        trajectory_samples = tuple(
            _sample(
                trajectory_id,
                ordinal,
                topology_sha256=trajectory_topology,
                operator_geometry_sha256=trajectory_operator,
                material_sha256=trajectory_material,
                dt_seconds=trajectory_provenance.dt_seconds,
            )
            for ordinal in range(sample_count)
        )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        scene_family=f"scene:{trajectory_id}",
        load_program_id=load_program_id or f"load:{trajectory_id}",
        load_program_sha256=_digest(f"load-program:{load_program_id or trajectory_id}"),
        source_chain_sha256=_digest(f"chain:{trajectory_id}"),
        topology_sha256=trajectory_topology,
        operator_geometry_sha256=trajectory_operator,
        material_sha256=trajectory_material,
        provenance=trajectory_provenance,
        source_transition_count=len(trajectory_samples),
        samples=trajectory_samples,
    )


class TestTrajectoryProvenance(unittest.TestCase):
    def test_exact_float64_bits_and_id_hashes_are_bound(self):
        provenance = _provenance("exact-bits")
        trajectory = _trajectory("canonical-id", provenance=provenance)

        expected_dt_bits = f"0x{struct.unpack('<Q', struct.pack('<d', 1.0 / 300.0))[0]:016x}"
        expected_density_bits = f"0x{struct.unpack('<Q', struct.pack('<d', 1000.0))[0]:016x}"
        self.assertEqual(provenance.dt_float64_bits, expected_dt_bits)
        self.assertEqual(provenance.density_float64_bits, expected_density_bits)
        self.assertEqual(
            trajectory.trajectory_id_sha256,
            hashlib.sha256(b"canonical-id").hexdigest(),
        )
        self.assertEqual(trajectory.as_dict()["provenance"], provenance.as_dict())

        adjacent = dataclasses.replace(
            provenance,
            dt_seconds=math.nextafter(provenance.dt_seconds, math.inf),
        )
        self.assertNotEqual(adjacent.dt_float64_bits, provenance.dt_float64_bits)
        self.assertNotEqual(adjacent.provenance_sha256, provenance.provenance_sha256)
        with self.assertRaisesRegex(ValueError, "sample time step disagrees with trajectory provenance"):
            dataclasses.replace(trajectory, provenance=adjacent)
        self.assertNotEqual(
            _trajectory("canonical-id", provenance=adjacent).trajectory_sha256,
            trajectory.trajectory_sha256,
        )

    def test_invalid_dt_seed_velocity_and_uri_fail_closed(self):
        for dt_seconds in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(dt_seconds=dt_seconds):
                with self.assertRaisesRegex(ValueError, "dt_seconds must be a positive finite float64"):
                    _provenance("invalid-dt", dt_seconds=dt_seconds)
        for density in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(density=density):
                with self.assertRaisesRegex(ValueError, "density_kg_m3 must be a positive finite float64"):
                    _provenance("invalid-density", density_kg_m3=density)
        for seed in (-1, True):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "generation_seed must be a non-negative integer"):
                    _provenance("invalid-seed", generation_seed=seed)
        with self.assertRaisesRegex(ValueError, r"initial_velocity_m_s\[1\] must be a finite float64"):
            _provenance("invalid-velocity", initial_velocity_m_s=(0.0, math.nan, 0.0))
        with self.assertRaisesRegex(ValueError, "must be an absolute canonical URI"):
            dataclasses.replace(_provenance("invalid-uri"), artifact_bundle_uri="relative/bundle.npz")

    def test_tampered_provenance_is_rejected_by_trajectory_authentication(self):
        provenance = _provenance("tampered")
        object.__setattr__(provenance, "dt_seconds", provenance.dt_seconds * 2.0)
        with self.assertRaisesRegex(ValueError, "trajectory provenance changed after authentication"):
            _trajectory("tampered-trajectory", provenance=provenance)

    def test_streaming_artifact_verifier_checks_bundle_and_source(self):
        bundle_bytes = b"durable trajectory bundle\x00with binary payload"
        source_bytes = b'{"source":"authenticated history"}\n'
        bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        provenance = _provenance(
            "durable",
            artifact_bundle_sha256=bundle_sha256,
            artifact_source_sha256=source_sha256,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "bundle.bin"
            source_path = root / "source.json"
            bundle_path.write_bytes(bundle_bytes)
            source_path.write_bytes(source_bytes)

            self.assertEqual(verify_file_sha256(bundle_path, bundle_sha256, chunk_size=3), bundle_sha256)
            self.assertEqual(provenance.verify_artifact_bundle(bundle_path, chunk_size=5), bundle_sha256)
            self.assertEqual(provenance.verify_artifact_source(source_path, chunk_size=7), source_sha256)

            bundle_path.write_bytes(bundle_bytes + b"tampered")
            with self.assertRaisesRegex(ValueError, "durable artifact SHA-256 mismatch"):
                provenance.verify_artifact_bundle(bundle_path, chunk_size=2)
            with self.assertRaisesRegex(ValueError, "chunk_size must be a positive integer"):
                verify_file_sha256(source_path, source_sha256, chunk_size=0)


class TestCanonicalDatasetRecords(unittest.TestCase):
    def test_topology_identity_uses_the_runtime_rest_and_ordered_tet_algorithm(self):
        rest = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
        topology = canonical_topology_sha256(rest, tets)
        trajectory = _trajectory("canonical-topology", topology=topology)

        self.assertEqual(verify_trajectory_topology(trajectory, rest, tets), topology)
        changed = rest.copy()
        changed[1, 0] = np.nextafter(changed[1, 0], 2.0)
        with self.assertRaisesRegex(ValueError, "materialized rest positions/connectivity"):
            verify_trajectory_topology(trajectory, changed, tets)

    def test_records_and_manifest_canonicalize_order_and_are_immutable(self):
        reverse_samples = tuple(_sample("second", ordinal) for ordinal in (2, 0, 1))
        second = _trajectory("second", samples=reverse_samples)
        first = _trajectory("first")

        manifest = SplitManifest(
            train=(second, first),
            validation=(),
            confirmation=(),
        )
        repeated = SplitManifest(
            train=(first, second),
            validation=(),
            confirmation=(),
        )

        self.assertEqual([record.trajectory_id for record in manifest.train], ["first", "second"])
        self.assertEqual([sample.ordinal for sample in second.samples], [0, 1, 2])
        self.assertEqual(manifest.manifest_sha256, repeated.manifest_sha256)
        self.assertEqual(manifest.as_dict(), repeated.as_dict())
        self.assertEqual(len(manifest.manifest_sha256), 64)
        self.assertEqual(first.provenance.provenance_sha256, first.as_dict()["provenance"]["provenance_sha256"])
        json.dumps(manifest.as_dict(), sort_keys=True, allow_nan=False)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.train = ()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            second.samples = ()

    def test_rejects_trajectory_and_load_program_overlap_across_roles(self):
        train = _trajectory("train")
        with self.assertRaisesRegex(ValueError, "trajectory overlap across roles"):
            SplitManifest(train=(train,), validation=(train,), confirmation=())

        source_alias = dataclasses.replace(
            _trajectory("source-alias"),
            source_chain_sha256=train.source_chain_sha256,
        )
        with self.assertRaisesRegex(ValueError, "trajectory source overlap across roles"):
            SplitManifest(train=(train,), validation=(source_alias,), confirmation=())

        validation = _trajectory("validation", load_program_id=train.load_program_id)
        with self.assertRaisesRegex(ValueError, "load-program overlap across roles"):
            SplitManifest(train=(train,), validation=(validation,), confirmation=())

        load_alias = dataclasses.replace(
            _trajectory("load-alias"),
            load_program_sha256=train.load_program_sha256,
        )
        with self.assertRaisesRegex(ValueError, "load-program overlap across roles"):
            SplitManifest(train=(train,), validation=(load_alias,), confirmation=())

    def test_rejects_every_numeric_content_hash_overlap_across_roles(self):
        train = _trajectory("train")
        train_sample = train.samples[0]
        for component in _COMPONENTS:
            with self.subTest(component=component):
                colliding_hash = getattr(train_sample, component).sha256
                replacement = NumericContentIdentity(
                    identifier=f"validation:distinct-id:{component}",
                    sha256=colliding_hash,
                )
                validation_sample = _sample("validation", 0, **{component: replacement})
                validation = _trajectory("validation", samples=(validation_sample,))
                with self.assertRaisesRegex(ValueError, "sample payload SHA-256 overlap across roles"):
                    SplitManifest(train=(train,), validation=(validation,), confirmation=())

        for component in ("physical_step_sha256", "common_objective_sha256"):
            with self.subTest(component=component):
                validation_sample = _sample(
                    "validation",
                    0,
                    **{component: getattr(train_sample, component)},
                )
                validation = _trajectory("validation", samples=(validation_sample,))
                with self.assertRaisesRegex(ValueError, "sample payload SHA-256 overlap across roles"):
                    SplitManifest(train=(train,), validation=(validation,), confirmation=())

    def test_objective_context_declarations_are_validated_and_bound_to_sample(self):
        sample = _sample("objective-context", 0)

        self.assertEqual(sample.as_dict()["physical_step_sha256"], sample.physical_step_sha256)
        self.assertEqual(sample.as_dict()["common_objective_sha256"], sample.common_objective_sha256)
        self.assertNotEqual(
            dataclasses.replace(sample, physical_step_sha256=_digest("changed-physical-step")).sample_sha256,
            sample.sample_sha256,
        )
        self.assertNotEqual(
            dataclasses.replace(sample, common_objective_sha256=_digest("changed-common-objective")).sample_sha256,
            sample.sample_sha256,
        )
        for component in ("physical_step_sha256", "common_objective_sha256"):
            with self.subTest(component=component):
                with self.assertRaisesRegex(ValueError, f"sample {component} must be a lowercase SHA-256 digest"):
                    dataclasses.replace(sample, **{component: "not-a-sha256"})

    def test_rejects_numeric_identifier_alias_even_when_bytes_differ(self):
        train = _trajectory("train")
        shared_identifier = train.samples[0].observed_f.identifier
        replacement = NumericContentIdentity(
            identifier=shared_identifier,
            sha256=_digest("different-numeric-bytes"),
        )
        validation = _trajectory("validation", samples=(_sample("validation", 0, observed_f=replacement),))
        with self.assertRaisesRegex(ValueError, "numeric content identifier overlap across roles"):
            SplitManifest(train=(train,), validation=(validation,), confirmation=())

    def test_reports_topology_operator_and_material_overlap_unless_configured_to_reject(self):
        topology = _digest("shared-topology")
        operator = _digest("shared-operator")
        material = _digest("shared-material")
        train = _trajectory("train", topology=topology, operator_geometry=operator, material=material)
        validation = _trajectory("validation", topology=topology, operator_geometry=operator, material=material)

        manifest = SplitManifest(train=(train,), validation=(validation,), confirmation=())
        self.assertTrue(manifest.overlap_report.has_topology_overlap)
        self.assertTrue(manifest.overlap_report.has_operator_geometry_overlap)
        self.assertTrue(manifest.overlap_report.has_material_overlap)
        self.assertEqual(len(manifest.overlap_report.role_pairs), 1)
        overlap = manifest.overlap_report.role_pairs[0]
        self.assertEqual((overlap.first, overlap.second), (DatasetRole.TRAIN, DatasetRole.VALIDATION))
        self.assertEqual(overlap.topology_sha256, (topology,))
        self.assertEqual(overlap.operator_geometry_sha256, (operator,))
        self.assertEqual(overlap.material_sha256, (material,))

        with self.assertRaisesRegex(ValueError, "topology overlap across roles"):
            SplitManifest(
                train=(train,),
                validation=(validation,),
                confirmation=(),
                reject_topology_overlap=True,
            )
        with self.assertRaisesRegex(ValueError, "material overlap across roles"):
            SplitManifest(
                train=(train,),
                validation=(validation,),
                confirmation=(),
                reject_material_overlap=True,
            )

    def test_same_topology_different_operator_geometry_separates_static_layouts(self):
        topology = _digest("one-topology")
        first = _trajectory("first-operator", topology=topology, operator_geometry=_digest("operator-a"))
        second = _trajectory("second-operator", topology=topology, operator_geometry=_digest("operator-b"))

        self.assertNotEqual(first.samples[0].static_layout_sha256, second.samples[0].static_layout_sha256)
        manifest = SplitManifest(train=(first, second), validation=(), confirmation=())
        schedule = build_sampling_schedule(manifest, steps=2, batch_size=2, seed=11)
        for batch in schedule.batches:
            self.assertEqual(
                {reference.operator_geometry_sha256 for reference in batch.samples},
                {batch.operator_geometry_sha256},
            )

    def test_complete_trajectory_rejects_slices_and_subranges_are_explicit(self):
        base = _trajectory("selection", sample_count=2)
        with self.assertRaisesRegex(ValueError, "complete trajectory selection must contain every source transition"):
            dataclasses.replace(base, source_transition_count=10)

        noncontiguous_samples = (_sample("selection", 7), _sample("selection", 42))
        with self.assertRaisesRegex(ValueError, "must form one contiguous selection"):
            dataclasses.replace(
                base,
                source_transition_count=43,
                samples=noncontiguous_samples,
                selection_contract="authenticated-contiguous-subrange-v1",
                selection_provenance_sha256=_digest("noncontiguous-selection-evidence"),
            )

        subrange = dataclasses.replace(
            base,
            source_transition_count=10,
            samples=(_sample("selection", 7), _sample("selection", 8)),
            selection_contract="authenticated-contiguous-subrange-v1",
            selection_provenance_sha256=_digest("subrange-selection-evidence"),
        )
        self.assertEqual([sample.ordinal for sample in subrange.samples], [7, 8])
        self.assertEqual(subrange.source_transition_count, 10)


class TestDataAccessLedger(unittest.TestCase):
    def setUp(self):
        self.manifest = SplitManifest(
            train=(_trajectory("train"),),
            validation=(_trajectory("validation"),),
            confirmation=(_trajectory("confirmation"),),
            consumed_regression=(_trajectory("consumed"),),
        )

    def test_confirmation_metadata_is_visible_but_training_and_selection_payloads_are_sealed(self):
        empty = DataAccessLedger(self.manifest)
        metadata = empty.record_access(
            "confirmation",
            purpose=DataAccessPurpose.MODEL_SELECTION,
            scope=DataAccessScope.METADATA,
        )

        self.assertEqual(len(empty.accesses), 0)
        self.assertEqual(len(metadata.accesses), 1)
        self.assertNotEqual(empty.ledger_sha256, metadata.ledger_sha256)
        self.assertEqual(metadata.accesses[0].previous_sha256, self.manifest.manifest_sha256)
        for purpose in (DataAccessPurpose.TRAINING, DataAccessPurpose.MODEL_SELECTION):
            with self.subTest(purpose=purpose):
                with self.assertRaisesRegex(ValueError, "confirmation payload access is forbidden"):
                    metadata.record_access(
                        "confirmation",
                        purpose=purpose,
                        scope=DataAccessScope.PAYLOAD,
                        payload_names=("reference_f",),
                    )

    def test_valid_role_specific_payload_access_is_hash_chained(self):
        ledger = DataAccessLedger(self.manifest)
        ledger = ledger.record_access(
            "train",
            purpose="training",
            scope="payload",
            payload_names=("reference_state", "observed_f"),
        )
        ledger = ledger.record_access(
            "validation",
            purpose="model_selection",
            scope="payload",
            payload_names=("reference_state",),
        )
        ledger = ledger.record_access(
            "confirmation",
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("reference_state",),
        )

        self.assertEqual([access.sequence for access in ledger.accesses], [0, 1, 2])
        self.assertEqual(ledger.accesses[0].payload_names, ("observed_f", "reference_state"))
        self.assertEqual(ledger.accesses[1].previous_sha256, ledger.accesses[0].access_sha256)
        self.assertEqual(ledger.accesses[2].previous_sha256, ledger.accesses[1].access_sha256)
        self.assertEqual(len(ledger.ledger_sha256), 64)
        json.dumps(ledger.as_dict(), sort_keys=True, allow_nan=False)

    def test_consumed_regression_payload_is_not_a_training_source(self):
        with self.assertRaisesRegex(ValueError, "consumed-regression payload access is forbidden"):
            DataAccessLedger(self.manifest).record_access(
                "consumed",
                purpose="training",
                scope="payload",
                payload_names=("observed_f",),
            )

    def test_confirmation_release_blocks_later_selection_on_the_same_ledger_branch(self):
        ledger = DataAccessLedger(self.manifest).record_access(
            "confirmation",
            purpose="confirmation_evaluation",
            scope="payload",
            payload_names=("reference_state",),
        )
        self.assertTrue(ledger.confirmation_payload_released)

        with self.assertRaisesRegex(ValueError, "cannot resume after confirmation release on this ledger branch"):
            ledger.record_access(
                "validation",
                purpose="model_selection",
                scope="payload",
                payload_names=("reference_state",),
            )

    def test_payload_access_is_bound_to_manifest_component_identities(self):
        empty = DataAccessLedger(self.manifest)
        with self.assertRaisesRegex(ValueError, "unknown payload components"):
            empty.record_access(
                "train",
                purpose="training",
                scope="payload",
                payload_names=("not_a_payload",),
            )

        valid = empty.record_access(
            "train",
            purpose="training",
            scope="payload",
            payload_names=("observed_f",),
        )
        objective_access = empty.record_access(
            "train",
            purpose="training",
            scope="payload",
            payload_names=("physical_step", "common_objective"),
        )
        self.assertEqual(objective_access.accesses[0].payload_names, ("common_objective", "physical_step"))
        self.assertNotEqual(
            objective_access.accesses[0].payload_identity_sha256,
            valid.accesses[0].payload_identity_sha256,
        )
        tampered = dataclasses.replace(
            valid.accesses[0],
            payload_identity_sha256=_digest("forged-payload-selection"),
        )
        with self.assertRaisesRegex(ValueError, "payload identity disagrees"):
            DataAccessLedger(self.manifest, (tampered,))


class TestTrajectoryFirstSampling(unittest.TestCase):
    def setUp(self):
        topology_a = _digest("topology-a")
        topology_b = _digest("topology-b")
        self.records = (
            _trajectory("a-short", sample_count=1, topology=topology_a),
            _trajectory("a-long", sample_count=7, topology=topology_a),
            _trajectory("b-short", sample_count=2, topology=topology_b),
            _trajectory("b-long", sample_count=5, topology=topology_b),
        )
        self.manifest = SplitManifest(train=self.records, validation=(), confirmation=())

    def test_schedule_is_replayable_homogeneous_and_balanced_by_trajectory(self):
        schedule = build_sampling_schedule(
            self.manifest,
            steps=len(self.records),
            batch_size=4,
            seed=1701,
        )
        repeated = build_sampling_schedule(
            self.manifest,
            steps=len(self.records),
            batch_size=4,
            seed=1701,
        )

        self.assertEqual(schedule, repeated)
        self.assertEqual(schedule.schedule_sha256, repeated.schedule_sha256)
        self.assertEqual(dict(schedule.trajectory_exposure), {record.trajectory_id: 4 for record in self.records})
        for batch in schedule.batches:
            self.assertEqual({sample.topology_sha256 for sample in batch.samples}, {batch.topology_sha256})

        selected_by_trajectory: dict[str, list[str]] = {record.trajectory_id: [] for record in self.records}
        for batch in schedule.batches:
            for sample in batch.samples:
                selected_by_trajectory[sample.trajectory_id].append(sample.sample_id)
        self.assertEqual(len(selected_by_trajectory["a-short"]), len(selected_by_trajectory["a-long"]))
        for record in self.records:
            counts = {
                sample.sample_id: selected_by_trajectory[record.trajectory_id].count(sample.sample_id)
                for sample in record.samples
            }
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

        different_seed = build_sampling_schedule(
            self.manifest,
            steps=len(self.records),
            batch_size=4,
            seed=1702,
        )
        self.assertNotEqual(schedule.schedule_sha256, different_seed.schedule_sha256)
        json.dumps(schedule.as_dict(), sort_keys=True, allow_nan=False)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            schedule.batches = ()

    def test_empty_role_and_invalid_work_settings_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "cannot sample the empty validation role"):
            build_sampling_schedule(
                self.manifest,
                role=DatasetRole.VALIDATION,
                steps=1,
                batch_size=1,
                seed=0,
            )
        with self.assertRaisesRegex(ValueError, "steps must be a positive integer"):
            build_sampling_schedule(self.manifest, steps=0, batch_size=1, seed=0)
        with self.assertRaisesRegex(ValueError, "steps must be divisible by the trajectory count"):
            build_sampling_schedule(self.manifest, steps=3, batch_size=1, seed=0)
        with self.assertRaisesRegex(ValueError, "seed must be a non-negative integer"):
            build_sampling_schedule(self.manifest, steps=1, batch_size=1, seed=-1)

    def test_schedule_rejects_valid_sample_repetition_that_disagrees_with_seed_replay(self):
        schedule = build_sampling_schedule(
            self.manifest,
            steps=len(self.records),
            batch_size=4,
            seed=1701,
        )
        target_index = next(
            index for index, batch in enumerate(schedule.batches) if batch.samples[0].trajectory_id == "a-long"
        )
        target = schedule.batches[target_index]
        repeated = dataclasses.replace(target, samples=(target.samples[0],) * len(target.samples))
        forged_batches = (*schedule.batches[:target_index], repeated, *schedule.batches[target_index + 1 :])

        with self.assertRaisesRegex(ValueError, "do not match deterministic PCG64 replay"):
            dataclasses.replace(schedule, batches=forged_batches)

    def test_schedule_cannot_be_relabelled_or_forge_a_sample_identity(self):
        manifest = SplitManifest(
            train=(_trajectory("train"),),
            validation=(),
            confirmation=(_trajectory("confirmation"),),
        )
        confirmation = build_sampling_schedule(
            manifest,
            role=DatasetRole.CONFIRMATION,
            steps=1,
            batch_size=1,
            seed=3,
        )

        with self.assertRaisesRegex(ValueError, "does not belong to the declared manifest role"):
            dataclasses.replace(confirmation, role=DatasetRole.TRAIN)

        original_batch = confirmation.batches[0]
        forged_reference = dataclasses.replace(
            original_batch.samples[0],
            sample_sha256=_digest("forged-sample"),
        )
        forged_batch = dataclasses.replace(original_batch, samples=(forged_reference,))
        with self.assertRaisesRegex(ValueError, "sample identity disagrees"):
            dataclasses.replace(confirmation, batches=(forged_batch,))

        for field, message in (
            ("physical_step_sha256", "physical-step identity disagrees"),
            ("common_objective_sha256", "common-objective identity disagrees"),
        ):
            with self.subTest(field=field):
                forged_reference = dataclasses.replace(
                    original_batch.samples[0],
                    **{field: _digest(f"forged-{field}")},
                )
                forged_batch = dataclasses.replace(original_batch, samples=(forged_reference,))
                with self.assertRaisesRegex(ValueError, message):
                    dataclasses.replace(confirmation, batches=(forged_batch,))

    def test_moving_pin_layouts_are_never_mixed_within_a_batch(self):
        topology = _digest("moving-pin-topology")
        material = _digest("moving-pin-material")
        provenance = _provenance("moving-pin")
        fixed_pins = _digest("pins:fixed")
        released_pins = _digest("pins:released")
        samples = tuple(
            _sample(
                "moving-pin",
                ordinal,
                topology_sha256=topology,
                material_sha256=material,
                pin_signature_sha256=fixed_pins if ordinal < 2 else released_pins,
                dt_seconds=provenance.dt_seconds,
            )
            for ordinal in range(4)
        )
        trajectory = _trajectory(
            "moving-pin",
            topology=topology,
            material=material,
            samples=samples,
            provenance=provenance,
        )
        manifest = SplitManifest(train=(trajectory,), validation=(), confirmation=())
        with self.assertRaisesRegex(ValueError, "must complete every trajectory's static-layout cycle"):
            build_sampling_schedule(manifest, steps=1, batch_size=2, seed=91)
        schedule = build_sampling_schedule(manifest, steps=2, batch_size=2, seed=91)

        self.assertEqual({batch.pin_signature_sha256 for batch in schedule.batches}, {fixed_pins, released_pins})
        self.assertEqual(
            {batch.static_layout_sha256 for batch in schedule.batches},
            {sample.static_layout_sha256 for sample in samples},
        )
        self.assertEqual({count for _trajectory_id, _layout, count in schedule.static_layout_exposure}, {2})
        for batch in schedule.batches:
            self.assertEqual({sample.topology_sha256 for sample in batch.samples}, {batch.topology_sha256})
            self.assertEqual({sample.pin_signature_sha256 for sample in batch.samples}, {batch.pin_signature_sha256})
            self.assertEqual({sample.static_layout_sha256 for sample in batch.samples}, {batch.static_layout_sha256})

        original_batch = schedule.batches[0]
        forged_reference = dataclasses.replace(
            original_batch.samples[0],
            static_layout_sha256=_digest("forged-static-layout"),
        )
        with self.assertRaisesRegex(ValueError, "sampling batch mixes static layouts"):
            dataclasses.replace(
                original_batch,
                samples=(forged_reference, original_batch.samples[1]),
            )

    def test_shared_static_layout_still_requires_per_sample_physical_objectives(self):
        trajectory = _trajectory("distinct-transitions", sample_count=2)
        manifest = SplitManifest(train=(trajectory,), validation=(), confirmation=())

        schedule = build_sampling_schedule(manifest, steps=1, batch_size=2, seed=123)
        batch = schedule.batches[0]
        selected_records = {
            reference.sample_id: next(
                sample for sample in trajectory.samples if sample.sample_id == reference.sample_id
            )
            for reference in batch.samples
        }

        self.assertEqual(len({reference.static_layout_sha256 for reference in batch.samples}), 1)
        self.assertEqual(len(selected_records), 2)
        self.assertEqual(len({sample.input_state.sha256 for sample in selected_records.values()}), 2)
        self.assertEqual(len({sample.reference_state.sha256 for sample in selected_records.values()}), 2)
        self.assertEqual(len({sample.physical_step_sha256 for sample in selected_records.values()}), 2)
        self.assertEqual(len({sample.common_objective_sha256 for sample in selected_records.values()}), 2)
        for reference in batch.samples:
            selected = selected_records[reference.sample_id]
            self.assertEqual(reference.physical_step_sha256, selected.physical_step_sha256)
            self.assertEqual(reference.common_objective_sha256, selected.common_objective_sha256)
        self.assertEqual(
            batch.as_dict()["physical_objective_routing"],
            "per-sample-unbatched-physical-objective-v1",
        )
        self.assertEqual(
            schedule.as_dict()["physical_objective_routing"],
            "per-sample-unbatched-physical-objective-v1",
        )


if __name__ == "__main__":
    unittest.main()
