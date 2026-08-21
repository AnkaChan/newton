# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for lazy reference-sequence portable sample materialization."""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path

import torch

from ..iterative_solver import validate_physical_objective_integration
from ..portable_dataset import (
    PortableDatasetSampleRecord,
    PortableReferenceSourceTransitionIdentity,
)
from ..reference_sequence_dataset import ReferenceSequenceDataset, ReferenceTransitionKey
from ..reference_sequence_v5_bridge import (
    ReferencePortableMaterializedSample,
    ReferenceSequencePortableDatasetBridge,
)
from ..torch_solver import (
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME,
    OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT,
)
from ..train_pr_history_v5 import V5TrainingSample
from ..v5_dataset import DatasetRole, TrajectorySampleRecord
from .test_reference_sequence_dataset import _write_index, _write_sequence_record


class TestReferenceSequencePortableBridge(unittest.TestCase):
    def _dataset(self, root: Path) -> ReferenceSequenceDataset:
        records = [
            _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="alpha",
                sequence_id="sample-000",
            ),
            _write_sequence_record(
                root,
                role=DatasetRole.VALIDATION,
                asset_id="beta",
                sequence_id="sample-000",
                offset=3.0,
            ),
            _write_sequence_record(
                root,
                role=DatasetRole.CONFIRMATION,
                asset_id="gamma",
                sequence_id="sample-000",
                offset=6.0,
            ),
        ]
        return ReferenceSequenceDataset.load(_write_index(root, records))

    def test_materializes_successor_record_and_bound_runtime_without_v5_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            bridge = ReferenceSequencePortableDatasetBridge(dataset, device="cpu")
            key = ReferenceTransitionKey("alpha", "sample-000", 1)

            loaded = bridge.materialize(key)

            self.assertIs(type(loaded), ReferencePortableMaterializedSample)
            self.assertIs(type(loaded.sample_record), PortableDatasetSampleRecord)
            self.assertIsNot(type(loaded.sample_record), TrajectorySampleRecord)
            self.assertFalse(hasattr(loaded, "training_sample"))
            self.assertNotIsInstance(loaded, V5TrainingSample)
            loaded.validate_immutable()
            validate_physical_objective_integration(
                loaded.projection_state,
                loaded.common_objective,
                loaded.physical_step,
            )

            record = loaded.sample_record
            state = loaded.projection_state
            objective = loaded.common_objective
            self.assertEqual(record.operator_geometry_policy, OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PORTABLE_VOLUME)
            self.assertEqual(record.operator_volume_policy, OPERATOR_VOLUME_POLICY_HOST_FLOAT64_SCALAR_POSE_DETERMINANT)
            self.assertEqual(record.operator_geometry_sha256, state.operator_geometry_sha256)
            self.assertEqual(record.operator_volume_sha256, state.operator_volume_sha256)
            self.assertEqual(record.operator_geometry_sha256, objective.operator_geometry_sha256)
            self.assertEqual(record.operator_volume_sha256, objective.operator_volume_sha256)
            self.assertEqual(record.common_objective_sha256, objective.common_objective_sha256)
            self.assertEqual(loaded.key, ("reference-sequence:alpha:sample-000", "step-00000001"))

            with self.assertRaisesRegex(TypeError, "TrajectorySampleRecord"):
                V5TrainingSample(
                    trajectory_id=loaded.key[0],
                    sample_record=record,
                    physical_step=loaded.physical_step,
                    common_objective=objective,
                    projection_state=state,
                    producer_attested_reference_positions=loaded.producer_attested_reference_positions,
                    producer_attested_reference_deformation_gradient=(
                        loaded.producer_attested_reference_deformation_gradient
                    ),
                )

    def test_bridge_is_on_demand_and_never_retains_dynamic_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            bridge = ReferenceSequencePortableDatasetBridge(dataset, device="cpu")
            keys = bridge.sample_keys(DatasetRole.TRAIN, count=3, seed=19)
            values = bridge.iter_materialized(keys)
            self.assertEqual(bridge.cached_asset_count, 0)
            first = next(values)
            self.assertEqual(bridge.cached_asset_count, 1)
            self.assertIsNot(first, bridge.materialize(first.transition_key))
            self.assertFalse(any(hasattr(value, "training_sample") for value in (first, *tuple(values))))

    def test_cpu_rematerialization_has_identical_durable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            key = ReferenceTransitionKey("alpha", "sample-000", 1)
            first = ReferenceSequencePortableDatasetBridge(dataset, device="cpu").materialize(key)
            second = ReferenceSequencePortableDatasetBridge(dataset, device=torch.device("cpu")).materialize(key)
            self.assertEqual(first.identities, second.identities)
            self.assertEqual(first.source_transition_sha256, second.source_transition_sha256)
            self.assertEqual(first.sample_record, second.sample_record)
            self.assertEqual(
                first.common_objective.common_objective_sha256, second.common_objective.common_objective_sha256
            )

    def test_materialized_wrapper_rejects_relabelled_volume_and_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            loaded = ReferenceSequencePortableDatasetBridge(dataset, device="cpu").materialize(
                ReferenceTransitionKey("alpha", "sample-000", 1)
            )

            changed_source = dataclasses.replace(
                loaded.sample_record.source_transition,
                operator_volume_sha256="0" * 64,
            )
            changed_numeric = {
                name: dataclasses.replace(identity, identifier=changed_source.numeric_identifier(name))
                for name, identity in loaded.sample_record.numeric_content
            }
            changed_volume = dataclasses.replace(
                loaded.sample_record,
                operator_volume_sha256="0" * 64,
                source_transition=changed_source,
                **changed_numeric,
            )
            with self.assertRaisesRegex(ValueError, "source transition|asset/runtime identities"):
                dataclasses.replace(loaded, sample_record=changed_volume)

            changed_objective = dataclasses.replace(
                loaded.sample_record,
                common_objective_sha256="1" * 64,
            )
            with self.assertRaisesRegex(ValueError, "common-objective identity"):
                dataclasses.replace(loaded, sample_record=changed_objective)

            relabelled_identity = dataclasses.replace(
                loaded.sample_record.observed_f,
                identifier="reference-sequence-portable-volume-v1:" + "2" * 64,
            )
            with self.assertRaisesRegex(ValueError, "identifier differs"):
                dataclasses.replace(
                    loaded.sample_record,
                    observed_f=relabelled_identity,
                )

    def test_materialized_wrapper_rejects_coherent_source_identity_relabels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            loaded = ReferenceSequencePortableDatasetBridge(dataset, device="cpu").materialize(
                ReferenceTransitionKey("alpha", "sample-000", 1)
            )

            def record_for_source(source):
                numeric = {
                    name: dataclasses.replace(identity, identifier=source.numeric_identifier(name))
                    for name, identity in loaded.sample_record.numeric_content
                }
                return dataclasses.replace(
                    loaded.sample_record,
                    source_transition=source,
                    **numeric,
                )

            changed_index = "f" * 64
            changed_source = dataclasses.replace(
                loaded.source_transition,
                reference_sequence_index_sha256=changed_index,
            )
            numeric = {
                name: dataclasses.replace(
                    identity,
                    identifier=changed_source.numeric_identifier(name),
                )
                for name, identity in loaded.sample_record.numeric_content
            }
            with self.assertRaisesRegex(ValueError, "reference-sequence index"):
                dataclasses.replace(
                    loaded,
                    identities=dataclasses.replace(
                        loaded.identities,
                        reference_sequence_index_sha256=changed_index,
                    ),
                    sample_record=dataclasses.replace(
                        loaded.sample_record,
                        source_transition=changed_source,
                        **numeric,
                    ),
                )

            for field in (
                "asset_source_sha256",
                "static_npz_sha256",
                "producer_topology_sha256",
                "producer_operator_sha256",
                "producer_material_sha256",
            ):
                changed_source = dataclasses.replace(loaded.source_transition, **{field: "e" * 64})
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source transition"):
                    dataclasses.replace(
                        loaded,
                        identities=dataclasses.replace(loaded.identities, **{field: "e" * 64}),
                        sample_record=record_for_source(changed_source),
                    )

    def test_materialized_wrapper_rejects_coherent_transition_key_relabels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            loaded = ReferenceSequencePortableDatasetBridge(dataset, device="cpu").materialize(
                ReferenceTransitionKey("alpha", "sample-000", 1)
            )

            def record_for_source(source, *, sample_id="step-00000001", ordinal=1):
                numeric = {
                    name: dataclasses.replace(identity, identifier=source.numeric_identifier(name))
                    for name, identity in loaded.sample_record.numeric_content
                }
                return dataclasses.replace(
                    loaded.sample_record,
                    sample_id=sample_id,
                    ordinal=ordinal,
                    source_transition=source,
                    **numeric,
                )

            asset_source = dataclasses.replace(loaded.source_transition, asset_id="relabeled-alpha")
            with self.assertRaisesRegex(ValueError, "source transition"):
                dataclasses.replace(
                    loaded,
                    transition_key=ReferenceTransitionKey("relabeled-alpha", "sample-000", 1),
                    identities=dataclasses.replace(loaded.identities, asset_id="relabeled-alpha"),
                    sample_record=record_for_source(asset_source),
                )
            sequence_source = dataclasses.replace(loaded.source_transition, sequence_id="relabeled-sequence")
            with self.assertRaisesRegex(ValueError, "source transition"):
                dataclasses.replace(
                    loaded,
                    transition_key=ReferenceTransitionKey("alpha", "relabeled-sequence", 1),
                    sample_record=record_for_source(sequence_source),
                )
            step_source = dataclasses.replace(loaded.source_transition, step_id=2)
            with self.assertRaisesRegex(ValueError, "source transition"):
                dataclasses.replace(
                    loaded,
                    transition_key=ReferenceTransitionKey("alpha", "sample-000", 2),
                    sample_record=record_for_source(step_source, sample_id="step-00000002", ordinal=2),
                )

    def test_materialized_sample_carries_sealed_source_transition_preimage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            loaded = ReferenceSequencePortableDatasetBridge(dataset, device="cpu").materialize(
                ReferenceTransitionKey("alpha", "sample-000", 1)
            )
            source = loaded.source_transition
            self.assertIs(type(source), PortableReferenceSourceTransitionIdentity)
            self.assertEqual(PortableReferenceSourceTransitionIdentity.from_dict(source.as_dict()), source)
            self.assertEqual(loaded.sample_record.source_transition, source)
            self.assertEqual(
                loaded.physical_step.source_evidence.source_transition_sha256,
                source.source_transition_sha256,
            )
            self.assertEqual(
                loaded.sample_record.as_dict()["physical_step"]["source_transition_sha256"],
                source.source_transition_sha256,
            )
            self.assertEqual(source.static_npz_sha256, loaded.identities.static_npz_sha256)
            for name, identity in loaded.sample_record.numeric_content:
                self.assertEqual(identity.identifier, source.numeric_identifier(name))
            changed_static = dataclasses.replace(source, static_npz_sha256="f" * 64)
            self.assertNotEqual(changed_static.source_transition_sha256, source.source_transition_sha256)
            self.assertIn("source_transition", loaded.sample_record.as_dict())

    @unittest.skipUnless(os.environ.get("PSS_RUN_CUDA_PARITY") == "1", "opt-in CUDA parity")
    def test_cpu_cuda_durable_identities_match_exactly(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            key = ReferenceTransitionKey("alpha", "sample-000", 1)
            cpu = ReferenceSequencePortableDatasetBridge(dataset, device="cpu").materialize(key)
            cuda = ReferenceSequencePortableDatasetBridge(dataset, device="cuda").materialize(key)
            self.assertEqual(cpu.identities, cuda.identities)
            self.assertEqual(cpu.source_transition_sha256, cuda.source_transition_sha256)
            self.assertEqual(cpu.sample_record, cuda.sample_record)
            self.assertEqual(
                cpu.common_objective.common_objective_sha256, cuda.common_objective.common_objective_sha256
            )


if __name__ == "__main__":
    unittest.main()
