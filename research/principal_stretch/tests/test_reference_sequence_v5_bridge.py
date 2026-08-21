# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for lazy reference-sequence materialization into v5 samples."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ..graph_transformer import GraphTransformerConfig
from ..iterative_solver import (
    PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
    validate_physical_objective_integration,
)
from ..reference_sequence_dataset import ReferenceSequenceDataset, ReferenceTransitionKey
from ..reference_sequence_v5_bridge import ReferenceSequencePortableObjectiveBridge, ReferenceSequenceV5Bridge
from ..torch_solver import (
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
)
from ..train_pr_history_v5 import SharedTopologyPredictorBank, V5TrainingSample
from ..v5_dataset import DatasetRole
from .test_reference_sequence_dataset import _write_index, _write_sequence_record


class TestReferenceSequenceV5Bridge(unittest.TestCase):
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
                role=DatasetRole.TRAIN,
                asset_id="alpha",
                sequence_id="sample-001",
            ),
            _write_sequence_record(
                root,
                role=DatasetRole.TRAIN,
                asset_id="beta",
                sequence_id="sample-000",
                offset=3.0,
            ),
        ]
        return ReferenceSequenceDataset.load(_write_index(root, records))

    def test_materializes_authenticated_free_body_v5_sample_and_keeps_identity_domains_separate(self) -> None:
        """Preserve legacy v5 materialization and every established identity domain."""
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            bridge = ReferenceSequenceV5Bridge(dataset, device="cpu")
            key = ReferenceTransitionKey("alpha", "sample-000", 1)
            source = dataset.transition(key)
            loaded = bridge.materialize(key)
            sample = loaded.training_sample
            state = sample.projection_state
            objective = sample.common_objective

            self.assertIs(type(sample), V5TrainingSample)
            sample.validate_immutable()
            validate_physical_objective_integration(state, objective, sample.physical_step)
            self.assertEqual(loaded.key, sample.key)
            self.assertEqual(loaded.transition_key, key)
            self.assertEqual(sample.trajectory_id, "reference-sequence:alpha:sample-000")
            self.assertEqual(sample.sample_record.sample_id, "step-00000001")
            self.assertEqual(sample.sample_record.ordinal, 1)
            self.assertEqual(
                sample.physical_step.integration_policy,
                PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            )
            self.assertEqual(
                sample.physical_step.source_evidence.source_transition_sha256,
                loaded.source_transition_sha256,
            )

            self.assertEqual(state.operator_geometry_policy, OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED)
            self.assertEqual(state.translation_gauge_policy, TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS)
            self.assertEqual(state.pinned.numel(), 0)
            self.assertEqual(objective.pinned.numel(), 0)
            self.assertEqual(tuple(sample.physical_step.pinned_targets.shape), (0, 3))
            self.assertTrue(torch.equal(sample.physical_step.pin, torch.zeros_like(sample.physical_step.pin)))
            self.assertTrue(torch.equal(state.center_of_mass_weights, objective.mass / objective.mass.sum()))

            identities = loaded.identities
            self.assertEqual(identities.asset_id, key.asset_id)
            self.assertEqual(identities.asset_source_sha256, source.asset_source_sha256)
            self.assertEqual(identities.producer_topology_sha256, source.topology_sha256)
            self.assertEqual(identities.producer_operator_sha256, source.operator_sha256)
            self.assertEqual(identities.producer_material_sha256, source.material_sha256)
            self.assertEqual(identities.v5_topology_sha256, sample.sample_record.topology_sha256)
            self.assertEqual(
                identities.v5_operator_geometry_sha256,
                sample.sample_record.operator_geometry_sha256,
            )
            self.assertEqual(identities.v5_material_sha256, sample.sample_record.material_sha256)
            self.assertEqual(identities.v5_pin_signature_sha256, sample.sample_record.pin_signature_sha256)
            self.assertNotEqual(identities.producer_operator_sha256, identities.v5_operator_geometry_sha256)
            self.assertNotEqual(identities.producer_material_sha256, identities.v5_material_sha256)
            self.assertEqual(
                source.reference_state_float64_sha256,
                sample.producer_attested_reference_positions_sha256,
            )
            self.assertEqual(
                state.operator_geometry_sha256,
                "96dd502257504fba191b31ff3c6d2d01ce81ea8cd285ffeb0a54a5517d072947",
            )
            self.assertEqual(
                state.projection_state_sha256,
                "588ceb6d5219c40c6e30e97a61804c11b8b8496ee69288efcf4c8a1cef042906",
            )
            self.assertEqual(
                objective.common_objective_sha256,
                "63bce0d69bfe565d2ac2f5dc4913cca95483a83dabe657bd687be6aad09eecdf",
            )
            self.assertEqual(
                sample.sample_record.sample_sha256,
                "df38817d96b00bb40c036ac2de0928fb4087a98ff738730bd06aa34e61fefdbe",
            )

    def test_portable_bridge_stops_before_current_v5_record_materialization(self) -> None:
        """Expose a bound portable objective without emitting a current v5 record."""
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            bridge = ReferenceSequencePortableObjectiveBridge(dataset, device="cpu")
            key = ReferenceTransitionKey("alpha", "sample-000", 1)

            context = bridge.materialize(key)

            self.assertEqual(context.transition_key, key)
            self.assertFalse(hasattr(context, "training_sample"))
            self.assertEqual(
                context.projection_state.operator_volume_sha256, context.common_objective.operator_volume_sha256
            )
            self.assertEqual(
                context.projection_state.operator_geometry_sha256,
                context.common_objective.operator_geometry_sha256,
            )
            context.validate_immutable()

    def test_reuses_one_solver_and_predictor_static_graph_per_asset_but_not_transition_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = ReferenceSequenceV5Bridge(self._dataset(Path(directory)), device=torch.device("cpu"))
            first = bridge.materialize(ReferenceTransitionKey("alpha", "sample-000", 0))
            second = bridge.materialize(ReferenceTransitionKey("alpha", "sample-000", 2))
            other_sequence = bridge.materialize(ReferenceTransitionKey("alpha", "sample-001", 1))
            other_asset = bridge.materialize(ReferenceTransitionKey("beta", "sample-000", 0))

            self.assertEqual(bridge.cached_asset_count, 2)
            self.assertEqual(tuple(identity.asset_id for identity in bridge.cached_asset_identities), ("alpha", "beta"))
            self.assertIs(first.training_sample.projection_state, second.training_sample.projection_state)
            self.assertIs(first.training_sample.projection_state, other_sequence.training_sample.projection_state)
            self.assertIsNot(first.training_sample.projection_state, other_asset.training_sample.projection_state)
            self.assertIsNot(first.training_sample, bridge.materialize(first.transition_key).training_sample)

            bank = SharedTopologyPredictorBank(
                GraphTransformerConfig(
                    hidden_dim=16,
                    num_heads=4,
                    n_levels=0,
                    cluster_size=2,
                    dt=float(first.training_sample.common_objective.dt),
                    architecture_version=5,
                ),
                torch.float32,
            )
            first_predictor = bank.ensure(first.training_sample)
            self.assertIs(first_predictor, bank.ensure(second.training_sample))
            self.assertIs(first_predictor, bank.ensure(other_sequence.training_sample))
            self.assertEqual(len(bank.predictors), 1)
            self.assertIsNot(first_predictor, bank.ensure(other_asset.training_sample))
            self.assertEqual(len(bank.predictors), 2)

    def test_iter_materialized_is_lazy_and_sampler_replay_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            bridge = ReferenceSequenceV5Bridge(dataset, device="cpu")
            keys = bridge.sample_keys(DatasetRole.TRAIN, count=6, seed=179)
            self.assertEqual(keys, dataset.sample_keys(DatasetRole.TRAIN, count=6, seed=179))

            materialized = bridge.iter_materialized(keys)
            self.assertEqual(bridge.cached_asset_count, 0)
            first = next(materialized)
            self.assertEqual(first.transition_key, keys[0])
            self.assertEqual(bridge.cached_asset_count, 1)
            remaining = tuple(materialized)
            self.assertEqual(tuple(item.transition_key for item in (first, *remaining)), keys)
            self.assertLessEqual(bridge.cached_asset_count, 2)


if __name__ == "__main__":
    unittest.main()
