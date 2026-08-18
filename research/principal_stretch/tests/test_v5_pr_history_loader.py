# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import torch

from ..pr_scene_history import AtomicCoordinate, HistoryCheckpoint, PRHistoryChain, PRSceneHistory, _advance_prefix
from ..v5_pr_history_loader import (
    LOADER_SCOPE_SHA256,
    SOURCE_BOUND_LOADER_SCOPE_SHA256,
    LoadedPRHistoryV5Sample,
    canonical_runtime_material_sha256,
    canonical_runtime_pin_signature_sha256,
    load_pr_history_v5_sample,
    load_source_bound_pr_history_v5_sample,
    loader_scope,
)


class TestV5PRHistoryLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = PRSceneHistory("stretch")
        cls.chain = cls.history.generate(stop=AtomicCoordinate.from_ordinal(2), max_transitions=2)
        cls.transition = cls.chain.transitions[0]

    def _load(self, chain=None, transition=None, **kwargs):
        selected_chain = self.chain if chain is None else chain
        selected_transition = self.transition if transition is None else transition
        return load_pr_history_v5_sample(
            self.history,
            selected_chain,
            selected_transition,
            trajectory_id="unittest-pr-stretch-root-one",
            expected_history_chain_sha256=selected_chain.chain_sha256,
            expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
            max_chain_transitions=kwargs.pop("max_chain_transitions", 2),
            **kwargs,
        )

    @staticmethod
    def _one_transition_chain(chain, transition):
        prefix = _advance_prefix(chain.initial_checkpoint.prefix_sha256, transition.transition_sha256)
        final = HistoryCheckpoint(
            manifest_sha256=chain.manifest.manifest_sha256,
            state=transition.output_state,
            prior_transition_sha256=transition.transition_sha256,
            prefix_sha256=prefix,
        )
        return PRHistoryChain(
            manifest=chain.manifest,
            initial_checkpoint=chain.initial_checkpoint,
            transitions=(transition,),
            timings=chain.timings[:1],
            final_checkpoint=final,
            termination=chain.termination,
        )

    def test_loads_real_reference_and_serializes_recomputed_identities(self):
        loaded = self._load()
        loaded.validate_immutable()
        sample = loaded.training_sample
        record = sample.sample_record
        objective = sample.common_objective
        state = sample.projection_state

        self.assertEqual(record.physical_step_sha256, sample.physical_step.physical_step_sha256)
        self.assertEqual(record.common_objective_sha256, objective.common_objective_sha256)
        self.assertEqual(
            record.material_sha256,
            canonical_runtime_material_sha256(objective.mass, objective.mu, objective.lam),
        )
        self.assertEqual(
            record.pin_signature_sha256,
            canonical_runtime_pin_signature_sha256(objective.pinned, objective.tets, objective.n_vertices),
        )
        self.assertNotEqual(record.material_sha256, self.history.manifest.material_sha256)
        self.assertEqual(loaded.reference_acceptance.source_chain_sha256, self.chain.chain_sha256)
        self.assertEqual(
            loaded.reference_acceptance.source_transition_sha256,
            self.transition.transition_sha256,
        )
        self.assertEqual(
            loaded.reference_acceptance.metrics["source_float64_accepted_reference"]["gradient_norm"],
            self.transition.reference_record["final_gradient_norm"],
        )
        self.assertGreater(
            loaded.reference_acceptance.metrics["source_float64_committed_image"]["gradient_norm"],
            loaded.reference_acceptance.metrics["source_float64_accepted_reference"]["gradient_norm"],
        )
        self.assertFalse(loaded.reference_acceptance.committed_image_equilibrium_claimed)
        self.assertFalse(loaded.reference_acceptance.dense_newton_replayed)
        self.assertEqual(state.projection_backend, "dense")
        self.assertEqual(state.tikhonov, 0.0)
        self.assertEqual(state.source_rest_q_exact.dtype, torch.float32)
        self.assertEqual(state.source_tet_poses.dtype, torch.float32)
        self.assertTrue(
            torch.equal(state.source_rest_q_exact, torch.as_tensor(np.array(self.history.static_bundle.rest_q)))
        )
        self.assertTrue(
            torch.equal(state.source_tet_poses, torch.as_tensor(np.array(self.history.static_bundle.tet_poses)))
        )
        self.assertEqual(state.Dm_inv.dtype, torch.float64)
        self.assertEqual(state.J.dtype, torch.float64)
        self.assertEqual(state.w.dtype, torch.float64)
        self.assertTrue(torch.equal(state.Dm_inv, state.source_tet_poses.to(torch.float64)))
        self.assertEqual(record.operator_geometry_sha256, state.operator_geometry_sha256)
        self.assertEqual(record.as_dict()["operator_geometry_sha256"], state.operator_geometry_sha256)
        self.assertEqual(sample.operator_geometry_sha256, state.operator_geometry_sha256)
        sample.validate_immutable()

    def test_source_bound_loader_matches_live_evidence_without_claiming_callback_replay(self):
        live = self._load()
        source_bound = load_source_bound_pr_history_v5_sample(
            self.history.manifest,
            self.history.static_bundle,
            self.history._base_scene,
            self.chain,
            self.transition,
            trajectory_id="unittest-pr-stretch-root-one",
            expected_history_chain_sha256=self.chain.chain_sha256,
            expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
            max_chain_transitions=2,
        )
        self.assertEqual(
            source_bound.training_sample.sample_record.as_dict(),
            live.training_sample.sample_record.as_dict(),
        )
        self.assertEqual(source_bound.reference_acceptance.as_dict(), live.reference_acceptance.as_dict())
        self.assertEqual(source_bound.loader_scope_sha256, SOURCE_BOUND_LOADER_SCOPE_SHA256)
        self.assertFalse(loader_scope(SOURCE_BOUND_LOADER_SCOPE_SHA256)["source_callback_replayed"])

    def test_loaded_sample_rejects_cross_transition_acceptance_swap(self):
        first = self._load()
        second = self._load(transition=self.chain.transitions[1])
        with self.assertRaisesRegex(ValueError, "different PR source"):
            LoadedPRHistoryV5Sample(
                training_sample=first.training_sample,
                reference_acceptance=second.reference_acceptance,
                loader_scope_sha256=first.loader_scope_sha256,
            )

    def test_requires_external_root_chain_bound_and_exact_membership(self):
        with self.assertRaisesRegex(ValueError, "externally pinned SHA-256"):
            load_pr_history_v5_sample(
                self.history,
                self.chain,
                self.transition,
                trajectory_id="unittest-pr-stretch-root-one",
                expected_history_chain_sha256="0" * 64,
                expected_root_checkpoint_sha256=self.history.initial_checkpoint.checkpoint_sha256,
            )
        with self.assertRaisesRegex(ValueError, "canonical PR root"):
            load_pr_history_v5_sample(
                self.history,
                self.chain,
                self.transition,
                trajectory_id="unittest-pr-stretch-root-one",
                expected_history_chain_sha256=self.chain.chain_sha256,
                expected_root_checkpoint_sha256="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "max_chain_transitions must be a positive integer"):
            self._load(max_chain_transitions=0)
        with self.assertRaisesRegex(ValueError, "exact object"):
            self._load(transition=dataclasses.replace(self.transition))

    def test_rejects_self_consistent_reference_and_objective_record_forgery(self):
        forged_reference_record = self.transition.as_dict()["reference_record"]
        forged_reference_record["final_objective"] += 1.0
        forged_transition = dataclasses.replace(self.transition, reference_record=forged_reference_record)
        forged_chain = self._one_transition_chain(self.chain, forged_transition)
        with self.assertRaisesRegex(ValueError, "final_objective"):
            self._load(chain=forged_chain, transition=forged_transition)

        forged_objective = dataclasses.replace(self.transition, objective_instance_sha256="0" * 64)
        forged_objective_chain = self._one_transition_chain(self.chain, forged_objective)
        with self.assertRaisesRegex(ValueError, "common-objective SHA-256"):
            self._load(chain=forged_objective_chain, transition=forged_objective)

    def test_rejects_raw_reference_material_pin_and_transition_tampering(self):
        transition = self.transition

        original_scene_sha256 = transition.scene_sha256
        object.__setattr__(transition, "scene_sha256", "0" * 64)
        try:
            with self.assertRaisesRegex(ValueError, "raw content changed"):
                self._load()
        finally:
            object.__setattr__(transition, "scene_sha256", original_scene_sha256)

        reference = transition.reference_positions
        original_reference = reference.copy()
        reference.setflags(write=True)
        reference[0, 0] += 1.0e-6
        try:
            with self.assertRaisesRegex(
                ValueError,
                "reference record position hash|raw content changed|not the float32 committed reference",
            ):
                self._load()
        finally:
            reference[...] = original_reference
            reference.setflags(write=False)

        materials = self.history.static_bundle.tet_materials
        original_materials = materials.copy()
        materials.setflags(write=True)
        materials[0, 0] += 1.0
        try:
            with self.assertRaisesRegex(ValueError, "static bundle|raw content"):
                self._load()
        finally:
            materials[...] = original_materials
            materials.setflags(write=False)

        targets = transition.applied_state.pin_targets
        original_targets = targets.copy()
        targets.setflags(write=True)
        targets[0, 0] += 1.0e-4
        try:
            with self.assertRaisesRegex(ValueError, "raw content changed|applied record"):
                self._load()
        finally:
            targets[...] = original_targets
            targets.setflags(write=False)

    def test_scope_refuses_durable_or_dense_replay_claims(self):
        scope = loader_scope()
        self.assertFalse(scope["durable_artifact_bundle_opened"])
        self.assertFalse(scope["durable_artifact_source_opened"])
        self.assertFalse(scope["trajectory_provenance_verified"])
        self.assertFalse(scope["dense_newton_replayed"])
        self.assertEqual(len(LOADER_SCOPE_SHA256), 64)
        self.assertEqual(torch.device("cpu"), self._load().training_sample.common_objective.device)


if __name__ == "__main__":
    unittest.main()
