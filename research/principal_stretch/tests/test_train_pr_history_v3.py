# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from research.principal_stretch.graph_transformer import GraphTransformerConfig
from research.principal_stretch.pr_scene_history import AtomicCoordinate, create_pr_scene_history
from research.principal_stretch.train_pr_history_v3 import (
    _SOURCE_PROVENANCE_AT_IMPORT,
    PRV3TrainingConfig,
    _canonical_digest,
    evaluate_pr_history_v3,
    load_pr_history_v3_checkpoint,
    train_pr_history_v3,
)


class TestPRHistoryV3Trainer(unittest.TestCase):
    @staticmethod
    def _reauthenticate_checkpoint(checkpoint):
        selection = checkpoint["metadata"].get("transition_selection")
        if isinstance(selection, dict) and "selection_sha256" in selection:
            selection_without_digest = dict(selection)
            selection_without_digest.pop("selection_sha256")
            selection["selection_sha256"] = _canonical_digest(selection_without_digest)
        checkpoint["metadata_sha256"] = _canonical_digest(checkpoint["metadata"])
        checkpoint["checkpoint_payload_sha256"] = _canonical_digest(
            {
                "contract": checkpoint["contract"],
                "state_dict_sha256": checkpoint["state_dict_sha256"],
                "metadata_sha256": checkpoint["metadata_sha256"],
            }
        )

    @classmethod
    def setUpClass(cls):
        cls.history = create_pr_scene_history("stretch")
        chain = cls.history.generate(
            stop=AtomicCoordinate.from_ordinal(6),
            max_transitions=6,
        )
        # f0s0 has the rest target; f1s0 applies the first nontrivial PR
        # boundary move.  Selecting both proves targets are sample-local.
        cls.transitions = (chain.transitions[0], chain.transitions[5])
        cls.graph_config = GraphTransformerConfig(
            hidden_dim=16,
            num_heads=4,
            n_levels=0,
            cluster_size=2,
            dropout=0.0,
            max_hencky_update=0.2,
            max_rotation_update=0.2,
            dt=cls.history.manifest.dt_seconds,
            architecture_version=3,
        )
        cls.training_config = PRV3TrainingConfig(
            steps=2,
            batch_size=2,
            learning_rate=1.0e-3,
            deformation_gradient_loss_weight=0.25,
            seed=17,
            log_every=1,
        )
        cls.result = train_pr_history_v3(
            cls.history,
            cls.transitions,
            graph_config=cls.graph_config,
            training_config=cls.training_config,
            device="cpu",
        )

    def test_moving_targets_and_authenticated_checkpoint(self):
        first_targets = self.transitions[0].model_inputs()["pin_targets"]
        moved_targets = self.transitions[1].model_inputs()["pin_targets"]
        self.assertFalse(np.array_equal(first_targets, moved_targets))

        checkpoint = self.result.checkpoint
        metadata = checkpoint["metadata"]
        self.assertEqual(metadata["history_manifest"]["manifest_sha256"], self.history.manifest.manifest_sha256)
        self.assertEqual(metadata["static_bundle"]["static_sha256"], self.history.static_bundle.static_sha256)
        selected = metadata["transition_selection"]["transitions"]
        self.assertEqual(
            [item["transition_sha256"] for item in selected],
            [transition.transition_sha256 for transition in self.transitions],
        )
        self.assertEqual(metadata["training_realized_hierarchy_levels"], 0)
        self.assertEqual(metadata["decoder_work"]["projection_backend"], "dense_cholesky")
        self.assertEqual(metadata["decoder_work"]["global_triangular_solves"], 1)
        self.assertEqual(metadata["decoder_work"]["local_polar_sweeps"], 0)
        self.assertEqual(
            metadata["training_work"]["predictor_passes_per_step"]["count"],
            "distinct_pin_signature_count_in_sampled_batch",
        )
        self.assertEqual(
            metadata["training_work"]["global_triangular_solves_per_step"]["count"],
            "distinct_pin_signature_count_in_sampled_batch",
        )
        self.assertEqual(metadata["training_work"]["available_pin_signature_count"], 1)
        self.assertEqual(metadata["training_work"]["predictor_passes_per_step"]["maximum"], 1)
        self.assertEqual(metadata["seed_contract"]["numpy_generator_seed"], 17)
        self.assertEqual(metadata["seed_contract"]["torch_manual_seed"], 17)
        self.assertIn("newton_revision", metadata["source_provenance"])
        self.assertIn("dirty_tree_sha256", metadata["source_provenance"])
        self.assertTrue(metadata["source_execution_binding"]["stable"])
        self.assertEqual(
            metadata["source_execution_binding"]["module_import"],
            metadata["source_provenance"],
        )
        self.assertEqual(
            metadata["source_execution_binding"]["training_start"],
            metadata["source_execution_binding"]["training_end"],
        )
        self.assertIn("checkpoint_payload_sha256", checkpoint)

    def test_common_objective_evaluation_preserves_dynamic_pins(self):
        report = evaluate_pr_history_v3(
            self.history,
            self.transitions,
            self.result.checkpoint,
            device="cpu",
            warmup=0,
            repeats=1,
        )
        deterministic = report["deterministic"]
        self.assertEqual(deterministic["summary"]["sample_count"], 2)
        self.assertIn("evaluation_sha256", deterministic)
        self.assertIn("timing_sha256", report["timing"])
        for transition, sample, timing in zip(
            self.transitions,
            deterministic["samples"],
            report["timing"]["samples"],
            strict=True,
        ):
            self.assertEqual(sample["transition_sha256"], transition.transition_sha256)
            self.assertEqual(sample["objective_instance_sha256"], transition.objective_instance_sha256)
            self.assertEqual(sample["metrics"]["max_pin_error_m"], 0.0)
            self.assertTrue(math.isfinite(sample["metrics"]["relative_residual"]))
            self.assertTrue(math.isfinite(sample["metrics"]["free_rms_error_m"]))
            self.assertEqual(timing["repeat_max_discrepancy_m"], 0.0)
            self.assertEqual(timing["repeat_discrepancy_tolerance_m"], 0.0)
            self.assertGreaterEqual(timing["input_preparation_seconds"], 0.0)
            self.assertEqual(len(timing["adapter_call_repeat_seconds"]), 1)
            self.assertEqual(timing["adapter_resident_max_discrepancy_m"], 0.0)
            self.assertIn("device-resident", timing["inference_scope"])
            self.assertEqual(timing["timing_temperature"], "unwarmed first resident repeat")
        self.assertEqual(deterministic["checkpoint_identity"]["schema_version"], 3)
        self.assertEqual(deterministic["checkpoint_identity"]["contract"], "pr2901-history-v3-checkpoint-v3")
        self.assertTrue(deterministic["training_evaluation_selection_match"])

    def test_seed_reproduces_transition_sampling(self):
        repeated = train_pr_history_v3(
            self.history,
            self.transitions,
            graph_config=self.graph_config,
            training_config=self.training_config,
            device="cpu",
        )
        first_log = self.result.checkpoint["metadata"]["training_log"]
        repeated_log = repeated.checkpoint["metadata"]["training_log"]
        self.assertEqual(
            [item["sample_indices"] for item in repeated_log],
            [item["sample_indices"] for item in first_log],
        )
        self.assertEqual(
            [item["transition_sha256"] for item in repeated_log],
            [item["transition_sha256"] for item in first_log],
        )
        self.assertEqual(
            repeated.checkpoint["metadata"]["seed_contract"],
            self.result.checkpoint["metadata"]["seed_contract"],
        )

    def test_normalized_raw_target_loss_has_unit_baseline_and_head_gradients(self):
        training_config = PRV3TrainingConfig(
            steps=1,
            batch_size=1,
            learning_rate=1.0e-3,
            position_loss_weight=0.0,
            deformation_gradient_loss_weight=0.0,
            loss_mode="normalized-raw-deformation-gradient",
            raw_deformation_gradient_floor=1.0e-8,
            seed=23,
            log_every=1,
        )
        # Raw target supervision must not pay for or differentiate through the
        # global projection used only by decoded-position training/evaluation.
        with mock.patch(
            "research.principal_stretch.train_pr_history_v3.torch_solver.project_deformation_gradient",
            side_effect=AssertionError("raw target training unexpectedly decoded positions"),
        ):
            result = train_pr_history_v3(
                self.history,
                self.transitions[1:],
                graph_config=self.graph_config,
                training_config=training_config,
                device="cpu",
            )

        checkpoint = result.checkpoint
        log = checkpoint["metadata"]["training_log"]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["loss_mode"], "normalized-raw-deformation-gradient")
        self.assertAlmostEqual(log[0]["normalized_raw_deformation_gradient_loss"], 1.0, places=10)
        self.assertAlmostEqual(
            log[0]["raw_target_deformation_gradient_loss"],
            log[0]["raw_deformation_gradient_normalizers"][0],
            places=14,
        )
        self.assertNotIn("normalized_position_loss", log[0])
        self.assertNotIn("volume_weighted_deformation_gradient_loss", log[0])
        self.assertGreater(log[0]["gradient_norm_before_clipping"], 0.0)

        state_dict = checkpoint["state_dict"]
        self.assertGreater(float(torch.linalg.vector_norm(state_dict["output_head.2.weight"])), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(state_dict["rotation_head.2.weight"])), 0.0)
        metadata = checkpoint["metadata"]
        self.assertEqual(metadata["training_config"]["loss_mode"], "normalized-raw-deformation-gradient")
        self.assertEqual(metadata["loss_contract"]["active_mode"], "normalized-raw-deformation-gradient")
        self.assertEqual(metadata["loss_contract"]["raw_deformation_gradient_floor"], 1.0e-8)
        self.assertEqual(metadata["training_work"]["global_triangular_solves_per_step"]["count"], 0)
        self.assertEqual(metadata["training_work"]["global_triangular_solves_per_step"]["maximum"], 0)
        self.assertFalse(metadata["training_work"]["decoded_position_loss_evaluated"])
        self.assertTrue(metadata["training_work"]["raw_target_deformation_gradient_loss_evaluated"])
        self.assertEqual(
            metadata["loss_contract"]["raw_deformation_gradient_precision"],
            "float64 target/reference/observed fields and rest-volume reduction",
        )
        self.assertEqual(checkpoint["schema_version"], 3)
        self.assertEqual(checkpoint["contract"], "pr2901-history-v3-checkpoint-v3")

        rest_config = dataclasses.replace(training_config, seed=1)
        with mock.patch(
            "research.principal_stretch.train_pr_history_v3.torch_solver.project_deformation_gradient",
            side_effect=AssertionError("raw target training unexpectedly decoded positions"),
        ):
            rest_result = train_pr_history_v3(
                self.history,
                self.transitions,
                graph_config=self.graph_config,
                training_config=rest_config,
                device="cpu",
            )
        rest_log = rest_result.checkpoint["metadata"]["training_log"][0]
        self.assertEqual(rest_log["sample_indices"], [0])
        # The zero heads reconstruct the observed gradient through A exp(H),
        # whose fp64 spectral round trip is not bit-exact even at rest.
        self.assertLess(rest_log["normalized_raw_deformation_gradient_loss"], 1.0e-8)
        self.assertLess(rest_log["raw_target_deformation_gradient_loss"], 1.0e-16)
        self.assertEqual(rest_log["raw_deformation_gradient_normalizers"], [1.0e-8])

    def test_loss_mode_validation_preserves_decoded_default(self):
        self.assertEqual(PRV3TrainingConfig().loss_mode, "decoded-position-deformation")
        with self.assertRaisesRegex(ValueError, "requires position_loss_weight=0"):
            PRV3TrainingConfig(loss_mode="normalized-raw-deformation-gradient")
        with self.assertRaisesRegex(ValueError, "decoded loss mode"):
            PRV3TrainingConfig(position_loss_weight=0.0, deformation_gradient_loss_weight=0.0)
        with self.assertRaisesRegex(ValueError, "loss_mode must be one of"):
            PRV3TrainingConfig(loss_mode="unknown")
        with self.assertRaisesRegex(ValueError, "floor must be finite and positive"):
            PRV3TrainingConfig(raw_deformation_gradient_floor=0.0)
        for name in ("steps", "batch_size", "seed", "log_every"):
            with self.subTest(integer_field=name):
                with self.assertRaisesRegex(ValueError, "integer"):
                    PRV3TrainingConfig(**{name: 1.5})

    def test_save_load_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history-v3.pt"
            saved = train_pr_history_v3(
                self.history,
                self.transitions[:1],
                graph_config=self.graph_config,
                training_config=PRV3TrainingConfig(steps=1, batch_size=1, seed=5),
                device="cpu",
                output_path=path,
            )
            self.assertTrue(path.is_file())
            predictor, dataset, loaded = load_pr_history_v3_checkpoint(
                path,
                self.history,
                self.transitions[:1],
                device="cpu",
            )
            self.assertEqual(predictor.model.n_levels, 0)
            self.assertEqual(len(dataset.samples), 1)
            self.assertEqual(loaded["checkpoint_payload_sha256"], saved.checkpoint["checkpoint_payload_sha256"])
            with self.assertRaises(FileExistsError):
                train_pr_history_v3(
                    self.history,
                    self.transitions[:1],
                    graph_config=self.graph_config,
                    training_config=PRV3TrainingConfig(steps=1, batch_size=1, seed=5),
                    device="cpu",
                    output_path=path,
                )

        tampered = copy.deepcopy(self.result.checkpoint)
        tampered["metadata"]["training_config"]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "metadata SHA-256"):
            load_pr_history_v3_checkpoint(
                tampered,
                self.history,
                self.transitions,
                device="cpu",
            )

    def test_legacy_v1_checkpoint_identity_remains_loadable(self):
        legacy = copy.deepcopy(self.result.checkpoint)
        legacy["schema_version"] = 1
        legacy["contract"] = "pr2901-history-v3-checkpoint-v1"
        metadata = legacy["metadata"]
        metadata["schema_version"] = 1
        metadata["contract"] = "pr2901-history-v3-checkpoint-v1"
        metadata["training_config"].pop("loss_mode")
        metadata["training_config"].pop("raw_deformation_gradient_floor")
        metadata.pop("training_work")
        metadata.pop("source_execution_binding")
        metadata["loss_contract"] = {
            "position": "sum_free(m_i*||x_i-x_ref_i||^2)/(sum_free(m_i)*rms_rest_edge^2)",
            "deformation_gradient": "sum_t(V_t*||F_t-F_ref_t||_F^2)/(9*sum_t(V_t))",
            "characteristic_length_m": metadata["loss_contract"]["characteristic_length_m"],
            "pins": "transition-local exact Dirichlet indices and targets excluded from position loss",
        }
        for selected in metadata["transition_selection"]["transitions"]:
            selected.pop("pin_signature")
            selected.pop("pin_signature_sha256")
            selected.pop("pin_count")
            selected.pop("observed_F_sha256")
            selected.pop("reference_F_sha256")
            selected.pop("raw_deformation_gradient_observed_loss")
        metadata["transition_selection"]["contract"] = "pr2901-history-selected-transition-set-v1"
        metadata["transition_selection"].pop("selection_sha256")
        metadata["transition_selection"]["selection_sha256"] = _canonical_digest(metadata["transition_selection"])
        for entry in metadata["training_log"]:
            entry.pop("loss_mode")
        self._reauthenticate_checkpoint(legacy)
        predictor, dataset, loaded = load_pr_history_v3_checkpoint(
            legacy,
            self.history,
            self.transitions,
            device="cpu",
        )
        self.assertEqual(predictor.model.n_levels, 0)
        self.assertEqual(len(dataset.samples), 2)
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["contract"], "pr2901-history-v3-checkpoint-v1")
        report = evaluate_pr_history_v3(
            self.history,
            self.transitions,
            legacy,
            device="cpu",
            warmup=0,
            repeats=1,
        )
        self.assertEqual(report["deterministic"]["checkpoint_identity"]["schema_version"], 1)
        self.assertEqual(
            report["deterministic"]["checkpoint_identity"]["contract"],
            "pr2901-history-v3-checkpoint-v1",
        )
        self.assertFalse(report["deterministic"]["training_evaluation_selection_match"])

    def test_schema_v3_fails_closed_on_loss_work_and_source_semantics(self):
        mutations = {
            "missing loss mode": lambda metadata: metadata["training_config"].pop("loss_mode"),
            "mislabeled loss": lambda metadata: metadata["loss_contract"].update(
                active_mode="normalized-raw-deformation-gradient"
            ),
            "mislabeled work": lambda metadata: metadata["training_work"].update(decoded_position_loss_evaluated=False),
            "unstable source": lambda metadata: metadata["source_execution_binding"].update(stable=False),
            "wrong selection contract": lambda metadata: metadata["transition_selection"].update(
                contract="pr2901-history-selected-transition-set-v1"
            ),
            "wrong selection scope": lambda metadata: metadata["transition_selection"].update(
                provenance_scope="complete-history"
            ),
            "wrong selection history": lambda metadata: metadata["transition_selection"].update(
                history_manifest_sha256="0" * 64
            ),
            "wrong selection static": lambda metadata: metadata["transition_selection"].update(static_sha256="1" * 64),
            "wrong pin count": lambda metadata: metadata["transition_selection"]["transitions"][0].update(
                pin_count=999
            ),
            "non-hex field digest": lambda metadata: metadata["transition_selection"]["transitions"][0].update(
                observed_F_sha256="g" * 64
            ),
            "duplicate transition": lambda metadata: metadata["transition_selection"]["transitions"].append(
                copy.deepcopy(metadata["transition_selection"]["transitions"][0])
            ),
            "mislabeled log": lambda metadata: metadata["training_log"][0].update(
                loss_mode="normalized-raw-deformation-gradient"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                malformed = copy.deepcopy(self.result.checkpoint)
                mutate(malformed["metadata"])
                self._reauthenticate_checkpoint(malformed)
                with self.assertRaisesRegex(ValueError, "schema-v3 checkpoint"):
                    load_pr_history_v3_checkpoint(
                        malformed,
                        self.history,
                        self.transitions,
                        device="cpu",
                    )

        experimental_v2 = copy.deepcopy(self.result.checkpoint)
        experimental_v2["schema_version"] = 2
        experimental_v2["contract"] = "pr2901-history-v3-checkpoint-v2"
        experimental_v2["metadata"]["schema_version"] = 2
        experimental_v2["metadata"]["contract"] = "pr2901-history-v3-checkpoint-v2"
        self._reauthenticate_checkpoint(experimental_v2)
        with self.assertRaisesRegex(ValueError, "unsupported PR history v3 checkpoint schema"):
            load_pr_history_v3_checkpoint(
                experimental_v2,
                self.history,
                self.transitions,
                device="cpu",
            )

    def test_training_rejects_a_source_tree_change(self):
        changed = dict(_SOURCE_PROVENANCE_AT_IMPORT)
        changed["dirty_tree_sha256"] = "f" * 64
        if changed == _SOURCE_PROVENANCE_AT_IMPORT:
            changed["newton_revision"] = "0" * 40

        with mock.patch(
            "research.principal_stretch.train_pr_history_v3._source_provenance",
            return_value=changed,
        ):
            with self.assertRaisesRegex(RuntimeError, "changed after this trainer module was imported"):
                train_pr_history_v3(
                    self.history,
                    self.transitions[:1],
                    graph_config=self.graph_config,
                    training_config=PRV3TrainingConfig(steps=1, batch_size=1),
                    device="cpu",
                )

        with mock.patch(
            "research.principal_stretch.train_pr_history_v3._source_provenance",
            side_effect=(_SOURCE_PROVENANCE_AT_IMPORT, changed),
        ):
            with self.assertRaisesRegex(RuntimeError, "changed during training"):
                train_pr_history_v3(
                    self.history,
                    self.transitions[:1],
                    graph_config=self.graph_config,
                    training_config=PRV3TrainingConfig(steps=1, batch_size=1),
                    device="cpu",
                )

    def test_fail_closed_on_config_or_history_mismatch(self):
        with self.assertRaisesRegex(ValueError, "unsupported projection_backend"):
            PRV3TrainingConfig(projection_backend="sparse_pcg")

        wrong_dt = GraphTransformerConfig(
            hidden_dim=16,
            num_heads=4,
            n_levels=0,
            cluster_size=2,
            dt=1.0 / 300.0,
            architecture_version=3,
        )
        self.assertNotEqual(wrong_dt.dt, self.history.manifest.dt_seconds)
        with self.assertRaisesRegex(ValueError, "dt must exactly equal"):
            train_pr_history_v3(
                self.history,
                self.transitions[:1],
                graph_config=wrong_dt,
                training_config=PRV3TrainingConfig(steps=1, batch_size=1),
                device="cpu",
            )

        other_history = create_pr_scene_history("twist")
        with self.assertRaisesRegex(ValueError, "different history manifest"):
            train_pr_history_v3(
                other_history,
                self.transitions[:1],
                graph_config=GraphTransformerConfig(
                    hidden_dim=16,
                    num_heads=4,
                    n_levels=0,
                    cluster_size=2,
                    dt=other_history.manifest.dt_seconds,
                    architecture_version=3,
                ),
                training_config=PRV3TrainingConfig(steps=1, batch_size=1),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
