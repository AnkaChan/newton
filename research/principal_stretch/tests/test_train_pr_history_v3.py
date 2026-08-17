# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.principal_stretch.graph_transformer import GraphTransformerConfig
from research.principal_stretch.pr_scene_history import AtomicCoordinate, create_pr_scene_history
from research.principal_stretch.train_pr_history_v3 import (
    PRV3TrainingConfig,
    evaluate_pr_history_v3,
    load_pr_history_v3_checkpoint,
    train_pr_history_v3,
)


class TestPRHistoryV3Trainer(unittest.TestCase):
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
        self.assertEqual(metadata["seed_contract"]["numpy_generator_seed"], 17)
        self.assertEqual(metadata["seed_contract"]["torch_manual_seed"], 17)
        self.assertIn("newton_revision", metadata["source_provenance"])
        self.assertIn("dirty_tree_sha256", metadata["source_provenance"])
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
