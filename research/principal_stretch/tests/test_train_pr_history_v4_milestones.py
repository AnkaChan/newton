# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from research.principal_stretch.graph_transformer import GraphTransformerConfig
from research.principal_stretch.pr_scene_history import AtomicCoordinate, create_pr_scene_history
from research.principal_stretch.train_pr_history_v3 import (
    PRV3TrainingConfig,
    PRV4MilestoneConfig,
    _build_phase_balanced_schedule_from_records,
    _canonical_digest,
    _state_dict_digest,
    _state_tree_digest,
    _validate_equal_milestone_exposure,
    evaluate_pr_history_v4_raw_f,
    load_pr_history_v3_checkpoint,
    train_pr_history_v3,
    train_pr_history_v4_milestones,
)


def _fake_transition_records(sample_count: int) -> list[dict[str, object]]:
    return [
        {
            "coordinate": {"frame": index // 5, "substep": index % 5, "ordinal": index},
            "transition_sha256": hashlib.sha256(f"transition-{index}".encode()).hexdigest(),
        }
        for index in range(sample_count)
    ]


class TestPhaseBalancedEpochSchedule(unittest.TestCase):
    def test_exact_phase_balance_equal_exposure_and_replay(self):
        records = _fake_transition_records(15)
        first = _build_phase_balanced_schedule_from_records(
            records,
            substeps_per_frame=5,
            steps=3,
            batch_size=10,
            seed=1701,
        )
        repeated = _build_phase_balanced_schedule_from_records(
            records,
            substeps_per_frame=5,
            steps=3,
            batch_size=10,
            seed=1701,
        )
        np.testing.assert_array_equal(first.batches, repeated.batches)
        self.assertEqual(first.record, repeated.record)
        self.assertEqual(first.batches.shape, (3, 10))
        self.assertEqual(first.record["exposure_by_sample_index"], [2] * 15)
        for batch in first.batches:
            self.assertEqual(len(set(int(index) for index in batch)), 10)
            phases = [records[int(index)]["coordinate"]["substep"] for index in batch]
            np.testing.assert_array_equal(np.bincount(phases, minlength=5), np.full(5, 2))

    def test_sampling_and_milestone_constraints_fail_closed(self):
        records = _fake_transition_records(15)
        with self.assertRaisesRegex(ValueError, "divisible by the sample count"):
            _build_phase_balanced_schedule_from_records(
                records,
                substeps_per_frame=5,
                steps=2,
                batch_size=10,
                seed=1701,
            )
        uneven = records[:-1]
        with self.assertRaisesRegex(ValueError, "same positive sample count"):
            _build_phase_balanced_schedule_from_records(
                uneven,
                substeps_per_frame=5,
                steps=7,
                batch_size=10,
                seed=1701,
            )
        schedule = _build_phase_balanced_schedule_from_records(
            records,
            substeps_per_frame=5,
            steps=3,
            batch_size=10,
            seed=1701,
        )
        with self.assertRaisesRegex(ValueError, "milestone update 1 does not give equal exposure"):
            _validate_equal_milestone_exposure(
                schedule,
                PRV4MilestoneConfig(milestone_updates=(1, 3)),
            )


class TestPRHistoryV4Milestones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_torch_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.history = create_pr_scene_history("stretch")
        chain = cls.history.generate(
            stop=AtomicCoordinate.from_ordinal(5),
            max_transitions=5,
        )
        cls.transitions = tuple(chain.transitions)
        cls.graph_config = GraphTransformerConfig(
            hidden_dim=8,
            num_heads=2,
            n_levels=0,
            cluster_size=2,
            dropout=0.0,
            max_hencky_update=0.2,
            max_rotation_update=0.2,
            max_multiplicative_update=0.25,
            dt=cls.history.manifest.dt_seconds,
            architecture_version=4,
        )
        cls.training_config = PRV3TrainingConfig(
            steps=4,
            batch_size=5,
            learning_rate=1.0e-3,
            weight_decay=1.0e-5,
            position_loss_weight=0.0,
            deformation_gradient_loss_weight=0.0,
            loss_mode="normalized-raw-deformation-gradient",
            raw_deformation_gradient_floor=1.0e-8,
            gradient_clip_norm=1.0e-12,
            seed=41,
            log_every=2,
        )
        cls.milestone_config = PRV4MilestoneConfig(
            milestone_updates=(2, 4),
            track_parameter_update_norm=True,
        )
        cls._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls._temporary_directory.name)
        cls.uninterrupted = train_pr_history_v4_milestones(
            cls.history,
            cls.transitions,
            graph_config=cls.graph_config,
            training_config=cls.training_config,
            milestone_config=cls.milestone_config,
            device="cpu",
            output_directory=root / "uninterrupted",
        )
        cls.resumed = train_pr_history_v4_milestones(
            cls.history,
            cls.transitions,
            graph_config=cls.graph_config,
            training_config=cls.training_config,
            milestone_config=cls.milestone_config,
            device="cpu",
            output_directory=root / "resumed",
            resume_from=cls.uninterrupted.checkpoint_paths[2],
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()
        torch.set_num_threads(cls._previous_torch_thread_count)

    @staticmethod
    def _reauthenticate(checkpoint: dict[str, object]) -> None:
        checkpoint["state_dict_sha256"] = _state_dict_digest(checkpoint["state_dict"])
        checkpoint["optimizer_state_sha256"] = _state_tree_digest(checkpoint["optimizer_state"])
        checkpoint["rng_state_sha256"] = _state_tree_digest(checkpoint["rng_state"])
        checkpoint["diagnostic_series_sha256"] = _canonical_digest(checkpoint["diagnostic_series"])
        checkpoint["restart_state"]["optimizer_state_sha256"] = checkpoint["optimizer_state_sha256"]
        checkpoint["restart_state"]["rng_state_sha256"] = checkpoint["rng_state_sha256"]
        checkpoint["restart_state"]["diagnostic_series_sha256"] = checkpoint["diagnostic_series_sha256"]
        checkpoint["restart_state_sha256"] = _canonical_digest(checkpoint["restart_state"])
        checkpoint["metadata_sha256"] = _canonical_digest(checkpoint["metadata"])
        checkpoint["checkpoint_payload_sha256"] = _canonical_digest(
            {
                "contract": checkpoint["contract"],
                "state_dict_sha256": checkpoint["state_dict_sha256"],
                "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
                "rng_state_sha256": checkpoint["rng_state_sha256"],
                "diagnostic_series_sha256": checkpoint["diagnostic_series_sha256"],
                "restart_state_sha256": checkpoint["restart_state_sha256"],
                "metadata_sha256": checkpoint["metadata_sha256"],
            }
        )

    def test_uninterrupted_and_resumed_states_are_identical(self):
        uninterrupted = self.uninterrupted.checkpoints[4]
        resumed = self.resumed.checkpoints[4]
        self.assertEqual(uninterrupted["state_dict_sha256"], resumed["state_dict_sha256"])
        self.assertEqual(uninterrupted["optimizer_state_sha256"], resumed["optimizer_state_sha256"])
        self.assertEqual(uninterrupted["diagnostic_series"], resumed["diagnostic_series"])
        for name, tensor in uninterrupted["state_dict"].items():
            self.assertTrue(torch.equal(tensor, resumed["state_dict"][name]), msg=name)
        self.assertEqual(
            _state_tree_digest(uninterrupted["optimizer_state"]),
            _state_tree_digest(resumed["optimizer_state"]),
        )
        self.assertEqual(uninterrupted["metadata"]["training_log"], resumed["metadata"]["training_log"])
        self.assertIsNone(uninterrupted["metadata"]["training_progress"]["external_parent_lineage"])
        parent = self.uninterrupted.checkpoints[2]
        lineage = resumed["metadata"]["training_progress"]["external_parent_lineage"]
        self.assertEqual(lineage["parent_checkpoint_payload_sha256"], parent["checkpoint_payload_sha256"])
        self.assertEqual(lineage["parent_completed_updates"], 2)
        self.assertEqual(lineage["parent_state_dict_sha256"], parent["state_dict_sha256"])
        self.assertEqual(lineage["parent_optimizer_state_sha256"], parent["optimizer_state_sha256"])
        self.assertEqual(lineage["parent_rng_state_sha256"], parent["rng_state_sha256"])
        self.assertEqual(lineage["parent_diagnostic_series_sha256"], parent["diagnostic_series_sha256"])
        self.assertEqual(lineage["parent_restart_state_sha256"], parent["restart_state_sha256"])
        self.assertEqual(lineage["parent_metadata_sha256"], parent["metadata_sha256"])
        self.assertEqual(
            lineage["parent_training_progress_sha256"],
            _canonical_digest(parent["metadata"]["training_progress"]),
        )
        self.assertIn("externally pin parent file", lineage["verification_scope"])

    def test_milestone_progress_sampling_and_all_step_diagnostics(self):
        for completed, checkpoint in self.uninterrupted.checkpoints.items():
            metadata = checkpoint["metadata"]
            progress = metadata["training_progress"]
            self.assertEqual(progress["completed_updates"], completed)
            self.assertEqual(progress["sampling_prefix"]["minimum_exposure"], completed)
            self.assertEqual(progress["sampling_prefix"]["maximum_exposure"], completed)
            self.assertTrue(progress["sampling_prefix"]["equal_exposure_at_this_milestone"])
            series = checkpoint["diagnostic_series"]
            self.assertEqual(len(series["gradient_norm_before_clipping"]), completed)
            self.assertEqual(len(series["parameter_update_norm"]), completed)
            diagnostics = metadata["optimization_diagnostics"]
            gradient = diagnostics["gradient_norm_before_clipping"]
            self.assertEqual(gradient["count"], completed)
            self.assertEqual(gradient["clipped_count"], completed)
            self.assertEqual(gradient["clipped_fraction"], 1.0)
            self.assertEqual(diagnostics["parameter_update_norm"]["count"], completed)
            self.assertTrue(all(value > 0.0 for value in series["parameter_update_norm"]))
            self.assertEqual(metadata["sampling"]["exposure_by_sample_index"], [4] * 5)
            self.assertTrue(self.uninterrupted.checkpoint_paths[completed].is_file())

    def test_checkpoint_load_and_raw_f_evaluation_never_project(self):
        checkpoint = self.resumed.checkpoints[4]
        predictor, dataset, loaded = load_pr_history_v3_checkpoint(
            checkpoint,
            self.history,
            self.transitions,
            device="cpu",
        )
        self.assertEqual(predictor.model.config.architecture_version, 4)
        self.assertEqual(len(dataset.samples), 5)
        self.assertEqual(loaded["schema_version"], 4)
        with mock.patch(
            "research.principal_stretch.train_pr_history_v3.torch_solver.project_deformation_gradient",
            side_effect=AssertionError("raw-F evaluation unexpectedly projected positions"),
        ):
            report = evaluate_pr_history_v4_raw_f(
                self.history,
                self.transitions,
                checkpoint,
                device="cpu",
            )
        self.assertEqual(report["projection_calls"], 0)
        self.assertEqual(report["summary"]["sample_count"], 5)
        values = [sample["normalized_raw_deformation_gradient_loss"] for sample in report["samples"]]
        self.assertAlmostEqual(
            report["summary"]["mean_normalized_raw_deformation_gradient_loss"],
            sum(values) / len(values),
        )
        self.assertLessEqual(
            report["summary"]["geometric_mean_normalized_raw_deformation_gradient_loss"],
            report["summary"]["maximum_normalized_raw_deformation_gradient_loss"],
        )
        self.assertIn("evaluation_sha256", report)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_restart_is_within_fixed_fixture_bounds(self):
        """Bound one fixed smoke fixture, not CUDA restart behavior generally."""
        graph_config = dataclasses.replace(
            self.graph_config,
            hidden_dim=32,
            num_heads=4,
            n_levels=2,
            cluster_size=8,
            max_hencky_update=0.35,
            max_rotation_update=0.75,
            max_multiplicative_update=0.5,
        )
        training_config = dataclasses.replace(
            self.training_config,
            learning_rate=3.0e-4,
            gradient_clip_norm=5.0,
            seed=1701,
        )
        root = Path(self._temporary_directory.name) / "cuda-fixed-fixture"
        uninterrupted = train_pr_history_v4_milestones(
            self.history,
            self.transitions,
            graph_config=graph_config,
            training_config=training_config,
            milestone_config=self.milestone_config,
            device="cuda",
            output_directory=root / "uninterrupted",
        )
        resumed = train_pr_history_v4_milestones(
            self.history,
            self.transitions,
            graph_config=graph_config,
            training_config=training_config,
            milestone_config=self.milestone_config,
            device="cuda",
            output_directory=root / "resumed",
            resume_from=uninterrupted.checkpoint_paths[2],
        )
        uninterrupted_checkpoint = uninterrupted.checkpoints[4]
        resumed_checkpoint = resumed.checkpoints[4]
        self.assertEqual(
            resumed_checkpoint["metadata"]["restart_reproducibility"]["mode"],
            "authenticated-stateful-cuda-tolerance-repeatable",
        )

        model_max_abs = 0.0
        for name, left in uninterrupted_checkpoint["state_dict"].items():
            right = resumed_checkpoint["state_dict"][name]
            if left.is_floating_point():
                model_max_abs = max(
                    model_max_abs,
                    float((left.to(torch.float64) - right.to(torch.float64)).abs().max()),
                )
            else:
                self.assertTrue(torch.equal(left, right), msg=name)
        self.assertLessEqual(model_max_abs, 1.0e-6)

        uninterrupted_optimizer = uninterrupted_checkpoint["optimizer_state"]
        resumed_optimizer = resumed_checkpoint["optimizer_state"]
        self.assertEqual(uninterrupted_optimizer["param_groups"], resumed_optimizer["param_groups"])
        adam_max_abs = 0.0
        for parameter_index, left_state in uninterrupted_optimizer["state"].items():
            right_state = resumed_optimizer["state"][parameter_index]
            self.assertEqual(set(left_state), set(right_state))
            for name, left in left_state.items():
                right = right_state[name]
                if isinstance(left, torch.Tensor) and left.is_floating_point():
                    adam_max_abs = max(
                        adam_max_abs,
                        float((left.to(torch.float64) - right.to(torch.float64)).abs().max()),
                    )
                elif isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right), msg=f"{parameter_index}:{name}")
                else:
                    self.assertEqual(left, right, msg=f"{parameter_index}:{name}")
        self.assertLessEqual(adam_max_abs, 1.0e-6)

        uninterrupted_series = uninterrupted_checkpoint["diagnostic_series"]
        resumed_series = resumed_checkpoint["diagnostic_series"]
        gradient_difference = max(
            abs(left - right)
            for left, right in zip(
                uninterrupted_series["gradient_norm_before_clipping"],
                resumed_series["gradient_norm_before_clipping"],
                strict=True,
            )
        )
        update_difference = max(
            abs(left - right)
            for left, right in zip(
                uninterrupted_series["parameter_update_norm"],
                resumed_series["parameter_update_norm"],
                strict=True,
            )
        )
        self.assertLessEqual(gradient_difference, 2.0e-2)
        self.assertLessEqual(update_difference, 1.0e-8)

    def test_hash_progress_and_exact_resume_gates_reject_tampering(self):
        checkpoint = self.uninterrupted.checkpoints[2]

        state_tamper = copy.deepcopy(checkpoint)
        name = next(iter(state_tamper["state_dict"]))
        state_tamper["state_dict"][name].reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(ValueError, "state_dict SHA-256"):
            load_pr_history_v3_checkpoint(state_tamper, self.history, self.transitions, device="cpu")

        optimizer_tamper = copy.deepcopy(checkpoint)
        optimizer_state = next(iter(optimizer_tamper["optimizer_state"]["state"].values()))
        optimizer_state["exp_avg"].reshape(-1)[0] += 1.0
        with self.assertRaisesRegex(ValueError, "optimizer-state SHA-256"):
            load_pr_history_v3_checkpoint(optimizer_tamper, self.history, self.transitions, device="cpu")

        progress_tamper = copy.deepcopy(checkpoint)
        progress_tamper["metadata"]["training_progress"]["next_batch_index"] = 1
        self._reauthenticate(progress_tamper)
        with self.assertRaisesRegex(ValueError, "training progress"):
            load_pr_history_v3_checkpoint(progress_tamper, self.history, self.transitions, device="cpu")

        reversed_transitions = tuple(reversed(self.transitions))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "training selection does not exactly match"):
                train_pr_history_v4_milestones(
                    self.history,
                    reversed_transitions,
                    graph_config=self.graph_config,
                    training_config=self.training_config,
                    milestone_config=self.milestone_config,
                    device="cpu",
                    output_directory=directory,
                    resume_from=checkpoint,
                )

    def test_seed_and_optimizer_binding_gates_reject_reauthenticated_tampering(self):
        checkpoint = self.uninterrupted.checkpoints[2]

        seed_tamper = copy.deepcopy(checkpoint)
        seed_tamper["metadata"]["seed_contract"]["numpy_generator_seed"] += 1
        self._reauthenticate(seed_tamper)
        with self.assertRaisesRegex(ValueError, "seed contract"):
            load_pr_history_v3_checkpoint(seed_tamper, self.history, self.transitions, device="cpu")

        order_tamper = copy.deepcopy(checkpoint)
        parameter_ids = order_tamper["optimizer_state"]["param_groups"][0]["params"]
        parameter_ids[0], parameter_ids[1] = parameter_ids[1], parameter_ids[0]
        self._reauthenticate(order_tamper)
        with self.assertRaisesRegex(ValueError, "parameter order"):
            load_pr_history_v3_checkpoint(order_tamper, self.history, self.transitions, device="cpu")

        coverage_tamper = copy.deepcopy(checkpoint)
        first_parameter_id = coverage_tamper["optimizer_state"]["param_groups"][0]["params"][0]
        del coverage_tamper["optimizer_state"]["state"][first_parameter_id]
        self._reauthenticate(coverage_tamper)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            load_pr_history_v3_checkpoint(coverage_tamper, self.history, self.transitions, device="cpu")

    def test_scope_and_exclusive_output_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "architecture version 4"):
                train_pr_history_v4_milestones(
                    self.history,
                    self.transitions,
                    graph_config=dataclasses.replace(self.graph_config, architecture_version=3),
                    training_config=self.training_config,
                    milestone_config=self.milestone_config,
                    output_directory=directory,
                )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "dropout=0"):
                train_pr_history_v4_milestones(
                    self.history,
                    self.transitions,
                    graph_config=dataclasses.replace(self.graph_config, dropout=0.1),
                    training_config=self.training_config,
                    milestone_config=self.milestone_config,
                    output_directory=directory,
                )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PRV4MilestoneConfig(milestone_updates=(2, 2))

        checkpoint_path = self.uninterrupted.checkpoint_paths[4]
        with self.assertRaises(FileExistsError):
            train_pr_history_v4_milestones(
                self.history,
                self.transitions,
                graph_config=self.graph_config,
                training_config=self.training_config,
                milestone_config=PRV4MilestoneConfig(milestone_updates=(4,)),
                output_directory=checkpoint_path.parent,
            )

    def test_legacy_schema_v3_trainer_identity_is_unchanged(self):
        legacy = train_pr_history_v3(
            self.history,
            self.transitions,
            graph_config=dataclasses.replace(self.graph_config, architecture_version=3),
            training_config=dataclasses.replace(
                self.training_config,
                steps=1,
                batch_size=1,
                gradient_clip_norm=5.0,
            ),
            device="cpu",
        ).checkpoint
        self.assertEqual(legacy["schema_version"], 3)
        self.assertEqual(legacy["contract"], "pr2901-history-v3-checkpoint-v3")
        self.assertEqual(legacy["state_dict_sha256"], _state_dict_digest(legacy["state_dict"]))
        self.assertNotIn("optimizer_state", legacy)
        self.assertEqual(
            set(legacy["metadata"]),
            {
                "schema_version",
                "contract",
                "history_manifest",
                "static_bundle",
                "transition_selection",
                "predictor_config",
                "training_realized_hierarchy_levels",
                "decoder_work",
                "training_work",
                "training_config",
                "seed_contract",
                "loss_contract",
                "training_log",
                "runtime",
                "source_provenance",
                "source_execution_binding",
                "software",
            },
        )
        self.assertEqual(
            set(legacy["metadata"]["runtime"]),
            {"train_seconds", "device_type", "parameter_count"},
        )


if __name__ == "__main__":
    unittest.main()
