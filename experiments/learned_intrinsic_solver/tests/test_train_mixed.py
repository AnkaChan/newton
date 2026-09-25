# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate detached mixed-query training and exact pool continuation."""

import importlib.util
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253 -- Optional experimental training tests.

from experiments.learned_intrinsic_solver.train_mixed import MixedTrainConfig, local_objective, run_training


class TestMixedTraining(unittest.TestCase):
    """Check the optimizer contract rather than a copied implementation."""

    def test_local_loss_cuts_history_and_keeps_each_member_gradient(self):
        """Only the current proposal receives gradients, with no hidden 1/K scale."""
        initial = torch.tensor([2.0, 0.2], requires_grad=True)
        previous = torch.tensor([1.5, 0.1], requires_grad=True)
        after = torch.tensor([1.0, 0.3], requires_grad=True)
        loss = local_objective(after, initial, previous, increase_weight=1.0).mean()
        loss.backward()
        self.assertIsNone(initial.grad)
        self.assertIsNone(previous.grad)
        torch.testing.assert_close(after.grad, torch.tensor([0.25, 1.0]))
        self.assertAlmostEqual(float(loss.detach()), -0.1, places=6)

    @staticmethod
    def config(epochs):
        """Use a tiny real hex problem with physical and inner transitions."""
        return MixedTrainConfig(
            cell_counts=(1, 1, 2),
            cell_size=0.1,
            hidden_dim=8,
            edge_hidden_dim=4,
            num_heads=2,
            batch_size=2,
            pool_multiplier=2,
            queries_per_epoch=8,
            max_epochs=epochs,
            stage_epochs=1,
            iteration_counts=(1, 2),
            physical_step_counts=(1, 2),
            validation_count=2,
            validation_iterations=3,
            validation_physical_steps=2,
            validation_physical_iterations=2,
            device="cpu",
            cpu_threads=1,
            preparation_workers=1,
            verbose=False,
            early_stopping=False,
        )

    def test_exact_resume_preserves_updates_and_active_trajectories(self):
        """A restart reproduces Adam and independently progressing pool members."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "experiments.learned_intrinsic_solver.curriculum.MixedCurriculum.available_counts",
                new=property(lambda self: ((1, 2), (1, 2))),
            ),
        ):
            root = Path(directory)
            full = run_training(root / "full", self.config(2))
            run_training(root / "split", self.config(1))
            resumed = run_training(root / "split", self.config(2), resume=root / "split/checkpoints/latest.pt")
            a = torch.load(root / "full/checkpoints/latest.pt", weights_only=False)
            b = torch.load(root / "split/checkpoints/latest.pt", weights_only=False)
            for key in a["network_state"]:
                torch.testing.assert_close(a["network_state"][key], b["network_state"][key], rtol=0, atol=0)
            for identity, state in a["optimizer_state"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, b["optimizer_state"]["state"][identity][key], rtol=0, atol=0)
            self.assertEqual(a["curriculum_state"], b["curriculum_state"])
            self.assertEqual(a["controller_state"], b["controller_state"])
            self.assertEqual(full["updates"], resumed["updates"])
            self.assertEqual(full["completed_updates"], 8)
            self.assertEqual(full["epochs"][-1]["query_count"], 8)
            self.assertEqual(full["epochs"][-1]["validation"]["sample_count"], 2)
            self.assertEqual(len(full["epochs"][-1]["validation"]["relative_energy"]), 4)
            self.assertEqual(a["rank_states"][0]["pool"]["next_seed"], b["rank_states"][0]["pool"]["next_seed"])
            for name in ("report.json", "updates.csv", "epochs.csv", "loss_curve.svg", "index.html"):
                self.assertTrue((root / "split" / name).is_file(), name)

    def test_resume_rejects_material_or_architecture_change_before_writing(self):
        """A checkpoint cannot silently change its physical problem or model."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_training(output, self.config(1))
            checkpoint = output / "checkpoints/latest.pt"
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                run_training(output, replace(self.config(2), time_step=0.01), resume=checkpoint)
            self.assertEqual(before, checkpoint.read_bytes())

    def test_resume_updates_only_descent_gate_and_records_effective_boundary(self):
        """Change the gate without rewriting prior decisions or resetting training state."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original_config = replace(self.config(1), stage_descent_rate=0.9)
            run_training(output, original_config)
            checkpoint = output / "checkpoints/latest.pt"
            saved = torch.load(checkpoint, weights_only=False)
            resumed = run_training(output, replace(original_config, stage_descent_rate=0.8), resume=checkpoint)
            restored = torch.load(output / "checkpoints/final.pt", weights_only=False)

            def assert_same(expected, actual):
                if isinstance(expected, torch.Tensor):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                elif isinstance(expected, np.ndarray):
                    np.testing.assert_array_equal(expected, actual)
                elif isinstance(expected, dict):
                    self.assertEqual(expected.keys(), actual.keys())
                    for name, value in expected.items():
                        assert_same(value, actual[name])
                elif isinstance(expected, (tuple, list)):
                    self.assertEqual(len(expected), len(actual))
                    for left, right in zip(expected, actual, strict=True):
                        assert_same(left, right)
                else:
                    self.assertEqual(expected, actual)

            for name in ("network_state", "optimizer_state", "controller_state", "rank_states"):
                assert_same(saved[name], restored[name])
            self.assertEqual(restored["curriculum_state"], {**saved["curriculum_state"], "min_descent_rate": 0.8})
            self.assertEqual(resumed["epochs"], saved["report"]["epochs"])
            self.assertEqual(resumed["updates"], saved["report"]["updates"])
            self.assertEqual(
                resumed["configuration_changes"],
                [
                    {
                        "field": "stage_descent_rate",
                        "previous": 0.9,
                        "current": 0.8,
                        "effective_from_epoch": 2,
                        "completed_updates": saved["report"]["completed_updates"],
                        "source": "checkpoint_resume",
                    }
                ],
            )
            self.assertEqual(restored["config"]["stage_descent_rate"], 0.8)

    def test_default_descent_gate_is_eighty_percent(self):
        """Use the user-selected descent gate for newly configured campaigns."""
        self.assertEqual(MixedTrainConfig().stage_descent_rate, 0.8)

    def test_stage_limit_defaults_and_legacy_configuration(self):
        """Give new campaigns twenty-epoch caps while old checkpoints remain uncapped."""
        self.assertEqual(MixedTrainConfig().stage_max_epochs, 20)
        legacy = asdict(self.config(1))
        legacy.pop("stage_max_epochs")
        self.assertIsNone(MixedTrainConfig.from_checkpoint_config(legacy).stage_max_epochs)
        for value in (0, 9, True, 20.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MixedTrainConfig(stage_max_epochs=value)

    def test_resume_applies_overdue_cap_without_new_validation(self):
        """Promote once at resume while leaving checkpoint history and active budgets intact."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = replace(self.config(1), stage_max_epochs=None, stage_descent_rate=1.0)
            run_training(output, config)
            checkpoint = output / "checkpoints/latest.pt"
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(saved["curriculum_state"]["stage"], 0)
            report = run_training(output, replace(config, stage_max_epochs=1), resume=checkpoint)
            restored = torch.load(output / "checkpoints/final.pt", weights_only=False)
            self.assertEqual(restored["curriculum_state"]["stage"], 1)
            self.assertEqual(restored["curriculum_state"]["stage_epochs"], 0)
            self.assertEqual(report["epochs"], saved["report"]["epochs"])
            self.assertEqual(report["updates"], saved["report"]["updates"])
            self.assertEqual(report["configuration_changes"][0]["field"], "stage_max_epochs")
            self.assertEqual(
                report["curriculum_events"],
                [
                    {
                        "source": "checkpoint_resume",
                        "advance_reason": "max_stage_epochs",
                        "previous_stage": 0,
                        "stage": 1,
                        "previous_stage_epochs": 1,
                        "effective_from_epoch": 2,
                        "completed_updates": saved["report"]["completed_updates"],
                    }
                ],
            )
            original_pool = saved["rank_states"][0]["pool"]
            resumed_pool = restored["rank_states"][0]["pool"]
            self.assertEqual(original_pool["iteration_counts"], resumed_pool["iteration_counts"])
            self.assertEqual(original_pool["physical_step_counts"], resumed_pool["physical_step_counts"])


if __name__ == "__main__":
    unittest.main()
