# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check actual CPU training updates, fixed holdout data, and exact resume."""

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.train_smoke import TrainSmokeConfig, run_training


class TestTrainSmoke(unittest.TestCase):
    def _config(self, updates):
        return TrainSmokeConfig(
            updates=updates,
            cell_counts=(2, 2, 3),
            hidden_dim=16,
            edge_hidden_dim=8,
            train_count=2,
            validation_count=2,
            validation_interval=2,
            checkpoint_interval=2,
            max_step_size=0.01,
            seed=127,
            verbose=False,
            device="cpu",
        )

    def test_training_changes_weights_and_preserves_holdout_split(self):
        """Perform real optimizer steps and save losses, weights, and disjoint sample identities."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run_training(output, self._config(2))
            initial = torch.load(output / "checkpoints/initial.pt", weights_only=False)
            final = torch.load(output / "checkpoints/final.pt", weights_only=False)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(initial["optimizer_updates"], 0)
            self.assertEqual(final["optimizer_updates"], 2)
            self.assertEqual(report["failure_count"], 0)
            self.assertFalse(
                torch.equal(
                    initial["network_state"]["correction_head.weight"], final["network_state"]["correction_head.weight"]
                )
            )
            self.assertTrue(final["optimizer_state"]["state"])
            self.assertTrue(set(report["train_seeds"]).isdisjoint(report["validation_seeds"]))
            self.assertEqual({entry["physical_seed"] for entry in report["training"]}, set(report["train_seeds"]))
            self.assertEqual(report["validation"][0]["optimizer_updates"], 0)
            self.assertEqual(report["validation"][0]["mean_normalized_loss"], 0)
            for entry in report["validation"]:
                self.assertEqual({case["physical_seed"] for case in entry["cases"]}, set(report["validation_seeds"]))
            for name in [
                "training.csv",
                "validation.csv",
                "loss_curve.png",
                "loss_curve.svg",
                "index.html",
                "report.json",
            ]:
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue(
                all(
                    value.dtype == torch.float32
                    for value in final["network_state"].values()
                    if value.is_floating_point()
                )
            )

    def test_resume_matches_uninterrupted_updates(self):
        """Restore optimizer and sampler RNG after reconstruction to match uninterrupted training."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_report = run_training(root / "full", self._config(3))
            run_training(root / "split", self._config(1))
            resumed = run_training(root / "split", self._config(3), resume=root / "split/checkpoints/final.pt")
            full = torch.load(root / "full/checkpoints/final.pt", weights_only=False)
            split = torch.load(root / "split/checkpoints/final.pt", weights_only=False)
            self.assertEqual(split["optimizer_updates"], 3)
            for key, value in full["network_state"].items():
                torch.testing.assert_close(value, split["network_state"][key], rtol=0, atol=0)
            for parameter_id, values in full["optimizer_state"]["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(
                        value, split["optimizer_state"]["state"][parameter_id][key], rtol=0, atol=0
                    )
            for a, b in zip(full_report["training"], resumed["training"], strict=True):
                for key in [
                    "physical_seed",
                    "candidate_mode",
                    "noise_seed",
                    "energy_before_joule",
                    "energy_after_joule",
                    "normalized_loss",
                ]:
                    self.assertEqual(a[key], b[key], key)
            self.assertEqual(full["sampler_state"], split["sampler_state"])

    def test_invalid_training_output_stops_without_fake_update(self):
        """Save an explicit failure checkpoint instead of repairing a learned proposal."""
        original = SolverLearnedIntrinsic.propose_update

        def invalid_training(solver, *args, **kwargs):
            if torch.is_grad_enabled():
                raise ValueError("Deliberate invalid learned Jacobian")
            return original(solver, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(SolverLearnedIntrinsic, "propose_update", invalid_training),
        ):
            report = run_training(Path(directory), self._config(2))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertEqual(report["failure_count"], 1)
            self.assertTrue((Path(directory) / "checkpoints/failure.pt").is_file())
            self.assertIn("Deliberate invalid learned Jacobian", report["failure"]["error"])

    def test_overlapping_seed_pools_are_rejected(self):
        """Reject a validation pool that overlaps the fixed physical training seeds."""
        with self.assertRaises(ValueError):
            replace(self._config(1), validation_seed_start=1)


if __name__ == "__main__":
    unittest.main()
