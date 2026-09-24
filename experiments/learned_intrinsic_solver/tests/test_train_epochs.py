# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify epoch counts, exact CPU resume, and failure preservation."""

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep
from experiments.learned_intrinsic_solver.train_epochs import EpochTrainConfig, _validate, run_training


def _equal_states(test, left, right):
    """Compare nested tensors and plain checkpoint values exactly."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _equal_states(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            _equal_states(test, a, b)
    else:
        test.assertEqual(left, right)


class TestEpochTraining(unittest.TestCase):
    def _config(self, epochs):
        return EpochTrainConfig(
            max_epochs=epochs,
            min_epochs=1,
            cell_counts=(2, 2, 3),
            hidden_dim=16,
            edge_hidden_dim=8,
            train_count=2,
            validation_count=2,
            batch_size=1,
            max_step_size=0.01,
            seed=127,
            verbose=False,
            device="cpu",
        )

    def test_epoch_counts_artifacts_and_resume_are_exact(self):
        """Visit each train seed once per epoch and replay Adam and weights exactly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_report = run_training(root / "full", self._config(3))
            first = run_training(root / "split", self._config(1))
            first_final = torch.load(root / "split/checkpoints/final.pt", weights_only=False)
            resumed = run_training(
                root / "split",
                self._config(3),
                resume=root / "split/checkpoints/final.pt",
            )
            full = torch.load(root / "full/checkpoints/final.pt", weights_only=False)
            split = torch.load(root / "split/checkpoints/final.pt", weights_only=False)
            initial = torch.load(root / "split/checkpoints/initial.pt", weights_only=False)
            self.assertEqual(first["completed_epochs"], 1)
            self.assertEqual(
                initial["controller_state"]["best_loss"], first["validation_initial"]["mean_normalized_loss"]
            )
            self.assertEqual(initial["controller_state"]["bad_epochs"], 0)
            self.assertEqual(first_final["controller_state"]["bad_epochs"], 1)
            self.assertEqual(
                first_final["controller_state"]["best_loss"], first["validation_initial"]["mean_normalized_loss"]
            )
            self.assertEqual(resumed["completed_epochs"], 3)
            self.assertEqual(resumed["optimizer_updates"], 6)
            self.assertEqual(initial["optimizer_updates"], 0)
            self.assertEqual([row["train"]["sample_count"] for row in resumed["history"]], [2, 2, 2])
            self.assertEqual([row["validation"]["sample_count"] for row in resumed["history"]], [2, 2, 2])
            self.assertTrue(set(resumed["train_seeds"]).isdisjoint(resumed["validation_seeds"]))
            self.assertEqual(full_report["validation_initial"], resumed["validation_initial"])
            for a, b in zip(full_report["history"], resumed["history"], strict=True):
                self.assertEqual(a["train"], b["train"])
                self.assertEqual(a["validation"], b["validation"])
            _equal_states(self, full["step_state"], split["step_state"])
            _equal_states(self, full["optimizer_state"], split["optimizer_state"])
            self.assertEqual(full["controller_state"], split["controller_state"])
            self.assertEqual(full["rank_states"][0]["dataset_identity"], split["rank_states"][0]["dataset_identity"])
            for name in (
                "initial.pt",
                "latest.pt",
                "best_validation.pt",
                "final.pt",
            ):
                self.assertTrue((root / "split/checkpoints" / name).is_file(), name)
            self.assertTrue((root / "split/data/rank_0.pt").is_file())
            for name in ("report.json", "epochs.csv", "loss_curve.svg", "index.html"):
                self.assertTrue((root / "split" / name).is_file(), name)

    def test_resume_mismatch_preserves_existing_artifacts(self):
        """Reject incompatible material before changing a completed checkpoint."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_training(output, self._config(1))
            checkpoint = output / "checkpoints/final.pt"
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                run_training(
                    output,
                    replace(self._config(3), lame_mu=12345.0),
                    resume=checkpoint,
                )
            self.assertEqual(before, checkpoint.read_bytes())
            with self.assertRaises(FileExistsError):
                run_training(output, self._config(1))

    def test_invalid_learned_proposal_saves_actual_input(self):
        """Stop before Adam and preserve the batch that caused the invalid proposal."""
        original = LearnedHexSolverStep.forward

        def fail_training(step, *args, **kwargs):
            if torch.is_grad_enabled():
                raise ValueError("deliberate invalid learned output")
            return original(step, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(LearnedHexSolverStep, "forward", fail_training):
            output = Path(directory)
            report = run_training(output, self._config(2))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertEqual(report["completed_epochs"], 0)
            self.assertEqual(report["failure"]["stage"], "before_backward")
            self.assertTrue((output / "checkpoints/failure.pt").is_file())
            actual = torch.load(output / "failure_input_rank_0.pt", weights_only=False)
            self.assertEqual(len(actual["physical_seeds"]), 1)
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                run_training(output, self._config(3), resume=output / "checkpoints/failure.pt")

    def test_validation_failure_keeps_complete_denominator(self):
        """Reevaluate a failed batch by query and count one invalid query against all queries."""

        class Step:
            def eval(self):
                pass

            def train(self):
                pass

            def energy(self, positions, inertial):
                return SimpleNamespace(total=10 + positions[:, 0, 0])

            def __call__(self, positions, inertial, *, fixed_positions):
                return SimpleNamespace(
                    positions=positions,
                    loss=SimpleNamespace(total=9 + positions[:, 0, 0]),
                )

        class Dataset:
            def validation_batches(self, batch_size):
                positions = torch.tensor([[[1.0, 0, 0]], [[2.0, 0, 0]]])
                yield {
                    "positions": positions,
                    "inertial_prediction": positions.clone(),
                    "fixed_positions": positions.clone(),
                    "physical_seeds": [10000, 10001],
                    "metadata": [{"physical_seed": 10000}, {"physical_seed": 10001}],
                }

        def reject_second(step, result, pins):
            if result.positions.shape[0] > 1 or result.positions[0, 0, 0] == 2:
                raise ValueError("invalid second query")

        with patch("experiments.learned_intrinsic_solver.train_epochs._screen_output", reject_second):
            result = _validate(Step(), Dataset(), 2, torch.device("cpu"), 1)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["descent_rate"], 0.5)
        self.assertIsNone(result["mean_normalized_loss"])
        self.assertIsNone(result["mean_after_joule"])
        self.assertEqual(result["failures"][0]["physical_seed"], 10001)

        def failed_transfer(batch, device):
            if len(batch["positions"]) > 1 or batch["positions"][0, 0, 0] == 2:
                raise RuntimeError("deliberate device transfer failure")
            return batch

        with (
            patch("experiments.learned_intrinsic_solver.train_epochs._screen_output", reject_second),
            patch("experiments.learned_intrinsic_solver.train_epochs._batch_on_device", failed_transfer),
        ):
            transferred = _validate(Step(), Dataset(), 2, torch.device("cpu"), 1)
        self.assertEqual(transferred["sample_count"], 2)
        self.assertEqual(transferred["valid_count"], 1)
        self.assertEqual(transferred["failed_count"], 1)
        self.assertEqual(transferred["descent_rate"], 0.5)
        self.assertIn("transfer failure", transferred["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
