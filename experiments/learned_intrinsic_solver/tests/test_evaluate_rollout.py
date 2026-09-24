# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify fixed-Y rollout trajectories, batching, and invalid-query accounting."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.evaluate_rollout import _rollout_batch, run_rank
from experiments.learned_intrinsic_solver.train_epochs import EpochTrainConfig, run_training


class TestEvaluateRollout(unittest.TestCase):
    def _config(self):
        return EpochTrainConfig(
            max_epochs=1,
            min_epochs=1,
            cell_counts=(2, 2, 2),
            hidden_dim=16,
            edge_hidden_dim=8,
            train_count=2,
            validation_count=2,
            batch_size=2,
            max_step_size=0.01,
            seed=127,
            verbose=False,
            device="cpu",
        )

    def test_real_fixed_queries_replay_with_batching_and_exact_ratio(self):
        """Keep original physical queries while matching serial and batched learned proposals."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            run_training(training, self._config())
            initial = run_rank(
                training / "checkpoints/initial.pt",
                training / "data",
                root / "initial",
                rank=0,
                world_size=1,
                iterations=3,
                batch_size=2,
                device="cpu",
            )
            with np.load(root / "initial/rank_0.npz") as data:
                initial_energies = data["energies"].copy()
                np.testing.assert_allclose(
                    data["relative_energies"], initial_energies / initial_energies[0], rtol=0, atol=0
                )
                self.assertEqual(data["physical_seeds"].tolist(), [10000, 10001])
            self.assertEqual(initial["checkpoint_epoch"], 0)
            self.assertTrue(initial["parameter_state_unchanged"])
            np.testing.assert_allclose(
                initial_energies, np.broadcast_to(initial_energies[0], initial_energies.shape), rtol=1e-5, atol=1e-5
            )
            checkpoint = training / "checkpoints/final.pt"
            batch = run_rank(
                checkpoint,
                training / "data",
                root / "batch",
                rank=0,
                world_size=1,
                iterations=3,
                batch_size=2,
                device="cpu",
            )
            serial = run_rank(
                checkpoint,
                training / "data",
                root / "serial",
                rank=0,
                world_size=1,
                iterations=3,
                batch_size=1,
                device="cpu",
            )
            with np.load(root / "batch/rank_0.npz") as batched, np.load(root / "serial/rank_0.npz") as one_by_one:
                np.testing.assert_array_equal(batched["physical_seeds"], one_by_one["physical_seeds"])
                np.testing.assert_allclose(batched["energies"], one_by_one["energies"], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(
                    batched["relative_energies"], one_by_one["relative_energies"], rtol=1e-5, atol=1e-5
                )
            self.assertEqual(batch["checkpoint_epoch"], 1)
            self.assertEqual(batch["sample_count"], 2)
            self.assertEqual(batch["failures"], [])
            self.assertEqual(serial["failures"], [])
            with self.assertRaises(FileExistsError):
                run_rank(
                    checkpoint,
                    training / "data",
                    root / "batch",
                    rank=0,
                    world_size=1,
                    iterations=3,
                    batch_size=2,
                    device="cpu",
                )

    def test_invalid_query_stops_at_first_failure_and_preserves_survivor(self):
        """Split a failing batch and leave future energies missing only for the failed query."""

        class Step:
            def energy(self, positions, inertial):
                return SimpleNamespace(total=10 + positions[:, 0, 0])

            def __call__(self, positions, inertial, *, fixed_positions):
                if (positions[:, 0, 0] >= 5).any():
                    raise ValueError("deliberate invalid proposal")
                proposed = positions + 1
                return SimpleNamespace(positions=proposed, loss=SimpleNamespace(total=10 + proposed[:, 0, 0]))

        positions = torch.tensor([[[1.0, 0, 0]], [[4.0, 0, 0]]])
        batch = {
            "positions": positions,
            "inertial_prediction": positions.clone(),
            "fixed_positions": positions.clone(),
            "physical_seeds": [10000, 10001],
        }
        with patch("experiments.learned_intrinsic_solver.train_epochs._screen_output"):
            seeds, energies, relative, failures = _rollout_batch(Step(), batch, 3, torch.device("cpu"))
        self.assertEqual(seeds.tolist(), [10000, 10001])
        self.assertEqual(energies[:, 0].tolist(), [11, 12, 13, 14])
        self.assertEqual(energies[:2, 1].tolist(), [14, 15])
        self.assertTrue(np.isnan(energies[2:, 1]).all())
        self.assertTrue(np.isnan(relative[2:, 1]).all())
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["physical_seed"], 10001)
        self.assertEqual(failures[0]["iteration"], 2)


if __name__ == "__main__":
    unittest.main()
