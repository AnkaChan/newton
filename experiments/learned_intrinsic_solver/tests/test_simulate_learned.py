# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for saved-state native simulation and honest step failures."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.simulate_learned import run_cases, select_validation_seeds
from experiments.learned_intrinsic_solver.train_epochs import EpochTrainConfig, run_training


class TestSimulateLearned(unittest.TestCase):
    def _config(self):
        return EpochTrainConfig(
            max_epochs=1,
            min_epochs=1,
            cell_counts=(2, 2, 2),
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

    def test_saved_velocity_and_two_iterations_per_physical_step(self):
        """Load saved X and V, then retain initial and sampled native states."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            run_training(training, self._config())
            checkpoint = training / "checkpoints/initial.pt"
            duration = 2 * self._config().time_step
            reports = run_cases(
                checkpoint, root / "simulation", seeds=[10000], duration=duration, fps=300, device="cpu"
            )
            report = reports[0]
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["completed_physical_steps"], 2)
            self.assertEqual(report["actual_optimizer_iteration_calls"], 4)
            self.assertEqual(report["optimizer_iterations_per_step"], 2)
            self.assertEqual(report["actual_duration_seconds"], duration)
            with np.load(root / "simulation/seed_10000/trajectory.npz") as trajectory:
                self.assertEqual(trajectory["positions"].shape[0], 3)
                np.testing.assert_allclose(trajectory["times"], [0, duration / 2, duration], rtol=0, atol=1e-12)
                fixed = trajectory["fixed_indices"]
                np.testing.assert_array_equal(
                    trajectory["positions"][:, fixed],
                    np.broadcast_to(trajectory["positions"][0, fixed], trajectory["positions"][:, fixed].shape),
                )
                data = torch.load(training / "data/rank_0.pt", weights_only=False)
                np.testing.assert_array_equal(trajectory["positions"][0], data["physical_samples"][10000]["positions"])
            self.assertGreater(report["initial_velocity_rms_m_per_s"], 0)
            with self.assertRaises(FileExistsError):
                run_cases(checkpoint, root / "simulation", seeds=[10000], duration=duration, fps=300, device="cpu")

    def test_failed_second_proposal_keeps_initial_state(self):
        """Count attempted learned calls and stop before committing an invalid physical step."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            run_training(training, self._config())
            original = SolverLearnedIntrinsic.propose_update
            attempted = 0

            def fail_second(solver, *args, **kwargs):
                nonlocal attempted
                attempted += 1
                if attempted == 2:
                    raise ValueError("deliberate invalid second proposal")
                return original(solver, *args, **kwargs)

            with patch.object(SolverLearnedIntrinsic, "propose_update", fail_second):
                report = run_cases(
                    training / "checkpoints/initial.pt",
                    root / "failed",
                    seeds=[10000],
                    duration=2 * self._config().time_step,
                    fps=300,
                    device="cpu",
                )[0]
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_physical_steps"], 0)
            self.assertEqual(report["actual_optimizer_iteration_calls"], 2)
            self.assertEqual(report["failure"]["physical_step"], 1)
            self.assertEqual(report["failure"]["time_seconds"], 0)
            with np.load(root / "failed/seed_10000/trajectory.npz") as trajectory:
                self.assertEqual(trajectory["positions"].shape[0], 1)
                self.assertEqual(trajectory["times"].tolist(), [0])

    def test_default_seed_selection_is_stable(self):
        """Select the ten requested held-out physical identities without replacement."""
        config = EpochTrainConfig()
        selected = select_validation_seeds(config)
        self.assertEqual(selected, (10015, 10112, 10136, 10153, 10248, 10356, 10395, 10411, 10488, 10502))
        self.assertEqual(len(set(selected)), 10)


if __name__ == "__main__":
    unittest.main()
