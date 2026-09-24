# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate detached mixed-query training and exact pool continuation."""

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
