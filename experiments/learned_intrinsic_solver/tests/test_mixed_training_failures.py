# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise mixed-training rejection, diagnostic retention, and validation cleanup."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import train_mixed
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.initial_state import InitialStateAugmenter
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork


class TestMixedTrainingFailures(unittest.TestCase):
    def _config(self):
        return train_mixed.MixedTrainConfig(
            cell_counts=(1, 1, 1),
            cell_size=0.1,
            hidden_dim=8,
            edge_hidden_dim=4,
            num_heads=2,
            batch_size=1,
            pool_multiplier=2,
            queries_per_epoch=1,
            max_epochs=1,
            iteration_counts=(1,),
            physical_step_counts=(1,),
            validation_count=2,
            validation_iterations=1,
            validation_physical_steps=1,
            validation_physical_iterations=1,
            device="cpu",
            cpu_threads=1,
            preparation_workers=1,
            verbose=False,
            early_stopping=False,
        )

    def _step(self, config):
        rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        network = IntrinsicSolverNetwork(
            rest.cell_counts,
            config.state_feature_dim,
            conditioning_dim=config.conditioning_dim,
            hidden_dim=8,
            edge_hidden_dim=4,
            num_heads=2,
        )
        step = MixedHexSolverStep(rest, fixed, network=network, time_step=config.time_step)
        self.addCleanup(step.close)
        return step, rest

    def test_partial_validation_reset_releases_prepared_contexts(self):
        """Release earlier native contexts when a later validation reset fails."""
        config = self._config()
        step, rest = self._step(config)
        factory = train_mixed._TrajectoryFactory(step, rest, config, rank=0, validation=True)
        reset = factory.reset

        def fail_second_reset(seed):
            if seed == 1:
                raise ValueError("deliberate validation reset failure")
            return reset(seed)

        with patch.object(factory, "reset", fail_second_reset), self.assertRaisesRegex(ValueError, "deliberate"):
            train_mixed._validation_chunk(step, factory, [0, 1], config, torch.device("cpu"))
        self.assertEqual(step.context_specs, {})
        healthy = train_mixed._validation_chunk(step, factory, [0], config, torch.device("cpu"))
        self.assertEqual(healthy[0]["seed"], 0)
        self.assertIsNone(healthy[0]["error"])
        self.assertEqual(step.context_specs, {})

    def test_training_and_validation_use_disjoint_augmentation_seeds(self):
        """Regenerate disjoint shape seeds even when logical pool identities overlap."""
        config = self._config()
        step, rest = self._step(config)
        training = train_mixed._TrajectoryFactory(step, rest, config, rank=0)
        validation = train_mixed._TrajectoryFactory(step, rest, config, rank=0, validation=True)
        for factory, expected_seed in ((training, 146), (validation, 147)):
            with self.subTest(validation=factory is validation):
                payload = factory.reset(73)
                self.assertEqual(payload["seed"], 73)
                self.assertEqual(payload["metadata"]["physical_seed"], expected_seed)
                expected = InitialStateAugmenter(
                    rest,
                    master_seed=factory.master_seed,
                    time_step=config.time_step,
                    material_ranges=config.material_ranges(),
                    strength_range=config.strength_range,
                    velocity_dt_range=config.velocity_dt_range,
                    perturbation_scale_range=config.perturbation_scale_range,
                ).reset(expected_seed)
                np.testing.assert_array_equal(payload["physical_positions"].numpy(), expected.positions)
                np.testing.assert_array_equal(payload["velocities"].numpy(), expected.velocities)
                factory.retire(payload)
        self.assertEqual(step.context_specs, {})

    def test_singular_center_proposal_is_rejected_before_backward(self):
        """Reject finite positive-Gauss proposals that cannot form the next local frame."""
        step, rest = self._step(self._config())
        step.register_context("case", lame_lambda=1000.0, lame_mu=1000.0, density=1000.0)
        positions = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        collapsed = positions.clone()
        collapsed[:, :, 2] *= 1e-8
        loss = step.energy(collapsed, positions, ("case",))
        self.assertTrue(torch.isfinite(loss.total).all())
        batch = {
            "candidate": positions,
            "physical_positions": positions,
            "inertial_prediction": positions,
            "context_ids": ("case",),
            "fixed_positions": positions[:, step.fixed_indices],
        }

        def collapsed_proposal(*args, **kwargs):
            return SimpleNamespace(positions=collapsed, loss=loss)

        with self.assertRaises(ValueError):
            train_mixed._checked_forward(collapsed_proposal, step, batch)
        self.assertTrue(all(parameter.grad is None for parameter in step.network.parameters()))

    def test_failed_training_saves_inputs_weights_and_failed_report(self):
        """Preserve the actual rejected query and optimizer state without recording an update."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(train_mixed, "_checked_forward", side_effect=ValueError("deliberate learned failure")),
        ):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "deliberate learned failure"):
                train_mixed.run_training(output, self._config())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_updates"], 0)
            failure = torch.load(output / "failure_rank_0.pt", weights_only=False)
            initial = torch.load(output / "checkpoints/initial.pt", weights_only=False)
            self.assertIn("deliberate learned failure", failure["error"])
            self.assertEqual(failure["completed_updates"], 0)
            self.assertEqual(len(failure["inputs"]), 1)
            actual = failure["inputs"][0]
            saved = initial["rank_states"][0]["pool"]["records"][0]
            self.assertEqual(actual["seed"], saved["seed"])
            self.assertEqual(actual["inner_iteration"], 0)
            payload = actual["payload"]
            self.assertEqual(failure["context_specs"][payload["context_id"]], payload["context_spec"])
            for key in ("candidate", "physical_positions", "velocities", "inertial_prediction", "fixed_positions"):
                torch.testing.assert_close(payload[key], saved["payload"][key], rtol=0, atol=0)
            for key, weights in initial["network_state"].items():
                torch.testing.assert_close(weights, failure["network_state"][key], rtol=0, atol=0)
            self.assertEqual(failure["optimizer_state"], initial["optimizer_state"])


if __name__ == "__main__":
    unittest.main()
