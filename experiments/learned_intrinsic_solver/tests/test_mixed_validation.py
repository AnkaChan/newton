# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check partial validation trajectories and absolute near-equilibrium diagnostics."""

import importlib.util
import unittest
from types import SimpleNamespace

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.mixed_validation import validate, validation_chunk


class _AnalyticStep(torch.nn.Module):
    """Provide exact quadratic energies while using the real proposal acceptance gate."""

    def __init__(self, *, fail_iteration=None, fail_physical=False, pin_target=0.0):
        super().__init__()
        self.contexts = {}
        self.batch_sizes = []
        self.fail_iteration = fail_iteration
        self.fail_physical = fail_physical
        self.pin_target = pin_target
        self.register_buffer("fixed_indices", torch.tensor([0]))
        self.register_buffer("cell_corner_indices", torch.arange(8).reshape(1, 8))
        self.register_buffer(
            "rest", torch.tensor([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
        )
        self.register_buffer("center_gradients", (2 * self.rest - 1) / 4)
        self.network = torch.nn.Linear(1, 1)

    def energy(self, positions, inertial_prediction, context_ids, *, previous_positions=None):
        free = positions[:, 7, 0] - 1
        pinned = positions[:, 0, 0] - self.pin_target
        return SimpleNamespace(total=free.square() + pinned.square())

    def forward(self, positions, inertial_prediction, context_ids, *, fixed_positions, previous_positions=None):
        self.batch_sizes.append(len(context_ids))
        for identity in context_ids:
            context = self.contexts[identity]
            context["iteration"] += 1
            if self.fail_iteration == context["iteration"]:
                raise ValueError("deliberate optimizer iteration failure")
            if self.fail_physical and context["physical_step"] == 1:
                raise ValueError("deliberate physical rollout failure")
        proposed = positions.clone()
        proposed[:, 7, 0] = 1 + (proposed[:, 7, 0] - 1) / 2
        return SimpleNamespace(positions=proposed, loss=self.energy(proposed, inertial_prediction, context_ids))


class _Factory:
    """Track real validation ownership without allocating native physical contexts."""

    def __init__(self, step, *, offset=2.0, fail_reset_seed=None):
        self.step = step
        self.offset = offset
        self.fail_reset_seed = fail_reset_seed

    def reset(self, seed):
        if seed == self.fail_reset_seed:
            raise ValueError("deliberate reset failure")
        identity = str(seed)
        if identity in self.step.contexts:
            raise ValueError("context already registered")
        self.step.contexts[identity] = {"iteration": 0, "physical_step": 0}
        positions = self.step.rest.clone()
        positions[7, 0] += self.offset
        return {
            "context_id": identity,
            "candidate": positions,
            "inertial_prediction": positions.clone(),
            "physical_positions": positions.clone(),
            "fixed_positions": positions[self.step.fixed_indices].clone(),
        }

    def advance(self, payload):
        context = self.step.contexts[payload["context_id"]]
        context["iteration"] = 0
        context["physical_step"] += 1
        return dict(payload)

    def retire(self, payload):
        self.step.contexts.pop(payload["context_id"])


class TestMixedValidation(unittest.TestCase):
    def _config(self, *, count=1, batch_size=1):
        return SimpleNamespace(
            validation_count=count,
            batch_size=batch_size,
            validation_iterations=3,
            validation_physical_steps=2,
            validation_physical_iterations=1,
        )

    def test_late_optimization_failure_preserves_first_query_and_physical_success(self):
        """Keep early population curve points and independent physical survival after iteration two fails."""
        step = _AnalyticStep(fail_iteration=2)
        report = validate(step, _Factory(step), self._config(), torch.device("cpu"), 0, 1)
        self.assertEqual(report["relative_energy"][0]["mean"], 1.0)
        self.assertEqual(report["relative_energy"][1]["mean"], 0.25)
        self.assertIsNone(report["relative_energy"][2]["mean"])
        self.assertEqual([point["failed_count"] for point in report["relative_energy"]], [0, 0, 1, 1])
        self.assertEqual(report["mean_normalized_loss"], -0.75)
        self.assertEqual(report["first_update_failed_count"], 0)
        self.assertEqual(report["optimization_failed_count"], 1)
        self.assertEqual(report["physical_failed_count"], 0)
        self.assertEqual(report["physical_survivors"], 1)
        self.assertEqual(report["failed_count"], 1)
        sample = report["samples"][0]
        self.assertEqual(sample["energies"], [4.0, 1.0])
        self.assertEqual(sample["failure_iteration"], 2)
        self.assertEqual(sample["physical_steps"], 2)
        self.assertIsNone(sample["physical_error"])
        self.assertEqual(step.contexts, {})

    def test_physical_failure_preserves_complete_optimization_curve(self):
        """Count physical failures separately while retaining all frozen-solve energies."""
        step = _AnalyticStep(fail_physical=True)
        report = validate(step, _Factory(step), self._config(), torch.device("cpu"), 0, 1)
        self.assertEqual([point["mean"] for point in report["relative_energy"]], [1.0, 0.25, 0.0625, 0.015625])
        self.assertEqual(report["optimization_failed_count"], 0)
        self.assertEqual(report["physical_failed_count"], 1)
        self.assertEqual(report["physical_survivors"], 0)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["mean_normalized_loss"], -0.75)
        sample = report["samples"][0]
        self.assertIsNone(sample["error"])
        self.assertIn("physical rollout", sample["physical_error"])
        self.assertEqual(sample["physical_steps"], 1)
        self.assertEqual(step.contexts, {})

    def test_near_zero_queries_keep_absolute_energy_displacement_and_free_residual(self):
        """Exclude unstable ratios and pinned forces while differentiating only the physical energy."""
        step = _AnalyticStep(pin_target=5e-5)
        report = validate(step, _Factory(step, offset=1e-5), self._config(), torch.device("cpu"), 0, 1)
        self.assertIn("near_zero", report)
        self.assertEqual(report["near_zero"]["sample_count"], 1)
        self.assertTrue(all(point["mean"] is None for point in report["relative_energy"]))
        points = report["near_zero"]["curves"]
        self.assertAlmostEqual(points[0]["mean_energy_joule"], 2.6e-9, delta=1e-11)
        self.assertEqual(points[0]["mean_displacement_rms_m"], 0)
        self.assertGreater(points[1]["mean_displacement_rms_m"], 0)
        self.assertAlmostEqual(points[0]["mean_free_force_residual_norm_n"], 2e-5, delta=1e-7)
        self.assertLess(points[-1]["mean_free_force_residual_norm_n"], points[0]["mean_free_force_residual_norm_n"])
        self.assertTrue(all(parameter.grad is None for parameter in step.network.parameters()))
        self.assertEqual(len(step.batch_sizes), 5)
        self.assertEqual(step.contexts, {})

    def test_partial_batch_reset_cleans_up_and_retries_healthy_member(self):
        """Isolate a failed reset while cleaning every prepared context and retaining both seeds."""
        step = _AnalyticStep()
        factory = _Factory(step, fail_reset_seed=1)
        config = self._config(count=2, batch_size=2)
        with self.assertRaisesRegex(ValueError, "reset failure"):
            validation_chunk(step, factory, [0, 1], config, torch.device("cpu"))
        self.assertEqual(step.contexts, {})
        report = validate(step, factory, config, torch.device("cpu"), 0, 1)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["first_update_failed_count"], 1)
        self.assertEqual(report["physical_survivors"], 1)
        self.assertIsNone(report["mean_normalized_loss"])
        self.assertEqual([sample["seed"] for sample in report["samples"]], [0, 1])
        self.assertIsNone(report["samples"][0]["error"])
        self.assertIn("reset failure", report["samples"][1]["error"])
        self.assertEqual(step.contexts, {})

    def test_successful_validation_keeps_batched_proposals_and_original_mode(self):
        """Evaluate successful groups together and restore the caller's module mode."""
        step = _AnalyticStep().eval()
        report = validate(step, _Factory(step), self._config(count=2, batch_size=2), torch.device("cpu"), 0, 1)
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(step.batch_sizes, [2, 2, 2, 2, 2])
        self.assertFalse(step.training)
        self.assertEqual(step.contexts, {})


if __name__ == "__main__":
    unittest.main()
