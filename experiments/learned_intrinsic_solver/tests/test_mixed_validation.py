# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check residual curves, selection eligibility, history carry and full-horizon validation."""

import importlib.util
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import history, mixed_validation, train_mixed
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.mixed_validation import (
    SELECTION_AGGREGATION,
    _full_horizon_seeds,
    validate,
    validate_full_horizon,
    validation_chunk,
)
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork

_FLOOR = 1e-6
_INERTIAL_SHIFT = (0.0, 0.1, 0.0)
"""Offset of the fixture inertial prediction from the physical start so anchor checks can tell them apart."""


class _AnalyticStep(torch.nn.Module):
    """Provide exact quadratic energies on one unit cell while using the real trainer boundary.

    The single free corner is corner 7; every other corner is treated as
    stationary and corner 0 is prescribed. Moving corner 7 by ``d`` along x
    gives ``E = d^2 + (x0 - pin)^2``, a free-corner residual of ``2|d|`` and a
    center Jacobian of ``1 + d / 4``. Each query halves ``d``.
    """

    def __init__(self, *, fail_iteration=None, fail_physical=False, fail_physical_contexts=(), pin_target=0.0):
        super().__init__()
        self.contexts = {}
        self.batch_sizes = []
        self.history_flags = []
        self.history_markers = []
        self.fail_iteration = fail_iteration
        self.fail_physical = fail_physical
        self.fail_physical_contexts = set(fail_physical_contexts)
        self.pin_target = pin_target
        self.energy_calls = []
        self.register_buffer("fixed_indices", torch.tensor([0]))
        self.register_buffer("cell_corner_indices", torch.arange(8).reshape(1, 8))
        self.register_buffer(
            "rest", torch.tensor([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
        )
        self.register_buffer("center_gradients", (2 * self.rest - 1) / 4)
        self.network = torch.nn.Linear(1, 1)

    def _quadratic(self, positions):
        free = positions[:, 7, 0] - 1
        pinned = positions[:, 0, 0] - self.pin_target
        return SimpleNamespace(total=free.square() + pinned.square())

    def energy(self, positions, inertial_prediction, context_ids, *, previous_positions=None):
        """Record the operands the validator differentiates; the quadratic itself ignores the anchor."""
        self.energy_calls.append(
            SimpleNamespace(
                context_ids=tuple(context_ids),
                positions=positions.detach().clone(),
                inertial_prediction=inertial_prediction.detach().clone(),
                anchor=None if previous_positions is None else previous_positions.detach().clone(),
            )
        )
        return self._quadratic(positions)

    def energy_floor(self, context_ids):
        return torch.full((len(context_ids),), _FLOOR)

    def forward(
        self, positions, inertial_prediction, context_ids, *, fixed_positions, previous_positions, history=None
    ):
        self.batch_sizes.append(len(context_ids))
        self.history_flags.append(None if history is None else history.valid.tolist())
        self.history_markers.append(None if history is None else history.axis_gradient_world[:, 0, 0, 0].tolist())
        markers = []
        for identity in context_ids:
            context = self.contexts[identity]
            context["iteration"] += 1
            markers.append(float(context["iteration"]))
            if self.fail_iteration == context["iteration"]:
                raise ValueError("deliberate optimizer iteration failure")
            if (self.fail_physical or identity in self.fail_physical_contexts) and context["physical_step"] == 1:
                raise ValueError("deliberate physical rollout failure")
        proposed = positions.clone()
        proposed[:, 7, 0] = 1 + (proposed[:, 7, 0] - 1) / 2
        gradient = torch.zeros(len(context_ids), 1, 3, 3)
        gradient[:, 0, 0, 0] = torch.tensor(markers)
        return SimpleNamespace(
            positions=proposed,
            loss=self._quadratic(proposed),
            axis_gradient_world=gradient,
            achieved_axis_update_world=torch.zeros_like(gradient),
        )


class _Factory:
    """Track real validation ownership without allocating native physical contexts."""

    def __init__(self, step, *, offset=2.0, offsets=None, fail_reset_seed=None):
        self.step = step
        self.offset = offset
        self.offsets = dict(offsets or {})
        self.fail_reset_seed = fail_reset_seed
        self.advanced_history = []
        self.physical_starts = {}

    def reset(self, seed):
        if seed == self.fail_reset_seed:
            raise ValueError("deliberate reset failure")
        identity = str(seed)
        if identity in self.step.contexts:
            raise ValueError("context already registered")
        self.step.contexts[identity] = {"iteration": 0, "physical_step": 0}
        positions = self.step.rest.clone()
        positions[7, 0] += self.offsets.get(seed, self.offset)
        self.physical_starts[identity] = positions.clone()
        return {
            "context_id": identity,
            "candidate": positions,
            "inertial_prediction": positions + torch.tensor(_INERTIAL_SHIFT),
            "physical_positions": positions.clone(),
            "fixed_positions": positions[self.step.fixed_indices].clone(),
            "metadata": {"perturbation_scale": 0.3},
            "candidate_mode": "inertial",
            **history.empty_history(1),
        }

    def advance(self, payload):
        context = self.step.contexts[payload["context_id"]]
        context["iteration"] = 0
        context["physical_step"] += 1
        prepared = {
            key: payload[key]
            for key in (
                "context_id",
                "candidate",
                "inertial_prediction",
                "physical_positions",
                "fixed_positions",
                "metadata",
                "candidate_mode",
            )
        }
        history.carry_history(payload, prepared)
        self.advanced_history.append(bool(prepared.get("history_valid", False)))
        return prepared

    def retire(self, payload):
        self.step.contexts.pop(payload["context_id"])


def _config(*, count=1, batch_size=1, full_count=0, iterations=3, physical_steps=2, physical_iterations=1):
    return SimpleNamespace(
        validation_count=count,
        validation_full_count=full_count,
        batch_size=batch_size,
        validation_iterations=iterations,
        validation_physical_steps=physical_steps,
        validation_physical_iterations=physical_iterations,
    )


def _reference_residual_norms(step, batch, anchor):
    """Return float64 free-corner residual norms [N] of the objective at the candidate with ``anchor`` as damping anchor."""
    with torch.enable_grad():
        positions = batch["candidate"].detach().clone().requires_grad_(True)
        total = step.energy(
            positions, batch["inertial_prediction"].detach(), batch["context_ids"], previous_positions=anchor.detach()
        ).total
        gradient = torch.autograd.grad(total.sum(), positions)[0]
    gradient = gradient.detach().clone()
    gradient[:, step.fixed_indices] = 0
    return torch.linalg.vector_norm(gradient.flatten(1).double(), dim=1).cpu().tolist()


class TestMixedValidation(unittest.TestCase):
    def test_residual_curve_and_diagnostics_recorded_for_every_sample_at_every_iteration(self):
        """Record residual, inversion and displacement for all samples and build the selection record."""
        step = _AnalyticStep()
        report = validate(step, _Factory(step), _config(count=2, batch_size=2), torch.device("cpu"), 0, 1)
        for sample in report["samples"]:
            self.assertEqual(sample["free_force_residual_norm_n"], [4.0, 2.0, 1.0, 0.5])
            self.assertEqual(sample["energies"], [4.0, 1.0, 0.25, 0.0625])
            self.assertEqual(sample["inverted_cell_counts"], [0, 0, 0, 0])
            self.assertEqual(len(sample["min_center_jacobians"]), 4)
            self.assertEqual(sample["perturbation_scale"], 0.3)
            self.assertEqual(sample["candidate_mode"], "inertial")
            self.assertAlmostEqual(sample["energy_floor_joule"], _FLOOR, places=12)
            self.assertEqual(len(sample["physical_records"]), 2)
        self.assertEqual([point["iteration"] for point in report["force_residual"]], [0, 1, 2, 3])
        self.assertEqual([point["mean"] for point in report["force_residual"]], [4.0, 2.0, 1.0, 0.5])
        self.assertEqual([point["max"] for point in report["force_residual"]], [4.0, 2.0, 1.0, 0.5])
        self.assertTrue(
            all(point["valid_count"] == 2 and point["failed_count"] == 0 for point in report["force_residual"])
        )
        self.assertEqual(
            report["selection"],
            {"metric": 0.5, "eligible": True, "aggregation": SELECTION_AGGREGATION, "survival_required": True},
        )
        self.assertEqual(SELECTION_AGGREGATION, "mean_final_free_force_residual_norm_n")
        self.assertAlmostEqual(report["mean_normalized_loss"], math.asinh(0.25), places=12)
        self.assertEqual(report["final_inverted_sample_count"], 0)
        self.assertEqual(len(report["inversion"]), 4)
        self.assertEqual(report["inversion"][0]["inverted_sample_count"], 0)
        self.assertEqual(report["inversion"][0]["max_inverted_cell_count"], 0)
        self.assertAlmostEqual(report["inversion"][0]["min_center_jacobian"], 1.5, places=6)
        self.assertEqual(report["physical_survivors"], 2)
        self.assertEqual([point["step"] for point in report["physical_curves"]], [1, 2])
        for point in report["physical_curves"]:
            self.assertEqual(point["valid_count"], 2)
            self.assertEqual(point["failed_count"], 0)
            self.assertGreater(point["free_force_residual_norm_n"]["mean"], 0)
            self.assertGreater(point["displacement_rms_m"]["mean"], 0)
            self.assertEqual(point["inverted_sample_count"], 0)
        self.assertEqual(report["physical_curves"][0]["energy_joule"]["mean"], 1.0)
        self.assertEqual(report["physical_curves"][0]["free_force_residual_norm_n"]["mean"], 2.0)
        self.assertAlmostEqual(report["physical_curves"][0]["displacement_rms_m"]["mean"], math.sqrt(1 / 8), places=9)
        self.assertEqual(report["final_physical_residual"], report["physical_curves"][-1]["free_force_residual_norm_n"])
        self.assertGreaterEqual(report["seconds"], 0.0)
        for removed in ("optimization_shortened_query_count", "physical_shortened_query_count", "acceptance_scales"):
            self.assertNotIn(removed, report)
            self.assertNotIn(removed, report["samples"][0])
        self.assertEqual(step.contexts, {})

    def test_residual_anchor_is_the_physical_step_start_at_every_observation(self):
        """Differentiate every recorded residual and energy with the unchanged physical start as the anchor."""
        step = _AnalyticStep()
        factory = _Factory(step)
        validate(step, factory, _config(count=2, batch_size=2), torch.device("cpu"), 0, 1)
        # One initial energy plus four residual observations, then one record per physical step.
        self.assertEqual(len(step.energy_calls), 7)
        moved = 0
        for call in step.energy_calls:
            self.assertIsNotNone(call.anchor)
            expected = torch.stack([factory.physical_starts[identity] for identity in call.context_ids])
            self.assertTrue(torch.equal(call.anchor, expected))
            self.assertFalse(torch.equal(call.anchor, call.inertial_prediction))
            if not torch.equal(call.positions, expected):
                moved += 1
                self.assertFalse(torch.equal(call.anchor, call.positions))
        # Three post-update optimization observations and two physical records differentiate a moved candidate.
        self.assertEqual(moved, 5)
        self.assertEqual(step.contexts, {})

    def test_selection_metric_is_the_mean_of_asymmetric_final_residuals(self):
        """Aggregate the final residuals with the mean rather than the median, maximum or one sample."""
        step = _AnalyticStep()
        factory = _Factory(step, offsets={0: 2.0, 1: 2.0, 2: 8.0})
        report = validate(step, factory, _config(count=3, batch_size=3), torch.device("cpu"), 0, 1)
        self.assertEqual(
            [sample["free_force_residual_norm_n"] for sample in report["samples"]],
            [[4.0, 2.0, 1.0, 0.5], [4.0, 2.0, 1.0, 0.5], [16.0, 8.0, 4.0, 2.0]],
        )
        self.assertEqual(report["selection"]["metric"], 1.0)
        self.assertTrue(report["selection"]["eligible"])
        self.assertEqual(
            report["force_residual"][0],
            {"iteration": 0, "mean": 8.0, "median": 4.0, "max": 16.0, "valid_count": 3, "failed_count": 0},
        )
        self.assertEqual(
            report["force_residual"][-1],
            {"iteration": 3, "mean": 1.0, "median": 0.5, "max": 2.0, "valid_count": 3, "failed_count": 0},
        )
        self.assertEqual(step.contexts, {})

    def test_history_is_stored_after_every_query_and_carried_across_physical_steps(self):
        """Feed each query the previous query's world gradient exactly as training does."""
        step = _AnalyticStep()
        factory = _Factory(step)
        validate(step, factory, _config(count=2, batch_size=2, physical_steps=3), torch.device("cpu"), 0, 1)
        self.assertEqual(
            step.history_flags,
            [[False, False], [True, True], [True, True], [False, False], [True, True], [True, True]],
        )
        self.assertEqual(
            step.history_markers,
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        )
        self.assertEqual(factory.advanced_history, [True, True, True, True])
        self.assertEqual(step.contexts, {})

    def test_selection_ineligible_when_an_optimization_sample_fails(self):
        """Keep the residual curve gaps visible and withhold the metric when a sample failed."""
        step = _AnalyticStep(fail_iteration=2)
        report = validate(step, _Factory(step), _config(), torch.device("cpu"), 0, 1)
        self.assertEqual([point["failed_count"] for point in report["force_residual"]], [0, 0, 1, 1])
        self.assertEqual([point["mean"] for point in report["force_residual"]], [4.0, 2.0, None, None])
        self.assertEqual(report["selection"]["metric"], None)
        self.assertFalse(report["selection"]["eligible"])
        self.assertEqual(report["relative_energy"][0]["mean"], 1.0)
        self.assertEqual(report["relative_energy"][1]["mean"], 0.25)
        self.assertIsNone(report["relative_energy"][2]["mean"])
        self.assertAlmostEqual(report["mean_normalized_loss"], math.asinh(0.25), places=12)
        self.assertEqual(report["first_update_failed_count"], 0)
        self.assertEqual(report["optimization_failed_count"], 1)
        self.assertEqual(report["physical_failed_count"], 0)
        self.assertEqual(report["physical_survivors"], 1)
        self.assertEqual(report["failed_count"], 1)
        sample = report["samples"][0]
        self.assertEqual(sample["energies"], [4.0, 1.0])
        self.assertEqual(sample["free_force_residual_norm_n"], [4.0, 2.0])
        self.assertEqual(sample["failure_iteration"], 2)
        self.assertEqual(sample["physical_steps"], 2)
        self.assertIsNone(sample["physical_error"])
        self.assertEqual(step.contexts, {})

    def test_selection_ineligible_when_a_physical_trajectory_dies(self):
        """Report the complete optimization metric but refuse selection without survival."""
        step = _AnalyticStep(fail_physical=True)
        report = validate(step, _Factory(step), _config(), torch.device("cpu"), 0, 1)
        self.assertEqual([point["mean"] for point in report["relative_energy"]], [1.0, 0.25, 0.0625, 0.015625])
        self.assertEqual(report["selection"]["metric"], 0.5)
        self.assertFalse(report["selection"]["eligible"])
        self.assertEqual(report["optimization_failed_count"], 0)
        self.assertEqual(report["physical_failed_count"], 1)
        self.assertEqual(report["physical_survivors"], 0)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual([point["valid_count"] for point in report["physical_curves"]], [1, 0])
        self.assertEqual([point["failed_count"] for point in report["physical_curves"]], [0, 1])
        self.assertIsNone(report["final_physical_residual"]["mean"])
        sample = report["samples"][0]
        self.assertIsNone(sample["error"])
        self.assertIn("physical rollout", sample["physical_error"])
        self.assertEqual(sample["physical_steps"], 1)
        self.assertEqual(sample["physical_failure_step"], 2)
        self.assertEqual(len(sample["physical_records"]), 1)
        self.assertEqual(step.contexts, {})

    def test_inversion_diagnostics_are_recorded_but_never_fail_a_sample(self):
        """Accept an inverted initial candidate and report its recovery as a diagnostic only."""
        step = _AnalyticStep()
        report = validate(step, _Factory(step, offset=-4.5), _config(), torch.device("cpu"), 0, 1)
        sample = report["samples"][0]
        self.assertIsNone(sample["error"])
        self.assertIsNone(sample["physical_error"])
        self.assertEqual(sample["inverted_cell_counts"], [1, 0, 0, 0])
        self.assertAlmostEqual(sample["min_center_jacobians"][0], -0.125, places=6)
        self.assertAlmostEqual(sample["min_center_jacobians"][1], 0.4375, places=6)
        self.assertEqual(report["failed_count"], 0)
        self.assertTrue(report["selection"]["eligible"])
        self.assertEqual(report["inversion"][0]["inverted_sample_count"], 1)
        self.assertEqual(report["inversion"][0]["max_inverted_cell_count"], 1)
        self.assertEqual(report["inversion"][0]["mean_inverted_cell_count"], 1.0)
        self.assertAlmostEqual(report["inversion"][0]["min_center_jacobian"], -0.125, places=6)
        self.assertEqual(report["inversion"][1]["inverted_sample_count"], 0)
        self.assertEqual(report["final_inverted_sample_count"], 0)
        self.assertEqual(report["physical_curves"][0]["inverted_sample_count"], 0)
        self.assertEqual(step.contexts, {})

    def test_near_zero_queries_keep_absolute_energy_displacement_and_free_residual(self):
        """Exclude unstable ratios and pinned forces while differentiating only the physical energy."""
        step = _AnalyticStep(pin_target=5e-5)
        report = validate(step, _Factory(step, offset=1e-5), _config(), torch.device("cpu"), 0, 1)
        self.assertIn("near_zero", report)
        self.assertEqual(report["near_zero"]["sample_count"], 1)
        self.assertTrue(all(point["mean"] is None for point in report["relative_energy"]))
        points = report["near_zero"]["curves"]
        self.assertAlmostEqual(points[0]["mean_energy_joule"], 2.6e-9, delta=1e-11)
        self.assertEqual(points[0]["mean_displacement_rms_m"], 0)
        self.assertGreater(points[1]["mean_displacement_rms_m"], 0)
        self.assertAlmostEqual(points[0]["mean_free_force_residual_norm_n"], 2e-5, delta=1e-7)
        self.assertLess(points[-1]["mean_free_force_residual_norm_n"], points[0]["mean_free_force_residual_norm_n"])
        self.assertAlmostEqual(report["force_residual"][0]["mean"], 2e-5, delta=1e-7)
        energies = report["samples"][0]["energies"]
        self.assertAlmostEqual(report["mean_normalized_loss"], math.asinh(energies[1] / _FLOOR), places=9)
        self.assertTrue(all(parameter.grad is None for parameter in step.network.parameters()))
        self.assertEqual(len(step.batch_sizes), 5)
        self.assertEqual(step.contexts, {})

    def test_partial_batch_reset_cleans_up_and_retries_healthy_member(self):
        """Isolate a failed reset while cleaning every prepared context and retaining both seeds."""
        step = _AnalyticStep()
        factory = _Factory(step, fail_reset_seed=1)
        config = _config(count=2, batch_size=2)
        with self.assertRaisesRegex(ValueError, "reset failure"):
            validation_chunk(step, factory, [0, 1], config, torch.device("cpu"))
        self.assertEqual(step.contexts, {})
        report = validate(step, factory, config, torch.device("cpu"), 0, 1)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["first_update_failed_count"], 1)
        self.assertEqual(report["physical_survivors"], 1)
        self.assertIsNone(report["mean_normalized_loss"])
        self.assertFalse(report["selection"]["eligible"])
        self.assertIsNone(report["selection"]["metric"])
        self.assertEqual([point["failed_count"] for point in report["force_residual"]], [1, 1, 1, 1])
        self.assertEqual([sample["seed"] for sample in report["samples"]], [0, 1])
        self.assertIsNone(report["samples"][0]["error"])
        self.assertIn("reset failure", report["samples"][1]["error"])
        self.assertEqual(step.contexts, {})

    def test_successful_validation_keeps_batched_proposals_and_original_mode(self):
        """Evaluate successful groups together and restore the caller's module mode."""
        step = _AnalyticStep().eval()
        report = validate(step, _Factory(step), _config(count=2, batch_size=2), torch.device("cpu"), 0, 1)
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(step.batch_sizes, [2, 2, 2, 2, 2])
        self.assertFalse(step.training)
        self.assertEqual(step.contexts, {})


class TestFullHorizonValidation(unittest.TestCase):
    def test_summary_keys_failed_sample_visibility_and_disjoint_seeds(self):
        """Run the fixed subset after the cheap seeds and keep a dead trajectory visible."""
        step = _AnalyticStep(fail_physical_contexts={"3"})
        factory = _Factory(step)
        config = _config(count=2, batch_size=2, full_count=3)
        report = validate_full_horizon(step, factory, config, torch.device("cpu"), 0, 1, iterations=2, physical_steps=3)
        self.assertEqual(
            set(report),
            {
                "sample_count",
                "physical_survivors",
                "failed_count",
                "iterations",
                "physical_steps",
                "final_free_force_residual_norm_n",
                "final_energy_joule",
                "final_physical_residual",
                "final_inverted_sample_count",
                "physical_curves",
                "samples",
                "seconds",
            },
        )
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["physical_survivors"], 2)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["iterations"], 2)
        self.assertEqual(report["physical_steps"], 3)
        self.assertGreaterEqual(report["seconds"], 0.0)
        self.assertEqual([sample["seed"] for sample in report["samples"]], [2, 3, 4])
        self.assertEqual([len(sample["physical_records"]) for sample in report["samples"]], [3, 1, 3])
        self.assertEqual([sample["physical_steps"] for sample in report["samples"]], [3, 1, 3])
        failed = report["samples"][1]
        self.assertIn("physical rollout", failed["physical_error"])
        self.assertEqual(failed["physical_failure_step"], 2)
        self.assertEqual(failed["perturbation_scale"], 0.3)
        self.assertEqual([point["valid_count"] for point in report["physical_curves"]], [3, 2, 2])
        self.assertEqual([point["failed_count"] for point in report["physical_curves"]], [0, 1, 1])
        self.assertEqual(report["physical_curves"][0]["energy_joule"]["mean"], 0.25)
        self.assertEqual(report["physical_curves"][0]["free_force_residual_norm_n"]["mean"], 1.0)
        self.assertIsNone(report["final_free_force_residual_norm_n"]["mean"])
        self.assertIsNone(report["final_energy_joule"]["mean"])
        self.assertEqual(report["final_physical_residual"], report["physical_curves"][-1]["free_force_residual_norm_n"])
        self.assertEqual(report["final_inverted_sample_count"], 0)
        record = report["samples"][0]["physical_records"][0]
        self.assertEqual(
            set(record),
            {
                "step",
                "energy_joule",
                "free_force_residual_norm_n",
                "displacement_rms_m",
                "inverted_cell_count",
                "min_center_jacobian",
            },
        )
        self.assertEqual(step.contexts, {})
        self.assertTrue(step.training)

    def test_complete_subset_reports_final_statistics_and_carries_history(self):
        """Aggregate final energy and residual over a fully surviving subset."""
        step = _AnalyticStep()
        factory = _Factory(step)
        report = validate_full_horizon(
            step,
            factory,
            _config(count=4, batch_size=4, full_count=2),
            torch.device("cpu"),
            0,
            1,
            iterations=1,
            physical_steps=2,
        )
        self.assertEqual([sample["seed"] for sample in report["samples"]], [4, 5])
        self.assertEqual(report["physical_survivors"], 2)
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(report["final_energy_joule"], {"mean": 0.25, "median": 0.25, "max": 0.25})
        self.assertEqual(report["final_free_force_residual_norm_n"], {"mean": 1.0, "median": 1.0, "max": 1.0})
        self.assertEqual(step.history_flags, [[False, False], [True, True]])
        self.assertEqual(step.history_markers, [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(step.contexts, {})

    def test_final_statistics_distinguish_mean_median_and_max(self):
        """Report mean, median and maximum of asymmetric final energies and residuals as separate values."""
        step = _AnalyticStep()
        factory = _Factory(step, offsets={3: 2.0, 4: 2.0, 5: 8.0})
        report = validate_full_horizon(
            step,
            factory,
            _config(count=3, batch_size=3, full_count=3),
            torch.device("cpu"),
            0,
            1,
            iterations=1,
            physical_steps=2,
        )
        self.assertEqual([sample["seed"] for sample in report["samples"]], [3, 4, 5])
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(
            [sample["physical_records"][-1]["free_force_residual_norm_n"] for sample in report["samples"]],
            [1.0, 1.0, 4.0],
        )
        self.assertEqual(
            [sample["physical_records"][-1]["energy_joule"] for sample in report["samples"]], [0.25, 0.25, 4.0]
        )
        self.assertEqual(report["final_free_force_residual_norm_n"], {"mean": 2.0, "median": 1.0, "max": 4.0})
        self.assertEqual(report["final_energy_joule"], {"mean": 1.5, "median": 0.25, "max": 4.0})
        self.assertEqual(step.contexts, {})

    def test_seeds_are_distributed_round_robin_and_arguments_are_validated(self):
        """Split the fixed subset over ranks and reject non-positive horizons."""
        config = _config(count=2, full_count=5)
        self.assertEqual(_full_horizon_seeds(config, 0, 2), [2, 4, 6])
        self.assertEqual(_full_horizon_seeds(config, 1, 2), [3, 5])
        self.assertEqual(_full_horizon_seeds(_config(count=2, full_count=0), 0, 1), [])
        step = _AnalyticStep()
        for iterations, physical_steps in ((0, 1), (1, 0), (True, 1), (1.5, 2)):
            with self.subTest(iterations=iterations, physical_steps=physical_steps), self.assertRaises(ValueError):
                validate_full_horizon(
                    step,
                    _Factory(step),
                    config,
                    torch.device("cpu"),
                    0,
                    1,
                    iterations=iterations,
                    physical_steps=physical_steps,
                )
        with self.assertRaises(ValueError):
            validate_full_horizon(
                step,
                _Factory(step),
                _config(count=2, full_count=-1),
                torch.device("cpu"),
                0,
                1,
                iterations=1,
                physical_steps=1,
            )
        empty = validate_full_horizon(
            step,
            _Factory(step),
            _config(count=2, full_count=0),
            torch.device("cpu"),
            0,
            1,
            iterations=1,
            physical_steps=1,
        )
        self.assertEqual(empty["sample_count"], 0)
        self.assertEqual(empty["physical_survivors"], 0)
        self.assertIsNone(empty["final_free_force_residual_norm_n"]["mean"])


class TestRealStepResidual(unittest.TestCase):
    """Compare the validator residual with the real mixed step on a damped unit cell."""

    def _config(self):
        return train_mixed.MixedTrainConfig(
            cell_counts=(1, 1, 1),
            cell_size=0.1,
            hidden_dim=8,
            edge_hidden_dim=4,
            num_heads=2,
            batch_size=2,
            pool_multiplier=2,
            queries_per_epoch=1,
            max_epochs=1,
            iteration_counts=(1,),
            physical_step_counts=(1,),
            validation_count=2,
            validation_iterations=2,
            validation_physical_steps=2,
            validation_physical_iterations=1,
            validation_full_count=1,
            validation_full_interval=1,
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
        # The correction and step heads start at zero; perturb every weight so the candidate moves.
        torch.manual_seed(0)
        with torch.no_grad():
            for parameter in step.network.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        return step, rest

    def test_validator_residual_matches_the_real_step_with_the_physical_anchor(self):
        """Match the step's own pre-update residual and reject the candidate or Y as the damping anchor."""
        config = self._config()
        step, rest = self._step(config)
        factory = train_mixed._TrajectoryFactory(step, rest, config, rank=0, validation=True)
        dampings = []
        reset = factory.reset

        def recording_reset(seed):
            payload = reset(seed)
            dampings.append(payload["context_spec"]["damping"])
            return payload

        forward_residuals = []
        forward = step.forward

        def recording_forward(*args, **kwargs):
            result = forward(*args, **kwargs)
            forward_residuals.append(result.force_residual_norm.detach().double().cpu().tolist())
            return result

        observations = []
        residual_norms = mixed_validation._free_force_residual_norms

        def observing_residual_norms(module, batch):
            returned = residual_norms(module, batch)
            observations.append(
                {
                    "returned": list(returned),
                    "physical": _reference_residual_norms(module, batch, batch["physical_positions"]),
                    "candidate": _reference_residual_norms(module, batch, batch["candidate"]),
                    "inertial": _reference_residual_norms(module, batch, batch["inertial_prediction"]),
                }
            )
            return returned

        with (
            patch.object(factory, "reset", recording_reset),
            patch.object(step, "forward", recording_forward),
            patch.object(mixed_validation, "_free_force_residual_norms", observing_residual_norms),
        ):
            report = validate(step, factory, config, torch.device("cpu"), 0, 1)
        self.assertEqual(len(dampings), 4)
        self.assertTrue(all(damping > 0 for damping in dampings))
        self.assertEqual(
            [(sample["error"], sample["physical_error"]) for sample in report["samples"]], [(None, None)] * 2
        )
        iterations, physical_steps = config.validation_iterations, config.validation_physical_steps
        self.assertEqual(len(forward_residuals), iterations + physical_steps * config.validation_physical_iterations)
        recorded = [sample["free_force_residual_norm_n"] for sample in report["samples"]]
        physical = [
            [record["free_force_residual_norm_n"] for record in sample["physical_records"]]
            for sample in report["samples"]
        ]
        for index, curve in enumerate(recorded):
            self.assertEqual(len(curve), iterations + 1)
            self.assertEqual(len(set(curve)), iterations + 1)
            for k in range(iterations):
                # Only pre-update candidates have a forward record; the step norms in float32.
                self.assertTrue(math.isclose(curve[k], forward_residuals[k][index], rel_tol=1e-6), (k, index))
        expected = [[curve[k] for curve in recorded] for k in range(iterations + 1)]
        expected += [[curve[p] for curve in physical] for p in range(physical_steps)]
        self.assertEqual([observation["returned"] for observation in observations], expected)
        for observation in observations:
            for value, physical_anchor, candidate_anchor, inertial_anchor in zip(
                observation["returned"],
                observation["physical"],
                observation["candidate"],
                observation["inertial"],
                strict=True,
            ):
                self.assertTrue(math.isclose(value, physical_anchor, rel_tol=1e-6))
                self.assertGreater(abs(physical_anchor - candidate_anchor), 1e-5 * physical_anchor)
                self.assertGreater(abs(physical_anchor - inertial_anchor), 1e-5 * physical_anchor)
        self.assertEqual(step.context_specs, {})


if __name__ == "__main__":
    unittest.main()
