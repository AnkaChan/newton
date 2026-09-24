# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify heterogeneous damping, its feature layout, and physical-step anchors."""

import copy
import importlib.util
import math
import unittest
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep


class TestMixedDamping(unittest.TestCase):
    def setUp(self):
        """Create small canonical geometry and distinct homogeneous materials."""
        torch.manual_seed(321)
        self.rest = generate_cuboid((2, 1, 2), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.positions = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)[None]
        self.dt = 0.01
        self.material = {"lame_lambda": 700.0, "lame_mu": 400.0, "density": 100.0}

    def _step(self, *, damped=True):
        network = IntrinsicSolverNetwork(
            self.rest.cell_counts,
            86 if damped else 38,
            conditioning_dim=6 if damped else 5,
            hidden_dim=16,
            edge_hidden_dim=8,
        )
        step = MixedHexSolverStep(self.rest, self.fixed, network=network, time_step=self.dt)
        self.addCleanup(step.close)
        return step

    def test_legacy_context_serializes_explicit_zero_damping(self):
        """Record the default zero coefficient while preserving legacy energy evaluation."""
        step = self._step(damped=False)
        step.register_context("legacy", **self.material)
        self.assertIn("damping", step.context_specs["legacy"])
        self.assertEqual(step.context_specs["legacy"]["damping"], 0.0)
        loss = step.energy(self.positions, self.positions, ("legacy",))
        torch.testing.assert_close(loss.damping, torch.zeros(1), rtol=0, atol=0)
        torch.testing.assert_close(loss.total, loss.elastic + loss.inertia, rtol=0, atol=0)

    def test_damped_features_append_gauss_metrics_and_keep_zero_coefficient_layout(self):
        """Preserve the original channels and append all eight symmetric metric differences."""
        step = self._step()
        legacy = self._step(damped=False)
        step.register_context("zero", damping=0.0, **self.material)
        step.register_context("damped", damping=0.8, **self.material)
        legacy.register_context("zero", **self.material)
        current = self.positions.clone()
        current[..., 2] *= 1.1
        original = legacy.prepare_inputs(current, self.positions, ("zero",))
        inputs = step.prepare_inputs(current, self.positions, ("zero",), previous_positions=self.positions)
        self.assertEqual(inputs.state_features.shape, (1, 4, 86))
        self.assertEqual(inputs.conditioning.shape, (1, 4, 6))
        torch.testing.assert_close(inputs.state_features[..., :38], original.state_features, rtol=0, atol=0)
        torch.testing.assert_close(inputs.conditioning[..., :5], original.conditioning, rtol=0, atol=0)
        torch.testing.assert_close(inputs.conditioning[..., 5], torch.zeros((1, 4)), rtol=0, atol=0)
        expected = torch.tensor([0.0, 0.0, 0.21, 0.0, 0.0, 0.0]).repeat(8)
        torch.testing.assert_close(inputs.state_features[..., 38:], expected.expand(1, 4, 48), rtol=2e-5, atol=5e-7)
        zero_metric = step.prepare_inputs(current, self.positions, ("zero",), previous_positions=current)
        torch.testing.assert_close(zero_metric.state_features[..., 38:], torch.zeros((1, 4, 48)), rtol=0, atol=0)
        damped = step.prepare_inputs(current, self.positions, ("damped",), previous_positions=self.positions)
        torch.testing.assert_close(damped.conditioning[..., 5], torch.full((1, 4), math.log1p(0.2)))
        undamped = step.energy(current, self.positions, ("zero",))
        reference = legacy.energy(current, self.positions, ("zero",))
        torch.testing.assert_close(undamped.total, reference.total, rtol=0, atol=0)

    def test_uniform_stretch_damping_matches_integrated_metric_difference(self):
        """Integrate the known affine metric change with each object's coefficient."""
        step = self._step()
        step.register_context("soft", damping=0.8, **self.material)
        step.register_context("strong", damping=3.2, **self.material)
        current = self.positions.repeat(2, 1, 1)
        current[..., 2] *= 1.1
        previous = self.positions.repeat(2, 1, 1)
        terms = step.energy(current, current, ("soft", "strong"), previous_positions=previous)
        expected = torch.tensor([0.8, 3.2]) * (4 * 0.1**3) * 0.21**2 / (2 * self.dt)
        torch.testing.assert_close(terms.damping, expected, rtol=2e-5, atol=1e-8)
        torch.testing.assert_close(terms.total, terms.elastic + terms.inertia + terms.damping, rtol=0, atol=0)

    def test_mixed_damped_forward_and_gradients_match_independent_steps(self):
        """Match separate single-material damped solvers with one batched network evaluation."""
        step = self._step()
        materials = {
            "soft": dict(self.material, damping=0.8),
            "stiff": {"lame_lambda": 3000.0, "lame_mu": 1200.0, "density": 250.0, "damping": 7.2},
        }
        for identity, specification in materials.items():
            step.register_context(identity, **specification)
        with torch.no_grad():
            step.network.correction_head.weight.normal_(std=0.003)
            for layer in step.network.layers:
                layer.film.weight.normal_(std=0.01)
        ids = ("stiff", "soft", "stiff")
        previous = self.positions.repeat(3, 1, 1)
        previous[1, :, 1] += 0.02 * previous[1, :, 2].square()
        current = previous.clone()
        current[:, :, 0] += 0.08 * current[:, :, 2].square()
        current[2, :, 1] -= 0.03 * current[2, :, 2].square()
        target = current + torch.tensor([0.0003, -0.001, 0.0001])
        pins = current[:, self.fixed].clone()
        reference_network = copy.deepcopy(step.network)
        references = []
        for index, identity in enumerate(ids):
            single = LearnedHexSolverStep(
                self.rest, self.fixed, network=reference_network, time_step=self.dt, **materials[identity]
            )
            references.append(
                single(
                    current[index : index + 1],
                    target[index : index + 1],
                    previous_positions=previous[index : index + 1],
                    fixed_positions=pins[index : index + 1],
                )
            )
        with patch.object(step.network, "forward", wraps=step.network.forward) as counted:
            output = step(current, target, ids, fixed_positions=pins, previous_positions=previous)
        self.assertEqual(counted.call_count, 1)
        torch.testing.assert_close(output.positions, torch.cat([r.positions for r in references]), rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(output.positions[:, self.fixed], pins, rtol=0, atol=0)
        for name in ("total", "elastic", "inertia", "damping"):
            torch.testing.assert_close(
                getattr(output.loss, name), torch.cat([getattr(r.loss, name) for r in references]), rtol=8e-5, atol=3e-7
            )
        output.loss.total.mean().backward()
        torch.cat([value.loss.total for value in references]).mean().backward()
        reference_parameters = dict(reference_network.named_parameters())
        for name, parameter in step.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, rtol=5e-4, atol=3e-7, msg=name)

    def test_anchor_is_held_until_physical_advance_without_changing_context(self):
        """Keep the physical starting shape through inner proposals and replace it only on advance."""
        step = self._step()
        step.register_context("case", damping=0.8, **self.material)
        payload = step.prepare("case", self.positions[0], torch.zeros_like(self.positions[0]))
        original_anchor = payload["physical_positions"].clone()
        original_target = payload["inertial_prediction"].clone()
        specification = step.context_specs["case"]
        candidate = payload["candidate"][None].clone()
        candidate[:, :, 0] += 0.05 * candidate[:, :, 2].square()
        for _ in range(2):
            result = step(
                candidate,
                payload["inertial_prediction"][None],
                ("case",),
                previous_positions=payload["physical_positions"][None],
                fixed_positions=payload["fixed_positions"][None],
            )
            self.assertGreater(result.loss.damping.item(), 0)
            candidate = result.positions.detach()
            torch.testing.assert_close(payload["physical_positions"], original_anchor, rtol=0, atol=0)
            torch.testing.assert_close(payload["inertial_prediction"], original_target, rtol=0, atol=0)
        payload["candidate"] = candidate[0]
        advanced = step.advance(payload)
        torch.testing.assert_close(advanced["physical_positions"], candidate[0], rtol=0, atol=0)
        torch.testing.assert_close(payload["physical_positions"], original_anchor, rtol=0, atol=0)
        self.assertEqual(step.context_specs["case"], specification)
        committed = step.energy(
            candidate, candidate, ("case",), previous_positions=advanced["physical_positions"][None]
        )
        torch.testing.assert_close(committed.damping, torch.zeros(1), rtol=0, atol=0)

    def test_invalid_schema_coefficients_and_missing_anchor_are_rejected(self):
        """Reject positive damping with a legacy schema and validate physical-step anchors."""
        legacy = self._step(damped=False)
        with self.assertRaisesRegex(ValueError, "damp"):
            legacy.register_context("positive", damping=0.8, **self.material)
        self.assertEqual(legacy.context_specs, {})
        step = self._step()
        for invalid in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(damping=invalid), self.assertRaises(ValueError):
                step.register_context("bad", damping=invalid, **self.material)
        step.register_context("case", damping=0.8, **self.material)
        step.register_context("zero", damping=0.0, **self.material)
        for identity in ("case", "zero"):
            for operation in (step.prepare_inputs, step.forward):
                with (
                    self.subTest(operation=operation.__name__, identity=identity),
                    self.assertRaisesRegex(ValueError, "previous_positions"),
                ):
                    operation(self.positions, self.positions, (identity,))
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            step.energy(self.positions, self.positions, ("case",))
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            step.energy(self.positions, self.positions, ("case",), previous_positions=self.positions.repeat(2, 1, 1))


if __name__ == "__main__":
    unittest.main()
