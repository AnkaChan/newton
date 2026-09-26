# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify heterogeneous damping, its axis-change block and conditioning, and physical-step anchors."""

import importlib.util
import math
import unittest

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.fusion import HexFusion
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork


class TestMixedDamping(unittest.TestCase):
    def setUp(self):
        """Create small canonical geometry and distinct homogeneous materials."""
        torch.manual_seed(321)
        self.rest = generate_cuboid((2, 1, 2), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.cell_count = len(self.rest.cell_corner_indices)
        self.positions = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)[None]
        self.dt = 0.01
        self.material = {"lame_lambda": 700.0, "lame_mu": 400.0, "density": 100.0}

    def _step(self):
        network = IntrinsicSolverNetwork(
            self.rest.cell_counts,
            features.STATE_FEATURE_DIM,
            conditioning_dim=features.CONDITIONING_DIM,
            hidden_dim=16,
            edge_hidden_dim=8,
        )
        with torch.no_grad():
            network.correction_head.weight.normal_(std=0.003)
            for layer in network.layers:
                layer.film.weight.normal_(std=0.01)
        step = MixedHexSolverStep(self.rest, self.fixed, network=network, time_step=self.dt)
        self.addCleanup(step.close)
        return step

    def _physical(self, damping, dtype=torch.float32):
        return HexImplicitEulerLoss(self.rest, time_step=self.dt, damping=damping, dtype=dtype, **self.material)

    def _perturbed(self, count):
        """Return distinct current, target, and physical-start batches with pins in place."""
        current = self.positions.repeat(count, 1, 1)
        for index in range(count):
            current[index, :, 0] += (0.05 + 0.02 * index) * current[index, :, 2].square()
            current[index, :, 1] += 0.003 * torch.sin(11 * current[index, :, 0] + index) * current[index, :, 2]
        previous = self.positions.repeat(count, 1, 1)
        previous[:, :, 1] += 0.02 * previous[:, :, 2].square()
        target = current + torch.tensor([0.0003, -0.001, 0.0001])
        return current, target, previous

    def test_contexts_serialize_damping_and_accept_positive_coefficients_in_one_schema(self):
        """Record zero damping explicitly and register damped materials with the same network."""
        step = self._step()
        step.register_context("plain", **self.material)
        step.register_context("damped", damping=0.8, **self.material)
        self.assertEqual(step.context_specs["plain"]["damping"], 0.0)
        self.assertEqual(step.context_specs["damped"]["damping"], 0.8)
        loss = step.energy(self.positions, self.positions, ("plain",))
        torch.testing.assert_close(loss.damping, torch.zeros(1), rtol=0, atol=0)
        torch.testing.assert_close(loss.total, loss.elastic + loss.inertia, rtol=0, atol=0)

    def test_physical_axis_change_block_and_damping_conditioning(self):
        """Encode R^T (F - F_previous) as the second block and log1p(eta / (mu dt)) as channel six."""
        step = self._step()
        step.register_context("zero", damping=0.0, **self.material)
        step.register_context("damped", damping=0.8, **self.material)
        current = self.positions.clone()
        current[..., 2] *= 1.1
        inputs = step.prepare_inputs(current, self.positions, ("zero",), previous_positions=self.positions)
        self.assertEqual(inputs.state_features.shape, (1, self.cell_count, features.STATE_FEATURE_DIM))
        self.assertEqual(inputs.conditioning.shape, (1, self.cell_count, features.CONDITIONING_DIM))
        torch.testing.assert_close(inputs.frames, torch.eye(3).expand(1, self.cell_count, 3, 3), rtol=0, atol=1e-6)
        change = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]).expand(1, self.cell_count, 9)
        torch.testing.assert_close(inputs.state_features[..., 9:18], change, rtol=2e-5, atol=5e-7)
        torch.testing.assert_close(inputs.state_features[..., 0:9], -change, rtol=2e-5, atol=5e-7)
        torch.testing.assert_close(inputs.conditioning[..., 5], torch.zeros(1, self.cell_count), rtol=0, atol=0)
        same_anchor = step.prepare_inputs(current, self.positions, ("zero",), previous_positions=current)
        torch.testing.assert_close(
            same_anchor.state_features[..., 9:18], torch.zeros(1, self.cell_count, 9), rtol=0, atol=0
        )
        damped = step.prepare_inputs(current, self.positions, ("damped",), previous_positions=self.positions)
        torch.testing.assert_close(damped.conditioning[..., 5], torch.full((1, self.cell_count), math.log1p(0.2)))
        torch.testing.assert_close(damped.conditioning[..., :5], inputs.conditioning[..., :5], rtol=0, atol=0)
        # Damping changes only the objective gradient blocks; geometry, boundary flags and history flag agree.
        keep = torch.ones(features.STATE_FEATURE_DIM, dtype=torch.bool)
        keep[18:27] = False
        keep[59] = False
        torch.testing.assert_close(damped.state_features[..., keep], inputs.state_features[..., keep], rtol=0, atol=0)
        self.assertFalse(torch.allclose(damped.axis_gradient_world, inputs.axis_gradient_world))
        reference = self._physical(0.0)(current, self.positions)
        undamped = step.energy(current, self.positions, ("zero",))
        torch.testing.assert_close(undamped.total, reference.total, rtol=1e-6, atol=0)

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

    def test_gradient_feature_includes_damping_and_matches_float64_twin(self):
        """Differentiate the complete objective, including damping, through the fusion adjoint."""
        step = self._step()
        coefficients = {"none": 0.0, "soft": 0.8, "strong": 3.2}
        for name, damping in coefficients.items():
            step.register_context(name, damping=damping, **self.material)
        current, target, previous = self._perturbed(2)
        damped = step.prepare_inputs(current, target, ("soft", "strong"), previous_positions=previous)
        undamped = step.prepare_inputs(current, target, ("none", "none"), previous_positions=previous)
        self.assertFalse(torch.allclose(damped.axis_gradient_world, undamped.axis_gradient_world, rtol=1e-3))
        self.assertGreater(
            damped.position_gradient.flatten(1).norm(dim=1).min().item(),
            0.5 * undamped.position_gradient.flatten(1).norm(dim=1).max().item(),
        )
        for index, name in enumerate(("soft", "strong")):
            physical = self._physical(coefficients[name], dtype=torch.float64)
            lam, mu = physical.lame_lambda, physical.lame_mu
            scale = torch.maximum(lam, mu)
            weights = mu * (3 - (mu / scale) / (lam / scale + mu / scale)) * self.rest.cell_size**3
            fusion = HexFusion(self.rest, self.fixed, cell_weights=weights, dtype=torch.float64)
            x = current[index : index + 1].double()
            y = target[index : index + 1].double()
            start = previous[index : index + 1].double()
            increment = torch.zeros((1, self.cell_count, 3, 3), dtype=torch.float64, requires_grad=True)
            energy = physical(fusion.fuse(x, increment, x[:, self.fixed]), y, previous_positions=start)
            self.assertGreater(energy.damping.item(), 0)
            expected = torch.autograd.grad(energy.total.sum(), increment)[0]
            magnitude = expected.abs().max().item()
            torch.testing.assert_close(
                damped.axis_gradient_world[index].double(), expected[0], rtol=1e-4, atol=1e-5 * magnitude, msg=name
            )

    def test_mixed_damped_forward_matches_independent_energy_and_keeps_gradients(self):
        """Evaluate the fused damped objective per object and differentiate to every parameter."""
        step = self._step()
        materials = {"soft": 0.8, "stiff": 7.2}
        for name, damping in materials.items():
            step.register_context(name, damping=damping, **self.material)
        ids = ("stiff", "soft", "stiff")
        current, target, previous = self._perturbed(3)
        pins = current[:, self.fixed].clone()
        output = step(current, target, ids, fixed_positions=pins, previous_positions=previous)
        self.assertEqual(output.step_size.shape, (3, self.cell_count))
        self.assertEqual(output.force_residual_norm.shape, (3,))
        torch.testing.assert_close(output.positions[:, self.fixed], pins, rtol=0, atol=0)
        self.assertTrue((output.loss.damping > 0).all())
        for index, name in enumerate(ids):
            reference = self._physical(materials[name])(
                output.positions[index : index + 1].detach(),
                target[index : index + 1],
                previous_positions=previous[index : index + 1],
            )
            for term in ("total", "elastic", "inertia", "damping"):
                torch.testing.assert_close(
                    getattr(output.loss, term)[index : index + 1],
                    getattr(reference, term),
                    rtol=2e-6,
                    atol=1e-9,
                    msg=term,
                )
        output.loss.total.mean().backward()
        for name, parameter in step.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(step.network.correction_head.weight.grad.abs().sum().item(), 0)

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

    def test_invalid_coefficients_and_missing_anchor_are_rejected(self):
        """Validate damping coefficients and require the physical-step anchor for every query."""
        step = self._step()
        for invalid in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(damping=invalid), self.assertRaises(ValueError):
                step.register_context("bad", damping=invalid, **self.material)
        self.assertEqual(step.context_specs, {})
        step.register_context("case", damping=0.8, **self.material)
        step.register_context("zero", damping=0.0, **self.material)
        for identity in ("case", "zero"):
            for operation in (step.prepare_inputs, step.forward):
                with self.subTest(operation=operation.__name__, identity=identity):
                    with self.assertRaisesRegex(ValueError, "previous_positions"):
                        operation(self.positions, self.positions, (identity,), previous_positions=None)
                    with self.assertRaises(TypeError):
                        operation(self.positions, self.positions, (identity,))
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            step.energy(self.positions, self.positions, ("case",))
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            step.energy(self.positions, self.positions, ("case",), previous_positions=self.positions.repeat(2, 1, 1))
        self.assertTrue(torch.isfinite(step.energy(self.positions, self.positions, ("zero",)).total).all())


if __name__ == "__main__":
    unittest.main()
