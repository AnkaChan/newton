# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check hexahedral physical energies and position gradients."""

import importlib.util
import unittest

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

from experiments.learned_intrinsic_solver.hex_energy import (
    HexImplicitEulerLoss,
    hex_gauss_quadrature,
    make_inertial_prediction,
)

# Preserve the former E=1000 Pa, nu=0.3 material in position-gradient checks.
_LAME_LAMBDA = 7500 / 13
_LAME_MU = 5000 / 13


class TestHexEnergy(unittest.TestCase):
    def test_quadrature_reproduces_affine_geometry(self):
        """Integrate the cube volume and reproduce affine material derivatives."""
        h = 0.3
        rule = hex_gauss_quadrature(h, dtype=np.float64)
        nodes = h * np.indices((2, 2, 2)).reshape(3, -1).T
        self.assertEqual(rule.shape_gradients.shape, (8, 8, 3))
        np.testing.assert_allclose(rule.shape_values.sum(axis=1), 1, atol=1e-15)
        np.testing.assert_allclose(rule.shape_gradients.sum(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(
            np.einsum("ki,qkj->qij", nodes, rule.shape_gradients), np.broadcast_to(np.eye(3), (8, 3, 3)), atol=1e-15
        )
        self.assertAlmostEqual(rule.weights.sum(), h**3)
        self.assertEqual(hex_gauss_quadrature(h).shape_gradients.dtype, np.float32)

    def test_lumped_mass_and_material_arrays(self):
        """Sum each cell's rho-volume-eighth contribution at shared corners."""
        import torch

        rest = generate_cuboid((2, 1, 1), cell_size=0.2)
        densities = np.array([2.0, 4.0])
        loss = HexImplicitEulerLoss(rest, lame_lambda=[0, 900], lame_mu=[300, 700], density=densities, time_step=0.01)
        expected = np.zeros(len(rest.corner_rest_positions))
        np.add.at(expected, rest.cell_corner_indices.reshape(-1), np.repeat(densities * 0.2**3 / 8, 8))
        np.testing.assert_allclose(loss.lumped_mass.numpy(), expected, rtol=2e-7)
        self.assertAlmostEqual(loss.lumped_mass.sum().item(), (densities * 0.2**3).sum(), places=8)
        self.assertEqual(loss.shape_gradients.dtype, torch.float32)
        self.assertEqual(loss.cell_corner_indices.dtype, torch.int64)
        torch.testing.assert_close(loss.lame_lambda, torch.tensor([0.0, 900.0]), rtol=0, atol=0)
        torch.testing.assert_close(loss.lame_mu, torch.tensor([300.0, 700.0]), rtol=0, atol=0)
        self.assertFalse(any(True for _ in loss.parameters()))

    def test_simple_shear_analytic_energy(self):
        """Match volume times mu times squared shear over two independently weighted cells."""
        import torch

        rest = generate_cuboid((2, 1, 1), cell_size=0.4)
        shear = 0.25
        deformation = torch.tensor([[1.0, shear, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
        positions = torch.tensor(rest.corner_rest_positions, dtype=torch.float64)[None] @ deformation.T
        # J=1 eliminates lambda entirely, including for heterogeneous lambda.
        loss = HexImplicitEulerLoss(rest, [0, 1700], [300, 700], 1000, 0.01, dtype=torch.float64)
        expected = 0.4**3 * (300 + 700) * shear**2 / 2
        self.assertAlmostEqual(loss(positions, positions).elastic.item(), expected, places=12)

    def test_uniform_dilation_analytic_energy(self):
        """Match the compressible log Neo-Hookean energy for uniform dilation."""
        import torch

        rest = generate_cuboid((2, 1, 1), cell_size=0.4)
        lam, mu = np.array([250.0, 900.0]), np.array([375.0, 400.0])
        scale = 1.12
        log_j = 3 * np.log(scale)
        expected = np.sum(0.4**3 * (0.5 * mu * (3 * scale**2 - 3) - mu * log_j + 0.5 * lam * log_j**2))
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 2e-5)):
            loss = HexImplicitEulerLoss(rest, lam, mu, 1000, 0.01, dtype=dtype)
            positions = (torch.tensor(rest.corner_rest_positions, dtype=dtype) * scale)[None]
            terms = loss(positions, positions)
            self.assertEqual(terms.total.shape, (1,))
            self.assertEqual(terms.total.dtype, dtype)
            torch.testing.assert_close(
                terms.elastic, torch.tensor([expected], dtype=dtype), rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(terms.inertia, torch.zeros_like(terms.inertia), rtol=0, atol=0)
            torch.testing.assert_close(terms.total, terms.elastic, rtol=0, atol=0)

    def test_zero_lambda_dilation_energy_and_derivative(self):
        """Retain the mu logarithmic term and its derivative when lambda is zero."""
        import torch

        rest = generate_cuboid((1, 1, 1), cell_size=0.4)
        mu, scale_value = 320.0, 0.8
        loss = HexImplicitEulerLoss(rest, lame_lambda=0, lame_mu=mu, density=1000, time_step=0.01, dtype=torch.float64)
        scale = torch.tensor(scale_value, dtype=torch.float64, requires_grad=True)
        positions = scale * torch.tensor(rest.corner_rest_positions, dtype=torch.float64)[None]
        energy = loss(positions, positions).elastic.sum()
        expected = 0.4**3 * mu * (1.5 * (scale_value**2 - 1) - 3 * np.log(scale_value))
        expected_derivative = 0.4**3 * 3 * mu * (scale_value - 1 / scale_value)
        self.assertAlmostEqual(energy.item(), expected, places=12)
        self.assertAlmostEqual(torch.autograd.grad(energy, scale)[0].item(), expected_derivative, places=11)
        torch.testing.assert_close(loss.lame_lambda, torch.zeros(1, dtype=torch.float64), rtol=0, atol=0)

    def test_rest_and_rigid_motion_zero_energy_force(self):
        """Keep rest and rigidly transformed cells stress free within arithmetic precision."""
        import torch

        rest = generate_cuboid((2, 2, 3), cell_size=0.25)
        angle = 0.63
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        rest_positions = rest.corner_rest_positions
        for dtype, tolerance in ((torch.float64, 2e-11), (torch.float32, 5e-4)):
            loss = HexImplicitEulerLoss(rest, _LAME_LAMBDA, _LAME_MU, 1000, 0.02, dtype=dtype)
            positions = torch.tensor(
                np.stack([rest_positions, rest_positions @ rotation.T + [0.3, -0.2, 0.1]]),
                dtype=dtype,
                requires_grad=True,
            )
            terms = loss(positions, positions.detach())
            gradient = torch.autograd.grad(terms.total.sum(), positions)[0]
            self.assertLess(terms.elastic.abs().max().item(), tolerance)
            self.assertLess(gradient.abs().max().item(), tolerance)

    def test_full_quadrature_sees_center_hourglass(self):
        """Penalize a nonaffine corner warp invisible to the cell-center gradient."""
        import torch

        rest = generate_cuboid((1, 1, 1), cell_size=0.5)
        nodes = np.indices((2, 2, 2)).reshape(3, -1).T
        warped = rest.corner_rest_positions.copy()
        warped[:, 0] += 0.025 * (-1.0) ** (nodes[:, 0] + nodes[:, 1])
        center_f = np.einsum("ki,kj->ij", warped - warped[:1], 2 * nodes - 1) / (4 * rest.cell_size)
        np.testing.assert_allclose(center_f, np.eye(3), atol=1e-15)
        positions = torch.tensor(warped[None], dtype=torch.float32, requires_grad=True)
        loss = HexImplicitEulerLoss(rest, _LAME_LAMBDA, _LAME_MU, 1000, 0.01)
        terms = loss(positions, positions.detach())
        self.assertGreater(terms.elastic.item(), 0.01)
        gradient = torch.autograd.grad(terms.elastic.sum(), positions)[0]
        self.assertGreater(gradient.norm().item(), 0.1)

    def test_inertial_prediction_and_exact_gradient(self):
        """Preserve the physical predictor and the mass-over-dt-squared gradient."""
        import torch

        rest = generate_cuboid((2, 1, 1), cell_size=0.2)
        loss = HexImplicitEulerLoss(rest, _LAME_LAMBDA, _LAME_MU, 900, 0.03)
        previous = torch.tensor(rest.corner_rest_positions[None], dtype=torch.float32)
        velocity = torch.full_like(previous, 0.2)
        acceleration = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float32)
        predicted = make_inertial_prediction(previous, velocity, 0.03, explicit_acceleration=acceleration)
        torch.testing.assert_close(predicted, previous + 0.03 * velocity + 0.03**2 * acceleration, rtol=0, atol=0)
        torch.testing.assert_close(make_inertial_prediction(previous, velocity, 0.03), previous + 0.03 * velocity)
        positions = (previous + 0.01).requires_grad_()
        terms = loss(positions, predicted)
        gradient = torch.autograd.grad(terms.inertia.sum(), positions)[0]
        expected = loss.lumped_mass[None, :, None] / 0.03**2 * (positions.detach() - predicted)
        torch.testing.assert_close(gradient, expected, rtol=3e-7, atol=1e-6)
        expected_energy = 0.5 * torch.sum(expected * (positions.detach() - predicted), dim=(1, 2))
        torch.testing.assert_close(terms.inertia, expected_energy, rtol=3e-7, atol=1e-6)

    def test_elasticity_gradient_and_float32_reference(self):
        """Check a nonaffine elastic gradient by differences and double precision."""
        import torch

        rest = generate_cuboid((1, 1, 1), cell_size=0.5)
        deformed = rest.corner_rest_positions @ np.array([[1.08, 0.02, 0], [0.03, 0.96, 0.04], [0, 0, 1.02]]).T
        deformed[-1] += [0.012, -0.008, 0.01]
        # Preserve the previous E=1500 Pa, nu=0.28 reference material.
        lam, mu = 1500 * 0.28 / (1.28 * 0.44), 1500 / (2 * 1.28)
        loss64 = HexImplicitEulerLoss(rest, lam, mu, 1000, 0.01, dtype=torch.float64)
        x64 = torch.tensor(deformed[None], dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(lambda x: loss64(x, x).elastic, (x64,), eps=1e-6, atol=2e-6, rtol=2e-5)
        )
        gradient64 = torch.autograd.grad(loss64(x64, x64).elastic.sum(), x64)[0]
        loss32 = HexImplicitEulerLoss(rest, lam, mu, 1000, 0.01)
        x32 = torch.tensor(deformed[None], dtype=torch.float32, requires_grad=True)
        gradient32 = torch.autograd.grad(loss32(x32, x32).elastic.sum(), x32)[0]
        torch.testing.assert_close(gradient32.double(), gradient64, atol=5e-4, rtol=8e-5)
        direction = torch.arange(x32.numel(), dtype=torch.float32).reshape_as(x32) / x32.numel() - 0.4
        epsilon = 1e-3
        finite_difference = (
            loss32(x32 + epsilon * direction, x32).elastic - loss32(x32 - epsilon * direction, x32).elastic
        ) / (2 * epsilon)
        torch.testing.assert_close(finite_difference, (gradient32 * direction).sum().reshape(1), atol=0.015, rtol=0.002)

    def test_near_rest_float32_energy_accuracy(self):
        """Reduce near-rest energy cancellation without changing its position gradient."""
        import torch

        rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        positions = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        z = positions[..., 2].clone()
        positions[..., 0] += 0.08 * z.square() + 0.001 * torch.sin(21 * positions[..., 1]) * z / 0.3
        positions[..., 1] += 0.02 * z.square()
        prediction = positions + torch.tensor([0.0003, -0.001, 0.0001])
        loss32 = HexImplicitEulerLoss(rest, _LAME_LAMBDA, _LAME_MU, 100, 0.04)
        loss64 = HexImplicitEulerLoss(rest, _LAME_LAMBDA, _LAME_MU, 100, 0.04, dtype=torch.float64)
        variable32 = positions.clone().requires_grad_()
        variable64 = positions.double().requires_grad_()
        energy32 = loss32(variable32, prediction).total.sum()
        energy64 = loss64(variable64, prediction.double()).total.sum()
        gradient32 = torch.autograd.grad(energy32, variable32)[0]
        gradient64 = torch.autograd.grad(energy64, variable64)[0]
        self.assertLess(abs(energy32.item() - energy64.item()), 5e-8)
        self.assertLess(((gradient32.double() - gradient64).norm() / gradient64.norm()).item(), 2e-5)
        generator = torch.Generator(device="cpu").manual_seed(11)
        direction = torch.randn(positions.shape, generator=generator, dtype=torch.float32)
        direction /= direction.norm()
        epsilon = 3e-4
        numerical = (
            (
                loss32(positions + epsilon * direction, prediction).total
                - loss32(positions - epsilon * direction, prediction).total
            )
            / (2 * epsilon)
        ).item()
        analytical = (gradient32 * direction).sum().item()
        self.assertLess(abs(analytical - numerical) / max(abs(analytical), abs(numerical)), 0.01)

    def test_invalid_jacobian_and_parameters(self):
        """Reject inverted or collapsed quadrature gradients without clamping."""
        import torch

        rest = generate_cuboid((1, 1, 1), cell_size=0.5)
        loss = HexImplicitEulerLoss(rest, _LAME_LAMBDA, _LAME_MU, 1000, 0.01)
        positions = torch.tensor(rest.corner_rest_positions[None], dtype=torch.float32)
        for scale in (0.0, -1.0):
            invalid = positions.clone()
            invalid[:, :, 0] *= scale
            with self.assertRaisesRegex(ValueError, "Jacobian"):
                loss(invalid, positions)
        # This corner fold leaves the center determinant positive (0.5),
        # but the far Gauss points detect an inverted region.
        invalid = positions.clone()
        invalid[:, -1, 0] -= 1.0
        with self.assertRaisesRegex(ValueError, "Jacobian"):
            loss(invalid, positions)
        for kwargs in (
            {"lame_lambda": -1},
            {"lame_lambda": np.nan},
            {"lame_lambda": np.inf},
            {"lame_mu": 0},
            {"lame_mu": -1},
            {"lame_mu": np.nan},
            {"lame_mu": np.inf},
            {"density": -1},
            {"time_step": 0},
            {"lame_lambda": [1, 2]},
            {"lame_mu": [1, 2]},
        ):
            values = {"lame_lambda": _LAME_LAMBDA, "lame_mu": _LAME_MU, "density": 1000, "time_step": 0.01}
            values.update(kwargs)
            with self.assertRaises(ValueError):
                HexImplicitEulerLoss(rest, **values)
        with self.assertRaises(ValueError):
            make_inertial_prediction(positions, positions, 0)
        with self.assertRaises(TypeError):
            loss(positions.double(), positions.double())


if __name__ == "__main__":
    unittest.main()
