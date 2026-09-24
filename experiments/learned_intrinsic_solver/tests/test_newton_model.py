# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify native Newton models carrying hexahedral solver metadata."""

import importlib.util
import unittest
from dataclasses import replace

import numpy as np
import warp as wp

import newton
from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model


class TestNewtonHexModel(unittest.TestCase):
    def setUp(self):
        """Create a small canonical cuboid and its material-z-min fixed face."""
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1, origin=(0.2, -0.3, 0.4))
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0.4)

    def _model(self, **overrides):
        arguments = {"lame_lambda": 1000 * 0.3 / (1.3 * 0.4), "lame_mu": 1000 / 2.6, "density": 900.0}
        arguments.update(overrides)
        return build_newton_hex_model(self.rest, self.fixed, **arguments)

    def test_canonical_topology_and_registered_attributes(self):
        """Preserve canonical hex data in registered model attributes after finalize."""
        model = self._model(gravity=(0.0, -9.81, 0.0))
        self.assertIsInstance(model, newton.Model)
        self.assertTrue(model.device.is_cpu)
        self.assertEqual(model.particle_count, len(self.rest.corner_rest_positions))
        self.assertEqual(model.get_custom_frequency_count("learned_intrinsic:hex"), 12)
        for count in (model.tet_count, model.tri_count, model.spring_count, model.body_count, model.shape_count):
            self.assertEqual(count, 0)
        metadata = model.learned_intrinsic
        np.testing.assert_array_equal(metadata.cell_counts.numpy(), [[2, 2, 3]])
        np.testing.assert_array_equal(metadata.cell_corner_indices.numpy(), self.rest.cell_corner_indices)
        np.testing.assert_array_equal(
            metadata.rest_positions.numpy(), self.rest.corner_rest_positions.astype(np.float32)
        )
        np.testing.assert_array_equal(model.particle_q.numpy(), metadata.rest_positions.numpy())
        np.testing.assert_allclose(metadata.cell_size.numpy(), [0.1], rtol=1e-7)
        np.testing.assert_allclose(model.gravity.numpy(), [[0, -9.81, 0]], rtol=1e-7)
        for field in (metadata.cell_size, metadata.lame_lambda, metadata.lame_mu, metadata.density):
            self.assertEqual(field.dtype, wp.float32)
            self.assertTrue(field.device.is_cpu)
        self.assertEqual(metadata.fixed.dtype, wp.bool)
        self.assertEqual(metadata.rest_positions.dtype, wp.vec3)
        self.assertFalse(hasattr(metadata, "young_modulus"))
        self.assertFalse(hasattr(metadata, "poisson_ratio"))

    def test_mass_remains_physical_when_fixed(self):
        """Retain every corner's positive mass while clearing fixed ACTIVE flags."""
        density = np.arange(12, dtype=np.float32) * 10 + 900
        lame_lambda = np.arange(12, dtype=np.float32) * 25
        lame_mu = np.arange(12, dtype=np.float32) * 50 + 300
        model = self._model(density=density, lame_lambda=lame_lambda, lame_mu=lame_mu)
        expected = np.zeros(model.particle_count)
        np.add.at(
            expected, self.rest.cell_corner_indices.reshape(-1), np.repeat(density.astype(np.float64) * 0.1**3 / 8, 8)
        )
        mass = model.particle_mass.numpy()
        np.testing.assert_allclose(mass, expected, rtol=3e-7)
        self.assertGreater(mass[self.fixed].min(), 0)
        self.assertAlmostEqual(float(mass.sum()), float(np.sum(density) * 0.1**3), places=5)
        np.testing.assert_allclose(model.particle_inv_mass.numpy(), 1 / mass, rtol=1e-7)
        fixed_flags = np.zeros(model.particle_count, dtype=bool)
        fixed_flags[self.fixed] = True
        np.testing.assert_array_equal(model.learned_intrinsic.fixed.numpy(), fixed_flags)
        np.testing.assert_array_equal(
            (model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0, ~fixed_flags
        )
        np.testing.assert_array_equal(model.learned_intrinsic.density.numpy(), density)
        np.testing.assert_array_equal(model.learned_intrinsic.lame_lambda.numpy(), lame_lambda)
        np.testing.assert_array_equal(model.learned_intrinsic.lame_mu.numpy(), lame_mu)

    def test_zero_lambda_preserves_physical_mass(self):
        """Accept scalar zero lambda without changing density-derived particle masses."""
        zero_lambda = self._model(lame_lambda=0.0)
        baseline = self._model()
        np.testing.assert_array_equal(zero_lambda.learned_intrinsic.lame_lambda.numpy(), np.zeros(12, dtype=np.float32))
        np.testing.assert_array_equal(zero_lambda.particle_mass.numpy(), baseline.particle_mass.numpy())
        self.assertGreater(zero_lambda.particle_mass.numpy()[self.fixed].min(), 0)

    def test_standard_state_and_control_creation(self):
        """Create independent standard Newton states with stationary initial particles."""
        model = self._model()
        initial = model.particle_q.numpy().copy()
        first, second, control = model.state(), model.state(), model.control()
        self.assertIsInstance(first, newton.State)
        self.assertIsInstance(control, newton.Control)
        np.testing.assert_array_equal(first.particle_q.numpy(), initial)
        np.testing.assert_array_equal(first.particle_qd.numpy(), np.zeros_like(initial))
        np.testing.assert_array_equal(first.particle_f.numpy(), np.zeros_like(initial))
        first.particle_q.assign(initial + np.float32(0.1))
        np.testing.assert_array_equal(second.particle_q.numpy(), initial)
        np.testing.assert_array_equal(model.particle_q.numpy(), initial)
        np.testing.assert_allclose(model.gravity.numpy(), [[0, 0, -9.81]], rtol=1e-7)

    def test_invalid_parameters_and_noncanonical_topology(self):
        """Reject invalid materials, pins, gravity, and altered canonical connectivity."""
        for arguments in (
            {"lame_lambda": -0.1},
            {"lame_lambda": np.inf},
            {"lame_lambda": [0, 1]},
            {"lame_mu": 0},
            {"lame_mu": -1},
            {"lame_mu": float("nan")},
            {"density": float("nan")},
            {"density": 1e-45},
            {"gravity": (0, 1)},
            {"gravity": (0, np.inf, 1)},
        ):
            with self.assertRaises(ValueError):
                self._model(**arguments)
        for fixed in ([0, 0], [-1], [len(self.rest.corner_rest_positions)], [[0]], [0.5], [True]):
            with self.assertRaises((ValueError, TypeError)):
                build_newton_hex_model(self.rest, fixed, lame_lambda=500, lame_mu=400, density=900)
        altered = self.rest.cell_corner_indices.copy()
        altered[0, 0], altered[0, 1] = altered[0, 1], altered[0, 0]
        with self.assertRaises(ValueError):
            build_newton_hex_model(
                replace(self.rest, cell_corner_indices=altered),
                self.fixed,
                lame_lambda=500,
                lame_mu=400,
                density=900,
            )


if __name__ == "__main__":
    unittest.main()
