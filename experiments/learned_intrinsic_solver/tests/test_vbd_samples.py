# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check initialization projection independently of the GPU solver."""

import unittest
from dataclasses import replace

import numpy as np

from experiments.learned_intrinsic_solver.data import augment_grid, generate_cuboid
from experiments.learned_intrinsic_solver.vbd_samples import CornerProjector, _ordering_map


class TestCornerProjection(unittest.TestCase):
    def test_rest_and_velocity(self):
        """Preserve identity geometry and average a constant velocity off the clamp."""
        rest = generate_cuboid((2, 3, 4), cell_size=0.1)
        projector = CornerProjector(rest)
        sample = replace(rest, cell_velocity=np.tile((0.1, -0.2, 0.3), (24, 1)))
        positions, velocity, residual = projector.project(sample)
        np.testing.assert_array_equal(positions, rest.corner_rest_positions)
        np.testing.assert_allclose(velocity[projector.free], np.tile((0.1, -0.2, 0.3), (48, 1)))
        np.testing.assert_array_equal(velocity[projector.fixed], 0)
        self.assertEqual(residual, 0)

    def test_compatible_affine_field(self):
        """Recover an affine deformation that leaves the clamped plane unchanged."""
        rest = generate_cuboid((3, 2, 4), cell_size=0.1)
        deformation = np.array([[1, 0, 0.13], [0, 1, -0.09], [0, 0, 1.07]])
        sample = replace(rest, cell_deformation=np.tile(deformation, (24, 1, 1)))
        positions, _, residual = CornerProjector(rest).project(sample)
        np.testing.assert_allclose(positions, rest.corner_rest_positions @ deformation.T, atol=1e-14)
        self.assertLess(residual, 1e-14)

    def test_seed_and_ordering(self):
        """Reproduce seeded projection and map corners into x-fast Newton order."""
        rest = generate_cuboid((2, 3, 4), cell_size=0.1)
        projector = CornerProjector(rest)
        first = projector.project(augment_grid(rest, seed=7))
        second = CornerProjector(rest).project(augment_grid(rest, seed=7))
        third = projector.project(augment_grid(rest, seed=8))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(np.array_equal(first[0], third[0]))
        np.testing.assert_array_equal(first[0][projector.fixed], rest.corner_rest_positions[projector.fixed])
        mapped = np.empty_like(rest.corner_rest_positions)
        mapped[_ordering_map(rest.cell_counts)] = rest.corner_rest_positions
        expected = np.array([(x, y, z) for z in range(5) for y in range(4) for x in range(3)]) * 0.1
        np.testing.assert_array_equal(mapped, expected)


if __name__ == "__main__":
    unittest.main()
