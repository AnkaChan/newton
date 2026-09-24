# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify compatible multiscale random initial geometries."""

import unittest

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.multiscale import (
    DeformationLevel,
    build_hierarchy,
    generate_multiscale,
    interpolate_control_grid,
    screen_geometry,
)


class TestMultiscaleDeformation(unittest.TestCase):
    def test_automatic_odd_hierarchy(self):
        """Cover shifted odd and singleton-axis grids without assuming divisibility."""
        for counts in ((7, 5, 19), (1, 3, 17)):
            rest = generate_cuboid(counts, cell_size=0.06, origin=(0.7, -0.3, 1.2))
            levels = build_hierarchy(rest, max_levels=3)
            self.assertEqual(levels[-1].control_counts, tuple(n + 1 for n in counts))
            self.assertLessEqual(max(n - 1 for n in levels[0].control_counts), 4)
            self.assertAlmostEqual(sum(level.amplitude_m for level in levels), 0.5 * min(counts) * 0.06)
            sample = generate_multiscale(rest, seed=4)
            fixed = rest.corner_rest_positions[:, 2] == 1.2
            np.testing.assert_array_equal(sample.positions[fixed], rest.corner_rest_positions[fixed])
            origin = rest.corner_rest_positions.min(axis=0)
            extent = np.array(counts) * rest.cell_size
            for control in sample.controls:
                ends = interpolate_control_grid(
                    control, np.array([origin, origin + extent]), origin=origin, extent=extent
                )
                np.testing.assert_allclose(ends, [control[0, 0, 0], control[-1, -1, -1]], atol=1e-16)

    def test_affine_interpolation(self):
        """Interpolate an affine displacement exactly on nonmatching grids."""
        rest = generate_cuboid((4, 3, 7), cell_size=0.1, origin=(0.3, -0.2, 0.7))
        origin = rest.corner_rest_positions.min(axis=0)
        extent = np.array(rest.cell_counts) * rest.cell_size
        xyz = np.stack(
            np.meshgrid(
                *[np.linspace(origin[a], origin[a] + extent[a], n) for a, n in enumerate((3, 2, 4))], indexing="ij"
            ),
            axis=-1,
        )
        matrix = np.array([[0.2, 0.3, -0.1], [-0.2, 0.1, 0.4], [0.1, -0.3, 0.2]])
        control = xyz @ matrix.T + [0.1, 0.2, -0.3]
        sampled = interpolate_control_grid(control, rest.corner_rest_positions, origin=origin, extent=extent)
        np.testing.assert_allclose(sampled, rest.corner_rest_positions @ matrix.T + [0.1, 0.2, -0.3], atol=3e-16)

    def test_seed_source_and_clamp(self):
        """Reproduce each level without changing the source or global random state."""
        rest = generate_cuboid((3, 4, 8), cell_size=0.1)
        before = rest.corner_rest_positions.copy()
        rng_before = np.random.get_state()  # noqa: NPY002 - verify the legacy global RNG is untouched
        first = generate_multiscale(rest, seed=13)
        second = generate_multiscale(rest, seed=13)
        third = generate_multiscale(rest, seed=14)
        rng_after = np.random.get_state()  # noqa: NPY002 - verify the legacy global RNG is untouched
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])
        np.testing.assert_array_equal(rest.corner_rest_positions, before)
        np.testing.assert_array_equal(first.positions, second.positions)
        self.assertFalse(np.array_equal(first.positions, third.positions))
        fixed = rest.corner_rest_positions[:, 2] == 0
        for control, displacement in zip(first.controls, first.level_displacements, strict=True):
            np.testing.assert_array_equal(control[:, :, 0], 0)
            np.testing.assert_array_equal(displacement[fixed], 0)
        np.testing.assert_array_equal(first.positions[fixed], before[fixed])
        np.testing.assert_allclose(
            first.positions, before + first.effective_scale * first.level_displacements.sum(axis=0), atol=1e-16
        )

    def test_zero_amplitude_and_independent_streams(self):
        """Keep zero-strength geometry exact and preserve other levels' random draws."""
        rest = generate_cuboid((5, 4, 12), cell_size=0.1)
        levels = build_hierarchy(rest)
        zero = tuple(DeformationLevel(level.name, level.control_counts, 0.0) for level in levels)
        sample = generate_multiscale(rest, seed=3, levels=zero)
        np.testing.assert_array_equal(sample.positions, rest.corner_rest_positions)
        self.assertEqual(sample.effective_scale, 1)
        modified = (DeformationLevel(levels[0].name, levels[0].control_counts, 0), *levels[1:])
        original = generate_multiscale(rest, seed=3)
        altered = generate_multiscale(rest, seed=3, levels=modified)
        for a, b in zip(original.controls[1:], altered.controls[1:], strict=True):
            np.testing.assert_array_equal(a, b)

    def test_orientation_and_backtracking(self):
        """Screen off-center inversion and record deterministic amplitude reduction."""
        rest = generate_cuboid((1, 1, 1))
        identity = screen_geometry(rest, rest.corner_rest_positions)
        self.assertAlmostEqual(identity["min_tet_volume_ratio"], 1)
        self.assertAlmostEqual(identity["min_sampled_jacobian"], 1)
        collapsed_corner = rest.corner_rest_positions.copy()
        collapsed_corner[7] = [0.2, 0.2, 0.2]
        screen = screen_geometry(rest, collapsed_corner)
        self.assertLess(screen["min_sampled_jacobian"], 0)
        large = (DeformationLevel("coarse", (2, 2, 5), 4.0),)
        rest = generate_cuboid((2, 2, 4), cell_size=0.1)
        sample = generate_multiscale(rest, seed=1, levels=large)
        self.assertGreater(sample.backtracking_steps, 0)
        self.assertEqual(sample.effective_scale, 0.5**sample.backtracking_steps)
        self.assertGreaterEqual(sample.screen["min_tet_volume_ratio"], 0.2)
        self.assertGreaterEqual(sample.screen["min_sampled_jacobian"], 0.2)

    def test_twenty_default_initial_states(self):
        """Keep all twenty full-size seeds and their contribution views finite and valid."""
        rest = generate_cuboid((10, 10, 40), cell_size=0.025)
        for seed in range(20):
            with self.subTest(seed=seed):
                sample = generate_multiscale(rest, seed=seed)
                self.assertTrue(np.isfinite(sample.positions).all())
                self.assertGreaterEqual(sample.screen["min_tet_volume_ratio"], 0.2)
                self.assertGreaterEqual(sample.screen["min_sampled_jacobian"], 0.2)
                self.assertGreater(
                    np.sqrt(np.mean(np.sum((sample.positions - rest.corner_rest_positions) ** 2, axis=1))), 0.01
                )
                for screen in sample.level_screens:
                    self.assertGreaterEqual(screen["min_tet_volume_ratio"], 0.2)
                    self.assertGreaterEqual(screen["min_sampled_jacobian"], 0.2)


if __name__ == "__main__":
    unittest.main()
