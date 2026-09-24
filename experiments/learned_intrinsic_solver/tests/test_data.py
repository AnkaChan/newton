# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.learned_intrinsic_solver.data import augment_grid, generate_cuboid


class TestCuboidGeneration(unittest.TestCase):
    def test_shared_corners_and_rest_geometry(self):
        """Build two adjacent cubes with one shared face and the requested dimensions."""
        grid = generate_cuboid((2, 1, 1), cell_size=0.5, origin=(-1.0, 2.0, 3.0))

        self.assertEqual(grid.corner_rest_positions.shape, (12, 3))
        self.assertEqual(grid.cell_corner_indices.shape, (2, 8))
        self.assertEqual(len(np.unique(grid.corner_rest_positions, axis=0)), 12)
        shared = np.intersect1d(grid.cell_corner_indices[0], grid.cell_corner_indices[1])
        self.assertEqual(len(shared), 4)
        np.testing.assert_allclose(grid.corner_rest_positions[shared, 0], -0.5)
        np.testing.assert_allclose(grid.corner_rest_positions.min(axis=0), (-1.0, 2.0, 3.0))
        np.testing.assert_allclose(grid.corner_rest_positions.max(axis=0), (0.0, 2.5, 3.5))
        np.testing.assert_allclose(grid.cell_rest_centers, [[-0.75, 2.25, 3.25], [-0.25, 2.25, 3.25]])

        corners = grid.corner_rest_positions[grid.cell_corner_indices[0]]
        np.testing.assert_allclose(corners[4] - corners[0], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(corners[2] - corners[0], [0.0, 0.5, 0.0])
        np.testing.assert_allclose(corners[1] - corners[0], [0.0, 0.0, 0.5])

    def test_face_connectivity_and_exposed_flags(self):
        """Connect shared material faces without connecting opposite cuboid boundaries."""
        grid = generate_cuboid((2, 1, 1))
        np.testing.assert_array_equal(grid.cell_neighbors, [[-1, 1, -1, -1, -1, -1], [0, -1, -1, -1, -1, -1]])
        np.testing.assert_array_equal(
            grid.cell_exposed_faces,
            [[True, False, True, True, True, True], [False, True, True, True, True, True]],
        )
        self.assertEqual(np.count_nonzero(grid.cell_exposed_faces), 10)

        interior = generate_cuboid((3, 3, 3))
        np.testing.assert_array_equal(interior.cell_neighbors[13], [4, 22, 10, 16, 12, 14])
        self.assertFalse(interior.cell_exposed_faces[13].any())
        self.assertEqual(np.count_nonzero(interior.cell_exposed_faces), 54)

    def test_rest_state_and_single_voxel(self):
        """Initialize undeformed axes and stationary cell velocities even for one voxel."""
        grid = generate_cuboid((1, 1, 1), cell_size=0.25)
        self.assertEqual(grid.cell_counts, (1, 1, 1))
        self.assertEqual(grid.cell_size, 0.25)
        self.assertEqual(grid.corner_rest_positions.shape, (8, 3))
        np.testing.assert_array_equal(grid.cell_deformation, np.eye(3)[None])
        np.testing.assert_array_equal(grid.cell_velocity, [[0.0, 0.0, 0.0]])
        self.assertTrue(grid.cell_exposed_faces.all())

    def test_invalid_generation_parameters(self):
        """Reject malformed dimensions, nonpositive spacing, and nonfinite origins."""
        for counts in [(0, 1, 1), (-1, 1, 1), (1.5, 1, 1), (True, 1, 1), (1, 1), (1, 1, 1, 1)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                generate_cuboid(counts)
        for cell_size in [0.0, -0.1, np.nan, np.inf]:
            with self.subTest(cell_size=cell_size), self.assertRaises(ValueError):
                generate_cuboid((1, 1, 1), cell_size=cell_size)
        for origin in [(0, 0), (0, np.nan, 0), (0, 0, np.inf)]:
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                generate_cuboid((1, 1, 1), origin=origin)


class TestGridAugmentation(unittest.TestCase):
    def test_seeded_per_cell_augmentation(self):
        """Reproduce seeded random fields while allowing different cells and seeds to differ."""
        source = generate_cuboid((3, 2, 2))
        first = augment_grid(source, deformation_amplitude=0.2, velocity_amplitude=0.7, seed=123)
        repeated = augment_grid(source, deformation_amplitude=0.2, velocity_amplitude=0.7, seed=123)
        different = augment_grid(source, deformation_amplitude=0.2, velocity_amplitude=0.7, seed=124)

        np.testing.assert_array_equal(first.cell_deformation, repeated.cell_deformation)
        np.testing.assert_array_equal(first.cell_velocity, repeated.cell_velocity)
        self.assertFalse(np.array_equal(first.cell_deformation, different.cell_deformation))
        self.assertFalse(np.array_equal(first.cell_velocity, different.cell_velocity))
        self.assertGreater(len(np.unique(first.cell_deformation.reshape(12, 9), axis=0)), 1)
        self.assertGreater(len(np.unique(first.cell_velocity, axis=0)), 1)
        self.assertLessEqual(np.abs(first.cell_velocity).max(), 0.7)

    def test_positive_orientation_and_bounded_deformation(self):
        """Keep identity-based deformation samples nonsingular with positive orientation."""
        augmented = augment_grid(generate_cuboid((8, 4, 3)), deformation_amplitude=0.32, velocity_amplitude=0.0, seed=4)
        self.assertTrue(np.all(np.linalg.det(augmented.cell_deformation) > 0.0))
        self.assertGreaterEqual(np.linalg.svd(augmented.cell_deformation, compute_uv=False).min(), 0.04 - 1.0e-12)
        self.assertLessEqual(np.abs(augmented.cell_deformation - np.eye(3)).max(), 0.32)
        np.testing.assert_array_equal(augmented.cell_velocity, np.zeros((96, 3)))

    def test_augmentation_preserves_source_and_rest_geometry(self):
        """Leave the source sample unchanged and give the result independent geometry arrays."""
        source = generate_cuboid((2, 1, 1))
        corners_before = source.corner_rest_positions.copy()
        augmented = augment_grid(source, seed=17)

        np.testing.assert_array_equal(source.cell_deformation, np.repeat(np.eye(3)[None], 2, axis=0))
        np.testing.assert_array_equal(source.cell_velocity, np.zeros((2, 3)))
        for name in ("corner_rest_positions", "cell_corner_indices", "cell_rest_centers", "cell_neighbors"):
            np.testing.assert_array_equal(getattr(augmented, name), getattr(source, name))
            self.assertFalse(np.shares_memory(getattr(augmented, name), getattr(source, name)))
        augmented.corner_rest_positions[0, 0] += 10.0
        np.testing.assert_array_equal(source.corner_rest_positions, corners_before)

    def test_zero_amplitudes_preserve_existing_state(self):
        """Make zero-amplitude augmentation preserve a previously deformed moving sample."""
        source = augment_grid(generate_cuboid((2, 2, 1)), seed=31)
        unchanged = augment_grid(source, deformation_amplitude=0.0, velocity_amplitude=0.0, seed=99)
        np.testing.assert_array_equal(unchanged.cell_deformation, source.cell_deformation)
        np.testing.assert_array_equal(unchanged.cell_velocity, source.cell_velocity)

    def test_deformation_composes_on_the_left(self):
        """Apply a spatial deformation increment to existing axes instead of resetting them."""
        rest = generate_cuboid((1, 1, 1))
        stretched = generate_cuboid((1, 1, 1))
        stretch = np.diag([2.0, 3.0, 4.0])
        stretched.cell_deformation[0] = stretch

        delta = augment_grid(rest, seed=12)
        result = augment_grid(stretched, seed=12)
        np.testing.assert_allclose(result.cell_deformation[0], delta.cell_deformation[0] @ stretch)

    def test_velocity_adds_to_existing_motion(self):
        """Perturb existing cell velocities without replacing their base motion."""
        stationary = generate_cuboid((2, 1, 1))
        moving = generate_cuboid((2, 1, 1))
        base_velocity = [3.0, -2.0, 5.0]
        moving.cell_velocity[:] = base_velocity

        delta = augment_grid(stationary, velocity_amplitude=0.4, seed=7)
        result = augment_grid(moving, velocity_amplitude=0.4, seed=7)
        np.testing.assert_allclose(result.cell_velocity, delta.cell_velocity + base_velocity)
        self.assertLessEqual(np.abs(result.cell_velocity - moving.cell_velocity).max(), 0.4)

    def test_invalid_augmentation_parameters(self):
        """Reject amplitudes that are negative, nonfinite, or cannot guarantee valid increments."""
        grid = generate_cuboid((1, 1, 1))
        for amplitude in [-0.1, np.nan, np.inf, 1.0 / 3.0, 0.5]:
            with self.subTest(deformation_amplitude=amplitude), self.assertRaises(ValueError):
                augment_grid(grid, deformation_amplitude=amplitude)
        for amplitude in [-0.1, np.nan, np.inf]:
            with self.subTest(velocity_amplitude=amplitude), self.assertRaises(ValueError):
                augment_grid(grid, velocity_amplitude=amplitude)

    def test_invalid_source_fields(self):
        """Reject singular, inverted, or nonfinite deformation and invalid velocity fields."""
        for deformation in [np.zeros((3, 3)), np.diag([-1.0, 1.0, 1.0]), np.full((3, 3), np.nan)]:
            grid = generate_cuboid((1, 1, 1))
            grid.cell_deformation[0] = deformation
            with self.subTest(deformation=deformation), self.assertRaises(ValueError):
                augment_grid(grid)
        grid = generate_cuboid((1, 1, 1))
        grid.cell_velocity[0, 1] = np.inf
        with self.assertRaises(ValueError):
            augment_grid(grid)


class TestDataCommand(unittest.TestCase):
    def test_export_consumable_sample(self):
        """Export cuboid topology and augmented cell fields in a pickle-free NumPy archive."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "sample.npz"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.learned_intrinsic_solver.data",
                    "--cells",
                    "2",
                    "1",
                    "1",
                    "--cell-size",
                    "0.25",
                    "--deformation-amplitude",
                    "0.1",
                    "--velocity-amplitude",
                    "0.3",
                    "--seed",
                    "42",
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(output, allow_pickle=False) as sample:
                np.testing.assert_array_equal(sample["cell_counts"], [2, 1, 1])
                self.assertEqual(float(sample["cell_size"]), 0.25)
                self.assertEqual(sample["corner_rest_positions"].shape, (12, 3))
                self.assertEqual(sample["cell_corner_indices"].shape, (2, 8))
                self.assertEqual(sample["cell_deformation"].shape, (2, 3, 3))
                self.assertEqual(sample["cell_velocity"].shape, (2, 3))
                self.assertEqual(np.count_nonzero(sample["cell_exposed_faces"]), 10)
                self.assertTrue(np.all(np.linalg.det(sample["cell_deformation"]) > 0.0))
                self.assertLessEqual(np.abs(sample["cell_velocity"]).max(), 0.3)


if __name__ == "__main__":
    unittest.main()
