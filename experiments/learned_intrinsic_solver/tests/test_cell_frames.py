# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify cell-center deformation and its local polar frame."""

import unittest

import numpy as np

from experiments.learned_intrinsic_solver.cell_frames import compute_cell_frames, generate_frame_sample
from experiments.learned_intrinsic_solver.data import generate_cuboid


class TestCellFrames(unittest.TestCase):
    def test_rest_geometry(self):
        """Recover identity axes and frames from canonical cells of arbitrary size."""
        rest = generate_cuboid((2, 3, 4), cell_size=0.07, origin=(0.2, -0.4, 0.3))
        result = compute_cell_frames(rest, rest.corner_rest_positions)
        identity = np.tile(np.eye(3), (24, 1, 1))
        np.testing.assert_allclose(result.deformation, identity, atol=2e-15)
        np.testing.assert_allclose(result.frames, identity, atol=2e-15)
        np.testing.assert_allclose(result.local_axes, identity, atol=2e-15)
        np.testing.assert_allclose(result.centers, rest.cell_rest_centers, atol=2e-15)
        self.assertTrue(result.valid.all())

    def test_affine_rotation_and_stretch(self):
        """Recover known rotation and symmetric stretch without normalizing axes."""
        rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        stretch = np.array([[1.3, 0.2, -0.1], [0.2, 0.8, 0.06], [-0.1, 0.06, 1.1]])
        deformation = rotation @ stretch
        positions = rest.corner_rest_positions @ deformation.T + [0.2, -0.1, 0.4]
        result = compute_cell_frames(rest, positions)
        np.testing.assert_allclose(result.deformation, np.tile(deformation, (12, 1, 1)), atol=2e-15)
        np.testing.assert_allclose(result.frames, np.tile(rotation, (12, 1, 1)), atol=3e-15)
        np.testing.assert_allclose(result.local_axes, np.tile(stretch, (12, 1, 1)), atol=3e-15)
        self.assertGreater(np.linalg.norm(result.local_axes[0, :, 0]), 1.3)

    def test_warped_cell_center_gradient(self):
        """Evaluate the center gradient for a warped hex rather than fitting one edge."""
        rest = generate_cuboid((1, 1, 1), cell_size=0.2)
        positions = rest.corner_rest_positions.copy()
        offset = np.array([0.03, -0.02, 0.01])
        positions[rest.cell_corner_indices[0, 7]] += offset
        result = compute_cell_frames(rest, positions)
        expected = np.eye(3) + np.tile(offset[:, None] / (4 * rest.cell_size), (1, 3))
        np.testing.assert_allclose(result.deformation[0], expected, atol=1e-15)
        np.testing.assert_allclose(result.centers[0], [0.1, 0.1, 0.1] + offset / 8, atol=1e-15)

    def test_rigid_equivariance(self):
        """Rotate frames with the object while preserving axes in their local frame."""
        rest, positions, _ = generate_frame_sample(cell_counts=(2, 3, 4), seed=19)
        original = compute_cell_frames(rest, positions)
        rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        translation = np.array([0.3, -0.1, 0.2])
        moved = compute_cell_frames(rest, positions @ rotation.T + translation)
        np.testing.assert_allclose(moved.frames, rotation @ original.frames, atol=1e-14)
        np.testing.assert_allclose(moved.local_axes, original.local_axes, atol=1e-14)
        np.testing.assert_allclose(moved.deformation, rotation @ original.deformation, atol=1e-14)
        np.testing.assert_allclose(moved.centers, original.centers @ rotation.T + translation, atol=1e-15)

    def test_invalid_orientation(self):
        """Flag inverted and collapsed cells without manufacturing a proper frame."""
        rest = generate_cuboid((1, 1, 1))
        for diagonal in ((-1, 1, 1), (1, 0, 1)):
            result = compute_cell_frames(rest, rest.corner_rest_positions * diagonal)
            self.assertFalse(result.valid[0])
            np.testing.assert_array_equal(result.frames, 0)
            np.testing.assert_array_equal(result.local_axes, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_cell_frames(rest, np.full_like(rest.corner_rest_positions, np.nan))

    def test_seeded_shared_corners(self):
        """Reproduce the actual projected geometry and distinguish different seeds."""
        rest, first, residual = generate_frame_sample(cell_counts=(2, 3, 4), seed=12)
        _, second, repeated_residual = generate_frame_sample(cell_counts=(2, 3, 4), seed=12)
        _, third, _ = generate_frame_sample(cell_counts=(2, 3, 4), seed=13)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(residual, repeated_residual)
        self.assertFalse(np.array_equal(first, third))
        fixed = rest.corner_rest_positions[:, 2] == 0
        np.testing.assert_array_equal(first[fixed], rest.corner_rest_positions[fixed])
        result = compute_cell_frames(rest, first)
        self.assertTrue(result.valid.all())
        np.testing.assert_allclose(result.frames @ result.local_axes, result.deformation, atol=2e-15)
        np.testing.assert_allclose(
            result.frames.transpose(0, 2, 1) @ result.frames, np.tile(np.eye(3), (24, 1, 1)), atol=2e-15
        )


if __name__ == "__main__":
    unittest.main()
