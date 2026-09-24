# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU geometry and camera contracts for offline learned-solver video."""

import unittest

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.render_learned import _boundary_geometry, _camera_for_trajectory


class TestRenderLearned(unittest.TestCase):
    def test_boundary_faces_keep_outward_winding_and_surface_grid(self):
        """Triangulate exposed voxel faces and retain their shared grid edges."""
        grid = generate_cuboid((1, 1, 1))
        triangles, edges = _boundary_geometry(grid.cell_corner_indices)
        self.assertEqual(triangles.shape, (12, 3))
        self.assertEqual(edges.shape, (12, 2))
        self.assertEqual(len(np.unique(np.sort(edges, axis=1), axis=0)), 12)
        positions = grid.corner_rest_positions
        normals = np.cross(
            positions[triangles[:, 1]] - positions[triangles[:, 0]],
            positions[triangles[:, 2]] - positions[triangles[:, 0]],
        )
        centers = positions[triangles].mean(axis=1) - positions.mean(axis=0)
        self.assertTrue(np.all(np.sum(normals * centers, axis=1) > 0))

    def test_shared_internal_face_is_not_rendered_as_surface(self):
        """Exclude an internal cell face while keeping adjacent surface grid lines."""
        grid = generate_cuboid((2, 1, 1))
        triangles, edges = _boundary_geometry(grid.cell_corner_indices)
        self.assertEqual(triangles.shape, (20, 3))
        self.assertTrue(np.all(triangles >= 0))
        self.assertTrue(np.all(triangles < len(grid.corner_rest_positions)))
        self.assertGreater(len(edges), 12)

    def test_camera_uses_every_saved_valid_frame(self):
        """Frame a late large deformation with one camera used for the whole video."""
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        camera = _camera_for_trajectory(positions)
        self.assertGreater(camera["target"][0], 1.0)
        self.assertGreater(camera["distance"], 4.0)
        self.assertEqual(camera["bounds_max"], [4.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
