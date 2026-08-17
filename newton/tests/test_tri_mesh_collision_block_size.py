# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.solvers.vbd.tri_mesh_collision import (
    _adaptive_collision_detection_block_size,
    _resolve_collision_detection_block_sizes,
)


class TestTriMeshCollisionBlockSize(unittest.TestCase):
    def test_adaptive_block_size_boundaries(self):
        """Select the largest block size that supplies the target grid."""
        expected_sizes = {
            57: 8,
            56: 4,
            29: 4,
            28: 2,
            15: 2,
            14: 1,
            8: 1,
            7: 1,
            0: 1,
        }
        for primitive_count, expected_size in expected_sizes.items():
            with self.subTest(primitive_count=primitive_count):
                self.assertEqual(
                    _adaptive_collision_detection_block_size(primitive_count, sm_count=1),
                    expected_size,
                )

    def test_adaptive_block_size_cloth_franka(self):
        """Resolve independent cloth Franka launch sizes on an L40."""
        self.assertEqual(
            _resolve_collision_detection_block_sizes(
                None,
                is_cuda=True,
                sm_count=142,
                particle_count=6436,
                edge_count=19174,
            ),
            (8, 4, 8),
        )

    def test_default_block_size_on_cpu(self):
        """Keep the default block size unchanged on CPU."""
        self.assertEqual(
            _resolve_collision_detection_block_sizes(
                None,
                is_cuda=False,
                sm_count=0,
                particle_count=1,
                edge_count=1,
            ),
            (8, 8, 8),
        )

    def test_explicit_block_size_preserved(self):
        """Preserve an explicit block size for both detector kernels."""
        self.assertEqual(
            _resolve_collision_detection_block_sizes(
                24,
                is_cuda=True,
                sm_count=142,
                particle_count=1,
                edge_count=1,
            ),
            (24, 24, 24),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
