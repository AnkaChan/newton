# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from research.principal_stretch.render_reference_gallery import (
    _exact_scalar,
    array_sha256,
    camera_from_sequence_bounds,
    ping_pong_stored_state_indices,
)


class TestRenderReferenceGallery(unittest.TestCase):
    def test_ping_pong_schedule_uses_only_stored_states(self):
        schedule = ping_pong_stored_state_indices(8, 3)
        self.assertEqual(len(schedule), 16 * 3)
        self.assertEqual(schedule[:6], (0, 0, 0, 1, 1, 1))
        self.assertEqual(schedule[-6:], (2, 2, 2, 1, 1, 1))
        self.assertEqual(set(schedule), set(range(9)))

    def test_camera_is_fixed_from_complete_sequence_bounds(self):
        q = np.array(
            [
                [[-1.0, -2.0, -0.5], [1.0, -2.0, -0.5], [1.0, 2.0, 0.5], [-1.0, 2.0, 0.5]],
                [[-1.5, -2.0, -0.5], [1.0, -2.0, -0.5], [1.0, 3.0, 0.5], [-1.0, 2.0, 0.5]],
            ],
            dtype=np.float64,
        )
        camera = camera_from_sequence_bounds(q)
        self.assertEqual(camera.bounds_min, (-1.5, -2.0, -0.5))
        self.assertEqual(camera.bounds_max, (1.0, 3.0, 0.5))
        self.assertEqual(camera.target, (-0.25, 0.5, 0.0))
        self.assertGreater(np.linalg.norm(np.subtract(camera.position, camera.target)), np.linalg.norm([2.5, 5.0, 1.0]))

    def test_array_hash_binds_dtype_shape_and_payload(self):
        values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        self.assertEqual(array_sha256(values), array_sha256(values.copy(order="F")))
        self.assertNotEqual(array_sha256(values), array_sha256(values.astype(np.float32)))
        self.assertNotEqual(array_sha256(values), array_sha256(values.reshape(3, 2, 2)))
        changed = values.copy()
        changed[0, 0, 0] = 1.0
        self.assertNotEqual(array_sha256(values), array_sha256(changed))

    def test_rejects_invalid_schedule_and_camera(self):
        with self.assertRaises(ValueError):
            ping_pong_stored_state_indices(0, 1)
        with self.assertRaises(ValueError):
            ping_pong_stored_state_indices(8, 0)
        with self.assertRaises(ValueError):
            camera_from_sequence_bounds(np.zeros((9, 4, 3), dtype=np.float64))

    def test_exact_scalar_prefers_canonical_shape(self):
        value, shape, legacy = _exact_scalar(np.asarray(1.0 / 300.0, dtype=np.float32), np.float32, "dt")
        self.assertEqual(value, float(np.float32(1.0 / 300.0)))
        self.assertEqual(shape, ())
        self.assertFalse(legacy)

    def test_exact_scalar_explicitly_accepts_legacy_singleton_vector(self):
        value, shape, legacy = _exact_scalar(np.asarray([1.0 / 300.0], dtype=np.float32), np.float32, "dt")
        self.assertEqual(value, float(np.float32(1.0 / 300.0)))
        self.assertEqual(shape, (1,))
        self.assertTrue(legacy)

    def test_exact_scalar_rejects_wrong_shape_or_dtype(self):
        with self.assertRaises(ValueError):
            _exact_scalar(np.asarray([1.0 / 300.0, 1.0 / 300.0], dtype=np.float32), np.float32, "dt")
        with self.assertRaises(ValueError):
            _exact_scalar(np.asarray(1.0 / 300.0, dtype=np.float64), np.float32, "dt")


if __name__ == "__main__":
    unittest.main()
