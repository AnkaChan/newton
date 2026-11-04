# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check a bag grip aperture change against the actual original hand geometry."""

import unittest

import numpy as np
import warp as wp

import newton
from experiments.collision_benchmark.bag_robot_scenes import BagRobotScene


@unittest.skipUnless(wp.is_cuda_available(), "The production hand texture SDF requires CUDA")
class BagGripTests(unittest.TestCase):
    def test_aperture_changes_only_grip_and_keeps_release_open(self):
        """Preserve free bag material while narrowing the original fingertips."""
        narrower = BagRobotScene("m3_franka_tote_empty", device="cuda:0", use_cuda_graph=False, closed_opening=0.002)
        baseline = BagRobotScene("m3_franka_tote_empty", device="cuda:0", use_cuda_graph=False)
        for name in (
            "particle_q",
            "particle_qd",
            "particle_mass",
            "particle_inv_mass",
            "particle_radius",
            "particle_flags",
            "tri_indices",
            "tri_materials",
            "edge_indices",
            "edge_bending_properties",
            "shape_type",
            "shape_scale",
            "shape_transform",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
        ):
            with self.subTest(array=name):
                np.testing.assert_array_equal(
                    getattr(baseline.model, name).numpy(), getattr(narrower.model, name).numpy()
                )
        for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
            self.assertEqual(getattr(baseline.model, name), getattr(narrower.model, name))
        self.assertTrue(np.all(narrower.model.particle_inv_mass.numpy() > 0))
        self.assertTrue(np.all(narrower.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)))
        np.testing.assert_array_equal(baseline.grip_centers, narrower.grip_centers)

        for original, changed in zip(baseline.initial_geometry, narrower.initial_geometry, strict=True):
            self.assertEqual(original["vertices_inside_robot_solids"], 0)
            self.assertEqual(changed["vertices_inside_robot_solids"], 0)
            self.assertAlmostEqual(original["physical_tip_aperture_m"], 0.00396, delta=1e-8)
            self.assertAlmostEqual(changed["physical_tip_aperture_m"], 0.00196, delta=1e-8)
        for original, changed in zip(baseline.drivers, narrower.drivers, strict=True):
            for name in ("times", "positions", "rotations"):
                np.testing.assert_array_equal(getattr(original, name).numpy(), getattr(changed, name).numpy())
            np.testing.assert_array_equal(original.openings.numpy()[:7], np.full(7, 0.004, dtype=np.float32))
            np.testing.assert_array_equal(changed.openings.numpy()[:7], np.full(7, 0.002, dtype=np.float32))
            np.testing.assert_array_equal(changed.openings.numpy()[7:], np.full(3, 0.08, dtype=np.float32))
            np.testing.assert_array_equal(original.openings.numpy()[7:], changed.openings.numpy()[7:])


if __name__ == "__main__":
    unittest.main()
