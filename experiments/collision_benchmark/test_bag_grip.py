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
    def _assert_same_soft_material(self, baseline, changed):
        """Compare actual soft state, surface geometry and contact materials."""
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
                    getattr(baseline.model, name).numpy(), getattr(changed.model, name).numpy()
                )
        for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
            self.assertEqual(getattr(baseline.model, name), getattr(changed.model, name))
        self.assertTrue(np.all(changed.model.particle_inv_mass.numpy() > 0))
        self.assertTrue(np.all(changed.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)))

    def test_aperture_changes_only_grip_and_keeps_release_open(self):
        """Preserve free bag material while narrowing the original fingertips."""
        narrower = BagRobotScene("m3_franka_tote_empty", device="cuda:0", use_cuda_graph=False, closed_opening=0.002)
        baseline = BagRobotScene("m3_franka_tote_empty", device="cuda:0", use_cuda_graph=False)
        self._assert_same_soft_material(baseline, narrower)
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

    def test_lower_pinch_seats_both_handle_vertices_inside_finite_pad_faces(self):
        """Shift only the grip trajectory while seating both finite finger pads."""
        shifted = BagRobotScene(
            "m3_franka_tote_empty",
            device="cuda:0",
            use_cuda_graph=False,
            closed_opening=0.003,
            grip_z_offset=-0.005,
        )
        baseline = BagRobotScene("m3_franka_tote_empty", device="cuda:0", use_cuda_graph=False, closed_opening=0.003)
        self._assert_same_soft_material(baseline, shifted)
        self.assertEqual(baseline.metrics()["physics_parameters"]["grip_z_offset_m"], 0.0)
        self.assertEqual(shifted.metrics()["physics_parameters"]["grip_z_offset_m"], -0.005)
        np.testing.assert_allclose(
            np.asarray(shifted.grip_centers) - baseline.grip_centers, [[0, 0, -0.005]] * 2, rtol=0, atol=1e-15
        )
        self.assertGreater(min(shifted.initial_hand_to_hand), 0.0)
        for original, changed in zip(baseline.drivers, shifted.drivers, strict=True):
            for name in ("times", "rotations", "openings"):
                np.testing.assert_array_equal(getattr(original, name).numpy(), getattr(changed, name).numpy())
            old_positions, new_positions = original.positions.numpy(), changed.positions.numpy()
            # Stored command coordinates are float32; permit one rounding unit.
            np.testing.assert_allclose(
                new_positions - old_positions, [[0, 0, -0.005]] * len(old_positions), rtol=0, atol=6e-8
            )
            np.testing.assert_allclose(
                new_positions - new_positions[0], old_positions - old_positions[0], rtol=0, atol=6e-8
            )
            np.testing.assert_array_equal(changed.openings.numpy()[7:], np.full(3, 0.08, dtype=np.float32))
        for scene, expected_count in ((baseline, 0), (shifted, 1)):
            for center, audit in zip(scene.grip_centers, scene.initial_geometry, strict=True):
                self.assertEqual(audit["vertices_inside_robot_solids"], 0)
                self.assertGreater(audit["robot_clearance_above_table_plane_m"], 0)
                self.assertAlmostEqual(audit["physical_tip_aperture_m"], 0.00296, delta=1e-8)
                # Use the actual FK-derived common face bounds, not nominal pad sizes.
                local = (scene.initial_q - center) * [1, -1, -1]
                xmin, xmax = audit["tip_contact_span_x_in_tcp_m"]
                zmin, zmax = audit["tip_contact_span_z_in_tcp_m"]
                inset = np.minimum.reduce(
                    (local[:, 0] - xmin, xmax - local[:, 0], local[:, 2] - zmin, zmax - local[:, 2])
                )
                between = np.abs(local[:, 1]) < audit["physical_tip_aperture_m"] / 2
                inside_face = between & (inset > 0)
                self.assertEqual(int(inside_face.sum()), expected_count)
                if expected_count:
                    self.assertGreater(float(inset[inside_face].min()), 0.0019)

    def test_center_pinch_on_original_handle_vertices(self):
        """Center both finite pads on handle vertices without changing material or relative motion."""
        centered = BagRobotScene(
            "m3_franka_tote_empty",
            device="cuda:0",
            use_cuda_graph=False,
            closed_opening=0.005,
            grip_x=0.054144661873579025,
            grip_z_offset=0.024055231662088783,
        )
        baseline = BagRobotScene("m3_franka_tote_empty", device="cuda:0", use_cuda_graph=False, closed_opening=0.005)
        self._assert_same_soft_material(baseline, centered)
        self.assertEqual(baseline.metrics()["physics_parameters"]["grip_x_m"], 0.04)
        self.assertEqual(baseline.metrics()["physics_parameters"]["grip_z_offset_m"], 0.0)
        self.assertEqual(centered.metrics()["physics_parameters"]["grip_x_m"], 0.054144661873579025)
        self.assertGreater(min(centered.initial_hand_to_hand), 0.0)
        for index, (original, changed, center, audit) in enumerate(
            zip(baseline.drivers, centered.drivers, centered.grip_centers, centered.initial_geometry, strict=True)
        ):
            self.assertEqual(audit["vertices_inside_robot_solids"], 0)
            self.assertGreater(audit["robot_clearance_above_table_plane_m"], 0)
            self.assertAlmostEqual(audit["physical_tip_aperture_m"], 0.00496, delta=1e-8)
            # Original vertices 459/513 lie on the two opposing handle shoulders.
            local = (centered.initial_q[(459, 513)[index]] - center) * [1, -1, -1]
            xmin, xmax = audit["tip_contact_span_x_in_tcp_m"]
            zmin, zmax = audit["tip_contact_span_z_in_tcp_m"]
            np.testing.assert_allclose(local[[0, 2]], [(xmin + xmax) / 2, (zmin + zmax) / 2], rtol=0, atol=8e-8)
            self.assertLess(abs(local[1]), audit["physical_tip_aperture_m"] / 2)
            self.assertGreater(min(local[0] - xmin, xmax - local[0]), 0.00874)
            self.assertGreater(min(local[2] - zmin, zmax - local[2]), 0.00924)
            for name in ("times", "rotations", "openings"):
                np.testing.assert_array_equal(getattr(original, name).numpy(), getattr(changed, name).numpy())
            old_positions, new_positions = original.positions.numpy(), changed.positions.numpy()
            offset = np.asarray(center) - baseline.grip_centers[index]
            np.testing.assert_allclose(
                new_positions - old_positions, np.tile(offset, (len(old_positions), 1)), rtol=0, atol=6e-8
            )
            np.testing.assert_allclose(
                new_positions - new_positions[0], old_positions - old_positions[0], rtol=0, atol=6e-8
            )
            np.testing.assert_array_equal(changed.openings.numpy()[7:], np.full(3, 0.08, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
