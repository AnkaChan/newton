# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for conservative sampled task evidence."""

import unittest
from types import SimpleNamespace

import numpy as np

from . import task_observer as module


def fixture():
    model = SimpleNamespace(
        shape_body=np.array([0]),
        shape_type=np.array([7]),
        shape_scale=np.array([[1.0, 1.0, 1.0]]),
        shape_transform=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        shape_margin=np.array([0.1]),
        particle_radius=np.array([0.2]),
        particle_mass=np.array([1.0]),
    )
    state = SimpleNamespace(
        particle_q=np.array([[2.0, 0.0, 2.0]]), body_q=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    )
    scene = SimpleNamespace(
        model=model,
        frame_dt=1 / 30,
        state=state,
        phase="hold",
        protocol={
            "family": "Robot manipulation",
            "hold_start": 0.0,
            "hold_end": 1.0,
            "penetration_surfaces": [{"kind": "plane", "normal": [0, 0, 1], "offset_m": 0}],
        },
        hand=SimpleNamespace(shape_ids=[0], label="test_hand"),
    )
    return scene


class TaskEvidenceTest(unittest.TestCase):
    def test_disabled_family_result(self):
        """Verify disabled family result."""
        scene = fixture()
        scene.protocol["family"] = "Soft contact"
        obs = module.TaskObserver(scene)
        result = obs.result()
        self.assertFalse(result["enabled"])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["support_planes"], [])
        self.assertIsNone(result["first_proven_hold_separation_s"])

    def test_closed_box_distance_and_overlap(self):
        """Verify closed box distance and overlap."""
        a = (np.zeros(3), np.ones(3))
        self.assertEqual(module.separation(a, a), 0)
        self.assertEqual(module.separation(a, (np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))), 0)
        self.assertAlmostEqual(module.separation(a, (np.array([4.0, 5.0, 1.0]), np.array([6.0, 6.0, 2.0]))), 5)

    def test_physical_shell_margin_and_full_transform(self):
        """Verify physical shell margin and full transform."""
        scene = fixture()
        obs = module.TaskObserver(scene)
        row = obs.rows[0]
        self.assertAlmostEqual(row["hand_separation_lower_bound_m"][0], np.sqrt(2) * 0.7)
        self.assertAlmostEqual(row["components"][0]["minimum_shell_z_m"], 1.8)
        scene.state.body_q[0, :3] = [1.0, 0.0, 1.0]
        obs.sample(scene.state, 1 / 30, "hold")
        self.assertFalse(obs.rows[-1]["fully_separate_from_all_hands"])

    def test_display_cadence_and_finite_suffix(self):
        """Verify display cadence and finite suffix."""
        scene = fixture()
        obs = module.TaskObserver(scene)
        for n in range(1, 21):
            obs.sample(scene.state, n / 300, "hold")
        self.assertEqual(len(obs.rows), 3)
        self.assertEqual(obs.result()["hold_fully_separated_sample_count"], 3)
        scene.state.particle_q[0, 0] = np.nan
        obs.sample(scene.state, 0.1, "hold")
        self.assertIsNone(obs.result()["final_fully_separated"])
        self.assertEqual(obs.result()["final_separation_suffix_observed_duration_s"], 0)

    def test_disjoint_components_union_is_only_a_lower_bound(self):
        """Verify disjoint components union is only a lower bound."""
        a = (np.array([-3.0, -1.0, -1.0]), np.array([3.0, 1.0, 1.0]))
        b = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(module.separation(a, b), 0)

    def test_original_mesh_scale_and_local_pose(self):
        """Verify original mesh scale and local pose."""
        scene = fixture()
        scene.model.shape_type[0] = 8
        scene.model.shape_source = [SimpleNamespace(vertices=np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]))]
        scene.model.shape_scale[0] = [-2.0, 3.0, 4.0]
        scene.model.shape_transform[0, :3] = [1.0, 2.0, 3.0]
        obs = module.TaskObserver(scene)
        lo, hi = obs.rows[0]["hand_bounds_m"][0]
        np.testing.assert_allclose(lo, [-1.1, -4.1, -9.1])
        np.testing.assert_allclose(hi, [3.1, 8.1, 15.1])

    def test_capsule_hemisphere_in_axial_extent(self):
        """Verify capsule hemisphere in axial extent."""
        scene = fixture()
        scene.model.shape_type[0] = 4
        scene.model.shape_scale[0] = [0.3, 0.7, 0]
        obs = module.TaskObserver(scene)
        lo, hi = obs.rows[0]["hand_bounds_m"][0]
        np.testing.assert_allclose(lo, [-0.4, -0.4, -1.1])
        np.testing.assert_allclose(hi, [0.4, 0.4, 1.1])

    def test_recipe_spec_field_layouts(self):
        """Verify recipe spec field layouts."""
        for name in ("hands", "specs"):
            scene = fixture()
            setattr(scene, name, [scene.hand])
            del scene.hand
            self.assertEqual(len(module.TaskObserver(scene).hands), 1)
        scene = fixture()
        del scene.hand
        obs = module.TaskObserver(scene)
        self.assertIsNone(obs.rows[0]["fully_separate_from_all_hands"])
        self.assertTrue(obs.warnings)

    def test_bag_payload_ranges_preserve_each_component(self):
        """Verify bag payload ranges preserve each component."""
        scene = fixture()
        scene.model.particle_radius = np.array([0.2, 0.1])
        scene.model.particle_mass = np.array([1.0, 2.0])
        scene.state.particle_q = np.array([[2.0, 0.0, 2.0], [3.0, 1.0, 3.0]])
        scene.bag = SimpleNamespace(particle_start=0, particle_end=1)
        scene.payloads = [SimpleNamespace(particle_start=1, particle_end=2)]
        obs = module.TaskObserver(scene)
        self.assertEqual([c["label"] for c in obs.rows[0]["components"]], ["bag", "payload_0"])
        np.testing.assert_allclose(obs.rows[0]["whole_object_bounds_m"][1], [3.1, 1.1, 3.1])

    def test_cable_bounds_and_duck_label_fallback(self):
        """Verify cable bounds and duck label fallback."""
        scene = fixture()
        scene.protocol["cable_body_indices"] = [0]
        scene.model.particle_radius = np.empty(0)
        scene.model.particle_mass = np.empty(0)
        scene.state.particle_q = np.empty((0, 3))
        obs = module.TaskObserver(scene)
        self.assertEqual(obs.rows[0]["components"][0]["label"], "whole_cable_arrangement")
        scene = fixture()
        del scene.hand
        scene.protocol["case_id"] = "r2_franka_duck"
        scene.model.body_label = ["franka/fr3_hand"]
        self.assertEqual(module.TaskObserver(scene).hands[0]["shape_ids"], [0])


if __name__ == "__main__":
    unittest.main()
