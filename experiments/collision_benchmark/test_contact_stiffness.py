# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check contact caps without stepping a simulation."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from experiments.collision_benchmark.contact_stiffness import apply_contact_policy, describe_policy


def cloth_model():
    return SimpleNamespace(
        soft_contact_ke=50000.0,
        soft_contact_kd=0.1,
        soft_contact_mu=1.0,
        shape_material_ke=wp.array([0.0, 1000.0, 50000.0, 100000.0], dtype=wp.float32, device="cpu"),
        tri_materials=wp.array([[1800.0, 1800.0, 0.1, 0.0, 0.0]], dtype=wp.float32, device="cpu"),
        tet_materials=None,
        joint_type=None,
    )


class ContactCapTests(unittest.TestCase):
    def test_cap_both_sides_without_changing_material_or_friction(self):
        """Cap both contact sides while preserving elasticity and friction."""
        model = cloth_model()
        elasticity = model.tri_materials.numpy().copy()
        report = apply_contact_policy(model, {"max_ratio": 10.0, "tet_policy": "numeric"})
        self.assertEqual(model.soft_contact_ke, 18000.0)
        np.testing.assert_array_equal(model.shape_material_ke.numpy(), [0.0, 1000.0, 18000.0, 18000.0])
        np.testing.assert_array_equal(model.tri_materials.numpy(), elasticity)
        self.assertEqual((model.soft_contact_kd, model.soft_contact_mu), (0.1, 1.0))
        self.assertEqual(report["cap_N_per_m"], 18000.0)
        self.assertLessEqual(0.5 * (model.soft_contact_ke + model.shape_material_ke.numpy().max()), 18000.0)

    def test_replacement_preserves_authored_values_and_rejects_policy_switch(self):
        """Keep authored provenance and reject a changed replacement policy."""
        model = cloth_model()
        policy = {"max_ratio": 3.0, "tet_policy": "numeric"}
        first = apply_contact_policy(model, policy)
        second = apply_contact_policy(model, policy)
        self.assertEqual(model.soft_contact_ke, 5400.0)
        self.assertEqual(first, second)
        self.assertEqual(first["authored"]["soft_contact_ke"], 50000.0)
        with self.assertRaises(ValueError):
            apply_contact_policy(model, {"max_ratio": 10.0, "tet_policy": "numeric"})

    def test_authored_is_unchanged(self):
        """Preserve the authored contact coefficients without an explicit cap."""
        model = cloth_model()
        report = apply_contact_policy(model, None)
        self.assertEqual(model.soft_contact_ke, 50000.0)
        np.testing.assert_array_equal(model.shape_material_ke.numpy(), [0.0, 1000.0, 50000.0, 100000.0])
        self.assertEqual(report["policy"]["name"], "authored")

    def test_invalid_caps_are_rejected(self):
        """Reject nonfinite, nonpositive, and over-limit contact ratios."""
        for value in (0, -1, 10.001, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                describe_policy({"max_ratio": value, "tet_policy": "numeric"})

    def test_tet_numeric_policy_reports_unlike_units(self):
        """Ignore inactive membrane terms and disclose tetrahedral units."""
        model = cloth_model()
        # Tet boundary faces have triangle records with no membrane energy.
        model.tri_materials = wp.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=wp.float32, device="cpu")
        model.tet_materials = wp.array([[8000.0, 8000.0, 0.01]], dtype=wp.float32, device="cpu")
        report = apply_contact_policy(model, {"max_ratio": 3.0, "tet_policy": "numeric"})
        self.assertEqual(model.soft_contact_ke, 24000.0)
        self.assertFalse(report["dimensionally_comparable"])

    def test_tet_mesh_reference_has_expected_length_scaling(self):
        """Scale the mesh-aware tetrahedral reference with element length."""
        references = []
        for scale in (1.0, 2.0):
            model = cloth_model()
            model.tri_materials = None
            model.tet_materials = wp.array([[8000.0, 8000.0, 0.01]], dtype=wp.float32, device="cpu")
            model.tet_indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, device="cpu")
            model.particle_q = wp.array(
                scale * np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]), dtype=wp.vec3, device="cpu"
            )
            report = apply_contact_policy(model, {"max_ratio": 10.0, "tet_policy": "mesh"})
            references.append(report["reference_coefficient"])
            self.assertTrue(report["dimensionally_comparable"])
        self.assertAlmostEqual(references[1], 2.0 * references[0])

    def test_rod_reference_excludes_bending_and_robot_actuators(self):
        """Use rod axial terms while excluding bending and robot actuators."""
        model = cloth_model()
        model.tri_materials = None
        model.joint_type = wp.array(
            [int(newton.JointType.REVOLUTE), int(newton.JointType.ROD)], dtype=wp.int32, device="cpu"
        )
        model.joint_qd_start = wp.array([0, 1, 5], dtype=wp.int32, device="cpu")
        model.joint_target_ke = wp.array([1e7, 5e4, 5e4, 0.008, 0.008], dtype=wp.float32, device="cpu")
        before = model.joint_target_ke.numpy().copy()
        report = apply_contact_policy(model, {"max_ratio": 1.0, "tet_policy": "numeric"})
        self.assertEqual(report["reference_coefficient"], 50000.0)
        self.assertEqual(model.shape_material_ke.numpy().max(), 50000.0)
        np.testing.assert_array_equal(model.joint_target_ke.numpy(), before)

    def test_mixed_bag_uses_smaller_reference(self):
        """Choose the smaller elastic reference for a mixed material model."""
        model = cloth_model()
        model.tet_materials = wp.array([[8000.0, 8000.0, 0.01]], dtype=wp.float32, device="cpu")
        report = apply_contact_policy(model, {"max_ratio": 10.0, "tet_policy": "numeric"})
        self.assertEqual(model.soft_contact_ke, 18000.0)
        self.assertEqual(len(report["reference_terms"]), 2)

    def test_replacement_rejects_late_material_change(self):
        """Reject replacement after a contact coefficient changes externally."""
        model = cloth_model()
        policy = {"max_ratio": 10.0, "tet_policy": "numeric"}
        apply_contact_policy(model, policy)
        model.soft_contact_ke = 50000.0
        with self.assertRaises(ValueError):
            apply_contact_policy(model, policy)


if __name__ == "__main__":
    unittest.main()
