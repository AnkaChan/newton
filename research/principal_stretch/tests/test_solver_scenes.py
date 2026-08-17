# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for common-objective adaptations of the PR #2901 scenes."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import warp as wp

import newton

from ..solver_benchmark import build_common_problem, run_vbd
from ..solver_scenes import (
    build_compression_scene,
    build_extension_scene,
    build_refinement_scene,
    build_sliver_scene,
    build_stretch_scene,
    build_twist_scene,
)
from ..torch_solver import compute_F


class TestSolverScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.extension = build_extension_scene()
        cls.stretch = build_stretch_scene()
        cls.twist = build_twist_scene()
        cls.compression_increment = build_compression_scene()
        cls.compression_reduced = build_compression_scene(dim=3, cell=0.1)
        cls.sliver = build_sliver_scene()

    def test_extension_preserves_pr_geometry_and_converts_material(self):
        scene = self.extension
        self.assertEqual((scene.n_vertices, scene.n_tets, scene.n_triangles), (525, 1600, 704))
        self.assertEqual(scene.pinned_indices.size, 25)
        np.testing.assert_array_equal(scene.gravity, [0.0, 0.0, np.float32(-9.81)])
        np.testing.assert_array_equal(scene.tet_materials[:, 0], np.full(scene.n_tets, 5.0e4))
        np.testing.assert_array_equal(scene.tet_materials[:, 1], np.full(scene.n_tets, 1.0e5))
        np.testing.assert_array_equal(scene.tet_materials[:, 2], np.zeros(scene.n_tets))
        self.assertTrue(np.all(scene.tri_materials == 0.0))
        self.assertEqual(scene.metadata["lambda_public_pa"], 5.0e4)
        self.assertEqual(scene.metadata["lambda_stored_pa"], 1.0e5)
        self.assertIn("a513d446", scene.metadata["source_revision"])
        self.assertEqual(scene.dt, float(np.float32(1.0 / 360.0)))

    def test_schedule_states_are_finite_and_orientation_preserving(self):
        scenes = (self.stretch, self.twist, self.compression_increment)
        for scene in scenes:
            with self.subTest(scene=scene.name):
                problem = build_common_problem(scene)
                for state in (scene.x_current, scene.vbd_inertial_target):
                    deformation = compute_F(
                        torch.from_numpy(np.array(state, copy=True)),
                        problem.tets,
                        problem.J,
                    )
                    determinant = torch.linalg.det(deformation)
                    self.assertTrue(torch.isfinite(deformation).all())
                    self.assertGreater(float(determinant.min()), 0.0)

    def test_driven_defaults_are_first_nontrivial_pr_increments(self):
        for scene in (self.stretch, self.twist, self.compression_increment):
            with self.subTest(scene=scene.name):
                self.assertTrue(scene.metadata["default_is_first_nontrivial_pr_increment"])
                self.assertEqual(scene.dt, float(np.float32(1.0 / 300.0)))
                self.assertFalse(
                    np.array_equal(scene.x_current[scene.pinned_indices], scene.pin_targets),
                    "moving boundary target must differ from the preceding state",
                )

    def test_stretch_preserves_zero_mass_left_and_inactive_massive_right_pins(self):
        scene = self.stretch
        rest = scene.rest_q
        left = np.isclose(rest[:, 0], rest[:, 0].min(), rtol=0.0, atol=1.0e-6)
        right = np.isclose(rest[:, 0], rest[:, 0].max(), rtol=0.0, atol=1.0e-6)
        self.assertTrue(np.all(scene.mass[left] == 0.0))
        self.assertTrue(np.all(scene.mass[right] > 0.0))
        self.assertTrue(np.all((scene.particle_flags[right] & int(newton.ParticleFlags.ACTIVE)) == 0))

    def test_reduced_compression_preserves_pr_physical_domain(self):
        extent = self.compression_reduced.rest_q.max(axis=0) - self.compression_reduced.rest_q.min(axis=0)
        np.testing.assert_allclose(extent, [0.3, 0.3, 0.3], rtol=0.0, atol=2.0e-7)
        self.assertEqual(self.compression_reduced.metadata["cell_size_m"], (0.1, 0.1, 0.1))

    def test_compression_release_requires_an_audited_trajectory(self):
        with self.assertRaisesRegex(ValueError, "audited trajectory"):
            build_compression_scene(released=True)

    def test_refinement_levels_share_the_same_physical_domain(self):
        expected = {
            "coarse": (99, 200),
            "medium": (525, 1600),
            "fine": (3321, 12800),
        }
        for level, counts in expected.items():
            with self.subTest(level=level):
                scene = build_refinement_scene(level)
                self.assertEqual((scene.n_vertices, scene.n_tets), counts)
                extent = scene.rest_q.max(axis=0) - scene.rest_q.min(axis=0)
                np.testing.assert_allclose(extent, [0.2, 0.2, 1.0], rtol=0.0, atol=2.0e-7)
                self.assertEqual(scene.metadata["refinement_level"], level)

    def test_sliver_cells_have_ten_to_one_aspect_ratio(self):
        scene = self.sliver
        extent = scene.rest_q.max(axis=0) - scene.rest_q.min(axis=0)
        np.testing.assert_allclose(extent, [0.4, 0.4, 0.2], rtol=0.0, atol=2.0e-7)
        self.assertEqual(scene.metadata["cell_aspect_ratio"], 10.0)
        self.assertEqual(scene.pinned_indices.size, 9)

    def test_vbd_adapter_runs_on_preserved_extension_topology(self):
        result = run_vbd(self.extension, 2, device="cpu", warmup=False, repeats=1)
        self.assertTrue(np.isfinite(result.positions).all())
        np.testing.assert_array_equal(
            result.positions[self.extension.pinned_indices],
            self.extension.pin_targets,
        )

    def test_invalid_scene_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "stretch_ratio"):
            build_stretch_scene(stretch_ratio=0.0)
        with self.assertRaisesRegex(ValueError, "compression_ratio"):
            build_compression_scene(compression_ratio=0.0)
        with self.assertRaisesRegex(ValueError, "audited history"):
            build_stretch_scene(stretch_ratio=2.0)
        with self.assertRaisesRegex(ValueError, "audited history"):
            build_twist_scene(twist_angle=0.5 * np.pi)
        with self.assertRaisesRegex(ValueError, "audited history"):
            build_compression_scene(compression_ratio=0.5)
        with self.assertRaisesRegex(ValueError, "history-bearing"):
            build_twist_scene(twist_angle=2.0 * np.pi)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            build_stretch_scene(dimensions=(3.5, 2, 2))
        with self.assertRaisesRegex(ValueError, "refinement level"):
            build_refinement_scene("unknown")

    def test_one_shot_diagnostics_are_explicitly_labeled(self):
        scene = build_compression_scene(compression_ratio=0.5, one_shot_diagnostic=True)
        self.assertTrue(scene.metadata["one_shot_diagnostic"])
        self.assertIn("not a PR trajectory checkpoint", scene.metadata["state_kind"])


if __name__ == "__main__":
    unittest.main()
