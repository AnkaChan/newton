# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check native DAT configuration and prescribed endpoint preservation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

import newton
from experiments.collision_benchmark import comparison_runtime as runtime
from experiments.collision_benchmark import main_dat_runtime, pouch_scene


class MainDATTests(unittest.TestCase):
    def tearDown(self):
        """Restore the process-wide benchmark and pipeline adapters."""
        if hasattr(runtime, "disable_main_dat"):
            runtime.disable_main_dat()
        runtime.MODE = False
        runtime.ACTIVE = None
        newton.solvers.SolverVBD = runtime.OriginalSolverVBD

    def enable(self):
        """Require the main-only adapter before testing its behavior."""
        self.assertTrue(hasattr(runtime, "configure_main_dat"), "Native DAT adapter is absent")
        runtime.configure_main_dat("m2_franka_pouch")

    def test_native_configuration_is_standard_only(self):
        """Select native DAT and reject unavailable ALM elasticity."""
        self.enable()
        options = runtime.paired_options(False)
        self.assertTrue(options["rigid_soft_enable_dat"])
        self.assertFalse(options["integrate_with_external_rigid_solver"])
        self.assertFalse(options["enable_fast_math"])
        self.assertNotIn("particle_elasticity_alm", options)
        self.assertNotIn("particle_self_contact_method", options)
        with self.assertRaisesRegex(ValueError, "standard"):
            runtime.paired_options(True)
        self.assertEqual(runtime.main_dat_configuration()["pipeline"]["soft_contact_gap"], 0.003)

    def test_endpoint_copy_preserves_dynamic_bodies(self):
        """Copy commanded kinematic endpoints without changing dynamic rows."""
        self.enable()

        model = SimpleNamespace(
            body_count=2,
            device=wp.get_device("cpu"),
            body_inv_mass=wp.array([1.0, 1.0], dtype=float, device="cpu"),
            body_flags=wp.array([int(newton.BodyFlags.KINEMATIC), 0], dtype=wp.int32, device="cpu"),
        )
        initial = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        endpoint = initial.copy()
        endpoint[:, 0] += 0.1
        state0 = SimpleNamespace(
            body_q=wp.array(initial, dtype=wp.transform, device="cpu"),
            body_qd=wp.zeros(2, dtype=wp.spatial_vector, device="cpu"),
        )
        state1 = SimpleNamespace(
            body_q=wp.array(endpoint, dtype=wp.transform, device="cpu"),
            body_qd=wp.ones(2, dtype=wp.spatial_vector, device="cpu"),
        )
        main_dat_runtime.prepare_kinematic_endpoints(model, state0, state1)
        np.testing.assert_array_equal(state0.body_q.numpy()[0], endpoint[0])
        np.testing.assert_array_equal(state0.body_q.numpy()[1], initial[1])
        np.testing.assert_array_equal(state0.body_qd.numpy()[0], np.ones(6))
        np.testing.assert_array_equal(state0.body_qd.numpy()[1], np.zeros(6))

    def test_collision_latch_retains_earlier_overflow(self):
        """Retain an overflow even when a later detection clears its counters."""
        self.enable()

        self.assertTrue(hasattr(main_dat_runtime, "record_contact_counts"))
        counts = wp.array([17, 29, 1, 0], dtype=wp.int32, device="cpu")
        latch = wp.zeros(5, dtype=wp.int32, device="cpu")
        main_dat_runtime.record_contact_counts(counts, latch)
        counts.assign(np.array([3, 5, 0, 1], dtype=np.int32))
        main_dat_runtime.record_contact_counts(counts, latch)
        counts.zero_()
        main_dat_runtime.record_contact_counts(counts, latch)
        np.testing.assert_array_equal(latch.numpy(), [17, 29, 1, 1, 3])

    def test_self_contact_latch_retains_row_capacity_excess(self):
        """Keep all row-family overflows after later clean self detections."""
        self.enable()
        self.assertTrue(hasattr(main_dat_runtime, "record_self_contact_counts"))
        info = SimpleNamespace()
        families = ("vertex_colliding_triangles", "edge_colliding_edges", "triangle_colliding_vertices")
        for family, counts, capacities in zip(families, ([2, 6], [7, 1], [9]), ([3, 4], [4, 2], [5]), strict=True):
            setattr(info, family + "_count", wp.array(counts, dtype=wp.int32, device="cpu"))
            setattr(info, family + "_buffer_sizes", wp.array(capacities, dtype=wp.int32, device="cpu"))
        latch = wp.zeros(7, dtype=wp.int32, device="cpu")
        main_dat_runtime.record_self_contact_counts(info, latch)
        for family in families:
            getattr(info, family + "_count").zero_()
        main_dat_runtime.record_self_contact_counts(info, latch)
        np.testing.assert_array_equal(latch.numpy(), [6, 7, 9, 2, 3, 4, 2])

    def test_historical_pouch_audit_keeps_independent_query(self):
        """Preserve the authored margin-only query for historical solvers."""

        self.assertTrue(hasattr(pouch_scene, "_initial_self_contact_audit"))
        detector = Mock()
        detector.vertex_colliding_triangles_count.numpy.return_value = np.array([1, 2])
        detector.edge_colliding_edges_count.numpy.return_value = np.array([4])
        solver = SimpleNamespace(trimesh_collision_detector=detector)
        state = SimpleNamespace(particle_q=object())
        result = pouch_scene._initial_self_contact_audit(solver, state, 0.002)
        detector.rebuild.assert_called_once_with(state.particle_q)
        detector.vertex_triangle_collision_detection.assert_called_once_with(0.002)
        detector.edge_edge_collision_detection.assert_called_once_with(0.002)
        self.assertEqual(result["vertex_triangle_pairs_inside_interaction_distance"], 3)
        self.assertEqual(result["edge_edge_pairs_inside_interaction_distance"], 4)

    def test_native_step_keeps_endpoint_and_history(self):
        """Preserve prescribed poses and measure their later discrete velocity."""
        self.enable()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, is_kinematic=True)
        builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.add_cloth_mesh(
            pos=(0, 0, 1),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=[(-0.01, 0, 0), (0.01, 0, 0), (0, 0.02, 0)],
            indices=[0, 1, 2],
            density=0.2,
            tri_ke=1800.0,
            tri_ka=1800.0,
            tri_kd=0.025,
            particle_radius=0.001,
        )
        builder.color()
        model = builder.finalize(device="cpu")
        solver = runtime.ObservedSolverVBD(
            model,
            iterations=2,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_method="penalty_dat",
            particle_elasticity_alm=False,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.005,
            particle_self_contact_gap=0.001,
        )
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.003, enable_rigid_soft_full_surface_contact=True)
        contacts = pipeline.contacts()
        self.assertIs(contacts, solver.contacts)
        self.assertIsNot(pipeline, solver.collision_pipeline)
        state0, state1 = model.state(), model.state()

        self.assertTrue(hasattr(pouch_scene, "_initial_self_contact_audit"))
        initial = state0.particle_q.numpy().copy()
        audit = pouch_scene._initial_self_contact_audit(solver, state0, 0.005)
        self.assertAlmostEqual(audit["query_radius_m"], 0.006)
        self.assertEqual(audit["detection_only_gap_m"], 0.001)
        self.assertNotIn("vertex_triangle_pairs_inside_interaction_distance", audit)
        np.testing.assert_array_equal(state0.particle_q.numpy(), initial)
        for endpoint in (0.001, 0.003):
            q = state1.body_q.numpy()
            q[body, 0] = endpoint
            state1.body_q.assign(q)
            with patch.object(solver.collision_pipeline, "collide", wraps=solver.collision_pipeline.collide) as collide:
                pipeline.collide(state0, contacts)
                self.assertEqual(collide.call_count, 0)
                solver.step(state0, state1, model.control(), contacts, 0.01)
                self.assertGreaterEqual(collide.call_count, 2)
            self.assertAlmostEqual(float(state1.body_q.numpy()[body, 0]), endpoint, places=7)
        self.assertAlmostEqual(float(state1.body_qd.numpy()[body, 0]), 0.2, places=5)
        self.assertEqual(int(solver.collision_pipeline.dat_self_contact_latch.numpy()[6]), 5)
        self.assertFalse(np.any(solver.collision_pipeline.dat_self_contact_latch.numpy()[3:6]))
        with self.assertRaisesRegex(ValueError, "pipeline"):
            newton.CollisionPipeline(model, soft_contact_gap=0.009, enable_rigid_soft_full_surface_contact=True)


if __name__ == "__main__":
    unittest.main()
