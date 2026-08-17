# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the common-objective solver benchmark."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import tempfile
import unittest

import numpy as np
import warp as wp
from newton.solvers import SolverVBD

from .. import solver_benchmark as benchmark


class TestSolverBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.scene = benchmark.build_structured_cantilever_scene()
        cls.problem = benchmark.build_common_problem(cls.scene)
        cls.newton_run = benchmark.run_newton(cls.scene, cls.problem, warmup=True, repeats=2)
        cls.newton_result = cls.newton_run.result
        cls.vbd_1 = benchmark.run_vbd(cls.scene, 1, device="cpu", warmup=True, repeats=2)
        cls.vbd_2 = benchmark.run_vbd(cls.scene, 2, device="cpu", warmup=True, repeats=1)
        cls.vbd_32 = benchmark.run_vbd(cls.scene, 32, device="cpu", warmup=True, repeats=2)

    def test_structured_scene_has_auditable_contract(self):
        scene = self.scene
        self.assertEqual(scene.n_vertices, 8)
        self.assertEqual(scene.n_tets, 5)
        self.assertEqual(scene.n_triangles, 12)
        np.testing.assert_array_equal(scene.pinned_indices, [0, 2, 4, 6])
        np.testing.assert_allclose(scene.external_force.sum(axis=0), [4.0, -3.0, -6.0])
        self.assertTrue(np.all(scene.tet_materials[:, 2] == 0.0))
        self.assertTrue(np.all(scene.tet_materials[:, 0] == 20.0))
        self.assertTrue(np.all(scene.tet_materials[:, 1] == 40.0))
        self.assertTrue(np.all(scene.tri_materials == 0.0))
        self.assertEqual(scene.metadata["lambda_linearized_pa"], 20.0)
        self.assertIn("consumed-directly", scene.metadata["coefficient_convention"])
        self.assertEqual(scene.color_group_offsets[0], 0)
        self.assertEqual(scene.color_group_offsets[-1], scene.n_vertices)
        np.testing.assert_array_equal(np.sort(scene.color_group_particles), np.arange(scene.n_vertices))
        self.assertFalse(scene.rest_q.flags.writeable)
        self.assertEqual(scene.manifest(), scene.manifest())
        caller_owned = scene.rest_q.copy()
        copied_scene = dataclasses.replace(scene, rest_q=caller_owned)
        caller_owned[0, 0] += 7.0
        self.assertNotEqual(caller_owned[0, 0], copied_scene.rest_q[0, 0])

    def test_scene_hash_changes_with_physical_inputs_and_ordering(self):
        original = self.scene.manifest()["scene_sha256"]
        mutations = {}
        for field in (
            "rest_q",
            "tet_poses",
            "tet_materials",
            "tri_poses",
            "tri_areas",
            "particle_flags",
            "x_current",
            "velocity",
            "gravity",
            "external_force",
            "pin_targets",
        ):
            value = getattr(self.scene, field).copy()
            if field == "particle_flags":
                value[self.scene.free_indices[0]] |= 2
            elif field == "tet_materials":
                value[0, 0] += 1.0
            elif field in ("tet_poses", "tri_poses"):
                value[0, 0, 0] += 0.01
            elif field == "pin_targets":
                value[0, 0] += 0.01
            else:
                value.reshape(-1)[0] += 0.01
            mutations[field] = value

        reordered_tets = self.scene.tet_indices.copy()
        reordered_tets[[0, 1]] = reordered_tets[[1, 0]]
        mutations["tet_indices"] = reordered_tets
        reordered_triangles = self.scene.tri_indices.copy()
        reordered_triangles[[0, 1]] = reordered_triangles[[1, 0]]
        mutations["tri_indices"] = reordered_triangles
        reordered_colors = self.scene.color_group_particles.copy()
        first_group_end = self.scene.color_group_offsets[1]
        if first_group_end >= 2:
            reordered_colors[:2] = reordered_colors[1::-1]
        else:
            reordered_colors[[0, first_group_end]] = reordered_colors[[first_group_end, 0]]
        mutations["color_group_particles"] = reordered_colors

        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = dataclasses.replace(self.scene, **{field: value})
                self.assertNotEqual(changed.manifest()["scene_sha256"], original)

        changed_mass = self.scene.mass.copy()
        changed_mass[self.scene.free_indices[0]] += np.float32(0.25)
        changed_inverse_mass = self.scene.particle_inv_mass.copy()
        changed_inverse_mass[self.scene.free_indices[0]] = np.float32(
            1.0 / np.float32(changed_mass[self.scene.free_indices[0]])
        )
        changed = dataclasses.replace(
            self.scene,
            mass=changed_mass,
            particle_inv_mass=changed_inverse_mass,
        )
        self.assertNotEqual(changed.manifest()["scene_sha256"], original)

        changed_dt = dataclasses.replace(self.scene, dt=self.scene.dt * 0.5)
        changed_metadata = dataclasses.replace(
            self.scene,
            metadata=dict(self.scene.metadata) | {"coefficient_convention": "deliberately-wrong"},
        )
        self.assertNotEqual(changed_dt.manifest()["scene_sha256"], original)
        self.assertNotEqual(changed_metadata.manifest()["scene_sha256"], original)

    def test_array_hash_is_layout_and_host_endian_independent(self):
        values = np.arange(12, dtype=np.float64).reshape(3, 4)
        fortran = np.asfortranarray(values)
        big_endian = values.astype(values.dtype.newbyteorder(">"))
        self.assertEqual(benchmark._array_digest(values), benchmark._array_digest(fortran))
        self.assertEqual(benchmark._array_digest(values), benchmark._array_digest(big_endian))

    def test_common_problem_rejects_damping(self):
        materials = self.scene.tet_materials.copy()
        materials[:, 2] = 1.0e-3
        damped = dataclasses.replace(self.scene, tet_materials=materials)
        with self.assertRaisesRegex(ValueError, "zero tet damping"):
            benchmark.build_common_problem(damped)

    def test_runtime_inputs_are_canonicalized_through_vbd_float32(self):
        scene = benchmark.build_structured_cantilever_scene(
            dimensions=(1, 2, 1),
            dt=1.0 / 60.0,
            gravity=(0.1, -0.2, -9.81),
            total_tip_force=(1.0, 2.0, -3.0),
            initial_velocity=(0.1, 0.2, 0.3),
        )
        self.assertEqual(scene.dt, float(np.float32(1.0 / 60.0)))
        for name in ("x_current", "velocity", "gravity", "external_force", "pin_targets"):
            value = getattr(scene, name)
            np.testing.assert_array_equal(value, value.astype(np.float32).astype(np.float64))
        problem = benchmark.build_common_problem(scene)
        np.testing.assert_array_equal(problem.inertial_target.numpy(), scene.vbd_inertial_target)
        benchmark._build_vbd_model(scene, "cpu")
        materials = scene.tet_materials.copy()
        materials[:] = 0.0
        disabled = dataclasses.replace(
            scene,
            tet_materials=materials,
            metadata=dict(scene.metadata) | {"fixture": "nonbinary-disabled-elastic-predictor"},
        )
        cpu = benchmark.run_vbd(disabled, 1, device="cpu", warmup=False, repeats=1)
        np.testing.assert_array_equal(cpu.positions, disabled.vbd_inertial_target)

    def test_common_evaluator_rejects_pin_violation(self):
        metrics = benchmark.evaluate_common_state(
            self.problem,
            self.newton_result.x,
            reference_positions=self.newton_result.x,
        )
        self.assertEqual(metrics.free_rms_error_m, 0.0)
        self.assertEqual(metrics.mass_weighted_rms_error_m, 0.0)
        self.assertEqual(metrics.max_pin_error_m, 0.0)

        invalid = self.newton_result.x.detach().numpy().copy()
        invalid[self.scene.pinned_indices[0], 0] += 1.0e-3
        with self.assertRaisesRegex(ValueError, "Dirichlet"):
            benchmark.evaluate_common_state(self.problem, invalid)

    def test_vbd_restarts_are_deterministic_and_preserve_pins(self):
        np.testing.assert_allclose(
            self.vbd_1.positions[self.scene.pinned_indices],
            self.scene.pin_targets,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(self.vbd_1.effective_tile_solve)
        self.assertEqual(self.vbd_1.color_group_count, self.scene.color_group_offsets.size - 1)
        self.assertEqual(len(self.vbd_1.repeat_seconds), 2)
        self.assertGreater(self.vbd_1.median_solve_seconds, 0.0)
        free_displacement = self.vbd_1.positions[self.scene.free_indices] - self.scene.x_current[self.scene.free_indices]
        self.assertGreater(np.linalg.norm(free_displacement), 0.0)

    def test_vbd_adapter_matches_free_predictor_and_both_pin_mechanisms(self):
        materials = self.scene.tet_materials.copy()
        materials[:] = 0.0
        flags = self.scene.particle_flags.copy()
        positive_mass_inactive = int(self.scene.free_indices[0])
        flags[positive_mass_inactive] &= ~int(benchmark.newton.ParticleFlags.ACTIVE)
        pinned = np.sort(np.append(self.scene.pinned_indices, positive_mass_inactive)).astype(np.int64)
        disabled = dataclasses.replace(
            self.scene,
            tet_materials=materials,
            particle_flags=flags,
            pinned_indices=pinned,
            pin_targets=self.scene.x_current[pinned],
            metadata=dict(self.scene.metadata) | {"fixture": "disabled-elastic-predictor"},
        )

        model = benchmark._build_vbd_model(disabled, "cpu")
        state_in, state_out, control = benchmark._make_vbd_state(model, disabled)
        q_before = state_in.particle_q.numpy().copy()
        qd_before = state_in.particle_qd.numpy().copy()
        force_before = state_in.particle_f.numpy().copy()
        solver = SolverVBD(
            model,
            iterations=1,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )
        solver.step(state_in, state_out, control, None, disabled.dt)

        expected = disabled.vbd_inertial_target.astype(np.float32)
        np.testing.assert_array_equal(state_out.particle_q.numpy(), expected)
        np.testing.assert_allclose(
            state_out.particle_qd.numpy(),
            (expected - q_before) / disabled.dt,
            rtol=0.0,
            atol=2.0e-6,
        )
        np.testing.assert_array_equal(state_in.particle_q.numpy(), expected)
        np.testing.assert_array_equal(state_in.particle_qd.numpy(), qd_before)
        np.testing.assert_array_equal(state_in.particle_f.numpy(), force_before)
        np.testing.assert_array_equal(state_out.particle_f.numpy(), np.zeros_like(force_before))
        self.assertGreater(disabled.mass[positive_mass_inactive], 0.0)
        self.assertEqual(np.linalg.norm(state_out.particle_qd.numpy()[pinned], axis=1).max(), 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_tile_vbd_preserves_boundary_model_and_matches_scalar(self):
        nonbinary = benchmark.build_structured_cantilever_scene(
            dimensions=(1, 2, 1),
            dt=1.0 / 60.0,
            gravity=(0.1, -0.2, -9.81),
            total_tip_force=(1.0, 2.0, -3.0),
            initial_velocity=(0.1, 0.2, 0.3),
        )
        materials = nonbinary.tet_materials.copy()
        materials[:] = 0.0
        disabled = dataclasses.replace(
            nonbinary,
            tet_materials=materials,
            metadata=dict(nonbinary.metadata) | {"fixture": "nonbinary-disabled-elastic-predictor"},
        )
        predictor = benchmark.run_vbd(
            disabled,
            1,
            device="cuda:0",
            tile_solve=True,
            warmup=False,
            repeats=1,
        )
        np.testing.assert_array_equal(predictor.positions, disabled.vbd_inertial_target)

        scalar = benchmark.run_vbd(
            self.scene,
            2,
            device="cuda:0",
            tile_solve=False,
            warmup=True,
            repeats=1,
        )
        tiled = benchmark.run_vbd(
            self.scene,
            2,
            device="cuda:0",
            tile_solve=True,
            warmup=True,
            repeats=1,
        )
        self.assertFalse(scalar.effective_tile_solve)
        self.assertTrue(tiled.effective_tile_solve)
        np.testing.assert_allclose(tiled.positions, scalar.positions, rtol=2.0e-6, atol=2.0e-7)

    def test_high_budget_vbd_and_newton_reach_same_stationary_point(self):
        self.assertTrue(self.newton_run.reference_accepted, self.newton_run.reference_failures)
        self.assertTrue(self.newton_result.converged, self.newton_result.reason)
        self.assertLess(self.newton_result.final_relative_residual, 1.0e-8)
        metrics = benchmark.evaluate_common_state(
            self.problem,
            self.vbd_32.positions,
            reference_positions=self.newton_result.x,
        )
        self.assertLess(metrics.relative_residual, 1.0e-4)
        self.assertLess(metrics.free_rms_error_m, 1.0e-6)
        self.assertLess(metrics.mass_weighted_rms_error_m, 1.0e-6)
        self.assertEqual(metrics.inverted_tet_fraction, 0.0)
        self.assertLess(abs(metrics.objective - self.newton_result.final_objective), 1.0e-8)
        self.assertLessEqual(self.newton_result.final_objective, metrics.objective + 1.0e-10)

    def test_two_timesteps_are_not_a_two_iteration_convergence_endpoint(self):
        model = benchmark._build_vbd_model(self.scene, "cpu")
        solver = SolverVBD(
            model,
            iterations=1,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )
        state_0, state_1, control = benchmark._make_vbd_state(model, self.scene)
        solver.step(state_0, state_1, control, None, self.scene.dt)

        state_1.clear_forces()
        state_1.particle_f.assign(
            wp.array(self.scene.external_force.astype(np.float32), dtype=wp.vec3, device=model.device)
        )
        state_2 = model.state()
        solver.step(state_1, state_2, control, None, self.scene.dt)
        sequential_positions = state_2.particle_q.numpy().astype(np.float64)
        difference = np.linalg.norm(sequential_positions - self.vbd_2.positions)
        self.assertGreater(difference, 1.0e-4)

    def test_bundle_round_trips_raw_states_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "result.json"
            benchmark.write_benchmark_bundle(
                output,
                self.scene,
                self.problem,
                self.newton_run,
                [self.vbd_1, self.vbd_32],
            )
            payload = json.loads(output.read_text())
            raw_path = output.with_suffix(".npz")
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["scene"]["scene_sha256"], self.scene.manifest()["scene_sha256"])
            self.assertEqual(
                payload["objective"]["objective_instance_sha256"],
                benchmark.common_objective_manifest(self.scene, self.problem)["objective_instance_sha256"],
            )
            self.assertEqual(payload["raw_npz"]["sha256"], hashlib.sha256(raw_path.read_bytes()).hexdigest())
            with np.load(raw_path) as raw:
                np.testing.assert_array_equal(raw["newton_positions"], self.newton_result.x.detach().numpy())
                np.testing.assert_array_equal(raw["vbd_positions_k32"], self.vbd_32.positions)

            recomputed = benchmark.evaluate_common_state(
                self.problem,
                self.vbd_32.positions,
                reference_positions=self.newton_result.x,
            )
            serialized = next(item for item in payload["vbd"] if item["iterations"] == 32)["metrics"]
            self.assertAlmostEqual(serialized["objective"], recomputed.objective, places=14)
            self.assertAlmostEqual(serialized["relative_residual"], recomputed.relative_residual, places=14)

    def test_bundle_rejects_cross_scene_problem_and_result(self):
        other_scene = benchmark.build_structured_cantilever_scene(total_tip_force=(5.0, -3.0, -6.0))
        other_problem = benchmark.build_common_problem(other_scene)
        other_vbd = benchmark.run_vbd(other_scene, 1, device="cpu", warmup=False, repeats=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "result.json"
            with self.assertRaisesRegex(ValueError, "problem does not match"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    other_problem,
                    self.newton_run,
                    [self.vbd_1],
                )
            with self.assertRaisesRegex(ValueError, "VBD result does not belong"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    self.problem,
                    self.newton_run,
                    [other_vbd],
                )

            changed_positions = self.vbd_1.positions.copy()
            changed_positions[self.scene.free_indices[0], 0] += 1.0e-4
            tampered = dataclasses.replace(self.vbd_1, positions=changed_positions)
            with self.assertRaisesRegex(ValueError, "modified after the bound run"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    self.problem,
                    self.newton_run,
                    [tampered],
                )

            relabeled = dataclasses.replace(self.vbd_1, iterations=999)
            with self.assertRaisesRegex(ValueError, "execution/configuration record"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    self.problem,
                    self.newton_run,
                    [relabeled],
                )

            fake_velocities = self.vbd_1.velocities.copy()
            fake_velocities[self.scene.free_indices[0], 0] += 1.0
            tampered_velocity = dataclasses.replace(self.vbd_1, velocities=fake_velocities)
            with self.assertRaisesRegex(ValueError, "execution/configuration record"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    self.problem,
                    self.newton_run,
                    [tampered_velocity],
                )

            fake_config = dataclasses.replace(
                self.newton_run,
                config=dataclasses.replace(self.newton_run.config, max_iterations=999),
            )
            with self.assertRaisesRegex(ValueError, "execution/configuration record"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    self.problem,
                    fake_config,
                    [self.vbd_1],
                )

            fake_timing = dataclasses.replace(self.vbd_1, repeat_seconds=(999.0,))
            with self.assertRaisesRegex(ValueError, "execution/configuration record"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    self.problem,
                    self.newton_run,
                    [fake_timing],
                )

            fake_iterate = dataclasses.replace(self.vbd_1, iterate_zero_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "iterate zero"):
                benchmark.write_benchmark_bundle(
                    output,
                    self.scene,
                    self.problem,
                    self.newton_run,
                    [fake_iterate],
                )


if __name__ == "__main__":
    unittest.main()
