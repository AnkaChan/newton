# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for multiplicative CPU MG-VBD quality evidence."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

from .. import correction_mg_vbd as integration
from ..correction_gpu import MatrixFreeCorrectionConfig, MatrixFreeStableNHOperator, solve_matrix_free_correction
from ..correction_multigrid import assemble_current_stable_nh_block_matrix
from ..solver_benchmark import build_common_problem, evaluate_common_state
from ..solver_scenes import build_stretch_scene


class TestMultiplicativeMGVBD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.scene = build_stretch_scene(dimensions=(2, 1, 1))
        cls.config = integration.MGVBDCorrectionConfig(coarse_node_limit=1)
        cls.captured_vbd = []
        cls.captured_newton = []
        original_vbd = integration.run_vbd
        original_newton = integration.run_newton

        def capture_vbd(*args, **kwargs):
            result = original_vbd(*args, **kwargs)
            cls.captured_vbd.append(result)
            return result

        def capture_newton(*args, **kwargs):
            result = original_newton(*args, **kwargs)
            cls.captured_newton.append(result)
            return result

        with (
            mock.patch.object(integration, "run_vbd", side_effect=capture_vbd) as vbd_mock,
            mock.patch.object(integration, "run_newton", side_effect=capture_newton) as newton_mock,
        ):
            cls.result = integration.run_multiplicative_mg_vbd(cls.scene, config=cls.config)
        cls.vbd_call_args = tuple(vbd_mock.call_args_list)
        cls.newton_call_count = newton_mock.call_count
        cls.k1_run, cls.k4_run = cls.captured_vbd
        (cls.newton_run,) = cls.captured_newton

    def test_fresh_runs_identity_and_tiny_gate_arithmetic(self):
        quality = self.result.quality
        self.assertEqual(self.newton_call_count, 1)
        self.assertEqual([call.args[1] for call in self.vbd_call_args], [1, 4])
        self.assertIsNot(self.k1_run, self.k4_run)
        self.assertEqual(quality.reference.provenance, "fresh-dense-newton")
        self.assertEqual(quality.vbd_k1.role, "vbd-k1")
        self.assertEqual(quality.vbd_k4.role, "vbd-k4")
        self.assertEqual(quality.vbd_k1.physical_state_sha256, quality.vbd_k4.physical_state_sha256)
        self.assertEqual(quality.vbd_k1.iterate_zero_sha256, quality.vbd_k4.iterate_zero_sha256)
        self.assertNotEqual(quality.vbd_k1.position_sha256, quality.vbd_k4.position_sha256)

        comparison = quality.gate.versus_k4
        tiny = np.finfo(np.float64).tiny
        self.assertEqual(
            comparison.objective_magnitude_ratio,
            abs(quality.final_metrics.objective) / max(abs(quality.k4_metrics.objective), tiny),
        )
        self.assertEqual(
            comparison.residual_ratio,
            quality.final_metrics.relative_residual / max(quality.k4_metrics.relative_residual, tiny),
        )
        self.assertEqual(
            comparison.free_rms_ratio,
            quality.final_metrics.free_rms_error_m / max(quality.k4_metrics.free_rms_error_m, tiny),
        )
        self.assertEqual(
            comparison.mass_weighted_rms_ratio,
            quality.final_metrics.mass_weighted_rms_error_m / max(quality.k4_metrics.mass_weighted_rms_error_m, tiny),
        )
        self.assertLess(comparison.objective_delta, 0.0)
        self.assertLess(comparison.residual_ratio, 0.01)
        self.assertLess(comparison.free_rms_ratio, 0.01)
        self.assertTrue(quality.gate.passed)
        self.assertEqual(quality.accepted_outer_correction_count, 3)
        self.assertEqual(quality.total_v_cycle_count, 12)
        self.assertEqual(quality.final_metrics.max_pin_error_m, 0.0)
        self.assertEqual(quality.final_metrics.inverted_tet_fraction, 0.0)
        self.assertGreater(quality.final_metrics.determinant_min, 0.0)
        np.testing.assert_array_equal(quality.final_positions[self.scene.pinned_indices], self.scene.pin_targets)

        expected_velocity = (quality.final_positions - self.scene.x_current) / self.scene.dt
        expected_velocity[self.scene.pinned_indices] = 0.0
        np.testing.assert_array_equal(quality.final_velocities, expected_velocity)
        self.assertFalse(self.result.timing.deterministic_record()["performance_evidence"])
        self.assertNotIn("timing", quality.deterministic_record())

    def test_outer_current_operator_true_residual_and_static_vcycles(self):
        quality = self.result.quality
        problem = build_common_problem(self.scene)
        current = quality.k1_positions
        for outer_index, outer in enumerate(quality.outer_corrections):
            with self.subTest(outer=outer_index):
                operator = MatrixFreeStableNHOperator.from_problem(problem, current)
                current_matrix = assemble_current_stable_nh_block_matrix(operator)
                pcg = outer.result.pcg
                self.assertIsNotNone(pcg)
                rhs = -operator.gradient_free()
                true_residual = rhs - current_matrix.matmul(pcg.solution)
                np.testing.assert_allclose(
                    np.linalg.norm(true_residual),
                    pcg.true_residual_norm,
                    rtol=2.0e-10,
                    atol=2.0e-14,
                )
                self.assertTrue(pcg.consumed_exact_iteration_count)
                self.assertEqual(pcg.completed_iterations, 4)
                self.assertEqual(len(outer.v_cycle_work), pcg.work.preconditioner_applications)
                self.assertEqual(len(outer.v_cycle_work), 4)
                self.assertTrue(
                    all(work.hierarchy_sha256 == quality.hierarchy.hierarchy_sha256 for work in outer.v_cycle_work)
                )
                self.assertEqual(outer.result.preconditioner_identity, quality.hierarchy.preconditioner_identity)
                current = outer.result.x

    def test_default_stretch_tuned_mg_beats_matched_block_jacobi_and_k4(self):
        scene = build_stretch_scene()
        run = integration.run_multiplicative_mg_vbd(scene)
        quality = run.quality
        problem = build_common_problem(scene)
        block_jacobi_positions = quality.k1_positions
        block_jacobi_applications = 0
        for _ in range(3):
            correction = solve_matrix_free_correction(
                problem,
                block_jacobi_positions,
                MatrixFreeCorrectionConfig(pcg_iterations=4),
            )
            self.assertTrue(correction.accepted)
            self.assertTrue(correction.pcg.consumed_exact_iteration_count)
            block_jacobi_applications += correction.pcg.work.preconditioner_applications
            block_jacobi_positions = correction.x
        block_jacobi = evaluate_common_state(
            problem,
            block_jacobi_positions,
            reference_positions=quality.reference_positions,
        )

        self.assertEqual(quality.total_v_cycle_count, block_jacobi_applications)
        self.assertEqual(block_jacobi_applications, 12)
        self.assertLess(quality.final_metrics.objective, block_jacobi.objective)
        self.assertLess(block_jacobi.objective, quality.k4_metrics.objective)
        self.assertLess(quality.final_metrics.relative_residual, block_jacobi.relative_residual)
        self.assertLess(block_jacobi.relative_residual, quality.k4_metrics.relative_residual)
        self.assertLess(quality.final_metrics.free_rms_error_m, block_jacobi.free_rms_error_m)
        self.assertLess(block_jacobi.free_rms_error_m, quality.k4_metrics.free_rms_error_m)
        self.assertLess(quality.gate.versus_k4.residual_ratio, 0.004)
        self.assertLess(quality.gate.versus_k4.free_rms_ratio, 0.004)
        self.assertTrue(quality.gate.passed)

    def test_identity_mismatch_and_safeguard_fallback_fail_closed(self):
        wrong_k4 = dataclasses.replace(self.k4_run, objective_instance_sha256="0" * 64)
        with (
            mock.patch.object(integration, "run_newton", return_value=self.newton_run),
            mock.patch.object(integration, "run_vbd", side_effect=(self.k1_run, wrong_k4)),
            self.assertRaisesRegex(ValueError, "vbd-k4.*objective_instance_sha256"),
        ):
            integration.run_multiplicative_mg_vbd(self.scene, config=self.config)

        stale_digest_k4 = dataclasses.replace(
            self.k4_run,
            repeat_seconds=(self.k4_run.repeat_seconds[0] + 1.0e-6,),
        )
        with (
            mock.patch.object(integration, "run_newton", return_value=self.newton_run),
            mock.patch.object(integration, "run_vbd", side_effect=(self.k1_run, stale_digest_k4)),
            self.assertRaisesRegex(ValueError, "run digest"),
        ):
            integration.run_multiplicative_mg_vbd(self.scene, config=self.config)

        fallback_config = dataclasses.replace(
            self.config,
            correction=dataclasses.replace(self.config.correction, minimum_determinant=2.0),
        )
        with (
            mock.patch.object(integration, "run_newton", return_value=self.newton_run),
            mock.patch.object(integration, "run_vbd", side_effect=(self.k1_run, self.k4_run)),
        ):
            fallback = integration.run_multiplicative_mg_vbd(self.scene, config=fallback_config).quality
        self.assertEqual(len(fallback.outer_corrections), 1)
        self.assertTrue(fallback.outer_corrections[0].result.used_fallback)
        self.assertEqual(fallback.outer_corrections[0].result.reason, "segment_inversion")
        self.assertFalse(fallback.gate.all_outer_work_completed)
        self.assertTrue(fallback.gate.fallback_used)
        self.assertFalse(fallback.gate.passed)
        np.testing.assert_array_equal(fallback.final_positions, fallback.k1_positions)
        np.testing.assert_array_equal(fallback.final_velocities, fallback.k1_velocities)

    def test_externally_pinned_supplied_reference_and_tamper_rejection(self):
        quality = self.result.quality
        record = integration._thaw_json(quality.reference.source_record)
        digest = quality.reference.source_record_sha256
        with (
            mock.patch.object(integration, "run_newton") as newton_mock,
            mock.patch.object(integration, "run_vbd", side_effect=(self.k1_run, self.k4_run)),
        ):
            supplied = integration.run_multiplicative_mg_vbd(
                self.scene,
                reference_positions=quality.reference_positions,
                reference_record=record,
                expected_reference_record_sha256=digest,
                config=self.config,
            ).quality
        newton_mock.assert_not_called()
        self.assertEqual(
            supplied.reference.provenance,
            "externally-pinned-supplied-reference:dense-cpu-newton-float64",
        )
        self.assertEqual(supplied.quality_sha256, supplied.deterministic_record()["quality_sha256"])

        with self.assertRaisesRegex(ValueError, "verified record"):
            integration.run_multiplicative_mg_vbd(
                self.scene,
                reference_positions=quality.reference_positions,
                config=self.config,
            )
        with self.assertRaisesRegex(ValueError, "expected SHA-256"):
            integration.run_multiplicative_mg_vbd(
                self.scene,
                reference_positions=quality.reference_positions,
                reference_record=record,
                expected_reference_record_sha256="f" * 64,
                config=self.config,
            )

        forged_cases = (
            (dict(record) | {"method": "forged-reference"}, "unsupported method"),
            (dict(record) | {"final_gradient_norm": 0.0}, "final_gradient_norm"),
            (dict(record) | {"verification_converged": False}, "independent verification"),
            (dict(record) | {"config": dict(record["config"]) | {"max_iterations": 49}}, "Newton config"),
        )
        for forged, message in forged_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                integration.run_multiplicative_mg_vbd(
                    self.scene,
                    reference_positions=quality.reference_positions,
                    reference_record=forged,
                    expected_reference_record_sha256=integration._canonical_digest(forged),
                    config=self.config,
                )

    def test_quality_container_recomputes_chain_gate_hierarchy_and_velocity(self):
        quality = self.result.quality
        wrong_gate = dataclasses.replace(quality.gate, exact_pins=False)
        with self.assertRaisesRegex(ValueError, "promotion gate"):
            dataclasses.replace(quality, gate=wrong_gate)

        wrong_start = dataclasses.replace(
            quality.outer_corrections[2],
            outer_index=1,
        )
        trailing = dataclasses.replace(quality.outer_corrections[1], outer_index=2)
        with self.assertRaisesRegex(ValueError, "previous endpoint"):
            dataclasses.replace(
                quality,
                outer_corrections=(quality.outer_corrections[0], wrong_start, trailing),
            )

        wrong_hierarchy = dataclasses.replace(quality.hierarchy, mode_kind="translation")
        with self.assertRaisesRegex(ValueError, "configured hierarchy policy"):
            dataclasses.replace(quality, hierarchy=wrong_hierarchy)

        wrong_config = dataclasses.replace(quality.config, outer_corrections=2)
        with self.assertRaisesRegex(ValueError, "invalid length"):
            dataclasses.replace(quality, config=wrong_config)

        altered_record = integration._thaw_json(quality.reference.source_record)
        altered_record["final_gradient_norm"] = 0.0
        altered_reference = dataclasses.replace(
            quality.reference,
            source_record=altered_record,
            source_record_sha256=integration._canonical_digest(altered_record),
        )
        with self.assertRaisesRegex(ValueError, "final_gradient_norm"):
            dataclasses.replace(quality, reference=altered_reference)

        wrong_velocity = quality.final_velocities.copy()
        wrong_velocity[self.scene.free_indices[0], 0] += 1.0e-12
        with self.assertRaisesRegex(ValueError, "velocity"):
            dataclasses.replace(quality, final_velocities=wrong_velocity)


if __name__ == "__main__":
    unittest.main()
