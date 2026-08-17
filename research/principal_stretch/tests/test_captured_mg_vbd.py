# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the first fully captured multiplicative MG-VBD composition."""

from __future__ import annotations

import inspect
import json
import os
import unittest

import numpy as np
import warp as wp

from research.principal_stretch.captured_mg_vbd import (
    CONTRACT_ID,
    OUTER_CORRECTIONS,
    PCG_ITERATIONS,
    CapturedMGVBDTiming,
    CapturedMultiplicativeMGVBD,
)
from research.principal_stretch.correction_gpu import (
    MatrixFreeCorrectionConfig,
    MatrixFreeStableNHOperator,
    minimum_determinant_on_segment,
)
from research.principal_stretch.correction_mg_vbd import MGVBDCorrectionConfig, run_multiplicative_mg_vbd
from research.principal_stretch.solver_benchmark import (
    build_common_problem,
    build_structured_cantilever_scene,
    evaluate_common_state,
)
from research.principal_stretch.solver_scenes import build_stretch_scene


def _scene(dimensions=(2, 2, 1), *, name="captured-mg-vbd-test"):
    return build_structured_cantilever_scene(
        dimensions=dimensions,
        dt=1.0 / 16.0,
        gravity=(0.1, -0.2, -2.0),
        total_tip_force=(4.0, -3.0, -6.0),
        initial_velocity=(0.03, -0.02, 0.01),
        name=name,
    )


class TestCapturedMultiplicativeMGVBD(unittest.TestCase):
    def test_source_uses_no_newton_internal_api_and_retains_strict_gates(self):
        module = __import__("research.principal_stretch.captured_mg_vbd", fromlist=["*"])
        source = inspect.getsource(module)
        self.assertNotIn("newton._src", source)
        self.assertNotIn("from newton import _src", source)
        self.assertIn("_segment_minimum", source)
        self.assertIn("end_objective < start_objective", source)
        self.assertIn("pcg_completed[0] == PCG_ITERATIONS", source)

    def test_contract_is_fixed_and_cpu_is_rejected(self):
        scene = _scene()
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            CapturedMultiplicativeMGVBD(scene, device="cpu")

    def test_tiny_scene_without_strict_coarsening_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strict coarsening"):
            CapturedMultiplicativeMGVBD(_scene((1, 1, 1), name="captured-mg-vbd-too-small"), device="cuda:0")

    def test_diagnostic_timing_requires_balanced_pairs_and_warmup(self):
        common = {
            "mg_seconds": (1.0e-3, 1.1e-3),
            "k4_seconds": (2.0e-4, 2.1e-4),
            "random_seed": 17,
            "device": "cuda:0",
        }
        with self.assertRaisesRegex(ValueError, "equal AB and BA"):
            CapturedMGVBDTiming(pair_orders=("AB", "AB"), warmup_replays=1, **common)
        with self.assertRaisesRegex(ValueError, "positive built-in int"):
            CapturedMGVBDTiming(pair_orders=("AB", "BA"), warmup_replays=0, **common)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestCapturedMultiplicativeMGVBDCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.scene = _scene()
        cls.solver = CapturedMultiplicativeMGVBD(cls.scene, device="cuda:0")
        cls.solver.capture_graphs(warmup_replays=1)

    def test_graph_replay_resets_poison_and_matches_uncaptured_lane(self):
        eager = self.solver.run(graph_replay=False)
        self.solver.poison(seed=123)
        captured = self.solver.run(graph_replay=True)
        self.solver.poison(seed=456)
        repeated = self.solver.run(graph_replay=True)

        self.assertEqual(eager.accepted, (True,) * OUTER_CORRECTIONS)
        self.assertEqual(eager.reasons, ("accepted",) * OUTER_CORRECTIONS)
        self.assertEqual(eager.pcg_completed_iterations, (PCG_ITERATIONS,) * OUTER_CORRECTIONS)
        np.testing.assert_array_equal(captured.positions, eager.positions)
        np.testing.assert_array_equal(captured.velocities, eager.velocities)
        np.testing.assert_array_equal(repeated.positions, eager.positions)
        np.testing.assert_array_equal(repeated.velocities, eager.velocities)
        self.assertEqual(captured.accepted, eager.accepted)
        self.assertEqual(captured.reasons, eager.reasons)
        self.assertTrue(captured.graph_replay)
        self.assertTrue(repeated.graph_replay)

        record = self.solver.deterministic_record()
        self.assertEqual(record["contract_id"], CONTRACT_ID)
        self.assertEqual(record["outer_corrections"], OUTER_CORRECTIONS)
        self.assertEqual(record["pcg_iterations_per_outer"], PCG_ITERATIONS)
        self.assertTrue(record["separate_k4_graph"])
        self.assertFalse(record["performance_evidence"])
        json.dumps(record, allow_nan=False)

    def test_paired_timing_is_balanced_and_diagnostic_only(self):
        timing = self.solver.benchmark_paired(pair_count=6, warmup_replays=2, random_seed=4817)
        self.assertEqual(timing.pair_orders.count("AB"), 3)
        self.assertEqual(timing.pair_orders.count("BA"), 3)
        self.assertGreater(timing.mg_median_seconds, 0.0)
        self.assertGreater(timing.k4_median_seconds, 0.0)
        self.assertTrue(timing.integrated_mg)
        self.assertFalse(timing.setup_included)
        self.assertFalse(timing.transfers_included)
        self.assertFalse(timing.performance_evidence)
        json.dumps(timing.deterministic_record(), allow_nan=False)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestCapturedMultiplicativeMGVBDRealStretchCuda(unittest.TestCase):
    """Cross-check the captured device gates against the CPU quality oracle."""

    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.scene = build_stretch_scene(dimensions=(2, 1, 1))
        cls.config = MGVBDCorrectionConfig(coarse_node_limit=1)
        cls.cpu = run_multiplicative_mg_vbd(cls.scene, config=cls.config)
        cls.solver = CapturedMultiplicativeMGVBD(cls.scene, device="cuda:0", config=cls.config)
        cls.solver.capture_graphs(warmup_replays=1)
        cls.endpoint = cls.solver.run(graph_replay=True)

    def test_fixed_work_gate_and_current_operator_match_cpu_oracle(self):
        endpoint = self.endpoint
        quality = self.cpu.quality
        self.assertEqual(endpoint.accepted, (True, True, False))
        self.assertEqual(endpoint.reasons, ("accepted", "accepted", "objective-increase"))
        self.assertEqual(endpoint.pcg_statuses, (1,) * OUTER_CORRECTIONS)
        self.assertEqual(endpoint.pcg_completed_iterations, (PCG_ITERATIONS,) * OUTER_CORRECTIONS)
        np.testing.assert_allclose(endpoint.outer_start_positions[0], quality.k1_positions, rtol=0.0, atol=4.0e-9)

        problem = build_common_problem(self.scene)
        operators = []
        for outer_index, (gpu_start, gpu_candidate) in enumerate(
            zip(endpoint.outer_start_positions, endpoint.outer_candidate_positions, strict=True)
        ):
            with self.subTest(outer=outer_index):
                np.testing.assert_array_equal(gpu_start, gpu_start.astype(np.float32).astype(np.float64))
                np.testing.assert_array_equal(gpu_candidate, gpu_candidate.astype(np.float32).astype(np.float64))
                initial = MatrixFreeStableNHOperator.from_problem(problem, gpu_start)
                candidate = MatrixFreeStableNHOperator.from_problem(problem, gpu_candidate)
                operators.append(initial)
                actual_step = gpu_candidate[initial.free] - gpu_start[initial.free]
                actual_directional = float(np.vdot(initial.gradient_free().reshape(-1), actual_step.reshape(-1)))
                actual_segment = minimum_determinant_on_segment(initial, candidate).determinant
                self.assertAlmostEqual(endpoint.initial_objectives[outer_index], initial.objective(), delta=5.0e-15)
                self.assertAlmostEqual(endpoint.candidate_objectives[outer_index], candidate.objective(), delta=5.0e-15)
                self.assertAlmostEqual(
                    endpoint.directional_derivatives[outer_index],
                    actual_directional,
                    delta=3.0e-13,
                )
                self.assertAlmostEqual(
                    endpoint.segment_minimum_determinants[outer_index],
                    actual_segment,
                    delta=1.0e-11,
                )
                if endpoint.accepted[outer_index]:
                    self.assertLess(
                        endpoint.candidate_objectives[outer_index], endpoint.initial_objectives[outer_index]
                    )
                    self.assertLessEqual(
                        endpoint.candidate_objectives[outer_index],
                        endpoint.initial_objectives[outer_index]
                        + self.config.correction.armijo * endpoint.directional_derivatives[outer_index],
                    )
                    self.assertGreater(
                        endpoint.segment_minimum_determinants[outer_index],
                        self.config.correction.minimum_determinant,
                    )
                else:
                    self.assertGreater(
                        endpoint.candidate_objectives[outer_index], endpoint.initial_objectives[outer_index]
                    )

        self.assertFalse(np.array_equal(operators[0].cofactors, operators[1].cofactors))
        self.assertFalse(np.array_equal(operators[1].cofactors, operators[2].cofactors))
        np.testing.assert_array_equal(endpoint.positions, endpoint.outer_start_positions[2])
        metrics = evaluate_common_state(problem, endpoint.positions, reference_positions=quality.reference_positions)
        self.assertLess(metrics.relative_residual, quality.k4_metrics.relative_residual)
        self.assertLess(metrics.free_rms_error_m, quality.k4_metrics.free_rms_error_m)

    def test_endpoint_pins_exact_segment_safety_bdf1_and_k4(self):
        endpoint = self.endpoint
        np.testing.assert_array_equal(endpoint.positions[self.scene.pinned_indices], self.scene.pin_targets)
        np.testing.assert_array_equal(endpoint.velocities[self.scene.pinned_indices], 0.0)
        free = self.scene.free_indices
        expected_velocity = (endpoint.positions[free] - self.scene.x_current[free]) / self.scene.dt
        np.testing.assert_allclose(endpoint.velocities[free], expected_velocity, rtol=0.0, atol=2.0e-9)
        metrics = evaluate_common_state(build_common_problem(self.scene), endpoint.positions)
        self.assertEqual(metrics.inverted_tet_fraction, 0.0)
        self.assertGreater(metrics.determinant_min, 0.0)

        k4 = self.solver.run_k4(graph_replay=True)
        np.testing.assert_allclose(k4.positions, self.cpu.quality.k4_positions, rtol=0.0, atol=4.0e-9)
        self.assertFalse(k4.integrated_mg)

    def test_poisoned_capture_replay_restores_all_retained_outer_evidence(self):
        expected = self.endpoint
        for replay_index in range(3):
            self.solver.poison(seed=9001 + replay_index)
            actual = self.solver.run(graph_replay=True)
            np.testing.assert_array_equal(actual.positions, expected.positions)
            np.testing.assert_array_equal(actual.velocities, expected.velocities)
            self.assertEqual(actual.accepted, expected.accepted)
            self.assertEqual(actual.reasons, expected.reasons)
            self.assertEqual(actual.initial_objectives, expected.initial_objectives)
            self.assertEqual(actual.candidate_objectives, expected.candidate_objectives)
            self.assertEqual(actual.directional_derivatives, expected.directional_derivatives)
            self.assertEqual(actual.segment_minimum_determinants, expected.segment_minimum_determinants)
            self.assertEqual(actual.endpoint_sha256, expected.endpoint_sha256)
            for actual_start, expected_start in zip(
                actual.outer_start_positions, expected.outer_start_positions, strict=True
            ):
                np.testing.assert_array_equal(actual_start, expected_start)
            for actual_candidate, expected_candidate in zip(
                actual.outer_candidate_positions, expected.outer_candidate_positions, strict=True
            ):
                np.testing.assert_array_equal(actual_candidate, expected_candidate)

    def test_endpoint_evidence_is_immutable_and_content_bound(self):
        endpoint = self.endpoint
        with self.assertRaises(ValueError):
            endpoint.positions.setflags(write=True)
        with self.assertRaises(ValueError):
            endpoint.outer_candidate_positions[0].setflags(write=True)
        record = endpoint.deterministic_record()
        self.assertEqual(record["endpoint_sha256"], endpoint.endpoint_sha256)
        self.assertEqual(len(record["current_operator_sha256s"]), OUTER_CORRECTIONS)
        self.assertEqual(len(set(record["current_operator_sha256s"])), OUTER_CORRECTIONS)
        json.dumps(record, allow_nan=False)

    def test_segment_rejection_is_fail_closed_and_masks_remaining_outer_work(self):
        rejecting_config = MGVBDCorrectionConfig(
            correction=MatrixFreeCorrectionConfig(minimum_determinant=2.0),
            coarse_node_limit=1,
        )
        solver = CapturedMultiplicativeMGVBD(self.scene, device="cuda:0", config=rejecting_config)
        solver.capture_graphs(warmup_replays=1)
        endpoint = solver.run(graph_replay=True)
        self.assertEqual(endpoint.accepted, (False, False, False))
        self.assertEqual(endpoint.reasons, ("segment-inversion", "masked-after-rejection", "masked-after-rejection"))
        np.testing.assert_array_equal(endpoint.positions, endpoint.outer_start_positions[0].astype(np.float32))
        self.assertGreater(endpoint.segment_minimum_determinants[0], 0.0)
        self.assertLess(endpoint.segment_minimum_determinants[0], 2.0)
        self.assertEqual(endpoint.initial_objectives[1:], (0.0, 0.0))
        self.assertEqual(endpoint.candidate_objectives[1:], (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
