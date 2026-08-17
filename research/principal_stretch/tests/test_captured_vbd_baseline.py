# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the captured public SolverVBD K1/K4 baseline."""

from __future__ import annotations

import inspect
import json
import os
import unittest

import numpy as np
import warp as wp

from research.principal_stretch.captured_vbd_baseline import (
    CONTRACT_ID,
    ITERATION_BUDGETS,
    MUTATION_HAZARDS,
    CapturedPublicVBDBaseline,
)
from research.principal_stretch.solver_benchmark import build_structured_cantilever_scene


def _scene():
    return build_structured_cantilever_scene(
        dimensions=(1, 1, 1),
        dt=1.0 / 16.0,
        gravity=(0.1, -0.2, -2.0),
        total_tip_force=(4.0, -3.0, -6.0),
        initial_velocity=(0.03, -0.02, 0.01),
        name="captured-public-vbd-baseline-test",
    )


class TestCapturedPublicVBDBaseline(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _scene()
        self.baseline = CapturedPublicVBDBaseline(self.scene, device="cpu", tile_solve=False)

    def test_contract_is_public_hermetic_and_fail_closed(self):
        source = inspect.getsource(__import__("research.principal_stretch.captured_vbd_baseline", fromlist=["*"]))
        self.assertNotIn("newton._src", source)
        self.assertNotIn("from newton import _src", source)

        record = self.baseline.deterministic_record()
        self.assertEqual(record["contract_id"], CONTRACT_ID)
        self.assertEqual(record["iteration_budgets"], list(ITERATION_BUDGETS))
        self.assertEqual(record["scene_sha256"], self.scene.manifest()["scene_sha256"])
        self.assertEqual(len(record["model_sha256"]), 64)
        self.assertEqual(len(record["pristine_state_sha256"]), 64)
        self.assertEqual(record["mutation_hazards"], list(MUTATION_HAZARDS))
        self.assertEqual(len(record["reset_arrays"]), 6)
        self.assertTrue(record["research_only"])
        self.assertTrue(record["diagnostic_baseline"])
        self.assertFalse(record["integrated_mg"])
        self.assertFalse(record["performance_evidence"])
        json.dumps(record, allow_nan=False)

        with self.assertRaisesRegex(ValueError, "K1 or K4"):
            self.baseline.run(2)
        with self.assertRaisesRegex(RuntimeError, "CUDA graph capture"):
            self.baseline.capture_graphs()
        with self.assertRaisesRegex(RuntimeError, "CUDA-event timing"):
            self.baseline.benchmark_paired()

    def test_cpu_poisoned_restarts_match_uncaptured_public_run_vbd_bitwise(self):
        endpoints = {}
        for iterations in ITERATION_BUDGETS:
            endpoint, reference = self.baseline.validate_against_run_vbd(iterations, graph_replay=False)
            endpoints[iterations] = endpoint
            self.assertEqual(endpoint.position_sha256, reference.result_state_sha256)
            self.assertEqual(endpoint.max_pin_error_m, 0.0)
            self.assertGreater(endpoint.minimum_determinant, 0.0)
            self.assertFalse(endpoint.graph_replay)
            self.assertFalse(endpoint.integrated_mg)
            self.assertFalse(endpoint.performance_evidence)

            for replay_index in range(3):
                self.baseline.poison_lane(iterations, seed=701 + 17 * iterations + replay_index)
                replay = self.baseline.run(iterations, graph_replay=False)
                np.testing.assert_array_equal(replay.positions, endpoint.positions)
                np.testing.assert_array_equal(replay.velocities, endpoint.velocities)
                self.assertEqual(replay.endpoint_sha256, endpoint.endpoint_sha256)
                self.assertEqual(replay.position_fp32_sha256, endpoint.position_fp32_sha256)
                self.assertEqual(replay.pristine_state_sha256, endpoint.pristine_state_sha256)

        self.assertNotEqual(endpoints[1].position_sha256, endpoints[4].position_sha256)

    def test_mutated_pristine_state_is_rejected(self):
        self.baseline.run(1)
        corrupted = self.baseline.pristine_input.particle_f.numpy()
        corrupted[0, 0] += np.float32(1.0)
        self.baseline.pristine_input.particle_f.assign(corrupted)
        with self.assertRaisesRegex(RuntimeError, "pristine input state"):
            self.baseline.run(1)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestCapturedPublicVBDBaselineCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.scene = _scene()
        cls.baseline = CapturedPublicVBDBaseline(cls.scene, device="cuda:0", tile_solve=False)
        cls.baseline.capture_graphs(warmup_replays=2)

    def test_graph_replays_reset_poison_and_match_public_run_vbd(self):
        for iterations in ITERATION_BUDGETS:
            expected, reference = self.baseline.validate_against_run_vbd(iterations, graph_replay=True)
            self.assertTrue(expected.graph_replay)
            self.assertEqual(expected.position_sha256, reference.result_state_sha256)
            self.assertEqual(expected.max_pin_error_m, 0.0)
            self.assertGreater(expected.minimum_determinant, 0.0)
            for replay_index in range(3):
                self.baseline.poison_lane(iterations, seed=1901 + 31 * iterations + replay_index)
                replay = self.baseline.run(iterations, graph_replay=True)
                np.testing.assert_array_equal(replay.positions, expected.positions)
                np.testing.assert_array_equal(replay.velocities, expected.velocities)
                self.assertEqual(replay.endpoint_sha256, expected.endpoint_sha256)

    def test_balanced_randomized_ab_ba_event_timing_excludes_setup_and_transfers(self):
        timing = self.baseline.benchmark_paired(pair_count=6, warmup_replays=3, random_seed=4817)
        self.assertEqual(timing.pair_orders.count("AB"), 3)
        self.assertEqual(timing.pair_orders.count("BA"), 3)
        self.assertGreater(timing.k1_median_seconds, 0.0)
        self.assertGreater(timing.k4_median_seconds, timing.k1_median_seconds)
        self.assertFalse(timing.setup_included)
        self.assertFalse(timing.transfers_included)
        self.assertFalse(timing.integrated_mg)
        self.assertFalse(timing.performance_evidence)
        record = timing.deterministic_record()
        json.dumps(record, allow_nan=False)
        print("CAPTURED_PUBLIC_VBD_TIMING=" + json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
