# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Task evidence must never convert incomplete frame data into a pass."""

import copy
import unittest

import numpy as np

from experiments.collision_benchmark.comparison_cases import ComparisonScene
from experiments.collision_benchmark.comparison_tasks import criterion_results


def _protocol(prepared=False):
    return {
        "frame_dt_s": 0.1,
        "duration_s": 65.0,
        "hold_start": 2.0,
        "hold_end": 62.0,
        "initial_hold_prepared": prepared,
        "category_applicability": {"grasp": not prepared, "hold": True, "transport": True, "place_release": True},
        "support_reference_height_m": 0.25,
        "transport_intervals": [{"start_s": 0.5, "end_s": 2.0}],
        "place_release_intervals": [{"start_s": 62.0, "end_s": 65.0}],
    }


def _frames(duration, height=0.5):
    return [
        {
            "time_s": i / 10,
            "minimum_shell_z_m": height,
            "particle_mean_m": [0.001 * i, 0.0, 0.5],
            "recipe_observation": {"hand_soft_contact_records": [10, 10]},
        }
        for i in range(round(duration * 10) + 1)
    ]


class TestTaskEvidence(unittest.TestCase):
    def test_partial_hold_cannot_fill_sixty_second_result(self):
        result = criterion_results(_frames(12), protocol=_protocol())
        ten, sixty = result["hold"]["measurements"]["checkpoints"]
        self.assertTrue(ten["observed_interval_complete_at_frame_cadence"])
        self.assertFalse(sixty["observed_interval_complete_at_frame_cadence"])
        self.assertEqual(result["hold"]["status"], "incomplete")

    def test_sag_and_centroid_motion_are_not_interface_slip(self):
        result = criterion_results(_frames(65), protocol=_protocol())
        hold = result["hold"]["measurements"]
        self.assertIsNone(hold["interface_slip_m"])
        self.assertGreater(hold["checkpoints"][1]["maximum_center_excursion_m"], 0)
        self.assertFalse(hold["checkpoints"][1]["center_motion_is_interface_slip"])
        self.assertFalse(hold["continuous_retention_qualified"])

    def test_prepared_contact_does_not_claim_acquisition(self):
        result = criterion_results(_frames(65), protocol=_protocol(prepared=True))
        self.assertEqual(result["grasp"]["status"], "not_applicable")
        self.assertFalse(result["grasp"]["measurements"]["acquisition_claimed_by_recipe"])

    def test_height_below_finite_table_does_not_prove_failure(self):
        result = criterion_results(_frames(65, height=0.1), protocol=_protocol())
        self.assertEqual(result["hold"]["status"], "incomplete")
        self.assertLess(
            result["hold"]["measurements"]["checkpoints"][0]["minimum_sampled_shell_height_above_support_m"], 0
        )

    def test_detected_contacts_do_not_prove_grasp_or_release(self):
        frames = _frames(65)
        for frame in frames:
            frame["recipe_observation"]["hand_soft_contact_records"] = [0, 0]
        result = criterion_results(frames, protocol=_protocol())
        self.assertEqual(result["grasp"]["status"], "incomplete")
        self.assertIsNone(result["place_release"]["measurements"]["final_hand_geometry"])
        self.assertFalse(result["place_release"]["measurements"]["released_on_intended_support_qualified"])

    def test_missing_frame_does_not_prove_continuity(self):
        frames = _frames(65)
        frames.pop(30)
        result = criterion_results(frames, protocol=_protocol())
        self.assertFalse(
            result["hold"]["measurements"]["checkpoints"][0]["observed_interval_complete_at_frame_cadence"]
        )


class TestPhysicalClock(unittest.TestCase):
    @staticmethod
    def _case():
        return {
            "case_id": "synthetic_clock_test",
            "frame_dt_s": 1 / 30,
            "substeps": 10,
            "iterations": 30,
            "hold_start": 5.5,
            "hold_end": 7.0,
            "stationary_windows": [[5.5, 7.0]],
            "applicability": {"analytic_freefall": False},
            "transport_windows": [[4.0, 5.5]],
            "place_release_windows": [[7.0, 11.5]],
        }

    def test_extend_only_stationary_command_time(self):
        class Raw:
            duration, time, frame_dt = 11.5, 0.0, 1 / 30

            def step(self):
                self.time += self.frame_dt

        scene = ComparisonScene(Raw(), self._case())
        self.assertEqual(scene.duration, 70)
        self.assertEqual(scene.protocol["hold_end"], 65.5)
        for elapsed, command in ((0, 0), (6.25, 6.25), (10, 6.25), (64.75, 6.25), (65.5, 7), (70, 11.5)):
            self.assertAlmostEqual(scene.motion_time_at(elapsed), command)
        for _ in range(2100):
            scene.step()
        self.assertEqual(scene.time, 70)
        self.assertAlmostEqual(scene.raw_scene.time, 11.5)

    def test_stationary_bounds_snap_inward_and_keep_requested_bounds(self):
        case = copy.deepcopy(self._case())
        case.update(hold_start=6.442184148847824, hold_end=7.942184148847824)
        case["stationary_windows"] = [[case["hold_start"], case["hold_end"]]]
        raw = type("Raw", (), {"duration": 15.53436829769565})()
        scene = ComparisonScene(raw, case)
        interval = scene.protocol["stationary_intervals"][0]
        self.assertGreaterEqual(interval["start_s"], interval["requested_start_s"])
        self.assertLessEqual(interval["end_s"], interval["requested_end_s"])
        self.assertAlmostEqual(interval["start_s"] / scene.dt, np.round(interval["start_s"] / scene.dt))
        self.assertAlmostEqual(interval["end_s"] / scene.dt, np.round(interval["end_s"] / scene.dt))


if __name__ == "__main__":
    unittest.main()
