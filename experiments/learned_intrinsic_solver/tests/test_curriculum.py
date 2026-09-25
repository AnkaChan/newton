# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify validation-gated mixed-trajectory curriculum progression."""

import copy
import json
import unittest

from experiments.learned_intrinsic_solver.curriculum import MixedCurriculum


def _validation(**updates):
    result = {
        "failed_count": 0,
        "sample_count": 10,
        "physical_survivors": 10,
        "mean_before_joule": 10.0,
        "mean_after_joule": 9.0,
        "descent_rate": 0.9,
    }
    result.update(updates)
    return result


class TestMixedCurriculum(unittest.TestCase):
    def test_require_minimum_residence_and_consecutive_qualified_validation(self):
        """Wait for both residence and an unbroken qualifying validation streak."""
        curriculum = MixedCurriculum(min_stage_epochs=3, patience=2)
        self.assertEqual(curriculum.available_counts, ((1,), (8,)))
        first = curriculum.observe(_validation())
        self.assertEqual(
            first,
            {
                "stage": 0,
                "advanced": False,
                "advance_reason": None,
                "qualified": True,
                "stage_epochs": 1,
                "qualified_epochs": 1,
            },
        )
        self.assertFalse(curriculum.observe(_validation())["advanced"])
        failure = curriculum.observe(_validation(failed_count=1))
        self.assertFalse(failure["qualified"])
        self.assertEqual(failure["qualified_epochs"], 0)
        self.assertFalse(curriculum.observe(_validation())["advanced"])
        advanced = curriculum.observe(_validation())
        self.assertEqual(
            advanced,
            {
                "stage": 1,
                "advanced": True,
                "advance_reason": "validation",
                "qualified": True,
                "stage_epochs": 0,
                "qualified_epochs": 0,
            },
        )
        self.assertEqual(curriculum.available_counts, ((1, 2), (8, 16)))

    def test_failed_stalled_or_incomplete_validation_never_advances(self):
        """Reject inversion failures, poor descent, lost trajectories and nonfinite energy."""
        failures = [
            {"failed_count": 1},
            {"mean_after_joule": 10.0},
            {"mean_after_joule": 11.0},
            {"descent_rate": 0.799},
            {"descent_rate": float("nan")},
            {"mean_before_joule": float("inf")},
            {"mean_after_joule": float("nan")},
            {"mean_after_joule": None},
            {"physical_survivors": 9},
            {"sample_count": 0, "physical_survivors": 0},
        ]
        for changes in failures:
            with self.subTest(changes=changes):
                curriculum = MixedCurriculum(min_stage_epochs=1, patience=1)
                for _ in range(12):
                    decision = curriculum.observe(_validation(**changes))
                    self.assertFalse(decision["qualified"])
                    self.assertFalse(decision["advanced"])
                    self.assertEqual(curriculum.stage, 0)
        self.assertFalse(MixedCurriculum(min_stage_epochs=1, patience=1).observe({})["qualified"])

    def test_all_caps_retain_shorter_counts_and_stop_at_final_stage(self):
        """Expose each approved count set while preserving shorter trajectories."""
        expected = [
            ((1,), (8,)),
            ((1, 2), (8, 16)),
            ((1, 2, 4), (8, 16, 32)),
            ((1, 2, 4, 8), (8, 16, 32, 64)),
            ((1, 2, 4, 8, 16), (8, 16, 32, 64, 128)),
            ((1, 2, 4, 8, 16, 32), (8, 16, 32, 64, 128)),
        ]
        curriculum = MixedCurriculum(min_stage_epochs=1, patience=1)
        for stage, counts in enumerate(expected):
            self.assertEqual(curriculum.stage, stage)
            self.assertEqual(curriculum.available_counts, counts)
            decision = curriculum.observe(_validation())
            self.assertEqual(decision["advanced"], stage < 5)
        for _ in range(12):
            self.assertFalse(curriculum.observe(_validation())["advanced"])
        self.assertEqual(curriculum.stage, 5)

    def test_configured_threshold_and_smaller_count_sets_are_preserved(self):
        """Use configured thresholds without creating unavailable K or H choices."""
        curriculum = MixedCurriculum((1, 2), (1, 2), min_stage_epochs=1, patience=1, min_descent_rate=0.8)
        self.assertEqual(curriculum.available_counts, ((1,), (1, 2)))
        self.assertFalse(curriculum.observe(_validation(descent_rate=0.79))["qualified"])
        for _ in range(5):
            self.assertTrue(curriculum.observe(_validation(descent_rate=0.8))["advanced"])
            self.assertEqual(curriculum.available_counts, ((1, 2), (1, 2)))

    def test_default_gate_accepts_eighty_percent_inclusively(self):
        """Advance at exactly the user-selected eighty-percent descent rate."""
        curriculum = MixedCurriculum(min_stage_epochs=1, patience=1)
        self.assertTrue(curriculum.observe(_validation(descent_rate=0.8))["advanced"])

    def test_hard_limit_advances_despite_failed_validation(self):
        """Force one stage transition at the epoch limit without claiming qualification."""
        curriculum = MixedCurriculum(min_stage_epochs=1, patience=2, max_stage_epochs=2)
        self.assertFalse(curriculum.observe(_validation(failed_count=1))["advanced"])
        decision = curriculum.observe(_validation(failed_count=1))
        self.assertTrue(decision["advanced"])
        self.assertFalse(decision["qualified"])
        self.assertEqual(decision["advance_reason"], "max_stage_epochs")
        self.assertEqual(curriculum.stage, 1)
        self.assertEqual(curriculum.stage_epochs, 0)
        self.assertEqual(curriculum.qualified_epochs, 0)

    def test_validation_can_advance_before_hard_limit(self):
        """Keep the existing quality gate as the earlier promotion route."""
        curriculum = MixedCurriculum(min_stage_epochs=1, patience=1, max_stage_epochs=2)
        self.assertEqual(curriculum.observe(_validation())["advance_reason"], "validation")

    def test_legacy_state_has_no_hard_limit_and_overdue_resume_advances_once(self):
        """Load old counters exactly and enforce a newly supplied cap without validation."""
        curriculum = MixedCurriculum()
        for _ in range(30):
            curriculum.observe(_validation(descent_rate=0.0))
        state = curriculum.state_dict()
        state.pop("max_stage_epochs", None)
        restored = MixedCurriculum(max_stage_epochs=20)
        restored.load_state_dict(state)
        self.assertIsNone(restored.max_stage_epochs)
        self.assertEqual(restored.stage_epochs, 30)
        restored.max_stage_epochs = 20
        self.assertTrue(restored.advance_if_overdue())
        self.assertEqual(restored.available_counts, ((1, 2), (8, 16)))
        self.assertEqual(restored.stage_epochs, 0)
        self.assertFalse(restored.advance_if_overdue())

    def test_hard_limit_round_trip_and_final_stage_do_not_skip_stages(self):
        """Preserve the cap on resume and stop advancing at the last stage."""
        curriculum = MixedCurriculum(min_stage_epochs=1, patience=2, max_stage_epochs=1)
        restored = MixedCurriculum()
        restored.load_state_dict(curriculum.state_dict())
        for _ in range(5):
            self.assertEqual(curriculum.observe({}), restored.observe({}))
        self.assertEqual(restored.stage, 5)
        self.assertFalse(restored.observe({})["advanced"])

    def test_resume_preserves_streak_and_future_decisions(self):
        """Resume serialized state with identical stage gates and available counts."""
        curriculum = MixedCurriculum(min_stage_epochs=3, patience=2, min_descent_rate=0.95)
        for _ in range(4):
            curriculum.observe(_validation(descent_rate=0.98))
        saved = json.loads(json.dumps(curriculum.state_dict()))
        resumed = MixedCurriculum()
        resumed.load_state_dict(saved)
        for changes in ({}, {"failed_count": 1}, {}, {}, {}, {}, {}, {}):
            value = _validation(descent_rate=0.98, **changes)
            self.assertEqual(curriculum.observe(value), resumed.observe(value))
            self.assertEqual(curriculum.available_counts, resumed.available_counts)
            self.assertEqual(curriculum.state_dict(), resumed.state_dict())

    def test_invalid_configuration_or_checkpoint_leaves_state_unchanged(self):
        """Validate budgets and checkpoint counters before mutating live progress."""
        for kwargs in (
            {"min_stage_epochs": 0},
            {"patience": True},
            {"min_descent_rate": float("nan")},
            {"min_descent_rate": 1.1},
            {"max_stage_epochs": 9},
            {"max_stage_epochs": True},
            {"max_stage_epochs": 20.5},
            {"iteration_counts": ()},
            {"iteration_counts": (2,)},
            {"iteration_counts": (1, 64)},
            {"physical_step_counts": (16,)},
            {"physical_step_counts": (8, 256)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MixedCurriculum(**kwargs)
        curriculum = MixedCurriculum()
        curriculum.observe(_validation())
        original = curriculum.state_dict()
        for updates in ({"stage": 6}, {"qualified_epochs": 2}, {"stage_epochs": -1}, {"version": 99}):
            corrupted = copy.deepcopy(original)
            corrupted.update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                curriculum.load_state_dict(corrupted)
            self.assertEqual(curriculum.state_dict(), original)


if __name__ == "__main__":
    unittest.main()
