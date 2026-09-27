# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for selection-metric plateau decisions."""

import unittest

from experiments.learned_intrinsic_solver.training_schedule import PlateauController


def validation(metric=1.0, *, eligible=True, descent=0.98, failed=0, survivors=10, count=10):
    return {
        "selection": {
            "metric": metric,
            "eligible": eligible,
            "aggregation": "mean_final_free_force_residual_norm_n",
            "survival_required": True,
        },
        "mean_normalized_loss": -0.01,
        "mean_before_joule": 10.0,
        "mean_after_joule": 9.9,
        "descent_rate": descent,
        "failed_count": failed,
        "physical_survivors": survivors,
        "sample_count": count,
    }


class TestTrainingSchedule(unittest.TestCase):
    def test_success_requires_minimum_epochs_and_good_recent_validation(self):
        controller = PlateauController()
        for epoch in range(1, 30):
            self.assertFalse(controller.observe(epoch, validation())["stop"])
        result = controller.observe(30, validation())
        self.assertEqual(result["status"], "plateau_converged")
        self.assertLess(result["learning_rate"], 1e-4)

    def test_flat_poor_descent_or_lost_trajectory_is_stalled(self):
        for changes in ({"descent": 0.5}, {"survivors": 9}):
            with self.subTest(changes=changes):
                controller = PlateauController()
                for epoch in range(1, 31):
                    result = controller.observe(epoch, validation(**changes))
                self.assertEqual(result["status"], "stalled")

    def test_ineligible_validation_cannot_improve_best(self):
        """Ignore a spectacular residual when the selection record is ineligible."""
        controller = PlateauController()
        controller.observe(1, validation(1.0))
        for epoch in range(2, 7):
            result = controller.observe(epoch, validation(1e-3, eligible=False, failed=1, survivors=9))
        self.assertEqual(result["learning_rate"], 5e-5)
        self.assertEqual(controller.state_dict()["best_loss"], 1.0)
        self.assertEqual(controller.bad_epochs, 5)

    def test_selection_metric_drives_improvement_with_relative_threshold(self):
        """Read the selection metric, not the first-update loss, and compare relatively."""
        controller = PlateauController()
        controller.observe(1, validation(1.0))
        self.assertEqual(controller.best_loss, 1.0)
        controller.observe(2, validation(0.9995))
        self.assertEqual(controller.best_loss, 1.0)
        self.assertEqual(controller.bad_epochs, 1)
        controller.observe(3, validation(0.998))
        self.assertEqual(controller.best_loss, 0.998)
        self.assertEqual(controller.bad_epochs, 0)
        small = PlateauController()
        small.observe(1, validation(1e-6))
        small.observe(2, validation(0.9e-6))
        self.assertEqual(small.best_loss, 0.9e-6)
        self.assertEqual(small.bad_epochs, 0)
        legacy = PlateauController()
        legacy.observe(1, {"mean_normalized_loss": -5.0, "failed_count": 0, "descent_rate": 1.0})
        self.assertIsNone(legacy.best_loss)
        self.assertEqual(legacy.bad_epochs, 1)
        for metric in (None, float("nan"), float("inf")):
            with self.subTest(metric=metric):
                fresh = PlateauController()
                fresh.observe(1, validation(metric))
                self.assertIsNone(fresh.best_loss)

    def test_only_validated_epochs_are_observed_and_must_increase(self):
        """Skipped epochs are absent: patience counts validations while the cap follows the training epoch."""
        controller = PlateauController(min_epochs=1, max_epochs=8, lr_patience=2)
        controller.observe(2, validation(1.0))
        controller.observe(4, validation(1.0))
        result = controller.observe(6, validation(1.0))
        # Two non-improving validations, not four elapsed epochs, trigger the halving.
        self.assertEqual((controller.bad_epochs, result["learning_rate"], controller.last_epoch), (2, 5e-5, 6))
        self.assertEqual(controller.observe(8, validation(1.0))["status"], "epoch_limit")
        for epoch in (8, 3, 0):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "increase"):
                controller.observe(epoch, validation())
        with self.assertRaises(ValueError):
            PlateauController().observe(0, validation())

    def test_reset_metric_history_forgets_the_best_and_patience_but_keeps_the_learning_rate(self):
        """A changed validation budget restarts comparisons without undoing learning-rate reductions."""
        controller = PlateauController(min_epochs=1, max_epochs=50, lr_patience=2)
        controller.observe(1, validation(1.0))
        controller.observe(2, validation(1.0, descent=0.5))
        controller.observe(3, validation(1.0))
        self.assertEqual((controller.best_loss, controller.bad_epochs, controller.reductions), (1.0, 2, 1))
        self.assertEqual((controller.learning_rate, controller.recent_good), (5e-5, [True, False, True]))
        controller.reset_metric_history()
        self.assertEqual(
            (controller.best_loss, controller.bad_epochs, controller.lr_bad_epochs, controller.recent_good),
            (None, 0, 0, []),
        )
        self.assertEqual((controller.reductions, controller.learning_rate, controller.last_epoch), (1, 5e-5, 3))
        # A worse metric under the new budget is the new best rather than a bad epoch.
        result = controller.observe(4, validation(2.0))
        self.assertEqual((controller.best_loss, controller.bad_epochs, result["learning_rate"]), (2.0, 0, 5e-5))
        self.assertEqual(set(controller.state_dict()), set(PlateauController().state_dict()))

    def test_resume_matches_uninterrupted_schedule(self):
        controller = PlateauController()
        for epoch in range(1, 13):
            controller.observe(epoch, validation())
        resumed = PlateauController()
        resumed.load_state_dict(controller.state_dict())
        for epoch in range(13, 31):
            self.assertEqual(controller.observe(epoch, validation()), resumed.observe(epoch, validation()))

    def test_meaningful_improvement_resets_plateau_and_epoch_limit_is_distinct(self):
        controller = PlateauController(min_epochs=3, max_epochs=8)
        for epoch in range(1, 9):
            result = controller.observe(epoch, validation(1.0 / epoch))
        self.assertEqual(result["status"], "epoch_limit")
        self.assertEqual(result["learning_rate"], 1e-4)

    def test_disabled_early_stopping_keeps_reducing_learning_rate_until_epoch_limit(self):
        """Continue useful or poor flat validation through the requested 500 epochs."""
        for descent in (0.98, 0.5):
            with self.subTest(descent=descent):
                controller = PlateauController(max_epochs=500)
                for epoch in range(1, 500):
                    result = controller.observe(epoch, validation(descent=descent), allow_early_stop=False)
                    self.assertEqual(result["status"], "running", f"epoch {epoch}")
                    self.assertFalse(result["stop"], f"epoch {epoch}")
                self.assertEqual(result["learning_rate"], 1e-6)
                result = controller.observe(500, validation(descent=descent), allow_early_stop=False)
                self.assertEqual(result, {"learning_rate": 1e-6, "stop": True, "status": "epoch_limit"})

    def test_old_controller_state_can_resume_with_early_stopping_disabled(self):
        """Extend the epoch cap without changing checkpoint controller fields."""
        controller = PlateauController()
        for epoch in range(1, 30):
            controller.observe(epoch, validation())
        resumed = PlateauController(max_epochs=500)
        resumed.load_state_dict(controller.state_dict())
        self.assertEqual(
            set(resumed.state_dict()),
            {
                "learning_rate",
                "min_epochs",
                "max_epochs",
                "lr_patience",
                "stop_patience",
                "threshold",
                "min_lr",
                "best_loss",
                "bad_epochs",
                "lr_bad_epochs",
                "reductions",
                "last_epoch",
                "recent_good",
            },
        )
        for epoch in range(30, 500):
            self.assertFalse(resumed.observe(epoch, validation(), allow_early_stop=False)["stop"])
        self.assertEqual(resumed.observe(500, validation(), allow_early_stop=False)["status"], "epoch_limit")


if __name__ == "__main__":
    unittest.main()
