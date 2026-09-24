# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for complete-validation plateau decisions."""

import unittest

from experiments.learned_intrinsic_solver.training_schedule import PlateauController


def validation(loss=-0.01, descent=0.98, failed=0):
    return {
        "mean_normalized_loss": loss,
        "mean_before_joule": 10.0,
        "mean_after_joule": 9.9,
        "descent_rate": descent,
        "failed_count": failed,
    }


class TestTrainingSchedule(unittest.TestCase):
    def test_success_requires_minimum_epochs_and_good_recent_validation(self):
        controller = PlateauController()
        for epoch in range(1, 30):
            self.assertFalse(controller.observe(epoch, validation())["stop"])
        result = controller.observe(30, validation())
        self.assertEqual(result["status"], "plateau_converged")
        self.assertLess(result["learning_rate"], 1e-4)

    def test_flat_poor_training_is_stalled(self):
        controller = PlateauController()
        for epoch in range(1, 31):
            result = controller.observe(epoch, validation(descent=0.5))
        self.assertEqual(result["status"], "stalled")

    def test_invalid_validation_cannot_improve_best(self):
        controller = PlateauController()
        controller.observe(1, validation())
        for epoch in range(2, 7):
            result = controller.observe(epoch, validation(loss=-100.0, failed=1))
        self.assertEqual(result["learning_rate"], 5e-5)
        self.assertEqual(controller.state_dict()["best_loss"], -0.01)

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
            result = controller.observe(epoch, validation(loss=-0.01 * epoch))
        self.assertEqual(result["status"], "epoch_limit")
        self.assertEqual(result["learning_rate"], 1e-4)


if __name__ == "__main__":
    unittest.main()
