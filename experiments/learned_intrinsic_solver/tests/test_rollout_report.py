# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check normalization and complete-population rollout statistics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.learned_intrinsic_solver.rollout_report import build_report, summarize_rollout


class TestRolloutReport(unittest.TestCase):
    def test_normalize_each_case_before_aggregating(self):
        rows = summarize_rollout(np.array([[2.0, 100.0, 4.0], [1.0, 10.0, 8.0]]))
        self.assertEqual(rows[0]["mean"], 1.0)
        self.assertEqual(rows[1]["max"], 2.0)
        self.assertEqual(rows[1]["median"], 0.5)
        self.assertAlmostEqual(rows[1]["mean"], (0.5 + 0.1 + 2.0) / 3)

    def test_invalid_case_is_not_silently_omitted_from_full_statistics(self):
        rows = summarize_rollout(np.array([[2.0, 100.0, 4.0], [1.0, np.nan, 8.0]]))
        self.assertEqual(rows[1]["failed_count"], 1)
        self.assertEqual(rows[1]["valid_count"], 2)
        self.assertIsNone(rows[1]["mean"])
        self.assertIsNone(rows[1]["median"])
        self.assertIsNone(rows[1]["max"])
        self.assertAlmostEqual(rows[1]["valid_only_mean"], 1.25)

    def test_zero_initial_energy_cannot_be_normalized(self):
        with self.assertRaisesRegex(ValueError, "initial"):
            summarize_rollout(np.array([[0.0, 1.0], [0.0, 0.5]]))

    def test_first_iteration_failure_still_produces_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            energies = np.array([[2.0, 4.0], [np.nan, 2.0]])
            np.savez(
                output / "rank_0.npz",
                physical_seeds=[10000, 10001],
                energies=energies,
                relative_energies=energies / energies[0],
            )
            (output / "rank_0.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "test-checkpoint",
                        "iterations": 1,
                        "checkpoint_epoch": 198,
                        "parameter_state_unchanged": True,
                        "failures": [{"physical_seed": 10000, "iteration": 1}],
                    }
                )
            )
            report = build_report(output, world_size=1)
            self.assertIsNone(report["step1_normalized_change_for_training_comparison"])
            self.assertEqual(report["rows"][1]["failed_count"], 1)
            self.assertTrue((output / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
