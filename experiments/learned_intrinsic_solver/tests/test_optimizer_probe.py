# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that optimizer diagnostics distinguish differentiation from descent."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.optimizer_probe import _direction_diagnostics, run_optimizer_probe


class TestOptimizerProbe(unittest.TestCase):
    def test_direction_uses_physical_gradient_not_energy_change(self):
        """Classify the literal gradient dot actual corner direction, including a no-op."""
        gradient = torch.tensor([3.0, -4.0])
        descent = _direction_diagnostics(gradient, torch.tensor([1.0, 1.0]))
        ascent = _direction_diagnostics(gradient, torch.tensor([-1.0, -1.0]))
        noop = _direction_diagnostics(gradient, torch.zeros(2))
        self.assertEqual(descent["gradient_dot_direction_joule"], -1.0)
        self.assertEqual(descent["direction_classification"], "descent")
        self.assertEqual(ascent["direction_classification"], "ascent")
        self.assertEqual(noop["direction_classification"], "zero")

    def test_two_updates_report_real_gradients_and_zero_head_noop(self):
        """Keep two real optimizer updates differentiable and serialize honest no-op diagnostics."""
        with tempfile.TemporaryDirectory() as directory:
            report = run_optimizer_probe(
                output_dir=Path(directory), cell_counts=(2, 2, 3), seeds=(0,), iterations=2, threads=2
            )
            self.assertTrue(report["all_graph_checks_passed"])
            self.assertEqual(report["network_hops"], [1])
            cases = {case["head_initialization"]: case for case in report["cases"]}
            self.assertEqual({tuple(case["network_hops"]) for case in cases.values()}, {(1,)})
            zero = cases["zero"]
            self.assertTrue(zero["zero_head_exact_noop"])
            self.assertGreater(zero["parameter_gradients"]["correction_head.weight"]["norm"], 0)
            self.assertEqual(zero["rows"][1]["direction_classification"], "zero")
            diagnostic = cases["diagnostic_nonzero"]
            self.assertEqual(diagnostic["completed_iterations"], 2)
            self.assertTrue(diagnostic["fixed_objective_unchanged"])
            self.assertTrue(diagnostic["source_state_unchanged"])
            self.assertGreater(diagnostic["retained_gradients"][0]["local_target_axes"]["norm"], 0)
            self.assertGreater(diagnostic["retained_gradients"][0]["raw_head"]["norm"], 0)
            self.assertEqual(diagnostic["rows"][-1]["fixed_corner_max_error_m"], 0)
            self.assertGreater(diagnostic["rows"][-1]["min_gauss_jacobian"], 0)
            self.assertEqual(diagnostic["dtype"], "torch.float32")
            saved = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(saved["all_graph_checks_passed"], report["all_graph_checks_passed"])
            with (Path(directory) / "iterations.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["head_initialization"] for row in rows}, {"zero", "diagnostic_nonzero"})


if __name__ == "__main__":
    unittest.main()
