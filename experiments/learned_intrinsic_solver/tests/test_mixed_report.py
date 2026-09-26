# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify live mixed-training report data and publication boundaries."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.learned_intrinsic_solver.mixed_report import write_mixed_report, write_progress
from experiments.learned_intrinsic_solver.publish_mixed_report import prepare_public_report

_PUBLIC_FILES = {
    "index.html",
    "report.json",
    "progress.json",
    "epochs.csv",
    "updates.csv",
    "loss_curve.svg",
    "validation_curve.svg",
    "residual_curve.svg",
}


class TestMixedReport(unittest.TestCase):
    def _validation(self, *, metric=0.125, eligible=True, survivors=512):
        return {
            "sample_count": 512,
            "failed_count": 0,
            "physical_survivors": survivors,
            "mean_normalized_loss": -0.3,
            "descent_rate": 0.9,
            "mean_before_joule": 2.0,
            "mean_after_joule": 1.5,
            "relative_energy": [
                {"iteration": 0, "mean": 1.0, "median": 1.0, "max": 1.0},
                {"iteration": 100, "mean": 0.5, "median": 0.4, "max": 0.9},
            ],
            "force_residual": [
                {"iteration": 0, "mean": 3.0, "median": 2.5, "max": 7.0, "valid_count": 512, "failed_count": 0},
                {"iteration": 50, "mean": None, "median": None, "max": None, "valid_count": 511, "failed_count": 1},
                {"iteration": 100, "mean": 0.125, "median": 0.1, "max": 0.75, "valid_count": 512, "failed_count": 0},
            ],
            "selection": {
                "metric": metric,
                "eligible": eligible,
                "aggregation": "mean_final_free_force_residual_norm_n",
                "survival_required": True,
            },
        }

    def _report(self):
        return {
            "config": {"max_epochs": 500, "validation_iterations": 100, "validation_full_interval": 5},
            "status": "running",
            "completed_epochs": 1,
            "completed_updates": 128,
            "best_selection": {"epoch": 1, "metric": 0.125, "aggregation": "mean_final_free_force_residual_norm_n"},
            "updates": [
                {
                    "update": 128,
                    "epoch": 1,
                    "loss": -0.25,
                    "before_joule": 2,
                    "after_joule": 1.5,
                    "mean_force_residual_n": 0.5,
                    "step_size_mean": 0.025,
                    "step_size_min": 0.02,
                    "step_size_max": 0.03,
                    "tie_cell_count": 3,
                }
            ],
            "epochs": [
                {
                    "epoch": 1,
                    "loss": -0.25,
                    "query_count": 8192,
                    "seconds": 10,
                    "mean_force_residual_n": 0.5,
                    "step_size_mean": 0.025,
                    "step_size_min": 0.02,
                    "step_size_max": 0.03,
                    "tie_cell_count": 3,
                    "candidate_modes": {"inertial": 4100, "perturbed_inertial": 4092},
                    "available_K": [1],
                    "available_H": [8],
                    "learning_rate": 1e-4,
                    "validation": self._validation(),
                    "full_horizon_validation": {
                        "sample_count": 16,
                        "physical_survivors": 16,
                        "failed_count": 0,
                        "iterations": 1,
                        "physical_steps": 8,
                        "final_free_force_residual_norm_n": {"mean": 0.25, "median": 0.2, "max": 0.9},
                        "final_energy_joule": {"mean": 1.25, "median": 1.0, "max": 3.0},
                        "seconds": 42.5,
                    },
                }
            ],
        }

    def test_report_renders_metrics_without_mutating_training_state(self):
        """Render the epoch loss, both validation curves and the selection headline into portable files."""

        report = self._report()
        original = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report, updated_at="2026-09-24T12:00:00+00:00")
            self.assertEqual(report, original)
            saved = json.loads((output / "report.json").read_text())
            self.assertEqual(saved["updated_at"], "2026-09-24T12:00:00+00:00")
            self.assertEqual(saved["epochs"], report["epochs"])
            html = (output / "index.html").read_text()
            self.assertIn("1 / 500", html)
            self.assertIn("0.5", html)
            self.assertIn("0.4", html)
            self.assertIn("0.9", html)
            self.assertIn("refresh", html)
            self.assertIn("30", html)
            self.assertIn("Mean local training loss", html)
            self.assertIn("asinh", html)
            self.assertIn("0.125 N, eligible for checkpoint selection", html)
            self.assertIn("Physical survivors: 512 / 512", html)
            self.assertIn("Best so far: 0.125 N at epoch 1", html)
            self.assertIn("<td>Force residual (N)</td><td>0.125</td><td>0.1</td><td>0.75</td>", html)
            self.assertIn("100", (output / "validation_curve.svg").read_text())
            residual = (output / "residual_curve.svg").read_text()
            self.assertIn("100", residual)
            self.assertIn('aria-label="median"', residual)
            self.assertNotIn("acceptance", html.lower())
            self.assertNotIn("shortened", html.lower())

    def test_residual_curve_leaves_gaps_for_incomplete_iterations(self):
        """Break the residual line where an iteration has failed samples, like the energy curve."""
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            residual = (output / "residual_curve.svg").read_text()
            mean_path = next(part for part in residual.split("<path ") if 'aria-label="mean"' in part)
            self.assertEqual(mean_path.count("M"), 2)
            self.assertEqual(mean_path.count("L"), 0)
            report["epochs"][0]["validation"]["force_residual"][1].update(mean=1.0, median=0.9, max=2.0)
            write_mixed_report(output, report)
            residual = (output / "residual_curve.svg").read_text()
            mean_path = next(part for part in residual.split("<path ") if 'aria-label="mean"' in part)
            self.assertEqual(mean_path.count("M"), 1)
            self.assertEqual(mean_path.count("L"), 2)

    def test_csv_tables_carry_residual_step_and_selection_columns_only(self):
        """Write the revised per-update and per-epoch columns without acceptance fields."""
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            with (output / "updates.csv").open() as handle:
                reader = csv.DictReader(handle)
                update_columns = reader.fieldnames
                update = next(reader)
            self.assertEqual(
                update_columns,
                [
                    "update",
                    "epoch",
                    "loss",
                    "before_joule",
                    "after_joule",
                    "mean_force_residual_n",
                    "step_size_mean",
                    "step_size_min",
                    "step_size_max",
                    "tie_cell_count",
                ],
            )
            self.assertEqual((update["mean_force_residual_n"], update["tie_cell_count"]), ("0.5", "3"))
            with (output / "epochs.csv").open() as handle:
                reader = csv.DictReader(handle)
                epoch_columns = reader.fieldnames
                epoch = next(reader)
            for removed in ("shortened_query_count", "mean_acceptance_scale"):
                self.assertNotIn(removed, update_columns)
                self.assertNotIn(removed, epoch_columns)
            self.assertEqual(
                (
                    epoch["selection_metric"],
                    epoch["selection_eligible"],
                    epoch["physical_survivors"],
                    epoch["sample_count"],
                ),
                ("0.125", "True", "512", "512"),
            )
            self.assertEqual(epoch["step_size_max"], "0.03")
            self.assertEqual(report["epochs"][0].get("selection_metric"), None)

    def test_full_horizon_table_uses_the_latest_completed_check(self):
        """Show K, H, survivors, final residual and seconds from the most recent full-horizon run."""
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn(
                "<td>1</td><td>1</td><td>8</td><td>16 / 16</td><td>0.25 / 0.2 / 0.9</td><td>1.25</td><td>42.5</td>",
                page,
            )
            later = copy.deepcopy(report["epochs"][0])
            later.update(epoch=2, full_horizon_validation=None)
            later["validation"] = self._validation(metric=None, eligible=False, survivors=511)
            report["epochs"].append(later)
            report["completed_epochs"] = 2
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn("<td>1</td><td>1</td><td>8</td><td>16 / 16</td>", page)
            self.assertIn("Unavailable, not eligible (a failed or incomplete trajectory)", page)
            self.assertIn("Physical survivors: 511 / 512", page)
            report["epochs"][0]["full_horizon_validation"] = None
            report["best_selection"] = None
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn("No full-horizon validation has completed yet.", page)
            self.assertIn("No eligible epoch has been selected yet.", page)

    def test_progress_keeps_physical_curriculum_and_latest_update(self):
        """Publish lightweight progress with actual current K/H and update counters."""

        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            write_progress(directory, report, phase="validation", epoch=2, available_K=[1, 2], available_H=[8, 16])
            progress = json.loads((Path(directory) / "progress.json").read_text())
            self.assertEqual(progress["phase"], "validation")
            self.assertEqual(progress["epoch"], 2)
            self.assertEqual(progress["available_K"], [1, 2])
            self.assertEqual(progress["available_H"], [8, 16])
            self.assertEqual(progress["completed_updates"], 128)
            self.assertNotIn("updates", progress)
            self.assertNotIn("epochs", progress)

    def test_mirror_excludes_checkpoints_and_preserves_failure(self):
        """Publish only generated metrics when local state and a failure coexist."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, public = root / "run", root / "public"
            write_mixed_report(run, self._report())
            write_progress(run, self._report(), phase="training", epoch=2)
            (run / "failure.json").write_text(json.dumps({"error": "bad <state>", "completed_updates": 129}))
            (run / "checkpoints").mkdir()
            (run / "checkpoints/latest.pt").write_bytes(b"private")
            (run / "failure_rank_0.pt").write_bytes(b"private")
            (run / "rank_0.log").write_text("private raw diagnostics")
            prepare_public_report(run, public, training_running=False, seen_training=True)
            self.assertEqual({path.name for path in public.iterdir()}, _PUBLIC_FILES)
            saved = json.loads((public / "report.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["failure"]["error"], "bad <state>")
            self.assertIn("bad &lt;state&gt;", (public / "index.html").read_text())

    def test_missing_run_is_preparing_without_creating_training_output(self):
        """Prepare a public waiting page without blocking the trainer's fresh-run guard."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_public_report(root / "absent", root / "public", training_running=False)
            self.assertFalse((root / "absent").exists())
            report = json.loads((root / "public/report.json").read_text())
            self.assertEqual(report["status"], "preparing")
            self.assertEqual(report["completed_epochs"], 0)
            self.assertEqual({path.name for path in (root / "public").iterdir()}, _PUBLIC_FILES)


if __name__ == "__main__":
    unittest.main()
