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


class TestMixedReport(unittest.TestCase):
    def _report(self):
        return {
            "config": {"max_epochs": 500, "validation_iterations": 100},
            "status": "running",
            "completed_epochs": 1,
            "completed_updates": 128,
            "updates": [{"update": 128, "epoch": 1, "loss": -0.25, "before_joule": 2, "after_joule": 1.5}],
            "epochs": [
                {
                    "epoch": 1,
                    "loss": -0.25,
                    "query_count": 8192,
                    "seconds": 10,
                    "available_K": [1],
                    "available_H": [8],
                    "validation": {
                        "sample_count": 512,
                        "failed_count": 0,
                        "physical_survivors": 512,
                        "relative_energy": [
                            {"iteration": 0, "mean": 1.0, "median": 1.0, "max": 1.0},
                            {"iteration": 100, "mean": 0.5, "median": 0.4, "max": 0.9},
                        ],
                    },
                }
            ],
        }

    def test_report_renders_metrics_without_mutating_training_state(self):
        """Render the epoch loss and all three validation statistics into portable files."""

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
            self.assertIn("100", (output / "validation_curve.svg").read_text())

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
            self.assertEqual(
                {path.name for path in public.iterdir()},
                {
                    "index.html",
                    "report.json",
                    "progress.json",
                    "epochs.csv",
                    "updates.csv",
                    "loss_curve.svg",
                    "validation_curve.svg",
                },
            )
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

    def test_acceptance_diagnostics_distinguish_missing_history_from_measured_zero(self):
        """Keep historical acceptance unknown and display recorded fractions after enabling the guard."""
        report = self._report()
        report["configuration_changes"] = [
            {"field": "geometry_backtracking", "previous": False, "current": True, "effective_from_epoch": 43}
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn("Geometry acceptance: enabled from epoch 43", page)
            self.assertIn("<td>Training epoch 1</td><td>not recorded</td><td>not recorded</td>", page)
            self.assertIn("<td>Validation optimizer</td><td>not recorded</td>", page)
            self.assertIn("<td>Validation physical rollout</td><td>not recorded</td>", page)
            for name in ("epochs", "updates"):
                with (output / f"{name}.csv").open() as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["shortened_query_count"], "")
                self.assertEqual(row["mean_acceptance_scale"], "")

            latest = copy.deepcopy(report["epochs"][0])
            latest.update(epoch=43, query_count=64, shortened_query_count=8, mean_acceptance_scale=0.9375)
            latest["validation"].update(
                optimization_query_count=200,
                optimization_shortened_query_count=0,
                physical_query_count=32,
                physical_shortened_query_count=4,
            )
            report["epochs"].append(latest)
            report["updates"].append(
                {"update": 5503, "epoch": 43, "shortened_query_count": 8, "mean_acceptance_scale": 0.9375}
            )
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn("<td>Training epoch 43</td><td>8 / 64</td><td>0.9375</td>", page)
            self.assertIn("<td>Validation optimizer</td><td>0 / 200</td>", page)
            self.assertIn("<td>Validation physical rollout</td><td>4 / 32</td>", page)
            for name in ("epochs", "updates"):
                with (output / f"{name}.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["shortened_query_count"], "")
                self.assertEqual(rows[-1]["shortened_query_count"], "8")
                self.assertEqual(rows[-1]["mean_acceptance_scale"], "0.9375")

    def test_acceptance_status_uses_configuration_without_a_resume_event(self):
        """Report enabled and disabled guard settings in fresh campaigns."""
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            for enabled, label in ((True, "enabled"), (False, "disabled")):
                with self.subTest(enabled=enabled):
                    report["config"]["geometry_backtracking"] = enabled
                    write_mixed_report(directory, report)
                    self.assertIn(
                        f"Geometry acceptance: {label}</summary>", (Path(directory) / "index.html").read_text()
                    )


if __name__ == "__main__":
    unittest.main()
