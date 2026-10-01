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
    "relative_residual_curve.svg",
    "per_trajectory_residual_curve.svg",
    "penetration_curve.svg",
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
            "penetration": [
                {"iteration": 0, "mean": 0.2, "max": 0.5, "valid_count": 512},
                {"iteration": 50, "mean": None, "max": None, "valid_count": 511},
                {"iteration": 100, "mean": 0.05, "max": 0.1, "valid_count": 512},
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
                    "gradient_norm": 0.7,
                    "contact_max_penetration_r": 0.3,
                    "contact_pair_mean": 2.5,
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
                    "contact_scene_fraction": 0.75,
                    "contact_realized_fraction": 0.5,
                    "contact_max_penetration_r": 0.3,
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
                        "final_max_penetration_r": {"mean": 0.02, "median": 0.01, "max": 0.06},
                        "selection": {
                            "metric": 0.25,
                            "eligible": True,
                            "aggregation": "mean_final_step_free_force_residual_norm_n",
                            "survival_required": True,
                        },
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
            self.assertIn("<td>Deepest penetration (r)</td><td>0.05</td><td>—</td><td>0.1</td>", html)
            self.assertIn("Contact scenes among this epoch&#x27;s rank-0 trajectories: 75.0%", html)
            self.assertIn("trajectories that made contact (at least one detected pair): 50.0%", html)
            self.assertIn("deepest training penetration: 0.3 r", html)
            self.assertIn("Validation deepest contact penetration (units of r)", html)
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

    def test_penetration_curve_is_written_with_gaps_and_embedded(self):
        """Plot mean and max penetration per iteration, leave gaps at incomplete iterations, and inline it."""
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            svg = (output / "penetration_curve.svg").read_text()
            self.assertIn("Validation deepest contact penetration (units of r)", svg)
            self.assertIn('aria-label="mean"', svg)
            self.assertIn('aria-label="max"', svg)
            self.assertNotIn('aria-label="median"', svg)
            max_path = next(part for part in svg.split("<path ") if 'aria-label="max"' in part)
            self.assertEqual((max_path.count("M"), max_path.count("L")), (2, 0))
            self.assertIn(svg, (output / "index.html").read_text())
            # A validation without the penetration curve (or no validation at all) renders a waiting plot.
            del report["epochs"][0]["validation"]["penetration"]
            write_mixed_report(output, report)
            self.assertIn("Waiting for completed measurements", (output / "penetration_curve.svg").read_text())
            report["epochs"] = []
            write_mixed_report(output, report)
            self.assertIn("Waiting for completed measurements", (output / "penetration_curve.svg").read_text())

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
                    "gradient_norm",
                    "contact_max_penetration_r",
                    "contact_pair_mean",
                ],
            )
            self.assertEqual((update["mean_force_residual_n"], update["tie_cell_count"]), ("0.5", "3"))
            self.assertEqual((update["contact_max_penetration_r"], update["contact_pair_mean"]), ("0.3", "2.5"))
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
            # The fixed-state regime columns follow every pool-regime column and stay blank for pool runs; the
            # full-horizon selection columns of schema 6 come last.
            self.assertEqual(
                epoch_columns[-10:],
                [
                    "contact_scene_fraction",
                    "contact_realized_fraction",
                    "contact_max_penetration_r",
                    "validation_final_max_penetration_r",
                    "regime_stage",
                    "regime_k_max",
                    "regime_h_max",
                    "regime_updates",
                    "full_horizon_selection_metric",
                    "full_horizon_selection_eligible",
                ],
            )
            self.assertEqual((epoch["regime_stage"], epoch["regime_updates"]), ("", ""))
            self.assertEqual(
                (epoch["full_horizon_selection_metric"], epoch["full_horizon_selection_eligible"]), ("0.25", "True")
            )
            # The validation column is the maximum deepest penetration at the final validation iteration.
            self.assertEqual(
                (
                    epoch["contact_scene_fraction"],
                    epoch["contact_realized_fraction"],
                    epoch["contact_max_penetration_r"],
                    epoch["validation_final_max_penetration_r"],
                ),
                ("0.75", "0.5", "0.3", "0.1"),
            )
            self.assertEqual(report["epochs"][0].get("selection_metric"), None)
            self.assertEqual(report["epochs"][0].get("validation_final_max_penetration_r"), None)

    def test_full_horizon_table_uses_the_latest_completed_check(self):
        """Show K, H, survivors, final residual and seconds from the most recent full-horizon run."""
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn(
                "<td>1</td><td>1</td><td>8</td><td>16 / 16</td><td>0.25 / 0.2 / 0.9</td><td>1.25</td>"
                "<td>0.02 / 0.06</td><td>42.5</td>",
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

    def test_selection_source_names_the_selecting_summary_and_its_record(self):
        """Name the summary that selects the best checkpoint, its metric and, for the full-horizon source, its record."""
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            # The legacy default: the cheap validation selects; the full-horizon check is shown for comparison.
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn(
                "Selection metric (cheap validation: mean free-corner force residual after 100 iterations): "
                "0.125 N, eligible for checkpoint selection. Physical survivors: 512 / 512. "
                "Best so far: 0.125 N at epoch 1.",
                page,
            )
            self.assertIn("squares mark full-horizon checks", page)
            svg = (output / "loss_curve.svg").read_text()
            self.assertIn("Cheap validation, final iteration (selects)", svg)
            self.assertNotIn("Full horizon, final step (selects)", svg)
            # The v4 source: the full-horizon check selects and its record explains the choice.
            report["config"]["selection_source"] = "full_horizon"
            report["best_selection"] = {
                "epoch": 1,
                "source": "full_horizon",
                "metric": 0.25,
                "aggregation": "mean_final_step_free_force_residual_norm_n",
                "completed_updates": 128,
                "iterations": 1,
                "physical_steps": 8,
                "sample_count": 16,
                "physical_survivors": 16,
                "final_energy_joule": {"mean": 1.25, "median": 1.0, "max": 3.0},
                "final_max_penetration_r": {"mean": 0.02, "median": 0.01, "max": 0.06},
            }
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn(
                "Selection metric (full-horizon check: mean final free-corner force residual after K = 1 learned "
                "iterations on each of H = 8 physical steps, epoch 1): 0.25 N, eligible for checkpoint selection. "
                "Full-horizon survivors: 16 / 16. Best so far: 0.25 N at epoch 1 (full-horizon check, K = 1, H = 8; "
                "final energy mean 1.25 J; deepest penetration mean 0.02 / max 0.06 r).",
                page,
            )
            self.assertIn(
                "Cheap validation metric (mean free-corner force residual after 100 iterations): 0.125 N, "
                "all trajectories survived. Physical survivors: 512 / 512.",
                page,
            )
            self.assertIn("The best checkpoint is selected on the full-horizon check", page)
            self.assertNotIn("squares mark full-horizon checks", page)
            svg = (output / "loss_curve.svg").read_text()
            self.assertIn("Full horizon, final step (selects)", svg)
            self.assertNotIn("Cheap validation, final iteration (selects)", svg)
            # An ineligible full-horizon check and no record yet.
            report["epochs"][0]["full_horizon_validation"]["selection"] = {"metric": None, "eligible": False}
            report["epochs"][0]["full_horizon_validation"]["physical_survivors"] = 15
            report["best_selection"] = None
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn(
                "Unavailable, not eligible (a failed or incomplete trajectory). Full-horizon survivors: 15 / 16. "
                "No eligible epoch has been selected yet.",
                page,
            )
            # Before any full-horizon check the headline says so instead of failing.
            report["epochs"][0]["full_horizon_validation"] = None
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn("after not evaluated yet): Unavailable, eligibility not recorded.", page)
            self.assertIn("Full-horizon survivors: Not evaluated.", page)

    def test_rows_without_validation_leave_gaps_and_show_the_latest_validated_epoch(self):
        """A skipped-validation row writes blank validation cells and keeps the last validated epoch's section."""
        report = self._report()
        report["config"]["validation_interval"] = 2
        skipped = copy.deepcopy(report["epochs"][0])
        skipped.update(epoch=2, validation=None, full_horizon_validation=None, learning_rate=9e-5)
        report["epochs"].append(skipped)
        report["completed_epochs"] = 2
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            with (output / "epochs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
            self.assertEqual([row["selection_metric"] for row in rows], ["0.125", ""])
            self.assertEqual(
                tuple(
                    rows[1][key]
                    for key in (
                        "selection_eligible",
                        "physical_survivors",
                        "sample_count",
                        "validation_final_max_penetration_r",
                    )
                ),
                ("", "", "", ""),
            )
            self.assertEqual(rows[1]["loss"], "-0.25")
            page = (output / "index.html").read_text()
            self.assertIn("2 / 500", page)
            self.assertIn("Validation runs every 2 epochs and on the final epoch.", page)
            self.assertIn("Epoch 1, 512 fixed validation seeds", page)
            self.assertIn("0.125 N, eligible for checkpoint selection", page)
            self.assertIn("<td>1</td><td>1</td><td>8</td><td>16 / 16</td>", page)
            self.assertIn("100", (output / "validation_curve.svg").read_text())
            self.assertIn("<svg", (output / "loss_curve.svg").read_text())
            self.assertIsNone(json.loads((output / "report.json").read_text())["epochs"][1]["validation"])
            # Without an interval above one the note is absent and a run without any validation still renders.
            del report["config"]["validation_interval"]
            report["epochs"][0].update(validation=None, full_horizon_validation=None)
            report["best_selection"] = None
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertNotIn("Validation runs every", page)
            self.assertIn("Epoch —, 0 fixed validation seeds", page)
            self.assertIn("No eligible epoch has been selected yet.", page)
            self.assertIn("Waiting for completed measurements", (output / "residual_curve.svg").read_text())

    def test_selection_reset_history_and_the_shown_validation_budget_are_labelled(self):
        """Name the earlier record after a selection reset; label the shown validation and full check by their budgets."""
        report = self._report()
        report["config"].update(
            validation_count=128, validation_iterations=32, validation_interval=4, validation_full_iterations=8
        )
        earlier = report["best_selection"]
        report["best_selection"] = None
        report["best_selection_history"] = [
            {"reset_at_epoch": 1, "reason": "validation budget changed", "record": earlier}
        ]
        skipped = copy.deepcopy(report["epochs"][0])
        skipped.update(epoch=2, validation=None, full_horizon_validation=None)
        report["epochs"].append(skipped)
        report["completed_epochs"] = 2
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn(
                "No eligible epoch has been selected yet. Selection restarted after epoch 1 (validation budget "
                "changed); the earlier record was 0.125 N at epoch 1.",
                page,
            )
            # The shown validation is the epoch-1 row measured over 100 iterations, not the configured 32.
            self.assertIn("Latest validation: 100 optimizer iterations", page)
            self.assertIn("force residual after 100 iterations", page)
            self.assertIn("<th>At iteration 100</th>", page)
            self.assertIn("Epoch 1, 512 fixed validation seeds", page)
            self.assertIn("128 fixed validation states. Validation runs every 4 epochs", page)
            self.assertIn("at the largest currently available H and at K capped at 8;", page)
            self.assertNotIn("largest currently available budgets", page)
            # A new best after the reset keeps the note; a reset without an earlier record says so.
            report["best_selection"] = {"epoch": 2, "metric": 0.5}
            report["best_selection_history"].append(
                {"reset_at_epoch": 2, "reason": "validation budget changed", "record": None}
            )
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn(
                "Best so far: 0.5 N at epoch 2. Selection restarted after epoch 2 (validation budget changed); "
                "there was no earlier eligible record.",
                page,
            )
            # Without a validated epoch the labels fall back to the configured budget; without a cap or a
            # reset the prose is unchanged.
            report["epochs"][0].update(validation=None, full_horizon_validation=None)
            del report["config"]["validation_full_iterations"]
            del report["best_selection_history"]
            write_mixed_report(output, report)
            page = (output / "index.html").read_text()
            self.assertIn("Latest validation: 32 optimizer iterations", page)
            self.assertIn("<th>At iteration 32</th>", page)
            self.assertIn("at the largest currently available budgets;", page)
            self.assertNotIn("Selection restarted", page)

    def test_fixed_state_runs_show_the_stage_counts_timetable_and_the_weights_only_origin(self):
        """Replace the pool's query and curriculum lines by the fixed-state stage, counts and timetable; name the origin."""
        report = self._report()
        report["config"].update(
            regime="fixed_states",
            state_count=2048,
            budget_cap=1024,
            growth_stage_epochs=2,
            growth_stages=[[1, 8], [2, 16], [4, 32]],
            queries_per_epoch=8192,
            stage_descent_rate=0.8,
            stage_max_epochs=20,
        )
        stage = {"name": "fixed_states", "stage": 0, "k_max": 1, "h_max": 8}
        report["epochs"][0]["regime"] = {**stage, "queries": 9300, "filler_queries": [3, 0, 2, 1], "updates": 146}
        report["epochs"][0]["curriculum"] = None
        report["initialized_from"] = {
            "checkpoint": "/runs/training_v3_contact_20260927_r2/checkpoints/best_validation.pt",
            "sha256": "abc",
            "completed_epochs": 62,
            "completed_updates": 7936,
            "best_selection": {"epoch": 60, "metric": 13.1},
        }
        report["progress"] = {
            "phase": "training",
            "epoch": 2,
            "regime": {**stage, "queries": 9400, "filler_queries": [1, 1, 1, 1], "updates": 147},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_mixed_report(output, report)
            html = (output / "index.html").read_text()
            self.assertIn(
                "Fixed-state regime: 2048 training states per epoch, K x H ≤ 1024; growth stage 0 (K ≤ 1, H ≤ 8): "
                "9400 sampled queries + 4 filler in 147 updates per rank and 512 fixed validation states.",
                html,
            )
            self.assertIn(
                "Growth timetable (K_max, H_max) from stage 0: (1, 8) → (2, 16) → (4, 32), advancing every 2 epochs; "
                "the final stage persists. The validation-gated curriculum is not used.",
                html,
            )
            self.assertIn(
                "Initialized weights-only (network and AdamW state) from "
                "training_v3_contact_20260927_r2/checkpoints/best_validation.pt after 62 completed epochs, "
                "best 13.1 N at epoch 60.",
                html,
            )
            self.assertNotIn("training queries per epoch", html)
            self.assertNotIn("Curriculum descent gate", html)
            with (output / "epochs.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                (row["regime_stage"], row["regime_k_max"], row["regime_h_max"], row["regime_updates"]),
                ("0", "1", "8", "146"),
            )
            # Before the first job list is assigned the heartbeat names the stage only.
            report["progress"] = {"phase": "initializing", "epoch": 1, "regime": stage}
            write_mixed_report(output, report)
            html = (output / "index.html").read_text()
            self.assertIn("growth stage 0 (K ≤ 1, H ≤ 8) and 512 fixed validation states.", html)
            # Without a heartbeat the latest completed row supplies the stage.
            del report["progress"]
            write_mixed_report(output, report)
            self.assertIn(
                "9300 sampled queries + 6 filler in 146 updates per rank", (output / "index.html").read_text()
            )
            # The pool regime's status line is unchanged.
            pool = self._report()
            pool["config"].update(queries_per_epoch=8192, stage_descent_rate=0.8, stage_max_epochs=20)
            write_mixed_report(output, pool)
            html = (output / "index.html").read_text()
            self.assertIn("8192 training queries per epoch and 512 fixed validation states.", html)
            self.assertIn("Curriculum descent gate: 80% · Hard cap: 20 epochs per stage</p>", html)
            self.assertNotIn("Fixed-state regime", html)
            self.assertNotIn("Initialized weights-only", html)

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
