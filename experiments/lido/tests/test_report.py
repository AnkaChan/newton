# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""report.py writes what the existing dashboard renderer and publisher read (docs/dashboard_keys.md)."""

import contextlib
import csv
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.learned_intrinsic_solver.mixed_report import write_mixed_report
from experiments.learned_intrinsic_solver.publish_mixed_report import prepare_public_report
from experiments.lido import report as R
from experiments.lido.config import TrainConfig
from experiments.lido.structs import FailureRecord

# Every key path the dashboard reads (docs/dashboard_keys.md sections 3 and 4). "0" indexes a list.
REPORT_PATHS = """
status updated_at completed_epochs completed_updates world_size parameter_count config epochs updates
best_selection best_selection_history initialized_from git_sha feature_schema_version
config.max_epochs config.validation_iterations config.validation_interval config.validation_count
config.validation_full_iterations config.validation_full_interval config.selection_source config.batch_size
config.damping_range config.regime config.state_count config.budget_cap config.growth_stages
config.growth_stage_epochs config.queries_per_epoch config.stage_descent_rate config.stage_max_epochs
config.energy_floor_scale config.energy_increase_weight
epochs.0.epoch epochs.0.loss epochs.0.query_count epochs.0.seconds epochs.0.mean_force_residual_n
epochs.0.step_size_mean epochs.0.step_size_min epochs.0.step_size_max epochs.0.tie_cell_count
epochs.0.gradient_norm_mean epochs.0.gradient_norm_max epochs.0.learning_rate
epochs.0.contact_scene_fraction epochs.0.contact_realized_fraction epochs.0.contact_max_penetration_r
epochs.0.available_K epochs.0.available_H
epochs.0.regime.name epochs.0.regime.stage epochs.0.regime.k_max epochs.0.regime.h_max epochs.0.regime.queries
epochs.0.regime.filler_queries epochs.0.regime.updates epochs.0.regime.step_cap epochs.0.regime.pinned_fraction
epochs.0.regime.resting_fraction
epochs.0.validation.mean_normalized_loss epochs.0.validation.descent_rate epochs.0.validation.mean_before_joule
epochs.0.validation.mean_after_joule epochs.0.validation.selection.metric epochs.0.validation.selection.eligible
epochs.0.validation.physical_survivors epochs.0.validation.sample_count epochs.0.validation.failed_count
epochs.0.validation.first_update_failed_count epochs.0.validation.failures
epochs.0.validation.relative_energy.0.iteration epochs.0.validation.relative_energy.0.mean
epochs.0.validation.relative_energy.0.median epochs.0.validation.relative_energy.0.max
epochs.0.validation.relative_energy.0.near_zero_count
epochs.0.validation.force_residual.0.iteration epochs.0.validation.force_residual.0.mean
epochs.0.validation.force_residual.0.median epochs.0.validation.force_residual.0.max
epochs.0.validation.penetration.0.iteration epochs.0.validation.penetration.0.mean epochs.0.validation.penetration.0.max
epochs.0.validation.samples
epochs.0.full_horizon_validation.iterations epochs.0.full_horizon_validation.physical_steps
epochs.0.full_horizon_validation.physical_survivors epochs.0.full_horizon_validation.sample_count
epochs.0.full_horizon_validation.seconds
epochs.0.full_horizon_validation.final_free_force_residual_norm_n.mean
epochs.0.full_horizon_validation.final_free_force_residual_norm_n.median
epochs.0.full_horizon_validation.final_free_force_residual_norm_n.max
epochs.0.full_horizon_validation.final_energy_joule.mean
epochs.0.full_horizon_validation.final_max_penetration_r.mean epochs.0.full_horizon_validation.final_max_penetration_r.max
epochs.0.full_horizon_validation.selection.metric epochs.0.full_horizon_validation.selection.eligible
epochs.0.full_horizon_validation.samples
best_selection.metric best_selection.epoch best_selection.source best_selection.iterations
best_selection.physical_steps best_selection.final_energy_joule.mean
best_selection.final_max_penetration_r.mean best_selection.final_max_penetration_r.max
best_selection_history.0.reset_at_epoch best_selection_history.0.reason
best_selection_history.0.record.metric best_selection_history.0.record.epoch
updates.0.update updates.0.epoch updates.0.loss updates.0.gradient_norm updates.0.step_size_mean
""".split()


def _lookup(obj, path):
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            if part not in obj:
                raise KeyError(path)
            obj = obj[part]
    return obj


def cheap_samples(n=4, K=3, failed_seed=None):
    return [
        {
            "seed": i,
            "residual_n": [10.0 / (k + 1) for k in range(K + 1)],
            "energy_joule": [-1.0 - 0.1 * k for k in range(K + 1)],
            "penetration_r": [0.1 * (k + 1) for k in range(K + 1)],
            "inverted_cells": [0] * (K + 1),
            "survived": i != failed_seed,
            "failed_first_update": i == failed_seed,
            "scale_joule": 2.0,
            "contact": True,
        }
        for i in range(n)
    ]


def full_samples(n=3, H=4, failed_seed=None):
    return [
        {
            "seed": 100 + i,
            "physical_records": [
                {"residual_n": 2.0 + i, "energy_joule": -0.5, "penetration_r": 0.2 * (i + 1), "inverted_cells": 0}
                for _ in range(H)
            ],
            "survived": 100 + i != failed_seed,
        }
        for i in range(n)
    ]


def epoch_record(epoch=1, loss=0.5, failed_seed=None):
    cheap = R.summarize_cheap_validation(cheap_samples(failed_seed=failed_seed), iterations=3)
    full = R.summarize_full_horizon(full_samples(), iterations=8, physical_steps=4, seconds=12.5)
    record = R.build_epoch_record(
        epoch=epoch,
        loss=loss,
        query_count=128,
        updates=8,
        seconds=61.0,
        lr=1e-4,
        grad_norm_mean=0.3,
        grad_norm_max=0.9,
        step_mean=0.02,
        step_min=0.01,
        step_max=0.05,
        tie_cell_count=3,
        mean_force_residual_n=4.5,
        resets=2,
        failures=[FailureRecord(seed=7, K=1, H=8, k=0, h=3, kind="nan", epoch=epoch, update=5)],
        regime={
            "stage": 0,
            "k_max": 1,
            "h_max": 8,
            "queries": 2048,
            "filler_queries": [0, 1, 0, 0],
            "updates": 32,
            "step_cap": 0.01,
            "pinned_fraction": 1.0,
        },
        available_K=[1],
        available_H=[8],
        contact_scene_fraction=0.8,
        contact_realized_fraction=0.6,
        contact_max_penetration_r=0.3,
        material_histograms={"E": [1, 2, 3]},
        validation=cheap,
        full_horizon_validation=full,
        rank_diagnostics=[{"rank": 0, "peak_cuda_bytes": 1}],
    )
    return record, cheap, full


class TestRunReport(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run_dir = self.root / "run"
        self.cfg = TrainConfig(updates_history_limit=5, max_epochs=10)
        self.rr = R.RunReport(self.run_dir, self.cfg, world_size=4, git_sha="abc123")

    def one_epoch(self, loss=0.5):
        self.rr.set_parameter_count(1234)
        self.rr.set_status("running", "training", epoch=1, regime={"stage": 0, "k_max": 1, "h_max": 8})
        self.rr.log_update(
            epoch=1, update=1, loss=0.7, grad_norm=0.4, lr=1e-4, step_mean=0.02, active=60, resets=1, seconds=0.5
        )
        record, cheap, full = epoch_record(loss=loss)
        self.assertTrue(R.selection_better(full, self.rr.report["best_selection"]))
        self.rr.set_best_selection(full, epoch=1)
        self.rr.log_epoch(record)
        # A budget change resets the selection and records the superseded record.
        self.rr.set_best_selection(full, epoch=2, reset_reason="full-horizon budget changed")
        self.rr.write()
        return record, cheap, full

    def test_every_dashboard_key_path_is_present(self):
        self.one_epoch()
        report = json.loads((self.run_dir / "report.json").read_text())
        for path in REPORT_PATHS:
            _lookup(report, path)  # raises KeyError when absent
        self.assertEqual(report["epochs"][0]["regime"]["name"], "fixed_states")
        self.assertEqual(report["completed_epochs"], 1)
        self.assertEqual(report["best_selection"]["epoch"], 2)
        self.assertEqual(report["best_selection"]["source"], self.cfg.selection_source)
        self.assertEqual(report["best_selection_history"][-1]["record"]["epoch"], 1)
        self.assertEqual(report["parameter_count"], 1234)
        self.assertEqual(report["world_size"], 4)
        self.assertEqual(report["config"]["max_epochs"], 10)

    def test_epochs_csv_has_the_25_columns_in_order(self):
        self.one_epoch()
        with open(self.run_dir / "epochs.csv", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], list(R.EPOCH_COLUMNS))
        self.assertEqual(len(R.EPOCH_COLUMNS), 25)
        self.assertEqual(len(rows), 2)
        row = dict(zip(rows[0], rows[1], strict=True))
        self.assertEqual(row["epoch"], "1")
        self.assertEqual(row["selection_eligible"], "True")
        self.assertEqual(row["regime_k_max"], "1")
        self.assertEqual(row["full_horizon_selection_metric"], "3.0")

    def test_progress_has_the_11_keys(self):
        self.one_epoch()
        progress = json.loads((self.run_dir / "progress.json").read_text())
        self.assertEqual(list(progress), list(R.PROGRESS_KEYS))
        self.assertEqual(len(progress), 11)
        self.assertEqual(progress["status"], "running")
        self.assertEqual(progress["phase"], "training")
        self.assertEqual(progress["max_epochs"], 10)
        self.assertEqual(progress["completed_epochs"], 1)
        self.assertEqual(progress["completed_updates"], 1)
        self.assertEqual(progress["latest_batch_loss"], 0.7)
        self.assertEqual(progress["available_K"], [1])
        self.assertEqual(progress["available_H"], [8])
        self.assertEqual(progress["regime"]["name"], "fixed_states")

    def test_updates_history_is_bounded(self):
        for i in range(1, 9):
            self.rr.log_update(
                epoch=1,
                update=i,
                loss=0.1 * i,
                grad_norm=0.1,
                lr=1e-4,
                step_mean=0.01,
                active=64,
                resets=0,
                seconds=0.1,
            )
        updates = self.rr.report["updates"]
        self.assertEqual(len(updates), 5)
        self.assertEqual([u["update"] for u in updates], [4, 5, 6, 7, 8])
        self.assertEqual(self.rr.report["completed_updates"], 8)
        self.assertEqual(sorted(os.listdir(self.run_dir)), ["progress.json"])

    def test_atomic_writes_leave_no_temporary_files(self):
        self.one_epoch()
        self.assertEqual(sorted(os.listdir(self.run_dir)), ["epochs.csv", "progress.json", "report.json"])
        self.rr.write_failure({"error": "boom", "record": FailureRecord(1, 1, 8, 0, 0, "nan", 1, 1)})
        self.rr.set_status("failed")
        self.assertEqual(
            sorted(os.listdir(self.run_dir)), ["epochs.csv", "failure.json", "progress.json", "report.json"]
        )
        failure = json.loads((self.run_dir / "failure.json").read_text())
        self.assertEqual(failure["error"], "boom")
        self.assertEqual(failure["record"]["kind"], "nan")

    def test_nan_and_tensors_become_plain_json(self):
        self.rr.log_update(
            epoch=torch.tensor(1),
            update=np.int64(1),
            loss=torch.tensor(float("nan")),
            grad_norm=np.float32(math.inf),
            lr=torch.tensor(1e-4),
            step_mean=np.float64(0.02),
            active=torch.tensor(3),
            resets=0,
            seconds=0.1,
        )
        row = self.rr.report["updates"][-1]
        self.assertIsNone(row["loss"])
        self.assertIsNone(row["gradient_norm"])
        self.assertIs(type(row["update"]), int)
        self.assertIs(type(row["learning_rate"]), float)
        self.assertIs(type(row["active"]), int)
        self.one_epoch(loss=float("nan"))
        text = (self.run_dir / "report.json").read_text()
        self.assertIn('"loss": null', text)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertIsNone(json.loads(text)["epochs"][0]["loss"])
        with self.assertRaises(ValueError):
            self.rr.set_status("exploded")

    def test_old_renderer_and_publisher_accept_the_run_dir(self):
        # Fresh run: no epochs, best_selection None, initialized_from None.
        self.rr.set_status("initializing")
        write_mixed_report(self.root / "public0", json.loads((self.run_dir / "report.json").read_text()))
        self.assertTrue((self.root / "public0" / "index.html").is_file())
        # One epoch, with a second validation-less epoch (gaps are allowed) and a failure file.
        self.one_epoch()
        record, _, _ = epoch_record(epoch=2)
        record["validation"] = None
        record["full_horizon_validation"] = None
        self.rr.log_epoch(record)
        report = json.loads((self.run_dir / "report.json").read_text())
        report["progress"] = json.loads((self.run_dir / "progress.json").read_text())
        write_mixed_report(self.root / "public1", report)
        self.assertTrue((self.root / "public1" / "index.html").is_file())
        self.assertEqual((self.root / "public1" / "epochs.csv").read_text(), (self.run_dir / "epochs.csv").read_text())
        self.rr.write_failure({"error": "boom"})
        published = prepare_public_report(self.run_dir, self.root / "public2")
        self.assertEqual(published["status"], "failed")
        self.assertTrue((self.root / "public2" / "index.html").is_file())
        self.assertIn("Training failure", (self.root / "public2" / "index.html").read_text())

    def test_cli_prints_headline(self):
        self.one_epoch()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(R.main([str(self.run_dir)]), 0)
        text = out.getvalue()
        self.assertIn("status running", text)
        self.assertIn("1 / 10 epochs completed", text)
        self.assertIn("best selection: 3 N at epoch 2", text)


class TestSummaries(unittest.TestCase):
    def test_cheap_validation_summary(self):
        v = R.summarize_cheap_validation(cheap_samples(failed_seed=2), iterations=3)
        self.assertEqual(v["sample_count"], 4)
        self.assertEqual(v["physical_survivors"], 3)
        self.assertEqual(v["failed_count"], 1)
        self.assertEqual(v["first_update_failed_count"], 1)
        self.assertEqual(v["failures"], [2])
        self.assertFalse(v["selection"]["eligible"])
        self.assertAlmostEqual(v["selection"]["metric"], 2.5)
        self.assertAlmostEqual(v["descent_rate"], 1.0)
        self.assertAlmostEqual(v["mean_before_joule"], -1.0)
        self.assertAlmostEqual(v["mean_after_joule"], -1.3)
        self.assertAlmostEqual(v["mean_normalized_loss"], -1.3 / 2.0)
        self.assertEqual([r["iteration"] for r in v["relative_energy"]], [0, 1, 2, 3])
        self.assertAlmostEqual(v["relative_energy"][-1]["mean"], 1.3)
        self.assertEqual(v["relative_energy"][-1]["near_zero_count"], 0)
        self.assertAlmostEqual(v["force_residual"][-1]["max"], 2.5)
        self.assertAlmostEqual(v["penetration"][-1]["max"], 0.4)
        self.assertEqual(len(v["samples"]), 4)
        # Near-zero initial energy is excluded from the ratios but keeps its residual.
        samples = cheap_samples(n=2)
        samples[0]["energy_joule"] = [0.0, -0.1, -0.2, -0.3]
        v = R.summarize_cheap_validation(samples, iterations=3)
        self.assertEqual(v["relative_energy"][-1]["near_zero_count"], 1)
        self.assertAlmostEqual(v["relative_energy"][-1]["mean"], 1.3)
        self.assertAlmostEqual(v["force_residual"][0]["mean"], 10.0)
        self.assertTrue(v["selection"]["eligible"])
        self.assertEqual(R.summarize_cheap_validation([], iterations=1)["sample_count"], 0)

    def test_full_horizon_summary_and_selection(self):
        f = R.summarize_full_horizon(full_samples(), iterations=8, physical_steps=4, seconds=12.5)
        self.assertEqual(
            (f["iterations"], f["physical_steps"], f["sample_count"], f["physical_survivors"]), (8, 4, 3, 3)
        )
        self.assertAlmostEqual(f["final_free_force_residual_norm_n"]["mean"], 3.0)
        self.assertAlmostEqual(f["final_free_force_residual_norm_n"]["median"], 3.0)
        self.assertAlmostEqual(f["final_free_force_residual_norm_n"]["max"], 4.0)
        self.assertAlmostEqual(f["final_energy_joule"]["mean"], -0.5)
        self.assertAlmostEqual(f["final_max_penetration_r"]["max"], 0.6)
        self.assertTrue(f["selection"]["eligible"])
        self.assertAlmostEqual(f["selection"]["metric"], 3.0)
        g = R.summarize_full_horizon(full_samples(failed_seed=102), iterations=8, physical_steps=4, seconds=1.0)
        self.assertFalse(g["selection"]["eligible"])
        self.assertAlmostEqual(g["selection"]["metric"], 2.5)
        none = R.summarize_full_horizon([dict(s, survived=False) for s in full_samples()], 8, 4, 1.0)
        self.assertIsNone(none["selection"]["metric"])
        self.assertFalse(none["selection"]["eligible"])
        # Eligible first, then lower metric; non-finite candidates never win; anything beats no best.
        self.assertTrue(R.selection_better(f, None))
        self.assertTrue(R.selection_better(f, g))
        self.assertFalse(R.selection_better(g, f))
        self.assertTrue(R.selection_better({"metric": 2.0, "eligible": True}, {"metric": 3.0, "eligible": True}))
        self.assertFalse(R.selection_better({"metric": 3.0, "eligible": True}, {"metric": 2.0, "eligible": True}))
        self.assertFalse(R.selection_better(none, None))
        self.assertFalse(R.selection_better({"metric": float("nan"), "eligible": True}, None))


if __name__ == "__main__":
    unittest.main()
