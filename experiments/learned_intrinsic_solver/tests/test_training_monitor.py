# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the experimental training watchdog's failure and delivery decisions."""

import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestTrainingMonitor(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("experiments.learned_intrinsic_solver.training_monitor")
        self.assertIsNotNone(spec, "The persistent training watchdog must be implemented")
        self.monitor = importlib.import_module("experiments.learned_intrinsic_solver.training_monitor")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.now = 10_000.0
        self.progress = {"status": "running", "phase": "training", "completed_updates": 64, "epoch": 1}
        self._write("progress.json", self.progress)
        self._write("report.json", {"status": "running", "epochs": []})
        (self.run / "checkpoints").mkdir()
        (self.run / "checkpoints/latest.pt").write_bytes(b"checkpoint")
        os.utime(self.run / "checkpoints/latest.pt", (self.now, self.now))
        self.workers = [{"pid": i + 1, "rank": i, "state": "S"} for i in range(4)]

    def _write(self, name, value):
        path = self.run / name
        path.write_text(json.dumps(value))
        os.utime(path, (self.now, self.now))

    def _check(self, **kwargs):
        return self.monitor.check_training_health(
            self.run, workers=kwargs.pop("workers", self.workers), dashboard={"ok": True}, now=self.now, **kwargs
        )

    def test_failed_run_remains_visible_after_all_workers_exit(self):
        """Raise an incident for the saved error even when the tmux pane survives."""
        self._write("failure.json", {"error": "trajectory 9238 invalid", "completed_updates": 5502})
        health = self._check(workers=[])
        self.assertEqual(health["health"], "attention")
        self.assertTrue(any("trajectory 9238 invalid" in issue for issue in health["issues"]))
        self.assertTrue(any("0/4" in issue for issue in health["issues"]))

    def test_worker_count_requires_unique_live_ranks(self):
        """Reject duplicate and stopped ranks instead of accepting four matching PIDs."""
        workers = [{"pid": i + 1, "rank": i % 2, "state": "T" if i == 3 else "S"} for i in range(4)]
        health = self._check(workers=workers)
        self.assertEqual(health["health"], "attention")
        self.assertTrue(any("rank" in issue for issue in health["issues"]))

    def test_shell_launcher_is_not_counted_as_an_additional_worker(self):
        """Count the Python trainer only when its shell parent forwards identical arguments."""
        proc = self.run / "proc"
        for pid, executable, prefix in ((100, "bash", ["-c", "run child"]), (101, "/venv/bin/python", ["-u"])):
            directory = proc / str(pid)
            directory.mkdir(parents=True)
            argv = [
                executable,
                *prefix,
                "-m",
                "experiments.learned_intrinsic_solver.train_mixed",
                "--output",
                str(self.run),
            ]
            (directory / "cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in argv) + b"\0")
            (directory / "environ").write_bytes(b"RANK=0\0")
            (directory / "status").write_text("State:\tS (sleeping)\n")
        with mock.patch.object(self.monitor, "_PROC", proc, create=True):
            workers = self.monitor._workers(self.run)
        self.assertEqual(workers, [{"pid": 101, "rank": 0, "state": "S"}])

    def test_epoch_limit_is_a_completed_campaign(self):
        """Avoid raising a missing-worker failure after the configured epoch limit."""
        self.progress["status"] = "epoch_limit"
        self._write("progress.json", self.progress)
        self.assertEqual(self._check(workers=[])["health"], "completed")

    def test_long_validation_is_not_immediately_a_stall(self):
        """Allow a twenty-minute validation heartbeat under a forty-five-minute threshold."""
        self.progress["phase"] = "validation"
        self._write("progress.json", self.progress)
        os.utime(self.run / "progress.json", (self.now - 1200, self.now - 1200))
        self.assertEqual(self._check()["health"], "healthy")

    def test_stale_progress_and_checkpoint_request_investigation(self):
        """Flag forty-six-minute inactivity without changing any training files."""
        os.utime(self.run / "progress.json", (self.now - 2760, self.now - 2760))
        os.utime(self.run / "checkpoints/latest.pt", (self.now - 2760, self.now - 2760))
        before = (self.run / "progress.json").read_bytes()
        health = self._check()
        self.assertEqual(health["health"], "attention")
        self.assertTrue(any("progress" in issue for issue in health["issues"]))
        self.assertTrue(any("checkpoint" in issue for issue in health["issues"]))
        self.assertEqual((self.run / "progress.json").read_bytes(), before)

    def test_paused_training_is_not_an_unexpected_worker_failure(self):
        """Respect a reported pause without triggering a restart investigation."""
        self.progress["status"] = "paused"
        self._write("progress.json", self.progress)
        health = self._check(workers=[])
        self.assertEqual(health["health"], "paused")
        self.assertFalse(health["issues"])

    def test_checkpoint_age_adapts_to_observed_epoch_duration(self):
        """Avoid stale-checkpoint alarms during a known forty-minute epoch."""
        self._write("report.json", {"status": "running", "epochs": [{"seconds": 2400}]})
        os.utime(self.run / "checkpoints/latest.pt", (self.now - 3000, self.now - 3000))
        self.assertEqual(self._check()["health"], "healthy")

    def test_corrupt_progress_is_reported_instead_of_crashing_watchdog(self):
        """Turn malformed training metadata into a recorded incident."""
        (self.run / "progress.json").write_text("{")
        health = self._check()
        self.assertEqual(health["health"], "attention")
        self.assertTrue(any("progress.json" in issue for issue in health["issues"]))

    def test_queue_dedup_is_thread_scoped_and_read_only(self):
        """Suppress only this monitor's pending messages for the selected thread."""
        database = self.run / "queue.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE queued_items (thread_id TEXT, payload_json TEXT)")
            connection.executemany(
                "INSERT INTO queued_items VALUES (?, ?)",
                [("other", '{"text":"[LIDO-v2 monitor] healthy"}'), ("root", '{"text":"user question"}')],
            )
        self.assertFalse(self.monitor.has_pending_monitor_message(database, "root"))
        with sqlite3.connect(database) as connection:
            connection.execute(
                "INSERT INTO queued_items VALUES (?, ?)", ("root", '{"text":"[LIDO-v2 monitor] attention"}')
            )
        self.assertTrue(self.monitor.has_pending_monitor_message(database, "root"))
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM queued_items").fetchone()[0], 3)


if __name__ == "__main__":
    unittest.main()
