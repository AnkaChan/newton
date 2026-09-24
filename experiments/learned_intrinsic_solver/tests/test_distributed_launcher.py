# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise distributed launcher cleanup with actual short CPU processes."""

import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class TestDistributedLauncher(unittest.TestCase):
    def _runner(self):
        """Load the process runner after asserting that the implementation exists."""
        name = "experiments.learned_intrinsic_solver.launch_distributed_probe"
        self.assertIsNotNone(importlib.util.find_spec(name), "distributed launcher must exist")
        return importlib.import_module(name)._run_workers

    def test_waits_for_all_successful_workers(self):
        """Capture every worker's output and wait for delayed successful completion."""
        runner = self._runner()
        with tempfile.TemporaryDirectory() as temporary:
            result = runner(
                [
                    [sys.executable, "-c", "print('rank zero')"],
                    [sys.executable, "-c", "import time; time.sleep(0.2); print('rank one')"],
                ],
                Path(temporary),
                timeout=5,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["exit_codes"], [0, 0])
            self.assertIn("rank zero", (Path(temporary) / "rank_0.log").read_text())
            self.assertIn("rank one", (Path(temporary) / "rank_1.log").read_text())

    def test_worker_failure_terminates_other_ranks(self):
        """Abort a waiting peer when another rank fails instead of leaving it alive."""
        runner = self._runner()
        with tempfile.TemporaryDirectory() as temporary:
            start = time.monotonic()
            result = runner(
                [
                    [sys.executable, "-c", "import time; time.sleep(0.2); raise SystemExit(7)"],
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                ],
                Path(temporary),
                timeout=5,
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["exit_codes"][0], 7)
            self.assertIsNotNone(result["exit_codes"][1])
            self.assertLess(time.monotonic() - start, 4)

    def test_timeout_terminates_every_rank(self):
        """Stop and reap processes that exceed the collective run's time limit."""
        runner = self._runner()
        with tempfile.TemporaryDirectory() as temporary:
            result = runner(
                [[sys.executable, "-c", "import time; time.sleep(30)"] for _ in range(2)],
                Path(temporary),
                timeout=0.2,
            )
            self.assertFalse(result["passed"])
            self.assertTrue(result["timed_out"])
            self.assertTrue(all(code is not None for code in result["exit_codes"]))

    def test_existing_output_is_preserved(self):
        """Reject an existing run directory before claims or workers can alter it."""
        self._runner()
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "report.json"
            marker.write_text(json.dumps({"preserve": True}))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.learned_intrinsic_solver.launch_distributed_probe",
                    "--output",
                    temporary,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(marker.read_text()), {"preserve": True})
            self.assertEqual(list(Path(temporary).iterdir()), [marker])

    def test_timeout_stops_descendant_that_ignores_termination(self):
        """Kill a surviving worker child even when its parent exited on SIGTERM."""
        runner = self._runner()
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "child.pid"
            child = (
                "import os, signal, sys, time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
            )
            parent = (
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); time.sleep(30)"
            )
            try:
                result = runner([[sys.executable, "-c", parent, child, str(pid_file)]], Path(temporary), timeout=0.5)
                self.assertTrue(result["timed_out"])
                pid = int(pid_file.read_text())
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    status = Path(f"/proc/{pid}/stat")
                    if not status.exists() or status.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("worker descendant survived launcher timeout cleanup")
            finally:
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_forwarded_output_cannot_bypass_fresh_directory_check(self):
        """Reject a second output argument before creating directories or claiming GPUs."""
        self._runner()
        module = importlib.import_module("experiments.learned_intrinsic_solver.launch_distributed_probe")
        with tempfile.TemporaryDirectory() as temporary:
            fresh = Path(temporary) / "fresh"
            for arguments in (("--output", temporary), (f"--output={temporary}",)):
                with self.assertRaisesRegex(ValueError, "output"):
                    module.launch_probe(fresh, probe_arguments=arguments)
                self.assertFalse(fresh.exists())


if __name__ == "__main__":
    unittest.main()
