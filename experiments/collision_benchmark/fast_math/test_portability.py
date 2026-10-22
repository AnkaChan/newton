# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check portable output handling and the immutable physics boundary."""

import hashlib
import importlib
import subprocess
import tempfile
import unittest
from pathlib import Path


class PortabilityTests(unittest.TestCase):
    """Exercise the guard against real temporary Git repositories."""

    def setUp(self):
        """Create a small repository with a pinned core file."""
        try:
            self.guard = importlib.import_module("experiments.collision_benchmark.fast_math.provenance")
        except ModuleNotFoundError:
            self.guard = None
        self.assertIsNotNone(self.guard, "The portable physics guard is not implemented")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.core = self.root / "newton/_src/solvers/example.py"
        self.core.parent.mkdir(parents=True)
        self.core.write_text("coefficient = 1\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture base")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.manifest = {
            "base_revision": self.base,
            "warp_version": "1.17.0",
            "files_sha256": {"newton/_src/solvers/example.py": hashlib.sha256(self.core.read_bytes()).hexdigest()},
        }

    def git(self, *args):
        """Run Git only in the isolated temporary test repository."""
        return subprocess.check_output(
            ["git", "-c", "user.name=Benchmark Test", "-c", "user.email=test@example.invalid", *args],
            cwd=self.root,
            text=True,
            stderr=subprocess.STDOUT,
        )

    def test_benchmark_only_child_commit_is_allowed(self):
        """Accept a child commit that leaves the frozen physics unchanged."""
        source = self.root / "experiments/benchmark.py"
        source.parent.mkdir()
        source.write_text("description = 'portable'\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture packaging")
        report = self.guard.verify_physics(self.root, self.manifest, warp_version="1.17.0")
        self.assertNotEqual(report["revision"], self.base)
        self.assertEqual(report["base_revision"], self.base)
        self.assertEqual(report["files_sha256"], self.manifest["files_sha256"])

    def test_modified_core_is_rejected_even_before_commit(self):
        """Reject a changed equation despite unchanged Git HEAD."""
        self.core.write_text("coefficient = 2\n")
        with self.assertRaisesRegex(ValueError, "physics.*changed"):
            self.guard.verify_physics(self.root, self.manifest, warp_version="1.17.0")

    def test_added_core_module_is_rejected(self):
        """Reject an extra Python module inside the frozen core."""
        self.core.with_name("extra.py").write_text("coefficient = 1\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.guard.verify_physics(self.root, self.manifest, warp_version="1.17.0")

    def test_missing_base_history_is_rejected(self):
        """Reject an unverified revision instead of bypassing the ancestry guard."""
        self.manifest["base_revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "ancestor"):
            self.guard.verify_physics(self.root, self.manifest, warp_version="1.17.0")

    def test_wrong_warp_version_is_rejected(self):
        """Reject a changed compiler dependency before creating the run."""
        with self.assertRaisesRegex(ValueError, "Warp"):
            self.guard.verify_physics(self.root, self.manifest, warp_version="2.0.0")

    def test_output_is_portable_but_existing_evidence_is_preserved(self):
        """Allow outputs outside the checkout and reject existing run data."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            self.guard.validate_output(output)
            output.mkdir()
            (output / "progress.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                self.guard.validate_output(output)


if __name__ == "__main__":
    unittest.main()
