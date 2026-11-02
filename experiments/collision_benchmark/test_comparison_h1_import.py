# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check explicit historical H1 initialization imports without a simulation."""

import hashlib
import json
import tempfile
import types
import unittest
from enum import IntEnum
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.collision_benchmark import comparison_h1 as h1


class HistoricalH1ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.prepare = {"state0.particle_q": np.zeros((2, 3), dtype=np.float32)}
        self.start = {**self.prepare, "driver0.fingers": np.ones((2, 12), dtype=np.float32)}
        payload = {f"prepare/{k}": v for k, v in self.prepare.items()}
        payload.update({f"start/{k}": v for k, v in self.start.items()})
        np.savez(self.folder / "common-start.npz", **payload)
        self.metadata = {
            "identity": {"case_id": "r7_h1_towel", "model_sha256": "model", "source_hashes": {}},
            "npz_sha256": hashlib.sha256((self.folder / "common-start.npz").read_bytes()).hexdigest(),
            "preparation_sha256": h1._digest_arrays(self.prepare),
            "state_sha256": h1._digest_arrays(self.start),
            "command_sha256": h1._digest_arrays({"driver0.fingers": self.start["driver0.fingers"]}),
            "python_fields": {},
        }
        self.write_metadata()

    def write_metadata(self):
        (self.folder / "common-start.json").write_text(json.dumps(self.metadata))

    def test_load_preserves_source_arrays_and_hashes(self):
        """Load exact initialized arrays while retaining source identity."""
        result = h1._load_initial_import(self.folder, "r7_h1_towel")
        self.assertEqual(result["provenance"]["npz_sha256"], self.metadata["npz_sha256"])
        np.testing.assert_array_equal(result["arrays"]["start"]["driver0.fingers"], self.start["driver0.fingers"])

    def test_reject_wrong_case(self):
        """Reject a cache belonging to another material and robot scene."""
        with self.assertRaisesRegex(ValueError, "case"):
            h1._load_initial_import(self.folder, "j3_h1_jacket")

    def test_reject_corrupt_npz(self):
        """Reject changed storage bytes before loading state arrays."""
        with (self.folder / "common-start.npz").open("ab") as output:
            output.write(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            h1._load_initial_import(self.folder, "r7_h1_towel")

    def test_reject_changed_command_digest(self):
        """Reject an internally inconsistent command fingerprint."""
        self.metadata["command_sha256"] = "wrong"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "command"):
            h1._load_initial_import(self.folder, "r7_h1_towel")

    def test_reject_changed_body_layout_before_restore(self):
        """Reject model correspondence changes before importing any state."""
        imported = h1._load_initial_import(self.folder, "r7_h1_towel")
        with mock.patch.object(h1, "_model_arrays", return_value={"shape_body": np.array([1, 0])}):
            with self.assertRaisesRegex(ValueError, "geometry/body-layout"):
                h1._validate_import_model(object(), imported)

    def test_reject_new_or_missing_state_field(self):
        """Reject array-layout drift rather than partially restoring old state."""
        with mock.patch.object(h1, "_array_slots", return_value={"new_state_array": object()}):
            with self.assertRaisesRegex(ValueError, "different state/driver array fields"):
                h1._restore(object(), self.start, drivers=True)

    def test_nested_solver_options_survive_cache_replay(self):
        """Keep enum-keyed collision schedules identical after JSON replay."""

        class Slot(IntEnum):
            RIGID = 0

        scene = types.SimpleNamespace(
            model=types.SimpleNamespace(
                soft_contact_ke=18000.0,
                collision_benchmark_contact_policy={"policy": {"name": "authored", "version": 1}},
            ),
            comparison_solver_kwargs={"collision_frequency_type": {Slot.RIGID: 2}},
            comparison_source_hashes={},
        )
        with (
            mock.patch.object(h1, "_model_arrays", return_value={}),
            mock.patch.object(h1.runtime, "main_dat_configuration", return_value=None, create=True),
        ):
            fresh_identity = h1._cache_identity("r7_h1_towel", scene)
            replayed_identity = json.loads(json.dumps(fresh_identity))
            self.assertEqual(fresh_identity, replayed_identity)


if __name__ == "__main__":
    unittest.main()
