# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check portable registry and replay inputs without constructing a simulation."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.collision_benchmark.comparison_cases import load_cases
from experiments.collision_benchmark.recipes.update_manifest import semantic_digest

PACKAGE = Path(__file__).resolve().parent
H1_CASES = {"j3_h1_jacket", "r7_h1_sheet", "r7_h1_towel", "r7_h1_gusset_bag"}


class TestComparisonPortability(unittest.TestCase):
    def test_registry_and_recipe_paths_from_another_directory(self):
        registry = PACKAGE / "cases.json"
        self.assertTrue(registry.is_file(), "The default registry must be included in the checkout")
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
                os.environ.pop("NEWTON_COMPARISON_CASES", None)
                os.chdir(directory)
                cases = load_cases()
                self.assertEqual(len(cases), 44)
                self.assertEqual({k for k, v in cases.items() if v["runner"] == "h1"}, H1_CASES)
                for case in cases.values():
                    self.assertLessEqual(case["substeps"], 10)
                    for key, value in case["env"].items():
                        if key.endswith("_RECIPE"):
                            self.assertTrue(Path(value).is_file(), value)
                    for record in case["replay"].values():
                        path = Path(record["path"])
                        self.assertTrue(path.is_file(), str(path))
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
        finally:
            os.chdir(previous)

    def test_h1_recipe_source_closure(self):
        manifest_path = PACKAGE / "recipes" / "h1" / "manifest.json"
        self.assertTrue(manifest_path.is_file(), "All four H1 replay inputs must be packaged")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(set(manifest["cases"]), H1_CASES)
        for case_id, record in manifest["cases"].items():
            self.assertTrue(record["files"])
            for relative, digest in record["files"].items():
                path = manifest_path.parent / case_id / relative
                self.assertTrue(path.is_file(), str(path))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                self.assertEqual(semantic_digest(path), record["original_semantics_sha256"][relative])
        self.assertNotIn("/home/", manifest_path.read_text())


if __name__ == "__main__":
    unittest.main()
