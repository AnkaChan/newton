# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the authenticated multiplicative MG-VBD CPU scene suite."""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from .. import bench_mg_vbd as benchmark
from ..solver_scenes import build_stretch_scene


class TestMGVBDSceneSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls.temporary_directory.name)
        cls.output_path = root / "tiny-suite.json"
        cls.report_path = root / "tiny-suite.md"
        cls.scene = build_stretch_scene(dimensions=(2, 1, 2))
        cls.results = []
        original = benchmark.run_multiplicative_mg_vbd

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            cls.results.append(result)
            return result

        with (
            mock.patch.dict(benchmark._SCENE_FACTORIES, {"stretch": lambda: cls.scene}, clear=True),
            mock.patch.object(benchmark, "run_multiplicative_mg_vbd", side_effect=capture) as solver,
        ):
            benchmark.run_scene_suite(
                benchmark.MGVBDSuiteRunConfig(
                    output_path=cls.output_path,
                    report_path=cls.report_path,
                    scene_keys=("stretch",),
                )
            )
        cls.solver_call = solver.call_args
        cls.payload = benchmark.verify_suite_file(cls.output_path, verify_current_sources=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_real_tiny_scene_binds_fresh_runs_exact_work_and_diagnostic_timing(self):
        self.assertEqual(len(self.results), 1)
        self.assertEqual(self.solver_call.kwargs["config"].deterministic_record(), benchmark._fixed_config_record())
        self.assertEqual(
            {
                name: self.solver_call.kwargs[name]
                for name in ("vbd_warmup", "vbd_repeats", "newton_warmup", "newton_repeats")
            },
            {"vbd_warmup": False, "vbd_repeats": 1, "newton_warmup": False, "newton_repeats": 1},
        )

        payload = self.payload
        entry = payload["scenes"][0]
        quality = entry["quality"]
        summary = entry["summary"]
        self.assertEqual(payload["status"], "complete")
        self.assertTrue(payload["summary"]["all_requested_gates_passed"])
        self.assertEqual(quality["reference"]["provenance"], "fresh-dense-newton")
        self.assertEqual(quality["vbd_k1"]["execution"], "fresh-scalar-cpu-run_vbd")
        self.assertEqual(quality["vbd_k4"]["execution"], "fresh-scalar-cpu-run_vbd")
        self.assertEqual(quality["vbd_k1"]["iterations"], 1)
        self.assertEqual(quality["vbd_k4"]["iterations"], 4)
        self.assertEqual(summary["work"]["total_preconditioner_applications"], 12)
        self.assertEqual(summary["work"]["total_v_cycle_records"], 12)
        self.assertEqual([item["v_cycle_records"] for item in summary["work"]["per_outer"]], [4, 4, 4])
        self.assertTrue(summary["safeguards"]["exact_pins"])
        self.assertTrue(summary["safeguards"]["inversion_free"])
        self.assertFalse(summary["safeguards"]["fallback_used"])
        self.assertEqual(entry["diagnostic_timing"]["performance_evidence"], False)
        self.assertEqual(
            entry["diagnostic_timing"]["integration_timing"]["performance_evidence"],
            False,
        )
        self.assertNotIn("timing", quality)
        self.assertIn(payload["suite_sha256"], self.report_path.read_text())
        self.assertIn("development-only", self.report_path.read_text())

    def test_complete_resume_is_verify_only_and_no_overwrite_is_default(self):
        before = self.output_path.read_bytes()
        with (
            mock.patch.dict(benchmark._SCENE_FACTORIES, {"stretch": lambda: self.scene}, clear=True),
            mock.patch.object(benchmark, "run_multiplicative_mg_vbd") as solver,
        ):
            resumed = benchmark.run_scene_suite(
                benchmark.MGVBDSuiteRunConfig(
                    output_path=self.output_path,
                    report_path=self.report_path,
                    scene_keys=("stretch",),
                    resume=True,
                )
            )
        solver.assert_not_called()
        self.assertEqual(resumed, self.output_path.resolve())
        self.assertEqual(self.output_path.read_bytes(), before)

        with (
            mock.patch.dict(benchmark._SCENE_FACTORIES, {"stretch": lambda: self.scene}, clear=True),
            self.assertRaisesRegex(FileExistsError, "refusing to overwrite"),
        ):
            benchmark.run_scene_suite(
                benchmark.MGVBDSuiteRunConfig(output_path=self.output_path, scene_keys=("stretch",))
            )

    def test_semantic_tamper_is_rejected_even_after_rehashing_every_container(self):
        payload = copy.deepcopy(self.payload)
        entry = payload["scenes"][0]
        quality = entry["quality"]
        quality["gate"]["versus_k4"]["residual_ratio"] *= 2.0
        quality.pop("quality_sha256")
        quality["quality_sha256"] = benchmark._canonical_sha256(quality)

        timing = entry["diagnostic_timing"]["integration_timing"]
        timing["quality_sha256"] = quality["quality_sha256"]
        timing.pop("timing_sha256")
        timing["timing_sha256"] = benchmark._canonical_sha256(timing)
        entry["diagnostic_timing"].pop("diagnostic_sha256")
        entry["diagnostic_timing"]["diagnostic_sha256"] = benchmark._canonical_sha256(entry["diagnostic_timing"])
        entry["summary"] = benchmark._summary_from_quality(quality)
        entry.pop("entry_sha256")
        entry["entry_sha256"] = benchmark._canonical_sha256(entry)
        payload = benchmark._seal_suite(payload)

        with self.assertRaisesRegex(ValueError, "ratios do not match raw common metrics"):
            benchmark.verify_suite_payload(payload)

    def test_partial_checkpoint_resumes_ordered_prefix_without_repeating_first_scene(self):
        root = pathlib.Path(self.temporary_directory.name)
        output = root / "resume-suite.json"
        factories = {"stretch": lambda: self.scene, "twist": lambda: self.scene}
        with (
            mock.patch.dict(benchmark._SCENE_FACTORIES, factories, clear=True),
            mock.patch.object(
                benchmark,
                "run_multiplicative_mg_vbd",
                side_effect=(self.results[0], KeyboardInterrupt()),
            ) as solver,
            self.assertRaises(KeyboardInterrupt),
        ):
            benchmark.run_scene_suite(
                benchmark.MGVBDSuiteRunConfig(output_path=output, scene_keys=("stretch", "twist"))
            )
        self.assertEqual(solver.call_count, 2)
        partial = benchmark.verify_suite_file(output)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([entry["key"] for entry in partial["scenes"]], ["stretch"])

        with (
            mock.patch.dict(benchmark._SCENE_FACTORIES, factories, clear=True),
            mock.patch.object(benchmark, "run_multiplicative_mg_vbd", return_value=self.results[0]) as solver,
        ):
            benchmark.run_scene_suite(
                benchmark.MGVBDSuiteRunConfig(
                    output_path=output,
                    scene_keys=("stretch", "twist"),
                    resume=True,
                )
            )
        solver.assert_called_once()
        complete = benchmark.verify_suite_file(output)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual([entry["key"] for entry in complete["scenes"]], ["stretch", "twist"])

    def test_execution_failure_is_retained_as_negative_row(self):
        output = pathlib.Path(self.temporary_directory.name) / "failure-suite.json"
        with (
            mock.patch.dict(benchmark._SCENE_FACTORIES, {"stretch": lambda: self.scene}, clear=True),
            mock.patch.object(
                benchmark,
                "run_multiplicative_mg_vbd",
                side_effect=ValueError("reference gate failed"),
            ),
        ):
            benchmark.run_scene_suite(benchmark.MGVBDSuiteRunConfig(output_path=output, scene_keys=("stretch",)))
        payload = benchmark.verify_suite_file(output)
        entry = payload["scenes"][0]
        self.assertEqual(entry["status"], "execution-failed")
        self.assertFalse(entry["summary"]["gate_passed"])
        self.assertEqual(entry["summary"]["execution_failure"]["exception_type"], "ValueError")
        self.assertEqual(payload["summary"]["execution_failed_scene_keys"], ["stretch"])
        self.assertFalse(payload["summary"]["all_requested_gates_passed"])

    def test_ceiling_fails_before_solver_and_output_creation(self):
        output = pathlib.Path(self.temporary_directory.name) / "ceiling-suite.json"
        with (
            mock.patch.dict(benchmark._SCENE_FACTORIES, {"stretch": lambda: self.scene}, clear=True),
            mock.patch.object(benchmark, "run_multiplicative_mg_vbd") as solver,
            self.assertRaisesRegex(ValueError, "free-DOF ceiling 17 exceeded"),
        ):
            benchmark.run_scene_suite(
                benchmark.MGVBDSuiteRunConfig(
                    output_path=output,
                    scene_keys=("stretch",),
                    max_newton_free_dofs=17,
                )
            )
        solver.assert_not_called()
        self.assertFalse(output.exists())

    def test_json_round_trip_and_source_pins_are_content_addressed(self):
        decoded = json.loads(self.output_path.read_text())
        self.assertEqual(decoded, self.payload)
        sources = decoded["source_manifest"]["files"]
        for filename, expected in benchmark._PINNED_SOURCE_SHA256.items():
            self.assertEqual(sources[filename]["sha256"], expected)
            self.assertEqual(sources[filename]["pinned_sha256"], expected)
            self.assertTrue(sources[filename]["reviewed"])


if __name__ == "__main__":
    unittest.main()
