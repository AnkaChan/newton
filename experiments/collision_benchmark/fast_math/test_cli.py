# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that the standalone command preserves runner controls."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from experiments.collision_benchmark import comparison_h1 as h1
from experiments.collision_benchmark import comparison_runtime as runtime
from experiments.collision_benchmark.contact_stiffness import apply_contact_policy
from experiments.collision_benchmark.run_comparison import parse_arguments as parse_runner

from .run_fast_math import main, parse_arguments, runner_arguments


class CommandTests(unittest.TestCase):
    """Verify explicit controls, safe defaults, and packaged help."""

    def test_forward_explicit_controls_without_treatment_leakage(self):
        """Keep mode and schedule flags while consuming only adapter options."""
        args = parse_arguments(
            [
                "s1_pads_2_10substeps",
                "--mode",
                "alm",
                "--math",
                "on",
                "--out",
                "outputs/on",
                "--initial-reference",
                "outputs/off/results.json",
                "--device",
                "cpu",
                "--hold-seconds",
                "10",
                "--duration",
                "0.2",
                "--no-video",
                "--video-fps",
                "15",
                "--width",
                "640",
                "--height",
                "480",
            ]
        )
        self.assertEqual(args.initial_reference, Path("outputs/off/results.json"))
        self.assertEqual(args.math, "on")
        self.assertEqual(
            runner_arguments(args),
            [
                "s1_pads_2_10substeps",
                "--mode",
                "alm",
                "--out",
                "outputs/on",
                "--device",
                "cpu",
                "--hold-seconds",
                "10.0",
                "--video-fps",
                "15.0",
                "--width",
                "640",
                "--height",
                "480",
                "--duration",
                "0.2",
                "--no-video",
            ],
        )

    def test_first_run_does_not_require_historical_outputs(self):
        """Allow a standalone first run and preserve the original hold default."""
        args = parse_arguments(["s1_pads_2_10substeps", "--mode", "vbd", "--math", "off", "--out", "outputs/off"])
        self.assertIsNone(args.initial_reference)
        self.assertEqual(args.hold_seconds, 60.0)
        self.assertIsNone(args.duration)
        self.assertFalse(args.no_video)
        self.assertIsNone(args.contact_max_ratio)
        self.assertEqual(args.contact_tet_policy, "numeric")
        self.assertNotIn("--contact-max-ratio", runner_arguments(args))

    def test_forward_contact_policy_explicitly(self):
        """Forward the selected contact cap and tetrahedral interpretation."""
        for ratio, policy in ((10.0, "numeric"), (0.25, "mesh")):
            with self.subTest(ratio=ratio, policy=policy):
                args = parse_arguments(
                    [
                        "r7_h1_sheet",
                        "--mode",
                        "vbd",
                        "--math",
                        "off",
                        "--out",
                        "outputs/contact",
                        "--contact-max-ratio",
                        str(ratio),
                        "--contact-tet-policy",
                        policy,
                    ]
                )
                forwarded = runner_arguments(args)
                self.assertEqual(forwarded[forwarded.index("--contact-max-ratio") + 1], str(ratio))
                self.assertEqual(forwarded[forwarded.index("--contact-tet-policy") + 1], policy)

    def test_reject_invalid_contact_ratio_before_output_creation(self):
        """Reject nonfinite, nonpositive, and over-limit caps before side effects."""
        with tempfile.TemporaryDirectory() as temporary:
            for value in ("nan", "inf", "-inf", "0", "-1", "10.00001"):
                with self.subTest(value=value):
                    out = Path(temporary) / "must-not-exist"
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        main(
                            [
                                "r7_h1_sheet",
                                "--mode",
                                "vbd",
                                "--math",
                                "off",
                                "--out",
                                str(out),
                                "--contact-max-ratio=" + value,
                            ]
                        )
                    self.assertEqual(caught.exception.code, 2)
                    self.assertFalse(out.exists())

    def test_help_exits_before_environment_or_scene_initialization(self):
        """Describe both switches without requiring a case or old recordings."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            parse_arguments(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--mode", output.getvalue())
        self.assertIn("--math", output.getvalue())


class ContactIntegrationTests(unittest.TestCase):
    """Check material installation and cache identity without physics steps."""

    def test_forwarded_policy_is_accepted_by_main_runner(self):
        """Preserve the exact cap when the adapter invokes the main runner."""
        args = parse_arguments(
            [
                "r7_h1_sheet",
                "--mode",
                "vbd",
                "--math",
                "off",
                "--out",
                "outputs/cap",
                "--contact-max-ratio",
                "2.5",
                "--contact-tet-policy",
                "mesh",
            ]
        )
        forwarded = parse_runner(runner_arguments(args))
        self.assertEqual(forwarded.contact_max_ratio, 2.5)
        self.assertEqual(forwarded.contact_tet_policy, "mesh")
        for value in ("nan", "inf", "0", "-1", "10.1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_runner(["case", "--mode", "vbd", "--out", "unused", "--contact-max-ratio=" + value])

    def test_constructor_applies_cap_before_solver_and_keeps_authored_report(self):
        """Cap before solver allocation and preserve authored values on replacement."""
        model = SimpleNamespace(
            soft_contact_ke=50000.0,
            shape_material_ke=wp.array([50000.0, 1000.0], dtype=wp.float32, device="cpu"),
            tri_materials=wp.array([[1800.0, 1800.0, 0.1, 0.0, 0.0]], dtype=wp.float32, device="cpu"),
            device=SimpleNamespace(is_cuda=False),
        )
        constructor_values = []

        def initialize(solver, actual_model, **kwargs):
            solver.model = actual_model
            constructor_values.append((actual_model.soft_contact_ke, actual_model.shape_material_ke.numpy().copy()))

        policy = {"max_ratio": 3.0, "tet_policy": "numeric"}
        with (
            patch.object(runtime, "CONTACT_STIFFNESS_POLICY", policy),
            patch.object(runtime.OriginalSolverVBD, "__init__", initialize),
            patch.object(runtime, "MODE", False),
        ):
            first = runtime.ObservedSolverVBD(model, iterations=1)
            raw = SimpleNamespace(solver=first)
            second = runtime.replace_solver(raw, False)
        self.assertIs(first.comparison_contact_policy, second.comparison_contact_policy)
        self.assertEqual(first.comparison_contact_policy["authored"]["soft_contact_ke"], 50000.0)
        self.assertEqual(len(constructor_values), 2)
        for scalar, shapes in constructor_values:
            self.assertEqual(scalar, 5400.0)
            np.testing.assert_array_equal(shapes, [5400.0, 1000.0])

    def test_h1_cache_is_separated_by_contact_policy_and_adapter(self):
        """Prevent authored or differently capped preparation from sharing a cache."""
        case = "r7_h1_sheet"
        policies = [
            None,
            {"max_ratio": 10.0, "tet_policy": "numeric"},
            {"max_ratio": 3.0, "tet_policy": "numeric"},
            {"max_ratio": 10.0, "tet_policy": "mesh"},
        ]
        folders = [h1._cache_folder_name(case, policy) for policy in policies]
        self.assertEqual(len(set(folders)), len(policies))
        self.assertEqual(folders[1], h1._cache_folder_name(case, {"tet_policy": "numeric", "max_ratio": 10}))
        with patch.object(h1, "_sha", return_value="different-adapter"):
            self.assertNotEqual(folders[1], h1._cache_folder_name(case, policies[1]))

    def test_h1_cache_identity_includes_actual_contact_materials(self):
        """Invalidate replay if the scalar or any shape stiffness differs."""
        model = SimpleNamespace(
            soft_contact_ke=50000.0,
            shape_material_ke=wp.array([50000.0, 1000.0], dtype=wp.float32, device="cpu"),
        )
        apply_contact_policy(model, None)
        scene = SimpleNamespace(model=model, comparison_source_hashes={}, comparison_solver_kwargs={})
        before = h1._cache_identity("r7_h1_sheet", scene)
        model.shape_material_ke.assign(np.array([50000.0, 999.0], dtype=np.float32))
        after_shape = h1._cache_identity("r7_h1_sheet", scene)
        self.assertNotEqual(before["model_sha256"], after_shape["model_sha256"])
        model.soft_contact_ke = 1000.0
        after_scalar = h1._cache_identity("r7_h1_sheet", scene)
        self.assertNotEqual(after_shape["soft_contact_ke"], after_scalar["soft_contact_ke"])


if __name__ == "__main__":
    unittest.main()
