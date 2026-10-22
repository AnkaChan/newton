# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that the standalone command preserves runner controls."""

import contextlib
import io
import unittest
from pathlib import Path

from .run_fast_math import parse_arguments, runner_arguments


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

    def test_help_exits_before_environment_or_scene_initialization(self):
        """Describe both switches without requiring a case or old recordings."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            parse_arguments(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--mode", output.getvalue())
        self.assertIn("--math", output.getvalue())


if __name__ == "__main__":
    unittest.main()
