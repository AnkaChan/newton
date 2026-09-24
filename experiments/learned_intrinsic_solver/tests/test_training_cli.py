# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check architecture selection at both training command-line boundaries."""

import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import train_epochs, train_smoke


class _ParsedConfig(Exception):
    """Stop at the training boundary after real CLI parsing and config validation."""

    def __init__(self, output, config, *, resume):
        self.config = config


class TestTrainingCli(unittest.TestCase):
    def _parse_config(self, module, arguments):
        def stop_before_training(output, config, *, resume):
            raise _ParsedConfig(output, config, resume=resume)

        with (
            patch("sys.argv", [module.__name__, "--output", "unused", *arguments]),
            patch.object(module, "run_training", stop_before_training),
            self.assertRaises(_ParsedConfig) as result,
        ):
            module._main()
        return result.exception.config

    def test_fresh_cli_uses_one_local_block(self):
        """Construct the one-block baseline when no hop sequence is supplied."""
        for module in (train_smoke, train_epochs):
            with self.subTest(module=module.__name__):
                self.assertEqual(self._parse_config(module, []).hops, (1,))

    def test_cli_accepts_explicit_hop_sequence(self):
        """Preserve an explicitly selected multiblock architecture in each CLI."""
        for module in (train_smoke, train_epochs):
            with self.subTest(module=module.__name__):
                self.assertEqual(self._parse_config(module, ["--hops", "1", "2", "1"]).hops, (1, 2, 1))

    def test_resume_cli_restores_legacy_hops(self):
        """Restore saved three-block hops instead of applying the new-run default."""
        for module, config_type in (
            (train_smoke, train_smoke.TrainSmokeConfig),
            (train_epochs, train_epochs.EpochTrainConfig),
        ):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "legacy.pt"
                torch.save({"config": asdict(config_type(hops=(1, 1, 1)))}, checkpoint)
                self.assertEqual(self._parse_config(module, ["--resume", str(checkpoint)]).hops, (1, 1, 1))
                self.assertEqual(self._parse_config(module, ["--resume", str(checkpoint), "--hops", "1"]).hops, (1,))

    def test_explicit_resume_mismatch_is_rejected_before_training(self):
        """Reject explicit conflicting hops without overwriting a legacy checkpoint."""
        for module, config_type in (
            (train_smoke, train_smoke.TrainSmokeConfig),
            (train_epochs, train_epochs.EpochTrainConfig),
        ):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                checkpoint = output / "legacy.pt"
                torch.save(
                    {"status": "complete", "config": asdict(config_type(hops=(1, 1, 1), device="cpu"))}, checkpoint
                )
                before = checkpoint.read_bytes()
                arguments = [
                    module.__name__,
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                    "--resume",
                    str(checkpoint),
                    "--hops",
                    "1",
                ]
                with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "resume configuration"):
                    module._main()
                self.assertEqual(checkpoint.read_bytes(), before)

    def test_cli_rejects_nonpositive_hops(self):
        """Validate hop distances before crossing the training boundary."""
        for module in (train_smoke, train_epochs):
            with self.subTest(module=module.__name__), self.assertRaisesRegex(ValueError, "hops"):
                self._parse_config(module, ["--hops", "0"])


if __name__ == "__main__":
    unittest.main()
