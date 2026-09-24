# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU launcher contracts; no GPU claims or workers are started."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.learned_intrinsic_solver import launch_training as launcher_module
from experiments.learned_intrinsic_solver.launch_training import launch_training

_WORKER_RESULT = {
    "passed": True,
    "exit_codes": [0, 0, 0, 0],
    "timed_out": False,
    "failure": None,
    "elapsed_seconds": 1.0,
}


class TestLaunchTraining(unittest.TestCase):
    def test_fresh_launch_claims_each_gpu_and_records_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim = root / "gpu-claim.sh"
            claim.write_text("# test placeholder\n")
            output = root / "run"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "experiments.learned_intrinsic_solver.launch_training._run_workers", return_value=_WORKER_RESULT
                ) as run,
            ):
                result = launch_training(
                    output,
                    workers=4,
                    timeout=86400,
                    training_arguments=("--batch-size", "16", "--max-epochs", "30"),
                    gpu_claim=claim,
                )
            commands, log_directory = run.call_args.args
            environments = run.call_args.kwargs["environments"]
            self.assertEqual(log_directory, output / "logs")
            self.assertEqual(run.call_args.kwargs["timeout"], 86400)
            self.assertEqual(len(commands), 4)
            for rank, (command, environment) in enumerate(zip(commands, environments, strict=True)):
                self.assertEqual(command[:3], ["bash", "-c", 'set -e; source "$1" "$2" occupy; shift 2; "$@"'])
                self.assertIn("experiments.learned_intrinsic_solver.train_epochs", command)
                self.assertEqual(command[command.index("--output") + 1], str(output))
                self.assertEqual(command[-4:], ["--batch-size", "16", "--max-epochs", "30"])
                self.assertEqual(environment["RANK"], str(rank))
                self.assertEqual(environment["WORLD_SIZE"], "4")
                self.assertEqual(environment["LOCAL_RANK"], "0")
                self.assertEqual(environment["NCCL_P2P_DISABLE"], "1")
            self.assertEqual(result["exit_codes"], [0, 0, 0, 0])
            recorded = json.loads((output / "launcher.json").read_text())
            self.assertEqual(recorded["resume"], None)
            self.assertEqual(recorded["config"]["training_arguments"], ["--batch-size", "16", "--max-epochs", "30"])
            self.assertEqual(recorded["exit_codes"], [0, 0, 0, 0])
            self.assertEqual(recorded["logs_directory"], str(output / "logs"))

    def test_resume_uses_new_logs_and_preserves_existing_nccl_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim = root / "gpu-claim.sh"
            claim.touch()
            output = root / "run"
            checkpoints = output / "checkpoints"
            checkpoints.mkdir(parents=True)
            resume = checkpoints / "latest.pt"
            resume.touch()
            (output / "logs").mkdir()
            (output / "logs_resume_001").mkdir()
            marker = output / "logs" / "rank_0.log"
            marker.write_text("original log")
            with (
                patch.dict(os.environ, {"NCCL_P2P_DISABLE": "0"}),
                patch(
                    "experiments.learned_intrinsic_solver.launch_training._run_workers", return_value=_WORKER_RESULT
                ) as run,
            ):
                result = launch_training(output, resume=resume, gpu_claim=claim)
            self.assertEqual(run.call_args.args[1], output / "logs_resume_002")
            self.assertEqual(run.call_args.kwargs["environments"][0]["NCCL_P2P_DISABLE"], "0")
            self.assertEqual(marker.read_text(), "original log")
            self.assertEqual(result["resume"], str(resume))
            self.assertEqual(run.call_args.args[0][0][-2:], ["--resume", str(resume)])

    def test_rejects_existing_fresh_output_and_unsafe_resume_before_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim = root / "gpu-claim.sh"
            claim.touch()
            output = root / "run"
            output.mkdir()
            outside = root / "outside.pt"
            outside.touch()
            with patch("experiments.learned_intrinsic_solver.launch_training._run_workers") as run:
                with self.assertRaises(FileExistsError):
                    launch_training(output, gpu_claim=claim)
                with self.assertRaisesRegex(ValueError, "checkpoints"):
                    launch_training(output, resume=outside, gpu_claim=claim)
                with self.assertRaises(FileNotFoundError):
                    launch_training(output, resume=output / "checkpoints/missing.pt", gpu_claim=claim)
                for arguments in (
                    ("--output", "elsewhere"),
                    ("--output=elsewhere",),
                    ("--resume", "elsewhere"),
                    ("--resume=elsewhere",),
                ):
                    with self.assertRaisesRegex(ValueError, "launcher"):
                        launch_training(root / "fresh", training_arguments=arguments, gpu_claim=claim)
            run.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])

    def test_cli_forwards_training_arguments_without_abbreviating_launcher_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["launch_training", "--output", str(output), "--workers", "2", "--", "--max-epochs", "30"],
                ),
                patch.object(launcher_module, "launch_training", return_value={"passed": True}) as launch,
            ):
                with self.assertRaises(SystemExit) as finished:
                    launcher_module._main()
            self.assertEqual(finished.exception.code, 0)
            self.assertEqual(launch.call_args.args, (output,))
            self.assertEqual(launch.call_args.kwargs["workers"], 2)
            self.assertEqual(launch.call_args.kwargs["training_arguments"], ("--max-epochs", "30"))
            with patch.object(sys, "argv", ["launch_training", "--out", str(output)]):
                with self.assertRaises(SystemExit) as rejected:
                    launcher_module._main()
            self.assertEqual(rejected.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
