# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Launch experimental learned-solver training with exclusive local GPU claims.

Each claimed GPU appears as cuda:0 to its rank. The local VM defaults to NCCL
shared-memory transport because peer-to-peer transport is unreliable here;
an explicit NCCL_P2P_DISABLE value in the environment takes precedence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from pathlib import Path

from .launch_distributed_probe import _run_workers

__all__ = ["launch_training"]


def _check_forwarded_arguments(arguments):
    for argument in arguments:
        if argument.split("=", 1)[0] in ("--output", "--resume", "--resume-weights-only"):
            raise ValueError("output and resume must be supplied only to the launcher")


def _next_log_directory(output, *, resume):
    if not resume:
        return output / "logs"
    number = 1
    while (output / f"logs_resume_{number:03d}").exists():
        number += 1
    return output / f"logs_resume_{number:03d}"


def launch_training(
    output: Path,
    *,
    workers: int = 4,
    timeout: float = 86400,
    resume: Path | None = None,
    resume_weights_only: bool = False,
    training_arguments: tuple[str, ...] = (),
    gpu_claim: Path | None = None,
    pipeline: str = "epochs",
) -> dict:
    """Run one, two, or four exclusively claimed GPU ranks under supervision.

    Fresh runs require a new output path. Resume requires an existing checkpoint
    inside that run's ``checkpoints`` directory and writes a new log directory.
    With ``resume_weights_only`` the checkpoint may come from any run (the mixed
    trainer's weights-only initialization): the output path must be fresh, the
    ranks receive ``--resume <checkpoint> --resume-weights-only`` and
    ``launcher.json`` records ``resume_weights_only``.
    """
    if isinstance(workers, bool) or workers not in (1, 2, 4):
        raise ValueError("workers must be one, two, or four")
    if pipeline not in ("epochs", "mixed"):
        raise ValueError("pipeline must be epochs or mixed")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    _check_forwarded_arguments(training_arguments)
    output = Path(output).resolve()
    if resume_weights_only and resume is None:
        raise ValueError("resume_weights_only requires a checkpoint")
    if resume_weights_only and pipeline != "mixed":
        raise ValueError("resume_weights_only is supported by the mixed pipeline only")
    fresh = resume is None or resume_weights_only
    if fresh:
        if output.exists():
            raise FileExistsError(f"Use a fresh output directory; refusing to overwrite {output}")
    if resume_weights_only:
        resume = Path(resume).resolve()
        if not resume.is_file():
            raise FileNotFoundError(f"weights-only checkpoint is missing: {resume}")
    elif resume is not None:
        resume = Path(resume).resolve()
        if not output.is_dir():
            raise FileNotFoundError(f"resume output directory is missing: {output}")
        if not resume.is_relative_to(output / "checkpoints"):
            raise ValueError("resume checkpoint must be contained within the output/checkpoints directory")
        if not resume.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing: {resume}")
    if gpu_claim is None:
        gpu_claim = Path(os.environ.get("AI_DOCS", "/home/horde/Code/AI-Docs")) / "Envs/scripts/gpu-claim.sh"
    gpu_claim = Path(gpu_claim).resolve()
    if not gpu_claim.is_file():
        raise FileNotFoundError(f"GPU claim script not found: {gpu_claim}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    if fresh:
        output.mkdir(parents=True)
    logs = _next_log_directory(output, resume=not fresh)
    commands, environments = [], []
    for rank in range(workers):
        environment = os.environ.copy()
        environment.update(
            RANK=str(rank),
            LOCAL_RANK="0",
            WORLD_SIZE=str(workers),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            PYTHONUNBUFFERED="1",
            TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        )
        environment.setdefault("NCCL_P2P_DISABLE", "1")
        environments.append(environment)
        command = [
            "bash",
            "-c",
            'set -e; source "$1" "$2" occupy; shift 2; "$@"',
            "learned-training-rank",
            str(gpu_claim),
            f"learned-intrinsic-train-{rank}",
            sys.executable,
            "-u",
            "-m",
            f"experiments.learned_intrinsic_solver.train_{pipeline}",
            "--output",
            str(output),
            *training_arguments,
        ]
        if resume is not None:
            command.extend(("--resume", str(resume)))
        if resume_weights_only:
            command.append("--resume-weights-only")
        commands.append(command)
    if resume_weights_only:
        print(f"Initializing network and optimizer weights only from {resume}; fresh run in {output}", flush=True)
    print(f"Launching {workers} ranks; logs: {logs}", flush=True)
    result = dict(_run_workers(commands, logs, timeout=timeout, environments=environments))
    result.update(
        config={
            "workers": workers,
            "timeout": timeout,
            "training_arguments": list(training_arguments),
            "pipeline": pipeline,
        },
        resume=str(resume) if resume is not None else None,
        resume_weights_only=bool(resume_weights_only),
        logs_directory=str(logs),
        world_size=workers,
        master_port=port,
    )
    temporary = output / "launcher.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "launcher.json")
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--timeout", type=float, default=86400)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="initialize a fresh mixed run's network and optimizer from --resume (any run's checkpoint)",
    )
    parser.add_argument("--pipeline", choices=("epochs", "mixed"), default="epochs")
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    result = launch_training(
        args.output,
        workers=args.workers,
        timeout=args.timeout,
        resume=args.resume,
        resume_weights_only=args.resume_weights_only,
        pipeline=args.pipeline,
        training_arguments=tuple(remaining),
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    _main()
