# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Launch an experimental local DDP probe with one exclusive claim per GPU.

This Linux development helper uses the workspace GPU-claim script. It does not
override existing reservations. Each rank sees its own claimed GPU as cuda:0.
The parent supervises ranks and stops their process groups on failure or timeout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

__all__ = ["launch_probe"]


def _stop_workers(processes):
    groups = {process.pid for process in processes}
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while groups and time.monotonic() < deadline:
        for process in processes:
            process.poll()
        for group in tuple(groups):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                groups.remove(group)
        if groups:
            time.sleep(0.05)
    # A child may outlive its shell and retain the GPU's inherited lock FD.
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.wait()


def _run_workers(commands, log_directory, *, timeout, environments=None):
    if not commands or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("workers and a finite positive timeout are required")
    if environments is not None and len(environments) != len(commands):
        raise ValueError("each worker must have one environment")
    log_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    start = time.monotonic()
    timed_out = False
    failure = None
    with ExitStack() as stack:
        try:
            for rank, command in enumerate(commands):
                log = stack.enter_context((log_directory / f"rank_{rank}.log").open("wb"))
                processes.append(
                    subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=None if environments is None else environments[rank],
                        start_new_session=True,
                    )
                )
            while True:
                codes = [process.poll() for process in processes]
                bad_ranks = [rank for rank, code in enumerate(codes) if code not in (None, 0)]
                if bad_ranks:
                    failure = f"rank {bad_ranks[0]} exited with status {codes[bad_ranks[0]]}"
                    break
                if all(code == 0 for code in codes):
                    break
                if time.monotonic() - start >= timeout:
                    timed_out = True
                    failure = "worker group exceeded the timeout"
                    break
                time.sleep(0.05)
        finally:
            # Also stop descendants if an exited shell left a process behind.
            _stop_workers(processes)
    codes = [process.returncode for process in processes]
    return {
        "passed": not failure and len(codes) == len(commands) and all(code == 0 for code in codes),
        "exit_codes": codes,
        "timed_out": timed_out,
        "failure": failure,
        "elapsed_seconds": time.monotonic() - start,
    }


def launch_probe(
    output: Path,
    *,
    workers: int = 4,
    timeout: float = 600,
    probe_arguments: tuple[str, ...] = (),
    gpu_claim: Path | None = None,
) -> dict:
    """Run the experimental probe on exclusively claimed local GPUs.

    Args:
        output: Fresh run directory; existing paths are rejected unchanged.
        workers: Number of GPU processes, one, two, or four.
        timeout: Maximum wall-clock time for the group, in seconds.
        probe_arguments: Additional arguments for distributed_probe.
        gpu_claim: Optional workspace GPU-claim script path.

    Returns:
        Worker exit codes, elapsed time, and aggregate process success. Numerical
        validation results are written by the probe and independent reference.
    """
    if isinstance(workers, bool) or workers not in (1, 2, 4):
        raise ValueError("workers must be one, two, or four")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if any(argument == "--output" or argument.startswith("--output=") for argument in probe_arguments):
        raise ValueError("output must be supplied only to the launcher")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory; refusing to overwrite {output}")
    if gpu_claim is None:
        gpu_claim = Path(os.environ.get("AI_DOCS", "/home/horde/Code/AI-Docs")) / "Envs/scripts/gpu-claim.sh"
    gpu_claim = Path(gpu_claim).resolve()
    if not gpu_claim.is_file():
        raise FileNotFoundError(f"GPU claim script not found: {gpu_claim}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    output.mkdir(parents=True)
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
        environments.append(environment)
        commands.append(
            [
                "bash",
                "-c",
                'set -e; source "$1" "$2" occupy; shift 2; "$@"',
                "distributed-probe-rank",
                str(gpu_claim),
                f"learned-intrinsic-ddp-{rank}",
                sys.executable,
                "-u",
                "-m",
                "experiments.learned_intrinsic_solver.distributed_probe",
                "--output",
                str(output),
                *probe_arguments,
            ]
        )
    print(f"Launching {workers} ranks; logs: {output / 'logs'}", flush=True)
    result = _run_workers(commands, output / "logs", timeout=timeout, environments=environments)
    result["world_size"] = workers
    result["master_port"] = port
    temporary = output / "launcher.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "launcher.json")
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--timeout", type=float, default=600)
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    result = launch_probe(args.output, workers=args.workers, timeout=args.timeout, probe_arguments=tuple(remaining))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    _main()
