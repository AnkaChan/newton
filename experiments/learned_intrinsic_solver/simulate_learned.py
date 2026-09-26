# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate saved physical states with two learned iterations per Newton step.

Experimental: the trained network supplies proposals without line search or
repair. A failed physical step ends that case and preserves its last valid state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

__all__ = ["run_cases", "select_validation_seeds"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def select_validation_seeds(config, seeds=None) -> tuple[int, ...]:
    """Choose ten reproducible saved validation states, or validate explicit seeds."""
    allowed = set(config.validation_seeds)
    if seeds is None:
        selected = tuple(
            sorted(int(seed) for seed in np.random.default_rng(73).choice(np.arange(10000, 10512), 10, replace=False))
        )
    else:
        selected = tuple(int(seed) for seed in seeds)
    if not selected or len(set(selected)) != len(selected) or any(seed not in allowed for seed in selected):
        raise ValueError("seeds must be unique saved validation physical seeds")
    return selected


def _sampling_schedule(duration: float, dt: float, fps: int) -> tuple[int, int]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")
    steps = round(duration / dt)
    stride = round(1 / (dt * fps))
    if steps < 1 or stride < 1 or not math.isclose(steps * dt, duration, rel_tol=0, abs_tol=1e-9):
        raise ValueError("duration must contain whole physical steps and fps cannot exceed the step rate")
    if not math.isclose(stride * dt * fps, 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("fps must divide the physical step rate exactly")
    return steps, stride


def _read_physical_samples(checkpoint, data_directory, config, seeds):
    """Validate immutable rank cache hashes before returning original X and V."""
    import torch

    world_size = int(checkpoint["world_size"])
    if world_size not in (1, 2, 4) or config.validation_count % world_size:
        raise ValueError("checkpoint validation pool cannot be evenly sharded")
    if len(checkpoint["rank_states"]) != world_size:
        raise ValueError("checkpoint rank-state count differs from world size")
    local_count = config.validation_count // world_size
    by_rank = {}
    for seed in seeds:
        rank = (seed - config.validation_seed_start) // local_count
        by_rank.setdefault(rank, []).append(seed)
    samples = {}
    hashes = {}
    for rank, rank_seeds in by_rank.items():
        path = data_directory / f"rank_{rank}.pt"
        digest = _sha256(path)
        if digest != checkpoint["rank_states"][rank]["dataset_identity"]:
            raise ValueError(f"rank {rank} dataset SHA256 differs from checkpoint")
        archive = torch.load(path, map_location="cpu", weights_only=False)
        metadata = archive.get("dataset_metadata", {})
        if metadata.get("rank") != rank or metadata.get("world_size") != world_size:
            raise ValueError(f"rank {rank} dataset metadata differs from checkpoint")
        if not set(rank_seeds).issubset(set(metadata.get("validation_seeds", ()))):
            raise ValueError(f"rank {rank} does not own requested validation seeds")
        for seed in rank_seeds:
            record = archive["physical_samples"][seed]
            samples[seed] = {
                "positions": np.asarray(record["positions"], dtype=np.float32).copy(),
                "velocity": np.asarray(record["velocity"], dtype=np.float32).copy(),
                "metadata": record["metadata"],
                "source_rank": rank,
            }
        hashes[rank] = digest
    return samples, hashes


def _state_is_valid(solver, result, output_positions, output_velocities, fixed, prescribed):
    """Screen finiteness and pins before accepting a step; inversion is a diagnostic.

    Finite inverted or collapsed cells are accepted (the stable Neo-Hookean
    objective is defined there); the minimum Gauss Jacobian is returned for
    reporting only. Nonfinite positions or energies remain explicit failures.
    """
    import torch

    from .train_smoke import _screen  # noqa: PLC0415 - Optional evaluation boundary.

    if result is None or len(result.updates) != 2:
        raise ValueError("physical step did not perform exactly two learned optimizer iterations")
    minimum = math.inf
    problem = SimpleNamespace(optimizer=solver.learned_step)
    for index, update in enumerate(result.updates, start=1):
        if not torch.isfinite(update.positions).all().item() or not torch.isfinite(update.loss.total).all().item():
            raise ValueError(f"learned iteration {index} has nonfinite positions or energy")
        screen = _screen(problem, update.positions)
        minimum = min(minimum, float(screen["min_gauss_j"]))
    if not np.isfinite(output_positions).all() or not np.isfinite(output_velocities).all():
        raise ValueError("physical step produced nonfinite positions or velocities")
    if not np.array_equal(output_positions[fixed], prescribed) or not np.all(output_velocities[fixed] == 0):
        raise ValueError("physical step moved prescribed pins")
    return minimum


def _simulate_case(
    solver,
    model,
    rest,
    record,
    *,
    seed,
    output_dir,
    checkpoint_path,
    checkpoint_hash,
    checkpoint_epoch,
    dataset_hash,
    dt,
    duration,
    requested_steps,
    frame_stride,
    fps,
    device,
):
    """Run one initial physical state and write its last valid trajectory."""
    import torch

    from .distributed_probe import _parameter_hash  # noqa: PLC0415 - Optional evaluation boundary.

    directory = output_dir / f"seed_{seed}"
    trajectory_path = directory / "trajectory.npz"
    report_path = directory / "report.json"
    if trajectory_path.exists() or report_path.exists():
        raise FileExistsError(f"simulation output already exists for seed {seed}")
    directory.mkdir(parents=True, exist_ok=True)
    initial_positions = record["positions"]
    initial_velocity = record["velocity"]
    rest_positions = rest.corner_rest_positions.astype(np.float32)
    fixed = np.flatnonzero(rest_positions[:, 2] == rest_positions[:, 2].min())
    if initial_positions.shape != rest_positions.shape or initial_velocity.shape != rest_positions.shape:
        raise ValueError(f"seed {seed} has incompatible saved positions or velocities")
    if not np.isfinite(initial_positions).all() or not np.isfinite(initial_velocity).all():
        raise ValueError(f"seed {seed} has nonfinite initial positions or velocities")
    if not np.array_equal(initial_positions[fixed], rest_positions[fixed]) or not np.all(initial_velocity[fixed] == 0):
        raise ValueError(f"seed {seed} does not satisfy initial prescribed pins")
    states = (model.state(), model.state())
    states[0].particle_q.assign(initial_positions)
    states[0].particle_qd.assign(initial_velocity)
    for state in states:
        state.particle_f.zero_()
    frames = [initial_positions.copy()]
    times = [0.0]
    completed = 0
    calls = 0
    min_gauss_j = math.inf
    failure = None
    start = time.perf_counter()
    parameters_before = _parameter_hash(solver.learned_step)
    original_propose = solver.propose_update

    def counted_proposal(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_propose(*args, **kwargs)

    solver.propose_update = counted_proposal
    # A new trajectory starts without optimizer history; step() then carries it.
    solver.last_result = None
    try:
        for step_number in range(1, requested_steps + 1):
            state_in = states[completed % 2]
            state_out = states[(completed + 1) % 2]
            state_in.particle_f.zero_()
            state_out.particle_f.zero_()
            prior_calls = calls
            try:
                solver.step(state_in, state_out, None, None, dt)
                result = solver.last_result
                positions = state_out.particle_q.numpy().copy()
                velocities = state_out.particle_qd.numpy().copy()
                minimum = _state_is_valid(solver, result, positions, velocities, fixed, initial_positions[fixed])
                if calls - prior_calls != 2:
                    raise ValueError("physical step made a number of learned proposal calls other than two")
                min_gauss_j = min(min_gauss_j, minimum)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                failure = {
                    "physical_step": step_number,
                    "time_seconds": completed * dt,
                    "last_valid_time_seconds": completed * dt,
                    "attempted_learned_iterations_in_step": calls - prior_calls,
                    "error": f"{type(error).__name__}: {error}",
                }
                break
            completed += 1
            if completed % frame_stride == 0 or completed == requested_steps:
                frames.append(positions)
                times.append(completed * dt)
            if completed % 100 == 0 or completed == requested_steps:
                print(f"seed={seed} physical_steps={completed}/{requested_steps} learned_calls={calls}", flush=True)
    finally:
        solver.propose_update = original_propose
    if failure is not None and times[-1] != completed * dt:
        frames.append(states[completed % 2].particle_q.numpy().copy())
        times.append(completed * dt)
    parameters_after = _parameter_hash(solver.learned_step)
    if parameters_after != parameters_before:
        raise RuntimeError("simulation changed checkpoint parameters")
    trajectory = np.stack(frames).astype(np.float32)
    frame_times = np.asarray(times, dtype=np.float64)
    report = {
        "schema_version": 1,
        "seed": seed,
        "status": "failed" if failure else "complete",
        "failure": failure,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "dataset_sha256": dataset_hash,
        "source_rank": record["source_rank"],
        "time_step": dt,
        "optimizer_iterations_per_step": 2,
        "initializer": "native rigid-guided fusion once per physical step",
        "requested_duration_seconds": duration,
        "actual_duration_seconds": completed * dt,
        "requested_physical_steps": requested_steps,
        "completed_physical_steps": completed,
        "actual_optimizer_iteration_calls": calls,
        "frame_rate": fps,
        "saved_frame_count": len(frames),
        "min_gauss_j": min_gauss_j if math.isfinite(min_gauss_j) else None,
        "initial_velocity_rms_m_per_s": float(
            np.sqrt(np.mean(np.sum(initial_velocity.astype(np.float64) ** 2, axis=1)))
        ),
        "initial_velocity_max_m_per_s": float(np.linalg.norm(initial_velocity.astype(np.float64), axis=1).max()),
        "elapsed_seconds": time.perf_counter() - start,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "parameter_state_unchanged": True,
        "working_dtype": "float32",
        "tf32_enabled": False,
        "autocast_enabled": False,
    }
    _atomic_npz(
        trajectory_path,
        positions=trajectory,
        times=frame_times,
        rest_positions=rest_positions,
        fixed_indices=fixed.astype(np.int64),
        cell_counts=np.asarray(rest.cell_counts, dtype=np.int64),
        cell_corner_indices=rest.cell_corner_indices.astype(np.int64),
    )
    _atomic_json(report_path, report)
    return report


def run_cases(
    checkpoint: Path,
    output_dir: Path,
    *,
    seeds=None,
    duration: float = 10.0,
    fps: int = 30,
    device: str = "cuda",
    data_directory: Path | None = None,
) -> list[dict]:
    """Simulate saved validation physical states without modifying network weights."""
    import torch

    from .data import generate_cuboid  # noqa: PLC0415 - Optional simulation boundary.
    from .features import CONDITIONING_DIM, STATE_FEATURE_DIM  # noqa: PLC0415
    from .network import IntrinsicSolverNetwork  # noqa: PLC0415
    from .newton_model import build_newton_hex_model  # noqa: PLC0415
    from .newton_solver import SolverLearnedIntrinsic  # noqa: PLC0415
    from .train_epochs import EpochTrainConfig  # noqa: PLC0415

    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    data_directory = Path(data_directory) if data_directory else checkpoint.parent.parent / "data"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema_version") != 1 or saved.get("status") == "failed":
        raise ValueError("checkpoint is not a usable learned training checkpoint")
    config = EpochTrainConfig(**saved["config"])
    selected = select_validation_seeds(config, seeds)
    dt = config.time_step
    requested_steps, frame_stride = _sampling_schedule(duration, dt, fps)
    physical, hashes = _read_physical_samples(saved, data_directory, config, selected)
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if target.type == "cuda":
        if torch.cuda.device_count() != 1 or int(os.environ.get("LOCAL_RANK", "0")) != 0:
            raise RuntimeError("each CUDA worker must claim one device as cuda:0")
        target = torch.device("cuda:0")
        torch.cuda.set_device(target)
        torch.cuda.reset_peak_memory_stats(target)
    checkpoint_hash = _sha256(checkpoint)
    previous_threads = torch.get_num_threads()
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
        model = build_newton_hex_model(
            rest,
            fixed,
            lame_lambda=config.lame_lambda,
            lame_mu=config.lame_mu,
            density=config.density,
            gravity=config.gravity,
        )
        network = IntrinsicSolverNetwork(
            config.cell_counts,
            STATE_FEATURE_DIM,
            conditioning_dim=CONDITIONING_DIM,
            hidden_dim=config.hidden_dim,
            edge_hidden_dim=config.edge_hidden_dim,
            num_heads=config.num_heads,
            hops=config.hops,
            max_step_size=config.max_step_size,
            query_chunk_size=config.query_chunk_size,
        ).to(target)
        solver = SolverLearnedIntrinsic(model, network=network, iterations=2)
        step = solver._step_for_dt(dt)
        step.load_state_dict(saved["step_state"])
        step.eval()
        results = []
        with torch.no_grad(), torch.autocast(device_type=target.type, enabled=False):
            for seed in selected:
                results.append(
                    _simulate_case(
                        solver,
                        model,
                        rest,
                        physical[seed],
                        seed=seed,
                        output_dir=output_dir,
                        checkpoint_path=checkpoint,
                        checkpoint_hash=checkpoint_hash,
                        checkpoint_epoch=int(saved["completed_epochs"]),
                        dataset_hash=hashes[physical[seed]["source_rank"]],
                        dt=dt,
                        duration=duration,
                        requested_steps=requested_steps,
                        frame_stride=frame_stride,
                        fps=fps,
                        device=target,
                    )
                )
        return results
    finally:
        torch.set_num_threads(previous_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    reports = run_cases(
        args.checkpoint,
        args.output_dir,
        seeds=args.seeds,
        duration=args.duration,
        fps=args.fps,
        device=args.device,
        data_directory=args.data_directory,
    )
    print(
        json.dumps(
            [
                {
                    "seed": report["seed"],
                    "status": report["status"],
                    "completed_physical_steps": report["completed_physical_steps"],
                }
                for report in reports
            ]
        ),
        flush=True,
    )
    if any(report["status"] != "complete" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
