# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate repeated learned proposals on fixed held-out physical queries.

Experimental: this is 100 optimization iterations against each query's original
physical predictor, not a simulation rollout or additional training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

__all__ = ["run_rank"]


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


def _checkpoint_validation(saved):
    epoch = int(saved["completed_epochs"])
    if epoch == 0:
        return saved["validation_initial"]
    matches = [row["validation"] for row in saved["history"] if row["epoch"] == epoch]
    if len(matches) != 1:
        raise ValueError("checkpoint has no unique validation for its completed epoch")
    return matches[0]


def _baseline(step, positions, inertial, seeds, indices, energies, failures):
    """Evaluate initial energy, splitting a bad batch to identify each bad case."""
    import torch

    if not indices:
        return
    selection = torch.tensor(indices, dtype=torch.long, device=positions.device)
    try:
        values = step.energy(positions[selection], inertial[selection]).total.detach()
        if not torch.isfinite(values).all().item() or (values <= 0).any().item():
            raise ValueError("initial physical energy must be finite and positive")
        energies[0, indices] = values.cpu().numpy().astype(np.float64)
    except (ValueError, RuntimeError) as error:
        if len(indices) == 1:
            failures.append({"physical_seed": int(seeds[indices[0]]), "iteration": 0, "error": str(error)})
            return
        middle = len(indices) // 2
        _baseline(step, positions, inertial, seeds, indices[:middle], energies, failures)
        _baseline(step, positions, inertial, seeds, indices[middle:], energies, failures)


def _propose(step, positions, inertial, pins, seeds, indices, iteration, energies, failures):
    """Advance good cases and recursively isolate invalid proposals."""
    import torch

    from .train_epochs import _screen_output  # noqa: PLC0415 - Optional Torch boundary.

    if not indices:
        return []
    selection = torch.tensor(indices, dtype=torch.long, device=positions.device)
    try:
        result = step(positions[selection], inertial[selection], fixed_positions=pins[selection])
        _screen_output(step, result, pins[selection])
        after = result.loss.total.detach()
        if not torch.isfinite(after).all().item():
            raise ValueError("learned proposal energy is nonfinite")
        energies[iteration, indices] = after.cpu().numpy().astype(np.float64)
        return [(index, result.positions[offset : offset + 1].detach()) for offset, index in enumerate(indices)]
    except (ValueError, RuntimeError) as error:
        if len(indices) == 1:
            failures.append({"physical_seed": int(seeds[indices[0]]), "iteration": iteration, "error": str(error)})
            return []
        middle = len(indices) // 2
        return _propose(
            step, positions, inertial, pins, seeds, indices[:middle], iteration, energies, failures
        ) + _propose(step, positions, inertial, pins, seeds, indices[middle:], iteration, energies, failures)


def _rollout_batch(step, batch, iterations, device, *, rank=0, batch_number=0):
    """Preserve original Y and pins while repeatedly updating candidate corners."""
    positions = batch["positions"].to(device)
    inertial = batch["inertial_prediction"].to(device)
    pins = batch["fixed_positions"].to(device)
    seeds = [int(seed) for seed in batch["physical_seeds"]]
    count = len(seeds)
    energies = np.full((iterations + 1, count), np.nan, dtype=np.float64)
    failures = []
    _baseline(step, positions, inertial, seeds, list(range(count)), energies, failures)
    active = [index for index in range(count) if math.isfinite(energies[0, index])]
    for iteration in range(1, iterations + 1):
        if not active:
            break
        survivors = _propose(step, positions, inertial, pins, seeds, active, iteration, energies, failures)
        active = [index for index, _ in survivors]
        for index, proposal in survivors:
            positions[index : index + 1] = proposal
        if iteration % 10 == 0 or iteration == iterations:
            print(
                f"rank={rank} batch={batch_number} iteration={iteration}/{iterations} surviving={len(active)}/{count}",
                flush=True,
            )
    relative = np.full_like(energies, np.nan)
    valid = np.isfinite(energies[0])
    relative[:, valid] = energies[:, valid] / energies[0, valid][None]
    return np.asarray(seeds, dtype=np.int64), energies, relative, failures


def run_rank(
    checkpoint: Path,
    data_directory: Path,
    output: Path,
    *,
    rank: int,
    world_size: int,
    iterations: int = 100,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """Write one independent rank's complete fixed-query energy trajectories.

    Failed cases retain NaN from their first invalid iteration onward. The
    caller aggregates the four rank files and validates the global step-one
    mean against the checkpoint's matching validation epoch.
    """
    import torch

    from .data import generate_cuboid  # noqa: PLC0415 - Optional evaluation boundary.
    from .distributed_probe import _parameter_hash  # noqa: PLC0415
    from .epoch_data import EpochDataset  # noqa: PLC0415
    from .network import IntrinsicSolverNetwork  # noqa: PLC0415
    from .newton_model import build_newton_hex_model  # noqa: PLC0415
    from .newton_solver import SolverLearnedIntrinsic  # noqa: PLC0415
    from .train_epochs import EpochTrainConfig  # noqa: PLC0415

    if world_size not in (1, 2, 4) or isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank/world_size must identify one of 1, 2, or 4 ranks")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    checkpoint, data_directory, output = Path(checkpoint), Path(data_directory), Path(output)
    npz_path, json_path = output / f"rank_{rank}.npz", output / f"rank_{rank}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"rank {rank} evaluation output already exists")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema_version") != 1 or saved.get("status") == "failed":
        raise ValueError("checkpoint is not a usable training checkpoint")
    if saved["world_size"] != world_size or len(saved["rank_states"]) != world_size:
        raise ValueError("checkpoint world size differs from evaluation world size")
    config = EpochTrainConfig(**saved["config"])
    expected_validation = _checkpoint_validation(saved)
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if target.type == "cuda":
        if torch.cuda.device_count() != 1 or int(os.environ.get("LOCAL_RANK", "0")) != 0:
            raise RuntimeError("each CUDA rank must claim one device as cuda:0")
        target = torch.device("cuda:0")
        torch.cuda.set_device(target)
        torch.cuda.reset_peak_memory_stats(target)
    checkpoint_hash = _sha256(checkpoint)
    start = time.perf_counter()
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
            38,
            hidden_dim=config.hidden_dim,
            edge_hidden_dim=config.edge_hidden_dim,
            num_heads=config.num_heads,
            hops=config.hops,
            max_step_size=config.max_step_size,
            query_chunk_size=config.query_chunk_size,
        ).to(target)
        solver = SolverLearnedIntrinsic(model, network=network, iterations=1)
        dataset = EpochDataset(
            config,
            rest,
            model,
            solver,
            rank=rank,
            world_size=world_size,
            dataset_dir=data_directory,
            resume=True,
        )
        if dataset.dataset_identity != saved["rank_states"][rank]["dataset_identity"]:
            raise ValueError("dataset SHA256 differs from checkpoint rank state")
        step = dataset.step
        step.load_state_dict(saved["step_state"])
        step.eval()
        weights_before = _parameter_hash(step)
        seed_parts, energy_parts, relative_parts, failures = [], [], [], []
        with torch.no_grad(), torch.autocast(device_type=target.type, enabled=False):
            for batch_number, batch in enumerate(dataset.validation_batches(batch_size), start=1):
                seeds, energies, relative, batch_failures = _rollout_batch(
                    step, batch, iterations, target, rank=rank, batch_number=batch_number
                )
                seed_parts.append(seeds)
                energy_parts.append(energies)
                relative_parts.append(relative)
                failures.extend(batch_failures)
        physical_seeds = np.concatenate(seed_parts)
        energies = np.concatenate(energy_parts, axis=1)
        relative = np.concatenate(relative_parts, axis=1)
        if physical_seeds.tolist() != list(dataset.validation_seeds):
            raise ValueError("evaluation did not retain every rank-local validation seed in fixed order")
        weights_after = _parameter_hash(step)
        if weights_after != weights_before:
            raise RuntimeError("inference changed checkpoint parameters")
        baseline = energies[0]
        one_step = energies[1]
        matched = np.isfinite(baseline) & np.isfinite(one_step)
        normalized = (one_step[matched] - baseline[matched]) / np.maximum(baseline[matched], 1.0)
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "rank": rank,
            "world_size": world_size,
            "iterations": iterations,
            "batch_size": batch_size,
            "sample_count": len(physical_seeds),
            "checkpoint_epoch": int(saved["completed_epochs"]),
            "checkpoint_optimizer_updates": int(saved["optimizer_updates"]),
            "checkpoint_sha256": checkpoint_hash,
            "dataset_sha256": dataset.dataset_identity,
            "parameter_sha256": weights_before,
            "parameter_state_unchanged": True,
            "validation_before_mean_joule": float(np.mean(baseline[matched])) if matched.any() else None,
            "validation_after_one_step_mean_joule": float(np.mean(one_step[matched])) if matched.any() else None,
            "validation_one_step_normalized_loss": float(np.mean(normalized)) if matched.any() else None,
            "one_step_valid_count": int(matched.sum()),
            "checkpoint_validation_mean_normalized_loss": expected_validation["mean_normalized_loss"],
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - start,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else 0,
            "gpu_name": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
            "working_dtype": "float32",
            "tf32_enabled": False,
            "autocast_enabled": False,
            "config": asdict(config),
        }
        if world_size == 1 and not failures and expected_validation["mean_normalized_loss"] is not None:
            if not math.isclose(
                metadata["validation_one_step_normalized_loss"],
                expected_validation["mean_normalized_loss"],
                rel_tol=1e-3,
                abs_tol=2e-5,
            ):
                raise ValueError("one-step rollout loss disagrees with checkpoint validation")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_npz(npz_path, physical_seeds=physical_seeds, energies=energies, relative_energies=relative)
        _atomic_json(json_path, metadata)
        return metadata
    finally:
        torch.set_num_threads(previous_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_rank(
        args.checkpoint,
        args.data_directory,
        args.output,
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        iterations=args.iterations,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        json.dumps({key: result[key] for key in ("rank", "sample_count", "iterations", "checkpoint_epoch")}), flush=True
    )


if __name__ == "__main__":
    _main()
