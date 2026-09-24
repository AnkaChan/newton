# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded, real four-GPU DDP diagnostic for one learned hex optimizer step.

Each process sees one exclusively claimed CUDA device as cuda:0. Native Newton
sampling and PARDISO fusion factorization remain on CPU. The fixed physical query
batch is replayed for a few Adam updates; this is not an epoch or a rollout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

__all__ = ["ProbeConfig", "build_step", "collate_queries", "run_probe"]


@dataclass(frozen=True)
class ProbeConfig:
    """Experimental fixed-size distributed diagnostic settings."""

    rank: int = 0
    world_size: int = 4
    batch_size: int = 16
    updates: int = 3
    cell_counts: tuple[int, int, int] = (10, 10, 40)
    seed: int = 73
    fail_rank: int | None = None
    fail_update: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "cell_counts", tuple(self.cell_counts))
        for name in ("world_size", "batch_size", "updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if len(self.cell_counts) != 3 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.cell_counts
        ):
            raise ValueError("cell_counts must contain three positive integers")
        if (self.fail_rank is None) != (self.fail_update is None):
            raise ValueError("fail_rank and fail_update must be set together")
        if self.fail_rank is not None and (
            isinstance(self.fail_rank, bool)
            or not isinstance(self.fail_rank, int)
            or not 0 <= self.fail_rank < self.world_size
        ):
            raise ValueError("fail_rank must be in [0, world_size)")
        if self.fail_update is not None and (
            isinstance(self.fail_update, bool)
            or not isinstance(self.fail_update, int)
            or not 1 <= self.fail_update <= self.updates
        ):
            raise ValueError("fail_update must be in [1, updates]")

    @property
    def train_seeds(self) -> tuple[int, ...]:
        """Return this rank's contiguous, disjoint physical seeds."""
        start = self.rank * self.batch_size
        return tuple(range(start, start + self.batch_size))

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        """Return a distinct seed used only by the shared sampler setup."""
        return (10000 + self.rank,)


def collate_queries(queries):
    """Pack fixed per-object candidates while retaining each original Y and pins."""
    import torch

    if not queries:
        raise ValueError("queries must be nonempty")
    fixed_indices = queries[0][0].fixed_indices
    for problem, candidate, metadata in queries:
        if not torch.equal(problem.fixed_indices, fixed_indices):
            raise ValueError("queries must have identical pin indices")
        if candidate.shape != problem.inertial_prediction.shape or candidate.shape[0] != 1:
            raise ValueError("candidate and original Y must have matching single-object shapes")
        if not torch.equal(candidate[:, fixed_indices], problem.fixed_positions):
            raise ValueError("candidate pins must equal the physical prescribed positions")
        if not isinstance(metadata.get("physical_seed"), int):
            raise ValueError("every query needs an integer physical_seed")
    return {
        "positions": torch.cat([candidate for _, candidate, _ in queries], dim=0).detach(),
        "inertial_prediction": torch.cat([problem.inertial_prediction for problem, _, _ in queries], dim=0).detach(),
        "fixed_positions": torch.cat([problem.fixed_positions for problem, _, _ in queries], dim=0).detach(),
        "physical_seeds": [metadata["physical_seed"] for _, _, metadata in queries],
        "metadata": [copy.deepcopy(metadata) for _, _, metadata in queries],
    }


def _network(config, device):
    import torch

    from .network import IntrinsicSolverNetwork  # noqa: PLC0415 - Optional training boundary.

    torch.manual_seed(config.seed)
    network = IntrinsicSolverNetwork(
        config.cell_counts,
        38,
        hidden_dim=128,
        edge_hidden_dim=64,
        num_heads=4,
        hops=(1, 1, 1),
        max_step_size=0.05,
        query_chunk_size=128,
    ).to(device)
    # A small correction head permits gradients through the encoder at update 1.
    with torch.no_grad():
        torch.nn.init.normal_(network.correction_head.weight, std=1e-4)
        torch.nn.init.normal_(network.correction_head.bias, std=1e-4)
    return network


def build_step(config: ProbeConfig, device):
    """Build the plain step used by a single-GPU replay of saved rank inputs."""
    from .data import generate_cuboid  # noqa: PLC0415 - Optional training boundary.
    from .solver_step import LearnedHexSolverStep  # noqa: PLC0415 - Optional training boundary.

    rest = generate_cuboid(config.cell_counts, cell_size=0.025)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    return LearnedHexSolverStep(
        rest,
        fixed,
        lame_lambda=288461.53846,
        lame_mu=192307.69231,
        density=1000.0,
        time_step=1 / 300,
        network=_network(config, device),
    )


def _make_fixed_batch(config, device):
    from .data import generate_cuboid  # noqa: PLC0415 - Optional training boundary.
    from .newton_model import build_newton_hex_model  # noqa: PLC0415 - Optional training boundary.
    from .newton_solver import SolverLearnedIntrinsic  # noqa: PLC0415 - Optional training boundary.
    from .train_smoke import TrainSmokeConfig, _Sampler  # noqa: PLC0415 - Optional training boundary.

    rest = generate_cuboid(config.cell_counts, cell_size=0.025)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    model = build_newton_hex_model(
        rest,
        fixed,
        lame_lambda=288461.53846,
        lame_mu=192307.69231,
        density=1000.0,
        gravity=(0.0, -9.81, 0.0),
    )
    solver = SolverLearnedIntrinsic(model, network=_network(config, device), iterations=1)
    sampler_config = TrainSmokeConfig(
        updates=1,
        cell_counts=config.cell_counts,
        train_count=config.batch_size,
        validation_count=1,
        train_seed_start=config.train_seeds[0],
        validation_seed_start=config.validation_seeds[0],
        seed=config.seed,
        verbose=False,
        device=str(device),
    )
    sampler = _Sampler(sampler_config, rest, model, solver)
    queries = []
    for seed in config.train_seeds:
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, seed, 811]))
        candidate, metadata = sampler.candidate(seed, rng)
        queries.append((sampler.problems[seed], candidate, metadata))
    return solver.learned_step, collate_queries(queries)


def _cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _flat_parameters(step, *, gradients=False):
    import torch

    parameters = list(step.parameters())
    values = [
        (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad) if gradients else parameter
        for parameter in parameters
    ]
    return torch.cat([value.detach().reshape(-1) for value in values]).cpu()


def _parameter_hash(step):
    flat = _flat_parameters(step)
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def _states_equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_states_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_states_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _replicas_equal(step, world_size, device):
    import torch
    import torch.distributed as dist

    digest = _parameter_hash(step)
    local = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    hashes = [bytes(value.cpu().tolist()).hex() for value in gathered]
    return len(set(hashes)) == 1, hashes


def _all_ranks_ok(local_error, device):
    import torch
    import torch.distributed as dist

    bad = torch.tensor(int(local_error is not None), device=device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if not bad.item():
        return None
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    return {str(rank): error for rank, error in enumerate(errors) if error is not None}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _phase(config, name):
    print(f"rank={config.rank} phase={name}", flush=True)


def run_probe(output: Path, config: ProbeConfig) -> dict:
    """Run bounded DDP updates and save exact rank data for independent replay."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if torch.cuda.device_count() != 1:
        raise RuntimeError("each rank must see exactly one exclusively claimed CUDA device")
    if os.environ.get("LOCAL_RANK") != "0":
        raise RuntimeError("LOCAL_RANK must be 0 within each exclusive device claim")
    if int(os.environ.get("RANK", "-1")) != config.rank or int(os.environ.get("WORLD_SIZE", "-1")) != config.world_size:
        raise RuntimeError("RANK and WORLD_SIZE must match ProbeConfig")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for path in (
        output / f"rank_{config.rank}.json",
        output / f"rank_{config.rank}.pt",
        output / f"first_update_rank_{config.rank}.pt",
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if config.rank == 0 and (output / "report.json").exists():
        raise FileExistsError("refusing to overwrite report.json")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    _phase(config, "process_group_start")
    dist.init_process_group("nccl", rank=config.rank, world_size=config.world_size, device_id=device)
    _phase(config, "process_group_ready")
    start = time.perf_counter()
    try:
        local_error = None
        step = batch = None
        try:
            step, batch = _make_fixed_batch(config, device)
        except Exception as exc:
            local_error = f"preparation: {type(exc).__name__}: {exc}"
        errors = _all_ranks_ok(local_error, device)
        if errors:
            raise RuntimeError(f"synchronized preparation failure: {errors}")
        _phase(config, "batch_ready")
        module = DistributedDataParallel(step, device_ids=[0], output_device=0, broadcast_buffers=False)
        _phase(config, "ddp_ready")
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
        initial_step_state = _cpu(step.state_dict()) if config.rank == 0 else None
        initial_optimizer_state = _cpu(optimizer.state_dict()) if config.rank == 0 else None
        input_batch = _cpu(batch)
        positions, inertial, pins = (batch[key] for key in ("positions", "inertial_prediction", "fixed_positions"))
        local_error = None
        before = None
        try:
            with torch.no_grad():
                before = step.energy(positions, inertial).total.detach()
            if not torch.isfinite(before).all().item():
                raise ValueError("nonfinite initial physical energy")
        except Exception as exc:
            local_error = f"initial objective: {type(exc).__name__}: {exc}"
        errors = _all_ranks_ok(local_error, device)
        if errors:
            raise RuntimeError(f"synchronized initial objective failure: {errors}")
        initial_equal, initial_hashes = _replicas_equal(step, config.world_size, device)
        if not initial_equal:
            raise ValueError(f"initial DDP replicas diverged: {initial_hashes}")
        _phase(config, "initial_replicas_equal")
        first_frames = first_after = first_normalized = first_grad = first_weights = None
        after_first_step_state = after_first_optimizer_state = after_first_rng = None
        rows = []
        failure = None
        completed = 0
        final_after = None
        for update in range(1, config.updates + 1):
            _phase(config, f"update_{update}_start")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            update_start = time.perf_counter()
            local_error = None
            result = normalized = None
            try:
                with torch.autocast(device_type="cuda", enabled=False):
                    result = module(positions, inertial, fixed_positions=pins)
                    normalized = (result.loss.total - before) / before.clamp_min(1.0)
                from .train_smoke import _screen  # noqa: PLC0415 - Optional training boundary.

                screen = _screen(SimpleNamespace(optimizer=step), result.positions)
                if not screen["valid"] or not torch.isfinite(result.loss.total).all().item():
                    raise ValueError(f"invalid learned output or energy: {screen}")
                if not torch.equal(result.positions[:, step.fixed_indices], pins):
                    raise ValueError("learned output moved prescribed pins")
                if config.rank == config.fail_rank and update == config.fail_update:
                    raise ValueError("intentional rank failure before backward")
            except Exception as exc:
                local_error = f"update {update} forward: {type(exc).__name__}: {exc}"
            errors = _all_ranks_ok(local_error, device)
            if errors:
                failure = {"stage": "before_backward", "update": update, "errors": errors}
                break
            _phase(config, f"update_{update}_forward_ready")
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - update_start
            local_error = None
            try:
                normalized.mean().backward()
                gradients = [parameter.grad for parameter in step.parameters()]
                if all(gradient is None for gradient in gradients) or any(
                    gradient is not None and not torch.isfinite(gradient).all().item() for gradient in gradients
                ):
                    raise ValueError("nonfinite or absent parameter gradients")
            except Exception as exc:
                local_error = f"update {update} backward: {type(exc).__name__}: {exc}"
            errors = _all_ranks_ok(local_error, device)
            if errors:
                failure = {"stage": "before_adam", "update": update, "errors": errors}
                break
            _phase(config, f"update_{update}_backward_ready")
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - update_start - forward_seconds
            if update == 1:
                first_frames = _cpu(result.frames)
                first_after = _cpu(result.loss.total)
                first_normalized = _cpu(normalized)
                first_grad = _flat_parameters(step, gradients=True)
            adam_start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            adam_seconds = time.perf_counter() - adam_start
            local_error = None
            if any(not torch.isfinite(parameter).all().item() for parameter in step.parameters()):
                local_error = f"update {update}: Adam produced nonfinite parameters"
            errors = _all_ranks_ok(local_error, device)
            if errors:
                failure = {"stage": "after_adam", "update": update, "errors": errors}
                break
            equal, hashes = _replicas_equal(step, config.world_size, device)
            if not equal:
                failure = {"stage": "replica_check", "update": update, "hashes": hashes}
                break
            _phase(config, f"update_{update}_replicas_equal")
            if update == 1:
                first_weights = _flat_parameters(step)
                after_first_step_state = _cpu(step.state_dict())
                after_first_optimizer_state = _cpu(optimizer.state_dict())
                after_first_rng = {
                    "torch": torch.get_rng_state().clone(),
                    "cuda": torch.cuda.get_rng_state(device).clone(),
                }
                torch.save(
                    {
                        "schema_version": 1,
                        "config": asdict(config),
                        "rank": config.rank,
                        "optimizer_updates": 1,
                        "step_state": after_first_step_state,
                        "optimizer_state": after_first_optimizer_state,
                        "rng_state": after_first_rng,
                    },
                    output / f"first_update_rank_{config.rank}.pt",
                )
            completed = update
            final_after = _cpu(result.loss.total)
            rows.append(
                {
                    "update": update,
                    "normalized_loss": float(normalized.mean().detach()),
                    "energy_before_mean_joule": float(before.mean()),
                    "energy_after_mean_joule": float(result.loss.total.mean().detach()),
                    "gradient_norm": float(first_grad.norm())
                    if update == 1
                    else float(_flat_parameters(step, gradients=True).norm()),
                    "nonzero_gradient_tensors": sum(
                        bool(gradient is not None and (gradient != 0).any()) for gradient in gradients
                    ),
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "adam_seconds": adam_seconds,
                    "full_iteration_seconds": time.perf_counter() - update_start,
                    "parameter_sha256": hashes[config.rank],
                    "replicas_equal": equal,
                }
            )
        if completed and failure is None:
            # Check the final checkpoint by replaying the same fixed query.
            with torch.no_grad():
                final_after = _cpu(module(positions, inertial, fixed_positions=pins).loss.total)
        torch.cuda.synchronize(device)
        training_elapsed_seconds = time.perf_counter() - start
        final_step_state = _cpu(step.state_dict())
        final_optimizer_state = _cpu(optimizer.state_dict())
        resume_replay = None
        if failure is None:
            replay_start = time.perf_counter()
            saved_first = torch.load(
                output / f"first_update_rank_{config.rank}.pt", map_location="cpu", weights_only=False
            )
            if saved_first["rank"] != config.rank or saved_first["optimizer_updates"] != 1:
                raise ValueError("serialized first-update checkpoint has the wrong rank or update count")
            step.load_state_dict(saved_first["step_state"])
            optimizer.load_state_dict(saved_first["optimizer_state"])
            torch.set_rng_state(saved_first["rng_state"]["torch"])
            torch.cuda.set_rng_state(saved_first["rng_state"]["cuda"], device)
            replay_losses = []
            for update in range(2, config.updates + 1):
                optimizer.zero_grad(set_to_none=True)
                replay_error = None
                try:
                    replay_output = module(positions, inertial, fixed_positions=pins)
                    replay_normalized = (replay_output.loss.total - before) / before.clamp_min(1.0)
                    if not torch.isfinite(replay_normalized).all().item():
                        raise ValueError("nonfinite replay objective")
                except Exception as exc:
                    replay_error = f"resume update {update} forward: {type(exc).__name__}: {exc}"
                errors = _all_ranks_ok(replay_error, device)
                if errors:
                    failure = {"stage": "resume_replay", "update": update, "errors": errors}
                    break
                replay_normalized.mean().backward()
                optimizer.step()
                replay_losses.append(float(replay_normalized.mean().detach()))
            if failure is None:
                weights_match = _states_equal(_cpu(step.state_dict()), final_step_state)
                optimizer_match = _states_equal(_cpu(optimizer.state_dict()), final_optimizer_state)
                losses_match = replay_losses == [row["normalized_loss"] for row in rows[1:]]
                local_mismatch = (
                    None
                    if weights_match and optimizer_match and losses_match
                    else (
                        f"resume mismatch: weights={weights_match}, optimizer={optimizer_match}, losses={losses_match}"
                    )
                )
                errors = _all_ranks_ok(local_mismatch, device)
                if errors:
                    failure = {"stage": "resume_replay", "update": config.updates, "errors": errors}
                resume_replay = {
                    "status": "passed" if errors is None else "failed",
                    "replayed_updates": max(0, config.updates - 1),
                    "weights_exact": weights_match,
                    "optimizer_exact": optimizer_match,
                    "losses_exact": losses_match,
                    "seconds": time.perf_counter() - replay_start,
                }
            step.load_state_dict(final_step_state)
            optimizer.load_state_dict(final_optimizer_state)
        torch.cuda.synchronize(device)
        result_report = {
            "status": "failed" if failure else "complete",
            "rank": config.rank,
            "world_size": config.world_size,
            "requested_updates": config.updates,
            "optimizer_updates": completed,
            "failure": failure,
            "physical_seeds": list(config.train_seeds),
            "validation_seeds": list(config.validation_seeds),
            "candidate_metadata": batch["metadata"],
            "initial_parameter_sha256": initial_hashes[config.rank],
            "parameter_count": sum(parameter.numel() for parameter in step.parameters()),
            "rows": rows,
            "resume_replay": resume_replay,
            "training_elapsed_seconds": training_elapsed_seconds,
            "elapsed_seconds": time.perf_counter() - start,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "device_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unknown")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "torch_version": torch.__version__,
            "working_dtype": "float32",
            "tf32_enabled": False,
            "autocast_enabled": False,
        }
        checkpoint = {
            "schema_version": 1,
            "config": asdict(config),
            "physics": {
                "cell_size": 0.025,
                "lame_lambda": 288461.53846,
                "lame_mu": 192307.69231,
                "density": 1000.0,
                "time_step": 1 / 300,
                "gravity": (0.0, -9.81, 0.0),
            },
            "network": {
                "state_feature_dim": 38,
                "hidden_dim": 128,
                "edge_hidden_dim": 64,
                "num_heads": 4,
                "hops": (1, 1, 1),
                "max_step_size": 0.05,
                "query_chunk_size": 128,
            },
            "learning_rate": 1e-4,
            "rank": config.rank,
            "optimizer_updates": completed,
            "input_batch": input_batch,
            "initial_step_state": initial_step_state,
            "initial_optimizer_state": initial_optimizer_state,
            "first_frames": first_frames,
            "first_before_per_sample": _cpu(before),
            "first_after_per_sample": first_after,
            "first_normalized_per_sample": first_normalized,
            "first_grad_flat": first_grad,
            "first_updated_parameters_flat": first_weights,
            "after_first_step_state": after_first_step_state,
            "after_first_optimizer_state": after_first_optimizer_state,
            "after_first_rng": after_first_rng,
            "final_step_state": final_step_state,
            "optimizer_state": final_optimizer_state,
            "final_after_per_sample": final_after,
        }
        torch.save(checkpoint, output / f"rank_{config.rank}.pt")
        _json(output / f"rank_{config.rank}.json", result_report)
        reports = [None] * config.world_size
        dist.all_gather_object(reports, result_report)
        if config.rank == 0:
            shared = {
                "schema_version": 1,
                "status": "complete" if all(row["status"] == "complete" for row in reports) else "failed",
                "config": asdict(config),
                "optimizer_updates": min(row["optimizer_updates"] for row in reports),
                "ranks": reports,
                "loss_definition": "mean((E_after - E_before.detach()) / max(E_before.detach(), 1 joule)) per rank; DDP averages rank gradients",
                "compute_partition": "Network, features, energy, Adam on each rank's exclusive GPU; native model sampling and PARDISO factorization/solves on CPU",
            }
            _json(output / "report.json", shared)
        dist.barrier()
        return result_report
    finally:
        dist.destroy_process_group()


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--cells", type=int, nargs=3, default=(10, 10, 40))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--fail-rank", type=int)
    parser.add_argument("--fail-update", type=int)
    args = parser.parse_args()
    config = ProbeConfig(
        rank=int(os.environ.get("RANK", "-1")),
        world_size=int(os.environ.get("WORLD_SIZE", "-1")),
        batch_size=args.batch_size,
        updates=args.updates,
        cell_counts=tuple(args.cells),
        seed=args.seed,
        fail_rank=args.fail_rank,
        fail_update=args.fail_update,
    )
    report = run_probe(args.output, config)
    print(
        json.dumps({"rank": config.rank, "status": report["status"], "optimizer_updates": report["optimizer_updates"]}),
        flush=True,
    )
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
