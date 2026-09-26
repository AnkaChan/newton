# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Epoch training for the experimental one-step learned hex optimizer.

One epoch visits each physical training state once. Candidates are freshly and
statelessly perturbed by the dataset. The original inertial prediction remains
fixed for the one learned optimization query.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .train_smoke import TrainSmokeConfig, _resolve_cli_hops

__all__ = ["EpochTrainConfig", "run_training"]


@dataclass(frozen=True)
class EpochTrainConfig(TrainSmokeConfig):
    """Experimental epoch campaign settings; physical units follow TrainSmokeConfig."""

    train_count: int = 8192
    validation_count: int = 512
    cpu_threads: int = 2
    batch_size: int = 16
    min_epochs: int = 30
    max_epochs: int = 200
    early_stopping: bool = True
    """Allow a validation plateau to end the run before max_epochs."""

    def __post_init__(self):
        """Validate the global seed pools and epoch controls."""
        super().__post_init__()
        for name in ("batch_size", "min_epochs", "max_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs cannot exceed max_epochs")


def _cpu(value):
    """Detach checkpoint values from live device storage."""
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


def _atomic_torch(path, value):
    """Replace a checkpoint only after its serialization completes."""
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _all_ranks_ok(error, device, world_size):
    """Return all local errors after a coordinated gate."""
    if world_size == 1:
        return {"0": error} if error is not None else None
    from .distributed_probe import _all_ranks_ok  # noqa: PLC0415 - Optional training boundary.

    return _all_ranks_ok(error, device)


def _sum_metrics(local, device, world_size):
    """Reduce weighted query totals, then form complete-population metrics."""
    import torch
    import torch.distributed as dist

    keys = ("loss", "before", "after", "descent", "valid", "failed", "sample")
    values = torch.tensor([local[key] for key in keys], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    totals = dict(zip(keys, values.cpu().tolist(), strict=True))
    valid = int(totals["valid"])
    samples = int(totals["sample"])
    failed = int(totals["failed"])
    return {
        "mean_normalized_loss": totals["loss"] / valid if valid and not failed else None,
        "mean_before_joule": totals["before"] / valid if valid else None,
        "mean_after_joule": totals["after"] / valid if valid and not failed else None,
        "descent_rate": totals["descent"] / samples if samples else 0.0,
        "failed_count": failed,
        "valid_count": valid,
        "sample_count": samples,
    }


def _empty_totals():
    return dict.fromkeys(("loss", "before", "after", "descent", "valid", "failed", "sample"), 0.0)


def _add_values(totals, before, after):
    """Add per-query values without converting a mean into an unweighted mean."""
    import torch

    normalized = (after - before) / before.clamp_min(1.0)
    if not (torch.isfinite(before).all() and torch.isfinite(after).all() and torch.isfinite(normalized).all()):
        raise ValueError("nonfinite physical energy or normalized loss")
    count = before.numel()
    totals["loss"] += float(normalized.sum())
    totals["before"] += float(before.sum())
    totals["after"] += float(after.sum())
    totals["descent"] += int((after < before).sum())
    totals["valid"] += count
    totals["sample"] += count
    return normalized


def _batch_on_device(batch, device):
    return {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}


def _screen_output(step, result, pins):
    """Reject nonfinite learned proposals or moved pins without altering them.

    Inverted or collapsed cells are accepted (the stable energy is finite there);
    the returned Jacobian screen is a diagnostic only, never a failure condition.
    """
    import torch

    from .train_smoke import _screen  # noqa: PLC0415 - Optional training boundary.

    screen = _screen(SimpleNamespace(optimizer=step), result.positions)
    if not torch.isfinite(result.positions).all().item():
        raise ValueError(f"nonfinite learned output positions: {screen}")
    if not torch.isfinite(result.loss.total).all().item():
        raise ValueError(f"nonfinite learned output energy: {screen}")
    if not torch.equal(result.positions[:, step.fixed_indices], pins):
        raise ValueError("learned output moved prescribed pins")
    return screen


def _evaluate_one(step, batch, totals):
    """Evaluate one or more validation queries with the unwrapped step."""
    import torch

    positions, inertial, pins, previous = (
        batch[key] for key in ("positions", "inertial_prediction", "fixed_positions", "previous_positions")
    )
    before = step.energy(positions, inertial, previous_positions=previous).total.detach()
    if not torch.isfinite(before).all().item():
        raise ValueError("nonfinite initial physical energy")
    result = step(positions, inertial, fixed_positions=pins, previous_positions=previous)
    _screen_output(step, result, pins)
    _add_values(totals, before, result.loss.total.detach())


def _validate(step, dataset, batch_size, device, world_size):
    """Count failures against every fixed validation query."""
    import torch

    totals = _empty_totals()
    failures = []
    step.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for cpu_batch in dataset.validation_batches(batch_size):
            try:
                batch = _batch_on_device(cpu_batch, device)
                _evaluate_one(step, batch, totals)
            except (ValueError, RuntimeError) as batch_error:
                # A bad item must not hide valid items in the same batch.
                for index, seed in enumerate(cpu_batch["physical_seeds"]):
                    single = {
                        key: value[index : index + 1]
                        if key in ("positions", "inertial_prediction", "fixed_positions", "previous_positions")
                        else value
                        for key, value in cpu_batch.items()
                    }
                    try:
                        single = _batch_on_device(single, device)
                        _evaluate_one(step, single, totals)
                    except (ValueError, RuntimeError) as error:
                        totals["failed"] += 1
                        totals["sample"] += 1
                        failures.append({"physical_seed": seed, "error": str(error)})
                if not cpu_batch["physical_seeds"]:
                    raise batch_error
    result = _sum_metrics(totals, device, world_size)
    if world_size > 1:
        import torch.distributed as dist

        gathered = [None] * world_size
        dist.all_gather_object(gathered, failures)
        failures = [failure for rank_failures in gathered for failure in rank_failures]
    result["failures"] = failures
    step.train()
    return result


def _config_for_resume(config):
    values = asdict(config)
    for key in ("max_epochs", "early_stopping", "verbose"):
        values.pop(key)
    return values


def _rank_state(rank, device, dataset):
    """Capture small rank-owned state for an epoch-boundary restart."""
    import torch

    return {
        "rank": rank,
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),  # noqa: NPY002 - Resume legacy global users.
        "python_rng_state": random.getstate(),
        "dataset_identity": dataset.dataset_identity,
        "dataset_state": dataset.state_dict(),
    }


def _gather_rank_states(rank, world_size, device, dataset):
    state = _rank_state(rank, device, dataset)
    if world_size == 1:
        return [state]
    import torch.distributed as dist

    states = [None] * world_size
    dist.all_gather_object(states, state)
    return states


def _restore_rank_state(state, device):
    import torch

    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    np.random.set_state(state["numpy_rng_state"])  # noqa: NPY002 - Restore exact checkpoint state.
    random.setstate(state["python_rng_state"])


def _run(output, config, resume, rank, world_size, device):
    import torch
    from torch.nn.parallel import DistributedDataParallel

    from .data import generate_cuboid  # noqa: PLC0415 - Optional training boundary.
    from .distributed_probe import _replicas_equal  # noqa: PLC0415
    from .epoch_data import EpochDataset  # noqa: PLC0415
    from .features import CONDITIONING_DIM, STATE_FEATURE_DIM  # noqa: PLC0415
    from .network import IntrinsicSolverNetwork  # noqa: PLC0415
    from .newton_model import build_newton_hex_model  # noqa: PLC0415
    from .newton_solver import SolverLearnedIntrinsic  # noqa: PLC0415
    from .training_report import write_report  # noqa: PLC0415
    from .training_schedule import PlateauController  # noqa: PLC0415

    start = time.perf_counter()
    checks = output / "checkpoints"
    data_dir = output / "data"
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if saved:
        if saved.get("status") == "failed":
            raise ValueError("failure checkpoints are diagnostic")
        if _config_for_resume(config) != _config_for_resume(EpochTrainConfig(**saved["config"])):
            raise ValueError("resume configuration differs beyond max_epochs/early_stopping/verbosity")
        if saved["world_size"] != world_size:
            raise ValueError("resume world size differs")
        if config.max_epochs < saved["completed_epochs"]:
            raise ValueError("max_epochs precedes completed epochs")
        if not data_dir.is_dir():
            raise ValueError("resume requires the original output data directory")
    else:
        conflict = rank == 0 and any(
            (output / name).exists() for name in ("checkpoints", "data", "report.json", "training.csv")
        )
        errors = _all_ranks_ok(
            "fresh training output already contains training artifacts" if conflict else None, device, world_size
        )
        if errors:
            raise FileExistsError(str(errors))
    output.mkdir(parents=True, exist_ok=True)
    checks.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    if rank == 0:
        print("phase=dataset_preparation", flush=True)
    if rank == 0 and not saved:
        write_report(
            output,
            {
                "status": "preparing",
                "config": asdict(config),
                "completed_epochs": 0,
                "optimizer_updates": 0,
                "train_count": config.train_count,
                "validation_count": config.validation_count,
                "world_size": world_size,
                "batch_size": config.batch_size,
                "global_batch_size": config.batch_size * world_size,
                "parameter_count": 0,
                "elapsed_seconds": 0.0,
                "history": [],
                "validation_initial": None,
            },
        )
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
    ).to(device)
    # The network constructor deliberately leaves the correction head at zero.
    solver = SolverLearnedIntrinsic(model, network=network, iterations=1)
    dataset = None
    prep_error = None
    try:
        dataset = EpochDataset(
            config,
            rest,
            model,
            solver,
            rank=rank,
            world_size=world_size,
            dataset_dir=data_dir,
            resume=bool(saved),
        )
        if saved and dataset.dataset_identity != saved["rank_states"][rank]["dataset_identity"]:
            raise ValueError("resume dataset identity differs")
    except Exception as error:
        prep_error = f"{type(error).__name__}: {error}"
    prep_errors = _all_ranks_ok(prep_error, device, world_size)
    if prep_errors:
        raise ValueError(f"dataset preparation failed: {prep_errors}")
    step = dataset.step
    module = (
        DistributedDataParallel(
            step,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if world_size > 1
        else step
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=config.learning_rate)
    controller = PlateauController(
        learning_rate=config.learning_rate,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
    )
    completed = saved["completed_epochs"] if saved else 0
    updates = saved["optimizer_updates"] if saved else 0
    history = copy.deepcopy(saved["history"]) if saved else []
    initial_validation = copy.deepcopy(saved["validation_initial"]) if saved else None
    best_loss = saved["best_validation_loss"] if saved else math.inf
    elapsed_prior = saved["elapsed_seconds"] if saved else 0.0
    status = "running"
    failure = None
    current_batch = None
    if saved:
        step.load_state_dict(saved["step_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        controller.load_state_dict(saved["controller_state"])
        for group in optimizer.param_groups:
            group["lr"] = controller.learning_rate
        _restore_rank_state(saved["rank_states"][rank], device)
    parameter_count = sum(parameter.numel() for parameter in step.parameters())

    def report():
        return {
            "schema_version": 1,
            "status": status,
            "config": asdict(config),
            "completed_epochs": completed,
            "optimizer_updates": updates,
            "train_count": config.train_count,
            "validation_count": config.validation_count,
            "world_size": world_size,
            "batch_size": config.batch_size,
            "global_batch_size": config.batch_size * world_size,
            "parameter_count": parameter_count,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - start,
            "history": history,
            "validation_initial": initial_validation,
            "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
            "failure": failure,
            "train_seeds": list(config.train_seeds),
            "validation_seeds": list(config.validation_seeds),
            "working_dtype": "float32",
            "tf32_enabled": False,
            "autocast_enabled": False,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }

    def checkpoint(name):
        states = _gather_rank_states(rank, world_size, device, dataset)
        if rank == 0:
            payload = {
                "schema_version": 1,
                "status": status,
                "config": asdict(config),
                "world_size": world_size,
                "completed_epochs": completed,
                "optimizer_updates": updates,
                "step_state": _cpu(step.state_dict()),
                "optimizer_state": _cpu(optimizer.state_dict()),
                "controller_state": copy.deepcopy(controller.state_dict()),
                "rank_states": states,
                "history": copy.deepcopy(history),
                "validation_initial": copy.deepcopy(initial_validation),
                "best_validation_loss": best_loss,
                "elapsed_seconds": report()["elapsed_seconds"],
                "failure": failure,
            }
            _atomic_torch(checks / name, payload)

    def save_report():
        if rank == 0:
            write_report(output, report())

    if not saved:
        initial_validation = _validate(step, dataset, config.batch_size, device, world_size)
        if initial_validation["sample_count"] != config.validation_count:
            raise ValueError("initial validation query count differs from configured count")
        if initial_validation["failed_count"] == 0:
            best_loss = initial_validation["mean_normalized_loss"]
            controller.best_loss = best_loss
        checkpoint("initial.pt")
        if best_loss < math.inf:
            checkpoint("best_validation.pt")
    save_report()
    if rank == 0:
        print(f"phase=training_ready epoch={completed} updates={updates}", flush=True)
    while completed < config.max_epochs:
        epoch = completed + 1
        epoch_start = time.perf_counter()
        used_lr = optimizer.param_groups[0]["lr"]
        train_totals = _empty_totals()
        batch_count = 0
        batches = iter(dataset.training_batches(epoch, config.batch_size))
        local_batch_count = math.ceil(len(dataset.train_seeds) / config.batch_size)
        for _ in range(local_batch_count):
            batch_count += 1
            current_batch = None
            cpu_batch = None
            sample_error = None
            try:
                cpu_batch = next(batches)
            except (StopIteration, ValueError, RuntimeError) as error:
                sample_error = f"epoch {epoch} batch {batch_count} sampling: {type(error).__name__}: {error}"
            errors = _all_ranks_ok(sample_error, device, world_size)
            if errors:
                failure = {"stage": "before_forward", "epoch": epoch, "batch": batch_count, "errors": errors}
                break
            current_batch = cpu_batch
            transfer_error = None
            try:
                batch = _batch_on_device(cpu_batch, device)
                positions, inertial, pins, previous = (
                    batch[key] for key in ("positions", "inertial_prediction", "fixed_positions", "previous_positions")
                )
            except (ValueError, RuntimeError) as error:
                transfer_error = f"epoch {epoch} batch {batch_count} transfer: {type(error).__name__}: {error}"
            errors = _all_ranks_ok(transfer_error, device, world_size)
            if errors:
                failure = {"stage": "before_forward", "epoch": epoch, "batch": batch_count, "errors": errors}
                break
            optimizer.zero_grad(set_to_none=True)
            local_error = None
            before = after = normalized = None
            try:
                with torch.no_grad():
                    before = step.energy(positions, inertial, previous_positions=previous).total.detach()
                if not torch.isfinite(before).all().item():
                    raise ValueError("nonfinite initial physical energy")
                with torch.autocast(device_type=device.type, enabled=False):
                    result = module(positions, inertial, fixed_positions=pins, previous_positions=previous)
                    _screen_output(step, result, pins)
                    after = result.loss.total
                    normalized = (after - before) / before.clamp_min(1.0)
                if not torch.isfinite(normalized).all().item():
                    raise ValueError("nonfinite normalized loss")
            except (ValueError, RuntimeError) as error:
                local_error = f"epoch {epoch} batch {batch_count} forward: {type(error).__name__}: {error}"
            errors = _all_ranks_ok(local_error, device, world_size)
            if errors:
                failure = {"stage": "before_backward", "epoch": epoch, "batch": batch_count, "errors": errors}
                break
            local_error = None
            try:
                normalized.mean().backward()
                gradients = [parameter.grad for parameter in step.parameters()]
                if all(gradient is None for gradient in gradients) or any(
                    gradient is not None and not torch.isfinite(gradient).all().item() for gradient in gradients
                ):
                    raise ValueError("nonfinite or absent parameter gradients")
            except (ValueError, RuntimeError) as error:
                local_error = f"epoch {epoch} batch {batch_count} backward: {type(error).__name__}: {error}"
            errors = _all_ranks_ok(local_error, device, world_size)
            if errors:
                failure = {"stage": "before_adam", "epoch": epoch, "batch": batch_count, "errors": errors}
                break
            optimizer.step()
            updates += 1
            local_error = None
            if any(not torch.isfinite(parameter).all().item() for parameter in step.parameters()):
                local_error = f"epoch {epoch} batch {batch_count}: Adam produced nonfinite parameters"
            errors = _all_ranks_ok(local_error, device, world_size)
            if errors:
                failure = {"stage": "after_adam", "epoch": epoch, "batch": batch_count, "errors": errors}
                break
            _add_values(train_totals, before, after.detach())
            if rank == 0 and config.verbose and (batch_count % 16 == 0 or batch_count == 1):
                print(
                    f"epoch={epoch} batch={batch_count} updates={updates} loss={float(normalized.detach().mean()):.6g}",
                    flush=True,
                )
        if failure:
            status = "failed"
            if current_batch is not None:
                _atomic_torch(output / f"failure_input_rank_{rank}.pt", _cpu(current_batch))
            checkpoint("failure.pt")
            save_report()
            break
        extra_error = None
        try:
            if next(batches, None) is not None:
                extra_error = f"epoch {epoch}: sampler yielded more batches than expected"
        except (ValueError, RuntimeError) as error:
            extra_error = f"epoch {epoch} sampler exhaustion: {type(error).__name__}: {error}"
        errors = _all_ranks_ok(extra_error, device, world_size)
        if errors:
            failure = {"stage": "sampler_count", "epoch": epoch, "errors": errors}
            status = "failed"
            checkpoint("failure.pt")
            save_report()
            break
        train_metrics = _sum_metrics(train_totals, device, world_size)
        if train_metrics["sample_count"] != config.train_count:
            raise ValueError("training epoch did not visit every physical seed exactly once")
        validation = _validate(step, dataset, config.batch_size, device, world_size)
        if validation["sample_count"] != config.validation_count:
            raise ValueError("validation query count differs from configured count")
        schedule = controller.observe(epoch, validation, allow_early_stop=config.early_stopping)
        for group in optimizer.param_groups:
            group["lr"] = schedule["learning_rate"]
        completed = epoch
        status = schedule["status"] if schedule["stop"] else "running"
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "optimizer_updates": updates,
                "learning_rate": used_lr,
                "next_learning_rate": schedule["learning_rate"],
                "train": train_metrics,
                "validation": validation,
                "elapsed_seconds": elapsed_prior + time.perf_counter() - start,
                "epoch_seconds": epoch_seconds,
            }
        )
        if world_size > 1:
            equal, hashes = _replicas_equal(step, world_size, device)
            if not equal:
                failure = {"stage": "replica_check", "epoch": epoch, "hashes": hashes}
                status = "failed"
        if failure:
            checkpoint("failure.pt")
            save_report()
            break
        if validation["failed_count"] == 0 and validation["mean_normalized_loss"] < best_loss:
            best_loss = validation["mean_normalized_loss"]
            checkpoint("best_validation.pt")
        checkpoint("latest.pt")
        if epoch % 10 == 0:
            checkpoint(f"epoch_{epoch:04d}.pt")
        save_report()
        if rank == 0:
            print(
                f"epoch={epoch} train_loss={train_metrics['mean_normalized_loss']} validation_loss={validation['mean_normalized_loss']} descent={validation['descent_rate']:.3f} failures={validation['failed_count']} status={status}",
                flush=True,
            )
        if schedule["stop"]:
            break
    if status != "failed":
        if status == "running":
            status = "epoch_limit"
        checkpoint("final.pt")
        save_report()
    return report()


def run_training(output: Path, config: EpochTrainConfig, *, resume: Path | None = None) -> dict:
    """Train one-step optimization queries and persist epoch-boundary state.

    CUDA ranks use the RANK/WORLD_SIZE/LOCAL_RANK environment from the launcher.
    An unlaunched CPU call runs as one rank for small deterministic tests.
    """
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in (1, 2, 4) or not 0 <= rank < world_size:
        raise ValueError("world size must be 1, 2, or 4 with a valid rank")
    if config.train_count % world_size:
        raise ValueError("train_count must divide evenly across ranks")
    if config.train_count // world_size < 1:
        raise ValueError("each rank needs a physical training state")
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda":
        if torch.cuda.device_count() != 1 or local_rank != 0:
            raise RuntimeError("each CUDA rank must claim exactly one device as cuda:0")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    elif world_size > 1:
        raise ValueError("distributed training requires CUDA")
    old_threads = torch.get_num_threads()
    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    old_numpy_rng = np.random.get_state()  # noqa: NPY002 - Preserve caller's legacy global state.
    old_python_rng = random.getstate()
    devices = [0] if device.type == "cuda" else []
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)  # noqa: NPY002 - Deterministic legacy consumers.
            random.seed(config.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(config.seed)
            if world_size > 1:
                dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
            try:
                return _run(Path(output), config, Path(resume) if resume else None, rank, world_size, device)
            finally:
                if world_size > 1:
                    dist.destroy_process_group()
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        np.random.set_state(old_numpy_rng)  # noqa: NPY002 - Restore caller's state.
        random.setstate(old_python_rng)


def _main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-count", type=int, default=8192)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument(
        "--no-early-stopping", action="store_true", help="Train until max-epochs despite a loss plateau"
    )
    parser.add_argument("--cells", type=int, nargs=3, default=(10, 10, 40))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--hops", nargs="+", type=int, help="Hop distance per block; default: saved hops on resume, otherwise 1"
    )
    parser.add_argument("--max-step-size", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    config = EpochTrainConfig(
        batch_size=args.batch_size,
        train_count=args.train_count,
        validation_count=args.validation_count,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        early_stopping=not args.no_early_stopping,
        cell_counts=tuple(args.cells),
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        hops=_resolve_cli_hops(args.hops, args.resume),
        max_step_size=args.max_step_size,
        verbose=not args.quiet,
    )
    result = run_training(args.output, config, resume=args.resume)
    print(json.dumps({key: result[key] for key in ("status", "completed_epochs", "optimizer_updates")}), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
