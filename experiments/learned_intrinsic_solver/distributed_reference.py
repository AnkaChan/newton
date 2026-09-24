# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay a distributed diagnostic on one CUDA device without distributed Torch.

The saved four rank microbatches are evaluated in sequence. Each local mean
loss contributes one quarter of the global loss, which reproduces the sample
weighting of distributed gradient averaging without an all-reduce. Frozen
first-step frames distinguish gradient differences from polar-SVD differences.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch  # noqa: TID253 -- Explicit opt-in PyTorch diagnostic.

from .data import generate_cuboid
from .network import IntrinsicSolverNetwork
from .solver_step import LearnedHexSolverStep

__all__ = ["run_reference"]


def _vector(values: torch.Tensor) -> torch.Tensor:
    """Move a tensor to a stable precision for comparison."""
    return values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Measure vector agreement without dividing by individual near-zero entries."""
    actual, expected = _vector(actual), _vector(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"comparison shape mismatch: {actual.shape} versus {expected.shape}")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {
            "finite": False,
            "count": actual.numel(),
            "actual_norm": None,
            "expected_norm": None,
            "error_norm": None,
            "relative_l2_error": None,
            "max_absolute_error": None,
            "cosine": None,
        }
    error = actual - expected
    actual_norm, expected_norm, error_norm = (float(x.norm()) for x in (actual, expected, error))
    scale = max(expected_norm, 1e-30)
    cosine = (
        float(torch.dot(actual, expected) / (actual_norm * expected_norm))
        if actual_norm and expected_norm
        else (1.0 if actual_norm == expected_norm else 0.0)
    )
    return {
        "finite": True,
        "count": actual.numel(),
        "actual_norm": actual_norm,
        "expected_norm": expected_norm,
        "error_norm": error_norm,
        "relative_l2_error": error_norm / scale,
        "max_absolute_error": float(error.abs().max()) if error.numel() else 0.0,
        "cosine": max(-1.0, min(1.0, cosine)),
    }


def _parameters_flat(network: IntrinsicSolverNetwork) -> torch.Tensor:
    """Flatten trainable parameters in optimizer registration order."""
    return torch.cat([parameter.detach().reshape(-1) for parameter in network.parameters()]).cpu()


def _gradients_flat(network: IntrinsicSolverNetwork) -> torch.Tensor:
    """Flatten all parameter gradients, rejecting an unexercised path."""
    missing = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    if missing:
        raise ValueError(f"missing gradients: {missing}")
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in network.parameters()]).cpu()


def _parameters_flat_from_state(state: dict, network: IntrinsicSolverNetwork) -> torch.Tensor:
    """Flatten checkpoint weights in live network parameter order."""
    return torch.cat([state[f"network.{name}"].reshape(-1) for name, _ in network.named_parameters()])


def _optimizer_moments(actual: dict, expected: dict) -> dict:
    """Compare Adam's first and second moments in parameter order."""
    if actual["state"].keys() != expected["state"].keys() or not _state_equal(
        actual["param_groups"], expected["param_groups"]
    ):
        raise ValueError("reference and DDP Adam states have different parameter registration or settings")
    return {
        name: _comparison(
            torch.cat([actual["state"][index][name].reshape(-1) for index in actual["state"]]),
            torch.cat([expected["state"][index][name].reshape(-1) for index in expected["state"]]),
        )
        for name in ("exp_avg", "exp_avg_sq")
    }


def _parameter_comparisons(network: IntrinsicSolverNetwork, expected: torch.Tensor, *, gradient: bool) -> dict:
    """Locate discrepancies within the flattened gradient or parameter vector."""
    offset, result = 0, {}
    expected = expected.reshape(-1)
    for name, parameter in network.named_parameters():
        count = parameter.numel()
        actual = parameter.grad if gradient else parameter.detach()
        if actual is None:
            raise ValueError(f"missing gradient for {name}")
        result[name] = _comparison(actual, expected[offset : offset + count])
        offset += count
    if offset != expected.numel():
        raise ValueError("saved flattened vector has the wrong parameter count")
    return result


def _build_step(config: dict, physics: dict, network_config: dict, device: torch.device) -> LearnedHexSolverStep:
    """Construct the physical step directly from saved experiment settings."""
    counts = tuple(config["cell_counts"])
    rest = generate_cuboid(counts, cell_size=float(physics["cell_size"]))
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    network = IntrinsicSolverNetwork(
        counts,
        int(network_config["state_feature_dim"]),
        hidden_dim=int(network_config["hidden_dim"]),
        edge_hidden_dim=int(network_config["edge_hidden_dim"]),
        num_heads=int(network_config["num_heads"]),
        hops=tuple(network_config["hops"]),
        max_step_size=float(network_config["max_step_size"]),
        query_chunk_size=int(network_config["query_chunk_size"]),
    ).to(device)
    return LearnedHexSolverStep(
        rest,
        fixed,
        lame_lambda=float(physics["lame_lambda"]),
        lame_mu=float(physics["lame_mu"]),
        density=float(physics["density"]),
        time_step=float(physics["time_step"]),
        network=network,
    )


def _on_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move only the saved physical query tensors to the reference device."""
    return tuple(batch[key].to(device) for key in ("positions", "inertial_prediction", "fixed_positions"))


def _state_equal(left, right) -> bool:
    """Check that all ranks saved identical synchronized checkpoint state."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _error_or_infinity(metric: dict, key: str) -> float:
    """Use infinity only inside pass checks when a metric is nonfinite."""
    value = metric[key]
    return math.inf if value is None else value


def run_reference(
    probe: Path,
    *,
    output: Path | None = None,
    overwrite: bool = False,
    device: str = "cuda:0",
    gradient_relative_tolerance: float = 1e-5,
    update_relative_tolerance: float = 1e-4,
) -> dict:
    """Replay the first global Adam step and final checkpoint inference.

    Args:
        probe: Directory containing the rank_R.pt files produced by the probe.
        output: JSON report path; defaults to reference.json within probe.
        overwrite: Permit replacing an existing report.
        device: One CUDA device used for the independent replay.
        gradient_relative_tolerance: Maximum global relative L2 gradient error.
        update_relative_tolerance: Maximum global relative L2 Adam update error.

    Returns:
        JSON-compatible diagnostics. A failed numerical check is reported in
        ``passed``; the command-line entry point also exits with status one.
    """
    probe = Path(probe)
    output = Path(output) if output is not None else probe / "reference.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    if any(
        not math.isfinite(value) or value <= 0 for value in (gradient_relative_tolerance, update_relative_tolerance)
    ):
        raise ValueError("relative tolerances must be finite and positive")
    reference_device = torch.device(device)
    if reference_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the numerical reference requires one available CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()

    first_path = probe / "rank_0.pt"
    rank_zero = torch.load(first_path, map_location="cpu", weights_only=False)
    config = rank_zero["config"]
    world_size = int(config["world_size"])
    if world_size != 4:
        raise ValueError(f"the diagnostic requires four ranks, got {world_size}")
    ranks = [rank_zero] + [
        torch.load(probe / f"rank_{rank}.pt", map_location="cpu", weights_only=False) for rank in range(1, world_size)
    ]
    shared_report = json.loads((probe / "report.json").read_text())
    if shared_report.get("status") != "complete" or shared_report.get("optimizer_updates") != config["updates"]:
        raise ValueError("the shared DDP report did not complete all requested updates")
    if len(shared_report.get("ranks", [])) != world_size:
        raise ValueError("the shared DDP report does not contain all ranks")
    rank_reports = [json.loads((probe / f"rank_{rank}.json").read_text()) for rank in range(world_size)]
    for rank, (local, shared) in enumerate(zip(rank_reports, shared_report["ranks"], strict=True)):
        replay = local.get("resume_replay")
        if local != shared or local.get("rank") != rank or local.get("status") != "complete":
            raise ValueError(f"rank {rank} DDP report is incomplete or differs from the shared report")
        if local.get("optimizer_updates") != config["updates"]:
            raise ValueError(f"rank {rank} did not finish all requested updates")
        if (
            not replay
            or replay.get("status") != "passed"
            or not all(replay.get(key) is True for key in ("weights_exact", "optimizer_exact", "losses_exact"))
        ):
            raise ValueError(f"rank {rank} serialized checkpoint resume replay did not pass")
        if replay.get("replayed_updates") != max(0, int(config["updates"]) - 1):
            raise ValueError(f"rank {rank} resumed the wrong number of updates")
    batch_sizes = [int(saved["input_batch"]["positions"].shape[0]) for saved in ranks]
    if len(set(batch_sizes)) != 1 or batch_sizes[0] != int(config["batch_size"]):
        raise ValueError(f"rank batch sizes differ from the configured size: {batch_sizes}")
    seeds = [seed for saved in ranks for seed in saved["input_batch"]["physical_seeds"]]
    if len(seeds) != sum(batch_sizes) or len(set(seeds)) != len(seeds):
        raise ValueError("physical seeds must be distinct across all saved queries")
    for rank, saved in enumerate(ranks):
        expected_config = config | {"rank": rank}
        if saved["config"] != expected_config or saved["rank"] != rank:
            raise ValueError(f"rank {rank} configuration or identity differs")
        if saved["physics"] != rank_zero["physics"] or saved["network"] != rank_zero["network"]:
            raise ValueError(f"rank {rank} physics or network settings differ")
        if saved["learning_rate"] != rank_zero["learning_rate"]:
            raise ValueError(f"rank {rank} learning rate differs")
        if saved["optimizer_updates"] != config["updates"]:
            raise ValueError(f"rank {rank} has only {saved['optimizer_updates']} completed updates")
    sample_count = sum(batch_sizes)
    device_batches = [_on_device(saved["input_batch"], reference_device) for saved in ranks]
    first_frames = [saved["first_frames"].to(reference_device) for saved in ranks]
    step = _build_step(config, rank_zero["physics"], rank_zero["network"], reference_device)
    step.load_state_dict(rank_zero["initial_step_state"], strict=True)
    step.network.train()
    optimizer = torch.optim.Adam(step.network.parameters(), lr=float(rank_zero["learning_rate"]))
    optimizer.load_state_dict(rank_zero["initial_optimizer_state"])
    initial_parameters = _parameters_flat(step.network)
    optimizer.zero_grad(set_to_none=True)

    first_rows = []
    first_loss_actual = 0.0
    cached_before = []
    for rank, saved in enumerate(ranks):
        positions, original_y, pins = device_batches[rank]
        if not torch.equal(positions[:, step.fixed_indices], pins):
            raise ValueError(f"rank {rank} candidate pins differ from saved prescribed positions")
        frames = first_frames[rank]
        with torch.autocast(device_type="cuda", enabled=False):
            before = step.energy(positions, original_y).total.detach()
            result = step(positions, original_y, fixed_positions=pins, frames=frames)
            after = result.loss.total
            normalized = (after - before) / before.clamp_min(1.0)
            (normalized.mean() / world_size).backward()
        cached_before.append(before)
        if not torch.equal(result.positions[:, step.fixed_indices], pins):
            raise ValueError(f"rank {rank} replay moved prescribed positions")
        first_loss_actual += float(normalized.mean().detach()) / world_size
        first_rows.append(
            {
                "rank": rank,
                "batch_size": batch_sizes[rank],
                "before_joule": _comparison(before, saved["first_before_per_sample"]),
                "after_joule": _comparison(after, saved["first_after_per_sample"]),
                "normalized_loss": _comparison(normalized, saved["first_normalized_per_sample"]),
            }
        )
        del result, after, normalized

    reference_gradient = _gradients_flat(step.network)
    gradient_rows = [_comparison(reference_gradient, saved["first_grad_flat"]) for saved in ranks]
    parameter_gradients = _parameter_comparisons(step.network, ranks[0]["first_grad_flat"], gradient=True)
    first_loss_expected = torch.cat([saved["first_normalized_per_sample"] for saved in ranks]).mean()
    optimizer.step()
    first_optimizer_moments = _optimizer_moments(optimizer.state_dict(), ranks[0]["after_first_optimizer_state"])
    updated_parameters = _parameters_flat(step.network)
    reference_update = _vector(updated_parameters) - _vector(initial_parameters)
    update_rows = []
    weight_rows = []
    for saved in ranks:
        saved_weights = saved["first_updated_parameters_flat"]
        update_rows.append(_comparison(reference_update, _vector(saved_weights) - _vector(initial_parameters)))
        weight_rows.append(_comparison(updated_parameters, saved_weights))
    parameter_updates = {}
    offset = 0
    for name, parameter in step.network.named_parameters():
        count = parameter.numel()
        parameter_updates[name] = _comparison(
            reference_update[offset : offset + count],
            _vector(ranks[0]["first_updated_parameters_flat"])[offset : offset + count]
            - _vector(initial_parameters)[offset : offset + count],
        )
        offset += count

    steady_step_seconds = []
    for _update in range(2, int(config["updates"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(reference_device)
        update_started = time.perf_counter()
        for (positions, original_y, pins), before in zip(device_batches, cached_before, strict=True):
            with torch.autocast(device_type="cuda", enabled=False):
                result = step(positions, original_y, fixed_positions=pins)
                normalized = (result.loss.total - before) / before.clamp_min(1.0)
                (normalized.mean() / world_size).backward()
            del result, normalized
        optimizer.step()
        torch.cuda.synchronize(reference_device)
        steady_step_seconds.append(time.perf_counter() - update_started)

    reference_final_parameters = _parameters_flat(step.network)
    final_optimizer_moments = _optimizer_moments(optimizer.state_dict(), ranks[0]["optimizer_state"])
    final_parameters = _parameters_flat_from_state(ranks[0]["final_step_state"], step.network)
    final_parameter_comparison = _comparison(reference_final_parameters, final_parameters)
    final_cumulative_update = _comparison(
        _vector(reference_final_parameters) - _vector(initial_parameters),
        _vector(final_parameters) - _vector(initial_parameters),
    )

    synchronized = all(
        _state_equal(saved["final_step_state"], ranks[0]["final_step_state"])
        and _state_equal(saved["optimizer_state"], ranks[0]["optimizer_state"])
        for saved in ranks[1:]
    )
    final_step = _build_step(config, rank_zero["physics"], rank_zero["network"], reference_device)
    final_step.load_state_dict(ranks[0]["final_step_state"], strict=True)
    final_step.network.eval()
    final_optimizer = torch.optim.Adam(final_step.network.parameters(), lr=float(rank_zero["learning_rate"]))
    final_optimizer.load_state_dict(ranks[0]["optimizer_state"])
    final_rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for rank, saved in enumerate(ranks):
            positions, original_y, pins = device_batches[rank]
            after = final_step(positions, original_y, fixed_positions=pins).loss.total
            final_rows.append({"rank": rank, "after_joule": _comparison(after, saved["final_after_per_sample"])})
    optimizer_steps = [
        int(state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"])
        for state in final_optimizer.state.values()
    ]
    expected_steps = int(config["updates"])
    first_finite = all(
        all(row[key]["finite"] for key in ("before_joule", "after_joule", "normalized_loss")) for row in first_rows
    )
    first_loss_max_error = max(_error_or_infinity(row["normalized_loss"], "max_absolute_error") for row in first_rows)
    final_max_relative_error = max(_error_or_infinity(row["after_joule"], "relative_l2_error") for row in final_rows)
    first_loss_gap = abs(first_loss_actual - float(first_loss_expected))
    reference_gradient_finite = bool(torch.isfinite(reference_gradient).all())
    reference_gradient_norm = float(reference_gradient.norm()) if reference_gradient_finite else None
    passed = bool(
        first_finite
        and reference_gradient.numel() > 0
        and reference_gradient_finite
        and reference_gradient_norm > 0
        and all(
            row["finite"] and _error_or_infinity(row, "relative_l2_error") <= gradient_relative_tolerance
            for row in gradient_rows
        )
        and all(
            row["finite"] and _error_or_infinity(row, "relative_l2_error") <= update_relative_tolerance
            for row in update_rows
        )
        and all(
            row["finite"] and _error_or_infinity(row, "relative_l2_error") <= 1e-5
            for row in (*first_optimizer_moments.values(), *final_optimizer_moments.values())
        )
        and first_loss_max_error <= 1e-6
        and first_loss_gap <= 1e-6
        and synchronized
        and final_max_relative_error <= 1e-6
        and final_cumulative_update["finite"]
        and _error_or_infinity(final_cumulative_update, "relative_l2_error") <= 1e-4
        and optimizer_steps
        and all(count == expected_steps for count in optimizer_steps)
    )
    worst_gradients = sorted(
        ((name, value["relative_l2_error"]) for name, value in parameter_gradients.items()),
        key=lambda item: math.inf if item[1] is None else item[1],
        reverse=True,
    )[:10]
    worst_updates = sorted(
        ((name, value["relative_l2_error"]) for name, value in parameter_updates.items()),
        key=lambda item: math.inf if item[1] is None else item[1],
        reverse=True,
    )[:10]
    report = {
        "passed": passed,
        "probe": str(probe),
        "device": str(reference_device),
        "sample_count": sample_count,
        "distinct_physical_seed_count": len(set(seeds)),
        "probe_status": shared_report["status"],
        "resume_replay_statuses": [row["resume_replay"]["status"] for row in rank_reports],
        "rank_batch_sizes": batch_sizes,
        "world_size": world_size,
        "expected_updates": expected_steps,
        "gradient_relative_tolerance": gradient_relative_tolerance,
        "update_relative_tolerance": update_relative_tolerance,
        "optimizer_moment_relative_tolerance": 1e-5,
        "first_loss_absolute_tolerance": 1e-6,
        "final_energy_relative_tolerance": 1e-6,
        "final_cumulative_update_relative_tolerance": 1e-4,
        "first_normalized_loss_ddp": float(first_loss_expected),
        "first_normalized_loss_reference": first_loss_actual if math.isfinite(first_loss_actual) else None,
        "first_rows": first_rows,
        "first_gradient": gradient_rows,
        "first_gradient_worst_parameters": worst_gradients,
        "first_gradient_norm": reference_gradient_norm,
        "first_updated_weights": weight_rows,
        "first_adam_update": update_rows,
        "first_adam_update_worst_parameters": worst_updates,
        "first_adam_moments": first_optimizer_moments,
        "single_gpu_steady_step_seconds": steady_step_seconds,
        "single_gpu_steady_mean_step_seconds": sum(steady_step_seconds) / len(steady_step_seconds)
        if steady_step_seconds
        else None,
        "final_parameter_values": final_parameter_comparison,
        "final_cumulative_update": final_cumulative_update,
        "final_adam_moments": final_optimizer_moments,
        "final_rank_states_identical": synchronized,
        "final_inference": final_rows,
        "final_optimizer_steps": {"minimum": min(optimizer_steps), "maximum": max(optimizer_steps)}
        if optimizer_steps
        else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gradient-relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--update-relative-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    report = run_reference(
        args.probe,
        output=args.output,
        overwrite=args.overwrite,
        device=args.device,
        gradient_relative_tolerance=args.gradient_relative_tolerance,
        update_relative_tolerance=args.update_relative_tolerance,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "sample_count": report["sample_count"],
                "gradient_relative_l2_error": max(
                    (row["relative_l2_error"] for row in report["first_gradient"]),
                    key=lambda value: math.inf if value is None else value,
                ),
                "adam_update_relative_l2_error": max(
                    (row["relative_l2_error"] for row in report["first_adam_update"]),
                    key=lambda value: math.inf if value is None else value,
                ),
                "final_rank_states_identical": report["final_rank_states_identical"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
