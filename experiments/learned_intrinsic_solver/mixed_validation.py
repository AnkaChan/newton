# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental fixed-seed validation with separate optimization and physical outcomes.

Every sample records, at every observation, the physical energy [J], the RMS
displacement [m], the Euclidean free-corner force residual norm [N] and the
inversion diagnostics of the cell-center deformation. Held-out queries receive
the same optimizer history the network was trained with (:mod:`history`).
Checkpoint selection uses the mean final residual of the frozen-problem
optimization phase and requires that no sample failed and every physical
trajectory survived. Inversion diagnostics are reported but never fail a
sample. ``validate_full_horizon`` evaluates a distinct held-out subset at the
full currently available K x H horizon and records its wall-clock cost.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from numbers import Integral

import numpy as np

__all__ = ["SELECTION_AGGREGATION", "validate", "validate_full_horizon", "validation_chunk"]

_NEAR_ZERO_ENERGY = 1e-8
SELECTION_AGGREGATION = "mean_final_free_force_residual_norm_n"
"""Checkpoint-selection metric: mean over all samples of the final optimization-phase residual [N]."""


def _positive_integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 1


def _new_sample(seed) -> dict:
    return {
        "seed": seed,
        "perturbation_scale": None,
        "candidate_mode": None,
        "energy_floor_joule": None,
        "energies": [],
        "displacements_rms_m": [],
        "free_force_residual_norm_n": [],
        "inverted_cell_counts": [],
        "min_center_jacobians": [],
        "displacement_rms": None,
        "error": None,
        "failure_iteration": None,
        "physical_steps": 0,
        "physical_error": None,
        "physical_failure_step": None,
        "physical_records": [],
    }


def _describe(samples, payloads) -> None:
    """Copy reset metadata that makes near-rest drift identifiable."""
    for sample, payload in zip(samples, payloads, strict=True):
        metadata = payload.get("metadata")
        scale = metadata.get("perturbation_scale") if isinstance(metadata, Mapping) else None
        sample["perturbation_scale"] = float(scale) if scale is not None else None
        sample["candidate_mode"] = payload.get("candidate_mode")


def _free_force_residual_norms(step, batch) -> list[float]:
    """Return the Euclidean free-corner force residual norm [N] of every candidate.

    Differentiates the implicit-Euler objective with respect to positions only,
    on a detached clone, with the physical-step start as the damping anchor.
    Rows of prescribed corners are zeroed; the norm is accumulated in float64.
    """
    import torch

    with torch.enable_grad():
        positions = batch["candidate"].detach().clone().requires_grad_(True)
        energy = step.energy(
            positions,
            batch["inertial_prediction"].detach(),
            batch["context_ids"],
            previous_positions=batch["physical_positions"].detach(),
        ).total
        gradient = torch.autograd.grad(energy.sum(), positions)[0]
    gradient = gradient.detach()
    gradient[:, step.fixed_indices] = 0
    if not torch.isfinite(gradient).all():
        raise ValueError("nonfinite free-corner force residual")
    return torch.linalg.vector_norm(gradient.flatten(1).double(), dim=1).cpu().tolist()


def _inversion_diagnostics(step, candidate) -> tuple[list[int], list[float]]:
    """Return per-sample inverted cell counts and minimum center Jacobians (diagnostics only)."""
    import torch

    from .features import center_deformation  # noqa: PLC0415 -- Optional training boundary.

    with torch.no_grad():
        deformation = center_deformation(candidate.detach(), step.cell_corner_indices, step.center_gradients)
        jacobian = torch.linalg.det(deformation)
    counts = (jacobian <= 0).sum(dim=1).cpu().tolist()
    minima = jacobian.min(dim=1).values.double().cpu().tolist()
    return [int(count) for count in counts], minima


def _rms_displacement(current, start) -> list[float]:
    return (current - start).double().square().sum(-1).mean(-1).sqrt().cpu().tolist()


def _record_iteration(step, batch, start, samples, energies) -> None:
    """Append one complete observation for every sample, including all diagnostics."""
    import torch

    if not torch.isfinite(energies).all():
        raise ValueError("nonfinite validation energy")
    values = energies.detach().cpu().tolist()
    residuals = _free_force_residual_norms(step, batch)
    inverted, minima = _inversion_diagnostics(step, batch["candidate"])
    displacement = _rms_displacement(batch["candidate"], start)
    for index, sample in enumerate(samples):
        sample["energies"].append(values[index])
        sample["displacements_rms_m"].append(displacement[index])
        sample["displacement_rms"] = displacement[index]
        sample["free_force_residual_norm_n"].append(residuals[index])
        sample["inverted_cell_counts"].append(inverted[index])
        sample["min_center_jacobians"].append(minima[index])


def _record_physical_step(step, batch, start, samples, energies, step_index) -> None:
    """Append the per-physical-step record after the inner updates and before advancing."""
    import torch

    if not torch.isfinite(energies).all():
        raise ValueError("nonfinite physical validation energy")
    values = energies.detach().cpu().tolist()
    residuals = _free_force_residual_norms(step, batch)
    inverted, minima = _inversion_diagnostics(step, batch["candidate"])
    displacement = _rms_displacement(batch["candidate"], start)
    for index, sample in enumerate(samples):
        sample["physical_records"].append(
            {
                "step": step_index,
                "energy_joule": values[index],
                "free_force_residual_norm_n": residuals[index],
                "displacement_rms_m": displacement[index],
                "inverted_cell_count": inverted[index],
                "min_center_jacobian": minima[index],
            }
        )


def _store_history(step, payloads, batch, result, device) -> None:
    """Write this query's history into the payloads and refresh the batch for the next query."""
    from . import history as history_module  # noqa: PLC0415 -- Optional training boundary.

    history_module.store_history(payloads, result)
    batch["history"] = history_module.batch_history(payloads, device, cell_count=len(step.cell_corner_indices))


def _optimization(step, factory, seeds, samples, config, device):
    from .train_mixed import _batch, _checked_forward  # noqa: PLC0415 -- Shared runtime acceptance boundary.

    payloads = []
    iteration = 0
    try:
        for seed in seeds:
            payloads.append(factory.reset(seed))
        _describe(samples, payloads)
        batch = _batch(payloads, device)
        start = batch["candidate"].clone()
        floors = step.energy_floor(batch["context_ids"]).detach().double().cpu().tolist()
        for sample, floor in zip(samples, floors, strict=True):
            sample["energy_floor_joule"] = floor
        energies = step.energy(
            start,
            batch["inertial_prediction"],
            batch["context_ids"],
            previous_positions=batch["physical_positions"],
        ).total
        _record_iteration(step, batch, start, samples, energies)
        for _ in range(config.validation_iterations):
            iteration += 1
            result = _checked_forward(step, step, batch)
            batch["candidate"] = result.positions.detach()
            _store_history(step, payloads, batch, result, device)
            _record_iteration(step, batch, start, samples, result.loss.total)
    except (RuntimeError, ValueError) as error:
        if len(seeds) > 1:
            raise
        samples[0]["error"] = str(error)
        samples[0]["failure_iteration"] = iteration
    finally:
        for payload in payloads:
            factory.retire(payload)


def _physical(step, factory, seeds, samples, device, *, iterations, physical_steps):
    """Run ``physical_steps`` physical steps of ``iterations`` learned queries per trajectory.

    Records one entry per completed physical step after its inner updates and
    before ``factory.advance``: total objective [J] of the last update, the
    free-corner residual norm [N] at the solved positions, the RMS displacement
    [m] from the trajectory's initial physical positions, and inversion
    diagnostics. Optimizer history is stored after every query and carried
    across physical steps by the factory.
    """
    from .train_mixed import _batch, _checked_forward  # noqa: PLC0415 -- Shared runtime acceptance boundary.

    if not _positive_integer(iterations) or not _positive_integer(physical_steps):
        raise ValueError("iterations and physical_steps must be positive integers")
    payloads = []
    completed = 0
    try:
        for seed in seeds:
            payloads.append(factory.reset(seed))
        _describe(samples, payloads)
        start = None
        for physical in range(physical_steps):
            batch = _batch(payloads, device)
            if start is None:
                start = batch["physical_positions"].clone()
            for _ in range(iterations):
                result = _checked_forward(step, step, batch)
                batch["candidate"] = result.positions.detach()
                _store_history(step, payloads, batch, result, device)
            _record_physical_step(step, batch, start, samples, result.loss.total, completed + 1)
            for index, payload in enumerate(payloads):
                payload["candidate"] = batch["candidate"][index].detach().cpu()
            completed += 1
            for sample in samples:
                sample["physical_steps"] = completed
            if physical + 1 < physical_steps:
                payloads = [factory.advance(payload) for payload in payloads]
    except (RuntimeError, ValueError) as error:
        if len(seeds) > 1:
            raise
        samples[0]["physical_error"] = str(error)
        samples[0]["physical_failure_step"] = completed + 1
    finally:
        for payload in payloads:
            factory.retire(payload)


def validation_chunk(step, factory, seeds, config, device):
    """Evaluate experimental fixed starts in a batch or retain an isolated failure.

    Optimization and physical rollouts each regenerate their initial payloads.
    A failed multi-member batch raises after cleanup so its caller can retry
    each seed independently. An isolated member retains all successful optimizer
    observations and attempts its physical rollout even if optimization failed.
    ``failure_iteration`` is zero for initialization failures and otherwise the
    one-based proposal index. ``physical_steps`` counts only completed steps.
    Every recorded iteration carries energy, displacement, residual norm and
    inversion diagnostics; ``physical_records`` holds one entry per completed
    physical step.
    """
    import torch

    samples = [_new_sample(seed) for seed in seeds]
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        _optimization(step, factory, seeds, samples, config, device)
        _physical(
            step,
            factory,
            seeds,
            samples,
            device,
            iterations=config.validation_physical_iterations,
            physical_steps=config.validation_physical_steps,
        )
    return samples


def _full_horizon_chunk(step, factory, seeds, device, *, iterations, physical_steps):
    import torch

    samples = [_new_sample(seed) for seed in seeds]
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        _physical(step, factory, seeds, samples, device, iterations=iterations, physical_steps=physical_steps)
    return samples


def _statistics(values, *, complete=True):
    if not complete or not values:
        return {"mean": None, "median": None, "max": None}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "max": float(np.max(values))}


def _inversion_summary(counts, minima, *, complete):
    """Summarize inversion diagnostics; the sample count is always reported over present samples."""
    return {
        "mean_inverted_cell_count": float(np.mean(counts)) if complete and counts else None,
        "max_inverted_cell_count": int(np.max(counts)) if complete and counts else None,
        "min_center_jacobian": float(np.min(minima)) if complete and minima else None,
        "inverted_sample_count": int(sum(count > 0 for count in counts)),
    }


def _iteration_curves(samples, iterations):
    residual_curve, inversion_curve = [], []
    for iteration in range(iterations + 1):
        present = [sample for sample in samples if len(sample["free_force_residual_norm_n"]) > iteration]
        missing = len(samples) - len(present)
        counts = {"valid_count": len(present), "failed_count": missing}
        residual_curve.append(
            {
                "iteration": iteration,
                **_statistics(
                    [sample["free_force_residual_norm_n"][iteration] for sample in present], complete=not missing
                ),
                **counts,
            }
        )
        inversion_curve.append(
            {
                "iteration": iteration,
                **_inversion_summary(
                    [sample["inverted_cell_counts"][iteration] for sample in present],
                    [sample["min_center_jacobians"][iteration] for sample in present],
                    complete=not missing,
                ),
                **counts,
            }
        )
    return residual_curve, inversion_curve


def _physical_curves(samples, physical_steps):
    curves = []
    for index in range(physical_steps):
        present = [sample for sample in samples if len(sample["physical_records"]) > index]
        missing = len(samples) - len(present)
        records = [sample["physical_records"][index] for sample in present]
        curves.append(
            {
                "step": index + 1,
                "valid_count": len(present),
                "failed_count": missing,
                "energy_joule": _statistics([record["energy_joule"] for record in records], complete=not missing),
                "free_force_residual_norm_n": _statistics(
                    [record["free_force_residual_norm_n"] for record in records], complete=not missing
                ),
                "displacement_rms_m": _statistics(
                    [record["displacement_rms_m"] for record in records], complete=not missing
                ),
                **_inversion_summary(
                    [record["inverted_cell_count"] for record in records],
                    [record["min_center_jacobian"] for record in records],
                    complete=not missing,
                ),
            }
        )
    return curves


def _selection(samples, iterations, *, failed_count, physical_survivors):
    """Build the checkpoint-selection record from the final optimization-phase residuals."""
    final = [
        sample["free_force_residual_norm_n"][iterations]
        for sample in samples
        if len(sample["free_force_residual_norm_n"]) > iterations
    ]
    complete = bool(samples) and len(final) == len(samples)
    return {
        "metric": float(np.mean(final)) if complete else None,
        "eligible": bool(complete and failed_count == 0 and physical_survivors == len(samples)),
        "aggregation": SELECTION_AGGREGATION,
        "survival_required": True,
    }


def _mean_first_update_loss(samples, *, increase_weight=1.0):
    """Return the mean LeCO first-update loss, or None when any sample lacks its first update.

    Uses the same increase weight as the trainer so the validation curve is
    comparable with the per-update training objective.
    """
    losses = []
    for sample in samples:
        energies = sample["energies"]
        floor = sample["energy_floor_joule"]
        if len(energies) < 2 or floor is None:
            return None
        before, after = energies[0], energies[1]
        scale = max(abs(before), floor)
        if not scale > 0:
            return None
        losses.append(math.asinh(after / scale) + increase_weight * max(0.0, (after - before) / scale))
    return float(np.mean(losses)) if losses else None


def _summarize(samples, config, *, seconds):
    iterations = config.validation_iterations
    failed = [sample for sample in samples if sample["error"] or sample["physical_error"]]
    optimization_failed = [sample for sample in samples if sample["error"]]
    physical_failed = [sample for sample in samples if sample["physical_error"]]
    near_zero = [sample for sample in samples if sample["energies"] and sample["energies"][0] <= _NEAR_ZERO_ENERGY]
    curves, absolute = [], []
    for iteration in range(iterations + 1):
        present = [sample for sample in samples if len(sample["energies"]) > iteration]
        missing = len(samples) - len(present)
        ratios = [
            sample["energies"][iteration] / sample["energies"][0]
            for sample in present
            if sample["energies"][0] > _NEAR_ZERO_ENERGY
        ]
        curves.append(
            {
                "iteration": iteration,
                **_statistics(ratios, complete=not missing),
                "mean_energy_joule": _statistics(
                    [sample["energies"][iteration] for sample in present], complete=not missing
                )["mean"],
                "valid_count": len(present),
                "failed_count": missing,
                "near_zero_count": len(near_zero),
                "relative_sample_count": len(ratios),
            }
        )
        small_present = [sample for sample in near_zero if len(sample["energies"]) > iteration]
        small_missing = len(near_zero) - len(small_present)
        energy = _statistics([sample["energies"][iteration] for sample in small_present], complete=not small_missing)
        displacement = _statistics(
            [sample["displacements_rms_m"][iteration] for sample in small_present], complete=not small_missing
        )
        residual = _statistics(
            [sample["free_force_residual_norm_n"][iteration] for sample in small_present], complete=not small_missing
        )
        absolute.append(
            {
                "iteration": iteration,
                "valid_count": len(small_present),
                "failed_count": small_missing,
                "mean_energy_joule": energy["mean"],
                "max_energy_joule": energy["max"],
                "mean_displacement_rms_m": displacement["mean"],
                "max_displacement_rms_m": displacement["max"],
                "mean_free_force_residual_norm_n": residual["mean"],
                "max_free_force_residual_norm_n": residual["max"],
            }
        )
    first = [sample for sample in samples if len(sample["energies"]) >= 2]
    before = np.asarray([sample["energies"][0] for sample in first])
    after = np.asarray([sample["energies"][1] for sample in first])
    first_complete = len(first) == len(samples) and bool(first)
    residual_curve, inversion_curve = _iteration_curves(samples, iterations)
    physical_curves = _physical_curves(samples, config.validation_physical_steps)
    survivors = sum(
        sample["physical_error"] is None and sample["physical_steps"] == config.validation_physical_steps
        for sample in samples
    )
    return {
        "sample_count": len(samples),
        "failed_count": len(failed),
        "optimization_failed_count": len(optimization_failed),
        "physical_failed_count": len(physical_failed),
        "first_update_failed_count": len(samples) - len(first),
        "failures": failed,
        "samples": samples,
        "mean_normalized_loss": _mean_first_update_loss(
            samples, increase_weight=getattr(config, "energy_increase_weight", 1.0)
        )
        if first_complete
        else None,
        "mean_before_joule": float(np.mean(before)) if first_complete else None,
        "mean_after_joule": float(np.mean(after)) if first_complete else None,
        "descent_rate": float(np.sum(after < before) / len(samples)) if samples else 0.0,
        "relative_energy": curves,
        "force_residual": residual_curve,
        "inversion": inversion_curve,
        "final_inverted_sample_count": inversion_curve[-1]["inverted_sample_count"],
        "selection": _selection(samples, iterations, failed_count=len(failed), physical_survivors=survivors),
        "near_zero": {
            "threshold_joule": _NEAR_ZERO_ENERGY,
            "sample_count": len(near_zero),
            "unclassified_count": sum(not sample["energies"] for sample in samples),
            "curves": absolute,
        },
        "physical_survivors": survivors,
        "physical_steps": config.validation_physical_steps,
        "physical_iterations": config.validation_physical_iterations,
        "physical_curves": physical_curves,
        "final_physical_residual": physical_curves[-1]["free_force_residual_norm_n"],
        "displacement_rms_mean": _statistics(
            [sample["displacement_rms"] for sample in samples if sample["displacement_rms"] is not None],
            complete=not optimization_failed,
        )["mean"],
        "seconds": seconds,
    }


def _summarize_full_horizon(samples, *, iterations, physical_steps, seconds):
    failed = [sample for sample in samples if sample["physical_error"]]
    survivors = sum(
        sample["physical_error"] is None and sample["physical_steps"] == physical_steps for sample in samples
    )
    curves = _physical_curves(samples, physical_steps)
    final = curves[-1]
    return {
        "sample_count": len(samples),
        "physical_survivors": survivors,
        "failed_count": len(failed),
        "iterations": iterations,
        "physical_steps": physical_steps,
        "final_free_force_residual_norm_n": final["free_force_residual_norm_n"],
        "final_energy_joule": final["energy_joule"],
        "final_physical_residual": final["free_force_residual_norm_n"],
        "final_inverted_sample_count": final["inverted_sample_count"],
        "physical_curves": curves,
        "samples": samples,
        "seconds": seconds,
    }


def _isolate_failures(seeds, batch_size, evaluate):
    """Evaluate seed groups, retrying the members of a failed group one at a time."""
    local = []
    for first in range(0, len(seeds), batch_size):
        group = seeds[first : first + batch_size]
        try:
            local.extend(evaluate(group))
        except (RuntimeError, ValueError):
            for seed in group:
                local.extend(evaluate([seed]))
    return local


def _gather(local, world_size):
    import torch.distributed as dist

    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    return sorted((sample for part in gathered for sample in part), key=lambda sample: sample["seed"])


def _full_horizon_seeds(config, rank, world_size) -> list[int]:
    """Return this rank's share of the fixed full-horizon subset, disjoint from the cheap seeds."""
    count = config.validation_full_count
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("validation_full_count must be a non-negative integer")
    first = config.validation_count
    return list(range(first + rank, first + count, world_size))


def validate(step, factory, config, device, rank, world_size):
    """Aggregate experimental fixed-seed validation without erasing earlier successes.

    Every sample reports Euclidean free-corner force-residual norms [N] at every
    iteration (``force_residual`` curve, iteration 0 = initial candidate) and
    inversion diagnostics (``inversion`` curve). ``selection`` carries the
    checkpoint-selection metric: the mean final residual over all samples,
    eligible only when no sample failed, every sample completed all iterations
    and every physical trajectory survived. ``mean_normalized_loss`` is the mean
    LeCO first-update loss ``asinh(E1/s) + relu((E1-E0)/s)`` with
    ``s = max(|E0|, energy_floor)``. Near-zero initial energies additionally
    report absolute energy [J], displacement [m] and residuals instead of
    ratios. ``physical_curves`` summarizes the cheap physical check per step.
    The residual differentiates the implicit-Euler energy with respect to
    positions only; network weights and their existing gradients are unchanged.
    """
    started = time.perf_counter()
    training = step.training
    step.eval()
    try:
        seeds = list(range(rank, config.validation_count, world_size))
        local = _isolate_failures(
            seeds, config.batch_size, lambda group: validation_chunk(step, factory, group, config, device)
        )
        samples = _gather(local, world_size)
        return _summarize(samples, config, seconds=time.perf_counter() - started)
    finally:
        step.train(training)


def validate_full_horizon(step, factory, config, device, rank, world_size, *, iterations, physical_steps):
    """Evaluate the fixed held-out subset at the full currently available K x H horizon.

    Seeds ``range(config.validation_count, config.validation_count +
    config.validation_full_count)`` are distinct from the cheap set and are
    distributed round-robin over ranks. Each trajectory runs ``physical_steps``
    physical steps of ``iterations`` learned queries and records, per step, the
    final energy [J], the final free-corner residual norm [N], the RMS
    displacement [m] and inversion diagnostics. Failed samples stay visible
    with their error and completed steps; final statistics are None when any
    sample is incomplete. ``seconds`` is this rank's wall-clock cost.

    Args:
        step: Mixed solver step in any mode; evaluation mode is restored afterwards.
        factory: Validation trajectory factory with ``reset``/``advance``/``retire``.
        config: Training configuration with ``validation_count``,
            ``validation_full_count`` and ``batch_size``.
        device: Device of the stacked batches.
        rank: This rank's index.
        world_size: Number of ranks sharing the subset.
        iterations: Learned queries per physical step (K), at least one.
        physical_steps: Physical steps per trajectory (H), at least one.

    Returns:
        Summary with ``sample_count``, ``physical_survivors``, ``failed_count``,
        ``iterations``, ``physical_steps``, ``final_free_force_residual_norm_n``,
        ``final_energy_joule`` (each ``{"mean", "median", "max"}``),
        ``final_physical_residual``, ``final_inverted_sample_count``,
        ``physical_curves``, ``samples`` and ``seconds``.

    Raises:
        ValueError: If ``iterations``, ``physical_steps`` or
            ``validation_full_count`` are not valid integers.
    """
    if not _positive_integer(iterations) or not _positive_integer(physical_steps):
        raise ValueError("iterations and physical_steps must be positive integers")
    started = time.perf_counter()
    seeds = _full_horizon_seeds(config, rank, world_size)
    training = step.training
    step.eval()
    try:
        local = _isolate_failures(
            seeds,
            config.batch_size,
            lambda group: _full_horizon_chunk(
                step, factory, group, device, iterations=iterations, physical_steps=physical_steps
            ),
        )
        samples = _gather(local, world_size)
        return _summarize_full_horizon(
            samples, iterations=iterations, physical_steps=physical_steps, seconds=time.perf_counter() - started
        )
    finally:
        step.train(training)
