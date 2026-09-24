# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental fixed-seed validation with separate optimization and physical outcomes."""

from __future__ import annotations

import numpy as np

__all__ = ["validate", "validation_chunk"]

_NEAR_ZERO_ENERGY = 1e-8


def _record_iteration(step, batch, start, samples, energies):
    """Append one complete observation, differentiating only near-zero physical energies."""
    import torch

    if not torch.isfinite(energies).all():
        raise ValueError("nonfinite validation energy")
    values = energies.detach().cpu().tolist()
    near_zero = [
        index
        for index, sample in enumerate(samples)
        if (sample["energies"][0] if sample["energies"] else values[index]) <= _NEAR_ZERO_ENERGY
    ]
    residuals = {}
    if near_zero:
        with torch.enable_grad():
            positions = batch["candidate"][near_zero].detach().clone().requires_grad_(True)
            physical_energy = step.energy(
                positions,
                batch["inertial_prediction"][near_zero].detach(),
                tuple(batch["context_ids"][index] for index in near_zero),
            ).total
            gradient = torch.autograd.grad(physical_energy.sum(), positions)[0]
        gradient[:, step.fixed_indices] = 0
        if not torch.isfinite(gradient).all():
            raise ValueError("nonfinite free-corner force residual")
        norms = torch.linalg.vector_norm(gradient.flatten(1).double(), dim=1).cpu().tolist()
        residuals = dict(zip(near_zero, norms, strict=True))
    displacement = (batch["candidate"] - start).double().square().sum(-1).mean(-1).sqrt().cpu().tolist()
    for index, sample in enumerate(samples):
        sample["energies"].append(values[index])
        sample["displacements_rms_m"].append(displacement[index])
        sample["displacement_rms"] = displacement[index]
        if index in residuals:
            sample["free_force_residual_norm_n"].append(residuals[index])


def _optimization(step, factory, seeds, samples, config, device):
    from .train_mixed import _batch, _checked_forward  # noqa: PLC0415 -- Shared runtime acceptance boundary.

    payloads = []
    iteration = 0
    try:
        for seed in seeds:
            payloads.append(factory.reset(seed))
        batch = _batch(payloads, device)
        start = batch["candidate"].clone()
        energies = step.energy(start, batch["inertial_prediction"], batch["context_ids"]).total
        _record_iteration(step, batch, start, samples, energies)
        for _ in range(config.validation_iterations):
            iteration += 1
            result = _checked_forward(step, step, batch)
            batch["candidate"] = result.positions.detach()
            _record_iteration(step, batch, start, samples, result.loss.total)
    except (RuntimeError, ValueError) as error:
        if len(seeds) > 1:
            raise
        samples[0]["error"] = str(error)
        samples[0]["failure_iteration"] = iteration
    finally:
        for payload in payloads:
            factory.retire(payload)


def _physical(step, factory, seeds, samples, config, device):
    from .train_mixed import _batch, _checked_forward  # noqa: PLC0415 -- Shared runtime acceptance boundary.

    payloads = []
    completed = 0
    try:
        for seed in seeds:
            payloads.append(factory.reset(seed))
        for physical in range(config.validation_physical_steps):
            batch = _batch(payloads, device)
            for _ in range(config.validation_physical_iterations):
                result = _checked_forward(step, step, batch)
                batch["candidate"] = result.positions.detach()
            for index, payload in enumerate(payloads):
                payload["candidate"] = batch["candidate"][index].detach().cpu()
            completed += 1
            for sample in samples:
                sample["physical_steps"] = completed
            if physical + 1 < config.validation_physical_steps:
                for index, payload in enumerate(payloads):
                    payloads[index] = factory.advance(payload)
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
    """
    import torch

    samples = [
        {
            "seed": seed,
            "energies": [],
            "displacements_rms_m": [],
            "free_force_residual_norm_n": [],
            "displacement_rms": None,
            "error": None,
            "failure_iteration": None,
            "physical_steps": 0,
            "physical_error": None,
            "physical_failure_step": None,
        }
        for seed in seeds
    ]
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        _optimization(step, factory, seeds, samples, config, device)
        _physical(step, factory, seeds, samples, config, device)
    return samples


def _statistics(values, *, complete=True):
    if not complete or not values:
        return {"mean": None, "median": None, "max": None}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "max": float(np.max(values))}


def _summarize(samples, config):
    failed = [sample for sample in samples if sample["error"] or sample["physical_error"]]
    optimization_failed = [sample for sample in samples if sample["error"]]
    physical_failed = [sample for sample in samples if sample["physical_error"]]
    near_zero = [sample for sample in samples if sample["energies"] and sample["energies"][0] <= _NEAR_ZERO_ENERGY]
    curves, absolute = [], []
    for iteration in range(config.validation_iterations + 1):
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
    return {
        "sample_count": len(samples),
        "failed_count": len(failed),
        "optimization_failed_count": len(optimization_failed),
        "physical_failed_count": len(physical_failed),
        "first_update_failed_count": len(samples) - len(first),
        "failures": failed,
        "samples": samples,
        "mean_normalized_loss": float(np.mean((after - before) / np.maximum(before, 1.0))) if first_complete else None,
        "mean_before_joule": float(np.mean(before)) if first_complete else None,
        "mean_after_joule": float(np.mean(after)) if first_complete else None,
        "descent_rate": float(np.sum(after < before) / len(samples)) if samples else 0.0,
        "relative_energy": curves,
        "near_zero": {
            "threshold_joule": _NEAR_ZERO_ENERGY,
            "sample_count": len(near_zero),
            "unclassified_count": sum(not sample["energies"] for sample in samples),
            "curves": absolute,
        },
        "physical_survivors": sum(
            sample["physical_error"] is None and sample["physical_steps"] == config.validation_physical_steps
            for sample in samples
        ),
        "physical_steps": config.validation_physical_steps,
        "physical_iterations": config.validation_physical_iterations,
        "displacement_rms_mean": _statistics(
            [sample["displacement_rms"] for sample in samples if sample["displacement_rms"] is not None],
            complete=not optimization_failed,
        )["mean"],
    }


def validate(step, factory, config, device, rank, world_size):
    """Aggregate experimental fixed-seed validation without erasing earlier successes.

    Near-zero initial energies have absolute energy [J], displacement [m], and
    Euclidean free-corner force-residual norms [N] instead of relative ratios.
    The residual differentiates the implicit-Euler energy with respect to
    positions only; network weights and their existing gradients are unchanged.
    """
    import torch.distributed as dist

    training = step.training
    step.eval()
    try:
        local = []
        seeds = list(range(rank, config.validation_count, world_size))
        for first in range(0, len(seeds), config.batch_size):
            group = seeds[first : first + config.batch_size]
            try:
                local.extend(validation_chunk(step, factory, group, config, device))
            except (RuntimeError, ValueError):
                for seed in group:
                    local.extend(validation_chunk(step, factory, [seed], config, device))
        gathered = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(gathered, local)
        else:
            gathered[0] = local
        samples = sorted((sample for part in gathered for sample in part), key=lambda sample: sample["seed"])
        return _summarize(samples, config)
    finally:
        step.train(training)
