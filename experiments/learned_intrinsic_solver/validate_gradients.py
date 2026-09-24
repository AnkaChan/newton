# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce finite-difference and float32/reference checks for the learned step.

Run with --full-grid to include a 10x10x40 forward/backward smoke test. No model
is trained or saved. Float64 runs are explicitly numerical references; the
working float32 path includes geometry, sparse factorization, and derivatives.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import generate_cuboid
from .multiscale import generate_multiscale
from .network import IntrinsicSolverNetwork
from .solver_step import LearnedHexSolverStep

__all__ = ["validate_gradients"]


def _fixture(dtype):
    import torch

    rest = generate_cuboid((2, 2, 3), cell_size=0.1)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
    # Create the same float32 data/parameters before casting the reference copy.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        model = IntrinsicSolverNetwork(rest.cell_counts, 38, hidden_dim=16, edge_hidden_dim=8)
        with torch.no_grad():
            model.correction_head.weight.normal_(std=0.004)
            for layer in model.layers:
                layer.film.weight.normal_(std=0.01)
    step = LearnedHexSolverStep(
        rest,
        fixed,
        lame_lambda=1000.0 * 0.3 / (1.3 * 0.4),
        lame_mu=1000.0 / 2.6,
        density=100.0,
        time_step=0.04,
        network=model.to(dtype=dtype),
        dtype=dtype,
    )
    x = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
    z = x[..., 2].clone()
    x[..., 0] += 0.08 * z.square() + 0.001 * torch.sin(21 * x[..., 1]) * z / 0.3
    x[..., 1] += 0.02 * z.square()
    y = x + torch.tensor([0.0003, -0.001, 0.0001])
    return step, x.to(dtype), y.to(dtype)


def _comparison(analytical: float, numerical: float) -> dict:
    absolute = abs(analytical - numerical)
    return {
        "autograd": analytical,
        "central_difference": numerical,
        "absolute_error": absolute,
        "relative_error": absolute / max(abs(analytical), abs(numerical), 1e-12),
    }


def _direction(shape, seed, dtype):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    return value / value.norm()


def _check_dtype(dtype):
    import torch

    step, x, y = _fixture(dtype)
    float32 = dtype == torch.float32
    results = {}
    saved_gradients = {}

    variable_x = x.clone().requires_grad_()
    energy = step.energy(variable_x, y).total.sum()
    gradient = torch.autograd.grad(energy, variable_x)[0]
    direction = _direction(x.shape, 11, dtype)
    analytical = (gradient * direction).sum().item()
    results["hex_energy_positions"] = []
    for epsilon in [1e-3, 3e-4, 1e-4] if float32 else [1e-5, 3e-6, 1e-6]:
        with torch.no_grad():
            numerical = (
                (step.energy(x + epsilon * direction, y).total - step.energy(x - epsilon * direction, y).total).sum()
                / (2 * epsilon)
            ).item()
        results["hex_energy_positions"].append({"epsilon_m": epsilon, **_comparison(analytical, numerical)})
    saved_gradients["energy"] = gradient.detach().double().numpy()

    delta = (0.01 * _direction((1, 12, 3, 3), 15, dtype)).requires_grad_()
    probe = _direction(x.shape, 19, dtype)
    fixed_positions = x[:, step.fixed_indices]
    fused = step.fusion.fuse(x, delta, fixed_positions)
    linear_loss = ((fused - x) * probe).sum()
    gradient = torch.autograd.grad(linear_loss, delta)[0]
    direction = _direction(delta.shape, 23, dtype)
    analytical = (gradient * direction).sum().item()
    results["fusion_targets"] = []
    for epsilon in [1e-2, 3e-3, 1e-3] if float32 else [1e-4, 1e-5, 1e-6]:
        with torch.no_grad():
            plus = step.fusion.fuse(x, delta + epsilon * direction, fixed_positions)
            minus = step.fusion.fuse(x, delta - epsilon * direction, fixed_positions)
            numerical = (((plus - minus) * probe).sum() / (2 * epsilon)).item()
        results["fusion_targets"].append({"epsilon_dimensionless": epsilon, **_comparison(analytical, numerical)})
    saved_gradients["fusion"] = gradient.detach().double().numpy()

    parameter = step.network.correction_head.bias
    loss = step(x, y).loss.total.sum()
    gradient = torch.autograd.grad(loss, parameter)[0]
    direction = _direction(parameter.shape, 29, dtype)
    analytical = (gradient * direction).sum().item()
    original = parameter.detach().clone()
    results["network_through_fusion_and_energy"] = []
    with torch.no_grad():
        for epsilon in [3e-3, 1e-3, 3e-4] if float32 else [1e-4, 1e-5, 1e-6]:
            parameter.copy_(original + epsilon * direction)
            plus = step(x, y).loss.total.sum().item()
            parameter.copy_(original - epsilon * direction)
            minus = step(x, y).loss.total.sum().item()
            results["network_through_fusion_and_energy"].append(
                {"epsilon_parameter": epsilon, **_comparison(analytical, (plus - minus) / (2 * epsilon))}
            )
        parameter.copy_(original)
        before = step(x, y).loss.total.item()
        parameter.copy_(original - 1e-3 * gradient / gradient.norm())
        after = step(x, y).loss.total.item()
        parameter.copy_(original)
    results["small_gradient_descent_check"] = {
        "before_joule": before,
        "after_joule": after,
        "decreased": after < before,
    }
    saved_gradients["network"] = gradient.detach().double().numpy()
    return results, saved_gradients


def _full_grid() -> dict:
    import torch

    start = time.perf_counter()
    rest = generate_cuboid((10, 10, 40), cell_size=0.025)
    sample = generate_multiscale(rest, seed=0, strength=0.2)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        step = LearnedHexSolverStep(
            rest,
            fixed,
            lame_lambda=5e5 * 0.3 / (1.3 * 0.4),
            lame_mu=5e5 / 2.6,
            density=1000,
            time_step=1 / 300,
        )
        # A nonzero head exercises backward through all attention blocks.
        with torch.no_grad():
            step.network.correction_head.weight.normal_(std=0.0001)
    x = torch.tensor(sample.positions, dtype=torch.float32)[None]
    acceleration = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float32)
    y = x + acceleration / 300**2
    result = step(x, y)
    result.loss.total.mean().backward()
    gradients = [p.grad for p in step.network.parameters()]
    finite = all(g is not None and torch.isfinite(g).all().item() for g in gradients)
    return {
        "cell_counts": [10, 10, 40],
        "cell_count": 4000,
        "corner_count": x.shape[1],
        "dtype": str(result.positions.dtype),
        "elapsed_seconds": time.perf_counter() - start,
        "loss_joule": result.loss.total.item(),
        "elastic_joule": result.loss.elastic.item(),
        "inertia_joule": result.loss.inertia.item(),
        "all_parameter_gradients_finite": finite,
        "correction_head_gradient_norm": step.network.correction_head.weight.grad.norm().item(),
        "edge_value_gradient_norm": step.network.layers[0].edge_val.weight.grad.norm().item(),
        "fixed_corner_max_error_m": (result.positions[:, fixed] - x[:, fixed]).abs().max().item(),
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def validate_gradients(*, full_grid: bool = False) -> dict:
    """Run deterministic CPU checks and return a JSON-serializable accuracy report.

    Experimental. Finite differences use three reported step sizes per check;
    acceptance uses the middle, preselected step rather than picking the best.
    Float64 diagnostics compare the same quantized input data and network values.

    Args:
        full_grid: Also run the requested 4,000-cell forward/backward workload.

    Returns:
        Results, error definitions, and a combined pass/fail flag.
    """
    import torch

    working, working_gradients = _check_dtype(torch.float32)
    reference, reference_gradients = _check_dtype(torch.float64)
    comparison = {}
    for name in working_gradients:
        a, b = working_gradients[name], reference_gradients[name]
        comparison[name] = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
    passed = all(
        result[name][1]["relative_error"] < threshold
        for result, threshold in ((working, 0.01), (reference, 1e-5))
        for name in ("hex_energy_positions", "fusion_targets", "network_through_fusion_and_energy")
    )
    passed = passed and all(value < 1e-3 for value in comparison.values())
    passed = passed and working["small_gradient_descent_check"]["decreased"]
    report = {
        "description": "Hex implicit Euler -> network axis update -> incremental hex fusion -> backward",
        "working_dtype": "float32",
        "reference_dtype": "float64 (diagnostics only)",
        "device": "CPU",
        "relative_error_definition": "abs(autograd-FD)/max(abs(autograd),abs(FD),1e-12)",
        "gradient_comparison_definition": "norm(gradient32-gradient64)/norm(gradient64)",
        "float32": working,
        "float64_reference": reference,
        "float32_vs_float64_gradient_relative_l2": comparison,
        "passed": bool(passed),
    }
    if full_grid:
        report["full_grid"] = _full_grid()
        report["passed"] = bool(
            report["passed"]
            and report["full_grid"]["all_parameter_gradients_finite"]
            and report["full_grid"]["fixed_corner_max_error_m"] == 0
        )
    return report


def main() -> None:
    """Write a reproducible gradient report and fail the command if checks fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "generated/gradient_validation/report.json"
    )
    args = parser.parse_args()
    report = validate_gradients(full_grid=args.full_grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
