# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe repeated learned proposals against one fixed implicit-Euler objective.

Experimental CPU float32 diagnostic, not training or a fallback optimizer. The
default workload unrolls five updates on each of three 4,000-cell samples with
both the default zero head and a small random diagnostic head. Frozen frames
are recomputed between updates; their decomposition is excluded from backward.
Graph checks and physical descent are reported separately in JSON and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .data import generate_cuboid
from .multiscale import generate_multiscale
from .network import IntrinsicSolverNetwork
from .newton_model import build_newton_hex_model
from .newton_solver import SolverLearnedIntrinsic

__all__ = ["run_optimizer_probe"]


def _direction_diagnostics(gradient, direction) -> dict:
    """Measure the physical directional derivative of the actual corner update."""
    import torch

    dot = (gradient.detach() * direction.detach()).sum().item()
    norm = direction.detach().norm().item()
    scale = gradient.detach().norm().item() * norm
    tolerance = 32 * torch.finfo(direction.dtype).eps * scale
    if not all(math.isfinite(value) for value in (dot, norm, scale)):
        classification = "nonfinite"
    elif norm == 0:
        classification = "zero"
    elif dot < -tolerance:
        classification = "descent"
    elif dot > tolerance:
        classification = "ascent"
    else:
        classification = "near_orthogonal"
    return {
        "direction_norm_m": norm,
        "gradient_dot_direction_joule": dot,
        "gradient_direction_cosine": dot / scale if scale else 0.0,
        "direction_classification": classification,
    }


def _gradient_diagnostics(tensor) -> dict:
    import torch

    gradient = tensor.grad
    if gradient is None:
        return {"present": False, "finite": False, "norm": None, "dtype": None, "nonzero_count": 0}
    return {
        "present": True,
        "finite": bool(torch.isfinite(gradient).all()),
        "norm": gradient.norm().item(),
        "dtype": str(gradient.dtype),
        "nonzero_count": int(torch.count_nonzero(gradient)),
    }


def _geometry_diagnostics(problem, positions) -> dict:
    import torch

    with torch.no_grad():
        energy = problem.optimizer.energy
        corners = positions[:, energy.cell_corner_indices]
        deformation = torch.einsum("bcki,qkj->bcqij", corners - corners[:, :, :1], energy.shape_gradients)
        return {
            "min_gauss_jacobian": torch.linalg.det(deformation).min().item(),
            "fixed_corner_max_error_m": (positions[:, problem.fixed_indices] - problem.fixed_positions)
            .abs()
            .max()
            .item(),
        }


def _energy_row(problem, positions, iteration: int, free_indices) -> tuple[dict, object]:
    import torch

    loss = problem.objective(positions)
    gradient = torch.autograd.grad(loss.total.sum(), positions)[0].detach()
    row = {
        "iteration": iteration,
        "loss_joule": loss.total.item(),
        "elastic_joule": loss.elastic.item(),
        "inertia_joule": loss.inertia.item(),
        "physical_gradient_norm_n": gradient.norm().item(),
        # This is the unconstrained stationarity residual; pin reactions are omitted.
        "free_stationarity_residual_norm_n": gradient[:, free_indices].norm().item(),
        **_geometry_diagnostics(problem, positions),
    }
    return row, gradient


def _fusion_residual(problem, update, free_indices) -> dict:
    """Measure stationarity from rounded output coordinates at all eight Gauss points.

    This includes float32 addition/subtraction error in Xnext - Xcurrent; it
    is not the sparse LU residual before the displacement is added to Xcurrent.
    """
    import torch

    with torch.no_grad():
        step = problem.optimizer
        energy = step.energy
        cells = energy.cell_corner_indices
        corners = update.current_positions[:, cells]
        deformation = torch.einsum("bcki,kj->bcij", corners - corners[:, :, :1], step.center_gradients)
        axes = update.frames.transpose(-1, -2) @ deformation
        target = update.frames @ (update.local_target_axes - axes)
        displacement = update.direction[:, cells]
        gradient = torch.einsum("bcki,qkj->bcqij", displacement - displacement[:, :, :1], energy.shape_gradients)
        weights = step.fusion_stiffness[:, None] * energy.quadrature_weights[None]

        def assemble(values):
            local = torch.einsum("bcqij,qkj,cq->bcki", values, energy.shape_gradients, weights)
            assembled = torch.zeros_like(update.positions)
            assembled.index_add_(1, cells.flatten(), local.flatten(1, 2))
            return assembled[:, free_indices]

        target = target[:, :, None].expand_as(gradient)
        residual = assemble(gradient - target).norm().item()
        load_norm = assemble(target).norm().item()
        return {
            "fusion_residual_norm": residual,
            "fusion_load_norm": load_norm,
            "fusion_relative_residual": residual / load_norm if load_norm else None,
        }


def _run_case(rest, sample, seed: int, *, iterations: int, nonzero_head: bool) -> dict:
    import torch

    started = time.perf_counter()
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == rest.corner_rest_positions[:, 2].min())
    model = build_newton_hex_model(
        rest, fixed, lame_lambda=5e5 * 0.3 / (1.3 * 0.4), lame_mu=5e5 / 2.6, density=1000.0, gravity=(0.0, -9.81, 0.0)
    )
    state = model.state()
    state.particle_q.assign(sample.positions.astype(np.float32))
    state.particle_qd.zero_()
    state.particle_f.zero_()
    original_state = [value.numpy().copy() for value in (state.particle_q, state.particle_qd, state.particle_f)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        network = IntrinsicSolverNetwork(rest.cell_counts, 38)
        if nonzero_head:
            with torch.no_grad():
                network.correction_head.weight.normal_(std=1e-4)
    solver = SolverLearnedIntrinsic(model, network=network, iterations=iterations)
    problem = solver.prepare_problem(state, 1 / 300)
    original_y = problem.inertial_prediction.clone()
    initial = solver.initialize_candidate(problem).clone().requires_grad_()
    free = problem.optimizer.fusion.free_indices
    raw_outputs = []

    def capture_raw(_module, _inputs, output):
        output.retain_grad()
        raw_outputs.append(output)

    hook = network.correction_head.register_forward_hook(capture_raw)
    result = {
        "seed": seed,
        "head_initialization": "diagnostic_nonzero" if nonzero_head else "zero",
        "head_weight_std": 1e-4 if nonzero_head else 0.0,
        "network_seed": 123,
        "network_hops": list(network.hops),
        "dtype": str(initial.dtype),
        "device": str(initial.device),
        "initialization_effective_scale": sample.effective_scale,
        "initialization_screen_halvings": sample.backtracking_steps,
        "previous_state_loss_joule": problem.objective(problem.previous_positions).total.item(),
        "initialized_displacement_norm_m": (initial.detach() - problem.previous_positions).norm().item(),
        "completed_iterations": 0,
        "graph_checks_passed": False,
        "error": None,
        "rows": [],
    }
    updates = []
    current = initial
    try:
        row, gradient = _energy_row(problem, current, 0, free)
        result["rows"].append(row)
        for index in range(iterations):
            update_started = time.perf_counter()
            update = solver.propose_update(current, problem)
            updates.append(update)
            for tensor in (update.positions, update.local_target_axes, update.axis_correction, update.direction):
                tensor.retain_grad()
            after, next_gradient = _energy_row(problem, update.positions, index + 1, free)
            after.update(_direction_diagnostics(gradient, update.direction))
            after.update(_fusion_residual(problem, update, free))
            after.update(
                {
                    "energy_change_joule": after["loss_joule"] - row["loss_joule"],
                    "energy_decreased": after["loss_joule"] < row["loss_joule"],
                    "raw_head_norm": raw_outputs[-1].detach().norm().item(),
                    "axis_correction_norm": update.axis_correction.detach().norm().item(),
                    "step_size": update.step_size.detach().item(),
                    "forward_seconds": time.perf_counter() - update_started,
                }
            )
            result["rows"].append(after)
            result["completed_iterations"] = index + 1
            current, row, gradient = update.positions, after, next_gradient

        # Diagnostic autograd.grad calls also run retained-gradient hooks. Clear
        # these so every following value is from the final physical loss only.
        initial.grad = None
        for update, raw in zip(updates, raw_outputs, strict=True):
            for tensor in (
                update.positions,
                update.local_target_axes,
                update.axis_correction,
                update.direction,
                raw,
            ):
                tensor.grad = None
        network.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        updates[-1].loss.total.sum().backward()
        result["backward_seconds"] = time.perf_counter() - backward_started
        result["initial_candidate_gradient"] = _gradient_diagnostics(initial)
        result["retained_gradients"] = [
            {
                "iteration": index + 1,
                "positions": _gradient_diagnostics(update.positions),
                "local_target_axes": _gradient_diagnostics(update.local_target_axes),
                "axis_correction": _gradient_diagnostics(update.axis_correction),
                "raw_head": _gradient_diagnostics(raw),
            }
            for index, (update, raw) in enumerate(zip(updates, raw_outputs, strict=True))
        ]
        parameters = {name: _gradient_diagnostics(parameter) for name, parameter in network.named_parameters()}
        result["parameter_gradients"] = parameters
        result["all_parameter_gradients_finite"] = all(value["finite"] for value in parameters.values())
        result["all_parameter_gradients_float32"] = all(
            value["dtype"] == "torch.float32" for value in parameters.values()
        )
        result["zero_head_exact_noop"] = (
            all(torch.equal(update.positions, initial) for update in updates) if not nonzero_head else None
        )
        retained = [
            value for item in result["retained_gradients"] for name, value in item.items() if name != "iteration"
        ]
        result["graph_checks_passed"] = bool(
            result["all_parameter_gradients_finite"]
            and result["all_parameter_gradients_float32"]
            and result["initial_candidate_gradient"]["finite"]
            and all(value["finite"] for value in retained)
            and result["retained_gradients"][0]["local_target_axes"]["norm"] > 0
            and result["retained_gradients"][0]["raw_head"]["norm"] > 0
            and parameters["correction_head.weight"]["norm"] > 0
            and all(item["fixed_corner_max_error_m"] == 0 for item in result["rows"])
            and (nonzero_head or result["zero_head_exact_noop"])
        )
    except (ValueError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        hook.remove()
    result["fixed_objective_unchanged"] = torch.equal(original_y, problem.inertial_prediction)
    result["source_state_unchanged"] = all(
        np.array_equal(value.numpy(), before)
        for value, before in zip((state.particle_q, state.particle_qd, state.particle_f), original_state, strict=True)
    )
    result["graph_checks_passed"] &= result["fixed_objective_unchanged"] and result["source_state_unchanged"]
    result["descent_direction_count"] = sum(row.get("direction_classification") == "descent" for row in result["rows"])
    result["ascent_direction_count"] = sum(row.get("direction_classification") == "ascent" for row in result["rows"])
    result["final_energy_decreased"] = (
        result["rows"][-1]["loss_joule"] < result["rows"][0]["loss_joule"] if result["rows"] else False
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _json_safe(value):
    """Keep failed nonfinite diagnostics readable in strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    rows = [
        {"seed": case["seed"], "head_initialization": case["head_initialization"], **row}
        for case in report["cases"]
        for row in case["rows"]
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "iterations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def run_optimizer_probe(
    *,
    output_dir: Path = Path("generated/optimizer_probe"),
    cell_counts: tuple[int, int, int] = (10, 10, 40),
    seeds: tuple[int, ...] = (0, 1, 2),
    iterations: int = 5,
    threads: int = 2,
) -> dict:
    """Run experimental CPU float32 optimizer checks and write JSON/CSV reports.

    No network weights are trained or updated. Initialization uses the existing
    screened multiscale generator before conversion to working float32. Each
    case receives one rigid initialization followed by repeated proposals for
    the same physical objective. Ascent is a measured outcome, not a failed
    differentiation check. Invalid proposals are reported without repair.

    Args:
        output_dir: Directory for report.json and iterations.csv.
        cell_counts: Canonical occupied cuboid dimensions; defaults to 4,000 cells.
        seeds: Deterministic multiscale sample seeds, with strength 0.2.
        iterations: Positive count of unrolled optimizer updates per case.
        threads: Positive Torch CPU worker count, restored on return.

    Returns:
        JSON-compatible report with per-iteration physical energies [J],
        corner directions [m], gradient norms [N], and final-loss derivatives.
    """
    import torch

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not seeds:
        raise ValueError("at least one sample seed is required")
    output_dir = Path(output_dir)
    rest = generate_cuboid(cell_counts, cell_size=0.025)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    report = {
        "cell_counts": list(cell_counts),
        "cell_count": int(np.prod(cell_counts)),
        "corner_count": len(rest.corner_rest_positions),
        "iterations": iterations,
        "seeds": list(seeds),
        "network_seed": 123,
        "working_dtype": "float32",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "torch_cpu_threads": threads,
        "cell_size_m": 0.025,
        "lame_lambda_pa": 5e5 * 0.3 / (1.3 * 0.4),
        "lame_mu_pa": 5e5 / 2.6,
        "density_kg_m3": 1000.0,
        "time_step_s": 1 / 300,
        "gravity_m_s2": [0.0, -9.81, 0.0],
        "initial_velocity_and_external_force": "zero",
        "network_hidden_dim": 128,
        "network_hops": None,
        "frame_derivatives": "frozen at every update; this is not the full polar-frame derivative",
        "interpretation": "Untrained proposals; graph checks do not assert descent or convergence.",
        "parameter_updates": 0,
        "line_search_or_fallback": False,
        "cases": [],
    }
    started = time.perf_counter()
    try:
        for seed in seeds:
            sample = generate_multiscale(rest, seed=seed, strength=0.2)
            for nonzero_head in (False, True):
                result = _run_case(rest, sample, seed, iterations=iterations, nonzero_head=nonzero_head)
                report["cases"].append(result)
                report["network_hops"] = result["network_hops"]
                report["all_graph_checks_passed"] = all(case["graph_checks_passed"] for case in report["cases"])
                report["elapsed_seconds"] = time.perf_counter() - started
                _write_report(report, output_dir)
    finally:
        torch.set_num_threads(previous_threads)
    report["cuda_initialized"] = torch.cuda.is_initialized()
    _write_report(report, output_dir)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated/optimizer_probe"))
    parser.add_argument("--cell-counts", type=int, nargs=3, default=(10, 10, 40))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run_optimizer_probe(
        output_dir=args.output_dir,
        cell_counts=tuple(args.cell_counts),
        seeds=tuple(args.seeds),
        iterations=args.iterations,
        threads=args.threads,
    )
    for case in report["cases"]:
        print(
            f"seed={case['seed']} head={case['head_initialization']} "
            f"iterations={case['completed_iterations']} graph_pass={case['graph_checks_passed']} "
            f"descent={case['descent_direction_count']} ascent={case['ascent_direction_count']} "
            f"final_energy_decreased={case['final_energy_decreased']} error={case['error']}"
        )
    print(f"Reports: {args.output_dir / 'report.json'} and {args.output_dir / 'iterations.csv'}")
    if not report["all_graph_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
