# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the FEM accuracy beam scenarios with the learned intrinsic hex solver.

Experimental. Each scenario of :mod:`.fem_accuracy_scenarios` is advanced on
the training beam with a trained mixed-pool checkpoint following the rollout
recipe of :mod:`.simulate_mixed`: the network and the step are rebuilt from
the checkpoint, one material context is registered with the scenario's SI
gravity, every physical substep starts from the inertial candidate and takes
``iterations`` learned updates with fusion after each one, and the committed
candidate becomes the next step's start. Positions are recorded every
:data:`.fem_accuracy_scenarios.SUBSTEPS` substeps in the layout that
``render_learned`` reads; ``metrics.json`` holds the shared metrics and
``run.json`` the checkpoint digest, the settings, the wall time and per-frame
diagnostics. There is no contact in these scenarios.

The far face is driven by building the step with the clamp corners followed by
the far-face corners as prescribed corners and passing the scheduled far-face
positions of the step end to ``prepare``/``advance`` every substep, so the
prescribed corners are fused exactly onto the schedule and their inertial
prediction follows it. The frame tie-break reference stays on the clamp
through ``reference_corners``. When a scenario releases the far face, the run
switches to a second step whose prescribed corners are the clamp only: the
committed positions, the finite-difference velocities and the per-cell
optimizer history are carried into the new step's payload unchanged
(``history.carry_history``), exactly as across any other physical step
boundary. A nonfinite proposal ends the scenario; the frames recorded so far
are written and the failure frame is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import fem_accuracy_scenarios as scenarios

__all__ = ["CONTEXT_ID", "StepSettings", "load_checkpoint", "run_scenario"]

CONTEXT_ID = "beam"
"""Identifier of the single material context registered on every step."""

_FAILURE_ERRORS = (ValueError, RuntimeError, FloatingPointError)
"""Errors that end a scenario as a recorded failure instead of aborting the run."""


@dataclass(frozen=True)
class StepSettings:
    """Step construction settings taken from the training configuration.

    Attributes:
        target_modes: Target vectors per cell of the network (3 or 7).
        energy_floor_scale: Multiplier of the material-aware energy floor.
        contact_max_pairs: Largest number of static-point pairs per sample.
        contact_tokens_per_cell: Token slots per cell of the contact encoder.
        contact_friction_epsilon: Friction smoothing band as a fraction of dt.
        cpu_threads: Torch CPU thread count used while running.
    """

    target_modes: int = 3
    energy_floor_scale: float = 1.0
    contact_max_pairs: int = 4
    contact_tokens_per_cell: int = 24
    contact_friction_epsilon: float = 1e-2
    cpu_threads: int = 2

    @classmethod
    def from_config(cls, config) -> StepSettings:
        """Read the step settings of a ``MixedTrainConfig``."""
        return cls(
            target_modes=int(config.target_modes),
            energy_floor_scale=float(config.energy_floor_scale),
            contact_max_pairs=int(getattr(config, "contact_max_pairs", 4)),
            contact_tokens_per_cell=int(getattr(config, "contact_tokens_per_cell", 24)),
            contact_friction_epsilon=float(getattr(config, "contact_friction_epsilon", 1e-2)),
            cpu_threads=int(getattr(config, "cpu_threads", 2)),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_digest(network) -> str:
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in network.parameters())).hexdigest()


def _json_ready(value):
    """Return ``value`` with tuples, arrays and NumPy scalars converted for ``json.dumps``."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_checkpoint(checkpoint: Path, device):
    """Rebuild the trained network of a ``mixed_pool_v2`` checkpoint on ``device``.

    Args:
        checkpoint: Path of the training checkpoint (read only).
        device: Torch device of the network and the step.

    Returns:
        ``(saved, config, network)``: the raw checkpoint dictionary, its
        ``MixedTrainConfig`` and the evaluation-mode network with the saved
        weights.

    Raises:
        ValueError: The checkpoint is not a mixed-pool checkpoint or was not
            trained on the FEM accuracy beam.
    """
    import torch

    from . import train_mixed  # noqa: PLC0415

    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if saved.get("format") != "mixed_pool_v2":
        raise ValueError("checkpoint is not a mixed_pool_v2 training checkpoint")
    config = train_mixed.MixedTrainConfig.from_checkpoint_config(saved["config"])
    if tuple(config.cell_counts) != scenarios.CELL_COUNTS or not math.isclose(
        config.cell_size, scenarios.CELL_SIZE, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("checkpoint was not trained on the FEM accuracy beam grid")
    if not math.isclose(config.time_step, scenarios.TIME_STEP, rel_tol=1e-9, abs_tol=0):
        raise ValueError("checkpoint time step differs from the scenario substep")
    network = train_mixed._build_network(config).to(device)
    network.load_state_dict(saved["network_state"])
    network.eval()
    return saved, config, network


def _build_step(rest, fixed, *, network, gravity, settings: StepSettings, device, reference_corners=None):
    """Build a step with ``fixed`` prescribed corners and register the beam material with ``gravity``."""
    from .mixed_physics import MixedHexSolverStep  # noqa: PLC0415

    step = MixedHexSolverStep(
        rest,
        np.asarray(fixed, dtype=np.int64),
        network=network,
        time_step=scenarios.TIME_STEP,
        gravity=gravity,
        energy_floor_scale=settings.energy_floor_scale,
        contact_max_pairs=settings.contact_max_pairs,
        contact_tokens_per_cell=settings.contact_tokens_per_cell,
        contact_friction_epsilon=settings.contact_friction_epsilon,
        target_modes=settings.target_modes,
        reference_corners=reference_corners,
    ).to(device)
    step.eval()
    step.register_context(
        CONTEXT_ID,
        lame_lambda=scenarios.LAME_LAMBDA,
        lame_mu=scenarios.LAME_MU,
        density=scenarios.DENSITY,
        damping=scenarios.DAMPING,
        gravity=gravity,
    )
    return step


def _inertial_candidate(payload: dict, fixed_indices) -> None:
    """Start the learned iterations from the inertial prediction with the prescribed rows set."""
    import torch

    candidate = payload["inertial_prediction"].clone()
    candidate[fixed_indices] = payload["fixed_positions"]
    if not torch.isfinite(candidate).all():
        raise ValueError("nonfinite candidate initialization")
    payload["candidate"] = candidate.detach()
    payload["candidate_mode"] = "inertial"


def _query(step, payload: dict, *, device, cell_count: int) -> tuple[float, float]:
    """Run one learned iteration in place; return the total energy [J] and the free force residual [N]."""
    from . import history as history_module  # noqa: PLC0415
    from . import train_mixed  # noqa: PLC0415

    batch = train_mixed._batch([payload], device, cell_count=cell_count, modes=step.target_modes)
    result = train_mixed._checked_forward(step, step, batch)
    history_module.store_history([payload], result)
    payload["candidate"] = result.positions[0].detach().cpu()
    return float(result.loss.total[0]), float(result.force_residual_norm[0])


def _carry(previous: dict, prepared: dict) -> dict:
    """Copy the optimizer history of ``previous`` into the freshly prepared payload."""
    from . import history as history_module  # noqa: PLC0415

    history_module.carry_history(previous, prepared)
    return prepared


def run_scenario(
    scenario: scenarios.Scenario,
    *,
    rest,
    network,
    settings: StepSettings,
    device,
    iterations: int,
    output_dir: Path,
    run_info: dict | None = None,
    progress_interval: int = 200,
) -> dict:
    """Run one scenario and write ``trajectory.npz``, ``metrics.json`` and ``run.json`` into ``output_dir``.

    Args:
        scenario: Scenario to run; its far-face schedule drives the prescribed corners.
        rest: Rest grid with the clamp at rest z = 0 and the far face at the largest rest z.
        network: Trained network on ``device`` (weights are never modified).
        settings: Step construction settings.
        device: Torch device of the network.
        iterations: Learned iterations per physical substep.
        output_dir: Directory of the three output files; created if needed.
        run_info: Extra JSON-serialisable entries written into ``run.json``
            (checkpoint path, digest and epoch).
        progress_interval: Print progress every this many substeps.

    Returns:
        The metrics dictionary written to ``metrics.json``.
    """
    import torch

    from . import history as history_module  # noqa: PLC0415
    from .frames import select_reference_corners  # noqa: PLC0415

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    started = time.perf_counter()
    rest_positions = np.asarray(rest.corner_rest_positions, dtype=np.float64)
    cells = np.asarray(rest.cell_corner_indices, dtype=np.int64)
    clamp = scenarios.clamp_indices(rest)
    far = scenarios.far_face_indices(rest)
    driven_fixed = np.concatenate([clamp, far])
    reference = select_reference_corners(rest_positions, clamp)
    gravity = tuple(float(value) for value in scenario.gravity)
    parameter_digest = _parameter_digest(network)
    steps = {"free": _build_step(rest, clamp, network=network, gravity=gravity, settings=settings, device=device)}
    if scenario.motion != "none":
        steps["driven"] = _build_step(
            rest,
            driven_fixed,
            network=network,
            gravity=gravity,
            settings=settings,
            device=device,
            reference_corners=reference,
        )
    clamp_rest = torch.tensor(rest_positions[clamp], dtype=torch.float32)

    def select(time_seconds: float):
        """Return the step and the prescribed positions [K, 3] of the substep ending at ``time_seconds``."""
        driven, positions, _ = scenario.prescribed_far_face(rest_positions, time_seconds)
        if not driven:
            return steps["free"], None
        return steps["driven"], torch.cat([clamp_rest, torch.tensor(positions, dtype=torch.float32)])

    cell_count = len(cells)
    end_times = scenarios.substep_times(scenario).ravel()
    total_substeps = len(end_times)
    frames = [rest_positions.astype(np.float32)]
    times = [0.0]
    energies: list[float] = []
    residuals: list[float] = []
    frame_min_jacobian: list[float] = [float(scenarios.centre_jacobian_ratios(frames[0], rest_positions, cells).min())]
    min_jacobian = frame_min_jacobian[0]
    learned_calls = 0
    completed = 0
    failure = None
    release_substep = None
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            step, fixed = select(end_times[0])
            x0 = torch.tensor(rest_positions, dtype=torch.float32)
            payload = step.prepare(CONTEXT_ID, x0, torch.zeros_like(x0), fixed_positions=fixed)
            payload.update(history_module.empty_history(cell_count, settings.target_modes))
            _inertial_candidate(payload, step.fixed_indices.cpu())
            for index, end_time in enumerate(end_times):
                substep = index + 1
                try:
                    energy = residual = math.nan
                    for _ in range(iterations):
                        energy, residual = _query(step, payload, device=device, cell_count=cell_count)
                        learned_calls += 1
                    committed = payload["candidate"].numpy()
                    completed = substep
                    energies.append(energy)
                    residuals.append(residual)
                    min_jacobian = min(
                        min_jacobian, float(scenarios.centre_jacobian_ratios(committed, rest_positions, cells).min())
                    )
                    if substep % scenarios.SUBSTEPS == 0:
                        frames.append(committed.astype(np.float32, copy=True))
                        times.append(float(end_time))
                        frame_min_jacobian.append(
                            float(scenarios.centre_jacobian_ratios(committed, rest_positions, cells).min())
                        )
                    if substep < total_substeps:
                        next_step, next_fixed = select(end_times[index + 1])
                        if next_step is step:
                            payload = _carry(payload, step.advance(payload, fixed_positions=next_fixed))
                        else:
                            # Release: the committed positions, the finite-difference velocities (the held far
                            # face has none) and the optimizer history move to the clamp-only step unchanged.
                            release_substep = substep
                            positions = payload["candidate"]
                            velocity = (positions - payload["physical_positions"]) / scenarios.TIME_STEP
                            prepared = next_step.prepare(CONTEXT_ID, positions, velocity, fixed_positions=next_fixed)
                            payload = _carry(payload, prepared)
                            step = next_step
                        _inertial_candidate(payload, step.fixed_indices.cpu())
                except _FAILURE_ERRORS as error:
                    failure = {
                        "frame": (substep - 1) // scenarios.SUBSTEPS + 1,
                        "substep": substep,
                        "time_seconds": float(end_time),
                        "last_recorded_frame": len(frames) - 1,
                        "learned_calls_before_failure": learned_calls,
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                    break
                if substep % progress_interval == 0 or substep == total_substeps:
                    print(
                        f"{scenario.name}: substep={substep}/{total_substeps} frame={len(frames) - 1} "
                        f"energy={energy:.4g}J residual={residual:.4g}N min_J={min_jacobian:.4f} "
                        f"elapsed={time.perf_counter() - started:.0f}s",
                        flush=True,
                    )
        if _parameter_digest(network) != parameter_digest:
            raise RuntimeError("scenario run changed checkpoint parameters")
    finally:
        for step in steps.values():
            step.close()
    elapsed = time.perf_counter() - started
    positions = np.stack(frames)
    times_array = np.asarray(times, dtype=np.float64)
    scenarios.write_trajectory(output_dir / "trajectory.npz", positions, times_array, rest_positions, clamp, cells)
    metrics = scenarios.compute_metrics(scenario, positions, rest_positions, cells, far, times_array)
    metrics.update({"solver": "learned", "failure": failure})
    scenarios.write_metrics(output_dir / "metrics.json", _json_ready(metrics))
    run = {
        **(run_info or {}),
        "solver": "learned",
        "scenario": scenarios.scenario_summary(scenario),
        "status": "failed" if failure else "complete",
        "failure": failure,
        "iterations_per_substep": iterations,
        "candidate_mode": "inertial",
        "device": str(device),
        "step_settings": asdict(settings),
        "material": {
            "lame_lambda": scenarios.LAME_LAMBDA,
            "lame_mu": scenarios.LAME_MU,
            "youngs_modulus": scenarios.YOUNG_MODULUS,
            "poisson_ratio": scenarios.POISSON_RATIO,
            "density": scenarios.DENSITY,
            "damping": scenarios.DAMPING,
        },
        "gravity": list(gravity),
        "clamp_corner_count": int(len(clamp)),
        "far_face_corner_count": int(len(far)),
        "reference_corners": None if reference is None else [int(value) for value in reference],
        "release_substep": release_substep,
        "release_handling": (
            "committed positions, finite-difference velocities and optimizer history carried into a clamp-only step"
            if scenario.release_frame is not None
            else None
        ),
        "completed_substeps": completed,
        "total_substeps": total_substeps,
        "recorded_frames": len(frames) - 1,
        "learned_calls": learned_calls,
        "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "diagnostics": {
            "min_centre_jacobian_ratio_all_substeps": min_jacobian,
            "frame_min_centre_jacobian_ratio": frame_min_jacobian,
            "substep_energy_last_iteration_joule": energies,
            "substep_free_force_residual_newton": residuals,
        },
    }
    (output_dir / "run.json").write_text(json.dumps(_json_ready(run), indent=2) + "\n")
    return metrics


def _summary_line(metrics: dict) -> str:
    """Return the one-line metric summary of one scenario."""
    name = metrics["scenario"]
    status = "complete" if metrics["completed"] else f"FAILED at frame {metrics['failure']['frame']}"
    if name == "extension":
        body = (
            f"tip={metrics['tip_displacement_final']:.5f}m (analytic {metrics['analytic_tip_displacement']:.5f}m) "
            f"V/V0={metrics['bulk_volume_ratio']:.5f} minJ={metrics['min_centre_jacobian_ratio']:.4f}"
        )
    elif name == "stretch":
        body = (
            f"V/V0={metrics['bulk_volume_ratio']:.4f} lateral={metrics['lateral_contraction']:.4f} "
            f"minJ={metrics['min_centre_jacobian_ratio']:.4f}"
        )
    elif name == "twist":
        body = (
            f"peak V/V0={metrics['bulk_volume_ratio_peak']} minJ_peak={metrics['min_centre_jacobian_ratio_peak']} "
            f"final V/V0={metrics['bulk_volume_ratio_final']:.4f} minJ_final={metrics['min_centre_jacobian_ratio_final']:.4f}"
        )
    else:
        body = (
            f"minJ_compression={metrics['min_centre_jacobian_ratio_compression']:.4f} "
            f"length_recovery={metrics['length_recovery_ratio']:.4f} V/V0={metrics['bulk_volume_ratio_final']:.4f}"
        )
    return f"{name}: {status}, frames={metrics['frame_count']}, {body}"


def _main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenarios", nargs="+", choices=tuple(scenarios.SCENARIOS), default=tuple(scenarios.SCENARIOS)
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    saved, config, network = load_checkpoint(args.checkpoint, device)
    settings = StepSettings.from_config(config)
    previous_threads = torch.get_num_threads()
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(settings.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    run_info = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch": int(saved["report"]["completed_epochs"]),
        "best_selection": saved["report"].get("best_selection"),
        "feature_schema_version": saved["config"].get("feature_schema_version"),
    }
    rest = scenarios.beam_rest()
    try:
        for name in args.scenarios:
            metrics = run_scenario(
                scenarios.SCENARIOS[name],
                rest=rest,
                network=network,
                settings=settings,
                device=device,
                iterations=args.iterations,
                output_dir=args.output_dir / name,
                run_info=run_info,
            )
            print(_summary_line(metrics), flush=True)
    finally:
        torch.set_num_threads(previous_threads)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32


if __name__ == "__main__":
    _main()
