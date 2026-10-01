# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the FEM accuracy beam scenarios with Newton's VBD solver.

Experimental reference driver. Each scenario from
:mod:`experiments.learned_intrinsic_solver.fem_accuracy_scenarios` is run on
the tetrahedral version of the training beam (``ModelBuilder.add_soft_grid``,
five tetrahedra per hex cell) with the material constants, time step, and
far-face schedule shared with the learned solver. The clamp and, while
driven, the far face are kinematic (zero mass on the model); the far face is
moved to its prescribed rigid motion before every substep and regains its
mass on release. Recorded frames are mapped into the hex corner ordering and
written in the layout read by
:mod:`experiments.learned_intrinsic_solver.render_learned`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import fem_accuracy_scenarios as scenarios
from .vbd_samples import ITERATIONS, _ordering_map, _tet_determinants, add_beam

__all__ = ["DEFAULT_OUTPUT", "format_summary", "run_scenario"]

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Newton checkout that holds the package and the ``generated`` output root."""

DEFAULT_OUTPUT = REPO_ROOT / "generated" / "fem_accuracy_20260930" / "vbd"
"""Default directory receiving one subdirectory per scenario."""

_RELEASE_MOTION_MINIMUM = 1e-3
"""Smallest far-face displacement [m] accepted as proof that a released face moves."""

_PROGRESS_FRAMES = 100
"""Print a progress line every this many recorded frames."""


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def format_summary(metrics: dict, wall_seconds: float | None = None) -> str:
    """Return a one-line summary of the shared metrics for one scenario.

    Args:
        metrics: Output of :func:`fem_accuracy_scenarios.compute_metrics`.
        wall_seconds: Optional wall time appended to the line.

    Returns:
        A single line naming the scenario and its key metric values; metrics
        that a truncated run did not reach print as ``None``.
    """

    def number(key: str, digits: int = 4) -> str:
        value = metrics.get(key)
        return "None" if value is None else f"{value:.{digits}g}"

    name = metrics["scenario"]
    if name == "extension":
        body = (
            f"tip={number('tip_displacement_final')} m (analytic {number('analytic_tip_displacement')} m) "
            f"V/V0={number('bulk_volume_ratio', 6)} minJ={number('min_centre_jacobian_ratio')}"
        )
    elif name == "stretch":
        body = (
            f"V/V0={number('bulk_volume_ratio', 6)} lateral={number('lateral_contraction')} "
            f"minJ={number('min_centre_jacobian_ratio')}"
        )
    elif name == "twist":
        body = (
            f"peak V/V0={number('bulk_volume_ratio_peak', 6)} peak minJ={number('min_centre_jacobian_ratio_peak')} "
            f"final V/V0={number('bulk_volume_ratio_final', 6)} final minJ={number('min_centre_jacobian_ratio_final')}"
        )
    elif name == "compression_release":
        body = (
            f"minJ(compression)={number('min_centre_jacobian_ratio_compression')} "
            f"length recovery={number('length_recovery_ratio')} V/V0={number('bulk_volume_ratio_final', 6)}"
        )
    else:
        raise ValueError(f"unknown scenario {name!r}")
    status = "" if metrics.get("completed", True) else f" INCOMPLETE ({metrics.get('frame_count')} frames)"
    wall = "" if wall_seconds is None else f" wall={wall_seconds:.1f}s"
    return f"{name}: {body}{status}{wall}"


def run_scenario(scenario: scenarios.Scenario, output_dir: Path, device: str, *, iterations: int = ITERATIONS) -> dict:
    """Simulate one scenario with SolverVBD and write its trajectory, metrics, and run record.

    Args:
        scenario: Scenario to run.
        output_dir: Directory receiving ``trajectory.npz``, ``metrics.json``,
            and ``run.json``; created if missing.
        device: Warp device for the model and solver, e.g. ``"cuda:0"``.
        iterations: VBD iterations per substep.

    Returns:
        The run record written to ``run.json`` (settings, environment,
        diagnostics, wall time, and the shared metrics).

    Raises:
        RuntimeError: If the tetrahedral particles do not coincide with the hex
            corners, the state becomes nonfinite, the clamp drifts, or a
            released far face fails to move.
    """
    import warp as wp  # noqa: PLC0415 - keep CPU imports of this module free of device initialization

    import newton  # noqa: PLC0415
    import newton.solvers  # noqa: PLC0415

    start = time.perf_counter()
    output_dir = Path(output_dir)
    rest = scenarios.beam_rest()
    cells = rest.cell_corner_indices
    clamp = scenarios.clamp_indices(rest)
    far = scenarios.far_face_indices(rest)
    mapping = _ordering_map(rest.cell_counts)
    clamp_particles = mapping[clamp]
    far_particles = mapping[far]
    fixed = np.zeros(len(mapping), dtype=bool)
    fixed[clamp_particles] = True

    builder = newton.ModelBuilder(gravity=scenario.gravity)
    tets, rest_determinants, total_mass = add_beam(builder, fixed, pos=wp.vec3(0.0), rot=wp.quat_identity())
    canonical = np.asarray(builder.particle_q, dtype=np.float64)
    if not np.allclose(canonical[mapping], rest.corner_rest_positions, rtol=0.0, atol=1e-6):
        raise RuntimeError("add_soft_grid particles do not coincide with the hex corners")
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=True,
    )
    states = [model.state(), model.state()]
    control = model.control()

    # The far face is kinematic while driven: zero both mass arrays the VBD kernels read from the model
    # (forward_step uses particle_inv_mass, the solve kernels particle_mass). Nothing is cached at construction.
    dynamic_mass = model.particle_mass.numpy().copy()
    dynamic_inv_mass = model.particle_inv_mass.numpy().copy()
    driven_mass = dynamic_mass.copy()
    driven_inv_mass = dynamic_inv_mass.copy()
    driven_mass[far_particles] = 0.0
    driven_inv_mass[far_particles] = 0.0
    far_index_array = wp.array(far_particles.astype(np.int32), dtype=wp.int32, device=model.device)
    far_positions = wp.empty(len(far_particles), dtype=wp.vec3, device=model.device)
    far_velocities = wp.empty(len(far_particles), dtype=wp.vec3, device=model.device)
    face_views = [
        (wp.indexedarray(state.particle_q, [far_index_array]), wp.indexedarray(state.particle_qd, [far_index_array]))
        for state in states
    ]
    driving = scenario.motion != "none"
    if driving:
        model.particle_mass.assign(driven_mass)
        model.particle_inv_mass.assign(driven_inv_mass)

    times = scenarios.frame_times(scenario)
    step_times = scenarios.substep_times(scenario)
    positions = np.empty((scenario.frame_count + 1, len(mapping), 3), dtype=np.float32)
    positions[0] = states[0].particle_q.numpy()[mapping]
    clamp_drift = 0.0
    min_tet_ratio = np.inf
    release_face = None
    release_motion = 0.0
    for frame in range(scenario.frame_count):
        for substep in range(scenarios.SUBSTEPS):
            state_in, state_out = states
            driven, face_positions, face_velocities = scenario.prescribed_far_face(rest, step_times[frame, substep])
            if driven:
                far_positions.assign(face_positions.astype(np.float32))
                far_velocities.assign(face_velocities.astype(np.float32))
                position_view, velocity_view = face_views[0]
                wp.copy(position_view, far_positions)
                wp.copy(velocity_view, far_velocities)
            elif driving:
                driving = False
                model.particle_mass.assign(dynamic_mass)
                model.particle_inv_mass.assign(dynamic_inv_mass)
                release_face = state_in.particle_q.numpy()[far_particles].astype(np.float64)
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, scenarios.TIME_STEP)
            states.reverse()
            face_views.reverse()
        particle_q = states[0].particle_q.numpy()
        if not np.isfinite(particle_q).all():
            raise RuntimeError(f"nonfinite particle positions at frame {frame + 1}")
        positions[frame + 1] = particle_q[mapping]
        clamp_drift = max(clamp_drift, float(np.abs(particle_q[clamp_particles] - canonical[clamp_particles]).max()))
        if clamp_drift > 1e-6:
            raise RuntimeError(f"clamp drifted by {clamp_drift:.3e} m at frame {frame + 1}")
        ratios = _tet_determinants(particle_q.astype(np.float64), tets) / rest_determinants
        min_tet_ratio = min(min_tet_ratio, float(ratios.min()))
        if release_face is not None:
            release_motion = max(release_motion, float(np.abs(particle_q[far_particles] - release_face).max()))
        if (frame + 1) % _PROGRESS_FRAMES == 0 or frame + 1 == scenario.frame_count:
            tip_z = float(particle_q[far_particles, 2].mean())
            print(
                f"{scenario.name}: frame {frame + 1}/{scenario.frame_count} t={times[frame + 1]:.3f}s "
                f"tip_z={tip_z:.5f} m min_tet_ratio={min_tet_ratio:.4f} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    if release_face is not None and release_motion < _RELEASE_MOTION_MINIMUM:
        raise RuntimeError(f"released far face moved only {release_motion:.3e} m; mass restore did not take effect")

    metrics = scenarios.compute_metrics(scenario, positions, rest, cells, far, times)
    scenarios.write_trajectory(output_dir / "trajectory.npz", positions, times, rest, clamp, cells)
    scenarios.write_metrics(output_dir / "metrics.json", metrics)
    wall_seconds = time.perf_counter() - start
    run = {
        "scenario": scenarios.scenario_summary(scenario),
        "solver": {
            "name": "Newton SolverVBD",
            "iterations_per_substep": iterations,
            "particle_enable_tile_solve": True,
            "particle_enable_self_contact": False,
            "time_step_seconds": scenarios.TIME_STEP,
            "substeps_per_frame": scenarios.SUBSTEPS,
            "frame_rate": scenarios.FPS,
            "device": str(model.device),
        },
        "mesh": {
            "cell_counts": list(rest.cell_counts),
            "cell_size": rest.cell_size,
            "particle_count": int(model.particle_count),
            "tet_count": int(len(tets)),
            "tets_per_hex_cell": 5,
            "particle_ordering": "add_soft_grid x-fast order; trajectory frames are mapped to the hex corner order",
            "mass_lumping": "tetrahedral rest volume, rho V / 4 per vertex",
            "total_mass_kg": total_mass,
        },
        "material": {
            "young_modulus_pa": scenarios.YOUNG_MODULUS,
            "poisson_ratio": scenarios.POISSON_RATIO,
            "lame_mu_pa": scenarios.LAME_MU,
            "lame_lambda_pa": scenarios.LAME_LAMBDA,
            "density_kg_per_m3": scenarios.DENSITY,
            "damping_pa_s": scenarios.DAMPING,
        },
        "boundary": {
            "clamp_particle_count": int(len(clamp_particles)),
            "far_face_particle_count": int(len(far_particles)),
            "far_face_driven": scenario.motion != "none",
            "release_frame": scenario.release_frame,
            "kinematic_particles": "zero particle_mass and particle_inv_mass on the model; far face restored on release",
            "far_face_release_displacement_max_m": release_motion if release_face is not None else None,
        },
        "environment": {
            "newton_version": version("newton"),
            "warp_version": version("warp-lang"),
            "numpy_version": np.__version__,
            "newton_git_revision": _git_revision(),
        },
        "diagnostics": {
            "clamp_drift_max_m": clamp_drift,
            "min_tet_volume_ratio": min_tet_ratio,
            "finite_all_frames": True,
        },
        "wall_seconds": wall_seconds,
        "metrics": {
            key: value for key, value in metrics.items() if not key.endswith("_series") and key != "series_times"
        },
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return run


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory receiving one subdirectory per scenario."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(scenarios.SCENARIOS),
        default=list(scenarios.SCENARIOS),
        metavar="NAME",
        help="Scenario names to run (default: all, in registry order).",
    )
    parser.add_argument("--device", default="cuda:0", help="Warp device for the model and solver.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help="VBD iterations per substep (default matches the training data generation).",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("iterations must be positive")
    summaries = []
    for name in args.scenarios:
        run = run_scenario(scenarios.SCENARIOS[name], args.output_dir / name, args.device, iterations=args.iterations)
        line = format_summary(scenarios.read_metrics(args.output_dir / name / "metrics.json"), run["wall_seconds"])
        print(line, flush=True)
        summaries.append(line)
    print("summary:", flush=True)
    for line in summaries:
        print("  " + line, flush=True)


if __name__ == "__main__":
    _main()
