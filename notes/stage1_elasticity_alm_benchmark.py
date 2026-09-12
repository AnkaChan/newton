# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the stage 1 experiment through the public SolverVBD interface.

Run with uv after claiming a GPU. CUDA timings include a reset and one cold
implicit step inside a captured graph. No contact or damping is present.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton

MODES = {
    "legacy": {"particle_elasticity_alm": False},
    "pressure": {"particle_elasticity_alm": True, "particle_elasticity_alm_deviatoric": False},
    "full": {"particle_elasticity_alm": True, "particle_elasticity_alm_deviatoric": True},
}


def make_grid(device, cells, lame):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=cells,
        dim_y=cells,
        dim_z=cells,
        cell_x=0.2,
        cell_y=0.2,
        cell_z=0.2,
        density=1000.0,
        k_mu=1.0e4,
        k_lambda=lame,
        k_damp=0.0,
    )
    q = np.asarray(builder.particle_q)
    for index in np.flatnonzero(q[:, 2] == 0.0):
        builder.particle_mass[index] = 0.0
    builder.color()
    model = builder.finalize(device=device)
    force = np.zeros((model.particle_count, 3), dtype=np.float32)
    force[np.isclose(q[:, 2], q[:, 2].max()), 2] = -10.0
    return model, force


def physical_residual(model, q, force, dt):
    """Evaluate true material equilibrium in float64, independent of ALM history."""
    q = q.astype(np.float64)
    q0 = model.particle_q.numpy().astype(np.float64)
    ids = model.tet_indices.numpy()
    inv_rest = model.tet_poses.numpy().astype(np.float64)
    material = model.tet_materials.numpy().astype(np.float64)
    ds = np.stack([q[ids[:, j]] - q[ids[:, 0]] for j in (1, 2, 3)], axis=2)
    F = ds @ inv_rest
    J = np.linalg.det(F)
    cof = np.stack(
        [
            np.cross(F[:, :, 1], F[:, :, 2]),
            np.cross(F[:, :, 2], F[:, :, 0]),
            np.cross(F[:, :, 0], F[:, :, 1]),
        ],
        axis=2,
    )
    mu = material[:, 0]
    pressure = (material[:, 1] + mu) * (J - 1.0) - mu
    stress = mu[:, None, None] * F + pressure[:, None, None] * cof
    volume = 1.0 / (6.0 * np.linalg.det(inv_rest))
    residual = force.astype(np.float64) - model.particle_mass.numpy()[:, None] * (q - q0) / dt**2
    weights = np.concatenate([-np.sum(inv_rest, axis=1)[:, None, :], inv_rest], axis=1)
    for j in range(4):
        element_force = -volume[:, None] * np.einsum("eij,ej->ei", stress, weights[:, j])
        np.add.at(residual, ids[:, j], element_force)
    free = model.particle_mass.numpy() > 0.0
    return float(np.linalg.norm(residual[free]) / np.linalg.norm(force[free])), float(J.min())


def sample(model, force, dt, mode, iterations, repeats):
    solver = newton.solvers.SolverVBD(model, iterations=iterations, **MODES[mode])
    state, output = model.state(), model.state()
    state.particle_f.assign(force)

    def cold_step():
        solver.reset(state)
        solver.step(state, output, None, None, dt)

    cold_step()
    q = output.particle_q.numpy().copy()
    milliseconds = None
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            cold_step()
        start = wp.Event(device=model.device, enable_timing=True)
        end = wp.Event(device=model.device, enable_timing=True)
        batches = []
        for _ in range(5):
            wp.record_event(start)
            for _ in range(repeats):
                wp.capture_launch(capture.graph)
            wp.record_event(end)
            batches.append(wp.get_event_elapsed_time(start, end) / repeats)
        milliseconds = float(np.median(batches))
    relative_residual, minimum_J = physical_residual(model, q, force, dt)
    return q, {
        "mode": mode,
        "iterations": iterations,
        "reset_step_ms": milliseconds,
        "relative_force_residual": relative_residual,
        "minimum_J": minimum_J,
    }


def rotating_history(device):
    """Record force artifacts when a world-space matrix memory crosses a rotation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for point in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=1.0e4, k_lambda=1.0e4, k_damp=0.0)
    builder.color()
    model = builder.finalize(device=device)
    theta = 0.2
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    q0 = model.particle_q.numpy().copy()
    q_rotated = ((q0 - q0.mean(axis=0)) @ rotation.T + q0.mean(axis=0)).astype(np.float32)
    result = []
    for mode, options in MODES.items():
        for reseed in (False, True):
            solver = newton.solvers.SolverVBD(model, iterations=1, **options)
            state, output = model.state(), model.state()
            solver.step(state, output, None, None, 1.0 / 60.0)
            state.particle_q.assign(q_rotated)
            state.particle_qd.zero_()
            if reseed:
                solver.reset(state, flags=0)
            solver.step(state, output, None, None, 1.0 / 60.0)
            displacement = output.particle_q.numpy() - q_rotated
            result.append(
                {
                    "mode": mode,
                    "reseed": reseed,
                    "maximum_vertex_motion_m": float(np.linalg.norm(displacement, axis=1).max()),
                    "maximum_speed_m_s": float(np.linalg.norm(output.particle_qd.numpy(), axis=1).max()),
                }
            )
    return result


def rest_drift(device, steps=1000):
    """Measure accumulated motion at rest with persistent history and no resets."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for point in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=1000.0, k_lambda=2000.0, k_damp=0.0)
    builder.color()
    model = builder.finalize(device=device)
    q0 = model.particle_q.numpy().copy()
    result = []
    for mode, options in MODES.items():
        solver = newton.solvers.SolverVBD(model, iterations=5, **options)
        states = [model.state(), model.state()]
        solver.step(states[0], states[1], None, None, 1.0 / 120.0)
        solver.reset(states[0])
        if model.device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                solver.step(states[0], states[1], None, None, 1.0 / 120.0)
                solver.step(states[1], states[0], None, None, 1.0 / 120.0)
            for _ in range(steps // 2):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(steps // 2):
                solver.step(states[0], states[1], None, None, 1.0 / 120.0)
                solver.step(states[1], states[0], None, None, 1.0 / 120.0)
        result.append(
            {
                "mode": mode,
                "steps": steps,
                "maximum_vertex_motion_m": float(np.linalg.norm(states[0].particle_q.numpy() - q0, axis=1).max()),
                "maximum_speed_m_s": float(np.linalg.norm(states[0].particle_qd.numpy(), axis=1).max()),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cells", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 5, 10, 30, 100])
    parser.add_argument("--reference-iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("notes/stage1_elasticity_alm_results.json"))
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    result = {"device": str(wp.get_device(args.device)), "warp": wp.__version__, "dt": 1.0 / 30.0, "cases": []}
    with wp.ScopedDevice(args.device):
        for lame in [1.0e4, 1.0e6]:
            model, force = make_grid(args.device, args.cells, lame)
            reference, reference_row = sample(model, force, result["dt"], "legacy", args.reference_iterations, 1)
            rows = []
            for mode in MODES:
                for iterations in args.iterations:
                    q, row = sample(model, force, result["dt"], mode, iterations, args.repeats)
                    row["maximum_position_error_to_reference_m"] = float(np.linalg.norm(q - reference, axis=1).max())
                    rows.append(row)
                    print(
                        f"lambda={lame:g} {mode:8} {iterations:3} iterations: residual={row['relative_force_residual']:.5g}, {row['reset_step_ms']:.5g} ms",
                        flush=True,
                    )
            result["cases"].append(
                {
                    "mu": 1.0e4,
                    "lambda": lame,
                    "particles": model.particle_count,
                    "tets": model.tet_count,
                    "reference": reference_row,
                    "rows": rows,
                }
            )
        result["rotation_probe"] = rotating_history(args.device)
        result["rest_drift"] = rest_drift(args.device)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
