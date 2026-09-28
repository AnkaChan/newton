# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run exact PR #2901 example checks with ALM on/off and record every step.

Run with PYTHONPATH pointing at this Newton worktree and uv run --no-sync.
Fixtures are unmodified files from PR commit c1d64b742085400f95b1595f8b3bb2c767a29560.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp
from residual import finish_metrics, numpy_forces, tet_forces, vertex_metrics

import newton
from newton.viewer import ViewerNull

HERE = Path(__file__).resolve().parent
PR_SHA = "c1d64b742085400f95b1595f8b3bb2c767a29560"
CASES = {
    "extension": ("beam_extension", 300),
    "stretch": ("beam_stretch", 400),
    "twist": ("beam_twist", 400),
    "compression": ("cube_compression", 400),
    "refinement": ("convergence_refinement", 50),
    "sliver": ("sliver_elements", 200),
}
NATIVE = newton.solvers.SolverVBD


class MeasuredSolver:
    """Delegate native stepping; record original-law residual without updating state."""

    def __init__(self, solver, name):
        self.solver, self.name = solver, name
        self.graphs = {}
        m = solver.model
        self.dt = None
        fingerprint = hashlib.sha256()
        for array in (m.particle_q, m.particle_mass, m.particle_flags, m.tet_indices, m.tet_poses, m.tet_materials):
            fingerprint.update(array.numpy().tobytes())
        self.initial_model_sha256 = fingerprint.hexdigest()
        self.elastic = wp.zeros(m.particle_count, dtype=wp.vec3d, device=m.device)
        self.damping = wp.zeros_like(self.elastic)
        self.initial_elastic = wp.zeros_like(self.elastic)
        self.determinants = wp.zeros(m.tet_count, dtype=wp.float64, device=m.device)
        self.metrics = wp.zeros((m.particle_count, 9), dtype=wp.float64, device=m.device)
        self.rows = wp.zeros((10000, 11), dtype=wp.float64, device=m.device)
        self.counter = wp.zeros(1, dtype=int, device=m.device)
        if m.spring_count or m.body_count:
            raise ValueError("Residual evaluator is restricted to the PR tet-only scenes")
        if m.edge_count and np.any(m.edge_bending_properties.numpy()):
            raise ValueError("Nonzero surface bending materials require extra residual terms")
        if m.tri_count and np.any(m.tri_materials.numpy()[:, :3]):
            raise ValueError("Nonzero triangle materials require extra residual terms")

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def record(self, output, contacts, dt):
        m = self.model
        self.elastic.zero_()
        self.damping.zero_()
        self.initial_elastic.zero_()
        wp.launch(
            tet_forces,
            m.tet_count,
            inputs=[
                output.particle_q,
                self.solver.particle_q_prev,
                m.tet_indices,
                m.tet_poses,
                m.tet_materials,
                wp.float64(dt),
                self.elastic,
                self.damping,
                self.initial_elastic,
                self.determinants,
            ],
            device=m.device,
        )
        wp.launch(
            vertex_metrics,
            m.particle_count,
            inputs=[
                output.particle_q,
                self.solver.particle_q_prev,
                self.solver.inertia,
                output.particle_qd,
                m.particle_mass,
                m.particle_inv_mass,
                m.particle_flags,
                self.elastic,
                self.damping,
                self.initial_elastic,
                wp.float64(dt),
                self.metrics,
            ],
            device=m.device,
        )
        count = contacts.soft_contact_count if contacts is not None else None
        wp.launch(
            finish_metrics, 1, inputs=[self.metrics, self.determinants, count, self.counter, self.rows], device=m.device
        )

    def step(self, state_in, state_out, control, contacts, dt):
        self.dt = float(np.float32(dt))
        if wp.get_device().is_capturing:
            self.solver.step(state_in, state_out, control, contacts, dt)
            self.record(state_out, contacts, self.dt)
            return
        # Cache both ping-pong directions. Compression replaces the flag array on
        # release; include its pointer so newly free vertices use a new graph.
        key = (id(state_in), id(state_out), self.model.particle_flags.ptr, self.dt)
        if key not in self.graphs:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.solver.step(state_in, state_out, control, contacts, dt)
                self.record(state_out, contacts, self.dt)
            self.graphs[key] = capture.graph
        wp.capture_launch(self.graphs[key])

    def save(self, directory, mode):
        count = int(self.counter.numpy()[0])
        if count > self.rows.shape[0]:
            raise RuntimeError("Residual recording capacity exceeded")
        rows = self.rows.numpy()[:count]
        if not np.isfinite(rows).all():
            raise RuntimeError("Nonfinite residual or state")
        if np.any(rows[:, 10]):
            raise RuntimeError("Contact activated: tet-only residual is incomplete")
        r = np.sqrt(rows[:, 0])
        r0 = np.sqrt(rows[:, 1])
        rms = r / np.sqrt(rows[:, 5])
        relative = r / np.maximum(r0, 1e-12)
        columns = np.column_stack(
            [
                np.arange(1, count + 1),
                np.arange(1, count + 1) * self.dt,
                r,
                rms,
                r0,
                relative,
                rows[:, 6],
                rows[:, 9],
                rows[:, 7],
                rows[:, 5],
                np.sqrt(rows[:, 2]),
                np.sqrt(rows[:, 3]),
                np.sqrt(rows[:, 4]),
            ]
        )
        filename = f"{self.name}-{mode}.csv"
        np.savetxt(
            directory / filename,
            columns,
            delimiter=",",
            fmt="%.9g",
            header="step,time_s,residual_l2_N,residual_rms_N,initial_residual_l2_N,relative_to_initial,max_vertex_residual_N,min_det_F,max_free_speed_m_s,free_vertices,elastic_l2_N,damping_l2_N,inertia_l2_N",
            comments="",
        )
        m = self.model
        summary = {
            "name": self.name,
            "mode": mode,
            "steps": count,
            "dt": self.dt,
            "iterations": self.iterations,
            "vertices": m.particle_count,
            "tets": m.tet_count,
            "csv": filename,
            "initial_model_sha256": self.initial_model_sha256,
            "contact_count_max": float(rows[:, 10].max()),
            "residual_rms_mean_N": float(rms.mean()),
            "residual_rms_median_N": float(np.median(rms)),
            "residual_rms_p95_N": float(np.percentile(rms, 95)),
            "residual_rms_final_N": float(rms[-1]),
            "relative_to_initial_median": float(np.median(relative)),
            "minimum_det_F": float(rows[:, 9].min()),
            "maximum_free_speed_m_s": float(rows[:, 7].max()),
        }
        if mode == "on":
            rho = self.solver._particle_elasticity_alm_state.tet_rho_pressure.numpy()
            summary["last_step_rho_pressure_min_max"] = [float(rho.min()), float(rho.max())]
        return summary


def validate_residual(device):
    """Verify GPU force evaluation against NumPy and finite-difference energy."""
    rng = np.random.default_rng(281)
    q0 = np.array([[0, 0, 0], [0.7, 0.1, 0], [0.2, 0.9, 0.1], [0, 0.2, 1.2]], np.float32)
    q = (q0 + rng.normal(size=(4, 3)) * 0.025).astype(np.float32)
    ids = np.array([[0, 1, 2, 3]], np.int32)
    poses = np.linalg.inv(np.stack([q0[k] - q0[0] for k in (1, 2, 3)], axis=1))[None].astype(np.float32)
    materials = np.array([[1e4, 1e5, 10.0]], np.float32)
    dt = float(np.float32(1 / 300))
    expected_e, expected_d, _ = numpy_forces(q, q0, ids, poses, materials, dt)
    arrays = [
        wp.array(q, dtype=wp.vec3, device=device),
        wp.array(q0, dtype=wp.vec3, device=device),
        wp.array(ids, dtype=int, device=device),
        wp.array(poses, dtype=wp.mat33, device=device),
        wp.array(materials, dtype=float, device=device),
    ]
    fe, fd, f0 = [wp.zeros(4, dtype=wp.vec3d, device=device) for _ in range(3)]
    determinants = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(tet_forces, 1, inputs=[*arrays, wp.float64(dt), fe, fd, f0, determinants], device=device)
    np.testing.assert_allclose(fe.numpy(), expected_e, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(fd.numpy(), expected_d, rtol=1e-10, atol=1e-9)
    q64 = q.astype(np.float64)
    fd_gradient = np.zeros_like(q64)
    h = 1e-6
    for i in range(4):
        for j in range(3):
            plus, minus = q64.copy(), q64.copy()
            plus[i, j] += h
            minus[i, j] -= h
            fd_gradient[i, j] = (
                numpy_forces(plus, q0, ids, poses, materials, dt)[2]
                - numpy_forces(minus, q0, ids, poses, materials, dt)[2]
            ) / (2 * h)
    error = float(np.linalg.norm(fd_gradient + expected_e + expected_d) / np.linalg.norm(expected_e + expected_d))
    if error > 1e-7:
        raise AssertionError(f"Energy-gradient residual validation failed: {error}")
    return {"gpu_matches_float64_numpy": True, "energy_gradient_relative_error": error}


def run_case(name, mode, output, frame_limit=None):
    fixture, frames = CASES[name]
    source = HERE / "sources" / f"example_soft_{fixture}.py"
    spec = importlib.util.spec_from_file_location(f"pr2901_{name}_{mode}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    measured = []
    labels = {
        "compression": ["compression50", "compression90"],
        "refinement": ["refinement_visual", "refinement_coarse", "refinement_medium", "refinement_fine"],
    }.get(name, [name])

    def factory(*args, **kwargs):
        kwargs["particle_elasticity_alm"] = mode == "on"
        kwargs["particle_elasticity_alm_deviatoric"] = False
        kwargs["particle_elasticity_alm_rho_scale"] = 1.0
        solver = MeasuredSolver(NATIVE(*args, **kwargs), labels[len(measured)])
        measured.append(solver)
        return solver

    failures = []
    newton.solvers.SolverVBD = factory
    start = time.perf_counter()
    try:
        example = module.Example(ViewerNull())
        actual_frames = min(frames, frame_limit) if frame_limit else frames
        for frame in range(actual_frames):
            example.step()
            if hasattr(example, "test_post_step"):
                try:
                    example.test_post_step()
                except (AssertionError, ValueError) as exc:
                    if not failures:
                        failures.append({"phase": "post_step", "frame": frame + 1, "message": str(exc)})
            if (frame + 1) % 50 == 0:
                print(
                    f"{name} ALM={mode}: {frame + 1}/{actual_frames} frames, {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
        if not frame_limit:
            try:
                example.test_final()
            except (AssertionError, ValueError) as exc:
                failures.append({"phase": "test_final", "message": str(exc)})
            # A failing main recovery check must not prevent the requested 90% run.
            if name == "compression" and len(measured) == 1:
                rec, vol, inv, nan = module._run_compression(compress_ratio=0.10)
                if nan or rec < 0.90 or vol < 0.90 or inv < module._INVERSION_TOL:
                    failures.append(
                        {
                            "phase": "auxiliary_compression90",
                            "message": f"height={rec}, volume={vol}, min_J={inv}, NaN={nan}",
                        }
                    )
        wp.synchronize()
        data = [solver.save(output, mode) for solver in measured]
        expected = [actual_frames * example.sim_substeps]
        if not frame_limit:
            if name == "compression":
                expected += [2000]
            if name == "refinement":
                expected += [900, 900, 900]
        assert [row["steps"] for row in data] == expected, (data, expected)
        q = example.state_0.particle_q.numpy()
        result = {
            "case": name,
            "mode": mode,
            "frames": actual_frames,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "checks_run": frame_limit is None,
            "original_checks_passed": not failures if not frame_limit else None,
            "failures": failures,
            "runs": data,
            "instrumented_wall_seconds": time.perf_counter() - start,
            "final_position_bounds": [q.min(axis=0).tolist(), q.max(axis=0).tolist()],
        }
        (output / f"{name}-{mode}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            f"FINISHED {name} ALM={mode}: checks={result['original_checks_passed']}; "
            + ", ".join(f"{r['name']}: mean RMS={r['residual_rms_mean_N']:.6g} N" for r in data),
            flush=True,
        )
        return result
    finally:
        newton.solvers.SolverVBD = NATIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--modes", nargs="+", choices=["off", "on"], default=["off", "on"])
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--smoke-frames", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    validation = validate_residual(wp.get_device())
    print("Residual validation:", validation, flush=True)
    provenance = {
        "pr": "https://github.com/newton-physics/newton/pull/2901",
        "pr_sha": PR_SHA,
        "solver_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "newton_source": str(Path(newton.__file__).resolve()),
        "warp_version": wp.__version__,
        "device": wp.get_device().name,
        "validation": validation,
        "alm_mode": "pressure-only; rho_scale=1; persistent history; no inter-step resets",
        "residual": "r = f_elastic(original law) + f_damping - m*(x-x_hat)/dt^2; x_hat is native float32 predictor, forces evaluated in float64; free vertices only",
        "normalization": "RMS = ||r||_2/sqrt(number of free vertices); relative = ||r||_2/max(||r(x_start)||_2,1e-12 N)",
        "measurement": "Every physical solver.step, including each frame substep. Each mode follows its own continuous trajectory. Equal iteration counts; instrumented wall time is not a speed benchmark.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not args.validate_only:
        for case in args.cases:
            for mode in args.modes:
                run_case(case, mode, args.output, args.smoke_frames)


if __name__ == "__main__":
    main()
