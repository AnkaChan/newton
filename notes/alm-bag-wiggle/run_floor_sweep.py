# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare original-force residuals from identical bag states with a rho-floor sweep."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from cloth_residual import bend_forces, bend_metrics, record_row, triangle_forces, triangle_metrics
from run_case import ROOT, Bag, drive_pins, pin_motion
from validate_cloth_residual import validate

from newton._src.solvers.vbd.particle_vbd_kernels import (
    accumulate_self_contact_force_and_hessian,
    gather_particle_body_contact_force_and_hessian,
)

MODES = {"off": None, "floor9": 9.0, "floor1": 1.0, "floor0p1": 0.1, "floor0p01": 0.01, "inertia": 0.0}
FIELDS = [
    "original_rms_N",
    "alm_rms_N",
    "constitutive_gap_rms_N",
    "elastic_rms_N",
    "damping_rms_N",
    "contact_rms_N",
    "inertia_rms_N",
    "free_vertices",
    "original_max_N",
    "iterate_change_rms_m",
]


class Measurement:
    def __init__(self, sim, floor):
        self.sim, self.floor = sim, floor
        m, s = sim.model, sim.solver
        self.elastic = wp.zeros(m.particle_count, dtype=wp.vec3d, device=m.device)
        self.damping = wp.zeros_like(self.elastic)
        self.gap = wp.zeros_like(self.elastic)
        self.contact = wp.zeros(m.particle_count, dtype=wp.vec3, device=m.device)
        self.hessian = wp.zeros(m.particle_count, dtype=wp.mat33, device=m.device)
        self.previous = wp.clone(sim.state_0.particle_q)
        self.ids = wp.array(np.arange(m.particle_count, dtype=np.int32), dtype=int, device=m.device)
        self.colors = wp.zeros(m.particle_count, dtype=int, device=m.device)
        self.rows = wp.zeros((s.iterations + 1, len(FIELDS)), dtype=wp.float64, device=m.device)

    def install(self):
        s = self.sim.solver
        initialize = s._initialize_particles
        iterate = s._solve_particle_iteration

        def measured_initialize(state_in, state_out, dt):
            initialize(state_in, state_out, dt)
            m, a = self.sim.model, s._particle_elasticity_alm_state
            if self.floor is not None:
                wp.launch(
                    triangle_metrics,
                    m.tri_count,
                    inputs=[
                        s.particle_q_prev,
                        m.tri_indices,
                        m.tri_poses,
                        m.tri_areas,
                        m.tri_materials,
                        m.particle_inv_mass,
                        m.particle_flags,
                        dt,
                        self.floor,
                        a,
                    ],
                    device=m.device,
                )
                wp.launch(
                    bend_metrics,
                    m.edge_count,
                    inputs=[
                        s.particle_q_prev,
                        m.edge_indices,
                        m.edge_rest_length,
                        m.edge_bending_properties,
                        m.particle_inv_mass,
                        m.particle_flags,
                        dt,
                        self.floor,
                        a,
                    ],
                    device=m.device,
                )
            self.record(state_in, 0, dt)

        def measured_iteration(state_in, state_out, contacts, dt, iter_num):
            iterate(state_in, state_out, contacts, dt, iter_num)
            self.record(state_in, iter_num + 1, dt)

        s._initialize_particles = measured_initialize
        s._solve_particle_iteration = measured_iteration

    def record(self, state, row, dt):
        m, s, contacts = self.sim.model, self.sim.solver, self.sim.contacts
        self.elastic.zero_()
        self.damping.zero_()
        self.gap.zero_()
        self.contact.zero_()
        self.hessian.zero_()
        wp.launch(
            gather_particle_body_contact_force_and_hessian,
            m.particle_count,
            inputs=[
                dt,
                self.ids,
                s.particle_q_prev,
                state.particle_q,
                s.friction_epsilon,
                m.particle_radius,
                contacts.soft_contact_indices,
                s._particle_contact_head,
                s._particle_contact_next,
                s.body_particle_contact_penalty_k,
                s.body_particle_contact_material_kd,
                s.body_particle_contact_material_mu,
                m.shape_body,
                state.body_q,
                s.body_q_prev,
                state.body_qd,
                m.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                m.shape_margin,
                contacts.soft_contact_barycentric,
                self.contact,
                self.hessian,
            ],
            device=m.device,
        )
        # All-zero diagnostic colors collect every vertex at the same accepted pose.
        wp.launch(
            accumulate_self_contact_force_and_hessian,
            s.particle_self_contact_evaluation_kernel_launch_size,
            inputs=[
                dt,
                0,
                s.particle_q_prev,
                state.particle_q,
                self.colors,
                m.tri_indices,
                m.edge_indices,
                s.trimesh_collision_info,
                s.particle_self_contact_margin,
                m.soft_contact_ke,
                m.soft_contact_kd,
                m.soft_contact_mu,
                s.friction_epsilon,
                s._self_contact_edge_edge_parallel_epsilon,
                s.particle_self_contact_evaluation_kernel_launch_size,
                self.contact,
                self.hessian,
            ],
            device=m.device,
        )
        dt64 = wp.float64(float(np.float32(dt)))
        wp.launch(
            triangle_forces,
            m.tri_count,
            inputs=[
                state.particle_q,
                s.particle_q_prev,
                m.tri_indices,
                m.tri_poses,
                m.tri_areas,
                m.tri_materials,
                dt64,
                s._particle_elasticity_alm_state,
                self.elastic,
                self.damping,
                self.gap,
            ],
            device=m.device,
        )
        wp.launch(
            bend_forces,
            m.edge_count,
            inputs=[
                state.particle_q,
                s.particle_q_prev,
                m.edge_indices,
                m.edge_rest_angle,
                m.edge_rest_length,
                m.edge_bending_properties,
                dt64,
                s._particle_elasticity_alm_state,
                self.elastic,
                self.damping,
                self.gap,
            ],
            device=m.device,
        )
        wp.launch(
            record_row,
            1,
            inputs=[
                state.particle_q,
                self.previous,
                s.inertia,
                m.particle_mass,
                m.particle_flags,
                self.elastic,
                self.damping,
                self.contact,
                self.gap,
                dt64,
                row,
                self.rows,
            ],
            device=m.device,
        )
        wp.copy(self.previous, state.particle_q)


def initialize_case(stiffness, snapshot, frame, mode, iterations):
    sim = Bag(stiffness, mode != "off")
    sim.solver.iterations = iterations
    sim.state_0.assign(snapshot)
    sim.state_1.assign(snapshot)
    sim.motion.assign(pin_motion((frame + 1) / 60.0))
    wp.launch(
        drive_pins,
        len(sim.pins),
        inputs=[sim.pins, sim.rest, sim.motion, sim.state_0.particle_q, sim.state_0.particle_qd],
        device=sim.model.device,
    )
    sim.state_0.clear_forces()
    sim.pipeline.collide(sim.state_0, sim.contacts)
    return sim


def fingerprint(state):
    h = hashlib.sha256()
    for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
        h.update(getattr(state, name).numpy().tobytes())
    return h.hexdigest()


def solve(sim, floor, measure=True):
    measurement = Measurement(sim, floor) if measure else None
    if measurement:
        measurement.install()
    before = fingerprint(sim.state_0)
    with wp.ScopedCapture(device=sim.model.device) as capture:
        sim.solver.step(sim.state_0, sim.state_1, sim.control, sim.contacts, sim.dt)
    assert before == fingerprint(sim.state_0), "Capture advanced the physical state"
    wp.capture_launch(capture.graph)
    assert np.isfinite(sim.state_1.particle_q.numpy()).all()
    if measurement is None:
        return None
    raw = measurement.rows.numpy()
    assert np.isfinite(raw).all()
    rows = raw.copy()
    for col in (*range(7), 9):
        rows[:, col] = np.sqrt(raw[:, col] / raw[:, 7])
    return rows


def main(args):
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    args.output.mkdir(exist_ok=True, parents=True)
    validation = validate(wp.get_device())
    print("force validation", validation, flush=True)
    cases = []
    for stiffness in args.stiffnesses:
        baseline = Bag(stiffness, False)
        baseline.capture()
        for frame in range(1, max(args.frames) + 1):
            baseline.step()
            if frame not in args.frames:
                continue
            snapshot = baseline.model.state()
            snapshot.assign(baseline.state_0)
            np.savez_compressed(
                args.output / f"checkpoint_ke{stiffness}_frame{frame}.npz",
                **{
                    name: getattr(snapshot, name).numpy() for name in ("particle_q", "particle_qd", "body_q", "body_qd")
                },
            )
            initial_hash = None
            first_residual = None
            for mode, floor in MODES.items():
                sim = initialize_case(stiffness, snapshot, frame, mode, args.iterations)
                sha = fingerprint(sim.state_0)
                if initial_hash is None:
                    initial_hash = sha
                assert sha == initial_hash
                rows = solve(sim, floor)
                if first_residual is None:
                    first_residual = rows[0, 0]
                np.testing.assert_allclose(rows[0, 0], first_residual, rtol=2.0e-5, atol=1.0e-7)
                stem = f"ke{stiffness}_frame{frame}_{mode}"
                np.savetxt(
                    args.output / f"{stem}.csv",
                    np.column_stack((np.arange(len(rows)), rows)),
                    delimiter=",",
                    header="iteration," + ",".join(FIELDS),
                    comments="",
                )
                a = sim.solver._particle_elasticity_alm_state
                rho = {}
                if floor is not None:
                    bending_k = sim.model.edge_bending_properties.numpy()[:, 0] * sim.model.edge_rest_length.numpy()
                    for name, values, k in (
                        ("triangle_stretch", a.tri_rho_stretch.numpy(), stiffness),
                        ("triangle_area", a.tri_rho_area.numpy(), 1.2 * stiffness),
                        ("bending", a.bend_rho.numpy(), bending_k),
                    ):
                        active = values > 0
                        r = values[active] / (k[active] if isinstance(k, np.ndarray) else k)
                        rho[name] = {
                            "min_over_k": float(r.min()),
                            "median_over_k": float(np.median(r)),
                            "max_over_k": float(r.max()),
                        }
                result = {
                    "stiffness": stiffness,
                    "checkpoint_frame": frame,
                    "mode": mode,
                    "floor": floor,
                    "input_state_sha256": sha,
                    "iterations": args.iterations,
                    "rho": rho,
                    "residual_initial_N": float(rows[0, 0]),
                    "residual_iteration10_N": float(rows[10, 0]),
                    "residual_final_N": float(rows[-1, 0]),
                    "gap_iteration10_N": float(rows[10, 2]),
                    "gap_final_N": float(rows[-1, 2]),
                    "soft_contact_count": int(sim.contacts.soft_contact_count.numpy()[0]),
                }
                # A separate uninstrumented native solve verifies read-only diagnostics and the floor9 override.
                if stiffness == args.stiffnesses[0] and frame == args.frames[0] and mode in ("off", "floor9"):
                    plain = initialize_case(stiffness, snapshot, frame, mode, args.iterations)
                    solve(plain, floor, measure=False)
                    difference = float(
                        np.max(np.abs(plain.state_1.particle_q.numpy() - sim.state_1.particle_q.numpy()))
                    )
                    result["instrumented_vs_native_max_position_difference_m"] = difference
                    assert difference < 1.0e-5, difference
                cases.append(result)
                (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
                print(
                    stem,
                    f"r0={rows[0, 0]:.6g} r10={rows[10, 0]:.6g} r{args.iterations}={rows[-1, 0]:.6g} gap10={rows[10, 2]:.6g}",
                    flush=True,
                )
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": str(wp.get_device()),
        "validation": validation,
        "cases": len(cases),
        "floors": MODES,
        "dt_s": 1 / 600,
        "iterations": args.iterations,
        "stiffnesses": args.stiffnesses,
        "checkpoint_frames": args.frames,
        "scene": baseline.params,
        "particles": baseline.model.particle_count,
        "triangles": baseline.model.tri_count,
        "edges": baseline.model.edge_count,
        "pinned_particles": len(baseline.pins),
        "residual": "Free-particle RMS of original float64 elasticity + damping + freshly evaluated native float32 contacts + float64 inertia at accepted positions. After each complete color sweep and dual update.",
        "initialization": "Identical full states from ALM-off baseline; next-frame pin motion; fresh solvers with constitutive stress seeds and cold rigid-contact history.",
        "limitations": "Particle force-balance diagnostic; does not include rigid-body residual or DAT constraint reactions. Contacts retain native scheduling and current penalty parameters. Instrumented timings are not speed benchmarks.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiffnesses", type=int, nargs="+", default=[1000, 10000, 100000, 1000000, 10000000])
    parser.add_argument("--frames", type=int, nargs="+", default=[60, 180])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results-floor-sweep")
    main(parser.parse_args())
