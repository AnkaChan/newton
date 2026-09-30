# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct the documented pinned-bag fixture and compare current ALM elasticity.

The missing May fixture was derived from sources/bag_parent.py. This driver
retains that parent's geometry, contents, and contact parameters, removes its
ground and gripper, and applies the pinned-rim schedule in the saved June notes.
It is a reconstruction, not an exact archived-fixture replay.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from metrics import _build_topology, _dihedral_angles, _measure, _triangle_angles
from sources import bag_parent

import newton
import newton.solvers

ROOT = Path(__file__).resolve().parent
PARENT_REVISION = "be3e604b"


class WithoutGround:
    """Let the archived builder author its bag and contents unchanged."""

    def __init__(self, builder):
        self.builder = builder

    def __getattr__(self, name):
        return getattr(self.builder, name)

    def add_shape_box(self, *args, **kwargs):
        if kwargs.get("label") == "ground":
            return -1
        return self.builder.add_shape_box(*args, **kwargs)


@wp.kernel
def drive_pins(
    indices: wp.array[int],
    rest: wp.array[wp.vec3],
    motion: wp.array[float],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    vertex = indices[wp.tid()]
    q[vertex] = rest[vertex] + wp.vec3(motion[0], 0.0, 0.0)
    qd[vertex] = wp.vec3(motion[1], 0.0, 0.0)


def pin_motion(time_s):
    t = max(0.0, time_s - 1.0)
    ramp = min(t / 0.6, 1.0)
    ramp_rate = 1.0 / 0.6 if 0.0 < t < 0.6 else 0.0
    omega = 2.0 * math.pi * 0.85
    return np.array(
        [
            0.07 * ramp * math.sin(omega * t),
            0.07 * (ramp_rate * math.sin(omega * t) + ramp * omega * math.cos(omega * t)),
        ],
        dtype=np.float32,
    )


class Bag:
    def __init__(self, stiffness, alm, *, rho_scale=1.0):
        self.params = copy.deepcopy(bag_parent.PARAMS)
        self.params.update(
            cloth_tri_ke=stiffness,
            cloth_tri_ka=0.2 * stiffness,
            cloth_tri_kd=0.1,
            settle_frames=60,
            wiggle_frames=300,
            wiggle_axis=0,
            wiggle_amplitude=0.07,
            wiggle_frequency=0.85,
            wiggle_ramp_duration=0.6,
        )
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.info = bag_parent.build_model(WithoutGround(builder), self.params, seed=42)
        pins = self.info["top_global_indices"]
        for vertex in pins:
            builder.particle_flags[vertex] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model = builder.finalize()
        for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
            setattr(self.model, name, self.params[name])
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.rest = wp.clone(self.state_0.particle_q)
        self.pins = wp.array(pins, dtype=int, device=self.model.device)
        self.motion = wp.zeros(2, dtype=float, device=self.model.device)
        self.dt = 1.0 / 600.0
        self.frame = 0
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=10,
            integrate_with_external_rigid_solver=False,
            rigid_body_particle_contact_buffer_size=self.params["rigid_body_particle_contact_buffer_size"],
            rigid_body_contact_buffer_size=self.params["rigid_body_contact_buffer_size"],
            particle_enable_self_contact=True,
            particle_self_contact_radius=self.params["particle_self_contact_radius"],
            particle_self_contact_margin=self.params["particle_self_contact_margin"],
            particle_topological_contact_filter_threshold=3,
            rigid_compliant_alm=False,
            rigid_contact_hard=True,
            particle_elasticity_alm=alm,
            particle_elasticity_alm_deviatoric=False,
            particle_elasticity_alm_rho_scale=rho_scale,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=self.params["soft_contact_creation_margin"],
        )
        self.contacts = self.pipeline.contacts()
        assert self.model.body_count == 3 and self.model.shape_count == 3
        assert self.model.tet_count == 0 and self.model.spring_count == 0
        self.graph = None

    def simulate(self):
        # The saved fixture notes prescribe a frame-level displacement and
        # analytic velocity, written to the pins before every substep.
        for _ in range(10):
            wp.launch(
                drive_pins,
                dim=len(self.pins),
                inputs=[self.pins, self.rest, self.motion],
                outputs=[self.state_0.particle_q, self.state_0.particle_qd],
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self):
        initial = self.state_0.particle_q.numpy()
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        np.testing.assert_array_equal(initial, self.state_0.particle_q.numpy())

    def step(self):
        self.frame += 1
        self.motion.assign(pin_motion(self.frame / 60.0))
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)


def run(args):
    wp.init()
    wp.config.quiet = True
    sim = Bag(args.stiffness, args.alm == "on")
    model = sim.model
    rest = sim.rest.numpy().astype(np.float64)
    topology = _build_topology(model.tri_indices.numpy())
    rest_lengths = np.linalg.norm(rest[topology.edges[:, 1]] - rest[topology.edges[:, 0]], axis=1)
    rest_angles = _triangle_angles(rest, topology.faces)
    rest_dihedrals = _dihedral_angles(rest, topology.faces, topology.bend_pairs)
    digest = hashlib.sha256()
    for value in (
        model.particle_q,
        model.particle_mass,
        model.particle_flags,
        model.tri_indices,
        model.tri_materials,
        model.edge_bending_properties,
        model.body_q,
        model.body_mass,
    ):
        digest.update(value.numpy().tobytes())
    sim.capture()
    positions, bodies, rows = [], [], []
    max_pin_error = 0.0
    for frame in range(args.frames + 1):
        points = sim.state_0.particle_q.numpy()
        body_q = sim.state_0.body_q.numpy()
        assert np.isfinite(points).all() and np.isfinite(body_q).all(), f"nonfinite frame {frame}"
        expected = rest[sim.info["top_global_indices"]].copy()
        expected[:, 0] += pin_motion(frame / 60.0)[0]
        max_pin_error = max(max_pin_error, float(np.abs(points[sim.info["top_global_indices"]] - expected).max()))
        stretch, shear, bend = _measure(points.astype(np.float64), topology, rest_lengths, rest_angles, rest_dihedrals)
        rows.append(
            {"frame": frame, "time_s": frame / 60.0, "stretch_score": stretch, "shear_score": shear, "bend_score": bend}
        )
        positions.append(points)
        bodies.append(body_q)
        if frame % 60 == 0:
            print(
                f"ke={args.stiffness:g} ALM={args.alm} frame={frame} stretch={stretch:.6g} bend={bend:.6g}", flush=True
            )
        if frame < args.frames:
            sim.step()
    assert max_pin_error < 1e-6, max_pin_error
    state = sim.solver._particle_elasticity_alm_state
    alm_stats = {}
    if args.alm == "on":
        active = model.edge_indices.numpy()[:, 0] >= 0
        rho = state.bend_rho.numpy()[active]
        history = state.bend_lambda.numpy()[active]
        assert np.isfinite(rho).all() and np.isfinite(history).all()
        assert np.max(np.abs(history)) > 0.0
        alm_stats = {
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "history_abs_max": float(np.abs(history).max()),
        }
        for name in ("tri_lambda_stretch", "tri_lambda_area", "tri_rho_stretch", "tri_rho_area"):
            values = getattr(state, name).numpy()
            assert np.isfinite(values).all() and np.any(values != 0.0), name
            alm_stats[name] = {"min": float(values.min()), "max": float(values.max())}
    result = {
        "stiffness": args.stiffness,
        "alm": args.alm,
        "alm_scope": "triangle_and_bending",
        "params": sim.params,
        "frames": args.frames,
        "solver_steps": args.frames * 10,
        "iterations_per_step": 10,
        "rho_scale": 1.0,
        "newton_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "parent_revision": subprocess.check_output(["git", "rev-parse", PARENT_REVISION], text=True).strip(),
        "geometry_source_sha256": hashlib.sha256((ROOT / "sources/bag_parent.py").read_bytes()).hexdigest(),
        "reconstruction": True,
        "model_sha256": digest.hexdigest(),
        "device": str(model.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "particle_count": model.particle_count,
        "triangle_count": model.tri_count,
        "edge_count": model.edge_count,
        "pin_count": len(sim.pins),
        "body_count": model.body_count,
        "max_pin_error_m": max_pin_error,
        "alm_stats": alm_stats,
        "summary": {
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "final": rows[-1][key],
                "peak": max(r[key] for r in rows),
            }
            for key in ("stretch_score", "shear_score", "bend_score")
        },
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"ke{int(args.stiffness):d}_{args.alm}"
    (args.output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / f"{stem}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / f"{stem}.npz", particle_q=np.array(positions), body_q=np.array(bodies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiffness", type=float, required=True)
    parser.add_argument("--alm", choices=["off", "on"], required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--output", type=Path, default=ROOT / "results-triangle-bending")
    run(parser.parse_args())
