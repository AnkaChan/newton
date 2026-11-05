# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example VBD Suture

A simulated half-circle needle with a thread sews two elastic cuboids together.
A kinematic holder rotates the needle about its arc center so the tip pierces
the top of the left block, crosses the gap, pierces the right block and exits
through its top. Every bite opens a virtual tunnel of tunnel points in the
tetrahedral tissue; the needle and the thread slide through the tunnels with
capped friction. The holder then lifts and pulls the needle so the thread,
clamped at its free end, tightens and closes the gap.

Success criteria (test mode):
- the gap between the two facing block faces closes below 0.5 mm,
- the largest transverse residual of any active tunnel point row stays below 0.2 mm,
- the needle group never penetrates an ungated tissue particle by more than 0.1 mm,
- the needle opens exactly two tunnels and both are through at the end.

Runs:
- `python -m newton.examples vbd_suture`
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD

PARAMS = {
    "fps": 60,
    "num_frames": 600,
    "sim_substeps": 10,
    "solver_iterations": 10,
    # tissue
    "block_cells": (14, 10, 4),
    "cell": 3.0e-3,
    "gap": 4.0e-3,
    "tissue_density": 1000.0,
    "tissue_youngs_modulus": 1.0e5,
    "tissue_poisson": 0.45,
    "tissue_damping": 0.0,
    "particle_radius": 5.0e-4,
    # needle: half circle
    "needle_radius": 8.0e-3,
    "wire_radius": 4.0e-4,
    "needle_capsules": 12,
    "needle_density": 7800.0,
    # thread
    "thread_radius": 1.5e-4,
    "thread_length": 0.10,
    "thread_segment": 2.5e-3,
    "thread_density": 1100.0,
    "thread_stretch_stiffness": 2.0e3,
    "thread_bend_stiffness": 1.0e-6,
    "clamp_position": (-0.0077, 0.0, 0.060),
    # holder drive
    "drive_linear_ke": 1.0e4,
    "drive_linear_kd": 1.0e2,
    "drive_angular_ke": 10.0,
    "drive_angular_kd": 0.1,
    # contact
    "rigid_gap": 1.0e-3,
    "soft_contact_gap": 1.0e-3,
    "soft_contact_ke": 2.0e4,
    "soft_contact_kd": 1.0,
    "soft_contact_mu": 0.3,
    "self_contact_margin": 1.0e-3,
    "self_contact_gap": 1.0e-3,
    # suture
    "tunnel_mu": 0.3,
    "grip": 1.0,
    "puncture_force": 0.1,
    "puncture_cos": 0.5,
    # holder script [s]
    "t_approach": 1.0,
    "t_bite_end": 7.0,
    "t_pull_end": 9.5,
    "bite_angle_deg": 340.0,
    "center_high": 0.030,
    "center_bite": 0.014,
    "pull_target": (0.008, 0.0, 0.060),
    # camera
    "camera_pos": (0.0, -0.16, 0.08),
    "camera_target": (0.0, 0.0, 0.015),
    "camera_fov": 30.0,
}


def _thread_polyline(start: wp.vec3, end: wp.vec3, length: float, segments: int) -> list[wp.vec3]:
    """Points of a two-leg polyline of total ``length``: from ``start`` up along +z, then straight to ``end``."""
    start_np = np.array([start[0], start[1], start[2]], dtype=np.float64)
    end_np = np.array([end[0], end[1], end[2]], dtype=np.float64)
    direct = float(np.linalg.norm(end_np - start_np))
    if length <= direct + 1.0e-6:
        raise ValueError("thread_length must exceed the distance from the needle tail to the clamp")
    lo, hi = 0.0, length
    for _ in range(60):
        rise = 0.5 * (lo + hi)
        knee = start_np + np.array([0.0, 0.0, rise])
        if rise + float(np.linalg.norm(end_np - knee)) > length:
            hi = rise
        else:
            lo = rise
    knee = start_np + np.array([0.0, 0.0, lo])
    leg2 = float(np.linalg.norm(end_np - knee))
    result = []
    for k in range(segments + 1):
        s = length * k / segments
        if s <= lo or leg2 <= 0.0:
            q = start_np + np.array([0.0, 0.0, min(s, lo)])
        else:
            q = knee + (end_np - knee) * min((s - lo) / leg2, 1.0)
        result.append(wp.vec3(*q))
    return result


def _needle_rotation(psi: float) -> wp.quat:
    """Rotation about -y: the tip, at -x in the body frame, descends as ``psi`` grows."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, -1.0, 0.0), float(psi))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.frame_dt = 1.0 / self.params["fps"]
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        builder = newton.ModelBuilder()
        # Millimetre-scale scene: the default rigid detection gap would pair every thread capsule with every other.
        builder.rigid_gap = self.params["rigid_gap"]
        SolverVBD.register_custom_attributes(builder)
        self._build_table(builder)
        self._build_tissue(builder)
        self._build_needle_and_thread(builder)
        builder.color()
        self.model = builder.finalize()

        p = self.params
        self.model.soft_contact_ke = p["soft_contact_ke"]
        self.model.soft_contact_kd = p["soft_contact_kd"]
        self.model.soft_contact_mu = p["soft_contact_mu"]
        self.model.shape_material_ke.fill_(p["soft_contact_ke"])
        self.model.shape_material_kd.fill_(p["soft_contact_kd"])
        self.model.vbd.suture_grip.fill_(p["grip"])
        self.model.vbd.suture_mu.fill_(p["tunnel_mu"])
        self.model.vbd.suture_puncture_force.fill_(p["puncture_force"])
        self.model.vbd.suture_puncture_cos.fill_(p["puncture_cos"])

        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=p["soft_contact_gap"])
        self.solver = SolverVBD(
            self.model,
            iterations=p["solver_iterations"],
            rigid_soft_enable_dat=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=p["self_contact_margin"],
            particle_self_contact_gap=p["self_contact_gap"],
            collision_pipeline=self.collision_pipeline,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.max_penetration = 0.0
        self.max_residual = 0.0
        self.tunnels_seen = 0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(*p["camera_pos"]), 0.0, 0.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = p["camera_fov"]
        self._look_at(wp.vec3(*p["camera_pos"]), wp.vec3(*p["camera_target"]))
        self.capture()

    # ── model construction ──────────────────────────────────────────────

    def _build_table(self, builder):
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, density=0.0)
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.005), wp.quat_identity()), hx=0.15, hy=0.10, hz=0.005, cfg=cfg
        )

    def _build_tissue(self, builder):
        p = self.params
        nx, ny, nz = p["block_cells"]
        cell = p["cell"]
        E, nu = p["tissue_youngs_modulus"], p["tissue_poisson"]
        k_mu = E / (2.0 * (1.0 + nu))
        k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        width = nx * cell
        depth = ny * cell
        self.block_width = width
        self.block_particle_ranges = []
        for sign in (-1.0, 1.0):
            x0 = sign * (0.5 * p["gap"]) + (-width if sign < 0 else 0.0)
            start = builder.particle_count
            builder.add_soft_grid(
                pos=wp.vec3(x0, -0.5 * depth, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=nx,
                dim_y=ny,
                dim_z=nz,
                cell_x=cell,
                cell_y=cell,
                cell_z=cell,
                density=p["tissue_density"],
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=p["tissue_damping"],
                particle_radius=p["particle_radius"],
            )
            self.block_particle_ranges.append((start, builder.particle_count))
        self.tissue_height = nz * cell

    def _build_needle_and_thread(self, builder):
        p = self.params
        R = p["needle_radius"]
        r = p["wire_radius"]
        center = wp.vec3(0.0, 0.0, p["center_high"])
        rot0 = _needle_rotation(0.0)

        # Holder: a kinematic body at the arc center, driven by the script.
        self.holder = builder.add_body(xform=wp.transform(center, rot0), label="suture_holder")

        # Needle: the upper half circle in the body frame, tip at -x, tail at +x.
        self.needle = builder.add_body(xform=wp.transform(center, rot0), label="suture_needle")
        count = p["needle_capsules"]
        angles = [math.pi - math.pi * k / count for k in range(count + 1)]
        points = [wp.vec3(R * math.cos(a), 0.0, R * math.sin(a)) for a in angles]
        cfg = newton.ModelBuilder.ShapeConfig(density=p["needle_density"], mu=0.2)
        self.needle_shapes = []
        for a, b in itertools.pairwise(points):
            mid = 0.5 * (a + b)
            d = b - a
            q = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), wp.normalize(d))
            half = max(0.5 * float(wp.length(d)) - 0.5 * r, 0.25 * r)
            self.needle_shapes.append(
                builder.add_shape_capsule(self.needle, xform=wp.transform(mid, q), radius=r, half_height=half, cfg=cfg)
            )
        tip_shape = self.needle_shapes[0]

        linear = [
            newton.ModelBuilder.JointDofConfig(
                axis=axis, target_ke=p["drive_linear_ke"], target_kd=p["drive_linear_kd"]
            )
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ]
        angular = [
            newton.ModelBuilder.JointDofConfig(
                axis=axis, target_ke=p["drive_angular_ke"], target_kd=p["drive_angular_kd"]
            )
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ]
        builder.add_joint_d6(
            self.holder, self.needle, linear_axes=linear, angular_axes=angular, label="suture_holder_drive"
        )

        # Thread: a rod from the tail straight up, then over to the clamp above the entry of the left
        # block, so both thread ends rise above the wound once the bite is done (like a simple stitch).
        tail_local = points[-1]
        tail_world = wp.transform_point(wp.transform(center, rot0), tail_local)
        segments = int(round(p["thread_length"] / p["thread_segment"]))
        clamp_world = wp.vec3(*p["clamp_position"])
        rod = newton.Rod(
            points=_thread_polyline(tail_world, clamp_world, p["thread_length"], segments),
            radius=p["thread_radius"],
        )
        thread_dir = wp.vec3(0.0, 0.0, 1.0)
        thread_cfg = newton.ModelBuilder.ShapeConfig(density=p["thread_density"], mu=0.3)
        self.thread_bodies, _joints = builder.add_rod(
            rod=rod,
            cfg=thread_cfg,
            stretch_stiffness=p["thread_stretch_stiffness"],
            bend_stiffness=p["thread_bend_stiffness"],
            label="suture_thread",
        )
        half_seg = 0.5 * p["thread_length"] / segments
        q_tail = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), wp.quat_rotate_inv(rot0, thread_dir))
        builder.add_joint_rod(
            self.needle,
            self.thread_bodies[0],
            parent_xform=wp.transform(tail_local, q_tail),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -half_seg), wp.quat_identity()),
            stretch_stiffness=p["thread_stretch_stiffness"],
            bend_stiffness=p["thread_bend_stiffness"],
            label="suture_swage",
        )
        clamp = builder.add_body(xform=wp.transform(clamp_world, wp.quat_identity()), label="suture_clamp")
        builder.add_joint_fixed(
            clamp,
            self.thread_bodies[-1],
            child_xform=wp.transform(wp.vec3(0.0, 0.0, half_seg), wp.quat_identity()),
            label="suture_clamp_joint",
        )
        builder.add_suture_needle(
            self.needle, tip_shape, centerline=points, thread_bodies=self.thread_bodies, max_tunnels=4
        )

    # ── holder script ───────────────────────────────────────────────────

    def _holder_pose(self, t: float) -> wp.transform:
        p = self.params
        if t < p["t_approach"]:
            s = t / p["t_approach"]
            z = p["center_high"] + (p["center_bite"] - p["center_high"]) * s
            return wp.transform(wp.vec3(0.0, 0.0, z), _needle_rotation(0.0))
        psi_max = math.radians(p["bite_angle_deg"])
        if t < p["t_bite_end"]:
            s = (t - p["t_approach"]) / (p["t_bite_end"] - p["t_approach"])
            return wp.transform(wp.vec3(0.0, 0.0, p["center_bite"]), _needle_rotation(psi_max * s))
        if t < p["t_pull_end"]:
            s = (t - p["t_bite_end"]) / (p["t_pull_end"] - p["t_bite_end"])
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            target = np.array(p["pull_target"])
            start = np.array([0.0, 0.0, p["center_bite"]])
            c = start + (target - start) * s
            return wp.transform(wp.vec3(*c), _needle_rotation(psi_max))
        return wp.transform(wp.vec3(*p["pull_target"]), _needle_rotation(psi_max))

    def _set_holder(self, t: float):
        pose = self._holder_pose(t)
        body_q = self.state_0.body_q.numpy()
        body_q[self.holder] = pose
        self.state_0.body_q.assign(body_q)

    # ── stepping ────────────────────────────────────────────────────────

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.sim_substeps % 2 == 0:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def step(self):
        self._set_holder(self.sim_time)
        self.frame += 1
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ── validation ──────────────────────────────────────────────────────

    def _look_at(self, pos, target):
        if not hasattr(self.viewer, "camera"):
            return
        d = target - pos
        dist_xy = math.hypot(d[0], d[1])
        self.viewer.camera.yaw = math.degrees(math.atan2(d[1], d[0]))
        self.viewer.camera.pitch = math.degrees(math.atan2(d[2], dist_xy))

    def gap_width(self) -> float:
        """Signed distance between the facing block faces, from their initial face particles."""
        q = self.state_0.particle_q.numpy()
        a0, a1 = self.block_particle_ranges[0]
        b0, b1 = self.block_particle_ranges[1]
        return float(q[b0:b1, 0][self._face_b].min() - q[a0:a1, 0][self._face_a].max())

    def _init_faces(self):
        q = self.model.particle_q.numpy()
        a0, a1 = self.block_particle_ranges[0]
        b0, b1 = self.block_particle_ranges[1]
        xa = q[a0:a1, 0]
        xb = q[b0:b1, 0]
        self._face_a = xa > xa.max() - 1.0e-6
        self._face_b = xb < xb.min() + 1.0e-6

    def max_row_residual(self) -> float:
        """Largest transverse distance from an active tunnel point to its rod place [m]."""
        solver = self.solver
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        tets = solver._suture_point_tet.numpy()
        active = solver._suture_point_active.numpy()
        segment = solver._suture_point_segment.numpy()
        if not ((tets >= 0) & (active != 0) & (segment >= 0)).any():
            return 0.0
        bary = solver._suture_point_bary.numpy()
        u = solver._suture_point_u.numpy()
        tangent = solver._suture_point_tangent.numpy()
        tet_indices = self.model.tet_indices.numpy().reshape(-1, 4)
        seg_body = self.model.suture_segment_body.numpy()
        seg_a = self.model.suture_segment_local_a.numpy()
        seg_b = self.model.suture_segment_local_b.numpy()
        worst = 0.0
        for i in np.flatnonzero((tets >= 0) & (active != 0) & (segment >= 0)):
            x = sum(bary[i][k] * q[tet_indices[tets[i]][k]] for k in range(4))
            s = segment[i]
            xf = wp.transform(*body_q[seg_body[s]])
            local = seg_a[s] + u[i] * (seg_b[s] - seg_a[s])
            place = np.array(wp.transform_point(xf, wp.vec3(*local)))
            t = tangent[i]
            d = x - place
            d -= np.dot(d, t) * t
            worst = max(worst, float(np.linalg.norm(d)))
        return worst

    def needle_penetration(self) -> float:
        """Deepest penetration of the needle group's capsules into ungated tissue particles [m]."""
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        gate = self.model.particle_contact_gate.numpy()
        shape_gate = self.model.shape_contact_gate.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_xf = self.model.shape_transform.numpy()
        shape_scale = self.model.shape_scale.numpy()
        radius = self.model.particle_radius.numpy()
        tissue = np.zeros(len(q), dtype=bool)
        for start, end in self.block_particle_ranges:
            tissue[start:end] = True
        free = tissue & (gate == 0)
        if not free.any():
            return 0.0
        pts = q[free]
        rad = radius[free]
        worst = 0.0
        for shape in np.flatnonzero(shape_gate != 0):
            body = shape_body[shape]
            xf = wp.transform_multiply(wp.transform(*body_q[body]), wp.transform(*shape_xf[shape]))
            r_cap, half = float(shape_scale[shape][0]), float(shape_scale[shape][1])
            a = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, -half)))
            b = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, half)))
            d = b - a
            dd = float(np.dot(d, d))
            t = np.clip(((pts - a) @ d) / dd, 0.0, 1.0) if dd > 0.0 else np.zeros(len(pts))
            closest = a + t[:, None] * d
            dist = np.linalg.norm(pts - closest, axis=1)
            worst = max(worst, float(np.max(r_cap + rad - dist)))
        return max(worst, 0.0)

    def test_post_step(self):
        if not hasattr(self, "_face_a"):
            self._init_faces()
        if self.frame % 10 == 0:
            self.max_penetration = max(self.max_penetration, self.needle_penetration())
            self.max_residual = max(self.max_residual, self.max_row_residual())
            self.tunnels_seen = max(self.tunnels_seen, int(self.solver._suture_tunnel_count.numpy()[0]))

    def test_final(self):
        if not hasattr(self, "_face_a"):
            self._init_faces()
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        if not (np.isfinite(q).all() and np.isfinite(body_q).all()):
            raise AssertionError("simulation produced non-finite state")
        gap = self.gap_width()
        if gap > 0.5e-3:
            raise AssertionError(f"gap between the blocks is {gap * 1e3:.2f} mm, expected below 0.5 mm")
        if self.max_residual > 0.2e-3:
            raise AssertionError(f"largest tunnel point residual {self.max_residual * 1e3:.3f} mm exceeds 0.2 mm")
        if self.max_penetration > 1.0e-4:
            raise AssertionError(f"needle group penetrated ungated tissue by {self.max_penetration * 1e3:.3f} mm")
        tunnels = int(self.solver._suture_tunnel_count.numpy()[0])
        states = self.solver._suture_tunnel_state.numpy()[:tunnels]
        if tunnels != 2 or not all(int(s) in (2, 3) for s in states):
            raise AssertionError(f"expected two fixed tunnels, found {tunnels} with states {states.tolist()}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
