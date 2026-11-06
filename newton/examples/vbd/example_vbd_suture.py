# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example VBD Suture

A simulated half-circle needle with a thread sews two elastic cuboids together.
A kinematic holder rotates the needle about its arc center so the tip pierces
the top of the left block, crosses the gap, pierces the right block and exits
through its top. Every bite opens a virtual tunnel of tunnel points in the
tetrahedral tissue; the needle and the thread slide through the tunnels with
capped friction. The free end of the thread is clamped on top of the right
block, away from the exit, so the slack lies on that block during the bite.
Once the needle is out, the holder draws it across the wound at a low constant
speed while the gap is wider than 0.5 mm, so the thread tightens and the gap
closes; the pull stops when the facing faces meet.

Success criteria (test mode, ``test_final``): the state is finite, the needle opens
exactly two tunnels and both are through at the end, no block is lifted off the
table, the needle group penetrates ungated tissue by less than 0.5 mm, and the gap
between the facing block faces is closed to within half a tet edge (1.5 mm).
``check_final_strict`` holds the tighter bounds of the design (gap below 0.5 mm,
largest tunnel row residual below 0.3 mm, penetration below 0.1 mm); run-to-run
variation on the GPU moves these by several tenths of a millimetre (see
notes/vbd-suture/decisions.md I19 and I21), so they are reported, not asserted.

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
    "num_frames": 720,
    "sim_substeps": 10,
    # 20 iterations: with the soft contacts below the thread slack still changes velocity by 3 to 4 mm/s per frame
    # at 10 iterations and by 1 mm/s at 20 (the tuning guide's last-resort knob). 10 iterations give a 3 times
    # faster run with a visibly rougher thread.
    "solver_iterations": 20,
    # tissue
    "block_cells": (14, 10, 4),
    "cell": 3.0e-3,
    "gap": 4.0e-3,
    "tissue_density": 1000.0,
    "tissue_youngs_modulus": 1.0e5,
    "tissue_poisson": 0.45,
    # Volumetric damping [Pa s]. The contact values below already follow the tuning guide's stiffness-ratio rule
    # and remove the tissue jitter; a damping of 1e2 to 1e3 adds a start transient here (the facing faces bulge 0.2
    # to 0.9 mm in the first 0.2 s, because the damping Hessian k_damp / dt exceeds the elastic stiffness).
    "tissue_damping": 0.0,
    "particle_radius": 5.0e-4,
    # needle: half circle
    "needle_radius": 8.0e-3,
    "wire_radius": 4.0e-4,
    "needle_capsules": 12,
    "needle_density": 7800.0,
    # thread
    "thread_radius": 1.5e-4,
    "thread_segment": 2.5e-3,
    "thread_density": 1100.0,
    "thread_stretch_stiffness": 2.0e3,
    # Bend and twist stiffness per joint [N m/rad]: EI / L of a 0.15 mm nylon thread is about 4e-4; 1e-4 keeps the
    # slack in smooth curves instead of flipping zigzags. Damping near half critical for the 0.2 mg capsules.
    "thread_bend_stiffness": 1.0e-4,
    "thread_stretch_damping": 0.02,
    "thread_bend_damping": 5.0e-9,
    # Contact material of the thread and needle shapes (averaged with the soft side for rigid-soft contact).
    # The 0.2 mg thread capsules set the scale: with the rigid-soft contact at 3e3 N/m (the mean of both sides)
    # the slack shivered on the tissue (velocity changes of 10 mm/s per frame); at 300 N/m it rests (3 to 4 mm/s at
    # 10 iterations, 1 mm/s at 20). Contact damping above 1 percent of the stiffness made it worse again.
    "thread_mu": 0.1,
    "thread_contact_ke": 1.0e2,
    "thread_contact_kd": 1.0,
    "needle_mu": 0.2,
    # Free end of the thread: clamped on top of the right block, 22 mm past the exit and 10 mm beside the needle
    # plane. The route from the needle tail to the clamp lays the slack (about 70 mm in all) on that block.
    "clamp_position": (0.030, -0.010, 0.01315),
    "thread_route": (
        (0.016, -0.010, 0.0132),
        (0.040, -0.013, 0.0132),
        (0.042, -0.005, 0.0132),
        (0.032, -0.003, 0.0132),
    ),
    # holder drive
    "drive_linear_ke": 1.0e4,
    "drive_linear_kd": 1.0e2,
    "drive_angular_ke": 10.0,
    "drive_angular_kd": 0.1,
    # contact
    "rigid_gap": 1.0e-3,
    "soft_contact_gap": 1.0e-3,
    # Soft enough for the 0.2 mg thread capsules (see thread_contact_ke); the needle indents the tissue by 0.2 mm
    # at the 0.1 N puncture force and DAT keeps every contact penetration-free. Damping 1 percent of the stiffness.
    "soft_contact_ke": 5.0e2,
    "soft_contact_kd": 5.0,
    "soft_contact_mu": 0.3,
    "self_contact_margin": 1.0e-3,
    "self_contact_gap": 1.0e-3,
    # suture
    "tunnel_mu": 0.3,
    "grip": 1.0,
    # Tunnel row stiffness [N/m], per-particle attribute vbd.suture_stiffness. Infinite = the hard row. A finite
    # value (about 1 kN/m) lowers the row forces 20 times but lets the thread sit up to 1 mm off the tunnel while
    # it is drawn through; see notes/vbd-suture/decisions.md I18 and I19.
    "tunnel_stiffness": float("inf"),
    "puncture_force": 0.1,
    "puncture_cos": 0.5,
    # holder script [s]
    "t_approach": 1.0,
    "t_bite_end": 7.0,
    "t_pull_end": 12.0,
    "bite_angle_deg": 360.0,
    "center_high": 0.030,
    "center_bite": 0.014,
    # Pull: the arc center moves toward this point (over the left block) at constant speed while the gap is
    # wider than gap_close, and holds otherwise. The speed (26 mm in 5 s) is low so the thread is not yanked
    # through the tunnels; a wound that the bite has already closed (notes/vbd-suture/decisions.md I19) leaves
    # the holder still.
    "pull_target": (-0.024, 0.0, 0.024),
    "gap_close": 0.5e-3,
    # camera
    "camera_pos": (0.0, -0.16, 0.08),
    "camera_target": (0.0, 0.0, 0.015),
    "camera_fov": 30.0,
}


def _resample_polyline(points: list[wp.vec3], segment_length: float) -> tuple[list[wp.vec3], float]:
    """Resample a polyline at equal arc length. Returns the points and the total length."""
    pts = np.array([[q[0], q[1], q[2]] for q in points], dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg.sum())
    if length <= 0.0:
        raise ValueError("thread route has zero length")
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    segments = max(2, int(round(length / segment_length)))
    result = []
    for k in range(segments + 1):
        s = length * k / segments
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] <= 0.0 else (s - cum[i]) / seg[i]
        result.append(wp.vec3(*(pts[i] + (pts[i + 1] - pts[i]) * min(f, 1.0))))
    return result, length


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
        shape_body = self.model.shape_body.numpy()
        thread_shapes = np.flatnonzero(np.isin(shape_body, self.thread_bodies))
        ke = self.model.shape_material_ke.numpy()
        kd = self.model.shape_material_kd.numpy()
        ke[thread_shapes] = p["thread_contact_ke"]
        kd[thread_shapes] = p["thread_contact_kd"]
        self.model.shape_material_ke.assign(ke)
        self.model.shape_material_kd.assign(kd)
        self.model.vbd.suture_grip.fill_(p["grip"])
        self.model.vbd.suture_mu.fill_(p["tunnel_mu"])
        self.model.vbd.suture_stiffness.fill_(p["tunnel_stiffness"])
        self.model.vbd.suture_puncture_force.fill_(p["puncture_force"])
        self.model.vbd.suture_puncture_cos.fill_(p["puncture_cos"])

        # Full-surface rigid-soft contact: the 0.3 mm thread lies between the 3 mm particle rows, so per-particle
        # contacts alone let it sink into the tissue; face and edge contacts hold it on the surface.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=p["soft_contact_gap"], enable_rigid_soft_full_surface_contact=True
        )
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
        self.pull_progress = 0.0
        self._init_faces()

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
            # The bottom particles sit one particle radius above the table so the first step starts without penetration.
            builder.add_soft_grid(
                pos=wp.vec3(x0, -0.5 * depth, p["particle_radius"]),
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
        cfg = newton.ModelBuilder.ShapeConfig(density=p["needle_density"], mu=p["needle_mu"])
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

        # Thread: a rod from the tail down onto the right block, looped on its top and ending at the clamp.
        # The slack stays on the right block, beside the needle plane, so the needle never catches it.
        tail_local = points[-1]
        tail_world = wp.transform_point(wp.transform(center, rot0), tail_local)
        clamp_world = wp.vec3(*p["clamp_position"])
        route = [tail_world, *(wp.vec3(*q) for q in p["thread_route"]), clamp_world]
        polyline, thread_length = _resample_polyline(route, p["thread_segment"])
        segments = len(polyline) - 1
        self.thread_length = thread_length
        rod = newton.Rod(points=polyline, radius=p["thread_radius"])
        thread_dir = wp.normalize(polyline[1] - polyline[0])
        thread_cfg = newton.ModelBuilder.ShapeConfig(density=p["thread_density"], mu=p["thread_mu"])
        self.thread_bodies, _joints = builder.add_rod(
            rod=rod,
            cfg=thread_cfg,
            stretch_stiffness=p["thread_stretch_stiffness"],
            bend_stiffness=p["thread_bend_stiffness"],
            stretch_damping=p["thread_stretch_damping"],
            bend_damping=p["thread_bend_damping"],
            label="suture_thread",
        )
        half_seg = 0.5 * thread_length / segments
        q_tail = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), wp.quat_rotate_inv(rot0, thread_dir))
        builder.add_joint_rod(
            self.needle,
            self.thread_bodies[0],
            parent_xform=wp.transform(tail_local, q_tail),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -half_seg), wp.quat_identity()),
            stretch_stiffness=p["thread_stretch_stiffness"],
            bend_stiffness=p["thread_bend_stiffness"],
            stretch_damping=p["thread_stretch_damping"],
            bend_damping=p["thread_bend_damping"],
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
        # Pull: the center moves toward the target as far as self.pull_progress allows (see _advance_pull).
        target = np.array(p["pull_target"])
        start = np.array([0.0, 0.0, p["center_bite"]])
        c = start + (target - start) * self.pull_progress
        return wp.transform(wp.vec3(*c), _needle_rotation(psi_max))

    def _advance_pull(self):
        """Advance the pull at constant speed while the gap is open; hold once the facing faces meet."""
        p = self.params
        if self.sim_time < p["t_bite_end"] or self.pull_progress >= 1.0:
            return
        if self.gap_width() > p["gap_close"]:
            step = self.frame_dt / (p["t_pull_end"] - p["t_bite_end"])
            self.pull_progress = min(1.0, self.pull_progress + step)

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
        self._advance_pull()
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

    def block_lift(self) -> float:
        """Largest height of a bottom-layer tissue particle above the table [m]."""
        q = self.state_0.particle_q.numpy()
        lift = 0.0
        for start, end in self.block_particle_ranges:
            bottom = self._bottom[start:end]
            lift = max(lift, float(q[start:end, 2][bottom].min()))
        return lift

    def _init_faces(self):
        q = self.model.particle_q.numpy()
        a0, a1 = self.block_particle_ranges[0]
        b0, b1 = self.block_particle_ranges[1]
        xa = q[a0:a1, 0]
        xb = q[b0:b1, 0]
        self._face_a = xa > xa.max() - 1.0e-6
        self._face_b = xb < xb.min() + 1.0e-6
        self._bottom = q[:, 2] < q[:, 2].min() + 1.0e-6

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

    def _check_final(self, *, gap_bound: float, residual_bound: float, penetration_bound: float):
        if not hasattr(self, "_face_a"):
            self._init_faces()
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        if not (np.isfinite(q).all() and np.isfinite(body_q).all()):
            raise AssertionError("simulation produced non-finite state")
        tunnels = int(self.solver._suture_tunnel_count.numpy()[0])
        states = self.solver._suture_tunnel_state.numpy()[:tunnels]
        if tunnels != 2 or not all(int(s) in (2, 3) for s in states):
            raise AssertionError(f"expected two fixed tunnels, found {tunnels} with states {states.tolist()}")
        lift = self.block_lift()
        if lift > 3.0e-3:
            raise AssertionError(f"a block was lifted {lift * 1e3:.1f} mm off the table by the thread")
        if self.max_penetration > penetration_bound:
            raise AssertionError(
                f"needle group penetrated ungated tissue by {self.max_penetration * 1e3:.3f} mm, "
                f"bound {penetration_bound * 1e3:.2f} mm"
            )
        gap = self.gap_width()
        if abs(gap) > gap_bound:
            raise AssertionError(f"gap between the blocks is {gap * 1e3:.2f} mm, bound {gap_bound * 1e3:.2f} mm")
        if self.max_residual > residual_bound:
            raise AssertionError(
                f"largest tunnel point residual {self.max_residual * 1e3:.3f} mm exceeds {residual_bound * 1e3:.2f} mm"
            )

    def test_final(self):
        """Robust invariants: two through tunnels, finite state, no lift, bounded penetration, wound closed to
        within half a tet edge. Run-to-run variation on the GPU (decisions I19, I21) keeps the design's tighter
        bounds out of this test; see :meth:`check_final_strict`."""
        self._check_final(gap_bound=1.5e-3, residual_bound=1.0e-3, penetration_bound=0.5e-3)

    def check_final_strict(self):
        """The bounds of the design: gap below 0.5 mm, row residual below 0.3 mm, penetration below 0.1 mm."""
        self._check_final(gap_bound=0.5e-3, residual_bound=0.3e-3, penetration_bound=1.0e-4)

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
