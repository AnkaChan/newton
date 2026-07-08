# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD DAT — Rigid-Rigid Impact (penetration-free truncation)
#
# A heavy sphere is shot into a pyramid of boxes resting on the ground.
# Every rigid-rigid contact reported by the collision pipeline defines a
# division plane between the two bodies (Divide and Truncate); with
# ``rigid_enable_penetration_free=True`` the pose updates of both bodies
# are truncated against these planes along their curved trajectories, so
# the impact scatters the boxes without bodies sinking into each other or
# into the ground. Set ``"enable_dat": False`` in PARAMS to compare.
#
# The scene sets ``ModelBuilder.rigid_gap`` so contacts (and thus division
# planes) are emitted before first touch, and a small shape ``margin`` so
# penalty forces engage in a shell outside the geometric surface that DAT
# protects.
#
# Command: python -m newton.examples vbd_dat_rigid_rigid
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

PARAMS = {
    # simulation
    "fps": 60,
    "sim_substeps": 4,
    "solver_iterations": 8,
    "num_frames": 300,
    # rigid DAT penetration-free truncation (the feature under demonstration)
    "enable_dat": True,
    # rigid contact detection gap = DAT query margin: contacts and division planes
    # appear at this separation, and per-substep body motion is capped at
    # 0.5 * relaxation * gap — it must exceed the fastest per-substep body motion
    # (|dx| + |omega| * bounding_radius), or DAT drains the body's momentum
    "rigid_gap": 0.12,
    # penalty-force shell outside the geometric surface. Contact stiffness must stop a
    # body within this shell (stopping distance ~ v * sqrt(m / ke) < margin), otherwise
    # bodies reach the geometric surface and DAT freezes them there (momentum drain)
    "shape_margin": 0.02,
    "shape_ke": 1.0e6,
    # box pyramid
    "box_half": 0.15,
    "box_mass": 1.0,
    "pyramid_rows": 3,
    # projectile
    "sphere_radius": 0.25,
    "sphere_mass": 10.0,
    "sphere_start": (-3.0, 0.0, 0.3),
    "sphere_velocity": (7.0, 0.0, 0.5),
    "sphere_spin": (0.0, -6.0, 0.0),
    # camera (fixed side view)
    "camera_pos": (1.2, -3.6, 1.1),
    "camera_pitch": -9.0,
    "camera_yaw": 105.0,
    "camera_fov": 45.0,
}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.frame_dt = 1.0 / self.params["fps"]
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        p = self.params
        builder = newton.ModelBuilder()  # Z up, gravity -Z
        builder.rigid_gap = p["rigid_gap"]
        builder.default_shape_cfg.margin = p["shape_margin"]
        builder.default_shape_cfg.ke = p["shape_ke"]
        builder.add_ground_plane()

        self._boxes = []
        self._build_pyramid(builder)

        r = p["sphere_radius"]
        inertia_val = 0.4 * p["sphere_mass"] * r * r
        inertia = wp.mat33(inertia_val, 0.0, 0.0, 0.0, inertia_val, 0.0, 0.0, 0.0, inertia_val)
        self._sphere = builder.add_body(
            xform=wp.transform(wp.vec3(*p["sphere_start"]), wp.quat_identity()),
            mass=p["sphere_mass"],
            inertia=inertia,
            lock_inertia=True,
        )
        builder.add_shape_sphere(body=self._sphere, radius=r, color=(0.85, 0.3, 0.25))

        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=p["solver_iterations"],
            rigid_enable_penetration_free=p["enable_dat"],
            rigid_penetration_free_query_margin=p["rigid_gap"],
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="nxn")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        qd = self.state_0.body_qd.numpy()
        qd[self._sphere][:3] = p["sphere_velocity"]
        qd[self._sphere][3:] = p["sphere_spin"]
        self.state_0.body_qd.assign(qd)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)

        self.max_ground_penetration = 0.0
        self.max_sphere_penetration = 0.0
        self.max_sphere_x = float(self.state_0.body_q.numpy()[self._sphere][0])
        self._top_box_start = self.state_0.body_q.numpy()[self._boxes[-1]][:3].copy()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(*p["camera_pos"]),
                p["camera_pitch"],
                p["camera_yaw"],
            )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

    # ── model construction ──────────────────────────────────────────────

    def _build_pyramid(self, builder):
        p = self.params
        h = p["box_half"]
        spacing = 2.0 * h + 0.01
        inertia_val = p["box_mass"] * h * h * 2.0 / 3.0
        inertia = wp.mat33(inertia_val, 0.0, 0.0, 0.0, inertia_val, 0.0, 0.0, 0.0, inertia_val)
        palette = [(0.25, 0.5, 0.85), (0.95, 0.75, 0.2), (0.4, 0.8, 0.4)]
        rows = p["pyramid_rows"]
        for level in range(rows):
            count = rows - level
            z = h + p["shape_margin"] + level * (spacing + 0.002)
            x0 = -0.5 * (count - 1) * spacing
            for i in range(count):
                body = builder.add_body(
                    xform=wp.transform(wp.vec3(x0 + i * spacing, 0.0, z), wp.quat_identity()),
                    mass=p["box_mass"],
                    inertia=inertia,
                    lock_inertia=True,
                )
                builder.add_shape_box(body=body, hx=h, hy=h, hz=h, color=palette[level % len(palette)])
                self._boxes.append(body)

    # ── simulation loop ─────────────────────────────────────────────────

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ── validation ──────────────────────────────────────────────────────

    def _penetrations(self):
        """(deepest body-below-ground, deepest box-corner-inside-sphere) in meters."""
        p = self.params
        h = p["box_half"]
        r = p["sphere_radius"]
        body_q = self.state_0.body_q.numpy()

        corners = np.array(
            [[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64
        )
        sphere_c = body_q[self._sphere][:3]

        ground_pen = max(0.0, -(float(body_q[self._sphere][2]) - r))
        sphere_pen = 0.0
        for box in self._boxes:
            bq = body_q[box]
            pos, (qx, qy, qz, qw) = bq[:3], bq[3:]
            rot = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            corners_world = corners @ rot.T + pos[None, :]
            ground_pen = max(ground_pen, -float(corners_world[:, 2].min()))
            sphere_pen = max(sphere_pen, -float((np.linalg.norm(corners_world - sphere_c[None, :], axis=1) - r).min()))
        return ground_pen, sphere_pen

    def test_post_step(self):
        ground_pen, sphere_pen = self._penetrations()
        self.max_ground_penetration = max(self.max_ground_penetration, ground_pen)
        self.max_sphere_penetration = max(self.max_sphere_penetration, sphere_pen)
        self.max_sphere_x = max(self.max_sphere_x, float(self.state_0.body_q.numpy()[self._sphere][0]))

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("simulation produced non-finite state")
        if self.max_ground_penetration > 1.0e-3:
            raise AssertionError(f"a body sank {self.max_ground_penetration:.6f} m into the ground")
        if self.max_sphere_penetration > 1.0e-3:
            raise AssertionError(f"a box corner entered the sphere by {self.max_sphere_penetration:.6f} m")
        # Guard against the DAT motion cap silently draining the projectile's momentum:
        # the sphere must actually reach the pyramid and scatter it.
        if self.max_sphere_x < -1.0:
            raise AssertionError(f"projectile stalled at x={self.max_sphere_x:.3f}; it never reached the pyramid")
        top_box_disp = float(np.linalg.norm(body_q[self._boxes[-1]][:3] - self._top_box_start))
        if top_box_disp < 0.2:
            raise AssertionError("the pyramid was not scattered by the impact")

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
