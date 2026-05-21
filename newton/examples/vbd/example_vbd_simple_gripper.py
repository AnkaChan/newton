# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Simple Gripper
#
# A minimal parallel-jaw gripper on a Cartesian gantry follows the same
# grab -> close -> lift -> wave trajectory used by the Franka pickup
# examples.  The mechanism is four prismatic joints:
#
#   World -[Z]-> gantry_z -[X]-> gantry_x -[Y]-> left_finger
#                                          -[Y]-> right_finger
#
# All joints are PD-position-driven.  VBD integrates everything
# (integrate_with_external_rigid_solver=False).  A small cube sits on
# a pedestal between the pads; the gripper closes, grips it by
# friction, and carries it through the lift-and-wave trajectory.
#
# Command: python -m newton.examples vbd_simple_gripper
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

PARAMS = {
    # ── simulation ──────────────────────────────────────────────────────
    "fps": 60,
    "sim_substeps": 20,
    "solver_iterations": 5,
    "gravity": -9.8,
    "num_frames": 420,
    # ── ground ──────────────────────────────────────────────────────────
    "ground_size": 0.80,
    "ground_thickness": 0.01,
    "ground_ke": 1.0e5,
    "ground_kd": 1.0e2,
    "ground_mu": 0.9,
    "ground_color": (0.45, 0.45, 0.48),
    # ── finger pad geometry (matches Franka-pickup finger pads) ────────
    "finger_pad_hx": 0.11,
    "finger_pad_hy": 0.0075,
    "finger_pad_hz": 0.026,
    "finger_density": 1000.0,
    "finger_z_offset": 0.09,
    "finger_color_left": (0.8, 0.3, 0.3),
    "finger_color_right": (0.3, 0.3, 0.8),
    # ── gripper gap (half-gap per finger) ──────────────────────────────
    "open_half_gap": 0.04,
    "closed_half_gap": 0.001,
    # ── trajectory (same as kinematic Franka pickup) ───────────────────
    "bag_floor_height": 0.004,
    "bag_size_z": 0.32,
    "grab_clearance": 0.09,
    "lift_height": 0.62,
    "wave_axis": 0,
    "wave_offset": 0.08,
    "close_duration": 0.6,
    "pinch_duration": 0.25,
    "lift_duration": 2.8,
    "wave_start_duration": 0.5,
    "wave_sweep_duration": 0.7,
    "wave_return_duration": 0.5,
    "hold_duration": 1.0,
    # ── PD drive gains ─────────────────────────────────────────────────
    "gantry_drive_ke": 5.0e4,
    "gantry_drive_kd": 5.0e3,
    "finger_drive_ke": 1.0e5,
    "finger_drive_kd": 1.0e2,
    # ── gantry carrier link mass ───────────────────────────────────────
    "gantry_link_mass": 0.01,
    # ── camera ─────────────────────────────────────────────────────────
    "camera_pos": (0.98, -1.38, 0.80),
    "camera_fov": 45.0,
    "camera_pitch": -16.9,
    "camera_yaw": 128.5,
    # ── cube (test object to grip) ────────────────────────────────────
    "cube_half": 0.015,
    "cube_density": 1000.0,
    "cube_ke": 5.0e5,
    "cube_kd": 5.0e1,
    "cube_mu": 0.5,
    "cube_color": (0.2, 0.8, 0.2),
    # ── pedestal (supports cube before grip) ──────────────────────────
    "pedestal_hx": 0.03,
    "pedestal_hy": 0.01,
    "pedestal_ke": 1.0e5,
    "pedestal_kd": 1.0e2,
    "pedestal_mu": 0.9,
    "pedestal_color": (0.6, 0.6, 0.5),
    # ── finger contact material ───────────────────────────────────────
    "finger_mu": 1.0,
    "finger_ke": 1.0e6,
    "finger_kd": 1.0e1,
    # ── collision ─────────────────────────────────────────────────────
    "rigid_body_contact_buffer_size": 512,
    "collision_broad_phase": "nxn",
    "soft_contact_margin": 0.01,
    # ── misc ───────────────────────────────────────────────────────────
    "draw_wireframe": False,
    "initial_paused": False,
}


def _vec_with_axis_offset(pos: wp.vec3, axis: int, offset: float) -> wp.vec3:
    v = [pos[0], pos[1], pos[2]]
    v[axis] += offset
    return wp.vec3(*v)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0
        self._current_waypoint = 0
        self._time_in_waypoint = 0.0
        self._gripper_frac = 0.0

        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self._build_gripper(builder)
        self._add_ground(builder)
        self._add_cube(builder)

        builder.color()
        self.model = builder.finalize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            integrate_with_external_rigid_solver=False,
            rigid_body_contact_buffer_size=self.params["rigid_body_contact_buffer_size"],
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.params["collision_broad_phase"],
            soft_contact_margin=self.params["soft_contact_margin"],
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)

        self._build_waypoints()
        self._set_targets(self._waypoints[0][0], 0.0)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = self.params["draw_wireframe"]
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = self.params["initial_paused"]
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(*self.params["camera_pos"]),
                self.params["camera_pitch"],
                self.params["camera_yaw"],
            )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

    # ── model construction ──────────────────────────────────────────────

    def _build_gripper(self, builder):
        p = self.params
        grab_z = p["bag_floor_height"] + p["bag_size_z"] + p["grab_clearance"]

        gantry_z = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, grab_z), wp.quat_identity()),
            mass=p["gantry_link_mass"],
            label="gantry_z",
        )
        gantry_x = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, grab_z), wp.quat_identity()),
            mass=p["gantry_link_mass"],
            label="gantry_x",
        )
        left_finger = builder.add_link(
            xform=wp.transform(
                wp.vec3(0.0, -p["open_half_gap"], grab_z - p["finger_z_offset"]),
                wp.quat_identity(),
            ),
            label="left_finger",
        )
        right_finger = builder.add_link(
            xform=wp.transform(
                wp.vec3(0.0, p["open_half_gap"], grab_z - p["finger_z_offset"]),
                wp.quat_identity(),
            ),
            label="right_finger",
        )

        finger_cfg = newton.ModelBuilder.ShapeConfig(
            density=p["finger_density"],
            ke=p["finger_ke"],
            kd=p["finger_kd"],
            mu=p["finger_mu"],
        )
        builder.add_shape_box(
            left_finger,
            hx=p["finger_pad_hx"],
            hy=p["finger_pad_hy"],
            hz=p["finger_pad_hz"],
            cfg=finger_cfg,
            color=p["finger_color_left"],
            label="left_pad",
        )
        builder.add_shape_box(
            right_finger,
            hx=p["finger_pad_hx"],
            hy=p["finger_pad_hy"],
            hz=p["finger_pad_hz"],
            cfg=finger_cfg,
            color=p["finger_color_right"],
            label="right_pad",
        )

        j_z = builder.add_joint_prismatic(
            parent=-1,
            child=gantry_z,
            axis=wp.vec3(0.0, 0.0, 1.0),
            target_ke=p["gantry_drive_ke"],
            target_kd=p["gantry_drive_kd"],
            target_pos=grab_z,
            label="gantry_z_joint",
        )
        j_x = builder.add_joint_prismatic(
            parent=gantry_z,
            child=gantry_x,
            axis=wp.vec3(1.0, 0.0, 0.0),
            target_ke=p["gantry_drive_ke"],
            target_kd=p["gantry_drive_kd"],
            target_pos=0.0,
            label="gantry_x_joint",
        )
        j_left = builder.add_joint_prismatic(
            parent=gantry_x,
            child=left_finger,
            axis=wp.vec3(0.0, -1.0, 0.0),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, p["finger_z_offset"]), wp.quat_identity()),
            target_ke=p["finger_drive_ke"],
            target_kd=p["finger_drive_kd"],
            target_pos=p["open_half_gap"],
            label="left_finger_joint",
        )
        j_right = builder.add_joint_prismatic(
            parent=gantry_x,
            child=right_finger,
            axis=wp.vec3(0.0, 1.0, 0.0),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, p["finger_z_offset"]), wp.quat_identity()),
            target_ke=p["finger_drive_ke"],
            target_kd=p["finger_drive_kd"],
            target_pos=p["open_half_gap"],
            label="right_finger_joint",
        )

        builder.add_articulation([j_z, j_x, j_left, j_right], label="gripper")

        builder.joint_q[0] = grab_z
        builder.joint_q[1] = 0.0
        builder.joint_q[2] = p["open_half_gap"]
        builder.joint_q[3] = p["open_half_gap"]

        self._dof_z = 0
        self._dof_x = 1
        self._dof_left = 2
        self._dof_right = 3

    def _add_ground(self, builder):
        p = self.params
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.ke = p["ground_ke"]
        ground_cfg.kd = p["ground_kd"]
        ground_cfg.mu = p["ground_mu"]
        builder.add_shape_box(
            -1,
            wp.transform(wp.vec3(0.0, 0.0, -p["ground_thickness"] / 2.0), wp.quat_identity()),
            hx=p["ground_size"] / 2.0,
            hy=p["ground_size"] / 2.0,
            hz=p["ground_thickness"] / 2.0,
            cfg=ground_cfg,
            color=p["ground_color"],
            label="ground",
        )

    def _add_cube(self, builder):
        p = self.params
        grab_z = p["bag_floor_height"] + p["bag_size_z"] + p["grab_clearance"]
        pad_center_z = grab_z - p["finger_z_offset"]
        cube_h = p["cube_half"]

        pedestal_top_z = pad_center_z - cube_h
        pedestal_hz = pedestal_top_z / 2.0
        ped_cfg = newton.ModelBuilder.ShapeConfig(
            ke=p["pedestal_ke"],
            kd=p["pedestal_kd"],
            mu=p["pedestal_mu"],
        )
        builder.add_shape_box(
            -1,
            wp.transform(wp.vec3(0.0, 0.0, pedestal_hz), wp.quat_identity()),
            hx=p["pedestal_hx"],
            hy=p["pedestal_hy"],
            hz=pedestal_hz,
            cfg=ped_cfg,
            color=p["pedestal_color"],
            label="pedestal",
        )

        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=p["cube_density"],
            ke=p["cube_ke"],
            kd=p["cube_kd"],
            mu=p["cube_mu"],
        )
        cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, pad_center_z), wp.quat_identity()),
            label="cube",
        )
        builder.add_shape_box(
            cube_body,
            hx=cube_h,
            hy=cube_h,
            hz=cube_h,
            cfg=cube_cfg,
            color=p["cube_color"],
            label="cube",
        )

    # ── trajectory ──────────────────────────────────────────────────────

    def _build_waypoints(self):
        p = self.params
        bag_top = p["bag_floor_height"] + p["bag_size_z"]
        grab_pos = wp.vec3(0.0, 0.0, bag_top + p["grab_clearance"])
        lift_pos = wp.vec3(0.0, 0.0, p["lift_height"])
        wave_left = _vec_with_axis_offset(lift_pos, p["wave_axis"], -p["wave_offset"])
        wave_right = _vec_with_axis_offset(lift_pos, p["wave_axis"], p["wave_offset"])
        self._waypoints = [
            (grab_pos, p["close_duration"], 0.0),
            (grab_pos, p["pinch_duration"], 1.0),
            (grab_pos, p["lift_duration"], 1.0),
            (lift_pos, p["wave_start_duration"], 1.0),
            (wave_left, p["wave_sweep_duration"], 1.0),
            (wave_right, p["wave_sweep_duration"], 1.0),
            (wave_left, p["wave_return_duration"], 1.0),
            (lift_pos, p["hold_duration"], 1.0),
        ]

    def _advance_waypoint(self):
        self._time_in_waypoint += self.frame_dt
        cur = self._waypoints[self._current_waypoint]
        nxt = self._waypoints[min(self._current_waypoint + 1, len(self._waypoints) - 1)]
        t = min(self._time_in_waypoint / cur[1], 1.0)

        target_pos = cur[0] * (1.0 - t) + nxt[0] * t
        self._gripper_frac = float(cur[2]) * (1.0 - t) + float(nxt[2]) * t

        if self._time_in_waypoint >= cur[1] and self._current_waypoint < len(self._waypoints) - 1:
            self._current_waypoint += 1
            self._time_in_waypoint = 0.0

        self._set_targets(target_pos, self._gripper_frac)

    def _set_targets(self, pos: wp.vec3, gripper_frac: float):
        p = self.params
        half_gap = p["open_half_gap"] * (1.0 - gripper_frac) + p["closed_half_gap"] * gripper_frac
        targets = np.zeros(self.model.joint_dof_count)
        targets[self._dof_z] = float(pos[2])
        targets[self._dof_x] = float(pos[0])
        targets[self._dof_left] = half_gap
        targets[self._dof_right] = half_gap
        self.control.joint_target_pos.assign(wp.array(targets, dtype=float, device=self.model.device))

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
        self._advance_waypoint()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

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
