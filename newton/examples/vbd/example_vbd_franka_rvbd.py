# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Franka RVBD
#
# A self-contained robot-only Franka FR3 scene for comparing maximal VBD
# rigid-body integration with RVBD reduced-coordinate projection.
#
# The arm is controlled by PD joint drives whose targets come from an IK
# solve each frame.  The same scene can be run with maximal rigid bodies or
# with reduced-coordinate projection enabled:
#
#   python -m newton.examples vbd_franka_rvbd
#   python -m newton.examples vbd_franka_rvbd --reduced-solve
#
###########################################################################

from __future__ import annotations

import copy

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

PARAMS = {
    "fps": 60,
    "num_frames": 240,
    "sim_substeps": 10,
    "solver_iterations": 10,
    "gravity": -9.8,
    "initial_paused": False,
    "initial_sim_time": 0.0,
    "initial_frame": 0,
    "initial_waypoint": 0,
    "initial_waypoint_time": 0.0,
    "initial_gripper_frac": 0.0,
    "ground_size": 0.80,
    "ground_thickness": 0.01,
    "ground_ke": 1.0e5,
    "ground_kd": 1.0e2,
    "ground_mu": 0.9,
    "ground_color": (0.45, 0.45, 0.48),
    "ground_label": "ground",
    "ground_center_xy": (0.0, 0.0),
    "collision_broad_phase": "nxn",
    "integrate_with_external_rigid_solver": False,
    "rigid_body_particle_contact_buffer_size": 2048,
    "rigid_body_contact_buffer_size": 512,
    "particle_enable_self_contact": False,
    "particle_self_contact_radius": 0.005,
    "particle_self_contact_margin": 0.01,
    "particle_topological_contact_filter_threshold": 3,
    "rigid_contact_hard": True,
    "rigid_joint_linear_ke": 1.0e6,
    "rigid_joint_angular_ke": 1.0e6,
    "rigid_joint_linear_kd": 1.0e3,
    "rigid_joint_angular_kd": 1.0e2,
    "body_enable_reduced_solve": False,
    "reduced_gn_iterations": 2,
    "reduced_gn_damping": 1.0e-6,
    "franka_asset_name": "franka_emika_panda",
    "franka_urdf_path": "urdf/fr3_franka_hand.urdf",
    "franka_base_pos": (-0.50, 0.0, 0.05),
    "franka_floating": False,
    "franka_scale": 1.0,
    "franka_enable_self_collisions": False,
    "franka_parse_visuals_as_colliders": True,
    "franka_zero_mass_body_mass": 0.05,
    "franka_zero_mass_body_inertia": 1.0e-4,
    "franka_init_q": (-3.6802e-03, 2.3902e-02, 3.6804e-03, -2.3683, -1.2919e-04, 2.3922, 7.8549e-01),
    "arm_joint_count": 7,
    "left_finger_body_suffix": "fr3_leftfinger",
    "right_finger_body_suffix": "fr3_rightfinger",
    "hand_body_suffix": "fr3_hand",
    "mesh_approximation_method": "convex_hull",
    "keep_visual_shapes": True,
    "arm_drive_ke": (1.0e6, 1.0e6, 8.0e5, 8.0e5, 6.0e5, 6.0e5, 6.0e5),
    "arm_drive_kd": (1.0e5, 1.0e5, 8.0e4, 8.0e4, 6.0e4, 6.0e4, 6.0e4),
    "gripper_drive_ke": 1.0e6,
    "gripper_drive_kd": 1.0e5,
    "gripper_open_gap": 0.08,
    "gripper_closed_gap": 0.001,
    "gripper_open_frac": 0.0,
    "gripper_closed_frac": 1.0,
    "gripper_joint_indices": (7, 8),
    "gripper_gap_test_tolerance": 1.0e-8,
    "finger_pad_density": 1000.0,
    "finger_shape_mu": 1.0,
    "finger_shape_ke": 1.0e6,
    "finger_shape_kd": 1.0e1,
    "finger_pad_half_width": 0.11,
    "finger_pad_half_thickness": 0.0075,
    "finger_pad_half_height": 0.026,
    "finger_pad_local_pos": (0.0, 0.00758, 0.0575),
    "finger_pad_left_label": "left_finger_pad",
    "finger_pad_right_label": "right_finger_pad",
    "ee_link_offset": (0.0, 0.0, 0.0),
    "grasp_xy": (0.0, 0.0),
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
    "waypoint_interp_max": 1.0,
    "tool_rotation_axis": (1.0, 0.0, 0.0),
    "tool_rotation_angle": np.pi,
    "ik_n_problems": 1,
    "ik_optimizer": "lbfgs",
    "ik_jacobian_mode": ik.IKJacobianType.ANALYTIC,
    "ik_lambda_initial": 0.1,
    "ik_iterations": 24,
    "pregrasp_ik_iterations": 48,
    "enable_ik_cuda_graph": True,
    "vertical_axis": 2,
    "hand_lift_test_min_z": 0.53,
    "draw_wireframe": False,
    "camera_pos": (0.92, -1.28, 0.74),
    "camera_fov": 44.0,
    "camera_pitch": -16.7,
    "camera_yaw": 129.6,
}


def _quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


def _vec_with_axis_offset(pos: wp.vec3, axis: int, offset: float) -> wp.vec3:
    values = [pos[0], pos[1], pos[2]]
    values[axis] += offset
    return wp.vec3(*values)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = copy.deepcopy(PARAMS)
        self.params["body_enable_reduced_solve"] = bool(getattr(args, "reduced_solve", False))
        self.params["reduced_gn_iterations"] = int(
            getattr(args, "reduced_gn_iterations", PARAMS["reduced_gn_iterations"])
        )
        self.params["reduced_gn_damping"] = float(getattr(args, "reduced_gn_damping", PARAMS["reduced_gn_damping"]))

        arm_drive_scale = float(getattr(args, "arm_drive_scale", 1.0))
        gripper_drive_scale_arg = getattr(args, "gripper_drive_scale", None)
        gripper_drive_scale = arm_drive_scale if gripper_drive_scale_arg is None else float(gripper_drive_scale_arg)
        self.params["arm_drive_ke"] = tuple(float(v) * arm_drive_scale for v in self.params["arm_drive_ke"])
        self.params["arm_drive_kd"] = tuple(float(v) * arm_drive_scale for v in self.params["arm_drive_kd"])
        self.params["gripper_drive_ke"] *= gripper_drive_scale
        self.params["gripper_drive_kd"] *= gripper_drive_scale

        self.sim_time = self.params["initial_sim_time"]
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = self.params["initial_frame"]
        self._current_waypoint = self.params["initial_waypoint"]
        self._time_in_waypoint = self.params["initial_waypoint_time"]
        self._gripper_frac = self.params["initial_gripper_frac"]

        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self._add_robot(builder)
        self._add_ground(builder)
        builder.color(include_bending=True)

        self.model = builder.finalize()
        self._regularize_robot_zero_mass_bodies()
        self._configure_finger_materials()
        self._configure_joint_drives()

        self.contacts = None
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.model.body_q, self.state_0.body_q)

        self._state_single = self._model_single.state()
        newton.eval_fk(self._model_single, self._model_single.joint_q, self._model_single.joint_qd, self._state_single)
        self._setup_ik()
        self._initialize_robot_pregrasp()

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.params["solver_iterations"],
            integrate_with_external_rigid_solver=self.params["integrate_with_external_rigid_solver"],
            rigid_body_particle_contact_buffer_size=self.params["rigid_body_particle_contact_buffer_size"],
            rigid_body_contact_buffer_size=self.params["rigid_body_contact_buffer_size"],
            particle_enable_self_contact=self.params["particle_enable_self_contact"],
            particle_self_contact_radius=self.params["particle_self_contact_radius"],
            particle_self_contact_margin=self.params["particle_self_contact_margin"],
            particle_topological_contact_filter_threshold=self.params["particle_topological_contact_filter_threshold"],
            rigid_contact_hard=self.params["rigid_contact_hard"],
            rigid_joint_linear_ke=self.params["rigid_joint_linear_ke"],
            rigid_joint_angular_ke=self.params["rigid_joint_angular_ke"],
            rigid_joint_linear_kd=self.params["rigid_joint_linear_kd"],
            rigid_joint_angular_kd=self.params["rigid_joint_angular_kd"],
            body_enable_reduced_solve=self.params["body_enable_reduced_solve"],
            reduced_gn_iterations=self.params["reduced_gn_iterations"],
            reduced_gn_damping=self.params["reduced_gn_damping"],
        )

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

    def _add_ground(self, builder: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(
            ke=self.params["ground_ke"],
            kd=self.params["ground_kd"],
            mu=self.params["ground_mu"],
        )
        builder.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(*self.params["ground_center_xy"], -self.params["ground_thickness"] / 2.0),
                wp.quat_identity(),
            ),
            hx=self.params["ground_size"] / 2.0,
            hy=self.params["ground_size"] / 2.0,
            hz=self.params["ground_thickness"] / 2.0,
            cfg=cfg,
            color=self.params["ground_color"],
            label=self.params["ground_label"],
        )

    def _add_robot(self, builder: newton.ModelBuilder) -> None:
        asset_path = newton.utils.download_asset(self.params["franka_asset_name"])
        builder.add_urdf(
            str(asset_path / self.params["franka_urdf_path"]),
            xform=wp.transform(wp.vec3(*self.params["franka_base_pos"]), wp.quat_identity()),
            floating=self.params["franka_floating"],
            scale=self.params["franka_scale"],
            enable_self_collisions=self.params["franka_enable_self_collisions"],
            parse_visuals_as_colliders=self.params["franka_parse_visuals_as_colliders"],
        )

        arm_joint_count = self.params["arm_joint_count"]
        builder.joint_q[:arm_joint_count] = self.params["franka_init_q"]
        open_gripper_value = self._gripper_joint_value(self.params["gripper_open_frac"])
        for joint_index in self.params["gripper_joint_indices"]:
            builder.joint_q[joint_index] = open_gripper_value

        self._left_finger_body = next(
            i for i, label in enumerate(builder.body_label) if label.endswith(self.params["left_finger_body_suffix"])
        )
        self._right_finger_body = next(
            i for i, label in enumerate(builder.body_label) if label.endswith(self.params["right_finger_body_suffix"])
        )
        self._hand_body = next(
            i for i, label in enumerate(builder.body_label) if label.endswith(self.params["hand_body_suffix"])
        )
        self._finger_contact_body_indices = {self._left_finger_body, self._right_finger_body}

        pad_cfg = newton.ModelBuilder.ShapeConfig(
            density=self.params["finger_pad_density"],
            mu=self.params["finger_shape_mu"],
            ke=self.params["finger_shape_ke"],
            kd=self.params["finger_shape_kd"],
        )
        pad_xform = wp.transform(wp.vec3(*self.params["finger_pad_local_pos"]), wp.quat_identity())
        builder.add_shape_box(
            body=self._left_finger_body,
            xform=pad_xform,
            hx=self.params["finger_pad_half_width"],
            hy=self.params["finger_pad_half_thickness"],
            hz=self.params["finger_pad_half_height"],
            cfg=pad_cfg,
            label=self.params["finger_pad_left_label"],
        )
        builder.add_shape_box(
            body=self._right_finger_body,
            xform=pad_xform,
            hx=self.params["finger_pad_half_width"],
            hy=self.params["finger_pad_half_thickness"],
            hz=self.params["finger_pad_half_height"],
            cfg=pad_cfg,
            label=self.params["finger_pad_right_label"],
        )

        finger_body_set = {self._left_finger_body, self._right_finger_body, self._hand_body}
        non_finger_shape_indices = [s for s, body in enumerate(builder.shape_body) if body not in finger_body_set]
        builder.approximate_meshes(
            method=self.params["mesh_approximation_method"],
            shape_indices=non_finger_shape_indices,
            keep_visual_shapes=self.params["keep_visual_shapes"],
        )

        self._model_single = copy.deepcopy(builder).finalize()
        self._robot_body_count = builder.body_count
        self._ee_body_index = self._hand_body

    def _regularize_robot_zero_mass_bodies(self) -> None:
        body_mass = self.model.body_mass.numpy().copy()
        zero_mass_indices = [
            i
            for i, label in enumerate(self.model.body_label[: self._robot_body_count])
            if body_mass[i] == 0.0 and not label.endswith("/base")
        ]
        if not zero_mass_indices:
            return

        mass = self.params["franka_zero_mass_body_mass"]
        inertia = self.params["franka_zero_mass_body_inertia"]
        body_inv_mass = self.model.body_inv_mass.numpy().copy()
        body_inertia = self.model.body_inertia.numpy().copy()
        body_inv_inertia = self.model.body_inv_inertia.numpy().copy()
        inertia_matrix = np.eye(3, dtype=np.float32) * inertia
        inv_inertia_matrix = np.eye(3, dtype=np.float32) / inertia

        for body_index in zero_mass_indices:
            body_mass[body_index] = mass
            body_inv_mass[body_index] = 1.0 / mass
            body_inertia[body_index] = inertia_matrix
            body_inv_inertia[body_index] = inv_inertia_matrix

        self.model.body_mass = wp.array(body_mass, dtype=float, device=self.model.device)
        self.model.body_inv_mass = wp.array(body_inv_mass, dtype=float, device=self.model.device)
        self.model.body_inertia = wp.array(body_inertia, dtype=wp.mat33, device=self.model.device)
        self.model.body_inv_inertia = wp.array(body_inv_inertia, dtype=wp.mat33, device=self.model.device)

    def _configure_finger_materials(self) -> None:
        shape_body = self.model.shape_body.numpy()
        shape_mu = self.model.shape_material_mu.numpy().copy()
        shape_ke = self.model.shape_material_ke.numpy().copy()
        shape_kd = self.model.shape_material_kd.numpy().copy()

        for shape_index, shape_body_index in enumerate(shape_body):
            body_index = int(shape_body_index)
            if body_index in self._finger_contact_body_indices:
                shape_mu[shape_index] = self.params["finger_shape_mu"]
                shape_ke[shape_index] = self.params["finger_shape_ke"]
                shape_kd[shape_index] = self.params["finger_shape_kd"]

        self.model.shape_material_mu = wp.array(shape_mu, dtype=float, device=self.model.device)
        self.model.shape_material_ke = wp.array(shape_ke, dtype=float, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=float, device=self.model.device)

    def _configure_joint_drives(self) -> None:
        ke = self.model.joint_target_ke.numpy().copy()
        kd = self.model.joint_target_kd.numpy().copy()
        arm_n = self.params["arm_joint_count"]
        ke[:arm_n] = self.params["arm_drive_ke"][:arm_n]
        kd[:arm_n] = self.params["arm_drive_kd"][:arm_n]
        for ji in self.params["gripper_joint_indices"]:
            ke[ji] = self.params["gripper_drive_ke"]
            kd[ji] = self.params["gripper_drive_kd"]
        self.model.joint_target_ke = wp.array(ke, dtype=float, device=self.model.device)
        self.model.joint_target_kd = wp.array(kd, dtype=float, device=self.model.device)

    def _tool_rotation(self) -> wp.quat:
        return wp.quat_from_axis_angle(
            wp.vec3(*self.params["tool_rotation_axis"]),
            self.params["tool_rotation_angle"],
        )

    def _setup_ik(self) -> None:
        ee_tf = wp.transform(*self._state_single.body_q.numpy()[self._ee_body_index])

        self._pos_obj = ik.IKObjectivePosition(
            link_index=self._ee_body_index,
            link_offset=wp.vec3(*self.params["ee_link_offset"]),
            target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=self._ee_body_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
        )
        self._joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self._model_single.joint_limit_lower,
            joint_limit_upper=self._model_single.joint_limit_upper,
        )
        joint_q_seed = self._model_single.joint_q.numpy().reshape(
            self.params["ik_n_problems"], self._model_single.joint_coord_count
        )
        self._joint_q_ik = wp.array(joint_q_seed.copy(), dtype=float, device=self._model_single.device)
        self._ik_solver = ik.IKSolver(
            model=self._model_single,
            n_problems=self.params["ik_n_problems"],
            objectives=[self._pos_obj, self._rot_obj, self._joint_limits_obj],
            optimizer=self.params["ik_optimizer"],
            lambda_initial=self.params["ik_lambda_initial"],
            jacobian_mode=self.params["ik_jacobian_mode"],
        )

        bag_top = self.params["bag_floor_height"] + self.params["bag_size_z"]
        grasp_x, grasp_y = self.params["grasp_xy"]
        grab_pos = wp.vec3(grasp_x, grasp_y, bag_top + self.params["grab_clearance"])
        lift_pos = wp.vec3(grasp_x, grasp_y, self.params["lift_height"])
        wave_axis = self.params["wave_axis"]
        wave_left_pos = _vec_with_axis_offset(lift_pos, wave_axis, -self.params["wave_offset"])
        wave_right_pos = _vec_with_axis_offset(lift_pos, wave_axis, self.params["wave_offset"])
        self._waypoints = [
            (grab_pos, self.params["close_duration"], self.params["gripper_open_frac"]),
            (grab_pos, self.params["pinch_duration"], self.params["gripper_closed_frac"]),
            (grab_pos, self.params["lift_duration"], self.params["gripper_closed_frac"]),
            (lift_pos, self.params["wave_start_duration"], self.params["gripper_closed_frac"]),
            (wave_left_pos, self.params["wave_sweep_duration"], self.params["gripper_closed_frac"]),
            (wave_right_pos, self.params["wave_sweep_duration"], self.params["gripper_closed_frac"]),
            (wave_left_pos, self.params["wave_return_duration"], self.params["gripper_closed_frac"]),
            (lift_pos, self.params["hold_duration"], self.params["gripper_closed_frac"]),
        ]

        if self.params["enable_ik_cuda_graph"] and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._ik_solver.step(
                    self._joint_q_ik,
                    self._joint_q_ik,
                    iterations=self.params["ik_iterations"],
                )
            self._graph_ik = capture.graph
        else:
            self._graph_ik = None

    def _gripper_joint_value(self, gripper_frac: float) -> float:
        open_value = 0.5 * self.params["gripper_open_gap"]
        closed_value = 0.5 * self.params["gripper_closed_gap"]
        open_frac = self.params["gripper_open_frac"]
        closed_frac = self.params["gripper_closed_frac"]
        frac_range = closed_frac - open_frac
        if frac_range == 0.0:
            return closed_value

        alpha = (gripper_frac - open_frac) / frac_range
        return open_value * (1.0 - alpha) + closed_value * alpha

    def _initialize_robot_pregrasp(self) -> None:
        start_pos = self._waypoints[0][0]
        start_frac = float(self._waypoints[0][2])
        start_rot = self._tool_rotation()

        self._pos_obj.set_target_positions(wp.array([start_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_quat_to_vec4(start_rot)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=self.params["pregrasp_ik_iterations"])

        joint_q = self.state_0.joint_q.numpy().copy()
        ik_solution = self._joint_q_ik.numpy()[0]
        arm_joint_count = self.params["arm_joint_count"]
        joint_q[:arm_joint_count] = ik_solution[:arm_joint_count]
        gripper_value = self._gripper_joint_value(start_frac)
        for joint_index in self.params["gripper_joint_indices"]:
            joint_q[joint_index] = gripper_value
        joint_q_wp = wp.array(joint_q, dtype=float, device=self.model.device)

        self._gripper_frac = start_frac
        self.model.joint_q.assign(joint_q_wp)
        self.state_0.joint_q.assign(joint_q_wp)
        self.state_1.joint_q.assign(joint_q_wp)
        self.model.joint_qd.zero_()
        self.state_0.joint_qd.zero_()
        self.state_1.joint_qd.zero_()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.model.body_q, self.state_0.body_q)

        single_coord_count = self._model_single.joint_coord_count
        joint_q_single = joint_q[:single_coord_count]
        joint_q_single_wp = wp.array(joint_q_single, dtype=float, device=self._model_single.device)
        self._model_single.joint_q.assign(joint_q_single_wp)
        self._model_single.joint_qd.zero_()
        self._state_single.joint_q.assign(joint_q_single_wp)
        self._state_single.joint_qd.zero_()
        newton.eval_fk(self._model_single, self._state_single.joint_q, self._state_single.joint_qd, self._state_single)

        self._update_drive_targets()

    def _set_joint_targets(self) -> None:
        self._time_in_waypoint += self.frame_dt
        current = self._waypoints[self._current_waypoint]
        next_waypoint = self._waypoints[min(self._current_waypoint + 1, len(self._waypoints) - 1)]
        t = min(self._time_in_waypoint / current[1], self.params["waypoint_interp_max"])

        target_pos = current[0] * (1.0 - t) + next_waypoint[0] * t
        self._gripper_frac = float(current[2]) * (1.0 - t) + float(next_waypoint[2]) * t

        target_rot = self._tool_rotation()
        self._pos_obj.set_target_positions(wp.array([target_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_quat_to_vec4(target_rot)], dtype=wp.vec4))

        if self._graph_ik is not None:
            wp.capture_launch(self._graph_ik)
        else:
            self._ik_solver.step(
                self._joint_q_ik,
                self._joint_q_ik,
                iterations=self.params["ik_iterations"],
            )

        if self._time_in_waypoint >= current[1] and self._current_waypoint < len(self._waypoints) - 1:
            self._current_waypoint += 1
            self._time_in_waypoint = 0.0

    def _update_drive_targets(self) -> None:
        ik_solution = self._joint_q_ik.numpy()[0]
        arm_n = self.params["arm_joint_count"]
        target_pos = np.zeros(self.model.joint_dof_count)
        target_pos[:arm_n] = ik_solution[:arm_n]
        gripper_value = self._gripper_joint_value(self._gripper_frac)
        for ji in self.params["gripper_joint_indices"]:
            target_pos[ji] = gripper_value
        self.control.joint_target_pos.assign(wp.array(target_pos, dtype=float, device=self.model.device))

    def simulate(self) -> None:
        self._update_drive_targets()

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.frame += 1
        self._set_joint_targets()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def metrics(self) -> dict[str, float]:
        body_q = self.state_0.body_q.numpy()
        joint_target = self.control.joint_target_pos.numpy()
        return {
            "time": float(self.sim_time),
            "hand_z": float(body_q[self._hand_body, self.params["vertical_axis"]]),
            "left_finger_y": float(body_q[self._left_finger_body, 1]),
            "right_finger_y": float(body_q[self._right_finger_body, 1]),
            "gripper_target": float(joint_target[self.params["gripper_joint_indices"][0]]),
            "reduced_solve": float(self.params["body_enable_reduced_solve"]),
        }

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        joint_q = self.state_0.joint_q.numpy()
        assert np.all(np.isfinite(body_q)), "Franka body poses contain non-finite values"
        assert np.all(np.isfinite(body_qd)), "Franka body velocities contain non-finite values"
        assert np.all(np.isfinite(joint_q)), "Franka joint coordinates contain non-finite values"

        open_joint_value = self._gripper_joint_value(self.params["gripper_open_frac"])
        closed_joint_value = self._gripper_joint_value(self.params["gripper_closed_frac"])
        assert abs(open_joint_value - 0.5 * self.params["gripper_open_gap"]) < self.params["gripper_gap_test_tolerance"]
        assert (
            abs(closed_joint_value - 0.5 * self.params["gripper_closed_gap"])
            < self.params["gripper_gap_test_tolerance"]
        )
        hand_z = float(body_q[self._hand_body][self.params["vertical_axis"]])
        assert hand_z > self.params["hand_lift_test_min_z"], f"Franka hand did not lift: z={hand_z:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--reduced-solve", action="store_true", help="Enable RVBD reduced-coordinate projection.")
        parser.add_argument("--reduced-gn-iterations", type=int, default=PARAMS["reduced_gn_iterations"])
        parser.add_argument("--reduced-gn-damping", type=float, default=PARAMS["reduced_gn_damping"])
        parser.add_argument("--arm-drive-scale", type=float, default=1.0)
        parser.add_argument("--gripper-drive-scale", type=float, default=None)
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
