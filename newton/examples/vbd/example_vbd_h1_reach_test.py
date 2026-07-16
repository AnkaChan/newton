# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
###########################################################################
# H1 arm reach test (robot only)
#
# Diagnostic companion to example_vbd_trash_bag_h1_grab_test.py: the SAME
# H1, joint drives, IK rig, and palm-up hook hand state — but no bag,
# rope, can, or table. The hands climb a ladder of targets (z 1.3 -> 1.9)
# and finish in the demo's up-and-back raise pose, printing target vs
# actual pinch position every half second. Use it to tell arm/IK reach
# limits apart from rope-load stalls.
###########################################################################

from __future__ import annotations

import example_vbd_trash_bag_h1_grab_test as grab
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

PARAMS = dict(grab.PARAMS)

# ladder rungs the hands climb (x, y magnitude, z), then the demo-like
# up-and-back finish. Stay inside the stable envelope mapped with this
# test: commanded z <= ~1.8, and no large lateral sweeps while overhead
# (~1.75+) — both trigger IK branch flips that detonate the stiff solve.
HOOK_POSE = (-0.05, 0.19, 1.17)
LADDER_Z = (1.3, 1.5, 1.7, 1.85)
FINISH_POSE = (-0.35, 0.17, 1.75)


class Example:
    def __init__(self, viewer, args, params: dict | None = None):
        self.viewer = viewer
        self.params = dict(PARAMS) if params is None else params
        p = self.params
        self.sim_time = 0.0
        self.fps = p["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = p["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0

        builder = newton.ModelBuilder(gravity=p["gravity"])
        robot_bodies, _robot_rigid_shapes, _hook_shapes = grab._add_h1(builder, p)
        builder.add_ground_plane()
        self.robot_coord_count = builder.joint_coord_count

        builder.enable_rigid_mesh_sdfs()
        builder.color()
        self.model = builder.finalize()
        self.robot_bodies = robot_bodies

        self.hand_bodies = [robot_bodies["left_hand"], robot_bodies["right_hand"]]
        self.hand_offsets = [wp.vec3(*values) for values in grab.HAND_OFFSETS]
        self.hand_rotations = [grab._normalized_quat(values) for values in grab.HAND_ROTATIONS]

        self.phase = "rest"
        self._phase_marks: list[tuple[float, str]] = []
        self._build_choreography()
        self._setup_ik()
        # start from the task-space rest pose, exactly like the demo
        self._solve_ik(
            np.asarray([p["rest_left"], p["rest_right"]], dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            iterations=48,
        )
        self.model.joint_q.assign(self.ik_joint_q_flat)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # solver AFTER the pose assignment (see the demo's setup_sim docstring)
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=p["solver_iterations"],
            rigid_body_contact_buffer_size=p["rigid_body_contact_buffer_size"],
            rigid_contact_hard=p["rigid_contact_hard"],
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
            rigid_joint_linear_kd=1.0e2,
            rigid_joint_angular_kd=1.0e2,
        )
        self.pipeline = newton.CollisionPipeline(self.model, broad_phase="nxn")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)
        wp.copy(self.control.joint_target_q, self.model.joint_q, count=self.robot_coord_count)
        self.control.joint_target_qd.zero_()
        self.previous_joint_targets = wp.clone(self.model.joint_q[: self.robot_coord_count])

        self._capture_graph()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(*p["camera_position"]), p["camera_pitch"], p["camera_yaw"])
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = p["camera_fov"]

    # ── choreography ─────────────────────────────────────────────────────

    def _build_choreography(self):
        p = self.params
        tracks: dict[str, grab._Track] = {}
        for side, rest in (("left", p["rest_left"]), ("right", p["rest_right"])):
            tracks[f"{side}_pos"] = grab._Track(np.asarray(rest, dtype=np.float64))
            tracks[f"{side}_thumb"] = grab._Track(0.0)
            tracks[f"{side}_index"] = grab._Track(0.0)
            tracks[f"{side}_other"] = grab._Track(0.0)
        self.tracks = tracks

        for side, sign in (("left", 1.0), ("right", -1.0)):
            cur = grab._HandCursor(tracks, side)
            cur.time = 0.5
            # hook-grab hand state: palm up, thumb perpendicular, hook curl
            cur.move(
                2.0,
                pos=(HOOK_POSE[0], sign * HOOK_POSE[1], HOOK_POSE[2]),
                thumb=1.0,
                index=p["hook_curl"],
                other=p["hook_curl"],
            )
            if side == "left":
                self._phase_marks.append((cur.time, "hook"))
            cur.wait(1.0)
            for z in LADDER_Z:
                cur.move(2.0, pos=(HOOK_POSE[0], sign * HOOK_POSE[1], z))
                if side == "left":
                    self._phase_marks.append((cur.time, f"z{z:.1f}"))
                cur.wait(1.0)
            cur.move(3.0, pos=(FINISH_POSE[0], sign * FINISH_POSE[1], FINISH_POSE[2]))
            if side == "left":
                self._phase_marks.append((cur.time, "finish"))
            cur.wait(2.0)
            self.total_time = cur.time

    # ── IK (identical rig to the grab demo) ──────────────────────────────

    def _setup_ik(self):
        initial_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, initial_state)
        body_q = initial_state.body_q.numpy()

        self.torso_body = self.robot_bodies["torso"]
        torso_transform = wp.transform(*body_q[self.torso_body])
        self.torso_position_objective = ik.IKObjectivePosition(
            link_index=self.torso_body,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(torso_transform)], dtype=wp.vec3),
            weight=self.params["torso_ik_position_weight"],
        )
        self.torso_rotation_objective = ik.IKObjectiveRotation(
            link_index=self.torso_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*wp.transform_get_rotation(torso_transform))], dtype=wp.vec4),
            weight=self.params["torso_ik_rotation_weight"],
        )

        self.position_objectives = []
        self.rotation_objectives = []
        for body, offset, rotation in zip(self.hand_bodies, self.hand_offsets, self.hand_rotations, strict=True):
            initial_position = wp.transform_point(wp.transform(*body_q[body]), offset)
            self.position_objectives.append(
                ik.IKObjectivePosition(
                    link_index=body,
                    link_offset=offset,
                    target_positions=wp.array([initial_position], dtype=wp.vec3),
                    weight=5.0,
                )
            )
            self.rotation_objectives.append(
                ik.IKObjectiveRotation(
                    link_index=body,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([wp.vec4(*rotation)], dtype=wp.vec4),
                    weight=0.2,
                )
            )

        self.elbow_objectives = []
        self.shoulder_positions = []
        off = self.params["elbow_target_offset"]
        for side, sign in (("left", 1.0), ("right", -1.0)):
            shoulder_pos = np.asarray(body_q[self.robot_bodies[f"{side}_shoulder"]][:3], dtype=np.float64)
            self.shoulder_positions.append((shoulder_pos, sign))
            elbow_target = shoulder_pos + np.asarray([off[0], sign * off[1], off[2]])
            self.elbow_objectives.append(
                ik.IKObjectivePosition(
                    link_index=self.robot_bodies[f"{side}_elbow"],
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.vec3(*elbow_target)], dtype=wp.vec3),
                    weight=0.3,
                )
            )

        # margin-shrunk arm limits: an IK target parked exactly AT a
        # mechanical stop makes the solver's limit constraint fight the
        # drive at the boundary and explode (see the grab demo's _setup_ik)
        ik_lo = self.model.joint_limit_lower.numpy().copy()
        ik_hi = self.model.joint_limit_upper.numpy().copy()
        qd_starts = self.model.joint_qd_start.numpy()
        for j, label in enumerate(self.model.joint_label):
            if "shoulder" in label or "elbow" in label:
                d = int(qd_starts[j])
                ik_lo[d] += self.params["ik_limit_margin"]
                ik_hi[d] -= self.params["ik_limit_margin"]
        joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(ik_lo, dtype=float),
            joint_limit_upper=wp.array(ik_hi, dtype=float),
            weight=5.0,
        )
        self.ik_joint_q = wp.clone(self.model.joint_q).reshape((1, self.model.joint_coord_count))
        self.ik_joint_q_flat = self.ik_joint_q.reshape((-1,))
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[
                *self.position_objectives,
                *self.rotation_objectives,
                *self.elbow_objectives,
                self.torso_position_objective,
                self.torso_rotation_objective,
                joint_limits,
            ],
            lambda_initial=1.0,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        q_starts = self.model.joint_q_start.numpy()
        finger_indices = []
        closed_values = []
        finger_groups = []
        for side, thumb_values in (("L", grab.THUMB_CLOSED_VALUES[0]), ("R", grab.THUMB_CLOSED_VALUES[1])):
            thumb_yaw, thumb_pitch, thumb_inter, thumb_distal = thumb_values
            finger_names_and_values = (
                ("thumb_proximal_yaw_joint", thumb_yaw),
                ("thumb_proximal_pitch_joint", thumb_pitch),
                ("thumb_intermediate_joint", thumb_inter),
                ("thumb_distal_joint", thumb_distal),
                ("index_proximal_joint", 1.2),
                ("index_intermediate_joint", 1.2),
                ("middle_proximal_joint", 1.0),
                ("middle_intermediate_joint", 1.0),
                ("ring_proximal_joint", 1.0),
                ("ring_intermediate_joint", 1.0),
                ("pinky_proximal_joint", 1.0),
                ("pinky_intermediate_joint", 1.0),
            )
            thumb_group = grab._FINGER_GROUP_LEFT_THUMB if side == "L" else grab._FINGER_GROUP_RIGHT_THUMB
            index_group = grab._FINGER_GROUP_LEFT_INDEX if side == "L" else grab._FINGER_GROUP_RIGHT_INDEX
            other_group = grab._FINGER_GROUP_LEFT_OTHER if side == "L" else grab._FINGER_GROUP_RIGHT_OTHER
            for suffix, value in finger_names_and_values:
                joint = grab._find_suffix(self.model.joint_label, f"{side}_{suffix}")
                finger_indices.append(int(q_starts[joint]))
                closed_values.append(value)
                if suffix.startswith("thumb_"):
                    finger_groups.append(thumb_group)
                elif suffix.startswith("index_"):
                    finger_groups.append(index_group)
                else:
                    finger_groups.append(other_group)
        self.finger_indices = wp.array(finger_indices, dtype=wp.int32, device=self.model.device)
        self.closed_finger_values = wp.array(closed_values, dtype=float, device=self.model.device)
        self.finger_groups = wp.array(finger_groups, dtype=wp.int32, device=self.model.device)
        self.finger_fractions = wp.zeros(6, dtype=float, device=self.model.device)

    def _solve_ik(self, positions: np.ndarray, fractions: np.ndarray, iterations: int = 24):
        for i, (objective, position) in enumerate(zip(self.position_objectives, positions, strict=True)):
            # same safe-envelope clamp as the grab demo (see its _solve_ik)
            shoulder, _sign = self.shoulder_positions[i]
            target = np.asarray(position, dtype=np.float64)
            rel = target - shoulder
            dist = float(np.linalg.norm(rel))
            max_reach = self.params["max_hand_reach"]
            if dist > max_reach:
                target = shoulder + rel * (max_reach / dist)
            objective.set_target_position(0, wp.vec3(*target))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=iterations)
        # diagnostic: largest single-frame jump in the IK solution since the
        # last report (a branch flip shows up as a ~radian spike)
        ik_q = self.ik_joint_q_flat.numpy()[: self.robot_coord_count]
        if getattr(self, "_prev_ik_q", None) is not None:
            self._dbg_max_dq = max(getattr(self, "_dbg_max_dq", 0.0), float(np.max(np.abs(ik_q - self._prev_ik_q))))
        self._prev_ik_q = ik_q
        self.finger_fractions.assign(np.asarray(fractions, dtype=np.float32))
        wp.launch(
            grab.set_finger_targets,
            dim=self.finger_indices.shape[0],
            inputs=[
                self.ik_joint_q_flat,
                self.finger_indices,
                self.closed_finger_values,
                self.finger_groups,
                self.finger_fractions,
            ],
        )

    def _update_trajectory(self):
        t = self.sim_time
        tr = self.tracks
        self.target_hand_positions = np.asarray([tr["left_pos"].sample(t), tr["right_pos"].sample(t)], dtype=np.float32)
        fractions = np.asarray(
            [
                tr["left_thumb"].sample(t),
                tr["right_thumb"].sample(t),
                tr["left_index"].sample(t),
                tr["right_index"].sample(t),
                tr["left_other"].sample(t),
                tr["right_other"].sample(t),
            ],
            dtype=np.float32,
        ).reshape(-1)
        for time, name in self._phase_marks:
            if t >= time:
                self.phase = name
        self._solve_ik(self.target_hand_positions, fractions)
        wp.launch(
            grab.update_control_targets,
            dim=self.robot_coord_count,
            inputs=[
                self.ik_joint_q_flat,
                self.previous_joint_targets,
                1.0 / self.frame_dt,
                self.params["joint_target_velocity_limit"],
            ],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
        )

    # ── simulation loop ──────────────────────────────────────────────────

    def _capture_graph(self):
        self.graph = None
        if not self.params["enable_cuda_graph"] or not wp.get_device().is_cuda:
            return
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self._update_trajectory()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        if self.frame % 30 == 0:
            self._report()

    def _report(self):
        # which arm joints the IK solution parks near a mechanical limit
        if not hasattr(self, "_limit_info"):
            lo = self.model.joint_limit_lower.numpy()
            hi = self.model.joint_limit_upper.numpy()
            q_starts = self.model.joint_q_start.numpy()
            qd_starts = self.model.joint_qd_start.numpy()
            self._limit_info = []
            for j, label in enumerate(self.model.joint_label):
                if ("shoulder" in label or "elbow" in label) and q_starts[j + 1] - q_starts[j] == 1:
                    d = int(qd_starts[j])
                    self._limit_info.append((int(q_starts[j]), float(lo[d]), float(hi[d]), label.split("/")[-1]))
        ik_q = self.ik_joint_q_flat.numpy()
        near = [
            f"{name} q{ik_q[qi]:+.2f} lim[{lo:+.2f},{hi:+.2f}]"
            for qi, lo, hi, name in self._limit_info
            if ik_q[qi] < lo + 0.08 or ik_q[qi] > hi - 0.08
        ]
        if near:
            print("[h1_reach]   NEAR LIMIT: " + " | ".join(near), flush=True)
        body_q = self.state_0.body_q.numpy()
        line = f"[h1_reach] t={self.sim_time:5.2f} {self.phase:7s}"
        for i, hand in enumerate(("L", "R")):
            pin = np.asarray(wp.transform_point(wp.transform(*body_q[self.hand_bodies[i]]), self.hand_offsets[i]))
            tgt = self.target_hand_positions[i]
            err = float(np.linalg.norm(pin - tgt))
            line += f" | {hand} tgt z{tgt[2]:.2f} got z{pin[2]:.2f} err {err:.3f}"
        line += f" | ik dq {getattr(self, '_dbg_max_dq', 0.0):.3f}"
        self._dbg_max_dq = 0.0
        print(line, flush=True)

    def gui(self, ui):
        ui.text(f"Phase: {self.phase}")
        ui.text(f"t = {self.sim_time:.2f} / {self.total_time:.2f} s")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1500)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
