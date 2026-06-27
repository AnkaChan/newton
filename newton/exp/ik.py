# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU IK controller shared by all experiments.

The IK runs on a robot-only model so the solver never sees the deformable
objects. The arm is added first in the simulation model, so its coordinates
``[0, n_coords)`` line up with this model's coordinates and the solved joints
can be copied straight into ``control.joint_target_q``.

Matching IsaacLab: each step the IK is re-seeded from the measured joint state
(:meth:`seed_from_state`) before solving, so it tracks the actual arm.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton.ik as ik

from .robots import find_label_index, set_gripper_q


class IKController:
    def __init__(
        self,
        ik_model,
        link_label,
        link_offset,
        home_pos,
        home_quat,
        *,
        iters=24,
        joint_limit_weight=0.0,
        lambda_initial=0.1,  # newton.ik.IKSolver default (matches IsaacLab)
    ):
        self.model = ik_model
        self.iters = int(iters)
        self.n_coords = ik_model.joint_coord_count
        self.joint_q = wp.array(ik_model.joint_q, shape=(1, self.n_coords))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.full(1, 0.04, dtype=float)
        hand_body = find_label_index(ik_model.body_label, link_label)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=hand_body,
            link_offset=link_offset,
            target_positions=wp.array([wp.vec3(*np.asarray(home_pos, dtype=float).tolist())], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*np.asarray(home_quat, dtype=float).tolist())], dtype=wp.vec4),
        )
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=ik_model.joint_limit_lower,
            joint_limit_upper=ik_model.joint_limit_upper,
            weight=float(joint_limit_weight),
        )
        self.solver = ik.IKSolver(
            model=ik_model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=lambda_initial,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def set_target(self, pos: wp.vec3, quat: wp.vec4):
        self.pos_obj.set_target_position(0, pos)
        self.rot_obj.set_target_rotation(0, quat)

    def set_finger(self, width: float):
        self.finger_pos_buf.fill_(float(width))

    def seed_from_state(self, joint_q):
        """Re-seed the IK warm start from the measured joint coordinates."""
        wp.copy(self.joint_q, joint_q, count=self.n_coords)

    def solve(self):
        self.solver.step(self.joint_q, self.joint_q, iterations=self.iters)

    def write_control(self, control):
        """Write solved arm joints + gripper width into ``control.joint_target_q``."""
        wp.launch(set_gripper_q, dim=1, inputs=[self.joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1])
        wp.copy(control.joint_target_q, self.joint_q, count=self.n_coords)

    def seed(self, joint_q_np):
        self.joint_q.assign(np.array([np.asarray(joint_q_np, dtype=np.float32)[: self.n_coords]], dtype=np.float32))
