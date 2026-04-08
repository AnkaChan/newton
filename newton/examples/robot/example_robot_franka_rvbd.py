# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka RVBD
#
# Simulates a Franka Panda arm using the VBD solver with reduced-coordinate
# projection (RVBD). The arm tracks sinusoidal joint targets under gravity,
# demonstrating exact kinematic constraint satisfaction.
#
# Command: python -m newton.examples robot_franka_rvbd
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )

        # Approximate mesh colliders as convex hulls for VBD compatibility
        builder.approximate_meshes(method="convex_hull", keep_visual_shapes=True)

        # Initial joint configuration (roughly upright pose)
        init_q = [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0, 0.04, 0.04]
        n_joints = min(len(init_q), len(builder.joint_q))
        builder.joint_q[:n_joints] = init_q[:n_joints]
        builder.joint_target_pos[:n_joints] = init_q[:n_joints]

        # Joint drive parameters
        builder.joint_target_ke[:n_joints] = [650.0] * n_joints
        builder.joint_target_kd[:n_joints] = [100.0] * n_joints
        builder.joint_effort_limit[:7] = [80.0] * min(7, n_joints)
        if n_joints > 7:
            builder.joint_effort_limit[7:n_joints] = [20.0] * (n_joints - 7)
        builder.joint_armature[:7] = [0.1] * min(7, n_joints)
        if n_joints > 7:
            builder.joint_armature[7:n_joints] = [0.5] * (n_joints - 7)

        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=10,
            body_enable_reduced_solve=True,
            reduced_gn_iterations=3,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()

        if viewer:
            self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            if self.viewer:
                self.viewer.apply_forces(self.state_0)

            # Sinusoidal joint targets for the 7 arm joints
            joint_target_pos = self.control.joint_target_pos.numpy()
            for i in range(min(7, len(joint_target_pos))):
                joint_target_pos[i] = 0.3 * math.sin(self.sim_time * 0.5 + i * 0.5)
            self.control.joint_target_pos.assign(joint_target_pos)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify joint constraint satisfaction after simulation."""
        state = self.state_0
        joint_q = wp.zeros(self.model.joint_coord_count, dtype=float, device=self.model.device)
        joint_qd = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.model.device)
        newton.eval_ik(self.model, state, joint_q, joint_qd)

        state_check = self.model.state()
        newton.eval_fk(self.model, joint_q, joint_qd, state_check)

        body_q = state.body_q.numpy().reshape(-1, 7)
        body_q_check = state_check.body_q.numpy().reshape(-1, 7)

        max_pos_err = np.max(np.abs(body_q[:, :3] - body_q_check[:, :3]))
        assert max_pos_err < 1e-3, f"Position constraint violation: {max_pos_err:.6f} m"


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    example = Example(viewer, args)

    newton.examples.run(example, args)
