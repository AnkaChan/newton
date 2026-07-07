# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Monolithic AVBD strategy: the rigid arm is integrated by VBD's AVBD path.

A single :class:`SolverVBD` owns both the cloth particles and the articulated
arm; there is no separate rigid solver. Arm/ground collisions are filtered so
only the cloth contacts the statics, and the arm tracks stiff implicit PD
targets through the AVBD augmented-Lagrangian coupling.
"""

from __future__ import annotations

import newton
from newton.solvers import SolverVBD

from .base import SolverStrategy
from . import register

# IsaacLab FRANKA_PANDA_AVBD_CFG actuator gains (defaults; scenes may override
# per (scene, solver) via Scene.robot_gains). Arm: stiff position drive
# (ke=1e6) with light damping. NOTE: the damping must stay small -- with substep
# dt ~1/600 a large kd (e.g. 1e4) makes the explicit damping term unstable
# (dt*kd >> 2) and the arm oscillates. Arm damping is task-dependent in IsaacLab
# (0.01 for cloth, 0.1 for the cube), so scenes override it. Fingers: ke=1e4,
# kd=0.1 (IsaacLab panda_hand), same across the AVBD tasks.
ARM_STIFFNESS = 1.0e6
ARM_DAMPING = 0.01
FINGER_STIFFNESS = 1.0e4
FINGER_DAMPING = 0.01
# Effort limits [N·m / N]: arm joints 1-4, joints 5-7, fingers.
EFFORT_LIMIT = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 200.0, 200.0]


@register
class MonolithicAvbdStrategy(SolverStrategy):
    key = "avbd"
    collapse_fixed_joints = True
    uses_mujoco = False
    decimation = 2  # IsaacLab Isaac-Pick-AVBD-Cloth-Direct-v0
    # Ramp the AVBD joint penalties before the first rendered frame; a cold
    # solver leaves the arm's joints slack and it collapses on frame 1.
    warmup_steps = 8

    def register_attributes(self, builder):
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

    def configure_robot(self, builder, robot_bodies, robot_joints):
        gains = {
            "arm_stiffness": ARM_STIFFNESS,
            "arm_damping": ARM_DAMPING,
            "finger_stiffness": FINGER_STIFFNESS,
            "finger_damping": FINGER_DAMPING,
        }
        gains.update(self.scene_robot_gains())  # scene overrides win
        builder.joint_target_ke[:7] = [gains["arm_stiffness"]] * 7
        builder.joint_target_kd[:7] = [gains["arm_damping"]] * 7
        builder.joint_target_ke[7:9] = [gains["finger_stiffness"]] * 2
        builder.joint_target_kd[7:9] = [gains["finger_damping"]] * 2
        builder.joint_effort_limit[:9] = EFFORT_LIMIT
        # Arm rotor inertia (IsaacLab FRANKA_PANDA armature=1e-3). Critical for
        # stability of the stiff 1e6/0.01 position drive -- without it the arm
        # blows up. The fingers have no armature.
        builder.joint_armature[:7] = [1.0e-3] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]

    def filter_collisions(self, builder, robot_shapes, static_shapes):
        # Only the cloth contacts the statics; cloth-vs-arm is VBD rigid contact.
        for rs in robot_shapes:
            for ss in static_shapes:
                builder.add_shape_collision_filter_pair(rs, ss)

    def apply_materials(self, model):
        mats = {
            "shape_material_ke": 1.0e4,
            "shape_material_kd": 1.0,
            "shape_material_mu": 1.5,
            "soft_contact_ke": 1.0e4,
            "soft_contact_kd": 1.0e-2,
            "soft_contact_mu": 1.5,
        }
        mats.update(self.scene_materials())  # scene overrides win
        model.shape_material_ke.fill_(mats["shape_material_ke"])
        model.shape_material_kd.fill_(mats["shape_material_kd"])
        model.shape_material_mu.fill_(mats["shape_material_mu"])
        model.soft_contact_ke = mats["soft_contact_ke"]
        model.soft_contact_kd = mats["soft_contact_kd"]
        model.soft_contact_mu = mats["soft_contact_mu"]

    def post_finalize(self, model, handles):
        # VBD uses model.body_q as the articulation rest pose.
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        self._robot_joints = handles.robot_joints

    def build_solver(self, model, handles, args):
        # Strategy defaults are the shirt_pick values; scenes override the
        # solver-specific pieces (e.g. self-contact radii/buffers for the
        # IsaacLab AVBD tasks) via Scene.solver_overrides.
        vbd_kwargs = dict(
            iterations=int(args.vbd_iterations),
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=2.0e-3,
            particle_self_contact_margin=2.0e-3,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.0,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=1.0e2,
            rigid_avbd_beta=1.0e5,
            rigid_avbd_gamma=0.99,
            rigid_joint_linear_k_start=1.0e4,
            rigid_joint_angular_k_start=1.0e1,
            rigid_joint_linear_ke=1.0e9,
            rigid_joint_angular_ke=1.0e9,
            rigid_joint_linear_kd=1.0e-2,
            rigid_joint_angular_kd=0.0,
            rigid_contact_history=False,
        )
        if getattr(args, "water_tight", False):
            # Water-tight adds edge/face cloth-mesh contacts on top of the per-particle
            # ones; raise the per-body soft-contact buffer so they are not dropped
            # (the default 256 overflows to ~400 on the shirt-pick grasp).
            vbd_kwargs["rigid_body_particle_contact_buffer_size"] = 2048
        vbd_kwargs.update(self.scene_solver_overrides())  # scene overrides win
        self.solver = SolverVBD(model=model, **vbd_kwargs)
        # NOTE: IsaacLab's NewtonVBDManager does NOT change joint constraint mode,
        # so joints stay in the default hard (augmented-Lagrangian) mode. Forcing
        # soft penalty-only constraints (set_joint_constraint_mode hard=False) is
        # far less stable for a stiff arm and makes it blow up.
        return self.solver

    def pre_substeps(self, solver, state):
        # IsaacLab's VBD manager rebuilds the particle BVH once per step.
        if hasattr(solver, "rebuild_bvh"):
            solver.rebuild_bvh(state)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--vbd-iterations", type=int, default=10, help="VBD iterations per substep (avbd).")
