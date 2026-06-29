# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shirt-pick scene: a Franka FR3 picks a unisex shirt off the ground.

Reproduces the IsaacLab ``Isaac-Pick-AVBD-Cloth-Direct-v0`` layout for every
solver (so AVBD and proxy runs are directly comparable): the robot's default
joint configuration, the cloth's placed rest state (taken verbatim from the
IsaacLab env so the deformable's world layout is identical), the home EE pose,
and the scripted pick state machine (``pick_cloth_sm``).
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton

from ..robots import HAND_BODY_SUFFIX, add_franka
from .base import Scene
from . import register

# Shirt cloth material (IsaacLab NewtonSurfaceDeformableBodyMaterialCfg).
CLOTH_DENSITY = 0.02
CLOTH_TRI_KE = 1.0e4
CLOTH_TRI_KA = 1.0e4
CLOTH_TRI_KD = 1.5e-6
CLOTH_EDGE_KE = 0.5
CLOTH_EDGE_KD = 1.0e-2
CLOTH_PARTICLE_RADIUS = 0.01

# Raw unisex-shirt mesh (vertices in cm, local frame) extracted once from
# newton/examples/assets/unisex_shirt.usd. It is placed at load time exactly the
# way IsaacLab's DeformableObjectCfg spawns it (scale + init_state rot/pos),
# rather than pre-baked into world coordinates.
_CLOTH_NPZ = os.path.join(os.path.dirname(__file__), "..", "assets", "unisex_shirt_mesh.npz")

# IsaacLab DeformableObjectCfg shirt spawn transform:
#   spawn.scale    = (0.01, 0.01, 0.01)   USD vertices are in cm
#   init_state.rot = (1, 0, 0, 0)         (qx,qy,qz,qw) = 180 deg about X; flips the
#                                         Y-up cm mesh into the Z-up world
#   init_state.pos = (0.5, 1.25, 0.10)
CLOTH_SCALE = 0.01
CLOTH_ROT = (1.0, 0.0, 0.0, 0.0)  # (qx, qy, qz, qw)
# CLOTH_POS = (0.5, 1.25, 0.10)
CLOTH_POS = (0.45, 1.20, 0.10)

# Spawn joint configuration (7 arm + 2 finger).
# ROBOT_INIT_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ROBOT_INIT_Q = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]

# Home / hover EE pose (IsaacLab AVBD env preset _ee_tf) and pick_cloth_sm
# descent height.
# HOME_POS = (0.7302, 0.0836, 0.3713)
HOME_POS = (0.69, 0.180, 0.48)
HOME_QUAT = (0.7140, -0.6664, -0.0916, 0.1943)  # (qx, qy, qz, qw)
GRASP_Z = 0.0981


@register
class ShirtPickScene(Scene):
    key = "shirt_pick"
    ik_link_label = HAND_BODY_SUFFIX
    ik_link_offset = wp.vec3(0.0, 0.0, 0.0)  # IsaacLab targets panda_hand directly
    ik_joint_limit_weight = 0.0  # IsaacLab disables the joint-limit objective
    ik_iters = 24
    default_sequence = "pick"

    def __init__(self, args):
        super().__init__(args)
        self.grasp_z = float(args.grasp_z)
        data = np.load(_CLOTH_NPZ)
        verts = data["vertices"].astype(np.float64)  # raw, cm, local frame
        tri = data["tri_indices"].astype(np.int32)  # (n_tri, 3) into verts
        self._vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]
        self._indices = list(tri.reshape(-1))

    def robot_init_q(self):
        return list(ROBOT_INIT_Q)

    # -- robot ------------------------------------------------------------
    def build_robot(self, builder, *, collapse_fixed_joints):
        return add_franka(builder, collapse_fixed_joints=collapse_fixed_joints)

    # -- world ------------------------------------------------------------
    def add_static(self, builder):
        start = builder.shape_count
        plane_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e-5, mu=1.0, margin=0.0, gap=0.01)
        builder.add_ground_plane(cfg=plane_cfg, label="ground_plane")
        return list(range(start, builder.shape_count))

    def add_deformables(self, builder):
        # Place the raw cm mesh exactly as IsaacLab's DeformableObjectCfg does:
        # scale to meters, rotate (Y-up cm -> Z-up world), translate to init pos.
        builder.add_cloth_mesh(
            vertices=self._vertices,
            indices=self._indices,
            pos=wp.vec3(*CLOTH_POS),
            rot=wp.quat(*CLOTH_ROT),
            scale=CLOTH_SCALE,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=CLOTH_DENSITY,
            tri_ke=CLOTH_TRI_KE,
            tri_ka=CLOTH_TRI_KA,
            tri_kd=CLOTH_TRI_KD,
            edge_ke=CLOTH_EDGE_KE,
            edge_kd=CLOTH_EDGE_KD,
            particle_radius=CLOTH_PARTICLE_RADIUS,
            label="shirt",
        )

    # -- task -------------------------------------------------------------
    def home_pose(self):
        return np.array(HOME_POS, dtype=np.float64), np.array(HOME_QUAT, dtype=np.float64)

    def sequences(self, home_pos, home_quat):
        from ..controllers.base import Keyframe
        from ..controllers.sequences import KeyframeSequence
        from ..robots import GRIP_CLOSE, GRIP_OPEN

        home = np.asarray(home_pos)
        grasp = home.copy()
        grasp[2] = self.grasp_z
        q = np.asarray(home_quat)
        return {
            # Scripted pick: hover -> descend -> grasp -> lift -> hold.
            "pick": KeyframeSequence(
                [
                    Keyframe(0.8, home, q, GRIP_OPEN),    # hover at home, gripper open
                    Keyframe(1.2, grasp, q, GRIP_OPEN),   # descend to grasp height
                    Keyframe(0.8, grasp, q, GRIP_CLOSE),  # close the gripper
                    Keyframe(1.2, home, q, GRIP_CLOSE),   # lift back to home
                    Keyframe(2.0, home, q, GRIP_CLOSE),   # hold
                ]
            ),
            # Hold the home pose, gripper open (debug / settle).
            "hold": KeyframeSequence([Keyframe(5.0, home, q, GRIP_OPEN)]),
        }

    # -- presentation -----------------------------------------------------
    def camera(self):
        return (wp.vec3(1.6, -1.2, 1.0), -25.0, 125.0, wp.vec3(0.5, 0.0, 0.15))

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--grasp_z", type=float, default=GRASP_Z, help="EE descent height before grasping [m].")
