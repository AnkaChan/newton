# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pick-AVBD-cube scene: a Franka picks a soft (tet-mesh) cube.

Ports the world layout of IsaacLab ``Isaac-Pick-AVBD-Cube-Direct-v0``: a 0.05 m
deformable cube (FEM tetrahedra, Young's modulus 2.5e5, Poisson 0.25, density
500) resting in front of the arm, plus the Franka at its default ready pose and
the pre-grasp home EE pose from the env's ``_ee_tf``.

Provides a scripted ``pick`` sequence (hover -> descend so the fingertips reach
the settled cube's Z-center -> close the gripper -> lift -> hold), mirroring the
shirt-pick example, plus a ``hold`` placeholder (pair with ``--solver avbd``, the
monolithic VBD solver the IsaacLab task uses).
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ..robots import HAND_BODY_SUFFIX, add_franka
from . import register
from .base import Scene

# Deformable cube (IsaacLab TetMeshCuboidCfg + DeformableBodyMaterialCfg).
CUBE_SIZE = 0.05  # edge length [m]
CUBE_POS = (0.35, 0.0, 0.05)  # centroid at spawn
CUBE_DENSITY = 500.0
CUBE_YOUNGS = 2.5e6
CUBE_POISSON = 0.25
CUBE_PARTICLE_RADIUS = 0.005
CUBE_DIM = 4  # hex cells per axis (each split into 5 tets) -> 125 nodes / 320 tets, matching IsaacLab

# fr3_hand -> fingertip (fr3_hand_tcp) offset from fr3_franka_hand.urdf: 0.1034 m
# along the hand's local +Z. IK drives the fr3_hand wrist frame directly
# (``ik_link_offset`` = 0), so to place the fingertips at a world point the
# hand-frame target must be pulled back by this offset rotated into world frame.
FINGERTIP_OFFSET = 0.1034  # [m]

# The cube spawns with its centroid at CUBE_POS but floating a half-edge above the
# ground; under gravity it drops ~0.021 m and settles resting on the plane, well
# before the grasp. Aiming at the spawn centroid (0.05) therefore hits the settled
# cube's TOP. Settled centroid Z, measured with the arm held at home under
# ``--solver avbd``: ~0.029 m -- the grasp aims the fingertips here so the pads
# straddle the SETTLED cube's Z-center.
CUBE_SETTLED_Z = 0.0293  # [m]

# Lame parameters from (E, nu): mu = E/(2(1+nu)), lambda = E*nu/((1+nu)(1-2nu)).
CUBE_K_MU = CUBE_YOUNGS / (2.0 * (1.0 + CUBE_POISSON))
CUBE_K_LAMBDA = CUBE_YOUNGS * CUBE_POISSON / ((1.0 + CUBE_POISSON) * (1.0 - 2.0 * CUBE_POISSON))
CUBE_K_DAMP = 0.0  # IsaacLab material specifies no explicit damping

# Franka spawn configuration (IsaacLab FRANKA_PANDA_CFG default ready pose).
ROBOT_INIT_Q = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]

# Pre-grasp home EE pose above the cube (IsaacLab pick_avbd_cube env _ee_tf).
# HOME_POS = (0.3022, 0.0000, 0.1257)
HOME_POS = (0.3800, 0.0000, 0.2057)
# HOME_QUAT = (0.9654, 0.0214, -0.2598, -0.0058)  # (qx, qy, qz, qw)
HOME_QUAT = (0.9916, 0.0220, -0.1277, -0.0029)
# HOME_QUAT = (1.0, 0.0, 0.0, 0.0)

# Model-level contact params for the AVBD solver (IsaacLab pick_avbd_cube MODEL_CFG).
_AVBD_MODEL_MATERIALS = {
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1.0,
    "soft_contact_mu": 1.5,
    "shape_material_ke": 1.0e5,
    "shape_material_kd": 1.0,
    "shape_material_mu": 1.5,
}

# VBD self-contact params for the AVBD solver (IsaacLab pick_avbd_cube
# VBDSolverCfg -- these differ from the strategy's shirt_pick defaults).
_AVBD_SOLVER_OVERRIDES = {
    "particle_enable_self_contact": True,
    "particle_self_contact_radius": 5.0e-3,
    "particle_self_contact_margin": 5.0e-3,
    "particle_topological_contact_filter_threshold": 2,
    "particle_vertex_contact_buffer_size": 32,
    "particle_edge_contact_buffer_size": 64,
}


def _quat_rotate(q, v):
    """Rotate vec3 ``v`` by quaternion ``q=(qx,qy,qz,qw)`` on the host (numpy)."""
    qv = np.asarray(q[:3], dtype=np.float64)
    w = float(q[3])
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _make_cube_tet_mesh(size: float, dim: int):
    """Regular tetrahedral cube mesh: ``(dim+1)^3`` grid nodes, 5 tets per hex
    cell. Vertices are in the local frame with the origin at a corner; returns
    ``(vertices, indices)`` where ``indices`` is flattened (4 per tet), suitable
    for :meth:`newton.ModelBuilder.add_soft_mesh`. Mirrors ``add_soft_grid``'s
    decomposition (alternating split by cell parity).
    """
    cell = size / dim
    n = dim + 1

    def gi(x, y, z):
        return n * n * z + n * y + x

    vertices = [wp.vec3(x * cell, y * cell, z * cell) for z in range(n) for y in range(n) for x in range(n)]
    indices: list[int] = []
    for z in range(dim):
        for y in range(dim):
            for x in range(dim):
                v0, v1, v2, v3 = gi(x, y, z), gi(x + 1, y, z), gi(x + 1, y, z + 1), gi(x, y, z + 1)
                v4, v5, v6, v7 = gi(x, y + 1, z), gi(x + 1, y + 1, z), gi(x + 1, y + 1, z + 1), gi(x, y + 1, z + 1)
                if (x & 1) ^ (y & 1) ^ (z & 1):
                    indices += [v0, v1, v4, v3, v2, v3, v6, v1, v5, v4, v1, v6, v7, v6, v3, v4, v4, v1, v6, v3]
                else:
                    indices += [v1, v2, v5, v0, v3, v0, v7, v2, v4, v7, v0, v5, v6, v5, v2, v7, v5, v2, v7, v0]
    return vertices, indices


@register
class PickAVBDCubeScene(Scene):
    key = "pick_avbd_cube"
    ik_link_label = HAND_BODY_SUFFIX
    ik_link_offset = wp.vec3(0.0, 0.0, 0.0)  # IsaacLab targets panda_hand directly
    ik_joint_limit_weight = 0.0  # IsaacLab disables the joint-limit objective
    ik_iters = 24
    default_sequence = "pick"

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
        # Build the cube as an explicit tetrahedral mesh and hand it to
        # add_soft_mesh, which computes proper FEM lumped node masses (summing
        # to density * volume) from the tet volumes -- no manual re-lumping.
        vertices, indices = _make_cube_tet_mesh(CUBE_SIZE, CUBE_DIM)
        # Local origin is a corner; offset so the cube is centered on CUBE_POS.
        origin = np.asarray(CUBE_POS, dtype=np.float64) - 0.5 * CUBE_SIZE
        builder.add_soft_mesh(
            pos=wp.vec3(*origin),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=CUBE_DENSITY,
            k_mu=CUBE_K_MU,
            k_lambda=CUBE_K_LAMBDA,
            k_damp=CUBE_K_DAMP,
            particle_radius=CUBE_PARTICLE_RADIUS,
            label="cube",
        )

    # -- solver-dependent physics overrides -------------------------------
    def model_materials(self, solver_key):
        # AVBD uses IsaacLab's pick_avbd_cube MODEL_CFG; other solvers keep
        # their strategy defaults for now.
        return dict(_AVBD_MODEL_MATERIALS) if solver_key == "avbd" else {}

    def solver_overrides(self, solver_key):
        # IsaacLab pick_avbd_cube VBDSolverCfg self-contact params (with
        # self-contact enabled on the cube).
        return dict(_AVBD_SOLVER_OVERRIDES) if solver_key == "avbd" else {}

    def robot_gains(self, solver_key):
        # IsaacLab pick_avbd_cube FRANKA_PANDA_AVBD_CFG: arm damping 0.1.
        gains = {"finger_stiffness": 1.0e5, "finger_damping": 1.0} if solver_key == "avbd" else {}  # at this gain, watertight collision drop the cube but old one doesn't
        # gains = {"finger_stiffness": 1.0e4, "finger_damping": 1.0} if solver_key == "avbd" else {}
        return gains

    # -- task -------------------------------------------------------------
    def home_pose(self):
        return np.array(HOME_POS, dtype=np.float64), np.array(HOME_QUAT, dtype=np.float64)

    def sequences(self, home_pos, home_quat):
        from ..controllers.base import Keyframe
        from ..controllers.sequences import KeyframeSequence
        from ..robots import GRIP_CLOSE, GRIP_OPEN

        home = np.asarray(home_pos)
        q = np.asarray(home_quat)
        # World point the fingertips aim for: the SETTLED cube's Z-center (see
        # CUBE_SETTLED_Z) with the spawn XY.
        tip_target = np.array([CUBE_POS[0], CUBE_POS[1], CUBE_SETTLED_Z], dtype=np.float64)
        # IK targets the fr3_hand wrist frame, so pull the hand-frame target back by
        # the tool offset rotated into world frame
        # (fingertip = hand_origin + R(q)*(0,0,FINGERTIP_OFFSET)).
        grasp = tip_target - _quat_rotate(q, (0.0, 0.0, FINGERTIP_OFFSET))
        return {
            # Scripted pick: hover -> descend to cube -> grasp -> lift -> hold.
            "pick": KeyframeSequence(
                [
                    Keyframe(0.8, home, q, GRIP_OPEN),  # hover at home, gripper open
                    Keyframe(1.2, grasp, q, GRIP_OPEN),  # descend to the cube center
                    Keyframe(0.8, grasp, q, GRIP_CLOSE),  # close the gripper
                    Keyframe(1.2, home, q, GRIP_CLOSE),  # lift back to home
                    # Keyframe(2.0, home, q, GRIP_CLOSE),  # hold
                    Keyframe(2.0, home, q, GRIP_OPEN),  # open the gripper
                ]
            ),
            # Hold the home pose, gripper open (debug / settle).
            "hold": KeyframeSequence([Keyframe(5.0, home, q, GRIP_OPEN)]),
        }

    # -- presentation -----------------------------------------------------
    def camera(self):
        """Initial GL camera as ``(pos, pitch, yaw, look_at)``.

        ``pos`` [m] and ``look_at`` [m] are world-space points. ``pitch`` and
        ``yaw`` are degrees in the Z-up spherical convention of
        :meth:`newton.viewer.Camera.get_front`: yaw is the XY-plane azimuth from
        +X toward +Y, pitch is elevation above the horizontal (negative looks
        down); ``front = (cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))``.

        The runner applies ``look_at`` after ``pitch``/``yaw`` (see
        :class:`~newton.exp.runner.Experiment`), so when ``look_at`` is given it
        recomputes the orientation from ``look_at - pos`` and the ``pitch``/``yaw``
        values here act only as a fallback for ``look_at = None``.
        """
        return (wp.vec3(1.2, -1.0, 0.2), -5.0, 130.0, wp.vec3(0.3, 0.0, 0.2))
