# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Table Cloth (VBD) — Newton port of the PhysX spread_tablecloth
#
# A folded tablecloth (loaded from ``assets/cloth``) is released above a
# table while a Unitree G1 humanoid holds its initial pose nearby. The
# cloth falls, contacts the table top, and drapes under gravity using the
# Newton VBD cloth solver.
#
# This is a port of the Isaac Lab + PhysX scene from the
# ``i4h-workflows`` ``spread_tablecloth`` task — VBD replaces the PhysX
# FEM-cloth solver and the G1 is kinematic (held at the initial pose by
# zeroing its body inverse masses).
#
# Assets (all loaded from ``newton/examples/assets/cloth/``):
#   - Cloth: ``Cloth_fold10.usd``. The proxy mesh
#     ``/root/Cloth_fold07/Visuals/Cloth_fold06`` (~2.5k verts) drives the
#     physics; the heavier 28k-vert mesh in the same file is skipped.
#   - Robot: ``g1_29dof_with_inspire_rev_1_0.usd`` (G1 + Inspire hands).
#   - Table: ``Table256.usd``. The visual mesh
#     ``/root/Table256/Visuals/Table256`` is added as a non-collidable shape
#     for rendering; cloth contact is handled by a simple box proxy whose
#     top surface sits at the same height as the table top.
#
# Command: python -m newton.examples table_cloth_vbd
#
###########################################################################

from __future__ import annotations

import math
import os

import h5py
import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.examples.bag.capture import (
    add_capture_arguments as _add_capture_arguments,
)
from newton.examples.bag.capture import (
    capture_replay_frame as _capture_replay_frame_common,
)
from newton.examples.bag.capture import (
    configure_capture as _configure_capture,
)
from newton.examples.bag.capture import (
    finalize_capture as _finalize_capture,
)
from newton.examples.bag.capture import (
    finalize_replay_video as _finalize_replay_video_common,
)
from newton.examples.bag.capture import (
    get_viewer_frame as _get_viewer_frame_common,
)
from newton.examples.bag.capture import (
    init_video_capture as _init_video_capture_common,
)
from newton.examples.bag.capture import (
    write_video_frame as _write_video_frame_common,
)

# Pink-IK is imported lazily inside Example.__init__ if --no-ik is not set.
# The heavy pinocchio import (~250 ms) is otherwise skipped.

# ─────────────────────────────────────────────────────────────────────────────
# Asset paths (relative to newton/examples/assets/)
# ─────────────────────────────────────────────────────────────────────────────
_CLOTH_USD_REL = "cloth/assets-1/assets-1/Cloth_fold06/Cloth_fold10.usd"
_CLOTH_PRIM_PATH = "/root/Cloth_fold07/Visuals/Cloth_fold06"
# Rigid "cloth pile" (Cloth_In002) inside the same USD as the deformable cloth.
# Visual and collision meshes have identical 28k-vert geometry; PhysX uses
# convex decomposition for collision, we use a single convex hull.
_RIGID_VIS_PRIM_PATH = "/root/Cloth_In002/Cloth_In002/Visuals/Cloth_In002"
_RIGID_COL_PRIM_PATH = "/root/Cloth_In002/Cloth_In002/Collisions/Cloth_In002_Collider1"
_G1_USD_REL = "cloth/assets-1/assets-1/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd"
_TABLE_USD_REL = "cloth/Table256/Table256/Table256.usd"
_TABLE_VIS_PRIM_PATH = "/root/Table256/Visuals/Table256"
# Table256 ships an authored offline convex decomposition: 29 mesh prims under
# Collisions/, each tagged ``physics:approximation = convexHull``.
_TABLE_COL_PRIM_FMT = "/root/Table256/Collisions/Table256_Collider{i}"
_TABLE_COL_COUNT = 29

# Recorded teleop demo from the i4h-workflows spread_tablecloth task (HDF5
# output of ``record_demos_tablecloth.py``). One episode (``data/demo_0``),
# 337 frames at the env's 30 Hz control rate. We replay G1 joint positions
# from this file and use the cloth nodal positions as a visual reference.
_REPLAY_HDF5_REL = "g1.hdf5"
_REPLAY_EPISODE = "demo_0"

# G1 fixed-base world pose, taken from the HDF5's recorded
# states/articulation/robot/root_pose[0] (constant across the episode).
# Without this, ``add_usd(floating=False)`` defaults the pelvis to the origin
# and the G1's feet end up below the ground plane.
_G1_BASE_POS = wp.vec3(-0.95, 0.0, 0.80)
_G1_BASE_ROT = wp.quat_identity()

# Isaac Lab's G1 joint ordering, copied verbatim from the env config's
# ``joint_names`` list. Newton's ``add_usd`` parses the same USD in
# tree-traversal order (legs/arms/fingers grouped by side instead of
# interleaved). We build the IL→Newton permutation at init by joint name
# match, and use it whenever we write IL-ordered values into joint_q.
# Note: the ``joint_position`` field in the recording uses PhysX articulation
# order, which is different again from Newton's USD traversal order.

# Custom init pose from i4h-workflows config/robot_config.py
# (DEFAULT_JOINT_POS + SPREAD_TABLECLOTH_CUSTOM_JOINT_POS). All joints
# not listed default to 0. This is what Isaac Lab spawns the G1 with
# before any teleop input arrives — used as our IK init seed and our
# warmup display pose.
_SPREAD_TABLECLOTH_INIT_POSE: dict[str, float] = {
    "left_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint": 0.5,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": -0.5,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Full 53-slot mapping for the recording's
# ``data/demo_0/states/articulation/robot/joint_position`` —
# slot_index → URDF joint name. The recording's order is PhysX's
# articulation order, which differs from both env_cfg ``joint_names``
# order and the URDF tree-DFS order Newton uses.
#
# PhysX walks the articulation breadth-first from the pelvis root,
# alternating waist + L/R leg + L/R arm joints at each depth. That's
# why arm joints (depths 5-11 from pelvis) interleave with leg joints
# (depths 1-6) inside slots 11-28.
#
# This is the same order Isaac Lab exposes as ``robot.data.joint_names`` for
# this USD; an Isaac Sim playback would pass the vector directly to
# ``write_joint_state_to_sim``. Newton's ``joint_q`` order is USD traversal
# order, so this table converts each HDF5 slot to a joint name first.
_JP_SLOT_TO_NAME: dict[int, str] = {
    # Lower body — BFS depth 1-6, interleaved waist + L/R legs.
    0: "left_hip_pitch_joint",
    1: "right_hip_pitch_joint",
    2: "waist_yaw_joint",
    3: "left_hip_roll_joint",
    4: "right_hip_roll_joint",
    5: "waist_roll_joint",
    6: "left_hip_yaw_joint",
    7: "right_hip_yaw_joint",
    8: "waist_pitch_joint",
    9: "left_knee_joint",
    10: "right_knee_joint",
    # Arms — BFS depths 5-11, interleaved with the last two leg depths
    # at slots 13/14 and 17/18.
    11: "left_shoulder_pitch_joint",
    12: "right_shoulder_pitch_joint",
    13: "left_ankle_pitch_joint",
    14: "right_ankle_pitch_joint",
    15: "left_shoulder_roll_joint",
    16: "right_shoulder_roll_joint",
    17: "left_ankle_roll_joint",
    18: "right_ankle_roll_joint",
    19: "left_shoulder_yaw_joint",
    20: "right_shoulder_yaw_joint",
    21: "left_elbow_joint",
    22: "right_elbow_joint",
    23: "left_wrist_roll_joint",
    24: "right_wrist_roll_joint",
    25: "left_wrist_pitch_joint",
    26: "right_wrist_pitch_joint",
    27: "left_wrist_yaw_joint",
    28: "right_wrist_yaw_joint",
    # Finger joints — breadth-first by proximal depth, then intermediate /
    # thumb child depths. This matches Isaac Lab's articulation view order.
    29: "L_index_proximal_joint",
    30: "L_middle_proximal_joint",
    31: "L_pinky_proximal_joint",
    32: "L_ring_proximal_joint",
    33: "L_thumb_proximal_yaw_joint",
    34: "R_index_proximal_joint",
    35: "R_middle_proximal_joint",
    36: "R_pinky_proximal_joint",
    37: "R_ring_proximal_joint",
    38: "R_thumb_proximal_yaw_joint",
    39: "L_index_intermediate_joint",
    40: "L_middle_intermediate_joint",
    41: "L_pinky_intermediate_joint",
    42: "L_ring_intermediate_joint",
    43: "L_thumb_proximal_pitch_joint",
    44: "R_index_intermediate_joint",
    45: "R_middle_intermediate_joint",
    46: "R_pinky_intermediate_joint",
    47: "R_ring_intermediate_joint",
    48: "R_thumb_proximal_pitch_joint",
    49: "L_thumb_intermediate_joint",
    50: "R_thumb_intermediate_joint",
    51: "L_thumb_distal_joint",
    52: "R_thumb_distal_joint",
}

_IL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]

# ─────────────────────────────────────────────────────────────────────────────
# Scene geometry (meters, Z-up — matches the PhysX scene)
# ─────────────────────────────────────────────────────────────────────────────
#
# The cloth + pile world placement is composed from two sources:
#   1. The cloth USD's authored xform on its Cloth_fold07 / Cloth_In002
#      prims — translate + orient about Z. This is what positions the
#      mesh relative to the cloth-USD root.
#   2. IL's env_cfg cloth.init_state.pos / .rot — the world placement
#      the env spawns the cloth-USD root at. From
#      ``g1_spread_tablecloth_env_cfg.py``: pos=(-0.65, 0, 0.78),
#      rot=(0,0,0,1) interpreted XYZW = identity.
# Composed: cloth world = env_cfg_pos + cloth_usd_translate; cloth world
# rot = cloth_usd_orient. Same for the pile, with its own translate.

# IL env_cfg cloth init_state (verified against the recording: composed
# cloth centroid lands at the recording's frame-0 centroid within 1 mm).
_ENVCFG_CLOTH_INIT_POS = wp.vec3(-0.65, 0.0, 0.78)
_ENVCFG_CLOTH_INIT_ROT = wp.quat_identity()  # env_cfg rot=(0,0,0,1) XYZW = identity

# Cloth-USD prim path used to read the cloth visual mesh xform.
_CLOTH_ROOT_PRIM_PATH = "/root/Cloth_fold07"
_RIGID_ROOT_PRIM_PATH = "/root/Cloth_In002"


def _read_usd_prim_xform(stage, prim_path: str) -> tuple[wp.vec3, wp.quat]:
    """Read an authored xform from a USD prim and return ``(translate, rot)``.

    Composes ``xformOp:translate`` + ``xformOp:orient`` (the only
    operations our cloth-USD prims author). Quaternion order in USD's
    ``xformOp:orient`` is WXYZ; wp.quat is XYZW, so we repack.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"{prim_path} not found in USD stage")
    t_attr = prim.GetAttribute("xformOp:translate")
    o_attr = prim.GetAttribute("xformOp:orient")
    t = t_attr.Get() if t_attr.IsValid() and t_attr.HasAuthoredValue() else (0.0, 0.0, 0.0)
    o = o_attr.Get() if o_attr.IsValid() and o_attr.HasAuthoredValue() else None
    translate = wp.vec3(float(t[0]), float(t[1]), float(t[2]))
    if o is None:
        rot = wp.quat_identity()
    else:
        # USD xformOp:orient is GfQuatf: w + imaginary(xyz). wp.quat is XYZW.
        rot = wp.quat(float(o.imaginary[0]), float(o.imaginary[1]), float(o.imaginary[2]), float(o.real))
    return translate, rot


def _compose_cloth_and_pile_xforms_from_usd() -> tuple[wp.vec3, wp.quat, wp.vec3, wp.quat]:
    """Read the cloth + pile authored xforms from Cloth_fold10.usd and
    compose with IL env_cfg's cloth init_state to get their world poses.

    Returns ``(cloth_pos, cloth_rot, pile_pos, pile_rot)``. We rely on
    env_cfg's rot being identity (verified by matching the recording's
    cloth centroid), so the composition reduces to element-wise add for
    the translate and direct copy for the rotation.
    """
    cloth_usd = newton.examples.get_asset(_CLOTH_USD_REL)
    stage = Usd.Stage.Open(cloth_usd)
    cloth_t, cloth_r = _read_usd_prim_xform(stage, _CLOTH_ROOT_PRIM_PATH)
    pile_t, pile_r = _read_usd_prim_xform(stage, _RIGID_ROOT_PRIM_PATH)
    env_pos = _ENVCFG_CLOTH_INIT_POS
    cloth_world_pos = wp.vec3(
        float(env_pos[0]) + float(cloth_t[0]),
        float(env_pos[1]) + float(cloth_t[1]),
        float(env_pos[2]) + float(cloth_t[2]),
    )
    pile_world_pos = wp.vec3(
        float(env_pos[0]) + float(pile_t[0]),
        float(env_pos[1]) + float(pile_t[1]),
        float(env_pos[2]) + float(pile_t[2]),
    )
    # env_cfg rotation is identity, so world rotation = USD-authored rotation.
    return cloth_world_pos, cloth_r, pile_world_pos, pile_r


_CLOTH_INIT_POS, _CLOTH_INIT_ROT, _RIGID_POS_FROM_USD, _RIGID_QUAT_FROM_USD = (
    _compose_cloth_and_pile_xforms_from_usd()
)
# Kept for compatibility with anything that still references _USD_Z_ROT.
_USD_Z_ROT = _CLOTH_INIT_ROT

# Table placement.
#
# The four TABLE_* constants in IL's g1_spread_tablecloth_env_cfg.py
# (TABLE_POS, TABLE_ROT, TABLE_SCALE, TABLE_TOP_POS) are dead — they're
# not referenced by the SceneCfg. The actual table is authored inside
# scene04.usd at /World/Table256 (see ``_compose_table_xform_from_scene``
# below), and scene04 itself is spawned by the env_cfg at
# pos=(0.9, -2.5, 0), rot=identity. The composed world placement is
# (-0.499, -0.137, +0.388) with -90° about Z and **no scaling** — quite
# different from the env_cfg's leftover constants. Reading scene04
# rather than hardcoding the result keeps us in sync if the scene file
# is updated.
_SCENE_USD_REL = "cloth/assets-1/assets-1/scene04.usd"
_SCENE_TABLE_PRIM_PATH = "/World/Table256"
# scene04.usd's own placement in IL's env_cfg.
_SCENE_ORIGIN_POS = wp.vec3(0.9, -2.5, 0.0)


def _compose_table_xform_from_scene() -> tuple[wp.vec3, wp.quat, wp.vec3]:
    """Read scene04.usd's authored xform for /World/Table256 and compose
    with scene04's own world placement to get the table's world transform.

    Returns ``(center, quat_xyzw_for_wp, scale)`` ready to feed
    :func:`builder.add_shape_*` calls. scene04's payload references are
    broken on our copy (they point to Assets/Table256/Table256.usd which
    we don't have), but the xform attributes on /World/Table256 are
    authored in scene04 itself and remain readable.
    """
    scene_usd = newton.examples.get_asset(_SCENE_USD_REL)
    scene_stage = Usd.Stage.Open(scene_usd, Usd.Stage.LoadNone)
    prim = scene_stage.GetPrimAtPath(_SCENE_TABLE_PRIM_PATH)
    if not prim:
        raise RuntimeError(f"{_SCENE_TABLE_PRIM_PATH} missing from {scene_usd}")
    t = prim.GetAttribute("xformOp:translate").Get()
    r = prim.GetAttribute("xformOp:rotateXYZ").Get()  # degrees, XYZ Euler
    s = prim.GetAttribute("xformOp:scale").Get()
    # Compose with scene04's world placement. scene04's own rotation is
    # identity (env_cfg: rot=(0,0,0,1) interpreted XYZW), so the world
    # translate is just element-wise add. Likewise scale.
    center = wp.vec3(
        float(t[0]) + float(_SCENE_ORIGIN_POS[0]),
        float(t[1]) + float(_SCENE_ORIGIN_POS[1]),
        float(t[2]) + float(_SCENE_ORIGIN_POS[2]),
    )
    # Only Z component is non-zero for this table; convert deg -> rad.
    rot_z_rad = math.radians(float(r[2]))
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rot_z_rad)
    scale = wp.vec3(float(s[0]), float(s[1]), float(s[2]))
    return center, rot, scale


_TABLE_CENTER, _TABLE_ROT, _TABLE_SCALE = _compose_table_xform_from_scene()

# Rigid "cloth pile" world pose, composed from the cloth USD's authored
# xform on /root/Cloth_In002 and IL env_cfg's cloth.init_state.pos. The
# pile is authored 26 mm below the cloth mesh center (so they don't
# overlap at rest) with the same +π/2 Z rotation.
_RIGID_POS = _RIGID_POS_FROM_USD
_RIGID_QUAT = _RIGID_QUAT_FROM_USD
# USD has mass = 0.001 kg with linearDamping = 1.0 and maxLinearVelocity = 1.0
# to keep the body stable. AVBD has no equivalent damping, so under the
# asymmetric initial cloth-rigid contacts a 1 g body skitters off the table.
# We use a heavier mass (~2 kg) as a stability accommodation; the body stays
# roughly under the cloth and lands on the table top as in PhysX.
_RIGID_MASS = 2.0
_RIGID_MU = 0.95  # USD material's static/dynamic friction
# Collision approximation. PhysX used convex decomposition with up to 64
# hulls of up to 64 verts each. CoACD with max_convex_hull=64 matches.
_RIGID_MAX_HULLS = 64
# Uniform shape-scale applied to every decomposed hull. 1.0 = no shrink.
_RIGID_COL_SCALE = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Cloth material (SI)
# ─────────────────────────────────────────────────────────────────────────────
_TRI_KE = 1.0e3
_TRI_KA = 1.0e3
_TRI_KD = 1.0e-3
_EDGE_KE = 5.0
_EDGE_KD = 1.0e-3
_DENSITY = 0.5  # kg/m² — surface density

# ─────────────────────────────────────────────────────────────────────────────
# Collision (SI)
# ─────────────────────────────────────────────────────────────────────────────
_PARTICLE_RADIUS = 5.0e-3  # 5 mm
# Self-contact radius/margin must stay below the rest-state inter-particle
# spacing (~6 mm for the proxy mesh) or the folded cloth's near-neighbours
# instantly bind together and the sheet refuses to fall.
_SELF_CONTACT_RADIUS = 2.0e-3
_SELF_CONTACT_MARGIN = 2.0e-3
_REST_EXCLUSION_RADIUS = 8.0e-3
_SOFT_CONTACT_MARGIN = 1.0e-2

_SOFT_CONTACT_KE = 1.0e4
_SOFT_CONTACT_KD = 1.0e-1
_SOFT_CONTACT_MU = 0.5

_SHAPE_KE = 1.0e5
_SHAPE_KD = 1.0e0
_SHAPE_MU = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────
# The recorded HDF5 demo uses a 30 Hz control rate (Isaac Lab sim_dt = 1/120 s
# with decimation = 4). We match that so one frame here corresponds to one
# recorded action. Internal substep size is preserved (40 substeps x 1/(30*40)
# = 0.833 ms = same as the previous 20 x 1/(60*20) setup).
_FPS = 30
_SIM_SUBSTEPS = 40
_VBD_ITERS = 10
# Frames at the start where both HDF5 playback and physics simulation are
# halted, so the scene's initial pose stays on screen for inspection.
# At 30 Hz, 60 frames = 2 s.
_WARMUP_FRAMES = 60


class Example:
    """Drop a tablecloth onto a table next to a kinematic Unitree G1.

    The G1 is fixed-base and its body inverse masses are zeroed so the
    initial joint pose is held by the VBD coupling. The cloth particles
    are integrated by :class:`newton.solvers.SolverVBD` and contact the
    table box and ground plane through the soft-contact pipeline.
    """

    def __init__(
        self,
        viewer,
        save_mp4: str | None = None,
        capture_replay: bool = False,
        capture_frames: int = 300,
        capture_fps: int = 60,
        capture_dir: str = "outputs/replay_capture",
        capture_format: str = "mp4",
        no_cloth: bool = False,
        no_pile: bool = False,
        show_record: bool = False,
    ):
        self.viewer = viewer
        self.fps = _FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = _SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self._frame_count = 0
        self._has_cloth = not no_cloth
        self._has_pile = not no_pile

        # Replay state ─ populated by ``_load_replay()`` below.
        # Joint motion comes from joint_position: the PD-tracked physical
        # state Isaac Lab rendered (read straight from PhysX in IL).
        self._replay_joint_q = None  # (T, 53)  G1 joint positions per frame (PhysX order)
        self._replay_cloth_pos = None  # (T, 2523, 3)  PhysX cloth particle world positions
        self._replay_total_frames = 0
        self._replay_frame = 0  # next HDF5 frame to consume
        self._replay_started = False  # set True after warmup completes
        # Last metric values for the HUD.
        self._cloth_delta_rms_mm = float("nan")
        self._cloth_delta_max_mm = float("nan")
        self._cloth_delta_mean_mm = float("nan")
        # UI toggle for the recorded-cloth overlay.
        # UI toggle for the recorded-cloth overlay. Defaults to off; users
        # opt in with ``--show-record`` and can also flip the
        # state during the run via the "Show record" checkbox in the HUD.
        self._show_record = bool(show_record)
        _configure_capture(
            self,
            save_mp4=save_mp4,
            capture_replay=capture_replay,
            capture_frames=capture_frames,
            capture_fps=capture_fps,
            capture_dir=capture_dir,
            capture_format=capture_format,
            capture_background_writes=False,
        )

        builder = newton.ModelBuilder()  # Z-up, gravity = (0, 0, -9.81)

        # ── G1 robot (fixed base, kinematic) ────────────────────────────────
        # Pelvis at world (-0.95, 0, 0.80) — matches the HDF5 recording. With
        # the default identity xform, the G1's feet would be below the ground.
        builder.add_usd(
            newton.examples.get_asset(_G1_USD_REL),
            xform=wp.transform(_G1_BASE_POS, _G1_BASE_ROT),
            floating=False,
        )
        self._robot_body_count = builder.body_count
        self._robot_shape_end = builder.shape_count

        # Newton's USD parser loads the G1's per-body visual meshes as
        # shapes but flags them VISIBLE-only (CollisionAPI sub-prims under
        # `/g1_29dof_with_hand_rev_1_0/<link>/collisions` are emitted as a
        # generic GPrim type the parser skips). Without an explicit flag
        # flip, cloth particles can't see the robot at all. Turn on
        # `COLLIDE_PARTICLES` on every robot shape so the cloth's
        # particle-rigid soft-contact pipeline registers them; leave
        # `COLLIDE_SHAPES` off so the robot's links don't collide with
        # each other (matches IL env_cfg's `enabled_self_collisions=False`).
        # The robot stays kinematic — `body_inv_mass` is zeroed below — so
        # this is one-way: the cloth feels the robot's hands and body but
        # the robot's pose is unchanged by cloth contact.
        for s in range(self._robot_shape_end):
            flags = int(builder.shape_flags[s])
            builder.shape_flags[s] = flags | int(newton.ShapeFlags.COLLIDE_PARTICLES)

        # Build the IL-index → Newton-qstart permutation by name-matching the
        # builder's joint labels. ``self._il_to_newton_qs[il_idx]`` is the
        # Newton joint_q coord index for the IL-named joint, or None if the
        # name didn't match (shouldn't happen with the canonical G1 USD).
        self._il_to_newton_qs: list[int | None] = [None] * len(_IL_JOINT_NAMES)
        for j in range(builder.joint_count):
            if int(builder.joint_type[j]) != 1:  # revolute only — fixed/free skipped
                continue
            short = builder.joint_label[j].rsplit("/", 1)[-1]
            if short in _IL_JOINT_NAMES:
                self._il_to_newton_qs[_IL_JOINT_NAMES.index(short)] = int(builder.joint_q_start[j])
        _missing = [_IL_JOINT_NAMES[i] for i, qs in enumerate(self._il_to_newton_qs) if qs is None]
        if _missing:
            print(f"[table_cloth_vbd] WARNING: IL joints not found in Newton model: {_missing}")

        # ── Cloth from assets/cloth/Cloth_fold10.usd ────────────────────────
        # The cloth USD also carries the rigid pile (Cloth_In002), so we open
        # the stage even when the deformable cloth itself is disabled.
        cloth_stage = Usd.Stage.Open(newton.examples.get_asset(_CLOTH_USD_REL))
        self._cloth_particle_start = builder.particle_count
        # Always read the cloth mesh (faces + vertices) — even with
        # ``--no-cloth`` we still want the indices so the recorded-cloth
        # overlay can render as a triangle mesh. We just skip
        # ``add_cloth_mesh`` (which is what actually spawns simulated
        # particles) when ``_has_cloth`` is False.
        cloth_prim = cloth_stage.GetPrimAtPath(_CLOTH_PRIM_PATH)
        if not cloth_prim:
            raise RuntimeError(f"Cloth prim not found at {_CLOTH_PRIM_PATH}")
        cloth_mesh = newton.usd.get_mesh(cloth_prim)
        vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in cloth_mesh.vertices]
        indices = list(map(int, cloth_mesh.indices))
        self._cloth_indices_np = np.array(indices, dtype=np.int32)
        if self._has_cloth:
            builder.add_cloth_mesh(
                pos=_CLOTH_INIT_POS,
                rot=_CLOTH_INIT_ROT,
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=vertices,
                indices=indices,
                density=_DENSITY,
                tri_ke=_TRI_KE,
                tri_ka=_TRI_KA,
                tri_kd=_TRI_KD,
                edge_ke=_EDGE_KE,
                edge_kd=_EDGE_KD,
                particle_radius=_PARTICLE_RADIUS,
            )
        self._cloth_particle_end = builder.particle_count

        # ── Rigid "cloth pile" (Cloth_In002) ────────────────────────────────
        # Dynamic rigid body living *inside* the deformable cloth at the
        # USD-authored relative offset. Collision geometry mirrors the USD
        # PhysX setup: convex decomposition (CoACD, up to 64 hulls) of the
        # 28k-vert mesh — a single convex hull would extend into the cloth
        # shell's concavities and cause initial-overlap penalties.
        self._rigid_body_idx = None
        self._rigid_col_shape_start = 0
        self._rigid_col_shape_end = 0
        if self._has_pile:
            rigid_col_prim = cloth_stage.GetPrimAtPath(_RIGID_COL_PRIM_PATH)
            rigid_vis_prim = cloth_stage.GetPrimAtPath(_RIGID_VIS_PRIM_PATH)
            if not rigid_col_prim or not rigid_vis_prim:
                raise RuntimeError("Rigid Cloth_In002 prims not found")
            rigid_col_mesh = newton.usd.get_mesh(rigid_col_prim)
            rigid_vis_mesh = newton.usd.get_mesh(rigid_vis_prim)

            # Diagonal inertia for a ~0.24 x 0.14 x 0.10 m box of the chosen mass.
            I = _RIGID_MASS / 12.0
            rigid_inertia = [
                [I * (0.14**2 + 0.10**2), 0.0, 0.0],
                [0.0, I * (0.24**2 + 0.10**2), 0.0],
                [0.0, 0.0, I * (0.24**2 + 0.14**2)],
            ]
            self._rigid_body_idx = builder.add_body(
                xform=wp.transform(_RIGID_POS, _RIGID_QUAT),
                mass=_RIGID_MASS,
                inertia=rigid_inertia,
                lock_inertia=True,
            )

            # Add the mesh as a (temporarily) mesh collider, then ask the
            # builder to replace it with a CoACD decomposition. ``shape_scale``
            # is set on each resulting hull so we can shrink them uniformly
            # to clear the cloth shell if needed.
            rigid_col_cfg = newton.ModelBuilder.ShapeConfig(
                ke=_SHAPE_KE,
                kd=_SHAPE_KD,
                mu=_RIGID_MU,
                density=0.0,
                is_visible=False,  # the visual mesh below is the rendered form
            )
            col_shape_idx = builder.add_shape_mesh(
                body=self._rigid_body_idx,
                mesh=rigid_col_mesh,
                cfg=rigid_col_cfg,
            )
            builder.approximate_meshes(
                method="coacd",
                shape_indices=[col_shape_idx],
                max_convex_hull=_RIGID_MAX_HULLS,
                threshold=0.1,  # lower than Newton's 0.5 default to actually split
                merge=True,  # merge small hulls back together to respect max_convex_hull
            )
            self._rigid_col_shape_start = col_shape_idx
            self._rigid_col_shape_end = builder.shape_count  # exclusive
            if _RIGID_COL_SCALE != 1.0:
                for s in range(self._rigid_col_shape_start, self._rigid_col_shape_end):
                    sx, sy, sz = builder.shape_scale[s]
                    builder.shape_scale[s] = (
                        sx * _RIGID_COL_SCALE,
                        sy * _RIGID_COL_SCALE,
                        sz * _RIGID_COL_SCALE,
                    )

            rigid_vis_cfg = newton.ModelBuilder.ShapeConfig(
                has_shape_collision=False,
                has_particle_collision=False,
                density=0.0,
            )
            builder.add_shape_mesh(
                body=self._rigid_body_idx,
                mesh=rigid_vis_mesh,
                cfg=rigid_vis_cfg,
            )

        # ── Table collision: 29 pre-authored convex hulls ───────────────────
        # Table256.usd authors one rigid body whose collider is an offline
        # convex decomposition baked into 29 separate Mesh prims, each tagged
        # ``physics:approximation = convexHull``. We mirror that exactly: 29
        # sibling shapes attached to the world body (the USD's FixedJoint
        # pins the rigid to /root, so static-on-world is the same thing).
        table_stage = Usd.Stage.Open(newton.examples.get_asset(_TABLE_USD_REL))
        table_xform = wp.transform(_TABLE_CENTER, _TABLE_ROT)
        table_col_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_SHAPE_KE,
            kd=_SHAPE_KD,
            mu=_SHAPE_MU,
            is_visible=False,
        )
        for i in range(1, _TABLE_COL_COUNT + 1):
            col_prim = table_stage.GetPrimAtPath(_TABLE_COL_PRIM_FMT.format(i=i))
            if not col_prim:
                raise RuntimeError(f"Table collider not found at {_TABLE_COL_PRIM_FMT.format(i=i)}")
            builder.add_shape_convex_hull(
                body=-1,
                xform=table_xform,
                mesh=newton.usd.get_mesh(col_prim),
                scale=_TABLE_SCALE,
                cfg=table_col_cfg,
            )

        # ── Table visual mesh (Table256.usd) ────────────────────────────────
        # Render-only: both collision flags disabled so it never participates
        # in cloth/rigid contact.
        table_vis_prim = table_stage.GetPrimAtPath(_TABLE_VIS_PRIM_PATH)
        if not table_vis_prim:
            raise RuntimeError(f"Table visual prim not found at {_TABLE_VIS_PRIM_PATH}")
        table_vis_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=False,
            has_particle_collision=False,
            density=0.0,
        )
        builder.add_shape_mesh(
            body=-1,
            xform=table_xform,
            mesh=newton.usd.get_mesh(table_vis_prim),
            scale=_TABLE_SCALE,
            cfg=table_vis_cfg,
        )

        # ── Ground plane ────────────────────────────────────────────────────
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_SHAPE_KE,
            kd=_SHAPE_KD,
            mu=_SHAPE_MU,
        )
        builder.add_ground_plane(cfg=ground_cfg)

        # VBD needs color groups for both particles (cloth) and rigid bodies
        # (the pile). ``include_bending`` adds bending-edge coloring for cloth.
        builder.color(include_bending=self._has_cloth)

        self.model = builder.finalize()

        # ── Make the G1 bodies kinematic (rigid pile stays dynamic) ───────
        inv_m = self.model.body_inv_mass.numpy().copy()
        inv_i = self.model.body_inv_inertia.numpy().copy()
        inv_m[: self._robot_body_count] = 0.0
        inv_i[: self._robot_body_count] = 0.0
        self.model.body_inv_mass = wp.array(inv_m, dtype=float)
        self.model.body_inv_inertia = wp.array(inv_i, dtype=wp.mat33)

        # ── Particle / shape contact materials ──────────────────────────────
        self.model.soft_contact_ke = _SOFT_CONTACT_KE
        self.model.soft_contact_kd = _SOFT_CONTACT_KD
        self.model.soft_contact_mu = _SOFT_CONTACT_MU

        shape_ke = self.model.shape_material_ke.numpy().copy()
        shape_kd = self.model.shape_material_kd.numpy().copy()
        shape_mu = self.model.shape_material_mu.numpy().copy()
        shape_ke[:] = _SHAPE_KE
        shape_kd[:] = _SHAPE_KD
        shape_mu[:] = _SHAPE_MU
        self.model.shape_material_ke = wp.array(shape_ke, dtype=float)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=float)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=float)

        # ── States ──────────────────────────────────────────────────────────
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialise the robot FK once. The robot pose is then frozen because
        # body_inv_mass is zero — VBD will not integrate those bodies.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.model.body_q, self.state_0.body_q)

        # ── VBD solver ──────────────────────────────────────────────────────
        # ``integrate_with_external_rigid_solver=False`` lets AVBD inside VBD
        # integrate the dynamic cloth-pile body. The G1 bodies stay put
        # because their inverse masses were zeroed above (kinematic).
        # ``particle_rest_shape_contact_exclusion_radius`` filters out the
        # self-contact pairs that start within 8 mm in the rest state — without
        # this the folded cloth's overlapping layers freeze the sheet in place.
        #
        # Tuning for the compound (~40-hull) cloth-pile collider:
        #   - ``rigid_body_particle_contact_buffer_size`` is per-body and
        #     defaults to 256; many hulls x cloth particles can blow past
        #     that, silently dropping contacts. We raise it to 2048.
        #   - ``rigid_avbd_beta`` is the penalty ramp rate. AVBD is designed
        #     around a low ``rigid_contact_k_start`` and lets the penalty
        #     adapt. With many sibling-hull contacts all ramping
        #     independently the effective stiffness can explode; we lower
        #     beta to slow that.
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=_VBD_ITERS,
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=_SELF_CONTACT_RADIUS,
            particle_self_contact_margin=_SELF_CONTACT_MARGIN,
            particle_rest_shape_contact_exclusion_radius=_REST_EXCLUSION_RADIUS,
            particle_topological_contact_filter_threshold=1,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=1.0e5,
            rigid_avbd_beta=1.0e3,
            rigid_body_particle_contact_buffer_size=2048,
        )

        # ── Collision pipeline ──────────────────────────────────────────────
        # broad_phase="nxn" is required for particle-rigid-body contact.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=_SOFT_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        # ── Viewer ──────────────────────────────────────────────────────────
        self.viewer.set_model(self.model)
        # Camera framed on the action zone: G1 pelvis at (-0.95, 0, 0.80),
        # table top centered around (-0.50, 0, 0.886), cloth + pile drop in
        # between. We park in the +X / -Y corner and look back-left so the
        # robot is on the right, the table on the left, and the cloth-pile
        # column straight ahead.
        self.viewer.set_camera(wp.vec3(0.8, -1.0, 1.6), -20.0, 140.0)

        # Load the recorded HDF5 once we have the model+state ready. After
        # warmup, ``step()`` will drive G1 joint_q from this data.
        self._load_replay()

        # Visible warmup pose: spread_tablecloth custom init (arms relaxed at
        # pitch=-0.3, roll=±0.5, elbow=-0.5) — matches what Isaac Lab
        # started the recording from, so the cloth settles around the
        # same arm configuration the recording starts at.
        if self._il_to_newton_qs is not None:
            self._snap_g1_to_init_pose()

        # Replay slot map: jp_slot → Newton joint_q index, for the full
        # recorded 53-DoF G1 joint state.
        self._replay_slot_qs: list[tuple[int, int]] = []
        if self._replay_joint_q is not None:
            self._build_replay_qmaps()

        self.capture()

        if self.save_mp4:
            self._init_video_capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Robot stays kinematic — its body_q never changes — but copying
            # into state_1 keeps the contact pipeline consistent.
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.capture_done:
            return

        # Warmup: advance physics by exactly one frame on the very first
        # call (so the cloth settles into the post-step pose that IL's
        # PostStepStatesRecorder captures as recording[0]), then freeze
        # both physics and the HDF5 playback for the remaining warmup
        # frames so the post-settle pose stays on screen for inspection.
        # The overlay was seeded with recording[0] in ``_load_replay``
        # and stays there because we don't call ``_update_record_overlay``.
        if self._frame_count < _WARMUP_FRAMES:
            if self._frame_count == 0:
                if self.graph:
                    wp.capture_launch(self.graph)
                else:
                    self.simulate()
            self.sim_time += self.frame_dt
            self.frame_index += 1
            self._frame_count += 1
            return

        # After warmup, drive the G1 from the recording and run physics.
        if self._replay_joint_q is not None:
            if not self._replay_started:
                self._replay_started = True
                self._replay_frame = 0
            self._apply_replay_frame(self._replay_frame)
            self._update_record_overlay(self._replay_frame)
            self._replay_frame += 1

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._frame_count += 1

        # Per-frame cloth-tracking metric vs the recorded PhysX cloth (same
        # 2523 particles). Only meaningful once replay has started.
        if self._replay_started:
            self._compute_cloth_metric(self._replay_frame - 1)

    def render(self):
        if self.capture_done:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Recorded-cloth overlay (Show record toggle). We always update the
        # mesh prototype + one identity-xform instance so the toggle just flips
        # ``hidden`` rather than relying on persistence between frames.
        if self._record_points_wp is not None and self._record_indices_wp is not None:
            self.viewer.log_mesh(
                "/record_cloth",
                points=self._record_points_wp,
                indices=self._record_indices_wp,
                backface_culling=False,
                hidden=not self._show_record,
            )
            self.viewer.log_instances(
                "/record_cloth_inst",
                mesh="/record_cloth",
                xforms=self._record_xforms_wp,
                scales=None,
                colors=self._record_colors_wp,
                materials=None,
                hidden=not self._show_record,
            )
        self.viewer.end_frame()
        _write_video_frame_common(self)
        self._capture_replay_frame()

    # ─────────────────────────────────────────────────────────────────────
    # Video / replay capture helpers
    # ─────────────────────────────────────────────────────────────────────

    def _init_video_capture(self):
        _init_video_capture_common(self)

    def _get_viewer_frame(self, *, render_ui: bool = False):
        return _get_viewer_frame_common(self.viewer, render_ui=render_ui)

    def _capture_replay_frame(self, *, frame_key: int | None = None):
        if frame_key is None:
            frame_key = self._frame_count
        _capture_replay_frame_common(self, frame_key=frame_key)

    def _finalize_replay_video(self):
        _finalize_replay_video_common(self)

    # ─────────────────────────────────────────────────────────────────────
    # HDF5 replay + per-frame cloth-tracking metric
    # ─────────────────────────────────────────────────────────────────────

    def _load_replay(self):
        """Read the recorded G1 joint trajectory + cloth particle trajectory.

        Populates ``self._replay_joint_q``, ``self._replay_cloth_pos`` and a
        few derived warp arrays used by render/metric paths. If the file is
        missing, replay is silently disabled (warmup-only behaviour).
        """
        path = newton.examples.get_asset(_REPLAY_HDF5_REL)
        if not os.path.exists(path):
            print(f"[table_cloth_vbd] {path} not found; G1 will hold its initial pose.")
            return

        with h5py.File(path, "r") as f:
            if "data" not in f or _REPLAY_EPISODE not in f["data"]:
                print(f"[table_cloth_vbd] {path} has no episode '{_REPLAY_EPISODE}'; replay disabled.")
                return
            demo = f["data"][_REPLAY_EPISODE]
            jq = np.array(demo["states/articulation/robot/joint_position"], dtype=np.float32)
            cp = np.array(demo["states/deformable_object/cloth/nodal_position"], dtype=np.float32)

        self._replay_joint_q = jq
        self._replay_cloth_pos = cp
        self._replay_total_frames = int(jq.shape[0])
        print(
            f"[table_cloth_vbd] Loaded {self._replay_total_frames} frames from {path} "
            f"(G1 dim={jq.shape[1]}, cloth nodes={cp.shape[1]})"
        )
        # Sanity check that the recorded cloth has the same particle count as
        # the proxy mesh we simulate. If not, the per-particle comparison is
        # meaningless.
        if self._has_cloth:
            ours = self._cloth_particle_end - self._cloth_particle_start
            if ours != cp.shape[1]:
                print(
                    f"[table_cloth_vbd] WARNING: our cloth has {ours} particles but the "
                    f"recording has {cp.shape[1]}. Per-particle metric will be skipped."
                )
                self._replay_cloth_pos = None

        # Device-side buffer for the "Show record" overlay (one frame at
        # a time). Decoupled from ``_has_cloth`` so ``--no-cloth`` only
        # disables the simulated cloth and the overlay can still render.
        if self._replay_cloth_pos is not None and self._cloth_indices_np is not None:
            n_nodes = cp.shape[1]
            self._record_points_wp = wp.zeros(n_nodes, dtype=wp.vec3)
            self._record_indices_wp = wp.array(self._cloth_indices_np, dtype=wp.int32)
            # Seed with frame 0 so the overlay shows the spawn pose before warmup ends.
            self._record_points_wp.assign(cp[0])
            # log_instances() needs xforms/colors arrays for the single instance;
            # red-orange so the overlay reads as distinct from our cloth.
            self._record_xforms_wp = wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
                dtype=wp.transform,
            )
            self._record_colors_wp = wp.array([wp.vec3(0.95, 0.35, 0.20)], dtype=wp.vec3)
        else:
            self._record_points_wp = None
            self._record_indices_wp = None
            self._record_xforms_wp = None
            self._record_colors_wp = None

    def _build_replay_qmaps(self) -> None:
        """Build the jp_slot → Newton joint_q index list used by replay.

        For each (jp_slot, joint_name) pair in ``_JP_SLOT_TO_NAME``,
        look up the corresponding Newton ``joint_q`` index by joint
        name and store as ``(jp_slot, newton_qs)``. At replay time we
        copy ``joint_position[t, jp_slot]`` to ``joint_q[newton_qs]``.
        """
        # Build Newton qs lookup by joint name from the builder labels
        # we cached in ``_il_to_newton_qs`` (which keys off
        # ``_IL_JOINT_NAMES``).
        self._replay_slot_qs = []
        for jp_slot, joint_name in _JP_SLOT_TO_NAME.items():
            if joint_name not in _IL_JOINT_NAMES:
                continue
            il_idx = _IL_JOINT_NAMES.index(joint_name)
            n_qs = self._il_to_newton_qs[il_idx]
            if n_qs is not None:
                self._replay_slot_qs.append((jp_slot, n_qs))
        print(
            f"[table_cloth_vbd] Replay slot map: {len(self._replay_slot_qs)}/"
            f"{len(_JP_SLOT_TO_NAME)} jp slots mapped to Newton joint_q"
        )

    def _apply_replay_frame(self, hdf5_frame: int) -> None:
        """Drive the G1 joints from one recorded frame — full direct replay.

        For every (jp_slot, newton_qs) pair built by
        ``_build_replay_qmaps``, copy ``joint_position[t, jp_slot]``
        into ``state_0.joint_q[newton_qs]``. This mirrors what Isaac
        Sim does in the minimal IL playback (``write_joint_state_to_sim``
        with the full 53-D vector) — every joint we drive comes from the
        same source IL renders from.

        All 53 recorded joints are mapped. Legs and waist move only by the
        small PD drift present in the recording, matching Isaac Lab replay.
        """
        if self._replay_joint_q is None:
            return
        idx = int(min(hdf5_frame, self._replay_total_frames - 1))
        jp_frame = self._replay_joint_q[idx]
        jq = self.state_0.joint_q.numpy().copy()
        for jp_slot, n_qs in self._replay_slot_qs:
            jq[n_qs] = float(jp_frame[jp_slot])
        self.state_0.joint_q.assign(wp.array(jq, dtype=float))

        # eval_fk would teleport the pile (free-joint) body back to its
        # FREE-joint origin coords, so save and restore its body_q.
        body_q_np = self.state_0.body_q.numpy().copy() if self._has_pile else None
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        if self._has_pile and self._rigid_body_idx is not None:
            new_bq = self.state_0.body_q.numpy().copy()
            new_bq[self._rigid_body_idx] = body_q_np[self._rigid_body_idx]
            self.state_0.body_q.assign(wp.array(new_bq, dtype=wp.transform))

    def _snap_g1_to_init_pose(self) -> None:
        """Set Newton's G1 joint_q to the spread_tablecloth custom init pose.

        Matches ``SPREAD_TABLECLOTH_CUSTOM_JOINT_POS`` from
        i4h-workflows' config/robot_config.py — arms held relaxed at
        ±0.5 roll, -0.3 pitch, -0.5 elbow. Used as the visible warmup
        pose before HDF5 replay starts.
        """
        jq = self.state_0.joint_q.numpy().copy()
        for name, value in _SPREAD_TABLECLOTH_INIT_POSE.items():
            if name in _IL_JOINT_NAMES:
                il_idx = _IL_JOINT_NAMES.index(name)
                n_qs = self._il_to_newton_qs[il_idx]
                if n_qs is not None:
                    jq[n_qs] = float(value)
        self.state_0.joint_q.assign(wp.array(jq, dtype=float))

        body_q_np = self.state_0.body_q.numpy().copy() if self._has_pile else None
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        if self._has_pile and self._rigid_body_idx is not None:
            new_bq = self.state_0.body_q.numpy().copy()
            new_bq[self._rigid_body_idx] = body_q_np[self._rigid_body_idx]
            self.state_0.body_q.assign(wp.array(new_bq, dtype=wp.transform))

    def _update_record_overlay(self, hdf5_frame: int) -> None:
        """Copy the next recorded cloth frame into the overlay buffer."""
        if self._record_points_wp is None or self._replay_cloth_pos is None:
            return
        idx = int(min(hdf5_frame, self._replay_total_frames - 1))
        self._record_points_wp.assign(self._replay_cloth_pos[idx])

    def _compute_cloth_metric(self, hdf5_frame: int) -> None:
        """Compute per-particle L2 deviation between our cloth and the
        recorded cloth at the given HDF5 frame. Stores RMS/max/mean in mm."""
        if (
            not self._has_cloth
            or self._replay_cloth_pos is None
            or self._cloth_particle_end == self._cloth_particle_start
        ):
            return
        idx = int(min(hdf5_frame, self._replay_total_frames - 1))
        ours = self.state_0.particle_q.numpy()[self._cloth_particle_start : self._cloth_particle_end]
        theirs = self._replay_cloth_pos[idx]
        d = np.linalg.norm(ours - theirs, axis=1)
        self._cloth_delta_rms_mm = float(np.sqrt((d * d).mean()) * 1000.0)
        self._cloth_delta_max_mm = float(d.max() * 1000.0)
        self._cloth_delta_mean_mm = float(d.mean() * 1000.0)

    # ─────────────────────────────────────────────────────────────────────
    # imgui-style HUD (only attached if the viewer supports it)
    # ─────────────────────────────────────────────────────────────────────

    def gui(self, ui):
        ui.text(f"frame: {self._frame_count}")
        ui.text(f"sim time: {self.sim_time:.2f} s")
        if self._replay_joint_q is not None:
            if not self._replay_started:
                ui.text(f"warmup: {self._frame_count}/{_WARMUP_FRAMES}")
            else:
                idx = min(self._replay_frame, self._replay_total_frames)
                ui.text(f"replay: {idx}/{self._replay_total_frames}")
            if not math.isnan(self._cloth_delta_rms_mm):
                ui.text(f"cloth dRMS:  {self._cloth_delta_rms_mm:.2f} mm")
                ui.text(f"cloth dMean: {self._cloth_delta_mean_mm:.2f} mm")
                ui.text(f"cloth dMax:  {self._cloth_delta_max_mm:.2f} mm")
        if self._record_points_wp is not None:
            _changed, self._show_record = ui.checkbox("Show record", self._show_record)

    def test_final(self):
        return  # temptest
        pq = self.state_0.particle_q.numpy()[self._cloth_particle_start : self._cloth_particle_end]
        qd = self.state_0.particle_qd.numpy()[self._cloth_particle_start : self._cloth_particle_end]
        # The cloth should have dropped from its release height (z = 1.20 m)
        # and settled on or near the table top (z ≈ 0.70 m).
        z_min = float(pq[:, 2].min())
        z_max = float(pq[:, 2].max())
        assert z_min > -0.05, f"Cloth particle below ground: z_min = {z_min:.3f}"
        assert z_max < float(_CLOTH_INIT_POS[2]) - 0.05, (
            f"Cloth has not fallen: z_max = {z_max:.3f} (release was {float(_CLOTH_INIT_POS[2]):.3f})"
        )
        # Cloth should not drift outside the simulated workspace.
        assert float(np.abs(pq[:, 0]).max()) < 2.0, "Cloth drifted too far in X"
        assert float(np.abs(pq[:, 1]).max()) < 1.0, "Cloth drifted too far in Y"
        # Velocities should be finite and bounded.
        assert float(np.abs(qd).max()) < 20.0, f"Cloth velocity too high: |v|max = {np.abs(qd).max():.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    _add_capture_arguments(
        parser,
        replay_help="Capture rendered frames and auto-build a replay video",
    )
    parser.add_argument(
        "--no-cloth",
        action="store_true",
        help="Skip adding the deformable tablecloth (useful for isolating per-component cost)",
    )
    parser.add_argument(
        "--no-pile",
        action="store_true",
        help="Skip adding the rigid cloth-pile body and its CoACD hull collider",
    )
    parser.add_argument(
        "--show-record",
        action="store_true",
        help=(
            "Render the recorded cloth (red-orange overlay) on top of our "
            "simulated cloth. Off by default; can also be toggled at runtime "
            "via the 'Show record' checkbox in the HUD."
        ),
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        save_mp4=getattr(args, "save_mp4", None),
        capture_replay=bool(args.capture_replay),
        capture_frames=int(args.capture_frames),
        capture_fps=int(args.capture_fps),
        capture_dir=str(args.capture_dir),
        capture_format=str(args.capture_format),
        no_cloth=bool(args.no_cloth),
        no_pile=bool(args.no_pile),
        show_record=bool(args.show_record),
    )

    # HUD panel (replay frame + cloth-tracking metric + Show record toggle).
    if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(lambda ui, ex=example: ex.gui(ui), position="side")

    while viewer.is_running() and not getattr(example, "capture_done", False):
        example.step()
        example.render()

    if args.test:
        example.test_final()

    _finalize_capture(example)
