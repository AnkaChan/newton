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

"""Shared constants and helpers for the table-cloth examples.

Both ``example_table_cloth_vbd`` and ``example_table_cloth_ppfcs`` port
the same Isaac Lab + PhysX ``spread_tablecloth`` scene from
``i4h-workflows``. They differ in the solver — VBD cloth + AVBD rigid
in one, PPFCS tri-shell + pinned-shell colliders in the other — but
the geometry, the asset paths, the HDF5 recording, and the world-frame
composition that places everything in the Newton scene are identical.
That shared infrastructure lives here so neither example depends on the
other.

This module exposes:

* USD / HDF5 asset paths and prim paths (``CLOTH_USD_REL`` etc.)
* World placement constants from IL's env_cfg (``G1_BASE_POS``,
  ``ENVCFG_CLOTH_INIT_POS``, ``SCENE_ORIGIN_POS``)
* Joint-ordering tables for the HDF5 recording (``IL_JOINT_NAMES``,
  ``JP_SLOT_TO_NAME``) and the spread_tablecloth init pose
  (``SPREAD_TABLECLOTH_INIT_POSE``)
* ``compose_table_xform_from_scene()`` — reads ``scene04.usd`` and
  returns the table's world ``(center, rot, scale)``
* ``compose_cloth_and_pile_xforms_from_usd()`` — reads the cloth USD
  and returns ``(cloth_pos, cloth_rot, pile_pos, pile_rot)``
* ``build_il_to_newton_qs(builder)`` — name-matched permutation from
  the IL joint-names list to Newton ``joint_q`` slot indices
* ``jp_slot_to_newton_qs(il_to_newton_qs)`` — slot map from
  ``joint_position[t, k]`` to Newton's ``joint_q[newton_qs]``
* ``apply_init_pose(jq, il_to_newton_qs)`` — overlay the
  spread_tablecloth custom init pose onto an existing joint_q array
* ``collect_link_meshes_in_link_local(stage, link_prim_path, subtree)`` —
  walk ``<link>/<subtree>`` and return concatenated ``(V, F)`` in the
  link's local frame; used to extract the USD-authored collision
  geometry (the G1's ``physics:approximation="convexHull"`` finger
  colliders) that Newton's ``add_usd`` parser doesn't honor on its own
* ``read_link_collision_approximation(stage, link_prim_path)`` — read
  the ``physics:approximation`` attribute on ``<link>/collisions``
* ``load_replay(path, episode)`` — opens the HDF5 and returns the
  recorded joint_position / actions / nodal_position arrays
"""

from __future__ import annotations

import math
import os

import h5py
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd

# ─────────────────────────────────────────────────────────────────────────────
# Asset paths (relative to newton/examples/assets/)
# ─────────────────────────────────────────────────────────────────────────────

CLOTH_USD_REL = "cloth/assets-1/assets-1/Cloth_fold06/Cloth_fold10.usd"
CLOTH_PRIM_PATH = "/root/Cloth_fold07/Visuals/Cloth_fold06"
CLOTH_ROOT_PRIM_PATH = "/root/Cloth_fold07"

# Rigid "cloth pile" (Cloth_In002) inside the same USD as the deformable cloth.
# Visual and collision meshes have identical 28k-vert geometry; PhysX uses
# convex decomposition for collision in the original, we use either CoACD hulls
# (VBD example) or a stiff tri-shell (PPFCS example).
RIGID_ROOT_PRIM_PATH = "/root/Cloth_In002"
RIGID_VIS_PRIM_PATH = "/root/Cloth_In002/Cloth_In002/Visuals/Cloth_In002"
RIGID_COL_PRIM_PATH = "/root/Cloth_In002/Cloth_In002/Collisions/Cloth_In002_Collider1"

G1_USD_REL = "cloth/assets-1/assets-1/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd"

TABLE_USD_REL = "cloth/Table256/Table256/Table256.usd"
TABLE_VIS_PRIM_PATH = "/root/Table256/Visuals/Table256"
# Table256 ships an authored offline convex decomposition: 29 mesh prims under
# Collisions/, each tagged ``physics:approximation = convexHull``.
TABLE_COL_PRIM_FMT = "/root/Table256/Collisions/Table256_Collider{i}"
TABLE_COL_COUNT = 29

SCENE_USD_REL = "cloth/assets-1/assets-1/scene04.usd"
SCENE_TABLE_PRIM_PATH = "/World/Table256"

# Recorded teleop demo from the i4h-workflows spread_tablecloth task (HDF5
# output of ``record_demos_tablecloth.py``). One episode (``data/demo_0``),
# 337 frames at the env's 30 Hz control rate.
REPLAY_HDF5_REL = "g1.hdf5"
REPLAY_EPISODE = "demo_0"


# ─────────────────────────────────────────────────────────────────────────────
# World placement constants — sourced from IL's env_cfg
# ─────────────────────────────────────────────────────────────────────────────

# G1 fixed-base world pose, taken from the HDF5's recorded
# states/articulation/robot/root_pose[0] (constant across the episode).
G1_BASE_POS = wp.vec3(-0.95, 0.0, 0.80)
G1_BASE_ROT = wp.quat_identity()

# scene04.usd's own placement in IL's env_cfg (the scene is spawned at
# pos=(0.9, -2.5, 0) with identity rotation). The table lives inside
# scene04 with its own authored xform; composing the two gives the
# table's world placement (see ``compose_table_xform_from_scene``).
SCENE_ORIGIN_POS = wp.vec3(0.9, -2.5, 0.0)

# IL env_cfg cloth init_state (verified against the recording: composed
# cloth centroid lands at the recording's frame-0 centroid within 1 mm).
ENVCFG_CLOTH_INIT_POS = wp.vec3(-0.65, 0.0, 0.78)
ENVCFG_CLOTH_INIT_ROT = wp.quat_identity()  # env_cfg rot=(0,0,0,1) XYZW = identity


# ─────────────────────────────────────────────────────────────────────────────
# Robot initial pose and HDF5 joint ordering
# ─────────────────────────────────────────────────────────────────────────────

# Custom init pose from i4h-workflows ``config/robot_config.py``
# (DEFAULT_JOINT_POS + SPREAD_TABLECLOTH_CUSTOM_JOINT_POS). All joints
# not listed default to 0.
SPREAD_TABLECLOTH_INIT_POSE: dict[str, float] = {
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

# Isaac Lab's G1 joint ordering — the env_cfg's ``joint_names`` list verbatim.
# Newton's ``add_usd`` parses the same USD in tree-traversal order (legs/arms/
# fingers grouped by side instead of interleaved). The IL→Newton permutation
# from this list is built by ``build_il_to_newton_qs``.
IL_JOINT_NAMES = [
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

# Full 53-slot mapping for the recording's
# ``data/demo_0/states/articulation/robot/joint_position`` — slot_index →
# URDF joint name. The recording's order is PhysX's articulation order,
# which differs from both env_cfg ``joint_names`` order and the URDF
# tree-DFS order Newton uses.
#
# PhysX walks the articulation breadth-first from the pelvis root,
# alternating waist + L/R leg + L/R arm joints at each depth. That's why
# arm joints (depths 5-11 from pelvis) interleave with leg joints
# (depths 1-6) inside slots 11-28.
JP_SLOT_TO_NAME: dict[int, str] = {
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
    # Finger joints — proximals (5 L + 5 R), intermediates (5 L + 5 R),
    # then thumb intermediates and distals, matching Isaac Lab's
    # articulation view order.
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


# ─────────────────────────────────────────────────────────────────────────────
# USD xform readers / world-pose composers
# ─────────────────────────────────────────────────────────────────────────────


def read_usd_prim_xform(stage: Usd.Stage, prim_path: str) -> tuple[wp.vec3, wp.quat]:
    """Read an authored ``xformOp:translate + xformOp:orient`` from a prim.

    Quaternion order in USD's ``xformOp:orient`` is WXYZ; ``wp.quat`` is
    XYZW, so we repack. Both attributes default to identity if not
    authored on the prim.
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
        rot = wp.quat(float(o.imaginary[0]), float(o.imaginary[1]), float(o.imaginary[2]), float(o.real))
    return translate, rot


def compose_cloth_and_pile_xforms_from_usd() -> tuple[wp.vec3, wp.quat, wp.vec3, wp.quat]:
    """Read the cloth + pile authored xforms from ``CLOTH_USD_REL`` and
    compose with IL env_cfg's cloth ``init_state`` to get their world
    poses.

    Returns ``(cloth_pos, cloth_rot, pile_pos, pile_rot)``. We rely on
    env_cfg's rot being identity (verified by matching the recording's
    cloth centroid), so the composition reduces to element-wise add for
    the translate and direct copy for the rotation.
    """
    cloth_usd = newton.examples.get_asset(CLOTH_USD_REL)
    stage = Usd.Stage.Open(cloth_usd)
    cloth_t, cloth_r = read_usd_prim_xform(stage, CLOTH_ROOT_PRIM_PATH)
    pile_t, pile_r = read_usd_prim_xform(stage, RIGID_ROOT_PRIM_PATH)
    env_pos = ENVCFG_CLOTH_INIT_POS
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
    return cloth_world_pos, cloth_r, pile_world_pos, pile_r


def compose_table_xform_from_scene() -> tuple[wp.vec3, wp.quat, wp.vec3]:
    """Read scene04.usd's authored xform for ``/World/Table256`` and
    compose with scene04's own world placement.

    Returns ``(center, rot, scale)`` in world frame. scene04's payload
    references for the table mesh are broken on our copy (they point
    to ``Assets/Table256/Table256.usd`` which we don't have), but the
    xform attributes on ``/World/Table256`` are authored in scene04
    itself and remain readable — we supply the geometry separately from
    our local copy at ``TABLE_USD_REL``.
    """
    scene_usd = newton.examples.get_asset(SCENE_USD_REL)
    scene_stage = Usd.Stage.Open(scene_usd, Usd.Stage.LoadNone)
    prim = scene_stage.GetPrimAtPath(SCENE_TABLE_PRIM_PATH)
    if not prim:
        raise RuntimeError(f"{SCENE_TABLE_PRIM_PATH} missing from {scene_usd}")
    t = prim.GetAttribute("xformOp:translate").Get()
    r = prim.GetAttribute("xformOp:rotateXYZ").Get()  # degrees, XYZ Euler
    s = prim.GetAttribute("xformOp:scale").Get()
    # Compose with scene04's world placement. scene04's own rotation is
    # identity (env_cfg: rot=(0,0,0,1) interpreted XYZW), so the world
    # translate is just element-wise add. Likewise scale.
    center = wp.vec3(
        float(t[0]) + float(SCENE_ORIGIN_POS[0]),
        float(t[1]) + float(SCENE_ORIGIN_POS[1]),
        float(t[2]) + float(SCENE_ORIGIN_POS[2]),
    )
    rot_z_rad = math.radians(float(r[2]))
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rot_z_rad)
    scale = wp.vec3(float(s[0]), float(s[1]), float(s[2]))
    return center, rot, scale


# ─────────────────────────────────────────────────────────────────────────────
# IL-joint-name ↔ Newton-joint_q-slot mapping
# ─────────────────────────────────────────────────────────────────────────────


def build_il_to_newton_qs(builder) -> list[int | None]:
    """Return ``il_to_newton_qs[il_idx] = newton_qs`` for every name in
    ``IL_JOINT_NAMES``, by matching builder.joint_label leaf names.

    Slots that don't match any builder joint (shouldn't happen with the
    canonical G1 USD) are ``None``.
    """
    out: list[int | None] = [None] * len(IL_JOINT_NAMES)
    for j in range(builder.joint_count):
        if int(builder.joint_type[j]) != 1:  # revolute only — fixed/free skipped
            continue
        short = builder.joint_label[j].rsplit("/", 1)[-1]
        if short in IL_JOINT_NAMES:
            out[IL_JOINT_NAMES.index(short)] = int(builder.joint_q_start[j])
    return out


def jp_slot_to_newton_qs(il_to_newton_qs: list[int | None]) -> list[tuple[int, int]]:
    """Compose the recording's slot map with the IL→Newton joint_q map.

    Returns a list of ``(jp_slot, newton_qs)`` pairs ready to consume
    via ``joint_q[newton_qs] = jp_frame[jp_slot]`` at replay time.
    Pairs with an unmapped joint are silently dropped.
    """
    pairs: list[tuple[int, int]] = []
    for jp_slot, joint_name in JP_SLOT_TO_NAME.items():
        if joint_name not in IL_JOINT_NAMES:
            continue
        il_idx = IL_JOINT_NAMES.index(joint_name)
        n_qs = il_to_newton_qs[il_idx]
        if n_qs is not None:
            pairs.append((jp_slot, n_qs))
    return pairs


def apply_init_pose(jq: np.ndarray, il_to_newton_qs: list[int | None]) -> None:
    """In-place: overlay :data:`SPREAD_TABLECLOTH_INIT_POSE` onto an
    existing Newton ``joint_q`` array.

    Only the joints named in ``SPREAD_TABLECLOTH_INIT_POSE`` are touched;
    everything else keeps the value it already had. Matches the visible
    warmup pose Isaac Lab spawns the G1 with (arms held relaxed at
    pitch=-0.3, roll=±0.5, elbow=-0.5).
    """
    for name, value in SPREAD_TABLECLOTH_INIT_POSE.items():
        if name not in IL_JOINT_NAMES:
            continue
        il_idx = IL_JOINT_NAMES.index(name)
        n_qs = il_to_newton_qs[il_idx]
        if n_qs is not None:
            jq[n_qs] = float(value)


# ─────────────────────────────────────────────────────────────────────────────
# Per-link USD geometry extraction
# ─────────────────────────────────────────────────────────────────────────────


def _gf_to_mat4(gf_matrix) -> np.ndarray:
    """Convert a USD ``Gf.Matrix4d`` (row-vector convention) to a contiguous
    ``(4, 4)`` numpy array."""
    return np.array([[gf_matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def _apply_mat4_to_pts_row(M: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Apply a row-major USD matrix ``M`` (4, 4) to ``(N, 3)`` vertices.

    USD matrices are right-multiplied with row vectors:
    ``v_world = v_local @ M``. Splitting that into rotation + translation
    avoids the per-vertex homogenisation cost.
    """
    R = M[:3, :3]
    t = M[3, :3]
    return V @ R + t


def collect_link_meshes_in_link_local(
    stage: Usd.Stage,
    link_prim_path: str,
    subtree: str = "collisions",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(V, F)`` for every Mesh prim under ``<link>/<subtree>``,
    expressed in the link prim's local frame.

    Walks the named subtree (typically ``"visuals"`` or ``"collisions"``)
    under a robot link prim in Newton's URDF-style USD layout. For each
    Mesh found, the helper computes its local-to-stage matrix and the
    link prim's local-to-stage matrix via :class:`UsdGeom.XformCache`,
    derives the relative mesh-to-link transform once, and bakes it into
    the vertices. Faces from multiple sibling Mesh prims are
    concatenated with an index offset.

    Used by both table-cloth examples to extract the USD-authored
    convex-hull finger colliders that Newton's ``add_usd`` parser
    doesn't honor on its own (it parses the rigid bodies + joints from
    the surrounding schemas but skips the meshes nested under
    ``collisions/`` because of how the parser categorises generic
    GPrim sub-trees).

    Returns ``None`` if the link or subtree is absent, or the subtree
    contains no Mesh prims.
    """
    link_prim = stage.GetPrimAtPath(link_prim_path)
    if not link_prim:
        return None
    sub = stage.GetPrimAtPath(f"{link_prim_path}/{subtree}")
    if not sub:
        return None
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link_to_stage = _gf_to_mat4(xform_cache.GetLocalToWorldTransform(link_prim))
    link_to_stage_inv = np.linalg.inv(link_to_stage)

    V_list: list[np.ndarray] = []
    F_list: list[np.ndarray] = []
    offset = 0
    for prim in Usd.PrimRange(sub):
        if prim.GetTypeName() != "Mesh":
            continue
        try:
            mesh = newton.usd.get_mesh(prim)
        except Exception:
            continue
        if mesh.vertices is None or mesh.indices is None:
            continue
        V_local = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        F = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
        mesh_to_stage = _gf_to_mat4(xform_cache.GetLocalToWorldTransform(prim))
        # USD's row-vector convention composes as ``mesh_to_link = mesh_to_stage @ link_to_stage_inv``.
        mesh_to_link = mesh_to_stage @ link_to_stage_inv
        V_in_link = _apply_mat4_to_pts_row(mesh_to_link, V_local)
        V_list.append(V_in_link)
        F_list.append(F + offset)
        offset += V_local.shape[0]
    if not V_list:
        return None
    return np.vstack(V_list), np.vstack(F_list)


def read_link_collision_approximation(stage: Usd.Stage, link_prim_path: str) -> str | None:
    """Read the ``physics:approximation`` attribute from
    ``<link>/collisions``.

    Returns the authored value (typical: ``"convexHull"``,
    ``"meshSimplification"``, ``"none"``) or ``None`` if the attribute
    or the ``collisions`` Xform is absent.
    """
    col = stage.GetPrimAtPath(f"{link_prim_path}/collisions")
    if not col:
        return None
    attr = col.GetAttribute("physics:approximation")
    if attr and attr.HasAuthoredValue():
        return str(attr.Get())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pile (Cloth_In002) — single-hull collision proxy
# ─────────────────────────────────────────────────────────────────────────────


def shrink_pile_hull_clear_of_cloth(
    pile_V_zup: np.ndarray,
    cloth_V_zup: np.ndarray,
    cloth_F: np.ndarray,
    *,
    s_min: float = 0.30,
    s_max: float = 1.00,
    n_iter: int = 18,
    ppfcs_dir=None,
) -> tuple[np.ndarray, float]:
    """Find the largest uniform centroid-shrink ``s`` in ``[s_min, s_max]``
    such that the scaled convex hull of ``pile_V_zup`` does NOT intersect
    ``(cloth_V_zup, cloth_F)``. Returns ``(V_shrunk, s)`` where
    ``V_shrunk`` is the scaled point cloud (suitable for
    ``add_shape_convex_hull`` or another SciPy hull pass) and ``s`` is
    the chosen scale.

    The pile is authored sitting inside the folded cloth, so a full-size
    hull clips the cloth shell. Centroid-shrinking the input verts is
    enough because the cloth has a real cavity around the pile — once
    the hull is small enough to live entirely inside that cavity, IPC /
    soft-contact sees no overlap at frame 0.

    Intersection checking uses ppf-contact-solver's
    :func:`frontend._intersection_.check_self_intersection` when
    ``ppfcs_dir`` resolves to the compiled solver (it's the same routine
    PPFCS will rerun at scene-build, so passing this check guarantees
    PPFCS will accept the scene). Falls back to a SciPy half-space
    inside-hull test on cloth verts when ppfcs is unavailable.
    """
    centroid = pile_V_zup.mean(axis=0)
    centered = pile_V_zup - centroid

    intersect_fn = None
    if ppfcs_dir is not None:
        try:
            import sys as _sys  # noqa: PLC0415

            ppfcs_root = str(ppfcs_dir)
            if ppfcs_root not in _sys.path:
                _sys.path.insert(0, ppfcs_root)
            from frontend._intersection_ import check_self_intersection  # noqa: PLC0415

            def intersect_fn(hull_V_zup: np.ndarray, hull_F: np.ndarray) -> bool:
                # Concatenate (cloth, hull) into one mesh; check_self_intersection
                # returns inter-mesh pairs since we don't mark anything as
                # collider here. Any pair counts as an overlap.
                n_c = cloth_V_zup.shape[0]
                V = np.vstack([cloth_V_zup, hull_V_zup]).astype(np.float64)
                F = np.vstack([cloth_F, hull_F + n_c]).astype(np.int32)
                pairs = check_self_intersection(
                    V, F, np.zeros(len(F), dtype=bool), verbose=False
                )
                return len(pairs) > 0
        except Exception:
            intersect_fn = None

    if intersect_fn is None:
        # Half-space inside-hull test on cloth vertices: conservative
        # (misses edge-crossings through cloth triangle interiors) but
        # cheap and SciPy-only.
        def intersect_fn(hull_V_zup: np.ndarray, hull_F: np.ndarray) -> bool:
            from scipy.spatial import ConvexHull  # noqa: PLC0415

            eqs = ConvexHull(hull_V_zup).equations  # (n_facets, 4): ax+b<=0 inside
            signed = cloth_V_zup @ eqs[:, :3].T + eqs[:, 3]
            return bool((signed.max(axis=1) < 0.0).any())

    from scipy.spatial import ConvexHull  # noqa: PLC0415

    def hull_at(s: float) -> tuple[np.ndarray, np.ndarray]:
        V = centroid + centered * s
        h = ConvexHull(V)
        used = np.unique(h.simplices.flatten())
        remap = -np.ones(V.shape[0], dtype=np.int64)
        remap[used] = np.arange(len(used))
        return V[used].astype(np.float64), remap[h.simplices].astype(np.int32)

    # Reject the trivial case where s_max itself doesn't overlap.
    Vh, Fh = hull_at(s_max)
    if not intersect_fn(Vh, Fh):
        return centroid + centered * s_max, s_max

    lo, hi = s_min, s_max
    best_s = s_min
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        Vh, Fh = hull_at(mid)
        if intersect_fn(Vh, Fh):
            hi = mid
        else:
            best_s = mid
            lo = mid
    return centroid + centered * best_s, best_s


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 replay loader
# ─────────────────────────────────────────────────────────────────────────────


def load_replay(path: str | None = None, episode: str = REPLAY_EPISODE):
    """Read the recorded G1 + cloth trajectory from the HDF5 file.

    Returns a dict with::

        {
            "joint_position": np.ndarray(T, 53),  # PhysX articulation order
            "actions": np.ndarray(T, 38) | None,
            "nodal_position": np.ndarray(T, 2523, 3),
            "n_frames": int,
        }

    If ``path`` is ``None``, resolves to
    ``newton.examples.get_asset(REPLAY_HDF5_REL)``. Returns ``None`` if
    the file or episode is missing.
    """
    if path is None:
        path = newton.examples.get_asset(REPLAY_HDF5_REL)
    if not os.path.exists(path):
        print(f"[table_cloth] {path} not found; replay data unavailable")
        return None

    with h5py.File(path, "r") as f:
        if "data" not in f or episode not in f["data"]:
            print(f"[table_cloth] {path} has no episode '{episode}'; replay disabled")
            return None
        demo = f["data"][episode]
        jq = np.array(demo["states/articulation/robot/joint_position"], dtype=np.float32)
        cp = np.array(demo["states/deformable_object/cloth/nodal_position"], dtype=np.float32)
        actions = np.array(demo["actions"], dtype=np.float32) if "actions" in demo else None

    return {
        "joint_position": jq,
        "actions": actions,
        "nodal_position": cp,
        "n_frames": int(jq.shape[0]),
    }
