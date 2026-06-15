# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example KFC Bag Lift via LS-DYNA shell replay
#
# This example builds the lift-start FR3 bag scene, writes an LS-DYNA
# keyword deck, launches LS-DYNA as a background process, then streams the
# results inside Newton.
#
# Flow:
#   1. Build the LS-DYNA keyword deck for a shell bag, rigid inserts, ground,
#      and rigid finger pads already gripping the bag at the lifted height.
#   2. Launch LS-DYNA as a background process and stream d3plot results while
#      the solve is still running.
#   3. Read d3plot results with lasso-python and replay the bag
#      deformation, shell stress, and rigid insert motion inside Newton.
#
# Runtime knobs:
#   - `--target-faces` lowers the shell mesh resolution written to LS-DYNA.
#   - `--capture-frames` shortens the replay horizon and output frame count.
#
# Command: python -m newton.examples.bag.example_kfc_bag_lift_ansys
#
###########################################################################

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import newton
import newton.examples
import newton.ik as ik
from newton.examples.bag.capture import (
    add_capture_arguments as _add_capture_arguments,
)
from newton.examples.bag.capture import (
    capture_replay_frame as _capture_replay_frame_common,
)
from newton.examples.bag.capture import (
    configure_capture as _configure_capture_common,
)
from newton.examples.bag.capture import (
    finalize_capture as _finalize_capture_common,
)
from newton.examples.bag.capture import (
    finalize_replay_video as _finalize_replay_video_common,
)
from newton.examples.bag.capture import (
    get_viewer_frame as _get_viewer_frame_common,
)
from newton.examples.bag.lift import (
    BAG_H_CM as _BAG_H,
)
from newton.examples.bag.lift import (
    BAG_X_CM as _BAG_X,
)
from newton.examples.bag.lift import (
    BAG_Y_CM as _BAG_Y,
)
from newton.examples.bag.lift import (
    DEFAULT_CLOSED_WIDTH_CM as _DEFAULT_CLOSED_WIDTH_CM,
)
from newton.examples.bag.lift import (
    FINGER_OPEN_Q_CM as _FINGER_OPEN_Q_CM,
)
from newton.examples.bag.lift import (
    FINGER_PAD_OFFSET_CM as _FINGER_PAD_OFFSET_CM,
)
from newton.examples.bag.lift import (
    FR3_BASE_CM as _FR3_BASE_CM,
)
from newton.examples.bag.lift import (
    FRAME_DT as _TARGET_FRAME_DT,
)
from newton.examples.bag.lift import (
    LIFT_Z_CM as _LIFT_Z,
)
from newton.examples.bag.lift import (
    TOTAL_DURATION_S as _TOTAL_DURATION_S,
)
from newton.examples.bag.lift import (
    add_fr3_hand as _add_fr3_hand,
)
from newton.examples.bag.lift import (
    add_lift_robot_arguments as _add_lift_robot_arguments,
)
from newton.examples.bag.lift import (
    finger_joint_q_from_gripper_fraction as _finger_joint_q_from_gripper_fraction,
)
from newton.examples.bag.lift import (
    finger_pad_half_extents_cm as _finger_pad_half_extents_cm_common,
)
from newton.examples.bag.lift import (
    gripper_fraction_from_closed_width_cm as _gripper_fraction_from_closed_width_cm,
)
from newton.examples.bag.lift import (
    lift_waypoints_cm as _lift_waypoints_cm,
)
from newton.examples.bag.lift import (
    log_content_placements_cm as _log_content_placements_cm,
)
from newton.examples.bag.mesh import (
    build_bary_map_with_logging as _shared_build_bary_map_with_logging,
)
from newton.examples.bag.mesh import (
    decimate_mesh as _shared_decimate_mesh,
)
from newton.examples.bag.mesh import (
    load_kfc_mesh_zup as _shared_load_kfc_mesh_zup,
)
from newton.examples.bag.render import render_bag_meshes as _render_bag_meshes

_LOG_PREFIX = "[KFC ansys]"

_FPS = 60.0
_DEFAULT_NUM_FRAMES = int(math.ceil(_TOTAL_DURATION_S / _TARGET_FRAME_DT)) + 1
_SHIFT_TARGET_FRAME_DT_S = 1.0 / 60.0

_DEFAULT_JOB_DIR = Path("outputs/lsdyna/kfc_bag_ansys_common")
_DEFAULT_LSDYNA_ROOT = Path(
    r"D:\Program Files\LS-DYNA Suite R16.1 Student\lsdyna\ls-dyna_smp_d_R16.1_180-gd50332db"
    r"e5_winx64_ifort190_sse2_studentversion.exe"
)

# Scene layout uses cm; LS-DYNA material/thickness values use SI units.
_VIZ_SCALE = 0.01

# Content objects [g]
_OBJECT_MASS_G = 1000.0

_BAG_PART_ID = 10
_GROUND_PART_ID = 20
_LEFT_PAD_PART_ID = 30
_RIGHT_PAD_PART_ID = 31
_SPHERE_PART_ID = 40
_BOX_PART_ID = 41
_CAPSULE_PART_ID = 42

_BAG_SECTION_ID = 10
_GROUND_SECTION_ID = 20
_LEFT_PAD_SECTION_ID = 30
_RIGHT_PAD_SECTION_ID = 31
_SPHERE_SECTION_ID = 40
_BOX_SECTION_ID = 41
_CAPSULE_SECTION_ID = 42

_BAG_MAT_ID = 10
_GROUND_MAT_ID = 20
_LEFT_PAD_MAT_ID = 30
_RIGHT_PAD_MAT_ID = 31
_SPHERE_MAT_ID = 40
_BOX_MAT_ID = 41
_CAPSULE_MAT_ID = 42

_BAG_THICKNESS_M = 3.5e-4
_BAG_DENSITY_KG_M3 = 240.0
_BAG_YOUNGS_MODULUS_PA = 2.5e9
_BAG_POISSON = 0.30

_RIGID_OBJECT_THICKNESS_M = 5.0e-3
_PAD_THICKNESS_M = 4.0e-3
_GROUND_THICKNESS_M = 1.0e-2
_RIGID_YOUNGS_MODULUS_PA = 2.0e11
_RIGID_POISSON = 0.30

_GROUND_HALF_EXTENTS_M = np.array([0.60, 0.60, _GROUND_THICKNESS_M * 0.5], dtype=np.float32)
_GROUND_CENTER_M = np.array([0.0, 0.0, -_GROUND_THICKNESS_M * 0.5], dtype=np.float32)

_CONTACT_FS = 0.60
_CONTACT_FD = 0.45
_STRESS_POINT_RADIUS_M = 0.0025
_DEFAULT_TARGET_FACES = 1200
_STUDENT_MAX_CPU = 4
_GRIP_ATTACH_MARGIN_M = 1.0e-3
_GRIP_ATTACH_MARGIN_SCALES = (1.0, 2.0, 4.0, 6.0, 8.0)
_GRIP_ATTACH_TARGET_NODE_COUNT = 8
_NON_PAD_CONTACT_PART_SET_ID = 1000
_STREAM_POLL_INTERVAL_S = 0.5
_STREAM_WAIT_SLEEP_S = 0.05
_STREAMING_VIEWER_MAX_FRAMES = 1_000_000_000
_D3PLOT_WRITE_RE = re.compile(r"\bt\s+([0-9.+\-Ee]+)\s+dt\s+[0-9.+\-Ee]+\s+write d3plot file\b")
_LSDYNA_DEBUG_SUMMARY_FILENAME = "lsdyna_debug_summary.json"
_LSDYNA_DEBUG_SUMMARY_SCHEMA = "newton.kfc_bag_ansys_common.lsdyna_debug"
_LSDYNA_DEBUG_SUMMARY_VERSION = 1
_LOGGED_CONTENT_PLACEMENTS: set[tuple[object, ...]] = set()
_LSDYNA_MESSAGE_FILE_GLOB = "messa" + "g*"
_LSDYNA_DIAGNOSTIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "solver_stdout": ("lsdyna.stdout.txt",),
    "d3hsp": ("d3hsp*",),
    "message": (_LSDYNA_MESSAGE_FILE_GLOB,),
    "status": ("status.out*",),
    "d3plot": ("d3plot*",),
    "glstat": ("glstat*",),
    "matsum": ("matsum*",),
    "nodout": ("nodout*",),
    "rcforc": ("rcforc*",),
    "rbdout": ("rbdout*",),
    "binout": ("binout*",),
}
_LSDYNA_TEXT_DIAGNOSTIC_KINDS = (
    "solver_stdout",
    "d3hsp",
    "message",
    "status",
    "glstat",
    "matsum",
    "nodout",
    "rcforc",
    "rbdout",
)
_LSDYNA_DEBUG_DATABASE_KEYWORDS = (
    "*DATABASE_GLSTAT",
    "*DATABASE_MATSUM",
    "*DATABASE_RCFORC",
    "*DATABASE_RBDOUT",
    "*DATABASE_NODOUT",
)


class _KeywordDeck:
    """Minimal LS-DYNA keyword deck writer used by this standalone example."""

    def __init__(self, title: str = ""):
        self.title = str(title)
        self.comment_header = ""
        self._blocks: list[str] = []

    def append(self, text: str, check: bool = True):
        _ = check
        block = str(text).rstrip("\r\n")
        if block.strip():
            self._blocks.append(block)

    def export_file(self, path: str):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["*KEYWORD"]
        if self.title:
            lines.append(f"$ {self.title}")
        if self.comment_header:
            for line in self.comment_header.splitlines():
                lines.append(f"$ {line}")
        lines.extend(self._blocks)
        if not lines or lines[-1].strip().upper() != "*END":
            lines.append("*END")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_keyword_deck():
    """Return the local keyword deck writer kept for inlined call sites."""
    return _KeywordDeck


def require_trimesh():
    """Import trimesh on demand with a targeted error."""
    try:
        import trimesh  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "This example requires trimesh. Install the examples/importers extras or run `uv pip install trimesh`."
        ) from exc
    return trimesh


def require_lasso():
    """Import lasso-python on demand with a targeted error."""
    try:
        from lasso.dyna import D3plot  # noqa: PLC0415
        from lasso.dyna.array_type import ArrayType  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "This example requires lasso-python to read LS-DYNA d3plot results. "
            "Install `newton[examples]` with Python 3.11 or newer."
        ) from exc
    return D3plot, ArrayType


def require_imageio():
    """Import imageio on demand for replay capture support."""
    try:
        import imageio.v2 as iio  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Replay capture requires imageio. Install with `uv pip install imageio imageio-ffmpeg`."
        ) from exc
    return iio


def _duration_from_capture_frames(capture_frames: int, output_dt_s: float) -> float:
    """Convert a requested replay frame count into an LS-DYNA end time."""
    if capture_frames < 2:
        raise ValueError("`--capture-frames` must be at least 2.")
    return float(capture_frames - 1) * float(output_dt_s)


def _full_capture_frame_count(output_dt_s: float) -> int:
    """Return the frame count needed to cover the full hold duration."""
    return int(math.ceil(_TOTAL_DURATION_S / float(output_dt_s))) + 1


def _disable_viewer_frame_limit_for_streaming(viewer):
    """Keep fixed-length viewers alive while LS-DYNA streams results."""
    if not hasattr(viewer, "num_frames"):
        return
    current = getattr(viewer, "num_frames", None)
    if current is None:
        return
    try:
        viewer.num_frames = max(int(current), _STREAMING_VIEWER_MAX_FRAMES)
    except (TypeError, ValueError):
        viewer.num_frames = _STREAMING_VIEWER_MAX_FRAMES


@dataclass
class ShellPart:
    """Surface mesh plus LS-DYNA metadata for one part."""

    label: str
    part_id: int
    section_id: int
    material_id: int
    vertices_m: np.ndarray
    faces: np.ndarray
    thickness_m: float
    density_kg_m3: float
    youngs_modulus_pa: float
    poisson_ratio: float
    rigid: bool = False
    fixed: bool = False
    prescribed_displacement_m: np.ndarray | None = None
    prescribed_rotation_rad: np.ndarray | None = None
    lock_rotation: bool = False


@dataclass
class MotionSamples:
    """FR3 replay samples used for both LS-DYNA drive curves and Newton replay."""

    times_s: np.ndarray
    robot_body_q_cm: np.ndarray
    left_pad_centers_m: np.ndarray
    right_pad_centers_m: np.ndarray
    left_pad_quat_xyzw: np.ndarray
    right_pad_quat_xyzw: np.ndarray


@dataclass
class DeckMetadata:
    """Metadata needed to map d3plot arrays back into viewer state."""

    deck_path: Path
    bag_part_id: int
    bag_faces: np.ndarray
    bag_node_ids: np.ndarray
    bag_node_count: int
    bag_element_count: int
    content_part_ids: dict[str, int]
    content_part_node_ids: dict[str, np.ndarray]
    content_part_initial_q_cm: dict[str, np.ndarray]


@dataclass
class ReplayData:
    """Preloaded LS-DYNA replay buffers."""

    times_s: np.ndarray
    bag_points_m: np.ndarray
    bag_von_mises: np.ndarray
    rigid_body_q_cm: dict[str, np.ndarray]


@dataclass
class ReplayLoadDiagnostics:
    """Diagnostics captured while loading LS-DYNA replay state from d3plot."""

    d3plot_path: Path
    raw_state_count: int
    resolved_time_count: int
    log_d3plot_write_count: int
    state_count_after_time_filter: int
    final_state_count: int
    dropped_by_time_filter: int
    dropped_by_geometry_filter: int
    using_fallback_times: bool


def _build_bary_map(
    full_verts: np.ndarray,
    phys_verts: np.ndarray,
    phys_faces: np.ndarray,
):
    """Map each full-res vertex to a nearby shell triangle with barycentrics."""
    return _shared_build_bary_map_with_logging(
        full_verts,
        phys_verts,
        phys_faces,
    )


def _load_kfc_mesh_zup() -> tuple[np.ndarray, np.ndarray]:
    """Load the KFC bag mesh from `kfc.usd`, convert to Z-up, scale to cm."""
    return _shared_load_kfc_mesh_zup(_BAG_H)


def _decimate_mesh(verts: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Isotropically remesh toward an approximate shell triangle budget."""
    return _shared_decimate_mesh(verts, faces, target_faces)


def _fit_bag_contents(phys_verts_cm: np.ndarray):
    """Return the rigid insert start poses used by the bag examples."""
    sphere_r_cm = 4.0
    box_h_cm = 3.0
    cap_r_cm, cap_half_len_cm = 3.0, 2.0

    bag_cx = 0.5 * (float(phys_verts_cm[:, 0].min()) + float(phys_verts_cm[:, 0].max()))
    bag_cy = 0.5 * (float(phys_verts_cm[:, 1].min()) + float(phys_verts_cm[:, 1].max()))

    def dist_sphere(cx: float, cy: float, cz: float, radius: float) -> float:
        dist = np.sqrt(
            (phys_verts_cm[:, 0] - cx) ** 2 + (phys_verts_cm[:, 1] - cy) ** 2 + (phys_verts_cm[:, 2] - cz) ** 2
        )
        return float(dist.min()) - radius

    def dist_box(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float) -> float:
        dx = np.maximum(np.abs(phys_verts_cm[:, 0] - cx) - hx, 0.0)
        dy = np.maximum(np.abs(phys_verts_cm[:, 1] - cy) - hy, 0.0)
        dz = np.maximum(np.abs(phys_verts_cm[:, 2] - cz) - hz, 0.0)
        return float(np.sqrt(dx**2 + dy**2 + dz**2).min())

    def dist_capsule_y(cx: float, cy: float, cz: float, radius: float, half_len: float) -> float:
        clamped_y = np.clip(phys_verts_cm[:, 1], cy - half_len, cy + half_len)
        dist = np.sqrt(
            (phys_verts_cm[:, 0] - cx) ** 2 + (phys_verts_cm[:, 1] - clamped_y) ** 2 + (phys_verts_cm[:, 2] - cz) ** 2
        )
        return float(dist.min()) - radius

    sphere_pos_cm = (bag_cx + 2.5, bag_cy, 5.0)
    box_pos_cm = (bag_cx - 5.5, bag_cy, 4.0)
    capsule_pos_cm = (bag_cx, bag_cy, 12.5)
    capsule_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)

    sphere_clearance_cm = dist_sphere(*sphere_pos_cm, sphere_r_cm)
    box_clearance_cm = dist_box(*box_pos_cm, box_h_cm, box_h_cm, box_h_cm)
    capsule_clearance_cm = dist_capsule_y(*capsule_pos_cm, cap_r_cm, cap_half_len_cm)
    placement_key = (
        tuple(round(float(value), 3) for value in sphere_pos_cm),
        round(float(sphere_clearance_cm), 3),
        tuple(round(float(value), 3) for value in box_pos_cm),
        round(float(box_clearance_cm), 3),
        tuple(round(float(value), 3) for value in capsule_pos_cm),
        round(float(capsule_clearance_cm), 3),
    )
    if placement_key not in _LOGGED_CONTENT_PLACEMENTS:
        _LOGGED_CONTENT_PLACEMENTS.add(placement_key)
        _log_content_placements_cm(
            sphere_pos_cm=sphere_pos_cm,
            sphere_clearance_cm=sphere_clearance_cm,
            box_pos_cm=box_pos_cm,
            box_clearance_cm=box_clearance_cm,
            capsule_pos_cm=capsule_pos_cm,
            capsule_clearance_cm=capsule_clearance_cm,
        )
    return sphere_pos_cm, box_pos_cm, capsule_pos_cm, capsule_quat


def _qv4(quat: wp.quat) -> wp.vec4:
    """Convert a Warp quaternion to a vec4 without changing xyzw ordering."""
    return wp.vec4(quat[0], quat[1], quat[2], quat[3])


def _wp_quat_to_xyzw_np(quat: wp.quat) -> np.ndarray:
    """Convert a Warp quaternion to an xyzw NumPy vector."""
    return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float32)


@lru_cache(maxsize=1)
def _lift_start_shift_cm() -> tuple[float, float, float]:
    """Return the world-space translation from the lift grab pose to the lift-start pose."""
    full_verts_cm, _ = _load_kfc_mesh_zup()

    grab_sampler = RobotTrajectorySampler(full_verts_cm, target_frame_dt_s=_SHIFT_TARGET_FRAME_DT_S)
    grab_motion = grab_sampler.sample(total_duration_s=0.65)

    lift_start_sampler = LiftStartRobotTrajectorySampler(full_verts_cm, target_frame_dt_s=_SHIFT_TARGET_FRAME_DT_S)
    lift_start_motion = lift_start_sampler.sample(total_duration_s=0.0)

    grab_centers_m = np.stack(
        [grab_motion.left_pad_centers_m[-1], grab_motion.right_pad_centers_m[-1]],
        axis=0,
    )
    lift_start_centers_m = np.stack(
        [lift_start_motion.left_pad_centers_m[0], lift_start_motion.right_pad_centers_m[0]],
        axis=0,
    )
    bag_shift_cm = (lift_start_centers_m.mean(axis=0) - grab_centers_m.mean(axis=0)) * 100.0
    return tuple(float(value) for value in bag_shift_cm)


def _load_lift_start_kfc_mesh_zup() -> tuple[np.ndarray, np.ndarray]:
    """Shift the bag mesh so frame 0 starts already gripped at lift height."""
    full_verts_cm, full_faces = _load_kfc_mesh_zup()
    bag_shift_cm = np.array(_lift_start_shift_cm(), dtype=np.float32)
    shifted_verts_cm = full_verts_cm.copy()
    shifted_verts_cm += bag_shift_cm
    return shifted_verts_cm, full_faces


def _fit_lift_start_bag_contents(bag_verts_cm: np.ndarray):
    """Shift the rigid inserts with the bag so the whole start state moves together."""
    sphere_pos_cm, box_pos_cm, capsule_pos_cm, capsule_quat_wp = _fit_bag_contents(bag_verts_cm)
    bag_shift_cm = np.array(_lift_start_shift_cm(), dtype=np.float32)
    return (
        np.asarray(sphere_pos_cm, dtype=np.float32) + bag_shift_cm,
        np.asarray(box_pos_cm, dtype=np.float32) + bag_shift_cm,
        np.asarray(capsule_pos_cm, dtype=np.float32) + bag_shift_cm,
        capsule_quat_wp,
    )


def _initial_content_body_q_cm(bag_verts_cm: np.ndarray) -> dict[str, np.ndarray]:
    """Return the original analytic rigid-body poses for the lift-start state."""
    sphere_pos_cm, box_pos_cm, capsule_pos_cm, capsule_quat_wp = _fit_lift_start_bag_contents(bag_verts_cm)
    return {
        "sphere": np.array([*sphere_pos_cm, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "box": np.array([*box_pos_cm, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "capsule": np.array([*capsule_pos_cm, *_wp_quat_to_xyzw_np(capsule_quat_wp)], dtype=np.float32),
    }


def _transform_point_cm(body_q_cm: np.ndarray, local_point_cm: np.ndarray) -> np.ndarray:
    """Apply a body transform (xyz + xyzw quat) to a local point."""
    rot = Rotation.from_quat(body_q_cm[3:7])
    return body_q_cm[:3] + rot.apply(local_point_cm)


def _quat_xyzw_from_matrices(mats: np.ndarray) -> np.ndarray:
    """Convert a batch of rotation matrices to xyzw quaternions."""
    quats = Rotation.from_matrix(mats.reshape(-1, 3, 3)).as_quat()
    return quats.reshape(*mats.shape[:-2], 4).astype(np.float32)


def _rigid_body_q_cm_from_points(initial_points_m: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    """Fit a rigid transform per frame from corresponding point clouds."""
    if points_m.ndim != 3 or points_m.shape[1:] != initial_points_m.shape:
        raise ValueError("Rigid-body point fitting expects shape [frames, node_count, 3].")

    initial_centroid = initial_points_m.mean(axis=0)
    initial_centered = initial_points_m - initial_centroid
    body_q_cm = np.zeros((points_m.shape[0], 7), dtype=np.float32)

    for frame_index, frame_points_m in enumerate(points_m):
        frame_centroid = frame_points_m.mean(axis=0)
        frame_centered = frame_points_m - frame_centroid
        covariance = initial_centered.T @ frame_centered
        u_mat, _, vh_mat = np.linalg.svd(covariance, full_matrices=False)
        rot_mat = vh_mat.T @ u_mat.T
        if np.linalg.det(rot_mat) < 0.0:
            vh_mat[-1, :] *= -1.0
            rot_mat = vh_mat.T @ u_mat.T

        quat_xyzw = Rotation.from_matrix(rot_mat).as_quat().astype(np.float32)
        body_q_cm[frame_index, :3] = (frame_centroid * 100.0).astype(np.float32)
        body_q_cm[frame_index, 3:] = quat_xyzw

    return body_q_cm


def _von_mises_from_stress(stress: np.ndarray) -> np.ndarray:
    """Collapse shell stress tensors to a per-element von Mises field."""
    if stress.ndim == 3:
        stress = stress[:, :, None, :]
    if stress.ndim != 4 or stress.shape[-1] < 6:
        raise ValueError(f"Unexpected shell stress shape: {stress.shape}")

    sxx = stress[..., 0]
    syy = stress[..., 1]
    szz = stress[..., 2]
    sxy = stress[..., 3]
    syz = stress[..., 4]
    szx = stress[..., 5]
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + szx**2))
    return vm.mean(axis=2).astype(np.float32)


def _stress_to_rgb(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a scalar stress field to a simple blue-green-red gradient."""
    scale = max(hi - lo, 1.0e-12)
    t_val = np.clip((values - lo) / scale, 0.0, 1.0).astype(np.float32)
    red = t_val
    green = 0.15 + 0.85 * (1.0 - np.abs(2.0 * t_val - 1.0))
    blue = 1.0 - t_val
    return np.column_stack([red, green, blue]).astype(np.float32)


def _node_positions_from_state(
    node_coordinates_m: np.ndarray,
    node_state_m: np.ndarray,
    sample_indexes: np.ndarray,
) -> np.ndarray:
    """Interpret the lasso nodal state array as positions or displacements."""
    if node_state_m.ndim != 3 or node_state_m.shape[1:] != node_coordinates_m.shape:
        raise ValueError(
            "Unexpected nodal state shape; expected [frame_count, node_count, 3], "
            f"got {node_state_m.shape} for coordinates {node_coordinates_m.shape}."
        )

    sample_indexes = np.asarray(sample_indexes, dtype=np.int32).reshape(-1)
    if sample_indexes.size == 0:
        sample_indexes = np.arange(min(len(node_coordinates_m), 32), dtype=np.int32)
    else:
        sample_indexes = sample_indexes[: min(sample_indexes.size, 32)]

    if np.allclose(
        node_state_m[0, sample_indexes],
        node_coordinates_m[sample_indexes],
        rtol=5.0e-2,
        atol=5.0e-3,
    ):
        return node_state_m

    return node_coordinates_m[None, :, :] + node_state_m


def _valid_state_indexes_from_positions(bag_points_m: np.ndarray) -> np.ndarray:
    """Filter replay frames by finite positions, extents, and max vertex step."""
    if bag_points_m.ndim != 3 or bag_points_m.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    initial_extent = np.ptp(bag_points_m[0], axis=0)
    max_extent = max(float(initial_extent.max()), 1.0e-6)
    min_allowed_extent = 0.3 * max_extent
    max_allowed_extent = 1.5 * max_extent
    max_allowed_step = max(10.0 * max_extent, 1.0)

    valid_indexes = [0]
    last_points = bag_points_m[0]

    for frame_index in range(1, bag_points_m.shape[0]):
        current_points = bag_points_m[frame_index]
        if not np.isfinite(current_points).all():
            continue

        current_extent = np.ptp(current_points, axis=0)
        current_max_extent = float(current_extent.max())
        if current_max_extent < min_allowed_extent or current_max_extent > max_allowed_extent:
            continue

        max_delta = float(np.linalg.norm(current_points - last_points, axis=1).max())
        if max_delta > max_allowed_step:
            continue

        valid_indexes.append(frame_index)
        last_points = current_points

    return np.asarray(valid_indexes, dtype=np.int32)


def _valid_state_indexes_from_times(raw_times_s: np.ndarray, output_dt_s: float) -> np.ndarray:
    """Filter obviously corrupted timestep entries from d3plot state arrays."""
    times_s = np.asarray(raw_times_s, dtype=np.float32).reshape(-1)
    if times_s.size == 0:
        return np.empty(0, dtype=np.int32)

    near_zero_tol = max(0.25 * float(output_dt_s), 1.0e-8)
    near_zero = np.where(np.isfinite(times_s) & (np.abs(times_s) <= near_zero_tol))[0]
    if len(near_zero) > 0:
        start_index = int(near_zero[0])
    else:
        nonnegative = np.where(np.isfinite(times_s) & (times_s >= 0.0))[0]
        if len(nonnegative) == 0:
            return np.arange(times_s.size, dtype=np.int32)
        start_index = int(nonnegative[0])

    valid_indexes = [start_index]
    last_time = float(times_s[start_index])

    for frame_index in range(start_index + 1, times_s.size):
        time_value = float(times_s[frame_index])
        if not np.isfinite(time_value) or time_value < 0.0:
            continue

        delta = time_value - last_time
        if delta < near_zero_tol:
            continue

        valid_indexes.append(frame_index)
        last_time = time_value

    return np.asarray(valid_indexes, dtype=np.int32)


def _kw_number(value: int | float) -> str:
    """Format a keyword numeric token so it fits LS-DYNA field limits."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3e}"


def _kw_line(*values: int | float) -> str:
    """Format a whitespace-delimited LS-DYNA card line."""
    tokens: list[str] = []
    for value in values:
        token = _kw_number(value)
        if len(token) > 10:
            raise ValueError(f"Keyword token `{token}` exceeds LS-DYNA's 10-character field width.")
        tokens.append(token)
    return " ".join(tokens)


def _kw_fixed_field(value: int | float | None, width: int) -> str:
    """Format a value for a fixed-width LS-DYNA table field."""
    if value is None:
        token = ""
    elif isinstance(value, (int, np.integer)):
        token = str(int(value))
    else:
        float_value = float(value)
        candidates = [
            f"{float_value:.17g}",
            f"{float_value:.14g}",
            f"{float_value:.11g}",
            f"{float_value:.8g}",
            f"{float_value:.5g}",
            f"{float_value:.4e}",
            f"{float_value:.3e}",
        ]
        token = ""
        max_token_len = max(width - 1, 1)
        for candidate in candidates:
            normalized_candidate = candidate
            if "e" not in normalized_candidate and "E" not in normalized_candidate and "." not in normalized_candidate:
                normalized_candidate += ".0"
            if len(normalized_candidate) <= max_token_len:
                token = normalized_candidate
                break
        if not token:
            raise ValueError(f"Could not fit float value `{float_value}` into width {width}.")

    if len(token) > width:
        raise ValueError(f"Fixed-width token `{token}` exceeds width {width}.")
    return f"{token:>{width}}"


def _kw_fixed_line(values: list[int | float | None], widths: list[int]) -> str:
    """Format a fixed-width LS-DYNA table line."""
    if len(values) != len(widths):
        raise ValueError("Values and widths must have the same length.")
    return "".join(_kw_fixed_field(value, width) for value, width in zip(values, widths, strict=True))


def _make_box_mesh(
    extents_m: tuple[float, float, float],
    center_m: np.ndarray,
    quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a triangulated box surface."""
    trimesh = require_trimesh()
    mesh = trimesh.creation.box(extents=np.asarray(extents_m, dtype=np.float64))
    xform = np.eye(4, dtype=np.float64)
    xform[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    xform[:3, 3] = np.asarray(center_m, dtype=np.float64)
    mesh.apply_transform(xform)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _make_sphere_mesh(radius_m: float, center_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create a triangulated sphere surface."""
    trimesh = require_trimesh()
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius_m)
    mesh.apply_translation(np.asarray(center_m, dtype=np.float64))
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _make_capsule_mesh(
    radius_m: float,
    cylindrical_length_m: float,
    center_m: np.ndarray,
    quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a triangulated capsule surface."""
    trimesh = require_trimesh()
    mesh = trimesh.creation.capsule(radius=radius_m, height=cylindrical_length_m, count=[12, 16])
    xform = np.eye(4, dtype=np.float64)
    xform[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    xform[:3, 3] = np.asarray(center_m, dtype=np.float64)
    mesh.apply_transform(xform)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _shell_density_for_mass(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    thickness_m: float,
    mass_kg: float,
) -> float:
    """Choose a shell density so surface area * thickness matches a target mass."""
    trimesh = require_trimesh()
    mesh = trimesh.Trimesh(vertices=vertices_m, faces=faces, process=False)
    area = max(float(mesh.area), 1.0e-12)
    return float(mass_kg / (area * thickness_m))


def _finger_pad_half_extents_cm(
    bag_verts_cm: np.ndarray,
    *,
    small_pad: bool,
) -> tuple[float, float, float]:
    """Compute the finger pad half extents used for replay and LS-DYNA."""
    return _finger_pad_half_extents_cm_common(
        bag_verts_cm,
        small_pad=small_pad,
        min_hx_cm=2.0,
    )


def _axis_aligned_box_from_vertices(vertices_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the center and half extents of an axis-aligned box mesh."""
    lo = np.min(vertices_m, axis=0)
    hi = np.max(vertices_m, axis=0)
    center = 0.5 * (lo + hi)
    half_extents = 0.5 * (hi - lo)
    return center.astype(np.float32), half_extents.astype(np.float32)


def _nodes_inside_axis_aligned_box(
    points_m: np.ndarray,
    center_m: np.ndarray,
    half_extents_m: np.ndarray,
) -> np.ndarray:
    """Return point indexes lying inside an axis-aligned box."""
    delta = np.abs(points_m - center_m[None, :])
    return np.where(np.all(delta <= half_extents_m[None, :], axis=1))[0].astype(np.int32)


def _take_closest_indexes(
    points_m: np.ndarray,
    center_m: np.ndarray,
    indexes: np.ndarray,
    count: int,
) -> np.ndarray:
    """Keep at most `count` indexes nearest to `center_m`."""
    indexes = np.asarray(indexes, dtype=np.int32)
    if len(indexes) <= count:
        return np.sort(indexes)

    distances = np.linalg.norm(points_m[indexes] - center_m[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    return np.sort(indexes[order[:count]])


def _relative_euler_xyz_rad(quats_xyzw: np.ndarray) -> np.ndarray:
    """Convert a quaternion sequence into relative XYZ Euler angles [rad]."""
    quats_xyzw = np.asarray(quats_xyzw, dtype=np.float64).reshape(-1, 4).copy()
    if len(quats_xyzw) == 0:
        return np.empty((0, 3), dtype=np.float32)

    for index in range(1, len(quats_xyzw)):
        if float(np.dot(quats_xyzw[index - 1], quats_xyzw[index])) < 0.0:
            quats_xyzw[index] *= -1.0

    initial_inv = Rotation.from_quat(quats_xyzw[0]).inv()
    relative_quats = np.stack(
        [(Rotation.from_quat(quat_xyzw) * initial_inv).as_quat() for quat_xyzw in quats_xyzw],
        axis=0,
    )
    relative_euler_rad = Rotation.from_quat(relative_quats).as_euler(
        "xyz",
        degrees=False,
    )
    return np.unwrap(relative_euler_rad, axis=0).astype(np.float32)


def _attached_bag_node_local_indexes(
    bag_vertices_m: np.ndarray,
    left_pad_vertices_m: np.ndarray,
    right_pad_vertices_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select bag nodes that will be rigidly attached to the finger pads."""
    left_center_m, left_half_extents_m = _axis_aligned_box_from_vertices(left_pad_vertices_m)
    right_center_m, right_half_extents_m = _axis_aligned_box_from_vertices(right_pad_vertices_m)
    grip_axis_m = right_center_m - left_center_m
    grip_axis_norm = np.linalg.norm(grip_axis_m)
    if grip_axis_norm <= 1.0e-8:
        raise RuntimeError("Finger pad centers are coincident; cannot build grip split.")
    grip_axis_m /= grip_axis_norm
    grip_midpoint_m = 0.5 * (left_center_m + right_center_m)

    left_indexes = np.empty(0, dtype=np.int32)
    right_indexes = np.empty(0, dtype=np.int32)
    for margin_scale in _GRIP_ATTACH_MARGIN_SCALES:
        margin_m = np.full(
            3,
            _GRIP_ATTACH_MARGIN_M * margin_scale,
            dtype=np.float32,
        )
        left_candidates = _nodes_inside_axis_aligned_box(
            bag_vertices_m,
            left_center_m,
            left_half_extents_m + margin_m,
        )
        right_candidates = _nodes_inside_axis_aligned_box(
            bag_vertices_m,
            right_center_m,
            right_half_extents_m + margin_m,
        )
        candidate_union = np.union1d(left_candidates, right_candidates)
        if len(candidate_union) == 0:
            continue

        signed_offsets = (bag_vertices_m[candidate_union] - grip_midpoint_m[None, :]) @ grip_axis_m
        left_indexes = candidate_union[signed_offsets <= 0.0].astype(np.int32)
        right_indexes = candidate_union[signed_offsets > 0.0].astype(np.int32)

        if len(left_indexes) >= _GRIP_ATTACH_TARGET_NODE_COUNT and len(right_indexes) >= _GRIP_ATTACH_TARGET_NODE_COUNT:
            left_indexes = _take_closest_indexes(
                bag_vertices_m,
                left_center_m,
                left_indexes,
                _GRIP_ATTACH_TARGET_NODE_COUNT,
            )
            right_indexes = _take_closest_indexes(
                bag_vertices_m,
                right_center_m,
                right_indexes,
                _GRIP_ATTACH_TARGET_NODE_COUNT,
            )
            break

    if len(left_indexes) > 0 and len(right_indexes) > 0:
        shared_count = min(
            len(left_indexes),
            len(right_indexes),
            _GRIP_ATTACH_TARGET_NODE_COUNT,
        )
        left_indexes = _take_closest_indexes(
            bag_vertices_m,
            left_center_m,
            left_indexes,
            shared_count,
        )
        right_indexes = _take_closest_indexes(
            bag_vertices_m,
            right_center_m,
            right_indexes,
            shared_count,
        )

    if len(left_indexes) == 0 or len(right_indexes) == 0:
        raise RuntimeError(
            "Could not identify bag vertices inside both finger pad boxes for the rigid attachment grip."
        )
    return left_indexes, right_indexes


class RobotTrajectorySampler:
    """Precompute the FR3 grasp trajectory used to drive the LS-DYNA finger pads."""

    def __init__(self, bag_verts_cm: np.ndarray, target_frame_dt_s: float, *, small_pad: bool = False):
        self.target_frame_dt = float(target_frame_dt_s)

        builder = newton.ModelBuilder(gravity=0.0)
        hand_bodies = _add_fr3_hand(
            builder,
            base_position=_FR3_BASE_CM,
            scale=100.0,
            finger_open_q=_FINGER_OPEN_Q_CM,
        )
        self._left_finger_body = hand_bodies.left_finger
        self._right_finger_body = hand_bodies.right_finger
        self._ee_body_index = hand_bodies.hand

        pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.001, mu=0.5, ke=1.0e4, kd=1.0)
        pad_xform = wp.transform(wp.vec3(*_FINGER_PAD_OFFSET_CM), wp.quat_identity())
        pad_hx_cm, pad_hy_cm, pad_hz_cm = _finger_pad_half_extents_cm(
            bag_verts_cm,
            small_pad=small_pad,
        )
        builder.add_shape_box(
            body=self._left_finger_body,
            xform=pad_xform,
            hx=pad_hx_cm,
            hy=pad_hy_cm,
            hz=pad_hz_cm,
            cfg=pad_cfg,
            label="left_finger_pad",
        )
        builder.add_shape_box(
            body=self._right_finger_body,
            xform=pad_xform,
            hx=pad_hx_cm,
            hy=pad_hy_cm,
            hz=pad_hz_cm,
            cfg=pad_cfg,
            label="right_finger_pad",
        )

        self.visual_model = builder.finalize()
        self.visual_state = self.visual_model.state()
        self._ik_model = self.visual_model
        self._ik_state = self.visual_state

        newton.eval_fk(self._ik_model, self._ik_model.joint_q, self._ik_model.joint_qd, self._ik_state)
        ee_tf = wp.transform(*self._ik_state.body_q.numpy()[self._ee_body_index])
        self._pos_obj = ik.IKObjectivePosition(
            link_index=self._ee_body_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=self._ee_body_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_qv4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
        )
        self._joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self._ik_model.joint_limit_lower,
            joint_limit_upper=self._ik_model.joint_limit_upper,
        )
        self._joint_q_ik = wp.array(self._ik_model.joint_q, shape=(1, self._ik_model.joint_coord_count))
        self._ik_solver = ik.IKSolver(
            model=self._ik_model,
            n_problems=1,
            objectives=[self._pos_obj, self._rot_obj, self._joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.waypoints = _lift_waypoints_cm(closed_fraction=1.0)
        self._current_waypoint = 0
        self._time_in_waypoint = 0.0
        self._gripper_frac = 0.0
        self._initialize_robot_pregrasp()

    def _initialize_robot_pregrasp(self):
        """Place the robot at the first waypoint before sampling begins."""
        start_pos = self.waypoints[0][0]
        start_frac = float(self.waypoints[0][2])
        start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)

        self._pos_obj.set_target_positions(wp.array([start_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_qv4(start_rot)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=48)
        self._gripper_frac = start_frac
        self._apply_current_pose()

    def _set_joint_targets(self):
        """Advance the end-effector waypoint schedule by one frame."""
        self._time_in_waypoint += self.target_frame_dt
        current = self.waypoints[self._current_waypoint]
        nxt = self.waypoints[min(self._current_waypoint + 1, len(self.waypoints) - 1)]
        alpha = min(self._time_in_waypoint / current[1], 1.0)

        target_pos = current[0] * (1.0 - alpha) + nxt[0] * alpha
        self._gripper_frac = float(current[2]) * (1.0 - alpha) + float(nxt[2]) * alpha

        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
        self._pos_obj.set_target_positions(wp.array([target_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_qv4(rot)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=24)

        if self._time_in_waypoint >= current[1] and self._current_waypoint < len(self.waypoints) - 1:
            self._current_waypoint += 1
            self._time_in_waypoint = 0.0

    def _apply_current_pose(self):
        """Copy the IK solution into the visual model and run FK."""
        joint_q = self.visual_model.joint_q.numpy().copy()
        ik_sol = self._joint_q_ik.numpy()[0]
        joint_q[:7] = ik_sol[:7]
        gripper_width = _finger_joint_q_from_gripper_fraction(self._gripper_frac, scale=100.0)
        joint_q[7], joint_q[8] = gripper_width, gripper_width
        joint_q_wp = wp.array(joint_q, dtype=float)

        self.visual_model.joint_q.assign(joint_q_wp)
        self.visual_state.joint_q.assign(joint_q_wp)
        self.visual_model.joint_qd.zero_()
        self.visual_state.joint_qd.zero_()
        newton.eval_fk(self.visual_model, self.visual_state.joint_q, self.visual_state.joint_qd, self.visual_state)

    def _pad_transforms_m(
        self,
        body_q_cm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the left/right finger pad centers and orientations."""
        local_pad_cm = np.array(_FINGER_PAD_OFFSET_CM, dtype=np.float32)
        left_body_q_cm = body_q_cm[self._left_finger_body]
        right_body_q_cm = body_q_cm[self._right_finger_body]
        left_cm = _transform_point_cm(left_body_q_cm, local_pad_cm)
        right_cm = _transform_point_cm(right_body_q_cm, local_pad_cm)
        left_quat_xyzw = left_body_q_cm[3:7].astype(np.float32).copy()
        right_quat_xyzw = right_body_q_cm[3:7].astype(np.float32).copy()
        return (
            left_cm * 0.01,
            left_quat_xyzw,
            right_cm * 0.01,
            right_quat_xyzw,
        )

    def sample(self, total_duration_s: float) -> MotionSamples:
        """Sample the robot trajectory at a fixed frame rate."""
        times_s = np.arange(
            0.0,
            total_duration_s + 0.5 * self.target_frame_dt,
            self.target_frame_dt,
            dtype=np.float32,
        )
        body_q_samples = []
        left_pad_samples = []
        right_pad_samples = []
        left_pad_quat_samples = []
        right_pad_quat_samples = []

        for frame_index, _ in enumerate(times_s):
            if frame_index > 0:
                self._set_joint_targets()
                self._apply_current_pose()

            body_q_cm = self.visual_state.body_q.numpy().copy().astype(np.float32)
            (
                left_pad_m,
                left_pad_quat_xyzw,
                right_pad_m,
                right_pad_quat_xyzw,
            ) = self._pad_transforms_m(body_q_cm)
            body_q_samples.append(body_q_cm)
            left_pad_samples.append(left_pad_m.astype(np.float32))
            right_pad_samples.append(right_pad_m.astype(np.float32))
            left_pad_quat_samples.append(left_pad_quat_xyzw)
            right_pad_quat_samples.append(right_pad_quat_xyzw)

        return MotionSamples(
            times_s=times_s,
            robot_body_q_cm=np.stack(body_q_samples),
            left_pad_centers_m=np.stack(left_pad_samples),
            right_pad_centers_m=np.stack(right_pad_samples),
            left_pad_quat_xyzw=np.stack(left_pad_quat_samples),
            right_pad_quat_xyzw=np.stack(right_pad_quat_samples),
        )


class LiftStartRobotTrajectorySampler(RobotTrajectorySampler):
    """Keep the finger pads closed at the lifted KFC bag pose."""

    def __init__(self, bag_verts_cm: np.ndarray, target_frame_dt_s: float, *, small_pad: bool = False):
        super().__init__(bag_verts_cm, target_frame_dt_s, small_pad=small_pad)
        lift_start_pos = wp.vec3(_BAG_X, _BAG_Y, _LIFT_Z)
        self.waypoints = [(lift_start_pos, max(self.target_frame_dt, 1.0), 1.0)]
        self._current_waypoint = 0
        self._time_in_waypoint = 0.0
        self._gripper_frac = 1.0
        self._initialize_robot_pregrasp()


def _build_shell_parts(
    bag_verts_cm: np.ndarray,
    bag_faces: np.ndarray,
    motion: MotionSamples,
    *,
    small_pad: bool = False,
) -> list[ShellPart]:
    """Create all shell surfaces that go into the LS-DYNA job."""
    bag_verts_m = (bag_verts_cm * 0.01).astype(np.float32)
    bag_faces = bag_faces.astype(np.int32)

    sphere_pos_cm, box_pos_cm, capsule_pos_cm, capsule_quat_wp = _fit_lift_start_bag_contents(bag_verts_cm)
    capsule_quat = _wp_quat_to_xyzw_np(capsule_quat_wp)
    object_mass_kg = _OBJECT_MASS_G * 1.0e-3

    sphere_vertices, sphere_faces = _make_sphere_mesh(0.04, np.array(sphere_pos_cm, dtype=np.float32) * 0.01)
    box_vertices, box_faces = _make_box_mesh(
        (0.06, 0.06, 0.06),
        np.array(box_pos_cm, dtype=np.float32) * 0.01,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    capsule_vertices, capsule_faces = _make_capsule_mesh(
        radius_m=0.03,
        cylindrical_length_m=0.04,
        center_m=np.array(capsule_pos_cm, dtype=np.float32) * 0.01,
        quat_xyzw=capsule_quat,
    )
    ground_vertices, ground_faces = _make_box_mesh(
        extents_m=tuple(float(v) * 2.0 for v in _GROUND_HALF_EXTENTS_M),
        center_m=_GROUND_CENTER_M,
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    pad_hx_cm, pad_hy_cm, pad_hz_cm = _finger_pad_half_extents_cm(
        bag_verts_cm,
        small_pad=small_pad,
    )
    left_pad_vertices, left_pad_faces = _make_box_mesh(
        extents_m=(2.0 * pad_hx_cm * 0.01, 2.0 * pad_hy_cm * 0.01, 2.0 * pad_hz_cm * 0.01),
        center_m=motion.left_pad_centers_m[0],
        quat_xyzw=motion.left_pad_quat_xyzw[0].astype(np.float32),
    )
    right_pad_vertices, right_pad_faces = _make_box_mesh(
        extents_m=(2.0 * pad_hx_cm * 0.01, 2.0 * pad_hy_cm * 0.01, 2.0 * pad_hz_cm * 0.01),
        center_m=motion.right_pad_centers_m[0],
        quat_xyzw=motion.right_pad_quat_xyzw[0].astype(np.float32),
    )

    sphere_density = _shell_density_for_mass(
        sphere_vertices,
        sphere_faces,
        _RIGID_OBJECT_THICKNESS_M,
        object_mass_kg,
    )
    box_density = _shell_density_for_mass(
        box_vertices,
        box_faces,
        _RIGID_OBJECT_THICKNESS_M,
        object_mass_kg,
    )
    capsule_density = _shell_density_for_mass(
        capsule_vertices,
        capsule_faces,
        _RIGID_OBJECT_THICKNESS_M,
        object_mass_kg,
    )

    left_disp = motion.left_pad_centers_m - motion.left_pad_centers_m[[0]]
    right_disp = motion.right_pad_centers_m - motion.right_pad_centers_m[[0]]
    left_rot_rad = _relative_euler_xyz_rad(motion.left_pad_quat_xyzw)
    right_rot_rad = _relative_euler_xyz_rad(motion.right_pad_quat_xyzw)

    return [
        ShellPart(
            label="bag_shell",
            part_id=_BAG_PART_ID,
            section_id=_BAG_SECTION_ID,
            material_id=_BAG_MAT_ID,
            vertices_m=bag_verts_m,
            faces=bag_faces,
            thickness_m=_BAG_THICKNESS_M,
            density_kg_m3=_BAG_DENSITY_KG_M3,
            youngs_modulus_pa=_BAG_YOUNGS_MODULUS_PA,
            poisson_ratio=_BAG_POISSON,
            rigid=False,
        ),
        ShellPart(
            label="ground",
            part_id=_GROUND_PART_ID,
            section_id=_GROUND_SECTION_ID,
            material_id=_GROUND_MAT_ID,
            vertices_m=ground_vertices,
            faces=ground_faces,
            thickness_m=_GROUND_THICKNESS_M,
            density_kg_m3=7800.0,
            youngs_modulus_pa=_RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_RIGID_POISSON,
            rigid=True,
            fixed=True,
        ),
        ShellPart(
            label="left_finger_pad",
            part_id=_LEFT_PAD_PART_ID,
            section_id=_LEFT_PAD_SECTION_ID,
            material_id=_LEFT_PAD_MAT_ID,
            vertices_m=left_pad_vertices,
            faces=left_pad_faces,
            thickness_m=_PAD_THICKNESS_M,
            density_kg_m3=7800.0,
            youngs_modulus_pa=_RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_RIGID_POISSON,
            rigid=True,
            prescribed_displacement_m=left_disp.astype(np.float32),
            prescribed_rotation_rad=left_rot_rad.astype(np.float32),
        ),
        ShellPart(
            label="right_finger_pad",
            part_id=_RIGHT_PAD_PART_ID,
            section_id=_RIGHT_PAD_SECTION_ID,
            material_id=_RIGHT_PAD_MAT_ID,
            vertices_m=right_pad_vertices,
            faces=right_pad_faces,
            thickness_m=_PAD_THICKNESS_M,
            density_kg_m3=7800.0,
            youngs_modulus_pa=_RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_RIGID_POISSON,
            rigid=True,
            prescribed_displacement_m=right_disp.astype(np.float32),
            prescribed_rotation_rad=right_rot_rad.astype(np.float32),
        ),
        ShellPart(
            label="sphere",
            part_id=_SPHERE_PART_ID,
            section_id=_SPHERE_SECTION_ID,
            material_id=_SPHERE_MAT_ID,
            vertices_m=sphere_vertices,
            faces=sphere_faces,
            thickness_m=_RIGID_OBJECT_THICKNESS_M,
            density_kg_m3=sphere_density,
            youngs_modulus_pa=_RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_RIGID_POISSON,
            rigid=True,
        ),
        ShellPart(
            label="box",
            part_id=_BOX_PART_ID,
            section_id=_BOX_SECTION_ID,
            material_id=_BOX_MAT_ID,
            vertices_m=box_vertices,
            faces=box_faces,
            thickness_m=_RIGID_OBJECT_THICKNESS_M,
            density_kg_m3=box_density,
            youngs_modulus_pa=_RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_RIGID_POISSON,
            rigid=True,
        ),
        ShellPart(
            label="capsule",
            part_id=_CAPSULE_PART_ID,
            section_id=_CAPSULE_SECTION_ID,
            material_id=_CAPSULE_MAT_ID,
            vertices_m=capsule_vertices,
            faces=capsule_faces,
            thickness_m=_RIGID_OBJECT_THICKNESS_M,
            density_kg_m3=capsule_density,
            youngs_modulus_pa=_RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_RIGID_POISSON,
            rigid=True,
        ),
    ]


def _append_keyword_block(deck, keyword: str, lines: list[str]):
    """Append one keyword block to an LS-DYNA deck."""
    deck.append("\n".join([keyword, *lines]), check=True)


def _append_lsdyna_debug_database_outputs(deck, output_dt_s: float):
    """Request cheap, solver-native LS-DYNA diagnostics alongside d3plot."""
    for keyword in _LSDYNA_DEBUG_DATABASE_KEYWORDS:
        _append_keyword_block(deck, keyword, [_kw_line(float(output_dt_s))])


def _read_text_tail(path: Path, *, max_lines: int = 20) -> str:
    """Return the last non-empty lines from a text diagnostic file."""
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    non_empty = [line.rstrip() for line in lines if line.strip()]
    if not non_empty:
        return ""
    return "\n".join(non_empty[-max_lines:])


def _lsdyna_debug_summary_path(job_dir: Path) -> Path:
    """Return the synthesized LS-DYNA debug-summary path for one job."""
    return job_dir / _LSDYNA_DEBUG_SUMMARY_FILENAME


def _lsdyna_diagnostic_matches(job_dir: Path) -> dict[str, list[Path]]:
    """Return all currently available LS-DYNA diagnostics grouped by kind."""
    matches_by_kind: dict[str, list[Path]] = {}
    for kind, patterns in _LSDYNA_DIAGNOSTIC_PATTERNS.items():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(path for path in sorted(job_dir.glob(pattern)) if path.is_file())
        deduped: dict[Path, None] = {}
        for path in matches:
            deduped[path] = None
        matches_by_kind[kind] = list(deduped.keys())
    return matches_by_kind


def _lsdyna_diagnostic_file_records(job_dir: Path) -> dict[str, list[dict[str, object]]]:
    """Return JSON-friendly metadata for existing LS-DYNA output files."""
    records_by_kind: dict[str, list[dict[str, object]]] = {}
    for kind, matches in _lsdyna_diagnostic_matches(job_dir).items():
        records: list[dict[str, object]] = []
        for path in matches:
            stat = path.stat()
            records.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        records_by_kind[kind] = records
    return records_by_kind


def _lsdyna_text_diagnostic_tails(job_dir: Path) -> dict[str, str]:
    """Return the tail of the newest LS-DYNA text diagnostics."""
    tails: dict[str, str] = {}
    matches_by_kind = _lsdyna_diagnostic_matches(job_dir)
    for kind in _LSDYNA_TEXT_DIAGNOSTIC_KINDS:
        matches = matches_by_kind.get(kind, [])
        if not matches:
            continue
        tail = _read_text_tail(matches[-1])
        if tail:
            tails[kind] = tail
    return tails


def _lsdyna_debug_hint(job_dir: Path) -> str:
    """Return a concise hint pointing to the richest available diagnostics."""
    return (
        f"See `{job_dir / 'lsdyna.stdout.txt'}`, `{job_dir / 'd3hsp'}`, "
        f"`{job_dir / _LSDYNA_MESSAGE_FILE_GLOB.rstrip('*')}`, "
        f"and `{_lsdyna_debug_summary_path(job_dir)}` for details."
    )


def _build_lsdyna_debug_summary(
    *,
    job_dir: Path,
    replay_diagnostics: ReplayLoadDiagnostics | None = None,
    target_frame_count: int | None = None,
    replay_frame_count: int | None = None,
    source_state_count: int | None = None,
    stream_mode: bool | None = None,
    stream_complete: bool | None = None,
    stream_solver_finished: bool | None = None,
    stream_solver_exit_code: int | None = None,
    last_replay_read_error: str = "",
    note: str = "",
) -> dict[str, object]:
    """Build a compact, file-backed summary of LS-DYNA debugging context."""
    log_d3plot_times_s = _read_d3plot_times_from_log(job_dir)
    solver_stdout_path = job_dir / "lsdyna.stdout.txt"
    return {
        "schema": _LSDYNA_DEBUG_SUMMARY_SCHEMA,
        "version": _LSDYNA_DEBUG_SUMMARY_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "job_dir": str(job_dir.resolve()),
        "summary_path": str(_lsdyna_debug_summary_path(job_dir).resolve()),
        "solver_stdout_path": str(solver_stdout_path.resolve()),
        "normal_termination_in_stdout": bool(_lsdyna_log_has_normal_termination(solver_stdout_path)),
        "last_d3plot_write_time_s": (None if log_d3plot_times_s.size == 0 else float(log_d3plot_times_s[-1])),
        "target_frame_count": None if target_frame_count is None else int(target_frame_count),
        "replay_frame_count": None if replay_frame_count is None else int(replay_frame_count),
        "source_state_count": None if source_state_count is None else int(source_state_count),
        "stream_mode": None if stream_mode is None else bool(stream_mode),
        "stream_complete": None if stream_complete is None else bool(stream_complete),
        "stream_solver_finished": None if stream_solver_finished is None else bool(stream_solver_finished),
        "stream_solver_exit_code": (None if stream_solver_exit_code is None else int(stream_solver_exit_code)),
        "last_replay_read_error": str(last_replay_read_error),
        "note": str(note),
        "diagnostic_files": _lsdyna_diagnostic_file_records(job_dir),
        "text_tails": _lsdyna_text_diagnostic_tails(job_dir),
        "replay_load": (
            None
            if replay_diagnostics is None
            else {
                "d3plot_path": str(replay_diagnostics.d3plot_path.resolve()),
                "raw_state_count": int(replay_diagnostics.raw_state_count),
                "resolved_time_count": int(replay_diagnostics.resolved_time_count),
                "log_d3plot_write_count": int(replay_diagnostics.log_d3plot_write_count),
                "state_count_after_time_filter": int(replay_diagnostics.state_count_after_time_filter),
                "final_state_count": int(replay_diagnostics.final_state_count),
                "dropped_by_time_filter": int(replay_diagnostics.dropped_by_time_filter),
                "dropped_by_geometry_filter": int(replay_diagnostics.dropped_by_geometry_filter),
                "using_fallback_times": bool(replay_diagnostics.using_fallback_times),
            }
        ),
    }


def _write_lsdyna_debug_summary(
    *,
    job_dir: Path,
    replay_diagnostics: ReplayLoadDiagnostics | None = None,
    target_frame_count: int | None = None,
    replay_frame_count: int | None = None,
    source_state_count: int | None = None,
    stream_mode: bool | None = None,
    stream_complete: bool | None = None,
    stream_solver_finished: bool | None = None,
    stream_solver_exit_code: int | None = None,
    last_replay_read_error: str = "",
    note: str = "",
) -> Path:
    """Write the synthesized LS-DYNA debug summary beside the solver outputs."""
    summary_path = _lsdyna_debug_summary_path(job_dir)
    payload = _build_lsdyna_debug_summary(
        job_dir=job_dir,
        replay_diagnostics=replay_diagnostics,
        target_frame_count=target_frame_count,
        replay_frame_count=replay_frame_count,
        source_state_count=source_state_count,
        stream_mode=stream_mode,
        stream_complete=stream_complete,
        stream_solver_finished=stream_solver_finished,
        stream_solver_exit_code=stream_solver_exit_code,
        last_replay_read_error=last_replay_read_error,
        note=note,
    )
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def _append_define_curve(deck, curve_id: int, pairs: np.ndarray, title: str | None = None):
    """Append a free-format *DEFINE_CURVE block."""
    lines = [_kw_fixed_line([curve_id, 0, 1.0, 1.0, 0.0, 0.0, 0, 0], [10, 10, 10, 10, 10, 10, 10, 10])]
    for x_val, y_val in pairs:
        lines.append(_kw_fixed_line([float(x_val), float(y_val)], [20, 20]))
    if title:
        lines.insert(0, title)
        _append_keyword_block(deck, "*DEFINE_CURVE_TITLE", lines)
    else:
        _append_keyword_block(deck, "*DEFINE_CURVE", lines)


def _write_keyword_deck(
    deck_path: Path,
    parts: list[ShellPart],
    motion: MotionSamples,
    output_dt_s: float,
) -> DeckMetadata:
    """Write the LS-DYNA keyword deck."""
    Deck = require_keyword_deck()

    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck = Deck(title="KFC bag lift-start shell replay deck")
    deck.comment_header = (
        "Generated by Newton's standalone LS-DYNA keyword writer.\n"
        "The actual solve is launched directly with the LS-DYNA executable."
    )

    _append_keyword_block(deck, "*CONTROL_TERMINATION", [_kw_line(float(motion.times_s[-1]))])
    _append_keyword_block(deck, "*CONTROL_TIMESTEP", [_kw_line(0.0, 0.90)])
    _append_keyword_block(deck, "*DATABASE_BINARY_D3PLOT", [_kw_line(float(output_dt_s))])
    _append_lsdyna_debug_database_outputs(deck, output_dt_s)

    gravity_curve_id = 1
    zero_curve_id = 2
    total_time_s = float(motion.times_s[-1])
    gravity_ramp_s = min(0.15, total_time_s)
    gravity_curve = np.array(
        [[0.0, 0.0], [gravity_ramp_s, 1.0], [total_time_s, 1.0]],
        dtype=np.float64,
    )
    if math.isclose(gravity_ramp_s, total_time_s):
        gravity_curve = np.array([[0.0, 0.0], [total_time_s, 1.0]], dtype=np.float64)
    zero_curve = np.array([[0.0, 0.0], [float(motion.times_s[-1]), 0.0]], dtype=np.float64)
    _append_define_curve(deck, gravity_curve_id, gravity_curve, title="gravity_ramp")
    _append_define_curve(deck, zero_curve_id, zero_curve, title="zero_rotation")
    active_until_s = float(motion.times_s[-1] + output_dt_s)

    curve_id = 100
    for part in parts:
        if part.prescribed_displacement_m is None:
            continue

        curve_id += 1
        _append_define_curve(
            deck,
            curve_id,
            np.column_stack([motion.times_s, part.prescribed_displacement_m[:, 0]]),
            title=f"{part.label}_dx",
        )
        _append_keyword_block(
            deck,
            "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
            [
                _kw_fixed_line(
                    [part.part_id, 1, 2, curve_id, 1.0, 0, active_until_s, 0.0], [10, 10, 10, 10, 10, 10, 10, 10]
                )
            ],
        )

        curve_id += 1
        _append_define_curve(
            deck,
            curve_id,
            np.column_stack([motion.times_s, part.prescribed_displacement_m[:, 1]]),
            title=f"{part.label}_dy",
        )
        _append_keyword_block(
            deck,
            "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
            [
                _kw_fixed_line(
                    [part.part_id, 2, 2, curve_id, 1.0, 0, active_until_s, 0.0], [10, 10, 10, 10, 10, 10, 10, 10]
                )
            ],
        )

        curve_id += 1
        _append_define_curve(
            deck,
            curve_id,
            np.column_stack([motion.times_s, part.prescribed_displacement_m[:, 2]]),
            title=f"{part.label}_dz",
        )
        _append_keyword_block(
            deck,
            "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
            [
                _kw_fixed_line(
                    [part.part_id, 3, 2, curve_id, 1.0, 0, active_until_s, 0.0], [10, 10, 10, 10, 10, 10, 10, 10]
                )
            ],
        )

        if part.prescribed_rotation_rad is not None:
            for axis, dof, label in ((0, 5, "rx"), (1, 6, "ry"), (2, 7, "rz")):
                curve_id += 1
                _append_define_curve(
                    deck,
                    curve_id,
                    np.column_stack([motion.times_s, part.prescribed_rotation_rad[:, axis]]),
                    title=f"{part.label}_{label}",
                )
                _append_keyword_block(
                    deck,
                    "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
                    [
                        _kw_fixed_line(
                            [part.part_id, dof, 2, curve_id, 1.0, 0, active_until_s, 0.0],
                            [10, 10, 10, 10, 10, 10, 10, 10],
                        )
                    ],
                )
        elif part.lock_rotation:
            for dof in (5, 6, 7):
                _append_keyword_block(
                    deck,
                    "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
                    [
                        _kw_fixed_line(
                            [part.part_id, dof, 2, zero_curve_id, 1.0, 0, active_until_s, 0.0],
                            [10, 10, 10, 10, 10, 10, 10, 10],
                        )
                    ],
                )

    _append_keyword_block(
        deck,
        "*LOAD_BODY_Z",
        [_kw_fixed_line([gravity_curve_id, 9.81, 0, 0.0, 0.0, 0.0, 0], [10, 10, 10, 10, 10, 10, 10])],
    )
    _append_keyword_block(
        deck,
        "*SET_PART_LIST",
        [
            _kw_fixed_line([_NON_PAD_CONTACT_PART_SET_ID], [10]),
            _kw_fixed_line(
                [
                    _BAG_PART_ID,
                    _GROUND_PART_ID,
                    _SPHERE_PART_ID,
                    _BOX_PART_ID,
                    _CAPSULE_PART_ID,
                    None,
                    None,
                    None,
                ],
                [10, 10, 10, 10, 10, 10, 10, 10],
            ),
        ],
    )
    _append_keyword_block(
        deck,
        "*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID",
        [
            "1,non_pad_contact",
            _kw_fixed_line([_NON_PAD_CONTACT_PART_SET_ID, 0, 2, 0], [10, 10, 10, 10]),
            _kw_line(_CONTACT_FS, _CONTACT_FD, 0.0, 0.0, active_until_s),
            _kw_line(1.0, 0.0, 0.0, 1.0),
        ],
    )

    for part in parts:
        _append_keyword_block(
            deck,
            "*PART",
            [
                part.label,
                _kw_fixed_line(
                    [part.part_id, part.section_id, part.material_id, 0, 0, 0, 0, 0], [10, 10, 10, 10, 10, 10, 10, 10]
                ),
            ],
        )
        _append_keyword_block(
            deck,
            "*SECTION_SHELL",
            [
                _kw_fixed_line([part.section_id, 2, 0.8333333, 5, 1.0, 0, 0, 1], [10, 10, 10, 10, 10, 10, 10, 10]),
                _kw_fixed_line(
                    [part.thickness_m, part.thickness_m, part.thickness_m, part.thickness_m, 0.0, 0.0, 0.0, 0],
                    [10, 10, 10, 10, 10, 10, 10, 10],
                ),
            ],
        )
        if part.rigid:
            _append_keyword_block(
                deck,
                "*MAT_RIGID",
                [
                    _kw_fixed_line(
                        [
                            part.material_id,
                            part.density_kg_m3,
                            part.youngs_modulus_pa,
                            part.poisson_ratio,
                            0.0,
                            0.0,
                            0.0,
                            None,
                        ],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                    ),
                    _kw_fixed_line([0.0, None, None], [10, 10, 10]),
                    _kw_fixed_line([None, None, None, None, None, None], [10, 10, 10, 10, 10, 10]),
                ],
            )
        else:
            _append_keyword_block(
                deck,
                "*MAT_ELASTIC",
                [
                    _kw_fixed_line(
                        [
                            part.material_id,
                            part.density_kg_m3,
                            part.youngs_modulus_pa,
                            part.poisson_ratio,
                            None,
                            None,
                            None,
                        ],
                        [10, 10, 10, 10, 10, 10, 10],
                    ),
                ],
            )

    next_node_id = 1
    next_elem_id = 1
    bag_node_count = len(parts[0].vertices_m)
    bag_elem_count = len(parts[0].faces)
    part_node_ids: dict[int, np.ndarray] = {}
    node_lines: list[str] = []

    for part in parts:
        node_ids = np.arange(next_node_id, next_node_id + len(part.vertices_m), dtype=np.int32)
        part_node_ids[part.part_id] = node_ids
        for node_id, vertex in zip(node_ids, part.vertices_m, strict=False):
            node_lines.append(
                _kw_fixed_line(
                    [int(node_id), float(vertex[0]), float(vertex[1]), float(vertex[2]), 0, 0],
                    [8, 16, 16, 16, 8, 8],
                )
            )
        next_node_id += len(node_ids)
    _append_keyword_block(deck, "*NODE", node_lines)

    elem_lines: list[str] = []
    for part in parts:
        node_ids = part_node_ids[part.part_id]
        for face in part.faces:
            n1 = int(node_ids[int(face[0])])
            n2 = int(node_ids[int(face[1])])
            n3 = int(node_ids[int(face[2])])
            elem_lines.append(
                _kw_fixed_line(
                    [next_elem_id, part.part_id, n1, n2, n3, n3, None, None, None, None],
                    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                )
            )
            next_elem_id += 1
    _append_keyword_block(deck, "*ELEMENT_SHELL", elem_lines)

    left_pad_part = next(part for part in parts if part.part_id == _LEFT_PAD_PART_ID)
    right_pad_part = next(part for part in parts if part.part_id == _RIGHT_PAD_PART_ID)
    left_attach_local, right_attach_local = _attached_bag_node_local_indexes(
        parts[0].vertices_m,
        left_pad_part.vertices_m,
        right_pad_part.vertices_m,
    )
    left_attach_node_ids = part_node_ids[_BAG_PART_ID][left_attach_local]
    right_attach_node_ids = part_node_ids[_BAG_PART_ID][right_attach_local]
    print(f"{_LOG_PREFIX} Attached grip nodes: left={len(left_attach_node_ids)} right={len(right_attach_node_ids)}")
    _append_keyword_block(
        deck,
        "*CONSTRAINED_EXTRA_NODES_NODE",
        [
            _kw_fixed_line(
                [_LEFT_PAD_PART_ID, int(node_id), 0],
                [10, 10, 10],
            )
            for node_id in left_attach_node_ids
        ],
    )
    _append_keyword_block(
        deck,
        "*CONSTRAINED_EXTRA_NODES_NODE",
        [
            _kw_fixed_line(
                [_RIGHT_PAD_PART_ID, int(node_id), 0],
                [10, 10, 10],
            )
            for node_id in right_attach_node_ids
        ],
    )

    fixed_parts = [part for part in parts if part.fixed]
    if fixed_parts:
        fixed_lines: list[str] = []
        for part in fixed_parts:
            for node_id in part_node_ids[part.part_id]:
                fixed_lines.append(
                    _kw_fixed_line([int(node_id), 0, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10, 10])
                )
        _append_keyword_block(deck, "*BOUNDARY_SPC_NODE", fixed_lines)

    deck.export_file(str(deck_path))

    return DeckMetadata(
        deck_path=deck_path,
        bag_part_id=_BAG_PART_ID,
        bag_faces=parts[0].faces.astype(np.int32),
        bag_node_ids=part_node_ids[_BAG_PART_ID].copy(),
        bag_node_count=bag_node_count,
        bag_element_count=bag_elem_count,
        content_part_ids={
            "sphere": _SPHERE_PART_ID,
            "box": _BOX_PART_ID,
            "capsule": _CAPSULE_PART_ID,
        },
        content_part_node_ids={
            label: part_node_ids[part_id].copy()
            for label, part_id in {
                "sphere": _SPHERE_PART_ID,
                "box": _BOX_PART_ID,
                "capsule": _CAPSULE_PART_ID,
            }.items()
        },
        content_part_initial_q_cm=_initial_content_body_q_cm(parts[0].vertices_m * 100.0),
    )


def _extract_shell_part_topology(
    arrays: dict,
    array_type,
    part_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one shell part's element rows, node ids, faces, and vertices from d3plot arrays."""
    part_ids = np.asarray(arrays[array_type.part_ids], dtype=np.int32).reshape(-1)
    shell_part_indexes = np.asarray(arrays[array_type.element_shell_part_indexes], dtype=np.int32).reshape(-1)
    shell_node_indexes = np.asarray(arrays[array_type.element_shell_node_indexes], dtype=np.int32)[:, :3]
    all_node_ids = np.asarray(arrays[array_type.node_ids], dtype=np.int32).reshape(-1)
    node_coordinates = np.asarray(arrays[array_type.node_coordinates], dtype=np.float32)

    part_matches = np.where(part_ids == int(part_id))[0]
    if len(part_matches) == 0:
        raise RuntimeError(f"Could not locate part id `{part_id}` in d3plot output.")
    part_index = int(part_matches[0])

    element_indexes = np.where(shell_part_indexes == part_index)[0]
    if len(element_indexes) == 0:
        raise RuntimeError(f"Could not locate shell elements for part id `{part_id}` in d3plot output.")

    global_node_indexes = np.unique(shell_node_indexes[element_indexes].reshape(-1))
    global_to_local = {
        int(global_index): local_index for local_index, global_index in enumerate(global_node_indexes.tolist())
    }
    faces = np.array(
        [
            [global_to_local[int(n1)], global_to_local[int(n2)], global_to_local[int(n3)]]
            for n1, n2, n3 in shell_node_indexes[element_indexes]
        ],
        dtype=np.int32,
    )
    node_ids = all_node_ids[global_node_indexes]
    vertices_m = node_coordinates[global_node_indexes].astype(np.float32)
    return element_indexes, global_node_indexes, node_ids, faces, vertices_m


def _find_lsdyna_executable(explicit_exe: str | None, search_root: Path) -> Path:
    """Locate the LS-DYNA executable or raise a targeted error."""
    if explicit_exe:
        return _ensure_lsdyna_executable(Path(explicit_exe), source="`--lsdyna-exe`")

    search_root = Path(os.path.expandvars(str(search_root))).expanduser()

    if search_root.is_file():
        return _ensure_lsdyna_executable(search_root, source="`--lsdyna-root`")

    if search_root.is_dir():
        patterns = ["ls-dyna*.exe", "lsdyna*_dp*.exe", "lsdyna*.exe", "*dyna*.exe"]
        for pattern in patterns:
            matches = sorted(search_root.rglob(pattern))
            if matches:
                return _ensure_lsdyna_executable(matches[0], source="`--lsdyna-root` match")

    if not search_root.exists():
        raise FileNotFoundError(
            "Could not locate an LS-DYNA executable. Pass `--lsdyna-exe` or "
            "configure `--lsdyna-root`. The configured "
            "`--lsdyna-root` does not exist: "
            f"`{search_root}`."
        )

    raise FileNotFoundError(
        "Could not locate an LS-DYNA executable. Pass `--lsdyna-exe` or "
        "configure `--lsdyna-root`. No executable matching `ls-dyna*.exe`, "
        "`lsdyna*_dp*.exe`, `lsdyna*.exe`, or `*dyna*.exe` was found under "
        f"`{search_root}`."
    )


def _ensure_lsdyna_executable(executable: Path, *, source: str = "LS-DYNA executable") -> Path:
    """Return a normalized LS-DYNA executable path or raise a clear error."""
    path = Path(os.path.expandvars(str(executable))).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{source} does not exist: `{path}`. Install LS-DYNA or pass a valid path with `--lsdyna-exe`."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"{source} is not a file: `{path}`. Pass the full LS-DYNA executable "
            "path via `--lsdyna-exe`, or use `--lsdyna-root` for a search folder."
        )
    return path


def _lsdyna_log_has_normal_termination(log_path: Path) -> bool:
    """Return whether the LS-DYNA log shows the solve reached normal termination."""
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "N o r m a l    t e r m i n a t i o n" in text or "*** termination time reached ***" in text


def _cleanup_lsdyna_outputs(job_dir: Path):
    """Remove stale LS-DYNA outputs before launching a fresh solve."""
    patterns = [
        "d3plot*",
        "d3hsp*",
        _LSDYNA_MESSAGE_FILE_GLOB,
        "status.out*",
        "part_des*",
        "group_file*",
        "binout*",
        "glstat*",
        "matsum*",
        "nodout*",
        "rcforc*",
        "rbdout*",
        "runrsf*",
        "d3dump*",
        "lsdyna.stdout.txt",
        _LSDYNA_DEBUG_SUMMARY_FILENAME,
    ]
    for pattern in patterns:
        for path in job_dir.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)


def _start_lsdyna_background(
    deck_path: Path,
    executable: Path,
    job_dir: Path,
    ncpu: int,
    memory: str,
) -> tuple[subprocess.Popen, object, Path]:
    """Launch LS-DYNA in the background and return the process plus log file handle."""
    executable = _ensure_lsdyna_executable(executable)
    log_path = job_dir / "lsdyna.stdout.txt"
    ncpu = min(int(ncpu), _STUDENT_MAX_CPU)
    cmd = [
        str(executable),
        f"i={deck_path.name}",
        f"ncpu={ncpu}",
        f"memory={memory}",
        "plabel=yes",
    ]
    env = os.environ.copy()
    env["LSTC_LICENSE"] = "ANSYS"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=job_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return process, log_file, log_path


def _d3plot_family_files(job_dir: Path) -> list[Path]:
    """Return the LS-DYNA d3plot file family in lexical order."""
    return sorted(path for path in job_dir.glob("d3plot*") if path.is_file())


def _d3plot_family_signature(job_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a change signature covering the full d3plot family."""
    matches = _d3plot_family_files(job_dir)
    if not matches:
        raise FileNotFoundError(f"No d3plot output found in `{job_dir}`.")
    signature: list[tuple[str, int, int]] = []
    for path in matches:
        stat = path.stat()
        signature.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(signature)


def _find_d3plot_file(job_dir: Path) -> Path:
    """Locate the first d3plot file produced by LS-DYNA."""
    direct = job_dir / "d3plot"
    if direct.exists():
        return direct
    matches = _d3plot_family_files(job_dir)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No d3plot output found in `{job_dir}`.")


def _read_d3plot_times_from_log(job_dir: Path) -> np.ndarray:
    """Parse actual LS-DYNA d3plot write times from the solver log."""
    log_path = job_dir / "lsdyna.stdout.txt"
    if not log_path.exists():
        return np.empty(0, dtype=np.float32)

    times_s: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _D3PLOT_WRITE_RE.search(line)
        if match is not None:
            times_s.append(float(match.group(1)))
    return np.asarray(times_s, dtype=np.float32)


def _resolve_replay_times_s(
    job_dir: Path,
    raw_times_s: np.ndarray,
    state_count: int,
    output_dt_s: float,
) -> tuple[np.ndarray, bool]:
    """Resolve per-state simulation times from d3plot or the solver log."""
    raw_times_s = np.asarray(raw_times_s, dtype=np.float32).reshape(-1)
    log_times_s = _read_d3plot_times_from_log(job_dir)

    if (
        raw_times_s.size == state_count
        and len(_valid_state_indexes_from_times(raw_times_s, output_dt_s)) == state_count
    ):
        return raw_times_s.astype(np.float32), False

    if log_times_s.size == state_count:
        return log_times_s.astype(np.float32), False
    if log_times_s.size > state_count:
        return log_times_s[:state_count].astype(np.float32), False
    if raw_times_s.size == state_count:
        return raw_times_s.astype(np.float32), False

    fallback = np.arange(state_count, dtype=np.float32) * float(output_dt_s)
    return fallback, True


def _load_replay_data_with_diagnostics(
    job_dir: Path,
    deck: DeckMetadata,
    output_dt_s: float,
) -> tuple[ReplayData, ReplayLoadDiagnostics]:
    """Load bag replay data plus diagnostics describing any dropped states."""
    D3plot, ArrayType = require_lasso()
    d3plot_path = _find_d3plot_file(job_dir)
    try:
        d3plot = D3plot(
            str(d3plot_path),
            state_array_filter=[
                ArrayType.node_displacement,
                ArrayType.element_shell_stress,
                ArrayType.rigid_body_coordinates,
                ArrayType.rigid_body_rotation_matrix,
            ],
        )
    except Exception as exc:
        log_path = job_dir / "lsdyna.stdout.txt"
        log_hint = f" {_lsdyna_debug_hint(job_dir)}" if log_path.exists() else ""
        raise RuntimeError(
            f"Failed to read LS-DYNA d3plot output from `{d3plot_path}`.{log_hint} "
            "If you are reusing prior results, try a fresh solve or clear the job directory."
        ) from exc

    arrays = d3plot.arrays
    if ArrayType.node_displacement not in arrays:
        raise RuntimeError(f"The d3plot file does not contain nodal displacements. {_lsdyna_debug_hint(job_dir)}")
    if ArrayType.element_shell_stress not in arrays:
        raise RuntimeError(f"The d3plot file does not contain shell stress output. {_lsdyna_debug_hint(job_dir)}")

    node_coordinates = np.asarray(arrays[ArrayType.node_coordinates], dtype=np.float32)
    node_displacement = np.asarray(arrays[ArrayType.node_displacement], dtype=np.float32)
    shell_stress = np.asarray(arrays[ArrayType.element_shell_stress], dtype=np.float32)
    raw_state_count = int(node_displacement.shape[0])
    log_d3plot_times_s = _read_d3plot_times_from_log(job_dir)
    raw_times_s = np.asarray(
        arrays.get(ArrayType.global_timesteps, arrays.get("timesteps", [])),
        dtype=np.float32,
    ).reshape(-1)
    resolved_times_s, using_fallback_times = _resolve_replay_times_s(
        job_dir=job_dir,
        raw_times_s=raw_times_s,
        state_count=node_displacement.shape[0],
        output_dt_s=output_dt_s,
    )
    resolved_time_count = int(len(resolved_times_s))

    filtered_state_indexes: np.ndarray | None = None
    geometry_state_indexes: np.ndarray | None = None
    if resolved_time_count == node_displacement.shape[0]:
        times_s = resolved_times_s.astype(np.float32)
    else:
        times_s = np.arange(node_displacement.shape[0], dtype=np.float32) * float(output_dt_s)

    state_count_after_time_filter = int(node_displacement.shape[0])
    dropped_by_time_filter = int(raw_state_count - state_count_after_time_filter)

    bag_shell_indexes, bag_node_indexes, bag_node_ids, bag_faces, _ = _extract_shell_part_topology(
        arrays,
        ArrayType,
        deck.bag_part_id,
    )
    deck.bag_faces = bag_faces
    deck.bag_node_ids = bag_node_ids
    deck.bag_node_count = len(bag_node_ids)
    deck.bag_element_count = len(bag_faces)

    node_positions_m = _node_positions_from_state(node_coordinates, node_displacement, sample_indexes=bag_node_indexes)
    bag_points_m = node_positions_m[:, bag_node_indexes, :]
    geometry_state_indexes = _valid_state_indexes_from_positions(bag_points_m)
    if 0 < len(geometry_state_indexes) < bag_points_m.shape[0]:
        node_positions_m = node_positions_m[geometry_state_indexes]
        bag_points_m = bag_points_m[geometry_state_indexes]
        shell_stress = shell_stress[geometry_state_indexes]
        times_s = times_s[geometry_state_indexes]

    final_state_count = int(bag_points_m.shape[0])
    dropped_by_geometry_filter = int(state_count_after_time_filter - final_state_count)
    bag_von_mises = _von_mises_from_stress(shell_stress[:, bag_shell_indexes])

    rigid_body_q_cm: dict[str, np.ndarray] = {}
    if ArrayType.rigid_body_coordinates in arrays and ArrayType.rigid_body_rotation_matrix in arrays:
        part_ids = np.asarray(arrays[ArrayType.part_ids], dtype=np.int32)
        rigid_part_indexes = np.asarray(arrays[ArrayType.rigid_body_part_indexes], dtype=np.int32)
        rigid_part_ids = part_ids[rigid_part_indexes]
        rigid_coords = np.asarray(arrays[ArrayType.rigid_body_coordinates], dtype=np.float32)
        rigid_rot = np.asarray(arrays[ArrayType.rigid_body_rotation_matrix], dtype=np.float32)
        if filtered_state_indexes is not None and len(filtered_state_indexes) > 0:
            rigid_coords = rigid_coords[filtered_state_indexes]
            rigid_rot = rigid_rot[filtered_state_indexes]
        if (
            geometry_state_indexes is not None
            and len(geometry_state_indexes) > 0
            and len(geometry_state_indexes) < rigid_coords.shape[0]
        ):
            rigid_coords = rigid_coords[geometry_state_indexes]
            rigid_rot = rigid_rot[geometry_state_indexes]
        rigid_rot = rigid_rot.reshape(rigid_coords.shape[0], rigid_coords.shape[1], 3, 3)
        rigid_quats = _quat_xyzw_from_matrices(rigid_rot)

        for label, part_id in deck.content_part_ids.items():
            matches = np.where(rigid_part_ids == part_id)[0]
            if len(matches) == 0:
                continue
            rigid_idx = int(matches[0])
            body_q_cm = np.concatenate([rigid_coords[:, rigid_idx] * 100.0, rigid_quats[:, rigid_idx]], axis=1)
            rigid_body_q_cm[label] = body_q_cm.astype(np.float32)
    else:
        for label, part_id in deck.content_part_ids.items():
            _, node_indexes, node_ids, _, _ = _extract_shell_part_topology(arrays, ArrayType, part_id)
            deck.content_part_node_ids[label] = node_ids
            initial_points_m = node_coordinates[node_indexes]
            current_points_m = node_positions_m[:, node_indexes, :]
            body_q_cm = _rigid_body_q_cm_from_points(initial_points_m, current_points_m)
            initial_body_q_cm = deck.content_part_initial_q_cm.get(label)
            if initial_body_q_cm is not None:
                delta_rot = Rotation.from_quat(body_q_cm[:, 3:])
                initial_rot = Rotation.from_quat(np.broadcast_to(initial_body_q_cm[3:], (body_q_cm.shape[0], 4)))
                body_q_cm[:, 3:] = (delta_rot * initial_rot).as_quat().astype(np.float32)
            rigid_body_q_cm[label] = body_q_cm

    replay_data = ReplayData(
        times_s=times_s.astype(np.float32),
        bag_points_m=bag_points_m.astype(np.float32),
        bag_von_mises=bag_von_mises.astype(np.float32),
        rigid_body_q_cm=rigid_body_q_cm,
    )
    diagnostics = ReplayLoadDiagnostics(
        d3plot_path=d3plot_path,
        raw_state_count=raw_state_count,
        resolved_time_count=resolved_time_count,
        log_d3plot_write_count=int(log_d3plot_times_s.size),
        state_count_after_time_filter=state_count_after_time_filter,
        final_state_count=final_state_count,
        dropped_by_time_filter=dropped_by_time_filter,
        dropped_by_geometry_filter=dropped_by_geometry_filter,
        using_fallback_times=bool(using_fallback_times),
    )
    return replay_data, diagnostics


def _load_replay_data(job_dir: Path, deck: DeckMetadata, output_dt_s: float) -> ReplayData:
    """Load bag deformation, shell stress, and rigid-body motion from d3plot."""
    replay_data, _ = _load_replay_data_with_diagnostics(job_dir, deck, output_dt_s)
    return replay_data


def _initial_replay_data(shell_verts_cm: np.ndarray, deck: DeckMetadata) -> ReplayData:
    """Construct a one-frame replay buffer from the known initial setup."""
    initial_body_q_cm = {
        label: pose[None, :].astype(np.float32) for label, pose in deck.content_part_initial_q_cm.items()
    }
    return ReplayData(
        times_s=np.array([0.0], dtype=np.float32),
        bag_points_m=(shell_verts_cm[None, :, :] * _VIZ_SCALE).astype(np.float32),
        bag_von_mises=np.zeros((1, len(deck.bag_faces)), dtype=np.float32),
        rigid_body_q_cm=initial_body_q_cm,
    )


def _resample_replay_data(
    replay: ReplayData,
    target_times_s: np.ndarray,
    *,
    hold_last: bool,
) -> tuple[ReplayData, bool]:
    """Align ordered LS-DYNA states onto Newton replay frame indexes."""
    if len(replay.times_s) == 0 or len(target_times_s) == 0:
        empty = ReplayData(
            times_s=np.empty(0, dtype=np.float32),
            bag_points_m=np.empty((0, *replay.bag_points_m.shape[1:]), dtype=np.float32),
            bag_von_mises=np.empty((0, *replay.bag_von_mises.shape[1:]), dtype=np.float32),
            rigid_body_q_cm={label: np.empty((0, 7), dtype=np.float32) for label in replay.rigid_body_q_cm},
        )
        return empty, False

    source_count = int(len(replay.times_s))
    if hold_last:
        frame_count = int(len(target_times_s))
        source_indexes = np.minimum(np.arange(frame_count, dtype=np.int32), source_count - 1)
    else:
        frame_count = min(source_count, int(len(target_times_s)))
        source_indexes = np.arange(frame_count, dtype=np.int32)

    if frame_count == 0:
        empty = ReplayData(
            times_s=np.empty(0, dtype=np.float32),
            bag_points_m=np.empty((0, *replay.bag_points_m.shape[1:]), dtype=np.float32),
            bag_von_mises=np.empty((0, *replay.bag_von_mises.shape[1:]), dtype=np.float32),
            rigid_body_q_cm={label: np.empty((0, 7), dtype=np.float32) for label in replay.rigid_body_q_cm},
        )
        return empty, False

    aligned = ReplayData(
        times_s=replay.times_s[source_indexes].astype(np.float32),
        bag_points_m=replay.bag_points_m[source_indexes].astype(np.float32),
        bag_von_mises=replay.bag_von_mises[source_indexes].astype(np.float32),
        rigid_body_q_cm={
            label: values[source_indexes].astype(np.float32) for label, values in replay.rigid_body_q_cm.items()
        },
    )
    used_tail_hold = bool(hold_last and frame_count > source_count)
    return aligned, used_tail_hold


def _build_visual_model(
    bag_verts_cm: np.ndarray,
    *,
    small_pad: bool = False,
) -> tuple[newton.Model, dict[str, int], int]:
    """Build the Newton-side replay model for robot and rigid contents."""
    builder = newton.ModelBuilder(gravity=0.0)
    hand_bodies = _add_fr3_hand(
        builder,
        base_position=_FR3_BASE_CM,
        scale=100.0,
        finger_open_q=_FINGER_OPEN_Q_CM,
    )
    robot_body_count = builder.body_count

    left_finger_body = hand_bodies.left_finger
    right_finger_body = hand_bodies.right_finger
    pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.001, mu=0.5, ke=1.0e4, kd=1.0)
    pad_xform = wp.transform(wp.vec3(*_FINGER_PAD_OFFSET_CM), wp.quat_identity())
    pad_hx_cm, pad_hy_cm, pad_hz_cm = _finger_pad_half_extents_cm(
        bag_verts_cm,
        small_pad=small_pad,
    )
    builder.add_shape_box(
        body=left_finger_body,
        xform=pad_xform,
        hx=pad_hx_cm,
        hy=pad_hy_cm,
        hz=pad_hz_cm,
        cfg=pad_cfg,
        label="left_finger_pad",
    )
    builder.add_shape_box(
        body=right_finger_body,
        xform=pad_xform,
        hx=pad_hx_cm,
        hy=pad_hy_cm,
        hz=pad_hz_cm,
        cfg=pad_cfg,
        label="right_finger_pad",
    )

    sphere_pos_cm, box_pos_cm, capsule_pos_cm, capsule_quat = _fit_lift_start_bag_contents(bag_verts_cm)
    content_cfg = newton.ModelBuilder.ShapeConfig(density=0.001, mu=0.4, ke=1.0e4, kd=10.0)

    builder.add_body(xform=wp.transform(wp.vec3(*sphere_pos_cm), wp.quat_identity()), mass=0.1)
    builder.add_shape_sphere(body=robot_body_count, radius=4.0, cfg=content_cfg)

    builder.add_body(xform=wp.transform(wp.vec3(*box_pos_cm), wp.quat_identity()), mass=0.1)
    builder.add_shape_box(body=robot_body_count + 1, hx=3.0, hy=3.0, hz=3.0, cfg=content_cfg)

    builder.add_body(xform=wp.transform(wp.vec3(*capsule_pos_cm), capsule_quat), mass=0.1)
    builder.add_shape_capsule(body=robot_body_count + 2, radius=3.0, half_height=2.0, cfg=content_cfg)

    builder.add_ground_plane()
    builder.color(include_bending=True)

    model = builder.finalize()
    body_map = {
        "sphere": robot_body_count,
        "box": robot_body_count + 1,
        "capsule": robot_body_count + 2,
    }
    return model, body_map, robot_body_count


class Example:
    """Replay an LS-DYNA shell bag lift-start solve inside Newton."""

    @staticmethod
    def create_parser():
        """Create the argument parser for the example."""
        parser = newton.examples.create_parser()
        parser.description = (
            "Replay an LS-DYNA shell bag lift-start inside Newton. The finger pads "
            "start already gripping the bag at the lifted height, the keyword "
            "deck is written by this standalone example, and the LS-DYNA "
            "solve is launched directly via the configured executable."
        )
        parser.set_defaults(num_frames=_DEFAULT_NUM_FRAMES)
        _add_capture_arguments(
            parser,
            replay_help="Capture rendered frames and auto-build a replay video or gif.",
            capture_frames_default=_DEFAULT_NUM_FRAMES,
            include_save_mp4=False,
        )
        parser.add_argument(
            "--lsdyna-root",
            type=str,
            default=str(_DEFAULT_LSDYNA_ROOT),
            help="Root folder to search for LS-DYNA if --lsdyna-exe is not given.",
        )
        parser.add_argument(
            "--lsdyna-exe",
            type=str,
            default=None,
            help="Full path to the LS-DYNA executable.",
        )
        parser.add_argument(
            "--job-dir",
            type=str,
            default=str(_DEFAULT_JOB_DIR),
            help="Directory used for the generated keyword deck and LS-DYNA outputs.",
        )
        parser.add_argument(
            "--target-faces",
            type=int,
            default=_DEFAULT_TARGET_FACES,
            help="Approximate shell face count for the bag mesh written to LS-DYNA.",
        )
        _add_lift_robot_arguments(parser, closed_width=False)
        parser.add_argument(
            "--output-dt",
            type=float,
            default=_TARGET_FRAME_DT,
            help="Requested d3plot output interval [s].",
        )
        parser.add_argument(
            "--ncpu",
            type=int,
            default=_STUDENT_MAX_CPU,
            help="CPU count passed to LS-DYNA (student builds are capped at 4).",
        )
        parser.add_argument(
            "--memory",
            type=str,
            default="200m",
            help="LS-DYNA memory argument, e.g. `200m` for 200 million words.",
        )
        return parser

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.target_frame_dt = float(args.output_dt)
        self._full_capture_frames = _full_capture_frame_count(self.target_frame_dt)
        requested_capture_frames = int(args.capture_frames)
        self.capture_frames = (
            self._full_capture_frames if requested_capture_frames == _DEFAULT_NUM_FRAMES else requested_capture_frames
        )
        self._frame_index = -1
        self._max_bag_top_z_cm = 0.0
        self._capture_duration_s = _duration_from_capture_frames(self.capture_frames, self.target_frame_dt)
        self._full_replay_requested = self._capture_duration_s >= (_TOTAL_DURATION_S - 0.5 * self.target_frame_dt)
        _configure_capture_common(
            self,
            capture_replay=bool(getattr(args, "capture_replay", False)),
            capture_frames=self.capture_frames,
            capture_fps=int(getattr(args, "capture_fps", 60)),
            capture_dir=str(getattr(args, "capture_dir", "outputs/replay_capture")),
            capture_format=str(getattr(args, "capture_format", "mp4")),
        )
        self._stream_mode = False
        self._source_replay: ReplayData | None = None
        self._target_frame_times_s = np.empty(0, dtype=np.float32)
        self._tail_hold_warned = False
        self._stream_solver_process: subprocess.Popen | None = None
        self._stream_solver_log_file = None
        self._stream_solver_log_path: Path | None = None
        self._stream_solver_finished = False
        self._stream_solver_exit_code: int | None = None
        self._stream_complete = False
        self._stream_last_d3plot_signature: tuple[tuple[str, int, int], ...] | None = None
        self._stream_next_poll_time_s = 0.0
        self._last_replay_diagnostics: ReplayLoadDiagnostics | None = None
        self._stream_last_replay_read_error = ""
        self.small_pad = bool(getattr(args, "small_pad", False))

        full_verts_cm, full_faces = _load_lift_start_kfc_mesh_zup()
        self.job_dir = Path(args.job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lsdyna_debug_summary_path = _lsdyna_debug_summary_path(self.job_dir)
        existing_deck_path = self.job_dir / "input.k"
        self._stream_mode = True
        _disable_viewer_frame_limit_for_streaming(self.viewer)

        print(f"{_LOG_PREFIX} LS-DYNA debug summary: {self._lsdyna_debug_summary_path.resolve()}")

        shell_verts_cm, shell_faces = _decimate_mesh(full_verts_cm, full_faces, int(args.target_faces))

        if not self._full_replay_requested or self.capture_frames != self._full_capture_frames:
            print(
                f"{_LOG_PREFIX} Requested {self.capture_frames} replay frames "
                f"({self._capture_duration_s:.3f}s at output_dt={self.target_frame_dt:.5f}s). "
                f"Full replay is {self._full_capture_frames} frames ({_TOTAL_DURATION_S:.3f}s)."
            )

        pad_hx_cm, _, pad_hz_cm = _finger_pad_half_extents_cm(
            shell_verts_cm,
            small_pad=self.small_pad,
        )
        print(
            f"{_LOG_PREFIX} Finger pad: hx={pad_hx_cm:.2f} hz={pad_hz_cm:.2f} cm  "
            f"{'(small-pad ~20% area)' if self.small_pad else '(full pad)'}"
        )

        self.motion = LiftStartRobotTrajectorySampler(
            shell_verts_cm,
            target_frame_dt_s=self.target_frame_dt,
            small_pad=self.small_pad,
        ).sample(self._capture_duration_s)
        self._target_frame_times_s = self.motion.times_s[: self.capture_frames].astype(np.float32).copy()

        parts = _build_shell_parts(
            shell_verts_cm,
            shell_faces,
            self.motion,
            small_pad=self.small_pad,
        )
        self.deck = _write_keyword_deck(
            deck_path=existing_deck_path,
            parts=parts,
            motion=self.motion,
            output_dt_s=self.target_frame_dt,
        )

        solver_exe = _find_lsdyna_executable(args.lsdyna_exe, Path(args.lsdyna_root))
        print(f"{_LOG_PREFIX} Running LS-DYNA: {solver_exe}")
        _cleanup_lsdyna_outputs(self.job_dir)
        (
            self._stream_solver_process,
            self._stream_solver_log_file,
            self._stream_solver_log_path,
        ) = _start_lsdyna_background(
            deck_path=self.deck.deck_path,
            executable=solver_exe,
            job_dir=self.job_dir,
            ncpu=int(args.ncpu),
            memory=str(args.memory),
        )
        self._write_lsdyna_debug_summary(note="LS-DYNA launched in streaming mode.")

        self._bary_vi0_np, self._bary_vi1_np, self._bary_vi2_np, self._bary_w_np = _build_bary_map(
            full_verts_cm,
            shell_verts_cm,
            self.deck.bag_faces,
        )
        self._n_full_verts = len(full_verts_cm)
        self._full_indices_wp = wp.array(full_faces.flatten().astype(np.int32), dtype=wp.int32)
        bary_proj_cm = (
            shell_verts_cm[self._bary_vi0_np] * self._bary_w_np[:, 0:1]
            + shell_verts_cm[self._bary_vi1_np] * self._bary_w_np[:, 1:2]
            + shell_verts_cm[self._bary_vi2_np] * self._bary_w_np[:, 2:3]
        )
        self._bary_disp_m = ((full_verts_cm - bary_proj_cm) * _VIZ_SCALE).astype(np.float32)

        self._source_replay = _initial_replay_data(shell_verts_cm, self.deck)
        self._update_replay_from_source(hold_last=False)
        self._write_lsdyna_debug_summary(
            note="Initial replay buffers prepared.",
        )

        self.model, self._content_body_indices, self._robot_body_count = _build_visual_model(
            shell_verts_cm,
            small_pad=self.small_pad,
        )
        self._initial_body_q_cm = self.model.body_q.numpy().copy().astype(np.float32)
        self.viz_state = self.model.state()
        self.state_0 = self.viz_state
        self._bag_indices_wp = wp.array(self.deck.bag_faces.flatten().astype(np.int32), dtype=wp.int32)
        self._proxy_indices_wp = self._bag_indices_wp
        self._bag_face0 = self.deck.bag_faces[:, 0]
        self._bag_face1 = self.deck.bag_faces[:, 1]
        self._bag_face2 = self.deck.bag_faces[:, 2]
        self._stress_radii_wp = wp.full(
            len(self.deck.bag_faces),
            float(_STRESS_POINT_RADIUS_M),
            dtype=wp.float32,
            device=self.model.device,
        )

        shape_xf = self.model.shape_transform.numpy().copy()
        shape_xf[:, :3] *= _VIZ_SCALE
        self.model.shape_transform = wp.array(shape_xf, dtype=wp.transform, device=self.model.device)
        shape_sc = self.model.shape_scale.numpy().copy()
        shape_sc *= _VIZ_SCALE
        self.model.shape_scale = wp.array(shape_sc, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(pos=wp.vec3(1.0, -1.0, 0.8), pitch=-10.0, yaw=135.0)
        if self._stream_mode:
            self._poll_streaming_replay(force=True)

    def _write_lsdyna_debug_summary(self, *, note: str = "") -> None:
        """Persist a compact LS-DYNA debug summary for later interruption analysis."""
        replay_frame_count = None if not hasattr(self, "replay") else len(self.replay.times_s)
        source_state_count = None if self._source_replay is None else len(self._source_replay.times_s)
        _write_lsdyna_debug_summary(
            job_dir=self.job_dir,
            replay_diagnostics=self._last_replay_diagnostics,
            target_frame_count=self.capture_frames,
            replay_frame_count=replay_frame_count,
            source_state_count=source_state_count,
            stream_mode=self._stream_mode,
            stream_complete=self._stream_complete,
            stream_solver_finished=self._stream_solver_finished,
            stream_solver_exit_code=self._stream_solver_exit_code,
            last_replay_read_error=self._stream_last_replay_read_error,
            note=note,
        )

    def _motion_body_q_cm_for_replay_time(self, replay_time_s: float) -> np.ndarray:
        """Interpolate the robot pose at one LS-DYNA replay time."""
        motion_times_s = self.motion.times_s
        if len(motion_times_s) <= 1:
            return self.motion.robot_body_q_cm[0]

        right_index = int(np.searchsorted(motion_times_s, float(replay_time_s), side="left"))
        if right_index <= 0:
            return self.motion.robot_body_q_cm[0]
        if right_index >= len(motion_times_s):
            return self.motion.robot_body_q_cm[-1]

        left_index = right_index - 1
        left_time_s = float(motion_times_s[left_index])
        right_time_s = float(motion_times_s[right_index])
        denom_s = max(right_time_s - left_time_s, 1.0e-12)
        alpha = float(np.clip((float(replay_time_s) - left_time_s) / denom_s, 0.0, 1.0))

        left_q_cm = self.motion.robot_body_q_cm[left_index]
        right_q_cm = self.motion.robot_body_q_cm[right_index]
        interp_q_cm = left_q_cm.copy()
        interp_q_cm[:, :3] = (1.0 - alpha) * left_q_cm[:, :3] + alpha * right_q_cm[:, :3]

        left_quat = left_q_cm[:, 3:7]
        right_quat = right_q_cm[:, 3:7].copy()
        flip = np.sum(left_quat * right_quat, axis=1) < 0.0
        right_quat[flip] *= -1.0
        interp_quat = (1.0 - alpha) * left_quat + alpha * right_quat
        norms = np.linalg.norm(interp_quat, axis=1, keepdims=True)
        interp_q_cm[:, 3:7] = interp_quat / np.maximum(norms, 1.0e-12)
        return interp_q_cm.astype(np.float32)

    def _frame_body_q_m(self, frame_index: int) -> np.ndarray:
        """Assemble Newton body transforms for one LS-DYNA replay frame."""
        body_q_cm = self._initial_body_q_cm.copy()
        replay_time_s = float(self.replay.times_s[frame_index])
        body_q_cm[: self._robot_body_count] = self._motion_body_q_cm_for_replay_time(replay_time_s)
        for label, body_index in self._content_body_indices.items():
            replay_q_cm = self.replay.rigid_body_q_cm.get(label)
            if replay_q_cm is not None:
                body_q_cm[body_index] = replay_q_cm[frame_index]
        body_q_m = body_q_cm.copy()
        body_q_m[:, :3] *= _VIZ_SCALE
        return body_q_m

    def _frame_visual_bag_points_m(self, frame_index: int) -> np.ndarray:
        """Interpolate the high-resolution visual bag mesh from the replay shell."""
        bag_points = self.replay.bag_points_m[frame_index]
        return (
            bag_points[self._bary_vi0_np] * self._bary_w_np[:, 0:1]
            + bag_points[self._bary_vi1_np] * self._bary_w_np[:, 1:2]
            + bag_points[self._bary_vi2_np] * self._bary_w_np[:, 2:3]
            + self._bary_disp_m
        ).astype(np.float32)

    def _frame_stress_points(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute shell centroid positions and colors for the current frame."""
        bag_points = self.replay.bag_points_m[frame_index]
        centroids = (bag_points[self._bag_face0] + bag_points[self._bag_face1] + bag_points[self._bag_face2]) / 3.0
        colors = _stress_to_rgb(self.replay.bag_von_mises[frame_index], self._stress_lo, self._stress_hi)
        return centroids.astype(np.float32), colors.astype(np.float32)

    def _update_replay_from_source(self, *, hold_last: bool):
        """Align raw LS-DYNA states onto Newton replay frame indexes."""
        replay, used_tail_hold = _resample_replay_data(
            self._source_replay,
            self._target_frame_times_s,
            hold_last=hold_last,
        )
        self.replay = replay

        if len(self.replay.bag_von_mises) > 0:
            stress_lo, stress_hi = np.nanpercentile(self.replay.bag_von_mises, [5.0, 95.0])
            self._stress_lo = float(stress_lo)
            self._stress_hi = float(max(stress_hi, stress_lo + 1.0e-6))
        else:
            self._stress_lo = 0.0
            self._stress_hi = 1.0

        if used_tail_hold and not self._tail_hold_warned and len(self._source_replay.times_s) > 0:
            source_end_s = float(self._source_replay.times_s[-1])
            target_end_s = (
                float(self._target_frame_times_s[-1]) if len(self._target_frame_times_s) > 0 else source_end_s
            )
            print(
                f"{_LOG_PREFIX} LS-DYNA provided usable states through {source_end_s:.3f}s; "
                f"holding the final state through {target_end_s:.3f}s to complete replay capture."
            )
            self._tail_hold_warned = True

    def _append_streaming_replay(self, update: ReplayData) -> int:
        """Append newly available raw LS-DYNA states, then align to target frames."""
        if self._source_replay is None:
            raise RuntimeError("Replay source buffers have not been initialized.")
        if len(update.times_s) == 0:
            return 0

        last_time_s = float(self._source_replay.times_s[-1]) if len(self._source_replay.times_s) > 0 else -math.inf
        start_idx = int(np.searchsorted(update.times_s, last_time_s + 1.0e-6, side="right"))
        append_count = len(update.times_s) - start_idx
        if append_count <= 0:
            return 0

        new_slice = slice(start_idx, start_idx + append_count)
        self._source_replay.times_s = np.concatenate([self._source_replay.times_s, update.times_s[new_slice]], axis=0)
        self._source_replay.bag_points_m = np.concatenate(
            [self._source_replay.bag_points_m, update.bag_points_m[new_slice]],
            axis=0,
        )
        self._source_replay.bag_von_mises = np.concatenate(
            [self._source_replay.bag_von_mises, update.bag_von_mises[new_slice]],
            axis=0,
        )

        for label in self._content_body_indices:
            existing = self._source_replay.rigid_body_q_cm.get(label)
            if existing is None:
                initial_pose = self.deck.content_part_initial_q_cm[label][None, :].astype(np.float32)
                existing = initial_pose
            update_body_q_cm = update.rigid_body_q_cm.get(label)
            if update_body_q_cm is None or len(update_body_q_cm) < (start_idx + append_count):
                fill = np.repeat(existing[-1:, :], append_count, axis=0)
            else:
                fill = update_body_q_cm[new_slice].astype(np.float32)
            self._source_replay.rigid_body_q_cm[label] = np.concatenate([existing, fill], axis=0)

        previous_frame_count = len(self.replay.times_s)
        self._update_replay_from_source(hold_last=self._stream_complete)
        return max(len(self.replay.times_s) - previous_frame_count, 0)

    def _poll_streaming_replay(self, *, force: bool = False):
        """Poll the growing d3plot file and append any newly readable states."""
        if not self._stream_mode or self._stream_complete:
            return

        now_s = time.monotonic()
        if not force and now_s < self._stream_next_poll_time_s:
            return
        self._stream_next_poll_time_s = now_s + _STREAM_POLL_INTERVAL_S

        if self._stream_solver_process is not None:
            exit_code = self._stream_solver_process.poll()
            if exit_code is not None and not self._stream_solver_finished:
                self._stream_solver_finished = True
                self._stream_solver_exit_code = int(exit_code)
                force = True

        try:
            d3plot_signature = _d3plot_family_signature(self.job_dir)
        except FileNotFoundError as exc:
            if self._stream_solver_finished:
                self._write_lsdyna_debug_summary(
                    note="Solver finished before any d3plot file became available.",
                )
                raise RuntimeError(
                    f"LS-DYNA completed without producing a d3plot file. {_lsdyna_debug_hint(self.job_dir)}"
                ) from exc
            return

        if not force and d3plot_signature == self._stream_last_d3plot_signature:
            return

        try:
            update, self._last_replay_diagnostics = _load_replay_data_with_diagnostics(
                self.job_dir,
                self.deck,
                output_dt_s=self.target_frame_dt,
            )
            self._stream_last_replay_read_error = ""
        except Exception as exc:
            self._stream_last_replay_read_error = str(exc)
            self._write_lsdyna_debug_summary(
                note="Replay read failed while polling streamed d3plot output.",
            )
            if self._stream_solver_finished:
                raise RuntimeError(
                    f"Failed to read final LS-DYNA replay data. {_lsdyna_debug_hint(self.job_dir)}"
                ) from exc
            return

        self._stream_last_d3plot_signature = d3plot_signature
        appended = self._append_streaming_replay(update)
        if appended > 0:
            total_frames = min(self.capture_frames, len(self.motion.times_s))
            print(f"{_LOG_PREFIX} Received {len(self.replay.times_s)}/{total_frames} replay frames")
        self._write_lsdyna_debug_summary(
            note=(
                "Streaming replay updated from d3plot."
                if appended > 0
                else "Observed d3plot change without any newly appendable replay state."
            ),
        )

        if self._stream_solver_finished:
            if self._stream_solver_log_file is not None and not self._stream_solver_log_file.closed:
                self._stream_solver_log_file.close()
            if self._stream_solver_exit_code not in (0, None):
                log_path = self._stream_solver_log_path or (self.job_dir / "lsdyna.stdout.txt")
                if _lsdyna_log_has_normal_termination(log_path):
                    print(
                        f"{_LOG_PREFIX} Warning: LS-DYNA exited with code "
                        f"{self._stream_solver_exit_code} after normal termination; "
                        "continuing with written results."
                    )
                    self._stream_solver_exit_code = 0
                else:
                    self._write_lsdyna_debug_summary(
                        note="Streaming solver exited with a non-zero code before replay completion.",
                    )
                    raise RuntimeError(
                        f"LS-DYNA exited with code {self._stream_solver_exit_code}. {_lsdyna_debug_hint(self.job_dir)}"
                    )
            self._stream_complete = True
            previous_frame_count = len(self.replay.times_s)
            self._update_replay_from_source(hold_last=True)
            if len(self.replay.times_s) > previous_frame_count:
                total_frames = min(self.capture_frames, len(self.motion.times_s))
                print(f"{_LOG_PREFIX} Received {len(self.replay.times_s)}/{total_frames} replay frames")
            self._write_lsdyna_debug_summary(
                note="Streaming solver finished; replay source finalized.",
            )

    def _stop_viewer(self, *, close: bool = False):
        """Request the active viewer loop to stop as soon as possible."""
        num_frames = getattr(self.viewer, "num_frames", None)
        if num_frames is not None:
            try:
                target = int(num_frames)
            except (TypeError, ValueError):
                target = 0
            if hasattr(self.viewer, "frame_count"):
                self.viewer.frame_count = max(int(getattr(self.viewer, "frame_count", 0)), target)
            if hasattr(self.viewer, "_frame_count"):
                self.viewer._frame_count = max(int(getattr(self.viewer, "_frame_count", 0)), target)
        if close and hasattr(self.viewer, "close"):
            self.viewer.close()

    def step(self):
        """Advance the replay by one stored LS-DYNA output frame."""
        if self._stream_mode:
            self._poll_streaming_replay()
        if self._frame_index < len(self.replay.times_s) - 1:
            self._frame_index += 1
        else:
            self._frame_index = len(self.replay.times_s) - 1
            if self._stream_mode and not self._stream_complete:
                time.sleep(_STREAM_WAIT_SLEEP_S)

        self.sim_time = float(self.replay.times_s[self._frame_index])
        top_z_cm = float(self.replay.bag_points_m[self._frame_index, :, 2].max() * 100.0)
        self._max_bag_top_z_cm = max(self._max_bag_top_z_cm, top_z_cm)
        if (
            self._stream_mode
            and self._stream_complete
            and not self.capture_replay
            and hasattr(self.viewer, "num_frames")
        ):
            self._stop_viewer()

    def render(self):
        """Render the current robot/content state plus the shell bag replay."""
        frame_index = max(self._frame_index, 0)
        body_q_m = self._frame_body_q_m(frame_index)
        self.viz_state.body_q.assign(wp.array(body_q_m, dtype=wp.transform, device=self.model.device))

        visual_bag_points_wp = wp.array(
            self._frame_visual_bag_points_m(frame_index), dtype=wp.vec3, device=self.model.device
        )
        bag_points_wp = wp.array(self.replay.bag_points_m[frame_index], dtype=wp.vec3, device=self.model.device)
        stress_points, stress_colors = self._frame_stress_points(frame_index)
        stress_points_wp = wp.array(stress_points, dtype=wp.vec3, device=self.model.device)
        stress_colors_wp = wp.array(stress_colors, dtype=wp.vec3, device=self.model.device)

        def _render_stress_overlay(proxy_mode: bool) -> None:
            self.viewer.log_points(
                "/bag_stress",
                stress_points_wp,
                radii=self._stress_radii_wp,
                colors=stress_colors_wp,
                hidden=not proxy_mode,
            )

        _render_bag_meshes(
            self.viewer,
            sim_time=self.sim_time,
            viz_state=self.viz_state,
            full_positions=visual_bag_points_wp,
            full_indices=self._full_indices_wp,
            proxy_positions=bag_points_wp,
            proxy_indices=self._proxy_indices_wp,
            render_proxy_overlay=_render_stress_overlay,
        )
        self._capture_replay_frame()

    def _get_viewer_frame(self, *, render_ui: bool = False):
        """Fetch the current viewer framebuffer if the viewer supports it."""
        return _get_viewer_frame_common(self.viewer, render_ui=render_ui)

    def _capture_replay_frame(self):
        """Capture the current rendered frame and finalize when enough are saved."""
        target_capture_count = self.capture_frames
        if self._stream_mode and self._stream_complete:
            target_capture_count = min(target_capture_count, len(self.replay.times_s))
        _capture_replay_frame_common(
            self,
            frame_key=self._frame_index,
            target_frame_count=target_capture_count,
        )

    def _finalize_replay_video(self):
        """Write the captured replay frames to an mp4 or gif."""
        _finalize_replay_video_common(self)

    def cleanup(self):
        """Release resources held by the example."""
        if self._stream_solver_process is not None:
            try:
                if self._stream_solver_process.poll() is None:
                    self._stream_solver_process.terminate()
                    try:
                        self._stream_solver_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self._stream_solver_process.kill()
                        self._stream_solver_process.wait(timeout=5.0)
                    self._stream_solver_finished = True
                    self._stream_solver_exit_code = int(self._stream_solver_process.returncode)
                else:
                    self._stream_solver_finished = True
                    self._stream_solver_exit_code = int(self._stream_solver_process.returncode)
            finally:
                self._stream_solver_process = None

        if self._stream_solver_log_file is not None and not self._stream_solver_log_file.closed:
            self._stream_solver_log_file.close()
        self._stream_solver_log_file = None
        _finalize_capture_common(self)
        self._write_lsdyna_debug_summary(note="Example cleanup completed.")

    def test_final(self):
        """Basic lift-start regression for the lifted, gripped replay."""
        assert len(self.replay.times_s) >= 1, "Lift-start replay produced no frames."
        assert np.isfinite(self.replay.bag_points_m).all(), "Lift-start replay contains non-finite bag points."
        initial_top_z_cm = float(self.replay.bag_points_m[0, :, 2].max() * 100.0)
        final_top_z_cm = float(self.replay.bag_points_m[-1, :, 2].max() * 100.0)
        assert initial_top_z_cm > 30.0, (
            f"Lift-start replay did not start lifted: top z = {initial_top_z_cm:.1f} cm (expected > 30 cm)"
        )
        assert final_top_z_cm > 20.0, (
            f"Lift-start replay dropped too far during replay: top z = {final_top_z_cm:.1f} cm (expected > 20 cm)"
        )


# Snapshot the inlined common implementation so the lift rollback subclass can
# use helper references without importing a temporary module.
from types import SimpleNamespace as _SimpleNamespace

_ansys_common = _SimpleNamespace(**globals())

_DEFAULT_JOB_DIR = Path("outputs/lsdyna/kfc_bag_lift_ansys")
_LIFT_BAG_YOUNGS_MODULUS_PA = 1.0e9
_LIFT_CONTACT_FS = 0.70
_LIFT_CONTACT_FD = 0.55
_DEFAULT_TSSFAC = 0.90
_DEFAULT_ROLLBACK_MAX_RETRIES = 2
_DEFAULT_ROLLBACK_BACKTRACK_FRAMES = 0
_DEFAULT_ROLLBACK_TSSFAC_SCALE = 0.75
_DEFAULT_ROLLBACK_MIN_TSSFAC = 0.35
_ROLLBACK_SUMMARY_FILENAME = "rollback_summary.json"


def _rollback_summary_path(job_dir: Path) -> Path:
    """Return the path used to persist rollback retry metadata."""
    return job_dir / _ROLLBACK_SUMMARY_FILENAME


def _write_rollback_summary(
    summary_path: Path,
    summary: dict[str, object],
) -> None:
    """Persist one JSON rollback summary for later inspection."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def _finalize_rollback_summary_ranges(summary: dict[str, object]) -> None:
    """Derive the final replay frame ranges regenerated by each retry."""
    events = summary.get("rollback_events", [])
    if not isinstance(events, list):
        summary["final_rollback_frame_ranges"] = []
        return

    final_frame_count = int(summary.get("final_replay_frame_count", 0))
    final_ranges: list[dict[str, object]] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue

        start_index = int(event.get("accepted_frame_count_before_retry", -1))
        end_index = int(event.get("merged_frame_count_after_attempt", -1)) - 1
        if start_index < 0 or end_index < 0:
            if bool(event.get("had_committed_prefix", False)):
                start_index = int(event["restart_frame_index"]) + 1
            else:
                start_index = 0
            next_anchor_index = final_frame_count - 1
            if event_index + 1 < len(events):
                next_event = events[event_index + 1]
                if isinstance(next_event, dict):
                    if bool(next_event.get("had_committed_prefix", False)):
                        next_anchor_index = int(next_event["restart_frame_index"])
                    else:
                        next_anchor_index = -1
            end_index = min(next_anchor_index, final_frame_count - 1)

        if start_index > end_index or start_index >= final_frame_count:
            event["final_retained_frame_start_index"] = None
            event["final_retained_frame_start_number"] = None
            event["final_retained_frame_end_index"] = None
            event["final_retained_frame_end_number"] = None
            event["final_retained_frame_count"] = 0
            continue

        end_index = min(end_index, final_frame_count - 1)
        range_record = {
            "retry_index": int(event["retry_index"]),
            "attempt_index": int(event["attempt_index"]),
            "retry_mode": str(event["retry_mode"]),
            "start_frame_index": start_index,
            "start_frame_number": start_index + 1,
            "end_frame_index": end_index,
            "end_frame_number": end_index + 1,
            "frame_count": end_index - start_index + 1,
            "tssfac": float(event["tssfac"]),
        }
        event["final_retained_frame_start_index"] = range_record["start_frame_index"]
        event["final_retained_frame_start_number"] = range_record["start_frame_number"]
        event["final_retained_frame_end_index"] = range_record["end_frame_index"]
        event["final_retained_frame_end_number"] = range_record["end_frame_number"]
        event["final_retained_frame_count"] = range_record["frame_count"]
        final_ranges.append(range_record)

    summary["final_rollback_frame_ranges"] = final_ranges


def _load_lift_kfc_mesh_zup() -> tuple[np.ndarray, np.ndarray]:
    """Load the original ground-resting KFC bag mesh."""
    return _ansys_common._load_kfc_mesh_zup()


def _fit_lift_bag_contents(
    bag_verts_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, wp.quat]:
    """Place the rigid inserts inside the original ground-resting bag."""
    return _ansys_common._fit_bag_contents(bag_verts_cm)


def _initial_lift_content_body_q_cm(
    bag_verts_cm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the analytic rigid-body poses for the lift start state."""
    sphere_pos_cm, box_pos_cm, capsule_pos_cm, capsule_quat_wp = _fit_lift_bag_contents(bag_verts_cm)
    return {
        "sphere": np.array(
            [*sphere_pos_cm, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "box": np.array(
            [*box_pos_cm, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "capsule": np.array(
            [*capsule_pos_cm, *_ansys_common._wp_quat_to_xyzw_np(capsule_quat_wp)],
            dtype=np.float32,
        ),
    }


class LiftRobotTrajectorySampler(_ansys_common.RobotTrajectorySampler):
    """Replay the original lift motion with a configurable final finger gap."""

    _LIFT_WAYPOINT_INDEX = 2

    def __init__(
        self,
        bag_verts_cm: np.ndarray,
        target_frame_dt_s: float,
        *,
        small_pad: bool = False,
        closed_width_cm: float = _DEFAULT_CLOSED_WIDTH_CM,
    ):
        super().__init__(bag_verts_cm, target_frame_dt_s, small_pad=small_pad)
        closed_frac = _gripper_fraction_from_closed_width_cm(closed_width_cm)
        self.waypoints = _lift_waypoints_cm(closed_fraction=closed_frac)
        self._current_waypoint = 0
        self._time_in_waypoint = 0.0
        self._gripper_frac = 0.0
        self._initialize_robot_pregrasp()

    @staticmethod
    def _smoothstep(alpha: float) -> float:
        """Ease the lift segment to avoid a vertical velocity jump at onset."""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def _set_joint_targets(self):
        """Advance the lift schedule, easing only the upward motion segment."""
        self._time_in_waypoint += self.target_frame_dt
        current = self.waypoints[self._current_waypoint]
        nxt = self.waypoints[min(self._current_waypoint + 1, len(self.waypoints) - 1)]
        alpha = min(self._time_in_waypoint / current[1], 1.0)
        phase_alpha = self._smoothstep(alpha) if self._current_waypoint == self._LIFT_WAYPOINT_INDEX else alpha

        target_pos = current[0] * (1.0 - phase_alpha) + nxt[0] * phase_alpha
        self._gripper_frac = float(current[2]) * (1.0 - phase_alpha) + float(nxt[2]) * phase_alpha

        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.pi))
        self._pos_obj.set_target_positions(wp.array([target_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_ansys_common._qv4(rot)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=24)

        if self._time_in_waypoint >= current[1] and self._current_waypoint < len(self.waypoints) - 1:
            self._current_waypoint += 1
            self._time_in_waypoint = 0.0


def _write_keyword_deck_friction(
    deck_path: Path,
    parts: list[_ansys_common.ShellPart],
    motion: _ansys_common.MotionSamples,
    output_dt_s: float,
    *,
    tssfac: float = _DEFAULT_TSSFAC,
    content_part_initial_q_cm: dict[str, np.ndarray] | None = None,
) -> _ansys_common.DeckMetadata:
    """Write a keyword deck that uses real pad-vs-bag friction contact."""
    Deck = _ansys_common.require_keyword_deck()

    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck = Deck(title="KFC bag lift shell replay deck")
    deck.comment_header = (
        "Generated by Newton's standalone LS-DYNA keyword writer.\n"
        "The actual solve is launched directly with the LS-DYNA executable."
    )

    _ansys_common._append_keyword_block(
        deck,
        "*CONTROL_TERMINATION",
        [_ansys_common._kw_line(float(motion.times_s[-1]))],
    )
    _ansys_common._append_keyword_block(
        deck,
        "*CONTROL_TIMESTEP",
        [_ansys_common._kw_line(0.0, float(tssfac))],
    )
    _ansys_common._append_keyword_block(
        deck,
        "*DATABASE_BINARY_D3PLOT",
        [_ansys_common._kw_line(float(output_dt_s))],
    )
    _ansys_common._append_lsdyna_debug_database_outputs(deck, output_dt_s)

    gravity_curve_id = 1
    zero_curve_id = 2
    total_time_s = float(motion.times_s[-1])
    gravity_ramp_s = min(0.15, total_time_s)
    gravity_curve = np.array(
        [[0.0, 0.0], [gravity_ramp_s, 1.0], [total_time_s, 1.0]],
        dtype=np.float64,
    )
    if np.isclose(gravity_ramp_s, total_time_s):
        gravity_curve = np.array(
            [[0.0, 0.0], [total_time_s, 1.0]],
            dtype=np.float64,
        )
    zero_curve = np.array(
        [[0.0, 0.0], [float(motion.times_s[-1]), 0.0]],
        dtype=np.float64,
    )
    _ansys_common._append_define_curve(
        deck,
        gravity_curve_id,
        gravity_curve,
        title="gravity_ramp",
    )
    _ansys_common._append_define_curve(
        deck,
        zero_curve_id,
        zero_curve,
        title="zero_rotation",
    )
    active_until_s = float(motion.times_s[-1] + output_dt_s)

    curve_id = 100
    for part in parts:
        if part.prescribed_displacement_m is None:
            continue

        for axis, dof, label in ((0, 1, "dx"), (1, 2, "dy"), (2, 3, "dz")):
            curve_id += 1
            _ansys_common._append_define_curve(
                deck,
                curve_id,
                np.column_stack([motion.times_s, part.prescribed_displacement_m[:, axis]]),
                title=f"{part.label}_{label}",
            )
            _ansys_common._append_keyword_block(
                deck,
                "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
                [
                    _ansys_common._kw_fixed_line(
                        [
                            part.part_id,
                            dof,
                            2,
                            curve_id,
                            1.0,
                            0,
                            active_until_s,
                            0.0,
                        ],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                    )
                ],
            )

        if part.prescribed_rotation_rad is not None:
            for axis, dof, label in ((0, 5, "rx"), (1, 6, "ry"), (2, 7, "rz")):
                curve_id += 1
                _ansys_common._append_define_curve(
                    deck,
                    curve_id,
                    np.column_stack([motion.times_s, part.prescribed_rotation_rad[:, axis]]),
                    title=f"{part.label}_{label}",
                )
                _ansys_common._append_keyword_block(
                    deck,
                    "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
                    [
                        _ansys_common._kw_fixed_line(
                            [
                                part.part_id,
                                dof,
                                2,
                                curve_id,
                                1.0,
                                0,
                                active_until_s,
                                0.0,
                            ],
                            [10, 10, 10, 10, 10, 10, 10, 10],
                        )
                    ],
                )
        elif part.lock_rotation:
            for dof in (5, 6, 7):
                _ansys_common._append_keyword_block(
                    deck,
                    "*BOUNDARY_PRESCRIBED_MOTION_RIGID",
                    [
                        _ansys_common._kw_fixed_line(
                            [
                                part.part_id,
                                dof,
                                2,
                                zero_curve_id,
                                1.0,
                                0,
                                active_until_s,
                                0.0,
                            ],
                            [10, 10, 10, 10, 10, 10, 10, 10],
                        )
                    ],
                )

    _ansys_common._append_keyword_block(
        deck,
        "*LOAD_BODY_Z",
        [
            _ansys_common._kw_fixed_line(
                [gravity_curve_id, 9.81, 0, 0.0, 0.0, 0.0, 0],
                [10, 10, 10, 10, 10, 10, 10],
            )
        ],
    )
    _ansys_common._append_keyword_block(
        deck,
        "*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID",
        [
            "1,all_parts_contact",
            _ansys_common._kw_line(0, 0),
            _ansys_common._kw_line(
                _LIFT_CONTACT_FS,
                _LIFT_CONTACT_FD,
                0.0,
                0.0,
                active_until_s,
            ),
            _ansys_common._kw_line(1.0, 0.0, 0.0, 1.0),
        ],
    )

    for part in parts:
        _ansys_common._append_keyword_block(
            deck,
            "*PART",
            [
                part.label,
                _ansys_common._kw_fixed_line(
                    [
                        part.part_id,
                        part.section_id,
                        part.material_id,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [10, 10, 10, 10, 10, 10, 10, 10],
                ),
            ],
        )
        _ansys_common._append_keyword_block(
            deck,
            "*SECTION_SHELL",
            [
                _ansys_common._kw_fixed_line(
                    [part.section_id, 2, 0.8333333, 5, 1.0, 0, 0, 1],
                    [10, 10, 10, 10, 10, 10, 10, 10],
                ),
                _ansys_common._kw_fixed_line(
                    [
                        part.thickness_m,
                        part.thickness_m,
                        part.thickness_m,
                        part.thickness_m,
                        0.0,
                        0.0,
                        0.0,
                        0,
                    ],
                    [10, 10, 10, 10, 10, 10, 10, 10],
                ),
            ],
        )
        if part.rigid:
            _ansys_common._append_keyword_block(
                deck,
                "*MAT_RIGID",
                [
                    _ansys_common._kw_fixed_line(
                        [
                            part.material_id,
                            part.density_kg_m3,
                            part.youngs_modulus_pa,
                            part.poisson_ratio,
                            0.0,
                            0.0,
                            0.0,
                            None,
                        ],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                    ),
                    _ansys_common._kw_fixed_line([0.0, None, None], [10, 10, 10]),
                    _ansys_common._kw_fixed_line(
                        [None, None, None, None, None, None],
                        [10, 10, 10, 10, 10, 10],
                    ),
                ],
            )
        else:
            _ansys_common._append_keyword_block(
                deck,
                "*MAT_ELASTIC",
                [
                    _ansys_common._kw_fixed_line(
                        [
                            part.material_id,
                            part.density_kg_m3,
                            part.youngs_modulus_pa,
                            part.poisson_ratio,
                            None,
                            None,
                            None,
                        ],
                        [10, 10, 10, 10, 10, 10, 10],
                    ),
                ],
            )

    next_node_id = 1
    next_elem_id = 1
    bag_node_count = len(parts[0].vertices_m)
    bag_elem_count = len(parts[0].faces)
    part_node_ids: dict[int, np.ndarray] = {}
    node_lines: list[str] = []

    for part in parts:
        node_ids = np.arange(
            next_node_id,
            next_node_id + len(part.vertices_m),
            dtype=np.int32,
        )
        part_node_ids[part.part_id] = node_ids
        for node_id, vertex in zip(node_ids, part.vertices_m, strict=False):
            node_lines.append(
                _ansys_common._kw_fixed_line(
                    [
                        int(node_id),
                        float(vertex[0]),
                        float(vertex[1]),
                        float(vertex[2]),
                        0,
                        0,
                    ],
                    [8, 16, 16, 16, 8, 8],
                )
            )
        next_node_id += len(node_ids)
    _ansys_common._append_keyword_block(deck, "*NODE", node_lines)

    elem_lines: list[str] = []
    for part in parts:
        node_ids = part_node_ids[part.part_id]
        for face in part.faces:
            n1 = int(node_ids[int(face[0])])
            n2 = int(node_ids[int(face[1])])
            n3 = int(node_ids[int(face[2])])
            elem_lines.append(
                _ansys_common._kw_fixed_line(
                    [
                        next_elem_id,
                        part.part_id,
                        n1,
                        n2,
                        n3,
                        n3,
                        None,
                        None,
                        None,
                        None,
                    ],
                    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                )
            )
            next_elem_id += 1
    _ansys_common._append_keyword_block(deck, "*ELEMENT_SHELL", elem_lines)

    fixed_parts = [part for part in parts if part.fixed]
    if fixed_parts:
        spc_lines: list[str] = []
        for part in fixed_parts:
            for node_id in part_node_ids[part.part_id]:
                spc_lines.append(
                    _ansys_common._kw_fixed_line(
                        [int(node_id), 0, 1, 1, 1, 1, 1, 1],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                    )
                )
        _ansys_common._append_keyword_block(deck, "*BOUNDARY_SPC_NODE", spc_lines)

    deck.append("*END", check=True)
    deck.export_file(str(deck_path))

    if content_part_initial_q_cm is None:
        content_part_initial_q_cm = _initial_lift_content_body_q_cm(parts[0].vertices_m * 100.0)

    return _ansys_common.DeckMetadata(
        deck_path=deck_path,
        bag_part_id=_ansys_common._BAG_PART_ID,
        bag_faces=parts[0].faces.astype(np.int32),
        bag_node_ids=part_node_ids[_ansys_common._BAG_PART_ID].copy(),
        bag_node_count=bag_node_count,
        bag_element_count=bag_elem_count,
        content_part_ids={
            "sphere": _ansys_common._SPHERE_PART_ID,
            "box": _ansys_common._BOX_PART_ID,
            "capsule": _ansys_common._CAPSULE_PART_ID,
        },
        content_part_node_ids={
            label: part_node_ids[part_id].copy()
            for label, part_id in {
                "sphere": _ansys_common._SPHERE_PART_ID,
                "box": _ansys_common._BOX_PART_ID,
                "capsule": _ansys_common._CAPSULE_PART_ID,
            }.items()
        },
        content_part_initial_q_cm={
            label: np.asarray(pose, dtype=np.float32).copy() for label, pose in content_part_initial_q_cm.items()
        },
    )


def _slice_motion_samples(
    motion: _ansys_common.MotionSamples,
    start_frame_index: int,
    end_frame_exclusive: int | None = None,
) -> _ansys_common.MotionSamples:
    """Return one motion slice with its local time reset to zero."""
    start_frame_index = int(np.clip(start_frame_index, 0, len(motion.times_s) - 1))
    if end_frame_exclusive is None:
        end_frame_exclusive = len(motion.times_s)
    end_frame_exclusive = int(np.clip(end_frame_exclusive, start_frame_index + 1, len(motion.times_s)))
    start_time_s = float(motion.times_s[start_frame_index])
    return _ansys_common.MotionSamples(
        times_s=(motion.times_s[start_frame_index:end_frame_exclusive] - start_time_s).astype(np.float32),
        robot_body_q_cm=motion.robot_body_q_cm[start_frame_index:end_frame_exclusive].astype(np.float32).copy(),
        left_pad_centers_m=motion.left_pad_centers_m[start_frame_index:end_frame_exclusive].astype(np.float32).copy(),
        right_pad_centers_m=motion.right_pad_centers_m[start_frame_index:end_frame_exclusive].astype(np.float32).copy(),
        left_pad_quat_xyzw=motion.left_pad_quat_xyzw[start_frame_index:end_frame_exclusive].astype(np.float32).copy(),
        right_pad_quat_xyzw=motion.right_pad_quat_xyzw[start_frame_index:end_frame_exclusive].astype(np.float32).copy(),
    )


def _build_shell_parts_from_state(
    bag_verts_cm: np.ndarray,
    bag_faces: np.ndarray,
    motion: _ansys_common.MotionSamples,
    content_part_initial_q_cm: dict[str, np.ndarray],
    *,
    small_pad: bool = False,
) -> list[_ansys_common.ShellPart]:
    """Create one LS-DYNA shell scene from an approximate restart state."""
    bag_verts_m = (bag_verts_cm * 0.01).astype(np.float32)
    bag_faces = bag_faces.astype(np.int32)

    default_content_q_cm = _initial_lift_content_body_q_cm(bag_verts_cm)
    content_q_cm = {
        label: np.asarray(
            content_part_initial_q_cm.get(label, default_content_q_cm[label]),
            dtype=np.float32,
        ).copy()
        for label in default_content_q_cm
    }

    sphere_vertices, sphere_faces = _ansys_common._make_sphere_mesh(
        0.04,
        content_q_cm["sphere"][:3].astype(np.float32) * 0.01,
    )
    box_vertices, box_faces = _ansys_common._make_box_mesh(
        (0.06, 0.06, 0.06),
        content_q_cm["box"][:3].astype(np.float32) * 0.01,
        content_q_cm["box"][3:].astype(np.float32),
    )
    capsule_vertices, capsule_faces = _ansys_common._make_capsule_mesh(
        radius_m=0.03,
        cylindrical_length_m=0.04,
        center_m=content_q_cm["capsule"][:3].astype(np.float32) * 0.01,
        quat_xyzw=content_q_cm["capsule"][3:].astype(np.float32),
    )
    ground_vertices, ground_faces = _ansys_common._make_box_mesh(
        extents_m=tuple(float(v) * 2.0 for v in _ansys_common._GROUND_HALF_EXTENTS_M),
        center_m=_ansys_common._GROUND_CENTER_M,
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    pad_hx_cm, pad_hy_cm, pad_hz_cm = _ansys_common._finger_pad_half_extents_cm(
        bag_verts_cm,
        small_pad=small_pad,
    )
    left_pad_vertices, left_pad_faces = _ansys_common._make_box_mesh(
        extents_m=(
            2.0 * pad_hx_cm * 0.01,
            2.0 * pad_hy_cm * 0.01,
            2.0 * pad_hz_cm * 0.01,
        ),
        center_m=motion.left_pad_centers_m[0],
        quat_xyzw=motion.left_pad_quat_xyzw[0].astype(np.float32),
    )
    right_pad_vertices, right_pad_faces = _ansys_common._make_box_mesh(
        extents_m=(
            2.0 * pad_hx_cm * 0.01,
            2.0 * pad_hy_cm * 0.01,
            2.0 * pad_hz_cm * 0.01,
        ),
        center_m=motion.right_pad_centers_m[0],
        quat_xyzw=motion.right_pad_quat_xyzw[0].astype(np.float32),
    )

    object_mass_kg = _ansys_common._OBJECT_MASS_G * 1.0e-3
    sphere_density = _ansys_common._shell_density_for_mass(
        sphere_vertices,
        sphere_faces,
        _ansys_common._RIGID_OBJECT_THICKNESS_M,
        object_mass_kg,
    )
    box_density = _ansys_common._shell_density_for_mass(
        box_vertices,
        box_faces,
        _ansys_common._RIGID_OBJECT_THICKNESS_M,
        object_mass_kg,
    )
    capsule_density = _ansys_common._shell_density_for_mass(
        capsule_vertices,
        capsule_faces,
        _ansys_common._RIGID_OBJECT_THICKNESS_M,
        object_mass_kg,
    )

    left_disp = motion.left_pad_centers_m - motion.left_pad_centers_m[[0]]
    right_disp = motion.right_pad_centers_m - motion.right_pad_centers_m[[0]]
    left_rot_rad = _ansys_common._relative_euler_xyz_rad(motion.left_pad_quat_xyzw)
    right_rot_rad = _ansys_common._relative_euler_xyz_rad(motion.right_pad_quat_xyzw)

    return [
        _ansys_common.ShellPart(
            label="bag_shell",
            part_id=_ansys_common._BAG_PART_ID,
            section_id=_ansys_common._BAG_SECTION_ID,
            material_id=_ansys_common._BAG_MAT_ID,
            vertices_m=bag_verts_m,
            faces=bag_faces,
            thickness_m=_ansys_common._BAG_THICKNESS_M,
            density_kg_m3=_ansys_common._BAG_DENSITY_KG_M3,
            youngs_modulus_pa=_LIFT_BAG_YOUNGS_MODULUS_PA,
            poisson_ratio=_ansys_common._BAG_POISSON,
            rigid=False,
        ),
        _ansys_common.ShellPart(
            label="ground",
            part_id=_ansys_common._GROUND_PART_ID,
            section_id=_ansys_common._GROUND_SECTION_ID,
            material_id=_ansys_common._GROUND_MAT_ID,
            vertices_m=ground_vertices,
            faces=ground_faces,
            thickness_m=_ansys_common._GROUND_THICKNESS_M,
            density_kg_m3=7800.0,
            youngs_modulus_pa=_ansys_common._RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_ansys_common._RIGID_POISSON,
            rigid=True,
            fixed=True,
        ),
        _ansys_common.ShellPart(
            label="left_finger_pad",
            part_id=_ansys_common._LEFT_PAD_PART_ID,
            section_id=_ansys_common._LEFT_PAD_SECTION_ID,
            material_id=_ansys_common._LEFT_PAD_MAT_ID,
            vertices_m=left_pad_vertices,
            faces=left_pad_faces,
            thickness_m=_ansys_common._PAD_THICKNESS_M,
            density_kg_m3=7800.0,
            youngs_modulus_pa=_ansys_common._RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_ansys_common._RIGID_POISSON,
            rigid=True,
            prescribed_displacement_m=left_disp.astype(np.float32),
            prescribed_rotation_rad=left_rot_rad.astype(np.float32),
        ),
        _ansys_common.ShellPart(
            label="right_finger_pad",
            part_id=_ansys_common._RIGHT_PAD_PART_ID,
            section_id=_ansys_common._RIGHT_PAD_SECTION_ID,
            material_id=_ansys_common._RIGHT_PAD_MAT_ID,
            vertices_m=right_pad_vertices,
            faces=right_pad_faces,
            thickness_m=_ansys_common._PAD_THICKNESS_M,
            density_kg_m3=7800.0,
            youngs_modulus_pa=_ansys_common._RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_ansys_common._RIGID_POISSON,
            rigid=True,
            prescribed_displacement_m=right_disp.astype(np.float32),
            prescribed_rotation_rad=right_rot_rad.astype(np.float32),
        ),
        _ansys_common.ShellPart(
            label="sphere",
            part_id=_ansys_common._SPHERE_PART_ID,
            section_id=_ansys_common._SPHERE_SECTION_ID,
            material_id=_ansys_common._SPHERE_MAT_ID,
            vertices_m=sphere_vertices,
            faces=sphere_faces,
            thickness_m=_ansys_common._RIGID_OBJECT_THICKNESS_M,
            density_kg_m3=sphere_density,
            youngs_modulus_pa=_ansys_common._RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_ansys_common._RIGID_POISSON,
            rigid=True,
        ),
        _ansys_common.ShellPart(
            label="box",
            part_id=_ansys_common._BOX_PART_ID,
            section_id=_ansys_common._BOX_SECTION_ID,
            material_id=_ansys_common._BOX_MAT_ID,
            vertices_m=box_vertices,
            faces=box_faces,
            thickness_m=_ansys_common._RIGID_OBJECT_THICKNESS_M,
            density_kg_m3=box_density,
            youngs_modulus_pa=_ansys_common._RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_ansys_common._RIGID_POISSON,
            rigid=True,
        ),
        _ansys_common.ShellPart(
            label="capsule",
            part_id=_ansys_common._CAPSULE_PART_ID,
            section_id=_ansys_common._CAPSULE_SECTION_ID,
            material_id=_ansys_common._CAPSULE_MAT_ID,
            vertices_m=capsule_vertices,
            faces=capsule_faces,
            thickness_m=_ansys_common._RIGID_OBJECT_THICKNESS_M,
            density_kg_m3=capsule_density,
            youngs_modulus_pa=_ansys_common._RIGID_YOUNGS_MODULUS_PA,
            poisson_ratio=_ansys_common._RIGID_POISSON,
            rigid=True,
        ),
    ]


def _truncate_replay(
    replay: _ansys_common.ReplayData,
    frame_count: int,
) -> _ansys_common.ReplayData:
    """Keep only the first `frame_count` replay frames."""
    frame_count = max(0, int(frame_count))
    return _ansys_common.ReplayData(
        times_s=replay.times_s[:frame_count].astype(np.float32).copy(),
        bag_points_m=replay.bag_points_m[:frame_count].astype(np.float32).copy(),
        bag_von_mises=replay.bag_von_mises[:frame_count].astype(np.float32).copy(),
        rigid_body_q_cm={
            label: values[:frame_count].astype(np.float32).copy() for label, values in replay.rigid_body_q_cm.items()
        },
    )


def _offset_replay_times(
    replay: _ansys_common.ReplayData,
    time_offset_s: float,
) -> _ansys_common.ReplayData:
    """Shift one replay's time base without changing its frame payloads."""
    return _ansys_common.ReplayData(
        times_s=(replay.times_s + float(time_offset_s)).astype(np.float32),
        bag_points_m=replay.bag_points_m,
        bag_von_mises=replay.bag_von_mises,
        rigid_body_q_cm=replay.rigid_body_q_cm,
    )


def _content_body_q_cm_for_frame(
    replay: _ansys_common.ReplayData,
    frame_index: int,
    bag_verts_cm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Approximate one restart state from the last replay frame."""
    content_q_cm = _initial_lift_content_body_q_cm(bag_verts_cm)
    for label, values in replay.rigid_body_q_cm.items():
        if len(values) > frame_index:
            content_q_cm[label] = values[frame_index].astype(np.float32).copy()
    return content_q_cm


class Example(_ansys_common.Example):
    """Replay an LS-DYNA shell bag lift solve inside Newton."""

    @staticmethod
    def create_parser():
        parser = _ansys_common.Example.create_parser()
        parser.description = (
            "Replay an LS-DYNA shell bag lift inside Newton. This rollback "
            "variant keeps the same world-time robot motion, but on solver "
            "failure it can approximately restart from the previous "
            "successful "
            "replay frame with a smaller LS-DYNA TSSFAC."
        )
        for action in parser._actions:
            if action.dest == "job_dir":
                action.default = str(_DEFAULT_JOB_DIR)
                break
        _add_lift_robot_arguments(parser, small_pad=False)
        parser.add_argument(
            "--no-rollback",
            dest="rollback",
            action="store_false",
            default=True,
            help=("Stop after the first failed LS-DYNA attempt instead of retrying from accepted replay frames."),
        )
        parser.add_argument(
            "--rollback-max-retries",
            type=int,
            default=_DEFAULT_ROLLBACK_MAX_RETRIES,
            help=(
                "Maximum number of approximate rollback retries after a "
                "failed LS-DYNA attempt. Set 0 to disable rollback and "
                "run only once."
            ),
        )
        parser.add_argument(
            "--rollback-backtrack-frames",
            type=int,
            default=_DEFAULT_ROLLBACK_BACKTRACK_FRAMES,
            help=(
                "How many already accepted replay frames to step back before "
                "a retry. The default 0 keeps the last accepted frame and "
                "retries only the next missing frame."
            ),
        )
        parser.add_argument(
            "--rollback-base-tssfac",
            type=float,
            default=_DEFAULT_TSSFAC,
            help=(
                "Initial LS-DYNA TSSFAC for the first attempt. Smaller "
                "values force finer internal explicit steps without "
                "changing world time."
            ),
        )
        parser.add_argument(
            "--rollback-tssfac-scale",
            type=float,
            default=_DEFAULT_ROLLBACK_TSSFAC_SCALE,
            help=(
                "Multiplier applied to TSSFAC on each retry, e.g. 0.75 turns 0.90 into 0.675, then 0.506, and so on."
            ),
        )
        parser.add_argument(
            "--rollback-min-tssfac",
            type=float,
            default=_DEFAULT_ROLLBACK_MIN_TSSFAC,
            help="Lower bound applied to retry TSSFAC values.",
        )
        return parser

    def __init__(self, viewer, args):
        closed_width_cm = float(args.closed_width_cm)
        rollback_enabled = bool(getattr(args, "rollback", True))
        rollback_max_retries = int(args.rollback_max_retries)
        if not rollback_enabled:
            rollback_max_retries = 0
        rollback_backtrack_frames = int(args.rollback_backtrack_frames)
        rollback_base_tssfac = float(args.rollback_base_tssfac)
        rollback_tssfac_scale = float(args.rollback_tssfac_scale)
        rollback_min_tssfac = float(args.rollback_min_tssfac)

        if rollback_max_retries < 0:
            raise ValueError("`--rollback-max-retries` must be non-negative.")
        if rollback_backtrack_frames < 0:
            raise ValueError("`--rollback-backtrack-frames` must be non-negative.")
        if not np.isfinite(rollback_base_tssfac) or rollback_base_tssfac <= 0.0:
            raise ValueError("`--rollback-base-tssfac` must be positive.")
        if not np.isfinite(rollback_tssfac_scale) or rollback_tssfac_scale <= 0.0:
            raise ValueError("`--rollback-tssfac-scale` must be positive.")
        if not np.isfinite(rollback_min_tssfac) or rollback_min_tssfac <= 0.0:
            raise ValueError("`--rollback-min-tssfac` must be positive.")

        print(f"{_ansys_common._LOG_PREFIX} Friction grip final finger gap: {closed_width_cm:.2f} cm")
        print(
            f"{_ansys_common._LOG_PREFIX} Lift bag shell Young's modulus: "
            f"{_LIFT_BAG_YOUNGS_MODULUS_PA * 1.0e-9:.2f} GPa"
        )
        print(
            f"{_ansys_common._LOG_PREFIX} Lift contact friction: "
            f"static={_LIFT_CONTACT_FS:.2f} dynamic={_LIFT_CONTACT_FD:.2f}"
        )
        if rollback_max_retries > 0:
            print(
                f"{_ansys_common._LOG_PREFIX} Approximate rollback enabled: "
                f"retries={rollback_max_retries} "
                f"backtrack={rollback_backtrack_frames} "
                f"base_tssfac={rollback_base_tssfac:.3f} "
                f"scale={rollback_tssfac_scale:.3f} "
                f"min_tssfac={rollback_min_tssfac:.3f}"
            )
        else:
            print(
                f"{_ansys_common._LOG_PREFIX} Approximate rollback disabled; "
                "LS-DYNA will not be restarted after a failed attempt."
            )

        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.target_frame_dt = float(args.output_dt)
        self._full_capture_frames = _ansys_common._full_capture_frame_count(self.target_frame_dt)
        requested_capture_frames = int(args.capture_frames)
        self.capture_frames = (
            self._full_capture_frames
            if requested_capture_frames == _ansys_common._DEFAULT_NUM_FRAMES
            else requested_capture_frames
        )
        self._frame_index = -1
        self._max_bag_top_z_cm = 0.0
        self._capture_duration_s = _ansys_common._duration_from_capture_frames(
            self.capture_frames,
            self.target_frame_dt,
        )
        self._full_replay_requested = self._capture_duration_s >= (
            _ansys_common._TOTAL_DURATION_S - 0.5 * self.target_frame_dt
        )
        _ansys_common._configure_capture_common(
            self,
            capture_replay=bool(getattr(args, "capture_replay", False)),
            capture_frames=self.capture_frames,
            capture_fps=int(getattr(args, "capture_fps", 60)),
            capture_dir=str(getattr(args, "capture_dir", "outputs/replay_capture")),
            capture_format=str(getattr(args, "capture_format", "mp4")),
        )
        self._stream_mode = False
        self._source_replay: _ansys_common.ReplayData | None = None
        self._target_frame_times_s = np.empty(0, dtype=np.float32)
        self._tail_hold_warned = False
        self._stream_solver_process = None
        self._stream_solver_log_file = None
        self._stream_solver_log_path: Path | None = None
        self._stream_solver_finished = False
        self._stream_solver_exit_code: int | None = None
        self._stream_complete = False
        self._stream_last_d3plot_signature = None
        self._stream_next_poll_time_s = 0.0
        self._last_replay_diagnostics: _ansys_common.ReplayLoadDiagnostics | None = None
        self._stream_last_replay_read_error = ""
        self.small_pad = bool(getattr(args, "small_pad", False))
        self._rollback_enabled = rollback_enabled
        self._rollback_max_retries = rollback_max_retries
        self._rollback_backtrack_frames = rollback_backtrack_frames
        self._rollback_base_tssfac = rollback_base_tssfac
        self._rollback_tssfac_scale = rollback_tssfac_scale
        self._rollback_min_tssfac = rollback_min_tssfac
        self._rollback_target_frame_count = 0
        self._rollback_shell_faces: np.ndarray | None = None
        self._rollback_solver_executable: Path | None = None
        self._rollback_ncpu = int(args.ncpu)
        self._rollback_memory = str(args.memory)
        self._rollback_summary: dict[str, object] | None = None
        self._rollback_active_attempt_record: dict[str, object] | None = None
        self._rollback_active_retry_event: dict[str, object] | None = None
        self._rollback_pending_retry_info: dict[str, object] | None = None
        self._rollback_current_attempt_index = -1
        self._rollback_current_attempt_mode = "base"
        self._rollback_current_rescue_depth = 0
        self._rollback_current_attempt_dir: Path | None = None
        self._rollback_current_time_offset_s = 0.0

        full_verts_cm, full_faces = _load_lift_kfc_mesh_zup()
        self.job_dir = Path(args.job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lsdyna_debug_summary_path = _ansys_common._lsdyna_debug_summary_path(self.job_dir)
        self.rollback_summary_path = _rollback_summary_path(self.job_dir)
        self._stream_mode = True
        _ansys_common.require_lasso()
        _ansys_common._disable_viewer_frame_limit_for_streaming(self.viewer)
        print(f"{_ansys_common._LOG_PREFIX} LS-DYNA debug summary: {self._lsdyna_debug_summary_path.resolve()}")
        print(f"{_ansys_common._LOG_PREFIX} Rollback summary: {self.rollback_summary_path.resolve()}")

        shell_verts_cm, shell_faces = _ansys_common._decimate_mesh(
            full_verts_cm,
            full_faces,
            int(args.target_faces),
        )

        if not self._full_replay_requested or self.capture_frames != self._full_capture_frames:
            print(
                f"{_ansys_common._LOG_PREFIX} Requested {self.capture_frames} "
                f"replay frames ({self._capture_duration_s:.3f}s at "
                f"output_dt={self.target_frame_dt:.5f}s). "
                f"Full replay is {self._full_capture_frames} frames "
                f"({_ansys_common._TOTAL_DURATION_S:.3f}s)."
            )

        pad_hx_cm, _, pad_hz_cm = _ansys_common._finger_pad_half_extents_cm(
            shell_verts_cm,
            small_pad=self.small_pad,
        )
        print(
            f"{_ansys_common._LOG_PREFIX} Finger pad: hx={pad_hx_cm:.2f} "
            f"hz={pad_hz_cm:.2f} cm  "
            f"{'(small-pad ~20% area)' if self.small_pad else '(full pad)'}"
        )

        self.motion = LiftRobotTrajectorySampler(
            shell_verts_cm,
            target_frame_dt_s=self.target_frame_dt,
            small_pad=self.small_pad,
            closed_width_cm=closed_width_cm,
        ).sample(self._capture_duration_s)
        self._target_frame_times_s = self.motion.times_s[: self.capture_frames].astype(np.float32).copy()
        self._rollback_target_frame_count = self._active_target_frame_count()

        solver_exe = _ansys_common._find_lsdyna_executable(
            args.lsdyna_exe,
            Path(args.lsdyna_root),
        )
        self._rollback_solver_executable = solver_exe
        self._rollback_shell_faces = shell_faces.astype(np.int32).copy()
        print(f"{_ansys_common._LOG_PREFIX} Running LS-DYNA: {solver_exe}")
        if self._rollback_enabled and self._rollback_max_retries > 0:
            print(f"{_ansys_common._LOG_PREFIX} Rollback attempt outputs are stored under `{self.job_dir}`.")
        else:
            print(f"{_ansys_common._LOG_PREFIX} Single-attempt outputs are stored under `{self.job_dir}`.")
        self._initialize_rollback_summary()
        self._start_rollback_attempt(
            attempt_index=0,
            restart_frame_index=0,
            attempt_mode="base",
            rescue_depth=0,
            restart_bag_verts_cm=shell_verts_cm.astype(np.float32).copy(),
            content_part_initial_q_cm=_initial_lift_content_body_q_cm(shell_verts_cm),
        )
        self._write_lsdyna_debug_summary(
            note=(
                "LS-DYNA launched in streaming rollback mode."
                if self._rollback_enabled and self._rollback_max_retries > 0
                else "LS-DYNA launched in single-attempt streaming mode."
            ),
        )

        (
            self._bary_vi0_np,
            self._bary_vi1_np,
            self._bary_vi2_np,
            self._bary_w_np,
        ) = _ansys_common._build_bary_map(
            full_verts_cm,
            shell_verts_cm,
            self.deck.bag_faces,
        )
        self._n_full_verts = len(full_verts_cm)
        self._full_indices_wp = wp.array(
            full_faces.flatten().astype(np.int32),
            dtype=wp.int32,
        )
        bary_proj_cm = (
            shell_verts_cm[self._bary_vi0_np] * self._bary_w_np[:, 0:1]
            + shell_verts_cm[self._bary_vi1_np] * self._bary_w_np[:, 1:2]
            + shell_verts_cm[self._bary_vi2_np] * self._bary_w_np[:, 2:3]
        )
        self._bary_disp_m = ((full_verts_cm - bary_proj_cm) * _ansys_common._VIZ_SCALE).astype(np.float32)

        self._source_replay = _ansys_common._initial_replay_data(
            shell_verts_cm,
            self.deck,
        )
        self._update_replay_from_source(hold_last=False)
        self._persist_rollback_summary()
        self._write_lsdyna_debug_summary(note="Initial replay buffers prepared.")

        self.model, self._content_body_indices, self._robot_body_count = _ansys_common._build_visual_model(
            shell_verts_cm,
            small_pad=self.small_pad,
        )
        self._initial_body_q_cm = self.model.body_q.numpy().copy().astype(np.float32)
        self.viz_state = self.model.state()
        self.state_0 = self.viz_state
        self._bag_indices_wp = wp.array(
            self.deck.bag_faces.flatten().astype(np.int32),
            dtype=wp.int32,
        )
        self._proxy_indices_wp = self._bag_indices_wp
        self._bag_face0 = self.deck.bag_faces[:, 0]
        self._bag_face1 = self.deck.bag_faces[:, 1]
        self._bag_face2 = self.deck.bag_faces[:, 2]
        self._stress_radii_wp = wp.full(
            len(self.deck.bag_faces),
            float(_ansys_common._STRESS_POINT_RADIUS_M),
            dtype=wp.float32,
            device=self.model.device,
        )

        shape_xf = self.model.shape_transform.numpy().copy()
        shape_xf[:, :3] *= _ansys_common._VIZ_SCALE
        self.model.shape_transform = wp.array(
            shape_xf,
            dtype=wp.transform,
            device=self.model.device,
        )
        shape_sc = self.model.shape_scale.numpy().copy()
        shape_sc *= _ansys_common._VIZ_SCALE
        self.model.shape_scale = wp.array(
            shape_sc,
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_camera(
                pos=wp.vec3(1.0, -1.0, 0.8),
                pitch=-10.0,
                yaw=135.0,
            )
        if self._stream_mode:
            self._poll_streaming_replay(force=True)

    def _active_target_frame_count(self) -> int:
        """Return the number of accepted replay frames this run needs."""
        return min(self.capture_frames, len(self.motion.times_s))

    def _persist_rollback_summary(self) -> None:
        """Write the current rollback summary to disk."""
        if self._rollback_summary is None:
            return
        replay_frame_count = 0
        if hasattr(self, "replay"):
            replay_frame_count = len(self.replay.times_s)
        elif self._source_replay is not None:
            replay_frame_count = len(self._source_replay.times_s)
        self._rollback_summary["final_replay_frame_count"] = int(replay_frame_count)
        _write_rollback_summary(self.rollback_summary_path, self._rollback_summary)

    def _initialize_rollback_summary(self) -> None:
        """Initialize the persistent rollback summary for one run."""
        self._rollback_summary = {
            "status": "running",
            "completed": False,
            "job_dir": str(self.job_dir),
            "summary_path": str(self.rollback_summary_path),
            "target_frame_count": int(self._active_target_frame_count()),
            "output_dt_s": float(self.target_frame_dt),
            "base_tssfac": float(self._rollback_base_tssfac),
            "tssfac_scale": float(self._rollback_tssfac_scale),
            "min_tssfac": float(self._rollback_min_tssfac),
            "rollback_enabled": bool(self._rollback_enabled),
            "max_retries": int(self._rollback_max_retries),
            "backtrack_frames": int(self._rollback_backtrack_frames),
            "rescue_policy": "single_frame_then_resume_base",
            "attempt_count": 0,
            "retry_event_count": 0,
            "rollback_event_count": 0,
            "final_replay_frame_count": 0,
            "attempts": [],
            "rollback_events": [],
        }
        self._persist_rollback_summary()

    def _register_rollback_attempt(
        self,
        *,
        attempt_index: int,
        attempt_dir: Path,
        restart_frame_index: int,
        attempt_mode: str,
        rescue_depth: int,
        tssfac: float,
    ) -> None:
        """Append one attempt record to the rollback summary."""
        if self._rollback_summary is None:
            return

        accepted_frame_count = len(self.replay.times_s) if hasattr(self, "replay") else 0
        had_committed_prefix = bool(accepted_frame_count > 0)
        attempt_record: dict[str, object] = {
            "attempt_index": int(attempt_index),
            "attempt_dir": attempt_dir.name,
            "attempt_mode": str(attempt_mode),
            "rescue_depth": int(rescue_depth),
            "had_committed_prefix": had_committed_prefix,
            "accepted_frame_count_before_attempt": int(accepted_frame_count),
            "restart_frame_index": int(restart_frame_index),
            "restart_frame_number": int(restart_frame_index + 1),
            "restart_time_s": float(self.motion.times_s[restart_frame_index]),
            "tssfac": float(tssfac),
        }
        self._rollback_summary["attempts"].append(attempt_record)
        self._rollback_summary["attempt_count"] = len(self._rollback_summary["attempts"])
        self._rollback_active_attempt_record = attempt_record
        self._rollback_active_retry_event = None

        if attempt_mode == "rescue":
            retry_event: dict[str, object] = {
                "retry_index": int(len(self._rollback_summary["rollback_events"]) + 1),
                "attempt_index": int(attempt_index),
                "attempt_dir": attempt_dir.name,
                "had_committed_prefix": had_committed_prefix,
                "rescue_depth": int(rescue_depth),
                "retry_mode": ("rollback" if had_committed_prefix else "restart_from_start"),
                "restart_frame_index": int(restart_frame_index),
                "restart_frame_number": int(restart_frame_index + 1),
                "restart_time_s": float(self.motion.times_s[restart_frame_index]),
                "tssfac": float(tssfac),
            }
            if self._rollback_pending_retry_info is not None:
                retry_event.update(self._rollback_pending_retry_info)
                self._rollback_pending_retry_info = None
            self._rollback_summary["rollback_events"].append(retry_event)
            self._rollback_summary["retry_event_count"] = len(self._rollback_summary["rollback_events"])
            self._rollback_summary["rollback_event_count"] = sum(
                1
                for event in self._rollback_summary["rollback_events"]
                if isinstance(event, dict) and bool(event.get("had_committed_prefix", False))
            )
            self._rollback_active_retry_event = retry_event

        self._persist_rollback_summary()

    def _close_stream_solver_log_file(self) -> None:
        """Close the active LS-DYNA stdout log file handle."""
        if self._stream_solver_log_file is not None and not self._stream_solver_log_file.closed:
            self._stream_solver_log_file.close()
        self._stream_solver_log_file = None

    def _start_rollback_attempt(
        self,
        *,
        attempt_index: int,
        restart_frame_index: int,
        attempt_mode: str,
        rescue_depth: int,
        restart_bag_verts_cm: np.ndarray,
        content_part_initial_q_cm: dict[str, np.ndarray],
    ) -> None:
        """Launch one rollback attempt in streaming LS-DYNA mode."""
        if self._rollback_shell_faces is None:
            raise RuntimeError("Rollback shell faces have not been initialized.")
        if self._rollback_solver_executable is None:
            raise RuntimeError("Rollback solver executable is not configured.")

        attempt_dir = self.job_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _ansys_common._cleanup_lsdyna_outputs(attempt_dir)

        if attempt_mode == "rescue":
            tssfac = max(
                float(self._rollback_min_tssfac),
                float(self._rollback_base_tssfac) * (float(self._rollback_tssfac_scale) ** int(rescue_depth)),
            )
            end_frame_exclusive = min(
                restart_frame_index + 2,
                self._active_target_frame_count(),
            )
        else:
            tssfac = float(self._rollback_base_tssfac)
            end_frame_exclusive = self._active_target_frame_count()

        attempt_motion = _slice_motion_samples(
            self.motion,
            restart_frame_index,
            end_frame_exclusive,
        )
        attempt_parts = _build_shell_parts_from_state(
            restart_bag_verts_cm,
            self._rollback_shell_faces,
            attempt_motion,
            content_part_initial_q_cm,
            small_pad=self.small_pad,
        )
        self.deck = _write_keyword_deck_friction(
            attempt_dir / "input.k",
            attempt_parts,
            attempt_motion,
            self.target_frame_dt,
            tssfac=tssfac,
            content_part_initial_q_cm=content_part_initial_q_cm,
        )

        self._rollback_current_attempt_index = int(attempt_index)
        self._rollback_current_attempt_mode = str(attempt_mode)
        self._rollback_current_rescue_depth = int(rescue_depth)
        self._rollback_current_attempt_dir = attempt_dir
        self._rollback_current_time_offset_s = float(self.motion.times_s[restart_frame_index])
        self._stream_last_d3plot_signature = None
        self._stream_next_poll_time_s = 0.0
        self._stream_solver_finished = False
        self._stream_solver_exit_code = None
        self._stream_last_replay_read_error = ""
        self._tail_hold_warned = False
        self._register_rollback_attempt(
            attempt_index=attempt_index,
            attempt_dir=attempt_dir,
            restart_frame_index=restart_frame_index,
            attempt_mode=attempt_mode,
            rescue_depth=rescue_depth,
            tssfac=tssfac,
        )

        if attempt_index == 0 and attempt_mode == "base":
            if self._rollback_enabled and self._rollback_max_retries > 0:
                print(
                    f"{_ansys_common._LOG_PREFIX} Rollback solve base "
                    f"TSSFAC={tssfac:.3f} "
                    f"(max retries={self._rollback_max_retries}, "
                    f"backtrack={self._rollback_backtrack_frames} frame(s))"
                )
            else:
                print(f"{_ansys_common._LOG_PREFIX} LS-DYNA single attempt TSSFAC={tssfac:.3f} (rollback disabled)")
        elif attempt_mode == "rescue":
            rescue_target_frame_number = min(
                restart_frame_index + 2,
                self._active_target_frame_count(),
            )
            print(
                f"{_ansys_common._LOG_PREFIX} Rollback retry "
                f"{rescue_depth}/{self._rollback_max_retries}: restart frame "
                f"{restart_frame_index + 1}/{self._active_target_frame_count()} at "
                f"t={self.motion.times_s[restart_frame_index]:.3f}s "
                f"to rescue frame {rescue_target_frame_number}/"
                f"{self._active_target_frame_count()} with TSSFAC={tssfac:.3f}"
            )
        else:
            print(
                f"{_ansys_common._LOG_PREFIX} Resuming base TSSFAC={tssfac:.3f} "
                f"from frame "
                f"{restart_frame_index + 1}/{self._active_target_frame_count()} at "
                f"t={self.motion.times_s[restart_frame_index]:.3f}s"
            )

        (
            self._stream_solver_process,
            self._stream_solver_log_file,
            self._stream_solver_log_path,
        ) = _ansys_common._start_lsdyna_background(
            deck_path=self.deck.deck_path,
            executable=self._rollback_solver_executable,
            job_dir=attempt_dir,
            ncpu=self._rollback_ncpu,
            memory=self._rollback_memory,
        )

    def _record_current_attempt_outcome(
        self,
        *,
        exit_code: int,
        normal_termination: bool,
        exit_observation: str,
        final_append_count: int,
    ) -> None:
        """Update the rollback summary for the just-finished attempt."""
        if self._rollback_active_attempt_record is None:
            return

        restart_frame_index = int(self._rollback_active_attempt_record["restart_frame_index"])
        accepted_frame_count = len(self.replay.times_s)
        appended_frame_count = max(
            accepted_frame_count - (restart_frame_index + 1),
            0,
        )
        self._rollback_active_attempt_record["return_code"] = int(exit_code)
        self._rollback_active_attempt_record["exit_code"] = int(exit_code)
        self._rollback_active_attempt_record["normal_termination"] = bool(normal_termination)
        self._rollback_active_attempt_record["exit_observation"] = str(exit_observation)
        self._rollback_active_attempt_record["aligned_frame_count"] = int(appended_frame_count)
        self._rollback_active_attempt_record["final_append_count"] = int(final_append_count)
        self._rollback_active_attempt_record["merged_frame_count_after_attempt"] = int(accepted_frame_count)
        self._rollback_active_attempt_record["completed_full_replay"] = bool(
            accepted_frame_count >= self._active_target_frame_count()
        )

        if self._rollback_active_retry_event is not None:
            self._rollback_active_retry_event["return_code"] = int(exit_code)
            self._rollback_active_retry_event["exit_code"] = int(exit_code)
            self._rollback_active_retry_event["normal_termination"] = bool(normal_termination)
            self._rollback_active_retry_event["exit_observation"] = str(exit_observation)
            self._rollback_active_retry_event["aligned_frame_count"] = int(appended_frame_count)
            self._rollback_active_retry_event["final_append_count"] = int(final_append_count)
            self._rollback_active_retry_event["merged_frame_count_after_attempt"] = int(accepted_frame_count)

        self._persist_rollback_summary()

    def _limit_rescue_replay_update(
        self,
        update: _ansys_common.ReplayData,
    ) -> _ansys_common.ReplayData:
        """Keep a one-frame rescue limited to restart plus rescued state."""
        if self._rollback_current_attempt_mode != "rescue":
            return update
        if len(update.times_s) <= 2:
            return update
        if self._rollback_active_attempt_record is not None:
            self._rollback_active_attempt_record["rescue_report_limit"] = 2
            self._rollback_active_attempt_record["discarded_rescue_report_count"] = int(len(update.times_s) - 2)
        if self._rollback_active_retry_event is not None:
            self._rollback_active_retry_event["rescue_report_limit"] = 2
            self._rollback_active_retry_event["discarded_rescue_report_count"] = int(len(update.times_s) - 2)
        return _truncate_replay(update, 2)

    def _append_final_attempt_replay_update(self, attempt_dir: Path) -> int:
        """Append any remaining readable frames after one attempt stops."""
        try:
            update, self._last_replay_diagnostics = _ansys_common._load_replay_data_with_diagnostics(
                attempt_dir,
                self.deck,
                output_dt_s=self.target_frame_dt,
            )
            self._stream_last_replay_read_error = ""
        except Exception as exc:
            self._stream_last_replay_read_error = str(exc)
            self._write_lsdyna_debug_summary(
                note="Final replay read failed while closing a rollback attempt.",
            )
            raise RuntimeError(f"Failed to read final LS-DYNA d3plot output after the solver finished. {exc}") from exc

        update = self._limit_rescue_replay_update(update)
        update = _offset_replay_times(
            update,
            self._rollback_current_time_offset_s,
        )
        appended = self._append_streaming_replay(update)
        if appended > 0:
            total_frames = self._active_target_frame_count()
            print(f"{_ansys_common._LOG_PREFIX} Received {len(self.replay.times_s)}/{total_frames} replay frames")
            self._persist_rollback_summary()
        return int(appended)

    @staticmethod
    def _describe_attempt_exit(
        exit_code: int,
        normal_termination: bool,
    ) -> str:
        """Classify the solver exit for rollback logging."""
        if exit_code == 0 and normal_termination:
            return "clean_exit_after_normal_termination"
        if exit_code == 0:
            return "clean_exit_without_normal_termination_marker"
        if normal_termination:
            return "nonzero_exit_after_normal_termination"
        return "nonzero_exit_before_normal_termination"

    def _trim_capture_to_frame_count(self, frame_count: int) -> None:
        """Delete captured PNGs beyond the accepted replay prefix."""
        _ansys_common._trim_replay_capture_common(
            self,
            frame_count,
            target_frame_count=self._active_target_frame_count(),
        )

    def _prepare_retry_from_prefix(
        self,
    ) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
        """Freeze the accepted prefix and prepare the next retry anchor."""
        if len(self.replay.times_s) == 0:
            raise RuntimeError("Rollback cannot retry without an accepted frame.")

        restart_frame_index = max(
            0,
            len(self.replay.times_s) - 1 - int(self._rollback_backtrack_frames),
        )
        if self._rollback_active_attempt_record is not None:
            self._rollback_active_attempt_record["scheduled_next_restart_frame_index"] = int(restart_frame_index)
            self._rollback_active_attempt_record["scheduled_next_restart_frame_number"] = int(restart_frame_index + 1)
            self._rollback_active_attempt_record["scheduled_next_restart_time_s"] = float(
                self.motion.times_s[restart_frame_index]
            )

        accepted_prefix = _truncate_replay(self.replay, restart_frame_index + 1)
        self._source_replay = accepted_prefix
        self._update_replay_from_source(hold_last=False)
        self._frame_index = min(
            max(self._frame_index, 0),
            len(self.replay.times_s) - 1,
        )
        if self.capture_replay:
            self._trim_capture_to_frame_count(restart_frame_index + 1)

        restart_bag_verts_cm = (self.replay.bag_points_m[restart_frame_index] * 100.0).astype(np.float32)
        restart_content_q_cm = _content_body_q_cm_for_frame(
            self.replay,
            restart_frame_index,
            restart_bag_verts_cm,
        )
        self._persist_rollback_summary()
        return restart_frame_index, restart_bag_verts_cm, restart_content_q_cm

    def _mark_rollback_completed(self) -> None:
        """Finalize the rollback summary for a successful run."""
        if self._rollback_summary is None:
            return
        self._rollback_summary["status"] = "completed"
        self._rollback_summary["completed"] = True
        self._rollback_summary.pop("error", None)
        _finalize_rollback_summary_ranges(self._rollback_summary)
        self._persist_rollback_summary()

    def _mark_rollback_failed(self, error: str) -> None:
        """Finalize the rollback summary for a failed or interrupted run."""
        if self._rollback_summary is None:
            return
        if self._rollback_summary.get("status") == "completed":
            return
        self._rollback_summary["status"] = "failed"
        self._rollback_summary["completed"] = False
        self._rollback_summary["error"] = str(error)
        _finalize_rollback_summary_ranges(self._rollback_summary)
        self._persist_rollback_summary()

    def _finish_or_retry_current_attempt(self) -> None:
        """Process one finished LS-DYNA attempt and start a retry if needed."""
        current_attempt_dir = self._rollback_current_attempt_dir or self.job_dir
        log_path = self._stream_solver_log_path or (current_attempt_dir / "lsdyna.stdout.txt")
        self._close_stream_solver_log_file()

        normal_termination = _ansys_common._lsdyna_log_has_normal_termination(log_path)
        exit_code = 0 if self._stream_solver_exit_code in (None, 0) else int(self._stream_solver_exit_code)
        final_append_count = self._append_final_attempt_replay_update(current_attempt_dir)
        exit_observation = self._describe_attempt_exit(
            exit_code,
            normal_termination,
        )
        self._record_current_attempt_outcome(
            exit_code=exit_code,
            normal_termination=normal_termination,
            exit_observation=exit_observation,
            final_append_count=final_append_count,
        )

        accepted_frame_count = len(self.replay.times_s)
        target_frame_count = self._active_target_frame_count()
        accepted_frame_count_before_attempt = 0
        current_tssfac = float(self._rollback_base_tssfac)
        if self._rollback_active_attempt_record is not None:
            accepted_frame_count_before_attempt = int(
                self._rollback_active_attempt_record.get(
                    "accepted_frame_count_before_attempt",
                    0,
                )
            )
            current_tssfac = float(
                self._rollback_active_attempt_record.get(
                    "tssfac",
                    self._rollback_base_tssfac,
                )
            )
        produced_new_frames = accepted_frame_count > accepted_frame_count_before_attempt

        if accepted_frame_count >= target_frame_count:
            if exit_code != 0 and normal_termination:
                print(
                    f"{_ansys_common._LOG_PREFIX} Warning: LS-DYNA exited with code "
                    f"{exit_code} after normal termination, but the accepted "
                    "replay is complete."
                )
            self._stream_complete = True
            self._mark_rollback_completed()
            self._write_lsdyna_debug_summary(
                note="Streaming rollback reached the target replay frame count.",
            )
            return

        if self._rollback_current_attempt_mode == "rescue" and produced_new_frames:
            resume_message = (
                f"Rescue attempt {self._rollback_current_rescue_depth}/"
                f"{self._rollback_max_retries} accepted "
                f"{accepted_frame_count}/{target_frame_count} replay frames "
                f"with TSSFAC={current_tssfac:.3f}. "
                f"Resuming base TSSFAC={self._rollback_base_tssfac:.3f}."
            )
            if self._rollback_active_attempt_record is not None:
                self._rollback_active_attempt_record["post_attempt_action"] = "resume_base_tssfac"
                self._rollback_active_attempt_record["post_attempt_message"] = resume_message
            print(f"{_ansys_common._LOG_PREFIX} {resume_message}")
            self._persist_rollback_summary()
            (
                restart_frame_index,
                restart_bag_verts_cm,
                restart_content_q_cm,
            ) = self._prepare_retry_from_prefix()
            self._rollback_pending_retry_info = None
            self._start_rollback_attempt(
                attempt_index=self._rollback_current_attempt_index + 1,
                restart_frame_index=restart_frame_index,
                attempt_mode="base",
                rescue_depth=0,
                restart_bag_verts_cm=restart_bag_verts_cm,
                content_part_initial_q_cm=restart_content_q_cm,
            )
            self._write_lsdyna_debug_summary(
                note="Streaming rollback resumed base TSSFAC after a one-frame rescue.",
            )
            return

        if not self._rollback_enabled or self._rollback_max_retries <= 0:
            stop_message = (
                f"LS-DYNA attempt accepted {accepted_frame_count}/"
                f"{target_frame_count} replay frames and stopped with "
                f"{exit_observation} (exit_code={exit_code}). "
                "Rollback is disabled, so no retry will be launched."
            )
            if self._rollback_active_attempt_record is not None:
                self._rollback_active_attempt_record["post_attempt_action"] = "stop_no_rollback"
                self._rollback_active_attempt_record["post_attempt_message"] = stop_message
            self._stream_complete = True
            self._mark_rollback_failed(stop_message)
            self._write_lsdyna_debug_summary(
                note="Streaming single LS-DYNA attempt ended before the target replay frame count.",
            )
            print(f"{_ansys_common._LOG_PREFIX} {stop_message} {_ansys_common._lsdyna_debug_hint(current_attempt_dir)}")
            self._stop_viewer(close=True)
            return

        if self._rollback_current_attempt_mode == "base":
            retry_reason_code = "incomplete_replay_after_base_attempt_exit"
            retry_reason_message = (
                f"Base attempt {self._rollback_current_attempt_index} accepted "
                f"{accepted_frame_count}/{target_frame_count} replay frames and "
                f"stopped with {exit_observation} (exit_code={exit_code})."
            )
            next_attempt_mode = "rescue"
            next_rescue_depth = 1
            retry_debug_note = "Streaming rollback launched a one-frame rescue attempt."
        else:
            missing_frame_number = min(
                accepted_frame_count_before_attempt + 1,
                target_frame_count,
            )
            retry_reason_code = "rescue_frame_not_accepted_after_attempt_exit"
            retry_reason_message = (
                f"Rescue attempt {self._rollback_current_rescue_depth}/"
                f"{self._rollback_max_retries} did not accept frame "
                f"{missing_frame_number}/{target_frame_count} and stopped with "
                f"{exit_observation} (exit_code={exit_code})."
            )
            next_attempt_mode = "rescue"
            next_rescue_depth = self._rollback_current_rescue_depth + 1
            retry_debug_note = "Streaming rollback launched another one-frame rescue attempt."

        if self._rollback_active_attempt_record is not None:
            self._rollback_active_attempt_record["retry_reason_code"] = retry_reason_code
            self._rollback_active_attempt_record["retry_reason_message"] = retry_reason_message
            self._rollback_active_attempt_record["post_attempt_action"] = "launch_rescue_retry"
        self._persist_rollback_summary()

        if next_attempt_mode == "rescue" and next_rescue_depth > self._rollback_max_retries:
            self._stream_complete = True
            self._mark_rollback_failed(
                "LS-DYNA rollback retries exhausted after an incomplete attempt "
                f"exit ({exit_observation}, exit_code={exit_code})."
            )
            self._write_lsdyna_debug_summary(
                note="Streaming rollback retries exhausted.",
            )
            print(
                f"{_ansys_common._LOG_PREFIX} Rollback retries exhausted; keeping "
                f"{accepted_frame_count}/{target_frame_count} accepted replay "
                f"frames. {_ansys_common._lsdyna_debug_hint(current_attempt_dir)}"
            )
            self._stop_viewer(close=True)
            return

        print(f"{_ansys_common._LOG_PREFIX} {retry_reason_message} Starting a rollback retry.")
        self._rollback_pending_retry_info = {
            "trigger_reason_code": retry_reason_code,
            "trigger_reason_message": retry_reason_message,
            "trigger_exit_code": int(exit_code),
            "trigger_normal_termination": bool(normal_termination),
            "trigger_exit_observation": exit_observation,
            "accepted_frame_count_before_retry": int(accepted_frame_count),
            "target_frame_count": int(target_frame_count),
            "trigger_attempt_mode": str(self._rollback_current_attempt_mode),
            "trigger_rescue_depth": int(self._rollback_current_rescue_depth),
        }
        (
            restart_frame_index,
            restart_bag_verts_cm,
            restart_content_q_cm,
        ) = self._prepare_retry_from_prefix()
        self._start_rollback_attempt(
            attempt_index=self._rollback_current_attempt_index + 1,
            restart_frame_index=restart_frame_index,
            attempt_mode=next_attempt_mode,
            rescue_depth=next_rescue_depth,
            restart_bag_verts_cm=restart_bag_verts_cm,
            content_part_initial_q_cm=restart_content_q_cm,
        )
        self._write_lsdyna_debug_summary(
            note=retry_debug_note,
        )

    def _poll_streaming_replay(self, *, force: bool = False):
        """Poll the current attempt's d3plot and start retries when needed."""
        if not self._stream_mode or self._stream_complete:
            return

        now_s = _ansys_common.time.monotonic()
        if not force and now_s < self._stream_next_poll_time_s:
            return
        self._stream_next_poll_time_s = now_s + _ansys_common._STREAM_POLL_INTERVAL_S

        if self._stream_solver_process is not None:
            exit_code = self._stream_solver_process.poll()
            if exit_code is not None and not self._stream_solver_finished:
                self._stream_solver_finished = True
                self._stream_solver_exit_code = int(exit_code)
                force = True

        current_attempt_dir = self._rollback_current_attempt_dir or self.job_dir
        try:
            d3plot_signature = _ansys_common._d3plot_family_signature(current_attempt_dir)
        except FileNotFoundError:
            if self._stream_solver_finished:
                self._write_lsdyna_debug_summary(
                    note="Streaming rollback attempt ended before any d3plot file appeared.",
                )
                self._finish_or_retry_current_attempt()
            return

        if not force and d3plot_signature == self._stream_last_d3plot_signature:
            if self._stream_solver_finished:
                self._finish_or_retry_current_attempt()
            return

        try:
            update, self._last_replay_diagnostics = _ansys_common._load_replay_data_with_diagnostics(
                current_attempt_dir,
                self.deck,
                output_dt_s=self.target_frame_dt,
            )
            self._stream_last_replay_read_error = ""
        except Exception as exc:
            self._stream_last_replay_read_error = str(exc)
            self._write_lsdyna_debug_summary(
                note="Replay read failed while polling streamed rollback output.",
            )
            if self._stream_solver_finished:
                self._finish_or_retry_current_attempt()
            return

        self._stream_last_d3plot_signature = d3plot_signature
        update = self._limit_rescue_replay_update(update)
        update = _offset_replay_times(update, self._rollback_current_time_offset_s)
        appended = self._append_streaming_replay(update)
        if appended > 0:
            total_frames = self._active_target_frame_count()
            print(f"{_ansys_common._LOG_PREFIX} Received {len(self.replay.times_s)}/{total_frames} replay frames")
            if self._rollback_active_attempt_record is not None:
                self._rollback_active_attempt_record["merged_frame_count_after_attempt"] = int(len(self.replay.times_s))
                self._rollback_active_attempt_record["completed_full_replay"] = bool(
                    len(self.replay.times_s) >= total_frames
                )
            self._persist_rollback_summary()
            if len(self.replay.times_s) >= total_frames:
                self._stream_complete = True
                self._mark_rollback_completed()
                self._write_lsdyna_debug_summary(
                    note="Streaming rollback reached the target replay frame count.",
                )
                return

        self._write_lsdyna_debug_summary(
            note=(
                "Streaming rollback updated accepted replay frames."
                if appended > 0
                else "Observed d3plot change without any newly accepted replay frame."
            ),
        )

        if self._stream_solver_finished:
            self._finish_or_retry_current_attempt()

    def cleanup(self):
        """Release resources and finalize the rollback summary."""
        if self._rollback_summary is not None:
            replay_frame_count = 0
            if hasattr(self, "replay"):
                replay_frame_count = len(self.replay.times_s)
            if replay_frame_count >= self._active_target_frame_count():
                self._mark_rollback_completed()
            elif self._rollback_summary.get("status") == "running":
                self._mark_rollback_failed("Rollback example cleaned up before reaching the target replay frame count.")
        super().cleanup()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        if (
            getattr(example, "capture_replay", False)
            and getattr(example, "capture_count", 0) > 0
            and getattr(example, "capture_video_path", None) is None
        ):
            example._finalize_replay_video()
        example.cleanup()
