# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.utils

FPS = 60.0
FRAME_DT = 1.0 / FPS

PHASE_OPEN_S = 0.45
PHASE_CLOSE_S = 0.20
PHASE_HOLD_S = 3.00
PHASE_LIFT_S = 2.50
TOTAL_DURATION_S = PHASE_OPEN_S + PHASE_CLOSE_S + PHASE_HOLD_S + PHASE_LIFT_S

BAG_H_CM = 27.9
BAG_H_M = BAG_H_CM * 0.01
BAG_X_CM = 0.0
BAG_Y_CM = 0.0
BAG_X_M = 0.0
BAG_Y_M = 0.0

GRAB_Z_CM = BAG_H_CM + 9.0
LIFT_Z_CM = 65.0
GRAB_EE_Z_M = GRAB_Z_CM * 0.01
LIFT_EE_Z_M = LIFT_Z_CM * 0.01

FR3_BASE_CM = (-50.0, 0.0, 5.0)
FR3_BASE_M = (-0.50, 0.0, 0.05)
FR3_INIT_Q = [
    -3.6802e-03,
    2.3902e-02,
    3.6804e-03,
    -2.3683,
    -1.2919e-04,
    2.3922,
    7.8549e-01,
]

FINGER_OPEN_Q_M = 0.04
FINGER_OPEN_Q_CM = 4.0
MAX_GRIPPER_WIDTH_CM = 4.0

FINGER_PAD_OFFSET_CM = (0.0, 0.758, 5.75)
FINGER_PAD_OFFSET_M = np.array(FINGER_PAD_OFFSET_CM, dtype=np.float64) * 0.01
FINGER_PAD_HALF_THICKNESS_CM = 0.75
FINGER_PAD_HALF_HEIGHT_CM = 2.60
FINGER_PAD_HALF_THICKNESS_M = FINGER_PAD_HALF_THICKNESS_CM * 0.01
FINGER_PAD_HALF_HEIGHT_M = FINGER_PAD_HALF_HEIGHT_CM * 0.01
FINGER_PAD_TOP_BAND_CM = 1.0

PAD_OPEN_CLEARANCE_M = 0.045
PAD_LIFT_Z_M = LIFT_EE_Z_M
PAD_Z_FROM_EE_M = 0.117
GRIP_BAND_TOP_FRAC = 0.85
SMALL_PAD_SCALE = math.sqrt(0.20)
DEFAULT_CLOSED_WIDTH_CM = 0.6
_LOG_PREFIX = "[bag]"


@dataclass(frozen=True)
class Fr3HandBodies:
    """Body indices for the FR3 hand links used by the bag lift examples."""

    left_finger: int
    right_finger: int
    hand: int


def add_lift_robot_arguments(parser, *, closed_width: bool = True, small_pad: bool = True) -> None:
    """Add common FR3 lift and finger-pad arguments."""
    if closed_width:
        parser.add_argument(
            "--closed-width-cm",
            type=float,
            default=DEFAULT_CLOSED_WIDTH_CM,
            help="Final finger-pad gap [cm] when fully closed.",
        )
    if small_pad:
        parser.add_argument(
            "--small-pad",
            action="store_true",
            help=("Use a smaller finger-pad visual/contact patch (approximately 20%% of the full pad face area)."),
        )


def add_fr3_hand(
    builder: newton.ModelBuilder,
    *,
    base_position: tuple[float, float, float],
    scale: float,
    finger_open_q: float,
    parse_visuals_as_colliders: bool = True,
) -> Fr3HandBodies:
    """Add the FR3 hand URDF and return its relevant body indices."""
    asset_path = newton.utils.download_asset("franka_emika_panda")
    builder.add_urdf(
        str(Path(asset_path) / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform(base_position, wp.quat_identity()),
        floating=False,
        scale=scale,
        enable_self_collisions=False,
        parse_visuals_as_colliders=parse_visuals_as_colliders,
    )
    builder.joint_q[:9] = [*FR3_INIT_Q, finger_open_q, finger_open_q]
    return Fr3HandBodies(
        left_finger=next(i for i, label in enumerate(builder.body_label) if label.endswith("fr3_leftfinger")),
        right_finger=next(i for i, label in enumerate(builder.body_label) if label.endswith("fr3_rightfinger")),
        hand=next(i for i, label in enumerate(builder.body_label) if label.endswith("fr3_hand")),
    )


def gripper_fraction_from_closed_width_cm(
    closed_width_cm: float,
    *,
    max_gap_cm: float = MAX_GRIPPER_WIDTH_CM,
) -> float:
    """Convert a target finger gap [cm] into FR3 gripper fraction, 0=open, 1=closed."""
    if not np.isfinite(closed_width_cm):
        raise ValueError("`closed_width_cm` must be finite.")
    if not np.isfinite(max_gap_cm) or max_gap_cm <= 0.0:
        raise ValueError("`max_gap_cm` must be finite and positive.")
    if closed_width_cm < 0.0 or closed_width_cm > max_gap_cm:
        raise ValueError(f"`closed_width_cm` must be within [0, {max_gap_cm}].")
    return float(1.0 - closed_width_cm / max_gap_cm)


def finger_joint_q_from_gripper_fraction(gripper_fraction: float, *, scale: float) -> float:
    """Return one FR3 finger joint coordinate for a gripper fraction and unit scale."""
    return float(scale) * FINGER_OPEN_Q_M * (1.0 - float(gripper_fraction))


def lift_waypoints_cm(*, closed_fraction: float = 1.0):
    """Return canonical lift waypoints in centimeters."""
    return [
        (wp.vec3(BAG_X_CM, BAG_Y_CM, GRAB_Z_CM), PHASE_OPEN_S, 0.0),
        (wp.vec3(BAG_X_CM, BAG_Y_CM, GRAB_Z_CM), PHASE_CLOSE_S, float(closed_fraction)),
        (wp.vec3(BAG_X_CM, BAG_Y_CM, GRAB_Z_CM), PHASE_HOLD_S, float(closed_fraction)),
        (wp.vec3(BAG_X_CM, BAG_Y_CM, LIFT_Z_CM), PHASE_LIFT_S, float(closed_fraction)),
    ]


def lift_waypoints_m(*, closed_fraction: float = 1.0):
    """Return canonical lift waypoints in meters."""
    return [
        (wp.vec3(BAG_X_M, BAG_Y_M, GRAB_EE_Z_M), PHASE_OPEN_S, 0.0),
        (wp.vec3(BAG_X_M, BAG_Y_M, GRAB_EE_Z_M), PHASE_CLOSE_S, float(closed_fraction)),
        (wp.vec3(BAG_X_M, BAG_Y_M, GRAB_EE_Z_M), PHASE_HOLD_S, float(closed_fraction)),
        (wp.vec3(BAG_X_M, BAG_Y_M, LIFT_EE_Z_M), PHASE_LIFT_S, float(closed_fraction)),
    ]


def finger_pad_half_extents_cm(
    bag_verts_cm: np.ndarray,
    *,
    small_pad: bool,
    min_hx_cm: float = 0.0,
) -> tuple[float, float, float]:
    """Compute shared FR3 finger-pad half extents in centimeters."""
    top_band_mask = bag_verts_cm[:, 2] >= (BAG_H_CM - FINGER_PAD_TOP_BAND_CM)
    top_band_verts = bag_verts_cm[top_band_mask] if np.any(top_band_mask) else bag_verts_cm
    pad_hx_cm = 0.5 * float(top_band_verts[:, 0].max() - top_band_verts[:, 0].min())
    pad_hx_cm = max(pad_hx_cm, float(min_hx_cm))
    pad_hy_cm = FINGER_PAD_HALF_THICKNESS_CM
    pad_hz_cm = FINGER_PAD_HALF_HEIGHT_CM
    if small_pad:
        pad_hx_cm *= SMALL_PAD_SCALE
        pad_hz_cm *= SMALL_PAD_SCALE
    return pad_hx_cm, pad_hy_cm, pad_hz_cm


def log_content_placements_cm(
    *,
    log_prefix: str = _LOG_PREFIX,
    sphere_pos_cm: tuple[float, float, float],
    sphere_clearance_cm: float,
    box_pos_cm: tuple[float, float, float],
    box_clearance_cm: float,
    capsule_pos_cm: tuple[float, float, float],
    capsule_clearance_cm: float,
    local_bag_coords: bool = False,
) -> None:
    """Print the shared sphere/box/capsule placement summary."""
    prefix = f"{log_prefix.strip()} " if log_prefix else ""
    coord_note = "local bag coords, " if local_bag_coords else ""
    print(
        f"{prefix}Object placements ({coord_note}clearance to bag wall):\n"
        f"  sphere  @ {_fmt_cm3(sphere_pos_cm)}  clr={sphere_clearance_cm:.2f} cm\n"
        f"  box     @ {_fmt_cm3(box_pos_cm)}  clr={box_clearance_cm:.2f} cm\n"
        f"  capsule @ {_fmt_cm3(capsule_pos_cm)} [Y-horiz]  "
        f"clr={capsule_clearance_cm:.2f} cm"
    )


def _fmt_cm3(value: tuple[float, float, float]) -> str:
    x, y, z = value
    return f"({x:.1f}, {y:.1f}, {z:.1f})"
