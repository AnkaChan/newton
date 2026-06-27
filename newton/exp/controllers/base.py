# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Controller interface: drives the EE target + gripper over time.

A controller maps simulation time to ``(pos, quat, finger_width)``. The runner
feeds that to the IK solver each step. ``KeyframeController`` plays a
scene-defined sequence; ``InteractiveController`` reads a mouse gizmo + keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class Keyframe:
    """One waypoint of a scripted task sequence."""

    duration: float  # time [s] to interpolate from the previous keyframe to this one
    pos: np.ndarray  # EE target position (3,) [m]
    quat: np.ndarray  # EE target orientation (4,) (qx, qy, qz, qw)
    finger: float  # finger width [m] (open ~0.04, closed 0.0)


def slerp(q0, q1, alpha):
    """Shortest-arc quaternion slerp; quaternions in (x, y, z, w) order."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + alpha * (q1 - q0)
    else:
        theta = np.arccos(d)
        q = (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)


class Controller:
    """Base controller. ``key`` is the registry name used by ``--control``."""

    key = "controller"

    def reset(self):
        """Reset internal state (called by the runner on reset)."""

    def update(self, t: float, dt: float):
        """Return ``(pos: wp.vec3, quat: wp.vec4, finger: float)`` for time ``t``."""
        raise NotImplementedError

    def render_overlay(self, viewer):
        """Draw any interactive overlay (e.g. a gizmo). Optional."""

    def consume_reset(self) -> bool:
        """Return True once if a reset was requested (e.g. the R key)."""
        return False

    @classmethod
    def build(cls, args, exp) -> "Controller":
        """Construct the controller for an assembled experiment."""
        raise NotImplementedError

    @classmethod
    def add_args(cls, parser):
        """Register controller-specific CLI arguments (optional)."""
