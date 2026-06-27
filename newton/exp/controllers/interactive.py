# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive controller: drag the TCP target gizmo, toggle/reset with keys.

The viewer mutates ``ee_tf`` in-place as the user drags the gizmo. ``G`` toggles
the gripper; ``R`` requests a reset (consumed by the runner).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..robots import GRIP_CLOSE, GRIP_OPEN
from .base import Controller
from . import register


@register
class InteractiveController(Controller):
    key = "interactive"

    def __init__(self, viewer, home_pos, home_quat):
        self.viewer = viewer
        self.home_pos = np.asarray(home_pos, dtype=np.float64)
        self.home_quat = np.asarray(home_quat, dtype=np.float64)
        self._has_keys = hasattr(viewer, "is_key_down")
        self.reset()

    @classmethod
    def build(cls, args, exp):
        return cls(exp.viewer, exp.home_pos, exp.home_quat)

    def reset(self):
        self.ee_tf = wp.transform(wp.vec3(*self.home_pos.tolist()), wp.quat(*self.home_quat.tolist()))
        self.gripper_closed = False
        self._g_prev = False
        self._r_prev = False
        self._reset_requested = False

    def _poll_keys(self):
        if not self._has_keys:
            return
        g = bool(self.viewer.is_key_down("g"))
        if g and not self._g_prev:
            self.gripper_closed = not self.gripper_closed
            print(f"[interactive] gripper {'closed' if self.gripper_closed else 'open'} (G)")
        self._g_prev = g

        r = bool(self.viewer.is_key_down("r"))
        if r and not self._r_prev:
            self._reset_requested = True
        self._r_prev = r

    def update(self, t: float, dt: float):
        self._poll_keys()
        p = wp.transform_get_translation(self.ee_tf)
        pos = wp.vec3(p[0], p[1], max(float(p[2]), 0.01))
        q = wp.transform_get_rotation(self.ee_tf)
        finger = GRIP_CLOSE if self.gripper_closed else GRIP_OPEN
        return pos, wp.vec4(q[0], q[1], q[2], q[3]), finger

    def render_overlay(self, viewer):
        if hasattr(viewer, "log_gizmo"):
            viewer.log_gizmo("target_tcp", self.ee_tf)

    def consume_reset(self) -> bool:
        if self._reset_requested:
            self._reset_requested = False
            print("[interactive] reset (R)")
            return True
        return False
