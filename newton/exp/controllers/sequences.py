# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Named task sequences for scripted control.

A :class:`Sequence` maps elapsed time to ``(pos, quat, finger)``. Scenes expose
a dict of named sequences (see ``Scene.sequences``); ``--sequence`` selects one.
All sequences are expressed as time-interpolated :class:`KeyframeSequence`
waypoints: positions are linearly interpolated, orientations slerped, and the
finger width stepped. Deterministic timing makes runs directly comparable.
"""

from __future__ import annotations

import numpy as np

from .base import Keyframe, slerp


class Sequence:
    """Base task sequence. ``step`` returns ``(pos (3,), quat (4,) xyzw, finger)``."""

    def reset(self):
        pass

    def step(self, t: float, dt: float):
        raise NotImplementedError


class KeyframeSequence(Sequence):
    """Time-interpolated waypoints (slerp orientation, stepped finger width)."""

    def __init__(self, keyframes: list[Keyframe]):
        self.keyframes = list(keyframes)
        self.key_times = np.cumsum([max(1e-6, kf.duration) for kf in self.keyframes])
        self._prev_seg = -1

    def reset(self):
        self._prev_seg = -1

    def step(self, t: float, dt: float):
        t = min(t, float(self.key_times[-1]) - 1e-6)
        seg = min(int(np.searchsorted(self.key_times, t)), len(self.keyframes) - 1)
        t_start = self.key_times[seg - 1] if seg > 0 else 0.0
        t_end = self.key_times[seg]
        alpha = float(np.clip((t - t_start) / max(t_end - t_start, 1e-6), 0.0, 1.0))
        cur = self.keyframes[seg]
        prev = self.keyframes[seg - 1] if seg > 0 else cur
        pos = (1.0 - alpha) * prev.pos + alpha * cur.pos
        quat = slerp(prev.quat, cur.quat, alpha)
        finger = (1.0 - alpha) * float(prev.finger) + alpha * float(cur.finger)
        if seg != self._prev_seg:
            print(f"[keyframe] segment {seg} target_z={float(pos[2]):.3f} finger={finger:.3f}")
            self._prev_seg = seg
        return np.asarray(pos), np.asarray(quat), finger
