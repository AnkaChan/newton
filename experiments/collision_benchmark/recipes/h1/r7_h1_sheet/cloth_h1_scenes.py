# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental H1 pair recipes for the preserved planar cloth collection."""

from __future__ import annotations

import os

import numpy as np

from .h1_bag_scenes import H1ClothScene, _original_placement
from .h1_open_start_scene import H1OpenStartScene

__all__ = ["build_scene"]

_NAMES = ("patch", "towel", "sheet", "trousers", "dress", "tshirt", "skirt")


def _front_grips(name):
    """Locate two upper-panel front patches on the unchanged R5 rest surface.

    These are material labels. The runtime observes their actual positions
    after free settling. It does not use their rest height as a live target.
    """
    _, _, original, vertices, _ = _original_placement(name)
    front_y = float(original[:, 1].min())
    front = vertices[vertices[:, 1] < front_y + 0.025]
    lo, hi = float(front[:, 0].min()), float(front[:, 0].max())
    grips = []
    for x in (lo + 0.025, hi - 0.025):
        nearby = front[abs(front[:, 0] - x) < 0.02]
        if not len(nearby):
            raise ValueError(f"No original material near {name} front grip")
        upper_z = float(nearby[:, 2].max())
        upper = nearby[nearby[:, 2] > upper_z - 0.005]
        target = np.array([x, front_y + 0.012, upper_z])
        grips.append(upper[np.argmin(np.linalg.norm(upper - target, axis=1))])
    return np.asarray(grips)


def build_scene(scene_id, **kwargs):
    name = scene_id.removeprefix("r7_h1_")
    if name not in _NAMES:
        raise ValueError(scene_id)
    kwargs.setdefault("rest_grip_positions", _front_grips(name))
    if os.environ.get("NEWTON_H1_START_MODE") == "audited_open":
        return H1OpenStartScene(scene_id, asset_name=name, **kwargs)
    return H1ClothScene(scene_id, asset_name=name, **kwargs)
