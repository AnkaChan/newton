# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene registry. Import a scene module to register it via :func:`register`."""

from __future__ import annotations

from .base import Scene, SceneHandles

SCENES: dict[str, type[Scene]] = {}


def register(cls: type[Scene]) -> type[Scene]:
    SCENES[cls.key] = cls
    return cls


def make_scene(key: str, args) -> Scene:
    if key not in SCENES:
        raise KeyError(f"Unknown scene {key!r}; available: {sorted(SCENES)}")
    return SCENES[key](args)


# Register built-in scenes.
from . import grasp_avbd_cloth  # noqa: E402,F401
from . import pick_avbd_cube  # noqa: E402,F401
from . import shirt_pick  # noqa: E402,F401

__all__ = ["SCENES", "Scene", "SceneHandles", "make_scene", "register"]
