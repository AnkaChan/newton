# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Controller registry."""

from __future__ import annotations

from .base import Controller, Keyframe, slerp

CONTROLLERS: dict[str, type[Controller]] = {}


def register(cls: type[Controller]) -> type[Controller]:
    CONTROLLERS[cls.key] = cls
    return cls


def make_controller(key: str, args, exp) -> Controller:
    if key not in CONTROLLERS:
        raise KeyError(f"Unknown controller {key!r}; available: {sorted(CONTROLLERS)}")
    return CONTROLLERS[key].build(args, exp)


# Register built-in controllers.
from . import state_machine  # noqa: E402,F401
from . import interactive  # noqa: E402,F401

__all__ = ["CONTROLLERS", "Controller", "Keyframe", "make_controller", "register", "slerp"]
