# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver/coupling strategy registry."""

from __future__ import annotations

from .base import SolverStrategy

SOLVERS: dict[str, type[SolverStrategy]] = {}


def register(cls: type[SolverStrategy]) -> type[SolverStrategy]:
    SOLVERS[cls.key] = cls
    return cls


def make_solver(key: str, args) -> SolverStrategy:
    if key not in SOLVERS:
        raise KeyError(f"Unknown solver {key!r}; available: {sorted(SOLVERS)}")
    return SOLVERS[key](args)


# Register built-in strategies.
from . import monolithic  # noqa: E402,F401
from . import proxy  # noqa: E402,F401
from . import soft_constraint  # noqa: E402,F401

__all__ = ["SOLVERS", "SolverStrategy", "make_solver", "register"]
