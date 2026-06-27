# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experiment framework for comparing coupling strategies across scenes.

Three independent, registry-based axes are mixed by the runner:

* **scenes** (:mod:`newton.exp.scenes`) -- world geometry + task sequence.
* **solvers** (:mod:`newton.exp.solvers`) -- coupling strategy (``avbd``, ``proxy``).
* **controllers** (:mod:`newton.exp.controllers`) -- scripted keyframes or interactive.

Run an experiment with::

    python -m newton.exp --solver proxy --control state_machine

Adding a comparison point is one file + a ``@register`` decorator.
"""

from __future__ import annotations

from .controllers import CONTROLLERS, make_controller
from .runner import Experiment, build_parser, main
from .scenes import SCENES, make_scene
from .solvers import SOLVERS, make_solver

__all__ = [
    "CONTROLLERS",
    "Experiment",
    "SCENES",
    "SOLVERS",
    "build_parser",
    "main",
    "make_controller",
    "make_scene",
    "make_solver",
]
