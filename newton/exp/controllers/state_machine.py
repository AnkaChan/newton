# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scripted controller: play a named, scene-defined task :class:`Sequence`.

``--sequence <name>`` picks one of the scene's sequences (``Scene.sequences``);
omitting it uses the scene's ``default_sequence``.
"""

from __future__ import annotations

import warp as wp

from .base import Controller
from . import register


@register
class StateMachineController(Controller):
    key = "state_machine"

    def __init__(self, sequence, name):
        self.sequence = sequence
        self.name = name

    @classmethod
    def build(cls, args, exp):
        sequences = exp.scene.sequences(exp.home_pos, exp.home_quat)
        name = getattr(args, "sequence", None) or exp.scene.default_sequence
        if name not in sequences:
            raise KeyError(f"Unknown sequence {name!r} for scene {exp.scene.key!r}; available: {sorted(sequences)}")
        print(f"[state_machine] scene={exp.scene.key!r} sequence={name!r}")
        return cls(sequences[name], name)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--sequence", type=str, default=None, help="Named task sequence (scene-defined); default per scene."
        )

    def reset(self):
        self.sequence.reset()

    def update(self, t: float, dt: float):
        pos, quat, finger = self.sequence.step(t, dt)
        return wp.vec3(*[float(x) for x in pos]), wp.vec4(*[float(x) for x in quat]), float(finger)
