# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run an experiment: ``python -m newton.exp [--scene ...] [--solver ...] ...``.

Examples (run from the repo root so this ``newton`` shadows any site-packages copy)::

    # monolithic AVBD vs proxy coupling on the shirt-pick scene (scripted pick)
    python -m newton.exp --solver avbd  --control state_machine
    python -m newton.exp --solver proxy --control state_machine

    # named task sequences (scene-defined): pick | keyframe | hold
    python -m newton.exp --solver avbd --sequence keyframe

    # interactive: drag the gizmo, G toggles the gripper, R resets
    python -m newton.exp --solver proxy --control interactive
"""

from __future__ import annotations

from .runner import main

if __name__ == "__main__":
    main(num_frames=600)
