# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene interface: the world geometry and task definition for an experiment.

A scene owns everything that is independent of the solver/coupling strategy and
of the controller: the robot it uses, the deformable object(s), static
obstacles, the home/hover EE pose, the IK target link, the camera, and its own
task sequence (keyframes). Solver-specific robot tuning and collision filtering
live in :mod:`newton.exp.solvers`; the time profile of the motion lives in
:mod:`newton.exp.controllers`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import warp as wp


@dataclass
class SceneHandles:
    """Semantic index sets into the finalized model, produced during assembly."""

    robot_bodies: list[int] = field(default_factory=list)
    robot_joints: list[int] = field(default_factory=list)
    robot_shapes: list[int] = field(default_factory=list)
    static_shapes: list[int] = field(default_factory=list)
    gripper_bodies: list[int] = field(default_factory=list)
    particle_count: int = 0


class Scene:
    """Base class for experiment scenes.

    Subclasses implement the geometry hooks. ``key`` is the registry name used
    by ``--scene``.
    """

    key: str = "scene"

    #: Body-label suffix the IK objective targets.
    ik_link_label: str = "fr3_hand"
    #: TCP offset from that link [m].
    ik_link_offset: wp.vec3 = wp.vec3(0.0, 0.0, 0.0)
    #: IK joint-limit objective weight (0 disables it, matching IsaacLab).
    ik_joint_limit_weight: float = 0.0
    #: IK solver iterations per control step.
    ik_iters: int = 24
    #: Default named task sequence (see :meth:`sequences`).
    default_sequence: str = "pick"

    def __init__(self, args):
        self.args = args

    #: Initial robot joint configuration the arm spawns at.
    def robot_init_q(self) -> list[float]:
        raise NotImplementedError

    # -- robot (shared by sim + IK models) --------------------------------
    def build_robot(self, builder, *, collapse_fixed_joints: bool):
        """Add the robot to ``builder`` at its rest config; return ``(bodies, joints, shapes)``."""
        raise NotImplementedError

    # -- task sequences ---------------------------------------------------
    def sequences(self, home_pos, home_quat) -> dict:
        """Return a name -> :class:`~newton.exp.controllers.sequences.Sequence` map."""
        raise NotImplementedError

    # -- world assembly ---------------------------------------------------
    def add_static(self, builder) -> list[int]:
        """Add static obstacles (ground, table, ...); return their shape ids."""
        raise NotImplementedError

    def add_deformables(self, builder) -> None:
        """Add cloth / cable / soft bodies."""
        raise NotImplementedError

    # -- task definition --------------------------------------------------
    def home_pose(self):
        """Return ``(pos (3,), quat (4,) xyzw)`` of the hover/home EE pose."""
        raise NotImplementedError

    # -- presentation -----------------------------------------------------
    def camera(self):
        """Return ``(pos, pitch, yaw, look_at)`` or ``None`` to keep defaults."""
        return None

    @classmethod
    def add_args(cls, parser):
        """Register scene-specific CLI arguments (optional)."""
