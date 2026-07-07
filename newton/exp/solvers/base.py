# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver/coupling strategy interface.

A strategy owns everything that differs between coupling approaches: how the
robot's actuators are tuned, which collisions are filtered, the contact
materials, how the solver is built, the per-substep hook, and how
solver-internal state is reset. This is the axis swept by ``--solver``.
"""

from __future__ import annotations


class SolverStrategy:
    """Base class. ``key`` is the registry name used by ``--solver``."""

    key = "solver"

    #: Whether the robot should be added with fixed joints collapsed.
    collapse_fixed_joints: bool = False
    #: Whether MuJoCo custom attributes must be registered on the builder.
    uses_mujoco: bool = False
    #: Physics sub-steps per control update (IsaacLab ``decimation``).
    decimation: int = 1

    def __init__(self, args):
        self.args = args
        self.solver = None
        #: Active scene, set by the runner before model assembly so the
        #: strategy can consult scene-provided physics overrides.
        self.scene = None

    # -- scene-provided overrides -----------------------------------------
    def scene_materials(self) -> dict:
        """Scene ``model_materials`` for this strategy's key (or ``{}``)."""
        return self.scene.model_materials(self.key) if self.scene is not None else {}

    def scene_solver_overrides(self) -> dict:
        """Scene ``solver_overrides`` for this strategy's key (or ``{}``)."""
        return self.scene.solver_overrides(self.key) if self.scene is not None else {}

    def scene_robot_gains(self) -> dict:
        """Scene ``robot_gains`` for this strategy's key (or ``{}``)."""
        return self.scene.robot_gains(self.key) if self.scene is not None else {}

    # -- model assembly ---------------------------------------------------
    def register_attributes(self, builder):
        """Register solver-specific builder custom attributes."""

    def configure_robot(self, builder, robot_bodies, robot_joints):
        """Set actuator gains / effort limits / armature / gravcomp."""

    def filter_collisions(self, builder, robot_shapes, static_shapes):
        """Add collision filter pairs needed by this strategy."""

    def apply_materials(self, model):
        """Set ``shape_material_*`` and ``soft_contact_*`` on the model."""

    def post_finalize(self, model, handles):
        """Hook after finalize (e.g. set the VBD articulation rest pose)."""

    def build_solver(self, model, handles, args):
        """Construct and return the solver (also stored as ``self.solver``)."""
        raise NotImplementedError

    # -- simulation hooks -------------------------------------------------
    def make_collision_pipeline(self, model):
        """Build the collision pipeline (IsaacLab uses broad_phase='explicit')."""
        import newton

        return newton.CollisionPipeline(model, broad_phase="explicit")

    def pre_substeps(self, solver, state):
        """Hook before each control update's sub-steps (e.g. VBD rebuild_bvh)."""

    def sync_initial(self, state):
        """Sync solver-internal state to ``state`` after the initial FK."""

    def post_step(self, model, state_out):
        """Per-substep hook after ``solver.step`` (e.g. ``eval_ik``)."""

    def reset_internal(self, state, device):
        """Reset solver-internal buffers (coupling forces, MuJoCo Data, ...)."""

    @classmethod
    def add_args(cls, parser):
        """Register solver-specific CLI arguments (optional)."""
