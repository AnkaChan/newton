# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable inner learned-optimizer unroll for one physical objective.

Experimental: every inner iteration uses the same original inertial predictor,
prescribed corners, material, and timestep. This module does not advance a
physical Newton State. Wrap this complete module, rather than its network, in
DDP when training a multi-iteration objective.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch  # noqa: TID253 -- This opt-in experimental module is a Torch nn.Module.
from torch import Tensor, nn  # noqa: TID253
from torch.utils.checkpoint import checkpoint  # noqa: TID253

from .solver_step import LearnedHexSolverStep

__all__ = ["UnrolledHexSolver", "UnrolledStepOutput"]


class UnrolledStepOutput(NamedTuple):
    """Final candidate, per-query physical energies [J], and training losses."""

    final_positions: Tensor
    energies: Tensor
    objective: Tensor
    per_sample_objective: Tensor
    mean_normalized_intermediate: Tensor
    mean_energy_increase_penalty: Tensor
    performed_iterations: int


class UnrolledHexSolver(nn.Module):
    """Unroll 1 to 32 learned proposals against one fixed physical Y and pin set.

    The objective averages each intermediate energy's normalized change from
    the initial candidate, then penalizes positive adjacent energy changes.
    Both terms use the same detached initial scale ``max(E0, 1 J)``. The energy
    increase weight multiplies the nonnegative penalty. Polar frames are
    recomputed at every iteration under LearnedHexSolverStep's existing frozen
    frame policy; fused positions and network outputs remain attached to the
    full first-order graph.

    Args:
        step: Complete learned hex step containing network, fusion, and energy.
        max_iterations: Maximum allowed inner proposals, at most 32.
        energy_increase_weight: Nonnegative multiplier on mean per-step energy
    increases, after normalization by the initial energy [J]. Each preceding
    energy is detached as the comparison target, so minimizing the penalty
    cannot raise an earlier iterate's energy to make a later increase smaller.
        checkpoint_activations: Recompute each learned step during backward
            using non-reentrant Torch activation checkpointing to reduce saved
            activations. The CPU sparse forward/adjoint work is repeated.
    """

    def __init__(
        self,
        step: LearnedHexSolverStep,
        *,
        max_iterations: int = 32,
        energy_increase_weight: float = 1.0,
        checkpoint_activations: bool = False,
    ):
        super().__init__()
        if not isinstance(step, LearnedHexSolverStep):
            raise TypeError("step must be a LearnedHexSolverStep")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 32:
            raise ValueError("max_iterations must be in [1, 32]")
        if not math.isfinite(energy_increase_weight) or energy_increase_weight < 0:
            raise ValueError("energy_increase_weight must be finite and nonnegative")
        if not isinstance(checkpoint_activations, bool):
            raise TypeError("checkpoint_activations must be boolean")
        self.step = step
        self.max_iterations = max_iterations
        self.energy_increase_weight = float(energy_increase_weight)
        self.checkpoint_activations = checkpoint_activations

    def _one_step(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        fixed_positions: Tensor,
        *,
        detach_energy_target: bool,
    ):
        result = self.step(
            positions,
            inertial_prediction,
            fixed_positions=fixed_positions,
            detach_energy_target=detach_energy_target,
        )
        return result.positions, result.loss.total

    def forward(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        *,
        fixed_positions: Tensor | None = None,
        iterations: int = 1,
        detach_energy_target: bool = False,
    ) -> UnrolledStepOutput:
        """Return all K+1 energies and the Kth candidate without detaching it.

        Args:
            positions: Initial candidate shared corners [m], [B,P,3].
            inertial_prediction: Original physical free-motion predictor [m],
                [B,P,3], held unchanged throughout the unroll.
            fixed_positions: Prescribed corners [m], [B,F,3]; defaults to rest.
            iterations: Runtime learned proposal count in [1,max_iterations].
            detach_energy_target: Detach Y only in physical energy terms while
                retaining its network-feature path. The default preserves the
                existing learned step's gradients.

        Raises:
            ValueError: Invalid count, shape, candidate, or physical energy.
        """
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 1 <= iterations <= self.max_iterations
        ):
            raise ValueError(f"iterations must be in [1, {self.max_iterations}]")
        if not isinstance(detach_energy_target, bool):
            raise TypeError("detach_energy_target must be boolean")
        if fixed_positions is None:
            fixed_positions = self.step.rest_positions[self.step.fixed_indices][None].expand(positions.shape[0], -1, -1)
        initial = self.step.energy(
            positions, inertial_prediction.detach() if detach_energy_target else inertial_prediction
        ).total
        energies = [initial]
        current = positions
        for _ in range(iterations):
            if self.checkpoint_activations and torch.is_grad_enabled():
                current, energy = checkpoint(
                    lambda x, y, pins: self._one_step(x, y, pins, detach_energy_target=detach_energy_target),
                    current,
                    inertial_prediction,
                    fixed_positions,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                current, energy = self._one_step(
                    current, inertial_prediction, fixed_positions, detach_energy_target=detach_energy_target
                )
            energies.append(energy)
        history = torch.stack(energies, dim=1)
        reference = initial.detach()
        scale = reference.clamp_min(1.0)
        normalized = (history[:, 1:] - reference[:, None]) / scale[:, None]
        increase = torch.relu(history[:, 1:] - history[:, :-1].detach()) / scale[:, None]
        normalized_per_sample = normalized.mean(dim=1)
        increase_per_sample = increase.mean(dim=1)
        per_sample_objective = normalized_per_sample + self.energy_increase_weight * increase_per_sample
        objective = per_sample_objective.mean()
        return UnrolledStepOutput(
            current,
            history,
            objective,
            per_sample_objective,
            normalized_per_sample.mean(),
            increase_per_sample.mean(),
            iterations,
        )
