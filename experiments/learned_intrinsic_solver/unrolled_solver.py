# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable inner learned-optimizer unroll for one physical objective.

Experimental: every inner iteration uses the same original inertial predictor,
physical-start positions, prescribed corners, material, and timestep. This
module does not advance a physical Newton State. Carried candidates are detached
between iterations by default; the detached optimizer history (previous
gradient feature and achieved axis change) is threaded through every iteration
and exposed for the next physical step. ``backward_detached`` streams local
backward passes on one device; its distributed trainer integration is separate work.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch  # noqa: TID253 -- This opt-in experimental module is a Torch nn.Module.
from torch import Tensor, nn  # noqa: TID253
from torch.utils.checkpoint import checkpoint  # noqa: TID253

from .input_assembly import OptimizerHistory
from .solver_step import LearnedHexSolverStep

__all__ = ["UnrolledHexSolver", "UnrolledStepOutput"]


class UnrolledStepOutput(NamedTuple):
    """Final candidate, per-query physical energies [J], training losses, and final history.

    ``history`` is the detached optimizer history after the last proposal
    (its world axis gradient feature and achieved center-deformation change);
    pass it to the next physical step's unroll to carry it across the boundary.
    """

    final_positions: Tensor
    energies: Tensor
    objective: Tensor
    per_sample_objective: Tensor
    mean_normalized_intermediate: Tensor
    mean_energy_increase_penalty: Tensor
    performed_iterations: int
    history: OptimizerHistory | None = None


class UnrolledHexSolver(nn.Module):
    """Unroll 1 to 32 learned proposals against one fixed physical Y and pin set.

    The objective averages each intermediate energy's normalized change from
    the initial candidate, then penalizes positive adjacent energy changes.
    Both terms use the same detached initial scale ``max(E0, 1 J)``. The energy
    increase weight multiplies the nonnegative penalty. Closest-rotation frames
    and the gradient feature are recomputed at every iteration under
    LearnedHexSolverStep's frozen-frame policy, and each proposal consumes the
    detached history of the preceding one. Each proposal's network, fusion and
    energy remain connected; carried positions are detached before the next
    proposal by default.

    Args:
        step: Complete learned hex step containing network, fusion, and energy.
        max_iterations: Maximum allowed inner proposals, at most 32.
        energy_increase_weight: Nonnegative multiplier on mean per-step energy
            increases, after normalization by the initial energy [J]. Each
            preceding energy is detached as the comparison target.
        detach_iterations: Cut gradients through carried positions between
            inner proposals. False retains the connected comparison mode.
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
        detach_iterations: bool = True,
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
        if not isinstance(detach_iterations, bool):
            raise TypeError("detach_iterations must be boolean")
        self.step = step
        self.max_iterations = max_iterations
        self.energy_increase_weight = float(energy_increase_weight)
        self.detach_iterations = detach_iterations
        self.checkpoint_activations = checkpoint_activations

    def _one_step(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        fixed_positions: Tensor,
        previous_positions: Tensor,
        history: OptimizerHistory | None,
        *,
        detach_energy_target: bool,
    ):
        result = self.step(
            positions,
            inertial_prediction,
            previous_positions=previous_positions,
            fixed_positions=fixed_positions,
            detach_energy_target=detach_energy_target,
            history=history,
        )
        return result.positions, result.loss.total, result.axis_gradient_world, result.achieved_axis_update_world

    def forward(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        *,
        previous_positions: Tensor,
        fixed_positions: Tensor | None = None,
        iterations: int = 1,
        detach_energy_target: bool = False,
        history: OptimizerHistory | None = None,
    ) -> UnrolledStepOutput:
        """Return all K+1 energies and the Kth candidate without detaching it.

        This forward-only API retains every local graph until the caller runs
        backward. Use ``backward_detached`` to release activations as you go.

        Args:
            positions: Initial candidate shared corners [m], [B,P,3].
            inertial_prediction: Original physical free-motion predictor [m],
                [B,P,3], held unchanged throughout the unroll.
            previous_positions: Original physical-step start [m], [B,P,3],
                held unchanged throughout the unroll. Required.
            fixed_positions: Prescribed corners [m], [B,F,3]; defaults to rest.
            iterations: Runtime learned proposal count in [1,max_iterations].
            detach_energy_target: Detach Y and previous_positions only in physical
                energy terms while retaining their network-feature paths. The
                default preserves the existing learned step's gradients.
            history: Detached optimizer history carried from the preceding
                physical step, or None to start a trajectory.

        Raises:
            ValueError: Invalid count, shape, candidate, or physical energy.
        """
        return self._run(
            positions,
            inertial_prediction,
            previous_positions=previous_positions,
            fixed_positions=fixed_positions,
            iterations=iterations,
            detach_energy_target=detach_energy_target,
            backward_each_iteration=False,
            history=history,
        )

    def backward_detached(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        *,
        previous_positions: Tensor,
        fixed_positions: Tensor | None = None,
        iterations: int = 1,
        detach_energy_target: bool = True,
        history: OptimizerHistory | None = None,
    ) -> UnrolledStepOutput:
        """Accumulate mean local parameter gradients and release each step's graph.

        Experimental single-device training helper. Backward each local loss
        divided by K immediately, without retaining its graph. All input
        tensors are treated as fixed physical data; outside input graphs are
        detached. Only parameter gradients are accumulated. Returned positions,
        energies and objective are detached diagnostics, not another loss to
        backward. Forward positions and loss values match ``forward``.

        The caller zeros gradients before this call and makes one optimizer
        update afterward. This method never clears gradients or changes weights.
        Discard partial gradients if a later proposal raises an error. Do not
        call this method through ``DDP.module``; that bypasses DDP's forward
        synchronization. The distributed streaming path is not implemented.

        Args:
            positions: Initial shared corners [m], shape [B,P,3].
            inertial_prediction: Fixed original free-motion prediction [m],
                shape [B,P,3]. Never recomputed between inner iterations.
            previous_positions: Fixed physical-step start [m], [B,P,3]. Required.
            fixed_positions: Prescribed corners [m], shape [B,F,3].
            iterations: Proposal count in [1,max_iterations].
            detach_energy_target: Energy target policy, as in ``forward``.
            history: Detached optimizer history, as in ``forward``.

        Raises:
            ValueError: Connected iteration policy or invalid physical input.
            RuntimeError: Autograd is disabled.
        """
        if not self.detach_iterations:
            raise ValueError("backward_detached requires detach_iterations=True")
        if not torch.is_grad_enabled():
            raise RuntimeError("backward_detached requires grad mode")
        if previous_positions is None:
            raise ValueError("previous_positions must supply the physical-step start positions")
        return self._run(
            positions.detach(),
            inertial_prediction.detach(),
            previous_positions=previous_positions.detach(),
            fixed_positions=None if fixed_positions is None else fixed_positions.detach(),
            iterations=iterations,
            detach_energy_target=detach_energy_target,
            backward_each_iteration=True,
            history=history,
        )

    def _run(
        self,
        positions: Tensor,
        inertial_prediction: Tensor,
        *,
        previous_positions: Tensor | None,
        fixed_positions: Tensor | None,
        iterations: int,
        detach_energy_target: bool,
        backward_each_iteration: bool,
        history: OptimizerHistory | None,
    ) -> UnrolledStepOutput:
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 1 <= iterations <= self.max_iterations
        ):
            raise ValueError(f"iterations must be in [1, {self.max_iterations}]")
        if not isinstance(detach_energy_target, bool):
            raise TypeError("detach_energy_target must be boolean")
        if previous_positions is None:
            raise ValueError("previous_positions must supply the physical-step start positions")
        if fixed_positions is None:
            fixed_positions = self.step.rest_positions[self.step.fixed_indices][None].expand(positions.shape[0], -1, -1)
        initial = self.step.energy(
            positions,
            inertial_prediction.detach() if detach_energy_target else inertial_prediction,
            previous_positions=previous_positions.detach() if detach_energy_target else previous_positions,
        ).total
        reference = initial.detach()
        scale = reference.clamp_min(1.0)
        energies = [initial]
        current = positions
        for index in range(iterations):
            if index and self.detach_iterations:
                current = current.detach()
            if self.checkpoint_activations and torch.is_grad_enabled():
                current, energy, gradient_feature, achieved = checkpoint(
                    lambda x, y, pins, previous, carried: self._one_step(
                        x, y, pins, previous, carried, detach_energy_target=detach_energy_target
                    ),
                    current,
                    inertial_prediction,
                    fixed_positions,
                    previous_positions,
                    history,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                current, energy, gradient_feature, achieved = self._one_step(
                    current,
                    inertial_prediction,
                    fixed_positions,
                    previous_positions,
                    history,
                    detach_energy_target=detach_energy_target,
                )
            # Detach at every carry: the next proposal reads history as fixed input.
            history = OptimizerHistory(
                gradient_feature.detach(),
                achieved.detach(),
                torch.ones(current.shape[0], dtype=torch.bool, device=current.device),
            )
            if backward_each_iteration:
                normalized = (energy - reference) / scale
                increase = torch.relu(energy - energies[-1].detach()) / scale
                local_objective = normalized + self.energy_increase_weight * increase
                (local_objective.mean() / iterations).backward()
                current, energy = current.detach(), energy.detach()
                del normalized, increase, local_objective
            energies.append(energy)
        energy_history = torch.stack(energies, dim=1)
        normalized = (energy_history[:, 1:] - reference[:, None]) / scale[:, None]
        increase = torch.relu(energy_history[:, 1:] - energy_history[:, :-1].detach()) / scale[:, None]
        normalized_per_sample = normalized.mean(dim=1)
        increase_per_sample = increase.mean(dim=1)
        per_sample_objective = normalized_per_sample + self.energy_increase_weight * increase_per_sample
        objective = per_sample_objective.mean()
        return UnrolledStepOutput(
            current,
            energy_history,
            objective,
            per_sample_objective,
            normalized_per_sample.mean(),
            increase_per_sample.mean(),
            iterations,
            history,
        )
