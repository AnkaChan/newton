# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable physical windows around the fixed-Y learned inner unroll.

Newton State and the CPU rigid proxy are deliberately outside the graph. Each
physical step's X, V, inertial predictor Y, rigid-guided candidate, learned
updates, and velocity advance otherwise stay connected inside a window. The
default training policy freezes Y only when evaluating physical energy; Y is
still an attached network feature. Windows detach X and V at their boundary so
callers can backward each window and release its graph before asking for the
next one. The default is one physical timestep per gradient window: train
each solve independently on the physical state produced by the previous one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import torch  # noqa: TID253 -- This opt-in experimental module uses Torch tensors.
from torch import Tensor  # noqa: TID253

from .hex_energy import make_inertial_prediction
from .newton_solver import SolverLearnedIntrinsic
from .unrolled_solver import UnrolledHexSolver

__all__ = ["PhysicalRollout", "PhysicalStep", "PhysicalWindow"]


class PhysicalStep(NamedTuple):
    """One physical advance, with K+1 inner energies [B,K+1] in joules."""

    previous_positions: Tensor
    previous_velocities: Tensor
    inertial_prediction: Tensor
    initial_candidate: Tensor
    energies: Tensor
    per_sample_objective: Tensor
    objective: Tensor
    positions: Tensor
    velocities: Tensor


class PhysicalWindow(NamedTuple):
    """A differentiable window [start_step,end_step) with mean local loss."""

    start_step: int
    end_step: int
    steps: tuple[PhysicalStep, ...]
    per_sample_objective: Tensor
    objective: Tensor
    final_positions: Tensor
    final_velocities: Tensor


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class PhysicalRollout:
    """Compose native rigid guidance with tensor-only physical time advancement.

    ``solver`` supplies immutable native masses, gravity and a detached CPU
    rigid proxy. ``unrolled`` must wrap that solver's step at ``time_step``.
    No Warp State is read or written by ``windows``. The yielded window owns
    its graph; call backward before requesting the next window to bound memory.
    """

    def __init__(
        self,
        solver: SolverLearnedIntrinsic,
        unrolled: UnrolledHexSolver,
        *,
        time_step: float,
        detach_energy_target: bool = True,
    ):
        if not isinstance(solver, SolverLearnedIntrinsic):
            raise TypeError("solver must be a SolverLearnedIntrinsic")
        if not isinstance(unrolled, UnrolledHexSolver):
            raise TypeError("unrolled must be an UnrolledHexSolver")
        if isinstance(time_step, bool) or not math.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        if not isinstance(detach_energy_target, bool):
            raise TypeError("detach_energy_target must be boolean")
        self.time_step = float(time_step)
        if unrolled.step is not solver._step_for_dt(self.time_step):
            raise ValueError("unrolled must use the solver step at time_step")
        self.solver = solver
        self.unrolled = unrolled
        self.detach_energy_target = detach_energy_target

    def _validate_inputs(
        self,
        positions: Tensor,
        velocities: Tensor,
        forces: Tensor | None,
        fixed_positions: Tensor | None,
        physical_steps: int,
        iterations: int,
        gradient_window: int,
    ) -> tuple[Tensor, Tensor]:
        _positive_int(physical_steps, "physical_steps")
        _positive_int(gradient_window, "gradient_window")
        _positive_int(iterations, "iterations")
        if iterations > self.unrolled.max_iterations:
            raise ValueError(f"iterations must be at most {self.unrolled.max_iterations}")
        step = self.unrolled.step
        expected_shape = (step.rest_positions.shape[0], 3)
        if (
            not isinstance(positions, Tensor)
            or positions.ndim != 3
            or positions.shape[0] < 1
            or positions.shape[1:] != expected_shape
            or positions.dtype != step.rest_positions.dtype
            or positions.device != step.rest_positions.device
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("positions must be finite [B,P,3] matching the learned step")
        if (
            not isinstance(velocities, Tensor)
            or velocities.shape != positions.shape
            or velocities.dtype != positions.dtype
            or velocities.device != positions.device
            or not torch.isfinite(velocities).all()
        ):
            raise ValueError("velocities must be finite [B,P,3] matching positions")
        if forces is None:
            forces = torch.zeros_like(positions)
        if (
            not isinstance(forces, Tensor)
            or forces.shape not in (positions.shape, (physical_steps, *positions.shape))
            or forces.dtype != positions.dtype
            or forces.device != positions.device
            or not torch.isfinite(forces).all()
        ):
            raise ValueError("forces must be finite [B,P,3] or [physical_steps,B,P,3] matching positions")
        fixed = step.fixed_indices
        if fixed_positions is None:
            fixed_positions = positions[:, fixed].detach().clone()
        elif (
            not isinstance(fixed_positions, Tensor)
            or fixed_positions.shape != (positions.shape[0], fixed.numel(), 3)
            or fixed_positions.dtype != positions.dtype
            or fixed_positions.device != positions.device
            or not torch.isfinite(fixed_positions).all()
        ):
            raise ValueError("fixed_positions must be finite [B,F,3] matching positions")
        elif not torch.equal(positions[:, fixed], fixed_positions):
            raise ValueError("initial positions must equal prescribed fixed_positions")
        return forces, fixed_positions.detach().clone()

    def _advance(
        self,
        positions: Tensor,
        velocities: Tensor,
        forces: Tensor,
        fixed_positions: Tensor,
        gravity: Tensor,
        *,
        iterations: int,
    ) -> PhysicalStep:
        step = self.unrolled.step
        acceleration = gravity + forces / step.energy.lumped_mass[None, :, None]
        inertial = make_inertial_prediction(positions, velocities, self.time_step, explicit_acceleration=acceleration)

        # Match SolverLearnedIntrinsic.prepare_problem's CPU rigid guidance,
        # but never let its detached snapshot replace the attached X in fusion.
        self.solver.rigid_predictor.model.set_gravity(tuple(gravity.detach().cpu().tolist()))
        rotations = []
        translations = []
        for batch_index in range(positions.shape[0]):
            rigid = self.solver.rigid_predictor.predict(
                positions[batch_index].detach().cpu().contiguous(),
                velocities[batch_index].detach().cpu().contiguous(),
                forces[batch_index].detach().cpu().contiguous(),
                self.time_step,
            )
            rotations.append(rigid.rigid_delta_rotation[0].to(device=positions.device))
            translations.append(rigid.rigid_delta_translation[0].to(device=positions.device))
        rotation = torch.stack(rotations)
        translation = torch.stack(translations)
        base = positions @ rotation.transpose(-1, -2) + translation[:, None, :]
        zero_increment = base.new_zeros((positions.shape[0], step.energy.cell_corner_indices.shape[0], 3, 3))
        initial = step.fusion.fuse(base, zero_increment, fixed_positions)
        solved = self.unrolled(
            initial,
            inertial,
            fixed_positions=fixed_positions,
            iterations=iterations,
            detach_energy_target=self.detach_energy_target,
        )
        new_positions = solved.final_positions
        free = torch.ones(positions.shape[1], dtype=torch.bool, device=positions.device)
        free[step.fixed_indices] = False
        new_velocities = torch.where(free[None, :, None], (new_positions - positions) / self.time_step, 0)
        return PhysicalStep(
            positions,
            velocities,
            inertial,
            initial,
            solved.energies,
            solved.per_sample_objective,
            solved.objective,
            new_positions,
            new_velocities,
        )

    def windows(
        self,
        positions: Tensor,
        velocities: Tensor,
        *,
        physical_steps: int,
        iterations: int = 1,
        gradient_window: int = 1,
        forces: Tensor | None = None,
        fixed_positions: Tensor | None = None,
        on_step_start: Callable[[int, Tensor, Tensor, Tensor, Tensor, Tensor], None] | None = None,
    ):
        """Yield consecutive differentiable windows, detaching X,V between them.

        Forces are external (gravity excluded), either constant [B,P,3] or
        scheduled [physical_steps,B,P,3]. Prescribed pins come from the initial
        physical X unless supplied explicitly. Each step recomputes Y from its
        current X,V; every inner query within that step sees that same Y.
        The local objective is the mean over physical steps and batch samples.
        By default, gradients stop at every physical timestep boundary while
        all optimizer iterations inside that timestep remain connected.
        Optional on_step_start receives (index, X, V, forces, pins, gravity)
        as detached, isolated tensor copies immediately before each solve.
        It can durably retain a failed solve's starting state without touching
        the autograd path or modifying the physical inputs.
        """
        if on_step_start is not None and not callable(on_step_start):
            raise TypeError("on_step_start must be callable or None")
        forces, pins = self._validate_inputs(
            positions, velocities, forces, fixed_positions, physical_steps, iterations, gradient_window
        )
        x, v = positions, velocities
        del positions, velocities, fixed_positions
        for start in range(0, physical_steps, gradient_window):
            steps = []
            end = min(start + gradient_window, physical_steps)
            for index in range(start, end):
                applied_forces = forces if forces.ndim == 3 else forces[index]
                gravity = torch.as_tensor(self.solver._gravity(), dtype=x.dtype, device=x.device)
                if on_step_start is not None:
                    on_step_start(
                        index,
                        x.detach().clone(),
                        v.detach().clone(),
                        applied_forces.detach().clone(),
                        pins.detach().clone(),
                        gravity.detach().clone(),
                    )
                result = self._advance(x, v, applied_forces, pins, gravity, iterations=iterations)
                steps.append(result)
                x, v = result.positions, result.velocities
            per_sample = torch.stack([step.per_sample_objective for step in steps]).mean(dim=0)
            yield PhysicalWindow(start, end, tuple(steps), per_sample, per_sample.mean(), x, v)
            x, v = x.detach(), v.detach()
            del per_sample, steps, result
