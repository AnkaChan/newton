# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fixed-work matrix-free physics corrector for learned v5 iterates.

This module implements one bounded stable-Neo-Hookean Gauss--Newton
correction.  It is a support layer for an architecture-v5 learned
principal-stretch iterate, not an end-to-end classical solver and not an
exact Newton method.  In particular, the tangent drops the generally
indefinite derivative-of-cofactor term from the exact Hessian.

The implementation is intentionally research-only.  It keeps only O(V + T)
state, executes a fixed number of PCG operator applications, evaluates every
registered step candidate (including the exact zero-step fallback), and
reports all scheduled and active work.  It is a Torch correctness foundation;
host-visible trace extraction and eager temporaries mean it is not yet a
CUDA-graph performance implementation.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Callable

import torch

from .v5_objective import (
    CommonObjectiveContext,
    _cofactor_3x3,
    _common_objective_components_trusted,
    _common_objective_residual_trusted,
    _deformation_gradient,
    _determinant_3x3,
    common_objective_residual,
)


def _require_nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _execution_scalar(
    name: str,
    value: float,
    reference: torch.Tensor,
    *,
    preserve_positive: bool,
) -> torch.Tensor:
    """Materialize a policy scalar without silent dtype overflow/underflow."""
    materialized = torch.tensor(value, dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(materialized).detach().cpu()):
        raise ValueError(f"{name} is not finite in execution dtype {reference.dtype}")
    if preserve_positive and value > 0.0 and not bool((materialized > 0.0).detach().cpu()):
        raise ValueError(f"positive {name} underflows in execution dtype {reference.dtype}")
    return materialized


def _validate_vector_field(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}, got {tuple(value.shape)}")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must share the reference device and dtype")


def _validate_positions(context: CommonObjectiveContext, positions: torch.Tensor) -> None:
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.shape != (context.n_vertices, 3):
        raise ValueError(f"positions must have shape ({context.n_vertices}, 3), got {tuple(positions.shape)}")
    if positions.device != context.device or positions.dtype != context.dtype:
        raise ValueError("positions and context must share one device and dtype")
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("the corrector supports torch.float32 and torch.float64 execution")


def common_objective_free_gradient(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Return the exact common-objective gradient with pinned rows eliminated.

    This is deliberately a thin alias for :func:`common_objective_residual`.
    Keeping one authoritative analytic expression prevents the corrector from
    quietly drifting away from the objective used to score the learned solver.

    Args:
        context: Authenticated common-objective problem.
        positions: Candidate positions [m], shape ``[V, 3]``.

    Returns:
        Exact free gradient [N], shape ``[V, 3]``. Pinned rows are zero.
    """
    return common_objective_residual(context, positions)


class StableNeoHookeanGaussNewtonOperator:
    """Matrix-free positive-definite tangent at one fixed position field.

    The elastic action for a direction ``p`` is

    ``dP = mu * dF + lambda * (cof(F) : dF) * cof(F)``.

    It is positive semidefinite; the positive free-vertex inertia makes the
    complete operator positive definite. Pinned input directions are removed
    before physical assembly, and pinned output rows reproduce the original
    direction exactly. Thus the full-space representation is symmetric with
    identity blocks on pins while the solve remains a free-variable solve.

    Persistent tensors are limited to positions, per-tet cofactors, a free
    mask, and 3-by-3 vertex diagonal blocks. No dense fine-level matrix is
    assembled or stored.
    """

    __slots__ = ("_block_diagonal", "_cofactor", "_context", "_free_mask", "_positions")

    def __init__(self, context: CommonObjectiveContext, positions: torch.Tensor) -> None:
        if not isinstance(context, CommonObjectiveContext):
            raise TypeError("context must be a CommonObjectiveContext")
        context.validate_immutable()
        _validate_positions(context, positions)
        self._context = context
        self._positions = positions.clone()
        deformation_gradient = _deformation_gradient(context, self._positions)
        self._cofactor = _cofactor_3x3(deformation_gradient)
        self._free_mask = torch.ones(context.n_vertices, dtype=torch.bool, device=context.device)
        self._free_mask[context._owned_tensor("pinned")] = False
        self._block_diagonal = self._assemble_block_diagonal()

    @property
    def n_vertices(self) -> int:
        """Number of operator vertices."""
        return self._context.n_vertices

    @property
    def device(self) -> torch.device:
        """Device holding the operator state."""
        return self._positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype used by the operator."""
        return self._positions.dtype

    def _assemble_block_diagonal(self) -> torch.Tensor:
        context = self._context
        J = context._owned_tensor("J")
        volume = context._owned_tensor("volume")
        mu = context._owned_tensor("mu")
        lam = context._owned_tensor("lam")
        tets = context._owned_tensor("tets")
        mass = context._owned_tensor("mass")

        identity = torch.eye(3, dtype=self.dtype, device=self.device)
        diagonal = context._inverse_dt_squared * mass[:, None, None] * identity
        j_squared_norm = J.square().sum(dim=-1)
        cofactor_j = torch.einsum("tdc,tac->tad", self._cofactor, J)
        elastic = mu[:, None, None, None] * j_squared_norm[:, :, None, None] * identity
        elastic = elastic + lam[:, None, None, None] * (cofactor_j[..., :, None] * cofactor_j[..., None, :])
        elastic = volume[:, None, None, None] * elastic
        diagonal = diagonal.index_add(0, tets.reshape(-1), elastic.reshape(-1, 3, 3))

        pinned = context._owned_tensor("pinned")
        if pinned.numel() > 0:
            diagonal = diagonal.index_copy(0, pinned, identity.expand(pinned.numel(), 3, 3))
        return diagonal

    def block_diagonal(self) -> torch.Tensor:
        """Return the analytic 3-by-3 block-Jacobi diagonal, shape ``[V, 3, 3]``."""
        return self._block_diagonal.clone()

    def matvec(self, direction: torch.Tensor) -> torch.Tensor:
        """Apply the stable-NH Gauss--Newton tangent without a fine matrix.

        Args:
            direction: Position direction [m], shape ``[V, 3]``.

        Returns:
            Tangent action [N], shape ``[V, 3]``. Pinned rows equal the input
            direction on those rows exactly.
        """
        _validate_vector_field("direction", direction, self._positions)
        context = self._context
        tets = context._owned_tensor("tets")
        J = context._owned_tensor("J")
        volume = context._owned_tensor("volume")
        mu = context._owned_tensor("mu")
        lam = context._owned_tensor("lam")
        mass = context._owned_tensor("mass")
        pinned = context._owned_tensor("pinned")

        free_direction = torch.where(self._free_mask[:, None], direction, torch.zeros_like(direction))
        direction_tet = free_direction[tets]
        delta_f = torch.einsum("tac,tad->tdc", J, direction_tet)
        cofactor_contraction = (self._cofactor * delta_f).sum(dim=(-2, -1))
        delta_p = mu[:, None, None] * delta_f
        delta_p = delta_p + (lam * cofactor_contraction)[:, None, None] * self._cofactor
        delta_p = volume[:, None, None] * delta_p
        tet_contribution = torch.einsum("tdc,tac->tad", delta_p, J)

        action = context._inverse_dt_squared * mass[:, None] * free_direction
        action = action.index_add(0, tets.reshape(-1), tet_contribution.reshape(-1, 3))
        if pinned.numel() > 0:
            action = action.index_fill(0, pinned, 0.0)
            action = action.index_copy(0, pinned, direction.index_select(0, pinned))
        return action


def stable_neo_hookean_gn_matvec(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Apply the analytic matrix-free stable-NH Gauss--Newton tangent."""
    return StableNeoHookeanGaussNewtonOperator(context, positions).matvec(direction)


def stable_neo_hookean_gn_block_diagonal(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Return the analytic vertex-block diagonal of the Gauss--Newton tangent."""
    return StableNeoHookeanGaussNewtonOperator(context, positions).block_diagonal()


@dataclasses.dataclass(frozen=True)
class FixedPCGConfig:
    """Fixed-work PCG policy.

    Args:
        iterations: Scheduled PCG iterations. Every scheduled operator and
            preconditioner call executes even after convergence or breakdown.
        relative_tolerance: Relative true linear-residual tolerance.
        absolute_tolerance: Absolute true linear-residual tolerance [N].
        curvature_relative_tolerance: Minimum accepted curvature divided by
            ``||p|| ||A p||``. Zero still requires strictly positive
            curvature.
    """

    iterations: int = 4
    relative_tolerance: float = 1.0e-6
    absolute_tolerance: float = 0.0
    curvature_relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, numbers.Integral):
            raise TypeError("iterations must be an integer")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        for name in ("relative_tolerance", "absolute_tolerance", "curvature_relative_tolerance"):
            _require_nonnegative_real(name, getattr(self, name))


@dataclasses.dataclass(frozen=True)
class FixedPCGTrace:
    """Host-visible diagnostics for one fixed-work PCG execution.

    ``algorithmic_scalar_reductions`` counts global dot products and norms.
    ``safeguard_scalar_reductions`` counts global finite, symmetry, and
    factor-validity checks outside the matrix-free operator itself. The sum is
    ``scalar_reductions``; per-tet contractions inside ``matvec`` are not
    global scalar reductions. ``active_iterations`` counts scheduled slots
    that entered with an active search direction, including a slot that
    detects breakdown before updating the solution.
    """

    scheduled_iterations: int
    active_iterations: int
    matrix_vector_products: int
    preconditioner_applications: int
    algorithmic_scalar_reductions: int
    safeguard_scalar_reductions: int
    scalar_reductions: int
    converged: bool
    stationary: bool
    breakdown: bool
    breakdown_iteration: int | None
    initial_residual_norm: float
    final_recursive_residual_norm: float
    final_true_residual_norm: float
    residual_norms: tuple[float, ...]
    curvatures: tuple[float, ...]
    step_lengths: tuple[float, ...]
    active_schedule: tuple[bool, ...]


@dataclasses.dataclass(frozen=True)
class FixedPCGResult:
    """PCG solution and complete fixed-work trace."""

    solution: torch.Tensor
    trace: FixedPCGTrace


def fixed_work_block_pcg(
    *,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    block_diagonal: torch.Tensor,
    pinned: torch.Tensor,
    config: FixedPCGConfig,
) -> FixedPCGResult:
    """Solve a 3-vector-block SPD system with fixed scheduled PCG work.

    Numerical failure is fail-closed: a non-finite quantity, non-SPD block
    diagonal, or insufficient curvature sets ``breakdown`` and returns the
    exact zero solution. The loop nevertheless performs every configured
    operator and block-preconditioner application with inactive zero vectors.

    Args:
        matvec: Matrix-free symmetric operator action on shape ``[V, 3]``.
        rhs: Linear-system right-hand side, shape ``[V, 3]``.
        block_diagonal: Symmetric 3-by-3 Jacobi blocks, shape ``[V, 3, 3]``.
        pinned: Eliminated vertex indices, shape ``[P]``.
        config: Fixed PCG work and numerical thresholds.

    Returns:
        A solution tensor and detailed scheduled/active work. Pinned solution
        rows are exact zero.
    """
    if not callable(matvec):
        raise TypeError("matvec must be callable")
    if not isinstance(config, FixedPCGConfig):
        raise TypeError("config must be a FixedPCGConfig")
    if not isinstance(rhs, torch.Tensor) or rhs.ndim != 2 or rhs.shape[1] != 3:
        raise ValueError("rhs must be a torch.Tensor with shape (V, 3)")
    if not rhs.is_floating_point():
        raise ValueError("rhs must have a floating dtype")
    expected_diagonal_shape = (rhs.shape[0], 3, 3)
    if not isinstance(block_diagonal, torch.Tensor) or block_diagonal.shape != expected_diagonal_shape:
        raise ValueError(f"block_diagonal must have shape {expected_diagonal_shape}")
    if block_diagonal.device != rhs.device or block_diagonal.dtype != rhs.dtype:
        raise ValueError("block_diagonal and rhs must share one device and dtype")
    if not isinstance(pinned, torch.Tensor) or pinned.ndim != 1 or pinned.dtype != torch.int64:
        raise ValueError("pinned must be an int64 tensor with shape (P,)")
    if pinned.device != rhs.device:
        raise ValueError("pinned and rhs must share one device")
    if pinned.numel() > 0:
        if (pinned < 0).any() or (pinned >= rhs.shape[0]).any():
            raise ValueError("pinned contains an out-of-range vertex")
        if torch.unique(pinned).numel() != pinned.numel():
            raise ValueError("pinned must not contain duplicates")

    relative_tolerance = _execution_scalar(
        "relative_tolerance",
        config.relative_tolerance,
        rhs,
        preserve_positive=True,
    )
    absolute_tolerance = _execution_scalar(
        "absolute_tolerance",
        config.absolute_tolerance,
        rhs,
        preserve_positive=True,
    )
    curvature_relative_tolerance = _execution_scalar(
        "curvature_relative_tolerance",
        config.curvature_relative_tolerance,
        rhs,
        preserve_positive=True,
    )

    diagonal = block_diagonal.clone()
    identity = torch.eye(3, dtype=rhs.dtype, device=rhs.device)
    if pinned.numel() > 0:
        diagonal = diagonal.index_copy(0, pinned, identity.expand(pinned.numel(), 3, 3))
    finite_diagonal = torch.isfinite(diagonal).all()
    diagonal_scale = diagonal.abs().amax().clamp_min(1.0)
    symmetry_error = (diagonal - diagonal.transpose(-2, -1)).abs().amax()
    symmetric_diagonal = symmetry_error <= 16.0 * torch.finfo(rhs.dtype).eps * diagonal_scale
    finite_for_factor = torch.nan_to_num(diagonal, nan=0.0, posinf=0.0, neginf=0.0)
    symmetric_for_factor = 0.5 * (finite_for_factor + finite_for_factor.transpose(-2, -1))
    cholesky, cholesky_info = torch.linalg.cholesky_ex(symmetric_for_factor, check_errors=False)
    factor_valid = finite_diagonal & symmetric_diagonal & (cholesky_info == 0).all()
    safe_cholesky = torch.where(factor_valid, cholesky, identity.expand_as(cholesky))

    def apply_preconditioner(value: torch.Tensor) -> torch.Tensor:
        solved = torch.cholesky_solve(value[..., None], safe_cholesky).squeeze(-1)
        if pinned.numel() > 0:
            solved = solved.index_fill(0, pinned, 0.0)
        return solved

    zero = torch.zeros_like(rhs)
    rhs_free = rhs.clone()
    if pinned.numel() > 0:
        rhs_free = rhs_free.index_fill(0, pinned, 0.0)
    rhs_finite = torch.isfinite(rhs_free).all()
    initial_norm = torch.linalg.vector_norm(rhs_free)
    tolerance = absolute_tolerance + relative_tolerance * initial_norm
    tolerance_finite = torch.isfinite(tolerance)
    stationary = rhs_finite & tolerance_finite & (initial_norm <= tolerance)
    setup_breakdown = (~rhs_finite) | (~factor_valid) | (~tolerance_finite)
    active = (~stationary) & (~setup_breakdown)
    breakdown = setup_breakdown.clone()
    breakdown_iteration = torch.where(
        setup_breakdown,
        torch.tensor(-1, dtype=torch.int64, device=rhs.device),
        torch.tensor(-2, dtype=torch.int64, device=rhs.device),
    )

    solution = zero.clone()
    residual = rhs_free.clone()
    preconditioned = apply_preconditioner(residual)
    direction = torch.where(active, preconditioned, zero)
    residual_preconditioned = (residual * preconditioned).sum()
    initial_scalar_valid = torch.isfinite(residual_preconditioned) & (stationary | (residual_preconditioned > 0.0))
    initial_breakdown = active & (~initial_scalar_valid)
    breakdown = breakdown | initial_breakdown
    breakdown_iteration = torch.where(
        initial_breakdown & (breakdown_iteration == -2),
        torch.tensor(-1, dtype=torch.int64, device=rhs.device),
        breakdown_iteration,
    )
    active = active & initial_scalar_valid

    iterations = int(config.iterations)
    residual_norms = torch.empty(iterations + 1, dtype=rhs.dtype, device=rhs.device)
    curvatures = torch.empty(iterations, dtype=rhs.dtype, device=rhs.device)
    step_lengths = torch.empty(iterations, dtype=rhs.dtype, device=rhs.device)
    active_schedule = torch.empty(iterations, dtype=torch.bool, device=rhs.device)
    residual_norms[0] = initial_norm
    active_iterations = torch.zeros((), dtype=torch.int64, device=rhs.device)
    converged = stationary.clone()

    for iteration in range(iterations):
        active_before = active
        active_schedule[iteration] = active_before
        active_direction = torch.where(active_before, direction, zero)
        action = matvec(active_direction)
        if not isinstance(action, torch.Tensor) or action.shape != rhs.shape:
            raise ValueError("matvec must return a tensor with rhs shape")
        if action.device != rhs.device or action.dtype != rhs.dtype:
            raise ValueError("matvec output must share rhs device and dtype")
        if pinned.numel() > 0:
            action = action.index_fill(0, pinned, 0.0)

        curvature = (active_direction * action).sum()
        direction_norm = torch.linalg.vector_norm(active_direction)
        action_norm = torch.linalg.vector_norm(action)
        curvature_threshold = curvature_relative_tolerance * direction_norm * action_norm
        curvature_valid = torch.isfinite(action).all() & torch.isfinite(curvature) & (curvature > curvature_threshold)
        iteration_breakdown = active_before & (~curvature_valid)
        safe_curvature = torch.where(curvature_valid, curvature, torch.ones_like(curvature))
        step = torch.where(active_before & curvature_valid, residual_preconditioned / safe_curvature, 0.0)
        step_finite = torch.isfinite(step)
        update_candidate = active_before & curvature_valid & step_finite

        proposed_solution = solution + step * active_direction
        proposed_residual = residual - step * action
        residual_for_preconditioner = torch.where(update_candidate, proposed_residual, residual)
        proposed_preconditioned = apply_preconditioner(residual_for_preconditioner)
        proposed_scalar = (residual_for_preconditioner * proposed_preconditioned).sum()
        proposed_norm = torch.linalg.vector_norm(residual_for_preconditioner)
        converged_now = update_candidate & torch.isfinite(proposed_norm) & (proposed_norm <= tolerance)
        proposed_valid = (
            torch.isfinite(proposed_solution).all()
            & torch.isfinite(residual_for_preconditioner).all()
            & torch.isfinite(proposed_preconditioned).all()
            & torch.isfinite(proposed_scalar)
            & (converged_now | (proposed_scalar > 0.0))
        )
        update = update_candidate & proposed_valid
        iteration_breakdown = iteration_breakdown | (update_candidate & (~proposed_valid))

        solution = torch.where(update, proposed_solution, solution)
        residual = torch.where(update, residual_for_preconditioner, residual)
        preconditioned_new = torch.where(update, proposed_preconditioned, preconditioned)
        scalar_new = torch.where(update, proposed_scalar, residual_preconditioned)
        safe_old_scalar = torch.where(
            torch.isfinite(residual_preconditioned) & (residual_preconditioned > 0.0),
            residual_preconditioned,
            torch.ones_like(residual_preconditioned),
        )
        beta = torch.where(update & (~converged_now), scalar_new / safe_old_scalar, 0.0)
        beta_valid = torch.isfinite(beta) & (beta >= 0.0)
        beta_breakdown = update & (~converged_now) & (~beta_valid)
        iteration_breakdown = iteration_breakdown | beta_breakdown
        next_active = update & (~converged_now) & beta_valid
        direction = torch.where(
            next_active,
            preconditioned_new + torch.where(beta_valid, beta, 0.0) * direction,
            zero,
        )
        preconditioned = preconditioned_new
        residual_preconditioned = scalar_new
        active_iterations = active_iterations + active_before.to(torch.int64)
        converged = converged | converged_now
        first_breakdown = iteration_breakdown & (breakdown_iteration == -2)
        breakdown_iteration = torch.where(
            first_breakdown,
            torch.tensor(iteration, dtype=torch.int64, device=rhs.device),
            breakdown_iteration,
        )
        breakdown = breakdown | iteration_breakdown
        active = next_active & (~breakdown)

        curvatures[iteration] = curvature
        step_lengths[iteration] = step
        residual_norms[iteration + 1] = torch.where(update, proposed_norm, residual_norms[iteration])

    fail_closed_solution = torch.where(breakdown, zero, solution)
    true_action = matvec(fail_closed_solution)
    if not isinstance(true_action, torch.Tensor) or true_action.shape != rhs.shape:
        raise ValueError("matvec must return a tensor with rhs shape")
    if true_action.device != rhs.device or true_action.dtype != rhs.dtype:
        raise ValueError("matvec output must share rhs device and dtype")
    if pinned.numel() > 0:
        true_action = true_action.index_fill(0, pinned, 0.0)
    true_residual_norm = torch.linalg.vector_norm(rhs_free - true_action)
    final_breakdown = (~torch.isfinite(true_action).all()) | (~torch.isfinite(true_residual_norm))
    first_final_breakdown = final_breakdown & (breakdown_iteration == -2)
    breakdown_iteration = torch.where(
        first_final_breakdown,
        torch.tensor(iterations, dtype=torch.int64, device=rhs.device),
        breakdown_iteration,
    )
    breakdown = breakdown | final_breakdown
    fail_closed_solution = torch.where(breakdown, zero, fail_closed_solution)
    final_recursive_norm = residual_norms[-1]
    converged = converged & (~breakdown) & torch.isfinite(true_residual_norm) & (true_residual_norm <= tolerance)

    breakdown_index = int(breakdown_iteration.detach().cpu())
    algorithmic_scalar_reductions = 5 * iterations + 3
    safeguard_scalar_reductions = 4 * iterations + 6
    trace = FixedPCGTrace(
        scheduled_iterations=iterations,
        active_iterations=int(active_iterations.detach().cpu()),
        matrix_vector_products=iterations + 1,
        preconditioner_applications=iterations + 1,
        algorithmic_scalar_reductions=algorithmic_scalar_reductions,
        safeguard_scalar_reductions=safeguard_scalar_reductions,
        scalar_reductions=algorithmic_scalar_reductions + safeguard_scalar_reductions,
        converged=bool(converged.detach().cpu()),
        stationary=bool(stationary.detach().cpu()),
        breakdown=bool(breakdown.detach().cpu()),
        breakdown_iteration=None if breakdown_index == -2 else breakdown_index,
        initial_residual_norm=float(initial_norm.detach().cpu()),
        final_recursive_residual_norm=float(final_recursive_norm.detach().cpu()),
        final_true_residual_norm=float(true_residual_norm.detach().cpu()),
        residual_norms=tuple(float(value) for value in residual_norms.detach().cpu()),
        curvatures=tuple(float(value) for value in curvatures.detach().cpu()),
        step_lengths=tuple(float(value) for value in step_lengths.detach().cpu()),
        active_schedule=tuple(bool(value) for value in active_schedule.detach().cpu()),
    )
    return FixedPCGResult(solution=fail_closed_solution, trace=trace)


@dataclasses.dataclass(frozen=True)
class CorrectorConfig:
    """One fixed stable-NH Gauss--Newton corrector policy.

    Args:
        pcg: Fixed linear-solve work.
        candidate_alphas: Ordered deterministic step candidates. Exactly one
            entry must be zero; every candidate is evaluated, and zero is the
            exact fallback rather than an accepted correction.
        minimum_determinant: Strict determinant lower bound.
        minimum_singular_value: Strict singular-value lower bound.
        objective_increase_tolerance: Absolute roundoff allowance [J] for the
            mandatory non-increasing-objective safeguard.
        require_residual_nonincrease: Require the raw free-residual norm not
            to increase in addition to the objective safeguard.
        residual_increase_tolerance: Absolute residual-norm allowance [N].
    """

    pcg: FixedPCGConfig = dataclasses.field(default_factory=FixedPCGConfig)
    candidate_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)
    minimum_determinant: float = 0.0
    minimum_singular_value: float = 0.0
    objective_increase_tolerance: float = 1.0e-12
    require_residual_nonincrease: bool = True
    residual_increase_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.pcg, FixedPCGConfig):
            raise TypeError("pcg must be a FixedPCGConfig")
        if not isinstance(self.candidate_alphas, tuple) or not self.candidate_alphas:
            raise TypeError("candidate_alphas must be a non-empty tuple")
        converted: list[float] = []
        for alpha in self.candidate_alphas:
            if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
                raise TypeError("candidate_alphas entries must be real numbers")
            value = float(alpha)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("candidate_alphas entries must be finite and non-negative")
            converted.append(value)
        if len(set(converted)) != len(converted):
            raise ValueError("candidate_alphas entries must be unique")
        if converted.count(0.0) != 1:
            raise ValueError("candidate_alphas must contain exactly one zero fallback")
        if not isinstance(self.require_residual_nonincrease, bool):
            raise TypeError("require_residual_nonincrease must be a bool")
        for name in (
            "minimum_determinant",
            "minimum_singular_value",
            "objective_increase_tolerance",
            "residual_increase_tolerance",
        ):
            _require_nonnegative_real(name, getattr(self, name))


@dataclasses.dataclass(frozen=True)
class CorrectorCandidateTrace:
    """Safeguard evidence for one fully evaluated line-search candidate."""

    index: int
    alpha: float
    finite: bool
    exact_pins: bool
    minimum_determinant: float
    minimum_singular_value: float
    objective: float
    raw_residual_norm: float
    determinant_valid: bool
    singular_value_valid: bool
    objective_nonincreasing: bool
    residual_nonincreasing: bool
    admissible_nonzero_step: bool


@dataclasses.dataclass(frozen=True)
class CorrectorWork:
    """Complete scheduled physics-state and linear-algebra work for one call."""

    gradient_state_evaluations: int
    tangent_builds: int
    block_diagonal_builds: int
    scheduled_pcg_iterations: int
    active_pcg_iterations: int
    tangent_matrix_vector_products: int
    block_preconditioner_applications: int
    pcg_scalar_reductions: int
    candidate_count: int
    candidate_objective_state_evaluations: int
    candidate_residual_state_evaluations: int
    candidate_determinant_state_evaluations: int
    candidate_singular_value_state_evaluations: int


@dataclasses.dataclass(frozen=True)
class CorrectorTrace:
    """Physics, line-search, and work evidence for one correction."""

    common_objective_sha256: str
    start_objective: float
    start_raw_residual_norm: float
    gradient_dot_direction: float
    descent_direction: bool
    pcg: FixedPCGTrace
    candidates: tuple[CorrectorCandidateTrace, ...]
    selected_candidate_index: int
    selected_alpha: float
    accepted: bool
    reason: str
    work: CorrectorWork


@dataclasses.dataclass(frozen=True)
class CorrectorResult:
    """Corrected positions, attempted direction, and complete trace."""

    positions: torch.Tensor
    direction: torch.Tensor
    trace: CorrectorTrace


def _candidate_geometry(
    context: CommonObjectiveContext,
    candidates: torch.Tensor,
    finite_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_candidates = torch.where(finite_positions[:, None, None], candidates, torch.zeros_like(candidates))
    deformation_gradient = _deformation_gradient(context, safe_candidates)
    determinant = _determinant_3x3(deformation_gradient)
    nan = torch.full((), float("nan"), dtype=candidates.dtype, device=candidates.device)
    minimum_determinant = torch.where(finite_positions, determinant.amin(dim=-1), nan)
    try:
        singular_values = torch.linalg.svdvals(deformation_gradient)
        minimum_singular_value = torch.where(finite_positions, singular_values.amin(dim=(-2, -1)), nan)
    except RuntimeError:
        # A backend SVD failure is a candidate-validity failure. Returning NaN
        # here makes every candidate fail closed while the caller still
        # preserves the supplied start exactly.
        minimum_singular_value = torch.full_like(minimum_determinant, float("nan"))
    return minimum_determinant, minimum_singular_value


def correct_common_objective(
    *,
    context: CommonObjectiveContext,
    start: torch.Tensor,
    config: CorrectorConfig,
) -> CorrectorResult:
    """Apply one fixed-work matrix-free correction to a learned iterate.

    The caller supplies the already-decoded learned principal-stretch iterate.
    Pinned coordinates in ``start`` are treated as the exact targets for this
    correction and are copied into every candidate. On any PCG breakdown or
    when no positive candidate passes every safeguard, the returned positions
    are a clone of ``start`` rather than a recomputed zero-step expression.

    Args:
        context: Authenticated common implicit objective.
        start: Learned-and-projected starting positions [m], shape ``[V, 3]``.
        config: Fixed PCG, candidate, and safeguard policy.

    Returns:
        Corrected positions, the attempted Gauss--Newton direction, and exact
        work/safeguard evidence.
    """
    if not isinstance(context, CommonObjectiveContext):
        raise TypeError("context must be a CommonObjectiveContext")
    if not isinstance(config, CorrectorConfig):
        raise TypeError("config must be a CorrectorConfig")
    context.validate_immutable()
    _validate_positions(context, start)

    gradient = _common_objective_residual_trusted(context, start)
    operator = StableNeoHookeanGaussNewtonOperator(context, start)
    pcg_result = fixed_work_block_pcg(
        matvec=operator.matvec,
        rhs=-gradient,
        block_diagonal=operator.block_diagonal(),
        pinned=context._owned_tensor("pinned"),
        config=config.pcg,
    )
    direction = pcg_result.solution
    gradient_dot_direction_tensor = (gradient * direction).sum()
    descent_direction_tensor = torch.isfinite(gradient_dot_direction_tensor) & (gradient_dot_direction_tensor < 0.0)

    alphas = torch.stack(
        tuple(
            _execution_scalar(
                f"candidate_alphas[{index}]",
                float(alpha),
                start,
                preserve_positive=True,
            )
            for index, alpha in enumerate(config.candidate_alphas)
        )
    )
    if torch.unique(alphas).numel() != alphas.numel():
        raise ValueError("candidate_alphas must remain distinct in the execution dtype")
    minimum_determinant = _execution_scalar(
        "minimum_determinant",
        config.minimum_determinant,
        start,
        preserve_positive=True,
    )
    minimum_singular_value = _execution_scalar(
        "minimum_singular_value",
        config.minimum_singular_value,
        start,
        preserve_positive=True,
    )
    objective_increase_tolerance = _execution_scalar(
        "objective_increase_tolerance",
        config.objective_increase_tolerance,
        start,
        preserve_positive=True,
    )
    residual_increase_tolerance = _execution_scalar(
        "residual_increase_tolerance",
        config.residual_increase_tolerance,
        start,
        preserve_positive=True,
    )
    candidates = start[None, :, :] + alphas[:, None, None] * direction[None, :, :]
    pinned = context._owned_tensor("pinned")
    if pinned.numel() > 0:
        pinned_targets = start.index_select(0, pinned)
        candidates = candidates.index_copy(
            1,
            pinned,
            pinned_targets.expand(candidates.shape[0], pinned.numel(), 3),
        )

    position_finite = torch.isfinite(candidates).all(dim=(-2, -1))
    components = _common_objective_components_trusted(context, candidates)
    candidate_objectives = components["total"]
    candidate_residuals = _common_objective_residual_trusted(context, candidates)
    candidate_residual_norms = torch.linalg.vector_norm(candidate_residuals.flatten(start_dim=-2), dim=-1)
    minimum_determinants, minimum_singular_values = _candidate_geometry(context, candidates, position_finite)
    if pinned.numel() == 0:
        exact_pins = torch.ones(candidates.shape[0], dtype=torch.bool, device=start.device)
    else:
        exact_pins = (
            candidates.index_select(1, pinned)
            == start.index_select(0, pinned).expand(candidates.shape[0], pinned.numel(), 3)
        ).all(dim=(-2, -1))

    zero_index = config.candidate_alphas.index(0.0)
    start_objective = candidate_objectives[zero_index]
    start_residual_norm = candidate_residual_norms[zero_index]
    finite = (
        position_finite
        & torch.isfinite(candidate_objectives)
        & torch.isfinite(candidate_residual_norms)
        & torch.isfinite(minimum_determinants)
        & torch.isfinite(minimum_singular_values)
    )
    determinant_valid = minimum_determinants > minimum_determinant
    singular_value_valid = minimum_singular_values > minimum_singular_value
    objective_valid = candidate_objectives <= start_objective + objective_increase_tolerance
    residual_valid = candidate_residual_norms <= start_residual_norm + residual_increase_tolerance
    if not config.require_residual_nonincrease:
        residual_valid = torch.ones_like(residual_valid)
    nonzero = alphas != 0.0
    admissible = (
        nonzero
        & finite
        & exact_pins
        & determinant_valid
        & singular_value_valid
        & objective_valid
        & residual_valid
        & descent_direction_tensor
        & (not pcg_result.trace.breakdown)
    )
    candidate_indices = torch.arange(candidates.shape[0], dtype=torch.int64, device=start.device)
    sentinel = torch.full_like(candidate_indices, candidates.shape[0])
    selected_positive = torch.where(admissible, candidate_indices, sentinel).amin()
    has_positive = selected_positive < candidates.shape[0]
    selected_index_tensor = torch.where(
        has_positive,
        selected_positive,
        torch.tensor(zero_index, dtype=torch.int64, device=start.device),
    )
    selected_index = int(selected_index_tensor.detach().cpu())
    accepted = bool(has_positive.detach().cpu())
    positions = candidates[selected_index].clone() if accepted else start.clone()

    if accepted:
        reason = "accepted"
    elif pcg_result.trace.breakdown:
        reason = "pcg-breakdown"
    elif not bool(finite[zero_index].detach().cpu()):
        reason = "invalid-start"
    elif pcg_result.trace.stationary:
        reason = "stationary"
    elif not bool(descent_direction_tensor.detach().cpu()):
        reason = "non-descent-direction"
    else:
        reason = "no-admissible-candidate"

    candidate_traces = tuple(
        CorrectorCandidateTrace(
            index=index,
            alpha=float(config.candidate_alphas[index]),
            finite=bool(finite[index].detach().cpu()),
            exact_pins=bool(exact_pins[index].detach().cpu()),
            minimum_determinant=float(minimum_determinants[index].detach().cpu()),
            minimum_singular_value=float(minimum_singular_values[index].detach().cpu()),
            objective=float(candidate_objectives[index].detach().cpu()),
            raw_residual_norm=float(candidate_residual_norms[index].detach().cpu()),
            determinant_valid=bool(determinant_valid[index].detach().cpu()),
            singular_value_valid=bool(singular_value_valid[index].detach().cpu()),
            objective_nonincreasing=bool(objective_valid[index].detach().cpu()),
            residual_nonincreasing=bool(residual_valid[index].detach().cpu()),
            admissible_nonzero_step=bool(admissible[index].detach().cpu()),
        )
        for index in range(candidates.shape[0])
    )
    work = CorrectorWork(
        gradient_state_evaluations=1,
        tangent_builds=1,
        block_diagonal_builds=1,
        scheduled_pcg_iterations=config.pcg.iterations,
        active_pcg_iterations=pcg_result.trace.active_iterations,
        tangent_matrix_vector_products=pcg_result.trace.matrix_vector_products,
        block_preconditioner_applications=pcg_result.trace.preconditioner_applications,
        pcg_scalar_reductions=pcg_result.trace.scalar_reductions,
        candidate_count=len(config.candidate_alphas),
        candidate_objective_state_evaluations=len(config.candidate_alphas),
        candidate_residual_state_evaluations=len(config.candidate_alphas),
        candidate_determinant_state_evaluations=len(config.candidate_alphas),
        candidate_singular_value_state_evaluations=len(config.candidate_alphas),
    )
    trace = CorrectorTrace(
        common_objective_sha256=context.common_objective_sha256,
        start_objective=float(start_objective.detach().cpu()),
        start_raw_residual_norm=float(start_residual_norm.detach().cpu()),
        gradient_dot_direction=float(gradient_dot_direction_tensor.detach().cpu()),
        descent_direction=bool(descent_direction_tensor.detach().cpu()),
        pcg=pcg_result.trace,
        candidates=candidate_traces,
        selected_candidate_index=selected_index,
        selected_alpha=float(config.candidate_alphas[selected_index]),
        accepted=accepted,
        reason=reason,
        work=work,
    )
    return CorrectorResult(positions=positions, direction=direction, trace=trace)


__all__ = [
    "CorrectorCandidateTrace",
    "CorrectorConfig",
    "CorrectorResult",
    "CorrectorTrace",
    "CorrectorWork",
    "FixedPCGConfig",
    "FixedPCGResult",
    "FixedPCGTrace",
    "StableNeoHookeanGaussNewtonOperator",
    "common_objective_free_gradient",
    "correct_common_objective",
    "fixed_work_block_pcg",
    "stable_neo_hookean_gn_block_diagonal",
    "stable_neo_hookean_gn_matvec",
]
