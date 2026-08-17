# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Basic dense CPU Newton baseline for volumetric elastic dynamics.

This module is intentionally a correctness/reference implementation, not a
production Newton solver.  It eliminates Dirichlet degrees of freedom, forms a
dense float64 Hessian with PyTorch autograd, regularizes indefinite Hessians,
and uses Armijo backtracking.  The implementation is suitable for small meshes
and deliberately exposes failures instead of silently returning the last
iterate as a converged result.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import time
from collections.abc import Sequence

import numpy as np
import torch

from .potentials import incremental_potential_stable_neo_hookean
from .torch_solver import _build_J, compute_F

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _cpu_float64(value: np.ndarray | torch.Tensor | Sequence[float], name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        # ``TetBenchmarkScene`` intentionally exposes read-only arrays.  Make
        # the ownership copy before constructing a Tensor so PyTorch does not
        # warn about undefined writes through a non-writable NumPy view.
        tensor = torch.from_numpy(np.array(value, dtype=np.float64, copy=True))
    else:
        tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor.detach().clone()


def _cpu_int64_indices(
    value: np.ndarray | torch.Tensor | Sequence[int],
    name: str,
) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.array(value, copy=True))
    else:
        tensor = torch.as_tensor(value, device="cpu")
    if tensor.numel() == 0:
        return torch.empty(tensor.shape, dtype=torch.int64, device="cpu")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must contain integers, got dtype {tensor.dtype}")
    return tensor.to(dtype=torch.int64).detach().clone()


def _per_tet(
    value: float | np.ndarray | torch.Tensor,
    n_tets: int,
    name: str,
) -> torch.Tensor:
    tensor = _cpu_float64(value, name)
    if tensor.ndim == 0:
        tensor = tensor.expand(n_tets).clone()
    if tensor.shape != (n_tets,):
        raise ValueError(f"{name} must be scalar or shape ({n_tets},), got {tuple(tensor.shape)}")
    return tensor


@dataclasses.dataclass(frozen=True)
class NewtonProblem:
    """One contact-free implicit-Euler tet problem on CPU in float64.

    ``residual_scale`` is shared by every method/warm start for this problem:
    the free-gradient norm at the force-shifted inertial target, floored at
    1 N. ``setup_seconds`` includes validation, rest-geometry construction,
    and this reference-gradient evaluation.
    """

    rest_q: torch.Tensor
    tets: torch.Tensor
    J: torch.Tensor
    volume: torch.Tensor
    mass: torch.Tensor
    mu: torch.Tensor
    lam: torch.Tensor
    inertial_target: torch.Tensor
    pinned: torch.Tensor
    free: torch.Tensor
    pin_targets: torch.Tensor
    dt: float
    residual_scale: float
    setup_seconds: float
    residual_scale_seconds: float

    @property
    def n_vertices(self) -> int:
        return int(self.rest_q.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    @property
    def n_free_dofs(self) -> int:
        return int(self.free.numel() * 3)

    def positions_from_free(self, z: torch.Tensor) -> torch.Tensor:
        """Assemble full positions from flattened free degrees of freedom."""
        if z.shape != (self.n_free_dofs,):
            raise ValueError(f"free vector must have shape ({self.n_free_dofs},), got {tuple(z.shape)}")
        x = self.rest_q.index_copy(0, self.free, z.reshape(-1, 3))
        return x.index_copy(0, self.pinned, self.pin_targets)

    def free_from_positions(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract a detached flattened free vector from full positions."""
        x_tensor = _cpu_float64(x, "x")
        if x_tensor.shape != self.rest_q.shape:
            raise ValueError(f"x must have shape {tuple(self.rest_q.shape)}, got {tuple(x_tensor.shape)}")
        return x_tensor[self.free].reshape(-1).detach().clone()

    def objective_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the common scalar potential at full positions."""
        return incremental_potential_stable_neo_hookean(
            x,
            self.inertial_target,
            self.mass,
            self.tets,
            self.J,
            self.mu,
            self.lam,
            self.volume,
            self.dt,
        )["total"]

    def objective_free(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the common scalar potential over free DOFs."""
        return self.objective_positions(self.positions_from_free(z))


def build_newton_problem(
    rest_q: np.ndarray | torch.Tensor,
    tet_indices: np.ndarray | torch.Tensor,
    tet_poses: np.ndarray | torch.Tensor,
    mass: np.ndarray | torch.Tensor,
    mu: float | np.ndarray | torch.Tensor,
    lam: float | np.ndarray | torch.Tensor,
    dt: float,
    *,
    x_current: np.ndarray | torch.Tensor | None = None,
    velocity: np.ndarray | torch.Tensor | None = None,
    gravity: Sequence[float] = (0.0, 0.0, 0.0),
    external_force: np.ndarray | torch.Tensor | None = None,
    pinned_indices: np.ndarray | torch.Tensor | Sequence[int] = (),
    pin_targets: np.ndarray | torch.Tensor | None = None,
    inertial_target: np.ndarray | torch.Tensor | None = None,
) -> NewtonProblem:
    """Construct a validated common-objective Newton problem.

    ``tet_poses`` are the per-tet inverse rest matrices used by Newton.  Gravity
    and external force are folded into VBD's force-shifted inertial target.

    Args:
        rest_q: Rest positions [m], shape ``[V, 3]``.
        tet_indices: Tet vertex indices, shape ``[T, 4]``.
        tet_poses: Inverse rest matrices [1/m], shape ``[T, 3, 3]``.
        mass: Lumped vertex masses [kg], shape ``[V]``.
        mu: Scalar or per-tet first material coefficient [Pa].
        lam: Scalar or per-tet second material coefficient [Pa].
        dt: Implicit substep duration [s].
        x_current: Positions at the beginning of the substep [m]. Defaults to
            ``rest_q``.
        velocity: Beginning-of-substep velocities [m/s]. Defaults to zero.
        gravity: Constant acceleration [m/s^2].
        external_force: Nodal forces [N], shape ``[V, 3]``. Defaults to zero.
        pinned_indices: Dirichlet vertex indices.
        pin_targets: Dirichlet target positions [m], shape ``[P, 3]``, in the
            same order as ``pinned_indices``. Defaults to ``x_current`` at the
            pinned vertices, matching VBD snapshot semantics.
        inertial_target: Optional precomputed inertial target [m], shape
            ``[V, 3]``. When supplied, it must contain the exact pin targets.
            This is useful when matching a lower-precision solver's arithmetic
            rather than recomputing its predictor in float64.

    Returns:
        A CPU float64 :class:`NewtonProblem`.
    """
    build_start = time.perf_counter()
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError(f"dt must be finite and positive, got {dt}")

    rest = _cpu_float64(rest_q, "rest_q")
    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError(f"rest_q must have shape (V, 3), got {tuple(rest.shape)}")
    n_vertices = int(rest.shape[0])

    tets = _cpu_int64_indices(tet_indices, "tet_indices")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tet_indices must have shape (T, 4), got {tuple(tets.shape)}")
    if tets.numel() == 0:
        raise ValueError("at least one tetrahedron is required")
    if int(tets.min()) < 0 or int(tets.max()) >= n_vertices:
        raise ValueError("tet_indices contains an out-of-range vertex")
    n_tets = int(tets.shape[0])

    dm_inv = _cpu_float64(tet_poses, "tet_poses")
    if dm_inv.shape != (n_tets, 3, 3):
        raise ValueError(f"tet_poses must have shape ({n_tets}, 3, 3), got {tuple(dm_inv.shape)}")
    det_inv = torch.linalg.det(dm_inv)
    volume = 1.0 / (6.0 * det_inv)
    if not torch.isfinite(volume).all() or (volume <= 0.0).any():
        raise ValueError("tet_poses must describe finite positively oriented rest tetrahedra")
    J = _build_J(dm_inv)
    rest_f = compute_F(rest, tets, J)
    identity = torch.eye(3, dtype=torch.float64).expand(n_tets, -1, -1)
    if not torch.allclose(rest_f, identity, rtol=2.0e-5, atol=2.0e-5):
        max_error = float((rest_f - identity).abs().max())
        raise ValueError(f"tet_poses do not match rest_q/tet_indices (max rest-F error {max_error:.3e})")

    masses = _cpu_float64(mass, "mass")
    if masses.shape != (n_vertices,):
        raise ValueError(f"mass must have shape ({n_vertices},), got {tuple(masses.shape)}")
    if (masses < 0.0).any():
        raise ValueError("mass must be non-negative")

    mu_t = _per_tet(mu, n_tets, "mu")
    lam_t = _per_tet(lam, n_tets, "lam")
    if (mu_t < 0.0).any():
        raise ValueError("mu must be non-negative")
    if (lam_t < 0.0).any() or (((mu_t > 0.0) | (lam_t > 0.0)) & (lam_t <= 0.0)).any():
        raise ValueError("lambda must be positive on active tets")

    pinned_input = _cpu_int64_indices(pinned_indices, "pinned_indices").reshape(-1)
    if pinned_input.numel() > 0:
        if int(pinned_input.min()) < 0 or int(pinned_input.max()) >= n_vertices:
            raise ValueError("pinned_indices contains an out-of-range vertex")
        if torch.unique(pinned_input).numel() != pinned_input.numel():
            raise ValueError("pinned_indices must not contain duplicates")
        pin_order = torch.argsort(pinned_input)
        pinned = pinned_input[pin_order]
    else:
        pin_order = torch.empty(0, dtype=torch.int64)
        pinned = pinned_input
    mask = torch.ones(n_vertices, dtype=torch.bool, device="cpu")
    mask[pinned] = False
    free = torch.where(mask)[0]
    if free.numel() == 0:
        raise ValueError("at least one free vertex is required")
    if (masses[free] <= 0.0).any():
        raise ValueError("every free vertex must have positive mass")

    x_n = rest.clone() if x_current is None else _cpu_float64(x_current, "x_current")
    if x_n.shape != rest.shape:
        raise ValueError(f"x_current must have shape {tuple(rest.shape)}")
    v_n = torch.zeros_like(rest) if velocity is None else _cpu_float64(velocity, "velocity")
    if v_n.shape != rest.shape:
        raise ValueError(f"velocity must have shape {tuple(rest.shape)}")
    f_ext = torch.zeros_like(rest) if external_force is None else _cpu_float64(external_force, "external_force")
    if f_ext.shape != rest.shape:
        raise ValueError(f"external_force must have shape {tuple(rest.shape)}")
    gravity_t = _cpu_float64(gravity, "gravity")
    if gravity_t.shape != (3,):
        raise ValueError("gravity must have shape (3,)")

    acceleration = gravity_t.expand_as(rest).clone()
    acceleration[free] += f_ext[free] / masses[free, None]
    computed_inertial_target = x_n + dt * v_n + dt * dt * acceleration

    if pin_targets is None:
        targets = x_n[pinned].clone()
    else:
        targets_input = _cpu_float64(pin_targets, "pin_targets")
        if targets_input.shape != (pinned_input.numel(), 3):
            raise ValueError(
                f"pin_targets must have shape ({pinned_input.numel()}, 3), got {tuple(targets_input.shape)}"
            )
        targets = targets_input[pin_order]
    if inertial_target is None:
        target = computed_inertial_target
        target[pinned] = targets
    else:
        target = _cpu_float64(inertial_target, "inertial_target")
        if target.shape != rest.shape:
            raise ValueError(f"inertial_target must have shape {tuple(rest.shape)}")
        if not torch.equal(target[pinned], targets):
            raise ValueError("inertial_target must contain the exact Dirichlet targets")

    problem = NewtonProblem(
        rest_q=rest,
        tets=tets,
        J=J,
        volume=volume,
        mass=masses,
        mu=mu_t,
        lam=lam_t,
        inertial_target=target,
        pinned=pinned,
        free=free,
        pin_targets=targets,
        dt=float(dt),
        residual_scale=math.nan,
        setup_seconds=math.nan,
        residual_scale_seconds=math.nan,
    )
    residual_scale_start = time.perf_counter()
    reference_x = target.index_copy(0, pinned, targets)
    reference_z = problem.free_from_positions(reference_x).requires_grad_(True)
    reference_value = problem.objective_free(reference_z)
    (reference_gradient,) = torch.autograd.grad(reference_value, reference_z)
    reference_gradient_norm = float(torch.linalg.vector_norm(reference_gradient))
    if not math.isfinite(reference_gradient_norm):
        raise ValueError("the inertial-target reference gradient must be finite")
    # Use one common denominator for every warm start and method. The 1 N floor
    # is the standard absolute-force fallback when the inertial target is
    # already stationary (for example an unloaded rest state).
    residual_scale = max(reference_gradient_norm, 1.0)
    residual_scale_seconds = time.perf_counter() - residual_scale_start
    return dataclasses.replace(
        problem,
        residual_scale=residual_scale,
        setup_seconds=time.perf_counter() - build_start,
        residual_scale_seconds=residual_scale_seconds,
    )


@dataclasses.dataclass(frozen=True)
class NewtonConfig:
    """Termination, regularization, and line-search settings."""

    max_iterations: int = 50
    gradient_absolute_tolerance: float = 1.0e-9
    gradient_relative_tolerance: float = 1.0e-8
    step_relative_tolerance: float = 1.0e-12
    armijo: float = 1.0e-4
    backtrack: float = 0.5
    max_line_search_steps: int = 30
    minimum_eigenvalue_relative: float = 1.0e-9
    regularization_growth: float = 10.0
    max_regularization_attempts: int = 12

    def validate(self) -> None:
        for name in ("max_iterations", "max_line_search_steps", "max_regularization_attempts"):
            if not isinstance(getattr(self, name), numbers.Integral) or isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be an integer")
        for name in (
            "gradient_absolute_tolerance",
            "gradient_relative_tolerance",
            "step_relative_tolerance",
            "armijo",
            "backtrack",
            "minimum_eigenvalue_relative",
            "regularization_growth",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.gradient_absolute_tolerance < 0.0 or self.gradient_relative_tolerance < 0.0:
            raise ValueError("gradient tolerances must be non-negative")
        if self.step_relative_tolerance < 0.0:
            raise ValueError("step_relative_tolerance must be non-negative")
        if not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must lie in (0, 1)")
        if not 0.0 < self.backtrack < 1.0:
            raise ValueError("backtrack must lie in (0, 1)")
        if self.max_line_search_steps < 1:
            raise ValueError("max_line_search_steps must be positive")
        if self.minimum_eigenvalue_relative <= 0.0:
            raise ValueError("minimum_eigenvalue_relative must be positive")
        if self.regularization_growth <= 1.0:
            raise ValueError("regularization_growth must exceed one")
        if self.max_regularization_attempts < 1:
            raise ValueError("max_regularization_attempts must be positive")


@dataclasses.dataclass(frozen=True)
class NewtonIteration:
    """One fully evaluated iterate in a Newton trace."""

    iteration: int
    objective: float
    gradient_norm: float
    relative_residual: float
    accepted_step_norm: float
    accepted_step_size: float
    regularization: float
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True)
class NewtonResult:
    """Newton solution, status, convergence trace, and work accounting.

    Objective counts are top-level scalar evaluations used for value/gradient
    passes or line-search trials. A Hessian request is counted separately as
    one composite derivative evaluation, irrespective of PyTorch's internal
    vectorization. Problem setup and its reference-gradient evaluation are
    timed separately from ``total_seconds`` so cold and reused-problem timings
    cannot be confused.
    """

    x: torch.Tensor
    converged: bool
    reason: str
    accepted_iterations: int
    total_seconds: float
    problem_setup_seconds: float
    residual_scale_setup_seconds: float
    objective_gradient_seconds: float
    hessian_seconds: float
    linear_solve_seconds: float
    line_search_seconds: float
    objective_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    eigenvalue_evaluations: int
    factorization_attempts: int
    line_search_trials: int
    trace: tuple[NewtonIteration, ...]

    @property
    def end_to_end_seconds(self) -> float:
        """Cold problem construction plus solve time [s].

        For repeated solves that reuse a :class:`NewtonProblem`, report
        ``total_seconds`` instead.
        """
        return self.problem_setup_seconds + self.total_seconds

    @property
    def final_objective(self) -> float:
        return self.trace[-1].objective if self.trace else math.nan

    @property
    def final_gradient_norm(self) -> float:
        return self.trace[-1].gradient_norm if self.trace else math.nan

    @property
    def final_relative_residual(self) -> float:
        return self.trace[-1].relative_residual if self.trace else math.nan


@dataclasses.dataclass
class _NewtonWork:
    objective_gradient_seconds: float = 0.0
    hessian_seconds: float = 0.0
    linear_solve_seconds: float = 0.0
    line_search_seconds: float = 0.0
    objective_evaluations: int = 0
    gradient_evaluations: int = 0
    hessian_evaluations: int = 0
    eigenvalue_evaluations: int = 0
    factorization_attempts: int = 0
    line_search_trials: int = 0


def _value_and_gradient(
    problem: NewtonProblem,
    z: torch.Tensor,
    work: _NewtonWork,
) -> tuple[torch.Tensor, torch.Tensor]:
    evaluation_start = time.perf_counter()
    z_var = z.detach().requires_grad_(True)
    value = problem.objective_free(z_var)
    (gradient,) = torch.autograd.grad(value, z_var)
    work.objective_evaluations += 1
    work.gradient_evaluations += 1
    work.objective_gradient_seconds += time.perf_counter() - evaluation_start
    return value.detach(), gradient.detach()


def _regularized_direction(
    problem: NewtonProblem,
    z: torch.Tensor,
    gradient: torch.Tensor,
    config: NewtonConfig,
    work: _NewtonWork,
) -> tuple[torch.Tensor | None, float, str | None]:
    hessian_start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(problem.objective_free, z, vectorize=True)
    hessian = 0.5 * (hessian + hessian.T)
    work.hessian_evaluations += 1
    work.hessian_seconds += time.perf_counter() - hessian_start
    if not torch.isfinite(hessian).all():
        return None, math.nan, "nonfinite"

    linear_solve_start = time.perf_counter()
    diagonal_scale = max(float(hessian.diagonal().abs().max()), 1.0)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    work.eigenvalue_evaluations += 1
    if not torch.isfinite(eigenvalues).all():
        work.linear_solve_seconds += time.perf_counter() - linear_solve_start
        return None, math.nan, "nonfinite"
    minimum_target = config.minimum_eigenvalue_relative * diagonal_scale
    regularization = max(0.0, minimum_target - float(eigenvalues[0]))
    identity = torch.eye(hessian.shape[0], dtype=hessian.dtype)

    for _ in range(config.max_regularization_attempts):
        work.factorization_attempts += 1
        shifted = hessian + regularization * identity
        factor, info = torch.linalg.cholesky_ex(shifted)
        if int(info) == 0:
            direction = torch.cholesky_solve((-gradient)[:, None], factor).squeeze(1)
            directional_derivative = float(torch.dot(gradient, direction))
            if torch.isfinite(direction).all() and directional_derivative < 0.0:
                work.linear_solve_seconds += time.perf_counter() - linear_solve_start
                return direction, regularization, None
        regularization = max(
            minimum_target,
            regularization * config.regularization_growth,
        )
    work.linear_solve_seconds += time.perf_counter() - linear_solve_start
    return None, regularization, "linear_solve"


def solve_newton(
    problem: NewtonProblem,
    x_initial: np.ndarray | torch.Tensor | None = None,
    config: NewtonConfig | None = None,
) -> NewtonResult:
    """Solve one common-objective step with dense line-searched Newton.

    Args:
        problem: Validated CPU float64 problem.
        x_initial: Initial full positions [m]. Defaults to the inertial target
            with Dirichlet targets imposed.
        config: Optional solver configuration.

    Returns:
        Result with an evaluated initial trace entry and one entry after every
        accepted update.
    """
    cfg = NewtonConfig() if config is None else config
    cfg.validate()
    start = time.perf_counter()
    work = _NewtonWork()
    trace: list[NewtonIteration] = []
    accepted_iterations = 0

    def finish(x: torch.Tensor, converged: bool, reason: str) -> NewtonResult:
        return NewtonResult(
            x=x,
            converged=converged,
            reason=reason,
            accepted_iterations=accepted_iterations,
            total_seconds=time.perf_counter() - start,
            problem_setup_seconds=problem.setup_seconds,
            residual_scale_setup_seconds=problem.residual_scale_seconds,
            objective_gradient_seconds=work.objective_gradient_seconds,
            hessian_seconds=work.hessian_seconds,
            linear_solve_seconds=work.linear_solve_seconds,
            line_search_seconds=work.line_search_seconds,
            objective_evaluations=work.objective_evaluations,
            gradient_evaluations=work.gradient_evaluations,
            hessian_evaluations=work.hessian_evaluations,
            eigenvalue_evaluations=work.eigenvalue_evaluations,
            factorization_attempts=work.factorization_attempts,
            line_search_trials=work.line_search_trials,
            trace=tuple(trace),
        )

    if x_initial is None:
        x0 = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets)
    else:
        if isinstance(x_initial, np.ndarray):
            x0 = torch.from_numpy(np.array(x_initial, dtype=np.float64, copy=True))
        else:
            x0 = torch.as_tensor(x_initial, dtype=torch.float64, device="cpu").detach().clone()
        if x0.shape != problem.rest_q.shape:
            raise ValueError(f"x_initial must have shape {tuple(problem.rest_q.shape)}")
        if not torch.isfinite(x0).all():
            x0 = x0.index_copy(0, problem.pinned, problem.pin_targets)
            return finish(x0, False, "nonfinite")
    z = problem.free_from_positions(x0)

    previous_step_norm = 0.0
    previous_step_size = 0.0
    previous_regularization = 0.0

    while True:
        value, gradient = _value_and_gradient(problem, z, work)
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            x = problem.positions_from_free(z).detach()
            return finish(x, False, "nonfinite")

        gradient_norm = float(torch.linalg.vector_norm(gradient))
        residual_denominator = max(problem.residual_scale, 1.0e-30)
        relative_residual = gradient_norm / residual_denominator
        trace.append(
            NewtonIteration(
                iteration=accepted_iterations,
                objective=float(value),
                gradient_norm=gradient_norm,
                relative_residual=relative_residual,
                accepted_step_norm=previous_step_norm,
                accepted_step_size=previous_step_size,
                regularization=previous_regularization,
                elapsed_seconds=time.perf_counter() - start,
            )
        )

        gradient_converged = (
            gradient_norm <= cfg.gradient_absolute_tolerance or relative_residual <= cfg.gradient_relative_tolerance
        )
        if gradient_converged:
            return finish(problem.positions_from_free(z).detach(), True, "gradient")
        if accepted_iterations >= cfg.max_iterations:
            return finish(problem.positions_from_free(z).detach(), False, "max_iterations")

        direction, regularization, direction_failure = _regularized_direction(problem, z, gradient, cfg, work)
        if direction is None:
            return finish(
                problem.positions_from_free(z).detach(),
                False,
                direction_failure or "linear_solve",
            )

        directional_derivative = float(torch.dot(gradient, direction))
        step_size = 1.0
        accepted = False
        candidate = z
        line_search_start = time.perf_counter()
        for _ in range(cfg.max_line_search_steps):
            trial = z + step_size * direction
            trial_value = problem.objective_free(trial)
            work.objective_evaluations += 1
            work.line_search_trials += 1
            sufficient_decrease = float(value) + cfg.armijo * step_size * directional_derivative
            if torch.isfinite(trial_value) and float(trial_value) <= sufficient_decrease:
                candidate = trial.detach()
                accepted = True
                break
            step_size *= cfg.backtrack
        work.line_search_seconds += time.perf_counter() - line_search_start
        if not accepted:
            return finish(problem.positions_from_free(z).detach(), False, "line_search")

        step_norm = float(torch.linalg.vector_norm(candidate - z))
        z_scale = max(float(torch.linalg.vector_norm(z)), 1.0)
        z = candidate
        accepted_iterations += 1
        previous_step_norm = step_norm
        previous_step_size = step_size
        previous_regularization = regularization
        if step_norm <= cfg.step_relative_tolerance * z_scale:
            # Evaluate and record the stalled point before returning.  A small
            # step without a converged gradient is not called success.
            value, gradient = _value_and_gradient(problem, z, work)
            if not torch.isfinite(value) or not torch.isfinite(gradient).all():
                return finish(problem.positions_from_free(z).detach(), False, "nonfinite")
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            relative_residual = gradient_norm / max(problem.residual_scale, 1.0e-30)
            trace.append(
                NewtonIteration(
                    iteration=accepted_iterations,
                    objective=float(value),
                    gradient_norm=gradient_norm,
                    relative_residual=relative_residual,
                    accepted_step_norm=previous_step_norm,
                    accepted_step_size=previous_step_size,
                    regularization=previous_regularization,
                    elapsed_seconds=time.perf_counter() - start,
                )
            )
            gradient_converged = (
                gradient_norm <= cfg.gradient_absolute_tolerance or relative_residual <= cfg.gradient_relative_tolerance
            )
            return finish(
                problem.positions_from_free(z).detach(),
                gradient_converged,
                "gradient" if gradient_converged else "stalled",
            )
