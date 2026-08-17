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


_RESIDUAL_POLISH_REASONS = (
    "gradient",
    "max_iterations",
    "nonfinite",
    "nonfinite_hessian",
    "non_spd_hessian",
    "factorization",
    "non_descent",
    "residual_line_search",
)
_RESIDUAL_POLISH_OBJECTIVE_ROUNDOFF_FACTOR = 8.0


@dataclasses.dataclass(frozen=True)
class NewtonResidualPolishConfig:
    """Bounded strict-reference root-polish configuration."""

    max_iterations: int
    gradient_absolute_tolerance: float
    gradient_relative_tolerance: float
    armijo: float
    backtrack: float
    max_line_search_steps: int

    @classmethod
    def from_newton_config(cls, config: NewtonConfig) -> NewtonResidualPolishConfig:
        """Copy only the primary settings used by residual polish."""
        config.validate()
        return cls(
            max_iterations=config.max_iterations,
            gradient_absolute_tolerance=config.gradient_absolute_tolerance,
            gradient_relative_tolerance=config.gradient_relative_tolerance,
            armijo=config.armijo,
            backtrack=config.backtrack,
            max_line_search_steps=config.max_line_search_steps,
        )

    def validate(self) -> None:
        """Validate bounded work and convergence settings."""
        for name in ("max_iterations", "max_line_search_steps"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        for name in (
            "gradient_absolute_tolerance",
            "gradient_relative_tolerance",
            "armijo",
            "backtrack",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.max_line_search_steps < 1:
            raise ValueError("max_line_search_steps must be positive")
        if self.gradient_absolute_tolerance < 0.0 or self.gradient_relative_tolerance < 0.0:
            raise ValueError("gradient tolerances must be non-negative")
        if not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must lie in (0, 1)")
        if not 0.0 < self.backtrack < 1.0:
            raise ValueError("backtrack must lie in (0, 1)")

    def deterministic_record(self) -> dict[str, object]:
        """Return the fixed numerical contract without timings."""
        return {
            **dataclasses.asdict(self),
            "contract": "strict-reference-residual-newton-v1",
            "merit": "half-squared-gradient-norm",
            "hessian_policy": "finite-symmetric-unregularized-spd",
            "line_search_policy": "armijo-and-strict-residual-decrease",
            "constraint_policy": "free-dof-elimination-exact-pins",
            "objective_roundoff_guard": "E1 <= E0 + 8*eps*max(1,abs(E0),abs(E1))",
            "objective_roundoff_factor": _RESIDUAL_POLISH_OBJECTIVE_ROUNDOFF_FACTOR,
        }


@dataclasses.dataclass(frozen=True)
class NewtonResidualPolishIteration:
    """One deterministic residual-polish iterate and its outgoing step."""

    iteration: int
    objective: float
    gradient_norm: float
    relative_residual: float
    residual_merit: float
    accepted_step_norm: float
    accepted_step_size: float
    merit_directional_derivative: float | None
    hessian_minimum_eigenvalue: float | None
    hessian_maximum_eigenvalue: float | None


@dataclasses.dataclass(frozen=True)
class NewtonResidualPolishTiming:
    """Timing-only residual-polish measurements."""

    total_seconds: float
    objective_gradient_seconds: float
    hessian_seconds: float
    linear_solve_seconds: float
    line_search_seconds: float
    trace_elapsed_seconds: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class NewtonResidualPolishResult:
    """Fail-closed stationary-root polish result with explicit work."""

    x: torch.Tensor
    converged: bool
    reason: str
    accepted_iterations: int
    residual_scale: float
    gradient_limit: float
    trace: tuple[NewtonResidualPolishIteration, ...]
    objective_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    eigenvalue_evaluations: int
    factorization_attempts: int
    line_search_trials: int
    timing: NewtonResidualPolishTiming

    def __post_init__(self) -> None:
        if self.reason not in _RESIDUAL_POLISH_REASONS:
            raise ValueError(f"unknown residual-polish reason: {self.reason}")
        if self.converged != (self.reason == "gradient"):
            raise ValueError("residual-polish convergence must agree with the gradient reason")
        for name in (
            "accepted_iterations",
            "objective_evaluations",
            "gradient_evaluations",
            "hessian_evaluations",
            "eigenvalue_evaluations",
            "factorization_attempts",
            "line_search_trials",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        if not math.isfinite(self.gradient_limit) or self.gradient_limit < 0.0:
            raise ValueError("gradient_limit must be finite and non-negative")
        object.__setattr__(self, "x", self.x.detach().clone())
        object.__setattr__(self, "trace", tuple(self.trace))
        if len(self.timing.trace_elapsed_seconds) != len(self.trace):
            raise ValueError("trace timings do not match residual-polish iterations")
        if not self.trace:
            if self.reason != "nonfinite" or self.accepted_iterations != 0:
                raise ValueError("only an initially nonfinite solve may have an empty residual-polish trace")
        elif len(self.trace) != self.accepted_iterations + 1:
            raise ValueError("residual-polish trace length does not match its accepted-update count")
        if tuple(item.iteration for item in self.trace) != tuple(range(len(self.trace))):
            raise ValueError("residual-polish trace iteration indices must be contiguous")
        accepted_outgoing_items = 0
        roundoff_factor = 8.0 * torch.finfo(torch.float64).eps
        for index, item in enumerate(self.trace):
            if math.isfinite(item.gradient_norm):
                expected_merit = 0.5 * item.gradient_norm * item.gradient_norm
                expected_relative = item.gradient_norm / self.residual_scale
                merit_roundoff = roundoff_factor * max(1.0, abs(item.residual_merit), abs(expected_merit))
                relative_roundoff = roundoff_factor * max(
                    1.0,
                    abs(item.relative_residual),
                    abs(expected_relative),
                )
                if not math.isfinite(item.residual_merit) or abs(item.residual_merit - expected_merit) > merit_roundoff:
                    raise ValueError("residual-polish trace merit does not match its gradient")
                if (
                    not math.isfinite(item.relative_residual)
                    or abs(item.relative_residual - expected_relative) > relative_roundoff
                ):
                    raise ValueError("residual-polish trace relative residual does not match its gradient")

            has_step_size = item.accepted_step_size != 0.0
            has_step_norm = item.accepted_step_norm != 0.0
            if has_step_size != has_step_norm:
                raise ValueError("residual-polish outgoing step size and norm are inconsistent")
            if not has_step_size:
                continue
            accepted_outgoing_items += 1
            if index + 1 >= len(self.trace):
                raise ValueError("residual-polish final trace item cannot contain an accepted outgoing step")
            if (
                not math.isfinite(item.accepted_step_size)
                or not 0.0 < item.accepted_step_size <= 1.0
                or not math.isfinite(item.accepted_step_norm)
                or item.accepted_step_norm <= 0.0
            ):
                raise ValueError("residual-polish accepted outgoing step is invalid")
            if item.merit_directional_derivative is None or not math.isfinite(item.merit_directional_derivative):
                raise ValueError("residual-polish accepted outgoing direction must be finite")
            if item.merit_directional_derivative >= 0.0:
                raise ValueError("residual-polish accepted outgoing direction must be descending")
            minimum_eigenvalue = item.hessian_minimum_eigenvalue
            maximum_eigenvalue = item.hessian_maximum_eigenvalue
            if (
                minimum_eigenvalue is None
                or maximum_eigenvalue is None
                or not math.isfinite(minimum_eigenvalue)
                or not math.isfinite(maximum_eigenvalue)
                or minimum_eigenvalue <= 0.0
                or maximum_eigenvalue < minimum_eigenvalue
            ):
                raise ValueError("residual-polish accepted outgoing Hessian must be finite SPD")
            next_gradient_norm = self.trace[index + 1].gradient_norm
            if not math.isfinite(next_gradient_norm) or next_gradient_norm >= item.gradient_norm:
                raise ValueError("residual-polish accepted outgoing step must strictly lower the gradient norm")
            next_objective = self.trace[index + 1].objective
            objective_roundoff_guard = (
                _RESIDUAL_POLISH_OBJECTIVE_ROUNDOFF_FACTOR
                * torch.finfo(torch.float64).eps
                * max(1.0, abs(item.objective), abs(next_objective))
            )
            if (
                not math.isfinite(item.objective)
                or not math.isfinite(next_objective)
                or next_objective > item.objective + objective_roundoff_guard
            ):
                raise ValueError("residual-polish accepted outgoing step violates the objective roundoff guard")
        if accepted_outgoing_items != self.accepted_iterations:
            raise ValueError("residual-polish accepted outgoing trace count is invalid")
        if (
            self.objective_evaluations != self.gradient_evaluations
            or self.objective_evaluations != len(self.trace) + self.line_search_trials
        ):
            raise ValueError("residual-polish objective/gradient work accounting is invalid")
        if not (self.accepted_iterations <= self.hessian_evaluations <= self.accepted_iterations + 1):
            raise ValueError("residual-polish Hessian work accounting is invalid")
        if not (self.eigenvalue_evaluations <= self.hessian_evaluations <= self.eigenvalue_evaluations + 1):
            raise ValueError("residual-polish eigenvalue work accounting is invalid")
        if not (self.accepted_iterations <= self.factorization_attempts <= self.eigenvalue_evaluations):
            raise ValueError("residual-polish factorization work accounting is invalid")
        if self.line_search_trials < self.accepted_iterations:
            raise ValueError("residual-polish line-search work accounting is invalid")

        timing_values = (
            self.timing.total_seconds,
            self.timing.objective_gradient_seconds,
            self.timing.hessian_seconds,
            self.timing.linear_solve_seconds,
            self.timing.line_search_seconds,
            *self.timing.trace_elapsed_seconds,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in timing_values):
            raise ValueError("residual-polish timings must be finite and non-negative")
        if any(
            after < before
            for before, after in zip(
                self.timing.trace_elapsed_seconds,
                self.timing.trace_elapsed_seconds[1:],
                strict=False,
            )
        ):
            raise ValueError("residual-polish trace timings must be monotone")

        if self.converged:
            if not torch.isfinite(self.x).all():
                raise ValueError("converged residual-polish positions must be finite")
            for item in self.trace:
                required_scalars = (
                    item.objective,
                    item.gradient_norm,
                    item.relative_residual,
                    item.residual_merit,
                    item.accepted_step_norm,
                    item.accepted_step_size,
                )
                optional_scalars = (
                    item.merit_directional_derivative,
                    item.hessian_minimum_eigenvalue,
                    item.hessian_maximum_eigenvalue,
                )
                if any(not math.isfinite(value) for value in required_scalars) or any(
                    value is not None and not math.isfinite(value) for value in optional_scalars
                ):
                    raise ValueError("converged residual-polish trace scalars must be finite")
            if self.final_gradient_norm > self.gradient_limit:
                raise ValueError("converged residual-polish gradient exceeds its configured limit")
            final = self.trace[-1]
            if (
                final.accepted_step_size != 0.0
                or final.accepted_step_norm != 0.0
                or final.merit_directional_derivative is not None
                or final.hessian_minimum_eigenvalue is not None
                or final.hessian_maximum_eigenvalue is not None
            ):
                raise ValueError("converged residual-polish final trace item must have no outgoing step")
            if not (
                self.hessian_evaluations
                == self.eigenvalue_evaluations
                == self.factorization_attempts
                == self.accepted_iterations
            ):
                raise ValueError("converged residual-polish derivative work must equal accepted updates")

    @property
    def final_objective(self) -> float:
        """Final common objective, or NaN when no iterate was evaluated."""
        return self.trace[-1].objective if self.trace else math.nan

    @property
    def final_gradient_norm(self) -> float:
        """Final free-gradient norm, or NaN when no iterate was evaluated."""
        return self.trace[-1].gradient_norm if self.trace else math.nan

    @property
    def final_relative_residual(self) -> float:
        """Final relative residual, or NaN when no iterate was evaluated."""
        return self.trace[-1].relative_residual if self.trace else math.nan

    def deterministic_record(self) -> dict[str, object]:
        """Return deterministic status, trace, and work counters."""

        def finite_or_none(value: float | None) -> float | None:
            if value is None or not math.isfinite(value):
                return None
            return value

        return {
            "converged": self.converged,
            "reason": self.reason,
            "accepted_iterations": self.accepted_iterations,
            "residual_scale": self.residual_scale,
            "gradient_limit": self.gradient_limit,
            "final_objective": finite_or_none(self.final_objective),
            "final_gradient_norm": finite_or_none(self.final_gradient_norm),
            "final_relative_residual": finite_or_none(self.final_relative_residual),
            "work": {
                "objective_evaluations": self.objective_evaluations,
                "gradient_evaluations": self.gradient_evaluations,
                "hessian_evaluations": self.hessian_evaluations,
                "eigenvalue_evaluations": self.eigenvalue_evaluations,
                "factorization_attempts": self.factorization_attempts,
                "line_search_trials": self.line_search_trials,
            },
            "trace": [
                {
                    "iteration": item.iteration,
                    "objective": finite_or_none(item.objective),
                    "gradient_norm": finite_or_none(item.gradient_norm),
                    "relative_residual": finite_or_none(item.relative_residual),
                    "residual_merit": finite_or_none(item.residual_merit),
                    "accepted_step_norm": finite_or_none(item.accepted_step_norm),
                    "accepted_step_size": finite_or_none(item.accepted_step_size),
                    "merit_directional_derivative": finite_or_none(item.merit_directional_derivative),
                    "hessian_minimum_eigenvalue": finite_or_none(item.hessian_minimum_eigenvalue),
                    "hessian_maximum_eigenvalue": finite_or_none(item.hessian_maximum_eigenvalue),
                }
                for item in self.trace
            ],
        }

    def timing_record(self) -> dict[str, object]:
        """Return timing fields kept outside deterministic content hashes."""
        return {
            "total_seconds": self.timing.total_seconds,
            "objective_gradient_seconds": self.timing.objective_gradient_seconds,
            "hessian_seconds": self.timing.hessian_seconds,
            "linear_solve_seconds": self.timing.linear_solve_seconds,
            "line_search_seconds": self.timing.line_search_seconds,
            "trace_elapsed_seconds": list(self.timing.trace_elapsed_seconds),
        }


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


@dataclasses.dataclass
class _ResidualPolishWork:
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


def solve_newton_residual_polish(
    problem: NewtonProblem,
    x_initial: np.ndarray | torch.Tensor,
    config: NewtonResidualPolishConfig,
) -> NewtonResidualPolishResult:
    """Polish a strict reference by globalizing Newton on residual norm.

    This is a separate stationary-root solver. It never changes the behavior,
    counters, or result of :func:`solve_newton`. Every accepted update must
    satisfy Armijo and strict decrease for ``0.5 * ||gradient||^2``. The
    common objective may increase only within a fixed ``8 * eps`` relative
    roundoff guard. Hessians must be finite and positive definite without
    regularization.
    """
    config.validate()
    start = time.perf_counter()
    work = _ResidualPolishWork()
    trace: list[NewtonResidualPolishIteration] = []
    trace_elapsed_seconds: list[float] = []
    accepted_iterations = 0
    gradient_limit = max(
        config.gradient_absolute_tolerance,
        config.gradient_relative_tolerance * problem.residual_scale,
    )

    if isinstance(x_initial, np.ndarray):
        x0 = torch.from_numpy(np.array(x_initial, dtype=np.float64, copy=True))
    else:
        x0 = torch.as_tensor(x_initial, dtype=torch.float64, device="cpu").detach().clone()
    if x0.shape != problem.rest_q.shape:
        raise ValueError(f"x_initial must have shape {tuple(problem.rest_q.shape)}")

    def finish(x: torch.Tensor, converged: bool, reason: str) -> NewtonResidualPolishResult:
        timing = NewtonResidualPolishTiming(
            total_seconds=time.perf_counter() - start,
            objective_gradient_seconds=work.objective_gradient_seconds,
            hessian_seconds=work.hessian_seconds,
            linear_solve_seconds=work.linear_solve_seconds,
            line_search_seconds=work.line_search_seconds,
            trace_elapsed_seconds=tuple(trace_elapsed_seconds),
        )
        return NewtonResidualPolishResult(
            x=x,
            converged=converged,
            reason=reason,
            accepted_iterations=accepted_iterations,
            residual_scale=problem.residual_scale,
            gradient_limit=gradient_limit,
            trace=tuple(trace),
            objective_evaluations=work.objective_evaluations,
            gradient_evaluations=work.gradient_evaluations,
            hessian_evaluations=work.hessian_evaluations,
            eigenvalue_evaluations=work.eigenvalue_evaluations,
            factorization_attempts=work.factorization_attempts,
            line_search_trials=work.line_search_trials,
            timing=timing,
        )

    if not torch.isfinite(x0).all():
        x0 = x0.index_copy(0, problem.pinned, problem.pin_targets)
        return finish(x0, False, "nonfinite")
    z = problem.free_from_positions(x0)

    def value_and_gradient(variable: torch.Tensor, *, line_search: bool) -> tuple[torch.Tensor, torch.Tensor]:
        evaluation_start = time.perf_counter()
        differentiable = variable.detach().requires_grad_(True)
        value = problem.objective_free(differentiable)
        (gradient,) = torch.autograd.grad(value, differentiable)
        elapsed = time.perf_counter() - evaluation_start
        if not line_search:
            work.objective_gradient_seconds += elapsed
        work.objective_evaluations += 1
        work.gradient_evaluations += 1
        return value.detach(), gradient.detach()

    def append_trace(
        iteration: int,
        objective: float,
        current_gradient_norm: float,
        current_relative_residual: float,
        current_residual_merit: float,
        *,
        accepted_step_norm: float = 0.0,
        accepted_step_size: float = 0.0,
        merit_directional_derivative: float | None = None,
        hessian_minimum_eigenvalue: float | None = None,
        hessian_maximum_eigenvalue: float | None = None,
    ) -> None:
        trace.append(
            NewtonResidualPolishIteration(
                iteration=iteration,
                objective=objective,
                gradient_norm=current_gradient_norm,
                relative_residual=current_relative_residual,
                residual_merit=current_residual_merit,
                accepted_step_norm=accepted_step_norm,
                accepted_step_size=accepted_step_size,
                merit_directional_derivative=merit_directional_derivative,
                hessian_minimum_eigenvalue=hessian_minimum_eigenvalue,
                hessian_maximum_eigenvalue=hessian_maximum_eigenvalue,
            )
        )
        trace_elapsed_seconds.append(time.perf_counter() - start)

    while True:
        value, gradient = value_and_gradient(z, line_search=False)
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        relative_residual = gradient_norm / max(problem.residual_scale, 1.0e-30)
        residual_merit = 0.5 * gradient_norm * gradient_norm
        current_trace = (
            accepted_iterations,
            float(value),
            gradient_norm,
            relative_residual,
            residual_merit,
        )

        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            append_trace(*current_trace)
            return finish(problem.positions_from_free(z).detach(), False, "nonfinite")
        if gradient_norm <= gradient_limit:
            append_trace(*current_trace)
            return finish(problem.positions_from_free(z).detach(), True, "gradient")
        if accepted_iterations >= config.max_iterations:
            append_trace(*current_trace)
            return finish(problem.positions_from_free(z).detach(), False, "max_iterations")

        hessian_start = time.perf_counter()
        matrix = torch.autograd.functional.hessian(problem.objective_free, z, vectorize=True)
        matrix = 0.5 * (matrix + matrix.T)
        work.hessian_evaluations += 1
        work.hessian_seconds += time.perf_counter() - hessian_start
        if not torch.isfinite(matrix).all():
            append_trace(*current_trace)
            return finish(problem.positions_from_free(z).detach(), False, "nonfinite_hessian")

        linear_solve_start = time.perf_counter()
        eigenvalues = torch.linalg.eigvalsh(matrix)
        work.eigenvalue_evaluations += 1
        minimum_eigenvalue = float(eigenvalues[0])
        maximum_eigenvalue = float(eigenvalues[-1])
        if not torch.isfinite(eigenvalues).all():
            work.linear_solve_seconds += time.perf_counter() - linear_solve_start
            append_trace(
                *current_trace,
                hessian_minimum_eigenvalue=minimum_eigenvalue,
                hessian_maximum_eigenvalue=maximum_eigenvalue,
            )
            return finish(problem.positions_from_free(z).detach(), False, "nonfinite_hessian")
        if minimum_eigenvalue <= 0.0:
            work.linear_solve_seconds += time.perf_counter() - linear_solve_start
            append_trace(
                *current_trace,
                hessian_minimum_eigenvalue=minimum_eigenvalue,
                hessian_maximum_eigenvalue=maximum_eigenvalue,
            )
            return finish(problem.positions_from_free(z).detach(), False, "non_spd_hessian")

        work.factorization_attempts += 1
        factor, info = torch.linalg.cholesky_ex(matrix)
        if int(info) != 0:
            work.linear_solve_seconds += time.perf_counter() - linear_solve_start
            append_trace(
                *current_trace,
                hessian_minimum_eigenvalue=minimum_eigenvalue,
                hessian_maximum_eigenvalue=maximum_eigenvalue,
            )
            return finish(problem.positions_from_free(z).detach(), False, "factorization")
        direction = torch.cholesky_solve((-gradient)[:, None], factor).squeeze(1)
        merit_directional_derivative = float(torch.dot(gradient, matrix @ direction))
        work.linear_solve_seconds += time.perf_counter() - linear_solve_start
        if not torch.isfinite(direction).all() or not math.isfinite(merit_directional_derivative):
            append_trace(
                *current_trace,
                merit_directional_derivative=merit_directional_derivative,
                hessian_minimum_eigenvalue=minimum_eigenvalue,
                hessian_maximum_eigenvalue=maximum_eigenvalue,
            )
            return finish(problem.positions_from_free(z).detach(), False, "nonfinite")
        if merit_directional_derivative >= 0.0:
            append_trace(
                *current_trace,
                merit_directional_derivative=merit_directional_derivative,
                hessian_minimum_eigenvalue=minimum_eigenvalue,
                hessian_maximum_eigenvalue=maximum_eigenvalue,
            )
            return finish(problem.positions_from_free(z).detach(), False, "non_descent")

        step_size = 1.0
        accepted = False
        candidate = z
        line_search_control_start = time.perf_counter()
        for _ in range(config.max_line_search_steps):
            trial = z + step_size * direction
            trial_value, trial_gradient = value_and_gradient(trial, line_search=True)
            work.line_search_trials += 1
            trial_gradient_norm = float(torch.linalg.vector_norm(trial_gradient))
            trial_merit = 0.5 * trial_gradient_norm * trial_gradient_norm
            merit_limit = residual_merit + config.armijo * step_size * merit_directional_derivative
            objective_roundoff_limit = (
                _RESIDUAL_POLISH_OBJECTIVE_ROUNDOFF_FACTOR
                * torch.finfo(torch.float64).eps
                * max(1.0, abs(float(value)), abs(float(trial_value)))
            )
            if (
                torch.isfinite(trial_value)
                and torch.isfinite(trial_gradient).all()
                and trial_gradient_norm < gradient_norm
                and trial_merit <= merit_limit
                and float(trial_value) <= float(value) + objective_roundoff_limit
            ):
                candidate = trial.detach()
                accepted = True
                break
            step_size *= config.backtrack
        work.line_search_seconds += time.perf_counter() - line_search_control_start

        if not accepted:
            append_trace(
                *current_trace,
                merit_directional_derivative=merit_directional_derivative,
                hessian_minimum_eigenvalue=minimum_eigenvalue,
                hessian_maximum_eigenvalue=maximum_eigenvalue,
            )
            return finish(problem.positions_from_free(z).detach(), False, "residual_line_search")

        step_norm = float(torch.linalg.vector_norm(candidate - z))
        append_trace(
            *current_trace,
            accepted_step_norm=step_norm,
            accepted_step_size=step_size,
            merit_directional_derivative=merit_directional_derivative,
            hessian_minimum_eigenvalue=minimum_eigenvalue,
            hessian_maximum_eigenvalue=maximum_eigenvalue,
        )
        z = candidate
        accepted_iterations += 1
