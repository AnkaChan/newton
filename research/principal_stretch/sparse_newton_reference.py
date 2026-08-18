# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exact sparse CPU Newton reference for scalable tetrahedral benchmarks.

The dense reference in :mod:`newton_baseline` deliberately uses PyTorch's
full autograd Hessian and dense eigendecomposition.  That is a useful oracle
for small scenes, but its quadratic storage prevents the 9,720-free-DOF fine
refinement scene from obtaining an independently converged reference.

This module assembles the same float64 Hessian from exact tetrahedron-local
blocks.  For stable Neo-Hookean elasticity,

``H_F = mu I + lambda (c c^T + (det(F) - alpha) H_det(F))``,

where ``c`` is the polynomial cofactor and ``H_det`` is the exact determinant
Hessian.  The inertial diagonal is added once, and Dirichlet degrees of
freedom are eliminated before sparse assembly.  SciPy is imported lazily and
is already available through Newton's ``importers`` and ``dev`` extras; it is
not made a new core dependency.

The sparse solve fails closed.  A smallest-algebraic Ritz pair proposes an
initial shift but is not treated as a global spectral bound.  Before any
linear solve, an unequilibrated symmetric-mode SuperLU factor must pass a
numerical LDL-style certificate: identical row/column permutations, unit
finite ``L`` diagonal, positive ``D = diag(U)`` with margin,
``U = diag(D) L.T``, and a full sparse factor residual.  A Gershgorin lower
bound is used only to seed rescue regularization after a factor or certificate
failure.  The independently recomputed linear residual, descent, and Armijo
acceptance remain mandatory.  Timing fields are kept separate from
deterministic records.  This is research reference infrastructure, not a
public Newton API or a performance solver.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import numbers
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .correction_gpu import MatrixFreeStableNHOperator
from .newton_baseline import NewtonConfig, NewtonProblem

SPARSE_NEWTON_CONTRACT = "exact-sparse-cpu-newton-float64-v2"
SPARSE_HESSIAN_CONTRACT = "exact-stable-nh-element-hessian-csr-v1"
SPARSE_LINEAR_SOLVER = "scipy-superlu-mmd-at-plus-a-symmetric-ldlt-certified-v2"
SPARSE_EIGEN_POLICY = "arpack-smallest-algebraic-ritz-heuristic-gershgorin-rescue-v2"
SPARSE_FACTOR_CERTIFICATE = "superlu-symmetric-ldlt-numerical-certificate-v2"
SPARSE_LINEAR_RESIDUAL_LIMIT = 5.0e-13
SPARSE_EIGEN_TOLERANCE = 1.0e-11
SPARSE_MAX_REFINEMENT_STEPS = 4
SPARSE_FACTOR_EQUILIBRATION = False
SPARSE_FACTOR_UNIT_DIAGONAL_LIMIT = 64.0 * np.finfo(np.float64).eps
SPARSE_FACTOR_PIVOT_RELATIVE_MARGIN = 512.0 * np.finfo(np.float64).eps
SPARSE_FACTOR_RELATION_RELATIVE_LIMIT = 5.0e-12
SPARSE_FACTORIZATION_RELATIVE_RESIDUAL_LIMIT = 5.0e-12

_SUPERLU_EQUILIBRATION_OPTION = "Equ" + "il"

_TERMINATION_REASONS = (
    "gradient",
    "max_iterations",
    "stalled",
    "hessian_eigensolve",
    "factorization",
    "factor_certificate",
    "linear_residual",
    "non_descent",
    "line_search",
)


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=canonical_dtype)


def _array_digest(value: np.ndarray) -> str:
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _readonly_float64(value: np.ndarray | torch.Tensor | Sequence[float], name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _load_scipy() -> tuple[Any, Any, str]:
    """Import the existing optional SciPy dependency only when requested."""
    try:
        import scipy  # noqa: PLC0415
        import scipy.sparse as sparse  # noqa: PLC0415
        import scipy.sparse.linalg as sparse_linalg  # noqa: PLC0415
    except ImportError as error:
        raise ModuleNotFoundError(
            "exact sparse Newton requires SciPy from Newton's existing 'dev' or 'importers' extra"
        ) from error
    return sparse, sparse_linalg, str(scipy.__version__)


def _epsilon_tensor() -> np.ndarray:
    epsilon = np.zeros((3, 3, 3), dtype=np.float64)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0
    return epsilon


_EPSILON = _epsilon_tensor()
_IDENTITY_3 = np.eye(3, dtype=np.float64)


@dataclasses.dataclass(frozen=True, eq=False)
class SparseExactHessian:
    """One exact free-DOF gradient and sparse Hessian assembly."""

    gradient: np.ndarray
    matrix: Any = dataclasses.field(repr=False, compare=False)
    objective: float
    minimum_determinant: float
    raw_triplet_count: int
    assembly_seconds: float

    def __post_init__(self) -> None:
        gradient = _readonly_float64(self.gradient, "gradient").reshape(-1)
        if self.matrix.shape != (gradient.size, gradient.size):
            raise ValueError("sparse Hessian shape does not match its gradient")
        if self.matrix.dtype != np.dtype(np.float64):
            raise ValueError("sparse Hessian must use float64 values")
        if not np.isfinite(self.matrix.data).all():
            raise ValueError("sparse Hessian values must be finite")
        if not math.isfinite(self.objective) or not math.isfinite(self.minimum_determinant):
            raise ValueError("sparse Hessian objective and determinant must be finite")
        if (
            isinstance(self.raw_triplet_count, bool)
            or not isinstance(self.raw_triplet_count, numbers.Integral)
            or self.raw_triplet_count < 0
        ):
            raise ValueError("raw_triplet_count must be a non-negative integer")
        if not math.isfinite(self.assembly_seconds) or self.assembly_seconds < 0.0:
            raise ValueError("assembly_seconds must be finite and non-negative")
        object.__setattr__(self, "gradient", gradient)

    @property
    def nnz(self) -> int:
        """Number of stored scalar Hessian entries after coalescing."""
        return int(self.matrix.nnz)


@dataclasses.dataclass(frozen=True)
class SparseNewtonIteration:
    """One evaluated iterate and its optional accepted outgoing direction."""

    iteration: int
    objective: float
    gradient_norm: float
    relative_residual: float
    minimum_determinant: float
    hessian_nnz: int
    minimum_eigenvalue: float | None = None
    eigenpair_residual: float | None = None
    diagonal_scale: float | None = None
    ritz_regularization: float | None = None
    gershgorin_lower_bound: float | None = None
    gershgorin_rescue_regularization: float | None = None
    gershgorin_rescue_used: bool = False
    regularization: float | None = None
    last_attempted_regularization: float | None = None
    factor_nnz: int | None = None
    factorization_attempts: int = 0
    factor_certificate_attempts: int = 0
    linear_solve_attempts: int = 0
    linear_refinement_steps: int = 0
    line_search_trials: int = 0
    factor_permutations_match: bool | None = None
    factor_l_unit_diagonal_error: float | None = None
    factor_minimum_diagonal: float | None = None
    factor_maximum_diagonal_magnitude: float | None = None
    factor_minimum_diagonal_relative: float | None = None
    factor_relation_relative_residual: float | None = None
    factorization_relative_residual: float | None = None
    factor_certificate_passed: bool | None = None
    linear_relative_residual: float | None = None
    directional_derivative: float | None = None
    accepted_step_norm: float | None = None
    accepted_step_size: float | None = None
    assembly_seconds: float = 0.0
    eigensolve_seconds: float = 0.0
    factorization_seconds: float = 0.0
    linear_solve_seconds: float = 0.0
    line_search_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "iteration",
            "hessian_nnz",
            "factorization_attempts",
            "factor_certificate_attempts",
            "linear_solve_attempts",
            "linear_refinement_steps",
            "line_search_trials",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.factor_nnz is not None and (
            isinstance(self.factor_nnz, bool)
            or not isinstance(self.factor_nnz, numbers.Integral)
            or self.factor_nnz < 0
        ):
            raise ValueError("factor_nnz must be a non-negative integer or None")
        required = (
            self.objective,
            self.gradient_norm,
            self.relative_residual,
            self.minimum_determinant,
            self.assembly_seconds,
            self.eigensolve_seconds,
            self.factorization_seconds,
            self.linear_solve_seconds,
            self.line_search_seconds,
            self.elapsed_seconds,
        )
        optional = (
            self.minimum_eigenvalue,
            self.eigenpair_residual,
            self.diagonal_scale,
            self.ritz_regularization,
            self.gershgorin_lower_bound,
            self.gershgorin_rescue_regularization,
            self.regularization,
            self.last_attempted_regularization,
            self.factor_l_unit_diagonal_error,
            self.factor_minimum_diagonal,
            self.factor_maximum_diagonal_magnitude,
            self.factor_minimum_diagonal_relative,
            self.factor_relation_relative_residual,
            self.factorization_relative_residual,
            self.linear_relative_residual,
            self.directional_derivative,
            self.accepted_step_norm,
            self.accepted_step_size,
        )
        if any(not math.isfinite(value) for value in required):
            raise ValueError("sparse Newton trace contains a non-finite required scalar")
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("sparse Newton trace contains a non-finite optional scalar")
        if self.gradient_norm < 0.0 or self.relative_residual < 0.0:
            raise ValueError("sparse Newton residual norms must be non-negative")
        if type(self.gershgorin_rescue_used) is not bool:
            raise ValueError("gershgorin_rescue_used must be an exact bool")
        for name in ("factor_permutations_match", "factor_certificate_passed"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be an exact bool or None")
        nonnegative_optional = (
            self.eigenpair_residual,
            self.diagonal_scale,
            self.ritz_regularization,
            self.gershgorin_rescue_regularization,
            self.regularization,
            self.last_attempted_regularization,
            self.factor_l_unit_diagonal_error,
            self.factor_maximum_diagonal_magnitude,
            self.factor_relation_relative_residual,
            self.factorization_relative_residual,
            self.linear_relative_residual,
        )
        if any(value is not None and value < 0.0 for value in nonnegative_optional):
            raise ValueError("sparse Newton certificate magnitudes must be non-negative")
        if self.factor_certificate_attempts > self.factorization_attempts:
            raise ValueError("factor certificate attempts cannot exceed factorization attempts")
        if self.factorization_attempts == 0:
            if self.regularization is not None or self.last_attempted_regularization is not None:
                raise ValueError("unattempted sparse factorization cannot record regularization")
        elif self.regularization != self.last_attempted_regularization:
            raise ValueError("regularization must equal the last attempted regularization")
        if self.gershgorin_rescue_used:
            if self.gershgorin_lower_bound is None or self.gershgorin_rescue_regularization is None:
                raise ValueError("Gershgorin rescue requires its lower bound and seed")
        elif self.gershgorin_lower_bound is not None or self.gershgorin_rescue_regularization is not None:
            raise ValueError("unused Gershgorin rescue cannot retain rescue evidence")
        if self.factor_certificate_passed is True:
            certificate_scalars = (
                self.factor_l_unit_diagonal_error,
                self.factor_minimum_diagonal,
                self.factor_maximum_diagonal_magnitude,
                self.factor_minimum_diagonal_relative,
                self.factor_relation_relative_residual,
                self.factorization_relative_residual,
            )
            if (
                self.factor_certificate_attempts < 1
                or self.factor_permutations_match is not True
                or any(value is None for value in certificate_scalars)
            ):
                raise ValueError("passed factor certificate lacks complete numerical evidence")
        if any(
            value < 0.0
            for value in (
                self.assembly_seconds,
                self.eigensolve_seconds,
                self.factorization_seconds,
                self.linear_solve_seconds,
                self.line_search_seconds,
                self.elapsed_seconds,
            )
        ):
            raise ValueError("sparse Newton timings must be non-negative")
        has_step_norm = self.accepted_step_norm is not None
        has_step_size = self.accepted_step_size is not None
        if has_step_norm != has_step_size:
            raise ValueError("accepted sparse Newton step size and norm must appear together")
        if has_step_size and (self.accepted_step_norm < 0.0 or not 0.0 < self.accepted_step_size <= 1.0):
            raise ValueError("accepted sparse Newton step must have a valid size and norm")
        if self.directional_derivative is not None and self.directional_derivative >= 0.0:
            raise ValueError("sparse Newton direction must be descending")
        if has_step_size and (self.directional_derivative is None or self.line_search_trials < 1):
            raise ValueError("accepted sparse Newton step requires line-search evidence")
        if has_step_size and self.factor_certificate_passed is not True:
            raise ValueError("accepted sparse Newton step requires a passed factor certificate")
        if self.line_search_trials and self.directional_derivative is None:
            raise ValueError("sparse Newton line-search trials require a direction")
        if self.linear_refinement_steps > SPARSE_MAX_REFINEMENT_STEPS:
            raise ValueError("sparse Newton refinement count exceeds its fixed limit")

    @property
    def has_accepted_outgoing_step(self) -> bool:
        """Whether this iterate records one accepted outgoing step."""
        return self.accepted_step_size is not None

    def deterministic_record(self) -> dict[str, object]:
        """Return timing-free numerical and work evidence."""
        timing_names = {
            "assembly_seconds",
            "eigensolve_seconds",
            "factorization_seconds",
            "linear_solve_seconds",
            "line_search_seconds",
            "elapsed_seconds",
        }
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in timing_names
        }

    def timing_record(self) -> dict[str, float]:
        """Return timing-only measurements for this iterate."""
        return {
            "assembly_seconds": self.assembly_seconds,
            "eigensolve_seconds": self.eigensolve_seconds,
            "factorization_seconds": self.factorization_seconds,
            "linear_solve_seconds": self.linear_solve_seconds,
            "line_search_seconds": self.line_search_seconds,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class SparseNewtonResult:
    """One fail-closed exact sparse Newton solve."""

    positions: np.ndarray
    converged: bool
    reason: str
    accepted_iterations: int
    residual_scale: float
    scipy_version: str
    total_seconds: float
    objective_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    eigenvalue_evaluations: int
    factorization_attempts: int
    factor_certificate_attempts: int
    linear_solve_attempts: int
    line_search_trials: int
    trace: tuple[SparseNewtonIteration, ...]

    def __post_init__(self) -> None:
        positions = _readonly_float64(self.positions, "positions")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("sparse Newton positions must have shape (V, 3)")
        if self.reason not in _TERMINATION_REASONS:
            raise ValueError(f"unknown sparse Newton termination reason: {self.reason}")
        if self.converged != (self.reason == "gradient"):
            raise ValueError("sparse Newton convergence must agree with the gradient reason")
        for name in (
            "accepted_iterations",
            "objective_evaluations",
            "gradient_evaluations",
            "hessian_evaluations",
            "eigenvalue_evaluations",
            "factorization_attempts",
            "factor_certificate_attempts",
            "linear_solve_attempts",
            "line_search_trials",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        if type(self.scipy_version) is not str or not self.scipy_version:
            raise ValueError("scipy_version must be a non-empty exact string")
        if not math.isfinite(self.total_seconds) or self.total_seconds < 0.0:
            raise ValueError("total_seconds must be finite and non-negative")
        trace = tuple(self.trace)
        if len(trace) != self.accepted_iterations + 1:
            raise ValueError("sparse Newton trace must contain one terminal iterate")
        if tuple(item.iteration for item in trace) != tuple(range(len(trace))):
            raise ValueError("sparse Newton trace iteration indices must be contiguous")
        if sum(item.has_accepted_outgoing_step for item in trace) != self.accepted_iterations:
            raise ValueError("sparse Newton accepted-step count does not match its trace")
        if trace[-1].has_accepted_outgoing_step:
            raise ValueError("sparse Newton terminal trace item cannot have an outgoing step")
        if any(after.elapsed_seconds < before.elapsed_seconds for before, after in itertools.pairwise(trace)):
            raise ValueError("sparse Newton trace timings must be monotone")
        if self.objective_evaluations < len(trace) or self.gradient_evaluations != len(trace):
            raise ValueError("sparse Newton objective/gradient work accounting is invalid")
        if self.hessian_evaluations != len(trace):
            raise ValueError("sparse Newton Hessian assembly count must equal evaluated iterates")
        if self.objective_evaluations != len(trace) + self.line_search_trials:
            raise ValueError("sparse Newton objective work does not match line-search trials")
        if self.eigenvalue_evaluations != sum(item.minimum_eigenvalue is not None for item in trace):
            raise ValueError("sparse Newton eigenvalue work accounting is invalid")
        if self.factorization_attempts != sum(item.factorization_attempts for item in trace):
            raise ValueError("sparse Newton factorization work accounting is invalid")
        if self.factor_certificate_attempts != sum(item.factor_certificate_attempts for item in trace):
            raise ValueError("sparse Newton factor-certificate work accounting is invalid")
        if self.linear_solve_attempts != sum(item.linear_solve_attempts for item in trace):
            raise ValueError("sparse Newton linear-solve work accounting is invalid")
        if self.line_search_trials != sum(item.line_search_trials for item in trace):
            raise ValueError("sparse Newton line-search work accounting is invalid")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "trace", trace)

    @property
    def final_objective(self) -> float:
        """Final common objective."""
        return self.trace[-1].objective

    @property
    def final_gradient_norm(self) -> float:
        """Final free-gradient norm [N]."""
        return self.trace[-1].gradient_norm

    @property
    def final_relative_residual(self) -> float:
        """Final free-gradient norm divided by the shared residual scale."""
        return self.trace[-1].relative_residual

    def deterministic_record(self) -> dict[str, object]:
        """Return timing-free solve evidence."""
        return {
            "contract": SPARSE_NEWTON_CONTRACT,
            "hessian_contract": SPARSE_HESSIAN_CONTRACT,
            "linear_solver": SPARSE_LINEAR_SOLVER,
            "eigen_policy": SPARSE_EIGEN_POLICY,
            "factor_certificate": SPARSE_FACTOR_CERTIFICATE,
            "factor_equilibration": SPARSE_FACTOR_EQUILIBRATION,
            "factor_unit_diagonal_limit": SPARSE_FACTOR_UNIT_DIAGONAL_LIMIT,
            "factor_pivot_relative_margin": SPARSE_FACTOR_PIVOT_RELATIVE_MARGIN,
            "factor_relation_relative_limit": SPARSE_FACTOR_RELATION_RELATIVE_LIMIT,
            "factorization_relative_residual_limit": SPARSE_FACTORIZATION_RELATIVE_RESIDUAL_LIMIT,
            "linear_residual_limit": SPARSE_LINEAR_RESIDUAL_LIMIT,
            "maximum_refinement_steps": SPARSE_MAX_REFINEMENT_STEPS,
            "positions_sha256": _array_digest(self.positions),
            "converged": self.converged,
            "reason": self.reason,
            "accepted_iterations": self.accepted_iterations,
            "residual_scale": self.residual_scale,
            "scipy_version": self.scipy_version,
            "final_objective": self.final_objective,
            "final_gradient_norm": self.final_gradient_norm,
            "final_relative_residual": self.final_relative_residual,
            "work": {
                "objective_evaluations": self.objective_evaluations,
                "gradient_evaluations": self.gradient_evaluations,
                "hessian_evaluations": self.hessian_evaluations,
                "eigenvalue_evaluations": self.eigenvalue_evaluations,
                "factorization_attempts": self.factorization_attempts,
                "factor_certificate_attempts": self.factor_certificate_attempts,
                "linear_solve_attempts": self.linear_solve_attempts,
                "line_search_trials": self.line_search_trials,
            },
            "trace": [item.deterministic_record() for item in self.trace],
        }

    def timing_record(self) -> dict[str, object]:
        """Return timing-only solve measurements."""
        return {
            "total_seconds": self.total_seconds,
            "trace": [item.timing_record() for item in self.trace],
        }


def assemble_sparse_exact_hessian(
    problem: NewtonProblem,
    positions: np.ndarray | torch.Tensor,
) -> SparseExactHessian:
    """Assemble the exact free-DOF stable-NH Hessian in SciPy CSR form.

    Args:
        problem: Validated CPU float64 common-objective problem.
        positions: Full positions [m], shape ``[vertex_count, 3]``. Pinned
            entries must equal the exact Dirichlet targets.

    Returns:
        Exact gradient, objective, determinant, and coalesced CSR Hessian.
    """
    if type(problem) is not NewtonProblem:
        raise TypeError("problem must be an exact NewtonProblem")
    sparse, _, _ = _load_scipy()
    start = time.perf_counter()
    operator = MatrixFreeStableNHOperator.from_problem(problem, positions)
    gradient = np.array(operator.gradient_free(), dtype=np.float64, order="C", copy=True)
    deformation = operator.deformation_gradients
    cofactor = operator.cofactors
    determinant = operator.determinants
    alpha = 1.0 + operator.mu / np.maximum(operator.lam, 1.0e-6)
    determinant_hessian = np.einsum(
        "dem,cln,tmn->tdcel",
        _EPSILON,
        _EPSILON,
        deformation,
        optimize=True,
    )
    cofactor_shape = np.einsum(
        "tdc,tac->tad",
        cofactor,
        operator.shape_gradients,
        optimize=False,
    )

    free_map = np.full(operator.n_vertices, -1, dtype=np.int64)
    free_map[operator.free] = np.arange(operator.free.size, dtype=np.int64)
    scalar = np.arange(3, dtype=np.int64)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for corner_a in range(4):
        shape_a = operator.shape_gradients[:, corner_a]
        free_a = free_map[operator.tets[:, corner_a]]
        for corner_b in range(4):
            shape_b = operator.shape_gradients[:, corner_b]
            free_b = free_map[operator.tets[:, corner_b]]
            keep = (free_a >= 0) & (free_b >= 0)
            count = int(np.count_nonzero(keep))
            if count == 0:
                continue
            determinant_block = np.einsum(
                "tc,tdcel,tl->tde",
                shape_a,
                determinant_hessian,
                shape_b,
                optimize=True,
            )
            material_block = (operator.mu * np.sum(shape_a * shape_b, axis=1))[:, None, None] * _IDENTITY_3[
                None
            ] + operator.lam[:, None, None] * (
                cofactor_shape[:, corner_a, :, None] * cofactor_shape[:, corner_b, None, :]
                + (determinant - alpha)[:, None, None] * determinant_block
            )
            local_block = operator.volumes[:, None, None] * material_block
            row = 3 * free_a[keep, None, None] + scalar[None, :, None]
            column = 3 * free_b[keep, None, None] + scalar[None, None, :]
            rows.append(np.broadcast_to(row, (count, 3, 3)).reshape(-1))
            columns.append(np.broadcast_to(column, (count, 3, 3)).reshape(-1))
            values.append(local_block[keep].reshape(-1))

    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    column = np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)
    value = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    raw_triplet_count = int(value.size)
    size = operator.n_free_dofs
    matrix = sparse.coo_matrix((value, (row, column)), shape=(size, size), dtype=np.float64).tocsr()
    inertia = np.repeat(operator.mass[operator.free] / (operator.dt * operator.dt), 3)
    matrix += sparse.diags(inertia, format="csr")
    matrix = (0.5 * (matrix + matrix.T)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return SparseExactHessian(
        gradient=gradient,
        matrix=matrix,
        objective=operator.objective(),
        minimum_determinant=operator.minimum_determinant,
        raw_triplet_count=raw_triplet_count,
        assembly_seconds=time.perf_counter() - start,
    )


@dataclasses.dataclass
class _Work:
    objective_evaluations: int = 0
    gradient_evaluations: int = 0
    hessian_evaluations: int = 0
    eigenvalue_evaluations: int = 0
    factorization_attempts: int = 0
    factor_certificate_attempts: int = 0
    linear_solve_attempts: int = 0
    line_search_trials: int = 0


@dataclasses.dataclass(frozen=True)
class _FactorCertificate:
    permutations_match: bool
    l_unit_diagonal_error: float | None
    minimum_diagonal: float | None
    maximum_diagonal_magnitude: float | None
    minimum_diagonal_relative: float | None
    relation_relative_residual: float | None
    factorization_relative_residual: float | None
    passed: bool


def _finite_sparse_frobenius_norm(matrix: Any) -> float | None:
    data = np.asarray(matrix.data, dtype=np.float64)
    if not np.isfinite(data).all():
        return None
    value = float(np.linalg.norm(data))
    return value if math.isfinite(value) else None


def _certify_symmetric_factor(
    shifted: Any,
    factor: Any,
    sparse: Any,
    diagonal_scale: float,
) -> _FactorCertificate:
    """Numerically certify one SuperLU factor as an SPD LDL-style factor."""
    failed = _FactorCertificate(False, None, None, None, None, None, None, False)
    size = shifted.shape[0]
    try:
        lower = factor.L.tocsc()
        upper = factor.U.tocsc()
        row_permutation = np.asarray(factor.perm_r)
        column_permutation = np.asarray(factor.perm_c)
    except (AttributeError, TypeError, ValueError):
        return failed
    if lower.shape != (size, size) or upper.shape != (size, size):
        return failed
    expected_permutation = np.arange(size, dtype=row_permutation.dtype)
    permutations_match = bool(
        row_permutation.shape == (size,)
        and column_permutation.shape == (size,)
        and np.array_equal(np.sort(row_permutation), expected_permutation)
        and np.array_equal(np.sort(column_permutation), expected_permutation)
        and np.array_equal(row_permutation, column_permutation)
    )

    lower_diagonal = np.asarray(lower.diagonal(), dtype=np.float64)
    upper_diagonal = np.asarray(upper.diagonal(), dtype=np.float64)
    factors_finite = bool(np.isfinite(lower.data).all() and np.isfinite(upper.data).all())
    l_unit_diagonal_error = None
    minimum_diagonal = None
    maximum_diagonal_magnitude = None
    minimum_diagonal_relative = None
    relation_relative_residual = None
    factorization_relative_residual = None
    if lower_diagonal.shape == (size,) and np.isfinite(lower_diagonal).all():
        l_unit_diagonal_error = float(np.max(np.abs(lower_diagonal - 1.0)))
    if upper_diagonal.shape == (size,) and np.isfinite(upper_diagonal).all():
        minimum_diagonal = float(np.min(upper_diagonal))
        maximum_diagonal_magnitude = float(np.max(np.abs(upper_diagonal)))
        pivot_scale = max(diagonal_scale, maximum_diagonal_magnitude, 1.0)
        minimum_diagonal_relative = minimum_diagonal / pivot_scale
    if factors_finite and upper_diagonal.shape == (size,) and np.isfinite(upper_diagonal).all():
        relation = upper - sparse.diags(upper_diagonal, format="csc") @ lower.T
        relation_norm = _finite_sparse_frobenius_norm(relation)
        upper_norm = _finite_sparse_frobenius_norm(upper)
        if relation_norm is not None and upper_norm is not None:
            relation_relative_residual = relation_norm / max(upper_norm, np.finfo(np.float64).tiny)
    if factors_finite and permutations_match:
        inverse_permutation = np.argsort(row_permutation)
        permuted = shifted[inverse_permutation, :][:, inverse_permutation]
        factorization_residual = permuted - lower @ upper
        residual_norm = _finite_sparse_frobenius_norm(factorization_residual)
        shifted_norm = _finite_sparse_frobenius_norm(permuted)
        if residual_norm is not None and shifted_norm is not None:
            factorization_relative_residual = residual_norm / max(
                shifted_norm,
                np.finfo(np.float64).tiny,
            )

    passed = bool(
        permutations_match
        and l_unit_diagonal_error is not None
        and l_unit_diagonal_error <= SPARSE_FACTOR_UNIT_DIAGONAL_LIMIT
        and minimum_diagonal_relative is not None
        and minimum_diagonal_relative > SPARSE_FACTOR_PIVOT_RELATIVE_MARGIN
        and relation_relative_residual is not None
        and relation_relative_residual <= SPARSE_FACTOR_RELATION_RELATIVE_LIMIT
        and factorization_relative_residual is not None
        and factorization_relative_residual <= SPARSE_FACTORIZATION_RELATIVE_RESIDUAL_LIMIT
    )
    return _FactorCertificate(
        permutations_match=permutations_match,
        l_unit_diagonal_error=l_unit_diagonal_error,
        minimum_diagonal=minimum_diagonal,
        maximum_diagonal_magnitude=maximum_diagonal_magnitude,
        minimum_diagonal_relative=minimum_diagonal_relative,
        relation_relative_residual=relation_relative_residual,
        factorization_relative_residual=factorization_relative_residual,
        passed=passed,
    )


def _gershgorin_lower_bound(matrix: Any) -> float:
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    absolute_row_sum = np.asarray(abs(matrix).sum(axis=1), dtype=np.float64).reshape(-1)
    lower_bound = float(np.min(diagonal - (absolute_row_sum - np.abs(diagonal))))
    if not math.isfinite(lower_bound):
        raise ValueError("sparse Hessian has a non-finite Gershgorin lower bound")
    return lower_bound


@dataclasses.dataclass(frozen=True)
class _Direction:
    value: np.ndarray | None = None
    reason: str | None = None
    minimum_eigenvalue: float | None = None
    eigenpair_residual: float | None = None
    diagonal_scale: float | None = None
    ritz_regularization: float | None = None
    gershgorin_lower_bound: float | None = None
    gershgorin_rescue_regularization: float | None = None
    gershgorin_rescue_used: bool = False
    regularization: float | None = None
    last_attempted_regularization: float | None = None
    factor_nnz: int | None = None
    factorization_attempts: int = 0
    factor_certificate_attempts: int = 0
    linear_solve_attempts: int = 0
    refinement_steps: int = 0
    factor_permutations_match: bool | None = None
    factor_l_unit_diagonal_error: float | None = None
    factor_minimum_diagonal: float | None = None
    factor_maximum_diagonal_magnitude: float | None = None
    factor_minimum_diagonal_relative: float | None = None
    factor_relation_relative_residual: float | None = None
    factorization_relative_residual: float | None = None
    factor_certificate_passed: bool | None = None
    linear_relative_residual: float | None = None
    eigensolve_seconds: float = 0.0
    factorization_seconds: float = 0.0
    linear_solve_seconds: float = 0.0


def _direction(
    matrix: Any,
    gradient: np.ndarray,
    config: NewtonConfig,
    sparse: Any,
    sparse_linalg: Any,
    work: _Work,
) -> _Direction:
    eigensolve_start = time.perf_counter()
    try:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            matrix,
            k=1,
            which="SA",
            tol=SPARSE_EIGEN_TOLERANCE,
            maxiter=max(1_000, 5 * matrix.shape[0]),
            v0=np.ones(matrix.shape[0], dtype=np.float64),
        )
    except (sparse_linalg.ArpackError, RuntimeError, ValueError):
        return _Direction(
            reason="hessian_eigensolve",
            eigensolve_seconds=time.perf_counter() - eigensolve_start,
        )
    eigensolve_seconds = time.perf_counter() - eigensolve_start
    work.eigenvalue_evaluations += 1
    minimum_eigenvalue = float(eigenvalues[0])
    eigenvector = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    eigenpair_residual = float(np.linalg.norm(matrix @ eigenvector - minimum_eigenvalue * eigenvector))
    diagonal_scale = max(float(np.max(np.abs(matrix.diagonal()))), 1.0)
    if not np.isfinite((minimum_eigenvalue, eigenpair_residual, diagonal_scale)).all():
        return _Direction(
            reason="hessian_eigensolve",
            eigensolve_seconds=eigensolve_seconds,
        )
    minimum_target = config.minimum_eigenvalue_relative * diagonal_scale
    ritz_regularization = max(0.0, minimum_target - minimum_eigenvalue)
    regularization = ritz_regularization
    identity = sparse.eye(matrix.shape[0], dtype=np.float64, format="csc")
    factorization_seconds = 0.0
    linear_solve_seconds = 0.0
    attempts = 0
    certificate_attempts = 0
    last_reason = "factorization"
    last_factor_nnz = None
    last_relative_residual = None
    last_refinement_steps = 0
    linear_solve_attempts = 0
    last_attempted_regularization = None
    gershgorin_lower_bound = None
    gershgorin_rescue_regularization = None
    gershgorin_rescue_used = False
    last_certificate = None

    def finish(value: np.ndarray | None, reason: str | None) -> _Direction:
        certificate = last_certificate
        return _Direction(
            value=value,
            reason=reason,
            minimum_eigenvalue=minimum_eigenvalue,
            eigenpair_residual=eigenpair_residual,
            diagonal_scale=diagonal_scale,
            ritz_regularization=ritz_regularization,
            gershgorin_lower_bound=gershgorin_lower_bound,
            gershgorin_rescue_regularization=gershgorin_rescue_regularization,
            gershgorin_rescue_used=gershgorin_rescue_used,
            regularization=last_attempted_regularization,
            last_attempted_regularization=last_attempted_regularization,
            factor_nnz=last_factor_nnz,
            factorization_attempts=attempts,
            factor_certificate_attempts=certificate_attempts,
            linear_solve_attempts=linear_solve_attempts,
            refinement_steps=last_refinement_steps,
            factor_permutations_match=None if certificate is None else certificate.permutations_match,
            factor_l_unit_diagonal_error=None if certificate is None else certificate.l_unit_diagonal_error,
            factor_minimum_diagonal=None if certificate is None else certificate.minimum_diagonal,
            factor_maximum_diagonal_magnitude=(None if certificate is None else certificate.maximum_diagonal_magnitude),
            factor_minimum_diagonal_relative=None if certificate is None else certificate.minimum_diagonal_relative,
            factor_relation_relative_residual=(None if certificate is None else certificate.relation_relative_residual),
            factorization_relative_residual=(
                None if certificate is None else certificate.factorization_relative_residual
            ),
            factor_certificate_passed=None if certificate is None else certificate.passed,
            linear_relative_residual=last_relative_residual,
            eigensolve_seconds=eigensolve_seconds,
            factorization_seconds=factorization_seconds,
            linear_solve_seconds=linear_solve_seconds,
        )

    def grow_after_factor_failure() -> None:
        nonlocal gershgorin_lower_bound
        nonlocal gershgorin_rescue_regularization
        nonlocal gershgorin_rescue_used
        nonlocal regularization
        if not gershgorin_rescue_used:
            gershgorin_lower_bound = _gershgorin_lower_bound(matrix)
            gershgorin_rescue_regularization = max(0.0, minimum_target - gershgorin_lower_bound)
            gershgorin_rescue_used = True
        assert last_attempted_regularization is not None
        assert gershgorin_rescue_regularization is not None
        regularization = max(
            minimum_target,
            last_attempted_regularization * config.regularization_growth,
            gershgorin_rescue_regularization,
        )

    for _ in range(config.max_regularization_attempts):
        attempts += 1
        work.factorization_attempts += 1
        last_attempted_regularization = regularization
        shifted = matrix.tocsc() + regularization * identity
        factorization_start = time.perf_counter()
        try:
            factor = sparse_linalg.splu(
                shifted,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={
                    "SymmetricMode": True,
                    _SUPERLU_EQUILIBRATION_OPTION: SPARSE_FACTOR_EQUILIBRATION,
                },
            )
        except RuntimeError:
            factorization_seconds += time.perf_counter() - factorization_start
            last_reason = "factorization"
            grow_after_factor_failure()
            continue
        last_factor_nnz = int(factor.L.nnz + factor.U.nnz)
        last_certificate = _certify_symmetric_factor(shifted, factor, sparse, diagonal_scale)
        certificate_attempts += 1
        work.factor_certificate_attempts += 1
        factorization_seconds += time.perf_counter() - factorization_start
        if not last_certificate.passed:
            last_reason = "factor_certificate"
            grow_after_factor_failure()
            continue
        work.linear_solve_attempts += 1
        linear_solve_attempts += 1
        linear_start = time.perf_counter()
        direction = np.asarray(factor.solve(-gradient), dtype=np.float64)
        rhs_norm = max(float(np.linalg.norm(gradient)), np.finfo(np.float64).tiny)
        refinement_steps = 0
        for _ in range(SPARSE_MAX_REFINEMENT_STEPS):
            residual = -gradient - shifted @ direction
            relative_residual = float(np.linalg.norm(residual)) / rhs_norm
            if relative_residual <= SPARSE_LINEAR_RESIDUAL_LIMIT:
                break
            direction += factor.solve(residual)
            refinement_steps += 1
        residual = -gradient - shifted @ direction
        relative_residual = float(np.linalg.norm(residual)) / rhs_norm
        linear_solve_seconds += time.perf_counter() - linear_start
        last_relative_residual = relative_residual
        last_refinement_steps = refinement_steps
        if not np.isfinite(direction).all() or not math.isfinite(relative_residual):
            last_relative_residual = None
            return finish(None, "linear_residual")
        if relative_residual > SPARSE_LINEAR_RESIDUAL_LIMIT:
            return finish(None, "linear_residual")
        if float(np.dot(gradient, direction)) < 0.0:
            return finish(direction, None)
        last_reason = "non_descent"
        regularization = max(
            minimum_target,
            last_attempted_regularization * config.regularization_growth,
        )

    return finish(None, last_reason)


def _positions_from_free(problem: NewtonProblem, free_positions: np.ndarray) -> np.ndarray:
    positions = np.empty((problem.n_vertices, 3), dtype=np.float64)
    positions[problem.free.detach().numpy()] = free_positions.reshape(-1, 3)
    positions[problem.pinned.detach().numpy()] = problem.pin_targets.detach().numpy()
    return positions


def _initial_positions(
    problem: NewtonProblem,
    positions: np.ndarray | torch.Tensor | None,
) -> np.ndarray:
    if positions is None:
        initial = problem.inertial_target.detach().numpy().copy()
    else:
        initial = _readonly_float64(positions, "x_initial").copy()
    expected = (problem.n_vertices, 3)
    if initial.shape != expected:
        raise ValueError(f"x_initial must have shape {expected}, got {initial.shape}")
    initial[problem.pinned.detach().numpy()] = problem.pin_targets.detach().numpy()
    return initial


def solve_sparse_newton(
    problem: NewtonProblem,
    x_initial: np.ndarray | torch.Tensor | None = None,
    config: NewtonConfig | None = None,
) -> SparseNewtonResult:
    """Solve one common-objective step with exact sparse CPU Newton.

    Args:
        problem: Validated CPU float64 problem.
        x_initial: Initial full positions [m]. Defaults to the force-shifted
            inertial target with exact Dirichlet targets.
        config: Dense-Newton-compatible convergence and globalization policy.

    Returns:
        Fail-closed sparse result with deterministic numerical evidence and
        separately serializable timings.
    """
    if type(problem) is not NewtonProblem:
        raise TypeError("problem must be an exact NewtonProblem")
    cfg = NewtonConfig() if config is None else config
    if type(cfg) is not NewtonConfig:
        raise TypeError("config must be an exact NewtonConfig")
    cfg.validate()
    sparse, sparse_linalg, scipy_version = _load_scipy()
    start = time.perf_counter()
    initial = _initial_positions(problem, x_initial)
    free = problem.free.detach().numpy()
    z = initial[free].reshape(-1).copy()
    trace: list[SparseNewtonIteration] = []
    work = _Work()
    accepted_iterations = 0
    stalled_after_update = False

    def finish(positions: np.ndarray, converged: bool, reason: str) -> SparseNewtonResult:
        return SparseNewtonResult(
            positions=positions,
            converged=converged,
            reason=reason,
            accepted_iterations=accepted_iterations,
            residual_scale=problem.residual_scale,
            scipy_version=scipy_version,
            total_seconds=time.perf_counter() - start,
            objective_evaluations=work.objective_evaluations,
            gradient_evaluations=work.gradient_evaluations,
            hessian_evaluations=work.hessian_evaluations,
            eigenvalue_evaluations=work.eigenvalue_evaluations,
            factorization_attempts=work.factorization_attempts,
            factor_certificate_attempts=work.factor_certificate_attempts,
            linear_solve_attempts=work.linear_solve_attempts,
            line_search_trials=work.line_search_trials,
            trace=tuple(trace),
        )

    while True:
        positions = _positions_from_free(problem, z)
        assembly = assemble_sparse_exact_hessian(problem, positions)
        work.objective_evaluations += 1
        work.gradient_evaluations += 1
        work.hessian_evaluations += 1
        gradient_norm = float(np.linalg.norm(assembly.gradient))
        relative_residual = gradient_norm / problem.residual_scale
        base = {
            "iteration": accepted_iterations,
            "objective": assembly.objective,
            "gradient_norm": gradient_norm,
            "relative_residual": relative_residual,
            "minimum_determinant": assembly.minimum_determinant,
            "hessian_nnz": assembly.nnz,
            "assembly_seconds": assembly.assembly_seconds,
        }
        converged = (
            gradient_norm <= cfg.gradient_absolute_tolerance or relative_residual <= cfg.gradient_relative_tolerance
        )
        if converged:
            trace.append(SparseNewtonIteration(**base, elapsed_seconds=time.perf_counter() - start))
            return finish(positions, True, "gradient")
        if stalled_after_update:
            trace.append(SparseNewtonIteration(**base, elapsed_seconds=time.perf_counter() - start))
            return finish(positions, False, "stalled")
        if accepted_iterations >= cfg.max_iterations:
            trace.append(SparseNewtonIteration(**base, elapsed_seconds=time.perf_counter() - start))
            return finish(positions, False, "max_iterations")

        direction = _direction(assembly.matrix, assembly.gradient, cfg, sparse, sparse_linalg, work)
        if direction.value is None:
            trace.append(
                SparseNewtonIteration(
                    **base,
                    minimum_eigenvalue=direction.minimum_eigenvalue,
                    eigenpair_residual=direction.eigenpair_residual,
                    diagonal_scale=direction.diagonal_scale,
                    ritz_regularization=direction.ritz_regularization,
                    gershgorin_lower_bound=direction.gershgorin_lower_bound,
                    gershgorin_rescue_regularization=direction.gershgorin_rescue_regularization,
                    gershgorin_rescue_used=direction.gershgorin_rescue_used,
                    regularization=direction.regularization,
                    last_attempted_regularization=direction.last_attempted_regularization,
                    factor_nnz=direction.factor_nnz,
                    factorization_attempts=direction.factorization_attempts,
                    factor_certificate_attempts=direction.factor_certificate_attempts,
                    linear_solve_attempts=direction.linear_solve_attempts,
                    linear_refinement_steps=direction.refinement_steps,
                    factor_permutations_match=direction.factor_permutations_match,
                    factor_l_unit_diagonal_error=direction.factor_l_unit_diagonal_error,
                    factor_minimum_diagonal=direction.factor_minimum_diagonal,
                    factor_maximum_diagonal_magnitude=direction.factor_maximum_diagonal_magnitude,
                    factor_minimum_diagonal_relative=direction.factor_minimum_diagonal_relative,
                    factor_relation_relative_residual=direction.factor_relation_relative_residual,
                    factorization_relative_residual=direction.factorization_relative_residual,
                    factor_certificate_passed=direction.factor_certificate_passed,
                    linear_relative_residual=direction.linear_relative_residual,
                    eigensolve_seconds=direction.eigensolve_seconds,
                    factorization_seconds=direction.factorization_seconds,
                    linear_solve_seconds=direction.linear_solve_seconds,
                    elapsed_seconds=time.perf_counter() - start,
                )
            )
            return finish(positions, False, direction.reason or "factorization")

        directional_derivative = float(np.dot(assembly.gradient, direction.value))
        step_size = 1.0
        candidate = z
        line_search_start = time.perf_counter()
        accepted = False
        iteration_line_search_trials = 0
        for _ in range(cfg.max_line_search_steps):
            trial = z + step_size * direction.value
            trial_positions = _positions_from_free(problem, trial)
            trial_objective = MatrixFreeStableNHOperator.from_problem(problem, trial_positions).objective()
            work.objective_evaluations += 1
            work.line_search_trials += 1
            iteration_line_search_trials += 1
            sufficient_decrease = assembly.objective + cfg.armijo * step_size * directional_derivative
            if math.isfinite(trial_objective) and trial_objective <= sufficient_decrease:
                candidate = trial
                accepted = True
                break
            step_size *= cfg.backtrack
        line_search_seconds = time.perf_counter() - line_search_start
        if not accepted:
            trace.append(
                SparseNewtonIteration(
                    **base,
                    minimum_eigenvalue=direction.minimum_eigenvalue,
                    eigenpair_residual=direction.eigenpair_residual,
                    diagonal_scale=direction.diagonal_scale,
                    ritz_regularization=direction.ritz_regularization,
                    gershgorin_lower_bound=direction.gershgorin_lower_bound,
                    gershgorin_rescue_regularization=direction.gershgorin_rescue_regularization,
                    gershgorin_rescue_used=direction.gershgorin_rescue_used,
                    regularization=direction.regularization,
                    last_attempted_regularization=direction.last_attempted_regularization,
                    factor_nnz=direction.factor_nnz,
                    factorization_attempts=direction.factorization_attempts,
                    factor_certificate_attempts=direction.factor_certificate_attempts,
                    linear_solve_attempts=direction.linear_solve_attempts,
                    linear_refinement_steps=direction.refinement_steps,
                    factor_permutations_match=direction.factor_permutations_match,
                    factor_l_unit_diagonal_error=direction.factor_l_unit_diagonal_error,
                    factor_minimum_diagonal=direction.factor_minimum_diagonal,
                    factor_maximum_diagonal_magnitude=direction.factor_maximum_diagonal_magnitude,
                    factor_minimum_diagonal_relative=direction.factor_minimum_diagonal_relative,
                    factor_relation_relative_residual=direction.factor_relation_relative_residual,
                    factorization_relative_residual=direction.factorization_relative_residual,
                    factor_certificate_passed=direction.factor_certificate_passed,
                    line_search_trials=iteration_line_search_trials,
                    linear_relative_residual=direction.linear_relative_residual,
                    directional_derivative=directional_derivative,
                    eigensolve_seconds=direction.eigensolve_seconds,
                    factorization_seconds=direction.factorization_seconds,
                    linear_solve_seconds=direction.linear_solve_seconds,
                    line_search_seconds=line_search_seconds,
                    elapsed_seconds=time.perf_counter() - start,
                )
            )
            return finish(positions, False, "line_search")

        step_norm = float(np.linalg.norm(candidate - z))
        trace.append(
            SparseNewtonIteration(
                **base,
                minimum_eigenvalue=direction.minimum_eigenvalue,
                eigenpair_residual=direction.eigenpair_residual,
                diagonal_scale=direction.diagonal_scale,
                ritz_regularization=direction.ritz_regularization,
                gershgorin_lower_bound=direction.gershgorin_lower_bound,
                gershgorin_rescue_regularization=direction.gershgorin_rescue_regularization,
                gershgorin_rescue_used=direction.gershgorin_rescue_used,
                regularization=direction.regularization,
                last_attempted_regularization=direction.last_attempted_regularization,
                factor_nnz=direction.factor_nnz,
                factorization_attempts=direction.factorization_attempts,
                factor_certificate_attempts=direction.factor_certificate_attempts,
                linear_solve_attempts=direction.linear_solve_attempts,
                linear_refinement_steps=direction.refinement_steps,
                factor_permutations_match=direction.factor_permutations_match,
                factor_l_unit_diagonal_error=direction.factor_l_unit_diagonal_error,
                factor_minimum_diagonal=direction.factor_minimum_diagonal,
                factor_maximum_diagonal_magnitude=direction.factor_maximum_diagonal_magnitude,
                factor_minimum_diagonal_relative=direction.factor_minimum_diagonal_relative,
                factor_relation_relative_residual=direction.factor_relation_relative_residual,
                factorization_relative_residual=direction.factorization_relative_residual,
                factor_certificate_passed=direction.factor_certificate_passed,
                line_search_trials=iteration_line_search_trials,
                linear_relative_residual=direction.linear_relative_residual,
                directional_derivative=directional_derivative,
                accepted_step_norm=step_norm,
                accepted_step_size=step_size,
                eigensolve_seconds=direction.eigensolve_seconds,
                factorization_seconds=direction.factorization_seconds,
                linear_solve_seconds=direction.linear_solve_seconds,
                line_search_seconds=line_search_seconds,
                elapsed_seconds=time.perf_counter() - start,
            )
        )
        z_scale = max(float(np.linalg.norm(z)), 1.0)
        z = candidate
        accepted_iterations += 1
        stalled_after_update = step_norm <= cfg.step_relative_tolerance * z_scale
