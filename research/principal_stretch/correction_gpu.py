# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Matrix-free stable-Neo-Hookean correction reference.

This module is the deterministic CPU oracle for the MG-VBD correction path.
It deliberately forms neither a global stiffness matrix nor an exact elastic
Hessian.  At a frozen position it evaluates the exact stable-Neo-Hookean
gradient and applies the positive-semidefinite Gauss-Newton operator

``A[p] = M p / dt^2 + sum_t B_t^T (mu I + lambda c c^T) B_t p``,

where ``c`` is the cofactor of the deformation gradient.  The indefinite
``(det(F) - alpha) d(cof(F))`` part of the exact Hessian is omitted.  Positive
free masses therefore make ``A`` symmetric positive definite even at flat or
inverted finite tetrahedra.

Dirichlet degrees of freedom are eliminated exactly: all Krylov vectors
contain free vertices only, and reconstruction writes the prescribed pin
targets.  The fixed-count PCG implementation records algebraic early
convergence as a work shortfall, and the fixed-alpha Armijo safeguard exposes
deterministic work and failure records.  This NumPy implementation is a
quality oracle; its timings must not be used as GPU performance evidence.
Its active CPU recurrence may exit on algebraic convergence or failure and is
not work-equivalent to a captured GPU implementation that launches a fixed
masked iteration schedule.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Callable, Sequence

import numpy as np
import torch

from .newton_baseline import NewtonProblem

_BLOCK_JACOBI_PRECONDITIONER_IDENTITY = "block-jacobi-3x3-v1"


def _readonly_float64(value: np.ndarray | torch.Tensor | Sequence[float], name: str) -> np.ndarray:
    """Return a finite float64 array backed by immutable bytes."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    owned = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(owned).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _readonly_int64(value: np.ndarray | torch.Tensor | Sequence[int], name: str) -> np.ndarray:
    """Return an owned, read-only int64 array without accepting float indices."""
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"{name} must contain integers")
        value = value.detach().cpu().numpy()
    else:
        input_array = np.asarray(value)
        if input_array.size and input_array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integers")
    owned = np.array(value, dtype=np.int64, order="C", copy=True)
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _readonly_copy(value: np.ndarray, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Freeze a computed float64 array and optionally enforce its shape."""
    owned = np.array(value, dtype=np.float64, order="C", copy=True)
    if shape is not None and owned.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {owned.shape}")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _finite_scaled_norm(value: np.ndarray) -> float | None:
    """Return a finite Euclidean norm without intermediate overflow.

    ``None`` means either an input element or the mathematical norm cannot be
    represented by float64.  Callers turn that state into an explicit failure
    record rather than leaking ``inf`` into a validated result.
    """
    vector = np.asarray(value, dtype=np.float64)
    if not np.isfinite(vector).all():
        return None
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        return 0.0
    scaled_norm = math.sqrt(float(np.sum((vector / scale) ** 2)))
    if not math.isfinite(scaled_norm) or scale > np.finfo(np.float64).max / scaled_norm:
        return None
    return scale * scaled_norm


def _resolve_preconditioner_identity(
    preconditioner: Preconditioner | None,
    identity: str | None,
) -> str:
    """Bind a deterministic identity to the selected preconditioner."""
    if preconditioner is None:
        if identity is not None and identity != _BLOCK_JACOBI_PRECONDITIONER_IDENTITY:
            raise ValueError("the default block-Jacobi preconditioner identity cannot be relabelled")
        return _BLOCK_JACOBI_PRECONDITIONER_IDENTITY
    if type(identity) is not str or not identity:
        raise ValueError("a custom preconditioner requires a non-empty exact string identity")
    return identity


def _cofactor_3x3(matrix: np.ndarray) -> np.ndarray:
    """Polynomial derivative of the 3x3 determinant.

    Unlike ``det(F) * inv(F).T``, this expression remains finite and exact at
    rank loss.  ``matrix`` may have arbitrary leading dimensions.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end in (3, 3), got {matrix.shape}")
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]
    return np.stack(
        (
            e * i - f * h,
            f * g - d * i,
            d * h - e * g,
            c * h - b * i,
            a * i - c * g,
            b * g - a * h,
            b * f - c * e,
            c * d - a * f,
            a * e - b * d,
        ),
        axis=-1,
    ).reshape(matrix.shape)


def _determinant_from_cofactor_polynomial(matrix: np.ndarray) -> np.ndarray:
    """Evaluate the matching polynomial determinant."""
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _scatter_tet_vectors(tets: np.ndarray, values: np.ndarray, n_vertices: int) -> np.ndarray:
    """Deterministically scatter ``[T, 4, 3]`` vectors to vertices."""
    result = np.zeros((n_vertices, 3), dtype=np.float64)
    # The local-corner loop fixes accumulation order and avoids relying on
    # implementation-specific parallel reduction order in this CPU oracle.
    for corner in range(4):
        np.add.at(result, tets[:, corner], values[:, corner])
    return result


@dataclasses.dataclass(frozen=True, eq=False)
class MatrixFreeStableNHOperator:
    """Frozen exact-gradient / PSD-Gauss-Newton operator at one position."""

    positions: np.ndarray
    tets: np.ndarray
    shape_gradients: np.ndarray
    volumes: np.ndarray
    mass: np.ndarray
    mu: np.ndarray
    lam: np.ndarray
    inertial_target: np.ndarray
    pinned: np.ndarray
    free: np.ndarray
    pin_targets: np.ndarray
    dt: float
    deformation_gradients: np.ndarray = dataclasses.field(init=False, repr=False)
    cofactors: np.ndarray = dataclasses.field(init=False, repr=False)
    determinants: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        positions = _readonly_float64(self.positions, "positions")
        tets = _readonly_int64(self.tets, "tets")
        shape_gradients = _readonly_float64(self.shape_gradients, "shape_gradients")
        volumes = _readonly_float64(self.volumes, "volumes")
        mass = _readonly_float64(self.mass, "mass")
        mu = _readonly_float64(self.mu, "mu")
        lam = _readonly_float64(self.lam, "lam")
        inertial_target = _readonly_float64(self.inertial_target, "inertial_target")
        pinned = _readonly_int64(self.pinned, "pinned").reshape(-1)
        free = _readonly_int64(self.free, "free").reshape(-1)
        pin_targets = _readonly_float64(self.pin_targets, "pin_targets")

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (V, 3), got {positions.shape}")
        n_vertices = positions.shape[0]
        if tets.ndim != 2 or tets.shape[1] != 4 or tets.shape[0] == 0:
            raise ValueError(f"tets must have non-empty shape (T, 4), got {tets.shape}")
        n_tets = tets.shape[0]
        if tets.min() < 0 or tets.max() >= n_vertices:
            raise ValueError("tets contains an out-of-range vertex")
        expected_shapes = {
            "shape_gradients": (n_tets, 4, 3),
            "volumes": (n_tets,),
            "mass": (n_vertices,),
            "mu": (n_tets,),
            "lam": (n_tets,),
            "inertial_target": (n_vertices, 3),
            "pin_targets": (pinned.size, 3),
        }
        arrays = {
            "shape_gradients": shape_gradients,
            "volumes": volumes,
            "mass": mass,
            "mu": mu,
            "lam": lam,
            "inertial_target": inertial_target,
            "pin_targets": pin_targets,
        }
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {arrays[name].shape}")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if (volumes <= 0.0).any():
            raise ValueError("volumes must be strictly positive")
        if (mass < 0.0).any() or (mass[free] <= 0.0).any():
            raise ValueError("mass must be non-negative and positive at every free vertex")
        active = (mu > 0.0) | (lam > 0.0)
        if (mu < 0.0).any() or (lam < 0.0).any() or (active & (lam <= 0.0)).any():
            raise ValueError("material coefficients must match stable Neo-Hookean's physical domain")
        if pinned.size + free.size != n_vertices:
            raise ValueError("pinned and free must partition all vertices")
        concatenated = np.concatenate((pinned, free))
        if np.unique(concatenated).size != n_vertices or concatenated.min() < 0 or concatenated.max() >= n_vertices:
            raise ValueError("pinned and free must be a disjoint in-range partition")
        if pinned.size and not np.array_equal(positions[pinned], pin_targets):
            raise ValueError("positions must contain the exact pin targets")

        with np.errstate(over="ignore", invalid="ignore"):
            deformation_gradients = np.einsum(
                "tac,tad->tdc",
                shape_gradients,
                positions[tets],
                optimize=False,
            )
            cofactors = _cofactor_3x3(deformation_gradients)
            determinants = _determinant_from_cofactor_polynomial(deformation_gradients)
        if not (
            np.isfinite(deformation_gradients).all()
            and np.isfinite(cofactors).all()
            and np.isfinite(determinants).all()
        ):
            raise ValueError("deformation geometry must remain finite")

        for name, value in (
            ("positions", positions),
            ("tets", tets),
            ("shape_gradients", shape_gradients),
            ("volumes", volumes),
            ("mass", mass),
            ("mu", mu),
            ("lam", lam),
            ("inertial_target", inertial_target),
            ("pinned", pinned),
            ("free", free),
            ("pin_targets", pin_targets),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "deformation_gradients", _readonly_copy(deformation_gradients))
        object.__setattr__(self, "cofactors", _readonly_copy(cofactors))
        object.__setattr__(self, "determinants", _readonly_copy(determinants))

    @classmethod
    def from_problem(
        cls,
        problem: NewtonProblem,
        positions: np.ndarray | torch.Tensor,
    ) -> MatrixFreeStableNHOperator:
        """Snapshot a validated common-objective problem at ``positions``.

        Pin coordinates supplied by the caller must already equal the exact
        targets.  Rejecting a mismatched state avoids silently changing the
        objective whose correction is being measured.
        """
        if not isinstance(problem, NewtonProblem):
            raise TypeError("problem must be a NewtonProblem")
        x = _readonly_float64(positions, "positions")
        expected = (problem.n_vertices, 3)
        if x.shape != expected:
            raise ValueError(f"positions must have shape {expected}, got {x.shape}")
        pinned = problem.pinned.detach().cpu().numpy()
        pin_targets = problem.pin_targets.detach().cpu().numpy()
        if pinned.size and not np.array_equal(x[pinned], pin_targets):
            raise ValueError("positions must contain the exact pin targets")
        return cls(
            positions=x,
            tets=problem.tets,
            shape_gradients=problem.J,
            volumes=problem.volume,
            mass=problem.mass,
            mu=problem.mu,
            lam=problem.lam,
            inertial_target=problem.inertial_target,
            pinned=problem.pinned,
            free=problem.free,
            pin_targets=pin_targets,
            dt=problem.dt,
        )

    @property
    def n_vertices(self) -> int:
        """Number of full vertices."""
        return int(self.positions.shape[0])

    @property
    def n_free_dofs(self) -> int:
        """Number of scalar unconstrained degrees of freedom."""
        return int(self.free.size * 3)

    @property
    def minimum_determinant(self) -> float:
        """Minimum determinant over every tetrahedron, including disabled rows."""
        return float(self.determinants.min())

    def positions_from_free(self, free_positions: np.ndarray | Sequence[float]) -> np.ndarray:
        """Reconstruct full positions and write exact Dirichlet targets."""
        free_array = np.asarray(free_positions, dtype=np.float64)
        if free_array.shape not in ((self.free.size, 3), (self.n_free_dofs,)):
            raise ValueError(
                f"free_positions must have shape ({self.free.size}, 3) or ({self.n_free_dofs},), got {free_array.shape}"
            )
        if not np.isfinite(free_array).all():
            raise ValueError("free_positions must be finite")
        result = np.empty_like(self.positions)
        result[self.free] = free_array.reshape(-1, 3)
        result[self.pinned] = self.pin_targets
        return _readonly_copy(result)

    def free_from_positions(self, positions: np.ndarray | torch.Tensor) -> np.ndarray:
        """Extract a flattened free vector after validating full positions."""
        full = _readonly_float64(positions, "positions")
        if full.shape != self.positions.shape:
            raise ValueError(f"positions must have shape {self.positions.shape}, got {full.shape}")
        return _readonly_copy(full[self.free].reshape(-1))

    def objective(self) -> float:
        """Evaluate the exact common stable-Neo-Hookean objective."""
        delta = self.positions - self.inertial_target
        inertia = 0.5 / (self.dt * self.dt) * float(np.sum(self.mass[:, None] * delta * delta))
        active = (self.mu > 0.0) | (self.lam > 0.0)
        alpha = 1.0 + self.mu / np.maximum(self.lam, 1.0e-6)
        frobenius_sq = np.sum(self.deformation_gradients * self.deformation_gradients, axis=(1, 2))
        density = 0.5 * self.mu * (frobenius_sq - 3.0) + 0.5 * self.lam * (self.determinants - alpha) ** 2
        elastic = float(np.sum(np.where(active, density, 0.0) * self.volumes))
        return inertia + elastic

    def gradient_full(self) -> np.ndarray:
        """Evaluate the exact nonlinear gradient at all vertices [N]."""
        alpha = 1.0 + self.mu / np.maximum(self.lam, 1.0e-6)
        first_piola = (
            self.mu[:, None, None] * self.deformation_gradients
            + (self.lam * (self.determinants - alpha))[:, None, None] * self.cofactors
        )
        contributions = (
            np.einsum(
                "tdc,tac->tad",
                first_piola,
                self.shape_gradients,
                optimize=False,
            )
            * self.volumes[:, None, None]
        )
        gradient = _scatter_tet_vectors(self.tets, contributions, self.n_vertices)
        gradient += self.mass[:, None] * (self.positions - self.inertial_target) / (self.dt * self.dt)
        return _readonly_copy(gradient)

    def gradient_free(self) -> np.ndarray:
        """Evaluate the exact gradient on eliminated free degrees of freedom."""
        return _readonly_copy(self.gradient_full()[self.free].reshape(-1))

    def apply_free(self, direction: np.ndarray | Sequence[float]) -> np.ndarray:
        """Apply the PSD Gauss-Newton operator to a free vector.

        The application uses tet-local deformation-gradient products and
        scatter-adds only; no global sparse or dense matrix is assembled.
        """
        free_direction = np.asarray(direction, dtype=np.float64)
        if free_direction.shape not in ((self.free.size, 3), (self.n_free_dofs,)):
            raise ValueError(
                f"direction must have shape ({self.free.size}, 3) or ({self.n_free_dofs},), got {free_direction.shape}"
            )
        if not np.isfinite(free_direction).all():
            raise ValueError("direction must be finite")
        full_direction = np.zeros_like(self.positions)
        full_direction[self.free] = free_direction.reshape(-1, 3)
        delta_f = np.einsum(
            "tac,tad->tdc",
            self.shape_gradients,
            full_direction[self.tets],
            optimize=False,
        )
        determinant_direction = np.sum(self.cofactors * delta_f, axis=(1, 2))
        delta_piola = (
            self.mu[:, None, None] * delta_f + (self.lam * determinant_direction)[:, None, None] * self.cofactors
        )
        contributions = (
            np.einsum(
                "tdc,tac->tad",
                delta_piola,
                self.shape_gradients,
                optimize=False,
            )
            * self.volumes[:, None, None]
        )
        result = _scatter_tet_vectors(self.tets, contributions, self.n_vertices)
        result += self.mass[:, None] * full_direction / (self.dt * self.dt)
        return _readonly_copy(result[self.free].reshape(-1))

    def apply_full(self, direction: np.ndarray | Sequence[float]) -> np.ndarray:
        """Apply to a full vector whose pinned coordinates are exactly zero."""
        full = np.asarray(direction, dtype=np.float64)
        if full.shape != self.positions.shape:
            raise ValueError(f"direction must have shape {self.positions.shape}, got {full.shape}")
        if not np.isfinite(full).all():
            raise ValueError("direction must be finite")
        if self.pinned.size and not np.array_equal(full[self.pinned], np.zeros((self.pinned.size, 3))):
            raise ValueError("pinned direction entries must be exactly zero")
        result = np.zeros_like(full)
        result[self.free] = self.apply_free(full[self.free]).reshape(-1, 3)
        return _readonly_copy(result)

    def block_diagonal(self) -> np.ndarray:
        """Return exact free-vertex 3x3 diagonal blocks of the operator."""
        blocks = np.zeros((self.n_vertices, 3, 3), dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        blocks += (self.mass / (self.dt * self.dt))[:, None, None] * identity
        for corner in range(4):
            shape = self.shape_gradients[:, corner]
            cofactor_shape = np.einsum("tdc,tc->td", self.cofactors, shape, optimize=False)
            local = self.volumes[:, None, None] * (
                (self.mu * np.sum(shape * shape, axis=1))[:, None, None] * identity
                + self.lam[:, None, None] * cofactor_shape[:, :, None] * cofactor_shape[:, None, :]
            )
            np.add.at(blocks, self.tets[:, corner], local)
        return _readonly_copy(blocks[self.free])

    def block_jacobi_inverse(self) -> np.ndarray:
        """Factor and invert the exact 3x3 free-vertex diagonal blocks."""
        blocks = self.block_diagonal()
        inverse = np.empty_like(blocks)
        identity = np.eye(3, dtype=np.float64)
        for vertex in range(self.free.size):
            try:
                factor = np.linalg.cholesky(blocks[vertex])
                inverse[vertex] = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(f"block {vertex} is not positive definite") from error
        if not np.isfinite(inverse).all():
            raise FloatingPointError("block-Jacobi inverse is nonfinite")
        return _readonly_copy(inverse)

    def apply_block_jacobi(
        self,
        residual: np.ndarray | Sequence[float],
        inverse: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the exact 3x3 block-Jacobi inverse to a free residual."""
        vector = np.asarray(residual, dtype=np.float64)
        if vector.shape not in ((self.free.size, 3), (self.n_free_dofs,)):
            raise ValueError("residual has the wrong free-vector shape")
        if not np.isfinite(vector).all():
            raise ValueError("residual must be finite")
        block_inverse = self.block_jacobi_inverse() if inverse is None else np.asarray(inverse, dtype=np.float64)
        if block_inverse.shape != (self.free.size, 3, 3) or not np.isfinite(block_inverse).all():
            raise ValueError("inverse must contain finite free-vertex 3x3 blocks")
        result = np.einsum("vij,vj->vi", block_inverse, vector.reshape(-1, 3), optimize=False)
        return _readonly_copy(result.reshape(-1))


@dataclasses.dataclass(frozen=True)
class DeterminantSegmentMinimum:
    """Minimum tet determinant over a straight position-space segment."""

    determinant: float
    fraction: float
    tet_index: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.determinant):
            raise ValueError("segment determinant must be finite")
        if not math.isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0:
            raise ValueError("segment fraction must lie in [0, 1]")
        if isinstance(self.tet_index, bool) or not isinstance(self.tet_index, numbers.Integral) or self.tet_index < 0:
            raise ValueError("segment tet index must be a non-negative integer")


def minimum_determinant_on_segment(
    start: MatrixFreeStableNHOperator,
    end: MatrixFreeStableNHOperator,
) -> DeterminantSegmentMinimum:
    """Find the exact cubic determinant minimum from ``start`` to ``end``.

    For each tet, ``det(F0 + s D)`` is a cubic.  Its coefficients follow from
    determinant multilinearity, and the only interior candidates are real
    roots of its quadratic derivative.  Checking these roots prevents a step
    with two positive endpoints from crossing the inverted region in between.
    """
    if not isinstance(start, MatrixFreeStableNHOperator) or not isinstance(end, MatrixFreeStableNHOperator):
        raise TypeError("start and end must be MatrixFreeStableNHOperator instances")
    structural_pairs = (
        (start.tets, end.tets),
        (start.shape_gradients, end.shape_gradients),
        (start.pinned, end.pinned),
        (start.free, end.free),
        (start.pin_targets, end.pin_targets),
    )
    if any(not np.array_equal(left, right) for left, right in structural_pairs):
        raise ValueError("segment endpoints must use the same mesh and constraints")

    delta_positions = end.positions - start.positions
    delta_f = np.einsum(
        "tac,tad->tdc",
        start.shape_gradients,
        delta_positions[start.tets],
        optimize=False,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        c0 = start.determinants
        c1 = np.sum(start.cofactors * delta_f, axis=(1, 2))
        c2 = np.sum(_cofactor_3x3(delta_f) * start.deformation_gradients, axis=(1, 2))
        c3 = _determinant_from_cofactor_polynomial(delta_f)
    coefficients = np.stack((c0, c1, c2, c3), axis=1)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("segment determinant polynomial is nonfinite")

    best_value = math.inf
    best_fraction = 0.0
    best_tet = 0
    epsilon = np.finfo(np.float64).eps
    for tet_index, (constant, linear, quadratic, cubic) in enumerate(coefficients):
        candidates = [0.0, 1.0]
        derivative = np.array((linear, 2.0 * quadratic, 3.0 * cubic), dtype=np.float64)
        scale = float(np.max(np.abs(derivative)))
        if scale > 0.0:
            c = derivative[0] / scale
            b = derivative[1] / scale
            a = derivative[2] / scale
            if abs(a) <= 32.0 * epsilon:
                if abs(b) > 32.0 * epsilon:
                    root = -c / b
                    if 0.0 < root < 1.0:
                        candidates.append(float(root))
            else:
                discriminant = b * b - 4.0 * a * c
                tolerance = 64.0 * epsilon * max(1.0, abs(b * b), abs(4.0 * a * c))
                if discriminant >= -tolerance:
                    square_root = math.sqrt(max(0.0, discriminant))
                    q = -0.5 * (b + math.copysign(square_root, b))
                    roots = (-b / (2.0 * a),) if q == 0.0 else (q / a, c / q)
                    for root in roots:
                        if 0.0 < root < 1.0:
                            candidates.append(float(root))
        for fraction in candidates:
            value = ((cubic * fraction + quadratic) * fraction + linear) * fraction + constant
            if value < best_value:
                best_value = float(value)
                best_fraction = fraction
                best_tet = tet_index
    return DeterminantSegmentMinimum(best_value, best_fraction, best_tet)


@dataclasses.dataclass(frozen=True)
class FixedPCGWork:
    """Exact primitive work completed by one fixed-count PCG solve."""

    operator_applications: int
    residual_verification_applications: int
    preconditioner_builds: int
    preconditioner_applications: int
    inner_products: int
    vector_updates: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{field.name} must be a non-negative integer")


@dataclasses.dataclass(frozen=True)
class FixedPCGIteration:
    """One completed fixed-count PCG update."""

    iteration: int
    residual_norm: float
    direction_curvature: float
    step_size: float
    conjugacy: float | None

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, numbers.Integral) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        values = (self.residual_norm, self.direction_curvature, self.step_size)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("completed PCG iteration scalars must be finite")
        if self.residual_norm < 0.0 or self.direction_curvature <= 0.0 or self.step_size <= 0.0:
            raise ValueError("completed PCG iteration scalars have invalid signs")
        if self.conjugacy is not None and (not math.isfinite(self.conjugacy) or self.conjugacy < 0.0):
            raise ValueError("PCG conjugacy must be finite and non-negative")


_PCG_REASONS = (
    "completed",
    "initial_residual_zero",
    "converged_early",
    "nonfinite_rhs",
    "rhs_norm_overflow",
    "recursive_residual_overflow",
    "true_residual_overflow",
    "preconditioner_factorization",
    "preconditioner_failure",
    "nonpositive_preconditioner",
    "operator_failure",
    "nonpositive_curvature",
    "nonfinite_update",
)


@dataclasses.dataclass(frozen=True, eq=False)
class FixedPCGResult:
    """Fixed-budget PCG solution with fail-closed status and exact work.

    ``success`` denotes a mathematically valid solution prefix.  Consumers
    that require identical work must additionally require
    :attr:`consumed_exact_iteration_count`; exact algebraic convergence can
    otherwise end the recurrence before its requested count.
    """

    solution: np.ndarray
    success: bool
    reason: str
    preconditioner_identity: str
    requested_iterations: int
    completed_iterations: int
    rhs_norm: float | None
    recursive_residual_norm: float | None
    true_residual_norm: float | None
    trace: tuple[FixedPCGIteration, ...]
    work: FixedPCGWork

    def __post_init__(self) -> None:
        solution = _readonly_copy(np.asarray(self.solution, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "trace", tuple(self.trace))
        if self.reason not in _PCG_REASONS:
            raise ValueError(f"unknown PCG reason: {self.reason}")
        expected_success = self.reason in ("completed", "initial_residual_zero", "converged_early")
        if self.success != expected_success:
            raise ValueError("PCG success flag does not agree with its reason")
        if type(self.preconditioner_identity) is not str or not self.preconditioner_identity:
            raise ValueError("PCG preconditioner identity must be a non-empty exact string")
        for name in ("requested_iterations", "completed_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.requested_iterations < 1 or self.completed_iterations > self.requested_iterations:
            raise ValueError("PCG iteration counts are inconsistent")
        if len(self.trace) != self.completed_iterations:
            raise ValueError("PCG trace length does not match completed iterations")
        if tuple(item.iteration for item in self.trace) != tuple(range(self.completed_iterations)):
            raise ValueError("PCG trace iteration indices must be contiguous")
        if self.reason == "completed" and self.completed_iterations != self.requested_iterations:
            raise ValueError("completed PCG must consume its exact requested iteration count")
        if self.rhs_norm is not None and (not math.isfinite(self.rhs_norm) or self.rhs_norm < 0.0):
            raise ValueError("rhs_norm must be finite and non-negative when present")
        if self.recursive_residual_norm is not None and (
            not math.isfinite(self.recursive_residual_norm) or self.recursive_residual_norm < 0.0
        ):
            raise ValueError("recursive_residual_norm must be finite and non-negative when present")
        if self.true_residual_norm is not None and (
            not math.isfinite(self.true_residual_norm) or self.true_residual_norm < 0.0
        ):
            raise ValueError("true_residual_norm must be finite and non-negative when present")
        if self.success and (
            self.rhs_norm is None or self.recursive_residual_norm is None or self.true_residual_norm is None
        ):
            raise ValueError("successful PCG must have finite norm records")
        if self.work.residual_verification_applications > self.work.operator_applications:
            raise ValueError("PCG residual-verification work exceeds total operator work")
        if self.success and not np.isfinite(solution).all():
            raise ValueError("successful PCG solution must be finite")

    def deterministic_record(self) -> dict[str, object]:
        """Serialize status, trace, and primitive work without timings."""
        return {
            "success": self.success,
            "reason": self.reason,
            "preconditioner_identity": self.preconditioner_identity,
            "requested_iterations": self.requested_iterations,
            "completed_iterations": self.completed_iterations,
            "consumed_exact_iteration_count": self.consumed_exact_iteration_count,
            "rhs_norm": self.rhs_norm,
            "recursive_residual_norm": self.recursive_residual_norm,
            "true_residual_norm": self.true_residual_norm,
            "work": dataclasses.asdict(self.work),
            "trace": [dataclasses.asdict(item) for item in self.trace],
        }

    @property
    def consumed_exact_iteration_count(self) -> bool:
        """Whether the recurrence completed every requested iteration."""
        return self.completed_iterations == self.requested_iterations

    @property
    def final_residual_norm(self) -> float:
        """Independently recomputed final residual norm."""
        return self.true_residual_norm if self.true_residual_norm is not None else math.nan


Preconditioner = Callable[[np.ndarray], np.ndarray]


def _pcg_result(
    solution: np.ndarray,
    reason: str,
    requested_iterations: int,
    completed_iterations: int,
    rhs_norm: float | None,
    residual_norm: float | None,
    trace: list[FixedPCGIteration],
    *,
    preconditioner_identity: str,
    true_residual_norm: float | None = None,
    operator_applications: int,
    residual_verification_applications: int,
    preconditioner_builds: int,
    preconditioner_applications: int,
    inner_products: int,
    vector_updates: int,
) -> FixedPCGResult:
    return FixedPCGResult(
        solution=solution,
        success=reason in ("completed", "initial_residual_zero", "converged_early"),
        reason=reason,
        preconditioner_identity=preconditioner_identity,
        requested_iterations=requested_iterations,
        completed_iterations=completed_iterations,
        rhs_norm=rhs_norm,
        recursive_residual_norm=residual_norm,
        true_residual_norm=true_residual_norm,
        trace=tuple(trace),
        work=FixedPCGWork(
            operator_applications=operator_applications,
            residual_verification_applications=residual_verification_applications,
            preconditioner_builds=preconditioner_builds,
            preconditioner_applications=preconditioner_applications,
            inner_products=inner_products,
            vector_updates=vector_updates,
        ),
    )


def solve_fixed_pcg(
    operator: MatrixFreeStableNHOperator,
    rhs: np.ndarray | Sequence[float],
    iterations: int,
    preconditioner: Preconditioner | None = None,
    preconditioner_identity: str | None = None,
) -> FixedPCGResult:
    """Solve ``A x = rhs`` from zero using a deterministic fixed PCG budget.

    Exact algebraic convergence is the only early-success condition.  Every
    other numerical breakdown returns a failure record and the last finite
    solution; it is never re-labelled as a completed fixed-budget solve.
    """
    if not isinstance(operator, MatrixFreeStableNHOperator):
        raise TypeError("operator must be a MatrixFreeStableNHOperator")
    if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    requested = int(iterations)
    resolved_preconditioner_identity = _resolve_preconditioner_identity(preconditioner, preconditioner_identity)
    vector = np.asarray(rhs, dtype=np.float64)
    if vector.shape != (operator.n_free_dofs,):
        raise ValueError(f"rhs must have shape ({operator.n_free_dofs},), got {vector.shape}")

    x = np.zeros(operator.n_free_dofs, dtype=np.float64)
    trace: list[FixedPCGIteration] = []
    counters = {
        "preconditioner_identity": resolved_preconditioner_identity,
        "operator_applications": 0,
        "residual_verification_applications": 0,
        "preconditioner_builds": 0,
        "preconditioner_applications": 0,
        "inner_products": 0,
        "vector_updates": 0,
    }

    def verify_true_residual(solution: np.ndarray) -> tuple[float | None, str | None]:
        counters["operator_applications"] += 1
        counters["residual_verification_applications"] += 1
        try:
            product = np.asarray(operator.apply_free(solution))
        except (FloatingPointError, ValueError):
            return None, "operator_failure"
        with np.errstate(over="ignore", invalid="ignore"):
            residual = vector - product
        if not np.isfinite(residual).all():
            return None, "true_residual_overflow"
        norm = _finite_scaled_norm(residual)
        return (norm, None) if norm is not None else (None, "true_residual_overflow")

    if not np.isfinite(vector).all():
        return _pcg_result(x, "nonfinite_rhs", requested, 0, None, None, trace, **counters)
    r = np.array(vector, copy=True)
    rhs_norm = _finite_scaled_norm(r)
    if rhs_norm is None:
        return _pcg_result(x, "rhs_norm_overflow", requested, 0, None, None, trace, **counters)
    if rhs_norm == 0.0:
        true_residual_norm, verification_failure = verify_true_residual(x)
        reason = "initial_residual_zero" if verification_failure is None else verification_failure
        return _pcg_result(
            x,
            reason,
            requested,
            0,
            0.0,
            0.0,
            trace,
            true_residual_norm=true_residual_norm,
            **counters,
        )

    if preconditioner is None:
        counters["preconditioner_builds"] = 1
        try:
            inverse = operator.block_jacobi_inverse()
        except (FloatingPointError, np.linalg.LinAlgError):
            return _pcg_result(
                x,
                "preconditioner_factorization",
                requested,
                0,
                rhs_norm,
                rhs_norm,
                trace,
                **counters,
            )

        def apply_preconditioner(value: np.ndarray) -> np.ndarray:
            return np.asarray(operator.apply_block_jacobi(value, inverse))

    else:
        if not callable(preconditioner):
            raise TypeError("preconditioner must be callable")
        apply_preconditioner = preconditioner

    def precondition(value: np.ndarray) -> np.ndarray | None:
        counters["preconditioner_applications"] += 1
        try:
            output = np.asarray(apply_preconditioner(np.array(value, copy=True)), dtype=np.float64)
        except Exception:
            return None
        if output.shape != value.shape or not np.isfinite(output).all():
            return None
        return np.array(output, copy=True)

    z = precondition(r)
    if z is None:
        return _pcg_result(x, "preconditioner_failure", requested, 0, rhs_norm, rhs_norm, trace, **counters)
    rho = float(np.dot(r, z))
    counters["inner_products"] += 1
    if not math.isfinite(rho) or rho <= 0.0:
        return _pcg_result(x, "nonpositive_preconditioner", requested, 0, rhs_norm, rhs_norm, trace, **counters)
    p = z
    residual_norm = rhs_norm

    for iteration in range(requested):
        counters["operator_applications"] += 1
        try:
            ap = np.asarray(operator.apply_free(p))
        except (FloatingPointError, ValueError):
            return _pcg_result(
                x,
                "operator_failure",
                requested,
                len(trace),
                rhs_norm,
                residual_norm,
                trace,
                **counters,
            )
        curvature = float(np.dot(p, ap))
        counters["inner_products"] += 1
        if not math.isfinite(curvature) or curvature <= 0.0:
            return _pcg_result(
                x,
                "nonpositive_curvature",
                requested,
                len(trace),
                rhs_norm,
                residual_norm,
                trace,
                **counters,
            )
        step_size = rho / curvature
        next_x = x + step_size * p
        next_r = r - step_size * ap
        counters["vector_updates"] += 2
        if not (
            math.isfinite(step_size) and step_size > 0.0 and np.isfinite(next_x).all() and np.isfinite(next_r).all()
        ):
            return _pcg_result(
                x,
                "nonfinite_update",
                requested,
                len(trace),
                rhs_norm,
                residual_norm,
                trace,
                **counters,
            )
        x = next_x
        r = next_r
        next_residual_norm = _finite_scaled_norm(r)
        if next_residual_norm is None:
            return _pcg_result(
                x,
                "recursive_residual_overflow",
                requested,
                len(trace),
                rhs_norm,
                None,
                trace,
                **counters,
            )
        residual_norm = next_residual_norm

        if residual_norm == 0.0:
            trace.append(FixedPCGIteration(iteration, residual_norm, curvature, step_size, None))
            true_residual_norm, verification_failure = verify_true_residual(x)
            successful_reason = "converged_early" if iteration + 1 < requested else "completed"
            return _pcg_result(
                x,
                successful_reason if verification_failure is None else verification_failure,
                requested,
                iteration + 1,
                rhs_norm,
                residual_norm,
                trace,
                true_residual_norm=true_residual_norm,
                **counters,
            )
        if iteration + 1 == requested:
            trace.append(FixedPCGIteration(iteration, residual_norm, curvature, step_size, None))
            break

        z = precondition(r)
        if z is None:
            trace.append(FixedPCGIteration(iteration, residual_norm, curvature, step_size, None))
            return _pcg_result(
                x,
                "preconditioner_failure",
                requested,
                iteration + 1,
                rhs_norm,
                residual_norm,
                trace,
                **counters,
            )
        next_rho = float(np.dot(r, z))
        counters["inner_products"] += 1
        if not math.isfinite(next_rho) or next_rho <= 0.0:
            trace.append(FixedPCGIteration(iteration, residual_norm, curvature, step_size, None))
            return _pcg_result(
                x,
                "nonpositive_preconditioner",
                requested,
                iteration + 1,
                rhs_norm,
                residual_norm,
                trace,
                **counters,
            )
        conjugacy = next_rho / rho
        next_p = z + conjugacy * p
        counters["vector_updates"] += 1
        if not math.isfinite(conjugacy) or conjugacy < 0.0 or not np.isfinite(next_p).all():
            trace.append(FixedPCGIteration(iteration, residual_norm, curvature, step_size, None))
            return _pcg_result(
                x,
                "nonfinite_update",
                requested,
                iteration + 1,
                rhs_norm,
                residual_norm,
                trace,
                **counters,
            )
        trace.append(FixedPCGIteration(iteration, residual_norm, curvature, step_size, conjugacy))
        p = next_p
        rho = next_rho

    true_residual_norm, verification_failure = verify_true_residual(x)
    return _pcg_result(
        x,
        "completed" if verification_failure is None else verification_failure,
        requested,
        requested,
        rhs_norm,
        residual_norm,
        trace,
        true_residual_norm=true_residual_norm,
        **counters,
    )


@dataclasses.dataclass(frozen=True)
class MatrixFreeCorrectionConfig:
    """Fixed-work PCG and fixed-alpha correction policy."""

    pcg_iterations: int = 4
    alpha: float = 1.0
    minimum_determinant: float = 0.0
    armijo: float = 1.0e-4

    def validate(self) -> None:
        """Validate fixed work and safeguard settings."""
        if (
            isinstance(self.pcg_iterations, bool)
            or not isinstance(self.pcg_iterations, numbers.Integral)
            or self.pcg_iterations < 1
        ):
            raise ValueError("pcg_iterations must be a positive integer")
        for name in ("alpha", "minimum_determinant", "armijo"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if self.minimum_determinant < 0.0:
            raise ValueError("minimum_determinant must be non-negative")
        if not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must lie in (0, 1)")

    def deterministic_record(self) -> dict[str, object]:
        """Serialize the numerical correction contract."""
        self.validate()
        return {
            **dataclasses.asdict(self),
            "contract": "matrix-free-stable-nh-gn-pcg-v1",
            "gradient": "exact-stable-neo-hookean",
            "operator": "psd-gauss-newton-plus-inertia",
            "constraint_policy": "free-dof-elimination-exact-pins",
            "preconditioner": "identity-bound-in-each-pcg-and-correction-record",
            "linear_policy": "zero-start-fixed-count-pcg-reject-work-shortfall",
            "cpu_execution_semantics": "active-early-exit-not-equivalent-to-captured-masked-fixed-launches",
            "update_policy": "fixed-alpha-armijo-fail-closed",
            "performance_evidence": False,
        }


@dataclasses.dataclass(frozen=True)
class MatrixFreeCorrectionWork:
    """Exact high-level work for one safeguarded correction attempt."""

    operator_builds: int
    objective_evaluations: int
    gradient_evaluations: int
    candidate_evaluations: int
    pcg: FixedPCGWork

    def __post_init__(self) -> None:
        for name in ("operator_builds", "objective_evaluations", "gradient_evaluations", "candidate_evaluations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


_CORRECTION_REASONS = (
    "accepted",
    "stationary",
    "pcg_failure",
    "pcg_work_shortfall",
    "non_descent",
    "candidate_nonfinite",
    "segment_inversion",
    "objective_increase",
)


@dataclasses.dataclass(frozen=True, eq=False)
class MatrixFreeCorrectionResult:
    """One fixed-alpha correction or the exact fail-closed fallback state."""

    x: np.ndarray
    accepted: bool
    reason: str
    alpha: float
    initial_objective: float
    candidate_objective: float | None
    final_objective: float
    initial_gradient_norm: float
    candidate_gradient_norm: float | None
    final_gradient_norm: float
    initial_minimum_determinant: float
    candidate_minimum_determinant: float | None
    final_minimum_determinant: float
    directional_derivative: float | None
    pcg: FixedPCGResult | None
    preconditioner_identity: str
    config: MatrixFreeCorrectionConfig
    work: MatrixFreeCorrectionWork
    segment_minimum_determinant: float | None = None
    segment_minimum_fraction: float | None = None

    def __post_init__(self) -> None:
        x = _readonly_copy(np.asarray(self.x, dtype=np.float64))
        object.__setattr__(self, "x", x)
        if self.reason not in _CORRECTION_REASONS:
            raise ValueError(f"unknown correction reason: {self.reason}")
        if self.accepted != (self.reason == "accepted"):
            raise ValueError("correction acceptance must agree with its reason")
        if type(self.config) is not MatrixFreeCorrectionConfig:
            raise ValueError("correction config must be an exact MatrixFreeCorrectionConfig")
        self.config.validate()
        if self.alpha != self.config.alpha:
            raise ValueError("correction alpha must match its config")
        if type(self.preconditioner_identity) is not str or not self.preconditioner_identity:
            raise ValueError("correction preconditioner identity must be a non-empty exact string")
        required = (
            self.alpha,
            self.initial_objective,
            self.final_objective,
            self.initial_gradient_norm,
            self.final_gradient_norm,
            self.initial_minimum_determinant,
            self.final_minimum_determinant,
        )
        optional = (
            self.candidate_objective,
            self.candidate_gradient_norm,
            self.candidate_minimum_determinant,
            self.directional_derivative,
            self.segment_minimum_determinant,
            self.segment_minimum_fraction,
        )
        if any(not math.isfinite(value) for value in required) or any(
            value is not None and not math.isfinite(value) for value in optional
        ):
            raise ValueError("correction scalar records must be finite")
        if not 0.0 < self.alpha <= 1.0 or self.initial_gradient_norm < 0.0 or self.final_gradient_norm < 0.0:
            raise ValueError("correction alpha or gradient norms are invalid")
        if self.candidate_gradient_norm is not None and self.candidate_gradient_norm < 0.0:
            raise ValueError("candidate gradient norm must be non-negative")
        if self.segment_minimum_fraction is not None and not 0.0 <= self.segment_minimum_fraction <= 1.0:
            raise ValueError("segment minimum fraction must lie in [0, 1]")
        if self.pcg is None:
            if self.work.pcg != _empty_pcg_work():
                raise ValueError("correction without PCG must have zero PCG work")
        elif self.work.pcg != self.pcg.work:
            raise ValueError("correction and PCG work records disagree")
        if self.pcg is not None and self.preconditioner_identity != self.pcg.preconditioner_identity:
            raise ValueError("correction and PCG preconditioner identities disagree")
        if self.accepted:
            if self.pcg is None or not self.pcg.success or not self.pcg.consumed_exact_iteration_count:
                raise ValueError("accepted correction requires a successful exact-work PCG solve")
            if self.candidate_objective != self.final_objective:
                raise ValueError("accepted correction must return its candidate objective")
            if (
                self.candidate_objective is None
                or self.candidate_gradient_norm is None
                or self.candidate_minimum_determinant is None
                or self.segment_minimum_determinant is None
                or self.segment_minimum_fraction is None
                or self.directional_derivative is None
            ):
                raise ValueError("accepted correction requires complete candidate records")
            if self.directional_derivative >= 0.0:
                raise ValueError("accepted correction direction must be descending")
            if (
                self.candidate_minimum_determinant <= self.config.minimum_determinant
                or self.segment_minimum_determinant <= self.config.minimum_determinant
            ):
                raise ValueError("accepted correction violates determinant safety")
            armijo_limit = self.initial_objective + self.config.armijo * self.config.alpha * self.directional_derivative
            if self.candidate_objective >= self.initial_objective or self.candidate_objective > armijo_limit:
                raise ValueError("accepted correction violates strict Armijo decrease")
        elif self.final_objective != self.initial_objective or self.final_gradient_norm != self.initial_gradient_norm:
            raise ValueError("rejected correction must report exact fallback metrics")
        if self.reason == "stationary" and (
            self.initial_gradient_norm != 0.0 or self.pcg is not None or self.candidate_objective is not None
        ):
            raise ValueError("stationary correction record is inconsistent")
        if self.reason == "segment_inversion" and (
            self.segment_minimum_determinant is None
            or self.segment_minimum_determinant > self.config.minimum_determinant
        ):
            raise ValueError("segment-inversion reason lacks a violating determinant")
        if self.reason == "objective_increase":
            if self.candidate_objective is None or self.directional_derivative is None:
                raise ValueError("objective-increase reason lacks candidate records")
            armijo_limit = self.initial_objective + self.config.armijo * self.config.alpha * self.directional_derivative
            if self.candidate_objective < self.initial_objective and self.candidate_objective <= armijo_limit:
                raise ValueError("objective-increase reason does not violate Armijo")

    @property
    def used_fallback(self) -> bool:
        """Whether the fixed-alpha candidate was not installed."""
        return not self.accepted

    def deterministic_record(self) -> dict[str, object]:
        """Serialize quality and work records without timings or position data."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "alpha": self.alpha,
            "initial_objective": self.initial_objective,
            "candidate_objective": self.candidate_objective,
            "final_objective": self.final_objective,
            "initial_gradient_norm": self.initial_gradient_norm,
            "candidate_gradient_norm": self.candidate_gradient_norm,
            "final_gradient_norm": self.final_gradient_norm,
            "initial_minimum_determinant": self.initial_minimum_determinant,
            "candidate_minimum_determinant": self.candidate_minimum_determinant,
            "final_minimum_determinant": self.final_minimum_determinant,
            "directional_derivative": self.directional_derivative,
            "segment_minimum_determinant": self.segment_minimum_determinant,
            "segment_minimum_fraction": self.segment_minimum_fraction,
            "pcg": None if self.pcg is None else self.pcg.deterministic_record(),
            "preconditioner_identity": self.preconditioner_identity,
            "config": self.config.deterministic_record(),
            "work": {
                "operator_builds": self.work.operator_builds,
                "objective_evaluations": self.work.objective_evaluations,
                "gradient_evaluations": self.work.gradient_evaluations,
                "candidate_evaluations": self.work.candidate_evaluations,
                "pcg": dataclasses.asdict(self.work.pcg),
            },
            "performance_evidence": False,
        }


def _empty_pcg_work() -> FixedPCGWork:
    return FixedPCGWork(0, 0, 0, 0, 0, 0)


def solve_matrix_free_correction(
    problem: NewtonProblem,
    x_initial: np.ndarray | torch.Tensor,
    config: MatrixFreeCorrectionConfig | None = None,
    *,
    preconditioner: Preconditioner | None = None,
    preconditioner_identity: str | None = None,
) -> MatrixFreeCorrectionResult:
    """Attempt one fixed-work, fixed-alpha, fail-closed correction.

    A rejected candidate returns the validated initial positions bit-for-bit.
    No line search or hidden retry is performed.
    """
    cfg = MatrixFreeCorrectionConfig() if config is None else config
    if not isinstance(cfg, MatrixFreeCorrectionConfig):
        raise TypeError("config must be a MatrixFreeCorrectionConfig")
    cfg.validate()
    resolved_preconditioner_identity = _resolve_preconditioner_identity(preconditioner, preconditioner_identity)
    initial = MatrixFreeStableNHOperator.from_problem(problem, x_initial)
    initial_objective = initial.objective()
    gradient = initial.gradient_free()
    gradient_norm = float(np.linalg.norm(gradient))
    initial_determinant = initial.minimum_determinant
    if not math.isfinite(initial_objective) or not math.isfinite(gradient_norm):
        raise ValueError("initial objective and gradient must be finite")
    if gradient_norm == 0.0:
        return MatrixFreeCorrectionResult(
            x=initial.positions,
            accepted=False,
            reason="stationary",
            alpha=cfg.alpha,
            initial_objective=initial_objective,
            candidate_objective=None,
            final_objective=initial_objective,
            initial_gradient_norm=gradient_norm,
            candidate_gradient_norm=None,
            final_gradient_norm=gradient_norm,
            initial_minimum_determinant=initial_determinant,
            candidate_minimum_determinant=None,
            final_minimum_determinant=initial_determinant,
            directional_derivative=None,
            pcg=None,
            preconditioner_identity=resolved_preconditioner_identity,
            config=cfg,
            work=MatrixFreeCorrectionWork(1, 1, 1, 0, _empty_pcg_work()),
        )

    pcg = solve_fixed_pcg(
        initial,
        -gradient,
        cfg.pcg_iterations,
        preconditioner,
        resolved_preconditioner_identity,
    )
    if not pcg.success or not pcg.consumed_exact_iteration_count:
        reason = "pcg_failure" if not pcg.success else "pcg_work_shortfall"
        return MatrixFreeCorrectionResult(
            x=initial.positions,
            accepted=False,
            reason=reason,
            alpha=cfg.alpha,
            initial_objective=initial_objective,
            candidate_objective=None,
            final_objective=initial_objective,
            initial_gradient_norm=gradient_norm,
            candidate_gradient_norm=None,
            final_gradient_norm=gradient_norm,
            initial_minimum_determinant=initial_determinant,
            candidate_minimum_determinant=None,
            final_minimum_determinant=initial_determinant,
            directional_derivative=None,
            pcg=pcg,
            preconditioner_identity=resolved_preconditioner_identity,
            config=cfg,
            work=MatrixFreeCorrectionWork(1, 1, 1, 0, pcg.work),
        )

    directional_derivative = float(np.dot(gradient, pcg.solution))
    if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
        return MatrixFreeCorrectionResult(
            x=initial.positions,
            accepted=False,
            reason="non_descent",
            alpha=cfg.alpha,
            initial_objective=initial_objective,
            candidate_objective=None,
            final_objective=initial_objective,
            initial_gradient_norm=gradient_norm,
            candidate_gradient_norm=None,
            final_gradient_norm=gradient_norm,
            initial_minimum_determinant=initial_determinant,
            candidate_minimum_determinant=None,
            final_minimum_determinant=initial_determinant,
            directional_derivative=directional_derivative,
            pcg=pcg,
            preconditioner_identity=resolved_preconditioner_identity,
            config=cfg,
            work=MatrixFreeCorrectionWork(1, 1, 1, 0, pcg.work),
        )

    candidate_free = initial.positions[initial.free].reshape(-1) + cfg.alpha * pcg.solution
    if not np.isfinite(candidate_free).all():
        return MatrixFreeCorrectionResult(
            x=initial.positions,
            accepted=False,
            reason="candidate_nonfinite",
            alpha=cfg.alpha,
            initial_objective=initial_objective,
            candidate_objective=None,
            final_objective=initial_objective,
            initial_gradient_norm=gradient_norm,
            candidate_gradient_norm=None,
            final_gradient_norm=gradient_norm,
            initial_minimum_determinant=initial_determinant,
            candidate_minimum_determinant=None,
            final_minimum_determinant=initial_determinant,
            directional_derivative=directional_derivative,
            pcg=pcg,
            preconditioner_identity=resolved_preconditioner_identity,
            config=cfg,
            work=MatrixFreeCorrectionWork(1, 1, 1, 0, pcg.work),
        )

    candidate_x = initial.positions_from_free(candidate_free)
    try:
        candidate = MatrixFreeStableNHOperator.from_problem(problem, candidate_x)
    except (FloatingPointError, OverflowError, ValueError):
        return MatrixFreeCorrectionResult(
            x=initial.positions,
            accepted=False,
            reason="candidate_nonfinite",
            alpha=cfg.alpha,
            initial_objective=initial_objective,
            candidate_objective=None,
            final_objective=initial_objective,
            initial_gradient_norm=gradient_norm,
            candidate_gradient_norm=None,
            final_gradient_norm=gradient_norm,
            initial_minimum_determinant=initial_determinant,
            candidate_minimum_determinant=None,
            final_minimum_determinant=initial_determinant,
            directional_derivative=directional_derivative,
            pcg=pcg,
            preconditioner_identity=resolved_preconditioner_identity,
            config=cfg,
            work=MatrixFreeCorrectionWork(1, 1, 1, 1, pcg.work),
        )
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_objective = candidate.objective()
        candidate_gradient_norm = float(np.linalg.norm(candidate.gradient_free()))
    candidate_determinant = candidate.minimum_determinant
    if not math.isfinite(candidate_objective) or not math.isfinite(candidate_gradient_norm):
        return MatrixFreeCorrectionResult(
            x=initial.positions,
            accepted=False,
            reason="candidate_nonfinite",
            alpha=cfg.alpha,
            initial_objective=initial_objective,
            candidate_objective=None,
            final_objective=initial_objective,
            initial_gradient_norm=gradient_norm,
            candidate_gradient_norm=None,
            final_gradient_norm=gradient_norm,
            initial_minimum_determinant=initial_determinant,
            candidate_minimum_determinant=None,
            final_minimum_determinant=initial_determinant,
            directional_derivative=directional_derivative,
            pcg=pcg,
            preconditioner_identity=resolved_preconditioner_identity,
            config=cfg,
            work=MatrixFreeCorrectionWork(2, 2, 2, 1, pcg.work),
        )
    try:
        segment_minimum = minimum_determinant_on_segment(initial, candidate)
    except (FloatingPointError, OverflowError, ValueError):
        return MatrixFreeCorrectionResult(
            x=initial.positions,
            accepted=False,
            reason="candidate_nonfinite",
            alpha=cfg.alpha,
            initial_objective=initial_objective,
            candidate_objective=None,
            final_objective=initial_objective,
            initial_gradient_norm=gradient_norm,
            candidate_gradient_norm=None,
            final_gradient_norm=gradient_norm,
            initial_minimum_determinant=initial_determinant,
            candidate_minimum_determinant=None,
            final_minimum_determinant=initial_determinant,
            directional_derivative=directional_derivative,
            pcg=pcg,
            preconditioner_identity=resolved_preconditioner_identity,
            config=cfg,
            work=MatrixFreeCorrectionWork(2, 2, 2, 1, pcg.work),
        )
    common_work = MatrixFreeCorrectionWork(2, 2, 2, 1, pcg.work)
    if segment_minimum.determinant <= cfg.minimum_determinant:
        reason = "segment_inversion"
    else:
        armijo_limit = initial_objective + cfg.armijo * cfg.alpha * directional_derivative
        reason = (
            "accepted"
            if candidate_objective < initial_objective and candidate_objective <= armijo_limit
            else "objective_increase"
        )
    accepted = reason == "accepted"
    return MatrixFreeCorrectionResult(
        x=candidate.positions if accepted else initial.positions,
        accepted=accepted,
        reason=reason,
        alpha=cfg.alpha,
        initial_objective=initial_objective,
        candidate_objective=candidate_objective,
        final_objective=candidate_objective if accepted else initial_objective,
        initial_gradient_norm=gradient_norm,
        candidate_gradient_norm=candidate_gradient_norm,
        final_gradient_norm=candidate_gradient_norm if accepted else gradient_norm,
        initial_minimum_determinant=initial_determinant,
        candidate_minimum_determinant=candidate_determinant,
        final_minimum_determinant=candidate_determinant if accepted else initial_determinant,
        directional_derivative=directional_derivative,
        pcg=pcg,
        preconditioner_identity=resolved_preconditioner_identity,
        config=cfg,
        work=common_work,
        segment_minimum_determinant=segment_minimum.determinant,
        segment_minimum_fraction=segment_minimum.fraction,
    )
