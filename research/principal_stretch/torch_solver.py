# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""PyTorch port of the local-global ARAP-with-target-stretch decoder.

Mirrors :class:`LocalGlobalRecover` but operates on torch tensors so the
decoder can sit inside an autograd graph. Used during training of the
Phase 2 stretch network.

Shapes throughout:
- x:        (V, 3)
- tets:    (T, 4) int64
- Dm_inv:  (T, 3, 3)
- J:       (T, 4, 3)  rows: a=0..3, cols: c=0..2.  F[t,:,c] = sum_a J[t,a,c] * x[tets[t,a],:]
- L:       (V, V) dense for the legacy direct backend
- L_ff:    (F, F) sparse CSR for the scalable PCG backend
- S*:      (T, 3, 3) symmetric
- F*:      (T, 3, 3) full target deformation gradient
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import torch

from .polar import polar_rotation

_DENSE_BACKEND = "dense"
_SPARSE_PCG_BACKEND = "sparse_pcg"
_JACOBI_PRECONDITIONER = "jacobi"
_MULTIGRID_PRECONDITIONER = "multigrid"
_TRANSLATION_GAUGE_NONE = "none"
TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS = "mass-weighted-center-of-mass"
OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE = "canonical-rest-inverse"
OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED = "source-tet-poses-promoted"
_LEGACY_OPERATOR_GEOMETRY_POLICY = "legacy-unverified"
_AUTHENTICATED_OPERATOR_GEOMETRY_POLICIES = (
    OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
)


def static_mesh_sha256(rest_q: np.ndarray, tet_indices: np.ndarray) -> str:
    """Hash canonical rest positions and ordered tetrahedral connectivity."""
    rest = np.ascontiguousarray(np.asarray(rest_q, dtype="<f8"))
    tets = np.ascontiguousarray(np.asarray(tet_indices, dtype="<i8"))
    if rest.ndim != 2 or rest.shape[1] != 3:
        raise ValueError(f"rest_q must have shape (V, 3), got {rest.shape}")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tet_indices must have shape (T, 4), got {tets.shape}")
    digest = hashlib.sha256()
    digest.update(b"principal-stretch-static-mesh-v1\0")
    for name, array in (("rest_q", rest), ("tet_indices", tets)):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _update_source_array_digest(digest, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    raw = memoryview(array).cast("B")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def operator_geometry_sha256(
    rest_q: np.ndarray,
    tet_indices: np.ndarray,
    tet_poses: np.ndarray,
    *,
    policy: str,
) -> str:
    """Hash the exact authenticated source operator geometry.

    The identity is deliberately independent of execution dtype and runtime
    assembly. Those are bound by :func:`projection_state_sha256`. Source
    arrays retain their exact dtype and C-order bytes; no float64
    canonicalization like :func:`static_mesh_sha256` is performed here.
    """
    if type(policy) is not str or policy not in _AUTHENTICATED_OPERATOR_GEOMETRY_POLICIES:
        raise ValueError("operator geometry policy is not an authenticated v5 policy")
    rest = np.ascontiguousarray(np.asarray(rest_q))
    tets = np.ascontiguousarray(np.asarray(tet_indices))
    poses = np.ascontiguousarray(np.asarray(tet_poses))
    _validate_authenticated_source_geometry(rest, tets, poses, policy)
    digest = hashlib.sha256(b"principal-stretch-operator-geometry-v1\0")
    digest.update(policy.encode("ascii") + b"\0")
    for name, array in (("rest_q", rest), ("tet_indices", tets), ("tet_poses", poses)):
        _update_source_array_digest(digest, name, array)
    return digest.hexdigest()


def _require_inverse_backward_error_numpy(source_matrix: np.ndarray, inverse: np.ndarray) -> None:
    """Require a fixed locally floored contribution-scaled inverse residual."""
    identity = np.broadcast_to(np.eye(3, dtype=inverse.dtype), inverse.shape)
    epsilon = np.finfo(inverse.dtype).eps
    gamma = 3.0 * epsilon / (1.0 - 3.0 * epsilon)
    for left, right in ((source_matrix, inverse), (inverse, source_matrix)):
        product = left @ right
        contribution_scale = np.abs(left) @ np.abs(right)
        local_scale = np.max(contribution_scale, axis=(-2, -1), keepdims=True)
        effective_scale = np.maximum(
            contribution_scale,
            np.asarray(epsilon, dtype=inverse.dtype) * local_scale,
        )
        bound = np.asarray(128.0 * gamma, dtype=inverse.dtype) * effective_scale
        if (
            not np.isfinite(product).all()
            or not np.isfinite(contribution_scale).all()
            or not np.isfinite(effective_scale).all()
            or not np.isfinite(bound).all()
            or not np.all(np.abs(product - identity) <= bound)
        ):
            raise ValueError("source tet_poses fail the fixed two-sided backward-error bound")


def _validate_authenticated_source_geometry(
    rest_q: np.ndarray,
    tet_indices: np.ndarray,
    tet_poses: np.ndarray,
    policy: str,
) -> None:
    """Validate exact source arrays under one non-relabelable v5 policy."""
    if policy not in _AUTHENTICATED_OPERATOR_GEOMETRY_POLICIES:
        raise ValueError("operator geometry policy is not an authenticated v5 policy")
    if tet_indices.dtype != np.dtype(np.int64):
        raise ValueError("authenticated operator source tet_indices must have exact int64 dtype")
    expected_float_dtype = (
        np.dtype(np.float64) if policy == OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE else np.dtype(np.float32)
    )
    if rest_q.dtype != expected_float_dtype or tet_poses.dtype != expected_float_dtype:
        raise ValueError(f"{policy} requires exact source rest_q and tet_poses dtype {expected_float_dtype}")
    if rest_q.ndim != 2 or rest_q.shape[1] != 3:
        raise ValueError(f"authenticated source rest_q must have shape (V, 3), got {rest_q.shape}")
    if tet_indices.ndim != 2 or tet_indices.shape[1] != 4:
        raise ValueError(f"authenticated source tet_indices must have shape (T, 4), got {tet_indices.shape}")
    if tet_poses.shape != (tet_indices.shape[0], 3, 3):
        raise ValueError("authenticated source tet_poses must have shape (T, 3, 3)")
    if not np.isfinite(rest_q).all() or not np.isfinite(tet_poses).all():
        raise ValueError("authenticated operator source geometry must be finite")
    if (tet_indices < 0).any() or (tet_indices >= rest_q.shape[0]).any():
        raise ValueError("authenticated source tet_indices contains an out-of-range vertex")
    if any(len(set(row.tolist())) != 4 for row in tet_indices):
        raise ValueError("authenticated source tetrahedra must contain four distinct vertices")

    rest_in_pose_dtype = rest_q
    origin = rest_in_pose_dtype[tet_indices[:, 0]]
    rest_matrix = np.stack(
        (
            rest_in_pose_dtype[tet_indices[:, 1]] - origin,
            rest_in_pose_dtype[tet_indices[:, 2]] - origin,
            rest_in_pose_dtype[tet_indices[:, 3]] - origin,
        ),
        axis=-1,
    )
    rest_det = np.linalg.det(rest_matrix)
    inverse_det = np.linalg.det(tet_poses)
    if (
        not np.isfinite(rest_det).all()
        or not np.isfinite(inverse_det).all()
        or (rest_det <= 0.0).any()
        or (inverse_det <= 0.0).any()
    ):
        raise ValueError("authenticated source tetrahedra must have finite positive orientation")
    source_volume = np.asarray(1.0, dtype=expected_float_dtype) / (
        np.asarray(6.0, dtype=expected_float_dtype) * inverse_det
    )
    if not np.isfinite(source_volume).all() or (source_volume <= 0.0).any():
        raise ValueError("authenticated source tet_poses produce invalid rest volumes")
    _require_inverse_backward_error_numpy(rest_matrix, tet_poses)


def _build_J(Dm_inv: torch.Tensor) -> torch.Tensor:
    """Per-tet shape-function gradient. J[t, a, c] = dF[t, :, c] / dx[i_a, :].

    For a tet with vertices (i0, i1, i2, i3) the deformation gradient is
    F = Ds @ Dm_inv where Ds = [x1 - x0, x2 - x0, x3 - x0].  Substituting:
        F[:, c] = sum_{a=1}^{3} Dm_inv[a-1, c] * (x[i_a] - x[i_0]).
    """
    T = Dm_inv.shape[0]
    J = torch.zeros(T, 4, 3, dtype=Dm_inv.dtype, device=Dm_inv.device)
    J[:, 1, :] = Dm_inv[:, 0, :]
    J[:, 2, :] = Dm_inv[:, 1, :]
    J[:, 3, :] = Dm_inv[:, 2, :]
    J[:, 0, :] = -(J[:, 1, :] + J[:, 2, :] + J[:, 3, :])
    return J


def compute_F(x: torch.Tensor, tets: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Compute deformation gradients, preserving optional batch dimensions."""
    x_tet = x[..., tets, :]  # (..., T, 4, 3)
    return torch.einsum("tac,...tad->...tdc", J, x_tet)


def polar_R(M: torch.Tensor) -> torch.Tensor:
    """Polar rotation ``R`` from ``M``, with an exact backward pass.

    Delegates to :func:`polar.polar_rotation` (Newton forward, analytic
    Sylvester backward).  The previous implementation here detached an SVD and
    re-routed the gradient through ``M inv(sym(R0^T M))``; that surrogate is
    exact only at ``S = I`` and reaches 27% relative Jacobian error at 50%
    stretch.  It is preserved in ``diag_polar_grad.py``, which measures it.
    """
    return polar_rotation(M)


def assemble_rhs(
    R: torch.Tensor, S_target: torch.Tensor, J: torch.Tensor, w: torch.Tensor, tets: torch.Tensor, n_verts: int
) -> torch.Tensor:
    """RHS[i_a, d] += w_e * (R S* @ J[t, a, :])[d]."""
    M = R @ S_target  # (T, 3, 3)
    # contrib[t, a, d] = w[t] * sum_c M[t, d, c] * J[t, a, c]
    contrib = torch.einsum("tdc,tac->tad", M, J) * w[:, None, None]  # (T, 4, 3)
    rhs = torch.zeros(n_verts, 3, dtype=R.dtype, device=R.device)
    rhs.index_add_(0, tets.reshape(-1), contrib.reshape(-1, 3))
    return rhs


@dataclasses.dataclass
class ProjectionDiagnostics:
    """Auditable work and convergence data for a full-gradient projection.

    ``matrix_vector_products`` counts fine Krylov and hierarchy sparse
    matrix-matrix calls, each of which advances all ``rhs_count`` scalar
    right-hand sides.  ``preconditioner_matrix_vector_products`` is the subset
    inside V-cycles.  A dense projection reports one factor solve and no
    iterative work; multigrid reports one coarsest factor solve per V-cycle.
    """

    backend: str
    converged: bool
    iterations: int
    rhs_count: int
    converged_rhs: int
    matrix_vector_products: int
    preconditioner_applications: int
    factor_solves: int
    rhs_norm_max: float
    initial_residual_norm_max: float
    residual_norm_max: float
    relative_residual_max: float
    relative_tolerance: float | None
    absolute_tolerance: float | None
    breakdown: bool = False
    preconditioner: str | None = None
    hierarchy_levels: int = 0
    preconditioner_matrix_vector_products: int = 0

    @property
    def scalar_rhs_matrix_vector_products(self) -> int:
        """Equivalent count if every right-hand side were solved separately."""
        return self.matrix_vector_products * self.rhs_count


@dataclasses.dataclass(frozen=True)
class _MultigridLevel:
    """One fixed Galerkin level in the sparse projection hierarchy."""

    matrix: torch.Tensor
    smoother_inverse: torch.Tensor | None
    aggregate: torch.Tensor | None


@dataclasses.dataclass(frozen=True)
class _MultigridHierarchy:
    """Fixed symmetric V-cycle data suitable for ordinary PCG."""

    levels: tuple[_MultigridLevel, ...]
    coarse_cholesky: torch.Tensor
    smoothing_steps: int


@dataclasses.dataclass
class SolverState:
    """Precomputed mesh + factorisation that does not depend on x or S*."""

    n_verts: int
    n_tets: int
    tets: torch.Tensor  # (T, 4) int64
    Dm_inv: torch.Tensor  # (T, 3, 3)
    J: torch.Tensor  # (T, 4, 3)
    w: torch.Tensor  # (T,) rest volumes
    pinned: torch.Tensor  # (P,) int64
    free: torch.Tensor  # (F,) int64
    L: torch.Tensor | None  # (V, V) dense, direct backend only
    L_ff_chol: torch.Tensor | None  # (F, F) Cholesky, direct backend only
    L_fp: torch.Tensor  # (F, P), dense or sparse CSR according to backend
    rest_q: torch.Tensor  # (V, 3)
    source_rest_q: torch.Tensor  # (V, 3) exact canonical float64 source geometry
    source_rest_q_exact: torch.Tensor  # (V, 3) exact authenticated input; legacy uses canonical placeholder
    source_tet_indices: torch.Tensor  # (T, 4) exact authenticated input; legacy uses canonical placeholder
    source_tet_poses: torch.Tensor  # (T, 3, 3) exact authenticated input; legacy uses runtime placeholder
    static_mesh_sha256: str
    operator_geometry_policy: str = _LEGACY_OPERATOR_GEOMETRY_POLICY
    operator_geometry_sha256: str | None = None
    projection_state_sha256: str = ""
    tikhonov: float = 0.0
    projection_backend: str = _DENSE_BACKEND
    L_ff_sparse: torch.Tensor | None = None  # (F, F) sparse CSR
    L_ff_inverse_diagonal: torch.Tensor | None = None  # (F,) Jacobi preconditioner
    pcg_relative_tolerance: float = 1.0e-8
    pcg_absolute_tolerance: float = 0.0
    pcg_max_iterations: int = 512
    pcg_raise_on_nonconvergence: bool = True
    pcg_preconditioner: str = _JACOBI_PRECONDITIONER
    multigrid_hierarchy: _MultigridHierarchy | None = None
    translation_gauge_policy: str = _TRANSLATION_GAUGE_NONE
    center_of_mass_weights: torch.Tensor | None = None  # (V,), normalized


def _update_projection_tensor_digest(
    digest,
    name: str,
    value: torch.Tensor | None,
) -> None:
    digest.update(name.encode("utf-8") + b"\0")
    if value is None:
        digest.update(b"none\0")
        return
    metadata = {
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "shape": list(value.shape),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\0")
    if value.layout == torch.strided:
        # Empty tensors can retain a zero last stride after ``contiguous``;
        # viewing them as bytes then fails even though their payload is empty.
        raw = b"" if value.numel() == 0 else value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if value.layout == torch.sparse_csr:
        _update_projection_tensor_digest(digest, f"{name}.crow_indices", value.crow_indices())
        _update_projection_tensor_digest(digest, f"{name}.col_indices", value.col_indices())
        _update_projection_tensor_digest(digest, f"{name}.values", value.values())
        return
    raise ValueError(f"projection-state tensor {name} has unsupported layout {value.layout}")


def projection_state_sha256(state: SolverState) -> str:
    """Hash every tensor and policy that defines compatibility projection."""
    if not isinstance(state, SolverState):
        raise TypeError("state must be a SolverState")
    if state.translation_gauge_policy not in (
        _TRANSLATION_GAUGE_NONE,
        TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
    ):
        raise ValueError("projection state has an unsupported translation gauge policy")
    has_center_of_mass_gauge = state.translation_gauge_policy == TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS
    if has_center_of_mass_gauge != (state.center_of_mass_weights is not None):
        raise ValueError("projection state translation gauge policy and mass weights disagree")
    authenticated_operator = state.operator_geometry_policy in _AUTHENTICATED_OPERATOR_GEOMETRY_POLICIES
    if has_center_of_mass_gauge:
        digest = hashlib.sha256(b"principal-stretch-projection-state-v4\0")
    else:
        digest = hashlib.sha256(
            b"principal-stretch-projection-state-v3\0"
            if authenticated_operator
            else b"principal-stretch-projection-state-v2\0"
        )
    metadata = {
        "n_verts": state.n_verts,
        "n_tets": state.n_tets,
        "static_mesh_sha256": state.static_mesh_sha256,
        "tikhonov": float(state.tikhonov).hex(),
        "projection_backend": state.projection_backend,
        "pcg_relative_tolerance": float(state.pcg_relative_tolerance).hex(),
        "pcg_absolute_tolerance": float(state.pcg_absolute_tolerance).hex(),
        "pcg_max_iterations": state.pcg_max_iterations,
        "pcg_raise_on_nonconvergence": state.pcg_raise_on_nonconvergence,
        "pcg_preconditioner": state.pcg_preconditioner,
    }
    if authenticated_operator:
        metadata.update(
            {
                "operator_geometry_policy": state.operator_geometry_policy,
                "operator_geometry_sha256": state.operator_geometry_sha256,
            }
        )
    if has_center_of_mass_gauge:
        metadata["translation_gauge_policy"] = state.translation_gauge_policy
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for name in (
        "tets",
        "Dm_inv",
        "J",
        "w",
        "pinned",
        "free",
        "L",
        "L_ff_chol",
        "L_fp",
        "rest_q",
        "source_rest_q",
        "L_ff_sparse",
        "L_ff_inverse_diagonal",
    ):
        _update_projection_tensor_digest(digest, name, getattr(state, name))
    if authenticated_operator:
        for name in ("source_rest_q_exact", "source_tet_indices", "source_tet_poses"):
            _update_projection_tensor_digest(digest, name, getattr(state, name))
    if has_center_of_mass_gauge:
        _update_projection_tensor_digest(digest, "center_of_mass_weights", state.center_of_mass_weights)
    hierarchy = state.multigrid_hierarchy
    if hierarchy is None:
        digest.update(b"multigrid_hierarchy\0none\0")
    else:
        digest.update(
            json.dumps(
                {"multigrid_levels": len(hierarchy.levels), "smoothing_steps": hierarchy.smoothing_steps},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for level_index, level in enumerate(hierarchy.levels):
            _update_projection_tensor_digest(digest, f"multigrid.{level_index}.matrix", level.matrix)
            _update_projection_tensor_digest(
                digest,
                f"multigrid.{level_index}.smoother_inverse",
                level.smoother_inverse,
            )
            _update_projection_tensor_digest(digest, f"multigrid.{level_index}.aggregate", level.aggregate)
        _update_projection_tensor_digest(digest, "multigrid.coarse_cholesky", hierarchy.coarse_cholesky)
    return digest.hexdigest()


def validate_authenticated_operator_geometry(state: SolverState) -> str:
    """Reauthenticate a v5 source operator and its exact runtime promotion."""
    if type(state) is not SolverState:
        raise TypeError("state must be a canonical SolverState")
    policy = state.operator_geometry_policy
    if type(policy) is not str or policy not in _AUTHENTICATED_OPERATOR_GEOMETRY_POLICIES:
        raise ValueError("solver state has no authenticated v5 operator geometry")
    for name in ("source_rest_q_exact", "source_tet_indices", "source_tet_poses"):
        value = getattr(state, name)
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device != state.rest_q.device:
            raise ValueError(f"solver state {name} has incompatible tensor metadata")
        if value.requires_grad:
            raise ValueError(f"solver state {name} must not require gradients")
    rest_source = state.source_rest_q_exact.detach().contiguous().cpu().numpy()
    tet_source = state.source_tet_indices.detach().contiguous().cpu().numpy()
    pose_source = state.source_tet_poses.detach().contiguous().cpu().numpy()
    expected_sha256 = operator_geometry_sha256(rest_source, tet_source, pose_source, policy=policy)
    if state.operator_geometry_sha256 != expected_sha256:
        raise ValueError("solver state operator_geometry_sha256 verification failed")
    if state.static_mesh_sha256 != static_mesh_sha256(rest_source, tet_source):
        raise ValueError("solver state static mesh differs from authenticated source geometry")
    if state.source_rest_q.dtype != torch.float64 or not torch.equal(
        state.source_rest_q, state.source_rest_q_exact.to(dtype=torch.float64)
    ):
        raise ValueError("solver state canonical source_rest_q differs from exact source geometry")
    if state.tets.dtype != torch.int64 or not torch.equal(state.tets, state.source_tet_indices.to(dtype=torch.int64)):
        raise ValueError("solver state runtime tetrahedra differ from exact source connectivity")
    if not state.rest_q.is_floating_point() or state.rest_q.dtype not in (torch.float32, torch.float64):
        raise ValueError("solver state runtime geometry must use float32 or float64")
    if policy == OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED and state.rest_q.dtype != torch.float64:
        raise ValueError("source-tet-poses-promoted requires float64 execution")
    if not torch.equal(state.rest_q, state.source_rest_q_exact.to(dtype=state.rest_q.dtype)):
        raise ValueError("solver state runtime rest_q is not the exact contracted source cast")
    expected_dm_inv = state.source_tet_poses.to(dtype=state.rest_q.dtype)
    if not torch.equal(state.Dm_inv, expected_dm_inv):
        raise ValueError("solver state Dm_inv is not the exact contracted source-pose cast")
    expected_j = _build_J(expected_dm_inv)
    expected_det = torch.linalg.det(expected_dm_inv)
    expected_volume = 1.0 / (6.0 * expected_det)
    if (
        not torch.isfinite(expected_det).all()
        or not torch.isfinite(expected_volume).all()
        or (expected_det <= 0.0).any()
        or (expected_volume <= 0.0).any()
    ):
        raise ValueError("solver state runtime source operator has invalid orientation or volume")
    if not torch.equal(state.J, expected_j) or not torch.equal(state.w, expected_volume):
        raise ValueError("solver state J or volume is not the exact source-pose-derived runtime operator")
    return expected_sha256


def _validate_pcg_options(relative_tolerance: float, absolute_tolerance: float, max_iterations: int) -> None:
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("pcg_relative_tolerance must be finite and non-negative")
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("pcg_absolute_tolerance must be finite and non-negative")
    if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
        raise ValueError("at least one PCG tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("pcg_max_iterations must be a positive integer")


def _validate_multigrid_options(
    preconditioner: str,
    coarse_size: int,
    max_levels: int,
    smoothing_steps: int,
    smoother_damping: float,
) -> None:
    if preconditioner not in (_JACOBI_PRECONDITIONER, _MULTIGRID_PRECONDITIONER):
        raise ValueError(f"pcg_preconditioner must be '{_JACOBI_PRECONDITIONER}' or '{_MULTIGRID_PRECONDITIONER}'")
    for name, value in (
        ("multigrid_coarse_size", coarse_size),
        ("multigrid_max_levels", max_levels),
        ("multigrid_smoothing_steps", smoothing_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(smoother_damping) or not 0.0 < smoother_damping < 1.0:
        raise ValueError("multigrid_smoother_damping must be finite and lie strictly between zero and one")


def _csr_numpy(matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy a CSR matrix into canonical host arrays for deterministic setup."""
    if matrix.layout != torch.sparse_csr:
        raise ValueError("multigrid setup requires a CSR matrix")
    crow = matrix.crow_indices().detach().cpu().numpy().astype(np.int64, copy=True)
    columns = matrix.col_indices().detach().cpu().numpy().astype(np.int64, copy=True)
    values = matrix.values().detach().cpu().numpy().copy()
    return crow, columns, values


def _csr_rows(crow: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(crow.size - 1, dtype=np.int64), np.diff(crow))


def _symmetrize_csr_values(
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Validate a square sparsity pattern and average transpose pairs."""
    if rows.size != columns.size or rows.size != values.size:
        raise ValueError("invalid CSR arrays in multigrid setup")
    keys = rows * n_rows + columns
    if keys.size and (np.diff(keys) <= 0).any():
        raise ValueError("multigrid setup requires canonical sorted CSR indices")
    transpose_keys = columns * n_rows + rows
    transpose_positions = np.searchsorted(keys, transpose_keys)
    if (transpose_positions >= keys.size).any() or not np.array_equal(keys[transpose_positions], transpose_keys):
        raise ValueError("multigrid Galerkin matrix must have a symmetric sparsity pattern")
    transpose_values = values[transpose_positions]
    scale = max(float(np.abs(values).max(initial=0.0)), 1.0)
    tolerance = 256.0 * np.finfo(values.dtype).eps * scale
    if not np.isfinite(values).all() or float(np.abs(values - transpose_values).max(initial=0.0)) > tolerance:
        raise ValueError("multigrid Galerkin matrix must be finite and symmetric")
    # Both entries perform the same pairwise addition, so transpose pairs become
    # bitwise equal.  This removes assembly roundoff before the matrix enters PCG.
    return 0.5 * (values + transpose_values)


def _csr_diagonal(crow: np.ndarray, columns: np.ndarray, values: np.ndarray) -> np.ndarray:
    n_rows = crow.size - 1
    rows = _csr_rows(crow)
    diagonal_mask = rows == columns
    diagonal = np.zeros(n_rows, dtype=values.dtype)
    diagonal[rows[diagonal_mask]] = values[diagonal_mask]
    if not np.isfinite(diagonal).all() or (diagonal <= 0.0).any():
        raise ValueError("multigrid Galerkin matrix must have a finite positive diagonal")
    return diagonal


def _deterministic_heavy_edge_aggregation(
    crow: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    diagonal: np.ndarray,
) -> np.ndarray:
    """Greedily pair graph neighbors, with normalized coupling and ID tie-breaks."""
    n_rows = crow.size - 1
    aggregate = np.full(n_rows, -1, dtype=np.int64)
    next_aggregate = 0
    for row in range(n_rows):
        if aggregate[row] >= 0:
            continue
        best_column = -1
        best_coupling = -1.0
        for offset in range(int(crow[row]), int(crow[row + 1])):
            column = int(columns[offset])
            if column == row or aggregate[column] >= 0 or values[offset] == 0.0:
                continue
            coupling = abs(float(values[offset])) / float(np.sqrt(diagonal[row] * diagonal[column]))
            if coupling > best_coupling or (coupling == best_coupling and column < best_column):
                best_coupling = coupling
                best_column = column
        aggregate[row] = next_aggregate
        if best_column >= 0:
            aggregate[best_column] = next_aggregate
        next_aggregate += 1
    return aggregate


def _galerkin_coarse_csr(
    crow: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    aggregate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form ``P.T @ A @ P`` for unsmoothed aggregate prolongation."""
    rows = _csr_rows(crow)
    n_coarse = int(aggregate.max(initial=-1)) + 1
    coarse_keys = aggregate[rows] * n_coarse + aggregate[columns]
    order = np.argsort(coarse_keys, kind="stable")
    sorted_keys = coarse_keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    keys = sorted_keys[starts]
    coarse_values = np.add.reduceat(values[order], starts)
    coarse_rows = keys // n_coarse
    coarse_columns = keys % n_coarse
    coarse_values = _symmetrize_csr_values(coarse_rows, coarse_columns, coarse_values, n_coarse)
    # Exact cancellation can leave explicit zero off-diagonals.  Drop them, but
    # retain every diagonal so hierarchy setup remains auditable and fail-closed.
    keep = (coarse_values != 0.0) | (coarse_rows == coarse_columns)
    coarse_rows = coarse_rows[keep]
    coarse_columns = coarse_columns[keep]
    coarse_values = coarse_values[keep]
    counts = np.bincount(coarse_rows, minlength=n_coarse)
    coarse_crow = np.empty(n_coarse + 1, dtype=np.int64)
    coarse_crow[0] = 0
    np.cumsum(counts, out=coarse_crow[1:])
    return coarse_crow, coarse_columns.astype(np.int64, copy=False), coarse_values


def _csr_from_numpy(
    crow: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_rows = crow.size - 1
    return torch.sparse_csr_tensor(
        torch.as_tensor(crow, dtype=torch.int64, device=device),
        torch.as_tensor(columns, dtype=torch.int64, device=device),
        torch.as_tensor(values, dtype=dtype, device=device),
        size=(n_rows, n_rows),
        dtype=dtype,
        device=device,
    )


def _build_multigrid_hierarchy(
    matrix: torch.Tensor,
    *,
    coarse_size: int,
    max_levels: int,
    smoothing_steps: int,
    smoother_damping: float,
) -> _MultigridHierarchy:
    """Build a deterministic heavy-edge/Galerkin hierarchy and coarse factor."""
    device = matrix.device
    dtype = matrix.dtype
    crow, columns, values = _csr_numpy(matrix)
    rows = _csr_rows(crow)
    values = _symmetrize_csr_values(rows, columns, values, crow.size - 1)
    levels: list[_MultigridLevel] = []
    current_matrix = _csr_from_numpy(crow, columns, values, device, dtype)

    for _level_index in range(max_levels):
        n_rows = crow.size - 1
        diagonal = _csr_diagonal(crow, columns, values)
        if n_rows <= coarse_size:
            coarse_dense = current_matrix.to_dense()
            try:
                coarse_cholesky = torch.linalg.cholesky(coarse_dense)
            except RuntimeError as exc:
                raise ValueError("multigrid coarsest Galerkin matrix is not positive definite") from exc
            levels.append(_MultigridLevel(current_matrix, None, None))
            return _MultigridHierarchy(tuple(levels), coarse_cholesky, smoothing_steps)

        row_absolute_sum = np.add.reduceat(np.abs(values), crow[:-1])
        if not np.isfinite(row_absolute_sum).all() or (row_absolute_sum <= 0.0).any():
            raise ValueError("multigrid smoother requires finite nonzero rows")
        # With B = damping * diag(1 / ||A_i||_1) and damping < 1,
        # 2 B^-1 - A is symmetric strictly diagonally dominant and positive.
        # Mirrored stationary sweeps therefore produce an SPD smoother.
        smoother_inverse = torch.as_tensor(
            smoother_damping / row_absolute_sum,
            dtype=dtype,
            device=device,
        )
        aggregate = _deterministic_heavy_edge_aggregation(crow, columns, values, diagonal)
        n_coarse = int(aggregate.max(initial=-1)) + 1
        if n_coarse <= 0 or n_coarse >= n_rows:
            raise ValueError("multigrid aggregation did not reduce the free-vertex graph")
        aggregate_tensor = torch.as_tensor(aggregate, dtype=torch.int64, device=device)
        levels.append(_MultigridLevel(current_matrix, smoother_inverse, aggregate_tensor))
        # Every fine row belongs to one nonempty aggregate, so the piecewise-
        # constant P has full column rank.  Thus P.T @ A @ P remains SPD.
        crow, columns, values = _galerkin_coarse_csr(crow, columns, values, aggregate)
        current_matrix = _csr_from_numpy(crow, columns, values, device, dtype)

    raise ValueError(f"multigrid hierarchy did not reach coarse_size={coarse_size} within max_levels={max_levels}")


def _validate_sparse_components(tet_indices: np.ndarray, pinned_indices: np.ndarray, n_verts: int) -> None:
    """Require a Dirichlet anchor in every connected vertex component."""
    parents = np.arange(n_verts, dtype=np.int64)

    def find(vertex: int) -> int:
        root = vertex
        while parents[root] != root:
            root = int(parents[root])
        while parents[vertex] != vertex:
            next_vertex = int(parents[vertex])
            parents[vertex] = root
            vertex = next_vertex
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    used = np.zeros(n_verts, dtype=bool)
    for tet in np.asarray(tet_indices, dtype=np.int64):
        used[tet] = True
        union(int(tet[0]), int(tet[1]))
        union(int(tet[0]), int(tet[2]))
        union(int(tet[0]), int(tet[3]))

    pinned_set = {int(vertex) for vertex in np.asarray(pinned_indices, dtype=np.int64)}
    anchored_roots = {find(vertex) for vertex in pinned_set}
    missing = sorted({find(vertex) for vertex in np.flatnonzero(used)} - anchored_roots)
    unused_free = [vertex for vertex in np.flatnonzero(~used) if vertex not in pinned_set]
    if missing or unused_free:
        raise ValueError("sparse projection requires every connected component and unused vertex to be pinned")


def _validate_center_of_mass_gauge_component(tet_indices: np.ndarray, n_verts: int) -> None:
    """Require the one translation mode fixed by a single center-of-mass gauge."""
    tets = np.asarray(tet_indices, dtype=np.int64)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"tet_indices must have shape (T, 4), got {tets.shape}")
    if n_verts <= 0 or tets.shape[0] == 0:
        raise ValueError("center-of-mass gauge requires one connected component with no unused vertices")
    if (tets < 0).any() or (tets >= n_verts).any():
        raise ValueError("tet_indices contains an out-of-range vertex")

    parents = np.arange(n_verts, dtype=np.int64)

    def find(vertex: int) -> int:
        root = vertex
        while parents[root] != root:
            root = int(parents[root])
        while parents[vertex] != vertex:
            next_vertex = int(parents[vertex])
            parents[vertex] = root
            vertex = next_vertex
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    used = np.zeros(n_verts, dtype=bool)
    for tet in tets:
        used[tet] = True
        union(int(tet[0]), int(tet[1]))
        union(int(tet[0]), int(tet[2]))
        union(int(tet[0]), int(tet[3]))
    roots = {find(int(vertex)) for vertex in np.flatnonzero(used)}
    if not used.all() or len(roots) != 1:
        raise ValueError("center-of-mass gauge requires one connected component with no unused vertices")


def _assemble_sparse_reduced_system(
    K: torch.Tensor,
    tets: torch.Tensor,
    free: torch.Tensor,
    pinned: torch.Tensor,
    n_verts: int,
    tikhonov: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble ``L_ff``, ``L_fp``, and inverse Jacobi diagonal in O(T) storage."""
    device = K.device
    dtype = K.dtype
    n_free = free.numel()
    n_pinned = pinned.numel()

    free_index = torch.full((n_verts,), -1, dtype=torch.int64, device=device)
    free_index[free] = torch.arange(n_free, dtype=torch.int64, device=device)
    pinned_index = torch.full((n_verts,), -1, dtype=torch.int64, device=device)
    pinned_index[pinned] = torch.arange(n_pinned, dtype=torch.int64, device=device)

    row_vertices = tets[:, :, None].expand(-1, -1, 4)
    col_vertices = tets[:, None, :].expand(-1, 4, -1)
    free_rows = free_index[row_vertices]
    free_cols = free_index[col_vertices]
    pinned_cols = pinned_index[col_vertices]

    ff_mask = (free_rows >= 0) & (free_cols >= 0)
    ff_indices = torch.stack((free_rows[ff_mask], free_cols[ff_mask]))
    ff_values = K[ff_mask]
    if tikhonov > 0.0:
        diagonal = torch.arange(n_free, dtype=torch.int64, device=device)
        ff_indices = torch.cat((ff_indices, torch.stack((diagonal, diagonal))), dim=1)
        ff_values = torch.cat((ff_values, torch.full((n_free,), tikhonov, dtype=dtype, device=device)))
    L_ff = torch.sparse_coo_tensor(
        ff_indices,
        ff_values,
        size=(n_free, n_free),
        dtype=dtype,
        device=device,
    ).coalesce()

    coalesced_indices = L_ff.indices()
    coalesced_values = L_ff.values()
    diagonal_mask = coalesced_indices[0] == coalesced_indices[1]
    diagonal = torch.zeros(n_free, dtype=dtype, device=device)
    diagonal.index_add_(0, coalesced_indices[0, diagonal_mask], coalesced_values[diagonal_mask])
    if n_free and (not torch.isfinite(diagonal).all() or (diagonal <= 0.0).any()):
        raise ValueError("sparse projection matrix must have a finite positive diagonal")

    fp_mask = (free_rows >= 0) & (pinned_cols >= 0)
    fp_indices = torch.stack((free_rows[fp_mask], pinned_cols[fp_mask]))
    fp_values = K[fp_mask]
    L_fp = torch.sparse_coo_tensor(
        fp_indices,
        fp_values,
        size=(n_free, n_pinned),
        dtype=dtype,
        device=device,
    ).coalesce()
    return L_ff.to_sparse_csr(), L_fp.to_sparse_csr(), diagonal.reciprocal()


def build_solver(
    rest_q: np.ndarray,
    tet_indices: np.ndarray,
    tet_poses: np.ndarray,
    pinned_indices: np.ndarray,
    device: torch.device,
    dtype=torch.float64,
    tikhonov: float = 0.0,
    projection_backend: str = _DENSE_BACKEND,
    pcg_relative_tolerance: float = 1.0e-8,
    pcg_absolute_tolerance: float = 0.0,
    pcg_max_iterations: int = 512,
    pcg_raise_on_nonconvergence: bool = True,
    pcg_preconditioner: str = _JACOBI_PRECONDITIONER,
    multigrid_coarse_size: int = 256,
    multigrid_max_levels: int = 12,
    multigrid_smoothing_steps: int = 1,
    multigrid_smoother_damping: float = 0.8,
    operator_geometry_policy: str | None = None,
    translation_gauge_policy: str | None = None,
    vertex_masses: np.ndarray | None = None,
) -> SolverState:
    r"""Build the fixed linear system used by the stretch decoder.

    ``projection_backend="dense"`` preserves the original Cholesky-backed
    state and remains the default.  ``"sparse_pcg"`` never allocates a dense
    vertex-by-vertex matrix: it assembles reduced CSR matrices with at most
    O(T) entries and either a Jacobi or fixed symmetric multigrid
    preconditioner.  Multigrid is an inference-only path until it has an
    implicit-adjoint backward.  Sparse states support
    :func:`project_deformation_gradient`, but not the legacy local-global
    :func:`solve` routine.

    Args:
        rest_q: Rest vertex positions [m], shape ``[V, 3]``.
        tet_indices: Tetrahedron vertex indices, shape ``[T, 4]``.
        tet_poses: Inverse rest matrices [1/m], shape ``[T, 3, 3]``.
        pinned_indices: Dirichlet vertex indices, shape ``[P]``.
        device: Torch device on which to assemble the state.
        dtype: Floating-point dtype for the solve.
        tikhonov: Diagonal regularization. Exact full-gradient projection
            requires zero regularization.
        projection_backend: ``"dense"`` or ``"sparse_pcg"``.
        pcg_relative_tolerance: Relative residual tolerance for sparse PCG.
        pcg_absolute_tolerance: Absolute residual tolerance for sparse PCG.
        pcg_max_iterations: Maximum sparse PCG iterations.
        pcg_raise_on_nonconvergence: Whether sparse projection fails closed
            when the requested tolerance is not reached.
        pcg_preconditioner: ``"jacobi"`` (the compatible default) or
            ``"multigrid"``.
        multigrid_coarse_size: Maximum vertex count for the prefactored exact
            coarsest solve.
        multigrid_max_levels: Maximum number of hierarchy levels, including
            the coarsest level.
        multigrid_smoothing_steps: Symmetric pre/post L1-Jacobi sweeps per
            non-coarse level and V-cycle.
        multigrid_smoother_damping: L1-Jacobi damping in ``(0, 1)``.
        operator_geometry_policy: Explicit authenticated v5 source-operator
            policy, or ``None`` for the backward-compatible but deliberately
            unauthenticated legacy path. V5 consumers reject legacy states.
        translation_gauge_policy: Optional translation gauge for an unpinned
            full-gradient projection. The only supported policy is
            ``"mass-weighted-center-of-mass"``. ``None`` preserves the
            physical-pin-only factorization.
        vertex_masses: Non-negative vertex masses [kg], shape ``[V]``. Required
            by the mass-weighted center-of-mass gauge and otherwise rejected.

    Returns:
        Precomputed solver state.
    """
    if not np.isfinite(tikhonov) or tikhonov < 0.0:
        raise ValueError("tikhonov must be finite and non-negative")
    if projection_backend not in (_DENSE_BACKEND, _SPARSE_PCG_BACKEND):
        raise ValueError(f"projection_backend must be '{_DENSE_BACKEND}' or '{_SPARSE_PCG_BACKEND}'")
    if projection_backend == _SPARSE_PCG_BACKEND:
        _validate_pcg_options(pcg_relative_tolerance, pcg_absolute_tolerance, pcg_max_iterations)
        _validate_multigrid_options(
            pcg_preconditioner,
            multigrid_coarse_size,
            multigrid_max_levels,
            multigrid_smoothing_steps,
            multigrid_smoother_damping,
        )
    elif pcg_preconditioner != _JACOBI_PRECONDITIONER:
        raise ValueError("pcg_preconditioner applies only to projection_backend='sparse_pcg'")
    if translation_gauge_policy is None:
        canonical_translation_gauge_policy = _TRANSLATION_GAUGE_NONE
    elif (
        type(translation_gauge_policy) is not str
        or translation_gauge_policy != TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS
    ):
        raise ValueError("translation_gauge_policy must be 'mass-weighted-center-of-mass' or be omitted")
    else:
        canonical_translation_gauge_policy = translation_gauge_policy
    if canonical_translation_gauge_policy == _TRANSLATION_GAUGE_NONE and vertex_masses is not None:
        raise ValueError("vertex_masses requires an explicit translation_gauge_policy")
    if operator_geometry_policy is None:
        canonical_operator_policy = _LEGACY_OPERATOR_GEOMETRY_POLICY
    elif type(operator_geometry_policy) is not str or operator_geometry_policy not in (
        OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    ):
        raise ValueError("operator_geometry_policy must name an explicit authenticated v5 policy or be omitted")
    else:
        canonical_operator_policy = operator_geometry_policy

    source_rest_array_exact = np.ascontiguousarray(np.asarray(rest_q))
    source_tet_indices_array = np.ascontiguousarray(np.asarray(tet_indices))
    source_tet_poses_array = np.ascontiguousarray(np.asarray(tet_poses))
    authenticated_operator_sha256 = None
    if canonical_operator_policy in _AUTHENTICATED_OPERATOR_GEOMETRY_POLICIES:
        if canonical_operator_policy == OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED and dtype != torch.float64:
            raise ValueError("source-tet-poses-promoted requires torch.float64 execution")
        authenticated_operator_sha256 = operator_geometry_sha256(
            source_rest_array_exact,
            source_tet_indices_array,
            source_tet_poses_array,
            policy=canonical_operator_policy,
        )

    source_rest_q = torch.as_tensor(np.asarray(rest_q, dtype=np.float64), dtype=torch.float64, device=device).clone()
    if authenticated_operator_sha256 is None:
        # These unbound compatibility placeholders deliberately preserve the
        # legacy conversion surface, including inputs whose raw NumPy dtype
        # has no Torch equivalent. V5 consumers reject this policy.
        source_rest_q_exact = source_rest_q.clone()
        source_tet_indices = torch.as_tensor(tet_indices, dtype=torch.int64, device=device).clone()
        source_tet_poses = torch.as_tensor(tet_poses, dtype=dtype, device=device).clone()
    else:
        source_rest_q_exact = torch.as_tensor(source_rest_array_exact, device=device).clone()
        source_tet_indices = torch.as_tensor(source_tet_indices_array, device=device).clone()
        source_tet_poses = torch.as_tensor(source_tet_poses_array, device=device).clone()
    rest_q_t = source_rest_q.to(dtype=dtype).clone()
    tets = torch.as_tensor(tet_indices, dtype=torch.int64, device=device)
    Dm_inv = (
        torch.as_tensor(tet_poses, dtype=dtype, device=device)
        if authenticated_operator_sha256 is None
        else source_tet_poses.to(dtype=dtype).clone()
    )
    pinned = torch.as_tensor(pinned_indices, dtype=torch.int64, device=device)

    n_verts = rest_q_t.shape[0]
    n_tets = tets.shape[0]

    center_of_mass_weights = None
    if canonical_translation_gauge_policy == TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS:
        if projection_backend != _DENSE_BACKEND:
            raise ValueError("mass-weighted center-of-mass gauge currently requires projection_backend='dense'")
        if tikhonov != 0.0:
            raise ValueError("mass-weighted center-of-mass gauge requires zero tikhonov")
        pinned_np = np.asarray(pinned_indices, dtype=np.int64)
        if pinned_np.ndim != 1:
            raise ValueError("pinned_indices must be one-dimensional")
        if pinned_np.size != 0:
            raise ValueError("center-of-mass gauge is separate from physical pins and requires no physical pins")
        if vertex_masses is None:
            raise ValueError("mass-weighted center-of-mass gauge requires vertex_masses")
        mass_array = np.asarray(vertex_masses, dtype=np.float64)
        if mass_array.shape != (n_verts,):
            raise ValueError(f"vertex_masses must have shape ({n_verts},), got {mass_array.shape}")
        if not np.isfinite(mass_array).all():
            raise ValueError("vertex_masses must be finite")
        if (mass_array < 0.0).any():
            raise ValueError("vertex_masses must be non-negative")
        if float(mass_array.sum(dtype=np.float64)) <= 0.0:
            raise ValueError("vertex_masses must have positive total mass")
        _validate_center_of_mass_gauge_component(np.asarray(tet_indices), n_verts)
        mass_tensor = torch.as_tensor(mass_array, dtype=dtype, device=device)
        execution_mass = mass_tensor.sum()
        if (
            not bool(torch.isfinite(mass_tensor).all().item())
            or not bool(torch.isfinite(execution_mass).item())
            or not bool((execution_mass > 0.0).item())
        ):
            raise ValueError("vertex_masses must remain finite in the solver execution dtype")
        center_of_mass_weights = (mass_tensor / execution_mass).clone()

    if projection_backend == _SPARSE_PCG_BACKEND:
        pinned_np = np.asarray(pinned_indices, dtype=np.int64)
        if pinned_np.ndim != 1 or len(np.unique(pinned_np)) != len(pinned_np):
            raise ValueError("sparse projection requires unique one-dimensional pinned_indices")
        if (pinned_np < 0).any() or (pinned_np >= n_verts).any():
            raise ValueError("pinned_indices contains an out-of-range vertex")
        _validate_sparse_components(np.asarray(tet_indices), pinned_np, n_verts)

    det_inv = torch.linalg.det(Dm_inv)
    w = 1.0 / (6.0 * det_inv)
    invalid_volume = (w <= 0).any()
    if authenticated_operator_sha256 is not None:
        invalid_volume = (
            invalid_volume | ~torch.isfinite(det_inv).all() | ~torch.isfinite(w).all() | (det_inv <= 0.0).any()
        )
    if invalid_volume:
        raise ValueError("non-positive rest volumes — check tet orientation")

    J = _build_J(Dm_inv)  # (T, 4, 3)

    # K[t, a, b] = w[t] * sum_c J[t, a, c] * J[t, b, c]
    K = torch.einsum("tac,tbc->tab", J, J) * w[:, None, None]  # (T, 4, 4)
    mask = torch.ones(n_verts, dtype=torch.bool, device=device)
    mask[pinned] = False
    free = torch.where(mask)[0]

    L = None
    L_ff_chol = None
    L_ff_sparse = None
    L_ff_inverse_diagonal = None
    multigrid_hierarchy = None
    if projection_backend == _DENSE_BACKEND:
        # Dense assembly of L: L = sum_e w_e * (J_e @ J_e^T) scattered.
        L = torch.zeros(n_verts, n_verts, dtype=dtype, device=device)
        rows = tets[:, :, None].expand(-1, -1, 4)  # (T, 4, 4)
        cols = tets[:, None, :].expand(-1, 4, -1)
        L.index_put_((rows.reshape(-1), cols.reshape(-1)), K.reshape(-1), accumulate=True)

        L_ff = L[free][:, free]
        if center_of_mass_weights is not None:
            L_ff = L_ff + center_of_mass_weights[:, None] * center_of_mass_weights[None, :]
        if tikhonov > 0.0:
            L_ff = L_ff + tikhonov * torch.eye(free.numel(), dtype=dtype, device=device)
        L_fp = L[free][:, pinned]
        L_ff_chol = torch.linalg.cholesky(L_ff)
    else:
        L_ff_sparse, L_fp, L_ff_inverse_diagonal = _assemble_sparse_reduced_system(
            K,
            tets,
            free,
            pinned,
            n_verts,
            tikhonov,
        )
        if pcg_preconditioner == _MULTIGRID_PRECONDITIONER:
            multigrid_hierarchy = _build_multigrid_hierarchy(
                L_ff_sparse,
                coarse_size=multigrid_coarse_size,
                max_levels=multigrid_max_levels,
                smoothing_steps=multigrid_smoothing_steps,
                smoother_damping=multigrid_smoother_damping,
            )
            # PCG must use the same bitwise-symmetric root operator that was
            # validated and used to build the Galerkin hierarchy.  The raw
            # GPU coalescing path is mathematically symmetric but may differ
            # between transpose entries by reduction roundoff on other
            # devices, which is not a sufficient contract for ordinary PCG.
            L_ff_sparse = multigrid_hierarchy.levels[0].matrix
            root_crow, root_columns, root_values = _csr_numpy(L_ff_sparse)
            root_diagonal = _csr_diagonal(root_crow, root_columns, root_values)
            L_ff_inverse_diagonal = torch.as_tensor(root_diagonal, dtype=dtype, device=device).reciprocal()

    state = SolverState(
        n_verts=n_verts,
        n_tets=n_tets,
        tets=tets,
        Dm_inv=Dm_inv,
        J=J,
        w=w,
        pinned=pinned,
        free=free,
        L=L,
        L_ff_chol=L_ff_chol,
        L_fp=L_fp,
        rest_q=rest_q_t,
        source_rest_q=source_rest_q,
        source_rest_q_exact=source_rest_q_exact,
        source_tet_indices=source_tet_indices,
        source_tet_poses=source_tet_poses,
        static_mesh_sha256=static_mesh_sha256(rest_q, tet_indices),
        operator_geometry_policy=canonical_operator_policy,
        operator_geometry_sha256=authenticated_operator_sha256,
        tikhonov=float(tikhonov),
        projection_backend=projection_backend,
        L_ff_sparse=L_ff_sparse,
        L_ff_inverse_diagonal=L_ff_inverse_diagonal,
        pcg_relative_tolerance=float(pcg_relative_tolerance),
        pcg_absolute_tolerance=float(pcg_absolute_tolerance),
        pcg_max_iterations=pcg_max_iterations,
        pcg_raise_on_nonconvergence=bool(pcg_raise_on_nonconvergence),
        pcg_preconditioner=pcg_preconditioner,
        multigrid_hierarchy=multigrid_hierarchy,
        translation_gauge_policy=canonical_translation_gauge_policy,
        center_of_mass_weights=center_of_mass_weights,
    )
    if authenticated_operator_sha256 is not None:
        validate_authenticated_operator_geometry(state)
    state.projection_state_sha256 = projection_state_sha256(state)
    return state


def compute_S_from_x(state: SolverState, x: torch.Tensor) -> torch.Tensor:
    """Per-tet symmetric polar ``S = R^T F`` at current positions.

    Supports a leading batch dimension: ``x`` of shape ``(V, 3)`` returns
    ``(T, 3, 3)``, ``(B, V, 3)`` returns ``(B, T, 3, 3)``.

    Previously used ``torch.linalg.svd`` *with* gradient, whose backward is
    ill-conditioned when singular values coincide — i.e. for every near-rest
    tet, which is most of the mesh most of the time.  This is called inside the
    K>1 rollout chain during training, so that mattered.
    """
    x_tet = x[..., state.tets, :]  # (..., T, 4, 3)
    F = torch.einsum("tac,...tad->...tdc", state.J, x_tet)
    R = polar_rotation(F)
    S = R.transpose(-1, -2) @ F
    return 0.5 * (S + S.transpose(-1, -2))


def inertial_predictor(
    state: SolverState, x_t: torch.Tensor, x_prev: torch.Tensor, pinned_targets: torch.Tensor
) -> torch.Tensor:
    """Constant-velocity extrapolation ``2 x_t - x_{t-1}``, with pins restored.

    This is the right warm start for the decoder in a temporal setting. Measured
    on the 8x4x4 article with exact target stretches, it moves the decoder's
    10-iteration error from 1.05e-2 m (warm start at ``x_t``) to 1.02e-3 m at
    identical cost, because local-global converges at only ~0.98 per iteration
    and therefore never travels far from wherever it starts.
    """
    x0 = 2.0 * x_t - x_prev
    x0 = x0.clone()
    x0[..., state.pinned, :] = pinned_targets
    return x0


def _projection_rhs(
    state: SolverState,
    F_target: torch.Tensor,
    pinned_targets: torch.Tensor,
) -> torch.Tensor:
    """Assemble the reduced normal-equation right-hand side."""
    batch = F_target.shape[:-3]
    contrib = torch.einsum("...tdc,tac->...tad", F_target, state.J) * state.w[:, None, None]
    rhs = torch.zeros(*batch, state.n_verts, 3, dtype=state.rest_q.dtype, device=state.rest_q.device)
    rhs.reshape(-1, state.n_verts, 3).index_add_(
        1,
        state.tets.reshape(-1),
        contrib.reshape(-1, state.n_tets * 4, 3),
    )

    if state.projection_backend == _DENSE_BACKEND:
        bc_rhs = torch.einsum("fp,...pd->...fd", state.L_fp, pinned_targets)
    else:
        pin_flat = pinned_targets.reshape(-1, state.pinned.numel(), 3)
        pin_columns = pin_flat.permute(1, 0, 2).reshape(state.pinned.numel(), -1)
        bc_columns = torch.sparse.mm(state.L_fp, pin_columns)
        bc_rhs = bc_columns.reshape(state.free.numel(), -1, 3).permute(1, 0, 2).reshape(*batch, state.free.numel(), 3)
    return rhs[..., state.free, :] - bc_rhs


def _relative_residual(residual_norm: torch.Tensor, rhs_norm: torch.Tensor) -> torch.Tensor:
    return torch.where(rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm)


@dataclasses.dataclass
class _PreconditionerWork:
    matrix_vector_products: int = 0
    factor_solves: int = 0


def _multigrid_v_cycle(
    hierarchy: _MultigridHierarchy,
    level_index: int,
    rhs: torch.Tensor,
    work: _PreconditionerWork,
) -> torch.Tensor:
    """Apply one fixed symmetric V-cycle to all RHS columns."""
    level = hierarchy.levels[level_index]
    if level.aggregate is None:
        work.factor_solves += 1
        return torch.cholesky_solve(rhs, hierarchy.coarse_cholesky)
    if level.smoother_inverse is None:
        raise RuntimeError("non-coarse multigrid level is missing its smoother")

    smoother = level.smoother_inverse[:, None]
    # Starting from zero makes the first pre-sweep exactly B r, without a
    # redundant A @ 0.  Remaining pre-sweeps, restriction, and the mirrored
    # post-sweeps are kept explicit.  The post smoother is the transpose of the
    # pre smoother; adding the recursively SPD coarse correction therefore
    # makes this fixed V-cycle symmetric positive definite for ordinary PCG.
    x = smoother * rhs
    for _ in range(1, hierarchy.smoothing_steps):
        residual = rhs - torch.sparse.mm(level.matrix, x)
        work.matrix_vector_products += 1
        x = x + smoother * residual

    residual = rhs - torch.sparse.mm(level.matrix, x)
    work.matrix_vector_products += 1
    n_coarse = hierarchy.levels[level_index + 1].matrix.shape[0]
    coarse_rhs = torch.zeros(n_coarse, rhs.shape[1], dtype=rhs.dtype, device=rhs.device)
    coarse_rhs = coarse_rhs.index_add(0, level.aggregate, residual)
    coarse_correction = _multigrid_v_cycle(hierarchy, level_index + 1, coarse_rhs, work)
    x = x + coarse_correction[level.aggregate]

    for _ in range(hierarchy.smoothing_steps):
        residual = rhs - torch.sparse.mm(level.matrix, x)
        work.matrix_vector_products += 1
        x = x + smoother * residual
    return x


def _apply_pcg_preconditioner(
    inverse_diagonal: torch.Tensor,
    hierarchy: _MultigridHierarchy | None,
    residual: torch.Tensor,
    work: _PreconditionerWork,
) -> torch.Tensor:
    if hierarchy is None:
        return inverse_diagonal[:, None] * residual
    return _multigrid_v_cycle(hierarchy, 0, residual, work)


def _pcg_solve(
    matrix: torch.Tensor,
    inverse_diagonal: torch.Tensor,
    multigrid_hierarchy: _MultigridHierarchy | None,
    rhs: torch.Tensor,
    initial_guess: torch.Tensor,
    relative_tolerance: float,
    absolute_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, ProjectionDiagnostics]:
    """Solve independent RHS columns with deterministic fixed-preconditioner PCG."""
    n_rows, rhs_count = rhs.shape
    x = initial_guess
    rhs_norm = torch.linalg.vector_norm(rhs, dim=0)
    threshold = absolute_tolerance + relative_tolerance * rhs_norm
    if n_rows:
        residual = rhs - torch.sparse.mm(matrix, x)
        matvec_count = 1
    else:
        residual = rhs
        matvec_count = 0
    residual_norm = torch.linalg.vector_norm(residual, dim=0)
    initial_residual_norm = residual_norm
    active = residual_norm > threshold
    failed = torch.zeros(rhs_count, dtype=torch.bool, device=rhs.device)
    preconditioner_count = 0
    iterations = 0
    residual_is_true = True
    preconditioner_work = _PreconditionerWork()

    if n_rows and bool(active.any().item()):
        z = _apply_pcg_preconditioner(inverse_diagonal, multigrid_hierarchy, residual, preconditioner_work)
        direction = torch.where(active[None, :], z, torch.zeros_like(z))
        residual_dot_z = (residual * z).sum(dim=0)
        preconditioner_count = 1
        # Avoid a device-to-host synchronization after every CUDA iteration.
        convergence_check_interval = 1 if rhs.device.type == "cpu" else 8

        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            matrix_direction = torch.sparse.mm(matrix, direction)
            matvec_count += 1
            denominator = (direction * matrix_direction).sum(dim=0)
            bad_denominator = active & ((denominator <= 0.0) | ~torch.isfinite(denominator))
            failed = failed | bad_denominator
            valid = active & ~bad_denominator
            safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
            alpha = torch.where(valid, residual_dot_z / safe_denominator, torch.zeros_like(denominator))
            x = x + direction * alpha[None, :]
            residual = residual - matrix_direction * alpha[None, :]
            residual_is_true = False
            residual_norm = torch.linalg.vector_norm(residual, dim=0)
            active = (residual_norm > threshold) & ~failed

            if iteration % convergence_check_interval == 0 and not bool(active.any().item()):
                # Recursive CG residuals can drift at tight tolerances.  Confirm
                # convergence against the actual normal equations, and restart
                # from that residual if necessary.
                residual = rhs - torch.sparse.mm(matrix, x)
                matvec_count += 1
                residual_is_true = True
                residual_norm = torch.linalg.vector_norm(residual, dim=0)
                active = (residual_norm > threshold) & ~failed
                if not bool(active.any().item()):
                    break
                z = _apply_pcg_preconditioner(
                    inverse_diagonal,
                    multigrid_hierarchy,
                    residual,
                    preconditioner_work,
                )
                direction = torch.where(active[None, :], z, torch.zeros_like(z))
                residual_dot_z = (residual * z).sum(dim=0)
                preconditioner_count += 1
                continue

            z_new = _apply_pcg_preconditioner(
                inverse_diagonal,
                multigrid_hierarchy,
                residual,
                preconditioner_work,
            )
            residual_dot_z_new = (residual * z_new).sum(dim=0)
            bad_numerator = active & (
                (residual_dot_z <= 0.0) | (residual_dot_z_new <= 0.0) | ~torch.isfinite(residual_dot_z_new)
            )
            failed = failed | bad_numerator
            active = active & ~bad_numerator
            safe_residual_dot_z = torch.where(active, residual_dot_z, torch.ones_like(residual_dot_z))
            beta = torch.where(active, residual_dot_z_new / safe_residual_dot_z, torch.zeros_like(residual_dot_z_new))
            direction = torch.where(active[None, :], z_new + direction * beta[None, :], torch.zeros_like(direction))
            residual_dot_z = residual_dot_z_new
            preconditioner_count += 1

    if n_rows and not residual_is_true:
        residual = rhs - torch.sparse.mm(matrix, x)
        matvec_count += 1
        residual_norm = torch.linalg.vector_norm(residual, dim=0)
    converged_mask = (residual_norm <= threshold) & ~failed
    relative = _relative_residual(residual_norm, rhs_norm)
    converged_rhs = int(converged_mask.sum().item())
    diagnostics = ProjectionDiagnostics(
        backend=_SPARSE_PCG_BACKEND,
        converged=converged_rhs == rhs_count,
        iterations=iterations,
        rhs_count=rhs_count,
        converged_rhs=converged_rhs,
        matrix_vector_products=matvec_count + preconditioner_work.matrix_vector_products,
        preconditioner_applications=preconditioner_count,
        factor_solves=preconditioner_work.factor_solves,
        rhs_norm_max=float(rhs_norm.max().item()),
        initial_residual_norm_max=float(initial_residual_norm.max().item()),
        residual_norm_max=float(residual_norm.max().item()),
        relative_residual_max=float(relative.max().item()),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        breakdown=bool(failed.any().item()),
        preconditioner=(_MULTIGRID_PRECONDITIONER if multigrid_hierarchy is not None else _JACOBI_PRECONDITIONER),
        hierarchy_levels=0 if multigrid_hierarchy is None else len(multigrid_hierarchy.levels),
        preconditioner_matrix_vector_products=preconditioner_work.matrix_vector_products,
    )
    return x, diagnostics


def project_deformation_gradient(
    state: SolverState,
    F_target: torch.Tensor,
    pinned_targets: torch.Tensor,
    *,
    center_of_mass_positions: torch.Tensor | None = None,
    center_of_mass_target: torch.Tensor | None = None,
    relative_tolerance: float | None = None,
    absolute_tolerance: float | None = None,
    max_iterations: int | None = None,
    raise_on_nonconvergence: bool | None = None,
    initial_positions: torch.Tensor | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ProjectionDiagnostics]:
    r"""Fit one globally compatible mesh to a full target-gradient field.

    The projection objective is

    .. math::

        \operatorname*{argmin}_{x,\;x_p=\bar{x}_p}
        \sum_t V_t\lVert F_t(x)-F_t^*\rVert_F^2.

    Since each ``F_t(x)`` is linear in the vertex positions, the normal matrix
    is the rest-mesh Laplacian in :class:`SolverState`.  Dense states obtain the
    floating-point minimizer with one prefactored Cholesky factor solve.  Sparse
    states approximate it to the declared normal-equation residual tolerance
    with conjugate gradients and the state's fixed Jacobi or symmetric
    Galerkin-multigrid preconditioner. Sparse convergence is diagnostics-bound
    and fails closed by default. Both paths preserve pins exactly and have no
    polar or local fixed-point step. Leading batch dimensions are broadcast and
    preserved.

    An unpinned dense state built with the mass-weighted center-of-mass gauge
    instead enforces ``sum_i w_i x_i = c``, where the normalized mass weights
    are fixed in the state. The caller supplies ``c`` directly or supplies
    positions from which it is computed; the rest pose is never an implicit
    translation target. The state prefactors ``L + w w^T``, so the per-call
    work remains one Cholesky solve. Per-call validation checks cheap tensor
    invariants; as with the state's matrix and factor tensors, trusted consumers
    reauthenticate factor/weight consistency with
    :func:`projection_state_sha256` before execution.

    The multigrid path is currently inference-only and rejects gradient-bearing
    inputs.  Its tolerance-terminated unrolled PCG backward is not an implicit
    derivative of the converged projection.  Dense and Jacobi autograd behavior
    remains available and unchanged.

    Args:
        state: Precomputed decoder state for the shared tetrahedral mesh.
        F_target: Full target deformation gradients, ``(..., T, 3, 3)``.
        pinned_targets: Pinned world positions [m], ``(..., P, 3)``.
        center_of_mass_positions: Optional positions [m], ``(..., V, 3)``,
            whose mass-weighted center is the translation-gauge target.
        center_of_mass_target: Optional explicit mass-weighted center [m],
            ``(..., 3)``. A center-of-mass-gauge state requires exactly one of
            this argument and ``center_of_mass_positions``.
        relative_tolerance: Optional per-call sparse relative residual
            tolerance override.
        absolute_tolerance: Optional per-call sparse absolute residual
            tolerance override.
        max_iterations: Optional per-call sparse PCG iteration limit.
        raise_on_nonconvergence: Optional per-call fail-closed override.
        initial_positions: Optional sparse PCG initial positions [m],
            ``(..., V, 3)``. Defaults to the rest positions. Ignored by the
            direct backend because its factor solve is not iterative.
        return_diagnostics: Return work and residual diagnostics with the
            positions. The default preserves the original tensor-only API.

    Returns:
        Projected vertex positions [m], ``(..., V, 3)``, optionally paired with
        :class:`ProjectionDiagnostics`.
    """
    dtype = state.rest_q.dtype
    device = state.rest_q.device
    if state.tikhonov != 0.0:
        raise ValueError("exact deformation-gradient projection requires a solver state without tikhonov")
    F_target = F_target.to(dtype=dtype, device=device)
    pinned_targets = pinned_targets.to(dtype=dtype, device=device)

    if F_target.shape[-3:] != (state.n_tets, 3, 3):
        raise ValueError(f"F_target must end in ({state.n_tets}, 3, 3), got {tuple(F_target.shape)}")
    if pinned_targets.shape[-2:] != (state.pinned.numel(), 3):
        raise ValueError(f"pinned_targets must end in ({state.pinned.numel()}, 3), got {tuple(pinned_targets.shape)}")

    translation_gauge_policy = state.translation_gauge_policy
    if translation_gauge_policy not in (
        _TRANSLATION_GAUGE_NONE,
        TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
    ):
        raise ValueError("solver state has an unsupported translation gauge policy")
    has_center_of_mass_gauge = translation_gauge_policy == TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS
    center = None
    if has_center_of_mass_gauge:
        if state.projection_backend != _DENSE_BACKEND:
            raise ValueError("mass-weighted center-of-mass gauge currently supports only dense projection")
        if state.pinned.numel() != 0:
            raise ValueError("center-of-mass gauge is separate from physical pins and requires no physical pins")
        weights = state.center_of_mass_weights
        if weights is None or weights.shape != (state.n_verts,):
            raise ValueError("center-of-mass-gauge solver state is missing its normalized mass weights")
        if weights.dtype != dtype or weights.device != device:
            raise ValueError("center-of-mass weights must match the solver state dtype and device")
        weights_are_finite = torch.isfinite(weights).all()
        weights_are_non_negative = (weights >= 0.0).all()
        weights_are_normalized = torch.isclose(
            weights.sum(),
            weights.new_tensor(1.0),
            rtol=0.0,
            atol=32.0 * torch.finfo(dtype).eps,
        )
        if (center_of_mass_positions is None) == (center_of_mass_target is None):
            raise ValueError(
                "center-of-mass-gauge projection requires exactly one of "
                "center_of_mass_positions and center_of_mass_target"
            )
        if center_of_mass_positions is not None:
            center_of_mass_positions = center_of_mass_positions.to(dtype=dtype, device=device)
            if center_of_mass_positions.shape[-2:] != (state.n_verts, 3):
                raise ValueError(
                    f"center_of_mass_positions must end in ({state.n_verts}, 3), "
                    f"got {tuple(center_of_mass_positions.shape)}"
                )
            gauge_input_is_finite = torch.isfinite(center_of_mass_positions).all()
            center = torch.einsum("v,...vd->...d", weights, center_of_mass_positions)
        else:
            center = center_of_mass_target.to(dtype=dtype, device=device)
            if center.shape[-1:] != (3,):
                raise ValueError(f"center_of_mass_target must end in (3,), got {tuple(center.shape)}")
            gauge_input_is_finite = torch.isfinite(center).all()
        center_is_finite = torch.isfinite(center).all()
        gauge_is_valid = (
            weights_are_finite
            & weights_are_non_negative
            & weights_are_normalized
            & gauge_input_is_finite
            & center_is_finite
        )
        if not bool(gauge_is_valid.item()):
            if not bool(weights_are_finite.item()):
                raise ValueError("center-of-mass weights must be finite")
            if not bool(weights_are_non_negative.item()):
                raise ValueError("center-of-mass weights must be non-negative")
            if not bool(weights_are_normalized.item()):
                raise ValueError("center-of-mass weights must be normalized to sum to one")
            if not bool(gauge_input_is_finite.item()):
                if center_of_mass_positions is not None:
                    raise ValueError("center_of_mass_positions must be finite")
                raise ValueError("center_of_mass_target must be finite")
            raise ValueError("mass-weighted center computed from positions must be finite")
    elif center_of_mass_positions is not None or center_of_mass_target is not None:
        raise ValueError("center-of-mass target requires a solver state built with its translation gauge policy")

    if initial_positions is not None:
        initial_positions = initial_positions.to(dtype=dtype, device=device)
        if initial_positions.shape[-2:] != (state.n_verts, 3):
            raise ValueError(
                f"initial_positions must end in ({state.n_verts}, 3), got {tuple(initial_positions.shape)}"
            )
    if state.multigrid_hierarchy is not None and (
        F_target.requires_grad
        or pinned_targets.requires_grad
        or (initial_positions is not None and initial_positions.requires_grad)
    ):
        raise ValueError(
            "multigrid sparse projection is inference-only until an implicit-adjoint backward is implemented"
        )

    initial_batch = () if initial_positions is None else initial_positions.shape[:-2]
    center_batch = () if center is None else center.shape[:-1]
    batch = torch.broadcast_shapes(F_target.shape[:-3], pinned_targets.shape[:-2], initial_batch, center_batch)
    F_target = F_target.expand(*batch, state.n_tets, 3, 3)
    pinned_targets = pinned_targets.expand(*batch, state.pinned.numel(), 3)
    if center is not None:
        center = center.expand(*batch, 3)
    b = _projection_rhs(state, F_target, pinned_targets)
    if center is not None:
        b = b + weights[:, None] * center[..., None, :]
    flat_b = b.reshape(-1, *b.shape[-2:])
    b_columns = flat_b.permute(1, 0, 2).reshape(flat_b.shape[1], -1)
    if state.projection_backend == _DENSE_BACKEND:
        if state.L_ff_chol is None:
            raise RuntimeError("dense projection state is missing its Cholesky factor")
        x_columns = torch.cholesky_solve(b_columns, state.L_ff_chol)
        if return_diagnostics:
            normal_residual = state.L_ff_chol @ (state.L_ff_chol.transpose(0, 1) @ x_columns) - b_columns
            residual_norm = torch.linalg.vector_norm(normal_residual, dim=0)
            rhs_norm = torch.linalg.vector_norm(b_columns, dim=0)
            relative = _relative_residual(residual_norm, rhs_norm)
            diagnostics = ProjectionDiagnostics(
                backend=_DENSE_BACKEND,
                converged=True,
                iterations=0,
                rhs_count=b_columns.shape[1],
                converged_rhs=b_columns.shape[1],
                matrix_vector_products=0,
                preconditioner_applications=0,
                factor_solves=1,
                rhs_norm_max=float(rhs_norm.max().item()),
                initial_residual_norm_max=float(rhs_norm.max().item()),
                residual_norm_max=float(residual_norm.max().item()),
                relative_residual_max=float(relative.max().item()),
                relative_tolerance=None,
                absolute_tolerance=None,
                preconditioner=None,
            )
    else:
        if state.L_ff_sparse is None or state.L_ff_inverse_diagonal is None:
            raise RuntimeError("sparse projection state is missing its CSR matrix or preconditioner")
        if state.pcg_preconditioner == _MULTIGRID_PRECONDITIONER and state.multigrid_hierarchy is None:
            raise RuntimeError("multigrid sparse projection state is missing its hierarchy")
        relative_tolerance = state.pcg_relative_tolerance if relative_tolerance is None else float(relative_tolerance)
        absolute_tolerance = state.pcg_absolute_tolerance if absolute_tolerance is None else float(absolute_tolerance)
        max_iterations = state.pcg_max_iterations if max_iterations is None else max_iterations
        _validate_pcg_options(relative_tolerance, absolute_tolerance, max_iterations)
        if initial_positions is None:
            initial_positions = state.rest_q.expand(*batch, state.n_verts, 3)
        else:
            initial_positions = initial_positions.expand(*batch, state.n_verts, 3)
        initial_flat = initial_positions[..., state.free, :].reshape(-1, state.free.numel(), 3)
        initial_columns = initial_flat.permute(1, 0, 2).reshape(state.free.numel(), -1)
        x_columns, diagnostics = _pcg_solve(
            state.L_ff_sparse,
            state.L_ff_inverse_diagonal,
            state.multigrid_hierarchy,
            b_columns,
            initial_columns,
            relative_tolerance,
            absolute_tolerance,
            max_iterations,
        )
        fail_closed = state.pcg_raise_on_nonconvergence if raise_on_nonconvergence is None else raise_on_nonconvergence
        if fail_closed and not diagnostics.converged:
            raise RuntimeError(
                "sparse deformation-gradient projection did not converge: "
                f"{diagnostics.converged_rhs}/{diagnostics.rhs_count} RHS, "
                f"iterations={diagnostics.iterations}, "
                f"max_relative_residual={diagnostics.relative_residual_max:.3e}, "
                f"breakdown={diagnostics.breakdown}"
            )
    x_free = x_columns.reshape(-1, flat_b.shape[0], 3).permute(1, 0, 2).reshape(b.shape)

    x = torch.zeros(*batch, state.n_verts, 3, dtype=dtype, device=device)
    x[..., state.free, :] = x_free
    x[..., state.pinned, :] = pinned_targets
    if center is not None:
        # Correct only roundoff in the null direction. This does not change any
        # deformation gradient and makes the caller's gauge exact to working
        # precision even if assembled normal-equation contributions do not sum
        # to bitwise zero.
        actual_center = torch.einsum("v,...vd->...d", weights, x)
        x = x + (center - actual_center)[..., None, :]
    if return_diagnostics:
        return x, diagnostics
    return x


def solve(
    state: SolverState,
    S_target: torch.Tensor,
    pinned_targets: torch.Tensor,
    x_init: torch.Tensor | None = None,
    n_iters: int = 6,
) -> torch.Tensor:
    """Differentiable local-global decode: (S_target, pins) -> x.

    Batched: a leading batch dimension is optional and preserved.  Every sample
    in the batch shares the mesh (and therefore the single Cholesky factor), so
    the whole batch is one triangular solve.

    Args:
        S_target: ``(..., T, 3, 3)`` symmetric target stretches.
        pinned_targets: ``(..., P, 3)`` world positions of pinned vertices.
        x_init: ``(..., V, 3)`` optional warm start.  Defaults to the rest pose.
            In a temporal setting pass :func:`inertial_predictor` — the
            iteration converges at only ~0.98 per step, so the warm start
            dominates the result at any practical ``n_iters``.
        n_iters: number of unrolled local-global iterations.
    """
    if state.translation_gauge_policy != _TRANSLATION_GAUGE_NONE:
        raise ValueError("translation gauges are supported only by project_deformation_gradient")
    if state.L_ff_chol is None:
        raise ValueError(
            "local-global solve requires projection_backend='dense'; "
            "a sparse_pcg state supports only project_deformation_gradient"
        )
    dtype = state.L_ff_chol.dtype
    device = state.L_ff_chol.device
    S_target = S_target.to(dtype=dtype, device=device)
    pinned_targets = pinned_targets.to(dtype=dtype, device=device)

    batch = S_target.shape[:-3]
    if x_init is None:
        x = state.rest_q.expand(*batch, -1, -1).clone()
    else:
        x = x_init.to(dtype=dtype, device=device).expand(*batch, -1, -1).clone()
    x[..., state.pinned, :] = pinned_targets

    bc_rhs = torch.einsum("fp,...pd->...fd", state.L_fp, pinned_targets)
    # index_add_ needs a flat leading dim; the mesh is shared so flatten/restore.
    flat_v = (-1, state.n_verts, 3)
    idx = state.tets.reshape(-1)

    for _ in range(n_iters):
        F = torch.einsum("tac,...tad->...tdc", state.J, x[..., state.tets, :])
        # Local step: R = polar(F S*^T).
        R = polar_rotation(F @ S_target.transpose(-1, -2))
        contrib = torch.einsum("...tdc,tac->...tad", R @ S_target, state.J) * state.w[:, None, None]
        rhs = torch.zeros(*batch, state.n_verts, 3, dtype=dtype, device=device)
        rhs.reshape(flat_v).index_add_(1, idx, contrib.reshape(-1, state.n_tets * 4, 3))
        # Global step: one solve against the pre-factored rest-mesh Laplacian.
        # Fold the batch into the RHS columns instead of broadcasting: a batched
        # B against an unbatched factor makes cholesky_solve materialise the
        # (F, F) factor per batch element (22 GB at 180 frames x 4k verts).
        b = rhs[..., state.free, :] - bc_rhs  # (*batch, F, 3)
        bf = b.reshape(-1, *b.shape[-2:])  # (B, F, 3)
        b_cols = bf.permute(1, 0, 2).reshape(bf.shape[1], -1)  # (F, B*3)
        x_cols = torch.cholesky_solve(b_cols, state.L_ff_chol)
        x_free = x_cols.reshape(-1, bf.shape[0], 3).permute(1, 0, 2).reshape(b.shape)
        x_new = x.clone()
        x_new[..., state.free, :] = x_free
        x_new[..., state.pinned, :] = pinned_targets
        x = x_new

    return x
