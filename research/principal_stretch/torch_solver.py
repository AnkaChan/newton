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
- L:       (V, V) dense  (small enough)
- S*:      (T, 3, 3) symmetric
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .polar import polar_rotation


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
    """F[t, d, c] = sum_a J[t, a, c] * x[tets[t, a], d]."""
    x_tet = x[tets]  # (T, 4, 3)
    return torch.einsum("tac,tad->tdc", J, x_tet)


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
    L: torch.Tensor  # (V, V) dense
    L_ff_chol: torch.Tensor  # (F, F) lower-triangular cholesky factor
    L_fp: torch.Tensor  # (F, P)
    rest_q: torch.Tensor  # (V, 3)


def build_solver(
    rest_q: np.ndarray,
    tet_indices: np.ndarray,
    tet_poses: np.ndarray,
    pinned_indices: np.ndarray,
    device: torch.device,
    dtype=torch.float64,
    tikhonov: float = 0.0,
) -> SolverState:
    rest_q_t = torch.as_tensor(rest_q, dtype=dtype, device=device)
    tets = torch.as_tensor(tet_indices, dtype=torch.int64, device=device)
    Dm_inv = torch.as_tensor(tet_poses, dtype=dtype, device=device)
    pinned = torch.as_tensor(pinned_indices, dtype=torch.int64, device=device)

    n_verts = rest_q_t.shape[0]
    n_tets = tets.shape[0]

    det_inv = torch.linalg.det(Dm_inv)
    w = 1.0 / (6.0 * det_inv)
    if (w <= 0).any():
        raise ValueError("non-positive rest volumes — check tet orientation")

    J = _build_J(Dm_inv)  # (T, 4, 3)

    # Dense assembly of L on rest mesh: L = sum_e w_e * (J_e @ J_e^T) scattered.
    # K[t, a, b] = w[t] * sum_c J[t, a, c] * J[t, b, c]
    K = torch.einsum("tac,tbc->tab", J, J) * w[:, None, None]  # (T, 4, 4)
    L = torch.zeros(n_verts, n_verts, dtype=dtype, device=device)
    rows = tets[:, :, None].expand(-1, -1, 4)  # (T, 4, 4)
    cols = tets[:, None, :].expand(-1, 4, -1)
    L.index_put_((rows.reshape(-1), cols.reshape(-1)), K.reshape(-1), accumulate=True)

    mask = torch.ones(n_verts, dtype=torch.bool, device=device)
    mask[pinned] = False
    free = torch.where(mask)[0]

    L_ff = L[free][:, free]
    if tikhonov > 0.0:
        L_ff = L_ff + tikhonov * torch.eye(free.numel(), dtype=dtype, device=device)
    L_fp = L[free][:, pinned]
    L_ff_chol = torch.linalg.cholesky(L_ff)

    return SolverState(
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
    )


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
        x_free = torch.cholesky_solve(rhs[..., state.free, :] - bc_rhs, state.L_ff_chol)
        x_new = x.clone()
        x_new[..., state.free, :] = x_free
        x_new[..., state.pinned, :] = pinned_targets
        x = x_new

    return x
