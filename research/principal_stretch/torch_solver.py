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
    """Polar rotation R from M with reflection correction.

    Uses SVD on the forward pass, but detaches U and Vh from the autograd
    graph and re-routes the gradient through the closed-form polar-projection
    identity to avoid the ill-conditioned SVD gradient when singular values
    coincide (which happens at convergence where F = R S, M = F S^T = R has
    all singular values equal to 1).

    Gradient identity: dR = sym((R^T dM)) ... but the cleanest stable form is
    to recompute R as the projection of M onto SO(3) via one step of Newton
    iteration starting from the (detached) SVD answer:
        R_init = polar_svd(M).detach()
        R = R_init + (M - R_init @ S_init) @ V S^{-1} V^T  ... (complicated)
    Instead we use a simpler approach: do SVD in detached mode, then do
    a single re-projection step that is differentiable through M only.
    """
    with torch.no_grad():
        U, _s, Vh = torch.linalg.svd(M)
        det = torch.linalg.det(U @ Vh)
        D = torch.eye(3, dtype=M.dtype, device=M.device).expand(M.shape[0], -1, -1).clone()
        D[:, 2, 2] = det
        R0 = U @ D @ Vh  # detached rotation guess

    # One Newton-iteration polish: R_{k+1} = 0.5 (R_k + R_k^{-T}).
    # For the autograd path we use a different identity: at the polar
    # decomposition M = R S with S = R^T M, dR satisfies
    #     R^T dR + dR^T R = (R^T dM + dM^T R) S^{-1} - (...)  (Lewis–Sendov)
    # The cheap differentiable surrogate is to project M onto the tangent
    # plane of SO(3) at R0: R = R0 + (I - R0 R0^T) @ M (M^T M)^{-1/2}, but
    # since R0 R0^T = I exactly, this collapses to a constant. We therefore
    # rely on the fact that the forward value R is already a function of M
    # via the (M S^{-1}) channel, where S = R0^T M is differentiable.
    S = R0.transpose(-1, -2) @ M  # symmetric stretch (positive definite-ish)
    # R = M @ S^{-1}; use Cholesky on S (regularised) for stable inverse.
    eye = torch.eye(3, dtype=M.dtype, device=M.device).expand_as(S)
    S_reg = 0.5 * (S + S.transpose(-1, -2)) + 1e-6 * eye
    S_inv = torch.linalg.inv(S_reg)
    R = M @ S_inv
    return R


def assemble_rhs(R: torch.Tensor, S_target: torch.Tensor, J: torch.Tensor,
                 w: torch.Tensor, tets: torch.Tensor, n_verts: int) -> torch.Tensor:
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


def build_solver(rest_q: np.ndarray, tet_indices: np.ndarray, tet_poses: np.ndarray,
                 pinned_indices: np.ndarray, device: torch.device, dtype=torch.float64,
                 tikhonov: float = 0.0) -> SolverState:
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
        n_verts=n_verts, n_tets=n_tets, tets=tets, Dm_inv=Dm_inv, J=J, w=w,
        pinned=pinned, free=free, L=L, L_ff_chol=L_ff_chol, L_fp=L_fp,
        rest_q=rest_q_t,
    )


def solve(state: SolverState, S_target: torch.Tensor, pinned_targets: torch.Tensor,
          x_init: torch.Tensor | None = None, n_iters: int = 6) -> torch.Tensor:
    """Differentiable local-global decode: (S_target, pins) -> x.

    Args:
        S_target: (T, 3, 3) symmetric target stretches.
        pinned_targets: (P, 3) world positions of pinned vertices.
        x_init: (V, 3) optional warm start.  Defaults to rest pose for free
            vertices, pinned targets at pinned vertices.
        n_iters: number of unrolled local-global iterations.
    """
    dtype = state.L_ff_chol.dtype
    device = state.L_ff_chol.device
    S_target = S_target.to(dtype=dtype, device=device)
    pinned_targets = pinned_targets.to(dtype=dtype, device=device)

    if x_init is None:
        x = state.rest_q.clone()
    else:
        x = x_init.to(dtype=dtype, device=device).clone()
    # Hard pin: pinned rows always equal targets.
    x = x.clone()
    x[state.pinned] = pinned_targets

    bc_rhs = state.L_fp @ pinned_targets  # (|free|, 3)

    for _ in range(n_iters):
        F = compute_F(x, state.tets, state.J)
        # Local: R = polar(F @ S*^T)
        R = polar_R(F @ S_target.transpose(-1, -2))
        rhs = assemble_rhs(R, S_target, state.J, state.w, state.tets, state.n_verts)
        b_free = rhs[state.free] - bc_rhs
        x_free = torch.cholesky_solve(b_free, state.L_ff_chol)
        x_new = x.clone()
        x_new[state.free] = x_free
        x_new[state.pinned] = pinned_targets
        x = x_new

    return x
