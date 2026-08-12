# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Batched 3x3 polar decomposition with an exact analytic backward pass.

Replaces ``torch.linalg.svd`` in the decoder's local step.  Two reasons:

1. **Correctness.**  The previous hand-rolled gradient in
   :func:`torch_solver.polar_R` re-routed through ``R = M inv(sym(R0^T M))``.
   That is exact only at ``S = I`` and is 22% wrong at 50% stretch
   (see ``diag_polar_grad.py``).  This module implements the exact
   differential instead.

2. **Speed.**  Batched cuSOLVER SVD on ``(T, 3, 3)`` is the decoder's dominant
   cost.  Newton's iteration for the orthogonal polar factor is pure batched
   matmul / elementwise work.

Forward
-------
Scaled Newton iteration (Higham),

    R_{k+1} = 0.5 (gamma_k R_k + gamma_k^-1 R_k^-T),
    gamma_k = (||R_k^-1||_F / ||R_k||_F)^(1/2)

started at ``R_0 = M``.  Converges quadratically to the orthogonal polar factor
whenever ``det M > 0``.  Elements with ``det M <= 0`` (which do not arise for
``M = F S*^T`` with ``det F > 0`` and SPD ``S*``, but are guarded anyway) fall
back to the reflection-corrected SVD.

Backward
--------
With ``M = R S``, ``S = R^T M`` symmetric, the differential of the rotation is
``dR = R Omega`` where the skew ``Omega`` solves the Sylvester equation

    Omega S + S Omega = R^T dM - dM^T R.

Writing skew matrices via their axial vectors and using the identity
``[b]_x S + S [b]_x = [(tr(S) I - S) b]_x`` for symmetric ``S``, that Sylvester
solve collapses to a 3x3 linear system.  The adjoint (what backward needs) is

    A     = skew(R^T grad_R)
    b     = (tr(S) I - S)^-1 axial(A)
    grad_M = 2 R [b]_x

``tr(S) I - S`` has eigenvalues ``(s2+s3, s1+s3, s1+s2)`` in the principal
stretches, so it is positive definite and well conditioned for any physical
deformation -- unlike the SVD backward, which blows up when singular values
coincide (i.e. near the rest state, which is most of the mesh most of the time).
"""

from __future__ import annotations

import torch


def _inv3(A: torch.Tensor) -> torch.Tensor:
    """Analytic inverse of a batch of 3x3 matrices via the adjugate."""
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    c00, c01, c02 = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * c00 + b * c01 + c * c02
    adj = torch.stack(
        [
            torch.stack([c00, c * h - b * i, b * f - c * e], dim=-1),
            torch.stack([c01, a * i - c * g, c * d - a * f], dim=-1),
            torch.stack([c02, b * g - a * h, a * e - b * d], dim=-1),
        ],
        dim=-2,
    )
    return adj / det[..., None, None]


def _det3(A: torch.Tensor) -> torch.Tensor:
    """Analytic determinant of a batch of 3x3 matrices."""
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    return a * (e * i - f * h) + b * (f * g - d * i) + c * (d * h - e * g)


def _svd_polar(M: torch.Tensor) -> torch.Tensor:
    """Reflection-corrected polar rotation via SVD (reference / fallback)."""
    U, _s, Vh = torch.linalg.svd(M)
    det = torch.linalg.det(U @ Vh)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(M.shape[0], -1, -1).clone()
    D[:, 2, 2] = det
    return U @ D @ Vh


def polar_rotation_forward(M: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Orthogonal polar factor of a batch of 3x3 matrices. No autograd graph."""
    with torch.no_grad():
        R = M.clone()
        for _ in range(iters):
            R_inv_t = _inv3(R).transpose(-1, -2)
            n_r = R.flatten(-2).norm(dim=-1, keepdim=True)[..., None]
            n_i = R_inv_t.flatten(-2).norm(dim=-1, keepdim=True)[..., None]
            gamma = torch.sqrt(n_i / n_r.clamp(min=1e-300))
            R = 0.5 * (gamma * R + R_inv_t / gamma)

        # Guard: Newton converges to a reflection when det M < 0, and to nothing
        # useful when M is singular.  Fall back to SVD on those elements only.
        bad = ~torch.isfinite(R).all(dim=(-2, -1)) | (torch.linalg.det(M) <= 0)
        if bad.any():
            R = R.clone()
            R[bad] = _svd_polar(M[bad])
    return R


_AXIAL = ((2, 1), (0, 2), (1, 0))


def _axial(A: torch.Tensor) -> torch.Tensor:
    """Axial vector of the skew part: a_k such that skew(A) = [a]_x."""
    return torch.stack([A[..., i, j] - A[..., j, i] for i, j in _AXIAL], dim=-1) * 0.5


def _cross_matrix(b: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(b[..., 0])
    return torch.stack(
        [
            torch.stack([z, -b[..., 2], b[..., 1]], dim=-1),
            torch.stack([b[..., 2], z, -b[..., 0]], dim=-1),
            torch.stack([-b[..., 1], b[..., 0], z], dim=-1),
        ],
        dim=-2,
    )


class _PolarRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, iters):
        R = polar_rotation_forward(M, iters)
        ctx.save_for_backward(R, R.transpose(-1, -2) @ M)
        return R

    @staticmethod
    def backward(ctx, grad_R):
        R, S = ctx.saved_tensors
        S = 0.5 * (S + S.transpose(-1, -2))
        a = _axial(R.transpose(-1, -2) @ grad_R)
        eye = torch.eye(3, dtype=S.dtype, device=S.device).expand_as(S)
        tr_S = S.diagonal(dim1=-2, dim2=-1).sum(-1)
        K = tr_S[..., None, None] * eye - S
        scale_floor = torch.finfo(S.dtype).tiny ** 0.5
        relative_floor = 1.0e-10 if S.dtype == torch.float64 else 1.0e-5
        scale = K.abs().amax(dim=(-2, -1)).clamp(min=scale_floor)
        shifted = K - relative_floor * scale[..., None, None] * eye
        minor_1 = shifted[..., 0, 0]
        minor_2 = shifted[..., 0, 0] * shifted[..., 1, 1] - shifted[..., 0, 1] * shifted[..., 1, 0]
        good = (
            torch.isfinite(S).all(dim=(-2, -1))
            & torch.isfinite(K).all(dim=(-2, -1))
            & (_det3(S) > 0.0)
            & (minor_1 > 0.0)
            & (minor_2 > 0.0)
            & (_det3(shifted) > 0.0)
        )
        # The reflection-corrected and rank-deficient proper-polar branches
        # are non-smooth, and their exact derivative is undefined or
        # unbounded.  Stop only that local polar path rather than feeding an
        # invalid/singular Sylvester solve into the decoder gradient.
        safe_K = torch.where(good[..., None, None], K, eye)
        candidate = (_inv3(safe_K) @ a.unsqueeze(-1)).squeeze(-1)
        b = torch.where(good[..., None], candidate, torch.zeros_like(candidate))
        return 2.0 * R @ _cross_matrix(b), None


def polar_rotation(M: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Differentiable orthogonal polar factor ``R`` of ``M = R S``.

    Args:
        M: Batch of 3x3 matrices, shape ``(..., 3, 3)``.
        iters: Newton iterations for the forward pass. Measured: 5 reaches
            fp64 round-off even at 17:1 principal-stretch anisotropy and
            ``det S = 0.09``; the default carries one iteration of margin.

    Returns:
        Rotations of the same shape as ``M``.
    """
    shape = M.shape
    R = _PolarRotation.apply(M.reshape(-1, 3, 3), iters)
    return R.reshape(shape)
