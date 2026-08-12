# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Batched 3x3 SPD matrix log/exp and SO(3) log with exact analytic backward.

The multi-resolution stretch representation lives in the matrix-log domain:
right-stretch tensors ``S`` (SPD, eigenvalues near 1) are mapped to symmetric
``H = log S`` where interpolation and network regression are unconstrained,
and mapped back with ``exp``.  Relative rotations between adjacent tets are
reduced to axis-angle vectors with the SO(3) log.

Why not autograd through ``torch.linalg.eigh`` directly?  Its backward
contains ``1 / (lam_i - lam_j)`` terms and returns NaN (or garbage) whenever
eigenvalues coincide -- and near-isotropic stretch, i.e. repeated eigenvalues,
is the rest state and therefore most of the mesh most of the time.  The
composed map ``f(M) = U diag(f(lam)) U^T`` is perfectly smooth there; only the
eigen-decomposition route is singular.  So the forward uses ``eigh`` but the
backward is the exact Daleckii-Krein differential, which stays finite.

Forward
-------
For symmetric ``M`` with ``M = U diag(lam) U^T``,

    f(M) = U diag(f(lam)) U^T,        f in {log, exp}.

Inputs are symmetrized (``0.5 (M + M^T)``) so the map -- and its gradient --
is well defined for the nearly-symmetric matrices produced upstream.

Backward (Daleckii-Krein)
-------------------------
The differential of ``f(M)`` in the eigenbasis is elementwise multiplication
by the divided-difference (Loewner) matrix ``G``:

    grad_M = U ( G .* (U^T grad_out U) ) U^T,   then symmetrized,
    G_ij   = (f(lam_i) - f(lam_j)) / (lam_i - lam_j)   if |lam_i - lam_j| > eps,
    G_ij   = f'((lam_i + lam_j) / 2)                   otherwise,

with ``eps = 1e-9`` in fp64 and ``1e-4`` in fp32.  The wider fp32 branch is
strictly safer: ``f'(mid)`` is off by ``O(gap^2 / 24)`` (~4e-10 at gap 1e-4,
invisible in fp32), while the quantized numerator fails catastrophically once
``f(lam)`` differences approach the dtype's ulp — ``exp(lam) ~= 1`` is stored
at ulp 1.19e-7 in fp32, so a 1e-8 gap rounds both exponentials to exactly
1.0f (``G = 0``, gradient component silently zeroed) and gaps just above a
fp64-sized threshold quantize the numerator to ``+-1 ulp`` (``|G|`` spikes up
to ~1e2).  ``G`` is bounded by ``max f'(lam)`` on the eigenvalue range, so
gradients are as well conditioned as ``f`` itself regardless of eigenvalue
multiplicity.

SO(3) log
---------
    theta = arccos(clamp((tr R - 1) / 2, -1, 1)),
    axial = theta / (2 sin theta) * [R32-R23, R13-R31, R21-R12],

with the Taylor limit factor ``1/2`` for ``theta < 1e-4``.  ``theta > 3.0``
raises: adjacent-tet relative rotations near pi are out of physical range,
and the axial formula degenerates there (``sin theta -> 0``).  Composed from
plain differentiable torch ops (no custom Function); the small-angle branch
is guarded with ``torch.where`` on safe inputs so no NaN leaks into gradients
through the unselected branch.
"""

from __future__ import annotations

import math

import torch

# Eigen-gap below which the divided difference switches to f'(mid), per dtype.
# The switch costs O(gap^2 / 24) accuracy; the divided difference costs
# ulp(f(lam)) / gap, which in fp32 zeroes or spikes G for gaps below ~1e-6
# (see the module docstring).  Dtypes below fp64 get the wide branch.
_EIG_EPS_F64 = 1e-9
_EIG_EPS_F32 = 1e-4
_SMALL_THETA = 1e-4  # rotation angle below which the axial factor is Taylor's 1/2
_MAX_THETA = 3.0  # rotation angles beyond this are out of physical range

# cusolverDnXsyevBatched rejects large batches with CUSOLVER_STATUS_INVALID_VALUE
# (workspace-size overflow; ~31.6k 3x3 fp64 matrices on torch 2.10/cu128, L40).
# Chunk CUDA eigh calls well below the observed limit; chunked cusolver measured
# ~20x faster than the magma backend for 138k matrices.
_EIGH_CHUNK = 16384


def _batched_eigh(Ms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.linalg.eigh`` chunked over the flattened batch on CUDA."""
    n = Ms.numel() // (Ms.shape[-1] * Ms.shape[-2])
    if not Ms.is_cuda or n <= _EIGH_CHUNK:
        return torch.linalg.eigh(Ms)
    flat = Ms.reshape(-1, Ms.shape[-2], Ms.shape[-1])
    lams, us = [], []
    for i in range(0, n, _EIGH_CHUNK):
        lam_i, u_i = torch.linalg.eigh(flat[i : i + _EIGH_CHUNK])
        lams.append(lam_i)
        us.append(u_i)
    lam = torch.cat(lams).reshape(*Ms.shape[:-1])
    u = torch.cat(us).reshape(Ms.shape)
    return lam, u


class _SymmetricMatrixFunction(torch.autograd.Function):
    """``f(M) = U diag(f(lam)) U^T`` with the Daleckii-Krein backward."""

    @staticmethod
    def forward(ctx, M, f, fprime):
        Ms = 0.5 * (M + M.transpose(-1, -2))
        lam, U = _batched_eigh(Ms)
        flam = f(lam)
        ctx.save_for_backward(lam, U, flam)
        ctx.fprime = fprime
        return U @ torch.diag_embed(flam) @ U.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_out):
        lam, U, flam = ctx.saved_tensors
        d = lam[..., :, None] - lam[..., None, :]
        num = flam[..., :, None] - flam[..., None, :]
        mid = 0.5 * (lam[..., :, None] + lam[..., None, :])
        eps = _EIG_EPS_F64 if lam.dtype == torch.float64 else _EIG_EPS_F32
        close = d.abs() <= eps
        G = torch.where(close, ctx.fprime(mid), num / torch.where(close, torch.ones_like(d), d))
        Ut = U.transpose(-1, -2)
        g = U @ (G * (Ut @ grad_out @ U)) @ Ut
        return 0.5 * (g + g.transpose(-1, -2)), None, None


def sym_log(S: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm of a batch of SPD 3x3 matrices.

    Args:
        S: SPD matrices (right-stretch tensors), shape ``(..., 3, 3)``.

    Returns:
        Symmetric ``log S`` of the same shape, differentiable with the exact
        Daleckii-Krein backward (finite at repeated eigenvalues).
    """
    return _SymmetricMatrixFunction.apply(S, torch.log, torch.reciprocal)


def sym_exp(H: torch.Tensor) -> torch.Tensor:
    """Matrix exponential of a batch of symmetric 3x3 matrices; inverse of :func:`sym_log`.

    Args:
        H: Symmetric matrices, shape ``(..., 3, 3)``.

    Returns:
        SPD ``exp H`` of the same shape, differentiable with the exact
        Daleckii-Krein backward (finite at repeated eigenvalues).
    """
    return _SymmetricMatrixFunction.apply(H, torch.exp, torch.exp)


def spd_floor(S: torch.Tensor, lam_min: float = 0.05) -> torch.Tensor:
    """Clamp symmetric-matrix eigenvalues to a positive floor.

    This is the spectral function ``f(lam) = max(lam, lam_min)`` evaluated
    through the same Daleckii-Krein path as :func:`sym_log`.  Its derivative
    therefore stays finite at repeated eigenvalues, unlike direct autograd
    through an eigendecomposition.

    Args:
        S: Symmetric matrices, shape ``(..., 3, 3)``.
        lam_min: Minimum output eigenvalue.

    Returns:
        Symmetric positive-definite matrices with the same shape as ``S``.
    """

    def clamp(lam: torch.Tensor) -> torch.Tensor:
        return lam.clamp(min=lam_min)

    def clamp_prime(lam: torch.Tensor) -> torch.Tensor:
        return (lam > lam_min).to(lam.dtype)

    return _SymmetricMatrixFunction.apply(S, clamp, clamp_prime)


def so3_log_axial(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle (axial) vector of a batch of rotation matrices.

    Args:
        R: Rotation matrices, shape ``(..., 3, 3)``.

    Returns:
        Axial vectors ``theta * axis`` [rad], shape ``(..., 3)``.  Plain-autograd
        differentiable, including at the identity.

    Raises:
        ValueError: If any rotation angle exceeds 3.0 rad -- adjacent-tet
            relative rotations near pi are out of physical range.
    """
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = torch.clamp(0.5 * (tr - 1.0), -1.0, 1.0)
    if bool((cos_theta < math.cos(_MAX_THETA)).any()):
        raise ValueError(f"so3_log_axial: rotation angle exceeds {_MAX_THETA} rad (out of physical range)")

    # theta < _SMALL_THETA  <=>  cos_theta > cos(_SMALL_THETA).  On the small
    # branch, feed arccos a dummy 0 so its infinite slope at cos_theta = 1
    # cannot leak NaN into the gradient via 0 * inf in the torch.where backward.
    small = cos_theta > math.cos(_SMALL_THETA)
    theta = torch.acos(torch.where(small, torch.zeros_like(cos_theta), cos_theta))
    factor = torch.where(small, torch.full_like(theta, 0.5), theta / (2.0 * torch.sin(theta)))
    w = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    return factor[..., None] * w
