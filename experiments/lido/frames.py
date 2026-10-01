# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cell frames: closest proper rotation of the centre deformation gradient, one float32 Warp kernel.

Route per cell (design spec 4.3): a converged Jacobi eigen-solve of F^T F gives the singular values (descending)
and the smallest right singular direction v3; det(F) < 0 reflects that direction, F <- F (I - 2 v3 v3^T); a scaled
Newton polar iteration (6 steps) gives R. Tie cells (gap <= 1e-4 max(s0, 1), gap = s1 - s2 if inverted else
s1 + s2) redo the eigen-solve on F + 1e-4 max(s0, 1) R_ref. Frames are constants: detached, no autograd.
"""

from __future__ import annotations

import torch
import warp as wp

MAX_SWEEPS = 12
POLAR_STEPS = 6
TIE_EPS = 1e-4


@wp.func
def jacobi_rotate(A: wp.mat33, V: wp.mat33, p: int, q: int):
    """One Jacobi rotation zeroing A[p, q] (A symmetric), accumulated into V."""
    if A[p, q] != 0.0:
        theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
        t = wp.sign(theta) / (wp.abs(theta) + wp.sqrt(theta * theta + 1.0))
        c = 1.0 / wp.sqrt(1.0 + t * t)
        s = t * c
        J = wp.identity(n=3, dtype=float)
        J[p, p] = c
        J[p, q] = s
        J[q, p] = -s
        J[q, q] = c
        A = wp.transpose(J) * A * J
        V = V * J
    return A, V


@wp.func
def jacobi_eigen(A: wp.mat33):
    """Eigen-solve of the symmetric positive semidefinite A = F^T F.

    Returns the singular values of F (sqrt of the eigenvalues, descending) and the matching eigenvectors as the
    columns of V. Cyclic sweeps until the off-diagonal norm is below 1e-7 of the trace, at most MAX_SWEEPS.
    """
    V = wp.identity(n=3, dtype=float)
    tol = 1e-7 * (A[0, 0] + A[1, 1] + A[2, 2])
    for _sweep in range(MAX_SWEEPS):
        off = wp.sqrt(A[0, 1] * A[0, 1] + A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2])
        if off <= tol:
            break
        A, V = jacobi_rotate(A, V, 0, 1)
        A, V = jacobi_rotate(A, V, 0, 2)
        A, V = jacobi_rotate(A, V, 1, 2)
    d = wp.vec3(A[0, 0], A[1, 1], A[2, 2])
    i0 = int(0)
    i1 = int(1)
    i2 = int(2)
    if d[i0] < d[i1]:
        i0, i1 = i1, i0
    if d[i1] < d[i2]:
        i1, i2 = i2, i1
    if d[i0] < d[i1]:
        i0, i1 = i1, i0
    s = wp.vec3(wp.sqrt(wp.max(d[i0], 0.0)), wp.sqrt(wp.max(d[i1], 0.0)), wp.sqrt(wp.max(d[i2], 0.0)))
    Vs = wp.matrix_from_cols(
        wp.vec3(V[0, i0], V[1, i0], V[2, i0]),
        wp.vec3(V[0, i1], V[1, i1], V[2, i1]),
        wp.vec3(V[0, i2], V[1, i2], V[2, i2]),
    )
    return s, Vs


@wp.func
def polar_newton(X: wp.mat33, steps: int):
    """Scaled Newton iteration for the orthogonal polar factor: X <- (g X + X^-T / g) / 2, g = |det X|^(-1/3)."""
    for _step in range(steps):
        g = wp.pow(wp.max(wp.abs(wp.determinant(X)), 1e-30), -1.0 / 3.0)
        X = 0.5 * (g * X + wp.transpose(wp.inverse(X)) / g)
    return X


@wp.kernel
def frames_kernel(F: wp.array[wp.mat33], R_ref: wp.array[wp.mat33], R: wp.array[wp.mat33]):
    i = wp.tid()
    Fi = F[i]
    s, V = jacobi_eigen(wp.transpose(Fi) * Fi)
    inverted = wp.determinant(Fi) < 0.0
    gap = wp.where(inverted, s[1] - s[2], s[1] + s[2])
    if gap <= TIE_EPS * wp.max(s[0], 1.0):  # tie: closest proper rotation of F + eps R_ref, same route
        Fi = Fi + TIE_EPS * wp.max(s[0], 1.0) * R_ref[i]
        s, V = jacobi_eigen(wp.transpose(Fi) * Fi)
        inverted = wp.determinant(Fi) < 0.0
    if inverted:  # reflect the smallest right singular direction
        v3 = wp.vec3(V[0, 2], V[1, 2], V[2, 2])
        Fi = Fi * (wp.identity(n=3, dtype=float) - 2.0 * wp.outer(v3, v3))
    R[i] = polar_newton(Fi, POLAR_STEPS)


def frames(F: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
    """Closest proper rotation of each F [C,3,3] float32; R_ref [C,3,3] breaks ties. Returns R [C,3,3], detached.

    Launched on the calling torch stream, so the result is usable by torch right away without a sync.
    """
    wp.init()
    F = F.detach().contiguous()
    R_ref = R_ref.detach().contiguous()
    R = torch.empty_like(F)
    count = F.shape[0]
    if count == 0:
        return R
    inputs = [wp.from_torch(F, dtype=wp.mat33), wp.from_torch(R_ref, dtype=wp.mat33), wp.from_torch(R, dtype=wp.mat33)]
    if F.is_cuda:
        wp.launch(
            frames_kernel, dim=count, inputs=inputs, stream=wp.stream_from_torch(torch.cuda.current_stream(F.device))
        )
    else:
        wp.launch(frames_kernel, dim=count, inputs=inputs, device="cpu")
    return R
