# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for principal-stretch recovery.

The core operations:

- compute F_e = Ds(x) * Dm^-1 per tet
- polar decompose F = R * S via svd3, with reflection correction
- assemble the global ARAP RHS b for the local-global solver
"""

from __future__ import annotations

import warp as wp


@wp.func
def _polar_RS(F: wp.mat33):
    """Return (R, S) with F = R @ S, R in SO(3), S = S^T."""
    U, sigma, V = wp.svd3(F)
    # Reflection correction: ensure det(R) = +1.
    det_uv = wp.determinant(U @ wp.transpose(V))
    D = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, det_uv)
    R = U @ D @ wp.transpose(V)
    Sigma = wp.mat33(sigma[0], 0.0, 0.0, 0.0, sigma[1], 0.0, 0.0, 0.0, sigma[2] * det_uv)
    S = V @ Sigma @ wp.transpose(V)
    return R, S


@wp.kernel
def compute_F_polar(
    x: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    # outputs
    F_out: wp.array[wp.mat33],
    R_out: wp.array[wp.mat33],
    S_out: wp.array[wp.mat33],
):
    tid = wp.tid()
    i = tet_indices[tid, 0]
    j = tet_indices[tid, 1]
    k = tet_indices[tid, 2]
    l = tet_indices[tid, 3]

    x0 = x[i]
    x10 = x[j] - x0
    x20 = x[k] - x0
    x30 = x[l] - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm_inv = tet_poses[tid]
    F = Ds @ Dm_inv

    R, S = _polar_RS(F)
    F_out[tid] = F
    R_out[tid] = R
    S_out[tid] = S


@wp.kernel
def compute_F_only(
    x: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    F_out: wp.array[wp.mat33],
):
    tid = wp.tid()
    i = tet_indices[tid, 0]
    j = tet_indices[tid, 1]
    k = tet_indices[tid, 2]
    l = tet_indices[tid, 3]
    x0 = x[i]
    Ds = wp.matrix_from_cols(x[j] - x0, x[k] - x0, x[l] - x0)
    F_out[tid] = Ds @ tet_poses[tid]


@wp.kernel
def local_step_R(
    F: wp.array[wp.mat33],
    S_target: wp.array[wp.mat33],
    R_out: wp.array[wp.mat33],
):
    """Local step of ARAP-with-target-stretch.

    R_e = argmin_{R in SO(3)} ||F_e - R * S_e^*||_F^2
        = polar_R(F_e * (S_e^*)^T)
    """
    tid = wp.tid()
    M = F[tid] @ wp.transpose(S_target[tid])
    R, _S = _polar_RS(M)
    R_out[tid] = R


@wp.kernel
def assemble_global_rhs(
    R: wp.array[wp.mat33],
    S_target: wp.array[wp.mat33],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    w: wp.array[wp.float32],
    # output: 3*|V| flattened gradient contributions, atomically accumulated
    rhs: wp.array[wp.vec3],
):
    """Accumulate per-tet contributions to the ARAP global-step RHS.

    The global step solves L x = rhs, where
        L  = sum_e w_e * G_e^T G_e          (constant, assembled on CPU)
        rhs= sum_e w_e * G_e^T (R_e S_e^*)
    G_e = [-1; I] @ (Dm_e^{-1})^T  rewritten as a 3x4 per-tet selector @ Dm_inv^T.

    Concretely, for column c of (R_e S_e^*), with B = Dm_inv^T:
        contribution to vertex j  is  w_e * B[j-1, c] * (R S^*)[:, c]   (j=1..3)
        contribution to vertex 0  is  -sum over j=1..3 of the above.
    """
    tid = wp.tid()
    we = w[tid]
    Dm_inv = tet_poses[tid]
    B = wp.transpose(Dm_inv)  # 3x3
    M = R[tid] @ S_target[tid]  # 3x3, target for F_e

    i0 = tet_indices[tid, 0]
    i1 = tet_indices[tid, 1]
    i2 = tet_indices[tid, 2]
    i3 = tet_indices[tid, 3]

    # F_e(x) = (x_j - x_0) packed as columns @ Dm_inv  (column c uses Dm_inv col c)
    # || F_e - M ||_F^2 expanded in terms of edge vectors gives gradient
    #   dE/d(x_j) = 2 * w_e * sum_c Dm_inv[j-1,c] * ((F_e - M)[:, c])  for j=1..3
    # In the global step (R fixed), the gradient w.r.t. x_j contributes a
    # quadratic form whose RHS piece is  w_e * sum_c Dm_inv[j-1, c] * M[:, c].
    # Equivalently RHS_j = w_e * (M @ Dm_inv[j-1, :]^T) for j=1..3.
    # RHS_0 = - (RHS_1 + RHS_2 + RHS_3).

    r1 = we * (M @ wp.vec3(Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2]))
    r2 = we * (M @ wp.vec3(Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2]))
    r3 = we * (M @ wp.vec3(Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2]))
    # Silence unused-warning on B (kept for clarity in derivation comment).
    _ = B

    wp.atomic_add(rhs, i1, r1)
    wp.atomic_add(rhs, i2, r2)
    wp.atomic_add(rhs, i3, r3)
    wp.atomic_add(rhs, i0, -(r1 + r2 + r3))
