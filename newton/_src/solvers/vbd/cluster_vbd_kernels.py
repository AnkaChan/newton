# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the multi-resolution (cluster-affine) coarse step in VBD.

The coarse step adds, before each per-vertex VBD sweep, a Gauss-Seidel sweep over cluster colours.
Each cluster solves a 12-DOF affine increment dq=(dA, dt) and prolongs dx_i = dA r_i + dt to its
(vertex-disjoint) free members. This module provides:

  * the per-triangle stable-Neo-Hookean element Hessian elem_H = area * PSD(d2Psi/dF2)  (6x6),
    matching Newton's evaluate_neo_hookean_membrane_force_hessian energy, and
  * (added incrementally) the per-vertex full-force eval and the cluster_solve gather/solve/prolong.

The 6x6 d2Psi/dF2 is formed by central finite difference of the analytic first Piola and
PSD-projected by an in-kernel cyclic Jacobi eigen-clamp (validated against numpy.linalg.eigh).
"""

from __future__ import annotations

import warp as wp

mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def nh_piola_vec(f0: wp.vec3, f1: wp.vec3, mu: float, lam: float):
    """Stable Neo-Hookean first Piola P (per unit area), packed column-major as vec6
    [P_col0.xyz, P_col1.xyz]. Matches membrane_PK1 / evaluate_neo_hookean_membrane_force_hessian:
    P = mu F + lambda_NH (J_s - alpha) [g0|g1], lambda_NH=lam+mu, alpha=1+mu/lambda_NH."""
    f00 = wp.dot(f0, f0)
    f11 = wp.dot(f1, f1)
    f01 = wp.dot(f0, f1)
    js = wp.sqrt(wp.max(f00 * f11 - f01 * f01, 1.0e-20))
    inv_j = 1.0 / js
    lmbd_nh = lam + mu
    lmbd_safe = wp.sign(lmbd_nh) * wp.max(wp.abs(lmbd_nh), 1.0e-6)
    alpha = 1.0 + mu / lmbd_safe
    g0 = inv_j * (f11 * f0 - f01 * f1)
    g1 = inv_j * (f00 * f1 - f01 * f0)
    s = lmbd_nh * (js - alpha)
    p0 = mu * f0 + s * g0
    p1 = mu * f1 + s * g1
    return vec6(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2])


@wp.func
def nh_dPdF_fd(f0: wp.vec3, f1: wp.vec3, mu: float, lam: float, eps: float):
    """6x6 d2Psi/dF2 (per unit area) by central FD of nh_piola_vec, symmetrized.
    vec(F)/vec(P) are column-major: index 3*col+row (col in {0,1} = F column, row in {0,1,2})."""
    h = mat66(0.0)
    for comp in range(6):
        col = comp // 3
        row = comp % 3
        d = wp.vec3(float(row == 0) * eps, float(row == 1) * eps, float(row == 2) * eps)
        if col == 0:
            pp = nh_piola_vec(f0 + d, f1, mu, lam)
            pm = nh_piola_vec(f0 - d, f1, mu, lam)
        else:
            pp = nh_piola_vec(f0, f1 + d, mu, lam)
            pm = nh_piola_vec(f0, f1 - d, mu, lam)
        cv = (pp - pm) / (2.0 * eps)
        for r in range(6):
            h[r, comp] = cv[r]
    hs = mat66(0.0)
    for i in range(6):
        for j in range(6):
            hs[i, j] = 0.5 * (h[i, j] + h[j, i])
    return hs


@wp.func
def psd_clamp6(a: mat66):
    """PSD-project a symmetric 6x6 (clamp negative eigenvalues to 0) via cyclic Jacobi eigen.
    Givens convention G=[[c,s],[-s,c]] -> zero a_pq with theta=0.5*atan2(2 a_pq, a_qq-a_pp)."""
    v = wp.identity(n=6, dtype=wp.float32)
    h = a
    for _sweep in range(16):
        for p in range(6):
            for q in range(p + 1, 6):
                apq = h[p, q]
                if wp.abs(apq) > 1.0e-14:
                    phi = 0.5 * wp.atan2(2.0 * apq, h[q, q] - h[p, p])
                    c = wp.cos(phi)
                    s = wp.sin(phi)
                    for i in range(6):
                        hip = h[i, p]
                        hiq = h[i, q]
                        h[i, p] = c * hip - s * hiq
                        h[i, q] = s * hip + c * hiq
                    for i in range(6):
                        hpi = h[p, i]
                        hqi = h[q, i]
                        h[p, i] = c * hpi - s * hqi
                        h[q, i] = s * hpi + c * hqi
                    for i in range(6):
                        vip = v[i, p]
                        viq = v[i, q]
                        v[i, p] = c * vip - s * viq
                        v[i, q] = s * vip + c * viq
    r = mat66(0.0)
    for i in range(6):
        lam_i = wp.max(h[i, i], 0.0)
        for a_ in range(6):
            for b_ in range(6):
                r[a_, b_] = r[a_, b_] + lam_i * v[a_, i] * v[b_, i]
    return r


@wp.kernel
def eval_elem_hessian(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_mu: wp.array(dtype=wp.float32),
    tri_lam: wp.array(dtype=wp.float32),
    tri_area: wp.array(dtype=wp.float32),
    eps: wp.float32,
    elem_h: wp.array(dtype=mat66),
):
    """Per-triangle stable-Neo-Hookean element Hessian elem_h[e] = area * PSD(d2Psi/dF2) (6x6)."""
    e = wp.tid()
    v0 = tri_indices[3 * e + 0]
    v1 = tri_indices[3 * e + 1]
    v2 = tri_indices[3 * e + 2]
    x01 = pos[v1] - pos[v0]
    x02 = pos[v2] - pos[v0]
    dm = tri_poses[e]
    f0 = x01 * dm[0, 0] + x02 * dm[1, 0]
    f1 = x01 * dm[0, 1] + x02 * dm[1, 1]
    h6 = nh_dPdF_fd(f0, f1, tri_mu[e], tri_lam[e], eps)
    elem_h[e] = tri_area[e] * psd_clamp6(h6)
