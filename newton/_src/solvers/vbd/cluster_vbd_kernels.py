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

from .particle_vbd_kernels import compute_angle_derivative, compute_normalized_vector_derivative

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
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_mu: wp.array[wp.float32],
    tri_lam: wp.array[wp.float32],
    tri_area: wp.array[wp.float32],
    eps: wp.float32,
    elem_h: wp.array[mat66],
):
    """Per-triangle stable-Neo-Hookean element Hessian elem_h[e] = area * PSD(d2Psi/dF2) (6x6)."""
    e = wp.tid()
    v0 = tri_indices[e, 0]
    v1 = tri_indices[e, 1]
    v2 = tri_indices[e, 2]
    x01 = pos[v1] - pos[v0]
    x02 = pos[v2] - pos[v0]
    dm = tri_poses[e]
    f0 = x01 * dm[0, 0] + x02 * dm[1, 0]
    f1 = x01 * dm[0, 1] + x02 * dm[1, 1]
    h6 = nh_dPdF_fd(f0, f1, tri_mu[e], tri_lam[e], eps)
    elem_h[e] = tri_area[e] * psd_clamp6(h6)


# --------------------------------------------------------------------------- #
# cluster solve: gather g_c + Galerkin H_c, 12x12 SPD solve, prolong
# --------------------------------------------------------------------------- #
mat12 = wp.types.matrix(shape=(12, 12), dtype=wp.float32)
vec12 = wp.types.vector(length=12, dtype=wp.float32)


@wp.func
def cross_block(h6: mat66, ck: wp.vec2, cl: wp.vec2):
    """3x3 element Hessian block J_k^T H6 J_l from per-corner dF/dx coeffs ck,cl (the k=l case is
    the per-vertex block). H6 sub-blocks: [0:3,0:3]=col0col0 ... [3:6,3:6]=col1col1 (column-major)."""
    b = wp.mat33(0.0)
    for a in range(3):
        for c in range(3):
            b[a, c] = (
                ck[0] * cl[0] * h6[a, c]
                + ck[0] * cl[1] * h6[a, 3 + c]
                + ck[1] * cl[0] * h6[3 + a, c]
                + ck[1] * cl[1] * h6[3 + a, 3 + c]
            )
    return b


@wp.func
def accum_ptbp(a: mat12, rk: wp.vec3, rl: wp.vec3, b: wp.mat33):
    """a += P(rk)^T B P(rl). P(r)=[ r^T (x) I3 | I3 ] (3x12), dq layout [dA row-major (9), dt (3)]:
    dx[d] = dA[d,:]·r + dt[d]  ->  P[d, 3d+j]=r[j], P[d, 9+d]=1."""
    for d in range(3):
        for e in range(3):
            bde = b[d, e]
            for j in range(3):
                rkj = rk[j] * bde
                for mm in range(3):
                    a[3 * d + j, 3 * e + mm] = a[3 * d + j, 3 * e + mm] + rkj * rl[mm]
                a[3 * d + j, 9 + e] = a[3 * d + j, 9 + e] + rk[j] * bde
            for mm in range(3):
                a[9 + d, 3 * e + mm] = a[9 + d, 3 * e + mm] + bde * rl[mm]
            a[9 + d, 9 + e] = a[9 + d, 9 + e] + bde
    return a


@wp.func
def accum_ptg(g: vec12, ri: wp.vec3, gi: wp.vec3):
    """g += P(ri)^T gi (the exact restricted gradient term)."""
    for d in range(3):
        for j in range(3):
            g[3 * d + j] = g[3 * d + j] + ri[j] * gi[d]
        g[9 + d] = g[9 + d] + gi[d]
    return g


@wp.func
def solve12_spd(a: mat12, b: vec12, ridge: float):
    """Solve (A + ridge·I) x = b for SPD A via in-register Cholesky L L^T + substitution (n=12)."""
    el = mat12(0.0)
    for j in range(12):
        s = a[j, j] + ridge
        for k in range(j):
            s = s - el[j, k] * el[j, k]
        s = wp.sqrt(wp.max(s, 1.0e-12))
        el[j, j] = s
        invs = 1.0 / s
        for i in range(j + 1, 12):
            t = a[i, j]
            for k in range(j):
                t = t - el[i, k] * el[j, k]
            el[i, j] = t * invs
    y = vec12(0.0)
    for i in range(12):
        s = b[i]
        for k in range(i):
            s = s - el[i, k] * y[k]
        y[i] = s / el[i, i]
    x = vec12(0.0)
    for ii in range(12):
        i = 11 - ii
        s = y[i]
        for k in range(i + 1, 12):
            s = s - el[k, i] * x[k]
        x[i] = s / el[i, i]
    return x


@wp.func
def dihedral_grads(
    edge_idx: int,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[wp.float32],
    stiffness: float,
):
    """Return (dtheta/dx0, dx1, dx2, dx3, k) for one dihedral bending edge; zeros if degenerate.

    Mirrors the geometry in evaluate_dihedral_angle_based_bending_force_hessian (vi0/vi1 are the
    wing tips, vi2/vi3 the shared edge). The per-edge Gauss-Newton bending Hessian is the rank-1
    k * g g^T over the 12-vector g = [dx0;dx1;dx2;dx3]."""
    z = wp.vec3(0.0)
    if edge_indices[edge_idx, 0] == -1 or edge_indices[edge_idx, 1] == -1:
        return z, z, z, z, float(0.0)
    eps = 1.0e-6
    x0 = pos[edge_indices[edge_idx, 0]]
    x1 = pos[edge_indices[edge_idx, 1]]
    x2 = pos[edge_indices[edge_idx, 2]]
    x3 = pos[edge_indices[edge_idx, 3]]
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)
    n1n = wp.length(n1)
    n2n = wp.length(n2)
    en = wp.length(e)
    if n1n < eps or n2n < eps or en < eps:
        return z, z, z, z, float(0.0)
    n1h = n1 / n1n
    n2h = n2 / n2n
    eh = e / en
    sin_t = wp.dot(wp.cross(n1h, n2h), eh)
    cos_t = wp.dot(n1h, n2h)
    skew_e = wp.skew(e)
    skew_x03 = wp.skew(x03)
    skew_x02 = wp.skew(x02)
    skew_x13 = wp.skew(x13)
    skew_x12 = wp.skew(x12)
    skew_n1 = wp.skew(n1h)
    skew_n2 = wp.skew(n2h)
    dn1_dx0 = compute_normalized_vector_derivative(n1n, n1h, skew_e)
    dn2_dx0 = wp.mat33(0.0)
    dn1_dx1 = wp.mat33(0.0)
    dn2_dx1 = compute_normalized_vector_derivative(n2n, n2h, -skew_e)
    dn1_dx2 = compute_normalized_vector_derivative(n1n, n1h, -skew_x03)
    dn2_dx2 = compute_normalized_vector_derivative(n2n, n2h, skew_x13)
    dn1_dx3 = compute_normalized_vector_derivative(n1n, n1h, skew_x02)
    dn2_dx3 = compute_normalized_vector_derivative(n2n, n2h, -skew_x12)
    g0 = compute_angle_derivative(n1h, n2h, eh, dn1_dx0, dn2_dx0, sin_t, cos_t, skew_n1, skew_n2)
    g1 = compute_angle_derivative(n1h, n2h, eh, dn1_dx1, dn2_dx1, sin_t, cos_t, skew_n1, skew_n2)
    g2 = compute_angle_derivative(n1h, n2h, eh, dn1_dx2, dn2_dx2, sin_t, cos_t, skew_n1, skew_n2)
    g3 = compute_angle_derivative(n1h, n2h, eh, dn1_dx3, dn2_dx3, sin_t, cos_t, skew_n1, skew_n2)
    return g0, g1, g2, g3, stiffness * edge_rest_length[edge_idx]


@wp.func
def cluster_assemble_solve(
    c: int,
    clu_vert_offsets: wp.array[wp.int32],
    clu_vert: wp.array[wp.int32],
    clu_vert_r: wp.array[wp.vec3],
    clu_ent_offsets: wp.array[wp.int32],
    ent_tri: wp.array[wp.int32],
    ent_k: wp.array[wp.int32],
    ent_l: wp.array[wp.int32],
    ent_rk: wp.array[wp.int32],
    ent_rl: wp.array[wp.int32],
    tri_coeff: wp.array[wp.vec2],
    elem_h: wp.array[mat66],
    g_vertex: wp.array[wp.vec3],
    minv_dt2: wp.array[wp.float32],
    contact_h: wp.array[wp.mat33],
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[wp.float32],
    edge_bending_properties: wp.array2d[wp.float32],
    bend_offsets: wp.array[wp.int32],
    bend_edge: wp.array[wp.int32],
    bend_r0: wp.array[wp.int32],
    bend_r1: wp.array[wp.int32],
    bend_r2: wp.array[wp.int32],
    bend_r3: wp.array[wp.int32],
    ridge_rel: float,
):
    """Assemble g_c + Galerkin H_c for cluster c and return the 12-DOF affine increment dq."""
    a = mat12(0.0)
    g = vec12(0.0)
    # Galerkin membrane Hessian: element-corner pairs both in c
    for ei in range(clu_ent_offsets[c], clu_ent_offsets[c + 1]):
        tri = ent_tri[ei]
        rk = clu_vert_r[ent_rk[ei]]
        rl = clu_vert_r[ent_rl[ei]]
        ck = tri_coeff[3 * tri + ent_k[ei]]
        cl = tri_coeff[3 * tri + ent_l[ei]]
        a = accum_ptbp(a, rk, rl, cross_block(elem_h[tri], ck, cl))
    # exact gradient + inertia/contact diagonal over members
    for mi in range(clu_vert_offsets[c], clu_vert_offsets[c + 1]):
        vi = clu_vert[mi]
        ri = clu_vert_r[mi]
        g = accum_ptg(g, ri, g_vertex[vi])
        d = minv_dt2[vi] * wp.identity(n=3, dtype=wp.float32) + contact_h[vi]
        a = accum_ptbp(a, ri, ri, d)
    # Galerkin dihedral bending: rank-1 per edge, k * G_c G_c^T with G_c the restricted angle gradient
    for bi in range(bend_offsets[c], bend_offsets[c + 1]):
        ed = bend_edge[bi]
        stiff = edge_bending_properties[ed, 0]
        if stiff > 0.0:
            g0, g1, g2, g3, kk = dihedral_grads(ed, pos, edge_indices, edge_rest_length, stiff)
            gc = vec12(0.0)
            if bend_r0[bi] >= 0:
                gc = accum_ptg(gc, clu_vert_r[bend_r0[bi]], g0)
            if bend_r1[bi] >= 0:
                gc = accum_ptg(gc, clu_vert_r[bend_r1[bi]], g1)
            if bend_r2[bi] >= 0:
                gc = accum_ptg(gc, clu_vert_r[bend_r2[bi]], g2)
            if bend_r3[bi] >= 0:
                gc = accum_ptg(gc, clu_vert_r[bend_r3[bi]], g3)
            for i in range(12):
                for j in range(12):
                    a[i, j] = a[i, j] + kk * gc[i] * gc[j]
    tr = float(0.0)
    for d in range(12):
        tr = tr + a[d, d]
    ridge = ridge_rel * tr / 12.0 + 1.0e-12
    neg = vec12(0.0)
    for d in range(12):
        neg[d] = -g[d]
    return solve12_spd(a, neg, ridge)


@wp.kernel
def cluster_solve(
    color_lo: int,
    color_clusters: wp.array[wp.int32],
    clu_vert_offsets: wp.array[wp.int32],
    clu_vert: wp.array[wp.int32],
    clu_vert_r: wp.array[wp.vec3],
    clu_ent_offsets: wp.array[wp.int32],
    ent_tri: wp.array[wp.int32],
    ent_k: wp.array[wp.int32],
    ent_l: wp.array[wp.int32],
    ent_rk: wp.array[wp.int32],
    ent_rl: wp.array[wp.int32],
    tri_coeff: wp.array[wp.vec2],
    elem_h: wp.array[mat66],
    g_vertex: wp.array[wp.vec3],
    minv_dt2: wp.array[wp.float32],
    contact_h: wp.array[wp.mat33],
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[wp.float32],
    edge_bending_properties: wp.array2d[wp.float32],
    bend_offsets: wp.array[wp.int32],
    bend_edge: wp.array[wp.int32],
    bend_r0: wp.array[wp.int32],
    bend_r1: wp.array[wp.int32],
    bend_r2: wp.array[wp.int32],
    bend_r3: wp.array[wp.int32],
    ridge_rel: float,
    maxstep: float,
    dq_out: wp.array[vec12],
    displacement: wp.array[wp.vec3],
):
    """One thread per cluster (launched per colour). Solve the 12-DOF affine and prolong it to the
    cluster's free members, clamping each per-vertex step to ``maxstep`` (no bias at convergence;
    guards the coarse Newton step against overshoot far from equilibrium). Colours are vertex-
    disjoint -> the displacement writes don't race."""
    c = color_clusters[color_lo + wp.tid()]
    dq = cluster_assemble_solve(
        c,
        clu_vert_offsets,
        clu_vert,
        clu_vert_r,
        clu_ent_offsets,
        ent_tri,
        ent_k,
        ent_l,
        ent_rk,
        ent_rl,
        tri_coeff,
        elem_h,
        g_vertex,
        minv_dt2,
        contact_h,
        pos,
        edge_indices,
        edge_rest_length,
        edge_bending_properties,
        bend_offsets,
        bend_edge,
        bend_r0,
        bend_r1,
        bend_r2,
        bend_r3,
        ridge_rel,
    )
    dq_out[c] = dq
    for mi in range(clu_vert_offsets[c], clu_vert_offsets[c + 1]):
        vi = clu_vert[mi]
        r = clu_vert_r[mi]
        dx = wp.vec3(
            dq[0] * r[0] + dq[1] * r[1] + dq[2] * r[2] + dq[9],
            dq[3] * r[0] + dq[4] * r[1] + dq[5] * r[2] + dq[10],
            dq[6] * r[0] + dq[7] * r[1] + dq[8] * r[2] + dq[11],
        )
        nrm = wp.length(dx)
        if nrm > maxstep:
            dx = dx * (maxstep / nrm)
        displacement[vi] = displacement[vi] + dx


# --------------------------------------------------------------------------- #
# energy evaluation + line-search helpers for the coarse step
# --------------------------------------------------------------------------- #
@wp.func
def nh_energy(f0: wp.vec3, f1: wp.vec3, mu: float, lam: float):
    """Stable Neo-Hookean membrane energy density (per unit area), matching nh_piola_vec."""
    f00 = wp.dot(f0, f0)
    f11 = wp.dot(f1, f1)
    f01 = wp.dot(f0, f1)
    js = wp.sqrt(wp.max(f00 * f11 - f01 * f01, 1.0e-20))
    lmbd_nh = lam + mu
    alpha = 1.0 + mu / (wp.sign(lmbd_nh) * wp.max(wp.abs(lmbd_nh), 1.0e-6))
    # Rest-relative energy (subtract the constant rest density 0.5*lmbd_nh*(1-alpha)^2): the line
    # search uses only energy differences, and removing the large constant offset (~mu^2/2 lmbd_nh
    # per area) keeps the float32 atomic sum from being swamped by it.
    return 0.5 * mu * (f00 + f11 - 2.0) + 0.5 * lmbd_nh * ((js - alpha) * (js - alpha) - (1.0 - alpha) * (1.0 - alpha))


@wp.kernel
def eval_membrane_energy(
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_mu: wp.array[wp.float32],
    tri_lam: wp.array[wp.float32],
    tri_area: wp.array[wp.float32],
    e_out: wp.array[wp.float32],
):
    e = wp.tid()
    dm = tri_poses[e]
    x01 = pos[tri_indices[e, 1]] - pos[tri_indices[e, 0]]
    x02 = pos[tri_indices[e, 2]] - pos[tri_indices[e, 0]]
    f0 = x01 * dm[0, 0] + x02 * dm[1, 0]
    f1 = x01 * dm[0, 1] + x02 * dm[1, 1]
    wp.atomic_add(e_out, 0, tri_area[e] * nh_energy(f0, f1, tri_mu[e], tri_lam[e]))


@wp.kernel
def eval_inertia_energy(
    pos: wp.array[wp.vec3],
    mass: wp.array[wp.float32],
    inertia: wp.array[wp.vec3],
    inv_dt2: float,
    e_out: wp.array[wp.float32],
):
    i = wp.tid()
    d = pos[i] - inertia[i]
    wp.atomic_add(e_out, 0, 0.5 * mass[i] * inv_dt2 * wp.dot(d, d))


@wp.kernel
def eval_bending_energy(
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angle: wp.array[wp.float32],
    edge_rest_length: wp.array[wp.float32],
    edge_bending_properties: wp.array2d[wp.float32],
    e_out: wp.array[wp.float32],
):
    """Dihedral bending energy 0.5*k*(theta - theta_rest)^2, k = stiffness*rest_length."""
    ei = wp.tid()
    stiff = edge_bending_properties[ei, 0]
    if edge_indices[ei, 0] == -1 or edge_indices[ei, 1] == -1 or stiff <= 0.0:
        return
    eps = 1.0e-6
    x0 = pos[edge_indices[ei, 0]]
    x1 = pos[edge_indices[ei, 1]]
    x2 = pos[edge_indices[ei, 2]]
    x3 = pos[edge_indices[ei, 3]]
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    n1n = wp.length(n1)
    n2n = wp.length(n2)
    en = wp.length(e)
    if n1n < eps or n2n < eps or en < eps:
        return
    n1h = n1 / n1n
    n2h = n2 / n2n
    eh = e / en
    theta = wp.atan2(wp.dot(wp.cross(n1h, n2h), eh), wp.dot(n1h, n2h))
    d = theta - edge_rest_angle[ei]
    wp.atomic_add(e_out, 0, 0.5 * stiff * edge_rest_length[ei] * d * d)


@wp.kernel
def axpy_trial(base: wp.array[wp.vec3], s: float, delta: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = base[i] + s * delta[i]


@wp.kernel
def apply_scaled(s: float, delta: wp.array[wp.vec3], pos: wp.array[wp.vec3]):
    i = wp.tid()
    pos[i] = pos[i] + s * delta[i]
