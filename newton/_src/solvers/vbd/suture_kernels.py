# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Suturing in SolverVBD: tunnel point rows, the needle state machine and the contact gate.

A suture needle (see :meth:`newton.ModelBuilder.add_suture_needle`) opens a virtual tunnel through a
tetrahedral soft body: a chain of tunnel points, each a barycentric point of one tet. Every point
carries a hard bilateral row in the two directions transverse to the rod tangent and the capped ALM
friction of body-body contact along the tangent. The design is in ``notes/vbd-suture/vbd-suture.md``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...geometry.soft_contacts_mesh import _soft_contact_force_enabled, _SoftContactFilterData
from ...math import orthonormal_basis
from ...utils.mesh import MeshAdjacencyData, get_vertex_adjacent_tet_id_order, get_vertex_num_adjacent_tets
from .particle_vbd_kernels import accumulate_elasticity_force_and_hessian
from .rigid_vbd_kernels import (
    _alm_relaxed_ascent,
    _attachment_auto_rho,
    _contact_auto_tangent_rho,
    _contact_delassus_w,
    _evaluate_rigid_soft_contact_force_norm,
    _material_force_terms,
    _point_force_to_body_torque_and_hessian,
    _regularized_coulomb_scale,
)

wp.set_module_options({"enable_backward": False})

# Needle states: 0 = no tunnel is growing (a puncture may open one), 1 = a tunnel is growing.
SUTURE_OUTSIDE = wp.constant(0)
SUTURE_PIERCING = wp.constant(1)

# Tunnel states.
TUNNEL_UNUSED = wp.constant(0)
TUNNEL_PIERCING = wp.constant(1)
TUNNEL_THROUGH = wp.constant(2)
TUNNEL_FINISHED = wp.constant(3)

# The face opposite local vertex i of a tet, as local indices.
_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def build_tet_neighbors(tet_indices: np.ndarray) -> np.ndarray:
    """Return the tet across the face opposite each local vertex, or -1 on the boundary.

    Args:
        tet_indices: Tet vertex indices, shape ``[tet_count, 4]``.

    Returns:
        int32 array of shape ``[tet_count, 4]``; entry ``[t, i]`` is the tet that shares the face
        opposite vertex ``i`` of tet ``t``.
    """
    tets = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)
    count = tets.shape[0]
    neighbors = np.full((count, 4), -1, dtype=np.int32)
    if count == 0:
        return neighbors
    faces = np.empty((count * 4, 3), dtype=np.int64)
    for i, (a, b, c) in enumerate(_TET_FACES):
        faces[i::4] = np.sort(tets[:, [a, b, c]], axis=1)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    sorted_faces = faces[order]
    same = np.all(sorted_faces[1:] == sorted_faces[:-1], axis=1)
    first = order[:-1][same]
    second = order[1:][same]
    neighbors[first // 4, first % 4] = second // 4
    neighbors[second // 4, second % 4] = first // 4
    return neighbors


# =====================================================================
# Chapter 1: the tunnel point row
# =====================================================================


@wp.struct
class SutureRodData:
    """The rod curve of every suture needle: one segment per rigid body, tip first, thread after."""

    segment_body: wp.array[wp.int32]
    segment_local_a: wp.array[wp.vec3]
    segment_local_b: wp.array[wp.vec3]
    needle_segment_start: wp.array[wp.int32]
    needle_body: wp.array[wp.int32]
    needle_tip_local: wp.array[wp.vec3]
    needle_group: wp.array[wp.int32]


@wp.struct
class SuturePointData:
    """Tunnel point rows of every suture needle; flat index ``(needle * T + tunnel) * Q + q``."""

    tet: wp.array[wp.int32]
    bary: wp.array[wp.vec4]
    active: wp.array[wp.int32]
    segment: wp.array[wp.int32]
    u: wp.array[wp.float32]
    tangent: wp.array[wp.vec3]
    segment_start: wp.array[wp.int32]
    u_start: wp.array[wp.float32]
    lam: wp.array[wp.vec3]
    lam_t: wp.array[wp.float32]
    rho: wp.array[wp.float32]
    rho_t: wp.array[wp.float32]
    ds: wp.array[wp.float32]
    stiffness: wp.array[wp.float32]
    cap_normal: wp.array[wp.float32]
    tunnel_capacity: int
    point_capacity: int
    friction_transverse_weight: float


@wp.func
def _suture_point_needle(index: int, points: SuturePointData) -> int:
    return index // (points.tunnel_capacity * points.point_capacity)


@wp.func
def _suture_point_is_row(index: int, points: SuturePointData) -> bool:
    return points.tet[index] >= 0 and points.active[index] != 0 and points.segment[index] >= 0


@wp.func
def _barycentric_point(tet: int, bary: wp.vec4, tet_indices: wp.array2d[wp.int32], q: wp.array[wp.vec3]) -> wp.vec3:
    return (
        bary[0] * q[tet_indices[tet, 0]]
        + bary[1] * q[tet_indices[tet, 1]]
        + bary[2] * q[tet_indices[tet, 2]]
        + bary[3] * q[tet_indices[tet, 3]]
    )


@wp.func
def _barycentric_scalar(
    tet: int, bary: wp.vec4, tet_indices: wp.array2d[wp.int32], values: wp.array[wp.float32]
) -> float:
    return (
        bary[0] * values[tet_indices[tet, 0]]
        + bary[1] * values[tet_indices[tet, 1]]
        + bary[2] * values[tet_indices[tet, 2]]
        + bary[3] * values[tet_indices[tet, 3]]
    )


@wp.func
def _segment_local_place(segment: int, u: float, rod: SutureRodData) -> wp.vec3:
    a = rod.segment_local_a[segment]
    b = rod.segment_local_b[segment]
    return a + u * (b - a)


@wp.func
def _segment_world_place(segment: int, u: float, rod: SutureRodData, body_q: wp.array[wp.transform]) -> wp.vec3:
    return wp.transform_point(body_q[rod.segment_body[segment]], _segment_local_place(segment, u, rod))


@wp.func
def _closest_parameter(x: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    d = b - a
    dd = wp.dot(d, d)
    if dd <= 0.0:
        return 0.0
    return wp.clamp(wp.dot(x - a, d) / dd, 0.0, 1.0)


@wp.func
def _series_support(a: float, b: float) -> float:
    """Two supports in series; zero means an immovable (kinematic) side."""
    if a <= 0.0:
        return b
    if b <= 0.0:
        return a
    return a * b / (a + b)


@wp.func
def _material_row_sum_norm(h: wp.mat33) -> float:
    return wp.max(
        wp.abs(h[0, 0]) + wp.abs(h[0, 1]) + wp.abs(h[0, 2]),
        wp.max(
            wp.abs(h[1, 0]) + wp.abs(h[1, 1]) + wp.abs(h[1, 2]),
            wp.abs(h[2, 0]) + wp.abs(h[2, 1]) + wp.abs(h[2, 2]),
        ),
    )


@wp.kernel
def initialize_suture_points(
    dt: float,
    lambda_retention: float,
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    adjacency: MeshAdjacencyData,
    tri_indices: wp.array2d[int],
    tri_poses: wp.array[wp.mat22],
    tri_areas: wp.array[float],
    tri_materials: wp.array2d[float],
    edge_indices: wp.array2d[int],
    edge_rest_angles: wp.array[float],
    edge_rest_lengths: wp.array[float],
    edge_materials: wp.array2d[float],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_structural_k: wp.array[float],
):
    """Once per step: retain the multipliers, fix the step-start rod place, set rho and rho_t.

    The soft support is the inverse mass of the four vertices, each reduced by its material Hessian
    as ``step_particle_body_attachment_lambda_rho`` does, weighted by the squared barycentric
    weights. The body support is its inertial Delassus term in the normal plane plus its structural
    stiffness, as the contact conditioning does (``_contact_body_conditioning_scale``). The row metric
    is the two supports in series, so a light needle held by a stiff drive still gets a firm row.
    """
    i = wp.tid()
    if points.tet[i] < 0 or points.active[i] == 0:
        points.lam[i] = wp.vec3(0.0)
        points.lam_t[i] = 0.0
        points.rho[i] = 0.0
        points.rho_t[i] = 0.0
        return
    segment = points.segment[i]
    if segment < 0:
        return
    points.lam[i] = lambda_retention * points.lam[i]
    points.lam_t[i] = lambda_retention * points.lam_t[i]
    points.segment_start[i] = segment
    points.u_start[i] = points.u[i]
    # The friction cap uses the converged transverse multiplier of the previous step, frozen for this
    # step, as body-body contact freezes its normal: a cap that followed the current multiplier would
    # grow with every iteration while the tissue lags and drag the tissue along the rod.
    points.cap_normal[i] = wp.length(points.lam[i])

    tet = points.tet[i]
    bary = points.bary[i]
    inv_dt_sq = 1.0 / (dt * dt)
    w_soft = float(0.0)
    for k in range(4):
        v = tet_indices[tet, k]
        w_v = particle_inv_mass[v]
        if (particle_flags[v] & ParticleFlags.ACTIVE) == 0:
            w_v = 0.0
        if w_v > 0.0:
            _f, material_h = accumulate_elasticity_force_and_hessian(
                v,
                dt,
                particle_q_prev,
                particle_q,
                tri_indices,
                tri_poses,
                tri_materials,
                tri_areas,
                edge_indices,
                edge_rest_angles,
                edge_rest_lengths,
                edge_materials,
                tet_indices,
                tet_poses,
                tet_materials,
                adjacency,
                wp.vec3(0.0),
                wp.mat33(0.0),
            )
            w_v = w_v / (1.0 + dt * dt * w_v * _material_row_sum_norm(material_h))
        w_soft += bary[k] * bary[k] * w_v
    support_soft = float(0.0)
    if w_soft > 0.0:
        support_soft = inv_dt_sq / w_soft

    t = points.tangent[i]
    n1, n2 = orthonormal_basis(t)
    body = rod.segment_body[segment]
    place_local = _segment_local_place(segment, points.u[i], rod)
    w_body = wp.max(
        _contact_delassus_w(body, place_local, n1, body_q, body_com, body_inv_mass, body_inv_inertia),
        _contact_delassus_w(body, place_local, n2, body_q, body_com, body_inv_mass, body_inv_inertia),
    )
    w_tangent = _contact_delassus_w(body, place_local, t, body_q, body_com, body_inv_mass, body_inv_inertia)
    support_body = float(0.0)
    if w_body > 0.0:
        support_body = inv_dt_sq / w_body + body_structural_k[body]
    support_body_t = float(0.0)
    if w_tangent > 0.0:
        support_body_t = inv_dt_sq / w_tangent + body_structural_k[body]

    rho = _attachment_auto_rho(_series_support(support_soft, support_body), points.stiffness[i])
    points.rho[i] = rho
    points.rho_t[i] = _contact_auto_tangent_rho(
        _series_support(support_soft, support_body_t), rho, body_structural_k[body]
    )


@wp.kernel
def refresh_suture_points(
    window: int,
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
):
    """Every iteration: find each active point's closest place on the rod curve and its tangent.

    A point with no cached segment searches the whole rod curve of its needle. A point whose
    closest place reaches the free end of the thread switches its row off (lifecycle rule 5).
    """
    i = wp.tid()
    if points.tet[i] < 0 or points.active[i] == 0:
        return
    needle = _suture_point_needle(i, points)
    segment_lo = rod.needle_segment_start[needle]
    segment_hi = rod.needle_segment_start[needle + 1]
    if segment_hi <= segment_lo:
        return
    x = _barycentric_point(points.tet[i], points.bary[i], tet_indices, particle_q)
    cached = points.segment[i]
    lo = segment_lo
    hi = segment_hi
    if cached >= 0:
        lo = wp.max(cached - window, segment_lo)
        hi = wp.min(cached + window + 1, segment_hi)
    best_segment = int(-1)
    best_u = float(0.0)
    best_d = float(1.0e30)
    for s in range(lo, hi):
        xf = body_q[rod.segment_body[s]]
        a = wp.transform_point(xf, rod.segment_local_a[s])
        b = wp.transform_point(xf, rod.segment_local_b[s])
        u = _closest_parameter(x, a, b)
        p = a + u * (b - a)
        d = wp.length_sq(x - p)
        if d < best_d:
            best_d = d
            best_segment = s
            best_u = u
    if best_segment < 0:
        return
    if best_segment == segment_hi - 1 and best_u >= 1.0 - 1.0e-6:
        # The free end of the thread passed the point: the row is off for good.
        points.active[i] = 0
        points.lam[i] = wp.vec3(0.0)
        points.lam_t[i] = 0.0
        return
    points.segment[i] = best_segment
    points.u[i] = best_u
    xf = body_q[rod.segment_body[best_segment]]
    a = wp.transform_point(xf, rod.segment_local_a[best_segment])
    b = wp.transform_point(xf, rod.segment_local_b[best_segment])
    points.tangent[i] = wp.normalize(b - a)
    if points.segment_start[i] < 0:
        points.segment_start[i] = best_segment
        points.u_start[i] = best_u


@wp.func
def evaluate_suture_row(
    i: int,
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    grip: wp.array[wp.float32],
    mu: wp.array[wp.float32],
    friction_epsilon: float,
    dt: float,
):
    """Force on the rod, its 3x3 Hessian and the rod place, for one tunnel point row.

    The transverse part is the attachment row projected onto the normal plane of the rod tangent
    (``_material_force_terms``: the default infinite stiffness gives ``k_eff = rho``, a finite
    ``vbd.suture_stiffness`` gives the compliant law). The
    tangential part is the capped ALM friction of body-body contact in one dimension: the force
    ``rho_t * slip + lambda_t`` is clamped to ``mu * (cap_normal + grip * ds)``, with ``cap_normal`` the
    previous step's converged transverse multiplier frozen for the step, and the Hessian is
    ``rho_t`` inside the cap and the regularized sliding metric outside it. The tissue receives the
    opposite force.
    """
    tet = points.tet[i]
    bary = points.bary[i]
    x = _barycentric_point(tet, bary, tet_indices, particle_q)
    x_prev = _barycentric_point(tet, bary, tet_indices, particle_q_prev)
    segment = points.segment[i]
    u = points.u[i]
    t = points.tangent[i]
    p = _segment_world_place(segment, u, rod, body_q)

    projector = wp.identity(n=3, dtype=float) - wp.outer(t, t)
    C = projector * (x - p)
    k_eff, lam_eff = _material_force_terms(points.rho[i], points.stiffness[i], points.lam[i], 1)
    f_perp = k_eff * C + lam_eff
    hessian = k_eff * projector

    segment_start = points.segment_start[i]
    u_start = points.u_start[i]
    p0 = _segment_world_place(segment_start, u_start, rod, body_q)
    p0_prev = _segment_world_place(segment_start, u_start, rod, body_q_prev)
    slip = wp.dot(t, (x - x_prev) - (p0 - p0_prev))

    g = _barycentric_scalar(tet, bary, tet_indices, grip)
    m = _barycentric_scalar(tet, bary, tet_indices, mu)
    cap = m * (points.friction_transverse_weight * points.cap_normal[i] + g * points.ds[i])
    rho_t = points.rho_t[i]
    f_t = rho_t * slip + points.lam_t[i]
    t_outer = wp.outer(t, t)
    if wp.abs(f_t) > cap:
        sliding_metric = cap * _regularized_coulomb_scale(wp.abs(slip), friction_epsilon * dt)
        hessian = hessian + sliding_metric * t_outer
        if f_t > 0.0:
            f_t = cap
        else:
            f_t = -cap
    else:
        hessian = hessian + rho_t * t_outer
    return f_perp + f_t * t, hessian, p


@wp.kernel
def accumulate_suture_row_force_and_hessian(
    dt: float,
    current_color: int,
    particle_colors: wp.array[int],
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    grip: wp.array[wp.float32],
    mu: wp.array[wp.float32],
    friction_epsilon: float,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Particle side, per color: scatter ``-w_i f`` and ``w_i^2 H`` to the colored tet vertices."""
    i = wp.tid()
    if not _suture_point_is_row(i, points):
        return
    force, hessian, _p = evaluate_suture_row(
        i, points, rod, tet_indices, particle_q, particle_q_prev, body_q, body_q_prev, grip, mu, friction_epsilon, dt
    )
    tet = points.tet[i]
    bary = points.bary[i]
    for k in range(4):
        v = tet_indices[tet, k]
        if particle_colors[v] == current_color:
            w = bary[k]
            wp.atomic_add(particle_forces, v, -w * force)
            wp.atomic_add(particle_hessians, v, (w * w) * hessian)


@wp.kernel
def accumulate_suture_rows_per_body(
    dt: float,
    color_group: wp.array[int],
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    grip: wp.array[wp.float32],
    mu: wp.array[wp.float32],
    friction_epsilon: float,
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """Rigid side, per body color: the wrench of every active row whose rod segment this body moves."""
    body = color_group[wp.tid()]
    if body_inv_mass[body] <= 0.0:
        return
    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)
    com_world = wp.transform_point(body_q[body], body_com[body])
    total = points.tet.shape[0]
    for i in range(total):
        if not _suture_point_is_row(i, points):
            continue
        if rod.segment_body[points.segment[i]] != body:
            continue
        force, hessian, p = evaluate_suture_row(
            i,
            points,
            rod,
            tet_indices,
            particle_q,
            particle_q_prev,
            body_q,
            body_q_prev,
            grip,
            mu,
            friction_epsilon,
            dt,
        )
        torque, h_al, h_aa = _point_force_to_body_torque_and_hessian(p - com_world, force, hessian)
        force_acc += force
        torque_acc += torque
        h_ll_acc += hessian
        h_al_acc += h_al
        h_aa_acc += h_aa
    # One thread per body of the color, so a plain read-add-write is enough.
    body_forces[body] = body_forces[body] + force_acc
    body_torques[body] = body_torques[body] + torque_acc
    body_hessian_ll[body] = body_hessian_ll[body] + h_ll_acc
    body_hessian_al[body] = body_hessian_al[body] + h_al_acc
    body_hessian_aa[body] = body_hessian_aa[body] + h_aa_acc


@wp.kernel
def update_duals_suture_points(
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    grip: wp.array[wp.float32],
    mu: wp.array[wp.float32],
):
    """After both passes: relaxed ascent of the transverse multiplier, clamped ascent of the tangential one."""
    i = wp.tid()
    if not _suture_point_is_row(i, points):
        return
    tet = points.tet[i]
    bary = points.bary[i]
    x = _barycentric_point(tet, bary, tet_indices, particle_q)
    x_prev = _barycentric_point(tet, bary, tet_indices, particle_q_prev)
    segment = points.segment[i]
    t = points.tangent[i]
    p = _segment_world_place(segment, points.u[i], rod, body_q)
    projector = wp.identity(n=3, dtype=float) - wp.outer(t, t)
    C = projector * (x - p)
    lam = projector * _alm_relaxed_ascent(points.lam[i], C, points.stiffness[i], points.rho[i])
    points.lam[i] = lam

    p0 = _segment_world_place(points.segment_start[i], points.u_start[i], rod, body_q)
    p0_prev = _segment_world_place(points.segment_start[i], points.u_start[i], rod, body_q_prev)
    slip = wp.dot(t, (x - x_prev) - (p0 - p0_prev))
    g = _barycentric_scalar(tet, bary, tet_indices, grip)
    m = _barycentric_scalar(tet, bary, tet_indices, mu)
    cap = m * (points.friction_transverse_weight * points.cap_normal[i] + g * points.ds[i])
    points.lam_t[i] = wp.clamp(points.lam_t[i] + points.rho_t[i] * slip, -cap, cap)


# =====================================================================
# Chapter 2: needle state, puncture, tunnel points, exit and the contact gate
# =====================================================================

_WALK_EPS = wp.constant(1.0e-6)
_WALK_MAX_STEPS = wp.constant(32)


@wp.func
def _tet_barycentric(tet: int, tet_indices: wp.array2d[wp.int32], q: wp.array[wp.vec3], x: wp.vec3) -> wp.vec4:
    a = q[tet_indices[tet, 0]]
    b = q[tet_indices[tet, 1]]
    c = q[tet_indices[tet, 2]]
    d = q[tet_indices[tet, 3]]
    m = wp.mat33(
        a[0] - d[0],
        b[0] - d[0],
        c[0] - d[0],
        a[1] - d[1],
        b[1] - d[1],
        c[1] - d[1],
        a[2] - d[2],
        b[2] - d[2],
        c[2] - d[2],
    )
    if wp.abs(wp.determinant(m)) < 1.0e-30:
        return wp.vec4(-1.0, -1.0, -1.0, -1.0)
    w = wp.inverse(m) * (x - d)
    return wp.vec4(w[0], w[1], w[2], 1.0 - w[0] - w[1] - w[2])


@wp.func
def _bary_min(b: wp.vec4) -> float:
    return wp.min(wp.min(b[0], b[1]), wp.min(b[2], b[3]))


@wp.func
def _bary_argmin(b: wp.vec4) -> int:
    k = int(0)
    m = b[0]
    for j in range(1, 4):
        if b[j] < m:
            m = b[j]
            k = j
    return k


@wp.func
def _bary_clamp(b: wp.vec4) -> wp.vec4:
    c = wp.vec4(wp.max(b[0], 0.0), wp.max(b[1], 0.0), wp.max(b[2], 0.0), wp.max(b[3], 0.0))
    s = c[0] + c[1] + c[2] + c[3]
    if s <= 0.0:
        return wp.vec4(0.25, 0.25, 0.25, 0.25)
    return c / s


@wp.func
def _segment_distance_sq(x: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    u = _closest_parameter(x, a, b)
    return wp.length_sq(x - (a + u * (b - a)))


@wp.func
def _tet_min_attribute(tet: int, tet_indices: wp.array2d[wp.int32], values: wp.array[wp.float32]) -> float:
    """Smallest per-particle value over the four vertices of ``tet``."""
    v = values[tet_indices[tet, 0]]
    for k in range(1, 4):
        v = wp.min(v, values[tet_indices[tet, k]])
    return v


@wp.func
def _append_suture_point(
    slot: int,
    tet: int,
    bary: wp.vec4,
    spacing: float,
    stiffness: float,
    max_points: int,
    points: SuturePointData,
    tunnel_point_count: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
) -> int:
    """Append one tunnel point to tunnel ``slot``; return the new count (unchanged when full)."""
    count = tunnel_point_count[slot]
    if count >= wp.min(points.point_capacity, max_points):
        overflow[0] = 1
        return count
    i = slot * points.point_capacity + count
    points.tet[i] = tet
    points.bary[i] = bary
    points.active[i] = 1
    points.segment[i] = -1
    points.segment_start[i] = -1
    points.u[i] = 0.0
    points.u_start[i] = 0.0
    points.tangent[i] = wp.vec3(0.0)
    points.lam[i] = wp.vec3(0.0)
    points.lam_t[i] = 0.0
    points.rho[i] = 0.0
    points.rho_t[i] = 0.0
    points.ds[i] = spacing
    points.stiffness[i] = stiffness
    points.cap_normal[i] = 0.0
    tunnel_point_count[slot] = count + 1
    return count + 1


@wp.kernel
def begin_suture_update(interval: int, counter: wp.array[wp.int32], due: wp.array[wp.int32]):
    """Advance the substep counter and decide whether this substep updates the needle state."""
    counter[0] = counter[0] + 1
    if counter[0] % interval == 0:
        due[0] = 1
    else:
        due[0] = 0


@wp.kernel
def accumulate_suture_tip_force(
    due: wp.array[wp.int32],
    shape_to_needle: wp.array[wp.int32],
    contact_filter: _SoftContactFilterData,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_max: int,
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    soft_contact_barycentric: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    contact_penalty_k: wp.array[float],
    use_log_barrier: bool,
    tip_force: wp.array[float],
    tip_record_key: wp.array[wp.int64],
):
    """Sum the normal contact force of every tip record per needle and keep the strongest record.

    The force law is the one the solver applies (``_evaluate_rigid_soft_contact_force_norm`` with the
    record's current penalty stiffness). The strongest record is selected through an int64 key of
    the force in micronewtons and the record index, so the choice is deterministic.
    """
    if due[0] == 0:
        return
    record = wp.tid()
    if record >= wp.min(soft_contact_count[0], soft_contact_max):
        return
    shape = soft_contact_shape[record]
    if shape < 0:
        return
    needle = shape_to_needle[shape]
    if needle < 0:
        return
    if not _soft_contact_force_enabled(record, contact_filter):
        return
    corners = soft_contact_indices[record]
    if corners[0] < 0:
        return
    body = shape_body[shape]
    X_wb = wp.transform_identity()
    if body >= 0:
        X_wb = body_q[body]
    cp_world = wp.transform_point(X_wb, soft_contact_body_pos[record])
    margin = float(0.0)
    if shape_margin.shape[0] > 0:
        margin = shape_margin[shape]
    n = soft_contact_normal[record]
    if corners[1] < 0:
        pos = particle_q[corners[0]]
        radius = particle_radius[corners[0]]
    else:
        bary = soft_contact_barycentric[record]
        pos = bary[0] * particle_q[corners[0]]
        radius = particle_radius[corners[0]]
        if corners[1] >= 0:
            pos += bary[1] * particle_q[corners[1]]
            radius = wp.max(radius, particle_radius[corners[1]])
        if corners[2] >= 0:
            pos += bary[2] * particle_q[corners[2]]
            radius = wp.max(radius, particle_radius[corners[2]])
    distance = wp.dot(n, pos - cp_world)
    collision_radius = radius + margin
    if collision_radius - distance <= 0.0:
        return
    dE_dD, _d2 = _evaluate_rigid_soft_contact_force_norm(
        distance, collision_radius, contact_penalty_k[record], use_log_barrier
    )
    f_n = -dE_dD
    if f_n <= 0.0:
        return
    wp.atomic_add(tip_force, needle, f_n)
    key = (wp.int64(wp.min(f_n * 1.0e6, 2.0e9)) << wp.int64(32)) + wp.int64(record) + wp.int64(1)
    wp.atomic_max(tip_record_key, needle, key)


@wp.kernel
def update_suture_needle_state(
    due: wp.array[wp.int32],
    armed: int,
    spacing: float,
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    particle_q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    puncture_force: wp.array[wp.float32],
    puncture_cos: wp.array[wp.float32],
    row_stiffness: wp.array[wp.float32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_normal: wp.array[wp.vec3],
    needle_max_tunnels: wp.array[wp.int32],
    needle_max_points: wp.array[wp.int32],
    gate_radius: wp.array[wp.float32],
    needle_state: wp.array[wp.int32],
    needle_tunnel: wp.array[wp.int32],
    needle_entered: wp.array[wp.int32],
    tunnel_count: wp.array[wp.int32],
    tunnel_state: wp.array[wp.int32],
    tunnel_point_count: wp.array[wp.int32],
    tip_tet: wp.array[wp.int32],
    last_tip: wp.array[wp.vec3],
    travel: wp.array[wp.float32],
    tip_force: wp.array[float],
    tip_record_key: wp.array[wp.int64],
    overflow: wp.array[wp.int32],
):
    """One thread per needle: puncture, tet walk, point creation and removal, exit and finish."""
    if due[0] == 0:
        return
    n = wp.tid()
    T = points.tunnel_capacity
    Q = points.point_capacity
    xf = body_q[rod.needle_body[n]]
    tip = wp.transform_point(xf, rod.needle_tip_local[n])
    segment_lo = rod.needle_segment_start[n]
    tip_dir = wp.normalize(
        _segment_world_place(segment_lo, 0.0, rod, body_q) - _segment_world_place(segment_lo, 1.0, rod, body_q)
    )
    state = needle_state[n]
    t = needle_tunnel[n]

    if state == SUTURE_OUTSIDE:
        key = tip_record_key[n]
        record = int(-1)
        if key > wp.int64(0):
            record = int((key - wp.int64(1)) & wp.int64(0xFFFFFFFF))
        punctured = bool(False)
        particle = int(-1)
        if record >= 0:
            particle = soft_contact_indices[record][0]
            if armed != 0:
                punctured = tip_force[n] > 0.0
            else:
                punctured = (
                    tip_force[n] >= puncture_force[particle]
                    and wp.dot(tip_dir, soft_contact_normal[record]) >= puncture_cos[particle]
                )
        if punctured and tunnel_count[n] < wp.min(needle_max_tunnels[n], T):
            best_tet = int(-1)
            best_min = float(-1.0e30)
            best_bary = wp.vec4(0.0)
            for k in range(get_vertex_num_adjacent_tets(adjacency, particle)):
                tet, _order = get_vertex_adjacent_tet_id_order(adjacency, particle, k)
                b = _tet_barycentric(tet, tet_indices, particle_q, tip)
                m = _bary_min(b)
                if m > best_min:
                    best_min = m
                    best_tet = tet
                    best_bary = b
            if best_tet >= 0:
                t = tunnel_count[n]
                tunnel_count[n] = t + 1
                slot = n * T + t
                tunnel_state[slot] = TUNNEL_PIERCING
                tunnel_point_count[slot] = 0
                _append_suture_point(
                    slot,
                    best_tet,
                    _bary_clamp(best_bary),
                    spacing,
                    _tet_min_attribute(best_tet, tet_indices, row_stiffness),
                    needle_max_points[n],
                    points,
                    tunnel_point_count,
                    overflow,
                )
                needle_state[n] = SUTURE_PIERCING
                needle_tunnel[n] = t
                needle_entered[n] = 0
                tip_tet[n] = best_tet
                travel[n] = 0.0
    elif state == SUTURE_PIERCING and tip_tet[n] >= 0:
        slot = n * T + t
        tet = tip_tet[n]
        bary = _tet_barycentric(tet, tet_indices, particle_q, tip)
        steps = int(0)
        exited = bool(False)
        inside = _bary_min(bary) >= -_WALK_EPS
        if inside:
            needle_entered[n] = 1
        while _bary_min(bary) < -_WALK_EPS and steps < _WALK_MAX_STEPS:
            face = _bary_argmin(bary)
            neighbor = tet_neighbors[tet, face]
            if neighbor < 0:
                if needle_entered[n] != 0:
                    _append_suture_point(
                        slot,
                        tet,
                        _bary_clamp(bary),
                        spacing,
                        _tet_min_attribute(tet, tet_indices, row_stiffness),
                        needle_max_points[n],
                        points,
                        tunnel_point_count,
                        overflow,
                    )
                    tunnel_state[slot] = TUNNEL_THROUGH
                    needle_state[n] = SUTURE_OUTSIDE  # the tunnel is fixed; the tip may open the next one
                    needle_tunnel[n] = -1
                    exited = True
                break
            tet = neighbor
            bary = _tet_barycentric(tet, tet_indices, particle_q, tip)
            steps += 1
            if _bary_min(bary) >= -_WALK_EPS:
                needle_entered[n] = 1
        if not exited:
            if _bary_min(bary) >= -_WALK_EPS or needle_entered[n] != 0:
                tip_tet[n] = tet
            if needle_entered[n] != 0:
                # Net forward travel since the last point: backward jitter of a dynamic needle subtracts
                # instead of resetting, so a held needle still reaches the spacing; retraction removes points below.
                advance = wp.dot(tip - last_tip[n], tip_dir)
                travel[n] = wp.max(travel[n] + advance, 0.0)
                if travel[n] >= spacing * (1.0 - 1.0e-4):
                    _append_suture_point(
                        slot,
                        tet,
                        _bary_clamp(bary),
                        spacing,
                        _tet_min_attribute(tet, tet_indices, row_stiffness),
                        needle_max_points[n],
                        points,
                        tunnel_point_count,
                        overflow,
                    )
                    travel[n] = 0.0
                count = tunnel_point_count[slot]
                if count > 1:
                    last = slot * Q + count - 1
                    if points.segment[last] == segment_lo and points.u[last] <= 1.0e-6:
                        x_last = _barycentric_point(points.tet[last], points.bary[last], tet_indices, particle_q)
                        if wp.length(x_last - tip) > spacing:
                            points.active[last] = 0
                            points.tet[last] = -1
                            tunnel_point_count[slot] = count - 1
            else:
                # The tip never entered and has moved away from the entry point: abandon the tunnel.
                entry = slot * Q
                x_entry = _barycentric_point(points.tet[entry], points.bary[entry], tet_indices, particle_q)
                if wp.length(x_entry - tip) > spacing + gate_radius[n]:
                    points.active[entry] = 0
                    tunnel_state[slot] = TUNNEL_FINISHED
                    needle_state[n] = SUTURE_OUTSIDE
                    needle_tunnel[n] = -1

    # Every fixed tunnel finishes once no point holds rod anymore (rule 6); it is never reused.
    for tt in range(tunnel_count[n]):
        slot = n * T + tt
        if tunnel_state[slot] != TUNNEL_THROUGH:
            continue
        count = tunnel_point_count[slot]
        any_active = bool(False)
        for q in range(count):
            if points.active[slot * Q + q] != 0:
                any_active = True
        if not any_active:
            tunnel_state[slot] = TUNNEL_FINISHED

    last_tip[n] = tip
    tip_force[n] = 0.0
    tip_record_key[n] = wp.int64(0)


@wp.kernel
def update_suture_gate(
    due: wp.array[wp.int32],
    needle_count: int,
    points: SuturePointData,
    rod: SutureRodData,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    tunnel_state: wp.array[wp.int32],
    tunnel_point_count: wp.array[wp.int32],
    needle_state: wp.array[wp.int32],
    needle_tunnel: wp.array[wp.int32],
    gate_radius: wp.array[wp.float32],
    particle_gate: wp.array[wp.int32],
):
    """One thread per particle: the gate bit of every needle whose tunnel polyline passes within the gate radius."""
    if due[0] == 0:
        return
    p = wp.tid()
    x = particle_q[p]
    T = points.tunnel_capacity
    Q = points.point_capacity
    bits = int(0)
    for n in range(needle_count):
        r2 = gate_radius[n] * gate_radius[n]
        hit = bool(False)
        for t in range(T):
            slot = n * T + t
            st = tunnel_state[slot]
            if st != TUNNEL_PIERCING and st != TUNNEL_THROUGH:
                continue
            count = tunnel_point_count[slot]
            prev = wp.vec3(0.0)
            have_prev = bool(False)
            for q in range(count):
                i = slot * Q + q
                if points.tet[i] < 0:
                    continue
                xq = _barycentric_point(points.tet[i], points.bary[i], tet_indices, particle_q)
                if have_prev:
                    if _segment_distance_sq(x, prev, xq) < r2:
                        hit = True
                elif wp.length_sq(x - xq) < r2:
                    hit = True
                prev = xq
                have_prev = True
            if st == TUNNEL_PIERCING and needle_state[n] == SUTURE_PIERCING and needle_tunnel[n] == t and have_prev:
                tip = wp.transform_point(body_q[rod.needle_body[n]], rod.needle_tip_local[n])
                if _segment_distance_sq(x, prev, tip) < r2:
                    hit = True
        if hit:
            bits |= rod.needle_group[n]
    particle_gate[p] = bits
