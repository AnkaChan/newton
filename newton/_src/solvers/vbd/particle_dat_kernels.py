# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ALM-DAT particle self-contact: stored division planes, plane placement,
final truncation, and the ALM contact row with gap-rate damping and Coulomb
friction.

Each stored self-contact record (vertex-triangle or edge-edge, one slot of the
detector's CSR rows) owns one division plane ``(p, n)``. The normal points
from the negative side (triangle, or edge ``e2``) toward the positive side
(vertex, or edge ``e1``). Planes are built at a certified (penetration-free)
anchor pose and only re-placed at later certified poses, so the committed
motion of every vertex can be capped against them without recomputing any
geometry at the trial pose.
"""

from __future__ import annotations

import warp as wp

from ...geometry.kernels import EE_PAIR_OVERFLOW, VT_PAIR_OVERFLOW, triangle_closest_point
from ...geometry.tri_mesh_collision import TriMeshCollisionInfo
from .rigid_vbd_kernels import (
    dat_separation_epsilon,
    find_edge_edge_separator,
    find_vertex_triangle_separator,
    planar_truncation_t,
)

# room reported when no vertex of a pair approaches its plane
_DAT_INFINITE_ROOM = wp.constant(1.0e30)
# below this closest-point distance the contact direction falls back to the plane normal
_CONTACT_DIRECTION_EPS = wp.constant(1.0e-9)


@wp.func
def _dat_plane_room(
    b: float,
    n: wp.vec3,
    origin: wp.vec3,
    negative_pos: wp.mat33,
    negative_disp: wp.mat33,
    negative_count: int,
    positive_pos: wp.mat33,
    positive_disp: wp.mat33,
    positive_count: int,
    margin: float,
    eps: float,
):
    """Smallest fraction of its remaining displacement any approaching vertex can
    take before reaching the plane at offset ``b`` (heights relative to ``origin``)."""
    room = float(_DAT_INFINITE_ROOM)
    for i in range(negative_count):
        rate = wp.dot(n, negative_disp[i])
        if rate > eps:
            height = wp.dot(n, negative_pos[i] - origin)
            room = wp.min(room, (b - margin - height) / rate)
    for j in range(positive_count):
        rate = -wp.dot(n, positive_disp[j])
        if rate > eps:
            height = wp.dot(n, positive_pos[j] - origin)
            room = wp.min(room, (height - margin - b) / rate)
    return room


@wp.func
def dat_plane_offset(
    n: wp.vec3,
    negative_pos: wp.mat33,
    negative_disp: wp.mat33,
    negative_count: int,
    positive_pos: wp.mat33,
    positive_disp: wp.mat33,
    positive_count: int,
    margin: float,
    eps: float,
):
    """Place a division plane along ``n`` between two separated primitives.

    Positions are the committed (certified) pose, displacements the remaining
    motion toward the trial pose. The plane is slid along ``n`` so that the
    slowest approaching vertex on each side can take the same fraction of its
    remaining displacement before touching it. With one approaching vertex per
    side this is the proportional split of the DAT paper. Sides that do not
    approach get no room reserved. Each side keeps at least ``eps`` (the DAT
    separation band) and at most half the current gap, so the thickness margin
    shrinks for pairs already closer than ``2 * margin``.

    Returns ``(plane_point, room, margin_eff)``: a point on the plane, the
    balanced fraction (``_DAT_INFINITE_ROOM`` when nobody approaches), and the
    margin actually reserved on each side.
    """
    origin = negative_pos[0]
    negative_max = float(-_DAT_INFINITE_ROOM)
    for i in range(negative_count):
        negative_max = wp.max(negative_max, wp.dot(n, negative_pos[i] - origin))
    positive_min = float(_DAT_INFINITE_ROOM)
    for j in range(positive_count):
        positive_min = wp.min(positive_min, wp.dot(n, positive_pos[j] - origin))
    gap = positive_min - negative_max
    # reserve at least the separation band on each side (never place the plane on a
    # primitive); fall back to the midpoint when the pair is thinner than the band
    margin_eff = wp.min(wp.max(margin, eps), wp.max(0.0, 0.5 * gap))
    b_lo = negative_max + margin_eff
    b_hi = positive_min - margin_eff

    negative_approaches = int(0)
    for i in range(negative_count):
        if wp.dot(n, negative_disp[i]) > eps:
            negative_approaches = 1
    positive_approaches = int(0)
    for j in range(positive_count):
        if -wp.dot(n, positive_disp[j]) > eps:
            positive_approaches = 1

    best_b = float(0.5 * (b_lo + b_hi))
    best_room = float(_DAT_INFINITE_ROOM)
    if negative_approaches == 1 and positive_approaches == 0:
        best_b = b_hi
    elif positive_approaches == 1 and negative_approaches == 0:
        best_b = b_lo
    if negative_approaches != positive_approaches:
        best_room = _dat_plane_room(
            best_b,
            n,
            origin,
            negative_pos,
            negative_disp,
            negative_count,
            positive_pos,
            positive_disp,
            positive_count,
            margin_eff,
            eps,
        )
    elif negative_approaches == 1 and positive_approaches == 1:
        # the balanced fraction is a concave piecewise-linear function of b whose
        # maximum sits where one negative-side line crosses one positive-side line
        best_b = b_lo
        best_room = _dat_plane_room(
            b_lo,
            n,
            origin,
            negative_pos,
            negative_disp,
            negative_count,
            positive_pos,
            positive_disp,
            positive_count,
            margin_eff,
            eps,
        )
        for i in range(negative_count):
            rate_i = wp.dot(n, negative_disp[i])
            if rate_i > eps:
                height_i = wp.dot(n, negative_pos[i] - origin)
                for j in range(positive_count):
                    rate_j = -wp.dot(n, positive_disp[j])
                    if rate_j > eps:
                        height_j = wp.dot(n, positive_pos[j] - origin)
                        b_ij = (rate_j * (height_i + margin_eff) + rate_i * (height_j - margin_eff)) / (rate_i + rate_j)
                        b_ij = wp.clamp(b_ij, b_lo, b_hi)
                        room_ij = _dat_plane_room(
                            b_ij,
                            n,
                            origin,
                            negative_pos,
                            negative_disp,
                            negative_count,
                            positive_pos,
                            positive_disp,
                            positive_count,
                            margin_eff,
                            eps,
                        )
                        if room_ij > best_room:
                            best_room = room_ij
                            best_b = b_ij
    return origin + best_b * n, best_room, margin_eff


@wp.struct
class ParticleContactDatPlanes:
    """One division plane per stored self-contact record, indexed by the
    detector's CSR row slot. Arrays are sized to each family's row capacity.
    ``*_valid`` is 1 when the slot owns a certified plane (0 for edge-edge
    records stored in the non-canonical direction and for pairs without a
    strict separator at the anchor)."""

    vt_normal: wp.array[wp.vec3]
    vt_point: wp.array[wp.vec3]
    vt_margin: wp.array[float]
    vt_valid: wp.array[wp.int32]
    ee_normal: wp.array[wp.vec3]
    ee_point: wp.array[wp.vec3]
    ee_margin: wp.array[float]
    ee_valid: wp.array[wp.int32]


def create_particle_contact_dat_planes(collision_info: TriMeshCollisionInfo, device) -> ParticleContactDatPlanes:
    """Allocate plane storage matching the row capacities of ``collision_info``."""
    vt_capacity = max(collision_info.vertex_colliding_triangles.shape[0] // 2, 1)
    ee_capacity = max(collision_info.edge_colliding_edges.shape[0] // 2, 1)
    planes = ParticleContactDatPlanes()
    planes.vt_normal = wp.zeros(vt_capacity, dtype=wp.vec3, device=device)
    planes.vt_point = wp.zeros(vt_capacity, dtype=wp.vec3, device=device)
    planes.vt_margin = wp.zeros(vt_capacity, dtype=float, device=device)
    planes.vt_valid = wp.zeros(vt_capacity, dtype=wp.int32, device=device)
    planes.ee_normal = wp.zeros(ee_capacity, dtype=wp.vec3, device=device)
    planes.ee_point = wp.zeros(ee_capacity, dtype=wp.vec3, device=device)
    planes.ee_margin = wp.zeros(ee_capacity, dtype=float, device=device)
    planes.ee_valid = wp.zeros(ee_capacity, dtype=wp.int32, device=device)
    return planes


def _plane_launch_size(planes: ParticleContactDatPlanes, device) -> int:
    capacity = max(planes.vt_normal.shape[0], planes.ee_normal.shape[0], 1)
    if device.is_cuda:
        return min(capacity, device.sm_count * 2048)
    return capacity


@wp.func
def _coordinate_scale(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    scale = float(1.0)
    scale = wp.max(scale, wp.max(wp.abs(a)))
    scale = wp.max(scale, wp.max(wp.abs(b)))
    scale = wp.max(scale, wp.max(wp.abs(c)))
    scale = wp.max(scale, wp.max(wp.abs(d)))
    return scale


@wp.kernel
def _build_contact_planes_kernel(
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
    margin: float,
    refresh_normals: int,
    stride: int,
    planes: ParticleContactDatPlanes,
):
    """Build or re-place one plane per stored record at the certified ``anchor``.

    With ``refresh_normals`` the normal is a certified separator found at the
    anchor (the stored normal serves as a candidate hint); otherwise the stored
    normal is kept and only the offset moves. Slots are re-assigned by every
    detection, so callers must clear the ``*_valid`` flags before the first
    build after a detection; within one detection a plane that can no longer be
    re-certified (pair thinner than the separation band) is kept as stored. Offsets follow
    :func:`dat_plane_offset` using the remaining displacement ``trial - anchor``.
    Edge-edge records are processed in the canonical direction only
    (``e1 < e2``); mirrored records are marked invalid.
    """
    t_id = wp.tid()

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        e1_idx = collision_info.edge_colliding_edges[2 * i]
        e2_idx = collision_info.edge_colliding_edges[2 * i + 1]
        if e1_idx < e2_idx:
            e1_a = edge_indices[e1_idx, 2]
            e1_b = edge_indices[e1_idx, 3]
            e2_a = edge_indices[e2_idx, 2]
            e2_b = edge_indices[e2_idx, 3]
            x_e1_a = anchor[e1_a]
            x_e1_b = anchor[e1_b]
            x_e2_a = anchor[e2_a]
            x_e2_b = anchor[e2_b]
            separation_eps = dat_separation_epsilon(_coordinate_scale(x_e1_a, x_e1_b, x_e2_a, x_e2_b))

            valid = int(planes.ee_valid[i])
            n = planes.ee_normal[i]
            if refresh_normals == 1 or valid == 0:
                hint = wp.vec3(0.0)
                if valid == 1:
                    hint = n
                # n points from e2 (negative) toward e1 (positive)
                found, n_found, _s0, _s1, _gap, _candidate = find_edge_edge_separator(
                    x_e1_a, x_e1_b, x_e2_a, x_e2_b, hint, separation_eps
                )
                if found:
                    n = n_found
                    valid = int(1)
                # not found: a pair thinner than the separation band keeps its stored
                # plane, which still separates the anchor because the committed motion
                # never crossed it; a slot without a stored plane stays unconstrained
            if valid == 1:
                positive = wp.mat33(0.0)
                positive[0] = x_e1_a
                positive[1] = x_e1_b
                positive_disp = wp.mat33(0.0)
                positive_disp[0] = trial[e1_a] - x_e1_a
                positive_disp[1] = trial[e1_b] - x_e1_b
                negative = wp.mat33(0.0)
                negative[0] = x_e2_a
                negative[1] = x_e2_b
                negative_disp = wp.mat33(0.0)
                negative_disp[0] = trial[e2_a] - x_e2_a
                negative_disp[1] = trial[e2_b] - x_e2_b
                point, _room, margin_eff = dat_plane_offset(
                    n, negative, negative_disp, 2, positive, positive_disp, 2, margin, separation_eps
                )
                planes.ee_normal[i] = n
                planes.ee_point[i] = point
                planes.ee_margin[i] = margin_eff
            planes.ee_valid[i] = valid
        else:
            planes.ee_valid[i] = 0
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        v = collision_info.vertex_colliding_triangles[2 * i]
        tri = collision_info.vertex_colliding_triangles[2 * i + 1]
        tri_a = tri_indices[tri, 0]
        tri_b = tri_indices[tri, 1]
        tri_c = tri_indices[tri, 2]
        x_v = anchor[v]
        x_a = anchor[tri_a]
        x_b = anchor[tri_b]
        x_c = anchor[tri_c]
        separation_eps = dat_separation_epsilon(_coordinate_scale(x_v, x_a, x_b, x_c))

        valid = int(planes.vt_valid[i])
        n = planes.vt_normal[i]
        if refresh_normals == 1 or valid == 0:
            hint = wp.vec3(0.0)
            if valid == 1:
                hint = n
            # n points from the triangle (negative) toward the vertex (positive)
            found, n_found, _s0, _s1, _gap, _candidate = find_vertex_triangle_separator(
                x_v, x_a, x_b, x_c, hint, separation_eps
            )
            if found:
                n = n_found
                valid = int(1)
            # not found: keep a stored plane (see the edge-edge branch)
        if valid == 1:
            positive = wp.mat33(0.0)
            positive[0] = x_v
            positive_disp = wp.mat33(0.0)
            positive_disp[0] = trial[v] - x_v
            negative = wp.mat33(0.0)
            negative[0] = x_a
            negative[1] = x_b
            negative[2] = x_c
            negative_disp = wp.mat33(0.0)
            negative_disp[0] = trial[tri_a] - x_a
            negative_disp[1] = trial[tri_b] - x_b
            negative_disp[2] = trial[tri_c] - x_c
            point, _room, margin_eff = dat_plane_offset(
                n, negative, negative_disp, 3, positive, positive_disp, 1, margin, separation_eps
            )
            planes.vt_normal[i] = n
            planes.vt_point[i] = point
            planes.vt_margin[i] = margin_eff
        planes.vt_valid[i] = valid
        i += stride


def build_contact_planes(
    model,
    anchor: wp.array,
    trial: wp.array,
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    *,
    margin: float,
    refresh_normals: bool,
    launch_size: int | None = None,
) -> None:
    """Launch :func:`_build_contact_planes_kernel` over the stored records."""
    if launch_size is None:
        launch_size = _plane_launch_size(planes, model.device)
    wp.launch(
        _build_contact_planes_kernel,
        dim=launch_size,
        inputs=[
            anchor,
            trial,
            model.tri_indices,
            model.edge_indices,
            collision_info,
            float(margin),
            1 if refresh_normals else 0,
            launch_size,
            planes,
        ],
        device=model.device,
    )


@wp.kernel
def _truncate_against_contact_planes_kernel(
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    gamma: float,
    stride: int,
    sigma: wp.array[float],
):
    """Cap every vertex's fraction of ``trial - anchor`` so it stays on its side
    of every valid stored plane (atomic minimum into ``sigma``)."""
    t_id = wp.tid()

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        if planes.ee_valid[i] == 1:
            e1_idx = collision_info.edge_colliding_edges[2 * i]
            e2_idx = collision_info.edge_colliding_edges[2 * i + 1]
            n = planes.ee_normal[i]
            p = planes.ee_point[i]
            m = planes.ee_margin[i]
            for k in range(2, 4):
                v = edge_indices[e1_idx, k]
                t = planar_truncation_t(anchor[v], trial[v] - anchor[v], n, p, gamma, m)
                if t < 1.0:
                    wp.atomic_min(sigma, v, t)
                v = edge_indices[e2_idx, k]
                t = planar_truncation_t(anchor[v], trial[v] - anchor[v], -n, p, gamma, m)
                if t < 1.0:
                    wp.atomic_min(sigma, v, t)
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            n = planes.vt_normal[i]
            p = planes.vt_point[i]
            m = planes.vt_margin[i]
            t = planar_truncation_t(anchor[v], trial[v] - anchor[v], n, p, gamma, m)
            if t < 1.0:
                wp.atomic_min(sigma, v, t)
            for k in range(3):
                u = tri_indices[tri, k]
                t = planar_truncation_t(anchor[u], trial[u] - anchor[u], -n, p, gamma, m)
                if t < 1.0:
                    wp.atomic_min(sigma, u, t)
        i += stride


def truncate_against_contact_planes(
    model,
    anchor: wp.array,
    trial: wp.array,
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    *,
    gamma: float,
    sigma: wp.array,
    launch_size: int | None = None,
) -> None:
    """Fill ``sigma`` with the per-particle fraction caps against the stored planes."""
    if launch_size is None:
        launch_size = _plane_launch_size(planes, model.device)
    sigma.fill_(1.0)
    wp.launch(
        _truncate_against_contact_planes_kernel,
        dim=launch_size,
        inputs=[
            anchor,
            trial,
            model.tri_indices,
            model.edge_indices,
            collision_info,
            planes,
            float(gamma),
            launch_size,
        ],
        outputs=[sigma],
        device=model.device,
    )


@wp.func
def budget_sphere_fraction(offset: wp.vec3, step: wp.vec3, budget: float):
    """Largest fraction of ``step`` whose end point ``offset + s * step`` stays within
    ``budget`` of the origin.

    ``offset`` is the anchor's displacement from the detection pose. A step whose
    end point lies inside the budget sphere is admitted whole, so a step back toward
    the detection pose is always admitted. A step that leaves the sphere is cut at
    the sphere. An anchor already outside the sphere (rounding) may only move to its
    closest approach to the detection pose.
    """
    dd = wp.dot(step, step)
    if dd == 0.0:
        return 1.0
    ad = wp.dot(offset, step)
    aa = wp.dot(offset, offset)
    if aa + 2.0 * ad + dd <= budget * budget:
        return 1.0
    disc = ad * ad - dd * (aa - budget * budget)
    if disc < 0.0:
        return wp.clamp(-ad / dd, 0.0, 1.0)
    return wp.clamp((-ad + wp.sqrt(disc)) / dd, 0.0, 1.0)


@wp.kernel
def _commit_truncated_displacement_kernel(
    x_start: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    sigma: wp.array[float],
    budget: float,
    anchor_out: wp.array[wp.vec3],
):
    """Advance the anchor by ``sigma`` of the remaining displacement, keeping its
    end point within the motion budget of ``x_start``.

    The budget is the same clamp as the penalty DAT path applies to the accumulated
    displacement (``apply_truncation_ts``): only the distance from the detection pose
    counts. A step whose end point stays inside the budget sphere is admitted whole,
    so a correction back toward ``x_start`` is always possible; a step that leaves the
    sphere is cut at the sphere.
    """
    i = wp.tid()
    step = sigma[i] * (trial[i] - anchor[i])
    offset = anchor[i] - x_start[i]
    step = step * budget_sphere_fraction(offset, step, budget)
    anchor_out[i] = anchor[i] + step


def commit_truncated_displacement(
    x_start: wp.array,
    anchor: wp.array,
    trial: wp.array,
    sigma: wp.array,
    *,
    budget: float,
    anchor_out: wp.array,
) -> None:
    """Launch :func:`_commit_truncated_displacement_kernel` over all particles."""
    wp.launch(
        _commit_truncated_displacement_kernel,
        dim=anchor.shape[0],
        inputs=[x_start, anchor, trial, sigma, float(budget)],
        outputs=[anchor_out],
        device=anchor.device,
    )


# ---------------------------------------------------------------------------
# Slack-gap ALM contact row with gap-rate damping and Coulomb friction
# ---------------------------------------------------------------------------


@wp.func
def contact_alm_rho(rho_inertia: float, stiffness: float, rho_scale: float, floor_ratio: float):
    """Effective ALM metric: scaled inertia metric, floored at ``floor_ratio * K``.

    At the plain inertia scale a soft cloth's contact rows have rho far below K,
    so a sweep transmits mostly the stale multiplier through a tiny Hessian and
    the coupled VBD iteration can pump lateral oscillations. The floor keeps the
    in-sweep response at least half penalty-like (s <= 1/2 when the ratio is 1)
    while the converged force stays rho-free.
    """
    return wp.max(rho_inertia * rho_scale, floor_ratio * stiffness)


@wp.func
def cushion_stiffness(gap: float, stiffness: float, rho: float, band: float, wall_ratio: float):
    """Gap-dependent cushion stiffness for a smooth compliant-to-hard transition.

    ``gap`` is the signed gap above the hard wall. Outside the transition band
    (``gap >= band``) the cushion keeps its authored stiffness ``K``. Inside the
    band the stiffness rises along a smoothstep to the wall stiffness
    ``wall_ratio * rho`` at the wall (``gap <= 0``), so the in-sweep stiffness
    ``K rho / (K + rho)`` and the ascent factor ``K / (K + rho)`` approach their
    hard-regime values ``rho`` and one before the slack reaches its bound. The
    wall stiffness stays finite: a stiffness that diverges at the wall would make
    the cushion force diverge and turn the row into a barrier. ``band <= 0``
    disables the transition and returns ``K``. A wall stiffness below ``K`` is
    clamped to ``K``, so the stiffness never drops toward the wall.
    """
    if band <= 0.0:
        return stiffness
    wall = wp.max(wall_ratio * rho, stiffness)
    u = wp.clamp(1.0 - gap / band, 0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)
    return stiffness + (wall - stiffness) * s


@wp.func
def alm_contact_row(gap: float, lam: float, stiffness: float, radius: float, rho: float):
    """Evaluate the cushion-plus-wall contact row for one pair.

    ``gap`` is the signed witness gap along the plane normal (negative past the
    plane), ``lam`` the stored compressive multiplier, ``stiffness`` the cushion
    stiffness ``K``, ``radius`` the cushion width ``R`` and ``rho`` the ALM
    metric. A slack ``z >= 0`` carries the cushion energy ``K/2 max(R - z, 0)^2``
    and ``z = gap`` is enforced by ALM; the slack solve is closed form and
    selects one of three regimes:

    - released (``gap - lam/rho >= R``): no force, no stiffness;
    - cushion (``0 < z``): force ``s lam + k_eff (R - gap)`` with
      ``s = K/(K + rho)``, ``k_eff = K rho/(K + rho)``;
    - hard wall (``z`` pinned at zero): force ``lam - rho gap``, stiffness ``rho``.

    The force is continuous across the regime switch. Returns
    ``(transmitted_force, gap_stiffness)``; the force is nonnegative and the
    multiplier update is ``lam <- transmitted_force`` evaluated at the trial pose.
    """
    if rho <= 0.0:
        return float(0.0), float(0.0)
    z_free = gap - lam / rho
    if z_free >= radius:
        return float(0.0), float(0.0)
    z0 = (stiffness * radius - lam + rho * gap) / (stiffness + rho)
    if z0 >= 0.0:
        s = stiffness / (stiffness + rho)
        k_eff = stiffness * rho / (stiffness + rho)
        return wp.max(s * lam + k_eff * (radius - gap), 0.0), k_eff
    return wp.max(lam - rho * gap, 0.0), rho


@wp.func
def cushion_force(z: float, stiffness: float, radius: float, rho: float, band: float, wall_ratio: float):
    """Cushion force at slack ``z``: the integral of the gap-dependent stiffness
    ``cushion_stiffness`` from ``z`` to the cushion width ``radius``.

    Outside the transition band this is the quadratic cushion ``K (R - z)``. Inside
    the band the smoothstep rise of the stiffness adds its exact integral, so the
    force is continuous, monotone in ``z`` and equals ``K R + (K_wall - K) band / 2``
    at the wall. Zero for ``z >= R``.
    """
    if z >= radius:
        return float(0.0)
    force = stiffness * (radius - z)
    if band > 0.0 and z < band:
        wall = wp.max(wall_ratio * rho, stiffness)
        u = 1.0 - wp.max(z, 0.0) / band
        # integral of the smoothstep 3t^2 - 2t^3 from 0 to u
        q = u * u * u - 0.5 * u * u * u * u
        force += (wall - stiffness) * band * q
    return force


@wp.func
def alm_contact_row_smooth(
    gap: float, lam: float, stiffness: float, radius: float, rho: float, band: float, wall_ratio: float
):
    """``alm_contact_row`` with the smooth compliant-to-hard transition.

    The cushion energy has the gap-dependent stiffness of ``cushion_stiffness``,
    so its force ``F(z)`` is ``cushion_force``. The slack solve ``F(z) = lam + rho (z - gap)``
    on ``z >= 0`` is a monotone scalar equation; ``F`` is convex in ``z``, so a few
    clamped Newton steps from the quadratic estimate converge. Regimes as in
    ``alm_contact_row``: released (``gap - lam/rho >= R``), cushion (``z > 0``) with
    transmitted force ``F(z)`` and gap stiffness ``K(z) rho / (K(z) + rho)``, hard
    wall (``z`` pinned at zero) with force ``lam - rho gap`` and stiffness ``rho``.
    With ``band <= 0`` it reduces to ``alm_contact_row`` exactly.
    """
    if band <= 0.0:
        return alm_contact_row(gap, lam, stiffness, radius, rho)
    if rho <= 0.0:
        return float(0.0), float(0.0)
    z_free = gap - lam / rho
    if z_free >= radius:
        return float(0.0), float(0.0)
    # g(z) = F(z) - lam - rho (z - gap) decreases in z. A negative value at z = 0 means the
    # unconstrained slack is negative: the slack pins at the wall and the wall carries the load.
    g0 = cushion_force(0.0, stiffness, radius, rho, band, wall_ratio) - lam + rho * gap
    if g0 < 0.0:
        return wp.max(lam - rho * gap, 0.0), rho
    # otherwise the root lies in [0, R): start from the authored-stiffness estimate and take
    # clamped Newton steps (F is convex in z, so the iteration converges)
    z = (stiffness * radius - lam + rho * gap) / (stiffness + rho)
    z = wp.clamp(z, 0.0, radius)
    for _it in range(8):
        g = cushion_force(z, stiffness, radius, rho, band, wall_ratio) - lam - rho * (z - gap)
        dg = -cushion_stiffness(z, stiffness, rho, band, wall_ratio) - rho
        z = wp.clamp(z - g / dg, 0.0, radius)
    k_z = cushion_stiffness(z, stiffness, rho, band, wall_ratio)
    return wp.max(cushion_force(z, stiffness, radius, rho, band, wall_ratio), 0.0), k_z * rho / (k_z + rho)


@wp.func
def alm_friction_row(lambda_t: wp.vec3, rho_t: float, rel: wp.vec3, n: wp.vec3, radius: float):
    """Coulomb friction as a tangential multiplier capped by the normal force.

    ``rel`` is the displacement of the positive side relative to the negative
    side since the step start (current barycentric weights on both time levels);
    its tangential part is the slip. The stored multiplier ``lambda_t`` is
    projected into the current tangent plane, advanced by ``rho_t`` times the
    slip, and clamped to the disk of radius ``radius`` (``mu`` times the normal
    force). Returns the clamped multiplier, which is the force on the negative
    side (the positive side receives its negative, so friction opposes the slip),
    and the local Jacobian with respect to the positive side's displacement:
    ``rho_t P`` inside the disk, ``rho_t r / |trial| (P - u u^T)`` on its rim
    and zero when released. Both blocks are positive semidefinite.
    """
    projector = wp.identity(3, float) - wp.outer(n, n)
    if radius <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)
    trial = projector * lambda_t + rho_t * (projector * rel)
    magnitude = wp.length(trial)
    if magnitude <= radius:
        return trial, rho_t * projector
    u = trial / magnitude
    return radius * u, rho_t * (radius / magnitude) * (projector - wp.outer(u, u))


@wp.func
def contact_damping_row(gap: float, normal_rel: float, damping: float, radius: float, dt: float):
    """Gap-rate damping of a closing contact (plain penalty formula, no multiplier).

    ``normal_rel`` is the change of the witness gap since the step start. Only the
    part of the closing motion that happens inside the margin is damped: the rate is
    measured from the margin crossing, ``(gap - min(gap_start, radius)) / dt``, so the
    force is zero before contact, continuous at the margin, and on a landing step
    acts as a dashpot anchored at the margin. A force that switches on abruptly at
    the margin makes the local solve chatter: the Newton step of a strongly damped
    row lands outside the margin, where every contact force vanishes, and the next
    iteration falls back. Returns the force magnitude along the contact direction on
    the positive side and its stiffness ``damping / dt``; zero unless closing.
    """
    if gap >= radius:
        return float(0.0), float(0.0)
    gap_start = gap - normal_rel
    closing = gap - wp.min(gap_start, radius)
    if closing >= -1.0e-9:
        return float(0.0), float(0.0)
    return -damping * closing / dt, damping / dt


@wp.struct
class ParticleContactAlmState:
    """Per-slot contact multipliers plus last step's sorted rows for matching.

    ``vt_lambda`` / ``ee_lambda`` (normal) and ``vt_lambda_t`` / ``ee_lambda_t``
    (tangential, world space) follow the detector's row capacity. The ``prev_*``
    arrays hold the previous step's CSR offsets, counterpart indices (sorted
    ascending within each row) and multipliers; matching merges them with the
    current sorted rows element by element.
    """

    vt_lambda: wp.array[float]
    ee_lambda: wp.array[float]
    vt_lambda_t: wp.array[wp.vec3]
    ee_lambda_t: wp.array[wp.vec3]
    prev_vt_offsets: wp.array[wp.int32]
    prev_vt_counterpart: wp.array[wp.int32]
    prev_vt_lambda: wp.array[float]
    prev_vt_lambda_t: wp.array[wp.vec3]
    prev_ee_offsets: wp.array[wp.int32]
    prev_ee_counterpart: wp.array[wp.int32]
    prev_ee_lambda: wp.array[float]
    prev_ee_lambda_t: wp.array[wp.vec3]


def create_particle_contact_alm_state(model, collision_info: TriMeshCollisionInfo, device) -> ParticleContactAlmState:
    """Allocate contact multipliers and matching history for ``collision_info``'s capacities."""
    vt_capacity = max(collision_info.vertex_colliding_triangles.shape[0] // 2, 1)
    ee_capacity = max(collision_info.edge_colliding_edges.shape[0] // 2, 1)
    state = ParticleContactAlmState()
    state.vt_lambda = wp.zeros(vt_capacity, dtype=float, device=device)
    state.ee_lambda = wp.zeros(ee_capacity, dtype=float, device=device)
    state.vt_lambda_t = wp.zeros(vt_capacity, dtype=wp.vec3, device=device)
    state.ee_lambda_t = wp.zeros(ee_capacity, dtype=wp.vec3, device=device)
    state.prev_vt_offsets = wp.zeros(model.particle_count + 1, dtype=wp.int32, device=device)
    state.prev_vt_counterpart = wp.zeros(vt_capacity, dtype=wp.int32, device=device)
    state.prev_vt_lambda = wp.zeros(vt_capacity, dtype=float, device=device)
    state.prev_vt_lambda_t = wp.zeros(vt_capacity, dtype=wp.vec3, device=device)
    state.prev_ee_offsets = wp.zeros(max(model.edge_count, 0) + 1, dtype=wp.int32, device=device)
    state.prev_ee_counterpart = wp.zeros(ee_capacity, dtype=wp.int32, device=device)
    state.prev_ee_lambda = wp.zeros(ee_capacity, dtype=float, device=device)
    state.prev_ee_lambda_t = wp.zeros(ee_capacity, dtype=wp.vec3, device=device)
    return state


@wp.func
def _vertex_triangle_row_geometry(
    v: int,
    tri: int,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    n: wp.vec3,
    dt: float,
):
    """Contact direction, signed witness gap, barycentric weights, the row's inertia
    metric and the relative displacement of the vertex against its witness since
    the step start.

    The force acts along the closest-point direction (the physical contact
    normal, as the penalty path uses); the stored plane normal ``n`` only fixes
    its sign, so a vertex that crossed the triangle reports a negative gap
    instead of a flipped normal. Degenerate zero distance falls back to ``n``.
    The relative displacement uses the current barycentric weights on both time
    levels, so a changing witness is not counted as motion.
    """
    ia = tri_indices[tri, 0]
    ib = tri_indices[tri, 1]
    ic = tri_indices[tri, 2]
    a = pos[ia]
    b = pos[ib]
    c = pos[ic]
    _closest, bary, _feature = triangle_closest_point(a, b, c, pos[v])
    witness = bary[0] * a + bary[1] * b + bary[2] * c
    diff = pos[v] - witness
    dis = wp.length(diff)
    direction = n
    gap = float(0.0)
    if dis > _CONTACT_DIRECTION_EPS:
        direction = diff / dis
        if wp.dot(direction, n) < 0.0:
            direction = -direction
        gap = wp.dot(direction, diff)
    witness_prev = bary[0] * pos_prev[ia] + bary[1] * pos_prev[ib] + bary[2] * pos_prev[ic]
    rel = (pos[v] - pos_prev[v]) - (witness - witness_prev)
    mobility = inv_mass[v]
    for k in range(3):
        mobility += bary[k] * bary[k] * inv_mass[tri_indices[tri, k]]
    rho = float(0.0)
    if mobility > 0.0:
        rho = 1.0 / (dt * dt * mobility)
    return direction, gap, bary, rho, rel


@wp.func
def _edge_edge_row_geometry(
    e1: int,
    e2: int,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    n: wp.vec3,
    dt: float,
    parallel_eps: float,
):
    """Contact direction (e1 positive), signed witness gap, the (s, t) weights, the
    inertia metric and the relative displacement of the e1 witness against the e2
    witness since the step start.

    Same convention as the vertex-triangle row: closest-point direction for the
    force, stored plane normal ``n`` for the sign only.
    """
    i1a = edge_indices[e1, 2]
    i1b = edge_indices[e1, 3]
    i2a = edge_indices[e2, 2]
    i2b = edge_indices[e2, 3]
    e1_a = pos[i1a]
    e1_b = pos[i1b]
    e2_a = pos[i2a]
    e2_b = pos[i2b]
    st = wp.closest_point_edge_edge(e1_a, e1_b, e2_a, e2_b, parallel_eps)
    s = st[0]
    t = st[1]
    c1 = e1_a + (e1_b - e1_a) * s
    c2 = e2_a + (e2_b - e2_a) * t
    diff = c1 - c2
    dis = wp.length(diff)
    direction = n
    gap = float(0.0)
    if dis > _CONTACT_DIRECTION_EPS:
        direction = diff / dis
        if wp.dot(direction, n) < 0.0:
            direction = -direction
        gap = wp.dot(direction, diff)
    c1_prev = pos_prev[i1a] + (pos_prev[i1b] - pos_prev[i1a]) * s
    c2_prev = pos_prev[i2a] + (pos_prev[i2b] - pos_prev[i2a]) * t
    rel = (c1 - c1_prev) - (c2 - c2_prev)
    mobility = (
        (1.0 - s) * (1.0 - s) * inv_mass[i1a]
        + s * s * inv_mass[i1b]
        + (1.0 - t) * (1.0 - t) * inv_mass[i2a]
        + t * t * inv_mass[i2b]
    )
    rho = float(0.0)
    if mobility > 0.0:
        rho = 1.0 / (dt * dt * mobility)
    return direction, gap, s, t, rho, rel


@wp.func
def _alm_contact_response(
    gap: float,
    rel: wp.vec3,
    n: wp.vec3,
    lam: float,
    lambda_t: wp.vec3,
    rho_inertia: float,
    dt: float,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
    transition_band: float,
    wall_stiffness_ratio: float,
    damping: float,
    friction_mu: float,
):
    """Evaluate one pair's full response at the current pose.

    Returns the normal force magnitude (ALM row plus gap-rate damping), the
    normal stiffness, the friction force on the negative side and its Jacobian,
    and the ALM row's transmitted force alone (the next normal multiplier and the
    friction budget).
    """
    rho = contact_alm_rho(rho_inertia, stiffness, rho_scale, rho_floor_ratio)
    transmitted, k = alm_contact_row_smooth(
        gap - hard_gap, lam, stiffness, radius - hard_gap, rho, transition_band, wall_stiffness_ratio
    )
    normal_force = transmitted
    normal_stiffness = k
    damping_force, damping_stiffness = contact_damping_row(gap, wp.dot(n, rel), damping, radius, dt)
    normal_force += damping_force
    normal_stiffness += damping_stiffness
    friction_force = wp.vec3(0.0)
    friction_jacobian = wp.mat33(0.0)
    if friction_mu > 0.0 and transmitted > 0.0:
        # the tangential metric stays at the inertia scale: with the stiffness floor a
        # row saturates its Coulomb disk after microns of slip and acts as kinetic friction
        friction_force, friction_jacobian = alm_friction_row(
            lambda_t, rho_inertia * rho_scale, rel, n, friction_mu * transmitted
        )
    return normal_force, normal_stiffness, friction_force, friction_jacobian, transmitted


@wp.kernel
def _accumulate_alm_contact_force_and_hessian_kernel(
    dt: float,
    current_color: int,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
    transition_band: float,
    wall_stiffness_ratio: float,
    damping: float,
    friction_mu: float,
    parallel_eps: float,
    stride: int,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Scatter the contact response of every valid stored pair onto its vertices of
    the current color (one thread per stored record): the slack-gap normal row,
    gap-rate damping while the pair is within the margin, and Coulomb friction as
    a tangential multiplier capped at ``friction_mu`` times the row's normal force.
    ``rho_scale`` and ``rho_floor_ratio`` shape the row's ALM metric; ``hard_gap``
    is the thickness at which the hard wall sits above the stored division plane,
    so a loaded contact rests on the wall and never touches the safety plane; the
    cushion spans ``hard_gap`` to ``radius``. A vertex with signed weight ``w``
    receives ``w * normal_force * n - w * friction_force`` and the Hessian
    ``w^2 (normal_stiffness n n^T + friction_jacobian)``."""
    t_id = wp.tid()

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        if planes.ee_valid[i] == 1:
            e1 = collision_info.edge_colliding_edges[2 * i]
            e2 = collision_info.edge_colliding_edges[2 * i + 1]
            n, gap, s, t, rho_inertia, rel = _edge_edge_row_geometry(
                e1, e2, pos, pos_prev, inv_mass, edge_indices, planes.ee_normal[i], dt, parallel_eps
            )
            normal_force, normal_stiffness, friction_force, friction_jacobian, _transmitted = _alm_contact_response(
                gap,
                rel,
                n,
                alm.ee_lambda[i],
                alm.ee_lambda_t[i],
                rho_inertia,
                dt,
                stiffness,
                radius,
                rho_scale,
                rho_floor_ratio,
                hard_gap,
                transition_band,
                wall_stiffness_ratio,
                damping,
                friction_mu,
            )
            if normal_force > 0.0 or normal_stiffness > 0.0 or wp.length_sq(friction_force) > 0.0:
                force_vec = normal_force * n
                hessian = normal_stiffness * wp.outer(n, n) + friction_jacobian
                w0 = 1.0 - s
                w1 = s
                w2 = -(1.0 - t)
                w3 = -t
                v0 = edge_indices[e1, 2]
                v1 = edge_indices[e1, 3]
                v2 = edge_indices[e2, 2]
                v3 = edge_indices[e2, 3]
                if particle_colors[v0] == current_color:
                    wp.atomic_add(particle_forces, v0, w0 * force_vec - w0 * friction_force)
                    wp.atomic_add(particle_hessians, v0, w0 * w0 * hessian)
                if particle_colors[v1] == current_color:
                    wp.atomic_add(particle_forces, v1, w1 * force_vec - w1 * friction_force)
                    wp.atomic_add(particle_hessians, v1, w1 * w1 * hessian)
                if particle_colors[v2] == current_color:
                    wp.atomic_add(particle_forces, v2, w2 * force_vec - w2 * friction_force)
                    wp.atomic_add(particle_hessians, v2, w2 * w2 * hessian)
                if particle_colors[v3] == current_color:
                    wp.atomic_add(particle_forces, v3, w3 * force_vec - w3 * friction_force)
                    wp.atomic_add(particle_hessians, v3, w3 * w3 * hessian)
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            n, gap, bary, rho_inertia, rel = _vertex_triangle_row_geometry(
                v, tri, pos, pos_prev, inv_mass, tri_indices, planes.vt_normal[i], dt
            )
            normal_force, normal_stiffness, friction_force, friction_jacobian, _transmitted = _alm_contact_response(
                gap,
                rel,
                n,
                alm.vt_lambda[i],
                alm.vt_lambda_t[i],
                rho_inertia,
                dt,
                stiffness,
                radius,
                rho_scale,
                rho_floor_ratio,
                hard_gap,
                transition_band,
                wall_stiffness_ratio,
                damping,
                friction_mu,
            )
            if normal_force > 0.0 or normal_stiffness > 0.0 or wp.length_sq(friction_force) > 0.0:
                force_vec = normal_force * n
                hessian = normal_stiffness * wp.outer(n, n) + friction_jacobian
                if particle_colors[v] == current_color:
                    wp.atomic_add(particle_forces, v, force_vec - friction_force)
                    wp.atomic_add(particle_hessians, v, hessian)
                for j in range(3):
                    u = tri_indices[tri, j]
                    if particle_colors[u] == current_color:
                        wp.atomic_add(particle_forces, u, -bary[j] * force_vec + bary[j] * friction_force)
                        wp.atomic_add(particle_hessians, u, bary[j] * bary[j] * hessian)
        i += stride


@wp.kernel
def _update_alm_contact_multipliers_kernel(
    dt: float,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
    transition_band: float,
    wall_stiffness_ratio: float,
    friction_mu: float,
    parallel_eps: float,
    stride: int,
    alm: ParticleContactAlmState,
):
    """Replace every valid pair's normal multiplier by the row's transmitted force at
    ``pos`` (the trial pose), then its tangential multiplier by the capped friction
    trial using the new normal force as the Coulomb budget."""
    t_id = wp.tid()

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        if planes.ee_valid[i] == 1:
            e1 = collision_info.edge_colliding_edges[2 * i]
            e2 = collision_info.edge_colliding_edges[2 * i + 1]
            n, gap, _s, _t, rho_inertia, rel = _edge_edge_row_geometry(
                e1, e2, pos, pos_prev, inv_mass, edge_indices, planes.ee_normal[i], dt, parallel_eps
            )
            rho = contact_alm_rho(rho_inertia, stiffness, rho_scale, rho_floor_ratio)
            force, _k = alm_contact_row_smooth(
                gap - hard_gap,
                alm.ee_lambda[i],
                stiffness,
                radius - hard_gap,
                rho,
                transition_band,
                wall_stiffness_ratio,
            )
            alm.ee_lambda[i] = force
            lambda_t, _jacobian = alm_friction_row(
                alm.ee_lambda_t[i], rho_inertia * rho_scale, rel, n, friction_mu * force
            )
            alm.ee_lambda_t[i] = lambda_t
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            n, gap, _bary, rho_inertia, rel = _vertex_triangle_row_geometry(
                v, tri, pos, pos_prev, inv_mass, tri_indices, planes.vt_normal[i], dt
            )
            rho = contact_alm_rho(rho_inertia, stiffness, rho_scale, rho_floor_ratio)
            force, _k = alm_contact_row_smooth(
                gap - hard_gap,
                alm.vt_lambda[i],
                stiffness,
                radius - hard_gap,
                rho,
                transition_band,
                wall_stiffness_ratio,
            )
            alm.vt_lambda[i] = force
            lambda_t, _jacobian = alm_friction_row(
                alm.vt_lambda_t[i], rho_inertia * rho_scale, rel, n, friction_mu * force
            )
            alm.vt_lambda_t[i] = lambda_t
        i += stride


@wp.kernel
def _match_rows_kernel(
    offsets: wp.array[wp.int32],
    values: wp.array[wp.int32],
    prev_offsets: wp.array[wp.int32],
    prev_counterpart: wp.array[wp.int32],
    prev_lambda: wp.array[float],
    prev_lambda_t: wp.array[wp.vec3],
    lam: wp.array[float],
    lam_t: wp.array[wp.vec3],
):
    """Merge one element's previous and current sorted rows; carry matched multipliers, zero the rest."""
    element = wp.tid()
    start = offsets[element]
    end = offsets[element + 1]
    p = prev_offsets[element]
    p_end = prev_offsets[element + 1]
    for slot in range(start, end):
        counterpart = values[2 * slot + 1]
        while p < p_end and prev_counterpart[p] < counterpart:
            p += 1
        carried = float(0.0)
        carried_t = wp.vec3(0.0)
        if p < p_end and prev_counterpart[p] == counterpart:
            carried = prev_lambda[p]
            carried_t = prev_lambda_t[p]
        lam[slot] = carried
        lam_t[slot] = carried_t


@wp.kernel
def _snapshot_rows_kernel(
    offsets: wp.array[wp.int32],
    values: wp.array[wp.int32],
    lam: wp.array[float],
    lam_t: wp.array[wp.vec3],
    prev_offsets: wp.array[wp.int32],
    prev_counterpart: wp.array[wp.int32],
    prev_lambda: wp.array[float],
    prev_lambda_t: wp.array[wp.vec3],
):
    """Copy one element's current row (counterparts and multipliers) into the history."""
    element = wp.tid()
    start = offsets[element]
    end = offsets[element + 1]
    prev_offsets[element] = start
    if element == offsets.shape[0] - 2:
        prev_offsets[element + 1] = end
    for slot in range(start, end):
        prev_counterpart[slot] = values[2 * slot + 1]
        prev_lambda[slot] = lam[slot]
        prev_lambda_t[slot] = lam_t[slot]


def accumulate_alm_contact_force_and_hessian(
    model,
    dt: float,
    color: int,
    pos: wp.array,
    pos_prev: wp.array,
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    *,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
    transition_band: float = 0.0,
    wall_stiffness_ratio: float = 9.0,
    damping: float,
    friction_mu: float,
    parallel_eps: float,
    launch_size: int,
    particle_forces: wp.array,
    particle_hessians: wp.array,
) -> None:
    """Launch the ALM contact scatter for one color."""
    wp.launch(
        _accumulate_alm_contact_force_and_hessian_kernel,
        dim=launch_size,
        inputs=[
            float(dt),
            int(color),
            pos,
            pos_prev,
            model.particle_colors,
            model.particle_inv_mass,
            model.tri_indices,
            model.edge_indices,
            collision_info,
            planes,
            alm,
            float(stiffness),
            float(radius),
            float(rho_scale),
            float(rho_floor_ratio),
            float(hard_gap),
            float(transition_band),
            float(wall_stiffness_ratio),
            float(damping),
            float(friction_mu),
            float(parallel_eps),
            launch_size,
        ],
        outputs=[particle_forces, particle_hessians],
        device=model.device,
    )


def update_alm_contact_multipliers(
    model,
    dt: float,
    pos: wp.array,
    pos_prev: wp.array,
    collision_info: TriMeshCollisionInfo,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    *,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
    transition_band: float = 0.0,
    wall_stiffness_ratio: float = 9.0,
    friction_mu: float,
    parallel_eps: float,
    launch_size: int,
) -> None:
    """Launch the once-per-sweep multiplier update at the trial pose."""
    wp.launch(
        _update_alm_contact_multipliers_kernel,
        dim=launch_size,
        inputs=[
            float(dt),
            pos,
            pos_prev,
            model.particle_inv_mass,
            model.tri_indices,
            model.edge_indices,
            collision_info,
            planes,
            float(stiffness),
            float(radius),
            float(rho_scale),
            float(rho_floor_ratio),
            float(hard_gap),
            float(transition_band),
            float(wall_stiffness_ratio),
            float(friction_mu),
            float(parallel_eps),
            launch_size,
        ],
        outputs=[alm],
        device=model.device,
    )


def match_contact_history(collision_info: TriMeshCollisionInfo, alm: ParticleContactAlmState, device) -> None:
    """Seed the multipliers from the previous step's rows (requires sorted rows)."""
    for offsets, values, prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t, lam, lam_t in (
        (
            collision_info.vertex_colliding_triangles_offsets,
            collision_info.vertex_colliding_triangles,
            alm.prev_vt_offsets,
            alm.prev_vt_counterpart,
            alm.prev_vt_lambda,
            alm.prev_vt_lambda_t,
            alm.vt_lambda,
            alm.vt_lambda_t,
        ),
        (
            collision_info.edge_colliding_edges_offsets,
            collision_info.edge_colliding_edges,
            alm.prev_ee_offsets,
            alm.prev_ee_counterpart,
            alm.prev_ee_lambda,
            alm.prev_ee_lambda_t,
            alm.ee_lambda,
            alm.ee_lambda_t,
        ),
    ):
        element_count = offsets.shape[0] - 1
        if element_count <= 0:
            continue
        wp.launch(
            _match_rows_kernel,
            dim=element_count,
            inputs=[offsets, values, prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t],
            outputs=[lam, lam_t],
            device=device,
        )


def snapshot_contact_history(collision_info: TriMeshCollisionInfo, alm: ParticleContactAlmState, device) -> None:
    """Store the current rows and multipliers for the next step's matching."""
    for offsets, values, lam, lam_t, prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t in (
        (
            collision_info.vertex_colliding_triangles_offsets,
            collision_info.vertex_colliding_triangles,
            alm.vt_lambda,
            alm.vt_lambda_t,
            alm.prev_vt_offsets,
            alm.prev_vt_counterpart,
            alm.prev_vt_lambda,
            alm.prev_vt_lambda_t,
        ),
        (
            collision_info.edge_colliding_edges_offsets,
            collision_info.edge_colliding_edges,
            alm.ee_lambda,
            alm.ee_lambda_t,
            alm.prev_ee_offsets,
            alm.prev_ee_counterpart,
            alm.prev_ee_lambda,
            alm.prev_ee_lambda_t,
        ),
    ):
        element_count = offsets.shape[0] - 1
        if element_count <= 0:
            continue
        wp.launch(
            _snapshot_rows_kernel,
            dim=element_count,
            inputs=[offsets, values, lam, lam_t],
            outputs=[prev_offsets, prev_counterpart, prev_lambda, prev_lambda_t],
            device=device,
        )


@wp.kernel
def _record_contact_coverage_kernel(global_pair_counts: wp.array[wp.int32], coverage: wp.array[wp.int32]):
    """Coverage is 1 only when neither family dropped records in the last detection."""
    ok = int(1)
    if global_pair_counts[VT_PAIR_OVERFLOW] != 0 or global_pair_counts[EE_PAIR_OVERFLOW] != 0:
        ok = 0
    coverage[0] = ok


def record_contact_coverage(collision_info: TriMeshCollisionInfo, coverage: wp.array, device) -> None:
    """Write the detection's coverage status (graph-safe, no host readback)."""
    wp.launch(
        _record_contact_coverage_kernel,
        dim=1,
        inputs=[collision_info.global_pair_counts],
        outputs=[coverage],
        device=device,
    )
