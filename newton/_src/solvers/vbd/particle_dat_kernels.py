# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ALM-DAT particle self-contact: stored division planes, plane placement,
and final truncation.

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
    collision_info_array: wp.array[TriMeshCollisionInfo],
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
    collision_info = collision_info_array[0]

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
    collision_info_array: wp.array,
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
            collision_info_array,
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
    collision_info_array: wp.array[TriMeshCollisionInfo],
    planes: ParticleContactDatPlanes,
    gamma: float,
    stride: int,
    sigma: wp.array[float],
):
    """Cap every vertex's fraction of ``trial - anchor`` so it stays on its side
    of every valid stored plane (atomic minimum into ``sigma``)."""
    t_id = wp.tid()
    collision_info = collision_info_array[0]

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
    collision_info_array: wp.array,
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
            collision_info_array,
            planes,
            float(gamma),
            launch_size,
        ],
        outputs=[sigma],
        device=model.device,
    )


@wp.kernel
def _commit_truncated_displacement_kernel(
    x_start: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    sigma: wp.array[float],
    budget: float,
    anchor_out: wp.array[wp.vec3],
):
    """Advance the anchor by ``sigma`` of the remaining displacement, keeping the
    total displacement from ``x_start`` within the motion budget."""
    i = wp.tid()
    step = sigma[i] * (trial[i] - anchor[i])
    step_length = wp.length(step)
    used = wp.length(anchor[i] - x_start[i])
    if used + step_length > budget and step_length > 0.0:
        step = step * (wp.max(budget - used, 0.0) / step_length)
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
# Slack-gap ALM contact row
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


@wp.struct
class ParticleContactAlmState:
    """Per-slot contact multipliers plus last step's sorted rows for matching.

    ``vt_lambda`` / ``ee_lambda`` follow the detector's row capacity. The
    ``prev_*`` arrays hold the previous step's CSR offsets, counterpart indices
    (sorted ascending within each row) and multipliers; matching merges them
    with the current sorted rows element by element.
    """

    vt_lambda: wp.array[float]
    ee_lambda: wp.array[float]
    prev_vt_offsets: wp.array[wp.int32]
    prev_vt_counterpart: wp.array[wp.int32]
    prev_vt_lambda: wp.array[float]
    prev_ee_offsets: wp.array[wp.int32]
    prev_ee_counterpart: wp.array[wp.int32]
    prev_ee_lambda: wp.array[float]


def create_particle_contact_alm_state(model, collision_info: TriMeshCollisionInfo, device) -> ParticleContactAlmState:
    """Allocate contact multipliers and matching history for ``collision_info``'s capacities."""
    vt_capacity = max(collision_info.vertex_colliding_triangles.shape[0] // 2, 1)
    ee_capacity = max(collision_info.edge_colliding_edges.shape[0] // 2, 1)
    state = ParticleContactAlmState()
    state.vt_lambda = wp.zeros(vt_capacity, dtype=float, device=device)
    state.ee_lambda = wp.zeros(ee_capacity, dtype=float, device=device)
    state.prev_vt_offsets = wp.zeros(model.particle_count + 1, dtype=wp.int32, device=device)
    state.prev_vt_counterpart = wp.zeros(vt_capacity, dtype=wp.int32, device=device)
    state.prev_vt_lambda = wp.zeros(vt_capacity, dtype=float, device=device)
    state.prev_ee_offsets = wp.zeros(max(model.edge_count, 0) + 1, dtype=wp.int32, device=device)
    state.prev_ee_counterpart = wp.zeros(ee_capacity, dtype=wp.int32, device=device)
    state.prev_ee_lambda = wp.zeros(ee_capacity, dtype=float, device=device)
    return state


@wp.func
def _vertex_triangle_row_geometry(
    v: int,
    tri: int,
    pos: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    n: wp.vec3,
    dt: float,
):
    """Contact direction, signed witness gap, barycentric weights, and the row's inertia metric.

    The force acts along the closest-point direction (the physical contact
    normal, as the penalty path uses); the stored plane normal ``n`` only fixes
    its sign, so a vertex that crossed the triangle reports a negative gap
    instead of a flipped normal. Degenerate zero distance falls back to ``n``.
    """
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]
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
    mobility = inv_mass[v]
    for k in range(3):
        mobility += bary[k] * bary[k] * inv_mass[tri_indices[tri, k]]
    rho = float(0.0)
    if mobility > 0.0:
        rho = 1.0 / (dt * dt * mobility)
    return direction, gap, bary, rho


@wp.func
def _edge_edge_row_geometry(
    e1: int,
    e2: int,
    pos: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    n: wp.vec3,
    dt: float,
    parallel_eps: float,
):
    """Contact direction (e1 positive), signed witness gap, the (s, t) weights, and the inertia metric.

    Same convention as the vertex-triangle row: closest-point direction for the
    force, stored plane normal ``n`` for the sign only.
    """
    e1_a = pos[edge_indices[e1, 2]]
    e1_b = pos[edge_indices[e1, 3]]
    e2_a = pos[edge_indices[e2, 2]]
    e2_b = pos[edge_indices[e2, 3]]
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
    mobility = (
        (1.0 - s) * (1.0 - s) * inv_mass[edge_indices[e1, 2]]
        + s * s * inv_mass[edge_indices[e1, 3]]
        + (1.0 - t) * (1.0 - t) * inv_mass[edge_indices[e2, 2]]
        + t * t * inv_mass[edge_indices[e2, 3]]
    )
    rho = float(0.0)
    if mobility > 0.0:
        rho = 1.0 / (dt * dt * mobility)
    return direction, gap, s, t, rho


@wp.kernel
def _accumulate_alm_contact_force_and_hessian_kernel(
    dt: float,
    current_color: int,
    pos: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
    parallel_eps: float,
    stride: int,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Scatter the slack-gap contact force and Hessian of every valid stored pair
    onto its vertices of the current color (one thread per stored record).
    ``rho_scale`` and ``rho_floor_ratio`` shape the row's ALM metric; ``hard_gap``
    is the thickness at which the hard wall sits above the stored division plane,
    so a loaded contact rests on the wall and never touches the safety plane; the
    cushion spans ``hard_gap`` to ``radius``."""
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        if planes.ee_valid[i] == 1:
            e1 = collision_info.edge_colliding_edges[2 * i]
            e2 = collision_info.edge_colliding_edges[2 * i + 1]
            n, gap, s, t, rho = _edge_edge_row_geometry(
                e1, e2, pos, inv_mass, edge_indices, planes.ee_normal[i], dt, parallel_eps
            )
            force, k = alm_contact_row(
                gap - hard_gap,
                alm.ee_lambda[i],
                stiffness,
                radius - hard_gap,
                contact_alm_rho(rho, stiffness, rho_scale, rho_floor_ratio),
            )
            if force > 0.0 or k > 0.0:
                nn = wp.outer(n, n)
                w0 = 1.0 - s
                w1 = s
                w2 = -(1.0 - t)
                w3 = -t
                v0 = edge_indices[e1, 2]
                v1 = edge_indices[e1, 3]
                v2 = edge_indices[e2, 2]
                v3 = edge_indices[e2, 3]
                if particle_colors[v0] == current_color:
                    wp.atomic_add(particle_forces, v0, w0 * force * n)
                    wp.atomic_add(particle_hessians, v0, w0 * w0 * k * nn)
                if particle_colors[v1] == current_color:
                    wp.atomic_add(particle_forces, v1, w1 * force * n)
                    wp.atomic_add(particle_hessians, v1, w1 * w1 * k * nn)
                if particle_colors[v2] == current_color:
                    wp.atomic_add(particle_forces, v2, w2 * force * n)
                    wp.atomic_add(particle_hessians, v2, w2 * w2 * k * nn)
                if particle_colors[v3] == current_color:
                    wp.atomic_add(particle_forces, v3, w3 * force * n)
                    wp.atomic_add(particle_hessians, v3, w3 * w3 * k * nn)
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            n, gap, bary, rho = _vertex_triangle_row_geometry(
                v, tri, pos, inv_mass, tri_indices, planes.vt_normal[i], dt
            )
            force, k = alm_contact_row(
                gap - hard_gap,
                alm.vt_lambda[i],
                stiffness,
                radius - hard_gap,
                contact_alm_rho(rho, stiffness, rho_scale, rho_floor_ratio),
            )
            if force > 0.0 or k > 0.0:
                nn = wp.outer(n, n)
                if particle_colors[v] == current_color:
                    wp.atomic_add(particle_forces, v, force * n)
                    wp.atomic_add(particle_hessians, v, k * nn)
                for j in range(3):
                    u = tri_indices[tri, j]
                    if particle_colors[u] == current_color:
                        wp.atomic_add(particle_forces, u, -bary[j] * force * n)
                        wp.atomic_add(particle_hessians, u, bary[j] * bary[j] * k * nn)
        i += stride


@wp.kernel
def _update_alm_contact_multipliers_kernel(
    dt: float,
    pos: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    planes: ParticleContactDatPlanes,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
    parallel_eps: float,
    stride: int,
    alm: ParticleContactAlmState,
):
    """Replace every valid pair's multiplier by its transmitted force at ``pos`` (the trial pose)."""
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    ee_offsets = collision_info.edge_colliding_edges_offsets
    ee_count = ee_offsets[ee_offsets.shape[0] - 1]
    i = t_id
    while i < ee_count:
        if planes.ee_valid[i] == 1:
            e1 = collision_info.edge_colliding_edges[2 * i]
            e2 = collision_info.edge_colliding_edges[2 * i + 1]
            _n, gap, _s, _t, rho = _edge_edge_row_geometry(
                e1, e2, pos, inv_mass, edge_indices, planes.ee_normal[i], dt, parallel_eps
            )
            force, _k = alm_contact_row(
                gap - hard_gap,
                alm.ee_lambda[i],
                stiffness,
                radius - hard_gap,
                contact_alm_rho(rho, stiffness, rho_scale, rho_floor_ratio),
            )
            alm.ee_lambda[i] = force
        i += stride

    vt_offsets = collision_info.vertex_colliding_triangles_offsets
    vt_count = vt_offsets[vt_offsets.shape[0] - 1]
    i = t_id
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            _n, gap, _bary, rho = _vertex_triangle_row_geometry(
                v, tri, pos, inv_mass, tri_indices, planes.vt_normal[i], dt
            )
            force, _k = alm_contact_row(
                gap - hard_gap,
                alm.vt_lambda[i],
                stiffness,
                radius - hard_gap,
                contact_alm_rho(rho, stiffness, rho_scale, rho_floor_ratio),
            )
            alm.vt_lambda[i] = force
        i += stride


@wp.kernel
def _match_rows_kernel(
    offsets: wp.array[wp.int32],
    values: wp.array[wp.int32],
    prev_offsets: wp.array[wp.int32],
    prev_counterpart: wp.array[wp.int32],
    prev_lambda: wp.array[float],
    lam: wp.array[float],
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
        if p < p_end and prev_counterpart[p] == counterpart:
            carried = prev_lambda[p]
        lam[slot] = carried


@wp.kernel
def _snapshot_rows_kernel(
    offsets: wp.array[wp.int32],
    values: wp.array[wp.int32],
    lam: wp.array[float],
    prev_offsets: wp.array[wp.int32],
    prev_counterpart: wp.array[wp.int32],
    prev_lambda: wp.array[float],
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


def accumulate_alm_contact_force_and_hessian(
    model,
    dt: float,
    color: int,
    pos: wp.array,
    collision_info_array: wp.array,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    *,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
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
            model.particle_colors,
            model.particle_inv_mass,
            model.tri_indices,
            model.edge_indices,
            collision_info_array,
            planes,
            alm,
            float(stiffness),
            float(radius),
            float(rho_scale),
            float(rho_floor_ratio),
            float(hard_gap),
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
    collision_info_array: wp.array,
    planes: ParticleContactDatPlanes,
    alm: ParticleContactAlmState,
    *,
    stiffness: float,
    radius: float,
    rho_scale: float,
    rho_floor_ratio: float,
    hard_gap: float,
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
            model.particle_inv_mass,
            model.tri_indices,
            model.edge_indices,
            collision_info_array,
            planes,
            float(stiffness),
            float(radius),
            float(rho_scale),
            float(rho_floor_ratio),
            float(hard_gap),
            float(parallel_eps),
            launch_size,
        ],
        outputs=[alm],
        device=model.device,
    )


def match_contact_history(collision_info: TriMeshCollisionInfo, alm: ParticleContactAlmState, device) -> None:
    """Seed the multipliers from the previous step's rows (requires sorted rows)."""
    for offsets, values, prev_offsets, prev_counterpart, prev_lambda, lam in (
        (
            collision_info.vertex_colliding_triangles_offsets,
            collision_info.vertex_colliding_triangles,
            alm.prev_vt_offsets,
            alm.prev_vt_counterpart,
            alm.prev_vt_lambda,
            alm.vt_lambda,
        ),
        (
            collision_info.edge_colliding_edges_offsets,
            collision_info.edge_colliding_edges,
            alm.prev_ee_offsets,
            alm.prev_ee_counterpart,
            alm.prev_ee_lambda,
            alm.ee_lambda,
        ),
    ):
        element_count = offsets.shape[0] - 1
        if element_count <= 0:
            continue
        wp.launch(
            _match_rows_kernel,
            dim=element_count,
            inputs=[offsets, values, prev_offsets, prev_counterpart, prev_lambda],
            outputs=[lam],
            device=device,
        )


def snapshot_contact_history(collision_info: TriMeshCollisionInfo, alm: ParticleContactAlmState, device) -> None:
    """Store the current rows and multipliers for the next step's matching."""
    for offsets, values, lam, prev_offsets, prev_counterpart, prev_lambda in (
        (
            collision_info.vertex_colliding_triangles_offsets,
            collision_info.vertex_colliding_triangles,
            alm.vt_lambda,
            alm.prev_vt_offsets,
            alm.prev_vt_counterpart,
            alm.prev_vt_lambda,
        ),
        (
            collision_info.edge_colliding_edges_offsets,
            collision_info.edge_colliding_edges,
            alm.ee_lambda,
            alm.prev_ee_offsets,
            alm.prev_ee_counterpart,
            alm.prev_ee_lambda,
        ),
    ):
        element_count = offsets.shape[0] - 1
        if element_count <= 0:
            continue
        wp.launch(
            _snapshot_rows_kernel,
            dim=element_count,
            inputs=[offsets, values, lam],
            outputs=[prev_offsets, prev_counterpart, prev_lambda],
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
