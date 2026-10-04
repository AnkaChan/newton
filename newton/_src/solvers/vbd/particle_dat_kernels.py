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

from ...geometry.tri_mesh_collision import TriMeshCollisionInfo
from .rigid_vbd_kernels import (
    dat_separation_epsilon,
    find_edge_edge_separator,
    find_vertex_triangle_separator,
    planar_truncation_t,
)

# room reported when no vertex of a pair approaches its plane
_DAT_INFINITE_ROOM = wp.constant(1.0e30)


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
    normal is kept and only the offset moves. Offsets follow
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
                found, n, _s0, _s1, _gap, _candidate = find_edge_edge_separator(
                    x_e1_a, x_e1_b, x_e2_a, x_e2_b, hint, separation_eps
                )
                valid = int(0)
                if found:
                    valid = int(1)
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
            found, n, _s0, _s1, _gap, _candidate = find_vertex_triangle_separator(
                x_v, x_a, x_b, x_c, hint, separation_eps
            )
            valid = int(0)
            if found:
                valid = int(1)
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
