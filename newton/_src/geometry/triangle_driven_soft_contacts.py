# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Triangle-driven soft-contact generation (water-tight rigid-soft path).

Companion to the legacy per-particle :func:`create_soft_contacts` kernel. The
legacy kernel emits V x surface records (one soft particle vs a rigid surface)
into the particle range ``[0, soft_contact_count[0])``. This module's kernel
adds the E x E and T x V records that per-particle SDF queries cannot see,
writing them into the E/F range
``[soft_contact_count[0], soft_contact_count[0] + soft_contact_count[1])``.

The two kernels are additive: the legacy kernel always runs, and this one runs
only when ``enable_water_tight_rigid_soft_contact`` is set. Both launch on the
same stream so the particle count read here is final.
"""

from __future__ import annotations

import warp as wp

from ..sim.contacts import SOFT_CONTACT_KIND_EDGE, SOFT_CONTACT_KIND_FACE
from .flags import ShapeFlags
from .kernels import closest_point_on_tri_to_segment, triangle_closest_point
from .types import GeoType

_DEGENERATE_EPS = wp.constant(1.0e-9)
"""Below this distance a contact direction is ill-defined; the test is skipped."""

_EDGE_EDGE_EPS = wp.constant(1.0e-6)
"""Parallel-edge tolerance passed to :func:`wp.closest_point_edge_edge`."""


@wp.func
def edge_bary(i: int, s: float):
    """Barycentric on the soft triangle of a point at parameter ``s`` along soft
    edge ``i``.

    Soft edge slots match :attr:`Mesh.tri_edges`: slot 0 = (V0, V1), slot 1 =
    (V1, V2), slot 2 = (V2, V0). ``s`` runs from the first endpoint (0) to the
    second (1).

    Args:
        i: Soft edge slot (0, 1, or 2).
        s: Parameter along the edge in ``[0, 1]``.

    Returns:
        Barycentric coordinates ``(u, v, w)`` on (V0, V1, V2).
    """
    if i == 0:
        return wp.vec3(1.0 - s, s, 0.0)
    if i == 1:
        return wp.vec3(0.0, 1.0 - s, s)
    return wp.vec3(s, 0.0, 1.0 - s)


@wp.func
def _soft_edge_endpoints(i: int, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Endpoints of soft edge slot ``i`` of triangle (a, b, c)."""
    if i == 0:
        return a, b
    if i == 1:
        return b, c
    return c, a


@wp.func
def _box_corner(ci: int, half: wp.vec3):
    """Corner ``ci`` (0..7) of an axis-aligned box with half-extents ``half``.

    Bit 0 selects +/-x, bit 1 selects +/-y, bit 2 selects +/-z.
    """
    fx = 2.0 * float(ci & 1) - 1.0
    fy = 2.0 * float((ci >> 1) & 1) - 1.0
    fz = 2.0 * float((ci >> 2) & 1) - 1.0
    return wp.vec3(fx * half[0], fy * half[1], fz * half[2])


@wp.func
def _emit_into_tri_range(
    # Sizing + offsets:
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    # New-only fields (indexed locally by the E/F count):
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    # Shared fields (indexed at particle_count + local):
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    # Record contents:
    soft_tri_id: wp.int32,
    kind: wp.uint8,
    bary: wp.vec3,
    shape_index: wp.int32,
    body_pos: wp.vec3,
    body_vel: wp.vec3,
    normal: wp.vec3,
):
    """Append one E/F record. New-only fields use the local E/F index ``j``;
    shared fields are packed at ``particle_count + j`` so they sit immediately
    after the legacy particle range.
    """
    j = wp.atomic_add(soft_contact_count, 1, 1)
    idx = particle_count + j
    if idx < soft_contact_max:
        soft_contact_primitive[j] = soft_tri_id
        soft_contact_kind[j] = kind
        soft_contact_barycentric[j] = bary
        soft_contact_shape[idx] = shape_index
        soft_contact_body_pos[idx] = body_pos
        soft_contact_body_vel[idx] = body_vel
        soft_contact_normal[idx] = normal


@wp.func
def _tri_vs_point_emit(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    p: wp.vec3,
    threshold: float,
    soft_tri_id: wp.int32,
    shape_index: wp.int32,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """T x V: rigid 0D feature ``p`` (shape-local) vs the soft triangle face.

    Emits a FACE record when the rigid point is within ``threshold`` of the
    triangle. ``p`` is the rigid contact point; the normal points from the
    rigid feature toward the soft triangle.
    """
    cp, bary, ft = triangle_closest_point(a, b, c, p)
    d = wp.length(cp - p)
    if d > _DEGENERATE_EPS and d < threshold:
        nrm = wp.transform_vector(X_ws, (cp - p) / d)
        _emit_into_tri_range(
            particle_count,
            soft_contact_max,
            soft_contact_count,
            soft_contact_primitive,
            soft_contact_kind,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            soft_tri_id,
            SOFT_CONTACT_KIND_FACE,
            bary,
            shape_index,
            wp.transform_point(X_bs, p),
            wp.vec3(0.0, 0.0, 0.0),
            nrm,
        )


@wp.func
def _tri_vs_sphere_emit(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    center: wp.vec3,
    surf_radius: float,
    margin_radius: float,
    soft_tri_id: wp.int32,
    shape_index: wp.int32,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """T x surface for a sphere-like feature: a point ``center`` carrying an
    offset ``surf_radius`` (sphere, or a cone base disk approximated as a disk
    of that radius). Emits a FACE record at the offset surface point.
    """
    cp, bary, ft = triangle_closest_point(a, b, c, center)
    dc = wp.length(cp - center)
    if dc > _DEGENERATE_EPS and dc < surf_radius + margin_radius:
        direction = (cp - center) / dc
        rpt = center + surf_radius * direction
        nrm = wp.transform_vector(X_ws, direction)
        _emit_into_tri_range(
            particle_count,
            soft_contact_max,
            soft_contact_count,
            soft_contact_primitive,
            soft_contact_kind,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            soft_tri_id,
            SOFT_CONTACT_KIND_FACE,
            bary,
            shape_index,
            wp.transform_point(X_bs, rpt),
            wp.vec3(0.0, 0.0, 0.0),
            nrm,
        )


@wp.func
def _tri_vs_segment_emit(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    p0: wp.vec3,
    p1: wp.vec3,
    surf_radius: float,
    margin_radius: float,
    soft_tri_id: wp.int32,
    shape_index: wp.int32,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """T x surface for a capped axis segment (capsule, or cylinder approximated
    by its axis). The rigid contact point sits ``surf_radius`` off the closest
    axis point toward the triangle. Emits a FACE record.
    """
    cp_tri, bary, cp_seg, dist = closest_point_on_tri_to_segment(a, b, c, p0, p1)
    if dist < surf_radius + margin_radius:
        if dist > _DEGENERATE_EPS:
            direction = (cp_tri - cp_seg) / dist
        else:
            direction = wp.normalize(wp.cross(b - a, c - a))
        rpt = cp_seg + surf_radius * direction
        nrm = wp.transform_vector(X_ws, direction)
        _emit_into_tri_range(
            particle_count,
            soft_contact_max,
            soft_contact_count,
            soft_contact_primitive,
            soft_contact_kind,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            soft_tri_id,
            SOFT_CONTACT_KIND_FACE,
            bary,
            shape_index,
            wp.transform_point(X_bs, rpt),
            wp.vec3(0.0, 0.0, 0.0),
            nrm,
        )


@wp.func
def _edge_vs_edge_emit(
    sa: wp.vec3,
    sb: wp.vec3,
    ra: wp.vec3,
    rb: wp.vec3,
    soft_edge_i: int,
    threshold: float,
    soft_tri_id: wp.int32,
    shape_index: wp.int32,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """E x E: soft edge (sa, sb) vs rigid 1D feature (ra, rb), both shape-local.

    Emits an EDGE record at the closest pair when within ``threshold``. The
    rigid contact point is the closest point on the rigid edge.
    """
    st = wp.closest_point_edge_edge(sa, sb, ra, rb, _EDGE_EDGE_EPS)
    spt = sa + (sb - sa) * st[0]
    rpt = ra + (rb - ra) * st[1]
    d = wp.length(spt - rpt)
    if d > _DEGENERATE_EPS and d < threshold:
        nrm = wp.transform_vector(X_ws, (spt - rpt) / d)
        _emit_into_tri_range(
            particle_count,
            soft_contact_max,
            soft_contact_count,
            soft_contact_primitive,
            soft_contact_kind,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            soft_tri_id,
            SOFT_CONTACT_KIND_EDGE,
            edge_bary(soft_edge_i, st[0]),
            shape_index,
            wp.transform_point(X_bs, rpt),
            wp.vec3(0.0, 0.0, 0.0),
            nrm,
        )


@wp.func
def _edge_vs_segment_emit(
    sa: wp.vec3,
    sb: wp.vec3,
    p0: wp.vec3,
    p1: wp.vec3,
    surf_radius: float,
    margin_radius: float,
    soft_edge_i: int,
    soft_tri_id: wp.int32,
    shape_index: wp.int32,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """E x surface: soft edge (sa, sb) vs a capped axis segment (capsule).

    The rigid contact point sits ``surf_radius`` off the closest axis point
    toward the soft edge. Emits an EDGE record.
    """
    st = wp.closest_point_edge_edge(sa, sb, p0, p1, _EDGE_EDGE_EPS)
    spt = sa + (sb - sa) * st[0]
    apt = p0 + (p1 - p0) * st[1]
    d = wp.length(spt - apt)
    if d > _DEGENERATE_EPS and d < surf_radius + margin_radius:
        direction = (spt - apt) / d
        rpt = apt + surf_radius * direction
        nrm = wp.transform_vector(X_ws, direction)
        _emit_into_tri_range(
            particle_count,
            soft_contact_max,
            soft_contact_count,
            soft_contact_primitive,
            soft_contact_kind,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            soft_tri_id,
            SOFT_CONTACT_KIND_EDGE,
            edge_bary(soft_edge_i, st[0]),
            shape_index,
            wp.transform_point(X_bs, rpt),
            wp.vec3(0.0, 0.0, 0.0),
            nrm,
        )


@wp.func
def _process_box_shape(
    soft_tri_id: wp.int32,
    soft_flag: wp.uint8,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_index: wp.int32,
    geo_scale: wp.vec3,
    margin: float,
    radius: float,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Box: 8 corners (T x V) and 12 edges (E x E). Box features are unique, so
    no rigid-side ownership dedup is needed.
    """
    mr = margin + radius

    # T x V: 8 corners.
    for ci in range(8):
        corner = _box_corner(ci, geo_scale)
        _tri_vs_point_emit(
            a, b, c, corner, mr, soft_tri_id, shape_index, X_ws, X_bs,
            particle_count, soft_contact_max, soft_contact_count,
            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
        )

    # E x E: 12 box edges (corners differing in exactly one axis bit).
    for axis in range(3):
        bit = 1 << axis
        for ci in range(8):
            if (ci & bit) == 0:
                corner_i = _box_corner(ci, geo_scale)
                corner_j = _box_corner(ci | bit, geo_scale)
                for i in range(3):
                    if (int(soft_flag) >> (i + 3)) & 1:
                        sa, sb = _soft_edge_endpoints(i, a, b, c)
                        _edge_vs_edge_emit(
                            sa, sb, corner_i, corner_j, i, mr, soft_tri_id, shape_index, X_ws, X_bs,
                            particle_count, soft_contact_max, soft_contact_count,
                            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
                            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
                        )


@wp.func
def _process_sphere_shape(
    soft_tri_id: wp.int32,
    soft_flag: wp.uint8,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_index: wp.int32,
    geo_scale: wp.vec3,
    margin: float,
    radius: float,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Sphere: a single T x surface record (no 1D feature)."""
    _tri_vs_sphere_emit(
        a, b, c, wp.vec3(0.0, 0.0, 0.0), geo_scale[0], margin + radius,
        soft_tri_id, shape_index, X_ws, X_bs,
        particle_count, soft_contact_max, soft_contact_count,
        soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
        soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
    )


@wp.func
def _process_capsule_shape(
    soft_tri_id: wp.int32,
    soft_flag: wp.uint8,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_index: wp.int32,
    geo_scale: wp.vec3,
    margin: float,
    radius: float,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Capsule (Axis.Z, scale = (radius, half_height, _)): T x surface vs the
    axis segment plus E x surface for each soft owned edge.
    """
    mr = margin + radius
    capsule_radius = geo_scale[0]
    p0 = wp.vec3(0.0, 0.0, geo_scale[1])
    p1 = wp.vec3(0.0, 0.0, -geo_scale[1])

    _tri_vs_segment_emit(
        a, b, c, p0, p1, capsule_radius, mr, soft_tri_id, shape_index, X_ws, X_bs,
        particle_count, soft_contact_max, soft_contact_count,
        soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
        soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
    )

    for i in range(3):
        if (int(soft_flag) >> (i + 3)) & 1:
            sa, sb = _soft_edge_endpoints(i, a, b, c)
            _edge_vs_segment_emit(
                sa, sb, p0, p1, capsule_radius, mr, i, soft_tri_id, shape_index, X_ws, X_bs,
                particle_count, soft_contact_max, soft_contact_count,
                soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
                soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
            )


@wp.func
def _process_cylinder_shape(
    soft_tri_id: wp.int32,
    soft_flag: wp.uint8,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_index: wp.int32,
    geo_scale: wp.vec3,
    margin: float,
    radius: float,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Cylinder (Axis.Z, scale = (cap_radius, half_height, _)): T x surface only,
    approximating the body by its axis segment. E x lateral is deferred.
    """
    _tri_vs_segment_emit(
        a, b, c, wp.vec3(0.0, 0.0, geo_scale[1]), wp.vec3(0.0, 0.0, -geo_scale[1]),
        geo_scale[0], margin + radius, soft_tri_id, shape_index, X_ws, X_bs,
        particle_count, soft_contact_max, soft_contact_count,
        soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
        soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
    )


@wp.func
def _process_cone_shape(
    soft_tri_id: wp.int32,
    soft_flag: wp.uint8,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_index: wp.int32,
    geo_scale: wp.vec3,
    margin: float,
    radius: float,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Cone (Axis.Z, scale = (base_radius, half_height, _)): apex (0D) and base
    disk (approximated as a disk of base_radius) are geometrically distinct, so
    they run as two independent T tests. Both can fire. E x lateral deferred.
    """
    mr = margin + radius
    apex = wp.vec3(0.0, 0.0, geo_scale[1])
    base_centre = wp.vec3(0.0, 0.0, -geo_scale[1])

    _tri_vs_point_emit(
        a, b, c, apex, mr, soft_tri_id, shape_index, X_ws, X_bs,
        particle_count, soft_contact_max, soft_contact_count,
        soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
        soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
    )
    _tri_vs_sphere_emit(
        a, b, c, base_centre, geo_scale[0], mr, soft_tri_id, shape_index, X_ws, X_bs,
        particle_count, soft_contact_max, soft_contact_count,
        soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
        soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
    )


@wp.func
def _process_mesh_shape(
    soft_tri_id: wp.int32,
    soft_flag: wp.uint8,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_index: wp.int32,
    geo_scale: wp.vec3,
    mesh_id: wp.uint64,
    shape_mesh_tri_edges: wp.array2d[wp.int32],
    shape_mesh_tri_feature_owner_flag: wp.array[wp.uint8],
    shape_mesh_ownership_range: wp.array[wp.vec3i],
    mesh_edge_indices: wp.array[wp.vec2i],
    shape_edge_range: wp.array[wp.vec2i],
    margin: float,
    radius: float,
    X_ws: wp.transform,
    X_bs: wp.transform,
    particle_count: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Triangulated rigid mesh. Inflate the soft triangle AABB by ``margin +
    radius``, walk the mesh BVH, and for each hit rigid triangle test only the
    rigid features it owns (flag bits 0-2 = vertices, bits 3-5 = edges). Owner
    gating guarantees each shared rigid vertex/edge is tested by exactly one
    rigid triangle.
    """
    rng = shape_mesh_ownership_range[shape_index]
    tri_start = rng[0]
    edge_start = rng[2]
    if tri_start < 0:
        return

    mr = margin + radius

    er = shape_edge_range[shape_index]
    edge_lo = er[0]
    edge_count = er[1]

    # Inflate the soft triangle AABB (shape-local), then unscale into the mesh
    # BVH's unscaled local space.
    lo_local = wp.min(wp.min(a, b), c) - wp.vec3(mr, mr, mr)
    hi_local = wp.max(wp.max(a, b), c) + wp.vec3(mr, mr, mr)
    lower = wp.cw_div(lo_local, geo_scale)
    upper = wp.cw_div(hi_local, geo_scale)

    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    rigid_tri = wp.int32(0)
    while wp.mesh_query_aabb_next(query, rigid_tri):
        rigid_flag = int(shape_mesh_tri_feature_owner_flag[tri_start + rigid_tri])

        i0 = wp.mesh_get_index(mesh_id, rigid_tri * 3 + 0)
        i1 = wp.mesh_get_index(mesh_id, rigid_tri * 3 + 1)
        i2 = wp.mesh_get_index(mesh_id, rigid_tri * 3 + 2)
        rv0 = wp.cw_mul(wp.mesh_get_point(mesh_id, i0), geo_scale)
        rv1 = wp.cw_mul(wp.mesh_get_point(mesh_id, i1), geo_scale)
        rv2 = wp.cw_mul(wp.mesh_get_point(mesh_id, i2), geo_scale)

        # T x V: rigid vertices owned by this rigid triangle.
        if (rigid_flag & 1) != 0:
            _tri_vs_point_emit(
                a, b, c, rv0, mr, soft_tri_id, shape_index, X_ws, X_bs,
                particle_count, soft_contact_max, soft_contact_count,
                soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
                soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
            )
        if (rigid_flag & 2) != 0:
            _tri_vs_point_emit(
                a, b, c, rv1, mr, soft_tri_id, shape_index, X_ws, X_bs,
                particle_count, soft_contact_max, soft_contact_count,
                soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
                soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
            )
        if (rigid_flag & 4) != 0:
            _tri_vs_point_emit(
                a, b, c, rv2, mr, soft_tri_id, shape_index, X_ws, X_bs,
                particle_count, soft_contact_max, soft_contact_count,
                soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
                soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
            )

        # E x E: rigid edges owned by this rigid triangle vs soft owned edges.
        # Skipped when the mesh is absent from mesh_edge_indices (no COLLIDE_SHAPES).
        if edge_lo >= 0:
            for jr in range(3):
                if (rigid_flag >> (jr + 3)) & 1:
                    own_eid = shape_mesh_tri_edges[tri_start + rigid_tri, jr]
                    local_e = own_eid - edge_start
                    if own_eid >= 0 and local_e >= 0 and local_e < edge_count:
                        ep = mesh_edge_indices[edge_lo + local_e]
                        re0 = wp.cw_mul(wp.mesh_get_point(mesh_id, ep[0]), geo_scale)
                        re1 = wp.cw_mul(wp.mesh_get_point(mesh_id, ep[1]), geo_scale)
                        for i in range(3):
                            if (int(soft_flag) >> (i + 3)) & 1:
                                sa, sb = _soft_edge_endpoints(i, a, b, c)
                                _edge_vs_edge_emit(
                                    sa, sb, re0, re1, i, mr, soft_tri_id, shape_index, X_ws, X_bs,
                                    particle_count, soft_contact_max, soft_contact_count,
                                    soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
                                    soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel,
                                    soft_contact_normal,
                                )


@wp.kernel
def create_soft_contacts_triangle_driven(
    # Grid dimension split.
    soft_tri_count: wp.int32,
    # Soft-side inputs.
    particle_q: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    particle_radius: wp.array[float],
    soft_tri_indices: wp.array2d[wp.int32],
    soft_tri_owner_flag: wp.array[wp.uint8],
    # Shape-side inputs (full table).
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_source_ptr: wp.array[wp.uint64],
    # Rigid-mesh ownership (Model-level).
    shape_mesh_tri_edges: wp.array2d[wp.int32],
    shape_mesh_tri_feature_owner_flag: wp.array[wp.uint8],
    shape_mesh_ownership_range: wp.array[wp.vec3i],
    mesh_edge_indices: wp.array[wp.vec2i],
    shape_edge_range: wp.array[wp.vec2i],
    # Contact config.
    margin: float,
    soft_contact_max: wp.int32,
    # Outputs (E/F range; reads slot 0 of soft_contact_count).
    soft_contact_count: wp.array[wp.int32],
    soft_contact_primitive: wp.array[wp.int32],
    soft_contact_kind: wp.array[wp.uint8],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """One thread per (rigid shape, soft triangle). The ``shape_index`` is
    uniform within each warp, so the geo_type dispatch is warp-coherent.

    Every soft triangle is a thread: its face participates in T x V regardless
    of feature ownership, while flag bits 3-5 gate which of its edges drive
    E x E. V x surface is handled by the legacy per-particle kernel.
    """
    tid = wp.tid()
    shape_index = tid // soft_tri_count
    soft_tri_local = tid % soft_tri_count

    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    soft_tri_id = soft_tri_local
    v0 = soft_tri_indices[soft_tri_id, 0]
    v1 = soft_tri_indices[soft_tri_id, 1]
    v2 = soft_tri_indices[soft_tri_id, 2]

    # World check (skip across-world pairs).
    if (
        particle_world[v0] != -1
        and shape_world[shape_index] != -1
        and particle_world[v0] != shape_world[shape_index]
    ):
        return

    # Conservative soft contact radius for this triangle.
    radius = wp.max(particle_radius[v0], wp.max(particle_radius[v1], particle_radius[v2]))

    # Transform soft triangle vertices into the shape's (unscaled) local frame.
    rigid_body = shape_body[shape_index]
    if rigid_body >= 0:
        X_wb = body_q[rigid_body]
    else:
        X_wb = wp.transform_identity()
    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    a = wp.transform_point(X_sw, particle_q[v0])
    b = wp.transform_point(X_sw, particle_q[v1])
    c = wp.transform_point(X_sw, particle_q[v2])

    soft_flag = soft_tri_owner_flag[soft_tri_id]
    geo_scale = shape_scale[shape_index]
    particle_count = soft_contact_count[0]

    geo = shape_type[shape_index]

    if geo == GeoType.MESH or geo == GeoType.CONVEX_MESH:
        _process_mesh_shape(
            soft_tri_id, soft_flag, a, b, c, shape_index, geo_scale,
            shape_source_ptr[shape_index],
            shape_mesh_tri_edges, shape_mesh_tri_feature_owner_flag,
            shape_mesh_ownership_range, mesh_edge_indices, shape_edge_range,
            margin, radius, X_ws, X_bs,
            particle_count, soft_contact_max, soft_contact_count,
            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
        )
    elif geo == GeoType.BOX:
        _process_box_shape(
            soft_tri_id, soft_flag, a, b, c, shape_index, geo_scale, margin, radius, X_ws, X_bs,
            particle_count, soft_contact_max, soft_contact_count,
            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
        )
    elif geo == GeoType.CAPSULE:
        _process_capsule_shape(
            soft_tri_id, soft_flag, a, b, c, shape_index, geo_scale, margin, radius, X_ws, X_bs,
            particle_count, soft_contact_max, soft_contact_count,
            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
        )
    elif geo == GeoType.SPHERE:
        _process_sphere_shape(
            soft_tri_id, soft_flag, a, b, c, shape_index, geo_scale, margin, radius, X_ws, X_bs,
            particle_count, soft_contact_max, soft_contact_count,
            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
        )
    elif geo == GeoType.CYLINDER:
        _process_cylinder_shape(
            soft_tri_id, soft_flag, a, b, c, shape_index, geo_scale, margin, radius, X_ws, X_bs,
            particle_count, soft_contact_max, soft_contact_count,
            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
        )
    elif geo == GeoType.CONE:
        _process_cone_shape(
            soft_tri_id, soft_flag, a, b, c, shape_index, geo_scale, margin, radius, X_ws, X_bs,
            particle_count, soft_contact_max, soft_contact_count,
            soft_contact_primitive, soft_contact_kind, soft_contact_barycentric,
            soft_contact_shape, soft_contact_body_pos, soft_contact_body_vel, soft_contact_normal,
        )
    # GeoType.PLANE / ELLIPSOID / SDF / HFIELD: no triangle-driven branch; their
    # V x surface contacts come from the legacy kernel into the particle range.


def launch_create_soft_contacts_triangle_driven(
    *,
    model,
    state,
    contacts,
    margin: float,
    device,
):
    """Launch the triangle-driven kernel once per :meth:`Model.collide`.

    All inputs are stable Warp arrays allocated at
    :meth:`ModelBuilder.finalize` or :class:`Contacts` construction, so the
    launch is safe to capture into a CUDA graph. Must run after the legacy
    ``create_soft_contacts`` launch on the same stream so ``soft_contact_count[0]``
    is final.

    Args:
        model: The finalized :class:`Model`.
        state: Current :class:`State` (provides ``particle_q`` and ``body_q``).
        contacts: The :class:`Contacts` buffer to append E/F records to.
        margin: Soft contact margin [m].
        device: Warp device to launch on.
    """
    soft_tri_count = model.tri_count
    if soft_tri_count == 0 or model.shape_count == 0:
        return

    wp.launch(
        kernel=create_soft_contacts_triangle_driven,
        dim=model.shape_count * soft_tri_count,
        inputs=[
            soft_tri_count,
            state.particle_q,
            model.particle_world,
            model.particle_radius,
            model.tri_indices,
            model.soft_mesh_adjacency.tri_feature_owner_flag,
            model.shape_body,
            model.shape_world,
            model.shape_type,
            model.shape_flags,
            model.shape_transform,
            model.shape_scale,
            state.body_q,
            model.shape_source_ptr,
            model.shape_mesh_tri_edges,
            model.shape_mesh_tri_feature_owner_flag,
            model.shape_mesh_ownership_range,
            model.mesh_edge_indices,
            model.shape_edge_range,
            margin,
            contacts.soft_contact_max,
        ],
        outputs=[
            contacts.soft_contact_count,
            contacts.soft_contact_primitive,
            contacts.soft_contact_kind,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ],
        device=device,
    )
