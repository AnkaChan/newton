# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The XFEM-type cut in SolverVBD (experimental, ``cut_method="xfem"``): level values, fragments,
virtual corners, wound particles, the tie and the surface rows.

A cut blade (see :meth:`newton.ModelBuilder.add_cut_blade`) cuts a tetrahedral soft body through
the tets where the blade plane is. A cut tet becomes two fragments. Its own line of the tet arrays
keeps the material on side A of the blade plane, and one spare tet of
:meth:`newton.ModelBuilder.add_cut_capacity` takes the material on side B. Both fragment rows have
the full rest shape of the tet and their share of its stiffness. All kernels have fixed launch
sizes and edit array contents only, so a captured CUDA graph can replay them.

Words: the *level value* of a tissue vertex is its signed distance from the blade plane, taken when
the stroke first needs the vertex (side A is negative, side B is positive or zero); a *crossed
edge* is a mesh edge whose two vertices are on different sides, and its *edge parameter* is the
place of the zero between them, from the side A vertex (0) to the side B vertex (1); a *fragment*
is the part of a cut tet on one side, a *fragment row* is the line of the tet arrays that holds
it, and the two fragment rows of one tet are *twins*; a *virtual corner* is a spare particle that
takes the place of a far-side corner in the fragment rows of one side; a *wound particle* is a
spare particle on a crossed edge that is a vertex of the collision triangles of the wound; the
*tie* holds a wound particle at its point on its edge; a *front edge* is a crossed edge with wound
particles that an uncut tet still holds; a *front row* is the edge row between the two wound
particles of a front edge.

A stored level value is never nearer to zero than ``level_min`` (the level clamp), so the cut
surface leaves the blade plane near a vertex that is close to the plane. The raw signed distance
is kept next to it: it gives the start gap of an edge in metres along the blade plane normal.

A cut advance is all or nothing. The kernels first count what the advance needs (spare tets,
spare particles for the wound particles, extra surface rows). When one of them is short, no tet
is cut in this cut update and the overflow entry of the short capacity grows by one.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ...geometry.tri_mesh_collision import TriMeshCollisionInfo
from ...utils.mesh import MeshAdjacencyData
from .cut_kernels import (
    _TET_EDGE_PAIRS_HOST,
    CUT_FORCE_UNCUTTABLE,
    TET_FACES,
    _blade_edge,
    _edge_index,
    _segment_distance_sq,
    _tet_rest_volume,
)

wp.set_module_options({"enable_backward": False})

# Level value of a particle that has none.
XFEM_LEVEL_UNSET = wp.constant(1.0e30)

# ``edge_owner`` entry of a mesh edge that no fragment row holds.
XFEM_NO_ROW = wp.constant(2147483647)

# Kinds of a triangle row and of an edge row (``tri_kind``, ``edge_kind``).
XFEM_ROW_FIXED = wp.constant(-1)  # a row without a tet source: the cut never writes it
XFEM_ROW_PARKED = wp.constant(0)  # a spare row that is not in use
XFEM_ROW_SKIN = wp.constant(1)  # a skin triangle (sub 0 to 2), or the side A part of a skin edge
XFEM_ROW_WOUND = wp.constant(2)  # a wound triangle of a cut slot (sub 0 to 3)
XFEM_EDGE_SKIN_B = wp.constant(2)  # the side B part of a cut skin edge
XFEM_EDGE_WOUND = wp.constant(3)  # a wound segment of a cut slot (sub 0 to 9)
XFEM_EDGE_FRONT = wp.constant(4)  # the part of a front edge between its two wound particles

# Wound triangle rows and wound edge rows of one cut slot.
XFEM_WOUND_TRIS_PER_TET = 4
XFEM_WOUND_EDGES_PER_TET = 10

_mat62i = wp.types.matrix(shape=(6, 2), dtype=wp.int32)

# The two local corners of each tet edge, in the order of ``_edge_index``.
TET_EDGE_CORNERS = wp.constant(_mat62i(0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3))


# =====================================================================
# Host helpers (solver construction)
# =====================================================================


def build_cut_mesh_edges(
    tet_indices: np.ndarray,
    first_tet_count: int,
    tri_source: np.ndarray,
    edge_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Number the mesh edges of the tissue tets and find the mesh edges of the surface rows.

    Args:
        tet_indices: Tet corners, shape ``[tet_count, 4]``.
        first_tet_count: Number of tissue tets; the spare tets and the idle tet follow them.
        tri_source: Tet source of each triangle row, shape ``[tri_count, 4]`` (tet, three local
            corners in row order), tet -1 for a row without a source.
        edge_source: Tet source of each edge row, shape ``[edge_count, 3]`` (tet, two local corners).

    Returns:
        ``mesh_edge`` ``[mesh_edge_count, 2]`` (the two vertices, smaller index first),
        ``tet_mesh_edge`` ``[tet_count, 6]`` (mesh edge of each local tet edge in the order (0,1)
        (0,2) (0,3) (1,2) (1,3) (2,3); -1 in the spare tets and the idle tet),
        ``tri_mesh_edge`` ``[tri_count, 3]`` (mesh edge between row corners k and k + 1; -1 without a
        source) and ``edge_row_mesh_edge`` ``[edge_count]`` (-1 without a source).
    """
    tets = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)
    tet_count = tets.shape[0]
    first = tets[:first_tet_count]
    pairs = np.array(_TET_EDGE_PAIRS_HOST, dtype=np.int64)
    ends = np.sort(first[:, pairs], axis=2).reshape(-1, 2)
    tet_mesh_edge = np.full((tet_count, 6), -1, dtype=np.int32)
    if ends.shape[0] > 0:
        mesh_edge, inverse = np.unique(ends, axis=0, return_inverse=True)
        tet_mesh_edge[:first_tet_count] = inverse.reshape(-1, 6)
    else:
        mesh_edge = np.zeros((0, 2), dtype=np.int64)

    def local_edge(a, b):
        return _TET_EDGE_PAIRS_HOST.index((min(a, b), max(a, b)))

    tri_source = np.asarray(tri_source).reshape(-1, 4)
    tri_mesh_edge = np.full((tri_source.shape[0], 3), -1, dtype=np.int32)
    for row in np.flatnonzero(tri_source[:, 0] >= 0):
        tet = int(tri_source[row, 0])
        corners = [int(c) for c in tri_source[row, 1:]]
        for k in range(3):
            tri_mesh_edge[row, k] = tet_mesh_edge[tet, local_edge(corners[k], corners[(k + 1) % 3])]
    edge_source = np.asarray(edge_source).reshape(-1, 3)
    edge_row_mesh_edge = np.full(edge_source.shape[0], -1, dtype=np.int32)
    for row in np.flatnonzero(edge_source[:, 0] >= 0):
        tet = int(edge_source[row, 0])
        edge_row_mesh_edge[row] = tet_mesh_edge[tet, local_edge(int(edge_source[row, 1]), int(edge_source[row, 2]))]
    return mesh_edge.astype(np.int32), tet_mesh_edge, tri_mesh_edge, edge_row_mesh_edge


# =====================================================================
# Small functions
# =====================================================================


@wp.func
def _level_side(level: float) -> int:
    """Side of a level value: 0 is side A (negative), 1 is side B (positive or zero)."""
    if level < 0.0:
        return 0
    return 1


@wp.func
def _edge_parameter(level_a: float, level_b: float, t_min: float) -> float:
    """Place of the zero level between a side A vertex (0) and a side B vertex (1), limited to [t_min, 1 - t_min]."""
    return wp.clamp(level_a / (level_a - level_b), t_min, 1.0 - t_min)


@wp.func
def tet_side_a_share(level: wp.vec4, t_min: float) -> float:
    """Volume share of a tet on side A of the blade plane, from the level values of its four corners.

    The cut polygon goes through the points of the edge parameters, which are limited to
    ``[t_min, 1 - t_min]`` as the wound particles are. The share is the sum of the sub-tets of the
    side A part in barycentric coordinates: one sub-tet when one corner is alone on a side, three
    when two corners are on each side.

    Returns:
        The share in [0, 1]; 0 or 1 for a tet that the plane does not cross.
    """
    count_a = int(0)
    for i in range(4):
        if level[i] < 0.0:
            count_a += 1
    if count_a == 0:
        return 0.0
    if count_a == 4:
        return 1.0
    if count_a == 1:
        lone = int(0)
        for i in range(4):
            if level[i] < 0.0:
                lone = i
        share = float(1.0)
        for j in range(4):
            if j != lone:
                share *= _edge_parameter(level[lone], level[j], t_min)
        return share
    if count_a == 3:
        lone = int(0)
        for i in range(4):
            if level[i] >= 0.0:
                lone = i
        share_b = float(1.0)
        for j in range(4):
            if j != lone:
                share_b *= 1.0 - _edge_parameter(level[j], level[lone], t_min)
        return 1.0 - share_b
    p = int(-1)
    q = int(-1)
    r = int(-1)
    s = int(-1)
    for i in range(4):
        if level[i] < 0.0:
            if p < 0:
                p = i
            else:
                q = i
        else:
            if r < 0:
                r = i
            else:
                s = i
    t_pr = _edge_parameter(level[p], level[r], t_min)
    t_ps = _edge_parameter(level[p], level[s], t_min)
    t_qr = _edge_parameter(level[q], level[r], t_min)
    t_qs = _edge_parameter(level[q], level[s], t_min)
    # The side A part is the prism (p, pr, ps; q, qr, qs), cut into the tets (p, pr, ps, q),
    # (pr, ps, q, qr) and (ps, q, qr, qs).
    return t_pr * t_ps + t_ps * t_qr * (1.0 - t_pr) + t_qr * t_qs * (1.0 - t_ps)


@wp.func
def _tie_corners(
    edge: int,
    side: int,
    edge_owner: wp.array2d[wp.int32],
    edge_owner_local: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
):
    """The side A corner and the side B corner of a crossed edge, read through the owner row of one side."""
    row = edge_owner[edge, side]
    local = edge_owner_local[edge, side]
    i = TET_EDGE_CORNERS[local, 0]
    j = TET_EDGE_CORNERS[local, 1]
    if particle_level[tet_vertex[row, i]] < 0.0:
        return tet_indices[row, i], tet_indices[row, j]
    return tet_indices[row, j], tet_indices[row, i]


@wp.func
def _node_label(
    node: int, count: int, base: int, record_group: wp.array[wp.int32], twin_group: wp.array[wp.int32]
) -> int:
    if node < count:
        return record_group[base + node]
    return twin_group[base + node - count]


@wp.func
def _set_node_label(
    node: int, count: int, base: int, record_group: wp.array[wp.int32], twin_group: wp.array[wp.int32], label: int
):
    if node < count:
        record_group[base + node] = label
    else:
        twin_group[base + node - count] = label


@wp.func
def _node_row(node: int, count: int, offset: int, adjacency: MeshAdjacencyData, tet_twin: wp.array[wp.int32]) -> int:
    """Tet row of a node of the group search: a tet record, or the twin of the row of a tet record."""
    if node < count:
        return adjacency.v_adj_tets[offset + 2 * node]
    return tet_twin[adjacency.v_adj_tets[offset + 2 * (node - count)]]


@wp.func
def _node_corner(node: int, count: int, offset: int, adjacency: MeshAdjacencyData) -> int:
    if node < count:
        return adjacency.v_adj_tets[offset + 2 * node + 1]
    return adjacency.v_adj_tets[offset + 2 * (node - count) + 1]


@wp.func
def _row_has_material_on_face(
    row: int,
    face: int,
    tet_side: wp.array[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
) -> bool:
    """A row that is not cut has material on each of its faces; a fragment has when the face has a vertex of its side."""
    side = tet_side[row]
    if side == 0:
        return True
    for k in range(3):
        if _level_side(particle_level[tet_vertex[row, TET_FACES[face, k]]]) == side - 1:
            return True
    return False


@wp.func
def _vertex_on_own_side(
    row: int,
    corner: int,
    tet_side: wp.array[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
) -> bool:
    """A corner of a row that is not cut is on its own side; a corner of a fragment is when it has the side of the fragment."""
    side = tet_side[row]
    if side == 0:
        return True
    return _level_side(particle_level[tet_vertex[row, corner]]) == side - 1


# =====================================================================
# One cut advance: level values, crossed edges, fragments
# =====================================================================


@wp.kernel
def update_cut_stroke(
    due: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    stroke_open: wp.array[wp.int32],
    stroke_number: wp.array[wp.int32],
):
    """One thread: count the strokes. The number grows by one when the blade edge has lost its last contact record."""
    if due[0] == 0:
        return
    if contact_count[0] > 0:
        stroke_open[0] = 1
    elif stroke_open[0] != 0:
        stroke_open[0] = 0
        stroke_number[0] = stroke_number[0] + 1


@wp.kernel
def set_cut_level_values(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    stroke_number: wp.array[wp.int32],
    level_range: float,
    level_min: float,
    blade_body: wp.array[wp.int32],
    blade_edge_shape: wp.array[wp.int32],
    blade_direction: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    cut_force: wp.array[wp.float32],
    wound_edge: wp.array[wp.int32],
    particle_level: wp.array[float],
    particle_level_raw: wp.array[float],
    particle_level_set: wp.array[wp.int32],
    particle_stroke: wp.array[wp.int32],
):
    """One thread per particle: a cuttable particle near the blade edge takes its level value for the stroke.

    ``particle_level_set`` is 0 without a value, 1 with a value of the current stroke and 2 with a
    value that stays (a vertex of a cut tet). The end of the stroke clears the values of state 1.

    The level clamp: a value nearer to zero than ``level_min`` is stored as ``+-level_min``. The
    stored value gives the side, the edge parameter and the material share, so no fragment is a
    sliver. The particle does not move, so near such a vertex the cut surface is not on the blade
    plane. ``particle_level_raw`` keeps the signed distance itself [m]; the start gap of an edge
    is taken from it (:func:`set_cut_edge_parameters`). ``particle_stroke`` keeps the number of
    the stroke that set the value.
    """
    p = wp.tid()
    if due[0] == 0:
        return
    if contact_count[0] == 0:
        if particle_level_set[p] == 1:  # the stroke ended
            particle_level_set[p] = 0
            particle_level[p] = XFEM_LEVEL_UNSET
        return
    if advance[0] == 0 or particle_level_set[p] != 0:
        return
    if cut_force[p] >= CUT_FORCE_UNCUTTABLE or wound_edge[p] >= 0:
        return
    a, b, d = _blade_edge(0, blade_body, blade_edge_shape, blade_direction, body_q, shape_transform, shape_scale)
    x = particle_q[p]
    if _segment_distance_sq(x, a, b) > level_range * level_range:
        return
    n = wp.cross(d, wp.normalize(b - a))
    level = wp.dot(n, x - a)
    particle_level_raw[p] = level
    if level < 0.0:
        level = wp.min(level, -level_min)
    else:
        level = wp.max(level, level_min)
    particle_level[p] = level
    particle_level_set[p] = 1
    particle_stroke[p] = stroke_number[0]


@wp.kernel
def set_cut_edge_parameters(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    start_gap: float,
    t_min: float,
    mesh_edge: wp.array2d[wp.int32],
    particle_level: wp.array[float],
    particle_level_raw: wp.array[float],
    particle_level_set: wp.array[wp.int32],
    edge_wound: wp.array2d[wp.int32],
    edge_t: wp.array[float],
    edge_half_gap_t: wp.array[float],
    edge_gap_limited: wp.array[wp.int32],
    edge_pending: wp.array[wp.int32],
):
    """One thread per mesh edge: the edge parameter of a crossed edge and the step of it that gives the start gap.

    The edge parameter comes from the stored (clamped) level values. The step ``edge_half_gap_t``
    comes from the raw signed distances: the two tie targets, one step to each side, are one start
    gap apart along the blade plane normal. Both tie targets must stay inside ``[t_min, 1 - t_min]``,
    so the step is at most ``0.5 * (1 - 2 * t_min)``. That limit acts on an edge that spans less than
    ``start_gap / (1 - 2 * t_min)`` along the normal; such an edge holds a smaller start gap and
    gets ``edge_gap_limited`` 1. The thread also clears the pending mark of its edge
    (:func:`mark_crossed_tets` sets it).
    """
    e = wp.tid()
    if due[0] == 0 or advance[0] == 0:
        return
    edge_pending[e] = 0
    if edge_wound[e, 0] >= 0:
        return
    v_a = mesh_edge[e, 0]
    v_b = mesh_edge[e, 1]
    if particle_level_set[v_a] == 0 or particle_level_set[v_b] == 0:
        return
    if _level_side(particle_level[v_a]) == _level_side(particle_level[v_b]):
        return
    if particle_level[v_a] >= 0.0:
        v_a = mesh_edge[e, 1]
        v_b = mesh_edge[e, 0]
    edge_t[e] = _edge_parameter(particle_level[v_a], particle_level[v_b], t_min)
    # The clamp keeps the sign, so the raw span is not negative.
    span = particle_level_raw[v_b] - particle_level_raw[v_a]
    step = 0.5 * (1.0 - 2.0 * t_min)
    limited = int(1)
    if span * step >= 0.5 * start_gap:
        step = 0.5 * start_gap / span
        limited = 0
    edge_half_gap_t[e] = step
    edge_gap_limited[e] = limited


@wp.kernel
def mark_crossed_tets(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    demand_slot: int,
    blade_body: wp.array[wp.int32],
    blade_edge_shape: wp.array[wp.int32],
    blade_direction: wp.array[wp.vec3],
    contact_range: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    cut_force: wp.array[wp.float32],
    particle_level: wp.array[float],
    particle_level_set: wp.array[wp.int32],
    edge_t: wp.array[float],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    tet_mesh_edge: wp.array2d[wp.int32],
    tet_side: wp.array[wp.int32],
    tet_pending: wp.array[wp.int32],
    edge_pending: wp.array[wp.int32],
    demand: wp.array[wp.int32],
):
    """One thread per tissue tet: mark a tet that is not cut yet and has a crossed edge near the blade edge.

    A crossed edge is pressed when its crossing point is within the contact range of the blade
    edge. The marked tet is *pending*: :func:`cut_crossed_tets` cuts it when the whole advance
    fits the capacity. The thread marks the crossed edges of its tet and adds one spare tet to
    the demand of the advance.
    """
    tet = wp.tid()
    if due[0] == 0 or advance[0] == 0:
        return
    tet_pending[tet] = 0
    if tet_side[tet] != 0:
        return
    level = wp.vec4(0.0)
    for c in range(4):
        v = tet_vertex[tet, c]
        if particle_level_set[v] == 0 or cut_force[tet_indices[tet, c]] >= CUT_FORCE_UNCUTTABLE:
            return
        level[c] = particle_level[v]
    a, b, _d = _blade_edge(0, blade_body, blade_edge_shape, blade_direction, body_q, shape_transform, shape_scale)
    pressed_range = contact_range[0]
    pressed = int(0)
    for local in range(6):
        i = TET_EDGE_CORNERS[local, 0]
        j = TET_EDGE_CORNERS[local, 1]
        if _level_side(level[i]) != _level_side(level[j]):
            x_a = particle_q[tet_indices[tet, i]]
            x_b = particle_q[tet_indices[tet, j]]
            if level[i] >= 0.0:
                x_a = particle_q[tet_indices[tet, j]]
                x_b = particle_q[tet_indices[tet, i]]
            crossing = x_a + edge_t[tet_mesh_edge[tet, local]] * (x_b - x_a)
            if _segment_distance_sq(crossing, a, b) <= pressed_range * pressed_range:
                pressed = 1
    if pressed == 0:
        return
    tet_pending[tet] = 1
    wp.atomic_add(demand, demand_slot, 1)
    for local in range(6):
        if _level_side(level[TET_EDGE_CORNERS[local, 0]]) != _level_side(level[TET_EDGE_CORNERS[local, 1]]):
            edge_pending[tet_mesh_edge[tet, local]] = 1


@wp.kernel
def count_cut_edge_demand(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    front_rows: int,
    edge_row_slot: int,
    particle_colors: wp.array[wp.int32],
    mesh_edge: wp.array2d[wp.int32],
    edge_pending: wp.array[wp.int32],
    edge_wound: wp.array2d[wp.int32],
    edge_cut: wp.array[wp.int32],
    demand: wp.array[wp.int32],
):
    """One thread per mesh edge: a crossed edge of a pending tet without wound particles needs one spare of the pool of each end.

    With ``front_rows`` it also needs one extra edge row.
    """
    e = wp.tid()
    if due[0] == 0 or advance[0] == 0:
        return
    if edge_pending[e] == 0 or edge_wound[e, 0] >= 0 or edge_cut[e] == 2:
        return
    wp.atomic_add(demand, particle_colors[mesh_edge[e, 0]], 1)
    wp.atomic_add(demand, particle_colors[mesh_edge[e, 1]], 1)
    if front_rows != 0:
        wp.atomic_add(demand, edge_row_slot, 1)


@wp.kernel
def count_cut_triangle_row_demand(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    tri_spare_start: int,
    triangle_row_slot: int,
    tet_pending: wp.array[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
    tri_source: wp.array2d[wp.int32],
    tri_kind: wp.array[wp.int32],
    tri_extra: wp.array[wp.int32],
    demand: wp.array[wp.int32],
):
    """One thread per original triangle row: a skin triangle that a pending tet clips needs two extra rows."""
    row = wp.tid()
    if due[0] == 0 or advance[0] == 0 or row >= tri_spare_start:
        return
    if tri_kind[row] != XFEM_ROW_SKIN or tri_extra[row] != -1:
        return
    tet = tri_source[row, 0]
    if tet_pending[tet] == 0:
        return
    side_0 = _level_side(particle_level[tet_vertex[tet, tri_source[row, 1]]])
    side_1 = _level_side(particle_level[tet_vertex[tet, tri_source[row, 2]]])
    side_2 = _level_side(particle_level[tet_vertex[tet, tri_source[row, 3]]])
    if side_0 == side_1 and side_1 == side_2:
        return
    wp.atomic_add(demand, triangle_row_slot, 2)


@wp.kernel
def count_cut_edge_row_demand(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    edge_spare_start: int,
    edge_row_slot: int,
    edge_pending: wp.array[wp.int32],
    edge_wound: wp.array2d[wp.int32],
    edge_cut: wp.array[wp.int32],
    edge_row_mesh_edge: wp.array[wp.int32],
    edge_kind: wp.array[wp.int32],
    edge_extra: wp.array[wp.int32],
    demand: wp.array[wp.int32],
):
    """One thread per original edge row: a skin edge that gets wound particles in this advance needs one extra row."""
    row = wp.tid()
    if due[0] == 0 or advance[0] == 0 or row >= edge_spare_start:
        return
    if edge_kind[row] != XFEM_ROW_SKIN or edge_extra[row] != -1:
        return
    e = edge_row_mesh_edge[row]
    if edge_pending[e] == 0 or edge_wound[e, 0] >= 0 or edge_cut[e] == 2:
        return
    wp.atomic_add(demand, edge_row_slot, 1)


@wp.kernel
def check_cut_demand(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    color_count: int,
    spare_count: int,
    triangle_row_count: int,
    edge_row_count: int,
    spare_tet_count: int,
    tri_extra_taken: wp.array[wp.int32],
    edge_extra_taken: wp.array[wp.int32],
    slot_taken: wp.array[wp.int32],
    pool_taken: wp.array[wp.int32],
    demand: wp.array[wp.int32],
    wound_base: wp.array[wp.int32],
    wound_next: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
    commit: wp.array[wp.int32],
):
    """One thread: let the cut advance go on only when the capacity holds all of it.

    ``demand`` has one entry per pool (spare particles for the wound particles of the advance),
    then the extra triangle rows, the extra edge rows and the spare tets. When an entry does not
    fit, the advance is refused: ``commit`` stays 0, no tet is cut in this cut update, and the
    overflow entry of that capacity grows by one. The blade then stays on the tissue, so a
    refused advance is counted again at every later cut update that wants it.

    When the advance fits, the thread reserves the spare particles of the wound particles: pool
    ``c`` gives the spares from ``wound_base[c]`` on, and ``pool_taken`` moves past them. The
    virtual corners of the advance take their spares after that block.
    """
    commit[0] = 0
    if due[0] == 0 or advance[0] == 0:
        return
    refused = int(0)
    for c in range(color_count):
        if demand[c] > 0 and pool_taken[c] + demand[c] > spare_count:
            overflow[c] = overflow[c] + 1
            refused = 1
    if demand[color_count] > 0 and tri_extra_taken[0] + demand[color_count] > triangle_row_count:
        overflow[color_count] = overflow[color_count] + 1
        refused = 1
    if demand[color_count + 1] > 0 and edge_extra_taken[0] + demand[color_count + 1] > edge_row_count:
        overflow[color_count + 1] = overflow[color_count + 1] + 1
        refused = 1
    if demand[color_count + 2] > 0 and slot_taken[0] + demand[color_count + 2] > spare_tet_count:
        overflow[color_count + 2] = overflow[color_count + 2] + 1
        refused = 1
    if refused == 0:
        for c in range(color_count):
            wound_base[c] = pool_taken[c]
            wound_next[c] = 0
            pool_taken[c] = pool_taken[c] + demand[c]
        commit[0] = 1
    for i in range(color_count + 3):
        demand[i] = 0


@wp.kernel
def cut_crossed_tets(
    due: wp.array[wp.int32],
    commit: wp.array[wp.int32],
    update_number: wp.array[wp.int32],
    spare_tet_start: int,
    spare_tet_count: int,
    full_copy: int,
    t_min: float,
    overflow_slot: int,
    particle_level: wp.array[float],
    tet_pending: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    tet_mesh_edge: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    tet_side: wp.array[wp.int32],
    tet_twin: wp.array[wp.int32],
    tet_first: wp.array[wp.int32],
    tet_slot: wp.array[wp.int32],
    tet_stamp: wp.array[wp.int32],
    tet_share: wp.array[float],
    slot_tet: wp.array[wp.int32],
    slot_taken: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
    edge_cut: wp.array[wp.int32],
    edge_owner: wp.array2d[wp.int32],
    particle_level_set: wp.array[wp.int32],
    particle_dirty: wp.array[wp.int32],
):
    """One thread per tissue tet: cut a pending tet (:func:`mark_crossed_tets`) of an advance that fits.

    The tet takes a cut slot. Its own line of the tet arrays keeps fragment A; the spare tet of
    the slot takes fragment B with the same corners, the same rest pose and the same mesh edges.
    With ``full_copy == 0`` each fragment row gets the stiffness share of its side.
    """
    tet = wp.tid()
    if due[0] == 0 or commit[0] == 0 or tet_pending[tet] == 0:
        return
    tet_pending[tet] = 0
    level = wp.vec4(0.0)
    for c in range(4):
        level[c] = particle_level[tet_vertex[tet, c]]
    slot = wp.atomic_add(slot_taken, 0, 1)
    if slot >= spare_tet_count:
        wp.atomic_add(overflow, overflow_slot, 1)  # not reached: check_cut_demand counted this tet
        return
    twin = spare_tet_start + slot
    share_a = tet_side_a_share(level, t_min)
    scale_a = share_a
    scale_b = 1.0 - share_a
    if full_copy != 0:
        scale_a = 1.0
        scale_b = 1.0
    for c in range(4):
        tet_indices[twin, c] = tet_indices[tet, c]
        tet_vertex[twin, c] = tet_vertex[tet, c]
        particle_level_set[tet_vertex[tet, c]] = 2  # the level value of a vertex of a cut tet stays
        particle_dirty[tet_indices[tet, c]] = 1
    tet_poses[twin] = tet_poses[tet]
    for m in range(3):
        stiffness = tet_materials[tet, m]
        tet_materials[twin, m] = scale_b * stiffness
        tet_materials[tet, m] = scale_a * stiffness
    for local in range(6):
        e = tet_mesh_edge[tet, local]
        tet_mesh_edge[twin, local] = e
        if _level_side(level[TET_EDGE_CORNERS[local, 0]]) != _level_side(level[TET_EDGE_CORNERS[local, 1]]):
            edge_cut[e] = wp.max(edge_cut[e], 1)
            wp.atomic_min(edge_owner, e, 0, tet)
            wp.atomic_min(edge_owner, e, 1, twin)
    stamp = update_number[0]
    tet_side[tet] = 1
    tet_side[twin] = 2
    tet_twin[tet] = twin
    tet_twin[twin] = tet
    tet_first[twin] = tet
    tet_slot[tet] = slot
    tet_slot[twin] = slot
    tet_stamp[tet] = stamp
    tet_stamp[twin] = stamp
    tet_share[tet] = share_a
    tet_share[twin] = 1.0 - share_a
    slot_tet[slot] = tet


# =====================================================================
# Virtual corners
# =====================================================================


@wp.kernel
def split_cut_tet_vertices(
    due: wp.array[wp.int32],
    update_number: wp.array[wp.int32],
    idle_tet: int,
    color_count: int,
    spare_count: int,
    full_copy_mass: int,
    gate_bit: int,
    ghost_gate_bit: int,
    overflow_record_slot: int,
    spare_start: wp.array[wp.int32],
    particle_colors: wp.array[wp.int32],
    particle_attached: wp.array[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    tet_side: wp.array[wp.int32],
    tet_twin: wp.array[wp.int32],
    tet_first: wp.array[wp.int32],
    tet_stamp: wp.array[wp.int32],
    tet_share: wp.array[float],
    tet_vertex: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    adjacency: MeshAdjacencyData,
    record_group: wp.array[wp.int32],
    twin_group: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_q_rest: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    cut_force: wp.array[wp.float32],
    suture_grip: wp.array[wp.float32],
    suture_mu: wp.array[wp.float32],
    suture_puncture_force: wp.array[wp.float32],
    suture_puncture_cos: wp.array[wp.float32],
    suture_stiffness: wp.array[wp.float32],
    particle_gate: wp.array[wp.int32],
    particle_ghost: wp.array[wp.int32],
    particle_level: wp.array[float],
    particle_level_raw: wp.array[float],
    particle_level_set: wp.array[wp.int32],
    particle_stroke: wp.array[wp.int32],
    pool_taken: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
    particle_dirty: wp.array[wp.int32],
):
    """One thread per corner particle of a tet cut in this advance: tet groups, virtual corners and tet records.

    The nodes of the group search are the tet records of the particle and, for a row cut in this
    advance, the twin row with the same corner. Two rows of different tets that share a face with
    the particle are joined when each has material on that face and they are not fragments of
    different sides. Every group that holds a row in which the particle is on its own side keeps
    the particle. Every other group gets a virtual corner (a ghost) from the pool of the particle
    color. So this method makes no copy that is material, and a spare of a pool is a virtual
    corner or a wound particle.

    A mesh with a pinch vertex (tets around a vertex that no face joins) has two groups with the
    particle on its own side. Both keep the particle: the blade does not separate them.

    The thread writes only the tet corners that hold its own particle, the tet records of its own
    particle and the data of the spares it took. A kept twin row takes the tet record of its moved
    twin, or a free tet record. When a pool is empty the group stays on the particle (counted in
    ``overflow``), so the two sides stay joined at this vertex.
    """
    v = wp.tid()
    if due[0] == 0 or particle_dirty[v] == 0:
        return
    particle_dirty[v] = 0
    stamp = update_number[0]
    offset = adjacency.v_adj_tets_offsets[v]
    count = (adjacency.v_adj_tets_offsets[v + 1] - offset) >> 1
    base = offset >> 1
    nodes = 2 * count

    # Every node starts as its own group. Node k < count is tet record k; node count + k is the twin of its row.
    for k in range(count):
        record_group[base + k] = -1
        twin_group[base + k] = -1
        row = adjacency.v_adj_tets[offset + 2 * k]
        if row != idle_tet:
            record_group[base + k] = k
            if tet_side[row] == 1 and tet_stamp[row] == stamp:
                twin_group[base + k] = count + k

    # A group label travels one join per sweep at least, so `nodes` sweeps always converge; the sweeps
    # after one without a join do nothing.
    joins = int(0)
    joins_before = int(-1)
    for _sweep in range(nodes):
        if joins != joins_before:
            joins_before = joins
            for n in range(nodes):
                if _node_label(n, count, base, record_group, twin_group) < 0:
                    continue
                row = _node_row(n, count, offset, adjacency, tet_twin)
                corner = _node_corner(n, count, offset, adjacency)
                side = tet_side[row]
                first = tet_first[row]
                for f in range(4):
                    if f == corner:
                        continue
                    other = tet_neighbors[first, f]
                    if other < 0:
                        continue
                    if not _row_has_material_on_face(row, f, tet_side, tet_vertex, particle_level):
                        continue
                    for m in range(nodes):
                        label_m = _node_label(m, count, base, record_group, twin_group)
                        if label_m < 0:
                            continue
                        row_m = _node_row(m, count, offset, adjacency, tet_twin)
                        if tet_first[row_m] != other:
                            continue
                        side_m = tet_side[row_m]
                        if side != 0 and side_m != 0 and side != side_m:
                            continue  # fragments of different sides
                        face_m = int(-1)
                        for k in range(4):
                            if tet_neighbors[other, k] == first:
                                face_m = k
                        if face_m < 0:
                            continue
                        if not _row_has_material_on_face(row_m, face_m, tet_side, tet_vertex, particle_level):
                            continue
                        label_n = _node_label(n, count, base, record_group, twin_group)
                        if label_m != label_n:
                            lowest = wp.min(label_m, label_n)
                            _set_node_label(n, count, base, record_group, twin_group, lowest)
                            _set_node_label(m, count, base, record_group, twin_group, lowest)
                            joins += 1

    # The first group that keeps the particle, and the material weight of all rows.
    keep = int(-1)
    lowest_label = int(-1)
    group_count = int(0)
    total_weight = float(0.0)
    for n in range(nodes):
        label = _node_label(n, count, base, record_group, twin_group)
        if label < 0:
            continue
        row = _node_row(n, count, offset, adjacency, tet_twin)
        corner = _node_corner(n, count, offset, adjacency)
        if label == n:
            group_count += 1
        total_weight += tet_share[row] * _tet_rest_volume(row, tet_poses)
        if lowest_label < 0 or label < lowest_label:
            lowest_label = label
        if _vertex_on_own_side(row, corner, tet_side, tet_vertex, particle_level):
            if keep < 0 or label < keep:
                keep = label
    if keep < 0:
        keep = lowest_label

    # A pinned or attached particle stays shared: all its rows keep it.
    can_split = (
        group_count >= 2
        and total_weight > 0.0
        and (particle_flags[v] & ParticleFlags.ACTIVE) != 0
        and particle_mass[v] > 0.0
        and particle_attached[v] == 0
    )
    if can_split:
        color = particle_colors[v]
        pool = color
        if color >= color_count:
            pool = color - color_count  # a copy takes from its own pool
        mass = particle_mass[v]
        moved_weight = float(0.0)
        for r in range(nodes):
            if r == keep or _node_label(r, count, base, record_group, twin_group) != r:
                continue
            # A group with the particle on its own side keeps the particle (a pinch vertex has two).
            own = int(0)
            for n in range(nodes):
                if _node_label(n, count, base, record_group, twin_group) == r:
                    if _vertex_on_own_side(
                        _node_row(n, count, offset, adjacency, tet_twin),
                        _node_corner(n, count, offset, adjacency),
                        tet_side,
                        tet_vertex,
                        particle_level,
                    ):
                        own = 1
            if own != 0:
                continue
            slot = wp.atomic_add(pool_taken, pool, 1)
            if slot >= spare_count:
                wp.atomic_add(overflow, pool, 1)
                continue  # the group stays on v
            s = spare_start[pool] + slot
            s_offset = adjacency.v_adj_tets_offsets[s]
            s_room = (adjacency.v_adj_tets_offsets[s + 1] - s_offset) >> 1
            s_count = int(0)
            weight = float(0.0)
            for n in range(nodes):
                if _node_label(n, count, base, record_group, twin_group) != r:
                    continue
                if s_count >= s_room:
                    wp.atomic_add(overflow, overflow_record_slot, 1)
                    continue  # the row stays on v
                row = _node_row(n, count, offset, adjacency, tet_twin)
                corner = _node_corner(n, count, offset, adjacency)
                tet_indices[row, corner] = s
                adjacency.v_adj_tets[s_offset + 2 * s_count] = row
                adjacency.v_adj_tets[s_offset + 2 * s_count + 1] = corner
                s_count += 1
                weight += tet_share[row] * _tet_rest_volume(row, tet_poses)
                if n < count:
                    record_group[base + n] = -2  # moved; its tet record is cleared below
                else:
                    twin_group[base + n - count] = -1
            copy_mass = mass
            if full_copy_mass == 0:
                copy_mass = mass * weight / total_weight
            particle_mass[s] = copy_mass
            particle_inv_mass[s] = 1.0 / copy_mass
            particle_q[s] = particle_q[v]
            particle_qd[s] = particle_qd[v]
            particle_q_rest[s] = particle_q_rest[v]
            cut_force[s] = cut_force[v]
            suture_grip[s] = suture_grip[v]
            suture_mu[s] = suture_mu[v]
            suture_puncture_force[s] = suture_puncture_force[v]
            suture_puncture_cos[s] = suture_puncture_cos[v]
            suture_stiffness[s] = suture_stiffness[v]
            particle_flags[s] = particle_flags[v]
            particle_level[s] = particle_level[v]
            particle_level_raw[s] = particle_level_raw[v]
            particle_level_set[s] = 2
            particle_stroke[s] = particle_stroke[v]
            particle_ghost[s] = 1
            particle_gate[s] = particle_gate[v] | gate_bit | ghost_gate_bit
            moved_weight += weight
        if moved_weight > 0.0 and full_copy_mass == 0:
            kept_mass = mass * (total_weight - moved_weight) / total_weight
            particle_mass[v] = kept_mass
            particle_inv_mass[v] = 1.0 / kept_mass

    # Tet records of v: a moved record gives its place to its kept twin, or becomes free.
    for k in range(count):
        if record_group[base + k] == -2:
            if twin_group[base + k] >= 0:
                adjacency.v_adj_tets[offset + 2 * k] = tet_twin[adjacency.v_adj_tets[offset + 2 * k]]
                twin_group[base + k] = -1
            else:
                adjacency.v_adj_tets[offset + 2 * k] = idle_tet
                adjacency.v_adj_tets[offset + 2 * k + 1] = 0
            record_group[base + k] = -1
    # A kept twin of a kept row takes a free tet record.
    cursor = int(0)
    for k in range(count):
        if record_group[base + k] >= 0 and twin_group[base + k] >= 0:
            twin_row = tet_twin[adjacency.v_adj_tets[offset + 2 * k]]
            corner = adjacency.v_adj_tets[offset + 2 * k + 1]
            placed = int(0)
            for j in range(count):
                if placed == 0 and j >= cursor and adjacency.v_adj_tets[offset + 2 * j] == idle_tet:
                    adjacency.v_adj_tets[offset + 2 * j] = twin_row
                    adjacency.v_adj_tets[offset + 2 * j + 1] = corner
                    placed = 1
                    cursor = j + 1
            if placed == 0:
                wp.atomic_add(overflow, overflow_record_slot, 1)


# =====================================================================
# Wound particles
# =====================================================================


@wp.kernel
def make_wound_particles(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    spare_count: int,
    t_min: float,
    cut_gate_bits: int,
    front_rows: int,
    front_row_start: int,
    front_row_count: int,
    overflow_edge_slot: int,
    spare_start: wp.array[wp.int32],
    particle_colors: wp.array[wp.int32],
    mesh_edge: wp.array2d[wp.int32],
    edge_t: wp.array[float],
    edge_half_gap_t: wp.array[float],
    edge_owner: wp.array2d[wp.int32],
    tet_mesh_edge: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    particle_level: wp.array[float],
    stroke_number: wp.array[wp.int32],
    wound_base: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_q_rest: wp.array[wp.vec3],
    cut_force: wp.array[wp.float32],
    suture_grip: wp.array[wp.float32],
    suture_mu: wp.array[wp.float32],
    suture_puncture_force: wp.array[wp.float32],
    suture_puncture_cos: wp.array[wp.float32],
    suture_stiffness: wp.array[wp.float32],
    particle_gate: wp.array[wp.int32],
    edge_cut: wp.array[wp.int32],
    edge_owner_local: wp.array2d[wp.int32],
    edge_wound: wp.array2d[wp.int32],
    edge_front: wp.array[wp.int32],
    wound_edge: wp.array[wp.int32],
    wound_side: wp.array[wp.int32],
    wound_new: wp.array[wp.int32],
    tie_s: wp.array[float],
    tie_dual: wp.array[wp.vec3],
    tie_count: wp.array[wp.int32],
    particle_stroke: wp.array[wp.int32],
    wound_next: wp.array[wp.int32],
    edge_row_mesh_edge: wp.array[wp.int32],
    edge_kind: wp.array[wp.int32],
    edge_extra_taken: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    """One thread per mesh edge: owner rows of a crossed edge of a cut tet, and its two wound particles.

    The wound particle of side A comes from the pool of the color of the side A vertex of the edge,
    the one of side B from the pool of the color of the side B vertex, so a wound particle never
    has the color of a corner of its tie. The spares are those that :func:`check_cut_demand`
    reserved for this advance, so the take cannot fail and no spare is lost. Each wound particle
    starts at its tie target. The thread also clears the front mark of its edge; the next launch
    sets the marks again. With ``front_rows`` the edge takes one extra edge row for the part
    between its two wound particles (see :func:`write_cut_edge_rows`).
    """
    e = wp.tid()
    if due[0] == 0 or advance[0] == 0:
        return
    edge_front[e] = 0
    if edge_cut[e] == 0:
        return
    for side in range(2):
        row = edge_owner[e, side]
        if row != XFEM_NO_ROW:
            for local in range(6):
                if tet_mesh_edge[row, local] == e:
                    edge_owner_local[e, side] = local
    if edge_wound[e, 0] >= 0 or edge_cut[e] != 1:
        return
    vertex_a = mesh_edge[e, 0]
    vertex_b = mesh_edge[e, 1]
    if particle_level[vertex_a] >= 0.0:
        vertex_a = mesh_edge[e, 1]
        vertex_b = mesh_edge[e, 0]
    pool_a = particle_colors[vertex_a]
    pool_b = particle_colors[vertex_b]
    slot_a = wound_base[pool_a] + wp.atomic_add(wound_next, pool_a, 1)
    slot_b = wound_base[pool_b] + wp.atomic_add(wound_next, pool_b, 1)
    if slot_a >= spare_count or slot_b >= spare_count:
        # Not reached: check_cut_demand reserved both spares. Kept so that a wrong count parks the
        # rows of this edge and shows in the overflow, and never writes outside a pool.
        if slot_a >= spare_count:
            wp.atomic_add(overflow, pool_a, 1)
        if slot_b >= spare_count:
            wp.atomic_add(overflow, pool_b, 1)
        edge_cut[e] = 2
        return
    # The two tie targets are one step (edge_half_gap_t) on each side of the crossing point. Near an
    # edge end both move together, so they keep their distance and stay inside [t_min, 1 - t_min].
    # That distance is one start gap along the blade plane normal, or less on an edge that is too
    # short along the normal (set_cut_edge_parameters).
    s_a = edge_t[e] - edge_half_gap_t[e]
    s_b = edge_t[e] + edge_half_gap_t[e]
    if s_b > 1.0 - t_min:
        s_a -= s_b - (1.0 - t_min)
        s_b = 1.0 - t_min
    if s_a < t_min:
        s_b += t_min - s_a
        s_a = t_min
    s_b = wp.min(s_b, 1.0 - t_min)
    for side in range(2):
        p = spare_start[pool_a] + slot_a
        s = s_a
        if side == 1:
            p = spare_start[pool_b] + slot_b
            s = s_b
        corner_a, corner_b = _tie_corners(
            e, side, edge_owner, edge_owner_local, tet_indices, tet_vertex, particle_level
        )
        own = corner_a
        if side == 1:
            own = corner_b
        particle_q[p] = (1.0 - s) * particle_q[corner_a] + s * particle_q[corner_b]
        particle_qd[p] = (1.0 - s) * particle_qd[corner_a] + s * particle_qd[corner_b]
        particle_q_rest[p] = (1.0 - s) * particle_q_rest[corner_a] + s * particle_q_rest[corner_b]
        cut_force[p] = wp.min(cut_force[corner_a], cut_force[corner_b])
        suture_grip[p] = suture_grip[own]
        suture_mu[p] = suture_mu[own]
        suture_puncture_force[p] = suture_puncture_force[own]
        suture_puncture_cos[p] = suture_puncture_cos[own]
        suture_stiffness[p] = suture_stiffness[own]
        particle_gate[p] = (particle_gate[corner_a] | particle_gate[corner_b]) & (~cut_gate_bits)
        wound_edge[p] = e
        wound_side[p] = side
        wound_new[p] = 1
        particle_stroke[p] = stroke_number[0]
        tie_s[p] = s
        tie_dual[p] = wp.vec3(0.0)
        wp.atomic_add(tie_count, corner_a, 1)
        wp.atomic_add(tie_count, corner_b, 1)
        edge_wound[e, side] = p
    if front_rows != 0:
        slot = wp.atomic_add(edge_extra_taken, 0, 1)
        if slot >= front_row_count:
            wp.atomic_add(overflow, overflow_edge_slot, 1)  # no front row: the blade meets the wound particles only
        else:
            edge_row_mesh_edge[front_row_start + slot] = e
            edge_kind[front_row_start + slot] = XFEM_EDGE_FRONT


@wp.kernel
def mark_cut_front_edges(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    cut_force: wp.array[wp.float32],
    tet_indices: wp.array2d[wp.int32],
    tet_mesh_edge: wp.array2d[wp.int32],
    tet_side: wp.array[wp.int32],
    edge_front: wp.array[wp.int32],
):
    """One thread per tissue tet: a cuttable tet that is not cut marks its six mesh edges as front edges."""
    tet = wp.tid()
    if due[0] == 0 or advance[0] == 0 or tet_side[tet] != 0:
        return
    for c in range(4):
        if cut_force[tet_indices[tet, c]] >= CUT_FORCE_UNCUTTABLE:
            return
    for local in range(6):
        edge_front[tet_mesh_edge[tet, local]] = 1


@wp.kernel
def finish_wound_particles(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    first_spare: int,
    gate_bit: int,
    mass_fraction: float,
    edge_owner: wp.array2d[wp.int32],
    edge_owner_local: wp.array2d[wp.int32],
    edge_front: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
    wound_edge: wp.array[wp.int32],
    wound_side: wp.array[wp.int32],
    tie_count: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_gate: wp.array[wp.int32],
    wound_new: wp.array[wp.int32],
    mass_taken: wp.array[float],
):
    """One thread per pool spare: the gate bit of a wound particle, and the mass of a new one.

    A wound particle on a front edge has no gate bit, so the blade rests on it; every other wound
    particle has the bit. A new wound particle takes, from each of the two corners of its tie,
    ``mass_fraction`` of the corner mass divided by the number of new ties of that corner. The next
    launch subtracts the taken mass from the corners, so the total mass stays.
    """
    p = first_spare + wp.tid()
    if due[0] == 0 or advance[0] == 0:
        return
    e = wound_edge[p]
    if e < 0:
        return
    if edge_front[e] != 0:
        particle_gate[p] = particle_gate[p] & (~gate_bit)
    else:
        particle_gate[p] = particle_gate[p] | gate_bit
    if wound_new[p] == 0:
        return
    wound_new[p] = 0
    corner_a, corner_b = _tie_corners(
        e, wound_side[p], edge_owner, edge_owner_local, tet_indices, tet_vertex, particle_level
    )
    mass = float(0.0)
    flags = int(0)
    for k in range(2):
        corner = corner_a
        if k == 1:
            corner = corner_b
        if (particle_flags[corner] & ParticleFlags.ACTIVE) != 0 and particle_mass[corner] > 0.0:
            taken = mass_fraction * particle_mass[corner] / float(wp.max(tie_count[corner], 1))
            wp.atomic_add(mass_taken, corner, taken)
            mass += taken
            flags = particle_flags[corner]
    if mass > 0.0:
        particle_mass[p] = mass
        particle_inv_mass[p] = 1.0 / mass
        particle_flags[p] = flags


@wp.kernel
def apply_wound_mass_take(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    tie_count: wp.array[wp.int32],
    mass_taken: wp.array[float],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
):
    """One thread per particle: subtract the mass that new wound particles took from a corner of their tie."""
    p = wp.tid()
    if due[0] == 0 or advance[0] == 0 or tie_count[p] == 0:
        return
    tie_count[p] = 0
    taken = mass_taken[p]
    mass_taken[p] = 0.0
    if taken > 0.0:
        mass = particle_mass[p] - taken
        particle_mass[p] = mass
        particle_inv_mass[p] = 1.0 / mass


# =====================================================================
# The tie
# =====================================================================


@wp.kernel
def accumulate_cut_tie_force_and_hessian(
    current_color: int,
    first_spare: int,
    stiffness: float,
    particle_colors: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    edge_owner: wp.array2d[wp.int32],
    edge_owner_local: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
    wound_edge: wp.array[wp.int32],
    wound_side: wp.array[wp.int32],
    tie_s: wp.array[float],
    tie_dual: wp.array[wp.vec3],
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """One thread per pool spare: the tie of a wound particle, added to its particles of the current color.

    The tie energy is ``dual . r + 0.5 * stiffness * |r|^2`` with ``r = x_p - (1 - s) * x_a - s * x_b``,
    where ``x_a`` and ``x_b`` are the two corners of the crossed edge in the owner row of the side
    of the wound particle. The corners are read at every call, so a corner that a later cut advance
    gives to a copy needs no update.
    """
    p = first_spare + wp.tid()
    e = wound_edge[p]
    if e < 0:
        return
    corner_a, corner_b = _tie_corners(
        e, wound_side[p], edge_owner, edge_owner_local, tet_indices, tet_vertex, particle_level
    )
    on_p = particle_colors[p] == current_color
    on_a = particle_colors[corner_a] == current_color
    on_b = particle_colors[corner_b] == current_color
    if not (on_p or on_a or on_b):
        return
    s = tie_s[p]
    residual = pos[p] - (1.0 - s) * pos[corner_a] - s * pos[corner_b]
    force = tie_dual[p] + stiffness * residual
    hessian = stiffness * wp.identity(n=3, dtype=float)
    if on_p:
        wp.atomic_add(particle_forces, p, -force)
        wp.atomic_add(particle_hessians, p, hessian)
    if on_a:
        wp.atomic_add(particle_forces, corner_a, (1.0 - s) * force)
        wp.atomic_add(particle_hessians, corner_a, (1.0 - s) * (1.0 - s) * hessian)
    if on_b:
        wp.atomic_add(particle_forces, corner_b, s * force)
        wp.atomic_add(particle_hessians, corner_b, s * s * hessian)


@wp.kernel
def update_duals_cut_ties(
    first_spare: int,
    stiffness: float,
    use_dual: int,
    pos: wp.array[wp.vec3],
    edge_owner: wp.array2d[wp.int32],
    edge_owner_local: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
    wound_edge: wp.array[wp.int32],
    wound_side: wp.array[wp.int32],
    tie_s: wp.array[float],
    tie_dual: wp.array[wp.vec3],
    tie_residual: wp.array[float],
):
    """One thread per pool spare, once per iteration: ``dual += stiffness * r``, and the tie residual ``|r|`` [m]."""
    p = first_spare + wp.tid()
    e = wound_edge[p]
    if e < 0:
        return
    corner_a, corner_b = _tie_corners(
        e, wound_side[p], edge_owner, edge_owner_local, tet_indices, tet_vertex, particle_level
    )
    s = tie_s[p]
    residual = pos[p] - (1.0 - s) * pos[corner_a] - s * pos[corner_b]
    if use_dual != 0:
        tie_dual[p] = tie_dual[p] + stiffness * residual
    tie_residual[p] = wp.length(residual)


# =====================================================================
# Surface rows
# =====================================================================


@wp.kernel
def take_cut_skin_triangle_rows(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    tri_spare_start: int,
    extra_start: int,
    extra_count: int,
    overflow_slot: int,
    tet_side: wp.array[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    particle_level: wp.array[float],
    tri_source: wp.array2d[wp.int32],
    tri_mesh_edge: wp.array2d[wp.int32],
    tri_kind: wp.array[wp.int32],
    tri_sub: wp.array[wp.int32],
    tri_extra: wp.array[wp.int32],
    extra_taken: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    """One thread per original triangle row: a skin triangle that a cut tet clips takes two extra rows."""
    row = wp.tid()
    if due[0] == 0 or advance[0] == 0 or row >= tri_spare_start:
        return
    if tri_kind[row] != XFEM_ROW_SKIN or tri_extra[row] != -1:
        return
    tet = tri_source[row, 0]
    if tet_side[tet] == 0:
        return
    side_0 = _level_side(particle_level[tet_vertex[tet, tri_source[row, 1]]])
    side_1 = _level_side(particle_level[tet_vertex[tet, tri_source[row, 2]]])
    side_2 = _level_side(particle_level[tet_vertex[tet, tri_source[row, 3]]])
    if side_0 == side_1 and side_1 == side_2:
        return
    slot = wp.atomic_add(extra_taken, 0, 2)
    if slot + 2 > extra_count:
        wp.atomic_add(overflow, overflow_slot, 1)
        tri_extra[row] = -2  # no extra rows: the quad part of this triangle stays open
        return
    first = extra_start + slot
    for j in range(2):
        for k in range(4):
            tri_source[first + j, k] = tri_source[row, k]
        for k in range(3):
            tri_mesh_edge[first + j, k] = tri_mesh_edge[row, k]
        tri_sub[first + j] = 1 + j
        tri_kind[first + j] = XFEM_ROW_SKIN
    tri_extra[row] = first


@wp.kernel
def take_cut_skin_edge_rows(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    edge_spare_start: int,
    extra_start: int,
    extra_count: int,
    overflow_slot: int,
    edge_wound: wp.array2d[wp.int32],
    edge_source: wp.array2d[wp.int32],
    edge_row_mesh_edge: wp.array[wp.int32],
    edge_kind: wp.array[wp.int32],
    edge_extra: wp.array[wp.int32],
    extra_taken: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    """One thread per original edge row: a skin edge that got wound particles takes one extra row for its side B part."""
    row = wp.tid()
    if due[0] == 0 or advance[0] == 0 or row >= edge_spare_start:
        return
    if edge_kind[row] != XFEM_ROW_SKIN or edge_extra[row] != -1:
        return
    e = edge_row_mesh_edge[row]
    if edge_wound[e, 0] < 0:
        return
    slot = wp.atomic_add(extra_taken, 0, 1)
    if slot >= extra_count:
        wp.atomic_add(overflow, overflow_slot, 1)
        edge_extra[row] = -2  # no extra row: the side B part of this edge stays open
        return
    extra = extra_start + slot
    for k in range(3):
        edge_source[extra, k] = edge_source[row, k]
    edge_row_mesh_edge[extra] = e
    edge_kind[extra] = XFEM_EDGE_SKIN_B
    edge_extra[row] = extra


@wp.kernel
def write_cut_triangle_rows(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    tri_spare_start: int,
    row_park_particle: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    tet_mesh_edge: wp.array2d[wp.int32],
    tet_side: wp.array[wp.int32],
    tet_twin: wp.array[wp.int32],
    slot_tet: wp.array[wp.int32],
    particle_level: wp.array[float],
    edge_wound: wp.array2d[wp.int32],
    tri_source: wp.array2d[wp.int32],
    tri_mesh_edge: wp.array2d[wp.int32],
    tri_kind: wp.array[wp.int32],
    tri_sub: wp.array[wp.int32],
    tri_slot: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
):
    """One thread per triangle row: write the three particles of the row from its kind.

    - Skin (sub 0 to 2): the face of a tet that is not cut; the face read through the fragment row
      of its side when the three corners are on one side; else the clipped face: sub 0 is the
      triangle of the lone corner, sub 1 and 2 are the quad of the other side as a fan from its
      first vertex. The order keeps the outward direction of the face.
    - Wound (sub 0 to 3; side ``sub // 2``): the cut polygon of the tet, three wound particles when
      one corner is alone on a side, four (as a fan) when two corners are on each side. The normal
      points from the side of the fragment to the far side.

    A row is all live or all parked. A row that needs a missing wound particle is parked.
    """
    row = wp.tid()
    if due[0] == 0 or advance[0] == 0:
        return
    kind = tri_kind[row]
    if kind == XFEM_ROW_FIXED:
        return
    park = row_park_particle[0]
    if row >= tri_spare_start:
        park = row_park_particle[row - tri_spare_start]
    i0 = int(-1)
    i1 = int(-1)
    i2 = int(-1)
    sub = tri_sub[row]
    if kind == XFEM_ROW_SKIN:
        tet = tri_source[row, 0]
        c0 = tri_source[row, 1]
        c1 = tri_source[row, 2]
        c2 = tri_source[row, 3]
        if tet_side[tet] == 0:
            if sub == 0:
                i0 = tet_indices[tet, c0]
                i1 = tet_indices[tet, c1]
                i2 = tet_indices[tet, c2]
        else:
            corners = wp.vec3i(c0, c1, c2)
            sides = wp.vec3i(0, 0, 0)
            count_b = int(0)
            for k in range(3):
                sides[k] = _level_side(particle_level[tet_vertex[tet, corners[k]]])
                count_b += sides[k]
            rows = wp.vec2i(tet, tet_twin[tet])  # the fragment row of side A and of side B
            if count_b == 0 or count_b == 3:
                if sub == 0:
                    i0 = tet_indices[rows[sides[0]], c0]
                    i1 = tet_indices[rows[sides[0]], c1]
                    i2 = tet_indices[rows[sides[0]], c2]
            else:
                lone = int(0)
                for k in range(3):
                    if (count_b == 1 and sides[k] == 1) or (count_b == 2 and sides[k] == 0):
                        lone = k
                k_p = (lone + 1) % 3
                k_q = (lone + 2) % 3
                side_lone = sides[lone]
                side_other = 1 - side_lone
                edge_lp = tri_mesh_edge[row, lone]  # between the lone corner and the next corner
                edge_ql = tri_mesh_edge[row, k_q]  # between the last corner and the lone corner
                if sub == 0:
                    i0 = tet_indices[rows[side_lone], corners[lone]]
                    i1 = edge_wound[edge_lp, side_lone]
                    i2 = edge_wound[edge_ql, side_lone]
                elif sub == 1:
                    i0 = edge_wound[edge_lp, side_other]
                    i1 = tet_indices[rows[side_other], corners[k_p]]
                    i2 = tet_indices[rows[side_other], corners[k_q]]
                else:
                    i0 = edge_wound[edge_lp, side_other]
                    i1 = tet_indices[rows[side_other], corners[k_q]]
                    i2 = edge_wound[edge_ql, side_other]
    elif kind == XFEM_ROW_WOUND:
        tet = slot_tet[tri_slot[row]]
        if tet >= 0:
            side = sub // 2
            part = sub % 2
            tet_sides = wp.vec4i(0, 0, 0, 0)
            count_b = int(0)
            for c in range(4):
                tet_sides[c] = _level_side(particle_level[tet_vertex[tet, c]])
                count_b += tet_sides[c]
            if count_b == 1 or count_b == 3:
                if part == 0:
                    lone = int(0)
                    for c in range(4):
                        if (count_b == 1 and tet_sides[c] == 1) or (count_b == 3 and tet_sides[c] == 0):
                            lone = c
                    # The triangle on the three edges at the lone corner, in the order of the face
                    # opposite to it, has its normal away from the lone corner.
                    w0 = edge_wound[tet_mesh_edge[tet, _edge_index(lone, TET_FACES[lone, 0])], side]
                    w1 = edge_wound[tet_mesh_edge[tet, _edge_index(lone, TET_FACES[lone, 1])], side]
                    w2 = edge_wound[tet_mesh_edge[tet, _edge_index(lone, TET_FACES[lone, 2])], side]
                    i0 = w0
                    if side == tet_sides[lone]:
                        i1 = w1
                        i2 = w2
                    else:
                        i1 = w2
                        i2 = w1
            elif count_b == 2:
                p = int(-1)
                q = int(-1)
                r = int(-1)
                s = int(-1)
                for c in range(4):
                    if tet_sides[c] == 0:
                        if p < 0:
                            p = c
                        else:
                            q = c
                    else:
                        if r < 0:
                            r = c
                        else:
                            s = c
                # The quad (pr, ps, qs, qr) has its normal from side A to side B when (p, q, r, s)
                # is an even permutation of the tet corners, which is when p + q is odd.
                w0 = edge_wound[tet_mesh_edge[tet, _edge_index(p, r)], side]
                w1 = edge_wound[tet_mesh_edge[tet, _edge_index(p, s)], side]
                w2 = edge_wound[tet_mesh_edge[tet, _edge_index(q, s)], side]
                w3 = edge_wound[tet_mesh_edge[tet, _edge_index(q, r)], side]
                a_to_b = (p + q) % 2 == 1
                if (a_to_b and side == 1) or ((not a_to_b) and side == 0):
                    w_swap = w1
                    w1 = w3
                    w3 = w_swap
                i0 = w0
                if part == 0:
                    i1 = w1
                    i2 = w2
                else:
                    i1 = w2
                    i2 = w3
    if i0 < 0 or i1 < 0 or i2 < 0:
        i0 = park
        i1 = park
        i2 = park
    tri_indices[row, 0] = i0
    tri_indices[row, 1] = i1
    tri_indices[row, 2] = i2


@wp.kernel
def write_cut_edge_rows(
    due: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    edge_spare_start: int,
    tri_spare_count: int,
    row_park_particle: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    tet_mesh_edge: wp.array2d[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    tet_side: wp.array[wp.int32],
    tet_twin: wp.array[wp.int32],
    slot_tet: wp.array[wp.int32],
    particle_level: wp.array[float],
    edge_wound: wp.array2d[wp.int32],
    edge_front: wp.array[wp.int32],
    edge_source: wp.array2d[wp.int32],
    edge_row_mesh_edge: wp.array[wp.int32],
    edge_kind: wp.array[wp.int32],
    edge_sub: wp.array[wp.int32],
    edge_slot: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
):
    """One thread per edge row: write the two edge particles of the row from its kind.

    - Skin: the mesh edge read through its source tet; through the fragment row of its side when the
      tet is cut; (side A vertex, wound particle of side A) when the edge has wound particles.
    - Side B part of a cut skin edge: (wound particle of side B, side B vertex).
    - Wound segment (sub 0 to 9; side ``sub // 5``): parts 0 to 3 are the segment in each tet face
      (the two wound particles on the two crossed edges of the face), part 4 is the diagonal of a
      quad. The tet with the smaller row writes the segment in a face that two cut tets share.
    - Front row: (wound particle of side A, wound particle of side B) of a front edge, the part of
      the mesh edge that still joins the two sides. The blade edge rests on it. It is parked when
      no tet that is not cut holds the edge any more, and it takes no part in self contact
      (:func:`filter_cut_self_contacts`).

    A row is all live or all parked. A row that needs a missing wound particle is parked.
    """
    row = wp.tid()
    if due[0] == 0 or advance[0] == 0:
        return
    kind = edge_kind[row]
    if kind == XFEM_ROW_FIXED:
        return
    park = row_park_particle[tri_spare_count]
    if row >= edge_spare_start:
        park = row_park_particle[tri_spare_count + row - edge_spare_start]
    i0 = int(-1)
    i1 = int(-1)
    if kind == XFEM_ROW_SKIN or kind == XFEM_EDGE_SKIN_B:
        tet = edge_source[row, 0]
        c0 = edge_source[row, 1]
        c1 = edge_source[row, 2]
        e = edge_row_mesh_edge[row]
        rows = wp.vec2i(tet, tet)  # the row that holds the material of side A and of side B
        if tet_side[tet] != 0:
            rows[1] = tet_twin[tet]
        side_0 = _level_side(particle_level[tet_vertex[tet, c0]])
        side_1 = _level_side(particle_level[tet_vertex[tet, c1]])
        if edge_wound[e, 0] >= 0 and edge_wound[e, 1] >= 0:
            c_a = c0
            c_b = c1
            if side_0 == 1:
                c_a = c1
                c_b = c0
            if kind == XFEM_ROW_SKIN:
                i0 = tet_indices[rows[0], c_a]
                i1 = edge_wound[e, 0]
            else:
                i0 = edge_wound[e, 1]
                i1 = tet_indices[rows[1], c_b]
        elif kind == XFEM_ROW_SKIN:
            if tet_side[tet] == 0:
                i0 = tet_indices[tet, c0]
                i1 = tet_indices[tet, c1]
            elif side_0 == side_1:
                i0 = tet_indices[rows[side_0], c0]
                i1 = tet_indices[rows[side_0], c1]
    elif kind == XFEM_EDGE_FRONT:
        e = edge_row_mesh_edge[row]
        if edge_front[e] != 0:
            i0 = edge_wound[e, 0]
            i1 = edge_wound[e, 1]
    elif kind == XFEM_EDGE_WOUND:
        tet = slot_tet[edge_slot[row]]
        if tet >= 0:
            sub = edge_sub[row]
            side = sub // 5
            part = sub % 5
            sides = wp.vec4i(0, 0, 0, 0)
            count_b = int(0)
            for c in range(4):
                sides[c] = _level_side(particle_level[tet_vertex[tet, c]])
                count_b += sides[c]
            if part < 4:
                other = tet_neighbors[tet, part]
                written_by_other = other >= 0 and other < tet and tet_side[other] != 0
                face_b = sides[TET_FACES[part, 0]] + sides[TET_FACES[part, 1]] + sides[TET_FACES[part, 2]]
                if (face_b == 1 or face_b == 2) and not written_by_other:
                    lone = int(0)
                    for k in range(3):
                        side_k = sides[TET_FACES[part, k]]
                        if (face_b == 1 and side_k == 1) or (face_b == 2 and side_k == 0):
                            lone = k
                    c_lone = TET_FACES[part, lone]
                    c_p = TET_FACES[part, (lone + 1) % 3]
                    c_q = TET_FACES[part, (lone + 2) % 3]
                    i0 = edge_wound[tet_mesh_edge[tet, _edge_index(c_lone, c_p)], side]
                    i1 = edge_wound[tet_mesh_edge[tet, _edge_index(c_lone, c_q)], side]
            elif count_b == 2:
                p = int(-1)
                q = int(-1)
                r = int(-1)
                s = int(-1)
                for c in range(4):
                    if sides[c] == 0:
                        if p < 0:
                            p = c
                        else:
                            q = c
                    else:
                        if r < 0:
                            r = c
                        else:
                            s = c
                i0 = edge_wound[tet_mesh_edge[tet, _edge_index(p, r)], side]
                i1 = edge_wound[tet_mesh_edge[tet, _edge_index(q, s)], side]
    if i0 < 0 or i1 < 0:
        i0 = park
        i1 = park
    edge_indices[row, 2] = i0
    edge_indices[row, 3] = i1


# =====================================================================
# Self contact near a cut
# =====================================================================


@wp.func
def _cut_side_tag(
    particle: int,
    wound_edge: wp.array[wp.int32],
    wound_side: wp.array[wp.int32],
    particle_level: wp.array[float],
    particle_level_set: wp.array[wp.int32],
    particle_stroke: wp.array[wp.int32],
) -> int:
    """Tag of the side of a cut that a particle belongs to; -1 for a particle that is not at a cut.

    The tag is ``2 * stroke + side`` with side 0 (A) or 1 (B), so the sides of two cuts made in
    two strokes never have the same tag. A wound particle has the side of its wound face and the
    stroke that made it. A vertex of a cut tet has the side of its level value and the stroke
    that set the value.
    """
    if wound_edge[particle] >= 0:
        return 2 * particle_stroke[particle] + wound_side[particle]
    if particle_level_set[particle] == 2:
        return 2 * particle_stroke[particle] + _level_side(particle_level[particle])
    return -1


@wp.kernel
def filter_cut_self_contacts(
    use_filter: int,
    front_rows: int,
    rest_radius: float,
    edge_kind: wp.array[wp.int32],
    particle_ghost: wp.array[wp.int32],
    wound_edge: wp.array[wp.int32],
    wound_side: wp.array[wp.int32],
    particle_level: wp.array[float],
    particle_level_set: wp.array[wp.int32],
    particle_stroke: wp.array[wp.int32],
    particle_q_rest: wp.array[wp.vec3],
    edge_owner: wp.array2d[wp.int32],
    edge_owner_local: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_vertex: wp.array2d[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info: TriMeshCollisionInfo,
):
    """One thread per particle and per edge row, after each self-contact detection: remove the pairs that a cut forbids.

    The contact force and the penetration-free truncation skip a list entry with the row -1 and a
    list with the count 0, so the removal acts on both.

    - A ghost (a virtual corner) is not material: it starts at the place of a real vertex of the
      other side. Its vertex-triangle list gets the count 0.
    - With ``use_filter``, a pair whose particles all have the same side tag (the same side of
      the same cut) and whose two parts are nearer than ``rest_radius`` in the rest shape is
      removed (rule "same side"): the wound face of a fragment lies half a start gap inside the
      material of its own side, next to the corners of that fragment, and its small triangles
      are closer to each other than the contact margin. The rest positions are those of
      ``particle_q_rest``; a wound particle has its tie target there. A pair of one side that is
      far apart in the rest shape stays: it is contact of a part that folds onto itself.
    - With ``use_filter``, a pair of a wound particle and a row that names a corner of its tie is
      removed (rule "tie corner"): a wound particle lies on the skin triangles of a tet that is
      not cut yet and holds its edge.
    - With ``front_rows``, every pair with a front row is removed: a front row lies between the two
      wound faces and is for the blade only.
    """
    i = wp.tid()
    if i < collision_info.vertex_colliding_triangles_count.shape[0]:
        if particle_ghost[i] != 0:
            collision_info.vertex_colliding_triangles_count[i] = 0
        elif use_filter != 0:
            tag = _cut_side_tag(i, wound_edge, wound_side, particle_level, particle_level_set, particle_stroke)
            if tag >= 0:
                corner_a = int(-1)
                corner_b = int(-1)
                if wound_edge[i] >= 0:
                    corner_a, corner_b = _tie_corners(
                        wound_edge[i],
                        wound_side[i],
                        edge_owner,
                        edge_owner_local,
                        tet_indices,
                        tet_vertex,
                        particle_level,
                    )
                offset = collision_info.vertex_colliding_triangles_offsets[i]
                count = wp.min(
                    collision_info.vertex_colliding_triangles_count[i],
                    collision_info.vertex_colliding_triangles_buffer_sizes[i],
                )
                for k in range(count):
                    tri = collision_info.vertex_colliding_triangles[2 * (offset + k) + 1]
                    if tri < 0:
                        continue
                    same_side = int(1)
                    tie_corner = int(0)
                    for c in range(3):
                        q = tri_indices[tri, c]
                        if (
                            _cut_side_tag(
                                q, wound_edge, wound_side, particle_level, particle_level_set, particle_stroke
                            )
                            != tag
                        ):
                            same_side = 0
                        if q == corner_a or q == corner_b:
                            tie_corner = 1
                    if same_side != 0:
                        rest = particle_q_rest[i]
                        closest, _bary, _feature = triangle_closest_point(
                            particle_q_rest[tri_indices[tri, 0]],
                            particle_q_rest[tri_indices[tri, 1]],
                            particle_q_rest[tri_indices[tri, 2]],
                            rest,
                        )
                        if wp.length_sq(rest - closest) >= rest_radius * rest_radius:
                            same_side = 0
                    if same_side != 0 or tie_corner != 0:
                        collision_info.vertex_colliding_triangles[2 * (offset + k) + 1] = -1
    if i < collision_info.edge_colliding_edges_count.shape[0]:
        if edge_kind[i] == XFEM_EDGE_FRONT:
            collision_info.edge_colliding_edges_count[i] = 0
        elif use_filter != 0 or front_rows != 0:
            v0 = edge_indices[i, 2]
            v1 = edge_indices[i, 3]
            tag_0 = _cut_side_tag(v0, wound_edge, wound_side, particle_level, particle_level_set, particle_stroke)
            tag_1 = _cut_side_tag(v1, wound_edge, wound_side, particle_level, particle_level_set, particle_stroke)
            if front_rows != 0 or tag_0 >= 0 or tag_1 >= 0:
                offset = collision_info.edge_colliding_edges_offsets[i]
                count = wp.min(
                    collision_info.edge_colliding_edges_count[i], collision_info.edge_colliding_edges_buffer_sizes[i]
                )
                for k in range(count):
                    other = collision_info.edge_colliding_edges[2 * (offset + k) + 1]
                    if other < 0:
                        continue
                    remove = int(0)
                    if edge_kind[other] == XFEM_EDGE_FRONT:
                        remove = 1
                    elif use_filter != 0 and (tag_0 >= 0 or tag_1 >= 0):
                        u0 = edge_indices[other, 2]
                        u1 = edge_indices[other, 3]
                        if tag_0 >= 0 and tag_0 == tag_1:
                            if (
                                _cut_side_tag(
                                    u0, wound_edge, wound_side, particle_level, particle_level_set, particle_stroke
                                )
                                == tag_0
                                and _cut_side_tag(
                                    u1, wound_edge, wound_side, particle_level, particle_level_set, particle_stroke
                                )
                                == tag_0
                            ):
                                # Same side: remove the pair when the two edges are near in the rest shape.
                                nearest = wp.closest_point_edge_edge(
                                    particle_q_rest[v0],
                                    particle_q_rest[v1],
                                    particle_q_rest[u0],
                                    particle_q_rest[u1],
                                    1.0e-9,
                                )
                                if nearest[2] < rest_radius:
                                    remove = 1
                        # Tie corner rule, the same from both rows of the pair.
                        for end in range(4):
                            w = v0
                            x0 = u0
                            x1 = u1
                            if end == 1:
                                w = v1
                            elif end == 2:
                                w = u0
                                x0 = v0
                                x1 = v1
                            elif end == 3:
                                w = u1
                                x0 = v0
                                x1 = v1
                            if wound_edge[w] >= 0:
                                corner_a, corner_b = _tie_corners(
                                    wound_edge[w],
                                    wound_side[w],
                                    edge_owner,
                                    edge_owner_local,
                                    tet_indices,
                                    tet_vertex,
                                    particle_level,
                                )
                                if x0 == corner_a or x0 == corner_b or x1 == corner_a or x1 == corner_b:
                                    remove = 1
                    if remove != 0:
                        collision_info.edge_colliding_edges[2 * (offset + k) + 1] = -1
