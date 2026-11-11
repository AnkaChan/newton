# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cutting in SolverVBD (experimental): blade force, cut faces, vertex split, start gap and surface rows.

A cut blade (see :meth:`newton.ModelBuilder.add_cut_blade`) cuts a tetrahedral soft body along
existing tet faces. No tet is added, removed or reshaped. A cut vertex keeps its particle for one
group of tets and gets a copy, taken from the spare particles of
:meth:`newton.ModelBuilder.add_cut_capacity`, for each other group. All kernels have fixed launch
sizes and edit array contents only, so a captured CUDA graph can replay them.

Words: a *tet record* is one (tet, corner) entry of the vertex-to-tet table; the *idle tet* has
zero stiffness and fills unused tet records; a *cut face* is an interior tet face that a cut
separated; a *fan* is the run of tets around one mesh edge that are joined through uncut faces.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...geometry.soft_contacts_mesh import _soft_contact_force_enabled, _SoftContactFilterData
from ...utils.mesh import MeshAdjacencyData
from .rigid_vbd_kernels import _evaluate_rigid_soft_contact_force_norm

wp.set_module_options({"enable_backward": False})

# A particle whose ``vbd.cut_force`` is at or above this value is not cuttable.
CUT_FORCE_UNCUTTABLE = wp.constant(1.0e30)

# Contact gate bit of the wound particles of a cut (needle bits are ``1 << needle`` with needle < 30).
CUT_GATE_BIT = 1 << 30

# Contact gate bit 31 (the sign bit of an int32) of a ghost: a virtual corner of the XFEM-type cut
# (``cut_method="xfem"``) is not material, so every shape carries this bit and has no contact with it.
CUT_GHOST_GATE_BIT = -(1 << 31)

# Largest number of distinct cut face normals at one particle that the start gap direction handles.
_MAX_GAP_NORMALS = 16

_mat43i = wp.types.matrix(shape=(4, 3), dtype=wp.int32)
_mat_normals = wp.types.matrix(shape=(_MAX_GAP_NORMALS, 3), dtype=wp.float32)

# The face opposite local vertex i of a tet, as local indices with outward winding
# (the same table as ``suture_kernels._TET_FACES``).
_TET_FACES_HOST = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
TET_FACES = wp.constant(_mat43i(1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1))

# The six edges of a tet as local vertex pairs (the same table as ``solver_vbd._TET_EDGE_PAIRS``).
_TET_EDGE_PAIRS_HOST = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


# =====================================================================
# Host helpers (solver construction)
# =====================================================================


def build_padded_tet_adjacency(
    v_adj_tets: np.ndarray,
    v_adj_tets_offsets: np.ndarray,
    spare_particles: np.ndarray,
    idle_tet: int,
    *,
    dropped_tets: np.ndarray | None = None,
    doubled_particles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a vertex-to-tet table in which every spare particle has room for the tets of a copy.

    Args:
        v_adj_tets: Flat (tet, corner) pairs of the model's table.
        v_adj_tets_offsets: Offsets into ``v_adj_tets`` in int32 values, shape ``[particle_count + 1]``.
        spare_particles: Indices of the spare particles.
        idle_tet: Index of the idle tet.
        dropped_tets: Indices of tets whose tet records are left out (the spare tets: they sit
            on the corners of the idle tet until a cut writes a fragment into them).
        doubled_particles: Indices of particles that get as many free tet records as they have tets
            (the XFEM-type cut: a vertex at the cut front is a corner of both fragment rows of a
            cut tet). With this argument a spare particle gets ``2 * Q`` tet records.

    Returns:
        The new flat pairs, the new offsets, and ``Q``, the largest number of tets at one vertex.
        A spare particle gets ``Q`` tet records ``(idle_tet, 0)``; a free tet record of another
        particle is ``(idle_tet, 0)`` too.
    """
    offsets = np.asarray(v_adj_tets_offsets, dtype=np.int64)
    pairs = np.asarray(v_adj_tets, dtype=np.int32).reshape(-1, 2)
    particle_count = offsets.shape[0] - 1
    owner = np.repeat(np.arange(particle_count), (offsets[1:] - offsets[:-1]) // 2)
    if dropped_tets is not None and len(dropped_tets) > 0:
        keep = ~np.isin(pairs[:, 0], dropped_tets)
        pairs = pairs[keep]
        owner = owner[keep]
    counts = np.bincount(owner, minlength=particle_count).astype(np.int64)
    is_spare = np.zeros(particle_count, dtype=bool)
    is_spare[spare_particles] = True
    q = int(counts[~is_spare].max()) if (~is_spare).any() else 0
    new_counts = counts.copy()
    if doubled_particles is not None:
        new_counts[doubled_particles] = 2 * counts[doubled_particles]
        new_counts[spare_particles] = 2 * q
    else:
        new_counts[spare_particles] = q
    new_offsets = np.zeros(particle_count + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(2 * new_counts)
    new_pairs = np.empty((int(new_offsets[-1]) // 2, 2), dtype=np.int32)
    new_pairs[:, 0] = idle_tet
    new_pairs[:, 1] = 0
    # The records of one particle stay in their order at the start of its block.
    first = np.zeros(particle_count + 1, dtype=np.int64)
    first[1:] = np.cumsum(counts)
    live = ~is_spare[owner]
    rank = np.arange(len(owner)) - first[owner]
    new_pairs[new_offsets[owner[live]] // 2 + rank[live]] = pairs[live]
    return new_pairs.reshape(-1), new_offsets.astype(np.int32), q


def build_cut_row_sources(
    tet_indices: np.ndarray,
    tet_neighbors: np.ndarray,
    tri_indices: np.ndarray,
    edge_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the tet face behind every skin triangle row and the tet edge behind every skin edge row.

    A row that matches no tet (a spare row, a cloth triangle) gets the source tet -1.

    Returns:
        ``tri_source`` ``[tri_count, 4]`` (tet, three local corners in row order),
        ``edge_source`` ``[edge_count, 3]`` (tet, two local corners in row order),
        ``tet_face_row`` ``[tet_count, 4]`` and ``tet_edge_row`` ``[tet_count, 6]`` (row index or -1).
    """
    tets = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)
    tris = np.asarray(tri_indices, dtype=np.int64).reshape(-1, 3)
    edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 4)
    tri_source = np.full((tris.shape[0], 4), -1, dtype=np.int32)
    edge_source = np.full((edges.shape[0], 3), -1, dtype=np.int32)
    tet_face_row = np.full((tets.shape[0], 4), -1, dtype=np.int32)
    tet_edge_row = np.full((tets.shape[0], 6), -1, dtype=np.int32)

    open_faces = {}
    for tet, face in zip(*np.nonzero(np.asarray(tet_neighbors) < 0), strict=True):
        corners = _TET_FACES_HOST[face]
        open_faces[tuple(sorted(int(tets[tet, c]) for c in corners))] = (int(tet), int(face))
    for row in range(tris.shape[0]):
        ids = [int(i) for i in tris[row]]
        if len(set(ids)) < 3:
            continue
        found = open_faces.get(tuple(sorted(ids)))
        if found is None:
            continue
        tet, face = found
        corner_of = {int(tets[tet, c]): c for c in range(4)}
        tri_source[row] = (tet, corner_of[ids[0]], corner_of[ids[1]], corner_of[ids[2]])
        tet_face_row[tet, face] = row

    # The tet with the lowest index that holds each mesh edge.
    first_tet = {}
    for tet in range(tets.shape[0] - 1, -1, -1):
        for a, b in _TET_EDGE_PAIRS_HOST:
            i, j = int(tets[tet, a]), int(tets[tet, b])
            first_tet[(i, j) if i < j else (j, i)] = tet
    for row in range(edges.shape[0]):
        i, j = int(edges[row, 2]), int(edges[row, 3])
        if i == j:
            continue
        tet = first_tet.get((i, j) if i < j else (j, i))
        if tet is None:
            continue
        corner_of = {int(tets[tet, c]): c for c in range(4)}
        a, b = corner_of[i], corner_of[j]
        edge_source[row] = (tet, a, b)
        tet_edge_row[tet, _TET_EDGE_PAIRS_HOST.index((min(a, b), max(a, b)))] = row
    return tri_source, edge_source, tet_face_row, tet_edge_row


# =====================================================================
# Small functions
# =====================================================================


@wp.func
def _edge_index(a: int, b: int) -> int:
    """Index of the tet edge between local corners ``a`` and ``b`` in the order (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)."""
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    if lo == 0:
        return hi - 1
    if lo == 1:
        return hi + 1
    return 5


@wp.func
def _segment_distance_sq(x: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    d = b - a
    dd = wp.dot(d, d)
    t = float(0.0)
    if dd > 0.0:
        t = wp.clamp(wp.dot(x - a, d) / dd, 0.0, 1.0)
    return wp.length_sq(x - (a + t * d))


@wp.func
def _blade_edge(
    blade: int,
    blade_body: wp.array[wp.int32],
    blade_edge_shape: wp.array[wp.int32],
    blade_direction: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
):
    """World-space blade edge segment (a, b) and cutting direction d of one blade."""
    X_wb = body_q[blade_body[blade]]
    shape = blade_edge_shape[blade]
    X_ws = wp.transform_multiply(X_wb, shape_transform[shape])
    half_height = shape_scale[shape][1]
    a = wp.transform_point(X_ws, wp.vec3(0.0, 0.0, -half_height))
    b = wp.transform_point(X_ws, wp.vec3(0.0, 0.0, half_height))
    d = wp.normalize(wp.transform_vector(X_wb, blade_direction[blade]))
    return a, b, d


@wp.func
def _tet_rest_volume(tet: int, tet_poses: wp.array[wp.mat33]) -> float:
    det = wp.determinant(tet_poses[tet])
    if det == 0.0:
        return 0.0
    return wp.abs(1.0 / (6.0 * det))


@wp.func
def _tet_holds(tet: int, particle: int, tet_indices: wp.array2d[wp.int32]) -> bool:
    return (
        tet_indices[tet, 0] == particle
        or tet_indices[tet, 1] == particle
        or tet_indices[tet, 2] == particle
        or tet_indices[tet, 3] == particle
    )


@wp.func
def _side_label(
    tet: int,
    keep_side_labels: int,
    a: wp.vec3,
    n: wp.vec3,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    tet_side: wp.array[wp.int32],
) -> int:
    """Side of the blade plane that holds the tet centroid: +1 (side B) or -1 (side A); kept for the stroke."""
    if keep_side_labels != 0:
        kept = tet_side[tet]
        if kept != 0:
            return kept
    centroid = 0.25 * (
        particle_q[tet_indices[tet, 0]]
        + particle_q[tet_indices[tet, 1]]
        + particle_q[tet_indices[tet, 2]]
        + particle_q[tet_indices[tet, 3]]
    )
    if wp.dot(n, centroid - a) >= 0.0:
        return 1
    return -1


# =====================================================================
# Chapter 2: the blade force
# =====================================================================


@wp.kernel
def begin_cut_update(
    interval: int,
    counter: wp.array[wp.int32],
    due: wp.array[wp.int32],
    update_number: wp.array[wp.int32],
    blade_force: wp.array[float],
    blade_record_key: wp.array[wp.int64],
    contact_count: wp.array[wp.int32],
):
    """Advance the substep counter, decide whether this substep is a cut update, and clear the blade sums."""
    counter[0] = counter[0] + 1
    if counter[0] % interval == 0:
        due[0] = 1
        update_number[0] = update_number[0] + 1
        for blade in range(blade_force.shape[0]):
            blade_force[blade] = 0.0
            blade_record_key[blade] = wp.int64(0)
            contact_count[blade] = 0
    else:
        due[0] = 0


@wp.kernel
def accumulate_cut_blade_force(
    due: wp.array[wp.int32],
    shape_blade: wp.array[wp.int32],
    blade_body: wp.array[wp.int32],
    blade_direction: wp.array[wp.vec3],
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
    cut_force: wp.array[wp.float32],
    contact_penalty_k: wp.array[float],
    use_log_barrier: bool,
    blade_force: wp.array[float],
    blade_record_key: wp.array[wp.int64],
    contact_count: wp.array[wp.int32],
):
    """One thread per rigid-soft contact record: sum the normal force of the blade edge along the cutting direction.

    The force law is the one the solver applies, as in ``accumulate_suture_tip_force``. The strongest
    record is kept through an int64 key of the force in micronewtons and the record index.
    """
    if due[0] == 0:
        return
    record = wp.tid()
    if record >= wp.min(soft_contact_count[0], soft_contact_max):
        return
    shape = soft_contact_shape[record]
    if shape < 0:
        return
    blade = shape_blade[shape]
    if blade < 0:
        return
    corners = soft_contact_indices[record]
    if corners[0] < 0:
        return
    wp.atomic_add(contact_count, blade, 1)  # the stroke goes on
    if not _soft_contact_force_enabled(record, contact_filter):
        return
    cuttable = cut_force[corners[0]] < CUT_FORCE_UNCUTTABLE
    if corners[1] >= 0 and cut_force[corners[1]] < CUT_FORCE_UNCUTTABLE:
        cuttable = True
    if corners[2] >= 0 and cut_force[corners[2]] < CUT_FORCE_UNCUTTABLE:
        cuttable = True
    if not cuttable:
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
    d = wp.normalize(wp.transform_vector(body_q[blade_body[blade]], blade_direction[blade]))
    f = -dE_dD * wp.max(wp.dot(n, d), 0.0)
    if f <= 0.0:
        return
    wp.atomic_add(blade_force, blade, f)
    key = (wp.int64(wp.min(f * 1.0e6, 2.0e9)) << wp.int64(32)) + wp.int64(record) + wp.int64(1)
    wp.atomic_max(blade_record_key, blade, key)


@wp.kernel
def decide_cut_advance(
    due: wp.array[wp.int32],
    armed: int,
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    cut_force: wp.array[wp.float32],
    blade_force: wp.array[float],
    blade_record_key: wp.array[wp.int64],
    advance: wp.array[wp.int32],
):
    """One thread per blade: the cut advances when the blade force reaches the cut force at the strongest record."""
    blade = wp.tid()
    advance[blade] = 0
    if due[0] == 0:
        return
    key = blade_record_key[blade]
    if key <= wp.int64(0):
        return
    record = int((key - wp.int64(1)) & wp.int64(0xFFFFFFFF))
    corners = soft_contact_indices[record]
    particle = int(-1)
    if corners[1] < 0:
        if cut_force[corners[0]] < CUT_FORCE_UNCUTTABLE:
            particle = corners[0]
    else:
        bary = soft_contact_barycentric[record]
        best = float(-1.0)
        for k in range(3):
            c = corners[k]
            if c >= 0 and cut_force[c] < CUT_FORCE_UNCUTTABLE and bary[k] > best:
                best = bary[k]
                particle = c
    if particle < 0:
        return
    if armed != 0 or blade_force[blade] >= cut_force[particle]:
        advance[blade] = 1


# =====================================================================
# Chapter 3: the cut faces
# =====================================================================


@wp.kernel
def mark_pressed_vertices(
    due: wp.array[wp.int32],
    press_wound: int,
    blade_body: wp.array[wp.int32],
    blade_edge_shape: wp.array[wp.int32],
    blade_direction: wp.array[wp.vec3],
    contact_range: wp.array[wp.float32],
    contact_count: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    cut_force: wp.array[wp.float32],
    particle_wound: wp.array[wp.int32],
    particle_pressed: wp.array[wp.int32],
):
    """One thread per particle: a cuttable active particle within the contact range of the blade edge is pressed.

    With ``press_wound`` a wound particle of the current stroke that lies inside the span of the
    blade edge is pressed too, so the cut advances from the cut front (fork: the blade rests on the
    two wound lips and cannot reach the cut front).
    """
    p = wp.tid()
    particle_pressed[p] = 0
    if due[0] == 0:
        return
    if contact_count[0] == 0:
        particle_wound[p] = 0  # the stroke ended
    if (particle_flags[p] & ParticleFlags.ACTIVE) == 0 or cut_force[p] >= CUT_FORCE_UNCUTTABLE:
        return
    a, b, _d = _blade_edge(0, blade_body, blade_edge_shape, blade_direction, body_q, shape_transform, shape_scale)
    x = particle_q[p]
    r = contact_range[0]
    if _segment_distance_sq(x, a, b) <= r * r:
        particle_pressed[p] = 1
    elif press_wound != 0 and particle_wound[p] != 0:
        e = b - a
        length = wp.length(e)
        if length > 0.0:
            s = wp.dot(x - a, e) / length
            if s >= -r and s <= length + r:
                particle_pressed[p] = 1


@wp.kernel
def mark_cut_faces(
    due: wp.array[wp.int32],
    keep_side_labels: int,
    blade_body: wp.array[wp.int32],
    blade_edge_shape: wp.array[wp.int32],
    blade_direction: wp.array[wp.vec3],
    contact_count: wp.array[wp.int32],
    advance: wp.array[wp.int32],
    update_number: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    cut_force: wp.array[wp.float32],
    tet_indices: wp.array2d[wp.int32],
    particle_pressed: wp.array[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    tet_side: wp.array[wp.int32],
    face_side: wp.array2d[wp.int32],
    face_stamp: wp.array2d[wp.int32],
    particle_dirty: wp.array[wp.int32],
    particle_wound: wp.array[wp.int32],
):
    """One thread per (tet, face): cut an uncut face that has a pressed vertex and two tets on different sides."""
    tid = wp.tid()
    tet = tid // 4
    face = tid % 4
    if due[0] == 0:
        return
    if contact_count[0] == 0:
        tet_side[tet] = 0  # the stroke ended: clear the side labels
        return
    other = tet_neighbors[tet, face]
    if other < tet:
        return  # skin, already cut, or the thread of the other tet handles this face
    if advance[0] == 0:
        return
    p0 = tet_indices[tet, TET_FACES[face, 0]]
    p1 = tet_indices[tet, TET_FACES[face, 1]]
    p2 = tet_indices[tet, TET_FACES[face, 2]]
    if particle_pressed[p0] == 0 and particle_pressed[p1] == 0 and particle_pressed[p2] == 0:
        return
    if (
        cut_force[p0] >= CUT_FORCE_UNCUTTABLE
        or cut_force[p1] >= CUT_FORCE_UNCUTTABLE
        or cut_force[p2] >= CUT_FORCE_UNCUTTABLE
    ):
        return
    a, b, d = _blade_edge(0, blade_body, blade_edge_shape, blade_direction, body_q, shape_transform, shape_scale)
    n = wp.cross(d, wp.normalize(b - a))
    side_tet = _side_label(tet, keep_side_labels, a, n, tet_indices, particle_q, tet_side)
    side_other = _side_label(other, keep_side_labels, a, n, tet_indices, particle_q, tet_side)
    tet_side[tet] = side_tet
    tet_side[other] = side_other
    if side_tet == side_other:
        return
    face_other = int(-1)
    for k in range(4):
        if tet_neighbors[other, k] == tet:
            face_other = k
    if face_other < 0:
        return
    stamp = update_number[0]
    tet_neighbors[tet, face] = -(2 + other)
    tet_neighbors[other, face_other] = -(2 + tet)
    face_side[tet, face] = side_tet
    face_side[other, face_other] = side_other
    face_stamp[tet, face] = stamp
    face_stamp[other, face_other] = stamp
    particle_dirty[p0] = 1
    particle_dirty[p1] = 1
    particle_dirty[p2] = 1
    particle_wound[p0] = 1
    particle_wound[p1] = 1
    particle_wound[p2] = 1


# =====================================================================
# Chapter 4: the split and the start gap
# =====================================================================


@wp.kernel
def split_cut_vertices(
    due: wp.array[wp.int32],
    idle_tet: int,
    color_count: int,
    spare_count: int,
    full_copy_mass: int,
    gate_bit: int,
    spare_start: wp.array[wp.int32],
    particle_colors: wp.array[wp.int32],
    particle_attached: wp.array[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    face_side: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    adjacency: MeshAdjacencyData,
    record_group: wp.array[wp.int32],
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
    particle_wound: wp.array[wp.int32],
    pool_taken: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
    particle_dirty: wp.array[wp.int32],
    particle_new: wp.array[wp.int32],
):
    """One thread per particle of a new cut face: one copy for each tet group but one.

    The tet groups of a vertex are the sets of its tets that are joined through uncut faces. The
    thread writes only the tet corners that hold its own vertex, the tet records of its own vertex
    and the data of the spares it took. Positions move in a later launch.
    """
    v = wp.tid()
    if due[0] == 0 or particle_dirty[v] == 0:
        return
    particle_dirty[v] = 0
    if (particle_flags[v] & ParticleFlags.ACTIVE) == 0 or particle_mass[v] == 0.0 or particle_attached[v] != 0:
        return  # a pinned or attached vertex stays shared

    offset = adjacency.v_adj_tets_offsets[v]
    count = (adjacency.v_adj_tets_offsets[v + 1] - offset) >> 1
    base = offset >> 1

    # Group search: every tet record starts as its own group; a sweep joins records across uncut faces.
    for k in range(count):
        if adjacency.v_adj_tets[offset + 2 * k] == idle_tet:
            record_group[base + k] = -1
        else:
            record_group[base + k] = k
    # A group label can travel at most count - 1 records, so count sweeps always converge.
    for _sweep in range(count):
        for k in range(count):
            if record_group[base + k] < 0:
                continue
            tet = adjacency.v_adj_tets[offset + 2 * k]
            corner = adjacency.v_adj_tets[offset + 2 * k + 1]
            for f in range(4):
                if f == corner:
                    continue
                other = tet_neighbors[tet, f]
                if other < 0:
                    continue
                for j in range(count):
                    if adjacency.v_adj_tets[offset + 2 * j] == other and record_group[base + j] >= 0:
                        lowest = wp.min(record_group[base + k], record_group[base + j])
                        record_group[base + k] = lowest
                        record_group[base + j] = lowest

    # The group that keeps the original: most side A cut faces at v; the lowest tet index decides a tie.
    group_count = int(0)
    keep = int(-1)
    keep_faces = int(-1)
    keep_tet = int(0)
    total_volume = float(0.0)
    for r in range(count):
        if record_group[base + r] < 0:
            continue
        total_volume += _tet_rest_volume(adjacency.v_adj_tets[offset + 2 * r], tet_poses)
        if record_group[base + r] != r:
            continue
        group_count += 1
        side_a_faces = int(0)
        lowest_tet = int(2147483647)
        for k in range(count):
            if record_group[base + k] != r:
                continue
            tet = adjacency.v_adj_tets[offset + 2 * k]
            corner = adjacency.v_adj_tets[offset + 2 * k + 1]
            lowest_tet = wp.min(lowest_tet, tet)
            for f in range(4):
                if f != corner and tet_neighbors[tet, f] <= -2 and face_side[tet, f] == -1:
                    side_a_faces += 1
        if side_a_faces > keep_faces or (side_a_faces == keep_faces and lowest_tet < keep_tet):
            keep = r
            keep_faces = side_a_faces
            keep_tet = lowest_tet
    if group_count < 2 or total_volume <= 0.0:
        return  # cut front

    color = particle_colors[v]
    pool = color
    if color >= color_count:
        pool = color - color_count  # a copy takes from its own pool
    mass = particle_mass[v]
    moved_volume = float(0.0)
    for r in range(count):
        if record_group[base + r] != r or r == keep:
            continue
        slot = wp.atomic_add(pool_taken, pool, 1)
        if slot >= spare_count:
            wp.atomic_add(overflow, pool, 1)
            continue  # the group stays on v
        s = spare_start[pool] + slot
        s_offset = adjacency.v_adj_tets_offsets[s]
        s_count = int(0)
        group_volume = float(0.0)
        for k in range(count):
            if record_group[base + k] != r:
                continue
            tet = adjacency.v_adj_tets[offset + 2 * k]
            corner = adjacency.v_adj_tets[offset + 2 * k + 1]
            tet_indices[tet, corner] = s
            adjacency.v_adj_tets[s_offset + 2 * s_count] = tet
            adjacency.v_adj_tets[s_offset + 2 * s_count + 1] = corner
            s_count += 1
            adjacency.v_adj_tets[offset + 2 * k] = idle_tet
            adjacency.v_adj_tets[offset + 2 * k + 1] = 0
            record_group[base + k] = -1
            group_volume += _tet_rest_volume(tet, tet_poses)
        copy_mass = mass
        if full_copy_mass == 0:
            copy_mass = mass * group_volume / total_volume
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
        particle_gate[s] = particle_gate[v] | gate_bit
        particle_wound[s] = 1
        particle_new[s] = 1
        moved_volume += group_volume
    if moved_volume > 0.0:
        if full_copy_mass == 0:
            kept_mass = mass * (total_volume - moved_volume) / total_volume
            particle_mass[v] = kept_mass
            particle_inv_mass[v] = 1.0 / kept_mass
        particle_gate[v] = particle_gate[v] | gate_bit
        particle_new[v] = 1


@wp.kernel
def compute_start_gap_offset(
    due: wp.array[wp.int32],
    start_gap: float,
    use_normal_sum: int,
    idle_tet: int,
    adjacency: MeshAdjacencyData,
    tet_indices: wp.array2d[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_new: wp.array[wp.int32],
    gap_offset: wp.array[wp.vec3],
    unsafe: wp.array[wp.int32],
):
    """One thread per new particle: half a start gap into its own material, away from every cut face at it.

    The opening of a direction u at a cut face is ``-dot(u, m)`` with m the outward unit normal of
    the tet face. The default direction makes the smallest opening over the cut faces of the
    particle as large as possible; its candidates are the directions with equal opening at one, two
    or three of the faces.
    """
    p = wp.tid()
    if due[0] == 0 or particle_new[p] == 0:
        return
    gap_offset[p] = wp.vec3(0.0)
    offset = adjacency.v_adj_tets_offsets[p]
    count = (adjacency.v_adj_tets_offsets[p + 1] - offset) >> 1

    inward = _mat_normals()  # distinct inward unit normals (-m) of the cut faces at p
    n_count = int(0)
    dropped = int(0)
    normal_sum = wp.vec3(0.0)
    for k in range(count):
        tet = adjacency.v_adj_tets[offset + 2 * k]
        if tet == idle_tet:
            continue
        corner = adjacency.v_adj_tets[offset + 2 * k + 1]
        for f in range(4):
            if f == corner:
                continue
            neighbor = tet_neighbors[tet, f]
            if neighbor > -2:
                continue
            if _tet_holds(-(neighbor + 2), p, tet_indices):
                continue  # this face is not separated at p
            x0 = particle_q[tet_indices[tet, TET_FACES[f, 0]]]
            x1 = particle_q[tet_indices[tet, TET_FACES[f, 1]]]
            x2 = particle_q[tet_indices[tet, TET_FACES[f, 2]]]
            area_normal = wp.cross(x1 - x0, x2 - x0)
            area2 = wp.length(area_normal)
            if area2 <= 1.0e-18:
                continue
            normal_sum += area_normal
            a_in = -area_normal / area2
            known = bool(False)
            for i in range(n_count):
                if wp.dot(a_in, wp.vec3(inward[i, 0], inward[i, 1], inward[i, 2])) > 1.0 - 1.0e-5:
                    known = True
            if not known:
                if n_count < _MAX_GAP_NORMALS:
                    inward[n_count, 0] = a_in[0]
                    inward[n_count, 1] = a_in[1]
                    inward[n_count, 2] = a_in[2]
                    n_count += 1
                else:
                    dropped = 1  # more distinct normals than the direction search holds

    if dropped != 0:
        wp.atomic_add(unsafe, 0, 1)

    if n_count == 0:
        wp.atomic_add(unsafe, 0, 1)
        return

    best_u = wp.vec3(0.0)
    best_open = float(-2.0)
    if use_normal_sum != 0:
        length = wp.length(normal_sum)
        if length > 0.0:
            best_u = -normal_sum / length
            best_open = float(2.0)
            for i in range(n_count):
                best_open = wp.min(best_open, wp.dot(best_u, wp.vec3(inward[i, 0], inward[i, 1], inward[i, 2])))
    else:
        for i in range(n_count):
            ai = wp.vec3(inward[i, 0], inward[i, 1], inward[i, 2])
            for j in range(i, n_count):
                aj = wp.vec3(inward[j, 0], inward[j, 1], inward[j, 2])
                for l in range(j, n_count):
                    al = wp.vec3(inward[l, 0], inward[l, 1], inward[l, 2])
                    u = wp.vec3(0.0)
                    valid = bool(False)
                    if i == j and j == l:
                        u = ai
                        valid = True
                    elif j == l:
                        s = ai + aj
                        if wp.length(s) > 1.0e-6:
                            u = wp.normalize(s)
                            valid = True
                    elif i != j:
                        m = wp.matrix_from_rows(ai, aj, al)
                        det = wp.determinant(m)
                        if wp.abs(det) > 1.0e-8:
                            w = wp.inverse(m) * wp.vec3(1.0, 1.0, 1.0)
                            if wp.length(w) > 1.0e-12:
                                u = wp.normalize(w)
                                valid = True
                    if valid:
                        least = float(2.0)
                        for q in range(n_count):
                            least = wp.min(least, wp.dot(u, wp.vec3(inward[q, 0], inward[q, 1], inward[q, 2])))
                        if least > best_open:
                            best_open = least
                            best_u = u

    if best_open <= 1.0e-6:
        wp.atomic_add(unsafe, 0, 1)  # no direction opens every cut face: p starts at distance zero on some faces
        return
    gap_offset[p] = 0.5 * start_gap * best_u


@wp.kernel
def clear_cut_gate_bit(gate_bit: int, particle_gate: wp.array[wp.int32]):
    """One thread per particle: remove the gate bit of the cut and keep the needle bits."""
    p = wp.tid()
    particle_gate[p] = particle_gate[p] & (~gate_bit)


@wp.kernel
def apply_start_gap_offset(
    due: wp.array[wp.int32],
    particle_new: wp.array[wp.int32],
    gap_offset: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
):
    """One thread per particle: move every new particle by its start gap offset."""
    p = wp.tid()
    if due[0] == 0 or particle_new[p] == 0:
        return
    particle_q[p] = particle_q[p] + gap_offset[p]
    particle_new[p] = 0


# =====================================================================
# Chapter 5: the surface rows
# =====================================================================


@wp.func
def _walk_fan(
    tet: int,
    face: int,
    a: int,
    b: int,
    max_steps: int,
    tet_indices: wp.array2d[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    tet_edge_row: wp.array2d[wp.int32],
):
    """Turn around the tet edge (a, b), away from ``face``, through uncut faces.

    Returns the last tet, the face where the fan ends, 1 if a tet of the fan is the source of an
    edge row, and 1 if the fan ended at a cut face or a skin face within ``max_steps`` tets.
    """
    pa = tet_indices[tet, a]
    pb = tet_indices[tet, b]
    has_row = int(0)
    if tet_edge_row[tet, _edge_index(a, b)] >= 0:
        has_row = 1
    for _step in range(max_steps):
        out_face = 6 - a - b - face
        other = tet_neighbors[tet, out_face]
        if other < 0:
            return tet, out_face, has_row, 1
        in_face = int(-1)
        na = int(-1)
        nb = int(-1)
        for k in range(4):
            if tet_neighbors[other, k] == tet:
                in_face = k
            q = tet_indices[other, k]
            if q == pa:
                na = k
            if q == pb:
                nb = k
        if in_face < 0 or na < 0 or nb < 0:
            return tet, out_face, has_row, 0
        tet = other
        face = in_face
        a = na
        b = nb
        if tet_edge_row[tet, _edge_index(a, b)] >= 0:
            has_row = 1
    return tet, face, has_row, 0


@wp.kernel
def take_surface_rows(
    due: wp.array[wp.int32],
    surface_edges: int,
    max_fan_tets: int,
    overflow_tri_slot: int,
    tri_spare_start: int,
    tri_spare_count: int,
    edge_spare_start: int,
    edge_spare_count: int,
    update_number: wp.array[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    face_stamp: wp.array2d[wp.int32],
    tri_taken: wp.array[wp.int32],
    edge_taken: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
    tri_source: wp.array2d[wp.int32],
    edge_source: wp.array2d[wp.int32],
    tet_face_row: wp.array2d[wp.int32],
    tet_edge_row: wp.array2d[wp.int32],
):
    """One thread per (tet, face): each side of a new cut face takes a triangle row, each new fan an edge row."""
    tid = wp.tid()
    tet = tid // 4
    face = tid % 4
    if due[0] == 0:
        return
    stamp = update_number[0]
    if face_stamp[tet, face] != stamp or tet_face_row[tet, face] >= 0:
        return
    c0 = TET_FACES[face, 0]
    c1 = TET_FACES[face, 1]
    c2 = TET_FACES[face, 2]
    slot = wp.atomic_add(tri_taken, 0, 1)
    if slot >= tri_spare_count:
        wp.atomic_add(overflow, overflow_tri_slot, 1)
    else:
        row = tri_spare_start + slot
        tri_source[row, 0] = tet
        tri_source[row, 1] = c0
        tri_source[row, 2] = c1
        tri_source[row, 3] = c2
        tet_face_row[tet, face] = row
    if surface_edges == 0:
        return
    for e in range(3):
        a = c0
        b = c1
        if e == 1:
            a = c1
            b = c2
        elif e == 2:
            a = c2
            b = c0
        end_tet, end_face, has_row, ended = _walk_fan(
            tet, face, a, b, max_fan_tets, tet_indices, tet_neighbors, tet_edge_row
        )
        if ended == 0:
            wp.atomic_add(overflow, overflow_tri_slot + 1, 1)
            continue
        if has_row != 0:
            continue  # the fan has its edge row
        if face_stamp[end_tet, end_face] == stamp and end_tet * 4 + end_face < tet * 4 + face:
            continue  # the thread of the other end takes the row
        edge_slot = wp.atomic_add(edge_taken, 0, 1)
        if edge_slot >= edge_spare_count:
            wp.atomic_add(overflow, overflow_tri_slot + 1, 1)
            continue
        edge_row = edge_spare_start + edge_slot
        edge_source[edge_row, 0] = tet
        edge_source[edge_row, 1] = a
        edge_source[edge_row, 2] = b
        tet_edge_row[tet, _edge_index(a, b)] = edge_row


@wp.kernel
def refresh_triangle_rows(
    due: wp.array[wp.int32],
    tri_spare_start: int,
    row_park_particle: wp.array[wp.int32],
    tri_source: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    tet_neighbors: wp.array2d[wp.int32],
    tet_face_row: wp.array2d[wp.int32],
    tri_indices: wp.array2d[wp.int32],
):
    """One thread per triangle row: rewrite a row that has a source from the tet corners."""
    row = wp.tid()
    if due[0] == 0:
        return
    tet = tri_source[row, 0]
    if tet < 0:
        return
    c0 = tri_source[row, 1]
    c1 = tri_source[row, 2]
    c2 = tri_source[row, 3]
    i0 = tet_indices[tet, c0]
    i1 = tet_indices[tet, c1]
    i2 = tet_indices[tet, c2]
    neighbor = tet_neighbors[tet, 6 - c0 - c1 - c2]
    if neighbor <= -2 and row >= tri_spare_start:
        other = -(neighbor + 2)
        if (
            other < tet
            and _tet_holds(other, i0, tet_indices)
            and _tet_holds(other, i1, tet_indices)
            and _tet_holds(other, i2, tet_indices)
        ):
            other_has_row = bool(False)
            for k in range(4):
                if tet_neighbors[other, k] == -(2 + tet) and tet_face_row[other, k] >= 0:
                    other_has_row = True
            if other_has_row:
                # No particle of this cut face is split yet: the row of the other side is enough.
                park = row_park_particle[row - tri_spare_start]
                i0 = park
                i1 = park
                i2 = park
    tri_indices[row, 0] = i0
    tri_indices[row, 1] = i1
    tri_indices[row, 2] = i2


@wp.kernel
def refresh_edge_rows(
    due: wp.array[wp.int32],
    edge_source: wp.array2d[wp.int32],
    tet_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
):
    """One thread per edge row: rewrite the two edge particles of a row that has a source from the tet corners."""
    row = wp.tid()
    if due[0] == 0:
        return
    tet = edge_source[row, 0]
    if tet < 0:
        return
    edge_indices[row, 2] = tet_indices[tet, edge_source[row, 1]]
    edge_indices[row, 3] = tet_indices[tet, edge_source[row, 2]]
