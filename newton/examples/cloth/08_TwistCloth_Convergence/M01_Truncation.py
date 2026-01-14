from Demos.mmapfile_demo import offset

import warp as wp
import numpy as np
import trimesh

from newton import ModelBuilder
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.vbd.solver_vbd import *
from newton._src.solvers.vbd.tri_mesh_collision import *
import glob
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tqdm

@wp.func
def point_plane_dis(
    v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    signed:bool=False,
):
    # Signed distance: positive if v is in the direction of n from the plane,
    # negative if opposite. If n is zero-length, return 0 to avoid NaNs.
    n_len = wp.length(n)
    if n_len > 0.0:
        dis = wp.dot(n, v - d) / n_len
        return dis if signed else wp.abs(dis)
    else:
        return 0.0

@wp.func
def segment_plane_intersects(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps_parallel: float,                 # e.g., 1e-8
    eps_intersect_near: float,                 # e.g., 1e-8
    eps_intersect_far: float,                 # e.g., 1e-8
    coplanar_counts: bool       # True if you want a coplanar segment to count as "hit"
) -> bool:
    # Plane eq: n·(p - d) = 0
    # Segment: p(t) = v + t * delta_v,  t in [0, 1]
    nv  = wp.dot(n, delta_v)
    num = -wp.dot(n, v - d)

    # Parallel (or nearly): either coplanar or no hit
    if wp.abs(nv) < eps_parallel:
        return coplanar_counts and (wp.abs(num) < eps_parallel)

    t = num / nv
    # consider tiny tolerance at ends
    return (t >= eps_intersect_near) and (t <= 1.0 + eps_intersect_far)

def segment_plane_intersects_debug(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps_parallel: float,                 # e.g., 1e-8
    eps_intersect_near: float,                 # e.g., 1e-8
    eps_intersect_far: float,                 # e.g., 1e-8
    coplanar_counts: bool       # True if you want a coplanar segment to count as "hit"
) -> bool:
    # Plane eq: n·(p - d) = 0
    # Segment: p(t) = v + t * delta_v,  t in [0, 1]
    nv  = wp.dot(n, delta_v)
    num = -wp.dot(n, v - d)

    # Parallel (or nearly): either coplanar or no hit
    if wp.abs(nv) < eps_parallel:
        intersect =  coplanar_counts and (wp.abs(num) < eps_parallel)
        if intersect:
            print("Parallel Branch: (wp.abs(num) < eps_parallel)", (wp.abs(num) < eps_parallel), " vs ", eps_parallel)
        return intersect

    t = num / nv
    # consider tiny tolerance at ends
    intersect = (t >= eps_intersect_near) and (t <= 1.0 + eps_intersect_far)
    print("Non Parallel Branch: t", t, " eps_near ", eps_intersect_near, " | eps_far:", eps_intersect_far)
    return (t >= eps_intersect_near) and (t <= 1.0 + eps_intersect_far)


@wp.func
def create_vertex_triangle_division_plane_closest_pt(
        v: wp.vec3,
        delta_v: wp.vec3,
        t1: wp.vec3,
        delta_t1: wp.vec3,
        t2: wp.vec3,
        delta_t2: wp.vec3,
        t3: wp.vec3,
        delta_t3: wp.vec3,
):
    """
    n points to the vertex side
    Args:
        v:
        delta_v:
        t1:
        delta_t1:
        t2:
        delta_t2:
        t3:
        delta_t3:

    Returns:


    """
    closest_p, bary, feature_type = triangle_closest_point(t1, t2, t3, v)

    n_hat = v - closest_p

    if wp.length(n_hat) < 1e-12:
        return wp.vector(False, False, False, False, length=4, dtype=wp.bool), wp.vec3(0.), v

    n = wp.normalize(n_hat)

    delta_v_n = wp.max(-wp.dot(n, delta_v), 0.)
    delta_t_n = wp.max(
        wp.vec4(
            wp.dot(n, delta_t1),
            wp.dot(n, delta_t2),
            wp.dot(n, delta_t3),
            0.,
        )
    )

    if delta_t_n + delta_v_n == 0.:
        d = closest_p + 0.5 * n_hat
    else:
        lmbd = delta_t_n / (delta_t_n + delta_v_n)
        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        # wp.printf("lambda: %f\n", lmbd)
        d = closest_p + lmbd * n_hat

    if delta_v_n == 0.:
        is_dummy_for_v = True
    else:
        is_dummy_for_v = not segment_plane_intersects(v, delta_v, n, d, 1e-6, -1e-8, 1e-8,False)

    if delta_t_n == 0.:
        is_dummy_for_t_1 = True
        is_dummy_for_t_2 = True
        is_dummy_for_t_3 = True
    else:
        is_dummy_for_t_1 = not segment_plane_intersects(t1, delta_t1, n, d, 1e-6, -1e-8, 1e-8,False)
        is_dummy_for_t_2 = not segment_plane_intersects(t2, delta_t2, n, d, 1e-6, -1e-8, 1e-8,False)
        is_dummy_for_t_3 = not segment_plane_intersects(t3, delta_t3, n, d, 1e-6, -1e-8, 1e-8,False)

    return wp.vector(is_dummy_for_v, is_dummy_for_t_1, is_dummy_for_t_2, is_dummy_for_t_3, length=4, dtype=wp.bool), n, d

@wp.func
def robust_edge_pair_normal(
    e0_v0_pos: wp.vec3,
    e0_v1_pos: wp.vec3,
    e1_v0_pos: wp.vec3,
    e1_v1_pos: wp.vec3,
    eps: float = 1.0e-6,
) -> wp.vec3:
    # Edge directions
    dir0 = e0_v1_pos - e0_v0_pos
    dir1 = e1_v1_pos - e1_v0_pos

    len0 = wp.length(dir0)
    len1 = wp.length(dir1)

    if len0 > eps:
        dir0 = dir0 / len0
    else:
        dir0 = wp.vec3(0.0, 0.0, 0.0)

    if len1 > eps:
        dir1 = dir1 / len1
    else:
        dir1 = wp.vec3(0.0, 0.0, 0.0)

    # Primary: cross of two valid directions
    n = wp.cross(dir0, dir1)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    # Parallel or degenerate: pick best non-zero direction
    reference = dir0
    if wp.length(reference) <= eps:
        reference = dir1

    if wp.length(reference) <= eps:
        # Both edges collapsed: fall back to canonical axis
        return wp.vec3(1.0, 0.0, 0.0)

    # Try bridge vector between midpoints
    bridge = 0.5 * ((e1_v0_pos + e1_v1_pos) - (e0_v0_pos + e0_v1_pos))
    bridge_len = wp.length(bridge)
    if bridge_len > eps:
        n = wp.cross(reference, bridge / bridge_len)
        len_n = wp.length(n)
        if len_n > eps:
            return n / len_n

    # Use an axis guaranteed (numerically) to be non-parallel
    fallback_axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(wp.dot(reference, fallback_axis)) > 0.9:
        fallback_axis = wp.vec3(0.0, 1.0, 0.0)

    n = wp.cross(reference, fallback_axis)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    # Final guard: use the remaining canonical axis
    fallback_axis = wp.vec3(0.0, 0.0, 1.0)
    n = wp.cross(reference, fallback_axis)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    return wp.vec3(1.0, 0.0, 0.0)

@wp.func
def create_edge_edge_division_plane_closest_pt(
        e0_v0_pos: wp.vec3,
        delta_e0_v0: wp.vec3,
        e0_v1_pos: wp.vec3,
        delta_e0_v1: wp.vec3,
        e1_v0_pos: wp.vec3,
        delta_e1_v0: wp.vec3,
        e1_v1_pos: wp.vec3,
        delta_e1_v1: wp.vec3,
):
    st = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, 1e-6)
    s = st[0]
    t = st[1]
    c1 = e0_v0_pos + (e0_v1_pos - e0_v0_pos) * s
    c2 = e1_v0_pos + (e1_v1_pos - e1_v0_pos) * t

    n_hat = c1 - c2

    if wp.length(n_hat) < 1e-12:
        return (wp.vector(False, False, False, False, length=4, dtype=wp.bool),
                robust_edge_pair_normal(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos),
                c1*0.5 + c2*0.5)

    n = wp.normalize(n_hat)

    delta_e0 = wp.max(
        wp.vec3(
            -wp.dot(n, delta_e0_v0),
            -wp.dot(n, delta_e0_v1),
            0.,
        )
    )
    delta_e1 = wp.max(
        wp.vec3(
            wp.dot(n, delta_e1_v0),
            wp.dot(n, delta_e1_v1),
            0.,
        )
    )

    if delta_e0 + delta_e1 == 0.:
        d = c2 + 0.5 * n_hat
    else:

        lmbd = delta_e1 / (delta_e1 + delta_e0)

        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        # wp.printf("lambda: %f\n", lmbd)
        d = c2 + lmbd * n_hat

    if delta_e0 == 0.:
        is_dummy_for_e0_v0 = True
        is_dummy_for_e0_v1 = True
    else:
        is_dummy_for_e0_v0 = not segment_plane_intersects(e0_v0_pos, delta_e0_v0, n, d, 1e-6, -1e-8,1e-6, False)
        is_dummy_for_e0_v1 = not segment_plane_intersects(e0_v1_pos, delta_e0_v1, n, d, 1e-6, -1e-8,1e-6, False)

    if delta_e1 == 0.:
        is_dummy_for_e1_v0 = True
        is_dummy_for_e1_v1 = True
    else:
        is_dummy_for_e1_v0 = not segment_plane_intersects(e1_v0_pos, delta_e1_v0, n, d, 1e-6, -1e-8, 1e-6,False)
        is_dummy_for_e1_v1 = not segment_plane_intersects(e1_v1_pos, delta_e1_v1, n, d, 1e-6, -1e-8, 1e-6,False)

    return wp.vector(is_dummy_for_e0_v0, is_dummy_for_e0_v1, is_dummy_for_e1_v0, is_dummy_for_e1_v1, length=4, dtype=wp.bool), n, d

def create_vertex_triangle_division_plane_closest_pt_np(
        v,
        delta_v,
        t1,
        delta_t1,
        t2,
        delta_t2,
        t3,
        delta_t3,
):
    closest_p, bary, feature_type = triangle_closest_point(t1, t2, t3, v)

    n_hat = v - closest_p

    if wp.length(n_hat) < 1e-12:
        return wp.vec4(1,1,1,1), wp.vec3(1.0, 0., 0.), v

    n = wp.normalize(n_hat)

    delta_v_n = wp.max(-wp.dot(n, delta_v), 0.)
    delta_t_n = np.max(
            [wp.dot(n, delta_t1),
            wp.dot(n, delta_t2),
            wp.dot(n, delta_t3),
            0.,]
    )

    if delta_t_n + delta_v_n == 0.:
        d = closest_p + 0.5 * n_hat
    else:
        lmbd = delta_t_n / (delta_t_n + delta_v_n)
        d = closest_p + wp.float32(lmbd) * n_hat

    if delta_v_n == 0.:
        is_dummy_for_v = True
    else:
        is_dummy_for_v = not segment_plane_intersects(v, delta_v, n, d, 1e-6, 1e-8, 1e-8, False)

    if delta_t_n == 0.:
        is_dummy_for_t_1 = True
        is_dummy_for_t_2 = True
        is_dummy_for_t_3 = True
    else:
        is_dummy_for_t_1 = not segment_plane_intersects(t1, delta_t1, n, d, 1e-6, 1e-8, 1e-8, False)
        is_dummy_for_t_2 = not segment_plane_intersects(t2, delta_t2, n, d, 1e-6, 1e-8, 1e-8,False)
        is_dummy_for_t_3 = not segment_plane_intersects(t3, delta_t3, n, d, 1e-6, 1e-8, 1e-8,False)

    return wp.vec4(is_dummy_for_v, is_dummy_for_t_1, is_dummy_for_t_2, is_dummy_for_t_3), n, d


@wp.func
def planar_truncation(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps: float,
    gamma_r: float,
    gamma_min: float=1e-3
):
    nv = wp.dot(n, delta_v)
    num = -wp.dot(n, v - d)

    # Parallel (or nearly): do not truncate
    if wp.abs(nv) < eps:
        return delta_v

    t = num / nv

    t = wp.max(wp.min(t * gamma_r, t - gamma_min), 0.)
    if t >= 1:
        return delta_v
    else:
        return t * delta_v

@wp.func
def vector_planar_projection(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps: float,
    gamma_r: float,
    gamma_min: float=1e-3
):
    nv = wp.dot(n, delta_v)
    num = -wp.dot(n, v - d)

    # Parallel (or nearly): do not truncate
    if wp.abs(nv) < eps:
        return delta_v

    t = num / nv
    t = wp.max(wp.min(t * gamma_r, t - gamma_min), 0.)

    if t >= 1:
        return delta_v
    else:
        return delta_v + (t - 1) * n * nv

@wp.func
def vector_planar_projection_dykstra(
    p: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
):
    """
    This function projects the p exactly to the plane and returns the new correction t_new
    Relaxiation is followed by a second truncation stage
    """
    # Half-space: (x - d) · n >= 0 is valid. n points to the valid side.
    # d is a point on the plane.
    #
    # We do *exact* orthogonal projection:
    # - if already inside (s >= 0), do nothing
    # - if outside (s < 0), move along +n just enough to reach the plane

    # signed coordinate along n from the plane
    s = wp.dot(p - d, n)

    # already in the valid half-space -> no change
    if s >= 0.0:
        return p, s

    # allow non-unit normals: projection distance along n is -s / ||n||^2
    nn = wp.dot(n, n)
    # guard a bit against degenerate normals
    if nn > 0.0:
        alpha = -s / nn        # move along +n
        p_proj = p + alpha * n
    else:
        # degenerate normal: can't project, just return original
        p_proj = p

    # recompute signed coordinate (should be ~0 if nn>0)
    t_new = wp.dot(p_proj - d, n)

    return p_proj, t_new

@wp.func()
def run_penetration_free_truncation(
        particle_index:int,
        particle_pos: wp.vec3,
        particle_displacement: wp.vec3,
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        parallel_eps: float,
        gamma: float,
        max_displacement:float,
):

    for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_index)):
        colliding_tri_index = get_vertex_colliding_triangles(collision_info, particle_index, i_v_collision)

        t1 = pos[tri_indices[colliding_tri_index, 0]]
        t2 = pos[tri_indices[colliding_tri_index, 1]]
        t3 = pos[tri_indices[colliding_tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[colliding_tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[colliding_tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[colliding_tri_index, 2]]

        is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
            particle_pos,
            particle_displacement,
            t1,
            delta_t1,
            t2,
            delta_t2,
            t3,
            delta_t3,
        )

        if not is_dummy[0]:
            particle_displacement = planar_truncation(particle_pos, particle_displacement, n, d, parallel_eps, gamma)

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        t1 = pos[tri_indices[tri_index, 0]]
        t2 = pos[tri_indices[tri_index, 1]]
        t3 = pos[tri_indices[tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[tri_index, 2]]

        for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
            colliding_v = get_triangle_colliding_vertices(collision_info, tri_index, i_t_collision)

            colliding_particle_pos = pos[colliding_v]
            colliding_particle_displacement = displacement_in[colliding_v]

            is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                colliding_particle_pos,
                colliding_particle_displacement,
                t1,
                delta_t1,
                t2,
                delta_t2,
                t3,
                delta_t3,
            )
            if not is_dummy[vertex_order + 1]:
                particle_displacement = planar_truncation(particle_pos, particle_displacement, n, d, parallel_eps,
                                                          gamma)
    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index,
                                                                                 i_adj_edge)

        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):
                colliding_e = get_edge_colliding_edges(collision_info, nei_edge_index, i_e_collision)

                e1_v1 = edge_indices[nei_edge_index, 2]
                e1_v2 = edge_indices[nei_edge_index, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[colliding_e, 2]
                e2_v2 = edge_indices[colliding_e, 3]

                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )

                if not is_dummy[vertex_order_on_edge - 2]:
                    particle_displacement = planar_truncation(particle_pos, particle_displacement, n, d, parallel_eps,
                                                              gamma)

    len_displacement = wp.length(particle_displacement)
    if len_displacement > max_displacement:
        particle_displacement = particle_displacement * max_displacement / len_displacement

    return particle_displacement

@wp.kernel
def penetration_free_truncation(
        # inputs
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        parallel_eps: float,
        gamma: float,
        max_displacement:float,
        # outputs
        displacement_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_index = tid
    particle_pos = pos[particle_index]
    particle_displacement = displacement_in[particle_index]

    particle_displacement = run_penetration_free_truncation(
        particle_index,
        particle_pos,
        particle_displacement,
        pos,
        displacement_in,
        tri_indices,
        edge_indices,
        adjacency,
        collision_info,
        parallel_eps,
        gamma,
        max_displacement,
    )

    displacement_out[particle_index] = particle_displacement

@wp.func
def initialize_projection(
        particle_index: int,
        particle_pos: wp.vec3,
        particle_displacement: wp.vec3,
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        division_plane_nds_vt: wp.array(dtype=wp.vec3),             # shape = vertex_colliding_triangles.shape
        projection_t_vt: wp.array(dtype=float),                     # shape = vertex_colliding_triangles.shape // 2
        division_plane_is_dummy_vt: wp.array(dtype=wp.bool),        # shape = vertex_colliding_triangles.shape // 2
        division_plane_nds_ee: wp.array(dtype=wp.vec3),             # shape = edge_colliding_edges.shape
        projection_t_ee: wp.array(dtype=float),                     # shape = edge_colliding_edges.shape, one for per vertex on the edge
        division_plane_is_dummy_ee: wp.array(dtype=wp.bool),        # shape = edge_colliding_edges.shape, one for per vertex on the edge
        division_plane_nds_tv: wp.array(dtype=wp.vec3),             # self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2
        projection_t_tv: wp.array(dtype=float),                     # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        division_plane_is_dummy_tv: wp.array(dtype=wp.bool),        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
):
    all_satisfied = wp.bool(True)
    for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_index)):
        colliding_tri_index = get_vertex_colliding_triangles(collision_info, particle_index, i_v_collision)

        t1 = pos[tri_indices[colliding_tri_index, 0]]
        t2 = pos[tri_indices[colliding_tri_index, 1]]
        t3 = pos[tri_indices[colliding_tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[colliding_tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[colliding_tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[colliding_tri_index, 2]]

        # n points to the vertex side
        is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
            particle_pos,
            particle_displacement,
            t1,
            delta_t1,
            t2,
            delta_t2,
            t3,
            delta_t3,
        )

        offset_v_t = collision_info.vertex_colliding_triangles_offsets[particle_index]
        # n points to the valid half of the half space
        division_plane_nds_vt[2 * (offset_v_t + i_v_collision)] = n
        division_plane_nds_vt[2 * (offset_v_t + i_v_collision) + 1] = d
        projection_t_vt[offset_v_t + i_v_collision] = 0.
        division_plane_is_dummy_vt[offset_v_t + i_v_collision] = is_dummy[0]

        wp.expect_eq(point_in_half_space(particle_pos, n, d), True)

        if not is_dummy[0]:
            all_satisfied = False

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        t1 = pos[tri_indices[tri_index, 0]]
        t2 = pos[tri_indices[tri_index, 1]]
        t3 = pos[tri_indices[tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[tri_index, 2]]

        for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
            colliding_v = get_triangle_colliding_vertices(collision_info, tri_index, i_t_collision)

            colliding_particle_pos = pos[colliding_v]
            colliding_particle_displacement = displacement_in[colliding_v]
            
            # n points to the vertex side
            is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                colliding_particle_pos,
                colliding_particle_displacement,
                t1,
                delta_t1,
                t2,
                delta_t2,
                t3,
                delta_t3,
            )

            # division_plane_nds_tv: wp.array(dtype=wp.vec3),             # self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2
            # projection_t_tv: wp.array(dtype=float),                     # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
            # division_plane_is_dummy_tv: wp.array(dtype=wp.bool),        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle

            offset_t_v = collision_info.triangle_colliding_vertices_offsets[tri_index]
            # n points to the valid half of the half space
            division_plane_nds_tv[2 * (offset_t_v + i_t_collision)] = -n
            division_plane_nds_tv[2 * (offset_t_v + i_t_collision) + 1] = d
            projection_t_tv[3 * (offset_t_v + i_t_collision) + vertex_order] = 0.
            division_plane_is_dummy_tv[3 * (offset_t_v + i_t_collision) + vertex_order] = is_dummy[vertex_order + 1]


            if not is_dummy[vertex_order + 1]:
                all_satisfied = False

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index,
                                                                                 i_adj_edge)

        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):
                colliding_e = get_edge_colliding_edges(collision_info, nei_edge_index, i_e_collision)

                e1_v1 = edge_indices[nei_edge_index, 2]
                e1_v2 = edge_indices[nei_edge_index, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[colliding_e, 2]
                e2_v2 = edge_indices[colliding_e, 3]

                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                # n points to the edge 1 side
                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )
                vertex_order_on_edge_2_v1_v2_only = vertex_order_on_edge - 2

                offset_e_e = collision_info.edge_colliding_edges_offsets[nei_edge_index]
                # n points to the valid half of the half space
                division_plane_nds_ee[2 * (offset_e_e + i_e_collision)] = n
                division_plane_nds_ee[2 * (offset_e_e + i_e_collision) + 1] = d
                projection_t_ee[2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only] = 0.
                division_plane_is_dummy_ee[2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only] = is_dummy[vertex_order_on_edge_2_v1_v2_only]

                if not is_dummy[vertex_order_on_edge_2_v1_v2_only]:
                    all_satisfied = False
    return all_satisfied

@wp.kernel
def calculate_vertex_collision_buffer(
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        projection_buffer_sizes: wp.array(dtype=int)
):
    particle_index = wp.tid()
    size_buffer = wp.int32(0)

    size_buffer += collision_info.vertex_colliding_triangles_buffer_sizes[particle_index]

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        size_buffer += collision_info.triangle_colliding_vertices_buffer_sizes[tri_index]

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index,
                                                                                 i_adj_edge)

        size_buffer += collision_info.edge_colliding_edges_buffer_sizes[nei_edge_index]
    projection_buffer_sizes[particle_index] = size_buffer

@wp.func
def initialize_projection_v2(
        particle_index: int,
        particle_pos: wp.vec3,
        particle_displacement: wp.vec3,
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        projection_vertex_offsets: wp.array(dtype=int),
        division_plane_nds: wp.array(dtype=wp.vec3),
        projection_ts: wp.array(dtype=float),
        num_projection_planes: wp.array(dtype=int),
):
    all_satisfied = wp.bool(True)
    num_plane = wp.int32(0)
    offset_v = projection_vertex_offsets[particle_index]
    # dont need to evaluate size
    for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_index)):
        colliding_tri_index = get_vertex_colliding_triangles(collision_info, particle_index, i_v_collision)

        t1 = pos[tri_indices[colliding_tri_index, 0]]
        t2 = pos[tri_indices[colliding_tri_index, 1]]
        t3 = pos[tri_indices[colliding_tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[colliding_tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[colliding_tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[colliding_tri_index, 2]]

        # n points to the vertex side
        is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
            particle_pos,
            particle_displacement,
            t1,
            delta_t1,
            t2,
            delta_t2,
            t3,
            delta_t3,
        )

        # n points to the valid half of the half space
        division_plane_nds[2 * (offset_v + num_plane)] = n
        division_plane_nds[2 * (offset_v + num_plane) + 1] = d
        num_plane += 1

        wp.expect_eq(point_in_half_space(particle_pos, n, d), True)

        if not is_dummy[0]:
            all_satisfied = False

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        t1 = pos[tri_indices[tri_index, 0]]
        t2 = pos[tri_indices[tri_index, 1]]
        t3 = pos[tri_indices[tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[tri_index, 2]]

        for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
            colliding_v = get_triangle_colliding_vertices(collision_info, tri_index, i_t_collision)

            colliding_particle_pos = pos[colliding_v]
            colliding_particle_displacement = displacement_in[colliding_v]

            # n points to the vertex side
            is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                colliding_particle_pos,
                colliding_particle_displacement,
                t1,
                delta_t1,
                t2,
                delta_t2,
                t3,
                delta_t3,
            )

            # n points to the valid half of the half space
            division_plane_nds[2 * (offset_v + num_plane)] = -n
            division_plane_nds[2 * (offset_v + num_plane) + 1] = d
            projection_ts[offset_v + num_plane] = 0.
            num_plane += 1

            wp.expect_eq(point_in_half_space(particle_pos, -n, d), True)

            if not is_dummy[vertex_order + 1]:
                all_satisfied = False

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index,
                                                                                 i_adj_edge)

        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):
                colliding_e = get_edge_colliding_edges(collision_info, nei_edge_index, i_e_collision)

                e1_v1 = edge_indices[nei_edge_index, 2]
                e1_v2 = edge_indices[nei_edge_index, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[colliding_e, 2]
                e2_v2 = edge_indices[colliding_e, 3]

                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                # n points to the edge 1 side
                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )
                vertex_order_on_edge_2_v1_v2_only = vertex_order_on_edge - 2

                # n points to the valid half of the half space
                division_plane_nds[2 * (offset_v + num_plane)] = n
                division_plane_nds[2 * (offset_v + num_plane) + 1] = d
                projection_ts[offset_v + num_plane] = 0.
                num_plane += 1

                wp.expect_eq(point_in_half_space(particle_pos, n, d), True)

                if not is_dummy[vertex_order_on_edge_2_v1_v2_only]:
                    all_satisfied = False

        num_projection_planes[particle_index] = num_plane

    return all_satisfied

@wp.kernel
def initialize_truncation_planes(
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        projection_vertex_offsets: wp.array(dtype=int),
        division_plane_nds: wp.array(dtype=wp.vec3),
        num_projection_planes: wp.array(dtype=int),
):
    particle_index = wp.tid()
    particle_pos = pos[particle_index]
    particle_displacement = displacement_in[particle_index]

    num_plane = wp.int32(0)
    offset_v = projection_vertex_offsets[particle_index]
    # dont need to evaluate size
    for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_index)):
        colliding_tri_index = get_vertex_colliding_triangles(collision_info, particle_index, i_v_collision)

        t1 = pos[tri_indices[colliding_tri_index, 0]]
        t2 = pos[tri_indices[colliding_tri_index, 1]]
        t3 = pos[tri_indices[colliding_tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[colliding_tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[colliding_tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[colliding_tri_index, 2]]

        # n points to the vertex side
        is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
            particle_pos,
            particle_displacement,
            t1,
            delta_t1,
            t2,
            delta_t2,
            t3,
            delta_t3,
        )

        # for truncation don't need to record the dummy plane
        if not is_dummy[0]:
            # n points to the valid half of the half space
            division_plane_nds[2 * (offset_v + num_plane)] = n
            division_plane_nds[2 * (offset_v + num_plane) + 1] = d
            num_plane += 1

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        t1 = pos[tri_indices[tri_index, 0]]
        t2 = pos[tri_indices[tri_index, 1]]
        t3 = pos[tri_indices[tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[tri_index, 2]]

        for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
            colliding_v = get_triangle_colliding_vertices(collision_info, tri_index, i_t_collision)

            colliding_particle_pos = pos[colliding_v]
            colliding_particle_displacement = displacement_in[colliding_v]

            # n points to the vertex side
            is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                colliding_particle_pos,
                colliding_particle_displacement,
                t1,
                delta_t1,
                t2,
                delta_t2,
                t3,
                delta_t3,
            )

            # for truncation don't need to record the dummy plane
            if not is_dummy[vertex_order + 1]:
                # n points to the valid half of the half space
                division_plane_nds[2 * (offset_v + num_plane)] = -n
                division_plane_nds[2 * (offset_v + num_plane) + 1] = d
                num_plane += 1

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index,
                                                                                 i_adj_edge)

        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):
                colliding_e = get_edge_colliding_edges(collision_info, nei_edge_index, i_e_collision)

                e1_v1 = edge_indices[nei_edge_index, 2]
                e1_v2 = edge_indices[nei_edge_index, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[colliding_e, 2]
                e2_v2 = edge_indices[colliding_e, 3]

                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                # n points to the edge 1 side
                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )
                vertex_order_on_edge_2_v1_v2_only = vertex_order_on_edge - 2

                # for truncation don't need to record the dummy plane
                if not is_dummy[vertex_order_on_edge_2_v1_v2_only]:
                    # n points to the valid half of the half space
                    division_plane_nds[2 * (offset_v + num_plane)] = n
                    division_plane_nds[2 * (offset_v + num_plane) + 1] = d
                    num_plane += 1

        num_projection_planes[particle_index] = num_plane


@wp.kernel
def run_penetration_free_truncation_v2(
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        parallel_eps: float,
        gamma: float,
        max_displacement:float,
        projection_vertex_offsets: wp.array(dtype=int),
        division_plane_nds: wp.array(dtype=wp.vec3),
        num_projection_planes: wp.array(dtype=int),
        displacement_out: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()
    particle_pos = pos[particle_index]
    particle_displacement = displacement_in[particle_index]

    offset_v = projection_vertex_offsets[particle_index]
    num_planes = num_projection_planes[particle_index]

    for constraint in range(num_planes):
        n = division_plane_nds[2 * (offset_v + constraint)]
        d = division_plane_nds[2 * (offset_v + constraint) + 1]
        particle_displacement = planar_truncation(particle_pos, particle_displacement, n, d, parallel_eps, gamma)

    len_displacement = wp.length(particle_displacement)
    if len_displacement > max_displacement:
        particle_displacement = particle_displacement * max_displacement / len_displacement
    displacement_out[particle_index] = particle_displacement




@wp.func
def point_in_half_space(
    p: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
):

    return wp.dot(p - d, n) >= -1e-6

@wp.func
def point_half_space_projection_dykstra(
    p: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
):
    # Half-space: (x - d) · n >= 0 valid.

    s = wp.dot(p - d, n)      # signed coord from plane along n
    # Already inside: projection is p itself, residual becomes zero.
    if s >= 0.0:
        # p_proj = p, residual = 0
        return p, 0.0

    # Outside: project onto plane
    # delta along +n to reach the plane:
    alpha = -s          # note: s < 0 -> alpha > 0
    p_proj = p + alpha * n

    # Dykstra residual r_new = p - p_proj = -alpha * n = s*n
    # encode as scalar along n:
    t_new = s           # s < 0 -> t_new < 0

    return p_proj, t_new

@wp.func
def run_dijkstra_projection(
        max_iter:int,
        particle_index: int,
        particle_pos: wp.vec3,
        particle_displacement: wp.vec3,
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        division_plane_nds_vt: wp.array(dtype=wp.vec3),             # shape = vertex_colliding_triangles.shape
        projection_t_vt: wp.array(dtype=float),                     # shape = vertex_colliding_triangles.shape // 2
        division_plane_is_dummy_vt: wp.array(dtype=wp.bool),        # shape = vertex_colliding_triangles.shape // 2
        division_plane_nds_ee: wp.array(dtype=wp.vec3),             # shape = edge_colliding_edges.shape
        projection_t_ee: wp.array(dtype=float),                     # shape = edge_colliding_edges.shape, one for per vertex on the edge
        division_plane_is_dummy_ee: wp.array(dtype=wp.bool),        # shape = edge_colliding_edges.shape, one for per vertex on the edge
        division_plane_nds_tv: wp.array(dtype=wp.vec3),             # self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2
        projection_t_tv: wp.array(dtype=float),                     # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        division_plane_is_dummy_tv: wp.array(dtype=wp.bool),        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
):
    x = particle_pos + particle_displacement

    for iter in range(max_iter):

        for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_index)):
            offset_v_t = collision_info.vertex_colliding_triangles_offsets[particle_index]
            is_dummy = division_plane_is_dummy_vt[offset_v_t + i_v_collision]
            if not is_dummy:
            # if True:
                n = division_plane_nds_vt[2 * (offset_v_t + i_v_collision)]
                d = division_plane_nds_vt[2 * (offset_v_t + i_v_collision) + 1]
                t = projection_t_vt[offset_v_t + i_v_collision]

                y = x + t * n
                y_proj, t_new = point_half_space_projection_dykstra(y, n, d)
                x = y_proj
                projection_t_vt[offset_v_t + i_v_collision] = t_new

                # wp.expect_eq(point_in_half_space(particle_pos, n, d), True)

        for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

            for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
                offset_t_v = collision_info.triangle_colliding_vertices_offsets[tri_index]
                is_dummy = division_plane_is_dummy_tv[3 * (offset_t_v + i_t_collision) + vertex_order]
                #
                if not is_dummy:
                # if True:
                    n = division_plane_nds_tv[2 * (offset_t_v + i_t_collision)]
                    d = division_plane_nds_tv[2 * (offset_t_v + i_t_collision) + 1]
                    t = projection_t_tv[3 * (offset_t_v + i_t_collision) + vertex_order]

                    y = x + t * n
                    y_proj, t_new = point_half_space_projection_dykstra(y, n, d)
                    x = y_proj
                    projection_t_tv[3 * (offset_t_v + i_t_collision) + vertex_order] = t_new

                    # wp.expect_eq(point_in_half_space(particle_pos, n, d), True)

        for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index,
                                                                                     i_adj_edge)

            if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
                vertex_order_on_edge_2_v1_v2_only = vertex_order_on_edge - 2
                for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):

                    offset_e_e = collision_info.edge_colliding_edges_offsets[nei_edge_index]

                    is_dummy = division_plane_is_dummy_ee[2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only]
                    if not is_dummy:
                    # if True:
                        # n points to the valid half of the half space
                        n = division_plane_nds_ee[2 * (offset_e_e + i_e_collision)]
                        d = division_plane_nds_ee[2 * (offset_e_e + i_e_collision) + 1]
                        t = projection_t_ee[2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only]

                        y = x + t * n
                        y_proj, t_new = point_half_space_projection_dykstra(y, n, d)
                        x = y_proj
                        projection_t_ee[2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only] = t_new

                        # wp.expect_eq(point_in_half_space(particle_pos, n, d), True)

    return x

@wp.func
def run_dijkstra_projection_v2(
        max_iter:int,
        particle_index: int,
        particle_pos: wp.vec3,
        particle_displacement: wp.vec3,
        projection_vertex_offsets: wp.array(dtype=int),
        division_plane_nds: wp.array(dtype=wp.vec3),
        projection_ts: wp.array(dtype=float),
        num_projection_planes: wp.array(dtype=int),
        tol: float=1e-5,
):
    x = particle_pos + particle_displacement

    offset_v = projection_vertex_offsets[particle_index]
    num_planes = num_projection_planes[particle_index]
    x_prev = wp.vec3(x)

    for iter in range(max_iter):
        for constraint in range(num_planes):
            n = division_plane_nds[2 * (offset_v + constraint)]
            d = division_plane_nds[2 * (offset_v + constraint) + 1]
            t = projection_ts[offset_v + constraint]

            y = x + t * n
            y_proj, t_new = point_half_space_projection_dykstra(y, n, d)
            x = y_proj

            dx = wp.length_sq(x - x_prev)
            if dx < tol * tol:
                break
            x_prev = x
            projection_ts[offset_v + constraint] = t_new


    return x

@wp.func()
def run_penetration_free_truncation_after_projection(
        particle_index:int,
        particle_pos: wp.vec3,
        particle_displacement: wp.vec3,
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        division_plane_nds_vt: wp.array(dtype=wp.vec3),  # shape = vertex_colliding_triangles.shape
        division_plane_is_dummy_vt: wp.array(dtype=wp.bool),  # shape = vertex_colliding_triangles.shape // 2
        division_plane_nds_ee: wp.array(dtype=wp.vec3),  # shape = edge_colliding_edges.shape
        division_plane_is_dummy_ee: wp.array(dtype=wp.bool),
        # shape = edge_colliding_edges.shape, one for per vertex on the edge
        division_plane_nds_tv: wp.array(dtype=wp.vec3),
        # self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2
        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        division_plane_is_dummy_tv: wp.array(dtype=wp.bool),
        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        parallel_eps: float,
        gamma: float,
        max_displacement:float,
):

    for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_index)):

        offset_v_t = collision_info.vertex_colliding_triangles_offsets[particle_index]
        is_dummy = division_plane_is_dummy_vt[offset_v_t + i_v_collision]

        if not is_dummy:
            n = division_plane_nds_vt[2 * (offset_v_t + i_v_collision)]
            d = division_plane_nds_vt[2 * (offset_v_t + i_v_collision) + 1]
            particle_displacement = planar_truncation(particle_pos, particle_displacement, n, d, parallel_eps, gamma)

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
            colliding_v = get_triangle_colliding_vertices(collision_info, tri_index, i_t_collision)

            offset_t_v = collision_info.triangle_colliding_vertices_offsets[tri_index]
            is_dummy = division_plane_is_dummy_tv[3 * (offset_t_v + i_t_collision) + vertex_order]

            n = division_plane_nds_tv[2 * (offset_t_v + i_t_collision)]
            d = division_plane_nds_tv[2 * (offset_t_v + i_t_collision) + 1]
            particle_displacement = planar_truncation(particle_pos, particle_displacement, n, d, parallel_eps,
                                                          gamma)

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index,
                                                                                 i_adj_edge)

        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):
                colliding_e = get_edge_colliding_edges(collision_info, nei_edge_index, i_e_collision)

                offset_e_e = collision_info.edge_colliding_edges_offsets[nei_edge_index]
                vertex_order_on_edge_2_v1_v2_only = vertex_order_on_edge - 2

                is_dummy = division_plane_is_dummy_ee[
                    2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only]
                # n points to the valid half of the half space
                n = division_plane_nds_ee[2 * (offset_e_e + i_e_collision)]
                d = division_plane_nds_ee[2 * (offset_e_e + i_e_collision) + 1]
                particle_displacement = planar_truncation(particle_pos, particle_displacement, n, d, parallel_eps,
                                                              gamma)

    len_displacement = wp.length(particle_displacement)
    if len_displacement > max_displacement:
        particle_displacement = particle_displacement * max_displacement / len_displacement

    return particle_displacement


@wp.kernel
def penetration_free_projection(
        # inputs
        max_iter: int,
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        division_plane_nds_vt: wp.array(dtype=wp.vec3),  # shape = vertex_colliding_triangles.shape
        projection_t_vt: wp.array(dtype=float),  # shape = vertex_colliding_triangles.shape // 2
        division_plane_is_dummy_vt: wp.array(dtype=wp.bool),  # shape = vertex_colliding_triangles.shape // 2
        division_plane_nds_ee: wp.array(dtype=wp.vec3),  # shape = edge_colliding_edges.shape
        projection_t_ee: wp.array(dtype=float),  # shape = vertex_colliding_triangles.shape // 2
        division_plane_is_dummy_ee: wp.array(dtype=wp.bool),  # shape = edge_colliding_edges.shape[0] // 2
        division_plane_nds_tv: wp.array(dtype=wp.vec3),
        # self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2
        projection_t_tv: wp.array(dtype=float),
        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        division_plane_is_dummy_tv: wp.array(dtype=wp.bool),
        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        parallel_eps: float,
        gamma: float,
        max_displacement:float,
        # outputs
        displacement_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_index = tid
    particle_pos = pos[particle_index]
    particle_displacement = displacement_in[particle_index]

    initialize_projection(
        particle_index,
        particle_pos,
        particle_displacement,
        pos,
        displacement_in,
        tri_indices,
        edge_indices,
        adjacency,
        collision_info,
        division_plane_nds_vt,
        projection_t_vt,
        division_plane_is_dummy_vt,
        division_plane_nds_ee,
        projection_t_ee,
        division_plane_is_dummy_ee,
        division_plane_nds_tv,
        projection_t_tv,
        division_plane_is_dummy_tv,
    )

    x_proj = run_dijkstra_projection(
        max_iter,
        particle_index,
        particle_pos,
        particle_displacement,
        adjacency,
        collision_info,
        division_plane_nds_vt,
        projection_t_vt,
        division_plane_is_dummy_vt,
        division_plane_nds_ee,
        projection_t_ee,
        division_plane_is_dummy_ee,
        division_plane_nds_tv,
        projection_t_tv,
        division_plane_is_dummy_tv,
    )

    particle_displacement_proj = x_proj - particle_pos
    particle_displacement_truncated = run_penetration_free_truncation_after_projection(
        particle_index,
        particle_pos,
        particle_displacement_proj,
        adjacency,
        collision_info,
        division_plane_nds_vt,
        division_plane_is_dummy_vt,
        division_plane_nds_ee,
        division_plane_is_dummy_ee,
        division_plane_nds_tv,
        division_plane_is_dummy_tv,
        parallel_eps,
        gamma,
        max_displacement,
    )

    displacement_out[particle_index] = particle_displacement_truncated

@wp.kernel
def penetration_free_projection_v2(
        # inputs
        max_iter: int,
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info: TriMeshCollisionInfo,
        projection_vertex_offsets: wp.array(dtype=int),
        division_plane_nds: wp.array(dtype=wp.vec3),
        projection_ts: wp.array(dtype=float),
        num_projection_planes: wp.array(dtype=int),
        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        parallel_eps: float,
        gamma: float,
        max_displacement:float,
        # outputs
        displacement_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_index = tid
    particle_pos = pos[particle_index]
    particle_displacement = displacement_in[particle_index]
    initialize_projection_v2(
        particle_index,
        particle_pos,
        particle_displacement,
        pos,
        displacement_in,
        tri_indices,
        edge_indices,
        adjacency,
        collision_info,
        projection_vertex_offsets,
        division_plane_nds,
        projection_ts,
        num_projection_planes
    )

    displacement_truncated = run_dijkstra_projection_v2(
        max_iter,
        particle_index,
        particle_pos,
        particle_displacement,
        projection_vertex_offsets,
        division_plane_nds,
        projection_ts,
        num_projection_planes,
    )

    displacement_out[particle_index] = displacement_truncated


def plot_scene(
    v: np.ndarray,
    tri: np.ndarray,
    vectors: np.ndarray,
    n: np.ndarray,
    p_on_plane: np.ndarray,
    plane_extent_scale: float = 1.2,
    plane_alpha: float = 0.25,
    ax=None,
    vectors_truncated=None
):
    """
    Plot a 3D vertex, triangle, 4 vectors from vertex, and a plane.

    v: (3,) array — the vertex
    tri: (3,3) array — triangle vertices
    vectors: (4,3) array — 4 vectors drawn from v
    n: (3,) array — plane normal
    p_on_plane: (3,) array — a point on the plane
    """
    v = np.asarray(v, float)
    tri = np.asarray(tri, float)
    vectors = np.asarray(vectors, float)
    n = np.asarray(n, float)
    p_on_plane = np.asarray(p_on_plane, float)

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

    # triangle face
    tri_poly = Poly3DCollection([tri], alpha=0.3, facecolor='C0', edgecolor='k')
    ax.add_collection3d(tri_poly)
    ax.scatter(tri[:,0], tri[:,1], tri[:,2], color='C0', s=40, label='triangle vertices')

    # vertex
    ax.scatter([v[0]], [v[1]], [v[2]], color='C3', s=60, label='vertex v')

    # 4 vectors
    for i, vec in enumerate(vectors):
        if i == 0:
            ax.quiver(v[0], v[1], v[2],
                      vec[0], vec[1], vec[2],
                      color=f"C{i+1}" if vectors_truncated is None else "C0", linewidth=2, arrow_length_ratio=0.1,
                      label=f'vector {i+1}')
        else:
            ax.quiver(tri[i-1, 0], tri[i-1, 1], tri[i-1, 2],
                      vec[0], vec[1], vec[2],
                      color=f"C{i+1}" if vectors_truncated is None else "C0", linewidth=2, arrow_length_ratio=0.1,
                      label=f'vector {i+1}')

    if vectors_truncated is not None:
        for i, vec in enumerate(vectors_truncated):

            if i == 0:
                ax.quiver(v[0], v[1], v[2],
                          vec[0], vec[1], vec[2],
                          color="C1", linewidth=4, arrow_length_ratio=0.2,
                          label=f'vector truncated {i+1}')
            else:
                ax.quiver(tri[i-1, 0], tri[i-1, 1], tri[i-1, 2],
                          vec[0], vec[1], vec[2],
                          color="C1", linewidth=4, arrow_length_ratio=0.2,
                          label=f'vector truncated {i+1}')

    # plane
    n_hat = n / np.linalg.norm(n)
    helper = np.array([1,0,0]) if not np.allclose(n_hat, [1,0,0]) else np.array([0,1,0])
    u1 = np.cross(n_hat, helper); u1 /= np.linalg.norm(u1)
    u2 = np.cross(n_hat, u1)
    pts = np.vstack([tri, v[None], v+vectors])
    diag = np.linalg.norm(pts.max(0)-pts.min(0))
    half = 0.5*plane_extent_scale*diag
    us = np.linspace(-half, half, 2)
    vs = np.linspace(-half, half, 2)
    plane_pts = np.array([p_on_plane+uu*u1+vv*u2 for uu in us for vv in vs]).reshape(2,2,3)
    quad = [plane_pts[0,0], plane_pts[1,0], plane_pts[1,1], plane_pts[0,1]]
    ax.add_collection3d(Poly3DCollection([quad], alpha=plane_alpha, facecolor='C2'))
    ax.plot([], [], [], color='C2', label='plane')

    # aspect
    all_pts = np.vstack([pts, plane_pts.reshape(-1,3)])
    center = all_pts.mean(0)
    r = (all_pts.max(0)-all_pts.min(0)).max()/2
    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)
    try: ax.set_box_aspect([1,1,1])
    except: pass

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    ax.set_title('Vertex, Triangle, 4 Vectors, and Plane')
    return ax
