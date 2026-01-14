import numpy as np
import warp as wp
import json

from newton import ModelBuilder
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton._src.solvers.vbd.solver_vbd import *

from M03_Evaluation import *
from M01_Truncation import *

# ---------------------------
# Simple tri-tri intersection (no-division variant + coplanar handler)
# ---------------------------
_EPS = 1e-8

def _dot(a, b): return float(np.dot(a, b))
def _sub(a, b): return (a - b).astype(np.float64)
def _cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1],
                     a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]], dtype=np.float64)

def _absmax_axis(v): return int(np.argmax(np.abs(v)))
def _orient2d(a, b, c): return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _project_coplanar_to_2d(N, P3):
    ax = _absmax_axis(N)
    if ax == 0:   # drop X
        return P3[:, [1, 2]]
    elif ax == 1: # drop Y
        return P3[:, [0, 2]]
    else:         # drop Z
        return P3[:, [0, 1]]

def _seg_seg_intersect_2d(a, b, c, d):
    def on_seg(p, q, r):
        minx, maxx = min(p[0], q[0]) - _EPS, max(p[0], q[0]) + _EPS
        miny, maxy = min(p[1], q[1]) - _EPS, max(p[1], q[1]) + _EPS
        if r[0] < minx or r[0] > maxx or r[1] < miny or r[1] > maxy: return False
        return abs(_orient2d(p, q, r)) <= 1e-12
    o1 = _orient2d(a, b, c); o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a); o4 = _orient2d(c, d, b)
    if (o1 == 0 and on_seg(a, b, c)) or (o2 == 0 and on_seg(a, b, d)) or \
       (o3 == 0 and on_seg(c, d, a)) or (o4 == 0 and on_seg(c, d, b)):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def _point_in_tri_2d(p, tri):
    a, b, c = tri
    w1 = _orient2d(a, b, p); w2 = _orient2d(b, c, p); w3 = _orient2d(c, a, p)
    has_pos = (w1 > 0) or (w2 > 0) or (w3 > 0)
    has_neg = (w1 < 0) or (w2 < 0) or (w3 < 0)
    return not (has_pos and has_neg)

def _coplanar_tri_tri(N, V0, V1, V2, U0, U1, U2):
    A2 = _project_coplanar_to_2d(N, np.array([V0, V1, V2], dtype=np.float64))
    B2 = _project_coplanar_to_2d(N, np.array([U0, U1, U2], dtype=np.float64))
    Ae = [(A2[0], A2[1]), (A2[1], A2[2]), (A2[2], A2[0])]
    Be = [(B2[0], B2[1]), (B2[1], B2[2]), (B2[2], B2[0])]
    for a0, a1 in Ae:
        for b0, b1 in Be:
            if _seg_seg_intersect_2d(a0, a1, b0, b1): return True
    if _point_in_tri_2d(A2[0], B2) or _point_in_tri_2d(B2[0], A2): return True
    return False

def tri_tri_intersect_no_div(V0, V1, V2, U0, U1, U2, eps=_EPS, use_epsilon_test=True):
    V0 = np.asarray(V0, dtype=np.float64); V1 = np.asarray(V1, dtype=np.float64); V2 = np.asarray(V2, dtype=np.float64)
    U0 = np.asarray(U0, dtype=np.float64); U1 = np.asarray(U1, dtype=np.float64); U2 = np.asarray(U2, dtype=np.float64)

    E1 = _sub(V1, V0); E2 = _sub(V2, V0)
    N1 = _cross(E1, E2); d1 = -_dot(N1, V0)
    du0 = _dot(N1, U0) + d1; du1 = _dot(N1, U1) + d1; du2 = _dot(N1, U2) + d1
    if use_epsilon_test:
        if abs(du0) < eps: du0 = 0.0
        if abs(du1) < eps: du1 = 0.0
        if abs(du2) < eps: du2 = 0.0
    if du0*du1 > 0.0 and du0*du2 > 0.0: return False

    E1 = _sub(U1, U0); E2 = _sub(U2, U0)
    N2 = _cross(E1, E2); d2 = -_dot(N2, U0)
    dv0 = _dot(N2, V0) + d2; dv1 = _dot(N2, V1) + d2; dv2 = _dot(N2, V2) + d2
    if use_epsilon_test:
        if abs(dv0) < eps: dv0 = 0.0
        if abs(dv1) < eps: dv1 = 0.0
        if abs(dv2) < eps: dv2 = 0.0
    if dv0*dv1 > 0.0 and dv0*dv2 > 0.0: return False

    D = _cross(N1, N2)
    idx = int(np.argmax(np.abs(D)))
    vp0, vp1, vp2 = V0[idx], V1[idx], V2[idx]
    up0, up1, up2 = U0[idx], U1[idx], U2[idx]

    def _new_compute_intervals(v0, v1, v2, d0, d1, d2):
        d0d1, d0d2 = d0*d1, d0*d2
        if d0d1 > 0.0:
            a = v2; b = (v0 - v2)*d2; c = (v1 - v2)*d2; x0 = d2 - d0; x1 = d2 - d1
        elif d0d2 > 0.0:
            a = v1; b = (v0 - v1)*d1; c = (v2 - v1)*d1; x0 = d1 - d0; x1 = d1 - d2
        elif d1*d2 > 0.0 or d0 != 0.0:
            a = v0; b = (v1 - v0)*d0; c = (v2 - v0)*d0; x0 = d0 - d1; x1 = d0 - d2
        elif d1 != 0.0:
            a = v1; b = (v0 - v1)*d1; c = (v2 - v1)*d1; x0 = d1 - d0; x1 = d1 - d2
        elif d2 != 0.0:
            a = v2; b = (v0 - v2)*d2; c = (v1 - v2)*d2; x0 = d2 - d0; x1 = d2 - d1
        else:
            return None
        return a, b, c, x0, x1

    out1 = _new_compute_intervals(vp0, vp1, vp2, dv0, dv1, dv2)
    if out1 is None: return _coplanar_tri_tri(N1, V0, V1, V2, U0, U1, U2)
    a, b, c, x0, x1 = out1
    out2 = _new_compute_intervals(up0, up1, up2, du0, du1, du2)
    if out2 is None: return _coplanar_tri_tri(N1, V0, V1, V2, U0, U1, U2)
    d, e, f, y0, y1 = out2

    xx = x0*x1; yy = y0*y1; xxyy = xx*yy
    i10 = a*xxyy + b*x1*yy; i11 = a*xxyy + c*x0*yy
    i20 = d*xxyy + e*xx*y1; i21 = d*xxyy + f*xx*y0
    if i10 > i11: i10, i11 = i11, i10
    if i20 > i21: i20, i21 = i21, i20
    return not (i11 < i20 or i21 < i10)

def _triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def _aabb(v0, v1, v2):
    P = np.stack([v0, v1, v2], axis=0)
    return P.min(axis=0), P.max(axis=0)

def _aabb_overlap(lo1, hi1, lo2, hi2):
    return not ((hi1 < lo2).any() or (hi2 < lo1).any())

# ---------------------------
# Simple nearest-K based random mesh generator
# ---------------------------
class DataGenerator:
    def __init__(self):
        self.data_counter = 0

    def get_data_set_size(self):
        return -1

    def get_mesh(self):
        # vertice, triangles, translation
        return np.zeros((10, 3), dtype=float), np.zeros((10, 3), dtype=wp.int32), np.zeros((10, 3), dtype=wp.int32),

class RandomMeshGenerator(DataGenerator):
    """
    Simple random mesh:
      1) Create N vertices uniformly in a box.
      2) For each triangle attempt:
         - pick a random seed vertex i
         - choose two of its K nearest neighbors whose edges are near target length
         - form triangle if non-degenerate and non-intersecting (adjacent is allowed)
      3) Stop when we have M triangles or fail_limit reached.
    """
    def __init__(self,
                 data_set_size:int,
                 num_triangles: int,
                 world_size=1.0,
                 target_edge=0.25,
                 edge_tol_low=0.1,         # accept edges in [(1-edge_tol), (1+edge_tol)] * target_edge
                 edge_tol_high=3.0,         # accept edges in [(1-edge_tol), (1+edge_tol)] * target_edge
                 k_neighbors=16,
                 displacement_scale=0.05,
                 shrink_ratio=0.95,
                 displacements_per_mesh=1,
                 seed=123,
                 fail_limit=50_000):
        super().__init__()

        self.num_triangles = int(num_triangles)
        self.world_size = np.array([world_size]*3, dtype=float) if np.isscalar(world_size) \
                          else np.array(world_size, dtype=float)
        self.target_edge = float(target_edge)
        self.edge_tol_low = float(edge_tol_low)
        self.edge_tol_high = float(edge_tol_high)
        self.k_neighbors = int(k_neighbors)
        self.seed = int(seed)
        self.fail_limit = int(fail_limit)
        self.displacement_scale=displacement_scale
        self.shrink_ratio=shrink_ratio

        self.rng = np.random.default_rng(self.seed)
        self.displacements_per_mesh = displacements_per_mesh
        self.data_set_size = data_set_size



    def _knn_lists(self, V):
        # very simple: compute KNN per vertex via distances (O(N^2), but simple)
        N = V.shape[0]
        K = min(self.k_neighbors, max(1, N-1))
        nbr_idx = np.empty((N, K), dtype=np.int32)
        nbr_dst = np.empty((N, K), dtype=np.float64)
        for i in range(N):
            d = np.linalg.norm(V - V[i], axis=1)
            d[i] = np.inf
            idx = np.argpartition(d, K)[:K]
            # sort those K by distance
            order = np.argsort(d[idx])
            idx = idx[order]
            nbr_idx[i, :] = idx
            nbr_dst[i, :] = d[idx]
        return nbr_idx, nbr_dst

    def generate_mesh(self):
        rng = self.rng
        V_all = np.zeros((3*self.num_triangles, 3))
        V = np.zeros((0, 3))

        tris = []                           # list of (i,j,k)
        tri_aabbs = []                      # list of (lo,hi) for quick reject

        fails = 0
        AREA_EPS = 1e-8


        while len(tris) < self.num_triangles and fails < self.fail_limit:

            # non-degenerate area
            center =  rng.uniform(low=0, high=self.world_size, size=3)
            v0 = center + rng.normal(loc=0, scale=self.target_edge, size=3)
            v1 = center + rng.normal(loc=0, scale=self.target_edge, size=3)
            v2 = center + rng.normal(loc=0, scale=self.target_edge, size=3)
            if _triangle_area(v0, v1, v2) < AREA_EPS:
                fails += 1
                continue

            # intersection test vs existing tris (adjacent allowed)
            lo, hi = _aabb(v0, v1, v2)
            ok = True
            for t_idx, (a, b, c) in enumerate(tris):

                lo_u, hi_u = tri_aabbs[t_idx]
                if not _aabb_overlap(lo, hi, lo_u, hi_u):
                    continue
                u0, u1, u2 = V[a], V[b], V[c]
                if tri_tri_intersect_no_div(v0, v1, v2, u0, u1, u2, eps=1e-5, use_epsilon_test=True):
                    ok = False
                    break

            if not ok:
                fails += 1
                continue

            # accept
            i = len(tris) * 3
            j = len(tris) * 3 + 1
            k = len(tris) * 3 + 2
            tris.append((i, j, k))
            tri_aabbs.append((lo, hi))

            V_all[i, :] = v0
            V_all[j, :] = v1
            V_all[k, :] = v2

            V = V_all[:3*len(tris)]

        T = np.asarray(tris, dtype=np.int32)
        V = np.asarray(V, dtype=float)

        # shrink triangles toward centroid
        if hasattr(self, "shrink_ratio") and self.shrink_ratio != 0.0:
            s = float(self.shrink_ratio)
            V_new = V.copy()
            for (i, j, k) in tris:
                c = (V[i] + V[j] + V[k]) / 3.0  # centroid
                V_new[i] = c + s * (V[i] - c)
                V_new[j] = c + s * (V[j] - c)
                V_new[k] = c + s * (V[k] - c)
            V = V_new

        self.vs = V.astype(float)
        self.ts = T

    def get_mesh(self):
        if not self.data_counter % self.displacements_per_mesh:
            self.generate_mesh()

        self.displacements = self.rng.normal(loc=0,size=3*self.vs.shape[0]).reshape(-1,3)
        self.displacements = self.displacement_scale * self.displacements / np.linalg.norm(self.displacements, axis=1)[..., None]

        return self.vs, self.ts, self.displacements

class Truncator:
    def __init__(self, collision_query_radius: float, bound_relaxation: float):
        self.bound_relaxation = bound_relaxation
        self.collision_query_radius = collision_query_radius

    def analyze_shape(self, collision_detector: TriMeshCollisionDetector, adjacency: ForceElementAdjacencyInfo,):
        self.model = collision_detector.model
        self.collision_detector = collision_detector
        self.adjacency = adjacency

        self.collision_detector.vertex_triangle_collision_detection(self.collision_query_radius)
        self.collision_detector.edge_edge_collision_detection(self.collision_query_radius)


    def truncate(self, positions, displacements):
        return np.zeros_like(displacements)

class TruncatorOffset(Truncator):
    @wp.kernel
    def offset_truncation(
        pos_prev_collision_detection: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        particle_conservative_bounds: wp.array(dtype=float),
        displacement_out: wp.array(dtype=wp.vec3),
    ):
        particle_index = wp.tid()

        pos = pos_prev_collision_detection[particle_index]

        displacement_out[particle_index] = apply_conservative_bound_truncation(
            particle_index, displacement_in[particle_index] + pos, pos_prev_collision_detection, particle_conservative_bounds
        ) - pos_prev_collision_detection[particle_index]

    def truncate(self, positions, displacements):

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.displacements_in = wp.array(displacements, dtype=wp.vec3)
        self.displacements_out = wp.zeros_like(self.displacements_in)

        self.particle_conservative_bounds = wp.zeros(displacements.shape[0], dtype=float)

        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.bound_relaxation,
                self.collision_query_radius,
                self.adjacency,
                self.collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
        )

        wp.launch(
            kernel=self.offset_truncation,
            inputs=[
                self.positions,
                self.displacements_in,
                self.particle_conservative_bounds,
            ],
            outputs=[
                self.displacements_out,
            ],
            dim=self.model.particle_count,
        )

        return self.displacements_out.numpy()

class TruncatorPlanar(Truncator):
    def truncate(self, positions, displacements):

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.displacements_in = wp.array(displacements, dtype=wp.vec3)
        self.displacements_out = wp.zeros_like(self.displacements_in)

        self.particle_conservative_bounds = wp.zeros(displacements.shape[0], dtype=float)

        wp.launch(
            kernel=penetration_free_truncation,
            inputs=[
                self.positions,
                self.displacements_in,
                self.model.tri_indices,
                self.model.edge_indices,
                self.adjacency,
                self.collision_detector.collision_info,
                1e-8,
                self.bound_relaxation,
                self.collision_query_radius * 0.45
            ],
            outputs=[self.displacements_out],
            dim = self.model.particle_count,
        )

        return self.displacements_out.numpy()

class TruncatorPlanar_v2(Truncator):
    def analyze_shape(self, collision_detector: TriMeshCollisionDetector, adjacency: ForceElementAdjacencyInfo,):
        self.model = collision_detector.model
        self.collision_detector = collision_detector
        self.adjacency = adjacency

        projection_buffer_sizes = wp.zeros(self.model.particle_count, dtype=int)

        wp.launch(
            dim=self.model.particle_count,
            kernel=calculate_vertex_collision_buffer,
            inputs=[
                self.adjacency,
                self.collision_detector.collision_info,
            ],
            outputs=[projection_buffer_sizes]
        )

        vertex_division_plane_buffer_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
        vertex_division_plane_buffer_offsets[1:] = np.cumsum(projection_buffer_sizes.numpy())[:]
        vertex_division_plane_buffer_offsets[0] = 0
        buffer_size_total = vertex_division_plane_buffer_offsets[-1]

        self.vertex_division_plane_buffer_offsets = wp.array(vertex_division_plane_buffer_offsets, dtype=int)

        self.collision_detector.vertex_triangle_collision_detection(self.collision_query_radius)
        self.collision_detector.edge_edge_collision_detection(self.collision_query_radius)

        self.division_plane_nds = wp.empty(buffer_size_total * 2, dtype=wp.vec3)
        self.division_num_planes = wp.empty(self.model.particle_count, dtype=int)

        #         particle_pos: wp.vec3,
        #         particle_displacement: wp.vec3,
        #         pos: wp.array(dtype=wp.vec3),
        #         displacement_in: wp.array(dtype=wp.vec3),
        #         tri_indices: wp.array(dtype=wp.int32, ndim=2),
        #         edge_indices: wp.array(dtype=wp.int32, ndim=2),
        #         adjacency: ForceElementAdjacencyInfo,
        #         collision_info: TriMeshCollisionInfo,
        #         projection_vertex_offsets: wp.array(dtype=int),
        #         division_plane_nds: wp.array(dtype=wp.vec3),
        #         num_projection_planes: wp.array(dtype=int),



    def truncate(self, positions, displacements):

        self.positions = wp.array(positions, dtype=wp.vec3)
        self.displacements_in = wp.array(displacements, dtype=wp.vec3)
        self.displacements_out = wp.zeros_like(self.displacements_in)

        self.particle_conservative_bounds = wp.zeros(displacements.shape[0], dtype=float)
        wp.launch(
            dim=self.model.particle_count,
            kernel=initialize_truncation_planes,
            inputs=[
                self.positions,
                self.displacements_in,
                self.model.tri_indices,
                self.model.edge_indices,
                self.adjacency,
                self.collision_detector.collision_info,
                self.vertex_division_plane_buffer_offsets,
                self.division_plane_nds,
                self.division_num_planes
            ]
        )

        wp.launch(
            kernel=run_penetration_free_truncation_v2,
            inputs=[
                self.positions,
                self.displacements_in,
                1e-8,
                self.bound_relaxation,
                self.collision_query_radius * 0.45,
                self.vertex_division_plane_buffer_offsets,
                self.division_plane_nds,
                self.division_num_planes
            ],
            outputs=[self.displacements_out],
            dim = self.model.particle_count,
        )

        return self.displacements_out.numpy()

class TruncatorProjection(TruncatorPlanar):
    def __init__(self, collision_query_radius: float, bound_relaxation: float, max_iter:int=5):
        super().__init__(collision_query_radius, bound_relaxation)
        self.max_iter = max_iter

    def analyze_shape(self, collision_detector: TriMeshCollisionDetector, adjacency: ForceElementAdjacencyInfo,):
        self.model = collision_detector.model
        self.collision_detector = collision_detector
        self.adjacency = adjacency

        self.collision_detector.vertex_triangle_collision_detection(self.collision_query_radius)
        self.collision_detector.edge_edge_collision_detection(self.collision_query_radius)

        self.division_plane_nds_vt =  wp.empty(self.collision_detector.collision_info.vertex_colliding_triangles.shape, dtype=wp.vec3)
        self.division_plane_is_dummy_vt =  wp.empty(self.collision_detector.collision_info.vertex_colliding_triangles.shape[0] // 2, dtype=wp.bool)
        self.projection_t_vt = wp.empty(self.collision_detector.collision_info.vertex_colliding_triangles.shape[0] // 2, dtype=float)

        self.division_plane_nds_ee =  wp.empty(self.collision_detector.collision_info.edge_colliding_edges.shape, dtype=wp.vec3)
        # 2 per collision pair: one for each vertex on the edge
        self.division_plane_is_dummy_ee =  wp.empty(self.collision_detector.collision_info.edge_colliding_edges.shape[0], dtype=wp.bool)
        # 2 per collision pair: one for each vertex on the edge
        self.projection_t_ee = wp.empty(self.collision_detector.collision_info.edge_colliding_edges.shape[0], dtype=float)

        # we also need to recompute the division plane from the plane triangle, because there is no easy way to access the n, d from the vertex side (index does not match)
        # also triangle_colliding_vertices records 1 element per v-t pair, therefore we need to double the size
        self.division_plane_nds_tv =  wp.empty(self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2, dtype=wp.vec3)
        # need 3 for each pair: one for one vertex on the triangle
        self.division_plane_is_dummy_tv =  wp.empty(self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 3, dtype=wp.bool)
        # y is recorded for each division plane, therefore each tv also need a y, since it's a different constraint
        # need 3 for each pair: one for one vertex on the triangle
        self.projection_t_tv = wp.empty(self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 3, dtype=float)

    def truncate(self, positions, displacements):
        self.positions = wp.array(positions, dtype=wp.vec3)
        self.displacements_in = wp.array(displacements, dtype=wp.vec3)
        self.displacements_out = wp.zeros_like(self.displacements_in)

        wp.launch(
            kernel=penetration_free_projection,
            inputs=[
                self.max_iter,
                self.positions,
                self.displacements_in,
                self.model.tri_indices,
                self.model.edge_indices,
                self.adjacency,
                self.collision_detector.collision_info,
                self.division_plane_nds_vt,  # shape = vertex_colliding_triangles.shape
                self.projection_t_vt,  # shape = vertex_colliding_triangles.shape // 2
                self.division_plane_is_dummy_vt,  # shape = vertex_colliding_triangles.shape // 2
                self.division_plane_nds_ee,  # shape = edge_colliding_edges.shape
                self.projection_t_ee,  # shape = vertex_colliding_triangles.shape // 2
                self.division_plane_is_dummy_ee,  # shape = edge_colliding_edges.shape[0] // 2
                self.division_plane_nds_tv,
                # self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2
                self.projection_t_tv,
                # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
                self.division_plane_is_dummy_tv,
                # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
                1e-8,
                self.bound_relaxation,
                self.collision_query_radius * 0.45
            ],
            outputs=[self.displacements_out],
            dim=self.model.particle_count,
        )

        return self.displacements_out.numpy()

class TruncatorProjection_v2(TruncatorPlanar):
    def __init__(self, collision_query_radius: float, bound_relaxation: float, max_iter:int):
        super().__init__(collision_query_radius, bound_relaxation)
        self.max_iter = max_iter

    def analyze_shape(self, collision_detector: TriMeshCollisionDetector, adjacency: ForceElementAdjacencyInfo,):
        self.model = collision_detector.model
        self.collision_detector = collision_detector
        self.adjacency = adjacency

        projection_buffer_sizes = wp.zeros(self.model.particle_count, dtype=int)

        wp.launch(
            dim=self.model.particle_count,
            kernel=calculate_vertex_collision_buffer,
            inputs=[
                self.adjacency,
                self.collision_detector.collision_info,
            ],
            outputs=[projection_buffer_sizes]
        )

        vertex_projection_buffer_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
        vertex_projection_buffer_offsets[1:] = np.cumsum(projection_buffer_sizes.numpy())[:]
        vertex_projection_buffer_offsets[0] = 0
        buffer_size_total = vertex_projection_buffer_offsets[-1]

        self.vertex_projection_buffer_offsets = wp.array(vertex_projection_buffer_offsets, dtype=int)

        self.collision_detector.vertex_triangle_collision_detection(self.collision_query_radius)
        self.collision_detector.edge_edge_collision_detection(self.collision_query_radius)

        self.projection_plane_nds =  wp.empty(buffer_size_total * 2, dtype=wp.vec3)
        self.projection_ts = wp.empty(buffer_size_total, dtype=float)
        self.projection_num_planes = wp.empty(self.model.particle_count, dtype=int)


    def truncate(self, positions, displacements):
        self.positions = wp.array(positions, dtype=wp.vec3)
        self.displacements_in = wp.array(displacements, dtype=wp.vec3)
        self.displacements_out = wp.zeros_like(self.displacements_in)

        wp.launch(
            kernel=penetration_free_projection_v2,
            inputs=[
                self.max_iter,
                self.positions,
                self.displacements_in,
                self.model.tri_indices,
                self.model.edge_indices,
                self.adjacency,
                self.collision_detector.collision_info,
                self.vertex_projection_buffer_offsets,
                self.projection_plane_nds,
                self.projection_ts,
                self.projection_num_planes,
                1e-8,
                self.bound_relaxation,
                self.collision_query_radius * 0.45
            ],
            outputs=[self.displacements_in],
            dim=self.model.particle_count,
        )

        # truncate after projection
        wp.launch(
            kernel=penetration_free_truncation,
            inputs=[
                self.positions,
                self.displacements_in,
                self.model.tri_indices,
                self.model.edge_indices,
                self.adjacency,
                self.collision_detector.collision_info,
                1e-8,
                self.bound_relaxation,
                self.collision_query_radius * 0.45
            ],
            outputs=[self.displacements_out],
            dim=self.model.particle_count,
        )

        return self.displacements_out.numpy()

def run_truncation(mesh_gen, truncator, hyper_parameters):
    query_radius = hyper_parameters["collision_query_radius"]

    evaluator = TruncationAnalyzer()
    reports = []
    for _ in range(mesh_gen.data_set_size):
        vs, ts, displacements = mesh_gen.get_mesh()

        builder = ModelBuilder()

        vertices = [wp.vec3(vs[i, :]) for i in range(vs.shape[0])]
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vertices,
            indices=ts.reshape(-1),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=2.0e-6,
            edge_ke=10,
        )
        builder.color()
        model = builder.finalize()

        vbd_integrator = SolverVBD(model)
        collision_detector = TriMeshCollisionDetector(
            model,
            vertex_collision_buffer_pre_alloc=512,
            triangle_collision_buffer_pre_alloc=512,
            edge_collision_buffer_pre_alloc=512,
            triangle_triangle_collision_buffer_pre_alloc=128,
            record_triangle_contacting_vertices=True
        )

        if hyper_parameters["evaluate_initial_state"]:
            if check_intersection(collision_detector, vs, None, plot=False):
                print("Input data ", mesh_gen.data_counter, " is intersection-free!")
            else:
                print("Input data ", mesh_gen.data_counter, " has intersection! Skipping this data! ")
                continue

        truncator.analyze_shape(collision_detector, vbd_integrator.adjacency)
        displacements_truncated = truncator.truncate(vs, displacements)

        # ps.register_surface_mesh("mesh_deformed", vs + displacements_truncated, ts)
        # mesh_org = ps.register_surface_mesh("mesh_org", vs, ts)
        # mesh_org.add_vector_quantity("rand vecs", displacements, enabled=True)
        # ps.show()

        if not check_intersection(collision_detector, vs, displacements_truncated, plot=True):
            print("Error! intersection happens after truncation!")

        report = evaluator.analyze(vs, ts, displacements, displacements_truncated, print_results=False)
        reports.append(report)
        print(json.dumps(report, indent=4))
    print("------------------------------------\nSummary\n------------------------------------\n")

    print(print(json.dumps(evaluator.combine_reports(reports), indent=4)))