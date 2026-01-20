# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings

import numpy as np
import warp as wp
from warp.types import matrix, vector

from ...core.types import override
from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
    TriMeshContinuousCollisionDetector,
    get_edge_colliding_edges,
    get_edge_colliding_edges_count,
    get_triangle_colliding_vertices,
    get_triangle_colliding_vertices_count,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
)

# TODO: Grab changes from Warp that has fixed the backward pass
wp.set_module_options({"enable_backward": False})

VBD_DEBUG_PRINTING_OPTIONS = {
    # "elasticity_force_hessian",
    # "contact_force_hessian",
    # "contact_force_hessian_vt",
    # "contact_force_hessian_ee",
    # "overall_force_hessian",
    # "inertia_force_hessian",
    # "connectivity",
    # "contact_info",
}

NUM_THREADS_PER_COLLISION_PRIMITIVE = 4
TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 16
TILE_SIZE_SELF_CONTACT_SOLVE = 64


class mat32(matrix(shape=(3, 2), dtype=wp.float32)):
    pass


class mat99(matrix(shape=(9, 9), dtype=wp.float32)):
    pass


class mat43(matrix(shape=(4, 3), dtype=wp.float32)):
    pass


class vec9(vector(length=9, dtype=wp.float32)):
    pass


@wp.struct
class ForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    v_adj_tets: wp.array(dtype=int)
    v_adj_tets_offsets: wp.array(dtype=int)

    v_adj_springs: wp.array(dtype=int)
    v_adj_springs_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_faces.device:
            return self
        else:
            adjacency_gpu = ForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            adjacency_gpu.v_adj_tets = self.v_adj_tets.to(device)
            adjacency_gpu.v_adj_tets_offsets = self.v_adj_tets_offsets.to(device)

            adjacency_gpu.v_adj_springs = self.v_adj_springs.to(device)
            adjacency_gpu.v_adj_springs_offsets = self.v_adj_springs_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_edges(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_tets(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_tets_offsets[vertex + 1] - adjacency.v_adj_tets_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_tet_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, tet: wp.int32):
    offset = adjacency.v_adj_tets_offsets[vertex]
    return adjacency.v_adj_tets[offset + tet * 2], adjacency.v_adj_tets[offset + tet * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.kernel
def _test_compute_force_element_adjacency(
    adjacency: ForceElementAdjacencyInfo,
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    face_indices: wp.array(dtype=wp.int32, ndim=2),
):
    wp.printf("num vertices: %d\n", adjacency.v_adj_edges_offsets.shape[0] - 1)
    for vertex in range(adjacency.v_adj_edges_offsets.shape[0] - 1):
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, vertex)
        for i_bd in range(num_adj_edges):
            bd_id, v_order = get_vertex_adjacent_edge_id_order(adjacency, vertex, i_bd)

            if edge_indices[bd_id, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_edges: %d\n", vertex, num_adj_edges)
                wp.printf("--iBd: %d | ", i_bd)
                wp.printf("edge id: %d | v_order: %d\n", bd_id, v_order)

        num_adj_faces = get_vertex_num_adjacent_faces(adjacency, vertex)

        for i_face in range(num_adj_faces):
            face, v_order = get_vertex_adjacent_face_id_order(
                adjacency,
                vertex,
                i_face,
            )

            if face_indices[face, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_faces: %d\n", vertex, num_adj_faces)
                wp.printf("--i_face: %d | face id: %d | v_order: %d\n", i_face, face, v_order)
                wp.printf(
                    "--face: %d %d %d\n",
                    face_indices[face, 0],
                    face_indices[face, 1],
                    face_indices[face, 2],
                )


@wp.func
def build_orthonormal_basis(n: wp.vec3):
    """
    Builds an orthonormal basis given a normal vector `n`. Return the two axes that is perpendicular to `n`.

    :param n: A 3D vector (list or array-like) representing the normal vector
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def evaluate_stvk_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    # StVK energy density: psi = mu * ||G||_F^2 + 0.5 * lambda * (trace(G))^2

    # Deformation gradient F = [f0, f1] (3x2 matrix as two 3D column vectors)
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    # Cache tri_pose elements
    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    # Compute F columns directly: F = [x01, x02] * tri_pose = [f0, f1]
    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # Green strain tensor: G = 0.5(F^T F - I) = [[G00, G01], [G01, G11]] (symmetric 2x2)
    f0_dot_f0 = wp.dot(f0, f0)
    f1_dot_f1 = wp.dot(f1, f1)
    f0_dot_f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0_dot_f0 - 1.0)
    G11 = 0.5 * (f1_dot_f1 - 1.0)
    G01 = 0.5 * f0_dot_f1

    # Frobenius norm squared of Green strain: ||G||_F^2 = G00^2 + G11^2 + 2 * G01^2
    G_frobenius_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    if G_frobenius_sq < 1.0e-20:
        return wp.vec3(0.0), wp.mat33(0.0)

    trace_G = G00 + G11

    # First Piola-Kirchhoff stress tensor (StVK model)
    # PK1 = 2*mu*F*G + lambda*trace(G)*F = [PK1_col0, PK1_col1] (3x2)
    lambda_trace_G = lmbd * trace_G
    two_mu = 2.0 * mu

    PK1_col0 = f0 * (two_mu * G00 + lambda_trace_G) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lambda_trace_G)

    # Vertex selection using masks to avoid branching
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)

    # Deformation gradient derivatives w.r.t. current vertex position
    df0_dx = DmInv00 * (mask1 - mask0) + DmInv10 * (mask2 - mask0)
    df1_dx = DmInv01 * (mask1 - mask0) + DmInv11 * (mask2 - mask0)

    # Force via chain rule: force = -(dpsi/dF) : (dF/dx)
    dpsi_dx = PK1_col0 * df0_dx + PK1_col1 * df1_dx
    force = -dpsi_dx

    # Hessian computation using Cauchy-Green invariants
    df0_dx_sq = df0_dx * df0_dx
    df1_dx_sq = df1_dx * df1_dx
    df0_df1_cross = df0_dx * df1_dx

    Ic = f0_dot_f0 + f1_dot_f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbd
    I33 = wp.identity(n=3, dtype=float)

    f0_outer_f0 = wp.outer(f0, f0)
    f1_outer_f1 = wp.outer(f1, f1)
    f0_outer_f1 = wp.outer(f0, f1)
    f1_outer_f0 = wp.outer(f1, f0)

    H_IIc00_scaled = mu * (f0_dot_f0 * I33 + 2.0 * f0_outer_f0 + f1_outer_f1)
    H_IIc11_scaled = mu * (f1_dot_f1 * I33 + 2.0 * f1_outer_f1 + f0_outer_f0)
    H_IIc01_scaled = mu * (f0_dot_f1 * I33 + f1_outer_f0)

    # d2(psi)/dF^2 components
    d2E_dF2_00 = lmbd * f0_outer_f0 + two_dpsi_dIc * I33 + H_IIc00_scaled
    d2E_dF2_01 = lmbd * f0_outer_f1 + H_IIc01_scaled
    d2E_dF2_11 = lmbd * f1_outer_f1 + two_dpsi_dIc * I33 + H_IIc11_scaled

    # Chain rule: H = (dF/dx)^T * (d2(psi)/dF^2) * (dF/dx)
    hessian = df0_dx_sq * d2E_dF2_00 + df1_dx_sq * d2E_dF2_11 + df0_df1_cross * (d2E_dF2_01 + wp.transpose(d2E_dF2_01))

    if damping > 0.0:
        inv_dt = 1.0 / dt

        # Previous deformation gradient for velocity
        x0_prev = pos_prev[v0]
        x01_prev = pos_prev[v1] - x0_prev
        x02_prev = pos_prev[v2] - x0_prev

        vel_x01 = (x01 - x01_prev) * inv_dt
        vel_x02 = (x02 - x02_prev) * inv_dt

        df0_dt = vel_x01 * DmInv00 + vel_x02 * DmInv10
        df1_dt = vel_x01 * DmInv01 + vel_x02 * DmInv11

        # First constraint: Cmu = ||G||_F (Frobenius norm of Green strain)
        Cmu = wp.sqrt(G_frobenius_sq)

        G00_normalized = G00 / Cmu
        G01_normalized = G01 / Cmu
        G11_normalized = G11 / Cmu

        # Time derivative of Green strain: dG/dt = 0.5 * (F^T * dF/dt + (dF/dt)^T * F)
        dG_dt_00 = wp.dot(f0, df0_dt)  # dG00/dt
        dG_dt_11 = wp.dot(f1, df1_dt)  # dG11/dt
        dG_dt_01 = 0.5 * (wp.dot(f0, df1_dt) + wp.dot(f1, df0_dt))  # dG01/dt

        # Time derivative of first constraint: dCmu/dt = (1/||G||_F) * (G : dG/dt)
        dCmu_dt = G00_normalized * dG_dt_00 + G11_normalized * dG_dt_11 + 2.0 * G01_normalized * dG_dt_01

        # Gradient of first constraint w.r.t. deformation gradient: dCmu/dF = (G/||G||_F) * F
        dCmu_dF_col0 = G00_normalized * f0 + G01_normalized * f1  # dCmu/df0
        dCmu_dF_col1 = G01_normalized * f0 + G11_normalized * f1  # dCmu/df1

        # Gradient of constraint w.r.t. vertex position: dCmu/dx = (dCmu/dF) : (dF/dx)
        dCmu_dx = df0_dx * dCmu_dF_col0 + df1_dx * dCmu_dF_col1

        # Damping force from first constraint: -mu * damping * (dCmu/dt) * (dCmu/dx)
        kd_mu = mu * damping
        force += -kd_mu * dCmu_dt * dCmu_dx

        # Damping Hessian: mu * damping * (1/dt) * (dCmu/dx) x (dCmu/dx)
        hessian += kd_mu * inv_dt * wp.outer(dCmu_dx, dCmu_dx)

        # Second constraint: Clmbd = trace(G) = G00 + G11 (trace of Green strain)
        # Time derivative of second constraint: dClmbd/dt = trace(dG/dt)
        dClmbd_dt = dG_dt_00 + dG_dt_11

        # Gradient of second constraint w.r.t. deformation gradient: dClmbd/dF = F
        dClmbd_dF_col0 = f0  # dClmbd/df0
        dClmbd_dF_col1 = f1  # dClmbd/df1

        # Gradient of Clmbd w.r.t. vertex position: dClmbd/dx = (dClmbd/dF) : (dF/dx)
        dClmbd_dx = df0_dx * dClmbd_dF_col0 + df1_dx * dClmbd_dF_col1

        # Damping force from second constraint: -lambda * damping * (dClmbd/dt) * (dClmbd/dx)
        kd_lmbd = lmbd * damping
        force += -kd_lmbd * dClmbd_dt * dClmbd_dx

        # Damping Hessian from second constraint: lambda * damping * (1/dt) * (dClmbd/dx) x (dClmbd/dx)
        hessian += kd_lmbd * inv_dt * wp.outer(dClmbd_dx, dClmbd_dx)

    # Apply area scaling
    force *= area
    hessian *= area

    return force, hessian


@wp.func
def compute_normalized_vector_derivative(
    unnormalized_vec_length: float, normalized_vec: wp.vec3, unnormalized_vec_derivative: wp.mat33
) -> wp.mat33:
    projection_matrix = wp.identity(n=3, dtype=float) - wp.outer(normalized_vec, normalized_vec)

    # d(normalized_vec)/dx = (1/|unnormalized_vec|) * (I - normalized_vec * normalized_vec^T) * d(unnormalized_vec)/dx
    return (1.0 / unnormalized_vec_length) * projection_matrix * unnormalized_vec_derivative


@wp.func
def compute_angle_derivative(
    n1_hat: wp.vec3,
    n2_hat: wp.vec3,
    e_hat: wp.vec3,
    dn1hat_dx: wp.mat33,
    dn2hat_dx: wp.mat33,
    sin_theta: float,
    cos_theta: float,
    skew_n1: wp.mat33,
    skew_n2: wp.mat33,
) -> wp.vec3:
    dsin_dx = wp.transpose(skew_n1 * dn2hat_dx - skew_n2 * dn1hat_dx) * e_hat
    dcos_dx = wp.transpose(dn1hat_dx) * n2_hat + wp.transpose(dn2hat_dx) * n1_hat

    # dtheta/dx = dsin/dx * cos - dcos/dx * sin
    return dsin_dx * cos_theta - dcos_dx * sin_theta


@wp.func
def evaluate_dihedral_angle_based_bending_force_hessian(
    bending_index: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    dt: float,
):
    # Skip invalid edges (boundary edges with missing opposite vertices)
    if edge_indices[bending_index, 0] == -1 or edge_indices[bending_index, 1] == -1:
        return wp.vec3(0.0), wp.mat33(0.0)

    eps = 1.0e-6

    vi0 = edge_indices[bending_index, 0]
    vi1 = edge_indices[bending_index, 1]
    vi2 = edge_indices[bending_index, 2]
    vi3 = edge_indices[bending_index, 3]

    x0 = pos[vi0]  # opposite 0
    x1 = pos[vi1]  # opposite 1
    x2 = pos[vi2]  # edge start
    x3 = pos[vi3]  # edge end

    # Compute edge vectors
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    # Compute normals
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)

    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    # Early exit for degenerate cases
    if n1_norm < eps or n2_norm < eps or e_norm < eps:
        return wp.vec3(0.0), wp.mat33(0.0)

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    k = stiffness * edge_rest_length[bending_index]
    dE_dtheta = k * (theta - edge_rest_angle[bending_index])

    # Pre-compute skew matrices (shared across all angle derivative computations)
    skew_e = wp.skew(e)
    skew_x03 = wp.skew(x03)
    skew_x02 = wp.skew(x02)
    skew_x13 = wp.skew(x13)
    skew_x12 = wp.skew(x12)
    skew_n1 = wp.skew(n1_hat)
    skew_n2 = wp.skew(n2_hat)

    # Compute the derivatives of unit normals with respect to each vertex; required for computing angle derivatives
    dn1hat_dx0 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_e)
    dn2hat_dx0 = wp.mat33(0.0)

    dn1hat_dx1 = wp.mat33(0.0)
    dn2hat_dx1 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_e)

    dn1hat_dx2 = compute_normalized_vector_derivative(n1_norm, n1_hat, -skew_x03)
    dn2hat_dx2 = compute_normalized_vector_derivative(n2_norm, n2_hat, skew_x13)

    dn1hat_dx3 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_x02)
    dn2hat_dx3 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_x12)

    # Compute all angle derivatives (required for damping)
    dtheta_dx0 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx0, dn2hat_dx0, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx1 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx1, dn2hat_dx1, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx2 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx2, dn2hat_dx2, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx3 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx3, dn2hat_dx3, sin_theta, cos_theta, skew_n1, skew_n2
    )

    # Use float masks for branch-free selection
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)
    mask3 = float(v_order == 3)

    # Select the derivative for the current vertex without branching
    dtheta_dx = dtheta_dx0 * mask0 + dtheta_dx1 * mask1 + dtheta_dx2 * mask2 + dtheta_dx3 * mask3

    # Compute elastic force and hessian
    bending_force = -dE_dtheta * dtheta_dx
    bending_hessian = k * wp.outer(dtheta_dx, dtheta_dx)

    if damping > 0.0:
        inv_dt = 1.0 / dt
        x_prev0 = pos_prev[vi0]
        x_prev1 = pos_prev[vi1]
        x_prev2 = pos_prev[vi2]
        x_prev3 = pos_prev[vi3]

        # Compute displacement vectors
        dx0 = x0 - x_prev0
        dx1 = x1 - x_prev1
        dx2 = x2 - x_prev2
        dx3 = x3 - x_prev3

        # Compute angular velocity using all derivatives
        dtheta_dt = (
            wp.dot(dtheta_dx0, dx0) + wp.dot(dtheta_dx1, dx1) + wp.dot(dtheta_dx2, dx2) + wp.dot(dtheta_dx3, dx3)
        ) * inv_dt

        damping_coeff = damping * k  # damping coefficients following the VBD convention
        damping_force = -damping_coeff * dtheta_dt * dtheta_dx
        damping_hessian = damping_coeff * inv_dt * wp.outer(dtheta_dx, dtheta_dx)

        bending_force = bending_force + damping_force
        bending_hessian = bending_hessian + damping_hessian

    return bending_force, bending_hessian


@wp.func
def assemble_tet_vertex_force_and_hessian(
    dE_dF: vec9,
    H: mat99,
    m1: float,
    m2: float,
    m3: float,
):
    f = wp.vec3(
        -(dE_dF[0] * m1 + dE_dF[3] * m2 + dE_dF[6] * m3),
        -(dE_dF[1] * m1 + dE_dF[4] * m2 + dE_dF[7] * m3),
        -(dE_dF[2] * m1 + dE_dF[5] * m2 + dE_dF[8] * m3),
    )
    h = wp.mat33()

    h[0, 0] += (
        m1 * (H[0, 0] * m1 + H[3, 0] * m2 + H[6, 0] * m3)
        + m2 * (H[0, 3] * m1 + H[3, 3] * m2 + H[6, 3] * m3)
        + m3 * (H[0, 6] * m1 + H[3, 6] * m2 + H[6, 6] * m3)
    )

    h[1, 0] += (
        m1 * (H[1, 0] * m1 + H[4, 0] * m2 + H[7, 0] * m3)
        + m2 * (H[1, 3] * m1 + H[4, 3] * m2 + H[7, 3] * m3)
        + m3 * (H[1, 6] * m1 + H[4, 6] * m2 + H[7, 6] * m3)
    )

    h[2, 0] += (
        m1 * (H[2, 0] * m1 + H[5, 0] * m2 + H[8, 0] * m3)
        + m2 * (H[2, 3] * m1 + H[5, 3] * m2 + H[8, 3] * m3)
        + m3 * (H[2, 6] * m1 + H[5, 6] * m2 + H[8, 6] * m3)
    )

    h[0, 1] += (
        m1 * (H[0, 1] * m1 + H[3, 1] * m2 + H[6, 1] * m3)
        + m2 * (H[0, 4] * m1 + H[3, 4] * m2 + H[6, 4] * m3)
        + m3 * (H[0, 7] * m1 + H[3, 7] * m2 + H[6, 7] * m3)
    )

    h[1, 1] += (
        m1 * (H[1, 1] * m1 + H[4, 1] * m2 + H[7, 1] * m3)
        + m2 * (H[1, 4] * m1 + H[4, 4] * m2 + H[7, 4] * m3)
        + m3 * (H[1, 7] * m1 + H[4, 7] * m2 + H[7, 7] * m3)
    )

    h[2, 1] += (
        m1 * (H[2, 1] * m1 + H[5, 1] * m2 + H[8, 1] * m3)
        + m2 * (H[2, 4] * m1 + H[5, 4] * m2 + H[8, 4] * m3)
        + m3 * (H[2, 7] * m1 + H[5, 7] * m2 + H[8, 7] * m3)
    )

    h[0, 2] += (
        m1 * (H[0, 2] * m1 + H[3, 2] * m2 + H[6, 2] * m3)
        + m2 * (H[0, 5] * m1 + H[3, 5] * m2 + H[6, 5] * m3)
        + m3 * (H[0, 8] * m1 + H[3, 8] * m2 + H[6, 8] * m3)
    )

    h[1, 2] += (
        m1 * (H[1, 2] * m1 + H[4, 2] * m2 + H[7, 2] * m3)
        + m2 * (H[1, 5] * m1 + H[4, 5] * m2 + H[7, 5] * m3)
        + m3 * (H[1, 8] * m1 + H[4, 8] * m2 + H[7, 8] * m3)
    )

    h[2, 2] += (
        m1 * (H[2, 2] * m1 + H[5, 2] * m2 + H[8, 2] * m3)
        + m2 * (H[2, 5] * m1 + H[5, 5] * m2 + H[8, 5] * m3)
        + m3 * (H[2, 8] * m1 + H[5, 8] * m2 + H[8, 8] * m3)
    )

    return f, h


@wp.func
def damp_force_and_hessian(
    particle_pos_prev: wp.vec3,
    particle_pos: wp.vec3,
    force: wp.vec3,
    hessian: wp.mat33,
    damping: float,
    dt: float,
):
    displacement = particle_pos_prev - particle_pos
    h_d = hessian * (damping / dt)
    f_d = h_d * displacement

    return force + f_d, hessian + h_d


@wp.func
def evaluate_volumetric_neo_hooken_force_and_hessian_4_vertices(
    tet_id: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_pose: wp.mat33,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    v0_idx = tet_indices[tet_id, 0]
    v1_idx = tet_indices[tet_id, 1]
    v2_idx = tet_indices[tet_id, 2]
    v3_idx = tet_indices[tet_id, 3]

    v0 = pos[v0_idx]
    v1 = pos[v1_idx]
    v2 = pos[v2_idx]
    v3 = pos[v3_idx]

    Dm_inv = tet_pose
    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    diff_1 = v1 - v0
    diff_2 = v2 - v0
    diff_3 = v3 - v0
    Ds = wp.mat33(
        diff_1[0],
        diff_2[0],
        diff_3[0],
        diff_1[1],
        diff_2[1],
        diff_3[1],
        diff_1[2],
        diff_2[2],
        diff_3[2],
    )

    F = Ds * Dm_inv

    a = 1.0 + mu / lmbd
    det_F = wp.determinant(F)

    F1_1 = F[0, 0]
    F2_1 = F[1, 0]
    F3_1 = F[2, 0]
    F1_2 = F[0, 1]
    F2_2 = F[1, 1]
    F3_2 = F[2, 1]
    F1_3 = F[0, 2]
    F2_3 = F[1, 2]
    F3_3 = F[2, 2]

    dPhi_D_dF = vec9(
        F1_1,
        F2_1,
        F3_1,
        F1_2,
        F2_2,
        F3_2,
        F1_3,
        F2_3,
        F3_3,
    )

    ddetF_dF = vec9(
        F2_2 * F3_3 - F2_3 * F3_2,
        F1_3 * F3_2 - F1_2 * F3_3,
        F1_2 * F2_3 - F1_3 * F2_2,
        F2_3 * F3_1 - F2_1 * F3_3,
        F1_1 * F3_3 - F1_3 * F3_1,
        F1_3 * F2_1 - F1_1 * F2_3,
        F2_1 * F3_2 - F2_2 * F3_1,
        F1_2 * F3_1 - F1_1 * F3_2,
        F1_1 * F2_2 - F1_2 * F2_1,
    )

    d2E_dF_dF = wp.outer(ddetF_dF, ddetF_dF)
    k = det_F - a
    d2E_dF_dF[0, 4] += k * F3_3
    d2E_dF_dF[4, 0] += k * F3_3
    d2E_dF_dF[0, 5] += k * -F2_3
    d2E_dF_dF[5, 0] += k * -F2_3
    d2E_dF_dF[0, 7] += k * -F3_2
    d2E_dF_dF[7, 0] += k * -F3_2
    d2E_dF_dF[0, 8] += k * F2_2
    d2E_dF_dF[8, 0] += k * F2_2

    d2E_dF_dF[1, 3] += k * -F3_3
    d2E_dF_dF[3, 1] += k * -F3_3
    d2E_dF_dF[1, 5] += k * F1_3
    d2E_dF_dF[5, 1] += k * F1_3
    d2E_dF_dF[1, 6] += k * F3_2
    d2E_dF_dF[6, 1] += k * F3_2
    d2E_dF_dF[1, 8] += k * -F1_2
    d2E_dF_dF[8, 1] += k * -F1_2

    d2E_dF_dF[2, 3] += k * F2_3
    d2E_dF_dF[3, 2] += k * F2_3
    d2E_dF_dF[2, 4] += k * -F1_3
    d2E_dF_dF[4, 2] += k * -F1_3
    d2E_dF_dF[2, 6] += k * -F2_2
    d2E_dF_dF[6, 2] += k * -F2_2
    d2E_dF_dF[2, 7] += k * F1_2
    d2E_dF_dF[7, 2] += k * F1_2

    d2E_dF_dF[3, 7] += k * F3_1
    d2E_dF_dF[7, 3] += k * F3_1
    d2E_dF_dF[3, 8] += k * -F2_1
    d2E_dF_dF[8, 3] += k * -F2_1

    d2E_dF_dF[4, 6] += k * -F3_1
    d2E_dF_dF[6, 4] += k * -F3_1
    d2E_dF_dF[4, 8] += k * F1_1
    d2E_dF_dF[8, 4] += k * F1_1

    d2E_dF_dF[5, 6] += k * F2_1
    d2E_dF_dF[6, 5] += k * F2_1
    d2E_dF_dF[5, 7] += k * -F1_1
    d2E_dF_dF[7, 5] += k * -F1_1

    d2E_dF_dF = d2E_dF_dF * lmbd

    d2E_dF_dF[0, 0] += mu
    d2E_dF_dF[1, 1] += mu
    d2E_dF_dF[2, 2] += mu
    d2E_dF_dF[3, 3] += mu
    d2E_dF_dF[4, 4] += mu
    d2E_dF_dF[5, 5] += mu
    d2E_dF_dF[6, 6] += mu
    d2E_dF_dF[7, 7] += mu
    d2E_dF_dF[8, 8] += mu

    d2E_dF_dF = d2E_dF_dF * rest_volume

    dPhi_D_dF = dPhi_D_dF * mu
    dPhi_H_dF = ddetF_dF * lmbd * k

    dE_dF = (dPhi_D_dF + dPhi_H_dF) * rest_volume

    Dm_inv_1_1 = Dm_inv[0, 0]
    Dm_inv_2_1 = Dm_inv[1, 0]
    Dm_inv_3_1 = Dm_inv[2, 0]
    Dm_inv_1_2 = Dm_inv[0, 1]
    Dm_inv_2_2 = Dm_inv[1, 1]
    Dm_inv_3_2 = Dm_inv[2, 1]
    Dm_inv_1_3 = Dm_inv[0, 2]
    Dm_inv_2_3 = Dm_inv[1, 2]
    Dm_inv_3_3 = Dm_inv[2, 2]

    ms = mat43(
        -Dm_inv_1_1 - Dm_inv_2_1 - Dm_inv_3_1,
        -Dm_inv_1_2 - Dm_inv_2_2 - Dm_inv_3_2,
        -Dm_inv_1_3 - Dm_inv_2_3 - Dm_inv_3_3,
        Dm_inv_1_1,
        Dm_inv_1_2,
        Dm_inv_1_3,
        Dm_inv_2_1,
        Dm_inv_2_2,
        Dm_inv_2_3,
        Dm_inv_3_1,
        Dm_inv_3_2,
        Dm_inv_3_3,
    )

    f1, h1 = assemble_tet_vertex_force_and_hessian(dE_dF, d2E_dF_dF, ms[0, 0], ms[0, 1], ms[0, 2])
    f1, h1 = damp_force_and_hessian(pos_prev[v0_idx], v0, f1, h1, damping, dt)
    f2, h2 = assemble_tet_vertex_force_and_hessian(dE_dF, d2E_dF_dF, ms[1, 0], ms[1, 1], ms[1, 2])
    f2, h2 = damp_force_and_hessian(pos_prev[v1_idx], v1, f2, h2, damping, dt)
    f3, h3 = assemble_tet_vertex_force_and_hessian(dE_dF, d2E_dF_dF, ms[2, 0], ms[2, 1], ms[2, 2])
    f3, h3 = damp_force_and_hessian(pos_prev[v2_idx], v2, f3, h3, damping, dt)
    f4, h4 = assemble_tet_vertex_force_and_hessian(dE_dF, d2E_dF_dF, ms[3, 0], ms[3, 1], ms[3, 2])
    f4, h4 = damp_force_and_hessian(pos_prev[v3_idx], v3, f4, h4, damping, dt)

    return f1, f2, f3, f4, h1, h2, h3, h4


@wp.func
def evaluate_volumetric_neo_hooken_force_and_hessian(
    tet_id: int,
    v_order: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_pose: wp.mat33,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    v0_idx = tet_indices[tet_id, 0]
    v1_idx = tet_indices[tet_id, 1]
    v2_idx = tet_indices[tet_id, 2]
    v3_idx = tet_indices[tet_id, 3]

    v0 = pos[v0_idx]
    v1 = pos[v1_idx]
    v2 = pos[v2_idx]
    v3 = pos[v3_idx]

    Dm_inv = tet_pose
    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    diff_1 = v1 - v0
    diff_2 = v2 - v0
    diff_3 = v3 - v0
    Ds = wp.mat33(
        diff_1[0],
        diff_2[0],
        diff_3[0],
        diff_1[1],
        diff_2[1],
        diff_3[1],
        diff_1[2],
        diff_2[2],
        diff_3[2],
    )

    F = Ds * Dm_inv

    a = 1.0 + mu / lmbd
    det_F = wp.determinant(F)

    F1_1 = F[0, 0]
    F2_1 = F[1, 0]
    F3_1 = F[2, 0]
    F1_2 = F[0, 1]
    F2_2 = F[1, 1]
    F3_2 = F[2, 1]
    F1_3 = F[0, 2]
    F2_3 = F[1, 2]
    F3_3 = F[2, 2]

    dPhi_D_dF = vec9(
        F1_1,
        F2_1,
        F3_1,
        F1_2,
        F2_2,
        F3_2,
        F1_3,
        F2_3,
        F3_3,
    )

    ddetF_dF = vec9(
        F2_2 * F3_3 - F2_3 * F3_2,
        F1_3 * F3_2 - F1_2 * F3_3,
        F1_2 * F2_3 - F1_3 * F2_2,
        F2_3 * F3_1 - F2_1 * F3_3,
        F1_1 * F3_3 - F1_3 * F3_1,
        F1_3 * F2_1 - F1_1 * F2_3,
        F2_1 * F3_2 - F2_2 * F3_1,
        F1_2 * F3_1 - F1_1 * F3_2,
        F1_1 * F2_2 - F1_2 * F2_1,
    )

    d2E_dF_dF = wp.outer(ddetF_dF, ddetF_dF)
    k = det_F - a
    d2E_dF_dF[0, 4] += k * F3_3
    d2E_dF_dF[4, 0] += k * F3_3
    d2E_dF_dF[0, 5] += k * -F2_3
    d2E_dF_dF[5, 0] += k * -F2_3
    d2E_dF_dF[0, 7] += k * -F3_2
    d2E_dF_dF[7, 0] += k * -F3_2
    d2E_dF_dF[0, 8] += k * F2_2
    d2E_dF_dF[8, 0] += k * F2_2

    d2E_dF_dF[1, 3] += k * -F3_3
    d2E_dF_dF[3, 1] += k * -F3_3
    d2E_dF_dF[1, 5] += k * F1_3
    d2E_dF_dF[5, 1] += k * F1_3
    d2E_dF_dF[1, 6] += k * F3_2
    d2E_dF_dF[6, 1] += k * F3_2
    d2E_dF_dF[1, 8] += k * -F1_2
    d2E_dF_dF[8, 1] += k * -F1_2

    d2E_dF_dF[2, 3] += k * F2_3
    d2E_dF_dF[3, 2] += k * F2_3
    d2E_dF_dF[2, 4] += k * -F1_3
    d2E_dF_dF[4, 2] += k * -F1_3
    d2E_dF_dF[2, 6] += k * -F2_2
    d2E_dF_dF[6, 2] += k * -F2_2
    d2E_dF_dF[2, 7] += k * F1_2
    d2E_dF_dF[7, 2] += k * F1_2

    d2E_dF_dF[3, 7] += k * F3_1
    d2E_dF_dF[7, 3] += k * F3_1
    d2E_dF_dF[3, 8] += k * -F2_1
    d2E_dF_dF[8, 3] += k * -F2_1

    d2E_dF_dF[4, 6] += k * -F3_1
    d2E_dF_dF[6, 4] += k * -F3_1
    d2E_dF_dF[4, 8] += k * F1_1
    d2E_dF_dF[8, 4] += k * F1_1

    d2E_dF_dF[5, 6] += k * F2_1
    d2E_dF_dF[6, 5] += k * F2_1
    d2E_dF_dF[5, 7] += k * -F1_1
    d2E_dF_dF[7, 5] += k * -F1_1

    d2E_dF_dF = d2E_dF_dF * lmbd

    d2E_dF_dF[0, 0] += mu
    d2E_dF_dF[1, 1] += mu
    d2E_dF_dF[2, 2] += mu
    d2E_dF_dF[3, 3] += mu
    d2E_dF_dF[4, 4] += mu
    d2E_dF_dF[5, 5] += mu
    d2E_dF_dF[6, 6] += mu
    d2E_dF_dF[7, 7] += mu
    d2E_dF_dF[8, 8] += mu

    d2E_dF_dF = d2E_dF_dF * rest_volume

    dPhi_D_dF = dPhi_D_dF * mu
    dPhi_H_dF = ddetF_dF * lmbd * k

    dE_dF = (dPhi_D_dF + dPhi_H_dF) * rest_volume

    Dm_inv_1_1 = Dm_inv[0, 0]
    Dm_inv_2_1 = Dm_inv[1, 0]
    Dm_inv_3_1 = Dm_inv[2, 0]
    Dm_inv_1_2 = Dm_inv[0, 1]
    Dm_inv_2_2 = Dm_inv[1, 1]
    Dm_inv_3_2 = Dm_inv[2, 1]
    Dm_inv_1_3 = Dm_inv[0, 2]
    Dm_inv_2_3 = Dm_inv[1, 2]
    Dm_inv_3_3 = Dm_inv[2, 2]

    ms = mat43(
        -Dm_inv_1_1 - Dm_inv_2_1 - Dm_inv_3_1,
        -Dm_inv_1_2 - Dm_inv_2_2 - Dm_inv_3_2,
        -Dm_inv_1_3 - Dm_inv_2_3 - Dm_inv_3_3,
        Dm_inv_1_1,
        Dm_inv_1_2,
        Dm_inv_1_3,
        Dm_inv_2_1,
        Dm_inv_2_2,
        Dm_inv_2_3,
        Dm_inv_3_1,
        Dm_inv_3_2,
        Dm_inv_3_3,
    )

    f, h = assemble_tet_vertex_force_and_hessian(dE_dF, d2E_dF_dF, ms[v_order, 0], ms[v_order, 1], ms[v_order, 2])
    f, h = damp_force_and_hessian(pos_prev[v0_idx], v0, f, h, damping, dt)

    return f, h


@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    dt: float,
):
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])

    n = contact_normal[contact_index]

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if penetration_depth > 0:
        body_contact_force_norm = penetration_depth * soft_contact_ke
        body_contact_force = n * body_contact_force_norm
        body_contact_hessian = soft_contact_ke * wp.outer(n, n)

        mu = shape_material_mu[shape_index]

        dx = particle_pos - particle_prev_pos

        damping_hessian = (soft_contact_kd / dt) * body_contact_hessian
        body_contact_hessian = body_contact_hessian + damping_hessian
        body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        if body_q_prev:
            # if body_q_prev is available, compute velocity using finite difference method
            # this is more accurate for simulating static friction
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        else:
            # otherwise use the instantaneous velocity
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]

            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)

            # compute the body velocity at the particle position
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # friction
        e0, e1 = build_orthonormal_basis(n)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * relative_translation
        eps_u = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(mu, body_contact_force_norm, T, u, eps_u)
        body_contact_force = body_contact_force + friction_force
        body_contact_hessian = body_contact_hessian + friction_hessian
    else:
        body_contact_force = wp.vec3(0.0, 0.0, 0.0)
        body_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return body_contact_force, body_contact_hessian


@wp.func
def evaluate_self_contact_force_norm(dis: float, collision_radius: float, k: float):
    # Adjust distance and calculate penetration depth

    penetration_depth = collision_radius - dis

    # Initialize outputs
    dEdD = wp.float32(0.0)
    d2E_dDdD = wp.float32(0.0)

    # C2 continuity calculation
    tau = collision_radius * 0.5
    if tau > dis > 1e-5:
        k2 = 0.5 * tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.func
def damp_collision(
    displacement: wp.vec3,
    collision_normal: wp.vec3,
    collision_hessian: wp.mat33,
    collision_damping: float,
    dt: float,
):
    if wp.dot(displacement, collision_normal) > 0:
        damping_hessian = (collision_damping / dt) * collision_hessian
        damping_force = damping_hessian * displacement
        return damping_force, damping_hessian
    else:
        return wp.vec3(0.0), wp.mat33(0.0)


@wp.func
def evaluate_edge_edge_contact(
    v: int,
    v_order: int,
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        pos_prev,
        edge_indices
        collision_radius
        collision_stiffness
        dt
        edge_edge_parallel_epsilon: threshold to determine whether 2 edges are parallel
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_prev[e1_v1] + (pos_prev[e1_v2] - pos_prev[e1_v1]) * s
        c2_prev = pos_prev[e2_v1] + (pos_prev[e2_v2] - pos_prev[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)
        friction_force = friction_force * v_bary
        friction_hessian = friction_hessian * v_bary * v_bary

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        if v_order == 0:
            displacement = pos_prev[e1_v1] - e1_v1_pos
        elif v_order == 1:
            displacement = pos_prev[e1_v2] - e1_v2_pos
        elif v_order == 2:
            displacement = pos_prev[e2_v1] - e2_v1_pos
        else:
            displacement = pos_prev[e2_v2] - e2_v2_pos

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + friction_force
        collision_hessian = collision_hessian + friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_edge_edge_contact_2_vertices(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        edge_indices
        collision_radius
        collision_stiffness
        dt
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_prev[e1_v1] + (pos_prev[e1_v2] - pos_prev[e1_v1]) * s
        c2_prev = pos_prev[e2_v1] + (pos_prev[e2_v2] - pos_prev[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        displacement_0 = pos_prev[e1_v1] - e1_v1_pos
        displacement_1 = pos_prev[e1_v2] - e1_v2_pos

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        return True, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return False, collision_force, collision_force, collision_hessian, collision_hessian


@wp.func
def evaluate_vertex_triangle_collision_force_hessian(
    v: int,
    v_order: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_prev[v]

        closest_p_prev = (
            bary[0] * pos_prev[tri_indices[tri, 0]]
            + bary[1] * pos_prev[tri_indices[tri, 1]]
            + bary[2] * pos_prev[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1], friction_force[2],
            )
        # fmt: on

        if v_order == 0:
            displacement = pos_prev[tri_indices[tri, 0]] - a
        elif v_order == 1:
            displacement = pos_prev[tri_indices[tri, 1]] - b
        elif v_order == 2:
            displacement = pos_prev[tri_indices[tri, 2]] - c
        else:
            displacement = pos_prev[v] - p

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + v_bary * friction_force
        collision_hessian = collision_hessian + v_bary * v_bary * friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_prev[v]

        closest_p_prev = (
            bary[0] * pos_prev[tri_indices[tri, 0]]
            + bary[1] * pos_prev[tri_indices[tri, 1]]
            + bary[2] * pos_prev[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1],
                friction_force[2],
            )
        # fmt: on

        displacement_0 = pos_prev[tri_indices[tri, 0]] - a
        displacement_1 = pos_prev[tri_indices[tri, 1]] - b
        displacement_2 = pos_prev[tri_indices[tri, 2]] - c
        displacement_3 = pos_prev[v] - p

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]
        collision_force_2 = collision_force * bs[2]
        collision_force_3 = collision_force * bs[3]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]
        collision_hessian_2 = collision_hessian * bs[2] * bs[2]
        collision_hessian_3 = collision_hessian * bs[3] * bs[3]

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_2,
            collision_normal * collision_normal_sign[2],
            collision_hessian_2,
            collision_damping,
            dt,
        )
        collision_force_2 += damping_force + bs[2] * friction_force
        collision_hessian_2 += damping_hessian + bs[2] * bs[2] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_3,
            collision_normal * collision_normal_sign[3],
            collision_hessian_3,
            collision_damping,
            dt,
        )
        collision_force_3 += damping_force + bs[3] * friction_force
        collision_hessian_3 += damping_hessian + bs[3] * bs[3] * friction_hessian
        return (
            True,
            collision_force_0,
            collision_force_1,
            collision_force_2,
            collision_force_3,
            collision_hessian_0,
            collision_hessian_1,
            collision_hessian_2,
            collision_hessian_3,
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            False,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
        )


@wp.func
def compute_friction(mu: float, normal_contact_force: float, T: mat32, u: wp.vec2, eps_u: float):
    """
    Returns the 1D friction force and hessian.
    Args:
        mu: Friction coefficient.
        normal_contact_force: normal contact force.
        T: Transformation matrix (3x2 matrix).
        u: 2D displacement vector.
    """
    # Friction
    u_norm = wp.length(u)

    if u_norm > 0.0:
        # IPC friction
        if u_norm > eps_u:
            # constant stage
            f1_SF_over_x = 1.0 / u_norm
        else:
            # smooth transition
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        force = -mu * normal_contact_force * T * (f1_SF_over_x * u)

        # Different from IPC, we treat the contact normal as constant
        # this significantly improves the stability
        hessian = mu * normal_contact_force * T * (f1_SF_over_x * wp.identity(2, float)) * wp.transpose(T)
    else:
        force = wp.vec3(0.0, 0.0, 0.0)
        hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return force, hessian


@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    inertia_out: wp.array(dtype=wp.vec3),
    displacements_out: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    pos_prev[particle] = pos[particle]
    if not particle_flags[particle] & ParticleFlags.ACTIVE:
        inertia_out[particle] = pos_prev[particle]
        return
    vel_new = vel[particle] + (gravity[0] + external_force[particle] * inv_mass[particle]) * dt
    inertia = pos[particle] + vel_new * dt
    inertia_out[particle] = inertia
    if displacements_out:
        displacements_out[particle] = vel_new * dt


@wp.kernel
def calculate_vertex_collision_buffer(
    adjacency: ForceElementAdjacencyInfo,
    collision_info: TriMeshCollisionInfo,
    projection_buffer_sizes: wp.array(dtype=wp.int32),
):
    particle_index = wp.tid()
    size_buffer = wp.int32(0)

    size_buffer += collision_info.vertex_colliding_triangles_buffer_sizes[particle_index]

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, _ = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        size_buffer += collision_info.triangle_colliding_vertices_buffer_sizes[tri_index]

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, _ = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)

        size_buffer += collision_info.edge_colliding_edges_buffer_sizes[nei_edge_index]
    projection_buffer_sizes[particle_index] = size_buffer


@wp.func
def segment_plane_intersects(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps_parallel: float,  # e.g., 1e-8
    eps_intersect_near: float,  # e.g., 1e-8
    eps_intersect_far: float,  # e.g., 1e-8
    coplanar_counts: bool,  # True if you want a coplanar segment to count as "hit"
) -> bool:
    # Plane eq: n·(p - d) = 0
    # Segment: p(t) = v + t * delta_v,  t in [0, 1]
    nv = wp.dot(n, delta_v)
    num = -wp.dot(n, v - d)

    # Parallel (or nearly): either coplanar or no hit
    if wp.abs(nv) < eps_parallel:
        return coplanar_counts and (wp.abs(num) < eps_parallel)

    t = num / nv
    # consider tiny tolerance at ends
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
    closest_p, _bary, _feature_type = triangle_closest_point(t1, t2, t3, v)

    n_hat = v - closest_p

    if wp.length(n_hat) < 1e-12:
        return wp.vector(False, False, False, False, length=4, dtype=wp.bool), wp.vec3(0.0), v

    n = wp.normalize(n_hat)

    delta_v_n = wp.max(-wp.dot(n, delta_v), 0.0)
    delta_t_n = wp.max(
        wp.vec4(
            wp.dot(n, delta_t1),
            wp.dot(n, delta_t2),
            wp.dot(n, delta_t3),
            0.0,
        )
    )

    if delta_t_n + delta_v_n == 0.0:
        d = closest_p + 0.5 * n_hat
    else:
        lmbd = delta_t_n / (delta_t_n + delta_v_n)
        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        # wp.printf("lambda: %f\n", lmbd)
        d = closest_p + lmbd * n_hat

    if delta_v_n == 0.0:
        is_dummy_for_v = True
    else:
        is_dummy_for_v = not segment_plane_intersects(v, delta_v, n, d, 1e-6, -1e-8, 1e-8, False)

    if delta_t_n == 0.0:
        is_dummy_for_t_1 = True
        is_dummy_for_t_2 = True
        is_dummy_for_t_3 = True
    else:
        is_dummy_for_t_1 = not segment_plane_intersects(t1, delta_t1, n, d, 1e-6, -1e-8, 1e-8, False)
        is_dummy_for_t_2 = not segment_plane_intersects(t2, delta_t2, n, d, 1e-6, -1e-8, 1e-8, False)
        is_dummy_for_t_3 = not segment_plane_intersects(t3, delta_t3, n, d, 1e-6, -1e-8, 1e-8, False)

    return (
        wp.vector(is_dummy_for_v, is_dummy_for_t_1, is_dummy_for_t_2, is_dummy_for_t_3, length=4, dtype=wp.bool),
        n,
        d,
    )


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
        return (
            wp.vector(False, False, False, False, length=4, dtype=wp.bool),
            robust_edge_pair_normal(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos),
            c1 * 0.5 + c2 * 0.5,
        )

    n = wp.normalize(n_hat)

    delta_e0 = wp.max(
        wp.vec3(
            -wp.dot(n, delta_e0_v0),
            -wp.dot(n, delta_e0_v1),
            0.0,
        )
    )
    delta_e1 = wp.max(
        wp.vec3(
            wp.dot(n, delta_e1_v0),
            wp.dot(n, delta_e1_v1),
            0.0,
        )
    )

    if delta_e0 + delta_e1 == 0.0:
        d = c2 + 0.5 * n_hat
    else:
        lmbd = delta_e1 / (delta_e1 + delta_e0)

        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        # wp.printf("lambda: %f\n", lmbd)
        d = c2 + lmbd * n_hat

    if delta_e0 == 0.0:
        is_dummy_for_e0_v0 = True
        is_dummy_for_e0_v1 = True
    else:
        is_dummy_for_e0_v0 = not segment_plane_intersects(e0_v0_pos, delta_e0_v0, n, d, 1e-6, -1e-8, 1e-6, False)
        is_dummy_for_e0_v1 = not segment_plane_intersects(e0_v1_pos, delta_e0_v1, n, d, 1e-6, -1e-8, 1e-6, False)

    if delta_e1 == 0.0:
        is_dummy_for_e1_v0 = True
        is_dummy_for_e1_v1 = True
    else:
        is_dummy_for_e1_v0 = not segment_plane_intersects(e1_v0_pos, delta_e1_v0, n, d, 1e-6, -1e-8, 1e-6, False)
        is_dummy_for_e1_v1 = not segment_plane_intersects(e1_v1_pos, delta_e1_v1, n, d, 1e-6, -1e-8, 1e-6, False)

    return (
        wp.vector(
            is_dummy_for_e0_v0, is_dummy_for_e0_v1, is_dummy_for_e1_v0, is_dummy_for_e1_v1, length=4, dtype=wp.bool
        ),
        n,
        d,
    )


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
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)

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


@wp.func
def planar_truncation(
    v: wp.vec3, delta_v: wp.vec3, n: wp.vec3, d: wp.vec3, eps: float, gamma_r: float, gamma_min: float = 1e-3
):
    nv = wp.dot(n, delta_v)
    num = wp.dot(n, d - v)

    # Parallel (or nearly): do not truncate
    if wp.abs(nv) < eps:
        return delta_v

    t = num / nv

    t = wp.max(wp.min(t * gamma_r, t - gamma_min), 0.0)
    if t >= 1:
        return delta_v
    else:
        return t * delta_v


@wp.func
def planar_truncation_t(
    v: wp.vec3, delta_v: wp.vec3, n: wp.vec3, d: wp.vec3, eps: float, gamma_r: float, gamma_min: float = 1e-3
):
    denom = wp.dot(n, delta_v)

    # Parallel (or nearly parallel) → no intersection
    if wp.abs(denom) < eps:
        return 1.0

    # Solve: dot(n, v + t*delta_v - d) = 0
    t = wp.dot(n, d - v) / denom

    if t < 0:
        return 1.0

    t = wp.clamp(wp.min(t * gamma_r, t - gamma_min), 0.0, 1.0)
    return t



@wp.func
def hessian_weighted_projection_onto_halfspace(
    x: wp.vec3, n: wp.vec3, p: wp.vec3, H_inv: wp.mat33, eps: float = 1e-8
):
    """
    Project x onto half-space {y : n^T (y - p) >= 0} using the hessian-weighted metric H.
    
    Args:
        x: Point to project
        n: Normal vector of the half-space (should be normalized)
        p: Point on the half-space boundary
        H_inv: Inverse of the hessian matrix H
        eps: Small epsilon for numerical stability
    
    Returns:
        Projected point
    """
    # Check if x is in the half-space: n^T (x - p) >= 0
    n_dot_x_minus_p = wp.dot(n, x - p)
    
    if n_dot_x_minus_p >= -eps:
        # x is already in the half-space
        return x
    
    # Compute projection: x - H^{-1} n * (n^T (x - p)) / (n^T H^{-1} n)
    H_inv_n = H_inv * n
    n_dot_H_inv_n = wp.dot(n, H_inv_n)
    
    if wp.abs(n_dot_H_inv_n) < eps:
        # Degenerate case: n is in null space of H
        return x
    
    lambda_proj = n_dot_x_minus_p / n_dot_H_inv_n
    return x - H_inv_n * lambda_proj

@wp.kernel
def apply_planar_truncation(
    pos: wp.array(dtype=wp.vec3),
    displacement_in: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    collision_info_arr: wp.array(dtype=TriMeshCollisionInfo),
    parallel_eps: float,
    gamma: float,
    max_displacement: float,
    displacements_out: wp.array(dtype=wp.vec3),
    pos_out: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()
    particle_pos = pos[particle_index]
    particle_displacement = displacement_in[particle_index]
    collision_info = collision_info_arr[0]

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
                particle_displacement = planar_truncation(
                    particle_pos, particle_displacement, n, d, parallel_eps, gamma
                )

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)

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
                    particle_displacement = planar_truncation(
                        particle_pos, particle_displacement, n, d, parallel_eps, gamma
                    )

    len_displacement = wp.length(particle_displacement)
    if len_displacement > max_displacement:
        particle_displacement = particle_displacement * max_displacement / len_displacement

    displacements_out[particle_index] = particle_displacement
    if pos_out:
        pos_out[particle_index] = pos[particle_index] + particle_displacement


@wp.kernel
def apply_planar_truncation_tile(
    pos: wp.array(dtype=wp.vec3),
    displacement_in: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    parallel_eps: float,
    gamma: float,
    max_displacement: float,
    displacements_out: wp.array(dtype=wp.vec3),
    pos_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_SELF_CONTACT_SOLVE
    thread_idx = tid % TILE_SIZE_SELF_CONTACT_SOLVE
    particle_index = block_idx

    particle_pos = pos[particle_index]
    particle_displacement = displacement_in[particle_index]
    collision_info = collision_info_array[0]

    t = float(1.0)

    num_colliding_tris = get_vertex_colliding_triangles_count(collision_info, particle_index)
    # dont need to evaluate size
    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_colliding_tris:
        colliding_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_SELF_CONTACT_SOLVE
        # elastic force and hessian
        colliding_tri_index = get_vertex_colliding_triangles(collision_info, particle_index, colliding_tri_counter)

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

        if not is_dummy[0]:
            t = wp.min(planar_truncation_t(particle_pos, particle_displacement, n, d, parallel_eps, gamma), t)

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        num_tri_colliding_vertices = get_triangle_colliding_vertices_count(collision_info, tri_index)

        t1 = pos[tri_indices[tri_index, 0]]
        t2 = pos[tri_indices[tri_index, 1]]
        t3 = pos[tri_indices[tri_index, 2]]

        delta_t1 = displacement_in[tri_indices[tri_index, 0]]
        delta_t2 = displacement_in[tri_indices[tri_index, 1]]
        delta_t3 = displacement_in[tri_indices[tri_index, 2]]

        batch_counter = wp.int32(0)
        while batch_counter + thread_idx < num_tri_colliding_vertices:
            colliding_vertex_counter = thread_idx + batch_counter
            batch_counter += TILE_SIZE_SELF_CONTACT_SOLVE
            colliding_vertex = get_triangle_colliding_vertices(collision_info, tri_index, colliding_vertex_counter)

            colliding_particle_pos = pos[colliding_vertex]
            colliding_particle_displacement = displacement_in[colliding_vertex]

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

            if not is_dummy[vertex_order + 1]:
                t = wp.min(planar_truncation_t(particle_pos, particle_displacement, n, d, parallel_eps, gamma), t)

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)

        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            batch_counter = wp.int32(0)
            num_e_colliding_e = get_edge_colliding_edges_count(collision_info, nei_edge_index)
            while batch_counter + thread_idx < num_e_colliding_e:
                colliding_edge_counter = thread_idx + batch_counter
                batch_counter += TILE_SIZE_SELF_CONTACT_SOLVE
                colliding_edge = get_edge_colliding_edges(collision_info, nei_edge_index, colliding_edge_counter)

                e1_v1 = edge_indices[nei_edge_index, 2]
                e1_v2 = edge_indices[nei_edge_index, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[colliding_edge, 2]
                e2_v2 = edge_indices[colliding_edge, 3]

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

                if not is_dummy[vertex_order_on_edge_2_v1_v2_only]:
                    t = wp.min(planar_truncation_t(particle_pos, particle_displacement, n, d, parallel_eps, gamma), t)

    t_tile = wp.tile(t)
    t_min = wp.tile_reduce(wp.min, t_tile)[0]

    if thread_idx == 0:
        particle_displacement = particle_displacement * t_min

        len_displacement = wp.length(particle_displacement)
        if len_displacement > max_displacement:
            particle_displacement = particle_displacement * max_displacement / len_displacement

        displacements_out[particle_index] = particle_displacement
        if pos_out:
            pos_out[particle_index] = pos[particle_index] + particle_displacement


@wp.kernel
def compute_particle_conservative_bound(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ForceElementAdjacencyInfo,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_index = wp.tid()
    min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])

    # bound from neighbor triangles
    for i_adj_tri in range(
        get_vertex_num_adjacent_faces(
            adjacency,
            particle_index,
        )
    ):
        tri_index, _vertex_order = get_vertex_adjacent_face_id_order(
            adjacency,
            particle_index,
            i_adj_tri,
        )
        min_dist = wp.min(min_dist, collision_info.triangle_colliding_vertices_min_dist[tri_index])

    # bound from neighbor edges
    for i_adj_edge in range(
        get_vertex_num_adjacent_edges(
            adjacency,
            particle_index,
        )
    ):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency,
            particle_index,
            i_adj_edge,
        )
        # vertex is on the edge; otherwise it only effects the bending energy
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor edges
            min_dist = wp.min(min_dist, collision_info.edge_colliding_edges_min_dist[nei_edge_index])

    particle_conservative_bounds[particle_index] = conservative_bound_relaxation * min_dist


@wp.kernel
def validate_conservative_bound(
    pos: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    v_index = wp.tid()

    displacement = wp.length(pos[v_index] - pos_prev_collision_detection[v_index])

    if displacement > particle_conservative_bounds[v_index] * 1.01 and displacement > 1e-5:
        # wp.expect_eq(displacement <= particle_conservative_bounds[v_index] * 1.01, True)
        wp.printf(
            "Vertex %d has moved by %f exceeded the limit of %f\n",
            v_index,
            displacement,
            particle_conservative_bounds[v_index],
        )


@wp.kernel
def apply_conservative_bound_truncation(
    particle_displacements: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    particle_q_out: wp.array(dtype=wp.vec3),
):
    particle_idx = wp.tid()

    particle_pos_prev_collision_detection = pos_prev_collision_detection[particle_idx]
    accumulated_displacement = particle_displacements[particle_idx]
    conservative_bound = particle_conservative_bounds[particle_idx]

    accumulated_displacement_norm = wp.length(accumulated_displacement)
    if accumulated_displacement_norm > conservative_bound and conservative_bound > 1e-6:
        accumulated_displacement = accumulated_displacement * (conservative_bound / accumulated_displacement_norm)
    particle_displacements[particle_idx] = accumulated_displacement
    particle_q_out[particle_idx] = particle_pos_prev_collision_detection + accumulated_displacement


@wp.kernel
def solve_trimesh_no_self_contact_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # # inertia force and hessian
    # f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    # h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    if tri_indices:
        num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)
        batch_counter = wp.int32(0)
        # loop through all the adjacent triangles using whole block
        while batch_counter + thread_idx < num_adj_faces:
            adj_tri_counter = thread_idx + batch_counter
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            # elastic force and hessian
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

            f_tri, h_tri = evaluate_stvk_force_hessian(
                tri_index,
                vertex_order,
                pos,
                pos_prev,
                tri_indices,
                tri_poses[tri_index],
                tri_areas[tri_index],
                tri_materials[tri_index, 0],
                tri_materials[tri_index, 1],
                tri_materials[tri_index, 2],
                dt,
            )
            # compute damping

            f += f_tri
            h += h_tri

            # fmt: off
            if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                    particle_index,
                    thread_idx,
                    vertex_order,
                    f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
                )
                # fmt: on

    if edge_indices:
        batch_counter = wp.int32(0)
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_edges:
            adj_edge_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
                adjacency, particle_index, adj_edge_counter
            )
            if edge_bending_properties[nei_edge_index, 0] != 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                    nei_edge_index,
                    vertex_order_on_edge,
                    pos,
                    pos_prev,
                    edge_indices,
                    edge_rest_angles,
                    edge_rest_length,
                    edge_bending_properties[nei_edge_index, 0],
                    edge_bending_properties[nei_edge_index, 1],
                    dt,
                )

                f += f_edge
                h += h_edge
    if tet_indices:
        # solve tet elasticity
        batch_counter = wp.int32(0)
        num_adj_tets = get_vertex_num_adjacent_tets(adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_tets:
            adj_tet_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                adjacency, particle_index, adj_tet_counter
            )

            f_tet, h_tet = evaluate_volumetric_neo_hooken_force_and_hessian(
                nei_tet_index,
                vertex_order_on_tet,
                pos_prev,
                pos,
                tet_indices,
                tet_poses[nei_tet_index],
                tet_materials[nei_tet_index, 0],
                tet_materials[nei_tet_index, 1],
                tet_materials[nei_tet_index, 2],
                dt,
            )

            f += f_tet
            h += h_tet

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-5:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )

            pos_new[particle_index] = particle_pos + h_inv * f_total


@wp.kernel
def solve_trimesh_no_self_contact(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    particle_index = particle_ids_in_color[tid]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    if tri_indices:
        # elastic force and hessian
        for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
            tri_id, particle_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

            # fmt: off
            if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d | num_adj_faces: %d | ",
                    particle_index,
                    get_vertex_num_adjacent_faces(particle_index, adjacency),
                )
                wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, particle_order)
                wp.printf(
                    "face: %d %d %d\n",
                    tri_indices[tri_id, 0],
                    tri_indices[tri_id, 1],
                    tri_indices[tri_id, 2],
                )
            # fmt: on

            f_tri, h_tri = evaluate_stvk_force_hessian(
                tri_id,
                particle_order,
                pos,
                pos_prev,
                tri_indices,
                tri_poses[tri_id],
                tri_areas[tri_id],
                tri_materials[tri_id, 0],
                tri_materials[tri_id, 1],
                tri_materials[tri_id, 2],
                dt,
            )

            f = f + f_tri
            h = h + h_tri

            # fmt: off
            if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                    particle_index,
                    i_adj_tri,
                    particle_order,
                    f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
                )
            # fmt: on
    if edge_indices:
        for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
                adjacency, particle_index, i_adj_edge
            )
            if edge_bending_properties[nei_edge_index, 0] != 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                    nei_edge_index,
                    vertex_order_on_edge,
                    pos,
                    pos_prev,
                    edge_indices,
                    edge_rest_angles,
                    edge_rest_length,
                    edge_bending_properties[nei_edge_index, 0],
                    edge_bending_properties[nei_edge_index, 1],
                    dt,
                )

                f += f_edge
                h += h_edge

    if tet_indices:
        for i_adj_tet in range(get_vertex_num_adjacent_tets(adjacency, particle_index)):
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(adjacency, particle_index, i_adj_tet)

            f_tet, h_tet = evaluate_volumetric_neo_hooken_force_and_hessian(
                nei_tet_index,
                vertex_order_on_tet,
                pos_prev,
                pos,
                tet_indices,
                tet_poses[nei_tet_index],
                tet_materials[nei_tet_index, 0],
                tet_materials[nei_tet_index, 1],
                tet_materials[nei_tet_index, 2],
                dt,
            )

            f += f_tet
            h += h_tet

    h += particle_hessians[particle_index]
    f += particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-5:
        hInv = wp.inverse(h)
        pos_new[particle_index] = particle_pos + hInv * f


@wp.kernel
def copy_particle_positions_back(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle = particle_ids_in_color[tid]

    pos[particle] = pos_new[particle]


@wp.kernel
def update_velocity(
    dt: float, pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt


@wp.kernel
def convert_body_particle_contact_data_kernel(
    # inputs
    body_particle_contact_buffer_pre_alloc: int,
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # outputs
    body_particle_contact_buffer: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
):
    contact_index = wp.tid()
    count = min(contact_max, contact_count[0])
    if contact_index >= count:
        return

    particle_index = soft_contact_particle[contact_index]
    offset = particle_index * body_particle_contact_buffer_pre_alloc

    contact_counter = wp.atomic_add(body_particle_contact_count, particle_index, 1)
    if contact_counter < body_particle_contact_buffer_pre_alloc:
        body_particle_contact_buffer[offset + contact_counter] = contact_index


@wp.kernel
def accumulate_self_contact_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                c_e1_v1 = particle_colors[e1_v1]
                c_e1_v2 = particle_colors[e1_v2]
                if c_e1_v1 == current_color or c_e1_v2 == current_color:
                    has_contact, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1 = (
                        evaluate_edge_edge_contact_2_vertices(
                            e1_idx,
                            e2_idx,
                            pos,
                            pos_prev,
                            edge_indices,
                            collision_radius,
                            soft_contact_ke,
                            soft_contact_kd,
                            friction_mu,
                            friction_epsilon,
                            dt,
                            edge_edge_parallel_epsilon,
                        )
                    )

                    if has_contact:
                        # here we only handle the e1 side, because e2 will also detection this contact and add force and hessian on its own
                        if c_e1_v1 == current_color:
                            wp.atomic_add(particle_forces, e1_v1, collision_force_0)
                            wp.atomic_add(particle_hessians, e1_v1, collision_hessian_0)
                        if c_e1_v2 == current_color:
                            wp.atomic_add(particle_forces, e1_v2, collision_force_1)
                            wp.atomic_add(particle_hessians, e1_v2, collision_hessian_1)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                c_v = particle_colors[particle_idx]
                c_tri_a = particle_colors[tri_a]
                c_tri_b = particle_colors[tri_b]
                c_tri_c = particle_colors[tri_c]

                if (
                    c_v == current_color
                    or c_tri_a == current_color
                    or c_tri_b == current_color
                    or c_tri_c == current_color
                ):
                    (
                        has_contact,
                        collision_force_0,
                        collision_force_1,
                        collision_force_2,
                        collision_force_3,
                        collision_hessian_0,
                        collision_hessian_1,
                        collision_hessian_2,
                        collision_hessian_3,
                    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                        particle_idx,
                        tri_idx,
                        pos,
                        pos_prev,
                        tri_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        dt,
                    )

                    if has_contact:
                        # particle
                        if c_v == current_color:
                            wp.atomic_add(particle_forces, particle_idx, collision_force_3)
                            wp.atomic_add(particle_hessians, particle_idx, collision_hessian_3)

                        # tri_a
                        if c_tri_a == current_color:
                            wp.atomic_add(particle_forces, tri_a, collision_force_0)
                            wp.atomic_add(particle_hessians, tri_a, collision_hessian_0)

                        # tri_b
                        if c_tri_b == current_color:
                            wp.atomic_add(particle_forces, tri_b, collision_force_1)
                            wp.atomic_add(particle_hessians, tri_b, collision_hessian_1)

                        # tri_c
                        if c_tri_c == current_color:
                            wp.atomic_add(particle_forces, tri_c, collision_force_2)
                            wp.atomic_add(particle_hessians, tri_c, collision_hessian_2)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE


@wp.kernel
def apply_planar_truncation_parallel_by_collision(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    displacement_in: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    parallel_eps: float,
    gamma: float,
    truncation_t_out: wp.array(dtype=float),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

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

                # For each, check the corresponding is_dummy entry in the vec4 is_dummy
                if not is_dummy[0]:
                    t = planar_truncation_t(e1_v1_pos, delta_e1_v1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e1_v1, t)
                if not is_dummy[1]:
                    t = planar_truncation_t(e1_v2_pos, delta_e1_v2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e1_v2, t)
                if not is_dummy[2]:
                    t = planar_truncation_t(e2_v1_pos, delta_e2_v1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e2_v1, t)
                if not is_dummy[3]:
                    t = planar_truncation_t(e2_v2_pos, delta_e2_v2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e2_v2, t)

                # planar truncation for 2 sides
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id

        colliding_particle_pos = pos[particle_idx]
        colliding_particle_displacement = displacement_in[particle_idx]

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                t1 = pos[tri_a]
                t2 = pos[tri_b]
                t3 = pos[tri_c]
                delta_t1 = displacement_in[tri_a]
                delta_t2 = displacement_in[tri_b]
                delta_t3 = displacement_in[tri_c]

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

                # planar truncation for 2 sides
                if not is_dummy[0]:
                    t = planar_truncation_t(
                        colliding_particle_pos, colliding_particle_displacement, n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, particle_idx, t)
                if not is_dummy[1]:
                    t = planar_truncation_t(t1, delta_t1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_a, t)
                if not is_dummy[2]:
                    t = planar_truncation_t(t2, delta_t2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_b, t)
                if not is_dummy[3]:
                    t = planar_truncation_t(t3, delta_t3, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_c, t)

            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # dont forget to do the final truncation based on the maximum displament allowance!


@wp.kernel
def apply_truncation_ts(
    pos: wp.array(dtype=wp.vec3),
    displacement_in: wp.array(dtype=wp.vec3),
    truncation_ts: wp.array(dtype=float),
    displacement_out: wp.array(dtype=wp.vec3),
    pos_out: wp.array(dtype=wp.vec3),
    max_displacement: float = 1e10,
):
    i = wp.tid()
    t = truncation_ts[i]
    particle_displacement = displacement_in[i] * t

    # Nuts-saving truncation: clamp displacement magnitude to max_displacement
    len_displacement = wp.length(particle_displacement)
    if len_displacement > max_displacement:
        particle_displacement = particle_displacement * max_displacement / len_displacement

    displacement_out[i] = particle_displacement
    if pos_out:
        pos_out[i] = pos[i] + particle_displacement


@wp.kernel
def hessian_dykstra_projection( # Anka's Dykstra Function, We use it here except that division planes are not precomputed and projections are hessian weighted
        max_iter:int,
        pos: wp.array(dtype=wp.vec3),
        displacement_in: wp.array(dtype=wp.vec3),
        particle_hessians: wp.array(dtype=wp.mat33),
        tri_indices: wp.array(dtype=wp.int32, ndim=2),
        edge_indices: wp.array(dtype=wp.int32, ndim=2),
        adjacency: ForceElementAdjacencyInfo,
        collision_info_arr: wp.array(dtype=TriMeshCollisionInfo),
        parallel_eps: float,
        #division_plane_nds_vt: wp.array(dtype=wp.vec3),             # shape = vertex_colliding_triangles.shape
        projection_t_vt: wp.array(dtype=float),                      # shape = (vertex_colliding_triangles.shape // 2)
        #division_plane_is_dummy_vt: wp.array(dtype=wp.bool),        # shape = vertex_colliding_triangles.shape // 2
        #division_plane_nds_ee: wp.array(dtype=wp.vec3),             # shape = edge_colliding_edges.shape
        projection_t_ee: wp.array(dtype=float),                      # shape = (edge_colliding_edges.shape, one for per vertex on the edge)
        #division_plane_is_dummy_ee: wp.array(dtype=wp.bool),        # shape = edge_colliding_edges.shape, one for per vertex on the edge
        #division_plane_nds_tv: wp.array(dtype=wp.vec3),             # self.collision_detector.collision_info.triangle_colliding_vertices.shape[0] * 2
        projection_t_tv: wp.array(dtype=float),                      # shape = (triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle)
        #division_plane_is_dummy_tv: wp.array(dtype=wp.bool),        # shape = triangle_colliding_vertices.shape[0] * 3, one for per vertex on the triangle
        displacements_out: wp.array(dtype=wp.vec3),
):
    particle_idx = wp.tid()
    particle_pos = pos[particle_idx]
    particle_displacement = displacement_in[particle_idx]
    x = pos[particle_idx] + particle_displacement
    delta_x_k = particle_displacement
    H_v = particle_hessians[particle_idx]
    det_H = wp.determinant(H_v)
    collision_info = collision_info_arr[0]
    '''
    if wp.abs(det_H) < 1e-8:
        # Degenerate case: fall back to no projection
        displacements_out[particle_idx] = particle_displacement
        return
    else:
    '''
    H_v_inv = wp.inverse(H_v)
    for iter in range(max_iter):

        for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_idx)):
            offset_v_t = collision_info.vertex_colliding_triangles_offsets[particle_idx]
            colliding_tri_index = get_vertex_colliding_triangles(collision_info, particle_idx, i_v_collision)
            t1 = pos[tri_indices[colliding_tri_index, 0]]
            t2 = pos[tri_indices[colliding_tri_index, 1]]
            t3 = pos[tri_indices[colliding_tri_index, 2]]
            
            delta_t1 = displacement_in[tri_indices[colliding_tri_index, 0]]
            delta_t2 = displacement_in[tri_indices[colliding_tri_index, 1]]
            delta_t3 = displacement_in[tri_indices[colliding_tri_index, 2]]
            
            is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                particle_pos,
                delta_x_k,
                t1,
                delta_t1,
                t2,
                delta_t2,
                t3,
                delta_t3,
            )
            
            if not is_dummy[0]:
                # Simplified: no dual variable storage (y_{v,t} = 0 for now)
                # Full implementation: x* = delta_x_k - y_{v,t}
                t = projection_t_vt[offset_v_t + i_v_collision]
                x_star = delta_x_k - t * n
                delta_x_k = hessian_weighted_projection_onto_halfspace(
                    particle_pos + x_star, n, d, H_v_inv, parallel_eps
                ) - particle_pos
                # Full implementation: y_{v,t} = x_star - delta_x_k
                t_new = wp.dot(x_star - delta_x_k, n)
                projection_t_vt[offset_v_t + i_v_collision] = t_new
                


        for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_idx)):
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_idx, i_adj_tri)
            t1 = pos[tri_indices[tri_index, 0]]
            t2 = pos[tri_indices[tri_index, 1]]
            t3 = pos[tri_indices[tri_index, 2]]
            
            delta_t1 = displacement_in[tri_indices[tri_index, 0]]
            delta_t2 = displacement_in[tri_indices[tri_index, 1]]
            delta_t3 = displacement_in[tri_indices[tri_index, 2]]

            for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
                offset_t_v = collision_info.triangle_colliding_vertices_offsets[tri_index]
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
                    t = projection_t_tv[3 * (offset_t_v + i_t_collision) + vertex_order]
                    x_star = delta_x_k - t * n
                    delta_x_k = hessian_weighted_projection_onto_halfspace(
                        particle_pos + x_star, n, d, H_v_inv, parallel_eps
                    ) - particle_pos
                    t_new = wp.dot(x_star - delta_x_k, n)
                    projection_t_tv[3 * (offset_t_v + i_t_collision) + vertex_order] = t_new
                    
                    

        for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_idx)):
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_idx,
                                                                                    i_adj_edge)

            if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
                vertex_order_on_edge_2_v1_v2_only = vertex_order_on_edge - 2
                for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):
                    offset_e_e = collision_info.edge_colliding_edges_offsets[nei_edge_index]
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
                    vertex_order_on_edge_2_v1_v2_only = vertex_order_on_edge - 2
                    
                    if not is_dummy[vertex_order_on_edge_2_v1_v2_only]:
                        t = projection_t_ee[2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only]
                        x_star = delta_x_k - t * n
                        delta_x_k = hessian_weighted_projection_onto_halfspace(
                            particle_pos + x_star, n, d, H_v_inv, parallel_eps
                        ) - particle_pos
                        projection_t_ee[2 * (offset_e_e + i_e_collision) + vertex_order_on_edge_2_v1_v2_only] = t_new
                        
    displacements_out[particle_idx] = delta_x_k

@wp.kernel
def accumulate_self_contact_force_and_hessian_tile(
    # inputs
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_SELF_CONTACT_SOLVE
    thread_idx = tid % TILE_SIZE_SELF_CONTACT_SOLVE
    particle_index = particle_ids_in_color[block_idx]
    collision_info = collision_info_array[0]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        return

    # elastic force and hessian
    num_colliding_tris = get_vertex_colliding_triangles_count(collision_info, particle_index)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_colliding_tris:
        colliding_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_SELF_CONTACT_SOLVE
        # elastic force and hessian
        colliding_t = get_vertex_colliding_triangles(collision_info, particle_index, colliding_tri_counter)

        collision_force, collision_hessian = evaluate_vertex_triangle_collision_force_hessian(
            particle_index,
            3,
            colliding_t,
            pos,
            pos_prev,
            tri_indices,
            collision_radius,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            dt,
        )

        f += collision_force
        h += collision_hessian

    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        num_tri_colliding_vertices = get_triangle_colliding_vertices_count(collision_info, tri_index)

        batch_counter = wp.int32(0)
        while batch_counter + thread_idx < num_tri_colliding_vertices:
            colliding_vertex_counter = thread_idx + batch_counter
            batch_counter += TILE_SIZE_SELF_CONTACT_SOLVE
            colliding_vertex = get_triangle_colliding_vertices(collision_info, tri_index, colliding_vertex_counter)

            collision_force, collision_hessian = evaluate_vertex_triangle_collision_force_hessian(
                colliding_vertex,
                vertex_order,
                tri_index,
                pos,
                pos_prev,
                tri_indices,
                collision_radius,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
                dt,
            )

            f = f + collision_force
            h = h + collision_hessian

    # edge-edge collision force and hessian
    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        # vertex is on the edge; otherwise it only effects the bending energy not the edge collision
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            batch_counter = wp.int32(0)
            num_e_colliding_e = get_edge_colliding_edges_count(collision_info, nei_edge_index)
            while batch_counter + thread_idx < num_e_colliding_e:
                colliding_edge_counter = thread_idx + batch_counter
                batch_counter += TILE_SIZE_SELF_CONTACT_SOLVE
                colliding_edge = get_edge_colliding_edges(collision_info, nei_edge_index, colliding_edge_counter)

                collision_force, collision_hessian = evaluate_edge_edge_contact(
                    particle_index,
                    vertex_order_on_edge - 2,
                    nei_edge_index,
                    colliding_edge,
                    pos,
                    pos_prev,
                    edge_indices,
                    collision_radius,
                    soft_contact_ke,
                    soft_contact_kd,
                    friction_mu,
                    friction_epsilon,
                    dt,
                    edge_edge_parallel_epsilon,
                )
                f = f + collision_force
                h = h + collision_hessian

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        particle_forces[particle_index] += f_total
        particle_hessians[particle_index] += h_total


def _csr_row(vals: np.ndarray, offs: np.ndarray, i: int) -> np.ndarray:
    return vals[offs[i] : offs[i + 1]]


def _set_to_csr(list_of_sets, dtype=np.int32, sort=True):
    offsets = np.zeros(len(list_of_sets) + 1, dtype=dtype)
    sizes = np.fromiter((len(s) for s in list_of_sets), count=len(list_of_sets), dtype=dtype)
    np.cumsum(sizes, out=offsets[1:])
    flat = np.empty(offsets[-1], dtype=dtype)
    idx = 0
    for s in list_of_sets:
        if sort:
            arr = np.fromiter(sorted(s), count=len(s), dtype=dtype)
        else:
            arr = np.fromiter(s, count=len(s), dtype=dtype)

        flat[idx : idx + len(arr)] = arr
        idx += len(arr)
    return flat, offsets


def one_ring_vertices(
    v: int, edge_indices: np.ndarray, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    e_u = edge_indices[:, 2]
    e_v = edge_indices[:, 3]
    # preserve only the adjacent edge information, remove the order information
    inc_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[::2]
    inc_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[1::2]
    if inc_edges.size == 0:
        return np.empty(0)
    us = e_u[inc_edges[np.where(inc_edges_order >= 2)]]
    vs = e_v[inc_edges[np.where(inc_edges_order >= 2)]]

    assert (np.logical_or(us == v, vs == v)).all()
    nbrs = np.unique(np.concatenate([us, vs]))
    return nbrs[nbrs != v]


def leq_n_ring_vertices(
    v: int, edge_indices: np.ndarray, n: int, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    visited = {v}
    frontier = {v}
    for _ in range(n):
        next_frontier = set()
        for u in frontier:
            for w in one_ring_vertices(u, edge_indices, v_adj_edges, v_adj_edges_offsets):  # iterable of neighbors of u
                if w not in visited:
                    visited.add(w)
                    next_frontier.add(w)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(visited, dtype=int)


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    num_vertices: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_faces: np.ndarray,
    v_adj_faces_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    v_nei_tri_sets = [set() for _ in range(num_vertices)]

    for v in range(num_vertices):
        # distance-1 vertices

        if n == 2:
            ring_n_minus_1 = one_ring_vertices(v, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1 = leq_n_ring_vertices(v, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        ring_1_tri_set = set(_csr_row(v_adj_faces, v_adj_faces_offsets, v)[::2])

        nei_tri_set = v_nei_tri_sets[v]
        for w in ring_n_minus_1:
            if w != v:
                # preserve only the adjacent edge information, remove the order information
                nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])

        nei_tri_set.difference_update(ring_1_tri_set)

    return v_nei_tri_sets


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    edge_nei_edge_sets = [set() for _ in range(edge_indices.shape[0])]

    for e_idx in range(edge_indices.shape[0]):
        # distance-1 vertices
        v1 = edge_indices[e_idx, 2]
        v2 = edge_indices[e_idx, 3]

        if n == 2:
            ring_n_minus_1_v1 = one_ring_vertices(v1, edge_indices, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = one_ring_vertices(v2, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1_v1 = leq_n_ring_vertices(v1, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = leq_n_ring_vertices(v2, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        all_neighbors = set(ring_n_minus_1_v1)
        all_neighbors.update(ring_n_minus_1_v2)

        ring_1_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2])
        ring_2_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v2)[::2])

        nei_edge_set = edge_nei_edge_sets[e_idx]
        for w in all_neighbors:
            if w != v1 and w != v2:
                # preserve only the adjacent edge information, remove the order information
                # nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])
                adj_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[::2]
                adj_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[1::2]
                adj_collision_edges = adj_edges[np.where(adj_edges_order >= 2)]
                nei_edge_set.update(adj_collision_edges)

        nei_edge_set.difference_update(ring_1_edge_set)
        nei_edge_set.difference_update(ring_2_edge_set)

    return edge_nei_edge_sets


def _csr_row(vals: np.ndarray, offs: np.ndarray, i: int) -> np.ndarray:
    """Extract CSR row `i` from the flattened adjacency arrays."""
    return vals[offs[i] : offs[i + 1]]


def _set_to_csr(list_of_sets, dtype=np.int32, sort=True):
    """
    Convert a list of integer sets into CSR (Compressed Sparse Row) structure.

    Args:
        list_of_sets: Iterable where each entry is a set of ints.
        dtype: Output dtype for the flattened arrays.
        sort: Whether to sort each row when writing into `flat`.

    Returns:
        A tuple `(flat, offsets)` representing the CSR values and offsets.
    """
    offsets = np.zeros(len(list_of_sets) + 1, dtype=dtype)
    sizes = np.fromiter((len(s) for s in list_of_sets), count=len(list_of_sets), dtype=dtype)
    np.cumsum(sizes, out=offsets[1:])
    flat = np.empty(offsets[-1], dtype=dtype)
    idx = 0
    for s in list_of_sets:
        if sort:
            arr = np.fromiter(sorted(s), count=len(s), dtype=dtype)
        else:
            arr = np.fromiter(s, count=len(s), dtype=dtype)

        flat[idx : idx + len(arr)] = arr
        idx += len(arr)
    return flat, offsets


def one_ring_vertices(
    v: int, edge_indices: np.ndarray, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """
    Find immediate neighboring vertices that share an edge with vertex `v`.

    Args:
        v: Vertex index whose neighborhood is queried.
        edge_indices: Array of shape [num_edges, 4] storing edge endpoint indices.
        v_adj_edges: Flattened CSR adjacency array listing edge ids and local order.
        v_adj_edges_offsets: CSR offsets indexing into `v_adj_edges`.

    Returns:
        Sorted array of neighboring vertex indices, excluding `v`.
    """
    e_u = edge_indices[:, 2]
    e_v = edge_indices[:, 3]
    # preserve only the adjacent edge information, remove the order information
    inc_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[::2]
    inc_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[1::2]
    if inc_edges.size == 0:
        return np.empty(0)
    us = e_u[inc_edges[np.where(inc_edges_order >= 2)]]
    vs = e_v[inc_edges[np.where(inc_edges_order >= 2)]]

    assert (np.logical_or(us == v, vs == v)).all()
    nbrs = np.unique(np.concatenate([us, vs]))
    return nbrs[nbrs != v]


def leq_n_ring_vertices(
    v: int, edge_indices: np.ndarray, n: int, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """
    Find all vertices within n-ring distance of vertex v using BFS.

    Args:
        v: Starting vertex index
        edge_indices: Edge connectivity array
        n: Maximum ring distance
        v_adj_edges: CSR values for vertex-edge adjacency
        v_adj_edges_offsets: CSR offsets for vertex-edge adjacency

    Returns:
        Array of all vertices within n-ring distance, including v itself
    """
    visited = {v}
    frontier = {v}
    for _ in range(n):
        next_frontier = set()
        for u in frontier:
            for w in one_ring_vertices(u, edge_indices, v_adj_edges, v_adj_edges_offsets):  # iterable of neighbors of u
                if w not in visited:
                    visited.add(w)
                    next_frontier.add(w)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(visited, dtype=int)


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    num_vertices: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_faces: np.ndarray,
    v_adj_faces_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    v_nei_tri_sets = [set() for _ in range(num_vertices)]

    for v in range(num_vertices):
        # distance-1 vertices

        if n == 2:
            ring_n_minus_1 = one_ring_vertices(v, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1 = leq_n_ring_vertices(v, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        ring_1_tri_set = set(_csr_row(v_adj_faces, v_adj_faces_offsets, v)[::2])

        nei_tri_set = v_nei_tri_sets[v]
        for w in ring_n_minus_1:
            if w != v:
                # preserve only the adjacent edge information, remove the order information
                nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])

        nei_tri_set.difference_update(ring_1_tri_set)

    return v_nei_tri_sets


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    edge_nei_edge_sets = [set() for _ in range(edge_indices.shape[0])]

    for e_idx in range(edge_indices.shape[0]):
        # distance-1 vertices
        v1 = edge_indices[e_idx, 2]
        v2 = edge_indices[e_idx, 3]

        if n == 2:
            ring_n_minus_1_v1 = one_ring_vertices(v1, edge_indices, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = one_ring_vertices(v2, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1_v1 = leq_n_ring_vertices(v1, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = leq_n_ring_vertices(v2, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        all_neighbors = set(ring_n_minus_1_v1)
        all_neighbors.update(ring_n_minus_1_v2)

        ring_1_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2])
        ring_2_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v2)[::2])

        nei_edge_set = edge_nei_edge_sets[e_idx]
        for w in all_neighbors:
            if w != v1 and w != v2:
                # preserve only the adjacent edge information, remove the order information
                # nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])
                adj_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[::2]
                adj_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[1::2]
                adj_collision_edges = adj_edges[np.where(adj_edges_order >= 2)]
                nei_edge_set.update(adj_collision_edges)

        nei_edge_set.difference_update(ring_1_edge_set)
        nei_edge_set.difference_update(ring_2_edge_set)

    return edge_nei_edge_sets


@wp.func
def evaluate_spring_force_and_hessian(
    particle_idx: int,
    spring_idx: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
):
    v0 = spring_indices[spring_idx * 2]
    v1 = spring_indices[spring_idx * 2 + 1]

    diff = pos[v0] - pos[v1]
    l = wp.length(diff)
    l0 = spring_rest_length[spring_idx]

    force_sign = 1.0 if particle_idx == v0 else -1.0

    spring_force = force_sign * spring_stiffness[spring_idx] * (l0 - l) / l * diff
    spring_hessian = spring_stiffness[spring_idx] * (
        wp.identity(3, float) - (l0 / l) * (wp.identity(3, float) - wp.outer(diff, diff) / (l * l))
    )

    # compute damping
    h_d = spring_hessian * (spring_damping[spring_idx] / dt)

    f_d = h_d * (pos_prev[particle_idx] - pos[particle_idx])

    spring_force = spring_force + f_d
    spring_hessian = spring_hessian + h_d

    return spring_force, spring_hessian


@wp.kernel
def accumulate_spring_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_ids_in_color: wp.array(dtype=int),
    adjacency: ForceElementAdjacencyInfo,
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    num_adj_springs = get_vertex_num_adjacent_springs(adjacency, particle_index)
    for spring_counter in range(num_adj_springs):
        spring_index = get_vertex_adjacent_spring_id(adjacency, particle_index, spring_counter)
        spring_force, spring_hessian = evaluate_spring_force_and_hessian(
            particle_index,
            spring_index,
            dt,
            pos,
            pos_prev,
            spring_indices,
            spring_rest_length,
            spring_stiffness,
            spring_damping,
        )

        particle_forces[particle_index] = particle_forces[particle_index] + spring_force
        particle_hessians[particle_index] = particle_hessians[particle_index] + spring_hessian


@wp.kernel
def accumulate_particle_body_contact_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_body_contact_count = min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


@wp.kernel
def solve_elasticity(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    particle_displacements: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # fmt: off
    if wp.static("inertia_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "particle: %d after accumulate inertia\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    if tri_indices:
        # elastic force and hessian
        for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

            # fmt: off
            if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d | num_adj_faces: %d | ",
                    particle_index,
                    get_vertex_num_adjacent_faces(particle_index, adjacency),
                )
                wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_index, vertex_order)
                wp.printf(
                    "face: %d %d %d\n",
                    tri_indices[tri_index, 0],
                    tri_indices[tri_index, 1],
                    tri_indices[tri_index, 2],
                )
            # fmt: on

            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
                f_tri, h_tri = evaluate_stvk_force_hessian(
                    tri_index,
                    vertex_order,
                    pos,
                    pos_prev,
                    tri_indices,
                    tri_poses[tri_index],
                    tri_areas[tri_index],
                    tri_materials[tri_index, 0],
                    tri_materials[tri_index, 1],
                    tri_materials[tri_index, 2],
                    dt,
                )

                f = f + f_tri
                h = h + h_tri

    if edge_indices:
        for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
            # vertex is on the edge; otherwise it only effects the bending energy n
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                    nei_edge_index, vertex_order_on_edge, pos, pos_prev, edge_indices, edge_rest_angles, edge_rest_length,
                    edge_bending_properties[nei_edge_index, 0], edge_bending_properties[nei_edge_index, 1], dt
                )

                f = f + f_edge
                h = h + h_edge

    if tet_indices:
        # solve tet elasticity
        num_adj_tets = get_vertex_num_adjacent_tets(adjacency, particle_index)
        for adj_tet_counter in range(num_adj_tets):
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                adjacency, particle_index, adj_tet_counter
            )
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_volumetric_neo_hooken_force_and_hessian(
                    nei_tet_index,
                    vertex_order_on_tet,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[nei_tet_index],
                    tet_materials[nei_tet_index, 0],
                    tet_materials[nei_tet_index, 1],
                    tet_materials[nei_tet_index, 2],
                    dt,
                )

                f += f_tet
                h += h_tet

    # fmt: off
    if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # # fmt: on
    h = h + particle_hessians[particle_index]
    f = f + particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-8:
        h_inv = wp.inverse(h)
        particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f

    particle_forces[particle_index] = f

@wp.kernel
def solve_elasticity_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    particle_displacements: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # elastic force and hessian
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    if tri_indices:
        # loop through all the adjacent triangles using whole block
        while batch_counter + thread_idx < num_adj_faces:
            adj_tri_counter = thread_idx + batch_counter
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            # elastic force and hessian
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

            # fmt: off
            if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d | num_adj_faces: %d | ",
                    particle_index,
                    get_vertex_num_adjacent_faces(particle_index, adjacency),
                )
                wp.printf("i_face: %d | face id: %d | v_order: %d | ", adj_tri_counter, tri_index, vertex_order)
                wp.printf(
                    "face: %d %d %d\n",
                    tri_indices[tri_index, 0],
                    tri_indices[tri_index, 1],
                    tri_indices[tri_index, 2],
                )
            # fmt: on

            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
                f_tri, h_tri = evaluate_stvk_force_hessian(
                    tri_index,
                    vertex_order,
                    pos,
                    pos_prev,
                    tri_indices,
                    tri_poses[tri_index],
                    tri_areas[tri_index],
                    tri_materials[tri_index, 0],
                    tri_materials[tri_index, 1],
                    tri_materials[tri_index, 2],
                    dt,
                )

                f += f_tri
                h += h_tri

    if edge_indices:
        batch_counter = wp.int32(0)
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_edges:
            adj_edge_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
                adjacency, particle_index, adj_edge_counter
            )
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                    nei_edge_index,
                    vertex_order_on_edge,
                    pos,
                    pos_prev,
                    edge_indices,
                    edge_rest_angles,
                    edge_rest_length,
                    edge_bending_properties[nei_edge_index, 0],
                    edge_bending_properties[nei_edge_index, 1],
                    dt,
                )

                f += f_edge
                h += h_edge

    if tet_indices:
        # solve tet elasticity
        batch_counter = wp.int32(0)
        num_adj_tets = get_vertex_num_adjacent_tets(adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_tets:
            adj_tet_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                adjacency, particle_index, adj_tet_counter
            )
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_volumetric_neo_hooken_force_and_hessian(
                    nei_tet_index,
                    vertex_order_on_tet,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[nei_tet_index],
                    tet_materials[nei_tet_index, 0],
                    tet_materials[nei_tet_index, 1],
                    tet_materials[nei_tet_index, 2],
                    dt,
                )

                f += f_tet
                h += h_tet

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-8:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )
            particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f_total


class SolverVBD(SolverBase):
    """An implicit solver using Vertex Block Descent (VBD) for cloth simulation.

    References:
        - Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block Descent. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages.
          https://doi.org/10.1145/3658179

    Note:
        `SolverVBD` requires particle coloring information through :attr:`newton.Model.particle_color_groups`.
        You may call :meth:`newton.ModelBuilder.color` to color particles or use :meth:`newton.ModelBuilder.set_coloring`
        to provide you own particle coloring.

    Example
    -------

    .. code-block:: python

        # color particles
        builder.color()
        # or you can use your custom coloring
        builder.set_coloring(user_provided_particle_coloring)

        model = builder.finalize()

        solver = newton.solvers.SolverVBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        handle_self_contact: bool = False,
        self_contact_radius: float = 0.2,
        self_contact_margin: float = 0.2,
        topological_contact_filter_threshold: int = 2,
        rest_shape_contact_exclusion_radius: float = 0.0,
        external_vertex_contact_filtering_map: dict | None = None,
        external_edge_contact_filtering_map: dict | None = None,
        integrate_with_external_rigid_solver: bool = False,
        penetration_free_conservative_bound_relaxation: float = 0.42,
        friction_epsilon: float = 1e-2,
        vertex_collision_buffer_pre_alloc: int = 32,
        edge_collision_buffer_pre_alloc: int = 64,
        collision_detection_interval: int = 0,
        edge_edge_parallel_epsilon: float = 1e-10,
        use_tile_solve: bool = True,
        truncation_mode: int = 1,  # 0: isometric, 1: planar (DAT), 2: CCD (global min t)
        dykstra_iterations: int = 20,  # Number of Dykstra iterations for mode 2
    ):
        """
        Args:
            model: The `Model` object used to initialize the integrator. Must be identical to the `Model` object passed
                to the `step` function.
            iterations: Number of VBD iterations per step.
            handle_self_contact: whether to self-contact.
            self_contact_radius: The radius used for self-contact detection. This is the distance at which vertex-triangle
                pairs and edge-edge pairs will start to interact with each other.
            self_contact_margin: The margin used for self-contact detection. This is the distance at which vertex-triangle
                pairs and edge-edge will be considered in contact generation. It should be larger than `self_contact_radius`
                to avoid missing contacts.
            topological_contact_filter_threshold: Maximum topological distance (measured in rings) under which candidate
                self-contacts are discarded. Set to a higher value to tolerate contacts between more closely connected mesh
                elements. Only used when `handle_self_contact` is `True`. Note that setting this to a value larger than 3 will
                result in a significant increase in computation time.
            rest_shape_contact_exclusion_radius: Additional world-space distance threshold for filtering topologically close
                primitives. Candidate contacts with a rest separation shorter than this value are ignored. The distance is
                evaluated in the rest configuration conveyed by `model.particle_q`. Only used when `handle_self_contact` is `True`.
            external_vertex_contact_filtering_map: Optional dictionary used to exclude additional vertex-triangle pairs during
                contact generation. Keys must be vertex primitive ids (integers), and each value must be a `list` or
                `set` containing the triangle primitives to be filtered out. Only used when `handle_self_contact` is `True`.
            external_edge_contact_filtering_map: Optional dictionary used to exclude additional edge-edge pairs during contact
                generation. Keys must be edge primitive ids (integers), and each value must be a `list` or `set`
                containing the edges to be filtered out. Only used when `handle_self_contact` is `True`.
            integrate_with_external_rigid_solver: an indicator of coupled rigid body - cloth simulation.  When set to
                `True`, the solver assumes the rigid body solve is handled  externally.
            penetration_free_conservative_bound_relaxation: Relaxation factor for conservative penetration-free projection.
            friction_epsilon: Threshold to smooth small relative velocities in friction computation.
            vertex_collision_buffer_pre_alloc: Preallocation size for each vertex's vertex-triangle collision buffer.
            edge_collision_buffer_pre_alloc: Preallocation size for edge's edge-edge collision buffer.
            edge_edge_parallel_epsilon: Threshold to detect near-parallel edges in edge-edge collision handling.
            collision_detection_interval: Controls how frequently collision detection is applied during the simulation.
                If set to a value < 0, collision detection is only performed once before the initialization step.
                If set to 0, collision detection is applied twice: once before and once immediately after initialization.
                If set to a value `k` >= 1, collision detection is applied before every `k` VBD iterations.
            use_tile_solve: whether to accelerate the solver using tile API
        Note:
            - The `integrate_with_external_rigid_solver` argument is an indicator of one-way coupling between rigid body
              and soft body solvers. If set to True, the rigid states should be integrated externally, with `state_in`
              passed to `step` function representing the previous rigid state and `state_out` representing the current one. Frictional forces are
              computed accordingly.
            - vertex_collision_buffer_pre_alloc` and `edge_collision_buffer_pre_alloc` are fixed and will not be
              dynamically resized during runtime.
              Setting them too small may result in undetected collisions.
              Setting them excessively large may increase memory usage and degrade performance.

        """
        super().__init__(model)
        self.iterations = iterations
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver
        self.collision_detection_interval = collision_detection_interval

        self.topological_contact_filter_threshold = topological_contact_filter_threshold
        self.rest_shape_contact_exclusion_radius = rest_shape_contact_exclusion_radius

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)

        self.handle_self_contact = handle_self_contact
        self.self_contact_radius = self_contact_radius
        self.self_contact_margin = self_contact_margin
        self.rest_shape = model.particle_q
        self.particle_conservative_bounds = wp.full((self.model.particle_count,), dtype=float, device=self.device)
        self.truncation_mode = truncation_mode
        self.dykstra_iterations = dykstra_iterations
        self.ccd_detector = None  # Initialized lazily when truncation_mode == 2

        if model.device.is_cpu and use_tile_solve:
            warnings.warn("Tiled solve requires model.device='cuda'. Tiled solve is disabled.", stacklevel=2)

        self.use_tile_solve = use_tile_solve and model.device.is_cuda

        if handle_self_contact:
            if self_contact_margin < self_contact_radius:
                raise ValueError(
                    "self_contact_margin is smaller than self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make self_contact_margin 1.5-2 times larger than self_contact_radius."
                )

            self.conservative_bound_relaxation = penetration_free_conservative_bound_relaxation
            # CCD safety margin: 2x the conservative bound relaxation (e.g., 0.42 -> 0.84)
            self.ccd_safety_margin = 2.0 * self.conservative_bound_relaxation

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
                edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
                edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
                record_triangle_contacting_vertices=True,
            )

            self.compute_contact_filtering_list(
                external_vertex_contact_filtering_map, external_edge_contact_filtering_map
            )

            self.trimesh_collision_detector.set_collision_filter_list(
                self.vertex_triangle_contact_filtering_list,
                self.vertex_triangle_contact_filtering_list_offsets,
                self.edge_edge_contact_filtering_list,
                self.edge_edge_contact_filtering_list_offsets,
            )

            self.compute_contact_filtering_list(
                external_vertex_contact_filtering_map, external_edge_contact_filtering_map
            )

            self.trimesh_collision_detector.set_collision_filter_list(
                self.vertex_triangle_contact_filtering_list,
                self.vertex_triangle_contact_filtering_list_offsets,
                self.edge_edge_contact_filtering_list,
                self.edge_edge_contact_filtering_list_offsets,
            )

            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
            )

            self.self_contact_evaluation_kernel_launch_size = max(
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                # soft_contact_max,
            )
        else:
            # Still need a valid size for body-particle contact evaluation
            self.self_contact_evaluation_kernel_launch_size = (
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE
            )

        self.truncation_ts = wp.zeros(self.model.particle_count, dtype=float, device=self.device)
        self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)

        # spaces for particle force and hessian
        self.particle_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_displacements = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.device)

        self.friction_epsilon = friction_epsilon

        if len(self.model.particle_color_groups) == 0:
            raise ValueError(
                "model.particle_color_groups is empty! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

        # tests
        # wp.launch(kernel=_test_compute_force_element_adjacency,
        #           inputs=[self.adjacency, model.edge_indices, model.tri_indices],
        #           dim=1, device=self.device)
        if self.truncation_mode == 3:
            self.project_t_vt = wp.zeros(dtype=float, shape=(len(self.trimesh_collision_detector.collision_info.vertex_colliding_triangles) // 2,), device=self.device)
            self.project_t_ee = wp.zeros(dtype=float, shape=self.trimesh_collision_detector.collision_info.edge_colliding_edges.shape, device=self.device)
            self.project_t_tv = wp.zeros(dtype=float, shape=(len(self.trimesh_collision_detector.collision_info.triangle_colliding_vertices) * 3,), device=self.device)
            self.dis_out = wp.zeros_like(self.particle_displacements, device=self.device)

    def compute_force_element_adjacency(self, model):
        adjacency = ForceElementAdjacencyInfo()

        with wp.ScopedDevice("cpu"):
            if model.edge_indices:
                edges_array = model.edge_indices.to("cpu")
                # build vertex-edge adjacency data
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                )

                num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
                vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
                vertex_adjacent_edges_offsets[0] = 0
                adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent edges has been filled to each vertex
                vertex_adjacent_edges_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
                # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
                adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        adjacency.v_adj_edges,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if model.tri_indices:
                face_indices = model.tri_indices.to("cpu")
                # compute adjacent triangles
                # count number of adjacent faces for each vertex
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(kernel=self.count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1)

                # preallocate memory based on counting results
                num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
                vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
                vertex_adjacent_faces_offsets[0] = 0
                adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32)

                vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
                # (face, vertex_order) * num_adj_faces * num_particles
                # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
                adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        adjacency.v_adj_faces,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if model.tet_indices:
                tet_indices = model.tet_indices.to("cpu")
                num_vertex_adjacent_tets = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_tets,
                    inputs=[tet_indices, num_vertex_adjacent_tets],
                    dim=1,
                )

                num_vertex_adjacent_tets = num_vertex_adjacent_tets.numpy()
                vertex_adjacent_tets_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_tets_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_tets)[:]
                vertex_adjacent_tets_offsets[0] = 0
                adjacency.v_adj_tets_offsets = wp.array(vertex_adjacent_tets_offsets, dtype=wp.int32)

                vertex_adjacent_tets_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                tet_adjacency_array_size = 2 * num_vertex_adjacent_tets.sum()
                adjacency.v_adj_tets = wp.empty(shape=(tet_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_tets,
                    inputs=[
                        tet_indices,
                        adjacency.v_adj_tets_offsets,
                        vertex_adjacent_tets_fill_count,
                        adjacency.v_adj_tets,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_tets_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_tets = wp.empty(shape=(0,), dtype=wp.int32)

            if model.spring_indices:
                spring_array = model.spring_indices.to("cpu")
                # build vertex-springs adjacency data
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                )

                num_vertex_adjacent_spring = num_vertex_adjacent_spring.numpy()
                vertex_adjacent_springs_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_springs_offsets[1:] = np.cumsum(num_vertex_adjacent_spring)[:]
                vertex_adjacent_springs_offsets[0] = 0
                adjacency.v_adj_springs_offsets = wp.array(vertex_adjacent_springs_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent springs has been filled to each vertex
                vertex_adjacent_springs_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(num_vertex_adjacent_spring.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        adjacency.v_adj_springs,
                    ],
                    dim=1,
                )

            else:
                adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

    # def initialize_division_plane_buffer(self):
    #     projection_buffer_sizes = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device=self.device)
    #
    #     wp.launch(
    #         dim=self.model.particle_count,
    #         kernel=calculate_vertex_collision_buffer,
    #         inputs=[
    #             self.adjacency,
    #             self.trimesh_collision_detector.collision_info,
    #         ],
    #         outputs=[projection_buffer_sizes],
    #         device=self.device,
    #     )
    #     vertex_division_plane_buffer_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=np.int32)
    #     vertex_division_plane_buffer_offsets[1:] = np.cumsum(projection_buffer_sizes.numpy())[:]
    #     vertex_division_plane_buffer_offsets[0] = 0
    #     buffer_size_total = vertex_division_plane_buffer_offsets[-1]
    #     self.vertex_division_plane_buffer_offsets = wp.array(
    #         vertex_division_plane_buffer_offsets, dtype=int, device=self.device
    #     )  # add this to the definition
    #
    #     self.division_plane_nds = wp.empty(buffer_size_total * 2, dtype=wp.vec3, device=self.device)
    #     self.division_gap_width = wp.empty(buffer_size_total * 2, dtype=float, device=self.device)
    #     self.division_num_planes = wp.empty(self.model.particle_count, dtype=int, device=self.device)

    def compute_contact_filtering_list(
        self, external_vertex_contact_filtering_map, external_edge_contact_filtering_map
    ):
        if self.model.tri_count:
            v_tri_filter_sets = None
            edge_edge_filter_sets = None
            if self.topological_contact_filter_threshold >= 2:
                if self.adjacency.v_adj_faces_offsets.size > 0:
                    v_tri_filter_sets = build_vertex_n_ring_tris_collision_filter(
                        self.topological_contact_filter_threshold,
                        self.model.particle_count,
                        self.model.edge_indices.numpy(),
                        self.adjacency.v_adj_edges.numpy(),
                        self.adjacency.v_adj_edges_offsets.numpy(),
                        self.adjacency.v_adj_faces.numpy(),
                        self.adjacency.v_adj_faces_offsets.numpy(),
                    )
                if self.adjacency.v_adj_edges_offsets.size > 0:
                    edge_edge_filter_sets = build_edge_n_ring_edge_collision_filter(
                        self.topological_contact_filter_threshold,
                        self.model.edge_indices.numpy(),
                        self.adjacency.v_adj_edges.numpy(),
                        self.adjacency.v_adj_edges_offsets.numpy(),
                    )

            if external_vertex_contact_filtering_map is not None:
                if v_tri_filter_sets is None:
                    v_tri_filter_sets = [set() for _ in range(self.model.particle_count)]
                for vertex_id, filter_set in external_vertex_contact_filtering_map.items():
                    v_tri_filter_sets[vertex_id].update(filter_set)

            if external_edge_contact_filtering_map is not None:
                if edge_edge_filter_sets is None:
                    edge_edge_filter_sets = [set() for _ in range(self.model.edge_count)]
                for edge_idx, filter_set in external_edge_contact_filtering_map.items():
                    edge_edge_filter_sets[edge_idx].update(filter_set)

            if v_tri_filter_sets is None:
                self.vertex_triangle_contact_filtering_list = None
                self.vertex_triangle_contact_filtering_list_offsets = None
            else:
                self.vertex_triangle_contact_filtering_list, self.vertex_triangle_contact_filtering_list_offsets = (
                    _set_to_csr(v_tri_filter_sets)
                )
                self.vertex_triangle_contact_filtering_list = wp.array(
                    self.vertex_triangle_contact_filtering_list, dtype=int, device=self.device
                )
                self.vertex_triangle_contact_filtering_list_offsets = wp.array(
                    self.vertex_triangle_contact_filtering_list_offsets, dtype=int, device=self.device
                )

            if edge_edge_filter_sets is None:
                self.edge_edge_contact_filtering_list = None
                self.edge_edge_contact_filtering_list_offsets = None
            else:
                self.edge_edge_contact_filtering_list, self.edge_edge_contact_filtering_list_offsets = _set_to_csr(
                    edge_edge_filter_sets
                )
                self.edge_edge_contact_filtering_list = wp.array(
                    self.edge_edge_contact_filtering_list, dtype=int, device=self.device
                )
                self.edge_edge_contact_filtering_list_offsets = wp.array(
                    self.edge_edge_contact_filtering_list_offsets, dtype=int, device=self.device
                )

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        if self.use_tile_solve:
            self.simulate_one_step_tile(state_in, state_out, control, contacts, dt)
        else:
            self.simulate_one_step_no_tile(state_in, state_out, control, contacts, dt)

    def simulate_one_step_no_tile(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ):
        self.collision_detection_penetration_free(state_in)

        model = self.model

        wp.launch(
            kernel=forward_step,
            inputs=[
                dt,
                model.gravity,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
            ],
            outputs=[
                self.inertia,
                self.particle_displacements,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        self.penetration_free_truncation(state_in.particle_q)

        for _iter in range(self.iterations):
            # after initialization, we need new collision detection to update the bounds
            if (self.collision_detection_interval == 0 and _iter == 0) or (
                self.collision_detection_interval >= 1 and _iter % self.collision_detection_interval == 0
            ):
                self.collision_detection_penetration_free(state_in)

            self.particle_forces.zero_()
            self.particle_hessians.zero_()

            for color in range(len(self.model.particle_color_groups)):
                if contacts is not None:
                    wp.launch(
                        kernel=accumulate_particle_body_contact_force_and_hessian,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_colors,
                            # body-particle contact
                            self.model.soft_contact_ke,
                            self.model.soft_contact_kd,
                            self.model.soft_contact_mu,
                            self.friction_epsilon,
                            self.model.particle_radius,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_count,
                            contacts.soft_contact_max,
                            self.model.shape_material_mu,
                            self.model.shape_body,
                            state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                            state_in.body_q if self.integrate_with_external_rigid_solver else None,
                            self.model.body_qd,
                            self.model.body_com,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        device=self.device,
                        # max_blocks=self.model.device.sm_count,
                    )

                if model.spring_count:
                    wp.launch(
                        kernel=accumulate_spring_force_and_hessian,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_color_groups[color],
                            self.adjacency,
                            self.model.spring_indices,
                            self.model.spring_rest_length,
                            self.model.spring_stiffness,
                            self.model.spring_damping,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        dim=self.model.particle_color_groups[color].size,
                        device=self.device,
                    )

                if self.handle_self_contact:
                    # wp.launch(
                    #     kernel=accumulate_self_contact_force_and_hessian_tile,
                    #     dim=self.model.particle_color_groups[color].size * TILE_SIZE_SELF_CONTACT_SOLVE,
                    #     block_dim=TILE_SIZE_SELF_CONTACT_SOLVE,
                    #     inputs=[
                    #         dt,
                    #         self.model.particle_color_groups[color],
                    #         self.particle_q_prev,
                    #         state_in.particle_q,
                    #         self.model.particle_flags,
                    #         self.model.tri_indices,
                    #         self.model.edge_indices,
                    #         self.adjacency,
                    #         # self contact
                    #         self.trimesh_collision_info,
                    #         self.self_contact_radius,
                    #         self.model.soft_contact_ke,
                    #         self.model.soft_contact_kd,
                    #         self.model.soft_contact_mu,
                    #         self.friction_epsilon,
                    #         self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    #         # outputs: particle force and hessian
                    #         self.particle_forces,
                    #         self.particle_hessians,
                    #     ],
                    #     device=self.device,
                    #     max_blocks=self.model.device.sm_count,
                    # )
                    wp.launch(
                        kernel=accumulate_self_contact_force_and_hessian,
                        dim=self.self_contact_evaluation_kernel_launch_size,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_colors,
                            self.model.tri_indices,
                            self.model.edge_indices,
                            # self-contact
                            self.trimesh_collision_info,
                            self.self_contact_radius,
                            self.model.soft_contact_ke,
                            self.model.soft_contact_kd,
                            self.model.soft_contact_mu,
                            self.friction_epsilon,
                            self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        device=self.device,
                        max_blocks=self.model.device.sm_count,
                    )

                wp.launch(
                    kernel=solve_elasticity,
                    dim=self.model.particle_color_groups[color].size,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.particle_q_prev,
                        state_in.particle_q,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.model.tet_indices,
                        self.model.tet_poses,
                        self.model.tet_materials,
                        self.adjacency,
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    outputs=[
                        self.particle_displacements,
                    ],
                    device=self.device,
                )
                if self.truncation_mode == 3:
                    # initialize dykstra arrays
                    # call the dykstra kernel
                    
                    wp.launch(
                        kernel=hessian_dykstra_projection,
                        inputs=[
                            self.dykstra_iterations,
                            self.pos_prev_collision_detection,
                            self.particle_displacements,
                            self.particle_hessians,
                            self.model.tri_indices,
                            self.model.edge_indices,
                            self.adjacency,
                            self.trimesh_collision_info,
                            1e-7,
                            self.project_t_vt,
                            self.project_t_ee,
                            self.project_t_tv,
                        ],
                        outputs=[
                            self.dis_out,
                        ],
                        dim=self.model.particle_count,
                        device=self.device,
                    )
                self.penetration_free_truncation(state_in.particle_q)

        wp.copy(state_out.particle_q, state_in.particle_q)

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def simulate_one_step_tile(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ):
        # collision detection before initialization to compute conservative bounds for initialization
        # To support two truncation methods, we have to edit this function to make it either
        # 0: compute conservative bounds
        # 1: compute division planes
        # which is determined by the truncation_mode
        self.collision_detection_penetration_free(state_in)

        model = self.model

        wp.launch(
            kernel=forward_step,
            inputs=[
                dt,
                model.gravity,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
            ],
            outputs=[
                self.inertia,
                self.particle_displacements,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        self.penetration_free_truncation(state_in.particle_q)

        for _iter in range(self.iterations):
            # after initialization, we need new collision detection to update the bounds
            if (self.collision_detection_interval == 0 and _iter == 0) or (
                self.collision_detection_interval >= 1 and _iter % self.collision_detection_interval == 0
            ):
                self.collision_detection_penetration_free(state_in)

            self.particle_forces.zero_()
            self.particle_hessians.zero_()

            for color in range(len(self.model.particle_color_groups)):
                if contacts is not None:
                    wp.launch(
                        kernel=accumulate_particle_body_contact_force_and_hessian,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_colors,
                            # body-particle contact
                            self.model.soft_contact_ke,
                            self.model.soft_contact_kd,
                            self.model.soft_contact_mu,
                            self.friction_epsilon,
                            self.model.particle_radius,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_count,
                            contacts.soft_contact_max,
                            self.model.shape_material_mu,
                            self.model.shape_body,
                            state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                            state_in.body_q if self.integrate_with_external_rigid_solver else None,
                            self.model.body_qd,
                            self.model.body_com,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        device=self.device,
                        # max_blocks=self.model.device.sm_count,
                    )

                if model.spring_count:
                    wp.launch(
                        kernel=accumulate_spring_force_and_hessian,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_color_groups[color],
                            self.adjacency,
                            self.model.spring_indices,
                            self.model.spring_rest_length,
                            self.model.spring_stiffness,
                            self.model.spring_damping,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        dim=self.model.particle_color_groups[color].size,
                        device=self.device,
                    )

                if self.handle_self_contact:
                    # wp.launch(
                    #     kernel=accumulate_self_contact_force_and_hessian_tile,
                    #     dim=self.model.particle_color_groups[color].size * TILE_SIZE_SELF_CONTACT_SOLVE,
                    #     block_dim=TILE_SIZE_SELF_CONTACT_SOLVE,
                    #     inputs=[
                    #         dt,
                    #         self.model.particle_color_groups[color],
                    #         self.particle_q_prev,
                    #         state_in.particle_q,
                    #         self.model.particle_flags,
                    #         self.model.tri_indices,
                    #         self.model.edge_indices,
                    #         self.adjacency,
                    #         # self contact
                    #         self.trimesh_collision_info,
                    #         self.self_contact_radius,
                    #         self.model.soft_contact_ke,
                    #         self.model.soft_contact_kd,
                    #         self.model.soft_contact_mu,
                    #         self.friction_epsilon,
                    #         self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    #         # outputs: particle force and hessian
                    #         self.particle_forces,
                    #         self.particle_hessians,
                    #     ],
                    #     device=self.device,
                    #     max_blocks=self.model.device.sm_count,
                    # )
                    wp.launch(
                        kernel=accumulate_self_contact_force_and_hessian,
                        dim=self.self_contact_evaluation_kernel_launch_size,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_colors,
                            self.model.tri_indices,
                            self.model.edge_indices,
                            # self-contact
                            self.trimesh_collision_info,
                            self.self_contact_radius,
                            self.model.soft_contact_ke,
                            self.model.soft_contact_kd,
                            self.model.soft_contact_mu,
                            self.friction_epsilon,
                            self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        device=self.device,
                        max_blocks=self.model.device.sm_count,
                    )

                wp.launch(
                    kernel=solve_elasticity_tile,
                    dim=self.model.particle_color_groups[color].size * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.particle_q_prev,
                        state_in.particle_q,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.model.tet_indices,
                        self.model.tet_poses,
                        self.model.tet_materials,
                        self.adjacency,
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    outputs=[
                        self.particle_displacements,
                    ],
                    device=self.device,
                )
                if self.truncation_mode == 3:
                    # initialize dykstra arrays
                    # call the dykstra kernel
                    
                    wp.launch(
                        kernel=hessian_dykstra_projection,
                        inputs=[
                            self.dykstra_iterations,
                            self.pos_prev_collision_detection,
                            self.particle_displacements,
                            self.particle_hessians,
                            self.model.tri_indices,
                            self.model.edge_indices,
                            self.adjacency,
                            self.trimesh_collision_info,
                            1e-7,
                            self.project_t_vt,
                            self.project_t_ee,
                            self.project_t_tv,
                        ],
                        outputs=[
                            self.dis_out,
                        ],
                        dim=self.model.particle_count,
                        device=self.device,
                    )

                self.penetration_free_truncation(state_in.particle_q)

        wp.copy(state_out.particle_q, state_in.particle_q)

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def collision_detection_penetration_free(self, current_state: State):
        self.pos_prev_collision_detection.assign(current_state.particle_q)
        self.particle_displacements.zero_()

        if self.handle_self_contact:
            self.trimesh_collision_detector.refit(current_state.particle_q)
            self.trimesh_collision_detector.vertex_triangle_collision_detection(
                self.self_contact_margin,
                min_query_radius=self.rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.rest_shape,
            )
            self.trimesh_collision_detector.edge_edge_collision_detection(
                self.self_contact_margin,
                min_query_radius=self.rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.rest_shape,
            )

            if self.truncation_mode == 0:
                wp.launch(
                    kernel=compute_particle_conservative_bound,
                    inputs=[
                        self.conservative_bound_relaxation,
                        self.self_contact_margin,
                        self.adjacency,
                        self.trimesh_collision_detector.collision_info,
                    ],
                    outputs=[
                        self.particle_conservative_bounds,
                    ],
                    dim=self.model.particle_count,
                    device=self.device,
                )

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (newton.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.handle_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)

    def resize_collision_buffers(self, shrink_to_fit: bool = False, growth_ratio: float = 1.5) -> bool:
        """Resize collision buffers based on actual collision counts from the last detection pass.

        This function analyzes the collision counts and resizes buffers that overflowed
        (or shrinks oversized buffers if shrink_to_fit=True). Use this after collision
        detection if you observe buffer overflow warnings or want to optimize memory usage.

        Buffer sizes are:
        - Multiplied by growth_ratio to provide headroom and reduce resize frequency
        - Rounded up to the next multiple of 4 for memory alignment
        - Clamped between pre_alloc and max_alloc settings

        Note:
            After resizing, you should re-run collision detection before the next simulation
            step to populate the new buffers.

        Args:
            shrink_to_fit: If True, also shrink buffers that are larger than needed.
                          If False (default), only grow buffers that overflowed.
            growth_ratio: Multiplier for collision counts to provide headroom. Default is 1.5
                         (50% extra space). Set to 1.0 for exact fit (no headroom).

        Returns:
            True if any buffer was resized, False otherwise.

        Example:
            .. code-block:: python

                # After running simulation and observing overflow
                if solver.resize_collision_buffers():
                    # Buffers were resized, re-run collision detection
                    solver.rebuild_bvh(state)

                # To reclaim memory after simulation settles
                solver.resize_collision_buffers(shrink_to_fit=True)
        """
        if not self.handle_self_contact:
            return False
        resized = self.trimesh_collision_detector.resize_collision_buffer_to_fit(shrink_to_fit, growth_ratio)
        if resized:
            # Update the collision_info array with new buffer references
            # This is critical - the old array contains stale pointers to deallocated memory
            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
            )
        return resized

    def penetration_free_truncation(self, particle_q_out=None):
        """
        Modify displacements_in in-place, also modify particle_q if its not None

        """
        if not self.handle_self_contact:
            self.truncation_ts.fill_(1.0)
            wp.launch(
                kernel=apply_truncation_ts,
                dim=self.model.particle_count,
                inputs=[
                    self.pos_prev_collision_detection,  # pos: wp.array(dtype=wp.vec3),
                    self.particle_displacements,  # displacement_in: wp.array(dtype=wp.vec3),
                    self.truncation_ts,  # truncation_ts: wp.array(dtype=float),
                ],
                outputs=[
                    self.particle_displacements,  # displacement_out: wp.array(dtype=wp.vec3),
                    particle_q_out,  # pos_out: wp.array(dtype=wp.vec3),
                    wp.inf,  # max_displacement: float
                ],
                device=self.device,
            )

        else:
            if self.truncation_mode == 0:
                # Mode 0: Isometric truncation (conservative bounds)
                wp.launch(
                    kernel=apply_conservative_bound_truncation,
                    inputs=[
                        self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                        self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array(dtype=wp.vec3),
                        self.particle_conservative_bounds,  # particle_conservative_bounds: wp.array(dtype=float),
                        particle_q_out,  # particle_q_out: wp.array(dtype=wp.vec3),
                    ],
                    dim=self.model.particle_count,
                    device=self.device,
                )
            elif self.truncation_mode == 2:
                # Mode 2: CCD truncation (global min t)
                self.penetration_free_truncation_ccd(particle_q_out)
            else:
                # ## IMIPLEMENTATION 1: parallel by vertex
                # wp.launch(
                #     kernel=apply_planar_truncation,
                #     inputs=[
                #         self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array(dtype=wp.vec3),
                #         self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                #         self.model.tri_indices,
                #         self.model.edge_indices,
                #         self.adjacency,
                #         self.trimesh_collision_info,
                #         self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                #         self.conservative_bound_relaxation * 2,
                #         self.self_contact_margin * self.conservative_bound_relaxation,
                #     ],
                #     outputs=[
                #         self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                #         particle_q_out,
                #     ],
                #     dim=self.model.particle_count,
                #     device=self.device,
                # )

                # pos: wp.array(dtype=wp.vec3),
                # displacement_in: wp.array(dtype=wp.vec3),
                # tri_indices: wp.array(dtype=wp.int32, ndim=2),
                # edge_indices: wp.array(dtype=wp.int32, ndim=2),
                # collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
                # parallel_eps: float,
                # gamma: float,
                # truncation_t_out: wp.array(dtype=wp.vec3),

                ## IMIPLEMENTATION 2: parallel by collision and atomic operation
                self.truncation_ts.fill_(1.0)
                wp.launch(
                    kernel=apply_planar_truncation_parallel_by_collision,
                    inputs=[
                        self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array(dtype=wp.vec3),
                        self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                        self.model.tri_indices,
                        self.model.edge_indices,
                        self.trimesh_collision_info,
                        self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                        self.conservative_bound_relaxation * 2,
                    ],
                    outputs=[
                        self.truncation_ts,
                    ],
                    dim=self.self_contact_evaluation_kernel_launch_size,
                    device=self.device,
                )

                wp.launch(
                    kernel=apply_truncation_ts,
                    dim=self.model.particle_count,
                    inputs=[
                        self.pos_prev_collision_detection,  # pos: wp.array(dtype=wp.vec3),
                        self.particle_displacements,  # displacement_in: wp.array(dtype=wp.vec3),
                        self.truncation_ts,  # truncation_ts: wp.array(dtype=float),
                        self.particle_displacements,  # displacement_out: wp.array(dtype=wp.vec3),
                        particle_q_out,  # pos_out: wp.array(dtype=wp.vec3),
                        self.self_contact_margin * self.conservative_bound_relaxation,  # max_displacement: float
                    ],
                    device=self.device,
                )

    def penetration_free_truncation_ccd(self, particle_q_out=None):
        """
        CCD-based truncation: finds the GLOBAL minimum collision time across all
        vertex-triangle and edge-edge pairs, then scales ALL displacements uniformly.
        
        This is the most conservative approach - if ANY primitive collides at time t,
        ALL vertices are scaled to stop at time t.
        """
        # Create or update CCD detector
        if self.ccd_detector is None:
            # CCD builds its own filter lists using the adjacency info
            # This is independent of DCD's topological_contact_filter_threshold
            self.ccd_detector = TriMeshContinuousCollisionDetector(
                self.trimesh_collision_detector,
                self.pos_prev_collision_detection,
                self.particle_displacements,
                adjacency=self.adjacency,  # Pass adjacency for building CCD's own filter lists
                filter_threshold=2,  # Always filter 2-ring neighbors for CCD
            )
        else:
            # Update positions and displacements, then refit BVH (faster than rebuild)
            wp.copy(self.ccd_detector.vertex_positions, self.pos_prev_collision_detection)
            wp.copy(self.ccd_detector.vertex_displacements, self.particle_displacements)
            self.ccd_detector.refit()
        
        # Run V-T and E-E CCD detection
        self.ccd_detector.detect_vertex_triangle_ccd()
        self.ccd_detector.detect_edge_edge_ccd()
        
        # Find GLOBAL minimum collision time
        vt_times = self.ccd_detector.vertex_collision_times.numpy()
        ee_times = self.ccd_detector.edge_collision_times.numpy()
        
        min_vt = float(vt_times.min()) if len(vt_times) > 0 else 1.0
        min_ee = float(ee_times.min()) if len(ee_times) > 0 else 1.0
        global_min_t = min(min_vt, min_ee)
        
        # Apply safety margin and clamp
        global_t = max(0.0, min(1.0, global_min_t * self.ccd_safety_margin))
        
        # Debug: print statistics (uncomment to debug CCD)
        # num_vt_collisions = int(np.sum(vt_times < 1.0))
        # num_ee_collisions = int(np.sum(ee_times < 1.0))
        # print(f"CCD Debug: min_vt={min_vt:.6f} ({num_vt_collisions} VT), min_ee={min_ee:.6f} ({num_ee_collisions} EE), global_t={global_t:.6f}")
        
        # Fill truncation_ts with global_t and apply
        self.truncation_ts.fill_(global_t)
        wp.launch(
            kernel=apply_truncation_ts,
            dim=self.model.particle_count,
            inputs=[
                self.pos_prev_collision_detection,  # pos: wp.array(dtype=wp.vec3),
                self.particle_displacements,  # displacement_in: wp.array(dtype=wp.vec3),
                self.truncation_ts,  # truncation_ts: wp.array(dtype=float),
                self.particle_displacements,  # displacement_out: wp.array(dtype=wp.vec3),
                particle_q_out,  # pos_out: wp.array(dtype=wp.vec3),
                wp.inf,  # max_displacement: float (no additional clamping for CCD)
            ],
            device=self.device,
        )

    def penetration_free_truncation_tile(self, particle_q_out=None):
        """
        Modify displacements_in in-place, also modify particle_q if its not None

        """
        if self.truncation_mode == 0:
            # Mode 0: Isometric truncation
            wp.launch(
                kernel=apply_conservative_bound_truncation,
                inputs=[
                    self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                    self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array(dtype=wp.vec3),
                    self.particle_conservative_bounds,  # particle_conservative_bounds: wp.array(dtype=float),
                    particle_q_out,  # particle_q_out: wp.array(dtype=wp.vec3),
                ],
                dim=self.model.particle_count,
                device=self.device,
            )
        elif self.truncation_mode == 2:
            # Mode 2: CCD truncation (global min t) - same as non-tile version
            self.penetration_free_truncation_ccd(particle_q_out)
        else:
            # Mode 1: Planar truncation (DAT)
            wp.launch(
                kernel=apply_planar_truncation_tile,
                inputs=[
                    self.pos_prev_collision_detection,  # pos_prev_collision_detection: wp.array(dtype=wp.vec3),
                    self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.adjacency,
                    self.trimesh_collision_info,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    self.conservative_bound_relaxation * 2,
                    self.self_contact_margin * self.conservative_bound_relaxation,
                ],
                outputs=[
                    self.particle_displacements,  # particle_displacements: wp.array(dtype=wp.vec3),
                    particle_q_out,
                ],
                dim=self.model.particle_count * TILE_SIZE_SELF_CONTACT_SOLVE,
                block_dim=TILE_SIZE_SELF_CONTACT_SOLVE,
                device=self.device,
                # max_blocks=self.model.device.sm_count,
            )

    @wp.kernel
    def count_num_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_edges: wp.array(dtype=wp.int32)
    ):
        for edge_id in range(edges_array.shape[0]):
            o0 = edges_array[edge_id, 0]
            o1 = edges_array[edge_id, 1]

            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
            num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

            if o0 != -1:
                num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
            if o1 != -1:
                num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1

    @wp.kernel
    def fill_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_edges_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_edges_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_edges: wp.array(dtype=wp.int32),
    ):
        for edge_id in range(edges_array.shape[0]):
            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
            vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
            vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

            o0 = edges_array[edge_id, 0]
            if o0 != -1:
                fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
                buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
                vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

            o1 = edges_array[edge_id, 1]
            if o1 != -1:
                fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
                buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
                vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1

    @wp.kernel
    def count_num_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_faces: wp.array(dtype=wp.int32)
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            num_vertex_adjacent_faces[v0] = num_vertex_adjacent_faces[v0] + 1
            num_vertex_adjacent_faces[v1] = num_vertex_adjacent_faces[v1] + 1
            num_vertex_adjacent_faces[v2] = num_vertex_adjacent_faces[v2] + 1

    @wp.kernel
    def fill_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_faces_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_faces_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_faces: wp.array(dtype=wp.int32),
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            fill_count_v0 = vertex_adjacent_faces_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_faces_offsets[v0]
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2] = face
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
            vertex_adjacent_faces_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_faces_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_faces_offsets[v1]
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2] = face
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
            vertex_adjacent_faces_fill_count[v1] = fill_count_v1 + 1

            fill_count_v2 = vertex_adjacent_faces_fill_count[v2]
            buffer_offset_v2 = vertex_adjacent_faces_offsets[v2]
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2] = face
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
            vertex_adjacent_faces_fill_count[v2] = fill_count_v2 + 1

    @wp.kernel
    def count_num_adjacent_tets(
        tet_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_tets: wp.array(dtype=wp.int32)
    ):
        for tet in range(tet_indices.shape[0]):
            v0 = tet_indices[tet, 0]
            v1 = tet_indices[tet, 1]
            v2 = tet_indices[tet, 2]
            v3 = tet_indices[tet, 3]

            num_vertex_adjacent_tets[v0] = num_vertex_adjacent_tets[v0] + 1
            num_vertex_adjacent_tets[v1] = num_vertex_adjacent_tets[v1] + 1
            num_vertex_adjacent_tets[v2] = num_vertex_adjacent_tets[v2] + 1
            num_vertex_adjacent_tets[v3] = num_vertex_adjacent_tets[v3] + 1

    @wp.kernel
    def fill_adjacent_tets(
        tet_indices: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_tets_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_tets_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_tets: wp.array(dtype=wp.int32),
    ):
        for tet in range(tet_indices.shape[0]):
            v0 = tet_indices[tet, 0]
            v1 = tet_indices[tet, 1]
            v2 = tet_indices[tet, 2]
            v3 = tet_indices[tet, 3]

            fill_count_v0 = vertex_adjacent_tets_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_tets_offsets[v0]
            vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2] = tet
            vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
            vertex_adjacent_tets_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_tets_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_tets_offsets[v1]
            vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2] = tet
            vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
            vertex_adjacent_tets_fill_count[v1] = fill_count_v1 + 1

            fill_count_v2 = vertex_adjacent_tets_fill_count[v2]
            buffer_offset_v2 = vertex_adjacent_tets_offsets[v2]
            vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2] = tet
            vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
            vertex_adjacent_tets_fill_count[v2] = fill_count_v2 + 1

            fill_count_v3 = vertex_adjacent_tets_fill_count[v3]
            buffer_offset_v3 = vertex_adjacent_tets_offsets[v3]
            vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2] = tet
            vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2 + 1] = 3
            vertex_adjacent_tets_fill_count[v3] = fill_count_v3 + 1

    @wp.kernel
    def count_num_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32), num_vertex_adjacent_springs: wp.array(dtype=wp.int32)
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            num_vertex_adjacent_springs[v0] = num_vertex_adjacent_springs[v0] + 1
            num_vertex_adjacent_springs[v1] = num_vertex_adjacent_springs[v1] + 1

    @wp.kernel
    def fill_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_springs: wp.array(dtype=wp.int32),
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            fill_count_v0 = vertex_adjacent_springs_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_springs_offsets[v0]
            vertex_adjacent_springs[buffer_offset_v0 + fill_count_v0] = spring_id
            vertex_adjacent_springs_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_springs_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_springs_offsets[v1]
            vertex_adjacent_springs[buffer_offset_v1 + fill_count_v1] = spring_id
            vertex_adjacent_springs_fill_count[v1] = fill_count_v1 + 1
