# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernel-level tests for the ALM-DAT particle contact planes and truncation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector, TriMeshCollisionInfo
from newton._src.solvers.vbd.particle_dat_kernels import (
    build_contact_planes,
    commit_truncated_displacement,
    create_particle_contact_dat_planes,
    dat_plane_offset,
    truncate_against_contact_planes,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _plane_offset_kernel(
    n: wp.vec3,
    negative_pos: wp.array[wp.vec3],
    negative_disp: wp.array[wp.vec3],
    negative_count: int,
    positive_pos: wp.array[wp.vec3],
    positive_disp: wp.array[wp.vec3],
    positive_count: int,
    margin: float,
    eps: float,
    out: wp.array[float],
):
    neg = wp.mat33(0.0)
    neg_d = wp.mat33(0.0)
    for i in range(negative_count):
        neg[i] = negative_pos[i]
        neg_d[i] = negative_disp[i]
    pos = wp.mat33(0.0)
    pos_d = wp.mat33(0.0)
    for j in range(positive_count):
        pos[j] = positive_pos[j]
        pos_d[j] = positive_disp[j]
    plane_point, room, margin_eff = dat_plane_offset(
        n, neg, neg_d, negative_count, pos, pos_d, positive_count, margin, eps
    )
    out[0] = wp.dot(n, plane_point)
    out[1] = room
    out[2] = margin_eff


def _run_plane_offset(device, negative, negative_disp, positive, positive_disp, margin=0.25, eps=1.0e-6):
    out = wp.zeros(3, dtype=float, device=device)
    wp.launch(
        _plane_offset_kernel,
        dim=1,
        inputs=[
            wp.vec3(0.0, 0.0, 1.0),
            wp.array(negative, dtype=wp.vec3, device=device),
            wp.array(negative_disp, dtype=wp.vec3, device=device),
            len(negative),
            wp.array(positive, dtype=wp.vec3, device=device),
            wp.array(positive_disp, dtype=wp.vec3, device=device),
            len(positive),
            margin,
            eps,
        ],
        outputs=[out],
        device=device,
    )
    return out.numpy()


def test_plane_offset_balances_slowest_vertices(test, device):
    """Place the plane where the slowest approaching vertex on each side gets the same fraction."""
    # Worked example from the design notes: triangle heights 0, -1, -2 moving up by 2, 5
    # and down by 1; vertex at height 3 moving down by 3; margin 0.25 per side.
    triangle = [(0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 1.0, -2.0)]
    triangle_disp = [(0.0, 0.0, 2.0), (0.0, 0.0, 5.0), (0.0, 0.0, -1.0)]
    vertex = [(0.3, 0.3, 3.0)]
    vertex_disp = [(0.0, 0.0, -3.0)]
    offset, room, margin_eff = _run_plane_offset(device, triangle, triangle_disp, vertex, vertex_disp)
    # the fast triangle vertex (rate 5) binds against the vertex (rate 3):
    # b = (3 * (-1 + 0.25) + 5 * (3 - 0.25)) / 8
    test.assertAlmostEqual(float(offset), 1.4375, places=5)
    test.assertAlmostEqual(float(room), 0.4375, places=5)
    test.assertAlmostEqual(float(margin_eff), 0.25, places=6)


def test_plane_offset_gives_all_room_to_the_only_approaching_side(test, device):
    """Put the plane at the band edge next to a side that does not approach."""
    triangle = [(0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 1.0, -2.0)]
    receding = [(0.0, 0.0, -2.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)]
    vertex = [(0.3, 0.3, 3.0)]
    vertex_disp = [(0.0, 0.0, -3.0)]
    offset, room, margin_eff = _run_plane_offset(device, triangle, receding, vertex, vertex_disp)
    # plane sits just above the highest triangle vertex; the vertex can take (3 - 0.25 - 0.25)/3
    test.assertAlmostEqual(float(offset), 0.25, places=5)
    test.assertAlmostEqual(float(room), 2.5 / 3.0, places=5)
    # thin pair: margin shrinks to half the gap
    thin_vertex = [(0.3, 0.3, 0.1)]
    _offset, _room, margin_eff = _run_plane_offset(device, triangle, receding, thin_vertex, vertex_disp)
    test.assertAlmostEqual(float(margin_eff), 0.05, places=5)


def _two_layer_model(device, layer_gap=0.012):
    """Two 3x3 cloth grids facing each other, close enough for self-contact pairs."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    grid, cell = 3, 0.05
    for layer in range(2):
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * grid * cell, -0.5 * grid * cell, 1.0 + layer * layer_gap),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=grid,
            dim_y=grid,
            cell_x=cell,
            cell_y=cell,
            mass=0.1,
            tri_ke=1.0e2,
            tri_ka=1.0e2,
            tri_kd=1.0e-1,
            edge_ke=1.0,
            particle_radius=0.01,
        )
    builder.color()
    return builder.finalize(device=device)


def _detect(model, query_radius=0.03):
    detector = TriMeshCollisionDetector(
        model, init_collision_info=True, topological_contact_filter_threshold=2, sort_contact_rows=True
    )
    detector.vertex_triangle_collision_detection(query_radius)
    detector.edge_edge_collision_detection(query_radius)
    info = detector.collision_info
    info_array = wp.array([info], dtype=TriMeshCollisionInfo, device=model.device)
    return detector, info, info_array


def _plane_sides(model, info, planes, anchor, margin_tol=1.0e-6):
    """Return the worst signed-distance violation over all valid planes (positive = violated)."""
    tri = model.tri_indices.numpy()
    edges = model.edge_indices.numpy()
    pos = anchor.numpy()
    worst = 0.0
    checked = 0
    rows = info.vertex_colliding_triangles.numpy()
    valid = planes.vt_valid.numpy()
    normal = planes.vt_normal.numpy()
    point = planes.vt_point.numpy()
    margin = planes.vt_margin.numpy()
    for slot in range(int(info.vertex_colliding_triangles_offsets.numpy()[-1])):
        if valid[slot] == 0:
            continue
        v, t = int(rows[2 * slot]), int(rows[2 * slot + 1])
        n, p, m = normal[slot], point[slot], float(margin[slot])
        worst = max(worst, m - float(np.dot(n, pos[v] - p)) - margin_tol)
        for k in range(3):
            worst = max(worst, float(np.dot(n, pos[tri[t, k]] - p)) + m - margin_tol)
        checked += 1
    rows = info.edge_colliding_edges.numpy()
    valid = planes.ee_valid.numpy()
    normal = planes.ee_normal.numpy()
    point = planes.ee_point.numpy()
    margin = planes.ee_margin.numpy()
    for slot in range(int(info.edge_colliding_edges_offsets.numpy()[-1])):
        if valid[slot] == 0:
            continue
        e1, e2 = int(rows[2 * slot]), int(rows[2 * slot + 1])
        n, p, m = normal[slot], point[slot], float(margin[slot])
        for k in (2, 3):
            worst = max(worst, m - float(np.dot(n, pos[edges[e1, k]] - p)) - margin_tol)
            worst = max(worst, float(np.dot(n, pos[edges[e2, k]] - p)) + m - margin_tol)
        checked += 1
    return worst, checked


def test_build_contact_planes_certifies_every_stored_pair(test, device):
    """Build one certified plane per stored pair with the vertex side positive and only canonical edge records valid."""
    model = _two_layer_model(device)
    _detector, info, info_array = _detect(model)
    planes = create_particle_contact_dat_planes(info, device)
    anchor = wp.clone(model.particle_q)
    trial = wp.clone(model.particle_q)
    build_contact_planes(model, anchor, trial, info_array, planes, margin=0.0, refresh_normals=True)
    worst, checked = _plane_sides(model, info, planes, anchor)
    test.assertGreater(checked, 0)
    test.assertLessEqual(worst, 0.0)
    rows = info.edge_colliding_edges.numpy()
    valid = planes.ee_valid.numpy()
    for slot in range(int(info.edge_colliding_edges_offsets.numpy()[-1])):
        if int(rows[2 * slot]) > int(rows[2 * slot + 1]):
            test.assertEqual(int(valid[slot]), 0)


def test_truncate_against_planes_keeps_every_vertex_on_its_side(test, device):
    """Commit random displacements through the stored planes without any vertex crossing them."""
    model = _two_layer_model(device)
    _detector, info, info_array = _detect(model)
    planes = create_particle_contact_dat_planes(info, device)
    anchor = wp.clone(model.particle_q)
    rng = np.random.default_rng(3)
    trial_np = model.particle_q.numpy() + rng.uniform(-0.03, 0.03, size=(model.particle_count, 3)).astype(np.float32)
    trial = wp.array(trial_np, dtype=wp.vec3, device=device)
    build_contact_planes(model, anchor, trial, info_array, planes, margin=0.0, refresh_normals=True)
    # negative control: the untruncated trial pose does cross stored planes
    worst_trial, _checked = _plane_sides(model, info, planes, trial)
    test.assertGreater(worst_trial, 0.0)
    sigma = wp.zeros(model.particle_count, dtype=float, device=device)
    truncate_against_contact_planes(model, anchor, trial, info_array, planes, gamma=0.85, sigma=sigma)
    committed = wp.clone(anchor)
    commit_truncated_displacement(model.particle_q, anchor, trial, sigma, budget=1.0, anchor_out=committed)
    worst, checked = _plane_sides(model, info, planes, committed)
    test.assertGreater(checked, 0)
    test.assertLessEqual(worst, 0.0)
    # something did move, and nothing moved past the trial pose
    moved = np.abs(committed.numpy() - anchor.numpy()).max()
    test.assertGreater(moved, 1.0e-4)
    test.assertTrue(np.all(sigma.numpy() <= 1.0) and np.all(sigma.numpy() >= 0.0))


devices = get_test_devices()


class TestParticleDatKernels(unittest.TestCase):
    pass


add_function_test(
    TestParticleDatKernels,
    "test_plane_offset_balances_slowest_vertices",
    test_plane_offset_balances_slowest_vertices,
    devices=devices,
)
add_function_test(
    TestParticleDatKernels,
    "test_plane_offset_gives_all_room_to_the_only_approaching_side",
    test_plane_offset_gives_all_room_to_the_only_approaching_side,
    devices=devices,
)
add_function_test(
    TestParticleDatKernels,
    "test_build_contact_planes_certifies_every_stored_pair",
    test_build_contact_planes_certifies_every_stored_pair,
    devices=devices,
)
add_function_test(
    TestParticleDatKernels,
    "test_truncate_against_planes_keeps_every_vertex_on_its_side",
    test_truncate_against_planes_keeps_every_vertex_on_its_side,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
