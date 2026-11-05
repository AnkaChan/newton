# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverVBD suture state: tet neighbor table, capacities and the tunnel point spacing rule."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.suture_kernels import build_tet_neighbors
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices


def test_build_tet_neighbors_on_two_tets(test, device):
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    neighbors = build_tet_neighbors(tets)
    test.assertEqual(neighbors.shape, (2, 4))
    # Tet 0 shares face (1, 2, 3), the face opposite its vertex 0, with tet 1.
    test.assertEqual(neighbors[0].tolist(), [1, -1, -1, -1])
    # Tet 1 shares face (1, 2, 3), the face opposite its vertex 4 (local index 3), with tet 0.
    test.assertEqual(neighbors[1].tolist(), [-1, -1, -1, 0])


def _grid_model(device, *, cell=3.0e-3, dims=(2, 2, 2), needle=True, max_tunnels=2, max_points=8):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    SolverVBD.register_custom_attributes(builder)
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dims[0],
        dim_y=dims[1],
        dim_z=dims[2],
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=1000.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
    )
    if needle:
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        tip = builder.add_shape_capsule(body, radius=4.0e-4, half_height=1.0e-3, cfg=cfg)
        builder.add_suture_needle(
            body,
            tip,
            centerline=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.01)],
            max_tunnels=max_tunnels,
            max_points_per_tunnel=max_points,
        )
    builder.color()
    return builder.finalize(skip_validation_inertia=True)


def test_build_tet_neighbors_on_soft_grid(test, device):
    with wp.ScopedDevice(device):
        model = _grid_model(device, needle=False)
        tets = model.tet_indices.numpy().reshape(-1, 4)
        neighbors = build_tet_neighbors(tets)
        for t in range(len(tets)):
            for face in range(4):
                other = neighbors[t, face]
                if other >= 0:
                    test.assertIn(t, neighbors[other].tolist())
        boundary_faces = int((neighbors < 0).sum())
        test.assertEqual(boundary_faces, model.tri_count)


def test_suture_solver_allocates_capacity(test, device):
    with wp.ScopedDevice(device):
        model = _grid_model(device, max_tunnels=2, max_points=8)
        solver = SolverVBD(model, iterations=2)
        test.assertEqual(solver._suture_point_tet.shape, (16,))
        test.assertTrue((solver._suture_point_tet.numpy() == -1).all())
        test.assertTrue(np.isinf(solver._suture_point_stiffness.numpy()).all())
        test.assertEqual(solver._suture_tunnel_state.shape, (2,))
        test.assertEqual(solver._suture_needle_state.numpy().tolist(), [0])
        test.assertEqual(solver._suture_tet_neighbors.shape, (model.tet_count, 4))
        test.assertGreater(solver._suture_gate_radius[0], 4.0e-4)


def test_suture_point_spacing_rule(test, device):
    with wp.ScopedDevice(device):
        model = _grid_model(device, cell=3.0e-3)
        test.assertAlmostEqual(SolverVBD(model, iterations=1)._suture_point_spacing, 0.5 * _mean_edge(model), places=7)
        model = _grid_model(device, cell=0.6e-3)
        test.assertAlmostEqual(SolverVBD(model, iterations=1)._suture_point_spacing, 0.5e-3, places=7)
        model = _grid_model(device, cell=1.0e-2)
        test.assertAlmostEqual(SolverVBD(model, iterations=1)._suture_point_spacing, 2.0e-3, places=7)
        model = _grid_model(device, cell=3.0e-3)
        test.assertAlmostEqual(SolverVBD(model, iterations=1, suture_point_spacing=1.0e-3)._suture_point_spacing, 1.0e-3, places=9)


def _mean_edge(model):
    tets = model.tet_indices.numpy().reshape(-1, 4)
    q = model.particle_q.numpy()
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    lengths = [np.linalg.norm(q[tets[:, a]] - q[tets[:, b]], axis=1) for a, b in pairs]
    return float(np.mean(np.concatenate(lengths)))


devices = get_test_devices()


class TestSutureSolverInit(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


add_function_test(TestSutureSolverInit, "test_build_tet_neighbors_on_two_tets", test_build_tet_neighbors_on_two_tets, devices=devices)
add_function_test(TestSutureSolverInit, "test_build_tet_neighbors_on_soft_grid", test_build_tet_neighbors_on_soft_grid, devices=devices)
add_function_test(TestSutureSolverInit, "test_suture_solver_allocates_capacity", test_suture_solver_allocates_capacity, devices=devices)
add_function_test(TestSutureSolverInit, "test_suture_point_spacing_rule", test_suture_point_spacing_rule, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
