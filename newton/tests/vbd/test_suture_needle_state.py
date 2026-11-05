# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The suture needle state machine: puncture, tet walk, tunnel points, exit, lifecycle, gate, capture."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices

DT = 1.0e-3
CELL = 3.0e-3
DIMS = (4, 4, 4)
TOP = DIMS[2] * CELL
TIP_RADIUS = 4.0e-4
TIP_HALF_HEIGHT = 1.0e-3
SPACING = 1.5e-3


def _scene(
    device,
    *,
    armed=False,
    puncture_force=None,
    puncture_cos=None,
    spacing=SPACING,
    max_points=256,
    needle_length=0.03,
    full_surface=False,
    tip_rotation=None,
    start=None,
):
    """A 12 mm tissue cube fixed at its bottom and a kinematic needle above it, tip pointing down."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    SolverVBD.register_custom_attributes(builder)
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=DIMS[0],
        dim_y=DIMS[1],
        dim_z=DIMS[2],
        cell_x=CELL,
        cell_y=CELL,
        cell_z=CELL,
        density=1000.0,
        k_mu=2.0e4,
        k_lambda=2.0e4,
        k_damp=0.0,
        fix_bottom=True,
        particle_radius=5.0e-4,
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    center = wp.vec3(0.5 * DIMS[0] * CELL, 0.5 * DIMS[1] * CELL, TOP + TIP_RADIUS + 1.0e-4) if start is None else start
    rot = wp.quat_identity() if tip_rotation is None else tip_rotation
    needle = builder.add_body(xform=wp.transform(center, rot))
    # The body origin is the tip; the needle extends along local +z (upward), so the tip direction is -z.
    tip = builder.add_shape_capsule(
        needle,
        xform=wp.transform(wp.vec3(0.0, 0.0, TIP_HALF_HEIGHT), wp.quat_identity()),
        radius=TIP_RADIUS,
        half_height=TIP_HALF_HEIGHT,
        cfg=cfg,
    )
    shaft_half = 0.5 * (needle_length - 2.0 * TIP_HALF_HEIGHT)
    builder.add_shape_capsule(
        needle,
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * TIP_HALF_HEIGHT + shaft_half), wp.quat_identity()),
        radius=TIP_RADIUS,
        half_height=shaft_half,
        cfg=cfg,
    )
    builder.add_suture_needle(
        needle,
        tip,
        centerline=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, needle_length)],
        max_points_per_tunnel=max_points,
    )
    builder.color()
    model = builder.finalize(skip_validation_inertia=True)
    if puncture_force is not None:
        model.vbd.suture_puncture_force.fill_(puncture_force)
    if puncture_cos is not None:
        model.vbd.suture_puncture_cos.fill_(puncture_cos)
    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=1.0e-3, enable_rigid_soft_full_surface_contact=full_surface
    )
    solver = SolverVBD(
        model,
        iterations=4,
        collision_pipeline=pipeline,
        suture_armed=armed,
        suture_point_spacing=spacing,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    return model, solver, state_0, state_1, control, needle


def _drive(solver, state_0, state_1, control, needle, poses):
    """Set the kinematic needle pose before each substep and step once."""
    for pose in poses:
        body_q = state_0.body_q.numpy()
        body_q[needle] = pose
        state_0.body_q.assign(body_q)
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def _descend(z_start, z_end, step, rot=None, x=0.5 * DIMS[0] * CELL, y=0.5 * DIMS[1] * CELL):
    rot = wp.quat_identity() if rot is None else rot
    z = z_start
    poses = []
    while z > z_end:
        poses.append(wp.transform(wp.vec3(x, y, z), rot))
        z -= step
    return poses


def _bary_of(model, state, tet, point):
    q = state.particle_q.numpy()
    idx = model.tet_indices.numpy().reshape(-1, 4)[tet]
    a, b, c, d = (q[i] for i in idx)
    mat = np.column_stack([a - d, b - d, c - d])
    w = np.linalg.solve(mat, point - d)
    return np.array([w[0], w[1], w[2], 1.0 - w.sum()])


def test_puncture_requires_force_above_threshold(test, device):
    with wp.ScopedDevice(device):
        for force, expected in ((100.0, 0), (1.0e-4, 1)):
            _model, solver, s0, s1, control, needle = _scene(device, puncture_force=force)
            z0 = TOP + TIP_RADIUS + 1.0e-4
            s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, z0 - 1.0e-3, 2.0e-4))
            test.assertEqual(int(solver.suture_needle_state()[0]), expected)
            if expected == 1:
                tets, barys = solver.suture_tunnel_points(0, 0)
                test.assertEqual(len(tets), 1)
                test.assertTrue((barys[0] >= -1.0e-6).all())
                np.testing.assert_allclose(barys[0].sum(), 1.0, atol=1.0e-6)


def test_puncture_requires_orientation(test, device):
    with wp.ScopedDevice(device):
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.deg2rad(80.0)))
        _model, solver, s0, s1, control, needle = _scene(device, puncture_force=1.0e-4, tip_rotation=rot)
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, z0 - 1.0e-3, 2.0e-4, rot=rot))
        test.assertEqual(int(solver.suture_needle_state()[0]), 0)


def test_armed_mode_punctures_on_contact(test, device):
    with wp.ScopedDevice(device):
        _model, solver, s0, s1, control, needle = _scene(device, armed=True, puncture_force=100.0)
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, z0 - 6.0e-4, 2.0e-4))
        test.assertEqual(int(solver.suture_needle_state()[0]), 1)


def test_tet_walk_appends_points_and_exits(test, device):
    with wp.ScopedDevice(device):
        model, solver, s0, s1, control, needle = _scene(device, armed=True)
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, -2.0e-3, 5.0e-4))
        test.assertEqual(int(solver.suture_needle_state()[0]), 0)  # the tunnel is fixed, the tip is free again
        test.assertEqual(int(solver._suture_tunnel_state.numpy()[0]), 2)
        tets, barys = solver.suture_tunnel_points(0, 0)
        expected = int(np.floor(TOP / SPACING)) + 2
        test.assertLessEqual(abs(len(tets) - expected), 1)
        q = s0.particle_q.numpy()
        idx = model.tet_indices.numpy().reshape(-1, 4)
        for _tet, bary in zip(tets, barys, strict=False):
            test.assertTrue((bary >= -1.0e-5).all())
            np.testing.assert_allclose(bary.sum(), 1.0, atol=1.0e-5)
        last = sum(barys[-1][k] * q[idx[tets[-1]][k]] for k in range(4))
        test.assertLess(abs(last[2]), 1.0e-3)
        first = sum(barys[0][k] * q[idx[tets[0]][k]] for k in range(4))
        test.assertLess(abs(first[2] - TOP), 1.0e-3)


def test_retraction_removes_points(test, device):
    with wp.ScopedDevice(device):
        model, solver, s0, s1, control, needle = _scene(device, armed=True)
        model.vbd.suture_mu.fill_(0.0)  # no friction drag, so the tissue stays put while the needle retracts
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, TOP - 6.0e-3, 5.0e-4))
        tets, barys = solver.suture_tunnel_points(0, 0)
        test.assertGreaterEqual(len(tets), 4)
        q = s0.particle_q.numpy()
        idx = model.tet_indices.numpy().reshape(-1, 4)
        depths = np.array(
            [TOP - sum(b[k] * q[idx[t]][k] for k in range(4))[2] for t, b in zip(tets, barys, strict=False)]
        )
        final_depth = 1.0e-3
        expected_removed = int((depths > final_depth + SPACING).sum())
        test.assertGreaterEqual(expected_removed, 2)
        poses = [
            wp.transform(wp.vec3(0.5 * DIMS[0] * CELL, 0.5 * DIMS[1] * CELL, TOP - z), wp.quat_identity())
            for z in np.arange(6.0e-3, final_depth - 1.0e-6, -5.0e-4)
        ]
        s0, s1 = _drive(solver, s0, s1, control, needle, poses)
        count_back = len(solver.suture_tunnel_points(0, 0)[0])
        test.assertEqual(len(tets) - count_back, expected_removed)


def test_rod_end_finishes_tunnel(test, device):
    with wp.ScopedDevice(device):
        _model, solver, s0, s1, control, needle = _scene(device, armed=True, needle_length=0.03)
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, -0.045, 5.0e-4))
        test.assertEqual(int(solver.suture_needle_state()[0]), 0)
        test.assertEqual(int(solver._suture_tunnel_state.numpy()[0]), 3)


def test_gate_bits_follow_the_tunnel(test, device):
    with wp.ScopedDevice(device):
        model, solver, s0, s1, control, needle = _scene(device, armed=True)
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, -2.0e-3, 5.0e-4))
        gate = model.particle_contact_gate.numpy()
        q = s0.particle_q.numpy()
        axis_x, axis_y = 0.5 * DIMS[0] * CELL, 0.5 * DIMS[1] * CELL
        r = np.hypot(q[:, 0] - axis_x, q[:, 1] - axis_y)
        r_gate = float(solver._suture_gate_radius[0])
        inside = r < r_gate - 1.0e-6
        far = r > 1.0e-2
        test.assertTrue(inside.any())
        test.assertTrue((gate[inside] == 1).all())
        test.assertTrue((gate[far] == 0).all())


def test_point_capacity_sets_overflow(test, device):
    with wp.ScopedDevice(device):
        _model, solver, s0, s1, control, needle = _scene(device, armed=True, max_points=3)
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, -2.0e-3, 5.0e-4))
        test.assertTrue(solver.suture_point_overflow())
        test.assertEqual(len(solver.suture_tunnel_points(0, 0)[0]), 3)


def test_face_record_walk_start(test, device):
    with wp.ScopedDevice(device):
        _model, solver, s0, s1, control, needle = _scene(device, armed=True, full_surface=True)
        z0 = TOP + TIP_RADIUS + 1.0e-4
        s0, s1 = _drive(solver, s0, s1, control, needle, _descend(z0, TOP - 2.0e-3, 5.0e-4))
        test.assertEqual(int(solver.suture_needle_state()[0]), 1)
        tets, barys = solver.suture_tunnel_points(0, 0)
        test.assertGreaterEqual(len(tets), 1)
        test.assertTrue((barys[0] >= -1.0e-5).all())


def test_graph_capture_matches_uncaptured(test, device):
    if not wp.get_device(device).is_cuda:
        return
    with wp.ScopedDevice(device):
        z0 = TOP + TIP_RADIUS + 1.0e-4
        poses = _descend(z0, TOP - 6.0e-3, 5.0e-4)

        model, solver, s0, s1, control, needle = _scene(device, armed=True)
        solver.step(s0, s1, control, None, DT)  # same warm-up step as the captured run
        s0, s1 = model.state(), model.state()
        s0, s1 = _drive(solver, s0, s1, control, needle, poses)
        q_plain = s0.particle_q.numpy().copy()
        tets_plain, barys_plain = solver.suture_tunnel_points(0, 0)

        model, solver, s0, s1, control, needle = _scene(device, armed=True)
        solver.step(s0, s1, control, None, DT)  # load modules before capture
        s0, s1 = model.state(), model.state()
        with wp.ScopedCapture() as capture:
            solver.step(s0, s1, control, None, DT)
            wp.copy(s0.particle_q, s1.particle_q)
            wp.copy(s0.particle_qd, s1.particle_qd)
            wp.copy(s0.body_q, s1.body_q)
            wp.copy(s0.body_qd, s1.body_qd)
        for pose in poses:
            body_q = s0.body_q.numpy()
            body_q[needle] = pose
            s0.body_q.assign(body_q)
            wp.capture_launch(capture.graph)
        q_graph = s0.particle_q.numpy()
        tets_graph, barys_graph = solver.suture_tunnel_points(0, 0)
        np.testing.assert_allclose(q_graph, q_plain, atol=1.0e-6)
        np.testing.assert_array_equal(tets_graph, tets_plain)
        np.testing.assert_allclose(barys_graph, barys_plain, atol=1.0e-6)


devices = get_test_devices()


class TestSutureNeedleState(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


for fn in (
    test_puncture_requires_force_above_threshold,
    test_puncture_requires_orientation,
    test_armed_mode_punctures_on_contact,
    test_tet_walk_appends_points_and_exits,
    test_retraction_removes_points,
    test_rod_end_finishes_tunnel,
    test_gate_bits_follow_the_tunnel,
    test_point_capacity_sets_overflow,
    test_face_record_walk_start,
    test_graph_capture_matches_uncaptured,
):
    add_function_test(TestSutureNeedleState, fn.__name__, fn, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
