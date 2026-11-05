# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tunnel point rows: transverse hold, one-sidedness for a kinematic rod, and the capped friction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices

DT = 1.0e-3
ROD_RADIUS = 3.0e-4
ROD_HALF_HEIGHT = 1.0e-2
TET_EDGE = 1.0e-2
BARY = (0.25, 0.25, 0.25, 0.25)


def _tet_points():
    """A regular tet centered at the origin with edge TET_EDGE."""
    s = TET_EDGE / np.sqrt(8.0)
    return [
        wp.vec3(s, s, s),
        wp.vec3(s, -s, -s),
        wp.vec3(-s, s, -s),
        wp.vec3(-s, -s, s),
    ]


def _scene(
    device, *, tet_fixed, rod_dynamic, rod_axis_x=0.0, grip=1.0, mu=0.3, ds=1.0e-3, point_spacing=1.0e-3, stiffness=None
):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    SolverVBD.register_custom_attributes(builder)
    mass = 0.0 if tet_fixed else 1.0e-3
    for p in _tet_points():
        builder.add_particle(pos=p, vel=wp.vec3(0.0), mass=mass, radius=1.0e-4)
    builder.add_tetrahedron(0, 2, 1, 3, k_mu=1.0, k_lambda=1.0, k_damp=0.0)
    cfg = newton.ModelBuilder.ShapeConfig(density=17600.0 if rod_dynamic else 0.0)
    rod = builder.add_body(xform=wp.transform(wp.vec3(rod_axis_x, 0.0, 0.0), wp.quat_identity()))
    capsule = builder.add_shape_capsule(rod, radius=ROD_RADIUS, half_height=ROD_HALF_HEIGHT, cfg=cfg)
    builder.add_suture_needle(
        rod, capsule, centerline=[wp.vec3(0.0, 0.0, ROD_HALF_HEIGHT), wp.vec3(0.0, 0.0, -ROD_HALF_HEIGHT)]
    )
    builder.color()
    model = builder.finalize(skip_validation_inertia=True)
    model.vbd.suture_grip.fill_(grip)
    model.vbd.suture_mu.fill_(mu)
    if stiffness is not None:
        model.vbd.suture_stiffness.fill_(stiffness)
    solver = SolverVBD(model, iterations=10, suture_point_spacing=point_spacing)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver.suture_set_points(0, 0, [0], [wp.vec4(*BARY)], state=state_0, ds=ds)
    return model, solver, state_0, state_1, control, rod


def _run(solver, state_0, state_1, control, steps, body_force=None, body=0):
    for _ in range(steps):
        if body_force is not None:
            f = np.zeros((state_0.body_f.shape[0], 6), dtype=np.float32)
            f[body, :3] = body_force
            state_0.body_f.assign(f)
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    return state_0


def _point_axis_distance(model, state, solver, rod):
    q = state.particle_q.numpy()
    tet = model.tet_indices.numpy().reshape(-1, 4)[0]
    x = sum(BARY[i] * q[tet[i]] for i in range(4))
    xf = wp.transform(*state.body_q.numpy()[rod])
    a = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, ROD_HALF_HEIGHT)))
    b = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, -ROD_HALF_HEIGHT)))
    d = b - a
    t = d / np.linalg.norm(d)
    r = x - a
    return float(np.linalg.norm(r - np.dot(r, t) * t))


def test_row_pulls_tissue_point_onto_kinematic_rod(test, device):
    with wp.ScopedDevice(device):
        model, solver, s0, s1, control, rod = _scene(device, tet_fixed=False, rod_dynamic=False, rod_axis_x=2.0e-3)
        body_q0 = s0.body_q.numpy().copy()
        test.assertGreater(_point_axis_distance(model, s0, solver, rod), 1.9e-3)
        state = _run(solver, s0, s1, control, steps=20)
        test.assertLess(_point_axis_distance(model, state, solver, rod), 1.0e-4)
        np.testing.assert_array_equal(state.body_q.numpy(), body_q0)


def test_friction_cap_sticks_below_and_slides_above(test, device):
    with wp.ScopedDevice(device):
        _model, solver, s0, s1, control, rod = _scene(device, tet_fixed=True, rod_dynamic=True)
        z0 = float(s0.body_q.numpy()[rod][2])
        state = _run(solver, s0, s1, control, steps=100, body_force=(0.0, 0.0, 1.0e-4), body=rod)
        test.assertLess(abs(float(state.body_q.numpy()[rod][2]) - z0), 1.0e-5)
        _model, solver, s0, s1, control, rod = _scene(device, tet_fixed=True, rod_dynamic=True)
        state = _run(solver, s0, s1, control, steps=20, body_force=(0.0, 0.0, 1.0e-3), body=rod)
        test.assertGreater(abs(float(state.body_q.numpy()[rod][2]) - z0), 1.0e-4)


def test_friction_zero_grip_slides_freely(test, device):
    with wp.ScopedDevice(device):
        _model, solver, s0, s1, control, rod = _scene(device, tet_fixed=True, rod_dynamic=True, grip=0.0)
        z0 = float(s0.body_q.numpy()[rod][2])
        state = _run(solver, s0, s1, control, steps=20, body_force=(0.0, 0.0, 1.0e-4), body=rod)
        test.assertGreater(abs(float(state.body_q.numpy()[rod][2]) - z0), 1.0e-4)


def test_transverse_multiplier_equals_side_load(test, device):
    with wp.ScopedDevice(device):
        model, solver, s0, s1, control, rod = _scene(device, tet_fixed=True, rod_dynamic=True)
        state = _run(solver, s0, s1, control, steps=200, body_force=(1.0e-3, 0.0, 0.0), body=rod)
        test.assertLess(_point_axis_distance(model, state, solver, rod), 1.0e-4)
        lam = solver._suture_point_lambda.numpy()[0]
        np.testing.assert_allclose(np.linalg.norm(lam), 1.0e-3, rtol=0.05)


def test_finite_stiffness_row_is_compliant(test, device):
    """With vbd.suture_stiffness = k the row carries the side load F at an offset F / k."""
    with wp.ScopedDevice(device):
        model, solver, s0, s1, control, rod = _scene(device, tet_fixed=True, rod_dynamic=True, stiffness=10.0)
        test.assertEqual(float(solver._suture_point_stiffness.numpy()[0]), 10.0)
        state = _run(solver, s0, s1, control, steps=200, body_force=(1.0e-3, 0.0, 0.0), body=rod)
        np.testing.assert_allclose(_point_axis_distance(model, state, solver, rod), 1.0e-3 / 10.0, rtol=0.1)
        lam = solver._suture_point_lambda.numpy()[0]
        np.testing.assert_allclose(np.linalg.norm(lam), 1.0e-3, rtol=0.05)


devices = get_test_devices()


class TestSutureRow(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


add_function_test(
    TestSutureRow,
    "test_row_pulls_tissue_point_onto_kinematic_rod",
    test_row_pulls_tissue_point_onto_kinematic_rod,
    devices=devices,
)
add_function_test(
    TestSutureRow,
    "test_friction_cap_sticks_below_and_slides_above",
    test_friction_cap_sticks_below_and_slides_above,
    devices=devices,
)
add_function_test(
    TestSutureRow, "test_friction_zero_grip_slides_freely", test_friction_zero_grip_slides_freely, devices=devices
)
add_function_test(
    TestSutureRow,
    "test_transverse_multiplier_equals_side_load",
    test_transverse_multiplier_equals_side_load,
    devices=devices,
)
add_function_test(
    TestSutureRow,
    "test_finite_stiffness_row_is_compliant",
    test_finite_stiffness_row_is_compliant,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
