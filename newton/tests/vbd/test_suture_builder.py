# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authoring of a suture needle: segments, gate bits, tissue parameters and input checks."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices

WIRE_RADIUS = 4.0e-4
THREAD_RADIUS = 1.5e-4
THREAD_LENGTH = 0.01
THREAD_SEGMENTS = 4


def _arc_points(count=5, radius=0.012, angle=0.5 * math.pi):
    """Centerline points of a circular arc in the body frame, tip first, in the x-z plane."""
    points = []
    for i in range(count):
        a = angle * i / (count - 1)
        points.append(wp.vec3(radius * math.sin(a), 0.0, radius * (1.0 - math.cos(a))))
    return points


def _add_needle(builder, centerline):
    """Add a needle body with one capsule per centerline segment and a tip capsule; return (body, tip_shape, shapes)."""
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
    shapes = []
    for a, b in zip(centerline[:-1], centerline[1:]):
        mid = 0.5 * (a + b)
        d = b - a
        q = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), wp.normalize(d))
        shapes.append(
            builder.add_shape_capsule(
                body, xform=wp.transform(mid, q), radius=WIRE_RADIUS, half_height=0.5 * wp.length(d) - WIRE_RADIUS
            )
        )
    tip = builder.add_shape_capsule(
        body, xform=wp.transform(centerline[0], wp.quat_identity()), radius=WIRE_RADIUS, half_height=WIRE_RADIUS
    )
    return body, tip, shapes


def _add_thread(builder, start, direction):
    rod = newton.Rod.create_straight(start, direction, THREAD_LENGTH, segment_count=THREAD_SEGMENTS, radius=THREAD_RADIUS)
    bodies, _joints = builder.add_rod(rod=rod)
    return bodies


def _scene(device):
    builder = newton.ModelBuilder()
    SolverVBD.register_custom_attributes(builder)
    table = builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.005)
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.01),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.01,
        cell_y=0.01,
        cell_z=0.01,
        density=1000.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
    )
    centerline = _arc_points()
    needle, tip, needle_shapes = _add_needle(builder, centerline)
    tail_world = wp.transform_point(builder.body_q[needle], centerline[-1])
    thread_bodies = _add_thread(builder, tail_world, wp.vec3(1.0, 0.0, 0.0))
    return builder, table, needle, tip, needle_shapes, centerline, thread_bodies


def test_add_suture_needle_segments_tip_first(test, device):
    with wp.ScopedDevice(device):
        builder, _table, needle, tip, _shapes, centerline, thread_bodies = _scene(device)
        index = builder.add_suture_needle(needle, tip, centerline=centerline, thread_bodies=thread_bodies)
        test.assertEqual(index, 0)
        model = builder.finalize(skip_validation_inertia=True)
        test.assertEqual(model.suture_needle_count, 1)
        seg_body = model.suture_segment_body.numpy()
        seg_a = model.suture_segment_local_a.numpy()
        seg_b = model.suture_segment_local_b.numpy()
        rest = model.suture_segment_rest_length.numpy()
        start = model.suture_needle_segment_start.numpy()
        n_needle = len(centerline) - 1
        test.assertEqual(start.tolist(), [0, n_needle + THREAD_SEGMENTS])
        np.testing.assert_array_equal(seg_body[:n_needle], needle)
        np.testing.assert_array_equal(seg_body[n_needle:], np.array(thread_bodies))
        for i in range(n_needle):
            np.testing.assert_allclose(seg_a[i], np.array(centerline[i]), atol=1.0e-7)
            np.testing.assert_allclose(seg_b[i], np.array(centerline[i + 1]), atol=1.0e-7)
        np.testing.assert_allclose(model.suture_needle_tip_local.numpy()[0], np.array(centerline[0]), atol=1.0e-7)
        body_q = model.body_q.numpy()
        previous_end = np.array(wp.transform_point(wp.transform(*body_q[needle]), centerline[-1]))
        for k, body in enumerate(thread_bodies):
            s = n_needle + k
            xf = wp.transform(*body_q[body])
            a_world = np.array(wp.transform_point(xf, wp.vec3(*seg_a[s])))
            b_world = np.array(wp.transform_point(xf, wp.vec3(*seg_b[s])))
            test.assertLess(np.linalg.norm(a_world - previous_end), np.linalg.norm(b_world - previous_end))
            previous_end = b_world
        np.testing.assert_allclose(rest, np.linalg.norm(seg_b - seg_a, axis=1), rtol=1.0e-6)


def test_add_suture_needle_sets_shape_gate(test, device):
    with wp.ScopedDevice(device):
        builder, table, needle, tip, shapes, centerline, thread_bodies = _scene(device)
        builder.add_suture_needle(needle, tip, centerline=centerline, thread_bodies=thread_bodies)
        model = builder.finalize(skip_validation_inertia=True)
        gate = model.shape_contact_gate.numpy()
        test.assertEqual(gate.shape, (model.shape_count,))
        test.assertEqual(int(gate[table]), 0)
        for shape in [*shapes, tip]:
            test.assertEqual(int(gate[shape]), 1)
        shape_body = model.shape_body.numpy()
        for body in thread_bodies:
            for shape in np.flatnonzero(shape_body == body):
                test.assertEqual(int(gate[shape]), 1)
        test.assertEqual(int(model.suture_needle_group.numpy()[0]), 1)
        particle_gate = model.particle_contact_gate.numpy()
        test.assertEqual(particle_gate.shape, (model.particle_count,))
        test.assertFalse(particle_gate.any())


def test_add_suture_needle_custom_attribute_defaults(test, device):
    with wp.ScopedDevice(device):
        builder, _table, needle, tip, _shapes, centerline, thread_bodies = _scene(device)
        builder.add_suture_needle(needle, tip, centerline=centerline, thread_bodies=thread_bodies)
        model = builder.finalize(skip_validation_inertia=True)
        count = model.particle_count
        for name, value in (
            ("suture_grip", 1.0),
            ("suture_mu", 0.3),
            ("suture_puncture_force", 0.5),
            ("suture_puncture_cos", 0.5),
        ):
            array = getattr(model.vbd, name).numpy()
            test.assertEqual(array.shape, (count,))
            np.testing.assert_allclose(array, value)


def test_add_suture_needle_rejects_bad_input(test, device):
    with wp.ScopedDevice(device):
        builder, table, needle, tip, _shapes, centerline, thread_bodies = _scene(device)
        with test.assertRaises(IndexError):
            builder.add_suture_needle(needle, table, centerline=centerline, thread_bodies=thread_bodies)
        with test.assertRaises(ValueError):
            builder.add_suture_needle(needle, tip, centerline=[centerline[0], centerline[0], centerline[1]])
        builder.add_suture_needle(needle, tip, centerline=centerline, thread_bodies=thread_bodies)
        for _ in range(30):
            body, tip_k, _s = _add_needle(builder, centerline)
            builder.add_suture_needle(body, tip_k, centerline=centerline)
        body, tip_k, _s = _add_needle(builder, centerline)
        with test.assertRaises(ValueError):
            builder.add_suture_needle(body, tip_k, centerline=centerline)


devices = get_test_devices()


class TestSutureBuilder(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


add_function_test(TestSutureBuilder, "test_add_suture_needle_segments_tip_first", test_add_suture_needle_segments_tip_first, devices=devices)
add_function_test(TestSutureBuilder, "test_add_suture_needle_sets_shape_gate", test_add_suture_needle_sets_shape_gate, devices=devices)
add_function_test(TestSutureBuilder, "test_add_suture_needle_custom_attribute_defaults", test_add_suture_needle_custom_attribute_defaults, devices=devices)
add_function_test(TestSutureBuilder, "test_add_suture_needle_rejects_bad_input", test_add_suture_needle_rejects_bad_input, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
