# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VBD solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder
from newton._src.geometry.kernels import SOFT_CONTACT_KIND_EDGE, SOFT_CONTACT_KIND_FACE
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_self_contact_force_norm
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    RigidContactHistory,
    init_body_body_contacts_avbd,
    snapshot_body_body_contact_history,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices(mode="basic")


def _soft_rigid_shape_config():
    cfg = ModelBuilder.ShapeConfig()
    cfg.density = 100.0
    cfg.ke = 1.0e5
    cfg.kd = 1.0e1
    cfg.mu = 0.0
    cfg.has_particle_collision = True
    cfg.margin = 0.0
    return cfg


def _add_pinned_soft_quad(builder):
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[
            wp.vec3(-0.5, -0.5, 0.0),
            wp.vec3(0.5, -0.5, 0.0),
            wp.vec3(-0.5, 0.5, 0.0),
            wp.vec3(0.5, 0.5, 0.0),
        ],
        indices=[0, 1, 2, 1, 3, 2],
        density=0.0,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=1.0e2,
        edge_ke=1.0e2,
        edge_kd=1.0,
        particle_radius=0.02,
    )


def _add_pinned_vertical_triangle(builder):
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[
            wp.vec3(-0.3, 0.0, 0.0),
            wp.vec3(0.3, 0.0, 0.0),
            wp.vec3(-0.3, 0.0, -0.4),
        ],
        indices=[0, 1, 2],
        density=0.0,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=1.0e2,
        edge_ke=1.0e2,
        edge_kd=1.0,
        particle_radius=0.02,
    )


def _box_mesh(hx, hy, hz):
    points = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [-hx, hy, hz],
            [hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0,
            2,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            5,
            7,
            6,
            0,
            1,
            4,
            1,
            5,
            4,
            2,
            6,
            3,
            3,
            6,
            7,
            0,
            4,
            2,
            2,
            4,
            6,
            1,
            3,
            5,
            3,
            7,
            5,
        ],
        dtype=np.int32,
    )
    return newton.Mesh(points, indices)


def _add_example_soft_contact_shape(builder, body, shape_name, cfg):
    radius = 0.05
    if shape_name == "mesh":
        builder.add_shape_mesh(body, mesh=_box_mesh(radius, radius, radius), cfg=cfg)
    elif shape_name == "cone":
        builder.add_shape_cone(body, radius=radius, half_height=radius, cfg=cfg)
    elif shape_name == "sphere":
        builder.add_shape_sphere(body, radius=radius, cfg=cfg)
    elif shape_name == "box":
        builder.add_shape_box(body, hx=radius, hy=radius, hz=radius, cfg=cfg)
    elif shape_name == "capsule":
        builder.add_shape_capsule(body, radius=radius * 0.7, half_height=radius, cfg=cfg)
    elif shape_name == "cylinder":
        builder.add_shape_cylinder(body, radius=radius, half_height=radius * 0.5, cfg=cfg)
    else:
        raise ValueError(shape_name)


def _collide_soft_contacts(builder, device, water_tight_soft_rigid, soft_contact_margin=0.0):
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="explicit", soft_contact_margin=soft_contact_margin, soft_contact_max=256
    )
    contacts = pipeline.contacts()
    state = model.state()

    pipeline.collide(state, contacts, water_tight_soft_rigid=water_tight_soft_rigid)
    soft_count = int(contacts.soft_contact_count.numpy()[0])
    contact_kinds = contacts.soft_contact_kind.numpy()[:soft_count].copy()
    contact_particles = contacts.soft_contact_particle.numpy()[:soft_count].copy()
    return soft_count, contact_kinds, contact_particles


def _run_vbd_step_with_soft_contacts(builder, body, device, water_tight_soft_rigid, soft_contact_margin=0.0):
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e5
    model.soft_contact_kd = 1.0e2
    model.soft_contact_mu = 0.0

    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=8,
        rigid_body_particle_contact_buffer_size=64,
        particle_enable_self_contact=False,
    )
    pipeline = newton.CollisionPipeline(
        model, broad_phase="explicit", soft_contact_margin=soft_contact_margin, soft_contact_max=128
    )
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    pipeline.collide(state_0, contacts, water_tight_soft_rigid=water_tight_soft_rigid)
    soft_count = int(contacts.soft_contact_count.numpy()[0])
    contact_kinds = contacts.soft_contact_kind.numpy()[:soft_count].copy()
    contact_particles = contacts.soft_contact_particle.numpy()[:soft_count].copy()
    initial_z = float(state_0.body_q.numpy()[body][2])

    solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
    final_z = float(state_1.body_q.numpy()[body][2])
    return initial_z, final_z, soft_count, contact_kinds, contact_particles


@wp.kernel
def _eval_self_contact_norm_kernel(
    distances: wp.array[float],
    collision_radius: float,
    k: float,
    dEdD_out: wp.array[float],
    d2E_out: wp.array[float],
):
    i = wp.tid()
    dEdD, d2E = evaluate_self_contact_force_norm(distances[i], collision_radius, k)
    dEdD_out[i] = dEdD
    d2E_out[i] = d2E


def test_self_contact_barrier_c2_at_tau(test, device):
    """Barrier must be C2-continuous at d = tau (= collision_radius / 2).

    The log-barrier region (d_min < d < tau) and the outer linear-penalty
    region (tau <= d < collision_radius) share the boundary d = tau.  For
    C2 continuity both the first derivative (force) and the second
    derivative (Hessian scalar) must agree there.

    Regression for GitHub issue #2154.
    """
    collision_radius = 0.02
    k = 1.0e3
    tau = collision_radius * 0.5
    eps = tau * 1e-5

    distances = wp.array([tau - eps, tau + eps], dtype=float, device=device)
    dEdD_out = wp.zeros(2, dtype=float, device=device)
    d2E_out = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        _eval_self_contact_norm_kernel,
        dim=2,
        inputs=[distances, collision_radius, k, dEdD_out, d2E_out],
        device=device,
    )

    dEdD = dEdD_out.numpy()
    d2E = d2E_out.numpy()

    np.testing.assert_allclose(
        dEdD[0],
        dEdD[1],
        rtol=1e-3,
        err_msg="Self-contact barrier force is not C1-continuous at d = tau",
    )
    np.testing.assert_allclose(
        d2E[0],
        d2E[1],
        rtol=1e-3,
        err_msg="Self-contact barrier Hessian is not C2-continuous at d = tau",
    )


def test_self_contact_barrier_c2_at_d_min(test, device):
    """Barrier must be C2-continuous at d = d_min (= 1e-5).

    The quadratic-extension region (d <= d_min) and the log-barrier region
    (d_min < d < tau) share the boundary d = d_min.  The quadratic is a
    Taylor expansion of the log-barrier at d_min, so both the first and
    second derivatives must match.
    """
    collision_radius = 0.02
    k = 1.0e3
    d_min = 1.0e-5
    eps = d_min * 1e-5

    distances = wp.array([d_min - eps, d_min + eps], dtype=float, device=device)
    dEdD_out = wp.zeros(2, dtype=float, device=device)
    d2E_out = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        _eval_self_contact_norm_kernel,
        dim=2,
        inputs=[distances, collision_radius, k, dEdD_out, d2E_out],
        device=device,
    )

    dEdD = dEdD_out.numpy()
    d2E = d2E_out.numpy()

    np.testing.assert_allclose(
        dEdD[0],
        dEdD[1],
        rtol=1e-3,
        err_msg="Self-contact barrier force is not C1-continuous at d = d_min",
    )
    np.testing.assert_allclose(
        d2E[0],
        d2E[1],
        rtol=1e-3,
        err_msg="Self-contact barrier Hessian is not C2-continuous at d = d_min",
    )


def _rigid_contact_history_restore_from_match_index(test, device):
    """VBD warm-start restores from explicit match_index rows."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([4], dtype=int, device=device)
        shape0 = wp.array([0, 0, 0, 0], dtype=int, device=device)
        shape1 = wp.array([1, 1, 1, 1], dtype=int, device=device)
        point0_in = np.array(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        point1_in = point0_in + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        point0 = wp.array(point0_in, dtype=wp.vec3, device=device)
        point1 = wp.array(point1_in, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)

        shape_ke = wp.array([100.0, 200.0], dtype=float, device=device)
        shape_kd = wp.array([1.0, 3.0], dtype=float, device=device)
        shape_mu = wp.array([0.25, 1.0], dtype=float, device=device)
        match_index = wp.array([2, -1, 0, -2], dtype=wp.int32, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 7.0]], dtype=wp.vec3, device=device)
        history.stick_flag = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        history.penalty_k = wp.array([20.0, 30.0, 40.0], dtype=float, device=device)
        history.point0 = wp.array([[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        history.point1 = wp.array([[20.0, 0.0, 1.0], [21.0, 0.0, 1.0], [22.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(4, dtype=float, device=device)
        lam = wp.zeros(4, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(4, dtype=float, device=device)
        material_mu = wp.zeros(4, dtype=float, device=device)
        material_ke = wp.zeros(4, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_avbd,
            dim=4,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                shape_ke,
                shape_kd,
                shape_mu,
                1,
                match_index,
                history,
                10.0,
            ],
            outputs=[
                point0,
                point1,
                penalty_k,
                lam,
                material_kd,
                material_mu,
                material_ke,
            ],
            device=device,
        )

        np.testing.assert_allclose(penalty_k.numpy(), [40.0, 10.0, 20.0, 10.0])
        np.testing.assert_allclose(lam.numpy(), [[0.0, 0.0, 7.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(material_ke.numpy(), [150.0] * 4)
        np.testing.assert_allclose(material_kd.numpy(), [2.0] * 4)
        np.testing.assert_allclose(material_mu.numpy(), [0.5] * 4)

        # slot 2 had DEADZONE, so contact 0 replays saved points. slot 0 was not sticky,
        # so contact 2 keeps the fresh narrow-phase points.
        point0_out = point0.numpy()
        point1_out = point1.numpy()
        np.testing.assert_allclose(point0_out[0], [22.0, 0.0, 0.0])
        np.testing.assert_allclose(point1_out[0], [22.0, 0.0, 1.0])
        np.testing.assert_allclose(point0_out[2], point0_in[2])
        np.testing.assert_allclose(point1_out[2], point1_in[2])
        np.testing.assert_allclose(point0_out[1], point0_in[1])
        np.testing.assert_allclose(point0_out[3], point0_in[3])


def _rigid_contact_history_soft_restores_penalty_only(test, device):
    """Soft contacts restore penalty state only; saved lambda and anchors stay unused."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([1], dtype=int, device=device)
        shape0 = wp.array([0], dtype=int, device=device)
        shape1 = wp.array([1], dtype=int, device=device)
        point0_in = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        point1_in = np.array([[10.0, 0.0, 1.0]], dtype=np.float32)
        point0 = wp.array(point0_in, dtype=wp.vec3, device=device)
        point1 = wp.array(point1_in, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
        history.stick_flag = wp.array([1], dtype=wp.int32, device=device)
        history.penalty_k = wp.array([40.0], dtype=float, device=device)
        history.point0 = wp.array([[20.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        history.point1 = wp.array([[20.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(1, dtype=float, device=device)
        lam = wp.zeros(1, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(1, dtype=float, device=device)
        material_mu = wp.zeros(1, dtype=float, device=device)
        material_ke = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_avbd,
            dim=1,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                wp.array([100.0, 200.0], dtype=float, device=device),
                wp.array([1.0, 3.0], dtype=float, device=device),
                wp.array([0.25, 1.0], dtype=float, device=device),
                0,
                wp.array([0], dtype=wp.int32, device=device),
                history,
                10.0,
            ],
            outputs=[
                point0,
                point1,
                penalty_k,
                lam,
                material_kd,
                material_mu,
                material_ke,
            ],
            device=device,
        )

        np.testing.assert_allclose(penalty_k.numpy(), [40.0])
        np.testing.assert_allclose(lam.numpy(), [[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(point0.numpy(), point0_in)
        np.testing.assert_allclose(point1.numpy(), point1_in)


def _rigid_contact_history_snapshot_copies_active_rows(test, device):
    """Snapshot writes solved state by active contact row and leaves inactive rows untouched."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([2], dtype=int, device=device)
        point0 = wp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        point1 = wp.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        lam = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=wp.vec3, device=device)
        stick = wp.array([1, 2, 3], dtype=wp.int32, device=device)
        penalty = wp.array([10.0, 20.0, 30.0], dtype=float, device=device)

        prev_lambda = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_stick = wp.zeros(3, dtype=wp.int32, device=device)
        prev_penalty = wp.zeros(3, dtype=float, device=device)
        prev_point0 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_point1 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_normal = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.launch(
            snapshot_body_body_contact_history,
            dim=3,
            inputs=[contact_count, point0, point1, normal, lam, stick, penalty],
            outputs=[prev_lambda, prev_stick, prev_penalty, prev_point0, prev_point1, prev_normal],
            device=device,
        )

        np.testing.assert_allclose(prev_lambda.numpy()[:2], lam.numpy()[:2])
        np.testing.assert_allclose(prev_stick.numpy()[:2], [1, 2])
        np.testing.assert_allclose(prev_penalty.numpy()[:2], [10.0, 20.0])
        np.testing.assert_allclose(prev_point0.numpy()[:2], point0.numpy()[:2])
        np.testing.assert_allclose(prev_point1.numpy()[:2], point1.numpy()[:2])
        np.testing.assert_allclose(prev_normal.numpy()[:2], normal.numpy()[:2])
        np.testing.assert_allclose(prev_lambda.numpy()[2], [0.0, 0.0, 0.0])
        test.assertEqual(prev_stick.numpy()[2], 0)
        test.assertEqual(prev_penalty.numpy()[2], 0.0)


def test_water_tight_soft_rigid_face_contact_pushes_body(test, device):
    """A two-triangle soft quad should support a rigid body through face contact."""
    with wp.ScopedDevice(device):
        cfg = _soft_rigid_shape_config()

        def build_scene():
            builder = ModelBuilder(gravity=0.0)
            _add_pinned_soft_quad(builder)
            body = builder.add_body(xform=wp.transform(wp.vec3(0.2, -0.1, 0.04), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
            return builder, body

        legacy_z0, legacy_z1, legacy_count, _legacy_kinds, _legacy_particles = _run_vbd_step_with_soft_contacts(
            *build_scene(), device, False
        )
        watertight_z0, watertight_z1, count, kinds, particles = _run_vbd_step_with_soft_contacts(
            *build_scene(), device, True
        )

        test.assertEqual(legacy_count, 0)
        test.assertLessEqual(abs(legacy_z1 - legacy_z0), 1.0e-6)
        test.assertGreater(count, 0)
        test.assertTrue(np.all(particles == -1))
        test.assertIn(int(SOFT_CONTACT_KIND_FACE), kinds)
        test.assertGreater(watertight_z1, watertight_z0 + 1.0e-4)


def test_water_tight_soft_rigid_face_contact_emits_for_example_shapes(test, device):
    """The example rigid shapes should contact a two-triangle soft quad by face/edge features."""
    with wp.ScopedDevice(device):
        cfg = _soft_rigid_shape_config()
        shape_names = ("mesh", "cone", "sphere", "box", "capsule", "cylinder")

        def build_scene(shape_name):
            builder = ModelBuilder(gravity=0.0)
            _add_pinned_soft_quad(builder)
            body = builder.add_body(xform=wp.transform(wp.vec3(0.2, -0.1, 0.06), wp.quat_identity()))
            _add_example_soft_contact_shape(builder, body, shape_name, cfg)
            return builder

        for shape_name in shape_names:
            legacy_count, _legacy_kinds, _legacy_particles = _collide_soft_contacts(
                build_scene(shape_name), device, False, 0.02
            )
            count, kinds, particles = _collide_soft_contacts(build_scene(shape_name), device, True, 0.02)

            test.assertEqual(legacy_count, 0, shape_name)
            test.assertGreater(count, 0, shape_name)
            test.assertTrue(np.all(particles == -1), shape_name)
            test.assertIn(int(SOFT_CONTACT_KIND_FACE), kinds, shape_name)


def test_water_tight_soft_rigid_edge_contact_pushes_rigid_edge_shapes(test, device):
    """Supported rigid-edge shapes should be pushed out by a soft triangle edge."""
    with wp.ScopedDevice(device):
        cfg = _soft_rigid_shape_config()

        def build_scene(shape_name):
            builder = ModelBuilder(gravity=0.0)
            _add_pinned_vertical_triangle(builder)
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.065), wp.quat_identity()))
            if shape_name == "box":
                builder.add_shape_box(body, hx=0.08, hy=0.005, hz=0.05, cfg=cfg)
            elif shape_name == "mesh":
                builder.add_shape_mesh(body, mesh=_box_mesh(0.08, 0.005, 0.05), cfg=cfg)
            else:
                raise ValueError(shape_name)
            return builder, body

        for shape_name in ("box", "mesh"):
            legacy_z0, legacy_z1, legacy_count, _legacy_kinds, _legacy_particles = _run_vbd_step_with_soft_contacts(
                *build_scene(shape_name), device, False
            )
            watertight_z0, watertight_z1, count, kinds, particles = _run_vbd_step_with_soft_contacts(
                *build_scene(shape_name), device, True, 0.02
            )

            test.assertEqual(legacy_count, 0, shape_name)
            test.assertLessEqual(abs(legacy_z1 - legacy_z0), 1.0e-6, shape_name)
            test.assertGreater(count, 0, shape_name)
            test.assertTrue(np.all(particles == -1), shape_name)
            test.assertIn(int(SOFT_CONTACT_KIND_EDGE), kinds, shape_name)
            test.assertGreater(watertight_z1, watertight_z0 + 1.0e-4, shape_name)


class TestSolverVBD(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBD, "test_self_contact_barrier_c2_at_tau", test_self_contact_barrier_c2_at_tau, devices=devices
)
add_function_test(
    TestSolverVBD, "test_self_contact_barrier_c2_at_d_min", test_self_contact_barrier_c2_at_d_min, devices=devices
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_restore_from_match_index",
    _rigid_contact_history_restore_from_match_index,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_soft_restores_penalty_only",
    _rigid_contact_history_soft_restores_penalty_only,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_snapshot_copies_active_rows",
    _rigid_contact_history_snapshot_copies_active_rows,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_water_tight_soft_rigid_face_contact_pushes_body",
    test_water_tight_soft_rigid_face_contact_pushes_body,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_water_tight_soft_rigid_face_contact_emits_for_example_shapes",
    test_water_tight_soft_rigid_face_contact_emits_for_example_shapes,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_water_tight_soft_rigid_edge_contact_pushes_rigid_edge_shapes",
    test_water_tight_soft_rigid_edge_contact_pushes_rigid_edge_shapes,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
