# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check particle elasticity ALM algebra and history preparation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import particle_alm_kernels as alm
from newton._src.solvers.vbd import particle_vbd_kernels as primal
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _coefficients_and_ascent(
    stiffness: wp.array[float], rho: wp.array[float], result: wp.array2d[float], matrices: wp.array[wp.mat33]
):
    i = wp.tid()
    s, k, a = alm.particle_alm_coefficients(stiffness[i], rho[i])
    result[i, 0] = s
    result[i, 1] = k
    result[i, 2] = a
    result[i, 3] = alm.particle_alm_ascent(2.0, 3.0, stiffness[i], rho[i])
    matrices[i] = alm.particle_alm_ascent(wp.mat33(2.0), wp.mat33(3.0), stiffness[i], rho[i])


def _tet_model(*, mu=12.0, lame=18.0, mass=1.0, skew=False, device="cpu"):
    builder = newton.ModelBuilder()
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    if skew:
        points = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 3.0)]
    for p in points:
        builder.add_particle(p, (0.0, 0.0, 0.0), mass)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=mu, k_lambda=lame)
    return builder.finalize(device=device)


@wp.kernel
def _evaluate_tet(
    pos: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    history: alm.ParticleElasticityAlmState,
    force: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    i = wp.tid()
    f, h = primal.evaluate_volumetric_neo_hookean_force_and_hessian_alm(
        0, i, pos, pos, indices, poses[0], materials[0, 0], materials[0, 1], 0.0, 0.1, history
    )
    force[i] = f
    hessian[i] = h


def _tet_force_hessian(model, positions, history):
    pos = wp.array(positions, dtype=wp.vec3, device=model.device)
    force = wp.empty(4, dtype=wp.vec3, device=model.device)
    hessian = wp.empty(4, dtype=wp.mat33, device=model.device)
    wp.launch(
        _evaluate_tet,
        dim=4,
        inputs=[pos, model.tet_indices, model.tet_poses, model.tet_materials, history, force, hessian],
        device=model.device,
    )
    return force.numpy(), hessian.numpy()


def _hinge_model(*, rest=0.0, stiffness=10.0, mass=1.0, device="cpu"):
    builder = newton.ModelBuilder()
    for p in [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]:
        builder.add_particle(wp.vec3(p), wp.vec3(0.0), mass)
    builder.add_edge(0, 1, 2, 3, rest=rest, edge_ke=stiffness, edge_kd=0.0)
    builder.add_spring(2, 3, stiffness, 0.0, 0.0)
    return builder.finalize(device=device)


def _metric_product_ranges(test, device):
    """Preserve representable metrics through overflowing or underflowing intermediate products."""
    minimum = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    maximum = float(np.finfo(np.float32).max)
    for stiffness in (1.0e-20, 1.0e38):
        builder = newton.ModelBuilder()
        for point in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
            builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
        builder.add_triangle(0, 1, 2, tri_ke=stiffness, tri_ka=0.0)
        models = (
            (_tet_model(mu=stiffness, lame=0.0, device=device), {"tet_rho_mu": 3.0, "tet_rho_pressure": 1.0}),
            (_hinge_model(stiffness=stiffness, device=device), {"spring_rho": 0.5, "bend_rho": 0.25}),
            (builder.finalize(device=device), {"tri_rho_stretch": 1.0, "tri_rho_area": 0.5}),
        )
        for model, row_factors in models:
            for scale, dt, inv_mass in (
                (1.0, 0.1, 1.0),
                (1.0e30, 1.0e20, 1.0e-20),
                (1.0e-30, 1.0e-20, 1.0e20),
                (1.0e30, 1.0e-20, 1.0e20),
                (1.0e-30, 1.0e20, 1.0e-20),
            ):
                model.particle_inv_mass.fill_(inv_mass)
                state = alm.create_particle_elasticity_alm_state(model, True, True, scale)
                alm.prepare_particle_elasticity_alm(model, model.particle_q, dt, state)
                # Analytic rest-pose mobilities; reference arithmetic stays double.
                inertia = float(np.float32(scale)) / (float(np.float32(dt)) ** 2 * float(np.float32(inv_mass)))
                for name, factor in row_factors.items():
                    with test.subTest(row=name, stiffness=stiffness, scale=scale, dt=dt, inv_mass=inv_mass):
                        floor = 0.0 if name == "tet_rho_pressure" else 9.0 * float(np.float32(stiffness))
                        expected = np.clip(max(factor * inertia, floor), minimum, maximum)
                        actual = float(getattr(state, name).numpy()[0])
                        test.assertTrue(np.isfinite(actual))
                        test.assertGreater(actual, 0.0)
                        np.testing.assert_allclose(actual, expected, rtol=4.0e-5, atol=minimum)


def _tet_metric_balanced_numerator(test, device):
    """Recover finite tet metrics when scale times the inverse rest determinant overflows."""
    model = _tet_model(mu=1.0, lame=1.0, device=device)
    model.tet_poses.assign(np.array([np.diag([1.0e8, 1.0e8, 1.0])], dtype=np.float32))
    positions = model.particle_q.numpy()
    positions[:, :2] *= 1.0e-8
    pos = wp.array(positions, dtype=wp.vec3, device=device)
    state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0e30)
    alm.prepare_particle_elasticity_alm(model, pos, 1.0e10, state)
    np.testing.assert_allclose(state.tet_rho_mu.numpy(), [4.5e10], rtol=4.0e-5)
    np.testing.assert_allclose(state.tet_rho_pressure.numpy(), [1.5e10], rtol=4.0e-5)


class TestParticleAlmKernels(unittest.TestCase):
    def test_tet_retained_history_rotation_covariance(self):
        """Rotate nonrest tet forces and curvature without rotating retained scalar history."""
        model = _tet_model(skew=True)
        history = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
        rest = model.particle_q.numpy()
        initial = rest @ np.array([[1.1, 0.2, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.3]]).T
        alm.prepare_particle_elasticity_alm(model, wp.array(initial, dtype=wp.vec3, device="cpu"), 0.1, history)
        positions = rest @ np.array([[0.8, 0.3, -0.1], [0.1, 1.2, 0.2], [0.2, 0.0, 1.1]]).T
        alm.update_particle_elasticity_alm(model, wp.array(positions, dtype=wp.vec3, device="cpu"), history)
        force, hessian = _tet_force_hessian(model, positions, history)
        axis = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
        angle = 0.73
        cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        rotation = np.cos(angle) * np.eye(3) + (1.0 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
        rotated_force, rotated_hessian = _tet_force_hessian(model, positions @ rotation.T, history)
        np.testing.assert_allclose(rotated_force, force @ rotation.T, rtol=3.0e-6, atol=5.0e-6)
        np.testing.assert_allclose(rotated_hessian, rotation @ hessian @ rotation.T, rtol=3.0e-6, atol=5.0e-6)

    def test_tet_reduced_energy_derivatives(self):
        """Differentiate minimized stretch and pressure energies with frozen stress histories."""
        model = _tet_model(skew=True)
        rest = model.particle_q.numpy().astype(np.float64)
        inverse = np.linalg.inv((rest[1:] - rest[0]).T)
        volume = np.linalg.det((rest[1:] - rest[0]).T) / 6.0
        positions = rest @ np.array([[0.8, 0.3, -0.1], [0.1, 1.2, 0.2], [0.2, 0.0, 1.1]]).T
        for stretch in (False, True):
            with self.subTest(stretch=stretch):
                history = alm.create_particle_elasticity_alm_state(model, True, stretch, 1.0)
                alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, history)
                # A nonstationary multiplier is essential: its geometric curvature
                # disappears from a matrix row or an incorrectly linearized norm.
                if stretch:
                    history.tet_lambda_mu.fill_(31.0)
                history.tet_lambda_pressure.fill_(-4.7)
                rho_mu = float(history.tet_rho_mu.numpy()[0]) if stretch else 0.0
                rho_p = float(history.tet_rho_pressure.numpy()[0])

                def reduced_row(constraint, stiffness, multiplier, rho):
                    auxiliary = (rho * constraint + multiplier) / (stiffness + rho)
                    gap = constraint - auxiliary
                    return 0.5 * stiffness * auxiliary**2 + multiplier * gap + 0.5 * rho * gap**2

                def energy(points, rho_mu=rho_mu, rho_p=rho_p, stretch=stretch):
                    deformation = (points[1:] - points[0]).T @ inverse
                    norm = np.linalg.norm(deformation)
                    stretch_energy = reduced_row(norm, 12.0, 31.0, rho_mu) if stretch else 6.0 * norm**2
                    pressure_energy = reduced_row(np.linalg.det(deformation) - 1.4, 30.0, -4.7, rho_p)
                    return volume * (stretch_energy + pressure_energy)

                force, hessian = _tet_force_hessian(model, positions, history)
                eps = 1.0e-4
                for vertex in range(4):
                    numerical_force = np.empty(3)
                    numerical_hessian = np.empty((3, 3))
                    for i in range(3):
                        ei = np.zeros_like(positions)
                        ei[vertex, i] = eps
                        numerical_force[i] = -(energy(positions + ei) - energy(positions - ei)) / (2.0 * eps)
                        for j in range(3):
                            ej = np.zeros_like(positions)
                            ej[vertex, j] = eps
                            numerical_hessian[i, j] = (
                                energy(positions + ei + ej)
                                - energy(positions + ei - ej)
                                - energy(positions - ei + ej)
                                + energy(positions - ei - ej)
                            ) / (4.0 * eps**2)
                    np.testing.assert_allclose(force[vertex], numerical_force, rtol=8.0e-6, atol=4.0e-6)
                    np.testing.assert_allclose(hessian[vertex], numerical_hessian, rtol=8.0e-6, atol=4.0e-6)

    def test_tet_fixed_point_matches_original_force(self):
        """Recover the original tet force from seeded scalar rows at a nonrest pose."""
        model = _tet_model(skew=True)
        positions = model.particle_q.numpy() @ np.array([[0.8, 0.3, -0.1], [0.1, 1.2, 0.2], [0.2, 0.0, 1.1]]).T
        original = alm.create_particle_elasticity_alm_state(model, False, True, 1.0)
        expected, _ = _tet_force_hessian(model, positions, original)
        for stretch in (False, True):
            with self.subTest(stretch=stretch):
                history = alm.create_particle_elasticity_alm_state(model, True, stretch, 1.0)
                pos = wp.array(positions, dtype=wp.vec3, device="cpu")
                alm.prepare_particle_elasticity_alm(model, pos, 0.1, history)
                for _ in range(3):
                    force, _ = _tet_force_hessian(model, positions, history)
                    np.testing.assert_allclose(force, expected, rtol=3.0e-6, atol=3.0e-6)
                    alm.update_particle_elasticity_alm(model, pos, history)

    def test_tet_collapsed_stretch_reseeds(self):
        """Retire collapsed norm rows and seed recovered stretch after preparation or ascent."""
        model = _tet_model()
        collapsed = wp.zeros(4, dtype=wp.vec3, device="cpu")
        for collapse_during_update in (False, True):
            with self.subTest(collapse_during_update=collapse_during_update):
                history = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
                alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, history)
                if collapse_during_update:
                    alm.update_particle_elasticity_alm(model, collapsed, history)
                else:
                    alm.prepare_particle_elasticity_alm(model, collapsed, 0.1, history)
                np.testing.assert_array_equal(history.tet_lambda_mu.numpy(), 0.0)
                self.assertEqual(int(history.tet_pending.numpy()[0]) & 1, 1)
                force, hessian = _tet_force_hessian(model, collapsed.numpy(), history)
                self.assertTrue(np.isfinite(force).all())
                self.assertTrue(np.isfinite(hessian).all())
                recovered = wp.array(model.particle_q.numpy() * 1.2, dtype=wp.vec3, device="cpu")
                alm.prepare_particle_elasticity_alm(model, recovered, 0.1, history)
                np.testing.assert_allclose(history.tet_lambda_mu.numpy(), 14.4 * np.sqrt(3.0), rtol=1.0e-6)
                self.assertEqual(int(history.tet_pending.numpy()[0]) & 1, 0)

    def test_small_hinge_stress_tracks_ten_sweeps(self):
        """A light, centimeter-scale hinge must not lose its authored moment."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        size = 0.01
        for point in [(0.0, size, 0.0), (0.0, -size, 0.0), (0.0, 0.0, 0.0), (size, 0.0, 0.0)]:
            builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0e-6)
        builder.add_edge(0, 1, 2, 3, rest=0.0, edge_ke=200.0, edge_kd=0.0)
        model = builder.finalize(device="cpu")
        state = alm.create_particle_elasticity_alm_state(model, True, False, 1.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 1.0 / 600.0, state)
        angle = 0.02
        points = model.particle_q.numpy()
        points[1] = (0.0, -size * np.cos(angle), size * np.sin(angle))
        pos = wp.array(points, dtype=wp.vec3, device="cpu")
        for _ in range(10):
            alm.update_particle_elasticity_alm(model, pos, state)
        expected_moment = -200.0 * size * angle
        self.assertAlmostEqual(float(state.bend_lambda.numpy()[0]), expected_moment, delta=0.01 * abs(expected_moment))

    def test_invalid_triangle_material_is_rejected(self):
        """Enabling ALM must validate membrane coefficients as it does tet coefficients."""
        builder = newton.ModelBuilder()
        for point in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
            builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
        builder.add_triangle(0, 1, 2, tri_ke=12.0, tri_ka=18.0)
        model = builder.finalize(device="cpu")
        for mu, lame in [(-1.0, 18.0), (12.0, -12.0), (np.nan, 18.0), (12.0, np.inf)]:
            with self.subTest(mu=mu, lame=lame):
                materials = model.tri_materials.numpy()
                materials[0, :2] = (mu, lame)
                model.tri_materials.assign(materials)
                with self.assertRaises(ValueError):
                    alm.create_particle_elasticity_alm_state(model, True, False, 1.0)

    def test_coefficients_and_matrix_ascent(self):
        """Preserve compliant recurrence and finite coefficients at extreme ratios."""
        stiffness = np.array([4.0, 4.0, 1.0e30, 1.0e-20, 3.0e38, 0.0, 4.0], dtype=np.float32)
        rho = np.array([2.0, 8.0, 1.0e-20, 1.0e30, 3.0e38, 2.0, 0.0], dtype=np.float32)
        result = wp.zeros((len(rho), 4), dtype=float, device="cpu")
        matrices = wp.zeros(len(rho), dtype=wp.mat33, device="cpu")
        wp.launch(
            _coefficients_and_ascent,
            dim=len(rho),
            inputs=[wp.array(stiffness, device="cpu"), wp.array(rho, device="cpu"), result, matrices],
            device="cpu",
        )
        expected = np.zeros((len(rho), 4), dtype=np.float64)
        for i, (k, r) in enumerate(zip(stiffness.astype(float), rho.astype(float), strict=True)):
            if k > 0.0 and r > 0.0:
                expected[i] = (k / (k + r), k * r / (k + r), r / (k + r), (2.0 * k + 3.0 * k * r) / (k + r))
            else:
                expected[i, 2] = 1.0
        # The stress itself can exceed float32; coefficients must remain finite.
        np.testing.assert_allclose(result.numpy()[:, :3], expected[:, :3], rtol=2.0e-6, atol=1.0e-30)
        np.testing.assert_allclose(result.numpy()[[0, 1, 2, 3, 5, 6], 3], expected[[0, 1, 2, 3, 5, 6], 3], rtol=2.0e-6)
        np.testing.assert_allclose(matrices.numpy()[:2], expected[:2, 3, None, None] * np.ones((2, 3, 3)), rtol=2.0e-6)

    def test_rest_seed_and_stress_metrics(self):
        """Seed canceling tet rest stresses and include rest volume in both metrics."""
        model = _tet_model()
        state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        np.testing.assert_allclose(state.tet_lambda_mu.numpy(), [12.0 * np.sqrt(3.0)])
        np.testing.assert_allclose(state.tet_lambda_pressure.numpy(), [-12.0], atol=2.0e-6)
        np.testing.assert_allclose(state.tet_rho_mu.numpy(), [300.0], rtol=1.0e-6)
        np.testing.assert_allclose(state.tet_rho_pressure.numpy(), [100.0], rtol=1.0e-6)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.05, state)
        np.testing.assert_allclose(state.tet_rho_mu.numpy(), [1200.0], rtol=1.0e-6)
        np.testing.assert_allclose(state.tet_rho_pressure.numpy(), [400.0], rtol=1.0e-6)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 1.0, state)
        np.testing.assert_allclose(state.tet_rho_mu.numpy(), [108.0], rtol=1.0e-6)
        np.testing.assert_allclose(state.tet_rho_pressure.numpy(), [1.0], rtol=1.0e-6)

    def test_skew_tet_metrics_use_inverse_rows(self):
        """Use rows of the inverse rest matrix in tet mobility metrics."""
        model = _tet_model(skew=True)
        model.particle_flags.assign(np.array([0, 1, 0, 0], dtype=np.int32))
        state = alm.create_particle_elasticity_alm_state(model, True, True, 2.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        # V0=1; only vertex 1 moves, with inverse row (.5,-.5,0).
        np.testing.assert_allclose(state.tet_rho_mu.numpy(), [1200.0], rtol=1.0e-6)
        np.testing.assert_allclose(state.tet_rho_pressure.numpy(), [400.0], rtol=1.0e-6)
        # Under F=diag(2,1,1), ||F*w/r||^2=(1+.25)/6; pressure mobility=1.25.
        pos = model.particle_q.numpy()
        pos[:, 0] *= 2.0
        alm.prepare_particle_elasticity_alm(model, wp.array(pos, dtype=wp.vec3, device="cpu"), 0.1, state)
        np.testing.assert_allclose(state.tet_rho_mu.numpy(), [960.0], rtol=1.0e-6)
        np.testing.assert_allclose(state.tet_rho_pressure.numpy(), [160.0], rtol=1.0e-6)

    def test_fixed_pose_tet_recurrence(self):
        """Update tet stress once from retained history at a changed pose."""
        model = _tet_model()
        state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        pos_np = model.particle_q.numpy()
        pos_np[1, 0] = 1.2
        pos = wp.array(pos_np, dtype=wp.vec3, device="cpu")
        alm.prepare_particle_elasticity_alm(model, pos, 0.1, state)
        np.testing.assert_allclose(state.tet_lambda_mu.numpy(), [12.0 * np.sqrt(3.0)])
        rho_mu = state.tet_rho_mu.numpy()[0]
        rho_p = state.tet_rho_pressure.numpy()[0]
        alm.update_particle_elasticity_alm(model, pos, state)
        expected_mu = 12.0 / (12.0 + rho_mu) * (12.0 * np.sqrt(3.0) + rho_mu * np.sqrt(3.44))
        expected_p = 30.0 / (30.0 + rho_p) * (-12.0 + rho_p * -0.2)
        np.testing.assert_allclose(state.tet_lambda_mu.numpy(), [expected_mu], rtol=1.0e-6)
        np.testing.assert_allclose(state.tet_lambda_pressure.numpy(), [expected_p], rtol=1.0e-6)

    def test_inactive_tets_clear_history(self):
        """Retire all-zero and immobile tet rows without invalid divisions."""
        for model in (_tet_model(mu=0.0, lame=0.0), _tet_model(mass=0.0)):
            state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
            state.tet_lambda_mu.fill_(5.0)
            state.tet_lambda_pressure.fill_(5.0)
            alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
            alm.update_particle_elasticity_alm(model, model.particle_q, state)
            np.testing.assert_array_equal(state.tet_lambda_mu.numpy(), 0.0)
            np.testing.assert_array_equal(state.tet_lambda_pressure.numpy(), 0.0)
            np.testing.assert_array_equal(state.tet_rho_pressure.numpy(), 0.0)

    def test_selected_world_reset_reseeds_incoming_pose(self):
        """Reseed selected-world spring history after pose edits with fixed buffers."""
        builder = newton.ModelBuilder()
        for world in range(2):
            builder.begin_world()
            start = builder.particle_count
            builder.add_particle((0.0, float(world), 0.0), (0.0, 0.0, 0.0), 1.0)
            builder.add_particle((1.0, float(world), 0.0), (0.0, 0.0, 0.0), 1.0)
            builder.add_spring(start, start + 1, 10.0, 0.0, 0.0)
            builder.end_world()
        builder.add_spring(2, 0, 10.0, 0.0, 0.0)
        model = builder.finalize(device="cpu")
        state = alm.create_particle_elasticity_alm_state(model, True, False, 1.0)
        ptr = state.spring_lambda.ptr
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        state.spring_lambda.assign(np.array([7.0, 11.0, 13.0], dtype=np.float32))
        mask = wp.array([True, False, False], dtype=wp.bool, device="cpu")
        alm.reset_particle_elasticity_alm(model, mask, state)
        pos_np = model.particle_q.numpy()
        pos_np[1, 0] = 1.3
        pos = wp.array(pos_np, dtype=wp.vec3, device="cpu")
        alm.prepare_particle_elasticity_alm(model, pos, 0.1, state)
        np.testing.assert_allclose(state.spring_lambda.numpy(), [3.0, 11.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(state.spring_rho.numpy(), [90.0, 90.0, 90.0])
        self.assertEqual(state.spring_lambda.ptr, ptr)

    def test_hinge_angle_sign_and_mobility(self):
        """Match the existing raw hinge angle and mobility at a right-angle fold."""
        model = _hinge_model()
        state = alm.create_particle_elasticity_alm_state(model, True, True, 4.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        np.testing.assert_allclose(state.bend_lambda.numpy(), [-5.0 * np.pi], rtol=1.0e-6)
        np.testing.assert_allclose(state.bend_rho.numpy(), [100.0], rtol=1.0e-6)
        pos_np = model.particle_q.numpy()
        pos_np[1] = (0.0, -1.0, 0.0)
        pos = wp.array(pos_np, dtype=wp.vec3, device="cpu")
        alm.update_particle_elasticity_alm(model, pos, state)
        np.testing.assert_allclose(state.bend_lambda.numpy(), [-5.0 * np.pi / 11.0], rtol=1.0e-6)

    def test_hinge_raw_angle_and_inactive_rows(self):
        """Preserve raw elastic angle residuals and clear zero or immobile line rows."""
        model = _hinge_model(rest=6.0)
        state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        np.testing.assert_allclose(state.bend_lambda.numpy(), [-60.0 - 5.0 * np.pi], rtol=1.0e-6)
        for model in (_hinge_model(mass=0.0), _hinge_model(stiffness=0.0)):
            state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
            state.bend_lambda.fill_(5.0)
            state.spring_lambda.fill_(5.0)
            alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
            alm.update_particle_elasticity_alm(model, model.particle_q, state)
            for row in (state.bend_lambda, state.spring_lambda, state.bend_rho, state.spring_rho):
                np.testing.assert_array_equal(row.numpy(), 0.0)

    def test_spring_recurrence_and_metric_scale(self):
        """Scale inertia before the spring floor and advance tension once per sweep."""
        model = _hinge_model()
        state = alm.create_particle_elasticity_alm_state(model, True, True, 4.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        np.testing.assert_allclose(state.spring_rho.numpy(), [200.0], rtol=1.0e-6)
        pos_np = model.particle_q.numpy()
        pos_np[3, 0] = 1.2
        pos = wp.array(pos_np, dtype=wp.vec3, device="cpu")
        alm.update_particle_elasticity_alm(model, pos, state)
        np.testing.assert_allclose(state.spring_lambda.numpy(), [40.0 / 21.0], rtol=1.0e-6)
        alm.update_particle_elasticity_alm(model, pos, state)
        np.testing.assert_allclose(state.spring_lambda.numpy(), [880.0 / 441.0], rtol=1.0e-6)

    def test_invalid_line_materials(self):
        """Reject nonfinite and negative spring or hinge material stiffness."""
        for material in (-1.0, np.nan, np.inf):
            model = _hinge_model()
            model.spring_stiffness.fill_(material)
            with self.assertRaises(ValueError):
                alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
            model = _hinge_model()
            properties = model.edge_bending_properties.numpy()
            properties[:, 0] = material
            model.edge_bending_properties.assign(properties)
            with self.assertRaises(ValueError):
                alm.create_particle_elasticity_alm_state(model, True, True, 1.0)

    def test_extreme_metrics_remain_finite(self):
        """Saturate unrepresentable metrics without retiring valid mobile rows."""
        maximum = float(np.finfo(np.float32).max)
        model = _hinge_model(stiffness=1.0e38)
        state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        self.assertEqual(state.spring_rho.numpy()[0], maximum)
        for model in (_hinge_model(), _tet_model()):
            state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
            alm.prepare_particle_elasticity_alm(model, model.particle_q, 1.0e-25, state)
            for rho in (state.tet_rho_mu, state.tet_rho_pressure, state.spring_rho, state.bend_rho):
                if rho.size:
                    np.testing.assert_array_equal(rho.numpy(), maximum)

    def test_configuration_validation(self):
        """Reject invalid enabled materials and nonpositive or nonfinite metrics."""
        for scale in (0.0, -1.0, np.nan, np.inf, 1.0e40, 1.0e-50):
            with self.assertRaises(ValueError):
                alm.create_particle_elasticity_alm_state(_tet_model(), True, True, scale)
        for mu, lame in ((1.0, -1.0), (-1.0, 2.0), (np.nan, 2.0), (1.0, np.inf)):
            with self.assertRaises(ValueError):
                alm.create_particle_elasticity_alm_state(_tet_model(mu=mu, lame=lame), True, True, 1.0)

    def test_tet_primal_high_stiffness(self):
        """Bound the ALM tet Hessian and preserve rest equilibrium at huge pressure stiffness."""
        self.assertTrue(hasattr(primal, "evaluate_volumetric_neo_hookean_force_and_hessian_alm"))

        @wp.kernel
        def evaluate(
            pos: wp.array[wp.vec3],
            indices: wp.array2d[int],
            poses: wp.array[wp.mat33],
            history: alm.ParticleElasticityAlmState,
            force: wp.array[wp.vec3],
            hessian: wp.array[wp.mat33],
        ):
            i = wp.tid()
            f, h = primal.evaluate_volumetric_neo_hookean_force_and_hessian_alm(
                0, i, pos, pos, indices, poses[0], 12.0, 1.0e25, 0.0, 0.1, history
            )
            force[i] = f
            hessian[i] = h

        model = _tet_model(lame=1.0e25)
        state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        force = wp.empty(4, dtype=wp.vec3, device="cpu")
        hessian = wp.empty(4, dtype=wp.mat33, device="cpu")
        wp.launch(
            evaluate,
            dim=4,
            inputs=[model.particle_q, model.tet_indices, model.tet_poses, state, force, hessian],
            device="cpu",
        )
        np.testing.assert_allclose(force.numpy(), 0.0, atol=1.0e-6)
        self.assertTrue(np.all(np.isfinite(hessian.numpy())))
        self.assertLess(np.linalg.norm(hessian.numpy(), axis=(1, 2)).max(), 100.0)

    def test_spring_primal_curvature(self):
        """Retain negative tangential spring curvature under ALM compression."""
        self.assertTrue(hasattr(primal, "evaluate_spring_force_and_hessian_both_vertices_alm"))

        @wp.kernel
        def evaluate(
            pos: wp.array[wp.vec3],
            indices: wp.array[int],
            rest: wp.array[float],
            stiffness: wp.array[float],
            damping: wp.array[float],
            history: alm.ParticleElasticityAlmState,
            force: wp.array[wp.vec3],
            hessian: wp.array[wp.mat33],
        ):
            _, _, f0, f1, h = primal.evaluate_spring_force_and_hessian_both_vertices_alm(
                0, 0.1, pos, pos, indices, rest, stiffness, damping, history
            )
            force[0] = f0
            force[1] = f1
            hessian[0] = h

        builder = newton.ModelBuilder()
        builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        builder.add_particle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        builder.add_spring(0, 1, 10.0, 0.0, 0.0)
        model = builder.finalize(device="cpu")
        state = alm.create_particle_elasticity_alm_state(model, True, True, 1.0)
        pos = wp.array([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        alm.prepare_particle_elasticity_alm(model, pos, 0.1, state)
        force = wp.empty(2, dtype=wp.vec3, device="cpu")
        hessian = wp.empty(1, dtype=wp.mat33, device="cpu")
        wp.launch(
            evaluate,
            dim=1,
            inputs=[
                pos,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
                state,
                force,
                hessian,
            ],
            device="cpu",
        )
        np.testing.assert_allclose(force.numpy(), [(-5.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
        np.testing.assert_allclose(hessian.numpy(), [np.diag([9.0, -10.0, -10.0])])

    def test_disabled_and_pressure_only_state(self):
        """Keep disabled history empty and pressure ALM active when authored lambda is zero."""
        model = _tet_model(lame=0.0)
        disabled = alm.create_particle_elasticity_alm_state(model, False, True, 1.0)
        self.assertEqual(disabled.tet_lambda_pressure.size, 0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, disabled)
        alm.update_particle_elasticity_alm(model, model.particle_q, disabled)
        alm.reset_particle_elasticity_alm(model, None, disabled)
        state = alm.create_particle_elasticity_alm_state(model, True, False, 1.0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        self.assertEqual(state.tet_lambda_mu.size, 0)
        np.testing.assert_allclose(state.tet_lambda_pressure.numpy(), [-12.0])
        self.assertGreater(state.tet_rho_pressure.numpy()[0], 0.0)


class TestParticleAlmMetricRanges(unittest.TestCase):
    pass


add_function_test(
    TestParticleAlmMetricRanges, "test_metric_product_ranges", _metric_product_ranges, devices=get_test_devices()
)
add_function_test(
    TestParticleAlmMetricRanges,
    "test_tet_metric_balanced_numerator",
    _tet_metric_balanced_numerator,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
