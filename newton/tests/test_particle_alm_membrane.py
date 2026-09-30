# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent energy, objectivity, and lifecycle checks for membrane ALM."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import particle_alm_kernels as alm
from newton._src.solvers.vbd import particle_vbd_kernels as primal


def _triangle_model(mu=12.0, lame=18.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for point in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
    builder.add_triangle(0, 1, 2, tri_ke=mu, tri_ka=lame, tri_kd=0.0)
    return builder.finalize(device="cpu")


class TestParticleAlmMembrane(unittest.TestCase):
    def _state(self, model):
        self.assertTrue(
            hasattr(primal, "evaluate_neo_hookean_membrane_force_hessian_alm"), "Membrane ALM is not implemented"
        )
        return alm.create_particle_elasticity_alm_state(model, True, False, 1.0)

    def _evaluate(self, model, state, points):
        @wp.kernel
        def evaluate(
            q: wp.array[wp.vec3],
            indices: wp.array2d[int],
            poses: wp.array[wp.mat22],
            materials: wp.array2d[float],
            history: alm.ParticleElasticityAlmState,
            forces: wp.array[wp.vec3],
            hessians: wp.array[wp.mat33],
        ):
            i = wp.tid()
            f, h = primal.evaluate_neo_hookean_membrane_force_hessian_alm(
                0, i, q, q, indices, poses[0], 0.5, materials[0, 0], materials[0, 1], 0.0, 0.1, history
            )
            forces[i] = f
            hessians[i] = h

        q = wp.array(points, dtype=wp.vec3, device="cpu")
        force = wp.empty(3, dtype=wp.vec3, device="cpu")
        hessian = wp.empty(3, dtype=wp.mat33, device="cpu")
        wp.launch(
            evaluate,
            3,
            inputs=[q, model.tri_indices, model.tri_poses, model.tri_materials, state, force, hessian],
            device="cpu",
        )
        return force.numpy(), hessian.numpy()

    def test_fixed_history_energy_derivatives(self):
        """Check force and unclamped Hessian against the eliminated scalar-row energy."""
        model = _triangle_model()
        state = self._state(model)
        state.tri_lambda_stretch.fill_(9.0)
        state.tri_lambda_area.fill_(40.0)
        state.tri_rho_stretch.fill_(7.0)
        state.tri_rho_area.fill_(11.0)
        points = np.array([[0.1, -0.1, 0.2], [1.6, 0.0, 0.3], [0.2, 1.4, 0.1]], dtype=np.float32)
        force, hessian = self._evaluate(model, state, points)

        def energy(q):
            f = np.column_stack((q[1] - q[0], q[2] - q[0]))
            norm = np.linalg.norm(f)
            c_area = np.linalg.norm(np.cross(f[:, 0], f[:, 1])) - 1.4
            return 0.5 * (
                0.5 * (84.0 / 19.0) * norm**2
                + (108.0 / 19.0) * norm
                + 0.5 * (330.0 / 41.0) * c_area**2
                + (1200.0 / 41.0) * c_area
            )

        q = points.astype(np.float64)
        eps = 2e-4
        expected_force = np.zeros((3, 3))
        expected_hessian = np.zeros((3, 3, 3))
        for i in range(3):
            for a in range(3):
                da = np.zeros_like(q)
                da[i, a] = eps
                expected_force[i, a] = -(energy(q + da) - energy(q - da)) / (2 * eps)
                for b in range(3):
                    db = np.zeros_like(q)
                    db[i, b] = eps
                    expected_hessian[i, a, b] = (
                        energy(q + da + db) - energy(q + da - db) - energy(q - da + db) + energy(q - da - db)
                    ) / (4 * eps**2)
        np.testing.assert_allclose(force, expected_force, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(hessian, expected_hessian, atol=4e-5, rtol=2e-5)

    def test_fixed_point_and_retained_history_rotation(self):
        """Recover the authored force and rotate retained history without world-space artifacts."""
        model = _triangle_model()
        state = self._state(model)
        rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        for points in (
            model.particle_q.numpy(),
            np.array([[0, 0, 0], [1.2, 0.1, 0.2], [0.2, 0.8, 0.1]], dtype=np.float32),
        ):
            alm.reset_particle_elasticity_alm(model, None, state)
            alm.prepare_particle_elasticity_alm(model, wp.array(points, dtype=wp.vec3, device="cpu"), 0.1, state)
            force, hessian = self._evaluate(model, state, points)
            q = points.astype(np.float64)
            f0, f1 = q[1] - q[0], q[2] - q[0]
            j = np.linalg.norm(np.cross(f0, f1))
            g0 = (np.dot(f1, f1) * f0 - np.dot(f0, f1) * f1) / j
            g1 = (np.dot(f0, f0) * f1 - np.dot(f0, f1) * f0) / j
            p0 = 12 * f0 + (30 * (j - 1) - 12) * g0
            p1 = 12 * f1 + (30 * (j - 1) - 12) * g1
            np.testing.assert_allclose(force, 0.5 * np.array([p0 + p1, -p0, -p1]), atol=5e-6)
            rotated_force, rotated_hessian = self._evaluate(model, state, points @ rotation.T)
            np.testing.assert_allclose(rotated_force, force @ rotation.T, atol=5e-6)
            np.testing.assert_allclose(rotated_hessian, rotation @ hessian @ rotation.T, atol=2e-5)

    def test_metrics_and_retirement(self):
        """Freeze correctly scaled row metrics and reseed after a collapsed pose."""
        model = _triangle_model(mu=1.0, lame=1.0)
        state = self._state(model)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        np.testing.assert_allclose(state.tri_rho_stretch.numpy(), [100.0], rtol=1e-6)
        np.testing.assert_allclose(state.tri_rho_area.numpy(), [50.0], rtol=1e-6)
        np.testing.assert_allclose(state.tri_lambda_stretch.numpy(), [np.sqrt(2)], rtol=1e-6)
        np.testing.assert_allclose(state.tri_lambda_area.numpy(), [-1.0], rtol=1e-6)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.05, state)
        np.testing.assert_allclose(state.tri_rho_stretch.numpy(), [400.0], rtol=1e-6)
        np.testing.assert_allclose(state.tri_rho_area.numpy(), [200.0], rtol=1e-6)
        frozen_area_rho = state.tri_rho_area.numpy()
        collapsed = wp.zeros(3, dtype=wp.vec3, device="cpu")
        alm.update_particle_elasticity_alm(model, collapsed, state)
        np.testing.assert_array_equal(state.tri_rho_area.numpy(), frozen_area_rho)
        alm.prepare_particle_elasticity_alm(model, collapsed, 0.1, state)
        np.testing.assert_array_equal(state.tri_lambda_stretch.numpy(), [0.0])
        np.testing.assert_array_equal(state.tri_lambda_area.numpy(), [0.0])
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        np.testing.assert_allclose(state.tri_lambda_stretch.numpy(), [np.sqrt(2)], rtol=1e-6)
        np.testing.assert_allclose(state.tri_lambda_area.numpy(), [-1.0], rtol=1e-6)

    def test_selected_world_reset_and_disabled_state(self):
        """Reset triangle rows in place for selected worlds, including global elements."""
        builder = newton.ModelBuilder()
        for world in range(3):
            if world < 2:
                builder.begin_world()
            start = len(builder.particle_q)
            for point in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
                builder.add_particle(wp.vec3(point), wp.vec3(0.0), mass=1.0)
            builder.add_triangle(start, start + 1, start + 2, tri_ke=1.0, tri_ka=1.0)
            if world < 2:
                builder.end_world()
        model = builder.finalize(device="cpu")
        state = self._state(model)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, state)
        state.tri_lambda_area.assign(np.array([7.0, 11.0, 13.0], dtype=np.float32))
        pointers = (state.tri_lambda_stretch.ptr, state.tri_lambda_area.ptr, state.tri_pending.ptr)
        mask = wp.array([True, False, True], dtype=wp.bool, device="cpu")
        alm.reset_particle_elasticity_alm(model, mask, state)
        points = model.particle_q.numpy()
        points[2::3, 1] = 1.5
        alm.prepare_particle_elasticity_alm(model, wp.array(points, dtype=wp.vec3, device="cpu"), 0.1, state)
        np.testing.assert_allclose(state.tri_lambda_area.numpy(), [0.0, 11.0, 0.0], atol=1e-6)
        self.assertEqual(pointers, (state.tri_lambda_stretch.ptr, state.tri_lambda_area.ptr, state.tri_pending.ptr))
        disabled = alm.create_particle_elasticity_alm_state(model, False, False, 1.0)
        self.assertEqual(disabled.tri_lambda_stretch.size, 0)
        self.assertEqual(disabled.tri_lambda_area.size, 0)
        alm.prepare_particle_elasticity_alm(model, model.particle_q, 0.1, disabled)
        alm.update_particle_elasticity_alm(model, model.particle_q, disabled)
        alm.reset_particle_elasticity_alm(model, None, disabled)


if __name__ == "__main__":
    unittest.main()
