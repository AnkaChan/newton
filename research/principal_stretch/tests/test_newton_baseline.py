# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the common stable-NH objective and dense CPU Newton baseline."""

from __future__ import annotations

import dataclasses
import math
import unittest

import numpy as np
import torch
import warp as wp

from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_volumetric_neo_hookean_force_and_hessian

from ..newton_baseline import (
    NewtonConfig,
    NewtonResidualPolishConfig,
    build_newton_problem,
    solve_newton,
    solve_newton_residual_polish,
)
from ..potentials import incremental_potential_stable_neo_hookean, stable_neo_hookean_energy_density


@wp.kernel
def _evaluate_vbd_single_tet(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    dm_inv: wp.array[wp.mat33],
    mu: float,
    lam: float,
    force: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    vertex_order = wp.tid()
    f, h = evaluate_volumetric_neo_hookean_force_and_hessian(
        0,
        vertex_order,
        positions,
        positions,
        tet_indices,
        dm_inv[0],
        mu,
        lam,
        0.0,
        0.1,
    )
    force[vertex_order] = f
    hessian[vertex_order] = h


def _single_tet_data():
    rest = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    dm_inv = np.eye(3, dtype=np.float64)[None]
    return rest, tets, dm_inv


def _two_tet_data():
    rest = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tets = np.array([[0, 1, 2, 3], [4, 2, 1, 3]], dtype=np.int64)
    dm = np.stack(
        [
            np.stack((rest[tet[1]] - rest[tet[0]], rest[tet[2]] - rest[tet[0]], rest[tet[3]] - rest[tet[0]]), axis=1)
            for tet in tets
        ]
    )
    self_check = np.linalg.det(dm)
    if np.any(self_check <= 0.0):
        raise RuntimeError(f"test fixture has non-positive volumes: {self_check}")
    return rest, tets, np.linalg.inv(dm)


def _cofactor(F: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.linalg.cross(F[:, 1], F[:, 2]),
            torch.linalg.cross(F[:, 2], F[:, 0]),
            torch.linalg.cross(F[:, 0], F[:, 1]),
        ),
        dim=1,
    )


class TestStableNeoHookean(unittest.TestCase):
    def _assert_vbd_parity(self, x_np):
        rest, tets, dm_inv = _single_tet_data()
        mu = 17.0
        lam = 31.0

        force_wp = wp.zeros(4, dtype=wp.vec3, device="cpu")
        hessian_wp = wp.zeros(4, dtype=wp.mat33, device="cpu")
        wp.launch(
            _evaluate_vbd_single_tet,
            dim=4,
            inputs=[
                wp.array(x_np, dtype=wp.vec3, device="cpu"),
                wp.array(tets.astype(np.int32), dtype=wp.int32, device="cpu"),
                wp.array(dm_inv.astype(np.float32), dtype=wp.mat33, device="cpu"),
                mu,
                lam,
            ],
            outputs=[force_wp, hessian_wp],
            device="cpu",
        )

        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.ones(4),
            mu=mu,
            lam=lam,
            dt=0.1,
        )
        z = torch.as_tensor(x_np, dtype=torch.float64).reshape(-1).requires_grad_(True)

        def elastic(flat_positions):
            positions = flat_positions.reshape(4, 3)
            F = torch.einsum("tac,tad->tdc", problem.J, positions[problem.tets])
            return stable_neo_hookean_energy_density(F[0], problem.mu[0], problem.lam[0]) * problem.volume[0]

        energy = elastic(z)
        (gradient,) = torch.autograd.grad(energy, z, create_graph=True)
        hessian = torch.autograd.functional.hessian(elastic, z)
        expected_blocks = torch.stack([hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] for i in range(4)])

        np.testing.assert_allclose(
            force_wp.numpy(),
            -gradient.detach().reshape(4, 3).numpy(),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            hessian_wp.numpy(),
            expected_blocks.detach().numpy(),
            rtol=5.0e-5,
            atol=5.0e-5,
        )

    def test_rest_stress_is_zero(self):
        F = torch.eye(3, dtype=torch.float64).requires_grad_(True)
        mu = torch.tensor(5.0e4, dtype=torch.float64)
        lam = torch.tensor(8.0e4, dtype=torch.float64)
        density = stable_neo_hookean_energy_density(F, mu, lam)
        (stress,) = torch.autograd.grad(density, F)
        self.assertLess(stress.abs().max().item(), 1.0e-10)
        # The VBD form retains a constant offset at rest.
        self.assertGreater(density.item(), 0.0)

    def test_first_piola_matches_closed_form(self):
        F = torch.tensor(
            [[1.2, 0.1, -0.2], [0.05, 0.9, 0.1], [0.02, -0.03, 1.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        mu = torch.tensor(17.0, dtype=torch.float64)
        lam = torch.tensor(31.0, dtype=torch.float64)
        (actual,) = torch.autograd.grad(stable_neo_hookean_energy_density(F, mu, lam), F)
        alpha = 1.0 + mu / lam
        expected = mu * F + lam * (torch.linalg.det(F) - alpha) * _cofactor(F)
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_small_lambda_uses_vbd_alpha_floor(self):
        F = torch.diag(torch.tensor([1.1, 0.9, 1.05], dtype=torch.float64)).requires_grad_(True)
        mu = torch.tensor(2.0, dtype=torch.float64)
        lam = torch.tensor(1.0e-8, dtype=torch.float64)
        (actual,) = torch.autograd.grad(stable_neo_hookean_energy_density(F, mu, lam), F)
        alpha = 1.0 + mu / 1.0e-6
        expected = mu * F + lam * (torch.linalg.det(F) - alpha) * _cofactor(F)
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_disabled_tet_has_zero_energy_and_derivatives(self):
        F = torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(9)).requires_grad_(True)
        density = stable_neo_hookean_energy_density(
            F,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )
        (gradient,) = torch.autograd.grad(density, F, create_graph=True)
        hessian = torch.autograd.functional.hessian(
            lambda value: stable_neo_hookean_energy_density(
                value,
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
            ),
            F,
        )
        self.assertEqual(density.item(), 0.0)
        self.assertEqual(gradient.abs().max().item(), 0.0)
        self.assertEqual(hessian.abs().max().item(), 0.0)

    def test_gradient_and_hessian_pass_gradcheck(self):
        F = torch.tensor(
            [[1.1, 0.2, 0.0], [-0.1, 0.95, 0.05], [0.02, 0.03, 1.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        mu = torch.tensor(11.0, dtype=torch.float64)
        lam = torch.tensor(23.0, dtype=torch.float64)

        def energy(value):
            return stable_neo_hookean_energy_density(value, mu, lam)

        self.assertTrue(torch.autograd.gradcheck(energy, (F,), eps=1.0e-6, atol=1.0e-6, rtol=1.0e-5))
        self.assertTrue(torch.autograd.gradgradcheck(energy, (F,), eps=1.0e-6, atol=2.0e-5, rtol=2.0e-4))

    def test_rigid_rotation_preserves_energy(self):
        F = torch.tensor(
            [[1.2, 0.1, 0.0], [0.0, 0.8, 0.2], [0.05, 0.0, 1.1]],
            dtype=torch.float64,
        )
        angle = 0.73
        c = np.cos(angle)
        s = np.sin(angle)
        Q = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        mu = torch.tensor(13.0, dtype=torch.float64)
        lam = torch.tensor(29.0, dtype=torch.float64)
        before = stable_neo_hookean_energy_density(F, mu, lam)
        after = stable_neo_hookean_energy_density(Q @ F, mu, lam)
        self.assertAlmostEqual(before.item(), after.item(), places=12)

    def test_vertex_force_matches_stress_assembly(self):
        rest, tets, dm_inv = _single_tet_data()
        x = torch.tensor(
            [[0.1, -0.2, 0.3], [1.25, 0.0, 0.2], [0.15, 0.85, 0.25], [0.0, -0.1, 1.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.ones(4),
            mu=17.0,
            lam=31.0,
            dt=0.1,
        )
        F = torch.einsum("tac,tad->tdc", problem.J, x[problem.tets])
        elastic = stable_neo_hookean_energy_density(F[0], problem.mu[0], problem.lam[0]) * problem.volume[0]
        (gradient,) = torch.autograd.grad(elastic, x)

        alpha = 1.0 + problem.mu[0] / problem.lam[0]
        P = problem.mu[0] * F[0] + problem.lam[0] * (torch.linalg.det(F[0]) - alpha) * _cofactor(F[0])
        assembled_force = -torch.einsum("dc,ac->ad", P, problem.J[0]) * problem.volume[0]
        torch.testing.assert_close(-gradient, assembled_force, rtol=1.0e-12, atol=1.0e-12)

    def test_force_and_diagonal_hessian_match_vbd_kernel(self):
        wp.init()
        x_np = np.array(
            [[0.1, -0.2, 0.3], [1.25, 0.0, 0.2], [0.15, 0.85, 0.25], [0.0, -0.1, 1.4]],
            dtype=np.float32,
        )
        self._assert_vbd_parity(x_np)

    def test_force_and_hessian_match_vbd_near_flat_and_inverted(self):
        wp.init()
        near_flat = np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.0, 0.9, 0.05], [0.02, -0.03, 1.0e-4]],
            dtype=np.float32,
        )
        inverted = near_flat.copy()
        inverted[3, 2] = -0.4
        exactly_flat = np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.0, 0.9, 0.0], [0.2, -0.3, 0.0]],
            dtype=np.float32,
        )
        self._assert_vbd_parity(near_flat)
        self._assert_vbd_parity(exactly_flat)
        self._assert_vbd_parity(inverted)

    def test_exactly_singular_hessian_is_finite(self):
        F = torch.diag(torch.tensor([1.1, 0.9, 0.0], dtype=torch.float64)).requires_grad_(True)

        def energy(value):
            return stable_neo_hookean_energy_density(
                value,
                torch.tensor(17.0, dtype=torch.float64),
                torch.tensor(31.0, dtype=torch.float64),
            )

        gradient = torch.autograd.functional.jacobian(energy, F)
        hessian = torch.autograd.functional.hessian(energy, F)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.isfinite(hessian).all())


class TestNewtonBaseline(unittest.TestCase):
    def _problem(self, pin_target=None):
        rest, tets, dm_inv = _single_tet_data()
        velocity = np.zeros_like(rest)
        velocity[1] = [1.5, 0.3, -0.2]
        force = np.zeros_like(rest)
        force[2] = [0.0, 4.0, 1.0]
        if pin_target is None:
            pin_target = np.array([[0.15, -0.1, 0.2]], dtype=np.float64)
        return build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.array([0.0, 1.0, 1.2, 0.9]),
            mu=20.0,
            lam=40.0,
            dt=0.08,
            x_current=rest,
            velocity=velocity,
            gravity=(0.0, 0.0, -9.81),
            external_force=force,
            pinned_indices=[0],
            pin_targets=pin_target,
        )

    def test_moving_pin_is_exact(self):
        target = np.array([[0.2, -0.3, 0.4]], dtype=np.float64)
        problem = self._problem(target)
        initial = problem.rest_q.clone()
        initial[0] = torch.tensor([9.0, 9.0, 9.0])
        result = solve_newton(problem, initial)
        self.assertTrue(result.converged, result.reason)
        torch.testing.assert_close(result.x[problem.pinned], torch.as_tensor(target), rtol=0.0, atol=0.0)

    def test_unsorted_pin_targets_follow_their_indices(self):
        rest, tets, dm_inv = _single_tet_data()
        current = rest.copy()
        current[0] = [0.2, 0.3, 0.4]
        current[3] = [-0.2, -0.3, -0.4]
        targets = np.array([[3.0, 3.1, 3.2], [0.1, 0.2, 0.3]], dtype=np.float64)
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.array([0.0, 1.0, 1.0, 0.0]),
            mu=20.0,
            lam=40.0,
            dt=0.1,
            x_current=current,
            pinned_indices=[3, 0],
            pin_targets=targets,
        )
        self.assertEqual(problem.pinned.tolist(), [0, 3])
        torch.testing.assert_close(problem.pin_targets[0], torch.as_tensor(targets[1]), rtol=0.0, atol=0.0)
        torch.testing.assert_close(problem.pin_targets[1], torch.as_tensor(targets[0]), rtol=0.0, atol=0.0)

    def test_default_pin_targets_use_current_positions(self):
        rest, tets, dm_inv = _single_tet_data()
        current = rest.copy()
        current[0] = [0.2, 0.3, 0.4]
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.array([0.0, 1.0, 1.0, 1.0]),
            mu=20.0,
            lam=40.0,
            dt=0.1,
            x_current=current,
            pinned_indices=[0],
        )
        torch.testing.assert_close(problem.pin_targets, torch.as_tensor(current[[0]]), rtol=0.0, atol=0.0)

    def test_converges_monotonically(self):
        problem = self._problem()
        result = solve_newton(
            problem,
            problem.rest_q,
            NewtonConfig(
                max_iterations=30,
                gradient_absolute_tolerance=1.0e-10,
                gradient_relative_tolerance=1.0e-10,
            ),
        )
        self.assertTrue(result.converged, result.reason)
        self.assertLess(result.final_relative_residual, 1.0e-9)
        objective = np.array([item.objective for item in result.trace])
        self.assertTrue(np.all(np.diff(objective) <= 1.0e-11), objective)
        self.assertGreater(result.accepted_iterations, 0)

    def test_initial_stationary_state(self):
        rest, tets, dm_inv = _single_tet_data()
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.ones(4),
            mu=20.0,
            lam=40.0,
            dt=0.08,
            x_current=rest,
        )
        result = solve_newton(problem, rest)
        self.assertTrue(result.converged, result.reason)
        self.assertEqual(result.accepted_iterations, 0)
        self.assertLess(result.final_gradient_norm, 1.0e-12)

    def test_load_only_minimizer_is_inertial_target(self):
        rest, tets, dm_inv = _single_tet_data()
        force = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [0.0, 3.0, 1.0], [-1.0, 0.0, 2.0]])
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.ones(4),
            mu=0.0,
            lam=0.0,
            dt=0.1,
            external_force=force,
            gravity=(0.0, 0.0, -9.81),
        )
        result = solve_newton(problem, rest)
        self.assertTrue(result.converged, result.reason)
        torch.testing.assert_close(result.x, problem.inertial_target, rtol=1.0e-12, atol=1.0e-12)

    def test_force_shifted_target_matches_explicit_load_potential(self):
        rest, tets, dm_inv = _single_tet_data()
        mass = np.array([1.0, 1.2, 0.9, 1.4])
        velocity = np.array(
            [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [0.2, 0.0, 0.1], [-0.3, 0.2, 0.0]],
            dtype=np.float64,
        )
        force = np.array(
            [[1.0, 0.0, -2.0], [0.0, 3.0, 1.0], [-1.0, 2.0, 0.0], [0.5, -0.5, 1.5]],
            dtype=np.float64,
        )
        gravity = np.array([0.0, 0.0, -9.81])
        dt = 0.07
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=mass,
            mu=0.0,
            lam=0.0,
            dt=dt,
            velocity=velocity,
            gravity=gravity,
            external_force=force,
        )
        candidate = torch.as_tensor(rest + 0.03, dtype=torch.float64).reshape(-1).requires_grad_(True)
        mass_t = torch.as_tensor(mass, dtype=torch.float64)
        velocity_t = torch.as_tensor(velocity, dtype=torch.float64)
        force_t = torch.as_tensor(force, dtype=torch.float64)
        gravity_t = torch.as_tensor(gravity, dtype=torch.float64)
        rest_t = torch.as_tensor(rest, dtype=torch.float64)

        def shifted(z):
            return problem.objective_free(z)

        def explicit(z):
            x = z.reshape(4, 3)
            unforced_target = rest_t + dt * velocity_t
            inertia = 0.5 / (dt * dt) * (mass_t[:, None] * (x - unforced_target) ** 2).sum()
            load = -((mass_t[:, None] * gravity_t + force_t) * x).sum()
            return inertia + load

        shifted_gradient = torch.autograd.functional.jacobian(shifted, candidate)
        explicit_gradient = torch.autograd.functional.jacobian(explicit, candidate)
        shifted_hessian = torch.autograd.functional.hessian(shifted, candidate)
        explicit_hessian = torch.autograd.functional.hessian(explicit, candidate)
        torch.testing.assert_close(shifted_gradient, explicit_gradient, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(shifted_hessian, explicit_hessian, rtol=1.0e-12, atol=1.0e-12)

    def test_multitet_hessian_scatter_includes_cross_vertex_blocks(self):
        rest, tets, dm_inv = _two_tet_data()
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.ones(5),
            mu=np.array([17.0, 23.0]),
            lam=np.array([31.0, 37.0]),
            dt=0.1,
        )
        x = torch.as_tensor(rest, dtype=torch.float64)
        x = (x + 0.04 * torch.randn(x.shape, dtype=x.dtype, generator=torch.Generator().manual_seed(4))).reshape(-1)

        def global_elastic(flat_positions):
            positions = flat_positions.reshape(5, 3)
            F = torch.einsum("tac,tad->tdc", problem.J, positions[problem.tets])
            density = stable_neo_hookean_energy_density(F, problem.mu, problem.lam)
            return (density * problem.volume).sum()

        actual = torch.autograd.functional.hessian(global_elastic, x)
        expected = torch.zeros_like(actual)
        for tet_index, tet in enumerate(problem.tets):
            local_x = x.reshape(5, 3)[tet].reshape(-1)

            def local_elastic(
                flat_positions,
                local_j=problem.J[tet_index],
                local_mu=problem.mu[tet_index],
                local_lam=problem.lam[tet_index],
                local_volume=problem.volume[tet_index],
            ):
                positions = flat_positions.reshape(4, 3)
                F = torch.einsum("ac,ad->dc", local_j, positions)
                return stable_neo_hookean_energy_density(F, local_mu, local_lam) * local_volume

            local_hessian = torch.autograd.functional.hessian(local_elastic, local_x)
            dofs = torch.stack((3 * tet, 3 * tet + 1, 3 * tet + 2), dim=1).reshape(-1)
            rows = dofs[:, None].expand(-1, 12)
            cols = dofs[None, :].expand(12, -1)
            expected.index_put_((rows.reshape(-1), cols.reshape(-1)), local_hessian.reshape(-1), accumulate=True)

        torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-11)
        shared_a = int(problem.tets[0, 1])
        shared_b = int(problem.tets[0, 2])
        cross_block = actual[3 * shared_a : 3 * shared_a + 3, 3 * shared_b : 3 * shared_b + 3]
        self.assertGreater(cross_block.abs().max().item(), 1.0e-6)

    def test_iteration_values_are_deterministic(self):
        problem = self._problem()
        config = NewtonConfig(max_iterations=20)
        first = solve_newton(problem, problem.rest_q, config)
        second = solve_newton(problem, problem.rest_q, config)
        self.assertEqual(first.converged, second.converged)
        self.assertEqual(first.reason, second.reason)
        torch.testing.assert_close(first.x, second.x, rtol=0.0, atol=0.0)
        self.assertEqual(len(first.trace), len(second.trace))
        for item_a, item_b in zip(first.trace, second.trace, strict=True):
            self.assertEqual(dataclasses_as_numeric(item_a), dataclasses_as_numeric(item_b))

    def test_residual_scale_is_shared_across_warm_starts(self):
        problem = self._problem()
        config = NewtonConfig(max_iterations=0)
        cold = solve_newton(problem, problem.rest_q, config)
        warm = solve_newton(problem, problem.inertial_target, config)
        self.assertGreater(problem.residual_scale, 0.0)
        self.assertAlmostEqual(warm.final_relative_residual, 1.0, places=12)
        self.assertAlmostEqual(
            cold.final_relative_residual,
            cold.final_gradient_norm / problem.residual_scale,
            places=12,
        )
        self.assertNotAlmostEqual(cold.final_relative_residual, 1.0, places=6)

    def test_nonfinite_start_reports_failure(self):
        problem = self._problem()
        initial = problem.rest_q.clone()
        initial[problem.free] = 1.0e200
        result = solve_newton(problem, initial)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "nonfinite")
        torch.testing.assert_close(result.x[problem.pinned], problem.pin_targets, rtol=0.0, atol=0.0)

        initial[problem.free[0], 0] = torch.nan
        result = solve_newton(problem, initial)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "nonfinite")
        self.assertEqual(len(result.trace), 0)
        self.assertTrue(np.isnan(result.final_objective))
        torch.testing.assert_close(result.x[problem.pinned], problem.pin_targets, rtol=0.0, atol=0.0)

    def test_exactly_flat_start_recovers(self):
        rest, tets, dm_inv = _single_tet_data()
        problem = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.ones(4),
            mu=20.0,
            lam=40.0,
            dt=0.1,
            pinned_indices=[0],
        )
        flat = rest.copy()
        flat[3] = [0.2, -0.3, 0.0]
        result = solve_newton(problem, flat)
        self.assertTrue(result.converged, result.reason)
        self.assertLess(result.final_relative_residual, 1.0e-10)
        objective = np.array([item.objective for item in result.trace])
        self.assertTrue(np.all(np.diff(objective) <= 1.0e-12), objective)

    def test_config_rejects_nonfinite_and_nonintegral_values(self):
        for kwargs in (
            {"gradient_absolute_tolerance": np.nan},
            {"gradient_relative_tolerance": np.inf},
            {"step_relative_tolerance": np.nan},
            {"minimum_eigenvalue_relative": np.nan},
            {"regularization_growth": np.inf},
            {"max_iterations": 2.5},
            {"max_line_search_steps": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NewtonConfig(**kwargs).validate()

    def test_problem_rejects_noninteger_indices_and_mismatched_rest_pose(self):
        rest, tets, dm_inv = _single_tet_data()
        with self.assertRaisesRegex(ValueError, "must contain integers"):
            build_newton_problem(rest, tets.astype(np.float64), dm_inv, np.ones(4), 20.0, 40.0, 0.1)
        with self.assertRaisesRegex(ValueError, "must contain integers"):
            build_newton_problem(
                rest,
                tets,
                dm_inv,
                np.ones(4),
                20.0,
                40.0,
                0.1,
                pinned_indices=np.array([0.0]),
            )
        wrong_pose = dm_inv.copy()
        wrong_pose[0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "do not match"):
            build_newton_problem(rest, tets, wrong_pose, np.ones(4), 20.0, 40.0, 0.1)

    def test_common_potential_rejects_bad_shapes_dtype_and_dt(self):
        problem = self._problem()
        x = problem.rest_q
        values = {
            "x_next": x,
            "inertial_target": problem.inertial_target,
            "mass": problem.mass,
            "tets": problem.tets,
            "J": problem.J,
            "mu": problem.mu,
            "lam": problem.lam,
            "volume": problem.volume,
            "dt": problem.dt,
        }
        for name, invalid in (
            ("mu", torch.ones(1, 1, dtype=torch.float64)),
            ("volume", torch.ones(2, dtype=torch.float64)),
            ("J", torch.ones(1, 3, 3, dtype=torch.float64)),
            ("tets", problem.tets.to(torch.int32)),
            ("mass", torch.ones(4, dtype=torch.float32)),
            ("dt", math.nan),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                kwargs = values | {name: invalid}
                incremental_potential_stable_neo_hookean(**kwargs)

    def test_tiny_step_status_checks_the_new_gradient(self):
        rest, tets, dm_inv = _single_tet_data()
        load_only = build_newton_problem(
            rest,
            tets,
            dm_inv,
            mass=np.ones(4),
            mu=0.0,
            lam=0.0,
            dt=0.1,
            external_force=np.full((4, 3), 0.5),
        )
        success = solve_newton(
            load_only,
            rest,
            NewtonConfig(step_relative_tolerance=10.0),
        )
        self.assertTrue(success.converged, success.reason)
        self.assertEqual(success.reason, "gradient")
        self.assertEqual(success.accepted_iterations, 1)

        nonlinear = solve_newton(
            self._problem(),
            self._problem().rest_q,
            NewtonConfig(
                gradient_absolute_tolerance=0.0,
                gradient_relative_tolerance=0.0,
                step_relative_tolerance=10.0,
            ),
        )
        self.assertFalse(nonlinear.converged)
        self.assertEqual(nonlinear.reason, "stalled")
        self.assertEqual(nonlinear.accepted_iterations, 1)

    def test_work_counters_and_phase_timings_are_complete(self):
        problem = self._problem()
        result = solve_newton(problem, problem.rest_q)
        self.assertTrue(result.converged, result.reason)
        self.assertEqual(result.objective_evaluations, result.gradient_evaluations + result.line_search_trials)
        self.assertEqual(result.hessian_evaluations, result.eigenvalue_evaluations)
        self.assertEqual(result.hessian_evaluations, result.accepted_iterations)
        self.assertGreaterEqual(result.factorization_attempts, result.hessian_evaluations)
        self.assertGreaterEqual(result.line_search_trials, result.accepted_iterations)
        self.assertGreater(result.problem_setup_seconds, 0.0)
        self.assertGreater(result.residual_scale_setup_seconds, 0.0)
        self.assertGreater(result.objective_gradient_seconds, 0.0)
        self.assertGreater(result.hessian_seconds, 0.0)
        self.assertGreater(result.linear_solve_seconds, 0.0)
        self.assertGreater(result.line_search_seconds, 0.0)
        self.assertGreaterEqual(result.end_to_end_seconds, result.total_seconds)

    def test_problem_constants_and_extracted_state_are_detached(self):
        rest, tets, dm_inv = _single_tet_data()
        rest_tensor = torch.as_tensor(rest).requires_grad_(True)
        problem = build_newton_problem(
            rest_tensor,
            tets,
            dm_inv,
            mass=np.ones(4),
            mu=20.0,
            lam=40.0,
            dt=0.1,
        )
        self.assertFalse(problem.rest_q.requires_grad)
        free = problem.free_from_positions(rest_tensor)
        self.assertFalse(free.requires_grad)
        self.assertIsNone(free.grad_fn)

    @staticmethod
    def _scalar_problem(objective):
        class ScalarProblem:
            rest_q = torch.zeros((1, 3), dtype=torch.float64)
            pinned = torch.empty(0, dtype=torch.int64)
            pin_targets = torch.empty((0, 3), dtype=torch.float64)
            residual_scale = 1.0

            @staticmethod
            def free_from_positions(x):
                return torch.as_tensor(x, dtype=torch.float64).reshape(-1).detach().clone()

            @staticmethod
            def positions_from_free(z):
                return z.reshape(1, 3)

            @staticmethod
            def objective_free(z):
                return objective(z)

        return ScalarProblem()

    @staticmethod
    def _polish_config(*, max_line_search_steps=30):
        return NewtonResidualPolishConfig(
            max_iterations=8,
            gradient_absolute_tolerance=1.0e-12,
            gradient_relative_tolerance=1.0e-12,
            armijo=1.0e-4,
            backtrack=0.5,
            max_line_search_steps=max_line_search_steps,
        )

    def test_residual_polish_converges_flat_offset_quadratic_with_exact_accounting(self):
        problem = self._scalar_problem(lambda z: torch.tensor(1.0e16, dtype=z.dtype) + 0.5 * (z * z).sum())
        config = self._polish_config()
        result = solve_newton_residual_polish(
            problem,
            np.array([[1.0e-8, 0.0, 0.0]], dtype=np.float64),
            config,
        )

        self.assertTrue(result.converged, result.reason)
        self.assertEqual(result.reason, "gradient")
        self.assertEqual(result.accepted_iterations, 1)
        self.assertEqual(len(result.trace), 2)
        self.assertLess(result.trace[1].gradient_norm, result.trace[0].gradient_norm)
        self.assertEqual(result.objective_evaluations, result.gradient_evaluations)
        self.assertEqual(result.objective_evaluations, len(result.trace) + result.line_search_trials)
        self.assertEqual(result.hessian_evaluations, result.eigenvalue_evaluations)
        self.assertEqual(result.hessian_evaluations, result.accepted_iterations)
        self.assertEqual(result.factorization_attempts, result.accepted_iterations)
        self.assertGreaterEqual(result.line_search_trials, result.accepted_iterations)
        phase_seconds = (
            result.timing.objective_gradient_seconds
            + result.timing.hessian_seconds
            + result.timing.linear_solve_seconds
            + result.timing.line_search_seconds
        )
        self.assertLessEqual(phase_seconds, result.timing.total_seconds)
        self.assertNotIn("total_seconds", result.deterministic_record())
        self.assertNotIn("elapsed_seconds", result.deterministic_record()["trace"][0])
        self.assertEqual(len(result.timing_record()["trace_elapsed_seconds"]), len(result.trace))
        self.assertIn("8*eps", config.deterministic_record()["objective_roundoff_guard"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.max_iterations = 9
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.trace[0].gradient_norm = 0.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.reason = "max_iterations"

    def test_residual_polish_rejects_non_spd_hessian(self):
        problem = self._scalar_problem(lambda z: -0.5 * z[0] * z[0] + 0.5 * (z[1:] * z[1:]).sum())
        result = solve_newton_residual_polish(
            problem,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            self._polish_config(),
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "non_spd_hessian")
        self.assertEqual(result.accepted_iterations, 0)
        self.assertEqual(result.hessian_evaluations, 1)
        self.assertEqual(result.eigenvalue_evaluations, 1)
        self.assertEqual(result.factorization_attempts, 0)
        self.assertLess(result.trace[0].hessian_minimum_eigenvalue, 0.0)

    def test_residual_polish_nonfinite_and_iteration_limit_are_closed(self):
        problem = self._scalar_problem(lambda z: 0.5 * (z * z).sum())
        nonfinite = solve_newton_residual_polish(
            problem,
            np.array([[math.nan, 0.0, 0.0]], dtype=np.float64),
            self._polish_config(),
        )
        self.assertFalse(nonfinite.converged)
        self.assertEqual(nonfinite.reason, "nonfinite")
        self.assertEqual(nonfinite.trace, ())
        self.assertEqual(nonfinite.objective_evaluations, 0)
        self.assertEqual(nonfinite.gradient_evaluations, 0)
        self.assertEqual(nonfinite.hessian_evaluations, 0)
        self.assertEqual(nonfinite.line_search_trials, 0)

        limited = solve_newton_residual_polish(
            problem,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            dataclasses.replace(self._polish_config(), max_iterations=0),
        )
        self.assertFalse(limited.converged)
        self.assertEqual(limited.reason, "max_iterations")
        self.assertEqual(limited.accepted_iterations, 0)
        self.assertEqual(len(limited.trace), 1)
        self.assertEqual(limited.objective_evaluations, 1)
        self.assertEqual(limited.gradient_evaluations, 1)
        self.assertEqual(limited.hessian_evaluations, 0)
        self.assertEqual(limited.line_search_trials, 0)

    def test_residual_polish_eliminates_pinned_dofs_exactly(self):
        problem = self._problem()
        initial = problem.rest_q.detach().clone()
        initial[problem.pinned] = 123.0
        result = solve_newton_residual_polish(
            problem,
            initial,
            NewtonResidualPolishConfig.from_newton_config(
                NewtonConfig(
                    max_iterations=20,
                    gradient_absolute_tolerance=1.0e-10,
                    gradient_relative_tolerance=1.0e-10,
                )
            ),
        )
        torch.testing.assert_close(result.x[problem.pinned], problem.pin_targets, rtol=0.0, atol=0.0)

    def test_residual_polish_config_validation(self):
        base = dataclasses.asdict(self._polish_config())
        for field, invalid in (
            ("max_iterations", True),
            ("max_iterations", -1),
            ("gradient_absolute_tolerance", -1.0),
            ("gradient_relative_tolerance", math.nan),
            ("armijo", 1.0),
            ("backtrack", 0.0),
            ("max_line_search_steps", 0),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                NewtonResidualPolishConfig(**(base | {field: invalid})).validate()

    def test_residual_polish_result_rejects_status_accounting_and_trace_tamper(self):
        problem = self._scalar_problem(lambda z: 0.5 * (z * z).sum())
        result = solve_newton_residual_polish(
            problem,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            self._polish_config(),
        )
        self.assertTrue(result.converged, result.reason)

        with self.assertRaisesRegex(ValueError, "convergence"):
            dataclasses.replace(result, reason="max_iterations")
        with self.assertRaisesRegex(ValueError, "objective/gradient work"):
            dataclasses.replace(result, objective_evaluations=result.objective_evaluations + 1)
        with self.assertRaisesRegex(ValueError, "trace length"):
            dataclasses.replace(result, accepted_iterations=result.accepted_iterations + 1)
        changed_indices = (
            dataclasses.replace(result.trace[0], iteration=1),
            *result.trace[1:],
        )
        with self.assertRaisesRegex(ValueError, "iteration indices"):
            dataclasses.replace(result, trace=changed_indices)
        changed_eigenvalue = (
            dataclasses.replace(result.trace[0], hessian_minimum_eigenvalue=math.nan),
            *result.trace[1:],
        )
        with self.assertRaisesRegex(ValueError, "finite SPD|trace scalars"):
            dataclasses.replace(result, trace=changed_eigenvalue)
        changed_objective = (
            result.trace[0],
            dataclasses.replace(result.trace[1], objective=result.trace[0].objective + 1.0),
            *result.trace[2:],
        )
        with self.assertRaisesRegex(ValueError, "objective roundoff guard"):
            dataclasses.replace(result, trace=changed_objective)
        with self.assertRaisesRegex(ValueError, "timings"):
            dataclasses.replace(
                result,
                timing=dataclasses.replace(result.timing, total_seconds=-1.0),
            )
        reversed_trace_timings = tuple(reversed(result.timing.trace_elapsed_seconds))
        with self.assertRaisesRegex(ValueError, "trace timings"):
            dataclasses.replace(
                result,
                timing=dataclasses.replace(result.timing, trace_elapsed_seconds=reversed_trace_timings),
            )

    def test_residual_polish_objective_roundoff_guard_fails_closed(self):
        a = -0.26814307801880943
        b = -1.5007641114752346

        def objective(z):
            x = z[0]
            return 0.5 * x * x + a * x**3 + b * x**4 + 0.5 * (z[1:] * z[1:]).sum()

        problem = self._scalar_problem(objective)
        result = solve_newton_residual_polish(
            problem,
            np.array([[0.17132823379228146, 0.0, 0.0]], dtype=np.float64),
            self._polish_config(max_line_search_steps=1),
        )

        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "residual_line_search")
        self.assertEqual(result.accepted_iterations, 0)
        self.assertEqual(result.line_search_trials, 1)
        self.assertEqual(result.trace[0].accepted_step_size, 0.0)
        self.assertEqual(result.objective_evaluations, result.gradient_evaluations)
        self.assertEqual(result.objective_evaluations, len(result.trace) + result.line_search_trials)
        self.assertEqual(result.hessian_evaluations, result.eigenvalue_evaluations)
        self.assertEqual(result.hessian_evaluations, result.accepted_iterations + 1)
        self.assertEqual(result.factorization_attempts, 1)


def dataclasses_as_numeric(item):
    """Return deterministic trace fields, excluding elapsed time."""
    return (
        item.iteration,
        item.objective,
        item.gradient_norm,
        item.relative_residual,
        item.accepted_step_norm,
        item.accepted_step_size,
        item.regularization,
    )


if __name__ == "__main__":
    unittest.main()
