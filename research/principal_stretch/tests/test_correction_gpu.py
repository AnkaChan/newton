# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the matrix-free stable-Neo-Hookean correction oracle."""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import torch

from research.principal_stretch.correction_gpu import (
    MatrixFreeCorrectionConfig,
    MatrixFreeStableNHOperator,
    minimum_determinant_on_segment,
    solve_fixed_pcg,
    solve_matrix_free_correction,
)
from research.principal_stretch.newton_baseline import NewtonProblem, build_newton_problem
from research.principal_stretch.solver_benchmark import build_common_problem
from research.principal_stretch.solver_scenes import build_stretch_scene


def _one_tet_problem(
    *,
    mass: np.ndarray | None = None,
    inertial_target: np.ndarray | None = None,
    mu: float = 20.0,
    lam: float = 40.0,
    dt: float = 0.1,
) -> NewtonProblem:
    rest = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return build_newton_problem(
        rest,
        np.array(((0, 1, 2, 3),), dtype=np.int64),
        np.eye(3, dtype=np.float64)[None],
        np.ones(4, dtype=np.float64) if mass is None else mass,
        mu,
        lam,
        dt,
        pinned_indices=np.array((0,), dtype=np.int64),
        pin_targets=rest[:1],
        inertial_target=inertial_target,
    )


def _deformed_positions(problem: NewtonProblem) -> np.ndarray:
    positions = problem.rest_q.numpy().copy()
    positions[1] += (0.12, 0.04, -0.02)
    positions[2] += (-0.03, -0.08, 0.06)
    positions[3] += (0.05, -0.02, 0.15)
    return positions


def _two_tet_problem() -> NewtonProblem:
    rest = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    tets = np.array(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int64)
    poses = []
    for tet in tets:
        corners = rest[tet]
        rest_matrix = np.stack(
            (corners[1] - corners[0], corners[2] - corners[0], corners[3] - corners[0]),
            axis=1,
        )
        poses.append(np.linalg.inv(rest_matrix))
    return build_newton_problem(
        rest,
        tets,
        np.stack(poses),
        np.array((1.0, 1.3, 0.9, 1.1, 1.4), dtype=np.float64),
        np.array((12.0, 31.0), dtype=np.float64),
        np.array((43.0, 67.0), dtype=np.float64),
        0.07,
        pinned_indices=np.array((0,), dtype=np.int64),
        pin_targets=rest[:1],
    )


def _two_tet_deformation(problem: NewtonProblem) -> np.ndarray:
    positions = problem.rest_q.numpy().copy()
    positions += np.array(
        (
            (0.0, 0.0, 0.0),
            (0.07, 0.03, -0.01),
            (-0.02, -0.05, 0.04),
            (0.03, -0.01, 0.08),
            (-0.04, 0.06, 0.02),
        ),
        dtype=np.float64,
    )
    return positions


def _dense_gauss_newton_oracle(operator: MatrixFreeStableNHOperator) -> np.ndarray:
    """Independently assemble the block formula used only by tests."""
    n_dofs = operator.n_free_dofs
    dense = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    free_lookup = {int(vertex): ordinal for ordinal, vertex in enumerate(operator.free)}
    for ordinal, vertex in enumerate(operator.free):
        block_slice = slice(3 * ordinal, 3 * ordinal + 3)
        dense[block_slice, block_slice] += operator.mass[vertex] / operator.dt**2 * identity
    for tet_index, tet in enumerate(operator.tets):
        cofactor = operator.cofactors[tet_index]
        for local_a, vertex_a in enumerate(tet):
            free_a = free_lookup.get(int(vertex_a))
            if free_a is None:
                continue
            ja = operator.shape_gradients[tet_index, local_a]
            cofactor_ja = cofactor @ ja
            row = slice(3 * free_a, 3 * free_a + 3)
            for local_b, vertex_b in enumerate(tet):
                free_b = free_lookup.get(int(vertex_b))
                if free_b is None:
                    continue
                jb = operator.shape_gradients[tet_index, local_b]
                cofactor_jb = cofactor @ jb
                block = operator.volumes[tet_index] * (
                    operator.mu[tet_index] * float(np.dot(ja, jb)) * identity
                    + operator.lam[tet_index] * np.outer(cofactor_ja, cofactor_jb)
                )
                column = slice(3 * free_b, 3 * free_b + 3)
                dense[row, column] += block
    return dense


class TestMatrixFreeStableNHOperator(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_exact_gradient_matches_common_objective_autograd(self):
        problem = _one_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
        z = problem.free_from_positions(operator.positions).requires_grad_(True)
        value = problem.objective_free(z)
        (expected,) = torch.autograd.grad(value, z)

        np.testing.assert_allclose(operator.objective(), float(value.detach()), rtol=2.0e-14, atol=2.0e-14)
        np.testing.assert_allclose(operator.gradient_free(), expected.numpy(), rtol=2.0e-13, atol=2.0e-13)

    def test_matrix_free_product_matches_dense_block_oracle(self):
        problem = _one_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
        dense = _dense_gauss_newton_oracle(operator)
        direction = np.random.default_rng(17).normal(size=operator.n_free_dofs)

        np.testing.assert_allclose(operator.apply_free(direction), dense @ direction, rtol=3.0e-14, atol=3.0e-14)
        np.testing.assert_allclose(dense, dense.T, rtol=0.0, atol=2.0e-14)
        self.assertGreater(float(np.linalg.eigvalsh(dense)[0]), 0.0)

        diagonal = operator.block_diagonal()
        expected_diagonal = np.stack(
            [dense[3 * index : 3 * index + 3, 3 * index : 3 * index + 3] for index in range(operator.free.size)]
        )
        np.testing.assert_allclose(diagonal, expected_diagonal, rtol=3.0e-14, atol=3.0e-14)

    def test_pin_elimination_is_exact(self):
        problem = _one_tet_problem()
        positions = _deformed_positions(problem)
        positions[0] = (91.0, -42.0, 7.0)
        with self.assertRaisesRegex(ValueError, "exact pin targets"):
            MatrixFreeStableNHOperator.from_problem(problem, positions)
        positions[0] = problem.pin_targets.numpy()[0]
        operator = MatrixFreeStableNHOperator.from_problem(problem, positions)
        full_direction = np.zeros_like(operator.positions)
        full_direction[operator.free] = 1.0
        product = operator.apply_full(full_direction)
        np.testing.assert_array_equal(product[operator.pinned], 0.0)
        full_direction[operator.pinned] = 1.0
        with self.assertRaisesRegex(ValueError, "pinned direction"):
            operator.apply_full(full_direction)

    def test_operator_remains_finite_and_spd_when_tet_is_inverted(self):
        problem = _one_tet_problem()
        near_flat_positions = problem.rest_q.numpy().copy()
        near_flat_positions[3, 2] = 1.0e-14
        near_flat = MatrixFreeStableNHOperator.from_problem(problem, near_flat_positions)
        self.assertGreater(near_flat.minimum_determinant, 0.0)
        self.assertTrue(np.isfinite(near_flat.gradient_free()).all())

        positions = problem.rest_q.numpy().copy()
        positions[3, 2] = -0.25
        operator = MatrixFreeStableNHOperator.from_problem(problem, positions)
        direction = np.random.default_rng(23).normal(size=operator.n_free_dofs)

        self.assertLess(operator.minimum_determinant, 0.0)
        self.assertTrue(np.isfinite(operator.gradient_free()).all())
        product = operator.apply_free(direction)
        self.assertTrue(np.isfinite(product).all())
        self.assertGreater(float(np.dot(direction, product)), 0.0)
        for block in operator.block_diagonal():
            self.assertGreater(float(np.linalg.eigvalsh(block)[0]), 0.0)

    def test_low_lambda_clamp_and_disabled_material_match_autograd(self):
        low_lambda = _one_tet_problem(mu=2.0e-7, lam=5.0e-7)
        low_operator = MatrixFreeStableNHOperator.from_problem(low_lambda, _deformed_positions(low_lambda))
        z = low_lambda.free_from_positions(low_operator.positions).requires_grad_(True)
        value = low_lambda.objective_free(z)
        (expected_gradient,) = torch.autograd.grad(value, z)
        np.testing.assert_allclose(low_operator.objective(), float(value.detach()), rtol=2.0e-13, atol=2.0e-13)
        np.testing.assert_allclose(low_operator.gradient_free(), expected_gradient.numpy(), rtol=3.0e-13, atol=3.0e-13)

        disabled = _one_tet_problem(mu=0.0, lam=0.0)
        disabled_operator = MatrixFreeStableNHOperator.from_problem(disabled, _deformed_positions(disabled))
        expected = (
            disabled.mass[disabled.free, None].numpy()
            * (disabled_operator.positions[disabled.free] - disabled.inertial_target[disabled.free].numpy())
            / disabled.dt**2
        ).reshape(-1)
        np.testing.assert_allclose(disabled_operator.gradient_free(), expected, rtol=0.0, atol=2.0e-14)
        direction = np.random.default_rng(31).normal(size=disabled_operator.n_free_dofs)
        diagonal = np.repeat(disabled.mass[disabled.free].numpy(), 3) / disabled.dt**2
        np.testing.assert_allclose(
            disabled_operator.apply_free(direction), diagonal * direction, rtol=0.0, atol=2.0e-14
        )
        self.assertTrue(np.isfinite(disabled_operator.minimum_determinant))

    def test_segment_minimum_catches_positive_endpoint_inversion_crossing(self):
        problem = _one_tet_problem()
        start = MatrixFreeStableNHOperator.from_problem(problem, problem.rest_q)
        end_positions = problem.rest_q.numpy().copy()
        end_positions[1] = (-2.0, 0.0, 0.0)
        end_positions[2] = (0.0, -0.5, 0.0)
        end = MatrixFreeStableNHOperator.from_problem(problem, end_positions)

        self.assertGreater(start.minimum_determinant, 0.0)
        self.assertGreater(end.minimum_determinant, 0.0)
        segment = minimum_determinant_on_segment(start, end)
        self.assertLess(segment.determinant, 0.0)
        self.assertGreater(segment.fraction, 0.0)
        self.assertLess(segment.fraction, 1.0)

    def test_shared_vertex_multi_tet_gradient_action_and_diagonal_oracle(self):
        problem = _two_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _two_tet_deformation(problem))
        z = problem.free_from_positions(operator.positions).requires_grad_(True)
        value = problem.objective_free(z)
        (expected_gradient,) = torch.autograd.grad(value, z)
        dense = _dense_gauss_newton_oracle(operator)
        direction = np.random.default_rng(37).normal(size=operator.n_free_dofs)

        np.testing.assert_allclose(operator.gradient_free(), expected_gradient.numpy(), rtol=3.0e-13, atol=3.0e-13)
        np.testing.assert_allclose(operator.apply_free(direction), dense @ direction, rtol=4.0e-14, atol=4.0e-14)
        expected_diagonal = np.stack(
            [dense[3 * index : 3 * index + 3, 3 * index : 3 * index + 3] for index in range(operator.free.size)]
        )
        np.testing.assert_allclose(operator.block_diagonal(), expected_diagonal, rtol=4.0e-14, atol=4.0e-14)

        rhs = -operator.gradient_free()
        pcg = solve_fixed_pcg(operator, rhs, operator.n_free_dofs)
        direct = np.linalg.solve(dense, rhs)
        self.assertTrue(pcg.success, pcg.deterministic_record())
        np.testing.assert_allclose(pcg.solution, direct, rtol=2.0e-10, atol=2.0e-12)
        expected_true_residual = float(np.linalg.norm(rhs - dense @ pcg.solution))
        self.assertAlmostEqual(pcg.true_residual_norm, expected_true_residual, places=12)


class TestFixedPCG(unittest.TestCase):
    def test_fixed_budget_has_exact_primitive_work(self):
        problem = _one_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
        result = solve_fixed_pcg(operator, -operator.gradient_free(), iterations=2)

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.preconditioner_identity, "block-jacobi-3x3-v1")
        self.assertEqual(result.completed_iterations, 2)
        self.assertEqual(len(result.trace), 2)
        self.assertEqual(result.work.operator_applications, 3)
        self.assertEqual(result.work.residual_verification_applications, 1)
        self.assertEqual(result.work.preconditioner_builds, 1)
        self.assertEqual(result.work.preconditioner_applications, 2)
        self.assertEqual(result.work.inner_products, 4)
        self.assertEqual(result.work.vector_updates, 5)
        dense = _dense_gauss_newton_oracle(operator)
        true_residual = np.linalg.norm(-operator.gradient_free() - dense @ result.solution)
        self.assertAlmostEqual(result.true_residual_norm, true_residual, places=13)
        self.assertFalse(result.solution.flags["W"])

    def test_breakdown_returns_failure_record(self):
        problem = _one_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))

        def zero_preconditioner(residual: np.ndarray) -> np.ndarray:
            return np.zeros_like(residual)

        result = solve_fixed_pcg(
            operator,
            -operator.gradient_free(),
            3,
            zero_preconditioner,
            "test-zero-preconditioner-v1",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonpositive_preconditioner")
        self.assertEqual(result.completed_iterations, 0)
        self.assertEqual(result.work.operator_applications, 0)
        self.assertEqual(result.work.preconditioner_applications, 1)
        self.assertEqual(result.work.inner_products, 1)

    def test_nonfinite_preconditioner_is_a_failure_record(self):
        problem = _one_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))

        def nonfinite_preconditioner(residual: np.ndarray) -> np.ndarray:
            return np.full_like(residual, np.nan)

        result = solve_fixed_pcg(
            operator,
            -operator.gradient_free(),
            2,
            nonfinite_preconditioner,
            "test-nonfinite-preconditioner-v1",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "preconditioner_failure")
        self.assertEqual(result.work.preconditioner_applications, 1)

    def test_finite_huge_rhs_norm_overflow_is_a_failure_record(self):
        problem = _one_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
        rhs = np.full(operator.n_free_dofs, 1.0e308, dtype=np.float64)

        result = solve_fixed_pcg(operator, rhs, 2)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "rhs_norm_overflow")
        self.assertIsNone(result.rhs_norm)
        self.assertIsNone(result.recursive_residual_norm)
        self.assertEqual(result.completed_iterations, 0)


class TestMatrixFreeCorrection(unittest.TestCase):
    def test_fixed_alpha_accepts_descent_and_records_work(self):
        problem = _one_tet_problem()
        initial = _deformed_positions(problem)
        config = MatrixFreeCorrectionConfig(pcg_iterations=3, alpha=0.5)
        result = solve_matrix_free_correction(problem, initial, config)

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")
        self.assertLess(result.final_objective, result.initial_objective)
        self.assertLessEqual(
            result.candidate_objective,
            result.initial_objective + config.armijo * config.alpha * result.directional_derivative,
        )
        self.assertGreater(result.final_minimum_determinant, 0.0)
        np.testing.assert_array_equal(result.x[problem.pinned.numpy()], problem.pin_targets.numpy())
        self.assertEqual(result.work.operator_builds, 2)
        self.assertEqual(result.work.objective_evaluations, 2)
        self.assertEqual(result.work.gradient_evaluations, 2)
        self.assertEqual(result.work.candidate_evaluations, 1)
        self.assertEqual(result.work.pcg, result.pcg.work)
        self.assertFalse(result.x.flags["W"])
        self.assertFalse(config.deterministic_record()["performance_evidence"])

    def test_inverting_candidate_falls_back_without_hidden_retry(self):
        rest = _one_tet_problem().rest_q.numpy()
        target = rest.copy()
        target[3, 2] = -1.0
        problem = _one_tet_problem(mass=np.full(4, 1.0e8), inertial_target=target, dt=0.1)
        initial = problem.rest_q.numpy().copy()
        result = solve_matrix_free_correction(
            problem,
            initial,
            MatrixFreeCorrectionConfig(pcg_iterations=4, alpha=1.0),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "segment_inversion")
        self.assertTrue(result.used_fallback)
        self.assertIsNotNone(result.candidate_minimum_determinant)
        self.assertLessEqual(result.candidate_minimum_determinant, 0.0)
        np.testing.assert_array_equal(result.x, initial)
        self.assertEqual(result.final_objective, result.initial_objective)
        self.assertEqual(result.work.candidate_evaluations, 1)
        self.assertEqual(result.pcg.requested_iterations, 4)

    def test_positive_endpoint_with_inverted_interior_falls_back(self):
        rest = _one_tet_problem().rest_q.numpy()
        target = rest.copy()
        target[1] = (-2.0, 0.0, 0.0)
        target[2] = (0.0, -0.5, 0.0)
        problem = _one_tet_problem(
            mass=np.full(4, 1.0e6),
            inertial_target=target,
            dt=0.1,
        )
        result = solve_matrix_free_correction(
            problem,
            rest,
            MatrixFreeCorrectionConfig(pcg_iterations=2),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "segment_inversion")
        self.assertGreater(result.initial_minimum_determinant, 0.0)
        self.assertGreater(result.candidate_minimum_determinant, 0.0)
        self.assertLess(result.segment_minimum_determinant, 0.0)
        self.assertGreater(result.segment_minimum_fraction, 0.0)
        self.assertLess(result.segment_minimum_fraction, 1.0)
        np.testing.assert_array_equal(result.x, rest)

    def test_early_algebraic_pcg_convergence_is_work_shortfall(self):
        rest = _one_tet_problem().rest_q.numpy()
        target = rest.copy()
        target[1, 0] += 0.2
        problem = _one_tet_problem(mu=0.0, lam=0.0, inertial_target=target)
        result = solve_matrix_free_correction(
            problem,
            rest,
            MatrixFreeCorrectionConfig(pcg_iterations=2, alpha=0.5),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "pcg_work_shortfall")
        self.assertTrue(result.pcg.success)
        self.assertEqual(result.pcg.reason, "converged_early")
        self.assertFalse(result.pcg.consumed_exact_iteration_count)
        np.testing.assert_array_equal(result.x, rest)

    def test_finite_huge_candidate_overflow_falls_back(self):
        rest = _one_tet_problem().rest_q.numpy()
        target = rest.copy()
        target[1] = (1.0e160, 0.0, 0.0)
        target[2] = (0.0, 1.0e160, 0.0)
        target[3] = (0.0, 0.0, 1.0e160)
        base_problem = _one_tet_problem(
            mass=np.full(4, 1.0e-200),
            mu=1.0e-200,
            lam=1.0e-200,
            dt=1.0,
        )
        problem = dataclasses.replace(base_problem, inertial_target=torch.from_numpy(target.copy()))
        result = solve_matrix_free_correction(
            problem,
            rest,
            MatrixFreeCorrectionConfig(pcg_iterations=1),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "candidate_nonfinite")
        np.testing.assert_array_equal(result.x, rest)
        self.assertEqual(result.work.candidate_evaluations, 1)

    def test_armijo_and_segment_records_reject_forgery(self):
        problem = _one_tet_problem()
        result = solve_matrix_free_correction(
            problem,
            _deformed_positions(problem),
            MatrixFreeCorrectionConfig(pcg_iterations=3, alpha=0.5),
        )
        self.assertTrue(result.accepted)
        with self.assertRaisesRegex(ValueError, "Armijo"):
            dataclasses.replace(
                result,
                candidate_objective=result.initial_objective,
                final_objective=result.initial_objective,
                directional_derivative=-1.0e-300,
            )
        with self.assertRaisesRegex(ValueError, "determinant safety"):
            dataclasses.replace(result, segment_minimum_determinant=-1.0)

    def test_custom_preconditioner_identity_flows_through_correction_record(self):
        problem = _one_tet_problem()

        def identity_preconditioner(residual: np.ndarray) -> np.ndarray:
            return residual.copy()

        with self.assertRaisesRegex(ValueError, "requires.*identity"):
            solve_matrix_free_correction(
                problem,
                _deformed_positions(problem),
                MatrixFreeCorrectionConfig(pcg_iterations=2, alpha=0.5),
                preconditioner=identity_preconditioner,
            )
        result = solve_matrix_free_correction(
            problem,
            _deformed_positions(problem),
            MatrixFreeCorrectionConfig(pcg_iterations=2, alpha=0.5),
            preconditioner=identity_preconditioner,
            preconditioner_identity="test-identity-preconditioner-v1",
        )
        self.assertEqual(result.preconditioner_identity, "test-identity-preconditioner-v1")
        self.assertEqual(result.pcg.preconditioner_identity, result.preconditioner_identity)
        self.assertEqual(
            result.deterministic_record()["pcg"]["preconditioner_identity"],
            result.preconditioner_identity,
        )

    def test_real_pr_stretch_scene_is_finite_pinned_and_descending(self):
        scene = build_stretch_scene(dimensions=(2, 1, 1))
        problem = build_common_problem(scene)
        initial = scene.x_current.copy()
        initial[scene.pinned_indices] = scene.pin_targets
        operator = MatrixFreeStableNHOperator.from_problem(problem, initial)
        direction = np.random.default_rng(29).normal(size=operator.n_free_dofs)

        self.assertTrue(np.isfinite(operator.gradient_free()).all())
        self.assertGreater(float(np.dot(direction, operator.apply_free(direction))), 0.0)
        result = solve_matrix_free_correction(
            problem,
            initial,
            MatrixFreeCorrectionConfig(pcg_iterations=3, alpha=0.5),
        )
        self.assertTrue(result.accepted, result.deterministic_record())
        self.assertLessEqual(result.final_objective, result.initial_objective)
        self.assertGreater(result.final_minimum_determinant, 0.0)
        np.testing.assert_array_equal(result.x[scene.pinned_indices], scene.pin_targets)

    def test_config_and_records_are_frozen(self):
        config = MatrixFreeCorrectionConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.alpha = 0.5
        with self.assertRaises(ValueError):
            MatrixFreeCorrectionConfig(pcg_iterations=0).validate()
        with self.assertRaises(ValueError):
            MatrixFreeCorrectionConfig(alpha=1.1).validate()

        problem = _one_tet_problem()
        operator = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
        correction = solve_matrix_free_correction(
            problem,
            _deformed_positions(problem),
            MatrixFreeCorrectionConfig(pcg_iterations=2, alpha=0.5),
        )
        for array in (operator.positions, operator.tets, operator.cofactors, correction.x, correction.pcg.solution):
            with self.assertRaises(ValueError):
                array.setflags(write=True)


if __name__ == "__main__":
    unittest.main()
