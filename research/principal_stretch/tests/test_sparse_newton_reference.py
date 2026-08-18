# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the exact sparse CPU Newton research reference."""

from __future__ import annotations

import builtins
import dataclasses
import types
import unittest
from unittest import mock

import numpy as np
import torch

from .. import solver_benchmark as benchmark
from .. import sparse_newton_reference as sparse_newton
from ..newton_baseline import NewtonConfig, build_newton_problem, solve_newton


def _single_tet_problem(*, isolated_free_vertex: bool = False):
    rest = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 2.0, 2.0),
        )[: 5 if isolated_free_vertex else 4],
        dtype=np.float64,
    )
    pinned = np.arange(4, dtype=np.int64) if isolated_free_vertex else np.array([0], dtype=np.int64)
    return build_newton_problem(
        rest,
        np.array([[0, 1, 2, 3]], dtype=np.int64),
        np.eye(3, dtype=np.float64)[None],
        np.ones(rest.shape[0], dtype=np.float64),
        0.0 if isolated_free_vertex else 7.0,
        0.0 if isolated_free_vertex else 13.0,
        0.25,
        pinned_indices=pinned,
        pin_targets=rest[pinned],
    )


class TestSparseExactHessian(unittest.TestCase):
    def test_matches_dense_autograd_at_regular_singular_and_inverted_states(self):
        problem = _single_tet_problem()
        deformation_gradients = (
            np.array(((1.1, 0.2, -0.1), (0.1, 0.8, 0.3), (0.0, -0.2, 1.2))),
            np.diag((1.0, 0.7, 0.0)),
            np.diag((1.0, 0.7, -0.4)),
        )
        for deformation_gradient in deformation_gradients:
            with self.subTest(determinant=np.linalg.det(deformation_gradient)):
                positions = problem.rest_q.numpy().copy()
                positions[1] = deformation_gradient[:, 0]
                positions[2] = deformation_gradient[:, 1]
                positions[3] = deformation_gradient[:, 2]
                free = problem.free_from_positions(positions).requires_grad_(True)
                objective = problem.objective_free(free)
                (gradient,) = torch.autograd.grad(objective, free)
                dense = torch.autograd.functional.hessian(problem.objective_free, free, vectorize=True)

                assembled = sparse_newton.assemble_sparse_exact_hessian(problem, positions)
                np.testing.assert_allclose(assembled.gradient, gradient.detach().numpy(), rtol=3.0e-14, atol=3.0e-14)
                np.testing.assert_allclose(
                    assembled.matrix.toarray(),
                    dense.detach().numpy(),
                    rtol=3.0e-14,
                    atol=3.0e-14,
                )
                self.assertAlmostEqual(assembled.objective, float(objective.detach()), places=13)
                self.assertAlmostEqual(
                    assembled.minimum_determinant,
                    float(np.linalg.det(deformation_gradient)),
                    places=14,
                )

    def test_inertia_only_isolated_free_vertex_needs_no_elastic_triplets(self):
        problem = _single_tet_problem(isolated_free_vertex=True)
        assembled = sparse_newton.assemble_sparse_exact_hessian(problem, problem.rest_q)
        expected_diagonal = np.full(3, 1.0 / problem.dt**2, dtype=np.float64)
        np.testing.assert_array_equal(assembled.gradient, np.zeros(3, dtype=np.float64))
        np.testing.assert_array_equal(assembled.matrix.toarray(), np.diag(expected_diagonal))
        self.assertEqual(assembled.raw_triplet_count, 0)
        self.assertEqual(assembled.nnz, 3)


class TestSymmetricFactorCertificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sparse, cls.sparse_linalg, _ = sparse_newton._load_scipy()

    def _certificate(self, diagonal, *, lower=None, upper=None, perm_r=None, perm_c=None):
        size = len(diagonal)
        shifted = self.sparse.diags(diagonal, format="csc")
        lower = self.sparse.eye(size, format="csc") if lower is None else self.sparse.csc_matrix(lower)
        upper = shifted if upper is None else self.sparse.csc_matrix(upper)
        identity = np.arange(size, dtype=np.int32)
        factor = types.SimpleNamespace(
            L=lower,
            U=upper,
            perm_r=identity if perm_r is None else np.asarray(perm_r, dtype=np.int32),
            perm_c=identity if perm_c is None else np.asarray(perm_c, dtype=np.int32),
        )
        return sparse_newton._certify_symmetric_factor(
            shifted,
            factor,
            self.sparse,
            max(float(np.max(np.abs(diagonal))), 1.0),
        )

    def test_rejects_mismatched_symmetric_permutations(self):
        certificate = self._certificate((1.0, 2.0), perm_r=(1, 0), perm_c=(0, 1))
        self.assertFalse(certificate.permutations_match)
        self.assertIsNone(certificate.factorization_relative_residual)
        self.assertFalse(certificate.passed)

    def test_rejects_negative_and_numerically_tiny_diagonal_pivots(self):
        for name, diagonal in (("negative", (-1.0, 2.0)), ("tiny", (1.0e-16, 1.0))):
            with self.subTest(name=name):
                certificate = self._certificate(diagonal)
                self.assertLessEqual(
                    certificate.minimum_diagonal_relative,
                    sparse_newton.SPARSE_FACTOR_PIVOT_RELATIVE_MARGIN,
                )
                self.assertFalse(certificate.passed)

    def test_rejects_broken_u_equals_d_l_transpose_relation(self):
        lower = np.array(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
        upper = np.array(((2.0, 0.75), (0.0, 3.0)), dtype=np.float64)
        shifted = lower @ upper
        certificate = self._certificate(np.diag(shifted), lower=lower, upper=upper)
        self.assertGreater(
            certificate.relation_relative_residual,
            sparse_newton.SPARSE_FACTOR_RELATION_RELATIVE_LIMIT,
        )
        self.assertFalse(certificate.passed)

    def test_rejects_relation_preserving_factor_of_another_matrix(self):
        certificate = self._certificate((1.0, 2.0), upper=np.eye(2, dtype=np.float64))
        self.assertEqual(certificate.relation_relative_residual, 0.0)
        self.assertGreater(
            certificate.factorization_relative_residual,
            sparse_newton.SPARSE_FACTORIZATION_RELATIVE_RESIDUAL_LIMIT,
        )
        self.assertFalse(certificate.passed)

    def test_missed_ritz_mode_uses_certified_gershgorin_rescue(self):
        matrix = self.sparse.diags((-1.0, 2.0, 3.0), format="csr")
        options = []

        def missed_mode(*args, **kwargs):
            return np.array([2.0]), np.array(((0.0,), (1.0,), (0.0,)))

        def recording_splu(*args, **kwargs):
            options.append(kwargs["options"])
            return self.sparse_linalg.splu(*args, **kwargs)

        sparse_linalg = types.SimpleNamespace(
            ArpackError=self.sparse_linalg.ArpackError,
            eigsh=missed_mode,
            splu=recording_splu,
        )
        config = NewtonConfig(max_regularization_attempts=3)
        direction = sparse_newton._direction(
            matrix,
            np.ones(3, dtype=np.float64),
            config,
            self.sparse,
            sparse_linalg,
            sparse_newton._Work(),
        )
        self.assertIsNotNone(direction.value)
        self.assertEqual(direction.ritz_regularization, 0.0)
        self.assertTrue(direction.gershgorin_rescue_used)
        self.assertEqual(direction.gershgorin_lower_bound, -1.0)
        self.assertEqual(direction.factorization_attempts, 2)
        self.assertEqual(direction.factor_certificate_attempts, 2)
        self.assertEqual(direction.linear_solve_attempts, 1)
        self.assertTrue(direction.factor_certificate_passed)
        self.assertEqual(direction.regularization, direction.gershgorin_rescue_regularization)
        self.assertEqual(direction.last_attempted_regularization, direction.regularization)
        self.assertEqual(
            options,
            [{"SymmetricMode": True, sparse_newton._SUPERLU_EQUILIBRATION_OPTION: False}] * 2,
        )

    def test_exhaustion_records_actual_last_attempt_not_untried_growth(self):
        matrix = self.sparse.diags((-1.0, 2.0, 3.0), format="csr")

        def missed_mode(*args, **kwargs):
            return np.array([2.0]), np.array(((0.0,), (1.0,), (0.0,)))

        sparse_linalg = types.SimpleNamespace(
            ArpackError=self.sparse_linalg.ArpackError,
            eigsh=missed_mode,
            splu=self.sparse_linalg.splu,
        )
        direction = sparse_newton._direction(
            matrix,
            np.ones(3, dtype=np.float64),
            NewtonConfig(max_regularization_attempts=1),
            self.sparse,
            sparse_linalg,
            sparse_newton._Work(),
        )
        self.assertIsNone(direction.value)
        self.assertEqual(direction.reason, "factor_certificate")
        self.assertEqual(direction.last_attempted_regularization, 0.0)
        self.assertEqual(direction.regularization, 0.0)
        self.assertGreater(direction.gershgorin_rescue_regularization, direction.last_attempted_regularization)
        self.assertFalse(direction.factor_certificate_passed)


class TestSparseNewtonReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = benchmark.build_structured_cantilever_scene(dimensions=(2, 1, 1))
        cls.problem = benchmark.build_common_problem(cls.scene)
        cls.config = NewtonConfig(
            max_iterations=50,
            gradient_absolute_tolerance=1.0e-10,
            gradient_relative_tolerance=1.0e-10,
            step_relative_tolerance=1.0e-14,
        )
        cls.iterate_zero = cls.problem.inertial_target.index_copy(
            0,
            cls.problem.pinned,
            cls.problem.pin_targets,
        )
        cls.sparse_result = sparse_newton.solve_sparse_newton(cls.problem, cls.iterate_zero, cls.config)
        cls.dense_result = solve_newton(cls.problem, cls.iterate_zero, cls.config)

    def test_solution_matches_dense_reference_and_authenticates_directions(self):
        result = self.sparse_result
        self.assertTrue(result.converged)
        self.assertEqual(result.reason, "gradient")
        self.assertEqual(result.accepted_iterations, self.dense_result.accepted_iterations)
        np.testing.assert_allclose(
            result.positions,
            self.dense_result.x.detach().numpy(),
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertAlmostEqual(result.final_objective, self.dense_result.final_objective, places=12)
        np.testing.assert_array_equal(
            result.positions[self.problem.pinned.numpy()],
            self.problem.pin_targets.numpy(),
        )
        self.assertFalse(result.positions.flags["W"])
        self.assertEqual(benchmark._sparse_newton_trace_failures(result, self.config, "test"), [])
        for item in result.trace[:-1]:
            self.assertLessEqual(item.linear_relative_residual, sparse_newton.SPARSE_LINEAR_RESIDUAL_LIMIT)
            self.assertGreater(item.factor_nnz, 0)
            self.assertTrue(item.factor_certificate_passed)
            self.assertTrue(item.factor_permutations_match)
            self.assertGreater(
                item.factor_minimum_diagonal_relative,
                sparse_newton.SPARSE_FACTOR_PIVOT_RELATIVE_MARGIN,
            )
            self.assertLessEqual(
                item.factor_relation_relative_residual,
                sparse_newton.SPARSE_FACTOR_RELATION_RELATIVE_LIMIT,
            )
            self.assertLessEqual(
                item.factorization_relative_residual,
                sparse_newton.SPARSE_FACTORIZATION_RELATIVE_RESIDUAL_LIMIT,
            )

    def test_work_accounting_and_timing_records_are_separate(self):
        result = self.sparse_result
        deterministic = result.deterministic_record()
        timing = result.timing_record()
        self.assertNotIn("total_seconds", deterministic)
        self.assertIn("total_seconds", timing)
        self.assertNotIn("elapsed_seconds", deterministic["trace"][0])
        self.assertIn("elapsed_seconds", timing["trace"][0])
        with self.assertRaisesRegex(ValueError, "objective work"):
            dataclasses.replace(result, objective_evaluations=result.objective_evaluations + 1)

    def test_zero_iteration_budget_fails_closed_without_linear_work(self):
        config = dataclasses.replace(self.config, max_iterations=0)
        result = sparse_newton.solve_sparse_newton(self.problem, self.iterate_zero, config)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "max_iterations")
        self.assertEqual(result.accepted_iterations, 0)
        self.assertEqual(result.eigenvalue_evaluations, 0)
        self.assertEqual(result.factorization_attempts, 0)
        self.assertEqual(result.factor_certificate_attempts, 0)
        self.assertEqual(result.linear_solve_attempts, 0)
        self.assertEqual(len(result.trace), 1)

    def test_trace_authentication_rejects_linear_residual_tampering(self):
        trace = list(self.sparse_result.trace)
        trace[0] = dataclasses.replace(
            trace[0],
            linear_relative_residual=10.0 * sparse_newton.SPARSE_LINEAR_RESIDUAL_LIMIT,
        )
        tampered = dataclasses.replace(self.sparse_result, trace=tuple(trace))
        failures = benchmark._sparse_newton_trace_failures(tampered, self.config, "tampered")
        self.assertTrue(any("linear residual gate" in failure for failure in failures))

    def test_trace_authentication_rejects_factor_certificate_tampering(self):
        trace = list(self.sparse_result.trace)
        trace[0] = dataclasses.replace(
            trace[0],
            factor_relation_relative_residual=10.0 * sparse_newton.SPARSE_FACTOR_RELATION_RELATIVE_LIMIT,
        )
        tampered = dataclasses.replace(self.sparse_result, trace=tuple(trace))
        failures = benchmark._sparse_newton_trace_failures(tampered, self.config, "tampered")
        self.assertTrue(any("symmetric factor certificate" in failure for failure in failures))

    def test_scipy_import_is_lazy_and_missing_extra_has_actionable_error(self):
        original_import = builtins.__import__

        def import_without_scipy(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("deliberately hidden")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_scipy):
            with self.assertRaisesRegex(ModuleNotFoundError, "existing 'dev' or 'importers' extra"):
                sparse_newton._load_scipy()

    def test_benchmark_adapter_accepts_repeats_and_binds_digest(self):
        run = benchmark.run_sparse_newton(
            self.scene,
            self.problem,
            config=self.config,
            warmup=False,
            repeats=2,
        )
        self.assertTrue(run.reference_accepted, run.reference_failures)
        self.assertEqual(run.reference_failures, ())
        self.assertEqual(len(run.repeat_seconds), 2)
        self.assertGreater(run.median_solve_seconds, 0.0)
        self.assertEqual(run.run_sha256, benchmark._sparse_newton_run_digest(run))
        self.assertEqual(run.verification_displacement_relative, 0.0)
        self.assertLessEqual(run.alternate_start_displacement_relative, 1.0e-9)
        with self.assertRaisesRegex(ValueError, "run digest"):
            dataclasses.replace(
                run,
                alternate_start_displacement_relative=run.alternate_start_displacement_relative + 1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()
