# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the research-only Warp matrix-free correction primitives."""

from __future__ import annotations

import inspect
import json
import os
import unittest

import numpy as np
import torch
import warp as wp

from research.principal_stretch.correction_gpu import MatrixFreeStableNHOperator, solve_fixed_pcg
from research.principal_stretch.correction_gpu_warp import (
    CONTRACT_ID,
    KERNEL_VERSION,
    WarpFixedPCGWorkspace,
    WarpMatrixFreeStableNHOperator,
)
from research.principal_stretch.newton_baseline import NewtonProblem, build_newton_problem


def _shared_vertex_problem() -> NewtonProblem:
    rest = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.4, 0.2, 0.8),
        ),
        dtype=np.float64,
    )
    # Deliberately non-monotone corner ownership stresses sorted shared-vertex
    # gathers independently of the free-vertex ordering.
    tets = np.array(((1, 2, 3, 4), (0, 1, 2, 3), (2, 1, 4, 5)), dtype=np.int64)
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
        np.array((0.8, 1.1, 1.4, 0.9, 1.3, 1.0), dtype=np.float64),
        np.array((13.0, 29.0, 47.0), dtype=np.float64),
        np.array((41.0, 73.0, 101.0), dtype=np.float64),
        0.061,
        pinned_indices=np.array((0, 4), dtype=np.int64),
        pin_targets=rest[[0, 4]],
        inertial_target=rest
        + np.array(
            (
                (0.0, 0.0, 0.0),
                (0.01, -0.02, 0.005),
                (-0.02, 0.01, 0.015),
                (0.005, 0.012, -0.007),
                (0.0, 0.0, 0.0),
                (-0.014, 0.006, 0.009),
            ),
            dtype=np.float64,
        ),
    )


def _deformed_positions(problem: NewtonProblem) -> np.ndarray:
    positions = problem.rest_q.numpy().copy()
    positions += np.array(
        (
            (0.0, 0.0, 0.0),
            (0.07, 0.03, -0.01),
            (-0.02, -0.05, 0.04),
            (0.03, -0.01, 0.08),
            (0.0, 0.0, 0.0),
            (-0.04, 0.06, 0.02),
        ),
        dtype=np.float64,
    )
    return positions


def _oracle_and_device(device: str) -> tuple[MatrixFreeStableNHOperator, WarpMatrixFreeStableNHOperator]:
    problem = _shared_vertex_problem()
    oracle = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
    return oracle, WarpMatrixFreeStableNHOperator.from_oracle(oracle, device=device)


class TestWarpMatrixFreeStableNHOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_dtype(torch.float64)
        cls.oracle, cls.operator = _oracle_and_device("cpu")

    def test_kernel_version_is_explicit(self):
        self.assertEqual(KERNEL_VERSION, "mg-vbd-warp-operator-v1")
        self.assertEqual(CONTRACT_ID, "mg-vbd-warp-fixed-pcg-research-v1")

    def test_sorted_gather_and_exact_free_elimination(self):
        operator = self.operator
        oracle = self.oracle
        np.testing.assert_array_equal(operator.free_host, oracle.free)
        expected_lookup = np.full(oracle.n_vertices, -1, dtype=np.int32)
        expected_lookup[oracle.free] = np.arange(oracle.free.size, dtype=np.int32)
        np.testing.assert_array_equal(operator.vertex_to_free_host, expected_lookup)
        np.testing.assert_array_equal(operator.vertex_to_free_host[oracle.pinned], -1)

        expected_entries = 0
        for free_index, vertex in enumerate(oracle.free):
            start = int(operator.incidence_offsets_host[free_index])
            end = int(operator.incidence_offsets_host[free_index + 1])
            pairs = list(
                zip(
                    operator.incidence_tets_host[start:end].tolist(),
                    operator.incidence_corners_host[start:end].tolist(),
                    strict=True,
                )
            )
            self.assertEqual(pairs, sorted(pairs))
            expected = sorted(
                (tet, corner)
                for tet, corners in enumerate(oracle.tets)
                for corner, candidate in enumerate(corners)
                if int(candidate) == int(vertex)
            )
            self.assertEqual(pairs, expected)
            expected_entries += len(expected)
        self.assertEqual(int(operator.incidence_offsets_host[-1]), expected_entries)
        self.assertNotIn("wp.atomic", inspect.getsource(__import__(operator.__module__, fromlist=["*"])))

        full_sized = wp.zeros(oracle.n_vertices, dtype=wp.vec3d, device="cpu")
        output = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        with self.assertRaisesRegex(ValueError, "direction must be a vec3d array"):
            operator.launch_apply(full_sized, output, operator.create_apply_workspace())

    def test_gradient_action_diagonal_and_geometry_match_oracle(self):
        operator = self.operator
        oracle = self.oracle
        gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        operator.launch_gradient(gradient)
        np.testing.assert_allclose(gradient.numpy().reshape(-1), oracle.gradient_free(), rtol=2.0e-14, atol=8.0e-14)
        np.testing.assert_allclose(
            operator.deformation_gradients.numpy(), oracle.deformation_gradients, rtol=2.0e-14, atol=8.0e-14
        )
        np.testing.assert_allclose(operator.cofactors.numpy(), oracle.cofactors, rtol=3.0e-14, atol=8.0e-14)
        np.testing.assert_allclose(operator.determinants.numpy(), oracle.determinants, rtol=3.0e-14, atol=8.0e-14)

        direction_host = np.random.default_rng(817).normal(size=(operator.n_free, 3))
        direction = wp.array(direction_host, dtype=wp.vec3d, device="cpu")
        product = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        operator.launch_apply(direction, product, operator.create_apply_workspace())
        np.testing.assert_allclose(
            product.numpy().reshape(-1), oracle.apply_free(direction_host), rtol=3.0e-14, atol=2.0e-13
        )

        diagonal = wp.empty(operator.n_free, dtype=wp.mat33d, device="cpu")
        operator.launch_block_diagonal(diagonal)
        np.testing.assert_allclose(diagonal.numpy(), oracle.block_diagonal(), rtol=3.0e-14, atol=1.0e-13)

    def test_repeated_gathers_are_bitwise_deterministic(self):
        operator = self.operator
        direction_host = np.random.default_rng(823).normal(size=(operator.n_free, 3))
        direction = wp.array(direction_host, dtype=wp.vec3d, device="cpu")
        gradient = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        product = wp.empty(operator.n_free, dtype=wp.vec3d, device="cpu")
        diagonal = wp.empty(operator.n_free, dtype=wp.mat33d, device="cpu")
        workspace = operator.create_apply_workspace()

        snapshots = []
        for _ in range(3):
            operator.launch_gradient(gradient)
            operator.launch_apply(direction, product, workspace)
            operator.launch_block_diagonal(diagonal)
            snapshots.append((gradient.numpy(), product.numpy(), diagonal.numpy()))
        for current in snapshots[1:]:
            for expected, actual in zip(snapshots[0], current, strict=True):
                np.testing.assert_array_equal(actual, expected)


class TestWarpFixedPCG(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_dtype(torch.float64)
        cls.oracle, cls.operator = _oracle_and_device("cpu")
        cls.rhs = -cls.oracle.gradient_free()

    def test_fixed_pcg_matches_numpy_oracle_and_reports_work(self):
        iterations = 4
        expected = solve_fixed_pcg(self.oracle, self.rhs, iterations)
        workspace = WarpFixedPCGWorkspace(self.operator, iterations)
        pointers_before = {
            name: int(getattr(workspace, name).ptr)
            for name in (
                "rhs",
                "solution",
                "residual",
                "preconditioned",
                "direction",
                "operator_direction",
                "operator_solution",
                "block_diagonal",
                "preconditioner_inverse",
                "state_status",
                "rho",
                "trace_curvature",
            )
        }
        workspace.set_rhs(self.rhs)
        workspace.launch()
        result = workspace.record()
        pointers_after = {name: int(getattr(workspace, name).ptr) for name in pointers_before}

        self.assertTrue(result.success, result.deterministic_record())
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.completed_iterations, iterations)
        self.assertEqual(result.requested_iterations, iterations)
        self.assertEqual(len(result.trace), iterations)
        self.assertEqual(result.preconditioner_identity, "block-jacobi-3x3-warp-v1")
        self.assertFalse(result.capture_replay)
        self.assertTrue(result.research_only)
        self.assertFalse(result.performance_evidence)
        self.assertEqual(pointers_after, pointers_before)
        self.assertEqual(result.work.preconditioner_builds, 1)
        self.assertEqual(result.work.operator_applications, iterations + 1)
        self.assertEqual(result.work.residual_verification_applications, 1)
        self.assertEqual(result.work.preconditioner_applications, iterations + 1)
        self.assertEqual(result.work.scalar_reductions, 2 * iterations + 2)
        self.assertEqual(result.work.kernel_launches, 2 + 7 * iterations + 6)
        np.testing.assert_allclose(result.solution.reshape(-1), expected.solution, rtol=3.0e-13, atol=3.0e-14)
        self.assertAlmostEqual(result.true_residual_norm, expected.true_residual_norm, places=13)

    def test_breakdown_is_masked_without_shortening_schedule(self):
        iterations = 3
        identity_inverse = np.repeat(np.eye(3, dtype=np.float64)[None], self.operator.n_free, axis=0)
        workspace = WarpFixedPCGWorkspace(
            self.operator,
            iterations,
            external_preconditioner_inverse=identity_inverse,
            preconditioner_identity="test-zero-block-preconditioner-v1",
        )
        # A post-validation device corruption emulates runtime memory failure;
        # the PCG schedule must fail closed without a host-side short circuit.
        workspace.preconditioner_inverse.assign(np.zeros_like(identity_inverse))
        workspace.set_rhs(self.rhs)
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonpositive_preconditioner")
        self.assertEqual(result.completed_iterations, 0)
        self.assertEqual(len(result.trace), iterations)
        self.assertEqual([item.status for item in result.trace], [result.reason] * iterations)
        self.assertTrue(all(not item.active_update_completed for item in result.trace))
        np.testing.assert_array_equal(result.solution, 0.0)
        self.assertEqual(result.work.preconditioner_builds, 0)
        self.assertEqual(result.work.operator_applications, iterations + 1)
        self.assertEqual(result.work.preconditioner_applications, iterations + 1)
        self.assertEqual(result.work.scalar_reductions, 2 * iterations + 2)
        self.assertEqual(result.work.kernel_launches, 7 * iterations + 6)

    def test_external_preconditioner_requires_exact_symmetric_positive_definite_blocks(self):
        blocks = np.repeat(np.eye(3, dtype=np.float64)[None], self.operator.n_free, axis=0)
        nonsymmetric = blocks.copy()
        nonsymmetric[0, 0, 1] = 0.25
        with self.assertRaisesRegex(ValueError, "exactly symmetric"):
            WarpFixedPCGWorkspace(
                self.operator,
                2,
                external_preconditioner_inverse=nonsymmetric,
                preconditioner_identity="test-nonsymmetric-v1",
            )
        indefinite = blocks.copy()
        indefinite[0, 0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive definite"):
            WarpFixedPCGWorkspace(
                self.operator,
                2,
                external_preconditioner_inverse=indefinite,
                preconditioner_identity="test-indefinite-v1",
            )

    def test_zero_rhs_is_successfully_masked_at_fixed_work(self):
        iterations = 5
        workspace = WarpFixedPCGWorkspace(self.operator, iterations)
        workspace.set_rhs(np.zeros(self.operator.n_free_dofs, dtype=np.float64))
        workspace.launch()
        result = workspace.record()

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "zero_rhs")
        self.assertEqual(result.completed_iterations, 0)
        self.assertEqual(len(result.trace), iterations)
        self.assertEqual(result.true_residual_norm, 0.0)
        self.assertEqual(result.work.operator_applications, iterations + 1)
        self.assertEqual(result.work.preconditioner_applications, iterations + 1)
        np.testing.assert_array_equal(result.solution, 0.0)

    def test_nonfinite_rhs_fails_closed_and_preserves_primary_reason(self):
        iterations = 2
        for nonfinite in (np.nan, np.inf, -np.inf):
            with self.subTest(nonfinite=nonfinite):
                rhs = self.rhs.copy()
                rhs[3] = nonfinite
                workspace = WarpFixedPCGWorkspace(self.operator, iterations)
                workspace.set_rhs(rhs)
                workspace.launch()
                result = workspace.record()

                self.assertFalse(result.success)
                self.assertEqual(result.reason, "nonfinite_rhs")
                self.assertEqual(result.completed_iterations, 0)
                self.assertIsNone(result.rhs_norm)
                self.assertIsNone(result.true_residual_norm)
                self.assertEqual([item.status for item in result.trace], [result.reason] * iterations)
                np.testing.assert_array_equal(result.solution, 0.0)
                self.assertEqual(result.work.operator_applications, iterations + 1)
                json.dumps(result.deterministic_record(), allow_nan=False)

    def test_nonfinite_preconditioner_corruption_is_finite_json_safe(self):
        iterations = 2
        identity_inverse = np.repeat(np.eye(3, dtype=np.float64)[None], self.operator.n_free, axis=0)
        workspace = WarpFixedPCGWorkspace(
            self.operator,
            iterations,
            external_preconditioner_inverse=identity_inverse,
            preconditioner_identity="test-corrupted-block-preconditioner-v1",
        )
        corrupted = identity_inverse.copy()
        corrupted[0, 0, 0] = np.nan
        workspace.preconditioner_inverse.assign(corrupted)
        workspace.set_rhs(self.rhs)
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonfinite_preconditioner")
        self.assertEqual(result.completed_iterations, 0)
        self.assertTrue(all(item.residual_norm is None for item in result.trace))
        json.dumps(result.deterministic_record(), allow_nan=False)

    def test_nonfinite_operator_corruption_is_finite_json_safe(self):
        oracle, operator = _oracle_and_device("cpu")
        corrupted = operator.cofactors.numpy()
        corrupted[0, 0, 0] = np.nan
        operator.cofactors.assign(corrupted)
        identity_inverse = np.repeat(np.eye(3, dtype=np.float64)[None], operator.n_free, axis=0)
        workspace = WarpFixedPCGWorkspace(
            operator,
            2,
            external_preconditioner_inverse=identity_inverse,
            preconditioner_identity="test-identity-block-preconditioner-v1",
        )
        workspace.set_rhs(-oracle.gradient_free())
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonfinite_operator")
        self.assertEqual(result.completed_iterations, 0)
        self.assertIsNone(result.true_residual_norm)
        self.assertTrue(all(item.direction_curvature is None for item in result.trace))
        json.dumps(result.deterministic_record(), allow_nan=False)

    def test_repeat_launch_reuses_buffers_and_is_bitwise_deterministic(self):
        workspace = WarpFixedPCGWorkspace(self.operator, 4)
        pointers = (
            int(workspace.solution.ptr),
            int(workspace.direction.ptr),
            int(workspace.apply_workspace.delta_piola.ptr),
        )
        solutions = []
        records = []
        for _ in range(3):
            workspace.set_rhs(self.rhs)
            workspace.launch()
            record = workspace.record()
            solutions.append(record.solution)
            records.append(record.deterministic_record())
        self.assertEqual(
            pointers,
            (int(workspace.solution.ptr), int(workspace.direction.ptr), int(workspace.apply_workspace.delta_piola.ptr)),
        )
        np.testing.assert_array_equal(solutions[1], solutions[0])
        np.testing.assert_array_equal(solutions[2], solutions[0])
        self.assertEqual(records[1], records[0])
        self.assertEqual(records[2], records[0])


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestWarpFixedPCGCudaCapture(unittest.TestCase):
    def test_fixed_schedule_is_cuda_graph_capturable(self):
        if wp.get_cuda_device_count() < 1:
            self.skipTest("no claimed CUDA device is visible")
        torch.set_default_dtype(torch.float64)
        oracle, operator = _oracle_and_device("cuda:0")
        rhs = -oracle.gradient_free()
        workspace = WarpFixedPCGWorkspace(operator, 4)

        # Warm compilation and allocations before capture.  The fixed launcher
        # itself performs neither allocation nor host synchronization.
        workspace.set_rhs(rhs)
        workspace.launch()
        expected = workspace.record()
        workspace.set_rhs(rhs)
        with wp.ScopedCapture(device=operator.device) as capture:
            workspace.launch()
        wp.capture_launch(capture.graph)
        captured = workspace.record(capture_replay=True)

        self.assertTrue(captured.success, captured.deterministic_record())
        self.assertTrue(captured.capture_replay)
        self.assertTrue(captured.research_only)
        self.assertFalse(captured.performance_evidence)
        self.assertEqual(captured.work, expected.work)
        np.testing.assert_array_equal(captured.solution, expected.solution)
        self.assertAlmostEqual(captured.true_residual_norm, expected.true_residual_norm, places=13)


if __name__ == "__main__":
    unittest.main()
