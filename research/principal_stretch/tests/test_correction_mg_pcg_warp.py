# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for static Warp multigrid used inside fixed current-A PCG."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import unittest

import numpy as np
import torch
import warp as wp

from research.principal_stretch.correction_gpu import MatrixFreeStableNHOperator, solve_fixed_pcg
from research.principal_stretch.correction_gpu_warp import WarpFixedPCGWorkspace, WarpMatrixFreeStableNHOperator
from research.principal_stretch.correction_multigrid import apply_v_cycle, build_stable_nh_rest_multigrid
from research.principal_stretch.correction_multigrid_warp import (
    WarpStaticMultigridHierarchy,
    WarpStaticMultigridPreconditioner,
)
from research.principal_stretch.newton_baseline import NewtonProblem, build_newton_problem

_ITERATIONS = 4


@wp.kernel(enable_backward=False)
def _inject_preconditioner_nonfinite(
    output: wp.array[wp.vec3d],
    retained_output: wp.array[wp.float64],
    inject: int,
):
    if wp.tid() == 0 and inject != 0:
        value = output[0]
        output[0] = wp.vec3d(wp.float64(wp.nan), value[1], value[2])
        retained_output[0] = wp.float64(wp.nan)


class _LaterFailingMultigridPreconditioner(WarpStaticMultigridPreconditioner):
    """Test boundary that injects nonfinite output at one retained apply."""

    def __init__(self, hierarchy: WarpStaticMultigridHierarchy, failure_application: int):
        super().__init__(hierarchy)
        self.failure_application = failure_application
        self._workspace_injection: dict[int, int] = {}
        self._workspace_count = 0
        self.application_kernel_launches += 1

    def create_application_workspace(self):
        workspace = super().create_application_workspace()
        self._workspace_injection[id(workspace)] = int(self._workspace_count == self.failure_application)
        self._workspace_count += 1
        return workspace

    def launch_apply(self, rhs, output, workspace) -> None:
        super().launch_apply(rhs, output, workspace)
        wp.launch(
            _inject_preconditioner_nonfinite,
            dim=1,
            inputs=[output, workspace.level_correction[0], self._workspace_injection[id(workspace)]],
            device=self.device,
        )

    def record_application(self, application_index, workspace, *, capture_replay):
        record = super().record_application(
            application_index,
            workspace,
            capture_replay=capture_replay,
        )
        return dataclasses.replace(record, scheduled_kernel_launches=self.application_kernel_launches)


def _problem() -> NewtonProblem:
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


def _oracle_and_hierarchy():
    problem = _problem()
    oracle = MatrixFreeStableNHOperator.from_problem(problem, _deformed_positions(problem))
    hierarchy = build_stable_nh_rest_multigrid(
        oracle,
        problem.rest_q.numpy(),
        coarse_node_limit=1,
    )
    return oracle, hierarchy


def _device_bundle(device: str):
    oracle, hierarchy = _oracle_and_hierarchy()
    operator = WarpMatrixFreeStableNHOperator.from_oracle(oracle, device=device)
    device_hierarchy = WarpStaticMultigridHierarchy.from_hierarchy(hierarchy, device=device)
    preconditioner = WarpStaticMultigridPreconditioner(device_hierarchy)
    workspace = WarpFixedPCGWorkspace(
        operator,
        _ITERATIONS,
        device_preconditioner=preconditioner,
    )
    return oracle, hierarchy, operator, device_hierarchy, preconditioner, workspace


def _numpy_solve(oracle, hierarchy, rhs):
    applications = []

    def precondition(residual):
        application = apply_v_cycle(hierarchy, residual)
        applications.append(application)
        return application.correction

    result = solve_fixed_pcg(
        oracle,
        rhs,
        _ITERATIONS,
        preconditioner=precondition,
        preconditioner_identity=f"static-mg-v-cycle-cpu-v1:{hierarchy.content_sha256}",
    )
    return result, tuple(applications)


def _workspace_pointers(workspace) -> tuple[int, ...]:
    pointers = [
        int(workspace.rhs.ptr),
        int(workspace.solution.ptr),
        int(workspace.residual.ptr),
        int(workspace.preconditioned.ptr),
        int(workspace.direction.ptr),
    ]
    for application in workspace.device_preconditioner_workspaces:
        pointers.extend(int(array.ptr) for array in application.level_rhs)
        pointers.extend(int(array.ptr) for array in application.level_correction)
    return tuple(pointers)


def _assert_exact_application_work(test, actual, expected) -> None:
    test.assertEqual(actual.static_preconditioner_sha256, expected.work.hierarchy_sha256)
    test.assertEqual(actual.rhs_count, expected.work.rhs_count)
    test.assertEqual(actual.level_visits, expected.work.level_visits)
    for field in (
        "matrix_block_products",
        "smoother_block_solves",
        "restriction_block_products",
        "prolongation_block_products",
        "coarsest_factor_solves",
    ):
        test.assertEqual(getattr(actual, field), getattr(expected.work, field), field)


class TestWarpMGFixedPCG(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_dtype(torch.float64)

    def test_current_operator_static_hierarchy_matches_numpy_and_retains_exact_work(self):
        oracle, hierarchy, operator, device_hierarchy, preconditioner, workspace = _device_bundle("cpu")
        rhs = -oracle.gradient_free()
        expected, expected_applications = _numpy_solve(oracle, hierarchy, rhs)
        pointers = _workspace_pointers(workspace)

        workspace.set_rhs(rhs)
        workspace.launch()
        actual = workspace.record()

        self.assertTrue(actual.success, actual.deterministic_record())
        self.assertEqual(actual.reason, "completed")
        self.assertEqual(actual.completed_iterations, _ITERATIONS)
        np.testing.assert_allclose(actual.solution.reshape(-1), expected.solution, rtol=4.0e-13, atol=4.0e-14)
        self.assertAlmostEqual(actual.true_residual_norm, expected.true_residual_norm, places=13)
        self.assertEqual(actual.work.preconditioner_builds, 0)
        self.assertEqual(actual.work.preconditioner_applications, _ITERATIONS)
        self.assertEqual(actual.work.operator_applications, _ITERATIONS + 1)
        self.assertEqual(actual.work.kernel_launches, 90)
        self.assertEqual(len(workspace.device_preconditioner_workspaces), _ITERATIONS)
        self.assertEqual(len(actual.preconditioner_evidence), _ITERATIONS)
        self.assertEqual(len(expected_applications), _ITERATIONS)
        self.assertEqual(_workspace_pointers(workspace), pointers)
        self.assertEqual(len(set(_workspace_pointers(workspace))), len(_workspace_pointers(workspace)))
        for index, (device_application, cpu_application) in enumerate(
            zip(actual.preconditioner_evidence, expected_applications, strict=True)
        ):
            self.assertEqual(device_application.application_index, index)
            self.assertTrue(device_application.output_finite)
            self.assertEqual(device_application.scheduled_kernel_launches, 14)
            _assert_exact_application_work(self, device_application, cpu_application)

        self.assertEqual(actual.current_operator_sha256, operator.current_operator_sha256)
        self.assertEqual(actual.static_preconditioner_sha256, hierarchy.content_sha256)
        self.assertEqual(actual.static_preconditioner_sha256, device_hierarchy.hierarchy_sha256)
        self.assertEqual(actual.preconditioner_identity, preconditioner.preconditioner_identity)
        self.assertEqual(len(actual.operator_preconditioner_binding_sha256), 64)
        self.assertNotEqual(actual.current_operator_sha256, actual.static_preconditioner_sha256)
        self.assertIsNone(actual.trace[-1].conjugacy)
        self.assertFalse(actual.capture_replay)
        self.assertTrue(actual.research_only)
        self.assertFalse(actual.performance_evidence)
        json.dumps(actual.deterministic_record(), allow_nan=False)

        launch_source = inspect.getsource(WarpFixedPCGWorkspace._launch_device_preconditioned)
        preconditioner_source = inspect.getsource(WarpStaticMultigridPreconditioner.launch_apply)
        self.assertNotIn(".numpy()", launch_source)
        self.assertNotIn(".numpy()", preconditioner_source)

    def test_nonfinite_mg_output_fails_before_initial_dot_and_keeps_fixed_work(self):
        oracle, _hierarchy, _operator, device_hierarchy, _preconditioner, workspace = _device_bundle("cpu")
        corrupted = device_hierarchy.coarse_cholesky.numpy()
        corrupted[0] = np.nan
        device_hierarchy.coarse_cholesky.assign(corrupted)
        workspace.set_rhs(-oracle.gradient_free())
        workspace.launch()
        result = workspace.record()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonfinite_preconditioner")
        self.assertEqual(result.completed_iterations, 0)
        np.testing.assert_array_equal(result.solution, 0.0)
        self.assertEqual(result.work.preconditioner_applications, _ITERATIONS)
        self.assertEqual(len(result.preconditioner_evidence), _ITERATIONS)
        self.assertTrue(all(not item.output_finite for item in result.preconditioner_evidence))
        self.assertEqual([item.status for item in result.trace], [result.reason] * _ITERATIONS)
        json.dumps(result.deterministic_record(), allow_nan=False)

    def test_later_preconditioner_failure_retains_completed_update_and_trace(self):
        oracle, hierarchy = _oracle_and_hierarchy()
        operator = WarpMatrixFreeStableNHOperator.from_oracle(oracle, device="cpu")
        device_hierarchy = WarpStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cpu")
        failure_application = 2
        preconditioner = _LaterFailingMultigridPreconditioner(device_hierarchy, failure_application)
        workspace = WarpFixedPCGWorkspace(
            operator,
            _ITERATIONS,
            device_preconditioner=preconditioner,
        )
        rhs = -oracle.gradient_free()
        cpu_application = 0

        def failing_cpu_preconditioner(residual):
            nonlocal cpu_application
            correction = apply_v_cycle(hierarchy, residual).correction
            if cpu_application == failure_application:
                correction = np.full_like(correction, np.nan)
            cpu_application += 1
            return correction

        expected = solve_fixed_pcg(
            oracle,
            rhs,
            _ITERATIONS,
            preconditioner=failing_cpu_preconditioner,
            preconditioner_identity="test-later-failing-static-mg",
        )
        workspace.set_rhs(rhs)
        workspace.launch()
        result = workspace.record()

        self.assertEqual(expected.reason, "preconditioner_failure")
        self.assertEqual(expected.completed_iterations, failure_application)
        self.assertEqual(len(expected.trace), failure_application)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "nonfinite_preconditioner")
        self.assertEqual(result.completed_iterations, failure_application)
        np.testing.assert_allclose(result.solution.reshape(-1), expected.solution, rtol=4.0e-13, atol=4.0e-14)
        self.assertEqual(
            [item.active_update_completed for item in result.trace],
            [True, True, False, False],
        )
        self.assertEqual(
            [item.status for item in result.trace],
            ["active", result.reason, result.reason, result.reason],
        )
        for index, expected_trace in enumerate(expected.trace):
            actual_trace = result.trace[index]
            self.assertAlmostEqual(actual_trace.residual_norm, expected_trace.residual_norm, places=13)
            self.assertAlmostEqual(actual_trace.direction_curvature, expected_trace.direction_curvature, places=12)
            self.assertAlmostEqual(actual_trace.step_size, expected_trace.step_size, places=13)
        self.assertIsNotNone(result.trace[0].conjugacy)
        self.assertIsNone(result.trace[1].conjugacy)
        self.assertEqual(result.work.preconditioner_applications, _ITERATIONS)
        self.assertEqual(result.work.kernel_launches, 94)
        self.assertEqual(len(result.preconditioner_evidence), _ITERATIONS)
        self.assertEqual(
            [item.output_finite for item in result.preconditioner_evidence],
            [True, True, False, True],
        )
        self.assertTrue(all(item.scheduled_kernel_launches == 15 for item in result.preconditioner_evidence))
        json.dumps(result.deterministic_record(), allow_nan=False)

    def test_device_boundary_rejects_relabel_and_keeps_block_path_independent(self):
        _oracle, _hierarchy, operator, _device_hierarchy, preconditioner, _workspace = _device_bundle("cpu")
        with self.assertRaisesRegex(ValueError, "cannot be relabelled"):
            WarpFixedPCGWorkspace(
                operator,
                _ITERATIONS,
                device_preconditioner=preconditioner,
                preconditioner_identity="forged-static-mg",
            )
        block_workspace = WarpFixedPCGWorkspace(operator, _ITERATIONS)
        self.assertEqual(block_workspace.work.preconditioner_applications, _ITERATIONS + 1)
        self.assertEqual(block_workspace.preconditioner_identity, "block-jacobi-3x3-warp-v1")
        self.assertEqual(block_workspace.device_preconditioner_workspaces, ())


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestWarpMGFixedPCGCudaCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        torch.set_default_dtype(torch.float64)

    def test_cuda_direct_matches_numpy_and_capture_replays_changed_rhs(self):
        oracle, hierarchy, _operator, device_hierarchy, preconditioner, direct_workspace = _device_bundle("cuda:0")
        rhs_a = -oracle.gradient_free()
        rhs_b = np.random.default_rng(829).normal(size=oracle.n_free_dofs)
        expected_a, _ = _numpy_solve(oracle, hierarchy, rhs_a)
        expected_b, _ = _numpy_solve(oracle, hierarchy, rhs_b)

        direct_workspace.set_rhs(rhs_a)
        direct_workspace.launch()
        direct_a = direct_workspace.record()
        direct_workspace.set_rhs(rhs_b)
        direct_workspace.launch()
        direct_b = direct_workspace.record()
        np.testing.assert_allclose(direct_a.solution.reshape(-1), expected_a.solution, rtol=7.0e-13, atol=8.0e-14)
        np.testing.assert_allclose(direct_b.solution.reshape(-1), expected_b.solution, rtol=7.0e-13, atol=8.0e-14)

        captured_workspace = WarpFixedPCGWorkspace(
            direct_workspace.operator,
            _ITERATIONS,
            device_preconditioner=preconditioner,
        )
        captured_workspace.set_rhs(rhs_a)
        captured_workspace.launch()
        warm = captured_workspace.record()
        pointers = _workspace_pointers(captured_workspace)
        captured_workspace.set_rhs(rhs_a)
        with wp.ScopedCapture(device=device_hierarchy.device) as capture:
            captured_workspace.launch()
        captured_workspace.set_rhs(rhs_b)
        wp.capture_launch(capture.graph)
        captured = captured_workspace.record(capture_replay=True)

        self.assertTrue(captured.success, captured.deterministic_record())
        self.assertTrue(captured.capture_replay)
        self.assertEqual(_workspace_pointers(captured_workspace), pointers)
        np.testing.assert_array_equal(captured.solution, direct_b.solution)
        self.assertNotEqual(captured.solution.tobytes(), warm.solution.tobytes())
        self.assertEqual(
            captured.preconditioner_evidence[0].input_sha256, direct_b.preconditioner_evidence[0].input_sha256
        )
        self.assertNotEqual(
            captured.preconditioner_evidence[0].input_sha256, warm.preconditioner_evidence[0].input_sha256
        )
        self.assertEqual(len(captured.preconditioner_evidence), _ITERATIONS)
        self.assertTrue(all(item.capture_replay for item in captured.preconditioner_evidence))
        self.assertTrue(all(item.output_finite for item in captured.preconditioner_evidence))
        for captured_application, direct_application in zip(
            captured.preconditioner_evidence,
            direct_b.preconditioner_evidence,
            strict=True,
        ):
            self.assertEqual(captured_application.input_sha256, direct_application.input_sha256)
            self.assertEqual(captured_application.output_sha256, direct_application.output_sha256)
            self.assertEqual(captured_application.algebraic_work_sha256, direct_application.algebraic_work_sha256)
        self.assertEqual(captured.work.preconditioner_applications, _ITERATIONS)
        self.assertEqual(captured.work.kernel_launches, 90)
        self.assertEqual(captured.current_operator_sha256, direct_b.current_operator_sha256)
        self.assertEqual(captured.static_preconditioner_sha256, direct_b.static_preconditioner_sha256)
        self.assertEqual(
            captured.operator_preconditioner_binding_sha256,
            direct_b.operator_preconditioner_binding_sha256,
        )
        self.assertFalse(captured.performance_evidence)
        np.testing.assert_allclose(captured.solution.reshape(-1), expected_b.solution, rtol=7.0e-13, atol=8.0e-14)


if __name__ == "__main__":
    unittest.main()
