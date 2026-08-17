# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the research-only device-resident static MG V-cycle."""

from __future__ import annotations

import inspect
import json
import os
import unittest

import numpy as np
import warp as wp

from research.principal_stretch.correction_multigrid import (
    StaticMultigridHierarchy,
    apply_v_cycle,
    build_static_multigrid,
)
from research.principal_stretch.correction_multigrid_warp import (
    CONTRACT_ID,
    KERNEL_VERSION,
    MAX_COARSE_SCALAR_SIZE,
    WarpStaticMultigridHierarchy,
)


def _hierarchy() -> StaticMultigridHierarchy:
    node_count = 32
    index = np.arange(node_count, dtype=np.float64)
    rest = np.column_stack((0.13 * index, np.sin(0.71 * index), np.cos(0.43 * index)))
    graph = 0.37 * np.eye(node_count, dtype=np.float64)
    for node in range(node_count - 1):
        weight = 0.7 + 0.03 * (node % 5)
        graph[node, node] += weight
        graph[node + 1, node + 1] += weight
        graph[node, node + 1] -= weight
        graph[node + 1, node] -= weight
    coordinate_metric = np.array(
        ((1.7, 0.21, -0.08), (0.21, 1.3, 0.14), (-0.08, 0.14, 0.9)),
        dtype=np.float64,
    )
    matrix = np.kron(graph, coordinate_metric)
    hierarchy = build_static_multigrid(
        matrix,
        rest,
        np.arange(node_count, dtype=np.int64),
        mode_kind="rigid",
        target_aggregate_size=4,
        minimum_aggregate_size=3,
        coarse_node_limit=2,
        pre_smooth_steps=2,
        post_smooth_steps=2,
        static_model_sha256="a" * 64,
    )
    if [(level.matrix.block_row_count, level.matrix.block_size) for level in hierarchy.levels] != [
        (32, 3),
        (8, 6),
        (2, 6),
    ]:
        raise RuntimeError("test fixture did not produce the intended mixed-block hierarchy")
    return hierarchy


def _rhs_set(scalar_size: int) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(817)
    return (
        generator.normal(size=scalar_size),
        np.linspace(-0.8, 1.1, scalar_size, dtype=np.float64),
        generator.normal(size=scalar_size) * np.repeat((0.1, 1.0, 7.0), scalar_size // 3),
        np.eye(1, scalar_size, scalar_size // 2, dtype=np.float64).reshape(-1),
    )


def _pointer_tuple(device_hierarchy: WarpStaticMultigridHierarchy, workspace) -> tuple[int, ...]:
    pointers = [int(device_hierarchy.coarse_cholesky.ptr), int(workspace.rhs.ptr), int(workspace.correction.ptr)]
    for level in device_hierarchy.levels:
        pointers.extend((int(level.row_offsets.ptr), int(level.column_indices.ptr), int(level.matrix_values.ptr)))
        for optional in (
            level.inverse_diagonal,
            level.aggregate,
            level.prolongation_blocks,
            level.member_offsets,
            level.member_fine_nodes,
        ):
            if optional is not None:
                pointers.append(int(optional.ptr))
    for arrays in (
        workspace.level_rhs,
        workspace.level_correction,
        workspace.level_product,
        workspace.level_residual,
    ):
        pointers.extend(int(array.ptr) for array in arrays)
    pointers.append(int(workspace.coarse_intermediate.ptr))
    return tuple(pointers)


def _assert_work_matches(test: unittest.TestCase, actual, expected) -> None:
    test.assertEqual(actual.hierarchy_sha256, expected.hierarchy_sha256)
    test.assertEqual(actual.rhs_sha256, expected.rhs_sha256)
    test.assertEqual(actual.rhs_count, expected.rhs_count)
    for field in (
        "level_visits",
        "matrix_block_products",
        "smoother_block_solves",
        "restriction_block_products",
        "prolongation_block_products",
        "coarsest_factor_solves",
    ):
        test.assertEqual(getattr(actual, field), getattr(expected, field), field)


class TestWarpStaticMultigridHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hierarchy = _hierarchy()
        cls.device_hierarchy = WarpStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cpu")

    def test_contract_is_explicit_and_has_no_floating_atomics_or_spectral_factorization(self):
        self.assertEqual(KERNEL_VERSION, "mg-vbd-warp-static-v-cycle-v1")
        self.assertEqual(CONTRACT_ID, "spectral-free-multiplicative-graph-vbd-warp-static-v1")
        source = inspect.getsource(__import__(self.device_hierarchy.__module__, fromlist=["*"]))
        self.assertNotIn("wp.atomic", source)
        self.assertNotIn("np.linalg.eig", source)
        self.assertNotIn("np.linalg.svd", source)
        self.assertEqual(self.device_hierarchy.hierarchy_sha256, self.hierarchy.content_sha256)
        self.assertEqual(self.device_hierarchy.solver_contract, self.hierarchy.solver_contract)
        self.assertEqual(len(self.device_hierarchy.device_snapshot_sha256), 64)

    def test_cpu_matches_numpy_for_several_rhs_and_exact_work(self):
        workspace = self.device_hierarchy.create_workspace()
        pointers = _pointer_tuple(self.device_hierarchy, workspace)
        for rhs in _rhs_set(self.hierarchy.levels[0].matrix.scalar_size):
            with self.subTest(rhs_norm=float(np.linalg.norm(rhs))):
                expected = apply_v_cycle(self.hierarchy, rhs)
                workspace.set_rhs(rhs)
                workspace.launch()
                actual = workspace.record()
                np.testing.assert_allclose(actual.correction, expected.correction, rtol=3.0e-14, atol=3.0e-14)
                _assert_work_matches(self, actual.work, expected.work)
                self.assertEqual(actual.work.hierarchy_sha256, self.hierarchy.content_sha256)
                self.assertEqual(actual.scheduled_kernel_launches, 37)
                self.assertFalse(actual.capture_replay)
                self.assertTrue(actual.research_only)
                self.assertFalse(actual.performance_evidence)
                json.dumps(actual.deterministic_record(), allow_nan=False)
                with self.assertRaises(ValueError):
                    actual.correction[0] = 1.0
        self.assertEqual(_pointer_tuple(self.device_hierarchy, workspace), pointers)

    def test_cpu_v_cycle_is_symmetric_and_positive(self):
        x, y, *_ = _rhs_set(self.hierarchy.levels[0].matrix.scalar_size)
        workspace = self.device_hierarchy.create_workspace()
        workspace.set_rhs(x)
        workspace.launch()
        preconditioned_x = workspace.record().correction
        workspace.set_rhs(y)
        workspace.launch()
        preconditioned_y = workspace.record().correction
        left = float(x @ preconditioned_y)
        right = float(y @ preconditioned_x)
        scale = max(1.0, abs(left), abs(right))
        self.assertLessEqual(abs(left - right), 8.0e-14 * scale)
        self.assertGreater(float(x @ preconditioned_x), 0.0)
        self.assertGreater(float(y @ preconditioned_y), 0.0)

    def test_repeated_cpu_launch_is_bitwise_deterministic(self):
        rhs = _rhs_set(self.hierarchy.levels[0].matrix.scalar_size)[0]
        workspace = self.device_hierarchy.create_workspace()
        pointers = _pointer_tuple(self.device_hierarchy, workspace)
        snapshots = []
        records = []
        for _ in range(4):
            workspace.set_rhs(rhs)
            workspace.launch()
            record = workspace.record()
            snapshots.append(record.correction)
            records.append(record.deterministic_record())
        for snapshot in snapshots[1:]:
            np.testing.assert_array_equal(snapshot, snapshots[0])
        self.assertEqual(records[1:], [records[0]] * 3)
        self.assertEqual(_pointer_tuple(self.device_hierarchy, workspace), pointers)

    def test_public_device_apply_rejects_wrong_shape_and_workspace_owner(self):
        hierarchy = self.device_hierarchy
        workspace = hierarchy.create_workspace()
        scalar = wp.zeros(hierarchy.n_free_dofs, dtype=wp.float64, device="cpu")
        output = wp.empty(hierarchy.n_free, dtype=wp.vec3d, device="cpu")
        with self.assertRaisesRegex(ValueError, "rhs must be a vec3d array"):
            hierarchy.launch_apply(scalar, output, workspace)
        other = WarpStaticMultigridHierarchy.from_hierarchy(self.hierarchy, device="cpu")
        with self.assertRaisesRegex(ValueError, "different device hierarchy"):
            hierarchy.launch_apply(workspace.rhs, output, other.create_workspace())
        with self.assertRaisesRegex(ValueError, "only finite"):
            bad_rhs = np.zeros(hierarchy.n_free_dofs, dtype=np.float64)
            bad_rhs[0] = np.nan
            workspace.set_rhs(bad_rhs)

    def test_coarsest_dense_solve_has_a_hard_scalar_bound(self):
        coarse_node_count = MAX_COARSE_SCALAR_SIZE // 3 + 1
        node_count = 2 * coarse_node_count
        rest = np.column_stack(
            (
                np.arange(node_count, dtype=np.float64),
                np.zeros(node_count, dtype=np.float64),
                np.zeros(node_count, dtype=np.float64),
            )
        )
        graph = np.eye(node_count, dtype=np.float64)
        for node in range(node_count - 1):
            graph[node, node] += 1.0
            graph[node + 1, node + 1] += 1.0
            graph[node, node + 1] = -1.0
            graph[node + 1, node] = -1.0
        hierarchy = build_static_multigrid(
            np.kron(graph, np.eye(3, dtype=np.float64)),
            rest,
            np.arange(node_count, dtype=np.int64),
            mode_kind="translation",
            target_aggregate_size=2,
            minimum_aggregate_size=2,
            coarse_node_limit=coarse_node_count,
        )
        self.assertEqual(hierarchy.levels[-1].matrix.scalar_size, 3 * coarse_node_count)
        with self.assertRaisesRegex(ValueError, "exceeds fixed bound"):
            WarpStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cpu")


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestWarpStaticMultigridCudaCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.hierarchy = _hierarchy()
        cls.device_hierarchy = WarpStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cuda:0")

    def test_cuda_matches_oracle_is_symmetric_positive_and_bitwise_deterministic(self):
        hierarchy = self.hierarchy
        device_hierarchy = self.device_hierarchy
        workspace = device_hierarchy.create_workspace()
        rhs_values = _rhs_set(hierarchy.levels[0].matrix.scalar_size)
        outputs = []
        for rhs in rhs_values:
            expected = apply_v_cycle(hierarchy, rhs)
            workspace.set_rhs(rhs)
            workspace.launch()
            actual = workspace.record()
            np.testing.assert_allclose(actual.correction, expected.correction, rtol=2.0e-13, atol=2.0e-13)
            _assert_work_matches(self, actual.work, expected.work)
            outputs.append(actual.correction)
        left = float(rhs_values[0] @ outputs[1])
        right = float(rhs_values[1] @ outputs[0])
        self.assertLessEqual(abs(left - right), 4.0e-13 * max(1.0, abs(left), abs(right)))
        self.assertGreater(float(rhs_values[0] @ outputs[0]), 0.0)
        self.assertGreater(float(rhs_values[1] @ outputs[1]), 0.0)

        snapshots = []
        pointers = _pointer_tuple(device_hierarchy, workspace)
        for _ in range(4):
            workspace.set_rhs(rhs_values[2])
            workspace.launch()
            snapshots.append(workspace.record().correction)
        for snapshot in snapshots[1:]:
            np.testing.assert_array_equal(snapshot, snapshots[0])
        self.assertEqual(_pointer_tuple(device_hierarchy, workspace), pointers)

    def test_fixed_v_cycle_is_cuda_graph_capturable(self):
        rhs = _rhs_set(self.hierarchy.levels[0].matrix.scalar_size)[0]
        workspace = self.device_hierarchy.create_workspace()
        workspace.set_rhs(rhs)
        workspace.launch()
        expected = workspace.record()
        workspace.set_rhs(rhs)
        with wp.ScopedCapture(device=self.device_hierarchy.device) as capture:
            workspace.launch()
        wp.capture_launch(capture.graph)
        captured = workspace.record(capture_replay=True)

        self.assertTrue(captured.capture_replay)
        self.assertTrue(captured.research_only)
        self.assertFalse(captured.performance_evidence)
        _assert_work_matches(self, captured.work, expected.work)
        self.assertEqual(captured.scheduled_kernel_launches, expected.scheduled_kernel_launches)
        np.testing.assert_array_equal(captured.correction, expected.correction)


if __name__ == "__main__":
    unittest.main()
