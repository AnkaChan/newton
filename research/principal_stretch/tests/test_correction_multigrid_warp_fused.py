# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the launch-fused research-only static MG V-cycle."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

from research.principal_stretch import correction_multigrid_warp_fused as fused_module
from research.principal_stretch.correction_gpu import MatrixFreeStableNHOperator
from research.principal_stretch.correction_graph_vbd import DirectGraphVBDConfig
from research.principal_stretch.correction_multigrid import (
    StaticMultigridHierarchy,
    apply_v_cycle,
    build_stable_nh_rest_multigrid,
    build_static_multigrid,
)
from research.principal_stretch.correction_multigrid_warp import WarpStaticMultigridHierarchy
from research.principal_stretch.correction_multigrid_warp_fused import (
    CONTRACT_ID,
    KERNEL_VERSION,
    SCHEDULE_VERSION,
    WarpFusedStaticMultigridHierarchy,
    WarpFusedStaticMultigridPreconditioner,
)
from research.principal_stretch.solver_benchmark import build_common_problem
from research.principal_stretch.solver_scenes import build_stretch_scene


def _mixed_hierarchy(*, smooth_steps: int = 2) -> StaticMultigridHierarchy:
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
    hierarchy = build_static_multigrid(
        np.kron(graph, coordinate_metric),
        rest,
        np.arange(node_count, dtype=np.int64),
        mode_kind="rigid",
        target_aggregate_size=4,
        minimum_aggregate_size=3,
        coarse_node_limit=2,
        pre_smooth_steps=smooth_steps,
        post_smooth_steps=smooth_steps,
        static_model_sha256="a" * 64,
    )
    expected_shapes = [(32, 3), (8, 6), (2, 6)]
    actual_shapes = [(level.matrix.block_row_count, level.matrix.block_size) for level in hierarchy.levels]
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"mixed-block fixture changed shape: {actual_shapes}")
    return hierarchy


def _translation_hierarchy(node_count: int) -> StaticMultigridHierarchy:
    rest = np.column_stack(
        (
            np.arange(node_count, dtype=np.float64),
            np.zeros(node_count, dtype=np.float64),
            np.zeros(node_count, dtype=np.float64),
        )
    )
    graph = 0.5 * np.eye(node_count, dtype=np.float64)
    for node in range(node_count - 1):
        graph[node, node] += 1.0
        graph[node + 1, node + 1] += 1.0
        graph[node, node + 1] = -1.0
        graph[node + 1, node] = -1.0
    return build_static_multigrid(
        np.kron(graph, np.eye(3, dtype=np.float64)),
        rest,
        np.arange(node_count, dtype=np.int64),
        mode_kind="translation",
        target_aggregate_size=16 if node_count > 32 else 4,
        minimum_aggregate_size=8 if node_count > 32 else 3,
        coarse_node_limit=32 if node_count > 32 else 2,
        maximum_levels=4,
        pre_smooth_steps=1,
        post_smooth_steps=1,
        static_model_sha256="b" * 64,
    )


def _coarsest_only_hierarchy() -> StaticMultigridHierarchy:
    source = _translation_hierarchy(8)
    level = source.levels[-1]
    node_count = level.matrix.block_row_count
    rest = np.column_stack(
        (
            np.arange(node_count, dtype=np.float64),
            np.zeros(node_count, dtype=np.float64),
            np.zeros(node_count, dtype=np.float64),
        )
    )
    return dataclasses.replace(
        source,
        levels=(level,),
        free_vertices=np.arange(node_count, dtype=np.int64),
        rest_positions=rest,
        free_masses=np.ones(node_count, dtype=np.float64),
        content_sha256="c" * 64,
    )


def _default_stretch_hierarchy() -> StaticMultigridHierarchy:
    scene = build_stretch_scene()
    problem = build_common_problem(scene)
    positions = np.array(scene.x_current, dtype=np.float64, copy=True)
    positions[scene.pinned_indices] = scene.pin_targets
    operator = MatrixFreeStableNHOperator.from_problem(problem, positions)
    config = DirectGraphVBDConfig()
    return build_stable_nh_rest_multigrid(
        operator,
        scene.rest_q,
        mode_kind=config.mode_kind,
        target_aggregate_size=config.target_aggregate_size,
        minimum_aggregate_size=config.minimum_aggregate_size,
        coarse_node_limit=config.coarse_node_limit,
        maximum_levels=config.maximum_levels,
        pre_smooth_steps=config.pre_smooth_steps,
        post_smooth_steps=config.post_smooth_steps,
        smoother_safety=config.smoother_safety,
    )


def _rhs_set(scalar_size: int) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(817)
    return (
        generator.normal(size=scalar_size),
        np.linspace(-0.8, 1.1, scalar_size, dtype=np.float64),
        generator.normal(size=scalar_size) * np.resize(np.array((0.1, 1.0, 7.0)), scalar_size),
        np.eye(1, scalar_size, scalar_size // 2, dtype=np.float64).reshape(-1),
        np.zeros(scalar_size, dtype=np.float64),
    )


def _pointer_tuple(hierarchy: WarpFusedStaticMultigridHierarchy, workspace) -> tuple[int, ...]:
    source = hierarchy.source_hierarchy
    pointers = [
        int(source.coarse_cholesky.ptr),
        int(workspace.rhs.ptr),
        int(workspace.correction.ptr),
        int(workspace.coarse_intermediate.ptr),
    ]
    for level in source.levels:
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
        workspace.level_correction_alt,
        workspace.level_residual,
    ):
        pointers.extend(int(array.ptr) for array in arrays)
    return tuple(pointers)


def _poison_workspace(workspace) -> None:
    workspace.correction.fill_(wp.vec3d(np.nan, np.nan, np.nan))
    for arrays in (
        workspace.level_rhs,
        workspace.level_correction,
        workspace.level_correction_alt,
        workspace.level_residual,
    ):
        for array in arrays:
            array.fill_(np.nan)
    workspace.coarse_intermediate.fill_(np.nan)


def _assert_canonical_work_matches(testcase: unittest.TestCase, actual, expected) -> None:
    for field in (
        "hierarchy_sha256",
        "rhs_sha256",
        "rhs_count",
        "level_visits",
        "matrix_block_products",
        "smoother_block_solves",
        "restriction_block_products",
        "prolongation_block_products",
        "coarsest_factor_solves",
    ):
        testcase.assertEqual(getattr(actual, field), getattr(expected, field), field)


class TestWarpFusedStaticMultigridHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hierarchy = _mixed_hierarchy(smooth_steps=2)
        cls.source = WarpStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cpu")
        cls.device_hierarchy = WarpFusedStaticMultigridHierarchy.from_device_hierarchy(cls.source)

    def test_contract_source_sharing_specialization_and_launch_path(self):
        hierarchy = self.device_hierarchy
        self.assertEqual(KERNEL_VERSION, "mg-vbd-warp-static-v-cycle-fused-v1")
        self.assertEqual(CONTRACT_ID, "spectral-free-multiplicative-graph-vbd-warp-static-fused-v1")
        self.assertEqual(SCHEDULE_VERSION, "zero-jacobi-residual-restrict-recurse-prolong-ping-pong-v1")
        self.assertIs(hierarchy.source_hierarchy, self.source)
        self.assertIs(hierarchy.levels, self.source.levels)
        self.assertEqual(int(hierarchy.coarse_cholesky.ptr), int(self.source.coarse_cholesky.ptr))
        self.assertEqual(len(hierarchy.schedule_sha256), 64)
        self.assertEqual(len(hierarchy.device_snapshot_sha256), 64)
        self.assertNotEqual(hierarchy.device_snapshot_sha256, hierarchy.source_device_snapshot_sha256)
        self.assertEqual(hierarchy.scheduled_kernel_launches, 17)
        self.assertEqual(self.source.scheduled_kernel_launches, 37)

        source = inspect.getsource(fused_module)
        self.assertNotIn("wp.atomic", source)
        self.assertNotIn("np.linalg.eig", source)
        self.assertNotIn("np.linalg.svd", source)
        for size in (3, 6):
            self.assertIn(f"_zero_start_block_jacobi_{size}", source)
            self.assertIn(f"_block_csr_residual_{size}", source)
            self.assertIn(f"_fused_block_jacobi_{size}", source)
        for method in (
            hierarchy.launch_apply,
            hierarchy._launch_level,
            hierarchy._launch_smooth,
        ):
            launch_source = inspect.getsource(method)
            self.assertNotIn("wp.empty", launch_source)
            self.assertNotIn("wp.zeros", launch_source)
            self.assertNotIn(".numpy(", launch_source)
            self.assertNotIn("synchronize", launch_source)

    def test_cpu_oracle_exact_schedule_work_hashes_and_pointer_stability(self):
        hierarchy = self.device_hierarchy
        workspace = hierarchy.create_workspace()
        pointers = _pointer_tuple(hierarchy, workspace)
        for rhs in _rhs_set(self.hierarchy.levels[0].matrix.scalar_size):
            with self.subTest(rhs_norm=float(np.linalg.norm(rhs))):
                expected = apply_v_cycle(self.hierarchy, rhs)
                workspace.set_rhs(rhs)
                with mock.patch.object(fused_module.wp, "launch", wraps=wp.launch) as launch:
                    workspace.launch()
                self.assertEqual(launch.call_count, 17)
                actual = workspace.record()
                np.testing.assert_allclose(actual.correction, expected.correction, rtol=4.0e-14, atol=4.0e-14)
                _assert_canonical_work_matches(self, actual.work, expected.work)
                self.assertEqual(actual.scheduled_kernel_launches, 17)
                self.assertEqual(
                    actual.physical_work.matrix_block_products_executed
                    + actual.physical_work.matrix_block_products_elided_zero_start,
                    actual.work.matrix_block_products,
                )
                self.assertEqual(
                    actual.physical_work.matrix_block_products_elided_zero_start,
                    sum(level.matrix.stored_block_count for level in self.hierarchy.levels[:-1]),
                )
                self.assertEqual(actual.physical_work.matrix_kernel_launches, 8)
                self.assertFalse(actual.capture_replay)
                self.assertTrue(actual.research_only)
                self.assertFalse(actual.performance_evidence)
                json.dumps(actual.deterministic_record(), allow_nan=False)
                with self.assertRaises(ValueError):
                    actual.correction.setflags(write=True)
        self.assertEqual(_pointer_tuple(hierarchy, workspace), pointers)

    def test_symmetric_positive_ping_pong_and_poison_independence(self):
        x, y, rhs, *_ = _rhs_set(self.hierarchy.levels[0].matrix.scalar_size)
        workspace = self.device_hierarchy.create_workspace()
        workspace.set_rhs(x)
        workspace.launch()
        preconditioned_x = workspace.record().correction
        workspace.set_rhs(y)
        workspace.launch()
        preconditioned_y = workspace.record().correction
        left = float(x @ preconditioned_y)
        right = float(y @ preconditioned_x)
        self.assertLessEqual(abs(left - right), 1.0e-13 * max(1.0, abs(left), abs(right)))
        self.assertGreater(float(x @ preconditioned_x), 0.0)
        self.assertGreater(float(y @ preconditioned_y), 0.0)

        expected = apply_v_cycle(self.hierarchy, rhs)
        pointers = _pointer_tuple(self.device_hierarchy, workspace)
        for _ in range(3):
            _poison_workspace(workspace)
            workspace.set_rhs(rhs)
            workspace.launch()
            actual = workspace.record()
            np.testing.assert_allclose(actual.correction, expected.correction, rtol=4.0e-14, atol=4.0e-14)
            self.assertEqual(_pointer_tuple(self.device_hierarchy, workspace), pointers)
        for level_index in range(len(self.hierarchy.levels) - 1):
            self.assertEqual(
                int(workspace._final_level_correction(level_index).ptr),
                int(workspace.level_correction_alt[level_index].ptr),
            )
            with self.assertRaisesRegex(RuntimeError, "distinct input and output"):
                self.device_hierarchy._launch_smooth(
                    level_index,
                    workspace.level_rhs[level_index],
                    workspace.level_correction[level_index],
                    workspace.level_correction[level_index],
                )

    def test_mismatch_nonfinite_and_tampered_evidence_fail_closed(self):
        hierarchy = self.device_hierarchy
        workspace = hierarchy.create_workspace()
        other = WarpFusedStaticMultigridHierarchy.from_hierarchy(self.hierarchy, device="cpu")
        with self.assertRaisesRegex(ValueError, "different fused device hierarchy"):
            hierarchy.launch_apply(workspace.rhs, workspace.correction, other.create_workspace())
        bad_rhs = np.zeros(hierarchy.n_free_dofs, dtype=np.float64)
        bad_rhs[0] = np.inf
        with self.assertRaisesRegex(ValueError, "only finite"):
            workspace.set_rhs(bad_rhs)

        rhs = _rhs_set(hierarchy.n_free_dofs)[0]
        workspace.set_rhs(rhs)
        workspace.launch()
        record = workspace.record()
        changed = np.array(record.correction, copy=True)
        changed[0] += 1.0
        with self.assertRaisesRegex(ValueError, "result hash"):
            dataclasses.replace(record, correction=changed)
        tampered_work = dataclasses.replace(record.work, matrix_block_products=record.work.matrix_block_products + 1)
        with self.assertRaisesRegex(ValueError, "untampered"):
            dataclasses.replace(record, work=tampered_work)
        with self.assertRaisesRegex(ValueError, "physical-work content_sha256"):
            dataclasses.replace(
                record.physical_work,
                matrix_block_products_executed=record.physical_work.matrix_block_products_executed + 1,
            )
        with self.assertRaisesRegex(ValueError, "complete fused V-cycle record"):
            dataclasses.replace(record, capture_replay=True)

        workspace.correction.fill_(wp.vec3d(np.nan, np.nan, np.nan))
        with self.assertRaisesRegex(FloatingPointError, "must remain finite"):
            workspace.record()
        original_schedule = workspace._schedule_sha256
        workspace._schedule_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "schedule binding"):
            workspace.launch()
        workspace._schedule_sha256 = original_schedule

    def test_preconditioner_boundary_retains_compatible_canonical_evidence(self):
        preconditioner = WarpFusedStaticMultigridPreconditioner(self.device_hierarchy)
        workspace = preconditioner.create_application_workspace()
        rhs = _rhs_set(self.device_hierarchy.n_free_dofs)[1]
        workspace.set_rhs(rhs)
        preconditioner.launch_apply(workspace.rhs, workspace.correction, workspace)
        application = preconditioner.record_application(0, workspace, capture_replay=False)
        self.assertEqual(preconditioner.application_kernel_launches, 17)
        self.assertEqual(application.scheduled_kernel_launches, 17)
        self.assertEqual(application.static_preconditioner_sha256, self.hierarchy.content_sha256)
        self.assertTrue(application.output_finite)
        self.assertFalse(application.capture_replay)

    def test_source_metadata_pointer_and_finite_content_tamper_fail_closed(self):
        source = WarpStaticMultigridHierarchy.from_hierarchy(self.hierarchy, device="cpu")
        hierarchy = WarpFusedStaticMultigridHierarchy.from_device_hierarchy(source)
        workspace = hierarchy.create_workspace()
        workspace.set_rhs(_rhs_set(hierarchy.n_free_dofs)[0])

        original_steps = source.pre_smooth_steps
        source.pre_smooth_steps = original_steps + 1
        with self.assertRaisesRegex(RuntimeError, "hierarchy identity changed"):
            workspace.launch()
        source.pre_smooth_steps = original_steps

        level = source.levels[0]
        original_values = level.matrix_values
        replacement = wp.empty(original_values.shape, dtype=wp.float64, device="cpu")
        object.__setattr__(level, "matrix_values", replacement)
        with self.assertRaisesRegex(RuntimeError, "hierarchy identity changed"):
            workspace.launch()
        object.__setattr__(level, "matrix_values", original_values)

        original_values.fill_(0.0)
        workspace.launch()
        with self.assertRaisesRegex(RuntimeError, "static device hierarchy content changed"):
            workspace.record()

    def test_coarsest_only_schedule_is_exactly_three(self):
        hierarchy = _coarsest_only_hierarchy()
        device_hierarchy = WarpFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cpu")
        workspace = device_hierarchy.create_workspace()
        rhs = np.linspace(-0.4, 0.9, hierarchy.levels[0].matrix.scalar_size)
        expected = apply_v_cycle(hierarchy, rhs)
        workspace.set_rhs(rhs)
        with mock.patch.object(fused_module.wp, "launch", wraps=wp.launch) as launch:
            workspace.launch()
        self.assertEqual(launch.call_count, 3)
        self.assertEqual(device_hierarchy.scheduled_kernel_launches, 3)
        self.assertEqual(len(workspace.level_correction_alt), 0)
        np.testing.assert_allclose(workspace.record().correction, expected.correction, rtol=3.0e-14, atol=3.0e-14)

    def test_257_block_row_boundary_and_translation_transfer(self):
        hierarchy = _translation_hierarchy(257)
        self.assertEqual(
            [(level.matrix.block_row_count, level.matrix.block_size) for level in hierarchy.levels],
            [(257, 3), (16, 3)],
        )
        device_hierarchy = WarpFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cpu")
        workspace = device_hierarchy.create_workspace()
        rhs = np.zeros(hierarchy.levels[0].matrix.scalar_size, dtype=np.float64)
        rhs[-3:] = (0.25, -0.5, 1.0)
        expected = apply_v_cycle(hierarchy, rhs)
        workspace.set_rhs(rhs)
        workspace.launch()
        actual = workspace.record()
        self.assertEqual(actual.scheduled_kernel_launches, 8)
        np.testing.assert_allclose(actual.correction, expected.correction, rtol=4.0e-14, atol=4.0e-14)
        self.assertGreater(float(np.linalg.norm(actual.correction[-3:])), 0.0)


class TestWarpFusedDefaultStretchCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hierarchy = _default_stretch_hierarchy()
        cls.device_hierarchy = WarpFusedStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cpu")

    def test_real_shared_default_hierarchy_matches_oracle_and_freezes_18_launch_schedule(self):
        shapes = [
            (level.matrix.block_row_count, level.matrix.block_size, level.matrix.stored_block_count)
            for level in self.hierarchy.levels
        ]
        self.assertEqual(shapes, [(144, 3, 1378), (32, 6, 270), (8, 6, 38), (2, 6, 4)])
        self.assertEqual(
            self.hierarchy.content_sha256, "2e08bcce552d135e3ec8010c6100ee6b18e6157e0c4e7013c2894280a1e9d493"
        )
        self.assertEqual(self.device_hierarchy.scheduled_kernel_launches, 18)
        self.assertEqual(self.device_hierarchy.source_hierarchy.scheduled_kernel_launches, 36)
        direct_correction_launches = 2 + 4 * ((7 + 2 * self.device_hierarchy.scheduled_kernel_launches) + 8)
        self.assertEqual(direct_correction_launches, 206)

        rhs = np.random.default_rng(230817).normal(size=self.hierarchy.levels[0].matrix.scalar_size)
        expected = apply_v_cycle(self.hierarchy, rhs)
        workspace = self.device_hierarchy.create_workspace()
        workspace.set_rhs(rhs)
        with mock.patch.object(fused_module.wp, "launch", wraps=wp.launch) as launch:
            workspace.launch()
        self.assertEqual(launch.call_count, 18)
        actual = workspace.record()
        np.testing.assert_allclose(actual.correction, expected.correction, rtol=7.0e-14, atol=7.0e-14)
        _assert_canonical_work_matches(self, actual.work, expected.work)
        self.assertEqual(actual.work.matrix_block_products, 5058)
        self.assertEqual(actual.physical_work.matrix_block_products_executed, 3372)
        self.assertEqual(actual.physical_work.matrix_block_products_elided_zero_start, 1686)
        self.assertEqual(actual.physical_work.zero_start_block_solves, 184)
        self.assertEqual(actual.physical_work.fused_smoother_block_solves, 184)
        self.assertFalse(actual.performance_evidence)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestWarpFusedStaticMultigridCudaCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.hierarchy = _mixed_hierarchy(smooth_steps=2)
        cls.device_hierarchy = WarpFusedStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cuda:0")

    def test_cuda_oracle_symmetry_positive_determinism_and_pointers(self):
        workspace = self.device_hierarchy.create_workspace()
        rhs_values = _rhs_set(self.hierarchy.levels[0].matrix.scalar_size)
        pointers = _pointer_tuple(self.device_hierarchy, workspace)
        outputs = []
        for rhs in rhs_values:
            expected = apply_v_cycle(self.hierarchy, rhs)
            workspace.set_rhs(rhs)
            workspace.launch()
            actual = workspace.record()
            np.testing.assert_allclose(actual.correction, expected.correction, rtol=3.0e-13, atol=3.0e-13)
            _assert_canonical_work_matches(self, actual.work, expected.work)
            outputs.append(actual.correction)
        left = float(rhs_values[0] @ outputs[1])
        right = float(rhs_values[1] @ outputs[0])
        self.assertLessEqual(abs(left - right), 6.0e-13 * max(1.0, abs(left), abs(right)))
        self.assertGreater(float(rhs_values[0] @ outputs[0]), 0.0)
        self.assertGreater(float(rhs_values[1] @ outputs[1]), 0.0)

        snapshots = []
        for _ in range(4):
            workspace.set_rhs(rhs_values[2])
            workspace.launch()
            snapshots.append(workspace.record().correction)
        for snapshot in snapshots[1:]:
            np.testing.assert_array_equal(snapshot, snapshots[0])
        self.assertEqual(_pointer_tuple(self.device_hierarchy, workspace), pointers)

    def test_capture_replay_accepts_changed_rhs_and_overwrites_poison(self):
        rhs_values = _rhs_set(self.hierarchy.levels[0].matrix.scalar_size)
        workspace = self.device_hierarchy.create_workspace()
        pointers = _pointer_tuple(self.device_hierarchy, workspace)
        workspace.set_rhs(rhs_values[0])
        workspace.launch()
        workspace.record()

        workspace.set_rhs(rhs_values[0])
        with wp.ScopedCapture(device=self.device_hierarchy.device) as capture:
            workspace.launch()

        workspace.set_rhs(rhs_values[1])
        wp.capture_launch(capture.graph)
        changed = workspace.record(capture_replay=True)
        expected_changed = apply_v_cycle(self.hierarchy, rhs_values[1])
        np.testing.assert_allclose(changed.correction, expected_changed.correction, rtol=3.0e-13, atol=3.0e-13)
        self.assertTrue(changed.capture_replay)
        self.assertFalse(changed.performance_evidence)

        _poison_workspace(workspace)
        workspace.set_rhs(rhs_values[2])
        wp.capture_launch(capture.graph)
        recovered = workspace.record(capture_replay=True)
        expected_recovered = apply_v_cycle(self.hierarchy, rhs_values[2])
        np.testing.assert_allclose(recovered.correction, expected_recovered.correction, rtol=3.0e-13, atol=3.0e-13)
        self.assertNotEqual(recovered.work.rhs_sha256, changed.work.rhs_sha256)
        self.assertNotEqual(recovered.work.result_sha256, changed.work.result_sha256)
        self.assertEqual(_pointer_tuple(self.device_hierarchy, workspace), pointers)

    def test_real_default_stretch_cuda_oracle_uses_18_launches(self):
        hierarchy = _default_stretch_hierarchy()
        device_hierarchy = WarpFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cuda:0")
        workspace = device_hierarchy.create_workspace()
        rhs = np.random.default_rng(99117).normal(size=hierarchy.levels[0].matrix.scalar_size)
        expected = apply_v_cycle(hierarchy, rhs)
        workspace.set_rhs(rhs)
        workspace.launch()
        actual = workspace.record()
        self.assertEqual(actual.scheduled_kernel_launches, 18)
        np.testing.assert_allclose(actual.correction, expected.correction, rtol=4.0e-13, atol=4.0e-13)
        _assert_canonical_work_matches(self, actual.work, expected.work)
        self.assertFalse(actual.performance_evidence)


if __name__ == "__main__":
    unittest.main()
