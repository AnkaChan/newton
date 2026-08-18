# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the scalar-row launch-fused research MG V-cycle."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

from research.principal_stretch import correction_multigrid_warp_scalar_fused as scalar_fused_module
from research.principal_stretch.correction_gpu import MatrixFreeStableNHOperator
from research.principal_stretch.correction_graph_vbd import DirectGraphVBDConfig
from research.principal_stretch.correction_multigrid import (
    StaticMultigridHierarchy,
    apply_v_cycle,
    build_stable_nh_rest_multigrid,
    build_static_multigrid,
)
from research.principal_stretch.correction_multigrid_warp import WarpStaticMultigridHierarchy
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    CONTRACT_ID,
    EXTERNAL_SHARED_PUBLICATION_ROUTE,
    KERNEL_VERSION,
    PUBLICATION_VERSION,
    SCHEDULE_VERSION,
    STANDALONE_PUBLICATION_ROUTE,
    WarpScalarFusedStaticMultigridHierarchy,
    WarpScalarFusedStaticMultigridPreconditioner,
)
from research.principal_stretch.solver_benchmark import build_common_problem
from research.principal_stretch.solver_scenes import build_stretch_scene


def _mixed_hierarchy(*, smooth_steps: int) -> StaticMultigridHierarchy:
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
    shapes = [(level.matrix.block_row_count, level.matrix.block_size) for level in hierarchy.levels]
    if shapes != [(32, 3), (8, 6), (2, 6)]:
        raise RuntimeError(f"mixed-block fixture changed shape: {shapes}")
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
    generator = np.random.default_rng(180817)
    return (
        generator.normal(size=scalar_size),
        np.linspace(-0.8, 1.1, scalar_size, dtype=np.float64),
        generator.normal(size=scalar_size) * np.resize(np.array((0.1, 1.0, 7.0)), scalar_size),
        np.eye(1, scalar_size, scalar_size // 2, dtype=np.float64).reshape(-1),
        np.zeros(scalar_size, dtype=np.float64),
    )


def _pointer_tuple(hierarchy: WarpScalarFusedStaticMultigridHierarchy, workspace) -> tuple[int, ...]:
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


class TestWarpScalarFusedStaticMultigridHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hierarchy = _mixed_hierarchy(smooth_steps=2)
        cls.source = WarpStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cpu")
        cls.device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_device_hierarchy(cls.source)

    def test_contract_scalar_owners_source_sharing_and_launch_path(self):
        hierarchy = self.device_hierarchy
        self.assertEqual(KERNEL_VERSION, "mg-vbd-warp-static-v-cycle-scalar-fused-v3")
        self.assertEqual(CONTRACT_ID, "spectral-free-multiplicative-graph-vbd-warp-static-scalar-fused-v1")
        self.assertEqual(
            SCHEDULE_VERSION,
            "scalar-core-and-versioned-publication-routes-v4",
        )
        self.assertEqual(PUBLICATION_VERSION, "scalar-fused-v-cycle-publication-routes-v2")
        self.assertEqual(EXTERNAL_SHARED_PUBLICATION_ROUTE, "external-shared-owner-scalar-to-vec3")
        self.assertIs(hierarchy.source_hierarchy, self.source)
        self.assertIs(hierarchy.levels, self.source.levels)
        self.assertEqual(int(hierarchy.coarse_cholesky.ptr), int(self.source.coarse_cholesky.ptr))
        self.assertEqual(len(hierarchy.schedule_sha256), 64)
        self.assertNotEqual(hierarchy.device_snapshot_sha256, hierarchy.source_device_snapshot_sha256)
        self.assertEqual(hierarchy.scheduled_kernel_launches, 22)
        self.assertEqual(hierarchy.core_kernel_launches, 21)
        self.assertNotEqual(hierarchy.schedule_sha256, hierarchy.core_schedule_sha256)
        self.assertNotEqual(hierarchy.device_snapshot_sha256, hierarchy.core_device_snapshot_sha256)
        self.assertEqual(self.source.scheduled_kernel_launches, 37)

        source = inspect.getsource(scalar_fused_module)
        self.assertNotIn("wp.atomic", source)
        self.assertNotIn("np.linalg.eig", source)
        self.assertNotIn("np.linalg.svd", source)
        for kernel in (
            "_fused_root_ingress_zero_start_scalar_jacobi",
            "_zero_start_scalar_jacobi",
            "_scalar_csr_residual",
            "_out_of_place_scalar_jacobi",
        ):
            self.assertIn(kernel, source)
        for method in (
            hierarchy.launch_apply,
            hierarchy._launch_level,
            hierarchy._launch_residual,
            hierarchy._launch_jacobi,
        ):
            launch_source = inspect.getsource(method)
            self.assertNotIn("wp.empty", launch_source)
            self.assertNotIn("wp.zeros", launch_source)
            self.assertNotIn(".numpy(", launch_source)
            self.assertNotIn("synchronize", launch_source)

    def test_cpu_oracle_p1_p2_exact_launches_work_and_fixed_b_result(self):
        for steps, expected_launches, expected_matrix_launches in ((1, 14, 4), (2, 22, 8)):
            with self.subTest(steps=steps):
                hierarchy = _mixed_hierarchy(smooth_steps=steps)
                device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cpu")
                workspace = device_hierarchy.create_workspace()
                pointers = _pointer_tuple(device_hierarchy, workspace)
                for rhs in _rhs_set(hierarchy.levels[0].matrix.scalar_size):
                    expected = apply_v_cycle(hierarchy, rhs)
                    workspace.set_rhs(rhs)
                    with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                        workspace.launch()
                    self.assertEqual(launch.call_count, expected_launches)
                    calls = launch.call_args_list

                    def launch_dims(kernel, calls=calls) -> list[int]:
                        return [call.kwargs["dim"] for call in calls if call.args[0] is kernel]

                    noncoarse_sizes = [level.matrix.scalar_size for level in hierarchy.levels[:-1]]
                    self.assertCountEqual(
                        launch_dims(scalar_fused_module._zero_start_scalar_jacobi),
                        noncoarse_sizes[1:],
                    )
                    self.assertEqual(
                        launch_dims(scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi),
                        noncoarse_sizes[:1],
                    )
                    self.assertEqual(launch_dims(scalar_fused_module._copy_vec3_to_scalar), [])
                    self.assertEqual(
                        launch_dims(scalar_fused_module._copy_scalar_to_vec3),
                        [hierarchy.levels[0].matrix.block_row_count],
                    )
                    self.assertCountEqual(
                        launch_dims(scalar_fused_module._scalar_csr_residual),
                        [size for size in noncoarse_sizes for _ in range(2 * steps)],
                    )
                    self.assertCountEqual(
                        launch_dims(scalar_fused_module._out_of_place_scalar_jacobi),
                        [size for size in noncoarse_sizes for _ in range(2 * steps - 1)],
                    )
                    self.assertCountEqual(
                        launch_dims(scalar_fused_module._restrict_owned_rows),
                        [level.matrix.scalar_size for level in hierarchy.levels[1:]],
                    )
                    self.assertCountEqual(
                        launch_dims(scalar_fused_module._prolong_add_owned_rows),
                        noncoarse_sizes,
                    )
                    actual = workspace.record()
                    np.testing.assert_allclose(actual.correction, expected.correction, rtol=4.0e-14, atol=4.0e-14)
                    _assert_canonical_work_matches(self, actual.work, expected.work)
                    self.assertEqual(actual.scheduled_kernel_launches, expected_launches)
                    self.assertEqual(actual.physical_work.core_kernel_launches, expected_launches - 1)
                    self.assertEqual(actual.physical_work.publication_kernel_launches, 1)
                    self.assertEqual(actual.physical_work.publication_version, PUBLICATION_VERSION)
                    self.assertEqual(actual.physical_work.publication_route, STANDALONE_PUBLICATION_ROUTE)
                    self.assertEqual(actual.physical_work.matrix_kernel_launches, expected_matrix_launches)
                    self.assertEqual(actual.physical_work.jacobi_kernel_launches, expected_matrix_launches)
                    stored_blocks = sum(level.matrix.stored_block_count for level in hierarchy.levels[:-1])
                    block_rows = sum(level.matrix.block_row_count for level in hierarchy.levels[:-1])
                    self.assertEqual(
                        actual.physical_work.matrix_block_products_executed,
                        stored_blocks * 2 * steps,
                    )
                    self.assertEqual(
                        actual.physical_work.matrix_block_products_elided_zero_start,
                        stored_blocks,
                    )
                    self.assertEqual(actual.physical_work.zero_start_block_solves, block_rows)
                    self.assertEqual(actual.physical_work.root_ingress_zero_start_fusions, 1)
                    self.assertEqual(
                        actual.physical_work.out_of_place_jacobi_block_solves,
                        block_rows * (2 * steps - 1),
                    )
                    self.assertEqual(
                        actual.physical_work.matrix_block_products_executed
                        + actual.physical_work.matrix_block_products_elided_zero_start,
                        actual.work.matrix_block_products,
                    )
                    self.assertFalse(actual.capture_replay)
                    self.assertTrue(actual.research_only)
                    self.assertFalse(actual.performance_evidence)
                    json.dumps(actual.deterministic_record(), allow_nan=False)
                    with self.assertRaises(ValueError):
                        actual.correction.setflags(write=True)

                    _poison_workspace(workspace)
                    workspace.set_rhs(rhs)
                    with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as core_launch:
                        workspace.launch_core()
                    self.assertEqual(core_launch.call_count, expected_launches - 1)
                    self.assertFalse(
                        any(
                            call.args[0] is scalar_fused_module._copy_scalar_to_vec3
                            for call in core_launch.call_args_list
                        )
                    )
                    with self.assertRaisesRegex(ValueError, "solver-private"):
                        workspace.record_core_application(token=object())
                    core_record = workspace.record_core_application(token=scalar_fused_module._CORE_RECORD_TOKEN)
                    np.testing.assert_array_equal(
                        core_record.correction.view(np.uint64),
                        actual.correction.view(np.uint64),
                    )
                    self.assertEqual(core_record.scheduled_kernel_launches, expected_launches - 1)
                    self.assertEqual(core_record.physical_work.core_kernel_launches, expected_launches - 1)
                    self.assertEqual(core_record.physical_work.publication_kernel_launches, 0)
                    self.assertEqual(core_record.physical_work.publication_version, PUBLICATION_VERSION)
                    self.assertEqual(core_record.physical_work.publication_route, EXTERNAL_SHARED_PUBLICATION_ROUTE)
                    self.assertEqual(core_record.schedule_sha256, device_hierarchy.core_schedule_sha256)
                    self.assertEqual(core_record.device_snapshot_sha256, device_hierarchy.core_device_snapshot_sha256)
                for level_index in range(len(hierarchy.levels) - 1):
                    self.assertEqual(
                        int(workspace._final_level_correction(level_index).ptr),
                        int(workspace.level_correction_alt[level_index].ptr),
                    )
                self.assertEqual(_pointer_tuple(device_hierarchy, workspace), pointers)

    def test_fused_root_ingress_is_bitwise_legacy_for_random_signed_zero_and_nonfinite(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cpu",
        )
        root = hierarchy.levels[0]
        self.assertEqual(root.block_size, 3)
        self.assertIsNotNone(root.inverse_diagonal)
        self.assertIsNotNone(root.omega)
        generator = np.random.default_rng(180818)
        random_rhs = generator.normal(size=(hierarchy.n_free, 3))
        signed_zero_rhs = np.zeros((hierarchy.n_free, 3), dtype=np.float64)
        signed_zero_rhs.reshape(-1)[::2] = -0.0
        nonfinite_rhs = generator.normal(size=(hierarchy.n_free, 3))
        nonfinite_rhs.reshape(-1)[:4] = (np.inf, -np.inf, np.nan, -0.0)

        for name, host_rhs in (
            ("random", random_rhs),
            ("signed_zero", signed_zero_rhs),
            ("nonfinite", nonfinite_rhs),
        ):
            with self.subTest(name=name):
                external_rhs = wp.array(host_rhs, dtype=wp.vec3d, device="cpu")
                legacy_rhs = wp.empty(root.scalar_size, dtype=wp.float64, device="cpu")
                legacy_correction = wp.empty(root.scalar_size, dtype=wp.float64, device="cpu")
                fused_rhs = wp.empty(root.scalar_size, dtype=wp.float64, device="cpu")
                fused_correction = wp.empty(root.scalar_size, dtype=wp.float64, device="cpu")
                wp.launch(
                    scalar_fused_module._copy_vec3_to_scalar,
                    dim=hierarchy.n_free,
                    inputs=[external_rhs, legacy_rhs],
                    device="cpu",
                )
                wp.launch(
                    scalar_fused_module._zero_start_scalar_jacobi,
                    dim=root.scalar_size,
                    inputs=[legacy_rhs, root.inverse_diagonal, root.block_size, root.omega, legacy_correction],
                    device="cpu",
                )
                wp.launch(
                    scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi,
                    dim=root.scalar_size,
                    inputs=[external_rhs, root.inverse_diagonal, root.omega, fused_rhs, fused_correction],
                    device="cpu",
                )
                np.testing.assert_array_equal(
                    fused_rhs.numpy().view(np.uint64),
                    legacy_rhs.numpy().view(np.uint64),
                )
                np.testing.assert_array_equal(
                    fused_correction.numpy().view(np.uint64),
                    legacy_correction.numpy().view(np.uint64),
                )
                np.testing.assert_array_equal(
                    fused_rhs.numpy().view(np.uint64),
                    host_rhs.reshape(-1).view(np.uint64),
                )

    def test_alias_spans_reject_fused_buffer_hazards_and_allow_rhs_output_identity(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cpu",
        )
        rhs = _rhs_set(hierarchy.n_free_dofs)[0]

        for internal_name, get_internal in (
            ("level_rhs", lambda workspace: workspace.level_rhs[0]),
            ("level_correction", lambda workspace: workspace.level_correction[0]),
            ("final_correction", lambda workspace: workspace.final_scalar_correction),
        ):
            for byte_offset in (0, 8):
                with self.subTest(internal_name=internal_name, byte_offset=byte_offset):
                    workspace = hierarchy.create_workspace()
                    internal = get_internal(workspace)
                    aliased_rhs = wp.array(
                        ptr=int(internal.ptr) + byte_offset,
                        dtype=wp.vec3d,
                        shape=(hierarchy.n_free,),
                        device=hierarchy.device,
                        copy=False,
                    )
                    with self.assertRaisesRegex(ValueError, "rhs and root"):
                        hierarchy.launch_apply(aliased_rhs, workspace.correction, workspace)

        for internal_name, get_internal in (
            ("level_rhs", lambda workspace: workspace.level_rhs[0]),
            ("level_correction", lambda workspace: workspace.level_correction[0]),
            ("final_correction", lambda workspace: workspace._final_level_correction(0)),
        ):
            with self.subTest(output_internal_name=internal_name):
                workspace = hierarchy.create_workspace()
                internal = get_internal(workspace)
                aliased_output = wp.array(
                    ptr=int(internal.ptr) + 8,
                    dtype=wp.vec3d,
                    shape=(hierarchy.n_free,),
                    device=hierarchy.device,
                    copy=False,
                )
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaisesRegex(ValueError, "output and root"):
                        hierarchy.launch_apply(workspace.rhs, aliased_output, workspace)
                self.assertEqual(launch.call_count, 0)

        workspace = hierarchy.create_workspace()
        shifted_primary = wp.array(
            ptr=int(workspace.level_rhs[0].ptr) + 8,
            dtype=wp.float64,
            shape=(hierarchy.n_free_dofs,),
            device=hierarchy.device,
            copy=False,
        )
        workspace.level_correction = (shifted_primary, *workspace.level_correction[1:])
        workspace._persistent_arrays = workspace._current_arrays()
        workspace._persistent_pointers = tuple(int(array.ptr) for array in workspace._persistent_arrays)
        with self.assertRaisesRegex(ValueError, "level_rhs.*primary correction"):
            hierarchy.launch_apply(workspace.rhs, workspace.correction, workspace)

        shared = wp.array(rhs.reshape(-1, 3), dtype=wp.vec3d, device="cpu")
        workspace = hierarchy.create_workspace()
        pointers = _pointer_tuple(hierarchy, workspace)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            hierarchy.launch_apply(shared, shared, workspace)
        self.assertEqual(launch.call_count, hierarchy.scheduled_kernel_launches)
        expected = apply_v_cycle(_mixed_hierarchy(smooth_steps=1), rhs)
        np.testing.assert_allclose(shared.numpy().reshape(-1), expected.correction, rtol=4.0e-14, atol=4.0e-14)
        internal_record = workspace.record_internal_application()
        np.testing.assert_array_equal(workspace.level_rhs[0].numpy(), rhs)
        np.testing.assert_allclose(internal_record.correction, expected.correction, rtol=4.0e-14, atol=4.0e-14)
        self.assertEqual(_pointer_tuple(hierarchy, workspace), pointers)

    def test_core_and_full_publication_are_bitwise_for_signed_zero_and_nonfinite(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cpu",
        )
        generator = np.random.default_rng(180819)
        random_rhs = generator.normal(size=(hierarchy.n_free, 3))
        signed_zero_rhs = np.zeros((hierarchy.n_free, 3), dtype=np.float64)
        signed_zero_rhs.reshape(-1)[::2] = -0.0
        nonfinite_rhs = generator.normal(size=(hierarchy.n_free, 3))
        nonfinite_rhs.reshape(-1)[:4] = (np.inf, -np.inf, np.nan, -0.0)
        for name, rhs in (
            ("random", random_rhs),
            ("signed_zero", signed_zero_rhs),
            ("nonfinite", nonfinite_rhs),
        ):
            with self.subTest(name=name):
                full = hierarchy.create_workspace()
                core = hierarchy.create_workspace()
                _poison_workspace(full)
                _poison_workspace(core)
                full.rhs.assign(rhs)
                core.rhs.assign(rhs)
                full.launch()
                core.launch_core()
                np.testing.assert_array_equal(
                    full.correction.numpy().reshape(-1).view(np.uint64),
                    core.final_scalar_correction.numpy().view(np.uint64),
                )

    def test_symmetric_positive_poison_independence_and_no_in_place_jacobi(self):
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
            np.testing.assert_allclose(workspace.record().correction, expected.correction, rtol=4.0e-14, atol=4.0e-14)
            self.assertEqual(_pointer_tuple(self.device_hierarchy, workspace), pointers)
        with self.assertRaisesRegex(RuntimeError, "distinct input and output"):
            self.device_hierarchy._launch_jacobi(
                0,
                workspace.level_residual[0],
                workspace.level_correction[0],
                workspace.level_correction[0],
            )

    def test_mismatch_nonfinite_static_and_evidence_tamper_fail_closed(self):
        hierarchy = self.device_hierarchy
        workspace = hierarchy.create_workspace()
        other = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(self.hierarchy, device="cpu")
        with self.assertRaisesRegex(ValueError, "different scalar-fused device hierarchy"):
            hierarchy.launch_apply(workspace.rhs, workspace.correction, other.create_workspace())
        bad_rhs = np.zeros(hierarchy.n_free_dofs, dtype=np.float64)
        bad_rhs[0] = np.inf
        with self.assertRaisesRegex(ValueError, "only finite"):
            workspace.set_rhs(bad_rhs)
        workspace.rhs.assign(bad_rhs.reshape(-1, 3))
        workspace.launch()
        with self.assertRaisesRegex(FloatingPointError, "must remain finite"):
            workspace.record()

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
        with self.assertRaisesRegex(ValueError, "complete scalar-fused V-cycle record"):
            dataclasses.replace(record, capture_replay=True)

        for field_name, forged_value in (
            ("root_ingress_zero_start_fusions", 0),
            ("zero_start_block_solves", record.physical_work.zero_start_block_solves + 1),
            ("matrix_kernel_launches", record.physical_work.matrix_kernel_launches + 1),
            ("core_kernel_launches", record.physical_work.core_kernel_launches + 1),
            ("publication_kernel_launches", 0),
            ("publication_version", PUBLICATION_VERSION + "-forged"),
            ("publication_route", EXTERNAL_SHARED_PUBLICATION_ROUTE),
            ("schedule_sha256", "0" * 64),
            ("content_sha256", "0" * 64),
        ):
            with self.subTest(forged_physical_field=field_name):
                forged_physical = dataclasses.replace(record.physical_work)
                object.__setattr__(forged_physical, field_name, forged_value)
                forged_record = dataclasses.replace(record)
                object.__setattr__(forged_record, "physical_work", forged_physical)
                with self.assertRaises(ValueError):
                    forged_record.deterministic_record()

        for field_name, forged_value in (
            ("kernel_version", "forged-kernel-version"),
            ("schedule_version", "forged-schedule-version"),
            ("schedule_sha256", "0" * 64),
            ("content_sha256", "0" * 64),
        ):
            with self.subTest(forged_record_field=field_name):
                forged_record = dataclasses.replace(record)
                object.__setattr__(forged_record, field_name, forged_value)
                with self.assertRaises(ValueError):
                    forged_record.deterministic_record()
        with mock.patch.object(scalar_fused_module, "SCHEDULE_VERSION", "forged-schedule-version"):
            with self.assertRaises(ValueError):
                record.deterministic_record()

        workspace.correction.fill_(wp.vec3d(np.nan, np.nan, np.nan))
        with self.assertRaisesRegex(FloatingPointError, "must remain finite"):
            workspace.record()
        original_schedule = workspace._schedule_sha256
        workspace._schedule_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "schedule binding"):
            workspace.launch()
        workspace._schedule_sha256 = original_schedule
        original_core_schedule = workspace._core_schedule_sha256
        workspace._core_schedule_sha256 = "1" * 64
        with self.assertRaisesRegex(RuntimeError, "schedule binding"):
            workspace.launch_core()
        workspace._core_schedule_sha256 = original_core_schedule

        original_residuals = workspace.level_residual
        workspace.level_residual = (
            wp.empty(original_residuals[0].shape, dtype=wp.float64, device="cpu"),
            *original_residuals[1:],
        )
        with self.assertRaisesRegex(RuntimeError, "persistent array pointers"):
            workspace.launch()
        workspace.level_residual = original_residuals

        source = WarpStaticMultigridHierarchy.from_hierarchy(self.hierarchy, device="cpu")
        static_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_device_hierarchy(source)
        static_workspace = static_hierarchy.create_workspace()
        static_workspace.set_rhs(rhs)

        original_steps = source.pre_smooth_steps
        source.pre_smooth_steps = original_steps + 1
        with self.assertRaisesRegex(RuntimeError, "hierarchy identity changed"):
            static_workspace.launch()
        source.pre_smooth_steps = original_steps

        level = source.levels[0]
        original_values = level.matrix_values
        replacement = wp.empty(original_values.shape, dtype=wp.float64, device="cpu")
        object.__setattr__(level, "matrix_values", replacement)
        with self.assertRaisesRegex(RuntimeError, "hierarchy identity changed"):
            static_workspace.launch()
        object.__setattr__(level, "matrix_values", original_values)

        source.levels[0].matrix_values.fill_(0.0)
        static_workspace.launch()
        with self.assertRaisesRegex(RuntimeError, "static device hierarchy content changed"):
            static_workspace.record()

        bad_root_source = WarpStaticMultigridHierarchy.from_hierarchy(self.hierarchy, device="cpu")
        bad_root = bad_root_source.levels[0]
        original_block_size = bad_root.block_size
        object.__setattr__(bad_root, "block_size", 6)
        with self.assertRaisesRegex(ValueError, "one 3-vector block per free vertex"):
            WarpScalarFusedStaticMultigridHierarchy.from_device_hierarchy(bad_root_source)
        object.__setattr__(bad_root, "block_size", original_block_size)

    def test_preconditioner_boundary_and_coarsest_only_schedule(self):
        preconditioner = WarpScalarFusedStaticMultigridPreconditioner(self.device_hierarchy)
        workspace = preconditioner.create_application_workspace()
        rhs = _rhs_set(self.device_hierarchy.n_free_dofs)[1]
        workspace.set_rhs(rhs)
        preconditioner.launch_apply(workspace.rhs, workspace.correction, workspace)
        application = preconditioner.record_application(0, workspace, capture_replay=False)
        self.assertEqual(preconditioner.application_kernel_launches, 22)
        self.assertEqual(application.scheduled_kernel_launches, 22)
        self.assertEqual(application.static_preconditioner_sha256, self.hierarchy.content_sha256)
        self.assertTrue(application.output_finite)
        self.assertFalse(application.capture_replay)

        coarse = _coarsest_only_hierarchy()
        coarse_device = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(coarse, device="cpu")
        coarse_workspace = coarse_device.create_workspace()
        coarse_rhs = np.linspace(-0.4, 0.9, coarse.levels[0].matrix.scalar_size)
        coarse_workspace.set_rhs(coarse_rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            coarse_workspace.launch()
        self.assertEqual(launch.call_count, 3)
        self.assertEqual(
            sum(
                call.args[0] is scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi
                for call in launch.call_args_list
            ),
            0,
        )
        self.assertEqual(
            sum(call.args[0] is scalar_fused_module._copy_vec3_to_scalar for call in launch.call_args_list),
            1,
        )
        self.assertEqual(coarse_device.scheduled_kernel_launches, 3)
        self.assertEqual(coarse_device.core_kernel_launches, 2)
        coarse_record = coarse_workspace.record()
        self.assertEqual(coarse_record.physical_work.root_ingress_zero_start_fusions, 0)
        np.testing.assert_allclose(
            coarse_record.correction,
            apply_v_cycle(coarse, coarse_rhs).correction,
            rtol=3.0e-14,
            atol=3.0e-14,
        )
        coarse_workspace.set_rhs(coarse_rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as core_launch:
            coarse_workspace.launch_core()
        self.assertEqual(core_launch.call_count, 2)
        core_record = coarse_workspace.record_core_application(token=scalar_fused_module._CORE_RECORD_TOKEN)
        self.assertEqual(core_record.scheduled_kernel_launches, 2)
        self.assertEqual(core_record.physical_work.publication_kernel_launches, 0)
        np.testing.assert_array_equal(core_record.correction.view(np.uint64), coarse_record.correction.view(np.uint64))

    def test_257_scalar_row_boundary_and_translation_transfer(self):
        hierarchy = _translation_hierarchy(257)
        self.assertEqual(
            [(level.matrix.block_row_count, level.matrix.block_size) for level in hierarchy.levels],
            [(257, 3), (16, 3)],
        )
        device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cpu")
        workspace = device_hierarchy.create_workspace()
        rhs = np.zeros(hierarchy.levels[0].matrix.scalar_size, dtype=np.float64)
        rhs[-3:] = (0.25, -0.5, 1.0)
        workspace.set_rhs(rhs)
        workspace.launch()
        actual = workspace.record()
        self.assertEqual(actual.scheduled_kernel_launches, 8)
        np.testing.assert_allclose(
            actual.correction,
            apply_v_cycle(hierarchy, rhs).correction,
            rtol=4.0e-14,
            atol=4.0e-14,
        )
        self.assertGreater(float(np.linalg.norm(actual.correction[-3:])), 0.0)


class TestWarpScalarFusedDefaultStretchCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hierarchy = _default_stretch_hierarchy()
        cls.device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cpu")

    def test_real_default_hierarchy_oracle_and_20_launch_schedule(self):
        shapes = [
            (level.matrix.block_row_count, level.matrix.block_size, level.matrix.stored_block_count)
            for level in self.hierarchy.levels
        ]
        self.assertEqual(shapes, [(144, 3, 1378), (32, 6, 270), (8, 6, 38), (2, 6, 4)])
        self.assertEqual(
            self.hierarchy.content_sha256, "2e08bcce552d135e3ec8010c6100ee6b18e6157e0c4e7013c2894280a1e9d493"
        )
        self.assertEqual(self.device_hierarchy.scheduled_kernel_launches, 20)
        self.assertEqual(self.device_hierarchy.core_kernel_launches, 19)
        self.assertEqual(self.device_hierarchy.source_hierarchy.scheduled_kernel_launches, 36)

        rhs = np.random.default_rng(230818).normal(size=self.hierarchy.levels[0].matrix.scalar_size)
        expected = apply_v_cycle(self.hierarchy, rhs)
        workspace = self.device_hierarchy.create_workspace()
        workspace.set_rhs(rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            workspace.launch()
        self.assertEqual(launch.call_count, 20)
        actual = workspace.record()
        np.testing.assert_allclose(actual.correction, expected.correction, rtol=7.0e-14, atol=7.0e-14)
        _assert_canonical_work_matches(self, actual.work, expected.work)
        self.assertEqual(actual.work.matrix_block_products, 5058)
        self.assertEqual(actual.physical_work.matrix_block_products_executed, 3372)
        self.assertEqual(actual.physical_work.matrix_block_products_elided_zero_start, 1686)
        self.assertEqual(actual.physical_work.zero_start_block_solves, 184)
        self.assertEqual(actual.physical_work.root_ingress_zero_start_fusions, 1)
        self.assertEqual(actual.physical_work.out_of_place_jacobi_block_solves, 184)
        self.assertEqual(actual.physical_work.matrix_kernel_launches, 6)
        self.assertEqual(actual.physical_work.jacobi_kernel_launches, 6)
        linear_prefix_launches = 4 + 2 * actual.physical_work.core_kernel_launches
        predicted_captured_launches = 2 + 4 * ((linear_prefix_launches + 1) + 3)
        self.assertEqual(predicted_captured_launches, 186)
        self.assertFalse(actual.performance_evidence)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestWarpScalarFusedStaticMultigridCudaCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.hierarchy = _mixed_hierarchy(smooth_steps=2)
        cls.device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(cls.hierarchy, device="cuda:0")

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

    def test_cuda_fused_root_ingress_is_bitwise_legacy_for_signed_zero(self):
        root = self.device_hierarchy.levels[0]
        host_rhs = np.zeros((self.device_hierarchy.n_free, 3), dtype=np.float64)
        host_rhs.reshape(-1)[::2] = -0.0
        host_rhs[1::5] = (0.25, -0.5, 1.0)
        external_rhs = wp.array(host_rhs, dtype=wp.vec3d, device=self.device_hierarchy.device)
        legacy_rhs = wp.empty(root.scalar_size, dtype=wp.float64, device=self.device_hierarchy.device)
        legacy_correction = wp.empty(root.scalar_size, dtype=wp.float64, device=self.device_hierarchy.device)
        fused_rhs = wp.empty(root.scalar_size, dtype=wp.float64, device=self.device_hierarchy.device)
        fused_correction = wp.empty(root.scalar_size, dtype=wp.float64, device=self.device_hierarchy.device)
        wp.launch(
            scalar_fused_module._copy_vec3_to_scalar,
            dim=self.device_hierarchy.n_free,
            inputs=[external_rhs, legacy_rhs],
            device=self.device_hierarchy.device,
        )
        wp.launch(
            scalar_fused_module._zero_start_scalar_jacobi,
            dim=root.scalar_size,
            inputs=[legacy_rhs, root.inverse_diagonal, root.block_size, root.omega, legacy_correction],
            device=self.device_hierarchy.device,
        )
        wp.launch(
            scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi,
            dim=root.scalar_size,
            inputs=[external_rhs, root.inverse_diagonal, root.omega, fused_rhs, fused_correction],
            device=self.device_hierarchy.device,
        )
        np.testing.assert_array_equal(fused_rhs.numpy().view(np.uint64), legacy_rhs.numpy().view(np.uint64))
        np.testing.assert_array_equal(
            fused_correction.numpy().view(np.uint64),
            legacy_correction.numpy().view(np.uint64),
        )

    def test_capture_replay_changed_rhs_overwrites_poison_and_repeats_hash(self):
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
        np.testing.assert_allclose(
            changed.correction,
            apply_v_cycle(self.hierarchy, rhs_values[1]).correction,
            rtol=3.0e-13,
            atol=3.0e-13,
        )
        self.assertTrue(changed.capture_replay)
        self.assertFalse(changed.performance_evidence)

        _poison_workspace(workspace)
        workspace.set_rhs(rhs_values[2])
        wp.capture_launch(capture.graph)
        recovered = workspace.record(capture_replay=True)
        np.testing.assert_allclose(
            recovered.correction,
            apply_v_cycle(self.hierarchy, rhs_values[2]).correction,
            rtol=3.0e-13,
            atol=3.0e-13,
        )
        wp.capture_launch(capture.graph)
        repeated = workspace.record(capture_replay=True)
        self.assertEqual(repeated.work.result_sha256, recovered.work.result_sha256)
        self.assertEqual(repeated.content_sha256, recovered.content_sha256)
        self.assertNotEqual(recovered.work.rhs_sha256, changed.work.rhs_sha256)
        self.assertEqual(_pointer_tuple(self.device_hierarchy, workspace), pointers)

    def test_real_default_stretch_cuda_oracle_uses_20_launches(self):
        hierarchy = _default_stretch_hierarchy()
        device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cuda:0")
        workspace = device_hierarchy.create_workspace()
        rhs = np.random.default_rng(99118).normal(size=hierarchy.levels[0].matrix.scalar_size)
        workspace.set_rhs(rhs)
        workspace.launch()
        actual = workspace.record()
        self.assertEqual(actual.scheduled_kernel_launches, 20)
        np.testing.assert_allclose(
            actual.correction,
            apply_v_cycle(hierarchy, rhs).correction,
            rtol=4.0e-13,
            atol=4.0e-13,
        )
        self.assertFalse(actual.performance_evidence)


if __name__ == "__main__":
    unittest.main()
