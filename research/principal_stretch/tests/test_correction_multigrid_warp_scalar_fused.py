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
    PHYSICAL_EXECUTION_AUTHENTICATION,
    PUBLICATION_VERSION,
    ROOT_INGRESS_COARSE_COPY_ROUTE,
    ROOT_INGRESS_EXTERNAL_SHARED_ROUTE,
    ROOT_INGRESS_INTERNAL_ROUTE,
    SCHEDULE_VERSION,
    STANDALONE_PUBLICATION_ROUTE,
    TERMINAL_FUSION_COARSEST_ONLY_ROUTE,
    TERMINAL_FUSION_CPU_FALLBACK_ROUTE,
    TERMINAL_FUSION_CUDA_ROUTE,
    TERMINAL_FUSION_OVERSIZE_FALLBACK_ROUTE,
    TERMINAL_FUSION_VERSION,
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


def _single_vertex_coarsest_only_hierarchy() -> StaticMultigridHierarchy:
    source = _translation_hierarchy(3)
    level = source.levels[-1]
    return dataclasses.replace(
        source,
        levels=(level,),
        free_vertices=np.array([0], dtype=np.int64),
        rest_positions=np.zeros((1, 3), dtype=np.float64),
        free_masses=np.ones(1, dtype=np.float64),
        content_sha256="d" * 64,
    )


def _rehash_physical_work(physical_work) -> None:
    object.__setattr__(
        physical_work,
        "content_sha256",
        scalar_fused_module._hash_parts(
            "warp-scalar-fused-v-cycle-physical-work-v9",
            tuple(
                (field.name, getattr(physical_work, field.name))
                for field in dataclasses.fields(physical_work)
                if field.name != "content_sha256"
            ),
        ),
    )


def _rehash_v_cycle_record(record) -> None:
    physical_work = record.physical_work
    object.__setattr__(
        record,
        "content_sha256",
        scalar_fused_module._hash_parts(
            "warp-scalar-fused-v-cycle-result-v9",
            (
                ("contract_id", record.contract_id),
                ("kernel_version", record.kernel_version),
                ("schedule_version", record.schedule_version),
                ("device_snapshot_sha256", record.device_snapshot_sha256),
                ("static_device_content_sha256", record.static_device_content_sha256),
                ("schedule_sha256", record.schedule_sha256),
                ("standalone_schedule_sha256", record.standalone_schedule_sha256),
                ("core_schedule_sha256", record.core_schedule_sha256),
                ("seeded_core_schedule_sha256", record.seeded_core_schedule_sha256),
                ("standalone_device_snapshot_sha256", record.standalone_device_snapshot_sha256),
                ("core_device_snapshot_sha256", record.core_device_snapshot_sha256),
                ("seeded_core_device_snapshot_sha256", record.seeded_core_device_snapshot_sha256),
                ("work_sha256", record.work.content_sha256),
                ("physical_work_sha256", physical_work.content_sha256),
                ("scheduled_kernel_launches", record.scheduled_kernel_launches),
                ("capture_replay", record.capture_replay),
                ("research_only", record.research_only),
                ("physical_execution_authentication", record.physical_execution_authentication),
                ("solver_issued_authentication", record.solver_issued_authentication),
                ("performance_evidence", record.performance_evidence),
            ),
        ),
    )


def _unchecked_record_clone(record):
    forged_record = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(forged_record, field.name, getattr(record, field.name))
    return forged_record


def _coordinated_rehash(record, *, physical_updates, record_updates=None):
    forged_physical = dataclasses.replace(record.physical_work)
    for field_name, value in physical_updates.items():
        object.__setattr__(forged_physical, field_name, value)
    _rehash_physical_work(forged_physical)
    forged_record = _unchecked_record_clone(record)
    object.__setattr__(forged_record, "physical_work", forged_physical)
    for field_name, value in (record_updates or {}).items():
        object.__setattr__(forged_record, field_name, value)
    _rehash_v_cycle_record(forged_record)
    return forged_record


def _default_stretch_hierarchy(*, smooth_steps: int = 1) -> StaticMultigridHierarchy:
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
        pre_smooth_steps=smooth_steps,
        post_smooth_steps=smooth_steps,
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


def _workspace_bit_patterns(workspace) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
    """Materialize every retained workspace array as exact host bytes."""
    patterns = []
    for array in workspace._current_arrays():
        host = np.ascontiguousarray(array.numpy())
        patterns.append((host.dtype.str, host.shape, host.tobytes()))
    return tuple(patterns)


def _external_vec3_view(
    *,
    size: int,
    stride: int,
    device: str | wp.context.Device,
) -> tuple[wp.array[wp.vec3d], wp.array[wp.vec3d]]:
    """Allocate backing storage and one explicit-stride external vec3 view."""
    element_size = wp.types.type_size_in_bytes(wp.vec3d)
    backing_size = 1 if stride == 0 else 1 + (size - 1) * abs(stride) // element_size
    backing = wp.empty(backing_size, dtype=wp.vec3d, device=device)
    pointer = int(backing.ptr) + ((size - 1) * -stride if stride < 0 else 0)
    view = wp.array(
        ptr=pointer,
        dtype=wp.vec3d,
        shape=(size,),
        strides=(stride,),
        device=device,
        copy=False,
    )
    return backing, view


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
        self.assertEqual(KERNEL_VERSION, "mg-vbd-warp-static-v-cycle-scalar-fused-v5")
        self.assertEqual(CONTRACT_ID, "spectral-free-multiplicative-graph-vbd-warp-static-scalar-fused-v1")
        self.assertEqual(
            SCHEDULE_VERSION,
            "scalar-core-terminal-fused-seeded-root-publication-routes-v7",
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
        self.assertEqual(hierarchy.seeded_core_kernel_launches, 20)
        self.assertTrue(hierarchy.supports_seeded_root_zero_start)
        self.assertFalse(hierarchy.supports_terminal_fusion)
        self.assertEqual(hierarchy.terminal_fusion_route, TERMINAL_FUSION_CPU_FALLBACK_ROUTE)
        self.assertEqual(hierarchy.terminal_fusion_kernel_launches, 0)
        self.assertEqual(hierarchy.terminal_level_index, 1)
        self.assertEqual(hierarchy.terminal_block_dim, 0)
        self.assertEqual(hierarchy.terminal_collective_count, 0)
        self.assertEqual(hierarchy.terminal_owner_thread, -1)
        self.assertNotEqual(hierarchy.schedule_sha256, hierarchy.core_schedule_sha256)
        self.assertNotEqual(hierarchy.core_schedule_sha256, hierarchy.seeded_core_schedule_sha256)
        self.assertNotEqual(hierarchy.device_snapshot_sha256, hierarchy.core_device_snapshot_sha256)
        self.assertNotEqual(hierarchy.core_device_snapshot_sha256, hierarchy.seeded_core_device_snapshot_sha256)
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
            "_terminal_restrict_ordered_solve_prolong",
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
                    self.assertEqual(actual.physical_work.terminal_fusion_kernel_launches, 0)
                    self.assertEqual(actual.physical_work.terminal_level_index, len(hierarchy.levels) - 2)
                    self.assertEqual(actual.physical_work.terminal_block_dim, 0)
                    self.assertEqual(actual.physical_work.terminal_collective_count, 0)
                    self.assertEqual(actual.physical_work.terminal_owner_thread, -1)
                    self.assertEqual(actual.physical_work.terminal_fusion_version, TERMINAL_FUSION_VERSION)
                    self.assertEqual(actual.physical_work.terminal_fusion_route, TERMINAL_FUSION_CPU_FALLBACK_ROUTE)
                    self.assertEqual(actual.physical_work.root_ingress_zero_start_fusions, 1)
                    self.assertEqual(actual.physical_work.root_ingress_route, ROOT_INGRESS_INTERNAL_ROUTE)
                    self.assertEqual(actual.physical_work.root_ingress_kernel_launches, 1)
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
                    self.assertEqual(
                        actual.physical_execution_authentication,
                        PHYSICAL_EXECUTION_AUTHENTICATION,
                    )
                    self.assertEqual(
                        actual.physical_work.physical_execution_authentication,
                        PHYSICAL_EXECUTION_AUTHENTICATION,
                    )
                    self.assertFalse(actual.solver_issued_authentication)
                    self.assertFalse(actual.physical_work.solver_issued_authentication)
                    self.assertFalse(actual.performance_evidence)
                    self.assertFalse(actual.physical_work.performance_evidence)
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
                    core_record = workspace.record_core_application()
                    np.testing.assert_array_equal(
                        core_record.correction.view(np.uint64),
                        actual.correction.view(np.uint64),
                    )
                    self.assertEqual(core_record.scheduled_kernel_launches, expected_launches - 1)
                    self.assertEqual(core_record.physical_work.core_kernel_launches, expected_launches - 1)
                    self.assertEqual(core_record.physical_work.publication_kernel_launches, 0)
                    self.assertEqual(core_record.physical_work.publication_version, PUBLICATION_VERSION)
                    self.assertEqual(core_record.physical_work.publication_route, EXTERNAL_SHARED_PUBLICATION_ROUTE)
                    self.assertEqual(core_record.physical_work.root_ingress_route, ROOT_INGRESS_INTERNAL_ROUTE)
                    self.assertEqual(core_record.physical_work.root_ingress_kernel_launches, 1)
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

    def test_seeded_root_tail_is_bitwise_old_core_for_p1_p2_and_records_physical_boundary(self):
        for steps in (1, 2):
            with self.subTest(steps=steps):
                hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
                    _mixed_hierarchy(smooth_steps=steps),
                    device="cpu",
                )
                rhs = np.random.default_rng(181001 + steps).normal(size=(hierarchy.n_free, 3))
                old = hierarchy.create_workspace()
                seeded = hierarchy.create_workspace()
                old.rhs.assign(rhs)
                seeded.rhs.assign(rhs)
                _poison_workspace(old)
                _poison_workspace(seeded)
                old.rhs.assign(rhs)
                seeded.rhs.assign(rhs)

                old.launch_core()
                seed = hierarchy.root_zero_start_seed_parameters(seeded.rhs, seeded)
                self.assertIsNotNone(seed)
                assert seed is not None
                inverse_diagonal, omega, scalar_rhs, root_primary = seed
                self.assertIs(scalar_rhs, seeded.level_rhs[0])
                self.assertIs(root_primary, seeded.level_correction[0])
                wp.launch(
                    scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi,
                    dim=hierarchy.levels[0].scalar_size,
                    inputs=[seeded.rhs, inverse_diagonal, omega, scalar_rhs, root_primary],
                    device=hierarchy.device,
                )
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    seeded.launch_seeded_core()
                self.assertEqual(launch.call_count, hierarchy.seeded_core_kernel_launches)
                self.assertFalse(
                    any(
                        call.args[0] is scalar_fused_module._fused_root_ingress_zero_start_scalar_jacobi
                        for call in launch.call_args_list
                    )
                )

                for group_name in (
                    "level_rhs",
                    "level_correction",
                    "level_correction_alt",
                    "level_residual",
                ):
                    for level_index, (old_array, seeded_array) in enumerate(
                        zip(getattr(old, group_name), getattr(seeded, group_name), strict=True)
                    ):
                        with self.subTest(steps=steps, group=group_name, level=level_index):
                            np.testing.assert_array_equal(
                                seeded_array.numpy().view(np.uint64),
                                old_array.numpy().view(np.uint64),
                            )
                np.testing.assert_array_equal(
                    seeded.coarse_intermediate.numpy().view(np.uint64),
                    old.coarse_intermediate.numpy().view(np.uint64),
                )
                record = seeded.record_seeded_core_application()
                self.assertEqual(record.scheduled_kernel_launches, hierarchy.seeded_core_kernel_launches)
                self.assertEqual(record.schedule_sha256, hierarchy.seeded_core_schedule_sha256)
                self.assertEqual(record.device_snapshot_sha256, hierarchy.seeded_core_device_snapshot_sha256)
                self.assertEqual(record.physical_work.root_ingress_zero_start_fusions, 1)
                self.assertEqual(record.physical_work.root_ingress_route, ROOT_INGRESS_EXTERNAL_SHARED_ROUTE)
                self.assertEqual(record.physical_work.root_ingress_kernel_launches, 0)
                self.assertEqual(record.physical_work.core_kernel_launches, hierarchy.seeded_core_kernel_launches)

    def test_route_schema_rejects_partial_and_allows_coherent_rewrite(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cpu",
        )
        workspace = hierarchy.create_workspace()
        rhs = _rhs_set(hierarchy.n_free_dofs)[0]

        workspace.set_rhs(rhs)
        workspace.launch()
        standalone_record = workspace.record()
        forged_standalone = _coordinated_rehash(
            standalone_record,
            physical_updates={
                "root_ingress_route": ROOT_INGRESS_EXTERNAL_SHARED_ROUTE,
                "root_ingress_kernel_launches": 0,
            },
        )
        self.assertEqual(forged_standalone.schedule_sha256, hierarchy.schedule_sha256)
        self.assertEqual(
            forged_standalone.physical_work.core_kernel_launches,
            hierarchy.core_kernel_launches,
        )
        with self.assertRaisesRegex(ValueError, "root ingress route"):
            forged_standalone.deterministic_record()

        workspace.set_rhs(rhs)
        workspace.launch_core()
        core_record = workspace.record_core_application()
        forged_core_route = _coordinated_rehash(
            core_record,
            physical_updates={
                "root_ingress_route": ROOT_INGRESS_EXTERNAL_SHARED_ROUTE,
                "root_ingress_kernel_launches": 0,
            },
        )
        self.assertEqual(forged_core_route.schedule_sha256, hierarchy.core_schedule_sha256)
        self.assertEqual(forged_core_route.physical_work.core_kernel_launches, hierarchy.core_kernel_launches)
        with self.assertRaisesRegex(ValueError, "root ingress route"):
            forged_core_route.deterministic_record()

        forged_core_schedule = _coordinated_rehash(
            core_record,
            physical_updates={"schedule_sha256": hierarchy.schedule_sha256},
            record_updates={
                "schedule_sha256": hierarchy.schedule_sha256,
                "device_snapshot_sha256": hierarchy.device_snapshot_sha256,
            },
        )
        self.assertEqual(forged_core_schedule.physical_work.core_kernel_launches, hierarchy.core_kernel_launches)
        with self.assertRaisesRegex(ValueError, "selected schedule"):
            forged_core_schedule.deterministic_record()

        coherent_seeded = _coordinated_rehash(
            core_record,
            physical_updates={
                "schedule_sha256": hierarchy.seeded_core_schedule_sha256,
                "root_ingress_route": ROOT_INGRESS_EXTERNAL_SHARED_ROUTE,
                "root_ingress_kernel_launches": 0,
                "core_kernel_launches": hierarchy.seeded_core_kernel_launches,
                "scheduled_kernel_launches": hierarchy.seeded_core_kernel_launches,
            },
            record_updates={
                "schedule_sha256": hierarchy.seeded_core_schedule_sha256,
                "device_snapshot_sha256": hierarchy.seeded_core_device_snapshot_sha256,
                "scheduled_kernel_launches": hierarchy.seeded_core_kernel_launches,
            },
        )
        coherent = coherent_seeded.deterministic_record()
        self.assertEqual(coherent["root_ingress_route"], ROOT_INGRESS_EXTERNAL_SHARED_ROUTE)
        self.assertEqual(coherent["scheduled_kernel_launches"], hierarchy.seeded_core_kernel_launches)
        self.assertEqual(
            coherent["physical_execution_authentication"],
            PHYSICAL_EXECUTION_AUTHENTICATION,
        )
        self.assertFalse(coherent["solver_issued_authentication"])
        self.assertFalse(coherent["performance_evidence"])

        coherent_terminal = _coordinated_rehash(
            standalone_record,
            physical_updates={
                "terminal_fusion_kernel_launches": 1,
                "terminal_block_dim": 64,
                "terminal_collective_count": 2,
                "terminal_owner_thread": 0,
                "terminal_fusion_route": TERMINAL_FUSION_CUDA_ROUTE,
                "core_kernel_launches": hierarchy.core_kernel_launches - 2,
                "scheduled_kernel_launches": hierarchy.scheduled_kernel_launches - 2,
            },
            record_updates={"scheduled_kernel_launches": hierarchy.scheduled_kernel_launches - 2},
        )
        coherent_terminal_record = coherent_terminal.deterministic_record()
        self.assertEqual(coherent_terminal_record["terminal_fusion_route"], TERMINAL_FUSION_CUDA_ROUTE)
        self.assertFalse(coherent_terminal_record["solver_issued_authentication"])
        self.assertFalse(coherent_terminal_record["performance_evidence"])

    def test_schema_policy_reconstruction_and_tamper_fail_closed(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cpu",
        )
        workspace = hierarchy.create_workspace()
        rhs = _rhs_set(hierarchy.n_free_dofs)[0]
        for private_authority_name in (
            "_bind_workspace_evidence",
            "_invalidate_workspace_evidence",
            "_register_workspace_execution",
            "_current_workspace_execution",
            "_prepare_workspace_record",
            "_cancel_workspace_record",
            "_consume_workspace_record",
            "_validate_issued_record",
            "_create_execution_evidence_authority",
            "_with_execution_authority",
            "_CORE_RECORD_TOKEN",
        ):
            with self.subTest(private_authority_name=private_authority_name):
                self.assertNotIn(private_authority_name, vars(scalar_fused_module))
        self.assertNotIn(
            "_issuance_capability",
            {field.name for field in dataclasses.fields(scalar_fused_module.WarpScalarFusedVCycleRecord)},
        )
        self.assertFalse(hasattr(scalar_fused_module.WarpScalarFusedVCycleRecord, "_require_issued"))

        workspace.set_rhs(rhs)
        workspace.launch_core()
        record = workspace.record_core_application()
        serialized = record.deterministic_record()
        self.assertEqual(
            serialized["physical_execution_authentication"],
            PHYSICAL_EXECUTION_AUTHENTICATION,
        )
        self.assertFalse(serialized["solver_issued_authentication"])
        self.assertFalse(serialized["performance_evidence"])
        self.assertNotIn("execution_route", serialized)
        self.assertNotIn("execution_generation", serialized)
        self.assertNotIn("execution_context_sha256", serialized)

        replacement = dataclasses.replace(record)
        self.assertIsNot(replacement, record)
        self.assertEqual(replacement.deterministic_record(), serialized)
        constructor_fields = {field.name: getattr(record, field.name) for field in dataclasses.fields(record)}
        reconstructed = type(record)(**constructor_fields)
        self.assertEqual(reconstructed.deterministic_record(), serialized)

        capture_rewrite = _unchecked_record_clone(record)
        object.__setattr__(capture_rewrite, "capture_replay", True)
        _rehash_v_cycle_record(capture_rewrite)
        capture_serialized = capture_rewrite.deterministic_record()
        self.assertTrue(capture_serialized["capture_replay"])
        self.assertFalse(capture_serialized["solver_issued_authentication"])

        for field_name, forged_value in (
            ("physical_execution_authentication", "solver-launch-authenticated-v1"),
            ("solver_issued_authentication", True),
            ("performance_evidence", True),
        ):
            with self.subTest(physical_policy=field_name):
                forged = _coordinated_rehash(
                    record,
                    physical_updates={field_name: forged_value},
                )
                with self.assertRaisesRegex(ValueError, "schema-validated and unauthenticated"):
                    forged.deterministic_record()
            with self.subTest(record_policy=field_name):
                forged = _unchecked_record_clone(record)
                object.__setattr__(forged, field_name, forged_value)
                _rehash_v_cycle_record(forged)
                with self.assertRaisesRegex(ValueError, "schema-validated and unauthenticated"):
                    forged.deterministic_record()

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
                    with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                        with self.assertRaisesRegex(ValueError, "rhs and root"):
                            hierarchy.launch_apply_core_seeded_root(aliased_rhs, workspace)
                    self.assertEqual(launch.call_count, 0)

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

        static_storage = hierarchy.levels[0].matrix_values
        for role in ("rhs", "output"):
            with self.subTest(external_static_alias=role):
                workspace = hierarchy.create_workspace()
                aliased = wp.array(
                    ptr=int(static_storage.ptr),
                    dtype=wp.vec3d,
                    shape=(hierarchy.n_free,),
                    device=hierarchy.device,
                    copy=False,
                )
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaisesRegex(ValueError, f"{role} and static"):
                        if role == "rhs":
                            hierarchy.launch_apply_core(aliased, workspace)
                        else:
                            hierarchy.launch_apply(workspace.rhs, aliased, workspace)
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

    def test_external_vec3_layouts_preserve_gapped_reversed_zero_stride_and_identity_on_cpu(self):
        source = _mixed_hierarchy(smooth_steps=1)
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(source, device="cpu")
        host_rhs = np.random.default_rng(181030).normal(size=(hierarchy.n_free, 3))
        expected = apply_v_cycle(source, host_rhs.reshape(-1)).correction.reshape(-1, 3)

        for stride in (48, -48):
            with self.subTest(stride=stride, role="separate-rhs-output"):
                _rhs_backing, rhs = _external_vec3_view(
                    size=hierarchy.n_free,
                    stride=stride,
                    device=hierarchy.device,
                )
                _output_backing, output = _external_vec3_view(
                    size=hierarchy.n_free,
                    stride=stride,
                    device=hierarchy.device,
                )
                rhs.assign(host_rhs)
                workspace = hierarchy.create_workspace()
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    hierarchy.launch_apply(rhs, output, workspace)
                self.assertEqual(launch.call_count, hierarchy.scheduled_kernel_launches)
                np.testing.assert_allclose(output.numpy(), expected, rtol=4.0e-14, atol=4.0e-14)

            with self.subTest(stride=stride, role="rhs-is-output"):
                _shared_backing, shared = _external_vec3_view(
                    size=hierarchy.n_free,
                    stride=stride,
                    device=hierarchy.device,
                )
                shared.assign(host_rhs)
                workspace = hierarchy.create_workspace()
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    hierarchy.launch_apply(shared, shared, workspace)
                self.assertEqual(launch.call_count, hierarchy.scheduled_kernel_launches)
                np.testing.assert_allclose(shared.numpy(), expected, rtol=4.0e-14, atol=4.0e-14)

        repeated_value = np.array([[0.25, -0.0, 1.5]], dtype=np.float64)
        zero_backing, zero_rhs = _external_vec3_view(
            size=hierarchy.n_free,
            stride=0,
            device=hierarchy.device,
        )
        zero_backing.assign(repeated_value)
        output = wp.empty(hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        workspace = hierarchy.create_workspace()
        hierarchy.launch_apply(zero_rhs, output, workspace)
        repeated_rhs = np.repeat(repeated_value, hierarchy.n_free, axis=0)
        repeated_expected = apply_v_cycle(source, repeated_rhs.reshape(-1)).correction.reshape(-1, 3)
        np.testing.assert_allclose(output.numpy(), repeated_expected, rtol=4.0e-14, atol=4.0e-14)

    def test_external_vec3_invalid_alignment_and_writable_overlap_are_zero_launch(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cpu",
        )
        rhs = wp.empty(hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        output_backing = wp.empty(2 * hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        invalid_outputs = (
            (
                "zero-stride",
                wp.array(
                    ptr=int(output_backing.ptr),
                    dtype=wp.vec3d,
                    shape=(hierarchy.n_free,),
                    strides=(0,),
                    device=hierarchy.device,
                    copy=False,
                ),
                "must not overlap",
            ),
            (
                "overlapping-stride",
                wp.array(
                    ptr=int(output_backing.ptr),
                    dtype=wp.vec3d,
                    shape=(hierarchy.n_free,),
                    strides=(16,),
                    device=hierarchy.device,
                    copy=False,
                ),
                "must not overlap",
            ),
            (
                "misaligned-stride",
                wp.array(
                    ptr=int(output_backing.ptr),
                    dtype=wp.vec3d,
                    shape=(hierarchy.n_free,),
                    strides=(28,),
                    device=hierarchy.device,
                    copy=False,
                ),
                "naturally aligned",
            ),
            (
                "misaligned-pointer",
                wp.array(
                    ptr=int(output_backing.ptr) + 4,
                    dtype=wp.vec3d,
                    shape=(hierarchy.n_free,),
                    strides=(24,),
                    device=hierarchy.device,
                    copy=False,
                ),
                "naturally aligned",
            ),
        )
        for label, output, message in invalid_outputs:
            with self.subTest(output=label):
                workspace = hierarchy.create_workspace()
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    with self.assertRaisesRegex(ValueError, message):
                        hierarchy.launch_apply(rhs, output, workspace)
                self.assertEqual(launch.call_count, 0)

        misaligned_rhs = wp.array(
            ptr=int(output_backing.ptr) + 4,
            dtype=wp.vec3d,
            shape=(hierarchy.n_free,),
            strides=(24,),
            device=hierarchy.device,
            copy=False,
        )
        workspace = hierarchy.create_workspace()
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "naturally aligned"):
                hierarchy.launch_apply_core(misaligned_rhs, workspace)
        self.assertEqual(launch.call_count, 0)

    def test_single_element_zero_stride_output_and_identity_are_valid_on_cpu(self):
        source = _single_vertex_coarsest_only_hierarchy()
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(source, device="cpu")
        host_rhs = np.array([[0.25, -0.5, 1.0]], dtype=np.float64)
        expected = apply_v_cycle(source, host_rhs.reshape(-1)).correction.reshape(1, 3)
        for shared in (False, True):
            with self.subTest(rhs_is_output=shared):
                backing, output = _external_vec3_view(size=1, stride=0, device=hierarchy.device)
                if shared:
                    backing.assign(host_rhs)
                    rhs = output
                else:
                    rhs = wp.array(host_rhs, dtype=wp.vec3d, device=hierarchy.device)
                workspace = hierarchy.create_workspace()
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    hierarchy.launch_apply(rhs, output, workspace)
                self.assertEqual(launch.call_count, hierarchy.scheduled_kernel_launches)
                np.testing.assert_allclose(output.numpy(), expected, rtol=4.0e-14, atol=4.0e-14)

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
            ("root_ingress_route", ROOT_INGRESS_EXTERNAL_SHARED_ROUTE),
            ("root_ingress_kernel_launches", 0),
            ("terminal_fusion_kernel_launches", 1),
            ("terminal_level_index", 0),
            ("terminal_block_dim", 64),
            ("terminal_collective_count", 2),
            ("terminal_owner_thread", 0),
            ("terminal_fusion_version", TERMINAL_FUSION_VERSION + "-forged"),
            ("terminal_fusion_route", TERMINAL_FUSION_CUDA_ROUTE),
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
                forged_record = _unchecked_record_clone(record)
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
                forged_record = _unchecked_record_clone(record)
                object.__setattr__(forged_record, field_name, forged_value)
                with self.assertRaises(ValueError):
                    forged_record.deterministic_record()
        with mock.patch.object(scalar_fused_module, "SCHEDULE_VERSION", "forged-schedule-version"):
            with self.assertRaisesRegex(ValueError, "schedule version"):
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
        original_seeded_schedule = workspace._seeded_core_schedule_sha256
        workspace._seeded_core_schedule_sha256 = "2" * 64
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(RuntimeError, "schedule binding"):
                workspace.launch_seeded_core()
        self.assertEqual(launch.call_count, 0)
        workspace._seeded_core_schedule_sha256 = original_seeded_schedule

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
        self.assertEqual(coarse_device.seeded_core_kernel_launches, 2)
        self.assertFalse(coarse_device.supports_seeded_root_zero_start)
        self.assertFalse(coarse_device.supports_terminal_fusion)
        self.assertEqual(coarse_device.terminal_fusion_route, TERMINAL_FUSION_COARSEST_ONLY_ROUTE)
        self.assertEqual(coarse_device.terminal_fusion_kernel_launches, 0)
        self.assertEqual(coarse_device.terminal_level_index, -1)
        self.assertEqual(coarse_device.terminal_block_dim, 0)
        self.assertEqual(coarse_device.terminal_collective_count, 0)
        self.assertEqual(coarse_device.terminal_owner_thread, -1)
        self.assertEqual(coarse_device.seeded_core_schedule_sha256, coarse_device.core_schedule_sha256)
        coarse_record = coarse_workspace.record()
        self.assertEqual(coarse_record.physical_work.root_ingress_zero_start_fusions, 0)
        self.assertEqual(coarse_record.physical_work.terminal_fusion_route, TERMINAL_FUSION_COARSEST_ONLY_ROUTE)
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
        core_record = coarse_workspace.record_core_application()
        self.assertEqual(core_record.scheduled_kernel_launches, 2)
        self.assertEqual(core_record.physical_work.publication_kernel_launches, 0)
        self.assertEqual(core_record.physical_work.root_ingress_route, ROOT_INGRESS_COARSE_COPY_ROUTE)
        self.assertEqual(core_record.physical_work.root_ingress_kernel_launches, 1)
        np.testing.assert_array_equal(core_record.correction.view(np.uint64), coarse_record.correction.view(np.uint64))

        coarse_workspace.set_rhs(coarse_rhs)
        self.assertIsNone(coarse_device.root_zero_start_seed_parameters(coarse_workspace.rhs, coarse_workspace))
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as seeded_launch:
            coarse_workspace.launch_seeded_core()
        self.assertEqual(seeded_launch.call_count, 2)
        seeded_record = coarse_workspace.record_seeded_core_application()
        self.assertEqual(seeded_record.schedule_sha256, coarse_device.core_schedule_sha256)
        self.assertEqual(seeded_record.physical_work.root_ingress_route, ROOT_INGRESS_COARSE_COPY_ROUTE)
        self.assertEqual(seeded_record.physical_work.root_ingress_kernel_launches, 1)
        np.testing.assert_array_equal(
            seeded_record.correction.view(np.uint64), coarse_record.correction.view(np.uint64)
        )

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

    def test_cuda_terminal_route_counts_and_1024_thread_boundary(self):
        hierarchy = self.device_hierarchy
        self.assertTrue(hierarchy.supports_terminal_fusion)
        self.assertEqual(hierarchy.terminal_fusion_route, TERMINAL_FUSION_CUDA_ROUTE)
        self.assertEqual(hierarchy.terminal_fusion_kernel_launches, 1)
        self.assertEqual(hierarchy.terminal_level_index, len(hierarchy.levels) - 2)
        self.assertEqual(hierarchy.terminal_block_dim, 64)
        self.assertEqual(hierarchy.terminal_collective_count, 2)
        self.assertEqual(hierarchy.terminal_owner_thread, 0)
        self.assertEqual(hierarchy.scheduled_kernel_launches, 20)
        self.assertEqual(hierarchy.core_kernel_launches, 19)
        self.assertEqual(hierarchy.seeded_core_kernel_launches, 18)

        supported_cpu = _translation_hierarchy(341)
        supported = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(supported_cpu, device="cuda:0")
        self.assertEqual(supported.levels[-2].scalar_size, 1023)
        self.assertTrue(supported.supports_terminal_fusion)
        self.assertEqual(supported.terminal_fusion_route, TERMINAL_FUSION_CUDA_ROUTE)
        self.assertEqual(supported.terminal_block_dim, 1024)
        boundary_rhs = np.random.default_rng(181032).normal(size=supported_cpu.levels[0].matrix.scalar_size)
        fused = supported.create_workspace()
        legacy = supported.create_workspace()
        _poison_workspace(fused)
        _poison_workspace(legacy)
        fused.set_rhs(boundary_rhs)
        legacy.set_rhs(boundary_rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as fused_launch:
            fused.launch()
        with (
            mock.patch.object(
                WarpScalarFusedStaticMultigridHierarchy,
                "supports_terminal_fusion",
                new=property(lambda _self: False),
            ),
            mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as legacy_launch,
        ):
            legacy.launch()
        self.assertEqual(fused_launch.call_count, supported.scheduled_kernel_launches)
        self.assertEqual(legacy_launch.call_count, fused_launch.call_count + 2)
        self.assertEqual(_workspace_bit_patterns(fused), _workspace_bit_patterns(legacy))

        oversized_cpu = _translation_hierarchy(342)
        oversized = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(oversized_cpu, device="cuda:0")
        self.assertEqual(oversized.levels[-2].scalar_size, 1026)
        self.assertFalse(oversized.supports_terminal_fusion)
        self.assertEqual(oversized.terminal_fusion_route, TERMINAL_FUSION_OVERSIZE_FALLBACK_ROUTE)
        self.assertEqual(oversized.terminal_block_dim, 0)
        self.assertEqual(oversized.scheduled_kernel_launches, 8)
        rhs = np.random.default_rng(181018).normal(size=oversized_cpu.levels[0].matrix.scalar_size)
        workspace = oversized.create_workspace()
        workspace.set_rhs(rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            workspace.launch()
        self.assertEqual(launch.call_count, 8)
        np.testing.assert_allclose(
            workspace.record().correction,
            apply_v_cycle(oversized_cpu, rhs).correction,
            rtol=5.0e-13,
            atol=5.0e-13,
        )

    def test_cuda_external_vec3_gapped_reversed_rhs_output_and_identity(self):
        hierarchy = self.device_hierarchy
        host_rhs = np.random.default_rng(181031).normal(size=(hierarchy.n_free, 3))
        expected = apply_v_cycle(self.hierarchy, host_rhs.reshape(-1)).correction.reshape(-1, 3)
        for stride in (48, -48):
            with self.subTest(stride=stride, role="separate-rhs-output"):
                _rhs_backing, rhs = _external_vec3_view(
                    size=hierarchy.n_free,
                    stride=stride,
                    device=hierarchy.device,
                )
                _output_backing, output = _external_vec3_view(
                    size=hierarchy.n_free,
                    stride=stride,
                    device=hierarchy.device,
                )
                rhs.assign(host_rhs)
                workspace = hierarchy.create_workspace()
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    hierarchy.launch_apply(rhs, output, workspace)
                self.assertEqual(launch.call_count, hierarchy.scheduled_kernel_launches)
                np.testing.assert_allclose(output.numpy(), expected, rtol=3.0e-13, atol=3.0e-13)

            with self.subTest(stride=stride, role="rhs-is-output"):
                _shared_backing, shared = _external_vec3_view(
                    size=hierarchy.n_free,
                    stride=stride,
                    device=hierarchy.device,
                )
                shared.assign(host_rhs)
                workspace = hierarchy.create_workspace()
                hierarchy.launch_apply(shared, shared, workspace)
                np.testing.assert_allclose(shared.numpy(), expected, rtol=3.0e-13, atol=3.0e-13)

        backing = wp.empty(2 * hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        invalid_output = wp.array(
            ptr=int(backing.ptr),
            dtype=wp.vec3d,
            shape=(hierarchy.n_free,),
            strides=(16,),
            device=hierarchy.device,
            copy=False,
        )
        workspace = hierarchy.create_workspace()
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                hierarchy.launch_apply(workspace.rhs, invalid_output, workspace)
        self.assertEqual(launch.call_count, 0)

    def test_cuda_single_element_zero_stride_output_and_identity_are_valid(self):
        source = _single_vertex_coarsest_only_hierarchy()
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(source, device="cuda:0")
        host_rhs = np.array([[0.25, -0.5, 1.0]], dtype=np.float64)
        expected = apply_v_cycle(source, host_rhs.reshape(-1)).correction.reshape(1, 3)
        for shared in (False, True):
            with self.subTest(rhs_is_output=shared):
                backing, output = _external_vec3_view(size=1, stride=0, device=hierarchy.device)
                if shared:
                    backing.assign(host_rhs)
                    rhs = output
                else:
                    rhs = wp.array(host_rhs, dtype=wp.vec3d, device=hierarchy.device)
                workspace = hierarchy.create_workspace()
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    hierarchy.launch_apply(rhs, output, workspace)
                self.assertEqual(launch.call_count, hierarchy.scheduled_kernel_launches)
                np.testing.assert_allclose(output.numpy(), expected, rtol=3.0e-13, atol=3.0e-13)

    def test_cuda_terminal_fusion_is_all_buffer_bitwise_legacy_for_p1_p2_and_edge_values(self):
        for steps, expected_full in ((1, 12), (2, 20)):
            with self.subTest(steps=steps):
                hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
                    _mixed_hierarchy(smooth_steps=steps),
                    device="cuda:0",
                )
                generator = np.random.default_rng(181020 + steps)
                random_rhs = generator.normal(size=(hierarchy.n_free, 3))
                signed_zero_rhs = np.zeros((hierarchy.n_free, 3), dtype=np.float64)
                signed_zero_rhs.reshape(-1)[::2] = -0.0
                nonfinite_rhs = generator.normal(size=(hierarchy.n_free, 3))
                nonfinite_rhs.reshape(-1)[:8] = (np.inf, -np.inf, np.nan, -0.0, 0.0, np.nan, np.inf, -np.inf)
                self.assertEqual(hierarchy.scheduled_kernel_launches, expected_full)
                self.assertEqual(hierarchy.core_kernel_launches, expected_full - 1)
                self.assertEqual(hierarchy.seeded_core_kernel_launches, expected_full - 2)
                for label, rhs in (
                    ("random", random_rhs),
                    ("signed-zero", signed_zero_rhs),
                    ("nonfinite", nonfinite_rhs),
                ):
                    with self.subTest(steps=steps, rhs=label):
                        fused = hierarchy.create_workspace()
                        legacy = hierarchy.create_workspace()
                        _poison_workspace(fused)
                        _poison_workspace(legacy)
                        fused.rhs.assign(rhs)
                        legacy.rhs.assign(rhs)
                        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as fused_launch:
                            fused.launch()
                        with (
                            mock.patch.object(
                                WarpScalarFusedStaticMultigridHierarchy,
                                "supports_terminal_fusion",
                                new=property(lambda _self: False),
                            ),
                            mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as legacy_launch,
                        ):
                            legacy.launch()
                        self.assertEqual(fused_launch.call_count, expected_full)
                        self.assertEqual(legacy_launch.call_count, expected_full + 2)
                        self.assertEqual(_workspace_bit_patterns(fused), _workspace_bit_patterns(legacy))

    def test_cuda_terminal_fusion_capture_replay_restores_poison_bitwise_legacy(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cuda:0",
        )
        fused = hierarchy.create_workspace()
        legacy = hierarchy.create_workspace()
        warmup_rhs = np.random.default_rng(181024).normal(size=(hierarchy.n_free, 3))
        fused.rhs.assign(warmup_rhs)
        legacy.rhs.assign(warmup_rhs)
        fused.launch()
        with mock.patch.object(
            WarpScalarFusedStaticMultigridHierarchy,
            "supports_terminal_fusion",
            new=property(lambda _self: False),
        ):
            legacy.launch()
        with wp.ScopedCapture(device=hierarchy.device) as fused_capture:
            fused.launch()
        with (
            mock.patch.object(
                WarpScalarFusedStaticMultigridHierarchy,
                "supports_terminal_fusion",
                new=property(lambda _self: False),
            ),
            wp.ScopedCapture(device=hierarchy.device) as legacy_capture,
        ):
            legacy.launch()

        generator = np.random.default_rng(181025)
        signed_zero_rhs = np.zeros((hierarchy.n_free, 3), dtype=np.float64)
        signed_zero_rhs.reshape(-1)[1::2] = -0.0
        nonfinite_rhs = generator.normal(size=(hierarchy.n_free, 3))
        nonfinite_rhs.reshape(-1)[:4] = (np.inf, -np.inf, np.nan, -0.0)
        for label, rhs in (
            ("changed", generator.normal(size=(hierarchy.n_free, 3))),
            ("signed-zero", signed_zero_rhs),
            ("nonfinite", nonfinite_rhs),
        ):
            with self.subTest(rhs=label):
                _poison_workspace(fused)
                _poison_workspace(legacy)
                fused.rhs.assign(rhs)
                legacy.rhs.assign(rhs)
                wp.capture_launch(fused_capture.graph)
                wp.capture_launch(legacy_capture.graph)
                self.assertEqual(_workspace_bit_patterns(fused), _workspace_bit_patterns(legacy))

    def test_cuda_terminal_preflight_rejects_alias_and_layout_before_launch(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cuda:0",
        )

        def bind_workspace(workspace) -> None:
            workspace._persistent_arrays = workspace._current_arrays()
            workspace._persistent_pointers = tuple(int(array.ptr) for array in workspace._persistent_arrays)

        workspace = hierarchy.create_workspace()
        workspace.level_rhs = (*workspace.level_rhs[:-1], workspace.coarse_intermediate)
        bind_workspace(workspace)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "workspace arrays"):
                workspace.launch()
        self.assertEqual(launch.call_count, 0)

        workspace = hierarchy.create_workspace()
        coarse_size = hierarchy.levels[-1].scalar_size
        workspace.coarse_intermediate = wp.array(
            ptr=int(hierarchy.coarse_cholesky.ptr),
            dtype=wp.float64,
            shape=(coarse_size,),
            device=hierarchy.device,
            copy=False,
        )
        bind_workspace(workspace)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "static .* mutable"):
                workspace.launch()
        self.assertEqual(launch.call_count, 0)

        workspace = hierarchy.create_workspace()
        workspace.coarse_intermediate = wp.array(
            ptr=int(workspace.coarse_intermediate.ptr) + 4,
            dtype=wp.float64,
            shape=(coarse_size,),
            device=hierarchy.device,
            copy=False,
        )
        bind_workspace(workspace)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "naturally aligned"):
                workspace.launch()
        self.assertEqual(launch.call_count, 0)

        workspace = hierarchy.create_workspace()
        workspace.coarse_intermediate = wp.array(
            ptr=int(workspace.coarse_intermediate.ptr),
            dtype=wp.float64,
            shape=(coarse_size,),
            strides=(16,),
            device=hierarchy.device,
            copy=False,
        )
        bind_workspace(workspace)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            with self.assertRaisesRegex(ValueError, "contiguous one-dimensional"):
                workspace.launch()
        self.assertEqual(launch.call_count, 0)

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
        self.assertEqual(changed.physical_execution_authentication, PHYSICAL_EXECUTION_AUTHENTICATION)
        self.assertFalse(changed.solver_issued_authentication)
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

    def test_real_default_stretch_cuda_oracle_uses_18_launches(self):
        hierarchy = _default_stretch_hierarchy()
        device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cuda:0")
        workspace = device_hierarchy.create_workspace()
        rhs = np.random.default_rng(99118).normal(size=hierarchy.levels[0].matrix.scalar_size)
        workspace.set_rhs(rhs)
        workspace.launch()
        actual = workspace.record()
        self.assertEqual(actual.scheduled_kernel_launches, 18)
        self.assertEqual(actual.physical_work.core_kernel_launches, 17)
        self.assertEqual(actual.physical_work.terminal_fusion_kernel_launches, 1)
        self.assertEqual(actual.physical_work.terminal_level_index, 2)
        self.assertEqual(actual.physical_work.terminal_block_dim, 64)
        self.assertEqual(actual.physical_work.terminal_collective_count, 2)
        self.assertEqual(actual.physical_work.terminal_owner_thread, 0)
        self.assertEqual(actual.physical_work.terminal_fusion_version, TERMINAL_FUSION_VERSION)
        self.assertEqual(actual.physical_work.terminal_fusion_route, TERMINAL_FUSION_CUDA_ROUTE)
        np.testing.assert_allclose(
            actual.correction,
            apply_v_cycle(hierarchy, rhs).correction,
            rtol=4.0e-13,
            atol=4.0e-13,
        )
        self.assertFalse(actual.performance_evidence)

    def test_real_default_stretch_four_level_p2_launch_equation(self):
        hierarchy = _default_stretch_hierarchy(smooth_steps=2)
        self.assertEqual(len(hierarchy.levels), 4)
        device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cuda:0")
        self.assertEqual(device_hierarchy.scheduled_kernel_launches, 30)
        self.assertEqual(device_hierarchy.core_kernel_launches, 29)
        self.assertEqual(device_hierarchy.seeded_core_kernel_launches, 28)
        workspace = device_hierarchy.create_workspace()
        rhs = np.random.default_rng(181033).normal(size=hierarchy.levels[0].matrix.scalar_size)
        workspace.set_rhs(rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            workspace.launch()
        self.assertEqual(launch.call_count, 30)
        actual = workspace.record()
        self.assertEqual(actual.physical_work.matrix_kernel_launches, 12)
        self.assertEqual(actual.physical_work.jacobi_kernel_launches, 12)
        np.testing.assert_allclose(
            actual.correction,
            apply_v_cycle(hierarchy, rhs).correction,
            rtol=5.0e-13,
            atol=5.0e-13,
        )


if __name__ == "__main__":
    unittest.main()
