# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the scalar-row launch-fused research MG V-cycle."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from research.principal_stretch import correction_multigrid_warp_scalar_fused as scalar_fused_module
from research.principal_stretch.correction_gpu import MatrixFreeStableNHOperator
from research.principal_stretch.correction_graph_vbd import DirectGraphVBDConfig
from research.principal_stretch.correction_multigrid import (
    StaticBlockMatrix,
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
    NONTERMINAL_GENERIC_CPU_ROUTE,
    NONTERMINAL_GENERIC_CUDA_ROUTE,
    NONTERMINAL_LITERAL_CUDA_ROUTE,
    NONTERMINAL_LITERAL_DEFAULT_PHYSICAL_NODE_MAP,
    NONTERMINAL_LITERAL_KERNEL_VERSION,
    PHYSICAL_EXECUTION_AUTHENTICATION,
    PUBLICATION_VERSION,
    ROOT_INGRESS_COARSE_COPY_ROUTE,
    ROOT_INGRESS_EXTERNAL_SHARED_ROUTE,
    ROOT_INGRESS_INTERNAL_ROUTE,
    SCHEDULE_VERSION,
    STANDALONE_PUBLICATION_ROUTE,
    TERMINAL_FIXED12_COARSE_SOLVE_KERNEL_VERSION,
    TERMINAL_FIXED12_COARSE_SOLVE_ROUTE,
    TERMINAL_FUSION_COARSEST_ONLY_ROUTE,
    TERMINAL_FUSION_CPU_FALLBACK_ROUTE,
    TERMINAL_FUSION_CUDA_ROUTE,
    TERMINAL_FUSION_OVERSIZE_FALLBACK_ROUTE,
    TERMINAL_FUSION_VERSION,
    TERMINAL_GENERIC_COARSE_SOLVE_KERNEL_VERSION,
    TERMINAL_GENERIC_COARSE_SOLVE_ROUTE,
    TERMINAL_MICROCYCLE_CUDA_ROUTE,
    TERMINAL_MICROCYCLE_FIXED12_CUDA_ROUTE,
    TERMINAL_MICROCYCLE_KERNEL_VERSION,
    TERMINAL_MICROCYCLE_LOGICAL_PHASES,
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


def _rigid_terminal_coarse_hierarchy(
    node_count: int,
    coarse_node_limit: int,
    *,
    device_hash_marker: str,
) -> StaticMultigridHierarchy:
    """Build a three-level rigid path with a selected coarse scalar size."""
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
        coarse_node_limit=coarse_node_limit,
        maximum_levels=3,
        pre_smooth_steps=1,
        post_smooth_steps=1,
        static_model_sha256=device_hash_marker * 64,
    )
    if len(hierarchy.levels) != 3:
        raise RuntimeError("fixed coarse-size fixture changed level count")
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


def _terminal_block_boundary_hierarchy(terminal_node_count: int) -> StaticMultigridHierarchy:
    """Build a cheap three-level path with the requested terminal block count."""
    node_count = 4 * terminal_node_count
    identity = np.eye(3, dtype=np.float64)
    entries = []
    for node in range(node_count):
        diagonal = 0.5 + int(node > 0) + int(node + 1 < node_count)
        entries.append((node, node, diagonal * identity))
        if node + 1 < node_count:
            entries.append((node, node + 1, -identity))
            entries.append((node + 1, node, -identity))
    matrix = StaticBlockMatrix.from_block_entries(node_count, entries, block_size=3)
    rest = np.column_stack(
        (
            np.arange(node_count, dtype=np.float64),
            np.zeros(node_count, dtype=np.float64),
            np.zeros(node_count, dtype=np.float64),
        )
    )
    front = build_static_multigrid(
        matrix,
        rest,
        np.arange(node_count, dtype=np.int64),
        mode_kind="translation",
        target_aggregate_size=4,
        minimum_aggregate_size=2,
        coarse_node_limit=terminal_node_count,
        maximum_levels=2,
        pre_smooth_steps=1,
        post_smooth_steps=1,
        static_model_sha256="e" * 64,
    )
    terminal_level = front.levels[-1]
    tail = build_static_multigrid(
        terminal_level.matrix,
        front.rest_positions,
        terminal_level.node_ids,
        mode_kind="translation",
        target_aggregate_size=16,
        minimum_aggregate_size=8,
        coarse_node_limit=32,
        maximum_levels=2,
        pre_smooth_steps=1,
        post_smooth_steps=1,
        static_model_sha256="e" * 64,
    )
    hierarchy = dataclasses.replace(
        front,
        levels=(front.levels[0], *tail.levels),
        coarse_cholesky=tail.coarse_cholesky,
        content_sha256=("e" if terminal_node_count == 341 else "f") * 64,
    )
    if hierarchy.levels[-2].matrix.block_row_count != terminal_node_count:
        raise RuntimeError("terminal boundary fixture changed aggregation")
    return hierarchy


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
            "warp-scalar-fused-v-cycle-physical-work-v14",
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
            "warp-scalar-fused-v-cycle-result-v14",
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


def _clear_workspace(workspace) -> None:
    workspace.correction.fill_(wp.vec3d(0.0, 0.0, 0.0))
    for arrays in (
        workspace.level_rhs,
        workspace.level_correction,
        workspace.level_correction_alt,
        workspace.level_residual,
    ):
        for array in arrays:
            array.fill_(0.0)
    workspace.coarse_intermediate.fill_(0.0)


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
        self.assertEqual(KERNEL_VERSION, "mg-vbd-warp-static-v-cycle-scalar-fused-v9")
        self.assertEqual(CONTRACT_ID, "spectral-free-multiplicative-graph-vbd-warp-static-scalar-fused-v3")
        self.assertEqual(
            SCHEDULE_VERSION,
            "scalar-core-literal-nonterminal-fixed12-terminal-seeded-root-routes-v12",
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
        self.assertEqual(hierarchy.nonterminal_literal_kernel_version, NONTERMINAL_LITERAL_KERNEL_VERSION)
        self.assertEqual(hierarchy.nonterminal_literal_kernel_route, NONTERMINAL_GENERIC_CPU_ROUTE)
        self.assertEqual(hierarchy.nonterminal_literal_physical_nodes, 0)
        self.assertEqual(hierarchy.nonterminal_literal_physical_node_map, "")
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
            "_terminal_zero_jacobi_residual_restrict_solve_prolong_residual_jacobi",
            "_terminal_zero_jacobi_residual_restrict_fixed12_solve_prolong_residual_jacobi",
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

    def test_literal_kernels_freeze_scalar_row_source_order_and_direct_selectors(self):
        residual_specs = (
            (scalar_fused_module._scalar_csr_residual_bs3, 3),
            (scalar_fused_module._scalar_csr_residual_bs6, 6),
        )
        jacobi_specs = (
            (scalar_fused_module._zero_start_scalar_jacobi_bs6, 6, "output[scalar_row] = omega * value"),
            (
                scalar_fused_module._out_of_place_scalar_jacobi_bs3,
                3,
                "output[scalar_row] = current[scalar_row] + omega * value",
            ),
            (
                scalar_fused_module._out_of_place_scalar_jacobi_bs6,
                6,
                "output[scalar_row] = current[scalar_row] + omega * value",
            ),
        )
        for kernel, block_size in residual_specs:
            source = inspect.getsource(kernel.func)
            self.assertNotIn("for local_column", source)
            self.assertEqual(source.count("product +="), block_size)
            additions = ["values[value_base]"] + [f"values[value_base + {index}]" for index in range(1, block_size)]
            positions = [source.index(term) for term in additions]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(source.count("residual[scalar_row] = rhs[scalar_row] - product"), 1)
        for kernel, block_size, retained_store in jacobi_specs:
            source = inspect.getsource(kernel.func)
            self.assertNotIn("for local_column", source)
            self.assertEqual(source.count("value +="), block_size)
            additions = ["inverse_diagonal[block_base]"] + [
                f"inverse_diagonal[block_base + {index}]" for index in range(1, block_size)
            ]
            positions = [source.index(term) for term in additions]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(source.count(retained_store), 1)
        transfer_specs = (
            (scalar_fused_module._restrict_owned_rows_3to6, 3, "coarse_value[scalar_row] = result"),
            (scalar_fused_module._restrict_owned_rows_6to6, 6, "coarse_value[scalar_row] = result"),
            (scalar_fused_module._prolong_add_owned_rows_3from6, 6, "fine_value[scalar_row] += result"),
            (scalar_fused_module._prolong_add_owned_rows_6from6, 6, "fine_value[scalar_row] += result"),
        )
        for kernel, addition_count, retained_store in transfer_specs:
            source = inspect.getsource(kernel.func)
            self.assertNotIn("for fine_local", source)
            self.assertNotIn("for coarse_local", source)
            self.assertEqual(source.count("result +="), addition_count)
            self.assertEqual(source.count(retained_store), 1)

        for selector_name in (
            "_nonterminal_zero_start_kernel",
            "_nonterminal_restriction_kernel",
            "_nonterminal_prolongation_kernel",
        ):
            selector_source = inspect.getsource(getattr(WarpScalarFusedStaticMultigridHierarchy, selector_name))
            self.assertNotIn("globals(", selector_source)
            self.assertNotIn(".get(", selector_source)
        mutable_kernel_globals = [
            name
            for name, value in vars(scalar_fused_module).items()
            if "kernel" in name.lower() and isinstance(value, (dict, list, set))
        ]
        self.assertEqual(mutable_kernel_globals, [])

    def test_literal_kernel_module_hash_ignores_plausible_mutable_global_injection(self):
        script_prefix = (
            "import hashlib\nfrom research.principal_stretch import correction_multigrid_warp_scalar_fused as module\n"
        )
        scripts = (
            script_prefix,
            script_prefix
            + "module._NONTERMINAL_LITERAL_KERNEL_MAP = {'bs3': object()}\n"
            + "module._BLOCK_SIZE = 17\n"
            + "module._BLOCK_OFFSETS = [9, 1, 4]\n"
            + "module._SCALAR_ROW_VECTOR_TYPE = object()\n",
        )
        hashes = []
        for prefix in scripts:
            script = (
                prefix
                + "kernels = (module._scalar_csr_residual_bs3, module._scalar_csr_residual_bs6, "
                + "module._zero_start_scalar_jacobi_bs6, module._out_of_place_scalar_jacobi_bs3, "
                + "module._out_of_place_scalar_jacobi_bs6, module._restrict_owned_rows_3to6, "
                + "module._restrict_owned_rows_6to6, module._prolong_add_owned_rows_3from6, "
                + "module._prolong_add_owned_rows_6from6)\n"
                + "digest = hashlib.sha256()\n"
                + "[digest.update(kernel.module.get_module_hash()) for kernel in kernels]\n"
                + "print('LITERAL_MODULE_SHA256=' + digest.hexdigest())\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                check=False,
                capture_output=True,
                text=True,
                timeout=60.0,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            matches = [
                line.removeprefix("LITERAL_MODULE_SHA256=")
                for line in completed.stdout.splitlines()
                if line.startswith("LITERAL_MODULE_SHA256=")
            ]
            self.assertEqual(len(matches), 1, completed.stdout + completed.stderr)
            hashes.append(matches[0])
        self.assertEqual(hashes[0], hashes[1])

    def test_literal_selectors_keep_zero_bs3_and_transfer_6to3_generic(self):
        probe = object.__new__(WarpScalarFusedStaticMultigridHierarchy)
        object.__setattr__(probe, "_nonterminal_literal_kernel_route", NONTERMINAL_LITERAL_CUDA_ROUTE)
        object.__setattr__(probe, "_terminal_level_index", 1)
        object.__setattr__(
            probe,
            "_levels",
            (
                SimpleNamespace(block_size=6, coarse_block_size=3),
                SimpleNamespace(block_size=3, coarse_block_size=None),
            ),
        )
        self.assertIs(probe._nonterminal_restriction_kernel(0), scalar_fused_module._restrict_owned_rows)
        self.assertIs(probe._nonterminal_prolongation_kernel(0), scalar_fused_module._prolong_add_owned_rows)
        object.__setattr__(
            probe,
            "_levels",
            (
                SimpleNamespace(block_size=3, coarse_block_size=3),
                SimpleNamespace(block_size=3, coarse_block_size=None),
            ),
        )
        self.assertIs(probe._nonterminal_zero_start_kernel(0), scalar_fused_module._zero_start_scalar_jacobi)
        self.assertIs(probe._nonterminal_restriction_kernel(0), scalar_fused_module._restrict_owned_rows)
        self.assertIs(probe._nonterminal_prolongation_kernel(0), scalar_fused_module._prolong_add_owned_rows)

    def test_terminal_microcycle_uses_literal_owner_and_six_unconditional_collectives(self):
        kernel = scalar_fused_module._terminal_zero_jacobi_residual_restrict_solve_prolong_residual_jacobi
        source = inspect.getsource(kernel.func)
        self.assertEqual(source.count("wp.tile_from_thread("), 6)
        self.assertEqual(source.count('storage="shared"'), 6)
        self.assertEqual(source.count("thread_idx=0"), 6)
        self.assertEqual(source.count("wp.tile_extract("), 6)
        self.assertNotIn("atomic_", source)
        for name in ("_TERMINAL_OWNER_THREAD", "_TERMINAL_COLLECTIVE_COUNT", "_TERMINAL_BLOCK_DIM"):
            self.assertNotIn(name, source)
        original_hash = kernel.module.get_module_hash()
        try:
            scalar_fused_module._TERMINAL_OWNER_THREAD = 17
            scalar_fused_module._TERMINAL_COLLECTIVE_COUNT = 1
            scalar_fused_module._TERMINAL_BLOCK_DIM = 32
            self.assertEqual(kernel.module.get_module_hash(), original_hash)
        finally:
            del scalar_fused_module._TERMINAL_OWNER_THREAD
            del scalar_fused_module._TERMINAL_COLLECTIVE_COUNT
            del scalar_fused_module._TERMINAL_BLOCK_DIM

    def test_fixed12_solve_is_literal_ordered_and_retains_all_global_stores(self):
        kernel = scalar_fused_module._terminal_zero_jacobi_residual_restrict_fixed12_solve_prolong_residual_jacobi
        source = inspect.getsource(kernel.func)
        self.assertEqual(source.count("wp.tile_from_thread("), 6)
        self.assertEqual(source.count('storage="shared"'), 6)
        self.assertEqual(source.count("thread_idx=0"), 6)
        self.assertEqual(source.count("wp.tile_extract("), 6)
        self.assertNotIn("atomic_", source)
        self.assertNotIn("_Vec12d", inspect.getsource(scalar_fused_module))
        self.assertNotIn("wp.types.vector", source)
        solve = source[source.index("value = coarse_rhs[0]") : source.index("solve_tile = wp.tile_from_thread(")]
        self.assertNotIn("for ", solve)
        self.assertNotIn("while ", solve)
        self.assertEqual(solve.count("intermediate["), 12)
        self.assertEqual(solve.count("coarse_solution["), 12)
        for row in range(12):
            self.assertIn(f"work{row} =", solve)
        forward_offsets = [solve.index(f"intermediate[{row}]") for row in range(12)]
        backward_offsets = [solve.index(f"coarse_solution[{row}]") for row in range(11, -1, -1)]
        self.assertEqual(forward_offsets, sorted(forward_offsets))
        self.assertEqual(backward_offsets, sorted(backward_offsets))
        self.assertIn("lower[143]", solve)

    def test_fresh_process_legacy_vec12_injection_cannot_change_prejit_module_hash(self):
        script_prefix = """
import importlib
import warp as wp
module = importlib.import_module("research.principal_stretch.correction_multigrid_warp_scalar_fused")
assert "_Vec12d" not in vars(module)
"""
        hashes = []
        for injection in (
            "\n",
            "\nmodule._Vec12d = wp.types.vector(length=11, dtype=wp.float64)\n",
            "\nmodule._Vec12d = wp.types.vector(length=12, dtype=wp.float32)\n",
        ):
            script = (
                script_prefix
                + injection
                + "kernel = module._terminal_zero_jacobi_residual_restrict_fixed12_solve_prolong_residual_jacobi\n"
                + 'print("FIXED12_MODULE_SHA256=" + kernel.module.get_module_hash(64).hex())\n'
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            marker = "FIXED12_MODULE_SHA256="
            matches = [line.removeprefix(marker) for line in completed.stdout.splitlines() if line.startswith(marker)]
            self.assertEqual(len(matches), 1, completed.stdout + completed.stderr)
            hashes.append(matches[0])
        self.assertEqual(len(set(hashes)), 1)

    def test_cpu_coarse_solve_sizes_and_non64_fixed12_candidate_use_generic_fallback(self):
        fixtures = (
            ("coarse6", _rigid_terminal_coarse_hierarchy(16, 1, device_hash_marker="6"), 6),
            ("coarse18", _rigid_terminal_coarse_hierarchy(48, 3, device_hash_marker="7"), 18),
            ("coarse96", _rigid_terminal_coarse_hierarchy(256, 16, device_hash_marker="8"), 96),
            ("coarse12-non64", _terminal_block_boundary_hierarchy(64), 12),
        )
        for label, source, coarse_size in fixtures:
            with self.subTest(label=label):
                hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(source, device="cpu")
                self.assertEqual(hierarchy.terminal_coarse_scalar_size, coarse_size)
                self.assertEqual(
                    hierarchy.terminal_coarse_solve_kernel_version,
                    TERMINAL_GENERIC_COARSE_SOLVE_KERNEL_VERSION,
                )
                self.assertEqual(hierarchy.terminal_coarse_solve_route, TERMINAL_GENERIC_COARSE_SOLVE_ROUTE)
                self.assertEqual(hierarchy.terminal_fusion_route, TERMINAL_FUSION_CPU_FALLBACK_ROUTE)
                self.assertFalse(hierarchy.supports_fixed12_terminal_microcycle)
        non64 = fixtures[-1][1]
        required_threads = max(non64.levels[-2].matrix.scalar_size, non64.levels[-1].matrix.scalar_size)
        self.assertEqual(((required_threads + 31) // 32) * 32, 192)

    def test_cpu_oracle_p1_p2_exact_launches_work_and_fixed_b_result(self):
        for steps, expected_launches, expected_recurrence_phases in ((1, 14, 4), (2, 22, 8)):
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
                    self.assertEqual(
                        actual.physical_work.matrix_recurrence_phases,
                        expected_recurrence_phases,
                    )
                    self.assertEqual(
                        actual.physical_work.jacobi_recurrence_phases,
                        expected_recurrence_phases,
                    )
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
                    self.assertEqual(
                        actual.physical_work.nonterminal_literal_kernel_version,
                        NONTERMINAL_LITERAL_KERNEL_VERSION,
                    )
                    self.assertEqual(
                        actual.physical_work.nonterminal_literal_kernel_route, NONTERMINAL_GENERIC_CPU_ROUTE
                    )
                    self.assertEqual(actual.physical_work.nonterminal_literal_physical_nodes, 0)
                    self.assertEqual(actual.physical_work.nonterminal_literal_physical_node_map, "")
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
                "terminal_fusion_launch_reduction": 2,
                "terminal_block_dim": 64,
                "terminal_collective_count": 2,
                "terminal_owner_thread": 0,
                "terminal_logical_phases": "restriction|ordered-coarse-cholesky|prolongation",
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
            ("terminal_fusion_launch_reduction", 2),
            ("terminal_level_index", 0),
            ("terminal_block_dim", 64),
            ("terminal_collective_count", 2),
            ("terminal_owner_thread", 0),
            ("terminal_microcycle_kernel_version", TERMINAL_MICROCYCLE_KERNEL_VERSION + "-forged"),
            ("terminal_logical_phases", "forged-phase"),
            ("terminal_fusion_version", TERMINAL_FUSION_VERSION + "-forged"),
            ("terminal_fusion_route", TERMINAL_FUSION_CUDA_ROUTE),
            ("zero_start_block_solves", record.physical_work.zero_start_block_solves + 1),
            ("matrix_recurrence_phases", record.physical_work.matrix_recurrence_phases + 1),
            ("jacobi_recurrence_phases", record.physical_work.jacobi_recurrence_phases + 1),
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
        self.assertEqual(actual.physical_work.matrix_recurrence_phases, 6)
        self.assertEqual(actual.physical_work.jacobi_recurrence_phases, 6)
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

    @staticmethod
    def _generic_nonterminal_kernels() -> dict[str, object]:
        return {
            "_scalar_csr_residual_bs3": scalar_fused_module._scalar_csr_residual,
            "_scalar_csr_residual_bs6": scalar_fused_module._scalar_csr_residual,
            "_zero_start_scalar_jacobi_bs6": scalar_fused_module._zero_start_scalar_jacobi,
            "_out_of_place_scalar_jacobi_bs3": scalar_fused_module._out_of_place_scalar_jacobi,
            "_out_of_place_scalar_jacobi_bs6": scalar_fused_module._out_of_place_scalar_jacobi,
            "_restrict_owned_rows_3to6": scalar_fused_module._restrict_owned_rows,
            "_restrict_owned_rows_6to6": scalar_fused_module._restrict_owned_rows,
            "_prolong_add_owned_rows_3from6": scalar_fused_module._prolong_add_owned_rows,
            "_prolong_add_owned_rows_6from6": scalar_fused_module._prolong_add_owned_rows,
        }

    def test_cuda_literal_route_exact_map_trace_and_generic_fallbacks(self):
        default_source = _default_stretch_hierarchy()
        default = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(default_source, device="cuda:0")
        self.assertEqual(default.nonterminal_literal_kernel_version, NONTERMINAL_LITERAL_KERNEL_VERSION)
        self.assertEqual(default.nonterminal_literal_kernel_route, NONTERMINAL_LITERAL_CUDA_ROUTE)
        self.assertEqual(default.nonterminal_literal_physical_nodes, 11)
        self.assertEqual(default.nonterminal_literal_physical_node_map, NONTERMINAL_LITERAL_DEFAULT_PHYSICAL_NODE_MAP)
        workspace = default.create_workspace()
        workspace.set_rhs(np.random.default_rng(190817).normal(size=default.n_free_dofs))
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            workspace.launch()
        kernels = [call.args[0] for call in launch.call_args_list]
        expected_counts = (
            (scalar_fused_module._scalar_csr_residual_bs3, 2),
            (scalar_fused_module._scalar_csr_residual_bs6, 2),
            (scalar_fused_module._zero_start_scalar_jacobi_bs6, 1),
            (scalar_fused_module._out_of_place_scalar_jacobi_bs3, 1),
            (scalar_fused_module._out_of_place_scalar_jacobi_bs6, 1),
            (scalar_fused_module._restrict_owned_rows_3to6, 1),
            (scalar_fused_module._restrict_owned_rows_6to6, 1),
            (scalar_fused_module._prolong_add_owned_rows_3from6, 1),
            (scalar_fused_module._prolong_add_owned_rows_6from6, 1),
        )
        for kernel, count in expected_counts:
            self.assertEqual(kernels.count(kernel), count, str(kernel))
        self.assertEqual(sum(kernels.count(kernel) for kernel, _count in expected_counts), 11)
        for generic_kernel in (
            scalar_fused_module._scalar_csr_residual,
            scalar_fused_module._zero_start_scalar_jacobi,
            scalar_fused_module._out_of_place_scalar_jacobi,
            scalar_fused_module._restrict_owned_rows,
            scalar_fused_module._prolong_add_owned_rows,
        ):
            self.assertNotIn(generic_kernel, kernels)
        record = workspace.record()
        self.assertEqual(record.physical_work.nonterminal_literal_kernel_version, NONTERMINAL_LITERAL_KERNEL_VERSION)
        self.assertEqual(record.physical_work.nonterminal_literal_kernel_route, NONTERMINAL_LITERAL_CUDA_ROUTE)
        self.assertEqual(record.physical_work.nonterminal_literal_physical_nodes, 11)
        self.assertEqual(
            record.physical_work.nonterminal_literal_physical_node_map,
            NONTERMINAL_LITERAL_DEFAULT_PHYSICAL_NODE_MAP,
        )
        self.assertEqual(record.physical_work.matrix_recurrence_phases, 6)
        self.assertEqual(record.physical_work.jacobi_recurrence_phases, 6)

        translation_source = _translation_hierarchy(16)
        translation = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(translation_source, device="cuda:0")
        self.assertEqual(translation.nonterminal_literal_kernel_route, NONTERMINAL_LITERAL_CUDA_ROUTE)
        self.assertEqual(translation.nonterminal_literal_physical_nodes, 3)
        self.assertEqual(
            translation.nonterminal_literal_physical_node_map,
            "residual-bs3=2|jacobi-bs3=1",
        )
        translation_workspace = translation.create_workspace()
        translation_workspace.set_rhs(np.random.default_rng(190818).normal(size=translation.n_free_dofs))
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as translation_launch:
            translation_workspace.launch()
        translation_kernels = [call.args[0] for call in translation_launch.call_args_list]
        self.assertEqual(translation_kernels.count(scalar_fused_module._scalar_csr_residual_bs3), 2)
        self.assertEqual(translation_kernels.count(scalar_fused_module._out_of_place_scalar_jacobi_bs3), 1)
        self.assertEqual(translation_kernels.count(scalar_fused_module._restrict_owned_rows), 1)
        self.assertEqual(translation_kernels.count(scalar_fused_module._prolong_add_owned_rows), 1)
        self.assertNotIn(scalar_fused_module._zero_start_scalar_jacobi_bs6, translation_kernels)
        self.assertNotIn(scalar_fused_module._restrict_owned_rows_3to6, translation_kernels)
        self.assertNotIn(scalar_fused_module._prolong_add_owned_rows_3from6, translation_kernels)

        terminal_root = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _translation_hierarchy(341),
            device="cuda:0",
        )
        self.assertEqual(terminal_root.nonterminal_literal_kernel_route, NONTERMINAL_GENERIC_CUDA_ROUTE)
        self.assertEqual(terminal_root.nonterminal_literal_physical_nodes, 0)
        self.assertEqual(terminal_root.nonterminal_literal_physical_node_map, "")

        p2 = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _default_stretch_hierarchy(smooth_steps=2),
            device="cuda:0",
        )
        self.assertEqual(p2.nonterminal_literal_physical_nodes, 19)
        self.assertEqual(
            p2.nonterminal_literal_physical_node_map,
            "residual-bs3=4|residual-bs6=4|zero-jacobi-bs6=1|jacobi-bs3=3|jacobi-bs6=3|"
            "restrict-3to6=1|restrict-6to6=1|prolong-3from6=1|prolong-6from6=1",
        )
        self.assertEqual(p2.scheduled_kernel_launches, 30)

    def test_cuda_literal_all17_direct_capture_poison_signedzero_nonfinite_and_sticky_are_bitwise_generic(self):
        source = _default_stretch_hierarchy()
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(source, device="cuda:0")
        literal = hierarchy.create_workspace()
        generic = hierarchy.create_workspace()
        pointers = (_pointer_tuple(hierarchy, literal), _pointer_tuple(hierarchy, generic))
        generator = np.random.default_rng(190819)
        random_rhs = generator.normal(size=(hierarchy.n_free, 3))
        signed_zero_rhs = np.zeros((hierarchy.n_free, 3), dtype=np.float64)
        signed_zero_rhs.reshape(-1)[1::2] = -0.0
        nonfinite_rhs = generator.normal(size=(hierarchy.n_free, 3))
        nonfinite_rhs.reshape(-1)[:8] = (np.inf, -np.inf, np.nan, -0.0, 0.0, np.nan, np.inf, -np.inf)
        generic_kernels = self._generic_nonterminal_kernels()

        def run_literal(rhs: np.ndarray) -> None:
            literal.rhs.assign(rhs)
            literal.launch()

        def run_generic(rhs: np.ndarray) -> None:
            generic.rhs.assign(rhs)
            with mock.patch.multiple(scalar_fused_module, **generic_kernels):
                generic.launch()

        for initialization in ("clean", "poison"):
            for order in ("literal-generic", "generic-literal"):
                for label, rhs in (
                    ("random", random_rhs),
                    ("signed-zero", signed_zero_rhs),
                    ("nonfinite", nonfinite_rhs),
                ):
                    with self.subTest(initialization=initialization, order=order, rhs=label):
                        initializer = _clear_workspace if initialization == "clean" else _poison_workspace
                        initializer(literal)
                        initializer(generic)
                        first, second = (
                            (run_literal, run_generic) if order == "literal-generic" else (run_generic, run_literal)
                        )
                        first(rhs)
                        second(rhs)
                        literal_patterns = _workspace_bit_patterns(literal)
                        generic_patterns = _workspace_bit_patterns(generic)
                        self.assertEqual(len(literal_patterns), 17)
                        self.assertEqual(literal_patterns, generic_patterns)

        run_literal(nonfinite_rhs)
        run_generic(nonfinite_rhs)
        run_literal(random_rhs)
        run_generic(random_rhs)
        self.assertEqual(_workspace_bit_patterns(literal), _workspace_bit_patterns(generic))

        literal.rhs.assign(random_rhs)
        generic.rhs.assign(random_rhs)
        with wp.ScopedCapture(device=hierarchy.device) as literal_capture:
            literal.launch()
        with (
            mock.patch.multiple(scalar_fused_module, **generic_kernels),
            wp.ScopedCapture(device=hierarchy.device) as generic_capture,
        ):
            generic.launch()
        for replay_index, (order, rhs) in enumerate(
            (
                ("literal-generic", signed_zero_rhs),
                ("generic-literal", nonfinite_rhs),
                ("literal-generic", random_rhs),
                ("generic-literal", random_rhs),
            )
        ):
            with self.subTest(replay=replay_index, order=order):
                _poison_workspace(literal)
                _poison_workspace(generic)
                literal.rhs.assign(rhs)
                generic.rhs.assign(rhs)
                first_graph, second_graph = (
                    (literal_capture.graph, generic_capture.graph)
                    if order == "literal-generic"
                    else (generic_capture.graph, literal_capture.graph)
                )
                wp.capture_launch(first_graph)
                wp.capture_launch(second_graph)
                literal_patterns = _workspace_bit_patterns(literal)
                generic_patterns = _workspace_bit_patterns(generic)
                self.assertEqual(len(literal_patterns), 17)
                self.assertEqual(literal_patterns, generic_patterns)
        self.assertEqual((_pointer_tuple(hierarchy, literal), _pointer_tuple(hierarchy, generic)), pointers)

    def test_cuda_literal_route_map_tamper_fails_before_launch_even_with_coherent_signature(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _default_stretch_hierarchy(),
            device="cuda:0",
        )
        workspace = hierarchy.create_workspace()
        workspace.set_rhs(np.random.default_rng(190821).normal(size=hierarchy.n_free_dofs))
        original = (
            hierarchy._nonterminal_literal_kernel_route,
            hierarchy._nonterminal_literal_physical_nodes,
            hierarchy._nonterminal_literal_physical_node_map,
        )
        attacks = (
            (
                NONTERMINAL_LITERAL_CUDA_ROUTE,
                11,
                "residual-bs6=2|residual-bs3=2|zero-jacobi-bs6=1|jacobi-bs3=1|jacobi-bs6=1|"
                "restrict-3to6=1|restrict-6to6=1|prolong-3from6=1|prolong-6from6=1",
            ),
            (
                NONTERMINAL_LITERAL_CUDA_ROUTE,
                13,
                NONTERMINAL_LITERAL_DEFAULT_PHYSICAL_NODE_MAP + "|residual-bs3=2",
            ),
            (
                NONTERMINAL_LITERAL_CUDA_ROUTE,
                9,
                "residual-bs3=2|residual-bs6=2|zero-jacobi-bs6=1|jacobi-bs3=1|jacobi-bs6=1|"
                "restrict-3to6=1|restrict-6to6=1",
            ),
            (NONTERMINAL_GENERIC_CUDA_ROUTE, 0, ""),
        )
        try:
            for route, count, physical_map in attacks:
                with self.subTest(route=route, count=count, map=physical_map):
                    object.__setattr__(hierarchy, "_nonterminal_literal_kernel_route", route)
                    object.__setattr__(hierarchy, "_nonterminal_literal_physical_nodes", count)
                    object.__setattr__(hierarchy, "_nonterminal_literal_physical_node_map", physical_map)
                    object.__setattr__(hierarchy, "_nonterminal_literal_route_signature", (route, count, physical_map))
                    with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                        with self.assertRaisesRegex(RuntimeError, "hierarchy identity changed"):
                            workspace.launch()
                    self.assertEqual(launch.call_count, 0)
        finally:
            object.__setattr__(hierarchy, "_nonterminal_literal_kernel_route", original[0])
            object.__setattr__(hierarchy, "_nonterminal_literal_physical_nodes", original[1])
            object.__setattr__(hierarchy, "_nonterminal_literal_physical_node_map", original[2])
            object.__setattr__(hierarchy, "_nonterminal_literal_route_signature", original)

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
        self.assertFalse(hierarchy.supports_terminal_microcycle)
        self.assertEqual(hierarchy.terminal_fusion_route, TERMINAL_FUSION_CUDA_ROUTE)
        self.assertEqual(hierarchy.terminal_fusion_kernel_launches, 1)
        self.assertEqual(hierarchy.terminal_fusion_launch_reduction, 2)
        self.assertEqual(hierarchy.terminal_level_index, len(hierarchy.levels) - 2)
        self.assertEqual(hierarchy.terminal_block_dim, 64)
        self.assertEqual(hierarchy.terminal_collective_count, 2)
        self.assertEqual(hierarchy.terminal_owner_thread, 0)
        self.assertEqual(hierarchy.terminal_logical_phases, "restriction|ordered-coarse-cholesky|prolongation")
        self.assertEqual(hierarchy.scheduled_kernel_launches, 20)
        self.assertEqual(hierarchy.core_kernel_launches, 19)
        self.assertEqual(hierarchy.seeded_core_kernel_launches, 18)

        terminal_root = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _translation_hierarchy(341),
            device="cuda:0",
        )
        self.assertEqual(terminal_root.terminal_level_index, 0)
        self.assertFalse(terminal_root.supports_terminal_microcycle)
        self.assertEqual(terminal_root.terminal_fusion_route, TERMINAL_FUSION_CUDA_ROUTE)
        self.assertEqual(terminal_root.terminal_fusion_launch_reduction, 2)

        supported_cpu = _terminal_block_boundary_hierarchy(341)
        supported = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(supported_cpu, device="cuda:0")
        self.assertEqual(supported.levels[-2].scalar_size, 1023)
        self.assertTrue(supported.supports_terminal_fusion)
        self.assertTrue(supported.supports_terminal_microcycle)
        self.assertEqual(supported.terminal_fusion_route, TERMINAL_MICROCYCLE_CUDA_ROUTE)
        self.assertEqual(supported.terminal_fusion_launch_reduction, 6)
        self.assertEqual(supported.terminal_block_dim, 1024)
        self.assertEqual(supported.terminal_collective_count, 6)
        boundary_rhs = np.random.default_rng(181032).normal(size=supported_cpu.levels[0].matrix.scalar_size)
        fused = supported.create_workspace()
        b2 = supported.create_workspace()
        _poison_workspace(fused)
        _poison_workspace(b2)
        fused.set_rhs(boundary_rhs)
        b2.set_rhs(boundary_rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as fused_launch:
            fused.launch()
        with (
            mock.patch.object(
                WarpScalarFusedStaticMultigridHierarchy,
                "supports_terminal_microcycle",
                new=property(lambda _self: False),
            ),
            mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as b2_launch,
        ):
            b2.launch()
        self.assertEqual(fused_launch.call_count, supported.scheduled_kernel_launches)
        self.assertEqual(b2_launch.call_count, fused_launch.call_count + 4)
        self.assertEqual(_workspace_bit_patterns(fused), _workspace_bit_patterns(b2))

        oversized_cpu = _terminal_block_boundary_hierarchy(342)
        oversized = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(oversized_cpu, device="cuda:0")
        self.assertEqual(oversized.levels[-2].scalar_size, 1026)
        self.assertFalse(oversized.supports_terminal_fusion)
        self.assertFalse(oversized.supports_terminal_microcycle)
        self.assertEqual(oversized.terminal_fusion_route, TERMINAL_FUSION_OVERSIZE_FALLBACK_ROUTE)
        self.assertEqual(oversized.terminal_block_dim, 0)
        self.assertEqual(oversized.scheduled_kernel_launches, 14)
        rhs = np.random.default_rng(181018).normal(size=oversized_cpu.levels[0].matrix.scalar_size)
        workspace = oversized.create_workspace()
        workspace.set_rhs(rhs)
        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
            workspace.launch()
        self.assertEqual(launch.call_count, 14)
        np.testing.assert_allclose(
            workspace.record().correction,
            apply_v_cycle(oversized_cpu, rhs).correction,
            rtol=5.0e-13,
            atol=5.0e-13,
        )

    def test_cuda_generic_coarse_solve_routes_cover_6_18_96_and_coarse12_non64(self):
        fixtures = (
            ("coarse6", _rigid_terminal_coarse_hierarchy(16, 1, device_hash_marker="6"), 6, 32),
            ("coarse18", _rigid_terminal_coarse_hierarchy(48, 3, device_hash_marker="7"), 18, 96),
            ("coarse96", _rigid_terminal_coarse_hierarchy(256, 16, device_hash_marker="8"), 96, 384),
            ("coarse12-non64", _terminal_block_boundary_hierarchy(64), 12, 192),
        )
        fixed_kernel = scalar_fused_module._terminal_zero_jacobi_residual_restrict_fixed12_solve_prolong_residual_jacobi
        generic_kernel = scalar_fused_module._terminal_zero_jacobi_residual_restrict_solve_prolong_residual_jacobi
        for label, source, coarse_size, block_dim in fixtures:
            with self.subTest(label=label):
                hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(source, device="cuda:0")
                self.assertTrue(hierarchy.supports_terminal_microcycle)
                self.assertFalse(hierarchy.supports_fixed12_terminal_microcycle)
                self.assertEqual(hierarchy.terminal_fusion_route, TERMINAL_MICROCYCLE_CUDA_ROUTE)
                self.assertEqual(hierarchy.terminal_block_dim, block_dim)
                self.assertEqual(hierarchy.terminal_coarse_scalar_size, coarse_size)
                self.assertEqual(
                    hierarchy.terminal_coarse_solve_kernel_version,
                    TERMINAL_GENERIC_COARSE_SOLVE_KERNEL_VERSION,
                )
                self.assertEqual(hierarchy.terminal_coarse_solve_route, TERMINAL_GENERIC_COARSE_SOLVE_ROUTE)
                workspace = hierarchy.create_workspace()
                rhs = np.random.default_rng(185000 + coarse_size).normal(size=hierarchy.n_free_dofs)
                workspace.set_rhs(rhs)
                with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch:
                    workspace.launch()
                kernels = [call.args[0] for call in launch.call_args_list]
                self.assertEqual(kernels.count(generic_kernel), 1)
                self.assertNotIn(fixed_kernel, kernels)
                np.testing.assert_allclose(
                    workspace.record().correction,
                    apply_v_cycle(source, rhs).correction,
                    rtol=5.0e-13,
                    atol=5.0e-13,
                )

    def test_cuda_fixed12_is_all_workspace_bitwise_generic_for_ab_ba_edge_and_random_factor(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cuda:0",
        )
        self.assertTrue(hierarchy.supports_fixed12_terminal_microcycle)
        self.assertEqual(hierarchy.terminal_block_dim, 64)
        self.assertEqual(hierarchy.terminal_coarse_scalar_size, 12)
        self.assertEqual(
            hierarchy.terminal_coarse_solve_kernel_version,
            TERMINAL_FIXED12_COARSE_SOLVE_KERNEL_VERSION,
        )
        self.assertEqual(hierarchy.terminal_coarse_solve_route, TERMINAL_FIXED12_COARSE_SOLVE_ROUTE)
        generator = np.random.default_rng(185120)
        random_rhs = generator.normal(size=(hierarchy.n_free, 3))
        signed_zero_rhs = np.zeros((hierarchy.n_free, 3), dtype=np.float64)
        signed_zero_rhs.reshape(-1)[1::2] = -0.0
        nonfinite_rhs = generator.normal(size=(hierarchy.n_free, 3))
        nonfinite_rhs.reshape(-1)[:8] = (np.inf, -np.inf, np.nan, -0.0, 0.0, np.nan, np.inf, -np.inf)
        original_lower = np.ascontiguousarray(hierarchy.coarse_cholesky.numpy())
        random_lower = np.tril(generator.uniform(-0.2, 0.2, size=(12, 12)))
        random_lower[np.diag_indices(12)] = generator.uniform(0.8, 1.8, size=12)
        cases = (
            ("random", random_rhs, original_lower),
            ("signed-zero", signed_zero_rhs, original_lower),
            ("nonfinite", nonfinite_rhs, original_lower),
            ("random-factor", random_rhs, np.ascontiguousarray(random_lower.reshape(-1))),
        )
        try:
            for label, rhs, lower in cases:
                for poisoned in (False, True):
                    for order in ("fixed-generic", "generic-fixed"):
                        with self.subTest(label=label, poisoned=poisoned, order=order):
                            fixed = hierarchy.create_workspace()
                            generic = hierarchy.create_workspace()
                            initialize = _poison_workspace if poisoned else _clear_workspace
                            initialize(fixed)
                            initialize(generic)
                            fixed.rhs.assign(rhs)
                            generic.rhs.assign(rhs)
                            hierarchy.coarse_cholesky.assign(lower)

                            def launch_fixed(fixed=fixed) -> None:
                                fixed.launch()

                            def launch_generic(generic=generic) -> None:
                                with mock.patch.object(
                                    WarpScalarFusedStaticMultigridHierarchy,
                                    "supports_fixed12_terminal_microcycle",
                                    new=property(lambda _self: False),
                                ):
                                    generic.launch()

                            first, second = (
                                (launch_fixed, launch_generic)
                                if order == "fixed-generic"
                                else (launch_generic, launch_fixed)
                            )
                            first()
                            second()
                            self.assertEqual(_workspace_bit_patterns(fixed), _workspace_bit_patterns(generic))
        finally:
            hierarchy.coarse_cholesky.assign(original_lower)

    def test_cuda_fixed12_generated_kernel_has_no_local_or_stack_spills_at_block64(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cuda:0",
        )
        self.assertTrue(hierarchy.supports_fixed12_terminal_microcycle)
        self.assertEqual(hierarchy.terminal_block_dim, 64)
        wp.load_module(module=scalar_fused_module, device=hierarchy.device, block_dim=64)
        symbol = "terminal_zero_jacobi_residual_restrict_fixed12_solve_prolong_residual_jacobi"
        module_hash_prefix = wp.get_module(scalar_fused_module.__name__).get_module_hash(64).hex()[:7]
        cache_root = Path(str(wp.config.kernel_cache_dir))
        sources = [
            path
            for path in cache_root.rglob("*.cu")
            if path.parent.name.endswith(module_hash_prefix) and symbol.encode() in path.read_bytes()
        ]
        self.assertTrue(sources, f"no generated CUDA source for {symbol!r} under {cache_root}")
        source = max(sources, key=lambda path: path.stat().st_mtime_ns)
        cubins = sorted(source.parent.glob("*.cubin"), key=lambda path: path.stat().st_mtime_ns)
        self.assertTrue(cubins, f"no cubin found beside {source}")
        cubin = cubins[-1]
        cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda-12.6/bin/cuobjdump"
        self.assertTrue(Path(cuobjdump).is_file(), "cuobjdump is required for the fixed12 resource gate")
        resource_output = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        resource_sections = [
            section for section in re.split(r"(?m)(?=^[ \t]*Function\b)", resource_output) if symbol in section
        ]
        self.assertTrue(resource_sections, f"resource output did not name {symbol!r}")
        resource = "\n".join(resource_sections)
        registers = [int(value) for value in re.findall(r"\bREG:(\d+)", resource)]
        local = [int(value) for value in re.findall(r"\bLOCAL:(\d+)", resource)]
        stack = [int(value) for value in re.findall(r"\bSTACK:(\d+)", resource)]
        shared = [int(value) for value in re.findall(r"\bSHARED:(\d+)", resource)]
        self.assertTrue(registers, resource)
        self.assertGreater(max(registers), 0)
        self.assertEqual(max(local, default=0), 0)
        self.assertEqual(max(stack, default=0), 0)
        self.assertGreater(max(shared, default=0), 0)
        sass_output = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        sass_sections = [
            section for section in re.split(r"(?m)(?=^[ \t]*Function\b)", sass_output) if symbol in section
        ]
        sass = "\n".join(sass_sections)
        self.assertTrue(sass_sections, f"SASS output did not name {symbol!r}")
        self.assertEqual(len(re.findall(r"\bLDL(?:\.|\b)", sass)), 0)
        self.assertEqual(len(re.findall(r"\bSTL(?:\.|\b)", sass)), 0)
        generated = source.read_text(errors="replace")
        excerpt = generated[generated.index(symbol) : generated.index(symbol) + 120_000]
        self.assertNotIn("vec_t<12", excerpt)
        self.assertIn("// work0 =", excerpt)
        self.assertIn("// work11 =", excerpt)

    def test_cuda_all_nine_literal_kernels_have_no_local_stack_or_ldl_stl_spills(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _default_stretch_hierarchy(),
            device="cuda:0",
        )
        workspace = hierarchy.create_workspace()
        workspace.set_rhs(np.random.default_rng(190820).normal(size=hierarchy.n_free_dofs))
        workspace.launch()
        symbols = (
            "scalar_csr_residual_bs3",
            "scalar_csr_residual_bs6",
            "zero_start_scalar_jacobi_bs6",
            "out_of_place_scalar_jacobi_bs3",
            "out_of_place_scalar_jacobi_bs6",
            "restrict_owned_rows_3to6",
            "restrict_owned_rows_6to6",
            "prolong_add_owned_rows_3from6",
            "prolong_add_owned_rows_6from6",
        )
        module_hash_prefix = wp.get_module(scalar_fused_module.__name__).get_module_hash().hex()[:7]
        cache_root = Path(str(wp.config.kernel_cache_dir))
        sources = [
            path
            for path in cache_root.rglob("*.cu")
            if path.parent.name.endswith(module_hash_prefix)
            and all(symbol.encode() in path.read_bytes() for symbol in symbols)
        ]
        self.assertTrue(sources, f"no generated CUDA source for all literal symbols under {cache_root}")
        source = max(sources, key=lambda path: path.stat().st_mtime_ns)
        cubins = sorted(source.parent.glob("*.cubin"), key=lambda path: path.stat().st_mtime_ns)
        self.assertTrue(cubins, f"no cubin found beside {source}")
        cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda-12.6/bin/cuobjdump"
        self.assertTrue(Path(cuobjdump).is_file(), "cuobjdump is required for the literal resource gate")
        resource_output = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubins[-1])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        sass_output = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubins[-1])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        resource_sections = re.split(r"(?m)(?=^[ \t]*Function\b)", resource_output)
        sass_sections = re.split(r"(?m)(?=^[ \t]*Function\b)", sass_output)
        for symbol in symbols:
            with self.subTest(symbol=symbol):
                resources = "\n".join(section for section in resource_sections if symbol in section)
                self.assertTrue(resources, f"resource output did not name {symbol!r}")
                registers = [int(value) for value in re.findall(r"\bREG:(\d+)", resources)]
                local = [int(value) for value in re.findall(r"\bLOCAL:(\d+)", resources)]
                stack = [int(value) for value in re.findall(r"\bSTACK:(\d+)", resources)]
                self.assertTrue(registers, resources)
                self.assertGreater(max(registers), 0, resources)
                self.assertEqual(max(local, default=0), 0, resources)
                self.assertEqual(max(stack, default=0), 0, resources)
                sass = "\n".join(section for section in sass_sections if symbol in section)
                self.assertTrue(sass, f"SASS output did not name {symbol!r}")
                self.assertEqual(len(re.findall(r"\bLDL(?:\.|\b)", sass)), 0, sass)
                self.assertEqual(len(re.findall(r"\bSTL(?:\.|\b)", sass)), 0, sass)

    def test_cuda_fixed12_route_and_solve_identity_tamper_fail_after_coordinated_rehash(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cuda:0",
        )
        workspace = hierarchy.create_workspace()
        workspace.set_rhs(np.random.default_rng(185121).normal(size=hierarchy.n_free_dofs))
        workspace.launch()
        record = workspace.record()
        attacks = (
            {"terminal_coarse_solve_kernel_version": TERMINAL_FIXED12_COARSE_SOLVE_KERNEL_VERSION + "-forged"},
            {"terminal_coarse_solve_route": TERMINAL_FIXED12_COARSE_SOLVE_ROUTE + "-forged"},
            {"terminal_coarse_scalar_size": 11},
            {
                "terminal_fusion_route": TERMINAL_MICROCYCLE_CUDA_ROUTE,
                "terminal_coarse_solve_kernel_version": TERMINAL_GENERIC_COARSE_SOLVE_KERNEL_VERSION,
                "terminal_coarse_solve_route": TERMINAL_GENERIC_COARSE_SOLVE_ROUTE,
            },
        )
        for updates in attacks:
            with self.subTest(updates=updates):
                forged = _coordinated_rehash(record, physical_updates=updates)
                with self.assertRaisesRegex(ValueError, "coarse-solve route|complete scalar schedule"):
                    forged.deterministic_record()

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

    def test_cuda_terminal_routes_are_all_buffer_bitwise_old_oracles_for_p1_p2_and_edge_values(self):
        for steps, expected_full, oracle_launch_delta in ((1, 8, 4), (2, 20, 2)):
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
                if steps == 1:
                    self.assertTrue(hierarchy.supports_terminal_microcycle)
                    self.assertEqual(hierarchy.terminal_fusion_route, TERMINAL_MICROCYCLE_FIXED12_CUDA_ROUTE)
                    self.assertTrue(hierarchy.supports_fixed12_terminal_microcycle)
                    self.assertEqual(hierarchy.terminal_fusion_launch_reduction, 6)
                    self.assertEqual(hierarchy.terminal_collective_count, 6)
                    self.assertEqual(
                        hierarchy.terminal_logical_phases.split("|"),
                        list(TERMINAL_MICROCYCLE_LOGICAL_PHASES),
                    )
                    patched_property = "supports_terminal_microcycle"
                else:
                    self.assertFalse(hierarchy.supports_terminal_microcycle)
                    self.assertEqual(hierarchy.terminal_fusion_route, TERMINAL_FUSION_CUDA_ROUTE)
                    self.assertEqual(hierarchy.terminal_fusion_launch_reduction, 2)
                    self.assertEqual(hierarchy.terminal_collective_count, 2)
                    patched_property = "supports_terminal_fusion"
                for label, rhs in (
                    ("random", random_rhs),
                    ("signed-zero", signed_zero_rhs),
                    ("nonfinite", nonfinite_rhs),
                ):
                    with self.subTest(steps=steps, rhs=label):
                        fused = hierarchy.create_workspace()
                        oracle = hierarchy.create_workspace()
                        _poison_workspace(fused)
                        _poison_workspace(oracle)
                        fused.rhs.assign(rhs)
                        oracle.rhs.assign(rhs)
                        with mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as fused_launch:
                            fused.launch()
                        with (
                            mock.patch.object(
                                WarpScalarFusedStaticMultigridHierarchy,
                                patched_property,
                                new=property(lambda _self: False),
                            ),
                            mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as oracle_launch,
                        ):
                            oracle.launch()
                        self.assertEqual(fused_launch.call_count, expected_full)
                        self.assertEqual(oracle_launch.call_count, expected_full + oracle_launch_delta)
                        self.assertEqual(_workspace_bit_patterns(fused), _workspace_bit_patterns(oracle))

    def test_cuda_fixed12_capture_replay_restores_poison_bitwise_generic(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cuda:0",
        )
        fused = hierarchy.create_workspace()
        generic = hierarchy.create_workspace()
        warmup_rhs = np.random.default_rng(181024).normal(size=(hierarchy.n_free, 3))
        fused.rhs.assign(warmup_rhs)
        generic.rhs.assign(warmup_rhs)
        fused.launch()
        with mock.patch.object(
            WarpScalarFusedStaticMultigridHierarchy,
            "supports_fixed12_terminal_microcycle",
            new=property(lambda _self: False),
        ):
            generic.launch()
        with wp.ScopedCapture(device=hierarchy.device) as fused_capture:
            fused.launch()
        with (
            mock.patch.object(
                WarpScalarFusedStaticMultigridHierarchy,
                "supports_fixed12_terminal_microcycle",
                new=property(lambda _self: False),
            ),
            wp.ScopedCapture(device=hierarchy.device) as generic_capture,
        ):
            generic.launch()

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
                _poison_workspace(generic)
                fused.rhs.assign(rhs)
                generic.rhs.assign(rhs)
                wp.capture_launch(fused_capture.graph)
                wp.capture_launch(generic_capture.graph)
                self.assertEqual(_workspace_bit_patterns(fused), _workspace_bit_patterns(generic))

    def test_cuda_terminal_preflight_rejects_alias_and_layout_before_launch(self):
        hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(
            _mixed_hierarchy(smooth_steps=1),
            device="cuda:0",
        )
        self.assertTrue(hierarchy.supports_terminal_microcycle)

        terminal = hierarchy.levels[hierarchy.terminal_level_index]
        probe = hierarchy.create_workspace()
        terminal_specs = (
            ("terminal.row_offsets", terminal.row_offsets),
            ("terminal.column_indices", terminal.column_indices),
            ("terminal.matrix_values", terminal.matrix_values),
            ("terminal.inverse_diagonal", terminal.inverse_diagonal),
            ("terminal.fine_rhs", probe.level_rhs[hierarchy.terminal_level_index]),
            ("terminal.fine_alternate", probe.level_correction_alt[hierarchy.terminal_level_index]),
            ("terminal.member_offsets", terminal.member_offsets),
            ("terminal.member_fine_nodes", terminal.member_fine_nodes),
            ("terminal.aggregate", terminal.aggregate),
            ("terminal.prolongation_blocks", terminal.prolongation_blocks),
            ("terminal.fine_residual", probe.level_residual[hierarchy.terminal_level_index]),
            ("terminal.coarse_rhs", probe.level_rhs[-1]),
            ("terminal.coarse_cholesky", hierarchy.coarse_cholesky),
            ("terminal.coarse_intermediate", probe.coarse_intermediate),
            ("terminal.coarse_solution", probe.level_correction[-1]),
            ("terminal.fine_correction", probe.level_correction[hierarchy.terminal_level_index]),
        )
        expected_names = [name for name, _array in terminal_specs]
        validated_names = []
        overlap_pairs = []
        original_validate = WarpScalarFusedStaticMultigridHierarchy._validate_exact_1d_array
        original_overlap = WarpScalarFusedStaticMultigridHierarchy._arrays_overlap.__func__

        def trace_validate(self, array, *, name, dtype, size):
            validated_names.append(name)
            return original_validate(self, array, name=name, dtype=dtype, size=size)

        def trace_overlap(cls, left, right):
            overlap_pairs.append((int(left.ptr), int(right.ptr)))
            return original_overlap(cls, left, right)

        with (
            mock.patch.object(
                WarpScalarFusedStaticMultigridHierarchy,
                "_validate_exact_1d_array",
                new=trace_validate,
            ),
            mock.patch.object(
                WarpScalarFusedStaticMultigridHierarchy,
                "_arrays_overlap",
                new=classmethod(trace_overlap),
            ),
        ):
            hierarchy._validate_terminal_fusion_preflight(probe)
        self.assertEqual(validated_names, expected_names)
        self.assertEqual(len(overlap_pairs), 120)
        self.assertEqual(len(set(overlap_pairs)), 120)

        for rejected_name in expected_names:
            with self.subTest(rejected_layout=rejected_name):
                workspace = hierarchy.create_workspace()

                def reject_one(self, array, *, name, dtype, size, rejected_name=rejected_name):
                    if name == rejected_name:
                        raise ValueError(f"{name} synthetic invalid layout")
                    return original_validate(self, array, name=name, dtype=dtype, size=size)

                with (
                    mock.patch.object(
                        WarpScalarFusedStaticMultigridHierarchy,
                        "_validate_exact_1d_array",
                        new=reject_one,
                    ),
                    mock.patch.object(scalar_fused_module.wp, "launch", wraps=wp.launch) as launch,
                ):
                    with self.assertRaisesRegex(ValueError, "synthetic invalid layout"):
                        workspace.launch()
                self.assertEqual(launch.call_count, 0)

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

    def test_real_default_stretch_cuda_oracle_uses_14_launches(self):
        hierarchy = _default_stretch_hierarchy()
        device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_hierarchy(hierarchy, device="cuda:0")
        workspace = device_hierarchy.create_workspace()
        rhs = np.random.default_rng(99118).normal(size=hierarchy.levels[0].matrix.scalar_size)
        workspace.set_rhs(rhs)
        workspace.launch()
        actual = workspace.record()
        self.assertEqual(actual.scheduled_kernel_launches, 14)
        self.assertEqual(actual.physical_work.core_kernel_launches, 13)
        self.assertEqual(actual.physical_work.terminal_fusion_kernel_launches, 1)
        self.assertEqual(actual.physical_work.terminal_fusion_launch_reduction, 6)
        self.assertEqual(actual.physical_work.terminal_level_index, 2)
        self.assertEqual(actual.physical_work.terminal_block_dim, 64)
        self.assertEqual(actual.physical_work.terminal_collective_count, 6)
        self.assertEqual(actual.physical_work.terminal_owner_thread, 0)
        self.assertEqual(actual.physical_work.terminal_microcycle_kernel_version, TERMINAL_MICROCYCLE_KERNEL_VERSION)
        self.assertEqual(
            actual.physical_work.terminal_logical_phases.split("|"),
            list(TERMINAL_MICROCYCLE_LOGICAL_PHASES),
        )
        self.assertEqual(actual.physical_work.terminal_fusion_version, TERMINAL_FUSION_VERSION)
        self.assertEqual(actual.physical_work.terminal_fusion_route, TERMINAL_MICROCYCLE_FIXED12_CUDA_ROUTE)
        self.assertEqual(
            actual.physical_work.terminal_coarse_solve_kernel_version,
            TERMINAL_FIXED12_COARSE_SOLVE_KERNEL_VERSION,
        )
        self.assertEqual(actual.physical_work.terminal_coarse_solve_route, TERMINAL_FIXED12_COARSE_SOLVE_ROUTE)
        self.assertEqual(actual.physical_work.terminal_coarse_scalar_size, 12)
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
        self.assertEqual(actual.physical_work.matrix_recurrence_phases, 12)
        self.assertEqual(actual.physical_work.jacobi_recurrence_phases, 12)
        np.testing.assert_allclose(
            actual.correction,
            apply_v_cycle(hierarchy, rhs).correction,
            rtol=5.0e-13,
            atol=5.0e-13,
        )


if __name__ == "__main__":
    unittest.main()
