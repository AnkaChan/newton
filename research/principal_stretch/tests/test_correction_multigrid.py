# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic residual-correction multigrid ceiling."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from .. import correction_multigrid as correction_multigrid_module
from ..correction_gpu import MatrixFreeStableNHOperator
from ..correction_multigrid import (
    SPECTRAL_FREE_CONTRACT,
    StaticBlockMatrix,
    apply_v_cycle,
    assemble_current_stable_nh_block_matrix,
    assemble_stable_nh_rest_block_matrix,
    build_block_jacobi,
    build_stable_nh_rest_multigrid,
    build_static_multigrid,
    rigid_enrichment,
    solve_pcg,
    stable_nh_static_model_digest,
)
from ..solver_benchmark import build_common_problem
from ..solver_scenes import build_stretch_scene


def _tetra_chain(group_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a sparse SPD vector-spring chain with tetrahedral aggregates."""
    if group_count < 2:
        raise ValueError("group_count must be at least two")
    local = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.30, 0.0, 0.0),
            (0.0, 0.27, 0.0),
            (0.0, 0.0, 0.24),
        ),
        dtype=np.float64,
    )
    rest = np.concatenate([local + np.array((0.65 * group, 0.0, 0.0)) for group in range(group_count)])
    node_count = rest.shape[0]
    dense = np.zeros((3 * node_count, 3 * node_count), dtype=np.float64)
    edges: list[tuple[int, int, float]] = []
    for group in range(group_count):
        nodes = [4 * group + offset for offset in range(4)]
        for local_i in range(4):
            for local_j in range(local_i + 1, 4):
                edges.append((nodes[local_i], nodes[local_j], 8.0))
        if group + 1 < group_count:
            edges.append((nodes[1], 4 * (group + 1), 0.8))
            edges.append((nodes[2], 4 * (group + 1) + 2, 0.4))

    identity = np.eye(3, dtype=np.float64)
    for first, second, weight in edges:
        direction = rest[second] - rest[first]
        direction /= np.linalg.norm(direction)
        longitudinal = np.outer(direction, direction)
        stiffness = weight * (longitudinal + 0.025 * (identity - longitudinal))
        first_slice = slice(3 * first, 3 * first + 3)
        second_slice = slice(3 * second, 3 * second + 3)
        dense[first_slice, first_slice] += stiffness
        dense[second_slice, second_slice] += stiffness
        dense[first_slice, second_slice] -= stiffness
        dense[second_slice, first_slice] -= stiffness
    dense += 2.0e-3 * np.eye(3 * node_count, dtype=np.float64)
    return dense, rest, np.arange(node_count, dtype=np.int64)


def _sparse_isotropic_tetra_chain(group_count: int) -> tuple[StaticBlockMatrix, np.ndarray, np.ndarray]:
    """Build a large local graph directly from O(V+E) block entries."""
    local = np.array(
        ((0.0, 0.0, 0.0), (0.30, 0.0, 0.0), (0.0, 0.27, 0.0), (0.0, 0.0, 0.24)),
        dtype=np.float64,
    )
    rest = np.concatenate([local + np.array((0.65 * group, 0.0, 0.0)) for group in range(group_count)])
    node_count = rest.shape[0]
    identity = np.eye(3, dtype=np.float64)
    entries: list[tuple[int, int, np.ndarray]] = [(node, node, 0.01 * identity) for node in range(node_count)]
    edges: list[tuple[int, int, float]] = []
    for group in range(group_count):
        nodes = [4 * group + offset for offset in range(4)]
        for first in range(4):
            for second in range(first + 1, 4):
                edges.append((nodes[first], nodes[second], 8.0))
        if group + 1 < group_count:
            edges.append((nodes[1], 4 * (group + 1), 0.5))
    for first, second, weight in edges:
        block = weight * identity
        entries.extend(
            (
                (first, first, block),
                (second, second, block),
                (first, second, -block),
                (second, first, -block),
            )
        )
    return (
        StaticBlockMatrix.from_block_entries(node_count, entries),
        rest,
        np.arange(node_count, dtype=np.int64),
    )


def _build(group_count: int, *, mode_kind: str = "rigid"):
    dense, rest, free = _tetra_chain(group_count)
    matrix = StaticBlockMatrix.from_dense(dense)
    hierarchy = build_static_multigrid(
        matrix,
        rest,
        free,
        mode_kind=mode_kind,
        target_aggregate_size=4,
        minimum_aggregate_size=3,
        coarse_node_limit=2,
    )
    return matrix, hierarchy, rest


def _block_neighbors(matrix: StaticBlockMatrix) -> tuple[set[int], ...]:
    neighbors = []
    for row in range(matrix.block_row_count):
        columns = matrix.column_indices[int(matrix.row_offsets[row]) : int(matrix.row_offsets[row + 1])]
        neighbors.append({int(column) for column in columns if int(column) != row})
    return tuple(neighbors)


class TestCorrectionMultigrid(unittest.TestCase):
    def test_repeat_build_and_work_hashes_are_identical_and_immutable(self):
        matrix_a, hierarchy_a, rest = _build(8)
        matrix_b = StaticBlockMatrix.from_dense(matrix_a.to_dense())
        hierarchy_b = build_static_multigrid(
            matrix_b,
            rest,
            np.arange(rest.shape[0], dtype=np.int64),
            coarse_node_limit=2,
        )

        self.assertEqual(matrix_a.content_sha256, matrix_b.content_sha256)
        self.assertEqual(hierarchy_a.content_sha256, hierarchy_b.content_sha256)
        self.assertEqual(hierarchy_a.storage.content_sha256, hierarchy_b.storage.content_sha256)
        self.assertEqual(
            [level.content_sha256 for level in hierarchy_a.levels],
            [level.content_sha256 for level in hierarchy_b.levels],
        )
        deeper_limit = build_static_multigrid(
            matrix_a,
            rest,
            np.arange(rest.shape[0], dtype=np.int64),
            coarse_node_limit=2,
            maximum_levels=9,
        )
        safer_smoothing = build_static_multigrid(
            matrix_a,
            rest,
            np.arange(rest.shape[0], dtype=np.int64),
            coarse_node_limit=2,
            smoother_safety=0.85,
        )
        self.assertNotEqual(hierarchy_a.content_sha256, deeper_limit.content_sha256)
        self.assertNotEqual(hierarchy_a.content_sha256, safer_smoothing.content_sha256)
        self.assertEqual(deeper_limit.maximum_levels, 9)
        self.assertEqual(safer_smoothing.smoother_safety, 0.85)
        rhs = np.linspace(-1.0, 1.0, matrix_a.scalar_size)
        first = apply_v_cycle(hierarchy_a, rhs)
        second = apply_v_cycle(hierarchy_b, rhs)
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.work.content_sha256, second.work.content_sha256)
        np.testing.assert_array_equal(first.correction, second.correction)
        with self.assertRaises(ValueError):
            first.correction[0] = 0.0
        with self.assertRaises(ValueError):
            first.correction.setflags(write=True)
        with self.assertRaises(ValueError):
            hierarchy_a.levels[0].matrix.values.setflags(write=True)

    def test_sparse_block_entry_assembly_matches_dense_without_dense_setup(self):
        dense, _rest, _free = _tetra_chain(3)
        node_count = dense.shape[0] // 3
        entries = []
        for row in range(node_count):
            for column in range(node_count):
                block = dense[3 * row : 3 * row + 3, 3 * column : 3 * column + 3]
                if row == column or np.any(block != 0.0):
                    # Split every block into two repeated local contributions
                    # to cover deterministic element/operator accumulation.
                    entries.append((row, column, 0.5 * block))
                    entries.append((row, column, 0.5 * block))
        sparse = StaticBlockMatrix.from_block_entries(node_count, entries)
        expected = StaticBlockMatrix.from_dense(dense)
        self.assertEqual(sparse.content_sha256, expected.content_sha256)
        np.testing.assert_array_equal(sparse.to_dense(), dense)

        with self.assertRaisesRegex(ValueError, "missing transpose"):
            StaticBlockMatrix.from_block_entries(
                2,
                (
                    (0, 0, np.eye(3)),
                    (1, 1, np.eye(3)),
                    (0, 1, -0.1 * np.eye(3)),
                ),
            )

    def test_aggregates_are_connected_large_enough_and_noncollinear(self):
        _matrix, hierarchy, rest = _build(10)
        for level_index, level in enumerate(hierarchy.levels[:-1]):
            neighbors = _block_neighbors(level.matrix)
            aggregate = level.aggregate
            for aggregate_id in range(int(aggregate.max()) + 1):
                members = {int(node) for node in np.flatnonzero(aggregate == aggregate_id)}
                self.assertGreaterEqual(len(members), 3)
                seen = {min(members)}
                stack = list(seen)
                while stack:
                    node = stack.pop()
                    for neighbor in neighbors[node]:
                        if neighbor in members and neighbor not in seen:
                            seen.add(neighbor)
                            stack.append(neighbor)
                self.assertEqual(seen, members, f"level {level_index} aggregate {aggregate_id}")
                if level_index == 0:
                    centered = rest[list(members)] - rest[list(members)].mean(axis=0)
                    self.assertGreaterEqual(np.linalg.matrix_rank(centered, tol=1.0e-12), 2)
            self.assertLess(level.prolongation.coarse_scalar_size, level.matrix.scalar_size)

    def test_rigid_enrichment_is_reproduced_with_rank_completed_local_bases(self):
        _matrix, hierarchy, _rest = _build(8)
        for level, coarse in zip(hierarchy.levels[:-1], hierarchy.levels[1:], strict=True):
            prolongation = level.prolongation
            np.testing.assert_allclose(
                prolongation.prolong(coarse.enrichment),
                level.enrichment,
                rtol=0.0,
                atol=3.0e-12,
            )
            dense = prolongation.to_dense()
            np.testing.assert_allclose(dense.T @ dense, np.eye(dense.shape[1]), rtol=0.0, atol=3.0e-12)

        first = hierarchy.levels[0]
        expected = rigid_enrichment(hierarchy.rest_positions)
        np.testing.assert_array_equal(first.enrichment, expected)
        # Every local basis has six columns even when the restricted empirical
        # enrichment is rank deficient; coordinate completion makes P full rank.
        self.assertEqual(first.prolongation.blocks.shape[2], 6)

    def test_mass_centering_and_length_scaling_are_coordinate_robust(self):
        matrix, _hierarchy, rest = _build(8)
        free = np.arange(rest.shape[0], dtype=np.int64)
        masses = np.linspace(0.5, 2.0, rest.shape[0])
        base = build_static_multigrid(matrix, rest, free, free_masses=masses, coarse_node_limit=2)
        transformed = build_static_multigrid(
            matrix,
            37.0 * rest + np.array((1.0e5, -2.0e5, 3.0e5)),
            free,
            free_masses=masses,
            coarse_node_limit=2,
        )
        self.assertNotEqual(base.content_sha256, transformed.content_sha256)
        for base_level, transformed_level in zip(base.levels[:-1], transformed.levels[:-1], strict=True):
            np.testing.assert_array_equal(base_level.aggregate, transformed_level.aggregate)
            np.testing.assert_allclose(
                base_level.prolongation.blocks,
                transformed_level.prolongation.blocks,
                rtol=0.0,
                atol=2.0e-11,
            )

        modes = rigid_enrichment(rest, masses).reshape(rest.shape[0], 3, 6)
        weighted_rotation_mean = np.sum(masses[:, None, None] * modes[:, :, 3:], axis=0)
        np.testing.assert_allclose(weighted_rotation_mean, 0.0, rtol=0.0, atol=2.0e-14)

    def test_galerkin_products_are_equal_symmetric_and_positive_definite(self):
        _matrix, hierarchy, _rest = _build(7)
        for fine, coarse in zip(hierarchy.levels[:-1], hierarchy.levels[1:], strict=True):
            prolongation = fine.prolongation.to_dense()
            expected = prolongation.T @ fine.matrix.to_dense() @ prolongation
            actual = coarse.matrix.to_dense()
            np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-12)
            np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=0.0)
            self.assertGreater(float(np.linalg.eigvalsh(actual)[0]), 0.0)

    def test_v_cycle_is_symmetric_positive_and_records_exact_work(self):
        matrix, hierarchy, _rest = _build(4)
        identity = np.eye(matrix.scalar_size, dtype=np.float64)
        result = apply_v_cycle(hierarchy, identity)
        inverse = result.correction
        np.testing.assert_allclose(inverse, inverse.T, rtol=0.0, atol=3.0e-11)
        self.assertGreater(float(np.linalg.eigvalsh(0.5 * (inverse + inverse.T))[0]), 0.0)
        self.assertEqual(result.work.level_visits, (1,) * len(hierarchy.levels))
        self.assertEqual(result.work.rhs_count, matrix.scalar_size)
        self.assertGreater(result.work.matrix_block_products, 0)
        self.assertGreater(result.work.smoother_block_solves, 0)
        self.assertGreater(result.work.restriction_block_products, 0)
        self.assertGreater(result.work.prolongation_block_products, 0)
        self.assertEqual(result.work.coarsest_factor_solves, matrix.scalar_size)
        self.assertEqual(result.work.hierarchy_sha256, hierarchy.content_sha256)

    def test_smoother_uses_independent_frobenius_row_bound(self):
        _matrix, hierarchy, _rest = _build(6)
        for level in hierarchy.levels[:-1]:
            matrix = level.matrix
            diagonal = matrix.diagonal_blocks()
            inverse_lower = []
            for block in diagonal:
                lower = np.linalg.cholesky(block)
                inverse_lower.append(np.linalg.solve(lower, np.eye(matrix.block_size)))
            row_bounds = []
            for row in range(matrix.block_row_count):
                bound = 0.0
                for entry in range(int(matrix.row_offsets[row]), int(matrix.row_offsets[row + 1])):
                    column = int(matrix.column_indices[entry])
                    normalized = inverse_lower[row] @ matrix.values[entry] @ inverse_lower[column].T
                    bound += float(np.linalg.norm(normalized, ord="fro"))
                row_bounds.append(bound)
            expected = max(row_bounds)
            self.assertAlmostEqual(level.smoother.normalized_spectral_upper_bound, expected, places=14)
            self.assertLessEqual(level.smoother.omega * expected, 0.9 + 2.0e-15)

    def test_rigid_rayleigh_is_nonzero_and_improves_rotation_solve_ablation(self):
        matrix, rigid, rest = _build(12, mode_kind="rigid")
        _same_matrix, translation, _same_rest = _build(12, mode_kind="translation")
        modes = rigid_enrichment(rest)
        translation_rayleigh = []
        rotation_rayleigh = []
        for column in range(6):
            mode = modes[:, column]
            rayleigh = float(mode @ matrix.matmul(mode) / (mode @ mode))
            (translation_rayleigh if column < 3 else rotation_rayleigh).append(rayleigh)
        self.assertGreater(min(rotation_rayleigh), 0.0)
        self.assertGreater(min(rotation_rayleigh), max(translation_rayleigh))

        rng = np.random.default_rng(921)
        random_rayleigh = []
        for _ in range(12):
            vector = rng.standard_normal(matrix.scalar_size)
            random_rayleigh.append(float(vector @ matrix.matmul(vector) / (vector @ vector)))
        self.assertLess(max(rotation_rayleigh), np.median(random_rayleigh))

        target = 0.7 * modes[:, 3] - 0.4 * modes[:, 4] + 0.2 * modes[:, 5]
        # Add a deterministic high-frequency component so PCG cannot terminate
        # merely by recognizing a single coarse vector.
        target += 0.03 * np.sin(np.arange(matrix.scalar_size, dtype=np.float64) * 1.7)
        rhs = matrix.matmul(target)
        rigid_result = solve_pcg(matrix, rhs, hierarchy=rigid, relative_tolerance=1.0e-9)
        translation_result = solve_pcg(matrix, rhs, hierarchy=translation, relative_tolerance=1.0e-9)
        jacobi_result = solve_pcg(
            matrix,
            rhs,
            jacobi=build_block_jacobi(matrix),
            relative_tolerance=1.0e-9,
        )
        self.assertTrue(rigid_result.converged)
        self.assertTrue(translation_result.converged)
        self.assertTrue(jacobi_result.converged)
        np.testing.assert_allclose(rigid_result.solution, target, rtol=2.0e-7, atol=2.0e-7)
        self.assertLess(rigid_result.iteration_count, translation_result.iteration_count)
        self.assertLess(rigid_result.iteration_count, jacobi_result.iteration_count)
        self.assertLessEqual(rigid_result.true_relative_residual, 1.0e-9)
        self.assertEqual(rigid_result.operator_applications, rigid_result.iteration_count + 1)
        self.assertEqual(rigid_result.preconditioner_applications, rigid_result.iteration_count)
        self.assertGreater(rigid_result.inner_products, 0)
        self.assertGreater(rigid_result.vector_updates, 0)
        tighter_record = solve_pcg(matrix, rhs, hierarchy=rigid, relative_tolerance=5.0e-10)
        self.assertNotEqual(rigid_result.content_sha256, tighter_record.content_sha256)
        self.assertEqual(rigid_result.rhs_sha256, tighter_record.rhs_sha256)

    def test_real_stable_nh_operator_sparse_blocks_and_static_hierarchy(self):
        scene = build_stretch_scene(dimensions=(3, 2, 1))
        problem = build_common_problem(scene)
        current_positions = scene.x_current.copy()
        current_positions[scene.pinned_indices] = scene.pin_targets
        current_operator = MatrixFreeStableNHOperator.from_problem(problem, current_positions)

        current_matrix = assemble_current_stable_nh_block_matrix(current_operator)
        generator = np.random.default_rng(119)
        for _ in range(3):
            direction = generator.normal(size=current_operator.n_free_dofs)
            np.testing.assert_allclose(
                current_matrix.matmul(direction),
                current_operator.apply_free(direction),
                rtol=5.0e-13,
                atol=5.0e-11,
            )

        rest_matrix = assemble_stable_nh_rest_block_matrix(current_operator, scene.rest_q)
        hierarchy = build_stable_nh_rest_multigrid(
            current_operator,
            scene.rest_q,
            coarse_node_limit=2,
        )
        self.assertEqual(hierarchy.solver_contract, SPECTRAL_FREE_CONTRACT)
        self.assertEqual(hierarchy.levels[0].matrix.content_sha256, rest_matrix.content_sha256)
        self.assertEqual(
            hierarchy.static_model_sha256,
            stable_nh_static_model_digest(current_operator, scene.rest_q),
        )

        other_positions = current_positions.copy()
        other_positions[current_operator.free] += generator.normal(size=(current_operator.free.size, 3)) * 0.01
        other_operator = MatrixFreeStableNHOperator.from_problem(problem, other_positions)
        other_hierarchy = build_stable_nh_rest_multigrid(other_operator, scene.rest_q, coarse_node_limit=2)
        self.assertNotEqual(
            assemble_current_stable_nh_block_matrix(current_operator).content_sha256,
            assemble_current_stable_nh_block_matrix(other_operator).content_sha256,
        )
        self.assertEqual(hierarchy.content_sha256, other_hierarchy.content_sha256)
        self.assertEqual(hierarchy.static_model_sha256, other_hierarchy.static_model_sha256)

        inverse = apply_v_cycle(hierarchy, np.eye(rest_matrix.scalar_size)).correction
        np.testing.assert_allclose(inverse, inverse.T, rtol=0.0, atol=2.0e-11)
        self.assertGreater(float(np.linalg.eigvalsh(0.5 * (inverse + inverse.T))[0]), 0.0)
        target = np.sin(np.arange(rest_matrix.scalar_size, dtype=np.float64) * 0.31)
        rhs = rest_matrix.matmul(target)
        result = solve_pcg(rest_matrix, rhs, hierarchy=hierarchy, relative_tolerance=1.0e-9)
        self.assertTrue(result.converged)
        self.assertLess(result.true_relative_residual, 1.0e-9)
        self.assertLess(result.residual_norms[-1], 1.0e-8 * result.residual_norms[0])

        invalid_rest = scene.rest_q.copy()
        invalid_rest[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "F=I"):
            assemble_stable_nh_rest_block_matrix(current_operator, invalid_rest)

    def test_sparse_storage_scales_with_graph_and_stores_no_dense_matrices(self):
        _small_matrix, small, _small_rest = _build(8)
        _large_matrix, large, _large_rest = _build(16)
        for hierarchy in (small, large):
            storage = hierarchy.storage
            graph_size = storage.fine_node_count + storage.fine_undirected_edge_count
            self.assertEqual(storage.dense_matrix_scalar_count_excluding_coarse_factor, 0)
            self.assertLess(storage.matrix_block_count, 6 * graph_size)
            self.assertLess(storage.prolongation_block_count, 2 * storage.fine_node_count)
            self.assertLess(storage.total_scalar_count, 300 * graph_size)
            retained_arrays = [
                hierarchy.coarse_cholesky,
                hierarchy.free_vertices,
                hierarchy.rest_positions,
                hierarchy.free_masses,
            ]
            for level in hierarchy.levels:
                retained_arrays.extend(
                    (
                        level.matrix.row_offsets,
                        level.matrix.column_indices,
                        level.matrix.values,
                        level.node_ids,
                        level.enrichment,
                    )
                )
                if level.aggregate is not None:
                    retained_arrays.append(level.aggregate)
                if level.prolongation is not None:
                    retained_arrays.extend((level.prolongation.aggregate, level.prolongation.blocks))
                if level.smoother is not None:
                    retained_arrays.append(level.smoother.inverse_diagonal)
            self.assertEqual(storage.total_bytes, sum(array.nbytes for array in retained_arrays))
            self.assertEqual(storage.total_bytes, 8 * storage.total_scalar_count)
            self.assertGreater(storage.factor_scalar_count, 0)
        self.assertLess(large.storage.total_scalar_count, 2.6 * small.storage.total_scalar_count)

    def test_large_sparse_setup_keeps_linear_graph_structure(self):
        matrix, rest, free = _sparse_isotropic_tetra_chain(128)
        hierarchy = build_static_multigrid(matrix, rest, free, coarse_node_limit=2)
        self.assertEqual(
            [level.matrix.block_row_count for level in hierarchy.levels],
            [512, 128, 32, 8, 2],
        )
        graph_size = hierarchy.storage.fine_node_count + hierarchy.storage.fine_undirected_edge_count
        self.assertLess(hierarchy.storage.matrix_block_count, 6 * graph_size)
        self.assertLess(hierarchy.storage.prolongation_block_count, 2 * hierarchy.storage.fine_node_count)

    def test_invalid_merges_do_not_recheck_an_already_valid_target(self):
        group_count = 64
        node_count = 3 * group_count
        identity = np.eye(3, dtype=np.float64)
        rest = np.zeros((node_count, 3), dtype=np.float64)
        rest[:3] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        entries: list[tuple[int, int, np.ndarray]] = [(node, node, 0.01 * identity) for node in range(node_count)]
        edges: list[tuple[int, int, float]] = []
        for group in range(group_count):
            first = 3 * group
            if group:
                rest[first : first + 3, 0] = (4.0 * group, 4.0 * group + 1.0, 4.0 * group + 2.0)
                edges.append((0, first, 0.1))
            edges.extend(((first, first + 1, 8.0), (first, first + 2, 8.0), (first + 1, first + 2, 8.0)))
        for first, second, weight in edges:
            block = weight * identity
            entries.extend(
                (
                    (first, first, block),
                    (second, second, block),
                    (first, second, -block),
                    (second, first, -block),
                )
            )
        matrix = StaticBlockMatrix.from_block_entries(node_count, entries)
        original = correction_multigrid_module._is_noncollinear
        with mock.patch.object(correction_multigrid_module, "_is_noncollinear", wraps=original) as geometry_check:
            aggregate = correction_multigrid_module._connected_aggregate(
                matrix,
                np.arange(node_count, dtype=np.int64),
                target_size=3,
                minimum_size=3,
                first_level_rest=rest,
            )
        self.assertEqual(geometry_check.call_count, group_count)
        np.testing.assert_array_equal(aggregate, np.zeros(node_count, dtype=np.int64))

    def test_pcg_scaled_norms_handle_huge_rhs_and_fail_closed_on_nonfinite(self):
        matrix, hierarchy, _rest = _build(4)
        direction = np.sin(np.arange(matrix.scalar_size, dtype=np.float64) * 0.37)
        ordinary_rhs = matrix.matmul(direction)
        huge_rhs = ordinary_rhs * 1.0e250
        result = solve_pcg(matrix, huge_rhs, hierarchy=hierarchy, relative_tolerance=1.0e-9)
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.solution).all())
        self.assertTrue(np.isfinite(result.residual_norms).all())
        self.assertTrue(np.isfinite(result.true_residual_norm))
        self.assertLessEqual(result.true_relative_residual, 1.0e-9)

        invalid_rhs = ordinary_rhs.copy()
        invalid_rhs[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            solve_pcg(matrix, invalid_rhs, hierarchy=hierarchy)
        with self.assertRaisesRegex(FloatingPointError, "not representable"):
            solve_pcg(matrix, np.full(matrix.scalar_size, np.finfo(np.float64).max), hierarchy=hierarchy)
        for invalid_iterations in (True, 1.5):
            with (
                self.subTest(maximum_iterations=invalid_iterations),
                self.assertRaisesRegex(ValueError, "exact positive integer"),
            ):
                solve_pcg(matrix, ordinary_rhs, hierarchy=hierarchy, maximum_iterations=invalid_iterations)

    def test_degenerate_rigid_aggregate_and_nonreducing_space_fail_closed(self):
        node_count = 6
        rest = np.column_stack((np.arange(node_count, dtype=np.float64), np.zeros((node_count, 2))))
        scalar_laplacian = np.diag(np.full(node_count, 2.0))
        scalar_laplacian += np.diag(np.full(node_count - 1, -0.9), 1)
        scalar_laplacian += np.diag(np.full(node_count - 1, -0.9), -1)
        dense = np.kron(scalar_laplacian, np.eye(3))
        with self.assertRaisesRegex(ValueError, "non-collinear"):
            build_static_multigrid(
                dense,
                rest,
                np.arange(node_count, dtype=np.int64),
                target_aggregate_size=3,
                minimum_aggregate_size=3,
                coarse_node_limit=1,
            )

        with self.assertRaisesRegex(ValueError, "minimum_aggregate_size"):
            build_static_multigrid(
                dense,
                rest,
                np.arange(node_count, dtype=np.int64),
                target_aggregate_size=3,
                minimum_aggregate_size=2,
                coarse_node_limit=1,
            )

        matrix, _hierarchy, valid_rest = _build(16)
        with self.assertRaisesRegex(ValueError, "maximum_levels=.*exhausted"):
            build_static_multigrid(
                matrix,
                valid_rest,
                np.arange(valid_rest.shape[0], dtype=np.int64),
                coarse_node_limit=1,
                maximum_levels=2,
            )


if __name__ == "__main__":
    unittest.main()
