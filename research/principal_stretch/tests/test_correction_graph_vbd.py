# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the direct two-V-cycle multiplicative graph solver."""

from __future__ import annotations

import copy
import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

from .. import correction_graph_vbd as graph_vbd
from ..correction_mg_vbd import _thaw_json
from ..correction_multigrid import apply_v_cycle, assemble_current_stable_nh_block_matrix
from ..solver_benchmark import build_common_problem
from ..solver_scenes import (
    build_compression_scene,
    build_extension_scene,
    build_sliver_scene,
    build_stretch_scene,
    build_twist_scene,
)


class TestDirectGraphVBD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.stretch_scene = build_stretch_scene()
        cls.sliver_scene = build_sliver_scene()
        cls.extension_scene = build_extension_scene()
        cls.twist_scene = build_twist_scene()
        cls.compression_scene = build_compression_scene()
        cls.stretch = graph_vbd.run_direct_graph_vbd(cls.stretch_scene)
        cls.sliver = graph_vbd.run_direct_graph_vbd(cls.sliver_scene)
        cls.extension = graph_vbd.run_direct_graph_vbd(cls.extension_scene)
        cls.twist = graph_vbd.run_direct_graph_vbd(cls.twist_scene)
        cls.compression = graph_vbd.run_direct_graph_vbd(cls.compression_scene)

    def test_stationary_primitive_matches_scratch_and_dense_current_operator(self):
        correction = self.stretch.quality.outer_corrections[0].correction
        operator = correction.operator
        hierarchy = correction.hierarchy
        dense = assemble_current_stable_nh_block_matrix(operator).to_dense()

        np.testing.assert_allclose(
            operator.apply_free(correction.first_correction),
            dense @ correction.first_correction,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        first = apply_v_cycle(hierarchy, correction.rhs)
        scratch_residual = correction.rhs - operator.apply_free(first.correction)
        second = apply_v_cycle(hierarchy, scratch_residual)
        scratch_direction = first.correction + second.correction
        scratch_true_residual = correction.rhs - operator.apply_free(scratch_direction)

        np.testing.assert_array_equal(correction.first_correction, first.correction)
        np.testing.assert_array_equal(correction.residual_after_first, scratch_residual)
        np.testing.assert_array_equal(correction.second_correction, second.correction)
        np.testing.assert_array_equal(correction.direction, scratch_direction)
        np.testing.assert_array_equal(correction.true_residual, scratch_true_residual)
        self.assertEqual(correction.v_cycle_work, (first.work, second.work))
        np.testing.assert_allclose(
            correction.true_residual,
            correction.rhs - dense @ correction.direction,
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_fixed_four_scene_matrix_completes_four_by_two_work(self):
        runs = (
            ("extension", self.extension),
            ("stretch", self.stretch),
            ("twist", self.twist),
            ("sliver", self.sliver),
        )
        for name, run in runs:
            with self.subTest(scene=name):
                quality = run.quality
                self.assertEqual(quality.config.coarse_node_limit, 4)
                self.assertEqual(quality.accepted_outer_correction_count, 4)
                self.assertEqual(quality.total_v_cycle_count, 8)
                self.assertTrue(quality.gate.all_outer_work_completed)
                self.assertFalse(quality.gate.fallback_used)
                self.assertTrue(quality.gate.passed)
                self.assertLess(quality.gate.versus_k4.objective_delta, 0.0)
                self.assertLess(quality.gate.versus_k4.residual_ratio, 1.0)
                self.assertLess(quality.gate.versus_k4.free_rms_ratio, 1.0)
                self.assertEqual(quality.final_metrics.max_pin_error_m, 0.0)
                self.assertEqual(quality.final_metrics.inverted_tet_fraction, 0.0)
                self.assertGreater(quality.final_metrics.determinant_min, 0.0)
                self.assertEqual(quality.vbd_k1.role, "vbd-k1")
                self.assertEqual(quality.vbd_k4.role, "vbd-k4")
                self.assertEqual(quality.vbd_k1.iterations, 1)
                self.assertEqual(quality.vbd_k4.iterations, 4)
                self.assertEqual(quality.vbd_k1.physical_state_sha256, quality.vbd_k4.physical_state_sha256)
                self.assertEqual(quality.vbd_k1.iterate_zero_sha256, quality.vbd_k4.iterate_zero_sha256)
                self.assertFalse(run.timing.deterministic_record()["performance_evidence"])
                self.assertNotIn("timing", quality.deterministic_record())

        self.assertLess(self.stretch.quality.gate.versus_k4.residual_ratio, 0.46)
        self.assertLess(self.stretch.quality.gate.versus_k4.free_rms_ratio, 0.50)
        self.assertLess(self.sliver.quality.gate.versus_k4.residual_ratio, 0.28)
        self.assertLess(self.sliver.quality.gate.versus_k4.free_rms_ratio, 0.32)
        self.assertLess(self.extension.quality.gate.versus_k4.residual_ratio, 0.002)
        self.assertLess(self.twist.quality.gate.versus_k4.residual_ratio, 0.0001)

    def test_compression_records_strict_fourth_outer_noop_without_relaxing_armijo(self):
        quality = self.compression.quality
        self.assertEqual(len(quality.outer_corrections), 4)
        self.assertEqual(quality.total_v_cycle_count, 8)
        self.assertEqual(quality.accepted_outer_correction_count, 3)
        rejected = quality.outer_corrections[-1].correction
        self.assertEqual(rejected.reason, "objective_increase")
        self.assertEqual(rejected.candidate_objective, rejected.initial_objective)
        self.assertFalse(quality.gate.all_outer_work_completed)
        self.assertTrue(quality.gate.fallback_used)
        self.assertFalse(quality.gate.passed)
        self.assertLess(quality.gate.versus_k4.objective_delta, 0.0)
        self.assertLess(quality.gate.versus_k4.residual_ratio, 0.002)
        self.assertLess(quality.gate.versus_k4.free_rms_ratio, 0.002)

    def test_each_outer_rebuilds_current_a_and_reuses_one_rest_hierarchy(self):
        quality = self.stretch.quality
        expected_start = quality.k1_metrics.position_sha256
        operator_hashes = []
        for outer_index, outer in enumerate(quality.outer_corrections):
            correction = outer.correction
            with self.subTest(outer=outer_index):
                self.assertEqual(outer.start_metrics.position_sha256, expected_start)
                self.assertEqual(
                    graph_vbd._array_digest(correction.operator.positions),
                    outer.start_metrics.position_sha256,
                )
                self.assertEqual(correction.hierarchy.content_sha256, quality.hierarchy.hierarchy_sha256)
                self.assertEqual(len(correction.v_cycle_work), 2)
                self.assertTrue(correction.exact_work_completed)
                self.assertTrue(outer.exact_work_completed)
                self.assertTrue(
                    all(work.hierarchy_sha256 == quality.hierarchy.hierarchy_sha256 for work in correction.v_cycle_work)
                )
                expected_start = outer.metrics.position_sha256
                operator_hashes.append(correction.operator_sha256)
        self.assertEqual(expected_start, quality.final_metrics.position_sha256)
        self.assertEqual(len(set(operator_hashes)), 4)

    def test_exact_pins_and_one_shot_bdf1_velocity(self):
        for scene, quality in (
            (self.stretch_scene, self.stretch.quality),
            (self.sliver_scene, self.sliver.quality),
        ):
            with self.subTest(scene=scene.name):
                for outer in quality.outer_corrections:
                    np.testing.assert_array_equal(
                        outer.correction.x[scene.pinned_indices],
                        scene.pin_targets,
                    )
                    np.testing.assert_array_equal(
                        outer.correction.operator.positions[scene.pinned_indices],
                        scene.pin_targets,
                    )
                expected_velocity = (quality.final_positions - scene.x_current) / scene.dt
                expected_velocity[scene.pinned_indices] = 0.0
                np.testing.assert_array_equal(quality.final_velocities, expected_velocity)
                np.testing.assert_array_equal(
                    quality.final_positions[scene.pinned_indices],
                    scene.pin_targets,
                )

    def test_safeguard_failure_is_exact_and_fail_closed_after_two_cycles(self):
        quality = self.stretch.quality
        problem = build_common_problem(self.stretch_scene)
        config = dataclasses.replace(quality.config, minimum_determinant=2.0)
        rejected = graph_vbd.solve_two_vcycle_stationary_correction(
            problem,
            quality.k1_positions,
            quality.hierarchy_object,
            config,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "segment_inversion")
        self.assertTrue(rejected.exact_work_completed)
        self.assertEqual(len(rejected.v_cycle_work), 2)
        self.assertLessEqual(rejected.segment_minimum_determinant, config.minimum_determinant)
        np.testing.assert_array_equal(rejected.x, quality.k1_positions)
        np.testing.assert_array_equal(
            rejected.x[self.stretch_scene.pinned_indices],
            self.stretch_scene.pin_targets,
        )

    def test_standalone_primitive_rejects_same_size_wrong_static_model(self):
        quality = self.stretch.quality
        problem = build_common_problem(self.stretch_scene)
        wrong_problem = dataclasses.replace(problem, mu=1.25 * problem.mu)
        wrong_operator = graph_vbd.MatrixFreeStableNHOperator.from_problem(
            wrong_problem,
            quality.k1_positions,
        )
        wrong_hierarchy = graph_vbd._build_hierarchy(
            wrong_operator,
            self.stretch_scene,
            quality.config,
        )
        self.assertEqual(
            wrong_hierarchy.levels[0].matrix.scalar_size,
            quality.hierarchy_object.levels[0].matrix.scalar_size,
        )
        self.assertNotEqual(
            wrong_hierarchy.static_model_sha256,
            quality.hierarchy_object.static_model_sha256,
        )
        with self.assertRaisesRegex(ValueError, "hierarchy static model"):
            graph_vbd.solve_two_vcycle_stationary_correction(
                problem,
                quality.k1_positions,
                wrong_hierarchy,
                quality.config,
            )
        with self.assertRaisesRegex(ValueError, "hierarchy static model"):
            dataclasses.replace(
                quality.outer_corrections[0].correction,
                hierarchy=wrong_hierarchy,
            )

        reversed_vertices = quality.hierarchy_object.free_vertices[::-1].copy()
        wrong_order = dataclasses.replace(quality.hierarchy_object, free_vertices=reversed_vertices)
        with self.assertRaisesRegex(ValueError, "free-vertex ordering"):
            dataclasses.replace(
                quality.outer_corrections[0].correction,
                hierarchy=wrong_order,
            )

        alternate_config = dataclasses.replace(quality.config, coarse_node_limit=16)
        alternate_hierarchy = graph_vbd._build_hierarchy(
            quality.outer_corrections[0].correction.operator,
            self.stretch_scene,
            alternate_config,
        )
        with self.assertRaisesRegex(ValueError, "hierarchy settings"):
            graph_vbd.solve_two_vcycle_stationary_correction(
                problem,
                quality.k1_positions,
                alternate_hierarchy,
                quality.config,
            )

    def test_quality_rejects_malformed_retained_hierarchy_arrays(self):
        quality = self.stretch.quality
        hierarchy = quality.hierarchy_object

        reversed_vertices = hierarchy.free_vertices[::-1].copy()
        wrong_order = dataclasses.replace(hierarchy, free_vertices=reversed_vertices)
        with self.assertRaisesRegex(ValueError, "free-vertex ordering"):
            dataclasses.replace(quality, hierarchy_object=wrong_order)

        changed_rest = hierarchy.rest_positions.copy()
        changed_rest[0, 0] += 1.0e-12
        wrong_rest = dataclasses.replace(hierarchy, rest_positions=changed_rest)
        with self.assertRaisesRegex(ValueError, "rest geometry"):
            dataclasses.replace(quality, hierarchy_object=wrong_rest)

        changed_masses = hierarchy.free_masses.copy()
        changed_masses[0] *= 1.01
        wrong_masses = dataclasses.replace(hierarchy, free_masses=changed_masses)
        with self.assertRaisesRegex(ValueError, "free masses"):
            dataclasses.replace(quality, hierarchy_object=wrong_masses)

    def test_stale_nested_hierarchy_hashes_are_recomputed_from_raw_content(self):
        quality = self.stretch.quality
        hierarchy = quality.hierarchy_object
        correction = quality.outer_corrections[0].correction
        problem = build_common_problem(self.stretch_scene)

        def frozen_copy(value):
            owned = np.array(value, dtype=value.dtype, order="C", copy=True)
            return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)

        root = hierarchy.levels[0]
        changed_values = root.matrix.values.copy()
        changed_values[0, 0, 0] += 1.0e-12
        stale_matrix = dataclasses.replace(root.matrix, values=frozen_copy(changed_values))
        stale_level = dataclasses.replace(root, matrix=stale_matrix)
        stale_matrix_hierarchy = dataclasses.replace(
            hierarchy,
            levels=(stale_level, *hierarchy.levels[1:]),
        )
        with self.assertRaisesRegex(ValueError, "matrix hash is stale"):
            graph_vbd.solve_two_vcycle_stationary_correction(
                problem,
                quality.k1_positions,
                stale_matrix_hierarchy,
                quality.config,
            )
        with self.assertRaisesRegex(ValueError, "matrix hash is stale"):
            dataclasses.replace(correction, hierarchy=stale_matrix_hierarchy)
        with self.assertRaisesRegex(ValueError, "matrix hash is stale"):
            dataclasses.replace(quality, hierarchy_object=stale_matrix_hierarchy)

        changed_factor = hierarchy.coarse_cholesky.copy()
        changed_factor[0, 0] += 1.0e-12
        stale_factor_hierarchy = dataclasses.replace(
            hierarchy,
            coarse_cholesky=frozen_copy(changed_factor),
        )
        with self.assertRaisesRegex(ValueError, "top-level hash is stale"):
            dataclasses.replace(correction, hierarchy=stale_factor_hierarchy)

    def test_coordinated_hierarchy_rehash_and_mutable_base_alias_are_fail_closed(self):
        quality = self.stretch.quality
        hierarchy = quality.hierarchy_object
        correction = quality.outer_corrections[0].correction

        def frozen_copy(value):
            owned = np.array(value, dtype=value.dtype, order="C", copy=True)
            return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)

        def level_sha256(level):
            return graph_vbd._hash_parts(
                "static-multigrid-level-v1",
                (
                    ("matrix_sha256", level.matrix.content_sha256),
                    ("node_ids", level.node_ids),
                    ("enrichment", level.enrichment),
                    ("aggregate", level.aggregate),
                    (
                        "prolongation_sha256",
                        None if level.prolongation is None else level.prolongation.content_sha256,
                    ),
                    ("smoother_sha256", None if level.smoother is None else level.smoother.content_sha256),
                ),
            )

        def hierarchy_sha256(candidate):
            return graph_vbd._hash_parts(
                "static-multigrid-hierarchy-v1",
                (
                    ("free_vertices", candidate.free_vertices),
                    ("rest_positions", candidate.rest_positions),
                    ("free_masses", candidate.free_masses),
                    ("solver_contract", candidate.solver_contract),
                    ("mode_kind", candidate.mode_kind),
                    ("target_aggregate_size", candidate.target_aggregate_size),
                    ("minimum_aggregate_size", candidate.minimum_aggregate_size),
                    ("coarse_node_limit", candidate.coarse_node_limit),
                    ("maximum_levels", candidate.maximum_levels),
                    ("pre_smooth_steps", candidate.pre_smooth_steps),
                    ("post_smooth_steps", candidate.post_smooth_steps),
                    ("smoother_safety", candidate.smoother_safety),
                    ("static_model_sha256", candidate.static_model_sha256),
                    ("coarse_cholesky", candidate.coarse_cholesky),
                    ("storage_sha256", candidate.storage.content_sha256),
                    *((f"level_{index}_sha256", level.content_sha256) for index, level in enumerate(candidate.levels)),
                ),
            )

        root = hierarchy.levels[0]
        changed_values = root.matrix.values.copy()
        changed_values[0, 0, 0] += 1.0e-12
        changed_values = frozen_copy(changed_values)
        changed_matrix_sha256 = graph_vbd._hash_parts(
            "static-block-matrix-v1",
            (
                ("block_row_count", root.matrix.block_row_count),
                ("block_size", root.matrix.block_size),
                ("row_offsets", root.matrix.row_offsets),
                ("column_indices", root.matrix.column_indices),
                ("values", changed_values),
            ),
        )
        rehashed_matrix = dataclasses.replace(
            root.matrix,
            values=changed_values,
            content_sha256=changed_matrix_sha256,
        )
        rehashed_level = dataclasses.replace(root, matrix=rehashed_matrix)
        rehashed_level = dataclasses.replace(rehashed_level, content_sha256=level_sha256(rehashed_level))
        rehashed_hierarchy = dataclasses.replace(
            hierarchy,
            levels=(rehashed_level, *hierarchy.levels[1:]),
        )
        rehashed_hierarchy = dataclasses.replace(
            rehashed_hierarchy,
            content_sha256=hierarchy_sha256(rehashed_hierarchy),
        )
        with self.assertRaisesRegex(ValueError, "deterministic A0 rebuild"):
            dataclasses.replace(correction, hierarchy=rehashed_hierarchy)
        with self.assertRaisesRegex(ValueError, "deterministic A0 rebuild"):
            dataclasses.replace(quality, hierarchy_object=rehashed_hierarchy)

        mutable_values = root.matrix.values.copy()
        readonly_alias = mutable_values.view()
        readonly_alias.setflags(write=False)
        alias_matrix = dataclasses.replace(root.matrix, values=readonly_alias)
        alias_level = dataclasses.replace(root, matrix=alias_matrix)
        alias_hierarchy = dataclasses.replace(
            hierarchy,
            levels=(alias_level, *hierarchy.levels[1:]),
        )
        canonical_correction = dataclasses.replace(correction, hierarchy=alias_hierarchy)
        canonical_quality = dataclasses.replace(quality, hierarchy_object=alias_hierarchy)
        correction_record = canonical_correction.deterministic_record()
        quality_record = canonical_quality.deterministic_record()

        mutable_values[0, 0, 0] += 1.0
        np.testing.assert_array_equal(canonical_correction.hierarchy.levels[0].matrix.values, root.matrix.values)
        np.testing.assert_array_equal(canonical_quality.hierarchy_object.levels[0].matrix.values, root.matrix.values)
        self.assertEqual(canonical_correction.deterministic_record(), correction_record)
        self.assertEqual(canonical_quality.deterministic_record(), quality_record)

    def test_content_evidence_rejects_vector_work_chain_gate_and_velocity_tampering(self):
        quality = self.stretch.quality
        outer = quality.outer_corrections[0]
        correction = outer.correction

        changed_direction = correction.direction.copy()
        changed_direction[0] += 1.0e-12
        with self.assertRaisesRegex(ValueError, "stationary direction"):
            dataclasses.replace(correction, direction=changed_direction)

        changed_residual = correction.residual_after_first.copy()
        changed_residual[0] += 1.0e-12
        with self.assertRaisesRegex(ValueError, "residual update"):
            dataclasses.replace(correction, residual_after_first=changed_residual)

        changed_work = dataclasses.replace(correction.v_cycle_work[0], result_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "first V-cycle"):
            dataclasses.replace(correction, v_cycle_work=(changed_work, correction.v_cycle_work[1]))

        with self.assertRaisesRegex(ValueError, "promotion gate"):
            dataclasses.replace(quality, gate=dataclasses.replace(quality.gate, exact_pins=False))

        changed_velocity = quality.final_velocities.copy()
        changed_velocity[self.stretch_scene.free_indices[0], 0] += 1.0e-12
        with self.assertRaisesRegex(ValueError, "BDF1"):
            dataclasses.replace(quality, final_velocities=changed_velocity)

        with self.assertRaises(ValueError):
            correction.direction[0] = 0.0

    def test_reference_policy_rejects_coordinated_rehash_tampering(self):
        quality = self.stretch.quality
        rebuilt = dataclasses.replace(quality, reference=quality.reference)
        self.assertEqual(rebuilt.quality_sha256, quality.quality_sha256)
        forged_provenance = dataclasses.replace(quality.reference, provenance="forged")
        with self.assertRaisesRegex(ValueError, "fresh dense Newton"):
            dataclasses.replace(quality, reference=forged_provenance)
        original = _thaw_json(quality.reference.source_record)

        def set_verification_flag(record):
            record["verification_converged"] = False

        def set_verification_reason(record):
            record["verification_reason"] = "step"

        def set_native_flag(record):
            record["native_converged"] = False

        def set_native_reason(record):
            record["native_reason"] = "line_search"

        def set_final_objective(record):
            record["final_objective"] += 1.0e-8

        def set_final_gradient(record):
            record["final_gradient_norm"] += 1.0e-8

        def set_final_residual(record):
            record["final_relative_residual"] += 1.0e-8

        def set_newton_config(record):
            record["config"]["max_iterations"] = 49

        def exceed_verification_gate(record):
            record["verification_displacement_relative"] = 1.0e-9

        def exceed_alternate_gate(record):
            record["alternate_start_displacement_relative"] = 1.0e-6

        attacks = (
            ("native flag", set_native_flag, "native Newton"),
            ("native reason", set_native_reason, "native Newton"),
            ("verification flag", set_verification_flag, "independent verification"),
            ("verification reason", set_verification_reason, "independent verification"),
            ("final objective", set_final_objective, "final_objective"),
            ("final gradient", set_final_gradient, "final_gradient_norm"),
            ("final residual", set_final_residual, "final_relative_residual"),
            ("Newton config", set_newton_config, "Newton config max_iterations"),
            ("verification gate", exceed_verification_gate, "verification_displacement_relative"),
            ("alternate gate", exceed_alternate_gate, "alternate_start_displacement_relative"),
        )
        for name, mutate, message in attacks:
            with self.subTest(attack=name):
                forged_record = copy.deepcopy(original)
                mutate(forged_record)
                forged_reference = dataclasses.replace(
                    quality.reference,
                    source_record=forged_record,
                    source_record_sha256=graph_vbd._canonical_digest(forged_record),
                )
                with self.assertRaisesRegex(ValueError, message):
                    dataclasses.replace(quality, reference=forged_reference)

    def test_independent_reference_acceptance_rejects_inversion(self):
        quality = self.stretch.quality
        forged_metrics = dataclasses.replace(
            quality.reference_metrics,
            determinant_min=-1.0,
            inverted_tet_fraction=1.0,
        )
        with (
            mock.patch.object(
                graph_vbd,
                "evaluate_common_state",
                side_effect=(forged_metrics, quality.k1_metrics, quality.k4_metrics, quality.final_metrics),
            ),
            self.assertRaisesRegex(ValueError, "inversion-free"),
        ):
            dataclasses.replace(quality, reference_metrics=forged_metrics)

    def test_mutable_vcycle_work_input_is_canonicalized_before_quality_hashing(self):
        quality = self.stretch.quality
        original_outer = quality.outer_corrections[0]
        mutable_work = list(original_outer.correction.v_cycle_work)
        canonical_correction = dataclasses.replace(original_outer.correction, v_cycle_work=mutable_work)
        self.assertIsInstance(canonical_correction.v_cycle_work, tuple)
        canonical_outer = dataclasses.replace(original_outer, correction=canonical_correction)
        canonical_quality = dataclasses.replace(
            quality,
            outer_corrections=(canonical_outer, *quality.outer_corrections[1:]),
        )
        self.assertEqual(canonical_quality.quality_sha256, quality.quality_sha256)
        frozen_record = canonical_quality.deterministic_record()

        mutable_work.clear()
        self.assertEqual(len(canonical_correction.v_cycle_work), 2)
        self.assertEqual(canonical_quality.quality_sha256, quality.quality_sha256)
        self.assertEqual(canonical_quality.deterministic_record(), frozen_record)

    def test_k4_velocity_and_scene_color_counts_reject_coordinated_rehashes(self):
        quality = self.stretch.quality

        def rehashed_evidence(evidence, **changes):
            payload = evidence.deterministic_record()
            payload.pop("evidence_sha256")
            payload.update(changes)
            return dataclasses.replace(
                evidence,
                **changes,
                evidence_sha256=graph_vbd._canonical_digest(payload),
            )

        forged_velocity = rehashed_evidence(quality.vbd_k4, velocity_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "K4 evidence does not bind K4 velocity"):
            dataclasses.replace(quality, vbd_k4=forged_velocity)

        impossible_count = int(self.stretch_scene.color_group_offsets.size - 1) + 100
        forged_k1_count = rehashed_evidence(quality.vbd_k1, color_group_count=impossible_count)
        with self.assertRaisesRegex(ValueError, "K1 color-group count"):
            dataclasses.replace(quality, vbd_k1=forged_k1_count)
        forged_k4_count = rehashed_evidence(quality.vbd_k4, color_group_count=impossible_count)
        with self.assertRaisesRegex(ValueError, "K4 color-group count"):
            dataclasses.replace(quality, vbd_k4=forged_k4_count)

        with self.assertRaises(ValueError):
            quality.k4_velocities[0, 0] = 0.0

    def test_frozen_candidate_rejects_work_schedule_changes(self):
        with self.assertRaisesRegex(ValueError, "four outer corrections and two V-cycles"):
            dataclasses.replace(graph_vbd.DirectGraphVBDConfig(), outer_corrections=3).validate()
        with self.assertRaisesRegex(ValueError, "four outer corrections and two V-cycles"):
            dataclasses.replace(graph_vbd.DirectGraphVBDConfig(), stationary_v_cycles=3).validate()
        with self.assertRaisesRegex(ValueError, "alpha=1"):
            dataclasses.replace(graph_vbd.DirectGraphVBDConfig(), alpha=0.5).validate()


if __name__ == "__main__":
    unittest.main()
