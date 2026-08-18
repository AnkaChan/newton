# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for captured direct multiplicative-graph VBD."""

from __future__ import annotations

import ctypes
import dataclasses
import inspect
import json
import os
import unittest
import weakref
from types import SimpleNamespace

import numpy as np
import warp as wp

from research.principal_stretch.captured_graph_vbd import (
    CONTRACT_ID,
    OUTER_CORRECTIONS,
    V_CYCLES_PER_OUTER,
    CapturedDirectGraphVBD,
    CapturedGraphVBDEndpoint,
    CapturedGraphVBDTiming,
    _hash_parts,
)
from research.principal_stretch.captured_vbd_baseline import CONTRACT_ID as VBD_BASELINE_CONTRACT_ID
from research.principal_stretch.correction_gpu import (
    MatrixFreeStableNHOperator,
    minimum_determinant_on_segment,
)
from research.principal_stretch.correction_gpu_warp import WarpMatrixFreeWorkspace
from research.principal_stretch.correction_graph_vbd import DirectGraphVBDConfig
from research.principal_stretch.correction_multigrid import apply_v_cycle
from research.principal_stretch.correction_multigrid_warp_scalar_fused import (
    WarpScalarFusedStaticMultigridHierarchy,
    WarpScalarFusedVCyclePhysicalWork,
    WarpScalarFusedVCycleRecord,
    WarpScalarFusedVCycleWorkspace,
)
from research.principal_stretch.solver_benchmark import (
    build_common_problem,
    build_structured_cantilever_scene,
    evaluate_common_state,
)
from research.principal_stretch.solver_scenes import build_stretch_scene


def _tiny_scene():
    return build_structured_cantilever_scene(
        dimensions=(2, 2, 1),
        dt=1.0 / 16.0,
        gravity=(0.1, -0.2, -2.0),
        total_tip_force=(4.0, -3.0, -6.0),
        initial_velocity=(0.03, -0.02, 0.01),
        name="captured-direct-graph-vbd-tiny",
    )


def _assert_float32_device_reconstruction(
    testcase: unittest.TestCase,
    solver: CapturedDirectGraphVBD,
    endpoint,
) -> None:
    """Independently replay current-A/two-B work and fp32 publication on CPU."""
    current = endpoint.outer_start_positions[0].copy()
    active = True
    for outer_index in range(OUTER_CORRECTIONS):
        work = endpoint.outer_work[outer_index]
        np.testing.assert_array_equal(endpoint.outer_start_positions[outer_index], current)
        operator = MatrixFreeStableNHOperator.from_problem(solver.problem, current)
        rhs = -operator.gradient_free().reshape(-1, 3)
        if not active:
            rhs.fill(0.0)
        first = apply_v_cycle(solver.hierarchy, rhs.reshape(-1)).correction.reshape(-1, 3)
        product = operator.apply_free(first.reshape(-1)).reshape(-1, 3)
        residual = rhs - product
        second = apply_v_cycle(solver.hierarchy, residual.reshape(-1)).correction.reshape(-1, 3)
        direction = first + second

        np.testing.assert_allclose(work.rhs, rhs, rtol=2.0e-12, atol=2.0e-13)
        np.testing.assert_allclose(work.first_correction, first, rtol=3.0e-12, atol=3.0e-13)
        np.testing.assert_allclose(work.operator_product_after_first, product, rtol=4.0e-12, atol=4.0e-13)
        np.testing.assert_allclose(work.residual_after_first, residual, rtol=4.0e-12, atol=4.0e-13)
        np.testing.assert_allclose(work.second_correction, second, rtol=4.0e-12, atol=4.0e-13)
        np.testing.assert_allclose(work.direction, direction, rtol=5.0e-12, atol=5.0e-13)

        candidate = current.copy()
        if active:
            candidate[operator.free] = (
                (current[operator.free] + direction.reshape(-1, 3)).astype(np.float32).astype(np.float64)
            )
        np.testing.assert_array_equal(endpoint.outer_candidate_positions[outer_index], candidate)
        if not active:
            testcase.assertEqual(endpoint.reasons[outer_index], "masked-after-rejection")
            continue

        candidate_operator = MatrixFreeStableNHOperator.from_problem(solver.problem, candidate)
        actual_step = candidate[operator.free] - current[operator.free]
        derivative = float(np.vdot(operator.gradient_free(), actual_step.reshape(-1)))
        segment = minimum_determinant_on_segment(operator, candidate_operator).determinant
        accepted = bool(
            derivative < 0.0
            and candidate_operator.objective() < operator.objective()
            and candidate_operator.objective() <= operator.objective() + solver.config.armijo * derivative
            and candidate_operator.minimum_determinant > solver.config.minimum_determinant
            and segment > solver.config.minimum_determinant
        )
        testcase.assertEqual(endpoint.accepted[outer_index], accepted)
        testcase.assertAlmostEqual(endpoint.initial_objectives[outer_index], operator.objective(), delta=2.0e-14)
        testcase.assertAlmostEqual(
            endpoint.candidate_objectives[outer_index], candidate_operator.objective(), delta=2.0e-14
        )
        testcase.assertAlmostEqual(endpoint.directional_derivatives[outer_index], derivative, delta=5.0e-12)
        testcase.assertAlmostEqual(endpoint.segment_minimum_determinants[outer_index], segment, delta=2.0e-11)
        if accepted:
            current = candidate
        else:
            active = False

    np.testing.assert_array_equal(endpoint.positions, current.astype(np.float32).astype(np.float64))


class TestCapturedDirectGraphVBD(unittest.TestCase):
    def test_source_is_public_research_only_and_contains_no_krylov_solver(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        source = inspect.getsource(module)
        self.assertNotIn("newton._src", source)
        self.assertNotIn("from newton import _src", source)
        self.assertNotIn("WarpFixedPCG", source)
        self.assertIn("b - A(x) B b", source)
        self.assertIn("end_objective < start_objective", source)
        self.assertIn("minimum_segment <= minimum_determinant", source)
        self.assertIn("WarpScalarFusedStaticMultigridHierarchy", source)
        self.assertNotIn("level_product", source)

    def test_contract_rejects_cpu_and_nondefault_coarse_bound(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            CapturedDirectGraphVBD(_tiny_scene(), device="cpu")
        with self.assertRaisesRegex(ValueError, "coarse_node_limit=4"):
            CapturedDirectGraphVBD(
                _tiny_scene(),
                device="cpu",
                config=DirectGraphVBDConfig(coarse_node_limit=1),
            )

    def test_diagnostic_timing_is_balanced_and_cannot_claim_performance(self):
        common = {
            "graph_seconds": (1.0e-3, 1.1e-3),
            "k4_seconds": (2.0e-4, 2.1e-4),
            "random_seed": 17,
            "device": "cuda:0",
            "contract_id": CONTRACT_ID,
            "scene_sha256": "0" * 64,
            "objective_instance_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "static_hierarchy_sha256": "3" * 64,
            "persistent_device_sha256": "4" * 64,
            "graph_identity_sha256": "5" * 64,
            "k4_graph_identity_sha256": "6" * 64,
            "comparator_contract_id": VBD_BASELINE_CONTRACT_ID,
        }
        with self.assertRaisesRegex(ValueError, "equal AB and BA"):
            CapturedGraphVBDTiming(pair_orders=("AB", "AB"), warmup_replays=1, **common)
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            CapturedGraphVBDTiming(
                pair_orders=("AB", "BA"),
                warmup_replays=1,
                performance_evidence=True,
                **common,
            )

    def test_public_endpoint_constructor_cannot_mint_unvalidated_evidence(self):
        fake_positions = tuple(np.full((1, 3), float(index), dtype=np.float64) for index in range(4))
        with self.assertRaisesRegex(TypeError, "_validation_context"):
            CapturedGraphVBDEndpoint(
                scene_sha256="0" * 64,
                objective_instance_sha256="1" * 64,
                static_hierarchy_sha256="2" * 64,
                config_sha256="3" * 64,
                k1_endpoint_sha256="4" * 64,
                k1_position_sha256="5" * 64,
                k1_velocity_sha256="6" * 64,
                k1_pristine_state_sha256="7" * 64,
                persistent_device_sha256="8" * 64,
                graph_identity_sha256="9" * 64,
                armijo=1.0e-4,
                minimum_determinant=0.0,
                free_vertices=np.array([0]),
                positions=np.full((1, 3), 4.0),
                velocities=np.zeros((1, 3)),
                accepted=(True,) * 4,
                reasons=("accepted",) * 4,
                initial_objectives=(1.0,) * 4,
                candidate_objectives=(0.5,) * 4,
                directional_derivatives=(-0.1,) * 4,
                segment_minimum_determinants=(1.0,) * 4,
                outer_start_positions=fake_positions,
                outer_candidate_positions=tuple(value + 1.0 for value in fake_positions),
                outer_work=(),
                graph_replay=True,
            )


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestCapturedDirectGraphVBDTinyCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.scene = _tiny_scene()
        cls.solver = CapturedDirectGraphVBD(cls.scene, device="cuda:0")
        cls.solver.capture_graphs(warmup_replays=1)
        cls.endpoint = cls.solver.run(graph_replay=True)

    @staticmethod
    def _owner_boundaries(solver: CapturedDirectGraphVBD):
        return (
            ("run", lambda: solver.run(graph_replay=True)),
            ("run_k4", lambda: solver.run_k4(graph_replay=True)),
            ("timing", lambda: solver.benchmark_paired(pair_count=2, warmup_replays=1)),
            ("serialization", solver.deterministic_record),
            ("recapture", lambda: solver.capture_graphs(warmup_replays=1)),
            ("poison", lambda: solver.poison(seed=4817)),
        )

    def test_float32_device_semantics_match_independent_cpu_reconstruction(self):
        _assert_float32_device_reconstruction(self, self.solver, self.endpoint)

    def test_exact_two_cycle_work_and_launch_count_are_retained(self):
        endpoint = self.endpoint
        cycle_launches = self.solver.device_hierarchy.scheduled_kernel_launches
        expected_linear = 7 + 2 * cycle_launches
        expected_total = 2 + OUTER_CORRECTIONS * (expected_linear + 8)
        self.assertEqual(endpoint.total_v_cycle_count, OUTER_CORRECTIONS * V_CYCLES_PER_OUTER)
        self.assertEqual(endpoint.correction_kernel_launches, expected_total)
        self.assertEqual(self.solver.correction_kernel_launches, expected_total)
        self.assertTrue(endpoint.exact_work_completed)
        for outer_index, work in enumerate(endpoint.outer_work):
            with self.subTest(outer=outer_index):
                self.assertEqual(work.outer_index, outer_index)
                self.assertEqual(work.linear_kernel_launches, expected_linear)
                self.assertTrue(work.exact_work_completed)
                self.assertEqual(len(work.v_cycles), 2)
                for record in work.v_cycles:
                    self.assertIs(type(record), WarpScalarFusedVCycleRecord)
                    self.assertIs(type(record.physical_work), WarpScalarFusedVCyclePhysicalWork)
                    self.assertEqual(record.scheduled_kernel_launches, cycle_launches)
                    self.assertEqual(record.work.hierarchy_sha256, endpoint.static_hierarchy_sha256)
                    self.assertEqual(record.work.rhs_count, 1)
                    self.assertEqual(record.work.coarsest_factor_solves, 1)
                    physical = record.physical_work
                    self.assertEqual(
                        physical.matrix_block_products_executed + physical.matrix_block_products_elided_zero_start,
                        record.work.matrix_block_products,
                    )
                    self.assertEqual(physical.scheduled_kernel_launches, cycle_launches)
                    self.assertEqual(physical.schedule_sha256, self.solver.device_hierarchy.schedule_sha256)
        schedule = self.solver.deterministic_record()
        self.assertEqual(schedule["contract_id"], CONTRACT_ID)
        self.assertEqual(schedule["krylov_iterations"], 0)
        self.assertEqual(schedule["v_cycles"], 8)
        self.assertEqual(schedule["correction_kernel_launches_excluding_public_k1"], expected_total)
        self.assertEqual(schedule["workspace_owner_identity_sha256"], self.solver._workspace_owner_identity_sha256())
        self.assertFalse(schedule["performance_evidence"])
        json.dumps(schedule, allow_nan=False)
        json.dumps(endpoint.deterministic_record(), allow_nan=False)

    def test_changed_poison_replay_restores_endpoint_and_all_linear_evidence(self):
        expected = self.endpoint
        for replay_index, seed in enumerate((1701, 99831, 42)):
            with self.subTest(replay=replay_index):
                self.solver.poison(seed=seed)
                actual = self.solver.run(graph_replay=True)
                self.assertEqual(actual.endpoint_sha256, expected.endpoint_sha256)
                np.testing.assert_array_equal(actual.positions, expected.positions)
                np.testing.assert_array_equal(actual.velocities, expected.velocities)
                self.assertEqual(actual.accepted, expected.accepted)
                self.assertEqual(actual.reasons, expected.reasons)
                for actual_work, expected_work in zip(actual.outer_work, expected.outer_work, strict=True):
                    self.assertEqual(actual_work.content_sha256, expected_work.content_sha256)
                    np.testing.assert_array_equal(actual_work.rhs, expected_work.rhs)
                    np.testing.assert_array_equal(actual_work.first_correction, expected_work.first_correction)
                    np.testing.assert_array_equal(actual_work.residual_after_first, expected_work.residual_after_first)
                    np.testing.assert_array_equal(actual_work.second_correction, expected_work.second_correction)
                    np.testing.assert_array_equal(actual_work.direction, expected_work.direction)

    def test_coordinated_rehash_cannot_forge_candidate_or_negative_vcycle_work(self):
        candidates = list(self.endpoint.outer_candidate_positions)
        candidates[0] = self.endpoint.outer_start_positions[0]
        with self.assertRaisesRegex(ValueError, "float32-publishable"):
            dataclasses.replace(self.endpoint, outer_candidate_positions=tuple(candidates))

        outer = self.endpoint.outer_work[0]
        record = outer.v_cycles[0]
        bad_work = dataclasses.replace(record.work, restriction_block_products=-7)
        work_sha256 = _hash_parts(
            "v-cycle-work-record-v1",
            (
                ("hierarchy_sha256", bad_work.hierarchy_sha256),
                ("rhs_sha256", bad_work.rhs_sha256),
                ("result_sha256", bad_work.result_sha256),
                ("rhs_count", bad_work.rhs_count),
                ("level_visits", np.asarray(bad_work.level_visits, dtype=np.int64)),
                ("matrix_block_products", bad_work.matrix_block_products),
                ("smoother_block_solves", bad_work.smoother_block_solves),
                ("restriction_block_products", bad_work.restriction_block_products),
                ("prolongation_block_products", bad_work.prolongation_block_products),
                ("coarsest_factor_solves", bad_work.coarsest_factor_solves),
            ),
        )
        bad_work = dataclasses.replace(bad_work, content_sha256=work_sha256)
        record_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-result-v1",
            (
                ("device_snapshot_sha256", record.device_snapshot_sha256),
                ("static_device_content_sha256", record.static_device_content_sha256),
                ("schedule_sha256", record.schedule_sha256),
                ("work_sha256", work_sha256),
                ("physical_work_sha256", record.physical_work.content_sha256),
                ("scheduled_kernel_launches", record.scheduled_kernel_launches),
                ("capture_replay", record.capture_replay),
            ),
        )
        bad_record = dataclasses.replace(record, work=bad_work, content_sha256=record_sha256)
        with self.assertRaisesRegex(ValueError, "canonical fixed work|non-negative"):
            dataclasses.replace(outer, v_cycles=(bad_record, outer.v_cycles[1]))

        physical = record.physical_work
        bad_executed = physical.matrix_block_products_executed + 1
        bad_elided = physical.matrix_block_products_elided_zero_start - 1
        physical_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-physical-work-v1",
            (
                ("hierarchy_sha256", physical.hierarchy_sha256),
                ("schedule_sha256", physical.schedule_sha256),
                ("rhs_sha256", physical.rhs_sha256),
                ("result_sha256", physical.result_sha256),
                ("matrix_block_products_executed", bad_executed),
                ("matrix_block_products_elided_zero_start", bad_elided),
                ("zero_start_block_solves", physical.zero_start_block_solves),
                ("out_of_place_jacobi_block_solves", physical.out_of_place_jacobi_block_solves),
                ("matrix_kernel_launches", physical.matrix_kernel_launches),
                ("jacobi_kernel_launches", physical.jacobi_kernel_launches),
                ("scheduled_kernel_launches", physical.scheduled_kernel_launches),
            ),
        )
        bad_physical = dataclasses.replace(
            physical,
            matrix_block_products_executed=bad_executed,
            matrix_block_products_elided_zero_start=bad_elided,
            content_sha256=physical_sha256,
        )
        physical_record_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-result-v1",
            (
                ("device_snapshot_sha256", record.device_snapshot_sha256),
                ("static_device_content_sha256", record.static_device_content_sha256),
                ("schedule_sha256", record.schedule_sha256),
                ("work_sha256", record.work.content_sha256),
                ("physical_work_sha256", physical_sha256),
                ("scheduled_kernel_launches", record.scheduled_kernel_launches),
                ("capture_replay", record.capture_replay),
            ),
        )
        bad_physical_record = dataclasses.replace(
            record,
            physical_work=bad_physical,
            content_sha256=physical_record_sha256,
        )
        with self.assertRaisesRegex(ValueError, "physical matrix_block_products_executed"):
            dataclasses.replace(outer, v_cycles=(bad_physical_record, outer.v_cycles[1]))

        for field_name, value in (("research_only", 1), ("performance_evidence", 0)):
            with self.subTest(policy_field=field_name):
                bad_policy_record = dataclasses.replace(record, **{field_name: value})
                with self.assertRaisesRegex(ValueError, "invalid V-cycle policy provenance"):
                    dataclasses.replace(outer, v_cycles=(bad_policy_record, outer.v_cycles[1]))

        provenance_forgeries = (
            ({"scene_sha256": "a" * 64}, "canonical retained scene"),
            ({"objective_instance_sha256": "b" * 64}, "canonical retained scene"),
            ({"config_sha256": "c" * 64}, "exact captured configuration"),
            ({"k1_endpoint_sha256": "d" * 64}, "K1 hashes"),
            ({"persistent_device_sha256": "e" * 64}, "persistent device identity"),
            ({"graph_identity_sha256": "f" * 64}, "graph identity"),
            ({"free_vertices": self.endpoint.free_vertices[::-1]}, "canonical problem ordering"),
        )
        for replacement, message in provenance_forgeries:
            with self.subTest(field=next(iter(replacement))):
                with self.assertRaisesRegex(ValueError, message):
                    dataclasses.replace(self.endpoint, **replacement)

        cloned_context = dataclasses.replace(
            self.endpoint._validation_context,
            persistent_device_sha256="a" * 64,
            graph_identity_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "exact live solver-issued object"):
            tuple(
                dataclasses.replace(
                    work,
                    persistent_device_sha256="a" * 64,
                    _validation_context=cloned_context,
                )
                for work in self.endpoint.outer_work
            )

        context = self.endpoint._validation_context
        original_schedule = context.v_cycle_schedule_sha256
        original_static = context.v_cycle_static_device_content_sha256
        original_snapshot = context.v_cycle_device_snapshot_sha256
        original_launches = context.v_cycle_kernel_launches
        original_capture_binding = context.capture_binding
        try:
            object.__setattr__(context, "v_cycle_schedule_sha256", "c" * 64)
            with self.assertRaisesRegex(ValueError, "scalar-fused schedule identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_schedule_sha256", original_schedule)
            object.__setattr__(context, "v_cycle_static_device_content_sha256", "d" * 64)
            with self.assertRaisesRegex(ValueError, "scalar-fused static content identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_static_device_content_sha256", original_static)
            object.__setattr__(context, "v_cycle_device_snapshot_sha256", "e" * 64)
            with self.assertRaisesRegex(ValueError, "scalar-fused device snapshot identity"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_device_snapshot_sha256", original_snapshot)
            object.__setattr__(context, "v_cycle_kernel_launches", original_launches + 1)
            with self.assertRaisesRegex(ValueError, "V-cycle launch count"):
                dataclasses.replace(self.endpoint)
            object.__setattr__(context, "v_cycle_kernel_launches", original_launches)
            object.__setattr__(context, "capture_binding", None)
            with self.assertRaisesRegex(ValueError, "fields do not match their solver-issued receipt"):
                dataclasses.replace(self.endpoint)
        finally:
            object.__setattr__(context, "v_cycle_schedule_sha256", original_schedule)
            object.__setattr__(context, "v_cycle_static_device_content_sha256", original_static)
            object.__setattr__(context, "v_cycle_device_snapshot_sha256", original_snapshot)
            object.__setattr__(context, "v_cycle_kernel_launches", original_launches)
            object.__setattr__(context, "capture_binding", original_capture_binding)

    def test_pristine_k1_reset_source_tamper_fails_before_execution(self):
        source = self.solver.baseline.pristine_input.particle_qd
        pristine = np.asarray(source.numpy(), dtype=np.float32)
        try:
            source.assign(np.full_like(pristine, 2.0))
            with self.assertRaisesRegex(RuntimeError, "pristine input state"):
                self.solver.run(graph_replay=True)
        finally:
            source.assign(pristine)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_static_device_hierarchy_tamper_fails_before_execution(self):
        values = self.solver.device_hierarchy.levels[0].matrix_values
        pristine = np.asarray(values.numpy(), dtype=np.float64)
        try:
            values.assign(np.zeros_like(pristine))
            with self.assertRaisesRegex(RuntimeError, "hierarchy.level_0.matrix_values"):
                self.solver.run(graph_replay=True)
        finally:
            values.assign(pristine)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_scalar_fused_owner_and_identity_labels_fail_closed(self):
        self.assertIs(type(self.solver.device_hierarchy), WarpScalarFusedStaticMultigridHierarchy)
        self.assertIs(self.solver.device_hierarchy.source_hierarchy, self.solver.source_device_hierarchy)
        fields = (
            "_schedule_sha256",
            "_static_device_content_sha256",
            "_device_snapshot_sha256",
        )
        for field in fields:
            with self.subTest(field=field):
                original = getattr(self.solver.device_hierarchy, field)
                try:
                    setattr(self.solver.device_hierarchy, field, "a" * 64)
                    with self.assertRaisesRegex(RuntimeError, "workspace identity|scalar-fused hierarchy"):
                        self.solver.deterministic_record()
                finally:
                    setattr(self.solver.device_hierarchy, field, original)

        original_source_snapshot = self.solver.source_device_hierarchy.device_snapshot_sha256
        try:
            self.solver.source_device_hierarchy.device_snapshot_sha256 = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "hierarchy metadata|scalar-fused hierarchy"):
                self.solver.deterministic_record()
        finally:
            self.solver.source_device_hierarchy.device_snapshot_sha256 = original_source_snapshot

        original_wrapper = self.solver.device_hierarchy
        replacement = WarpScalarFusedStaticMultigridHierarchy.from_device_hierarchy(self.solver.source_device_hierarchy)
        try:
            self.solver.device_hierarchy = replacement
            with self.assertRaisesRegex(RuntimeError, "hierarchy owner object|workspace owner object"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.device_hierarchy = original_wrapper

    def test_scalar_fused_alt_buffers_are_pointer_bound_and_named(self):
        signatures = dict(self.solver._persistent_array_signatures())
        self.assertTrue(any("level_correction_alt" in name for name in signatures))
        self.assertFalse(any("level_product" in name for name in signatures))
        cycle = self.solver.workspaces[0].first_cycle
        self.assertIs(type(cycle), WarpScalarFusedVCycleWorkspace)
        original = cycle.level_correction_alt
        replacement = wp.empty(
            original[0].shape,
            dtype=wp.float64,
            device=self.solver.device,
        )
        try:
            cycle.level_correction_alt = (replacement, *original[1:])
            with self.assertRaisesRegex(
                RuntimeError,
                "workspace persistent array pointers|allocation or pointer|level_correction_alt container",
            ):
                self.solver.deterministic_record()
        finally:
            cycle.level_correction_alt = original

    def test_workspace_container_swap_fails_at_every_public_boundary(self):
        original = self.solver.workspaces
        for name, operation in self._owner_boundaries(self.solver):
            with self.subTest(boundary=name):
                try:
                    self.solver.workspaces = list(original)
                    with self.assertRaisesRegex(RuntimeError, "workspace tuple container"):
                        operation()
                finally:
                    self.solver.workspaces = original

    def test_outer_position_container_attacks_fail_at_every_public_boundary(self):
        class StatefulPositionSequence:
            def __init__(self, canonical, redirected):
                self.canonical = canonical
                self.redirected = redirected
                self.reads = 0

            def __len__(self):
                self.reads += 1
                return len(self.canonical)

            def __iter__(self):
                self.reads += 1
                return iter(self.canonical if self.reads < 4 else self.redirected)

            def __getitem__(self, index):
                self.reads += 1
                values = self.canonical if self.reads < 4 else self.redirected
                return values[index]

        for field_name in ("outer_start_positions", "outer_candidate_positions"):
            original = getattr(self.solver, field_name)
            alternate = wp.empty(original[0].shape, dtype=wp.vec3d, device=self.solver.device)
            redirected = (alternate, *original[1:])
            attacks = (
                ("list", lambda values=original: list(values)),
                ("shallow-tuple", lambda values=original: tuple(iter(values))),
                (
                    "stateful",
                    lambda canonical=original, alternate_values=redirected: StatefulPositionSequence(
                        canonical,
                        alternate_values,
                    ),
                ),
            )
            for attack_name, make_attack in attacks:
                for boundary_name, operation in self._owner_boundaries(self.solver):
                    with self.subTest(field=field_name, attack=attack_name, boundary=boundary_name):
                        attack = make_attack()
                        try:
                            setattr(self.solver, field_name, attack)
                            with self.assertRaisesRegex(RuntimeError, f"{field_name} tuple container or order"):
                                operation()
                            if isinstance(attack, StatefulPositionSequence):
                                self.assertEqual(attack.reads, 0)
                        finally:
                            setattr(self.solver, field_name, original)

    def test_particle_color_group_container_attacks_fail_for_k1_k4_and_all_boundaries(self):
        class StatefulColorGroups(list):
            def __init__(self, canonical, redirected):
                super().__init__(canonical)
                self.canonical = canonical
                self.redirected = redirected
                self.reads = 0

            def __len__(self):
                self.reads += 1
                return super().__len__()

            def __iter__(self):
                self.reads += 1
                return iter(self.canonical if self.reads < 4 else self.redirected)

            def __getitem__(self, index):
                self.reads += 1
                values = self.canonical if self.reads < 4 else self.redirected
                return values[index]

        model = self.solver.baseline.model
        original = model.particle_color_groups
        alternate = wp.array(
            np.asarray(original[0].numpy(), dtype=np.int32)[::-1].copy(),
            dtype=wp.int32,
            device=self.solver.device,
        )
        redirected = [alternate, *original[1:]]
        attacks = (
            ("shallow-list", lambda: list(original)),
            ("stateful-list", lambda: StatefulColorGroups(original, redirected)),
        )
        for attack_name, make_attack in attacks:
            for boundary_name, operation in self._owner_boundaries(self.solver):
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    attack = make_attack()
                    try:
                        model.particle_color_groups = attack
                        with self.assertRaisesRegex(RuntimeError, "particle color-group list or order"):
                            operation()
                        if isinstance(attack, StatefulColorGroups):
                            self.assertEqual(attack.reads, 0)
                    finally:
                        model.particle_color_groups = original

    def test_coordinated_hierarchy_level_container_attacks_fail_at_all_boundaries(self):
        class StatefulLevels:
            def __init__(self, canonical, redirected):
                self.canonical = canonical
                self.redirected = redirected
                self.reads = 0

            def __len__(self):
                self.reads += 1
                return len(self.canonical)

            def __iter__(self):
                self.reads += 1
                return iter(self.canonical if self.reads < 5 else self.redirected)

            def __getitem__(self, index):
                self.reads += 1
                values = self.canonical if self.reads < 5 else self.redirected
                return values[index]

        source = self.solver.source_device_hierarchy
        scalar = self.solver.device_hierarchy
        original = source.levels
        alternate_values = wp.empty(
            original[0].matrix_values.shape,
            dtype=wp.float64,
            device=self.solver.device,
        )
        redirected_level = dataclasses.replace(original[0], matrix_values=alternate_values)
        redirected = (redirected_level, *original[1:])
        attacks = (
            ("list", lambda: list(original)),
            ("shallow-tuple", lambda: tuple(iter(original))),
            ("stateful", lambda: StatefulLevels(original, redirected)),
        )
        for attack_name, make_attack in attacks:
            for boundary_name, operation in self._owner_boundaries(self.solver):
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    attack = make_attack()
                    try:
                        source.levels = attack
                        scalar._levels = attack
                        with self.assertRaisesRegex(RuntimeError, "hierarchy levels tuple container"):
                            operation()
                        if isinstance(attack, StatefulLevels):
                            self.assertEqual(attack.reads, 0)
                    finally:
                        source.levels = original
                        scalar._levels = original

    def test_each_hierarchy_level_array_owner_is_construction_bound(self):
        fields = (
            "row_offsets",
            "column_indices",
            "matrix_values",
            "inverse_diagonal",
            "aggregate",
            "prolongation_blocks",
            "member_offsets",
            "member_fine_nodes",
        )
        for level_index, level in enumerate(self.solver.source_device_hierarchy.levels):
            for field_name in fields:
                original = getattr(level, field_name)
                if original is None:
                    continue
                with self.subTest(level=level_index, field=field_name):
                    replacement = wp.empty(
                        original.shape,
                        dtype=original.dtype,
                        device=self.solver.device,
                    )
                    try:
                        object.__setattr__(level, field_name, replacement)
                        with self.assertRaisesRegex(RuntimeError, f"level {level_index} {field_name} owner"):
                            self.solver.deterministic_record()
                    finally:
                        object.__setattr__(level, field_name, original)

    def test_outer_proxy_and_shallow_clone_cannot_redirect_recapture(self):
        original_container = self.solver.workspaces
        original = original_container[0]
        alternate = wp.empty(
            original.first_correction.shape,
            dtype=wp.vec3d,
            device=self.solver.device,
        )

        class StatefulOuterProxy:
            def __init__(self, canonical, redirected):
                self.canonical = canonical
                self.redirected = redirected
                self.first_correction_reads = 0

            def __getattr__(self, name):
                if name == "first_correction":
                    self.first_correction_reads += 1
                    if self.first_correction_reads > 2:
                        return self.redirected
                return getattr(self.canonical, name)

        proxy = StatefulOuterProxy(original, alternate)
        graph = self.solver.graph
        k4_graph = self.solver.k4_graph
        try:
            self.solver.workspaces = (proxy, *original_container[1:])
            with self.assertRaisesRegex(RuntimeError, "workspace tuple container"):
                self.solver.capture_graphs(warmup_replays=1)
            self.assertEqual(proxy.first_correction_reads, 0)
            self.assertIs(self.solver.graph, graph)
            self.assertIs(self.solver.k4_graph, k4_graph)
        finally:
            self.solver.workspaces = original_container

        shallow_outer = dataclasses.replace(original)
        try:
            self.solver.workspaces = (shallow_outer, *original_container[1:])
            with self.assertRaisesRegex(RuntimeError, "workspace tuple container"):
                self.solver.deterministic_record()
        finally:
            self.solver.workspaces = original_container

    def test_stateful_registry_dispatch_cannot_redirect_any_public_boundary(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        solver = CapturedDirectGraphVBD(self.scene, device="cuda:0")
        solver.capture_graphs(warmup_replays=1)
        expected_positions = solver.run(graph_replay=True).positions
        original_lookup = module._lookup_workspace_owners
        canonical = original_lookup(solver)
        alternate = wp.empty(
            canonical.outer[0].first_correction.shape,
            dtype=wp.vec3d,
            device=solver.device,
        )
        sentinel = np.full((alternate.shape[0], 3), 734.25, dtype=np.float64)
        alternate.assign(sentinel)
        registry = next(
            cell.cell_contents
            for cell in original_lookup.__closure__ or ()
            if isinstance(cell.cell_contents, weakref.WeakKeyDictionary)
        )

        for dispatch_kind in ("module-global", "closure-registry-get"):
            for boundary_name, operation in self._owner_boundaries(solver):
                with self.subTest(dispatch=dispatch_kind, boundary=boundary_name):
                    canonical = original_lookup(solver)
                    forged_outer = canonical.outer[0]._replace(first_correction=alternate)
                    forged = canonical._replace(outer=(forged_outer, *canonical.outer[1:]))
                    calls = 0

                    def stateful_lookup(
                        owner,
                        default=None,
                        canonical_binding=canonical,
                        forged_binding=forged,
                    ):
                        nonlocal calls
                        calls += 1
                        if owner is not solver:
                            return original_lookup(owner)
                        return canonical_binding if calls == 1 else forged_binding

                    alternate.assign(sentinel)
                    try:
                        if dispatch_kind == "module-global":
                            module._lookup_workspace_owners = stateful_lookup
                        else:
                            registry.get = stateful_lookup
                        operation()
                        self.assertEqual(calls, 1)
                        np.testing.assert_array_equal(np.asarray(alternate.numpy(), dtype=np.float64), sentinel)
                    finally:
                        if dispatch_kind == "module-global":
                            module._lookup_workspace_owners = original_lookup
                        elif "get" in vars(registry):
                            del registry.get
                    if boundary_name == "recapture":
                        published = original_lookup(solver)
                        self.assertIsNot(published, canonical)
                        self.assertIs(published.persistent, canonical.persistent)
                        self.assertIs(published.claims, canonical.claims)
                        self.assertEqual(published.capture.generation, canonical.capture.generation + 1)
        np.testing.assert_array_equal(solver.run(graph_replay=True).positions, expected_positions)

    def test_cycle_owner_and_all_tuple_container_clones_fail_closed(self):
        cycles = tuple(
            cycle for workspace in self.solver.workspaces for cycle in (workspace.first_cycle, workspace.second_cycle)
        )
        for cycle_index, cycle in enumerate(cycles):
            for field_name in (
                "level_rhs",
                "level_correction",
                "level_correction_alt",
                "level_residual",
                "_persistent_arrays",
                "_persistent_pointers",
            ):
                with self.subTest(cycle=cycle_index, container=field_name):
                    original = getattr(cycle, field_name)
                    try:
                        setattr(cycle, field_name, list(original))
                        with self.assertRaisesRegex(RuntimeError, f"{field_name} container"):
                            self.solver.deterministic_record()
                    finally:
                        setattr(cycle, field_name, original)

        original_cycle = self.solver.workspaces[0].first_cycle
        shallow_cycle = object.__new__(WarpScalarFusedVCycleWorkspace)
        for slot in WarpScalarFusedVCycleWorkspace.__slots__:
            setattr(shallow_cycle, slot, getattr(original_cycle, slot))
        try:
            self.solver.workspaces[0].first_cycle = shallow_cycle
            with self.assertRaisesRegex(RuntimeError, "scalar-fused workspace owner object"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.workspaces[0].first_cycle = original_cycle

        foreign_cycle = SimpleNamespace(
            **{slot: getattr(original_cycle, slot) for slot in WarpScalarFusedVCycleWorkspace.__slots__}
        )
        try:
            self.solver.workspaces[0].first_cycle = foreign_cycle
            with self.assertRaisesRegex(RuntimeError, "scalar-fused workspace owner object"):
                self.solver.run_k4(graph_replay=True)
        finally:
            self.solver.workspaces[0].first_cycle = original_cycle

    def test_cycle_arrays_and_operator_apply_owner_clones_fail_closed(self):
        workspace = self.solver.workspaces[0]
        cycle = workspace.first_cycle
        for field_name in ("rhs", "correction", "coarse_intermediate"):
            with self.subTest(cycle_array=field_name):
                original = getattr(cycle, field_name)
                replacement = wp.empty(
                    original.shape,
                    dtype=original.dtype,
                    device=self.solver.device,
                )
                try:
                    setattr(cycle, field_name, replacement)
                    with self.assertRaisesRegex(RuntimeError, f"{field_name} owner"):
                        self.solver.deterministic_record()
                finally:
                    setattr(cycle, field_name, original)

        original_apply = workspace.operator_apply
        shallow_apply = object.__new__(WarpMatrixFreeWorkspace)
        shallow_apply.__dict__.update(original_apply.__dict__)
        try:
            workspace.operator_apply = shallow_apply
            with self.assertRaisesRegex(RuntimeError, "matrix-free apply workspace owner"):
                self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
        finally:
            workspace.operator_apply = original_apply

        foreign_apply = SimpleNamespace(**original_apply.__dict__)
        try:
            workspace.operator_apply = foreign_apply
            with self.assertRaisesRegex(RuntimeError, "matrix-free apply workspace owner"):
                self.solver.capture_graphs(warmup_replays=1)
        finally:
            workspace.operator_apply = original_apply

    def test_operator_and_endpoint_source_tamper_fail_before_execution(self):
        cases = (
            ("mass", self.solver.operator.mass, "operator.mass"),
            ("canonical positions", self.solver.canonical_positions, "canonical_positions"),
            ("x current", self.solver.x_current, "x_current"),
        )
        for name, array, message in cases:
            with self.subTest(source=name):
                pristine = np.asarray(array.numpy()).copy()
                try:
                    array.assign(pristine + 1.0)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.solver.run(graph_replay=True)
                finally:
                    array.assign(pristine)

    def test_every_direct_correction_array_owner_is_construction_bound(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver).direct
        for field_name in binding._fields:
            if field_name in ("outer_start_positions", "outer_candidate_positions"):
                continue
            with self.subTest(field=field_name):
                original = getattr(self.solver, field_name)
                replacement = wp.empty(original.shape, dtype=original.dtype, device=self.solver.device)
                try:
                    setattr(self.solver, field_name, replacement)
                    with self.assertRaisesRegex(RuntimeError, f"direct correction {field_name} owner"):
                        self.solver.deterministic_record()
                finally:
                    setattr(self.solver, field_name, original)

    def test_public_model_and_config_tamper_fail_before_execution(self):
        model_mass = self.solver.baseline.model.particle_mass
        pristine_mass = np.asarray(model_mass.numpy()).copy()
        try:
            model_mass.assign(pristine_mass + 1.0)
            with self.assertRaisesRegex(RuntimeError, "public static model"):
                self.solver.run(graph_replay=True)
        finally:
            model_mass.assign(pristine_mass)

        pristine_config = self.solver.config
        try:
            self.solver.config = dataclasses.replace(pristine_config, armijo=2.0e-4)
            with self.assertRaisesRegex(RuntimeError, "scene or configuration identity"):
                self.solver.deterministic_record()
        finally:
            self.solver.config = pristine_config

    def test_persistent_array_pointer_replacement_fails_before_execution(self):
        pristine_array = self.solver.operator.mass
        replacement = wp.array(
            np.asarray(pristine_array.numpy(), dtype=np.float64),
            dtype=wp.float64,
            device=self.solver.device,
        )
        try:
            self.solver.operator.mass = replacement
            with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.operator.mass = pristine_array

    def test_complete_persistent_array_binding_defeats_recomputed_cache_at_all_boundaries(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        persistent = binding.persistent
        self.assertIsNotNone(persistent)
        self.assertEqual(len(persistent.arrays), 384)
        self.assertEqual(len(persistent.signatures), 384)
        self.assertIs(self.solver._persistent_array_identity, persistent.signatures)
        for (array_name, array), (signature_name, _signature) in zip(
            persistent.arrays,
            persistent.signatures,
            strict=True,
        ):
            self.assertEqual(array_name, signature_name)
            self.assertIs(dict(self.solver._persistent_input_arrays(binding))[array_name], array)

        original_identity = self.solver._persistent_array_identity
        for iterations in (1, 4):
            lane_solver = self.solver.baseline._lane(iterations).solver
            original = lane_solver.particle_displacements
            alternate = wp.empty(original.shape, dtype=original.dtype, device=self.solver.device)
            for boundary_name, operation in self._owner_boundaries(self.solver):
                with self.subTest(iterations=iterations, boundary=boundary_name):
                    try:
                        lane_solver.particle_displacements = alternate
                        self.solver._persistent_array_identity = self.solver._persistent_array_signatures(binding)
                        with self.assertRaisesRegex(
                            RuntimeError,
                            rf"persistent array owner lane_{iterations}\.solver\.particle_displacements",
                        ):
                            operation()
                    finally:
                        lane_solver.particle_displacements = original
                        self.solver._persistent_array_identity = original_identity
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_each_lane_adjacency_content_tamper_fails_before_execution(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                adjacency = self.solver.baseline._lane(iterations).solver.particle_adjacency
                source = adjacency.v_adj_tets
                pristine = np.asarray(source.numpy(), dtype=np.int32).copy()
                tampered = pristine.copy()
                tampered[0] = np.int32(tampered[0] + 1)
                try:
                    source.assign(tampered)
                    operation = self.solver.run if iterations == 1 else self.solver.run_k4
                    with self.assertRaisesRegex(RuntimeError, "adjacency"):
                        operation(graph_replay=True)
                finally:
                    source.assign(pristine)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_lane_snapshot_names_every_required_adjacency_and_scratch_array(self):
        signatures = dict(self.solver._persistent_array_signatures())
        adjacency_names = (
            "v_adj_faces",
            "v_adj_faces_offsets",
            "v_adj_edges",
            "v_adj_edges_offsets",
            "v_adj_springs",
            "v_adj_springs_offsets",
            "v_adj_tets",
            "v_adj_tets_offsets",
        )
        scratch_names = (
            "particle_q_prev",
            "inertia",
            "particle_displacements",
            "pos_prev_collision_detection",
            "truncation_ts",
            "particle_forces",
            "particle_hessians",
        )
        for iterations in (1, 4):
            for name in scratch_names:
                self.assertIn(f"lane_{iterations}.solver.{name}", signatures)
            for name in adjacency_names:
                key = f"lane_{iterations}.solver.particle_adjacency.{name}"
                self.assertIn(key, signatures)
                array = getattr(self.solver.baseline._lane(iterations).solver.particle_adjacency, name)
                if array.size == 0:
                    self.assertIsNone(array.ptr)
                    self.assertEqual(signatures[key][1], 0)

    def test_each_lane_adjacency_pointer_tamper_blocks_serialization(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                adjacency = self.solver.baseline._lane(iterations).solver.particle_adjacency
                source = adjacency.v_adj_tets
                replacement = wp.array(
                    np.asarray(source.numpy(), dtype=np.int32),
                    dtype=wp.int32,
                    device=self.solver.device,
                )
                try:
                    adjacency.v_adj_tets = replacement
                    with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                        self.solver.deterministic_record()
                finally:
                    adjacency.v_adj_tets = source
        json.dumps(self.solver.deterministic_record(), allow_nan=False)

    def test_each_lane_adjacency_cached_ctype_data_tamper_fails_closed(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                adjacency = self.solver.baseline._lane(iterations).solver.particle_adjacency
                descriptor = adjacency._ctype.v_adj_tets
                original_data = descriptor.data
                try:
                    descriptor.data = int(adjacency.v_adj_faces.ptr)
                    with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                        if iterations == 1:
                            self.solver.run(graph_replay=True)
                        else:
                            self.solver.capture_graphs(warmup_replays=1)
                    if iterations == 4:
                        with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                            self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
                        with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                            self.solver.deterministic_record()
                finally:
                    descriptor.data = original_data

    def test_each_lane_adjacency_cached_ctype_shape_tamper_fails_closed(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                descriptor = self.solver.baseline._lane(iterations).solver.particle_adjacency._ctype.v_adj_tets
                original_shape = descriptor.shape[0]
                try:
                    descriptor.shape[0] = original_shape + 1
                    with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                        if iterations == 1:
                            self.solver.deterministic_record()
                        else:
                            self.solver.run_k4(graph_replay=True)
                finally:
                    descriptor.shape[0] = original_shape

    def test_each_lane_adjacency_cached_ctype_stride_tamper_fails_closed(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                descriptor = self.solver.baseline._lane(iterations).solver.particle_adjacency._ctype.v_adj_tets
                original_stride = descriptor.strides[0]
                try:
                    descriptor.strides[0] = original_stride + 4
                    with self.assertRaisesRegex(RuntimeError, "cached C descriptor"):
                        if iterations == 1:
                            self.solver.run(graph_replay=True)
                        else:
                            self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
                finally:
                    descriptor.strides[0] = original_stride

    def test_direct_array_fake_same_layout_cached_ctype_fails_closed(self):
        source = self.solver.operator.mass
        original = source.__ctype__()

        class FakeArrayDescriptor(ctypes.Structure):
            _fields_ = type(original)._fields_

        fake = FakeArrayDescriptor()
        fake.data = original.data
        fake.grad = original.grad
        fake.ndim = original.ndim
        for index in range(4):
            fake.shape[index] = original.shape[index]
            fake.strides[index] = original.strides[index]
        try:
            source.ctype = fake
            with self.assertRaisesRegex(RuntimeError, "type or field layout"):
                self.solver.deterministic_record()
        finally:
            source.ctype = original

    def test_each_lane_scratch_pointer_tamper_fails_run_and_timing(self):
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                lane_solver = self.solver.baseline._lane(iterations).solver
                source = lane_solver.particle_forces
                replacement = wp.array(
                    np.asarray(source.numpy(), dtype=np.float32),
                    dtype=wp.vec3,
                    device=self.solver.device,
                )
                try:
                    lane_solver.particle_forces = replacement
                    if iterations == 1:
                        with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                            self.solver.run(graph_replay=True)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "allocation or pointer"):
                            self.solver.benchmark_paired(pair_count=2, warmup_replays=1)
                finally:
                    lane_solver.particle_forces = source

    def test_k4_iteration_schedule_tamper_fails_before_recapture(self):
        k4_solver = self.solver.baseline._lane(4).solver
        original_iterations = k4_solver.iterations
        try:
            k4_solver.iterations = 1
            with self.assertRaisesRegex(RuntimeError, "K4 solver iteration schedule"):
                self.solver.capture_graphs(warmup_replays=1)
            with self.assertRaisesRegex(RuntimeError, "K4 solver iteration schedule"):
                self.solver.run_k4(graph_replay=True)
        finally:
            k4_solver.iterations = original_iterations
        self.assertEqual(
            self.solver.run_k4(graph_replay=True).endpoint_sha256,
            self.solver._construction_k4.endpoint_sha256,
        )

    def test_record_path_requires_an_unconsumed_private_execution(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        self.assertFalse(hasattr(self.solver, "record"))
        self.assertFalse(hasattr(module, "_EXECUTION_TOKEN"))
        self.solver._pending_execution = (False, 999)
        try:
            with self.assertRaisesRegex(RuntimeError, "exact solver-issued execution receipt"):
                self.solver._record(execution_receipt=object())
        finally:
            del self.solver._pending_execution
        graph = self.solver.graph
        try:
            self.solver.graph = self.solver.k4_graph
            with self.assertRaisesRegex(RuntimeError, "captured graph object"):
                self.solver.run(graph_replay=True)
        finally:
            self.solver.graph = graph

    def test_solver_graph_identity_labels_cannot_be_reassigned(self):
        fields = (
            "_uncaptured_graph_identity_sha256",
            "graph_identity_sha256",
            "k4_graph_identity_sha256",
        )
        for field in fields:
            with self.subTest(field=field):
                original = getattr(self.solver, field)
                try:
                    setattr(self.solver, field, "a" * 64)
                    with self.assertRaisesRegex(RuntimeError, "identity label"):
                        self.solver.deterministic_record()
                finally:
                    setattr(self.solver, field, original)

    def test_graph_pair_generation_and_recomputed_facades_fail_at_all_execution_boundaries(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        capture_binding = binding.capture
        self.assertIsNotNone(capture_binding)
        with wp.ScopedCapture(device=self.solver.device) as clone_capture:
            self.solver._enqueue_integrated(binding)
        graph_clone = clone_capture.graph
        self.assertIsNot(graph_clone, capture_binding.graph)

        original_facade = (
            self.solver.graph,
            self.solver.k4_graph,
            self.solver._captured_graph_object_identity,
            self.solver._capture_generation,
            self.solver.graph_identity_sha256,
            self.solver.k4_graph_identity_sha256,
        )

        def install_forged_facade(graph, k4_graph, generation):
            self.solver.graph = graph
            self.solver.k4_graph = k4_graph
            self.solver._captured_graph_object_identity = (id(graph), id(k4_graph))
            self.solver._capture_generation = generation
            self.solver.graph_identity_sha256 = self.solver._derive_graph_identity(
                captured=True,
                comparator=False,
                owner_binding=binding,
                graph=graph,
                generation=generation,
            )
            self.solver.k4_graph_identity_sha256 = self.solver._derive_graph_identity(
                captured=True,
                comparator=True,
                owner_binding=binding,
                graph=k4_graph,
                generation=generation,
            )

        def install_fresh_equal_object_identity():
            install_forged_facade(
                capture_binding.graph,
                capture_binding.k4_graph,
                capture_binding.generation,
            )
            self.solver._captured_graph_object_identity = (
                capture_binding.object_identity[0],
                capture_binding.object_identity[1],
            )

        attacks = (
            (
                "integrated-to-k4",
                lambda: install_forged_facade(
                    capture_binding.k4_graph,
                    capture_binding.k4_graph,
                    capture_binding.generation,
                ),
            ),
            (
                "swap-integrated-k4",
                lambda: install_forged_facade(
                    capture_binding.k4_graph,
                    capture_binding.graph,
                    capture_binding.generation,
                ),
            ),
            (
                "integrated-clone",
                lambda: install_forged_facade(
                    graph_clone,
                    capture_binding.k4_graph,
                    capture_binding.generation,
                ),
            ),
            (
                "generation-and-labels",
                lambda: install_forged_facade(
                    capture_binding.graph,
                    capture_binding.k4_graph,
                    capture_binding.generation + 1,
                ),
            ),
            ("fresh-equal-object-identity", install_fresh_equal_object_identity),
        )
        boundaries = (
            ("run", lambda: self.solver.run(graph_replay=True)),
            ("run_k4", lambda: self.solver.run_k4(graph_replay=True)),
            ("timing", lambda: self.solver.benchmark_paired(pair_count=2, warmup_replays=1)),
            ("serialization", self.solver.deterministic_record),
        )
        for attack_name, install_attack in attacks:
            for boundary_name, operation in boundaries:
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    try:
                        install_attack()
                        with self.assertRaisesRegex(RuntimeError, "captured graph object facade or generation"):
                            operation()
                    finally:
                        (
                            self.solver.graph,
                            self.solver.k4_graph,
                            self.solver._captured_graph_object_identity,
                            self.solver._capture_generation,
                            self.solver.graph_identity_sha256,
                            self.solver.k4_graph_identity_sha256,
                        ) = original_facade
        self.assertIs(module._lookup_workspace_owners(self.solver), binding)
        self.assertEqual(self.solver.run(graph_replay=True).endpoint_sha256, self.endpoint.endpoint_sha256)

    def test_native_graph_exec_swaps_fail_at_all_launch_and_evidence_boundaries(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        capture_binding = binding.capture
        self.assertIsNotNone(capture_binding)
        graph = capture_binding.graph
        k4_graph = capture_binding.k4_graph
        graph_exec = capture_binding.graph_native.graph_exec
        k4_graph_exec = capture_binding.k4_graph_native.graph_exec
        self.assertIs(graph.graph_exec, graph_exec)
        self.assertIs(k4_graph.graph_exec, k4_graph_exec)
        self.assertNotEqual(graph_exec.value, k4_graph_exec.value)

        attacks = (
            ("integrated-to-k4-exec", lambda: setattr(graph, "graph_exec", k4_graph_exec)),
            ("k4-to-integrated-exec", lambda: setattr(k4_graph, "graph_exec", graph_exec)),
            (
                "swap-both-execs",
                lambda: (
                    setattr(graph, "graph_exec", k4_graph_exec),
                    setattr(k4_graph, "graph_exec", graph_exec),
                ),
            ),
        )
        boundaries = (
            ("run", lambda: self.solver.run(graph_replay=True)),
            ("run_k4", lambda: self.solver.run_k4(graph_replay=True)),
            ("timing", lambda: self.solver.benchmark_paired(pair_count=2, warmup_replays=1)),
            ("serialization", self.solver.deterministic_record),
            ("record", lambda: dataclasses.replace(self.endpoint)),
            ("recapture", lambda: self.solver.capture_graphs(warmup_replays=1)),
        )
        marker = np.asarray(self.solver.final_positions.numpy(), dtype=np.float32).copy()
        for attack_name, install_attack in attacks:
            for boundary_name, operation in boundaries:
                with self.subTest(attack=attack_name, boundary=boundary_name):
                    try:
                        install_attack()
                        self.solver.graph_identity_sha256 = self.solver._graph_identity(
                            captured=True,
                            comparator=False,
                            owner_binding=binding,
                        )
                        self.solver.k4_graph_identity_sha256 = self.solver._graph_identity(
                            captured=True,
                            comparator=True,
                            owner_binding=binding,
                        )
                        with self.assertRaisesRegex(RuntimeError, "native graph_exec owner or handle value"):
                            operation()
                        np.testing.assert_array_equal(
                            np.asarray(self.solver.final_positions.numpy(), dtype=np.float32),
                            marker,
                        )
                    finally:
                        graph.graph_exec = graph_exec
                        k4_graph.graph_exec = k4_graph_exec

    def test_all_native_graph_and_replay_stream_fields_are_bound(self):
        module = __import__("research.principal_stretch.captured_graph_vbd", fromlist=["*"])
        binding = module._lookup_workspace_owners(self.solver)
        capture_binding = binding.capture
        self.assertIsNotNone(capture_binding)
        graph_native = capture_binding.graph_native
        graph = capture_binding.graph
        replay = binding.replay_stream
        cases = (
            (
                "graph-handle-object",
                graph,
                "graph",
                capture_binding.k4_graph_native.graph_handle,
                "native graph owner or handle value",
            ),
            ("capture-id", graph, "capture_id", graph.capture_id + 1, "capture ID"),
            ("module-execs", graph, "module_execs", set(graph.module_execs), "module-exec set"),
            (
                "stream-handle",
                replay.stream,
                "cuda_stream",
                int(self.solver.device.stream.cuda_stream),
                "replay stream owner or native handle",
            ),
        )
        for name, owner, field_name, replacement, message in cases:
            with self.subTest(field=name):
                original = getattr(owner, field_name)
                try:
                    setattr(owner, field_name, replacement)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.solver.deterministic_record()
                finally:
                    setattr(owner, field_name, original)

        original_exec_value = graph_native.graph_exec.value
        try:
            graph_native.graph_exec.value = capture_binding.k4_graph_native.graph_exec_value
            with self.assertRaisesRegex(RuntimeError, "native graph_exec owner or handle value"):
                self.solver.run(graph_replay=True)
        finally:
            graph_native.graph_exec.value = original_exec_value

        original_context = self.solver.device._context
        try:
            self.solver.device._context = original_context + 1
            with self.assertRaisesRegex(RuntimeError, "replay device or context owner"):
                self.solver.deterministic_record()
        finally:
            self.solver.device._context = original_context

    def test_segment_rejection_is_sticky_fail_closed_but_keeps_fixed_work(self):
        solver = CapturedDirectGraphVBD(
            self.scene,
            device="cuda:0",
            config=DirectGraphVBDConfig(minimum_determinant=2.0),
        )
        solver.capture_graphs(warmup_replays=1)
        endpoint = solver.run(graph_replay=True)
        self.assertEqual(endpoint.accepted, (False, False, False, False))
        self.assertEqual(
            endpoint.reasons,
            ("segment-inversion", "masked-after-rejection", "masked-after-rejection", "masked-after-rejection"),
        )
        np.testing.assert_array_equal(endpoint.positions, endpoint.outer_start_positions[0])
        self.assertEqual(endpoint.initial_objectives[1:], (0.0, 0.0, 0.0))
        self.assertEqual(endpoint.candidate_objectives[1:], (0.0, 0.0, 0.0))
        rejected_state = endpoint.outer_start_positions[0]
        for outer_index, work in enumerate(endpoint.outer_work[1:], start=1):
            np.testing.assert_array_equal(endpoint.outer_start_positions[outer_index], rejected_state)
            np.testing.assert_array_equal(endpoint.outer_candidate_positions[outer_index], rejected_state)
            for name in (
                "rhs",
                "first_correction",
                "operator_product_after_first",
                "residual_after_first",
                "second_correction",
                "direction",
            ):
                np.testing.assert_array_equal(getattr(work, name), 0.0)
        self.assertTrue(endpoint.exact_work_completed)

        masked = endpoint.outer_work[1]
        with self.assertRaisesRegex(ValueError, "exact solver-issued schedule slot"):
            dataclasses.replace(masked, outer_index=0)
        slot_zero = endpoint.outer_work[0]
        with self.assertRaisesRegex(ValueError, "exact solver-issued schedule slot"):
            dataclasses.replace(
                masked,
                outer_index=0,
                start_position_sha256=slot_zero.start_position_sha256,
                current_operator_sha256=slot_zero.current_operator_sha256,
                rhs=slot_zero.rhs,
                first_correction=slot_zero.first_correction,
                operator_product_after_first=slot_zero.operator_product_after_first,
                residual_after_first=slot_zero.residual_after_first,
                second_correction=slot_zero.second_correction,
                direction=slot_zero.direction,
                v_cycles=slot_zero.v_cycles,
                accepted=slot_zero.accepted,
                reason=slot_zero.reason,
                _validation_operator=slot_zero._validation_operator,
            )

    def test_timing_boundary_is_balanced_diagnostic_only(self):
        timing = self.solver.benchmark_paired(pair_count=4, warmup_replays=2, random_seed=4817)
        self.assertEqual(timing.pair_orders.count("AB"), 2)
        self.assertEqual(timing.pair_orders.count("BA"), 2)
        self.assertGreater(timing.graph_median_seconds, 0.0)
        self.assertGreater(timing.k4_median_seconds, 0.0)
        self.assertTrue(timing.integrated_direct_graph)
        self.assertFalse(timing.setup_included)
        self.assertFalse(timing.transfers_included)
        self.assertFalse(timing.performance_evidence)
        self.assertEqual(timing.contract_id, CONTRACT_ID)
        self.assertEqual(timing.scene_sha256, self.solver.scene_sha256)
        self.assertEqual(timing.config_sha256, self.solver.config_sha256)
        self.assertEqual(timing.persistent_device_sha256, self.solver._persistent_device_sha256)
        self.assertEqual(timing.graph_identity_sha256, self.solver.graph_identity_sha256)
        self.assertEqual(timing.k4_graph_identity_sha256, self.solver.k4_graph_identity_sha256)
        json.dumps(timing.deterministic_record(), allow_nan=False)

    def test_endpoint_evidence_is_bytes_backed_and_immutable(self):
        with self.assertRaises(ValueError):
            self.endpoint.positions.setflags(write=True)
        with self.assertRaises(ValueError):
            self.endpoint.outer_work[0].direction.setflags(write=True)
        self.assertEqual(len(self.endpoint.outer_start_position_sha256s), OUTER_CORRECTIONS)
        self.assertEqual(len(self.endpoint.outer_candidate_position_sha256s), OUTER_CORRECTIONS)
        self.assertEqual(len(set(self.endpoint.current_operator_sha256s)), OUTER_CORRECTIONS)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestCapturedDirectGraphVBDDefaultStretchCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        cls.scene = build_stretch_scene()
        cls.problem = build_common_problem(cls.scene)
        cls.solver = CapturedDirectGraphVBD(cls.scene, device="cuda:0")
        cls.solver.capture_graphs(warmup_replays=1)
        cls.endpoint = cls.solver.run(graph_replay=True)
        cls.k4 = cls.solver.run_k4(graph_replay=True)

    def test_real_default_stretch_accepts_four_current_operator_corrections(self):
        endpoint = self.endpoint
        self.assertEqual(endpoint.accepted, (True,) * OUTER_CORRECTIONS)
        self.assertEqual(endpoint.reasons, ("accepted",) * OUTER_CORRECTIONS)
        self.assertEqual(len(set(endpoint.current_operator_sha256s)), OUTER_CORRECTIONS)
        for index in range(OUTER_CORRECTIONS):
            self.assertLess(endpoint.candidate_objectives[index], endpoint.initial_objectives[index])
            self.assertLessEqual(
                endpoint.candidate_objectives[index],
                endpoint.initial_objectives[index]
                + self.solver.config.armijo * endpoint.directional_derivatives[index],
            )
            self.assertGreater(endpoint.segment_minimum_determinants[index], 0.0)

    def test_real_default_stretch_scalar_fused_schedule_and_physical_work_are_exact(self):
        self.assertEqual(self.solver.device_hierarchy.scheduled_kernel_launches, 21)
        self.assertEqual(self.solver.linear_kernel_launches_per_outer, 49)
        self.assertEqual(self.solver.correction_kernel_launches, 230)
        for outer in self.endpoint.outer_work:
            self.assertEqual(outer.linear_kernel_launches, 49)
            for record in outer.v_cycles:
                self.assertEqual(record.work.matrix_block_products, 5058)
                self.assertEqual(record.physical_work.matrix_block_products_executed, 3372)
                self.assertEqual(record.physical_work.matrix_block_products_elided_zero_start, 1686)
                self.assertEqual(record.physical_work.zero_start_block_solves, 184)
                self.assertEqual(record.physical_work.out_of_place_jacobi_block_solves, 184)
                self.assertEqual(record.physical_work.matrix_kernel_launches, 6)
                self.assertEqual(record.physical_work.jacobi_kernel_launches, 6)
                self.assertEqual(record.physical_work.scheduled_kernel_launches, 21)
                self.assertFalse(record.performance_evidence)
        schedule = self.solver.deterministic_record()
        self.assertEqual(schedule["v_cycle_kernel_launches"], 21)
        self.assertEqual(schedule["linear_kernel_launches_per_outer"], 49)
        self.assertEqual(schedule["correction_kernel_launches_excluding_public_k1"], 230)
        self.assertEqual(schedule["v_cycle_canonical_matrix_block_products"], 5058)
        self.assertEqual(schedule["v_cycle_matrix_block_products_executed"], 3372)
        self.assertEqual(schedule["v_cycle_matrix_block_products_elided_zero_start"], 1686)
        self.assertEqual(schedule["v_cycle_zero_start_block_solves"], 184)
        self.assertEqual(schedule["v_cycle_out_of_place_jacobi_block_solves"], 184)
        self.assertEqual(schedule["v_cycle_matrix_kernel_launches"], 6)
        self.assertEqual(schedule["v_cycle_jacobi_kernel_launches"], 6)

    def test_real_default_stretch_is_safe_exactly_pinned_and_better_than_k4(self):
        endpoint = self.endpoint
        np.testing.assert_array_equal(endpoint.positions[self.scene.pinned_indices], self.scene.pin_targets)
        np.testing.assert_array_equal(endpoint.velocities[self.scene.pinned_indices], 0.0)
        expected_velocity = (
            ((endpoint.positions - self.scene.x_current) * np.float64(1.0 / self.scene.dt))
            .astype(np.float32)
            .astype(np.float64)
        )
        expected_velocity[self.scene.pinned_indices] = 0.0
        np.testing.assert_array_equal(endpoint.velocities, expected_velocity)
        metrics = evaluate_common_state(self.problem, endpoint.positions)
        k4_metrics = evaluate_common_state(self.problem, self.k4.positions)
        self.assertEqual(metrics.inverted_tet_fraction, 0.0)
        self.assertGreater(metrics.determinant_min, 0.0)
        self.assertLess(metrics.relative_residual, k4_metrics.relative_residual)
        self.assertFalse(self.k4.integrated_mg)

    def test_real_default_stretch_paired_timing_is_diagnostic(self):
        timing = self.solver.benchmark_paired(pair_count=4, warmup_replays=2, random_seed=9901)
        record = timing.deterministic_record()
        self.assertGreater(timing.graph_median_seconds, 0.0)
        self.assertGreater(timing.k4_median_seconds, 0.0)
        self.assertFalse(timing.performance_evidence)
        print("CAPTURED_DIRECT_GRAPH_VBD_TIMING=" + json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
