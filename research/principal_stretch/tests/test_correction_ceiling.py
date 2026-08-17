# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dense residual-correction quality ceiling."""

from __future__ import annotations

import copy
import dataclasses
import types
import unittest
from unittest import mock

import numpy as np

from ..correction_ceiling import (
    CorrectionLadder,
    _validate_accepted_reference_record,
    _validate_vbd_timing_record,
    _verify_self_hashed_record,
    _verify_transition_by_canonical_replay,
    build_correction_start,
    build_vbd_correction_start,
    compare_endpoint_to_state,
    evaluate_pr_transition_correction_ceiling,
    run_dense_residual_prefix,
    smallest_passing_budget,
    verify_correction_endpoint_record,
)
from ..newton_baseline import NewtonProblem, NewtonResidualPolishConfig
from ..pr_scene_history import (
    AtomicCoordinate,
    FrameSchedule,
    HistoryCheckpoint,
    PRHistoryChain,
    PRSceneHistory,
    _advance_prefix,
    _root_prefix,
)
from ..solver_benchmark import (
    TetBenchmarkScene,
    _array_digest,
    _canonical_digest,
    _vbd_run_digest,
    build_common_problem,
    build_structured_cantilever_scene,
    evaluate_common_state,
    run_newton,
    run_vbd,
)


class TestCorrectionCeiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = build_structured_cantilever_scene()
        cls.problem = build_common_problem(cls.scene)
        cls.reference_run = run_newton(cls.scene, cls.problem, warmup=False, repeats=1)
        if not cls.reference_run.reference_accepted:
            raise RuntimeError(f"test dense reference was rejected: {cls.reference_run.reference_failures}")
        cls.reference = cls.reference_run.result.x.detach().cpu().numpy()
        cls.vbd_1 = run_vbd(cls.scene, 1, device="cpu", warmup=False, repeats=1)
        cls.vbd_4 = run_vbd(cls.scene, 4, device="cpu", warmup=False, repeats=1)
        cls.start = build_vbd_correction_start(cls.scene, cls.problem, cls.vbd_1, cls.reference)
        cls.comparator = build_vbd_correction_start(cls.scene, cls.problem, cls.vbd_4, cls.reference)

    def test_one_correction_from_vbd_k1_beats_fresh_vbd_k4(self):
        endpoint = run_dense_residual_prefix(
            self.scene,
            self.problem,
            self.start,
            self.reference,
            1,
        )
        comparison = compare_endpoint_to_state(endpoint, self.comparator)
        work = endpoint.solver_record["work"]

        self.assertTrue(self.reference_run.reference_accepted)
        self.assertEqual(self.start.provenance["iterations"], 1)
        self.assertEqual(self.comparator.provenance["iterations"], 4)
        self.assertTrue(endpoint.exact_budget_completed)
        self.assertFalse(endpoint.saturated)
        self.assertTrue(endpoint.state_valid)
        self.assertEqual(endpoint.solver_record["reason"], "max_iterations")
        self.assertEqual(endpoint.applied_corrections, 1)
        self.assertEqual(len(endpoint.solver_record["trace"]), 2)
        self.assertEqual(work["hessian_evaluations"], 1)
        self.assertEqual(work["eigenvalue_evaluations"], 1)
        self.assertEqual(work["factorization_attempts"], 1)
        self.assertTrue(comparison.passed)
        self.assertLess(comparison.residual_ratio, 1.0)
        self.assertLess(comparison.free_rms_ratio, 1.0)
        self.assertLess(endpoint.final_metrics.objective, endpoint.initial_metrics.objective)
        self.assertEqual(endpoint.final_metrics.max_pin_error_m, 0.0)
        self.assertEqual(endpoint.final_metrics.inverted_tet_fraction, 0.0)
        self.assertFalse(endpoint.positions.flags["W"])
        with self.assertRaises(ValueError):
            endpoint.positions.setflags(write=True)
        with self.assertRaises(ValueError):
            endpoint.start.positions.setflags(write=True)

    def test_two_corrections_apply_exact_full_budget(self):
        endpoint = run_dense_residual_prefix(
            self.scene,
            self.problem,
            self.start,
            self.reference,
            2,
        )
        work = endpoint.solver_record["work"]

        self.assertTrue(endpoint.exact_budget_completed)
        self.assertFalse(endpoint.saturated)
        self.assertFalse(endpoint.solver_record["converged"])
        self.assertEqual(endpoint.solver_record["reason"], "max_iterations")
        self.assertEqual(endpoint.applied_corrections, 2)
        self.assertEqual(len(endpoint.solver_record["trace"]), 3)
        self.assertEqual(work["hessian_evaluations"], 2)
        self.assertEqual(work["eigenvalue_evaluations"], 2)
        self.assertEqual(work["factorization_attempts"], 2)
        self.assertEqual([item["accepted_step_size"] for item in endpoint.solver_record["trace"]], [1.0, 1.0, 0.0])
        self.assertLess(endpoint.final_metrics.relative_residual, 1.0e-12)

    def test_stationary_start_is_saturated_not_exact_larger_budget(self):
        scene = build_structured_cantilever_scene(
            gravity=(0.0, 0.0, 0.0),
            total_tip_force=(0.0, 0.0, 0.0),
        )
        problem = build_common_problem(scene)
        stationary = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets).numpy()
        start = build_correction_start(
            scene,
            problem,
            stationary,
            stationary,
            role="stationary-control",
            provenance={"contract": "unittest-stationary-control-v1"},
        )
        endpoint = run_dense_residual_prefix(scene, problem, start, stationary, 2)

        self.assertEqual(start.metrics.gradient_norm, 0.0)
        self.assertTrue(endpoint.saturated)
        self.assertFalse(endpoint.exact_budget_completed)
        self.assertEqual(endpoint.applied_corrections, 0)
        self.assertEqual(endpoint.solver_record["reason"], "gradient")

    def test_terminal_gradient_after_requested_update_completes_exact_budget(self):
        inertia_scene = dataclasses.replace(
            self.scene,
            tet_materials=np.zeros_like(self.scene.tet_materials),
            gravity=np.zeros(3),
        )
        problem = build_common_problem(inertia_scene)
        reference = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets).numpy()
        start = build_correction_start(
            inertia_scene,
            problem,
            inertia_scene.x_current,
            reference,
            role="pure-inertia-control",
            provenance={"contract": "unittest-pure-inertia-control-v1"},
        )
        endpoint = run_dense_residual_prefix(inertia_scene, problem, start, reference, 1)

        self.assertEqual(endpoint.solver_record["reason"], "gradient")
        self.assertEqual(endpoint.applied_corrections, 1)
        self.assertTrue(endpoint.exact_budget_completed)
        self.assertFalse(endpoint.saturated)
        self.assertEqual(endpoint.final_metrics.gradient_norm, 0.0)

    def test_nonbinary_backtrack_uses_solver_multiplication_order(self):
        endpoint = run_dense_residual_prefix(
            self.scene,
            self.problem,
            self.start,
            self.reference,
            1,
            base_config=NewtonResidualPolishConfig(
                max_iterations=1,
                gradient_absolute_tolerance=0.0,
                gradient_relative_tolerance=0.0,
                armijo=0.96,
                backtrack=0.3,
                max_line_search_steps=30,
            ),
        )

        self.assertEqual(endpoint.solver_record["trace"][0]["accepted_step_size"], 0.027)
        self.assertTrue(endpoint.exact_budget_completed)

    def test_deterministic_record_excludes_and_binds_timing(self):
        first = run_dense_residual_prefix(self.scene, self.problem, self.start, self.reference, 1)
        second = run_dense_residual_prefix(self.scene, self.problem, self.start, self.reference, 1)

        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.endpoint_sha256, second.endpoint_sha256)
        self.assertNotIn("timing", first.as_dict())
        self.assertGreater(first.timing_record()["solver"]["total_seconds"], 0.0)
        self.assertEqual(first.timing_record()["endpoint_sha256"], first.endpoint_sha256)
        self.assertEqual(len(first.timing_record()["timing_sha256"]), 64)
        self.assertFalse(first.timing_record()["performance_evidence"])
        self.assertEqual(
            first.timing_record()["measurement_provenance"],
            "self-reported-diagnostic-not-performance-evidence-v1",
        )
        verify_correction_endpoint_record(
            first.as_dict(),
            scene=self.scene,
            problem=self.problem,
            reference_positions=self.reference,
            start_positions=first.start.positions,
            endpoint_positions=first.positions,
            start_vbd_result=self.vbd_1,
        )
        with self.assertRaisesRegex(ValueError, "original run result"):
            verify_correction_endpoint_record(
                first.as_dict(),
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=first.start.positions,
                endpoint_positions=first.positions,
            )
        record = copy.deepcopy(first.as_dict())
        record["forged_unvalidated_claim"] = {"passed": True}
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "record fields changed"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=first.start.positions,
                endpoint_positions=first.positions,
                start_vbd_result=self.vbd_1,
            )

    def test_generic_serialized_start_requires_canonical_mapping_provenance(self):
        class OneShotProvenance(dict):
            def __init__(self):
                super().__init__(contract="generic-unittest-start-v1", source="test")
                self.traversals = 0

            def items(self):
                self.traversals += 1
                if self.traversals > 1:
                    raise AssertionError("provenance input was traversed more than once")
                return super().items()

        provenance = OneShotProvenance()
        generic_start = build_correction_start(
            self.scene,
            self.problem,
            self.vbd_1.positions,
            self.reference,
            role="generic-unittest-start",
            provenance=provenance,
        )
        self.assertEqual(provenance.traversals, 1)
        endpoint = run_dense_residual_prefix(
            self.scene,
            self.problem,
            generic_start,
            self.reference,
            1,
        )
        verify_correction_endpoint_record(
            endpoint.as_dict(),
            scene=self.scene,
            problem=self.problem,
            reference_positions=self.reference,
            start_positions=generic_start.positions,
            endpoint_positions=endpoint.positions,
        )
        with self.assertRaisesRegex(ValueError, "provenance must be a mapping"):
            dataclasses.replace(generic_start, provenance="generic-unittest-start-v1")

        class FalseStartsWithString(str):
            def startswith(self, *args, **kwargs):
                return False

        with self.assertRaisesRegex(ValueError, "role must be nonempty"):
            build_correction_start(
                self.scene,
                self.problem,
                self.vbd_1.positions,
                self.reference,
                role=FalseStartsWithString("vbd-k1"),
                provenance={"contract": "generic-unittest-start-v1"},
            )

        class FalseEqualityString(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return False

        with self.assertRaisesRegex(ValueError, "canonical JSON scalar types"):
            build_correction_start(
                self.scene,
                self.problem,
                self.vbd_1.positions,
                self.reference,
                role="generic-unittest-start",
                provenance={"contract": FalseEqualityString("pss-solver-vbd-state-evidence-v1")},
            )

        record = copy.deepcopy(endpoint.as_dict())
        record["start"]["provenance"] = "generic-unittest-start-v1"
        start_payload = dict(record["start"])
        start_payload.pop("evidence_sha256")
        record["start"]["evidence_sha256"] = _canonical_digest(start_payload)
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "provenance must be a mapping"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=generic_start.positions,
                endpoint_positions=endpoint.positions,
            )

        class FalseInequalityString(str):
            def __ne__(self, other):
                return False

        record = copy.deepcopy(endpoint.as_dict())
        record["contract"] = FalseInequalityString("attacker-endpoint-contract-v1")
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "canonical JSON scalar types"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=generic_start.positions,
                endpoint_positions=endpoint.positions,
            )

    def test_serialized_endpoint_mapping_is_snapshotted_once(self):
        endpoint = run_dense_residual_prefix(self.scene, self.problem, self.start, self.reference, 1)
        forged = copy.deepcopy(endpoint.as_dict())
        forged["config"]["gradient_relative_tolerance"] = 0
        payload = dict(forged)
        payload.pop("endpoint_sha256")
        forged["endpoint_sha256"] = _canonical_digest(payload)

        class SplitRecord(dict):
            def __init__(self, honest, alternate):
                super().__init__(honest)
                self.alternate = alternate
                self.traversals = 0

            def items(self):
                self.traversals += 1
                if self.traversals > 1:
                    raise AssertionError("endpoint record was traversed more than once")
                return self.alternate.items()

        record = SplitRecord(endpoint.as_dict(), forged)
        with self.assertRaisesRegex(ValueError, "gradient_relative_tolerance must be a float"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        self.assertEqual(record.traversals, 1)

    def test_record_objective_and_vbd_tampering_fail_closed(self):
        endpoint = run_dense_residual_prefix(self.scene, self.problem, self.start, self.reference, 1)
        forged_vbd = dataclasses.replace(self.vbd_1, iterations=4, run_sha256="")
        forged_vbd = dataclasses.replace(forged_vbd, run_sha256=_vbd_run_digest(forged_vbd))
        with self.assertRaisesRegex(ValueError, "deterministic CPU replay"):
            build_vbd_correction_start(
                self.scene,
                self.problem,
                forged_vbd,
                self.reference,
            )
        for field, value in (
            ("requested_tile_solve", 0),
            ("effective_tile_solve", 0),
            ("color_group_count", float(self.vbd_1.color_group_count)),
        ):
            with self.subTest(vbd_type_alias=field):
                forged_vbd = dataclasses.replace(self.vbd_1, **{field: value}, run_sha256="")
                forged_vbd = dataclasses.replace(forged_vbd, run_sha256=_vbd_run_digest(forged_vbd))
                with self.assertRaisesRegex(ValueError, "invalid"):
                    build_vbd_correction_start(
                        self.scene,
                        self.problem,
                        forged_vbd,
                        self.reference,
                    )

        record = copy.deepcopy(endpoint.as_dict())
        record["start"]["provenance"] = {"contract": "generic-attacker-controlled-v1"}
        start_payload = dict(record["start"])
        start_payload.pop("evidence_sha256")
        record["start"]["evidence_sha256"] = _canonical_digest(start_payload)
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "reserved vbd-k role"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
            )

        record = copy.deepcopy(endpoint.as_dict())
        provenance = record["start"]["provenance"]
        provenance["device"] = "cuda:malicious"
        provenance["velocities_sha256"] = "f" * 64
        execution_payload = dict(provenance)
        execution_payload.pop("vbd_execution_sha256")
        provenance["vbd_execution_sha256"] = _canonical_digest(execution_payload)
        start_payload = dict(record["start"])
        start_payload.pop("evidence_sha256")
        record["start"]["evidence_sha256"] = _canonical_digest(start_payload)
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "changed from its original run result"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )

        record = copy.deepcopy(endpoint.as_dict())
        record["requested_corrections"] = 2
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        record = copy.deepcopy(endpoint.as_dict())
        record["final_metrics"]["position_sha256"] = "f" * 64
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "endpoint position hash"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        record = copy.deepcopy(endpoint.as_dict())
        record["solver"]["trace"][0]["gradient_norm"] *= 2.0
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "relative residual|merit"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        solver = copy.deepcopy(endpoint.as_dict()["solver"])
        solver["reason"] = "gradient"
        solver["converged"] = True
        with self.assertRaisesRegex(ValueError, "gradient termination"):
            dataclasses.replace(endpoint, solver_record=solver)
        record = copy.deepcopy(endpoint.as_dict())
        record["solver"]["final_objective"] = 1.0e99
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "final_objective"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        record = copy.deepcopy(endpoint.as_dict())
        record["position_sha256"] = "banana"
        record["final_metrics"]["position_sha256"] = "banana"
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "position_sha256"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        record = copy.deepcopy(endpoint.as_dict())
        record["solver"]["reason"] = "residual_line_search"
        record["solver"]["converged"] = False
        for name in record["solver"]["work"]:
            record["solver"]["work"][name] = -1
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        record = copy.deepcopy(endpoint.as_dict())
        record["applied_corrections"] = 1.0
        record["exact_budget_completed"] = 1
        record["saturated"] = 0
        record["state_valid"] = 1
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "applied-correction"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        record = copy.deepcopy(endpoint.as_dict())
        record["solver"]["trace"][0]["iteration"] = False
        record["solver"]["trace"][1]["iteration"] = True
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "trace indices"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        record = copy.deepcopy(endpoint.as_dict())
        record["config"]["gradient_relative_tolerance"] = 0
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "gradient_relative_tolerance must be a float"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )
        with self.assertRaisesRegex(ValueError, "different accepted reference"):
            dataclasses.replace(endpoint, reference_position_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            dataclasses.replace(endpoint, config=dataclasses.replace(endpoint.config, max_iterations=2))
        with self.assertRaisesRegex(ValueError, "result-state hash"):
            build_vbd_correction_start(
                self.scene,
                self.problem,
                dataclasses.replace(self.vbd_1, result_state_sha256="0" * 64),
                self.reference,
            )
        with self.assertRaises(ValueError):
            endpoint.positions[0, 0] = 123.0
        with self.assertRaisesRegex(ValueError, "metrics disagree"):
            dataclasses.replace(
                endpoint,
                final_metrics=dataclasses.replace(endpoint.final_metrics, gradient_norm=-1.0),
            )
        with self.assertRaisesRegex(ValueError, "metrics disagree"):
            dataclasses.replace(
                endpoint.start,
                metrics=dataclasses.replace(endpoint.start.metrics, objective=endpoint.start.metrics.objective + 1.0),
            )
        forged_metrics = dataclasses.replace(
            endpoint.final_metrics,
            objective=endpoint.final_metrics.objective - 1.0,
            elastic=endpoint.final_metrics.elastic - 1.0,
            gradient_norm=0.0,
            relative_residual=0.0,
            free_rms_error_m=0.0,
            mass_weighted_rms_error_m=0.0,
        )
        forged_solver = copy.deepcopy(endpoint.as_dict()["solver"])
        forged_solver["trace"][-1].update(
            objective=forged_metrics.objective,
            gradient_norm=0.0,
            relative_residual=0.0,
            residual_merit=0.0,
        )
        forged_solver.update(
            final_objective=forged_metrics.objective,
            final_gradient_norm=0.0,
            final_relative_residual=0.0,
        )
        with self.assertRaisesRegex(ValueError, "independent objective evaluation"):
            dataclasses.replace(endpoint, final_metrics=forged_metrics, solver_record=forged_solver)
        forged_record = copy.deepcopy(endpoint.as_dict())
        forged_record["final_metrics"] = forged_metrics.as_dict()
        forged_record["solver"] = forged_solver
        forged_payload = dict(forged_record)
        forged_payload.pop("endpoint_sha256")
        forged_record["endpoint_sha256"] = _canonical_digest(forged_payload)
        with self.assertRaisesRegex(ValueError, "independent objective evaluation"):
            verify_correction_endpoint_record(
                forged_record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )

    def test_endpoint_and_verifier_require_canonical_solver_replay(self):
        endpoint = run_dense_residual_prefix(self.scene, self.problem, self.start, self.reference, 1)
        reference_metrics = evaluate_common_state(
            self.problem,
            self.reference,
            reference_positions=self.reference,
        )
        initial = endpoint.initial_metrics
        initial_merit = 0.5 * initial.gradient_norm * initial.gradient_norm
        final_merit = 0.5 * reference_metrics.gradient_norm * reference_metrics.gradient_norm
        forged_solver = {
            "converged": False,
            "reason": "max_iterations",
            "accepted_iterations": 1,
            "residual_scale": endpoint.solver_record["residual_scale"],
            "gradient_limit": 0.0,
            "final_objective": reference_metrics.objective,
            "final_gradient_norm": reference_metrics.gradient_norm,
            "final_relative_residual": reference_metrics.relative_residual,
            "work": {
                "objective_evaluations": 3,
                "gradient_evaluations": 3,
                "hessian_evaluations": 1,
                "eigenvalue_evaluations": 1,
                "factorization_attempts": 1,
                "line_search_trials": 1,
            },
            "trace": [
                {
                    "iteration": 0,
                    "objective": initial.objective,
                    "gradient_norm": initial.gradient_norm,
                    "relative_residual": initial.relative_residual,
                    "residual_merit": initial_merit,
                    "accepted_step_norm": float(np.linalg.norm(self.reference - endpoint.start.positions)),
                    "accepted_step_size": 1.0,
                    "merit_directional_derivative": -(initial.gradient_norm * initial.gradient_norm),
                    "hessian_minimum_eigenvalue": 1.0,
                    "hessian_maximum_eigenvalue": 1.0,
                },
                {
                    "iteration": 1,
                    "objective": reference_metrics.objective,
                    "gradient_norm": reference_metrics.gradient_norm,
                    "relative_residual": reference_metrics.relative_residual,
                    "residual_merit": final_merit,
                    "accepted_step_norm": 0.0,
                    "accepted_step_size": 0.0,
                    "merit_directional_derivative": None,
                    "hessian_minimum_eigenvalue": None,
                    "hessian_maximum_eigenvalue": None,
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "deterministic solver replay"):
            dataclasses.replace(
                endpoint,
                positions=self.reference,
                final_metrics=reference_metrics,
                solver_record=forged_solver,
            )

        record = copy.deepcopy(endpoint.as_dict())
        record["position_sha256"] = reference_metrics.position_sha256
        record["final_metrics"] = reference_metrics.as_dict()
        record["solver"] = forged_solver
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "deterministic solver replay"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=self.reference,
                start_vbd_result=self.vbd_1,
            )

    def test_validation_rejects_problem_subclasses(self):
        class OverriddenProblem(NewtonProblem):
            def free_from_positions(self, x):
                return super().free_from_positions(x)

        evil = OverriddenProblem(
            **{
                field.name: getattr(self.problem, field.name)
                for field in dataclasses.fields(NewtonProblem)
                if field.init
            }
        )
        with self.assertRaisesRegex(ValueError, "canonical NewtonProblem type"):
            build_correction_start(
                self.scene,
                evil,
                self.start.positions,
                self.reference,
                role="subclass-control",
                provenance={"contract": "unittest-subclass-control-v1"},
            )

        class OverriddenScene(TetBenchmarkScene):
            def manifest(self):
                record = super().manifest()
                record["scene_sha256"] = "0" * 64
                return record

        evil_scene = OverriddenScene(
            **{
                field.name: getattr(self.scene, field.name)
                for field in dataclasses.fields(TetBenchmarkScene)
                if field.init
            }
        )
        with self.assertRaisesRegex(ValueError, "canonical TetBenchmarkScene"):
            build_correction_start(
                evil_scene,
                self.problem,
                self.start.positions,
                self.reference,
                role="scene-subclass-control",
                provenance={"contract": "unittest-scene-subclass-control-v1"},
            )

    def test_cross_scene_problem_start_and_comparator_are_rejected(self):
        other_scene = build_structured_cantilever_scene(total_tip_force=(5.0, -3.0, -6.0))
        other_problem = build_common_problem(other_scene)
        other_reference_run = run_newton(other_scene, other_problem, warmup=False, repeats=1)
        self.assertTrue(other_reference_run.reference_accepted)
        other_reference = other_reference_run.result.x.detach().cpu().numpy()
        other_vbd = run_vbd(other_scene, 4, device="cpu", warmup=False, repeats=1)
        other_comparator = build_vbd_correction_start(
            other_scene,
            other_problem,
            other_vbd,
            other_reference,
        )
        endpoint = run_dense_residual_prefix(self.scene, self.problem, self.start, self.reference, 1)

        with self.assertRaisesRegex(ValueError, "does not match"):
            run_dense_residual_prefix(self.scene, other_problem, self.start, self.reference, 1)
        with self.assertRaisesRegex(ValueError, "different scene"):
            compare_endpoint_to_state(endpoint, other_comparator)

    def test_ladder_requires_independent_strict_budgets(self):
        endpoints = tuple(
            run_dense_residual_prefix(self.scene, self.problem, self.start, self.reference, budget) for budget in (1, 2)
        )
        comparisons = tuple(compare_endpoint_to_state(endpoint, self.comparator) for endpoint in endpoints)
        ladder = CorrectionLadder(
            start=self.start,
            comparator=self.comparator,
            endpoints=endpoints,
            comparisons=comparisons,
        )

        self.assertEqual(ladder.smallest_passing_budget, 1)
        self.assertEqual(smallest_passing_budget(endpoints, comparisons, self.comparator), 1)
        self.assertEqual(len(ladder.ladder_sha256), 64)
        forged = dataclasses.replace(comparisons[0], residual_ratio=123.0)
        with self.assertRaisesRegex(ValueError, "independent metrics"):
            CorrectionLadder(
                start=self.start,
                comparator=self.comparator,
                endpoints=endpoints,
                comparisons=(forged, comparisons[1]),
            )
        with self.assertRaisesRegex(ValueError, "independent metrics"):
            smallest_passing_budget(endpoints, (forged, comparisons[1]), self.comparator)
        forged_later = dataclasses.replace(comparisons[1], residual_ratio=123.0)
        with self.assertRaisesRegex(ValueError, "independent metrics"):
            smallest_passing_budget(endpoints, (comparisons[0], forged_later), self.comparator)
        mixed_endpoint = run_dense_residual_prefix(
            self.scene,
            self.problem,
            self.comparator,
            self.reference,
            2,
        )
        mixed_comparison = compare_endpoint_to_state(mixed_endpoint, self.comparator)
        with self.assertRaisesRegex(ValueError, "one initializer"):
            smallest_passing_budget(
                (endpoints[0], mixed_endpoint),
                (comparisons[0], mixed_comparison),
                self.comparator,
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            CorrectionLadder(
                start=self.start,
                comparator=self.comparator,
                endpoints=(endpoints[1], endpoints[0]),
                comparisons=(comparisons[1], comparisons[0]),
            )

    def test_budget_config_and_reference_validation(self):
        class OverriddenConfig(NewtonResidualPolishConfig):
            def deterministic_record(self):
                record = super().deterministic_record()
                record["armijo"] = 1.0e-8
                return record

        overridden_config = OverriddenConfig(
            max_iterations=1,
            gradient_absolute_tolerance=0.0,
            gradient_relative_tolerance=0.0,
            armijo=1.0e-4,
            backtrack=0.5,
            max_line_search_steps=30,
        )
        with self.assertRaisesRegex(ValueError, "canonical NewtonResidualPolishConfig"):
            run_dense_residual_prefix(
                self.scene,
                self.problem,
                self.start,
                self.reference,
                1,
                base_config=overridden_config,
            )
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                run_dense_residual_prefix(
                    self.scene,
                    self.problem,
                    self.start,
                    self.reference,
                    invalid,
                )
        endpoint = run_dense_residual_prefix(
            self.scene,
            self.problem,
            self.start,
            self.reference,
            1,
            base_config=NewtonResidualPolishConfig(
                max_iterations=9,
                gradient_absolute_tolerance=1.0,
                gradient_relative_tolerance=1.0,
                armijo=5.0e-4,
                backtrack=0.25,
                max_line_search_steps=10,
            ),
        )
        self.assertEqual(endpoint.config.max_iterations, 1)
        self.assertEqual(endpoint.config.gradient_absolute_tolerance, 0.0)
        self.assertEqual(endpoint.config.gradient_relative_tolerance, 0.0)
        self.assertEqual(endpoint.config.armijo, 5.0e-4)
        self.assertEqual(endpoint.config.backtrack, 0.25)

        class LyingBudget(int):
            __hash__ = int.__hash__

            def __eq__(self, other):
                return True

            def __lt__(self, other):
                return False

        with self.assertRaisesRegex(ValueError, "positive integer"):
            dataclasses.replace(endpoint, requested_corrections=LyingBudget(999))
        record = copy.deepcopy(endpoint.as_dict())
        record["requested_corrections"] = LyingBudget(999)
        payload = dict(record)
        payload.pop("endpoint_sha256")
        record["endpoint_sha256"] = _canonical_digest(payload)
        with self.assertRaisesRegex(ValueError, "canonical JSON scalar types"):
            verify_correction_endpoint_record(
                record,
                scene=self.scene,
                problem=self.problem,
                reference_positions=self.reference,
                start_positions=endpoint.start.positions,
                endpoint_positions=endpoint.positions,
                start_vbd_result=self.vbd_1,
            )

        class AlwaysEqualDigest(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            dataclasses.replace(endpoint, scene_sha256=AlwaysEqualDigest("0" * 64))

        wrong_reference = self.reference.copy()
        wrong_reference[0, 0] += 1.0e-6
        with self.assertRaisesRegex(ValueError, "different accepted reference"):
            run_dense_residual_prefix(self.scene, self.problem, self.start, wrong_reference, 1)

    def test_real_pr_transition_reconstruction_binds_fresh_k1_k4(self):
        history = PRSceneHistory("stretch")
        chain = history.generate(max_transitions=1)
        chain.verify()
        self.assertEqual(len(chain.transitions), 1)
        transition = chain.transitions[0]

        list_backed_chain = dataclasses.replace(chain)
        object.__setattr__(list_backed_chain, "transitions", list(list_backed_chain.transitions))
        with self.assertRaisesRegex(ValueError, "tuple-backed PR history chain"):
            evaluate_pr_transition_correction_ceiling(
                history,
                list_backed_chain,
                transition,
                expected_history_chain_sha256=chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        split_record_transition = dataclasses.replace(transition)
        split_record_chain = dataclasses.replace(chain, transitions=(split_record_transition,))

        class SplitBacking(dict):
            def __init__(self, canonical, forged):
                super().__init__(canonical)
                self.forged = forged
                self.forged["forged_unpinned_claim"] = {"promotion_authorized": True}
                self.traversals = 0

            def items(self):
                self.traversals += 1
                source = self if self.traversals == 1 else self.forged
                return dict.items(source)

        split_backing = SplitBacking(
            transition.reference_record,
            copy.deepcopy(transition.as_dict()["reference_record"]),
        )
        object.__setattr__(
            split_record_transition,
            "reference_record",
            types.MappingProxyType(split_backing),
        )
        split_result = evaluate_pr_transition_correction_ceiling(
            history,
            split_record_chain,
            split_record_transition,
            expected_history_chain_sha256=chain.chain_sha256,
            correction_budgets=(1,),
            vbd_device="cpu",
            vbd_warmup=False,
            vbd_repeats=1,
        )
        self.assertEqual(split_backing.traversals, 1)
        self.assertEqual(split_result.transition_sha256, transition.transition_sha256)
        self.assertEqual(
            split_result.reference_record_sha256,
            _canonical_digest(transition.as_dict()["reference_record"]),
        )

        class StretchThatSerializesAsAttackerKind(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return other == "stretch"

        relabeled_history = PRSceneHistory(StretchThatSerializesAsAttackerKind("attacker-kind"))
        relabeled_history_chain = relabeled_history.generate(max_transitions=1)
        with self.assertRaisesRegex(ValueError, "canonical PR history kind"):
            evaluate_pr_transition_correction_ceiling(
                relabeled_history,
                relabeled_history_chain,
                relabeled_history_chain.transitions[0],
                expected_history_chain_sha256=relabeled_history_chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        class OverriddenHistory(PRSceneHistory):
            pass

        overridden_history = OverriddenHistory("stretch")
        with self.assertRaisesRegex(ValueError, "canonical PRSceneHistory"):
            evaluate_pr_transition_correction_ceiling(
                overridden_history,
                chain,
                transition,
                expected_history_chain_sha256=chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        overridden_instance = PRSceneHistory("stretch")
        overridden_instance.frame_schedule = lambda frame: FrameSchedule(
            frame,
            "drive",
            "stretch_ratio",
            1.5,
            False,
        )
        overridden_chain = overridden_instance.generate(max_transitions=1)
        with self.assertRaisesRegex(ValueError, "unmodified canonical PRSceneHistory"):
            evaluate_pr_transition_correction_ceiling(
                overridden_instance,
                overridden_chain,
                overridden_chain.transitions[0],
                expected_history_chain_sha256=overridden_chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        mutated_base = PRSceneHistory("stretch")
        mutated_base._base_scene.gravity.setflags(write=True)
        mutated_base._base_scene.gravity[1] = 123.0
        mutated_chain = mutated_base.generate(max_transitions=1)
        with self.assertRaisesRegex(ValueError, "base scene differs"):
            evaluate_pr_transition_correction_ceiling(
                mutated_base,
                mutated_chain,
                mutated_chain.transitions[0],
                expected_history_chain_sha256=mutated_chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        result = evaluate_pr_transition_correction_ceiling(
            history,
            chain,
            transition,
            expected_history_chain_sha256=chain.chain_sha256,
            correction_budgets=(1,),
            vbd_device="cpu",
            vbd_warmup=False,
            vbd_repeats=1,
        )

        self.assertTrue(transition.reference_record["accepted"])
        self.assertEqual(result.transition_sha256, transition.transition_sha256)
        self.assertEqual(result.scene_sha256, transition.scene_sha256)
        self.assertEqual(result.objective_instance_sha256, transition.objective_instance_sha256)
        self.assertEqual(result.vbd_k1.provenance["iterations"], 1)
        self.assertEqual(result.vbd_k4.provenance["iterations"], 4)
        self.assertEqual(result.primary_smallest_passing_budget, 1)
        self.assertTrue(result.ladders[0].comparisons[0].passed)
        self.assertEqual(result.reference_metrics.max_pin_error_m, 0.0)
        self.assertEqual(result.reference_metrics.inverted_tet_fraction, 0.0)
        self.assertEqual(result.timing_record()["result_sha256"], result.result_sha256)
        self.assertFalse(result.timing_record()["performance_evidence"])
        self.assertFalse(result.timing_record()["values"]["vbd_k1"]["performance_evidence"])

        class OneShotCoordinate(dict):
            def __init__(self):
                super().__init__(transition.coordinate.as_dict())
                self.traversals = 0

            def items(self):
                self.traversals += 1
                if self.traversals > 1:
                    raise AssertionError("coordinate input was traversed more than once")
                return super().items()

        coordinate = OneShotCoordinate()
        coordinate_result = dataclasses.replace(result, coordinate=coordinate)
        self.assertEqual(coordinate.traversals, 1)
        self.assertEqual(dict(coordinate_result.coordinate), transition.coordinate.as_dict())
        relabeled_chain = dataclasses.replace(chain)
        object.__setattr__(relabeled_chain, "chain_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "chain SHA-256 changed"):
            evaluate_pr_transition_correction_ceiling(
                history,
                relabeled_chain,
                transition,
                expected_history_chain_sha256="0" * 64,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )
        with self.assertRaisesRegex(ValueError, "history_manifest_sha256 changed"):
            dataclasses.replace(
                result,
                history_manifest_sha256="0" * 64,
                history_static_sha256="1" * 64,
                history_chain_sha256="2" * 64,
                transition_sha256="3" * 64,
                reference_record_sha256="4" * 64,
                coordinate={"forged": "not-a-coordinate"},
            )
        swapped_timing = copy.deepcopy(result.timing_record()["values"])
        swapped_timing["vbd_k1"], swapped_timing["vbd_k4"] = swapped_timing["vbd_k4"], swapped_timing["vbd_k1"]
        with self.assertRaisesRegex(ValueError, "wrong state evidence"):
            dataclasses.replace(result, timings=swapped_timing)
        negative_timing = copy.deepcopy(result.timing_record()["values"])
        negative_timing["vbd_k1"]["setup_seconds"] = -1.0
        timing_payload = dict(negative_timing["vbd_k1"])
        timing_payload.pop("timing_sha256")
        negative_timing["vbd_k1"]["timing_sha256"] = _canonical_digest(timing_payload)
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            dataclasses.replace(result, timings=negative_timing)
        wrong_run = copy.deepcopy(result.timing_record()["values"]["vbd_k1"])
        wrong_run["run_sha256"] = "0" * 64
        timing_payload = dict(wrong_run)
        timing_payload.pop("timing_sha256")
        wrong_run["timing_sha256"] = _canonical_digest(timing_payload)
        with self.assertRaisesRegex(ValueError, "inconsistent with its execution"):
            _validate_vbd_timing_record(wrong_run, result.vbd_k1)
        forged_provenance = dict(result.vbd_k1.provenance)
        forged_provenance["physical_state_sha256"] = "0" * 64
        execution_payload = dict(forged_provenance)
        execution_payload.pop("vbd_execution_sha256")
        forged_provenance["vbd_execution_sha256"] = _canonical_digest(execution_payload)
        with self.assertRaisesRegex(ValueError, "provenance changed from its original run"):
            dataclasses.replace(result.vbd_k1, provenance=forged_provenance)

        mutable_state = dataclasses.replace(transition.input_state)
        mutable_state.qd.setflags(write=True)
        mutable_state.qd[0, 0] += np.float32(1.0)
        with self.assertRaisesRegex(ValueError, "raw content changed"):
            _verify_self_hashed_record(mutable_state.as_dict(), "state_sha256", "mutated test state")

        alternate_qd = transition.input_state.qd.copy()
        alternate_qd[0, 0] += np.float32(1.0)
        alternate_state = dataclasses.replace(transition.input_state, qd=alternate_qd)
        alternate_checkpoint = HistoryCheckpoint(
            manifest_sha256=chain.manifest.manifest_sha256,
            state=alternate_state,
            prior_transition_sha256=None,
            prefix_sha256=_root_prefix(chain.manifest.manifest_sha256, alternate_state.state_sha256),
        )
        alternate_chain = PRHistoryChain(
            manifest=chain.manifest,
            initial_checkpoint=alternate_checkpoint,
            transitions=(),
            timings=(),
            final_checkpoint=alternate_checkpoint,
            termination="range_complete",
        )
        with self.assertRaisesRegex(ValueError, "exact object stored"):
            evaluate_pr_transition_correction_ceiling(
                history,
                alternate_chain,
                transition,
                expected_history_chain_sha256=alternate_chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        tampered_transition = dataclasses.replace(transition, input_prefix_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "exact object stored"):
            evaluate_pr_transition_correction_ceiling(
                history,
                chain,
                tampered_transition,
                expected_history_chain_sha256=chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        bad_reference_record = copy.deepcopy(transition.as_dict()["reference_record"])
        bad_reference_record["scene_sha256"] = "0" * 64
        bad_reference_record["final_gradient_norm"] = 1.0e99
        bad_transition = dataclasses.replace(transition, reference_record=bad_reference_record)
        bad_final_checkpoint = HistoryCheckpoint(
            manifest_sha256=chain.final_checkpoint.manifest_sha256,
            state=chain.final_checkpoint.state,
            prior_transition_sha256=bad_transition.transition_sha256,
            prefix_sha256=_advance_prefix(chain.initial_checkpoint.prefix_sha256, bad_transition.transition_sha256),
        )
        bad_chain = PRHistoryChain(
            manifest=chain.manifest,
            initial_checkpoint=chain.initial_checkpoint,
            transitions=(bad_transition,),
            timings=chain.timings,
            final_checkpoint=bad_final_checkpoint,
            termination=chain.termination,
        )
        with self.assertRaisesRegex(ValueError, "canonical dense-reference replay|changed scene_sha256"):
            evaluate_pr_transition_correction_ceiling(
                history,
                bad_chain,
                bad_transition,
                expected_history_chain_sha256=bad_chain.chain_sha256,
                correction_budgets=(1,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )

        scene = history.build_atomic_scene(transition.input_state, history.apply_callback(transition.input_state))
        problem = build_common_problem(scene)
        reference_metrics = evaluate_common_state(
            problem,
            transition.reference_positions,
            reference_positions=transition.reference_positions,
        )
        for method, policy_name, policy_value in (
            (
                "dense-cpu-newton-float64-with-strict-residual-polish",
                "residual_polish_policy",
                "strict-reference-residual-newton-three-start-v1",
            ),
            (
                "dense-cpu-newton-float64-with-alternate-residual-verification",
                "alternate_residual_policy",
                "alternate-start-only-residual-verification-v1",
            ),
        ):
            with self.subTest(method=method):
                recovered_record = copy.deepcopy(transition.as_dict()["reference_record"])
                recovered_record["method"] = method
                recovered_record[policy_name] = policy_value
                recovered_record["config"]["step_relative_tolerance"] = 0.0
                recovered_record[policy_name] = "wrong-policy"
                with self.assertRaisesRegex(ValueError, f"changed {policy_name}"):
                    _validate_accepted_reference_record(
                        dataclasses.replace(transition, reference_record=recovered_record),
                        problem,
                        reference_metrics,
                    )
                recovered_record[policy_name] = policy_value
                recovered_record["config"]["gradient_absolute_tolerance"] = 1.0e30
                with self.assertRaisesRegex(ValueError, "canonical history policy"):
                    _validate_accepted_reference_record(
                        dataclasses.replace(transition, reference_record=recovered_record),
                        problem,
                        reference_metrics,
                    )

    def test_nonzero_transition_replay_regenerates_the_canonical_root_prefix(self):
        history = PRSceneHistory("stretch")
        honest = history.generate(
            stop=AtomicCoordinate.from_ordinal(2),
            max_transitions=2,
        )
        first = honest.transitions[0]
        reference = first.reference_positions.copy()
        free = np.setdiff1d(
            np.arange(reference.shape[0], dtype=np.int64),
            first.applied_state.pinned_indices,
        )
        reference[free[0], 0] += 1.0e-6
        reference_record = copy.deepcopy(first.as_dict()["reference_record"])
        reference_record["position_sha256"] = _array_digest(reference)
        output_state = PRSceneHistory._commit_reference(
            history,
            first.input_state,
            first.applied_state,
            reference,
        )
        forked_first = dataclasses.replace(
            first,
            reference_record=reference_record,
            reference_positions=reference,
            output_state=output_state,
        )
        forked_checkpoint = HistoryCheckpoint(
            manifest_sha256=honest.manifest.manifest_sha256,
            state=output_state,
            prior_transition_sha256=forked_first.transition_sha256,
            prefix_sha256=_advance_prefix(
                honest.initial_checkpoint.prefix_sha256,
                forked_first.transition_sha256,
            ),
        )
        forked_prefix = PRHistoryChain(
            manifest=honest.manifest,
            initial_checkpoint=honest.initial_checkpoint,
            transitions=(forked_first,),
            timings=honest.timings[:1],
            final_checkpoint=forked_checkpoint,
            termination="range_complete",
        )
        suffix = history.generate(
            checkpoint=forked_checkpoint,
            start=AtomicCoordinate.from_ordinal(1),
            stop=AtomicCoordinate.from_ordinal(2),
            max_transitions=1,
            prior_chain=forked_prefix,
        )
        forked_chain = PRHistoryChain(
            manifest=honest.manifest,
            initial_checkpoint=honest.initial_checkpoint,
            transitions=(forked_first, suffix.transitions[0]),
            timings=(honest.timings[0], suffix.timings[0]),
            final_checkpoint=suffix.final_checkpoint,
            termination="range_complete",
        )

        self.assertNotEqual(forked_first.output_state.state_sha256, first.output_state.state_sha256)
        with self.assertRaisesRegex(ValueError, "canonical root dense-reference replay"):
            _verify_transition_by_canonical_replay(
                history,
                forked_chain,
                forked_chain.transitions[1],
            )

    def test_terminal_transition_accepts_history_complete_replay(self):
        history = PRSceneHistory("stretch")
        transition = mock.Mock()
        transition.coordinate = AtomicCoordinate.from_ordinal(0)
        transition.next_coordinate = history.manifest.end_coordinate
        transition.as_dict.return_value = {"transition": "terminal"}
        checkpoint = mock.Mock()
        checkpoint.as_dict.return_value = {"checkpoint": "terminal"}
        chain = mock.Mock()
        chain.transitions = (transition,)
        replay = mock.Mock()
        replay.termination = "history_complete"
        replay.transitions = (transition,)
        replay.final_checkpoint = checkpoint

        with (
            mock.patch.object(PRSceneHistory, "generate", return_value=replay),
            mock.patch.object(PRHistoryChain, "checkpoint_at", return_value=checkpoint),
        ):
            _verify_transition_by_canonical_replay(history, chain, transition)

    def test_real_recovered_reference_methods_match_canonical_replay(self):
        history = PRSceneHistory("stretch")
        chain = history.generate(
            stop=AtomicCoordinate.from_ordinal(17),
            max_transitions=17,
        )
        expected_methods = {
            14: "dense-cpu-newton-float64-with-strict-residual-polish",
            16: "dense-cpu-newton-float64-with-alternate-residual-verification",
        }
        for ordinal, method in expected_methods.items():
            with self.subTest(ordinal=ordinal):
                transition = chain.transitions[ordinal]
                self.assertEqual(transition.reference_record["method"], method)
                result = evaluate_pr_transition_correction_ceiling(
                    history,
                    chain,
                    transition,
                    expected_history_chain_sha256=chain.chain_sha256,
                    correction_budgets=(1,),
                    vbd_device="cpu",
                    vbd_warmup=False,
                    vbd_repeats=1,
                )
                self.assertEqual(result.transition_sha256, transition.transition_sha256)
                self.assertEqual(result.primary_smallest_passing_budget, 1)

    def test_pr_transition_and_optional_start_mismatch_fail_closed(self):
        history = PRSceneHistory("stretch")
        chain = history.generate(max_transitions=1)
        transition = chain.transitions[0]

        with self.assertRaisesRegex(ValueError, "another transition objective"):
            evaluate_pr_transition_correction_ceiling(
                history,
                chain,
                transition,
                expected_history_chain_sha256=chain.chain_sha256,
                correction_budgets=(1,),
                optional_starts=(self.start,),
                vbd_device="cpu",
                vbd_warmup=False,
                vbd_repeats=1,
            )
        for invalid in ((), (2, 1), (1, 1), (0,)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "strictly increasing"):
                evaluate_pr_transition_correction_ceiling(
                    history,
                    chain,
                    transition,
                    expected_history_chain_sha256=chain.chain_sha256,
                    correction_budgets=invalid,
                    vbd_device="cpu",
                    vbd_warmup=False,
                    vbd_repeats=1,
                )


if __name__ == "__main__":
    unittest.main()
