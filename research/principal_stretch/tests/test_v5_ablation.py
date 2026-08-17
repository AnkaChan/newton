# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the development-only v5 identical-corrector ablation harness."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import unittest
from unittest import mock

import numpy as np
import torch

from .. import torch_solver as ts
from .. import v5_ablation
from ..graph_transformer import GraphTransformerConfig
from ..iterative_solver import PhysicalStepContext
from ..predictor import build_stretch_predictor
from ..tests.test_graph_transformer import _chain_mesh, _inputs, _tet_poses
from ..v5_ablation import (
    INERTIAL_ARM,
    LEARNED_ARM,
    MANDATORY_ARM_NAMES,
    PERMUTED_ARM,
    PERSISTENCE_ARM,
    VBD_K1_ARM,
    ZERO_ARM,
    AttestedVBDK1Start,
    V5AblationConfig,
    V5AblationResult,
    VBDK1MethodRecord,
    _replay_candidate_safeguards,
    compare_ablation_arms,
    pin_binding_sha256,
    position_sha256,
    run_v5_identical_corrector_ablation,
)
from ..v5_corrector import CorrectorConfig, FixedPCGConfig
from ..v5_objective import CommonObjectiveContext, common_objective_components, common_objective_residual


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class TestV5IdenticalCorrectorAblation(unittest.TestCase):
    def setUp(self) -> None:
        self.rest, self.tets = _chain_mesh(4)
        self.projection = ts.build_solver(
            self.rest,
            self.tets,
            _tet_poses(self.rest, self.tets),
            np.asarray([0, 1, 2], dtype=np.int64),
            device=torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        x_current, x_previous, force, gravity, mu, lam, pin = _inputs(self.rest, self.tets)
        rest_positions = torch.as_tensor(self.rest, dtype=torch.float64)
        x_current = rest_positions + 0.02 * (x_current - rest_positions)
        self.mass = torch.linspace(0.8, 1.2, self.rest.shape[0], dtype=torch.float64)
        self.dt = 1.0 / 60.0
        inertial_target = 2.0 * x_current - x_previous
        inertial_target = inertial_target + self.dt * self.dt * (gravity + force / self.mass[:, None])
        self.objective = CommonObjectiveContext(
            tets=self.projection.tets,
            J=self.projection.J,
            volume=self.projection.w,
            mass=self.mass,
            mu=mu.double(),
            lam=lam.double(),
            inertial_target=inertial_target,
            pinned=self.projection.pinned,
            dt=self.dt,
        )
        self.physical_step = PhysicalStepContext(
            x_current=x_current,
            x_previous=x_previous,
            force=force,
            gravity=gravity,
            mu=mu.double(),
            lam=lam.double(),
            pin=pin.double(),
            pinned_targets=x_current[self.projection.pinned],
        )
        self.predictor = build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float64,
            residual=True,
            graph_config=GraphTransformerConfig(
                hidden_dim=8,
                num_heads=2,
                n_levels=2,
                cluster_size=2,
                max_hencky_update=0.01,
                max_rotation_update=0.015,
                architecture_version=5,
            ),
        )
        generator = torch.Generator().manual_seed(170817)
        with torch.no_grad():
            self.predictor.model.output_head[-1].weight.normal_(std=5.0e-4, generator=generator)
            self.predictor.model.rotation_head[-1].weight.normal_(std=5.0e-4, generator=generator)
        self.predictor.eval()
        self.config = V5AblationConfig(iterations=2, head_permutation=(2, 3, 0, 1))
        self.corrector_config = CorrectorConfig(
            pcg=FixedPCGConfig(iterations=2),
            candidate_alphas=(0.5, 0.0),
        )
        self.vbd_k1 = self._vbd_attestation()

    def _vbd_attestation(
        self,
        *,
        physical_step: PhysicalStepContext | None = None,
        positions: torch.Tensor | None = None,
        **identity_overrides: str,
    ) -> AttestedVBDK1Start:
        physical = self.physical_step if physical_step is None else physical_step
        x_current = physical.x_current
        targets = physical.pinned_targets
        if positions is None:
            positions = x_current.clone()
            free = self.projection.free
            phase = torch.linspace(0.0, math.pi, free.numel(), dtype=positions.dtype)
            positions[free, 0] += 4.0e-4 * torch.sin(phase)
            positions[free, 1] -= 2.0e-4 * torch.cos(phase)
            positions[self.projection.pinned] = targets
        identities = {
            "physical_step_sha256": physical.physical_step_sha256,
            "common_objective_sha256": self.objective.common_objective_sha256,
            "static_mesh_sha256": self.projection.static_mesh_sha256,
            "operator_geometry_sha256": self.projection.operator_geometry_sha256,
            "projection_state_sha256": self.projection.projection_state_sha256,
            "pin_binding_sha256": pin_binding_sha256(self.projection.pinned, targets),
        }
        identities.update(identity_overrides)
        return AttestedVBDK1Start(
            positions=positions,
            method_record=VBDK1MethodRecord(source_run_sha256=_digest("caller-vbd-k1-receipt")),
            **identities,
        )

    def _run(
        self,
        *,
        corrector_config: CorrectorConfig | None = None,
        vbd_k1: AttestedVBDK1Start | object | None = None,
        config: V5AblationConfig | None = None,
        physical_step: PhysicalStepContext | None = None,
    ):
        physical = self.physical_step if physical_step is None else physical_step
        return run_v5_identical_corrector_ablation(
            predictor=self.predictor,
            projection_state=self.projection,
            objective=self.objective,
            physical_step=physical,
            expected_physical_step_sha256=physical.physical_step_sha256,
            corrector_config=self.corrector_config if corrector_config is None else corrector_config,
            vbd_k1=self.vbd_k1 if vbd_k1 is None else vbd_k1,
            config=self.config if config is None else config,
        )

    def test_all_rows_execute_fresh_work_and_one_identical_corrector(self) -> None:
        predictor_type = type(self.predictor)
        predictor_method = predictor_type.predict_principal_stretch_update
        predictor_modes: list[str] = []

        def record_predictor_mode(instance, *args, **kwargs):
            predictor_modes.append(kwargs["head_mode"])
            return predictor_method(instance, *args, **kwargs)

        corrector = v5_ablation.correct_common_objective
        with (
            mock.patch.object(predictor_type, "predict_principal_stretch_update", record_predictor_mode),
            mock.patch.object(v5_ablation, "correct_common_objective", wraps=corrector) as corrector_spy,
        ):
            result = self._run()

        self.assertEqual(tuple(arm.name for arm in result.arms), MANDATORY_ARM_NAMES)
        self.assertEqual(len(predictor_modes), 3 * self.config.iterations)
        self.assertEqual(predictor_modes.count("learned"), self.config.iterations)
        self.assertEqual(predictor_modes.count("zero"), self.config.iterations)
        self.assertEqual(predictor_modes.count("permuted"), self.config.iterations)
        self.assertEqual(corrector_spy.call_count, len(MANDATORY_ARM_NAMES))
        for call in corrector_spy.call_args_list:
            self.assertIs(call.kwargs["context"], self.objective)
            self.assertIs(call.kwargs["config"], self.corrector_config)

        learned_work = dataclasses.asdict(result.arm(LEARNED_ARM).iterative_work)
        self.assertEqual(learned_work, dataclasses.asdict(result.arm(ZERO_ARM).iterative_work))
        self.assertEqual(learned_work, dataclasses.asdict(result.arm(PERMUTED_ARM).iterative_work))
        self.assertEqual(learned_work["predictor_passes"], self.config.iterations)
        self.assertEqual(learned_work["projection_calls"], self.config.iterations)
        scheduled_hashes = {arm.corrector_scheduled_work_sha256 for arm in result.arms}
        config_hashes = {arm.corrector_config_sha256 for arm in result.arms}
        self.assertEqual(scheduled_hashes, {result.corrector_scheduled_work_sha256})
        self.assertEqual(config_hashes, {result.corrector_config_sha256})
        self.assertEqual(result.corrector_call_count, len(MANDATORY_ARM_NAMES))

        self.assertNotEqual(
            result.arm(LEARNED_ARM).pre_corrector_metrics.positions_sha256,
            result.arm(PERMUTED_ARM).pre_corrector_metrics.positions_sha256,
        )
        self.assertNotEqual(
            result.arm(ZERO_ARM).pre_corrector_metrics.positions_sha256,
            result.arm(PERMUTED_ARM).pre_corrector_metrics.positions_sha256,
        )
        self.assertEqual(result.arm(PERSISTENCE_ARM).pin_overwrite_vertices, 0)
        self.assertEqual(result.arm(INERTIAL_ARM).pin_overwrite_vertices, self.projection.pinned.numel())
        self.assertIn("caller-attested", result.vbd_freshness_scope)
        self.assertIn("caller-attested", result.arm(VBD_K1_ARM).vbd_freshness_scope)
        self.assertTrue(result.development_only)
        self.assertFalse(result.learned_value_claim)
        self.assertIn("no-learned-value-claim", result.claim_scope)

    def test_corrected_endpoint_metrics_and_retention_are_independent(self) -> None:
        result = self._run()
        persistence = self.physical_step.x_current
        targets = self.physical_step.pinned_targets
        for arm in result.arms:
            with self.subTest(arm=arm.name):
                self.assertEqual(
                    position_sha256(arm.pre_corrector_positions), arm.pre_corrector_metrics.positions_sha256
                )
                self.assertEqual(position_sha256(arm.corrected_positions), arm.corrected_metrics.positions_sha256)
                components = common_objective_components(self.objective, arm.corrected_positions)
                residual = common_objective_residual(self.objective, arm.corrected_positions)
                deformation = ts.compute_F(arm.corrected_positions, self.projection.tets, self.projection.J)
                determinant = torch.linalg.det(deformation)
                singular_values = torch.linalg.svdvals(deformation)
                self.assertAlmostEqual(arm.corrected_metrics.total_objective, float(components["total"]), places=10)
                self.assertAlmostEqual(
                    arm.corrected_metrics.raw_residual_norm,
                    float(torch.linalg.vector_norm(residual)),
                    places=9,
                )
                self.assertAlmostEqual(
                    arm.corrected_metrics.normalized_residual_norm,
                    float(torch.linalg.vector_norm(residual) / self.objective.residual_scale),
                    places=12,
                )
                self.assertAlmostEqual(arm.corrected_metrics.minimum_determinant, float(determinant.amin()), places=10)
                self.assertAlmostEqual(
                    arm.corrected_metrics.minimum_singular_value,
                    float(singular_values.amin()),
                    places=10,
                )
                self.assertTrue(torch.equal(arm.corrected_positions[self.projection.pinned], targets))
                correction = arm.corrected_positions - arm.pre_corrector_positions
                self.assertAlmostEqual(arm.correction_norm, float(torch.linalg.vector_norm(correction)), places=12)
                self.assertAlmostEqual(
                    arm.corrected_displacement_from_persistence_norm,
                    float(torch.linalg.vector_norm(arm.corrected_positions - persistence)),
                    places=12,
                )
                if not arm.corrector_trace.accepted:
                    self.assertTrue(arm.fallback_preserved_start)
                    self.assertTrue(torch.equal(arm.corrected_positions, arm.pre_corrector_positions))
        learned = result.arm(LEARNED_ARM)
        self.assertIs(learned.learned_displacement_retention, learned.initializer_displacement_retention)

    def test_learned_zero_comparison_reports_deltas_without_quality_claim(self) -> None:
        result = self._run()
        comparison = compare_ablation_arms(result, LEARNED_ARM, ZERO_ARM)
        learned = result.arm(LEARNED_ARM)
        zero = result.arm(ZERO_ARM)
        self.assertEqual(comparison.left_arm, LEARNED_ARM)
        self.assertEqual(comparison.right_arm, ZERO_ARM)
        self.assertIsNone(comparison.quality_verdict)
        self.assertIn("no-quality-verdict", comparison.claim_scope)
        self.assertAlmostEqual(
            comparison.corrected_total_objective_delta,
            learned.corrected_metrics.total_objective - zero.corrected_metrics.total_objective,
        )
        self.assertAlmostEqual(
            comparison.corrected_raw_residual_norm_delta,
            learned.corrected_metrics.raw_residual_norm - zero.corrected_metrics.raw_residual_norm,
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare_ablation_arms(result, LEARNED_ARM, LEARNED_ARM)
        with self.assertRaisesRegex(ValueError, "mandatory canonical row"):
            dataclasses.replace(result, arms=(result.arms[0], *result.arms[:-1]))

    def test_breakdown_and_safeguard_fallback_preserve_every_start(self) -> None:
        fallback_config = CorrectorConfig(
            pcg=FixedPCGConfig(iterations=2, curvature_relative_tolerance=2.0),
            candidate_alphas=(0.5, 0.0),
            minimum_singular_value=10.0,
        )
        result = self._run(corrector_config=fallback_config)
        self.assertTrue(any(arm.corrector_trace.pcg.breakdown for arm in result.arms))
        for arm in result.arms:
            with self.subTest(arm=arm.name):
                self.assertFalse(arm.corrector_trace.accepted)
                self.assertTrue(arm.fallback_preserved_start)
                self.assertEqual(
                    arm.pre_corrector_metrics.positions_sha256,
                    arm.corrected_metrics.positions_sha256,
                )
                self.assertTrue(torch.equal(arm.pre_corrector_positions, arm.corrected_positions))
                self.assertEqual(arm.correction_norm, 0.0)

    def test_result_reauthentication_rejects_tensor_metric_work_and_label_tamper(self) -> None:
        result = self._run()
        result.validate_immutable()

        tensor_tamper = copy.deepcopy(result)
        tensor_tamper.arms[0].pre_corrector_positions[self.projection.free[0], 0] += 1.0e-5
        with self.assertRaisesRegex(RuntimeError, "positions changed"):
            tensor_tamper.validate_immutable()

        metric_tamper = copy.deepcopy(result)
        object.__setattr__(
            metric_tamper.arms[0].pre_corrector_metrics,
            "total_objective",
            metric_tamper.arms[0].pre_corrector_metrics.total_objective + 1.0,
        )
        with self.assertRaises(RuntimeError):
            metric_tamper.validate_immutable()

        work_tamper = copy.deepcopy(result)
        work = work_tamper.arms[0].corrector_trace.work
        object.__setattr__(work, "active_pcg_iterations", work.active_pcg_iterations + 1)
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            work_tamper.validate_immutable()

        candidate_tamper = copy.deepcopy(result)
        trace = candidate_tamper.arms[0].corrector_trace
        candidate_index = 0 if trace.selected_candidate_index != 0 else 1
        candidate = trace.candidates[candidate_index]
        direction = -math.inf if candidate.objective_nonincreasing else math.inf
        object.__setattr__(candidate, "objective", math.nextafter(candidate.objective, direction))
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            candidate_tamper.validate_immutable()

        pcg_tamper = copy.deepcopy(result)
        pcg = pcg_tamper.arms[0].corrector_trace.pcg
        object.__setattr__(pcg, "initial_residual_norm", math.nextafter(pcg.initial_residual_norm, math.inf))
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            pcg_tamper.validate_immutable()

        timing_tamper = copy.deepcopy(result)
        timing = timing_tamper.arms[0].timing
        object.__setattr__(timing, "corrector_seconds", timing.corrector_seconds + 1.0e-6)
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            timing_tamper.validate_immutable()

        relabel_tamper = copy.deepcopy(result)
        object.__setattr__(relabel_tamper.arms[0], "name", ZERO_ARM)
        with self.assertRaisesRegex(ValueError, "mandatory canonical row"):
            relabel_tamper.validate_immutable()

        scope_tamper = copy.deepcopy(result)
        object.__setattr__(scope_tamper, "dat_scope", "verified-dat")
        with self.assertRaisesRegex(ValueError, "checkpoint or DAT limitation"):
            scope_tamper.validate_immutable()

        operator_tamper = copy.deepcopy(result)
        object.__setattr__(operator_tamper, "operator_geometry_sha256", _digest("wrong-operator"))
        with self.assertRaisesRegex(RuntimeError, "operator_geometry_sha256"):
            operator_tamper.validate_immutable()

        comparison_tamper = copy.deepcopy(result)
        object.__setattr__(
            comparison_tamper.arms[0].corrected_metrics,
            "raw_residual_norm",
            comparison_tamper.arms[0].corrected_metrics.raw_residual_norm + 1.0,
        )
        with self.assertRaises(RuntimeError):
            compare_ablation_arms(comparison_tamper, LEARNED_ARM, ZERO_ARM)

        class _ResultSubclass(V5AblationResult):
            pass

        subclass_result = _ResultSubclass(
            **{field.name: getattr(result, field.name) for field in dataclasses.fields(result)}
        )
        with self.assertRaisesRegex(TypeError, "V5AblationResult"):
            compare_ablation_arms(subclass_result, LEARNED_ARM, ZERO_ARM)

        float32_reference = torch.zeros((), dtype=torch.float32)
        boundary = float(float32_reference.new_tensor(0.1))
        boundary_config = dataclasses.replace(result.corrector_config, minimum_determinant=0.1)
        boundary_candidate = dataclasses.replace(
            result.arms[0].corrector_trace.candidates[0],
            minimum_determinant=boundary,
        )
        determinant_valid, *_ = _replay_candidate_safeguards(
            boundary_candidate,
            result.arms[0].corrector_trace,
            boundary_config,
            float32_reference,
        )
        self.assertFalse(determinant_valid)

    def test_predictor_execution_surface_rejects_child_train_and_hooks(self) -> None:
        child = next(module for name, module in self.predictor.named_modules() if name)
        child.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            self._run()
        self.predictor.eval()

        handle = child.register_forward_hook(lambda _module, _inputs, output: output)
        try:
            with self.assertRaisesRegex(ValueError, "active _forward_hooks hook"):
                self._run()
        finally:
            handle.remove()

    def test_attestation_permutation_and_moving_pin_tamper_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one implicit step and one solver sweep"):
            VBDK1MethodRecord(source_run_sha256=_digest("bad-sweep"), solver_sweeps=2)

        wrong_common = self._vbd_attestation(common_objective_sha256=_digest("wrong-common"))
        with self.assertRaisesRegex(ValueError, "common_objective_sha256"):
            self._run(vbd_k1=wrong_common)

        wrong_operator = self._vbd_attestation(operator_geometry_sha256=_digest("wrong-operator"))
        with self.assertRaisesRegex(ValueError, "operator_geometry_sha256"):
            self._run(vbd_k1=wrong_operator)

        tampered = self._vbd_attestation()
        object.__getattribute__(tampered, "positions")[self.projection.free[0], 0] += 1.0e-3
        with self.assertRaisesRegex(RuntimeError, "positions changed"):
            self._run(vbd_k1=tampered)

        identity_config = dataclasses.replace(self.config, head_permutation=tuple(range(self.tets.shape[0])))
        with self.assertRaisesRegex(ValueError, "nonidentity"):
            self._run(config=identity_config)
        with self.assertRaisesRegex(TypeError, "AttestedVBDK1Start"):
            self._run(vbd_k1=self.physical_step.x_current)

        changed_targets = self.physical_step.pinned_targets.clone()
        changed_targets[0, 0] += 1.0e-4
        moving_pin_mismatch = dataclasses.replace(self.physical_step, pinned_targets=changed_targets)
        with self.assertRaisesRegex(ValueError, "persistence.*exact current pinned"):
            self._run(physical_step=moving_pin_mismatch)


if __name__ == "__main__":
    unittest.main()
