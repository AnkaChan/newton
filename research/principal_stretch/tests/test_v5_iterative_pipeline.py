# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Representation and orchestration tests for iterative principal-stretch v5."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import numpy as np
import torch

from research.principal_stretch import iterative_solver as iterative_solver_module
from research.principal_stretch import torch_solver as ts
from research.principal_stretch.graph_transformer import (
    GraphTransformerConfig,
    _radially_bound_symmetric,
    _radially_bound_vector,
    _skew,
    covariant_observation_frame,
)
from research.principal_stretch.iterative_solver import (
    PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
    ConstraintApplication,
    ConstraintObservation,
    IdentityConstraintHook,
    IterativeSolverConfig,
    PhysicalStepContext,
    ProposalSafeguardConfig,
    SolverVBDStagedFloat32Evidence,
    _validate_config_execution_dtype,
    solve_iterative_principal_stretch,
    validate_physical_objective_integration,
)
from research.principal_stretch.predictor import (
    build_stretch_predictor,
    checkpoint_predictor_config,
    load_stretch_predictor_state,
    predictor_decoder_work,
    resolve_solver_iterations,
)
from research.principal_stretch.spd_log import spd_floor, sym_exp, sym_log
from research.principal_stretch.tests.test_graph_transformer import (
    _chain_mesh,
    _inputs,
    _rotation,
    _tet_poses,
)
from research.principal_stretch.v5_objective import (
    CommonObjectiveContext,
    common_objective_components,
    common_objective_residual,
)


class _HalfStepConstraint:
    """Small differentiable stand-in for DAT's displacement truncation seam."""

    def __init__(self):
        self.prepared: list[int] = []
        self.applied: list[int] = []

    def descriptor(self) -> dict[str, object]:
        return {"schema_version": 1, "kind": "test-half-step", "reference": "current-iterate"}

    def begin_step(
        self,
        positions: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> object:
        del pinned, pinned_targets
        return {"initial": positions}

    def prepare_iteration(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
    ) -> ConstraintObservation:
        self.prepared.append(iteration)
        normal = torch.zeros_like(positions)
        normal[..., 1] = 1.0
        slack = torch.full(positions.shape[:-1], 0.25, dtype=positions.dtype, device=positions.device)
        return ConstraintObservation(
            state=state,
            normal=normal,
            normalized_slack=slack,
            diagnostics={"refreshes": 1},
        )

    def constrain(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
        proposed: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> ConstraintApplication:
        self.applied.append(iteration)
        constrained = positions + 0.5 * (proposed - positions)
        constrained = constrained.index_copy(-2, pinned, pinned_targets)
        return ConstraintApplication(
            state=state,
            positions=constrained,
            diagnostics={"truncation_calls": 1, "minimum_fraction": 0.5},
        )


class _TransientContextMutationConstraint(_HalfStepConstraint):
    """Adversarial hook that plans to restore context after prediction."""

    def __init__(self, mutate, restore):
        super().__init__()
        self._mutate = mutate
        self._restore = restore

    def prepare_iteration(self, state, iteration, positions):
        self._mutate()
        return super().prepare_iteration(state, iteration, positions)

    def constrain(self, state, iteration, positions, proposed, pinned, pinned_targets):
        self._restore()
        return super().constrain(state, iteration, positions, proposed, pinned, pinned_targets)


class _InvalidConstraint(_HalfStepConstraint):
    def __init__(self, *, change_pin: bool):
        super().__init__()
        self.change_pin = change_pin

    def constrain(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
        proposed: torch.Tensor,
        pinned: torch.Tensor,
        pinned_targets: torch.Tensor,
    ) -> ConstraintApplication:
        del positions
        invalid = proposed.clone()
        if self.change_pin:
            invalid[..., pinned[0], 0] += 1.0
        else:
            invalid[..., 0, 0] = torch.nan
        return ConstraintApplication(state=state, positions=invalid, diagnostics={})


class _InvalidObservationConstraint(_HalfStepConstraint):
    def prepare_iteration(
        self,
        state: object,
        iteration: int,
        positions: torch.Tensor,
    ) -> ConstraintObservation:
        observation = super().prepare_iteration(state, iteration, positions)
        return dataclasses.replace(observation, normal=observation.normal.float())


class _RecordingCandidateConstraint(_HalfStepConstraint):
    """Candidate-aware hook that records the fixed schedule in call order."""

    def __init__(self, transform=None):
        super().__init__()
        self.candidates: list[tuple[int, int, float]] = []
        self.candidate_input_states: list[object] = []
        self.candidate_pinned_inputs: list[torch.Tensor] = []
        self.candidate_pinned_target_inputs: list[torch.Tensor] = []
        self.prepared_input_states: list[object] = []
        self.begin_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.transform = transform

    def begin_step(self, positions, pinned, pinned_targets):
        self.begin_inputs = (positions, pinned, pinned_targets)
        return super().begin_step(positions, pinned, pinned_targets)

    def prepare_iteration(self, state, iteration, positions):
        self.prepared_input_states.append(state)
        return super().prepare_iteration(state, iteration, positions)

    def constrain(self, state, iteration, positions, proposed, pinned, pinned_targets):
        raise AssertionError("candidate mode must not call the legacy constrain method")

    def constrain_candidate(
        self,
        state,
        iteration,
        candidate_index,
        candidate_step_fraction,
        positions,
        candidate_positions,
        pinned,
        pinned_targets,
    ):
        self.candidates.append((iteration, candidate_index, candidate_step_fraction))
        self.candidate_input_states.append(state)
        self.candidate_pinned_inputs.append(pinned)
        self.candidate_pinned_target_inputs.append(pinned_targets)
        constrained = candidate_positions
        if self.transform is not None:
            constrained = self.transform(iteration, candidate_index, constrained)
        return ConstraintApplication(
            state={"iteration": iteration, "selected_candidate": candidate_index},
            positions=constrained,
            diagnostics={"candidate_index": candidate_index},
        )


class TestV5PrincipalStretchRepresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest, cls.tets = _chain_mesh(8)
        cls.state = ts.build_solver(
            cls.rest,
            cls.tets,
            _tet_poses(cls.rest, cls.tets),
            np.array([0, 1, 2], dtype=np.int64),
            device=torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        cls.inputs = _inputs(cls.rest, cls.tets)

    def _predictor(self, version: int = 5):
        return build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float32,
            residual=True,
            graph_config=GraphTransformerConfig(
                hidden_dim=32,
                num_heads=4,
                n_levels=5,
                cluster_size=2,
                max_hencky_update=0.2,
                max_rotation_update=0.3,
                architecture_version=version,
            ),
        )

    def _context(self):
        x_iterate = self.inputs[0] + 0.15 * (self.inputs[0] - self.inputs[1])
        x_iterate[self.state.pinned] = self.inputs[0][self.state.pinned]
        residual = torch.zeros_like(x_iterate)
        normal = torch.zeros_like(x_iterate)
        slack = torch.zeros(x_iterate.shape[0], dtype=x_iterate.dtype)
        return x_iterate, residual, normal, slack

    def _predict_update(self, predictor, *, transformed=None, physical_dt=1.0 / 60.0):
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        x_iterate, residual, normal, slack = self._context()
        if transformed is not None:
            x_current, x_previous, x_iterate, force, gravity, residual, normal = transformed
        return predictor.predict_principal_stretch_update(
            self.state,
            x_current,
            x_previous,
            x_iterate,
            force,
            gravity,
            mu,
            lam,
            pin,
            residual,
            normal,
            slack,
            iteration_fraction=0.25,
            physical_dt=physical_dt,
        )

    def test_v5_heads_follow_explicit_bounded_log_stretch_formula(self):
        predictor = self._predictor()
        raw_symmetric = torch.tensor([2.0, -1.0, 0.5, 1.2, -0.7, 0.3])
        raw_axial = torch.tensor([1.5, -0.8, 0.4])
        with torch.no_grad():
            predictor.model.output_head[-1].bias.copy_(raw_symmetric)
            predictor.model.rotation_head[-1].bias.copy_(raw_axial)

        target, delta_h, omega = self._predict_update(predictor)
        x_iterate = self._context()[0]
        observed = ts.compute_F(x_iterate, self.state.tets, self.state.J)
        observed_h = 0.5 * sym_log(spd_floor(observed.transpose(-1, -2) @ observed, lam_min=0.05**2))
        frame = covariant_observation_frame(observed, observed_h)
        expected_delta = _radially_bound_symmetric(
            raw_symmetric.expand(self.tets.shape[0], -1), predictor.model.config.max_hencky_update
        ).double()
        expected_omega = _radially_bound_vector(
            raw_axial.expand(self.tets.shape[0], -1), predictor.model.config.max_rotation_update
        ).double()
        expected = frame @ torch.matrix_exp(_skew(expected_omega)) @ sym_exp(observed_h + expected_delta)

        torch.testing.assert_close(delta_h, expected_delta, rtol=1.0e-7, atol=1.0e-7)
        torch.testing.assert_close(omega, expected_omega, rtol=1.0e-7, atol=1.0e-7)
        torch.testing.assert_close(target, expected, rtol=3.0e-7, atol=3.0e-7)
        self.assertLess(torch.linalg.matrix_norm(delta_h, ord="fro").max().item(), 0.2)
        self.assertLess(torch.linalg.vector_norm(omega, dim=-1).max().item(), 0.3)
        self.assertTrue(torch.equal(delta_h, delta_h.transpose(-1, -2)))
        self.assertGreater(torch.linalg.det(target).min().item(), 0.0)

    def test_zero_heads_are_identity_and_one_projection_recovers_iterate(self):
        predictor = self._predictor()
        target, delta_h, omega = self._predict_update(predictor)
        x_iterate = self._context()[0]
        observed = ts.compute_F(x_iterate, self.state.tets, self.state.J)
        projected = ts.project_deformation_gradient(self.state, target, x_iterate[self.state.pinned])

        torch.testing.assert_close(target, observed, rtol=1.0e-12, atol=1.0e-12)
        self.assertEqual(delta_h.abs().max().item(), 0.0)
        self.assertEqual(omega.abs().max().item(), 0.0)
        torch.testing.assert_close(projected, x_iterate, rtol=2.0e-12, atol=2.0e-12)
        self.assertTrue(torch.equal(projected[self.state.pinned], x_iterate[self.state.pinned]))

    def test_residual_context_is_active_and_full_update_is_se3_equivariant(self):
        predictor = self._predictor()
        generator = torch.Generator().manual_seed(91)
        with torch.no_grad():
            predictor.model.v5_context_encoder[-1].weight.normal_(std=0.04, generator=generator)
            predictor.model.output_head[-1].weight.normal_(std=0.03, generator=generator)
            predictor.model.rotation_head[-1].weight.normal_(std=0.03, generator=generator)
        x_current, x_previous, force, gravity, *_ = self.inputs
        x_iterate, residual, normal, slack = self._context()
        residual[-2] = torch.tensor([0.4, -0.2, 0.3], dtype=residual.dtype)
        normal[-3] = torch.tensor([0.0, 1.0, 0.0], dtype=normal.dtype)
        slack[-3] = -0.1
        base = predictor.predict_principal_stretch_update(
            self.state,
            x_current,
            x_previous,
            x_iterate,
            force,
            gravity,
            *self.inputs[4:],
            residual,
            normal,
            slack,
            iteration_fraction=0.6,
            physical_dt=1.0 / 60.0,
        )[0]
        zero_residual = predictor.predict_principal_stretch_update(
            self.state,
            x_current,
            x_previous,
            x_iterate,
            force,
            gravity,
            *self.inputs[4:],
            torch.zeros_like(residual),
            normal,
            slack,
            iteration_fraction=0.6,
            physical_dt=1.0 / 60.0,
        )[0]
        self.assertGreater((base - zero_residual).abs().max().item(), 1.0e-8)

        rotation = _rotation()
        translation = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64)

        def rotate(value: torch.Tensor) -> torch.Tensor:
            return value @ rotation.T

        transformed = predictor.predict_principal_stretch_update(
            self.state,
            rotate(x_current) + translation,
            rotate(x_previous) + translation,
            rotate(x_iterate) + translation,
            rotate(force),
            rotate(gravity),
            *self.inputs[4:],
            rotate(residual),
            rotate(normal),
            slack,
            iteration_fraction=0.6,
            physical_dt=1.0 / 60.0,
        )[0]
        expected = torch.einsum("ij,tjk->tik", rotation, base)
        torch.testing.assert_close(transformed, expected, rtol=6.0e-6, atol=6.0e-6)

    def test_v5_checkpoint_is_distinct_and_old_schemas_remain_strict(self):
        source = self._predictor()
        config = source.checkpoint_config()
        graph = config["graph_transformer"]
        self.assertEqual(graph["architecture_version"], 5)
        self.assertIn("max_rotation_update", graph)
        self.assertNotIn("max_multiplicative_update", graph)
        self.assertTrue(any(name.startswith("v5_context_encoder.") for name in source.model.state_dict()))

        checkpoint = {"predictor_config": config, "state_dict": source.model.state_dict()}
        self.assertEqual(checkpoint_predictor_config(checkpoint), config)
        rebuilt = self._predictor()
        load_stretch_predictor_state(rebuilt, checkpoint)
        for name, value in source.model.state_dict().items():
            self.assertTrue(torch.equal(value, rebuilt.model.state_dict()[name]), name)

        v3 = self._predictor(3)
        self.assertFalse(any(name.startswith("v5_context_encoder.") for name in v3.model.state_dict()))
        with self.assertRaises(RuntimeError):
            load_stretch_predictor_state(v3, checkpoint)

        malformed = dataclasses.asdict(GraphTransformerConfig(architecture_version=5))
        with self.assertRaisesRegex(ValueError, "max_multiplicative_update"):
            checkpoint_predictor_config(
                {
                    "predictor_config": {"kind": "graph-transformer", "residual": True, "graph_transformer": malformed},
                    "state_dict": source.model.state_dict(),
                }
            )

    def test_v5_requires_explicit_iterative_route_and_accounts_fixed_work(self):
        predictor = self._predictor()
        with self.assertRaisesRegex(ValueError, "explicit"):
            resolve_solver_iterations(predictor, None)
        self.assertEqual(resolve_solver_iterations(predictor, 3), 3)
        self.assertEqual(
            predictor_decoder_work(predictor, 3, 1),
            {
                "schema_version": 3,
                "target": "principal-log-stretch-full-deformation-gradient",
                "decoder": "iterative-weighted-global-projection",
                "predictor_passes": 3,
                "compatibility_projection_calls": 3,
                "local_polar_sweeps": 0,
                "common_residual_evaluations": 4,
                "common_objective_evaluations": 4,
                "state_validity_evaluations": 4,
                "constraint_preparations": 3,
                "constraint_applications": 3,
                "physical_step_authentications": 9,
                "common_objective_authentications": 9,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "iterative"):
            predictor.predict_deformation_gradient(self.state, *self.inputs)

    def test_permuted_head_ablation_reuses_the_bounded_learned_outputs(self):
        predictor = self._predictor()
        generator = torch.Generator().manual_seed(117)
        with torch.no_grad():
            predictor.model.output_head[-1].weight.normal_(std=0.05, generator=generator)
            predictor.model.rotation_head[-1].weight.normal_(std=0.05, generator=generator)
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        x_iterate, residual, normal, slack = self._context()
        learned = predictor.predict_principal_stretch_update(
            self.state,
            x_current,
            x_previous,
            x_iterate,
            force,
            gravity,
            mu,
            lam,
            pin,
            residual,
            normal,
            slack,
            iteration_fraction=0.4,
            physical_dt=1.0 / 60.0,
        )
        permutation = torch.arange(self.tets.shape[0] - 1, -1, -1)
        permuted = predictor.predict_principal_stretch_update(
            self.state,
            x_current,
            x_previous,
            x_iterate,
            force,
            gravity,
            mu,
            lam,
            pin,
            residual,
            normal,
            slack,
            iteration_fraction=0.4,
            physical_dt=1.0 / 60.0,
            head_mode="permuted",
            head_permutation=permutation,
        )
        torch.testing.assert_close(permuted[1], learned[1].index_select(-3, permutation))
        torch.testing.assert_close(permuted[2], learned[2].index_select(-2, permutation))

    def test_v5_batch_matches_independent_iterations(self):
        predictor = self._predictor()
        generator = torch.Generator().manual_seed(90210)
        with torch.no_grad():
            predictor.model.v5_context_encoder[-1].weight.normal_(std=0.04, generator=generator)
            predictor.model.output_head[-1].weight.normal_(std=0.03, generator=generator)
            predictor.model.rotation_head[-1].weight.normal_(std=0.03, generator=generator)
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        x_iterate, residual, normal, slack = self._context()
        offset = 0.003 * torch.sin(torch.arange(x_iterate.shape[0], dtype=x_iterate.dtype))[:, None]
        second_iterate = x_iterate + offset
        second_iterate[self.state.pinned] = x_current[self.state.pinned]
        batches = (
            torch.stack((x_current, x_current)),
            torch.stack((x_previous, x_previous)),
            torch.stack((x_iterate, second_iterate)),
            torch.stack((force, 0.7 * force)),
            torch.stack((gravity, gravity)),
            torch.stack((residual, residual + 0.02)),
            torch.stack((normal, normal)),
            torch.stack((slack, slack)),
        )
        fractions = torch.tensor([0.2, 0.7], dtype=x_current.dtype)
        physical_dt = torch.tensor([1.0 / 60.0, 1.0 / 120.0], dtype=x_current.dtype)
        batched = predictor.predict_principal_stretch_update(
            self.state,
            batches[0],
            batches[1],
            batches[2],
            batches[3],
            batches[4],
            mu,
            lam,
            pin,
            batches[5],
            batches[6],
            batches[7],
            iteration_fraction=fractions,
            physical_dt=physical_dt,
        )
        independent = [
            predictor.predict_principal_stretch_update(
                self.state,
                batches[0][index],
                batches[1][index],
                batches[2][index],
                batches[3][index],
                batches[4][index],
                mu,
                lam,
                pin,
                batches[5][index],
                batches[6][index],
                batches[7][index],
                iteration_fraction=fractions[index],
                physical_dt=physical_dt[index],
            )
            for index in range(2)
        ]
        for output_index, output in enumerate(batched):
            expected = torch.stack([item[output_index] for item in independent])
            torch.testing.assert_close(output, expected, rtol=2.0e-6, atol=2.0e-6)
        self.assertGreater((batched[1][0] - batched[1][1]).abs().max().item(), 1.0e-7)
        self.assertGreater((batched[2][0] - batched[2][1]).abs().max().item(), 1.0e-7)

    def test_rest_repeated_spectrum_has_finite_iterate_gradient(self):
        predictor = self._predictor()
        with torch.no_grad():
            predictor.model.output_head[-1].bias.copy_(torch.tensor([0.02, -0.01, 0.01, 0.003, 0.004, -0.002]))
            predictor.model.rotation_head[-1].bias.copy_(torch.tensor([0.01, -0.005, 0.007]))
        rest = torch.as_tensor(self.rest, dtype=torch.float64)
        x_iterate = rest.clone().requires_grad_(True)
        zeros = torch.zeros_like(rest)
        mu, lam, pin = self.inputs[4:]
        target, _, _ = predictor.predict_principal_stretch_update(
            self.state,
            rest,
            rest,
            x_iterate,
            zeros,
            torch.zeros(3, dtype=rest.dtype),
            mu,
            lam,
            pin,
            zeros,
            zeros,
            torch.zeros(rest.shape[0], dtype=rest.dtype),
            iteration_fraction=0.0,
            physical_dt=1.0 / 60.0,
        )
        target.square().sum().backward()
        self.assertIsNotNone(x_iterate.grad)
        self.assertTrue(torch.isfinite(x_iterate.grad).all())

    def test_v5_velocity_features_use_runtime_physical_dt(self):
        predictor = self._predictor()
        generator = torch.Generator().manual_seed(213)
        with torch.no_grad():
            predictor.model.output_head[-1].weight.normal_(std=0.04, generator=generator)
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        x_iterate, residual, normal, slack = self._context()

        def update(dt):
            return predictor.predict_principal_stretch_update(
                self.state,
                x_current,
                x_previous,
                x_iterate,
                force,
                gravity,
                mu,
                lam,
                pin,
                residual,
                normal,
                slack,
                iteration_fraction=0.3,
                physical_dt=dt,
            )[1]

        at_sixty_hz = update(1.0 / 60.0)
        at_one_twenty_hz = update(1.0 / 120.0)
        self.assertGreater((at_sixty_hz - at_one_twenty_hz).abs().max().item(), 1.0e-9)

        with self.assertRaisesRegex(ValueError, "physical_dt"):
            self._predict_update(predictor, physical_dt=0.0)


class TestV5IterativeSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest, cls.tets = _chain_mesh(5)
        cls.state = ts.build_solver(
            cls.rest,
            cls.tets,
            _tet_poses(cls.rest, cls.tets),
            np.array([0, 1, 2], dtype=np.int64),
            device=torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )
        cls.inputs = _inputs(cls.rest, cls.tets)
        mass = torch.linspace(0.8, 1.2, cls.rest.shape[0], dtype=torch.float64)
        dt = 1.0 / 60.0
        inertial_target = 2.0 * cls.inputs[0] - cls.inputs[1]
        inertial_target = inertial_target + dt * dt * (cls.inputs[3] + cls.inputs[2] / mass[:, None])
        cls.objective = CommonObjectiveContext(
            tets=cls.state.tets,
            J=cls.state.J,
            volume=cls.state.w,
            mass=mass,
            mu=cls.inputs[4].double(),
            lam=cls.inputs[5].double(),
            inertial_target=inertial_target,
            pinned=cls.state.pinned,
            dt=dt,
        )
        cls.physical_step = PhysicalStepContext(
            x_current=cls.inputs[0],
            x_previous=cls.inputs[1],
            force=cls.inputs[2],
            gravity=cls.inputs[3],
            mu=cls.inputs[4].double(),
            lam=cls.inputs[5].double(),
            pin=cls.inputs[6].double(),
            pinned_targets=cls.inputs[0][cls.state.pinned],
        )

    def _predictor(self):
        return build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float32,
            residual=True,
            graph_config=GraphTransformerConfig(
                hidden_dim=16,
                num_heads=4,
                n_levels=3,
                cluster_size=2,
                max_hencky_update=0.12,
                max_rotation_update=0.15,
                architecture_version=5,
            ),
        )

    def _solve(self, predictor, *, iterations=3, constraint=None, **config_kwargs):
        config_kwargs.setdefault("objective_policy", "record")
        config_kwargs.setdefault("residual_policy", "record")
        return solve_iterative_principal_stretch(
            predictor=predictor,
            projection_state=self.state,
            objective=self.objective,
            physical_step=self.physical_step,
            expected_physical_step_sha256=self.physical_step.physical_step_sha256,
            config=IterativeSolverConfig(iterations=iterations, **config_kwargs),
            constraint=constraint,
        )

    def test_zero_head_k_iterations_are_fixed_and_fully_accounted(self):
        result = self._solve(
            self._predictor(),
            iterations=3,
            objective_policy="require-nonincreasing",
            residual_policy="require-nonincreasing",
        )
        torch.testing.assert_close(result.positions, self.inputs[0], rtol=3.0e-12, atol=3.0e-12)
        self.assertTrue(torch.equal(result.positions[self.state.pinned], self.inputs[0][self.state.pinned]))
        self.assertEqual(len(result.trace), 3)
        self.assertEqual(
            dataclasses.asdict(result.work),
            {
                "predictor_passes": 3,
                "projection_calls": 3,
                "residual_evaluations": 4,
                "objective_evaluations": 4,
                "state_validity_evaluations": 4,
                "constraint_preparations": 3,
                "constraint_applications": 3,
                "physical_step_authentications": 9,
                "common_objective_authentications": 9,
                "projection_backend": "dense",
                "projection_diagnostics_recorded": True,
                "projection_iterations": 0,
                "projection_matrix_vector_products": 0,
                "projection_preconditioner_applications": 0,
                "projection_factor_solves": 3,
            },
        )
        self.assertEqual(result.constraint_descriptor["kind"], "identity")
        self.assertEqual(len(result.constraint_descriptor_sha256), 64)
        self.assertEqual(result.constraint_registration, "registered-identity-development")

    def test_candidate_globalization_runs_the_full_schedule_before_accepting_full_step(self):
        predictor = self._predictor()
        current = self.inputs[0]
        raw_residual = common_objective_residual(self.objective, current)
        proposed = current - 1.0e-7 * raw_residual
        proposed = proposed.index_copy(-2, self.state.pinned, current[self.state.pinned])
        constraint = _RecordingCandidateConstraint()
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        self.assertEqual(constraint.candidates, [(0, 0, 1.0), (0, 1, 0.5), (0, 2, 0.0)])
        self.assertEqual(result.trace[0].selected_candidate_index, 0)
        self.assertEqual(result.trace[0].selected_step_fraction, 1.0)
        self.assertTrue(result.trace[0].proposal_accepted)
        self.assertTrue(result.trace[0].learned_contribution_retained)
        torch.testing.assert_close(
            result.trace[0].learned_displacement_retention,
            torch.tensor(1.0, dtype=current.dtype),
        )
        self.assertEqual(
            (
                result.work.predictor_passes,
                result.work.projection_calls,
                result.work.residual_evaluations,
                result.work.objective_evaluations,
                result.work.state_validity_evaluations,
                result.work.constraint_preparations,
                result.work.constraint_applications,
                result.work.physical_step_authentications,
                result.work.common_objective_authentications,
            ),
            (1, 1, 4, 4, 4, 1, 3, 7, 7),
        )

    def test_candidate_globalization_selects_half_step_and_scores_constrained_positions(self):
        predictor = self._predictor()
        current = self.inputs[0]
        raw_residual = common_objective_residual(self.objective, current)
        proposed = current - 1.5e-6 * raw_residual
        proposed = proposed.index_copy(-2, self.state.pinned, current[self.state.pinned])

        def truncate_full(_iteration, candidate_index, candidate_positions):
            if candidate_index == 0:
                return current + 0.5 * (candidate_positions - current)
            return candidate_positions

        constraint = _RecordingCandidateConstraint(transform=truncate_full)
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        iteration = result.trace[0]
        self.assertEqual(iteration.selected_candidate_index, 0)
        self.assertEqual(iteration.selected_step_fraction, 1.0)
        expected = current + 0.5 * (proposed - current)
        torch.testing.assert_close(iteration.positions, expected)
        torch.testing.assert_close(
            iteration.candidate_evaluations[0].objective,
            common_objective_components(self.objective, expected)["total"],
        )
        self.assertNotEqual(
            iteration.candidate_evaluations[0].objective.item(),
            common_objective_components(self.objective, proposed)["total"].item(),
        )
        torch.testing.assert_close(
            iteration.learned_displacement_retention,
            torch.tensor(0.5, dtype=current.dtype),
        )

        identity_result = None
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            identity_result = self._solve(
                predictor,
                iterations=1,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )
        self.assertEqual(identity_result.trace[0].selected_candidate_index, 1)
        self.assertEqual(identity_result.trace[0].selected_step_fraction, 0.5)
        torch.testing.assert_close(identity_result.positions, expected)

    def test_opposite_constraint_motion_does_not_retain_the_learned_displacement(self):
        predictor = self._predictor()
        current = self.inputs[0]
        raw_residual = common_objective_residual(self.objective, current)
        proposed = current - 1.0e-7 * raw_residual
        proposed = proposed.index_copy(-2, self.state.pinned, current[self.state.pinned])

        def reverse_full_candidate(_iteration, candidate_index, candidate_positions):
            if candidate_index == 0:
                return current - (candidate_positions - current)
            return candidate_positions

        constraint = _RecordingCandidateConstraint(transform=reverse_full_candidate)
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        reversed_candidate = result.trace[0].candidate_evaluations[0]
        torch.testing.assert_close(
            reversed_candidate.displacement_retention,
            torch.tensor(-1.0, dtype=current.dtype),
        )
        self.assertFalse(reversed_candidate.learned_contribution_retained)
        self.assertEqual(result.trace[0].selected_step_fraction, 0.5)

    def test_objective_null_state_uses_exact_zero_candidate(self):
        current = self.inputs[0]
        zeros = torch.zeros_like
        zero_material = torch.zeros(self.state.n_tets, dtype=current.dtype)
        physical_step = PhysicalStepContext(
            x_current=current,
            x_previous=current,
            force=zeros(current),
            gravity=torch.zeros(3, dtype=current.dtype),
            mu=zero_material,
            lam=zero_material,
            pin=self.inputs[6].double(),
            pinned_targets=current[self.state.pinned],
        )
        objective = CommonObjectiveContext(
            tets=self.state.tets,
            J=self.state.J,
            volume=self.state.w,
            mass=self.objective.mass,
            mu=zero_material,
            lam=zero_material,
            inertial_target=current,
            pinned=self.state.pinned,
            dt=self.objective.dt,
        )
        proposed = current.clone()
        proposed[-1, 0] += 1.0e-3
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=objective,
                physical_step=physical_step,
                expected_physical_step_sha256=physical_step.physical_step_sha256,
                config=IterativeSolverConfig(
                    iterations=1,
                    proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
                ),
            )

        iteration = result.trace[0]
        self.assertTrue(torch.equal(result.positions, current))
        self.assertEqual(iteration.selected_candidate_index, 2)
        self.assertEqual(iteration.selected_step_fraction, 0.0)
        self.assertFalse(iteration.proposal_accepted)
        self.assertFalse(iteration.learned_contribution_retained)
        self.assertEqual(iteration.selection_reason, "no-admissible-positive")
        self.assertEqual(iteration.learned_displacement_retention.item(), 0.0)
        self.assertEqual(result.proposal_accepted_iterations, 0)
        self.assertEqual(result.zero_step_iterations, 1)
        self.assertEqual(result.learned_contribution_retained_iterations, 0)

    def test_candidate_invalid_values_and_pins_are_rejected_after_fixed_scoring(self):
        predictor = self._predictor()
        current = self.inputs[0]
        raw_residual = common_objective_residual(self.objective, current)
        proposed = current - 1.0e-7 * raw_residual
        proposed = proposed.index_copy(-2, self.state.pinned, current[self.state.pinned])

        def tamper(_iteration, candidate_index, candidate_positions):
            tampered = candidate_positions.clone()
            if candidate_index == 0:
                tampered[-1, 0] = torch.nan
            elif candidate_index == 1:
                tampered[self.state.pinned[0], 0] = torch.nextafter(
                    tampered[self.state.pinned[0], 0],
                    torch.tensor(torch.inf, dtype=tampered.dtype),
                )
            return tampered

        constraint = _RecordingCandidateConstraint(transform=tamper)
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        evaluations = result.trace[0].candidate_evaluations
        self.assertEqual(len(evaluations), 3)
        self.assertIn("non-finite-positions", evaluations[0].rejection_reasons)
        self.assertFalse(evaluations[0].objective_finite)
        self.assertFalse(evaluations[0].residual_finite)
        self.assertIsNone(evaluations[0].displacement_retention)
        self.assertFalse(evaluations[0].learned_contribution_retained)
        self.assertIn("changed-exact-pins", evaluations[1].rejection_reasons)
        self.assertTrue(evaluations[1].objective_finite)
        self.assertTrue(evaluations[1].residual_finite)
        self.assertTrue(evaluations[2].admissible)
        self.assertEqual(result.trace[0].selected_step_fraction, 0.0)
        self.assertEqual(constraint.candidates, [(0, 0, 1.0), (0, 1, 0.5), (0, 2, 0.0)])

    def test_zero_candidate_must_be_a_bitwise_noop_after_all_candidate_calls(self):
        predictor = self._predictor()
        current = self.inputs[0]

        def move_zero(_iteration, candidate_index, candidate_positions):
            moved = candidate_positions.clone()
            if candidate_index == 2:
                moved[-1, 0] = torch.nextafter(
                    moved[-1, 0],
                    torch.tensor(torch.inf, dtype=moved.dtype),
                )
            return moved

        constraint = _RecordingCandidateConstraint(transform=move_zero)
        with (
            mock.patch(
                "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
                return_value=(
                    current,
                    mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "zero-step constraint candidate.*bitwise"),
        ):
            self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )
        self.assertEqual(constraint.candidates, [(0, 0, 1.0), (0, 1, 0.5), (0, 2, 0.0)])

    def test_zero_candidate_rejects_a_signed_zero_bit_flip(self):
        predictor = self._predictor()
        translation = torch.zeros_like(self.inputs[0])
        translation[:, 0] = -self.inputs[0][0, 0]
        current = self.inputs[0] + translation
        positive_zero = (current == 0.0) & ~torch.signbit(current)
        self.assertTrue(positive_zero.any())
        zero_index = tuple(int(index) for index in torch.nonzero(positive_zero, as_tuple=False)[0])
        physical_step = PhysicalStepContext(
            x_current=current,
            x_previous=self.inputs[1] + translation,
            force=self.inputs[2],
            gravity=self.inputs[3],
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            pin=self.inputs[6].double(),
            pinned_targets=current[self.state.pinned],
        )
        objective = CommonObjectiveContext(
            tets=self.state.tets,
            J=self.state.J,
            volume=self.state.w,
            mass=self.objective.mass,
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            inertial_target=self.objective.inertial_target + translation,
            pinned=self.state.pinned,
            dt=self.objective.dt,
        )

        def flip_zero_sign(_iteration, candidate_index, candidate_positions):
            changed = candidate_positions.clone()
            if candidate_index == 2:
                changed[zero_index] = -0.0
            return changed

        constraint = _RecordingCandidateConstraint(transform=flip_zero_sign)
        with (
            mock.patch(
                "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
                return_value=(
                    current,
                    mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "zero-step constraint candidate.*bitwise"),
        ):
            solve_iterative_principal_stretch(
                predictor=predictor,
                projection_state=self.state,
                objective=objective,
                physical_step=physical_step,
                expected_physical_step_sha256=physical_step.physical_step_sha256,
                config=IterativeSolverConfig(
                    iterations=1,
                    objective_policy="require-nonincreasing",
                    residual_policy="require-nonincreasing",
                    proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
                ),
                constraint=constraint,
            )

    def test_candidate_evidence_snapshots_a_reused_constraint_output_buffer(self):
        predictor = self._predictor()
        current = self.inputs[0]
        proposed = current.clone()
        proposed[-1, 0] += 1.0e-3
        buffer = torch.empty_like(current)

        def reuse_buffer(_iteration, _candidate_index, candidate_positions):
            buffer.copy_(candidate_positions)
            return buffer

        constraint = _RecordingCandidateConstraint(transform=reuse_buffer)
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        evaluations = result.trace[0].candidate_evaluations
        self.assertTrue(torch.equal(evaluations[0].constrained_positions, proposed))
        self.assertTrue(torch.equal(evaluations[2].constrained_positions, current))
        self.assertNotEqual(evaluations[0].constrained_positions.data_ptr(), buffer.data_ptr())

    def test_candidate_hook_cannot_mutate_raw_proposal_or_candidate_evidence(self):
        predictor = self._predictor()
        current = self.inputs[0]
        raw_residual = common_objective_residual(self.objective, current)
        proposed = current - 1.0e-7 * raw_residual
        proposed = proposed.index_copy(-2, self.state.pinned, current[self.state.pinned])
        doubled = current + 2.0 * (proposed - current)

        def mutate_full_candidate_in_place(_iteration, candidate_index, candidate_positions):
            if candidate_index == 0:
                candidate_positions.copy_(doubled)
            return candidate_positions

        constraint = _RecordingCandidateConstraint(transform=mutate_full_candidate_in_place)
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed.clone(),
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        iteration = result.trace[0]
        evaluations = iteration.candidate_evaluations
        self.assertTrue(torch.equal(iteration.proposed_positions, proposed))
        self.assertTrue(torch.equal(evaluations[0].candidate_positions, proposed))
        self.assertTrue(torch.equal(evaluations[0].constrained_positions, doubled))
        torch.testing.assert_close(
            evaluations[1].displacement_retention,
            torch.tensor(0.5, dtype=current.dtype),
        )

    def test_candidate_hook_cannot_mutate_iteration_base_or_prior_trace(self):
        predictor = self._predictor()

        class MutatingInputConstraint(_RecordingCandidateConstraint):
            def __init__(self, objective):
                super().__init__()
                self.objective = objective

            def constrain_candidate(
                self,
                state,
                iteration,
                candidate_index,
                candidate_step_fraction,
                positions,
                candidate_positions,
                pinned,
                pinned_targets,
            ):
                self.candidates.append((iteration, candidate_index, candidate_step_fraction))
                self.candidate_input_states.append(state)
                if iteration == 1 and candidate_index == 0:
                    with torch.no_grad():
                        residual = common_objective_residual(self.objective, positions)
                        positions.add_(-1.0e-7 * residual)
                        positions.index_copy_(0, pinned, pinned_targets)
                constrained = candidate_positions.clone()
                if iteration == 1 and candidate_step_fraction > 0.0:
                    constrained[-1, 0] = torch.nan
                return ConstraintApplication(
                    state={"iteration": iteration, "selected_candidate": candidate_index},
                    positions=constrained,
                    diagnostics={"candidate_index": candidate_index},
                )

        constraint = MutatingInputConstraint(self.objective)
        result = self._solve(
            predictor,
            iterations=2,
            constraint=constraint,
            objective_policy="require-nonincreasing",
            residual_policy="require-nonincreasing",
            proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
        )

        first, second = result.trace
        self.assertEqual(second.selected_step_fraction, 0.0)
        self.assertTrue(torch.equal(second.positions_before, first.positions))
        self.assertTrue(torch.equal(second.positions, first.positions))
        self.assertTrue(torch.equal(result.positions, first.positions))
        self.assertNotEqual(second.positions_before.data_ptr(), first.positions.data_ptr())
        self.assertNotEqual(second.positions.data_ptr(), first.positions.data_ptr())
        self.assertNotEqual(result.positions.data_ptr(), second.positions.data_ptr())

    def test_nonfinite_projected_proposal_falls_back_to_exact_zero_at_fixed_work(self):
        predictor = self._predictor()
        proposed = self.inputs[0].clone()
        proposed[-1, 0] = torch.nan
        constraint = _RecordingCandidateConstraint()
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            return_value=(
                proposed,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )
        self.assertTrue(torch.equal(result.positions, self.inputs[0]))
        self.assertEqual(result.trace[0].selected_step_fraction, 0.0)
        self.assertIsNone(result.trace[0].learned_displacement_retention)
        self.assertEqual(result.work.constraint_applications, 3)
        self.assertEqual(result.work.residual_evaluations, 4)

    def test_candidate_iterations_use_same_prepared_state_and_selected_successor(self):
        predictor = self._predictor()
        constraint = _RecordingCandidateConstraint()
        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            side_effect=lambda *_args, **_kwargs: (
                _args[0].rest_q,
                mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=2,
                constraint=constraint,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        self.assertEqual(
            constraint.candidates,
            [(0, 0, 1.0), (0, 1, 0.5), (0, 2, 0.0), (1, 0, 1.0), (1, 1, 0.5), (1, 2, 0.0)],
        )
        self.assertIs(constraint.candidate_input_states[0], constraint.candidate_input_states[1])
        self.assertIs(constraint.candidate_input_states[1], constraint.candidate_input_states[2])
        self.assertIs(constraint.candidate_input_states[3], constraint.candidate_input_states[4])
        self.assertIs(constraint.candidate_input_states[4], constraint.candidate_input_states[5])
        self.assertIs(constraint.prepared_input_states[0], constraint.candidate_input_states[0])
        self.assertIs(constraint.prepared_input_states[1], constraint.candidate_input_states[3])
        self.assertEqual(
            constraint.prepared_input_states[1],
            {"iteration": 0, "selected_candidate": result.trace[0].selected_candidate_index},
        )
        self.assertIsNotNone(constraint.begin_inputs)
        begin_positions, begin_pinned, begin_targets = constraint.begin_inputs
        owned_positions, *_unused, owned_targets = self.physical_step._owned_tensors()
        self.assertNotEqual(begin_positions.data_ptr(), owned_positions.data_ptr())
        self.assertNotEqual(begin_pinned.data_ptr(), self.state.pinned.data_ptr())
        self.assertNotEqual(begin_targets.data_ptr(), owned_targets.data_ptr())
        self.assertNotIn(begin_pinned.data_ptr(), {value.data_ptr() for value in constraint.candidate_pinned_inputs})
        self.assertNotIn(
            begin_targets.data_ptr(),
            {value.data_ptr() for value in constraint.candidate_pinned_target_inputs},
        )
        self.assertEqual(
            len({value.data_ptr() for value in constraint.candidate_pinned_inputs}),
            len(constraint.candidate_pinned_inputs),
        )
        self.assertEqual(
            len({value.data_ptr() for value in constraint.candidate_pinned_target_inputs}),
            len(constraint.candidate_pinned_target_inputs),
        )
        self.assertTrue(all(torch.equal(value, self.state.pinned) for value in constraint.candidate_pinned_inputs))
        self.assertTrue(
            all(
                torch.equal(value, self.physical_step.pinned_targets)
                for value in constraint.candidate_pinned_target_inputs
            )
        )
        self.assertEqual(result.work.predictor_passes, 2)
        self.assertEqual(result.work.projection_calls, 2)
        self.assertEqual(result.work.constraint_applications, 6)
        self.assertEqual(result.work.residual_evaluations, 7)
        self.assertEqual(result.work.physical_step_authentications, 11)

    def test_proposal_safeguard_configuration_fails_closed_on_dtype_and_policy_tamper(self):
        with self.assertRaisesRegex(TypeError, "built-in floats"):
            ProposalSafeguardConfig(candidate_step_fractions=(1.0, np.float64(0.5), 0.0))
        config = IterativeSolverConfig(
            iterations=1,
            proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 1.0e-50, 0.0)),
        )
        with self.assertRaisesRegex(ValueError, "execution dtype"):
            _validate_config_execution_dtype(config, torch.zeros((), dtype=torch.float32))

        safeguard = ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0))
        config = IterativeSolverConfig(iterations=1, proposal_safeguard=safeguard)
        object.__setattr__(safeguard, "selection_policy", "last-admissible")
        with self.assertRaisesRegex(ValueError, "selection_policy"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=self.objective,
                physical_step=self.physical_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=config,
            )

    def test_solver_revalidates_the_exact_config_at_the_execution_boundary(self):
        predictor = self._predictor()

        def solve(config):
            return solve_iterative_principal_stretch(
                predictor=predictor,
                projection_state=self.state,
                objective=self.objective,
                physical_step=self.physical_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=config,
            )

        cases = (
            ("iterations", True, TypeError, "iterations must be an integer"),
            ("minimum_determinant", -1.0, ValueError, "minimum_determinant must be finite and non-negative"),
            ("objective_policy", "record", ValueError, "requires strict objective and residual"),
            ("head_mode", "permuted", ValueError, "permuted head mode requires"),
            ("head_permutation", (0,), ValueError, "only valid with head_mode"),
        )
        for name, value, error, message in cases:
            with self.subTest(name=name):
                config = IterativeSolverConfig(
                    iterations=1,
                    proposal_safeguard=(
                        ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0))
                        if name == "objective_policy"
                        else None
                    ),
                )
                object.__setattr__(config, name, value)
                with self.assertRaisesRegex(error, message):
                    solve(config)

        class DerivedConfig(IterativeSolverConfig):
            pass

        with self.assertRaisesRegex(TypeError, "exact IterativeSolverConfig"):
            solve(DerivedConfig(iterations=1))

        accepted = IterativeSolverConfig(
            iterations=np.int64(1),
            minimum_determinant=np.float64(0.0),
            head_mode="permuted",
            head_permutation=tuple(np.int64(index) for index in range(self.state.n_tets)),
        )
        accepted.validate()

    def test_initial_and_direct_committed_residual_norms_must_be_finite(self):
        real_norm = iterative_solver_module._normalized_residual_norm
        cases = (
            (1, "initial raw residual norm"),
            (2, "initial normalized residual norm"),
            (3, "committed raw residual norm"),
            (4, "committed normalized residual norm"),
        )
        for failing_call, message in cases:
            with self.subTest(message=message):
                call_count = 0

                def inject_nonfinite_norm(residual, failing_call=failing_call):
                    nonlocal call_count
                    call_count += 1
                    result = real_norm(residual)
                    if call_count == failing_call:
                        result = torch.full_like(result, torch.inf)
                    return result

                with (
                    mock.patch.object(
                        iterative_solver_module,
                        "_normalized_residual_norm",
                        side_effect=inject_nonfinite_norm,
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    self._solve(self._predictor(), iterations=1)

    def test_candidate_residual_finite_includes_the_normalized_norm(self):
        predictor = self._predictor()
        current = self.inputs[0]
        raw_residual = common_objective_residual(self.objective, current)
        proposed = current - 1.0e-7 * raw_residual
        proposed = proposed.index_copy(-2, self.state.pinned, current[self.state.pinned])
        real_norm = iterative_solver_module._normalized_residual_norm
        call_count = 0

        def overflow_first_candidate_normalized_norm(residual):
            nonlocal call_count
            call_count += 1
            result = real_norm(residual)
            if call_count == 4:
                result = torch.full_like(result, torch.inf)
            return result

        with (
            mock.patch.object(
                iterative_solver_module,
                "_normalized_residual_norm",
                side_effect=overflow_first_candidate_normalized_norm,
            ),
            mock.patch(
                "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
                return_value=(
                    proposed,
                    mock.Mock(iterations=1, matrix_vector_products=0, preconditioner_applications=0, factor_solves=1),
                ),
            ),
        ):
            result = self._solve(
                predictor,
                iterations=1,
                objective_policy="require-nonincreasing",
                residual_policy="require-nonincreasing",
                proposal_safeguard=ProposalSafeguardConfig(candidate_step_fractions=(1.0, 0.5, 0.0)),
            )

        candidate = result.trace[0].candidate_evaluations[0]
        self.assertFalse(candidate.residual_finite)
        self.assertIn("non-finite-residual", candidate.rejection_reasons)

    def test_config_rejects_an_inversion_permitting_determinant_bound(self):
        default = IterativeSolverConfig(iterations=1)
        self.assertEqual(default.objective_policy, "require-nonincreasing")
        self.assertEqual(default.residual_policy, "require-nonincreasing")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            IterativeSolverConfig(iterations=1, minimum_determinant=-1.0e-6)
        with self.assertRaisesRegex(ValueError, "initializer_policy"):
            IterativeSolverConfig(iterations=1, initializer_policy="accepted-reference")
        with self.assertRaisesRegex(ValueError, "registered small safeguard tolerance"):
            IterativeSolverConfig(iterations=1, objective_increase_tolerance=1.0e300)

        float32 = torch.zeros((), dtype=torch.float32)
        for name in ("minimum_determinant", "minimum_singular_value"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name}.*execution dtype"):
                _validate_config_execution_dtype(
                    IterativeSolverConfig(iterations=1, **{name: 1.0e-300}),
                    float32,
                )

    def test_physical_step_requires_exact_floating_feature_dtype_and_binary_pins(self):
        with self.assertRaisesRegex(ValueError, "mu must share the position dtype"):
            dataclasses.replace(self.physical_step, mu=self.inputs[4])
        invalid_pin = self.physical_step.pin
        invalid_pin[0] = 0.5
        with self.assertRaisesRegex(ValueError, "zero or one"):
            dataclasses.replace(self.physical_step, pin=invalid_pin)

    def test_public_integration_validator_rejects_broadcastable_wrong_tet_shape(self):
        wrong_shape = dataclasses.replace(
            self.physical_step,
            mu=self.physical_step.mu.repeat(2),
            lam=self.physical_step.lam.repeat(2),
            pin=self.physical_step.pin.repeat(2),
        )
        with self.assertRaisesRegex(ValueError, "wrong exact tet shape"):
            validate_physical_objective_integration(self.state, self.objective, wrong_shape)

    def test_solver_vbd_float32_history_and_target_are_authenticated_separately(self):
        dt32 = np.float32(1.0 / 300.0)
        pre_event = self.inputs[0].float()
        velocity = torch.linspace(
            -0.071,
            0.083,
            pre_event.numel(),
            dtype=torch.float32,
        ).reshape_as(pre_event)
        velocity[self.state.pinned] = 0.0
        source_mass = torch.linspace(0.8, 1.2, pre_event.shape[0], dtype=torch.float32)
        inverse_mass = torch.reciprocal(source_mass)
        force32 = self.inputs[2].float()
        gravity32 = self.inputs[3].float()
        pinned_targets32 = pre_event[self.state.pinned].clone()
        pinned_targets32[:, 1] += torch.tensor(0.0125, dtype=torch.float32)
        x_current32 = pre_event.clone()
        x_current32[self.state.pinned] = pinned_targets32
        x_previous32 = pre_event - velocity * torch.tensor(dt32, dtype=torch.float32)
        acceleration32 = gravity32 + force32 * inverse_mass[:, None]
        velocity_new32 = velocity + acceleration32 * torch.tensor(dt32, dtype=torch.float32)
        target32 = x_current32 + velocity_new32 * torch.tensor(dt32, dtype=torch.float32)
        target32[self.state.pinned] = x_current32[self.state.pinned]

        source_evidence = SolverVBDStagedFloat32Evidence(
            source_transition_sha256="a" * 64,
            dt_seconds=float(dt32),
            pre_event_positions=pre_event,
            velocity=velocity,
            mass=source_mass,
            inverse_mass=inverse_mass,
        )
        physical_step = PhysicalStepContext(
            x_current=x_current32.double(),
            x_previous=x_previous32.double(),
            force=force32.double(),
            gravity=gravity32.double(),
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            pin=self.inputs[6].double(),
            pinned_targets=pinned_targets32.double(),
            integration_policy=PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
            source_evidence=source_evidence,
        )
        objective = CommonObjectiveContext(
            tets=self.state.tets,
            J=self.state.J,
            volume=self.state.w,
            mass=source_mass.double(),
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            inertial_target=target32.double(),
            pinned=self.state.pinned,
            dt=float(dt32),
        )

        free = torch.ones(self.state.n_verts, dtype=torch.bool)
        free[self.state.pinned.cpu()] = False
        algebraic = 2.0 * physical_step.x_current - physical_step.x_previous
        algebraic = algebraic + objective.dt**2 * (
            physical_step.gravity + physical_step.force / objective.mass[:, None]
        )
        self.assertGreater((algebraic[free] - objective.inertial_target[free]).abs().max().item(), 1.0e-14)
        validate_physical_objective_integration(self.state, objective, physical_step)
        self.assertEqual(physical_step.source_evidence.evidence_sha256, source_evidence.evidence_sha256)
        self.assertNotEqual(physical_step.physical_step_sha256, self.physical_step.physical_step_sha256)

        changed_velocity = velocity.clone()
        changed_velocity[-1, 0] = torch.nextafter(
            changed_velocity[-1, 0],
            torch.tensor(torch.inf, dtype=torch.float32),
        )
        changed_evidence = dataclasses.replace(source_evidence, velocity=changed_velocity)
        changed_step = dataclasses.replace(physical_step, source_evidence=changed_evidence)
        self.assertNotEqual(changed_step.physical_step_sha256, physical_step.physical_step_sha256)

        changed_velocity[-1, 0] += torch.tensor(0.01, dtype=torch.float32)
        changed_evidence = dataclasses.replace(source_evidence, velocity=changed_velocity)
        changed_step = dataclasses.replace(physical_step, source_evidence=changed_evidence)
        with self.assertRaisesRegex(ValueError, "x_previous|inertial target"):
            validate_physical_objective_integration(self.state, objective, changed_step)

        invalid_inverse_mass = inverse_mass.clone()
        invalid_inverse_mass[-1] = torch.nextafter(
            invalid_inverse_mass[-1],
            torch.tensor(torch.inf, dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, "exact float32 reciprocal"):
            dataclasses.replace(source_evidence, inverse_mass=invalid_inverse_mass)

        tampered_dt = dataclasses.replace(source_evidence)
        object.__setattr__(tampered_dt, "dt_seconds", float(np.float32(1.0 / 150.0)))
        with self.assertRaisesRegex(RuntimeError, "timestep changed"):
            tampered_dt.validate_immutable()

        tampered_dt_type = dataclasses.replace(source_evidence)
        object.__setattr__(tampered_dt_type, "dt_seconds", np.float64(source_evidence.dt_seconds))
        with self.assertRaisesRegex(RuntimeError, "timestep changed type"):
            tampered_dt_type.validate_immutable()

        class StringSubclass(str):
            pass

        tampered_transition_type = dataclasses.replace(source_evidence)
        object.__setattr__(
            tampered_transition_type,
            "source_transition_sha256",
            StringSubclass(source_evidence.source_transition_sha256),
        )
        with self.assertRaisesRegex(RuntimeError, "transition identity changed type"):
            tampered_transition_type.validate_immutable()

        tampered_step = dataclasses.replace(physical_step)
        tampered_step._owned_tensors()[-1].add_(0.125)
        with self.assertRaisesRegex(RuntimeError, "physical-step context changed"):
            validate_physical_objective_integration(self.state, objective, tampered_step)

        tampered_policy_type = dataclasses.replace(physical_step)
        object.__setattr__(
            tampered_policy_type,
            "integration_policy",
            StringSubclass(physical_step.integration_policy),
        )
        with self.assertRaisesRegex(RuntimeError, "integration policy changed type"):
            tampered_policy_type.validate_immutable()

        with self.assertRaisesRegex(ValueError, "promoted float64"):
            PhysicalStepContext(
                x_current=x_current32,
                x_previous=x_previous32,
                force=force32,
                gravity=gravity32,
                mu=self.inputs[4].float(),
                lam=self.inputs[5].float(),
                pin=self.inputs[6].float(),
                pinned_targets=pinned_targets32,
                integration_policy=PHYSICAL_INTEGRATION_POLICY_SOLVER_VBD_STAGED_FLOAT32,
                source_evidence=source_evidence,
            )

    def test_singular_value_and_nonincrease_safeguards_fail_closed(self):
        predictor = self._predictor()
        with self.assertRaisesRegex(RuntimeError, "singular-value bound"):
            self._solve(predictor, iterations=1, minimum_singular_value=2.0)

        scalar = torch.tensor(1.0, dtype=torch.float64)
        with (
            mock.patch(
                "research.principal_stretch.iterative_solver._common_objective_components_trusted",
                side_effect=({"total": scalar}, {"total": scalar + 1.0}),
            ),
            self.assertRaisesRegex(RuntimeError, "increased common objective"),
        ):
            self._solve(predictor, iterations=1, objective_policy="require-nonincreasing")

        zeros = torch.zeros_like(self.physical_step.x_current)
        ones = torch.ones_like(zeros)
        with (
            mock.patch(
                "research.principal_stretch.iterative_solver._common_objective_residual_trusted",
                side_effect=(zeros, ones),
            ),
            self.assertRaisesRegex(RuntimeError, "increased normalized residual norm"),
        ):
            self._solve(predictor, iterations=1, residual_policy="require-nonincreasing")

    def test_result_records_raw_and_normalized_physics_safeguards(self):
        result = self._solve(self._predictor(), iterations=1)
        expected_raw = common_objective_residual(self.objective, result.positions)
        expected_raw_norm = torch.linalg.vector_norm(expected_raw.flatten())

        torch.testing.assert_close(result.raw_residual_norm, expected_raw_norm)
        torch.testing.assert_close(
            result.normalized_residual_norm,
            expected_raw_norm / self.objective.residual_scale,
        )
        self.assertEqual(result.common_objective_sha256, self.objective.common_objective_sha256)
        self.assertGreater(result.minimum_determinant.item(), 0.0)
        self.assertGreater(result.minimum_singular_value.item(), 0.0)

    def test_solver_rejects_features_from_a_different_common_objective(self):
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        mismatched_force = force.clone()
        mismatched_force[-1, 0] += 1.0
        mismatched_step = PhysicalStepContext(
            x_current=x_current,
            x_previous=x_previous,
            force=mismatched_force,
            gravity=gravity,
            mu=mu.double(),
            lam=lam.double(),
            pin=pin.double(),
            pinned_targets=x_current[self.state.pinned],
        )
        with self.assertRaisesRegex(ValueError, "bound inertial target"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=self.objective,
                physical_step=mismatched_step,
                expected_physical_step_sha256=mismatched_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1),
            )

    def test_equal_inertial_target_does_not_authenticate_a_different_history(self):
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        displacement = torch.zeros_like(x_current)
        free = torch.ones(self.state.n_verts, dtype=torch.bool)
        free[self.state.pinned.cpu()] = False
        displacement[free] = torch.tensor([0.002, -0.003, 0.001], dtype=x_current.dtype)
        alternate_step = PhysicalStepContext(
            x_current=x_current + displacement,
            x_previous=x_previous + 2.0 * displacement,
            force=force,
            gravity=gravity,
            mu=mu.double(),
            lam=lam.double(),
            pin=pin.double(),
            pinned_targets=x_current[self.state.pinned],
        )
        alternate_target = 2.0 * alternate_step.x_current - alternate_step.x_previous
        alternate_target = alternate_target + self.objective.dt**2 * (
            alternate_step.gravity + alternate_step.force / self.objective.mass[:, None]
        )
        torch.testing.assert_close(alternate_target[free], self.objective.inertial_target[free], rtol=0.0, atol=2.0e-16)
        self.assertNotEqual(alternate_step.physical_step_sha256, self.physical_step.physical_step_sha256)

        with self.assertRaisesRegex(ValueError, "physical-step identity"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=self.objective,
                physical_step=alternate_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1),
            )

    def test_objective_null_pinned_loads_are_masked_before_learned_features(self):
        predictor = self._predictor()
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        force_with_pinned_loads = force.clone()
        force_with_pinned_loads[self.state.pinned] = torch.tensor(
            [[11.0, -7.0, 3.0], [5.0, 13.0, -2.0], [-17.0, 19.0, 23.0]],
            dtype=force.dtype,
        )
        physical_step = PhysicalStepContext(
            x_current=x_current,
            x_previous=x_previous,
            force=force_with_pinned_loads,
            gravity=gravity,
            mu=mu.double(),
            lam=lam.double(),
            pin=pin.double(),
            pinned_targets=x_current[self.state.pinned],
        )
        with mock.patch.object(
            predictor,
            "predict_principal_stretch_update",
            wraps=predictor.predict_principal_stretch_update,
        ) as predict:
            solve_iterative_principal_stretch(
                predictor=predictor,
                projection_state=self.state,
                objective=self.objective,
                physical_step=physical_step,
                expected_physical_step_sha256=physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1),
            )

        network_force = predict.call_args.args[4]
        self.assertTrue(
            torch.equal(network_force[self.state.pinned], torch.zeros_like(network_force[self.state.pinned]))
        )
        free = torch.ones(self.state.n_verts, dtype=torch.bool)
        free[self.state.pinned.cpu()] = False
        self.assertTrue(torch.equal(network_force[free], force[free]))
        self.assertFalse(torch.equal(force_with_pinned_loads[self.state.pinned], network_force[self.state.pinned]))

    def test_physical_step_public_tensor_mutation_cannot_change_sealed_inputs(self):
        physical_step = dataclasses.replace(self.physical_step)
        expected_current = physical_step.x_current
        expected_force = physical_step.force
        physical_step.x_current.data.fill_(123.0)
        physical_step.force.numpy()[...] = -456.0

        torch.testing.assert_close(physical_step.x_current, expected_current, rtol=0.0, atol=0.0)
        torch.testing.assert_close(physical_step.force, expected_force, rtol=0.0, atol=0.0)
        result = solve_iterative_principal_stretch(
            predictor=self._predictor(),
            projection_state=self.state,
            objective=self.objective,
            physical_step=physical_step,
            expected_physical_step_sha256=physical_step.physical_step_sha256,
            config=IterativeSolverConfig(iterations=1),
        )
        self.assertEqual(result.physical_step_sha256, physical_step.physical_step_sha256)

    def test_solver_reauthenticates_internal_context_bytes_at_the_trust_boundary(self):
        physical_step = dataclasses.replace(self.physical_step)
        physical_step._owned_tensors()[1].add_(0.001)
        with self.assertRaisesRegex(RuntimeError, "physical-step context changed after authentication"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=self.objective,
                physical_step=physical_step,
                expected_physical_step_sha256=physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1),
            )

    def test_hook_cannot_transiently_mutate_then_restore_bound_contexts_around_prediction(self):
        physical_step = dataclasses.replace(self.physical_step)
        physical_mu = physical_step._owned_tensors()[4]
        physical_hook = _TransientContextMutationConstraint(
            lambda: physical_mu.add_(1.0),
            lambda: physical_mu.sub_(1.0),
        )
        with self.assertRaisesRegex(RuntimeError, "physical-step context changed after authentication"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=self.objective,
                physical_step=physical_step,
                expected_physical_step_sha256=physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1, objective_policy="record", residual_policy="record"),
                constraint=physical_hook,
            )

        objective = dataclasses.replace(self.objective)
        objective_mu = objective._owned_tensor("mu")
        objective_hook = _TransientContextMutationConstraint(
            lambda: objective_mu.add_(1.0),
            lambda: objective_mu.sub_(1.0),
        )
        with self.assertRaisesRegex(RuntimeError, "common-objective context changed after authentication"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=objective,
                physical_step=self.physical_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1, objective_policy="record", residual_policy="record"),
                constraint=objective_hook,
            )

        objective = dataclasses.replace(self.objective)
        objective._owned_tensor("inertial_target").add_(0.001)
        with self.assertRaisesRegex(RuntimeError, "common-objective context changed after authentication"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=self.state,
                objective=objective,
                physical_step=self.physical_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1),
            )

    def test_solver_rejects_a_predictor_built_for_a_different_static_mesh(self):
        wrong_tets = self.tets[::-1].copy()
        wrong_predictor = build_stretch_predictor(
            "graph-transformer",
            self.rest,
            wrong_tets,
            torch.device("cpu"),
            torch.float32,
            residual=True,
            graph_config=GraphTransformerConfig(
                hidden_dim=16,
                num_heads=4,
                n_levels=3,
                cluster_size=2,
                max_hencky_update=0.12,
                max_rotation_update=0.15,
                architecture_version=5,
            ),
        )
        with self.assertRaisesRegex(ValueError, "static-mesh"):
            self._solve(wrong_predictor, iterations=1)

    def test_solver_rejects_mutated_projection_and_graph_static_state(self):
        self.assertIsNotNone(self.state.L_ff_chol)
        tampered_factor = self.state.L_ff_chol.clone()
        tampered_factor[0, 0] *= 1.01
        tampered_projection = dataclasses.replace(self.state, L_ff_chol=tampered_factor)
        with self.assertRaisesRegex(ValueError, "projection state"):
            solve_iterative_principal_stretch(
                predictor=self._predictor(),
                projection_state=tampered_projection,
                objective=self.objective,
                physical_step=self.physical_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1),
            )

        tampered_predictor = self._predictor()
        with torch.no_grad():
            tampered_predictor.model.corner_force_weight[0, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "static graph"):
            self._solve(tampered_predictor, iterations=1)

    def test_k1_is_exactly_one_learned_projection_iteration(self):
        predictor = self._predictor()
        with torch.no_grad():
            predictor.model.output_head[-1].bias.copy_(torch.tensor([0.03, -0.01, 0.02, 0.0, 0.005, -0.004]))
            predictor.model.rotation_head[-1].bias.copy_(torch.tensor([0.01, -0.02, 0.015]))
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        residual = common_objective_residual(self.objective, x_current, normalize=True, detach=True)
        target, expected_delta_h, expected_omega = predictor.predict_principal_stretch_update(
            self.state,
            x_current,
            x_previous,
            x_current,
            force,
            gravity,
            mu,
            lam,
            pin,
            residual,
            torch.zeros_like(x_current),
            torch.zeros(x_current.shape[:-1], dtype=x_current.dtype),
            iteration_fraction=0.0,
            physical_dt=self.objective.dt,
        )
        expected = ts.project_deformation_gradient(self.state, target, x_current[self.state.pinned])
        result = self._solve(predictor, iterations=1)

        torch.testing.assert_close(result.positions, expected)
        torch.testing.assert_close(result.trace[0].delta_h, expected_delta_h)
        torch.testing.assert_close(result.trace[0].omega, expected_omega)

    def test_sparse_projection_receives_the_current_iterate_as_warm_start(self):
        predictor = self._predictor()
        sparse_state = ts.build_solver(
            self.rest,
            self.tets,
            _tet_poses(self.rest, self.tets),
            np.array([0, 1, 2], dtype=np.int64),
            device=torch.device("cpu"),
            dtype=torch.float64,
            projection_backend="sparse_pcg",
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
        )

        def project(_state, _target, _pins, **kwargs):
            return kwargs["initial_positions"]

        with mock.patch(
            "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
            side_effect=project,
        ) as projection:
            result = solve_iterative_principal_stretch(
                predictor=predictor,
                projection_state=sparse_state,
                objective=self.objective,
                physical_step=self.physical_step,
                expected_physical_step_sha256=self.physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=2, return_projection_diagnostics=False),
            )

        self.assertEqual(projection.call_count, 2)
        torch.testing.assert_close(
            projection.call_args_list[0].kwargs["initial_positions"],
            self.physical_step.x_current,
        )
        self.assertIs(projection.call_args_list[1].kwargs["initial_positions"], result.trace[0].positions)

    def test_free_body_projection_uses_the_objective_inertial_center_on_both_return_paths(self):
        predictor = self._predictor()
        masses = torch.linspace(0.7, 1.9, self.rest.shape[0], dtype=torch.float64)
        state = ts.build_solver(
            self.rest,
            self.tets,
            _tet_poses(self.rest, self.tets),
            np.empty(0, dtype=np.int64),
            device=torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
            translation_gauge_policy=ts.TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
            vertex_masses=masses.numpy(),
        )
        current = self.inputs[0]
        shift = torch.tensor([0.031, -0.017, 0.023], dtype=torch.float64)
        inertial_target = current + shift
        physical_step = PhysicalStepContext(
            x_current=current,
            x_previous=current - shift,
            force=torch.zeros_like(current),
            gravity=torch.zeros(3, dtype=torch.float64),
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            pin=torch.zeros(self.tets.shape[0], dtype=torch.float64),
            pinned_targets=torch.empty(0, 3, dtype=torch.float64),
        )
        objective = CommonObjectiveContext(
            tets=state.tets,
            J=state.J,
            volume=state.w,
            mass=masses,
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            inertial_target=inertial_target,
            pinned=torch.empty(0, dtype=torch.int64),
            dt=1.0 / 60.0,
        )

        for return_diagnostics in (False, True):
            with self.subTest(return_diagnostics=return_diagnostics):
                with mock.patch(
                    "research.principal_stretch.iterative_solver.torch_solver.project_deformation_gradient",
                    wraps=ts.project_deformation_gradient,
                ) as projection:
                    result = solve_iterative_principal_stretch(
                        predictor=predictor,
                        projection_state=state,
                        objective=objective,
                        physical_step=physical_step,
                        expected_physical_step_sha256=physical_step.physical_step_sha256,
                        config=IterativeSolverConfig(
                            iterations=1,
                            objective_policy="record",
                            residual_policy="record",
                            return_projection_diagnostics=return_diagnostics,
                        ),
                    )

                self.assertIs(
                    projection.call_args.kwargs["center_of_mass_positions"],
                    objective._owned_tensor("inertial_target"),
                )
                weights = state.center_of_mass_weights
                expected_center = torch.einsum("v,vd->d", weights, inertial_target)
                actual_center = torch.einsum("v,vd->d", weights, result.positions)
                torch.testing.assert_close(actual_center, expected_center, rtol=0.0, atol=2.0e-14)

    def test_free_body_projection_requires_exact_objective_mass_weights(self):
        predictor = self._predictor()
        gauge_masses = torch.linspace(0.7, 1.9, self.rest.shape[0], dtype=torch.float64)
        state = ts.build_solver(
            self.rest,
            self.tets,
            _tet_poses(self.rest, self.tets),
            np.empty(0, dtype=np.int64),
            device=torch.device("cpu"),
            dtype=torch.float64,
            operator_geometry_policy=ts.OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
            translation_gauge_policy=ts.TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
            vertex_masses=gauge_masses.numpy(),
        )
        objective_masses = gauge_masses.clone()
        objective_masses[[0, 1]] = objective_masses[[1, 0]]
        current = self.inputs[0]
        objective = CommonObjectiveContext(
            tets=state.tets,
            J=state.J,
            volume=state.w,
            mass=objective_masses,
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            inertial_target=current,
            pinned=torch.empty(0, dtype=torch.int64),
            dt=1.0 / 60.0,
        )
        physical_step = PhysicalStepContext(
            x_current=current,
            x_previous=current,
            force=torch.zeros_like(current),
            gravity=torch.zeros(3, dtype=torch.float64),
            mu=self.inputs[4].double(),
            lam=self.inputs[5].double(),
            pin=torch.zeros(self.tets.shape[0], dtype=torch.float64),
            pinned_targets=torch.empty(0, 3, dtype=torch.float64),
        )

        with self.assertRaisesRegex(ValueError, "center-of-mass weights.*common-objective mass"):
            solve_iterative_principal_stretch(
                predictor=predictor,
                projection_state=state,
                objective=objective,
                physical_step=physical_step,
                expected_physical_step_sha256=physical_step.physical_step_sha256,
                config=IterativeSolverConfig(iterations=1),
            )

    def test_zero_head_ablation_executes_both_heads_at_identical_work(self):
        predictor = self._predictor()
        with torch.no_grad():
            predictor.model.output_head[-1].bias.fill_(0.04)
            predictor.model.rotation_head[-1].bias.fill_(0.03)
        with (
            mock.patch.object(predictor.model.output_head, "forward", wraps=predictor.model.output_head.forward) as h,
            mock.patch.object(
                predictor.model.rotation_head,
                "forward",
                wraps=predictor.model.rotation_head.forward,
            ) as r,
        ):
            result = self._solve(predictor, iterations=2, head_mode="zero")

        self.assertEqual(h.call_count, 2)
        self.assertEqual(r.call_count, 2)
        self.assertEqual(result.head_mode, "zero")
        torch.testing.assert_close(result.positions, self.inputs[0], rtol=3.0e-12, atol=3.0e-12)
        for iteration in result.trace:
            self.assertEqual(iteration.delta_h.abs().max().item(), 0.0)
            self.assertEqual(iteration.omega.abs().max().item(), 0.0)

    def test_constraint_is_between_projection_and_next_residual_aware_pass(self):
        predictor = self._predictor()
        with torch.no_grad():
            predictor.model.output_head[-1].bias.copy_(torch.tensor([0.04, -0.02, 0.01, 0.0, 0.01, -0.01]))
            predictor.model.rotation_head[-1].bias.copy_(torch.tensor([0.03, -0.01, 0.02]))
        constraint = _HalfStepConstraint()
        with mock.patch.object(
            predictor,
            "predict_principal_stretch_update",
            wraps=predictor.predict_principal_stretch_update,
        ) as predict:
            result = self._solve(predictor, iterations=2, constraint=constraint)

        self.assertEqual(result.constraint_registration, "unregistered-custom-no-authenticated-execution")

        self.assertEqual(constraint.prepared, [0, 1])
        self.assertEqual(constraint.applied, [0, 1])
        self.assertEqual(predict.call_count, 2)
        first, second = result.trace
        torch.testing.assert_close(
            first.positions,
            first.positions_before + 0.5 * (first.proposed_positions - first.positions_before),
        )
        torch.testing.assert_close(second.positions_before, first.positions)
        second_iterate = predict.call_args_list[1].args[3]
        torch.testing.assert_close(second_iterate, first.positions)
        physical_current = predict.call_args_list[0].args[1]
        physical_previous = predict.call_args_list[0].args[2]
        for call in predict.call_args_list:
            self.assertIs(call.args[1], physical_current)
            self.assertIs(call.args[2], physical_previous)
        torch.testing.assert_close(physical_current, self.inputs[0])
        torch.testing.assert_close(physical_previous, self.inputs[1])
        expected_second_residual = (
            common_objective_residual(
                self.objective,
                first.positions,
                detach=True,
            )
            / self.objective.residual_scale
        )
        torch.testing.assert_close(predict.call_args_list[1].args[9], expected_second_residual)
        self.assertTrue(torch.equal(result.positions[self.state.pinned], self.inputs[0][self.state.pinned]))

    def test_constraint_pin_or_finite_violation_fails_closed(self):
        predictor = self._predictor()
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self._solve(predictor, iterations=1, constraint=_InvalidConstraint(change_pin=False))
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            self._solve(predictor, iterations=1, constraint=_InvalidConstraint(change_pin=True))
        with self.assertRaisesRegex(RuntimeError, "constraint normal.*dtype"):
            self._solve(predictor, iterations=1, constraint=_InvalidObservationConstraint())

    def test_identity_unroll_is_differentiable_through_both_principal_heads(self):
        predictor = self._predictor()
        generator = torch.Generator().manual_seed(314159)
        with torch.no_grad():
            predictor.model.v5_context_encoder[-1].weight.normal_(std=0.01, generator=generator)
            predictor.model.output_head[-1].weight.normal_(std=0.01, generator=generator)
            predictor.model.rotation_head[-1].weight.normal_(std=0.01, generator=generator)
        result = self._solve(predictor, iterations=2, constraint=IdentityConstraintHook())
        target = self.inputs[0] + torch.linspace(0.0, 0.01, self.rest.shape[0])[:, None]
        loss = (result.positions - target).square().mean()
        loss.backward()
        parameters = (
            predictor.model.v5_context_encoder[0].weight,
            predictor.model.encoders[0][0].weight,
            predictor.model.output_head[0].weight,
            predictor.model.output_head[-1].weight,
            predictor.model.rotation_head[0].weight,
            predictor.model.rotation_head[-1].weight,
        )
        for parameter in parameters:
            gradient = parameter.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
