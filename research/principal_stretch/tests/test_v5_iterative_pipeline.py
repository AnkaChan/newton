# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Representation and orchestration tests for iterative principal-stretch v5."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import numpy as np
import torch

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
from research.principal_stretch.v5_objective import CommonObjectiveContext, common_objective_residual


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
