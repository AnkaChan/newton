# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify Newton Model/State integration with the learned hex optimizer."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

import newton
from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.input_assembly import OptimizerHistory
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic


class TestNewtonLearnedSolver(unittest.TestCase):
    def setUp(self):
        """Create a small deformed clamped hex model with nonzero aggregate rotation."""
        torch.manual_seed(117)
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.model = build_newton_hex_model(
            self.rest,
            self.fixed,
            lame_lambda=1000 * 0.3 / (1.3 * 0.4),
            lame_mu=1000 / 2.6,
            density=100,
            gravity=(0, -9.81, 0),
        )
        self.state = self.model.state()
        self.output = self.model.state()
        x = self.rest.corner_rest_positions.astype(np.float32)
        x[:, 0] += 0.06 * x[:, 2] ** 2
        velocity = np.zeros_like(x)
        velocity[:, 0] = 0.3 * x[:, 2]
        velocity[:, 1] = 0.05 * x[:, 2]
        force = np.zeros_like(x)
        force[:, 2] = np.float32(0.0007)
        self.state.particle_q.assign(x)
        self.state.particle_qd.assign(velocity)
        self.state.particle_f.assign(force)

    def _solver(self, *, iterations=1, nonzero_head=False):
        network = IntrinsicSolverNetwork(
            self.rest.cell_counts,
            features.STATE_FEATURE_DIM,
            conditioning_dim=features.CONDITIONING_DIM,
            hidden_dim=16,
            edge_hidden_dim=8,
        )
        if nonzero_head:
            with torch.no_grad():
                network.correction_head.weight.normal_(std=0.002)
        return SolverLearnedIntrinsic(self.model, network=network, iterations=iterations)

    @staticmethod
    def _record_queries(step):
        """Return a forward wrapper that records (consumed history, history flag column, output) per query."""
        records = []
        original = step.forward

        def recording(*args, **kwargs):
            inputs = step.prepare_inputs(
                args[0], args[1], previous_positions=kwargs["previous_positions"], history=kwargs.get("history")
            )
            result = original(*args, **kwargs)
            records.append((kwargs.get("history"), inputs.state_features[0, :, 60].unique().tolist(), result))
            return result

        return records, recording

    def test_default_network_uses_revised_schema(self):
        """Build the 61/6 default network regardless of damping and reject legacy schemas."""
        solver = SolverLearnedIntrinsic(self.model)
        self.assertEqual(
            (solver.network.state_feature_dim, solver.network.conditioning_dim),
            (features.STATE_FEATURE_DIM, features.CONDITIONING_DIM),
        )
        legacy = SolverLearnedIntrinsic(
            self.model, network=IntrinsicSolverNetwork(self.rest.cell_counts, 38, conditioning_dim=5)
        )
        with self.assertRaisesRegex(ValueError, "schema"):
            legacy.prepare_problem(self.state, 0.01)

    def test_solve_carries_detached_history_across_iterations_and_native_steps(self):
        """Feed each update's detached history to the next query, expose it, and carry it across step() calls."""
        solver = self._solver(iterations=3, nonzero_head=True)
        problem = solver.prepare_problem(self.state, 0.01)
        step = problem.optimizer
        records, recording = self._record_queries(step)
        with patch.object(step, "forward", recording):
            result = solver.solve(problem)
        self.assertEqual(len(records), 3)
        self.assertIsNone(records[0][0])
        self.assertEqual(records[0][1], [0.0])
        for index in (1, 2):
            consumed, flags, _ = records[index]
            previous = records[index - 1][2]
            self.assertEqual(flags, [1.0])
            self.assertEqual(consumed.valid.tolist(), [True])
            self.assertFalse(consumed.axis_gradient_world.requires_grad)
            torch.testing.assert_close(consumed.axis_gradient_world, previous.axis_gradient_world, rtol=0, atol=0)
            torch.testing.assert_close(consumed.axis_update_world, previous.achieved_axis_update_world, rtol=0, atol=0)
        self.assertIsInstance(result.history, OptimizerHistory)
        torch.testing.assert_close(
            result.history.axis_gradient_world, records[2][2].axis_gradient_world, rtol=0, atol=0
        )
        for update in result.updates:
            self.assertEqual(update.force_residual_norm.shape, (1,))
            self.assertEqual(update.tie_mask.shape, (1, 12))
            self.assertIsNotNone(update.achieved_axis_update_world)
        # Without history the second query differs, so the history is a real input.
        alternative = solver.propose_update(result.updates[0].positions.detach(), problem)
        self.assertGreater((alternative.positions - result.updates[1].positions).abs().max().item(), 0)
        # Native step() carries last_result.history into the next physical step; None restarts a trajectory.
        records.clear()
        solver.last_result = None
        with patch.object(step, "forward", recording):
            solver.step(self.state, self.output, None, None, 0.01)
            first_history = solver.last_result.history
            solver.step(self.output, self.state, None, None, 0.01)
        self.assertIsNone(records[0][0])
        consumed = records[3][0]
        torch.testing.assert_close(consumed.axis_gradient_world, first_history.axis_gradient_world, rtol=0, atol=0)
        self.assertEqual(records[3][1], [1.0])
        records.clear()
        solver.last_result = None
        with patch.object(step, "forward", recording):
            solver.predict(self.state, 0.01)
        self.assertIsNone(records[0][0])

    def test_inverted_candidate_is_accepted_end_to_end(self):
        """Solve from a folded candidate: proper frames, finite energies, and a finite differentiable result."""
        solver = self._solver(iterations=2, nonzero_head=True)
        problem = solver.prepare_problem(self.state, 0.01)
        candidate = problem.previous_positions.clone()
        top = torch.from_numpy(self.rest.corner_rest_positions[:, 2] == self.rest.corner_rest_positions[:, 2].max())
        candidate[:, top, 2] = 0.15
        deformation = features.center_deformation(
            candidate, problem.optimizer.cell_corner_indices, problem.optimizer.center_gradients
        )
        self.assertLess(torch.linalg.det(deformation).min().item(), 0)
        update = solver.propose_update(candidate, problem)
        self.assertTrue(torch.isfinite(update.loss.total).all())
        self.assertTrue((torch.linalg.det(update.frames) > 0).all())
        self.assertTrue(torch.isfinite(update.axis_gradient_world).all())
        result = solver.solve(problem, initial_positions=candidate)
        self.assertTrue(torch.isfinite(result.positions).all())
        self.assertTrue(torch.isfinite(result.loss.total).all())
        result.loss.total.sum().backward()
        gradient = solver.network.correction_head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)

    def test_inverted_committed_state_advances_through_native_step(self):
        """Carry a folded committed State through prepare_problem and step() without repair or rejection."""
        solver = self._solver(iterations=2, nonzero_head=True)
        x = self.state.particle_q.numpy().copy()
        top = self.rest.corner_rest_positions[:, 2] == self.rest.corner_rest_positions[:, 2].max()
        x[top, 2] = 0.15
        self.state.particle_q.assign(x)
        step = solver.prepare_problem(self.state, 0.01).optimizer

        def min_det(positions):
            deformation = features.center_deformation(positions, step.cell_corner_indices, step.center_gradients)
            return torch.linalg.det(deformation).min().item()

        self.assertLess(min_det(torch.from_numpy(x)[None]), 0)
        solver.step(self.state, self.output, None, None, 0.01)
        result = solver.last_result
        self.assertTrue(np.isfinite(self.output.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(self.output.particle_qd.numpy()).all())
        np.testing.assert_array_equal(self.output.particle_q.numpy()[self.fixed], x[self.fixed])
        self.assertIsInstance(result.history, OptimizerHistory)
        self.assertTrue(result.history.valid.all())
        # The rigid-fused initializer carries the fold through instead of repairing or shortening it.
        self.assertLess(min_det(result.initial_positions), 0)
        # The timestep boundary carries history produced by an inverted-state solve.
        solver.step(self.output, self.state, None, None, 0.01)
        self.assertTrue(np.isfinite(self.state.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(self.state.particle_qd.numpy()).all())
        self.assertTrue(solver.last_result.history.valid.all())

    def test_native_step_preserves_input_and_commits_once(self):
        """Read Newton state, preserve it, and write only the final positions and velocities."""
        solver = self._solver()
        self.assertIsInstance(solver, newton.solvers.SolverBase)
        saved = [
            value.numpy().copy() for value in (self.state.particle_q, self.state.particle_qd, self.state.particle_f)
        ]
        solver.step(self.state, self.output, self.model.control(), None, 0.01)
        result = solver.last_result
        np.testing.assert_array_equal(self.output.particle_q.numpy(), result.positions.detach().numpy()[0])
        expected_velocity = (self.output.particle_q.numpy() - saved[0]) / np.float32(0.01)
        expected_velocity[self.fixed] = 0
        np.testing.assert_allclose(self.output.particle_qd.numpy(), expected_velocity, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(self.output.particle_q.numpy()[self.fixed], saved[0][self.fixed])
        for actual, wanted in zip(
            (self.state.particle_q, self.state.particle_qd, self.state.particle_f), saved, strict=True
        ):
            np.testing.assert_array_equal(actual.numpy(), wanted)

    def test_original_inertial_prediction_and_runtime_gravity(self):
        """Use original corner motion and forces for Y independently of the rigid target."""
        solver = self._solver()
        self.model.set_gravity((1, -2, 0.5))
        result = solver.predict(self.state, 0.02)
        x, v, f = (a.numpy() for a in (self.state.particle_q, self.state.particle_qd, self.state.particle_f))
        mass = self.model.particle_mass.numpy()
        expected = (
            x + np.float32(0.02) * v + np.float32(0.02) ** 2 * (f / mass[:, None] + np.array([1, -2, 0.5], np.float32))
        )
        np.testing.assert_allclose(result.inertial_prediction.numpy()[0], expected, rtol=1e-6, atol=2e-8)
        total_mass = mass.sum()
        com_velocity = (mass[:, None] * v).sum(0) / total_mass
        expected_v = com_velocity + 0.02 * (f.sum(0) / total_mass + np.array([1, -2, 0.5], np.float32))
        np.testing.assert_allclose(
            result.rigid_prediction.predicted_linear_velocity.numpy(), expected_v, rtol=2e-6, atol=1e-7
        )
        predicted_rigid = (
            x @ result.rigid_prediction.rigid_delta_rotation.numpy()[0].T
            + result.rigid_prediction.rigid_delta_translation.numpy()[0]
        )
        self.assertGreater(np.linalg.norm(expected - predicted_rigid), 1e-5)

    def test_inner_iterations_do_not_repeat_rigid_integration(self):
        """Apply the single physical rigid prediction only once across zero-correction iterations."""
        solver = self._solver(iterations=3)
        result = solver.predict(self.state, 0.01)
        self.assertGreater((result.positions[0] - torch.from_numpy(self.state.particle_q.numpy())).norm().item(), 1e-5)
        self.assertEqual(len(result.updates), 3)
        for update in result.updates[1:]:
            torch.testing.assert_close(update.positions, result.updates[0].positions, rtol=0, atol=0)

    def test_network_gradient_survives_newton_state_write(self):
        """Retain the Torch loss graph while exposing detached positions as Newton State arrays."""
        solver = self._solver(nonzero_head=True)
        solver.step(self.state, self.output, None, None, 0.01)
        solver.last_result.loss.total.mean().backward()
        for parameter in (solver.network.correction_head.weight, solver.network.layers[0].edge_val.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertFalse(solver.last_result.rigid_prediction.rigid_delta_rotation.requires_grad)

    def test_dt_change_and_material_notification_refresh(self):
        """Refresh physical dt and model materials without replacing the trainable network."""
        solver = self._solver()
        first = solver.predict(self.state, 0.01)
        second = solver.predict(self.state, 0.02)
        self.assertFalse(torch.equal(first.inertial_prediction, second.inertial_prediction))
        for name in ("lame_lambda", "lame_mu"):
            coefficient = getattr(self.model.learned_intrinsic, name)
            coefficient.assign(coefficient.numpy() * 2)
        previous_network = solver.network
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        third = solver.predict(self.state, 0.02)
        self.assertIs(solver.network, previous_network)
        torch.testing.assert_close(third.loss.elastic, second.loss.elastic * 2, rtol=2e-4, atol=2e-7)
        torch.testing.assert_close(third.loss.inertia, second.loss.inertia, rtol=2e-4, atol=2e-7)

    def test_invalid_step_does_not_commit_output(self):
        """Reject invalid dt and populated contacts before touching output state."""
        solver = self._solver()
        before = self.output.particle_q.numpy().copy()
        with self.assertRaises(ValueError):
            solver.step(self.state, self.output, None, None, 0)
        contacts = newton.CollisionPipeline(self.model).contacts()
        contacts.soft_contact_count.assign(np.array([1], dtype=np.int32))
        with self.assertRaises(NotImplementedError):
            solver.step(self.state, self.output, None, contacts, 0.01)
        np.testing.assert_array_equal(self.output.particle_q.numpy(), before)


if __name__ == "__main__":
    unittest.main()
