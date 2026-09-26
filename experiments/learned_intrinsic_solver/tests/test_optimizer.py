# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check repeated learned directions for one fixed physical timestep objective."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features, input_assembly
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic


class TestLearnedOptimizer(unittest.TestCase):
    def setUp(self):
        """Prepare reproducible candidates with a fixed end and nonzero physical motion."""
        torch.manual_seed(225)
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
        x = self.state.particle_q.numpy()
        x[:, 0] += 0.07 * x[:, 2] ** 2
        velocity = np.zeros_like(x)
        velocity[:, 0] = 0.3 * x[:, 2]
        self.state.particle_q.assign(x)
        self.state.particle_qd.assign(velocity)

    def _solver(self, *, nonzero=True):
        network = IntrinsicSolverNetwork(
            self.rest.cell_counts,
            features.STATE_FEATURE_DIM,
            conditioning_dim=features.CONDITIONING_DIM,
            hidden_dim=16,
            edge_hidden_dim=8,
        )
        if nonzero:
            with torch.no_grad():
                network.correction_head.weight.normal_(std=0.002)
        return SolverLearnedIntrinsic(self.model, network=network, iterations=3)

    def test_arbitrary_candidate_query_does_not_redefine_problem(self):
        """Evaluate a supplied feasible candidate without resetting Y or advancing rigid motion again."""
        solver = self._solver(nonzero=False)
        problem = solver.prepare_problem(self.state, 0.01)
        candidate = problem.previous_positions.clone()
        candidate[..., 1] += 0.04 * candidate[..., 2].square()
        saved_y = problem.inertial_prediction.clone()
        update = solver.propose_update(candidate, problem)
        torch.testing.assert_close(update.current_positions, candidate, rtol=0, atol=0)
        torch.testing.assert_close(update.positions, candidate, rtol=0, atol=0)
        torch.testing.assert_close(update.direction, torch.zeros_like(candidate), rtol=0, atol=0)
        torch.testing.assert_close(problem.inertial_prediction, saved_y, rtol=0, atol=0)
        initialized = solver.initialize_candidate(problem)
        self.assertGreater((initialized - problem.previous_positions).norm().item(), 1e-5)
        solved = solver.solve(problem, initial_positions=candidate)
        torch.testing.assert_close(solved.positions, candidate, rtol=0, atol=0)

    def test_updates_read_latest_candidate_and_return_actual_direction(self):
        """Feed every fused output to the next invocation of the same network."""
        solver = self._solver()
        problem = solver.prepare_problem(self.state, 0.01)
        result = solver.solve(problem)
        current = result.initial_positions
        self.assertEqual(len(result.updates), 3)
        for update in result.updates:
            torch.testing.assert_close(update.current_positions, current, rtol=0, atol=0)
            torch.testing.assert_close(update.direction, update.positions - current, rtol=0, atol=0)
            torch.testing.assert_close(
                update.direction[:, self.fixed], torch.zeros(1, len(self.fixed), 3), rtol=0, atol=0
            )
            self.assertGreater(update.direction.norm().item(), 0)
            torch.testing.assert_close(update.loss.total, problem.objective(update.positions).total)
            current = update.positions
        self.assertGreater((result.updates[0].direction - result.updates[1].direction).norm().item(), 1e-8)

    def test_final_loss_reaches_every_iteration_and_initial_candidate(self):
        """Backpropagate one final physical loss through all fused updates without detaching iterates."""
        solver = self._solver()
        problem = solver.prepare_problem(self.state, 0.01)
        initial = solver.initialize_candidate(problem).detach().requires_grad_()
        result = solver.solve(problem, initial_positions=initial)
        for update in result.updates:
            update.local_target_axes.retain_grad()
            update.positions.retain_grad()
        result.loss.total.mean().backward()
        for tensor in [initial] + [u.local_target_axes for u in result.updates] + [u.positions for u in result.updates]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.norm().item(), 0)
        self.assertGreater(solver.network.layers[0].edge_val.weight.grad.norm().item(), 0)

    def test_prepared_objectives_survive_interleaved_timesteps(self):
        """Retain the original physical dt after another problem changes the solver cache."""
        solver = self._solver(nonzero=False)
        first = solver.prepare_problem(self.state, 0.01)
        solver.prepare_problem(self.state, 0.03)
        candidate = first.previous_positions.clone()
        candidate[..., 1] += 0.04 * candidate[..., 2].square()
        update = solver.propose_update(candidate, first)
        mass = torch.from_numpy(self.model.particle_mass.numpy())
        expected_inertia = (
            0.5 * (mass[None, :, None] * (candidate - first.inertial_prediction).square()).sum((1, 2)) / 0.01**2
        )
        torch.testing.assert_close(update.loss.inertia, expected_inertia)

    def test_frozen_frame_unroll_gradient_matches_finite_difference(self):
        """Check a three-update parameter derivative while replaying frozen frames, gradient features and history."""
        solver = self._solver()
        problem = solver.prepare_problem(self.state, 0.01)
        initial = solver.initialize_candidate(problem)
        recorded_gradients = []
        original_gradient = input_assembly.objective_gradient

        def record(*args, **kwargs):
            value = original_gradient(*args, **kwargs)
            recorded_gradients.append(value)
            return value

        with patch.object(input_assembly, "objective_gradient", record):
            result = solver.solve(problem, initial_positions=initial)
        self.assertEqual(len(recorded_gradients), 3)
        frames = [update.frames for update in result.updates]
        # Frames, the gradient feature and the consumed history are detached inputs of each query; the
        # finite difference must hold all three sequences fixed to match the frozen-graph derivative.
        histories = [None] + [update.next_history() for update in result.updates[:-1]]
        parameter = solver.network.correction_head.bias
        gradient = torch.autograd.grad(result.loss.total.sum(), parameter)[0]
        direction = torch.tensor([1, -2, 3, -1, 0.5, -0.5, 2, -1, 1], dtype=torch.float32)
        direction /= direction.norm()
        analytical = (gradient * direction).sum().item()
        original = parameter.detach().clone()
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                parameter.copy_(original + sign * 1e-3 * direction)
                current = initial
                replay = iter(recorded_gradients)

                def replay_gradient(*args, _replay=replay, **kwargs):
                    return next(_replay)

                with patch.object(input_assembly, "objective_gradient", replay_gradient):
                    for frame, history in zip(frames, histories, strict=True):
                        current = solver.propose_update(current, problem, frames=frame, history=history).positions
                values.append(problem.objective(current).total.sum().item())
            parameter.copy_(original)
        numerical = (values[0] - values[1]) / 2e-3
        self.assertAlmostEqual(analytical, numerical, delta=max(1e-5, abs(analytical) * 0.01))

    def test_infeasible_candidate_is_not_reported_as_an_optimizer_direction(self):
        """Reject changed prescribed corners instead of hiding boundary projection in a search direction."""
        solver = self._solver()
        problem = solver.prepare_problem(self.state, 0.01)
        candidate = problem.previous_positions.clone()
        candidate[:, self.fixed[0], 0] += 0.01
        with self.assertRaises(ValueError):
            solver.propose_update(candidate, problem)


if __name__ == "__main__":
    unittest.main()
