# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for differentiable physical windows around learned inner solves."""

import unittest
from unittest.mock import patch

import numpy as np
import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.physical_rollout import PhysicalRollout
from experiments.learned_intrinsic_solver.unrolled_solver import UnrolledHexSolver


class TestPhysicalRollout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.rest = generate_cuboid((1, 1, 2), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.model = build_newton_hex_model(
            self.rest,
            self.fixed,
            lame_lambda=576.9230769,
            lame_mu=384.6153846,
            density=100,
            gravity=(0, -9.81, 0),
        )
        network = IntrinsicSolverNetwork(
            self.rest.cell_counts,
            features.STATE_FEATURE_DIM,
            conditioning_dim=features.CONDITIONING_DIM,
            hidden_dim=16,
            edge_hidden_dim=8,
        )
        with torch.no_grad():
            network.correction_head.weight.normal_(std=1e-4)
        self.solver = SolverLearnedIntrinsic(self.model, network=network, iterations=2)
        self.dt = 1 / 300
        self.step = self.solver._step_for_dt(self.dt)
        self.rollout = PhysicalRollout(self.solver, UnrolledHexSolver(self.step), time_step=self.dt)
        x = self.rest.corner_rest_positions.astype(np.float32).copy()
        x[:, 0] += np.float32(0.02) * x[:, 2] ** 2
        v = np.zeros_like(x)
        v[:, 0] = np.float32(0.05) * x[:, 2]
        f = np.zeros_like(x)
        f[:, 1] = np.float32(0.0007)
        self.x = torch.from_numpy(x)[None]
        self.v = torch.from_numpy(v)[None]
        self.f = torch.from_numpy(f)[None]

    def test_two_steps_match_native_detached_forward_and_keep_original_pins(self):
        state = self.model.state()
        state.particle_q.assign(self.x[0].numpy())
        state.particle_qd.assign(self.v[0].numpy())
        state.particle_f.assign(self.f[0].numpy())
        native = []
        history = None
        for _ in range(2):
            result = self.solver.predict(state, self.dt, history=history)
            history = result.history
            native.append(result)
            state.particle_q.assign(result.positions.detach().numpy()[0])
            state.particle_qd.assign(result.velocities.detach().numpy()[0])
        windows = list(
            self.rollout.windows(self.x, self.v, forces=self.f, physical_steps=2, iterations=2, gradient_window=2)
        )
        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(len(window.steps), 2)
        for got, expected in zip(window.steps, native, strict=True):
            torch.testing.assert_close(got.inertial_prediction, expected.inertial_prediction, rtol=0, atol=2e-8)
            torch.testing.assert_close(got.initial_candidate, expected.initial_positions, rtol=0, atol=2e-8)
            torch.testing.assert_close(got.positions, expected.positions, rtol=0, atol=2e-8)
            torch.testing.assert_close(got.velocities, expected.velocities, rtol=0, atol=2e-8)
            torch.testing.assert_close(got.positions[:, self.fixed], self.x[:, self.fixed], rtol=0, atol=0)
            torch.testing.assert_close(got.velocities[:, self.fixed], torch.zeros_like(got.velocities[:, self.fixed]))
            self.assertEqual(got.energies.shape, (1, 3))
            torch.testing.assert_close(
                got.history.axis_gradient_world, expected.history.axis_gradient_world, rtol=0, atol=2e-8
            )
        self.assertTrue(window.objective.requires_grad)

    def test_history_is_carried_across_physical_steps_and_cleared_for_a_new_rollout(self):
        """Consume each query's detached history in the next query, across steps and windows, until a new rollout."""
        records = []
        original = self.step.forward

        def recording(*args, **kwargs):
            result = original(*args, **kwargs)
            records.append((kwargs.get("history"), result))
            return result

        with patch.object(self.step, "forward", recording):
            windows = list(self.rollout.windows(self.x, self.v, forces=self.f, physical_steps=2, iterations=2))
        self.assertEqual(len(records), 4)
        self.assertIsNone(records[0][0])
        for index in range(1, 4):
            consumed, previous = records[index][0], records[index - 1][1]
            self.assertEqual(consumed.valid.tolist(), [True])
            self.assertFalse(consumed.axis_gradient_world.requires_grad)
            torch.testing.assert_close(consumed.axis_gradient_world, previous.axis_gradient_world, rtol=0, atol=0)
            torch.testing.assert_close(consumed.axis_update_world, previous.achieved_axis_update_world, rtol=0, atol=0)
        # Step 2 (records[2]) consumed the history exposed by step 1, i.e. it crossed the timestep boundary.
        torch.testing.assert_close(
            records[2][0].axis_gradient_world, windows[0].steps[0].history.axis_gradient_world, rtol=0, atol=0
        )
        records.clear()
        with patch.object(self.step, "forward", recording):
            resumed = next(
                self.rollout.windows(
                    windows[0].final_positions.detach(),
                    windows[0].final_velocities.detach(),
                    forces=self.f,
                    physical_steps=1,
                    iterations=2,
                    history=windows[0].steps[0].history,
                )
            )
            fresh = next(self.rollout.windows(self.x, self.v, forces=self.f, physical_steps=1, iterations=2))
        torch.testing.assert_close(resumed.steps[0].positions, windows[1].steps[0].positions, rtol=0, atol=0)
        self.assertIsNotNone(records[0][0])
        self.assertIsNone(records[2][0])
        self.assertIsNone(fresh.steps[0].previous_positions.grad)

    def test_default_trains_each_timestep_without_resetting_physical_state(self):
        x = self.x.clone().requires_grad_()
        v = self.v.clone().requires_grad_()
        optimizer = torch.optim.SGD(self.step.network.parameters(), lr=1e-6)
        iterator = self.rollout.windows(x, v, physical_steps=2, iterations=2)
        first = next(iterator)
        self.assertEqual((first.start_step, first.end_step), (0, 1))
        first.objective.backward()
        first_x_gradient, first_v_gradient = x.grad.clone(), v.grad.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second = next(iterator)
        self.assertEqual((second.start_step, second.end_step), (1, 2))
        previous = second.steps[0]
        self.assertFalse(previous.previous_positions.requires_grad)
        self.assertFalse(previous.previous_velocities.requires_grad)
        torch.testing.assert_close(previous.previous_positions, first.final_positions.detach(), rtol=0, atol=0)
        torch.testing.assert_close(previous.previous_velocities, first.final_velocities.detach(), rtol=0, atol=0)
        second.objective.backward()
        torch.testing.assert_close(x.grad, first_x_gradient, rtol=0, atol=0)
        torch.testing.assert_close(v.grad, first_v_gradient, rtol=0, atol=0)
        gradients = [parameter.grad for parameter in self.step.network.parameters() if parameter.grad is not None]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.square().sum().item() for gradient in gradients), 0)
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_two_step_window_retains_cross_step_gradient_and_cuts_only_at_boundary(self):
        x = self.x.clone().requires_grad_()
        v = self.v.clone().requires_grad_()
        iterator = self.rollout.windows(x, v, forces=self.f, physical_steps=3, iterations=1, gradient_window=2)
        first = next(iterator)
        self.assertEqual((first.start_step, first.end_step), (0, 2))
        intermediate = first.steps[0].positions
        cross_step = torch.autograd.grad(first.steps[1].energies[:, -1].sum(), intermediate, retain_graph=True)[0]
        self.assertGreater(cross_step.norm().item(), 0)
        first.objective.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.norm().item(), 0)
        second = next(iterator)
        self.assertEqual((second.start_step, second.end_step), (2, 3))
        self.assertFalse(second.steps[0].previous_positions.requires_grad)
        torch.testing.assert_close(second.steps[0].previous_positions, first.final_positions.detach())
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_streamed_training_matches_local_gradients_and_advances_after_adam(self):
        """Accumulate each solve's local gradients before one Adam update and carry its state."""
        expected = next(self.rollout.windows(self.x, self.v, physical_steps=1, iterations=4))
        expected.objective.backward()
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in self.step.named_parameters()
            if parameter.grad is not None
        }
        optimizer = torch.optim.Adam(self.step.parameters(), lr=1e-6)
        optimizer.zero_grad(set_to_none=True)
        iterator = self.rollout.windows(
            self.x.clone().requires_grad_(),
            self.v.clone().requires_grad_(),
            physical_steps=2,
            iterations=4,
            backward_each_iteration=True,
        )
        first = next(iterator)
        torch.testing.assert_close(first.objective, expected.objective, rtol=0, atol=0)
        torch.testing.assert_close(first.final_positions, expected.final_positions, rtol=0, atol=0)
        self.assertFalse(first.objective.requires_grad)
        self.assertFalse(first.final_positions.requires_grad)
        self.assertFalse(first.final_velocities.requires_grad)
        for name, parameter in self.step.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-5, atol=1e-7)
        self.assertEqual(len(optimizer.state), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second = next(iterator)
        torch.testing.assert_close(second.steps[0].previous_positions, first.final_positions, rtol=0, atol=0)
        torch.testing.assert_close(second.steps[0].previous_velocities, first.final_velocities, rtol=0, atol=0)
        self.assertGreater(sum(p.grad.square().sum().item() for p in self.step.parameters() if p.grad is not None), 0)
        optimizer.step()
        self.assertTrue(all(state["step"].item() == 2 for state in optimizer.state.values()))
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_streamed_training_rejects_connected_physical_windows(self):
        """Reject streaming backward across more than one physical timestep."""
        with self.assertRaisesRegex(ValueError, "gradient_window"):
            next(
                self.rollout.windows(self.x, self.v, physical_steps=2, gradient_window=2, backward_each_iteration=True)
            )

    def test_detached_energy_target_preserves_network_feature_path(self):
        x = self.x.clone().requires_grad_()
        v = self.v.clone().requires_grad_()
        result = next(self.rollout.windows(x, v, physical_steps=1, iterations=1))
        self.assertTrue(result.steps[0].inertial_prediction.requires_grad)
        self.assertTrue(result.steps[0].positions.requires_grad)
        gradient = torch.autograd.grad(result.steps[0].positions.sum(), v, allow_unused=True)[0]
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.norm().item(), 0)

    def test_batched_rigid_guidance_matches_independent_problems(self):
        x2 = self.x.clone()
        v2 = self.v.clone()
        free = np.setdiff1d(np.arange(x2.shape[1]), self.fixed)
        x2[:, free, 0] += 0.0003
        v2[:, free, 1] += 0.002
        positions = torch.cat((self.x, x2))
        velocities = torch.cat((self.v, v2))
        forces = torch.cat((self.f, self.f * 2))
        batched = next(self.rollout.windows(positions, velocities, forces=forces, physical_steps=1))
        singles = [
            next(
                self.rollout.windows(
                    positions[i : i + 1], velocities[i : i + 1], forces=forces[i : i + 1], physical_steps=1
                )
            )
            for i in range(2)
        ]
        for field in ("inertial_prediction", "initial_candidate", "positions", "velocities"):
            actual = getattr(batched.steps[0], field)
            expected = torch.cat([getattr(single.steps[0], field) for single in singles])
            torch.testing.assert_close(actual, expected, rtol=0, atol=3e-8)

    def test_rejects_invalid_lengths_and_mismatched_tensor_metadata(self):
        with self.assertRaisesRegex(ValueError, "physical_steps"):
            list(self.rollout.windows(self.x, self.v, physical_steps=0))
        with self.assertRaisesRegex(ValueError, "gradient_window"):
            list(self.rollout.windows(self.x, self.v, physical_steps=1, gradient_window=0))
        with self.assertRaisesRegex(ValueError, "iterations"):
            list(self.rollout.windows(self.x, self.v, physical_steps=1, iterations=33))
        with self.assertRaisesRegex(ValueError, "forces"):
            list(self.rollout.windows(self.x, self.v, physical_steps=1, forces=self.f.double()))


if __name__ == "__main__":
    unittest.main()
