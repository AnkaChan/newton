# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for differentiable physical windows around learned inner solves."""

import unittest

import numpy as np
import torch  # noqa: TID253

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
        network = IntrinsicSolverNetwork(self.rest.cell_counts, 38, hidden_dim=16, edge_hidden_dim=8)
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
        for _ in range(2):
            result = self.solver.predict(state, self.dt)
            native.append(result)
            state.particle_q.assign(result.positions.detach().numpy()[0])
            state.particle_qd.assign(result.velocities.detach().numpy()[0])
        windows = list(self.rollout.windows(self.x, self.v, forces=self.f, physical_steps=2, iterations=2))
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
        self.assertTrue(window.objective.requires_grad)

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
