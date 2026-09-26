# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify CUDA learned optimization with the fixed CPU sparse-solve bridge."""

import copy
import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TestCudaLearnedSolver(unittest.TestCase):
    def setUp(self):
        """Use identical float32 weights and a clamped deformed native state on both devices."""
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.model = build_newton_hex_model(
            self.rest, self.fixed, lame_lambda=600, lame_mu=400, density=100, gravity=(0, -9.81, 0)
        )
        self.state = self.model.state()
        x = self.state.particle_q.numpy()
        x[:, 0] += 0.08 * x[:, 2] ** 2
        self.state.particle_q.assign(x)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(711)
            self.network = IntrinsicSolverNetwork(
                self.rest.cell_counts,
                features.STATE_FEATURE_DIM,
                conditioning_dim=features.CONDITIONING_DIM,
                hidden_dim=16,
                edge_hidden_dim=8,
            )
            with torch.no_grad():
                self.network.correction_head.weight.normal_(std=0.001)

    def test_cuda_unroll_and_native_commit_keep_gradients(self):
        """Keep all Torch optimizer work and its gradients on CUDA across three sparse solves."""
        network = self.network.cuda()
        solver = SolverLearnedIntrinsic(self.model, network=network, iterations=3)
        output = self.model.state()
        solver.step(self.state, output, None, None, 0.01)
        result = solver.last_result
        self.assertTrue(result.positions.is_cuda)
        self.assertTrue(result.loss.total.is_cuda)
        for update in result.updates:
            self.assertTrue(update.frames.is_cuda)
            self.assertTrue(update.local_target_axes.is_cuda)
            update.local_target_axes.retain_grad()
        self.assertTrue(all(buffer.is_cuda for buffer in solver.learned_step.buffers()))
        result.loss.total.sum().backward()
        for update in result.updates:
            self.assertTrue(update.local_target_axes.grad.is_cuda)
            self.assertGreater(update.local_target_axes.grad.norm().item(), 0)
        gradient = network.layers[0].edge_val.weight.grad
        self.assertTrue(gradient.is_cuda)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)
        np.testing.assert_array_equal(output.particle_q.numpy(), result.positions.detach().cpu().numpy()[0])
        np.testing.assert_array_equal(output.particle_q.numpy()[self.fixed], self.state.particle_q.numpy()[self.fixed])

    def test_cuda_matches_cpu_frozen_frame_unroll_and_parameter_gradient(self):
        """Compare the full CUDA forward/adjoint path against the CPU reference with shared frozen frames."""
        cpu = SolverLearnedIntrinsic(self.model, network=self.network, iterations=3)
        gpu = SolverLearnedIntrinsic(self.model, network=copy.deepcopy(self.network).cuda(), iterations=3)
        cpu_problem = cpu.prepare_problem(self.state, 0.01)
        gpu_problem = gpu.prepare_problem(self.state, 0.01)
        cpu_result = cpu.solve(cpu_problem)
        current = cpu_result.initial_positions.detach().cuda()
        history = None
        for update in cpu_result.updates:
            gpu_update = gpu.propose_update(current, gpu_problem, frames=update.frames.cuda(), history=history)
            current = gpu_update.positions
            history = gpu_update.next_history()
        self.assertEqual(cpu_result.history.axis_gradient_world.device.type, "cpu")
        torch.testing.assert_close(
            history.axis_gradient_world.cpu(), cpu_result.history.axis_gradient_world, rtol=2e-4, atol=1e-6
        )
        torch.testing.assert_close(current.cpu(), cpu_result.positions, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(gpu_update.loss.total.cpu(), cpu_result.loss.total, rtol=2e-4, atol=2e-6)
        cpu_result.loss.total.sum().backward()
        gpu_update.loss.total.sum().backward()
        for name in ("correction_head.weight", "layers.0.edge_val.weight"):
            reference = dict(cpu.network.named_parameters())[name].grad
            actual = dict(gpu.network.named_parameters())[name].grad.cpu()
            relative_error = (actual - reference).norm() / reference.norm().clamp_min(1e-10)
            self.assertLess(relative_error.item(), 5e-4, name)


if __name__ == "__main__":
    unittest.main()
