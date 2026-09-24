# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration checks for the experimental learned hexahedral solver step."""

import importlib.util
import unittest

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep


class TestLearnedHexSolverStep(unittest.TestCase):
    def setUp(self):
        """Build a small clamped cuboid with a repeatable non-affine initial shape."""
        torch.manual_seed(31)
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)

    def _fixture(self, *, dtype=torch.float32, nonzero_head=False):
        model = IntrinsicSolverNetwork(self.rest.cell_counts, 38, hidden_dim=16, edge_hidden_dim=8)
        model.to(dtype=dtype)
        if nonzero_head:
            with torch.no_grad():
                model.correction_head.weight.normal_(std=0.004)
                for layer in model.layers:
                    layer.film.weight.normal_(std=0.01)
        step = LearnedHexSolverStep(
            self.rest,
            self.fixed,
            lame_lambda=1000.0 * 0.3 / (1.3 * 0.4),
            lame_mu=1000.0 / 2.6,
            density=100.0,
            time_step=0.04,
            network=model,
            dtype=dtype,
        )
        x = torch.tensor(self.rest.corner_rest_positions, dtype=dtype).unsqueeze(0)
        z = x[..., 2]
        x[..., 0] += 0.08 * z.square() + 0.001 * torch.sin(21 * x[..., 1]) * z / 0.3
        x[..., 1] += 0.02 * z.square()
        y = x + torch.tensor([0.0003, -0.001, 0.0001], dtype=dtype)
        return step, x, y

    def test_per_cell_lame_conditioning_and_fusion_stiffness(self):
        """Keep zero lambda finite and preserve distinct shear/volume inputs and fusion weights."""
        rest = generate_cuboid((2, 1, 1), cell_size=0.025)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        step = LearnedHexSolverStep(
            rest,
            fixed,
            lame_lambda=[0, 1e5],
            lame_mu=[1e5, 3e5],
            density=[1000, 2000],
            time_step=1 / 60,
        )
        x = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        inputs = step.prepare_inputs(x, x)
        expected = torch.tensor(
            [[[0, np.log(2), 0, 0, 0], [np.log(2), np.log(4), np.log(2), 0, 0]]], dtype=torch.float32
        )
        torch.testing.assert_close(inputs.conditioning, expected)
        torch.testing.assert_close(step.fusion_stiffness, torch.tensor([2e5, 6.75e5]))

    def test_zero_update_preserves_shape_and_trains_head(self):
        """Preserve warped corners exactly while allowing the physical loss to train the output head."""
        step, x, y = self._fixture()
        result = step(x, y)
        torch.testing.assert_close(result.positions, x, rtol=0, atol=0)
        torch.testing.assert_close(result.positions[:, self.fixed], x[:, self.fixed], rtol=0, atol=0)
        self.assertEqual(result.positions.dtype, torch.float32)
        result.loss.total.mean().backward()
        gradient = step.network.correction_head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)

    def test_frames_frozen_and_inertial_vectors_in_receiver_frame(self):
        """Detach only polar rotations and rotate each inertial vector into its receiving frame."""
        step, x, _ = self._fixture(dtype=torch.float64)
        rotation = torch.tensor([[0.0, -1, 0], [1.0, 0, 0], [0.0, 0, 1]], dtype=x.dtype)
        x = (torch.tensor(self.rest.corner_rest_positions, dtype=x.dtype) @ rotation.T)[None].requires_grad_()
        offset = torch.tensor([0.01, 0.02, -0.03], dtype=x.dtype)
        inputs = step.prepare_inputs(x, x.detach() + offset)
        self.assertFalse(inputs.frames.requires_grad)
        self.assertTrue(inputs.local_axes.requires_grad)
        torch.testing.assert_close(inputs.local_axes, torch.eye(3, dtype=x.dtype).expand(1, 12, 3, 3))
        expected = (rotation.T @ offset / 0.1).expand(1, 12, 8, 3)
        torch.testing.assert_close(inputs.state_features[..., :24].reshape(1, 12, 8, 3), expected)
        inputs.local_axes.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)

    def test_rigid_target_changes_only_fusion(self):
        """Keep the original physical inertia when an external rigid rotation guides fusion."""
        step, x, y = self._fixture(dtype=torch.float64)
        angle = torch.tensor(0.04, dtype=x.dtype)
        c, s = angle.cos(), angle.sin()
        rotation = torch.stack(
            (torch.stack((c, -s, c * 0)), torch.stack((s, c, c * 0)), torch.tensor([0.0, 0, 1], dtype=x.dtype))
        )[None]
        saved_y = y.clone()
        result = step(
            x,
            y,
            rigid_delta_rotation=rotation,
            rigid_delta_translation=torch.tensor([[0.02, -0.01, 0.03]], dtype=x.dtype),
        )
        expected = (
            0.5 * (step.energy.lumped_mass[None, :, None] * (result.positions - y).square()).sum((1, 2)) / 0.04**2
        )
        torch.testing.assert_close(result.loss.inertia, expected)
        torch.testing.assert_close(y, saved_y, rtol=0, atol=0)
        torch.testing.assert_close(result.positions[:, self.fixed], x[:, self.fixed], rtol=0, atol=0)
        self.assertGreater((result.positions - x).norm().item(), 1e-4)
        no_translation = step(x, y, rigid_delta_rotation=rotation)
        torch.testing.assert_close(result.positions, no_translation.positions, rtol=1e-11, atol=1e-13)

    def test_loss_backpropagates_through_fusion_and_attention(self):
        """Deliver nonzero physical gradients through fusion into both edge projections and FiLM."""
        step, x, y = self._fixture(nonzero_head=True)
        output = step(x, y)
        output.local_target_axes.retain_grad()
        output.loss.total.sum().backward()
        self.assertGreater(output.local_target_axes.grad.norm().item(), 0)
        for name in (
            "node_encoder.0.weight",
            "edge_encoder.0.weight",
            "layers.0.edge_bias.weight",
            "layers.0.edge_val.weight",
            "layers.0.film.weight",
            "step_head.weight",
        ):
            gradient = dict(step.network.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0, name)

    def test_network_parameter_gradient_matches_finite_difference(self):
        """Compare the entire energy-through-fusion derivative with central differences in network weights."""
        step, x, y = self._fixture(dtype=torch.float64, nonzero_head=True)
        parameter = step.network.layers[0].edge_val.weight
        loss = step(x, y).loss.total.sum()
        gradient = torch.autograd.grad(loss, parameter)[0]
        direction = torch.randn_like(parameter)
        direction /= direction.norm()
        analytical = (gradient * direction).sum().item()
        original = parameter.detach().clone()
        epsilon = 1e-5
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                parameter.copy_(original + sign * epsilon * direction)
                values.append(step(x, y).loss.total.sum().item())
            parameter.copy_(original)
        numerical = (values[0] - values[1]) / (2 * epsilon)
        self.assertAlmostEqual(analytical, numerical, delta=max(1e-10, abs(analytical) * 1e-5))

    def test_position_gradient_with_frozen_frames(self):
        """Differentiate features and fused base positions while excluding frame decomposition."""
        step, x, y = self._fixture(dtype=torch.float64, nonzero_head=True)
        frames = step.prepare_inputs(x, y).frames
        x.requires_grad_()
        gradient = torch.autograd.grad(step(x, y, frames=frames).loss.total.sum(), x)[0]
        direction = torch.randn_like(x)
        direction /= direction.norm()
        analytical = (gradient * direction).sum().item()
        epsilon = 1e-6
        with torch.no_grad():
            plus = step(x + epsilon * direction, y, frames=frames).loss.total.sum()
            minus = step(x - epsilon * direction, y, frames=frames).loss.total.sum()
        numerical = ((plus - minus) / (2 * epsilon)).item()
        self.assertAlmostEqual(analytical, numerical, delta=max(1e-9, abs(analytical) * 1e-6))


if __name__ == "__main__":
    unittest.main()
