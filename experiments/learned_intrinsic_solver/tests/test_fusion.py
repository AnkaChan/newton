# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check incremental hexahedral fusion and its CPU/CUDA tensor adjoint."""

import itertools
import unittest
from importlib.util import find_spec

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if find_spec("torch") is None:
    raise unittest.SkipTest("HexFusion tests require the optional Torch dependency")

from experiments.learned_intrinsic_solver.fusion import HexFusion


def _reference_energy(rest, base, result, increments, weights):
    """Evaluate the fitting energy independently through trilinear basis values."""
    import torch

    cells = torch.tensor(rest.cell_corner_indices, dtype=torch.long)
    displacement = (result - base)[:, cells]
    total = result.new_zeros(())
    for point in itertools.product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=3):
        derivatives = []
        for corner in itertools.product((-1, 1), repeat=3):
            derivative = []
            for axis in range(3):
                value = corner[axis] / (4 * rest.cell_size)
                for other in range(3):
                    if other != axis:
                        value *= 1 + corner[other] * point[other]
                derivative.append(value)
            derivatives.append(derivative)
        gradient = torch.tensor(derivatives, dtype=result.dtype)
        sampled = torch.einsum("bcvw,va->bcwa", displacement, gradient)
        total = total + ((sampled - increments).square().sum(dim=(-1, -2)) * weights).sum() / 8
    return total


class TestHexFusion(unittest.TestCase):
    def test_single_pin_reproduces_full_affine_increment(self):
        """Recover all affine axes and a nonzero translation with only one pin."""
        import torch

        rest = generate_cuboid((1, 1, 1), cell_size=0.4)
        fusion = HexFusion(rest, [0])
        base = torch.tensor(rest.corner_rest_positions, dtype=torch.float32).unsqueeze(0)
        increment = torch.tensor([[[[0.1, 0.2, -0.1], [0.05, -0.15, 0.2], [0.1, 0.05, 0.1]]]])
        translation = torch.tensor([0.07, -0.03, 0.02])
        expected = base + base @ increment[0, 0].T + translation
        result = fusion.fuse(base, increment, expected[:, :1])
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(result[:, :1], expected[:, :1], rtol=0, atol=0)

    def test_batch_affine_recovery_with_nonzero_fixed_displacements(self):
        """Keep batch channels independent while enforcing displaced boundary pins."""
        import torch

        rest = generate_cuboid((2, 1, 3), cell_size=0.25)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        fusion = HexFusion(rest, fixed)
        base = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None].repeat(2, 1, 1)
        gradients = torch.tensor(
            [
                [[0.1, 0.0, 0.2], [0.0, -0.1, 0.05], [0.0, 0.02, 0.1]],
                [[-0.05, 0.2, 0.0], [0.1, 0.05, -0.1], [0.02, 0.0, 0.15]],
            ],
            dtype=torch.float32,
        )
        translations = torch.tensor([[0.02, -0.03, 0.01], [-0.07, 0.02, 0.04]])
        expected = base + base @ gradients.transpose(-1, -2) + translations[:, None]
        increments = gradients[:, None].expand(-1, len(rest.cell_corner_indices), -1, -1)
        result = fusion.fuse(base, increments, expected[:, fixed])
        torch.testing.assert_close(result, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(result[:, fixed], expected[:, fixed], rtol=0, atol=0)

    def test_zero_increment_preserves_a_warped_base_exactly(self):
        """Leave non-affine shared-corner geometry unchanged for a zero update."""
        import torch

        rest = generate_cuboid((2, 2, 3), cell_size=0.25)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        fusion = HexFusion(rest, fixed)
        random = torch.Generator().manual_seed(8)
        base = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        base = base + 0.03 * torch.randn(base.shape, generator=random)
        increments = torch.zeros((1, len(rest.cell_corner_indices), 3, 3))
        result = fusion.fuse(base, increments, base[:, fixed])
        self.assertTrue(torch.equal(result, base))

    def test_incompatible_targets_satisfy_free_vertex_stationarity(self):
        """Minimize full quadrature error while keeping every prescribed pin exact."""
        import torch

        rest = generate_cuboid((2, 2, 2), cell_size=0.3)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        weights = torch.linspace(0.5, 2.0, len(rest.cell_corner_indices))
        fusion = HexFusion(rest, fixed, cell_weights=weights)
        random = torch.Generator().manual_seed(11)
        base = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        base = base + 0.01 * torch.randn(base.shape, generator=random)
        increments = 0.1 * torch.randn((1, len(weights), 3, 3), generator=random)
        prescribed = base[:, fixed] + 0.01 * torch.randn((1, len(fixed), 3), generator=random)
        result = fusion.fuse(base, increments, prescribed)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[:, fixed], prescribed, rtol=0, atol=0)
        independent = result.detach().requires_grad_()
        energy = _reference_energy(rest, base, independent, increments, weights)
        gradient = torch.autograd.grad(energy, independent)[0]
        self.assertGreater(float(energy.detach()), 0.01)
        self.assertLess(float(gradient[:, fusion.free_indices].abs().max()), 5e-5)

    def test_float32_adjoint_and_finite_difference(self):
        """Differentiate target, base, and pin inputs through the sparse solve."""
        import torch

        rest = generate_cuboid((2, 1, 2), cell_size=0.5)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        fusion = HexFusion(rest, fixed)
        random = torch.Generator().manual_seed(21)
        base = torch.randn((2, len(rest.corner_rest_positions), 3), generator=random, requires_grad=True)
        increments = torch.randn((2, len(rest.cell_corner_indices), 3, 3), generator=random, requires_grad=True)
        prescribed = torch.randn((2, len(fixed), 3), generator=random, requires_grad=True)
        probe = torch.randn(base.shape, generator=random)
        inputs = (base, increments, prescribed)
        output = fusion.fuse(*inputs)
        gradients = torch.autograd.grad((output * probe).sum(), inputs)
        directions = tuple(torch.randn(value.shape, generator=random) for value in inputs)
        adjoint = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions, strict=True))
        forward = (fusion.fuse(*directions) * probe).sum()
        torch.testing.assert_close(adjoint, forward, rtol=2e-5, atol=2e-5)
        epsilon = 0.002
        positive = tuple(
            value.detach() + epsilon * direction for value, direction in zip(inputs, directions, strict=True)
        )
        negative = tuple(
            value.detach() - epsilon * direction for value, direction in zip(inputs, directions, strict=True)
        )
        finite_difference = ((fusion.fuse(*positive) - fusion.fuse(*negative)) * probe).sum() / (2 * epsilon)
        torch.testing.assert_close(adjoint, finite_difference, rtol=2e-3, atol=2e-3)
        for gradient in gradients:
            self.assertEqual(gradient.dtype, torch.float32)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_float64_reference_gradcheck(self):
        """Check every input derivative against an independent finite difference."""
        import torch

        rest = generate_cuboid((1, 1, 1), cell_size=0.5)
        fusion = HexFusion(rest, [0, 2], dtype=torch.float64)
        random = torch.Generator().manual_seed(32)
        inputs = (
            torch.randn((1, 8, 3), generator=random, dtype=torch.float64, requires_grad=True),
            torch.randn((1, 1, 3, 3), generator=random, dtype=torch.float64, requires_grad=True),
            torch.randn((1, 2, 3), generator=random, dtype=torch.float64, requires_grad=True),
        )
        self.assertTrue(torch.autograd.gradcheck(fusion.fuse, inputs, eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_all_pinned_vertices_ignore_deformation_targets(self):
        """Return prescribed positions and their gradients when no free vertices remain."""
        import torch

        rest = generate_cuboid((1, 1, 1))
        indices = [7, 3, 2, 1, 5, 6, 0, 4]
        fusion = HexFusion(rest, indices)
        base = torch.zeros((1, 8, 3), requires_grad=True)
        increments = torch.ones((1, 1, 3, 3), requires_grad=True)
        prescribed = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3).requires_grad_()
        result = fusion.fuse(base, increments, prescribed)
        torch.testing.assert_close(result[:, indices], prescribed, rtol=0, atol=0)
        gradients = torch.autograd.grad(result.sum(), (base, increments, prescribed))
        torch.testing.assert_close(gradients[0], torch.zeros_like(base), rtol=0, atol=0)
        torch.testing.assert_close(gradients[1], torch.zeros_like(increments), rtol=0, atol=0)
        torch.testing.assert_close(gradients[2], torch.ones_like(prescribed), rtol=0, atol=0)

    def test_reject_unsupported_gauges_and_invalid_static_inputs(self):
        """Reject missing anchors, invalid pins, nonpositive weights, and unsupported precision."""
        import torch

        rest = generate_cuboid((1, 1, 1))
        for indices in ([], [0, 0], [-1], [8], [0.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                HexFusion(rest, indices)
        for weights in ([0.0], [-1.0], [float("nan")], [1.0, 1.0]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                HexFusion(rest, [0], cell_weights=weights)
        with self.assertRaises(ValueError):
            HexFusion(rest, [0], dtype=torch.float16)
        with self.assertRaises(ValueError):
            HexFusion(rest, [0], cell_weights=torch.ones(1, requires_grad=True))

    def test_reject_incompatible_dynamic_inputs(self):
        """Reject implicit dtype conversion and malformed coordinate batches."""
        import torch

        rest = generate_cuboid((1, 1, 1))
        fusion = HexFusion(rest, [0])
        base = torch.zeros((1, 8, 3))
        increments = torch.zeros((1, 1, 3, 3))
        prescribed = torch.zeros((1, 1, 3))
        with self.assertRaises(TypeError):
            fusion.fuse(base.double(), increments, prescribed)
        with self.assertRaises(ValueError):
            fusion.fuse(base[0], increments, prescribed)
        with self.assertRaises(ValueError):
            fusion.fuse(base, increments, prescribed[:, :0])

    def test_cuda_forward_and_all_input_adjoints_match_cpu(self):
        """Bridge nonzero boundary data and gradients without changing device or precision."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        rest = generate_cuboid((2, 1, 2), cell_size=0.5)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        fusion = HexFusion(rest, fixed, cell_weights=np.array([0.5, 0.8, 1.3, 2.0], np.float32))
        random = torch.Generator().manual_seed(46)
        inputs = (
            torch.randn((2, len(rest.corner_rest_positions), 3), generator=random, requires_grad=True),
            torch.randn((2, len(rest.cell_corner_indices), 3, 3), generator=random, requires_grad=True),
            torch.randn((2, len(fixed), 3), generator=random, requires_grad=True),
        )
        probe = torch.randn(inputs[0].shape, generator=random)
        expected = fusion.fuse(*inputs)
        expected_gradients = torch.autograd.grad((expected * probe).sum(), inputs)
        cuda_inputs = tuple(value.detach().to("cuda").requires_grad_() for value in inputs)
        actual = fusion.fuse(*cuda_inputs)
        actual_gradients = torch.autograd.grad((actual * probe.to("cuda")).sum(), cuda_inputs)
        self.assertEqual(actual.device, cuda_inputs[0].device)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(actual[:, fixed], cuda_inputs[2], rtol=0, atol=0)
        for gradient, reference in zip(actual_gradients, expected_gradients, strict=True):
            self.assertEqual(gradient.device, cuda_inputs[0].device)
            self.assertEqual(gradient.dtype, torch.float32)
            torch.testing.assert_close(gradient.cpu(), reference, rtol=2e-6, atol=2e-6)

    def test_cuda_zero_update_and_mixed_device_rejection(self):
        """Preserve CUDA geometry exactly and reject implicit input-device transfers."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        rest = generate_cuboid((1, 1, 2), cell_size=0.25)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        fusion = HexFusion(rest, fixed)
        random = torch.Generator().manual_seed(47)
        base = torch.randn((1, len(rest.corner_rest_positions), 3), generator=random).to("cuda")
        increments = torch.zeros((1, len(rest.cell_corner_indices), 3, 3), device="cuda")
        prescribed = base[:, fixed].clone()
        self.assertTrue(torch.equal(fusion.fuse(base, increments, prescribed), base))
        with self.assertRaises(ValueError):
            fusion.fuse(base, increments.cpu(), prescribed)
        with self.assertRaises(ValueError):
            fusion.fuse(base, increments, prescribed.cpu())


if __name__ == "__main__":
    unittest.main()
