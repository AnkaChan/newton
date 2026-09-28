# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check incremental hexahedral fusion and its CPU/CUDA tensor adjoint."""

import itertools
import os
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if find_spec("torch") is None:
    raise unittest.SkipTest("HexFusion tests require the optional Torch dependency")

from experiments.learned_intrinsic_solver.fusion import HexFusion
from experiments.learned_intrinsic_solver.hex_energy import hex_gauss_quadrature
from experiments.learned_intrinsic_solver.pardiso import PardisoFactor


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


def _projection_case(dtype, *, seed, batch_count=2):
    """Build a clamped 2x2x3 grid with a warped base, random targets, pins, and a position gradient."""
    import torch

    rest = generate_cuboid((2, 2, 3), cell_size=0.25)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
    fusion = HexFusion(rest, fixed, dtype=dtype)
    random = torch.Generator().manual_seed(seed)
    base = torch.tensor(rest.corner_rest_positions, dtype=dtype)[None].repeat(batch_count, 1, 1)
    base = base + 0.02 * torch.randn(base.shape, generator=random, dtype=dtype)
    increments = torch.randn((batch_count, fusion.cell_count, 3, 3), generator=random, dtype=dtype)
    prescribed = base[:, fixed] + 0.01 * torch.randn((batch_count, len(fixed), 3), generator=random, dtype=dtype)
    gradient = torch.randn(base.shape, generator=random, dtype=dtype)
    return rest, fusion, base, increments, prescribed, gradient


def _mode_vectors(positions: np.ndarray, cells: np.ndarray, cell_size: float) -> np.ndarray:
    """Return the seven shared-basis target vectors [B, C, 3, 7] from corner positions [B, P, 3].

    Independent NumPy implementation of the shared definition: corner
    coordinates ``xi_k = (2x - 1, 2y - 1, 2z - 1)`` for local corner
    ``k = 4x + 2y + z``, modes ``xi^1, xi^2, xi^3, xi^1 xi^2, xi^1 xi^3,
    xi^2 xi^3, xi^1 xi^2 xi^3``, coefficients ``c_m = (1/8) sum_k e_m(xi_k) x_k``
    and ``v_m = (2 / h) c_m``.
    """
    corners = positions[:, cells]
    bits = (np.arange(8)[:, None] >> np.array([2, 1, 0])[None]) & 1
    xi = (2 * bits - 1).astype(positions.dtype)
    modes = np.stack(
        [
            xi[:, 0],
            xi[:, 1],
            xi[:, 2],
            xi[:, 0] * xi[:, 1],
            xi[:, 0] * xi[:, 2],
            xi[:, 1] * xi[:, 2],
            xi[:, 0] * xi[:, 1] * xi[:, 2],
        ],
        axis=1,
    )
    coefficients = np.einsum("km,bckj->bcjm", modes, corners) / positions.dtype.type(8)
    return coefficients * positions.dtype.type(2 / cell_size)


def _legacy_three_mode_outputs(rest, fixed, weights, dtype, base, increments, prescribed, gradient):
    """Reproduce the pre-seven-mode fuse and project_gradient with the original weighted-repeat operator.

    This is a verbatim copy of the affine-only operator assembly and solve
    sequence that predates ``target_modes``; the three-mode path must stay
    bit-identical to it.
    """
    import torch
    from scipy import sparse

    numpy_dtype = np.dtype(np.float32 if dtype == torch.float32 else np.float64)
    cells = np.asarray(rest.cell_corner_indices)
    corner_count = len(rest.corner_rest_positions)
    cell_count = len(cells)
    fixed = np.asarray(fixed, dtype=np.int64)
    free_mask = np.ones(corner_count, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    weights = np.asarray(weights, dtype=numpy_dtype)
    quadrature = hex_gauss_quadrature(rest.cell_size, dtype=numpy_dtype)
    gradients = np.asarray(quadrature.shape_gradients, dtype=numpy_dtype)
    quadrature_weights = np.asarray(quadrature.weights, dtype=numpy_dtype)
    quadrature_weights = quadrature_weights / quadrature_weights.sum(dtype=numpy_dtype)
    row_count = cell_count * 8 * 3
    rows = np.repeat(np.arange(row_count), 8)
    columns = np.broadcast_to(cells[:, None, None, :], (cell_count, 8, 3, 8)).reshape(-1)
    values = np.broadcast_to(gradients.transpose(0, 2, 1)[None], (cell_count, 8, 3, 8)).reshape(-1)
    gradient_operator = sparse.coo_matrix(
        (values, (rows, columns)), shape=(row_count, corner_count), dtype=numpy_dtype
    ).tocsr()
    row_weights = np.repeat((weights[:, None] * quadrature_weights[None]).reshape(-1), 3)
    stiffness = (gradient_operator.T @ gradient_operator.multiply(row_weights[:, None])).tocsc()
    target_columns = np.arange(cell_count * 3).reshape(cell_count, 1, 3)
    target_columns = np.broadcast_to(target_columns, (cell_count, 8, 3)).reshape(-1)
    weighted_repeat = sparse.coo_matrix(
        (row_weights, (np.arange(row_count), target_columns)),
        shape=(row_count, cell_count * 3),
        dtype=numpy_dtype,
    ).tocsr()
    target_operator = (gradient_operator.T @ weighted_repeat).tocsr()[free].tocsr()
    fixed_coupling = stiffness[free][:, fixed].tocsr()
    factor = PardisoFactor(stiffness[free][:, free].tocsc())

    def pack(values):
        return np.asfortranarray(values.transpose(1, 0, 2).reshape(values.shape[1], values.shape[0] * values.shape[2]))

    def unpack(columns, batch_count):
        return columns.reshape(columns.shape[0], batch_count, 3).transpose(1, 0, 2)

    batch_count = base.shape[0]
    target_rows = increments.numpy().swapaxes(-1, -2).reshape(batch_count, 3 * cell_count, 3)
    fixed_delta = (prescribed - base[:, fixed]).numpy()
    rhs = target_operator @ pack(target_rows) - fixed_coupling @ pack(fixed_delta)
    fused = base.numpy().copy()
    fused[:, free] += unpack(factor.solve(np.asfortranarray(rhs)), batch_count)
    fused[:, fixed] = prescribed.numpy()
    adjoint = factor.solve(pack(gradient[:, free].numpy()), transpose=True)
    projected = unpack(target_operator.T @ adjoint, batch_count).reshape(batch_count, cell_count, 3, 3)
    factor.close()
    return fused, projected.swapaxes(-1, -2)


def _seven_mode_case(dtype, *, seed, batch_count=2, cell_counts=(2, 3, 3)):
    """Build a clamped grid with a warped base, a random corner displacement, and its exact seven-mode targets."""
    import torch

    rest = generate_cuboid(cell_counts, cell_size=0.25)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
    fusion = HexFusion(rest, fixed, dtype=dtype, target_modes=7)
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    random = np.random.default_rng(seed)
    base = rest.corner_rest_positions[None] + 0.02 * random.standard_normal((batch_count, fusion.corner_count, 3))
    base = base.astype(numpy_dtype)
    displacement = (0.05 * random.standard_normal(base.shape)).astype(numpy_dtype)
    cells = rest.cell_corner_indices
    targets = _mode_vectors(base + displacement, cells, rest.cell_size) - _mode_vectors(base, cells, rest.cell_size)
    return rest, fusion, torch.from_numpy(base), torch.from_numpy(displacement), torch.from_numpy(targets)


class TestHexFusion(unittest.TestCase):
    def test_missing_selected_pardiso_runtime_does_not_fall_back(self):
        """Fail clearly when the requested sparse runtime cannot be loaded."""
        rest = generate_cuboid((1, 1, 1))
        with patch.dict(os.environ, {"MKL_RT": "/missing-newton-test-runtime/libmkl_rt.so"}):
            with self.assertRaises(ImportError):
                HexFusion(rest, [0])

    def test_backend_configuration_error_remains_visible(self):
        """Preserve backend setup advice rather than misdiagnosing constraints."""
        rest = generate_cuboid((1, 1, 1))
        with patch.dict(os.environ, {"MKL_INTERFACE_LAYER": "ILP64"}):
            with self.assertRaisesRegex(ValueError, "LP64"):
                HexFusion(rest, [0])

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


class TestHexFusionProjectGradient(unittest.TestCase):
    def test_float64_adjoint_identity(self):
        """Match <project_gradient(g), D> with <g_free, fuse(D) - fuse(0)> per batch element."""
        import torch

        _, fusion, base, increments, prescribed, gradient = _projection_case(torch.float64, seed=61)
        projected = fusion.project_gradient(gradient)
        self.assertEqual(projected.shape, increments.shape)
        self.assertEqual(projected.dtype, torch.float64)
        self.assertEqual(projected.device, gradient.device)
        self.assertFalse(projected.requires_grad)
        with torch.no_grad():
            moved = fusion.fuse(base, increments, prescribed)
            unmoved = fusion.fuse(base, torch.zeros_like(increments), prescribed)
        delta = moved - unmoved
        free = fusion.free_indices
        self.assertTrue(torch.equal(delta[:, fusion.fixed_indices], torch.zeros_like(delta[:, fusion.fixed_indices])))
        left = torch.einsum("bcij,bcij->b", projected, increments)
        right = torch.einsum("bpi,bpi->b", gradient[:, free], delta[:, free])
        self.assertGreater(float(right.abs().min()), 1e-3)
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-12)

    def test_float64_matches_autograd_of_fuse(self):
        """Equal the autograd increment gradient of <g, fuse(base, D, fixed)> for any D."""
        import torch

        _, fusion, base, increments, prescribed, gradient = _projection_case(torch.float64, seed=62)
        variable = increments.clone().requires_grad_()
        objective = (gradient * fusion.fuse(base, variable, prescribed)).sum()
        expected = torch.autograd.grad(objective, variable)[0]
        torch.testing.assert_close(fusion.project_gradient(gradient), expected, rtol=1e-12, atol=1e-12)
        other = torch.zeros_like(increments, requires_grad=True)
        expected_other = torch.autograd.grad((gradient * fusion.fuse(base, other, prescribed)).sum(), other)[0]
        torch.testing.assert_close(expected_other, expected, rtol=1e-12, atol=1e-12)

    def test_fixed_rows_are_ignored(self):
        """Read only free-corner rows; prescribed-corner rows never reach the solve."""
        import torch

        _, fusion, _, increments, _, gradient = _projection_case(torch.float64, seed=63)
        fixed = fusion.fixed_indices
        masked = gradient.clone()
        masked[:, fixed] = 0.0
        self.assertTrue(torch.equal(fusion.project_gradient(gradient), fusion.project_gradient(masked)))
        polluted = gradient.clone()
        polluted[:, fixed] = float("nan")
        self.assertTrue(torch.equal(fusion.project_gradient(polluted), fusion.project_gradient(masked)))
        only_fixed = gradient - masked
        self.assertTrue(torch.equal(fusion.project_gradient(only_fixed), torch.zeros_like(increments)))

    def test_float32_smoke_matches_reference_and_autograd(self):
        """Return finite detached float32 results close to the float64 reference on CPU."""
        import torch

        rest, fusion, base, increments, prescribed, gradient = _projection_case(torch.float32, seed=64)
        projected = fusion.project_gradient(gradient.clone().requires_grad_())
        self.assertEqual(projected.shape, increments.shape)
        self.assertEqual(projected.dtype, torch.float32)
        self.assertEqual(projected.device, gradient.device)
        self.assertFalse(projected.requires_grad)
        self.assertTrue(torch.isfinite(projected).all())
        self.assertGreater(float(projected.abs().max()), 0.0)
        variable = increments.clone().requires_grad_()
        expected = torch.autograd.grad((gradient * fusion.fuse(base, variable, prescribed)).sum(), variable)[0]
        torch.testing.assert_close(projected, expected, rtol=1e-6, atol=1e-6)
        reference = HexFusion(rest, fusion.fixed_indices, dtype=torch.float64).project_gradient(gradient.double())
        torch.testing.assert_close(projected.double(), reference, rtol=1e-4, atol=1e-5)

    def test_cuda_input_matches_cpu_and_keeps_device(self):
        """Accept a CUDA position gradient and return the CPU result on the input device."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        _, fusion, _, increments, _, gradient = _projection_case(torch.float32, seed=65)
        expected = fusion.project_gradient(gradient)
        cuda_gradient = gradient.to("cuda").requires_grad_()
        projected = fusion.project_gradient(cuda_gradient)
        self.assertEqual(projected.device, cuda_gradient.device)
        self.assertEqual(projected.dtype, torch.float32)
        self.assertEqual(projected.shape, increments.shape)
        self.assertFalse(projected.requires_grad)
        torch.testing.assert_close(projected.cpu(), expected, rtol=0, atol=0)

    def test_all_pinned_projection_is_zero(self):
        """Return zero axis gradients when no free corner remains."""
        import torch

        rest = generate_cuboid((1, 1, 1))
        fusion = HexFusion(rest, [7, 3, 2, 1, 5, 6, 0, 4])
        gradient = torch.arange(48, dtype=torch.float32).reshape(2, 8, 3)
        projected = fusion.project_gradient(gradient)
        self.assertTrue(torch.equal(projected, torch.zeros((2, 1, 3, 3))))

    def test_reject_invalid_position_gradients(self):
        """Reject implicit dtype conversion, non-tensor input, and malformed shapes."""
        import torch

        rest = generate_cuboid((1, 1, 1))
        fusion = HexFusion(rest, [0])
        gradient = torch.zeros((1, 8, 3))
        with self.assertRaises(TypeError):
            fusion.project_gradient(gradient.double())
        with self.assertRaises(TypeError):
            fusion.project_gradient(gradient.numpy())
        for malformed in (gradient[0], gradient[:, :7], gradient[:0], gradient[..., :2]):
            with self.subTest(shape=tuple(malformed.shape)), self.assertRaises(ValueError):
                fusion.project_gradient(malformed)


class TestHexFusionTargetModes(unittest.TestCase):
    def test_default_three_modes_match_legacy_implementation_bitwise(self):
        """Keep the affine-only fuse and project_gradient bit-identical to the pre-seven-mode operator path."""
        import torch

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                rest = generate_cuboid((2, 3, 3), cell_size=0.25)
                fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
                weights = np.linspace(0.5, 2.0, len(rest.cell_corner_indices))
                fusion = HexFusion(rest, fixed, cell_weights=weights, dtype=dtype)
                self.assertEqual(fusion.target_modes, 3)
                random = torch.Generator().manual_seed(71)
                base = torch.tensor(rest.corner_rest_positions, dtype=dtype)[None].repeat(2, 1, 1)
                base = base + 0.02 * torch.randn(base.shape, generator=random, dtype=dtype)
                increments = torch.randn((2, fusion.cell_count, 3, 3), generator=random, dtype=dtype)
                prescribed = base[:, fixed] + 0.01 * torch.randn((2, len(fixed), 3), generator=random, dtype=dtype)
                gradient = torch.randn(base.shape, generator=random, dtype=dtype)
                fused, projected = _legacy_three_mode_outputs(
                    rest, fixed, weights, dtype, base, increments, prescribed, gradient
                )
                self.assertTrue(np.array_equal(fusion.fuse(base, increments, prescribed).numpy(), fused))
                self.assertTrue(np.array_equal(fusion.project_gradient(gradient).numpy(), projected))

    def test_reject_invalid_mode_counts_and_mismatched_target_shapes(self):
        """Accept only 3 or 7 modes and require targets shaped for the constructed mode count."""
        import torch

        rest = generate_cuboid((1, 1, 1))
        for modes in (0, 1, 2, 4, 6, 8, True, "7"):
            with self.subTest(target_modes=modes), self.assertRaises(ValueError):
                HexFusion(rest, [0], target_modes=modes)
        affine = HexFusion(rest, [0])
        warped = HexFusion(rest, [0], target_modes=7)
        self.assertEqual(warped.target_modes, 7)
        base = torch.zeros((1, 8, 3))
        prescribed = torch.zeros((1, 1, 3))
        with self.assertRaises(ValueError):
            affine.fuse(base, torch.zeros((1, 1, 3, 7)), prescribed)
        with self.assertRaises(ValueError):
            warped.fuse(base, torch.zeros((1, 1, 3, 3)), prescribed)
        self.assertEqual(warped.fuse(base, torch.zeros((1, 1, 3, 7)), prescribed).shape, (1, 8, 3))
        self.assertEqual(warped.project_gradient(torch.ones((2, 8, 3))).shape, (2, 1, 3, 7))

    def test_seven_modes_recover_random_corner_displacement_exactly(self):
        """Reproduce any free-corner displacement from its seven-vector targets and consistent pins."""
        import torch

        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                rest, fusion, base, displacement, targets = _seven_mode_case(dtype, seed=81)
                fixed = fusion.fixed_indices
                result = fusion.fuse(base, targets, (base + displacement)[:, fixed])
                self.assertEqual(result.dtype, dtype)
                scale = float(displacement.abs().max())
                error = float(((result - base) - displacement).abs().max()) / scale
                self.assertLess(error, 1e-5)
                torch.testing.assert_close(result[:, fixed], (base + displacement)[:, fixed], rtol=0, atol=0)
                # The affine part alone cannot express the same displacement.
                affine = HexFusion(rest, fixed, dtype=dtype)
                truncated = affine.fuse(base, targets[..., :3].contiguous(), (base + displacement)[:, fixed])
                affine_error = float(((truncated - base) - displacement).abs().max()) / scale
                self.assertGreater(affine_error, 0.1)

    def test_pure_warping_target_moves_corners_only_with_seven_modes(self):
        """Move free corners for a warping-only target while a zero affine target leaves them fixed."""
        import torch

        rest, fusion, base, _, targets = _seven_mode_case(torch.float64, seed=82)
        fixed = fusion.fixed_indices
        warping = torch.zeros_like(targets)
        warping[..., 3:] = targets[..., 3:]
        self.assertGreater(float(warping.abs().max()), 0.1)
        moved = fusion.fuse(base, warping, base[:, fixed])
        motion = (moved - base)[:, fusion.free_indices]
        self.assertGreater(float(motion.abs().max()), 1e-2)
        self.assertTrue(torch.equal(moved[:, fixed], base[:, fixed]))
        affine = HexFusion(rest, fixed, dtype=torch.float64)
        still = affine.fuse(base, warping[..., :3].contiguous(), base[:, fixed])
        self.assertTrue(torch.equal(still, base))

    def test_seven_modes_with_zero_warping_match_three_modes(self):
        """Reduce to the affine-only reconstruction when the four warping columns are zero."""
        import torch

        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
            with self.subTest(dtype=dtype):
                rest, fusion, base, displacement, targets = _seven_mode_case(dtype, seed=83)
                fixed = fusion.fixed_indices
                prescribed = (base + displacement)[:, fixed]
                affine_only = torch.zeros_like(targets)
                affine_only[..., :3] = targets[..., :3]
                affine = HexFusion(rest, fixed, dtype=dtype)
                expected = affine.fuse(base, targets[..., :3].contiguous(), prescribed)
                actual = fusion.fuse(base, affine_only, prescribed)
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                gradient = torch.randn(base.shape, generator=torch.Generator().manual_seed(84), dtype=dtype)
                projected = fusion.project_gradient(gradient)
                torch.testing.assert_close(
                    projected[..., :3], affine.project_gradient(gradient), rtol=tolerance, atol=tolerance
                )
                self.assertGreater(float(projected[..., 3:].abs().max()), 0.0)

    def test_seven_modes_project_gradient_is_adjoint_of_fuse(self):
        """Match <project_gradient(g), T> with <g_free, fuse(T) - fuse(0)> and the autograd target gradient."""
        import torch

        _, fusion, base, displacement, targets = _seven_mode_case(torch.float64, seed=85)
        fixed = fusion.fixed_indices
        free = fusion.free_indices
        prescribed = (base + displacement)[:, fixed]
        gradient = torch.randn(base.shape, generator=torch.Generator().manual_seed(86), dtype=torch.float64)
        projected = fusion.project_gradient(gradient)
        self.assertEqual(projected.shape, targets.shape)
        self.assertFalse(projected.requires_grad)
        with torch.no_grad():
            delta = fusion.fuse(base, targets, prescribed) - fusion.fuse(base, torch.zeros_like(targets), prescribed)
        self.assertTrue(torch.equal(delta[:, fixed], torch.zeros_like(delta[:, fixed])))
        left = torch.einsum("bcij,bcij->b", projected, targets)
        right = torch.einsum("bpi,bpi->b", gradient[:, free], delta[:, free])
        self.assertGreater(float(right.abs().min()), 1e-3)
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-12)
        variable = targets.clone().requires_grad_()
        expected = torch.autograd.grad((gradient * fusion.fuse(base, variable, prescribed)).sum(), variable)[0]
        torch.testing.assert_close(projected, expected, rtol=1e-12, atol=1e-12)

    def test_seven_modes_float64_gradcheck(self):
        """Differentiate base, seven-vector targets, and pins through the sparse solve."""
        import torch

        rest = generate_cuboid((1, 1, 1), cell_size=0.5)
        fusion = HexFusion(rest, [0, 2], dtype=torch.float64, target_modes=7)
        random = torch.Generator().manual_seed(87)
        inputs = (
            torch.randn((1, 8, 3), generator=random, dtype=torch.float64, requires_grad=True),
            torch.randn((1, 1, 3, 7), generator=random, dtype=torch.float64, requires_grad=True),
            torch.randn((1, 2, 3), generator=random, dtype=torch.float64, requires_grad=True),
        )
        self.assertTrue(torch.autograd.gradcheck(fusion.fuse, inputs, eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_seven_modes_batch_consistency_and_float32_reference(self):
        """Solve batch channels independently and keep float32 close to the float64 reference."""
        import torch

        rest, fusion, base, displacement, targets = _seven_mode_case(torch.float64, seed=88, batch_count=3)
        fixed = fusion.fixed_indices
        prescribed = (base + displacement)[:, fixed]
        gradient = torch.randn(base.shape, generator=torch.Generator().manual_seed(89), dtype=torch.float64)
        batched = fusion.fuse(base, targets, prescribed)
        projected = fusion.project_gradient(gradient)
        for index in range(3):
            single = fusion.fuse(base[index : index + 1], targets[index : index + 1], prescribed[index : index + 1])
            torch.testing.assert_close(single, batched[index : index + 1], rtol=1e-12, atol=1e-12)
            single_projected = fusion.project_gradient(gradient[index : index + 1])
            torch.testing.assert_close(single_projected, projected[index : index + 1], rtol=1e-12, atol=1e-12)
        single_fusion = HexFusion(rest, fixed, dtype=torch.float32, target_modes=7)
        fused32 = single_fusion.fuse(base.float(), targets.float(), prescribed.float())
        self.assertEqual(fused32.dtype, torch.float32)
        torch.testing.assert_close(fused32.double(), batched, rtol=1e-4, atol=1e-5)
        projected32 = single_fusion.project_gradient(gradient.float())
        self.assertEqual(projected32.dtype, torch.float32)
        self.assertEqual(projected32.shape, (3, fusion.cell_count, 3, 7))
        torch.testing.assert_close(projected32.double(), projected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
