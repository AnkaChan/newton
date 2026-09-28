# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the seven-mode trilinear basis against the quadrature and feature conventions on the CPU."""

import importlib.util
import unittest

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is an optional dependency")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features, hex_modes
from experiments.learned_intrinsic_solver.hex_energy import hex_gauss_quadrature


def _corners(cell_counts, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return rest corners [1, P, 3], long cell corner indices [C, 8] and the cell size of a small grid."""
    grid = generate_cuboid(cell_counts, cell_size=0.25)
    positions = torch.tensor(grid.corner_rest_positions, dtype=dtype)[None]
    return positions, torch.tensor(grid.cell_corner_indices, dtype=torch.long), grid.cell_size


def _random_positions(batch: int, rest: torch.Tensor, generator: torch.Generator, scale: float = 0.05) -> torch.Tensor:
    """Perturb rest corners so every cell carries affine and warping deformation."""
    noise = torch.randn(batch, *rest.shape[1:], dtype=rest.dtype, generator=generator)
    return rest + scale * noise


def _gauss_points(cell_size: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the eight Gauss points, shape gradients and weights of hex_energy as tensors."""
    rule = hex_gauss_quadrature(cell_size, dtype=np.float64 if dtype == torch.float64 else np.float32)
    return (
        torch.tensor(rule.points, dtype=dtype),
        torch.tensor(rule.shape_gradients, dtype=dtype),
        torch.tensor(rule.weights, dtype=dtype),
    )


def _proper_rotations(*shape: int, generator: torch.Generator, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    left, _, right_transpose = torch.linalg.svd(torch.randn(*shape, 3, 3, dtype=dtype, generator=generator))
    rotations = left @ right_transpose
    flip = torch.linalg.det(rotations) < 0
    left[flip, :, -1] *= -1
    return left @ right_transpose


class TestBasisDefinition(unittest.TestCase):
    def test_constants(self):
        """Pin the seven-vector target width shared with the network and fusion."""
        self.assertEqual(hex_modes.MODE_COUNT, 7)
        self.assertEqual(hex_modes.TARGET_DIM, 21)

    def test_corner_coordinates_follow_z_fast_grid_order(self):
        """Match the corner order of generate_cuboid and hex_gauss_quadrature, with z varying fastest."""
        xi = hex_modes.corner_local_coordinates(dtype=torch.float64)
        self.assertEqual(xi.shape, (8, 3))
        expected = torch.tensor(2 * np.indices((2, 2, 2)).reshape(3, -1).T - 1, dtype=torch.float64)
        torch.testing.assert_close(xi, expected, rtol=0, atol=0)
        positions, corners, cell_size = _corners((1, 1, 1))
        rest_corners = positions[0, corners[0]]
        centre = rest_corners.mean(dim=0)
        torch.testing.assert_close((rest_corners - centre) * (2 / cell_size), xi, rtol=0, atol=1e-15)
        self.assertEqual(hex_modes.corner_local_coordinates().dtype, torch.get_default_dtype())
        self.assertEqual(hex_modes.corner_local_coordinates(dtype=torch.float32).dtype, torch.float32)

    def test_mode_matrix_is_orthogonal(self):
        """Verify E^T E = 8 I, the constant first column and the documented mode order."""
        matrix = hex_modes.mode_matrix(dtype=torch.float64)
        self.assertEqual(matrix.shape, (8, 8))
        torch.testing.assert_close(matrix.T @ matrix, 8 * torch.eye(8, dtype=torch.float64), rtol=0, atol=0)
        self.assertTrue((matrix.abs() == 1).all())
        xi = hex_modes.corner_local_coordinates(dtype=torch.float64)
        x, y, z = xi.unbind(-1)
        expected = torch.stack((torch.ones_like(x), x, y, z, x * y, x * z, y * z, x * y * z), dim=-1)
        torch.testing.assert_close(matrix, expected, rtol=0, atol=0)
        self.assertEqual(hex_modes.mode_matrix(dtype=torch.float32).dtype, torch.float32)

    def test_gradient_directions_reproduce_shape_gradients(self):
        """Recover the trilinear shape gradients as (1/8) E[:, 1:] @ grad_X e_m at the Gauss points."""
        cell_size = 0.3
        points, shape_gradients, _ = _gauss_points(cell_size, torch.float64)
        directions = hex_modes.mode_gradient_directions(points, cell_size)
        self.assertEqual(directions.shape, (8, 7, 3))
        matrix = hex_modes.mode_matrix(dtype=torch.float64)
        reconstructed = torch.einsum("km,qmj->qkj", matrix[:, 1:], directions) / 8
        torch.testing.assert_close(reconstructed, shape_gradients, rtol=1e-14, atol=1e-14)
        # Affine modes carry the constant material gradient (2 / h) e_a.
        torch.testing.assert_close(
            directions[:, :3], (2 / cell_size) * torch.eye(3, dtype=torch.float64).expand(8, 3, 3), rtol=0, atol=0
        )
        x, y, z = points.unbind(-1)
        zero = torch.zeros_like(x)
        expected_warp = torch.stack(
            (
                torch.stack((y, x, zero), dim=-1),
                torch.stack((z, zero, x), dim=-1),
                torch.stack((zero, z, y), dim=-1),
                torch.stack((y * z, x * z, x * y), dim=-1),
            ),
            dim=-2,
        )
        torch.testing.assert_close(directions[:, 3:], (2 / cell_size) * expected_warp, rtol=1e-15, atol=0)
        self.assertEqual(hex_modes.mode_gradient_directions(points.float(), cell_size).dtype, torch.float32)


class TestModeVectors(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(20260928)

    def test_affine_cell_has_zero_warping(self):
        """Recover the affine columns (2 / h) A and vanishing warping for positions A xi + t."""
        positions, corners, cell_size = _corners((1, 1, 1))
        affine = torch.tensor([[1.2, 0.3, -0.1], [0.0, 0.8, 0.25], [0.4, -0.2, 1.1]], dtype=torch.float64)
        translation = torch.tensor([0.7, -1.3, 2.1], dtype=torch.float64)
        xi = hex_modes.corner_local_coordinates(dtype=torch.float64)
        cell_positions = torch.zeros_like(positions)
        cell_positions[0, corners[0]] = xi @ affine.T + translation
        vectors = hex_modes.mode_vectors(cell_positions, corners, cell_size)
        self.assertEqual(vectors.shape, (1, 1, 3, 7))
        torch.testing.assert_close(vectors[0, 0, :, :3], (2 / cell_size) * affine, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(vectors[0, 0, :, 3:], torch.zeros(3, 4, dtype=torch.float64), rtol=0, atol=1e-14)
        gradients = xi / (4 * cell_size)
        centre = features.center_deformation(cell_positions, corners, gradients)
        torch.testing.assert_close(vectors[..., :3], centre, rtol=1e-14, atol=1e-14)

    def test_first_columns_match_center_deformation(self):
        """Match features.center_deformation in the first three columns for random corners in both dtypes."""
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
            with self.subTest(dtype=dtype):
                rest, corners, cell_size = _corners((2, 2, 1), dtype)
                positions = _random_positions(3, rest, self.generator)
                vectors = hex_modes.mode_vectors(positions, corners, cell_size)
                self.assertEqual(vectors.shape, (3, 4, 3, 7))
                self.assertEqual(vectors.dtype, dtype)
                gradients = hex_modes.corner_local_coordinates(dtype=dtype) / (4 * cell_size)
                centre = features.center_deformation(positions, corners, gradients)
                torch.testing.assert_close(vectors[..., :3], centre, rtol=tolerance, atol=tolerance)
                # Random corners do warp: the last four columns are not degenerate.
                self.assertGreater(vectors[..., 3:].abs().max().item(), 1e-3)

    def test_gauss_point_deformation_matches_shape_gradient_deformation(self):
        """Reproduce the hex_energy Gauss-point F from the seven vectors for random corners to 1e-6."""
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
            with self.subTest(dtype=dtype):
                rest, corners, cell_size = _corners((2, 1, 2), dtype)
                positions = _random_positions(2, rest, self.generator)
                points, shape_gradients, _ = _gauss_points(cell_size, dtype)
                vectors = hex_modes.mode_vectors(positions, corners, cell_size)
                deformation = hex_modes.gauss_point_deformation(vectors, points)
                self.assertEqual(deformation.shape, (2, 4, 8, 3, 3))
                self.assertEqual(deformation.dtype, dtype)
                cell_corners = positions[:, corners]
                direct = torch.einsum("bcki,qkj->bcqij", cell_corners - cell_corners[:, :, :1], shape_gradients)
                torch.testing.assert_close(deformation, direct, rtol=tolerance, atol=tolerance)
                centre = hex_modes.gauss_point_deformation(vectors, torch.zeros(1, 3, dtype=dtype))
                torch.testing.assert_close(centre[:, :, 0], vectors[..., :3], rtol=0, atol=0)

    def test_rotation_equivariance(self):
        """Rotate every target vector by Q when the corners are rotated by a proper rotation Q."""
        rest, corners, cell_size = _corners((2, 2, 1))
        positions = _random_positions(2, rest, self.generator)
        rotations = _proper_rotations(2, generator=self.generator)
        rotated = torch.einsum("bij,bpj->bpi", rotations, positions)
        vectors = hex_modes.mode_vectors(positions, corners, cell_size)
        rotated_vectors = hex_modes.mode_vectors(rotated, corners, cell_size)
        expected = torch.einsum("bij,bcjm->bcim", rotations, vectors)
        torch.testing.assert_close(rotated_vectors, expected, rtol=1e-12, atol=1e-12)
        # The local representation R^T V is invariant under the same frame change.
        local = torch.einsum("bji,bcjm->bcim", rotations, rotated_vectors)
        torch.testing.assert_close(local, vectors, rtol=1e-12, atol=1e-12)

    def test_translation_invariance(self):
        """Leave every target vector unchanged under a rigid translation of all corners."""
        rest, corners, cell_size = _corners((1, 2, 2))
        positions = _random_positions(2, rest, self.generator)
        vectors = hex_modes.mode_vectors(positions, corners, cell_size)
        shifted = hex_modes.mode_vectors(
            positions + torch.tensor([3.0, -2.0, 0.5], dtype=positions.dtype), corners, cell_size
        )
        torch.testing.assert_close(shifted, vectors, rtol=1e-13, atol=1e-13)

    def test_projection_is_the_weighted_adjoint(self):
        """Verify <G, gauss_point_deformation(V)>_w == <project_gauss_gradients(G), V> in float64."""
        cell_size = 0.25
        points, _, weights = _gauss_points(cell_size, torch.float64)
        vectors = torch.randn(3, 5, 3, 7, dtype=torch.float64, generator=self.generator)
        gradients = torch.randn(3, 5, 8, 3, 3, dtype=torch.float64, generator=self.generator)
        deformation = hex_modes.gauss_point_deformation(vectors, points)
        projected = hex_modes.project_gauss_gradients(gradients, points, weights)
        self.assertEqual(projected.shape, (3, 5, 3, 7))
        left = torch.einsum("q,bcqij,bcqij->bc", weights, gradients, deformation)
        right = torch.einsum("bcim,bcim->bc", projected, vectors)
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
        # The projection is the autograd gradient of the weighted pairing.
        leaf = vectors.clone().requires_grad_(True)
        pairing = torch.einsum("q,bcqij,bcqij->", weights, gradients, hex_modes.gauss_point_deformation(leaf, points))
        (autograd,) = torch.autograd.grad(pairing, leaf)
        torch.testing.assert_close(autograd, projected, rtol=1e-12, atol=1e-12)

    def test_projection_of_the_affine_columns_matches_quadrature_volume(self):
        """Project a constant gradient onto the affine modes as the summed weights times G."""
        cell_size = 0.5
        points, _, weights = _gauss_points(cell_size, torch.float64)
        gradient = torch.randn(3, 3, dtype=torch.float64, generator=self.generator)
        projected = hex_modes.project_gauss_gradients(gradient.expand(8, 3, 3), points, weights)
        torch.testing.assert_close(projected[:, :3], weights.sum() * gradient, rtol=1e-12, atol=1e-12)
        # Symmetric Gauss points cancel the odd warping directions of a constant gradient.
        torch.testing.assert_close(projected[:, 3:], torch.zeros(3, 4, dtype=torch.float64), rtol=0, atol=1e-13)

    def test_mode_vectors_gradcheck(self):
        """Pass a float64 finite-difference gradient check of mode_vectors in the positions."""
        rest, corners, cell_size = _corners((2, 1, 1))
        positions = _random_positions(1, rest, self.generator).requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: hex_modes.mode_vectors(x, corners, cell_size), (positions,), eps=1e-6, atol=1e-6, rtol=1e-5
            )
        )
        points, _, weights = _gauss_points(cell_size, torch.float64)
        vectors = torch.randn(1, 2, 3, 7, dtype=torch.float64, generator=self.generator).requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(lambda v: hex_modes.gauss_point_deformation(v, points), (vectors,), eps=1e-6)
        )
        gradients = torch.randn(1, 2, 8, 3, 3, dtype=torch.float64, generator=self.generator).requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda g: hex_modes.project_gauss_gradients(g, points, weights), (gradients,), eps=1e-6
            )
        )


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.rest, self.corners, self.cell_size = _corners((1, 1, 1))
        self.points, _, self.weights = _gauss_points(self.cell_size, torch.float64)

    def test_mode_vectors_rejects_bad_inputs(self):
        """Reject wrong shapes, non-long indices, device-agnostic dtype errors and invalid cell sizes."""
        with self.assertRaises(ValueError):
            hex_modes.mode_vectors(self.rest[0], self.corners, self.cell_size)
        with self.assertRaises(ValueError):
            hex_modes.mode_vectors(self.rest, self.corners[:, :4], self.cell_size)
        with self.assertRaises(TypeError):
            hex_modes.mode_vectors(self.rest, self.corners.int(), self.cell_size)
        with self.assertRaises(TypeError):
            hex_modes.mode_vectors(self.rest.long(), self.corners, self.cell_size)
        with self.assertRaises(TypeError):
            hex_modes.mode_vectors(self.rest.numpy(), self.corners, self.cell_size)
        for bad in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(cell_size=bad), self.assertRaises(ValueError):
                hex_modes.mode_vectors(self.rest, self.corners, bad)
        with self.assertRaises(ValueError):
            hex_modes.mode_gradient_directions(self.points, 0.0)
        with self.assertRaises(ValueError):
            hex_modes.mode_gradient_directions(self.points[:, :2], self.cell_size)

    def test_point_evaluation_rejects_mismatched_points_and_weights(self):
        """Reject points and weights whose shape or dtype disagree with the mode tensors."""
        vectors = torch.zeros(1, 1, 3, 7, dtype=torch.float64)
        gradients = torch.zeros(1, 1, 8, 3, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            hex_modes.gauss_point_deformation(vectors[..., :3], self.points)
        with self.assertRaises(ValueError):
            hex_modes.gauss_point_deformation(vectors, self.points[0])
        with self.assertRaises(TypeError):
            hex_modes.gauss_point_deformation(vectors, self.points.float())
        with self.assertRaises(ValueError):
            hex_modes.project_gauss_gradients(gradients[:, :, 0], self.points, self.weights)
        with self.assertRaises(ValueError):
            hex_modes.project_gauss_gradients(gradients[:, :, :4], self.points, self.weights)
        with self.assertRaises(ValueError):
            hex_modes.project_gauss_gradients(gradients, self.points, self.weights[:4])
        with self.assertRaises(TypeError):
            hex_modes.project_gauss_gradients(gradients, self.points, self.weights.float())
        with self.assertRaises(TypeError):
            hex_modes.project_gauss_gradients(gradients.float(), self.points, self.weights)

    def test_float32_round_trip_preserves_dtype(self):
        """Keep float32 tensors in float32 through every function of the module."""
        rest = self.rest.float()
        vectors = hex_modes.mode_vectors(rest, self.corners, self.cell_size)
        points = self.points.float()
        deformation = hex_modes.gauss_point_deformation(vectors, points)
        projected = hex_modes.project_gauss_gradients(deformation, points, self.weights.float())
        self.assertEqual(vectors.dtype, torch.float32)
        self.assertEqual(deformation.dtype, torch.float32)
        self.assertEqual(projected.dtype, torch.float32)
        self.assertEqual(projected.shape, (1, 1, 3, 7))


if __name__ == "__main__":
    unittest.main()
