# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify closest proper rotation frames and the clamped-face tie-break."""

import math
import unittest

import numpy as np
import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.cell_frames import compute_cell_frames
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.frames import (
    FrameResult,
    closest_proper_rotations,
    reference_rotation,
    select_reference_corners,
)


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    """Return a Haar-like random proper rotation in float64."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _unit(generator: torch.Generator) -> torch.Tensor:
    vector = torch.randn(3, generator=generator, dtype=torch.float64)
    return vector / vector.norm()


def _complete_basis(axis: torch.Tensor) -> torch.Tensor:
    """Return a proper orthonormal basis [axis, b, axis x b] as columns."""
    other = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype)
    if abs(float(axis @ other)) > 0.9:
        other = torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype)
    other = other - (axis @ other) * axis
    other = other / other.norm()
    return torch.stack((axis, other, torch.linalg.cross(axis, other)), dim=-1)


def _axis_rotation(angle: float, dtype=torch.float64) -> torch.Tensor:
    """Return diag(1, Q(angle)) with Q = [[c, -s], [s, c]]."""
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=dtype)


def _closed_form_tie_break(u1: torch.Tensor, v1: torch.Tensor, reference: torch.Tensor):
    """Return the tie-family member U diag(1, Q(theta*)) V^T closest to the reference and its bases.

    With M = V^T R_ref^T U and N its lower-right 2x2 block, tr(R_ref^T R(theta)) =
    M00 + cos(theta) (N00 + N11) + sin(theta) (N01 - N10), so theta* = atan2(N01 - N10, N00 + N11).
    """
    left, right = _complete_basis(u1), _complete_basis(v1)
    block = (right.T @ reference.T @ left)[1:, 1:]
    angle = math.atan2(float(block[0, 1] - block[1, 0]), float(block[0, 0] + block[1, 1]))
    return left @ _axis_rotation(angle) @ right.T, left, right


def _mixed_batch(dtype: torch.dtype, seed: int = 7) -> tuple[torch.Tensor, dict[str, int]]:
    """Return [2, 8, 3, 3] deformations covering generic, inverted, zero, rank-one and reflection cells."""
    generator = torch.Generator().manual_seed(seed)
    batch = torch.randn(2, 8, 3, 3, generator=generator, dtype=torch.float64) * 0.6 + torch.eye(3, dtype=torch.float64)
    cells = {"zero": 0, "rank_one": 1, "reflection": 2, "inverted_equal": 3, "inverted": 4}
    batch[:, cells["zero"]] = 0.0
    batch[:, cells["rank_one"]] = 1.7 * torch.outer(_unit(generator), _unit(generator))
    batch[:, cells["reflection"]] = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
    batch[:, cells["inverted_equal"]] = torch.diag(torch.tensor([2.0, 0.5, -0.5], dtype=torch.float64))
    inverted = batch[:, cells["inverted"]]
    batch[:, cells["inverted"]] = torch.where(torch.linalg.det(inverted)[:, None, None] < 0, inverted, -inverted)
    return batch.to(dtype), cells


class TestSelectReferenceCorners(unittest.TestCase):
    def test_canonical_grid_z_min_face(self):
        """Pick the origin, the far diagonal corner and the smallest-ID equal-area face corner."""
        rest = generate_cuboid((10, 10, 40), cell_size=0.025)
        rest_positions = rest.corner_rest_positions
        fixed = np.flatnonzero(rest_positions[:, 2] == 0)
        self.assertEqual(len(fixed), 121)
        corners = select_reference_corners(rest_positions, fixed)
        self.assertIsInstance(corners, np.ndarray)
        self.assertEqual(corners.dtype, np.int64)
        self.assertEqual(corners.shape, (3,))
        np.testing.assert_allclose(rest_positions[corners[0]], [0.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(rest_positions[corners[1]], [0.25, 0.25, 0.0], atol=1e-12)
        np.testing.assert_allclose(rest_positions[corners[2]], [0.0, 0.25, 0.0], atol=1e-12)
        # Both remaining face corners have equal area; the smaller corner ID wins deterministically.
        other = np.flatnonzero(np.all(np.isclose(rest_positions, [0.25, 0.0, 0.0]), axis=1))[0]
        self.assertLess(corners[2], other)
        self.assertTrue(np.isin(corners, fixed).all())

    def test_ordering_and_tie_rules(self):
        """Order by lexicographic minimum, farthest distance and largest area with smallest-ID ties."""
        points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 3.0]])
        np.testing.assert_array_equal(select_reference_corners(points, [0, 1, 2, 3]), [1, 3, 0])
        np.testing.assert_array_equal(select_reference_corners(points, [3, 2, 1, 0]), [1, 3, 0])
        np.testing.assert_array_equal(select_reference_corners(points, [2, 1, 0]), [1, 0, 2])
        np.testing.assert_array_equal(select_reference_corners(points, np.array([1, 1, 2, 3])), [1, 3, 2])
        np.testing.assert_array_equal(select_reference_corners(points, torch.tensor([0, 1, 4])), [1, 4, 0])

    def test_none_for_too_few_or_collinear(self):
        """Return None without three noncollinear prescribed corners."""
        rest = generate_cuboid((3, 3, 3), cell_size=0.1)
        rest_positions = rest.corner_rest_positions
        self.assertIsNone(select_reference_corners(rest_positions, []))
        self.assertIsNone(select_reference_corners(rest_positions, [0, 1]))
        self.assertIsNone(select_reference_corners(rest_positions, [0, 0, 0]))
        line = np.flatnonzero((rest_positions[:, 1] == 0) & (rest_positions[:, 2] == 0))
        self.assertEqual(len(line), 4)
        self.assertIsNone(select_reference_corners(rest_positions, line))
        coincident = np.zeros((4, 3))
        self.assertIsNone(select_reference_corners(coincident, [0, 1, 2, 3]))
        nearly = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-13, 0.0]])
        self.assertIsNone(select_reference_corners(nearly, [0, 1, 2]))
        face = np.flatnonzero(rest_positions[:, 2] == 0)
        self.assertIsNotNone(select_reference_corners(rest_positions, face))

    def test_invalid_inputs(self):
        """Reject malformed positions and indices explicitly."""
        points = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            select_reference_corners(np.zeros((4, 2)), [0, 1, 2])
        with self.assertRaises(ValueError):
            select_reference_corners(np.array([[0.0, 0.0, np.nan]] * 4), [0, 1, 2])
        with self.assertRaises(ValueError):
            select_reference_corners(points, [0, 1, 4])
        with self.assertRaises(ValueError):
            select_reference_corners(points, [0, -1, 2])
        with self.assertRaises(ValueError):
            select_reference_corners(points, [0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            select_reference_corners(points, [[0, 1, 2]])


class TestReferenceRotation(unittest.TestCase):
    def test_columns_orthonormal_and_detached(self):
        """Build [e1, e2, n] columns from the three ordered corners without gradient."""
        generator = torch.Generator().manual_seed(3)
        positions = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64).requires_grad_(True)
        indices = torch.tensor([4, 1, 5])
        frame = reference_rotation(positions, indices)
        self.assertEqual(frame.shape, (2, 3, 3))
        self.assertEqual(frame.dtype, torch.float64)
        self.assertFalse(frame.requires_grad)
        identity = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
        torch.testing.assert_close(frame.transpose(-1, -2) @ frame, identity, atol=1e-14, rtol=0)
        torch.testing.assert_close(torch.linalg.det(frame), torch.ones(2, dtype=torch.float64), atol=1e-14, rtol=0)
        corners = positions.detach()[:, indices]
        edge = corners[:, 1] - corners[:, 0]
        torch.testing.assert_close(frame[..., 0], edge / edge.norm(dim=-1, keepdim=True), atol=1e-14, rtol=0)
        plane = corners[:, 2] - corners[:, 0]
        self.assertLess((frame[..., 2] * plane).sum(-1).abs().max().item(), 1e-14)
        self.assertLess((frame[..., 2] * edge).sum(-1).abs().max().item(), 1e-14)
        # Accept array-like indices too, and float32 positions.
        torch.testing.assert_close(reference_rotation(positions, [4, 1, 5]), frame)
        self.assertEqual(reference_rotation(positions.float(), np.array([4, 1, 5])).dtype, torch.float32)

    def test_rigid_equivariance(self):
        """Rotate the reference with the whole problem; translation has no effect."""
        generator = torch.Generator().manual_seed(4)
        positions = torch.randn(3, 5, 3, generator=generator, dtype=torch.float64)
        rotation = _random_rotation(generator)
        moved = positions @ rotation.T + torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
        indices = torch.tensor([0, 2, 3])
        original = reference_rotation(positions, indices)
        torch.testing.assert_close(reference_rotation(moved, indices), rotation @ original, atol=1e-13, rtol=0)

    def test_degenerate_and_invalid(self):
        """Raise for coincident or collinear corners and malformed inputs."""
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        reference_rotation(positions, torch.tensor([0, 1, 3]))
        with self.assertRaisesRegex(ValueError, "collinear"):
            reference_rotation(positions, torch.tensor([0, 1, 2]))
        coincident = positions.clone()
        coincident[0, 1] = coincident[0, 0]
        with self.assertRaisesRegex(ValueError, "coincide"):
            reference_rotation(coincident, torch.tensor([0, 1, 3]))
        with self.assertRaises(TypeError):
            reference_rotation(positions.numpy(), torch.tensor([0, 1, 3]))
        with self.assertRaises(ValueError):
            reference_rotation(positions[0], torch.tensor([0, 1, 3]))
        with self.assertRaises(ValueError):
            reference_rotation(positions, torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            reference_rotation(positions, torch.tensor([0, 1, 1]))
        with self.assertRaises(ValueError):
            reference_rotation(positions, torch.tensor([0, 1, 4]))
        with self.assertRaises(ValueError):
            reference_rotation(positions, torch.tensor([0.0, 1.0, 3.0]))
        nonfinite = positions.clone()
        nonfinite[0, 3, 1] = float("nan")
        with self.assertRaises(ValueError):
            reference_rotation(nonfinite, torch.tensor([0, 1, 3]))


class TestClosestProperRotations(unittest.TestCase):
    def test_proper_orthonormal_and_exact_reconstruction(self):
        """Return proper rotations with R (R^T F) = F for generic, inverted, zero and rank-deficient cells."""
        for dtype, tolerance in ((torch.float32, 2e-6), (torch.float64, 1e-14)):
            with self.subTest(dtype=dtype):
                deformation, cells = _mixed_batch(dtype)
                original = deformation.clone()
                result = closest_proper_rotations(deformation.clone().requires_grad_(True))
                self.assertIsInstance(result, FrameResult)
                frames = result.frames
                self.assertEqual(frames.shape, (2, 8, 3, 3))
                self.assertEqual(frames.dtype, dtype)
                self.assertFalse(frames.requires_grad)
                self.assertFalse(result.singular_values.requires_grad)
                self.assertEqual(result.tie_mask.shape, (2, 8))
                self.assertEqual(result.tie_mask.dtype, torch.bool)
                identity = torch.eye(3, dtype=dtype).expand(2, 8, 3, 3)
                torch.testing.assert_close(frames.transpose(-1, -2) @ frames, identity, atol=tolerance, rtol=0)
                torch.testing.assert_close(
                    torch.linalg.det(frames), torch.ones(2, 8, dtype=dtype), atol=4 * tolerance, rtol=0
                )
                axes = frames.transpose(-1, -2) @ deformation
                torch.testing.assert_close(frames @ axes, deformation, atol=4 * tolerance, rtol=0)
                torch.testing.assert_close(deformation, original, atol=0, rtol=0)
                singular_values = result.singular_values
                self.assertEqual(singular_values.shape, (2, 8, 3))
                self.assertTrue((singular_values[..., :-1] >= singular_values[..., 1:]).all())
                torch.testing.assert_close(singular_values, torch.linalg.svdvals(deformation), atol=tolerance, rtol=0)
                self.assertTrue((torch.linalg.det(deformation[:, cells["inverted"]]) < 0).all())
                self.assertTrue(result.tie_mask[:, cells["zero"]].all())
                self.assertTrue(result.tie_mask[:, cells["rank_one"]].all())
                self.assertTrue(result.tie_mask[:, cells["reflection"]].all())
                self.assertTrue(result.tie_mask[:, cells["inverted_equal"]].all())
                self.assertFalse(result.tie_mask[:, cells["inverted"]].any())
                self.assertFalse(result.tie_mask[:, 5:].any())

    def test_positive_deformation_is_polar_factor(self):
        """Recover U Vh for positively oriented F and the exact rotation of a rotation-stretch product."""
        generator = torch.Generator().manual_seed(9)
        deformation = torch.randn(1, 6, 3, 3, generator=generator, dtype=torch.float64) * 0.4 + torch.eye(
            3, dtype=torch.float64
        )
        self.assertTrue((torch.linalg.det(deformation) > 0).all())
        left, _, right_transpose = np.linalg.svd(deformation.numpy())
        result = closest_proper_rotations(deformation)
        np.testing.assert_allclose(result.frames.numpy(), left @ right_transpose, atol=1e-12)
        self.assertFalse(result.tie_mask.any())
        rotation = _random_rotation(generator)
        stretch = torch.tensor([[1.3, 0.2, -0.1], [0.2, 0.8, 0.06], [-0.1, 0.06, 1.1]], dtype=torch.float64)
        product = closest_proper_rotations((rotation @ stretch)[None, None])
        torch.testing.assert_close(product.frames[0, 0], rotation, atol=1e-13, rtol=0)
        torch.testing.assert_close(product.frames[0, 0].T @ (rotation @ stretch), stretch, atol=1e-13, rtol=0)

    def test_inverted_flips_smallest_direction_and_is_closest_proper(self):
        """Match U diag(1, 1, -1) Vh for det F < 0 and beat every other proper candidate in Frobenius distance."""
        generator = torch.Generator().manual_seed(10)
        deformation = torch.randn(1, 12, 3, 3, generator=generator, dtype=torch.float64)
        deformation[..., :, 0] = -deformation[..., :, 0].abs() - 1.0
        deformation = torch.where(torch.linalg.det(deformation)[..., None, None] < 0, deformation, -deformation)
        self.assertTrue((torch.linalg.det(deformation) < 0).all())
        left, singular, right_transpose = np.linalg.svd(deformation.numpy())
        self.assertTrue((singular[..., 1] - singular[..., 2] > 1e-3).all(), "test needs distinct singular values")
        flipped = left @ np.diag([1.0, 1.0, -1.0]) @ right_transpose
        result = closest_proper_rotations(deformation)
        frames = result.frames.numpy()
        np.testing.assert_allclose(frames, flipped, atol=1e-12)
        self.assertFalse(result.tie_mask.any())
        improper = left @ right_transpose
        self.assertTrue((np.linalg.det(improper) < 0).all(), "U Vh is improper for inverted F")
        distance = np.linalg.norm(deformation.numpy() - frames, axis=(-2, -1))
        for flip in ([-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]):
            candidate = left @ np.diag(flip) @ right_transpose
            np.testing.assert_allclose(np.linalg.det(candidate), 1.0, atol=1e-12)
            other = np.linalg.norm(deformation.numpy() - candidate, axis=(-2, -1))
            self.assertTrue((other > distance + 1e-6).all())
        for _ in range(200):
            rotation = _random_rotation(generator).numpy()
            other = np.linalg.norm(deformation.numpy() - rotation, axis=(-2, -1))
            self.assertTrue((other >= distance - 1e-12).all())

    def test_tie_detection_thresholds(self):
        """Flag ties from s2 + s3 (proper) or s2 - s3 (inverted) relative to max(s1, 1)."""
        cases = [
            ([1.0, 1.0, 1.0], False),
            ([1.0, 1.0, -1.0], True),
            ([1.0, 0.0, 0.0], True),
            ([1.0, 0.5, 0.0], False),
            ([1.0, 0.6e-4, 0.5e-4], False),
            ([1.0, 0.5e-4, 0.4e-4], True),
            ([10.0, 5e-4, 4e-4], True),
            ([10.0, 6e-4, 5e-4], False),
            ([0.1, 6e-5, 3e-5], True),
            ([0.1, 8e-5, 3e-5], False),
            ([2.0, 0.5, -0.5], True),
            ([2.0, 0.5, -0.4], False),
            ([2.0, 0.5, -0.49995], True),
        ]
        deformation = torch.stack([torch.diag(torch.tensor(values, dtype=torch.float64)) for values, _ in cases])[None]
        result = closest_proper_rotations(deformation)
        expected = torch.tensor([[flag for _, flag in cases]])
        self.assertEqual(result.tie_mask.tolist(), expected.tolist())
        loose = closest_proper_rotations(deformation, tie_tolerance=1e-3)
        self.assertTrue(loose.tie_mask[0, 4].item())
        self.assertFalse(loose.tie_mask[0, 3].item())

    def test_zero_deformation_uses_reference(self):
        """Return the reference frame for F = 0 and a proper rotation without one."""
        generator = torch.Generator().manual_seed(12)
        reference = torch.stack([_random_rotation(generator) for _ in range(2)])
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-14)):
            with self.subTest(dtype=dtype):
                zero = torch.zeros(2, 3, 3, 3, dtype=dtype)
                plain = closest_proper_rotations(zero)
                self.assertTrue(plain.tie_mask.all())
                identity = torch.eye(3, dtype=dtype).expand(2, 3, 3, 3)
                torch.testing.assert_close(
                    plain.frames.transpose(-1, -2) @ plain.frames, identity, atol=tolerance, rtol=0
                )
                broken = closest_proper_rotations(zero, reference.to(dtype))
                self.assertTrue(broken.tie_mask.all())
                torch.testing.assert_close(
                    broken.frames, reference.to(dtype)[:, None].expand(2, 3, 3, 3), atol=tolerance, rtol=0
                )
                torch.testing.assert_close(broken.singular_values, torch.zeros(2, 3, 3, dtype=dtype), atol=0, rtol=0)

    def test_reference_only_modifies_tie_cells(self):
        """Keep non-tie frames bit-identical with or without a reference; fall back to the plain formula without one."""
        generator = torch.Generator().manual_seed(13)
        reference = torch.stack([_random_rotation(generator) for _ in range(2)])
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                deformation, cells = _mixed_batch(dtype)
                plain = closest_proper_rotations(deformation)
                broken = closest_proper_rotations(deformation, reference.to(dtype))
                self.assertTrue(torch.equal(plain.tie_mask, broken.tie_mask))
                self.assertTrue(torch.equal(plain.frames[~plain.tie_mask], broken.frames[~plain.tie_mask]))
                self.assertTrue(torch.equal(plain.singular_values, broken.singular_values))
                difference = (plain.frames - broken.frames).abs().amax(dim=(-2, -1))
                self.assertTrue((difference[plain.tie_mask] > 1e-3).all())
                left, _, right_transpose = torch.linalg.svd(deformation)
                orientation = torch.where(torch.linalg.det(left @ right_transpose) < 0, -1.0, 1.0).to(dtype)
                flip = torch.ones(2, 8, 3, dtype=dtype)
                flip[..., 2] = orientation
                expected = (left * flip[..., None, :]) @ right_transpose
                torch.testing.assert_close(plain.frames, expected, atol=0, rtol=0)
                self.assertTrue(torch.equal(plain.frames[:, cells["zero"]], torch.eye(3, dtype=dtype).expand(2, 3, 3)))

    def test_rank_one_tie_break_matches_closed_form(self):
        """Map v1 to u1 and choose the family member closest to the reference (closed-form theta*)."""
        generator = torch.Generator().manual_seed(14)
        angles = torch.linspace(0, 2 * math.pi, 721, dtype=torch.float64)[:-1]
        evaluated = 0
        for trial in range(12):
            u1, v1 = _unit(generator), _unit(generator)
            scale = (0.5, 1.0, 4.0)[trial % 3]
            reference = _random_rotation(generator)
            deformation = scale * torch.outer(u1, v1)
            expected, left, right = _closed_form_tie_break(u1, v1, reference)
            # The closed form is the family minimizer of the distance to the reference (brute-force check).
            family = torch.stack([left @ _axis_rotation(float(angle)) @ right.T for angle in angles])
            distances = torch.linalg.norm(family - reference, dim=(-2, -1))
            self.assertLessEqual(torch.linalg.norm(expected - reference).item(), distances.min().item() + 1e-9)
            torch.testing.assert_close(expected @ v1, u1, atol=1e-14, rtol=0)
            if distances.max().item() - distances.min().item() < 0.2:
                # theta* is ill conditioned when the reference is nearly a reflection in the tie plane;
                # every family member is then almost equally close, so the comparison is not meaningful.
                continue
            evaluated += 1
            # Rotating the result by pi about u1 gives the farthest family member.
            opposite_map = left @ _axis_rotation(math.pi) @ left.T
            for dtype, closed_tolerance, map_tolerance in ((torch.float64, 1e-3, 1e-3), (torch.float32, 5e-2, 2e-3)):
                with self.subTest(trial=trial, dtype=dtype):
                    result = closest_proper_rotations(deformation.to(dtype)[None, None], reference.to(dtype)[None])
                    self.assertTrue(result.tie_mask.all())
                    frame = result.frames[0, 0].double()
                    torch.testing.assert_close(frame @ v1, u1, atol=map_tolerance, rtol=0)
                    torch.testing.assert_close(frame, expected, atol=closed_tolerance, rtol=0)
                    torch.testing.assert_close(frame.T @ frame, torch.eye(3, dtype=torch.float64), atol=1e-6, rtol=0)
                    # Distinctly closer to the reference than the opposite member of the tie family.
                    opposite = opposite_map @ frame
                    torch.testing.assert_close(opposite @ v1, u1, atol=map_tolerance, rtol=0)
                    self.assertLess(
                        torch.linalg.norm(frame - reference).item(),
                        torch.linalg.norm(opposite - reference).item() - 0.1,
                    )
        self.assertGreaterEqual(evaluated, 6)

    def test_inverted_equal_singular_values_tie_break(self):
        """Resolve the inverted s2 = s3 tie to the member of the closest family nearest the reference."""
        generator = torch.Generator().manual_seed(15)
        deformation = torch.diag(torch.tensor([2.0, 0.5, -0.5], dtype=torch.float64))
        angles = torch.linspace(0, 2 * math.pi, 1441, dtype=torch.float64)[:-1]
        # U diag(1, Q) Vh with det Q = -1 and Vh = diag(1, 1, -1): every rotation about the first axis.
        family = torch.stack([_axis_rotation(angle) for angle in angles.tolist()])
        # Every family member is a proper rotation equally close to F.
        torch.testing.assert_close(torch.linalg.det(family), torch.ones(len(family), dtype=torch.float64))
        member_distance = torch.linalg.norm(family - deformation, dim=(-2, -1))
        torch.testing.assert_close(member_distance, member_distance[:1].expand_as(member_distance))
        for _ in range(4):
            reference = _random_rotation(generator)
            result = closest_proper_rotations(deformation[None, None], reference[None])
            self.assertTrue(result.tie_mask.all())
            frame = result.frames[0, 0]
            torch.testing.assert_close(frame.T @ frame, torch.eye(3, dtype=torch.float64), atol=1e-14, rtol=0)
            self.assertAlmostEqual(torch.linalg.det(frame).item(), 1.0, places=12)
            torch.testing.assert_close(frame @ deformation[:, 0], deformation[:, 0], atol=1e-3, rtol=0)
            torch.testing.assert_close(
                torch.linalg.norm(frame - deformation).item(), member_distance[0].item(), atol=1e-3, rtol=0
            )
            best = torch.linalg.norm(family - reference, dim=(-2, -1)).min().item()
            self.assertLessEqual(torch.linalg.norm(frame - reference).item(), best + 1e-3)
            plain = closest_proper_rotations(deformation[None, None]).frames[0, 0]
            self.assertLessEqual(
                torch.linalg.norm(frame - reference).item(), torch.linalg.norm(plain - reference).item() + 1e-3
            )

    def test_whole_problem_rotation_equivariance(self):
        """Give Q R for every cell, ties included, when F and the reference both rotate by Q."""
        generator = torch.Generator().manual_seed(16)
        reference = torch.stack([_random_rotation(generator) for _ in range(2)])
        rotation = _random_rotation(generator)
        deformation, _ = _mixed_batch(torch.float64)
        for dtype, tie_tolerance, plain_tolerance in ((torch.float64, 1e-9, 1e-13), (torch.float32, 1e-2, 1e-5)):
            with self.subTest(dtype=dtype):
                original = closest_proper_rotations(deformation.to(dtype), reference.to(dtype))
                rotated = closest_proper_rotations((rotation @ deformation).to(dtype), (rotation @ reference).to(dtype))
                self.assertTrue(torch.equal(original.tie_mask, rotated.tie_mask))
                self.assertTrue(original.tie_mask.any() and (~original.tie_mask).any())
                error = (rotated.frames.double() - rotation @ original.frames.double()).abs().amax(dim=(-2, -1))
                self.assertLess(error[original.tie_mask].max().item(), tie_tolerance)
                self.assertLess(error[~original.tie_mask].max().item(), plain_tolerance)
                torch.testing.assert_close(
                    rotated.singular_values,
                    original.singular_values,
                    atol=1e-5 if dtype == torch.float32 else 1e-13,
                    rtol=0,
                )

    def test_whole_problem_rotation_on_collapsed_grid(self):
        """Rotate a clamped grid with collapsed cells; frames follow Q via the rotating clamped-face reference."""
        rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        rest_positions = rest.corner_rest_positions
        fixed = np.flatnonzero(rest_positions[:, 2] == 0)
        corners = select_reference_corners(rest_positions, fixed)
        self.assertIsNotNone(corners)
        positions = rest_positions.copy()
        positions[np.isclose(rest_positions[:, 2], 0.2)] = [0.1, 0.1, 0.25]
        positions[np.isclose(rest_positions[:, 2], 0.3)] = [0.12, 0.09, 0.36]
        collapsed = np.flatnonzero(np.isclose(rest.cell_rest_centers[:, 2], 0.25))
        self.assertEqual(len(collapsed), 4)
        generator = torch.Generator().manual_seed(17)
        rotation = _random_rotation(generator)
        translation = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
        original_positions = torch.tensor(positions, dtype=torch.float64)
        moved_positions = original_positions @ rotation.T + translation
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
            with self.subTest(dtype=dtype):
                current = original_positions.to(dtype)[None]
                moved = moved_positions.to(dtype)[None]
                deformation = torch.tensor(
                    compute_cell_frames(rest, current[0].double().numpy()).deformation, dtype=dtype
                )
                moved_deformation = torch.tensor(
                    compute_cell_frames(rest, moved[0].double().numpy()).deformation, dtype=dtype
                )
                original = closest_proper_rotations(
                    deformation[None], reference_rotation(current, torch.tensor(corners))
                )
                rotated = closest_proper_rotations(
                    moved_deformation[None], reference_rotation(moved, torch.tensor(corners))
                )
                expected_ties = torch.zeros(1, 12, dtype=torch.bool)
                expected_ties[0, collapsed] = True
                self.assertTrue(torch.equal(original.tie_mask, expected_ties))
                self.assertTrue(torch.equal(rotated.tie_mask, expected_ties))
                torch.testing.assert_close(
                    rotated.frames.double(), rotation @ original.frames.double(), atol=tolerance, rtol=0
                )
                axes = original.frames.transpose(-1, -2) @ deformation[None]
                torch.testing.assert_close(original.frames @ axes, deformation[None], atol=tolerance, rtol=0)
                moved_axes = rotated.frames.transpose(-1, -2) @ moved_deformation[None]
                torch.testing.assert_close(moved_axes.double(), axes.double(), atol=tolerance, rtol=0)

    def test_tie_tolerance_zero_disables_tie_break(self):
        """Keep the plain formula everywhere with a zero tolerance while still flagging exact ties."""
        generator = torch.Generator().manual_seed(18)
        reference = _random_rotation(generator)[None]
        deformation = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
        deformation[0, 1] = torch.diag(torch.tensor([1.0, 1e-6, 1e-6], dtype=torch.float64))
        result = closest_proper_rotations(deformation, reference, tie_tolerance=0.0)
        self.assertEqual(result.tie_mask.tolist(), [[True, False]])
        torch.testing.assert_close(result.frames, torch.eye(3, dtype=torch.float64).expand(1, 2, 3, 3), atol=0, rtol=0)

    def test_invalid_inputs(self):
        """Reject malformed deformations, references and tolerances explicitly."""
        generator = torch.Generator().manual_seed(19)
        deformation = torch.randn(2, 3, 3, 3, generator=generator)
        reference = torch.stack([_random_rotation(generator) for _ in range(2)]).float()
        closest_proper_rotations(deformation, reference)
        with self.assertRaises(TypeError):
            closest_proper_rotations(deformation.numpy())
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation[0])
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation[..., :2])
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation.to(torch.int64))
        nonfinite = deformation.clone()
        nonfinite[0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            closest_proper_rotations(nonfinite)
        with self.assertRaises(TypeError):
            closest_proper_rotations(deformation, reference.numpy())
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation, reference[:1])
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation, reference[:, None])
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation, reference.double())
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation, reference * 1.01)
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation, -reference)
        with self.assertRaises(ValueError):
            closest_proper_rotations(deformation, reference * float("nan"))
        for tolerance in (-1e-4, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                closest_proper_rotations(deformation, tie_tolerance=tolerance)
        for tolerance in (True, "1e-4", None):
            with self.assertRaises(TypeError):
                closest_proper_rotations(deformation, tie_tolerance=tolerance)


if __name__ == "__main__":
    unittest.main()
