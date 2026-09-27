# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check exposed-face contact samples, centroids, outward normals and their gradients on the CPU."""

import dataclasses
import importlib.util
import unittest

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is an optional dependency")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import contact_geometry

CELL_COUNTS = (2, 2, 3)
CELL_SIZE = 0.025
ORIGIN = (0.1, -0.2, 0.3)


def _expected_face_centers_and_normals(cell_counts, cell_size, origin):
    """Enumerate exposed faces of a cuboid directly from the cell coordinates.

    Returns a dict keyed by (cell_index, face_index) with (face_center, outward_normal)
    computed without touching the grid's corner tables.
    """
    nx, ny, nz = cell_counts
    expected = {}
    for cell in range(nx * ny * nz):
        coords = np.array([cell // (ny * nz), (cell // nz) % ny, cell % nz])
        center = np.asarray(origin) + cell_size * (coords + 0.5)
        for face in range(6):
            axis, sign = face // 2, (-1.0 if face % 2 == 0 else 1.0)
            at_boundary = coords[axis] == (0 if sign < 0 else cell_counts[axis] - 1)
            if not at_boundary:
                continue
            normal = np.zeros(3)
            normal[axis] = sign
            expected[(cell, face)] = (center + 0.5 * cell_size * normal, normal)
    return expected


class TestFaceLocalCorners(unittest.TestCase):
    def test_local_corner_quads_lie_on_their_face(self):
        """Pin the z-fast local corner ids of each material face and their outward diagonal normal."""
        local = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
        self.assertEqual(len(contact_geometry.FACE_LOCAL_CORNERS), 6)
        for face, quad_ids in enumerate(contact_geometry.FACE_LOCAL_CORNERS):
            axis, side = face // 2, face % 2
            self.assertEqual(sorted(quad_ids), sorted(i for i in range(8) if local[i, axis] == side))
            quad = local[list(quad_ids)]
            normal = np.cross(quad[2] - quad[0], quad[3] - quad[1])
            expected = np.zeros(3)
            expected[axis] = -2.0 if side == 0 else 2.0
            np.testing.assert_allclose(normal, expected)


class TestExposedFaceSamples(unittest.TestCase):
    def setUp(self):
        self.grid = generate_cuboid(CELL_COUNTS, cell_size=CELL_SIZE, origin=ORIGIN)
        self.samples = contact_geometry.exposed_face_samples(self.grid)
        self.rest = torch.tensor(self.grid.corner_rest_positions, dtype=torch.float64)[None]

    def test_sample_count_matches_exposed_faces(self):
        """Produce exactly one sample per exposed face with the contracted dtypes and shapes."""
        nx, ny, nz = CELL_COUNTS
        expected_count = 2 * (nx * ny + ny * nz + nx * nz)
        self.assertEqual(expected_count, int(self.grid.cell_exposed_faces.sum()))
        samples = self.samples
        self.assertEqual(samples.cell_index.shape, (expected_count,))
        self.assertEqual(samples.face_index.shape, (expected_count,))
        self.assertEqual(samples.corners.shape, (expected_count, 4))
        self.assertEqual(samples.rest_normals.shape, (expected_count, 3))
        self.assertEqual(samples.cell_index.dtype, torch.int64)
        self.assertEqual(samples.face_index.dtype, torch.int64)
        self.assertEqual(samples.corners.dtype, torch.int64)
        self.assertEqual(samples.rest_normals.dtype, torch.float32)
        exposed = torch.tensor(self.grid.cell_exposed_faces)
        self.assertTrue(exposed[samples.cell_index, samples.face_index].all())
        pairs = list(zip(samples.cell_index.tolist(), samples.face_index.tolist(), strict=True))
        self.assertEqual(pairs, sorted(pairs))
        self.assertEqual(len(set(pairs)), expected_count)

    def test_rest_centroids_and_outward_normals(self):
        """Match face centers and outward unit normals computed independently from the cuboid layout."""
        expected = _expected_face_centers_and_normals(CELL_COUNTS, CELL_SIZE, ORIGIN)
        points = contact_geometry.sample_points(self.rest, self.samples.corners)[0]
        normals = contact_geometry.sample_normals(self.rest, self.samples.corners, self.samples.rest_normals)[0]
        body_center = self.rest[0].mean(dim=0)
        self.assertEqual(len(expected), points.shape[0])
        for index, (cell, face) in enumerate(
            zip(self.samples.cell_index.tolist(), self.samples.face_index.tolist(), strict=True)
        ):
            center, normal = expected[(cell, face)]
            np.testing.assert_allclose(points[index].numpy(), center, atol=1e-12)
            np.testing.assert_allclose(normals[index].numpy(), normal, atol=1e-12)
            np.testing.assert_allclose(self.samples.rest_normals[index].numpy(), normal, atol=1e-7)
            self.assertGreater(float(torch.dot(normals[index], points[index] - body_center)), 0.0)
        np.testing.assert_allclose(torch.linalg.vector_norm(normals, dim=-1).numpy(), 1.0, atol=1e-12)

    def test_inward_corner_order_raises(self):
        """Refuse a grid whose corner tables make a rest face normal point into its cell."""
        mirrored = dataclasses.replace(
            self.grid,
            corner_rest_positions=self.grid.corner_rest_positions * np.array([-1.0, 1.0, 1.0]),
            cell_rest_centers=self.grid.cell_rest_centers * np.array([-1.0, 1.0, 1.0]),
        )
        with self.assertRaisesRegex(ValueError, "points inward"):
            contact_geometry.exposed_face_samples(mirrored)
        swapped = dataclasses.replace(self.grid, cell_corner_indices=self.grid.cell_corner_indices[:, ::-1].copy())
        with self.assertRaisesRegex(ValueError, "points inward"):
            contact_geometry.exposed_face_samples(swapped)

    def test_invalid_grid_arrays_raise(self):
        """Reject corner tables with the wrong width or out-of-range ids."""
        narrow = dataclasses.replace(self.grid, cell_corner_indices=self.grid.cell_corner_indices[:, :7].copy())
        with self.assertRaisesRegex(ValueError, r"\[C, 8\]"):
            contact_geometry.exposed_face_samples(narrow)
        out_of_range = dataclasses.replace(self.grid, cell_corner_indices=self.grid.cell_corner_indices + 1000)
        with self.assertRaisesRegex(ValueError, "index into"):
            contact_geometry.exposed_face_samples(out_of_range)


class TestSampleFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.grid = generate_cuboid(CELL_COUNTS, cell_size=CELL_SIZE, origin=ORIGIN)
        self.samples = contact_geometry.exposed_face_samples(self.grid)
        rest = torch.tensor(self.grid.corner_rest_positions, dtype=torch.float64)
        self.positions = rest[None] + 0.2 * CELL_SIZE * torch.randn(3, *rest.shape, dtype=torch.float64)

    def test_sample_points_gradcheck(self):
        """Pass a float64 finite-difference gradient check for the centroids."""
        positions = self.positions[:1].clone().requires_grad_(True)
        corners = self.samples.corners
        self.assertTrue(torch.autograd.gradcheck(lambda p: contact_geometry.sample_points(p, corners), (positions,)))

    def test_sample_normals_gradcheck(self):
        """Pass a float64 finite-difference gradient check for the diagonal normals on a deformed grid."""
        positions = self.positions[:1].clone().requires_grad_(True)
        corners, rest_normals = self.samples.corners, self.samples.rest_normals
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda p: contact_geometry.sample_normals(p, corners, rest_normals),
                (positions,),
                eps=1e-6,
                atol=1e-6,
            )
        )

    def test_normals_are_unit_and_rotation_equivariant(self):
        """Return unit normals that rotate with a rigid rotation of the deformed grid."""
        corners, rest_normals = self.samples.corners, self.samples.rest_normals
        normals = contact_geometry.sample_normals(self.positions, corners, rest_normals)
        np.testing.assert_allclose(torch.linalg.vector_norm(normals, dim=-1).numpy(), 1.0, atol=1e-12)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        if torch.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        rotated = contact_geometry.sample_normals(self.positions @ rotation.T, corners, rest_normals)
        self.assertTrue(torch.allclose(rotated, normals @ rotation.T, atol=1e-12))
        points = contact_geometry.sample_points(self.positions, corners)
        rotated_points = contact_geometry.sample_points(self.positions @ rotation.T, corners)
        self.assertTrue(torch.allclose(rotated_points, points @ rotation.T, atol=1e-12))

    def test_degenerate_face_falls_back_to_rest_normal(self):
        """Use the rest normal with finite gradients when all four corners of a face coincide."""
        corners, rest_normals = self.samples.corners, self.samples.rest_normals
        positions = self.positions[:1].clone()
        collapsed = 5
        positions[0, corners[collapsed]] = positions[0, corners[collapsed, 0]]
        positions.requires_grad_(True)
        normals = contact_geometry.sample_normals(positions, corners, rest_normals)
        self.assertTrue(torch.isfinite(normals).all())
        self.assertTrue(torch.allclose(normals[0, collapsed], rest_normals[collapsed].double()))
        reference = contact_geometry.sample_normals(positions.detach(), corners, rest_normals)
        self.assertTrue(torch.equal(normals.detach(), reference))
        (normals.sum() + contact_geometry.sample_points(positions, corners).sum()).backward()
        self.assertTrue(torch.isfinite(positions.grad).all())

    def test_fully_collapsed_body_returns_rest_normals(self):
        """Fall back to every rest normal when the whole body collapses to one point."""
        positions = torch.zeros(2, self.positions.shape[1], 3, dtype=torch.float32, requires_grad=True)
        normals = contact_geometry.sample_normals(positions, self.samples.corners, self.samples.rest_normals)
        self.assertTrue(torch.equal(normals, self.samples.rest_normals[None].expand(2, -1, -1)))
        normals.sum().backward()
        self.assertTrue(torch.equal(positions.grad, torch.zeros_like(positions)))

    def test_batching_matches_per_object(self):
        """Match per-object results exactly when three objects are evaluated as one batch."""
        corners, rest_normals = self.samples.corners, self.samples.rest_normals
        points = contact_geometry.sample_points(self.positions, corners)
        normals = contact_geometry.sample_normals(self.positions, corners, rest_normals)
        self.assertEqual(points.shape, (3, corners.shape[0], 3))
        self.assertEqual(normals.shape, (3, corners.shape[0], 3))
        for b in range(3):
            single = self.positions[b : b + 1]
            self.assertTrue(torch.equal(points[b : b + 1], contact_geometry.sample_points(single, corners)))
            self.assertTrue(
                torch.equal(normals[b : b + 1], contact_geometry.sample_normals(single, corners, rest_normals))
            )

    def test_invalid_arguments_raise(self):
        """Raise ValueError or TypeError on malformed positions, corners, rest normals and eps."""
        corners, rest_normals = self.samples.corners, self.samples.rest_normals
        positions = self.positions[:1]
        with self.assertRaisesRegex(ValueError, r"\[B, P, 3\]"):
            contact_geometry.sample_points(positions[0], corners)
        with self.assertRaisesRegex(ValueError, r"\[S, 4\]"):
            contact_geometry.sample_points(positions, corners[:, :3])
        with self.assertRaises(TypeError):
            contact_geometry.sample_points(positions, corners.to(torch.int32))
        with self.assertRaises(TypeError):
            contact_geometry.sample_points(positions.numpy(), corners)
        with self.assertRaisesRegex(ValueError, r"\[S, 3\]"):
            contact_geometry.sample_normals(positions, corners, rest_normals[:-1])
        with self.assertRaisesRegex(ValueError, "eps"):
            contact_geometry.sample_normals(positions, corners, rest_normals, eps=0.0)


if __name__ == "__main__":
    unittest.main()
