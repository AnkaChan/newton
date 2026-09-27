# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the contact scene geometry and camera used by the learned-solver renderer."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.render_learned import (
    _CONTACT_KEYS,
    _PLANE_MIN_HALF_EXTENT,
    _camera_for_trajectory,
    _contact_framing_points,
    _ContactGeometry,
    _load_trajectory,
    _look_at,
    _plane_geometry,
    _plane_quad,
)


def _beam_frames(cell_counts=(1, 1, 2), cell_size=0.1):
    """Return two frames of a tiny beam whose free end sags along -y, plus its topology."""
    grid = generate_cuboid(cell_counts, cell_size=cell_size)
    rest = grid.corner_rest_positions.astype(np.float32)
    fixed = np.flatnonzero(rest[:, 2] == rest[:, 2].min())
    sagged = rest.copy()
    sagged[:, 1] -= 0.05 * (sagged[:, 2] - rest[:, 2].min())
    positions = np.stack((rest, sagged)).astype(np.float32)
    return grid, positions, rest, fixed


def _contact_arrays(plane_present=True, plane_height=-0.12, point_count=3):
    rng = np.random.default_rng(5)
    normals = rng.normal(size=(point_count, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return {
        "contact_plane_present": np.asarray(plane_present, dtype=bool),
        "contact_plane_point": np.array((0.05, plane_height, 0.1), dtype=np.float32),
        "contact_plane_normal": np.array((0.0, 1.0, 0.0), dtype=np.float32),
        "contact_point_positions": rng.uniform(-0.1, 0.3, size=(point_count, 3)).astype(np.float32),
        "contact_point_normals": normals.astype(np.float32),
        "contact_point_radii": rng.uniform(0.05, 0.2, size=point_count).astype(np.float32),
    }


def _write_trajectory(path, positions, rest, fixed, grid, **extra):
    np.savez(
        path,
        positions=positions,
        times=np.array([0.0, 1 / 300]),
        rest_positions=rest,
        fixed_indices=fixed.astype(np.int64),
        cell_counts=np.asarray(grid.cell_counts, dtype=np.int64),
        cell_corner_indices=grid.cell_corner_indices.astype(np.int64),
        **extra,
    )


class TestRenderContactGeometry(unittest.TestCase):
    def test_plane_quad_lies_in_the_plane_and_faces_along_the_normal(self):
        """Build a square quad at the plane height, centered under the beam, wound toward the normal."""
        vertices, triangles = _plane_quad(
            np.array((0.0, -0.1, 0.0)), np.array((0.0, 1.0, 0.0)), center=np.array((0.2, 0.7, 0.5)), half_extent=1.5
        )
        self.assertEqual(vertices.shape, (4, 3))
        self.assertEqual(triangles.shape, (2, 3))
        np.testing.assert_allclose(vertices[:, 1], -0.1, rtol=0, atol=1e-6)
        np.testing.assert_allclose(vertices.mean(axis=0), (0.2, -0.1, 0.5), rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(vertices[:, 0].max() - vertices[:, 0].min()), 3.0, places=5)
        self.assertAlmostEqual(float(vertices[:, 2].max() - vertices[:, 2].min()), 3.0, places=5)
        for triangle in triangles:
            a, b, c = vertices[triangle].astype(np.float64)
            normal = np.cross(b - a, c - a)
            self.assertGreater(normal[1], 0)
            self.assertAlmostEqual(abs(normal[0]) + abs(normal[2]), 0.0, places=5)

    def test_tilted_plane_quad_keeps_its_vertices_on_the_plane(self):
        """Project the center onto an inclined plane and keep both triangle normals aligned with it."""
        normal = np.array((1.0, 1.0, 0.0)) / math.sqrt(2)
        point = np.array((0.3, -0.2, 0.0))
        vertices, triangles = _plane_quad(point, normal, center=np.array((1.0, 2.0, -0.5)), half_extent=2.0)
        offsets = (vertices.astype(np.float64) - point) @ normal
        np.testing.assert_allclose(offsets, 0.0, rtol=0, atol=1e-5)
        for triangle in triangles:
            a, b, c = vertices[triangle].astype(np.float64)
            face = np.cross(b - a, c - a)
            face /= np.linalg.norm(face)
            np.testing.assert_allclose(face, normal, rtol=0, atol=1e-5)
        with self.assertRaises(ValueError):
            _plane_quad(point, normal, center=np.zeros(3), half_extent=0.0)

    def test_plane_geometry_spans_at_least_three_meters_around_the_beam(self):
        """Center the ground quad under the trajectory bounds and never shrink it below the minimum."""
        _, positions, _, _ = _beam_frames()
        contact = _ContactGeometry(
            True,
            np.array((0.0, -0.3, 0.0), dtype=np.float32),
            np.array((0.0, 1.0, 0.0), dtype=np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros(0, np.float32),
        )
        vertices, _, half_extent = _plane_geometry(positions, contact)
        self.assertEqual(half_extent, _PLANE_MIN_HALF_EXTENT)
        center = (positions.reshape(-1, 3).min(axis=0) + positions.reshape(-1, 3).max(axis=0)) / 2
        np.testing.assert_allclose(vertices.mean(axis=0), (center[0], -0.3, center[2]), rtol=0, atol=1e-6)
        wide = positions.copy()
        wide[1, :, 0] += 4.0
        _, _, half_extent = _plane_geometry(wide, contact)
        self.assertGreater(half_extent, _PLANE_MIN_HALF_EXTENT)

    def test_camera_frames_the_floor_under_the_beam_and_the_static_points(self):
        """Keep the camera above the floor while the floor region under the beam and every sphere stay in view."""
        _, positions, _, _ = _beam_frames()
        arrays = _contact_arrays(plane_height=-0.3, point_count=2)
        arrays["contact_point_positions"] = np.array([[0.9, 0.0, 0.1], [0.0, 0.0, -0.6]], dtype=np.float32)
        arrays["contact_point_radii"] = np.array([0.05, 0.1], dtype=np.float32)
        contact = _ContactGeometry(
            True,
            arrays["contact_plane_point"],
            arrays["contact_plane_normal"],
            arrays["contact_point_positions"],
            arrays["contact_point_normals"],
            arrays["contact_point_radii"],
        )
        extra = _contact_framing_points(positions, contact)
        self.assertEqual(extra.shape, (8 + 4, 3))
        camera = _camera_for_trajectory(positions, extra_points=extra)
        self.assertEqual(camera["up_axis"], "Y")
        self.assertLessEqual(camera["bounds_min"][1], -0.3 + 1e-6)
        self.assertGreaterEqual(camera["bounds_max"][0], 0.95 - 1e-6)
        self.assertLessEqual(camera["bounds_min"][2], -0.7 + 1e-6)
        self.assertGreater(camera["position"][1], -0.3)
        plain = _camera_for_trajectory(positions)
        self.assertGreater(camera["distance"], plain["distance"])
        self.assertEqual(_contact_framing_points(positions, _ContactGeometry.empty()).shape, (0, 3))

    def test_look_at_reproduces_the_y_up_viewer_front_vector(self):
        """Match the ViewerGL Y-up yaw and pitch convention for an arbitrary view."""
        position = np.array((2.0, 1.5, -0.7))
        target = np.array((0.1, -0.2, 0.4))
        yaw, pitch = _look_at(position, target)
        front = np.array(
            (
                math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)),
                math.sin(math.radians(pitch)),
                math.sin(math.radians(yaw)) * math.cos(math.radians(pitch)),
            )
        )
        expected = (target - position) / np.linalg.norm(target - position)
        np.testing.assert_allclose(front, expected, rtol=0, atol=1e-9)

    def test_contact_arrays_round_trip_through_load_trajectory(self):
        """Read the saved plane and static points back exactly; older files without them are contact-free."""
        grid, positions, rest, fixed = _beam_frames()
        arrays = _contact_arrays()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            _write_trajectory(path, positions, rest, fixed, grid, **arrays)
            loaded = _load_trajectory(path)
            self.assertEqual(len(loaded), 6)
            contact = loaded[5]
            self.assertTrue(contact.plane_present)
            self.assertEqual(contact.point_count, 3)
            np.testing.assert_array_equal(contact.plane_point, arrays["contact_plane_point"])
            np.testing.assert_array_equal(contact.plane_normal, arrays["contact_plane_normal"])
            np.testing.assert_array_equal(contact.point_positions, arrays["contact_point_positions"])
            np.testing.assert_array_equal(contact.point_normals, arrays["contact_point_normals"])
            np.testing.assert_array_equal(contact.point_radii, arrays["contact_point_radii"])
            np.testing.assert_array_equal(loaded[0], positions)

            legacy = Path(directory) / "legacy.npz"
            _write_trajectory(legacy, positions, rest, fixed, grid)
            contact = _load_trajectory(legacy)[5]
            self.assertFalse(contact.plane_present)
            self.assertEqual(contact.point_count, 0)

            empty = Path(directory) / "empty.npz"
            _write_trajectory(
                empty, positions, rest, fixed, grid, **_contact_arrays(plane_present=False, point_count=0)
            )
            contact = _load_trajectory(empty)[5]
            self.assertFalse(contact.plane_present)
            self.assertEqual(contact.point_positions.shape, (0, 3))
            self.assertEqual(contact.point_radii.shape, (0,))

            partial = Path(directory) / "partial.npz"
            _write_trajectory(partial, positions, rest, fixed, grid, contact_plane_present=np.asarray(True))
            with self.assertRaisesRegex(ValueError, "missing contact arrays"):
                _load_trajectory(partial)

            invalid = dict(arrays)
            invalid["contact_point_radii"] = np.zeros(3, dtype=np.float32)
            broken = Path(directory) / "broken.npz"
            _write_trajectory(broken, positions, rest, fixed, grid, **invalid)
            with self.assertRaisesRegex(ValueError, "positive"):
                _load_trajectory(broken)
        self.assertEqual(len(_CONTACT_KEYS), 6)


if __name__ == "__main__":
    unittest.main()
