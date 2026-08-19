# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic hierarchy-driven random initial states."""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from research.principal_stretch.hierarchy import build_hierarchy
from research.principal_stretch.hierarchy_random_state import (
    HierarchyRandomStateConfig,
    _hierarchy_levels,
    _prolong_affine_to_tets,
    generate_hierarchy_random_state,
)


def _two_tet_mesh() -> tuple[np.ndarray, np.ndarray]:
    rest_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tet_indices = np.array([[0, 1, 2, 3], [4, 2, 1, 3]], dtype=np.int64)
    return rest_positions, tet_indices


def _unit_box_mesh(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    def vertex_index(i: int, j: int, k: int) -> int:
        width = resolution + 1
        return (i * width + j) * width + k

    coordinates = np.linspace(0.0, 1.0, resolution + 1)
    rest_positions = np.array(
        [[x, y, z] for x in coordinates for y in coordinates for z in coordinates], dtype=np.float64
    )
    permutations = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    tet_indices: list[list[int]] = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                for permutation in permutations:
                    corner = np.array([i, j, k], dtype=np.int64)
                    path = [corner.copy()]
                    for axis in permutation:
                        corner[axis] += 1
                        path.append(corner.copy())
                    tet_indices.append([vertex_index(*point) for point in path])
    return rest_positions, np.asarray(tet_indices, dtype=np.int64)


def _deformation_metrics(
    rest_positions: np.ndarray,
    deformed_positions: np.ndarray,
    tet_indices: np.ndarray,
) -> tuple[float, float]:
    rest = rest_positions[tet_indices]
    deformed = deformed_positions[tet_indices]
    rest_matrices = np.stack(
        (rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0], rest[:, 3] - rest[:, 0]),
        axis=-1,
    )
    deformed_matrices = np.stack(
        (
            deformed[:, 1] - deformed[:, 0],
            deformed[:, 2] - deformed[:, 0],
            deformed[:, 3] - deformed[:, 0],
        ),
        axis=-1,
    )
    deformation_gradients = deformed_matrices @ np.linalg.inv(rest_matrices)
    return (
        float(np.linalg.det(deformation_gradients).min()),
        float(np.linalg.svd(deformation_gradients, compute_uv=False).min()),
    )


class TestHierarchyRandomState(unittest.TestCase):
    def setUp(self):
        self.rest_positions, self.tet_indices = _two_tet_mesh()
        self.hierarchy = build_hierarchy(self.tet_indices, self.rest_positions, n_levels=2, target=2)

    def _generate(self, deformation_seed: int = 17, velocity_seed: int = 29, **kwargs):
        return generate_hierarchy_random_state(
            self.rest_positions,
            self.tet_indices,
            self.hierarchy,
            deformation_seed=deformation_seed,
            velocity_seed=velocity_seed,
            **kwargs,
        )

    def test_deterministic_read_only_result_and_independent_seeds(self):
        first = self._generate()
        repeated = self._generate()
        velocity_changed = self._generate(velocity_seed=30)
        deformation_changed = self._generate(deformation_seed=18)

        for name in ("rest_positions", "deformed_positions", "displacements", "velocities"):
            first_array = getattr(first, name)
            np.testing.assert_array_equal(first_array, getattr(repeated, name))
            self.assertEqual(first_array.dtype, np.float64)
            self.assertFalse(first_array.flags["W"])
            with self.assertRaises(ValueError):
                first_array.flat[0] = 0.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.deformation_scale = 0.0
        np.testing.assert_array_equal(first.displacements, first.deformed_positions - first.rest_positions)

        np.testing.assert_array_equal(first.deformed_positions, velocity_changed.deformed_positions)
        np.testing.assert_array_equal(first.displacements, velocity_changed.displacements)
        self.assertFalse(np.array_equal(first.velocities, velocity_changed.velocities))
        np.testing.assert_array_equal(first.velocities, deformation_changed.velocities)
        self.assertFalse(np.array_equal(first.deformed_positions, deformation_changed.deformed_positions))

    def test_global_bounds_quality_diagnostics_and_exact_pins(self):
        config = HierarchyRandomStateConfig(
            translation_fraction=0.12,
            rotation_radians=0.35,
            log_stretch=0.20,
            max_displacement_fraction=0.15,
            velocity_fraction_per_second=0.40,
            angular_velocity_radians_per_second=1.0,
            log_stretch_rate_per_second=0.50,
            max_velocity_fraction_per_second=0.45,
            minimum_singular_value=0.60,
        )
        result = self._generate(config=config, pinned_indices=np.array([0, 4], dtype=np.int64))

        for array in (result.deformed_positions, result.displacements, result.velocities):
            self.assertTrue(np.isfinite(array).all())
        np.testing.assert_array_equal(result.deformed_positions[[0, 4]], self.rest_positions[[0, 4]])
        np.testing.assert_array_equal(result.displacements[[0, 4]], np.zeros((2, 3)))
        np.testing.assert_array_equal(result.velocities[[0, 4]], np.zeros((2, 3)))

        max_displacement_fraction = (
            float(np.linalg.norm(result.displacements, axis=1).max()) / result.characteristic_length
        )
        mean_displacement = result.displacements[np.unique(self.tet_indices)].mean(axis=0)
        max_centered_displacement_fraction = (
            float(np.linalg.norm(result.displacements[np.unique(self.tet_indices)] - mean_displacement, axis=1).max())
            / result.characteristic_length
        )
        max_velocity_fraction = float(np.linalg.norm(result.velocities, axis=1).max()) / result.characteristic_length
        self.assertAlmostEqual(result.max_displacement_fraction, max_displacement_fraction)
        self.assertAlmostEqual(result.max_centered_displacement_fraction, max_centered_displacement_fraction)
        self.assertAlmostEqual(result.max_velocity_fraction_per_second, max_velocity_fraction)
        self.assertLessEqual(result.max_displacement_fraction, config.max_displacement_fraction + 1.0e-15)
        self.assertLessEqual(result.max_velocity_fraction_per_second, config.max_velocity_fraction_per_second + 1.0e-15)

        minimum_determinant, minimum_singular_value = _deformation_metrics(
            self.rest_positions, result.deformed_positions, self.tet_indices
        )
        self.assertAlmostEqual(result.minimum_determinant, minimum_determinant)
        self.assertAlmostEqual(result.minimum_singular_value, minimum_singular_value)
        self.assertGreater(result.minimum_determinant, 0.0)
        self.assertGreaterEqual(result.minimum_singular_value, config.minimum_singular_value)

    def test_empty_coarsening_hierarchy_still_uses_tet_level(self):
        hierarchy = build_hierarchy(self.tet_indices[:1], self.rest_positions, n_levels=0, target=2)
        config = HierarchyRandomStateConfig(
            translation_fraction=0.10,
            rotation_radians=0.0,
            log_stretch=0.0,
            max_displacement_fraction=0.20,
            velocity_fraction_per_second=0.0,
            angular_velocity_radians_per_second=0.0,
            log_stretch_rate_per_second=0.0,
            max_velocity_fraction_per_second=0.0,
        )
        result = generate_hierarchy_random_state(
            self.rest_positions,
            self.tet_indices[:1],
            hierarchy,
            deformation_seed=7,
            velocity_seed=11,
            config=config,
        )
        displacement = result.displacements[self.tet_indices[0]]
        np.testing.assert_allclose(displacement, np.broadcast_to(displacement[0], displacement.shape), atol=1.0e-15)
        self.assertGreater(np.linalg.norm(displacement[0]), 0.0)
        np.testing.assert_array_equal(result.velocities, np.zeros_like(result.velocities))

    def test_characteristic_length_is_invariant_to_tet_refinement(self):
        lengths = []
        for resolution in (1, 2):
            rest_positions, tet_indices = _unit_box_mesh(resolution)
            hierarchy = build_hierarchy(tet_indices, rest_positions, n_levels=2, target=2)
            result = generate_hierarchy_random_state(
                rest_positions,
                tet_indices,
                hierarchy,
                deformation_seed=17,
                velocity_seed=29,
            )
            lengths.append(result.characteristic_length)
            self.assertLessEqual(result.max_displacement_fraction, 0.18 + 1.0e-15)
            self.assertLessEqual(result.max_velocity_fraction_per_second, 0.50 + 1.0e-15)
        self.assertEqual(lengths[0], lengths[1])
        self.assertAlmostEqual(lengths[0], np.sqrt(3.0))

    def test_smooth_hierarchy_field_remains_meaningful_on_refined_box(self):
        rest_positions, tet_indices = _unit_box_mesh(8)
        hierarchy = build_hierarchy(tet_indices, rest_positions, n_levels=3, target=8)
        config = HierarchyRandomStateConfig(level_decay=0.50)
        result = generate_hierarchy_random_state(
            rest_positions,
            tet_indices,
            hierarchy,
            deformation_seed=17,
            velocity_seed=29,
            config=config,
        )

        self.assertGreaterEqual(result.max_displacement_fraction, 0.04)
        self.assertGreaterEqual(result.max_centered_displacement_fraction, 0.02)
        self.assertLessEqual(result.max_displacement_fraction, config.max_displacement_fraction + 1.0e-15)
        self.assertGreater(result.deformation_scale, 0.50)
        self.assertGreater(result.minimum_determinant, 0.0)
        self.assertGreaterEqual(result.minimum_singular_value, config.minimum_singular_value)

    def test_prolongation_reproduces_affine_fields(self):
        rest_positions, tet_indices = _unit_box_mesh(2)
        hierarchy = build_hierarchy(tet_indices, rest_positions, n_levels=3, target=2)
        levels = _hierarchy_levels(hierarchy, tet_indices.shape[0])
        source_level = len(levels) - 1
        source_centroids = levels[source_level][0]
        linear_matrix = np.array([[0.10, -0.04, 0.02], [0.03, -0.08, 0.01], [-0.02, 0.05, 0.06]], dtype=np.float64)
        offset = np.array([0.30, -0.20, 0.10], dtype=np.float64)
        linear = np.broadcast_to(linear_matrix, (source_centroids.shape[0], 3, 3)).copy()
        center_values = offset + np.einsum("ij,nj->ni", linear_matrix, source_centroids)

        tet_linear, tet_center_values = _prolong_affine_to_tets(
            linear, center_values, np.asarray(hierarchy.tet_c0), levels, source_level
        )

        expected_linear = np.broadcast_to(linear_matrix, tet_linear.shape)
        expected_center_values = offset + np.einsum("ij,nj->ni", linear_matrix, hierarchy.tet_c0)
        np.testing.assert_allclose(tet_linear, expected_linear, rtol=0.0, atol=2.0e-16)
        np.testing.assert_allclose(tet_center_values, expected_center_values, rtol=0.0, atol=3.0e-16)

    def test_generation_is_stable_far_from_the_origin(self):
        rest_positions, tet_indices = _unit_box_mesh(2)
        shift = np.array([1.0e9, -2.0e9, 3.0e9])
        shifted_positions = rest_positions + shift
        hierarchy = build_hierarchy(tet_indices, rest_positions, n_levels=3, target=2)
        shifted_hierarchy = build_hierarchy(tet_indices, shifted_positions, n_levels=3, target=2)
        base = generate_hierarchy_random_state(
            rest_positions, tet_indices, hierarchy, deformation_seed=17, velocity_seed=29
        )
        shifted = generate_hierarchy_random_state(
            shifted_positions, tet_indices, shifted_hierarchy, deformation_seed=17, velocity_seed=29
        )

        self.assertEqual(base.characteristic_length, shifted.characteristic_length)
        np.testing.assert_allclose(shifted.displacements, base.displacements, rtol=0.0, atol=3.0e-7)
        np.testing.assert_allclose(shifted.velocities, base.velocities, rtol=0.0, atol=1.0e-7)

    def test_config_and_seed_types_are_exact(self):
        with self.assertRaisesRegex(TypeError, "translation_fraction must be a built-in float"):
            HierarchyRandomStateConfig(translation_fraction=np.float64(0.1))
        with self.assertRaisesRegex(TypeError, "max_rescale_steps must be a built-in int"):
            HierarchyRandomStateConfig(max_rescale_steps=True)
        with self.assertRaisesRegex(TypeError, "deformation_seed must be a built-in int"):
            self._generate(deformation_seed=True)
        with self.assertRaisesRegex(TypeError, "config must be exactly HierarchyRandomStateConfig"):
            self._generate(config=None)

    def test_validity_backoff_fails_closed_when_disabled(self):
        config = HierarchyRandomStateConfig(
            translation_fraction=0.0,
            rotation_radians=2.0,
            log_stretch=2.0,
            max_displacement_fraction=10.0,
            minimum_singular_value=0.999999,
            max_rescale_steps=0,
        )
        with self.assertRaisesRegex(RuntimeError, "unable to generate a valid deformation"):
            self._generate(config=config)


if __name__ == "__main__":
    unittest.main()
