# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mass-weighted free-body translation gauge."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from research.principal_stretch import torch_solver as ts


def _regular_star(n_tets: int) -> tuple[np.ndarray, np.ndarray]:
    """Return well-conditioned tetrahedra in one vertex-connected component."""
    rest = np.zeros((3 * n_tets + 1, 3), dtype=np.float64)
    tets = np.empty((n_tets, 4), dtype=np.int64)
    for tet_index in range(n_tets):
        angle = 2.0 * math.pi * tet_index / max(n_tets, 1)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        scale = 1.0 + 0.05 * tet_index
        vertices = np.arange(3 * tet_index + 1, 3 * tet_index + 4)
        rest[vertices] = scale * rotation
        tets[tet_index] = np.concatenate(([0], vertices))
    return rest, tets


def _tet_poses(rest: np.ndarray, tets: np.ndarray) -> np.ndarray:
    corners = rest[tets]
    matrices = np.stack(
        [corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]],
        axis=-1,
    )
    return np.linalg.inv(matrices)


def _build_dense(
    rest: np.ndarray,
    tets: np.ndarray,
    pinned: np.ndarray,
    masses: np.ndarray | None = None,
):
    gauge_options = {}
    if masses is not None:
        gauge_options = {
            "translation_gauge_policy": ts.TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
            "vertex_masses": masses,
        }
    return ts.build_solver(
        rest,
        tets,
        _tet_poses(rest, tets),
        pinned,
        torch.device("cpu"),
        torch.float64,
        **gauge_options,
    )


class TestFreeBodyProjection(unittest.TestCase):
    def test_batched_position_derived_center_of_mass_recovers_compatible_positions(self):
        rest, tets = _regular_star(4)
        masses = np.linspace(0.2, 2.3, len(rest), dtype=np.float64)
        state = _build_dense(rest, tets, np.empty(0, dtype=np.int64), masses)
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        phase = torch.linspace(0.0, math.pi, len(rest), dtype=torch.float64)
        deformation = torch.stack(
            (0.04 * torch.sin(phase), -0.02 * torch.sin(2.0 * phase), 0.03 * torch.cos(phase)),
            dim=-1,
        )
        positions = torch.stack(
            (
                rest_tensor + deformation + torch.tensor([0.7, -0.4, 1.1]),
                rest_tensor - 0.6 * deformation + torch.tensor([-1.2, 0.3, 0.5]),
            )
        )
        target = ts.compute_F(positions, state.tets, state.J)

        projected = ts.project_deformation_gradient(
            state,
            target,
            torch.empty(0, 3, dtype=torch.float64),
            center_of_mass_positions=positions,
        )

        self.assertEqual(tuple(projected.shape), tuple(positions.shape))
        torch.testing.assert_close(projected, positions, rtol=3.0e-12, atol=3.0e-12)

    def test_explicit_center_of_mass_is_exact_and_translation_covariant(self):
        rest, tets = _regular_star(3)
        masses = np.linspace(0.5, 2.0, len(rest), dtype=np.float64)
        state = _build_dense(rest, tets, np.empty(0, dtype=np.int64), masses)
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        generator = torch.Generator().manual_seed(8301)
        target = ts.compute_F(rest_tensor, state.tets, state.J) + 0.05 * torch.randn(
            2,
            len(tets),
            3,
            3,
            dtype=torch.float64,
            generator=generator,
        )
        center = torch.tensor([[0.2, -0.1, 0.7], [-0.4, 0.8, 0.3]], dtype=torch.float64)
        shift = torch.tensor([1.3, -0.6, 0.2], dtype=torch.float64)
        pins = torch.empty(0, 3, dtype=torch.float64)

        projected = ts.project_deformation_gradient(
            state,
            target,
            pins,
            center_of_mass_target=center,
        )
        shifted = ts.project_deformation_gradient(
            state,
            target,
            pins,
            center_of_mass_target=center + shift,
        )
        weights = torch.as_tensor(masses / masses.sum(), dtype=torch.float64)
        actual_center = torch.einsum("v,bvd->bd", weights, projected)

        torch.testing.assert_close(actual_center, center, rtol=0.0, atol=2.0e-14)
        torch.testing.assert_close(shifted, projected + shift, rtol=2.0e-12, atol=2.0e-12)

    def test_batched_projection_is_differentiable_in_target_and_gauge_positions(self):
        rest, tets = _regular_star(2)
        masses = np.linspace(0.7, 1.4, len(rest), dtype=np.float64)
        state = _build_dense(rest, tets, np.empty(0, dtype=np.int64), masses)
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        target = ts.compute_F(rest_tensor, state.tets, state.J).expand(2, -1, -1, -1).clone()
        target = (target + 0.01).requires_grad_(True)
        gauge_positions = torch.stack((rest_tensor + 0.2, rest_tensor - 0.3)).requires_grad_(True)
        pins = torch.empty(0, 3, dtype=torch.float64)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda gradient, positions: ts.project_deformation_gradient(
                    state,
                    gradient,
                    pins,
                    center_of_mass_positions=positions,
                ),
                (target, gauge_positions),
                eps=1.0e-6,
                atol=2.0e-6,
                rtol=2.0e-5,
            )
        )

    def test_pins_and_center_of_mass_gauge_remain_separate(self):
        rest, tets = _regular_star(2)
        state = _build_dense(rest, tets, np.array([0], dtype=np.int64))
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        target = ts.compute_F(rest_tensor, state.tets, state.J)
        pins = rest_tensor[state.pinned]

        expected = ts.project_deformation_gradient(state, target, pins)
        torch.testing.assert_close(expected, rest_tensor, rtol=2.0e-12, atol=2.0e-12)
        with self.assertRaisesRegex(ValueError, "translation gauge policy"):
            ts.project_deformation_gradient(
                state,
                target,
                pins,
                center_of_mass_target=rest_tensor.mean(dim=0),
            )
        with self.assertRaisesRegex(ValueError, "physical pins"):
            _build_dense(
                rest,
                tets,
                np.array([0], dtype=np.int64),
                np.ones(len(rest), dtype=np.float64),
            )

    def test_default_pinned_projection_state_digest_is_unchanged(self):
        rest, tets = _regular_star(2)
        state = _build_dense(rest, tets, np.array([0], dtype=np.int64))

        self.assertEqual(
            state.projection_state_sha256,
            "bf8612fe7457d704821ac98e445048e0bbe305515b0f507e7e1176feac0fb432",
        )

    def test_mass_distribution_is_bound_into_projection_state_digest(self):
        rest, tets = _regular_star(2)
        uniform = _build_dense(
            rest,
            tets,
            np.empty(0, dtype=np.int64),
            np.ones(len(rest), dtype=np.float64),
        )
        nonuniform = _build_dense(
            rest,
            tets,
            np.empty(0, dtype=np.int64),
            np.linspace(0.3, 1.7, len(rest), dtype=np.float64),
        )

        self.assertNotEqual(uniform.projection_state_sha256, nonuniform.projection_state_sha256)
        original_digest = nonuniform.projection_state_sha256
        nonuniform.center_of_mass_weights.copy_(torch.roll(nonuniform.center_of_mass_weights, shifts=1))
        self.assertNotEqual(original_digest, ts.projection_state_sha256(nonuniform))

    def test_invalid_gauge_inputs_and_disconnected_mesh_fail_clearly(self):
        rest, tets = _regular_star(1)
        masses = np.ones(len(rest), dtype=np.float64)
        state = _build_dense(rest, tets, np.empty(0, dtype=np.int64), masses)
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        target = ts.compute_F(rest_tensor, state.tets, state.J)
        pins = torch.empty(0, 3, dtype=torch.float64)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            ts.project_deformation_gradient(state, target, pins)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ts.project_deformation_gradient(
                state,
                target,
                pins,
                center_of_mass_positions=rest_tensor,
                center_of_mass_target=rest_tensor.mean(dim=0),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _build_dense(
                rest,
                tets,
                np.empty(0, dtype=np.int64),
                np.array([1.0, 1.0, -1.0, 1.0]),
            )
        with self.assertRaisesRegex(ValueError, "requires vertex_masses"):
            ts.build_solver(
                rest,
                tets,
                _tet_poses(rest, tets),
                np.empty(0, dtype=np.int64),
                torch.device("cpu"),
                torch.float64,
                translation_gauge_policy=ts.TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
            )
        with self.assertRaisesRegex(ValueError, "projection_backend='dense'"):
            ts.build_solver(
                rest,
                tets,
                _tet_poses(rest, tets),
                np.empty(0, dtype=np.int64),
                torch.device("cpu"),
                torch.float64,
                projection_backend="sparse_pcg",
                translation_gauge_policy=ts.TRANSLATION_GAUGE_MASS_WEIGHTED_CENTER_OF_MASS,
                vertex_masses=masses,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            ts.project_deformation_gradient(
                state,
                target,
                pins,
                center_of_mass_target=torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float64),
            )
        nonfinite_positions = rest_tensor.clone()
        nonfinite_positions[0, 1] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            ts.project_deformation_gradient(
                state,
                target,
                pins,
                center_of_mass_positions=nonfinite_positions,
            )

        invalid_weights = (
            (torch.tensor([float("nan"), 0.3, 0.3, 0.4], dtype=torch.float64), "finite"),
            (torch.tensor([-0.1, 0.3, 0.3, 0.5], dtype=torch.float64), "non-negative"),
            (torch.full((len(rest),), 0.5, dtype=torch.float64), "normalized"),
        )
        for weights, message in invalid_weights:
            with self.subTest(weight_invariant=message):
                mutated = _build_dense(rest, tets, np.empty(0, dtype=np.int64), masses)
                mutated.center_of_mass_weights.copy_(weights)
                with self.assertRaisesRegex(ValueError, message):
                    ts.project_deformation_gradient(
                        mutated,
                        target,
                        pins,
                        center_of_mass_target=rest_tensor.mean(dim=0),
                    )

        other_rest = rest + np.array([4.0, 0.0, 0.0])
        disconnected_rest = np.concatenate((rest, other_rest))
        disconnected_tets = np.concatenate((tets, tets + len(rest)))
        with self.assertRaisesRegex(ValueError, "one connected component"):
            _build_dense(
                disconnected_rest,
                disconnected_tets,
                np.empty(0, dtype=np.int64),
                np.ones(len(disconnected_rest), dtype=np.float64),
            )
        rest_with_unused_vertex = np.concatenate((rest, np.array([[3.0, 2.0, 1.0]])))
        with self.assertRaisesRegex(ValueError, "no unused vertices"):
            _build_dense(
                rest_with_unused_vertex,
                tets,
                np.empty(0, dtype=np.int64),
                np.ones(len(rest_with_unused_vertex), dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
