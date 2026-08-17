# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the scalable full-deformation-gradient projection."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

from research.principal_stretch import torch_solver as ts


def _tet_poses(rest: np.ndarray, tets: np.ndarray) -> np.ndarray:
    corners = rest[tets]
    matrices = np.stack(
        [corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]],
        axis=-1,
    )
    return np.linalg.inv(matrices)


def _irregular_chain(n_tets: int) -> tuple[np.ndarray, np.ndarray]:
    count = n_tets + 3
    parameter = np.linspace(-1.0, 1.0, count)
    index = np.arange(count)
    rest = np.stack(
        [
            parameter,
            0.3 * parameter**2 + 0.1 * np.sin(1.7 * index),
            0.15 * parameter**3 + 0.1 * np.cos(1.3 * index),
        ],
        axis=1,
    )
    tets = np.stack([np.arange(i, i + 4) for i in range(n_tets)]).astype(np.int64)
    for tet in tets:
        matrix = np.stack(
            [rest[tet[1]] - rest[tet[0]], rest[tet[2]] - rest[tet[0]], rest[tet[3]] - rest[tet[0]]],
            axis=1,
        )
        if np.linalg.det(matrix) < 0.0:
            tet[2], tet[3] = tet[3], tet[2]
    return rest, tets


def _regular_star(n_tets: int) -> tuple[np.ndarray, np.ndarray]:
    """Well-conditioned tets sharing one anchored vertex."""
    rest = np.zeros((3 * n_tets + 1, 3), dtype=np.float64)
    tets = np.empty((n_tets, 4), dtype=np.int64)
    for tet_index in range(n_tets):
        angle = 2.0 * math.pi * tet_index / max(n_tets, 1)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        scale = 1.0 + 0.05 * tet_index / max(n_tets, 1)
        vertices = np.arange(3 * tet_index + 1, 3 * tet_index + 4)
        rest[vertices] = scale * rotation
        tets[tet_index] = np.concatenate(([0], vertices))
    return rest, tets


def _states(rest: np.ndarray, tets: np.ndarray, pinned: np.ndarray):
    poses = _tet_poses(rest, tets)
    dense = ts.build_solver(rest, tets, poses, pinned, torch.device("cpu"), torch.float64)
    sparse = ts.build_solver(
        rest,
        tets,
        poses,
        pinned,
        torch.device("cpu"),
        torch.float64,
        projection_backend="sparse_pcg",
        pcg_relative_tolerance=1.0e-12,
        pcg_max_iterations=1024,
    )
    return dense, sparse


class TestSparseGradientProjection(unittest.TestCase):
    def test_regular_compatible_field_matches_dense_and_preserves_pins(self):
        rest, tets = _regular_star(12)
        pinned = np.array([0], dtype=np.int64)
        dense, sparse = _states(rest, tets, pinned)
        x = torch.as_tensor(rest, dtype=torch.float64)
        phase = torch.linspace(0.0, math.pi, len(rest), dtype=torch.float64)
        x = x + torch.stack((0.02 * torch.sin(phase), 0.01 * torch.cos(phase), 0.015 * torch.sin(2.0 * phase)), -1)
        pin_targets = x[sparse.pinned]
        target = ts.compute_F(x, sparse.tets, sparse.J)

        expected = ts.project_deformation_gradient(dense, target, pin_targets)
        actual, diagnostics = ts.project_deformation_gradient(sparse, target, pin_targets, return_diagnostics=True)

        torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=2.0e-11)
        self.assertTrue(torch.equal(actual[sparse.pinned], pin_targets))
        self.assertTrue(diagnostics.converged)
        self.assertEqual(diagnostics.converged_rhs, 3)
        self.assertGreater(diagnostics.iterations, 0)
        self.assertGreaterEqual(diagnostics.matrix_vector_products, diagnostics.iterations)
        self.assertEqual(
            diagnostics.scalar_rhs_matrix_vector_products,
            diagnostics.matrix_vector_products * diagnostics.rhs_count,
        )
        self.assertLessEqual(
            diagnostics.residual_norm_max,
            diagnostics.absolute_tolerance + diagnostics.relative_tolerance * diagnostics.rhs_norm_max,
        )

    def test_irregular_noisy_field_broadcast_batch_matches_dense(self):
        rest, tets = _irregular_chain(7)
        pinned = np.array([0, 1, 2], dtype=np.int64)
        dense, sparse = _states(rest, tets, pinned)
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        generator = torch.Generator().manual_seed(73)
        base = ts.compute_F(rest_tensor, sparse.tets, sparse.J)
        target = base[None] + 0.04 * torch.randn(2, len(tets), 3, 3, generator=generator, dtype=torch.float64)
        pin_targets = rest_tensor[sparse.pinned][None] + torch.tensor([[[0.03, -0.02, 0.01]]])

        expected = ts.project_deformation_gradient(dense, target, pin_targets)
        actual, diagnostics = ts.project_deformation_gradient(sparse, target, pin_targets, return_diagnostics=True)

        self.assertEqual(tuple(actual.shape), (2, len(rest), 3))
        torch.testing.assert_close(actual, expected, rtol=3.0e-10, atol=3.0e-10)
        self.assertTrue(torch.equal(actual[:, sparse.pinned], pin_targets.expand(2, -1, -1)))
        self.assertEqual(diagnostics.rhs_count, 6)
        self.assertLess(diagnostics.relative_residual_max, 1.0e-11)

    def test_sparse_autograd_agrees_with_dense_implicit_solution(self):
        rest, tets = _irregular_chain(6)
        pinned = np.array([0, 1, 2], dtype=np.int64)
        dense, sparse = _states(rest, tets, pinned)
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        generator = torch.Generator().manual_seed(91)
        target_value = ts.compute_F(rest_tensor, sparse.tets, sparse.J) + 0.03 * torch.randn(
            len(tets), 3, 3, generator=generator, dtype=torch.float64
        )
        pin_value = rest_tensor[sparse.pinned] + torch.tensor([0.02, -0.01, 0.03], dtype=torch.float64)
        loss_weight = torch.randn(len(rest), 3, generator=generator, dtype=torch.float64)

        gradients = []
        for state in (dense, sparse):
            target = target_value.clone().requires_grad_(True)
            pins = pin_value.clone().requires_grad_(True)
            projected = ts.project_deformation_gradient(
                state,
                target,
                pins,
                relative_tolerance=1.0e-13,
                max_iterations=1024,
            )
            gradients.append(torch.autograd.grad((projected * loss_weight).sum(), (target, pins)))

        for sparse_gradient, dense_gradient in zip(gradients[1], gradients[0], strict=True):
            self.assertTrue(torch.isfinite(sparse_gradient).all())
            torch.testing.assert_close(sparse_gradient, dense_gradient, rtol=2.0e-8, atol=2.0e-9)

    def test_nonconvergence_is_reported_and_fails_closed(self):
        rest, tets = _irregular_chain(7)
        pinned = np.array([0, 1, 2], dtype=np.int64)
        _dense, sparse = _states(rest, tets, pinned)
        rest_tensor = torch.as_tensor(rest, dtype=torch.float64)
        generator = torch.Generator().manual_seed(111)
        target = ts.compute_F(rest_tensor, sparse.tets, sparse.J) + torch.randn(
            len(tets), 3, 3, generator=generator, dtype=torch.float64
        )
        pins = rest_tensor[sparse.pinned]

        _projected, diagnostics = ts.project_deformation_gradient(
            sparse,
            target,
            pins,
            relative_tolerance=1.0e-15,
            max_iterations=1,
            raise_on_nonconvergence=False,
            return_diagnostics=True,
        )
        self.assertFalse(diagnostics.converged)
        self.assertEqual(diagnostics.iterations, 1)
        self.assertLess(diagnostics.converged_rhs, diagnostics.rhs_count)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            ts.project_deformation_gradient(
                sparse,
                target,
                pins,
                relative_tolerance=1.0e-15,
                max_iterations=1,
            )

    def test_large_sparse_build_never_requests_dense_vertex_matrix(self):
        rest, tets = _regular_star(2048)
        poses = _tet_poses(rest, tets)
        pinned = np.array([0], dtype=np.int64)
        n_verts = len(rest)
        original_zeros = torch.zeros

        def guarded_zeros(*size, **kwargs):
            shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
            if shape == (n_verts, n_verts):
                raise AssertionError("sparse build requested a dense vertex matrix")
            return original_zeros(*size, **kwargs)

        with mock.patch.object(torch, "zeros", side_effect=guarded_zeros):
            state = ts.build_solver(
                rest,
                tets,
                poses,
                pinned,
                torch.device("cpu"),
                torch.float64,
                projection_backend="sparse_pcg",
            )

        self.assertIsNone(state.L)
        self.assertIsNone(state.L_ff_chol)
        self.assertEqual(state.L_ff_sparse.layout, torch.sparse_csr)
        self.assertEqual(state.L_fp.layout, torch.sparse_csr)
        self.assertLessEqual(state.L_ff_sparse._nnz(), 16 * len(tets))

    def test_sparse_state_rejects_unanchored_components_and_local_global_solve(self):
        rest_a, tets_a = _regular_star(2)
        rest_b, tets_b = _regular_star(2)
        rest_b = rest_b + np.array([4.0, 0.0, 0.0])
        rest = np.concatenate((rest_a, rest_b))
        tets = np.concatenate((tets_a, tets_b + len(rest_a)))
        with self.assertRaisesRegex(ValueError, "every connected component"):
            ts.build_solver(
                rest,
                tets,
                _tet_poses(rest, tets),
                np.array([0], dtype=np.int64),
                torch.device("cpu"),
                torch.float64,
                projection_backend="sparse_pcg",
            )

        rest, tets = _regular_star(2)
        state = ts.build_solver(
            rest,
            tets,
            _tet_poses(rest, tets),
            np.array([0], dtype=np.int64),
            torch.device("cpu"),
            torch.float64,
            projection_backend="sparse_pcg",
        )
        target = torch.eye(3, dtype=torch.float64).expand(len(tets), -1, -1)
        with self.assertRaisesRegex(ValueError, "projection_backend='dense'"):
            ts.solve(state, target, state.rest_q[state.pinned])


if __name__ == "__main__":
    unittest.main()
