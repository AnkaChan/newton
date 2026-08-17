# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the symmetric Galerkin preconditioner used by sparse projection."""

from __future__ import annotations

import os
import pathlib
import unittest

import numpy as np
import torch

from .. import torch_solver as ts
from ..gaia_assets import GAIA_ASSETS, load_gaia_tet_mesh
from ..solver_scenes import build_extension_scene


def _build_beam_states(dim_z: int, *, coarse_size: int = 12):
    scene = build_extension_scene(dim_xy=2, dim_z=dim_z, cell=0.01)
    common = {
        "rest_q": np.array(scene.rest_q, copy=True),
        "tet_indices": np.array(scene.tet_indices, copy=True),
        "tet_poses": np.array(scene.tet_poses, copy=True),
        "pinned_indices": np.array(scene.pinned_indices, copy=True),
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    dense = ts.build_solver(**common)
    jacobi = ts.build_solver(
        **common,
        projection_backend="sparse_pcg",
        pcg_relative_tolerance=1.0e-10,
        pcg_max_iterations=2048,
    )
    multigrid = ts.build_solver(
        **common,
        projection_backend="sparse_pcg",
        pcg_relative_tolerance=1.0e-10,
        pcg_max_iterations=2048,
        pcg_preconditioner="multigrid",
        multigrid_coarse_size=coarse_size,
    )
    return dense, jacobi, multigrid


def _compatible_bend(state: ts.SolverState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rest = state.rest_q
    height = rest[:, 2].max() - rest[:, 2].min()
    parameter = (rest[:, 2] - rest[:, 2].min()) / height
    displacement = torch.zeros_like(rest)
    displacement[:, 0] = 0.02 * (1.0 - parameter).square()
    target = rest + displacement
    initial = rest + 0.9 * displacement
    target[state.pinned] = rest[state.pinned]
    initial[state.pinned] = rest[state.pinned]
    gradient = ts.compute_F(target, state.tets, state.J)
    return gradient, target[state.pinned], initial


class TestMultigridProjection(unittest.TestCase):
    def test_small_projection_matches_dense_and_reduces_iterations(self):
        dense, jacobi, multigrid = _build_beam_states(40, coarse_size=16)
        gradient, pins, initial = _compatible_bend(multigrid)

        expected = ts.project_deformation_gradient(dense, gradient, pins)
        jacobi_result, jacobi_diagnostics = ts.project_deformation_gradient(
            jacobi,
            gradient,
            pins,
            initial_positions=initial,
            return_diagnostics=True,
        )
        actual, diagnostics = ts.project_deformation_gradient(
            multigrid,
            gradient,
            pins,
            initial_positions=initial,
            return_diagnostics=True,
        )

        torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)
        torch.testing.assert_close(jacobi_result, expected, rtol=2.0e-9, atol=2.0e-10)
        self.assertTrue(torch.equal(actual[multigrid.pinned], pins))
        self.assertTrue(diagnostics.converged)
        self.assertEqual(diagnostics.preconditioner, "multigrid")
        self.assertGreaterEqual(diagnostics.hierarchy_levels, 3)
        self.assertGreater(diagnostics.preconditioner_matrix_vector_products, 0)
        self.assertEqual(diagnostics.factor_solves, diagnostics.preconditioner_applications)
        self.assertLess(diagnostics.iterations, jacobi_diagnostics.iterations // 2)

    def test_v_cycle_is_symmetric_positive_and_galerkin(self):
        _dense, _jacobi, state = _build_beam_states(10, coarse_size=8)
        hierarchy = state.multigrid_hierarchy
        self.assertIsNotNone(hierarchy)
        self.assertIs(state.L_ff_sparse, hierarchy.levels[0].matrix)
        root = state.L_ff_sparse.to_dense()
        torch.testing.assert_close(root, root.T, rtol=0.0, atol=0.0)

        identity = torch.eye(state.free.numel(), dtype=torch.float64)
        work = ts._PreconditionerWork()
        inverse = ts._multigrid_v_cycle(hierarchy, 0, identity, work)
        torch.testing.assert_close(inverse, inverse.T, rtol=2.0e-12, atol=2.0e-12)
        self.assertGreater(float(torch.linalg.eigvalsh(inverse).min()), 0.0)

        for fine, coarse in zip(hierarchy.levels[:-1], hierarchy.levels[1:], strict=True):
            prolongation = torch.zeros(fine.matrix.shape[0], coarse.matrix.shape[0], dtype=torch.float64)
            prolongation[torch.arange(fine.matrix.shape[0]), fine.aggregate] = 1.0
            expected = prolongation.T @ fine.matrix.to_dense() @ prolongation
            torch.testing.assert_close(coarse.matrix.to_dense(), expected, rtol=2.0e-12, atol=2.0e-12)

    def test_setup_is_deterministic_and_invalid_hierarchy_fails_closed(self):
        _dense_a, _jacobi_a, state_a = _build_beam_states(10, coarse_size=8)
        _dense_b, _jacobi_b, state_b = _build_beam_states(10, coarse_size=8)
        hierarchy_a = state_a.multigrid_hierarchy
        hierarchy_b = state_b.multigrid_hierarchy
        self.assertEqual(len(hierarchy_a.levels), len(hierarchy_b.levels))
        for level_a, level_b in zip(hierarchy_a.levels, hierarchy_b.levels, strict=True):
            self.assertTrue(torch.equal(level_a.matrix.crow_indices(), level_b.matrix.crow_indices()))
            self.assertTrue(torch.equal(level_a.matrix.col_indices(), level_b.matrix.col_indices()))
            self.assertTrue(torch.equal(level_a.matrix.values(), level_b.matrix.values()))
            if level_a.aggregate is not None:
                self.assertTrue(torch.equal(level_a.aggregate, level_b.aggregate))

        scene = build_extension_scene(dim_xy=2, dim_z=10, cell=0.01)
        common = {
            "rest_q": np.array(scene.rest_q, copy=True),
            "tet_indices": np.array(scene.tet_indices, copy=True),
            "tet_poses": np.array(scene.tet_poses, copy=True),
            "pinned_indices": np.array(scene.pinned_indices, copy=True),
            "device": torch.device("cpu"),
            "dtype": torch.float64,
            "projection_backend": "sparse_pcg",
            "pcg_preconditioner": "multigrid",
        }
        with self.assertRaisesRegex(ValueError, "did not reach"):
            ts.build_solver(**common, multigrid_coarse_size=2, multigrid_max_levels=1)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            ts.build_solver(**common, multigrid_smoother_damping=1.0)

    def test_multigrid_is_explicitly_inference_only(self):
        _dense, _jacobi, state = _build_beam_states(10, coarse_size=8)
        gradient, pins, initial = _compatible_bend(state)
        inputs = {
            "target gradient": (gradient.clone().requires_grad_(True), pins, initial),
            "pin targets": (gradient, pins.clone().requires_grad_(True), initial),
            "initial positions": (gradient, pins, initial.clone().requires_grad_(True)),
        }
        for name, (target_value, pin_value, initial_value) in inputs.items():
            with self.subTest(input=name), self.assertRaisesRegex(ValueError, "inference-only"):
                ts.project_deformation_gradient(
                    state,
                    target_value,
                    pin_value,
                    initial_positions=initial_value,
                )


@unittest.skipUnless(os.environ.get("PSS_GAIA_ASSET_ROOT"), "set PSS_GAIA_ASSET_ROOT for Gaia asset tests")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMultigridGaiaSmoke(unittest.TestCase):
    def test_armadillo_and_spaghetti_converge(self):
        root = pathlib.Path(os.environ["PSS_GAIA_ASSET_ROOT"])
        cases = (
            ("Armadilo_lowres", 1.0, False, 64),
            ("spaghetti", 0.1, True, 128),
        )
        for name, scale, exact_end_only, iteration_limit in cases:
            with self.subTest(asset=name):
                specification = GAIA_ASSETS[name]
                mesh = load_gaia_tet_mesh(
                    root / specification.relative_path,
                    unit_scale_m_per_source_unit=scale,
                    expected_file_sha256=specification.sha256,
                )
                rest = np.array(mesh.vertices_m, copy=True)
                tets = np.array(mesh.tet_indices, copy=True)
                extent = np.ptp(rest, axis=0)
                axis = int(np.argmax(extent))
                coordinate = rest[:, axis]
                threshold = coordinate.min() if exact_end_only else coordinate.min() + 0.06 * extent[axis]
                pinned = np.flatnonzero(coordinate <= threshold).astype(np.int64)
                corners = rest[tets]
                poses = np.linalg.inv(
                    np.stack(
                        (corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]),
                        axis=-1,
                    )
                )
                state = ts.build_solver(
                    rest,
                    tets,
                    poses,
                    pinned,
                    torch.device("cuda"),
                    torch.float64,
                    projection_backend="sparse_pcg",
                    pcg_preconditioner="multigrid",
                    pcg_relative_tolerance=1.0e-8,
                    pcg_max_iterations=iteration_limit,
                )
                rest_tensor = state.rest_q
                parameter = torch.as_tensor(
                    (coordinate - coordinate.min()) / extent[axis],
                    dtype=torch.float64,
                    device=rest_tensor.device,
                )
                displacement = torch.zeros_like(rest_tensor)
                displacement[:, (axis + 1) % 3] = 0.02 * parameter.square()
                target = rest_tensor + displacement
                initial = rest_tensor + 0.9 * displacement
                target[state.pinned] = rest_tensor[state.pinned]
                initial[state.pinned] = rest_tensor[state.pinned]
                gradient = ts.compute_F(target, state.tets, state.J)

                actual, diagnostics = ts.project_deformation_gradient(
                    state,
                    gradient,
                    target[state.pinned],
                    initial_positions=initial,
                    return_diagnostics=True,
                )

                self.assertTrue(diagnostics.converged)
                self.assertLessEqual(diagnostics.iterations, iteration_limit)
                self.assertLess(diagnostics.relative_residual_max, 1.0e-8)
                self.assertLess(float(torch.sqrt(torch.mean((actual - target).square())).cpu()), 1.0e-8)
                self.assertTrue(torch.equal(actual[state.pinned], target[state.pinned]))


if __name__ == "__main__":
    unittest.main()
