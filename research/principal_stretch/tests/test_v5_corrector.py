# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixed-work architecture-v5 physics corrector."""

from __future__ import annotations

import unittest

import torch

from ..v5_corrector import (
    CorrectorConfig,
    FixedPCGConfig,
    StableNeoHookeanGaussNewtonOperator,
    common_objective_free_gradient,
    correct_common_objective,
    fixed_work_block_pcg,
    stable_neo_hookean_gn_block_diagonal,
    stable_neo_hookean_gn_matvec,
)
from ..v5_objective import CommonObjectiveContext, common_objective_components, common_objective_residual


def _single_tet() -> tuple[torch.Tensor, torch.Tensor]:
    rest = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return rest, torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)


def _two_tets() -> tuple[torch.Tensor, torch.Tensor]:
    rest = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return rest, torch.tensor([[0, 1, 2, 3], [4, 2, 1, 3]], dtype=torch.int64)


def _shape_data(rest: torch.Tensor, tets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge_matrix = torch.stack(
        (
            rest[tets[:, 1]] - rest[tets[:, 0]],
            rest[tets[:, 2]] - rest[tets[:, 0]],
            rest[tets[:, 3]] - rest[tets[:, 0]],
        ),
        dim=-1,
    )
    dm_inverse = torch.linalg.inv(edge_matrix)
    shape_gradient = torch.zeros(tets.shape[0], 4, 3, dtype=rest.dtype, device=rest.device)
    shape_gradient[:, 1:] = dm_inverse
    shape_gradient[:, 0] = -shape_gradient[:, 1:].sum(dim=1)
    return shape_gradient, torch.linalg.det(edge_matrix) / 6.0


def _context(
    rest: torch.Tensor,
    tets: torch.Tensor,
    *,
    inertial_target: torch.Tensor | None = None,
    pinned: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    lam: torch.Tensor | None = None,
) -> CommonObjectiveContext:
    shape_gradient, volume = _shape_data(rest, tets)
    n_vertices = rest.shape[0]
    n_tets = tets.shape[0]
    target_offset = torch.linspace(-0.025, 0.035, n_vertices, dtype=rest.dtype)[:, None]
    target_direction = torch.tensor([[0.4, -0.7, 0.2]], dtype=rest.dtype)
    target = rest + target_offset * target_direction if inertial_target is None else inertial_target
    return CommonObjectiveContext(
        tets=tets,
        J=shape_gradient,
        volume=volume,
        mass=torch.linspace(0.8, 1.4, n_vertices, dtype=rest.dtype),
        mu=torch.linspace(17.0, 23.0, n_tets, dtype=rest.dtype) if mu is None else mu,
        lam=torch.linspace(31.0, 41.0, n_tets, dtype=rest.dtype) if lam is None else lam,
        inertial_target=target,
        pinned=torch.empty(0, dtype=torch.int64) if pinned is None else pinned,
        dt=0.08,
    )


def _deformation_gradient(
    positions: torch.Tensor,
    tets: torch.Tensor,
    shape_gradient: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("tac,tad->tdc", shape_gradient, positions[tets])


def _explicit_operator_matrix(
    context: CommonObjectiveContext,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Independently assemble the tiny dense GN matrix for test comparison."""
    n_dofs = 3 * context.n_vertices
    matrix = torch.diag((context.mass / (context.dt * context.dt)).repeat_interleave(3))
    shape_gradient = context.J
    deformation_gradient = _deformation_gradient(positions, context.tets, shape_gradient)
    for tet_index, vertices in enumerate(context.tets):
        local_map = torch.zeros(9, 12, dtype=positions.dtype)
        for corner in range(4):
            for spatial in range(3):
                for material in range(3):
                    local_map[3 * spatial + material, 3 * corner + spatial] = shape_gradient[
                        tet_index, corner, material
                    ]
        local_f = deformation_gradient[tet_index].detach().clone().requires_grad_(True)
        (cofactor,) = torch.autograd.grad(torch.linalg.det(local_f), local_f)
        cofactor_flat = cofactor.reshape(-1)
        density_tangent = context.mu[tet_index] * torch.eye(9, dtype=positions.dtype)
        density_tangent = density_tangent + context.lam[tet_index] * torch.outer(cofactor_flat, cofactor_flat)
        local_matrix = context.volume[tet_index] * local_map.T @ density_tangent @ local_map
        local_dofs = torch.stack((3 * vertices, 3 * vertices + 1, 3 * vertices + 2), dim=-1).reshape(-1)
        matrix[local_dofs[:, None], local_dofs[None, :]] += local_matrix

    for vertex in context.pinned:
        dofs = 3 * vertex + torch.arange(3)
        matrix[dofs, :] = 0.0
        matrix[:, dofs] = 0.0
        matrix[dofs, dofs] = 1.0
    if matrix.shape != (n_dofs, n_dofs):
        raise AssertionError("test assembly produced an invalid matrix")
    return matrix


class TestStableNeoHookeanGaussNewton(unittest.TestCase):
    def test_exact_free_gradient_reuses_common_residual_and_matches_autograd(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0, 4], dtype=torch.int64))
        positions = rest + torch.tensor(
            [
                [0.2, -0.1, 0.05],
                [0.08, 0.03, -0.02],
                [-0.04, 0.09, 0.01],
                [0.02, -0.05, 0.11],
                [-0.08, 0.04, -0.03],
            ],
            dtype=torch.float64,
        )

        actual = common_objective_free_gradient(context, positions)
        authoritative = common_objective_residual(context, positions)
        self.assertTrue(torch.equal(actual, authoritative))

        differentiable = positions.clone().requires_grad_(True)
        (expected,) = torch.autograd.grad(common_objective_components(context, differentiable)["total"], differentiable)
        expected = expected.detach()
        expected[context.pinned] = 0.0
        torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)
        self.assertTrue(torch.equal(actual[context.pinned], torch.zeros_like(actual[context.pinned])))

    def test_matvec_matches_independently_assembled_small_operator(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0, 4], dtype=torch.int64))
        positions = rest + torch.tensor(
            [
                [0.02, -0.01, 0.03],
                [0.11, 0.05, -0.04],
                [-0.03, -0.07, 0.06],
                [0.05, 0.01, 0.08],
                [-0.06, 0.12, -0.02],
            ],
            dtype=torch.float64,
        )
        direction = torch.tensor(
            [
                [0.3, -0.2, 0.1],
                [-0.4, 0.5, 0.2],
                [0.2, 0.1, -0.3],
                [0.1, -0.2, 0.4],
                [-0.5, 0.3, -0.1],
            ],
            dtype=torch.float64,
        )
        explicit = _explicit_operator_matrix(context, positions)
        expected = (explicit @ direction.reshape(-1)).reshape_as(direction)

        actual = stable_neo_hookean_gn_matvec(context, positions, direction)
        torch.testing.assert_close(actual, expected, rtol=2.0e-13, atol=2.0e-13)
        self.assertTrue(torch.equal(actual[context.pinned], direction[context.pinned]))

        direction_without_pins = direction.clone()
        direction_without_pins[context.pinned] = 0.0
        free_action = stable_neo_hookean_gn_matvec(context, positions, direction_without_pins)
        torch.testing.assert_close(actual[1:4], free_action[1:4], rtol=0.0, atol=0.0)

    def test_matvec_matches_exact_residual_jvp_when_volumetric_stress_is_zero(self):
        rest, tets = _single_tet()
        mu = torch.tensor([2.0], dtype=torch.float64)
        lam = torch.tensor([4.0], dtype=torch.float64)
        context = _context(rest, tets, mu=mu, lam=lam)
        alpha = 1.0 + float(mu[0] / lam[0])
        positions = alpha ** (1.0 / 3.0) * rest
        direction = torch.tensor(
            [
                [0.11, -0.07, 0.03],
                [-0.02, 0.06, 0.09],
                [0.05, -0.04, 0.08],
                [-0.03, 0.02, -0.01],
            ],
            dtype=torch.float64,
        )

        _, exact_jvp = torch.autograd.functional.jvp(
            lambda value: common_objective_residual(context, value),
            positions,
            direction,
        )
        actual = stable_neo_hookean_gn_matvec(context, positions, direction)
        torch.testing.assert_close(actual, exact_jvp, rtol=2.0e-12, atol=2.0e-12)

    def test_operator_is_symmetric_positive_definite_and_has_exact_pin_rows(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0], dtype=torch.int64))
        positions = rest + 0.07 * torch.tensor(
            [
                [1.0, -2.0, 0.5],
                [-1.0, 0.2, 0.3],
                [0.4, 0.8, -0.9],
                [-0.5, 0.7, 1.1],
                [0.9, -0.3, 0.6],
            ],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(931)
        left = torch.randn(rest.shape, dtype=rest.dtype, generator=generator)
        right = torch.randn(rest.shape, dtype=rest.dtype, generator=generator)
        left_action = stable_neo_hookean_gn_matvec(context, positions, left)
        right_action = stable_neo_hookean_gn_matvec(context, positions, right)

        torch.testing.assert_close((left * right_action).sum(), (right * left_action).sum(), rtol=3.0e-13, atol=3.0e-13)
        self.assertGreater(float((left * left_action).sum()), 0.0)
        self.assertGreater(float((right * right_action).sum()), 0.0)
        self.assertTrue(torch.equal(left_action[context.pinned], left[context.pinned]))
        self.assertTrue(torch.equal(right_action[context.pinned], right[context.pinned]))

    def test_block_diagonal_matches_dense_diagonal_blocks(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0, 4], dtype=torch.int64))
        positions = rest + 0.05 * torch.tensor(
            [
                [0.2, -0.4, 0.8],
                [1.0, 0.3, -0.7],
                [-0.1, 0.9, 0.5],
                [0.6, -0.8, 0.4],
                [-0.5, 0.2, 1.1],
            ],
            dtype=torch.float64,
        )
        dense = _explicit_operator_matrix(context, positions).reshape(context.n_vertices, 3, context.n_vertices, 3)
        expected = torch.stack([dense[vertex, :, vertex, :] for vertex in range(context.n_vertices)])

        actual = stable_neo_hookean_gn_block_diagonal(context, positions)
        torch.testing.assert_close(actual, expected, rtol=3.0e-13, atol=3.0e-13)
        identity = torch.eye(3, dtype=rest.dtype).expand(context.pinned.numel(), 3, 3)
        self.assertTrue(torch.equal(actual[context.pinned], identity))

    def test_operator_persistent_state_has_no_dense_vertex_matrix(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0], dtype=torch.int64))
        operator = StableNeoHookeanGaussNewtonOperator(context, rest)
        tensors = [
            getattr(operator, name) for name in operator.__slots__ if isinstance(getattr(operator, name), torch.Tensor)
        ]
        forbidden_shape = (3 * context.n_vertices, 3 * context.n_vertices)
        self.assertNotIn(forbidden_shape, [tuple(value.shape) for value in tensors])
        self.assertLessEqual(sum(value.numel() for value in tensors), 13 * context.n_vertices + 9 * context.n_tets)


class TestFixedWorkPCG(unittest.TestCase):
    def test_fixed_work_matches_small_dense_solve(self):
        generator = torch.Generator().manual_seed(1203)
        factor = torch.randn(6, 6, dtype=torch.float64, generator=generator)
        matrix = factor.T @ factor + 0.7 * torch.eye(6, dtype=torch.float64)
        rhs = torch.randn(2, 3, dtype=torch.float64, generator=generator)
        blocks = torch.stack((matrix[:3, :3], matrix[3:, 3:]))
        config = FixedPCGConfig(iterations=8, relative_tolerance=1.0e-12)

        result = fixed_work_block_pcg(
            matvec=lambda value: (matrix @ value.reshape(-1)).reshape_as(value),
            rhs=rhs,
            block_diagonal=blocks,
            pinned=torch.empty(0, dtype=torch.int64),
            config=config,
        )
        expected = torch.linalg.solve(matrix, rhs.reshape(-1)).reshape_as(rhs)
        torch.testing.assert_close(result.solution, expected, rtol=2.0e-11, atol=2.0e-11)
        self.assertTrue(result.trace.converged)
        self.assertFalse(result.trace.breakdown)
        self.assertEqual(result.trace.scheduled_iterations, 8)
        self.assertEqual(result.trace.matrix_vector_products, 9)
        self.assertEqual(result.trace.preconditioner_applications, 9)
        self.assertEqual(result.trace.algorithmic_scalar_reductions, 5 * config.iterations + 3)
        self.assertEqual(result.trace.safeguard_scalar_reductions, 4 * config.iterations + 6)
        self.assertEqual(
            result.trace.scalar_reductions,
            result.trace.algorithmic_scalar_reductions + result.trace.safeguard_scalar_reductions,
        )
        self.assertLessEqual(result.trace.active_iterations, 6)
        self.assertEqual(len(result.trace.active_schedule), 8)

    def test_stationary_system_still_executes_scheduled_work(self):
        rhs = torch.zeros(3, 3, dtype=torch.float64)
        blocks = 2.0 * torch.eye(3, dtype=rhs.dtype).expand(3, 3, 3).clone()
        calls = 0

        def matvec(value: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return 2.0 * value

        result = fixed_work_block_pcg(
            matvec=matvec,
            rhs=rhs,
            block_diagonal=blocks,
            pinned=torch.tensor([0], dtype=torch.int64),
            config=FixedPCGConfig(iterations=5),
        )
        self.assertEqual(calls, 6)
        self.assertTrue(result.trace.stationary)
        self.assertTrue(result.trace.converged)
        self.assertFalse(result.trace.breakdown)
        self.assertEqual(result.trace.active_iterations, 0)
        self.assertEqual(result.trace.active_schedule, (False,) * 5)
        self.assertTrue(torch.equal(result.solution, rhs))

    def test_zero_curvature_breakdown_is_fail_closed_without_shortening_work(self):
        rhs = torch.ones(2, 3, dtype=torch.float64)
        blocks = torch.eye(3, dtype=rhs.dtype).expand(2, 3, 3).clone()
        calls = 0

        def zero_matvec(value: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return torch.zeros_like(value)

        result = fixed_work_block_pcg(
            matvec=zero_matvec,
            rhs=rhs,
            block_diagonal=blocks,
            pinned=torch.empty(0, dtype=torch.int64),
            config=FixedPCGConfig(iterations=4),
        )
        self.assertEqual(calls, 5)
        self.assertTrue(result.trace.breakdown)
        self.assertEqual(result.trace.breakdown_iteration, 0)
        self.assertFalse(result.trace.converged)
        self.assertEqual(result.trace.active_iterations, 1)
        self.assertEqual(result.trace.active_schedule, (True, False, False, False))
        self.assertTrue(torch.equal(result.solution, torch.zeros_like(rhs)))

    def test_non_spd_block_diagonal_is_fail_closed(self):
        rhs = torch.ones(2, 3, dtype=torch.float64)
        blocks = torch.eye(3, dtype=rhs.dtype).expand(2, 3, 3).clone()
        blocks[1, 2, 2] = -1.0
        result = fixed_work_block_pcg(
            matvec=lambda value: value,
            rhs=rhs,
            block_diagonal=blocks,
            pinned=torch.empty(0, dtype=torch.int64),
            config=FixedPCGConfig(iterations=3),
        )
        self.assertTrue(result.trace.breakdown)
        self.assertEqual(result.trace.breakdown_iteration, -1)
        self.assertEqual(result.trace.active_schedule, (False, False, False))
        self.assertTrue(torch.equal(result.solution, torch.zeros_like(rhs)))

    def test_policy_scalars_must_be_representable_in_execution_dtype(self):
        rhs = torch.ones(2, 3, dtype=torch.float32)
        blocks = torch.eye(3, dtype=rhs.dtype).expand(2, 3, 3).clone()
        common = {
            "matvec": lambda value: value,
            "rhs": rhs,
            "block_diagonal": blocks,
            "pinned": torch.empty(0, dtype=torch.int64),
        }
        for name, config in (
            ("overflow", FixedPCGConfig(relative_tolerance=1.0e300)),
            ("underflow", FixedPCGConfig(curvature_relative_tolerance=1.0e-300)),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fixed_work_block_pcg(**common, config=config)


class TestV5Corrector(unittest.TestCase):
    def test_normal_correction_preserves_pins_and_improves_objective_and_residual(self):
        rest, tets = _two_tets()
        pinned = torch.tensor([0], dtype=torch.int64)
        context = _context(rest, tets, pinned=pinned)
        start = rest.clone()
        start[1:] += torch.tensor([0.2, -0.1, 0.1], dtype=rest.dtype)
        config = CorrectorConfig(pcg=FixedPCGConfig(iterations=8, relative_tolerance=1.0e-10))

        result = correct_common_objective(context=context, start=start, config=config)
        self.assertTrue(result.trace.accepted)
        self.assertGreater(result.trace.selected_alpha, 0.0)
        self.assertTrue(result.trace.descent_direction)
        self.assertTrue(torch.equal(result.positions[pinned], start[pinned]))
        selected = result.trace.candidates[result.trace.selected_candidate_index]
        self.assertLessEqual(selected.objective, result.trace.start_objective + config.objective_increase_tolerance)
        self.assertLessEqual(
            selected.raw_residual_norm,
            result.trace.start_raw_residual_norm + config.residual_increase_tolerance,
        )
        self.assertGreater(selected.minimum_determinant, config.minimum_determinant)
        self.assertGreater(selected.minimum_singular_value, config.minimum_singular_value)
        self.assertEqual(result.trace.work.candidate_count, len(config.candidate_alphas))
        self.assertEqual(result.trace.work.candidate_objective_state_evaluations, len(config.candidate_alphas))
        self.assertEqual(result.trace.work.candidate_residual_state_evaluations, len(config.candidate_alphas))
        self.assertEqual(result.trace.work.scheduled_pcg_iterations, config.pcg.iterations)
        self.assertEqual(result.trace.work.tangent_matrix_vector_products, config.pcg.iterations + 1)
        self.assertIn(0.0, [candidate.alpha for candidate in result.trace.candidates])

    def test_stationary_start_preserves_bits_but_runs_all_work(self):
        rest, tets = _single_tet()
        context = _context(
            rest,
            tets,
            inertial_target=rest,
            pinned=torch.tensor([0], dtype=torch.int64),
            mu=torch.tensor([10.0], dtype=torch.float64),
            lam=torch.tensor([20.0], dtype=torch.float64),
        )
        config = CorrectorConfig(pcg=FixedPCGConfig(iterations=5))
        result = correct_common_objective(context=context, start=rest, config=config)

        self.assertFalse(result.trace.accepted)
        self.assertEqual(result.trace.reason, "stationary")
        self.assertTrue(result.trace.pcg.stationary)
        self.assertEqual(result.trace.pcg.matrix_vector_products, 6)
        self.assertEqual(result.trace.pcg.preconditioner_applications, 6)
        self.assertEqual(result.trace.pcg.active_schedule, (False,) * 5)
        self.assertTrue(torch.equal(result.positions, rest))
        self.assertTrue(torch.equal(result.direction, torch.zeros_like(rest)))
        self.assertEqual(len(result.trace.candidates), len(config.candidate_alphas))

    def test_forced_curvature_breakdown_preserves_start_exactly(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0], dtype=torch.int64))
        start = rest.clone()
        start[1:] += torch.tensor([0.15, -0.08, 0.05], dtype=rest.dtype)
        config = CorrectorConfig(
            pcg=FixedPCGConfig(iterations=4, curvature_relative_tolerance=2.0),
            candidate_alphas=(1.0, 0.5, 0.0),
        )
        result = correct_common_objective(context=context, start=start, config=config)

        self.assertFalse(result.trace.accepted)
        self.assertEqual(result.trace.reason, "pcg-breakdown")
        self.assertTrue(result.trace.pcg.breakdown)
        self.assertEqual(result.trace.pcg.breakdown_iteration, 0)
        self.assertEqual(result.trace.pcg.matrix_vector_products, 5)
        self.assertEqual(len(result.trace.candidates), 3)
        self.assertTrue(torch.equal(result.positions, start))
        self.assertTrue(torch.equal(result.direction, torch.zeros_like(start)))

    def test_optional_residual_safeguard_rejects_objective_decreasing_step(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0], dtype=torch.int64))
        offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-0.21345709850697103, 0.6542624358990924, -0.27941392858392683],
                [-0.535250086058693, -0.9221562948350408, -0.23055854239939147],
                [0.18400218816982686, 0.1783438421050755, 0.07394257280323879],
                [-0.17615067749006008, 0.9739324748374905, -0.22393042238545058],
            ],
            dtype=torch.float64,
        )
        start = rest + offsets
        pcg = FixedPCGConfig(iterations=4)
        without_residual_guard = correct_common_objective(
            context=context,
            start=start,
            config=CorrectorConfig(
                pcg=pcg,
                candidate_alphas=(2.0, 0.0),
                require_residual_nonincrease=False,
            ),
        )
        with_residual_guard = correct_common_objective(
            context=context,
            start=start,
            config=CorrectorConfig(
                pcg=pcg,
                candidate_alphas=(2.0, 0.0),
                require_residual_nonincrease=True,
            ),
        )

        unchecked_candidate = without_residual_guard.trace.candidates[0]
        self.assertTrue(unchecked_candidate.objective_nonincreasing)
        self.assertGreater(unchecked_candidate.raw_residual_norm, without_residual_guard.trace.start_raw_residual_norm)
        self.assertTrue(without_residual_guard.trace.accepted)
        self.assertFalse(with_residual_guard.trace.candidates[0].residual_nonincreasing)
        self.assertFalse(with_residual_guard.trace.accepted)
        self.assertTrue(torch.equal(with_residual_guard.positions, start))

    def test_geometry_and_finite_safeguards_preserve_start(self):
        rest, tets = _two_tets()
        context = _context(rest, tets, pinned=torch.tensor([0], dtype=torch.int64))
        start = rest.clone()
        start[1:] += torch.tensor([0.2, -0.1, 0.1], dtype=rest.dtype)

        geometry_result = correct_common_objective(
            context=context,
            start=start,
            config=CorrectorConfig(
                pcg=FixedPCGConfig(iterations=4),
                candidate_alphas=(1.0, 0.0),
                minimum_determinant=2.0,
                minimum_singular_value=1.5,
            ),
        )
        self.assertFalse(geometry_result.trace.accepted)
        self.assertFalse(geometry_result.trace.candidates[0].determinant_valid)
        self.assertFalse(geometry_result.trace.candidates[0].singular_value_valid)
        self.assertTrue(torch.equal(geometry_result.positions, start))

        finite_result = correct_common_objective(
            context=context,
            start=start,
            config=CorrectorConfig(
                pcg=FixedPCGConfig(iterations=4),
                candidate_alphas=(1.0e200, 0.0),
            ),
        )
        self.assertFalse(finite_result.trace.accepted)
        self.assertFalse(finite_result.trace.candidates[0].finite)
        self.assertTrue(torch.equal(finite_result.positions, start))
        for candidate in finite_result.trace.candidates:
            self.assertTrue(candidate.exact_pins)

    def test_candidate_and_safeguard_scalars_must_survive_execution_dtype(self):
        rest64, tets = _single_tet()
        rest = rest64.to(torch.float32)
        context = _context(rest, tets, pinned=torch.tensor([0], dtype=torch.int64))
        cases = (
            CorrectorConfig(candidate_alphas=(1.0e300, 0.0)),
            CorrectorConfig(candidate_alphas=(1.0e-300, 0.0)),
            CorrectorConfig(minimum_determinant=1.0e300),
            CorrectorConfig(minimum_singular_value=1.0e-300),
            CorrectorConfig(objective_increase_tolerance=1.0e300),
            CorrectorConfig(residual_increase_tolerance=1.0e-300),
        )
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                correct_common_objective(context=context, start=rest, config=config)

        with self.assertRaisesRegex(ValueError, "remain distinct"):
            correct_common_objective(
                context=context,
                start=rest,
                config=CorrectorConfig(candidate_alphas=(1.0, 1.0 + 1.0e-10, 0.0)),
            )


if __name__ == "__main__":
    unittest.main()
