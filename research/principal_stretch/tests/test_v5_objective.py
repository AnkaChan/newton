# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the iterative-v5 common-objective residual."""

from __future__ import annotations

import dataclasses
import math
import unittest

import torch

from ..potentials import incremental_potential_stable_neo_hookean
from ..v5_objective import (
    CommonObjectiveContext,
    _common_objective_components_trusted,
    _common_objective_residual_trusted,
    common_objective_components,
    common_objective_residual,
)


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
    dm_inv = torch.linalg.inv(edge_matrix)
    J = torch.zeros(tets.shape[0], 4, 3, dtype=rest.dtype, device=rest.device)
    J[:, 1:] = dm_inv
    J[:, 0] = -J[:, 1:].sum(dim=1)
    volume = torch.linalg.det(edge_matrix) / 6.0
    return J, volume


def _context(
    rest: torch.Tensor,
    tets: torch.Tensor,
    *,
    inertial_target: torch.Tensor | None = None,
    pinned: torch.Tensor | None = None,
) -> CommonObjectiveContext:
    J, volume = _shape_data(rest, tets)
    n_vertices = rest.shape[0]
    n_tets = tets.shape[0]
    target_offset = torch.linspace(-0.025, 0.035, n_vertices, dtype=rest.dtype)[:, None]
    target_direction = torch.tensor([[0.4, -0.7, 0.2]], dtype=rest.dtype)
    target = rest + target_offset * target_direction if inertial_target is None else inertial_target
    return CommonObjectiveContext(
        tets=tets,
        J=J,
        volume=volume,
        mass=torch.linspace(0.8, 1.4, n_vertices, dtype=rest.dtype),
        mu=torch.linspace(17.0, 23.0, n_tets, dtype=rest.dtype),
        lam=torch.linspace(31.0, 41.0, n_tets, dtype=rest.dtype),
        inertial_target=target,
        pinned=torch.empty(0, dtype=torch.int64) if pinned is None else pinned,
        dt=0.08,
    )


def _existing_components(context: CommonObjectiveContext, positions: torch.Tensor) -> dict[str, torch.Tensor]:
    return incremental_potential_stable_neo_hookean(
        positions,
        context.inertial_target,
        context.mass,
        context.tets,
        context.J,
        context.mu,
        context.lam,
        context.volume,
        context.dt,
    )


def _existing_residual(context: CommonObjectiveContext, positions: torch.Tensor) -> torch.Tensor:
    candidate = positions.detach().clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_existing_components(context, candidate)["total"], candidate)
    expected = gradient.detach().clone()
    expected[context.pinned] = 0.0
    return expected


class TestV5CommonObjective(unittest.TestCase):
    def test_unbound_common_objective_hash_remains_golden(self) -> None:
        """Preserve the existing unbound objective identity exactly."""
        rest, tets = _single_tet()
        context = _context(rest, tets)

        self.assertEqual(
            context.common_objective_sha256,
            "ee96547e4a041ac5512b5492a12ce46914771ca35bc63708ce0e78ba3aadce53",
        )

    def test_small_lambda_preserves_the_authenticated_vbd_alpha_floor(self):
        rest, tets = _single_tet()
        J, volume = _shape_data(rest, tets)
        context = CommonObjectiveContext(
            tets=tets,
            J=J,
            volume=volume,
            mass=torch.ones(rest.shape[0], dtype=rest.dtype),
            mu=torch.tensor([2.0], dtype=rest.dtype),
            lam=torch.tensor([1.0e-8], dtype=rest.dtype),
            inertial_target=rest,
            pinned=torch.empty(0, dtype=torch.int64),
            dt=0.08,
        )

        actual_components = common_objective_components(context, rest)
        expected_components = _existing_components(context, rest)
        actual_residual = common_objective_residual(context, rest)
        expected_residual = _existing_residual(context, rest)

        for name in ("total", "inertia", "elastic"):
            torch.testing.assert_close(actual_components[name], expected_components[name], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_residual, expected_residual, rtol=2.0e-12, atol=2.0e-12)
        self.assertGreater(actual_residual.abs().max().item(), 0.0)

    def test_components_and_residual_match_existing_autograd_on_distorted_meshes(self):
        single_rest, single_tets = _single_tet()
        healthy_rest, healthy_tets = _two_tets()
        healthy = healthy_rest + torch.tensor(
            [
                [0.03, -0.01, 0.02],
                [0.11, 0.04, -0.03],
                [-0.02, -0.08, 0.06],
                [0.05, 0.02, 0.09],
                [-0.07, 0.12, -0.04],
            ],
            dtype=torch.float64,
        )
        inverted = single_rest.clone()
        inverted[1] += torch.tensor([0.12, 0.03, -0.04], dtype=torch.float64)
        inverted[3] = torch.tensor([0.04, -0.03, -0.35], dtype=torch.float64)
        near_flat = single_rest.clone()
        near_flat[1] += torch.tensor([0.07, -0.02, 0.01], dtype=torch.float64)
        near_flat[2] += torch.tensor([-0.03, 0.05, -0.02], dtype=torch.float64)
        near_flat[3] = torch.tensor([0.2, 0.25, 1.0e-12], dtype=torch.float64)

        cases = {
            "healthy-two-tet": (healthy_rest, healthy_tets, healthy),
            "inverted": (single_rest, single_tets, inverted),
            "near-flat": (single_rest, single_tets, near_flat),
        }
        for name, (rest, tets, positions) in cases.items():
            with self.subTest(name=name):
                context = _context(rest, tets)
                actual_components = common_objective_components(context, positions)
                expected_components = _existing_components(context, positions)
                for component in ("total", "inertia", "elastic"):
                    torch.testing.assert_close(
                        actual_components[component],
                        expected_components[component],
                        rtol=2.0e-13,
                        atol=2.0e-13,
                    )

                actual_residual = common_objective_residual(context, positions)
                expected_residual = _existing_residual(context, positions)
                torch.testing.assert_close(actual_residual, expected_residual, rtol=2.0e-12, atol=2.0e-12)
                self.assertTrue(torch.isfinite(actual_residual).all())
                self.assertTrue(all(torch.isfinite(value).all() for value in actual_components.values()))

    def test_leading_batch_dimensions_equal_independent_evaluations(self):
        rest, tets = _two_tets()
        context = _context(rest, tets)
        offsets = torch.tensor(
            [
                [0.02, -0.03, 0.01],
                [-0.04, 0.01, 0.03],
                [0.01, 0.02, -0.02],
                [0.03, -0.01, 0.04],
                [-0.02, 0.05, -0.03],
            ],
            dtype=torch.float64,
        )
        candidates = torch.stack((rest + offsets, rest - 0.7 * offsets, rest + 1.3 * offsets, rest))
        positions = candidates.reshape(2, 2, *rest.shape)

        batched_components = common_objective_components(context, positions)
        independent_components = {
            name: torch.stack(
                [common_objective_components(context, candidate)[name] for candidate in candidates]
            ).reshape(2, 2)
            for name in ("total", "inertia", "elastic")
        }
        for name, expected in independent_components.items():
            torch.testing.assert_close(batched_components[name], expected, rtol=0.0, atol=0.0)

        batched_residual = common_objective_residual(context, positions)
        independent_residual = torch.stack([common_objective_residual(context, candidate) for candidate in candidates])
        independent_residual = independent_residual.reshape_as(positions)
        torch.testing.assert_close(batched_residual, independent_residual, rtol=0.0, atol=0.0)

    def test_pinned_rows_are_exact_zero_and_free_rows_match_autograd(self):
        rest, tets = _two_tets()
        pinned = torch.tensor([0, 4], dtype=torch.int64)
        context = _context(rest, tets, pinned=pinned)
        positions = rest + torch.tensor(
            [
                [0.3, -0.2, 0.1],
                [0.04, 0.02, -0.01],
                [-0.03, 0.05, 0.02],
                [0.01, -0.04, 0.03],
                [-0.2, 0.1, 0.25],
            ],
            dtype=torch.float64,
        )
        actual = common_objective_residual(context, positions)
        expected = _existing_residual(context, positions)
        torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)
        self.assertTrue(torch.equal(actual[pinned], torch.zeros_like(actual[pinned])))

    def test_detached_normalized_feature_is_bound_to_scale_and_does_not_mutate_positions(self):
        rest, tets = _two_tets()
        context = _context(rest, tets)
        positions = (rest + 0.03 * torch.sin(2.0 * rest + 0.4)).requires_grad_(True)
        original = positions.detach().clone()

        raw = common_objective_residual(context, positions)
        feature = common_objective_residual(context, positions, normalize=True, detach=True)

        self.assertTrue(raw.requires_grad)
        self.assertFalse(feature.requires_grad)
        self.assertIsNone(feature.grad_fn)
        torch.testing.assert_close(feature, raw.detach() / context.residual_scale, rtol=0.0, atol=0.0)
        torch.testing.assert_close(positions.detach(), original, rtol=0.0, atol=0.0)

    def test_active_rigid_motion_covariance(self):
        rest, tets = _two_tets()
        pinned = torch.tensor([0], dtype=torch.int64)
        context = _context(rest, tets, pinned=pinned)
        positions = rest + 0.04 * torch.cos(1.7 * rest + 0.2)
        angle = 0.63
        axis = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        axis = axis / torch.linalg.vector_norm(axis)
        cross = torch.tensor(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=torch.float64,
        )
        rotation = (
            torch.eye(3, dtype=torch.float64) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
        )
        translation = torch.tensor([0.7, -0.25, 0.4], dtype=torch.float64)

        transformed_target = context.inertial_target @ rotation.T + translation
        transformed_context = CommonObjectiveContext(
            tets=context.tets,
            J=context.J,
            volume=context.volume,
            mass=context.mass,
            mu=context.mu,
            lam=context.lam,
            inertial_target=transformed_target,
            pinned=context.pinned,
            dt=context.dt,
        )
        transformed_positions = positions @ rotation.T + translation

        original_components = common_objective_components(context, positions)
        transformed_components = common_objective_components(transformed_context, transformed_positions)
        for name in ("total", "inertia", "elastic"):
            torch.testing.assert_close(
                transformed_components[name], original_components[name], rtol=2.0e-13, atol=2.0e-13
            )

        original_residual = common_objective_residual(context, positions)
        transformed_residual = common_objective_residual(transformed_context, transformed_positions)
        torch.testing.assert_close(
            transformed_residual,
            original_residual @ rotation.T,
            rtol=3.0e-13,
            atol=3.0e-13,
        )

    def test_context_owns_immutable_validated_device_tensors(self):
        rest, tets = _single_tet()
        J, volume = _shape_data(rest, tets)
        mass = torch.ones(4, dtype=torch.float64, requires_grad=True)
        context = CommonObjectiveContext(
            tets=tets,
            J=J,
            volume=volume,
            mass=mass,
            mu=torch.tensor([17.0], dtype=torch.float64),
            lam=torch.tensor([31.0], dtype=torch.float64),
            inertial_target=rest,
            pinned=torch.tensor([0], dtype=torch.int64),
            dt=0.1,
        )
        with torch.no_grad():
            mass.fill_(99.0)
        self.assertTrue(torch.equal(context.mass, torch.ones_like(context.mass)))
        self.assertFalse(any(tensor.requires_grad for tensor in (context.mass, context.J, context.inertial_target)))
        self.assertEqual(context.device, rest.device)
        self.assertEqual(context.dtype, rest.dtype)
        self.assertEqual(len(context.common_objective_sha256), 64)
        length = context.volume.sum().pow(1.0 / 3.0)
        expected_material_scale = (context.volume * (context.mu + context.lam)).sum() / length
        expected_inertial_scale = context.mass[1:].sum() * length / context.dt**2
        self.assertEqual(context.residual_scale, float(torch.maximum(expected_material_scale, expected_inertial_scale)))
        changed = dataclasses.replace(context, mass=context.mass + 1.0)
        self.assertNotEqual(changed.common_objective_sha256, context.common_objective_sha256)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            context.residual_scale = 3.0

        with self.assertRaisesRegex(ValueError, "duplicates"):
            dataclasses.replace(context, pinned=torch.tensor([0, 0], dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "inverse square"):
            dataclasses.replace(context, dt=1.0e-300)
        with self.assertRaisesRegex(ValueError, "inverse square"):
            dataclasses.replace(context, dt=1.0e300)
        with self.assertRaisesRegex(ValueError, "execution dtype|derived residual scale"):
            dataclasses.replace(
                context,
                mu=torch.full_like(context.mu, torch.finfo(context.dtype).max),
                lam=torch.full_like(context.lam, torch.finfo(context.dtype).max),
            )

        expected_residual = common_objective_residual(context, rest)
        context.mass.data.fill_(123.0)
        context.inertial_target.numpy()[...] = -456.0
        torch.testing.assert_close(common_objective_residual(context, rest), expected_residual, rtol=0.0, atol=0.0)

        # Internal zero-copy views exist only for authenticated solver scopes;
        # even an explicit mutation through one must invalidate the context at
        # the next trust boundary.
        context._owned_tensor("mass").fill_(7.0)
        with self.assertRaisesRegex(RuntimeError, "changed after authentication"):
            context.validate_immutable()
        with self.assertRaisesRegex(RuntimeError, "changed after authentication"):
            common_objective_residual(context, rest)

    def test_float32_context_rejects_unrepresentable_timestep_derivations(self):
        rest64, tets = _single_tet()
        context = _context(rest64.to(torch.float32), tets)

        with self.assertRaisesRegex(ValueError, "inverse timestep square"):
            dataclasses.replace(context, dt=1.0e-20)
        with self.assertRaisesRegex(ValueError, "inverse timestep square"):
            dataclasses.replace(context, dt=1.0e23)
        with self.assertRaisesRegex(ValueError, "mass-times-inverse-timestep-square"):
            dataclasses.replace(context, dt=6.0e-20)

    def test_float32_context_rejects_unrepresentable_material_derivations(self):
        rest64, tets = _single_tet()
        context = _context(rest64.to(torch.float32), tets)

        with self.assertRaisesRegex(ValueError, "stable Neo-Hookean alpha"):
            dataclasses.replace(
                context,
                mu=torch.tensor([3.0e38], dtype=torch.float32),
                lam=torch.tensor([1.0e-6], dtype=torch.float32),
            )
        with self.assertRaisesRegex(ValueError, "alpha-squared"):
            dataclasses.replace(
                context,
                mu=torch.tensor([1.0e14], dtype=torch.float32),
                lam=torch.tensor([1.0e-6], dtype=torch.float32),
            )
        with self.assertRaisesRegex(ValueError, "volume-times-mu"):
            dataclasses.replace(
                context,
                volume=torch.tensor([2.0], dtype=torch.float32),
                mu=torch.tensor([2.0e38], dtype=torch.float32),
                lam=torch.tensor([2.0e38], dtype=torch.float32),
            )

    def test_float32_context_rejects_unrepresentable_residual_scale(self):
        rest64, tets = _single_tet()
        context = _context(rest64.to(torch.float32), tets)
        with self.assertRaisesRegex(ValueError, "derived residual scale"):
            dataclasses.replace(
                context,
                volume=torch.tensor([1.0e30], dtype=torch.float32),
                mass=torch.ones(4, dtype=torch.float32),
                mu=torch.zeros(1, dtype=torch.float32),
                lam=torch.zeros(1, dtype=torch.float32),
                dt=math.sqrt(1.0e-29),
            )

    def test_public_float32_evaluators_reject_nonfinite_dynamic_outputs(self):
        rest64, tets = _single_tet()
        rest = rest64.to(torch.float32)
        context = _context(rest, tets)
        overflowing_positions = 1.0e20 * rest

        with self.assertRaisesRegex(RuntimeError, "common-objective .* non-finite"):
            common_objective_components(context, overflowing_positions)
        with self.assertRaisesRegex(RuntimeError, "common-objective residual .* non-finite"):
            common_objective_residual(context, overflowing_positions)

        # Trusted hot helpers avoid a host-visible finite reduction. Their
        # iterative/corrector callers own the pre-commit validity check.
        trusted_components = _common_objective_components_trusted(context, overflowing_positions)
        trusted_residual = _common_objective_residual_trusted(context, overflowing_positions)
        self.assertTrue(any(not torch.isfinite(value).all() for value in trusted_components.values()))
        self.assertFalse(torch.isfinite(trusted_residual).all())


if __name__ == "__main__":
    unittest.main()
