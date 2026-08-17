# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for architecture-v5 representation labels and staged losses."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import torch

from research.principal_stretch.v5_objective import CommonObjectiveContext, common_objective_components
from research.principal_stretch.v5_training import (
    CompatibleStateLossConfig,
    PotentialExcessLossConfig,
    PrincipalStretchLabelConfig,
    RepresentationLossConfig,
    build_principal_stretch_labels,
    common_potential_excess_loss,
    common_potential_excess_loss_batch,
    compatible_state_loss,
    principal_stretch_representation_loss,
)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vector[..., 0])
    return torch.stack(
        (
            torch.stack((zero, -vector[..., 2], vector[..., 1]), dim=-1),
            torch.stack((vector[..., 2], zero, -vector[..., 0]), dim=-1),
            torch.stack((-vector[..., 1], vector[..., 0], zero), dim=-1),
        ),
        dim=-2,
    )


def _rotation(vector: torch.Tensor) -> torch.Tensor:
    return torch.matrix_exp(_skew(vector))


def _tet_data(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    return {
        "positions": positions,
        "tets": torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
        "J": torch.tensor(
            [[[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=dtype,
        ),
        "volume": torch.tensor([1.0 / 6.0], dtype=dtype),
        "mass": torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=dtype),
        "pinned": torch.tensor([0], dtype=torch.int64),
    }


class TestPrincipalStretchLabels(unittest.TestCase):
    def _fields(
        self, *, dtype: torch.dtype = torch.float64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = torch.tensor([[0.08, 0.01, 0.00], [0.01, -0.04, 0.015], [0.00, 0.015, 0.02]], dtype=dtype)
        delta_h = torch.tensor([[0.025, -0.008, 0.004], [-0.008, -0.015, 0.006], [0.004, 0.006, 0.01]], dtype=dtype)
        a0 = _rotation(torch.tensor([0.12, -0.08, 0.04], dtype=dtype))
        omega = torch.tensor([-0.06, 0.09, 0.03], dtype=dtype)
        f0 = a0 @ torch.matrix_exp(h0)
        target = a0 @ _rotation(omega) @ torch.matrix_exp(h0 + delta_h)
        return f0, target, delta_h, omega

    def test_exact_labels_reconstruct_and_preserve_arbitrary_batch(self):
        f0, target, expected_delta_h, expected_omega = self._fields()
        f0 = torch.stack((f0, f0, f0, f0)).reshape(2, 2, 3, 3)
        target = torch.stack((target, target, target, target)).reshape(2, 2, 3, 3)
        labels = build_principal_stretch_labels(
            f0,
            target,
            PrincipalStretchLabelConfig(max_hencky_update=0.2, max_rotation_update=0.4),
        )

        self.assertTrue(dataclasses.is_dataclass(labels))
        self.assertEqual(labels.delta_H.shape, (2, 2, 3, 3))
        self.assertEqual(labels.omega.shape, (2, 2, 3))
        torch.testing.assert_close(labels.delta_H[0, 0], expected_delta_h, rtol=1.0e-10, atol=1.0e-11)
        torch.testing.assert_close(labels.omega[0, 0], expected_omega, rtol=1.0e-10, atol=1.0e-11)
        torch.testing.assert_close(labels.reconstructed_F, target, rtol=1.0e-10, atol=1.0e-11)
        torch.testing.assert_close(labels.H_star, labels.H_target)
        torch.testing.assert_close(labels.A_star, labels.A_target)
        self.assertFalse(labels.diagnostics.floor_would_activate)
        self.assertFalse(labels.diagnostics.near_pi_branch)
        self.assertLess(labels.diagnostics.maximum_hencky_cap_ratio, 1.0)
        self.assertLess(labels.diagnostics.maximum_rotation_cap_ratio, 1.0)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            labels.delta_H = torch.zeros_like(labels.delta_H)

    def test_float32_identity_labels_are_finite_and_exact(self):
        identity = torch.eye(3, dtype=torch.float32).expand(2, 3, 3).clone()
        labels = build_principal_stretch_labels(
            identity,
            identity,
            PrincipalStretchLabelConfig(max_hencky_update=0.2, max_rotation_update=0.4),
        )

        for value in (
            labels.delta_H,
            labels.omega,
            labels.H_star,
            labels.A_star,
            labels.H_target,
            labels.A_target,
            labels.reconstructed_F,
        ):
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(labels.delta_H, torch.zeros_like(labels.delta_H), rtol=0.0, atol=0.0)
        torch.testing.assert_close(labels.omega, torch.zeros_like(labels.omega), rtol=0.0, atol=0.0)
        torch.testing.assert_close(labels.reconstructed_F, identity, rtol=0.0, atol=0.0)

    def test_invalid_labels_fail_closed(self):
        f0, target, _delta_h, _omega = self._fields()
        config = PrincipalStretchLabelConfig(max_hencky_update=0.2, max_rotation_update=0.4)
        cases = (
            (torch.full_like(f0, float("nan")), target, "finite"),
            (f0, -target, "positive determinant"),
            (f0, target[..., :2, :2], "same shape"),
        )
        for initial, reference, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                build_principal_stretch_labels(initial, reference, config)

        too_flat = torch.diag(torch.tensor([0.01, 1.0, 1.0], dtype=f0.dtype))
        with self.assertRaisesRegex(ValueError, "principal-stretch floor"):
            build_principal_stretch_labels(f0, too_flat, config)

        too_much_stretch = target @ torch.diag(torch.tensor([1.8, 1.0, 1.0], dtype=f0.dtype))
        with self.assertRaisesRegex(ValueError, "Hencky label exceeds"):
            build_principal_stretch_labels(f0, too_much_stretch, config)

        near_pi = f0 @ _rotation(torch.tensor([3.05, 0.0, 0.0], dtype=f0.dtype))
        branch_config = PrincipalStretchLabelConfig(max_hencky_update=2.0, max_rotation_update=2.9)
        with self.assertRaisesRegex(ValueError, r"principal SO\(3\) branch"):
            build_principal_stretch_labels(f0, near_pi, branch_config)

    def test_reconstruction_miss_is_not_accepted(self):
        f0, target, _delta_h, _omega = self._fields(dtype=torch.float32)
        strict = PrincipalStretchLabelConfig(
            max_hencky_update=0.2,
            max_rotation_update=0.4,
            reconstruction_relative_tolerance=1.0e-12,
            reconstruction_absolute_tolerance=1.0e-12,
        )
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            build_principal_stretch_labels(f0, target, strict)

    def test_label_config_must_remain_representable_in_execution_dtype(self):
        f0, target, _delta_h, _omega = self._fields(dtype=torch.float32)
        cases = (
            (PrincipalStretchLabelConfig(max_hencky_update=1.0e300), "max_hencky_update.*execution dtype"),
            (
                PrincipalStretchLabelConfig(minimum_principal_stretch=1.0e-30),
                "right_cauchy_green_eigenvalue_floor.*execution dtype",
            ),
            (
                PrincipalStretchLabelConfig(reconstruction_absolute_tolerance=1.0e-50),
                "reconstruction_absolute_tolerance.*execution dtype",
            ),
        )
        for config, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_principal_stretch_labels(f0, target, config)


class TestV5StagedLosses(unittest.TestCase):
    def test_supervision_targets_are_detached(self):
        predicted_h = torch.zeros(1, 3, 3, dtype=torch.float64, requires_grad=True)
        predicted_omega = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
        target_h = torch.eye(3, dtype=torch.float64).mul(0.01).unsqueeze(0).requires_grad_()
        target_omega = torch.tensor([[0.01, -0.02, 0.015]], dtype=torch.float64, requires_grad=True)
        representation = principal_stretch_representation_loss(
            predicted_h,
            predicted_omega,
            target_h,
            target_omega,
            RepresentationLossConfig(),
            volume=torch.ones(1, dtype=torch.float64),
        )
        representation.total.backward()
        self.assertIsNone(target_h.grad)
        self.assertIsNone(target_omega.grad)
        self.assertGreater(predicted_h.grad.abs().max().item(), 0.0)
        self.assertGreater(predicted_omega.grad.abs().max().item(), 0.0)

        data = _tet_data()
        reference = data["positions"].clone().requires_grad_()
        predicted = reference.detach().clone()
        predicted[1:] *= 1.02
        predicted.requires_grad_()
        compatible = compatible_state_loss(
            predicted,
            reference,
            tets=data["tets"],
            J=data["J"],
            volume=data["volume"],
            mass=data["mass"],
            pinned=data["pinned"],
            config=CompatibleStateLossConfig(),
        )
        compatible.total.backward()
        self.assertIsNone(reference.grad)
        self.assertGreater(predicted.grad.abs().max().item(), 0.0)

    def test_distinct_physical_objectives_are_routed_per_batch_sample(self):
        data = _tet_data()
        shifted_target = data["positions"].clone()
        shifted_target[1:] += torch.tensor([0.03, -0.02, 0.01], dtype=torch.float64)
        contexts = tuple(
            CommonObjectiveContext(
                tets=data["tets"],
                J=data["J"],
                volume=data["volume"],
                mass=data["mass"],
                mu=torch.tensor([4.0], dtype=torch.float64),
                lam=torch.tensor([7.0], dtype=torch.float64),
                inertial_target=target,
                pinned=data["pinned"],
                dt=0.5,
            )
            for target in (data["positions"], shifted_target)
        )
        reference = torch.stack((data["positions"], data["positions"]))
        predicted = reference.clone()
        predicted[:, 1:] *= 1.02
        baseline = reference.clone()
        baseline[:, 1:] *= 1.05
        config = PotentialExcessLossConfig()

        batched = common_potential_excess_loss_batch(contexts, predicted, reference, baseline, config)
        independent = tuple(
            common_potential_excess_loss(context, predicted[index], reference[index], baseline[index], config)
            for index, context in enumerate(contexts)
        )
        torch.testing.assert_close(
            batched.per_sample_total,
            torch.stack(tuple(result.total for result in independent)),
        )
        shared_context = common_potential_excess_loss(contexts[0], predicted, reference, baseline, config)
        self.assertGreater(
            (shared_context.per_sample_total[1] - batched.per_sample_total[1]).abs().item(),
            1.0e-6,
        )

    def test_representation_loss_is_cap_normalized_and_differentiable(self):
        predicted_h = torch.zeros(2, 3, 3, dtype=torch.float64, requires_grad=True)
        predicted_omega = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        target_h = torch.zeros_like(predicted_h)
        target_h[1, 0, 0] = 0.1
        target_omega = torch.zeros_like(predicted_omega)
        target_omega[1, 2] = 0.2
        volume = torch.tensor([1.0, 3.0], dtype=torch.float64)
        config = RepresentationLossConfig(
            max_hencky_update=0.2,
            max_rotation_update=0.4,
            hencky_weight=2.0,
            rotation_weight=3.0,
        )
        result = principal_stretch_representation_loss(
            predicted_h,
            predicted_omega,
            target_h,
            target_omega,
            config,
            volume=volume,
        )

        self.assertAlmostEqual(result.hencky.item(), 0.1875)
        self.assertAlmostEqual(result.rotation.item(), 0.1875)
        self.assertAlmostEqual(result.total.item(), 0.9375)
        result.total.backward()
        self.assertTrue(torch.isfinite(predicted_h.grad).all())
        self.assertTrue(torch.isfinite(predicted_omega.grad).all())

    def test_compatible_loss_uses_fixed_physical_floors_and_exact_pins(self):
        data = _tet_data()
        reference = data["positions"]
        predicted = reference.clone()
        predicted[1:, :] *= 1.02
        predicted = predicted.requires_grad_()
        config = CompatibleStateLossConfig(
            characteristic_length_m=2.0,
            position_denominator_floor_kg_m2=10.0,
            deformation_denominator_floor_m3=5.0,
            position_weight=2.0,
            deformation_weight=3.0,
        )
        result = compatible_state_loss(
            predicted,
            reference,
            tets=data["tets"],
            J=data["J"],
            volume=data["volume"],
            mass=data["mass"],
            pinned=data["pinned"],
            config=config,
        )

        position_numerator = (data["mass"][1:, None] * (predicted[1:] - reference[1:]).square()).sum()
        f_delta = 0.02 * torch.eye(3, dtype=torch.float64)
        deformation_numerator = (data["volume"] * f_delta.square().sum()).sum()
        self.assertAlmostEqual(result.position.item(), (position_numerator / 10.0).item())
        self.assertAlmostEqual(result.projected_F.item(), (deformation_numerator / 5.0).item())
        self.assertTrue(result.position_floor_active)
        self.assertTrue(result.deformation_floor_active)
        result.total.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

        bad_pin = predicted.detach().clone()
        bad_pin[0, 0] = 1.0e-8
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            compatible_state_loss(
                bad_pin,
                reference,
                tets=data["tets"],
                J=data["J"],
                volume=data["volume"],
                mass=data["mass"],
                pinned=data["pinned"],
                config=config,
            )

        inverted = reference.clone()
        inverted[[1, 2]] = inverted[[2, 1]]
        with self.assertRaisesRegex(ValueError, "positive determinant"):
            compatible_state_loss(
                inverted,
                reference,
                tets=data["tets"],
                J=data["J"],
                volume=data["volume"],
                mass=data["mass"],
                pinned=data["pinned"],
                config=config,
            )

    def test_common_potential_excess_keeps_signed_excess_and_is_first_order(self):
        data = _tet_data()
        reference = data["positions"].clone()
        reference[1:] *= 1.05
        predicted = data["positions"].clone()
        predicted[1:] *= 1.02
        predicted.requires_grad_()
        baseline = data["positions"].clone()
        baseline[1:] *= 1.10
        context = CommonObjectiveContext(
            tets=data["tets"],
            J=data["J"],
            volume=data["volume"],
            mass=data["mass"],
            mu=torch.tensor([4.0], dtype=torch.float64),
            lam=torch.tensor([7.0], dtype=torch.float64),
            inertial_target=data["positions"],
            pinned=data["pinned"],
            dt=0.5,
        )
        config = PotentialExcessLossConfig(denominator_floor_joules=1.0e-8)
        with mock.patch("torch.autograd.grad", side_effect=AssertionError("loss must not form residual Hessians")):
            result = common_potential_excess_loss(context, predicted, reference, baseline, config)

        components = {
            name: common_objective_components(context, value)["total"]
            for name, value in (("predicted", predicted), ("reference", reference), ("baseline", baseline))
        }
        expected_excess = components["predicted"] - components["reference"]
        expected_baseline = components["baseline"] - components["reference"]
        expected_denominator = max(abs(expected_baseline.item()), config.denominator_floor_joules)
        self.assertLess(result.excess.item(), 0.0)
        self.assertAlmostEqual(result.excess.item(), expected_excess.item())
        self.assertAlmostEqual(result.denominator.item(), expected_denominator)
        self.assertAlmostEqual(result.total.item(), expected_excess.item() / expected_denominator)
        self.assertTrue(result.negative_excess)
        (gradient,) = torch.autograd.grad(result.total, predicted)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_staged_loss_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            principal_stretch_representation_loss(
                torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
                torch.zeros(1, 3),
                torch.zeros(1, 3, 3),
                torch.zeros(1, 3),
                RepresentationLossConfig(),
                volume=torch.ones(1),
            )
        with self.assertRaisesRegex(ValueError, "predicted Hencky"):
            principal_stretch_representation_loss(
                0.3 * torch.eye(3)[None],
                torch.zeros(1, 3),
                torch.zeros(1, 3, 3),
                torch.zeros(1, 3),
                RepresentationLossConfig(max_hencky_update=0.2),
                volume=torch.ones(1),
            )

        data = _tet_data()
        context = CommonObjectiveContext(
            tets=data["tets"],
            J=data["J"],
            volume=data["volume"],
            mass=data["mass"],
            mu=torch.tensor([4.0], dtype=torch.float64),
            lam=torch.tensor([7.0], dtype=torch.float64),
            inertial_target=data["positions"],
            pinned=data["pinned"],
            dt=0.5,
        )
        inverted = data["positions"].clone()
        inverted[[1, 2]] = inverted[[2, 1]]
        with self.assertRaisesRegex(ValueError, "positive determinant"):
            common_potential_excess_loss(
                context,
                inverted,
                data["positions"],
                data["positions"],
                PotentialExcessLossConfig(),
            )

        deformed_reference = data["positions"].clone()
        deformed_reference[1:] *= 1.05
        with self.assertRaisesRegex(ValueError, "higher potential"):
            common_potential_excess_loss(
                context,
                deformed_reference,
                deformed_reference,
                data["positions"],
                PotentialExcessLossConfig(negative_baseline_tolerance_joules=0.0),
            )

    def test_execution_dtype_overflow_and_underflow_fail_closed(self):
        dtype = torch.float32
        predicted_h = torch.zeros(1, 3, 3, dtype=dtype, requires_grad=True)
        target_h = torch.zeros_like(predicted_h)
        target_h[0, 0, 0] = 0.1
        predicted_omega = torch.zeros(1, 3, dtype=dtype, requires_grad=True)
        target_omega = torch.zeros_like(predicted_omega)
        volume = torch.ones(1, dtype=dtype)

        with self.assertRaisesRegex(ValueError, "max_hencky_update.*execution dtype"):
            principal_stretch_representation_loss(
                predicted_h,
                predicted_omega,
                target_h,
                target_omega,
                RepresentationLossConfig(max_hencky_update=1.0e300),
                volume=volume,
            )
        with self.assertRaisesRegex(ValueError, "Hencky cap-squared denominator"):
            principal_stretch_representation_loss(
                predicted_h,
                predicted_omega,
                target_h,
                target_omega,
                RepresentationLossConfig(max_hencky_update=1.0e30),
                volume=volume,
            )

        data = _tet_data(dtype)
        compatible_candidate = data["positions"].clone()
        compatible_candidate[1:] *= 1.02
        with self.assertRaisesRegex(ValueError, "characteristic_length_m.*execution dtype"):
            compatible_state_loss(
                compatible_candidate,
                data["positions"],
                tets=data["tets"],
                J=data["J"],
                volume=data["volume"],
                mass=data["mass"],
                pinned=data["pinned"],
                config=CompatibleStateLossConfig(
                    characteristic_length_m=1.0e150,
                    position_weight=1.0,
                    deformation_weight=0.0,
                ),
            )
        with self.assertRaisesRegex(ValueError, "characteristic-length squared"):
            compatible_state_loss(
                compatible_candidate,
                data["positions"],
                tets=data["tets"],
                J=data["J"],
                volume=data["volume"],
                mass=data["mass"],
                pinned=data["pinned"],
                config=CompatibleStateLossConfig(
                    characteristic_length_m=1.0e30,
                    position_weight=1.0,
                    deformation_weight=0.0,
                ),
            )

        context = CommonObjectiveContext(
            tets=data["tets"],
            J=data["J"],
            volume=data["volume"],
            mass=data["mass"],
            mu=torch.tensor([4.0], dtype=dtype),
            lam=torch.tensor([7.0], dtype=dtype),
            inertial_target=data["positions"],
            pinned=data["pinned"],
            dt=0.5,
        )
        reference = data["positions"]
        predicted = compatible_candidate.requires_grad_()
        with self.assertRaisesRegex(ValueError, "denominator_floor_joules.*execution dtype"):
            common_potential_excess_loss(
                context,
                predicted,
                reference,
                reference,
                PotentialExcessLossConfig(denominator_floor_joules=1.0e-50),
            )
        with self.assertRaisesRegex(ValueError, "common-potential loss.*non-finite"):
            common_potential_excess_loss(
                context,
                predicted,
                reference,
                reference,
                PotentialExcessLossConfig(weight=3.0e38),
            )

    def test_configs_are_frozen_and_validate_physical_scales(self):
        config = RepresentationLossConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.max_hencky_update = 1.0
        invalid = (
            (PrincipalStretchLabelConfig, {"minimum_principal_stretch": 0.0}),
            (RepresentationLossConfig, {"max_rotation_update": float("nan")}),
            (CompatibleStateLossConfig, {"characteristic_length_m": 0.0}),
            (PotentialExcessLossConfig, {"denominator_floor_joules": 0.0}),
        )
        for cls, kwargs in invalid:
            with self.subTest(cls=cls.__name__), self.assertRaisesRegex(ValueError, "finite and positive"):
                cls(**kwargs)

    def test_compatible_and_potential_results_preserve_batch_axes(self):
        data = _tet_data()
        reference = torch.stack((data["positions"], data["positions"]))
        predicted = reference.clone()
        predicted[0, 1:] *= 1.01
        predicted[1, 1:] *= 1.02
        compatible = compatible_state_loss(
            predicted,
            reference,
            tets=data["tets"],
            J=data["J"],
            volume=data["volume"],
            mass=data["mass"],
            pinned=data["pinned"],
            config=CompatibleStateLossConfig(),
        )
        self.assertEqual(compatible.per_sample_total.shape, (2,))
        self.assertEqual(compatible.per_sample_position.shape, (2,))
        self.assertEqual(compatible.per_sample_projected_F.shape, (2,))

        baseline = reference.clone()
        baseline[:, 1:] *= 1.05
        context = CommonObjectiveContext(
            tets=data["tets"],
            J=data["J"],
            volume=data["volume"],
            mass=data["mass"],
            mu=torch.tensor([4.0], dtype=torch.float64),
            lam=torch.tensor([7.0], dtype=torch.float64),
            inertial_target=data["positions"],
            pinned=data["pinned"],
            dt=0.5,
        )
        potential = common_potential_excess_loss(
            context,
            predicted,
            reference,
            baseline,
            PotentialExcessLossConfig(),
        )
        self.assertEqual(potential.per_sample_total.shape, (2,))
        self.assertEqual(potential.per_sample_excess.shape, (2,))
        self.assertEqual(potential.denominator.shape, (2,))


if __name__ == "__main__":
    unittest.main()
