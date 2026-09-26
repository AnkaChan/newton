# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the revised nine-value feature schema constants and pure functions on the CPU."""

import importlib.util
import math
import unittest

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is an optional dependency")

import torch  # noqa: TID253


def _center_gradients(cell_size: float, dtype: torch.dtype) -> torch.Tensor:
    """Rest-cube center shape gradients in the z-fast corner order used by the solver steps."""
    signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=dtype)
    return signs / (4 * cell_size)


def _proper_rotations(*shape: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    left, _, right_transpose = torch.linalg.svd(torch.randn(*shape, 3, 3, dtype=dtype))
    rotations = left @ right_transpose
    flip = torch.linalg.det(rotations) < 0
    left[flip, :, -1] *= -1
    return left @ right_transpose


class TestSchemaConstants(unittest.TestCase):
    def test_dimensions_and_order(self):
        """Pin the schema-3 widths and packing order the network and trainer depend on."""
        self.assertEqual(
            features.MATRIX_BLOCKS,
            (
                "inertial_axis_offset",
                "physical_axis_change",
                "current_axis_gradient",
                "previous_axis_gradient",
                "previous_axis_update",
            ),
        )
        self.assertEqual(features.MATRIX_FEATURE_DIM, 45)
        self.assertEqual(features.BOUNDARY_DIM, 14)
        self.assertEqual(features.SCALAR_FEATURES, ("log_gradient_rms", "history_valid"))
        self.assertEqual(features.STATE_FEATURE_DIM, 61)
        self.assertEqual(features.CONDITIONING_DIM, 6)
        self.assertEqual(len(features.CONDITIONING_CHANNELS), 6)
        self.assertEqual(features.EDGE_FEATURE_DIM, 24)
        self.assertEqual(features.FEATURE_SCHEMA_VERSION, 3)
        self.assertEqual(features.RMS_FLOOR, 1e-12)
        self.assertEqual(features.CLIP, 10.0)


class TestPackStateFeatures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch, self.cells = 2, 3
        count = self.batch * self.cells * 9
        self.blocks = {
            name: 1000.0 * index + torch.arange(count, dtype=torch.float32).reshape(self.batch, self.cells, 3, 3)
            for index, name in enumerate(features.MATRIX_BLOCKS)
        }
        self.boundary = torch.rand(self.cells, features.BOUNDARY_DIM).round()
        self.log_rms = torch.tensor([-2.5, 1.25])

    def _pack(self, **overrides):
        arguments = dict(
            self.blocks,
            boundary_features=self.boundary,
            log_gradient_rms=self.log_rms,
            history_valid=torch.tensor([True, True]),
        )
        arguments.update(overrides)
        return features.pack_state_features(**arguments)

    def test_packing_order_and_width(self):
        """Place five row-major matrix blocks, boundary flags, log RMS, and the flag in order."""
        state = self._pack()
        self.assertEqual(state.shape, (self.batch, self.cells, 61))
        self.assertEqual(state.dtype, torch.float32)
        for index, name in enumerate(features.MATRIX_BLOCKS):
            block = self.blocks[name]
            torch.testing.assert_close(state[..., 9 * index : 9 * index + 9], block.flatten(-2))
            for i in range(3):
                for j in range(3):
                    torch.testing.assert_close(state[..., 9 * index + 3 * i + j], block[..., i, j])
        torch.testing.assert_close(state[..., 45:51], self.boundary[None, :, :6].expand(self.batch, -1, -1))
        torch.testing.assert_close(state[..., 51:59], self.boundary[None, :, 6:].expand(self.batch, -1, -1))
        torch.testing.assert_close(state[..., 59], self.log_rms[:, None].expand(-1, self.cells))
        torch.testing.assert_close(state[..., 60], torch.ones(self.batch, self.cells))

    def test_history_flag_zeroes_history_blocks(self):
        """Write 1.0/0.0 for the flag and zero both history blocks of objects without history."""
        state = self._pack(history_valid=torch.tensor([True, False]))
        torch.testing.assert_close(state[0, :, 60], torch.ones(self.cells))
        torch.testing.assert_close(state[1, :, 60], torch.zeros(self.cells))
        torch.testing.assert_close(state[0, :, 27:45], self._pack()[0, :, 27:45])
        self.assertTrue((state[1, :, 27:45] == 0).all())
        self.assertTrue((self.blocks["previous_axis_gradient"][1] != 0).all())
        torch.testing.assert_close(state[1, :, :27], self._pack()[1, :, :27])
        torch.testing.assert_close(state[1, :, 45:60], self._pack()[1, :, 45:60])

    def test_numeric_flag_and_broadcast_scalar_shapes(self):
        """Accept [B, 1, 1, 1] log RMS, numeric flags, and batched boundary flags."""
        reference = self._pack(history_valid=torch.tensor([True, False]))
        state = self._pack(
            history_valid=torch.tensor([1.0, 0.0]),
            log_gradient_rms=self.log_rms.reshape(self.batch, 1, 1, 1),
            boundary_features=self.boundary[None].expand(self.batch, -1, -1).clone(),
        )
        torch.testing.assert_close(state, reference)

    def test_rejects_malformed_inputs(self):
        """Refuse mismatched block shapes, boundary widths, and per-object scalar counts."""
        with self.assertRaises(ValueError):
            self._pack(boundary_features=self.boundary[:, :13])
        with self.assertRaises(ValueError):
            self._pack(previous_axis_update=self.blocks["previous_axis_update"][:, :2])
        with self.assertRaises(ValueError):
            self._pack(log_gradient_rms=torch.zeros(3))
        with self.assertRaises(ValueError):
            self._pack(history_valid=torch.tensor([True]))
        with self.assertRaises(TypeError):
            self._pack(physical_axis_change=self.blocks["physical_axis_change"].double())
        with self.assertRaises(TypeError):
            self._pack(boundary_features=self.boundary.bool())


class TestRmsNormalize(unittest.TestCase):
    def test_zero_input_uses_floor_and_stays_finite(self):
        """Floor a measured zero RMS at 1e-12 so the normalized field and its log stay finite."""
        normalized, rms = features.rms_normalize(torch.zeros(2, 3, 3, 3))
        self.assertEqual(rms.shape, (2, 1, 1, 1))
        torch.testing.assert_close(rms, torch.full((2, 1, 1, 1), 1e-12))
        torch.testing.assert_close(normalized, torch.zeros(2, 3, 3, 3))
        self.assertTrue(torch.isfinite(rms.log()).all())
        torch.testing.assert_close(rms.log(), torch.full((2, 1, 1, 1), math.log(1e-12)))

    def test_rms_value_and_log(self):
        """Compute the per-object RMS over all cell components and expose it for the log input."""
        values = torch.cat((torch.full((1, 4, 3, 3), 2.0), torch.full((1, 4, 3, 3), -3.0)))
        normalized, rms = features.rms_normalize(values)
        torch.testing.assert_close(rms.flatten(), torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(rms.log().flatten(), torch.tensor([math.log(2.0), math.log(3.0)]))
        torch.testing.assert_close(normalized[0], torch.ones(4, 3, 3))
        torch.testing.assert_close(normalized[1], -torch.ones(4, 3, 3))
        mixed = torch.zeros(1, 2, 3, 3)
        mixed[0, 0, 0, 0] = 3.0
        mixed[0, 1, 2, 1] = -3.0
        _, rms = features.rms_normalize(mixed)
        torch.testing.assert_close(rms.flatten(), torch.tensor([math.sqrt(18.0 / 18.0)]))

    def test_clips_to_plus_minus_ten(self):
        """Clip outliers to +/-10 after dividing by the own RMS or by a supplied RMS."""
        values = torch.zeros(1, 30, 3, 3)
        values[0, 0, 0, 0] = 5.0
        values[0, 29, 2, 2] = -5.0
        normalized, rms = features.rms_normalize(values)
        torch.testing.assert_close(rms.flatten(), torch.tensor([5.0 * math.sqrt(2.0 / 270.0)]))
        self.assertGreater(5.0 / rms.item(), 10.0)
        self.assertEqual(normalized.max().item(), 10.0)
        self.assertEqual(normalized.min().item(), -10.0)
        self.assertEqual((normalized != 0).sum().item(), 2)
        supplied, returned = features.rms_normalize(torch.tensor([[[[1e3, -1e3, 0.5]] * 3]]), rms=torch.ones(1))
        torch.testing.assert_close(returned, torch.ones(1, 1, 1, 1))
        torch.testing.assert_close(supplied, torch.tensor([[[[10.0, -10.0, 0.5]] * 3]]))

    def test_reuses_supplied_rms(self):
        """Normalize the previous gradient with the current RMS and floor a supplied zero RMS."""
        torch.manual_seed(3)
        current = torch.randn(2, 5, 3, 3)
        previous = 4.0 * torch.randn(2, 5, 3, 3)
        _, current_rms = features.rms_normalize(current)
        normalized, rms = features.rms_normalize(previous, rms=current_rms)
        torch.testing.assert_close(rms, current_rms)
        torch.testing.assert_close(normalized, (previous / current_rms).clamp(-10.0, 10.0))
        flat, flat_rms = features.rms_normalize(previous, rms=current_rms.flatten())
        torch.testing.assert_close(flat, normalized)
        torch.testing.assert_close(flat_rms, current_rms)
        _, floored = features.rms_normalize(previous, rms=torch.zeros(2))
        torch.testing.assert_close(floored, torch.full((2, 1, 1, 1), 1e-12))

    def test_rejects_bad_shapes(self):
        """Refuse non-[B, C, 3, 3] fields and RMS tensors with the wrong element count."""
        with self.assertRaises(ValueError):
            features.rms_normalize(torch.zeros(2, 3, 9))
        with self.assertRaises(ValueError):
            features.rms_normalize(torch.zeros(2, 3, 3, 3), rms=torch.ones(3))
        with self.assertRaises(TypeError):
            features.rms_normalize(torch.zeros(2, 3, 3, 3, dtype=torch.long))


class TestCenterDeformation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        self.corners = torch.as_tensor(self.rest.cell_corner_indices, dtype=torch.long)

    def _rest_positions(self, dtype):
        return torch.tensor(self.rest.corner_rest_positions, dtype=dtype)[None]

    def test_identity_on_rest_grid(self):
        """Return the identity for every cell of the undeformed grid in both precisions."""
        for dtype, tolerance in ((torch.float64, 1e-13), (torch.float32, 1e-6)):
            deformation = features.center_deformation(
                self._rest_positions(dtype), self.corners, _center_gradients(self.rest.cell_size, dtype)
            )
            self.assertEqual(deformation.shape, (1, 12, 3, 3))
            self.assertEqual(deformation.dtype, dtype)
            expected = torch.eye(3, dtype=dtype).expand(1, 12, 3, 3)
            torch.testing.assert_close(deformation, expected, atol=tolerance, rtol=0.0)

    def test_affine_map_recovers_matrix(self):
        """Recover the analytic F of x -> M x + t per object, including an inverted map."""
        matrices = torch.randn(2, 3, 3, dtype=torch.float64)
        matrices[1, :, 0] *= -1.0
        self.assertLess(torch.linalg.det(matrices[1]).item() * torch.linalg.det(matrices[0]).item(), 0.0)
        translation = torch.randn(2, 1, 3, dtype=torch.float64)
        positions = self._rest_positions(torch.float64) @ matrices.transpose(-1, -2) + translation
        deformation = features.center_deformation(
            positions, self.corners, _center_gradients(self.rest.cell_size, torch.float64)
        )
        torch.testing.assert_close(deformation, matrices[:, None].expand(-1, 12, -1, -1), atol=1e-12, rtol=0.0)
        single = features.center_deformation(
            positions.float(), self.corners, _center_gradients(self.rest.cell_size, torch.float32)
        )
        torch.testing.assert_close(single, matrices[:, None].expand(-1, 12, -1, -1).float(), atol=2e-5, rtol=0.0)

    def test_translation_invariant_and_differentiable(self):
        """Ignore rigid translations and propagate gradients whose corner sums vanish."""
        positions = self._rest_positions(torch.float64) + 0.01 * torch.randn(1, 36, 3, dtype=torch.float64)
        gradients = _center_gradients(self.rest.cell_size, torch.float64)
        base = features.center_deformation(positions, self.corners, gradients)
        shifted = features.center_deformation(positions + torch.tensor([1.0, -2.0, 0.5]), self.corners, gradients)
        torch.testing.assert_close(shifted, base, atol=1e-12, rtol=0.0)
        leaf = positions.clone().requires_grad_(True)
        weights = torch.randn(1, 12, 3, 3, dtype=torch.float64)
        (features.center_deformation(leaf, self.corners, gradients) * weights).sum().backward()
        self.assertTrue(torch.isfinite(leaf.grad).all())
        torch.testing.assert_close(leaf.grad.sum(1), torch.zeros(1, 3, dtype=torch.float64), atol=1e-12, rtol=0.0)

    def test_rejects_malformed_inputs(self):
        """Refuse wrong ranks, corner counts, index dtypes, and gradient shapes or dtypes."""
        positions = self._rest_positions(torch.float32)
        gradients = _center_gradients(self.rest.cell_size, torch.float32)
        with self.assertRaises(ValueError):
            features.center_deformation(positions[0], self.corners, gradients)
        with self.assertRaises(ValueError):
            features.center_deformation(positions, self.corners[:, :7], gradients)
        with self.assertRaises(TypeError):
            features.center_deformation(positions, self.corners.int(), gradients)
        with self.assertRaises(ValueError):
            features.center_deformation(positions, self.corners, gradients[:, :2])
        with self.assertRaises(TypeError):
            features.center_deformation(positions, self.corners, gradients.double())


class TestToLocal(unittest.TestCase):
    def test_is_frame_transpose_times_matrix(self):
        """Return R^T M so that R @ to_local(R, M) reconstructs M."""
        torch.manual_seed(5)
        frames = _proper_rotations(2, 4)
        matrices = torch.randn(2, 4, 3, 3, dtype=torch.float64)
        local = features.to_local(frames, matrices)
        self.assertEqual(local.shape, (2, 4, 3, 3))
        torch.testing.assert_close(local, frames.transpose(-1, -2) @ matrices)
        torch.testing.assert_close(frames @ local, matrices, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(
            features.to_local(frames, frames), torch.eye(3, dtype=torch.float64).expand(2, 4, 3, 3)
        )

    def test_rejects_shape_and_dtype_mismatch(self):
        """Refuse mismatched shapes, non-3x3 trailing dimensions, and dtype mixtures."""
        frames = _proper_rotations(2, 4)
        with self.assertRaises(ValueError):
            features.to_local(frames, torch.zeros(2, 3, 3, 3, dtype=torch.float64))
        with self.assertRaises(ValueError):
            features.to_local(
                torch.zeros(2, 4, 3, 2, dtype=torch.float64), torch.zeros(2, 4, 3, 2, dtype=torch.float64)
            )
        with self.assertRaises(TypeError):
            features.to_local(frames, torch.zeros(2, 4, 3, 3))


class TestConditioningChannels(unittest.TestCase):
    def test_channel_formulas(self):
        """Match the six documented channel formulas including log1p(eta / (mu dt))."""
        time_step = 1.0 / 60.0
        channels = features.conditioning_channels(
            torch.tensor([1e5, 2e5]),
            torch.tensor([1e5, 5e4]),
            torch.tensor([1000.0, 2000.0]),
            torch.tensor([0.0, 3.0]),
            0.025,
            time_step,
        )
        self.assertEqual(channels.shape, (2, 6))
        expected = torch.tensor(
            [
                [math.log1p(1.0), math.log1p(1.0), 0.0, 0.0, 0.0, 0.0],
                [math.log1p(2.0), math.log1p(0.5), math.log(2.0), 0.0, 0.0, math.log1p(3.0 / (5e4 * time_step))],
            ]
        )
        torch.testing.assert_close(channels, expected)
        scaled = features.conditioning_channels(
            torch.tensor([1e5]), torch.tensor([1e5]), torch.tensor([1000.0]), torch.tensor([0.0]), 0.05, 1.0 / 300.0
        )
        torch.testing.assert_close(scaled[0, 3:5], torch.tensor([math.log(2.0), math.log(60.0 / 300.0)]))

    def test_rejects_invalid_inputs(self):
        """Refuse nonpositive or nonfinite scalars and mismatched material tensors."""
        lam, mu, rho, eta = (torch.tensor([1e5]) for _ in range(4))
        with self.assertRaises(ValueError):
            features.conditioning_channels(lam, mu, rho, eta, 0.0, 0.01)
        with self.assertRaises(ValueError):
            features.conditioning_channels(lam, mu, rho, eta, 0.025, math.nan)
        with self.assertRaises(ValueError):
            features.conditioning_channels(lam, mu, rho, eta, 0.025, True)
        with self.assertRaises(ValueError):
            features.conditioning_channels(torch.tensor([1e5, 1e5]), mu, rho, eta, 0.025, 0.01)
        with self.assertRaises(ValueError):
            features.conditioning_channels(lam[None], mu[None], rho[None], eta[None], 0.025, 0.01)
        with self.assertRaises(TypeError):
            features.conditioning_channels(lam.double(), mu, rho, eta, 0.025, 0.01)


if __name__ == "__main__":
    unittest.main()
