# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration checks for the experimental learned hexahedral solver step (revised schema)."""

import copy
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features, input_assembly
from experiments.learned_intrinsic_solver.frames import (
    closest_proper_rotations,
    reference_rotation,
    select_reference_corners,
)
from experiments.learned_intrinsic_solver.input_assembly import OptimizerHistory
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep

MATERIAL = {"lame_lambda": 1000.0 * 0.3 / (1.3 * 0.4), "lame_mu": 1000.0 / 2.6, "density": 100.0}
GRADIENT_SLICE = slice(18, 27)
PREVIOUS_GRADIENT_SLICE = slice(27, 36)
PREVIOUS_UPDATE_SLICE = slice(36, 45)


def revised_network(cell_counts, *, dtype=torch.float32, nonzero_head=False, **kwargs):
    """Build a small revised-schema network, optionally with nonzero heads."""
    model = IntrinsicSolverNetwork(
        cell_counts,
        features.STATE_FEATURE_DIM,
        conditioning_dim=features.CONDITIONING_DIM,
        hidden_dim=16,
        edge_hidden_dim=8,
        **kwargs,
    )
    model.to(dtype=dtype)
    if nonzero_head:
        with torch.no_grad():
            model.correction_head.weight.normal_(std=0.004)
            model.step_head.weight.normal_(std=0.01)
            for layer in model.layers:
                layer.film.weight.normal_(std=0.01)
    return model


class TestLearnedHexSolverStep(unittest.TestCase):
    def setUp(self):
        """Build a small clamped cuboid with a repeatable non-affine initial shape."""
        torch.manual_seed(31)
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)

    def _fixture(self, *, dtype=torch.float32, nonzero_head=False, damping=0.0):
        model = revised_network(self.rest.cell_counts, dtype=dtype, nonzero_head=nonzero_head)
        step = LearnedHexSolverStep(
            self.rest, self.fixed, **MATERIAL, time_step=0.04, damping=damping, network=model, dtype=dtype
        )
        previous = torch.tensor(self.rest.corner_rest_positions, dtype=dtype).unsqueeze(0)
        x = previous.clone()
        z = x[..., 2]
        x[..., 0] += 0.08 * z.square() + 0.001 * torch.sin(21 * x[..., 1]) * z / 0.3
        x[..., 1] += 0.02 * z.square()
        y = x + torch.tensor([0.0003, -0.001, 0.0001], dtype=dtype)
        return step, x, y, previous

    def _inverted(self, *, dtype=torch.float32):
        """Fold the top corner layer below the layer beneath it so the four top cells have det F < 0."""
        step, x, y, previous = self._fixture(dtype=dtype, nonzero_head=True)
        z = self.rest.corner_rest_positions[:, 2]
        top = torch.from_numpy(z == z.max())
        x, y = x.clone(), y.clone()
        x[:, top, 2] = 0.15
        y[:, top, 2] = 0.15
        return step, x, y, previous

    @staticmethod
    def _deformation(step, positions):
        return features.center_deformation(positions.detach(), step.cell_corner_indices, step.center_gradients)

    def _assert_proper_reconstruction(self, frames, local_axes, deformation, atol):
        eye = torch.eye(3, dtype=frames.dtype).expand_as(frames)
        torch.testing.assert_close(frames.transpose(-1, -2) @ frames, eye, rtol=0, atol=atol)
        self.assertTrue((torch.linalg.det(frames) > 0).all())
        self.assertFalse(frames.requires_grad)
        torch.testing.assert_close(frames @ local_axes.detach(), deformation, rtol=0, atol=atol)

    def test_default_network_uses_revised_schema_and_legacy_networks_are_rejected(self):
        """Construct the 61/6/24 default and reject 38/5 and 86/6 networks or checkpoints explicitly."""
        step = LearnedHexSolverStep(self.rest, self.fixed, **MATERIAL, time_step=0.04)
        self.assertEqual(
            (step.network.state_feature_dim, step.network.conditioning_dim, step.network.edge_input_dim),
            (features.STATE_FEATURE_DIM, features.CONDITIONING_DIM, features.EDGE_FEATURE_DIM),
        )
        self.assertEqual(step.conditioning.shape, (12, features.CONDITIONING_DIM))
        for state, conditioning in ((38, 5), (86, 6), (61, 5), (60, 6)):
            legacy = IntrinsicSolverNetwork(
                self.rest.cell_counts, state, conditioning_dim=conditioning, hidden_dim=16, edge_hidden_dim=8
            )
            with self.subTest(schema=(state, conditioning)), self.assertRaisesRegex(ValueError, "schema"):
                LearnedHexSolverStep(self.rest, self.fixed, **MATERIAL, time_step=0.04, network=legacy)
        legacy_state = IntrinsicSolverNetwork(self.rest.cell_counts, 38, conditioning_dim=5).state_dict()
        with self.assertRaises(RuntimeError):
            step.network.load_state_dict(legacy_state, strict=True)
        with self.assertRaisesRegex(ValueError, "edge"):
            LearnedHexSolverStep(
                self.rest,
                self.fixed,
                **MATERIAL,
                time_step=0.04,
                network=IntrinsicSolverNetwork(
                    self.rest.cell_counts, features.STATE_FEATURE_DIM, edge_input_dim=20, hidden_dim=16
                ),
            )

    def test_per_cell_lame_conditioning_and_fusion_stiffness(self):
        """Keep zero lambda finite, expose all six channels including zero viscosity, and keep fusion weights."""
        rest = generate_cuboid((2, 1, 1), cell_size=0.025)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        step = LearnedHexSolverStep(
            rest,
            fixed,
            lame_lambda=[0, 1e5],
            lame_mu=[1e5, 3e5],
            density=[1000, 2000],
            time_step=1 / 60,
        )
        x = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        inputs = step.prepare_inputs(x, x, previous_positions=x)
        expected = torch.tensor(
            [[[0, np.log(2), 0, 0, 0, 0], [np.log(2), np.log(4), np.log(2), 0, 0, 0]]], dtype=torch.float32
        )
        torch.testing.assert_close(inputs.conditioning, expected)
        torch.testing.assert_close(step.fusion_stiffness, torch.tensor([2e5, 6.75e5]))
        damped = LearnedHexSolverStep(
            rest, fixed, lame_lambda=[0, 1e5], lame_mu=[1e5, 3e5], density=[1000, 2000], time_step=1 / 60, damping=2e3
        )
        torch.testing.assert_close(
            damped.conditioning[:, 5], torch.tensor([np.log1p(2e3 * 60 / 1e5), np.log1p(2e3 * 60 / 3e5)]).float()
        )
        self.assertEqual(inputs.state_features.shape, (1, 2, features.STATE_FEATURE_DIM))

    def test_previous_positions_and_history_are_validated(self):
        """Require the physical-step start and reject malformed history explicitly."""
        step, x, y, previous = self._fixture()
        with self.assertRaises(TypeError):
            step(x, y)
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            step(x, y, previous_positions=None)
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            step.prepare_inputs(x, y, previous_positions=previous.repeat(2, 1, 1))
        with self.assertRaisesRegex(ValueError, "finite"):
            step(x, y, previous_positions=previous * float("nan"))
        cells = len(step.cell_corner_indices)
        good = OptimizerHistory(torch.zeros(1, cells, 3, 3), torch.zeros(1, cells, 3, 3), torch.tensor([False]))
        step.prepare_inputs(x, y, previous_positions=previous, history=good)
        with self.assertRaises(TypeError):
            step.prepare_inputs(x, y, previous_positions=previous, history=tuple(good))
        for malformed in (
            OptimizerHistory(torch.zeros(2, cells, 3, 3), good.axis_update_world, good.valid),
            OptimizerHistory(good.axis_gradient_world.double(), good.axis_update_world, good.valid),
            OptimizerHistory(good.axis_gradient_world, torch.full((1, cells, 3, 3), float("nan")), good.valid),
            OptimizerHistory(good.axis_gradient_world, good.axis_update_world, torch.tensor([1.0])),
        ):
            with self.subTest(history=malformed), self.assertRaises(ValueError):
                step(x, y, previous_positions=previous, history=malformed)

    def test_zero_update_preserves_shape_and_trains_head(self):
        """Preserve warped corners exactly while allowing the physical loss to train the output head."""
        step, x, y, previous = self._fixture()
        result = step(x, y, previous_positions=previous)
        torch.testing.assert_close(result.positions, x, rtol=0, atol=0)
        torch.testing.assert_close(result.positions[:, self.fixed], x[:, self.fixed], rtol=0, atol=0)
        self.assertEqual(result.positions.dtype, torch.float32)
        self.assertEqual(result.step_size.shape, (1, 12))
        torch.testing.assert_close(result.achieved_axis_update_world, torch.zeros(1, 12, 3, 3), rtol=0, atol=0)
        result.loss.total.mean().backward()
        gradient = step.network.correction_head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)

    def test_frames_frozen_and_axis_blocks_in_receiver_frame(self):
        """Detach only the rotations and express the inertial offset and physical change as R^T dF."""
        step, _, _, previous = self._fixture(dtype=torch.float64)
        rotation = torch.tensor([[0.0, -1, 0], [1.0, 0, 0], [0.0, 0, 1]], dtype=previous.dtype)
        x = (previous[0] @ rotation.T)[None].requires_grad_()
        stretch = torch.tensor([[0.01, 0.002, 0.0], [0.0, -0.005, 0.003], [0.004, 0.0, 0.02]], dtype=x.dtype)
        y = (x.detach() @ (torch.eye(3, dtype=x.dtype) + stretch.T)).detach()
        inputs = step.prepare_inputs(x, y, previous_positions=previous)
        self.assertFalse(inputs.frames.requires_grad)
        self.assertTrue(inputs.local_axes.requires_grad)
        torch.testing.assert_close(inputs.frames, rotation.expand(1, 12, 3, 3))
        torch.testing.assert_close(inputs.local_axes, torch.eye(3, dtype=x.dtype).expand(1, 12, 3, 3))
        expected_offset = (rotation.T @ stretch @ rotation).expand(1, 12, 3, 3)
        torch.testing.assert_close(inputs.state_features[..., :9].reshape(1, 12, 3, 3), expected_offset)
        expected_change = (torch.eye(3, dtype=x.dtype) - rotation.T).expand(1, 12, 3, 3)
        torch.testing.assert_close(inputs.state_features[..., 9:18].reshape(1, 12, 3, 3), expected_change)
        torch.testing.assert_close(inputs.state_features[..., 45:59], step.boundary_features[None])
        self.assertEqual(inputs.state_features[..., 60].unique().tolist(), [0.0])
        torch.testing.assert_close(inputs.state_features[..., 27:45], torch.zeros(1, 12, 18, dtype=x.dtype))
        self.assertFalse(inputs.axis_gradient_world.requires_grad)
        self.assertFalse(inputs.position_gradient.requires_grad)
        self.assertEqual(inputs.position_gradient[:, self.fixed].abs().max().item(), 0)
        inputs.local_axes.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)

    def test_gradient_feature_is_the_projected_objective_gradient(self):
        """Match the world axis gradient with the fusion adjoint of the zero-pinned energy gradient."""
        step, x, y, previous = self._fixture(dtype=torch.float64, damping=3.0)
        candidate = x.clone().requires_grad_()
        energy = step.energy(candidate, y, previous_positions=previous).total.sum()
        gradient = torch.autograd.grad(energy, candidate)[0]
        gradient[:, step.fixed_indices] = 0
        expected = step.fusion.project_gradient(gradient)
        inputs = step.prepare_inputs(x, y, previous_positions=previous)
        torch.testing.assert_close(inputs.axis_gradient_world, expected, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(inputs.position_gradient, gradient, rtol=1e-12, atol=1e-14)
        local = inputs.frames.transpose(-1, -2) @ expected
        rms = local.square().mean().sqrt()
        torch.testing.assert_close(
            inputs.state_features[..., GRADIENT_SLICE].reshape(1, 12, 3, 3), (local / rms).clamp(-10, 10)
        )
        torch.testing.assert_close(inputs.state_features[..., 59], rms.log().expand(1, 12))
        result = step(x, y, previous_positions=previous)
        torch.testing.assert_close(result.axis_gradient_world, expected, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(result.force_residual_norm, gradient.flatten(1).norm(dim=1))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(step(x, y, previous_positions=previous).loss.total).all())

    def test_history_blocks_follow_the_previous_query(self):
        """Fill the history blocks and flag from the previous query and zero them for invalid objects."""
        step, x, y, previous = self._fixture(nonzero_head=True)
        first = step(x, y, previous_positions=previous)
        history = OptimizerHistory(first.axis_gradient_world, first.achieved_axis_update_world, torch.tensor([True]))
        second = step.prepare_inputs(first.positions.detach(), y, previous_positions=previous, history=history)
        self.assertEqual(second.state_features[..., 60].unique().tolist(), [1.0])
        frames = second.frames
        _, rms = features.rms_normalize(features.to_local(frames, second.axis_gradient_world))
        expected_gradient, _ = features.rms_normalize(features.to_local(frames, history.axis_gradient_world), rms=rms)
        expected_update, _ = features.rms_normalize(features.to_local(frames, history.axis_update_world))
        torch.testing.assert_close(
            second.state_features[..., PREVIOUS_GRADIENT_SLICE].reshape(1, 12, 3, 3), expected_gradient
        )
        torch.testing.assert_close(
            second.state_features[..., PREVIOUS_UPDATE_SLICE].reshape(1, 12, 3, 3), expected_update
        )
        self.assertGreater(second.state_features[..., PREVIOUS_UPDATE_SLICE].abs().max().item(), 0)
        invalid = OptimizerHistory(history.axis_gradient_world, history.axis_update_world, torch.tensor([False]))
        cleared = step.prepare_inputs(first.positions.detach(), y, previous_positions=previous, history=invalid)
        none = step.prepare_inputs(first.positions.detach(), y, previous_positions=previous)
        torch.testing.assert_close(cleared.state_features, none.state_features, rtol=0, atol=0)
        self.assertEqual(none.state_features[..., 60].unique().tolist(), [0.0])
        torch.testing.assert_close(none.state_features[..., 27:45], torch.zeros(1, 12, 18), rtol=0, atol=0)
        with_history = step(first.positions.detach(), y, previous_positions=previous, history=history)
        without = step(first.positions.detach(), y, previous_positions=previous)
        self.assertGreater((with_history.positions - without.positions).abs().max().item(), 0)

    def test_inputs_match_mixed_step_for_the_same_single_object(self):
        """Produce the same frames, state, conditioning, gradient feature and update as MixedHexSolverStep."""
        network = revised_network(self.rest.cell_counts, nonzero_head=True)
        step = LearnedHexSolverStep(self.rest, self.fixed, **MATERIAL, damping=5.0, time_step=0.04, network=network)
        mixed = MixedHexSolverStep(self.rest, self.fixed, network=copy.deepcopy(network), time_step=0.04)
        self.addCleanup(mixed.close)
        mixed.register_context("one", **MATERIAL, damping=5.0)
        previous = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)[None]
        x = previous.clone()
        x[..., 0] += 0.08 * x[..., 2].square()
        x[..., 1] -= 0.03 * x[..., 2]
        y = x + torch.tensor([0.0003, -0.001, 0.0001])
        generator = torch.Generator().manual_seed(9)
        history = OptimizerHistory(
            torch.randn((1, 12, 3, 3), generator=generator) * 0.05,
            torch.randn((1, 12, 3, 3), generator=generator) * 1e-3,
            torch.tensor([True]),
        )
        for carried in (None, history):
            with self.subTest(history=carried is not None):
                single = step.prepare_inputs(x, y, previous_positions=previous, history=carried)
                shared = mixed.prepare_inputs(x, y, ("one",), previous_positions=previous, history=carried)
                torch.testing.assert_close(single.frames, shared.frames, rtol=0, atol=0)
                torch.testing.assert_close(single.local_axes, shared.local_axes, rtol=0, atol=0)
                torch.testing.assert_close(single.conditioning, shared.conditioning, rtol=0, atol=0)
                torch.testing.assert_close(single.state_features, shared.state_features, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(single.axis_gradient_world, shared.axis_gradient_world, rtol=1e-5, atol=1e-9)
                torch.testing.assert_close(single.position_gradient, shared.position_gradient, rtol=1e-5, atol=1e-8)
                self.assertEqual(single.tie_mask.tolist(), shared.tie_mask.tolist())
                for hop in single.edge_features:
                    torch.testing.assert_close(single.edge_features[hop], shared.edge_features[hop], rtol=0, atol=0)
                pins = x[:, self.fixed]
                one = step(x, y, previous_positions=previous, fixed_positions=pins, history=carried)
                many = mixed(x, y, ("one",), previous_positions=previous, fixed_positions=pins, history=carried)
                torch.testing.assert_close(one.positions, many.positions, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(one.step_size, many.step_size, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(one.loss.total, many.loss.total, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(one.force_residual_norm, many.force_residual_norm, rtol=1e-5, atol=1e-8)
                torch.testing.assert_close(
                    one.achieved_axis_update_world, many.achieved_axis_update_world, rtol=1e-4, atol=1e-7
                )

    def test_inverted_candidate_gets_proper_frames_and_finite_energy(self):
        """Accept a folded layer: right-handed frames reconstruct the inverted F and everything stays finite."""
        for dtype, atol in ((torch.float32, 2e-5), (torch.float64, 1e-12)):
            step, x, y, previous = self._inverted(dtype=dtype)
            x.requires_grad_()
            deformation = self._deformation(step, x)
            self.assertLess(torch.linalg.det(deformation).min().item(), 0)
            inputs = step.prepare_inputs(x, y, previous_positions=previous)
            self._assert_proper_reconstruction(inputs.frames, inputs.local_axes, deformation, atol)
            self.assertLess(torch.linalg.det(inputs.local_axes).min().item(), 0)
            self.assertTrue(inputs.local_axes.requires_grad)
            self.assertEqual(inputs.tie_mask.shape, (1, 12))
            self.assertEqual(inputs.tie_mask.dtype, torch.bool)
            self.assertFalse(inputs.tie_mask.any())
            self.assertTrue(torch.isfinite(inputs.state_features).all())
            result = step(x, y, previous_positions=previous)
            self.assertTrue(torch.isfinite(result.loss.total).all())
            self.assertTrue(torch.isfinite(result.positions).all())
            self.assertEqual(result.step_size.shape, (1, 12))
            self.assertTrue(torch.equal(result.tie_mask, inputs.tie_mask))
            for name in ("axis_gradient_world", "achieved_axis_update_world", "force_residual_norm"):
                value = getattr(result, name)
                self.assertFalse(value.requires_grad, name)
                self.assertTrue(torch.isfinite(value).all(), name)
            torch.testing.assert_close(
                result.achieved_axis_update_world, self._deformation(step, result.positions) - deformation
            )
            gradients = torch.autograd.grad(result.loss.total.sum(), (x, step.network.correction_head.weight))
            for gradient in gradients:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.norm().item(), 0)

    def test_mirrored_cell_uses_clamped_face_reference_and_rotates_with_the_problem(self):
        """Break the reflection tie with the pinned-face reference and keep whole-problem rotation equivariance."""
        dtype = torch.float64
        rest = generate_cuboid((1, 1, 2), cell_size=0.1)
        z = rest.corner_rest_positions[:, 2]
        fixed = np.flatnonzero(z == 0)
        material = {"lame_lambda": 700.0, "lame_mu": 300.0, "density": 100.0, "time_step": 0.02, "dtype": dtype}
        step = LearnedHexSolverStep(rest, fixed, **material, network=revised_network(rest.cell_counts, dtype=dtype))
        np.testing.assert_array_equal(
            step.reference_corners.numpy(), select_reference_corners(rest.corner_rest_positions, fixed)
        )
        self.assertNotIn("reference_corners", step.state_dict())
        previous = torch.tensor(rest.corner_rest_positions, dtype=dtype)[None]
        x = previous.clone()
        x[:, torch.from_numpy(z == z.max()), 2] = 0.0
        deformation = self._deformation(step, x)
        torch.testing.assert_close(deformation[0, 1], torch.diag(x.new_tensor([1.0, 1.0, -1.0])))
        inputs = step.prepare_inputs(x, x, previous_positions=previous)
        self.assertEqual(inputs.tie_mask.tolist(), [[False, True]])
        self._assert_proper_reconstruction(inputs.frames, inputs.local_axes, deformation, 1e-12)
        reference = reference_rotation(x, step.reference_corners)
        torch.testing.assert_close(
            inputs.frames, closest_proper_rotations(deformation, reference).frames, rtol=0, atol=0
        )
        axis = x.new_tensor([1.0, 2.0, 3.0])
        axis /= axis.norm()
        skew = torch.stack(
            (
                torch.stack((axis[0] * 0, -axis[2], axis[1])),
                torch.stack((axis[2], axis[0] * 0, -axis[0])),
                torch.stack((-axis[1], axis[0], axis[0] * 0)),
            )
        )
        rotation = torch.linalg.matrix_exp(0.7 * skew)
        rotated = step.prepare_inputs(x @ rotation.T, x @ rotation.T, previous_positions=previous @ rotation.T)
        torch.testing.assert_close(rotated.frames, rotation @ inputs.frames, rtol=0, atol=1e-8)
        torch.testing.assert_close(rotated.local_axes, inputs.local_axes, rtol=0, atol=1e-8)
        torch.testing.assert_close(rotated.state_features, inputs.state_features, rtol=1e-7, atol=1e-7)
        self.assertTrue(torch.isfinite(step(x, x, previous_positions=previous).loss.total).all())
        two_pins = LearnedHexSolverStep(
            rest, fixed[:2], **material, network=revised_network(rest.cell_counts, dtype=dtype)
        )
        self.assertEqual(two_pins.reference_corners.numel(), 0)
        plain = two_pins.prepare_inputs(x, x, previous_positions=previous)
        self.assertEqual(plain.tie_mask.tolist(), [[False, True]])
        self._assert_proper_reconstruction(plain.frames, plain.local_axes, deformation, 1e-12)

    def test_supplied_frames_are_validated_and_frozen(self):
        """Replay supplied rotations detached and reject malformed, non-orthonormal, or improper frames."""
        step, x, y, previous = self._fixture(dtype=torch.float64)
        frames = step.prepare_inputs(x, y, previous_positions=previous).frames
        inputs = step.prepare_inputs(x, y, previous_positions=previous, frames=frames.clone().requires_grad_())
        torch.testing.assert_close(inputs.frames, frames, rtol=0, atol=0)
        self.assertFalse(inputs.frames.requires_grad)
        self.assertIsNone(inputs.tie_mask)
        self.assertIsNone(step(x, y, previous_positions=previous, frames=frames).tie_mask)
        with self.assertRaisesRegex(ValueError, "shape, dtype, and device"):
            step.prepare_inputs(x, y, previous_positions=previous, frames=frames[:, :-1])
        with self.assertRaisesRegex(ValueError, "shape, dtype, and device"):
            step.prepare_inputs(x, y, previous_positions=previous, frames=frames.to(torch.float32))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            step.prepare_inputs(x, y, previous_positions=previous, frames=frames * 1.01)
        improper = frames.clone()
        improper[..., 2] *= -1
        with self.assertRaisesRegex(ValueError, "positive determinant"):
            step(x, y, previous_positions=previous, frames=improper)

    def test_rigid_target_changes_only_fusion(self):
        """Keep the original physical inertia when an external rigid rotation guides fusion."""
        step, x, y, previous = self._fixture(dtype=torch.float64)
        angle = torch.tensor(0.04, dtype=x.dtype)
        c, s = angle.cos(), angle.sin()
        rotation = torch.stack(
            (torch.stack((c, -s, c * 0)), torch.stack((s, c, c * 0)), torch.tensor([0.0, 0, 1], dtype=x.dtype))
        )[None]
        saved_y = y.clone()
        result = step(
            x,
            y,
            previous_positions=previous,
            rigid_delta_rotation=rotation,
            rigid_delta_translation=torch.tensor([[0.02, -0.01, 0.03]], dtype=x.dtype),
        )
        expected = (
            0.5 * (step.energy.lumped_mass[None, :, None] * (result.positions - y).square()).sum((1, 2)) / 0.04**2
        )
        torch.testing.assert_close(result.loss.inertia, expected)
        torch.testing.assert_close(y, saved_y, rtol=0, atol=0)
        torch.testing.assert_close(result.positions[:, self.fixed], x[:, self.fixed], rtol=0, atol=0)
        self.assertGreater((result.positions - x).norm().item(), 1e-4)
        no_translation = step(x, y, previous_positions=previous, rigid_delta_rotation=rotation)
        torch.testing.assert_close(result.positions, no_translation.positions, rtol=1e-11, atol=1e-13)

    def test_loss_backpropagates_through_fusion_and_attention(self):
        """Deliver nonzero physical gradients through fusion into both edge projections and FiLM."""
        step, x, y, previous = self._fixture(nonzero_head=True)
        output = step(x, y, previous_positions=previous)
        output.local_target_axes.retain_grad()
        output.loss.total.sum().backward()
        self.assertGreater(output.local_target_axes.grad.norm().item(), 0)
        for name in (
            "node_encoder.0.weight",
            "edge_encoder.0.weight",
            "layers.0.edge_bias.weight",
            "layers.0.edge_val.weight",
            "layers.0.film.weight",
            "condition_encoder.0.weight",
            "step_head.weight",
        ):
            gradient = dict(step.network.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0, name)

    def test_network_parameter_gradient_matches_finite_difference(self):
        """Compare the entire energy-through-fusion derivative with central differences in network weights."""
        step, x, y, previous = self._fixture(dtype=torch.float64, nonzero_head=True)
        parameter = step.network.layers[0].edge_val.weight
        loss = step(x, y, previous_positions=previous).loss.total.sum()
        gradient = torch.autograd.grad(loss, parameter)[0]
        direction = torch.randn_like(parameter)
        direction /= direction.norm()
        analytical = (gradient * direction).sum().item()
        original = parameter.detach().clone()
        epsilon = 1e-5
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                parameter.copy_(original + sign * epsilon * direction)
                values.append(step(x, y, previous_positions=previous).loss.total.sum().item())
            parameter.copy_(original)
        numerical = (values[0] - values[1]) / (2 * epsilon)
        self.assertAlmostEqual(analytical, numerical, delta=max(1e-10, abs(analytical) * 1e-5))

    def test_position_gradient_with_frozen_frames_and_gradient_feature(self):
        """Differentiate features and fused base positions while excluding the frozen frames and gradient feature."""
        step, x, y, previous = self._fixture(dtype=torch.float64, nonzero_head=True)
        inputs = step.prepare_inputs(x, y, previous_positions=previous)
        frames = inputs.frames
        x.requires_grad_()
        gradient = torch.autograd.grad(step(x, y, previous_positions=previous, frames=frames).loss.total.sum(), x)[0]
        direction = torch.randn_like(x)
        direction /= direction.norm()
        analytical = (gradient * direction).sum().item()
        epsilon = 1e-6
        # The gradient feature is detached by design; hold it fixed in the finite difference as well.
        with (
            torch.no_grad(),
            patch.object(input_assembly, "objective_gradient", return_value=inputs.position_gradient),
        ):
            plus = step(x + epsilon * direction, y, previous_positions=previous, frames=frames).loss.total.sum()
            minus = step(x - epsilon * direction, y, previous_positions=previous, frames=frames).loss.total.sum()
        numerical = ((plus - minus) / (2 * epsilon)).item()
        self.assertAlmostEqual(analytical, numerical, delta=max(1e-9, abs(analytical) * 1e-6))


if __name__ == "__main__":
    unittest.main()
