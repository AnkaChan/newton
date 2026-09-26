# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the revised mixed-material step: schema, frames, gradient feature, normalization, physics."""

import copy
import importlib.util
import io
import json
import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.frames import (
    closest_proper_rotations,
    reference_rotation,
    select_reference_corners,
)
from experiments.learned_intrinsic_solver.fusion import HexFusion
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep, OptimizerHistory
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.network_geometry import build_edge_features
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.solver_step import LearnedHexStepOutput

MATERIALS = {
    "soft": {"lame_lambda": 700 * 0.2 / (1.2 * 0.6), "lame_mu": 700 / 2.4, "density": 90.0},
    "stiff": {"lame_lambda": 3000 * 0.4 / (1.4 * 0.2), "lame_mu": 3000 / 2.8, "density": 180.0},
}


def make_network(cell_counts, **kwargs):
    """Build a small revised-schema network with nonzero heads so outputs vary per cell."""
    network = IntrinsicSolverNetwork(
        cell_counts,
        features.STATE_FEATURE_DIM,
        conditioning_dim=features.CONDITIONING_DIM,
        hidden_dim=16,
        edge_hidden_dim=8,
        **kwargs,
    )
    with torch.no_grad():
        network.correction_head.weight.normal_(std=0.003)
        network.step_head.weight.normal_(std=0.01)
        for layer in network.layers:
            layer.film.weight.normal_(std=0.01)
    return network


def fusion_weights(physical, cell_size):
    """Return the fusion cell weights the mixed step derives from a material."""
    lam, mu = physical.lame_lambda, physical.lame_mu
    scale = torch.maximum(lam, mu)
    return mu * (3 - (mu / scale) / (lam / scale + mu / scale)) * cell_size**3


class TestMixedHexSolverStep(unittest.TestCase):
    def setUp(self):
        """Construct distinct materials and a non-affine clamped cuboid."""
        torch.manual_seed(302)
        self.rest = generate_cuboid((2, 1, 2), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.free = np.setdiff1d(np.arange(len(self.rest.corner_rest_positions)), self.fixed)
        self.cell_count = len(self.rest.cell_corner_indices)
        self.dt = 0.01
        self.gravity = (0.0, -9.81, 0.0)
        self.specs = {name: dict(spec) for name, spec in MATERIALS.items()}
        self.network = make_network(self.rest.cell_counts)
        self.step = MixedHexSolverStep(
            self.rest, self.fixed, network=self.network, time_step=self.dt, gravity=self.gravity
        )
        self.addCleanup(self.step.close)
        for name, spec in self.specs.items():
            self.step.register_context(name, **spec)
        self.rest_positions = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)
        self.x = self.rest_positions.clone()
        self.x[:, 0] += 0.06 * self.x[:, 2].square()
        self.x[:, 1] += 0.002 * torch.sin(17 * self.x[:, 0]) * self.x[:, 2]
        self.velocity = torch.zeros_like(self.x)
        self.velocity[:, 0] = 0.3 * self.x[:, 2]
        self.velocity[:, 1] = 0.05 * self.x[:, 2]
        self.forces = torch.zeros_like(self.x)
        self.forces[:, 2] = 0.0007

    def _batch(self, ids):
        """Return distinct candidate, target, and physical-start batches for the given contexts."""
        positions = self.x[None].repeat(len(ids), 1, 1)
        for index in range(1, len(ids)):
            positions[index, :, index % 3] += 0.02 * index * positions[index, :, 2].square()
        target = positions + torch.tensor([0.0003, -0.001, 0.0001])
        previous = self.rest_positions[None].repeat(len(ids), 1, 1)
        previous[:, :, 1] -= 0.01 * previous[:, :, 2].square()
        return positions, target, previous

    def _center(self, positions):
        return features.center_deformation(positions, self.step.cell_corner_indices, self.step.center_gradients)

    def _native(self, context_id, positions, velocities, forces):
        model = build_newton_hex_model(self.rest, self.fixed, gravity=self.gravity, **self.specs[context_id])
        state = model.state()
        state.particle_q.assign(positions.detach().numpy())
        state.particle_qd.assign(velocities.detach().numpy())
        state.particle_f.assign(forces.detach().numpy())
        # The native rigid initializer never reads weights; the single-material solver shares the revised schema.
        reference = IntrinsicSolverNetwork(
            self.rest.cell_counts, features.STATE_FEATURE_DIM, hidden_dim=16, edge_hidden_dim=8
        )
        solver = SolverLearnedIntrinsic(model, network=reference)
        problem = solver.prepare_problem(state, self.dt)
        return solver, problem

    def _single_object(self, spec, network, positions, target, previous, pins, history=None):
        """Compose the schema-3 query for one object without the mixed step."""
        rest = self.rest
        physical = HexImplicitEulerLoss(
            rest, spec["lame_lambda"], spec["lame_mu"], spec["density"], self.dt, damping=spec.get("damping", 0.0)
        )
        fusion = HexFusion(rest, self.fixed, cell_weights=fusion_weights(physical, rest.cell_size))
        cells = physical.cell_corner_indices
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
        gradients = signs / (4 * rest.cell_size)
        deformation = features.center_deformation(positions, cells, gradients)
        corners = select_reference_corners(rest.corner_rest_positions, self.fixed)
        frame_result = closest_proper_rotations(deformation, reference_rotation(positions, corners))
        frames = frame_result.frames
        axes = features.to_local(frames, deformation)
        with torch.enable_grad():
            candidate = positions.detach().requires_grad_(True)
            energy = physical(candidate, target.detach(), previous_positions=previous.detach()).total.sum()
            gradient = torch.autograd.grad(energy, candidate)[0]
        gradient[:, self.fixed] = 0
        world_gradient = fusion.project_gradient(gradient)
        current, rms = features.rms_normalize(features.to_local(frames, world_gradient))
        if history is None:
            previous_gradient = torch.zeros_like(current)
            previous_update = torch.zeros_like(current)
            valid = torch.zeros(1, dtype=torch.bool)
        else:
            previous_gradient, _ = features.rms_normalize(
                features.to_local(frames, history.axis_gradient_world), rms=rms
            )
            previous_update, _ = features.rms_normalize(features.to_local(frames, history.axis_update_world))
            valid = history.valid
        flags = torch.zeros(len(rest.corner_rest_positions))
        flags[self.fixed] = 1
        boundary = torch.cat((torch.tensor(rest.cell_exposed_faces, dtype=torch.float32), flags[cells]), -1)
        state = features.pack_state_features(
            inertial_axis_offset=features.to_local(
                frames, features.center_deformation(target, cells, gradients) - deformation
            ),
            physical_axis_change=features.to_local(
                frames, deformation - features.center_deformation(previous, cells, gradients)
            ),
            current_axis_gradient=current,
            previous_axis_gradient=previous_gradient,
            previous_axis_update=previous_update,
            boundary_features=boundary,
            log_gradient_rms=rms.log(),
            history_valid=valid,
        )
        centers = torch.tensor(rest.cell_rest_centers, dtype=torch.float32)
        edges = {
            hop: build_edge_features(
                centers, positions[:, cells].mean(-2), frames, axes, rest.cell_size, *network.neighborhood(hop)
            )
            for hop in set(network.hops)
        }
        material = (physical.lame_lambda[:1], physical.lame_mu[:1], physical.density[:1], physical.damping[:1])
        conditioning = features.conditioning_channels(*material, rest.cell_size, self.dt)
        prediction = network(axes, state, edges, conditioning[:, None].expand(-1, len(cells), -1))
        fused = fusion.fuse(positions, frames @ (prediction.local_target_axes - axes), pins)
        return {
            "positions": fused,
            "local_target_axes": prediction.local_target_axes,
            "axis_correction": prediction.axis_correction,
            "step_size": prediction.step_size,
            "frames": frames,
            "tie_mask": frame_result.tie_mask,
            "axis_gradient_world": world_gradient,
            "force_residual_norm": gradient.flatten(1).norm(dim=1),
            "achieved_axis_update_world": features.center_deformation(fused.detach(), cells, gradients)
            - deformation.detach(),
            "state_features": state,
            "loss": physical(fused, target, previous_positions=previous),
        }

    def test_network_schema_and_removed_options_are_validated(self):
        """Accept only the revised 61/6/24 schema and reject the removed backtracking controls."""
        for state, conditioning, edge in ((38, 5, 24), (86, 6, 24), (61, 5, 24), (61, 6, 20), (60, 6, 24)):
            network = IntrinsicSolverNetwork(
                self.rest.cell_counts,
                state,
                conditioning_dim=conditioning,
                edge_input_dim=edge,
                hidden_dim=16,
                edge_hidden_dim=8,
            )
            with self.subTest(schema=(state, conditioning, edge)), self.assertRaisesRegex(ValueError, "schema"):
                MixedHexSolverStep(self.rest, self.fixed, network=network, time_step=self.dt)
        with self.assertRaises(TypeError):
            MixedHexSolverStep(
                self.rest, self.fixed, network=self.network, time_step=self.dt, geometry_backtracking=True
            )
        for invalid in (0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(energy_floor_scale=invalid), self.assertRaises(ValueError):
                MixedHexSolverStep(
                    self.rest, self.fixed, network=self.network, time_step=self.dt, energy_floor_scale=invalid
                )
        self.assertFalse(hasattr(self.step, "feasibility"))
        self.assertFalse(hasattr(self.step, "geometry_backtracking"))
        self.assertNotIn("acceptance_scale", LearnedHexStepOutput._fields)
        self.assertEqual(self.step.energy_floor_scale, 1.0)
        self.assertEqual(self.step.network.state_feature_dim, features.STATE_FEATURE_DIM)

    def test_reference_corners_buffer_and_fallback_without_three_pins(self):
        """Register the clamped-face corners and keep the plain frame formula with two pins."""
        expected = select_reference_corners(self.rest.corner_rest_positions, self.fixed)
        self.assertEqual(self.step.reference_corners.dtype, torch.long)
        self.assertEqual(self.step.reference_corners.tolist(), expected.tolist())
        self.assertIn("reference_corners", dict(self.step.named_buffers()))
        pins = np.array([0, 1])
        fallback = MixedHexSolverStep(self.rest, pins, network=self.network, time_step=self.dt)
        self.addCleanup(fallback.close)
        fallback.register_context("soft", **self.specs["soft"])
        self.assertEqual(fallback.reference_corners.numel(), 0)
        mirrored = self.rest_positions.clone()
        mirrored[:, 2] *= -1
        batch = torch.stack((self.x, mirrored))
        with patch("experiments.learned_intrinsic_solver.input_assembly.reference_rotation") as reference:
            inputs = fallback.prepare_inputs(batch, batch, ("soft", "soft"), previous_positions=batch)
        reference.assert_not_called()
        self.assertEqual(inputs.tie_mask.tolist(), [[False] * self.cell_count, [True] * self.cell_count])
        deformation = self._center(batch)
        torch.testing.assert_close(inputs.frames @ inputs.local_axes, deformation, rtol=0, atol=2e-6)
        torch.testing.assert_close(torch.linalg.det(inputs.frames), torch.ones(2, self.cell_count), rtol=0, atol=1e-5)
        output = fallback(batch, batch, ("soft", "soft"), previous_positions=batch, fixed_positions=batch[:, pins])
        self.assertTrue(torch.isfinite(output.loss.total).all())

    def test_frames_are_proper_and_reconstruct_inverted_and_tied_deformation(self):
        """Return right-handed frames whose local axes reproduce F for regular, inverted, and tie cells."""
        inverted = self.x.clone()
        inverted[:, 0] *= -1
        mirrored = self.rest_positions.clone()
        mirrored[:, 2] *= -1
        batch = torch.stack((self.x, inverted, mirrored))
        ids = ("soft", "stiff", "soft")
        previous = self.rest_positions[None].repeat(3, 1, 1)
        inputs = self.step.prepare_inputs(batch, batch, ids, previous_positions=previous)
        frames = inputs.frames
        self.assertEqual(frames.shape, (3, self.cell_count, 3, 3))
        self.assertFalse(frames.requires_grad)
        identity = torch.eye(3).expand(3, self.cell_count, 3, 3)
        torch.testing.assert_close(frames.transpose(-1, -2) @ frames, identity, rtol=0, atol=2e-6)
        torch.testing.assert_close(torch.linalg.det(frames), torch.ones(3, self.cell_count), rtol=0, atol=1e-5)
        deformation = self._center(batch)
        torch.testing.assert_close(frames @ inputs.local_axes, deformation, rtol=0, atol=2e-6)
        determinant = torch.linalg.det(deformation)
        self.assertTrue((determinant[0] > 0).all())
        self.assertTrue((determinant[1] < 0).all())
        self.assertTrue((torch.linalg.det(inputs.local_axes[1]) < 0).all())
        self.assertEqual(inputs.tie_mask.dtype, torch.bool)
        self.assertEqual(inputs.tie_mask.tolist(), [[False] * 4, [False] * 4, [True] * 4])
        # The reference frame from the clamped face resolves the mirrored tie deterministically.
        expected = closest_proper_rotations(
            deformation[2:], reference_rotation(batch[2:], self.step.reference_corners)
        ).frames
        torch.testing.assert_close(frames[2:], expected, rtol=0, atol=1e-6)
        output = self.step(batch, batch, ids, previous_positions=previous, fixed_positions=batch[:, self.fixed])
        torch.testing.assert_close(output.frames, frames, rtol=0, atol=0)
        torch.testing.assert_close(output.tie_mask, inputs.tie_mask, rtol=0, atol=0)

    def test_inverted_and_collapsed_candidates_are_accepted_with_finite_energy_and_gradients(self):
        """Evaluate inverted and collapsed candidates without rejection, shortening, or nonfinite values."""
        inverted = self.x.clone()
        inverted[:, 0] *= -1
        collapsed = self.x.clone()
        collapsed[self.free, 2] = 0
        batch = torch.stack((inverted, collapsed))
        ids = ("soft", "stiff")
        previous = self.rest_positions[None].repeat(2, 1, 1)
        determinant = torch.linalg.det(self._center(batch))
        self.assertTrue((determinant[0] < 0).all())
        torch.testing.assert_close(determinant[1], torch.zeros(self.cell_count), rtol=0, atol=1e-6)
        terms = self.step.energy(batch, batch, ids, previous_positions=previous)
        for name in ("total", "elastic", "inertia", "damping"):
            self.assertTrue(torch.isfinite(getattr(terms, name)).all(), name)
        self.assertTrue((terms.elastic > 0).all())
        output = self.step(batch, batch, ids, previous_positions=previous, fixed_positions=batch[:, self.fixed])
        self.assertTrue(torch.isfinite(output.loss.total).all())
        for name in ("positions", "axis_gradient_world", "achieved_axis_update_world", "force_residual_norm"):
            self.assertTrue(torch.isfinite(getattr(output, name)).all(), name)
        torch.testing.assert_close(output.positions[:, self.fixed], batch[:, self.fixed], rtol=0, atol=0)
        self.assertNotEqual(output.positions.detach().sub(batch).abs().max().item(), 0.0)
        output.loss.total.sum().backward()
        for name, parameter in self.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_gradient_feature_matches_float64_fused_energy_derivative(self):
        """Match the derivative of the fused energy with respect to a world axis increment at zero."""
        ids = ("soft", "stiff")
        positions, target, previous = self._batch(ids)
        inputs = self.step.prepare_inputs(positions, target, ids, previous_positions=previous)
        self.assertEqual(inputs.axis_gradient_world.shape, (2, self.cell_count, 3, 3))
        self.assertFalse(inputs.axis_gradient_world.requires_grad)
        self.assertEqual(inputs.position_gradient.shape, positions.shape)
        torch.testing.assert_close(
            inputs.position_gradient[:, self.fixed], torch.zeros(2, len(self.fixed), 3), rtol=0, atol=0
        )
        generator = torch.Generator().manual_seed(7)
        for index, name in enumerate(ids):
            spec = self.specs[name]
            physical = HexImplicitEulerLoss(
                self.rest, spec["lame_lambda"], spec["lame_mu"], spec["density"], self.dt, dtype=torch.float64
            )
            fusion = HexFusion(
                self.rest, self.fixed, cell_weights=fusion_weights(physical, self.rest.cell_size), dtype=torch.float64
            )
            x = positions[index : index + 1].double()
            y = target[index : index + 1].double()
            start = previous[index : index + 1].double()

            def fused_energy(increment, x=x, y=y, start=start, physical=physical, fusion=fusion):
                return physical(fusion.fuse(x, increment, x[:, self.fixed]), y, previous_positions=start).total.sum()

            zero = torch.zeros((1, self.cell_count, 3, 3), dtype=torch.float64, requires_grad=True)
            expected = torch.autograd.grad(fused_energy(zero), zero)[0]
            direction = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
            epsilon = 1e-6
            numerical = (fused_energy(epsilon * direction) - fused_energy(-epsilon * direction)) / (2 * epsilon)
            torch.testing.assert_close(numerical, (expected * direction).sum(), rtol=1e-6, atol=1e-10)
            scale = expected.abs().max().item()
            self.assertGreater(scale, 0)
            torch.testing.assert_close(
                inputs.axis_gradient_world[index].double(), expected[0], rtol=1e-4, atol=1e-5 * scale, msg=name
            )
            # The position gradient equals the physical objective gradient with zeroed pins.
            candidate = x.clone().requires_grad_(True)
            position_gradient = torch.autograd.grad(
                physical(candidate, y, previous_positions=start).total.sum(), candidate
            )[0]
            position_gradient[:, self.fixed] = 0
            torch.testing.assert_close(
                inputs.position_gradient[index].double(),
                position_gradient[0],
                rtol=1e-4,
                atol=1e-5 * position_gradient.abs().max().item(),
            )

    def test_state_features_follow_leco_normalization_and_history_contract(self):
        """Pack six blocks with the current RMS shared by both gradients and an own RMS for the update."""
        ids = ("soft", "stiff")
        positions, target, previous = self._batch(ids)
        inputs = self.step.prepare_inputs(positions, target, ids, previous_positions=previous)
        state = inputs.state_features
        cells = self.cell_count
        self.assertEqual(state.shape, (2, cells, features.STATE_FEATURE_DIM))
        self.assertEqual(inputs.conditioning.shape, (2, cells, features.CONDITIONING_DIM))
        frames = inputs.frames
        deformation = self._center(positions)
        offset = frames.transpose(-1, -2) @ (self._center(target) - deformation)
        change = frames.transpose(-1, -2) @ (deformation - self._center(previous))
        torch.testing.assert_close(state[..., 0:9], offset.flatten(-2), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(state[..., 9:18], change.flatten(-2), rtol=1e-6, atol=1e-7)
        local_gradient = frames.transpose(-1, -2) @ inputs.axis_gradient_world
        rms = local_gradient.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(features.RMS_FLOOR)
        expected_gradient = (local_gradient / rms).clamp(-features.CLIP, features.CLIP)
        torch.testing.assert_close(state[..., 18:27], expected_gradient.flatten(-2), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(state[..., 27:45], torch.zeros(2, cells, 18), rtol=0, atol=0)
        torch.testing.assert_close(
            state[..., 45:59], self.step.boundary_features[None].expand(2, -1, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(state[..., 59], rms.log().reshape(2, 1).expand(-1, cells), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(state[..., 60], torch.zeros(2, cells), rtol=0, atol=0)
        self.assertGreater(rms.min().item(), features.RMS_FLOOR)
        # Object 0 carries history; object 1 is flagged invalid despite nonzero tensors.
        generator = torch.Generator().manual_seed(11)
        previous_gradient = torch.randn((2, cells, 3, 3), generator=generator) * 40 * rms
        previous_update = torch.randn((2, cells, 3, 3), generator=generator) * 1e-3
        history = OptimizerHistory(previous_gradient, previous_update, torch.tensor([True, False]))
        with_history = self.step.prepare_inputs(positions, target, ids, previous_positions=previous, history=history)
        keep = torch.ones(features.STATE_FEATURE_DIM, dtype=torch.bool)
        keep[27:45] = False
        keep[60] = False
        torch.testing.assert_close(with_history.state_features[..., keep], state[..., keep], rtol=0, atol=0)
        torch.testing.assert_close(with_history.frames, frames, rtol=0, atol=0)
        local_previous = frames.transpose(-1, -2) @ previous_gradient
        expected_previous = (local_previous / rms).clamp(-features.CLIP, features.CLIP)
        block = with_history.state_features[0, :, 27:36]
        torch.testing.assert_close(block, expected_previous[0].flatten(-2), rtol=1e-5, atol=1e-5)
        self.assertEqual(block.abs().max().item(), features.CLIP)
        self.assertGreater((block.abs() == features.CLIP).sum().item(), 0)
        local_update = frames.transpose(-1, -2) @ previous_update
        own_rms = local_update.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(features.RMS_FLOOR)
        expected_update = (local_update / own_rms).clamp(-features.CLIP, features.CLIP)
        update_block = with_history.state_features[0, :, 36:45]
        torch.testing.assert_close(update_block, expected_update[0].flatten(-2), rtol=1e-5, atol=1e-5)
        shared_rms_update = (local_update / rms).clamp(-features.CLIP, features.CLIP)[0].flatten(-2)
        self.assertFalse(torch.allclose(update_block, shared_rms_update, rtol=1e-3, atol=1e-3))
        torch.testing.assert_close(with_history.state_features[0, :, 60], torch.ones(cells), rtol=0, atol=0)
        torch.testing.assert_close(with_history.state_features[1, :, 27:45], torch.zeros(cells, 18), rtol=0, atol=0)
        torch.testing.assert_close(with_history.state_features[1, :, 60], torch.zeros(cells), rtol=0, atol=0)
        # A measured zero update normalizes to zero through the RMS floor instead of NaN.
        zero_history = OptimizerHistory(
            previous_gradient, torch.zeros_like(previous_update), torch.tensor([True, True])
        )
        zero_update = self.step.prepare_inputs(
            positions, target, ids, previous_positions=previous, history=zero_history
        )
        torch.testing.assert_close(zero_update.state_features[..., 36:45], torch.zeros(2, cells, 9), rtol=0, atol=0)
        torch.testing.assert_close(zero_update.state_features[..., 60], torch.ones(2, cells), rtol=0, atol=0)
        # Fully prescribed corners give a zero projected gradient: RMS floors at 1e-12 and log RMS is finite.
        pinned = MixedHexSolverStep(
            self.rest, np.arange(len(self.rest_positions)), network=self.network, time_step=self.dt
        )
        self.addCleanup(pinned.close)
        pinned.register_context("soft", **self.specs["soft"])
        floored = pinned.prepare_inputs(self.x[None], target[:1], ("soft",), previous_positions=previous[:1])
        torch.testing.assert_close(floored.axis_gradient_world, torch.zeros(1, cells, 3, 3), rtol=0, atol=0)
        torch.testing.assert_close(floored.state_features[..., 18:27], torch.zeros(1, cells, 9), rtol=0, atol=0)
        torch.testing.assert_close(
            floored.state_features[..., 59], torch.full((1, cells), math.log(features.RMS_FLOOR)), rtol=1e-6, atol=0
        )
        self.assertTrue(torch.isfinite(floored.state_features).all())

    def test_gradient_feature_is_detached_and_learned_path_reaches_all_parameters(self):
        """Freeze the gradient feature while keeping network, fusion, and energy differentiable."""
        ids = ("stiff", "soft")
        positions, target, previous = self._batch(ids)
        positions.requires_grad_(True)
        inputs = self.step.prepare_inputs(positions, target, ids, previous_positions=previous)
        state = inputs.state_features
        self.assertTrue(state.requires_grad)
        self.assertTrue(inputs.local_axes.requires_grad)
        for name in ("frames", "axis_gradient_world", "position_gradient"):
            self.assertFalse(getattr(inputs, name).requires_grad, name)
        frozen = torch.autograd.grad(
            state[..., 18:45].sum() + state[..., 59:61].sum(), positions, retain_graph=True, allow_unused=True
        )[0]
        self.assertTrue(frozen is None or not frozen.any())
        # The two geometric blocks sum to R^T (F_Y - F_prev), so probe the inertial offset alone.
        live = torch.autograd.grad(state[..., :9].sum(), positions, retain_graph=True)[0]
        self.assertGreater(live.abs().max().item(), 0)
        self.network.zero_grad()
        output = self.step(
            positions.detach(),
            target,
            ids,
            previous_positions=previous,
            fixed_positions=positions.detach()[:, self.fixed],
        )
        self.assertTrue(output.positions.requires_grad)
        self.assertTrue(output.loss.total.requires_grad)
        self.assertEqual(output.step_size.shape, (2, self.cell_count))
        self.assertEqual(output.force_residual_norm.shape, (2,))
        self.assertEqual(output.achieved_axis_update_world.shape, (2, self.cell_count, 3, 3))
        for name in ("axis_gradient_world", "achieved_axis_update_world", "force_residual_norm", "tie_mask"):
            self.assertFalse(getattr(output, name).requires_grad, name)
        expected_residual = torch.linalg.vector_norm(inputs.position_gradient.flatten(1), dim=1)
        torch.testing.assert_close(output.force_residual_norm, expected_residual, rtol=1e-6, atol=0)
        achieved = self._center(output.positions.detach()) - self._center(positions.detach())
        torch.testing.assert_close(output.achieved_axis_update_world, achieved, rtol=0, atol=0)
        torch.testing.assert_close(output.axis_gradient_world, inputs.axis_gradient_world, rtol=0, atol=0)
        output.loss.total.mean().backward()
        for name, parameter in self.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for head in ("correction_head", "step_head", "node_encoder", "edge_encoder", "condition_encoder"):
            module = getattr(self.network, head)
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0, head)

    def test_mixed_forward_and_all_parameter_gradients_match_independent_single_objects(self):
        """Match independently composed per-object queries with one batched network evaluation."""
        ids = ("stiff", "soft", "stiff")
        positions, target, previous = self._batch(ids)
        pins = positions[:, self.fixed].clone()
        generator = torch.Generator().manual_seed(5)
        history = OptimizerHistory(
            torch.randn((3, self.cell_count, 3, 3), generator=generator) * 0.05,
            torch.randn((3, self.cell_count, 3, 3), generator=generator) * 1e-3,
            torch.tensor([True, False, True]),
        )
        reference_network = copy.deepcopy(self.network)
        references = [
            self._single_object(
                self.specs[name],
                reference_network,
                positions[i : i + 1],
                target[i : i + 1],
                previous[i : i + 1],
                pins[i : i + 1],
                OptimizerHistory(
                    history.axis_gradient_world[i : i + 1],
                    history.axis_update_world[i : i + 1],
                    history.valid[i : i + 1],
                ),
            )
            for i, name in enumerate(ids)
        ]
        with patch.object(self.network, "forward", wraps=self.network.forward) as counted:
            output = self.step(
                positions, target, ids, fixed_positions=pins, previous_positions=previous, history=history
            )
        self.assertEqual(counted.call_count, 1)
        inputs = self.step.prepare_inputs(positions, target, ids, previous_positions=previous, history=history)
        torch.testing.assert_close(
            inputs.state_features, torch.cat([r["state_features"] for r in references]), rtol=2e-5, atol=2e-5
        )
        tolerances = {
            "positions": (2e-5, 2e-7),
            "local_target_axes": (2e-5, 1e-6),
            "axis_correction": (2e-5, 2e-7),
            "step_size": (2e-5, 2e-7),
            "frames": (0, 1e-6),
            "axis_gradient_world": (1e-4, 1e-7),
            "force_residual_norm": (1e-5, 1e-7),
            "achieved_axis_update_world": (2e-4, 2e-6),
        }
        for name, (rtol, atol) in tolerances.items():
            expected = torch.cat([r[name] for r in references])
            torch.testing.assert_close(getattr(output, name), expected, rtol=rtol, atol=atol, msg=name)
        self.assertEqual(output.tie_mask.tolist(), torch.cat([r["tie_mask"] for r in references]).tolist())
        torch.testing.assert_close(output.positions[:, self.fixed], pins, rtol=0, atol=0)
        for name in ("total", "elastic", "inertia", "damping"):
            expected = torch.cat([getattr(r["loss"], name) for r in references])
            torch.testing.assert_close(getattr(output.loss, name), expected, rtol=8e-5, atol=3e-7, msg=name)
        output.loss.total.sum().backward()
        torch.cat([r["loss"].total for r in references]).sum().backward()
        expected_parameters = dict(reference_network.named_parameters())
        for name, parameter in self.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, expected_parameters[name].grad, rtol=5e-4, atol=3e-7, msg=name)

    def test_energy_floor_matches_material_formula(self):
        """Return c * eps32 * V * (lambda + 2 mu + eta / dt + rho h^2 / dt^2) per object."""
        self.step.register_context("damped", damping=0.8, **self.specs["soft"])
        ids = ("soft", "stiff", "damped", "soft")
        floor = self.step.energy_floor(ids)
        self.assertEqual(floor.shape, (4,))
        self.assertEqual(floor.dtype, torch.float32)
        self.assertEqual(floor.device, self.step.rest_positions.device)
        self.assertFalse(floor.requires_grad)
        volume = self.cell_count * self.rest.cell_size**3
        expected = []
        for name in ids:
            spec = self.step.context_specs[name]
            modulus = (
                spec["lame_lambda"]
                + 2 * spec["lame_mu"]
                + spec["damping"] / self.dt
                + spec["density"] * self.rest.cell_size**2 / self.dt**2
            )
            expected.append(2.0**-23 * volume * modulus)
        torch.testing.assert_close(floor.double(), torch.tensor(expected, dtype=torch.float64), rtol=2e-6, atol=0)
        self.assertGreater(floor[2].item(), floor[0].item())
        scaled = MixedHexSolverStep(
            self.rest, self.fixed, network=self.network, time_step=self.dt, energy_floor_scale=2.5
        )
        self.addCleanup(scaled.close)
        scaled.register_context("soft", **self.specs["soft"])
        torch.testing.assert_close(scaled.energy_floor(("soft",)), 2.5 * floor[:1], rtol=1e-6, atol=0)
        for invalid in (["soft"], (), "soft"):
            with self.subTest(context_ids=invalid), self.assertRaises(ValueError):
                self.step.energy_floor(invalid)
        with self.assertRaises(KeyError):
            self.step.energy_floor(("missing",))

    def test_context_creation_is_independent_of_network_and_broadcast_buffers(self):
        """Keep worker-created material state out of weights and DDP broadcast buffers."""
        state_keys = tuple(self.step.state_dict())
        with patch.object(self.network, "parameters", side_effect=AssertionError("worker read weights")):
            with patch.object(self.network, "forward", side_effect=AssertionError("worker called network")):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(self.step.register_context, f"worker-{i}", **spec)
                        for i, spec in enumerate(self.specs.values())
                    ]
                    for future in futures:
                        future.result()
        self.assertEqual(tuple(self.step.state_dict()), state_keys)
        for name, _ in self.step.named_buffers():
            self.assertFalse(
                any(part in name for part in ("lame", "density", "lumped_mass", "conditioning", "fusion")), name
            )
        self.assertEqual(
            json.loads(json.dumps(self.step.context_specs))["soft"], {**self.specs["soft"], "damping": 0.0}
        )
        snapshot = self.step.context_specs
        snapshot["soft"]["density"] = -1
        self.assertEqual(self.step.context_specs["soft"]["density"], self.specs["soft"]["density"])

    def test_prepare_matches_native_rigid_candidate_and_original_inertia_target(self):
        """Match native physical preparation without network calls or energy evaluation."""
        solver, problem = self._native("soft", self.x, self.velocity, self.forces)
        expected_candidate = solver.initialize_candidate(problem)
        with patch.object(self.network, "forward", side_effect=AssertionError("prepare called network")):
            with patch.object(self.step, "energy", side_effect=AssertionError("prepare evaluated energy")):
                payload = self.step.prepare("soft", self.x.requires_grad_(), self.velocity, forces=self.forces)
        self.assertEqual(
            set(payload),
            {
                "context_id",
                "physical_positions",
                "velocities",
                "candidate",
                "inertial_prediction",
                "fixed_positions",
                "forces",
            },
        )
        self.assertEqual(payload["context_id"], "soft")
        torch.testing.assert_close(payload["candidate"], expected_candidate[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(payload["inertial_prediction"], problem.inertial_prediction[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(payload["candidate"][self.fixed], self.x[self.fixed], rtol=0, atol=0)
        for name, value in payload.items():
            if name != "context_id":
                self.assertEqual(value.device.type, "cpu")
                self.assertEqual(value.dtype, torch.float32)
                self.assertFalse(value.requires_grad)
        torch.save(payload, io.BytesIO())

    def test_advance_recomputes_native_problem_once_from_committed_candidate(self):
        """Reset velocity from displacement and recompute physical guidance exactly once."""
        payload = self.step.prepare("stiff", self.x, self.velocity, forces=self.forces)
        candidate = payload["candidate"].clone()
        candidate[:, 0] += 0.01 * candidate[:, 2].square()
        payload["candidate"] = candidate
        expected_velocity = (candidate - self.x) / self.dt
        expected_velocity[self.fixed] = 0
        solver, problem = self._native("stiff", candidate, expected_velocity, self.forces)
        with patch.object(self.step, "prepare", wraps=self.step.prepare) as counted:
            advanced = self.step.advance(payload)
        self.assertEqual(counted.call_count, 1)
        torch.testing.assert_close(advanced["physical_positions"], candidate, rtol=0, atol=0)
        torch.testing.assert_close(advanced["velocities"], expected_velocity, rtol=0, atol=0)
        torch.testing.assert_close(advanced["candidate"], solver.initialize_candidate(problem)[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(
            advanced["inertial_prediction"], problem.inertial_prediction[0], rtol=2e-6, atol=5e-8
        )

    def test_advance_carries_inverted_committed_candidate_without_repair(self):
        """Advance a folded committed candidate: finite payload, exact positions, and a still-inverted initializer."""
        payload = self.step.prepare("soft", self.x, self.velocity, forces=self.forces)
        candidate = payload["candidate"].clone()
        top = self.rest_positions[:, 2] == self.rest_positions[:, 2].max()
        candidate[top, 2] = 0.05
        self.assertLess(torch.linalg.det(self._center(candidate[None])).min().item(), 0)
        payload["candidate"] = candidate
        advanced = self.step.advance(payload)
        self.assertEqual(set(advanced), set(payload))
        for name, value in advanced.items():
            if name != "context_id":
                self.assertTrue(torch.isfinite(value).all(), name)
        torch.testing.assert_close(advanced["physical_positions"], candidate, rtol=0, atol=0)
        expected_velocity = (candidate - self.x) / self.dt
        expected_velocity[self.fixed] = 0
        torch.testing.assert_close(advanced["velocities"], expected_velocity, rtol=0, atol=0)
        # The next rigid initializer keeps the fold instead of repairing or shortening it, and still solves.
        self.assertLess(torch.linalg.det(self._center(advanced["candidate"][None])).min().item(), 0)
        output = self.step(
            advanced["candidate"][None],
            advanced["inertial_prediction"][None],
            ("soft",),
            previous_positions=advanced["physical_positions"][None],
            fixed_positions=advanced["fixed_positions"][None],
        )
        self.assertTrue(torch.isfinite(output.positions).all())
        self.assertTrue(torch.isfinite(output.loss.total).all())

    def test_invalid_contexts_inputs_and_history_are_rejected_explicitly(self):
        """Reject missing contexts, bad materials, nonfinite or mismatched inputs, and malformed history."""
        with self.assertRaises(ValueError):
            self.step.register_context("soft", **self.specs["soft"])
        with self.assertRaises(ValueError):
            self.step.register_context("invalid", lame_lambda=-1, lame_mu=1, density=1)
        positions = self.x[None]
        with self.assertRaises(ValueError):
            self.step.energy(positions, positions, ("soft", "stiff"))
        self.step.discard_context("soft")
        self.assertNotIn("soft", self.step.context_specs)
        with self.assertRaises(KeyError):
            self.step.energy(positions, positions, ("soft",))
        self.step.register_context("soft", **self.specs["soft"])
        nonfinite = positions.clone()
        nonfinite[0, self.free[0], 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.step.energy(nonfinite, positions, ("soft",))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.step.prepare_inputs(positions, nonfinite, ("soft",), previous_positions=positions)
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            self.step.prepare_inputs(positions, positions, ("soft",), previous_positions=None)
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            self.step(positions, positions, ("soft",), previous_positions=positions.repeat(2, 1, 1))
        with self.assertRaises(TypeError):
            self.step(positions, positions, ("soft",))
        cells = self.cell_count
        good = OptimizerHistory(torch.zeros(1, cells, 3, 3), torch.zeros(1, cells, 3, 3), torch.tensor([False]))
        self.step.prepare_inputs(positions, positions, ("soft",), previous_positions=positions, history=good)
        with self.assertRaises(TypeError):
            self.step.prepare_inputs(positions, positions, ("soft",), previous_positions=positions, history=tuple(good))
        malformed = (
            OptimizerHistory(torch.zeros(2, cells, 3, 3), good.axis_update_world, good.valid),
            OptimizerHistory(good.axis_gradient_world.double(), good.axis_update_world, good.valid),
            OptimizerHistory(good.axis_gradient_world, torch.full((1, cells, 3, 3), float("nan")), good.valid),
            OptimizerHistory(good.axis_gradient_world, good.axis_update_world, torch.tensor([1.0])),
            OptimizerHistory(good.axis_gradient_world, good.axis_update_world, torch.tensor([False, False])),
        )
        for history in malformed:
            with self.subTest(history=history), self.assertRaises(ValueError):
                self.step(positions, positions, ("soft",), previous_positions=positions, history=history)
        with self.assertRaisesRegex(ValueError, "fixed_positions"):
            self.step(positions, positions, ("soft",), previous_positions=positions, fixed_positions=positions)


if __name__ == "__main__":
    unittest.main()
