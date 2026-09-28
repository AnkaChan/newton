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

from experiments.learned_intrinsic_solver import features, hex_modes
from experiments.learned_intrinsic_solver.contact_geometry import sample_points
from experiments.learned_intrinsic_solver.contact_scene import KIND_PLANE, KIND_POINT, ContactPartners, detect_contacts
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.frames import (
    closest_proper_rotations,
    reference_rotation,
    select_reference_corners,
)
from experiments.learned_intrinsic_solver.fusion import HexFusion
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss, HexLossTerms
from experiments.learned_intrinsic_solver.input_assembly import assemble_inputs
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep, OptimizerHistory
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork, IntrinsicSolverOutput
from experiments.learned_intrinsic_solver.network_geometry import build_edge_features
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep, LearnedHexStepOutput

MATERIALS = {
    "soft": {"lame_lambda": 700 * 0.2 / (1.2 * 0.6), "lame_mu": 700 / 2.4, "density": 90.0},
    "stiff": {"lame_lambda": 3000 * 0.4 / (1.4 * 0.2), "lame_mu": 3000 / 2.8, "density": 180.0},
}


def make_network(cell_counts, *, target_modes=3, **kwargs):
    """Build a small revised-schema network with ``target_modes`` modes and nonzero heads so outputs vary per cell."""
    network = IntrinsicSolverNetwork(
        cell_counts,
        features.state_feature_dim(target_modes),
        target_modes=target_modes,
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


def floor_partners(height, *, ke=500.0, kd=1.0, mu=0.3):
    """Return partners with only a ground plane at ``height`` [m] and no static points."""
    return ContactPartners(
        plane_present=True,
        plane_point=torch.tensor([0.0, height, 0.0]),
        plane_normal=torch.tensor([0.0, 1.0, 0.0]),
        point_positions=torch.zeros((0, 3)),
        point_normals=torch.zeros((0, 3)),
        point_radii=torch.zeros((0,)),
        ke=ke,
        kd=kd,
        mu=mu,
    )


def contact_batch(payloads):
    """Collate the contact pair tensors of several payloads, padded to the largest Q with a mask."""
    counts = [int(payload["contact_sample_index"].shape[0]) for payload in payloads]
    width = max(counts)

    def pad(name, *tail):
        rows = []
        for payload in payloads:
            value = payload[name]
            padded = torch.zeros((width, *tail), dtype=value.dtype)
            padded[: value.shape[0]] = value
            rows.append(padded)
        return torch.stack(rows)

    return {
        "sample_index": pad("contact_sample_index"),
        "kind": pad("contact_kind"),
        "partner_point": pad("contact_partner_point", 3),
        "partner_normal": pad("contact_partner_normal", 3),
        "partner_radius": pad("contact_partner_radius"),
        "mask": torch.stack([torch.arange(width) < count for count in counts]),
    }


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
        """Compose the SI query for one object without the mixed step.

        The mixed step evaluates the objective in cell units, so its gradient
        feature (and the history it consumes) is the SI fusion-projected
        gradient divided by ``mu h^3``; the position gradient and the force
        residual stay in newtons. Uniform physical fusion weights and unit
        weights give the same fit.
        """
        rest = self.rest
        energy_unit = spec["lame_mu"] * rest.cell_size**3
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
        world_gradient = fusion.project_gradient(gradient) / energy_unit
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
        conditioning = features.conditioning_channels(*material, rest.cell_size, self.dt, self.gravity)
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
        """Accept only the 61/7/24 widths (schema 5, dimensionless conditioning) and reject removed controls."""
        for state, conditioning, edge in (
            (38, 5, 24),
            (86, 6, 24),
            (61, 6, 24),
            (61, 5, 24),
            (61, 9, 24),
            (61, 7, 20),
            (60, 7, 24),
        ):
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
        # The reference frame from the clamped face resolves the mirrored tie deterministically. The step decomposes
        # the deformation of the positions in cell units; a tie is a degenerate decomposition, so the SI deformation
        # (identical up to 1e-7) may resolve to a frame a few 1e-5 away, hence the production arithmetic is used.
        unit = batch[2:] / self.rest.cell_size
        expected = closest_proper_rotations(
            features.center_deformation(unit, self.step.cell_corner_indices, self.step.unit_center_gradients),
            reference_rotation(unit, self.step.reference_corners),
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
        """Match the derivative of the fused SI energy with respect to a world axis increment at zero.

        The step reports the axis gradient in ``mu h^3`` and the position
        gradient in newtons, the convention shared with LearnedHexSolverStep.
        """
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
            energy_unit = spec["lame_mu"] * self.rest.cell_size**3
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
                inputs.axis_gradient_world[index].double() * energy_unit,
                expected[0],
                rtol=1e-4,
                atol=1e-5 * scale,
                msg=name,
            )
            # The position gradient equals the physical objective gradient [N] with zeroed pins.
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
        # The step forms F from the positions in cell units; the SI reference agrees to float32 rounding.
        torch.testing.assert_close(state[..., 0:9], offset.flatten(-2), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(state[..., 9:18], change.flatten(-2), rtol=1e-6, atol=1e-6)
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
        # The residual is the norm of the position gradient; both are reported in newtons.
        expected_residual = torch.linalg.vector_norm(inputs.position_gradient.flatten(1), dim=1)
        torch.testing.assert_close(output.force_residual_norm, expected_residual, rtol=0, atol=0)
        achieved = self._center(output.positions.detach()) - self._center(positions.detach())
        torch.testing.assert_close(output.achieved_axis_update_world, achieved, rtol=0, atol=2e-6)
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
            # The gradient feature is O(1) in cell units; two factorisations agree to float32 rounding.
            "axis_gradient_world": (1e-4, 1e-5),
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

    def test_gravity_is_kept_in_float64_for_the_conditioning_and_validated(self):
        """Form the gravity channel from the float64 vector so it equals the single-material step's bit for bit.

        A float32 copy of (0, -9.81, 0) has magnitude 9.8100004, which moves
        ``log1p(|g| dt^2 / h)`` by one float32 ulp at ``dt = 1/300``, ``h =
        0.025``; the float32 vector is used only for the inertial prediction.
        """
        rest = generate_cuboid((2, 1, 1), cell_size=0.025)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        dt, gravity = 1.0 / 300.0, (0.0, -9.81, 0.0)
        network = make_network(rest.cell_counts)
        mixed = MixedHexSolverStep(rest, fixed, network=network, time_step=dt, gravity=gravity)
        self.addCleanup(mixed.close)
        self.assertEqual(mixed.gravity, gravity)
        self.assertIsInstance(mixed.gravity[1], float)
        spec = self.specs["soft"]
        mixed.register_context("soft", **spec)
        single = LearnedHexSolverStep(rest, fixed, **spec, time_step=dt, gravity=gravity, network=network)
        x = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        channel = features.CONDITIONING_CHANNELS.index("log1p_gravity_ratio")
        shared = mixed.prepare_inputs(x, x, ("soft",), previous_positions=x).conditioning[..., channel]
        expected = torch.full_like(shared, math.log1p(9.81 * dt**2 / rest.cell_size))
        torch.testing.assert_close(shared, expected, rtol=0, atol=0)
        torch.testing.assert_close(shared, single.conditioning[None, :, channel], rtol=0, atol=0)
        rounded = torch.full_like(shared, math.log1p(float(np.float32(9.81)) * dt**2 / rest.cell_size))
        self.assertFalse(torch.equal(shared, rounded), "the float32 magnitude would give a different channel")
        # The inertial prediction still adds g dt^2 in float32.
        payload = mixed.prepare("soft", x[0], torch.zeros_like(x[0]))
        drop = torch.tensor(gravity, dtype=torch.float32) * dt**2
        torch.testing.assert_close(payload["inertial_prediction"] - x[0], drop.expand_as(x[0]), rtol=0, atol=1e-9)
        for bad in ("g", None, (0.0, -9.81), (0.0, math.nan, 0.0), ((0.0, 1.0), 2.0)):
            with self.subTest(gravity=bad), self.assertRaisesRegex(ValueError, "gravity"):
                MixedHexSolverStep(rest, fixed, network=network, time_step=dt, gravity=bad)

    def test_far_origin_builds_the_unit_grid_exactly(self):
        """Accept a rest grid whose origin / h is large: the unit lattice is generated, not divided.

        Dividing the SI corners by h leaves a 1e-12 deviation from the canonical
        unit lattice at an origin of 1000 m with h = 0.025, which the grid check
        of HexImplicitEulerLoss rejects although the SI grid itself passes.
        """
        far = generate_cuboid((2, 2, 3), cell_size=0.025, origin=(1000.0, 1000.0, 1000.0))
        fixed = np.flatnonzero(far.corner_rest_positions[:, 2] == far.corner_rest_positions[:, 2].min())
        HexImplicitEulerLoss(far, 0.0, 1.0, 1.0, self.dt)  # the SI grid is canonical
        step = MixedHexSolverStep(far, fixed, network=make_network(far.cell_counts), time_step=self.dt)
        self.addCleanup(step.close)
        unit = step._unit_rest
        lattice = generate_cuboid(far.cell_counts, cell_size=1.0, origin=tuple(far.corner_rest_positions[0] / 0.025))
        np.testing.assert_array_equal(unit.corner_rest_positions, lattice.corner_rest_positions)
        np.testing.assert_array_equal(unit.cell_corner_indices, far.cell_corner_indices)
        self.assertEqual(unit.cell_size, 1.0)
        np.testing.assert_allclose(unit.cell_rest_centers, far.cell_rest_centers / 0.025, rtol=0, atol=1e-9)
        step.register_context("soft", **self.specs["soft"])
        x = torch.tensor(far.corner_rest_positions, dtype=torch.float32)[None]
        terms = step.energy(x, x, ("soft",))
        # X / h is about 4e4 in float32, so the rest energy is zero only to rounding (5.7e-18 J observed).
        self.assertLessEqual(terms.total.item(), 1e-9 * step.energy_floor(("soft",)).item())
        self.assertTrue(torch.isfinite(step(x, x, ("soft",), previous_positions=x).positions).all())

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
            json.loads(json.dumps(self.step.context_specs))["soft"],
            {**self.specs["soft"], "damping": 0.0, "gravity": [0.0, -9.81, 0.0]},
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
                "contact_sample_index",
                "contact_kind",
                "contact_partner_point",
                "contact_partner_normal",
                "contact_partner_radius",
            },
        )
        self.assertEqual(payload["context_id"], "soft")
        torch.testing.assert_close(payload["candidate"], expected_candidate[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(payload["inertial_prediction"], problem.inertial_prediction[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(payload["candidate"][self.fixed], self.x[self.fixed], rtol=0, atol=0)
        for name, value in payload.items():
            if name != "context_id":
                self.assertEqual(value.device.type, "cpu")
                expected_dtype = torch.int64 if name in ("contact_sample_index", "contact_kind") else torch.float32
                self.assertEqual(value.dtype, expected_dtype, name)
                self.assertFalse(value.requires_grad)
        # A context registered without partners is contact-free: Q = 0.
        self.assertEqual(payload["contact_sample_index"].shape, (0,))
        self.assertEqual(payload["contact_partner_point"].shape, (0, 3))
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

    def _seven_mode_step(self, network):
        step = MixedHexSolverStep(
            self.rest, self.fixed, network=network, time_step=self.dt, gravity=self.gravity, target_modes=7
        )
        self.addCleanup(step.close)
        for name, spec in self.specs.items():
            step.register_context(name, **spec)
        return step

    def test_target_modes_must_match_the_network(self):
        """Reject a network whose mode count or state width disagrees with the step, and invalid mode counts."""
        with self.assertRaisesRegex(ValueError, "schema"):
            MixedHexSolverStep(self.rest, self.fixed, network=self.network, time_step=self.dt, target_modes=7)
        seven = make_network(self.rest.cell_counts, target_modes=7)
        with self.assertRaisesRegex(ValueError, "schema"):
            MixedHexSolverStep(self.rest, self.fixed, network=seven, time_step=self.dt)
        wide = IntrinsicSolverNetwork(
            self.rest.cell_counts, features.state_feature_dim(7), hidden_dim=16, edge_hidden_dim=8
        )
        with self.assertRaisesRegex(ValueError, "schema"):
            MixedHexSolverStep(self.rest, self.fixed, network=wide, time_step=self.dt, target_modes=7)
        for bad in (5, 1, True, "7"):
            with self.subTest(target_modes=bad), self.assertRaisesRegex(ValueError, "target_modes"):
                MixedHexSolverStep(self.rest, self.fixed, network=seven, time_step=self.dt, target_modes=bad)
        step = self._seven_mode_step(seven)
        self.assertEqual((step.target_modes, step.network.target_modes), (7, 7))

    def test_seven_mode_forward_shapes_energy_and_history(self):
        """Run the seven-mode mixed step on the tiny grid: [B, C, 3, 7] blocks, finite fused positions,
        an energy that energy() reproduces exactly and a history that round-trips through forward."""
        network = make_network(self.rest.cell_counts, target_modes=7, contact_tokens=True)
        self.assertEqual(
            network.node_encoder[0].in_features, 21 + features.state_feature_dim(7) + features.CONTACT_FEATURE_DIM
        )
        step = self._seven_mode_step(network)
        ids = ("soft", "stiff")
        positions, target, previous = self._batch(ids)
        pins = positions[:, self.fixed].clone()
        cells, h = self.cell_count, self.rest.cell_size
        inputs = step.prepare_inputs(positions, target, ids, previous_positions=previous)
        self.assertEqual(inputs.local_axes.shape, (2, cells, 3, 7))
        self.assertEqual(inputs.state_features.shape, (2, cells, 121))
        self.assertEqual(inputs.axis_gradient_world.shape, (2, cells, 3, 7))
        self.assertEqual(inputs.contact_tokens.shape[:2], (2, cells))
        corners = step.cell_corner_indices
        vectors = hex_modes.mode_vectors(positions / h, corners, 1.0)
        torch.testing.assert_close(inputs.frames @ inputs.local_axes, vectors, rtol=0, atol=2e-6)
        self.assertGreater(vectors[..., 3:].abs().max().item(), 1e-5, "the fixture warps its cells")
        expected_frames = closest_proper_rotations(
            vectors[..., :3], reference_rotation(positions / h, step.reference_corners)
        ).frames
        torch.testing.assert_close(inputs.frames, expected_frames, rtol=0, atol=0)
        offset = inputs.frames.transpose(-1, -2) @ (hex_modes.mode_vectors(target / h, corners, 1.0) - vectors)
        torch.testing.assert_close(inputs.state_features[..., :21], offset.flatten(-2), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(inputs.state_features[..., 63:105], torch.zeros(2, cells, 42), rtol=0, atol=0)
        torch.testing.assert_close(inputs.state_features[..., 120], torch.zeros(2, cells), rtol=0, atol=0)
        # The gradient feature is the seven-mode adjoint projection: affine columns match the three-mode step.
        reference = self.step.prepare_inputs(positions, target, ids, previous_positions=previous)
        scale = reference.axis_gradient_world.abs().max().item()
        torch.testing.assert_close(
            inputs.axis_gradient_world[..., :3], reference.axis_gradient_world, rtol=1e-5, atol=1e-5 * scale
        )
        self.assertGreater(inputs.axis_gradient_world[..., 3:].abs().max().item(), 0)
        torch.testing.assert_close(inputs.position_gradient, reference.position_gradient, rtol=1e-6, atol=1e-6)

        output = step(positions, target, ids, previous_positions=previous, fixed_positions=pins)
        self.assertEqual(output.local_target_axes.shape, (2, cells, 3, 7))
        self.assertEqual(output.axis_correction.shape, (2, cells, 3, 7))
        self.assertEqual(output.achieved_axis_update_world.shape, (2, cells, 3, 7))
        self.assertEqual(output.step_size.shape, (2, cells))
        self.assertTrue(torch.isfinite(output.positions).all())
        torch.testing.assert_close(output.positions[:, self.fixed], pins, rtol=0, atol=0)
        self.assertGreater((output.positions.detach() - positions).abs().max().item(), 0)
        energy = step.energy(output.positions.detach(), target, ids, previous_positions=previous)
        torch.testing.assert_close(energy.total, output.loss.total.detach(), rtol=0, atol=0)
        achieved = hex_modes.mode_vectors(output.positions.detach() / h, corners, 1.0) - vectors
        torch.testing.assert_close(output.achieved_axis_update_world, achieved, rtol=0, atol=2e-6)
        self.assertGreater(output.achieved_axis_update_world[..., 3:].abs().max().item(), 0)
        output.loss.total.sum().backward()
        gradient = network.correction_head.weight.grad
        warping_rows = torch.arange(21).reshape(3, 7)[:, 3:].flatten()
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[warping_rows].abs().sum().item(), 0)
        for name, parameter in network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
        history = OptimizerHistory(
            output.axis_gradient_world, output.achieved_axis_update_world, torch.tensor([True, False])
        )
        second = step.prepare_inputs(
            output.positions.detach(), target, ids, previous_positions=previous, history=history
        )
        torch.testing.assert_close(second.state_features[:, :, 120], torch.tensor([[1.0] * cells, [0.0] * cells]))
        self.assertGreater(second.state_features[0, :, 84:105].abs().max().item(), 0)
        torch.testing.assert_close(second.state_features[1, :, 63:105], torch.zeros(cells, 42), rtol=0, atol=0)
        legacy = OptimizerHistory(torch.zeros(2, cells, 3, 3), torch.zeros(2, cells, 3, 3), torch.tensor([True, True]))
        with self.assertRaisesRegex(ValueError, r"\[B, C, 3, 7\]"):
            step.prepare_inputs(positions, target, ids, previous_positions=previous, history=legacy)
        # prepare() fuses a seven-mode zero increment and yields the same candidate as the three-mode step.
        payload = step.prepare("soft", self.x, self.velocity, forces=self.forces)
        reference_payload = self.step.prepare("soft", self.x, self.velocity, forces=self.forces)
        torch.testing.assert_close(payload["candidate"], reference_payload["candidate"], rtol=0, atol=0)

    def test_seven_modes_reproduce_three_modes_for_affine_only_targets(self):
        """Match the three-mode step when the warping outputs are zero: identical zero-increment fusions and
        1e-6 agreement for a hand-set affine increment fed through both fusions."""
        three_network = IntrinsicSolverNetwork(
            self.rest.cell_counts, features.STATE_FEATURE_DIM, hidden_dim=16, edge_hidden_dim=8
        )
        seven_network = IntrinsicSolverNetwork(
            self.rest.cell_counts, features.state_feature_dim(7), target_modes=7, hidden_dim=16, edge_hidden_dim=8
        )
        three = MixedHexSolverStep(self.rest, self.fixed, network=three_network, time_step=self.dt)
        self.addCleanup(three.close)
        seven = MixedHexSolverStep(self.rest, self.fixed, network=seven_network, time_step=self.dt, target_modes=7)
        self.addCleanup(seven.close)
        for step in (three, seven):
            for name, spec in self.specs.items():
                step.register_context(name, **spec)
        ids = ("soft", "stiff")
        positions, target, previous = self._batch(ids)
        pins = positions[:, self.fixed] + torch.tensor([0.002, -0.001, 0.0015])
        # Zero-initialised heads: targets equal the inputs, so both steps fuse a zero increment with displaced pins.
        a = three(positions, target, ids, previous_positions=previous, fixed_positions=pins)
        b = seven(positions, target, ids, previous_positions=previous, fixed_positions=pins)
        self.assertGreater((a.positions - positions).abs().max().item(), 1e-4, "the displaced pins move corners")
        torch.testing.assert_close(b.positions, a.positions, rtol=0, atol=0)
        torch.testing.assert_close(b.loss.total, a.loss.total, rtol=0, atol=0)
        torch.testing.assert_close(b.frames, a.frames, rtol=0, atol=0)
        # A hand-set affine-only local increment through the whole update path of both steps.
        delta = 0.01 * torch.randn((2, self.cell_count, 3, 3), generator=torch.Generator().manual_seed(4))

        def affine_only(local_axes, state_features, edge_features, conditioning, **kwargs):
            correction = torch.zeros_like(local_axes)
            correction[..., :3] = delta
            return IntrinsicSolverOutput(local_axes + correction, correction, torch.ones(local_axes.shape[:2]))

        with patch.object(three_network, "forward", affine_only), patch.object(seven_network, "forward", affine_only):
            a = three(positions, target, ids, previous_positions=previous, fixed_positions=pins)
            b = seven(positions, target, ids, previous_positions=previous, fixed_positions=pins)
        self.assertGreater((a.positions - positions).abs().max().item(), 1e-3, "the increment moves the corners")
        torch.testing.assert_close(b.positions, a.positions, rtol=0, atol=1e-6)
        torch.testing.assert_close(b.loss.total, a.loss.total, rtol=1e-5, atol=1e-9)
        # ... and directly through the two shared unit-grid fusions with the same world increment.
        h = self.rest.cell_size
        world = a.frames @ delta
        padded = torch.zeros((2, self.cell_count, 3, 7))
        padded[..., :3] = world
        torch.testing.assert_close(
            seven._fusion.fuse(positions / h, padded, pins / h),
            three._fusion.fuse(positions / h, world, pins / h),
            rtol=0,
            atol=1e-6,
        )

    def test_per_context_gravity_drives_prediction_conditioning_predictor_and_specs(self):
        """Give every context its own gravity: inertial prediction, conditioning channel, rigid predictor
        and context_specs follow it; None falls back to the step's constructor gravity."""
        rest = generate_cuboid((2, 1, 1), cell_size=0.025)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        h, dt = rest.cell_size, 1.0 / 300.0
        network = make_network(rest.cell_counts)
        step = MixedHexSolverStep(rest, fixed, network=network, time_step=dt, gravity=(0.0, -9.81, 0.0))
        self.addCleanup(step.close)
        spec = self.specs["soft"]
        gravities = {"earth": (0.0, -9.81, 0.0), "heavy": (0.0, -30.0, 0.0), "tilted": (1.0, -2.0, 0.5)}
        step.register_context("earth", **spec)
        step.register_context("heavy", **spec, gravity=(0.0, -30.0, 0.0))
        step.register_context("tilted", **spec, gravity=torch.tensor([1.0, -2.0, 0.5]))
        specs = step.context_specs
        for name, gravity in gravities.items():
            self.assertEqual(specs[name]["gravity"], gravity, name)
            self.assertTrue(all(isinstance(value, float) for value in specs[name]["gravity"]), name)
        self.assertEqual(set(specs["heavy"]), {"lame_lambda", "lame_mu", "density", "damping", "gravity"})
        # A reported specification rebuilds an identical context.
        step.register_context("rebuilt", **specs["heavy"])
        self.assertEqual(step.context_specs["rebuilt"], specs["heavy"])
        ids = tuple(gravities)
        x = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None].repeat(len(ids), 1, 1)
        inputs = step.prepare_inputs(x, x, ids, previous_positions=x)
        channel = features.CONDITIONING_CHANNELS.index("log1p_gravity_ratio")
        material = torch.tensor([[spec["lame_lambda"], spec["lame_mu"], spec["density"], 0.0]], dtype=torch.float32)
        expected = torch.stack(
            [features.conditioning_channels(*material.unbind(-1), h, dt, gravity)[0] for gravity in gravities.values()]
        )
        torch.testing.assert_close(inputs.conditioning[:, 0], expected, rtol=0, atol=0)
        self.assertEqual(inputs.conditioning[:, 0, channel].unique().numel(), 3)
        others = [index for index in range(features.CONDITIONING_DIM) if index != channel]
        torch.testing.assert_close(
            inputs.conditioning[1:, :, others], inputs.conditioning[:1, :, others].expand(2, -1, -1), rtol=0, atol=0
        )
        # The single-material step at the same gravity forms the same channel bit for bit.
        single = LearnedHexSolverStep(rest, fixed, **spec, time_step=dt, gravity=(0.0, -30.0, 0.0), network=network)
        torch.testing.assert_close(inputs.conditioning[1, :, channel], single.conditioning[:, channel], rtol=0, atol=0)
        # prepare() adds the context's g dt^2 to the inertial prediction and drives the predictor with it.
        velocity = torch.zeros_like(x[0])
        payloads = {name: step.prepare(name, x[0], velocity) for name in ids}
        for name, gravity in gravities.items():
            drop = torch.tensor(gravity, dtype=torch.float32) * dt**2
            torch.testing.assert_close(
                payloads[name]["inertial_prediction"] - x[0], drop.expand_as(x[0]), rtol=0, atol=1e-8, msg=name
            )
            model_gravity = step._contexts[name].predictor.model.gravity.numpy()[0]
            np.testing.assert_allclose(model_gravity, np.array(gravity, dtype=np.float32), rtol=0, atol=0, err_msg=name)
        heavy = build_newton_hex_model(rest, fixed, gravity=(0.0, -30.0, 0.0), **spec)
        state = heavy.state()
        state.particle_q.assign(x[0].numpy())
        state.particle_qd.assign(velocity.numpy())
        solver = SolverLearnedIntrinsic(
            heavy, network=IntrinsicSolverNetwork(rest.cell_counts, features.STATE_FEATURE_DIM)
        )
        problem = solver.prepare_problem(state, dt)
        torch.testing.assert_close(
            payloads["heavy"]["inertial_prediction"], problem.inertial_prediction[0], rtol=2e-6, atol=5e-8
        )
        torch.testing.assert_close(
            payloads["heavy"]["candidate"], solver.initialize_candidate(problem)[0], rtol=2e-6, atol=5e-8
        )
        for bad in ((0.0, -9.81), "g", (0.0, math.nan, 0.0), ((0.0, 1.0), 2.0)):
            with self.subTest(gravity=bad), self.assertRaisesRegex(ValueError, "gravity"):
                step.register_context(f"bad-{bad!r}", **spec, gravity=bad)
        self.assertNotIn("bad-'g'", step.context_specs)


class TestMixedHexSolverStepContact(unittest.TestCase):
    """Contact handling of the mixed step: detection, energy, tokens, conditioning and diagnostics."""

    def setUp(self):
        """Build a (2, 1, 2) grid clamped at z = 0 whose bottom faces lie at y = 0, with r = 0.05 m."""
        torch.manual_seed(404)
        self.rest = generate_cuboid((2, 1, 2), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.cell_count = len(self.rest.cell_corner_indices)
        self.dt = 0.01
        self.spec = dict(MATERIALS["soft"])
        self.network = make_network(self.rest.cell_counts)
        self.step = MixedHexSolverStep(self.rest, self.fixed, network=self.network, time_step=self.dt)
        self.addCleanup(self.step.close)
        self.rest_positions = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)
        self.bottom = torch.tensor(self.rest.corner_rest_positions[:, 1] == 0.0)
        self.radius = 0.5 * self.rest.cell_size

    def _contact_step(self, **kwargs):
        """Return a step whose network consumes contact tokens, with a nonzero pooled projection."""
        network = make_network(self.rest.cell_counts, contact_tokens=True)
        with torch.no_grad():
            network.contact_encoder.pool_projection.weight.normal_(std=0.1)
        step = MixedHexSolverStep(self.rest, self.fixed, network=network, time_step=self.dt, **kwargs)
        self.addCleanup(step.close)
        return step

    def _payloads(self, step, ids, velocities=None):
        velocity = torch.zeros_like(self.rest_positions) if velocities is None else velocities
        return [step.prepare(name, self.rest_positions, velocity) for name in ids]

    @staticmethod
    def _stack(payloads, name):
        return torch.stack([payload[name] for payload in payloads])

    def test_constructor_defaults_and_validation(self):
        """Default r to 0.5 h, register the face tables as buffers and reject invalid contact parameters."""
        self.assertEqual(self.step.contact_radius, self.radius)
        self.assertEqual(self.step.contact_max_pairs, 4)
        self.assertEqual(self.step.contact_tokens_per_cell, 24)
        self.assertEqual(self.step.contact_friction_epsilon, 1e-2)
        buffers = dict(self.step.named_buffers())
        self.assertEqual(buffers["face_corners"].shape, (16, 4))
        self.assertEqual(buffers["face_cell_index"].shape, (16,))
        self.assertEqual(buffers["face_corners"].dtype, torch.int64)
        custom = MixedHexSolverStep(
            self.rest,
            self.fixed,
            network=self.network,
            time_step=self.dt,
            contact_radius=0.02,
            contact_max_pairs=2,
            contact_tokens_per_cell=6,
            contact_friction_epsilon=0.5,
        )
        self.addCleanup(custom.close)
        self.assertEqual(
            (
                custom.contact_radius,
                custom.contact_max_pairs,
                custom.contact_tokens_per_cell,
                custom.contact_friction_epsilon,
            ),
            (0.02, 2, 6, 0.5),
        )
        for invalid in (
            {"contact_radius": 0.0},
            {"contact_radius": math.nan},
            {"contact_max_pairs": -1},
            {"contact_max_pairs": 2.0},
            {"contact_tokens_per_cell": 0},
            {"contact_friction_epsilon": 0.0},
        ):
            with self.subTest(**invalid), self.assertRaises(ValueError):
                MixedHexSolverStep(self.rest, self.fixed, network=self.network, time_step=self.dt, **invalid)
        with self.assertRaises(ValueError):
            self.step.register_context("bad", **self.spec, contact={"plane_present": False})

    def test_contact_free_context_reproduces_contact_less_objective(self):
        """Give a zero contact term, total = elastic + inertia + damping, zero contact channels and diagnostics."""
        self.step.register_context("free", **self.spec)
        self.step.register_context("damped", damping=0.5, **self.spec)
        ids = ("free", "damped")
        payloads = self._payloads(self.step, ids)
        positions = self._stack(payloads, "candidate")
        positions[:, ~self.bottom, 1] -= 0.01
        target = self._stack(payloads, "inertial_prediction")
        previous = self._stack(payloads, "physical_positions")
        terms = self.step.energy(positions, target, ids, previous_positions=previous)
        self.assertIsInstance(terms, HexLossTerms)
        self.assertEqual(terms.contact.shape, (2,))
        torch.testing.assert_close(terms.contact, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(terms.total, terms.elastic + terms.inertia + terms.damping, rtol=0, atol=0)
        self.assertGreater(terms.damping[1].item(), 0.0)
        # A padded batch with Q = 0 pairs is the same as no contact argument.
        contact = contact_batch(payloads)
        self.assertEqual(contact["sample_index"].shape, (2, 0))
        with_empty = self.step.energy(positions, target, ids, previous_positions=previous, contact=contact)
        for name in HexLossTerms._fields:
            torch.testing.assert_close(getattr(with_empty, name), getattr(terms, name), rtol=0, atol=0)
        inputs = self.step.prepare_inputs(positions, target, ids, previous_positions=previous, contact=contact)
        self.assertEqual(inputs.conditioning.shape, (2, self.cell_count, features.CONDITIONING_DIM))
        torch.testing.assert_close(inputs.conditioning[..., 4:], torch.zeros(2, self.cell_count, 3), rtol=0, atol=0)
        self.assertIsNone(inputs.contact_tokens)
        self.assertIsNone(inputs.contact_mask)
        output = self.step(
            positions,
            target,
            ids,
            previous_positions=previous,
            fixed_positions=positions[:, self.fixed],
            contact=contact,
        )
        torch.testing.assert_close(output.contact_energy, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(output.contact_max_penetration, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(output.loss.contact, torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(
            output.loss.total, output.loss.elastic + output.loss.inertia + output.loss.damping, rtol=0, atol=0
        )
        for name in ("contact_energy", "contact_max_penetration"):
            self.assertFalse(getattr(output, name).requires_grad, name)

    def test_prepare_detects_plane_pairs_within_search_band(self):
        """Find the bottom faces when the floor lies within r + margin of them and nothing when it is far."""
        threshold = 2 * self.radius
        self.step.register_context("near", **self.spec, contact=floor_partners(-(threshold - 0.02)))
        self.step.register_context("far", **self.spec, contact=floor_partners(-0.5))
        self.step.register_context("band", **self.spec, contact=floor_partners(-(threshold + 0.005)))
        near, far, band = self._payloads(self.step, ("near", "far", "band"))
        for payload in (near, far, band):
            for name in ("contact_sample_index", "contact_kind"):
                self.assertEqual(payload[name].dtype, torch.int64)
                self.assertEqual(payload[name].device.type, "cpu")
            for name in ("contact_partner_point", "contact_partner_normal", "contact_partner_radius"):
                self.assertEqual(payload[name].dtype, torch.float32)
        # Only the four -y faces (gap 0.08 m) lie inside the band of 2 r = 0.1 m; side faces sit at gap 0.13 m.
        self.assertEqual(near["contact_sample_index"].shape, (4,))
        self.assertTrue((near["contact_kind"] == KIND_PLANE).all())
        faces = self.step.face_samples.face_index[near["contact_sample_index"]]
        self.assertEqual(faces.tolist(), [2, 2, 2, 2])
        self.assertEqual(sorted(self.step.face_samples.cell_index[near["contact_sample_index"]].tolist()), [0, 1, 2, 3])
        torch.testing.assert_close(near["contact_partner_normal"], torch.tensor([[0.0, 1.0, 0.0]] * 4), rtol=0, atol=0)
        torch.testing.assert_close(
            near["contact_partner_point"][:, 1], torch.full((4,), -(threshold - 0.02)), rtol=0, atol=1e-6
        )
        self.assertEqual(far["contact_sample_index"].shape, (0,))
        self.assertEqual(band["contact_sample_index"].shape, (0,))
        # A downward step-start velocity widens the band by |v| dt and captures the same floor.
        velocity = torch.zeros_like(self.rest_positions)
        velocity[:, 1] = -1.0
        moving = self.step.prepare("band", self.rest_positions, velocity)
        self.assertEqual(moving["contact_sample_index"].shape, (4,))
        # advance() refreshes the pairs through prepare() on the committed candidate: sinking toward the far
        # floor turns the empty pair list into a populated one that matches a direct prepare() call.
        payload = dict(far)
        payload["candidate"] = payload["candidate"].clone()
        payload["candidate"][:, 1] -= 0.3
        advanced = self.step.advance(payload)
        expected_velocity = (payload["candidate"] - far["physical_positions"]) / self.dt
        expected_velocity[self.fixed] = 0
        expected = self.step.prepare("far", payload["candidate"], expected_velocity)
        self.assertGreater(advanced["contact_sample_index"].shape[0], 0)
        for name in ("contact_sample_index", "contact_kind", "contact_partner_point", "contact_partner_normal"):
            torch.testing.assert_close(advanced[name], expected[name], rtol=0, atol=0)
        torch.save(near, io.BytesIO())

    def test_prepare_drops_partners_that_do_not_oppose_the_face_normal(self):
        """Pass the step-start face normals to detection so grazing and back-facing partners pair with nothing.

        A floor 0.03 m below the (2, 1, 2) grid lies within the 2 r = 0.1 m
        band of the four bottom faces (gap 0.03) and of the eight side faces
        (centroids at y = 0.05, gap 0.08); only the bottom faces oppose its
        normal. A static disk beside the +x face whose normal points away from
        the body is a candidate of the two +x faces without the filter and of
        nothing with it; the same disk facing the body pairs with exactly those
        two faces.
        """
        beside = torch.tensor([[0.23, 0.05, 0.1]])
        self.step.register_context("floor", **self.spec, contact=floor_partners(-0.03))
        for name, normal in (("away", [[1.0, 0.0, 0.0]]), ("facing", [[-1.0, 0.0, 0.0]])):
            self.step.register_context(
                name,
                **self.spec,
                contact=ContactPartners(
                    plane_present=False,
                    plane_point=torch.zeros(3),
                    plane_normal=torch.tensor([0.0, 1.0, 0.0]),
                    point_positions=beside,
                    point_normals=torch.tensor(normal),
                    point_radii=torch.tensor([0.08]),
                    ke=500.0,
                    kd=0.0,
                    mu=0.0,
                ),
            )
        floor, away, facing = self._payloads(self.step, ("floor", "away", "facing"))
        samples = sample_points(self.rest_positions[None], self.step.face_samples.corners)[0]
        unfiltered = {
            name: detect_contacts(
                samples,
                torch.zeros_like(samples),
                self.step._contexts[name].contact,
                radius=self.radius,
                time_step=self.dt,
            )
            for name in ("floor", "away")
        }
        self.assertEqual(unfiltered["floor"].sample_index.shape, (12,))
        self.assertEqual(floor["contact_sample_index"].shape, (4,))
        self.assertEqual(self.step.face_samples.face_index[floor["contact_sample_index"]].tolist(), [2, 2, 2, 2])
        self.assertTrue((floor["contact_kind"] == KIND_PLANE).all())
        self.assertEqual(unfiltered["away"].sample_index.shape, (2,))
        self.assertEqual(away["contact_sample_index"].shape, (0,))
        self.assertEqual(facing["contact_sample_index"].shape, (2,))
        self.assertTrue((facing["contact_kind"] == KIND_POINT).all())
        self.assertEqual(self.step.face_samples.face_index[facing["contact_sample_index"]].tolist(), [1, 1])

    def test_penetrating_floor_raises_energy_and_pushes_bottom_corners_upward(self):
        """Add a positive contact energy for a penetrating floor whose force lifts the bottom corners."""
        self.step.register_context("free", **self.spec)
        self.step.register_context("floor", **self.spec, contact=floor_partners(-0.02))
        ids = ("free", "floor")
        payloads = self._payloads(self.step, ids)
        contact = contact_batch(payloads)
        # Only the four bottom faces oppose the floor normal; the eight side faces within the band are dropped.
        self.assertEqual(contact["sample_index"].shape[1], 4)
        self.assertEqual(contact["mask"].tolist(), [[False] * 4, [True] * 4])
        # Evaluate at the step start so the friction and damping anchors coincide with the positions.
        positions = self._stack(payloads, "physical_positions")
        target = self._stack(payloads, "inertial_prediction")
        previous = positions.clone()
        terms = self.step.energy(positions, target, ids, previous_positions=previous, contact=contact)
        self.assertEqual(terms.contact[0].item(), 0.0)
        self.assertGreater(terms.contact[1].item(), 0.0)
        torch.testing.assert_close(terms.total, terms.elastic + terms.inertia + terms.damping + terms.contact)
        torch.testing.assert_close(terms.elastic[0], terms.elastic[1])
        torch.testing.assert_close(terms.inertia[0], terms.inertia[1])
        self.assertGreater(terms.total[1].item(), terms.total[0].item())
        # Only the four bottom samples penetrate (depth d = r - gap = 0.03 m). At zero slip the IPC friction
        # smoothing still contributes mu * ke * d * eps_u / 3 per pair with eps_u = friction_epsilon * dt.
        depth, eps_u = 0.03, 1e-2 * self.dt
        expected = 4 * (0.5 * 500.0 * depth**2 + 0.3 * 500.0 * depth * eps_u / 3)
        self.assertAlmostEqual(terms.contact[1].item(), expected, places=6)
        candidate = positions.clone().requires_grad_(True)
        total = self.step.energy(candidate, target, ids, previous_positions=previous, contact=contact).contact.sum()
        gradient = torch.autograd.grad(total, candidate)[0]
        force = -gradient[1]
        self.assertTrue((force[self.bottom, 1] > 0).all())
        torch.testing.assert_close(force[~self.bottom], torch.zeros_like(force[~self.bottom]), rtol=0, atol=0)
        torch.testing.assert_close(force[:, [0, 2]], torch.zeros_like(force[:, [0, 2]]), rtol=0, atol=0)
        torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            self.step.energy(positions, target, ids, contact=contact)
        # The gradient feature of the network input includes the contact force.
        inputs = self.step.prepare_inputs(positions, target, ids, previous_positions=previous, contact=contact)
        plain = self.step.prepare_inputs(positions, target, ids, previous_positions=previous)
        torch.testing.assert_close(inputs.position_gradient[0], plain.position_gradient[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(inputs.position_gradient[1], plain.position_gradient[1]))
        torch.testing.assert_close(
            inputs.conditioning[1, 0, 4:],
            torch.tensor([math.log1p(500.0 / (700.0 * 0.1)), 1.0 / (500.0 * self.dt), 0.3]),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(inputs.conditioning[0, 0, 4:], torch.zeros(3), rtol=0, atol=0)

    def test_forward_with_contact_network_builds_tokens_and_reports_penetration(self):
        """Pass built tokens to a contact-token network and fill the contact energy and penetration outputs."""
        step = self._contact_step(contact_tokens_per_cell=5)
        step.register_context("free", **self.spec)
        step.register_context("floor", **self.spec, contact=floor_partners(-0.02))
        ids = ("free", "floor")
        payloads = self._payloads(step, ids)
        contact = contact_batch(payloads)
        positions = self._stack(payloads, "candidate")
        target = self._stack(payloads, "inertial_prediction")
        previous = self._stack(payloads, "physical_positions")
        inputs = step.prepare_inputs(positions, target, ids, previous_positions=previous, contact=contact)
        self.assertEqual(inputs.contact_tokens.shape, (2, self.cell_count, 5, features.CONTACT_TOKEN_DIM))
        self.assertEqual(inputs.contact_mask.shape, (2, self.cell_count, 5))
        self.assertFalse(inputs.contact_tokens.requires_grad)
        self.assertEqual(inputs.contact_mask[0].sum().item(), 0)
        # Every cell owns one floor pair, its -y face; its side faces within the band do not oppose the floor.
        self.assertEqual(inputs.contact_mask[1].sum(-1).tolist(), [1] * self.cell_count)
        kinds = inputs.contact_tokens[1][inputs.contact_mask[1]][:, 15:18]
        torch.testing.assert_close(kinds, torch.tensor([[1.0, 0.0, 0.0]]).expand(4, 3), rtol=0, atol=0)
        with patch.object(step.network, "forward", wraps=step.network.forward) as counted:
            output = step(
                positions,
                target,
                ids,
                previous_positions=previous,
                fixed_positions=positions[:, self.fixed],
                contact=contact,
            )
        self.assertEqual(counted.call_count, 1)
        passed = counted.call_args.kwargs
        torch.testing.assert_close(passed["contact_tokens"], inputs.contact_tokens, rtol=0, atol=0)
        self.assertEqual(passed["contact_mask"].tolist(), inputs.contact_mask.tolist())
        self.assertEqual(output.contact_energy.shape, (2,))
        self.assertEqual(output.contact_max_penetration.shape, (2,))
        torch.testing.assert_close(output.contact_energy, output.loss.contact.detach(), rtol=0, atol=0)
        self.assertEqual(output.contact_energy[0].item(), 0.0)
        self.assertGreater(output.contact_energy[1].item(), 0.0)
        self.assertEqual(output.contact_max_penetration[0].item(), 0.0)
        self.assertGreater(output.contact_max_penetration[1].item(), 0.0)
        self.assertLess(output.contact_max_penetration[1].item(), 1.0)
        # Penetration is measured at the fused positions in units of r.
        samples = step._sample_positions(output.positions.detach())
        bottom = samples[1, contact["sample_index"][1]]
        depth = torch.relu(self.radius - (bottom[:, 1] + 0.02)).max() / self.radius
        self.assertAlmostEqual(output.contact_max_penetration[1].item(), depth.item(), places=5)
        output.loss.total.sum().backward()
        for name, parameter in step.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        encoder = [p.grad.abs().sum().item() for n, p in step.network.named_parameters() if "contact_encoder" in n]
        self.assertGreater(sum(encoder), 0)
        # A contact-free batch keeps every encoder parameter in the graph (needed by DDP) with zero output.
        step.network.zero_grad()
        plain = step(
            positions[:1],
            target[:1],
            ids[:1],
            previous_positions=previous[:1],
            fixed_positions=positions[:1, self.fixed],
        )
        plain.loss.total.sum().backward()
        for name, parameter in step.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
        torch.testing.assert_close(plain.contact_energy, torch.zeros(1), rtol=0, atol=0)
        reference = self.step
        reference.register_context("free", **self.spec)
        baseline_inputs = reference.prepare_inputs(positions[:1], target[:1], ids[:1], previous_positions=previous[:1])
        torch.testing.assert_close(
            step.prepare_inputs(positions[:1], target[:1], ids[:1], previous_positions=previous[:1]).state_features,
            baseline_inputs.state_features,
            rtol=0,
            atol=0,
        )

    def test_assemble_inputs_passes_contact_tokens_through(self):
        """Return the supplied tokens and mask unchanged and refuse a half-supplied pair."""
        self.step.register_context("free", **self.spec)
        payload = self._payloads(self.step, ("free",))[0]
        positions = payload["candidate"][None]
        target = payload["inertial_prediction"][None]
        previous = payload["physical_positions"][None]
        tokens = torch.randn(1, self.cell_count, 3, features.CONTACT_TOKEN_DIM)
        mask = torch.rand(1, self.cell_count, 3) < 0.5
        contexts = self.step._lookup(("free",), 1)
        conditioning = torch.zeros(1, self.cell_count, features.CONDITIONING_DIM)
        # The assembly runs in the step's cell units with the shared unit fusion factor.
        h = self.rest.cell_size
        unit_positions, unit_target, unit_previous = positions / h, target / h, previous / h

        def energy_total(candidate, y, start):
            return self.step._unit_energy(candidate, y, contexts, start).total

        project_gradient = self.step._fusion.project_gradient
        inputs = assemble_inputs(
            self.step._unit_geometry(),
            unit_positions,
            unit_target,
            unit_previous,
            energy_total=energy_total,
            project_gradient=project_gradient,
            conditioning=conditioning,
            contact_tokens=tokens,
            contact_mask=mask,
        )
        self.assertIs(inputs.contact_tokens, tokens)
        self.assertIs(inputs.contact_mask, mask)
        with self.assertRaisesRegex(ValueError, "contact_tokens"):
            assemble_inputs(
                self.step._unit_geometry(),
                unit_positions,
                unit_target,
                unit_previous,
                energy_total=energy_total,
                project_gradient=project_gradient,
                conditioning=conditioning,
                contact_tokens=tokens,
            )

    def test_contact_batch_validation(self):
        """Reject malformed padded pair batches explicitly."""
        self.step.register_context("floor", **self.spec, contact=floor_partners(-0.02))
        payloads = self._payloads(self.step, ("floor",))
        contact = contact_batch(payloads)
        positions = self._stack(payloads, "candidate")
        target = self._stack(payloads, "inertial_prediction")
        previous = self._stack(payloads, "physical_positions")
        good = self.step.energy(positions, target, ("floor",), previous_positions=previous, contact=contact)
        self.assertGreater(good.contact.item(), 0.0)
        broken = [
            {key: value for key, value in contact.items() if key != "mask"},
            {**contact, "mask": contact["mask"].float()},
            {**contact, "partner_point": contact["partner_point"][:, :, :2]},
            {**contact, "sample_index": contact["sample_index"].float()},
            {**contact, "partner_radius": contact["partner_radius"].double()},
            {**contact, "sample_index": contact["sample_index"].repeat(2, 1)},
            {**contact, "partner_normal": contact["partner_normal"].clone().fill_(float("nan"))},
            {**contact, "sample_index": torch.full_like(contact["sample_index"], 99)},
            [contact],
        ]
        for index, bad in enumerate(broken):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.step.energy(positions, target, ("floor",), previous_positions=previous, contact=bad)
        # Masked rows may hold arbitrary values.
        poisoned = {name: value.clone() for name, value in contact.items()}
        poisoned["mask"][0, :6] = False
        poisoned["partner_point"][0, :6] = float("nan")
        poisoned["sample_index"][0, :6] = -1
        partial = self.step.energy(positions, target, ("floor",), previous_positions=previous, contact=poisoned)
        self.assertTrue(torch.isfinite(partial.contact).all())
        self.assertLess(partial.contact.item(), good.contact.item())


if __name__ == "__main__":
    unittest.main()
