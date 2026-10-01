# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of prescribed far-face motion in the mixed step and of the learned FEM accuracy driver."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features, train_mixed
from experiments.learned_intrinsic_solver import fem_accuracy_learned as learned
from experiments.learned_intrinsic_solver import fem_accuracy_scenarios as scenarios
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.frames import select_reference_corners
from experiments.learned_intrinsic_solver.history import empty_history
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork

MATERIAL = {"lame_lambda": 700 * 0.2 / (1.2 * 0.6), "lame_mu": 700 / 2.4, "density": 90.0}


def make_network(cell_counts, *, target_modes=3):
    """Build a small revised-schema network with nonzero heads so proposals move free corners."""
    network = IntrinsicSolverNetwork(
        cell_counts,
        features.state_feature_dim(target_modes),
        target_modes=target_modes,
        conditioning_dim=features.CONDITIONING_DIM,
        hidden_dim=16,
        edge_hidden_dim=8,
    )
    with torch.no_grad():
        network.correction_head.weight.normal_(std=0.003)
        network.step_head.weight.normal_(std=0.01)
        for layer in network.layers:
            layer.film.weight.normal_(std=0.01)
    return network


class TestPrescribedFarFace(unittest.TestCase):
    def setUp(self):
        """Pin the clamp first and then the far face of a tiny grid, as the driver does."""
        torch.manual_seed(7)
        self.rest = generate_cuboid((2, 1, 3), cell_size=0.1)
        self.rest_positions = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)
        self.clamp = scenarios.clamp_indices(self.rest)
        self.far = scenarios.far_face_indices(self.rest)
        self.fixed = np.concatenate([self.clamp, self.far])
        self.free = np.setdiff1d(np.arange(len(self.rest_positions)), self.fixed)
        self.dt = 0.01
        self.network = make_network(self.rest.cell_counts)
        self.step = MixedHexSolverStep(
            self.rest, self.fixed, network=self.network, time_step=self.dt, gravity=(0.0, -9.81, 0.0)
        )
        self.addCleanup(self.step.close)
        self.step.register_context("soft", **MATERIAL)
        self.velocity = torch.zeros_like(self.rest_positions)
        self.velocity[:, 2] = 0.2 * self.rest_positions[:, 2]
        self.translated = self.rest_positions[self.fixed].clone()
        self.translated[len(self.clamp) :, 2] += 0.02

    def test_prepare_fuses_candidate_and_inertia_onto_prescribed_positions(self):
        """Move the far-face rows of the candidate and the inertial prediction exactly; leave the rest as before."""
        default = self.step.prepare("soft", self.rest_positions, self.velocity)
        payload = self.step.prepare("soft", self.rest_positions, self.velocity, fixed_positions=self.translated)
        torch.testing.assert_close(payload["fixed_positions"], self.translated, rtol=0, atol=0)
        torch.testing.assert_close(payload["candidate"][self.fixed], self.translated, rtol=0, atol=0)
        torch.testing.assert_close(payload["inertial_prediction"][self.fixed], self.translated, rtol=0, atol=0)
        torch.testing.assert_close(
            payload["inertial_prediction"][self.free], default["inertial_prediction"][self.free], rtol=0, atol=0
        )
        torch.testing.assert_close(payload["physical_positions"], self.rest_positions, rtol=0, atol=0)
        # The default keeps the prescribed corners at their input positions.
        torch.testing.assert_close(default["fixed_positions"], self.rest_positions[self.fixed], rtol=0, atol=0)
        torch.testing.assert_close(default["candidate"][self.fixed], self.rest_positions[self.fixed], rtol=0, atol=0)
        # Free corners of the fused candidate follow the stretched far face instead of staying at rest.
        self.assertGreater((payload["candidate"][self.free] - default["candidate"][self.free]).abs().max().item(), 1e-4)
        with self.assertRaisesRegex(ValueError, "fixed_positions"):
            self.step.prepare("soft", self.rest_positions, self.velocity, fixed_positions=self.translated[:2])

    def test_learned_update_moves_far_face_exactly_while_clamp_stays(self):
        """Run the trainer's checked forward on the prescribed payload and read the prescribed rows back exactly."""
        payload = self.step.prepare("soft", self.rest_positions, self.velocity, fixed_positions=self.translated)
        payload.update(empty_history(len(self.rest.cell_corner_indices), self.step.target_modes))
        learned._inertial_candidate(payload, self.step.fixed_indices)
        torch.testing.assert_close(payload["candidate"][self.fixed], self.translated, rtol=0, atol=0)
        batch = train_mixed._batch(
            [payload], torch.device("cpu"), cell_count=len(self.rest.cell_corner_indices), modes=self.step.target_modes
        )
        with torch.no_grad():
            result = train_mixed._checked_forward(self.step, self.step, batch)
        positions = result.positions[0]
        torch.testing.assert_close(positions[self.far], self.rest_positions[self.far] + torch.tensor([0.0, 0.0, 0.02]))
        torch.testing.assert_close(positions[self.clamp], self.rest_positions[self.clamp], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(positions).all())
        self.assertGreater((positions[self.free] - self.rest_positions[self.free]).abs().max().item(), 1e-4)

    def test_advance_keeps_prescribed_velocity_only_when_driven(self):
        """Keep the far-face finite-difference velocity for a driven next step; zero it in the default path."""
        payload = self.step.prepare("soft", self.rest_positions, self.velocity, fixed_positions=self.translated)
        learned._inertial_candidate(payload, self.step.fixed_indices)
        next_positions = self.translated.clone()
        next_positions[len(self.clamp) :, 2] += 0.02
        driven = self.step.advance(payload, fixed_positions=next_positions)
        torch.testing.assert_close(driven["physical_positions"][self.fixed], self.translated, rtol=0, atol=0)
        torch.testing.assert_close(driven["fixed_positions"], next_positions, rtol=0, atol=0)
        expected = torch.zeros(len(self.far), 3)
        expected[:, 2] = 0.02 / self.dt
        torch.testing.assert_close(driven["velocities"][self.far], expected)
        torch.testing.assert_close(driven["velocities"][self.clamp], torch.zeros(len(self.clamp), 3), rtol=0, atol=0)
        torch.testing.assert_close(driven["inertial_prediction"][self.fixed], next_positions, rtol=0, atol=0)
        held = self.step.advance(payload)
        torch.testing.assert_close(held["velocities"][self.fixed], torch.zeros(len(self.fixed), 3), rtol=0, atol=0)
        torch.testing.assert_close(held["fixed_positions"], self.translated, rtol=0, atol=0)

    def test_reference_corners_override_and_validation(self):
        """Keep the tie-break reference on the clamp when asked; reject IDs outside the prescribed set."""
        geometric = select_reference_corners(self.rest.corner_rest_positions, self.fixed)
        clamp_reference = select_reference_corners(self.rest.corner_rest_positions, self.clamp)
        self.assertTrue(np.isin(geometric, self.far).any())
        np.testing.assert_array_equal(self.step.reference_corners.numpy(), geometric)
        pinned = MixedHexSolverStep(
            self.rest,
            self.fixed,
            network=self.network,
            time_step=self.dt,
            reference_corners=clamp_reference,
        )
        self.addCleanup(pinned.close)
        np.testing.assert_array_equal(pinned.reference_corners.numpy(), clamp_reference)
        free_corner = int(self.free[0])
        for bad in ([free_corner, *clamp_reference[:2]], clamp_reference[:2], [clamp_reference[0]] * 3):
            with self.assertRaisesRegex(ValueError, "reference_corners"):
                MixedHexSolverStep(
                    self.rest, self.fixed, network=self.network, time_step=self.dt, reference_corners=bad
                )


class TestRunScenario(unittest.TestCase):
    def test_compression_release_on_tiny_grid_writes_layout_and_releases(self):
        """Drive, hold and release the far face of a tiny grid with one learned iteration per substep."""
        torch.manual_seed(11)
        rest = generate_cuboid((2, 1, 3), cell_size=0.1)
        scenario = scenarios.Scenario(
            name="compression_release",
            gravity=(0.0, 0.0, 0.0),
            frame_count=3,
            ramp_frames=1,
            hold_frames=1,
            motion="translation",
            amplitude=-0.03,
        )
        network = make_network(rest.cell_counts, target_modes=7)
        settings = learned.StepSettings(target_modes=7)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "compression_release"
            metrics = learned.run_scenario(
                scenario,
                rest=rest,
                network=network,
                settings=settings,
                device="cpu",
                iterations=1,
                output_dir=output,
                run_info={"checkpoint": "none"},
                progress_interval=1000,
            )
            trajectory = scenarios.read_trajectory(output / "trajectory.npz")
            run = json.loads((output / "run.json").read_text())
            written = scenarios.read_metrics(output / "metrics.json")
        self.assertEqual(metrics["scenario"], "compression_release")
        self.assertTrue(metrics["completed"])
        self.assertIsNone(metrics["failure"])
        self.assertEqual(written["frame_count"], 3)
        self.assertEqual(trajectory.positions.shape, (4, len(rest.corner_rest_positions), 3))
        np.testing.assert_array_equal(trajectory.fixed_indices, scenarios.clamp_indices(rest))
        far = scenarios.far_face_indices(rest)
        rest_far = rest.corner_rest_positions[far]
        # Frame 1 ends the ramp and frame 2 holds: the far face sits exactly on the schedule.
        for frame in (1, 2):
            np.testing.assert_allclose(
                trajectory.positions[frame, far], rest_far + np.array([0.0, 0.0, -0.03]), rtol=0, atol=1e-6
            )
        # After the release the far face is free and springs back from the compressed position.
        self.assertGreater(trajectory.positions[3, far, 2].mean(), trajectory.positions[2, far, 2].mean())
        self.assertEqual(run["release_substep"], 2 * scenarios.SUBSTEPS)
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["learned_calls"], 3 * scenarios.SUBSTEPS)
        self.assertEqual(len(run["diagnostics"]["frame_min_centre_jacobian_ratio"]), 4)
        self.assertEqual(run["checkpoint"], "none")
        self.assertEqual(metrics["release_frame"], 2)
        self.assertLess(metrics["min_centre_jacobian_ratio_compression"], 1.0)


if __name__ == "__main__":
    unittest.main()
