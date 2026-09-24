# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the frozen Newton rigid proxy against analytical mass-point cases."""

import itertools
import unittest
from importlib.util import find_spec

if find_spec("torch") is None:
    raise unittest.SkipTest("Rigid predictor tests require the optional Torch dependency")

from experiments.learned_intrinsic_solver.rigid_predictor import RigidPosePredictor
from newton.solvers import SolverBase


class TestRigidPosePredictor(unittest.TestCase):
    def setUp(self):
        import torch

        self.positions = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=torch.float32)
        self.masses = torch.ones(8, dtype=torch.float32)
        self.zeros = torch.zeros_like(self.positions)

    def test_gravity_and_external_force_enter_once(self):
        """Recover semi-implicit translation from the physical mass and net force."""
        import torch

        predictor = RigidPosePredictor(self.masses, gravity=(0.0, 0.0, -10.0))
        positions = self.positions + torch.tensor([1.0, 2.0, 3.0])
        forces = torch.tensor([2.0, 0.0, 0.0]).expand_as(positions)
        prediction = predictor.predict(positions, self.zeros, forces, 0.1)
        self.assertIsInstance(predictor.solver, SolverBase)
        self.assertEqual(predictor.model.device.alias, "cpu")
        torch.testing.assert_close(prediction.predicted_linear_velocity, torch.tensor([0.2, 0.0, -1.0]))
        torch.testing.assert_close(prediction.predicted_position, torch.tensor([1.02, 2.0, 2.9]))
        torch.testing.assert_close(prediction.rigid_delta_rotation, torch.eye(3)[None])
        torch.testing.assert_close(
            prediction.rigid_delta_translation, torch.tensor([[0.02, 0.0, -0.1]]), atol=2e-7, rtol=1e-5
        )
        predictor.model.set_gravity((0.0, 0.0, 0.0))
        no_gravity = predictor.predict(positions, self.zeros, self.zeros, 0.1)
        torch.testing.assert_close(no_gravity.predicted_position, prediction.center_of_mass)

    def test_torque_uses_newton_quaternion_step(self):
        """Recover angular acceleration and Newton's normalized quaternion update."""
        import torch

        predictor = RigidPosePredictor(self.masses, gravity=(0.0, 0.0, 0.0))
        acceleration = torch.tensor([0.0, 0.0, 0.6])
        forces = torch.linalg.cross(acceleration.expand_as(self.positions), self.positions)
        prediction = predictor.predict(self.positions, self.zeros, forces, 0.2)
        torch.testing.assert_close(prediction.predicted_angular_velocity, torch.tensor([0.0, 0.0, 0.12]))
        tangent = 0.5 * 0.2 * 0.12
        cosine = (1 - tangent**2) / (1 + tangent**2)
        sine = 2 * tangent / (1 + tangent**2)
        expected = torch.tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(prediction.predicted_rotation, expected, atol=2e-7, rtol=1e-6)

    def test_recover_rigid_spin_from_corner_velocities(self):
        """Extract mass-center velocity and spin without treating deformation as another step."""
        import torch

        predictor = RigidPosePredictor(self.masses, gravity=(0.0, 0.0, 0.0))
        omega = torch.tensor([0.4, -0.6, 0.8])
        velocity = torch.tensor([0.3, -0.2, 0.1])
        velocities = velocity + torch.linalg.cross(omega.expand_as(self.positions), self.positions)
        prediction = predictor.predict(self.positions, velocities, self.zeros, 0.05)
        torch.testing.assert_close(prediction.center_of_mass_velocity, velocity)
        torch.testing.assert_close(prediction.angular_momentum, 16 * omega)
        torch.testing.assert_close(prediction.inertia, 16 * torch.eye(3))
        torch.testing.assert_close(prediction.predicted_angular_velocity, omega)
        torch.testing.assert_close(prediction.predicted_position, 0.05 * velocity)

    def test_unequal_masses_use_weighted_center_and_momentum(self):
        """Match hand-computed mass reductions instead of averaging corner samples."""
        import torch

        positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        velocities = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        predictor = RigidPosePredictor(torch.tensor([1.0, 2.0, 3.0, 4.0]), gravity=(0.0, 0.0, 0.0))
        prediction = predictor.predict(positions, velocities, torch.zeros_like(positions), 0.1)
        torch.testing.assert_close(prediction.center_of_mass, torch.tensor([0.4, 0.6, 0.8]))
        torch.testing.assert_close(prediction.center_of_mass_velocity, torch.tensor([0.5, 0.6, 0.7]))
        torch.testing.assert_close(prediction.angular_momentum, torch.tensor([-1.4, 6.8, 4.6]))
        torch.testing.assert_close(
            prediction.inertia, torch.tensor([[18.0, 2.4, 3.2], [2.4, 16.0, 4.8], [3.2, 4.8, 14.8]])
        )

    def test_newton_gyroscopic_response_for_nonprincipal_spin(self):
        """Retain the public integrator's torque-free gyroscopic acceleration."""
        import torch

        predictor = RigidPosePredictor(self.masses, gravity=(0.0, 0.0, 0.0))
        positions = self.positions * torch.tensor([2.0, 1.0, 1.0])
        omega = torch.tensor([1.0, 2.0, 3.0])
        velocities = torch.linalg.cross(omega.expand_as(positions), positions)
        prediction = predictor.predict(positions, velocities, self.zeros, 0.1)
        torch.testing.assert_close(prediction.predicted_angular_velocity, torch.tensor([1.0, 2.18, 2.88]))

    def test_refresh_deformed_inertia_at_fixed_angular_momentum(self):
        """Recompute the proxy tensor when the same object changes shape."""
        import torch

        predictor = RigidPosePredictor(self.masses, gravity=(0.0, 0.0, 0.0))
        omega_first = torch.tensor([0.0, 0.0, 0.5])
        velocity_first = torch.linalg.cross(omega_first.expand_as(self.positions), self.positions)
        first = predictor.predict(self.positions, velocity_first, self.zeros, 0.1)
        stretched = self.positions * torch.tensor([2.0, 1.0, 1.0])
        omega_second = torch.tensor([0.0, 0.0, 0.2])
        velocity_second = torch.linalg.cross(omega_second.expand_as(stretched), stretched)
        second = predictor.predict(stretched, velocity_second, self.zeros, 0.1)
        torch.testing.assert_close(first.angular_momentum, second.angular_momentum)
        torch.testing.assert_close(second.inertia, torch.diag(torch.tensor([16.0, 40.0, 40.0])))
        torch.testing.assert_close(first.predicted_angular_velocity, omega_first)
        torch.testing.assert_close(second.predicted_angular_velocity, omega_second)
        torch.testing.assert_close(torch.from_numpy(predictor.model.body_inertia.numpy()[0]), second.inertia)
        torch.testing.assert_close(
            torch.from_numpy(predictor.model.body_inv_inertia.numpy()[0]), torch.linalg.inv(second.inertia)
        )

    def test_impulses_are_applied_once_and_calls_do_not_accumulate(self):
        """Apply supplied impulses to a fresh proxy rather than integrating scratch history."""
        import torch

        predictor = RigidPosePredictor(self.masses, gravity=(0.0, 0.0, 0.0))
        arguments = {"linear_impulse": torch.tensor([8.0, 0.0, 0.0]), "angular_impulse": torch.tensor([0.0, 0.0, 16.0])}
        first = predictor.predict(self.positions, self.zeros, self.zeros, 0.1, **arguments)
        second = predictor.predict(self.positions, self.zeros, self.zeros, 0.1, **arguments)
        torch.testing.assert_close(first.predicted_linear_velocity, torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(first.predicted_angular_velocity, torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(first.predicted_position, torch.tensor([0.1, 0.0, 0.0]))
        torch.testing.assert_close(second.predicted_rotation, first.predicted_rotation)
        torch.testing.assert_close(second.predicted_position, first.predicted_position)
        cleared = predictor.predict(self.positions, self.zeros, self.zeros, 0.1)
        torch.testing.assert_close(cleared.rigid_delta_rotation, torch.eye(3)[None])
        torch.testing.assert_close(cleared.rigid_delta_translation, torch.zeros((1, 3)))

    def test_source_inputs_and_returned_snapshots_stay_unchanged(self):
        """Freeze predictor derivatives without mutating input or previously returned tensors."""
        import torch

        predictor = RigidPosePredictor(self.masses)
        positions = self.positions.clone().requires_grad_()
        velocities = self.zeros.clone().requires_grad_()
        forces = self.zeros.clone().requires_grad_()
        inputs = (positions, velocities, forces)
        copies = tuple(value.detach().clone() for value in inputs)
        prediction = predictor.predict(*inputs, 0.1)
        saved = tuple(value.clone() for value in prediction)
        predictor.predict(positions + 10, velocities + 1, forces + 2, 0.2)
        for value, original in zip(inputs, copies, strict=True):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
            self.assertIsNone(value.grad)
        for value, original in zip(prediction, saved, strict=True):
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_rigid_change_of_world_coordinates_transforms_the_prediction(self):
        """Rotate geometry, motion, loads, gravity, and impulses together."""
        import torch

        random = torch.Generator().manual_seed(73)
        rotation = torch.tensor([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
        translation = torch.tensor([2.0, -1.0, 0.7])
        positions = self.positions * torch.tensor([1.5, 0.7, 1.0]) + torch.tensor([0.2, 0.3, -0.1])
        velocities = torch.randn(self.positions.shape, generator=random)
        forces = torch.randn(self.positions.shape, generator=random)
        masses = torch.arange(1, 9, dtype=torch.float32)
        gravity = torch.tensor([0.7, -0.4, -9.6])
        impulse = torch.tensor([0.3, -0.5, 0.2])
        angular_impulse = torch.tensor([-0.2, 0.4, 0.6])
        first = RigidPosePredictor(masses, gravity=gravity).predict(
            positions, velocities, forces, 0.04, linear_impulse=impulse, angular_impulse=angular_impulse
        )
        transformed = RigidPosePredictor(masses, gravity=rotation @ gravity).predict(
            positions @ rotation.T + translation,
            velocities @ rotation.T,
            forces @ rotation.T,
            0.04,
            linear_impulse=rotation @ impulse,
            angular_impulse=rotation @ angular_impulse,
        )
        torch.testing.assert_close(transformed.inertia, rotation @ first.inertia @ rotation.T, rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(
            transformed.predicted_position, rotation @ first.predicted_position + translation, rtol=3e-6, atol=3e-6
        )
        torch.testing.assert_close(
            transformed.rigid_delta_rotation[0],
            rotation @ first.rigid_delta_rotation[0] @ rotation.T,
            rtol=3e-6,
            atol=3e-6,
        )
        first_points = positions @ first.rigid_delta_rotation[0].T + first.rigid_delta_translation
        second_points = (positions @ rotation.T + translation) @ transformed.rigid_delta_rotation[
            0
        ].T + transformed.rigid_delta_translation
        torch.testing.assert_close(second_points, first_points @ rotation.T + translation, rtol=3e-6, atol=3e-6)

    def test_reject_singular_geometry_and_invalid_inputs(self):
        """Reject undefined spin extraction and unsupported runtime precision."""
        import torch

        for masses in (torch.zeros(8), -self.masses, torch.tensor([float("nan")])):
            with self.subTest(masses=masses), self.assertRaises(ValueError):
                RigidPosePredictor(masses)
        predictor = RigidPosePredictor(self.masses)
        collinear = torch.zeros_like(self.positions)
        collinear[:, 0] = torch.arange(8)
        with self.assertRaises(ValueError):
            predictor.predict(collinear, self.zeros, self.zeros, 0.1)
        with self.assertRaises(TypeError):
            predictor.predict(self.positions.double(), self.zeros, self.zeros, 0.1)
        for dt in (0.0, -0.1, float("nan"), True):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                predictor.predict(self.positions, self.zeros, self.zeros, dt)
        with self.assertRaises(ValueError):
            predictor.predict(self.positions, self.zeros, self.zeros, 0.1, linear_impulse=torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
