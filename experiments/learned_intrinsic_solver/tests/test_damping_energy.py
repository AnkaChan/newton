# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the physical VBD metric damping potential and shared features."""

import importlib.util
import unittest

import numpy as np
import warp as wp

from experiments.learned_intrinsic_solver.data import generate_cuboid

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

from experiments.learned_intrinsic_solver.damping import damping_metric_difference, pack_damping_features
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_volumetric_neo_hookean_force_and_hessian


@wp.kernel
def _tet_damping_force_oracle(
    positions: wp.array[wp.vec3],
    previous_positions: wp.array[wp.vec3],
    cells: wp.array2d[wp.int32],
    inverse_rest: wp.mat33,
    forces: wp.array[wp.vec3],
):
    vertex = wp.tid()
    damped, _ = evaluate_volumetric_neo_hookean_force_and_hessian(
        0, vertex, previous_positions, positions, cells, inverse_rest, 30.0, 40.0, 3.0, 0.02
    )
    undamped, _ = evaluate_volumetric_neo_hookean_force_and_hessian(
        0, vertex, previous_positions, positions, cells, inverse_rest, 30.0, 40.0, 0.0, 0.02
    )
    forces[vertex] = damped - undamped


class TestDampingEnergy(unittest.TestCase):
    def _fixture(self, *, damping=3.0, time_step=0.02, counts=(1, 1, 1)):
        import torch

        rest = generate_cuboid(counts, cell_size=0.4)
        loss = HexImplicitEulerLoss(rest, 500.0, 300.0, 1000.0, time_step, damping=damping, dtype=torch.float64)
        anchor = torch.tensor(rest.corner_rest_positions[None], dtype=torch.float64)
        deformation = torch.tensor([[1.1, 0.2, 0.0], [0.0, 0.95, 0.1], [0.0, 0.0, 1.04]], dtype=torch.float64)
        return rest, loss, anchor, anchor @ deformation.T

    def test_affine_energy_and_analytic_gradient(self):
        """Match metric viscosity with doubled shear entries and its spatial force."""
        import torch

        _, loss, anchor, positions = self._fixture()
        positions.requires_grad_()
        terms = loss(positions, positions.detach(), previous_positions=anchor)
        deformation = np.array([[1.1, 0.2, 0.0], [0.0, 0.95, 0.1], [0.0, 0.0, 1.04]])
        metric_change = deformation.T @ deformation - np.eye(3)
        expected_energy = 0.4**3 * 3.0 / (2 * 0.02) * np.sum(metric_change**2)
        self.assertAlmostEqual(terms.damping.item(), expected_energy, places=12)
        # Integral of each cube shape gradient is its corner sign times h^2/4.
        signs = 2 * np.indices((2, 2, 2)).reshape(3, -1).T - 1
        integrated_gradients = signs * 0.4**2 / 4
        expected_gradient = (2 * 3.0 / 0.02 * deformation @ metric_change @ integrated_gradients.T).T
        gradient = torch.autograd.grad(terms.damping.sum(), positions)[0]
        np.testing.assert_allclose(gradient.numpy()[0], expected_gradient, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(terms.total, terms.elastic + terms.inertia + terms.damping)

    def test_nonaffine_position_and_anchor_gradcheck(self):
        """Preserve both position derivatives through nonaffine Gauss metrics."""
        import torch

        _, loss, anchor, positions = self._fixture()
        anchor[:, -1] += torch.tensor([0.006, -0.003, 0.004])
        positions[:, -1] += torch.tensor([-0.004, 0.007, 0.003])
        anchor.requires_grad_()
        positions.requires_grad_()
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda current, previous: loss(current, current.detach(), previous_positions=previous).damping,
                (positions, anchor),
                eps=1e-6,
                atol=2e-6,
                rtol=2e-5,
            )
        )
        previous_gradient = torch.autograd.grad(
            loss(positions, positions.detach(), previous_positions=anchor).damping.sum(), anchor
        )[0]
        self.assertGreater(previous_gradient.norm().item(), 1.0)

    def test_finite_rigid_motion_of_deformed_anchor(self):
        """Leave a deformed material undamped under finite rigid rotation and translation."""
        import torch

        _, loss, _, anchor = self._fixture()
        anchor[:, -1] += torch.tensor([0.005, 0.006, -0.004])
        angle = 1.1
        rotation = torch.tensor(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=torch.float64
        )
        positions = (anchor @ rotation.T + torch.tensor([2.0, -1.0, 0.5])).requires_grad_()
        damping = loss(positions, positions.detach(), previous_positions=anchor).damping
        self.assertLess(damping.item(), 1e-24)
        self.assertLess(torch.autograd.grad(damping.sum(), positions)[0].abs().max().item(), 1e-11)

    def test_zero_damping_preserves_legacy_energy_and_gradients(self):
        """Preserve exact undamped values without requiring an anchor."""
        import torch

        rest, zero, anchor, positions = self._fixture(damping=0)
        legacy = HexImplicitEulerLoss(rest, 500.0, 300.0, 1000.0, 0.02, dtype=torch.float64)
        positions.requires_grad_()
        actual = zero(positions, anchor)
        expected = legacy(positions, anchor)
        torch.testing.assert_close(actual.total, actual.elastic + actual.inertia, rtol=0, atol=0)
        torch.testing.assert_close(actual.damping, torch.zeros(1, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(actual.total, expected.total, rtol=0, atol=0)
        gradients = [
            torch.autograd.grad(terms.total.sum(), positions, retain_graph=True)[0] for terms in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, rtol=0, atol=0)

    def test_physical_anchor_is_required_and_changes_force(self):
        """Distinguish the physical anchor from the inertial target and candidate."""
        import torch

        _, loss, anchor, positions = self._fixture()
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            loss(positions, anchor)
        positions.requires_grad_()
        physical = loss(positions, positions.detach(), previous_positions=anchor).damping
        unchanged = loss(positions, positions.detach(), previous_positions=positions.detach()).damping
        self.assertGreater(physical.item(), 0.1)
        self.assertEqual(unchanged.item(), 0.0)
        self.assertGreater(torch.autograd.grad(physical.sum(), positions)[0].norm().item(), 1.0)

    def test_inverse_timestep_and_per_cell_coefficients(self):
        """Scale independently authored cell viscosities by inverse physical timestep."""
        import torch

        _, loss, anchor, positions = self._fixture(damping=[0.0, 6.0], counts=(2, 1, 1))
        _, uniform, _, _ = self._fixture(damping=3.0, counts=(2, 1, 1))
        _, slower, _, _ = self._fixture(damping=[0.0, 6.0], time_step=0.04, counts=(2, 1, 1))
        actual = loss(positions, positions, previous_positions=anchor).damping
        torch.testing.assert_close(actual, uniform(positions, positions, previous_positions=anchor).damping)
        torch.testing.assert_close(actual, 2 * slower(positions, positions, previous_positions=anchor).damping)
        # Deform only the outer face of the zero-viscosity cell.
        positions = anchor.clone()
        positions[:, positions[0, :, 0] == 0, 0] -= 0.02
        self.assertLess(loss(positions, positions, previous_positions=anchor).damping.item(), 1e-25)

    def test_validate_coefficients_and_anchor(self):
        """Reject malformed viscosity and anchor data instead of broadcasting it."""
        for damping in (-1, np.inf, np.nan, [1, 2]):
            with self.subTest(damping=damping), self.assertRaises(ValueError):
                self._fixture(damping=damping)
        _, loss, anchor, positions = self._fixture()
        for invalid in (anchor[:, :-1], anchor.expand(2, -1, -1), anchor * float("nan")):
            with self.assertRaises(ValueError):
                loss(positions, positions, previous_positions=invalid)
        with self.assertRaises(TypeError):
            loss(positions, positions, previous_positions=anchor.float())

    def test_legacy_checkpoint_requires_zero_configured_damping(self):
        """Load legacy energy weights strictly only into an undamped configuration."""
        _, zero, _, _ = self._fixture(damping=0)
        legacy = {key: value for key, value in zero.state_dict().items() if key != "damping"}
        self.assertEqual(zero.load_state_dict(legacy, strict=True).missing_keys, [])
        _, positive, _, _ = self._fixture()
        for strict in (True, False):
            with self.assertRaisesRegex(RuntimeError, "damping"):
                positive.load_state_dict(legacy, strict=strict)
        zero.load_state_dict(positive.state_dict(), strict=True)
        _, _, anchor, positions = self._fixture()
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            zero(positions, anchor)

    def test_metric_features_keep_gauss_order_and_frobenius_norm(self):
        """Pack six metric channels per Gauss point without losing shear norm."""
        import torch

        _, loss, anchor, positions = self._fixture()
        difference = damping_metric_difference(positions, anchor, loss.cell_corner_indices, loss.shape_gradients)
        packed = pack_damping_features(difference)
        expected = torch.tensor([0.21, -0.0575, 0.0916, 0.22, 0.0, 0.095], dtype=torch.float64)
        expected[3:] *= np.sqrt(2)
        torch.testing.assert_close(packed.reshape(1, 1, 8, 6), expected.expand(1, 1, 8, 6), atol=1e-14, rtol=1e-12)
        torch.testing.assert_close(packed.square().sum(-1), difference.square().sum((-1, -2, -3)))
        nonaffine = positions.clone()
        nonaffine[:, -1] += torch.tensor([0.004, -0.008, 0.01])
        varied = damping_metric_difference(nonaffine, anchor, loss.cell_corner_indices, loss.shape_gradients)
        self.assertGreater(varied.var(dim=2).sum().item(), 1e-4)

    def test_native_vbd_tet_force_matches_hex_metric_stress(self):
        """Match native VBD damping forces after rest-volume and shape-gradient assembly."""
        import torch

        rest, loss, _, _ = self._fixture()
        deformation = torch.tensor(
            [[1.1, 0.2, 0.0], [0.0, 0.95, 0.1], [0.0, 0.0, 1.04]], dtype=torch.float64, requires_grad=True
        )
        previous_deformation = torch.tensor(
            [[0.97, 0.04, 0.0], [0.02, 1.02, 0.01], [0.01, 0.0, 0.98]], dtype=torch.float64
        )
        hex_rest = torch.tensor(rest.corner_rest_positions[None], dtype=torch.float64)
        positions = hex_rest @ deformation.T
        previous = hex_rest @ previous_deformation.T
        energy = loss(positions, positions.detach(), previous_positions=previous).damping.sum()
        integrated_stress = torch.autograd.grad(energy, deformation)[0].numpy()
        rest_edges = np.array([[0.4, 0.05, 0], [0, 0.35, 0.02], [0, 0, 0.45]])
        inverse_rest = np.linalg.inv(rest_edges)
        tet_rest = np.vstack((np.zeros((1, 3)), rest_edges.T))
        gradients = np.vstack((-inverse_rest.sum(axis=0), inverse_rest))
        tet_volume = np.linalg.det(rest_edges) / 6
        expected = -gradients @ integrated_stress.T * (tet_volume / 0.4**3)
        forces = wp.zeros(4, dtype=wp.vec3, device="cpu")
        wp.launch(
            _tet_damping_force_oracle,
            dim=4,
            inputs=[
                wp.array(tet_rest @ deformation.detach().numpy().T, dtype=wp.vec3, device="cpu"),
                wp.array(tet_rest @ previous_deformation.numpy().T, dtype=wp.vec3, device="cpu"),
                wp.array([[0, 1, 2, 3]], dtype=wp.int32, ndim=2, device="cpu"),
                wp.mat33(inverse_rest.astype(np.float32)),
            ],
            outputs=[forces],
            device="cpu",
        )
        np.testing.assert_allclose(forces.numpy(), expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
