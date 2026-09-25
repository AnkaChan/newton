# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check orientation acceptance beyond the energy's Gauss sample points."""

import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import hex_validity
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.multiscale import screen_geometry
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork


class TestHexFeasibility(unittest.TestCase):
    def setUp(self):
        """Create a corner inversion invisible to Gauss and five-tet checks."""
        self.rest = generate_cuboid((1, 1, 1))
        self.base = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)[None]
        self.proposal = torch.tensor(
            [
                [0, 0, 0],
                [0, -0.6, -0.01],
                [0, 1, 0],
                [0, 0.4, 0.5],
                [1, 0, 0],
                [1, -0.6, 0.1],
                [1, 1, 0],
                [1, 0.4, 0.6],
            ],
            dtype=torch.float32,
        )[None]

    def test_corner_inversion_is_shortened_before_energy_evaluation(self):
        """Reject a folded corner despite positive Gauss and auxiliary tet volumes."""
        guard = hex_validity.HexFeasibility(self.rest)
        energy = HexImplicitEulerLoss(self.rest, 1, 1, 1, 1)
        self.assertTrue(torch.isfinite(energy(self.proposal, self.base).total).all())
        screen = screen_geometry(self.rest, self.proposal[0].numpy())
        self.assertAlmostEqual(screen["min_sampled_jacobian"], -0.01, places=6)
        self.assertGreater(screen["min_tet_volume_ratio"], 0)
        accepted, scale = guard(self.base, self.proposal)
        self.assertEqual(scale.tolist(), [0.5])
        self.assertGreater(min(screen_geometry(self.rest, accepted[0].numpy()).values()), 0)
        torch.testing.assert_close(accepted[:, [0, 2, 4, 6]], self.base[:, [0, 2, 4, 6]], rtol=0, atol=0)

    def test_acceptance_is_per_instance_and_preserves_valid_proposals_exactly(self):
        """Keep a valid member bit exact while shortening only its invalid neighbor."""
        guard = hex_validity.HexFeasibility(self.rest)
        valid = self.base * 1.1
        proposal = torch.cat((valid, self.proposal))
        accepted, scale = guard(self.base.expand(2, -1, -1), proposal)
        self.assertEqual(scale.tolist(), [1.0, 0.5])
        self.assertTrue(torch.equal(accepted[0], valid[0]))

    def test_gradient_matches_finite_difference_on_fixed_acceptance_branch(self):
        """Differentiate the shortened displacement while freezing its discrete choice."""
        guard = hex_validity.HexFeasibility(self.rest)
        proposal = self.proposal.clone().requires_grad_()
        weights = torch.arange(24, dtype=torch.float32).reshape_as(proposal) / 24
        accepted, scale = guard(self.base, proposal)
        (accepted * weights).sum().backward()
        direction = torch.zeros_like(proposal)
        direction[0, 1, 2] = 1
        epsilon = 1e-3
        values = []
        for sign in (-1, 1):
            moved, moved_scale = guard(self.base, proposal.detach() + sign * epsilon * direction)
            self.assertTrue(torch.equal(moved_scale, scale))
            values.append(float((moved * weights).sum()))
        numerical = (values[1] - values[0]) / (2 * epsilon)
        self.assertAlmostEqual(float((proposal.grad * direction).sum()), numerical, delta=4e-4)
        self.assertFalse(scale.requires_grad)

    def test_bad_base_and_nonfinite_proposals_fail_explicitly(self):
        """Reject an invalid starting shape and nonfinite updates without freezing them."""
        guard = hex_validity.HexFeasibility(self.rest)
        with self.assertRaisesRegex(ValueError, "base"):
            guard(self.proposal, self.base)
        nonfinite = self.base.clone()
        nonfinite[0, 1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            guard(self.base, nonfinite)

    def test_float32_positive_roundoff_does_not_accept_a_folded_corner(self):
        """Shorten a corner whose tiny float32 determinant has the wrong sign."""
        guard = hex_validity.HexFeasibility(self.rest)
        proposal = self.base.clone()
        proposal[0, 0] = torch.tensor([0.5928649306297302, 0.11281482130289078, 0.2943202555179596])
        self.assertLess(screen_geometry(self.rest, proposal[0].numpy())["min_sampled_jacobian"], 0)
        accepted, scale = guard(self.base, proposal)
        self.assertEqual(scale.tolist(), [0.5])
        self.assertGreater(min(screen_geometry(self.rest, accepted[0].numpy()).values()), 0)

    def test_resolved_compression_remains_unshortened(self):
        """Preserve positive compressed shapes whose orientation clears roundoff."""
        guard = hex_validity.HexFeasibility(self.rest)
        proposal = self.base.clone()
        proposal[..., 2] *= 0.001
        accepted, scale = guard(self.base, proposal)
        self.assertEqual(scale.tolist(), [1.0])
        self.assertTrue(torch.equal(accepted, proposal))

    def test_unresolved_search_fails_and_singular_center_is_rejected(self):
        """Stop a bounded search explicitly and reject float32-singular positive cells."""
        guard = hex_validity.HexFeasibility(self.rest)
        huge = self.base.clone()
        huge[..., 2] *= -1e15
        with self.assertRaisesRegex(ValueError, "halvings"):
            guard(self.base, huge)
        thin = self.base.clone()
        thin[..., 2] *= 1e-8
        with self.assertRaisesRegex(ValueError, "base"):
            guard(thin, self.base)

    def test_screen_agrees_with_reference_on_non_cubic_grid_dimensions(self):
        """Use shared screen topology for both cell parity patterns across a grid."""
        rest = generate_cuboid((2, 1, 3), cell_size=0.1)
        guard = hex_validity.HexFeasibility(rest)
        x = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        rng = np.random.default_rng(137)
        for amplitude in (0.01, 0.05, 0.2):
            proposal = x + torch.tensor(rng.normal(size=x.shape), dtype=x.dtype) * amplitude
            expected = min(screen_geometry(rest, proposal[0].numpy()).values()) > 0
            self.assertEqual(bool(guard.valid(proposal)[0]), expected)

    def test_mixed_step_backtracks_before_log_energy_and_keeps_fusion_gradient(self):
        """Shorten a real learned inversion before log energy and differentiate fusion."""
        network = IntrinsicSolverNetwork((1, 1, 1), 38, hidden_dim=16, edge_hidden_dim=8, max_step_size=4)
        with torch.no_grad():
            network.correction_head.bias[-1] = -10
        fixed = np.array([0, 2, 4, 6])
        step = MixedHexSolverStep(self.rest, fixed, network=network, time_step=0.01, geometry_backtracking=True)
        self.addCleanup(step.close)
        step.register_context("fixture", lame_lambda=1, lame_mu=1, density=1)
        result = step(self.base, self.base, ("fixture",))
        self.assertEqual(result.acceptance_scale.tolist(), [0.5])
        self.assertGreater(min(screen_geometry(self.rest, result.positions[0].detach().numpy()).values()), 0)
        self.assertTrue(torch.equal(result.positions[:, fixed], self.base[:, fixed]))
        result.loss.total.sum().backward()
        derivative = float(network.correction_head.bias.grad[-1])
        self.assertTrue(np.isfinite(derivative))
        values = []
        for bias in (-10.01, -9.99):
            with torch.no_grad():
                network.correction_head.bias[-1] = bias
                moved = step(self.base, self.base, ("fixture",))
            self.assertEqual(moved.acceptance_scale.tolist(), [0.5])
            values.append(float(moved.loss.total[0]))
        numerical = (values[1] - values[0]) / 0.02
        self.assertAlmostEqual(derivative, numerical, delta=max(abs(numerical) * 0.03, 0.01))


if __name__ == "__main__":
    unittest.main()
