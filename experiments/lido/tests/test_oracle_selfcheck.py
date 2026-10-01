# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-check of the black-box oracle over the old learned hex solver (CPU, float64, 2x2x3 grid)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.hex_energy import HexImplicitEulerLoss
from experiments.lido.tests import oracle

CELL_COUNTS = (2, 2, 3)
H = 0.025
DT = 1.0 / 300.0
E = 1e5
NU = 0.3
RHO = 1000.0
ETA = 100.0
MATERIAL = (E, NU, RHO, ETA)


def _states(seed: int, scale: float = 0.05):
    """Rest grid plus three random small perturbations (pinned rows at rest)."""
    rng = np.random.default_rng(seed)
    rest, cells, pinned = oracle.rest_grid(CELL_COUNTS, H)

    def perturb():
        delta = rng.normal(size=rest.shape) * scale * H
        delta[pinned] = 0.0
        return rest + delta

    return rest, cells, pinned, perturb(), perturb(), perturb(), rng


class OracleSelfCheck(unittest.TestCase):
    def test_conventions(self):
        conventions = oracle.conventions()
        corners = conventions["corner_order"]
        self.assertEqual(len(corners), 8)
        self.assertEqual(len({tuple(c) for c in corners}), 8)
        self.assertTrue(all(set(c) <= {-1, 1} for c in corners))
        # z fastest: consecutive corners differ in xi3 first.
        self.assertEqual(corners[0], (-1, -1, -1))
        self.assertEqual(corners[1], (-1, -1, 1))
        self.assertEqual(corners[4], (1, -1, -1))
        points = np.asarray(conventions["gauss_points"])
        self.assertEqual(points.shape, (8, 3))
        np.testing.assert_allclose(points, np.asarray(corners) / np.sqrt(3.0))
        self.assertEqual(len(conventions["mode_order"]), 7)
        self.assertTrue(conventions["mode_scaled_by_two_over_h"])
        self.assertEqual(len(conventions["reference_corners_canonical_10x10x40"]), 3)
        self.assertEqual(conventions["reference_corners_2x2x3"], list(oracle.reference_corners(CELL_COUNTS, H)))
        for name, width in (("node_feature_layout", 159), ("edge_feature_layout", 24), ("contact_token_layout", 19)):
            self.assertEqual(sum(w for _, w in conventions[name]), width, name)

    def test_rest_grid(self):
        rest, cells, pinned = oracle.rest_grid(CELL_COUNTS, H)
        self.assertEqual(rest.shape, (36, 3))
        self.assertEqual(cells.shape, (12, 8))
        self.assertEqual(pinned.shape, (36,))
        self.assertEqual(int(pinned.sum()), 9)
        self.assertTrue(np.all(rest[pinned, 2] == 0.0))
        self.assertTrue(np.all(rest[~pinned, 2] > 0.0))
        # Local corner k = 4x + 2y + z sits at rest offset h (x, y, z) from corner 0 of its cell.
        offsets = rest[cells] - rest[cells[:, :1]]
        expected = H * (np.asarray(oracle.conventions()["corner_order"]) + 1) / 2
        np.testing.assert_allclose(offsets, np.broadcast_to(expected, offsets.shape), atol=1e-15)
        # Corner numbering: z fastest.
        np.testing.assert_allclose(rest[1] - rest[0], (0.0, 0.0, H))

    def test_energy_terms_and_rest_state(self):
        rest, _, _, X, Y, X_prev, _ = _states(1)
        zero = oracle.energy_terms(CELL_COUNTS, H, DT, rest, rest, rest, *MATERIAL)
        self.assertAlmostEqual(zero["total"], 0.0, delta=1e-24)  # float rounding of 1 / (4 h); energy unit is 0.6 J
        terms = oracle.energy_terms(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        for name in ("elastic", "inertia", "damping"):
            self.assertGreater(terms[name], 0.0)
        self.assertAlmostEqual(terms["total"], terms["elastic"] + terms["inertia"] + terms["damping"], places=14)
        energy, _ = oracle.energy_and_grad(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        self.assertEqual(energy, terms["total"])

    def test_gradient_matches_finite_difference(self):
        _, _, _, X, Y, X_prev, rng = _states(2)
        energy, gradient = oracle.energy_and_grad(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        self.assertTrue(np.isfinite(energy))
        self.assertEqual(gradient.shape, X.shape)
        self.assertTrue(np.isfinite(gradient).all())
        step = 1e-7
        for _ in range(4):
            direction = rng.normal(size=X.shape)
            direction /= np.linalg.norm(direction)
            plus = oracle.energy_and_grad(CELL_COUNTS, H, DT, X + step * direction, Y, X_prev, *MATERIAL)[0]
            minus = oracle.energy_and_grad(CELL_COUNTS, H, DT, X - step * direction, Y, X_prev, *MATERIAL)[0]
            finite = (plus - minus) / (2 * step)
            analytic = float(np.sum(gradient * direction))
            self.assertAlmostEqual(finite / analytic, 1.0, delta=1e-5)

    def test_normalised_units_reproduce_the_si_objective(self):
        """The old training step's cell-unit objective (h = mu = dt = 1) equals mu h^3 times the SI one."""
        _, _, _, X, Y, X_prev, _ = _states(3)
        lam, mu = oracle.lame(E, NU)
        unit_rest = generate_cuboid(CELL_COUNTS, cell_size=1.0)
        normalised = HexImplicitEulerLoss(
            unit_rest, lam / mu, 1.0, RHO * H**2 / (mu * DT**2), 1.0, damping=ETA / (mu * DT), dtype=torch.float64
        )
        as_unit = [torch.from_numpy(np.ascontiguousarray(value / H))[None] for value in (X, Y, X_prev)]
        unit_terms = normalised(as_unit[0], as_unit[1], previous_positions=as_unit[2])
        si_terms = oracle.energy_terms(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        scale = oracle.energy_unit(H, E, NU)
        for name in ("total", "elastic", "inertia", "damping"):
            self.assertAlmostEqual(float(getattr(unit_terms, name)[0]) * scale / si_terms[name], 1.0, delta=1e-12)

    def test_frames(self):
        rest, _, _, X, _, _, rng = _states(4)
        rotations, ties = oracle.frames(CELL_COUNTS, H, X, return_tie_mask=True)
        self.assertEqual(rotations.shape, (12, 3, 3))
        self.assertEqual(ties.shape, (12,))
        self.assertTrue(np.isfinite(rotations).all())
        np.testing.assert_allclose(
            rotations.transpose(0, 2, 1) @ rotations, np.broadcast_to(np.eye(3), rotations.shape), atol=1e-12
        )
        np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-12)
        self.assertFalse(ties.any())
        # A rigidly rotated rest grid has every frame equal to the rotation.
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rotation *= np.sign(np.linalg.det(rotation))
        rotated = rest @ rotation.T
        np.testing.assert_allclose(
            oracle.frames(CELL_COUNTS, H, rotated), np.broadcast_to(rotation, (12, 3, 3)), atol=1e-12
        )
        np.testing.assert_allclose(
            oracle.frames(CELL_COUNTS, H, rest), np.broadcast_to(np.eye(3), (12, 3, 3)), atol=1e-15
        )

    def test_modes(self):
        rest, cells, _, X, _, _, rng = _states(5)
        vectors = oracle.modes(CELL_COUNTS, H, X)
        self.assertEqual(vectors.shape, (12, 7, 3))
        self.assertTrue(np.isfinite(vectors).all())
        at_rest = oracle.modes(CELL_COUNTS, H, rest)
        np.testing.assert_allclose(at_rest[:, :3], np.broadcast_to(np.eye(3), (12, 3, 3)), atol=1e-15)
        np.testing.assert_allclose(at_rest[:, 3:], 0.0, atol=1e-15)
        # First three vectors are the columns of the centre F = sum_k (x_k - x_0) signs_k^T / (4 h).
        signs = np.asarray(oracle.conventions()["corner_order"], dtype=float)
        corners = X[cells]
        centre = np.einsum("cki,kj->cij", corners - corners[:, :1], signs) / (4 * H)
        np.testing.assert_allclose(vectors[:, :3].transpose(0, 2, 1), centre, atol=1e-13)
        # Translation invariance; a single warped cell has nonzero warping vectors.
        np.testing.assert_allclose(oracle.modes(CELL_COUNTS, H, X + rng.normal(size=3)), vectors, atol=1e-12)
        self.assertGreater(np.abs(vectors[:, 3:]).max(), 0.0)

    def test_fuse(self):
        _, _, pinned, X, _, _, rng = _states(6)
        vectors = oracle.modes(CELL_COUNTS, H, X)
        zero = oracle.fuse(CELL_COUNTS, H, X, np.zeros_like(vectors))
        self.assertEqual(zero.shape, X.shape)
        np.testing.assert_array_equal(zero, 0.0)
        # A compatible increment (from a displacement with zero pinned rows) is reproduced exactly.
        delta = rng.normal(size=X.shape) * 1e-3 * H
        delta[pinned] = 0.0
        increment = oracle.modes(CELL_COUNTS, H, X + delta) - vectors
        fused = oracle.fuse(CELL_COUNTS, H, X, increment)
        self.assertTrue(np.isfinite(fused).all())
        np.testing.assert_array_equal(fused[pinned], 0.0)
        np.testing.assert_allclose(fused, delta, atol=1e-12 * H)
        # Independent of the base positions.
        np.testing.assert_allclose(oracle.fuse(CELL_COUNTS, H, X + 1e-3 * H, increment), fused, atol=1e-15)

    def test_project_gradient_is_the_fusion_adjoint(self):
        _, _, pinned, X, Y, X_prev, rng = _states(7)
        _, gradient = oracle.energy_and_grad(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        projected = oracle.project_gradient(CELL_COUNTS, H, X, gradient)
        self.assertEqual(projected.shape, (12, 7, 3))
        self.assertTrue(np.isfinite(projected).all())
        for _ in range(3):
            increment = rng.normal(size=(12, 7, 3))
            fused = oracle.fuse(CELL_COUNTS, H, X, increment)
            self.assertAlmostEqual(np.sum(projected * increment) / np.sum(gradient * fused), 1.0, delta=1e-9)
        unit = oracle.energy_unit(H, E, NU)
        np.testing.assert_allclose(
            oracle.project_gradient(CELL_COUNTS, H, X, gradient, energy_unit=unit), projected / unit, rtol=1e-15
        )
        # Pinned rows of the gradient do not enter.
        poisoned = gradient.copy()
        poisoned[pinned] = 1e6
        np.testing.assert_array_equal(oracle.project_gradient(CELL_COUNTS, H, X, poisoned), projected)

    def test_network_inputs(self):
        _, _, _, X, Y, X_prev, rng = _states(8)
        inputs = oracle.network_inputs(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        expected = {
            "frames": (12, 3, 3),
            "tie_mask": (12,),
            "local_axes": (12, 7, 3),
            "state_features": (12, 121),
            "node_features": (12, 159),
            "edge_features": (12, 27, 24),
            "neighbor_indices": (12, 27),
            "neighbor_mask": (12, 27),
            "conditioning": (7,),
            "axis_gradient_world": (12, 7, 3),
            "position_gradient": (36, 3),
        }
        self.assertEqual({name: value.shape for name, value in inputs.items()}, expected)
        for name, value in inputs.items():
            self.assertTrue(np.isfinite(value).all(), name)
        np.testing.assert_allclose(inputs["frames"], oracle.frames(CELL_COUNTS, H, X), atol=1e-15)
        local = np.einsum("cji,cmj->cmi", inputs["frames"], oracle.modes(CELL_COUNTS, H, X))
        np.testing.assert_allclose(inputs["local_axes"], local, atol=1e-13)
        node = inputs["node_features"]
        np.testing.assert_allclose(node[:, :21], inputs["local_axes"].transpose(0, 2, 1).reshape(12, 21))
        np.testing.assert_array_equal(node[:, 21:142], inputs["state_features"])
        np.testing.assert_array_equal(node[:, 142:], 0.0)
        self.assertTrue(np.all(inputs["state_features"][:, -1] == 0.0))  # history_valid
        self.assertTrue(np.all(inputs["neighbor_mask"][:, 0]))  # self slot
        _, gradient = oracle.energy_and_grad(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        np.testing.assert_allclose(
            inputs["axis_gradient_world"],
            oracle.project_gradient(CELL_COUNTS, H, X, gradient, energy_unit=oracle.energy_unit(H, E, NU)),
            rtol=1e-12,
        )
        np.testing.assert_allclose(inputs["conditioning"], oracle.conditioning(H, DT, *MATERIAL))
        history = (rng.normal(size=(12, 7, 3)), rng.normal(size=(12, 7, 3)))
        with_history = oracle.network_inputs(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL, history=history)
        self.assertTrue(np.all(with_history["state_features"][:, -1] == 1.0))
        self.assertGreater(np.abs(with_history["state_features"][:, 63:105]).max(), 0.0)

    @unittest.skipUnless(oracle.V4_CHECKPOINT.is_file(), "v4 checkpoint not present")
    def test_network_state_dict_summary(self):
        summary = oracle.network_state_dict_summary(oracle.V4_CHECKPOINT)
        names = dict(summary)
        self.assertEqual(len(summary), 60)
        self.assertEqual(names["node_encoder.0.weight"], (192, 159))
        self.assertEqual(names["edge_encoder.0.weight"], (96, 24))
        self.assertEqual(names["contact_encoder.token_encoder.0.weight"], (64, 19))
        self.assertEqual(names["layers.0.edge_update.0.weight"], (192, 480))
        self.assertEqual(names["correction_head.weight"], (21, 192))

    @unittest.skipUnless(oracle.V4_CHECKPOINT.is_file(), "v4 checkpoint not present")
    def test_network_forward(self):
        _, _, _, X, Y, X_prev, _ = _states(9)
        inputs = oracle.network_inputs(CELL_COUNTS, H, DT, X, Y, X_prev, *MATERIAL)
        output = oracle.network_forward(oracle.V4_CHECKPOINT, CELL_COUNTS, inputs)
        self.assertEqual(output["local_target"].shape, (12, 7, 3))
        self.assertEqual(output["correction"].shape, (12, 7, 3))
        self.assertEqual(output["step_size"].shape, (12,))
        for value in output.values():
            self.assertTrue(np.isfinite(value).all())
        self.assertTrue(np.all(output["step_size"] > 0.0) and np.all(output["step_size"] < 0.05))
        self.assertLess(np.sqrt((output["correction"] ** 2).sum((1, 2))).max(), 1.0)
        reconstructed = inputs["local_axes"] + output["step_size"][:, None, None] * output["correction"]
        np.testing.assert_allclose(output["local_target"], reconstructed, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
