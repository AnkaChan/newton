# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check information loss and honest shared-corner reconstruction."""

import unittest
from dataclasses import replace

import numpy as np

from experiments.learned_intrinsic_solver.cell_frames import compute_cell_frames
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.round_trip import (
    FrameDecoder,
    build_measurement_operators,
    make_hourglass,
)


class TestFrameRoundTrip(unittest.TestCase):
    def setUp(self):
        """Create a small odd-node cross section with a fixed end."""
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)

    def test_operator_matches_existing_encoder(self):
        """Match opposite-face gradients and corner-average centers exactly."""
        gradient, centers = build_measurement_operators(self.rest)
        rng = np.random.default_rng(42)
        positions = self.rest.corner_rest_positions + rng.uniform(-0.001, 0.001, (36, 3))
        encoded = compute_cell_frames(self.rest, positions)
        np.testing.assert_allclose(
            (gradient @ positions).reshape(-1, 3, 3).transpose(0, 2, 1), encoded.deformation, atol=2e-15
        )
        np.testing.assert_allclose(centers @ positions, encoded.centers, atol=1e-16)

    def test_identity_and_noncanonical_anchors(self):
        """Recover identity and exactly enforce only the supplied fixed face."""
        positions = self.rest.corner_rest_positions
        for include_centers in (False, True):
            decoder = FrameDecoder(self.rest, self.fixed, include_centers=include_centers)
            identity, _ = decoder.decode(compute_cell_frames(self.rest, positions), positions[self.fixed])
            np.testing.assert_allclose(identity, positions, atol=1e-12)
            translated = positions + np.array((0.1, -0.2, 0.07))
            recovered, _ = decoder.decode(compute_cell_frames(self.rest, translated), translated[self.fixed])
            np.testing.assert_array_equal(recovered[self.fixed], translated[self.fixed])
            np.testing.assert_allclose(
                compute_cell_frames(self.rest, recovered).deformation,
                np.broadcast_to(np.eye(3), (len(self.rest.cell_corner_indices), 3, 3)),
                atol=2e-10,
            )

    def test_hourglass_is_invisible_and_not_recovered(self):
        """Show distinct corners with the same centers, frames, axes and boundary."""
        rest = self.rest.corner_rest_positions
        warped = make_hourglass(self.rest, amplitude=0.001)
        original = compute_cell_frames(self.rest, rest)
        hidden = compute_cell_frames(self.rest, warped)
        np.testing.assert_array_equal(warped[self.fixed], rest[self.fixed])
        self.assertAlmostEqual(np.linalg.norm(warped - rest, axis=1).max(), 0.001)
        np.testing.assert_allclose(hidden.centers, original.centers, atol=1e-16)
        np.testing.assert_allclose(hidden.deformation, original.deformation, atol=2e-15)
        np.testing.assert_allclose(hidden.frames @ hidden.local_axes, original.deformation, atol=3e-15)
        for include_centers in (False, True):
            recovered, _ = FrameDecoder(self.rest, self.fixed, include_centers=include_centers).decode(
                hidden, warped[self.fixed]
            )
            np.testing.assert_allclose(recovered, rest, atol=1e-12)
            self.assertGreater(np.linalg.norm(recovered - warped, axis=1).max(), 0.00099)

    def test_decode_uses_polar_factors_not_original_gradient(self):
        """Reconstruct through R and U even when the encoder's raw F is erased."""
        positions = self.rest.corner_rest_positions.copy()
        positions[:, 0] += 0.08 * positions[:, 2]
        encoded = compute_cell_frames(self.rest, positions)
        erased = replace(encoded, deformation=np.full_like(encoded.deformation, np.nan))
        for include_centers in (False, True):
            decoder = FrameDecoder(self.rest, self.fixed, include_centers=include_centers)
            expected, _ = decoder.decode(encoded, positions[self.fixed])
            actual, _ = decoder.decode(erased, positions[self.fixed])
            np.testing.assert_array_equal(actual, expected)

    def test_rank_deficiency_with_centers_and_clamp(self):
        """Demonstrate free-corner nullspace despite the overdetermined row count."""
        gradient, centers = build_measurement_operators(self.rest)
        free = np.ones(len(self.rest.corner_rest_positions), dtype=bool)
        free[self.fixed] = False
        for matrix in (
            gradient[:, free].toarray(),
            np.vstack((gradient[:, free].toarray(), centers[:, free].toarray())),
        ):
            rank = np.linalg.matrix_rank(matrix)
            self.assertLess(rank, free.sum())
            hidden = make_hourglass(self.rest, amplitude=0.001) - self.rest.corner_rest_positions
            np.testing.assert_allclose(matrix @ hidden[free], 0, atol=2e-15)


if __name__ == "__main__":
    unittest.main()
