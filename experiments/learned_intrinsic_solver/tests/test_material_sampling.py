# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for deterministic trajectory-level material sampling."""

import math
import unittest
from dataclasses import FrozenInstanceError, asdict

import numpy as np

from experiments.learned_intrinsic_solver.material_sampling import MaterialRanges, MaterialSample, sample_material


class TestMaterialSampling(unittest.TestCase):
    def test_same_seed_repeats_and_sample_is_model_ready(self):
        """Repeat the same material sample and expose native model keywords."""
        first = sample_material(73)
        self.assertEqual(first, sample_material(73))
        self.assertIsInstance(first, MaterialSample)
        self.assertEqual(set(asdict(first)), {"lame_lambda", "lame_mu", "density"})
        with self.assertRaises(FrozenInstanceError):
            first.density = 1000.0

    def test_seed_changes_independent_material_components(self):
        """Change all three draws when the trajectory seed changes."""
        first, second = sample_material(73), sample_material(74)
        self.assertNotEqual(first.lame_lambda, second.lame_lambda)
        self.assertNotEqual(first.lame_mu, second.lame_mu)
        self.assertNotEqual(first.density, second.density)

    def test_known_rng_draws_map_uniformly_in_log_space(self):
        """Map the three seeded uniform draws to the specified log ranges."""
        ranges = MaterialRanges(lame_lambda=(10.0, 100.0), lame_mu=(100.0, 10000.0), density=(1.0, 1000.0))
        draws = np.random.default_rng(31).random(3)
        sampled = sample_material(31, ranges=ranges)
        for value, bounds, draw in zip(asdict(sampled).values(), asdict(ranges).values(), draws, strict=True):
            expected = math.exp(math.log(bounds[0]) + float(draw) * (math.log(bounds[1]) - math.log(bounds[0])))
            self.assertAlmostEqual(value, expected, places=12)

    def test_default_samples_stay_within_requested_bounds(self):
        """Bound every component over distinct deterministic trajectory seeds."""
        ranges = MaterialRanges()
        self.assertEqual(ranges.lame_lambda, (1e3, 1e6))
        self.assertEqual(ranges.lame_mu, (1e3, 1e6))
        self.assertEqual(ranges.density, (100.0, 10000.0))
        for seed in range(100):
            sample = sample_material(seed)
            for value, bounds in zip(asdict(sample).values(), asdict(ranges).values(), strict=True):
                self.assertGreaterEqual(value, bounds[0])
                self.assertLessEqual(value, bounds[1])

    def test_invalid_seed_or_ranges_fail(self):
        """Reject ambiguous seeds and nonpositive or unordered log bounds."""
        for seed in (-1, True, 1.5, "73"):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                sample_material(seed)
        for bounds in ((0.0, 1.0), (-1.0, 1.0), (1.0, 1.0), (2.0, 1.0), (1.0, math.inf), (math.nan, 2.0)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                MaterialRanges(lame_mu=bounds)


if __name__ == "__main__":
    unittest.main()
