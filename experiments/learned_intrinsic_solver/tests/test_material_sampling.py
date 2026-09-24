# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for deterministic trajectory-level material sampling."""

import math
import unittest
from dataclasses import FrozenInstanceError, asdict

import numpy as np

from experiments.learned_intrinsic_solver.material_sampling import (
    MaterialRanges,
    MaterialSample,
    lame_from_youngs_modulus,
    sample_material,
)


class TestMaterialSampling(unittest.TestCase):
    def test_same_seed_repeats_and_sample_is_model_ready(self):
        """Repeat the same material sample and expose native model keywords."""
        first = sample_material(73)
        self.assertEqual(first, sample_material(73))
        self.assertIsInstance(first, MaterialSample)
        self.assertEqual(set(asdict(first)), {"lame_lambda", "lame_mu", "density"})
        with self.assertRaises(FrozenInstanceError):
            first.density = 1000.0

    def test_seed_changes_independent_material_draws(self):
        """Change Young's modulus, Poisson's ratio, and density with the seed."""
        ranges = MaterialRanges()
        first, second = sample_material(73, ranges=ranges), sample_material(74, ranges=ranges)
        self.assertNotEqual(first.youngs_modulus, second.youngs_modulus)
        self.assertNotAlmostEqual(first.poissons_ratio, second.poissons_ratio)
        self.assertNotEqual(first.density, second.density)

    def test_standard_lame_conversion_and_inverse_properties(self):
        """Derive standard Lamé parameters and recover Young's modulus and ratio."""
        self.assertEqual(lame_from_youngs_modulus(1000.0, 0.25), (400.0, 400.0))
        self.assertEqual(lame_from_youngs_modulus(1000.0, 0.0), (0.0, 500.0))
        upper_lambda, upper_mu = lame_from_youngs_modulus(2980.0, 0.49)
        self.assertAlmostEqual(upper_lambda, 49000.0)
        self.assertAlmostEqual(upper_mu, 1000.0)
        sample = MaterialSample(*lame_from_youngs_modulus(1000.0, 0.25), density=1234.0)
        self.assertAlmostEqual(sample.youngs_modulus, 1000.0)
        self.assertAlmostEqual(sample.poissons_ratio, 0.25)
        with self.assertRaises(FrozenInstanceError):
            sample.poissons_ratio = 0.3

    def test_known_rng_draws_map_to_log_and_linear_ranges(self):
        """Map seeded modulus and density log draws and the linear ratio draw."""
        ranges = MaterialRanges(youngs_modulus=(10.0, 100.0), poissons_ratio=(0.2, 0.49), density=(1.0, 1000.0))
        draws = np.random.default_rng(31).random(3)
        sampled = sample_material(31, ranges=ranges)
        expected_e = math.exp(math.log(10.0) + float(draws[0]) * math.log(10.0))
        expected_nu = 0.2 + float(draws[1]) * 0.29
        expected_rho = math.exp(math.log(1.0) + float(draws[2]) * math.log(1000.0))
        expected_lam, expected_mu = lame_from_youngs_modulus(expected_e, expected_nu)
        self.assertAlmostEqual(sampled.lame_lambda, expected_lam, places=12)
        self.assertAlmostEqual(sampled.lame_mu, expected_mu, places=12)
        self.assertAlmostEqual(sampled.density, expected_rho, places=12)
        self.assertAlmostEqual(sampled.youngs_modulus, expected_e, places=12)
        self.assertAlmostEqual(sampled.poissons_ratio, expected_nu, places=12)

    def test_default_samples_stay_within_requested_bounds(self):
        """Bound every component over distinct deterministic trajectory seeds."""
        ranges = MaterialRanges()
        self.assertEqual(ranges.youngs_modulus, (1e3, 1e6))
        self.assertEqual(ranges.poissons_ratio, (0.2, 0.49))
        self.assertEqual(ranges.density, (100.0, 10000.0))
        for seed in range(100):
            sample = sample_material(seed)
            self.assertGreaterEqual(sample.youngs_modulus, ranges.youngs_modulus[0])
            self.assertLessEqual(sample.youngs_modulus, ranges.youngs_modulus[1])
            self.assertGreaterEqual(sample.poissons_ratio, ranges.poissons_ratio[0])
            self.assertLessEqual(sample.poissons_ratio, ranges.poissons_ratio[1])
            self.assertGreaterEqual(sample.density, ranges.density[0])
            self.assertLessEqual(sample.density, ranges.density[1])

    def test_fixed_ranges_return_exact_material(self):
        """Allow deterministic fixed modulus, ratio, and density bounds."""
        ranges = MaterialRanges(youngs_modulus=(1000.0, 1000.0), poissons_ratio=(0.25, 0.25), density=(1200.0, 1200.0))
        self.assertEqual(sample_material(0, ranges=ranges), MaterialSample(400.0, 400.0, 1200.0))
        self.assertEqual(sample_material(1, ranges=ranges), sample_material(0, ranges=ranges))

    def test_invalid_seed_or_ranges_fail(self):
        """Reject ambiguous seeds and invalid modulus, ratio, or density domains."""
        for seed in (-1, True, 1.5, "73"):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                sample_material(seed)
        for bounds in ((0.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (1.0, math.inf), (math.nan, 2.0)):
            for field in ("youngs_modulus", "density"):
                with self.subTest(field=field, bounds=bounds), self.assertRaises(ValueError):
                    MaterialRanges(**{field: bounds})
        for bounds in ((-0.1, 0.3), (0.0, 0.5), (0.45, 0.2), (math.nan, 0.3), (0.2, math.inf)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                MaterialRanges(poissons_ratio=bounds)
        for args in ((0.0, 0.3), (1000.0, -0.1), (1000.0, 0.5), (math.inf, 0.3), (1000.0, math.nan)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                lame_from_youngs_modulus(*args)


if __name__ == "__main__":
    unittest.main()
