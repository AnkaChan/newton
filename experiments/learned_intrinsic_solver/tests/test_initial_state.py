# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for seeded, fresh physical trajectory initial states."""

import unittest
from dataclasses import asdict

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.initial_state import InitialStateAugmenter
from experiments.learned_intrinsic_solver.material_sampling import MaterialRanges, sample_material
from experiments.learned_intrinsic_solver.train_smoke import TrainSmokeConfig, _Sampler


class TestInitialStateAugmenter(unittest.TestCase):
    def setUp(self):
        self.rest = generate_cuboid((2, 2, 3), cell_size=0.1)
        self.augmenter = InitialStateAugmenter(self.rest, seed=7, master_seed=73, time_step=1 / 300)

    def test_repeated_reordered_and_fresh_instance_resets_are_exact(self):
        first = self.augmenter.reset()
        self.augmenter.reset(91)
        repeated = self.augmenter.reset(7)
        fresh = InitialStateAugmenter(self.rest, seed=7, master_seed=73, time_step=1 / 300).reset()
        for other in (repeated, fresh):
            np.testing.assert_array_equal(other.positions, first.positions)
            np.testing.assert_array_equal(other.velocities, first.velocities)
            np.testing.assert_array_equal(other.fixed_indices, first.fixed_indices)
            self.assertEqual(other.material, first.material)
            self.assertEqual(other.metadata, first.metadata)
        np.testing.assert_array_equal(self.augmenter.reset().positions, first.positions)

    def test_result_and_source_mutation_cannot_change_future_reset(self):
        reference = self.augmenter.reset(7)
        positions = reference.positions.copy()
        velocities = reference.velocities.copy()
        rest_positions = self.rest.corner_rest_positions.copy()
        reference.positions[:] = 42
        reference.velocities[:] = -42
        reference.fixed_indices[:] = 0
        reference.metadata["physical_seed"] = -1
        self.rest.corner_rest_positions[:] = 99
        self.rest.cell_corner_indices[:] = 0
        again = self.augmenter.reset(7)
        np.testing.assert_array_equal(again.positions, positions)
        np.testing.assert_array_equal(again.velocities, velocities)
        np.testing.assert_array_equal(
            again.positions[again.fixed_indices], rest_positions[again.fixed_indices].astype(np.float32)
        )
        self.assertEqual(again.metadata["physical_seed"], 7)

    def test_legacy_sampler_shape_and_velocity_stream_match_exactly(self):
        config = TrainSmokeConfig(
            cell_counts=self.rest.cell_counts, cell_size=self.rest.cell_size, seed=73, device="cpu"
        )
        sampler = object.__new__(_Sampler)
        sampler.config, sampler.rest = config, self.rest
        fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == self.rest.corner_rest_positions[:, 2].min())
        for seed in (0, 7, 19):
            expected = sampler._physical(seed, fixed)
            actual = self.augmenter.reset(seed)
            np.testing.assert_array_equal(actual.positions, expected["positions"])
            np.testing.assert_array_equal(actual.velocities, expected["velocity"])
            self.assertEqual(actual.metadata["strength"], expected["metadata"]["strength"])
            self.assertEqual(
                actual.metadata["requested_velocity_dt_rms_m"], expected["metadata"]["requested_velocity_dt_rms_m"]
            )

    def test_material_stream_independent_of_shape_and_within_bounds(self):
        seed = 31
        state = self.augmenter.reset(seed)
        material_seed = int(np.random.SeedSequence([73, seed, 1103]).generate_state(1, dtype=np.uint64)[0])
        self.assertEqual(state.material, sample_material(material_seed))
        self.assertEqual(state.metadata["material_seed"], material_seed)
        different_shape = InitialStateAugmenter(self.rest, master_seed=73, strength_range=(0.0, 0.0)).reset(seed)
        self.assertEqual(different_shape.material, state.material)
        custom_ranges = MaterialRanges(lame_lambda=(1e4, 1e5), lame_mu=(1e3, 1e4), density=(200, 500))
        different_material = InitialStateAugmenter(self.rest, master_seed=73, material_ranges=custom_ranges).reset(seed)
        np.testing.assert_array_equal(different_material.positions, state.positions)
        np.testing.assert_array_equal(different_material.velocities, state.velocities)
        for value, bounds in zip(
            asdict(different_material.material).values(), asdict(custom_ranges).values(), strict=True
        ):
            self.assertGreaterEqual(value, bounds[0])
            self.assertLessEqual(value, bounds[1])

    def test_float32_screen_pins_and_metadata(self):
        state = self.augmenter.reset(2)
        self.assertEqual(state.positions.dtype, np.float32)
        self.assertEqual(state.velocities.dtype, np.float32)
        self.assertEqual(state.fixed_indices.dtype, np.int64)
        self.assertTrue(np.isfinite(state.positions).all())
        self.assertTrue(np.isfinite(state.velocities).all())
        np.testing.assert_array_equal(
            state.positions[state.fixed_indices],
            self.rest.corner_rest_positions[state.fixed_indices].astype(np.float32),
        )
        np.testing.assert_array_equal(state.velocities[state.fixed_indices], 0)
        self.assertGreaterEqual(state.metadata["screen"]["min_tet_volume_ratio"], 0.2)
        self.assertGreaterEqual(state.metadata["screen"]["min_sampled_jacobian"], 0.2)
        self.assertEqual(state.metadata["schema_version"], 1)
        self.assertEqual(state.metadata["generator_version"], "initial_state_v1")
        self.assertEqual(state.metadata["physical_seed_sequence"], [73, 2, 701])

    def test_reset_does_not_advance_global_numpy_random_state(self):
        original = np.random.get_state()  # noqa: NPY002 -- Verify the legacy global RNG is untouched.
        try:
            np.random.seed(201)  # noqa: NPY002 -- Set a known global state for the isolation check.
            before = np.random.get_state()  # noqa: NPY002
            self.augmenter.reset(12)
            after = np.random.get_state()  # noqa: NPY002
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
        finally:
            np.random.set_state(original)  # noqa: NPY002 -- Restore the caller's global RNG state.

    def test_invalid_configuration_and_unrecoverable_float32_shape_fail(self):
        for kwargs in (
            {"seed": -1},
            {"master_seed": True},
            {"time_step": 0},
            {"strength_range": (0.2, 0.1)},
            {"velocity_dt_range": (-1, 0.1)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                InitialStateAugmenter(self.rest, **kwargs)
        with self.assertRaises((TypeError, ValueError)):
            self.augmenter.reset(-1)
        unrepresentable = generate_cuboid((1, 1, 1), cell_size=0.1, origin=(1e8, 1e8, 1e8))
        with self.assertRaisesRegex(ValueError, "float32"):
            InitialStateAugmenter(unrepresentable).reset(3)


if __name__ == "__main__":
    unittest.main()
