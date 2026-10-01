# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the VBD reference driver's beam construction and summary line."""

import unittest

import numpy as np
import warp as wp

import newton
from experiments.learned_intrinsic_solver import fem_accuracy_scenarios as scenarios
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.fem_accuracy_vbd import format_summary
from experiments.learned_intrinsic_solver.vbd_samples import DENSITY, _ordering_map, add_beam


class TestAddBeam(unittest.TestCase):
    def test_particles_match_hex_corners_and_masses_are_lumped(self):
        """Place the tetrahedral particles on the hex corners and lump rho V / 4 with a zero-mass clamp."""
        rest = scenarios.beam_rest()
        mapping = _ordering_map(rest.cell_counts)
        fixed = np.zeros(len(mapping), dtype=bool)
        fixed[mapping[scenarios.clamp_indices(rest)]] = True
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, scenarios.GRAVITY_MAGNITUDE))
        tets, determinants, total_mass = add_beam(builder, fixed, pos=wp.vec3(0.0), rot=wp.quat_identity())
        particles = np.asarray(builder.particle_q, dtype=np.float64)
        np.testing.assert_allclose(particles[mapping], rest.corner_rest_positions, rtol=0.0, atol=1e-6)
        rest_volume = scenarios.BEAM_WIDTH**2 * scenarios.BEAM_LENGTH
        self.assertEqual(tets.shape, (5 * len(rest.cell_corner_indices), 4))
        self.assertTrue(np.all(determinants > 0.0))
        self.assertAlmostEqual(determinants.sum() / 6.0, rest_volume, places=6)
        self.assertAlmostEqual(total_mass, DENSITY * rest_volume, places=3)
        mass = np.asarray(builder.particle_mass)
        self.assertEqual(len(mass), len(mapping))
        self.assertTrue(np.all(mass[fixed] == 0.0))
        self.assertTrue(np.all(mass[~fixed] > 0.0))
        self.assertEqual(list(builder.gravity), [0.0, 0.0, np.float32(scenarios.GRAVITY_MAGNITUDE)])
        with self.assertRaises(ValueError):
            add_beam(builder, fixed, pos=wp.vec3(0.0), rot=wp.quat_identity())


class TestSummary(unittest.TestCase):
    def test_format_summary_covers_every_scenario(self):
        """Name each scenario, flag truncated runs, and print unreached metrics as None."""
        rest = generate_cuboid((2, 2, 4), cell_size=0.25)
        far = scenarios.far_face_indices(rest)
        for scenario in scenarios.SCENARIOS.values():
            positions = np.repeat(rest.corner_rest_positions[None], 3, axis=0)
            times = scenarios.frame_times(scenario)[:3]
            metrics = scenarios.compute_metrics(scenario, positions, rest, rest.cell_corner_indices, far, times)
            line = format_summary(metrics, 1.5)
            self.assertTrue(line.startswith(f"{scenario.name}: "))
            self.assertIn("INCOMPLETE (2 frames)", line)
            self.assertTrue(line.endswith("wall=1.5s"))
            self.assertEqual(line.count("\n"), 0)
            if scenario.name == "twist":
                self.assertIn("peak V/V0=None", line)
        complete = {"scenario": "extension", "completed": True, "tip_displacement_final": 0.0123}
        self.assertEqual(format_summary(complete), "extension: tip=0.0123 m (analytic None m) V/V0=None minJ=None")


if __name__ == "__main__":
    unittest.main()
