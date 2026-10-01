# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the shared FEM accuracy scenarios, schedules, metrics, and file layout."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.learned_intrinsic_solver import fem_accuracy_scenarios as scenarios
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.render_learned import _load_trajectory

SMALL_COUNTS = (2, 3, 4)
SMALL_SIZE = 0.025


def _rotation_z(theta: float) -> np.ndarray:
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


class TestConstants(unittest.TestCase):
    def test_material_and_time_constants(self):
        """Derive Lame parameters and the substep from the shared VBD sample constants."""
        self.assertAlmostEqual(scenarios.LAME_MU, 5.0e5 / 2.6)
        self.assertAlmostEqual(scenarios.LAME_LAMBDA, 5.0e5 * 0.3 / (1.3 * 0.4))
        self.assertAlmostEqual(scenarios.TIME_STEP, 1.0 / 300.0)
        self.assertEqual(scenarios.BEAM_LENGTH, 1.0)
        self.assertEqual(scenarios.BEAM_WIDTH, 0.25)
        self.assertAlmostEqual(scenarios.analytic_extension_tip_displacement(), 9.81e-3)

    def test_beam_rest_and_corner_sets(self):
        """Build the 10x10x40 beam and select its clamp, far face, and mid-length section."""
        rest = scenarios.beam_rest()
        self.assertEqual(len(rest.corner_rest_positions), 4961)
        self.assertEqual(len(rest.cell_corner_indices), 4000)
        clamp = scenarios.clamp_indices(rest)
        far = scenarios.far_face_indices(rest)
        mid = scenarios.mid_length_indices(rest)
        self.assertEqual((len(clamp), len(far), len(mid)), (121, 121, 121))
        self.assertTrue(np.all(rest.corner_rest_positions[clamp, 2] == 0.0))
        self.assertTrue(np.allclose(rest.corner_rest_positions[far, 2], 1.0))
        self.assertTrue(np.allclose(rest.corner_rest_positions[mid, 2], 0.5))
        self.assertEqual(len(np.intersect1d(clamp, far)), 0)


class TestGeometryMetrics(unittest.TestCase):
    def setUp(self):
        self.rest = generate_cuboid(SMALL_COUNTS, cell_size=SMALL_SIZE)
        self.positions = self.rest.corner_rest_positions
        self.cells = self.rest.cell_corner_indices

    def test_rest_volume_and_jacobian(self):
        """Recover h^3 per cell and a unit centre Jacobian ratio at rest."""
        volumes = scenarios.cell_volumes(self.positions, self.cells)
        self.assertEqual(volumes.shape, (24,))
        np.testing.assert_allclose(volumes, SMALL_SIZE**3, rtol=1e-12)
        ratios = scenarios.centre_jacobian_ratios(self.positions, self.positions, self.cells)
        np.testing.assert_allclose(ratios, 1.0, rtol=1e-12)
        self.assertAlmostEqual(scenarios.bulk_volume_ratio(self.positions, self.positions, self.cells), 1.0)

    def test_affine_deformation_scales_by_determinant(self):
        """Scale volumes and centre Jacobians by det(A) under x = A X + b."""
        matrix = np.array([[1.2, 0.1, -0.05], [0.0, 0.9, 0.2], [0.3, -0.1, 1.1]])
        deformed = self.positions @ matrix.T + np.array([0.3, -0.2, 0.1])
        determinant = np.linalg.det(matrix)
        np.testing.assert_allclose(
            scenarios.cell_volumes(deformed, self.cells), determinant * SMALL_SIZE**3, rtol=1e-12
        )
        np.testing.assert_allclose(
            scenarios.centre_jacobian_ratios(deformed, self.positions, self.cells), determinant, rtol=1e-12
        )
        self.assertAlmostEqual(scenarios.bulk_volume_ratio(deformed, self.positions, self.cells), determinant)

    def test_frustum_volume_is_exact(self):
        """Match the closed-form volume of a square frustum, whose faces stay planar."""
        rest = generate_cuboid((1, 1, 1), cell_size=1.0)
        positions = rest.corner_rest_positions.copy()
        top = positions[:, 2] > 0.5
        bottom_side, top_side = 1.0, 0.4
        positions[top, :2] = 0.5 + (positions[top, :2] - 0.5) * top_side / bottom_side
        expected = (bottom_side**2 + bottom_side * top_side + top_side**2) / 3.0
        np.testing.assert_allclose(scenarios.cell_volumes(positions, rest.cell_corner_indices), expected, rtol=1e-12)

    def test_inverted_cell_reports_negative_measures(self):
        """Flag inversion with negative volume and centre Jacobian ratio."""
        rest = generate_cuboid((1, 1, 1), cell_size=1.0)
        positions = rest.corner_rest_positions.copy()
        positions[:, 2] = 1.0 - positions[:, 2]
        self.assertLess(scenarios.cell_volumes(positions, rest.cell_corner_indices)[0], 0.0)
        self.assertLess(
            scenarios.centre_jacobian_ratios(positions, rest.corner_rest_positions, rest.cell_corner_indices)[0], 0.0
        )

    def test_tip_displacement_and_lateral_contraction(self):
        """Measure the far-face z offset per frame and the mid-length width ratio."""
        far = scenarios.far_face_indices(self.rest)
        mid = scenarios.mid_length_indices(self.rest)
        shifted = self.positions + np.array([0.0, 0.0, 0.01])
        self.assertAlmostEqual(scenarios.tip_displacement(shifted, self.rest, far), 0.01)
        series = scenarios.tip_displacement(np.stack([self.positions, shifted]), self.rest, far)
        np.testing.assert_allclose(series, [0.0, 0.01], atol=1e-15)
        squeezed = self.positions * np.array([0.8, 0.8, 1.0])
        self.assertAlmostEqual(scenarios.lateral_contraction(squeezed, self.rest, mid), 0.8)


class TestSchedules(unittest.TestCase):
    def setUp(self):
        self.rest = scenarios.beam_rest()
        self.far = scenarios.far_face_indices(self.rest)
        self.face = self.rest.corner_rest_positions[self.far]

    def test_registry_and_timing(self):
        """Expose the four scenarios with their frame counts, gravity, and time grids."""
        self.assertEqual(list(scenarios.SCENARIOS), ["extension", "stretch", "twist", "compression_release"])
        expected_frames = {"extension": 300, "stretch": 400, "twist": 300, "compression_release": 400}
        for name, scenario in scenarios.SCENARIOS.items():
            self.assertEqual(scenario.frame_count, expected_frames[name])
            times = scenarios.frame_times(scenario)
            substeps = scenarios.substep_times(scenario)
            self.assertEqual(times.shape, (scenario.frame_count + 1,))
            self.assertEqual(times[0], 0.0)
            self.assertAlmostEqual(times[-1], scenario.frame_count / 30)
            self.assertEqual(substeps.shape, (scenario.frame_count, 10))
            np.testing.assert_array_equal(substeps[:, -1], times[1:])
            np.testing.assert_allclose(np.diff(substeps.reshape(-1)), scenarios.TIME_STEP, rtol=1e-9)
            json.dumps(scenarios.scenario_summary(scenario))
        self.assertEqual(scenarios.SCENARIOS["extension"].gravity, (0.0, 0.0, 9.81))
        for name in ("stretch", "twist", "compression_release"):
            self.assertEqual(scenarios.SCENARIOS[name].gravity, (0.0, 0.0, 0.0))

    def test_extension_never_drives(self):
        """Leave the far face free for the whole hanging-beam run."""
        scenario = scenarios.SCENARIOS["extension"]
        for time_seconds in (0.0, 1.0, 5.0, 10.0):
            self.assertEqual(scenario.prescribed_far_face(self.rest, time_seconds), (False, None, None))

    def test_stretch_reaches_double_length_and_holds(self):
        """Translate the far face to z = 2L at the ramp end and hold it with zero velocity."""
        scenario = scenarios.SCENARIOS["stretch"]
        driven, positions, velocities = scenario.prescribed_far_face(self.rest, 100 / 30)
        self.assertTrue(driven)
        np.testing.assert_allclose(positions, self.face + np.array([0.0, 0.0, 0.5]), atol=1e-12)
        np.testing.assert_allclose(velocities, np.tile([0.0, 0.0, 1.0 / (200 / 30)], (121, 1)), atol=1e-12)
        for time_seconds in (200 / 30, 300 / 30, 400 / 30):
            driven, positions, velocities = scenario.prescribed_far_face(self.rest, time_seconds)
            self.assertTrue(driven)
            np.testing.assert_allclose(positions[:, 2], 2.0, atol=1e-12)
            np.testing.assert_allclose(positions[:, :2], self.face[:, :2], atol=1e-12)
            np.testing.assert_array_equal(velocities, 0.0)

    def test_twist_returns_to_rest_at_full_turn(self):
        """Rotate the far face about its centroid: half a turn flips it, a full turn restores it."""
        scenario = scenarios.SCENARIOS["twist"]
        centroid = self.face.mean(axis=0)
        driven, positions, velocities = scenario.prescribed_far_face(self.rest, 100 / 30)
        self.assertTrue(driven)
        flipped = centroid + (self.face - centroid) @ _rotation_z(np.pi).T
        np.testing.assert_allclose(positions, flipped, atol=1e-12)
        np.testing.assert_allclose(positions[:, :2], 2 * centroid[:2] - self.face[:, :2], atol=1e-12)
        omega = 2 * np.pi / (200 / 30)
        expected_velocity = omega * np.cross([0.0, 0.0, 1.0], positions - centroid)
        np.testing.assert_allclose(velocities, expected_velocity, atol=1e-12)
        for time_seconds in (200 / 30, 300 / 30):
            driven, positions, velocities = scenario.prescribed_far_face(self.rest, time_seconds)
            self.assertTrue(driven)
            np.testing.assert_allclose(positions, self.face, atol=1e-12)
            np.testing.assert_array_equal(velocities, 0.0)
        driven, positions, _ = scenario.prescribed_far_face(self.rest, 50 / 30)
        np.testing.assert_allclose(positions, centroid + (self.face - centroid) @ _rotation_z(np.pi / 2).T, atol=1e-12)

    def test_compression_reaches_half_length_then_releases(self):
        """Push the far face to z = 0.5 L, hold it through frame 150, and free it afterwards."""
        scenario = scenarios.SCENARIOS["compression_release"]
        self.assertEqual(scenario.release_frame, 150)
        driven, positions, velocities = scenario.prescribed_far_face(self.rest, 50 / 30)
        self.assertTrue(driven)
        np.testing.assert_allclose(positions[:, 2], 0.75, atol=1e-12)
        np.testing.assert_allclose(velocities[:, 2], -0.5 / (100 / 30), atol=1e-12)
        for time_seconds in (100 / 30, 150 / 30, 1500 * scenarios.TIME_STEP):
            driven, positions, velocities = scenario.prescribed_far_face(self.rest, time_seconds)
            self.assertTrue(driven)
            np.testing.assert_allclose(positions[:, 2], 0.5, atol=1e-12)
            np.testing.assert_array_equal(velocities, 0.0)
        for time_seconds in (151 / 30, 1501 * scenarios.TIME_STEP, 400 / 30):
            self.assertEqual(scenario.prescribed_far_face(self.rest, time_seconds), (False, None, None))


class TestMetricsAndFiles(unittest.TestCase):
    def setUp(self):
        self.rest = generate_cuboid(SMALL_COUNTS, cell_size=SMALL_SIZE)
        self.positions = self.rest.corner_rest_positions
        self.cells = self.rest.cell_corner_indices
        self.clamp = scenarios.clamp_indices(self.rest)
        self.far = scenarios.far_face_indices(self.rest)

    def _trajectory(self, frame_count: int):
        """Stretch the beam linearly along z over the recorded frames, keeping the clamp fixed."""
        times = np.arange(frame_count + 1) / scenarios.FPS
        factors = 1.0 + 0.5 * times / times[-1]
        positions = np.stack([self.positions * np.array([1.0, 1.0, factor]) for factor in factors])
        return positions, times

    def test_compute_metrics_per_scenario(self):
        """Produce the listed, JSON-serialisable metrics for every scenario."""
        positions, times = self._trajectory(4)
        extension = scenarios.compute_metrics(
            scenarios.SCENARIOS["extension"], positions, self.rest, self.cells, self.far, times
        )
        self.assertEqual(extension["frame_count"], 4)
        self.assertFalse(extension["completed"])
        self.assertAlmostEqual(extension["tip_displacement_final"], 0.05)
        self.assertEqual(len(extension["tip_displacement_series"]), 5)
        self.assertEqual(extension["series_times"], list(times))
        self.assertAlmostEqual(extension["analytic_tip_displacement"], 9.81e-3)
        self.assertAlmostEqual(extension["bulk_volume_ratio"], 1.5)
        self.assertAlmostEqual(extension["min_centre_jacobian_ratio"], 1.5)
        stretch = scenarios.compute_metrics(
            scenarios.SCENARIOS["stretch"], positions, self.rest, self.cells, self.far, times
        )
        self.assertEqual(
            set(stretch) - {"scenario", "frame_count", "completed", "final_time_seconds"},
            {"bulk_volume_ratio", "lateral_contraction", "min_centre_jacobian_ratio"},
        )
        self.assertAlmostEqual(stretch["lateral_contraction"], 1.0)
        twist = scenarios.compute_metrics(
            scenarios.SCENARIOS["twist"], positions, self.rest, self.cells, self.far, times
        )
        self.assertIsNone(twist["bulk_volume_ratio_peak"])
        self.assertAlmostEqual(twist["bulk_volume_ratio_final"], 1.5)
        compression = scenarios.compute_metrics(
            scenarios.SCENARIOS["compression_release"], positions, self.rest, self.cells, self.far, times
        )
        self.assertAlmostEqual(compression["min_centre_jacobian_ratio_compression"], 1.0)
        self.assertAlmostEqual(compression["length_recovery_ratio"], 1.5)
        self.assertAlmostEqual(compression["bulk_volume_ratio_final"], 1.5)
        for metrics in (extension, stretch, twist, compression):
            json.dumps(metrics)

    def test_twist_peak_metrics_use_ramp_end_frame(self):
        """Evaluate the twist peak at the ramp-end frame once it has been recorded."""
        scenario = scenarios.SCENARIOS["twist"]
        positions, times = self._trajectory(scenario.frame_count)
        metrics = scenarios.compute_metrics(scenario, positions, self.rest, self.cells, self.far, times)
        self.assertTrue(metrics["completed"])
        self.assertEqual(metrics["peak_frame"], 200)
        peak_factor = 1.0 + 0.5 * 200 / 300
        self.assertAlmostEqual(metrics["bulk_volume_ratio_peak"], peak_factor)
        self.assertAlmostEqual(metrics["min_centre_jacobian_ratio_peak"], peak_factor)
        self.assertAlmostEqual(metrics["bulk_volume_ratio_final"], 1.5)

    def test_write_trajectory_round_trips_through_renderer_loader(self):
        """Write the npz layout and load it with render_learned._load_trajectory."""
        positions, times = self._trajectory(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned" / "stretch" / "trajectory.npz"
            written = scenarios.write_trajectory(path, positions, times, self.rest, self.clamp, self.cells)
            self.assertEqual(written, path)
            loaded_positions, loaded_times, loaded_rest, loaded_fixed, loaded_cells, _ = _load_trajectory(path)
            self.assertEqual(loaded_positions.dtype, np.float32)
            self.assertEqual(loaded_positions.shape, (4, len(self.positions), 3))
            np.testing.assert_allclose(loaded_positions, positions, rtol=1e-6)
            np.testing.assert_array_equal(loaded_times, times)
            np.testing.assert_allclose(loaded_rest, self.positions, rtol=1e-6)
            np.testing.assert_array_equal(loaded_fixed, self.clamp)
            np.testing.assert_array_equal(loaded_cells, self.cells)
            trajectory = scenarios.read_trajectory(path)
            self.assertEqual(trajectory.positions.dtype, np.float32)
            self.assertEqual(trajectory.rest_positions.dtype, np.float64)
            np.testing.assert_array_equal(trajectory.fixed_indices, self.clamp)
            metrics_path = scenarios.write_metrics(path.with_name("metrics.json"), {"bulk_volume_ratio": 1.5})
            self.assertEqual(scenarios.read_metrics(metrics_path), {"bulk_volume_ratio": 1.5})

    def test_write_trajectory_rejects_moving_pins_and_bad_times(self):
        """Refuse to list a moving face as fixed or to store nonmonotone times."""
        positions, times = self._trajectory(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            with self.assertRaises(ValueError):
                scenarios.write_trajectory(path, positions, times, self.rest, self.far, self.cells)
            with self.assertRaises(ValueError):
                scenarios.write_trajectory(path, positions, times[::-1], self.rest, self.clamp, self.cells)
            with self.assertRaises(ValueError):
                scenarios.write_trajectory(path, positions[:, :-1], times, self.rest, self.clamp, self.cells)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
