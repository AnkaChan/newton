# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that the batch uses actual float32 encoding and solver iterates."""

import sys
import unittest

import numpy as np

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.round_trip_batch import (
    Float32Decoder,
    analyze_result,
    compute_frames_float32,
    project_null_component,
)
from experiments.learned_intrinsic_solver.round_trip_batch_report import summarize_samples


class TestFloat32RoundTrip(unittest.TestCase):
    def test_isotropic_rest_identity(self):
        """Encode repeated singular values and reconstruct float32 rest positions."""
        rest = generate_cuboid((2, 2, 3), cell_size=0.025)
        decoder = Float32Decoder(rest)
        original = decoder.rest_positions.copy()
        encoded = compute_frames_float32(rest, original)
        self.assertTrue(encoded.valid.all())
        np.testing.assert_allclose(
            encoded.frames, np.broadcast_to(np.eye(3, dtype=np.float32), encoded.frames.shape), atol=2e-6
        )
        recovered, displacement, rhs, _ = decoder.decode(encoded, original[decoder.fixed])
        metrics = analyze_result(decoder, original, encoded, recovered, displacement, rhs)
        self.assertLess(metrics["corner_max_error_m"], 1e-7)
        self.assertEqual(metrics["boundary_max_error_m"], 0)

    def test_native_float32_encoder(self):
        """Keep the polar path float32 and recover a known positive affine F."""
        rest = generate_cuboid((2, 2, 3), cell_size=0.025)
        affine = np.array([[1.1, 0.04, 0.1], [0.02, 0.9, 0.03], [0, 0, 1]], dtype=np.float32)
        positions = rest.corner_rest_positions.astype(np.float32) @ affine.T
        encoded = compute_frames_float32(rest, positions)
        for array in (encoded.centers, encoded.deformation, encoded.frames, encoded.local_axes, encoded.determinants):
            self.assertEqual(array.dtype, np.float32)
        self.assertTrue(encoded.valid.all())
        np.testing.assert_allclose(encoded.deformation, np.broadcast_to(affine, encoded.deformation.shape), atol=2e-6)
        np.testing.assert_allclose(encoded.frames @ encoded.local_axes, encoded.deformation, atol=2e-6)
        with self.assertRaises(TypeError):
            compute_frames_float32(rest, positions.astype(np.float64))

    def test_solver_recurrence_stays_float32(self):
        """Trace every observed LSQR x/u/v/w/dk array and verify float32 solves."""
        from scipy.sparse.linalg import lsqr

        rest = generate_cuboid((2, 2, 3), cell_size=0.025)
        decoder = Float32Decoder(rest)
        positions = decoder.rest_positions.copy()
        positions[:, 0] += np.float32(0.1) * positions[:, 2]
        encoded = compute_frames_float32(rest, positions)
        observed = {name: set() for name in ("x", "u", "v", "w", "dk")}

        def trace(frame, event, arg):
            if frame.f_code is lsqr.__code__ and event == "line":
                for name, dtypes in observed.items():
                    value = frame.f_locals.get(name)
                    if isinstance(value, np.ndarray):
                        dtypes.add(str(value.dtype))
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            recovered, displacement, rhs, diagnostics = decoder.decode(encoded, positions[decoder.fixed])
        finally:
            sys.settrace(previous)
        for name, dtypes in observed.items():
            self.assertEqual(dtypes, {"float32"}, name)
        for array in (decoder.matrix, decoder.free_matrix, rhs, displacement, recovered):
            self.assertEqual(array.dtype, np.float32)
        self.assertTrue(all(axis["stop_code"] in (0, 1, 2, 4, 5) for axis in diagnostics))
        np.testing.assert_array_equal(recovered[decoder.fixed], positions[decoder.fixed])
        metrics = analyze_result(decoder, positions, encoded, recovered, displacement, rhs)
        self.assertLess(metrics["equation_component_rms"], 1e-5)
        self.assertGreater(decoder.matvec_calls, 0)
        self.assertGreater(decoder.rmatvec_calls, 0)

    def test_null_diagnostic_is_a_projection(self):
        """Keep the offline null projection orthogonal and outside the decoder."""
        rest = generate_cuboid((2, 2, 3), cell_size=0.025)
        rng = np.random.default_rng(3)
        displacement = rng.normal(size=rest.corner_rest_positions.shape) * 0.001
        decoder = Float32Decoder(rest)
        displacement[decoder.fixed] = 0
        null = project_null_component(rest, displacement)
        np.testing.assert_allclose(project_null_component(rest, null), null, atol=1e-18)
        np.testing.assert_allclose(decoder.matrix.astype(np.float64) @ null, 0, atol=1e-17)
        self.assertLess(abs(np.sum(null * (displacement - null))), 1e-20)

    def test_summary_retains_failed_seeds(self):
        samples = [
            {"seed": seed, "status": "complete", "effective_scale": 0.5, "metrics": {"corner_rmse_m": value}}
            for seed, value in enumerate((1.0, 2.0, 4.0))
        ]
        samples.append({"seed": 3, "status": "failed", "error": "Deliberate test failure"})
        summary = summarize_samples(samples)
        self.assertEqual(summary["requested_count"], 4)
        self.assertEqual(summary["complete_count"], 3)
        self.assertEqual(summary["failed_seeds"], [3])
        metric = summary["statistics"]["corner_rmse_m"]
        self.assertEqual(metric["median"], 2)
        self.assertAlmostEqual(metric["mean"], 7 / 3)
        self.assertAlmostEqual(metric["p95"], 3.8)
        self.assertEqual(metric["worst_seed"], 2)


if __name__ == "__main__":
    unittest.main()
