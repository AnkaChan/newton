# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the optional oneMKL PARDISO fusion factorization."""

import ctypes
import gc
import importlib.metadata
import os
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse  # noqa: TID253 -- Optional sparse backend tests.

from experiments.learned_intrinsic_solver.pardiso import PardisoFactor


def _installed_runtime() -> Path | None:
    try:
        distribution = importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        return None
    return next(
        (
            Path(distribution.locate_file(file)).resolve()
            for file in distribution.files or ()
            if file.name == "libmkl_rt.so.2"
        ),
        None,
    )


class TestPardisoValidation(unittest.TestCase):
    def setUp(self):
        """Construct a small full-matrix sparse input."""
        self.matrix = sparse.csr_matrix(np.array([[5.0, 2.0], [1.0, 4.0]], dtype=np.float32))

    def test_explicit_missing_runtime_does_not_fall_back(self):
        """Reject an explicit missing backend without trying another library."""
        missing = Path("/missing/newton-pardiso-test/libmkl_rt.so.2")
        with self.assertRaisesRegex(ImportError, "requirements-pardiso.txt"):
            PardisoFactor(self.matrix, library_path=missing)

    def test_invalid_mkl_rt_override_does_not_fall_back(self):
        """Reject a bad MKL_RT override even when an installed runtime exists."""
        with mock.patch.dict(os.environ, {"MKL_RT": "/missing/newton-pardiso-test/libmkl_rt.so.2"}):
            with self.assertRaisesRegex(ImportError, "MKL_RT"):
                PardisoFactor(self.matrix)

    def test_bad_matrices_are_rejected(self):
        """Reject malformed, nonfinite, or unsupported sparse matrices."""
        cases = (
            sparse.csr_matrix(np.ones((2, 3), dtype=np.float32)),
            sparse.csr_matrix((0, 0), dtype=np.float32),
            sparse.csr_matrix(np.array([[np.nan, 0], [0, 1]], dtype=np.float32)),
            sparse.csr_matrix(np.eye(2, dtype=np.int32)),
        )
        for matrix in cases:
            with self.subTest(shape=matrix.shape, dtype=matrix.dtype), self.assertRaises((TypeError, ValueError)):
                PardisoFactor(matrix, library_path="/missing/newton-pardiso-test/libmkl_rt.so.2")


@unittest.skipUnless(_installed_runtime() is not None, "optional oneMKL runtime is not installed")
class TestPardisoFactor(unittest.TestCase):
    def setUp(self):
        """Prepare a nonsymmetric matrix and independent dense reference."""
        self.matrix = np.array([[5.0, 2.0, 0.0], [1.0, 4.0, 1.0], [0.0, 2.0, 3.0]], dtype=np.float32)
        self.rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    def test_float32_multirhs_forward_and_transpose(self):
        """Preserve float32 and solve a full nonsymmetric matrix both ways."""
        with PardisoFactor(sparse.csr_matrix(self.matrix), threads=2) as factor:
            self.assertEqual(factor.dtype, np.dtype(np.float32))
            self.assertEqual(factor.mkl_max_threads, 2)
            with mock.patch.object(factor, "_call", wraps=factor._call) as calls:
                for transpose in (False, True):
                    result = factor.solve(self.rhs, transpose=transpose)
                    expected = np.linalg.solve(self.matrix.T if transpose else self.matrix, self.rhs)
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=1e-6)
                self.assertEqual([call.args[0] for call in calls.call_args_list], [33, 33])

    def test_float64_reference_and_repeated_cached_solve(self):
        """Reuse one factor for repeated vector and matrix solves in float64."""
        matrix = self.matrix.astype(np.float64)
        rhs = self.rhs.astype(np.float64)
        with PardisoFactor(sparse.csr_matrix(matrix), threads=2) as factor:
            with mock.patch.object(factor, "_call", wraps=factor._call) as calls:
                for _ in range(3):
                    np.testing.assert_allclose(factor.solve(rhs[:, 0]), np.linalg.solve(matrix, rhs[:, 0]), atol=1e-12)
                self.assertEqual([call.args[0] for call in calls.call_args_list], [33, 33, 33])
            self.assertEqual(factor.dtype, np.dtype(np.float64))

    def test_rhs_validation_and_close(self):
        """Reject wrong RHS inputs and make close idempotent."""
        factor = PardisoFactor(sparse.csr_matrix(self.matrix), threads=2)
        for rhs in (self.rhs.astype(np.float64), self.rhs[:2], np.full((3, 1), np.nan, dtype=np.float32)):
            with self.subTest(shape=rhs.shape, dtype=rhs.dtype), self.assertRaises(ValueError):
                factor.solve(rhs)
        with self.assertRaises(TypeError):
            factor.solve(self.rhs, transpose="T")
        factor.close()
        factor.close()
        with self.assertRaises(RuntimeError):
            factor.solve(self.rhs)

    def test_scoped_thread_count_restores_previous_setting(self):
        """Restore the caller's local MKL thread count after factor and solve."""
        # Match backend setup even when this test is run alone, before the
        # first native query can initialize oneMKL's OpenMP runtime.
        environment = mock.patch.dict(os.environ, {"MKL_THREADING_LAYER": "GNU"})
        environment.start()
        self.addCleanup(environment.stop)
        lib = ctypes.CDLL(str(_installed_runtime()))
        get_threads = lib.MKL_Get_Max_Threads
        get_threads.argtypes = []
        get_threads.restype = ctypes.c_int
        before = get_threads()
        with PardisoFactor(sparse.csr_matrix(self.matrix), threads=2) as factor:
            self.assertEqual(factor.mkl_max_threads, 2)
            self.assertEqual(get_threads(), before)
            factor.solve(self.rhs)
            self.assertEqual(get_threads(), before)
        self.assertEqual(get_threads(), before)

    def test_gc_releases_factor(self):
        """Release a factor when its last Python reference is collected."""
        released = []
        original = PardisoFactor._call

        def record(instance, phase, rhs, result):
            if phase == -1:
                released.append(id(instance))
            return original(instance, phase, rhs, result)

        with mock.patch.object(PardisoFactor, "_call", record):
            factor = PardisoFactor(sparse.csr_matrix(self.matrix), threads=2)
            identifier = id(factor)
            reference = weakref.ref(factor)
            del factor
            gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(released.count(identifier), 1)


if __name__ == "__main__":
    unittest.main()
