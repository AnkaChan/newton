# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental cached oneMKL PARDISO factor for CPU hex fusion solves.

The optional oneMKL runtime is loaded only when a factor is constructed.
Float32 matrices, factors, right-hand sides, and adjoints stay in float32.
This LP64 bridge is local to the experimental learned intrinsic solver.
"""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import importlib.metadata
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import sparse  # noqa: TID253 -- Optional backend is imported only when selected.

__all__ = ["PardisoFactor"]

_INT32_MAX = np.iinfo(np.int32).max
_C_INT_P = ct.POINTER(ct.c_int32)
_INSTALL = "uv pip install -r experiments/learned_intrinsic_solver/requirements-pardiso.txt"


def _library_path(library_path: str | os.PathLike[str] | None) -> str:
    """Find an explicit or optional installed oneMKL runtime without fallback."""
    requested = library_path if library_path is not None else os.environ.get("MKL_RT")
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise ImportError(f"oneMKL runtime requested via library_path/MKL_RT does not exist: {path}; {_INSTALL}")
        return str(path)
    try:
        distribution = importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        for record in distribution.files or ():
            if record.name in ("libmkl_rt.so.2", "libmkl_rt.so", "mkl_rt.dll"):
                path = Path(distribution.locate_file(record)).resolve()
                if path.is_file():
                    return str(path)
    for name in ("libmkl_rt.so.2", "libmkl_rt.so", "mkl_rt.dll"):
        path = Path(sys.prefix) / "lib" / name
        if path.is_file():
            return str(path.resolve())
    found = ctypes.util.find_library("mkl_rt")
    if found:
        return found
    raise ImportError(f"oneMKL runtime not found; install the optional backend with `{_INSTALL}` or set MKL_RT")


def _load_runtime(library_path: str | os.PathLike[str] | None):
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        if os.environ["MKL_THREADING_LAYER"].upper() != "GNU":
            raise RuntimeError("oneMKL PARDISO requires MKL_THREADING_LAYER=GNU with this Torch/OpenMP build")
    if "ILP64" in os.environ.get("MKL_INTERFACE_LAYER", "").upper():
        raise RuntimeError("PardisoFactor requires the LP64 oneMKL interface")
    if ct.sizeof(ct.c_void_p) != 8:
        raise RuntimeError("PardisoFactor requires a 64-bit process")
    path = _library_path(library_path)
    try:
        return ct.CDLL(path)
    except OSError as error:
        raise ImportError(f"cannot load oneMKL runtime {path}: {error}; install with `{_INSTALL}`") from error


class PardisoFactor:
    """Cache one full-matrix real PARDISO LU factorization and its adjoint.

    Experimental: this class supports float32 and float64 SciPy sparse matrices
    and one or multiple NumPy right-hand sides. Factors stay on CPU. A single
    handle is serialized across threads; construct a new factor after forking.
    ``close()`` releases native memory promptly, and garbage collection is a
    fallback. Sparse matrix values and topology are copied at construction.

    Args:
        matrix: Nonempty square SciPy sparse real matrix.
        library_path: Optional explicit oneMKL runtime path. ``MKL_RT`` has
            the same role when this is omitted; neither permits fallback.
        threads: Scoped oneMKL thread count. Defaults to ``MKL_NUM_THREADS``
            when set, otherwise 30; the prior thread-local setting is restored.
    """

    def __init__(self, matrix, *, library_path=None, threads: int | None = None):
        if threads is None:
            configured = os.environ.get("MKL_NUM_THREADS")
            try:
                threads = int(configured) if configured is not None else 30
            except ValueError as error:
                raise ValueError("MKL_NUM_THREADS must be a positive integer") from error
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if not sparse.issparse(matrix) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            raise ValueError("matrix must be a nonempty square SciPy sparse matrix")
        if matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("matrix must be float32 or float64")
        if matrix.shape[0] > _INT32_MAX or matrix.nnz > _INT32_MAX:
            raise ValueError("matrix exceeds oneMKL LP64 limits")
        csr = matrix.tocsr(copy=True)
        csr.sum_duplicates()
        csr.sort_indices()
        if not np.isfinite(csr.data).all():
            raise ValueError("matrix must have finite entries")

        self.dtype = csr.dtype
        self.n = int(csr.shape[0])
        self.threads = threads
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        self._factor_attempted = False
        self._data = np.ascontiguousarray(csr.data)
        self._indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
        self._indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
        self._perm = np.zeros(self.n, dtype=np.int32)
        self._pt = (ct.c_void_p * 64)()
        self._iparm = (ct.c_int32 * 64)()
        self._lib = _load_runtime(library_path)
        self._pardisoinit = self._lib.pardisoinit
        self._pardisoinit.argtypes = [ct.POINTER(ct.c_void_p), _C_INT_P, _C_INT_P]
        self._pardisoinit.restype = None
        self._pardiso = self._lib.pardiso
        self._pardiso.argtypes = [
            ct.POINTER(ct.c_void_p),
            _C_INT_P,
            _C_INT_P,
            _C_INT_P,
            _C_INT_P,
            _C_INT_P,
            ct.c_void_p,
            _C_INT_P,
            _C_INT_P,
            _C_INT_P,
            _C_INT_P,
            _C_INT_P,
            _C_INT_P,
            ct.c_void_p,
            ct.c_void_p,
            _C_INT_P,
        ]
        self._pardiso.restype = None
        # The uppercase spelling is the C int-by-value API. The lowercase
        # symbol is a Fortran pointer-argument ABI and segfaults via ctypes.
        self._set_threads = self._lib.MKL_Set_Num_Threads_Local
        self._set_threads.argtypes = [ct.c_int]
        self._set_threads.restype = ct.c_int
        self._get_threads = self._lib.MKL_Get_Max_Threads
        self._get_threads.argtypes = []
        self._get_threads.restype = ct.c_int
        with self._scoped_threads():
            self.mkl_max_threads = self._get_threads()
            self._pardisoinit(self._pt, ct.byref(ct.c_int32(11)), self._iparm)
        self._iparm[0] = 1  # Preserve pardisoinit's matrix-type defaults.
        self._iparm[11] = 0  # A, not A.T, for the first solve.
        self._iparm[26] = 1  # Validate the CSR structure in the analysis phase.
        self._iparm[27] = 1 if self.dtype == np.float32 else 0
        self._iparm[34] = 1  # SciPy CSR is zero-based.
        dummy = np.zeros((self.n, 1), dtype=self.dtype, order="F")
        self._factor_attempted = True
        try:
            self._call(12, dummy, dummy.copy(order="F"))
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    @contextmanager
    def _scoped_threads(self):
        previous = self._set_threads(self.threads)
        try:
            yield
        finally:
            self._set_threads(previous)

    def _call(self, phase: int, rhs: np.ndarray, result: np.ndarray) -> None:
        error = ct.c_int32()
        with self._scoped_threads():
            self._pardiso(
                self._pt,
                ct.byref(ct.c_int32(1)),
                ct.byref(ct.c_int32(1)),
                ct.byref(ct.c_int32(11)),
                ct.byref(ct.c_int32(phase)),
                ct.byref(ct.c_int32(self.n)),
                ct.c_void_p(self._data.ctypes.data),
                self._indptr.ctypes.data_as(_C_INT_P),
                self._indices.ctypes.data_as(_C_INT_P),
                self._perm.ctypes.data_as(_C_INT_P),
                ct.byref(ct.c_int32(rhs.shape[1])),
                self._iparm,
                ct.byref(ct.c_int32(0)),
                ct.c_void_p(rhs.ctypes.data),
                ct.c_void_p(result.ctypes.data),
                ct.byref(error),
            )
        if error.value:
            raise RuntimeError(f"oneMKL PARDISO phase {phase} failed with error {error.value}")

    def solve(self, rhs: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Solve ``A X = rhs`` or ``A.T X = rhs`` with cached factors.

        Args:
            rhs: NumPy vector [N] or column-major-packed matrix [N, R].
                The input precision must match the factor precision.
            transpose: Use the exact transposed full matrix for adjoints.

        Returns:
            One vector [N] or matrix [N, R], preserving input precision.
        """
        if os.getpid() != self._pid:
            raise RuntimeError("PardisoFactor cannot be reused after fork; construct a new factor")
        if not isinstance(transpose, bool):
            raise TypeError("transpose must be boolean")
        with self._lock:
            if self._closed:
                raise RuntimeError("PardisoFactor is closed")
            values = np.asarray(rhs)
            if values.dtype != self.dtype or values.ndim not in (1, 2) or values.shape[0] != self.n:
                raise ValueError(f"rhs must be {self.dtype} with shape [{self.n}] or [{self.n},R]")
            if values.ndim == 2 and (values.shape[1] < 1 or values.shape[1] > _INT32_MAX):
                raise ValueError("rhs must have a positive LP64-compatible column count")
            if not np.isfinite(values).all():
                raise ValueError("rhs must be finite")
            vector = values.ndim == 1
            packed = np.asfortranarray(values[:, None] if vector else values)
            result = np.empty_like(packed, order="F")
            self._iparm[11] = 2 if transpose else 0
            self._call(33, packed, result)
            return result[:, 0].copy() if vector else result

    def close(self) -> None:
        """Release the native factor once; child processes never touch it."""
        if getattr(self, "_closed", True):
            return
        if os.getpid() != self._pid:
            self._closed = True
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._factor_attempted:
                dummy = np.zeros((self.n, 1), dtype=self.dtype, order="F")
                self._call(-1, dummy, dummy.copy(order="F"))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
