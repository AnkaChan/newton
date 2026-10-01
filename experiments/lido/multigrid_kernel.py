# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp CSR kernels for the multigrid cycle (design spec section 10, "Canonical-cube meshes of arbitrary shape").

The right-hand sides of the fusion solve come in triples (object, axis), so every kernel works on [n, k] arrays of
3-vectors (k = columns / 3) with one thread per (row, triple): the three threads of a row that share a 128-byte
line of the matrix broadcast its loads, the vector gathers are 12-byte contiguous. Fusing the sparse product with
the vector updates around it makes one smoother step one launch:

- `spmm`: out = alpha A x + beta y (the PCG product A p, the restriction P^T r, the prolongation x + P e, with
  beta = 0 the y operand is not read);
- `jacobi`: r = b - A x, d = D^-1 r / theta, x += d (the first Chebyshev step from a nonzero iterate);
- `cheb`: r -= A d, d = c1 d + c2 D^-1 r, x += d (the following Chebyshev steps; r is the residual of the step's
  input iterate, the smoothed iterate's residual for the coarse level is one more `spmm`);
- `residual`: R = B - A64 X in float64 together with its float32 copy (the PCG residual recomputed exactly);
- `axpy`: X += alpha_j p (float64 iterate, float32 direction and column scalars).

cuSPARSE's CSR SpMM measured 1.06 ms (fp32) / 4.3 ms (fp64) for 27M nonzeros and 3 columns on an idle L40 against
0.42 / 0.84 ms for these kernels, and 0.081 against 0.017 ms at 2.1M nonzeros. Kernels are generated per (float64,
float32) pair and cached, each in its own Warp module: adding an instantiation to a shared module would rebuild and
reload it, and a CUDA graph recorded with the previous build then faults on replay. Launches go on the calling torch
stream, so they record into CUDA graphs as they are.
"""

import torch
import warp as wp

Tensor = torch.Tensor

_WP_DTYPE = {torch.float32: wp.float32, torch.float64: wp.float64}
_kernels: dict[tuple, tuple] = {}


def kernels(T_hi, T_lo) -> tuple:
    """(spmm, jacobi, cheb, residual, axpy, V_lo, V_hi) for the working scalar type T_lo and the residual type T_hi."""
    key = (T_hi, T_lo)
    if key in _kernels:
        return _kernels[key]
    V = wp.types.vector(length=3, dtype=T_lo)
    Vh = wp.types.vector(length=3, dtype=T_hi)

    @wp.kernel(module="unique", enable_backward=False)
    def spmm_kernel(
        crow: wp.array[wp.int32],
        col: wp.array[wp.int32],
        val: wp.array[T_lo],
        x: wp.array2d[V],  # [cols(A), k]
        y: wp.array2d[V],  # [rows(A), k], read only when beta != 0
        alpha: T_lo,
        beta: T_lo,
        k: int,
        out: wp.array2d[V],  # [rows(A), k]
    ):
        tid = wp.tid()
        i = tid // k
        j = tid - i * k
        acc = V()
        for e in range(crow[i], crow[i + 1]):
            acc += val[e] * x[col[e], j]
        if beta != T_lo(0.0):
            acc = alpha * acc + beta * y[i, j]
        else:
            acc = alpha * acc
        out[i, j] = acc

    @wp.kernel(module="unique", enable_backward=False)
    def jacobi_kernel(
        crow: wp.array[wp.int32],
        col: wp.array[wp.int32],
        val: wp.array[T_lo],
        b: wp.array2d[V],
        x: wp.array2d[V],
        dinv: wp.array[T_lo],  # D^-1 / theta
        k: int,
        r_out: wp.array2d[V],
        d_out: wp.array2d[V],
        x_out: wp.array2d[V],
    ):
        tid = wp.tid()
        i = tid // k
        j = tid - i * k
        acc = V()
        for e in range(crow[i], crow[i + 1]):
            acc += val[e] * x[col[e], j]
        r = b[i, j] - acc
        d = dinv[i] * r
        r_out[i, j] = r
        d_out[i, j] = d
        x_out[i, j] = x[i, j] + d

    @wp.kernel(module="unique", enable_backward=False)
    def cheb_kernel(
        crow: wp.array[wp.int32],
        col: wp.array[wp.int32],
        val: wp.array[T_lo],
        r: wp.array2d[V],
        d: wp.array2d[V],
        x: wp.array2d[V],
        dinv: wp.array[T_lo],  # c2 D^-1
        c1: T_lo,
        k: int,
        r_out: wp.array2d[V],
        d_out: wp.array2d[V],
        x_out: wp.array2d[V],
    ):
        tid = wp.tid()
        i = tid // k
        j = tid - i * k
        acc = V()
        for e in range(crow[i], crow[i + 1]):
            acc += val[e] * d[col[e], j]
        rn = r[i, j] - acc
        dn = c1 * d[i, j] + dinv[i] * rn
        r_out[i, j] = rn
        d_out[i, j] = dn
        x_out[i, j] = x[i, j] + dn

    @wp.kernel(module="unique", enable_backward=False)
    def residual_kernel(
        crow: wp.array[wp.int32],
        col: wp.array[wp.int32],
        val: wp.array[T_hi],
        b: wp.array2d[Vh],
        x: wp.array2d[Vh],
        k: int,
        r_out: wp.array2d[Vh],
        rw_out: wp.array2d[V],
    ):
        tid = wp.tid()
        i = tid // k
        j = tid - i * k
        acc = Vh()
        for e in range(crow[i], crow[i + 1]):
            acc += val[e] * x[col[e], j]
        res = b[i, j] - acc
        r_out[i, j] = res
        rw_out[i, j] = V(T_lo(res[0]), T_lo(res[1]), T_lo(res[2]))

    @wp.kernel(module="unique", enable_backward=False)
    def axpy_kernel(
        x: wp.array2d[Vh],  # updated in place
        p: wp.array2d[V],
        alpha: wp.array[V],  # [k]: one scalar per column
        k: int,
    ):
        tid = wp.tid()
        i = tid // k
        j = tid - i * k
        a = alpha[j]
        q = p[i, j]
        x[i, j] = x[i, j] + Vh(T_hi(a[0]) * T_hi(q[0]), T_hi(a[1]) * T_hi(q[1]), T_hi(a[2]) * T_hi(q[2]))

    _kernels[key] = (spmm_kernel, jacobi_kernel, cheb_kernel, residual_kernel, axpy_kernel, V, Vh)
    return _kernels[key]


def csr_arrays(A: Tensor, T) -> tuple:
    """Descriptor triple (crow, col, val) of an int32 CSR tensor for the Warp kernels (no copies)."""
    return (
        wp.from_torch(A.crow_indices(), dtype=wp.int32, requires_grad=False, return_ctype=True),
        wp.from_torch(A.col_indices(), dtype=wp.int32, requires_grad=False, return_ctype=True),
        wp.from_torch(A.values(), dtype=T, requires_grad=False, return_ctype=True),
    )


class WarpOps:
    """The cycle's vector operations on CUDA for right-hand sides with a multiple of three columns.

    `dtype` is the working precision of the hierarchy, `hi` that of the recomputed residual (float64 or `dtype`)."""

    name = "warp"

    def __init__(self, dtype, hi, device: torch.device):
        wp.init()
        self.dtype, self.hi, self.device = dtype, hi, device
        self.T_lo, self.T_hi = _WP_DTYPE[dtype], _WP_DTYPE[hi]
        self._spmm, self._jacobi, self._cheb, self._residual, self._axpy, self.V, self.Vh = kernels(
            self.T_hi, self.T_lo
        )

    @staticmethod
    def accepts(cols: int) -> bool:
        return cols % 3 == 0

    def csr(self, A: Tensor) -> tuple:
        return csr_arrays(A, _WP_DTYPE[A.dtype])

    def _launch(self, kernel, dim: int, inputs: list) -> None:
        wp.launch(kernel, dim=dim, inputs=inputs, stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def _vecs(self, t: Tensor, V) -> wp.array:
        n, m = t.shape
        return wp.from_torch(t.view(n, m // 3, 3), dtype=V, requires_grad=False, return_ctype=True)

    def _scalars(self, t: Tensor, T) -> wp.array:
        return wp.from_torch(t, dtype=T, requires_grad=False, return_ctype=True)

    def spmm(self, csr: tuple, rows: int, x: Tensor, y: Tensor | None, alpha: float, beta: float) -> Tensor:
        """alpha A x + beta y, A given by its descriptors and row count; y unread (may be None) when beta == 0."""
        k = x.shape[1] // 3
        out = torch.empty(rows, x.shape[1], dtype=x.dtype, device=x.device)
        yv = self._vecs(out if y is None else y, self.V)
        self._launch(
            self._spmm,
            rows * k,
            [*csr, self._vecs(x, self.V), yv, self.T_lo(alpha), self.T_lo(beta), k, self._vecs(out, self.V)],
        )
        return out

    def jacobi(self, csr: tuple, b: Tensor, x: Tensor, dinv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(r, d, x) with r = b - A x, d = dinv r, x + d."""
        k = b.shape[1] // 3
        r, d, xn = torch.empty_like(b), torch.empty_like(b), torch.empty_like(b)
        v = self._vecs
        self._launch(
            self._jacobi,
            b.shape[0] * k,
            [
                *csr,
                v(b, self.V),
                v(x, self.V),
                self._scalars(dinv, self.T_lo),
                k,
                v(r, self.V),
                v(d, self.V),
                v(xn, self.V),
            ],
        )
        return r, d, xn

    def cheb(self, csr: tuple, r: Tensor, d: Tensor, x: Tensor, dinv: Tensor, c1: float) -> tuple:
        """(r, d, x) with r - A d, c1 d + dinv r, x + d."""
        k = r.shape[1] // 3
        rn, dn, xn = torch.empty_like(r), torch.empty_like(r), torch.empty_like(r)
        v = self._vecs
        self._launch(
            self._cheb,
            r.shape[0] * k,
            [
                *csr,
                v(r, self.V),
                v(d, self.V),
                v(x, self.V),
                self._scalars(dinv, self.T_lo),
                self.T_lo(c1),
                k,
                v(rn, self.V),
                v(dn, self.V),
                v(xn, self.V),
            ],
        )
        return rn, dn, xn

    def residual(self, csr_hi: tuple, B: Tensor, X: Tensor) -> tuple[Tensor, Tensor]:
        """(R, R in the working dtype) with R = B - A_hi X."""
        k = B.shape[1] // 3
        R = torch.empty_like(B)
        Rw = torch.empty(B.shape, dtype=self.dtype, device=B.device)
        v = self._vecs
        self._launch(
            self._residual, B.shape[0] * k, [*csr_hi, v(B, self.Vh), v(X, self.Vh), k, v(R, self.Vh), v(Rw, self.V)]
        )
        return R, Rw

    def axpy(self, X: Tensor, P: Tensor, alpha: Tensor) -> Tensor:
        """X + alpha_j P_j column by column with X in the residual precision (in place when the precisions differ)."""
        if X.dtype == P.dtype:
            return torch.addcmul(X, P, alpha)
        k = P.shape[1] // 3
        a = wp.from_torch(alpha.contiguous().view(k, 3), dtype=self.V, requires_grad=False, return_ctype=True)
        self._launch(self._axpy, P.shape[0] * k, [self._vecs(X, self.Vh), self._vecs(P, self.V), a, k])
        return X
