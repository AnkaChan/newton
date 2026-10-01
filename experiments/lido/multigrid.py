# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometric multigrid preconditioned conjugate gradient for the fusion matrix on voxel grids (design spec section 10,
"Canonical-cube meshes of arbitrary shape").

Level 0 is K_s[free, free] (`fusion.assemble_free_csr`). The hierarchy lives on the voxel lattice: a coarse voxel exists
where any of its 2x2x2 fine voxels exists, the coarse corners are the lattice corners of the coarse voxels, the
prolongation is trilinear interpolation on the lattice (weights 1, 1/2, 1/4, 1/8 by the parity of the fine corner;
every stencil node is a corner of the coarse voxel containing a fine voxel of the fine corner, so it exists) with
pinned fine corners left out (zero Dirichlet rows) and coarse corners with an empty column removed; the restriction
is the transpose and the coarse operators are the Galerkin products P^T A P, built once in setup until the coarsest
level has at most `coarse_size` unknowns, where a dense inverse is stored. The smoother is a Chebyshev-Jacobi
polynomial (degree 2 by default; the largest eigenvalue of D^-1 A from a few power iterations at setup, the smoothing
interval [lmax / eig_ratio, lmax] with eig_ratio 4: on carved shapes, plates, rods and sparse pin sets the PCG takes
5-9 iterations to 1e-8 in float64 and 4-5 to 1e-5 in float32 whatever the degree and ratio, so the cheapest cycle
wins), which has no sequential dependence and makes the V-cycle a symmetric preconditioner, so all 3n right-hand sides
are solved at once by a vectorised conjugate gradient with one set of scalars per column.

A float32 iterative solve has an accuracy floor of about eps x cond(K_ff) whatever the tolerance (7.6e-4 relative
error on a 100^3 box, 1.8e-4 on a carved 50^3, where a direct float32 solve reaches 5e-7), so by default the residual
is recomputed in float64 from a float64 copy of the level-0 operator (`fp64_residual`, every `fp64_every` iterations,
the float32 recursive update in between: residual replacement), which brings the error to the requested tolerance.

Execution (the method above is fixed). By default the solve runs a fixed number of PCG iterations (`iterations`, 8:
the float64 relative residual after 4 / 6 / 8 / 10 iterations is at worst 5.0e-5 / 1.0e-7 / 7.7e-10 / 2.3e-12 over
boxes, plates, rods, slabs with sparse pins and carved shapes from 27 to a million free corners, so 8 leaves four
orders of magnitude to the 1e-5 requirement and 6 would leave two; the float32 result is within 3e-8 of the float64
solve) with no host synchronisation and no data-dependent control flow, so it records into a CUDA graph: inside a
capture (the inference query, `capture.CapturedQuery`) it is recorded as part of the outer graph; otherwise `solve`
records one graph per right-hand-side shape on first use and replays it afterwards (the autograd backward, the same
solve, replays it from the autograd thread). `iterations=None` selects the adaptive path (stop at `rtol`, at most
`max_iter`, one host sync per iteration; `stats` reports the count). The cycle's vector operations are Warp CSR
kernels fused with the surrounding updates on CUDA (`multigrid_kernel`: one launch per smoother step, one per
residual, restriction and prolongation, a float64 residual kernel that also writes the float32 copy) and torch
sparse kernels on the CPU: 7 launches per level, 32 kernels per PCG iteration on a 3-level hierarchy and 54 on 6
levels (108 and 212 with cuSPARSE and unfused torch ops). Measured on an idle L40 (3 right-hand sides, captured):
0.7 ms at 4-8.5k free corners, 1.9 ms at 78k, 19 ms at 596k, 33 ms at 1.02M (cuDSS 0.22-0.37, 1.0, 6.4, 17 ms; the
eager adaptive solve before this: 3.4-4.7, 9.7, 32, 33 ms); details in the design spec, section 10.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch

from .fusion import _SparseSolve, apply_cell_blocks, assemble_free_csr, cell_block
from .grid import Grid
from .hex import CORNER_OFFSETS

Tensor = torch.Tensor


def _to_csr(coo: Tensor) -> Tensor:
    """Coalesced COO -> CSR with 32-bit indices (half the index memory; cuSPARSE, the CPU kernel and the Warp kernels
    take them)."""
    csr = coo.coalesce().to_sparse_csr()
    return torch.sparse_csr_tensor(
        csr.crow_indices().to(torch.int32), csr.col_indices().to(torch.int32), csr.values(), csr.shape
    )


def _diagonal(coo: Tensor) -> Tensor:
    idx = coo.indices()
    on_diag = idx[0] == idx[1]
    return torch.zeros(coo.shape[0], dtype=coo.dtype, device=coo.device).index_put_(
        (idx[0][on_diag],), coo.values()[on_diag]
    )


def _linear(lat: Tensor, shape: tuple) -> Tensor:
    return (lat[:, 0] * shape[1] + lat[:, 1]) * shape[2] + lat[:, 2]


def _unravel(ids: Tensor, shape: tuple) -> Tensor:
    z = ids % shape[2]
    y = (ids // shape[2]) % shape[1]
    x = ids // (shape[1] * shape[2])
    return torch.stack([x, y, z], 1)


def coarsen(voxels: Tensor, corners: Tensor, shape: tuple, dtype) -> tuple[Tensor, Tensor, Tensor, tuple]:
    """One coarsening step on the lattice.

    voxels [C,3]: lattice coordinates of the present cells of this level (lattice shape `shape`); corners [n,3]: lattice
    coordinates of this level's unknowns. Returns (P as a coalesced COO [n, nc], coarse voxels [Cc,3], coarse corner
    lattice coordinates [nc,3], coarse lattice shape). Trilinear weights arise from the eight half-steps (x + s) // 2,
    s in {0,1}^3, each worth 1/8: an even coordinate sends both half-steps to the same node (weight 1), an odd one
    splits them (1/2 each).
    """
    dev = voxels.device
    cshape = tuple((int(s) + 1) // 2 for s in shape)
    cnode_shape = tuple(s + 1 for s in cshape)
    cvox = _unravel(torch.unique(_linear(voxels // 2, cshape)), cshape)
    ccorner_ids = torch.unique(_linear((cvox[:, None, :] + CORNER_OFFSETS.to(dev)[None]).reshape(-1, 3), cnode_shape))
    pos = torch.full((int(torch.tensor(cnode_shape).prod()),), -1, dtype=torch.int64, device=dev)
    pos[ccorner_ids] = torch.arange(ccorner_ids.numel(), device=dev)
    n = corners.shape[0]
    half = (corners[:, None, :] + CORNER_OFFSETS.to(dev)[None]) // 2  # [n,8,3]
    col = pos[_linear(half.reshape(-1, 3), cnode_shape)]
    if bool((col < 0).any()):
        raise RuntimeError("multigrid: a trilinear stencil node is missing on the coarse level")
    row = torch.arange(n, device=dev)[:, None].expand(n, 8).reshape(-1)
    P = torch.sparse_coo_tensor(
        torch.stack([row, col]), torch.full((n * 8,), 1.0 / 8.0, dtype=dtype, device=dev), (n, ccorner_ids.numel())
    ).coalesce()
    used = torch.bincount(P.indices()[1], minlength=ccorner_ids.numel()) > 0
    renum = torch.full((ccorner_ids.numel(),), -1, dtype=torch.int64, device=dev)
    renum[used] = torch.arange(int(used.sum()), device=dev)
    idx = P.indices()
    P = torch.sparse_coo_tensor(torch.stack([idx[0], renum[idx[1]]]), P.values(), (n, int(used.sum()))).coalesce()
    return P, cvox, _unravel(ccorner_ids[used], cnode_shape), cshape


class TorchOps:
    """The cycle's vector operations with torch sparse kernels: the CPU, and CUDA right-hand sides whose column count
    is not a multiple of three. Operator handles are the CSR tensors themselves."""

    name = "torch"

    def __init__(self, dtype, hi):
        self.dtype, self.hi = dtype, hi

    @staticmethod
    def accepts(cols: int) -> bool:
        return True

    def csr(self, A: Tensor) -> Tensor:
        return A

    def spmm(self, A: Tensor, rows: int, x: Tensor, y: Tensor | None, alpha: float, beta: float) -> Tensor:
        """alpha A x + beta y (y unread when beta == 0)."""
        if beta == 0.0:
            out = torch.sparse.mm(A, x)
            return out if alpha == 1.0 else alpha * out
        return torch.sparse.addmm(y, A, x, beta=beta, alpha=alpha)

    def jacobi(self, A: Tensor, b: Tensor, x: Tensor, dinv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(r, d, x) with r = b - A x, d = dinv r, x + d."""
        r = torch.sparse.addmm(b, A, x, alpha=-1.0)
        d = dinv[:, None] * r
        return r, d, x + d

    def cheb(self, A: Tensor, r: Tensor, d: Tensor, x: Tensor, dinv: Tensor, c1: float) -> tuple:
        """(r, d, x) with r - A d, c1 d + dinv r, x + d."""
        r = torch.sparse.addmm(r, A, d, alpha=-1.0)
        d = torch.addcmul(c1 * d, dinv[:, None], r)
        return r, d, x + d

    def residual(self, A_hi: Tensor, B: Tensor, X: Tensor) -> tuple[Tensor, Tensor]:
        """(R, R in the working dtype) with R = B - A_hi X."""
        R = torch.sparse.addmm(B, A_hi, X, alpha=-1.0)
        return R, R.to(self.dtype)

    def axpy(self, X: Tensor, P: Tensor, alpha: Tensor) -> Tensor:
        """X + alpha_j P_j column by column, in the precision of X."""
        return torch.addcmul(X, P.to(X.dtype), alpha.to(X.dtype))


@dataclass
class Level:
    A: Tensor  # CSR [n,n]
    dinv_a: Tensor  # [n] D^-1 / theta: the first Chebyshev step
    cheb: list  # [(c1, c2 D^-1 [n])] for the Chebyshev steps 2..degree
    P: Tensor | None = None  # CSR [n, nc]
    PT: Tensor | None = None  # CSR [nc, n]
    inv: Tensor | None = None  # dense [n,n] on the coarsest level
    handles: dict = field(default_factory=dict)  # ops name -> (A, P, PT) operator handles

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def nc(self) -> int:
        return int(self.P.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.A.values().numel())


class MultigridFactor:
    """Multigrid-preconditioned conjugate gradient on K_s[free, free] for any voxel grid (same interface as the other
    factors: `solve`, `K_fp`, `apply_full`).

    iterations: fixed PCG iteration count, sync-free and capturable (default 8); None for the adaptive stop at `rtol`
    within `max_iter` iterations. fp64_residual: recompute the residual in float64 every `fp64_every` iterations
    (float32 recursive update in between). capture: on CUDA with fixed iterations, `solve` records one CUDA graph per
    right-hand-side shape and replays it (default on; never while another capture is in progress, where the solve is
    recorded into that graph). `stats` describes the last solve."""

    def __init__(
        self,
        grid: Grid,
        dtype=torch.float32,
        iterations: int | None = 8,
        rtol: float = 1e-5,
        max_iter: int = 50,
        degree: int = 2,
        eig_ratio: float = 4.0,
        coarse_size: int = 400,
        power_iters: int = 10,
        max_levels: int = 12,
        fp64_residual: bool = True,
        fp64_every: int = 1,
        capture: bool | None = None,
    ):
        if grid.voxel_index is None or grid.corner_lattice is None:
            raise ValueError("MultigridFactor needs the lattice coordinates (Grid.build or Grid.from_voxels)")
        if iterations is not None and iterations < 1:
            raise ValueError("iterations must be at least 1 (or None for the adaptive stop)")
        if fp64_every < 1:
            raise ValueError("fp64_every must be at least 1")
        self.grid = grid
        self.free = grid.free  # the corner set the solve covers (all but the reference corner for an unpinned body)
        self.dtype = dtype
        self.iterations = iterations
        self.rtol = rtol
        self.max_iter = max_iter
        self.degree = degree
        self.eig_ratio = eig_ratio
        self.power_iters = power_iters
        self.fp64_residual = fp64_residual and dtype != torch.float64
        self.fp64_every = fp64_every
        self.hi = torch.float64 if self.fp64_residual else dtype
        on_cuda = grid.device.type == "cuda"
        self.capture = (on_cuda if capture is None else capture) and iterations is not None
        if self.capture and not on_cuda:
            raise ValueError("CUDA-graph capture needs a CUDA grid")
        self._A = cell_block().to(device=grid.device, dtype=dtype)
        self.levels: list[Level] = []
        self._A64: Tensor | None = None
        self._h64: dict = {}  # ops name -> handle of the float64 level-0 operator
        self._torch = TorchOps(dtype, self.hi)
        self._warp = None
        if on_cuda and dtype in (torch.float32, torch.float64):
            from .multigrid_kernel import WarpOps

            self._warp = WarpOps(dtype, self.hi, grid.device)
        self._graphs: dict[tuple, tuple] = {}  # (shape, dtype) -> (graph, static rhs, static solution)
        self._stats: dict | None = None  # adaptive path
        self._last: tuple | None = None  # fixed path: (B, X) of the last solve in the residual precision
        self._build(coarse_size, max_levels)

    # ------------------------------------------------------------------ setup
    def _ops(self) -> list:
        return [ops for ops in (self._warp, self._torch) if ops is not None]

    def _build(self, coarse_size: int, max_levels: int) -> None:
        g = self.grid
        if self.fp64_residual:  # assemble once in float64, keep it for the residual, round it for the cycle
            A64 = assemble_free_csr(g, torch.float64).to_sparse_coo().coalesce()
            self._A64 = _to_csr(A64)
            self._h64 = {ops.name: ops.csr(self._A64) for ops in self._ops()}
            A = A64.to(self.dtype).coalesce()
        else:
            A = assemble_free_csr(g, self.dtype).to_sparse_coo().coalesce()
        voxels, corners, shape = g.voxel_index, g.corner_lattice[g.free], tuple(g.cell_counts)
        self.sizes: list[int] = []
        while True:
            n = int(A.shape[0])
            self.sizes.append(n)
            if n <= coarse_size or len(self.levels) + 1 >= max_levels:
                self.levels.append(self._level(A, coarsest=True))
                break
            P, voxels, corners, shape = coarsen(voxels, corners, shape, self.dtype)
            if int(P.shape[1]) >= n:
                self.levels.append(self._level(A, coarsest=True))
                break
            PT = P.t().coalesce()
            Ac = torch.sparse.mm(PT, torch.sparse.mm(A, P)).coalesce()
            Ac = (0.5 * (Ac + Ac.t())).coalesce()  # exact symmetry for the conjugate gradient
            lvl = self._level(A, coarsest=False)
            lvl.P, lvl.PT = _to_csr(P), _to_csr(PT)
            self.levels.append(lvl)
            A = Ac
        self.coarse_shape = shape
        for lvl in self.levels:
            for ops in self._ops():
                lvl.handles[ops.name] = tuple(None if t is None else ops.csr(t) for t in (lvl.A, lvl.P, lvl.PT))

    def _level(self, A_coo: Tensor, coarsest: bool) -> Level:
        dinv = 1.0 / _diagonal(A_coo)
        A = _to_csr(A_coo)
        if coarsest:
            inv = torch.linalg.inv(A_coo.to_dense().to(torch.float64)).to(self.dtype)
            return Level(A=A, dinv_a=dinv, cheb=[], inv=inv)
        lmax = 1.1 * self._largest_eigenvalue(A, dinv)
        lmin = lmax / self.eig_ratio
        theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
        sigma = theta / delta
        cheb, rho_old = [], 1.0 / sigma
        for _ in range(self.degree - 1):
            rho = 1.0 / (2.0 * sigma - rho_old)
            cheb.append((rho * rho_old, (2.0 * rho / delta) * dinv))
            rho_old = rho
        return Level(A=A, dinv_a=dinv / theta, cheb=cheb)

    def _largest_eigenvalue(self, A: Tensor, dinv: Tensor) -> float:
        gen = torch.Generator(device=A.device).manual_seed(0)
        v = torch.randn(A.shape[0], 1, dtype=self.dtype, device=A.device, generator=gen)
        v = v / v.norm()
        lam = 1.0
        for _ in range(self.power_iters):
            w = dinv[:, None] * torch.sparse.mm(A, v)
            lam = float(w.norm())
            v = w / max(lam, 1e-30)
        return lam

    def memory_bytes(self) -> int:
        total = 0
        for lvl in self.levels:
            for t in (lvl.A, lvl.P, lvl.PT, self._A64 if lvl is self.levels[0] else None):
                if t is not None:
                    total += sum(x.numel() * x.element_size() for x in (t.crow_indices(), t.col_indices(), t.values()))
            for t in (lvl.dinv_a, *(d for _, d in lvl.cheb)):
                total += t.numel() * t.element_size()
            if lvl.inv is not None:
                total += lvl.inv.numel() * lvl.inv.element_size()
        return total

    # ------------------------------------------------------------------ cycle
    def _vcycle(self, ops, l: int, b: Tensor) -> Tensor:
        """One V-cycle on level l from a zero start: Chebyshev pre-smoothing, Galerkin coarse correction, Chebyshev
        post-smoothing; the dense inverse on the coarsest level."""
        lvl = self.levels[l]
        if lvl.inv is not None:
            return lvl.inv @ b
        A, P, PT = lvl.handles[ops.name]
        d = lvl.dinv_a[:, None] * b  # first step from zero: r = b
        x, r = d, b
        for c1, dinv in lvl.cheb:
            r, d, x = ops.cheb(A, r, d, x, dinv, c1)
        r = ops.spmm(A, lvl.n, x, b, -1.0, 1.0)  # the smoothed iterate's residual (the step's r is one step behind)
        e = self._vcycle(ops, l + 1, ops.spmm(PT, lvl.nc, r, None, 1.0, 0.0))
        x = ops.spmm(P, lvl.n, e, x, 1.0, 1.0)
        r, d, x = ops.jacobi(A, b, x, lvl.dinv_a)
        for c1, dinv in lvl.cheb:
            r, d, x = ops.cheb(A, r, d, x, dinv, c1)
        return x

    def _ops_for(self, cols: int):
        return self._warp if self._warp is not None and self._warp.accepts(cols) else self._torch

    def _solve_fixed(self, rhs: Tensor) -> Tensor:
        """rhs [Pf, m] -> K_ff^-1 rhs by exactly `iterations` PCG iterations over all columns at once, no host sync."""
        ops = self._ops_for(rhs.shape[1])
        lvl0 = self.levels[0]
        A = lvl0.handles[ops.name][0]
        Rw = rhs.detach().contiguous()
        B = Rw.to(self.hi)  # a copy in float64, the right-hand side itself otherwise (never written)
        X = torch.zeros_like(B)
        Z = self._vcycle(ops, 0, Rw)
        Pd = Z
        rz = (Rw * Z).sum(0)
        for i in range(self.iterations):
            AP = ops.spmm(A, lvl0.n, Pd, None, 1.0, 0.0)
            pAp = (Pd * AP).sum(0)
            alpha = torch.where(pAp > 0, rz / pAp, 0.0)  # zero columns and converged columns stay put
            X = ops.axpy(X, Pd, alpha)
            if i == self.iterations - 1:
                break
            if self.fp64_residual and (i + 1) % self.fp64_every == 0:
                _, Rw = ops.residual(self._h64[ops.name], B, X)
            else:
                Rw = torch.addcmul(Rw, AP, alpha, value=-1.0)
            Z = self._vcycle(ops, 0, Rw)
            rz_new = (Rw * Z).sum(0)
            beta = torch.where(rz > 0, rz_new / rz, 0.0)
            Pd = torch.addcmul(Z, Pd, beta)
            rz = rz_new
        self._last = (B, X)
        return X.to(rhs.dtype)

    def _solve_adaptive(self, rhs: Tensor) -> Tensor:
        """The same PCG stopping when every column's relative residual is below `rtol` (one host sync per
        iteration), at most `max_iter` iterations, the residual recomputed exactly whenever `fp64_residual`."""
        ops = self._ops_for(rhs.shape[1])
        lvl0 = self.levels[0]
        A = lvl0.handles[ops.name][0]
        Rw = rhs.detach().contiguous()
        B = Rw.to(self.hi)
        bnorm = B.norm(dim=0)
        scale = torch.where(bnorm > 0, bnorm, torch.ones_like(bnorm))
        X = torch.zeros_like(B)
        Z = self._vcycle(ops, 0, Rw)
        Pd = Z
        rz = (Rw * Z).sum(0)
        iterations, converged, res = 0, False, bnorm.new_zeros(bnorm.shape)
        for _ in range(self.max_iter):
            iterations += 1
            AP = ops.spmm(A, lvl0.n, Pd, None, 1.0, 0.0)
            pAp = (Pd * AP).sum(0)
            alpha = torch.where(pAp > 0, rz / pAp, 0.0)
            X = ops.axpy(X, Pd, alpha)
            if self.fp64_residual:
                R, Rw = ops.residual(self._h64[ops.name], B, X)
            else:
                R = Rw = torch.addcmul(Rw, AP, alpha, value=-1.0)
            res = R.norm(dim=0) / scale
            if bool((res <= self.rtol).all()):
                converged = True
                break
            Z = self._vcycle(ops, 0, Rw)
            rz_new = (Rw * Z).sum(0)
            beta = torch.where(rz > 0, rz_new / rz, 0.0)
            Pd = torch.addcmul(Z, Pd, beta)
            rz = rz_new
        self._stats = {"iterations": iterations, "residual": float(res.max()), "converged": converged}
        if not converged:
            warnings.warn(
                f"MultigridFactor: {self.max_iter} iterations reached, relative residual {self._stats['residual']:.2e} "
                f"(rtol {self.rtol:.1e})",
                stacklevel=2,
            )
        return X.to(rhs.dtype)

    # ---------------------------------------------------------------- capture
    def _record(self, rhs: Tensor) -> tuple:
        """One warm-up solve (kernel modules, cuBLAS workspaces) then the capture of the fixed solve on a side stream;
        returns (graph, static right-hand side, static solution)."""
        dev = rhs.device
        static_in = rhs.detach().clone()
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            self._solve_fixed(static_in)
        torch.cuda.current_stream(dev).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            static_out = self._solve_fixed(static_in)
        return graph, static_in, static_out

    def _solve_cols(self, rhs: Tensor) -> Tensor:
        """rhs [Pf, m] -> K_ff^-1 rhs: adaptive, or fixed-iteration eagerly / inside an outer capture / by replay."""
        if self.iterations is None:
            return self._solve_adaptive(rhs)
        if not self.capture or torch.cuda.is_current_stream_capturing():
            return self._solve_fixed(rhs)
        key = (tuple(rhs.shape), rhs.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._graphs[key] = self._record(rhs)
        graph, static_in, static_out = entry
        static_in.copy_(rhs)
        graph.replay()
        return static_out.clone()

    @property
    def stats(self) -> dict:
        """The last solve: iterations, the largest relative residual over the columns, converged (<= rtol). On the
        fixed path the residual is computed from the final iterate when read (an operator apply and a host sync), so
        the solve itself stays synchronisation-free."""
        if self.iterations is None:
            return self._stats or {"iterations": 0, "residual": 0.0, "converged": True}
        if self._last is None:
            return {"iterations": 0, "residual": 0.0, "converged": True}
        B, X = self._last
        A = self._A64 if self.fp64_residual else self.levels[0].A
        res = torch.sparse.addmm(B, A, X, alpha=-1.0).norm(dim=0)
        bnorm = B.norm(dim=0)
        rel = float((res / torch.where(bnorm > 0, bnorm, torch.ones_like(bnorm))).max())
        return {"iterations": self.iterations, "residual": rel, "converged": rel <= self.rtol}

    # ------------------------------------------------------------------ interface
    def solve(self, r: Tensor) -> Tensor:
        """r [n, Pf, 3] -> K_ff^-1 r; differentiable (the backward is the same solve, K is symmetric)."""
        n, Pf, _ = r.shape
        sol = _SparseSolve.apply(r.permute(1, 0, 2).reshape(Pf, n * 3), self)
        return sol.reshape(Pf, n, 3).permute(1, 0, 2)

    def apply_full(self, v: Tensor) -> Tensor:
        """K_s v on the whole grid through the cell blocks: v [n, P, 3] -> [n, P, 3]."""
        return apply_cell_blocks(self.grid, self._A.to(v.dtype), v)

    def K_fp(self, dp: Tensor) -> Tensor:
        g = self.grid
        v = torch.zeros(dp.shape[0], g.P, 3, dtype=dp.dtype, device=dp.device).index_copy_(1, g.pinned, dp)
        return self.apply_full(v)[:, g.free]
