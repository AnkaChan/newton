# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Weighted least-squares fusion of per-Gauss-point target increments (design spec 4.3).

K = B^T W B = I_3 (x) K_s with one scalar matrix K_s per grid shape. On a box grid of unit trilinear cells K_s is
exactly the Kronecker sum  Kx (x) My (x) Mz + Mx (x) Ky (x) Mz + Mx (x) My (x) Kz  of the assembled 1D stiffness and
mass matrices (2-point Gauss is exact for these integrands), and pinning the z-min face removes one Dirichlet plane
of the z factor. The free block is therefore solved exactly by fast diagonalisation: generalised eigenvectors of
(K_a, M_a) per axis, three small matmuls in, a pointwise division, three matmuls out. No factor is stored, the cost
is O(P (nx + ny + nz)) and it scales to any grid; the dense float32 inverse (94 MB and 0.05 ms per solve for the
canonical grid, 5 GB at 20x20x80) remains available as `Fusion(solver="dense")` and as the test reference.
For meshes that are not box grids (any cell table, any pin set) the same interface offers a geometric multigrid
preconditioned conjugate gradient on the voxel lattice (`multigrid.MultigridFactor`, the default on CUDA) and cuDSS
through nvmath-python (`SparseFactor`: fp32 solve 0.21 ms for 3 right-hand sides on the canonical grid, 0.65 ms at
32k cells, factor once in 0.1-0.5 s, 13 GB at a million corners); both solves are wrapped in an autograd Function
whose backward is the same solve. Prescribed corners are eliminated exactly. Autograd through the matmuls is the
adjoint solve (K is symmetric).

Unpinned bodies (derivation note section 7). Without pins K is singular on the three translations (7.1) and the
right-hand side B^T W dF is orthogonal to them for every dF, so the shape solve is translation free: KronFactor
uses the pseudo-inverse (the zero eigenvalue's inverse set to 0), the other factors fix one reference corner
r = grid.ref_corners[0] as the Dirichlet node (K_ff with one artificially fixed corner is SPD, Corollary 4.2) and
set d_r = 0. `fuse(..., centroid_target=c_t)` then adds the translation t = c_t - c(x_k) - c(d_hat) per free object
(eq. 7.21), which is the solution of the centroid-only blend (7.17)/(7.19): c(x_k + d) = c_t exactly for every
lambda, and lambda drops out of the shape solve. `project_gradient` removes the translation component of the
gradient before the solve: the modes cannot carry it (B Z = 0, eq. 7.1).
"""

from __future__ import annotations

import dataclasses

import torch

from .grid import Grid
from .hex import MODE_COUNT, HexConstants

Tensor = torch.Tensor


def cell_block(hc: HexConstants | None = None) -> Tensor:
    """A [8,8] = sum_q w_q g_{q,k} . g_{q,k'} for the unit hex (float64)."""
    from . import hex as hx

    return torch.einsum("q,qka,qla->kl", hx.WEIGHTS, hx.GQ, hx.GQ)


def assemble_scalar(grid: Grid) -> Tensor:
    """K_s [P,P] float64 on the grid's device."""
    A = cell_block().to(grid.device)
    C, P = grid.C, grid.P
    rows = grid.cells[:, :, None].expand(C, 8, 8).reshape(-1)
    cols = grid.cells[:, None, :].expand(C, 8, 8).reshape(-1)
    K = torch.zeros(P, P, dtype=torch.float64, device=grid.device)
    K.index_put_((rows, cols), A[None].expand(C, 8, 8).reshape(-1), accumulate=True)
    return K


def one_d_matrices(n: int, device, dtype=torch.float64) -> tuple[Tensor, Tensor]:
    """Assembled 1D mass and stiffness matrices on n + 1 nodes of unit linear elements, [n+1, n+1]."""
    M = torch.zeros(n + 1, n + 1, dtype=dtype, device=device)
    K = torch.zeros(n + 1, n + 1, dtype=dtype, device=device)
    m = torch.tensor([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=dtype, device=device)
    k = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype, device=device)
    idx = torch.arange(n, device=device)
    for a in range(2):
        for b in range(2):
            M.index_put_((idx + a, idx + b), m[a, b].expand(n), accumulate=True)
            K.index_put_((idx + a, idx + b), k[a, b].expand(n), accumulate=True)
    return M, K


def generalised_eigh(K: Tensor, M: Tensor) -> tuple[Tensor, Tensor]:
    """K V = M V diag(lam) with V^T M V = I (float64)."""
    L = torch.linalg.cholesky(M)
    Li = torch.linalg.inv(L)
    lam, W = torch.linalg.eigh(Li @ K @ Li.T)
    return lam, Li.T @ W


def dirichlet_view(grid: Grid) -> Grid:
    """The corner split a factor solves on: the grid itself when it has pins; for an unpinned body a shallow copy with
    the reference corner r = grid.ref_corners[0] as the only Dirichlet node (derivation 7.6: K_ff with one fixed
    corner is SPD by Corollary 4.2; the solution with d_r = 0 is one particular solution of the singular system)."""
    if grid.pinned.numel() > 0:
        return grid
    r = grid.ref_corners[:1]
    mask = torch.zeros(grid.P, dtype=torch.bool, device=grid.device)
    mask[r] = True
    return dataclasses.replace(grid, free=(~mask).nonzero().flatten(), pinned=r.clone(), pinned_mask=mask)


class KronFactor:
    """Fast-diagonalisation solver of the free block of K_s for a box grid with the z-min face pinned, or the
    pseudo-inverse of the whole K_s for an unpinned box (pins="none": the constant mode's inverse is 0, exact on
    right-hand sides with zero translation component).

    `free` is the corner set the solve covers (grid.free in both cases)."""

    def __init__(self, grid: Grid, dtype=torch.float32):
        if grid.kind != "box" or grid.pins not in ("zmin_face", "none"):
            raise ValueError("KronFactor needs a box grid with the z-min face pinned or no pins")
        nx, ny, nz = grid.cell_counts
        dev = grid.device
        Mx, Kx = one_d_matrices(nx, dev)
        My, Ky = one_d_matrices(ny, dev)
        Mz, Kz = one_d_matrices(nz, dev)
        lx, Vx = generalised_eigh(Kx, Mx)
        ly, Vy = generalised_eigh(Ky, My)
        self.free_body = grid.pins == "none"
        if self.free_body:
            lz, Vz = generalised_eigh(Kz, Mz)
            self.shape = (nx + 1, ny + 1, nz + 1)
        else:
            lz, Vz = generalised_eigh(Kz[1:, 1:], Mz[1:, 1:])  # free z nodes 1..nz
            self.shape = (nx + 1, ny + 1, nz)
        self.V = (Vx.to(dtype), Vy.to(dtype), Vz.to(dtype))
        denom = lx[:, None, None] + ly[None, :, None] + lz[None, None, :]
        if self.free_body:
            # eigh sorts ascending, so (0, 0, 0) is the constant mode of the three unpinned 1D stiffness matrices
            # (the translation null space of K, Lemma 4.1); its inverse is 0 (division by inf)
            if max(float(lx[0].abs()), float(ly[0].abs()), float(lz[0].abs())) > 1e-9:
                raise RuntimeError("KronFactor: the unpinned 1D stiffness has no zero eigenvalue")
            denom[0, 0, 0] = float("inf")
        self.denom = denom.to(dtype)
        self.full_shape = (nx + 1, ny + 1, nz + 1)
        self.M = (Mx.to(dtype), My.to(dtype), Mz.to(dtype))
        self.K = (Kx.to(dtype), Ky.to(dtype), Kz.to(dtype))
        self.grid = grid
        self.free = grid.free

    @staticmethod
    def _axis(A: Tensor, T: Tensor, axis: int) -> Tensor:
        """Contract A [m, n_axis] with T [n, X, Y, Z, 3] along the given spatial axis (0, 1 or 2).

        Each contraction is one batched matmul on a contiguous view (no operand permutes, no copies)."""
        n, X, Y, Z, c = T.shape
        if axis == 0:
            return torch.matmul(A, T.reshape(n, X, Y * Z * c)).reshape(n, -1, Y, Z, c)
        if axis == 1:
            return torch.matmul(A, T.reshape(n * X, Y, Z * c)).reshape(n, X, -1, Z, c)
        return torch.matmul(A, T.reshape(n * X * Y, Z, c)).reshape(n, X, Y, -1, c)

    def solve(self, r: Tensor) -> Tensor:
        """r [n, Pf, 3] -> K_ff^-1 r, exact up to rounding (the pseudo-inverse K^+ r for an unpinned box)."""
        n = r.shape[0]
        T = r.reshape(n, *self.shape, 3)
        for axis, V in enumerate(self.V):
            T = self._axis(V.T, T, axis)
        T = T / self.denom[None, ..., None]
        for axis, V in enumerate(self.V):
            T = self._axis(V, T, axis)
        return T.reshape(n, -1, 3)

    def apply_full(self, v: Tensor) -> Tensor:
        """K_s v on the whole grid: v [n, P, 3] -> [n, P, 3]."""
        n = v.shape[0]
        T = v.reshape(n, *self.full_shape, 3)
        out = torch.zeros_like(T)
        for a in range(3):
            U = T
            for axis in range(3):
                U = self._axis(self.K[axis] if axis == a else self.M[axis], U, axis)
            out = out + U
        return out.reshape(n, -1, 3)

    def K_fp(self, dp: Tensor) -> Tensor:
        """K_s[free, pinned] dp: dp [n, Pp, 3] -> [n, Pf, 3]."""
        g = self.grid
        v = torch.zeros(dp.shape[0], g.P, 3, dtype=dp.dtype, device=dp.device).index_copy_(1, g.pinned, dp)
        return self.apply_full(v)[:, g.free]


def apply_cell_blocks(grid: Grid, A: Tensor, v: Tensor) -> Tensor:
    """K_s v on the whole grid through the identical cell blocks A [8,8]: v [n, P, 3] -> [n, P, 3]."""
    n = v.shape[0]
    vc = v[:, grid.cells]  # [n, C, 8, 3]
    contrib = torch.einsum("kl,nclr->nckr", A, vc)
    return torch.zeros_like(v).index_add_(1, grid.cells.reshape(-1), contrib.reshape(n, -1, 3))


def assemble_free_csr(grid: Grid, dtype=torch.float64) -> Tensor:
    """K_s[free, free] as a CSR tensor on the grid's device, assembled from the identical cell blocks.

    Works for any cell table (not only box grids): the only structure used is `grid.cells` and the pin set."""
    A = cell_block().to(device=grid.device, dtype=dtype)
    C, P, Pf = grid.C, grid.P, grid.Pf
    rows = grid.cells[:, :, None].expand(C, 8, 8).reshape(-1)
    cols = grid.cells[:, None, :].expand(C, 8, 8).reshape(-1)
    pos = torch.full((P,), -1, dtype=torch.long, device=grid.device)
    pos[grid.free] = torch.arange(Pf, device=grid.device)
    keep = (pos[rows] >= 0) & (pos[cols] >= 0)
    coo = torch.sparse_coo_tensor(
        torch.stack([pos[rows[keep]], pos[cols[keep]]]), A[None].expand(C, 8, 8).reshape(-1)[keep], (Pf, Pf)
    ).coalesce()
    return coo.to_sparse_csr()


class _SparseSolve(torch.autograd.Function):
    """Differentiable solve with a symmetric positive definite sparse factor: the backward is the same solve."""

    @staticmethod
    def forward(ctx, rhs: Tensor, factor) -> Tensor:
        ctx.factor = factor
        return factor._solve_cols(rhs)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return ctx.factor._solve_cols(grad), None


class SparseFactor:
    """cuDSS (through nvmath-python) direct factorisation of K_s[free, free]: general meshes and pins, GPU only."""

    def __init__(self, grid: Grid, dtype=torch.float32):
        import nvmath.sparse.advanced as sa  # optional dependency: `uv pip install nvmath-python[cu12]`

        self.sa = sa
        self.dtype = dtype
        idx = grid.device.index if grid.device.index is not None else torch.cuda.current_device()
        self.device = torch.device("cuda", idx)
        self.grid = grid
        self.free = grid.free
        self._activate()
        self.csr = assemble_free_csr(grid, dtype)
        self.solvers: dict[int, object] = {}
        self._A = cell_block().to(device=grid.device, dtype=dtype)

    def _solver(self, ncol: int, rhs_cm: Tensor):
        if ncol not in self.solvers:
            opts = self.sa.DirectSolverOptions(sparse_system_type=self.sa.DirectSolverMatrixType.SPD)
            solver = self.sa.DirectSolver(self.csr, rhs_cm, options=opts, stream=torch.cuda.current_stream(self.device))
            solver.plan()
            solver.factorize()
            self.solvers[ncol] = solver
        return self.solvers[ncol]

    def _activate(self) -> None:
        """nvmath binds to the thread's current CUDA device (cuda.core keeps one Device object per thread), and
        autograd runs the backward on a worker thread, so activate on every call."""
        from cuda.core import Device as _CudaDevice

        torch.cuda.set_device(self.device)
        _CudaDevice(self.device.index).set_current()

    def _solve_cols(self, rhs: Tensor) -> Tensor:
        """rhs [Pf, k] (any layout) -> K_ff^-1 rhs [Pf, k]; cuDSS wants a column-major right-hand side."""
        self._activate()
        k = rhs.shape[1]
        rhs_cm = rhs.detach().t().contiguous().t()
        solver = self._solver(k, rhs_cm)
        solver.reset_operands(b=rhs_cm, stream=torch.cuda.current_stream(self.device))
        x = solver.solve(stream=torch.cuda.current_stream(self.device))
        return x.contiguous()

    def solve(self, r: Tensor) -> Tensor:
        n, Pf, _ = r.shape
        sol = _SparseSolve.apply(r.permute(1, 0, 2).reshape(Pf, n * 3), self)
        return sol.reshape(Pf, n, 3).permute(1, 0, 2)

    def apply_full(self, v: Tensor) -> Tensor:
        """K_s v on the whole grid through the cell blocks: v [n, P, 3] -> [n, P, 3]."""
        return apply_cell_blocks(self.grid, self._A, v)

    def K_fp(self, dp: Tensor) -> Tensor:
        g = self.grid
        v = torch.zeros(dp.shape[0], g.P, 3, dtype=dp.dtype, device=dp.device).index_copy_(1, g.pinned, dp)
        return self.apply_full(v)[:, g.free]


def sparse_solver_available() -> bool:
    try:
        import nvmath.sparse.advanced  # noqa: F401

        return True
    except Exception:
        return False


class DenseFactor:
    """Stored dense inverse of K_ff (reference solver; memory grows as Pf^2)."""

    def __init__(self, grid: Grid, dtype=torch.float32, refine: bool = False):
        Ks = assemble_scalar(grid)
        f, p = grid.free, grid.pinned
        Kff = Ks[f][:, f]
        self.K_inv = torch.linalg.inv(Kff).to(dtype)
        self.K_fp_mat = Ks[f][:, p].to(dtype)
        self.Kff = Kff.to(dtype) if refine else None
        self.free = grid.free

    def solve(self, r: Tensor) -> Tensor:
        n, Pf, _ = r.shape
        rhs = r.permute(1, 0, 2).reshape(Pf, n * 3)
        sol = self.K_inv @ rhs
        if self.Kff is not None:
            sol = sol + self.K_inv @ (rhs - self.Kff @ sol)
        return sol.reshape(Pf, n, 3).permute(1, 0, 2)

    def K_fp(self, dp: Tensor) -> Tensor:
        return torch.einsum("fp,npa->nfa", self.K_fp_mat, dp)


class Fusion:
    """solver: "auto" (structured solve for box grids with the z-min face pinned, multigrid PCG for anything else on
    CUDA, dense inverse otherwise), or one of "kron", "mg", "sparse" (cuDSS), "dense" to force a path."""

    def __init__(self, solver: str = "auto", refine: bool = False, options: dict | None = None):
        self.solver = solver
        self.refine = refine
        self.options = dict(options or {})  # MultigridFactor keyword options (iterations, capture, rtol, degree, ...)
        self.sparse_max_free = int(
            self.options.pop("sparse_max_free", 300_000)
        )  # "auto": cuDSS up to this many free corners
        self.factors: dict[
            tuple, object
        ] = {}  # (grid key, dtype) -> KronFactor | MultigridFactor | SparseFactor | DenseFactor

    def factor(self, grid: Grid, dtype=torch.float32):
        """The solver of the grid's shape system; every factor exposes `free`, the corner set its solve covers
        (grid.free, or all corners but the reference corner for an unpinned body on a non-Kron solver)."""
        key = (grid.key, dtype)
        if key not in self.factors:
            structured = grid.kind == "box" and grid.pins in ("zmin_face", "none")
            if self.solver == "kron" or (self.solver == "auto" and structured):
                self.factors[key] = KronFactor(grid, dtype)
            else:
                g = dirichlet_view(grid)
                if self.solver == "sparse" or (
                    self.solver == "auto"
                    and grid.device.type == "cuda"
                    and g.Pf <= self.sparse_max_free
                    and sparse_solver_available()
                ):
                    # cuDSS is the fastest general solver while its factor fits (measured: 0.2 ms at 5k, 1 ms at
                    # 80k, 6 ms / ~5 GB at 600k free corners); beyond that the multigrid wins on memory (0.66 GB at 1M)
                    self.factors[key] = SparseFactor(g, dtype)
                elif self.solver == "mg" or (self.solver == "auto" and grid.device.type == "cuda"):
                    from .multigrid import MultigridFactor

                    self.factors[key] = MultigridFactor(g, dtype, **self.options)
                else:
                    self.factors[key] = DenseFactor(g, dtype, self.refine)
        return self.factors[key]

    def rhs(self, grid: Grid, dF: Tensor, hc: HexConstants) -> Tensor:
        """B^T W dF for n objects: dF [n*C,8,3,3] -> [n,P,3]."""
        n = dF.shape[0] // grid.C
        contrib = torch.einsum("q,ncqra,qka->nckr", hc.weights, dF.view(n, grid.C, 8, 3, 3), hc.Gq)
        out = torch.zeros(n, grid.P, 3, dtype=dF.dtype, device=dF.device)
        return out.index_add_(1, grid.cells.reshape(-1), contrib.reshape(n, grid.C * 8, 3))

    def fuse(self, batch, dF: Tensor, d_pinned: Tensor | None = None, centroid_target: Tensor | None = None) -> Tensor:
        """Corner displacement [N,3] fitting the target increments dF [C,8,3,3]; pinned rows = d_pinned (or 0).

        `centroid_target` [O,3] (normalised units): for the unpinned objects the translation-free shape solution
        d_hat (d_hat = 0 at the reference corner, or the pseudo-inverse solution) is completed by the rigid
        translation t = c_t - c(x_k) - c(d_hat) of eq. 7.21, with c the mass-weighted centroid over the normalised
        corner masses rho m_i and x_k = batch.x, so that c(x_k + d) = c_t exactly (Proposition 7.3(b)); pinned
        objects are untouched. Without a target the unpinned shape solution is returned as it is.
        """
        hc = batch.hc
        d = torch.zeros(batch.N, 3, dtype=dF.dtype, device=dF.device)
        for grp in batch.groups:
            grid = grp.grid
            n = grp.objects.numel()
            fac = self.factor(grid, dF.dtype)
            b = self.rhs(grid, dF[grp.cell_idx], hc)[:, fac.free]
            dp = None
            if d_pinned is not None and grid.pinned.numel() > 0:
                dp = d_pinned[grp.corner_idx].view(n, grid.P, 3)[:, grid.pinned]
                b = b - fac.K_fp(dp)
            sol = fac.solve(b)
            dg = torch.zeros(n, grid.P, 3, dtype=dF.dtype, device=dF.device).index_copy_(1, fac.free, sol)
            if dp is not None:
                dg = dg.index_copy_(1, grid.pinned, dp)
            if centroid_target is not None and grid.pinned.numel() == 0:
                m = batch.material.rho[grp.objects].to(dF.dtype)[:, None] * grid.mass.to(dF.dtype)[None]  # [n,P]
                w = (m / m.sum(1, keepdim=True))[..., None]
                c_x = (w * batch.x.detach()[grp.corner_idx].view(n, grid.P, 3)).sum(1)
                c_d = (w * dg).sum(1)
                dg = dg + (centroid_target[grp.objects] - c_x - c_d)[:, None, :]
            d = d.index_copy_(0, grp.corner_idx, dg.reshape(-1, 3))
        return d

    def project_gradient(self, batch, gX: Tensor) -> Tensor:
        """Gamma^T W B K_ff^-1 gX_free per cell: [N,3] (pinned rows ignored) -> [C,7,3] world-axis mode gradient.

        Unpinned objects: the translation component of gX (its mean over the object's corners) is removed before
        the solve. It is zero for the elastic and viscous parts (eq. 1.8); the inertia and contact parts carry
        M_tot (c(x) - c(y_n)) - F_con(x) (eq. 7.8), which vanishes only at the fixed point. K is singular on it
        (7.1) and the mode targets cannot represent it (B Z = 0), so it does not enter the modes either way;
        removing it makes the result independent of the particular solution the factor picks.
        """
        hc = batch.hc
        out = torch.zeros(batch.C, MODE_COUNT, 3, dtype=gX.dtype, device=gX.device)
        for grp in batch.groups:
            grid = grp.grid
            n = grp.objects.numel()
            fac = self.factor(grid, gX.dtype)
            g_all = gX[grp.corner_idx].view(n, grid.P, 3)
            if grid.pinned.numel() == 0:
                g_all = g_all - g_all.mean(1, keepdim=True)
            g = g_all[:, fac.free]
            z = torch.zeros(n, grid.P, 3, dtype=gX.dtype, device=gX.device).index_copy_(1, fac.free, fac.solve(g))
            dFz = torch.einsum("nckr,qka->ncqra", z[:, grid.cells], hc.Gq) * hc.weights[:, None, None]
            m = torch.einsum("qij,ncqi->ncj", hc.Gamma, dFz.reshape(n, grid.C, 8, 9))
            out = out.index_copy_(0, grp.cell_idx, m.reshape(-1, MODE_COUNT, 3))
        return out
