# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fusion against a float64 least squares of the full B^T W B system; separability; mixed-grid batch; the Kronecker
factor on every pinned face against the dense inverse."""

import unittest

import torch

from experiments.lido import hex as hx
from experiments.lido.batch import Batch
from experiments.lido.fusion import (
    DenseFactor,
    Fusion,
    KronFactor,
    SparseFactor,
    assemble_scalar,
    sparse_solver_available,
)
from experiments.lido.grid import FACE_PINS, Grid


def full_B(grid):
    """B [72 C, 3 P] float64 with vec F index 3 r + a and corner coordinate index 3 p + r."""
    C, P = grid.C, grid.P
    B = torch.zeros(C, 8, 3, 3, P, 3, dtype=torch.float64)
    for c in range(C):
        for k in range(8):
            p = grid.cells[c, k].item()
            for r in range(3):
                B[c, :, r, :, p, r] += hx.GQ[:, k, :]
    return B.reshape(C * 72, P * 3)


class TestFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.grid = Grid.build((2, 2, 3))
        self.hc64 = hx.HexConstants.get("cpu", torch.float64)

    def test_separability(self):
        g = self.grid
        B = full_B(g)
        W = torch.diag(hx.WEIGHTS.repeat_interleave(9).repeat(g.C))
        K = B.t() @ W @ B
        Ks = assemble_scalar(g)
        K_kron = torch.kron(Ks, torch.eye(3, dtype=torch.float64))
        self.assertLess((K - K_kron).abs().max().item() / K.abs().max().item(), 1e-12)

    def test_fuse_matches_dense_least_squares(self):
        g = self.grid
        batch = Batch.build([g], "cpu", torch.float64)
        fusion = Fusion()
        dm = torch.randn(g.C, 7, 3, dtype=torch.float64)
        dF = hx.modes_to_gauss(dm, self.hc64)
        dp = torch.zeros(g.P, 3, dtype=torch.float64)
        dp[g.pinned] = 0.1 * torch.randn(g.pinned.numel(), 3, dtype=torch.float64)
        d = fusion.fuse(batch, dF, dp)
        B = full_B(g)
        W = torch.diag(hx.WEIGHTS.repeat_interleave(9).repeat(g.C))
        free3 = torch.stack([3 * g.free + r for r in range(3)], 1).reshape(-1)
        pin3 = torch.stack([3 * g.pinned + r for r in range(3)], 1).reshape(-1)
        target = dF.reshape(-1)
        rhs = B[:, free3].t() @ W @ (target - B[:, pin3] @ dp[g.pinned].reshape(-1))
        Kff = B[:, free3].t() @ W @ B[:, free3]
        ref = torch.zeros(g.P * 3, dtype=torch.float64)
        ref[free3] = torch.linalg.solve(Kff, rhs)
        ref[pin3] = dp[g.pinned].reshape(-1)
        self.assertLess((d.reshape(-1) - ref).abs().max().item(), 1e-9)

    def test_single_cell_fit_is_exact(self):
        g = Grid.build((1, 1, 1), pins="none")
        # a free single cell: K is singular (translation); pin one corner set instead
        g = Grid.build((1, 1, 2))
        batch = Batch.build([g], "cpu", torch.float64)
        fusion = Fusion()
        dm = torch.randn(g.C, 7, 3, dtype=torch.float64)
        dF = hx.modes_to_gauss(dm, self.hc64)
        d = fusion.fuse(batch, dF)
        # achieved increments differ from the targets only through shared-corner agreement; check the residual is orthogonal
        achieved = hx.gauss_deformation(d[g.cells], self.hc64)
        res = achieved - dF
        B = full_B(g)
        W = torch.diag(hx.WEIGHTS.repeat_interleave(9).repeat(g.C))
        free3 = torch.stack([3 * g.free + r for r in range(3)], 1).reshape(-1)
        self.assertLess((B[:, free3].t() @ W @ res.reshape(-1)).abs().max().item(), 1e-10)

    def test_project_gradient_matches_dense(self):
        g = self.grid
        batch = Batch.build([g], "cpu", torch.float64)
        fusion = Fusion()
        gX = torch.randn(g.P, 3, dtype=torch.float64)
        gX[g.pinned] = 0
        out = fusion.project_gradient(batch, gX)
        B = full_B(g)
        W = torch.diag(hx.WEIGHTS.repeat_interleave(9).repeat(g.C))
        free3 = torch.stack([3 * g.free + r for r in range(3)], 1).reshape(-1)
        Kff = B[:, free3].t() @ W @ B[:, free3]
        z = torch.zeros(g.P * 3, dtype=torch.float64)
        z[free3] = torch.linalg.solve(Kff, gX.reshape(-1)[free3])
        proj = (W @ B @ z).reshape(g.C, 8, 3, 3)
        ref = hx.gauss_to_modes(proj, self.hc64)
        self.assertLess((out - ref).abs().max().item(), 1e-9)

    def test_mixed_grid_batch_equals_per_object(self):
        g1, g2 = Grid.build((2, 2, 3)), Grid.build((1, 2, 2))
        grids = [g1, g2, g1]
        batch = Batch.build(grids, "cpu", torch.float64)
        fusion = Fusion()
        dm = torch.randn(batch.C, 7, 3, dtype=torch.float64)
        dF = hx.modes_to_gauss(dm, self.hc64)
        d = fusion.fuse(batch, dF)
        for o, g in enumerate(grids):
            single = Batch.build([g], "cpu", torch.float64)
            cs, ce = batch.cell_off[o].item(), batch.cell_off[o + 1].item()
            ps, pe = batch.corner_off[o].item(), batch.corner_off[o + 1].item()
            d1 = Fusion().fuse(single, dF[cs:ce])
            self.assertLess((d[ps:pe] - d1).abs().max().item(), 1e-10)
        self.assertEqual(len(batch.groups), 2)

    def test_autograd_through_solve(self):
        g = self.grid
        batch = Batch.build([g], "cpu", torch.float64)
        fusion = Fusion()
        dm = torch.randn(g.C, 7, 3, dtype=torch.float64, requires_grad=True)
        d = fusion.fuse(batch, hx.modes_to_gauss(dm, self.hc64))
        v = torch.randn_like(d)
        (grad,) = torch.autograd.grad((d * v).sum(), dm)
        # adjoint: d = S(dm) linear, so grad = S^T v; check with a directional finite difference
        e = torch.randn_like(dm)
        lhs = (grad * e).sum()
        rhs = (fusion.fuse(batch, hx.modes_to_gauss(e.detach(), self.hc64)) * v).sum()
        self.assertAlmostEqual(lhs.item(), rhs.item(), places=9)

    def test_kron_equals_dense_small(self):
        for cc in ((2, 2, 3), (1, 2, 2), (3, 2, 4)):
            g = Grid.build(cc)
            batch = Batch.build([g], "cpu", torch.float64)
            dm = torch.randn(g.C, 7, 3, dtype=torch.float64)
            dF = hx.modes_to_gauss(dm, self.hc64)
            dp = torch.zeros(g.P, 3, dtype=torch.float64)
            dp[g.pinned] = 0.1 * torch.randn(g.pinned.numel(), 3, dtype=torch.float64)
            gX = torch.randn(g.P, 3, dtype=torch.float64)
            d_k = Fusion("kron").fuse(batch, dF, dp)
            d_d = Fusion("dense").fuse(batch, dF, dp)
            self.assertLess((d_k - d_d).abs().max().item(), 1e-11)
            p_k = Fusion("kron").project_gradient(batch, gX)
            p_d = Fusion("dense").project_gradient(batch, gX)
            self.assertLess((p_k - p_d).abs().max().item(), 1e-11)

    def test_kron_every_pinned_face_equals_dense(self):
        """One lattice face clamped (any of the six): the Kronecker solve, K_fp, the fused displacement with
        prescribed pins and the projected gradient equal the dense inverse's within 1e-11 (float64); "auto" picks
        the Kronecker factor for these grids."""
        for cc in ((2, 3, 4), (3, 2, 2)):
            for pins in FACE_PINS:
                g = Grid.build(cc, pins)
                axis, side = "xyz".index(pins[0]), pins[1:4]
                self.assertEqual(g.pinned.numel(), (cc[(axis + 1) % 3] + 1) * (cc[(axis + 2) % 3] + 1))
                self.assertTrue((g.rest[g.pinned, axis] == (0 if side == "min" else cc[axis])).all())
                kron, dense = KronFactor(g, torch.float64), DenseFactor(g, torch.float64)
                self.assertEqual(kron.shape[axis], cc[axis])
                r = torch.randn(2, g.Pf, 3, dtype=torch.float64)
                self.assertLess((kron.solve(r) - dense.solve(r)).abs().max().item(), 1e-11, pins)
                dp = torch.randn(2, g.pinned.numel(), 3, dtype=torch.float64)
                self.assertLess((kron.K_fp(dp) - dense.K_fp(dp)).abs().max().item(), 1e-11, pins)
                batch = Batch.build([g], "cpu", torch.float64)
                dF = hx.modes_to_gauss(torch.randn(g.C, 7, 3, dtype=torch.float64), self.hc64)
                d_pinned = torch.zeros(g.P, 3, dtype=torch.float64)
                d_pinned[g.pinned] = 0.1 * torch.randn(g.pinned.numel(), 3, dtype=torch.float64)
                auto = Fusion()
                self.assertIsInstance(auto.factor(g, torch.float64), KronFactor)
                d_k = auto.fuse(batch, dF, d_pinned)
                d_d = Fusion("dense").fuse(batch, dF, d_pinned)
                self.assertLess((d_k - d_d).abs().max().item(), 1e-11, pins)
                self.assertTrue(torch.equal(d_k[g.pinned], d_pinned[g.pinned]))
                gX = torch.randn(g.P, 3, dtype=torch.float64)
                p_k = auto.project_gradient(batch, gX)
                p_d = Fusion("dense").project_gradient(batch, gX)
                self.assertLess((p_k - p_d).abs().max().item(), 1e-11, pins)
        with self.assertRaises(ValueError):
            KronFactor(Grid.from_voxels(torch.ones(2, 2, 2, dtype=torch.bool)))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_float32_cuda_accuracy_canonical(self):
        import time

        g = Grid.build((10, 10, 40), device="cuda")
        batch = Batch.build([g], "cuda")
        dm = 0.01 * torch.randn(g.C, 7, 3, device="cuda")
        dF = hx.modes_to_gauss(dm, batch.hc)
        Ks = assemble_scalar(g)
        rhs = Fusion().rhs(g, dF.double(), hx.HexConstants.get("cuda", torch.float64))[0][g.free]
        ref = torch.linalg.solve(Ks[g.free][:, g.free], rhs)
        for solver in ("kron", "dense"):
            fusion = Fusion(solver)
            d = fusion.fuse(batch, dF)
            rel = (d[g.free].double() - ref).norm() / ref.norm()
            self.assertLess(rel.item(), 1e-5, solver)
            fac = fusion.factor(g)
            r = torch.randn(1, g.Pf, 3, device="cuda")
            for _ in range(3):
                fac.solve(r)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(20):
                fac.solve(r)
            torch.cuda.synchronize()
            print(
                f"fusion solve {solver}: {(time.perf_counter() - t) / 20 * 1e3:.3f} ms (canonical grid, 3 columns), rel err {rel.item():.1e}"
            )

    @unittest.skipUnless(torch.cuda.is_available() and sparse_solver_available(), "cuda + nvmath")
    def test_sparse_equals_dense_and_backward(self):
        import time

        g = Grid.build((3, 2, 4), device="cuda")
        batch = Batch.build([g], "cuda", torch.float64)
        dm = torch.randn(g.C, 7, 3, dtype=torch.float64, device="cuda", requires_grad=True)
        dF = hx.modes_to_gauss(dm, batch.hc)
        dp = torch.zeros(g.P, 3, dtype=torch.float64, device="cuda")
        dp[g.pinned] = 0.1 * torch.randn(g.pinned.numel(), 3, dtype=torch.float64, device="cuda")
        d_s = Fusion("sparse").fuse(batch, dF, dp)
        d_d = Fusion("dense").fuse(batch, dF, dp)
        self.assertLess((d_s - d_d).abs().max().item(), 1e-10)
        v = torch.randn_like(d_s)
        (grad_s,) = torch.autograd.grad((d_s * v).sum(), dm, retain_graph=True)
        (grad_d,) = torch.autograd.grad((d_d * v).sum(), dm)
        self.assertLess((grad_s - grad_d).abs().max().item(), 1e-10)
        gX = torch.randn(g.P, 3, dtype=torch.float64, device="cuda")
        self.assertLess(
            (Fusion("sparse").project_gradient(batch, gX) - Fusion("dense").project_gradient(batch, gX))
            .abs()
            .max()
            .item(),
            1e-10,
        )
        # float32 canonical timing
        g = Grid.build((10, 10, 40), device="cuda")
        fac = SparseFactor(g)
        r = torch.randn(1, g.Pf, 3, device="cuda")
        for _ in range(3):
            fac.solve(r)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            fac.solve(r)
        torch.cuda.synchronize()
        print(f"fusion solve sparse (cuDSS) canonical grid, 3 columns: {(time.perf_counter() - t) / 10 * 1e3:.3f} ms")

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_kron_large_grid(self):
        import time

        g = Grid.build((20, 20, 80), device="cuda")
        fac = KronFactor(g)
        r = torch.randn(1, g.Pf, 3, device="cuda")
        u = fac.solve(r)
        # residual check through the full operator: K_s u (free rows) == r
        full = torch.zeros(1, g.P, 3, device="cuda").index_copy(1, g.free, u)
        res = fac.apply_full(full)[:, g.free] - r
        self.assertLess((res.norm() / r.norm()).item(), 1e-4)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            fac.solve(r)
        torch.cuda.synchronize()
        print(
            f"fusion solve kron 20x20x80 ({g.C} cells, {g.Pf} free corners): {(time.perf_counter() - t) / 10 * 1e3:.3f} ms"
        )


class TestBatchedKron(unittest.TestCase):
    """`Fusion(batched=True)`: the padded multi-group solve (`BatchedKron`) reproduces the per-group loop on a batch
    of free and pinned boxes of different shapes (fuse with a centroid target, fuse without, project_gradient),
    carries the gradient to dF, and is cached per layout."""

    def batch(self, device, dtype):
        from experiments.lido.tests.test_capture import set_material

        pins = ["none", "zmin_face", "none", "ymax_face", "xmin_face", "none"]
        sides = [(2, 3, 2), (2, 2, 3), (4, 2, 2), (3, 3, 2), (2, 2, 2), (3, 2, 4)]
        grids = [Grid.build(s, pins=p, device=device) for s, p in zip(sides, pins, strict=True)]
        b = Batch.build(grids, device, dtype)
        set_material(b, grids, dtype, device)
        gen = torch.Generator().manual_seed(4)
        b.X = b.rest.to(dtype) + 0.05 * torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
        b.X[b.pinned] = b.rest.to(dtype)[b.pinned]
        b.x = b.X.clone()
        dF = 0.1 * torch.randn(b.C, 8, 3, 3, generator=gen, dtype=dtype).to(device)
        gX = torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
        c_t = torch.randn(b.O, 3, generator=gen, dtype=dtype).to(device) + b.X.mean(0)
        return b, dF, gX, c_t

    def check(self, device, dtype, tol):
        from experiments.lido import physics

        b, dF, gX, c_t = self.batch(device, dtype)
        self.assertEqual(len(b.groups), 6)
        loop, batched = Fusion(), Fusion(batched=True)
        self.assertIsNone(loop.batched_solver(b, dtype))
        kron = batched.batched_solver(b, dtype)
        self.assertIsNotNone(kron)
        self.assertIs(batched.batched_solver(b, dtype), kron)  # cached on the batch
        self.assertEqual(
            kron.shape, (5, 4, 5)
        )  # the largest free lattice per axis (nx + 1, or nx with the face pinned)
        for target in (None, c_t):
            d1 = loop.fuse(b, dF, centroid_target=target)
            d2 = batched.fuse(b, dF, centroid_target=target)
            self.assertLess(
                (d1 - d2).abs().max().item(), tol * max(1.0, d1.abs().max().item()), f"target {target is not None}"
            )
            self.assertTrue((d2[b.pinned] == 0).all())
            if target is not None:
                c = physics.centroid(b, b.x + d2)
                self.assertLess((c - target)[b.free_objects].abs().max().item(), 10 * tol)
        p1, p2 = loop.project_gradient(b, gX), batched.project_gradient(b, gX)
        self.assertLess((p1 - p2).abs().max().item(), tol * max(1.0, p1.abs().max().item()))
        # gradient to the targets through the batched path, as through the loop
        dFg = dF.clone().requires_grad_(True)
        w = torch.randn_like(d1)
        (g2,) = torch.autograd.grad((batched.fuse(b, dFg, centroid_target=c_t) * w).sum(), dFg)
        (g1,) = torch.autograd.grad((loop.fuse(b, dFg, centroid_target=c_t) * w).sum(), dFg)
        self.assertLess((g1 - g2).abs().max().item(), tol * max(1.0, g1.abs().max().item()))
        # the cache follows the layout
        b.relayout([Grid.build((2, 2, 2), pins="none", device=device), *b.grids[1:]])
        self.assertIsNone(b.fusion_cache)
        self.assertIsNot(batched.batched_solver(b, dtype), kron)
        # one group (body mode): the loop path whatever the flag
        one = Batch.build([Grid.build((2, 2, 3), device=device)] * 3, device, dtype)
        self.assertIsNone(batched.batched_solver(one, dtype))

    def test_cpu_float64(self):
        self.check("cpu", torch.float64, 1e-11)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_cuda_float32(self):
        self.check("cuda:0", torch.float32, 2e-5)


if __name__ == "__main__":
    unittest.main()
