# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""`BatchedSparse` (2026-10-02): the block-diagonal cuDSS factorisation of a batch of voxel grids and boxes
reproduces the per-group loop (fuse with and without a centroid target, project_gradient, the backward), its
blocks are the loop's matrices, it is cached per batch layout and dtype, it falls back to the loop without nvmath
or off the GPU, and a benchmark of the loop against the batched factor on a 64k-cell batch of ~150 voxel bodies
(the v6 scene size)."""

import time
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.lido import fusion as fu
from experiments.lido import physics, shapes
from experiments.lido.batch import Batch
from experiments.lido.fusion import BatchedKron, BatchedSparse, Fusion, sparse_solver_available
from experiments.lido.grid import Grid
from experiments.lido.tests.test_capture import set_material

CUDA_NVMATH = torch.cuda.is_available() and sparse_solver_available()


def voxel_grids(device, rng: np.random.Generator) -> list:
    """Eight small connected voxel grids (5-13 voxels of a 3 x 3 x 3 lattice, cropped): pinned on the z-min face,
    free, or pinned by a mask (the x = 0 lattice corners); the first grid repeated, so one group has two objects."""
    pins = ["zmin_face", "none", "none", "zmin_face", "mask", "none", "zmin_face"]
    grids = []
    for p in pins:
        occ = torch.from_numpy(shapes.trim(shapes.grow_coarse(rng, (3, 3, 3), int(rng.integers(5, 14)))))
        pin = p
        if p == "mask":
            pin = torch.zeros(tuple(n + 1 for n in occ.shape), dtype=torch.bool)
            pin[0] = True
        grids.append(Grid.from_voxels(occ, pin, device))
    return [*grids, grids[0]]


def make_batch(device, dtype, grids: list | None = None):
    """Eight voxel objects plus a free box and a box pinned on its y-max face (ten objects, nine groups), a noisy
    rest state, random targets dF, gradient gX and centroid targets c_t."""
    rng = np.random.default_rng(3)
    if grids is None:
        grids = [
            *voxel_grids(device, rng),
            Grid.build((2, 3, 2), "none", device),
            Grid.build((2, 2, 3), "ymax_face", device),
        ]
    b = Batch.build(grids, device, dtype)
    set_material(b, grids, dtype, device)
    gen = torch.Generator().manual_seed(5)
    b.X = b.rest.to(dtype) + 0.05 * torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
    b.X[b.pinned] = b.rest.to(dtype)[b.pinned]
    b.x = b.X.clone()
    dF = 0.1 * torch.randn(b.C, 8, 3, 3, generator=gen, dtype=dtype).to(device)
    gX = torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
    c_t = torch.randn(b.O, 3, generator=gen, dtype=dtype).to(device) + b.X.mean(0)
    return b, dF, gX, c_t


def free_total(grids: list) -> int:
    """Rows of the block-diagonal system: the free corners of every pinned object, all but one of a free one."""
    return sum(g.Pf if g.pinned.numel() > 0 else g.P - 1 for g in grids)


@unittest.skipUnless(CUDA_NVMATH, "cuda + nvmath")
class TestBatchedSparse(unittest.TestCase):
    device = "cuda:0"

    def check(self, dtype, tol, fd_tol):
        b, dF, gX, c_t = make_batch(self.device, dtype)
        self.assertEqual((b.O, len(b.groups)), (10, 9))
        loop, batched = Fusion(), Fusion(batched=True)
        self.assertIsNone(loop.batched_solver(b, dtype))
        sp = batched.batched_solver(b, dtype)
        self.assertIsInstance(sp, BatchedSparse)
        self.assertIs(batched.batched_solver(b, dtype), sp)  # cached on the batch
        Nf = free_total(b.grids)
        self.assertEqual(sp.Nf, Nf)
        self.assertEqual(tuple(sp.solver.csr.shape), (Nf, Nf))
        # fuse with a centroid target: the target fixes the translation, so the result is the loop's
        d1 = loop.fuse(b, dF, centroid_target=c_t)
        d2 = batched.fuse(b, dF, centroid_target=c_t)
        scale = max(1.0, d1.abs().max().item())
        self.assertLess((d1 - d2).abs().max().item(), tol * scale)
        self.assertTrue((d2[b.pinned] == 0).all())
        c = physics.centroid(b, b.x + d2)
        self.assertLess((c - c_t)[b.free_objects].abs().max().item(), 10 * tol)
        # without a target: the loop's result on the pinned objects and on the free voxel bodies (both solvers fix
        # the reference corner); on the free box the loop's KronFactor takes the pseudo-inverse solution instead,
        # so the two differ by a rigid translation of that object only
        d1, d2 = loop.fuse(b, dF), batched.fuse(b, dF)
        o = b.corner_obj
        diff = d1 - d2
        counts = torch.bincount(o, minlength=b.O).to(dtype)[:, None]
        mean = torch.zeros(b.O, 3, dtype=dtype, device=diff.device).index_add_(0, o, diff) / counts
        self.assertLess((diff - mean[o]).abs().max().item(), tol * scale)
        box_free = torch.tensor([g.kind == "box" and g.pins == "none" for g in b.grids], device=diff.device)
        self.assertEqual(int(box_free.sum()), 1)
        self.assertLess(mean[~box_free].abs().max().item(), tol * scale)
        # project_gradient never sees the particular solution
        p1, p2 = loop.project_gradient(b, gX), batched.project_gradient(b, gX)
        self.assertLess((p1 - p2).abs().max().item(), tol * max(1.0, p1.abs().max().item()))
        # one cuDSS plan (three right-hand-side columns) serves fuse and project_gradient
        self.assertEqual(list(sp.solver.solvers), [3])
        # the backward through the batched solve: the loop's gradient, and the directional derivative (fuse is
        # affine in dF, so a central difference is exact up to rounding)
        dFg = dF.clone().requires_grad_(True)
        w = torch.randn_like(d1)
        (g2,) = torch.autograd.grad((batched.fuse(b, dFg, centroid_target=c_t) * w).sum(), dFg)
        (g1,) = torch.autograd.grad((loop.fuse(b, dFg, centroid_target=c_t) * w).sum(), dFg)
        self.assertLess((g1 - g2).abs().max().item(), tol * max(1.0, g1.abs().max().item()))
        e = torch.randn_like(dF)
        eps = 0.5
        with torch.no_grad():
            fd = (
                batched.fuse(b, dF + eps * e, centroid_target=c_t) - batched.fuse(b, dF - eps * e, centroid_target=c_t)
            ) / (2 * eps)
        lhs, rhs = (g2 * e).sum().item(), (fd * w).sum().item()
        self.assertLess(abs(lhs - rhs), fd_tol * max(1.0, abs(lhs)))
        # the cache follows the layout
        b.relayout([Grid.build((2, 2, 2), pins="none", device=self.device), *b.grids[1:]])
        self.assertIsNone(b.fusion_cache)
        sp2 = batched.batched_solver(b, dtype)
        self.assertIsInstance(sp2, BatchedSparse)
        self.assertIsNot(sp2, sp)
        self.assertEqual(sp2.Nf, free_total(b.grids))
        # an all-box batch keeps the Kron chain; one group (body mode) the loop
        boxes = Batch.build(
            [Grid.build((2, 2, 3), device=self.device), Grid.build((2, 3, 2), "none", device=self.device)],
            self.device,
            dtype,
        )
        self.assertIsInstance(batched.batched_solver(boxes, dtype), BatchedKron)
        one = Batch.build([b.grids[1]] * 3, self.device, dtype)
        self.assertIsNone(batched.batched_solver(one, dtype))

    def test_float32(self):
        self.check(torch.float32, 1e-4, 1e-3)

    def test_float64(self):
        self.check(torch.float64, 1e-9, 1e-8)

    def test_blocks_are_the_loop_matrices(self):
        """Block o of the block-diagonal matrix is K_s[free, free] of object o's dirichlet view, in batch corner
        order, with nothing off the diagonal, and `rows` are the batch corner rows of those free corners."""
        b, *_ = make_batch(self.device, torch.float64)
        sp = Fusion(batched=True).batched_solver(b, torch.float64)
        dense = sp.solver.csr.to_dense()
        off = 0
        for o, g in enumerate(b.grids):
            dv = fu.dirichlet_view(g)
            n = dv.Pf
            K = fu.assemble_scalar(g)[dv.free][:, dv.free]
            self.assertLess((dense[off : off + n, off : off + n] - K).abs().max().item(), 1e-13)
            self.assertEqual(dense[off : off + n, :off].abs().max().item() if off else 0.0, 0.0)
            self.assertEqual(dense[off : off + n, off + n :].abs().max().item() if off + n < sp.Nf else 0.0, 0.0)
            self.assertTrue(torch.equal(sp.rows[off : off + n], dv.free + b.corner_off[o]))
            off += n
        self.assertEqual(off, sp.Nf)

    def test_fallback_to_the_loop(self):
        """Without nvmath, off the GPU, with multigrid or dense forced, or over `sparse_max_free`, the batched flag
        leaves a non-Kron batch to the per-group loop; a forced "sparse" solver ignores the bound."""
        b, dF, _gX, c_t = make_batch(self.device, torch.float32)
        batched = Fusion(batched=True)
        with mock.patch.object(fu, "sparse_solver_available", return_value=False):
            self.assertIsNone(batched.batched_solver(b, torch.float32))
            d = batched.fuse(b, dF, centroid_target=c_t)
        self.assertIs(b.fusion_cache[torch.float32], False)
        # two loop runs differ at the ULP level (index_add_ atomics), so not torch.equal
        self.assertLess((d - Fusion().fuse(b, dF, centroid_target=c_t)).abs().max().item(), 1e-5)
        fresh = lambda: Batch.build(b.grids, self.device, torch.float32)  # noqa: E731
        self.assertIsNone(Fusion("dense", batched=True).batched_solver(fresh(), torch.float32))
        self.assertIsNone(Fusion("mg", batched=True).batched_solver(fresh(), torch.float32))
        self.assertIsNone(Fusion(batched=True, options={"sparse_max_free": 10}).batched_solver(fresh(), torch.float32))
        self.assertIsInstance(
            Fusion("sparse", batched=True, options={"sparse_max_free": 10}).batched_solver(fresh(), torch.float32),
            BatchedSparse,
        )
        cpu, *_ = make_batch("cpu", torch.float64)
        self.assertIsNone(Fusion(batched=True).batched_solver(cpu, torch.float64))

    def test_benchmark_v6_batch(self):
        """Fuse and project_gradient per query on a 64k-cell batch of ~150 voxel bodies (`shapes.sample_voxel_shape`
        with seed 1; every fourth body pinned on its z-min face, the rest free): the per-group loop (one cuDSS
        factor and one solve per body) against `BatchedSparse`, with the factorisation times."""
        rng = np.random.default_rng(1)
        grids, cells = [], 0
        while cells < 64_000:
            occ = torch.from_numpy(shapes.sample_voxel_shape(rng))
            g = Grid.from_voxels(occ, "zmin_face" if len(grids) % 4 == 0 else "none", self.device)
            grids.append(g)
            cells += g.C
        b, dF, gX, c_t = make_batch(self.device, torch.float32, grids)
        Nf = free_total(grids)

        def timed(fn, repeat=5, warm=2):
            for _ in range(warm):
                fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(repeat):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / repeat * 1e3

        loop, batched = Fusion(), Fusion(batched=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        d_loop = loop.fuse(b, dF, centroid_target=c_t)
        torch.cuda.synchronize()
        t_loop_first = time.perf_counter() - t
        t = time.perf_counter()
        sp = batched.batched_solver(b, torch.float32)
        torch.cuda.synchronize()
        t_build = time.perf_counter() - t
        t = time.perf_counter()
        d_batched = batched.fuse(b, dF, centroid_target=c_t)
        torch.cuda.synchronize()
        t_batched_first = time.perf_counter() - t
        self.assertIsInstance(sp, BatchedSparse)
        rel = (d_loop - d_batched).abs().max().item() / d_loop.abs().max().item()
        self.assertLess(rel, 1e-3)
        fuse_loop = timed(lambda: loop.fuse(b, dF, centroid_target=c_t))
        fuse_batched = timed(lambda: batched.fuse(b, dF, centroid_target=c_t))
        pg_loop = timed(lambda: loop.project_gradient(b, gX))
        pg_batched = timed(lambda: batched.project_gradient(b, gX))
        print(
            f"\nBatchedSparse benchmark: {b.O} voxel bodies, {b.C} cells, {b.N} corners, {Nf} free rows, "
            f"{len(b.groups)} groups, nnz {sp.solver.csr.values().numel()}\n"
            f"  loop: first query {t_loop_first * 1e3:.0f} ms ({len(loop.factors)} cuDSS factors), "
            f"fuse {fuse_loop:.2f} ms, project_gradient {pg_loop:.2f} ms per query\n"
            f"  BatchedSparse: block-diagonal CSR {t_build * 1e3:.0f} ms, factorise + first solve "
            f"{t_batched_first * 1e3:.0f} ms, fuse {fuse_batched:.2f} ms, project_gradient {pg_batched:.2f} ms "
            f"per query\n"
            f"  max |d_loop - d_batched| / max |d_loop| = {rel:.1e}"
        )


if __name__ == "__main__":
    unittest.main()
