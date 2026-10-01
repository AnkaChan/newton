# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multigrid PCG fusion solver: fixed-iteration and adaptive solves against the dense inverse (float64, CPU), the
fixed solve against a float64 solve, CUDA-graph replay, cuDSS and scale (float32, CUDA)."""

import time
import unittest
import warnings

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
from experiments.lido.grid import Grid
from experiments.lido.multigrid import MultigridFactor
from experiments.lido.tests.test_voxel_grid import carved, holed_box

CPU_SHAPES = {
    "box 2x2x3": lambda: Grid.build((2, 2, 3)),
    "holed zmin": lambda: Grid.from_voxels(holed_box()),
    "holed random pins": lambda: Grid.from_voxels(holed_box(), random_pin_mask((6, 6, 8), 0.05, 2)),
    "carved 10x10x14 random pins": lambda: Grid.from_voxels(carved(10, 10, 14), random_pin_mask((10, 10, 14), 0.02, 5)),
}


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def random_pin_mask(shape, p: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(tuple(s + 1 for s in shape), generator=gen) < p
    mask[:, :, 0] = True  # the base plus scattered pins
    return mask


def timed(fn, reps: int):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps, out


class TestMultigridCPU(unittest.TestCase):
    """float64 on the CPU with a small coarsest level so that every case has several levels."""

    def setUp(self):
        torch.manual_seed(0)

    def check_against_dense(self, g: Grid, max_iter_at_1e8: int = 25):
        dense = DenseFactor(g, torch.float64)
        r = torch.randn(2, g.Pf, 3, dtype=torch.float64)
        ref = dense.solve(r)
        mg = MultigridFactor(g, torch.float64, iterations=None, rtol=1e-8, coarse_size=16)
        self.assertGreaterEqual(len(mg.levels), 2)
        self.assertEqual(mg.sizes[0], g.Pf)
        self.assertTrue(all(a > b for a, b in zip(mg.sizes[:-1], mg.sizes[1:], strict=True)))
        x = mg.solve(r)
        self.assertTrue(mg.stats["converged"])
        self.assertLessEqual(mg.stats["iterations"], max_iter_at_1e8)
        self.assertLess(rel(x, ref), 1e-6)
        tight = MultigridFactor(g, torch.float64, iterations=None, rtol=1e-13, coarse_size=16)
        self.assertLess(rel(tight.solve(r), ref), 1e-9)
        fixed = MultigridFactor(g, torch.float64, coarse_size=16)  # the default: 8 iterations on the same hierarchy
        self.assertEqual(fixed.sizes, mg.sizes)
        self.assertLess(rel(fixed.solve(r), ref), 1e-8)
        self.assertEqual(fixed.stats["iterations"], 8)
        self.assertTrue(fixed.stats["converged"])
        # the operator pieces: K_fp and apply_full equal the assembled matrix
        Ks = assemble_scalar(g)
        v = torch.randn(1, g.P, 3, dtype=torch.float64)
        self.assertLess(rel(mg.apply_full(v), torch.einsum("pq,nqa->npa", Ks, v)), 1e-12)
        if g.pinned.numel():
            dp = torch.randn(1, g.pinned.numel(), 3, dtype=torch.float64)
            self.assertLess(rel(mg.K_fp(dp), dense.K_fp(dp)), 1e-12)
        return mg

    def test_small_box(self):
        g = Grid.build((2, 2, 3))
        dense = DenseFactor(g, torch.float64)
        r = torch.randn(1, g.Pf, 3, dtype=torch.float64)
        one_level = MultigridFactor(g, torch.float64, iterations=None, rtol=1e-8)  # 27 unknowns: dense coarsest only
        self.assertEqual(len(one_level.levels), 1)
        self.assertLess(rel(one_level.solve(r), dense.solve(r)), 1e-12)
        self.assertEqual(one_level.stats["iterations"], 1)
        fixed = MultigridFactor(g, torch.float64)  # the exact preconditioner leaves nothing for the other 7 iterations
        self.assertLess(rel(fixed.solve(r), dense.solve(r)), 1e-12)
        self.assertEqual(fixed.stats["iterations"], 8)
        mg = MultigridFactor(g, torch.float64, iterations=None, rtol=1e-8, coarse_size=8)
        self.assertEqual(mg.sizes, [27, 12, 8])
        self.assertLess(rel(mg.solve(r), dense.solve(r)), 1e-9)
        self.assertLessEqual(mg.stats["iterations"], 25)

    def test_fixed_eight_iterations_match_dense(self):
        """The default fixed solve (no stopping test) against the dense float64 inverse on the CPU shapes."""
        for name, make in CPU_SHAPES.items():
            g = make()
            r = torch.randn(2, g.Pf, 3, dtype=torch.float64)
            ref = DenseFactor(g, torch.float64).solve(r)
            mg = MultigridFactor(g, torch.float64)
            self.assertIsNone(mg._warp)
            self.assertFalse(mg.capture)
            x = mg.solve(r)
            self.assertLess(rel(x, ref), 1e-9, name)
            stats = mg.stats
            self.assertEqual(stats["iterations"], 8)
            self.assertLess(stats["residual"], 1e-9, name)
            self.assertTrue(stats["converged"])
            # the residual table: more iterations keep improving (no stagnation from the fixed schedule)
            mg.iterations = 4
            self.assertGreater(rel(mg.solve(r), ref), rel(x, ref) * 0.5, name)

    def test_arguments(self):
        g = Grid.from_voxels(holed_box())
        with self.assertRaises(ValueError):
            MultigridFactor(g, torch.float64, iterations=0)
        with self.assertRaises(ValueError):
            MultigridFactor(g, torch.float64, fp64_every=0)
        with self.assertRaises(ValueError):
            MultigridFactor(g, torch.float64, capture=True)  # no CUDA graphs on the CPU
        self.assertFalse(MultigridFactor(g, torch.float64).capture)
        self.assertFalse(MultigridFactor(g, torch.float64).fp64_residual)  # float64 cycle: nothing to recompute

    def test_carved_zmin_pins(self):
        mg = self.check_against_dense(Grid.from_voxels(holed_box()))
        self.assertEqual(mg.sizes[0], 368)

    def test_carved_random_pin_mask(self):
        g = Grid.from_voxels(holed_box(), random_pin_mask((6, 6, 8), 0.05, 2))
        self.assertGreater(g.pinned.numel(), 48)
        self.check_against_dense(g)

    def test_cylinder_carving_random_pins(self):
        g = Grid.from_voxels(carved(10, 10, 14), random_pin_mask((10, 10, 14), 0.02, 5))
        self.check_against_dense(g)

    def test_zero_columns_and_stats(self):
        g = Grid.from_voxels(holed_box())
        mg = MultigridFactor(g, torch.float64, iterations=None, rtol=1e-8, coarse_size=16)
        r = torch.randn(2, g.Pf, 3, dtype=torch.float64)
        r[1] = 0
        x = mg.solve(r)
        self.assertTrue((x[1] == 0).all())
        self.assertLess(rel(x[:1], DenseFactor(g, torch.float64).solve(r[:1])), 1e-6)
        self.assertEqual(set(mg.stats), {"iterations", "residual", "converged"})
        self.assertLessEqual(mg.stats["residual"], 1e-8)
        fixed = MultigridFactor(g, torch.float64, coarse_size=16)
        xf = fixed.solve(r)
        self.assertTrue((xf[1] == 0).all())  # zero columns stay exactly zero through the fixed schedule
        self.assertEqual(set(fixed.stats), {"iterations", "residual", "converged"})
        short = MultigridFactor(g, torch.float64, iterations=None, rtol=1e-12, max_iter=1, coarse_size=16)
        with self.assertWarns(UserWarning):
            short.solve(r)
        self.assertFalse(short.stats["converged"])
        self.assertEqual(short.stats["iterations"], 1)

    def test_fusion_mg_equals_dense_with_autograd(self):
        g = Grid.from_voxels(holed_box(), random_pin_mask((6, 6, 8), 0.05, 7))
        batch = Batch.build([g], "cpu", torch.float64)
        hc = hx.HexConstants.get("cpu", torch.float64)
        dense_fusion = Fusion("dense")
        dm = torch.randn(g.C, 7, 3, dtype=torch.float64, requires_grad=True)
        dp = torch.zeros(g.P, 3, dtype=torch.float64)
        dp[g.pinned] = 0.1 * torch.randn(g.pinned.numel(), 3, dtype=torch.float64)
        d_d = dense_fusion.fuse(batch, hx.modes_to_gauss(dm, hc), dp)
        v = torch.randn_like(d_d)
        (grad_d,) = torch.autograd.grad((d_d * v).sum(), dm, retain_graph=True)
        gX = torch.randn(g.P, 3, dtype=torch.float64)
        # the adaptive solve at a tight tolerance and the default fixed 8-iteration solve
        for options in ({"iterations": None, "rtol": 1e-13, "coarse_size": 16}, {}):
            mg_fusion = Fusion("mg", options=options)
            d_m = mg_fusion.fuse(batch, hx.modes_to_gauss(dm, hc), dp)
            fac = mg_fusion.factor(g, torch.float64)
            self.assertIsInstance(fac, MultigridFactor)
            self.assertEqual(fac.iterations, options.get("iterations", 8))
            self.assertLess(rel(d_m, d_d), 1e-9, options)
            (grad_m,) = torch.autograd.grad((d_m * v).sum(), dm, retain_graph=True)
            self.assertLess(rel(grad_m, grad_d), 1e-9, options)
            self.assertLess(
                rel(mg_fusion.project_gradient(batch, gX), dense_fusion.project_gradient(batch, gX)), 1e-9, options
            )

    def test_auto_selection(self):
        box = Grid.build((2, 2, 3))
        vox = Grid.from_voxels(holed_box())
        auto = Fusion()
        self.assertIsInstance(auto.factor(box, torch.float64), KronFactor)
        self.assertIsInstance(auto.factor(vox, torch.float64), DenseFactor)  # CPU: dense reference
        self.assertIsInstance(Fusion("mg").factor(vox, torch.float64), MultigridFactor)
        with self.assertRaises(ValueError):
            KronFactor(vox)


@unittest.skipUnless(torch.cuda.is_available(), "cuda")
class TestMultigridCUDA(unittest.TestCase):
    """float32 on the GPU; timings are measured while the GPU is shared, so the printed numbers are upper bounds."""

    def setUp(self):
        torch.manual_seed(0)
        warnings.simplefilter("ignore", UserWarning)

    @unittest.skipUnless(sparse_solver_available(), "nvmath")
    def test_equals_cudss_on_carved_20x20x30_random_pins(self):
        shape = (20, 20, 30)
        g = Grid.from_voxels(carved(*shape), random_pin_mask(shape, 0.02, 1), "cuda")
        r = torch.randn(1, g.Pf, 3, device="cuda")
        sp = SparseFactor(g)
        ref = sp.solve(r)
        t_mg_setup, mg = timed(lambda: MultigridFactor(g), 1)
        x = mg.solve(r)
        self.assertTrue(mg.stats["converged"])
        self.assertLess(rel(x, ref), 1e-4)
        lean = MultigridFactor(g, fp64_residual=False)
        x_lean = lean.solve(r)
        self.assertLess(rel(x_lean, ref), 1e-4)
        t_sp, _ = timed(lambda: sp.solve(r), 10)
        t_mg, _ = timed(lambda: mg.solve(r), 10)
        t_lean, _ = timed(lambda: lean.solve(r), 10)
        print(
            f"multigrid 20x20x30 carved, {g.C} cells, {g.Pf} free, {g.pinned.numel()} pins: levels {mg.sizes}, "
            f"{mg.stats['iterations']} iterations, rel err vs cuDSS {rel(x, ref):.1e} (float32 residual "
            f"{rel(x_lean, ref):.1e}); setup {t_mg_setup:.2f} s, solve {t_mg * 1e3:.1f} ms (float32 residual "
            f"{t_lean * 1e3:.1f} ms, cuDSS {t_sp * 1e3:.2f} ms), hierarchy {mg.memory_bytes() / 1e6:.1f} MB"
        )

    def test_fixed_matches_float64_solve_and_replay_is_exact(self):
        """The fixed 8-iteration float32 solve against the dense float64 solve on the test shapes, eagerly and by
        CUDA-graph replay (the replayed graph is the eager computation: identical bits)."""
        shapes = {
            "box 2x2x3": Grid.build((2, 2, 3), device="cuda"),
            "holed zmin": Grid.from_voxels(holed_box(), "zmin_face", "cuda"),
            "holed random pins": Grid.from_voxels(holed_box(), random_pin_mask((6, 6, 8), 0.05, 2), "cuda"),
            "carved 20x20x30 random pins": Grid.from_voxels(
                carved(20, 20, 30), random_pin_mask((20, 20, 30), 0.02, 1), "cuda"
            ),
        }
        for name, g in shapes.items():
            Ks = assemble_scalar(g)
            r = torch.randn(2, g.Pf, 3, device="cuda")
            ref = torch.linalg.solve(Ks[g.free][:, g.free], r.double().permute(1, 0, 2).reshape(g.Pf, 6))
            ref = ref.reshape(g.Pf, 2, 3).permute(1, 0, 2)
            del Ks
            eager = MultigridFactor(g, capture=False)
            replay = MultigridFactor(g)
            self.assertTrue(replay.capture and not eager.capture)
            x_e = eager.solve(r)
            self.assertLess(rel(x_e.double(), ref), 1e-5, name)
            self.assertLess(eager.stats["residual"], 1e-5, name)
            x_r = replay.solve(r)  # records the graph for this shape, then replays it
            self.assertEqual(list(replay._graphs), [((g.Pf, 6), torch.float32)])
            self.assertTrue(torch.equal(x_r, x_e), name)
            x_r2 = replay.solve(0.5 * r)  # replay with a new right-hand side
            self.assertEqual(len(replay._graphs), 1)
            self.assertTrue(torch.equal(x_r2, replay.solve(0.5 * r)), name)
            self.assertLess((x_r2 - 0.5 * x_e).abs().max().item(), 1e-6 * x_e.abs().max().item(), name)
            self.assertLess(replay.stats["residual"], 1e-5, name)
            # the torch sparse path (a column count that is not a multiple of three) gives the same solve
            rhs = r.permute(1, 0, 2).reshape(g.Pf, 6)
            x_t = eager._solve_cols(rhs[:, :4])
            self.assertLess(rel(x_t, x_e.permute(1, 0, 2).reshape(g.Pf, 6)[:, :4]), 1e-6, name)

    @unittest.skipUnless(sparse_solver_available(), "nvmath")
    def test_graph_per_shape_and_backward_on_the_autograd_thread(self):
        """Fusion("mg") on a voxel batch: the forward records one graph per right-hand-side shape, the backward (the
        same solve, run by autograd's worker thread) replays it; both match cuDSS."""
        g = Grid.from_voxels(holed_box(), random_pin_mask((6, 6, 8), 0.05, 7), "cuda")
        mg_fusion, sp_fusion = Fusion("mg"), Fusion("sparse")
        for n in (1, 3):
            batch = Batch.build([g] * n, "cuda")
            dm = (0.01 * torch.randn(batch.C, 7, 3, device="cuda")).requires_grad_(True)
            d_m = mg_fusion.fuse(batch, hx.modes_to_gauss(dm, batch.hc))
            d_s = sp_fusion.fuse(batch, hx.modes_to_gauss(dm, batch.hc))
            fac = mg_fusion.factor(g)
            self.assertEqual(
                list(fac._graphs), [((g.Pf, 3 * k), torch.float32) for k in range(1, n + 1) if k in (1, n)]
            )
            self.assertLess(rel(d_m, d_s), 1e-4)
            v = torch.randn_like(d_m)
            (grad_m,) = torch.autograd.grad((d_m * v).sum(), dm, retain_graph=True)
            (grad_d,) = torch.autograd.grad((d_s * v).sum(), dm)
            self.assertLess(rel(grad_m, grad_d), 1e-4)
            self.assertEqual(len(fac._graphs), 1 if n == 1 else 2)  # the backward replayed, no new graph
            gX = torch.randn(batch.N, 3, device="cuda")
            self.assertLess(rel(mg_fusion.project_gradient(batch, gX), sp_fusion.project_gradient(batch, gX)), 1e-4)

    def _report(self, name: str, g: Grid, reference):
        """Build both variants on g, time them, check the true residual and (if a reference solve is given) the error."""
        r = torch.randn(1, g.Pf, 3, device="cuda")
        ref = reference(g, r) if reference is not None else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t_setup, mg = timed(lambda: MultigridFactor(g), 1)
        peak = torch.cuda.max_memory_allocated() - base
        mg.solve(r)
        t_solve, x = timed(lambda: mg.solve(r), 3)
        self.assertTrue(mg.stats["converged"])
        full = torch.zeros(1, g.P, 3, dtype=torch.float64, device="cuda").index_copy(1, g.free, x.double())
        true_res = rel(mg.apply_full(full)[:, g.free], r.double())
        self.assertLess(true_res, 1e-4)
        lean = MultigridFactor(g, fp64_residual=False)
        lean.solve(r)
        t_lean, x_lean = timed(lambda: lean.solve(r), 3)
        errs = ""
        if ref is not None:
            errs = f", rel err vs float64 reference {rel(x.double(), ref):.1e} (float32 residual {rel(x_lean.double(), ref):.1e})"
            self.assertLess(rel(x.double(), ref), 1e-4)
            self.assertLess(rel(x_lean.double(), ref), 1e-2)
        print(
            f"multigrid {name}, {g.C} cells, {g.Pf} free: levels {mg.sizes}, nnz {[lv.nnz for lv in mg.levels]}, "
            f"{mg.stats['iterations']} iterations (float32 residual {lean.stats['iterations']}), true rel residual "
            f"{true_res:.1e}{errs}; setup {t_setup:.2f} s, solve (3 rhs) {t_solve * 1e3:.1f} ms (float32 residual "
            f"{t_lean * 1e3:.1f} ms), hierarchy {mg.memory_bytes() / 1e6:.0f} MB (float32 residual "
            f"{lean.memory_bytes() / 1e6:.0f} MB), setup peak {peak / 1e9:.2f} GB"
        )

    def test_large_carved_shapes_report(self):
        cudss = (lambda g, r: SparseFactor(g, torch.float64).solve(r.double())) if sparse_solver_available() else None
        self._report("50^3 carved", Grid.from_voxels(carved(50, 50, 50), "zmin_face", "cuda"), cudss)
        torch.cuda.empty_cache()
        self._report("100^3 carved", Grid.from_voxels(carved(100, 100, 100), "zmin_face", "cuda"), None)
        torch.cuda.empty_cache()
        # a million cells: the hierarchy memory figure, with the float64 structured solve as the reference
        self._report(
            "100^3 box",
            Grid.build((100, 100, 100), device="cuda"),
            lambda g, r: KronFactor(g, torch.float64).solve(r.double()),
        )

    @unittest.skipUnless(sparse_solver_available(), "nvmath")
    def test_fusion_mg_equals_sparse_on_a_voxel_batch(self):
        g1 = Grid.from_voxels(carved(10, 10, 14), random_pin_mask((10, 10, 14), 0.02, 3), "cuda")
        g2 = Grid.from_voxels(holed_box(), "zmin_face", "cuda")
        batch = Batch.build([g1, g2, g1], "cuda")
        self.assertEqual(len(batch.groups), 2)
        hc = batch.hc
        dm = 0.01 * torch.randn(batch.C, 7, 3, device="cuda")
        dF = hx.modes_to_gauss(dm, hc)
        dp = torch.zeros(batch.N, 3, device="cuda")
        dp[batch.pinned] = 0.01 * torch.randn(int(batch.pinned.sum()), 3, device="cuda")
        mg_fusion, sp_fusion = Fusion("mg"), Fusion("sparse")
        d_m, d_s = mg_fusion.fuse(batch, dF, dp), sp_fusion.fuse(batch, dF, dp)
        self.assertLess(rel(d_m, d_s), 1e-4)
        gX = torch.randn(batch.N, 3, device="cuda")
        self.assertLess(rel(mg_fusion.project_gradient(batch, gX), sp_fusion.project_gradient(batch, gX)), 1e-4)
        auto = Fusion()  # "auto": cuDSS while the factor is small (default up to 300k free corners), multigrid beyond
        self.assertIsInstance(auto.factor(g1), SparseFactor)
        self.assertIsInstance(Fusion(options={"sparse_max_free": 0}).factor(g1), MultigridFactor)
        self.assertIsInstance(auto.factor(Grid.build((2, 2, 3), device="cuda")), KronFactor)
        d_a = auto.fuse(batch, dF, dp)
        self.assertLess(rel(d_a, d_s), 1e-4)
        d_a2 = Fusion(options={"sparse_max_free": 0}).fuse(batch, dF, dp)
        self.assertLess(rel(d_a2, d_s), 1e-4)


if __name__ == "__main__":
    unittest.main()
