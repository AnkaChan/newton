# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multigrid fusion solver on an idle GPU: residual against the iteration count, the cost of the float64 residual,
kernels per iteration, and the solve timings (eager adaptive, eager fixed, captured fixed, cuDSS, the structured solve
on the box) for 3 and 48 right-hand sides, all back to back in one process.

    CUDA_VISIBLE_DEVICES=1 python -m experiments.lido.bench_multigrid --sizes all [--old path/to/multigrid_old.py]

`--old` loads a previous `MultigridFactorOld` from a file for the "before" columns.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
import warnings

import torch

from .fusion import KronFactor, SparseFactor, sparse_solver_available
from .grid import Grid
from .multigrid import MultigridFactor
from .tests.test_multigrid import random_pin_mask
from .tests.test_voxel_grid import carved, holed_box


def beam_with_hole(nx=10, ny=10, nz=40, radius=3.0) -> torch.Tensor:
    """The canonical beam with a through-hole of the given radius along z (voxel centres)."""
    x, y, _ = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij")
    occ = torch.ones(nx, ny, nz, dtype=torch.bool)
    return occ & ~(((x + 0.5 - nx / 2) ** 2 + (y + 0.5 - ny / 2) ** 2) <= radius**2)


def meshes(which: str, device) -> list[tuple[str, Grid]]:
    out = [
        ("box 2x2x3", Grid.build((2, 2, 3), device=device)),
        ("holed 6x6x8", Grid.from_voxels(holed_box(), "zmin_face", device)),
        ("holed 6x6x8 random pins", Grid.from_voxels(holed_box(), random_pin_mask((6, 6, 8), 0.05, 2), device)),
        ("plate 30x30x2", Grid.build((30, 30, 2), device=device)),
        ("rod 3x3x60", Grid.build((3, 3, 60), device=device)),
        (
            "slab 40x4x40 sparse pins",
            Grid.from_voxels(torch.ones(40, 4, 40, dtype=torch.bool), random_pin_mask((40, 4, 40), 0.01, 3), device),
        ),
        ("beam 10x10x40 hole r3", Grid.from_voxels(beam_with_hole(), "zmin_face", device)),
        ("box 10x10x40", Grid.build((10, 10, 40), device=device)),
        (
            "carved 20x20x30 random pins",
            Grid.from_voxels(carved(20, 20, 30), random_pin_mask((20, 20, 30), 0.02, 1), device),
        ),
    ]
    if which == "all":
        out += [
            ("carved 50^3", Grid.from_voxels(carved(50, 50, 50), "zmin_face", device)),
            ("carved 100^3", Grid.from_voxels(carved(100, 100, 100), "zmin_face", device)),
            ("box 100^3", Grid.build((100, 100, 100), device=device)),
        ]
    return out


def timed(fn, arg, reps: int, device) -> float:
    """Milliseconds per `fn(arg)` after one warm-up call."""
    fn(arg)
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    for _ in range(reps):
        fn(arg)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t) / reps * 1e3


def kernels_per_call(fn, arg, device) -> int:
    from torch.profiler import ProfilerActivity, profile

    fn(arg)
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(arg)
        torch.cuda.synchronize(device)
    return sum(e.count for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA)


def true_residual(g: Grid, fac, x: torch.Tensor, r: torch.Tensor) -> float:
    full = torch.zeros(r.shape[0], g.P, 3, dtype=torch.float64, device=r.device).index_copy(1, g.free, x.double())
    return ((fac.apply_full(full)[:, g.free] - r.double()).norm() / r.double().norm()).item()


def rel(a, b) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def residual_table(grids, device, counts=(4, 6, 8, 10)) -> None:
    print(
        f"\nfloat64 relative residual after a fixed number of PCG iterations (fp32 cycle, 3 rhs)\n{'mesh':30s} {'free':>8s} lv  "
        + " ".join(f"{k:>9d}" for k in counts)
        + "   err vs fp64 ref at 8"
    )
    for name, g in grids:
        mg = MultigridFactor(g, capture=False)
        r = torch.randn(1, g.Pf, 3, device=device)
        ref = reference(g, r)
        res, err = [], float("nan")
        for k in counts:
            mg.iterations = k
            x = mg.solve(r)
            res.append(mg.stats["residual"])
            if k == 8 and ref is not None:
                err = rel(x, ref)
        print(f"{name:30s} {g.Pf:8d} {len(mg.levels):2d}  " + " ".join(f"{v:9.1e}" for v in res) + f"   {err:9.1e}")
        del mg
        torch.cuda.empty_cache()


def reference(g: Grid, r: torch.Tensor):
    """float64 solve: the structured solve on boxes with the z-min face pinned, cuDSS up to 100k free corners."""
    if g.kind == "box" and g.pins == "zmin_face":
        return KronFactor(g, torch.float64).solve(r.double())
    if g.Pf <= 100_000 and sparse_solver_available():
        return SparseFactor(g, torch.float64).solve(r.double())
    return None


def fp64_variants(grids, device, reps) -> None:
    print(
        f"\nthe float64 residual: captured fixed-8 solve, 3 rhs, ms (rel err vs fp64 reference / true fp64 residual)\n{'mesh':30s} {'every 1 (default)':>24s} {'every 2':>24s} {'every 4':>24s} {'fp32 residual':>24s}"
    )
    for name, g in grids:
        r = torch.randn(1, g.Pf, 3, device=device)
        ref = reference(g, r)
        cells = []
        for kw in ({}, {"fp64_every": 2}, {"fp64_every": 4}, {"fp64_residual": False}):
            mg = MultigridFactor(g, **kw)
            x = mg.solve(r)
            t = timed(mg.solve, r, reps, device)
            e = rel(x, ref) if ref is not None else float("nan")
            cells.append(f"{t:7.2f} ({e:7.1e} / {true_residual(g, mg, x, r):7.1e})")
            del mg
            torch.cuda.empty_cache()
        print(f"{name:30s} " + " ".join(f"{c:>24s}" for c in cells))


def load_old(path: str | None):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("multigrid_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MultigridFactorOld


def timing_table(grids, device, reps, old_cls) -> None:
    has_cudss = sparse_solver_available()
    print(
        "\nsolve timings, ms (median of back-to-back blocks); setup s; hierarchy MB; kernels per PCG iteration\n"
        f"{'mesh':30s} {'free':>8s} rhs {'old adaptive':>14s} {'adaptive':>10s} {'fixed 8':>9s} {'captured':>9s} {'cuDSS':>8s} {'kron':>7s}   setup   MB  kernels/it old->new  err captured vs ref"
    )
    for name, g in grids:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        mg = MultigridFactor(g)
        torch.cuda.synchronize(device)
        t_setup = time.perf_counter() - t0
        adaptive = MultigridFactor(g, iterations=None)
        eager = MultigridFactor(g, capture=False)
        old = old_cls(g) if old_cls is not None else None
        kron = KronFactor(g) if g.kind == "box" and g.pins == "zmin_face" else None
        sp = None
        if has_cudss:
            try:
                sp = SparseFactor(g)
            except Exception as exc:
                print(f"  cuDSS unavailable for {name}: {exc}")
        for n in (1, 16):
            r = torch.randn(n, g.Pf, 3, device=device)
            ref = reference(g, r) if n == 1 else None
            row = {}
            row["old"] = timed(old.solve, r, reps, device) if old is not None else float("nan")
            old_it = old.stats["iterations"] if old is not None else 0
            row["adaptive"] = timed(adaptive.solve, r, reps, device)
            row["fixed"] = timed(eager.solve, r, reps, device)
            row["captured"] = timed(mg.solve, r, reps, device)
            row["cudss"] = timed(sp.solve, r, reps, device) if sp is not None else float("nan")
            row["kron"] = timed(kron.solve, r, reps, device) if kron is not None else float("nan")
            k_new = kernels_per_call(eager.solve, r, device) / mg.iterations
            k_old = kernels_per_call(old.solve, r, device) / max(old_it, 1) if old is not None else float("nan")
            err = rel(mg.solve(r), ref) if ref is not None else float("nan")
            print(
                f"{name:30s} {g.Pf:8d} {3 * n:3d} {row['old']:10.2f} ({old_it}) {row['adaptive']:6.2f} ({adaptive.stats['iterations']}) "
                f"{row['fixed']:9.2f} {row['captured']:9.2f} {row['cudss']:8.2f} {row['kron']:7.2f}   {t_setup:5.2f} {mg.memory_bytes() / 1e6:5.0f}  "
                f"{k_old:5.0f} -> {k_new:4.0f}   {err:8.1e}"
            )
        del mg, adaptive, eager, old, kron, sp
        torch.cuda.empty_cache()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="LIDO multigrid fusion solver benchmark")
    p.add_argument("--sizes", choices=("small", "all"), default="all")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument(
        "--old", default=None, help="path to the previous multigrid.py (MultigridFactorOld) for the before columns"
    )
    p.add_argument("--skip", default="", help="comma-separated parts to skip: residual,fp64,timing")
    a = p.parse_args(argv)
    warnings.simplefilter("ignore")
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    skip = set(a.skip.split(",")) if a.skip else set()
    grids = meshes(a.sizes, device)
    props = torch.cuda.get_device_properties(device)
    print(f"{props.name}, torch {torch.__version__}, {len(grids)} meshes, {a.reps} repetitions per timing")
    if "residual" not in skip:
        residual_table(grids, device)
    big = [(n, g) for n, g in grids if g.Pf >= 4000]
    if "fp64" not in skip:
        fp64_variants(big, device, a.reps)
    if "timing" not in skip:
        timing_table(big, device, a.reps, load_old(a.old))


if __name__ == "__main__":
    main()
