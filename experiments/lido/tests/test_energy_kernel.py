# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Warp elastic + damping kernel against the torch reference: per-cell energies, gradients through
energy_and_grad (with and without a graph on the candidate), inverted cells, dispatch, and the timing of
energy_and_grad on the canonical 10x10x40 grid at batch 1 and 16."""

import contextlib
import time
import unittest

import torch

from experiments.lido import hex as hx
from experiments.lido import physics
from experiments.lido.batch import Batch
from experiments.lido.energy_kernel import elastic_damping_warp
from experiments.lido.grid import Grid
from experiments.lido.structs import Material
from experiments.lido.units import material_from_si

MATERIALS = [{"E": 1e5, "nu": 0.3, "eta": 100.0}, {"E": 4e5, "nu": 0.45, "eta": 20.0}]
CANONICAL = (10, 10, 40)


def make_batch(cell_counts, O, seed, device="cuda", dtype=torch.float32, mirror=()):
    """O copies of one grid alternating the two materials, randomly deformed, C_prev from the step-start shape.
    Objects listed in `mirror` are reflected in x so that every one of their cells has det F < 0."""
    torch.manual_seed(seed)
    g = Grid.build(cell_counts, "zmin_face", device)
    b = Batch.build([g] * O, device, dtype)
    b.material = Material.cat(
        [
            material_from_si(
                **MATERIALS[o % 2],
                rho=1000.0,
                gravity=(0, -9.81, 0),
                h=0.025,
                dt=1 / 300,
                cell_count=g.C,
                sample_count=g.S,
                device=device,
            )
            for o in range(O)
        ]
    )
    for f in ("lam", "rho", "eta", "g", "ke", "kd", "mu_f", "kappa", "beta", "friction_eps", "floor", "h", "dt", "mu"):
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    rest = g.rest.to(dtype).repeat(O, 1)
    b.X = rest + 0.05 * torch.randn(b.N, 3, dtype=dtype, device=device)
    b.X[b.pinned] = rest[b.pinned]
    b.V = 0.02 * torch.randn(b.N, 3, dtype=dtype, device=device)
    b.V[b.pinned] = 0
    b.Y = b.X + b.V + b.material.g[b.corner_obj]
    b.Y[b.pinned] = b.X[b.pinned]
    F_prev = hx.gauss_deformation(b.X[b.cells], b.hc)
    b.C_prev = hx.mat3_tn(F_prev, F_prev)
    b.x = b.Y + 0.03 * torch.randn(b.N, 3, dtype=dtype, device=device)
    b.x[b.pinned] = b.X[b.pinned]
    for o in mirror:
        rows = b.corner_obj == o
        b.x[rows, 0] = 2.0 * rest[rows, 0].mean() - b.x[rows, 0]
    return b


@contextlib.contextmanager
def torch_path():
    saved = physics.USE_WARP
    physics.USE_WARP = False
    try:
        yield
    finally:
        physics.USE_WARP = saved


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def timed(fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3


@unittest.skipUnless(torch.cuda.is_available(), "needs cuda")
class TestEnergyKernel(unittest.TestCase):
    def check_against_torch(self, b, energy_tol=1e-5, grad_tol=1e-4):
        E_warp = elastic_damping_warp(b, b.x)
        E_el, E_damp = physics.elastic_damping(b, b.x)
        self.assertEqual(E_warp.shape, (b.C,))
        self.assertTrue(torch.isfinite(E_warp).all())
        self.assertLess(rel_err(E_warp, E_el + E_damp), energy_tol)
        E1, g1 = physics.energy_and_grad(b, b.x)
        with torch_path():
            E2, g2 = physics.energy_and_grad(b, b.x)
        self.assertTrue(torch.isfinite(g1).all())
        self.assertTrue((g1[b.pinned] == 0).all())
        self.assertLess(rel_err(E1, E2), energy_tol)
        self.assertLess(rel_err(g1, g2), grad_tol)

    def test_small_grid_two_materials(self):
        self.check_against_torch(make_batch((2, 2, 3), 2, 0))

    def test_canonical_grid_two_materials(self):
        self.check_against_torch(make_batch(CANONICAL, 2, 1))

    def test_inverted_cells_finite_and_matching(self):
        b = make_batch((2, 2, 3), 2, 2, mirror=(1,))
        J = torch.linalg.det(hx.gauss_deformation(b.x[b.cells], b.hc))
        self.assertTrue((J[b.cell_obj == 1] < 0).all())
        self.assertTrue((J[b.cell_obj == 0] > 0).all())
        self.check_against_torch(b)

    def test_backward_through_candidate(self):
        """Training route: the energy keeps its graph to the candidate's source and per-object weights scale it."""
        b = make_batch((2, 2, 3), 2, 3)
        w = torch.tensor([1.0, 3.0], device=b.device)
        grads = []
        for ctx in (contextlib.nullcontext(), torch_path()):
            with ctx:
                src = torch.zeros(b.N, 3, device=b.device, requires_grad=True)
                E, gX = physics.energy_and_grad(b, b.x + src)
                self.assertTrue(E.requires_grad)
                self.assertFalse(gX.requires_grad)
                (E * w).sum().backward()
                grads.append(src.grad)
        self.assertLess(rel_err(grads[0], grads[1]), 1e-4)

    def test_dispatch_keeps_torch_for_float64(self):
        b = make_batch((2, 2, 3), 2, 4, dtype=torch.float64)
        E_el, E_damp = physics.elastic_damping(b, b.x)
        self.assertTrue(torch.equal(physics.elastic_damping_total(b, b.x), E_el + E_damp))
        b32 = make_batch((2, 2, 3), 2, 4)
        self.assertTrue(torch.equal(physics.elastic_damping_total(b32, b32.x), elastic_damping_warp(b32, b32.x)))

    def test_timing(self):
        lines = []
        for O in (1, 16):
            b = make_batch(CANONICAL, O, 5)
            t_warp = timed(lambda b=b: physics.energy_and_grad(b, b.x), 50)
            with torch_path():
                t_torch = timed(lambda b=b: physics.energy_and_grad(b, b.x), 20)
            lines.append(f"energy_and_grad {CANONICAL} batch {O}: warp {t_warp:.3f} ms, torch {t_torch:.3f} ms")
        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
