# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Energy: stress against Newton's formula, finite-difference gradient, damping invariance, units."""

import unittest

import torch

from experiments.lido import hex as hx
from experiments.lido import physics
from experiments.lido.batch import Batch
from experiments.lido.grid import Grid
from experiments.lido.units import material_from_si


def make_batch(grid, dtype=torch.float64, **mat):
    b = Batch.build([grid], "cpu", dtype)
    m = {
        "E": 1e5,
        "nu": 0.3,
        "rho": 1000.0,
        "eta": 100.0,
        "gravity": (0, -9.81, 0),
        "h": 0.025,
        "dt": 1 / 300,
        "cell_count": grid.C,
        "sample_count": grid.S,
    }
    m.update(mat)
    b.material = material_from_si(**m)
    for f in ("lam", "rho", "eta", "g", "ke", "kd", "mu_f", "kappa", "beta", "friction_eps", "floor", "h", "dt", "mu"):
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    b.X = grid.rest.to(dtype) + 0.05 * torch.randn(grid.P, 3, dtype=dtype)
    b.X[grid.pinned] = grid.rest[grid.pinned].to(dtype)
    b.V = 0.02 * torch.randn(grid.P, 3, dtype=dtype)
    b.V[grid.pinned] = 0
    b.Y = b.X + b.V + b.material.g[0]
    b.Y[grid.pinned] = b.X[grid.pinned]
    F_prev = hx.gauss_deformation(b.X[b.cells], b.hc)
    b.C_prev = F_prev.transpose(-1, -2) @ F_prev
    b.x = b.Y + 0.03 * torch.randn(grid.P, 3, dtype=dtype)
    b.x[grid.pinned] = b.X[grid.pinned]
    return b


class TestPhysics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.grid = Grid.build((2, 2, 3))

    def test_stress_matches_newton_formula(self):
        F = torch.eye(3, dtype=torch.float64) + 0.3 * torch.randn(16, 3, 3, dtype=torch.float64)
        F.requires_grad_(True)
        lam = torch.tensor(2.5, dtype=torch.float64)
        psi = physics.stable_neo_hookean(F, lam)
        (P,) = torch.autograd.grad(psi.sum(), F)
        ref = physics.neo_hookean_stress(F.detach(), lam)
        self.assertLess((P - ref).abs().max().item(), 1e-10)
        self.assertLess(physics.stable_neo_hookean(torch.eye(3, dtype=torch.float64), lam).abs().item(), 1e-15)

    def test_energy_finite_at_inversion(self):
        F = torch.zeros(3, 3, dtype=torch.float64)
        lam = torch.tensor(1.0, dtype=torch.float64)
        self.assertTrue(torch.isfinite(physics.stable_neo_hookean(F, lam)))
        Fi = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
        self.assertTrue(torch.isfinite(physics.stable_neo_hookean(Fi, lam)))

    def test_gradient_finite_difference(self):
        b = make_batch(self.grid)
        _E, gX = physics.energy_and_grad(b, b.x)
        self.assertTrue((gX[b.pinned] == 0).all())
        for _ in range(3):
            e = torch.randn_like(b.x)
            e[b.pinned] = 0
            eps = 1e-6
            Ep = physics.energy(b, b.x + eps * e).sum()
            Em = physics.energy(b, b.x - eps * e).sum()
            fd = (Ep - Em) / (2 * eps)
            self.assertAlmostEqual(fd.item(), (gX * e).sum().item(), delta=1e-6 * max(1.0, abs(fd.item())))

    def test_damping_vanishes_under_rigid_motion(self):
        b = make_batch(self.grid, eta=500.0)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        q = q * torch.sign(torch.det(q))
        x = b.X @ q.t() + 0.7
        _, E_damp = physics.elastic_damping(b, x)
        self.assertLess(E_damp.abs().max().item(), 1e-12)

    def test_elastic_energy_rotation_invariant(self):
        b = make_batch(self.grid)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        q = q * torch.sign(torch.det(q))
        E1, _ = physics.elastic_damping(b, b.x)
        E2, _ = physics.elastic_damping(b, b.x @ q.t() + 0.2)
        self.assertLess((E1 - E2).abs().max().item(), 1e-10)

    def test_units_energy_scale(self):
        """Normalised energy times mu h^3 equals the SI incremental potential of a uniformly stretched block."""
        g = self.grid
        b = make_batch(g)
        s = 1.05
        x = g.rest.to(torch.float64) * torch.tensor([s, 1.0, 1.0], dtype=torch.float64)
        b.Y = x.clone()
        F_prev = hx.gauss_deformation(x[b.cells], b.hc)
        b.C_prev = F_prev.transpose(-1, -2) @ F_prev
        E_norm = physics.energy(b, x)[0].item()
        si = b.material.si[0]
        mu, lam = si["mu"], si["lam"]
        torch.diag(torch.tensor([s, 1.0, 1.0], dtype=torch.float64))
        J = s
        I_C = s * s + 2
        psi = 0.5 * mu * (I_C - 3) + 0.5 * (lam + mu) * (J - 1) ** 2 - mu * (J - 1)
        E_si = psi * g.C * si["h"] ** 3
        self.assertAlmostEqual(E_norm * si["energy_scale"], E_si, delta=1e-9 * abs(E_si))

    def test_residual_and_inverted(self):
        b = make_batch(self.grid)
        _, gX = physics.energy_and_grad(b, b.x)
        r = physics.residual(gX, b)
        self.assertEqual(r.shape, (1,))
        self.assertAlmostEqual(r.item(), gX.norm().item(), places=10)
        _, Fc = physics.modes_and_center(b.x, b)
        self.assertEqual(physics.inverted_cells(Fc, b).item(), 0.0)

    def test_energy_and_grad_with_graph(self):
        b = make_batch(self.grid)
        src = torch.randn(self.grid.P, 3, dtype=torch.float64, requires_grad=True)
        x = b.x + 0.01 * src
        E, gX = physics.energy_and_grad(b, x)
        self.assertTrue(E.requires_grad)
        self.assertFalse(gX.requires_grad)
        E.sum().backward()
        self.assertTrue(torch.isfinite(src.grad).all())


if __name__ == "__main__":
    unittest.main()
