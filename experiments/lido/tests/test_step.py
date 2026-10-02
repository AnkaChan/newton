# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step: energy reuse identity, zero-init no-op, equivariance of the fused update and the loss, sub-batch advance."""

import unittest

import torch

from experiments.lido import physics
from experiments.lido.augment import Augmenter
from experiments.lido.batch import Batch
from experiments.lido.config import TrainConfig
from experiments.lido.fusion import Fusion
from experiments.lido.grid import Grid
from experiments.lido.network import Net
from experiments.lido.step import Step
from experiments.lido.structs import Material
from experiments.lido.units import material_from_si
from experiments.lido.validation import local_objective


def make(O=2, seed=0, randomize=True, dtype=torch.float32, net_kwargs=None):
    torch.manual_seed(seed)
    g = Grid.build((2, 2, 3))
    b = Batch.build([g] * O, "cpu", dtype)
    mats = [
        material_from_si(
            E=1e5 * (o + 1),
            nu=0.3,
            rho=1000.0,
            eta=100.0,
            gravity=(0, -9.81, 0),
            h=0.025,
            dt=1 / 300,
            cell_count=g.C,
            sample_count=g.S,
        )
        for o in range(O)
    ]
    b.material = Material.cat(mats)
    for f in ("lam", "rho", "eta", "g", "ke", "kd", "mu_f", "kappa", "beta", "friction_eps", "floor", "h", "dt", "mu"):
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    rest = g.rest.to(dtype).repeat(O, 1)
    b.X = rest + 0.05 * torch.randn(b.N, 3, dtype=dtype)
    b.X[b.pinned] = rest[b.pinned]
    b.V = 0.02 * torch.randn(b.N, 3, dtype=dtype)
    b.V[b.pinned] = 0
    b.X_prev = b.X.clone()
    b.x = b.X.clone()
    net = Net(**(net_kwargs or {})).to(dtype)
    if randomize:
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
    step = Step(net, Fusion(), Augmenter("cpu"), noise_prob=0.0)
    gens = [torch.Generator().manual_seed(o) for o in range(O)]
    step.prepare(b, torch.ones(O, dtype=torch.bool), gens)
    return g, b, step, gens


def rotate_batch(b, q, t):
    for name in ("X", "V", "X_prev", "x", "Y"):
        v = getattr(b, name)
        setattr(b, name, v @ q.t() + (t if name != "V" else 0))


class TestStep(unittest.TestCase):
    def test_zero_init_leaves_candidate(self):
        _g, b, step, _gens = make(randomize=False)
        out = step.query(b)
        self.assertLess((out.cand_after - b.x).abs().max().item(), 1e-12)
        self.assertLess((out.E_after - out.E_before).abs().max().item(), 1e-6)

    def test_energy_reuse_identity(self):
        _g, b, step, _gens = make()
        out = step.query(b)
        step.commit(b, out)
        E, gX = physics.energy_and_grad(b, b.x)
        self.assertLess((E - b.E).abs().max().item(), 1e-5 * E.abs().max().item())
        self.assertLess((gX - b.gX).abs().max().item(), 1e-5 * gX.abs().max().item() + 1e-7)
        out2 = step.query(b)
        self.assertTrue(torch.equal(out2.E_before, b.E))

    def test_equivariance(self):
        _g, b, step, _gens = make(dtype=torch.float64)
        out = step.query(b)
        loss = local_objective(out.E_after, out.E_before, b.material.floor)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        q = q * torch.sign(torch.det(q))
        t = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
        _g2, b2, step2, _ = make(dtype=torch.float64)
        step2.net.load_state_dict(step.net.state_dict())
        rotate_batch(b2, q, t)
        b2.material.g = b2.material.g @ q.t()  # gravity rotates with the scene
        step2.prepare(b2, torch.ones(2, dtype=torch.bool), None)
        b2.x = b.x @ q.t() + t
        b2.E, b2.gX = physics.energy_and_grad(b2, b2.x)
        self.assertLess((b2.E - b.E).abs().max().item(), 1e-9)
        out2 = step2.query(b2)
        d1 = (out.cand_after - b.x) @ q.t()
        d2 = out2.cand_after - b2.x
        self.assertLess((d1 - d2).abs().max().item(), 1e-5 * max(1.0, d1.abs().max().item()))  # float32 frames
        loss2 = local_objective(out2.E_after, out2.E_before, b2.material.floor)
        self.assertLess((loss - loss2).abs().max().item(), 1e-5)

    def test_backward_reaches_network(self):
        _g, b, step, _gens = make()
        out = step.query(b)
        loss = local_objective(out.E_after, out.E_before, b.material.floor).mean()
        loss.backward()
        total = sum(p.grad.abs().sum().item() for p in step.net.parameters() if p.grad is not None)
        self.assertGreater(total, 0.0)

    def test_advance_subset_leaves_others(self):
        _g, b, step, gens = make(O=3)
        for _ in range(2):
            out = step.query(b)
            step.commit(b, out)
        X0, V0, x0 = b.X.clone(), b.V.clone(), b.x.clone()
        sel = torch.tensor([False, True, False])
        step.advance(b, sel, gens)
        rows = sel[b.corner_obj]
        self.assertTrue(torch.equal(b.X[~rows], X0[~rows]))
        self.assertTrue(torch.equal(b.V[~rows], V0[~rows]))
        self.assertTrue(torch.equal(b.x[~rows], x0[~rows]))
        self.assertTrue(torch.equal(b.X[rows], x0[rows]))
        self.assertTrue(torch.allclose(b.V[rows], x0[rows] - X0[rows]))
        self.assertTrue((b.x[rows & b.pinned] == b.X[rows & b.pinned]).all())
        E, _gX = physics.energy_and_grad(b, b.x)
        self.assertLess((E - b.E).abs().max().item(), 1e-5 * E.abs().max().item())


class TestBatchedCandidateNoise(unittest.TestCase):
    """`Step.prepare` with ONE generator for the batch (v5 scenes): the candidate noise of every grid group in one
    call; about half the selected objects perturbed with the configured RMS, free bodies keep the inertial
    centroid, pinned rows stay at X, equal seeds give equal candidates, a different seed different ones."""

    def make(self, seed):
        torch.manual_seed(0)
        grids = [
            Grid.build(s, pins=p)
            for s, p in (((2, 2, 3), "zmin_face"), ((3, 2, 2), "none"), ((2, 2, 3), "zmin_face"), ((2, 3, 2), "none"))
        ]
        grids = grids * 4  # 16 objects in 4 groups
        b = Batch.build(grids, "cpu", torch.float64)
        parts = []
        for g in grids:
            parts.append(
                material_from_si(
                    E=1e5,
                    nu=0.3,
                    rho=1000.0,
                    eta=100.0,
                    gravity=(0, -9.81, 0),
                    h=0.025,
                    dt=1 / 300,
                    cell_count=g.C,
                    sample_count=g.S,
                    dtype=torch.float64,
                )
            )
        b.material = Material.cat(parts)
        b.X = b.rest.to(torch.float64).clone()
        b.V = torch.zeros_like(b.X)
        b.X_prev, b.x = b.X.clone(), b.X.clone()
        step = Step(
            Net.from_config(TrainConfig(hidden_dim=24, edge_hidden_dim=12, num_heads=2, contact_hidden_dim=8)).double(),
            Fusion(),
            Augmenter("cpu"),
            noise_prob=0.5,
            noise_range=(0.02, 0.04),
        )
        gen = torch.Generator().manual_seed(seed)
        return b, step, gen

    def test_batched_noise(self):
        b, step, gen = self.make(1)
        sel = torch.ones(b.O, dtype=torch.bool)
        sel[3] = False  # an unselected object keeps its candidate
        b.x[b.corner_obj == 3] = -7.0
        step.prepare(b, sel, gen)
        noise = b.x - b.Y
        per_obj = physics.seg_sum((noise**2).sum(-1), b.corner_obj, b.O)
        perturbed = (per_obj > 0) & sel
        self.assertTrue(3 <= int(perturbed.sum()) <= 12)  # about half of the 15 selected objects
        self.assertTrue((b.x[b.corner_obj == 3] == -7.0).all())
        self.assertTrue((b.x[b.pinned] == b.X[b.pinned]).all())
        counts = torch.bincount(b.corner_obj, minlength=b.O).double()
        rms = (per_obj / counts).sqrt()[perturbed]
        self.assertTrue(((rms > 0.015) & (rms < 0.045)).all(), rms)  # U(0.02, 0.04) RMS over the corners
        # a free body's candidate keeps the inertial centroid c(Y)
        for o in range(b.O):
            if bool(b.free_objects[o]) and bool(perturbed[o]):
                m = physics.corner_mass(b)[b.corner_obj == o][:, None]
                self.assertLess(((m * noise[b.corner_obj == o]).sum(0) / m.sum()).abs().max().item(), 1e-12)
        # deterministic in the seed
        rows = sel[b.corner_obj]
        b2, step2, gen2 = self.make(1)
        step2.prepare(b2, sel, gen2)
        self.assertTrue(torch.equal(b2.x[rows], b.x[rows]))
        b3, step3, gen3 = self.make(2)
        step3.prepare(b3, sel, gen3)
        self.assertFalse(torch.equal(b3.x[rows], b.x[rows]))
        # perturb overrides the draw
        b4, step4, gen4 = self.make(1)
        step4.prepare(b4, sel, gen4, perturb=torch.zeros(b.O, dtype=torch.bool))
        self.assertTrue(torch.equal(b4.x[rows], b4.Y[rows]))  # inertial candidates (the unselected object keeps X)
        self.assertTrue(torch.isfinite(b.E).all() and torch.isfinite(b.gX).all())


if __name__ == "__main__":
    unittest.main()
