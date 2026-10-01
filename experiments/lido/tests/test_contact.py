# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact: detection layout and rules, Newton's law and its gradient, gating, tokens, penetration metric."""

import unittest

import torch

from experiments.lido import contact
from experiments.lido import hex as hx
from experiments.lido.batch import Batch
from experiments.lido.grid import Grid
from experiments.lido.scenes import cat_scenes
from experiments.lido.structs import ContactScene
from experiments.lido.units import material_from_si

DT = torch.float64
DOWN = (0.0, -1.0, 0.0)
UP = (0.0, 1.0, 0.0)


def make_scene(points=(), normals=(), radii=(), plane_d=-0.3, plane_present=True):
    pts = torch.tensor(points, dtype=DT).reshape(-1, 3)
    return ContactScene(
        plane_n=torch.tensor([UP], dtype=DT),
        plane_d=torch.tensor([plane_d], dtype=DT),
        plane_present=torch.tensor([plane_present]),
        points=pts,
        normals=torch.tensor(normals, dtype=DT).reshape(-1, 3),
        radii=torch.tensor(radii, dtype=DT).reshape(-1),
        point_offsets=torch.tensor([0, pts.shape[0]]),
    )


def set_material(b, grids, **mat):
    m = {
        "E": 1e5,
        "nu": 0.3,
        "rho": 1000.0,
        "eta": 100.0,
        "gravity": (0, -9.81, 0),
        "h": 0.025,
        "dt": 1 / 300,
        "kappa": 50.0,
        "beta": 0.5,
        "mu_f": 0.6,
    }
    m.update(mat)
    parts = [material_from_si(cell_count=g.C, sample_count=g.S, **m) for g in grids]
    b.material = parts[0].cat(parts)
    for f in ("lam", "rho", "eta", "g", "ke", "kd", "mu_f", "kappa", "beta", "friction_eps", "floor", "h", "dt", "mu"):
        setattr(b.material, f, getattr(b.material, f).to(DT))


def make_batch(grid, scene=None, noise=0.05, vel=0.02, **mat):
    b = Batch.build([grid], "cpu", DT)
    set_material(b, [grid], **mat)
    b.X = grid.rest.to(DT) + noise * torch.randn(grid.P, 3, dtype=DT)
    b.X[grid.pinned] = grid.rest[grid.pinned].to(DT)
    b.V = vel * torch.randn(grid.P, 3, dtype=DT)
    b.V[grid.pinned] = 0
    b.Y = b.X + b.V + b.material.g[0]
    b.Y[grid.pinned] = b.X[grid.pinned]
    F_prev = hx.gauss_deformation(b.X[b.cells], b.hc)
    b.C_prev = F_prev.transpose(-1, -2) @ F_prev
    b.x = b.Y.clone()
    if scene is not None:
        b.scene = scene
    b.pairs = contact.detect(b, b.X, b.V)
    return b


def point_ids(pairs, scene):
    """Index into scene.points of every kind-1 pair."""
    d = (pairs.partner_point[:, None, :] - scene.points[None]).norm(dim=-1)
    return d.argmin(1)


def proper_rotation(n=None):
    q, _ = torch.linalg.qr(torch.randn(*((n, 3, 3) if n else (3, 3)), dtype=DT))
    return q * torch.sign(torch.linalg.det(q))[..., None, None]


class TestDetection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.grid = Grid.build((2, 2, 3))  # rest x, y in [0, 2], z in [0, 3]; top (+y) face at y = 2
        # one point above the top face with an opposing normal, one with a non-opposing normal, one far away
        self.scene = make_scene(
            points=[(1.0, 2.8, 1.5), (1.0, 2.8, 1.5), (1.0, -10.0, 1.5)],
            normals=[DOWN, UP, UP],
            radii=[1.5, 1.5, 1.5],  # lateral reach covers all six top samples (farthest at 1.118)
        )

    def test_sample_normals_outward_at_rest(self):
        b = make_batch(self.grid, noise=0.0)
        n = contact.sample_normals(b, self.grid.rest.to(DT))
        self.assertTrue(torch.allclose(n, hx.FACE_NORMALS[b.sample_face]))

    def test_csr_and_sorting(self):
        b = make_batch(self.grid, self.scene)
        p = b.pairs
        self.assertGreater(p.count, 0)
        counts = torch.bincount(p.cell, minlength=b.C)
        self.assertTrue(torch.equal(p.token_offsets[1:] - p.token_offsets[:-1], counts))
        self.assertEqual(int(p.token_offsets[0]), 0)
        self.assertTrue(torch.equal(p.cell, b.sample_cell[p.sample]))
        self.assertTrue((p.cell[1:] >= p.cell[:-1]).all())
        self.assertTrue((p.sample[1:] >= p.sample[:-1]).all())
        same = p.sample[1:] == p.sample[:-1]
        self.assertTrue((p.kind[1:][same] >= p.kind[:-1][same]).all())  # plane before points
        ids = point_ids(p, self.scene)
        both_pts = same & (p.kind[1:] == 1) & (p.kind[:-1] == 1)
        self.assertTrue((ids[1:][both_pts] > ids[:-1][both_pts]).all())  # points by id
        self.assertTrue((p.obj == 0).all())
        self.assertTrue(p.valid.all())
        self.assertTrue(torch.allclose(p.anchor, contact.sample_positions(b, b.X)[p.sample]))

    def test_plane_pairs_on_downward_faces(self):
        b = make_batch(self.grid, make_scene(plane_d=-0.3), noise=0.0, vel=0.0)
        p = b.pairs
        down = (b.sample_face == 2).nonzero().flatten()
        plane_samples = p.sample[p.kind == 0]
        self.assertEqual(plane_samples.numel(), down.numel())
        self.assertTrue(torch.equal(plane_samples.sort().values, down))
        self.assertTrue((p.kind == 0).all())
        self.assertTrue(torch.allclose(p.partner_point[:, 1], torch.full((p.count,), -0.3, dtype=DT)))
        b2 = make_batch(self.grid, make_scene(plane_d=-0.3, plane_present=False), noise=0.0, vel=0.0)
        self.assertEqual(b2.pairs.count, 0)

    def test_normal_opposition_and_far_rejection(self):
        b = make_batch(self.grid, self.scene, noise=0.0, vel=0.0)
        p = b.pairs
        pts = p.kind == 1
        self.assertEqual(int(pts.sum()), int((b.sample_face == 3).sum()))  # one pair per top sample
        self.assertTrue((point_ids(p, self.scene)[pts] == 0).all())
        self.assertTrue(torch.allclose(p.partner_normal[pts], torch.tensor([DOWN], dtype=DT)))
        self.assertTrue((p.radius[pts] == 1.5).all())

    def test_one_sided_disc(self):
        # disc below the body, normal pointing away from it: the top face opposes the normal but lies behind the disc
        b = make_batch(
            self.grid,
            make_scene(points=[(1.0, -0.3, 1.5)], normals=[DOWN], radii=[3.0], plane_present=False),
            noise=0.0,
        )
        self.assertEqual(b.pairs.count, 0)

    def test_nearest_m_pair_kept(self):
        heights = [2.6 + 0.1 * i for i in range(6)]
        scene = make_scene(
            points=[(1.0, y, 1.5) for y in heights], normals=[DOWN] * 6, radii=[2.0] * 6, plane_present=False
        )
        b = make_batch(self.grid, scene, noise=0.0, vel=0.0)
        p = b.pairs
        ids = point_ids(p, scene)
        for s in p.sample.unique():
            rows = p.sample == s
            self.assertEqual(int(rows.sum()), contact.M_PAIR)
            self.assertTrue(torch.equal(ids[rows], torch.arange(4)))

    def test_points_pair_within_their_object(self):
        g = self.grid
        b = Batch.build([g, g], "cpu", DT)
        set_material(b, [g, g])
        b.scene = cat_scenes([make_scene(), make_scene(points=[(1.0, 2.8, 1.5)], normals=[DOWN], radii=[1.0])])
        b.X = b.rest.to(DT)
        b.V = torch.zeros_like(b.X)
        p = contact.detect(b, b.X, b.V)
        self.assertEqual(p.token_offsets.shape[0], b.C + 1)
        self.assertTrue((p.obj[p.kind == 1] == 1).all())
        self.assertTrue((p.sample[p.kind == 1] >= b.sample_off[1]).all())
        self.assertEqual(int((p.kind == 0).sum()), 2 * int((g.samples.face == 2).sum()))
        self.assertTrue(torch.equal(p.obj, b.cell_obj[p.cell]))


class TestEnergy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.grid = Grid.build((2, 2, 3))
        self.scene = make_scene(points=[(1.0, 2.8, 1.5)], normals=[DOWN], radii=[1.0], plane_d=-0.3)

    def check_fd(self, b, x, eps=1e-6, tol=1e-6):
        """Autograd gradient against central differences of the energy with the friction load frozen at x."""
        x = x.detach().requires_grad_(True)
        E = contact.contact_energy(b, x)
        self.assertEqual(E.shape, (1,))
        self.assertGreater(E.item(), 0.0)
        (g,) = torch.autograd.grad(E.sum(), x)
        with torch.no_grad():
            _, _, gap, r_total = contact._geometry(b, x, b.pairs)
            load = b.material.ke[b.pairs.obj] * torch.relu(r_total - gap)
            frozen = lambda xx: sum(contact.pair_energies(b, xx, b.pairs, load)).sum()  # noqa: E731
            for _ in range(3):
                e = torch.randn_like(x)
                fd = (frozen(x + eps * e) - frozen(x - eps * e)) / (2 * eps)
                self.assertAlmostEqual(fd.item(), (g * e).sum().item(), delta=tol * max(1.0, abs(fd.item())))
        # the unfrozen difference is exactly the detached load's variation (Newton's constant-normal-force friction)
        e = torch.randn_like(x)
        full = (contact.contact_energy(b, x + eps * e).sum() - contact.contact_energy(b, x - eps * e).sum()) / (2 * eps)
        self.assertNotAlmostEqual(full.item(), (g * e).sum().item(), delta=1e-4 * max(1.0, abs(full.item())))

    def test_gradient_finite_difference(self):
        b = make_batch(self.grid, self.scene)
        _, E_d, E_f = contact.pair_energies(b, b.x, b.pairs)
        self.assertGreater(E_f.sum().item(), 0.0)  # slips outside the band
        self.check_fd(b, b.x)
        x_small = (
            b.X + torch.tensor([0.0, -0.05, 0.0], dtype=DT) + 5e-4 * torch.randn_like(b.X)
        )  # slips inside the band
        _, E_d, E_f = contact.pair_energies(b, x_small, b.pairs)
        self.assertGreater(E_f.sum().item(), 0.0)
        self.assertGreater(E_d.sum().item(), 0.0)
        self.check_fd(b, x_small, eps=1e-7, tol=1e-5)

    def test_normal_law_matches_newton(self):
        """Newton: penetration_depth = collision_radius - distance, force = ke penetration_depth n (zero when clear)."""
        g = Grid.build((1, 1, 1), pins="none")
        bottom = g.samples.corners[g.samples.face == 2][0]
        for gap in (0.1, 0.3, 0.45, 0.6):
            b = make_batch(g, make_scene(plane_d=-gap), noise=0.0, vel=0.0, mu_f=0.0, beta=0.0)
            x = b.X.detach().requires_grad_(True)
            E = contact.contact_energy(b, x)
            (gX,) = torch.autograd.grad(E.sum(), x)
            ke = b.material.ke[0].item()
            pen = max(contact.R_SAMPLE - gap, 0.0)
            self.assertAlmostEqual(E.item(), 0.5 * ke * pen**2, delta=1e-12 * max(1.0, ke))
            force = -gX * 4  # the sample is the mean of 4 corners
            expected = torch.tensor([[0.0, ke * pen, 0.0]], dtype=DT)
            self.assertTrue(torch.allclose(force[bottom], expected.expand(4, 3), atol=1e-9 * max(1.0, ke)))
            others = torch.ones(g.P, dtype=torch.bool)
            others[bottom] = False
            self.assertTrue((gX[others] == 0).all())

    def test_damping_and_friction_gated_on_penetration(self):
        b = make_batch(self.grid, make_scene(plane_d=-0.9), noise=0.0, vel=0.0)
        self.assertGreater(b.pairs.count, 0)  # detected within the band, not penetrating
        x = (b.X + torch.tensor([0.05, -0.1, 0.0], dtype=DT)).requires_grad_(True)  # approaching and slipping
        E = contact.contact_energy(b, x)
        self.assertEqual(E.item(), 0.0)
        (gX,) = torch.autograd.grad(E.sum(), x)
        self.assertTrue((gX == 0).all())
        x2 = b.X + torch.tensor([0.05, -0.5, 0.0], dtype=DT)  # gap 0.4 < r: all three terms on
        E_n, E_d, E_f = contact.pair_energies(b, x2, b.pairs)
        self.assertTrue((E_n > 0).all() and (E_d > 0).all() and (E_f > 0).all())
        b.pairs.anchor = b.pairs.anchor - torch.tensor([0.0, 0.6, 0.0], dtype=DT)  # now separating: damping off
        _, E_d, E_f = contact.pair_energies(b, x2, b.pairs)
        self.assertTrue((E_d == 0).all() and (E_f > 0).all())

    def test_friction_invariant_to_rigid_translation(self):
        b = make_batch(self.grid, make_scene(plane_d=-0.3), noise=0.0, vel=0.0, beta=0.0)
        x = b.X + torch.tensor([0.03, -0.2, 0.01], dtype=DT)
        t = torch.tensor([0.7, 0.0, -0.4], dtype=DT)  # tangential: gaps unchanged
        E1 = contact.contact_energy(b, x)
        E3 = contact.contact_energy(b, x + t)  # anchor fixed: different slip
        b.pairs.anchor = b.pairs.anchor + t
        E2 = contact.contact_energy(b, x + t)
        self.assertAlmostEqual(E1.item(), E2.item(), delta=1e-10 * abs(E1.item()))
        self.assertGreater(abs(E3.item() - E1.item()), 1e-3 * abs(E1.item()))

    def test_tokens_rotation_invariant(self):
        grid = Grid.build((2, 2, 3), pins="none")  # no exactly planar face: no partner normal exactly orthogonal
        b1 = make_batch(grid, self.scene)
        R1 = proper_rotation(b1.C)
        t1 = contact.contact_tokens(b1, b1.x, R1)
        self.assertEqual(t1.shape, (b1.pairs.count, contact.TOKEN_DIM))
        self.assertTrue(torch.equal(t1[:, 15:18].argmax(1), b1.pairs.kind))
        self.assertTrue((t1[:, 18] == 0).all())
        Q = proper_rotation()
        s = self.scene
        scene2 = ContactScene(
            plane_n=s.plane_n @ Q.t(),
            plane_d=s.plane_d,
            plane_present=s.plane_present,
            points=s.points @ Q.t(),
            normals=s.normals @ Q.t(),
            radii=s.radii,
            point_offsets=s.point_offsets,
        )
        b2 = Batch.build([grid], "cpu", DT)
        set_material(b2, [grid])
        b2.scene = scene2
        b2.X, b2.V = b1.X @ Q.t(), b1.V @ Q.t()
        b2.pairs = contact.detect(b2, b2.X, b2.V)
        self.assertEqual(b2.pairs.count, b1.pairs.count)
        self.assertTrue(torch.equal(b2.pairs.sample, b1.pairs.sample))
        self.assertTrue(torch.equal(b2.pairs.kind, b1.pairs.kind))
        t2 = contact.contact_tokens(b2, b1.x @ Q.t(), Q @ R1)
        self.assertLess((t1 - t2).abs().max().item(), 1e-9)

    def test_penetration_metric(self):
        b = make_batch(self.grid, make_scene(plane_d=-0.9), noise=0.0, vel=0.0)
        self.assertGreater(b.pairs.count, 0)
        self.assertEqual(contact.penetration(b, b.X).item(), 0.0)
        b = make_batch(self.grid, make_scene(plane_d=-0.3), noise=0.0, vel=0.0)
        self.assertAlmostEqual(contact.penetration(b, b.X).item(), 0.4, places=12)
        b = make_batch(
            self.grid, self.scene, noise=0.0, vel=0.0
        )  # point: gap 0.8 > r = 0.5 -> detected, not penetrating
        self.assertGreater(b.pairs.count, 0)
        self.assertAlmostEqual(
            contact.penetration(b, b.X).item(), 0.4, places=12
        )  # only the fixture's plane penetrates

    def test_no_pairs(self):
        b = make_batch(self.grid)
        self.assertEqual(b.pairs.count, 0)
        self.assertTrue(torch.equal(contact.contact_energy(b, b.x), torch.zeros(1, dtype=DT)))
        self.assertEqual(contact.contact_tokens(b, b.x, torch.eye(3, dtype=DT).expand(b.C, 3, 3)).shape, (0, 19))
        self.assertEqual(contact.penetration(b, b.x).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
