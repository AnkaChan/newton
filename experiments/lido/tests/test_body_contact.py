# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-body contact (design spec section 11): mesh detection between free boxes in one world frame (facing samples,
partner faces and normals, no self pairs, both directions), the closest point on a quad, the energy gradient with
respect to both bodies' corners (finite differences, Newton's third law), a body resting on another body, the Warp
pair kernel and the fused pass against the torch path with partner-corner gradients, capacity against compaction,
and the captured query with body pairs."""

import unittest

import torch

from experiments.lido import contact, physics
from experiments.lido import hex as hx
from experiments.lido.batch import Batch
from experiments.lido.capture import CapturedQuery
from experiments.lido.config import TrainConfig
from experiments.lido.frames import frames
from experiments.lido.fusion import Fusion
from experiments.lido.grid import Grid, reference_rotation
from experiments.lido.network import Net
from experiments.lido.step import Step
from experiments.lido.structs import ContactScene, Material
from experiments.lido.tests.test_capture import FullFloat32, small_net
from experiments.lido.tests.test_contact_kernel import rel_err, torch_paths
from experiments.lido.units import material_from_si

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
SMALL_CFG = TrainConfig(hidden_dim=24, edge_hidden_dim=12, num_heads=2, contact_hidden_dim=8)
MATERIAL_FIELDS = (
    "lam",
    "rho",
    "eta",
    "g",
    "ke",
    "kd",
    "mu_f",
    "kappa",
    "beta",
    "friction_eps",
    "floor",
    "h",
    "dt",
    "mu",
)
SI = {"E": 1e5, "nu": 0.3, "rho": 1000.0, "eta": 50.0, "gravity": (0.0, -9.81, 0.0), "h": 0.025, "dt": 1.0 / 300.0}
R = contact.R_SAMPLE


def world_scene(O: int, plane_d, dtype, device) -> ContactScene:
    """One world frame for all O objects: the y-plane at offset plane_d for every object (absent when None)."""
    z = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
    return ContactScene(
        plane_n=torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device).expand(O, 3).clone(),
        plane_d=torch.full((O,), 0.0 if plane_d is None else float(plane_d), dtype=dtype, device=device),
        plane_present=torch.full((O,), plane_d is not None, dtype=torch.bool, device=device),
        points=z(0, 3),
        normals=z(0, 3),
        radii=z(0),
        point_offsets=torch.zeros(O + 1, dtype=torch.int64, device=device),
    )


def two_boxes(
    dtype=torch.float64,
    device="cpu",
    sides=((3, 3, 3), (3, 3, 3)),
    gap=0.4,
    shift=(0.0, 0.0),
    plane_d=None,
    kappa=10.0,
    beta=0.3,
    mu_f=0.4,
    noise=0.0,
    seed=0,
) -> Batch:
    """Two unpinned boxes in one world frame: A with its rest corner at the origin, B above A with its bottom faces
    `gap` cells over A's top face (gap < r: touching in the sample-sphere sense), shifted by `shift` in (x, z); the
    step-start shape is the rest shape plus `noise`; V = 0, X_prev = x = X."""
    device = torch.device(device)
    gen = torch.Generator().manual_seed(seed)
    grids = [Grid.build(s, pins="none", device=device) for s in sides]
    b = Batch.build(grids, device, dtype)
    parts = [
        material_from_si(cell_count=g.C, sample_count=g.S, kappa=kappa, beta=beta, mu_f=mu_f, device=device, **SI)
        for g in grids
    ]
    b.material = Material.cat(parts)
    for f in MATERIAL_FIELDS:
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    b.scene = world_scene(2, plane_d, dtype, device)
    b.body_contact = True
    X = b.rest.to(dtype).clone()
    rows = b.corner_obj == 1
    X[rows] += torch.tensor([shift[0], sides[0][1] + gap, shift[1]], dtype=dtype, device=device)
    X += noise * torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
    b.X, b.V = X, torch.zeros_like(X)
    b.X_prev, b.x, b.Y = X.clone(), X.clone(), X.clone()
    F_prev = hx.gauss_deformation(X[b.cells], b.hc)
    b.C_prev = hx.mat3_tn(F_prev, F_prev)
    b.m_Y, b.m_prev = hx.modes(X[b.cells], b.hc), hx.modes(X[b.cells], b.hc)
    b.R_ref = reference_rotation(X, b.ref_corners)
    return b


def body_rows(pairs, obj: int):
    return (pairs.kind == contact.KIND_BODY) & (pairs.obj == obj) & pairs.valid


class TestDetection(unittest.TestCase):
    """Two touching 3x3x3 boxes without a plane: every bottom sample of B pairs with the top face of A under it and
    every top sample of A with the bottom face of B above it; nothing else pairs."""

    def test_facing_samples_both_directions(self):
        for shift in ((0.0, 0.0), (0.3, -0.2)):
            b = two_boxes(shift=shift)
            p = contact.detect(b, b.X, b.V)
            self.assertGreater(p.count, 0)
            self.assertTrue((p.kind == contact.KIND_BODY).all())  # no plane, no points
            self.assertTrue(p.valid.all())
            self.assertFalse((p.partner_body == p.obj).any())  # no self pairs
            self.assertTrue(torch.equal(p.partner_body, b.sample_obj[p.partner_face]))
            self.assertTrue(torch.equal(p.obj, b.sample_obj[p.sample]))
            # B's bottom faces (face 2) against A's top faces (face 3) and the reverse: 9 + 9 pairs
            self.assertEqual(p.count, 18)
            rows_b, rows_a = body_rows(p, 1), body_rows(p, 0)
            self.assertEqual(int(rows_b.sum()), 9)
            self.assertEqual(int(rows_a.sum()), 9)
            self.assertTrue((b.sample_face[p.sample[rows_b]] == 2).all())
            self.assertTrue((b.sample_face[p.partner_face[rows_b]] == 3).all())
            self.assertTrue((b.sample_face[p.sample[rows_a]] == 3).all())
            self.assertTrue((b.sample_face[p.partner_face[rows_a]] == 2).all())
            # partner normals are the partner faces' normals at X; partner points are the closest points on them
            ns = contact.sample_normals(b, b.X)
            self.assertTrue(torch.allclose(p.partner_normal, ns[p.partner_face]))
            self.assertTrue(torch.allclose(p.partner_normal[rows_b], torch.tensor([[0.0, 1.0, 0.0]]).double()))
            xs = contact.sample_positions(b, b.X)
            self.assertTrue(torch.allclose(p.anchor, xs[p.sample]))
            gap = ((xs[p.sample] - p.partner_point) * p.partner_normal).sum(-1)
            self.assertTrue(torch.allclose(gap, torch.full((p.count,), 0.4, dtype=torch.float64), atol=1e-6))
            self.assertTrue((p.radius == R).all())
            # the recorded face is the one under the sample: its quad contains the closest point
            c = b.X[b.sample_corners[p.partner_face]]
            pt, w = contact.closest_point_on_quad(xs[p.sample], c[:, 0], c[:, 1], c[:, 2], c[:, 3])
            self.assertLess((pt - p.partner_point).abs().max().item(), 1e-6)
            self.assertTrue((w >= -1e-12).all() and torch.allclose(w.sum(-1), torch.ones(p.count).double()))
            # symmetric: the set of (sample face, partner face) of A's rows mirrors B's rows
            fwd = set(zip(p.sample[rows_b].tolist(), p.partner_face[rows_b].tolist(), strict=True))
            rev = set(zip(p.partner_face[rows_a].tolist(), p.sample[rows_a].tolist(), strict=True))
            if shift == (0.0, 0.0):
                self.assertEqual(fwd, rev)
            # sorted by owning cell, sample
            self.assertTrue((p.cell[1:] >= p.cell[:-1]).all() and (p.sample[1:] >= p.sample[:-1]).all())
            counts = torch.bincount(p.cell, minlength=b.C)
            self.assertTrue(torch.equal(p.token_offsets[1:] - p.token_offsets[:-1], counts))

    def test_separated_and_penetrating(self):
        b = two_boxes(gap=1.2)  # surface distance 1.2 - r = 0.7 > margin r: nothing
        self.assertEqual(contact.detect(b, b.X, b.V).count, 0)
        b.V[:, 1] = 0.5  # both bodies moving together: no relative speed, no approach, nothing
        self.assertEqual(contact.detect(b, b.X, b.V).count, 0)
        b.V[b.corner_obj == 0] = 0.0
        b.V[b.corner_obj == 1, 1] = -0.5  # B coming down at A at rest: the relative speed 0.5 reaches it ...
        p = contact.detect(b, b.X, b.V)
        self.assertEqual(p.count, 18)  # ... from both sides: A's samples see B approaching although A does not move
        self.assertEqual(int(body_rows(p, 0).sum()), 9)
        self.assertEqual(int(body_rows(p, 1).sum()), 9)
        b.V[b.corner_obj == 1, 1] = 0.5  # B moving away from A: |v_s - v_f| = 0.5 as well (the margin is a speed)
        self.assertEqual(contact.detect(b, b.X, b.V).count, 18)
        b.V[b.corner_obj == 1] = 0.0
        b.V[b.corner_obj == 0, 1] = 0.5  # A coming up at B at rest: the same pairs
        self.assertEqual(contact.detect(b, b.X, b.V).count, 18)
        b.V[:] = 0.0
        b.V[b.corner_obj == 0, 0] = 0.7  # A sliding sideways under B: relative speed 0.7, pairs; B's own speed 0
        self.assertEqual(contact.detect(b, b.X, b.V).count, 18)
        b = two_boxes(gap=-0.3)  # B's bottom samples inside A: the sign parity keeps them, gap negative
        b.pairs = p = contact.detect(b, b.X, b.V)
        self.assertEqual(p.count, 18)
        _, _, _, gap, _, _ = contact._geometry(b, b.X, p)
        self.assertTrue(torch.allclose(gap, torch.full((18,), -0.3, dtype=torch.float64), atol=1e-6))
        self.assertTrue(torch.allclose(contact.penetration(b, b.X), torch.full((2,), 0.8 / R, dtype=torch.float64)))

    def test_body_contact_off_and_single_body(self):
        b = two_boxes()
        b.body_contact = False
        self.assertEqual(contact.detect(b, b.X, b.V).count, 0)
        b.body_contact = True
        cap = contact.detect(b, b.X, b.V, capacity=True)
        self.assertEqual(cap.count, b.S * 2)  # 1 plane slot + min(M_PAIR, O - 1) = 1
        g = Grid.build((2, 2, 2), pins="none")
        single = Batch.build([g], "cpu", torch.float64)
        single.scene = world_scene(1, None, torch.float64, "cpu")
        single.body_contact = True
        single.X = g.rest.double()
        single.V = torch.zeros_like(single.X)
        self.assertEqual(contact.detect(single, single.X, single.V).count, 0)

    def test_tokens_use_the_body_slot(self):
        b = two_boxes(noise=0.02)
        b.pairs = contact.detect(b, b.X, b.V)
        Rf = torch.eye(3, dtype=torch.float64).expand(b.C, 3, 3)
        t = contact.contact_tokens(b, b.X, Rf)
        self.assertEqual(t.shape, (b.pairs.count, contact.TOKEN_DIM))
        self.assertTrue((t[:, 15:18].argmax(1) == contact.KIND_BODY).all())
        self.assertTrue((t[:, 11] == 1.0).all())  # r_p / r
        self.assertTrue(torch.isfinite(t).all())

    def test_meshes_rebuilt_after_relayout(self):
        b = two_boxes()
        contact.detect(b, b.X, b.V)
        m = b.meshes
        self.assertIsNotNone(m)
        contact.detect(b, b.X + 0.1, b.V)
        self.assertIs(b.meshes, m)  # refreshed in place
        b.relayout([Grid.build((3, 3, 3), pins="none"), Grid.build((2, 2, 2), pins="none")])
        self.assertIsNone(b.meshes)


class TestClosestPoint(unittest.TestCase):
    def test_against_brute_force(self):
        gen = torch.Generator().manual_seed(5)
        n = 64
        c = torch.randn(n, 4, 3, dtype=torch.float64, generator=gen)
        c[:, 2] = c[:, 0] + (c[:, 1] - c[:, 0]) + (c[:, 3] - c[:, 0]) + 0.3 * torch.randn(n, 3, generator=gen).double()
        p = 2.0 * torch.randn(n, 3, dtype=torch.float64, generator=gen)
        pt, w = contact.closest_point_on_quad(p, c[:, 0], c[:, 1], c[:, 2], c[:, 3])
        self.assertTrue((w >= -1e-12).all())
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(n, dtype=torch.float64)))
        self.assertTrue(torch.allclose(pt, (w[..., None] * c).sum(1)))
        self.assertTrue(((w > 0).sum(-1) <= 3).all())  # one triangle
        m = 200
        u = torch.linspace(0, 1, m, dtype=torch.float64)
        uu, vv = torch.meshgrid(u, u, indexing="ij")
        keep = uu + vv <= 1.0
        bary = torch.stack([1 - uu[keep] - vv[keep], uu[keep], vv[keep]], -1)  # [G,3]
        best = torch.full((n,), float("inf"), dtype=torch.float64)
        for tri in ((0, 1, 2), (0, 2, 3)):
            pts = torch.einsum("gk,nkd->ngd", bary, c[:, list(tri)])  # [n,G,3]
            best = torch.minimum(best, (pts - p[:, None]).norm(dim=-1).min(1).values)
        d = (pt - p).norm(dim=-1)
        self.assertTrue((d <= best + 1e-12).all())
        self.assertLess((best - d).max().item(), 0.05)  # the grid's resolution
        # points projecting into the interior of a planar quad: the projection, weights sum to one on one triangle
        q = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
        pp = torch.tensor([[0.7, 0.4, 1.4], [1.6, -0.2, 0.3]], dtype=torch.float64)
        pt, _ = contact.closest_point_on_quad(pp, *(q[i].expand(2, 3) for i in range(4)))
        self.assertTrue(torch.allclose(pt, pp * torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)))

    def test_matches_the_warp_function(self):
        """The kernel's weights (float32, through the pair kernel's geometry) agree with the torch ones: checked
        indirectly by TestKernel; here the tie rule: a point over the shared diagonal gives the same weights."""
        q = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        pp = torch.tensor([[0.5, 0.3, 0.5]], dtype=torch.float64)
        _, w = contact.closest_point_on_quad(pp, *(q[i][None] for i in range(4)))
        self.assertTrue(torch.allclose(w, torch.tensor([[0.5, 0.0, 0.5, 0.0]], dtype=torch.float64)))


def affine_world(b: Batch, gen: torch.Generator) -> None:
    """One affine map (rotation, stretch, shear) of the whole world: faces stay planar, contacts stay contacts."""
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=gen))
    q = q * torch.sign(torch.linalg.det(q))
    A = q @ (torch.eye(3, dtype=torch.float64) + 0.1 * torch.randn(3, 3, dtype=torch.float64, generator=gen))
    X = b.X @ A.t()
    b.X, b.X_prev, b.x, b.Y = X, X.clone(), X.clone(), X.clone()


class TestGradient(unittest.TestCase):
    """The torch gradient against finite differences of the energy (friction load frozen) with respect to the
    corners of both bodies, and Newton's third law."""

    def setUp(self):
        self.gen = torch.Generator().manual_seed(11)

    def pushed(self, b: Batch) -> torch.Tensor:
        """Candidate: B pressed into A along the contact normal and slid sideways, A nudged; rigid per body, so the
        material-point term of the slip is exact and the faces stay planar."""
        n = b.pairs.partner_normal[body_rows(b.pairs, 1)].mean(0)  # A's top normal (world axes)
        n = n / n.norm()
        t1 = torch.linalg.cross(n, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        t1 = t1 / t1.norm()
        t2 = torch.linalg.cross(n, t1)
        x = b.X.clone()
        x[b.corner_obj == 1] += 0.05 * t1 - 0.12 * n + 0.03 * t2
        x[b.corner_obj == 0] += -0.02 * t1 + 0.01 * n + 0.015 * t2
        return x

    def test_finite_differences_both_bodies(self):
        for shift, affine in (((0.0, 0.0), False), ((0.4, -0.3), True), ((0.2, 0.1), False)):
            b = two_boxes(shift=shift, beta=0.5, mu_f=0.6)
            if affine:
                affine_world(b, self.gen)
            b.pairs = contact.detect(b, b.X, b.V)
            self.assertGreaterEqual(
                b.pairs.count, 18
            )  # the affine map tilts B's side faces towards A's top: grazing pairs
            x = self.pushed(b).requires_grad_(True)
            E = contact.contact_energy(b, x)
            self.assertTrue((E > 0).all())
            (g,) = torch.autograd.grad(E.sum(), x)
            self.assertGreater(g[b.corner_obj == 0].abs().max().item(), 0.0)
            self.assertGreater(g[b.corner_obj == 1].abs().max().item(), 0.0)
            with torch.no_grad():
                _, _, _, gap, r_total, _ = contact._geometry(b, x, b.pairs)
                load = b.material.ke[b.pairs.obj] * torch.relu(r_total - gap)
                _, E_d, E_f = contact.pair_energies(b, x, b.pairs, load)
                self.assertTrue((E_f > 0).sum() > 0 and (E_d > 0).sum() > 0)  # grazing side pairs carry no load
                frozen = lambda xx, b=b, load=load: sum(contact.pair_energies(b, xx, b.pairs, load)).sum()  # noqa: E731
                eps = 1e-6
                for obj in (0, 1):
                    for _ in range(3):
                        e = torch.randn(b.N, 3, dtype=torch.float64, generator=self.gen)
                        e[b.corner_obj != obj] = 0  # one body's corners at a time
                        fd = (frozen(x + eps * e) - frozen(x - eps * e)) / (2 * eps)
                        self.assertAlmostEqual(
                            fd.item(), (g * e).sum().item(), delta=1e-7 * max(1.0, abs(fd.item())), msg=f"{shift} {obj}"
                        )

    def test_third_law(self):
        """No plane: the contact gradient sums to zero over all corners of both bodies (translation invariance of
        gap, normal and relative slip), for warped faces and all three energy terms."""
        for seed in range(3):
            b = two_boxes(shift=(0.3, 0.2), noise=0.05, beta=0.5, mu_f=0.6, seed=seed)
            b.pairs = contact.detect(b, b.X, b.V)
            self.assertGreater(b.pairs.count, 0)
            x = b.X + 0.03 * torch.randn(b.N, 3, dtype=torch.float64, generator=self.gen)
            x[b.corner_obj == 1] += torch.tensor([0.02, -0.15, -0.01], dtype=torch.float64)
            x.requires_grad_(True)
            E_n, E_d, E_f = contact.pair_energies(b, x, b.pairs)
            self.assertTrue((E_n > 0).sum() > 0 and (E_d > 0).sum() > 0 and (E_f > 0).sum() > 0)
            (g,) = torch.autograd.grad(contact.contact_energy(b, x).sum(), x)
            scale = g.abs().sum(0).max().item()
            self.assertGreater(scale, 0.0)
            self.assertLess(g.sum(0).abs().max().item(), 1e-10 * scale)
            F = contact.contact_force(b, x.detach())
            self.assertLess((F[0] + F[1]).abs().max().item(), 1e-10 * scale)
            # the partner's translation stiffness counts the pairs it is the partner of
            ke_sum, H = contact.active_stiffness(b, x.detach())
            active = (contact._geometry(b, x.detach(), b.pairs)[4] - contact._geometry(b, x.detach(), b.pairs)[3]) > 0
            ke = b.material.ke[0].item()
            n_a = int((active & (b.pairs.obj == 0)).sum()) + int((active & (b.pairs.partner_body == 0)).sum())
            self.assertAlmostEqual(ke_sum[0].item(), n_a * ke, places=9)
            self.assertTrue(torch.allclose(H, H.transpose(-1, -2)))

    def test_co_moving_bodies_have_no_slip(self):
        """Both bodies translated by the same vector since step start: no damping, no friction between them."""
        b = two_boxes(gap=0.2, beta=0.5, mu_f=0.6)
        b.pairs = contact.detect(b, b.X, b.V)
        t = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
        E_n, E_d, E_f = contact.pair_energies(b, b.X + t, b.pairs)
        self.assertTrue((E_n > 0).all())
        self.assertTrue((E_d == 0).all())
        eps = b.material.friction_eps[0].item()
        self.assertTrue(torch.allclose(E_f, b.material.mu_f[0] * b.material.ke[0] * (R - 0.2) * eps / 3.0))  # f0(0)
        # only B moving: slip and approach on both directions' rows
        x = b.X.clone()
        x[b.corner_obj == 1] += t
        E_n, E_d, E_f = contact.pair_energies(b, x, b.pairs)
        self.assertTrue((E_d > 0).all() and (E_f > 0).all())


class TestResting(unittest.TestCase):
    """A 2x2x2 box set onto a 3x3x3 box standing on the plane, zero-init network, implicit-contact centroid update
    (the coupled Newton step on both centroids with damping): both bodies come to rest at the static penetrations of
    the penalty law; nothing falls through. B's footprint overhangs A's top samples at x = 2.5 and z = 0.5 by 0.1:
    those pair through the edge tie-break and hold B as well (13 pairs: 4 of B's bottom, 9 of A's top)."""

    def test_rest_on_top_face(self):
        b = two_boxes(sides=((3, 3, 3), (2, 2, 2)), gap=R, shift=(0.4, 0.6), plane_d=-R, kappa=20.0, beta=0.3, mu_f=0.0)
        net = Net.from_config(SMALL_CFG).double().eval()
        step = Step(net, Fusion(), translation="implicit_contact")
        sel = torch.ones(2, dtype=torch.bool)
        step.prepare(b, sel)
        self.assertGreater(int((b.pairs.kind == contact.KIND_BODY).sum()), 0)
        top_a = b.sample_face == 3
        bottom_b = (b.sample_face == 2) & (b.sample_obj == 1)
        ys = []
        for _ in range(100):
            for _ in range(8):
                step.commit(b, step.query(b))
            step.advance(b, sel)
            xs = contact.sample_positions(b, b.X)
            ys.append((xs[top_a & (b.sample_obj == 0), 1].mean().item(), xs[bottom_b, 1].mean().item()))
            self.assertTrue(torch.isfinite(b.X).all())
        # B stays on A: its bottom samples never drop below A's top samples by more than r
        self.assertTrue(all(yb > ya - R for ya, yb in ys))
        self.assertLess(max(abs(ys[-1][k] - ys[-10][k]) for k in range(2)), 1e-6)  # static
        # static penetrations: B held by its pairs with A (both directions) at equal depth, A by the plane
        M = physics.total_mass(b)
        g_mag = b.material.g[0].norm().item()
        ke = b.material.ke[0].item()
        p = b.pairs
        _, _, _, gap, r_total, _ = contact._geometry(b, b.X, p)
        active = (r_total - gap > 0) & p.valid
        hold = int((active & (p.kind == contact.KIND_BODY)).sum())  # pairs between A and B
        self.assertGreater(hold, 0)
        pen_ab = (r_total - gap)[active & (p.kind == contact.KIND_BODY)]
        self.assertLess((pen_ab - pen_ab.mean()).abs().max().item(), 1e-6 * pen_ab.mean().item())
        self.assertAlmostEqual(
            pen_ab.mean().item(), M[1].item() * g_mag / (hold * ke), delta=1e-4 * pen_ab.mean().item()
        )
        pen_plane = (r_total - gap)[active & (p.kind == 0)]
        self.assertAlmostEqual(
            pen_plane.mean().item(), (M[0] + M[1]).item() * g_mag / (9 * ke), delta=1e-4 * pen_plane.mean().item()
        )
        self.assertLess(contact.penetration(b, b.X)[1].item() * R - pen_ab.mean().item(), 1e-9)


def kernel_batch(seed: int, variant: str, device=CUDA) -> Batch:
    """Two warped boxes in contact on CUDA float32 with the plane under A; the candidate selects the regime."""
    gen = torch.Generator().manual_seed(seed)
    b = two_boxes(torch.float32, device, shift=(0.3, -0.2), plane_d=-0.3, noise=0.05, beta=0.5, mu_f=0.6, seed=seed)
    b.V = 0.1 * torch.randn(b.N, 3, generator=gen).to(device)
    noise = torch.randn(b.N, 3, generator=gen).to(device)
    push = torch.tensor([0.0, -1.0, 0.0], device=device)
    x = b.X.clone()
    rows = b.corner_obj == 1
    if variant == "slip":
        x = x + 0.02 * noise
        x[rows] += 0.3 * torch.tensor([1.0, 0.0, 0.5], device=device) + 0.2 * push
    elif variant == "band":
        x = x + 3e-4 * noise
        x[rows] += 0.15 * push
    elif variant == "approach":
        x = x + 0.01 * noise
        x[rows] += 0.3 * push
    b.x = x
    b.pairs = contact.detect(b, b.X, b.V, capacity=True)
    return b


@unittest.skipUnless(HAS_CUDA, "needs cuda")
class TestKernel(unittest.TestCase):
    """The Warp pair kernel (through the autograd Function and the fused pass) against the torch path with body
    pairs: energies, sample and partner-corner gradients."""

    def grads(self, b):
        x = b.x.clone().requires_grad_(True)
        E_w = contact.contact_energy(b, x)
        (g_w,) = torch.autograd.grad(E_w.sum(), x)
        with torch_paths(contact_only=True):
            E_t = contact.contact_energy(b, x)
            (g_t,) = torch.autograd.grad(E_t.sum(), x)
        return E_w, g_w, E_t, g_t

    def test_regimes_match_torch(self):
        for seed in range(2):
            for variant in ("slip", "band", "approach", "rest"):
                b = kernel_batch(seed, variant)
                tag = f"seed {seed} {variant}"
                self.assertGreater(int(body_rows(b.pairs, 0).sum()) + int(body_rows(b.pairs, 1).sum()), 0, tag)
                E_w, g_w, E_t, g_t = self.grads(b)
                self.assertGreater(E_t.abs().max().item(), 0.0, tag)
                self.assertLess(rel_err(E_w, E_t), 1e-5, f"{tag}: energy")
                self.assertLess(rel_err(g_w, g_t), 1e-4, f"{tag}: gradient")
                # fused inference pass
                self.assertTrue(physics.fused_pass_applies(b, b.x))
                E, gX = physics.energy_and_grad(b, b.x)
                with torch_paths():
                    E_ref, g_ref = physics.energy_and_grad(b, b.x)
                self.assertLess(rel_err(E, E_ref), 1e-5, f"{tag}: fused energy")
                self.assertLess(rel_err(gX, g_ref), 1e-4, f"{tag}: fused gradient")

    def test_partner_corner_gradients(self):
        """Only B's rows active: A's corners receive the partner gradient alone, equal and opposite in total."""
        b = kernel_batch(3, "slip")
        b.pairs.valid &= b.pairs.obj == 1
        self.assertGreater(int(body_rows(b.pairs, 1).sum()), 0)
        E_w, g_w, E_t, g_t = self.grads(b)
        a = b.corner_obj == 0
        self.assertGreater(g_t[a].abs().max().item(), 0.0)
        self.assertLess(rel_err(g_w[a], g_t[a]), 1e-4)
        self.assertLess(rel_err(g_w[~a], g_t[~a]), 1e-4)
        self.assertLess(rel_err(E_w, E_t), 1e-5)
        self.assertEqual(E_w[0].item(), 0.0)  # the energy belongs to the owner
        # the plane pairs of B are off (valid rows are all body pairs): third law in float32
        self.assertTrue((b.pairs.kind[b.pairs.valid] == contact.KIND_BODY).all())
        self.assertLess(g_w.sum(0).abs().max().item(), 1e-4 * g_w.abs().sum(0).max().item())

    def test_geometry_kernel_matches_torch(self):
        """The capacity rows' geometry (tokens, translation stiffness, penetration) through `pair_geometry_kernel`
        against the torch chain, padded rows zero."""
        for seed, variant in ((0, "slip"), (1, "approach"), (2, "rest")):
            b = kernel_batch(seed, variant)
            Rf = frames(hx.center_deformation(hx.modes(b.x[b.cells], b.hc)), b.R_ref[b.cell_obj])
            tag = f"seed {seed} {variant}"
            geo_w = contact._geometry(b, b.x, b.pairs)
            tok_w = contact.contact_tokens(b, b.x, Rf)
            ke_w, J_w = contact.translation_hessian(b, b.x)
            pen_w = contact.penetration(b, b.x)
            with torch_paths(contact_only=True):
                geo_t = contact._geometry(b, b.x, b.pairs)
                tok_t = contact.contact_tokens(b, b.x, Rf)
                ke_t, J_t = contact.translation_hessian(b, b.x)
                pen_t = contact.penetration(b, b.x)
            valid = b.pairs.valid
            self.assertGreater(int(valid.sum()), 0)
            self.assertLess(int(valid.sum()), b.pairs.count)
            for name, gw, gt in zip(("xs", "p", "n", "gap", "r", "delta"), geo_w, geo_t, strict=True):
                self.assertLess(rel_err(gw[valid], gt[valid]), 1e-5, f"{tag}: {name}")
                if name != "r":
                    self.assertEqual(gw[~valid].abs().max().item(), 0.0, f"{tag}: {name} padded")
            self.assertLess(rel_err(tok_w[valid], tok_t[valid]), 1e-5, f"{tag}: tokens")
            self.assertTrue(torch.isfinite(tok_w).all())
            self.assertLess(rel_err(ke_w, ke_t), 1e-5, f"{tag}: ke")
            self.assertLess(rel_err(J_w, J_t), 1e-4, f"{tag}: J")
            self.assertLess(rel_err(pen_w, pen_t), 1e-5, f"{tag}: penetration")

    def test_backward_scales_with_object_weights(self):
        b = kernel_batch(1, "approach")
        w = torch.tensor([2.0, -0.5], device=CUDA)
        grads = []
        for use_torch in (False, True):
            with torch_paths(contact_only=use_torch):
                x = b.x.clone().requires_grad_(True)
                (contact.contact_energy(b, x) * w).sum().backward()
                grads.append(x.grad)
        self.assertLess(rel_err(grads[0], grads[1]), 1e-4)


class TestCapacity(unittest.TestCase):
    """detect(capacity=True) carries the compacted body pairs in the same order; energy, gradient, tokens and
    penetration are identical between the layouts."""

    def test_same_pairs_same_values(self):
        for seed in range(3):
            b = two_boxes(shift=(0.3, -0.2), plane_d=-0.3, noise=0.05, seed=seed)
            b.V = 0.1 * torch.randn(b.N, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
            compact = contact.detect(b, b.X, b.V)
            cap = contact.detect(b, b.X, b.V, capacity=True)
            self.assertTrue(cap.padded and not compact.padded)
            self.assertEqual(cap.count, b.S * 2)
            self.assertEqual(int(cap.valid.sum()), compact.count)
            self.assertGreater(int((compact.kind == contact.KIND_BODY).sum()), 0)
            self.assertGreater(int((compact.kind == 0).sum()), 0)
            for name in ("sample", "cell", "obj", "partner_point", "partner_normal", "kind", "radius", "anchor"):
                self.assertTrue(torch.equal(getattr(cap, name)[cap.valid], getattr(compact, name)), name)
            for name in ("partner_body", "partner_face"):
                self.assertTrue(torch.equal(getattr(cap, name)[cap.valid], getattr(compact, name)), name)
            x = (b.X + 0.02 * torch.randn(b.N, 3, dtype=torch.float64)).requires_grad_(True)
            x.data[b.corner_obj == 1, 1] -= 0.1
            Rf = frames(hx.center_deformation(hx.modes(x.detach()[b.cells], b.hc)).float(), b.R_ref[b.cell_obj].float())
            out = []
            for pairs in (compact, cap):
                b.pairs = pairs
                E = contact.contact_energy(b, x)
                (g,) = torch.autograd.grad(E.sum(), x)
                out.append(
                    (E, g, contact.penetration(b, x.detach()), contact.contact_tokens(b, x.detach(), Rf.double()))
                )
            (E1, g1, pen1, tok1), (E2, g2, pen2, tok2) = out
            self.assertGreater(E1.sum().item(), 0.0)
            self.assertLess((E1 - E2).abs().max().item(), 1e-12 * E1.abs().max().item())
            self.assertLess((g1 - g2).abs().max().item(), 1e-12 * g1.abs().max().item())
            self.assertTrue(torch.equal(pen1, pen2))
            self.assertTrue(torch.equal(tok2[cap.valid], tok1))


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestCapturedQueryBodies(FullFloat32):
    """The captured query (capacity pairs, Warp contact kernel) replays the eager compacted query (torch contact
    path) on two free boxes in contact over the plane, implicit-contact centroid update. Tolerance 5e-5 as the
    free-body capture tests: positions and energies agree to 1e-5, the gradients to 3e-5 at kappa 5 (the contact
    gradient comes from different float32 paths, and the stiff two-body coupling feeds the difference back through
    both bodies' positions; kappa 20 reaches 3e-4)."""

    def states(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        net = small_net(gen, CUDA, SMALL_CFG)
        pair = []
        for capacity in (False, True):
            b = two_boxes(
                torch.float32, CUDA, shift=(0.3, -0.2), gap=0.3, plane_d=-0.3, noise=0.03, kappa=5.0, seed=seed
            )
            gen_b = torch.Generator().manual_seed(seed + 100)
            b.V = 0.1 * torch.randn(b.N, 3, generator=gen_b).to(CUDA)
            b.X_prev = b.X - b.V
            step = Step(net, Fusion(), pair_capacity=capacity, translation="implicit_contact")
            step.prepare(b, torch.ones(2, dtype=torch.bool, device=CUDA))
            pair.append((b, step))
        return pair

    def assert_same(self, b1, b2, tag, tol=5e-5):
        for name in ("x", "E", "gX", "hist_grad", "hist_update", "picard_constant"):
            a, c = getattr(b1, name), getattr(b2, name)
            err = (a - c).abs().max().item()
            self.assertLess(err, tol * max(1.0, c.abs().max().item()), f"{tag}: {name} differs by {err:.3e}")

    def test_three_queries_and_an_advance(self):
        (b1, s1), (b2, s2) = self.states()
        self.assertGreater(int(body_rows(b2.pairs, 0).sum()) + int(body_rows(b2.pairs, 1).sum()), 0)
        self.assertGreater(int((b2.pairs.kind[b2.pairs.valid] == 0).sum()), 0)
        sel = torch.ones(2, dtype=torch.bool, device=CUDA)
        with torch.no_grad():
            cq = CapturedQuery(s2, b2)
            torch.cuda.synchronize()
            self.assert_same(b1, b2, "before")
            for k in range(3):
                s1.commit(b1, s1.query(b1))
                cq.replay()
                torch.cuda.synchronize()
                self.assert_same(b1, b2, f"query {k}")
            s1.advance(b1, sel)
            s2.advance(b2, sel)
            self.assertTrue(b2.pairs is not cq.pairs and b2.X is not cq.state["X"])
            cq.sync()
            self.assertTrue(b2.pairs is cq.pairs and b2.X is cq.state["X"])
            self.assertGreater(int(body_rows(b2.pairs, 0).sum()) + int(body_rows(b2.pairs, 1).sum()), 0)
            self.assert_same(b1, b2, "after advance")
            s1.commit(b1, s1.query(b1))
            cq.replay()
            torch.cuda.synchronize()
            self.assert_same(b1, b2, "query after advance")


if __name__ == "__main__":
    unittest.main()
