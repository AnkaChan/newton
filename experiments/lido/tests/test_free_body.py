# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unpinned bodies (derivation note section 7): translation-free shape solve with the centroid-only blend (7.17-7.21),
the rigid semi-implicit centroid target recomputed per query (7.11, Picard) and the Newton step on the centroid (7.15,
`translation="implicit_contact"`);
free fall, soft and stiff contact, equivariance, the pinned path bitwise unchanged, capture, rollout and SolverLIDO."""

import itertools
import math
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
from experiments.lido import contact, physics
from experiments.lido import hex as hx
from experiments.lido.augment import Augmenter
from experiments.lido.batch import Batch
from experiments.lido.config import TrainConfig
from experiments.lido.fusion import Fusion, sparse_solver_available
from experiments.lido.grid import Grid
from experiments.lido.network import Net
from experiments.lido.newton_solver import SolverLIDO, add_hex_body, attach_bodies
from experiments.lido.rollout import rollout
from experiments.lido.step import Step, translation_residual
from experiments.lido.structs import ContactScene, Material
from experiments.lido.tests import test_capture as capture_tests  # module import: no duplicate collection
from experiments.lido.tests.test_fusion import full_B
from experiments.lido.tests.test_step import make as make_pinned
from experiments.lido.tests.test_voxel_grid import holed_box
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
SI = {"E": 1e5, "nu": 0.3, "rho": 1000.0, "eta": 50.0, "gravity": (0.0, -9.81, 0.0), "h": 0.05, "dt": 1.0 / 300.0}
REFERENCE = Path(__file__).parent / "reference" / "pinned_beam_reference.pt"


def plane_scene(plane_d: float | None, dtype, device) -> ContactScene:
    """The y-plane n = (0, 1, 0) at offset plane_d (signed distance of x is y - plane_d); absent when None."""
    z = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
    return ContactScene(
        plane_n=torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device),
        plane_d=torch.tensor([0.0 if plane_d is None else plane_d], dtype=dtype, device=device),
        plane_present=torch.tensor([plane_d is not None], device=device),
        points=z(0, 3),
        normals=z(0, 3),
        radii=z(0),
        point_offsets=torch.tensor([0, 0], dtype=torch.int64, device=device),
    )


def free_batch(kappa: float, plane_d: float | None, dtype=torch.float64, device="cpu", grid=None, y0=0.0):
    """One unpinned 2x2x3 box at rest (shifted by y0 along y), normal-penalty contact only (beta = mu_f = 0)."""
    g = grid or Grid.build((2, 2, 3), pins="none", device=device)
    b = Batch.build([g], device, dtype)
    m = material_from_si(cell_count=g.C, sample_count=g.S, kappa=kappa, beta=0.0, mu_f=0.0, device=device, **SI)
    b.material = Material.cat([m])
    for f in MATERIAL_FIELDS:
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    b.scene = plane_scene(plane_d, dtype, device)
    b.X = g.rest.to(dtype).clone()
    b.X[:, 1] += y0
    b.V = torch.zeros_like(b.X)
    b.X_prev = b.X.clone()
    b.x = b.X.clone()
    return g, b


def zero_step(translation="picard", dtype=torch.float64) -> Step:
    """A fresh network: zero-init heads, so the proposal is zero and the fused update is the translation alone."""
    return Step(Net.from_config(SMALL_CFG).to(dtype).eval(), Fusion(), translation=translation)


def rest_shape_error(X: torch.Tensor, rest: torch.Tensor) -> float:
    return ((X - X.mean(0)) - (rest.to(X.dtype) - rest.to(X.dtype).mean(0))).abs().max().item()


def static_penetration(b: Batch, n_bottom: int) -> float:
    """Normalised penetration at which n_bottom active pairs balance gravity: M_tot |g| / (n_bottom ke)."""
    M_tot = physics.total_mass(b)[0].item()
    return M_tot * b.material.g[0].norm().item() / (n_bottom * b.material.ke[0].item())


class TestFreeFall(unittest.TestCase):
    """(a) Contact-free free body: the centroid follows free fall and the shape stays the rest shape."""

    def test_centroid_free_fall_and_rigid_shape(self):
        for translation in ("picard", "implicit_contact"):
            g, b = free_batch(0.0, None)
            step = zero_step(translation)
            sel = torch.ones(1, dtype=torch.bool)
            step.prepare(b, sel)
            self.assertEqual(b.pairs.count, 0)
            c0, cd0, grav = physics.centroid(b, b.X)[0], physics.centroid(b, b.V)[0], b.material.g[0]
            for n in range(1, 21):
                for _ in range(8):
                    out = step.query(b)
                    step.commit(b, out)
                step.advance(b, sel)
                expected = c0 + n * cd0 + n * (n + 1) / 2 * grav
                self.assertLess((physics.centroid(b, b.X)[0] - expected).abs().max().item(), 1e-10, translation)
                self.assertLess(rest_shape_error(b.X, g.rest), 1e-12, translation)
            self.assertTrue(torch.equal(b.picard_constant, torch.zeros(1, dtype=torch.float64)))

    def test_perturbed_candidate_keeps_the_inertial_centroid(self):
        """prepare's noisy candidate has c(x_0) = c(Y) = c_n + cdot_n + g (7.6), and the inertia term has no masked rows."""
        _g, b = free_batch(0.0, None)
        b.V = 0.01 * torch.randn(b.N, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        b.X_prev = b.X - b.V
        step = Step(Net.from_config(SMALL_CFG).double(), Fusion(), Augmenter("cpu"), noise_prob=1.0)
        gens = [torch.Generator().manual_seed(9)]
        step.prepare(b, torch.ones(1, dtype=torch.bool), gens)
        self.assertGreater((b.x - b.Y).abs().max().item(), 1e-3)  # the candidate is perturbed ...
        c_Y = physics.centroid(b, b.X)[0] + physics.centroid(b, b.V)[0] + b.material.g[0]
        self.assertLess((physics.centroid(b, b.Y)[0] - c_Y).abs().max().item(), 1e-12)
        self.assertLess((physics.centroid(b, b.x)[0] - c_Y).abs().max().item(), 1e-12)  # ... at the inertial centroid
        self.assertTrue((physics.inertia(b, b.x) > 0).all())


class TestSoftContact(unittest.TestCase):
    """(b) Soft floor contact, Picard constant below 1: the translation residual (7.14) contracts by the Picard
    factor (7.13) per query and the dropped body comes to rest at the static penetration of the penalty law."""

    KAPPA = 1.0  # sum_bottom ke / M_tot = 6 x 2.6 / 70.2 = 0.222

    def test_residual_contracts_by_the_picard_factor(self):
        g, b = free_batch(self.KAPPA, None)
        pen = static_penetration(b, 6)
        g, b = free_batch(self.KAPPA, -contact.R_SAMPLE + pen)  # resting at the static penetration, V = 0
        step = zero_step("picard")
        sel = torch.ones(1, dtype=torch.bool)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            step.prepare(b, sel)
        L = b.picard_constant[0].item()
        self.assertAlmostEqual(L, 6 * b.material.ke[0].item() / physics.total_mass(b)[0].item(), places=12)
        self.assertLess(L, 1.0)
        res = [translation_residual(b, b.x)[0].norm().item()]
        for _ in range(8):
            out = step.query(b)
            step.commit(b, out)
            self.assertAlmostEqual(out.picard_constant[0].item(), L, places=12)
            res.append(translation_residual(b, b.x)[0].norm().item())
        self.assertGreater(res[0], 0.1)
        for r0, r1 in itertools.pairwise(res):
            self.assertLessEqual(r1, L * r0 * (1 + 1e-6) + 1e-12)
        self.assertLess(res[-1], 1e-5 * res[0])
        # the candidate stays rigid (zero proposal): only the translation moves
        self.assertLess(rest_shape_error(b.x, g.rest), 1e-12)

    def test_dropped_body_rests_at_the_static_penetration(self):
        for translation in ("picard", "implicit_contact"):
            g, b = free_batch(self.KAPPA, -1.5)  # bottom faces one cell above the plane
            pen_static = static_penetration(b, 6)
            step = zero_step(translation)
            sel = torch.ones(1, dtype=torch.bool)
            step.prepare(b, sel)
            ys = []
            for _ in range(200):
                for _ in range(8):
                    step.commit(b, step.query(b))
                step.advance(b, sel)
                ys.append(physics.centroid(b, b.X)[0, 1].item())
            pen = contact.penetration(b, b.X)[0].item() * contact.R_SAMPLE
            self.assertLess(abs(pen - pen_static), 1e-3 * pen_static, translation)
            self.assertLess(max(abs(a - c) for a, c in zip(ys[-20:-1], ys[-19:], strict=True)), 1e-5, translation)
            self.assertLess(min(ys), ys[-1])  # it bounced below the rest position and came back up
            self.assertGreater(ys[-1], -pen_static - 1e-6)  # and does not sink further
            self.assertLess(rest_shape_error(b.X, g.rest), 1e-10, translation)


class TestStiffContact(unittest.TestCase):
    """(c) Stiff floor contact, Picard constant above 1: Newton converges in finitely many queries (Proposition
    7.2(i)); Picard is detected (warning) and does not contract.

    Observed with the zero proposal (rigid body): Picard is an exact 2-cycle between the penetrating inertial
    position and a lifted position with no contact, so r_tr stays constant and the state returned after an even
    number of queries is the free-fall position: the body falls through the floor (y = -42 cells after 200 steps at
    kappa = 20). Newton lands on the root in one query once the active set is the solution's."""

    KAPPA = 20.0  # sum_bottom ke / M_tot = 4.44

    def resting(self, translation):
        _g, b = free_batch(self.KAPPA, None)
        pen = static_penetration(b, 6)
        _g, b = free_batch(self.KAPPA, -contact.R_SAMPLE + pen)
        step = zero_step(translation)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step.prepare(b, torch.ones(1, dtype=torch.bool))
        return b, step, caught

    def residuals(self, b, step, K=6):
        res = [translation_residual(b, b.x)[0].norm().item()]
        for _ in range(K):
            step.commit(b, step.query(b))
            res.append(translation_residual(b, b.x)[0].norm().item())
        return res

    def test_picard_is_detected_and_does_not_contract(self):
        b, step, caught = self.resting("picard")
        self.assertGreater(b.picard_constant[0].item(), 1.0)
        self.assertEqual(len([w for w in caught if "Picard constant" in str(w.message)]), 1)
        res = self.residuals(b, step)
        self.assertGreater(min(r1 / r0 for r0, r1 in itertools.pairwise(res)), 0.99)
        # one warning per object: a second prepare of the same object is silent
        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter("always")
            step.prepare(b, torch.ones(1, dtype=torch.bool))
        self.assertEqual([w for w in again if "Picard constant" in str(w.message)], [])

    def test_newton_converges_in_a_few_queries(self):
        b, step, _caught = self.resting("implicit_contact")
        res = self.residuals(b, step)
        self.assertGreater(res[0], 0.1)
        self.assertLess(res[1], 1e-10 * res[0])  # affine region with the solution's active set: one step is exact
        self.assertLess(max(res[1:]), 1e-10 * res[0])

    def test_tilted_body_terminates_finitely(self):
        """A body tilted by 5 degrees about z has two bottom-sample heights (two breakpoints of r_tr along the
        normal). Started so that both groups penetrate at the inertial candidate while the root has only the lower
        group active, the Newton iterate moves up monotonically (Newton-Fourier) and lands exactly on the root once
        the active set is the root's: two queries here (Proposition 7.2(i))."""
        _g, b = free_batch(self.KAPPA, None)
        a = math.radians(5.0)
        R = torch.tensor(
            [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        c = b.X.mean(0)
        b.X = c + (b.X - c) @ R.t()
        xs = contact.sample_positions(b, b.X)
        bottom = b.sample_face == 2  # -y faces
        heights = xs[bottom, 1]
        self.assertGreater(heights.max().item() - heights.min().item(), 0.08)
        b.scene.plane_d[:] = heights.min().item() - contact.R_SAMPLE + 0.01  # the lower group penetrates by 0.01
        b.scene.plane_present[:] = True
        b.V[:, 1] = -0.09
        b.X_prev, b.x = b.X - b.V, b.X.clone()
        step = zero_step("implicit_contact")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            step.prepare(b, torch.ones(1, dtype=torch.bool))
        ke = b.material.ke[0].item()
        active = lambda: round(contact.active_stiffness(b, b.x)[0][0].item() / ke)  # noqa: E731
        self.assertEqual(active(), 6)
        res = [translation_residual(b, b.x)[0].norm().item()]
        counts = [active()]
        for _ in range(4):
            step.commit(b, step.query(b))
            res.append(translation_residual(b, b.x)[0].norm().item())
            counts.append(active())
        self.assertGreater(res[1], 1e-3 * res[0])  # the first step, with six active pairs, stops short of the root
        self.assertEqual(counts[1], 3)
        self.assertLess(res[2], 1e-12 * res[0])  # the second lands on it
        self.assertEqual(counts[2:], [3, 3, 3])
        self.assertLess(max(res[2:]), 1e-12 * res[0])

    def test_newton_drop_rests_at_the_static_penetration(self):
        _g, b = free_batch(self.KAPPA, -1.5)
        pen_static = static_penetration(b, 6)
        step = zero_step("implicit_contact")
        sel = torch.ones(1, dtype=torch.bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            step.prepare(b, sel)
        ys = []
        for _ in range(200):
            for _ in range(8):
                step.commit(b, step.query(b))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                step.advance(b, sel)
            ys.append(physics.centroid(b, b.X)[0, 1].item())
        pen = contact.penetration(b, b.X)[0].item() * contact.R_SAMPLE
        self.assertLess(abs(pen - pen_static), 1e-6 * pen_static)
        self.assertLess(abs(ys[-1] - ys[-50]), 1e-9)


def rotate_free_batch(b: Batch, q: torch.Tensor, t: torch.Tensor) -> None:
    for name in ("X", "V", "X_prev", "x"):
        v = getattr(b, name)
        setattr(b, name, v @ q.t() + (t if name != "V" else 0))
    b.material.g = b.material.g @ q.t()
    n = b.scene.plane_n @ q.t()
    b.scene.plane_d = b.scene.plane_d + n @ t
    b.scene.plane_n = n


class TestEquivariance(unittest.TestCase):
    """(d) Rotating and translating the scene (positions, velocities, gravity, plane) rotates the fused update,
    translation included, for both centroid targets."""

    def make(self, translation, dtype=torch.float64):
        torch.manual_seed(3)
        g, b = free_batch(5.0, -contact.R_SAMPLE + 0.05)  # bottom faces 0.05 cells into the floor
        b.X = b.X + 0.05 * torch.randn(b.N, 3, dtype=dtype)
        b.V = 0.02 * torch.randn(b.N, 3, dtype=dtype)
        b.X_prev = b.X - b.V
        b.x = b.X.clone()
        net = Net.from_config(SMALL_CFG).to(dtype)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        return g, b, Step(net, Fusion(), translation=translation)

    def test_rotated_scene(self):
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)))
        q = q * torch.sign(torch.det(q))
        t = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
        for translation in ("picard", "implicit_contact"):
            _g, b, step = self.make(translation)
            _g2, b2, step2 = self.make(translation)
            step2.net.load_state_dict(step.net.state_dict())
            rotate_free_batch(b2, q, t)
            sel = torch.ones(1, dtype=torch.bool)
            step.prepare(b, sel)
            step2.prepare(b2, sel)
            self.assertGreater(int(b.pairs.valid.sum()), 0)
            self.assertEqual(b.pairs.count, b2.pairs.count)
            self.assertLess((b2.E - b.E).abs().max().item(), 1e-9)
            self.assertLess((b2.c_n - (b.c_n @ q.t() + t)).abs().max().item(), 1e-12)
            self.assertLess((b2.cdot_n - b.cdot_n @ q.t()).abs().max().item(), 1e-12)
            out, out2 = step.query(b), step2.query(b2)
            d1 = (out.cand_after - b.x) @ q.t()
            d2 = out2.cand_after - b2.x
            self.assertGreater(d1.abs().max().item(), 1e-4)
            self.assertLess((d1 - d2).abs().max().item(), 1e-5 * max(1.0, d1.abs().max().item()), translation)
            self.assertLess((out.picard_constant - out2.picard_constant).abs().max().item(), 1e-12)
            c1 = physics.centroid(b, out.cand_after.detach())[0] @ q.t() + t
            c2 = physics.centroid(b2, out2.cand_after.detach())[0]
            self.assertLess((c1 - c2).abs().max().item(), 1e-9)


class TestPinnedPathUnchanged(unittest.TestCase):
    """(e) The pinned beam's query results are bitwise those recorded before the free-body changes (CPU, one
    thread, fixed seeds: tests.test_step.make and tests.test_capture.make_batch with a plane and static points).
    The reference was re-recorded after the aggregated edge values of the network (spec section 10, 2026-10-01),
    which reorder the network's float sums: the new recording differs from the first at rounding level only."""

    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def assert_record(self, rec: dict, out, tag: str):
        for name, value in (
            ("cand_after", out.cand_after.detach()),
            ("E_after", out.E_after.detach()),
            ("gX_after", out.gX_after),
        ):
            self.assertTrue(
                torch.equal(rec[name], value), f"{tag}: {name} differs by {(rec[name] - value).abs().max()}"
            )

    def test_bitwise_reference(self):
        ref = torch.load(REFERENCE)
        for dtype, tag in ((torch.float64, "f64"), (torch.float32, "f32")):
            _g, b, step, gens = make_pinned(O=2, seed=0, randomize=True, dtype=dtype)
            self.assertFalse(b.any_free)
            rec = ref[f"nocontact_{tag}"]
            for k in range(2):
                out = step.query(b)
                self.assert_record(rec[k], out, f"nocontact_{tag}[{k}]")
                self.assertIsNone(out.picard_constant)
                step.commit(b, out)
            step.advance(b, torch.ones(2, dtype=torch.bool), gens)
            self.assert_record(rec[2], step.query(b), f"nocontact_{tag}[2]")
        gen = torch.Generator().manual_seed(7)
        # the reference was recorded with the a02 edge module (TrainConfig's default since changed to "pair")
        net = capture_tests.small_net(gen, "cpu", TrainConfig(**capture_tests.SMALL, edge_module="a02")).double()
        grids = [Grid.build((2, 2, 3)), Grid.build((1, 2, 2))]
        scenes = [capture_tests.random_scene(gen, gr, 6, torch.float64, "cpu") for gr in grids]
        b = capture_tests.make_batch(grids, scenes, torch.Generator().manual_seed(107), torch.float64, "cpu")
        step = Step(net, Fusion())
        step.prepare(b, torch.ones(b.O, dtype=torch.bool))
        self.assertGreater(b.pairs.count, 0)
        for k in range(2):
            out = step.query(b)
            self.assert_record(ref["contact_f64"][k], out, f"contact_f64[{k}]")
            step.commit(b, out)


class TestCentroidBlendSolution(unittest.TestCase):
    """(f) `fuse` with a centroid target reproduces the centroid-only blend (7.19) solved densely for every lambda:
    the same d, c(x_k + d) = c_t exactly (Proposition 7.3(b)); the solvers agree with each other."""

    def dense_reference(self, g: Grid, b: Batch, dF: torch.Tensor, c_t: torch.Tensor, lam: float) -> torch.Tensor:
        B = full_B(g)
        W = torch.diag(hx.WEIGHTS.repeat_interleave(9).repeat(g.C))
        K = B.t() @ W @ B
        m = physics.corner_mass(b)  # [P]
        M_tot = m.sum()
        Z = torch.kron(torch.ones(g.P, 1, dtype=torch.float64), torch.eye(3, dtype=torch.float64))  # [3P,3]
        V = m.repeat_interleave(3)[:, None] * Z  # M Z
        A = K + lam / M_tot * V @ V.t()
        self.assertGreater(torch.linalg.eigvalsh(A).min().item(), 0.0)
        c_x = physics.centroid(b, b.x)[0]
        rhs = B.t() @ W @ dF.reshape(-1) + lam * V @ (c_t[0] - c_x)
        return torch.linalg.solve(A, rhs).reshape(g.P, 3)

    def check(self, device, solvers):
        g, b = free_batch(1.0, None, device=device, grid=Grid.build((2, 2, 2), pins="none", device=device))
        gen = torch.Generator().manual_seed(11)
        b.x = g.rest.to(torch.float64) + 0.1 * torch.randn(g.P, 3, dtype=torch.float64, generator=gen).to(device)
        dm = torch.randn(g.C, 7, 3, dtype=torch.float64, generator=gen).to(device)
        dF = hx.modes_to_gauss(dm, b.hc)
        c_t = (physics.centroid(b, b.x) + torch.tensor([[0.4, -0.3, 0.2]], dtype=torch.float64, device=device)).detach()
        g_cpu, b_cpu = free_batch(1.0, None, grid=Grid.build((2, 2, 2), pins="none"))
        b_cpu.x = b.x.cpu()
        refs = [self.dense_reference(g_cpu, b_cpu, dF.cpu(), c_t.cpu(), lam) for lam in (1e-3, 1.0, 1e3)]
        for r in refs[1:]:
            self.assertLess((r - refs[0]).abs().max().item(), 1e-9)  # lambda drops out (7.20-7.21)
        d_ref = refs[0].to(device)
        gX = torch.randn(g.P, 3, dtype=torch.float64, generator=gen).to(device)
        for solver, tol in solvers:
            fusion = Fusion(solver, options={"capture": False} if solver == "mg" else None)
            d = fusion.fuse(b, dF, centroid_target=c_t)
            self.assertLess((d - d_ref).abs().max().item(), tol, solver)
            c = physics.centroid(b, b.x + d)
            self.assertLess((c - c_t).abs().max().item(), 1e-13, solver)
            # without a target the shape solution is translation free up to the particular solution
            d0 = fusion.fuse(b, dF)
            diff = d0 - d
            self.assertLess((diff - diff.mean(0)).abs().max().item(), tol, solver)
            # and project_gradient agrees between the solvers (translation component of gX removed)
            p = fusion.project_gradient(b, gX)
            if solver == solvers[0][0]:
                p_first = p
            else:
                self.assertLess((p - p_first).abs().max().item(), 1e-9, solver)

    def test_cpu_kron_dense_multigrid(self):
        self.check("cpu", [("kron", 1e-10), ("dense", 1e-10), ("mg", 1e-9)])

    @unittest.skipUnless(HAS_CUDA, "cuda")
    def test_cuda_sparse_multigrid(self):
        solvers = [("kron", 1e-10), ("dense", 1e-10), ("mg", 1e-9)]
        if sparse_solver_available():
            solvers.append(("sparse", 1e-10))
        self.check(CUDA, solvers)

    def test_kron_free_box_pseudo_inverse(self):
        """K^+ on a mean-zero right-hand side solves K d = r; the constant mode's coefficient is dropped."""
        g = Grid.build((2, 3, 2), pins="none")
        b = Batch.build([g], "cpu", torch.float64)
        fac = Fusion("kron").factor(g, torch.float64)
        self.assertTrue(torch.isinf(fac.denom[0, 0, 0]))
        r = torch.randn(2, g.P, 3, dtype=torch.float64)
        r = r - r.mean(1, keepdim=True)
        d = fac.solve(r)
        self.assertLess((fac.apply_full(d) - r).abs().max().item(), 1e-10)
        self.assertLess(d.abs().max().item(), 1e3)
        self.assertEqual(fac.free.numel(), g.P)
        self.assertEqual(b.groups[0].grid.pins, "none")


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestCapturedQueryFreeVoxelBody(capture_tests.TestCapturedQuery):
    """(g) The captured query with an unpinned voxel body (multigrid fusion with one fixed reference corner, the
    centroid target recorded into the graph) replays the eager query, Picard constant included.

    kappa 5: the Picard constant of this scene (plane plus six static discs) is 2.1 at the initial candidate and 0.4
    after a few queries, once the body has lifted out of the discs. Tolerance 5e-5 instead of the pinned 1e-5: the
    contact force that drives the translation comes from the torch pair energies in the eager compacted path and from
    the Warp kernel in the captured one, and a free body feeds that float32 difference back into the next query
    through its position (with kappa 50 the non-contracting Picard target amplifies it to 5e-4 by the fourth query;
    the implicit-contact target stays below 1e-5 there, see the subclass)."""

    grids = staticmethod(lambda: [Grid.from_voxels(holed_box(), "none", CUDA)])
    fusions = staticmethod(lambda: (Fusion("mg", options={"capture": False}), Fusion("mg")))
    KAPPA = 5.0
    TRANSLATION = "picard"

    def states(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        net = capture_tests.small_net(gen, CUDA)
        grids = self.grids()
        scenes = [capture_tests.random_scene(gen, g, 6, torch.float32, CUDA) for g in grids]
        pair = []
        for capacity, fusion in zip((False, True), self.fusions(), strict=True):
            gen_b = torch.Generator().manual_seed(seed + 100)
            b = capture_tests.make_batch(grids, scenes, gen_b, torch.float32, CUDA)
            capture_tests.set_material(b, grids, torch.float32, CUDA, kappa=self.KAPPA)
            step = Step(net, fusion, pair_capacity=capacity, translation=self.TRANSLATION)
            step.prepare(b, torch.ones(b.O, dtype=torch.bool, device=CUDA))
            pair.append((b, step))
        return pair

    def assert_same(self, b1, b2, tag, tol=5e-5):
        super().assert_same(b1, b2, tag, tol)
        self.assertTrue(b1.any_free and b2.any_free)
        err = (b1.picard_constant - b2.picard_constant).abs().max().item()
        self.assertLess(err, tol * max(1.0, b1.picard_constant.abs().max().item()))

    def test_three_queries_and_an_advance(self):
        super().test_three_queries_and_an_advance()
        _, (b2, s2) = self.states()
        self.assertEqual(s2.fusion.factor(b2.grids[0]).free.numel(), b2.grids[0].P - 1)
        self.assertGreater(b2.picard_constant[0].item(), 0.0)  # written by prepare at the initial candidate


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestCapturedQueryFreeVoxelBodyImplicitContact(TestCapturedQueryFreeVoxelBody):
    """The same with stiff contact (kappa 50, Picard constant 16 at the initial candidate) and the implicit-contact
    target."""

    KAPPA = 50.0
    TRANSLATION = "implicit_contact"

    def test_three_queries_and_an_advance(self):
        super().test_three_queries_and_an_advance()
        _, (b2, _) = self.states()
        self.assertGreater(b2.picard_constant[0].item(), 1.0)


class TestRolloutAndSolverFreeBody(unittest.TestCase):
    """(h) rollout and SolverLIDO with `pins="none"`: all particle masses positive, 3 steps run, the centroid
    follows the free-fall formula without a plane."""

    def test_rollout_free_fall(self):
        scenario = dict(cell_counts=(2, 2, 3), pins="none", **SI)
        device = "cuda:0" if HAS_CUDA else "cpu"
        for translation in ("picard", "implicit_contact"):
            r = rollout(None, SMALL_CFG.to_dict(), scenario, 3, 2, device=device, translation=translation)
            self.assertEqual(r["fixed_indices"].shape, (0,))
            dt, grav = SI["dt"], np.array(SI["gravity"])
            pos = r["positions"].astype(np.float64)
            c0 = pos[0].mean(0)
            for n in range(1, 4):
                expected = pos[n - 1] + dt * r["velocities"][n - 1] + dt**2 * grav
                self.assertLess(np.abs(pos[n] - expected).max(), 1e-6, translation)
                self.assertLess(np.abs(pos[n].mean(0) - (c0 + n * (n + 1) / 2 * dt**2 * grav)).max(), 1e-6)
            self.assertTrue((r["penetration_r"] == 0).all())

    def test_solver_free_bodies(self):
        wp.init()
        device = "cuda:0" if HAS_CUDA else "cpu"
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        material = {"E": 1e5, "nu": 0.3, "rho": 1000.0, "eta": 50.0}
        for i, cc in enumerate(((2, 2, 3), (1, 2, 2))):
            add_hex_body(builder, cc, SI["h"], pos=(0.5 * i, 0.3, 0.0), material=material, pins="none")
        model = builder.finalize(device=device)
        attach_bodies(model, builder)
        self.assertTrue((model.particle_mass.numpy() > 0).all())
        self.assertTrue((model.particle_inv_mass.numpy() > 0).all())
        solver = SolverLIDO(model, None, iterations=2, cfg=SMALL_CFG, translation="implicit_contact")
        self.assertTrue(solver.batch.any_free)
        self.assertFalse(solver.batch.pinned.any())
        self.assertFalse(solver.batch.scene.plane_present.any())
        s0, s1 = model.state(), model.state()
        q0 = s0.particle_q.numpy().copy().astype(np.float64)
        grav = model.gravity.numpy()[-1].astype(np.float64)
        dt = SI["dt"]
        for n in range(1, 4):
            q_before, qd_before = s0.particle_q.numpy().copy(), s0.particle_qd.numpy().copy()
            solver.step(s0, s1, None, None, dt)
            q, qd = s1.particle_q.numpy().copy(), s1.particle_qd.numpy().copy()
            self.assertTrue(np.isfinite(q).all() and np.isfinite(qd).all())
            self.assertLess(np.abs(q - (q_before + dt * qd_before + dt**2 * grav)).max(), 1e-6)
            for body in solver.bodies:
                sl = slice(body.particle_start, body.particle_start + body.particle_count)
                c = q[sl].mean(0)
                self.assertLess(np.abs(c - (q0[sl].mean(0) + n * (n + 1) / 2 * dt**2 * grav)).max(), 1e-6)
            s0, s1 = s1, s0
        with self.assertRaises(ValueError):
            Step(solver.net, Fusion(), translation="blend")


if __name__ == "__main__":
    unittest.main()


class TestTranslationTrustRegion(unittest.TestCase):
    """The centroid target's trust region (2026-10-02): a blow-up of the contact force moves a body by at most
    `translation_step_max` cells per query, 0 removes the bound, and a physical step is left untouched."""

    def test_bound_binds_only_on_huge_forces(self):
        import experiments.lido.step as step_module

        with self.assertRaises(ValueError):
            Step(Net.from_config(SMALL_CFG).eval(), Fusion(), translation_step_max=-1.0)
        results = {}
        for bound in (1.0, 0.25, 0.0):
            _g, b = free_batch(100.0, -0.3)  # resting on the plane, 0.2 cells into the sample band
            step = Step(Net.from_config(SMALL_CFG).to(torch.float64).eval(), Fusion(), translation_step_max=bound)
            sel = torch.ones(1, dtype=torch.bool)
            step.prepare(b, sel)
            self.assertGreater(b.pairs.count, 0)
            c_x = physics.centroid(b, b.x)
            c_t, _ = step.centroid_target(b, b.x)
            results[bound] = (c_t - c_x).norm(dim=-1).item()
            # a contact force blow-up: 1e4 x the body's weight along +x, as the squeezed body of the fourth v5 attempt
            real = step_module.contact.contact_force
            M = physics.total_mass(b)
            fake = lambda batch, x: real(batch, x) + 1e4 * M[:, None] * torch.tensor([[1.0, 0.0, 0.0]], dtype=x.dtype)  # noqa: E731
            step_module.contact.contact_force = fake
            try:
                c_huge, _ = step.centroid_target(b, b.x)
            finally:
                step_module.contact.contact_force = real
            d = (c_huge - c_x).norm(dim=-1).item()
            if bound > 0.0:
                self.assertAlmostEqual(d, bound, places=9, msg=f"bound {bound}")
                self.assertGreater((c_huge - c_x)[0, 0].item(), 0.0)  # the direction of the step is kept
            else:
                self.assertGreater(d, 1e3)
        # the physical step is far below the bounds and identical with and without them
        self.assertLess(results[1.0], 0.25)
        self.assertAlmostEqual(results[1.0], results[0.0], places=12)
        self.assertAlmostEqual(results[0.25], results[0.0], places=12)
