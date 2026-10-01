# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capacity-layout pairs against the compacted detection, the padded contact encoder, CUDA-graph replay of the
query against eager queries, rollout and SolverLIDO with capture."""

import unittest
import unittest.mock

import numpy as np
import torch

from experiments.lido import contact
from experiments.lido import hex as hx
from experiments.lido.batch import Batch
from experiments.lido.capture import CapturedQuery
from experiments.lido.config import TrainConfig
from experiments.lido.features import features
from experiments.lido.frames import frames
from experiments.lido.fusion import Fusion
from experiments.lido.grid import Grid, reference_rotation
from experiments.lido.network import ContactEncoder, Net
from experiments.lido.newton_solver import SolverLIDO
from experiments.lido.rollout import compile_inference, rollout
from experiments.lido.scenes import cat_scenes
from experiments.lido.step import Step
from experiments.lido.structs import ContactScene, Material
from experiments.lido.tests import test_newton_solver as solver_tests  # module import: no duplicate collection
from experiments.lido.tests.test_voxel_grid import holed_box
from experiments.lido.units import material_from_si

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
SMALL = {"hidden_dim": 24, "edge_hidden_dim": 12, "num_heads": 2, "contact_hidden_dim": 8}
SMALL_CFG = TrainConfig(**SMALL)
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


def random_scene(gen: torch.Generator, grid: Grid, n_points: int, dtype, device) -> ContactScene:
    """A plane just under the rest bottom and `n_points` discs around the body with roughly opposing normals."""
    nx, ny, nz = grid.cell_counts
    centre = torch.tensor([nx / 2, ny / 2, nz / 2], dtype=dtype)
    pos = torch.rand(n_points, 3, generator=gen, dtype=dtype) * torch.tensor([nx + 2, ny + 2, nz + 2], dtype=dtype)
    pos = pos - 1.0
    toward = centre - pos
    toward = toward / toward.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    normals = toward + 0.3 * torch.randn(n_points, 3, generator=gen, dtype=dtype)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    radii = 0.5 + 2.0 * torch.rand(n_points, generator=gen, dtype=dtype)
    plane_d = -0.2 - 0.6 * torch.rand((), generator=gen, dtype=dtype)
    return ContactScene(
        plane_n=torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device),
        plane_d=plane_d.reshape(1).to(device),
        plane_present=torch.tensor([True], device=device),
        points=pos.to(device),
        normals=normals.to(device),
        radii=radii.to(device),
        point_offsets=torch.tensor([0, n_points], device=device),
    )


def set_material(b: Batch, grids: list, dtype, device, **mat) -> None:
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
    parts = [material_from_si(cell_count=g.C, sample_count=g.S, device=device, **m) for g in grids]
    b.material = Material.cat(parts)
    for f in MATERIAL_FIELDS:
        setattr(b.material, f, getattr(b.material, f).to(dtype))


def make_batch(grids: list, scenes: list, gen: torch.Generator, dtype=torch.float64, device="cpu") -> Batch:
    """Random noisy rest state with velocities; Y, C_prev and the candidate as prepare would set them."""
    b = Batch.build(grids, device, dtype)
    set_material(b, grids, dtype, device)
    b.scene = cat_scenes(scenes)
    rest = b.rest.to(dtype)
    b.X = rest + 0.05 * torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
    b.X[b.pinned] = rest[b.pinned]
    b.V = 0.3 * torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
    b.V[b.pinned] = 0
    b.X_prev = b.X - b.V
    b.Y = b.X + b.V + b.material.g[b.corner_obj]
    b.Y[b.pinned] = b.X[b.pinned]
    b.m_Y = hx.modes(b.Y[b.cells], b.hc)
    b.m_prev = hx.modes(b.X[b.cells], b.hc)
    F_prev = hx.gauss_deformation(b.X[b.cells], b.hc)
    b.C_prev = hx.mat3_tn(F_prev, F_prev)
    b.R_ref = reference_rotation(b.X, b.ref_corners)
    b.x = b.Y + 0.02 * torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
    b.x[b.pinned] = b.X[b.pinned]
    return b


class TestCapacityLayout(unittest.TestCase):
    """detect(capacity=True) carries the compacted pairs in the same order, padded with invalid rows."""

    def pairs_both(self, seed, grids=None, n_points=6):
        gen = torch.Generator().manual_seed(seed)
        grids = grids or [Grid.build((2, 2, 3)), Grid.build((1, 2, 2))]
        scenes = [random_scene(gen, g, n_points, torch.float64, "cpu") for g in grids]
        b = make_batch(grids, scenes, gen)
        compact = contact.detect(b, b.X, b.V)
        cap = contact.detect(b, b.X, b.V, capacity=True)
        return b, compact, cap

    def test_same_pairs_same_order(self):
        for seed in range(4):
            b, compact, cap = self.pairs_both(seed)
            k = min(contact.M_PAIR, b.scene.points.shape[0])
            cols = 1 + k
            self.assertTrue(cap.padded)
            self.assertEqual(cap.count, b.S * cols)
            self.assertTrue(torch.equal(cap.sample, torch.arange(b.S).repeat_interleave(cols)))
            self.assertGreater(compact.count, 0)
            self.assertLess(compact.count, cap.count)
            self.assertEqual(int(cap.valid.sum()), compact.count)
            for name in ("sample", "cell", "obj", "partner_point", "partner_normal", "kind", "radius", "anchor"):
                self.assertTrue(torch.equal(getattr(cap, name)[cap.valid], getattr(compact, name)), name)
            # token_offsets: static capacity CSR; valid counts per cell reproduce the compacted offsets
            per_cell = torch.bincount(b.sample_cell, minlength=b.C) * cols
            self.assertEqual(cap.token_offsets.tolist(), [0, *per_cell.cumsum(0).tolist()])
            counts = torch.zeros(b.C, dtype=torch.long).index_add(0, cap.cell, cap.valid.long())
            self.assertEqual(counts.cumsum(0).tolist(), compact.token_offsets[1:].tolist())
            # every token sits inside its cell's capacity range, and the attention pairs cover exactly those ranges
            tok = torch.arange(cap.count)
            self.assertTrue(
                (tok >= cap.token_offsets[cap.cell]).all() and (tok < cap.token_offsets[cap.cell + 1]).all()
            )
            src, dst = cap.attn_pairs
            self.assertTrue(torch.equal(cap.cell[src], cap.cell[dst]))
            lengths = cap.token_offsets[1:] - cap.token_offsets[:-1]
            self.assertEqual(int(cap.attn_offsets[-1]), int((lengths**2).sum()))

    def test_layout_cached_per_batch(self):
        b, _, cap = self.pairs_both(0)
        cap2 = contact.detect(b, b.X + 0.1, b.V, capacity=True)
        self.assertIs(cap2.token_offsets, cap.token_offsets)
        self.assertIs(cap2.attn_pairs, cap.attn_pairs)

    def test_no_points_and_no_plane(self):
        gen = torch.Generator().manual_seed(1)
        g = Grid.build((2, 2, 3))
        b = make_batch([g], [random_scene(gen, g, 0, torch.float64, "cpu")], gen)
        cap = contact.detect(b, b.X, b.V, capacity=True)
        self.assertEqual(cap.count, b.S)
        self.assertTrue((cap.kind == 0).all())
        b.scene.plane_present[:] = False
        cap = contact.detect(b, b.X, b.V, capacity=True)
        self.assertFalse(cap.valid.any())
        self.assertTrue(torch.equal(contact.contact_energy(b, b.x), torch.zeros(1, dtype=torch.float64)))

    def test_energy_tokens_penetration_identical(self):
        for seed in range(3):
            b, compact, cap = self.pairs_both(seed)
            R = frames(
                hx.center_deformation(hx.modes(b.x[b.cells], b.hc)).float(), b.R_ref[b.cell_obj].float()
            ).double()
            x = b.x.clone().requires_grad_(True)
            b.pairs = compact
            E1 = contact.contact_energy(b, x)
            (g1,) = torch.autograd.grad(E1.sum(), x)
            pen1 = contact.penetration(b, b.x)
            tok1 = contact.contact_tokens(b, b.x, R)
            b.pairs = cap
            E2 = contact.contact_energy(b, x)
            (g2,) = torch.autograd.grad(E2.sum(), x)
            pen2 = contact.penetration(b, b.x)
            tok2 = contact.contact_tokens(b, b.x, R)
            self.assertGreater(E1.sum().item(), 0.0)
            self.assertLess((E1 - E2).abs().max().item(), 1e-12 * E1.abs().max().item())
            self.assertLess((g1 - g2).abs().max().item(), 1e-12 * g1.abs().max().item())
            self.assertTrue(torch.equal(pen1, pen2))
            self.assertTrue(torch.equal(tok2[cap.valid], tok1))
            self.assertTrue(torch.isfinite(tok2).all())


class TestPaddedEncoder(unittest.TestCase):
    """The contact encoder gives the same per-cell output from padded tokens as from the compacted ones."""

    def encoder_outputs(self, device, dtype, seed=0):
        gen = torch.Generator().manual_seed(seed)
        grids = [Grid.build((2, 2, 3), device=device), Grid.build((1, 2, 2), device=device)]
        scenes = [random_scene(gen, g, 6, dtype, device) for g in grids]
        b = make_batch(grids, scenes, gen, dtype, device)
        enc = ContactEncoder(16, 2).to(device=device, dtype=dtype)
        with torch.no_grad():
            for p in enc.parameters():
                p.add_(0.2 * torch.randn(p.shape, generator=gen, dtype=dtype).to(device))
        R = frames(hx.center_deformation(hx.modes(b.x[b.cells], b.hc)).float(), b.R_ref[b.cell_obj].float()).to(dtype)
        outs = []
        for capacity in (False, True):
            b.pairs = contact.detect(b, b.X, b.V, capacity=capacity)
            f = features(b, b.x, R, *_mode_inputs(b), torch.zeros(b.C, hx.MODE_COUNT, 3, dtype=dtype, device=device))
            if capacity:
                self.assertIsNotNone(f.token_valid)
                self.assertIsNotNone(f.token_attn)
            else:
                self.assertIsNone(f.token_valid)
            outs.append(enc(f.tokens, f.token_offsets, f.token_valid, f.token_cell, f.token_attn))
        return outs, b

    def test_cpu_float64(self):
        (compact, padded), b = self.encoder_outputs("cpu", torch.float64)
        self.assertEqual(padded.shape, (b.C, 17))
        self.assertGreater(compact.abs().max().item(), 0.0)
        self.assertLess((compact - padded).abs().max().item(), 1e-12)

    @unittest.skipUnless(HAS_CUDA, "cuda")
    def test_cuda_warp_attention(self):
        (compact, padded), _ = self.encoder_outputs(CUDA, torch.float32)
        self.assertLess((compact - padded).abs().max().item(), 1e-5 * compact.abs().max().item() + 1e-6)


def _mode_inputs(b):
    m = hx.modes(b.x[b.cells], b.hc)
    return m, hx.center_deformation(m)


def small_net(gen: torch.Generator, device, cfg=SMALL_CFG) -> Net:
    net = Net.from_config(cfg).to(device).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen).to(device))
    return net


class FullFloat32(unittest.TestCase):
    """TF32 matmuls patched out: cuBLAS picks different algorithms for the compacted and the padded token rows,
    whose TF32 rounding differences would hide behind the tolerances; in full float32 only atomics noise remains."""

    def setUp(self):
        self.precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.patch = unittest.mock.patch("torch.set_float32_matmul_precision")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        torch.set_float32_matmul_precision(self.precision)


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestCapturedQuery(FullFloat32):
    """Replaying the captured query reproduces eager queries (compacted pairs) on the same inputs."""

    grids = staticmethod(lambda: [Grid.build((2, 2, 3), device=CUDA), Grid.build((1, 2, 2), device=CUDA)])
    fusions = staticmethod(lambda: (Fusion(), Fusion()))  # (eager reference, captured)
    cfg = SMALL_CFG

    def states(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        net = small_net(gen, CUDA, self.cfg)
        grids = self.grids()
        scenes = [random_scene(gen, g, 6, torch.float32, CUDA) for g in grids]
        pair = []
        for capacity, fusion in zip((False, True), self.fusions(), strict=True):
            gen_b = torch.Generator().manual_seed(seed + 100)
            b = make_batch(grids, scenes, gen_b, torch.float32, CUDA)
            step = Step(net, fusion, pair_capacity=capacity)
            step.prepare(b, torch.ones(b.O, dtype=torch.bool, device=CUDA))
            pair.append((b, step))
        return pair

    def assert_same(self, b1, b2, tag, tol=1e-5):
        for name in ("x", "E", "gX", "hist_grad", "hist_update"):
            a, c = getattr(b1, name), getattr(b2, name)
            err = (a - c).abs().max().item()
            self.assertLess(err, tol * max(1.0, c.abs().max().item()), f"{tag}: {name} differs by {err:.3e}")
        self.assertTrue(torch.equal(b1.hist_valid, b2.hist_valid))

    def test_three_queries_and_an_advance(self):
        (b1, s1), (b2, s2) = self.states()
        self.assertGreater(int(b2.pairs.valid.sum()), 0)
        sel = torch.ones(b1.O, dtype=torch.bool, device=CUDA)
        x0, E0, hv0 = b2.x.clone(), b2.E.clone(), b2.hist_valid.clone()
        with torch.no_grad():
            cq = CapturedQuery(s2, b2)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(b2.x, x0) and torch.equal(b2.E, E0) and torch.equal(b2.hist_valid, hv0))
            self.assert_same(b1, b2, "before")
            for k in range(3):
                s1.commit(b1, s1.query(b1))
                out = cq.replay()
                torch.cuda.synchronize()
                self.assertFalse(torch.equal(b2.x, x0))  # the network moves the candidate
                self.assertTrue(torch.equal(out.E_after, b2.E))
                self.assert_same(b1, b2, f"query {k}")
            s1.advance(b1, sel)
            s2.advance(b2, sel)
            self.assertTrue(b2.pairs is not cq.pairs)  # prepare rebound the pairs ...
            cq.sync()
            self.assertTrue(b2.pairs is cq.pairs)  # ... sync copied them into the static buffers
            self.assertTrue(b2.x is cq.state["x"])
            self.assert_same(b1, b2, "after advance")
            s1.commit(b1, s1.query(b1))
            cq.replay()
            torch.cuda.synchronize()
            self.assert_same(b1, b2, "query after advance")

    def test_requires_capacity_pairs(self):
        (b1, s1), _ = self.states()
        with self.assertRaises(ValueError):
            CapturedQuery(s1, b1)


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestCapturedQueryVoxelGrid(TestCapturedQuery):
    """One voxel grid in the batch with the multigrid fusion solver: the fixed-iteration PCG is recorded into the
    query's graph (the eager reference runs the PCG without its own graphs)."""

    grids = staticmethod(lambda: [Grid.from_voxels(holed_box(), "zmin_face", CUDA)])
    fusions = staticmethod(lambda: (Fusion("mg", options={"capture": False}), Fusion("mg")))

    def test_three_queries_and_an_advance(self):
        super().test_three_queries_and_an_advance()
        _, (b2, s2) = self.states()
        from experiments.lido.multigrid import MultigridFactor

        fac = s2.fusion.factor(b2.grids[0])
        self.assertIsInstance(fac, MultigridFactor)
        self.assertEqual(fac.iterations, 8)

    def test_solver_graphs_outside_the_captured_query(self):
        """The captured step's factor records its own graph only when solving outside a capture (the warm-up)."""
        (b1, s1), (b2, s2) = self.states()
        with torch.no_grad():
            cq = CapturedQuery(s2, b2)
            fac = s2.fusion.factor(b2.grids[0])
            self.assertEqual(len(fac._graphs), 1)  # the warm-up queries recorded the (Pf, 3) graph
            self.assertEqual(len(s1.fusion.factor(b1.grids[0])._graphs), 0)
            cq.replay()
            torch.cuda.synchronize()
            self.assertEqual(len(fac._graphs), 1)


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestCompiledInference(FullFloat32):
    """The compiled rollout chains (layer as one graph through the attention custom op, edge encoder, feature
    chains), compiled exactly as the rollout does it, give the eager query's results; one object as in the
    rollout and two objects as in the Newton solver."""

    cfg = SMALL_CFG

    def check(self, grids):
        gen = torch.Generator().manual_seed(5)
        net = small_net(gen, CUDA, self.cfg)
        scenes = [random_scene(gen, g, 6, torch.float32, CUDA) for g in grids]
        b = make_batch(grids, scenes, torch.Generator().manual_seed(105), torch.float32, CUDA)
        step = Step(net, Fusion(), pair_capacity=True)
        step.prepare(b, torch.ones(b.O, dtype=torch.bool, device=CUDA))
        with torch.no_grad():
            ref = step.query(b)
            compile_inference(net, step)
            out = step.query(b)
        for name in ("cand_after", "E_after", "gX_after", "step", "dm_world"):
            a, c = getattr(out, name), getattr(ref, name)
            err = (a - c).abs().max().item()
            self.assertLess(err, 1e-5 * max(1.0, c.abs().max().item()), f"{name} differs by {err:.3e}")
        self.assertGreater((ref.cand_after - b.x).abs().max().item(), 0.0)

    def test_one_object(self):
        self.check([Grid.build((2, 2, 3), device=CUDA)])

    def test_two_objects(self):
        self.check([Grid.build((2, 2, 3), device=CUDA), Grid.build((1, 2, 2), device=CUDA)])


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestRolloutCapture(FullFloat32):
    def test_capture_matches_eager(self):
        gen = torch.Generator().manual_seed(3)
        net = small_net(gen, "cpu")
        ck = {"network_state": net.state_dict(), "config": SMALL_CFG.to_dict()}
        scenario = {
            "cell_counts": (2, 2, 3),
            "h": 0.05,
            "dt": 1.0 / 300.0,
            "E": 1e5,
            "nu": 0.3,
            "rho": 1000.0,
            "eta": 50.0,
            "gravity": (0.0, -9.81, 0.0),
            "contact": {
                "plane_present": True,
                "plane_height": -0.02,
                "kappa": 100.0,
                "beta": 0.1,
                "mu_f": 0.3,
                "points": [[0.05, -0.03, 0.1], [0.1, 0.12, 0.05]],
                "normals": [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                "radii": [0.03, 0.04],
            },
        }
        eager = rollout(ck, {}, scenario, 3, 2, device=CUDA, capture=False)
        captured = rollout(ck, {}, scenario, 3, 2, device=CUDA, capture=True)
        self.assertFalse(eager["captured"])
        self.assertTrue(captured["captured"])
        self.assertGreater(np.abs(eager["positions"][-1] - eager["positions"][0]).max(), 1e-4)
        scale = np.abs(eager["positions"]).max()
        self.assertLess(np.abs(eager["positions"] - captured["positions"]).max(), 2e-5 * scale)
        self.assertLess(
            np.abs(eager["velocities"] - captured["velocities"]).max(), 2e-5 * np.abs(eager["velocities"]).max()
        )
        for key in ("energy_joule", "residual_n", "penetration_r"):
            self.assertLess(np.abs(eager[key] - captured[key]).max(), 1e-4 * max(1.0, np.abs(eager[key]).max()), key)


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestSolverCapture(unittest.TestCase):
    def test_two_bodies_three_steps(self):
        model = solver_tests.build_model(contact={"kappa": 100.0, "beta": 0.0, "mu_f": 0.2}, device="cuda:0")
        solver = SolverLIDO(model, None, iterations=2, cfg=SMALL_CFG, capture=True)
        self.assertTrue(solver.capture)
        self.assertIsNone(solver.captured)
        checker = solver_tests.TestSolverLIDO()
        checker.run_steps(model, solver, 3)
        self.assertIsNotNone(solver.captured)
        self.assertTrue(solver.capture)
        self.assertTrue(solver.batch.pairs.padded)  # advance re-prepared: the next step's sync copies them in
        self.assertTrue(solver.batch.x is not solver.captured.state["x"])
        solver.captured.sync()
        self.assertTrue(solver.batch.x is solver.captured.state["x"])
        self.assertTrue(solver.batch.pairs is solver.captured.pairs)


if __name__ == "__main__":
    unittest.main()
