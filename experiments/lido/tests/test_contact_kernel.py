# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Warp contact kernel and the fused inference energy pass against the torch paths: pair energies and
gradients on random two-object scenes (penetration, approach, tangential slip inside and outside the friction
band, padded rows), `physics.energy_and_grad` through the fused pass (values, pinned rows, launch count), and
the dispatch rules that keep the training path on torch autograd."""

import contextlib
import unittest
import unittest.mock

import torch
from torch.profiler import ProfilerActivity, profile

from experiments.lido import contact, contact_kernel, physics
from experiments.lido.batch import _empty_pairs
from experiments.lido.grid import Grid
from experiments.lido.tests.test_capture import make_batch, random_scene

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")


@contextlib.contextmanager
def torch_paths(contact_only: bool = False):
    saved = contact.USE_WARP, physics.USE_WARP
    contact.USE_WARP = False
    if not contact_only:
        physics.USE_WARP = False
    try:
        yield
    finally:
        contact.USE_WARP, physics.USE_WARP = saved


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def scene_batch(seed: int, variant: str):
    """Two objects over a plane with six discs each; the candidate relative to the friction anchor (X) selects
    the regime: `slip` (large tangential motion), `band` (sub-eps slip, pushed into the plane), `approach`
    (moving into the plane), `rest` (the anchor itself, zero slip)."""
    gen = torch.Generator().manual_seed(seed)
    grids = [Grid.build((2, 2, 3), device=CUDA), Grid.build((3, 2, 2), device=CUDA)]
    scenes = [random_scene(gen, g, 6, torch.float32, CUDA) for g in grids]
    b = make_batch(grids, scenes, gen, torch.float32, CUDA)
    noise = torch.randn(b.N, 3, generator=gen).to(CUDA)
    down = torch.tensor([0.0, -1.0, 0.0], device=CUDA)
    if variant == "slip":
        x = b.X + 0.3 * noise + 0.3 * down
    elif variant == "band":
        x = b.X + 3e-4 * noise + 0.15 * down
    elif variant == "approach":
        x = b.X + 0.02 * noise + 0.4 * down
    else:
        x = b.X.clone()
    x[b.pinned] = b.X[b.pinned]
    b.x = x
    b.pairs = contact.detect(b, b.X, b.V, capacity=True)
    return b


def regimes(b, x):
    """Per valid pair: penetrating, approaching, slip inside the band, slip outside the band (torch geometry)."""
    pairs = b.pairs
    xs, n, gap, r_total = contact._geometry(b, x, pairs)
    d = r_total - gap
    delta = xs - pairs.anchor
    vn = (n * delta).sum(-1)
    u = delta - vn[:, None] * n
    y = u.norm(dim=-1)
    eps = b.material.friction_eps[pairs.obj]
    pen = (d > 0) & pairs.valid
    return pen, pen & (vn < 0), pen & (y < eps) & (y > 0), pen & (y >= eps)


@unittest.skipUnless(HAS_CUDA, "needs cuda")
class TestContactKernel(unittest.TestCase):
    def energies_and_grads(self, b):
        x = b.x.clone().requires_grad_(True)
        E_w = contact.contact_energy(b, x)
        (g_w,) = torch.autograd.grad(E_w.sum(), x)
        with torch_paths(contact_only=True):
            E_t = contact.contact_energy(b, x)
            (g_t,) = torch.autograd.grad(E_t.sum(), x)
        return E_w, g_w, E_t, g_t

    def check(self, b, tag):
        E_w, g_w, E_t, g_t = self.energies_and_grads(b)
        self.assertEqual(E_w.shape, (b.O,))
        self.assertTrue(torch.isfinite(E_w).all() and torch.isfinite(g_w).all())
        self.assertGreater(E_t.abs().max().item(), 0.0, tag)
        self.assertLess(rel_err(E_w, E_t), 1e-5, f"{tag}: energy")
        self.assertLess(rel_err(g_w, g_t), 1e-4, f"{tag}: gradient")

    def test_regimes_match_torch(self):
        seen = {"pen": 0, "approach": 0, "band": 0, "slip": 0}
        for seed in range(3):
            for variant in ("slip", "band", "approach", "rest"):
                b = scene_batch(seed, variant)
                self.assertTrue(b.pairs.padded and not b.pairs.valid.all())
                pen, app, band, slip = regimes(b, b.x)
                for key, mask in zip(seen, (pen, app, band, slip), strict=True):
                    seen[key] += int(mask.sum())
                if variant == "band":
                    self.assertGreater(int(band.sum()), 0, f"seed {seed}: no sub-eps slip under load")
                if variant == "slip":
                    self.assertGreater(int(slip.sum()), 0, f"seed {seed}: no slip beyond the band")
                if variant == "approach":
                    self.assertGreater(int(app.sum()), 0, f"seed {seed}: nothing approaching")
                self.check(b, f"seed {seed} {variant}")
        self.assertTrue(all(v > 0 for v in seen.values()), seen)

    def test_rest_has_no_friction_gradient(self):
        """At the anchor the slip is zero: the floored |u| passes no gradient, as the torch clamp_min."""
        b = scene_batch(1, "rest")
        pen, _, _, _ = regimes(b, b.x)
        self.assertGreater(int(pen.sum()), 0)
        self.check(b, "rest")

    def test_padded_rows_ignored(self):
        b = scene_batch(2, "slip")
        E_w, g_w, _, _ = self.energies_and_grads(b)
        invalid = ~b.pairs.valid
        self.assertGreater(int(invalid.sum()), 0)
        # deeply penetrating partners on the padded rows must not register
        b.pairs.partner_point[invalid] = b.pairs.partner_point[invalid] + 5.0 * b.pairs.partner_normal[invalid]
        b.pairs.anchor[invalid] = b.pairs.anchor[invalid] + 1.0
        E_w2, g_w2, E_t2, g_t2 = self.energies_and_grads(b)
        self.assertLess(rel_err(E_w2, E_w), 1e-6)  # atomics: the summation order may differ between runs
        self.assertLess(rel_err(g_w2, g_w), 1e-6)
        self.assertLess(rel_err(E_w2, E_t2), 1e-5)
        self.assertLess(rel_err(g_w2, g_t2), 1e-4)

    def test_backward_scales_with_object_weights(self):
        b = scene_batch(3, "slip")
        w = torch.tensor([2.0, -0.5], device=CUDA)
        grads = []
        for ctx in (contextlib.nullcontext(), torch_paths(contact_only=True)):
            with ctx:
                x = b.x.clone().requires_grad_(True)
                (contact.contact_energy(b, x) * w).sum().backward()
                grads.append(x.grad)
        self.assertLess(rel_err(grads[0], grads[1]), 1e-4)

    def test_compacted_pairs_stay_on_torch(self):
        b = scene_batch(4, "slip")
        b.pairs = contact.detect(b, b.X, b.V)
        self.assertFalse(b.pairs.padded)
        with unittest.mock.patch.object(contact_kernel, "launch_contact", side_effect=AssertionError("kernel")):
            E = contact.contact_energy(b, b.x)
        self.assertGreater(E.abs().max().item(), 0.0)
        with torch_paths(contact_only=True):
            self.assertLess(rel_err(E, contact.contact_energy(b, b.x)), 1e-6)  # index_add atomics: not bitwise


@unittest.skipUnless(HAS_CUDA, "needs cuda")
class TestFusedPass(unittest.TestCase):
    def check_fused(self, b, tag, with_contact=True):
        self.assertTrue(physics.fused_pass_applies(b, b.x), tag)
        E, gX = physics.energy_and_grad(b, b.x)
        with torch_paths():
            self.assertFalse(physics.fused_pass_applies(b, b.x))
            E_t, g_t = physics.energy_and_grad(b, b.x)
        self.assertEqual(E.shape, (b.O,))
        self.assertFalse(E.requires_grad or gX.requires_grad)
        self.assertTrue(torch.isfinite(E).all() and torch.isfinite(gX).all())
        self.assertTrue((gX[b.pinned] == 0).all())
        self.assertLess(rel_err(E, E_t), 1e-5, f"{tag}: energy")
        self.assertLess(rel_err(gX, g_t), 1e-4, f"{tag}: gradient")
        if with_contact:
            with torch_paths():
                self.assertGreater(contact.contact_energy(b, b.x).abs().max().item(), 0.0, tag)
        # energy() alone (segment sums plus the contact Function) gives the same per-object totals
        self.assertLess(rel_err(physics.energy(b, b.x), E), 1e-5, f"{tag}: energy()")
        return E, gX

    def test_matches_torch_in_every_regime(self):
        for seed in range(2):
            for variant in ("slip", "band", "approach"):
                self.check_fused(scene_batch(seed, variant), f"seed {seed} {variant}")

    def test_without_pairs(self):
        b = scene_batch(0, "slip")
        b.pairs = _empty_pairs(b.C, CUDA)
        self.check_fused(b, "no pairs", with_contact=False)
        b = scene_batch(0, "slip")
        b.scene.plane_present[:] = False
        b.scene.points = b.scene.points + 100.0
        b.pairs = contact.detect(b, b.X, b.V, capacity=True)
        self.assertFalse(b.pairs.valid.any())
        self.check_fused(b, "all rows padded", with_contact=False)

    def test_launch_count(self):
        b = scene_batch(1, "slip")
        physics.energy_and_grad(b, b.x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            physics.energy_and_grad(b, b.x)
            torch.cuda.synchronize()
        launches = sum(e.count for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA)
        self.assertLessEqual(launches, 4, f"{launches} device launches")  # zero fill + cells + pairs + corners

    def test_dispatch(self):
        b = scene_batch(2, "slip")
        self.assertTrue(physics.fused_pass_applies(b, b.x))
        self.assertFalse(physics.fused_pass_applies(b, b.x.clone().requires_grad_(True)))
        self.assertFalse(physics.fused_pass_applies(b, b.x.double()))
        b.pairs = contact.detect(b, b.X, b.V)  # compacted: training layout
        self.assertFalse(physics.fused_pass_applies(b, b.x))
        with torch_paths(contact_only=True):
            b.pairs = contact.detect(b, b.X, b.V, capacity=True)
            self.assertFalse(physics.fused_pass_applies(b, b.x))

    def test_training_route_unchanged(self):
        """A candidate with a graph takes the autograd route (Warp cells Function, contact Function on padded
        pairs) and agrees with the torch path."""
        b = scene_batch(3, "approach")
        grads = []
        for ctx in (contextlib.nullcontext(), torch_paths()):
            with ctx:
                src = torch.zeros(b.N, 3, device=CUDA, requires_grad=True)
                E, gX = physics.energy_and_grad(b, b.x + src)
                self.assertTrue(E.requires_grad)
                (E * torch.tensor([1.0, 3.0], device=CUDA)).sum().backward()
                grads.append((E.detach(), gX, src.grad))
        for a, c in zip(grads[0], grads[1], strict=True):
            self.assertLess(rel_err(a, c), 1e-4)


if __name__ == "__main__":
    unittest.main()
