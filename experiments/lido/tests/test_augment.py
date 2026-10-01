# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from experiments.lido.augment import Augmenter
from experiments.lido.grid import Grid
from experiments.lido.structs import SceneSpec

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_spec(seed, cell_counts, perturbation_scale=0.7, strength=0.05, velocity_dt=0.05):
    return SceneSpec(
        seed=seed,
        cell_counts=cell_counts,
        h=0.025,
        dt=1.0 / 300.0,
        pins="zmin_face",
        E=1e4,
        nu=0.3,
        rho=1e3,
        eta=10.0,
        gravity=(0.0, -9.81, 0.0),
        perturbation_scale=perturbation_scale,
        strength=strength,
        velocity_dt=velocity_dt,
    )


def rms(f):
    return f.double().pow(2).sum(-1).mean(-1).sqrt().item()


def neighbour_diff(f, cell_counts):
    """Mean norm of the difference between axis-adjacent corners."""
    nx, ny, nz = cell_counts
    v = f.reshape(nx + 1, ny + 1, nz + 1, 3)
    d = [(v[1:] - v[:-1]), (v[:, 1:] - v[:, :-1]), (v[:, :, 1:] - v[:, :, :-1])]
    return torch.cat([x.norm(dim=-1).flatten() for x in d]).mean().item()


class TestAugmenter(unittest.TestCase):
    def gen(self, device, seed):
        return torch.Generator(device=device).manual_seed(seed)

    def test_field_rms(self):
        for device in DEVICES:
            grid = Grid.build((10, 10, 40), device=device)
            aug = Augmenter(device)
            for target in (0.03, 1.0, 3.5):
                f = aug.field(grid, self.gen(device, 3), target)
                self.assertEqual(tuple(f.shape), (grid.P, 3))
                self.assertLess(abs(rms(f) - target) / target, 1e-6, device)

    def test_zero_rms_gives_zeros(self):
        for device in DEVICES:
            grid = Grid.build((4, 4, 8), device=device)
            f = Augmenter(device).field(grid, self.gen(device, 1), 0.0)
            self.assertTrue(torch.all(f == 0))
            self.assertTrue(torch.isfinite(f).all())

    def test_smoothness(self):
        grid = Grid.build((10, 10, 40))
        aug = Augmenter("cpu")
        self.assertEqual(aug.wavelengths(grid), [2, 4, 8, 16, 32])
        o8 = aug.octave(grid, self.gen("cpu", 5), 8)
        ratio8 = neighbour_diff(o8, grid.cell_counts) / rms(o8)
        o2 = aug.octave(grid, self.gen("cpu", 5), 2)
        ratio2 = neighbour_diff(o2, grid.cell_counts) / rms(o2)
        self.assertLess(ratio8, 0.3)  # measured about 0.20; a white field would give about 1.4
        self.assertLess(ratio8, 0.5 * ratio2)
        f = aug.field(grid, self.gen("cpu", 5), 1.0)
        self.assertLess(neighbour_diff(f, grid.cell_counts), 0.2)  # multiscale field, RMS 1

    def test_determinism(self):
        for device in DEVICES:
            grid = Grid.build((6, 6, 12), device=device)
            aug = Augmenter(device)
            a = aug.field(grid, self.gen(device, 11), 1.0)
            b = aug.field(grid, self.gen(device, 11), 1.0)
            c = aug.field(grid, self.gen(device, 12), 1.0)
            self.assertTrue(torch.equal(a, b), device)
            self.assertFalse(torch.equal(a, c), device)
            spec = make_spec(0, grid.cell_counts)
            X1, V1 = aug.initial_state(grid, spec, self.gen(device, 21))
            X2, V2 = aug.initial_state(grid, spec, self.gen(device, 21))
            X3, V3 = aug.initial_state(grid, spec, self.gen(device, 22))
            self.assertTrue(torch.equal(X1, X2) and torch.equal(V1, V2))
            self.assertFalse(torch.equal(X1, X3) or torch.equal(V1, V3))
            n1 = aug.candidate_noise(grid, self.gen(device, 31))
            n2 = aug.candidate_noise(grid, self.gen(device, 31))
            n3 = aug.candidate_noise(grid, self.gen(device, 32))
            self.assertTrue(torch.equal(n1, n2))
            self.assertFalse(torch.equal(n1, n3))

    def test_initial_state_pins_and_scales(self):
        for device in DEVICES:
            grid = Grid.build((10, 10, 40), device=device)
            aug = Augmenter(device)
            spec = make_spec(0, grid.cell_counts, perturbation_scale=0.5, strength=0.04, velocity_dt=0.08)
            X, V = aug.initial_state(grid, spec, self.gen(device, 7))
            self.assertTrue(torch.equal(X[grid.pinned], grid.rest[grid.pinned]))
            self.assertTrue(torch.all(V[grid.pinned] == 0))
            self.assertFalse(torch.equal(X[grid.free], grid.rest[grid.free]))
            # the fields before the pin overwrite have RMS scale*strength*4.45 cells and scale*velocity_dt
            aug2 = Augmenter(device)
            g = self.gen(device, 7)
            d = aug2.field(grid, g, 0.5 * 0.04 * 4.45)
            v = aug2.field(grid, g, 0.5 * 0.08)
            self.assertLess(abs(rms(d) - 0.089) / 0.089, 1e-6)
            self.assertTrue(torch.allclose(X[grid.free], grid.rest[grid.free] + d[grid.free], atol=1e-6))
            self.assertTrue(torch.allclose(V[grid.free], v[grid.free], atol=1e-7))

    def test_candidate_noise(self):
        for device in DEVICES:
            grid = Grid.build((10, 10, 40), device=device)
            aug = Augmenter(device)
            for seed in range(5):
                n = aug.candidate_noise(grid, self.gen(device, seed), 0.01, 0.10)
                self.assertTrue(torch.all(n[grid.pinned] == 0))
                r = rms(n)
                self.assertGreater(r, 0.0)
                self.assertLessEqual(r, 0.10)
            free_rms = rms(aug.candidate_noise(grid, self.gen(device, 0), 0.05, 0.05)[grid.free])
            self.assertGreater(free_rms, 0.03)  # a fixed 5 % request stays near 5 % on the free corners
            self.assertLess(free_rms, 0.07)

    def test_batched_equals_single(self):
        for device in DEVICES:
            aug = Augmenter(device)
            g1 = Grid.build((4, 4, 8), device=device)
            g2 = Grid.build((3, 5, 6), device=device)
            grids = [g1, g2, g1, g1, g2, g2]
            specs = [
                make_spec(i, g.cell_counts, perturbation_scale=0.2 + 0.1 * i, strength=0.03 + 0.01 * i)
                for i, g in enumerate(grids)
            ]
            Xb, Vb = aug.initial_states(grids, specs, [self.gen(device, 100 + i) for i in range(len(grids))])
            for i, (g, s) in enumerate(zip(grids, specs, strict=True)):
                X, V = aug.initial_state(g, s, self.gen(device, 100 + i))
                self.assertTrue(torch.allclose(X, Xb[i], rtol=1e-6, atol=1e-6), (device, i))
                self.assertTrue(torch.allclose(V, Vb[i], rtol=1e-6, atol=1e-7), (device, i))
                self.assertTrue(torch.equal(Xb[i][g.pinned], g.rest[g.pinned]))
                self.assertTrue(torch.all(Vb[i][g.pinned] == 0))


if __name__ == "__main__":
    unittest.main()
