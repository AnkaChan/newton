# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pinned bodies in v5 scenes (design spec section 11, Anka 2026-10-02): the six face pin patterns of `Grid.build`
against `Grid.from_voxels` masks, a pinned body hanging in the air with a free body dropped onto it (pins exactly
held over the steps, the free body lands with finite energies and a bounded penetration), and a mixed scene (half
the bodies pinned) through the scene runner's epoch and the v5 validation records on the CPU."""

import dataclasses
import math
import unittest

import torch

from experiments.lido import contact, physics, scenes_v5
from experiments.lido import jobs as J
from experiments.lido import validation as V
from experiments.lido.batch import Batch
from experiments.lido.fusion import Fusion, KronFactor
from experiments.lido.grid import FACE_PINS, Grid, face_pin_mask
from experiments.lido.network import Net
from experiments.lido.runner import scene_batch
from experiments.lido.step import Step
from experiments.lido.structs import Material
from experiments.lido.tests.test_body_contact import MATERIAL_FIELDS, SI, SMALL_CFG, world_scene
from experiments.lido.tests.test_scene_runner import MASTER, make_runner_cpu, scene_cfg
from experiments.lido.units import material_from_si

R = contact.R_SAMPLE


class TestGridFaces(unittest.TestCase):
    def test_build_face_variants_equal_voxel_masks(self):
        """Every face pattern of `Grid.build` equals `Grid.from_voxels` on full occupancy with the face's corner
        mask, field by field (pins, flags, exposure, samples and the reference corners included)."""
        for cc in ((2, 3, 4), (3, 2, 2)):
            nx, ny, nz = cc
            lx, ly, lz = torch.meshgrid(torch.arange(nx + 1), torch.arange(ny + 1), torch.arange(nz + 1), indexing="ij")
            lattice = torch.stack([lx, ly, lz], -1).reshape(-1, 3)
            for pins in FACE_PINS:
                g = Grid.build(cc, pins)
                self.assertEqual(g.key, (*cc, pins))
                self.assertEqual((g.kind, g.pins), ("box", pins))
                mask = face_pin_mask(lattice, cc, pins)
                v = Grid.from_voxels(torch.ones(*cc, dtype=torch.bool), mask.reshape(nx + 1, ny + 1, nz + 1))
                for f in dataclasses.fields(Grid):
                    a, b = getattr(g, f.name), getattr(v, f.name)
                    if f.name == "key":
                        continue
                    if f.name == "samples":
                        for n in ("cell", "face", "corners"):
                            self.assertTrue(torch.equal(getattr(a, n), getattr(b, n)), (pins, n))
                    elif isinstance(a, torch.Tensor):
                        self.assertTrue(torch.equal(a, b), (pins, f.name))
                self.assertTrue(torch.equal(g.pinned_mask, mask))
                self.assertTrue(torch.equal(g.fixed_flags, mask[g.cells]))
                self.assertTrue(
                    (g.rest[g.ref_corners][:, "xyz".index(pins[0])] == g.rest[g.pinned[0], "xyz".index(pins[0])]).all()
                )
        # "zmin_face" and "none" are the patterns they were (the hard-coded reference corners)
        z = Grid.build((2, 3, 4))
        self.assertTrue(torch.equal(z.pinned_mask, z.rest[:, 2] == 0))
        self.assertEqual(z.ref_corners.tolist(), [0, 55, 15])
        self.assertEqual(Grid.build((2, 3, 4), "none").ref_corners.tolist(), [0, 55, 15])
        with self.assertRaises(ValueError):
            Grid.build((2, 2, 2), "ymid_face")


def hanging_and_falling(dtype=torch.float64, gap: float = 1.0 + R, kappa: float = 20.0) -> tuple[Batch, Step]:
    """A 3x2x3 block hanging from its pinned top face (y = 6) above the plane at y = 0 and a free 2x2x2 box whose
    bottom is `gap` cells above the block's top face, offset in (x, z); zero-init network (no shape proposal),
    implicit-contact centroid update, damping on."""
    gA, gB = Grid.build((3, 2, 3), "ymax_face"), Grid.build((2, 2, 2), "none")
    b = Batch.build([gA, gB], "cpu", dtype)
    b.material = Material.cat(
        [
            material_from_si(cell_count=g.C, sample_count=g.S, kappa=kappa, beta=0.3, mu_f=0.0, device="cpu", **SI)
            for g in (gA, gB)
        ]
    )
    for f in MATERIAL_FIELDS:
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    b.scene = world_scene(2, 0.0, dtype, "cpu")
    b.body_contact = True
    X = b.rest.to(dtype).clone()
    X[b.corner_obj == 0] += torch.tensor([0.0, 4.0, 0.0], dtype=dtype)  # A: y in [4, 6], the y = 6 face pinned
    X[b.corner_obj == 1] += torch.tensor([0.4, 6.0 + gap, 0.6], dtype=dtype)
    b.X, b.V = X, torch.zeros_like(X)
    b.X_prev, b.x = X.clone(), X.clone()
    step = Step(Net.from_config(SMALL_CFG).to(dtype).eval(), Fusion(), translation="implicit_contact")
    return b, step


class TestHangingBodyAndFreeFall(unittest.TestCase):
    def test_pins_held_free_body_lands(self):
        b, step = hanging_and_falling()
        sel = torch.ones(2, dtype=torch.bool)
        step.prepare(b, sel)
        self.assertEqual(b.free_objects.tolist(), [False, True])
        self.assertEqual(len(b.groups), 2)  # a pinned and a free grid never share a fusion group
        for grp in b.groups:
            self.assertIsInstance(step.fusion.factor(grp.grid, torch.float64), KronFactor)
        self.assertEqual(b.pairs.count, 0)  # a cell above: nothing touches at the start
        X0 = b.X.clone()
        pin = b.pinned
        rows_b = b.corner_obj == 1
        top_a = (b.sample_face == 3) & (b.sample_obj == 0)
        self.assertTrue(pin[b.sample_corners[top_a]].all())  # the top face's samples sit on pinned corners
        ys, body_pairs, pens = [], [], []
        for _ in range(30):
            for _ in range(4):
                out = step.query(b)
                self.assertTrue(torch.isfinite(out.E_after).all())
                step.commit(b, out)
            step.advance(b, sel)
            self.assertTrue(torch.isfinite(b.X).all() and torch.isfinite(b.E).all())
            # the pinned corners never move: exactly X over the steps, zero velocity, candidate at X
            self.assertTrue(torch.equal(b.X[pin], X0[pin]))
            self.assertLess((b.x[pin] - X0[pin]).abs().max().item(), 1e-12)
            self.assertEqual(b.V[pin].abs().max().item(), 0.0)
            self.assertLess((b.Y[pin] - X0[pin]).abs().max().item(), 1e-12)
            ys.append(physics.centroid(b, b.X)[1, 1].item())
            body_pairs.append(int(((b.pairs.kind == contact.KIND_BODY) & b.pairs.valid).sum()))
            pens.append(contact.kind_penetration(b, b.X)[contact.KIND_BODY].item())
        # the free body falls (free fall until it touches), then rests on the hanging body
        g = b.material.g[1].norm().item()
        y_start = physics.centroid(b, X0)[1, 1].item()
        first = next(i for i, n in enumerate(body_pairs) if n > 0)
        self.assertGreater(first, 5)
        for i in range(first):
            self.assertAlmostEqual(ys[i], y_start - 0.5 * g * (i + 1) * (i + 2), delta=1e-9)
        self.assertGreater(body_pairs[-1], 0)
        self.assertLess(ys[-1], ys[0])
        self.assertLess(abs(ys[-1] - ys[-2]), 1e-6)  # at rest
        # it landed on the hanging body, not on the plane: its bottom rests one sample radius over the pinned face
        # less the penetration, which stays bounded (the static value with kappa 20 is under a tenth of the radius)
        bottom = b.X[rows_b, 1].min().item()
        self.assertGreater(bottom, 6.0 + R - 0.1 * R)
        self.assertLess(bottom, 6.0 + R + 1e-6)
        self.assertLess(max(pens), 0.1)
        self.assertEqual(int(((b.pairs.kind == 0) & b.pairs.valid & (b.pairs.obj == 1)).sum()), 0)
        # the pinned body's free corners moved (the zero-init network lets its lower part sag) but stayed finite
        self.assertGreater((b.X[(b.corner_obj == 0) & ~pin] - X0[(b.corner_obj == 0) & ~pin]).abs().max().item(), 0.0)

    def test_pins_held_in_float32(self):
        b, step = hanging_and_falling(torch.float32, gap=R)
        sel = torch.ones(2, dtype=torch.bool)
        step.prepare(b, sel)
        X0 = b.X.clone()
        for _ in range(5):
            step.commit(b, step.query(b))
            step.advance(b, sel)
            self.assertTrue(torch.equal(b.X[b.pinned], X0[b.pinned]))
            self.assertEqual(b.V[b.pinned].abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(b.E).all())


def mixed_scene(cfg, validation: bool = False) -> scenes_v5.SceneV5:
    """The first scene (by seed) of the stream with both pinned and free bodies (the test scenes hold 3-5 bodies)."""
    for seed in range(32):
        scene = scenes_v5.sample_scene(MASTER, seed, cfg, validation=validation)
        if 0 < sum(1 for b in scene.bodies if b.pinned) < len(scene.bodies):
            return scene
    raise AssertionError("no mixed scene in 32 seeds")


class TestMixedScene(unittest.TestCase):
    """A scene of about 1500 cells with half the bodies pinned on the CPU: realise, the runner's epoch, the
    validation records and the momentum drift of the free bodies."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = scene_cfg(pinned_body_fraction=0.5)

    def test_realised_scene_holds_its_pins_over_steps(self):
        scene = mixed_scene(self.cfg)
        runner, step = make_runner_cpu(self.cfg)
        b = scene_batch(scene, runner.grids, runner.aug, "cpu")
        self.assertEqual(b.free_objects.tolist(), [not body.pinned for body in scene.bodies])
        self.assertTrue(b.any_free)
        grid_keys = {grp.grid.key for grp in b.groups}
        self.assertEqual(len(grid_keys), len(b.groups))
        for grp in b.groups:
            self.assertEqual(len({bool(b.free_objects[o]) for o in grp.objects.tolist()}), 1)
        X0 = b.X.clone()
        self.assertEqual(b.V[b.pinned].abs().max().item(), 0.0)
        step.prepare(b, b.active, None)
        for _ in range(3):
            step.commit(b, step.query(b))
            step.advance(b, b.active, None)
            self.assertTrue(torch.equal(b.X[b.pinned], X0[b.pinned]))
            self.assertEqual(b.V[b.pinned].abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(b.E).all())
        # the free bodies fell, the pinned bodies' centroids stayed within their deformation
        c0, c1 = physics.centroid(b, X0), physics.centroid(b, b.X)
        free = b.free_objects
        self.assertTrue((c1[free, 1] < c0[free, 1]).all())

    def test_runner_epoch_and_validation_records(self):
        cfg = self.cfg
        runner, step = make_runner_cpu(cfg)
        runner.start_epoch(1)
        jobs = J.sample_epoch_jobs(MASTER, 1, cfg)
        pinned_seen = free_seen = 0
        for _ in range(runner.U):
            self.assertIsNotNone(runner.job)
            pinned_seen += int((~runner.batch.free_objects).sum())
            free_seen += int(runner.batch.free_objects.sum())
            runner.commit(step.query(runner.batch))
        self.assertGreater(pinned_seen, 0)
        self.assertGreater(free_seen, 0)
        self.assertEqual(runner.loaded_jobs, cfg.scene_count)
        self.assertEqual(runner.failures, [])
        self.assertEqual(runner.queries, sum(s["bodies"] * s["K"] * s["H"] for s in runner.scene_summaries))
        self.assertEqual(len(runner.pair_history), sum(1 + j.H for j in jobs))
        summary = runner.epoch_summary()
        self.assertGreater(summary["pinned_bodies_mean"], 0.0)
        self.assertLess(summary["pinned_bodies_mean"], summary["bodies_mean"])
        self.assertTrue(all("pinned_bodies" in s for s in summary["scenes_served"]))
        # validation records on the mixed held-out scenes
        grids, aug = runner.grids, runner.aug
        records = V.validate_cheap_v5(step, cfg, grids, aug, "cpu", MASTER)
        self.assertEqual(len(records), cfg.validation_scene_count)
        for r in records:
            self.assertTrue(r["survived"])
            self.assertIn("pinned_bodies", r["scene"])
            self.assertTrue(all(math.isfinite(v) for v in r["residual_n"] + r["energy_joule"]))
        full, _ = V.validate_full_horizon_v5(step, cfg, grids, aug, "cpu", MASTER, 1, 2)
        self.assertTrue(all(r["survived"] for r in full))
        drift = full[0]["momentum_drift"]
        self.assertTrue(math.isfinite(drift))
        self.assertLess(drift, 1e-3)

    def test_momentum_check_counts_free_bodies_only(self):
        cfg = self.cfg
        runner, step = make_runner_cpu(cfg)
        scene = mixed_scene(cfg, validation=True)
        free = V.contact_free_copy(scene, 8)
        b = scene_batch(free, runner.grids, runner.aug, "cpu", plane=False)
        step.prepare(b, b.active, None)
        self.assertEqual(b.pairs.count, 0)
        mass_free = sum(
            float(b.material.si[o]["rho"]) * float(b.mass[b.corner_obj == o].sum()) * scene.h**3
            for o in range(b.O)
            if bool(b.free_objects[o])
        )
        self.assertAlmostEqual(V.total_mass_si(b), mass_free, places=9)
        p = V.momentum_si(b, b.V)
        self.assertEqual(tuple(p.shape), (3,))
        drift = V.momentum_drift_check(step, cfg, runner.grids, runner.aug, "cpu", MASTER, 1, 8, scene)
        self.assertTrue(math.isfinite(drift))
        self.assertLess(drift, 1e-3)
        # a scene of pinned bodies only has no free fall to measure
        all_pinned = dataclasses.replace(
            scene,
            bodies=[dataclasses.replace(body, pins="zmin_face", velocity=(0.0, 0.0, 0.0)) for body in scene.bodies],
        )
        self.assertTrue(
            math.isnan(V.momentum_drift_check(step, cfg, runner.grids, runner.aug, "cpu", MASTER, 1, 1, all_pinned))
        )


if __name__ == "__main__":
    unittest.main()
