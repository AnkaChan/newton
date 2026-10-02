# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adversarial review of the body-body contact physics and the v5 data path (2026-10-02): the tests that pinned
the confirmed findings (as `expectedFailure` while open) and now guard the fixes.

Finding 1 (contact.py / physics.py / step.py, units; fixed by the common normaliser per scene, `material_from_si(mu_ref)`).
Every body's energy was in ITS OWN normalised units (mu_o h^3), and a body pair's energy belongs to the sample's
body, so its gradient on the PARTNER's corners was in the owner's force units added to rows in the partner's units:
in SI the partner felt the reaction scaled by mu_partner / mu_owner (90 % of the larger force lost at E = 1e5 vs
1e6), and the coupled centroid update changed the SI momentum of two touching bodies at rest. With one reference
modulus per scene (`realise`: the geometric mean of the bodies' moduli) all bodies share the unit of energy; the
third law and momentum conservation hold in SI, the scene's SI energy is the sum of the bodies' SI energies in body
mode, and the conditioning channels (the body's own dimensionless groups) are unchanged.

Finding 2 (runner.py / train.py; fixed in `SceneRunner.load`). A scene that fails (non-finite energy) when the rank's
queue is already empty stayed as the rank's batch with `active` all False and non-finite energies, so the trainer's
masked loss was NaN (NaN * 0): the idle batch is now reset to a finite state (candidate at X, zero energies).
"""

import dataclasses
import unittest

import torch

from experiments.lido import contact, physics, scenes_v5
from experiments.lido import hex as hx
from experiments.lido.augment import Augmenter
from experiments.lido.batch import Batch
from experiments.lido.config import TrainConfig
from experiments.lido.grid import Grid, GridCache, reference_rotation
from experiments.lido.step import Step
from experiments.lido.structs import _MATERIAL_TENSORS, Material
from experiments.lido.tests.test_body_contact import world_scene
from experiments.lido.tests.test_scene_runner import make_runner_cpu, scene_cfg
from experiments.lido.units import conditioning, energy_scale, force_scale, material_from_si, reference_modulus
from experiments.lido.validation import local_objective

H, DT = 0.025, 1.0 / 300.0


def two_boxes_si(moduli, gap=0.4, kappa=10.0, dtype=torch.float64, common_unit=True):
    """Two free 3x3x3 boxes in one world frame, B on top of A with its bottom samples `gap` cells over A's top face
    (gap < r: penetrating in the sample-sphere sense), one SI material per box (Young's modulus from `moduli`,
    the rest shared), no gravity, no plane, V = 0, pairs detected at X. `common_unit`: the scene's reference modulus
    as `scenes_v5.realise` passes it (False: each body its own unit, the configuration of the finding)."""
    dev = torch.device("cpu")
    grids = [Grid.build((3, 3, 3), pins="none", device=dev) for _ in moduli]
    b = Batch.build(grids, dev, dtype)
    mu_ref = reference_modulus([{"E": E, "nu": 0.3} for E in moduli]) if common_unit else None
    parts = [
        material_from_si(
            E=E,
            nu=0.3,
            rho=1000.0,
            eta=50.0,
            gravity=(0.0, 0.0, 0.0),
            h=H,
            dt=DT,
            cell_count=g.C,
            sample_count=g.S,
            kappa=kappa,
            beta=0.0,
            mu_f=0.0,
            device=dev,
            mu_ref=mu_ref,
        )
        for E, g in zip(moduli, grids, strict=True)
    ]
    b.material = Material.cat(parts)
    for f in _MATERIAL_TENSORS:
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    b.scene = world_scene(2, None, dtype, dev)
    b.body_contact = True
    X = b.rest.to(dtype).clone()
    X[b.corner_obj == 1] += torch.tensor([0.0, 3.0 + gap, 0.0], dtype=dtype)
    b.X, b.V = X, torch.zeros_like(X)
    b.X_prev, b.x, b.Y = X.clone(), X.clone(), X.clone()
    F_prev = hx.gauss_deformation(X[b.cells], b.hc)
    b.C_prev = hx.mat3_tn(F_prev, F_prev)
    b.m_Y = b.m_prev = hx.modes(X[b.cells], b.hc)
    b.R_ref = reference_rotation(X, b.ref_corners)
    b.c_n = physics.centroid(b, X)
    b.cdot_n = torch.zeros_like(b.c_n)
    b.pairs = contact.detect(b, b.X, b.V)
    return b


def si_mass(b) -> torch.Tensor:
    """Total mass per object [O] in kg: rho (SI) times the lumped cell volume."""
    rho = torch.tensor([d["rho"] for d in b.material.si], dtype=b.mass.dtype)
    return rho * physics.seg_sum(b.mass, b.corner_obj, b.O) * H**3


class TestThirdLawInSI(unittest.TestCase):
    """Finding 1: the body-pair reaction on the partner was in the owner's units; with the scene's common unit the
    third law and momentum conservation hold in SI for any stiffness ratio."""

    def test_equal_materials_control(self):
        b = two_boxes_si([1e5, 1e5])
        self.assertEqual(b.pairs.count, 18)  # 9 samples of each body against the other's face: both directions
        F = contact.contact_force(b, b.x) * force_scale(b.material)[:, None]  # newtons
        self.assertLess(float(F.sum(0).norm()), 1e-9 * float(F.norm(dim=1).max()))
        # effective stiffness between the two bodies: (n_A + n_B) ke, each direction a spring of ke
        ke_si, d = b.material.si[0]["ke"], (contact.R_SAMPLE - 0.4) * H
        self.assertAlmostEqual(float(F[1, 1]), 18 * ke_si * d, delta=1e-6 * 18 * ke_si * d)

    def test_own_units_record_the_finding(self):
        b = two_boxes_si([1e5, 1e6], common_unit=False)
        F = contact.contact_force(b, b.x) * force_scale(b.material)[:, None]
        self.assertGreater(float(F.sum(0).norm()), 0.8 * float(F.norm(dim=1).max()))  # 90 % of |F_B| lost

    def test_contact_force_sums_to_zero_in_si(self):
        for moduli in ([1e5, 1e6], [1e3, 1e6]):
            b = two_boxes_si(moduli)
            self.assertEqual(b.pairs.count, 18)
            scale = force_scale(b.material)
            self.assertTrue(torch.equal(scale[0], scale[1]))  # one unit of force for the scene
            F_norm = contact.contact_force(b, b.x)
            self.assertLess(float(F_norm.sum(0).norm()), 1e-9 * float(F_norm.norm(dim=1).max()))
            F = F_norm * scale[:, None]  # newtons
            self.assertLess(float(F.sum(0).norm()), 1e-9 * float(F.norm(dim=1).max()), moduli)
            # the pair law in SI is unchanged: 9 springs of each body's ke_si (kappa E_o h), one per direction
            ke_a, ke_b, d = b.material.si[0]["ke"], b.material.si[1]["ke"], (contact.R_SAMPLE - 0.4) * H
            self.assertAlmostEqual(float(F[1, 1]), 9 * (ke_a + ke_b) * d, delta=1e-6 * 9 * (ke_a + ke_b) * d)

    def test_coupled_centroid_update_conserves_si_momentum(self):
        for moduli in ([1e5, 1e6], [1e3, 1e6]):
            b = two_boxes_si(moduli)
            c_t, _ = Step(None).centroid_target(b, b.x)  # the joint Newton step over both free bodies, no gravity
            dc = (c_t - b.c_n) * H  # metres
            impulse = si_mass(b)[:, None] * dc  # kg m per step
            self.assertGreater(float(impulse.norm(dim=1).max()), 0.0)  # the bodies do separate
            self.assertLess(float(impulse.sum(0).norm()), 1e-6 * float(impulse.norm(dim=1).max()), moduli)


class TestCommonUnit(unittest.TestCase):
    """The common normaliser per scene (`material_from_si(mu_ref)`, `scenes_v5.realise`): the scene's SI energy is
    the sum of the bodies' SI energies in body mode, and the conditioning channels are those of body mode."""

    @staticmethod
    def step_start(b: Batch, X: torch.Tensor, V: torch.Tensor) -> None:
        b.X, b.V = X.clone(), V.clone()
        b.X_prev, b.x = X.clone(), X.clone()
        b.Y = X + V + b.material.g[b.corner_obj]
        b.Y = torch.where(b.pinned[:, None], b.X, b.Y)
        F = hx.gauss_deformation((X - V)[b.cells], b.hc)
        b.C_prev = hx.mat3_tn(F, F)
        b.pairs = contact.detect(b, X, V)

    def test_scene_energy_is_the_sum_of_body_energies(self):
        cfg = TrainConfig(scene_cells=250, body_sides=(3, 5), pinned_body_fraction=0.5)
        grids, aug = GridCache("cpu"), Augmenter("cpu")
        scene = scenes_v5.sample_scene(11, 0, cfg)
        scene = dataclasses.replace(scene, bodies=scene.bodies[:3])
        self.assertEqual(len(scene.bodies), 3)
        gs, X, V, material, cs = scenes_v5.realise(scene, grids, aug, "cpu", torch.float64)
        moduli = [m["mu"] for m in material.si]
        self.assertGreater(max(moduli) / min(moduli), 1.5)  # different stiffnesses
        self.assertTrue(torch.equal(material.mu_norm, material.mu_norm[:1].expand(3)))
        mu_ref = reference_modulus([b.material for b in scene.bodies])
        self.assertAlmostEqual(float(material.mu_norm[0]), mu_ref, delta=1e-6 * mu_ref)
        b = Batch.build(gs, "cpu", torch.float64)
        b.material, b.scene = material, cs
        self.step_start(b, X, V)
        E_scene = physics.energy(b, X + 0.01 * V) * energy_scale(material)  # joules per body
        self.assertTrue(torch.isfinite(E_scene).all() and (E_scene > 0).all())
        # body mode: each body alone with its own unit and the same plane
        E_bodies = []
        for o, (g, body) in enumerate(zip(gs, scene.bodies, strict=True)):
            rows = b.corner_obj == o
            one = Batch.build([g], "cpu", torch.float64)
            m = material_from_si(
                **body.material,
                gravity=scene.gravity,
                h=scene.h,
                dt=scene.dt,
                cell_count=g.C,
                sample_count=g.S,
                kappa=float(cs_kappa := scene.contact["kappa"]),
                beta=float(scene.contact["beta"]),
                mu_f=float(scene.contact["mu_f"]),
                friction_epsilon=float(scene.contact["friction_epsilon"]),
                floor_scale=float(scene.contact["floor_scale"]),
                dtype=torch.float64,
            )
            self.assertEqual(float(m.mu_scale[0]), 1.0)
            self.assertEqual(float(m.kappa[0]), float(cs_kappa))
            one.material = m
            one.scene = world_scene(1, 0.0, torch.float64, "cpu")
            self.step_start(one, X[rows], V[rows])
            self.assertEqual(one.pairs.count, int((b.pairs.obj == o).sum()))
            E_bodies.append(float(physics.energy(one, (X + 0.01 * V)[rows])[0] * energy_scale(m)[0]))
        for o in range(3):
            self.assertAlmostEqual(float(E_scene[o]), E_bodies[o], delta=1e-9 * E_bodies[o])
        self.assertAlmostEqual(float(E_scene.sum()), sum(E_bodies), delta=1e-9 * sum(E_bodies))
        # the floor scales with the energy, the force unit with the energy unit
        floors = torch.tensor([d["floor"] for d in material.si], dtype=torch.float64)
        self.assertTrue(torch.allclose(material.floor * energy_scale(material), floors))
        self.assertTrue(torch.allclose(force_scale(material) * material.h, energy_scale(material)))

    def test_conditioning_channels_unchanged(self):
        cfg = TrainConfig(scene_cells=400, body_sides=(3, 5))
        scene = scenes_v5.sample_scene(3, 1, cfg)
        grids, aug = GridCache("cpu"), Augmenter("cpu")
        gs, _X, _V, material, _cs = scenes_v5.realise(scene, grids, aug, "cpu")
        cond = conditioning(material)
        self.assertEqual(tuple(cond.shape), (len(gs), 7))
        for o, (g, body) in enumerate(zip(gs, scene.bodies, strict=True)):
            own = material_from_si(
                **body.material,
                gravity=scene.gravity,
                h=scene.h,
                dt=scene.dt,
                cell_count=g.C,
                sample_count=g.S,
                kappa=float(scene.contact["kappa"]),
                beta=float(scene.contact["beta"]),
                mu_f=float(scene.contact["mu_f"]),
                friction_epsilon=float(scene.contact["friction_epsilon"]),
                floor_scale=float(scene.contact["floor_scale"]),
            )
            self.assertTrue(torch.equal(cond[o], conditioning(own)[0]))  # bitwise
            for f in ("lam", "rho", "eta", "g", "mu_f", "kappa", "beta", "friction_eps", "h", "dt", "mu"):
                self.assertTrue(torch.equal(getattr(material, f)[o : o + 1], getattr(own, f)), f)
            # ke, kd, floor and the unit: the body's values rescaled by mu / mu_ref
            ratio = float(material.mu_scale[o])
            self.assertAlmostEqual(ratio, float(own.mu[0]) / float(material.mu_norm[o]), places=6)
            for f in ("ke", "kd", "floor"):
                common, alone = float(getattr(material, f)[o]), float(getattr(own, f)[0])
                self.assertAlmostEqual(common / ratio, alone, delta=1e-6 * abs(alone), msg=f)
        # body mode is bitwise what it was: no mu_ref gives mu_scale 1 and mu_norm = mu
        self.assertTrue(torch.equal(own.mu_scale, torch.ones(1)) and torch.equal(own.mu_norm, own.mu))
        self.assertEqual(own.si[0]["energy_scale"], own.si[0]["mu"] * scene.h**3)


class TestRunnerFailureTail(unittest.TestCase):
    """Finding 2: a failed last scene left a non-finite batch that the trainer's masked mean could not mask; the
    idle batch is now finite, the loss on it finite and its gradients zero."""

    def test_failed_last_scene_leaves_a_finite_loss(self):
        cfg = scene_cfg(scene_count=1)
        runner, step = make_runner_cpu(cfg)
        runner.start_epoch(1)
        out = step.query(runner.batch)
        out.E_after = out.E_after.clone()
        out.E_after[0] = float("nan")
        out.cand_after = out.cand_after.detach().clone()
        out.cand_after[runner.batch.corner_obj == 0] = float("nan")  # a real failure: the candidate blew up too
        runner.commit(out)  # the scene fails, the queue is empty: the rank idles on this batch for the whole epoch
        self.assertEqual(len(runner.failures), 1)
        self.assertIsNone(runner.job)
        self.assertFalse(bool(runner.batch.active.any()))
        batch = runner.batch
        self.assertTrue(torch.isfinite(batch.x).all() and torch.isfinite(batch.E).all())
        self.assertTrue(torch.equal(batch.x, batch.X))
        # the trainer's loss on the idle updates (train.py), in both the masked-fill and the product form
        for _ in range(2):
            out = step.query(batch)
            loss_vec = local_objective(out.E_after, out.E_before, batch.material.floor, cfg.energy_increase_weight)
            self.assertTrue(torch.isfinite(loss_vec).all())
            mask = batch.active & torch.isfinite(loss_vec)
            product = ((loss_vec * mask).sum() / mask.sum().clamp_min(1)).detach()
            self.assertTrue(bool(torch.isfinite(product)), f"loss {float(product)} on the idle batch after a failure")
            loss = loss_vec.masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)
            self.assertEqual(float(loss), 0.0)
            loss.backward()
            # no NaN reaches the parameters (DDP would all-reduce them): the gradients are finite and zero
            for name, p in step.net.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertEqual(float(p.grad.abs().max()), 0.0, name)
            step.net.zero_grad(set_to_none=True)
            runner.commit(out)
        self.assertEqual(runner.idle_updates, 2)


if __name__ == "__main__":
    unittest.main()
