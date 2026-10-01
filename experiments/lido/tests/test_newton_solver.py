# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverLIDO inside Newton: pins prescribed from particle_q, untrained net leaves the candidate at Y, grouping,
ground-plane and static-point conversion, GPU query timing on the canonical beam."""

import time
import unittest

import numpy as np
import torch
import warp as wp

import newton
from experiments.lido.batch import Batch
from experiments.lido.config import TrainConfig
from experiments.lido.fusion import Fusion
from experiments.lido.grid import Grid
from experiments.lido.network import Net
from experiments.lido.newton_solver import SolverLIDO, add_hex_body, attach_bodies, ground_plane_from_model
from experiments.lido.step import Step
from experiments.lido.units import material_from_si

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SMALL_CFG = TrainConfig(hidden_dim=24, edge_hidden_dim=12, num_heads=2, contact_hidden_dim=8)
MATERIAL = {"E": 1e5, "nu": 0.3, "rho": 1000.0, "eta": 50.0}
H = 0.05
DT = 1.0 / 300.0
PLANE_HEIGHT = -0.1


def build_model(cell_counts=((2, 2, 3), (1, 2, 2)), plane=True, contact=None, device=DEVICE):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    for i, cc in enumerate(cell_counts):
        add_hex_body(builder, cc, H, pos=(0.5 * i, 0.3, 0.0), material=MATERIAL, contact=contact, key=f"body{i}")
    if plane:
        builder.add_ground_plane(height=PLANE_HEIGHT)
    model = builder.finalize(device=device)
    attach_bodies(model, builder)
    return model


def to_np(a: wp.array) -> np.ndarray:
    return a.numpy().copy()


class TestAddHexBody(unittest.TestCase):
    def test_particles_and_registry(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        rot = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # 90 degrees about z
        idx = add_hex_body(builder, (2, 2, 3), H, pos=(1.0, 2.0, 3.0), rot=rot, material=MATERIAL)
        self.assertEqual(idx, 0)
        body = builder.lido_bodies[0]
        grid = Grid.build((2, 2, 3))
        self.assertEqual((body.particle_start, body.particle_count), (0, grid.P))
        q = np.asarray(builder.particle_q, dtype=np.float64)
        rest = grid.rest.numpy() * H
        expected = np.array([1.0, 2.0, 3.0]) + np.stack([-rest[:, 1], rest[:, 0], rest[:, 2]], -1)
        self.assertLess(np.abs(q - expected).max(), 1e-6)
        mass = np.asarray(builder.particle_mass)
        self.assertTrue((mass[grid.pinned_mask.numpy()] == 0).all())
        self.assertTrue((mass[~grid.pinned_mask.numpy()] > 0).all())
        lumped = grid.mass.numpy()  # cells sharing the corner / 8, in rho h^3 units; pinned corners are zeroed
        self.assertAlmostEqual(mass.sum(), MATERIAL["rho"] * H**3 * lumped[~grid.pinned_mask.numpy()].sum(), places=9)
        model = builder.finalize(device="cpu")
        attach_bodies(model, builder)
        self.assertEqual(len(model.lido_bodies), 1)


class TestSolverLIDO(unittest.TestCase):
    def setUp(self):
        wp.init()
        torch.manual_seed(0)

    def run_steps(self, model, solver, steps):
        s0, s1 = model.state(), model.state()
        q_in = to_np(s0.particle_q)
        g = model.gravity.numpy()[-1]
        inv_mass = to_np(model.particle_inv_mass)
        pinned = inv_mass == 0
        for k in range(steps):
            q_before, qd_before = to_np(s0.particle_q), to_np(s0.particle_qd)
            solver.step(s0, s1, None, None, DT)
            q_after, qd_after = to_np(s1.particle_q), to_np(s1.particle_qd)
            self.assertTrue(np.isfinite(q_after).all() and np.isfinite(qd_after).all())
            self.assertTrue((q_after[pinned] == q_before[pinned]).all(), f"pins moved at step {k}")
            self.assertTrue((qd_after[pinned] == qd_before[pinned]).all())
            if k == 0:
                expected = q_before + DT * qd_before + DT**2 * g
                self.assertLess(np.abs(q_after[~pinned] - expected[~pinned]).max(), 1e-6)
                self.assertLess(np.abs(qd_after[~pinned] - (qd_before + DT * g)[~pinned]).max(), 1e-5)
            s0, s1 = s1, s0
        q_final = to_np(s0.particle_q)
        self.assertTrue((q_final[~pinned, 1] < q_in[~pinned, 1]).all(), "free corners did not fall")
        return s0, s1

    def test_two_bodies_three_steps(self):
        model = build_model()
        solver = SolverLIDO(model, None, iterations=2, cfg=SMALL_CFG)
        self.assertEqual(len(solver.groups), 2)
        self.assertEqual(solver.batch.O, 2)
        self.run_steps(model, solver, 3)

    def test_equal_grids_share_a_group(self):
        model = build_model(cell_counts=((2, 2, 3), (2, 2, 3)))
        solver = SolverLIDO(model, None, iterations=1, cfg=SMALL_CFG)
        self.assertEqual(len(solver.groups), 1)
        self.assertEqual(solver.groups[0].objects.numel(), 2)
        self.run_steps(model, solver, 1)

    def test_prescribed_pins_follow_state_in(self):
        model = build_model(cell_counts=((2, 2, 3),))
        solver = SolverLIDO(model, None, iterations=1, cfg=SMALL_CFG)
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, DT)
        pinned = to_np(model.particle_inv_mass) == 0
        q = to_np(s1.particle_q)
        q[pinned, 0] += 0.01
        s1.particle_q.assign(q)
        solver.step(s1, s0, None, None, DT)
        q_out = to_np(s0.particle_q)
        self.assertTrue((q_out[pinned] == q[pinned]).all())
        self.assertTrue(np.isfinite(q_out).all())
        # the batch holds the new pin positions in cell units
        X_pin = solver.batch.X[solver.batch.pinned].cpu().numpy()
        expected = (q[pinned] - np.array(solver.bodies[0].origin)) / H
        self.assertLess(np.abs(X_pin - expected).max(), 1e-5)

    def test_ground_plane_and_static_points_conversion(self):
        model = build_model(contact={"kappa": 100.0, "beta": 0.0, "mu_f": 0.2})
        plane = ground_plane_from_model(model)
        self.assertIsNotNone(plane)
        point, normal = plane
        self.assertLess(np.abs(normal - np.array([0.0, 1.0, 0.0])).max(), 1e-6)
        self.assertAlmostEqual(float(point[1]), PLANE_HEIGHT, places=6)
        pts = {"points": [[0.1, 0.0, 0.05]], "normals": [[0.0, 1.0, 0.0]], "radii": [0.02]}
        solver = SolverLIDO(model, None, iterations=1, cfg=SMALL_CFG, static_points=pts)
        scene = solver.batch.scene
        self.assertTrue(scene.plane_present.all())
        for i, b in enumerate(solver.bodies):
            o = np.array(b.origin)
            self.assertAlmostEqual(scene.plane_d[i].item(), float(normal @ (point - o)) / H, places=5)
            self.assertLess(np.abs(scene.plane_n[i].cpu().numpy() - normal).max(), 1e-6)
            p = scene.points[scene.point_offsets[i] : scene.point_offsets[i + 1]].cpu().numpy()
            self.assertLess(np.abs(p - (np.array(pts["points"]) - o) / H).max(), 1e-5)
            self.assertAlmostEqual(scene.radii[i].item(), 0.02 / H, places=5)
        self.assertIsNone(solver.batch.material)  # built at the first step from its dt
        self.run_steps(model, solver, 2)
        self.assertTrue((solver.batch.material.ke > 0).all())
        self.assertEqual(solver.batch.material.count, 2)

    def test_ground_flag(self):
        model = build_model(plane=False)
        solver = SolverLIDO(model, None, iterations=1, cfg=SMALL_CFG)
        self.assertFalse(solver.batch.scene.plane_present.any())
        with self.assertRaises(ValueError):
            SolverLIDO(model, None, iterations=1, cfg=SMALL_CFG, ground=True)
        model = build_model(plane=True)
        solver = SolverLIDO(model, None, iterations=1, cfg=SMALL_CFG, ground=False)
        self.assertFalse(solver.batch.scene.plane_present.any())

    def test_reinitialises_from_modified_state(self):
        model = build_model(cell_counts=((2, 2, 3),))
        solver = SolverLIDO(model, None, iterations=1, cfg=SMALL_CFG)
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, DT)
        q = to_np(s1.particle_q)
        free = to_np(model.particle_inv_mass) > 0
        s1.particle_qd.assign(np.zeros_like(q))  # user reset of the velocities
        solver.step(s1, s0, None, None, DT)
        g = model.gravity.numpy()[-1]
        q_out = to_np(s0.particle_q)
        self.assertLess(np.abs(q_out[free] - (q + DT**2 * g)[free]).max(), 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "needs cuda")
class TestQueryTiming(unittest.TestCase):
    def test_canonical_beam_query_time(self):
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        cfg = TrainConfig()
        net = Net.from_config(cfg).to(device).eval()
        with torch.no_grad():
            for p in net.parameters():
                torch.nn.init.normal_(p, std=0.02)
        grid = Grid.build(cfg.cell_counts, cfg.pins, device)
        batch = Batch.build([grid], device)
        batch.material = material_from_si(
            E=1e5,
            nu=0.3,
            rho=1000.0,
            eta=100.0,
            gravity=cfg.gravity,
            h=cfg.cell_size,
            dt=cfg.time_step,
            cell_count=grid.C,
            sample_count=grid.S,
            device=device,
        )
        batch.X = grid.rest.clone()
        batch.V = torch.zeros_like(batch.X)
        batch.X_prev = batch.X.clone()
        batch.x = batch.X.clone()
        step = Step(net, Fusion())
        sel = torch.ones(1, dtype=torch.bool, device=device)
        K, n_timed = 8, 20
        with torch.no_grad():
            step.prepare(batch, sel)
            for _ in range(3):
                step.commit(batch, step.query(batch))
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(n_timed):
                step.commit(batch, step.query(batch))
            torch.cuda.synchronize(device)
        per_query = (time.perf_counter() - t0) / n_timed
        print(
            f"\ncanonical beam {cfg.cell_counts}: {per_query * 1e3:.2f} ms/query, "
            f"{per_query * K * 1e3:.1f} ms/step (K={K})"
        )
        self.assertTrue(torch.isfinite(batch.x).all())
        self.assertLess(per_query, 0.1)


if __name__ == "__main__":
    unittest.main()
