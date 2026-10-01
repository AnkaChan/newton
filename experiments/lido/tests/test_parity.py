# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parity with the old package through the black-box oracle adapter (design spec 1b "Parity gate" (a)):
energy, gradient, modes, frames, fusion, projected gradient, node/edge features and the v4 network forward."""

import os
import unittest

import numpy as np
import torch

from experiments.lido import hex as hx
from experiments.lido import physics
from experiments.lido.batch import Batch
from experiments.lido.features import features
from experiments.lido.frames import frames
from experiments.lido.fusion import Fusion
from experiments.lido.grid import Grid, reference_rotation
from experiments.lido.network import Net
from experiments.lido.tests import oracle
from experiments.lido.units import material_from_si

CC = (2, 2, 3)
H, DT, E_MOD, NU, RHO, ETA = 0.025, 1.0 / 300.0, 1e5, 0.3, 1000.0, 100.0
GRAVITY = (0.0, -9.81, 0.0)


def si_state(seed=0):
    rng = np.random.default_rng(seed)
    rest, cells, pinned = oracle.rest_grid(CC, H)
    X = rest + 0.08 * H * rng.standard_normal(rest.shape)
    X_prev = rest + 0.05 * H * rng.standard_normal(rest.shape)
    V = 0.3 * H / DT * rng.standard_normal(rest.shape) * 0.1
    X[pinned] = rest[pinned]
    X_prev[pinned] = rest[pinned]
    V[pinned] = 0
    Y = X_prev + DT * V + DT * DT * np.array(GRAVITY)
    Y[pinned] = rest[pinned]
    return rest, cells, pinned, X, Y, X_prev


def new_batch(X, Y, X_prev, dtype=torch.float64):
    g = Grid.build(CC)
    b = Batch.build([g], "cpu", dtype)
    b.material = material_from_si(
        E=E_MOD, nu=NU, rho=RHO, eta=ETA, gravity=GRAVITY, h=H, dt=DT, cell_count=g.C, sample_count=g.S
    )
    for f in ("lam", "rho", "eta", "g", "ke", "kd", "mu_f", "kappa", "beta", "friction_eps", "floor", "h", "dt", "mu"):
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    t = lambda a: torch.as_tensor(a / H, dtype=dtype)  # noqa: E731
    b.X, b.Y, b.X_prev = t(X), t(Y), t(X_prev)
    b.x = b.X.clone()
    F_prev = hx.gauss_deformation(b.X_prev[b.cells], b.hc)
    b.C_prev = F_prev.transpose(-1, -2) @ F_prev
    b.m_Y = hx.modes(b.Y[b.cells], b.hc)
    b.m_prev = hx.modes(b.X_prev[b.cells], b.hc)
    b.R_ref = reference_rotation(b.X, b.ref_corners)
    return g, b


class TestParity(unittest.TestCase):
    def setUp(self):
        self.rest, self.cells, self.pinned, self.X, self.Y, self.X_prev = si_state()
        self.grid, self.b = new_batch(self.X, self.Y, self.X_prev)
        self.scale = oracle.energy_unit(H, E_MOD, NU)

    def test_conventions_match(self):
        conv = oracle.conventions()
        self.assertEqual(conv["corner_order"], [tuple(int(v) for v in row) for row in hx.XI_CORNERS.tolist()])
        self.assertTrue(np.array_equal(self.cells, self.grid.cells.numpy()))
        self.assertTrue(np.allclose(self.rest / H, self.grid.rest.numpy()))
        self.assertTrue(np.array_equal(self.pinned, self.grid.pinned_mask.numpy()))
        self.assertEqual(conv["reference_corners_2x2x3"], self.grid.ref_corners.tolist())

    def test_energy_and_gradient(self):
        E_ref, g_ref = oracle.energy_and_grad(CC, H, DT, self.X, self.Y, self.X_prev, E_MOD, NU, RHO, ETA)
        E, gX = physics.energy_and_grad(self.b, self.b.x)
        self.assertAlmostEqual(E.item() * self.scale, E_ref, delta=1e-6 * abs(E_ref))  # old constants are float32
        free = ~self.pinned
        g_si = gX.numpy() * self.b.material.mu.item() * H**2
        self.assertLess(np.abs(g_si[free] - g_ref[free]).max(), 1e-6 * np.abs(g_ref[free]).max())

    def test_modes(self):
        m_ref = oracle.modes(CC, H, self.X)
        m = hx.modes(self.b.X[self.b.cells], self.b.hc).numpy()
        self.assertLess(np.abs(m - m_ref).max(), 1e-12)

    def test_frames(self):
        R_ref = oracle.frames(CC, H, self.X)
        _, F_c = physics.modes_and_center(self.b.X, self.b)
        R = frames(F_c.float(), self.b.R_ref.float()[self.b.cell_obj]).double().numpy()
        self.assertLess(np.abs(R - R_ref).max(), 1e-5)

    def test_fuse(self):
        rng = np.random.default_rng(1)
        dm = 0.05 * rng.standard_normal((self.grid.C, 7, 3))
        d_ref = oracle.fuse(CC, H, self.X, dm)
        d = Fusion().fuse(self.b, hx.modes_to_gauss(torch.as_tensor(dm), self.b.hc)).numpy() * H
        self.assertLess(np.abs(d - d_ref).max(), 1e-12 * max(1.0, np.abs(d_ref).max() / 1e-3))

    def test_project_gradient(self):
        _, g_ref = oracle.energy_and_grad(CC, H, DT, self.X, self.Y, self.X_prev, E_MOD, NU, RHO, ETA)
        p_ref = oracle.project_gradient(CC, H, self.X, g_ref, energy_unit=self.scale)
        _, gX = physics.energy_and_grad(self.b, self.b.x)
        p = Fusion().project_gradient(self.b, gX).numpy()
        self.assertLess(np.abs(p - p_ref).max(), 1e-6 * np.abs(p_ref).max())

    def test_features_and_network_forward(self):
        inputs = oracle.network_inputs(CC, H, DT, self.X, self.Y, self.X_prev, E_MOD, NU, RHO, ETA, gravity=GRAVITY)
        g, b = new_batch(self.X, self.Y, self.X_prev, torch.float32)
        _, b.gX = physics.energy_and_grad(b, b.x)
        m_c, F_c = physics.modes_and_center(b.x, b)
        R = frames(F_c, b.R_ref[b.cell_obj])
        g_m = Fusion().project_gradient(b, b.gX)
        f = features(b, b.x, R, m_c, F_c, g_m)
        node_ref = inputs["node_features"]
        self.assertEqual(f.node.shape, (g.C, 142))
        self.assertLess(np.abs(f.node.numpy() - node_ref[:, :142]).max(), 2e-4)
        self.assertLess(np.abs(f.cond.numpy()[0] - inputs["conditioning"]).max(), 1e-5)
        # edge features: map (src, dst) to the old slot layout (slot 0 self, then lexicographic offsets)
        offsets = [(a, b_, c) for a in (-1, 0, 1) for b_ in (-1, 0, 1) for c in (-1, 0, 1) if (a, b_, c) != (0, 0, 0)]
        slot_of = {o: i + 1 for i, o in enumerate(offsets)}
        slot_of[(0, 0, 0)] = 0
        edge_ref = inputs["edge_features"]  # [C,27,24]
        src, dst = b.edges
        rest_off = (g.edge_rest).round().to(torch.int64).tolist()
        maxerr = 0.0
        for e in range(src.numel()):
            s = slot_of[tuple(rest_off[e])]
            maxerr = max(maxerr, float(np.abs(f.edge_attr[e].numpy() - edge_ref[dst[e], s]).max()))
        self.assertLess(maxerr, 2e-4)
        if not os.path.exists(str(oracle.V4_CHECKPOINT)):
            self.skipTest("v4 checkpoint")
        out_ref = oracle.network_forward(oracle.V4_CHECKPOINT, CC, inputs)
        net = Net()
        state = torch.load(str(oracle.V4_CHECKPOINT), map_location="cpu", weights_only=False)["network_state"]
        net.load_state_dict({k: v for k, v in state.items() if not k.startswith("neighbor_")})
        with torch.no_grad():
            out = net(f, b.edges, b.edge_offsets, b.cell_obj)
        self.assertLess(np.abs(out.corr.numpy() - out_ref["correction"]).max(), 2e-4)
        self.assertLess(np.abs(out.step.numpy() - out_ref["step_size"]).max(), 2e-5)


if __name__ == "__main__":
    unittest.main()
