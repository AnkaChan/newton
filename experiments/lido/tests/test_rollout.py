# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""rollout(): documented keys and shapes, npz round trip with the renderer's keys, fresh net = free fall."""

import os
import tempfile
import unittest

import numpy as np
import torch

from experiments.lido.rollout import CONTACT_KEYS, NPZ_KEYS, load_network, rollout, save_npz

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SMALL = {"hidden_dim": 24, "edge_hidden_dim": 12, "num_heads": 2, "contact_hidden_dim": 8}
SCENARIO = {
    "cell_counts": (2, 2, 3),
    "h": 0.05,
    "dt": 1.0 / 300.0,
    "E": 1e5,
    "nu": 0.3,
    "rho": 1000.0,
    "eta": 50.0,
    "gravity": (0.0, -9.81, 0.0),
}


class TestRollout(unittest.TestCase):
    def test_keys_shapes_and_free_fall(self):
        steps, K = 3, 2
        r = rollout(None, SMALL, SCENARIO, steps, K, device=DEVICE)
        P, C = 3 * 3 * 4, 2 * 2 * 3
        for k in NPZ_KEYS:
            self.assertIn(k, r)
        self.assertEqual(r["positions"].shape, (steps + 1, P, 3))
        self.assertEqual(r["velocities"].shape, (steps + 1, P, 3))
        self.assertEqual(r["times"].shape, (steps + 1,))
        self.assertEqual(r["rest_positions"].shape, (P, 3))
        self.assertEqual(r["cell_corner_indices"].shape, (C, 8))
        self.assertEqual(r["fixed_indices"].shape, (9,))
        for k in ("energy_joule", "residual_n", "penetration_r"):
            self.assertEqual(r[k].shape, (steps,))
            self.assertTrue(np.isfinite(r[k]).all())
        self.assertTrue(np.isfinite(r["positions"]).all())
        self.assertEqual(int(r["iterations"]), K)
        self.assertEqual(tuple(r["cell_counts"]), (2, 2, 3))
        self.assertTrue(np.isfinite(r["seconds_per_query"]))
        self.assertEqual(r["config"]["hidden_dim"], 24)
        # fresh network: the candidate stays at Y, so the free corners free-fall and the pins stay
        fixed = r["fixed_indices"]
        free = np.setdiff1d(np.arange(P), fixed)
        self.assertTrue((r["positions"][:, fixed] == r["positions"][0, fixed]).all())
        dt, g = SCENARIO["dt"], np.array(SCENARIO["gravity"])
        for s in range(steps):
            expected = r["positions"][s] + dt * r["velocities"][s] + dt**2 * g
            self.assertLess(np.abs(r["positions"][s + 1, free] - expected[free]).max(), 1e-6)
        self.assertTrue((r["penetration_r"] == 0).all())
        self.assertFalse(any(k in r for k in CONTACT_KEYS))

    def test_npz_round_trip_with_contact(self):
        scenario = dict(
            SCENARIO,
            contact={
                "plane_present": True,
                "plane_height": -0.02,
                "kappa": 100.0,
                "beta": 0.1,
                "mu_f": 0.3,
                "points": [[0.05, -0.03, 0.1]],
                "normals": [[0.0, 1.0, 0.0]],
                "radii": [0.03],
            },
        )
        r = rollout(None, SMALL, scenario, 2, 1, device=DEVICE)
        for k in CONTACT_KEYS:
            self.assertIn(k, r)
        self.assertEqual(r["contact_plane_present"].dtype, np.bool_)
        self.assertEqual(r["contact_plane_point"].shape, (3,))
        self.assertAlmostEqual(float(r["contact_plane_point"][1]), -0.02, places=6)
        self.assertEqual(r["contact_point_positions"].shape, (1, 3))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "traj.npz")
            save_npz(r, path)
            with np.load(path) as data:
                files = set(data.files)
                renderer_required = {"positions", "times", "rest_positions", "fixed_indices", "cell_corner_indices"}
                self.assertTrue(renderer_required <= files)
                self.assertTrue(set(CONTACT_KEYS) <= files)
                self.assertTrue(np.array_equal(data["positions"], r["positions"]))
                self.assertEqual(data["times"].dtype, np.float64)
                self.assertEqual(data["cell_corner_indices"].dtype, np.int64)
                self.assertEqual(float(data["h"]), SCENARIO["h"])

    def test_load_network_fresh_and_round_trip(self):
        net, cfg = load_network(None, overrides=SMALL)
        self.assertEqual(cfg.hidden_dim, 24)
        ck = {"network_state": net.state_dict(), "config": cfg.to_dict(), "epoch": 3}
        net2, cfg2 = load_network(ck)
        self.assertEqual(cfg2.to_dict(), cfg.to_dict())
        for a, b in zip(net.state_dict().values(), net2.state_dict().values(), strict=True):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
