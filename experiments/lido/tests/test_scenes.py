# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene sampler: shell and rejection rules, kappa load floor, JSON spec, SI -> normalised conversion, offsets."""

import json
import unittest

import torch

from experiments.lido import contact, scenes
from experiments.lido.batch import Batch
from experiments.lido.config import TrainConfig
from experiments.lido.grid import Grid
from experiments.lido.units import material_from_si

H = 0.025


def material_si(grid, **mat):
    m = {
        "E": 1e5,
        "nu": 0.3,
        "rho": 1000.0,
        "eta": 100.0,
        "gravity": (0, -9.81, 0),
        "h": H,
        "dt": 1 / 300,
        "cell_count": grid.C,
        "sample_count": grid.S,
    }
    m.update(mat)
    return material_from_si(**m).si[0]


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = TrainConfig(contact_plane_probability=1.0)
        self.grid = Grid.build((4, 4, 8))

    def sample(self, seed, cfg=None, **mat):
        rng = torch.Generator().manual_seed(seed)
        return scenes.sample_contact_spec(rng, cfg or self.cfg, material_si(self.grid, **mat), self.grid)

    def test_points_in_shell_outside_body(self):
        nx, ny, nz = self.grid.cell_counts
        total = 0
        for seed in range(6):
            spec = self.sample(seed)
            pts = torch.tensor(spec["points"], dtype=torch.float64).reshape(-1, 3) / H
            nrm = torch.tensor(spec["normals"], dtype=torch.float64).reshape(-1, 3)
            rad = torch.tensor(spec["radii"], dtype=torch.float64) / H
            total += pts.shape[0]
            if pts.shape[0] == 0:
                continue
            lo = torch.tensor([-scenes.SHELL_XZ / H, -scenes.SHELL_Y / H, -scenes.SHELL_XZ / H], dtype=torch.float64)
            hi = torch.tensor([nx + scenes.SHELL_XZ / H, ny, nz + scenes.SHELL_XZ / H], dtype=torch.float64)
            self.assertTrue(((pts > lo) & (pts < hi)).all())
            grown = torch.tensor([nx + 1, ny + 1, nz + 1], dtype=torch.float64)
            self.assertFalse(((pts > -1.0) & (pts < grown)).all(1).any())
            self.assertTrue((pts[:, 2] >= 1.0).all())
            self.assertTrue(torch.allclose(nrm.norm(dim=-1), torch.ones(pts.shape[0], dtype=torch.float64)))
            centre = torch.tensor([nx, ny, nz], dtype=torch.float64) / 2
            self.assertTrue(((nrm * (centre - pts)).sum(-1) > 0).all())
            lo_r, hi_r = self.cfg.contact_point_radius_range
            self.assertTrue(((rad >= lo_r) & (rad <= hi_r)).all())
            # no rest sample is a detection candidate at rest
            b = Batch.build([self.grid], "cpu", torch.float64)
            b.scene = scenes.scene_from_spec(spec, H, "cpu", torch.float64)
            X = b.rest.to(torch.float64)
            p = contact.detect(b, X, torch.zeros_like(X))
            self.assertEqual(int((p.kind == 1).sum()), 0)
        self.assertGreater(total, 20)

    def test_kappa_floor(self):
        heavy = TrainConfig(contact_kappa_range=(10.0, 20.0))
        grid = Grid.build((10, 10, 40))
        rng = torch.Generator().manual_seed(5)
        spec = scenes.sample_contact_spec(rng, heavy, material_si(grid, E=1e3, rho=1e4), grid)
        expected = 1e4 * grid.C * H**3 * 9.81 / (400 * 0.5 * 0.5 * H)
        self.assertAlmostEqual(spec["ke_floor"], expected, delta=1e-9 * expected)
        self.assertAlmostEqual(spec["ke_floor"], 2452.5, delta=0.1)
        self.assertTrue(spec["floor_bound"])
        self.assertEqual(spec["ke"], spec["ke_floor"])
        self.assertAlmostEqual(spec["kappa"], spec["ke"] / (1e3 * H), delta=1e-12)
        self.assertLess(spec["kappa_drawn"], spec["kappa"])
        m = material_from_si(
            E=1e3,
            nu=0.3,
            rho=1e4,
            eta=100.0,
            gravity=(0, -9.81, 0),
            h=H,
            dt=1 / 300,
            cell_count=grid.C,
            sample_count=grid.S,
            kappa=spec["kappa"],
        )
        self.assertAlmostEqual(m.si[0]["ke"], spec["ke"], delta=1e-9 * spec["ke"])
        light = scenes.sample_contact_spec(
            torch.Generator().manual_seed(5), heavy, material_si(grid, E=1e5, rho=1e3), grid
        )
        self.assertFalse(light["floor_bound"])
        self.assertAlmostEqual(light["kappa"], light["kappa_drawn"])
        off = TrainConfig(contact_kappa_range=(10.0, 20.0), contact_static_penetration_max=None)
        spec_off = scenes.sample_contact_spec(
            torch.Generator().manual_seed(5), off, material_si(grid, E=1e3, rho=1e4), grid
        )
        self.assertEqual(spec_off["ke_floor"], 0.0)
        self.assertFalse(spec_off["floor_bound"])

    def test_spec_json_ranges_and_disabled(self):
        spec = self.sample(11)
        json.dumps(spec)
        self.assertTrue(spec["plane_present"])
        lo, hi = self.cfg.contact_plane_height_range
        self.assertTrue(lo <= spec["plane_height"] <= hi)
        self.assertTrue(0.0 <= spec["beta"] <= 1.0 and 0.0 <= spec["mu_f"] <= 1.0)
        self.assertEqual(spec, self.sample(11))
        self.assertEqual(
            scenes.sample_contact_spec(
                torch.Generator().manual_seed(1), TrainConfig(contact=False), material_si(self.grid), self.grid
            ),
            {},
        )
        no_plane = self.sample(11, TrainConfig(contact_plane_probability=0.0))
        self.assertFalse(no_plane["plane_present"])

    def test_scene_from_spec_and_offsets(self):
        spec_a = {
            "plane_present": True,
            "plane_height": -0.1,
            "points": [[0.05, -0.2, 0.3], [0.1, 0.1, 0.1]],
            "normals": [[0, 1, 0], [1, 0, 0]],
            "radii": [0.0125, 0.05],
        }
        spec_c = {
            "plane_present": False,
            "plane_height": -0.02,
            "points": [[0.0, 0.0, 0.5]] * 3,
            "normals": [[0, 0, -1]] * 3,
            "radii": [0.1] * 3,
        }
        s = scenes.scenes_for_objects([spec_a, {}, spec_c], [0.025, 0.05, 0.1], "cpu")
        self.assertTrue(torch.equal(s.point_offsets, torch.tensor([0, 2, 2, 5])))
        self.assertTrue(torch.equal(s.plane_present, torch.tensor([True, False, False])))
        self.assertTrue(torch.allclose(s.plane_d, torch.tensor([-4.0, 0.0, -0.2])))
        self.assertTrue(torch.allclose(s.plane_n, torch.tensor([[0.0, 1.0, 0.0]]).expand(3, 3)))
        self.assertTrue(torch.allclose(s.points[:2], torch.tensor(spec_a["points"]) / 0.025))
        self.assertTrue(torch.allclose(s.points[2:], torch.tensor(spec_c["points"]) / 0.1))
        self.assertTrue(torch.allclose(s.radii, torch.tensor([0.5, 2.0, 1.0, 1.0, 1.0])))
        self.assertEqual(s.points.shape, (5, 3))
        one = scenes.scene_from_spec(spec_a, 0.025, "cpu", torch.float64)
        self.assertEqual(one.points.dtype, torch.float64)
        self.assertTrue(torch.equal(one.point_offsets, torch.tensor([0, 2])))
        empty = scenes.scene_from_spec({}, 0.025, "cpu")
        self.assertEqual(empty.points.shape, (0, 3))
        self.assertFalse(bool(empty.plane_present[0]))


if __name__ == "__main__":
    unittest.main()
