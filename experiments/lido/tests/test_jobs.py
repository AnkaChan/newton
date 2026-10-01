# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math
import unittest

from experiments.lido import jobs as J
from experiments.lido.config import TrainConfig
from experiments.lido.structs import Job, SceneSpec


class TestGrowthAndJobs(unittest.TestCase):
    def setUp(self):
        self.cfg = TrainConfig()

    def test_growth_stage_table(self):
        expected = {
            1: (0, 1, 8),
            2: (0, 1, 8),
            3: (1, 2, 16),
            4: (1, 2, 16),
            5: (2, 4, 32),
            6: (2, 4, 32),
            7: (3, 8, 64),
            8: (3, 8, 64),
            9: (4, 16, 128),
            10: (4, 16, 128),
            11: (5, 32, 128),
            12: (5, 32, 128),
            13: (5, 32, 128),
            48: (5, 32, 128),
        }
        for epoch, want in expected.items():
            self.assertEqual(J.growth_stage(epoch, self.cfg), want, epoch)

    def test_jobs_respect_caps(self):
        for epoch in (1, 3, 6, 9, 12):
            _, K_max, H_max = J.growth_stage(epoch, self.cfg)
            jobs = J.sample_epoch_jobs(73, epoch, self.cfg)
            self.assertEqual(len(jobs), self.cfg.state_count)
            self.assertEqual([j.seed for j in jobs], list(range(self.cfg.state_count)))
            for j in jobs:
                self.assertIsInstance(j, Job)
                self.assertLessEqual(j.K * j.H, self.cfg.budget_cap)
                self.assertLessEqual(j.K, K_max)
                self.assertTrue(1 <= j.H <= H_max)
                self.assertIn(j.K, self.cfg.iteration_counts)
        jobs = J.sample_epoch_jobs(73, 12, self.cfg)
        self.assertEqual(max(j.K for j in jobs), 32)
        self.assertEqual(max(j.H for j in jobs), 128)
        self.assertTrue(any(j.K * j.H == 2048 for j in jobs))

    def test_jobs_deterministic(self):
        a = J.sample_epoch_jobs(73, 5, self.cfg)
        b = J.sample_epoch_jobs(73, 5, self.cfg)
        self.assertEqual(a, b)
        self.assertNotEqual(a, J.sample_epoch_jobs(73, 6, self.cfg))
        self.assertNotEqual(a, J.sample_epoch_jobs(74, 5, self.cfg))

    def test_assign_balance_and_U(self):
        jobs = J.sample_epoch_jobs(73, 12, self.cfg)
        queues, U = J.assign(jobs, 4, 16)
        self.assertEqual(len(queues), 4)
        self.assertEqual(sorted(j.seed for q in queues for j in q), list(range(len(jobs))))
        loads = [sum(j.K * j.H for j in q) for q in queues]
        self.assertLess(max(loads) / min(loads), 1.05)
        longest = max(j.K * j.H for j in jobs)
        self.assertEqual(U, max(math.ceil(max(loads) / 16), longest))
        for q in queues:  # LPT order: non-increasing K H
            costs = [j.K * j.H for j in q]
            self.assertEqual(costs, sorted(costs, reverse=True))
        # a longest job may exceed the ceiling on tiny configurations
        small = [Job(0, 8, 128), Job(1, 1, 1), Job(2, 1, 1)]
        queues, U = J.assign(small, 2, 16)
        self.assertEqual(U, 1024)
        self.assertEqual([len(q) for q in queues], [1, 2])


class TestSceneSpec(unittest.TestCase):
    def setUp(self):
        self.cfg = TrainConfig(contact=False)

    def check_ranges(self, s, cfg):
        lo, hi = cfg.youngs_modulus_range
        self.assertTrue(lo <= s.E <= hi)
        lo, hi = cfg.poissons_ratio_range
        self.assertTrue(lo <= s.nu <= hi)
        lo, hi = cfg.density_range
        self.assertTrue(lo <= s.rho <= hi)
        lo, hi = cfg.damping_range
        self.assertTrue(lo <= s.eta <= hi)
        lo, hi = cfg.gravity_magnitude_range
        g = math.sqrt(sum(v * v for v in s.gravity))
        self.assertTrue(lo <= g <= hi)
        self.assertAlmostEqual(s.gravity[0], 0.0)
        self.assertAlmostEqual(s.gravity[2], 0.0)
        self.assertLess(s.gravity[1], 0.0)
        lo, hi = cfg.perturbation_scale_range
        self.assertTrue(lo <= s.perturbation_scale <= hi)
        lo, hi = cfg.strength_range
        self.assertTrue(lo <= s.strength <= hi)
        lo, hi = cfg.velocity_dt_range
        self.assertTrue(lo <= s.velocity_dt <= hi)
        self.assertEqual(s.cell_counts, tuple(cfg.cell_counts))
        self.assertEqual((s.h, s.dt, s.pins), (cfg.cell_size, cfg.time_step, cfg.pins))

    def test_ranges_and_log_uniform_spread(self):
        specs = [J.sample_scene_spec(73, i, self.cfg) for i in range(256)]
        for i, s in enumerate(specs):
            self.assertEqual(s.seed, i)
            self.check_ranges(s, self.cfg)
        # log-uniform: about a third of the E draws fall in each decade of [1e3, 1e6]
        decades = [sum(1 for s in specs if 10**k <= s.E < 10 ** (k + 1)) for k in (3, 4, 5)]
        for n in decades:
            self.assertTrue(50 <= n <= 120, decades)

    def test_deterministic_and_epoch_independent(self):
        a = J.sample_scene_spec(73, 17, self.cfg)
        self.assertEqual(a, J.sample_scene_spec(73, 17, self.cfg))
        self.assertNotEqual(a, J.sample_scene_spec(73, 18, self.cfg))
        self.assertNotEqual(a, J.sample_scene_spec(74, 17, self.cfg))
        jobs1 = J.sample_epoch_jobs(73, 1, self.cfg)[:64]
        jobs9 = J.sample_epoch_jobs(73, 9, self.cfg)[:64]
        specs1 = J.sample_scene_specs(jobs1, 73, 1, self.cfg)
        specs9 = J.sample_scene_specs(jobs9, 73, 9, self.cfg)
        self.assertEqual(set(specs1), set(range(64)))
        self.assertEqual(specs1, specs9)
        self.assertEqual(specs1[17], a)

    def test_validation_seeds_disjoint(self):
        train = [J.sample_scene_spec(73, i, self.cfg) for i in range(128)]
        val = [J.sample_scene_spec(73, i, self.cfg, validation=True) for i in range(64)]
        train_E = {s.E for s in train}
        for i, v in enumerate(val):
            self.assertEqual(v.seed, i)
            self.check_ranges(v, self.cfg)
            self.assertNotIn(v.E, train_E)
            self.assertNotEqual(v, train[i])
        self.assertEqual(val[3], J.sample_scene_spec(73, 3, self.cfg, validation=True))

    def test_json_serialisable(self):
        s = J.sample_scene_spec(73, 5, self.cfg)
        d = dataclasses.asdict(s)
        text = json.dumps(d)
        back = SceneSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(text).items()})
        self.assertEqual(back, s)
        self.assertEqual(s.contact, {})

    def test_contact_flag(self):
        cfg = TrainConfig(contact=True)
        s = J.sample_scene_spec(73, 5, cfg)
        self.assertIsInstance(s.contact, dict)
        # the material draws do not depend on the contact flag
        off = J.sample_scene_spec(73, 5, self.cfg)
        self.assertEqual(
            (s.E, s.nu, s.rho, s.eta, s.gravity, s.perturbation_scale, s.strength, s.velocity_dt),
            (off.E, off.nu, off.rho, off.eta, off.gravity, off.perturbation_scale, off.strength, off.velocity_dt),
        )


if __name__ == "__main__":
    unittest.main()
