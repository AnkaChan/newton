# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loss scale decisions of 2026-10-02: physical energy floor, bounded increase term, blow-up guard."""

import unittest

import torch

from experiments.lido.augment import Augmenter
from experiments.lido.config import TrainConfig
from experiments.lido.fusion import Fusion
from experiments.lido.grid import GridCache
from experiments.lido.network import Net
from experiments.lido.runner import JobRunner
from experiments.lido.step import Step
from experiments.lido.units import EPS32, material_from_si
from experiments.lido.validation import local_objective


class TestLossScale(unittest.TestCase):
    def test_physical_floor_is_the_weight_times_one_cell(self):
        kw = {
            "E": 1e5,
            "nu": 0.3,
            "rho": 1000.0,
            "eta": 100.0,
            "gravity": (0, -9.81, 0),
            "h": 0.025,
            "dt": 1 / 300,
            "cell_count": 4000,
            "sample_count": 1800,
        }
        m_round = material_from_si(physical_floor=False, **kw)
        m_phys = material_from_si(physical_floor=True, **kw)
        weight_h = 1000.0 * 4000 * 0.025**3 * 9.81 * 0.025  # rho V |g| h in joules
        self.assertAlmostEqual(m_phys.si[0]["floor"], weight_h, delta=1e-9 * weight_h)
        self.assertGreater(m_phys.floor.item(), 1e3 * m_round.floor.item())
        self.assertLess(
            m_round.si[0]["floor"], 10 * EPS32 * 4000 * 0.025**3 * (1e6 + 1e5 / (1 / 300) + 1000 * 0.025**2 * 300**2)
        )

    def test_bounded_increase_grows_logarithmically(self):
        E_before = torch.tensor([1.0, 1.0, 1.0])
        E_after = torch.tensor([1.0, 100.0, 1e5])
        floor = torch.tensor([1.0, 1.0, 1.0])
        linear = local_objective(E_after, E_before, floor, bounded=False)
        bounded = local_objective(E_after, E_before, floor, bounded=True)
        self.assertAlmostEqual(linear[0].item(), bounded[0].item(), places=6)  # same zero point
        self.assertGreater(linear[2].item(), 1e5 - 10)
        self.assertLess(bounded[2].item(), 30)
        self.assertTrue((bounded[1:] > bounded[0]).all())  # still penalises increases, monotonically
        # decreases are rewarded identically in both forms
        dec = local_objective(torch.tensor([0.1]), torch.tensor([1.0]), torch.tensor([1.0]), bounded=True)
        dec_lin = local_objective(torch.tensor([0.1]), torch.tensor([1.0]), torch.tensor([1.0]), bounded=False)
        self.assertAlmostEqual(dec.item(), dec_lin.item(), places=6)

    def test_blowup_guard_reloads_the_slot(self):
        cfg = TrainConfig.from_dict(
            {
                "cell_counts": (2, 2, 3),
                "batch_size": 2,
                "state_count": 4,
                "budget_cap": 16,
                "growth_stages": ((2, 3),),
                "growth_stage_epochs": 1,
                "max_epochs": 1,
                "device": "cpu",
                "validation_count": 1,
                "validation_full_count": 1,
                "validation_full_steps": 1,
                "validation_iterations": 1,
                "validation_full_iterations": 1,
                "contact": False,
                "blowup_energy_factor": 1e3,
            }
        )
        grids = GridCache("cpu")
        aug = Augmenter("cpu")
        step = Step(Net.from_config(cfg), Fusion(), aug)
        runner = JobRunner(cfg, step, aug, grids, 0, 1, "cpu", 5)
        runner.start_epoch(1)
        out = step.query(runner.batch)
        out.E_after = out.E_after.clone()
        out.E_after[0] = 1e9 * runner.batch.material.floor[0]  # finite but diverged
        runner.commit(out)
        self.assertEqual(runner.resets, 1)
        self.assertTrue(torch.isfinite(runner.batch.E).all())
        self.assertTrue((runner.batch.E <= 1e3 * runner.batch.material.floor).all())


if __name__ == "__main__":
    unittest.main()


class TestStepCapCurriculum(unittest.TestCase):
    def test_cap_schedule(self):
        from experiments.lido.train import step_cap

        cfg = TrainConfig.from_dict({"max_step_size": 0.05, "step_cap_start": 0.1, "step_cap_ramp_epochs": 8})
        self.assertAlmostEqual(step_cap(cfg, 0), 0.005)
        self.assertAlmostEqual(step_cap(cfg, 4), 0.05 * (0.1 + 0.9 * 0.5))
        self.assertAlmostEqual(step_cap(cfg, 8), 0.05)
        self.assertAlmostEqual(step_cap(cfg, 20), 0.05)
        self.assertAlmostEqual(step_cap(TrainConfig.from_dict({"step_cap_ramp_epochs": 0}), 0), 0.05)

    def test_network_step_bounded_by_cap(self):
        from experiments.lido.features import EDGE_WIDTH, NODE_WIDTH, TOKEN_WIDTH
        from experiments.lido.structs import Features

        net = Net()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.5 * torch.randn_like(p))
        C = 12
        dst = torch.arange(C).repeat_interleave(3)
        src = (dst + torch.tensor([0, 1, -1]).repeat(C)) % C
        f = Features(
            node=torch.randn(C, NODE_WIDTH),
            edge_attr=torch.randn(3 * C, EDGE_WIDTH),
            cond=torch.randn(1, 7),
            tokens=torch.zeros(0, TOKEN_WIDTH),
            token_offsets=torch.zeros(C + 1, dtype=torch.long),
        )
        for cap in (0.005, 0.02, 0.05):
            net.step_cap.fill_(cap)
            out = net(f, torch.stack([src, dst]), torch.arange(0, 3 * C + 1, 3), torch.zeros(C, dtype=torch.long))
            self.assertTrue((out.step <= cap + 1e-7).all() and (out.step > 0).all())
        self.assertNotIn("step_cap", net.state_dict())  # non-persistent: old checkpoints load unchanged
