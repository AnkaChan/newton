# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from experiments.lido.augment import Augmenter
from experiments.lido.config import TrainConfig
from experiments.lido.fusion import Fusion
from experiments.lido.grid import GridCache
from experiments.lido.network import Net
from experiments.lido.runner import JobRunner
from experiments.lido.step import Step


def tiny_cfg(**kw):
    d = {
        "cell_counts": (2, 2, 3),
        "batch_size": 3,
        "state_count": 7,
        "budget_cap": 16,
        "growth_stages": ((2, 3), (4, 4)),
        "growth_stage_epochs": 1,
        "max_epochs": 2,
        "device": "cpu",
        "validation_count": 2,
        "validation_full_count": 1,
        "validation_full_steps": 2,
        "validation_iterations": 2,
        "validation_full_iterations": 2,
        "contact": True,
        "log_every": 1,
    }
    d.update(kw)
    return TrainConfig.from_dict(d)


class TestRunner(unittest.TestCase):
    def test_epoch_runs_to_completion(self):
        cfg = tiny_cfg()
        grids = GridCache("cpu")
        aug = Augmenter("cpu")
        step = Step(Net.from_config(cfg), Fusion(), aug)
        runner = JobRunner(cfg, step, aug, grids, 0, 1, "cpu", 5)
        runner.start_epoch(1)
        self.assertGreater(runner.U, 0)
        served = 0
        for _ in range(runner.U):
            out = step.query(runner.batch)
            served += int(runner.batch.active.sum())
            runner.commit(out)
        self.assertEqual(
            served,
            sum(j.K * j.H for j in __import__("experiments.lido.jobs", fromlist=["x"]).sample_epoch_jobs(5, 1, cfg)),
        )
        self.assertEqual(runner.loaded_jobs, cfg.state_count)
        self.assertFalse(runner.batch.active.any())
        self.assertEqual(runner.failures, [])

    def test_failure_reloads_slot(self):
        cfg = tiny_cfg()
        grids = GridCache("cpu")
        aug = Augmenter("cpu")
        step = Step(Net.from_config(cfg), Fusion(), aug)
        runner = JobRunner(cfg, step, aug, grids, 0, 1, "cpu", 5)
        runner.start_epoch(1)
        out = step.query(runner.batch)
        out.E_after = out.E_after.clone()
        out.E_after[1] = float("nan")
        runner.commit(out)
        self.assertEqual(len(runner.failures), 1)
        self.assertEqual(runner.resets, 1)
        self.assertTrue(torch.isfinite(runner.batch.E).all())


if __name__ == "__main__":
    unittest.main()
