# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""v5 scene regime (design spec section 11): job sampling and U with `scene_count`, the SceneRunner serving an
epoch of small scenes on the CPU (exactly sum K H queries), idle updates on a lighter rank, failure handling."""

import os
import socket
import tempfile
import unittest

import torch

from experiments.lido import jobs as J
from experiments.lido import scenes_v5 as S
from experiments.lido.augment import Augmenter
from experiments.lido.config import TrainConfig
from experiments.lido.fusion import Fusion
from experiments.lido.grid import GridCache
from experiments.lido.network import Net
from experiments.lido.runner import JobRunner, SceneRunner, make_runner
from experiments.lido.step import Step

MASTER = 5


def scene_cfg(**kw):
    d = {
        "scene_mode": "v5",
        "scene_curriculum": False,  # the tests describe the final mix; the curriculum has its own tests
        "scene_cells": 1500,
        "scene_count": 3,
        "budget_cap": 16,
        "growth_stages": ((2, 3), (4, 4)),
        "growth_stage_epochs": 1,
        "max_epochs": 2,
        "device": "cpu",
        "validation_scene_count": 2,
        "validation_full_scene_count": 1,
        "validation_iterations": 2,
        "validation_full_iterations": 2,
        "validation_full_steps": 2,
        "hidden_dim": 24,
        "edge_hidden_dim": 12,
        "num_heads": 2,
        "contact_hidden_dim": 8,
    }
    d.update(kw)
    return TrainConfig.from_dict(d)


def make_runner_cpu(cfg, rank=0, world=1):
    grids = GridCache("cpu")
    aug = Augmenter("cpu")
    step = Step(Net.from_config(cfg), Fusion(), aug)
    return SceneRunner(cfg, step, aug, grids, rank, world, "cpu", MASTER), step


class TestJobsV5(unittest.TestCase):
    def test_scene_jobs_and_U(self):
        cfg = scene_cfg(scene_count=5)
        jobs = J.sample_epoch_jobs(MASTER, 1, cfg)
        self.assertEqual([j.seed for j in jobs], [0, 1, 2, 3, 4])
        _, K_max, H_max = J.growth_stage(1, cfg)
        for j in jobs:
            self.assertTrue(1 <= j.H <= H_max and j.K <= K_max and j.K * j.H <= cfg.budget_cap)
        # the same seed stream as the body regime: the first scene_count draws coincide
        body = J.sample_epoch_jobs(MASTER, 1, scene_cfg(scene_mode="body", state_count=7, scene_count=5))
        self.assertEqual(len(body), 7)
        self.assertEqual(jobs, body[:5])
        self.assertEqual(J.job_count(cfg), 5)
        # one scene per rank at a time: U is the heaviest rank's sum of K H
        queues, U = J.assign(jobs, 1, 1)
        self.assertEqual(U, sum(j.K * j.H for j in jobs))
        queues, U = J.assign(jobs, 2, 1)
        loads = [sum(j.K * j.H for j in q) for q in queues]
        self.assertEqual(U, max(loads))
        self.assertEqual(sorted(j.seed for q in queues for j in q), [0, 1, 2, 3, 4])
        # the default configuration's regime (docstring table): 64 scenes, stage 0
        default = TrainConfig(scene_mode="v5")
        jobs = J.sample_epoch_jobs(73, 1, default)
        self.assertEqual(len(jobs), 64)
        self.assertTrue(all(j.K == 1 and 1 <= j.H <= 8 for j in jobs))
        _, U = J.assign(jobs, 4, 1)
        self.assertTrue(60 <= U <= 100, U)


def _ddp_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    """One gloo rank of `TestIdleRankUnderDDP`: an epoch of two scenes over two ranks, rank 1's scene fails at the
    first update, so it idles through the rest of the epoch while rank 0 trains; every update's gradients are
    checked finite on both ranks after the all-reduce."""
    import torch.distributed as dist

    from experiments.lido.validation import local_objective

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}")
    try:
        cfg = scene_cfg(scene_count=2)
        grids, aug = GridCache("cpu"), Augmenter("cpu")
        torch.manual_seed(0)
        net = Net.from_config(cfg)
        model = torch.nn.parallel.DistributedDataParallel(net, broadcast_buffers=False)
        step = Step(model, Fusion(), aug)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        runner = SceneRunner(cfg, step, aug, grids, rank, world, "cpu", MASTER)
        runner.start_epoch(1)
        finite, idle, losses = [], 0, []
        for update in range(runner.U):
            b = runner.batch
            out = step.query(b)
            if rank == 1 and update == 0:  # a real failure: non-finite energy (graph kept) and candidate
                out.E_after = out.E_after.clone()
                out.E_after[0] = float("nan")
                out.cand_after = out.cand_after.detach().clone()
                out.cand_after[b.corner_obj == 0] = float("nan")
            loss_vec = local_objective(out.E_after, out.E_before, b.material.floor, cfg.energy_increase_weight)
            mask = b.active & torch.isfinite(loss_vec)
            loss = loss_vec.masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            finite.append(bool(torch.isfinite(gn)))
            if torch.isfinite(gn):  # train.py skips the step of a non-finite gradient
                opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            runner.commit(out)
            idle += int(runner.job is None)
        torch.save(
            {"finite": finite, "idle": idle, "failures": len(runner.failures), "U": runner.U, "losses": losses},
            os.path.join(out_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class TestIdleRankUnderDDP(unittest.TestCase):
    """Two gloo ranks on the CPU: the rank whose only scene fails idles on a finite batch; its zero gradients enter
    the all-reduce and both ranks keep finite gradients and losses for the rest of the epoch (review finding 2).
    (The failing update itself: with a NaN produced inside the energy its gradient reaches the parameters through
    the masked loss on both ranks via the all-reduce and train.py skips that one optimizer step on the non-finite
    gradient norm; the test's injection overwrites the energy after the fact, which cuts that path.)"""

    def test_two_ranks_one_failing(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as out_dir:
            torch.multiprocessing.spawn(_ddp_worker, args=(2, port, out_dir), nprocs=2, join=True)
            results = [torch.load(os.path.join(out_dir, f"rank{r}.pt")) for r in range(2)]
        for r, res in enumerate(results):
            self.assertGreater(res["U"], 2)
            self.assertEqual(res["finite"][1:], [True] * (res["U"] - 1), f"rank {r}: non-finite gradient")
            self.assertTrue(all(v == v for v in res["losses"]), f"rank {r}: NaN loss")
        self.assertEqual(results[1]["failures"], 1)
        self.assertEqual(results[0]["failures"], 0)
        self.assertEqual(results[1]["idle"], results[1]["U"])  # the job is gone from the failing commit on
        self.assertTrue(all(v == 0.0 for v in results[1]["losses"][1:]))
        self.assertGreater(results[0]["losses"][0], 0.0)


class TestSceneRunner(unittest.TestCase):
    def test_epoch_serves_every_query(self):
        cfg = scene_cfg()
        runner, step = make_runner_cpu(cfg)
        runner.start_epoch(1)
        jobs = J.sample_epoch_jobs(MASTER, 1, cfg)
        self.assertEqual(runner.U, sum(j.K * j.H for j in jobs))
        b = runner.batch
        self.assertTrue(b.body_contact and b.any_free)
        self.assertEqual(b.free_objects.tolist(), [not body.pinned for body in runner.scene.bodies])
        self.assertTrue(bool((b.origin == 0).all()) and bool(b.scene.plane_present.all()))
        self.assertGreaterEqual(b.C, cfg.scene_cells)
        self.assertEqual(runner.active_count, b.O)
        served, batches = 0, set()
        for _ in range(runner.U):
            self.assertIsNotNone(runner.job)
            batches.add(id(runner.batch))
            out = step.query(runner.batch)
            served += 1
            runner.commit(out)
        self.assertEqual(served, sum(j.K * j.H for j in jobs))
        self.assertEqual(runner.loaded_jobs, cfg.scene_count)
        self.assertEqual(runner.queries, sum(s["bodies"] * s["K"] * s["H"] for s in runner.scene_summaries))
        self.assertIsNone(runner.job)  # the queue ran dry exactly at U
        self.assertEqual(runner.active_count, 0)
        self.assertFalse(bool(runner.batch.active.any()))
        self.assertEqual(runner.failures, [])
        self.assertEqual(runner.idle_updates, 0)
        self.assertEqual(len(batches), cfg.scene_count)  # a fresh Batch per scene
        # one detection at every load and after every advance
        self.assertEqual(len(runner.pair_history), sum(1 + j.H for j in jobs))
        summary = runner.epoch_summary()
        self.assertEqual(summary["scenes"], cfg.scene_count)
        self.assertEqual(
            [s["seed"] for s in summary["scenes_served"]],
            [S.epoch_scene_seed(1, j.seed) for j in J.assign(jobs, 1, 1)[0][0]],  # fresh scenes every epoch
        )
        for key in ("pairs_mean", "body_pairs_mean", "body_pairs_max", "plane_pairs_mean", "steps_with_body_pairs"):
            self.assertIn(key, summary)
        # a further update on the idle runner is harmless and counted
        runner.commit(step.query(runner.batch))
        self.assertEqual(runner.idle_updates, 1)

    def test_lighter_rank_idles_through_U(self):
        cfg = scene_cfg(scene_count=3)
        jobs = J.sample_epoch_jobs(MASTER, 1, cfg)
        queues, U = J.assign(jobs, 2, 1)
        loads = [sum(j.K * j.H for j in q) for q in queues]
        light = loads.index(min(loads))
        self.assertLess(loads[light], U)
        runner, step = make_runner_cpu(cfg, rank=light, world=2)
        runner.start_epoch(1)
        self.assertEqual(runner.U, U)
        served = 0
        for _ in range(U):
            served += int(runner.job is not None)
            runner.commit(step.query(runner.batch))
        self.assertEqual(served, loads[light])
        self.assertEqual(runner.idle_updates, U - loads[light])
        self.assertEqual(runner.loaded_jobs, len(queues[light]))

    def test_failure_loads_the_next_scene(self):
        cfg = scene_cfg()
        runner, step = make_runner_cpu(cfg)
        runner.start_epoch(1)
        first = runner.job
        out = step.query(runner.batch)
        out.E_after = out.E_after.clone()
        out.E_after[0] = float("nan")
        runner.commit(out)
        self.assertEqual(len(runner.failures), 1)
        f = runner.failures[0]
        self.assertEqual(
            (f.seed, f.K, f.H, f.k, f.h, f.kind, f.epoch, f.update),
            (first.seed, first.K, first.H, 1, 0, "non_finite", 1, 1),
        )
        self.assertEqual(runner.resets, 1)
        self.assertEqual(runner.loaded_jobs, 2)
        self.assertIsNotNone(runner.job)
        self.assertNotEqual(runner.job, first)
        self.assertTrue(torch.isfinite(runner.batch.E).all())
        self.assertEqual((runner.k, runner.h), (0, 0))

    def test_make_runner_and_world_check(self):
        cfg = scene_cfg()
        grids, aug = GridCache("cpu"), Augmenter("cpu")
        step = Step(Net.from_config(cfg), Fusion(), aug)
        self.assertIsInstance(make_runner(cfg, step, aug, grids, 0, 1, "cpu", MASTER), SceneRunner)
        body = scene_cfg(scene_mode="body", cell_counts=(2, 2, 3), batch_size=2, state_count=3)
        self.assertIsInstance(make_runner(body, step, aug, grids, 0, 1, "cpu", MASTER), JobRunner)
        with self.assertRaises(ValueError):
            make_runner(scene_cfg(scene_mode="other"), step, aug, grids, 0, 1, "cpu", MASTER)
        runner = SceneRunner(scene_cfg(scene_count=2), step, aug, grids, 0, 3, "cpu", MASTER)
        with self.assertRaises(ValueError):
            runner.start_epoch(1)


if __name__ == "__main__":
    unittest.main()


class TestSceneCurriculumRunner(unittest.TestCase):
    def test_fresh_scenes_every_epoch_with_the_epoch_mix(self):
        cfg = scene_cfg(scene_count=2, scene_curriculum=True, scene_curriculum_epochs=(1, 3), max_epochs=3)
        runner, _ = make_runner_cpu(cfg)
        runner.start_epoch(1)
        first = runner.scene
        self.assertEqual(first.seed, S.epoch_scene_seed(1, runner.job.seed))
        self.assertTrue(all(b.pinned for b in first.bodies))  # epoch 1: everything pinned
        self.assertEqual(runner.epoch_summary()["pinned_fraction"], 1.0)
        self.assertEqual(runner.epoch_summary()["resting_fraction"], 0.0)
        runner.start_epoch(3)
        third = runner.scene
        self.assertEqual(third.seed, S.epoch_scene_seed(3, runner.job.seed))
        self.assertNotEqual([b.cell_counts for b in third.bodies], [b.cell_counts for b in first.bodies])
        self.assertEqual(runner.mix, S.scene_mix(cfg, 3))
        self.assertEqual(runner.mix.pinned_fraction, cfg.pinned_body_fraction)
        self.assertEqual(runner.epoch_summary()["pinned_fraction"], cfg.pinned_body_fraction)
