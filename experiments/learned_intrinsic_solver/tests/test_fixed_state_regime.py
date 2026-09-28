# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-state epoch regime: job sampling, rank assignment, pool job lists, trainer loop and weights-only starts."""

import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import threading
import unittest
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np

from experiments.learned_intrinsic_solver.train_mixed import (
    _ARCHITECTURE_FIELDS,
    JobAssignment,
    MixedTrainConfig,
    _fixed_state_counts,
    assign_jobs,
    fixed_state_stage,
    sample_epoch_jobs,
)

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253 -- Optional experimental training tests.

from experiments.learned_intrinsic_solver import launch_training as launcher_module
from experiments.learned_intrinsic_solver import train_epochs, train_mixed
from experiments.learned_intrinsic_solver.launch_training import launch_training
from experiments.learned_intrinsic_solver.train_mixed import run_training
from experiments.learned_intrinsic_solver.trajectory_pool import ActiveTrajectoryPool

_WORKER_RESULT = {"passed": True, "exit_codes": [0], "timed_out": False, "failure": None, "elapsed_seconds": 1.0}


def _is_power_of_two(value):
    return value >= 1 and value & (value - 1) == 0


class _Physics:
    """Payload stub recording every reset, advance and retirement by seed."""

    def __init__(self):
        self.resets, self.advances, self.retired = [], [], []
        self.lock = threading.Lock()

    def reset(self, seed):
        with self.lock:
            self.resets.append(seed)
        return {"seed": seed, "candidate": torch.tensor([0.0])}

    def advance(self, payload):
        with self.lock:
            self.advances.append(payload["seed"])
        return dict(payload)

    def retire(self, payload):
        with self.lock:
            self.retired.append(payload["seed"])


def _job_pool(jobs, *, batch_size=2, capacity=4, workers=1):
    physics = _Physics()
    pool = ActiveTrajectoryPool(
        capacity,
        batch_size,
        physics.reset,
        physics.advance,
        physics.retire,
        iteration_counts=(1,),
        physical_step_counts=(1,),
        seed=0,
        workers=workers,
        initialize=False,
    )
    pool.set_jobs(jobs)
    return pool, physics


def _drain(pool):
    """Run a job list to exhaustion; return per-record (seed, K, H, queries) and the batch sizes."""
    served = {}
    sizes = []
    while not pool.exhausted:
        records = pool.take_batch()
        sizes.append(len({record.id for record in records}))
        for record in records:
            entry = served.setdefault(record.id, [record.seed, record.iteration_budget, record.step_budget, 0])
            entry[3] += 1
            record.payload["candidate"] = record.payload["candidate"] + 1
        pool.finish_batch(records)
    return served, sizes


class TestJobSampling(unittest.TestCase):
    def test_sample_epoch_jobs_is_deterministic_and_seed_sensitive(self):
        """Draw one job per training state from a stream fixed by the master seed and the epoch."""
        config = MixedTrainConfig(regime="fixed_states", state_count=64)
        first = sample_epoch_jobs(config, 5, 73)
        self.assertEqual(first, sample_epoch_jobs(config, 5, 73))
        self.assertEqual([seed for seed, _, _ in first], list(range(64)))
        self.assertNotEqual(first, sample_epoch_jobs(config, 5, 74))
        self.assertNotEqual(first, sample_epoch_jobs(config, 6, 73))

    def test_sample_epoch_jobs_respects_the_stage_ranges_and_the_budget_cap(self):
        """Keep 1 <= H <= H_max, K a power of two <= K_max and K x H <= budget_cap; K = 1 stays available."""
        config = MixedTrainConfig(regime="fixed_states", state_count=512, budget_cap=128)
        for epoch in range(1, 15):
            stage, k_max, h_max = fixed_state_stage(config, epoch)
            jobs = sample_epoch_jobs(config, epoch, 73)
            with self.subTest(epoch=epoch, stage=stage):
                for _, iterations, steps in jobs:
                    self.assertTrue(1 <= steps <= h_max)
                    self.assertTrue(_is_power_of_two(iterations) and iterations <= k_max)
                    self.assertLessEqual(iterations * steps, config.budget_cap)
                self.assertIn(1, {steps for _, _, steps in jobs})
                self.assertIn(h_max, {steps for _, _, steps in jobs})
                if k_max > 1:
                    self.assertGreater(max(iterations for _, iterations, _ in jobs), 1)
        # A horizon beyond the cap still offers K = 1.
        tiny = MixedTrainConfig(regime="fixed_states", state_count=32, budget_cap=1, growth_stages=((4, 8),))
        self.assertTrue(all(iterations == 1 for _, iterations, _ in sample_epoch_jobs(tiny, 1, 1)))

    def test_stage_progression_follows_the_epoch_timetable(self):
        """Advance one stage every growth_stage_epochs epochs and hold the final stage."""
        config = MixedTrainConfig(regime="fixed_states", growth_stage_epochs=2)
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5]
        self.assertEqual([fixed_state_stage(config, epoch)[0] for epoch in range(1, 15)], expected)
        self.assertEqual(fixed_state_stage(config, 100), (5, 32, 128))
        self.assertEqual(_fixed_state_counts(config, 3), ((1, 2, 4, 8), (64,)))
        self.assertEqual(_fixed_state_counts(replace(config, budget_cap=4), 5), ((1, 2, 4), (128,)))


class TestJobAssignment(unittest.TestCase):
    def test_assign_jobs_balances_ranks_and_pads_every_rank_to_the_common_update_count(self):
        """Balance K x H loads within one job, keep every sampled job once and pad with own-seed fillers."""
        config = MixedTrainConfig(regime="fixed_states", state_count=256)
        jobs = sample_epoch_jobs(config, 9, 73)
        assignment = assign_jobs(jobs, 4, 16)
        self.assertIsInstance(assignment, JobAssignment)
        self.assertEqual(assignment, assign_jobs(list(reversed(jobs)), 4, 16))
        sampled = [
            rank[: len(rank) - fillers]
            for rank, fillers in zip(assignment.rank_jobs, assignment.filler_queries, strict=True)
        ]
        loads = [sum(k * h for _, k, h in rank) for rank in sampled]
        longest = max(k * h for _, k, h in jobs)
        self.assertLessEqual(max(loads) - min(loads), longest)
        self.assertEqual(sorted(job for rank in sampled for job in rank), sorted(jobs))
        self.assertEqual(assignment.queries, sum(k * h for _, k, h in jobs))
        self.assertEqual(assignment.updates, max(-(-max(loads) // 16), longest))
        for rank, fillers in zip(assignment.rank_jobs, assignment.filler_queries, strict=True):
            self.assertEqual(sum(k * h for _, k, h in rank), assignment.updates * 16)
            own = {seed for seed, _, _ in rank[: len(rank) - fillers]}
            self.assertTrue(all(job[1:] == (1, 1) and job[0] in own for job in rank[len(rank) - fillers :]))
            sizes = [k * h for _, k, h in rank[: len(rank) - fillers]]
            self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_assign_jobs_shuffles_each_rank_in_a_seeded_order_and_keeps_the_fillers_last(self):
        """Permute every rank's sampled jobs from SeedSequence([*shuffle_seed, rank, 7332]); counts and fillers are unchanged."""
        config = MixedTrainConfig(regime="fixed_states", state_count=256)
        jobs = sample_epoch_jobs(config, 11, 73)
        ordered = assign_jobs(jobs, 4, 16)
        shuffled = assign_jobs(jobs, 4, 16, shuffle_seed=(73, 11))
        self.assertEqual(shuffled, assign_jobs(list(reversed(jobs)), 4, 16, shuffle_seed=(73, 11)))
        self.assertNotEqual(shuffled.rank_jobs, assign_jobs(jobs, 4, 16, shuffle_seed=(73, 12)).rank_jobs)
        self.assertEqual(
            (shuffled.updates, shuffled.queries, shuffled.filler_queries),
            (ordered.updates, ordered.queries, ordered.filler_queries),
        )
        for rank, (before, after, fillers) in enumerate(
            zip(ordered.rank_jobs, shuffled.rank_jobs, shuffled.filler_queries, strict=True)
        ):
            with self.subTest(rank=rank):
                sampled_before, sampled_after = before[: len(before) - fillers], after[: len(after) - fillers]
                permutation = np.random.default_rng(np.random.SeedSequence([73, 11, rank, 7332])).permutation(
                    len(sampled_before)
                )
                self.assertEqual(list(sampled_after), [sampled_before[index] for index in permutation])
                self.assertEqual(sorted(sampled_after), sorted(sampled_before))
                self.assertEqual(after[len(after) - fillers :], before[len(before) - fillers :])
                # The epoch no longer runs from the longest budgets down to the shortest.
                sizes = [k * h for _, k, h in sampled_after]
                self.assertNotEqual(sizes, sorted(sizes, reverse=True))
        self.assertEqual(len({rank[:8] for rank in shuffled.rank_jobs}), 4)

    def test_assign_jobs_raises_the_update_count_to_the_longest_job(self):
        """A trajectory receives one query per update, so U is at least the longest K x H."""
        assignment = assign_jobs([(0, 8, 1), (1, 1, 1), (2, 1, 1)], 1, 2)
        self.assertEqual(assignment.updates, 8)
        self.assertEqual(assignment.filler_queries, (6,))
        self.assertEqual(assignment.rank_jobs[0][:3], ((0, 8, 1), (1, 1, 1), (2, 1, 1)))
        self.assertEqual(Counter(assignment.rank_jobs[0][3:]), Counter({(0, 1, 1): 2, (1, 1, 1): 2, (2, 1, 1): 2}))

    def test_assign_jobs_rejects_empty_ranks_and_bad_counts(self):
        """Every rank needs a seed of its own for the filler jobs."""
        with self.assertRaisesRegex(ValueError, "state_count"):
            assign_jobs([(0, 1, 1)], 2, 2)
        for world_size, batch_size in ((0, 2), (2, 0), (True, 2), (1, 1.5)):
            with self.subTest(world_size=world_size, batch_size=batch_size), self.assertRaises(ValueError):
                assign_jobs([(0, 1, 1), (1, 1, 1)], world_size, batch_size)


class TestPoolJobMode(unittest.TestCase):
    def test_pool_runs_exactly_the_listed_jobs_with_full_batches_until_exhaustion(self):
        """Run each job once with its own K and H, serve full batches to the last query and report exhaustion."""
        jobs = [(0, 2, 2), (1, 1, 1), (2, 1, 1), (3, 1, 2), (4, 1, 1), (0, 1, 1)]
        pool, physics = _job_pool(jobs, capacity=3)
        self.addCleanup(pool.close)
        self.assertEqual(pool.remaining_queries, 10)
        self.assertFalse(pool.exhausted)
        served, sizes = _drain(pool)
        self.assertEqual(sizes, [2] * 5)
        self.assertEqual(sorted((seed, k, h) for seed, k, h, _ in served.values()), sorted(jobs))
        self.assertTrue(all(queries == k * h for _, k, h, queries in served.values()))
        self.assertEqual(pool.remaining_queries, 0)
        self.assertTrue(pool.exhausted)
        self.assertEqual(pool.records, ())
        pool.quiesce()
        self.assertEqual(sorted(physics.resets), sorted(seed for seed, _, _ in jobs))
        self.assertEqual(sorted(physics.advances), [0, 3])
        self.assertEqual(sorted(physics.retired), sorted(seed for seed, _, _ in jobs))
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            pool.take_batch()

    def test_pool_completes_long_tails_that_fifo_rotation_would_strand(self):
        """Serve critical trajectories first so a long job among fillers still ends on a full batch."""
        for jobs in (
            [(0, 8, 1)] + [(seed, 1, 1) for seed in range(1, 9)],
            [(seed, 1, 1) for seed in range(1, 9)] + [(0, 8, 1)],
            [(0, 4, 1), (1, 4, 1), (2, 4, 1), (3, 1, 1), (4, 1, 1), (5, 1, 1), (6, 1, 1)],
            [(0, 2, 2), (1, 1, 4), (2, 1, 1), (3, 1, 1), (4, 1, 1), (5, 1, 1)],
        ):
            with self.subTest(jobs=jobs):
                pool, _ = _job_pool(jobs, capacity=4)
                self.addCleanup(pool.close)
                served, sizes = _drain(pool)
                total = sum(k * h for _, k, h in jobs)
                self.assertEqual(sizes, [2] * (total // 2))
                self.assertEqual(sorted((seed, k, h) for seed, k, h, _ in served.values()), sorted(jobs))
                self.assertTrue(all(queries == k * h for _, k, h, queries in served.values()))

    def test_pool_serves_random_feasible_job_lists_with_full_batches_only(self):
        """Every list whose total is a batch multiple and whose longest job fits in the update count completes.

        The order is immaterial: the trainer's lists (fillers last, sampled jobs
        shuffled) and fully shuffled lists both end on a full batch.
        """
        rng = random.Random(7)
        for trial in range(40):
            batch_size = rng.choice((2, 3, 4))
            jobs = [(seed, rng.choice((1, 2, 4, 8)), rng.randint(1, 6)) for seed in range(rng.randint(1, 12))]
            total = sum(k * h for _, k, h in jobs)
            longest = max(k * h for _, k, h in jobs)
            updates = max(-(-total // batch_size), longest)
            jobs += [(seed % len(jobs), 1, 1) for seed in range(updates * batch_size - total)]
            shuffled = list(jobs)
            rng.shuffle(shuffled)
            for order, listed in (("fillers last", jobs), ("shuffled", shuffled)):
                with self.subTest(trial=trial, batch_size=batch_size, order=order, jobs=listed):
                    pool, _ = _job_pool(listed, batch_size=batch_size, capacity=batch_size + 1, workers=2)
                    self.addCleanup(pool.close)
                    served, sizes = _drain(pool)
                    self.assertEqual(sizes, [batch_size] * updates)
                    self.assertEqual(sorted((seed, k, h) for seed, k, h, _ in served.values()), sorted(jobs))

    def test_set_jobs_validates_the_list_and_the_pool_state(self):
        """Reject unbalanced totals, over-long jobs, malformed jobs and live trajectories; no checkpoint in job mode."""
        physics = _Physics()
        empty = ActiveTrajectoryPool(
            3,
            2,
            physics.reset,
            physics.advance,
            iteration_counts=(1,),
            physical_step_counts=(1,),
            seed=0,
            initialize=False,
        )
        self.addCleanup(empty.close)
        self.assertIsNone(empty.remaining_queries)
        self.assertFalse(empty.exhausted)
        for jobs in ([(0, 1, 1), (1, 2, 1)], [(0, 4, 1), (1, 1, 1), (2, 1, 1)], [(0, 1)], [(0, 0, 1)], [(True, 1, 1)]):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                empty.set_jobs(jobs)
        self.assertEqual(physics.resets, [])
        empty.set_jobs([(0, 1, 1), (1, 1, 1)])
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            empty.state_dict()
        with self.assertRaisesRegex(ValueError, "live"):
            empty.set_jobs([(2, 1, 1), (3, 1, 1)])
        streaming = ActiveTrajectoryPool(
            3, 2, physics.reset, physics.advance, iteration_counts=(1,), physical_step_counts=(1,), seed=0
        )
        self.addCleanup(streaming.close)
        with self.assertRaisesRegex(ValueError, "live"):
            streaming.set_jobs([(0, 1, 1), (1, 1, 1)])
        self.assertIsNone(streaming.remaining_queries)
        streaming.state_dict()


class TestFixedStateTraining(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        """Tiny CPU problem: six fixed states, two growth stages of one epoch each, K x H <= 8."""
        values = {
            "regime": "fixed_states",
            "cell_counts": (2, 2, 3),
            "cell_size": 0.1,
            "hidden_dim": 8,
            "edge_hidden_dim": 4,
            "num_heads": 2,
            "batch_size": 2,
            "pool_multiplier": 2,
            "state_count": 6,
            "budget_cap": 8,
            "growth_stage_epochs": 1,
            "growth_stages": ((1, 2), (2, 4)),
            "max_epochs": 2,
            "validation_count": 2,
            "validation_iterations": 3,
            "validation_physical_steps": 2,
            "validation_physical_iterations": 2,
            "validation_full_count": 1,
            "validation_full_interval": 2,
            "device": "cpu",
            "cpu_threads": 1,
            "preparation_workers": 1,
            "verbose": False,
            "early_stopping": False,
        }
        values.update(overrides)
        return MixedTrainConfig(**values)

    def test_config_validation_rejects_bad_regime_settings(self):
        """Reject unknown regimes, non-power-of-two K_max, H_max above 128, decreasing stages and empty budgets."""
        for overrides in (
            {"regime": "fixed"},
            {"growth_stages": ((3, 8),)},
            {"growth_stages": ((1, 129),)},
            {"growth_stages": ((2, 16), (1, 32))},
            {"growth_stages": ((1, 16), (2, 8))},
            {"growth_stages": ()},
            {"growth_stages": ((1,),)},
            {"growth_stages": ((64, 8),)},
            {"budget_cap": 0},
            {"state_count": 0},
            {"growth_stage_epochs": 0},
        ):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                MixedTrainConfig(**overrides)
        # JSON lists become tuples and checkpoints predating the regime default to the pool.
        config = MixedTrainConfig(growth_stages=[[1, 8], [2, 16]])
        self.assertEqual(config.growth_stages, ((1, 8), (2, 16)))
        legacy = {name: value for name, value in asdict(MixedTrainConfig()).items() if name != "regime"}
        del legacy["growth_stages"]
        restored = MixedTrainConfig.from_checkpoint_config(legacy)
        self.assertEqual((restored.regime, restored.growth_stages), ("pool", MixedTrainConfig().growth_stages))

    def test_fixed_state_training_runs_the_growth_timetable_end_to_end(self):
        """Perform exactly U updates per epoch on the fixed states, validate every epoch and write checkpoints."""
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(train_mixed, "write_progress", wraps=train_mixed.write_progress) as heartbeat:
                report = run_training(output, config)
            expected = [
                assign_jobs(
                    sample_epoch_jobs(config, epoch, config.seed),
                    1,
                    config.batch_size,
                    shuffle_seed=(config.seed, epoch),
                )
                for epoch in (1, 2)
            ]
            # Every heartbeat carries the stage: before the first job list exists, during and after the run.
            phases = [(call.kwargs["phase"], call.kwargs.get("regime")) for call in heartbeat.call_args_list]
            self.assertEqual(phases[0], ("initializing", {"name": "fixed_states", "stage": 0, "k_max": 1, "h_max": 2}))
            self.assertEqual(phases[-1], ("complete", report["epochs"][-1]["regime"]))
            self.assertTrue(all(regime is not None for _, regime in phases))
            self.assertEqual(report["completed_epochs"], 2)
            self.assertEqual(report["completed_updates"], sum(a.updates for a in expected))
            self.assertNotIn("initialized_from", report)
            for row, assignment, epoch in zip(report["epochs"], expected, (1, 2), strict=True):
                stage, k_max, h_max = fixed_state_stage(config, epoch)
                self.assertEqual(stage, epoch - 1)
                self.assertEqual(
                    row["regime"],
                    {
                        "name": "fixed_states",
                        "stage": stage,
                        "k_max": k_max,
                        "h_max": h_max,
                        "queries": assignment.queries,
                        "filler_queries": list(assignment.filler_queries),
                        "updates": assignment.updates,
                    },
                )
                self.assertIsNone(row["curriculum"])
                self.assertFalse(row["allow_early_stop"])
                self.assertEqual(row["query_count"], assignment.updates * config.batch_size)
                self.assertEqual(sum(row["rank_0_budgets"].values()), row["query_count"])
                self.assertEqual(set(row["rank_0_budgets"]), {f"{k}/{h}" for _, k, h in assignment.rank_jobs[0]})
                self.assertEqual(tuple(row["available_K"]), _fixed_state_counts(config, stage)[0])
                self.assertEqual(tuple(row["available_H"]), (h_max,))
                self.assertIn("selection", row["validation"])
                self.assertIn("force_residual", row["validation"])
            self.assertIsNone(report["epochs"][0]["full_horizon_validation"])
            full = report["epochs"][1]["full_horizon_validation"]
            self.assertEqual((full["iterations"], full["physical_steps"], full["sample_count"]), (2, 4, 1))
            pool_stats = report["epochs"][-1]["rank_0_pool"]
            self.assertEqual(pool_stats["retired"], sum(len(a.rank_jobs[0]) for a in expected))
            self.assertEqual(pool_stats["inner_iterations"], report["completed_updates"] * config.batch_size)
            self.assertEqual(pool_stats["resets"], pool_stats["retired"])
            for name in ("initial.pt", "latest.pt", "final.pt"):
                self.assertTrue((output / "checkpoints" / name).is_file(), name)
            saved = torch.load(output / "checkpoints/latest.pt", weights_only=False)
            self.assertIsNone(saved["rank_states"][0]["pool"])
            self.assertEqual(saved["rank_states"][0]["context_specs"], {})
            self.assertEqual(saved["report"]["completed_epochs"], 2)
            if report["best_selection"] is not None:
                self.assertTrue((output / "checkpoints/best_validation.pt").is_file())
            with (output / "epochs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["regime_stage"] for row in rows], ["0", "1"])
            self.assertEqual([row["regime_updates"] for row in rows], [str(a.updates) for a in expected])
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual((progress["phase"], progress["completed_epochs"]), ("complete", 2))
            self.assertEqual(progress["available_H"], [4])
            self.assertEqual(progress["regime"], report["epochs"][-1]["regime"])

    def test_unserved_queries_fail_through_the_coordinated_gate_and_keep_the_stage_in_the_heartbeat(self):
        """Route a pool invariant violation through _all_ranks_ok like every other loop failure; the failed heartbeat keeps the regime."""
        config = self.config(max_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(ActiveTrajectoryPool, "exhausted", new_callable=PropertyMock, return_value=False),
                patch.object(train_epochs, "_all_ranks_ok", wraps=train_epochs._all_ranks_ok) as gate,
                self.assertRaisesRegex(RuntimeError, "unserved"),
            ):
                run_training(output, config)
            self.assertTrue(any("unserved" in str(call.args[0]) for call in gate.call_args_list))
            self.assertTrue((output / "failure.json").is_file())
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(
                (progress["phase"], progress["regime"]["stage"], progress["regime"]["k_max"]), ("failed", 0, 1)
            )

    def test_fixed_state_resume_restarts_the_next_epoch_from_a_fresh_job_list(self):
        """An ordinary resume continues the epoch counter and reproduces the uninterrupted run's job lists."""
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = run_training(root / "full", config)
            run_training(root / "split", replace(config, max_epochs=1))
            resumed = run_training(root / "split", config, resume=root / "split/checkpoints/latest.pt")
            self.assertEqual([row["epoch"] for row in resumed["epochs"]], [1, 2])
            self.assertEqual([row["regime"] for row in resumed["epochs"]], [row["regime"] for row in full["epochs"]])
            self.assertEqual(resumed["completed_updates"], full["completed_updates"])
            self.assertEqual(
                [row["rank_0_budgets"] for row in resumed["epochs"]], [row["rank_0_budgets"] for row in full["epochs"]]
            )
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                run_training(
                    root / "split", replace(config, regime="pool"), resume=root / "split/checkpoints/latest.pt"
                )

    def test_weights_only_initialization_from_a_pool_checkpoint(self):
        """Load only the network and AdamW state; the epoch counter, report and pool start fresh."""
        pool_config = self.config(
            regime="pool",
            queries_per_epoch=8,
            iteration_counts=(1, 2),
            physical_step_counts=(1, 2),
            stage_epochs=1,
            max_epochs=1,
        )
        fixed_config = self.config(youngs_modulus_range=(2e3, 5e5), contact_kappa_range=(0.5, 5.0), max_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_report = run_training(root / "pool", pool_config)
            checkpoint = root / "pool/checkpoints/latest.pt"
            source = torch.load(checkpoint, weights_only=False)
            for overrides in ({"hidden_dim": 16}, {"cell_counts": (1, 1, 2)}, {"contact": False}):
                with self.subTest(**overrides), self.assertRaisesRegex(ValueError, "architecture"):
                    run_training(
                        root / "rejected",
                        replace(fixed_config, **overrides),
                        resume=checkpoint,
                        resume_weights_only=True,
                    )
            self.assertFalse((root / "rejected").exists())
            # A parameter shape the architecture fields do not cover (a feature change without a schema
            # bump) is rejected as an architecture mismatch before any output is written.
            mangled_state = dict(source["network_state"])
            first = next(iter(mangled_state))
            mangled_state[first] = torch.zeros(mangled_state[first].shape[0] + 1, *mangled_state[first].shape[1:])
            del mangled_state[list(mangled_state)[-1]]
            torch.save(dict(source, network_state=mangled_state), root / "mangled.pt")
            with self.assertRaisesRegex(ValueError, f"architecture.*{re.escape(first)}"):
                run_training(root / "rejected", fixed_config, resume=root / "mangled.pt", resume_weights_only=True)
            self.assertFalse((root / "rejected").exists())
            with self.assertRaisesRegex(ValueError, "requires a checkpoint"):
                run_training(root / "rejected", fixed_config, resume_weights_only=True)
            report = run_training(root / "fixed", fixed_config, resume=checkpoint, resume_weights_only=True)
            self.assertEqual([row["epoch"] for row in report["epochs"]], [1])
            self.assertEqual(report["completed_epochs"], 1)
            self.assertEqual(report["epochs"][0]["regime"]["stage"], 0)
            origin = report["initialized_from"]
            self.assertEqual(origin["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(origin["sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(origin["completed_epochs"], 1)
            self.assertEqual(origin["completed_updates"], source_report["completed_updates"])
            self.assertEqual(origin["best_selection"], source_report["best_selection"])
            differences = origin["config_differences"]
            self.assertEqual(differences["youngs_modulus_range"], {"checkpoint": (1e3, 1e6), "current": (2e3, 5e5)})
            self.assertEqual(differences["regime"], {"checkpoint": "pool", "current": "fixed_states"})
            self.assertTrue(set(differences).isdisjoint(_ARCHITECTURE_FIELDS))
            # The fresh run's initial checkpoint carries the source weights and Adam moments before any update.
            initial = torch.load(root / "fixed/checkpoints/initial.pt", weights_only=False)
            self.assertEqual(initial["report"]["completed_epochs"], 0)
            self.assertEqual(initial["report"]["initialized_from"], origin)
            for key in source["network_state"]:
                torch.testing.assert_close(initial["network_state"][key], source["network_state"][key], rtol=0, atol=0)
            for identity, state in source["optimizer_state"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(
                        initial["optimizer_state"]["state"][identity][key], value, rtol=0, atol=0
                    )
            self.assertEqual(initial["config"]["regime"], "fixed_states")
            # The learning rate follows the new schedule from epoch 1.
            self.assertEqual(
                report["epochs"][0]["learning_rate"], train_mixed._scheduled_learning_rate(fixed_config, 1, 0.0)
            )
            with self.assertRaises(FileExistsError):
                run_training(root / "fixed", fixed_config, resume=checkpoint, resume_weights_only=True)


class TestWeightsOnlyCommandLines(unittest.TestCase):
    def test_trainer_cli_forwards_the_weights_only_flag_with_the_new_configuration(self):
        """Take the whole configuration from --config and pass resume_weights_only to run_training."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"regime": "fixed_states", "state_count": 6, "device": "cpu"}))
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"never read by the parser")
            arguments = [
                "train_mixed",
                "--output",
                str(root / "run"),
                "--resume",
                str(checkpoint),
                "--resume-weights-only",
                "--config",
                str(config_path),
            ]
            with patch.object(sys, "argv", arguments), patch.object(train_mixed, "run_training") as run:
                train_mixed._main()
            self.assertEqual(run.call_args.kwargs, {"resume": checkpoint, "resume_weights_only": True})
            config = run.call_args.args[1]
            self.assertEqual((config.regime, config.state_count, config.device), ("fixed_states", 6, "cpu"))
            with (
                patch.object(sys, "argv", ["train_mixed", "--output", str(root / "run"), "--resume-weights-only"]),
                patch.object(train_mixed, "run_training") as run,
                self.assertRaises(SystemExit),
            ):
                train_mixed._main()
            run.assert_not_called()

    def test_launcher_weights_only_starts_a_fresh_run_from_any_checkpoint(self):
        """Accept a checkpoint outside the output, require a fresh output and forward the flag to every rank."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim = root / "claim.sh"
            claim.touch()
            checkpoint = root / "previous/checkpoints/best_validation.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            output = root / "fresh"
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(launcher_module, "_run_workers", return_value=_WORKER_RESULT) as run,
            ):
                result = launch_training(
                    output,
                    workers=1,
                    pipeline="mixed",
                    resume=checkpoint,
                    resume_weights_only=True,
                    training_arguments=("--config", "fixed.json"),
                    gpu_claim=claim,
                )
            command, logs = run.call_args.args
            self.assertEqual(logs, output / "logs")
            self.assertEqual(
                command[0][-5:], ["--config", "fixed.json", "--resume", str(checkpoint), "--resume-weights-only"]
            )
            recorded = json.loads((output / "launcher.json").read_text())
            self.assertEqual((recorded["resume"], recorded["resume_weights_only"]), (str(checkpoint), True))
            self.assertTrue(result["resume_weights_only"])
            with patch.object(launcher_module, "_run_workers") as run:
                with self.assertRaises(FileExistsError):
                    launch_training(
                        output, pipeline="mixed", resume=checkpoint, resume_weights_only=True, gpu_claim=claim
                    )
                with self.assertRaises(FileNotFoundError):
                    launch_training(
                        root / "other",
                        pipeline="mixed",
                        resume=root / "missing.pt",
                        resume_weights_only=True,
                        gpu_claim=claim,
                    )
                with self.assertRaisesRegex(ValueError, "mixed"):
                    launch_training(root / "other", resume=checkpoint, resume_weights_only=True, gpu_claim=claim)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    launch_training(root / "other", pipeline="mixed", resume_weights_only=True, gpu_claim=claim)
                with self.assertRaisesRegex(ValueError, "launcher"):
                    launch_training(
                        root / "other", pipeline="mixed", training_arguments=("--resume-weights-only",), gpu_claim=claim
                    )
            run.assert_not_called()
            self.assertFalse((root / "other").exists())


if __name__ == "__main__":
    unittest.main()
