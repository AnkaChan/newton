# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for sharded, replayable learned-solver epochs."""

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.epoch_data import EpochDataset
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.train_smoke import TrainSmokeConfig


def _setup(config):
    rest = generate_cuboid(config.cell_counts, cell_size=config.cell_size)
    fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
    model = build_newton_hex_model(
        rest,
        fixed,
        lame_lambda=config.lame_lambda,
        lame_mu=config.lame_mu,
        density=config.density,
        gravity=config.gravity,
    )
    return rest, model, SolverLearnedIntrinsic(model, iterations=1)


def _config():
    return TrainSmokeConfig(
        updates=1,
        cell_counts=(2, 2, 2),
        hidden_dim=16,
        edge_hidden_dim=8,
        train_count=8,
        validation_count=4,
        train_seed_start=20,
        validation_seed_start=100,
        seed=127,
        verbose=False,
        device="cpu",
    )


class TestEpochDataset(unittest.TestCase):
    def test_rank_shards_cover_disjoint_physical_pools_without_drops(self):
        config = _config()
        rest, model, solver = _setup(config)
        with tempfile.TemporaryDirectory() as directory:
            shards = [
                EpochDataset(config, rest, model, solver, rank=rank, world_size=2, dataset_dir=Path(directory))
                for rank in range(2)
            ]
            self.assertEqual([shard.train_seeds for shard in shards], [(20, 21, 22, 23), (24, 25, 26, 27)])
            self.assertEqual([shard.validation_seeds for shard in shards], [(100, 101), (102, 103)])
            self.assertEqual(shards[0].step, solver.learned_step)
            for shard in shards:
                batches = list(shard.training_batches(0, batch_size=3))
                self.assertEqual([len(batch["physical_seeds"]) for batch in batches], [3, 1])
                self.assertEqual(
                    {seed for batch in batches for seed in batch["physical_seeds"]}, set(shard.train_seeds)
                )
                self.assertEqual(
                    {seed for batch in shard.validation_batches(3) for seed in batch["physical_seeds"]},
                    set(shard.validation_seeds),
                )
            self.assertTrue(set(shards[0].train_seeds).isdisjoint(shards[1].train_seeds))
            self.assertTrue(set(config.train_seeds).isdisjoint(config.validation_seeds))
            for world_size in (1, 4):
                world_dir = Path(directory) / f"world_{world_size}"
                world_shards = [
                    EpochDataset(config, rest, model, solver, rank=rank, world_size=world_size, dataset_dir=world_dir)
                    for rank in range(world_size)
                ]
                self.assertEqual(
                    [shard.train_seeds for shard in world_shards],
                    [
                        config.train_seeds[rank * (8 // world_size) : (rank + 1) * (8 // world_size)]
                        for rank in range(world_size)
                    ],
                )
                self.assertEqual(
                    [shard.validation_seeds for shard in world_shards],
                    [
                        config.validation_seeds[rank * (4 // world_size) : (rank + 1) * (4 // world_size)]
                        for rank in range(world_size)
                    ],
                )

    def test_epoch_candidates_replay_by_seed_and_preserve_original_y_and_pins(self):
        config = _config()
        rest, model, solver = _setup(config)
        with tempfile.TemporaryDirectory() as directory:
            dataset = EpochDataset(config, rest, model, solver, rank=0, world_size=2, dataset_dir=Path(directory))
            first = list(dataset.training_batches(3, batch_size=2))
            repeated = list(dataset.training_batches(3, batch_size=3))
            by_seed = {
                seed: (batch["positions"][i], batch["inertial_prediction"][i], batch["metadata"][i])
                for batch in first
                for i, seed in enumerate(batch["physical_seeds"])
            }
            for batch in repeated:
                for i, seed in enumerate(batch["physical_seeds"]):
                    candidate, original_y, metadata = by_seed[seed]
                    torch.testing.assert_close(batch["positions"][i], candidate, rtol=0, atol=0)
                    torch.testing.assert_close(batch["inertial_prediction"][i], original_y, rtol=0, atol=0)
                    self.assertEqual(batch["metadata"][i], metadata)
                    problem = dataset.sampler.problems[seed]
                    torch.testing.assert_close(batch["inertial_prediction"][i], problem.inertial_prediction[0])
                    torch.testing.assert_close(batch["positions"][i, problem.fixed_indices], problem.fixed_positions[0])
            other = {
                seed: batch["metadata"][i]["noise_seed"]
                for batch in dataset.training_batches(4, batch_size=3)
                for i, seed in enumerate(batch["physical_seeds"])
            }
            self.assertTrue(all(other[seed] != by_seed[seed][2]["noise_seed"] for seed in dataset.train_seeds))

    def test_resume_reloads_identical_validation_and_rejects_mismatch(self):
        config = _config()
        rest, model, solver = _setup(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            initial = EpochDataset(config, rest, model, solver, rank=0, world_size=2, dataset_dir=path)
            validation = list(initial.validation_batches(1))
            identity = initial.dataset_identity
            with self.assertRaisesRegex(FileExistsError, "resume"):
                EpochDataset(config, rest, model, solver, rank=0, world_size=2, dataset_dir=path)
            reloaded = EpochDataset(
                replace(config, updates=2, verbose=True),
                rest,
                model,
                solver,
                rank=0,
                world_size=2,
                dataset_dir=path,
                resume=True,
            )
            self.assertEqual(reloaded.dataset_identity, identity)
            self.assertEqual(reloaded.state_dict()["dataset_identity"], identity)
            reloaded_validation = list(reloaded.validation_batches(1))
            self.assertEqual(len(reloaded_validation), len(validation))
            for before, after in zip(validation, reloaded_validation, strict=True):
                self.assertEqual(before["physical_seeds"], after["physical_seeds"])
                self.assertEqual(before["metadata"], after["metadata"])
                for field in ("positions", "inertial_prediction", "fixed_positions"):
                    torch.testing.assert_close(before[field], after[field], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "configuration"):
                EpochDataset(
                    replace(config, seed=128), rest, model, solver, rank=0, world_size=2, dataset_dir=path, resume=True
                )
            with self.assertRaisesRegex(ValueError, "world_size"):
                EpochDataset(config, rest, model, solver, rank=0, world_size=4, dataset_dir=path, resume=True)

    def test_invalid_sharding_is_rejected_before_sampling(self):
        config = replace(_config(), train_count=6)
        rest, model, solver = _setup(config)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "divisible"):
                EpochDataset(config, rest, model, solver, rank=0, world_size=4, dataset_dir=Path(directory))
            self.assertFalse((Path(directory) / "rank_0.pt").exists())


if __name__ == "__main__":
    unittest.main()
