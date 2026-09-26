# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for fixed multi-rank physical queries."""

import importlib.util
import unittest
from types import SimpleNamespace

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.distributed_probe import ProbeConfig, collate_queries


class TestDistributedProbe(unittest.TestCase):
    def test_rank_seed_pools_are_disjoint_and_validation_is_held_out(self):
        configs = [
            ProbeConfig(rank=rank, world_size=4, batch_size=3, updates=1, cell_counts=(2, 2, 3)) for rank in range(4)
        ]
        self.assertEqual([config.train_seeds for config in configs], [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)])
        self.assertTrue(all(set(config.train_seeds).isdisjoint(config.validation_seeds) for config in configs))
        self.assertEqual(len(set.union(*(set(config.validation_seeds) for config in configs))), 4)

    def test_collation_preserves_each_original_y_and_pins(self):
        fixed = torch.tensor([0, 3])
        queries = []
        for seed in (7, 8):
            candidate = torch.full((1, 4, 3), float(seed))
            pins = torch.full((1, 2, 3), float(seed + 1))
            candidate[:, fixed] = pins
            original_y = torch.full((1, 4, 3), float(seed + 2))
            previous = torch.full((1, 4, 3), float(seed + 3))
            problem = SimpleNamespace(
                inertial_prediction=original_y, previous_positions=previous, fixed_positions=pins, fixed_indices=fixed
            )
            queries.append((problem, candidate, {"physical_seed": seed}))
        batch = collate_queries(queries)
        self.assertEqual(batch["physical_seeds"], [7, 8])
        torch.testing.assert_close(batch["positions"], torch.cat([query[1] for query in queries]))
        torch.testing.assert_close(
            batch["inertial_prediction"], torch.cat([query[0].inertial_prediction for query in queries])
        )
        torch.testing.assert_close(
            batch["previous_positions"], torch.cat([query[0].previous_positions for query in queries])
        )
        torch.testing.assert_close(batch["fixed_positions"], batch["positions"][:, fixed])
        self.assertFalse(torch.equal(batch["positions"], batch["inertial_prediction"]))

    def test_collation_rejects_broken_pins(self):
        fixed = torch.tensor([0])
        problem = SimpleNamespace(
            inertial_prediction=torch.ones(1, 2, 3),
            previous_positions=torch.ones(1, 2, 3),
            fixed_positions=torch.ones(1, 1, 3),
            fixed_indices=fixed,
        )
        with self.assertRaisesRegex(ValueError, "pins"):
            collate_queries([(problem, torch.zeros(1, 2, 3), {"physical_seed": 0})])

    def test_invalid_rank_and_failure_update_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            ProbeConfig(rank=4, world_size=4)
        with self.assertRaisesRegex(ValueError, "fail_update"):
            ProbeConfig(rank=0, world_size=4, updates=1, fail_rank=1, fail_update=2)


if __name__ == "__main__":
    unittest.main()
