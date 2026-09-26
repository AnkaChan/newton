# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU lifecycle and exact-resume checks for independently sampled trajectories."""

import copy
import threading
import unittest

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.trajectory_pool import ActiveTrajectoryPool


class _Physics:
    def __init__(self):
        self.resets = []
        self.advances = []
        self.retired = []
        self.lock = threading.Lock()

    def reset(self, seed):
        with self.lock:
            self.resets.append(seed)
        return {
            "seed": seed,
            "physical_positions": torch.tensor([0.0]),
            "candidate": torch.tensor([0.0]),
            "velocities": torch.tensor([0.0]),
            "nested": {"value": [torch.tensor([1.0])]},
        }

    def advance(self, payload):
        with self.lock:
            self.advances.append(payload["seed"])
        # Carry every stored key, including optimizer history, across the boundary.
        result = dict(payload)
        result["velocities"] = payload["candidate"] - payload["physical_positions"]
        result["physical_positions"] = payload["candidate"].clone()
        return result

    def retire(self, payload):
        with self.lock:
            self.retired.append(payload["seed"])


def _signature(records):
    return [(r.id, r.seed, r.iteration_budget, r.step_budget, r.inner_iteration, r.physical_step) for r in records]


def _iterate(pool):
    records = pool.take_batch()
    signature = _signature(records)
    for record in records:
        record.payload["candidate"] = record.payload["candidate"] + 1
        record.payload["history_axis_update_world"] = record.payload["candidate"].sum()
        record.payload["history_valid"] = True
    pool.finish_batch(records)
    return signature


class TestTrajectoryPool(unittest.TestCase):
    def make_pool(self, *, physics=None, **kwargs):
        physics = physics or _Physics()
        settings = {
            "capacity": 5,
            "batch_size": 2,
            "iteration_counts": (1, 3),
            "physical_step_counts": (1, 2),
            "seed": 43,
        }
        settings.update(kwargs)
        pool = ActiveTrajectoryPool(reset=physics.reset, advance=physics.advance, retire=physics.retire, **settings)
        self.addCleanup(pool.close)
        return pool, physics

    def test_mixed_budgets_and_seeds_are_deterministic(self):
        """Sample independent budgets once per trajectory regardless of worker count."""
        first, first_physics = self.make_pool(workers=1)
        second, _ = self.make_pool(workers=3)
        expected = [_iterate(first) for _ in range(30)]
        actual = [_iterate(second) for _ in range(30)]
        self.assertEqual(actual, expected)
        observed = {}
        for batch in actual:
            self.assertEqual(len(batch), 2)
            self.assertEqual(len({row[0] for row in batch}), 2)
            for identity, seed, iterations, steps, _, _ in batch:
                self.assertEqual(observed.setdefault(identity, (seed, iterations, steps)), (seed, iterations, steps))
        self.assertEqual({row[1] for row in observed.values()}, {1, 3})
        self.assertEqual({row[2] for row in observed.values()}, {1, 2})
        self.assertGreater(len(observed), 5)
        first.quiesce()
        self.assertEqual(sorted(first_physics.resets), list(range(43, 43 + len(first_physics.resets))))

    def test_dispatch_reaches_every_initial_member_before_reusing_a_batch(self):
        """Prevent a long-K cohort from monopolizing all updates in the active pool."""
        for counts in ((32,), (1, 32)):
            with self.subTest(iteration_counts=counts):
                pool, _ = self.make_pool(capacity=8, iteration_counts=counts, physical_step_counts=(2,))
                pool.quiesce()
                first_round = [_iterate(pool) for _ in range(4)]
                self.assertEqual([[row[0] for row in batch] for batch in first_round], [[0, 1], [2, 3], [4, 5], [6, 7]])
                self.assertTrue(all(row[4:] == (0, 0) for batch in first_round for row in batch))
                second_round = [_iterate(pool) for _ in range(4)]
                self.assertEqual(
                    [[row[0] for row in batch] for batch in second_round], [[0, 1], [2, 3], [4, 5], [6, 7]]
                )
                self.assertTrue(all(row[4] + row[5] == 1 for batch in second_round for row in batch))
                if counts == (1, 32):
                    self.assertEqual({row[4:] for batch in second_round for row in batch}, {(1, 0), (0, 1)})
                if counts == (32,):
                    later = [_iterate(pool) for _ in range(24)]
                    self.assertEqual({row[0] for batch in later for row in batch}, set(range(8)))
                    self.assertTrue(all(record.inner_iteration == 8 for record in pool.records))

    def test_other_ready_records_run_while_advance_is_blocked(self):
        """Fill the next full batch from ready trajectories during CPU preparation."""
        physics = _Physics()
        released = threading.Event()
        blocked = threading.Event()
        original = physics.advance
        blocked_seed = []

        def advance(payload):
            if payload["seed"] == blocked_seed[0]:
                blocked.set()
                if not released.wait(5):
                    raise RuntimeError("test advance timed out")
            return original(payload)

        physics.advance = advance
        pool, _ = self.make_pool(physics=physics, capacity=4, iteration_counts=(1,), physical_step_counts=(2,))
        self.addCleanup(released.set)
        first = pool.take_batch()
        blocked_seed.append(first[0].seed)
        pool.finish_batch(first)
        self.assertTrue(blocked.wait(5))
        second = pool.take_batch()
        self.assertEqual([record.id for record in second], [2, 3])
        pool.finish_batch(second)
        released.set()
        third = pool.take_batch()
        self.assertEqual([record.id for record in third], [0, 1])
        pool.finish_batch(third)

    def test_advance_updates_velocity_once_at_each_physical_boundary(self):
        """Delegate velocity updates only after all inner iterations of a step; history is carried."""
        pool, physics = self.make_pool(capacity=3, batch_size=1, iteration_counts=(2,), physical_step_counts=(2,))
        initial_seed = None
        observations = []
        while len(observations) < 4:
            records = pool.take_batch()
            record = records[0]
            if initial_seed is None:
                initial_seed = record.seed
            if record.seed == initial_seed:
                observations.append((record.inner_iteration, record.physical_step, record.payload["velocities"].item()))
                if record.physical_step == 0 and record.inner_iteration == 0:
                    self.assertNotIn("history_valid", record.payload)
                if record.physical_step == 1 and record.inner_iteration == 0:
                    self.assertTrue(record.payload["history_valid"])
                    self.assertEqual(record.payload["history_axis_update_world"].item(), 9.0)
            record.payload["candidate"] += 1
            record.payload["history_axis_update_world"] = torch.tensor(9.0)
            record.payload["history_valid"] = True
            pool.finish_batch(records)
        pool.quiesce()
        self.assertEqual(observations, [(0, 0, 0.0), (1, 0, 0.0), (0, 1, 2.0), (1, 1, 2.0)])
        self.assertEqual(physics.advances.count(initial_seed), 1)
        self.assertEqual(physics.retired.count(initial_seed), 1)

    def test_maximum_horizon_runs_32_iterations_on_each_of_128_steps(self):
        """Complete the maximum scheduler horizon without an unused 129th preparation."""
        resets, advances, retired = [], [], []

        def reset(seed):
            resets.append(seed)
            return {"seed": seed}

        def advance(payload):
            advances.append(payload["seed"])
            return dict(payload)

        def retire(payload):
            retired.append(payload["seed"])

        with ActiveTrajectoryPool(
            2, 1, reset, advance, retire, iteration_counts=(32,), physical_step_counts=(128,), seed=43, workers=1
        ) as pool:
            observed = []
            while 43 not in retired:
                records = pool.take_batch()
                record = records[0]
                if record.seed == 43:
                    observed.append((record.inner_iteration, record.physical_step))
                pool.finish_batch(records)
            pool.quiesce()
            self.assertEqual(len(observed), 4096)
            self.assertEqual(observed[0], (0, 0))
            self.assertEqual(observed[31:33], [(31, 0), (0, 1)])
            self.assertEqual(observed[-1], (31, 127))
            for physical_step in range(128):
                self.assertEqual(sum(step == physical_step for _, step in observed), 32)
            self.assertEqual(advances.count(43), 127)
            self.assertEqual(retired, [43])
            self.assertEqual(sorted(resets), [43, 44, 45])
            self.assertEqual(pool.stats["retired"], 1)

    def test_single_step_trajectories_retire_without_advancing(self):
        """Reset finished trajectories without computing an unused future state."""
        pool, physics = self.make_pool(iteration_counts=(1,), physical_step_counts=(1,))
        for _ in range(4):
            _iterate(pool)
        pool.quiesce()
        self.assertEqual(physics.advances, [])
        self.assertEqual(len(physics.retired), 8)
        self.assertEqual(len(physics.resets), 13)

    def test_curriculum_changes_only_new_trajectory_budgets(self):
        """Keep active K and H fixed when curriculum availability changes."""
        pool, _ = self.make_pool(iteration_counts=(2,), physical_step_counts=(2,))
        pool.quiesce()
        initial_ids = {record.id for record in pool.records}
        pool.set_available_counts((1,), (1,))
        saw_new = False
        for _ in range(20):
            for identity, _, iterations, steps, _, _ in _iterate(pool):
                if identity in initial_ids:
                    self.assertEqual((iterations, steps), (2, 2))
                else:
                    saw_new = True
                    self.assertEqual((iterations, steps), (1, 1))
        self.assertTrue(saw_new)

    def test_checkpoint_preserves_pending_queue_rng_and_payload(self):
        """Resume the identical future batches without calling any initial resets."""
        pool, _ = self.make_pool(iteration_counts=(1, 2), physical_step_counts=(1, 3))
        for _ in range(2):
            _iterate(pool)
        state = pool.state_dict()
        self.assertGreater(len(state["pending"]), 0)
        self.assertGreater(len(state["ready"]), 0)
        restore_physics = _Physics()
        restored = ActiveTrajectoryPool.from_state_dict(
            state,
            reset=restore_physics.reset,
            advance=restore_physics.advance,
            retire=restore_physics.retire,
            workers=3,
        )
        self.addCleanup(restored.close)
        self.assertEqual(restore_physics.resets, [])
        self.assertEqual(restored.state_dict()["pending"], state["pending"])
        self.assertEqual(restored.state_dict()["ready"], state["ready"])
        self.assertEqual(restored.state_dict()["dispatch"], state["dispatch"])
        for _ in range(30):
            first = pool.take_batch()
            second = restored.take_batch()
            self.assertEqual(_signature(first), _signature(second))
            for a, b in zip(first, second, strict=True):
                torch.testing.assert_close(a.payload["candidate"], b.payload["candidate"], rtol=0, atol=0)
                a.payload["candidate"] += 1
                b.payload["candidate"] += 1
            pool.finish_batch(first)
            restored.finish_batch(second)

    def test_checkpoint_payloads_are_independent_cpu_detached_copies(self):
        """Make serialized tensor storage independent from all live payloads."""
        pool, _ = self.make_pool()
        pool.quiesce()
        state = pool.state_dict()
        identity = state["records"][0]["id"]
        original = next(record for record in pool.records if record.id == identity)
        saved = state["records"][0]["payload"]
        original.payload["nested"]["value"][0].add_(7)
        self.assertEqual(saved["nested"]["value"][0].item(), 1.0)
        self.assertEqual(saved["candidate"].device.type, "cpu")
        self.assertFalse(saved["candidate"].requires_grad)

    def test_iteration_boundaries_detach_all_payload_tensors(self):
        """Release autograd references at every inner iteration boundary."""
        pool, _ = self.make_pool(iteration_counts=(3,), physical_step_counts=(2,))
        records = pool.take_batch()
        inputs = torch.tensor([2.0], requires_grad=True)
        for record in records:
            record.payload["candidate"] = inputs.square()
            record.payload["nested"] = {"value": [inputs * 3]}
            record.payload["history_axis_gradient_world"] = inputs.sum()
        pool.finish_batch(records)
        for record in records:
            for tensor in (
                record.payload["candidate"],
                record.payload["nested"]["value"][0],
                record.payload["history_axis_gradient_world"],
            ):
                self.assertIsNone(tensor.grad_fn)
                self.assertFalse(tensor.requires_grad)

    def test_preparation_failure_is_not_replaced_by_fresh_reset(self):
        """Propagate failed physical advancement without silently resetting data."""
        physics = _Physics()

        def fail(payload):
            raise ValueError("invalid physical state")

        physics.advance = fail
        pool, _ = self.make_pool(
            physics=physics, capacity=3, batch_size=1, iteration_counts=(1,), physical_step_counts=(2,)
        )
        for _ in range(3):
            _iterate(pool)
        with self.assertRaisesRegex(RuntimeError, "invalid physical state"):
            pool.take_batch()
        self.assertEqual(len(physics.resets), 3)
        with self.assertRaisesRegex(RuntimeError, "invalid physical state"):
            pool.close()
        self.assertEqual(sorted(physics.retired), sorted(physics.resets))

    def test_close_drains_preparation_and_retires_each_context_once(self):
        """Release both ready and prepared pending contexts on repeated close."""
        pool, physics = self.make_pool(iteration_counts=(1,), physical_step_counts=(2,))
        _iterate(pool)
        pool.close()
        pool.close()
        self.assertEqual(sorted(physics.retired), sorted(physics.resets))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.take_batch()

    def test_rejects_invalid_pool_sizes_and_counts(self):
        """Reject dimensions that cannot supply a distinct full active batch."""
        for settings in (
            {"capacity": 2},
            {"batch_size": 0},
            {"workers": 0},
            {"iteration_counts": ()},
            {"iteration_counts": (0,)},
            {"physical_step_counts": (True,)},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.make_pool(**settings)

    def test_checked_out_batches_cannot_be_lost_or_finished_twice(self):
        """Reject invalid ownership transitions before mutating any trajectory."""
        pool, _ = self.make_pool()
        records = pool.take_batch()
        with self.assertRaisesRegex(RuntimeError, "batch"):
            pool.take_batch()
        with self.assertRaisesRegex(RuntimeError, "batch"):
            pool.state_dict()
        with self.assertRaises(ValueError):
            pool.finish_batch([records[0], records[0]])
        self.assertEqual(records[0].inner_iteration, 0)
        pool.finish_batch(records)
        with self.assertRaises(ValueError):
            pool.finish_batch(records)

    def test_restore_rejects_inconsistent_queue_membership(self):
        """Reject corrupt snapshots instead of dropping or repeating a trajectory."""
        pool, _ = self.make_pool()
        state = pool.state_dict()
        for invalid_queue in ("ready", "dispatch"):
            with self.subTest(invalid_queue=invalid_queue):
                corrupt = copy.deepcopy(state)
                corrupt[invalid_queue].append(corrupt["pending"][0])
                physics = _Physics()
                with self.assertRaises(ValueError):
                    ActiveTrajectoryPool.from_state_dict(corrupt, reset=physics.reset, advance=physics.advance)
                self.assertEqual(physics.resets, [])


if __name__ == "__main__":
    unittest.main()
