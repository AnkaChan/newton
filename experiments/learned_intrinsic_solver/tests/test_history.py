# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from types import SimpleNamespace

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.history import (
    HISTORY_KEYS,
    batch_history,
    carry_history,
    empty_history,
    store_history,
)
from experiments.learned_intrinsic_solver.mixed_physics import OptimizerHistory


class TestHistoryPlumbing(unittest.TestCase):
    def test_empty_history_has_zero_blocks_and_false_flag(self):
        payload = empty_history(5)
        self.assertEqual(tuple(payload), HISTORY_KEYS)
        self.assertEqual(payload["history_axis_gradient_world"].shape, (5, 3, 3))
        self.assertEqual(payload["history_axis_update_world"].dtype, torch.float32)
        self.assertFalse(payload["history_valid"])
        with self.assertRaises(ValueError):
            empty_history(0)

    def test_batch_history_treats_missing_or_invalid_entries_as_zero(self):
        stored = empty_history(4)
        gradient = torch.arange(36.0).reshape(4, 3, 3)
        update = -gradient
        payloads = [
            {},
            stored,
            {"history_axis_gradient_world": gradient, "history_axis_update_world": update, "history_valid": True},
        ]
        history = batch_history(payloads, torch.device("cpu"), cell_count=4)
        self.assertIsInstance(history, OptimizerHistory)
        self.assertEqual(history.axis_gradient_world.shape, (3, 4, 3, 3))
        self.assertEqual(history.valid.tolist(), [False, False, True])
        self.assertTrue(torch.equal(history.axis_gradient_world[:2], torch.zeros(2, 4, 3, 3)))
        self.assertTrue(torch.equal(history.axis_gradient_world[2], gradient))
        self.assertTrue(torch.equal(history.axis_update_world[2], update))
        self.assertFalse(history.axis_gradient_world.requires_grad)
        with self.assertRaises(ValueError):
            batch_history([], torch.device("cpu"), cell_count=4)
        with self.assertRaises(ValueError):
            batch_history([{"history_valid": True}], torch.device("cpu"), cell_count=4)

    def test_store_then_batch_round_trip_and_carry(self):
        gradient = torch.randn(2, 3, 3, 3, requires_grad=True)
        update = torch.randn(2, 3, 3, 3, requires_grad=True)
        result = SimpleNamespace(axis_gradient_world=gradient, achieved_axis_update_world=update)
        payloads = [empty_history(3), {}]
        store_history(payloads, result)
        for index, payload in enumerate(payloads):
            self.assertTrue(payload["history_valid"])
            self.assertFalse(payload["history_axis_gradient_world"].requires_grad)
            self.assertTrue(torch.equal(payload["history_axis_gradient_world"], gradient[index].detach()))
            self.assertTrue(torch.equal(payload["history_axis_update_world"], update[index].detach()))
        history = batch_history(payloads, torch.device("cpu"), cell_count=3)
        self.assertTrue(history.valid.all())
        self.assertTrue(torch.equal(history.axis_gradient_world, gradient.detach()))
        advanced = {"candidate": torch.zeros(1)}
        carry_history(payloads[0], advanced)
        self.assertEqual(set(HISTORY_KEYS) | {"candidate"}, set(advanced))
        self.assertIs(advanced["history_axis_gradient_world"], payloads[0]["history_axis_gradient_world"])
        untouched = {}
        carry_history({}, untouched)
        self.assertEqual(untouched, {})
        with self.assertRaises(ValueError):
            store_history(payloads, SimpleNamespace(axis_gradient_world=None, achieved_axis_update_world=update))
        with self.assertRaises(ValueError):
            store_history(payloads[:1], result)


if __name__ == "__main__":
    unittest.main()
