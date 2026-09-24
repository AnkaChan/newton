# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistence, atomicity and reproducible sampling of retained physical states."""

import dataclasses
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import disk_replay


class TestDiskReplay(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "rank_0.sqlite3"
        self.positions = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
        self.metadata = {"time_step": 1 / 300, "gravity": [0, -9.81, 0], "lame_lambda": 20, "lame_mu": 10}
        self.arrays = {"rest_positions": self.positions, "fixed_indices": np.array([0], dtype=np.int64)}

    def context(self, store):
        return store.register_context(metadata=self.metadata, arrays=self.arrays)

    def state(self, context_id, step=0, trajectory="trajectory-a"):
        return disk_replay.ReplayState(
            context_id=context_id,
            trajectory_id=trajectory,
            step_index=step,
            positions=torch.from_numpy(self.positions.copy()).requires_grad_(),
            velocities=torch.full((2, 3), float(step), requires_grad=True),
            forces=np.full((2, 3), 0.25, dtype=np.float32),
            fixed_positions=self.positions[:1].copy(),
            metadata={"physical_seed": 7, "optimizer_update": step, "iterations": 4},
        )

    def test_disk_roundtrip_survives_restart_and_detaches_owned_values(self):
        # Catches losing records on restart, storing a graph, or keeping caller aliases.
        with disk_replay.DiskReplayStore(self.path) as store:
            context_id = self.context(store)
            self.assertEqual(context_id, self.context(store))
            state = self.state(context_id, 3)
            (record_id,) = store.append([state])
            with torch.no_grad():
                state.positions.add_(8)
            state.metadata["physical_seed"] = -1
        with disk_replay.DiskReplayStore(self.path, read_only=True) as store:
            saved = store.get(record_id)
            np.testing.assert_array_equal(saved.positions, self.positions)
            np.testing.assert_array_equal(saved.velocities, np.full((2, 3), 3, dtype=np.float32))
            np.testing.assert_array_equal(saved.forces, np.full((2, 3), 0.25, dtype=np.float32))
            self.assertEqual(saved.metadata["physical_seed"], 7)
            self.assertEqual(saved.positions.dtype, np.float32)
            np.testing.assert_array_equal(store.get_context(context_id)["arrays"]["fixed_indices"], [0])
            self.assertEqual(store.get_context(context_id)["metadata"], self.metadata)
            with self.assertRaises(PermissionError):
                store.append([self.state(context_id, 4)])
        with disk_replay.DiskReplayStore(self.path) as store:
            store.append([self.state(context_id, 4)])
            self.assertEqual(len(store), 2)

    def test_duplicate_batch_rolls_back_and_invalid_input_leaves_no_partial_write(self):
        # Catches partial batches, overwriting old states, and silently accepting bad states.
        with disk_replay.DiskReplayStore(self.path) as store:
            context_id = self.context(store)
            store.append([self.state(context_id)])
            with self.assertRaises(sqlite3.IntegrityError):
                store.append([self.state(context_id, 1), self.state(context_id)])
            self.assertEqual(len(store), 1)
            for bad in (
                dataclasses.replace(self.state(context_id, 2), positions=np.zeros((3, 3), dtype=np.float32)),
                dataclasses.replace(self.state(context_id, 2), velocities=np.full((2, 3), np.nan)),
                dataclasses.replace(self.state(context_id, 2), step_index=-1),
                dataclasses.replace(self.state(context_id, 2), context_id="missing"),
                dataclasses.replace(self.state(context_id, 2), fixed_positions=np.ones((2, 3))),
                dataclasses.replace(self.state(context_id, 2), metadata={"loss": float("nan")}),
            ):
                with self.subTest(bad=bad):
                    with self.assertRaises((ValueError, KeyError)):
                        store.append([self.state(context_id, 1), bad])
                    self.assertEqual(len(store), 1)

    def test_snapshot_sampling_across_rank_files_refresh_and_context_filter(self):
        # Catches missing ranks, moving sample populations, and nondeterministic replay.
        paths = [self.path, self.path.with_name("rank_1.sqlite3")]
        for rank, path in enumerate(paths):
            with disk_replay.DiskReplayStore(path) as store:
                context_id = self.context(store)
                store.append([self.state(context_id, step, f"rank-{rank}") for step in range(4)])
        with disk_replay.DiskReplayDataset(paths) as dataset:
            self.assertEqual(len(dataset), 8)

            def signature(records):
                return [(record.trajectory_id, record.step_index) for record in records]

            first = signature(dataset.sample(6, seed=42))
            self.assertEqual(first, signature(dataset.sample(6, seed=42)))
            self.assertEqual(len(set(first)), 6)
            self.assertEqual({record.trajectory_id for record in dataset.sample(8, seed=1)}, {"rank-0", "rank-1"})
            with disk_replay.DiskReplayStore(paths[0]) as store:
                other_id = store.register_context(metadata={**self.metadata, "lame_mu": 50}, arrays=self.arrays)
                store.append([self.state(other_id, 4, "other-material")])
            self.assertEqual(len(dataset), 8)
            dataset.refresh()
            self.assertEqual(len(dataset), 9)
            self.assertEqual(dataset[-1].trajectory_id, "rank-1")
            with self.assertRaises(IndexError):
                _ = dataset[9]
            with self.assertRaises(ValueError):
                dataset.sample(10, seed=1)
        with disk_replay.DiskReplayDataset(paths, context_id=context_id) as dataset:
            self.assertEqual(len(dataset), 8)

    def test_reader_does_not_create_missing_files_or_accept_unknown_schema(self):
        # Catches silently opening the wrong dataset during resume.
        with self.assertRaises(FileNotFoundError):
            disk_replay.DiskReplayStore(self.path, read_only=True)
        self.assertFalse(self.path.exists())
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE unrelated (value INTEGER)")
        with self.assertRaises(ValueError):
            disk_replay.DiskReplayStore(self.path)

    def test_committed_state_survives_writer_exit_without_close(self):
        # Catches acknowledging a write before a durable transaction commits.
        script = """
import os
import sys
import numpy as np
from experiments.learned_intrinsic_solver.disk_replay import DiskReplayStore, ReplayState
store = DiskReplayStore(sys.argv[1])
x = np.zeros((2, 3), dtype=np.float32)
context = store.register_context(metadata={}, arrays={
    'rest_positions': x, 'fixed_indices': np.array([], dtype=np.int64)})
store.append([ReplayState(context, 'crash-test', 0, x, x + 2)])
os._exit(0)
"""
        subprocess.run([sys.executable, "-c", script, str(self.path)], check=True, capture_output=True)
        with disk_replay.DiskReplayStore(self.path, read_only=True) as store:
            self.assertEqual(len(store), 1)
            np.testing.assert_array_equal(store.get(1).velocities, np.full((2, 3), 2, dtype=np.float32))

    def test_context_identity_tracks_material_and_arrays_and_zero_forces_are_compact(self):
        # Catches replaying the right X/V under a different material or rest grid.
        with disk_replay.DiskReplayStore(self.path) as store:
            first = self.context(store)
            reordered = store.register_context(metadata=dict(reversed(list(self.metadata.items()))), arrays=self.arrays)
            self.assertEqual(first, reordered)
            other_material = store.register_context(metadata={**self.metadata, "lame_mu": 1000}, arrays=self.arrays)
            other_geometry = store.register_context(
                metadata=self.metadata, arrays={**self.arrays, "rest_positions": self.positions * 2}
            )
            self.assertEqual(len({first, other_material, other_geometry}), 3)
            state = dataclasses.replace(self.state(first), forces=np.zeros((2, 3), dtype=np.float32))
            (identifier,) = store.append([state])
            self.assertIsNone(store.get(identifier).forces)

    def test_trajectory_cannot_silently_change_material_on_continuation(self):
        # Catches resuming a trajectory with a newly sampled material.
        with disk_replay.DiskReplayStore(self.path) as store:
            first = self.context(store)
            second = store.register_context(metadata={**self.metadata, "lame_mu": 1000}, arrays=self.arrays)
            store.append([self.state(first)])
            with self.assertRaisesRegex(ValueError, "trajectory.*context"):
                store.append([self.state(second, 1)])
            self.assertEqual(len(store), 1)
            with self.assertRaisesRegex(ValueError, "trajectory.*context"):
                store.append([self.state(first, 0, "new"), self.state(second, 1, "new")])
            self.assertEqual(len(store), 1)


if __name__ == "__main__":
    unittest.main()
