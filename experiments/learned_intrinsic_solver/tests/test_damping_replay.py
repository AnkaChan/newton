# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve absolute damping and physical anchors across disk replay."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.disk_replay import DiskReplayStore, ReplayState
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.physical_rollout import PhysicalRollout
from experiments.learned_intrinsic_solver.replay_rollout import (
    build_replay_rollout,
    physical_context,
    replay_state,
    retain_windows,
)
from experiments.learned_intrinsic_solver.unrolled_solver import UnrolledHexSolver


class TestDampingReplay(unittest.TestCase):
    def _fixture(self, *, damped=True):
        torch.manual_seed(137)
        rest = generate_cuboid((1, 1, 2), cell_size=0.1)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        model = build_newton_hex_model(
            rest,
            fixed,
            lame_lambda=[700.0, 1000.0],
            lame_mu=[300.0, 600.0],
            density=100.0,
            damping=[10.0, 100.0] if damped else 0.0,
            gravity=(0, -9.81, 0),
        )
        network = IntrinsicSolverNetwork(
            rest.cell_counts,
            86 if damped else 38,
            conditioning_dim=6 if damped else 5,
            hidden_dim=16,
            edge_hidden_dim=8,
        )
        with torch.no_grad():
            network.correction_head.weight.normal_(std=1e-4)
        solver = SolverLearnedIntrinsic(model, network=network)
        step = solver._step_for_dt(0.01)
        rollout = PhysicalRollout(solver, UnrolledHexSolver(step), time_step=0.01)
        x = torch.tensor(rest.corner_rest_positions, dtype=torch.float32)[None]
        x[..., 0] += 0.05 * x[..., 2]
        velocity = torch.zeros_like(x)
        velocity[..., 1] = 0.1 * x[..., 2]
        return rollout, x, velocity

    def test_damped_retention_and_reconstruction_preserve_the_solve(self):
        """Round-trip per-cell viscosity and both physical starts without changing energies."""
        rollout, x, velocity = self._fixture()
        with tempfile.TemporaryDirectory() as directory, DiskReplayStore(Path(directory) / "states.sqlite3") as store:
            windows = list(
                retain_windows(
                    rollout,
                    store,
                    x,
                    velocity,
                    trajectory_ids=("damped-137",),
                    physical_steps=2,
                    iterations=2,
                )
            )
            states = [store.get(int(record)) for record in store.record_ids()]
            context = store.get_context(states[0].context_id)
        self.assertEqual(context["metadata"]["schema_version"], 2)
        np.testing.assert_array_equal(context["arrays"]["damping"], [10.0, 100.0])
        restored = build_replay_rollout(
            context, copy.deepcopy(rollout.unrolled.step.network), gravity=states[0].metadata["gravity"]
        )
        torch.testing.assert_close(restored.unrolled.step.energy.damping, torch.tensor([10.0, 100.0]))
        for state, window in zip(states, windows, strict=True):
            actual = replay_state(restored, context, state)
            original = window.steps[0]
            for name in ("previous_positions", "inertial_prediction", "energies", "positions", "velocities"):
                torch.testing.assert_close(getattr(actual, name), getattr(original, name).detach(), rtol=0, atol=0)

    def test_legacy_missing_damping_rebuilds_only_zero_physics(self):
        """Read an old zero-damping context and reproduce its physical solve."""
        rollout, x, velocity = self._fixture(damped=False)
        legacy = physical_context(rollout)
        legacy["metadata"]["schema_version"] = 1
        legacy["arrays"].pop("damping", None)
        restored = build_replay_rollout(legacy, copy.deepcopy(rollout.unrolled.step.network), gravity=(0, -9.81, 0))
        torch.testing.assert_close(restored.unrolled.step.energy.damping, torch.zeros(2), rtol=0, atol=0)
        expected = next(rollout.windows(x, velocity, physical_steps=1, iterations=2)).steps[0]
        actual = next(restored.windows(x, velocity, physical_steps=1, iterations=2)).steps[0]
        torch.testing.assert_close(actual.energies, expected.energies, rtol=0, atol=0)
        torch.testing.assert_close(actual.positions, expected.positions, rtol=0, atol=0)

    def test_legacy_context_cannot_hide_active_damping(self):
        """Reject replay of a legacy zero context through an already damped rollout."""
        rollout, x, velocity = self._fixture()
        legacy = physical_context(rollout)
        legacy["metadata"]["schema_version"] = 1
        legacy["arrays"].pop("damping", None)
        legacy["context_id"] = "legacy-zero"
        state = ReplayState(
            context_id="legacy-zero",
            trajectory_id="legacy-state",
            step_index=0,
            positions=x[0].numpy(),
            velocities=velocity[0].numpy(),
            metadata={"gravity": [0, -9.81, 0], "iterations": 1},
        )
        with self.assertRaisesRegex(ValueError, "damping"):
            replay_state(rollout, legacy, state)

    def test_new_schema_missing_damping_and_legacy_positive_damping_fail(self):
        """Reject incomplete modern material data and positive damping mislabeled legacy."""
        rollout, _, _ = self._fixture()
        context = physical_context(rollout)
        context["metadata"]["schema_version"] = 2
        context["arrays"].pop("damping", None)
        with self.assertRaisesRegex(ValueError, "damping"):
            build_replay_rollout(context, rollout.unrolled.step.network, gravity=(0, -9.81, 0))
        context["metadata"]["schema_version"] = 1
        context["arrays"]["damping"] = np.array([10.0, 100.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "damping"):
            build_replay_rollout(context, rollout.unrolled.step.network, gravity=(0, -9.81, 0))


if __name__ == "__main__":
    unittest.main()
