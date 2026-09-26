# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU retention/replay of actual physical step-start states."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.disk_replay import DiskReplayDataset, DiskReplayStore
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


def _network(cell_counts):
    return IntrinsicSolverNetwork(
        cell_counts,
        features.STATE_FEATURE_DIM,
        conditioning_dim=features.CONDITIONING_DIM,
        hidden_dim=16,
        edge_hidden_dim=8,
    )


class TestReplayRollout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        rest = generate_cuboid((1, 1, 2), cell_size=0.1)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        model = build_newton_hex_model(
            rest,
            fixed,
            lame_lambda=576.9230769,
            lame_mu=384.6153846,
            density=100,
            gravity=(0, -9.81, 0),
        )
        network = _network(rest.cell_counts)
        with torch.no_grad():
            network.correction_head.weight.normal_(std=1e-4)
        solver = SolverLearnedIntrinsic(model, network=network, iterations=2)
        step = solver._step_for_dt(1 / 300)
        self.rollout = PhysicalRollout(solver, UnrolledHexSolver(step), time_step=1 / 300)
        x = rest.corner_rest_positions.astype(np.float32).copy()
        x[:, 0] += np.float32(0.02) * x[:, 2] ** 2
        v = np.zeros_like(x)
        v[:, 0] = np.float32(0.05) * x[:, 2]
        self.x = torch.from_numpy(x)[None].requires_grad_()
        self.v = torch.from_numpy(v)[None].requires_grad_()
        self.forces = torch.zeros((2, *self.x.shape), dtype=torch.float32)
        self.forces[0, :, :, 1] = 0.0007
        self.forces[1, :, :, 0] = 0.0003
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "states.sqlite3"

    def _retain(self, store, *, physical_steps=2):
        return retain_windows(
            self.rollout,
            store,
            self.x,
            self.v,
            physical_steps=physical_steps,
            iterations=2,
            forces=self.forces[:physical_steps],
            trajectory_ids=("test-seed-71",),
            provenance={"source_checkpoint": "sha256:example", "physical_seed": 71},
        )

    def test_step_starts_are_saved_before_guess_and_replay_current_network(self):
        with DiskReplayStore(self.path) as store:
            windows = list(self._retain(store))
            self.assertEqual(len(store), 2)
        self.assertEqual(len(windows), 2)
        windows[0].objective.backward()
        self.assertIsNotNone(self.x.grad)
        self.assertGreater(self.x.grad.norm().item(), 0)
        with DiskReplayDataset([self.path]) as dataset:
            states = [dataset[i] for i in range(2)]
        np.testing.assert_array_equal(states[0].positions, self.x.detach().numpy()[0])
        np.testing.assert_array_equal(states[1].positions, windows[0].final_positions.detach().numpy()[0])
        np.testing.assert_array_equal(states[1].velocities, windows[0].final_velocities.detach().numpy()[0])
        np.testing.assert_array_equal(states[1].forces, self.forces[1, 0].numpy())
        self.assertFalse(np.array_equal(states[0].positions, windows[0].steps[0].initial_candidate.detach().numpy()[0]))
        self.assertEqual(states[0].metadata["source_checkpoint"], "sha256:example")
        self.assertEqual(states[1].metadata["iterations"], 2)
        # The retained history is the detached history entering each step: none at the start, then carried.
        self.assertIsNone(states[0].history)
        carried = windows[0].steps[0].history
        self.assertEqual(carried.valid.tolist(), [True])
        np.testing.assert_array_equal(states[1].history["axis_gradient_world"], carried.axis_gradient_world[0].numpy())
        np.testing.assert_array_equal(states[1].history["axis_update_world"], carried.axis_update_world[0].numpy())
        self.assertGreater(np.abs(states[1].history["axis_update_world"]).max(), 0)
        with DiskReplayStore(self.path, read_only=True) as store:
            context = store.get_context(states[0].context_id)
        for name in ("lame_lambda", "lame_mu", "density", "lumped_mass"):
            np.testing.assert_array_equal(
                context["arrays"][name], getattr(self.rollout.unrolled.step.energy, name).detach().numpy()
            )
        replay_network = _network((1, 1, 2))
        replay_network.load_state_dict(self.rollout.unrolled.step.network.state_dict(), strict=True)
        replay_rollout = build_replay_rollout(context, replay_network, gravity=states[0].metadata["gravity"])
        for state, window in zip(states, windows, strict=True):
            replayed = replay_state(replay_rollout, context, state)
            original = window.steps[0]
            torch.testing.assert_close(
                replayed.inertial_prediction, original.inertial_prediction.detach(), rtol=0, atol=0
            )
            torch.testing.assert_close(replayed.initial_candidate, original.initial_candidate.detach(), rtol=0, atol=0)
            torch.testing.assert_close(replayed.energies, original.energies.detach(), rtol=0, atol=0)
            torch.testing.assert_close(replayed.positions, original.positions.detach(), rtol=0, atol=0)
        with torch.no_grad():
            replay_network.correction_head.bias.add_(0.001)
        current = replay_state(replay_rollout, context, states[0])
        self.assertFalse(torch.equal(current.positions, windows[0].steps[0].positions.detach()))

    def test_failed_solve_keeps_its_step_start_and_rejects_changed_gravity(self):
        with DiskReplayStore(self.path) as store:
            with patch.object(self.rollout, "_advance", side_effect=RuntimeError("learned failure")):
                with self.assertRaisesRegex(RuntimeError, "learned failure"):
                    list(self._retain(store, physical_steps=1))
            self.assertEqual(len(store), 1)
            with DiskReplayDataset([self.path]) as dataset:
                state = dataset[0]
            context = store.get_context(state.context_id)
        self.rollout.solver.model.set_gravity((0, -1, 0))
        with self.assertRaisesRegex(ValueError, "gravity"):
            replay_state(self.rollout, context, state)

    def test_compacted_zero_force_replays_as_zero_load(self):
        with DiskReplayStore(self.path) as store:
            window = next(
                retain_windows(
                    self.rollout,
                    store,
                    self.x,
                    self.v,
                    trajectory_ids=("zero-force",),
                    physical_steps=1,
                    iterations=1,
                    provenance={"physical_seed": 1},
                )
            )
            (record_id,) = store.record_ids()
            state = store.get(int(record_id))
            context = store.get_context(state.context_id)
        self.assertIsNone(state.forces)
        replayed = replay_state(self.rollout, context, state)
        torch.testing.assert_close(replayed.inertial_prediction, window.steps[0].inertial_prediction, rtol=0, atol=0)

    def test_distinct_per_cell_material_and_authoritative_masses_round_trip(self):
        rest = self.rollout.solver._rest
        fixed = self.rollout.unrolled.step.fixed_indices.numpy()
        material = {
            "lame_lambda": np.array([1e3, 1e6], dtype=np.float32),
            "lame_mu": np.array([1e6, 1e3], dtype=np.float32),
            "density": np.array([100, 10000], dtype=np.float32),
        }
        model = build_newton_hex_model(rest, fixed, gravity=(0, -9.81, 0), **material)
        network = _network(rest.cell_counts)
        solver = SolverLearnedIntrinsic(model, network=network)
        original = PhysicalRollout(solver, UnrolledHexSolver(solver._step_for_dt(1 / 300)), time_step=1 / 300)
        context = physical_context(original)
        restored_network = _network(rest.cell_counts)
        restored = build_replay_rollout(context, restored_network, gravity=(0, -9.81, 0))
        for name, values in material.items():
            np.testing.assert_array_equal(context["arrays"][name], values)
            np.testing.assert_array_equal(physical_context(restored)["arrays"][name], values)
        np.testing.assert_array_equal(
            physical_context(restored)["arrays"]["lumped_mass"], context["arrays"]["lumped_mass"]
        )

    def test_replay_rejects_a_record_from_another_material_context(self):
        with DiskReplayStore(self.path) as store:
            _ = next(self._retain(store, physical_steps=1))
            (record_id,) = store.record_ids()
            state = store.get(int(record_id))
            source = store.get_context(state.context_id)
            other_arrays = {name: value.copy() for name, value in source["arrays"].items()}
            other_arrays["lame_mu"] *= 2
            other_id = store.register_context(metadata=source["metadata"], arrays=other_arrays)
            other = store.get_context(other_id)
        network = _network((1, 1, 2))
        wrong_physics = build_replay_rollout(other, network, gravity=state.metadata["gravity"])
        with self.assertRaisesRegex(ValueError, "context_id"):
            replay_state(wrong_physics, other, state)

    def test_callable_provenance_captures_current_update_before_each_step(self):
        update = {"number": 4}
        with DiskReplayStore(self.path) as store:
            iterator = retain_windows(
                self.rollout,
                store,
                self.x,
                self.v,
                trajectory_ids=("changing-update",),
                physical_steps=2,
                iterations=1,
                step_offset=5,
                provenance=lambda step: {"optimizer_update": update["number"], "step_seen": step},
            )
            _ = next(iterator)
            update["number"] = 5
            _ = next(iterator)
            first_id, second_id = store.record_ids()
            first, second = store.get(int(first_id)), store.get(int(second_id))
        self.assertEqual(first.metadata["optimizer_update"], 4)
        self.assertEqual(second.metadata["optimizer_update"], 5)
        self.assertEqual([first.metadata["step_seen"], second.metadata["step_seen"]], [5, 6])
        self.assertEqual([first.step_index, second.step_index], [5, 6])

    def test_physical_context_reuses_state_under_a_new_loss_policy(self):
        with DiskReplayStore(self.path) as store:
            original = next(self._retain(store, physical_steps=1)).steps[0]
            (record_id,) = store.record_ids()
            state = store.get(int(record_id))
            context = store.get_context(state.context_id)
        network = _network((1, 1, 2))
        network.load_state_dict(self.rollout.unrolled.step.network.state_dict(), strict=True)
        changed_policy = build_replay_rollout(
            context,
            network,
            gravity=state.metadata["gravity"],
            energy_increase_weight=2.0,
            detach_energy_target=False,
        )
        self.assertNotEqual(state.metadata["energy_increase_weight"], changed_policy.unrolled.energy_increase_weight)
        self.assertNotEqual(state.metadata["detach_energy_target"], changed_policy.detach_energy_target)
        replayed = replay_state(changed_policy, context, state)
        torch.testing.assert_close(replayed.inertial_prediction, original.inertial_prediction, rtol=0, atol=0)
        torch.testing.assert_close(replayed.energies, original.energies, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
