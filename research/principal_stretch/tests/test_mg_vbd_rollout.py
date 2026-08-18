# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for history-preserving MG-VBD rollout state."""

from __future__ import annotations

import dataclasses
import inspect
import os
import unittest

import numpy as np
import warp as wp

from research.principal_stretch.captured_graph_vbd import CapturedDirectGraphVBD, _lookup_workspace_owners
from research.principal_stretch.mg_vbd_rollout import (
    CONTRACT_ID,
    MGVBDRollout,
    MGVBDRolloutBackend,
    MGVBDRolloutCapturedBackend,
    MGVBDRolloutStepEndpoint,
    MGVBDRolloutStepInput,
)
from research.principal_stretch.solver_benchmark import (
    TetBenchmarkScene,
    build_structured_cantilever_scene,
)


def _scene() -> TetBenchmarkScene:
    return build_structured_cantilever_scene(
        dimensions=(1, 1, 1),
        dt=1.0 / 32.0,
        gravity=(0.25, -0.5, -1.5),
        total_tip_force=(1.0, -0.75, 0.5),
        initial_velocity=(0.03, -0.02, 0.01),
        name="mg-vbd-rollout-test",
    )


def _published_velocity(
    positions: np.ndarray,
    x_current: np.ndarray,
    pinned_indices: np.ndarray,
    dt: float,
) -> np.ndarray:
    velocity = ((positions - x_current) * np.float64(1.0 / dt)).astype(np.float32)
    velocity[pinned_indices] = np.float32(0.0)
    return velocity.astype(np.float64)


def _deterministic_endpoint(
    scene: TetBenchmarkScene,
    step_input: MGVBDRolloutStepInput,
) -> MGVBDRolloutStepEndpoint:
    """Stand in for one deterministic float32-published MG-VBD solve."""
    positions = np.array(step_input.inertial_target, dtype=np.float64, copy=True)
    free = scene.free_indices
    restoring = (scene.rest_q[free] - step_input.positions[free]) * np.float64(0.03125)
    forced = (
        step_input.external_force[free]
        * scene.particle_inv_mass[free, None]
        * np.float64(step_input.dt * step_input.dt * 0.125)
    )
    positions[free] += restoring + forced
    positions = positions.astype(np.float32).astype(np.float64)
    positions[step_input.pinned_indices] = step_input.pin_targets
    velocities = _published_velocity(
        positions,
        step_input.positions,
        step_input.pinned_indices,
        step_input.dt,
    )
    return MGVBDRolloutStepEndpoint(
        positions=positions,
        velocities=velocities,
        accepted=(True, True, False, False),
        reasons=("accepted", "accepted", "objective-increase", "masked-after-rejection"),
        graph_replay=True,
    )


class _DeterministicBackend(MGVBDRolloutBackend):
    def __init__(self, scene: TetBenchmarkScene):
        self._scene = scene
        self._resource_identity = ("static-hierarchy", id(self), "captured-graph")
        self.inputs: list[MGVBDRolloutStepInput] = []
        self.return_wrong_velocity = False

    @property
    def scene(self) -> TetBenchmarkScene:
        return self._scene

    @property
    def resource_identity(self) -> tuple[object, ...]:
        return self._resource_identity

    def solve_step(self, step_input: MGVBDRolloutStepInput) -> MGVBDRolloutStepEndpoint:
        self.inputs.append(step_input)
        endpoint = _deterministic_endpoint(self._scene, step_input)
        if self.return_wrong_velocity:
            velocities = np.zeros_like(endpoint.velocities)
            return dataclasses.replace(endpoint, velocities=velocities)
        return endpoint


def _assert_same_state(
    test: unittest.TestCase,
    left,
    right,
) -> None:
    test.assertEqual(left.step_index, right.step_index)
    test.assertEqual(left.time_seconds, right.time_seconds)
    test.assertEqual(left.accepted, right.accepted)
    test.assertEqual(left.reasons, right.reasons)
    test.assertEqual(left.graph_replay, right.graph_replay)
    for name in (
        "positions",
        "velocities",
        "external_force",
        "inertial_target",
        "pinned_indices",
        "pin_targets",
    ):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


class TestMGVBDRollout(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _scene()
        self.backend = _DeterministicBackend(self.scene)
        self.rollout = MGVBDRollout(self.scene, self.backend)

    def test_initial_state_is_exactly_pinned_and_immutable(self):
        state = self.rollout.state
        self.assertEqual(state.contract_id, CONTRACT_ID)
        self.assertEqual(state.step_index, 0)
        self.assertEqual(state.time_seconds, 0.0)
        self.assertEqual(state.accepted, ())
        self.assertFalse(state.graph_replay)
        np.testing.assert_array_equal(state.positions[state.pinned_indices], state.pin_targets)
        np.testing.assert_array_equal(state.velocities[state.pinned_indices], 0.0)
        np.testing.assert_array_equal(state.inertial_target, self.scene.vbd_inertial_target)
        for name in (
            "positions",
            "velocities",
            "external_force",
            "inertial_target",
            "pinned_indices",
            "pin_targets",
        ):
            self.assertTrue(memoryview(getattr(state, name)).readonly)
        with self.assertRaises(ValueError):
            state.positions[0, 0] = 7.0

    def test_step_propagates_exact_endpoint_and_persistent_force(self):
        force = np.linspace(-2.0, 3.0, self.scene.n_vertices * 3, dtype=np.float64).reshape(-1, 3)
        self.rollout.set_external_force(force)
        first = self.rollout.step(self.scene.dt)
        second = self.rollout.step(self.scene.dt)

        self.assertEqual(len(self.backend.inputs), 2)
        np.testing.assert_array_equal(self.backend.inputs[0].external_force, force.astype(np.float32))
        np.testing.assert_array_equal(self.backend.inputs[1].external_force, force.astype(np.float32))
        np.testing.assert_array_equal(self.backend.inputs[1].positions, first.positions)
        np.testing.assert_array_equal(self.backend.inputs[1].velocities, first.velocities)
        np.testing.assert_array_equal(second.external_force, force.astype(np.float32))
        np.testing.assert_array_equal(
            first.velocities,
            _published_velocity(
                first.positions, self.backend.inputs[0].positions, self.scene.pinned_indices, self.scene.dt
            ),
        )
        self.assertEqual(first.step_index, 1)
        self.assertEqual(second.step_index, 2)
        self.assertEqual(first.accepted, (True, True, False, False))
        self.assertEqual(first.reasons[-1], "masked-after-rejection")
        self.assertTrue(first.graph_replay)
        self.assertEqual(self.rollout.resource_identity, self.backend.resource_identity)

    def test_moving_pin_targets_preserve_positions_and_zero_velocities(self):
        targets = self.scene.pin_targets.copy()
        targets[:, 1] += np.float64(0.125)
        targets[:, 2] -= np.float64(0.0625)
        self.rollout.set_pin_targets(targets)
        before = self.rollout.state
        result = self.rollout.step(self.scene.dt)

        expected_targets = targets.astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(before.positions[self.scene.pinned_indices], expected_targets)
        np.testing.assert_array_equal(before.velocities[self.scene.pinned_indices], 0.0)
        np.testing.assert_array_equal(self.backend.inputs[-1].pin_targets, expected_targets)
        np.testing.assert_array_equal(
            self.backend.inputs[-1].inertial_target[self.scene.pinned_indices],
            expected_targets,
        )
        np.testing.assert_array_equal(result.positions[self.scene.pinned_indices], expected_targets)
        np.testing.assert_array_equal(result.velocities[self.scene.pinned_indices], 0.0)

    def test_reset_replay_is_bitwise_deterministic(self):
        schedules = tuple(
            np.full((self.scene.n_vertices, 3), fill_value=value, dtype=np.float64) for value in (0.25, -0.5, 1.25)
        )

        def run_schedule():
            states = []
            for force in schedules:
                self.rollout.set_external_force(force)
                states.append(self.rollout.step(self.scene.dt))
            return tuple(states)

        first = run_schedule()
        reset_state = self.rollout.reset()
        np.testing.assert_array_equal(reset_state.external_force, self.scene.external_force)
        np.testing.assert_array_equal(reset_state.pin_targets, self.scene.pin_targets)
        second = run_schedule()

        self.assertEqual(self.backend.resource_identity, self.rollout.resource_identity)
        for left, right in zip(first, second, strict=True):
            _assert_same_state(self, left, right)

    def test_matches_manually_reconstructed_one_step_sequence(self):
        positions = np.array(self.scene.x_current, dtype=np.float32).astype(np.float64)
        velocities = np.array(self.scene.velocity, dtype=np.float32).astype(np.float64)
        pin_targets = np.array(self.scene.pin_targets, dtype=np.float32).astype(np.float64)
        positions[self.scene.pinned_indices] = pin_targets
        velocities[self.scene.pinned_indices] = 0.0

        for index, scale in enumerate((0.75, -0.25, 1.5)):
            force = np.array(self.scene.external_force * scale, dtype=np.float32).astype(np.float64)
            pin_targets = pin_targets.copy()
            pin_targets[:, 2] = (pin_targets[:, 2] + np.float64(0.01 * index)).astype(np.float32)
            positions = positions.copy()
            velocities = velocities.copy()
            positions[self.scene.pinned_indices] = pin_targets
            velocities[self.scene.pinned_indices] = 0.0
            manual_scene = dataclasses.replace(
                self.scene,
                x_current=positions,
                velocity=velocities,
                external_force=force,
                pin_targets=pin_targets,
            )
            manual_input = MGVBDRolloutStepInput(
                step_index=index,
                dt=manual_scene.dt,
                positions=manual_scene.x_current,
                velocities=manual_scene.velocity,
                external_force=manual_scene.external_force,
                inertial_target=manual_scene.vbd_inertial_target,
                pinned_indices=manual_scene.pinned_indices,
                pin_targets=manual_scene.pin_targets,
            )
            manual_endpoint = _deterministic_endpoint(self.scene, manual_input)

            self.rollout.set_pin_targets(pin_targets)
            self.rollout.set_external_force(force)
            actual = self.rollout.step(self.scene.dt)
            np.testing.assert_array_equal(actual.positions, manual_endpoint.positions)
            np.testing.assert_array_equal(actual.velocities, manual_endpoint.velocities)
            np.testing.assert_array_equal(self.backend.inputs[-1].inertial_target, manual_scene.vbd_inertial_target)
            positions = manual_endpoint.positions
            velocities = manual_endpoint.velocities

    def test_invalid_dt_does_not_launch_or_advance(self):
        before = self.rollout.state
        with self.assertRaisesRegex(ValueError, "timestep is fixed"):
            self.rollout.step(self.scene.dt * 0.5)
        after = self.rollout.state
        self.assertEqual(self.backend.inputs, [])
        _assert_same_state(self, before, after)

    def test_wrong_backend_velocity_does_not_advance(self):
        before = self.rollout.state
        self.backend.return_wrong_velocity = True
        with self.assertRaisesRegex(ValueError, "exact BDF1"):
            self.rollout.step(self.scene.dt)
        after = self.rollout.state
        self.assertEqual(len(self.backend.inputs), 1)
        _assert_same_state(self, before, after)

    def test_replaced_backend_resource_is_rejected_before_launch(self):
        before = self.rollout.state
        self.backend._resource_identity = ("replacement",)
        with self.assertRaisesRegex(RuntimeError, "resources changed"):
            self.rollout.reset()
        with self.assertRaisesRegex(RuntimeError, "resources changed"):
            self.rollout.step(self.scene.dt)
        self.assertEqual(self.backend.inputs, [])
        _assert_same_state(self, before, self.rollout.state)

    def test_captured_backend_uses_transactional_graph_replay(self):
        source = inspect.getsource(MGVBDRolloutCapturedBackend.solve_step)
        validation = source.index("self._solver._validate_persistent_sources")
        assignment = source.index("self._assign_dynamic_inputs(step_input)")
        self.assertLess(validation, assignment)
        self.assertIn("wp.capture_launch(graph, stream=replay_stream)", source)
        self.assertIn("self._restore_one_step_inputs()", source)
        self.assertEqual(source.count("wp.synchronize_stream(replay_stream)"), 1)
        self.assertNotIn("self._solver.run(", source)

        identity_source = inspect.getsource(MGVBDRolloutCapturedBackend._live_resource_identity)
        self.assertIn("_lookup_workspace_owners(self._solver)", identity_source)
        self.assertIn("self._solver._validate_workspace_owner_bindings(owner_binding)", identity_source)


@unittest.skipUnless(os.environ.get("MG_VBD_TEST_CUDA") == "1", "set MG_VBD_TEST_CUDA=1 after claiming a GPU")
class TestMGVBDRolloutCuda(unittest.TestCase):
    def setUp(self) -> None:
        if wp.get_cuda_device_count() < 1:
            raise unittest.SkipTest("no claimed CUDA device is visible")
        self.scene = build_structured_cantilever_scene(
            dimensions=(6, 3, 3),
            dt=1.0 / 16.0,
            gravity=(0.1, -0.2, -2.0),
            total_tip_force=(4.0, -3.0, -6.0),
            initial_velocity=(0.03, -0.02, 0.01),
            name="captured-direct-graph-vbd-tiny",
        )
        self.solver = CapturedDirectGraphVBD(self.scene, device="cuda:0", tile_solve=False)
        self.solver.capture_graphs(warmup_replays=1)
        self.backend = MGVBDRolloutCapturedBackend(self.solver)
        self.rollout = MGVBDRollout(self.scene, self.backend)

    def test_static_device_mass_tamper_rejects_without_advancing_rollout(self):
        binding = _lookup_workspace_owners(self.solver)
        original_mass = np.array(binding.operator.mass.numpy(), copy=True)
        tampered_mass = original_mass.copy()
        tampered_mass[self.scene.free_indices[0]] *= np.float64(1.125)
        before = self.rollout.state
        final_positions = np.array(binding.direct.final_positions.numpy(), copy=True)
        try:
            binding.operator.mass.assign(tampered_mass)
            with self.assertRaisesRegex(RuntimeError, "persistent operator.mass changed"):
                self.rollout.step(self.scene.dt)
            _assert_same_state(self, before, self.rollout.state)
            np.testing.assert_array_equal(binding.direct.final_positions.numpy(), final_positions)
        finally:
            binding.operator.mass.assign(original_mass)
        self.solver._validate_persistent_sources(owner_binding=binding)

    def test_recapture_changes_live_identity_and_rejects_reset_and_step(self):
        before = self.rollout.state
        original_identity = self.backend.resource_identity
        self.solver.capture_graphs(warmup_replays=1)
        self.assertNotEqual(self.backend.resource_identity, original_identity)
        with self.assertRaisesRegex(RuntimeError, "resources changed"):
            self.rollout.reset()
        with self.assertRaisesRegex(RuntimeError, "resources changed"):
            self.rollout.step(self.scene.dt)
        _assert_same_state(self, before, self.rollout.state)


if __name__ == "__main__":
    unittest.main()
