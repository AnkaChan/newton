# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic free-body reference rollout shards."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from .. import reference_rollout as rollout_module
from ..reference_rollout import (
    FreeBodyReferenceProtocol,
    build_free_body_scene,
    normalize_initial_state,
    repair_tet_orientation,
    run_reference_rollout,
    write_reference_rollout_shard,
)
from ..solver_benchmark import CommonStateMetrics, build_common_problem, common_objective_manifest


def _unit_tet() -> tuple[np.ndarray, np.ndarray]:
    rest = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return rest, np.array(((0, 1, 2, 3),), dtype=np.int64)


def _normalized_state():
    rest, tets = _unit_tet()
    deformed = rest + np.array((0.2, -0.1, 0.05))
    velocity = np.array(
        (
            (0.1, 0.0, 0.0),
            (0.1, 0.1, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, 0.1),
        ),
        dtype=np.float64,
    )
    return normalize_initial_state(rest, deformed, velocity, tets)


def _metrics(
    positions: np.ndarray,
    *,
    objective: float,
    relative_residual: float,
    determinant_min: float = 0.8,
    minimum_singular_value: float = 0.7,
) -> CommonStateMetrics:
    return CommonStateMetrics(
        objective=objective,
        inertia=0.25 * objective,
        elastic=0.75 * objective,
        gradient_norm=relative_residual,
        relative_residual=relative_residual,
        determinant_min=determinant_min,
        determinant_max=1.2,
        inverted_tet_fraction=0.0 if determinant_min > 0.0 else 1.0,
        minimum_singular_value=minimum_singular_value,
        free_rms_error_m=None,
        mass_weighted_rms_error_m=None,
        max_pin_error_m=0.0,
        position_sha256=rollout_module._array_digest(np.asarray(positions, dtype=np.float64)),
    )


class TestFreeBodyReferenceProtocol(unittest.TestCase):
    def test_default_protocol_is_the_registered_pilot(self):
        protocol = FreeBodyReferenceProtocol()
        self.assertEqual(protocol.requested_dt_seconds, 1.0 / 300.0)
        self.assertEqual(protocol.execution_dt_seconds, float(np.float32(1.0 / 300.0)))
        self.assertEqual(protocol.execution_dt_float32_bits, "0x3b5a740e")
        self.assertEqual(protocol.normalized_characteristic_length_m, 1.0)
        self.assertEqual(protocol.density_kg_m3, 1000.0)
        self.assertEqual(protocol.shear_modulus_pa, 1.0e4)
        self.assertEqual(protocol.linear_lame_lambda_pa, 1.0e5)
        self.assertEqual(protocol.vbd_stored_lambda_pa, 1.1e5)
        self.assertEqual(protocol.tet_damping, 0.0)
        self.assertEqual(protocol.gravity_m_s2, (0.0, 0.0, 0.0))
        self.assertEqual(protocol.iteration_budgets, (20, 50, 100))
        self.assertEqual(protocol.rollout_steps, 8)
        self.assertEqual(protocol.maximum_relative_residual, 2.0e-2)
        self.assertEqual(protocol.maximum_residual_ratio, 1.0e-1)

        record = protocol.as_dict()
        self.assertEqual(record["lambda_convention"], "vbd_stored_lambda=mu+linear_lame_lambda")
        self.assertEqual(record["boundary_condition"], "free-body-no-pins")
        self.assertEqual(record["contact"], False)

    def test_protocol_rejects_drift_from_the_fixed_pilot(self):
        with self.assertRaisesRegex(ValueError, "exactly 1/300"):
            FreeBodyReferenceProtocol(requested_dt_seconds=1.0 / 120.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            FreeBodyReferenceProtocol(iteration_budgets=(20, 20, 100))
        with self.assertRaisesRegex(ValueError, "zero gravity"):
            FreeBodyReferenceProtocol(gravity_m_s2=(0.0, 0.0, -9.81))


class TestInitialStateCanonicalization(unittest.TestCase):
    def test_orientation_repair_changes_only_negative_tet_corner_order(self):
        rest, positive = _unit_tet()
        negative = positive[:, (0, 2, 1, 3)]
        mixed = np.concatenate((positive, negative), axis=0)

        repaired = repair_tet_orientation(rest, mixed)

        self.assertEqual(repaired.repaired_tet_indices, (1,))
        np.testing.assert_array_equal(repaired.tet_indices[0], positive[0])
        changed_corners = np.flatnonzero(repaired.tet_indices[1] != mixed[1])
        np.testing.assert_array_equal(changed_corners, np.array((2, 3)))
        corners = rest[repaired.tet_indices]
        matrices = np.stack(
            (corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]),
            axis=-1,
        )
        self.assertTrue(np.all(np.linalg.det(matrices) > 0.0))
        self.assertFalse(repaired.tet_indices.flags["W"])
        np.testing.assert_array_equal(mixed[1], negative[0])

    def test_orientation_repair_rejects_degenerate_tets(self):
        rest, tets = _unit_tet()
        rest[3] = rest[2]
        with self.assertRaisesRegex(ValueError, "degenerate"):
            repair_tet_orientation(rest, tets)

    def test_normalization_centers_rest_scale_and_velocity_consistently(self):
        rest, tets = _unit_tet()
        source_center = rest.mean(axis=0)
        source_length = float(np.linalg.norm(np.ptp(rest, axis=0)))
        translation = np.array((3.0, -2.0, 1.0))
        deformed = rest + translation
        velocity = np.broadcast_to(np.array((source_length, 0.0, 0.0)), rest.shape)

        state = normalize_initial_state(rest, deformed, velocity, tets)

        np.testing.assert_allclose(state.source_center, source_center, rtol=0.0, atol=0.0)
        self.assertEqual(state.source_characteristic_length, source_length)
        self.assertAlmostEqual(float(np.linalg.norm(np.ptp(state.rest_q, axis=0))), 1.0, delta=1.0e-7)
        np.testing.assert_allclose(state.rest_q.mean(axis=0), 0.0, rtol=0.0, atol=2.0e-8)
        np.testing.assert_allclose(
            state.x_initial - state.rest_q,
            np.broadcast_to(translation / source_length, rest.shape),
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(state.velocity_initial[:, 0], 1.0, rtol=0.0, atol=0.0)
        self.assertEqual(state.rest_q.dtype, np.float32)
        self.assertEqual(state.tet_indices.dtype, np.int32)
        self.assertFalse(state.x_initial.flags["W"])
        self.assertEqual(state.orientation_repaired_count, 0)


class TestFreeBodyScene(unittest.TestCase):
    def test_scene_uses_positive_active_mass_and_no_pins(self):
        protocol = FreeBodyReferenceProtocol(rollout_steps=1)
        scene = build_free_body_scene(
            _normalized_state(),
            protocol=protocol,
            device="cpu",
            asset_id="unit-tet",
            source="synthetic",
            source_sha256="a" * 64,
            deformation_seed=17,
            velocity_seed=29,
        )

        self.assertEqual(scene.template.n_vertices, 4)
        self.assertEqual(scene.template.n_tets, 1)
        self.assertEqual(scene.model.edge_count, 0)
        self.assertGreater(scene.model.tri_count, 0)
        self.assertTrue(np.all(scene.template.mass > 0.0))
        self.assertEqual(scene.template.pinned_indices.shape, (0,))
        self.assertEqual(scene.template.pin_targets.shape, (0, 3))
        active = int(rollout_module.newton.ParticleFlags.ACTIVE)
        self.assertTrue(np.all((scene.template.particle_flags & active) != 0))
        np.testing.assert_array_equal(scene.template.tet_materials[:, 0], protocol.shear_modulus_pa)
        np.testing.assert_array_equal(scene.template.tet_materials[:, 1], protocol.vbd_stored_lambda_pa)
        np.testing.assert_array_equal(scene.template.tet_materials[:, 2], 0.0)
        np.testing.assert_array_equal(scene.template.gravity, 0.0)
        np.testing.assert_array_equal(scene.template.external_force, 0.0)

        objective = common_objective_manifest(scene.template, build_common_problem(scene.template))
        self.assertRegex(objective["objective_instance_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(objective["derived_arrays"]["pin_targets"]["shape"], [0, 3])


class TestReferenceRollout(unittest.TestCase):
    def _scene(self, *, steps: int = 2):
        return build_free_body_scene(
            _normalized_state(),
            protocol=FreeBodyReferenceProtocol(rollout_steps=steps),
            device="cpu",
            asset_id="unit-tet",
            source="synthetic",
            source_sha256="b" * 64,
            deformation_seed=101,
            velocity_seed=202,
        )

    def test_reused_runner_final_budget_is_bitwise_equal_to_a_fresh_runner(self):
        scene = self._scene(steps=1)
        positions = scene.template.x_current
        velocities = scene.template.velocity
        force = np.zeros_like(positions, dtype=np.float32)

        reused = rollout_module._ReusableVBDRunner(scene)
        reused.solve(positions, velocities, force, 20)
        reused.solve(positions, velocities, force, 50)
        reused_final = reused.solve(positions, velocities, force, 100)
        fresh_final = rollout_module._ReusableVBDRunner(scene).solve(
            positions,
            velocities,
            force,
            100,
        )

        self.assertEqual(reused_final.positions.tobytes(), fresh_final.positions.tobytes())
        self.assertEqual(reused_final.velocities.tobytes(), fresh_final.velocities.tobytes())
        self.assertEqual(reused_final.position_float32_sha256, fresh_final.position_float32_sha256)
        self.assertEqual(reused_final.velocity_float32_sha256, fresh_final.velocity_float32_sha256)

    def test_fresh_budget_restarts_share_one_solver_and_only_final_candidate_commits(self):
        scene = self._scene(steps=2)
        protocol = scene.protocol
        runtime_instances: list[object] = []

        class FakeRuntime:
            def __init__(self, supplied_scene):
                self.scene = supplied_scene
                self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []
                runtime_instances.append(self)

            def solve(self, positions, velocities, external_force, iterations):
                self.calls.append((positions.copy(), velocities.copy(), iterations))
                delta = np.float32(iterations) * np.float32(1.0e-6)
                input_positions = np.asarray(positions, dtype=np.float32)
                output_positions = np.asarray(input_positions + delta, dtype=np.float32)
                output_velocities = np.asarray(
                    (output_positions - input_positions) / np.float32(protocol.execution_dt_seconds),
                    dtype=np.float32,
                )
                return rollout_module._CandidateState(
                    positions=output_positions,
                    velocities=output_velocities,
                    effective_tile_solve=False,
                )

        per_budget_residual = {20: 0.10, 50: 0.04, 100: 0.01}
        evaluation_count = 0

        def evaluate(problem, positions):
            nonlocal evaluation_count
            values = np.asarray(positions, dtype=np.float64)
            slot = evaluation_count % (len(protocol.iteration_budgets) + 1)
            evaluation_count += 1
            if slot == 0:
                return _metrics(values, objective=10.0, relative_residual=1.0)
            budget = protocol.iteration_budgets[slot - 1]
            return _metrics(
                values,
                objective=9.0 - budget * 1.0e-3,
                relative_residual=per_budget_residual[budget],
            )

        with (
            mock.patch.object(rollout_module, "_ReusableVBDRunner", FakeRuntime),
            mock.patch.object(rollout_module, "evaluate_common_state", side_effect=evaluate),
        ):
            result = run_reference_rollout(scene)

        self.assertEqual(len(runtime_instances), 1)
        calls = runtime_instances[0].calls
        self.assertEqual([item[2] for item in calls], [20, 50, 100, 20, 50, 100])
        for step in range(2):
            starts = [item[0] for item in calls[step * 3 : (step + 1) * 3]]
            np.testing.assert_array_equal(starts[0], starts[1])
            np.testing.assert_array_equal(starts[1], starts[2])
        self.assertEqual(result.q.shape, (3, 4, 3))
        self.assertEqual(result.qd.shape, (3, 4, 3))
        self.assertEqual(result.inertial_target.shape, (2, 4, 3))
        self.assertEqual(result.q.dtype, np.float64)
        self.assertEqual(result.qd.dtype, np.float64)
        self.assertEqual(result.inertial_target.dtype, np.float64)
        self.assertEqual(len(result.steps), 2)
        self.assertTrue(all(step.reference_accepted for step in result.steps))
        self.assertTrue(all(step.selected_iterations == 100 for step in result.steps))
        self.assertTrue(all(len(step.candidates) == 3 for step in result.steps))
        self.assertTrue(all(step.reference_failures == () for step in result.steps))
        self.assertTrue(all(len(step.dynamic_scene_sha256) == 64 for step in result.steps))
        self.assertTrue(all(len(step.objective_instance_sha256) == 64 for step in result.steps))
        expected_first = (scene.template.x_current.astype(np.float32) + np.float32(100.0e-6)).astype(np.float64)
        np.testing.assert_array_equal(result.q[1], expected_first)
        self.assertFalse(result.q.flags["W"])

        with self.assertRaisesRegex(ValueError, "objective identity"):
            dataclasses.replace(result.steps[0], objective_instance_sha256="0" * 64)

    def test_independent_gate_marks_candidate_only_and_stops_sequence(self):
        scene = self._scene(steps=3)
        protocol = scene.protocol

        class FakeRuntime:
            def __init__(self, _scene):
                pass

            def solve(self, positions, velocities, external_force, iterations):
                input_positions = np.asarray(positions, dtype=np.float32)
                output_positions = np.asarray(input_positions + np.float32(iterations * 1.0e-6), dtype=np.float32)
                output_velocities = np.asarray(
                    (output_positions - input_positions) / np.float32(protocol.execution_dt_seconds),
                    dtype=np.float32,
                )
                return rollout_module._CandidateState(output_positions, output_velocities, False)

        failures = rollout_module._reference_failures(
            _metrics(scene.template.x_current, objective=10.0, relative_residual=0.1),
            _metrics(scene.template.x_current, objective=9.0, relative_residual=0.015),
            protocol=protocol,
            exact_velocity_commit=True,
        )
        self.assertIn("residual ratio", " ".join(failures))

        evaluation_count = 0

        def evaluate(problem, positions):
            nonlocal evaluation_count
            values = np.asarray(positions, dtype=np.float64)
            is_iterate_zero = evaluation_count == 0
            evaluation_count += 1
            if is_iterate_zero:
                return _metrics(values, objective=10.0, relative_residual=1.0)
            return _metrics(values, objective=9.0, relative_residual=0.03)

        with (
            mock.patch.object(rollout_module, "_ReusableVBDRunner", FakeRuntime),
            mock.patch.object(rollout_module, "evaluate_common_state", side_effect=evaluate),
        ):
            result = run_reference_rollout(scene)

        self.assertEqual(len(result.steps), 1)
        self.assertFalse(result.steps[0].reference_accepted)
        self.assertIn("relative residual", " ".join(result.steps[0].reference_failures))
        self.assertEqual(result.q.shape[0], 2)
        self.assertFalse(result.reference_accepted)


class TestReferenceShard(unittest.TestCase):
    def _accepted_rollout(self, *, source: str = "synthetic"):
        scene = build_free_body_scene(
            _normalized_state(),
            protocol=FreeBodyReferenceProtocol(rollout_steps=1, iteration_budgets=(100,)),
            device="cpu",
            asset_id="unit-tet",
            source=source,
            source_sha256="c" * 64,
            deformation_seed=303,
            velocity_seed=404,
        )
        protocol = scene.protocol

        class FakeRuntime:
            def __init__(self, _scene):
                pass

            def solve(self, positions, velocities, external_force, iterations):
                input_positions = np.asarray(positions, dtype=np.float32)
                output_positions = np.asarray(input_positions + np.float32(1.0e-4), dtype=np.float32)
                output_velocities = np.asarray(
                    (output_positions - input_positions) / np.float32(protocol.execution_dt_seconds),
                    dtype=np.float32,
                )
                return rollout_module._CandidateState(output_positions, output_velocities, False)

        evaluation_count = 0

        def evaluate(problem, positions):
            nonlocal evaluation_count
            values = np.asarray(positions, dtype=np.float64)
            is_iterate_zero = evaluation_count == 0
            evaluation_count += 1
            if is_iterate_zero:
                return _metrics(values, objective=10.0, relative_residual=1.0)
            return _metrics(values, objective=9.0, relative_residual=0.01)

        with (
            mock.patch.object(rollout_module, "_ReusableVBDRunner", FakeRuntime),
            mock.patch.object(rollout_module, "evaluate_common_state", side_effect=evaluate),
        ):
            return run_reference_rollout(scene)

    def test_shard_is_byte_deterministic_and_contains_exact_training_arrays(self):
        result = self._accepted_rollout()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            first = write_reference_rollout_shard(root / "first", result, sequence_id="sample-000")
            second = write_reference_rollout_shard(root / "second", result, sequence_id="sample-000")

            for name in ("static_npz", "sequence_npz", "evidence_json", "manifest_json"):
                self.assertEqual(getattr(first, name).read_bytes(), getattr(second, name).read_bytes())

            with np.load(first.sequence_npz, allow_pickle=False) as arrays:
                self.assertEqual(
                    set(arrays.files),
                    {
                        "deformation_seed",
                        "dt",
                        "external_force",
                        "gravity",
                        "inertial_target",
                        "pin_targets",
                        "pinned_indices",
                        "q",
                        "qd",
                        "step_ids",
                        "velocity_seed",
                    },
                )
                np.testing.assert_array_equal(arrays["q"], result.q)
                np.testing.assert_array_equal(arrays["qd"], result.qd)
                self.assertEqual(arrays["q"].dtype, np.float64)
                self.assertEqual(arrays["qd"].dtype, np.float64)
                self.assertEqual(arrays["inertial_target"].dtype, np.float64)
                self.assertEqual(arrays["dt"].dtype, np.float32)
                self.assertEqual(arrays["pinned_indices"].shape, (0,))
                self.assertEqual(arrays["pin_targets"].shape, (1, 0, 3))

            manifest = json.loads(first.manifest_json.read_text())
            self.assertEqual(manifest["schema"], "pss-free-body-reference-shard-v1")
            self.assertTrue(manifest["reference_accepted"])
            self.assertEqual(manifest["sequence_id"], "sample-000")
            self.assertEqual(manifest["protocol"]["execution_dt_float32_bits"], "0x3b5a740e")
            for name in (
                "material_sha256",
                "operator_sha256",
                "protocol_sha256",
                "topology_sha256",
            ):
                self.assertRegex(manifest["identities"][name], r"^[0-9a-f]{64}$")
            self.assertEqual(manifest["initial_scene_sha256"], manifest["initial_scene"]["scene_sha256"])
            evidence = json.loads(first.evidence_json.read_text())
            self.assertTrue(evidence["steps"][0]["reference_accepted"])
            self.assertEqual(evidence["steps"][0]["candidates"][0]["iterations"], 100)
            self.assertRegex(evidence["steps"][0]["objective_instance_sha256"], r"^[0-9a-f]{64}$")

    def test_local_source_relocation_does_not_change_canonical_shard_bytes(self):
        first_source = rollout_module._logical_source_name(pathlib.Path("/machine-a/data/unit-tet.vtk"))
        second_source = rollout_module._logical_source_name(pathlib.Path("/machine-b/cache/unit-tet.vtk"))
        self.assertEqual(first_source, second_source)
        first_result = self._accepted_rollout(source=first_source)
        second_result = self._accepted_rollout(source=second_source)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            first = write_reference_rollout_shard(root / "first", first_result, sequence_id="sample-000")
            second = write_reference_rollout_shard(root / "second", second_result, sequence_id="sample-000")
            for name in ("static_npz", "sequence_npz", "evidence_json", "manifest_json"):
                self.assertEqual(getattr(first, name).read_bytes(), getattr(second, name).read_bytes())

    def test_rejected_rollout_is_never_written_as_training_data(self):
        accepted = self._accepted_rollout()
        rejected_step = types.SimpleNamespace(reference_accepted=False)
        rejected = types.SimpleNamespace(
            **{name: getattr(accepted, name) for name in accepted.__dataclass_fields__ if name != "steps"},
            steps=(rejected_step,),
            reference_accepted=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = pathlib.Path(temporary_directory) / "rejected"
            with self.assertRaisesRegex(ValueError, "not accepted"):
                write_reference_rollout_shard(destination, rejected, sequence_id="sample-000")
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
