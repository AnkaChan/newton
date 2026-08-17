# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import numpy as np

from .. import pr_scene_history as history


class TestPRSceneHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stretch = history.create_pr_scene_history("stretch")
        cls.twist = history.create_pr_scene_history("twist")
        cls.compression_50 = history.create_pr_scene_history("compression-50")
        cls.compression_90 = history.create_pr_scene_history("compression-90")

    @staticmethod
    def _state_at(builder, coordinate, *, q=None, qd=None, flags=None):
        initial = builder.initial_checkpoint.state
        return history.CommittedState(
            manifest_sha256=builder.manifest.manifest_sha256,
            coordinate=coordinate,
            q=initial.q if q is None else q,
            qd=initial.qd if qd is None else qd,
            particle_flags=initial.particle_flags if flags is None else flags,
        )

    @staticmethod
    def _accepted_reference(scene, _config):
        positions = scene.vbd_inertial_target.copy()
        return history._ReferenceStep(
            positions=positions,
            accepted=True,
            failures=(),
            deterministic_record={
                "method": "unittest-exact-inertial-target",
                "accepted": True,
                "position_sha256": history._array_digest(positions),
            },
            timing_record={"seconds": 0.125},
        )

    @staticmethod
    def _stalled_reference(scene, config, *, failures=None):
        positions = scene.vbd_inertial_target.copy()
        rejected_failures = (
            "native termination: stalled",
            "independent gradient 5.000e-07 N exceeds 2.000e-09 N",
            "verification termination: stalled",
            "verification displacement 2.000e-12 exceeds 1e-12",
        )
        if failures is not None:
            rejected_failures = failures
        return history._ReferenceStep(
            positions=positions,
            accepted=False,
            failures=rejected_failures,
            deterministic_record={
                "method": "unittest-stalled-reference",
                "config": dataclasses.asdict(config),
                "scene_sha256": "1" * 64,
                "objective_instance_sha256": "2" * 64,
                "accepted": False,
                "failures": list(rejected_failures),
                "native_converged": False,
                "native_reason": "stalled",
                "final_objective": 5.0,
                "final_gradient_norm": 5.0e-7,
                "final_relative_residual": 2.5e-8,
                "verification_displacement_relative": 2.0e-12,
                "alternate_start_displacement_relative": 4.0e-12,
                "alternate_start_gradient_norm": 1.0e-11,
                "alternate_start_relative_residual": 5.0e-13,
                "position_sha256": history._array_digest(positions),
            },
            timing_record={"seconds": 0.25},
        )

    def test_exact_schedule_endpoints(self):
        self.assertEqual(self.stretch.frame_schedule(0).value, 1.0)
        self.assertEqual(self.stretch.frame_schedule(1).value, 1.0 + 1.0 / 200.0)
        self.assertEqual(self.stretch.frame_schedule(199).value, 1.995)
        self.assertEqual(self.stretch.frame_schedule(200).value, 2.0)
        self.assertEqual(self.stretch.frame_schedule(399).value, 2.0)

        self.assertEqual(self.twist.frame_schedule(0).value, 0.0)
        self.assertEqual(self.twist.frame_schedule(1).value, 2.0 * np.pi / 200.0)
        self.assertEqual(self.twist.frame_schedule(200).value, 2.0 * np.pi)

        self.assertEqual(self.compression_50.frame_schedule(0).value, 1.0)
        self.assertEqual(self.compression_50.frame_schedule(149).value, 0.5)
        self.assertEqual(
            self.compression_90.frame_schedule(149).value,
            1.0 - (149 / 149) * (1.0 - 0.1),
        )
        self.assertEqual(self.compression_50.frame_schedule(150).action, "release")
        self.assertEqual(self.compression_50.frame_schedule(151).action, "none")
        self.assertEqual(self.stretch.manifest.as_dict()["dt_float32_bits"], "0x3b5a740e")
        self.assertEqual(self.stretch.manifest.transition_count, 2000)

    def test_callback_is_constant_inside_each_five_substep_frame(self):
        frame_start = self._state_at(self.stretch, history.AtomicCoordinate(1, 0))
        applied_start = self.stretch.apply_callback(frame_start)
        self.assertTrue(applied_start.callback_applied)
        self.assertEqual(applied_start.action, "drive")

        inside_frame = self._state_at(
            self.stretch,
            history.AtomicCoordinate(1, 1),
            q=applied_start.q,
            flags=applied_start.particle_flags,
        )
        applied_inside = self.stretch.apply_callback(inside_frame)
        self.assertFalse(applied_inside.callback_applied)
        self.assertEqual(applied_inside.action, "none")
        np.testing.assert_array_equal(applied_inside.q, applied_start.q)
        np.testing.assert_array_equal(applied_inside.pin_targets, applied_start.pin_targets)

        last_inside = dataclasses.replace(inside_frame, coordinate=history.AtomicCoordinate(1, 4))
        applied_last = self.stretch.apply_callback(last_inside)
        np.testing.assert_array_equal(applied_last.pin_targets, applied_start.pin_targets)
        self.assertEqual(last_inside.coordinate.next(), history.AtomicCoordinate(2, 0))

    def test_compression_release_changes_flags_without_position_overwrite(self):
        q = self.compression_50.initial_checkpoint.state.q.copy()
        driven = self.compression_50._driven
        q[driven, 0] += np.float32(0.0125)
        q[driven, 2] = np.float32(1.173)
        before = self._state_at(self.compression_50, history.AtomicCoordinate(150, 0), q=q)
        applied = self.compression_50.apply_callback(before)

        self.assertEqual(applied.action, "release")
        np.testing.assert_array_equal(applied.q, q)
        self.assertTrue(np.all((applied.particle_flags[driven] & history._ACTIVE_FLAG) != 0))
        self.assertEqual(np.intersect1d(applied.pinned_indices, driven).size, 0)
        bottom = np.where(np.isclose(q[:, 2], np.float32(1.0), rtol=0.0, atol=1.0e-6))[0]
        np.testing.assert_array_equal(applied.pinned_indices, bottom)

    def test_full_turn_float32_pin_target_hash(self):
        state = self._state_at(self.twist, history.AtomicCoordinate(200, 0))
        applied = self.twist.apply_callback(state)
        self.assertEqual(applied.schedule_value, 2.0 * np.pi)
        self.assertEqual(
            history._array_digest(applied.pin_targets),
            "7799f6e76d08ba923786d942e4575f2d3489ab4220788f93401aa7f182369768",
        )

    def test_velocity_commit_uses_float32_vbd_operation_order(self):
        state = self._state_at(self.stretch, history.AtomicCoordinate(1, 0))
        applied = self.stretch.apply_callback(state)
        reference = applied.q.astype(np.float64)
        free = np.setdiff1d(np.arange(reference.shape[0]), applied.pinned_indices)
        reference[free[0]] += np.array([1.0e-4, -2.0e-4, 3.0e-4])
        committed = self.stretch._commit_reference(state, applied, reference)

        expected_q = reference.astype(np.float32)
        expected_qd = (expected_q - applied.q) / np.float32(self.stretch.manifest.dt_seconds)
        expected_qd[applied.pinned_indices] = np.float32(0.0)
        np.testing.assert_array_equal(committed.q, expected_q)
        np.testing.assert_array_equal(committed.qd, expected_qd)
        np.testing.assert_array_equal(
            committed.qd[applied.pinned_indices],
            np.zeros((applied.pinned_indices.size, 3), dtype=np.float32),
        )

    def test_chain_rejects_tampering_and_transition_reordering(self):
        with mock.patch.object(history, "_solve_dense_reference", side_effect=self._accepted_reference):
            chain = self.stretch.generate(
                stop=history.AtomicCoordinate(0, 2),
                max_transitions=2,
            )
        self.assertEqual(len(chain.transitions), 2)
        chain.verify()

        with self.assertRaisesRegex(ValueError, "order|disconnected"):
            dataclasses.replace(chain, transitions=tuple(reversed(chain.transitions)))

        tampered_q = chain.transitions[0].output_state.q.copy()
        tampered_q[0, 1] += np.float32(0.01)
        tampered_state = dataclasses.replace(chain.transitions[0].output_state, q=tampered_q)
        with self.assertRaisesRegex(ValueError, "float32 committed reference"):
            dataclasses.replace(chain.transitions[0], output_state=tampered_state)

        content_hash = chain.chain_sha256
        timing = dataclasses.replace(chain.timings[0], values={"seconds": 999.0})
        retimed = dataclasses.replace(chain, timings=(timing, chain.timings[1]))
        self.assertEqual(retimed.chain_sha256, content_hash)
        self.assertNotEqual(retimed.timings[0].timing_sha256, chain.timings[0].timing_sha256)

    def test_transition_exports_dynamic_training_sample(self):
        with mock.patch.object(history, "_solve_dense_reference", side_effect=self._accepted_reference):
            chain = self.compression_50.generate(max_transitions=1)
        transition = chain.transitions[0]
        arrays = transition.training_arrays()
        record = transition.training_record()

        np.testing.assert_array_equal(arrays["x_current"], chain.initial_checkpoint.state.q)
        np.testing.assert_array_equal(arrays["velocity"], chain.initial_checkpoint.state.qd)
        np.testing.assert_array_equal(arrays["pin_targets"], transition.applied_state.pin_targets)
        np.testing.assert_array_equal(arrays["inertial_target"], transition.inertial_target)
        np.testing.assert_array_equal(arrays["x_reference"], transition.reference_positions)
        self.assertEqual(record["dt_seconds"], float(np.float32(1.0 / 300.0)))
        self.assertEqual(record["topology_sha256"], self.compression_50.manifest.topology_sha256)
        self.assertEqual(record["material_sha256"], self.compression_50.manifest.material_sha256)
        self.assertFalse(arrays["x_current"].flags["W"])

        static = self.compression_50.static_bundle
        self.assertEqual(
            tuple(static.training_arrays()),
            ("rest_q", "tet_indices", "tet_poses", "mass", "tet_materials", "gravity", "external_force"),
        )
        self.assertEqual(static.manifest_sha256, transition.manifest_sha256)
        self.assertEqual(static.topology_sha256, transition.topology_sha256)
        self.assertEqual(static.material_sha256, transition.material_sha256)
        self.assertEqual(static.as_dict()["static_sha256"], static.static_sha256)
        self.assertEqual(static.rest_q.dtype, np.float32)
        self.assertEqual(static.tet_indices.dtype, np.int64)
        self.assertFalse(static.rest_q.flags["W"])

    def test_model_inputs_use_applied_current_and_exact_previous_contract(self):
        with mock.patch.object(history, "_solve_dense_reference", side_effect=self._accepted_reference):
            chain = self.stretch.generate(
                stop=history.AtomicCoordinate(1, 1),
                max_transitions=6,
            )
        transition = chain.transitions[-1]
        self.assertEqual(transition.coordinate, history.AtomicCoordinate(1, 0))
        inputs = transition.model_inputs()
        self.assertEqual(
            tuple(inputs),
            ("x_current", "x_previous", "pinned_indices", "pin_targets", "inertial_target"),
        )
        expected_displacement = (transition.input_state.qd * np.float32(transition.dt_seconds)).astype(np.float32)
        expected_previous = (transition.input_state.q - expected_displacement).astype(np.float32)
        np.testing.assert_array_equal(inputs["x_current"], transition.applied_state.q)
        np.testing.assert_array_equal(inputs["x_previous"], expected_previous)
        np.testing.assert_array_equal(inputs["pinned_indices"], transition.applied_state.pinned_indices)
        np.testing.assert_array_equal(inputs["pin_targets"], transition.applied_state.pin_targets)
        self.assertFalse(inputs["x_previous"].flags["W"])
        self.assertFalse(
            np.array_equal(
                inputs["x_current"][self.stretch._driven],
                transition.input_state.q[self.stretch._driven],
            )
        )

    def test_transition_recomputes_the_commit_contract(self):
        with mock.patch.object(history, "_solve_dense_reference", side_effect=self._accepted_reference):
            transition = self.stretch.generate(max_transitions=1).transitions[0]

        bad_qd = transition.output_state.qd.copy()
        bad_qd[transition.output_state.qd.shape[0] - 1, 0] += np.float32(0.25)
        bad_velocity_state = dataclasses.replace(transition.output_state, qd=bad_qd)
        with self.assertRaisesRegex(ValueError, "exact float32 commit formula"):
            dataclasses.replace(transition, output_state=bad_velocity_state)

        bad_flags = transition.output_state.particle_flags.copy()
        bad_flags[-1] ^= history._ACTIVE_FLAG
        bad_flag_state = dataclasses.replace(transition.output_state, particle_flags=bad_flags)
        with self.assertRaisesRegex(ValueError, "output flags"):
            dataclasses.replace(transition, output_state=bad_flag_state)

        bad_q = transition.output_state.q.copy()
        bad_q[-1, 0] += np.float32(0.01)
        bad_position_state = dataclasses.replace(transition.output_state, q=bad_q)
        with self.assertRaisesRegex(ValueError, "float32 committed reference"):
            dataclasses.replace(transition, output_state=bad_position_state)

        bad_reference_record = dict(transition.reference_record)
        bad_reference_record["position_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reference record position hash"):
            dataclasses.replace(transition, reference_record=bad_reference_record)

    def test_failed_reference_stops_before_commit(self):
        calls = 0

        def accept_then_fail(scene, config):
            nonlocal calls
            calls += 1
            accepted = self._accepted_reference(scene, config)
            if calls == 1:
                return accepted
            return history._ReferenceStep(
                positions=accepted.positions,
                accepted=False,
                failures=("synthetic alternate-start gate failure",),
                deterministic_record={"method": "unittest-rejection", "accepted": False},
                timing_record={"seconds": 0.25},
            )

        with mock.patch.object(history, "_solve_dense_reference", side_effect=accept_then_fail):
            chain = self.stretch.generate(
                stop=history.AtomicCoordinate(0, 3),
                max_transitions=3,
            )
        self.assertEqual(chain.termination, "failed_reference")
        self.assertEqual(len(chain.transitions), 1)
        self.assertEqual(chain.final_checkpoint.state.coordinate, history.AtomicCoordinate(0, 1))
        self.assertEqual(chain.failed_reference.coordinate, history.AtomicCoordinate(0, 1))
        self.assertIn("alternate-start", chain.failed_reference.failures[0])
        self.assertEqual(len(chain.timings), 2)
        self.assertFalse(chain.timings[-1].accepted)

    def test_healthy_native_stall_retries_only_step_tolerance_and_authenticates_attempts(self):
        configs = []

        def stall_then_accept(scene, config):
            configs.append(config)
            if len(configs) == 1:
                return self._stalled_reference(scene, config)
            accepted = self._accepted_reference(scene, config)
            deterministic_record = dict(accepted.deterministic_record)
            deterministic_record.update(
                {
                    "config": dataclasses.asdict(config),
                    "scene_sha256": "1" * 64,
                    "objective_instance_sha256": "2" * 64,
                }
            )
            return dataclasses.replace(accepted, deterministic_record=deterministic_record)

        with mock.patch.object(history, "_solve_dense_reference", side_effect=stall_then_accept):
            chain = self.stretch.generate(max_transitions=1)

        self.assertEqual(chain.termination, "range_complete")
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0], history._default_newton_config())
        self.assertEqual(configs[1], dataclasses.replace(configs[0], step_relative_tolerance=0.0))

        transition = chain.transitions[0]
        record = transition.reference_record
        self.assertEqual(record["retry_policy"], history._STALLED_RETRY_POLICY)
        self.assertEqual(record["selected_attempt"], 1)
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(record["attempts"][0]["record"]["native_reason"], "stalled")
        self.assertTrue(record["attempts"][1]["record"]["accepted"])

        timing = chain.timings[0]
        self.assertEqual(timing.values["selected_attempt"], 1)
        self.assertEqual(len(timing.values["attempts"]), 2)
        tampered_values = history._thaw_json(timing.values)
        tampered_values["attempts"][0]["record"]["seconds"] = 999.0
        tampered_timing = dataclasses.replace(timing, values=tampered_values)
        self.assertNotEqual(tampered_timing.timing_sha256, timing.timing_sha256)

        tampered_record = history._thaw_json(record)
        tampered_record["attempts"][0]["record"]["final_gradient_norm"] = 1.0
        tampered_transition = dataclasses.replace(transition, reference_record=tampered_record)
        self.assertNotEqual(tampered_transition.transition_sha256, transition.transition_sha256)

    def test_primary_acceptance_returns_the_exact_unwrapped_reference_step(self):
        scene = self.stretch.build_atomic_scene(
            self.stretch.initial_checkpoint.state,
            self.stretch.apply_callback(self.stretch.initial_checkpoint.state),
        )
        primary = self._accepted_reference(scene, history._default_newton_config())
        with mock.patch.object(history, "_solve_dense_reference", return_value=primary) as solve:
            selected = history._solve_dense_reference_with_retry(scene, history._default_newton_config())

        solve.assert_called_once_with(scene, history._default_newton_config())
        self.assertIs(selected, primary)
        self.assertEqual(selected.deterministic_record, primary.deterministic_record)
        self.assertEqual(selected.timing_record, primary.timing_record)

    def test_retry_config_tamper_fails_closed_and_preserves_both_attempts(self):
        calls = 0

        def stalled_then_tampered_acceptance(scene, config):
            nonlocal calls
            calls += 1
            if calls == 1:
                return self._stalled_reference(scene, config)
            accepted = self._accepted_reference(scene, config)
            tampered_config = dataclasses.asdict(config)
            tampered_config["max_iterations"] += 1
            record = dict(accepted.deterministic_record)
            record.update(
                {
                    "config": tampered_config,
                    "scene_sha256": "1" * 64,
                    "objective_instance_sha256": "2" * 64,
                }
            )
            return dataclasses.replace(accepted, deterministic_record=record)

        with mock.patch.object(
            history,
            "_solve_dense_reference",
            side_effect=stalled_then_tampered_acceptance,
        ):
            chain = self.stretch.generate(max_transitions=1)

        self.assertEqual(calls, 2)
        self.assertEqual(chain.termination, "failed_reference")
        self.assertEqual(
            chain.failed_reference.failures,
            ("dense reference retry config does not match the requested retry config",),
        )
        record = chain.failed_reference.reference_record
        self.assertFalse(record["accepted"])
        self.assertEqual(record["selected_attempt"], 1)
        self.assertEqual(record["provenance_failures"], chain.failed_reference.failures)
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(record["attempts"][1]["record"]["config"]["max_iterations"], 51)
        self.assertEqual(len(chain.timings[0].values["attempts"]), 2)
        self.assertFalse(chain.timings[0].accepted)

        tampered_record = history._thaw_json(record)
        tampered_record["attempts"][1]["record"]["config"]["max_iterations"] = 52
        tampered_failure = dataclasses.replace(chain.failed_reference, reference_record=tampered_record)
        self.assertNotEqual(tampered_failure.failure_sha256, chain.failed_reference.failure_sha256)
        tampered_timing_values = history._thaw_json(chain.timings[0].values)
        tampered_timing_values["attempts"][1]["record"]["seconds"] = 999.0
        tampered_timing = dataclasses.replace(chain.timings[0], values=tampered_timing_values)
        self.assertNotEqual(tampered_timing.timing_sha256, chain.timings[0].timing_sha256)

    def test_retry_combine_anomaly_fails_closed_and_preserves_both_attempts(self):
        cases = (
            ("scene_sha256", "3" * 64, "changed scene_sha256"),
            ("objective_instance_sha256", "4" * 64, "changed objective_instance_sha256"),
            ("position_sha256", "5" * 64, "position hash does not match"),
        )
        for field, value, expected_failure in cases:
            with self.subTest(field=field):
                calls = 0

                def stalled_then_anomalous_acceptance(
                    scene,
                    config,
                    *,
                    anomaly_field=field,
                    anomaly_value=value,
                ):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return self._stalled_reference(scene, config)
                    accepted = self._accepted_reference(scene, config)
                    record = dict(accepted.deterministic_record)
                    record.update(
                        {
                            "config": dataclasses.asdict(config),
                            "scene_sha256": "1" * 64,
                            "objective_instance_sha256": "2" * 64,
                            anomaly_field: anomaly_value,
                        }
                    )
                    return dataclasses.replace(accepted, deterministic_record=record)

                with mock.patch.object(
                    history,
                    "_solve_dense_reference",
                    side_effect=stalled_then_anomalous_acceptance,
                ):
                    chain = self.stretch.generate(max_transitions=1)

                self.assertEqual(calls, 2)
                self.assertEqual(chain.termination, "failed_reference")
                self.assertIn(expected_failure, chain.failed_reference.failures[0])
                record = chain.failed_reference.reference_record
                self.assertFalse(record["accepted"])
                self.assertEqual(record["selected_attempt"], 1)
                self.assertEqual(record["provenance_failures"], chain.failed_reference.failures)
                self.assertEqual(len(record["attempts"]), 2)
                self.assertEqual(len(chain.timings[0].values["attempts"]), 2)
                self.assertFalse(chain.timings[0].accepted)

    def test_retry_policy_rejects_inversion_nonfinite_and_other_termination(self):
        cases = (
            (
                "inversion",
                "stalled",
                ("native termination: stalled", "reference contains inverted tetrahedra"),
            ),
            ("nonfinite", "nonfinite", ("native termination: nonfinite",)),
            ("other", "max_iterations", ("native termination: max_iterations",)),
        )
        for label, reason, failures in cases:
            with self.subTest(label=label):
                calls = 0

                def rejected(scene, config, *, expected_reason=reason, expected_failures=failures):
                    nonlocal calls
                    calls += 1
                    reference = self._stalled_reference(scene, config, failures=expected_failures)
                    record = dict(reference.deterministic_record)
                    record["native_reason"] = expected_reason
                    return dataclasses.replace(reference, deterministic_record=record)

                with mock.patch.object(history, "_solve_dense_reference", side_effect=rejected):
                    chain = self.stretch.generate(max_transitions=1)
                self.assertEqual(calls, 1)
                self.assertEqual(chain.termination, "failed_reference")
                self.assertNotIn("attempts", chain.failed_reference.reference_record)

    def test_stalled_retry_exception_remains_fail_closed_with_both_attempts(self):
        calls = 0

        def stall_then_raise(scene, config):
            nonlocal calls
            calls += 1
            if calls == 1:
                return self._stalled_reference(scene, config)
            raise RuntimeError("synthetic retry factorization failure")

        with mock.patch.object(history, "_solve_dense_reference", side_effect=stall_then_raise):
            chain = self.stretch.generate(max_transitions=1)

        self.assertEqual(calls, 2)
        self.assertEqual(chain.termination, "failed_reference")
        self.assertIn("stalled retry raised RuntimeError", chain.failed_reference.failures[0])
        self.assertEqual(chain.failed_reference.reference_record["selected_attempt"], 1)
        self.assertEqual(len(chain.failed_reference.reference_record["attempts"]), 2)
        self.assertEqual(chain.timings[0].values["selected_attempt"], 1)

    def test_real_bounded_ordinal8_retry_connects_through_ordinal9(self):
        chain = self.stretch.generate(
            stop=history.AtomicCoordinate.from_ordinal(10),
            max_transitions=10,
        )

        self.assertEqual(chain.termination, "range_complete")
        self.assertEqual(len(chain.transitions), 10)
        chain.verify()
        ordinal8 = chain.transitions[8]
        ordinal9 = chain.transitions[9]
        self.assertEqual(ordinal8.coordinate.ordinal, 8)
        self.assertEqual(ordinal8.reference_record["selected_attempt"], 1)
        attempts = ordinal8.reference_record["attempts"]
        self.assertEqual(attempts[0]["record"]["native_reason"], "stalled")
        self.assertFalse(attempts[0]["record"]["accepted"])
        self.assertTrue(attempts[1]["record"]["accepted"])
        self.assertEqual(attempts[0]["record"]["config"]["step_relative_tolerance"], 1.0e-14)
        self.assertEqual(attempts[1]["record"]["config"]["step_relative_tolerance"], 0.0)
        self.assertEqual(ordinal9.coordinate.ordinal, 9)
        self.assertEqual(ordinal9.input_state_sha256, ordinal8.output_state.state_sha256)
        self.assertNotIn("attempts", ordinal9.reference_record)

    def test_reference_exception_stops_before_commit(self):
        with mock.patch.object(
            history,
            "_solve_dense_reference",
            side_effect=RuntimeError("synthetic factorization failure"),
        ):
            chain = self.stretch.generate(max_transitions=1)
        self.assertEqual(chain.termination, "failed_reference")
        self.assertEqual(chain.transitions, ())
        self.assertEqual(chain.final_checkpoint.checkpoint_sha256, chain.initial_checkpoint.checkpoint_sha256)
        self.assertIn("RuntimeError", chain.failed_reference.failures[0])

    def test_bounded_ranges_and_checkpoint_resume(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_transitions"):
            self.stretch.generate(stop=history.AtomicCoordinate(2, 0))

        with mock.patch.object(history, "_solve_dense_reference", side_effect=self._accepted_reference):
            prefix = self.stretch.generate(
                stop=history.AtomicCoordinate(0, 2),
                max_transitions=2,
            )
            checkpoint = prefix.checkpoint_at(history.AtomicCoordinate(0, 1))
            resumed = self.stretch.generate(
                start=history.AtomicCoordinate(0, 1),
                stop=history.AtomicCoordinate(0, 3),
                checkpoint=checkpoint,
                prior_chain=prefix,
                max_transitions=2,
            )
            direct = self.stretch.generate(
                stop=history.AtomicCoordinate(0, 3),
                max_transitions=3,
            )

        self.assertEqual(resumed.final_checkpoint.state.state_sha256, direct.final_checkpoint.state.state_sha256)
        self.assertEqual(resumed.final_checkpoint.prefix_sha256, direct.final_checkpoint.prefix_sha256)
        self.assertEqual(resumed.prior_chain_sha256, prefix.chain_sha256)
        self.assertEqual(resumed.as_dict()["verification_scope"]["prior_history_proof"], prefix.chain_sha256)
        with self.assertRaisesRegex(ValueError, "verified prior-chain proof"):
            self.stretch.generate(
                start=history.AtomicCoordinate(0, 1),
                stop=history.AtomicCoordinate(0, 2),
                checkpoint=checkpoint,
            )

        wrong_prefix = dataclasses.replace(checkpoint, prefix_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "does not match the verified prior-chain proof"):
            self.stretch.generate(
                start=history.AtomicCoordinate(0, 1),
                stop=history.AtomicCoordinate(0, 2),
                checkpoint=wrong_prefix,
                prior_chain=prefix,
            )
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.stretch.generate(start=history.AtomicCoordinate(0, 1))


if __name__ == "__main__":
    unittest.main()
