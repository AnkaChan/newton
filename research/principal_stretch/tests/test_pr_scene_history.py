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
