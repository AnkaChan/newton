# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for raw PSS candidates on the independent common objective."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import warp as wp

import newton

from .. import pss_benchmark as pss
from ..graph_transformer import GraphTransformerConfig, PrincipalStretchGraphTransformer
from ..predictor import build_stretch_predictor
from ..solver_benchmark import build_common_problem, build_structured_cantilever_scene, evaluate_common_state
from ..solver_scenes import build_extension_scene


class TestPSSBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.scene = build_structured_cantilever_scene(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, 0.0),
            total_tip_force=(0.0, 0.0, 0.0),
            name="pss-zero-update-fixture",
        )
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.checkpoint_path = pathlib.Path(cls.temporary_directory.name) / "tiny-graph.pt"
        cls._write_checkpoint(cls.checkpoint_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    @classmethod
    def _checkpoint(
        cls,
        *,
        config: GraphTransformerConfig | None = None,
        rest_q: np.ndarray | None = None,
        tet_indices: np.ndarray | None = None,
    ) -> dict[str, object]:
        if config is None:
            config = GraphTransformerConfig(
                hidden_dim=16,
                num_heads=4,
                n_levels=0,
                cluster_size=2,
                dt=cls.scene.dt,
                architecture_version=2,
            )
        if rest_q is None:
            rest_q = cls.scene.rest_q
        if tet_indices is None:
            tet_indices = cls.scene.tet_indices
        predictor = build_stretch_predictor(
            "graph-transformer",
            np.array(rest_q, copy=True),
            np.array(tet_indices, copy=True),
            torch.device("cpu"),
            torch.float32,
            residual=True,
            graph_config=config,
        )
        return {
            "state_dict": predictor.model.state_dict(),
            "predictor_config": predictor.checkpoint_config(),
            "args": {
                "predictor": "graph-transformer",
                "residual": True,
                "blocks": 1,
                "warm": "inertial",
                "dt": config.dt,
                "loss": "pos",
                "gt_hidden": config.hidden_dim,
                "gt_heads": config.num_heads,
                "gt_levels": config.n_levels,
                "gt_cluster_size": config.cluster_size,
                "gt_dropout": config.dropout,
                "gt_max_delta": config.max_hencky_update,
                "gt_architecture_version": config.architecture_version,
            },
            "training_realized_hierarchy_levels": predictor.model.n_levels,
            "torch_version": str(torch.__version__),
        }

    @classmethod
    def _write_checkpoint(cls, path: pathlib.Path, mutation=None, **checkpoint_kwargs) -> None:
        checkpoint = cls._checkpoint(**checkpoint_kwargs)
        if mutation is not None:
            mutation(checkpoint)
        torch.save(checkpoint, path)

    def test_raw_candidate_is_finite_pin_exact_and_common_scorable(self):
        result = pss.run_pss(
            self.scene,
            self.checkpoint_path,
            2,
            device="cpu",
            warmup=True,
            repeats=2,
        )
        problem = build_common_problem(self.scene)
        metrics = evaluate_common_state(problem, result.positions)
        record = pss.pss_run_record(result, metrics, scene=self.scene)

        np.testing.assert_allclose(result.positions, self.scene.rest_q, rtol=0.0, atol=2.0e-7)
        np.testing.assert_array_equal(
            result.positions[self.scene.pinned_indices],
            self.scene.pin_targets,
        )
        self.assertTrue(np.isfinite(result.target_stretch).all())
        self.assertGreater(np.linalg.eigvalsh(result.target_stretch).min(), 0.0)
        self.assertLess(metrics.relative_residual, 1.0e-10)
        self.assertEqual(metrics.position_sha256, result.result_state_sha256)
        self.assertFalse(result.positions.flags["W"])
        self.assertFalse(result.target_stretch.flags["W"])
        self.assertFalse(result.previous_positions.flags["W"])
        self.assertEqual(len(result.repeat_seconds), 2)
        self.assertEqual(len(result.predictor_seconds), 2)
        self.assertEqual(len(result.decoder_seconds), 2)
        self.assertEqual(len(result.transfer_seconds), 2)
        self.assertGreater(result.median_solve_seconds, 0.0)
        self.assertEqual(result.checkpoint_sha256, hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest())
        self.assertEqual(record["method"], "principal-stretch-graph-transformer-diagnostic")
        self.assertFalse(record["claim_boundary"]["common_objective_convergence"])
        self.assertEqual(record["work"]["predictor_passes_per_repeat"], 1)
        self.assertEqual(record["work"]["surrogate_decoder_sweeps_per_repeat"], 2)
        self.assertEqual(record["repeat_determinism"]["target_tolerance"], 0.0)
        self.assertTrue(record["repeat_determinism"]["target_required_exact"])
        self.assertEqual(record["repeat_determinism"]["position_tolerance_m"], 0.0)
        self.assertEqual(
            record["metrics_provenance"]["objective_instance_sha256"],
            result.objective_instance_sha256,
        )
        self.assertIsNone(record["metrics_provenance"]["reference_state_sha256"])
        self.assertEqual(record["repeat_determinism"]["position_max_abs_discrepancy_m"], 0.0)
        json.dumps(record, allow_nan=False)

    def test_every_repeat_restarts_from_common_inertial_iterate(self):
        starts = []
        original_solve = pss.torch_solver.solve

        def recording_solve(state, target, pin_targets, x_init=None, n_iters=6):
            starts.append(x_init.detach().cpu().numpy().copy())
            return original_solve(state, target, pin_targets, x_init=x_init, n_iters=n_iters)

        with mock.patch.object(pss.torch_solver, "solve", side_effect=recording_solve):
            pss.run_pss(
                self.scene,
                self.checkpoint_path,
                3,
                device="cpu",
                warmup=True,
                repeats=2,
            )

        self.assertEqual(len(starts), 3)
        for start in starts:
            np.testing.assert_array_equal(start, self.scene.vbd_inertial_target)

    def test_history_is_derived_from_velocity_and_bound(self):
        scene = build_structured_cantilever_scene(
            dt=self.scene.dt,
            gravity=(0.0, 0.0, 0.0),
            total_tip_force=(0.0, 0.0, 0.0),
            initial_velocity=(0.125, -0.25, 0.5),
            name="pss-history-fixture",
        )
        result = pss.run_pss(
            scene,
            self.checkpoint_path,
            1,
            device="cpu",
            warmup=False,
            repeats=1,
        )
        expected = scene.x_current - scene.dt * scene.velocity
        np.testing.assert_array_equal(result.previous_positions, expected)
        self.assertEqual(result.previous_state_sha256, pss._array_digest(expected))

    def test_static_topology_is_rebuilt_for_a_different_mesh(self):
        scene = build_structured_cantilever_scene(
            dimensions=(1, 2, 1),
            dt=self.scene.dt,
            gravity=(0.0, 0.0, 0.0),
            total_tip_force=(0.0, 0.0, 0.0),
            name="pss-cross-topology-fixture",
        )
        result = pss.run_pss(
            scene,
            self.checkpoint_path,
            1,
            device="cpu",
            warmup=False,
            repeats=1,
        )
        self.assertEqual(result.positions.shape, scene.x_current.shape)
        self.assertEqual(result.target_stretch.shape, (scene.n_tets, 3, 3))
        self.assertEqual(result.realized_hierarchy_levels, 0)

    def test_scene_timestep_pin_and_history_ambiguity_are_rejected(self):
        wrong_dt = dataclasses.replace(self.scene, dt=self.scene.dt * 0.5)
        with self.assertRaisesRegex(ValueError, "checkpoint dt"):
            pss.run_pss(wrong_dt, self.checkpoint_path, 1, device="cpu", warmup=False, repeats=1)

        current = self.scene.x_current.copy()
        current[self.scene.pinned_indices[0], 0] += np.float32(0.25)
        wrong_pin = dataclasses.replace(self.scene, x_current=current)
        with self.assertRaisesRegex(ValueError, "exact Dirichlet"):
            pss.run_pss(wrong_pin, self.checkpoint_path, 1, device="cpu", warmup=False, repeats=1)

        velocity = self.scene.velocity.copy()
        velocity[self.scene.pinned_indices[0], 1] = np.float32(0.125)
        moving_pin = dataclasses.replace(self.scene, velocity=velocity)
        with self.assertRaisesRegex(ValueError, "zero pinned velocity"):
            pss.run_pss(moving_pin, self.checkpoint_path, 1, device="cpu", warmup=False, repeats=1)

    def test_unanchored_component_is_rejected_before_factorization(self):
        mass = self.scene.mass.copy()
        replacement_mass = np.float32(self.scene.mass[self.scene.free_indices[0]])
        mass[self.scene.pinned_indices] = replacement_mass
        inverse_mass = np.zeros_like(mass, dtype=np.float32)
        positive = mass.astype(np.float32) > 0.0
        inverse_mass[positive] = np.float32(1.0) / mass.astype(np.float32)[positive]
        flags = self.scene.particle_flags.copy()
        flags |= int(newton.ParticleFlags.ACTIVE)
        floating = dataclasses.replace(
            self.scene,
            mass=mass,
            particle_inv_mass=inverse_mass,
            particle_flags=flags,
            pinned_indices=np.empty(0, dtype=np.int64),
            pin_targets=np.empty((0, 3), dtype=np.float64),
            metadata=dict(self.scene.metadata) | {"fixture": "unanchored"},
        )
        with self.assertRaisesRegex(ValueError, "must contain a pinned vertex"):
            pss.run_pss(floating, self.checkpoint_path, 1, device="cpu", warmup=False, repeats=1)

    def test_non_graph_and_incompatible_checkpoint_semantics_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = pathlib.Path(temporary_directory)
            not_graph = directory / "not-graph.pt"
            self._write_checkpoint(
                not_graph,
                lambda checkpoint: checkpoint["predictor_config"].update({"kind": "mlp"}),
            )
            with self.assertRaisesRegex(ValueError, "requires a graph-transformer"):
                pss.run_pss(self.scene, not_graph, 1, device="cpu", warmup=False, repeats=1)

            blocks = directory / "blocks.pt"
            self._write_checkpoint(blocks, lambda checkpoint: checkpoint["args"].update({"blocks": 2}))
            with self.assertRaisesRegex(ValueError, "blocks=1"):
                pss.run_pss(self.scene, blocks, 1, device="cpu", warmup=False, repeats=1)

            args_dt = directory / "args-dt.pt"
            self._write_checkpoint(
                args_dt,
                lambda checkpoint: checkpoint["args"].update({"dt": self.scene.dt * 0.5}),
            )
            with self.assertRaisesRegex(ValueError, "args dt"):
                pss.run_pss(self.scene, args_dt, 1, device="cpu", warmup=False, repeats=1)

            mismatches = {
                "gt_hidden": 32,
                "gt_heads": 2,
                "gt_levels": 1,
                "gt_cluster_size": 3,
                "gt_dropout": 0.25,
                "gt_max_delta": 0.125,
                "gt_architecture_version": 1,
            }
            for argument_name, incompatible_value in mismatches.items():
                with self.subTest(argument_name=argument_name):
                    path = directory / f"mismatch-{argument_name}.pt"
                    self._write_checkpoint(
                        path,
                        lambda checkpoint, name=argument_name, value=incompatible_value: checkpoint["args"].update(
                            {name: value}
                        ),
                    )
                    with self.assertRaisesRegex(ValueError, f"args {argument_name}"):
                        pss.run_pss(self.scene, path, 1, device="cpu", warmup=False, repeats=1)

            legacy_rotation_default = directory / "legacy-no-rotation-bound.pt"
            self._write_checkpoint(
                legacy_rotation_default,
                lambda checkpoint: checkpoint["predictor_config"]["graph_transformer"].pop("max_rotation_update"),
            )
            legacy_result = pss.run_pss(
                self.scene,
                legacy_rotation_default,
                1,
                device="cpu",
                warmup=False,
                repeats=1,
            )
            self.assertIn("max_rotation_update", json.loads(legacy_result.predictor_config_json)["graph_transformer"])

            def make_v3_without_rotation_metadata(checkpoint):
                checkpoint["predictor_config"]["graph_transformer"]["architecture_version"] = 3
                checkpoint["predictor_config"]["graph_transformer"].pop("max_rotation_update")
                checkpoint["args"]["gt_architecture_version"] = 3

            v3_missing_rotation = directory / "v3-no-rotation-bound.pt"
            self._write_checkpoint(v3_missing_rotation, make_v3_without_rotation_metadata)
            with self.assertRaisesRegex(ValueError, "missing max_rotation_update"):
                pss.run_pss(self.scene, v3_missing_rotation, 1, device="cpu", warmup=False, repeats=1)

    def test_training_hierarchy_depth_guards_cross_resolution_inference(self):
        one_tet_rest = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        one_tet_indices = np.array([[0, 1, 2, 3]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = pathlib.Path(temporary_directory)
            for architecture_version in (1, 2):
                with self.subTest(architecture_version=architecture_version):
                    config = GraphTransformerConfig(
                        hidden_dim=16,
                        num_heads=4,
                        n_levels=2,
                        cluster_size=2,
                        dt=self.scene.dt,
                        architecture_version=architecture_version,
                    )
                    path = directory / f"depth-v{architecture_version}.pt"
                    self._write_checkpoint(
                        path,
                        config=config,
                        rest_q=one_tet_rest,
                        tet_indices=one_tet_indices,
                    )
                    message = (
                        "exceeds.*training-realized" if architecture_version == 2 else "trained at hierarchy depth"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        pss.run_pss(self.scene, path, 1, device="cpu", warmup=False, repeats=1)

            equal_depth_config = GraphTransformerConfig(
                hidden_dim=16,
                num_heads=4,
                n_levels=2,
                cluster_size=2,
                dt=self.scene.dt,
                architecture_version=2,
            )
            equal_depth = directory / "equal-depth-v2.pt"
            self._write_checkpoint(equal_depth, config=equal_depth_config)
            result = pss.run_pss(self.scene, equal_depth, 1, device="cpu", warmup=False, repeats=1)
            self.assertEqual(result.realized_hierarchy_levels, result.training_hierarchy_levels)

    def test_v2_positive_depth_requires_valid_training_depth_metadata(self):
        config = GraphTransformerConfig(
            hidden_dim=16,
            num_heads=4,
            n_levels=2,
            cluster_size=2,
            dt=self.scene.dt,
            architecture_version=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = pathlib.Path(temporary_directory)
            missing = directory / "missing-depth.pt"
            self._write_checkpoint(
                missing,
                lambda checkpoint: checkpoint.pop("training_realized_hierarchy_levels"),
                config=config,
            )
            with self.assertRaisesRegex(ValueError, "must record training_realized_hierarchy_levels"):
                pss.run_pss(self.scene, missing, 1, device="cpu", warmup=False, repeats=1)

            malformed = directory / "malformed-depth.pt"
            self._write_checkpoint(
                malformed,
                lambda checkpoint: checkpoint.update({"training_realized_hierarchy_levels": True}),
                config=config,
            )
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                pss.run_pss(self.scene, malformed, 1, device="cpu", warmup=False, repeats=1)

    def test_metric_objective_and_reference_provenance_are_recomputed(self):
        result = pss.run_pss(
            self.scene,
            self.checkpoint_path,
            1,
            device="cpu",
            warmup=False,
            repeats=1,
        )
        problem = build_common_problem(self.scene)
        reference_a = self.scene.rest_q.copy()
        reference_a[self.scene.free_indices[0], 0] += 1.0e-3
        reference_b = self.scene.rest_q.copy()
        reference_b[self.scene.free_indices[0], 0] += 2.0e-3
        metrics_a = evaluate_common_state(problem, result.positions, reference_positions=reference_a)
        record = pss.pss_run_record(
            result,
            metrics_a,
            scene=self.scene,
            reference_positions=reference_a,
        )
        self.assertEqual(record["metrics_provenance"]["reference_state_sha256"], pss._array_digest(reference_a))

        with self.assertRaisesRegex(ValueError, "supplied objective and reference"):
            pss.pss_run_record(
                result,
                metrics_a,
                scene=self.scene,
                reference_positions=reference_b,
            )
        with self.assertRaisesRegex(ValueError, "scene is required"):
            pss.pss_run_record(result, metrics_a, reference_positions=reference_a)

        wrong_scene = build_structured_cantilever_scene(
            dt=self.scene.dt,
            gravity=(0.0, -1.0, 0.0),
            total_tip_force=(0.0, 0.0, 0.0),
            name="wrong-common-objective",
        )
        wrong_metrics = evaluate_common_state(
            build_common_problem(wrong_scene),
            result.positions,
            reference_positions=reference_a,
        )
        with self.assertRaisesRegex(ValueError, "metrics scene"):
            pss.pss_run_record(
                result,
                wrong_metrics,
                scene=wrong_scene,
                reference_positions=reference_a,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_extension_repeat_discrepancy_is_bounded_and_recorded(self):
        checkpoint = pathlib.Path("/tmp/pss-gt-toy-flat-k4-6000.pt")
        if not checkpoint.is_file():
            self.skipTest(f"preserved integration checkpoint is unavailable: {checkpoint}")
        scene = build_extension_scene(dt=1.0 / 60.0)
        result = pss.run_pss(
            scene,
            checkpoint,
            4,
            device="cuda:0",
            warmup=True,
            repeats=4,
        )
        self.assertEqual(result.target_repeat_max_abs_discrepancy, 0.0)
        self.assertEqual(
            result.target_repeat_tolerance,
            pss._cuda_target_repeat_tolerance(result.target_repeat_scale),
        )
        self.assertLessEqual(
            result.position_repeat_max_abs_discrepancy_m,
            result.position_repeat_tolerance_m,
        )
        self.assertEqual(
            result.position_repeat_tolerance_m,
            pss._cuda_position_repeat_tolerance_m(result.position_repeat_scale_m),
        )
        record = pss.pss_run_record(result)
        self.assertFalse(record["repeat_determinism"]["target_required_exact"])
        self.assertEqual(
            record["repeat_determinism"]["position_max_abs_discrepancy_m"],
            result.position_repeat_max_abs_discrepancy_m,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_multires_nonzero_head_repeat_discrepancy_is_bounded(self):
        config = GraphTransformerConfig(
            hidden_dim=16,
            num_heads=4,
            n_levels=4,
            cluster_size=2,
            dt=self.scene.dt,
            architecture_version=2,
        )

        def make_head_nonzero(checkpoint):
            state_dict = checkpoint["state_dict"]
            weight = state_dict["output_head.2.weight"]
            bias = state_dict["output_head.2.bias"]
            weight.copy_(torch.linspace(-0.02, 0.02, weight.numel()).reshape_as(weight))
            bias.copy_(torch.linspace(-0.01, 0.01, bias.numel()))

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = pathlib.Path(temporary_directory) / "multires-nonzero-head.pt"
            self._write_checkpoint(checkpoint_path, make_head_nonzero, config=config)
            result = pss.run_pss(
                self.scene,
                checkpoint_path,
                4,
                device="cuda:0",
                warmup=True,
                repeats=8,
            )

        self.assertGreater(result.realized_hierarchy_levels, 0)
        self.assertLessEqual(result.target_repeat_max_abs_discrepancy, result.target_repeat_tolerance)
        self.assertLessEqual(
            result.position_repeat_max_abs_discrepancy_m,
            result.position_repeat_tolerance_m,
        )
        self.assertEqual(
            result.target_repeat_tolerance,
            pss._CUDA_TARGET_REPEAT_EPS_MULTIPLIER * np.finfo(np.float32).eps * result.target_repeat_scale,
        )
        self.assertEqual(
            result.position_repeat_tolerance_m,
            pss._CUDA_POSITION_REPEAT_EPS_MULTIPLIER * np.finfo(np.float32).eps * result.position_repeat_scale_m,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_target_repeat_tolerance_rejects_material_drift(self):
        original_forward = PrincipalStretchGraphTransformer.forward
        call_count = 0

        def perturbed_forward(model, *args, **kwargs):
            nonlocal call_count
            target = original_forward(model, *args, **kwargs)
            call_count += 1
            if call_count == 2:
                target = target.clone()
                tolerance = pss._cuda_target_repeat_tolerance(pss._target_repeat_scale(target))
                target.reshape(-1)[0] += 2.0 * tolerance
            return target

        with mock.patch.object(PrincipalStretchGraphTransformer, "forward", new=perturbed_forward):
            with self.assertRaisesRegex(RuntimeError, "predictor target.*exceeding"):
                pss.run_pss(
                    self.scene,
                    self.checkpoint_path,
                    1,
                    device="cuda:0",
                    warmup=False,
                    repeats=2,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_position_repeat_tolerance_rejects_material_drift(self):
        original_solve = pss.torch_solver.solve
        call_count = 0
        tolerance = pss._cuda_position_repeat_tolerance_m(pss._position_repeat_scale_m(self.scene))

        def perturbed_solve(state, target, pin_targets, x_init=None, n_iters=6):
            nonlocal call_count
            positions = original_solve(state, target, pin_targets, x_init=x_init, n_iters=n_iters)
            call_count += 1
            if call_count == 2:
                positions = positions.clone()
                positions[state.free[0], 0] += 2.0 * tolerance
            return positions

        with mock.patch.object(pss.torch_solver, "solve", side_effect=perturbed_solve):
            with self.assertRaisesRegex(RuntimeError, "exceeding"):
                pss.run_pss(
                    self.scene,
                    self.checkpoint_path,
                    1,
                    device="cuda:0",
                    warmup=False,
                    repeats=2,
                )

    def test_checkpoint_bytes_and_result_tampering_are_detected(self):
        result = pss.run_pss(
            self.scene,
            self.checkpoint_path,
            1,
            device="cpu",
            warmup=False,
            repeats=1,
        )
        positions = result.positions.copy()
        positions[self.scene.free_indices[0], 0] += 1.0e-3
        tampered = dataclasses.replace(result, positions=positions)
        with self.assertRaisesRegex(ValueError, "positions were modified"):
            pss.pss_run_record(tampered)

        checkpoint = torch.load(io.BytesIO(self.checkpoint_path.read_bytes()), map_location="cpu", weights_only=False)
        changed_path = pathlib.Path(self.temporary_directory.name) / "tiny-graph-changed.pt"
        checkpoint["args"]["provenance_probe"] = 1
        torch.save(checkpoint, changed_path)
        changed = pss.run_pss(
            self.scene,
            changed_path,
            1,
            device="cpu",
            warmup=False,
            repeats=1,
        )
        self.assertNotEqual(changed.checkpoint_sha256, result.checkpoint_sha256)
        self.assertNotEqual(changed.run_sha256, result.run_sha256)

    def test_invalid_budgets_are_rejected(self):
        for iterations in (0, -1, True, 1.5):
            with self.subTest(iterations=iterations):
                with self.assertRaisesRegex(ValueError, "decoder_iterations"):
                    pss.run_pss(
                        self.scene,
                        self.checkpoint_path,
                        iterations,
                        device="cpu",
                        warmup=False,
                        repeats=1,
                    )


if __name__ == "__main__":
    unittest.main()
