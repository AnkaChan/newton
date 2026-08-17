# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""End-to-end routing tests for the full-gradient graph architecture."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from research.principal_stretch import bench_pareto, eval_singlestep, rollout, train
from research.principal_stretch import torch_solver as ts
from research.principal_stretch.graph_transformer import GraphTransformerConfig
from research.principal_stretch.predictor import (
    build_stretch_predictor,
    decode_predictor_step,
    predictor_architecture_version,
    predictor_decoder_work,
    resolve_solver_iterations,
    validate_static_pin_trajectory,
)
from research.principal_stretch.tests.test_graph_transformer import _chain_mesh, _inputs, _rotation, _tet_poses


class TestV3Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest, cls.tets = _chain_mesh(8)
        cls.state = ts.build_solver(
            cls.rest,
            cls.tets,
            _tet_poses(cls.rest, cls.tets),
            np.array([0, 1, 2], dtype=np.int64),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        cls.inputs = _inputs(cls.rest, cls.tets)

    def _predictor(self, version: int):
        return build_stretch_predictor(
            "graph-transformer",
            self.rest,
            self.tets,
            torch.device("cpu"),
            torch.float32,
            residual=True,
            graph_config=GraphTransformerConfig(
                hidden_dim=32,
                num_heads=4,
                n_levels=5,
                cluster_size=2,
                architecture_version=version,
            ),
        )

    def _decode(self, predictor, x_current=None, x_previous=None, force=None, *, x_init=None):
        default = self.inputs
        x_current = default[0] if x_current is None else x_current
        x_previous = default[1] if x_previous is None else x_previous
        force = default[2] if force is None else force
        if predictor_architecture_version(predictor) == 3:
            S_current = None
            S_previous = None
        else:
            S_current = ts.compute_S_from_x(self.state, x_current)
            S_previous = ts.compute_S_from_x(self.state, x_previous)
        return decode_predictor_step(
            predictor,
            self.state,
            x_current,
            x_previous,
            force,
            *default[3:],
            S_current,
            S_previous,
            x_current[self.state.pinned],
            x_init=x_init,
            solver_iterations=1,
            blocks=1,
        )

    def test_v3_work_is_explicit_and_ambiguous_iterations_are_rejected(self):
        v3 = self._predictor(3)
        self.assertEqual(resolve_solver_iterations(v3, None), 1)
        work = predictor_decoder_work(v3, 1, 1)
        self.assertEqual(
            work,
            {
                "schema_version": 1,
                "target": "full-deformation-gradient",
                "decoder": "weighted-global-projection",
                "predictor_passes": 1,
                "global_triangular_solves": 1,
                "local_polar_sweeps": 0,
            },
        )
        with self.assertRaisesRegex(ValueError, "solver_iterations must be 1"):
            resolve_solver_iterations(v3, 2)
        with self.assertRaisesRegex(ValueError, "blocks must be 1"):
            predictor_decoder_work(v3, 1, 2)

        v2 = self._predictor(2)
        self.assertEqual(resolve_solver_iterations(v2, None), 10)
        self.assertEqual(predictor_decoder_work(v2, 10, 3)["global_triangular_solves"], 9)

    def test_v3_route_calls_one_projection_and_never_legacy_decoder(self):
        predictor = self._predictor(3)
        with torch.no_grad():
            predictor.model.output_head[-1].bias.copy_(torch.tensor([0.1, -0.05, 0.03, 0.02, 0.01, -0.04]))
            predictor.model.rotation_head[-1].bias.copy_(torch.tensor([0.2, -0.1, 0.05]))
        expected_F = predictor.predict_deformation_gradient(self.state, *self.inputs)
        expected = ts.project_deformation_gradient(self.state, expected_F, self.inputs[0][self.state.pinned])

        with (
            mock.patch.object(predictor, "forward", side_effect=AssertionError("legacy forward called")),
            mock.patch.object(ts, "solve", side_effect=AssertionError("legacy solve called")),
            mock.patch.object(ts, "compute_S_from_x", side_effect=AssertionError("legacy polar called")),
            mock.patch.object(ts, "project_deformation_gradient", wraps=ts.project_deformation_gradient) as projection,
        ):
            actual = self._decode(predictor, x_init=torch.full_like(self.inputs[0], 99.0))
        projection.assert_called_once()
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_v2_route_is_numerically_unchanged(self):
        predictor = self._predictor(2)
        with torch.no_grad():
            predictor.model.output_head[-1].bias.copy_(torch.tensor([0.1, -0.05, 0.03, 0.02, 0.01, -0.04]))
        x_current, x_previous, force, gravity, mu, lam, pin = self.inputs
        S_current = ts.compute_S_from_x(self.state, x_current)
        S_previous = ts.compute_S_from_x(self.state, x_previous)
        x_init = ts.inertial_predictor(self.state, x_current, x_previous, x_current[self.state.pinned])
        target = predictor(
            self.state,
            x_current,
            x_previous,
            force,
            gravity,
            mu,
            lam,
            pin,
            S_current,
            S_previous,
        )
        expected = ts.solve(
            self.state,
            target.double(),
            x_current[self.state.pinned],
            x_init=x_init,
            n_iters=4,
        )

        with (
            mock.patch.object(
                predictor, "predict_deformation_gradient", side_effect=AssertionError("v3 target called")
            ),
            mock.patch.object(ts, "project_deformation_gradient", side_effect=AssertionError("v3 projection called")),
        ):
            actual = decode_predictor_step(
                predictor,
                self.state,
                x_current,
                x_previous,
                force,
                gravity,
                mu,
                lam,
                pin,
                S_current,
                S_previous,
                x_current[self.state.pinned],
                x_init=x_init,
                solver_iterations=4,
                blocks=1,
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_rotation_head_trains_through_the_global_projection(self):
        student = self._predictor(3)
        teacher = self._predictor(3)
        with torch.no_grad():
            teacher.model.rotation_head[-1].bias.copy_(torch.tensor([0.12, -0.08, 0.05]))
            target_positions = self._decode(teacher)

        optimizer = torch.optim.Adam(student.model.rotation_head.parameters(), lr=3.0e-2)
        initial_bias = student.model.rotation_head[-1].bias.detach().clone()
        losses = []
        first_gradient = None
        for _ in range(20):
            optimizer.zero_grad()
            predicted = self._decode(student)
            loss = (predicted - target_positions).square().mean()
            loss.backward()
            if first_gradient is None:
                first_gradient = student.model.rotation_head[-1].bias.grad.detach().clone()
            optimizer.step()
            losses.append(float(loss.detach()))

        self.assertGreater(first_gradient.abs().max().item(), 0.0)
        self.assertGreater((student.model.rotation_head[-1].bias - initial_bias).abs().max().item(), 0.0)
        self.assertLess(losses[-1], 0.2 * losses[0])

    def test_v3_end_to_end_decode_is_active_se3_equivariant(self):
        predictor = self._predictor(3)
        with torch.no_grad():
            predictor.model.output_head[-1].bias.copy_(torch.tensor([0.1, -0.05, 0.03, 0.02, 0.01, -0.04]))
            predictor.model.rotation_head[-1].bias.copy_(torch.tensor([0.2, -0.1, 0.05]))
        decoded = self._decode(predictor)

        Q = _rotation()
        translation = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)

        def rotate(vector):
            return vector @ Q.T

        transformed = self._decode(
            predictor,
            x_current=rotate(self.inputs[0]) + translation,
            x_previous=rotate(self.inputs[1]) + translation,
            force=rotate(self.inputs[2]),
        )
        torch.testing.assert_close(transformed, rotate(decoded) + translation, rtol=4.0e-6, atol=4.0e-6)

    def test_legacy_npz_route_rejects_moving_pins(self):
        positions = np.broadcast_to(self.rest, (3, *self.rest.shape)).copy()
        validate_static_pin_trajectory(self.rest, self.state.pinned.numpy(), positions)
        positions[1, 0, 1] += 1.0e-4
        with self.assertRaisesRegex(ValueError, "only static rest pins"):
            validate_static_pin_trajectory(self.rest, self.state.pinned.numpy(), positions)

    def test_eval_and_rollout_use_the_shared_versioned_router(self):
        self.assertIs(eval_singlestep.decode_predictor_step, decode_predictor_step)
        self.assertIs(rollout.decode_predictor_step, decode_predictor_step)

    def test_legacy_pareto_rejects_v3_checkpoint_semantics(self):
        with self.assertRaisesRegex(ValueError, "legacy right-stretch"):
            bench_pareto.validate_legacy_net_checkpoint(
                {
                    "kind": "graph-transformer",
                    "graph_transformer": {"architecture_version": 3},
                }
            )
        bench_pareto.validate_legacy_net_checkpoint(
            {
                "kind": "graph-transformer",
                "graph_transformer": {"architecture_version": 2},
            }
        )

    def test_train_cli_writes_v3_rotation_and_actual_decoder_metadata(self):
        frames = []
        rest = self.rest.astype(np.float64)
        phase = np.linspace(0.0, np.pi, rest.shape[0])[:, None]
        direction = np.array([[0.02, -0.01, 0.015]])
        for frame in range(4):
            x = rest + frame * np.sin(phase) * direction
            x[self.state.pinned.numpy()] = rest[self.state.pinned.numpy()]
            frames.append(x)

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            data_path = root / "tiny-static-pins.npz"
            checkpoint_path = root / "v3.pt"
            np.savez_compressed(
                data_path,
                rest_q=rest,
                tet_indices=self.tets,
                tet_poses=_tet_poses(rest, self.tets),
                pinned_indices=self.state.pinned.numpy(),
                x=np.stack(frames),
                f_ext=np.zeros((4, rest.shape[0], 3), dtype=np.float64),
                traj_start=np.array([0], dtype=np.int64),
                particle_mass=np.ones(rest.shape[0], dtype=np.float64),
                mu_per_tet=np.full(self.tets.shape[0], 2.0e4, dtype=np.float32),
                lam_per_tet=np.full(self.tets.shape[0], 3.0e4, dtype=np.float32),
                gravity=np.zeros(3, dtype=np.float64),
            )
            argv = [
                "train",
                "--train",
                str(data_path),
                "--out",
                str(checkpoint_path),
                "--device",
                "cpu",
                "--steps",
                "1",
                "--batch",
                "1",
                "--max-rollout",
                "1",
                "--log-every",
                "1",
                "--predictor",
                "graph-transformer",
                "--gt-architecture-version",
                "3",
                "--gt-max-rotation",
                "0.4",
                "--gt-hidden",
                "16",
                "--gt-heads",
                "4",
                "--gt-levels",
                "2",
                "--gt-cluster-size",
                "2",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(train, "compute_S_from_x", side_effect=AssertionError("legacy polar called")),
            ):
                train.main()

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            graph_config = checkpoint["predictor_config"]["graph_transformer"]
            self.assertEqual(graph_config["architecture_version"], 3)
            self.assertEqual(graph_config["max_rotation_update"], 0.4)
            self.assertEqual(checkpoint["args"]["solver_iters"], 1)
            self.assertEqual(checkpoint["decoder_work"]["global_triangular_solves"], 1)
            self.assertEqual(checkpoint["decoder_work"]["local_polar_sweeps"], 0)
            self.assertEqual(
                checkpoint["training_realized_hierarchy_levels"],
                checkpoint["runtime"]["realized_hierarchy_levels"],
            )
            self.assertTrue(any(name.startswith("rotation_head.") for name in checkpoint["state_dict"]))


if __name__ == "__main__":
    unittest.main()
