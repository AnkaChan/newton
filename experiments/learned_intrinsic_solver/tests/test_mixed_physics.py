# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare heterogeneous batches with the existing single-material physics."""

import copy
import importlib.util
import io
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.mixed_physics import MixedHexSolverStep
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep


class TestMixedHexSolverStep(unittest.TestCase):
    def setUp(self):
        """Construct distinct materials and a non-affine clamped cuboid."""
        torch.manual_seed(302)
        self.rest = generate_cuboid((2, 1, 2), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)
        self.dt = 0.01
        self.gravity = (0.0, -9.81, 0.0)
        self.specs = {
            "soft": {"lame_lambda": 700 * 0.2 / (1.2 * 0.6), "lame_mu": 700 / 2.4, "density": 90.0},
            "stiff": {"lame_lambda": 3000 * 0.4 / (1.4 * 0.2), "lame_mu": 3000 / 2.8, "density": 180.0},
        }
        self.network = IntrinsicSolverNetwork(self.rest.cell_counts, 38, hidden_dim=16, edge_hidden_dim=8)
        with torch.no_grad():
            self.network.correction_head.weight.normal_(std=0.003)
            for layer in self.network.layers:
                layer.film.weight.normal_(std=0.01)
        self.step = MixedHexSolverStep(
            self.rest, self.fixed, network=self.network, time_step=self.dt, gravity=self.gravity
        )
        self.addCleanup(self.step.close)
        for name, spec in self.specs.items():
            self.step.register_context(name, **spec)
        self.x = torch.tensor(self.rest.corner_rest_positions, dtype=torch.float32)
        self.x[:, 0] += 0.06 * self.x[:, 2].square()
        self.x[:, 1] += 0.002 * torch.sin(17 * self.x[:, 0]) * self.x[:, 2]
        self.velocity = torch.zeros_like(self.x)
        self.velocity[:, 0] = 0.3 * self.x[:, 2]
        self.velocity[:, 1] = 0.05 * self.x[:, 2]
        self.forces = torch.zeros_like(self.x)
        self.forces[:, 2] = 0.0007

    def _native(self, context_id, positions, velocities, forces):
        model = build_newton_hex_model(self.rest, self.fixed, gravity=self.gravity, **self.specs[context_id])
        state = model.state()
        state.particle_q.assign(positions.detach().numpy())
        state.particle_qd.assign(velocities.detach().numpy())
        state.particle_f.assign(forces.detach().numpy())
        solver = SolverLearnedIntrinsic(model, network=copy.deepcopy(self.network))
        problem = solver.prepare_problem(state, self.dt)
        return solver, problem

    def test_mixed_forward_and_all_parameter_gradients_match_independent_steps(self):
        """Match independent materials with one batched network evaluation and exact pins."""
        ids = ("stiff", "soft", "stiff")
        positions = torch.stack((self.x, self.x, self.x))
        positions[1, :, 0] += 0.02 * positions[1, :, 2].square()
        positions[2, :, 1] -= 0.03 * positions[2, :, 2].square()
        target = positions + torch.tensor([0.0003, -0.001, 0.0001])
        fixed = positions[:, self.fixed].clone()
        reference_network = copy.deepcopy(self.network)
        references = []
        for index, context_id in enumerate(ids):
            step = LearnedHexSolverStep(
                self.rest, self.fixed, network=reference_network, time_step=self.dt, **self.specs[context_id]
            )
            references.append(
                step(positions[index : index + 1], target[index : index + 1], fixed_positions=fixed[index : index + 1])
            )
        with patch.object(self.network, "forward", wraps=self.network.forward) as counted:
            output = self.step(positions, target, ids, fixed_positions=fixed)
        self.assertEqual(counted.call_count, 1)
        for field in ("positions", "local_target_axes", "axis_correction", "step_size", "frames"):
            expected = torch.cat([getattr(value, field) for value in references])
            # Batched float32 SVD changes the frozen frames by a few ulps.
            tolerance = 1e-6 if field in ("frames", "local_target_axes") else 2e-7
            torch.testing.assert_close(getattr(output, field), expected, rtol=2e-5, atol=tolerance)
        self.assertFalse(output.frames.requires_grad)
        torch.testing.assert_close(output.positions[:, self.fixed], fixed, rtol=0, atol=0)
        for field in ("total", "elastic", "inertia"):
            expected = torch.cat([getattr(value.loss, field) for value in references])
            torch.testing.assert_close(getattr(output.loss, field), expected, rtol=8e-5, atol=3e-7)
        output.loss.total.mean().backward()
        torch.cat([value.loss.total for value in references]).mean().backward()
        expected_parameters = dict(reference_network.named_parameters())
        for name, parameter in self.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(parameter.grad, expected_parameters[name].grad, rtol=5e-4, atol=3e-7, msg=name)

    def test_context_creation_is_independent_of_network_and_broadcast_buffers(self):
        """Keep worker-created material state out of weights and DDP broadcast buffers."""
        state_keys = tuple(self.step.state_dict())
        with patch.object(self.network, "parameters", side_effect=AssertionError("worker read weights")):
            with patch.object(self.network, "forward", side_effect=AssertionError("worker called network")):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(self.step.register_context, f"worker-{i}", **spec)
                        for i, spec in enumerate(self.specs.values())
                    ]
                    for future in futures:
                        future.result()
        self.assertEqual(tuple(self.step.state_dict()), state_keys)
        for name, _ in self.step.named_buffers():
            self.assertFalse(
                any(part in name for part in ("lame", "density", "lumped_mass", "conditioning", "fusion")), name
            )
        self.assertEqual(json.loads(json.dumps(self.step.context_specs))["soft"], self.specs["soft"])
        snapshot = self.step.context_specs
        snapshot["soft"]["density"] = -1
        self.assertEqual(self.step.context_specs["soft"]["density"], self.specs["soft"]["density"])

    def test_prepare_matches_native_rigid_candidate_and_original_inertia_target(self):
        """Match native physical preparation without network calls or energy evaluation."""
        solver, problem = self._native("soft", self.x, self.velocity, self.forces)
        expected_candidate = solver.initialize_candidate(problem)
        with patch.object(self.network, "forward", side_effect=AssertionError("prepare called network")):
            with patch.object(self.step, "energy", side_effect=AssertionError("prepare evaluated energy")):
                payload = self.step.prepare("soft", self.x.requires_grad_(), self.velocity, forces=self.forces)
        self.assertEqual(
            set(payload),
            {
                "context_id",
                "physical_positions",
                "velocities",
                "candidate",
                "inertial_prediction",
                "fixed_positions",
                "forces",
            },
        )
        self.assertEqual(payload["context_id"], "soft")
        torch.testing.assert_close(payload["candidate"], expected_candidate[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(payload["inertial_prediction"], problem.inertial_prediction[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(payload["candidate"][self.fixed], self.x[self.fixed], rtol=0, atol=0)
        for name, value in payload.items():
            if name != "context_id":
                self.assertEqual(value.device.type, "cpu")
                self.assertEqual(value.dtype, torch.float32)
                self.assertFalse(value.requires_grad)
        torch.save(payload, io.BytesIO())

    def test_advance_recomputes_native_problem_once_from_committed_candidate(self):
        """Reset velocity from displacement and recompute physical guidance exactly once."""
        payload = self.step.prepare("stiff", self.x, self.velocity, forces=self.forces)
        candidate = payload["candidate"].clone()
        candidate[:, 0] += 0.01 * candidate[:, 2].square()
        payload["candidate"] = candidate
        expected_velocity = (candidate - self.x) / self.dt
        expected_velocity[self.fixed] = 0
        solver, problem = self._native("stiff", candidate, expected_velocity, self.forces)
        with patch.object(self.step, "prepare", wraps=self.step.prepare) as counted:
            advanced = self.step.advance(payload)
        self.assertEqual(counted.call_count, 1)
        torch.testing.assert_close(advanced["physical_positions"], candidate, rtol=0, atol=0)
        torch.testing.assert_close(advanced["velocities"], expected_velocity, rtol=0, atol=0)
        torch.testing.assert_close(advanced["candidate"], solver.initialize_candidate(problem)[0], rtol=2e-6, atol=5e-8)
        torch.testing.assert_close(
            advanced["inertial_prediction"], problem.inertial_prediction[0], rtol=2e-6, atol=5e-8
        )

    def test_invalid_context_and_inversion_are_rejected(self):
        """Reject missing contexts, invalid scalar materials, and inverted Gauss points."""
        with self.assertRaises(ValueError):
            self.step.register_context("soft", **self.specs["soft"])
        with self.assertRaises(ValueError):
            self.step.register_context("invalid", lame_lambda=-1, lame_mu=1, density=1)
        with self.assertRaises(ValueError):
            self.step.energy(self.x[None], self.x[None], ("soft", "stiff"))
        self.step.discard_context("soft")
        self.assertNotIn("soft", self.step.context_specs)
        with self.assertRaises(KeyError):
            self.step.energy(self.x[None], self.x[None], ("soft",))
        self.step.register_context("soft", **self.specs["soft"])
        inverted = self.x.clone()
        inverted[:, 0] *= -1
        with self.assertRaisesRegex(ValueError, "Jacobian"):
            self.step.energy(inverted[None], self.x[None], ("soft",))


if __name__ == "__main__":
    unittest.main()
