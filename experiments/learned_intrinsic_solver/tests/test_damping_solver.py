# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise metric damping through learned updates and physical time advances."""

import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.input_assembly import OptimizerHistory
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic
from experiments.learned_intrinsic_solver.physical_rollout import PhysicalRollout
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep
from experiments.learned_intrinsic_solver.unrolled_solver import UnrolledHexSolver


class TestDampingSolver(unittest.TestCase):
    def setUp(self):
        """Create two clamped cells and fix the random network initialization."""
        torch.manual_seed(317)
        self.rest = generate_cuboid((1, 1, 2), cell_size=0.1)
        self.fixed = np.flatnonzero(self.rest.corner_rest_positions[:, 2] == 0)

    def _network(self, *, nonzero=False, dtype=torch.float64, state_feature_dim=None, conditioning_dim=None):
        network = IntrinsicSolverNetwork(
            self.rest.cell_counts,
            features.STATE_FEATURE_DIM if state_feature_dim is None else state_feature_dim,
            conditioning_dim=features.CONDITIONING_DIM if conditioning_dim is None else conditioning_dim,
            hidden_dim=16,
            edge_hidden_dim=8,
            max_step_size=0.01,
        ).to(dtype=dtype)
        if nonzero:
            with torch.no_grad():
                network.correction_head.weight.normal_(std=0.01)
        return network

    def _step(self, *, damping=4.0, nonzero=False, dtype=torch.float64, network=None):
        if network is None:
            network = self._network(nonzero=nonzero, dtype=dtype)
        step = LearnedHexSolverStep(
            self.rest,
            self.fixed,
            lame_lambda=700.0,
            lame_mu=300.0,
            density=100.0,
            damping=damping,
            time_step=0.02,
            network=network,
            dtype=dtype,
        )
        previous = torch.tensor(self.rest.corner_rest_positions, dtype=dtype)[None]
        candidate = previous.clone()
        candidate[..., 0] += 0.05 * candidate[..., 2]
        inertial = previous.clone()
        inertial[..., 1] -= 0.0002
        return step, candidate, inertial, previous

    def test_damping_enters_energy_conditioning_and_physical_change_block(self):
        """Expose the viscosity channel and the physical axis change, and evaluate the metric damping energy."""
        step, candidate, inertial, previous = self._step()
        inputs = step.prepare_inputs(candidate, inertial, previous_positions=previous)
        self.assertEqual(inputs.state_features.shape, (1, 2, features.STATE_FEATURE_DIM))
        torch.testing.assert_close(inputs.conditioning[..., 5], candidate.new_full((1, 2), np.log1p(4 / 6)))
        change = inputs.frames @ inputs.state_features[..., 9:18].reshape(1, 2, 3, 3)
        expected_change = torch.zeros(1, 2, 3, 3, dtype=candidate.dtype)
        expected_change[..., 0, 2] = 0.05
        torch.testing.assert_close(change, expected_change, rtol=0, atol=1e-12)
        undamped, _, _, _ = self._step(damping=0)
        undamped.network.load_state_dict(step.network.state_dict())
        plain = undamped.prepare_inputs(candidate, inertial, previous_positions=previous)
        torch.testing.assert_close(plain.state_features[..., :18], inputs.state_features[..., :18], rtol=0, atol=0)
        torch.testing.assert_close(plain.conditioning[..., :5], inputs.conditioning[..., :5], rtol=0, atol=0)
        torch.testing.assert_close(plain.conditioning[..., 5], torch.zeros(1, 2, dtype=candidate.dtype))
        # Damping changes the physical gradient, hence the normalized gradient block and its log RMS.
        self.assertGreater((plain.state_features[..., 18:27] - inputs.state_features[..., 18:27]).abs().max(), 0)
        saved_y = inertial.clone()
        output = step(candidate, inertial, previous_positions=previous)
        expected_energy = 0.002 * 4 / (2 * 0.02) * (2 * 0.05**2 + 0.05**4)
        torch.testing.assert_close(output.loss.damping, candidate.new_tensor([expected_energy]))
        torch.testing.assert_close(inertial, saved_y, rtol=0, atol=0)
        output.loss.total.sum().backward()
        gradient = step.network.correction_head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)
        with self.assertRaises(TypeError):
            step(candidate, inertial)
        with self.assertRaisesRegex(ValueError, "previous_positions"):
            step(candidate, inertial, previous_positions=None)

    def test_zero_damping_shares_the_schema_and_legacy_networks_are_rejected(self):
        """Keep zero-damping energy checkpoints strict-loadable and refuse legacy 38/5 and 86/6 networks."""
        step, candidate, inertial, previous = self._step(damping=0)
        state = step.state_dict()
        state.pop("energy.damping", None)
        step.load_state_dict(state, strict=True)
        baseline = step(candidate, inertial, previous_positions=previous)
        repeated = step(candidate, inertial, previous_positions=previous)
        torch.testing.assert_close(baseline.positions, repeated.positions, rtol=0, atol=0)
        torch.testing.assert_close(baseline.loss.total, repeated.loss.total, rtol=0, atol=0)
        torch.testing.assert_close(baseline.loss.damping, torch.zeros(1, dtype=candidate.dtype), rtol=0, atol=0)
        for state_dim, conditioning_dim in ((38, 5), (86, 6)):
            with self.subTest(schema=(state_dim, conditioning_dim)), self.assertRaisesRegex(ValueError, "schema"):
                self._step(network=self._network(state_feature_dim=state_dim, conditioning_dim=conditioning_dim))
            with (
                self.subTest(schema=(state_dim, conditioning_dim), damping=0),
                self.assertRaisesRegex(ValueError, "schema"),
            ):
                self._step(
                    damping=0, network=self._network(state_feature_dim=state_dim, conditioning_dim=conditioning_dim)
                )
        damped, _, _, _ = self._step(network=self._network())
        self.assertEqual(
            damped.prepare_inputs(candidate, inertial, previous_positions=previous).conditioning.shape[-1],
            features.CONDITIONING_DIM,
        )

    def test_energy_anchor_detachment_preserves_feature_gradients(self):
        """Detach only the physical anchor energy path while retaining its network path."""
        step, candidate, inertial, previous = self._step()
        previous.requires_grad_()
        connected = step(candidate, inertial, previous_positions=previous)
        direct = torch.autograd.grad(connected.loss.total.sum(), previous)[0]
        self.assertGreater(direct.norm().item(), 0)
        detached = step(candidate, inertial, previous_positions=previous, detach_energy_target=True)
        stopped = torch.autograd.grad(detached.loss.total.sum(), previous)[0]
        torch.testing.assert_close(stopped, torch.zeros_like(stopped), rtol=0, atol=0)
        with torch.no_grad():
            step.network.correction_head.weight.normal_(std=0.01)
        feature_connected = step(candidate, inertial, previous_positions=previous, detach_energy_target=True)
        feature_gradient = torch.autograd.grad(feature_connected.positions.square().sum(), previous)[0]
        self.assertGreater(feature_gradient.norm().item(), 0)
        self.assertTrue(torch.isfinite(feature_gradient).all())

    def test_unroll_keeps_anchor_across_candidates_and_checkpoint_backward(self):
        """Match a fixed-anchor manual solve that carries history, with and without activation recomputation."""
        step, candidate, inertial, previous = self._step(nonzero=True)
        manual = candidate
        history = None
        energies = [step.energy(manual, inertial, previous_positions=previous).total]
        for _ in range(3):
            result = step(manual, inertial, previous_positions=previous, history=history)
            manual = result.positions.detach()
            history = OptimizerHistory(
                result.axis_gradient_world, result.achieved_axis_update_world, torch.tensor([True])
            )
            energies.append(result.loss.total.detach())
        expected = torch.stack(energies, dim=1).detach()
        gradients = []
        for checkpoint in (False, True):
            step.zero_grad(set_to_none=True)
            unrolled = UnrolledHexSolver(step, checkpoint_activations=checkpoint)
            result = unrolled(candidate, inertial, previous_positions=previous, iterations=3)
            torch.testing.assert_close(result.energies, expected)
            torch.testing.assert_close(result.history.axis_gradient_world, history.axis_gradient_world)
            result.objective.backward()
            gradients.append({name: parameter.grad.clone() for name, parameter in step.network.named_parameters()})
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name])
        step.zero_grad(set_to_none=True)
        streamed = UnrolledHexSolver(step).backward_detached(
            candidate, inertial, previous_positions=previous, iterations=3
        )
        torch.testing.assert_close(streamed.energies, expected)
        for name, parameter in step.network.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[0][name])
        self.assertFalse(streamed.final_positions.requires_grad)

    def _native(self, *, damping=4.0, nonzero=False):
        model = build_newton_hex_model(
            self.rest,
            self.fixed,
            lame_lambda=700.0,
            lame_mu=300.0,
            density=100.0,
            damping=damping,
            gravity=(0, -9.81, 0),
        )
        solver = SolverLearnedIntrinsic(model, network=self._network(nonzero=nonzero, dtype=torch.float32))
        state = model.state()
        velocity = np.zeros_like(self.rest.corner_rest_positions, dtype=np.float32)
        velocity[:, 0] = 0.5 * self.rest.corner_rest_positions[:, 2]
        state.particle_qd.assign(velocity)
        return model, solver, state

    def test_native_metadata_and_problem_objective(self):
        """Carry native damping and the snapshotted physical start into every update."""
        model, solver, state = self._native(damping=[3.0, 7.0])
        np.testing.assert_array_equal(model.learned_intrinsic.damping.numpy(), [3.0, 7.0])
        problem = solver.prepare_problem(state, 0.02)
        candidate = problem.previous_positions.clone()
        candidate[..., 0] += 0.05 * candidate[..., 2]
        expected = 0.001 * (3 + 7) / (2 * 0.02) * (2 * 0.05**2 + 0.05**4)
        torch.testing.assert_close(problem.objective(candidate).damping, candidate.new_tensor([expected]))
        update = solver.propose_update(candidate, problem)
        torch.testing.assert_close(update.loss.damping, candidate.new_tensor([expected]))
        state.particle_q.assign(np.zeros_like(state.particle_q.numpy()))
        torch.testing.assert_close(problem.objective(candidate).damping, candidate.new_tensor([expected]))

    def test_native_default_uses_revised_schema_and_legacy_metadata_falls_back(self):
        """Use one revised schema for any damping and interpret an old model's absent damping as zero."""
        model, _, _ = self._native()
        default = SolverLearnedIntrinsic(model).network
        self.assertEqual(
            (default.state_feature_dim, default.conditioning_dim),
            (features.STATE_FEATURE_DIM, features.CONDITIONING_DIM),
        )
        del model.learned_intrinsic.damping
        legacy = SolverLearnedIntrinsic(model)
        self.assertEqual(legacy.network.state_feature_dim, features.STATE_FEATURE_DIM)
        torch.testing.assert_close(legacy._step_for_dt(0.02).energy.damping, torch.zeros(2))

    def test_connected_physical_rollout_uses_each_physical_start(self):
        """Preserve cross-step derivatives while holding the correct anchor in each solve."""
        _, solver, state = self._native(nonzero=True)
        step = solver._step_for_dt(0.02)
        unrolled = UnrolledHexSolver(step, detach_iterations=False, checkpoint_activations=True)
        rollout = PhysicalRollout(solver, unrolled, time_step=0.02, detach_energy_target=False)
        positions = torch.from_numpy(state.particle_q.numpy())[None].requires_grad_()
        velocities = torch.from_numpy(state.particle_qd.numpy())[None]
        window = next(rollout.windows(positions, velocities, physical_steps=2, iterations=2, gradient_window=2))
        for physical in window.steps:
            expected = step.energy(
                physical.positions, physical.inertial_prediction, previous_positions=physical.previous_positions
            ).total
            torch.testing.assert_close(physical.energies[:, -1], expected)
        cross_gradient = torch.autograd.grad(window.steps[1].energies[:, -1].sum(), window.steps[0].positions)[0]
        self.assertTrue(torch.isfinite(cross_gradient).all())
        self.assertGreater(cross_gradient.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
