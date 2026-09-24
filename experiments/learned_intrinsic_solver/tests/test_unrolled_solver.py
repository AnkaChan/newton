# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for complete learned-optimizer unrolls and first-order gradients."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch dependency is not installed")

import torch  # noqa: TID253

from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep
from experiments.learned_intrinsic_solver.unrolled_solver import UnrolledHexSolver


class TestUnrolledHexSolver(unittest.TestCase):
    def _problem(self, *, nonzero_head=False):
        torch.manual_seed(19)
        rest = generate_cuboid((1, 1, 2), cell_size=0.025)
        fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
        network = IntrinsicSolverNetwork(
            rest.cell_counts,
            38,
            hidden_dim=16,
            edge_hidden_dim=8,
            num_heads=4,
            hops=(1, 1, 1),
            max_step_size=0.01,
            query_chunk_size=16,
        )
        if nonzero_head:
            with torch.no_grad():
                network.correction_head.weight.normal_(std=1e-4)
                network.correction_head.bias.normal_(std=1e-4)
        step = LearnedHexSolverStep(
            rest,
            fixed,
            lame_lambda=288461.53846,
            lame_mu=192307.69231,
            density=1000.0,
            time_step=1 / 300,
            network=network,
        )
        positions = torch.from_numpy(rest.corner_rest_positions.astype(np.float32))[None].clone()
        inertial = positions.clone()
        inertial[:, :, 1] -= 0.0002
        pins = positions[:, fixed].clone()
        return step, positions, inertial, pins

    def test_k1_matches_existing_step_and_k2_k4_k32_report_all_energies(self):
        """Match one baseline proposal and retain exactly K+1 physical energies."""
        step, positions, inertial, pins = self._problem()
        module = UnrolledHexSolver(step)
        baseline = step(positions, inertial, fixed_positions=pins)
        single = module(positions, inertial, fixed_positions=pins, iterations=1)
        torch.testing.assert_close(single.final_positions, baseline.positions, rtol=0, atol=0)
        torch.testing.assert_close(single.energies[:, 0], step.energy(positions, inertial).total, rtol=0, atol=0)
        torch.testing.assert_close(single.energies[:, 1], baseline.loss.total, rtol=0, atol=0)
        self.assertEqual(single.performed_iterations, 1)
        self.assertEqual(single.per_sample_objective.shape, (1,))
        torch.testing.assert_close(single.objective, single.per_sample_objective.mean(), rtol=0, atol=0)
        scale = single.energies[:, 0].detach().clamp_min(1.0)
        expected = ((single.energies[:, 1] - single.energies[:, 0].detach()) / scale).mean()
        penalty = (torch.relu(single.energies[:, 1] - single.energies[:, 0]) / scale).mean()
        torch.testing.assert_close(single.objective, expected + penalty, rtol=0, atol=0)
        no_penalty = UnrolledHexSolver(step, energy_increase_weight=0)(
            positions, inertial, fixed_positions=pins, iterations=1
        )
        torch.testing.assert_close(no_penalty.objective, expected, rtol=0, atol=0)
        for count in (2, 4, 32):
            result = module(positions, inertial, fixed_positions=pins, iterations=count)
            self.assertEqual(result.energies.shape, (1, count + 1))
            self.assertEqual(result.performed_iterations, count)
            self.assertTrue(torch.isfinite(result.energies).all())
            torch.testing.assert_close(result.final_positions[:, step.fixed_indices], pins, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "iterations"):
            module(positions, inertial, fixed_positions=pins, iterations=33)

    def test_final_energy_gradient_reaches_first_fused_positions(self):
        """Retain cross-iteration gradients only when explicitly requested."""
        step, positions, inertial, pins = self._problem(nonzero_head=True)
        module = UnrolledHexSolver(step, detach_iterations=False)
        captured = []
        original_forward = step.forward

        def capture(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            captured.append(result.positions)
            return result

        with patch.object(step, "forward", capture):
            result = module(positions, inertial, fixed_positions=pins, iterations=2)
        first_position_gradient = torch.autograd.grad(result.energies[:, -1].sum(), captured[0], retain_graph=True)[0]
        self.assertGreater(first_position_gradient.norm().item(), 0)
        result.objective.backward()
        gradient = step.network.correction_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)

    def test_default_cuts_cross_iteration_gradient_but_trains_every_update(self):
        """Train all proposals locally without connecting their carried positions."""
        step, positions, inertial, pins = self._problem(nonzero_head=True)
        captured = []

        def capture(_module, _args, output):
            output.positions.retain_grad()
            captured.append(output.positions)

        handle = step.register_forward_hook(capture)
        try:
            result = UnrolledHexSolver(step)(positions, inertial, fixed_positions=pins, iterations=4)
        finally:
            handle.remove()
        cross_gradient = torch.autograd.grad(
            result.energies[:, -1].sum(), captured[0], allow_unused=True, retain_graph=True
        )[0]
        self.assertTrue(cross_gradient is None or torch.count_nonzero(cross_gradient).item() == 0)
        for value in captured:
            value.grad = None
        result.objective.backward()
        for value in captured:
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.norm().item(), 0)

    def test_streamed_backward_matches_independent_local_losses(self):
        """Average every local gradient, preserve weights, and return detached diagnostics."""
        step, positions, inertial, pins = self._problem(nonzero_head=True)
        count = 4
        with torch.no_grad():
            states = [positions]
            energies = [step.energy(positions, inertial).total]
            for _ in range(count):
                update = step(states[-1], inertial, fixed_positions=pins)
                states.append(update.positions)
                energies.append(update.loss.total)
        scale = energies[0].clamp_min(1.0)
        for index in range(count):
            update = step(states[index], inertial, fixed_positions=pins)
            loss = (update.loss.total - energies[0] + torch.relu(update.loss.total - energies[index])) / scale
            (loss.mean() / count).backward()
        expected = {
            name: parameter.grad.clone() for name, parameter in step.named_parameters() if parameter.grad is not None
        }
        weights = {name: parameter.detach().clone() for name, parameter in step.named_parameters()}
        # These may be nonleaf tensors from physical prediction. Streaming
        # parameter training must not retain or traverse their outside graph.
        source = positions.clone().requires_grad_()
        prediction = inertial + source.square() * 0
        for checkpoint in (False, True):
            step.zero_grad(set_to_none=True)
            module = UnrolledHexSolver(step, checkpoint_activations=checkpoint)
            for repeat in (1, 2):
                result = module.backward_detached(source, prediction, fixed_positions=pins, iterations=count)
                for name, parameter in step.named_parameters():
                    torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
                    if name in expected:
                        torch.testing.assert_close(parameter.grad, expected[name] * repeat, rtol=2e-5, atol=1e-7)
                self.assertIsNone(source.grad)
                for value in result:
                    if isinstance(value, torch.Tensor):
                        self.assertFalse(value.requires_grad)
                torch.testing.assert_close(result.final_positions, states[-1], rtol=0, atol=0)
                torch.testing.assert_close(result.energies, torch.stack(energies, dim=1), rtol=0, atol=0)

    def test_streamed_backward_releases_saved_activations_each_iteration(self):
        """Bound saved activation storage by one proposal rather than the sequence length."""
        step, positions, inertial, pins = self._problem(nonzero_head=True)
        module = UnrolledHexSolver(step)
        peaks = []
        for count in (1, 4):
            counts = {"live": 0, "peak": 0}

            class SavedTensor:
                def __init__(self, value, counters=counts):
                    self.value = value
                    self.counters = counters
                    counters["live"] += 1
                    counters["peak"] = max(counters["peak"], counters["live"])

                def __del__(self):
                    self.counters["live"] -= 1

            step.zero_grad(set_to_none=True)
            with torch.autograd.graph.saved_tensors_hooks(SavedTensor, lambda saved: saved.value):
                module.backward_detached(positions, inertial, fixed_positions=pins, iterations=count)
            self.assertEqual(counts["live"], 0)
            peaks.append(counts["peak"])
        self.assertGreater(peaks[0], 0)
        self.assertEqual(peaks[0], peaks[1])

    def test_streamed_backward_rejects_connected_or_no_grad_usage(self):
        """Reject modes incompatible with immediate local backward."""
        step, positions, inertial, pins = self._problem()
        with self.assertRaisesRegex(ValueError, "detach_iterations"):
            UnrolledHexSolver(step, detach_iterations=False).backward_detached(
                positions, inertial, fixed_positions=pins
            )
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "grad"):
            UnrolledHexSolver(step).backward_detached(positions, inertial, fixed_positions=pins)

    def test_nonreentrant_checkpoint_matches_plain_k4_gradient(self):
        """Recompute each inner step in backward without changing first-order gradients."""
        step, positions, inertial, pins = self._problem(nonzero_head=True)
        plain = UnrolledHexSolver(step)
        memory = UnrolledHexSolver(step, checkpoint_activations=True)
        result_plain = plain(positions, inertial, fixed_positions=pins, iterations=4)
        result_plain.objective.backward()
        reference = step.network.correction_head.weight.grad.detach().clone()
        step.zero_grad(set_to_none=True)
        result_memory = memory(positions, inertial, fixed_positions=pins, iterations=4)
        result_memory.objective.backward()
        torch.testing.assert_close(result_memory.energies, result_plain.energies, rtol=0, atol=0)
        torch.testing.assert_close(result_memory.objective, result_plain.objective, rtol=0, atol=0)
        torch.testing.assert_close(step.network.correction_head.weight.grad, reference, rtol=1e-5, atol=1e-7)
        self.assertEqual(result_memory.performed_iterations, 4)

    def test_detached_energy_target_retains_network_feature_gradient(self):
        """Remove Y's direct energy gradient while preserving its network-feature path."""
        step, positions, inertial, pins = self._problem(nonzero_head=True)
        inertial.requires_grad_()
        module = UnrolledHexSolver(step)
        full = module(positions, inertial, fixed_positions=pins, iterations=1)
        detached = module(positions, inertial, fixed_positions=pins, iterations=1, detach_energy_target=True)
        torch.testing.assert_close(full.final_positions, detached.final_positions, rtol=0, atol=0)
        torch.testing.assert_close(full.energies, detached.energies, rtol=0, atol=0)
        full_gradient = torch.autograd.grad(full.energies[:, -1].sum(), inertial, retain_graph=True)[0]
        feature_gradient = torch.autograd.grad(detached.energies[:, -1].sum(), inertial, retain_graph=True)[0]
        direct_gradient = torch.autograd.grad(
            step.energy(detached.final_positions.detach(), inertial).total.sum(), inertial
        )[0]
        self.assertGreater(feature_gradient.norm().item(), 0)
        torch.testing.assert_close(full_gradient - feature_gradient, direct_gradient, rtol=1e-4, atol=1e-5)

        plain = module(positions, inertial, fixed_positions=pins, iterations=2, detach_energy_target=True)
        memory = UnrolledHexSolver(step, checkpoint_activations=True)(
            positions, inertial, fixed_positions=pins, iterations=2, detach_energy_target=True
        )
        torch.testing.assert_close(plain.objective, memory.objective, rtol=0, atol=0)
        plain_gradient = torch.autograd.grad(plain.objective, step.network.correction_head.weight, retain_graph=True)[0]
        memory_gradient = torch.autograd.grad(memory.objective, step.network.correction_head.weight)[0]
        torch.testing.assert_close(plain_gradient, memory_gradient, rtol=1e-5, atol=1e-7)

    def test_increase_penalty_freezes_each_preceding_energy(self):
        """Penalize later increases without giving a gradient that raises prior energy."""
        step, positions, inertial, pins = self._problem()
        module = UnrolledHexSolver(step)
        initial = torch.tensor([1.0], requires_grad=True)
        first = torch.tensor([3.0], requires_grad=True)
        second = torch.tensor([5.0], requires_grad=True)
        later_energies = iter((first, second))

        def next_energy(current, prediction, fixed, *, detach_energy_target, previous_positions=None):
            return current, next(later_energies)

        with (
            patch.object(step.energy, "forward", return_value=SimpleNamespace(total=initial)),
            patch.object(module, "_one_step", next_energy),
        ):
            result = module(positions, inertial, fixed_positions=pins, iterations=2)
        before_gradient, first_gradient, second_gradient = torch.autograd.grad(
            result.mean_energy_increase_penalty, (initial, first, second), allow_unused=True
        )
        torch.testing.assert_close(before_gradient, torch.tensor([0.0]), rtol=0, atol=0)
        torch.testing.assert_close(first_gradient, torch.tensor([0.5]), rtol=0, atol=0)
        torch.testing.assert_close(second_gradient, torch.tensor([0.5]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
