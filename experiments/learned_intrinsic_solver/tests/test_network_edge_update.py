# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the A02 state-dependent edge update and chunk checkpointing on CPU."""

import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("PyTorch is an optional dependency") from error

from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork, IntrinsicTransformerLayer

GRID = (2, 2, 3)
STATE_DIM = 5


def _inputs(model, batch=2, requires_grad=False):
    count = math.prod(model.cell_counts)
    # Centre axes start at the identity; any warping columns start at zero.
    axes = torch.eye(3, model.target_modes).expand(batch, count, 3, model.target_modes).clone()
    state = torch.randn(batch, count, model.state_feature_dim)
    conditioning = torch.randn(batch, count, model.conditioning_dim)
    edges = {}
    for hop in dict.fromkeys(model.hops):
        indices, _ = model.neighborhood(hop)
        edges[hop] = torch.randn(batch, count, indices.shape[1], model.edge_input_dim)
    if requires_grad:
        state.requires_grad_()
        conditioning.requires_grad_()
        for values in edges.values():
            values.requires_grad_()
    return axes, state, edges, conditioning


def _network(seed, **flags):
    torch.manual_seed(seed)
    return IntrinsicSolverNetwork(GRID, STATE_DIM, hidden_dim=16, num_heads=4, **flags)


def _perturb_heads(model, seed):
    """Make the output sensitive to the block; keep the network deterministic."""
    with torch.no_grad():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.correction_head.weight.normal_(std=0.1)
            model.step_head.weight.normal_(std=0.1)
            for layer in model.layers:
                layer.film.weight.normal_(std=0.01)


def _perturb_edge_update(model, seed):
    with torch.no_grad():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for layer in model.layers:
                layer.edge_update[2].weight.normal_(std=0.1)
                layer.edge_update[2].bias.normal_(std=0.1)


def _edge_update_keys(model):
    return sorted(name for name, _ in model.named_parameters() if ".edge_update." in name)


class TestLayerEdgeUpdate(unittest.TestCase):
    def setUp(self):
        """Seed the layer fixtures for repeatable float32 comparisons."""
        torch.manual_seed(31)

    def _layers(self, **variant_flags):
        torch.manual_seed(3)
        baseline = IntrinsicTransformerLayer(8, 3, num_heads=2, conditioning_dim=2, query_chunk_size=2)
        torch.manual_seed(3)
        variant = IntrinsicTransformerLayer(8, 3, num_heads=2, conditioning_dim=2, query_chunk_size=2, **variant_flags)
        result = variant.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(all(key.startswith("edge_update.") for key in result.missing_keys))
        with torch.no_grad():
            for layer in (baseline, variant):
                layer.film.weight.normal_(std=0.05)
            variant.film.weight.copy_(baseline.film.weight)
        return baseline, variant

    def test_rejects_non_boolean_flags(self):
        """Refuse truthy non-bool flags so configuration typos surface early."""
        with self.assertRaises(ValueError):
            IntrinsicTransformerLayer(8, 3, edge_network=1)
        with self.assertRaises(ValueError):
            IntrinsicTransformerLayer(8, 3, checkpoint_chunks="yes")
        with self.assertRaises(ValueError):
            IntrinsicSolverNetwork(GRID, STATE_DIM, hidden_dim=16, edge_network=None)

    def test_zero_initialized_update_matches_baseline_exactly(self):
        """Start bit-identical to the geometry-only layer, including attention weights."""
        baseline, variant = self._layers(edge_network=True)
        self.assertIsNone(baseline.edge_update)
        self.assertEqual(torch.count_nonzero(variant.edge_update[2].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(variant.edge_update[2].bias).item(), 0)
        self.assertEqual(variant.edge_update[0].in_features, 2 * 8 + 3)
        self.assertEqual(variant.edge_update[2].out_features, 3)
        features = torch.randn(2, 5, 8)
        edges = torch.randn(2, 5, 3, 3)
        conditioning = torch.randn(2, 5, 2)
        indices = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 0, 0]])
        mask = torch.tensor(
            [[True, True, True], [True, True, False], [True, True, True], [True, False, False], [True, True, True]]
        )
        expected, expected_attention = baseline(
            features, edges, indices, mask, conditioning=conditioning, return_attention=True
        )
        actual, attention = variant(features, edges, indices, mask, conditioning=conditioning, return_attention=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(attention, expected_attention, rtol=0, atol=0)

    def test_poisoned_invalid_slots_are_ignored(self):
        """Keep the output unchanged when invalid edge slots hold NaN or inf with the edge update active."""
        _, variant = self._layers(edge_network=True)
        with torch.no_grad():
            variant.edge_update[2].weight.normal_(std=0.2)
            variant.edge_update[2].bias.normal_(std=0.2)
        features = torch.randn(1, 3, 8, requires_grad=True)
        edges = torch.randn(1, 3, 4, 3)
        conditioning = torch.randn(1, 3, 2)
        mask = torch.tensor([[True, True, False, False], [True, False, False, True], [False, False, False, False]])
        indices = torch.tensor([[0, 1, -1, 10000], [1, -8, 10000, 2], [-2, 5, 6, 7]])
        clean = variant(features, edges, indices, mask, conditioning=conditioning)
        poisoned = edges.clone()
        invalid = ~mask.unsqueeze(0)
        poisoned[invalid] = float("nan")
        poisoned[0, 2, :2] = float("inf")
        poisoned[0, 1, 1] = -float("inf")
        poisoned.requires_grad_()
        output = variant(features, poisoned, indices, mask, conditioning=conditioning)
        torch.testing.assert_close(output, clean, rtol=0, atol=0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(poisoned.grad).all())
        self.assertEqual(torch.count_nonzero(poisoned.grad[invalid]).item(), 0)
        self.assertGreater(poisoned.grad[~invalid].abs().sum().item(), 0)

    def test_checkpointed_layer_matches_and_recomputes(self):
        """Recompute each chunk in backward with identical outputs, attention, and gradients."""
        _, variant = self._layers(edge_network=True, checkpoint_chunks=True)
        torch.manual_seed(3)
        reference = IntrinsicTransformerLayer(
            8, 3, num_heads=2, conditioning_dim=2, query_chunk_size=2, edge_network=True
        )
        with torch.no_grad():
            variant.edge_update[2].weight.normal_(std=0.2)
            variant.edge_update[2].bias.normal_(std=0.2)
        reference.load_state_dict(variant.state_dict(), strict=True)
        calls = []
        original = variant._attend_chunk

        def counting(*args):
            calls.append(1)
            return original(*args)

        variant._attend_chunk = counting
        features = torch.randn(2, 5, 8, requires_grad=True)
        other = features.detach().clone().requires_grad_()
        edges = torch.randn(2, 5, 3, 3)
        conditioning = torch.randn(2, 5, 2)
        indices = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 0, 0]])
        mask = torch.tensor(
            [[True, True, True], [True, True, False], [True, True, True], [True, False, False], [True, True, True]]
        )
        expected, expected_attention = reference(
            other, edges, indices, mask, conditioning=conditioning, return_attention=True
        )
        actual, attention = variant(features, edges, indices, mask, conditioning=conditioning, return_attention=True)
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(attention, expected_attention, rtol=1e-6, atol=1e-6)
        expected.square().sum().backward()
        actual.square().sum().backward()
        self.assertEqual(len(calls), 6)
        torch.testing.assert_close(features.grad, other.grad, rtol=1e-6, atol=1e-6)
        for (name, parameter), (other_name, other_parameter) in zip(
            variant.named_parameters(), reference.named_parameters(), strict=True
        ):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, other_parameter.grad, rtol=1e-6, atol=1e-6)
        calls.clear()
        with torch.no_grad():
            inference = variant(features.detach(), edges, indices, mask, conditioning=conditioning)
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(inference, expected.detach(), rtol=1e-6, atol=1e-6)


class TestNetworkEdgeUpdate(unittest.TestCase):
    def setUp(self):
        """Seed the network fixtures for repeatable float32 comparisons."""
        torch.manual_seed(41)

    def _pair(self, target_modes=3, **variant_flags):
        baseline = _network(7, target_modes=target_modes)
        variant = _network(7, target_modes=target_modes, **variant_flags)
        result = variant.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(sorted(result.missing_keys), _edge_update_keys(variant))
        _perturb_heads(baseline, 11)
        _perturb_heads(variant, 11)
        return baseline, variant

    def test_identity_at_initialization(self):
        """Match the baseline exactly with zero-initialized edge updates and shared weights."""
        baseline, variant = self._pair(edge_network=True)
        self.assertFalse(baseline.edge_network)
        self.assertTrue(variant.edge_network)
        self.assertFalse(variant.checkpoint_chunks)
        hidden, edge = 16, 64
        extra = (2 * hidden + edge) * hidden + hidden + hidden * edge + edge
        self.assertEqual(
            sum(p.numel() for p in variant.parameters()) - sum(p.numel() for p in baseline.parameters()),
            extra * len(variant.layers),
        )
        for layer in variant.layers:
            self.assertEqual(torch.count_nonzero(layer.edge_update[2].weight).item(), 0)
            self.assertEqual(torch.count_nonzero(layer.edge_update[2].bias).item(), 0)
        inputs = _inputs(baseline)
        expected = baseline(*inputs)
        actual = variant(*inputs)
        self.assertGreater(torch.count_nonzero(expected.axis_correction).item(), 0)
        for wanted, got in zip(expected, actual, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)

    def test_identity_at_initialization_with_seven_modes(self):
        """Start bit-identical to the geometry-only network when seven target vectors are predicted."""
        baseline, variant = self._pair(target_modes=7, edge_network=True)
        self.assertEqual(variant.target_modes, 7)
        self.assertEqual(variant.correction_head.out_features, 21)
        axes, state, edges, conditioning = _inputs(baseline)
        self.assertEqual(axes.shape, (2, 12, 3, 7))
        axes = axes + 0.1 * torch.randn_like(axes)
        expected = baseline(axes, state, edges, conditioning)
        actual = variant(axes, state, edges, conditioning)
        self.assertEqual(actual.local_target_axes.shape, (2, 12, 3, 7))
        self.assertGreater(torch.count_nonzero(expected.axis_correction[..., 3:]).item(), 0)
        for wanted, got in zip(expected, actual, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        _perturb_edge_update(variant, 13)
        changed = variant(axes, state, edges, conditioning)
        self.assertTrue(torch.isfinite(changed.local_target_axes).all())
        self.assertFalse(torch.allclose(changed.local_target_axes, expected.local_target_axes))

    def test_perturbed_edge_update_changes_output_and_receives_gradients(self):
        """Let the state-dependent path alter predictions and train all of its parameters."""
        baseline, variant = self._pair(edge_network=True)
        inputs = _inputs(baseline, requires_grad=True)
        expected = baseline(*inputs)
        _perturb_edge_update(variant, 13)
        changed = variant(*inputs)
        self.assertTrue(torch.isfinite(changed.local_target_axes).all())
        self.assertFalse(torch.allclose(changed.local_target_axes, expected.local_target_axes))
        self.assertFalse(torch.allclose(changed.step_size, expected.step_size))
        loss = (changed.local_target_axes - (inputs[0] + 0.1)).square().mean() + changed.step_size.mean()
        loss.backward()
        edge_update_parameters = [
            (name, parameter) for name, parameter in variant.named_parameters() if ".edge_update." in name
        ]
        self.assertEqual(len(edge_update_parameters), 4 * len(variant.layers))
        for name, parameter in edge_update_parameters:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        for parameter in variant.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        _, state, edges, conditioning = inputs
        for value in (state, conditioning, *edges.values()):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_poisoned_invalid_slots_leave_network_output_unchanged(self):
        """Ignore NaN and inf in invalid neighbor slots before the shared encoder and the edge update."""
        _, variant = self._pair(edge_network=True)
        _perturb_edge_update(variant, 13)
        axes, state, edges, conditioning = _inputs(variant)
        clean = variant(axes, state, edges, conditioning)
        poisoned = {}
        for hop, values in edges.items():
            _, mask = variant.neighborhood(hop)
            invalid = ~mask
            self.assertGreater(int(invalid.sum()), 0)
            broken = values.clone()
            broken[:, invalid] = float("nan")
            first = invalid.nonzero()[0]
            broken[:, first[0], first[1]] = float("inf")
            poisoned[hop] = broken.requires_grad_()
        output = variant(axes, state, poisoned, conditioning)
        for wanted, got in zip(clean, output, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        output.local_target_axes.sum().backward()
        for hop, values in poisoned.items():
            _, mask = variant.neighborhood(hop)
            self.assertTrue(torch.isfinite(values.grad).all())
            self.assertEqual(torch.count_nonzero(values.grad[:, ~mask]).item(), 0)

    def test_checkpoint_chunks_matches_forward_and_gradients(self):
        """Reproduce outputs and every parameter gradient when chunks are recomputed."""
        plain = _network(7, edge_network=True, query_chunk_size=4)
        recomputed = _network(7, edge_network=True, query_chunk_size=4, checkpoint_chunks=True)
        _perturb_heads(plain, 11)
        _perturb_edge_update(plain, 13)
        recomputed.load_state_dict(plain.state_dict(), strict=True)
        self.assertTrue(recomputed.checkpoint_chunks)
        self.assertTrue(all(layer.checkpoint_chunks for layer in recomputed.layers))
        inputs = _inputs(plain, requires_grad=True)
        axes, state, edges, conditioning = inputs
        other_state = state.detach().clone().requires_grad_()
        other_conditioning = conditioning.detach().clone().requires_grad_()
        other_edges = {hop: values.detach().clone().requires_grad_() for hop, values in edges.items()}
        expected = plain(axes, state, edges, conditioning)
        actual = recomputed(axes, other_state, other_edges, other_conditioning)
        for wanted, got in zip(expected, actual, strict=True):
            torch.testing.assert_close(got, wanted, rtol=1e-6, atol=1e-6)
        target = axes + 0.1
        (expected.local_target_axes - target).square().mean().add(expected.step_size.mean()).backward()
        (actual.local_target_axes - target).square().mean().add(actual.step_size.mean()).backward()
        for (name, parameter), (other_name, other_parameter) in zip(
            plain.named_parameters(), recomputed.named_parameters(), strict=True
        ):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(other_parameter.grad, name)
            torch.testing.assert_close(other_parameter.grad, parameter.grad, rtol=1e-6, atol=1e-6, msg=name)
        torch.testing.assert_close(other_state.grad, state.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(other_conditioning.grad, conditioning.grad, rtol=1e-6, atol=1e-6)
        for hop in edges:
            torch.testing.assert_close(other_edges[hop].grad, edges[hop].grad, rtol=1e-6, atol=1e-6)
        with torch.no_grad():
            inference = recomputed(axes, state, edges, conditioning)
        torch.testing.assert_close(
            inference.local_target_axes, expected.local_target_axes.detach(), rtol=1e-6, atol=1e-6
        )

    def test_multi_hop_schedule_runs_with_both_flags(self):
        """Run hops=(1, 2, 1) with the edge network and checkpointing, starting identical to the baseline."""
        torch.manual_seed(9)
        baseline = IntrinsicSolverNetwork(GRID, STATE_DIM, hidden_dim=16, num_heads=4, hops=(1, 2, 1))
        torch.manual_seed(9)
        variant = IntrinsicSolverNetwork(
            GRID, STATE_DIM, hidden_dim=16, num_heads=4, hops=(1, 2, 1), edge_network=True, checkpoint_chunks=True
        )
        self.assertEqual(len(variant.layers), 3)
        result = variant.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(sorted(result.missing_keys), _edge_update_keys(variant))
        _perturb_heads(baseline, 11)
        _perturb_heads(variant, 11)
        inputs = _inputs(baseline, requires_grad=True)
        expected = baseline(*inputs)
        actual = variant(*inputs)
        self.assertEqual(actual.local_target_axes.shape, (2, 12, 3, 3))
        self.assertEqual(actual.step_size.shape, (2, 12))
        for wanted, got in zip(expected, actual, strict=True):
            torch.testing.assert_close(got, wanted, rtol=1e-6, atol=1e-6)
        _perturb_edge_update(variant, 13)
        changed = variant(*inputs)
        self.assertTrue(torch.isfinite(changed.local_target_axes).all())
        self.assertFalse(torch.allclose(changed.local_target_axes, expected.local_target_axes))
        changed.local_target_axes.square().mean().backward()
        for name, parameter in variant.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
