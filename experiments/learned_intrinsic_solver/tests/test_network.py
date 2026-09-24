# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise masked geometric attention and the learned target network on CPU."""

import io
import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("PyTorch is an optional dependency") from error

from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork, IntrinsicTransformerLayer


class TestIntrinsicTransformer(unittest.TestCase):
    def setUp(self):
        """Make small CPU examples deterministic without changing global thread settings."""
        torch.manual_seed(17)

    def test_edge_bias_and_value(self):
        """Use edge scores for attention and edge values for the delivered message."""
        layer = IntrinsicTransformerLayer(4, 1, num_heads=1)
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
            layer.edge_bias.weight.fill_(1)
            layer.edge_val.weight[0, 0] = 1
            layer.out_projection.weight.copy_(torch.eye(4))
        features = torch.zeros(1, 2, 4)
        indices = torch.tensor([[0, 1], [1, 0]])
        mask = torch.ones(2, 2, dtype=torch.bool)
        edges = torch.tensor([[[[0.0], [math.log(3)]], [[0.0], [0.0]]]])
        output, attention = layer(features, edges, indices, mask, return_attention=True)
        torch.testing.assert_close(attention[0, 0, 0], torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(output[0, 0], torch.tensor([0.75 * math.log(3), 0, 0, 0]))
        with torch.no_grad():
            layer.edge_val.weight.zero_()
        torch.testing.assert_close(layer(features, edges, indices, mask), features)

    def test_masked_edges_and_empty_row(self):
        """Ignore invalid indices and poisoned edge values in forward and backward."""
        layer = IntrinsicTransformerLayer(8, 3, num_heads=2)
        features = torch.randn(1, 2, 8, requires_grad=True)
        edges = torch.randn(1, 2, 3, 3)
        mask = torch.tensor([[True, False, False], [False, False, False]])
        indices = torch.tensor([[0, -1, 10000], [-8, 10000, -2]])
        edges[~mask.unsqueeze(0)] = float("nan")
        edges.requires_grad_()
        output, attention = layer(features, edges, indices, mask, return_attention=True)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(attention[0, 0, :, 0], torch.ones(2))
        self.assertEqual(torch.count_nonzero(attention[0, 0, :, 1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(attention[0, 1]).item(), 0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(edges.grad).all())
        self.assertEqual(torch.count_nonzero(edges.grad[~mask.unsqueeze(0)]).item(), 0)

    def test_neighbor_slot_permutation(self):
        """Keep output unchanged when neighbor slots and their edge features are reordered."""
        layer = IntrinsicTransformerLayer(8, 3, num_heads=2)
        features = torch.randn(2, 3, 8)
        edges = torch.randn(2, 3, 3, 3)
        indices = torch.tensor([[0, 1, 2], [1, 0, 0], [2, 1, 0]])
        mask = torch.tensor([[True, True, True], [True, True, False], [True, True, True]])
        order = torch.tensor([2, 0, 1])
        expected = layer(features, edges, indices, mask)
        actual = layer(features, edges[:, :, order], indices[:, order], mask[:, order])
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_conditioning_and_backpropagation(self):
        """Propagate derivatives through nodes, edges, and learned FiLM conditioning."""
        layer = IntrinsicTransformerLayer(8, 3, num_heads=2, conditioning_dim=2)
        with torch.no_grad():
            layer.film.weight.normal_(std=0.05)
        features = torch.randn(2, 3, 8, requires_grad=True)
        edges = torch.randn(2, 3, 2, 3, requires_grad=True)
        conditioning = torch.randn(2, 3, 2, requires_grad=True)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0]])
        mask = torch.ones(3, 2, dtype=torch.bool)
        output = layer(features, edges, indices, mask, conditioning=conditioning)
        output.square().mean().backward()
        for value in (features, edges, conditioning):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        changed = layer(features.detach(), edges.detach(), indices, mask, conditioning=conditioning.detach() + 1)
        self.assertFalse(torch.allclose(changed, output))

    def test_query_chunking_and_cell_permutation(self):
        """Preserve attention and derivatives across query chunk sizes and cell numbering."""
        layer = IntrinsicTransformerLayer(8, 3, num_heads=2, query_chunk_size=1)
        unsplit = IntrinsicTransformerLayer(8, 3, num_heads=2, query_chunk_size=16)
        unsplit.load_state_dict(layer.state_dict())
        features = torch.randn(1, 3, 8, requires_grad=True)
        other_features = features.detach().clone().requires_grad_()
        edges = torch.randn(1, 3, 2, 3)
        indices = torch.tensor([[0, 2], [1, 0], [2, 1]])
        mask = torch.ones(3, 2, dtype=torch.bool)
        chunked, attention = layer(features, edges, indices, mask, return_attention=True)
        full, full_attention = unsplit(other_features, edges, indices, mask, return_attention=True)
        torch.testing.assert_close(chunked, full)
        torch.testing.assert_close(attention, full_attention)
        chunked.square().sum().backward()
        full.square().sum().backward()
        torch.testing.assert_close(features.grad, other_features.grad)
        order = torch.tensor([2, 0, 1])
        inverse = torch.argsort(order)
        permuted = layer(features[:, order], edges[:, order], inverse[indices[order]], mask[order])
        torch.testing.assert_close(permuted, chunked[:, order])


class TestIntrinsicSolverNetwork(unittest.TestCase):
    def setUp(self):
        """Seed the network fixture for repeatable float32 comparisons."""
        torch.manual_seed(23)

    @staticmethod
    def _inputs(model, batch=2):
        count = math.prod(model.cell_counts)
        axes = torch.eye(3).expand(batch, count, 3, 3).clone()
        state = torch.randn(batch, count, model.state_feature_dim)
        conditioning = torch.randn(batch, count, model.conditioning_dim)
        edges = {}
        for hop in set(model.hops):
            indices, _ = model.neighborhood(hop)
            edges[hop] = torch.randn(batch, count, indices.shape[1], model.edge_input_dim)
        return axes, state, edges, conditioning

    def test_initial_target_and_state_dict(self):
        """Start with unchanged local axes and preserve learned outputs across serialization."""
        model = IntrinsicSolverNetwork((2, 3, 4), 5, hidden_dim=16, num_heads=4)
        inputs = self._inputs(model)
        output = model(*inputs)
        torch.testing.assert_close(output.local_target_axes, inputs[0], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(output.axis_correction).item(), 0)
        self.assertEqual(output.step_size.shape, (2,))
        self.assertTrue(((output.step_size > 0) & (output.step_size < 1)).all())
        with torch.no_grad():
            model.correction_head.weight.normal_(std=0.1)
        expected = model(*inputs)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = IntrinsicSolverNetwork((2, 3, 4), 5, hidden_dim=16, num_heads=4)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        for actual, wanted in zip(restored(*inputs), expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_batch_independence_and_output_bounds(self):
        """Keep independent objects separate and bound each cell's nine-component correction."""
        model = IntrinsicSolverNetwork((2, 2, 3), 5, hidden_dim=16, max_step_size=0.2)
        with torch.no_grad():
            model.correction_head.weight.normal_()
        axes, state, edges, conditioning = self._inputs(model)
        together = model(axes, state, edges, conditioning)
        first = model(axes[:1], state[:1], {h: e[:1] for h, e in edges.items()}, conditioning[:1])
        torch.testing.assert_close(together.local_target_axes[:1], first.local_target_axes)
        norms = torch.linalg.vector_norm(together.axis_correction.flatten(-2), dim=-1)
        self.assertTrue((norms < 1).all())
        self.assertTrue(((together.step_size > 0) & (together.step_size <= 0.2)).all())
        expected = axes + together.step_size[:, None, None, None] * together.axis_correction
        torch.testing.assert_close(together.local_target_axes, expected)

    def test_solver_network_gradients(self):
        """Train through all three local blocks and the shared object step controller."""
        model = IntrinsicSolverNetwork((2, 2, 3), 5, hidden_dim=16)
        with torch.no_grad():
            model.correction_head.weight.normal_(std=0.1)
            model.step_head.weight.normal_(std=0.1)
            for layer in model.layers:
                layer.film.weight.normal_(std=0.01)
        axes, state, edges, conditioning = self._inputs(model)
        state.requires_grad_()
        conditioning.requires_grad_()
        for values in edges.values():
            values.requires_grad_()
        output = model(axes, state, edges, conditioning)
        loss = (output.local_target_axes - (axes + 0.1)).square().mean()
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(state.grad.abs().sum().item(), 0)
        self.assertGreater(conditioning.grad.abs().sum().item(), 0)
        for values in edges.values():
            self.assertGreater(values.grad.abs().sum().item(), 0)

    def test_network_masks_before_edge_encoder(self):
        """Exclude NaN padding before the shared edge MLP as well as before attention."""
        model = IntrinsicSolverNetwork((1, 1, 1), 0, hidden_dim=16)
        axes, state, edges, conditioning = self._inputs(model, batch=1)
        for hop, values in edges.items():
            _, mask = model.neighborhood(hop)
            values[:, ~mask] = float("nan")
            values.requires_grad_()
        output = model(axes, state, edges, conditioning)
        self.assertTrue(torch.isfinite(output.local_target_axes).all())
        output.local_target_axes.sum().backward()
        for values in edges.values():
            self.assertTrue(torch.isfinite(values.grad).all())

    def test_full_grid_float32_inference(self):
        """Run the requested 4,000-cell grid with the fixed-slot hop schedule."""
        model = IntrinsicSolverNetwork((10, 10, 40), 5)
        with torch.no_grad():
            with torch.random.fork_rng(devices=[]):
                inputs = self._inputs(model, batch=1)
            output = model(*inputs)
        self.assertEqual(output.local_target_axes.shape, (1, 4000, 3, 3))
        self.assertEqual(output.local_target_axes.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.local_target_axes).all())
        self.assertEqual([model.neighborhood(h)[0].shape[1] for h in model.hops], [27, 27, 27])


if __name__ == "__main__":
    unittest.main()
