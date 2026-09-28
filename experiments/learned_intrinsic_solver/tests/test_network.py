# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise masked geometric attention and the learned target network on CPU."""

import io
import math
import unittest
from dataclasses import asdict

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("PyTorch is an optional dependency") from error

from experiments.learned_intrinsic_solver import features
from experiments.learned_intrinsic_solver.contact_network import ContactEncoder
from experiments.learned_intrinsic_solver.distributed_probe import ProbeConfig, _network
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork, IntrinsicTransformerLayer
from experiments.learned_intrinsic_solver.train_smoke import TrainSmokeConfig


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
        # Centre axes start at the identity; any warping columns start at zero.
        axes = torch.eye(3, model.target_modes).expand(batch, count, 3, model.target_modes).clone()
        state = torch.randn(batch, count, model.state_feature_dim)
        conditioning = torch.randn(batch, count, model.conditioning_dim)
        edges = {}
        for hop in set(model.hops):
            indices, _ = model.neighborhood(hop)
            edges[hop] = torch.randn(batch, count, indices.shape[1], model.edge_input_dim)
        return axes, state, edges, conditioning

    def test_initial_target_and_state_dict(self):
        """Start with unchanged axes and a uniform half-maximum per-cell step; preserve outputs across saves."""
        model = IntrinsicSolverNetwork((2, 3, 4), 5, hidden_dim=16, num_heads=4)
        inputs = self._inputs(model)
        output = model(*inputs)
        torch.testing.assert_close(output.local_target_axes, inputs[0], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(output.axis_correction).item(), 0)
        self.assertEqual(output.step_size.shape, (2, 24))
        torch.testing.assert_close(output.step_size, torch.full((2, 24), 0.5), rtol=0, atol=0)
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

    def test_default_one_layer_radius_one_and_training_config(self):
        """Use one masked 27-slot neighborhood with the revised 61-feature, seven-channel width by default."""
        model = IntrinsicSolverNetwork((3, 3, 3), features.STATE_FEATURE_DIM)
        self.assertEqual(model.conditioning_dim, features.CONDITIONING_DIM)
        self.assertEqual(TrainSmokeConfig().hops, (1,))
        self.assertEqual(model.hops, (1,))
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 323342)
        self.assertEqual(model.node_encoder[0].in_features, 9 + features.STATE_FEATURE_DIM)
        self.assertFalse(model.contact_tokens)
        self.assertIsNone(model.contact_encoder)
        self.assertEqual(model.condition_encoder[0].in_features, features.CONDITIONING_DIM)
        indices, mask = model.neighborhood(1)
        self.assertEqual(indices.shape, (27, 27))
        self.assertEqual(int(mask[13].sum()), 27)
        self.assertEqual(int(mask[0].sum()), 8)

    def test_saved_explicit_three_layer_config_restores_strictly(self):
        """Keep old checkpoint architecture when saved hops explicitly name three blocks."""
        saved_config = asdict(TrainSmokeConfig(hops=(1, 1, 1)))
        legacy = IntrinsicSolverNetwork((2, 2, 2), 38, hops=saved_config["hops"])
        stream = io.BytesIO()
        torch.save({"config": saved_config, "network_state": legacy.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored_config = TrainSmokeConfig(**saved["config"])
        self.assertEqual(restored_config.hops, (1, 1, 1))
        restored = IntrinsicSolverNetwork((2, 2, 2), 38, hops=restored_config.hops)
        restored.load_state_dict(saved["network_state"], strict=True)
        self.assertEqual(len(restored.layers), 3)
        with self.assertRaises(RuntimeError):
            IntrinsicSolverNetwork((2, 2, 2), 38).load_state_dict(saved["network_state"], strict=True)

    def test_distributed_probe_uses_current_default_architecture(self):
        """Build the active diagnostic probe with the same single block."""
        model = _network(ProbeConfig(world_size=1, batch_size=1, cell_counts=(2, 2, 2)), torch.device("cpu"))
        self.assertEqual(model.hops, (1,))
        self.assertEqual(len(model.layers), 1)

    def test_batch_independence_and_output_bounds(self):
        """Keep independent objects separate and bound each cell's nine-component correction."""
        model = IntrinsicSolverNetwork((2, 2, 3), 5, hidden_dim=16, max_step_size=0.2)
        with torch.no_grad():
            model.correction_head.weight.normal_()
        axes, state, edges, conditioning = self._inputs(model)
        together = model(axes, state, edges, conditioning)
        first = model(axes[:1], state[:1], {h: e[:1] for h, e in edges.items()}, conditioning[:1])
        torch.testing.assert_close(together.local_target_axes[:1], first.local_target_axes)
        torch.testing.assert_close(together.step_size[:1], first.step_size)
        norms = torch.linalg.vector_norm(together.axis_correction.flatten(-2), dim=-1)
        self.assertTrue((norms < 1).all())
        self.assertEqual(together.step_size.shape, (2, 12))
        self.assertTrue(((together.step_size > 0) & (together.step_size < 0.2)).all())
        expected = axes + together.step_size[..., None, None] * together.axis_correction
        torch.testing.assert_close(together.local_target_axes, expected)

    def test_per_cell_step_varies_across_cells(self):
        """Give cells of one object different steps once the step head has nonzero weights."""
        model = IntrinsicSolverNetwork((2, 2, 2), 5, hidden_dim=16, max_step_size=0.2)
        axes, state, edges, conditioning = self._inputs(model)
        uniform = model(axes, state, edges, conditioning).step_size
        torch.testing.assert_close(uniform, torch.full((2, 8), 0.1))
        with torch.no_grad():
            model.step_head.weight.normal_()
        varied = model(axes, state, edges, conditioning).step_size
        self.assertEqual(varied.shape, (2, 8))
        self.assertTrue(torch.isfinite(varied).all())
        self.assertTrue(((varied > 0) & (varied < 0.2)).all())
        for row in varied:
            self.assertGreater((row - row[0]).abs().amax().item(), 1e-4)

    def test_solver_network_gradients(self):
        """Train through the local block and the per-cell step controller."""
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
        self.assertGreater(model.step_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.step_head.bias.grad.abs().sum().item(), 0)
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
        self.assertEqual(output.step_size.shape, (1, 4000))
        self.assertEqual(output.local_target_axes.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.local_target_axes).all())
        self.assertEqual([model.neighborhood(h)[0].shape[1] for h in model.hops], [27])


class TestContactTokenFlag(unittest.TestCase):
    """Exercise the schema-4 contact token path of the solver network."""

    GRID = (2, 2, 3)
    STATE_DIM = 7

    def setUp(self):
        """Seed the fixtures so float32 comparisons are repeatable."""
        torch.manual_seed(29)

    def _network(self, seed, **flags):
        torch.manual_seed(seed)
        return IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, hidden_dim=16, num_heads=4, **flags)

    def _pair(self):
        """Return a flag-off baseline and a flag-on variant sharing every common weight."""
        baseline = self._network(5)
        variant = self._network(5, contact_tokens=True)
        state = {name: value for name, value in baseline.state_dict().items() if name != "node_encoder.0.weight"}
        result = variant.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(
            all(key.startswith("contact_encoder.") or key == "node_encoder.0.weight" for key in result.missing_keys)
        )
        with torch.no_grad():
            width = baseline.node_encoder[0].in_features
            variant.node_encoder[0].weight[:, :width].copy_(baseline.node_encoder[0].weight)
            variant.node_encoder[0].weight[:, width:].normal_()
            for model in (baseline, variant):
                model.correction_head.weight.normal_(std=0.1)
                model.step_head.weight.normal_(std=0.1)
            variant.correction_head.weight.copy_(baseline.correction_head.weight)
            variant.step_head.weight.copy_(baseline.step_head.weight)
        return baseline, variant

    def _inputs(self, model, batch=2, slots=3):
        axes, state, edges, conditioning = TestIntrinsicSolverNetwork._inputs(model, batch=batch)
        cells = math.prod(model.cell_counts)
        tokens = torch.randn(batch, cells, slots, features.CONTACT_TOKEN_DIM)
        mask = torch.rand(batch, cells, slots) < 0.5
        mask[0, 0] = True
        mask[-1, -1] = False
        return axes, state, edges, conditioning, tokens, mask

    def test_flag_widens_node_input_and_owns_encoder(self):
        """Own a zero-initialized ContactEncoder and widen the node input by CONTACT_FEATURE_DIM."""
        model = self._network(1, contact_tokens=True)
        self.assertTrue(model.contact_tokens)
        self.assertIsInstance(model.contact_encoder, ContactEncoder)
        self.assertEqual(model.contact_encoder.output_dim, features.CONTACT_FEATURE_DIM)
        self.assertEqual(model.node_encoder[0].in_features, 9 + self.STATE_DIM + features.CONTACT_FEATURE_DIM)
        self.assertEqual(torch.count_nonzero(model.contact_encoder.pool_projection.weight).item(), 0)
        with self.assertRaises(ValueError):
            IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, hidden_dim=16, contact_tokens=1)

    def test_contact_free_scene_matches_flag_off_network_at_initialization(self):
        """Reproduce the flag-off output with matching weights for omitted and all-masked tokens."""
        baseline, variant = self._pair()
        axes, state, edges, conditioning, tokens, mask = self._inputs(baseline)
        expected = baseline(axes, state, edges, conditioning)
        self.assertGreater(torch.count_nonzero(expected.axis_correction).item(), 0)
        omitted = variant(axes, state, edges, conditioning)
        for wanted, got in zip(expected, omitted, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        masked = variant(axes, state, edges, conditioning, contact_tokens=tokens, contact_mask=torch.zeros_like(mask))
        for wanted, got in zip(expected, masked, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        # With the zero-initialized pool projection even valid tokens only add the count channel.
        counted = variant(axes, state, edges, conditioning, contact_tokens=tokens, contact_mask=mask)
        self.assertTrue(torch.isfinite(counted.local_target_axes).all())
        self.assertFalse(torch.allclose(counted.local_target_axes, expected.local_target_axes))

    def test_rejects_tokens_without_flag_and_half_supplied_inputs(self):
        """Raise ValueError for tokens on a flag-off network and for tokens or mask given alone."""
        baseline, variant = self._pair()
        axes, state, edges, conditioning, tokens, mask = self._inputs(baseline)
        with self.assertRaisesRegex(ValueError, "contact_tokens"):
            baseline(axes, state, edges, conditioning, contact_tokens=tokens, contact_mask=mask)
        with self.assertRaisesRegex(ValueError, "contact_tokens"):
            baseline(axes, state, edges, conditioning, contact_mask=mask)
        with self.assertRaises(ValueError):
            variant(axes, state, edges, conditioning, contact_tokens=tokens)
        with self.assertRaises(ValueError):
            variant(axes, state, edges, conditioning, contact_mask=mask)
        with self.assertRaises(ValueError):
            variant(axes, state, edges, conditioning, contact_tokens=tokens[:, :1], contact_mask=mask[:, :1])

    def test_trained_encoder_changes_output_and_receives_gradients(self):
        """Let valid tokens alter predictions and train every contact encoder parameter."""
        _, variant = self._pair()
        with torch.no_grad():
            variant.contact_encoder.pool_projection.weight.normal_(std=0.2)
        axes, state, edges, conditioning, tokens, mask = self._inputs(variant)
        tokens.requires_grad_()
        clean = variant(axes, state, edges, conditioning)
        output = variant(axes, state, edges, conditioning, contact_tokens=tokens, contact_mask=mask)
        self.assertFalse(torch.allclose(output.local_target_axes, clean.local_target_axes))
        (output.local_target_axes.square().mean() + output.step_size.mean()).backward()
        contact_parameters = [(n, p) for n, p in variant.named_parameters() if n.startswith("contact_encoder.")]
        self.assertGreater(len(contact_parameters), 0)
        for name, parameter in contact_parameters:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertEqual(torch.count_nonzero(tokens.grad[~mask]).item(), 0)
        self.assertGreater(tokens.grad[mask].abs().sum().item(), 0)


class TestTargetModes(unittest.TestCase):
    """Exercise the v4 seven-vector target head next to the legacy three-axis path."""

    GRID = (2, 2, 3)
    STATE_DIM = 7

    def setUp(self):
        """Seed the fixtures so float32 comparisons are repeatable."""
        torch.manual_seed(37)

    def _network(self, seed, **flags):
        torch.manual_seed(seed)
        return IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, hidden_dim=16, num_heads=4, **flags)

    @staticmethod
    def _perturb_heads(model, seed):
        with torch.no_grad():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model.correction_head.weight.normal_(std=0.1)
                model.step_head.weight.normal_(std=0.1)

    def test_seven_modes_shapes_and_widths(self):
        """Read and predict [B, N, 3, 7] with a 21-wide head whose first three columns stay the centre axes."""
        model = self._network(1, target_modes=7)
        self.assertEqual(model.target_modes, 7)
        self.assertEqual(model.node_encoder[0].in_features, 21 + self.STATE_DIM)
        self.assertEqual(model.correction_head.out_features, 21)
        axes, state, edges, conditioning = TestIntrinsicSolverNetwork._inputs(model)
        self.assertEqual(axes.shape, (2, 12, 3, 7))
        torch.testing.assert_close(axes[..., :3], torch.eye(3).expand(2, 12, 3, 3))
        self.assertEqual(torch.count_nonzero(axes[..., 3:]).item(), 0)
        self._perturb_heads(model, 2)
        output = model(axes, state, edges, conditioning)
        self.assertEqual(output.local_target_axes.shape, (2, 12, 3, 7))
        self.assertEqual(output.axis_correction.shape, (2, 12, 3, 7))
        self.assertEqual(output.step_size.shape, (2, 12))
        self.assertGreater(torch.count_nonzero(output.axis_correction[..., 3:]).item(), 0)
        norms = torch.linalg.vector_norm(output.axis_correction.flatten(-2), dim=-1)
        self.assertEqual(norms.shape, (2, 12))
        self.assertTrue((norms < 1).all())
        expected = axes + output.step_size[..., None, None] * output.axis_correction
        torch.testing.assert_close(output.local_target_axes, expected)
        with self.assertRaisesRegex(ValueError, r"3, 7"):
            model(axes[..., :3], state, edges, conditioning)
        legacy = self._network(1)
        with self.assertRaisesRegex(ValueError, r"3, 3"):
            legacy(axes, state, edges, conditioning)

    def test_rejects_invalid_target_modes(self):
        """Refuse non-positive, boolean, float, and string mode counts."""
        for bad in (0, -1, True, 3.0, "7"):
            with self.subTest(target_modes=bad):
                with self.assertRaisesRegex(ValueError, "target_modes"):
                    IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, hidden_dim=16, target_modes=bad)
        for modes in (1, 2, 7):
            model = IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, hidden_dim=16, target_modes=modes)
            self.assertEqual(model.correction_head.out_features, 3 * modes)

    def test_default_is_bit_identical_to_explicit_three_modes(self):
        """Keep the default network, its parameters, and its outputs unchanged by the new option."""
        default = self._network(5)
        explicit = self._network(5, target_modes=3)
        self.assertEqual(default.target_modes, 3)
        self.assertEqual(default.node_encoder[0].in_features, 9 + self.STATE_DIM)
        self.assertEqual(default.correction_head.out_features, 9)
        default_state, explicit_state = default.state_dict(), explicit.state_dict()
        self.assertEqual(list(default_state), list(explicit_state))
        for key, value in default_state.items():
            torch.testing.assert_close(explicit_state[key], value, rtol=0, atol=0, msg=key)
        self._perturb_heads(default, 6)
        self._perturb_heads(explicit, 6)
        axes, state, edges, conditioning = TestIntrinsicSolverNetwork._inputs(default)
        axes = axes + 0.1 * torch.randn_like(axes)
        expected = default(axes, state, edges, conditioning)
        actual = explicit(axes, state, edges, conditioning)
        self.assertGreater(torch.count_nonzero(expected.axis_correction).item(), 0)
        for wanted, got in zip(expected, actual, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)

    def test_zero_initialized_head_returns_input_for_both_mode_counts(self):
        """Return the input vectors exactly with zero corrections and a half-maximum step for m = 3 and m = 7."""
        for modes in (3, 7):
            with self.subTest(target_modes=modes):
                model = self._network(9, target_modes=modes, max_step_size=0.4)
                self.assertEqual(torch.count_nonzero(model.correction_head.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(model.correction_head.bias).item(), 0)
                axes, state, edges, conditioning = TestIntrinsicSolverNetwork._inputs(model)
                axes = axes + 0.3 * torch.randn_like(axes)
                output = model(axes, state, edges, conditioning)
                self.assertEqual(output.local_target_axes.shape, (2, 12, 3, modes))
                torch.testing.assert_close(output.local_target_axes, axes, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(output.axis_correction).item(), 0)
                torch.testing.assert_close(output.step_size, torch.full((2, 12), 0.2), rtol=0, atol=0)

    def test_v4_width_forward_backward_smoke(self):
        """Train one step at the v4 width (192 hidden, 6 heads, 96 edge) with warping modes, edge network, and contact tokens."""
        v4 = {"hidden_dim": 192, "num_heads": 6, "edge_hidden_dim": 96}
        baseline = {"hidden_dim": 128, "num_heads": 4, "edge_hidden_dim": 64}
        flags = {"target_modes": 7, "edge_network": True, "contact_tokens": True}
        counts = {}
        for label, width in (("baseline", baseline), ("v4", v4)):
            torch.manual_seed(11)
            model = IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, **width, **flags)
            counts[label] = sum(parameter.numel() for parameter in model.parameters())
            torch.manual_seed(11)
            affine = IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, **width, **{**flags, "target_modes": 3})
            affine_count = sum(parameter.numel() for parameter in affine.parameters())
            # The extra modes only widen the node encoder input and the correction head.
            self.assertEqual(counts[label] - affine_count, 24 * width["hidden_dim"] + 12)
        self.assertEqual(counts["baseline"], 428522)
        self.assertEqual(counts["v4"], 881484)

        torch.manual_seed(13)
        model = IntrinsicSolverNetwork(self.GRID, self.STATE_DIM, **v4, **flags)
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.layers[0].num_heads, 6)
        self.assertEqual(model.layers[0].head_dim, 32)
        self.assertEqual(model.layers[0].edge_dim, 96)
        self.assertEqual(model.node_encoder[0].in_features, 21 + self.STATE_DIM + features.CONTACT_FEATURE_DIM)
        self._perturb_heads(model, 14)
        with torch.no_grad():
            model.layers[0].edge_update[2].weight.normal_(std=0.1)
            model.layers[0].film.weight.normal_(std=0.01)
            model.contact_encoder.pool_projection.weight.normal_(std=0.2)
        axes, state, edges, conditioning = TestIntrinsicSolverNetwork._inputs(model)
        axes = (axes + 0.1 * torch.randn_like(axes)).requires_grad_()
        state.requires_grad_()
        conditioning.requires_grad_()
        for values in edges.values():
            values.requires_grad_()
        tokens = torch.randn(2, 12, 3, features.CONTACT_TOKEN_DIM, requires_grad=True)
        mask = torch.rand(2, 12, 3) < 0.5
        mask[0, 0] = True
        output = model(axes, state, edges, conditioning, contact_tokens=tokens, contact_mask=mask)
        self.assertEqual(output.local_target_axes.shape, (2, 12, 3, 7))
        self.assertEqual(output.axis_correction.shape, (2, 12, 3, 7))
        self.assertEqual(output.step_size.shape, (2, 12))
        for value in output:
            self.assertTrue(torch.isfinite(value).all())
        loss = (output.local_target_axes - (axes.detach() + 0.1)).square().mean() + output.step_size.mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.correction_head.weight.grad.abs().sum().item(), 0)
        for value in (axes, state, conditioning, tokens, *edges.values()):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(tokens.grad[~mask]).item(), 0)


if __name__ == "__main__":
    unittest.main()
