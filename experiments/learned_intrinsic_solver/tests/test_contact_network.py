# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the masked per-cell contact token encoder on CPU."""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("PyTorch is an optional dependency") from error

from experiments.learned_intrinsic_solver.contact_network import (
    CONTACT_POOL_DIM,
    CONTACT_TOKEN_DIM,
    ContactEncoder,
)


def _perturb(encoder: ContactEncoder, std: float = 0.1) -> None:
    """Move the zero-initialized projection so upstream parameters receive gradient."""
    with torch.no_grad():
        encoder.pool_projection.weight.normal_(std=std)
        encoder.pool_projection.bias.normal_(std=std)


class TestContactEncoder(unittest.TestCase):
    def setUp(self):
        """Seed the fixtures so float32 comparisons are repeatable."""
        torch.manual_seed(31)

    @staticmethod
    def _inputs(batch=2, cells=3, slots=5, token_dim=CONTACT_TOKEN_DIM):
        tokens = torch.randn(batch, cells, slots, token_dim)
        mask = torch.rand(batch, cells, slots) < 0.6
        mask[0, 0] = False  # one cell with no contact
        mask[-1, -1] = True  # one full cell
        mask[0, -1, 0] = True  # guarantee at least one token elsewhere
        return tokens, mask

    def test_output_shape_and_constants(self):
        """Return [B, C, pool_dim + 1] with the contract widths."""
        self.assertEqual(CONTACT_TOKEN_DIM, 19)
        self.assertEqual(CONTACT_POOL_DIM, 16)
        encoder = ContactEncoder()
        self.assertEqual(encoder.output_dim, 17)
        tokens, mask = self._inputs(batch=2, cells=3, slots=5)
        output = encoder(tokens, mask)
        self.assertEqual(tuple(output.shape), (2, 3, 17))
        self.assertTrue(torch.isfinite(output).all())

    def test_zero_initialization_and_count_channel(self):
        """Emit exact zeros in the pooled channels and count / M in the last channel at start."""
        encoder = ContactEncoder()
        tokens, mask = self._inputs(batch=2, cells=4, slots=6)
        output = encoder(tokens, mask)
        self.assertEqual(torch.count_nonzero(output[..., :CONTACT_POOL_DIM]).item(), 0)
        torch.testing.assert_close(output[..., -1], mask.sum(-1).float() / 6)

    def test_empty_cells_are_zero_after_training(self):
        """Keep every channel exactly zero for cells without valid tokens even with a nonzero bias."""
        encoder = ContactEncoder()
        _perturb(encoder)
        with torch.no_grad():
            encoder.pool_projection.bias.fill_(1.5)
        tokens, mask = self._inputs(batch=2, cells=3, slots=4)
        output = encoder(tokens, mask)
        empty = ~mask.any(-1)
        self.assertTrue(empty.any())
        self.assertEqual(torch.count_nonzero(output[empty]).item(), 0)
        self.assertGreater(output[~empty][..., :CONTACT_POOL_DIM].abs().sum().item(), 0)

    def test_masked_rows_do_not_affect_output_or_gradients(self):
        """Ignore NaN and inf in padding rows for the forward value and for gradients."""
        encoder = ContactEncoder()
        _perturb(encoder)
        tokens, mask = self._inputs(batch=2, cells=3, slots=5)
        clean = tokens.clone().requires_grad_()
        poisoned = tokens.clone()
        padding = ~mask.unsqueeze(-1).expand_as(poisoned)
        poisoned[padding] = float("nan")
        poisoned[0, 1, ~mask[0, 1], 0] = float("inf")
        poisoned.requires_grad_()
        expected = encoder(clean, mask)
        actual = encoder(poisoned, mask)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(poisoned.grad).all())
        torch.testing.assert_close(poisoned.grad, clean.grad)
        self.assertEqual(torch.count_nonzero(poisoned.grad[padding]).item(), 0)
        for parameter in encoder.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_permutation_invariance_within_cell(self):
        """Return the same summary when the tokens of a cell are reordered together with the mask."""
        encoder = ContactEncoder()
        _perturb(encoder)
        tokens, mask = self._inputs(batch=2, cells=3, slots=6)
        expected = encoder(tokens, mask)
        order = torch.randperm(6)
        actual = encoder(tokens[:, :, order], mask[:, :, order])
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        # Different permutations per cell also leave the summary unchanged.
        orders = torch.stack([torch.randperm(6) for _ in range(6)]).reshape(2, 3, 6)
        gathered_tokens = torch.gather(tokens, 2, orders[..., None].expand_as(tokens))
        gathered_mask = torch.gather(mask, 2, orders)
        torch.testing.assert_close(encoder(gathered_tokens, gathered_mask), expected, rtol=2e-6, atol=2e-6)

    def test_gradients_reach_all_parameters(self):
        """Propagate derivatives to every parameter once the output projection is nonzero."""
        encoder = ContactEncoder()
        _perturb(encoder)
        tokens, mask = self._inputs(batch=2, cells=3, slots=5)
        tokens.requires_grad_()
        output = encoder(tokens, mask)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad.abs().sum().item(), 0)
        for name, parameter in encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_padding_slots_do_not_change_pooling(self):
        """Match the summary of the unpadded token set when extra padding slots hold large values."""
        encoder = ContactEncoder()
        _perturb(encoder)
        batch, cells, valid_slots, padding_slots = 2, 3, 3, 4
        tokens = torch.randn(batch, cells, valid_slots, CONTACT_TOKEN_DIM)
        mask = torch.ones(batch, cells, valid_slots, dtype=torch.bool)
        compact = encoder(tokens, mask)
        padded_tokens = torch.cat([tokens, torch.full((batch, cells, padding_slots, CONTACT_TOKEN_DIM), 1e6)], dim=2)
        padded_mask = torch.cat([mask, torch.zeros(batch, cells, padding_slots, dtype=torch.bool)], dim=2)
        padded = encoder(padded_tokens, padded_mask)
        torch.testing.assert_close(padded[..., :CONTACT_POOL_DIM], compact[..., :CONTACT_POOL_DIM])
        torch.testing.assert_close(
            padded[..., -1], torch.full((batch, cells), valid_slots / (valid_slots + padding_slots))
        )
        # Interleaving the padding among the valid slots is equally harmless.
        order = torch.tensor([3, 0, 4, 1, 5, 2, 6])
        interleaved = encoder(padded_tokens[:, :, order], padded_mask[:, :, order])
        torch.testing.assert_close(interleaved, padded, rtol=2e-6, atol=2e-6)

    def test_max_pool_reflects_valid_tokens_only(self):
        """Change the summary when a padding slot becomes valid, proving the pools see the mask."""
        encoder = ContactEncoder()
        _perturb(encoder)
        tokens = torch.randn(1, 1, 3, CONTACT_TOKEN_DIM)
        tokens[0, 0, 2] = 8.0  # a large token that would dominate an unmasked max pool
        mask = torch.tensor([[[True, True, False]]])
        masked = encoder(tokens, mask)
        unmasked = encoder(tokens, torch.ones_like(mask))
        self.assertFalse(torch.allclose(masked[..., :CONTACT_POOL_DIM], unmasked[..., :CONTACT_POOL_DIM]))

    def test_sparse_batches_save_activations_for_contacting_cells_only(self):
        """Scale saved-for-backward memory with the cells that own a token, not with B * C * M."""
        encoder = ContactEncoder()
        _perturb(encoder)
        tokens = torch.randn(1, 400, 24, CONTACT_TOKEN_DIM)
        sparse_mask = torch.zeros(1, 400, 24, dtype=torch.bool)
        sparse_mask[:, ::10, 0] = True  # 10% of the cells hold one token

        def saved_bytes(mask):
            total = [0]

            def pack(tensor):
                total[0] += tensor.numel() * tensor.element_size()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                encoder(tokens, mask).sum().backward()
            return total[0]

        sparse = saved_bytes(sparse_mask)
        dense = saved_bytes(torch.ones_like(sparse_mask))
        self.assertLess(sparse, 0.25 * dense, (sparse, dense))
        # Compaction leaves the summary unchanged against the dense evaluation of every cell.
        torch.testing.assert_close(
            encoder(tokens, sparse_mask), encoder._encode_dense(tokens, sparse_mask), rtol=1e-5, atol=1e-6
        )

    def test_batch_without_any_token_is_zero_and_keeps_parameters_in_the_graph(self):
        """Return exact zeros for an all-masked batch while every parameter still receives a gradient."""
        encoder = ContactEncoder()
        _perturb(encoder)
        tokens = torch.randn(2, 3, 4, CONTACT_TOKEN_DIM)
        output = encoder(tokens, torch.zeros(2, 3, 4, dtype=torch.bool))
        self.assertEqual(tuple(output.shape), (2, 3, 17))
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        output.sum().backward()
        for name, parameter in encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_constructor_validation(self):
        """Reject invalid widths, head counts, and input shapes with ValueError."""
        with self.assertRaises(ValueError):
            ContactEncoder(width=6, num_heads=4)
        with self.assertRaises(ValueError):
            ContactEncoder(token_dim=0)
        with self.assertRaises(ValueError):
            ContactEncoder(pool_dim=True)
        encoder = ContactEncoder()
        tokens, mask = self._inputs(batch=1, cells=2, slots=3)
        with self.assertRaises(ValueError):
            encoder(tokens[..., :5], mask)
        with self.assertRaises(ValueError):
            encoder(tokens, mask.float())
        with self.assertRaises(ValueError):
            encoder(tokens, mask[:, :1])
        with self.assertRaises(ValueError):
            encoder(tokens[:, :, :0], mask[:, :, :0])


if __name__ == "__main__":
    unittest.main()
