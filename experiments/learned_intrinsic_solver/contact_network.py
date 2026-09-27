# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental per-cell encoder for variable-count contact tokens.

Each material cell owns up to ``M`` contact candidate tokens (schema version 4
in :mod:`features`, ``CONTACT_TOKEN_DIM`` channels each, all expressed in the
owning cell's frame). :class:`ContactEncoder` maps the padded, masked token
set of every cell to a fixed-width summary that the caller appends to the
cell's state features. The summary is permutation invariant over tokens,
ignores padding slots entirely, and is exactly zero for cells with no valid
token, so a contact-free scene reproduces the schema-3 network input at
initialization. This API may change; importing this optional experimental
module requires PyTorch, importing Newton does not.
"""

import torch  # noqa: TID253 -- Explicit opt-in module defining PyTorch nn.Module classes.
from torch import Tensor, nn  # noqa: TID253

__all__ = ["CONTACT_POOL_DIM", "CONTACT_TOKEN_DIM", "ContactEncoder"]

CONTACT_TOKEN_DIM = 19
"""Channels per contact token; see the schema-4 layout in the contact design note."""

CONTACT_POOL_DIM = 16
"""Learned pooled channels per cell, excluding the trailing token-count channel."""


def _integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class ContactEncoder(nn.Module):
    """Summarize the masked contact tokens of every cell into a fixed vector.

    The block is a token MLP, one pre-LayerNorm masked self-attention block over
    the tokens of the same cell (tokens never attend across cells), and masked
    mean plus masked max pooling followed by a zero-initialized projection. The
    output has ``pool_dim + 1`` channels: the projected pooling result and the
    fraction of valid tokens ``count / M``.

    Padding slots are removed before every learned projection, excluded from the
    attention softmax, and excluded from both pooling reductions, so arbitrary
    values (including NaN or inf) in masked token rows do not influence the
    output or the gradients. Cells without any valid token return exact zeros in
    every channel regardless of the learned bias.

    Only cells with at least one valid token pass through the block (gather,
    encode, scatter), so activations saved for backpropagation scale with the
    number of contacting cells rather than with ``B * C * M``. A batch without
    any valid token still encodes one empty cell so every parameter stays in the
    autograd graph, as distributed data parallel training requires.

    Args:
        token_dim: Channels per token; the schema supplies CONTACT_TOKEN_DIM.
        width: Token hidden width inside the attention block.
        num_heads: Attention heads; must divide width.
        ffn_multiplier: Expansion factor for the SiLU feed-forward branch.
        pool_dim: Learned output channels before the appended count channel.
    """

    def __init__(
        self,
        token_dim: int = CONTACT_TOKEN_DIM,
        width: int = 64,
        num_heads: int = 2,
        ffn_multiplier: int = 4,
        pool_dim: int = CONTACT_POOL_DIM,
    ):
        super().__init__()
        for name, value in (
            ("token_dim", token_dim),
            ("width", width),
            ("num_heads", num_heads),
            ("ffn_multiplier", ffn_multiplier),
            ("pool_dim", pool_dim),
        ):
            _integer(value, name)
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.token_dim = token_dim
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.pool_dim = pool_dim
        self.output_dim = pool_dim + 1
        self.token_encoder = nn.Sequential(nn.Linear(token_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out_projection = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_multiplier * width),
            nn.SiLU(),
            nn.Linear(ffn_multiplier * width, width),
        )
        # Zero start: contact tokens are invisible to the downstream network until trained.
        self.pool_projection = nn.Linear(2 * width, pool_dim)
        nn.init.zeros_(self.pool_projection.weight)
        nn.init.zeros_(self.pool_projection.bias)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """Encode every cell's token set independently.

        Args:
            tokens: Contact tokens in the owning cell frame, shape [B, C, M, token_dim].
                Masked rows may hold arbitrary values, including NaN.
            mask: Valid-token flags, shape [B, C, M], dtype bool.

        Returns:
            Per-cell summaries, shape [B, C, pool_dim + 1]. The first pool_dim
            channels are the projected pooled features; the last channel is the
            number of valid tokens divided by M. Cells without valid tokens are
            exactly zero in every channel.
        """
        if tokens.ndim != 4 or tokens.shape[-1] != self.token_dim:
            raise ValueError("tokens must have shape [batch, cells, slots, token_dim]")
        batch, cells, slots, _ = tokens.shape
        if not batch or not cells or not slots:
            raise ValueError("tokens must contain at least one object, cell, and slot")
        if mask.shape != (batch, cells, slots) or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [batch, cells, slots]")

        flat_tokens = tokens.reshape(batch * cells, slots, self.token_dim)
        flat_mask = mask.reshape(batch * cells, slots)
        active = flat_mask.any(dim=-1).nonzero().squeeze(-1)
        if not active.numel():
            # Keep the parameters in the graph on contact-free batches; the empty cell encodes to zeros.
            active = active.new_zeros((1,))
        encoded = self._encode_dense(flat_tokens[active][None], flat_mask[active][None])[0]
        output = encoded.new_zeros((batch * cells, self.output_dim)).index_copy(0, active, encoded)
        return output.reshape(batch, cells, self.output_dim)

    def _encode_dense(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """Encode every cell of ``tokens`` [B, C, M, token_dim] with ``mask`` [B, C, M] into [B, C, output_dim]."""
        batch, cells, slots, _ = tokens.shape
        valid = mask[..., None]
        has_token = mask.any(dim=-1, keepdim=True)
        # Remove poisoned padding before learned projections, not just after softmax.
        hidden = self.token_encoder(torch.where(valid, tokens, 0))
        normalized = self.attention_norm(hidden)
        query, key, value = (
            self.qkv(normalized)
            .reshape(batch, cells, slots, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 1, 4, 2, 5)
            .unbind(0)
        )
        scores = (query @ key.transpose(-1, -2)) * (self.head_dim**-0.5)
        key_valid = mask[:, :, None, None, :]
        scores = scores.masked_fill(~key_valid, -torch.inf)
        # Cells without tokens must not send NaNs through softmax or its backward.
        scores = torch.where(has_token[:, :, None, :, None], scores, 0)
        weights = scores.softmax(dim=-1).masked_fill(~key_valid, 0)
        message = (weights @ value).permute(0, 1, 3, 2, 4).reshape(batch, cells, slots, self.width)
        attended = hidden + self.out_projection(message)
        encoded = attended + self.ffn(self.ffn_norm(attended))

        encoded = torch.where(valid, encoded, 0)
        count = mask.sum(dim=-1, dtype=encoded.dtype)
        mean_pool = encoded.sum(dim=2) / count.clamp(min=1)[..., None]
        max_pool = encoded.masked_fill(~valid, -torch.inf).amax(dim=2)
        max_pool = torch.where(has_token, max_pool, 0)
        pooled = self.pool_projection(torch.cat([mean_pool, max_pool], dim=-1))
        # The bias would otherwise leak into contact-free cells after training.
        pooled = torch.where(has_token, pooled, 0)
        return torch.cat([pooled, (count / slots)[..., None]], dim=-1)
