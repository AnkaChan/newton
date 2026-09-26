# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental PyTorch layers for predicting intrinsic cell deformation targets.

This API may change. Geometry preprocessing, physical feature normalization,
frame extraction, fusion, and time integration belong to the caller. Importing
this optional experimental module requires PyTorch; importing Newton does not.
"""

from collections.abc import Mapping
from math import isfinite, prod
from typing import NamedTuple

import torch  # noqa: TID253 -- Explicit opt-in module defining PyTorch nn.Module classes.
from torch import Tensor, nn  # noqa: TID253

from .network_geometry import build_grid_neighborhood

__all__ = ["IntrinsicSolverNetwork", "IntrinsicSolverOutput", "IntrinsicTransformerLayer"]


def _integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _expand_conditioning(conditioning: Tensor, batch: int, cells: int, channels: int) -> Tensor:
    if conditioning.shape == (batch, channels):
        return conditioning[:, None, :].expand(batch, cells, channels)
    if conditioning.shape != (batch, cells, channels):
        raise ValueError("conditioning must have shape [batch, channels] or [batch, cells, channels]")
    return conditioning


class IntrinsicTransformerLayer(nn.Module):
    """Apply experimental masked graph attention, FiLM, and a residual MLP.

    Each directed edge supplies a learned per-head score bias and a learned value
    addition. There is no fixed geometric attention prior. Ordinary hidden feature
    channels are not spatial vectors; the caller supplies geometric edge features
    already expressed in the receiving cell's frame.

    Layer normalization precedes each residual branch. FiLM starts as identity.
    Query chunking reduces temporary gathers while retaining the complete neighbor
    softmax for every query; it does not bound total training activation memory.

    Args:
        hidden_dim: Number of cell feature channels.
        edge_dim: Number of encoded edge feature channels.
        num_heads: Attention heads; must divide hidden_dim.
        conditioning_dim: Scalar-conditioning channels; zero disables FiLM.
        ffn_multiplier: Expansion factor for the SiLU feed-forward branch.
        query_chunk_size: Maximum queried cells per attention chunk.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        *,
        num_heads: int = 4,
        conditioning_dim: int = 0,
        ffn_multiplier: int = 4,
        query_chunk_size: int = 128,
    ):
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("edge_dim", edge_dim),
            ("num_heads", num_heads),
            ("ffn_multiplier", ffn_multiplier),
            ("query_chunk_size", query_chunk_size),
        ):
            _integer(value, name)
        _integer(conditioning_dim, "conditioning_dim", minimum=0)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.conditioning_dim = conditioning_dim
        self.query_chunk_size = query_chunk_size
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.edge_val = nn.Linear(edge_dim, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.SiLU(),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
        )
        self.film = nn.Linear(conditioning_dim, 4 * hidden_dim) if conditioning_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(
        self,
        features: Tensor,
        edge_features: Tensor,
        neighbor_indices: Tensor,
        neighbor_mask: Tensor,
        *,
        conditioning: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Update cell features without attending to nonexistent edges.

        Args:
            features: Cell hidden features, shape [B, N, hidden_dim].
            edge_features: Directed features, shape [B, N, S, edge_dim].
            neighbor_indices: Common topology, shape [N, S], dtype int64.
                Invalid slots may contain arbitrary sentinel indices.
            neighbor_mask: Valid-edge flags, shape [N, S], dtype bool.
            conditioning: Prepared dimensionless scalars, shape [B, N, C]
                or [B, C]. Required when conditioning_dim is nonzero.
            return_attention: Also return per-head weights for inspection.

        Returns:
            Updated features [B, N, hidden_dim], optionally paired with
            attention [B, N, num_heads, S]. Invalid weights are exactly zero.
            A fully masked row has zero attention; its residual/MLP remains
            defined. Canonical neighborhoods always include an active self.
        """
        if features.ndim != 3 or features.shape[-1] != self.hidden_dim:
            raise ValueError("features must have shape [batch, cells, hidden_dim]")
        batch, cells, _ = features.shape
        if not batch or not cells:
            raise ValueError("features must contain at least one object and cell")
        if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != cells or not neighbor_indices.shape[1]:
            raise ValueError("neighbor_indices must have shape [cells, nonempty_slots]")
        if neighbor_indices.dtype != torch.int64:
            raise ValueError("neighbor_indices must have dtype int64")
        if neighbor_mask.shape != neighbor_indices.shape or neighbor_mask.dtype != torch.bool:
            raise ValueError("neighbor_mask must be boolean with the same shape as neighbor_indices")
        slots = neighbor_indices.shape[1]
        if edge_features.shape != (batch, cells, slots, self.edge_dim):
            raise ValueError("edge_features must have shape [batch, cells, slots, edge_dim]")

        modulation = None
        if self.film is not None:
            if conditioning is None:
                raise ValueError("conditioning is required when FiLM is enabled")
            conditioning = _expand_conditioning(conditioning, batch, cells, self.conditioning_dim)
            modulation = self.film(conditioning).chunk(4, dim=-1)
        elif conditioning is not None:
            raise ValueError("conditioning was supplied but this layer has no FiLM channels")

        normalized = self.attention_norm(features)
        if modulation is not None:
            normalized = normalized * (1 + modulation[0]) + modulation[1]
        query, key, value = self.qkv(normalized).reshape(batch, cells, 3, self.num_heads, self.head_dim).unbind(2)
        safe_indices = torch.where(neighbor_mask, neighbor_indices, 0)
        chunks, attention_chunks = [], []
        for start in range(0, cells, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, cells)
            valid = neighbor_mask[None, start:stop, :, None]
            indices = safe_indices[start:stop]
            # Remove poisoned padding before learned projections, not just after softmax.
            edges = torch.where(valid, edge_features[:, start:stop], 0)
            scores = (query[:, start:stop, None] * key[:, indices]).sum(-1) * (self.head_dim**-0.5)
            scores = (scores + self.edge_bias(edges)).masked_fill(~valid, -torch.inf)
            # All-masked rows must not send NaNs through softmax or its backward.
            has_neighbor = valid.any(dim=2, keepdim=True)
            scores = torch.where(has_neighbor, scores, 0)
            weights = scores.softmax(dim=2).masked_fill(~valid, 0)
            edge_values = self.edge_val(edges).reshape(batch, stop - start, slots, self.num_heads, self.head_dim)
            message = (weights[..., None] * (value[:, indices] + edge_values)).sum(dim=2)
            chunks.append(message.reshape(batch, stop - start, self.hidden_dim))
            if return_attention:
                attention_chunks.append(weights.permute(0, 1, 3, 2))
        attended = features + self.out_projection(torch.cat(chunks, dim=1))
        normalized = self.ffn_norm(attended)
        if modulation is not None:
            normalized = normalized * (1 + modulation[2]) + modulation[3]
        output = attended + self.ffn(normalized)
        if return_attention:
            return output, torch.cat(attention_chunks, dim=1)
        return output


class IntrinsicSolverOutput(NamedTuple):
    """Experimental local deformation targets; these are not integrated positions."""

    local_target_axes: Tensor
    """Dimensionless target matrices with axes as columns, shape [B, N, 3, 3]."""
    axis_correction: Tensor
    """Dimensionless corrections with joint nine-value norm below one, shape [B, N, 3, 3]."""
    step_size: Tensor
    """Dimensionless per-cell step in (0, max_step_size), shape [B, N]."""


class IntrinsicSolverNetwork(nn.Module):
    """Stack experimental graph-transformer blocks into a local-target predictor.

    The baseline is one local block with corner-connected hop distance
    (1,), width 128, and four heads. Topology is a full canonical cuboid;
    each batch entry has the same topology and is one separate object. Indices
    and masks are registered buffers, so state_dict and device moves retain
    them. Do not use this topology to connect separate or empty material cells.

    Scalar/node feature normalization is caller-owned. Current axes are always
    encoded and need not be repeated in state_features. A shared edge MLP is
    evaluated once per distinct hop, with separate bias/value projections in
    each layer. Shared scalar conditioning drives each layer's FiLM.

    The correction head starts at zero, so initial local targets equal current
    axes. This does not guarantee a no-op after global fusion. The step head is
    applied per cell: every cell receives its own bounded sigmoid step in
    (0, max_step_size), equal to 0.5 * max_step_size at the zero initialization.
    This per-cell step controller is an experimental configurable choice, not a
    guarantee of physical stability or energy descent.

    Args:
        cell_counts: Positive cell counts along material x, y, z; z varies fastest.
        state_feature_dim: Additional prepared per-cell features, excluding axes.
        conditioning_dim: Prepared material/size/timestep/viscosity scalar
            channels; the revised schema supplies six (features.CONDITIONING_DIM).
        hidden_dim: Cell-token width.
        num_heads: Attention heads per block.
        edge_input_dim: Raw directed edge channels; network_geometry supplies 24.
        edge_hidden_dim: Shared encoded edge width.
        hops: Exact graph-hop distance used by each block, plus self. Defaults
            to one radius-1 block (27 masked slots); pass an explicit sequence
            to restore a saved architecture with more blocks.
        max_step_size: Upper bound on the dimensionless per-cell step.
        query_chunk_size: Query cells per attention chunk; all neighbor slots remain visible.
    """

    def __init__(
        self,
        cell_counts: tuple[int, int, int],
        state_feature_dim: int,
        *,
        conditioning_dim: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        edge_input_dim: int = 24,
        edge_hidden_dim: int = 64,
        hops: tuple[int, ...] = (1,),
        max_step_size: float = 1.0,
        query_chunk_size: int = 128,
    ):
        super().__init__()
        _integer(state_feature_dim, "state_feature_dim", minimum=0)
        for name, value in (
            ("conditioning_dim", conditioning_dim),
            ("hidden_dim", hidden_dim),
            ("edge_input_dim", edge_input_dim),
            ("edge_hidden_dim", edge_hidden_dim),
        ):
            _integer(value, name)
        self.cell_counts = tuple(cell_counts)
        self.hops = tuple(hops)
        if not self.hops:
            raise ValueError("hops must contain at least one attention block")
        if not isfinite(max_step_size) or max_step_size <= 0:
            raise ValueError("max_step_size must be finite and positive")
        for hop in dict.fromkeys(self.hops):
            indices, mask = build_grid_neighborhood(self.cell_counts, hop)
            self.register_buffer(f"neighbor_indices_{hop}", indices)
            self.register_buffer(f"neighbor_mask_{hop}", mask)
        self.state_feature_dim = state_feature_dim
        self.conditioning_dim = conditioning_dim
        self.edge_input_dim = edge_input_dim
        self.max_step_size = float(max_step_size)
        self.node_encoder = nn.Sequential(
            nn.Linear(9 + state_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.SiLU(), nn.Linear(edge_hidden_dim, edge_hidden_dim)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList(
            IntrinsicTransformerLayer(
                hidden_dim,
                edge_hidden_dim,
                num_heads=num_heads,
                conditioning_dim=hidden_dim,
                query_chunk_size=query_chunk_size,
            )
            for _ in self.hops
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.correction_head = nn.Linear(hidden_dim, 9)
        self.step_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)

    def neighborhood(self, hop: int) -> tuple[Tensor, Tensor]:
        """Return registered indices and masks [N, S] for a configured hop."""
        if hop not in self.hops:
            raise ValueError(f"hop {hop} is not configured")
        return getattr(self, f"neighbor_indices_{hop}"), getattr(self, f"neighbor_mask_{hop}")

    def forward(
        self,
        local_axes: Tensor,
        state_features: Tensor,
        edge_features: Mapping[int, Tensor],
        conditioning: Tensor,
    ) -> IntrinsicSolverOutput:
        """Predict local targets in the same fixed frames as the input axes.

        Args:
            local_axes: Current dimensionless axes as columns, shape [B, N, 3, 3].
            state_features: Additional normalized inputs, shape [B, N, state_feature_dim].
                Examples include inertial offsets/rest length, boundary flags,
                and normalized optimizer history; the layout is caller-defined.
            edge_features: Raw directed features [B, N, S_hop, edge_input_dim],
                indexed by hop. Use the matching neighborhood() indices/masks.
            conditioning: Prepared scalar channels [B, N, conditioning_dim]
                or [B, conditioning_dim]. Normalize/log physical magnitudes
                before this call; the network does not infer physical units.

        Returns:
            Local targets, bounded corrections, and a per-cell step [B, N] with
            target = local_axes + step_size[..., None, None] * axis_correction.
            Frame extraction and global reconstruction are external operations.
        """
        cells = prod(self.cell_counts)
        if local_axes.ndim != 4 or local_axes.shape[1:] != (cells, 3, 3) or not local_axes.shape[0]:
            raise ValueError("local_axes must have shape [nonempty_batch, cell_count, 3, 3]")
        batch = local_axes.shape[0]
        if state_features.shape != (batch, cells, self.state_feature_dim):
            raise ValueError("state_features must match the batch, cells, and configured feature count")
        conditioning = _expand_conditioning(conditioning, batch, cells, self.conditioning_dim)
        condition = self.condition_encoder(conditioning)
        features = self.node_encoder(torch.cat((local_axes.flatten(-2), state_features), dim=-1))
        encoded_edges = {}
        for hop in dict.fromkeys(self.hops):
            indices, mask = self.neighborhood(hop)
            if hop not in edge_features:
                raise ValueError(f"edge_features is missing hop {hop}")
            edges = edge_features[hop]
            if edges.shape != (batch, cells, indices.shape[1], self.edge_input_dim):
                raise ValueError(f"edge_features[{hop}] has the wrong shape")
            edges = torch.where(mask[None, :, :, None], edges, 0)
            encoded_edges[hop] = self.edge_encoder(edges)
        for hop, layer in zip(self.hops, self.layers, strict=True):
            indices, mask = self.neighborhood(hop)
            features = layer(features, encoded_edges[hop], indices, mask, conditioning=condition)
        features = self.output_norm(features)
        raw = self.correction_head(features)
        correction = (raw / torch.sqrt(1 + raw.square().sum(dim=-1, keepdim=True))).reshape(batch, cells, 3, 3)
        step_size = self.max_step_size * self.step_head(features).squeeze(-1).sigmoid()
        target = local_axes + step_size[..., None, None] * correction
        return IntrinsicSolverOutput(target, correction, step_size)
