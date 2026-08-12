# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Multiresolution graph transformer for principal-stretch dynamics.

The learned state is the Hencky tensor

``H = log(U) = 0.5 log(F.T @ F)``,

whose eigenvalues are the principal log-stretches.  The implementation never
regresses eigenvectors: their signs, ordering, and bases inside repeated
eigenspaces are not continuous.  Instead, the complete symmetric tensor is
carried in the material frame and the output is

``U_target = exp(H_current + delta_H)``.

This is positive definite by construction and invariant to an active world
rigid transform.  World vectors (gravity, load, velocity, and current edge
offsets) are pulled into each node's polar frame before entering ordinary
MLPs.  Under ``x' = Q x + t`` and ``f' = Q f``, the polar frame becomes
``R' = Q R``, so every learned feature and the material-frame target stretch
remain unchanged.  The downstream local-global decoder then makes the full
position solver SE(3)-equivariant.

The topology path follows HOOD's useful multiresolution principle.  Tets are
coarsened only along shared faces, each quotient graph runs content-dependent
sparse attention, and a top-down U-Net pass lets fine nodes cross-attend to
their parent and topology-adjacent parent tokens.  No Euclidean proximity
edge, absolute node embedding, or per-mesh learned parameter is used.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
import torch.nn as nn

from .hierarchy import Hierarchy
from .model import sym_to_vec, vec_to_sym
from .polar import polar_rotation
from .spd_log import spd_floor, sym_exp, sym_log
from .torch_solver import SolverState

NODE_FEATURE_DIM = 36
EDGE_FEATURE_DIM = 20


@dataclasses.dataclass(frozen=True)
class GraphTransformerConfig:
    """Hyperparameters saved with graph-transformer checkpoints."""

    hidden_dim: int = 64
    num_heads: int = 4
    n_levels: int = 5
    cluster_size: int = 8
    dropout: float = 0.0
    max_hencky_update: float = 0.35
    dt: float = 1.0 / 60.0

    def __post_init__(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.n_levels < 0:
            raise ValueError("n_levels must be non-negative")
        if self.cluster_size < 2:
            raise ValueError("cluster_size must be at least 2")
        if self.max_hencky_update <= 0.0:
            raise ValueError("max_hencky_update must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _batch_pool_sum(values: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    """Sum ``(B, N, ...)`` child values into their parent nodes."""
    n_parent = int(assign.max().item()) + 1
    shape = list(values.shape)
    shape[1] = n_parent
    return values.new_zeros(shape).index_add(1, assign, values)


def _batch_pool_mean(values: torch.Tensor, assign: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """Rest-volume-weighted mean of ``(B, N, ...)`` child values."""
    weight_shape = (1, volume.shape[0], *([1] * (values.dim() - 2)))
    numerator = _batch_pool_sum(values * volume.reshape(weight_shape), assign)
    denominator = volume.new_zeros(int(assign.max().item()) + 1).index_add(0, assign, volume)
    denom_shape = (1, denominator.shape[0], *([1] * (values.dim() - 2)))
    return numerator / denominator.reshape(denom_shape)


def _spectral_invariants(H: torch.Tensor) -> torch.Tensor:
    """Smooth symmetric polynomials of the principal Hencky strains."""
    trace = H.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(3, dtype=H.dtype, device=H.device)
    dev = H - trace[..., None, None] * eye / 3.0
    dev_norm = torch.sqrt((dev * dev).sum(dim=(-2, -1)).clamp(min=0.0) + 1.0e-16)
    dev_det = torch.linalg.det(dev)
    return torch.stack([trace, dev_norm, dev_det], dim=-1)


def _radially_bound_symmetric(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    """Bound a symmetric generator by its Frobenius norm, without axis bias."""
    matrix = vec_to_sym(raw)
    norm = torch.sqrt((matrix * matrix).sum(dim=(-2, -1), keepdim=True) + 1.0e-16)
    scale = maximum / torch.sqrt(maximum * maximum + norm * norm)
    return matrix * scale


class RelationAttentionBlock(nn.Module):
    """Pre-norm sparse graph attention with relation-biased keys and values."""

    def __init__(self, hidden_dim: int, num_heads: int, edge_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_encoder = _mlp(edge_dim, hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(hidden_dim, num_heads, bias=False)
        self.edge_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def _attention(
        self,
        hidden: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attention update and rows, including the self edge."""
        batch_size, n_nodes, _ = hidden.shape
        adjacency = adjacency.long()
        valid = adjacency >= 0
        index = adjacency.clamp(min=0)
        n_slots = adjacency.shape[1]

        normalized = self.norm_attention(hidden)
        query = self.query(normalized).reshape(batch_size, n_nodes, self.num_heads, self.head_dim)
        key = self.key(normalized).reshape(batch_size, n_nodes, self.num_heads, self.head_dim)
        value = self.value(normalized).reshape(batch_size, n_nodes, self.num_heads, self.head_dim)

        neighbor_key = key[:, index]
        neighbor_value = value[:, index]
        key = torch.cat([key[:, :, None], neighbor_key], dim=2)
        value = torch.cat([value[:, :, None], neighbor_value], dim=2)

        self_edge = edge_features.new_zeros(batch_size, n_nodes, 1, edge_features.shape[-1])
        relation = torch.cat([self_edge, edge_features], dim=2)
        relation_hidden = self.edge_encoder(relation)
        bias = self.edge_bias(relation_hidden)
        value = value + self.edge_value(relation_hidden).reshape(
            batch_size, n_nodes, n_slots + 1, self.num_heads, self.head_dim
        )

        logits = (query[:, :, None] * key).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + bias

        # Shared-face/quotient area is a physically meaningful prior, but it
        # is not the attention itself.  Content-dependent q/k and relation
        # bias can override it.  Normalize per row so the self prior is zero.
        weight = edge_weight.to(dtype=hidden.dtype, device=hidden.device)
        count = valid.sum(dim=1).clamp(min=1).to(hidden.dtype)
        row_mean = (weight * valid).sum(dim=1) / count
        log_prior = torch.log((weight / row_mean[:, None].clamp(min=1.0e-12)).clamp(min=1.0e-12))
        log_prior = torch.where(valid, log_prior, torch.zeros_like(log_prior))
        logits = logits + torch.cat([log_prior.new_zeros(n_nodes, 1), log_prior], dim=1)[None, :, :, None]

        mask = torch.cat([torch.ones(n_nodes, 1, dtype=torch.bool, device=valid.device), valid], dim=1)
        logits = logits.masked_fill(~mask[None, :, :, None], torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=2)
        update = (self.dropout(attention)[..., None] * value).sum(dim=2).reshape(batch_size, n_nodes, -1)
        return self.output(update), attention

    def attention_weights(
        self,
        hidden: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Expose attention rows for diagnostics and invariance tests."""
        return self._attention(hidden, edge_features, adjacency, edge_weight)[1]

    def forward(
        self,
        hidden: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        update, _attention = self._attention(hidden, edge_features, adjacency, edge_weight)
        hidden = hidden + self.dropout(update)
        return hidden + self.dropout(self.ffn(self.norm_ffn(hidden)))


class ParentCrossAttention(nn.Module):
    """Topology-only coarse-to-fine attention over PoU parent candidates."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm_child = nn.LayerNorm(hidden_dim)
        self.norm_parent = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        child: torch.Tensor,
        parent: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_child, _ = child.shape
        candidate_index = candidate_index.long()
        valid = candidate_index >= 0
        index = candidate_index.clamp(min=0)

        query = self.query(self.norm_child(child)).reshape(batch_size, n_child, self.num_heads, self.head_dim)
        parent_norm = self.norm_parent(parent)
        key = self.key(parent_norm).reshape(batch_size, parent.shape[1], self.num_heads, self.head_dim)[:, index]
        value = self.value(parent_norm).reshape(batch_size, parent.shape[1], self.num_heads, self.head_dim)[:, index]
        logits = (query[:, :, None] * key).sum(dim=-1) / math.sqrt(self.head_dim)
        prior = torch.log(candidate_weight.to(logits).clamp(min=1.0e-12))
        logits = logits + prior[None, :, :, None]
        logits = logits.masked_fill(~valid[None, :, :, None], torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=2)
        update = (self.dropout(attention)[..., None] * value).sum(dim=2)
        child = child + self.dropout(self.output(update.reshape(batch_size, n_child, -1)))
        return child + self.dropout(self.ffn(self.norm_ffn(child)))


class PrincipalStretchGraphTransformer(nn.Module):
    """Sparse graph U-transformer that predicts an SPD right-stretch field."""

    def __init__(
        self,
        hierarchy: Hierarchy,
        tets: np.ndarray,
        n_verts: int,
        config: GraphTransformerConfig | None = None,
    ):
        super().__init__()
        self.config = config or GraphTransformerConfig()
        self.n_levels = len(hierarchy.levels)
        hidden_dim = self.config.hidden_dim
        num_heads = self.config.num_heads

        tets = np.asarray(tets, dtype=np.int64)
        self.register_buffer("tets", torch.as_tensor(tets, dtype=torch.int64))

        # Distribute each vertex load among incident tets in proportion to
        # rest volume.  Summing the resulting tet loads exactly recovers the
        # total vertex load, including on irregular-valence meshes.
        incident_volume = np.zeros(n_verts, dtype=np.float64)
        np.add.at(incident_volume, tets.reshape(-1), np.repeat(hierarchy.tet_vol, 4))
        corner_weight = hierarchy.tet_vol[:, None] / incident_volume[tets]
        self.register_buffer("corner_force_weight", torch.as_tensor(corner_weight, dtype=torch.float32))

        self._register_level(0, hierarchy.tet_adj, hierarchy.tet_w_adj, hierarchy.tet_vol, hierarchy.tet_c0)
        child_volume = hierarchy.tet_vol
        for level, hierarchy_level in enumerate(hierarchy.levels, start=1):
            self._register_level(
                level,
                hierarchy_level.adj,
                hierarchy_level.w_adj,
                hierarchy_level.vol,
                hierarchy_level.c0,
            )
            self.register_buffer(f"assign_{level}", torch.as_tensor(hierarchy_level.assign, dtype=torch.int64))
            self.register_buffer(f"child_volume_{level}", torch.as_tensor(child_volume, dtype=torch.float32))
            self.register_buffer(f"pou_index_{level}", torch.as_tensor(hierarchy_level.pou_idx, dtype=torch.int64))
            self.register_buffer(f"pou_weight_{level}", torch.as_tensor(hierarchy_level.pou_w, dtype=torch.float32))
            child_volume = hierarchy_level.vol

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(NODE_FEATURE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(self.n_levels + 1)
            ]
        )
        self.down_attention = nn.ModuleList(
            [
                RelationAttentionBlock(hidden_dim, num_heads, EDGE_FEATURE_DIM, self.config.dropout)
                for _ in range(self.n_levels + 1)
            ]
        )
        self.down_fusion = nn.ModuleList([_mlp(2 * hidden_dim, hidden_dim, hidden_dim) for _ in range(self.n_levels)])
        self.up_cross_attention = nn.ModuleList(
            [ParentCrossAttention(hidden_dim, num_heads, self.config.dropout) for _ in range(self.n_levels)]
        )
        self.up_attention = nn.ModuleList(
            [
                RelationAttentionBlock(hidden_dim, num_heads, EDGE_FEATURE_DIM, self.config.dropout)
                for _ in range(self.n_levels)
            ]
        )
        self.output_head = _mlp(hidden_dim, hidden_dim, 6)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def _register_level(
        self,
        level: int,
        adjacency: np.ndarray,
        edge_weight: np.ndarray,
        volume: np.ndarray,
        rest_centroid: np.ndarray,
    ):
        valid = adjacency >= 0
        index = np.clip(adjacency, 0, None)
        neighbor_rest = rest_centroid[index]
        rest_offset = neighbor_rest - rest_centroid[:, None, :]
        rest_length = np.linalg.norm(rest_offset, axis=-1)
        rest_length = np.where(valid, np.maximum(rest_length, 1.0e-12), 1.0)
        rest_direction = rest_offset / rest_length[:, :, None]
        positive = edge_weight[valid]
        scale = float(positive.mean()) if positive.size else 1.0
        log_weight = np.where(valid, np.log(np.maximum(edge_weight / scale, 1.0e-12)), 0.0)

        self.register_buffer(f"adjacency_{level}", torch.as_tensor(adjacency, dtype=torch.int64))
        self.register_buffer(f"edge_weight_{level}", torch.as_tensor(edge_weight, dtype=torch.float32))
        self.register_buffer(f"volume_{level}", torch.as_tensor(volume, dtype=torch.float32))
        self.register_buffer(f"rest_length_{level}", torch.as_tensor(rest_length, dtype=torch.float32))
        self.register_buffer(f"rest_direction_{level}", torch.as_tensor(rest_direction, dtype=torch.float32))
        self.register_buffer(f"log_edge_weight_{level}", torch.as_tensor(log_weight, dtype=torch.float32))

    def _level_buffer(self, name: str, level: int) -> torch.Tensor:
        return getattr(self, f"{name}_{level}")

    def conservative_tet_load(self, force: torch.Tensor) -> torch.Tensor:
        """Convert vertex forces [N] to exactly conservative per-tet loads [N]."""
        batched = force.dim() == 3
        if not batched:
            force = force[None]
        weight = self.corner_force_weight.to(dtype=force.dtype, device=force.device)
        tet_force = (force[:, self.tets] * weight[None, :, :, None]).sum(dim=2)
        return tet_force if batched else tet_force[0]

    @staticmethod
    def _project(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bnji,bnj->bni", rotation, vector)

    def _node_features(self, fields: dict[str, torch.Tensor], gravity: torch.Tensor, level: int) -> torch.Tensor:
        H = fields["H"]
        H_previous = fields["H_previous"]
        rotation = fields["rotation"]
        gravity_material = self._project(rotation, gravity[:, None].expand(-1, H.shape[1], -1))
        force_material = self._project(rotation, fields["force"])
        velocity_material = self._project(rotation, fields["velocity"])
        determinant = torch.linalg.det(fields["F"])
        volume = self._level_buffer("volume", level).to(H)
        log_relative_volume = torch.log(volume / volume.mean()).expand(H.shape[0], -1)

        feature = torch.cat(
            [
                sym_to_vec(H),
                sym_to_vec(H_previous),
                sym_to_vec(H - H_previous),
                torch.asinh(gravity_material / 10.0),
                torch.asinh(force_material / 30.0),
                torch.asinh(velocity_material),
                torch.log(fields["mu"].clamp(min=1.0) / 1.0e5),
                torch.log(fields["lam"].clamp(min=1.0) / 1.0e5),
                fields["pin"],
                log_relative_volume[..., None],
                torch.sign(determinant)[..., None],
                torch.log(determinant.abs().clamp(min=1.0e-6)).clamp(min=-8.0, max=8.0)[..., None],
                _spectral_invariants(H),
            ],
            dim=-1,
        )
        if feature.shape[-1] != NODE_FEATURE_DIM:
            raise RuntimeError(f"internal node feature size {feature.shape[-1]} != {NODE_FEATURE_DIM}")
        return feature.to(dtype=self.output_head[-1].weight.dtype)

    def _edge_features(self, fields: dict[str, torch.Tensor], level: int) -> torch.Tensor:
        adjacency = self._level_buffer("adjacency", level)
        valid = adjacency >= 0
        index = adjacency.clamp(min=0)
        centroid = fields["centroid"]
        rotation = fields["rotation"]
        H = fields["H"]
        offset = centroid[:, index] - centroid[:, :, None]
        rest_length = self._level_buffer("rest_length", level).to(offset)[None, :, :, None]
        offset_material = torch.einsum("bnji,bnkj->bnki", rotation, offset) / rest_length
        extension = offset.norm(dim=-1, keepdim=True) / rest_length - 1.0
        neighbor_rotation = rotation[:, index]
        relative_rotation = torch.einsum("bnji,bnkjl->bnkil", rotation, neighbor_rotation)

        neighbor_H = H[:, index]
        H_difference = neighbor_H - H[:, :, None]
        trace_difference = H_difference.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        commutator = H[:, :, None] @ neighbor_H - neighbor_H @ H[:, :, None]
        tensor_relation = torch.stack(
            [
                torch.sqrt((H_difference * H_difference).sum(dim=(-2, -1)) + 1.0e-16),
                trace_difference,
                torch.sqrt((commutator * commutator).sum(dim=(-2, -1)) + 1.0e-16),
            ],
            dim=-1,
        )
        rest_direction = self._level_buffer("rest_direction", level).to(offset_material)
        log_weight = self._level_buffer("log_edge_weight", level).to(offset_material)
        feature = torch.cat(
            [
                offset_material,
                rest_direction[None].expand(offset.shape[0], -1, -1, -1),
                extension,
                relative_rotation.flatten(-2),
                tensor_relation,
                log_weight[None, :, :, None].expand(offset.shape[0], -1, -1, -1),
            ],
            dim=-1,
        )
        feature = torch.where(valid[None, :, :, None], feature, torch.zeros_like(feature))
        if feature.shape[-1] != EDGE_FEATURE_DIM:
            raise RuntimeError(f"internal edge feature size {feature.shape[-1]} != {EDGE_FEATURE_DIM}")
        return feature.to(dtype=self.output_head[-1].weight.dtype)

    def _prepare(
        self,
        state: SolverState,
        x_current: torch.Tensor,
        x_previous: torch.Tensor,
        force: torch.Tensor,
        gravity: torch.Tensor,
        mu: torch.Tensor,
        lam: torch.Tensor,
        pin: torch.Tensor,
    ) -> tuple[bool, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batched = x_current.dim() == 3
        if not batched:
            x_current = x_current[None]
            x_previous = x_previous[None]
            force = force[None]
        batch_size = x_current.shape[0]
        if gravity.dim() == 1:
            gravity = gravity[None].expand(batch_size, -1)
        if mu.dim() == 1:
            mu = mu[None].expand(batch_size, -1)
            lam = lam[None].expand(batch_size, -1)
            pin = pin[None].expand(batch_size, -1)

        x_tet = x_current[:, self.tets]
        x_previous_tet = x_previous[:, self.tets]
        F = torch.einsum("tac,btad->btdc", state.J, x_tet)
        F_previous = torch.einsum("tac,btad->btdc", state.J, x_previous_tet)
        C = F.transpose(-1, -2) @ F
        C_previous = F_previous.transpose(-1, -2) @ F_previous
        H = 0.5 * sym_log(spd_floor(C, lam_min=0.05**2))
        H_previous = 0.5 * sym_log(spd_floor(C_previous, lam_min=0.05**2))
        centroid = x_tet.mean(dim=2)
        previous_centroid = x_previous_tet.mean(dim=2)
        force_tet = self.conservative_tet_load(force)
        rotation = polar_rotation(F)

        fields: dict[str, torch.Tensor] = {
            "H": H,
            "H_previous": H_previous,
            "F": F,
            "rotation": rotation,
            "centroid": centroid,
            "velocity": (centroid - previous_centroid) / self.config.dt,
            "force": force_tet,
            "mu": mu[..., None],
            "lam": lam[..., None],
            "pin": pin[..., None],
        }
        base_H = H.to(dtype=self.output_head[-1].weight.dtype)
        node_features: list[torch.Tensor] = []
        edge_features: list[torch.Tensor] = []

        for level in range(self.n_levels + 1):
            node_features.append(self._node_features(fields, gravity, level))
            edge_features.append(self._edge_features(fields, level))
            if level == self.n_levels:
                break
            next_level = level + 1
            assign = self._level_buffer("assign", next_level)
            child_volume = self._level_buffer("child_volume", next_level).to(H)
            pooled: dict[str, torch.Tensor] = {}
            for name, value in fields.items():
                if name == "force":
                    pooled[name] = _batch_pool_sum(value, assign)
                elif name != "rotation":
                    pooled[name] = _batch_pool_mean(value, assign, child_volume)
            pooled["rotation"] = polar_rotation(pooled["F"])
            fields = pooled

        return batched, base_H, node_features, edge_features

    def forward(
        self,
        state: SolverState,
        x_current: torch.Tensor,
        x_previous: torch.Tensor,
        force: torch.Tensor,
        gravity: torch.Tensor,
        mu: torch.Tensor,
        lam: torch.Tensor,
        pin: torch.Tensor,
    ) -> torch.Tensor:
        """Predict material-frame target right stretches.

        Args:
            state: Local-global decoder state for the shared tet mesh.
            x_current: Current vertex positions [m], ``(V, 3)`` or ``(B, V, 3)``.
            x_previous: Previous positions [m], same shape as ``x_current``.
            force: Current per-vertex external force [N], same shape as positions.
            gravity: Gravity [m/s^2], ``(3,)`` or ``(B, 3)``.
            mu: Per-tet first Lamé parameter [Pa], ``(T,)`` or ``(B, T)``.
            lam: Per-tet second Lamé parameter [Pa], same shape as ``mu``.
            pin: Per-tet pin-incidence flag, same shape as ``mu``.

        Returns:
            SPD target right stretches, ``(T, 3, 3)`` or ``(B, T, 3, 3)``.
        """
        batched, base_H, node_features, edge_features = self._prepare(
            state, x_current, x_previous, force, gravity, mu, lam, pin
        )

        hidden = self.encoders[0](node_features[0])
        hidden = self.down_attention[0](
            hidden,
            edge_features[0],
            self._level_buffer("adjacency", 0),
            self._level_buffer("edge_weight", 0),
        )
        skip = [hidden]

        for level in range(1, self.n_levels + 1):
            assign = self._level_buffer("assign", level)
            child_volume = self._level_buffer("child_volume", level).to(hidden)
            restricted = _batch_pool_mean(hidden, assign, child_volume)
            encoded = self.encoders[level](node_features[level])
            hidden = self.down_fusion[level - 1](torch.cat([restricted, encoded], dim=-1))
            hidden = self.down_attention[level](
                hidden,
                edge_features[level],
                self._level_buffer("adjacency", level),
                self._level_buffer("edge_weight", level),
            )
            skip.append(hidden)

        for level in range(self.n_levels - 1, -1, -1):
            parent_level = level + 1
            hidden = self.up_cross_attention[level](
                skip[level],
                hidden,
                self._level_buffer("pou_index", parent_level),
                self._level_buffer("pou_weight", parent_level),
            )
            hidden = self.up_attention[level](
                hidden,
                edge_features[level],
                self._level_buffer("adjacency", level),
                self._level_buffer("edge_weight", level),
            )

        delta_H = _radially_bound_symmetric(self.output_head(hidden), self.config.max_hencky_update)
        target = sym_exp(base_H + delta_H)
        return target if batched else target[0]
