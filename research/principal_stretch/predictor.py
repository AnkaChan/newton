# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared construction and invocation of principal-stretch predictors."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .graph_transformer import GraphTransformerConfig, PrincipalStretchGraphTransformer
from .hierarchy import build_hierarchy
from .model import StretchNet, build_face_adjacency, build_features
from .torch_solver import SolverState

PREDICTOR_KINDS = ("mlp", "graph-transformer")


class StretchPredictor(nn.Module):
    """Adapter that gives the flat baseline and graph transformer one API."""

    def __init__(self, kind: str, model: nn.Module, residual: bool, face_adjacency: np.ndarray | None = None):
        super().__init__()
        if kind not in PREDICTOR_KINDS:
            raise ValueError(f"unknown predictor {kind!r}; expected one of {PREDICTOR_KINDS}")
        self.kind = kind
        self.model = model
        self.residual = residual
        if face_adjacency is not None:
            self.register_buffer("face_adjacency", torch.as_tensor(face_adjacency, dtype=torch.int64))
        else:
            self.register_buffer("face_adjacency", None)

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
        S_current: torch.Tensor,
        S_previous: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target right stretches with either implementation."""
        if self.kind == "graph-transformer":
            return self.model(state, x_current, x_previous, force, gravity, mu, lam, pin)

        dtype = next(self.model.parameters()).dtype
        S_current_network = S_current.to(dtype)
        feature = build_features(
            S_current_network,
            S_previous.to(dtype),
            gravity.to(dtype),
            force.to(dtype),
            mu.to(dtype),
            lam.to(dtype),
            pin.to(dtype),
            state.tets,
            self.face_adjacency,
        )
        return self.model(feature, S_base=S_current_network if self.residual else None)

    def checkpoint_config(self) -> dict[str, Any]:
        """Serializable architecture metadata for checkpoint reconstruction."""
        config: dict[str, Any] = {"kind": self.kind, "residual": self.residual}
        if self.kind == "graph-transformer":
            config["graph_transformer"] = dataclasses.asdict(self.model.config)
        return config


def build_stretch_predictor(
    kind: str,
    rest_q: np.ndarray,
    tets: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    *,
    residual: bool,
    graph_config: GraphTransformerConfig | dict[str, Any] | None = None,
) -> StretchPredictor:
    """Build a predictor and all mesh-static topology exactly once."""
    if kind == "mlp":
        model = StretchNet().to(device=device, dtype=dtype)
        return StretchPredictor(kind, model, residual, build_face_adjacency(tets)).to(device)
    if kind != "graph-transformer":
        raise ValueError(f"unknown predictor {kind!r}; expected one of {PREDICTOR_KINDS}")

    if graph_config is None:
        config = GraphTransformerConfig()
    elif isinstance(graph_config, dict):
        config = GraphTransformerConfig(**graph_config)
    else:
        config = graph_config
    hierarchy = build_hierarchy(
        tets,
        rest_q,
        n_levels=config.n_levels,
        target=config.cluster_size,
    )
    model = PrincipalStretchGraphTransformer(hierarchy, tets, rest_q.shape[0], config)
    return StretchPredictor(kind, model, residual=True).to(device=device, dtype=dtype)


def checkpoint_predictor_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Read new metadata, falling back to pre-adapter flat checkpoints."""
    if "predictor_config" in checkpoint:
        return dict(checkpoint["predictor_config"])
    args = checkpoint.get("args", {})
    return {
        "kind": args.get("predictor", "mlp"),
        "residual": bool(args.get("residual", False)),
    }
