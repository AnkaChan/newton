# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared construction and invocation of principal-stretch predictors."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import torch_solver
from .graph_transformer import GraphTransformerConfig, PrincipalStretchGraphTransformer
from .hierarchy import build_hierarchy
from .model import StretchNet, build_face_adjacency, build_features
from .torch_solver import SolverState

PREDICTOR_KINDS = ("mlp", "graph-transformer")

_STATIC_GRAPH_KEYS = ("tets", "corner_force_weight")
_STATIC_GRAPH_PREFIXES = (
    "adjacency_",
    "edge_weight_",
    "volume_",
    "rest_length_",
    "rest_direction_",
    "log_edge_weight_",
    "assign_",
    "child_volume_",
    "pou_index_",
    "pou_weight_",
    "representative_",
)


def _is_static_graph_key(name: str) -> bool:
    return name in _STATIC_GRAPH_KEYS or name.startswith(_STATIC_GRAPH_PREFIXES)


def predictor_architecture_version(predictor: StretchPredictor) -> int | None:
    """Return the graph architecture version, or ``None`` for the flat MLP."""
    if predictor.kind != "graph-transformer":
        return None
    return int(predictor.model.config.architecture_version)


def validate_static_pin_trajectory(
    rest_q: np.ndarray,
    pinned_indices: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Reject moving-boundary data from the legacy single-pin-target schema."""
    rest_q = np.asarray(rest_q)
    pinned_indices = np.asarray(pinned_indices, dtype=np.int64)
    positions = np.asarray(positions)
    expected = np.broadcast_to(rest_q[pinned_indices], (positions.shape[0], pinned_indices.size, 3))
    actual = positions[:, pinned_indices]
    if not np.array_equal(actual, expected):
        discrepancy = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        raise ValueError(
            "architecture v3 legacy trajectory routing supports only static rest pins; "
            f"observed maximum pinned displacement {discrepancy:.6e} m"
        )


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
            graph_config = dataclasses.asdict(self.model.config)
            if self.model.config.architecture_version != 4:
                # Preserve the learned checkpoint schema of v0-v3 exactly.
                # V5 also uses the explicit log-stretch formula and must not
                # authenticate the unrelated v4 multiplicative cap.
                del graph_config["max_multiplicative_update"]
            config["graph_transformer"] = graph_config
        return config

    def predict_deformation_gradient(
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
        """Predict v3/v4 full deformation gradients through the shared adapter.

        The ordinary :meth:`forward` path deliberately remains the legacy SPD
        right-stretch API so v0-v3 training, evaluation, and checkpoints keep
        their existing semantics.  Architecture v4 rejects that legacy path.
        """
        if self.kind != "graph-transformer":
            raise RuntimeError("full deformation-gradient prediction requires a graph-transformer predictor")
        return self.model.predict_deformation_gradient(state, x_current, x_previous, force, gravity, mu, lam, pin)

    def predict_principal_stretch_update(
        self,
        state: SolverState,
        x_current: torch.Tensor,
        x_previous: torch.Tensor,
        x_iterate: torch.Tensor,
        force: torch.Tensor,
        gravity: torch.Tensor,
        mu: torch.Tensor,
        lam: torch.Tensor,
        pin: torch.Tensor,
        normalized_residual: torch.Tensor,
        constraint_normal: torch.Tensor,
        normalized_constraint_slack: torch.Tensor,
        *,
        iteration_fraction: float | torch.Tensor,
        physical_dt: float | torch.Tensor,
        head_mode: str = "learned",
        head_permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict one architecture-v5 bounded principal-stretch update."""
        if self.kind != "graph-transformer":
            raise RuntimeError("iterative principal-stretch prediction requires a graph-transformer predictor")
        return self.model.predict_principal_stretch_update(
            state,
            x_current,
            x_previous,
            x_iterate,
            force,
            gravity,
            mu,
            lam,
            pin,
            normalized_residual,
            constraint_normal,
            normalized_constraint_slack,
            iteration_fraction=iteration_fraction,
            physical_dt=physical_dt,
            head_mode=head_mode,
            head_permutation=head_permutation,
        )


def resolve_solver_iterations(predictor: StretchPredictor, requested: int | None) -> int:
    """Resolve legacy defaults and enforce the one-shot full-gradient decoder."""
    version = predictor_architecture_version(predictor)
    if requested is None:
        if version == 5:
            raise ValueError("architecture v5 requires an explicit fixed solver iteration count")
        return 1 if version in (3, 4) else 10
    if requested < 1:
        raise ValueError("solver_iterations must be positive")
    if version in (3, 4) and requested != 1:
        raise ValueError(f"architecture v{version} uses exactly one global projection; solver_iterations must be 1")
    return requested


def predictor_decoder_work(
    predictor: StretchPredictor,
    solver_iterations: int,
    blocks: int,
) -> dict[str, int | str]:
    """Return explicit, serializable decoder work for one predicted step."""
    if blocks < 1:
        raise ValueError("blocks must be positive")
    version = predictor_architecture_version(predictor)
    if version == 5:
        if blocks != 1:
            raise ValueError("architecture v5 iterative routing requires blocks=1")
        if solver_iterations < 1:
            raise ValueError("architecture v5 solver_iterations must be positive")
        return {
            "schema_version": 3,
            "target": "principal-log-stretch-full-deformation-gradient",
            "decoder": "iterative-weighted-global-projection",
            "predictor_passes": solver_iterations,
            "compatibility_projection_calls": solver_iterations,
            "local_polar_sweeps": 0,
            "common_residual_evaluations": solver_iterations + 1,
            "common_objective_evaluations": solver_iterations + 1,
            "state_validity_evaluations": solver_iterations + 1,
            "constraint_preparations": solver_iterations,
            "constraint_applications": solver_iterations,
            "physical_step_authentications": 2 * solver_iterations + 3,
            "common_objective_authentications": 2 * solver_iterations + 3,
        }
    if version in (3, 4):
        if blocks != 1:
            raise ValueError(
                f"architecture v{version} uses one predictor pass and one global projection; blocks must be 1"
            )
        if solver_iterations != 1:
            raise ValueError(f"architecture v{version} uses exactly one global projection; solver_iterations must be 1")
        return {
            "schema_version": 1,
            "target": "full-deformation-gradient",
            "decoder": "weighted-global-projection",
            "predictor_passes": 1,
            "global_triangular_solves": 1,
            "local_polar_sweeps": 0,
        }

    iterations_per_block = max(1, solver_iterations // blocks)
    sweeps = blocks * iterations_per_block
    return {
        "schema_version": 1,
        "target": "right-stretch",
        "decoder": "local-global-polar",
        "predictor_passes": blocks,
        "global_triangular_solves": sweeps,
        "local_polar_sweeps": sweeps,
    }


def decode_predictor_step(
    predictor: StretchPredictor,
    state: SolverState,
    x_current: torch.Tensor,
    x_previous: torch.Tensor,
    force: torch.Tensor,
    gravity: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
    pin: torch.Tensor,
    S_current: torch.Tensor | None,
    S_previous: torch.Tensor | None,
    pinned_targets: torch.Tensor,
    *,
    x_init: torch.Tensor | None,
    solver_iterations: int,
    blocks: int,
) -> torch.Tensor:
    """Predict and reconstruct one step with version-correct decoder work.

    Architectures v3 and v4 predict a complete deformation-gradient field and
    use exactly one global compatibility projection.  Earlier graph versions
    and the flat MLP retain their existing unrolled stretch/polar decoder.
    """
    work = predictor_decoder_work(predictor, solver_iterations, blocks)
    if predictor_architecture_version(predictor) == 5:
        raise RuntimeError("architecture v5 requires the iterative solver route with objective and constraint state")
    if work["target"] == "full-deformation-gradient":
        F_target = predictor.predict_deformation_gradient(state, x_current, x_previous, force, gravity, mu, lam, pin)
        return torch_solver.project_deformation_gradient(state, F_target, pinned_targets)

    if x_init is None:
        raise ValueError("the legacy right-stretch decoder requires x_init")
    if S_current is None or S_previous is None:
        raise ValueError("the legacy right-stretch decoder requires current and previous stretches")
    iterations_per_block = max(1, solver_iterations // blocks)
    x_next = x_init
    S_block = S_current
    for block in range(blocks):
        S_target = predictor(
            state,
            x_current,
            x_previous,
            force,
            gravity,
            mu,
            lam,
            pin,
            S_block,
            S_previous,
        )
        x_next = torch_solver.solve(
            state,
            S_target.double(),
            pinned_targets,
            x_init=x_next,
            n_iters=iterations_per_block,
        )
        if block + 1 < blocks:
            S_block = torch_solver.compute_S_from_x(state, x_next)
    return x_next


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
    model = PrincipalStretchGraphTransformer(hierarchy, tets, rest_q.shape[0], config, rest_q=rest_q)
    predictor = StretchPredictor(kind, model, residual=True).to(device=device, dtype=dtype)
    if config.architecture_version == 5:
        predictor.model.static_graph_sha256 = predictor.model.compute_static_graph_sha256()
    return predictor


def checkpoint_predictor_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Read new metadata, falling back to pre-adapter flat checkpoints."""
    if "predictor_config" in checkpoint:
        config = dict(checkpoint["predictor_config"])
        if config.get("kind") == "graph-transformer":
            graph_config = dict(config.get("graph_transformer", {}))
            if "architecture_version" not in graph_config:
                state_dict = checkpoint.get("state_dict", {})
                graph_config["architecture_version"] = 0 if any(_is_static_graph_key(k) for k in state_dict) else 1
            architecture_version = graph_config["architecture_version"]
            if architecture_version in (0, 1, 2):
                # This field is unused by v0-v2 and did not exist in their
                # saved metadata.  Materialize its default so provenance sees
                # a canonical configuration without changing legacy inference.
                graph_config.setdefault("max_rotation_update", GraphTransformerConfig.max_rotation_update)
            elif architecture_version == 3:
                if "max_rotation_update" not in graph_config:
                    raise ValueError("architecture-v3 checkpoint is missing max_rotation_update metadata")
            elif architecture_version == 4:
                if "max_multiplicative_update" not in graph_config:
                    raise ValueError("architecture-v4 checkpoint is missing max_multiplicative_update metadata")
                # Rotation-head parameters are reused as the skew part of the
                # joint v4 correction; the independent v3 cap is not used.
                graph_config.setdefault("max_rotation_update", GraphTransformerConfig.max_rotation_update)
            elif architecture_version == 5:
                if "max_rotation_update" not in graph_config:
                    raise ValueError("architecture-v5 checkpoint is missing max_rotation_update metadata")
                if "max_multiplicative_update" in graph_config:
                    raise ValueError("architecture-v5 checkpoint must not contain max_multiplicative_update metadata")
            else:
                raise ValueError(f"unsupported graph architecture version {architecture_version!r}")
            config["graph_transformer"] = graph_config
        return config
    args = checkpoint.get("args", {})
    return {
        "kind": args.get("predictor", "mlp"),
        "residual": bool(args.get("residual", False)),
    }


def load_stretch_predictor_state(predictor: StretchPredictor, checkpoint: dict[str, Any]) -> None:
    """Strictly load learned state while rebuilding mesh-static graph buffers."""
    state_dict = dict(checkpoint["state_dict"])
    if predictor.kind == "graph-transformer":
        state_dict = {name: value for name, value in state_dict.items() if not _is_static_graph_key(name)}
    predictor.model.load_state_dict(state_dict, strict=True)
