# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Authenticated checkpoint and evaluation contracts for iterative PSS v5.

This module is intentionally separate from the v3/v4 trainer and loaders.  A
v5 checkpoint is self-verifying: it binds learned tensors, integrity-only
optimizer/RNG continuation snapshots, the
whole-trajectory training split, and every fixed-work solver choice.  That
self-contained verification does not bind an evaluation problem.  A separate
:class:`V5EvaluationBinding` authenticates a held-out trajectory, sample
selection, and the physical timestep used by the common objective.

The legacy ``GraphTransformerConfig.dt`` field remains serialized because it
is part of that model configuration, but v5 runtime physics must obtain its
timestep from each common-objective context.  Evaluation bindings therefore
record ``physical_dt_seconds`` explicitly and never require the held-out mesh,
history, or timestep to equal the training data.  This foundation deliberately
does not claim exact optimizer/RNG resume capability; that requires the future
trainer to bind parameter order and prove a save/load/next-update parity test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import struct
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .graph_transformer import GraphTransformerConfig, PrincipalStretchGraphTransformer
from .hierarchy import build_hierarchy
from .iterative_solver import (
    CandidateEvaluation,
    ConstraintApplication,
    ConstraintObservation,
    IdentityConstraintHook,
    IterativeSolverConfig,
    IterativeSolverIteration,
    IterativeSolverResult,
    IterativeSolverWork,
    PhysicalStepContext,
    ProposalSafeguardConfig,
    _tensor_bytes_equal,
    _validate_config_execution_dtype,
    validate_physical_objective_integration,
)
from .predictor import StretchPredictor, predictor_architecture_version
from .torch_solver import (
    OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
    OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
    ProjectionDiagnostics,
    SolverState,
    compute_F,
    project_deformation_gradient,
    projection_state_sha256,
    validate_authenticated_operator_geometry,
)
from .v5_dataset import (
    DataAccessLedger,
    DataAccessPurpose,
    DataAccessScope,
    DatasetRole,
    SamplingSchedule,
    SplitManifest,
    TrajectoryRecord,
    verify_trajectory_topology,
)
from .v5_objective import (
    CommonObjectiveContext,
    _common_objective_components_trusted,
    _common_objective_residual_trusted,
    common_objective_components,
    common_objective_residual,
)
from .v5_training import (
    CompatibleStateLossConfig,
    PotentialExcessLossConfig,
    PrincipalStretchLabelConfig,
    RepresentationLossConfig,
)

_SCHEMA_VERSION = 5
_CHECKPOINT_CONTRACT = "pss-iterative-principal-stretch-checkpoint-v5"
_SOLVER_CONTRACT = "pss-iterative-principal-stretch-solver-contract-v5"
_EVALUATION_SCHEMA_VERSION = 3
_EVALUATION_CONTRACT = "pss-v5-held-out-evaluation-binding-v3"
_TRAJECTORY_CONTRACT = "pss-v5-dataset-trajectory-v3"
_MODEL_SEMANTICS = "weight-shared-residual-aware-principal-stretch-v5"
_REPRESENTATION_FORMULA = "F_target=A_iterate@exp(skew(omega))@exp(H_iterate+delta_H)"
_RESIDUAL_DEFINITION = "exact-common-objective-gradient-at-current-iterate-v1"
_PHYSICAL_TIMESTEP_SOURCE = "common-objective-context-per-sample"
_LEGACY_GRAPH_DT_POLICY = "serialized-for-graph-config-compatibility-not-used-by-v5-runtime"
_FALLBACK_POLICY = "none"
_MINIMUM_PRINCIPAL_STRETCH = 0.05
_RESIDUAL_NORMALIZATION = "divide-by-common-objective-context-residual-scale"
_RESIDUAL_SCALE_SOURCE = "derived-max-material-or-inertial-force-with-1e-12N-floor-v1"
_CONSTRAINT_PREPARE_CADENCE = "before-every-learned-iteration"
_CONSTRAINT_APPLY_CADENCE = "after-every-projection-before-residual"
_CONSTRAINT_GRADIENT_POLICY = "hook-defined-autograd"
_OBJECTIVE_POLICY = "require-nonincreasing"
_RESIDUAL_POLICY = "require-nonincreasing"
_INITIALIZER_POLICY = "persistence"
_IDENTITY_CONSTRAINT_DESCRIPTOR = {
    "schema_version": 1,
    "kind": "identity",
    "refresh_policy": "none",
    "displacement_reference": "current-iterate",
}
_V5_PARAMETER_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}
_REPLAY_TOLERANCES = {
    "torch.float32": (1.0e-6, 1.0e-7),
    "torch.float64": (1.0e-12, 1.0e-12),
}
_MAX_OBJECTIVE_INCREASE_TOLERANCE = 1.0e-12
_MAX_NORMALIZED_RESIDUAL_INCREASE_TOLERANCE = 1.0e-12
_MAX_PROJECTION_RELATIVE_TOLERANCE = 1.0e-5
_REPRESENTATION_LABEL_FORMULA = "principal-stretch-labels-explicit-log-stretch-v1"
_REPRESENTATION_LOSS_FORMULA = "cap-normalized-delta-h-plus-omega-mse-v1"
_COMPATIBLE_STATE_LOSS_FORMULA = "mass-position-plus-volume-deformation-relative-mse-v1"
_POTENTIAL_EXCESS_LOSS_FORMULA = "signed-common-objective-potential-excess-v1"
_RNG_ALGORITHM = "torch-cpu-plus-numpy-pcg64"
_BATCH_STREAM_CONTRACT = "pss-v5-static-layout-homogeneous-trajectory-first-sampling-v3"
_RUNTIME_CLAIM_SCOPE = "authenticated-development-replay-not-learned-contribution-or-promotion-evidence"
_SCHEMA_V6_VERSION = 6
_CHECKPOINT_V6_CONTRACT = "pss-iterative-principal-stretch-checkpoint-v6"
_SOLVER_V6_CONTRACT = "pss-iterative-principal-stretch-solver-contract-v6"
_EVALUATION_V6_SCHEMA_VERSION = 4
_EVALUATION_V6_CONTRACT = "pss-v6-fixed-candidate-held-out-evaluation-binding-v4"
_MODEL_V6_SEMANTICS = "weight-shared-residual-aware-principal-stretch-fixed-candidate-v6"
_TRAINING_PROPOSAL_POLICY = "direct-unselected-record-v1"


def _jsonable(value: object) -> object:
    """Return a finite canonical-JSON-compatible value."""
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("authenticated JSON mappings require non-empty string keys")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("authenticated JSON floats must be finite")
        return value
    raise TypeError(f"unsupported authenticated JSON type {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Hash finite JSON data with sorted keys and no insignificant spaces."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_text(value: object) -> str:
    return _canonical_json_bytes(value).decode("utf-8")


def _decoded_canonical_json(value: str, name: str) -> object:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be canonical JSON text")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if _canonical_json_text(decoded) != value:
        raise ValueError(f"{name} is not canonical JSON")
    return decoded


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    result = _require_nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_finite_float(value: object, name: str, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array, dtype=canonical_dtype)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(json.dumps(canonical.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    if tensor.layout != torch.strided:
        raise ValueError("authenticated state tensors must use strided layout")
    if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
        raise ValueError("authenticated state tensors must be finite")
    try:
        array = tensor.detach().cpu().contiguous().numpy()
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"tensor dtype {tensor.dtype} cannot be hashed canonically") from error
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": _array_sha256(array),
    }


def learned_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash learned tensor names, dtypes, shapes, and canonical CPU bytes."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("learned state_dict must be a non-empty mapping")
    records: list[dict[str, object]] = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not name:
            raise ValueError("learned state_dict keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"learned state_dict entry {name!r} is not a tensor")
        records.append({"name": name, **_tensor_record(tensor)})
    return canonical_json_sha256({"contract": "named-tensor-state-v1", "tensors": records})


def _canonical_v5_graph_config(graph_config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(graph_config, Mapping):
        raise TypeError("graph_config must be a mapping")
    try:
        config = GraphTransformerConfig(**dict(graph_config))
    except (TypeError, ValueError) as error:
        raise ValueError("graph_config is not a valid architecture-v5 GraphTransformerConfig") from error
    canonical = dataclasses.asdict(config)
    canonical.pop("max_multiplicative_update")
    if config.architecture_version != 5:
        raise ValueError("graph architecture_version must be exactly 5")
    if not math.isfinite(config.dropout) or not 0.0 <= config.dropout < 1.0:
        raise ValueError("graph dropout must be finite and in [0, 1)")
    if _jsonable(graph_config) != canonical:
        raise ValueError("graph_config must contain the exact canonical architecture-v5 fields")
    return canonical


def _expected_v5_learned_schema(graph_config: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """Derive the mesh-independent learned schema from the real v5 module."""
    canonical = _canonical_v5_graph_config(graph_config)
    config = GraphTransformerConfig(**canonical)
    rest = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    hierarchy = build_hierarchy(tets, rest, n_levels=config.n_levels, target=config.cluster_size)
    with torch.random.fork_rng(devices=[]):
        model = PrincipalStretchGraphTransformer(hierarchy, tets, rest.shape[0], config, rest_q=rest)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def _validate_v5_learned_state(
    state_dict: Mapping[str, torch.Tensor],
    graph_config: Mapping[str, object],
    parameter_dtype: str,
) -> None:
    """Reject legacy, partial, topology-persistent, or arbitrary tensor maps."""
    expected_dtype = _V5_PARAMETER_DTYPES.get(parameter_dtype)
    if expected_dtype is None:
        raise ValueError(f"unsupported schema-v5 learned parameter dtype {parameter_dtype!r}")
    static_names = {"tets", "corner_force_weight"}
    static_prefixes = (
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
    persistent = [
        name
        for name in state_dict
        if name in static_names or any(name.startswith(prefix) for prefix in static_prefixes)
    ]
    if persistent:
        raise ValueError(f"schema-v5 learned state contains mesh-static keys: {sorted(persistent)}")
    required_families = (
        "encoders.",
        "down_attention.",
        "output_head.",
        "rotation_head.",
        "v5_context_encoder.",
    )
    missing_families = [
        prefix for prefix in required_families if not any(name.startswith(prefix) for name in state_dict)
    ]
    if missing_families:
        raise ValueError(f"schema-v5 learned state is missing required key families: {missing_families}")
    expected = _expected_v5_learned_schema(graph_config)
    if set(state_dict) != set(expected):
        missing = sorted(set(expected) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected))
        raise ValueError(
            "learned state does not match the exact architecture-v5 learned key schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, expected_shape in expected.items():
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
            raise ValueError(f"schema-v5 learned tensor {name!r} has the wrong shape")
        if not tensor.is_floating_point():
            raise ValueError(f"schema-v5 learned tensor {name!r} must have a floating dtype")
        if tensor.dtype != expected_dtype:
            raise ValueError(f"schema-v5 learned tensor {name!r} has dtype {tensor.dtype}; expected {parameter_dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"schema-v5 learned tensor {name!r} must be finite")


def _state_tree_record(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", **_tensor_record(value)}
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
            raise ValueError("authenticated state arrays must be finite")
        return {
            "kind": "numpy-array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_sha256(array),
        }
    if isinstance(value, Mapping):
        items: list[dict[str, object]] = []
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise TypeError(f"unsupported state mapping key type {type(key).__name__}")
            items.append(
                {
                    "key": {"type": type(key).__name__, "value": key},
                    "value": _state_tree_record(item),
                }
            )
        items.sort(key=lambda item: _canonical_json_text(item["key"]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_state_tree_record(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_state_tree_record(item) for item in value]}
    if isinstance(value, np.generic):
        return _state_tree_record(value.item())
    if isinstance(value, (str, int, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("authenticated state floats must be finite")
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported authenticated state type {type(value).__name__}")


def state_tree_sha256(value: object) -> str:
    """Hash a nested tensor/scalar continuation snapshot without pickle bytes."""
    return canonical_json_sha256(_state_tree_record(value))


def _clone_state_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {key: _clone_state_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_state_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_state_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported authenticated state type {type(value).__name__}")


def _strict_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        raise ValueError(f"{name} fields differ: missing={missing}, unexpected={unexpected}")


@dataclasses.dataclass(frozen=True)
class TrainingStage:
    """One registered immutable segment of the v5 training curriculum."""

    name: str
    start_update: int
    end_update: int
    label_config: PrincipalStretchLabelConfig | None = None
    representation_loss_config: RepresentationLossConfig | None = None
    compatible_state_loss_config: CompatibleStateLossConfig | None = None
    potential_excess_loss_config: PotentialExcessLossConfig | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.start_update, "training stage start_update")
        _require_positive_integer(self.end_update, "training stage end_update")
        if self.end_update <= self.start_update:
            raise ValueError("training stage end_update must exceed start_update")
        if self.name == "representation":
            if type(self.label_config) is not PrincipalStretchLabelConfig:
                raise ValueError("representation stage requires PrincipalStretchLabelConfig")
            if type(self.representation_loss_config) is not RepresentationLossConfig:
                raise ValueError("representation stage requires RepresentationLossConfig")
            if self.compatible_state_loss_config is not None or self.potential_excess_loss_config is not None:
                raise ValueError("representation stage cannot declare physics-stage losses")
            if (
                self.label_config.max_hencky_update != self.representation_loss_config.max_hencky_update
                or self.label_config.max_rotation_update != self.representation_loss_config.max_rotation_update
            ):
                raise ValueError("representation label and loss caps must match")
            return
        if self.name == "physics":
            if type(self.compatible_state_loss_config) is not CompatibleStateLossConfig:
                raise ValueError("physics stage requires CompatibleStateLossConfig")
            if type(self.potential_excess_loss_config) is not PotentialExcessLossConfig:
                raise ValueError("physics stage requires PotentialExcessLossConfig")
            if self.label_config is not None or self.representation_loss_config is not None:
                raise ValueError("physics stage cannot declare representation-stage losses")
            return
        raise ValueError("training stage name must be the registered 'representation' or 'physics' stage")

    @staticmethod
    def _term(formula: str, config: object) -> dict[str, object]:
        if not dataclasses.is_dataclass(config):
            raise TypeError("training loss config must be a dataclass")
        return {"formula": formula, "config": dataclasses.asdict(config)}

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        result: dict[str, object] = {
            "name": self.name,
            "start_update": self.start_update,
            "end_update": self.end_update,
        }
        if self.name == "representation":
            result["loss_contract"] = {
                "terms": ["principal_stretch_representation"],
                "label_construction": self._term(_REPRESENTATION_LABEL_FORMULA, self.label_config),
                "principal_stretch_representation": self._term(
                    _REPRESENTATION_LOSS_FORMULA,
                    self.representation_loss_config,
                ),
            }
        else:
            result["loss_contract"] = {
                "terms": ["compatible_state", "common_potential_excess"],
                "compatible_state": self._term(
                    _COMPATIBLE_STATE_LOSS_FORMULA,
                    self.compatible_state_loss_config,
                ),
                "common_potential_excess": self._term(
                    _POTENTIAL_EXCESS_LOSS_FORMULA,
                    self.potential_excess_loss_config,
                ),
            }
        return result

    @staticmethod
    def _config(
        term: object,
        *,
        formula: str,
        config_type: type,
        name: str,
    ) -> object:
        if not isinstance(term, Mapping):
            raise ValueError(f"training {name} contract must be a mapping")
        _strict_keys(term, {"formula", "config"}, f"training {name} contract")
        if term["formula"] != formula or not isinstance(term["config"], Mapping):
            raise ValueError(f"training {name} formula or config is not canonical")
        try:
            config = config_type(**dict(term["config"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"training {name} config is invalid") from error
        if dataclasses.asdict(config) != term["config"]:
            raise ValueError(f"training {name} config is not canonical")
        return config

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrainingStage:
        """Reconstruct and validate one canonical stage."""
        _strict_keys(value, {"name", "start_update", "end_update", "loss_contract"}, "training stage")
        loss = value["loss_contract"]
        if not isinstance(loss, Mapping):
            raise ValueError("training loss_contract must be a mapping")
        if value["name"] == "representation":
            _strict_keys(
                loss,
                {"terms", "label_construction", "principal_stretch_representation"},
                "representation loss contract",
            )
            if loss["terms"] != ["principal_stretch_representation"]:
                raise ValueError("representation stage terms are not canonical")
            return cls(
                name="representation",
                start_update=value["start_update"],
                end_update=value["end_update"],
                label_config=cls._config(
                    loss["label_construction"],
                    formula=_REPRESENTATION_LABEL_FORMULA,
                    config_type=PrincipalStretchLabelConfig,
                    name="label construction",
                ),
                representation_loss_config=cls._config(
                    loss["principal_stretch_representation"],
                    formula=_REPRESENTATION_LOSS_FORMULA,
                    config_type=RepresentationLossConfig,
                    name="principal-stretch representation loss",
                ),
            )
        if value["name"] == "physics":
            _strict_keys(
                loss,
                {"terms", "compatible_state", "common_potential_excess"},
                "physics loss contract",
            )
            if loss["terms"] != ["compatible_state", "common_potential_excess"]:
                raise ValueError("physics stage terms are not canonical")
            return cls(
                name="physics",
                start_update=value["start_update"],
                end_update=value["end_update"],
                compatible_state_loss_config=cls._config(
                    loss["compatible_state"],
                    formula=_COMPATIBLE_STATE_LOSS_FORMULA,
                    config_type=CompatibleStateLossConfig,
                    name="compatible-state loss",
                ),
                potential_excess_loss_config=cls._config(
                    loss["common_potential_excess"],
                    formula=_POTENTIAL_EXCESS_LOSS_FORMULA,
                    config_type=PotentialExcessLossConfig,
                    name="potential-excess loss",
                ),
            )
        raise ValueError("training stage name is not registered")


@dataclasses.dataclass(frozen=True)
class ResidualContract:
    """Exact residual feature definition and normalization."""

    definition: str
    normalization: str
    scale_source: str
    detach_features: bool

    def __post_init__(self) -> None:
        if self.definition != _RESIDUAL_DEFINITION:
            raise ValueError(f"residual definition must be exactly {_RESIDUAL_DEFINITION!r}")
        if self.normalization != _RESIDUAL_NORMALIZATION:
            raise ValueError(f"residual normalization must be exactly {_RESIDUAL_NORMALIZATION!r}")
        if self.scale_source != _RESIDUAL_SCALE_SOURCE:
            raise ValueError(f"residual scale_source must be exactly {_RESIDUAL_SCALE_SOURCE!r}")
        _require_bool(self.detach_features, "residual detach_features")

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResidualContract:
        """Reconstruct and validate a residual contract."""
        _strict_keys(value, {"definition", "normalization", "scale_source", "detach_features"}, "residual")
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class RepresentationContract:
    """Explicit bounded principal-log-stretch representation."""

    minimum_principal_stretch: float
    max_hencky_update: float
    max_rotation_update: float
    formula: str = _REPRESENTATION_FORMULA

    def __post_init__(self) -> None:
        if self.formula != _REPRESENTATION_FORMULA:
            raise ValueError(f"v5 representation formula must be exactly {_REPRESENTATION_FORMULA!r}")
        minimum = _require_finite_float(
            self.minimum_principal_stretch,
            "minimum_principal_stretch",
            strictly_positive=True,
        )
        if minimum != _MINIMUM_PRINCIPAL_STRETCH:
            raise ValueError(f"minimum_principal_stretch must be exactly {_MINIMUM_PRINCIPAL_STRETCH}")
        object.__setattr__(self, "minimum_principal_stretch", minimum)
        for name in ("max_hencky_update", "max_rotation_update"):
            object.__setattr__(
                self,
                name,
                _require_finite_float(getattr(self, name), name, strictly_positive=True),
            )

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return {
            "formula": self.formula,
            "minimum_principal_stretch": self.minimum_principal_stretch,
            "right_cauchy_green_eigenvalue_floor": self.minimum_principal_stretch**2,
            "max_hencky_update": self.max_hencky_update,
            "max_rotation_update": self.max_rotation_update,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RepresentationContract:
        """Reconstruct and validate a representation contract."""
        _strict_keys(
            value,
            {
                "formula",
                "minimum_principal_stretch",
                "right_cauchy_green_eigenvalue_floor",
                "max_hencky_update",
                "max_rotation_update",
            },
            "representation",
        )
        result = cls(
            minimum_principal_stretch=value["minimum_principal_stretch"],
            max_hencky_update=value["max_hencky_update"],
            max_rotation_update=value["max_rotation_update"],
            formula=value["formula"],
        )
        if value["right_cauchy_green_eigenvalue_floor"] != result.minimum_principal_stretch**2:
            raise ValueError("right-Cauchy-Green eigenvalue floor is inconsistent")
        return result


@dataclasses.dataclass(frozen=True)
class ProjectionContract:
    """Fixed compatibility projection backend and bounded work."""

    backend: str
    relative_tolerance: float | None
    absolute_tolerance: float | None
    max_iterations: int
    warm_start: str
    raise_on_nonconvergence: bool
    preconditioner: str
    require_runtime_diagnostics: bool = True
    execution_dtype: str = "torch.float64"
    operator_geometry_policy: str = OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE

    def __post_init__(self) -> None:
        if self.backend not in ("dense", "sparse_pcg"):
            raise ValueError("projection backend must be 'dense' or 'sparse_pcg'")
        if self.execution_dtype not in _V5_PARAMETER_DTYPES:
            raise ValueError("projection execution_dtype must be registered float32 or float64")
        if type(self.operator_geometry_policy) is not str or self.operator_geometry_policy not in (
            OPERATOR_GEOMETRY_POLICY_CANONICAL_REST_INVERSE,
            OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED,
        ):
            raise ValueError("projection operator_geometry_policy must be an authenticated v5 policy")
        if (
            self.operator_geometry_policy == OPERATOR_GEOMETRY_POLICY_SOURCE_TET_POSES_PROMOTED
            and self.execution_dtype != "torch.float64"
        ):
            raise ValueError("source-tet-poses-promoted requires torch.float64 projection execution")
        _require_bool(self.raise_on_nonconvergence, "projection raise_on_nonconvergence")
        if not _require_bool(self.require_runtime_diagnostics, "projection require_runtime_diagnostics"):
            raise ValueError("v5 runtime replay evidence must require projection diagnostics")
        if self.backend == "dense":
            if self.relative_tolerance is not None or self.absolute_tolerance is not None:
                raise ValueError("dense projection tolerances must be None")
            if self.max_iterations != 0:
                raise ValueError("dense projection max_iterations must be zero")
            if self.warm_start != "not-applicable" or self.preconditioner != "none":
                raise ValueError("dense projection must use no warm start or preconditioner")
            return
        relative = _require_finite_float(self.relative_tolerance, "projection relative_tolerance")
        absolute = _require_finite_float(self.absolute_tolerance, "projection absolute_tolerance")
        if relative <= 0.0 or relative > _MAX_PROJECTION_RELATIVE_TOLERANCE:
            raise ValueError(
                "sparse_pcg relative_tolerance must be positive and no greater than "
                f"{_MAX_PROJECTION_RELATIVE_TOLERANCE}"
            )
        if absolute != 0.0:
            raise ValueError("schema-v5 sparse_pcg absolute_tolerance must be exactly zero")
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "absolute_tolerance", absolute)
        _require_positive_integer(self.max_iterations, "projection max_iterations")
        if self.warm_start != "current-iterate":
            raise ValueError("sparse_pcg warm_start must be 'current-iterate'")
        if not self.raise_on_nonconvergence:
            raise ValueError("sparse_pcg must fail closed on nonconvergence")
        if self.preconditioner != "jacobi":
            raise ValueError("schema-v5 sparse_pcg currently authenticates only the exact Jacobi preconditioner")

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProjectionContract:
        """Reconstruct and validate a projection contract."""
        _strict_keys(
            value,
            {
                "backend",
                "relative_tolerance",
                "absolute_tolerance",
                "max_iterations",
                "warm_start",
                "raise_on_nonconvergence",
                "preconditioner",
                "require_runtime_diagnostics",
                "execution_dtype",
                "operator_geometry_policy",
            },
            "projection",
        )
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class ConstraintContract:
    """Canonical stateful constraint descriptor and iteration semantics."""

    descriptor_json: str
    prepare_cadence: str
    apply_cadence: str
    gradient_policy: str
    descriptor_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        descriptor = _decoded_canonical_json(self.descriptor_json, "constraint descriptor")
        if not isinstance(descriptor, Mapping):
            raise ValueError("constraint descriptor must decode to a mapping")
        if descriptor != _IDENTITY_CONSTRAINT_DESCRIPTOR:
            raise ValueError("only the exact registered identity constraint descriptor is currently supported")
        if self.prepare_cadence != _CONSTRAINT_PREPARE_CADENCE:
            raise ValueError(f"constraint prepare_cadence must be exactly {_CONSTRAINT_PREPARE_CADENCE!r}")
        if self.apply_cadence != _CONSTRAINT_APPLY_CADENCE:
            raise ValueError(f"constraint apply_cadence must be exactly {_CONSTRAINT_APPLY_CADENCE!r}")
        if self.gradient_policy != _CONSTRAINT_GRADIENT_POLICY:
            raise ValueError(f"constraint gradient_policy must be exactly {_CONSTRAINT_GRADIENT_POLICY!r}")
        object.__setattr__(self, "descriptor_sha256", canonical_json_sha256(descriptor))

    @classmethod
    def build(
        cls,
        descriptor: Mapping[str, object],
        *,
        prepare_cadence: str,
        apply_cadence: str,
        gradient_policy: str,
    ) -> ConstraintContract:
        """Build an immutable contract from finite JSON descriptor data."""
        return cls(
            descriptor_json=_canonical_json_text(descriptor),
            prepare_cadence=prepare_cadence,
            apply_cadence=apply_cadence,
            gradient_policy=gradient_policy,
        )

    @property
    def descriptor(self) -> dict[str, object]:
        """Return a fresh mutable copy of the authenticated descriptor."""
        decoded = _decoded_canonical_json(self.descriptor_json, "constraint descriptor")
        if not isinstance(decoded, dict):
            raise RuntimeError("validated constraint descriptor changed type")
        return decoded

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return {
            "descriptor": self.descriptor,
            "descriptor_sha256": self.descriptor_sha256,
            "prepare_cadence": self.prepare_cadence,
            "apply_cadence": self.apply_cadence,
            "gradient_policy": self.gradient_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ConstraintContract:
        """Reconstruct and verify a constraint contract."""
        _strict_keys(
            value,
            {"descriptor", "descriptor_sha256", "prepare_cadence", "apply_cadence", "gradient_policy"},
            "constraint",
        )
        descriptor = value["descriptor"]
        if not isinstance(descriptor, Mapping):
            raise ValueError("constraint descriptor must be a mapping")
        result = cls.build(
            descriptor,
            prepare_cadence=value["prepare_cadence"],
            apply_cadence=value["apply_cadence"],
            gradient_policy=value["gradient_policy"],
        )
        if result.descriptor_sha256 != value["descriptor_sha256"]:
            raise ValueError("constraint descriptor SHA-256 verification failed")
        return result


@dataclasses.dataclass(frozen=True)
class CorrectorContract:
    """Explicit classical corrector kind and fixed numerical budget."""

    kind: str
    iterations: int
    residual_operator_calls: int
    preconditioner_calls: int
    line_search_candidates: int

    def __post_init__(self) -> None:
        _require_string(self.kind, "corrector kind")
        for name in (
            "iterations",
            "residual_operator_calls",
            "preconditioner_calls",
            "line_search_candidates",
        ):
            _require_nonnegative_integer(getattr(self, name), f"corrector {name}")
        budgets = (
            self.iterations,
            self.residual_operator_calls,
            self.preconditioner_calls,
            self.line_search_candidates,
        )
        if self.kind != "identity":
            raise ValueError("only the registered identity corrector is currently supported")
        if any(budgets):
            raise ValueError("identity corrector must have an all-zero fixed budget")

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CorrectorContract:
        """Reconstruct and validate a corrector contract."""
        _strict_keys(
            value,
            {"kind", "iterations", "residual_operator_calls", "preconditioner_calls", "line_search_candidates"},
            "corrector",
        )
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class SafeguardContract:
    """Fail-closed endpoint policies; fallback is forbidden for learned v5."""

    minimum_determinant: float
    minimum_singular_value: float
    objective_policy: str
    residual_policy: str
    objective_increase_tolerance: float
    normalized_residual_increase_tolerance: float
    replay_relative_tolerance: float
    replay_absolute_tolerance: float
    initializer_policy: str
    require_finite: bool = True
    exact_dirichlet: bool = True
    invalid_state_policy: str = "reject"
    fallback: str = _FALLBACK_POLICY

    def __post_init__(self) -> None:
        determinant = _require_finite_float(self.minimum_determinant, "minimum_determinant")
        if determinant < 0.0:
            raise ValueError("minimum_determinant must be non-negative")
        object.__setattr__(self, "minimum_determinant", determinant)
        singular = _require_finite_float(self.minimum_singular_value, "minimum_singular_value")
        if singular < 0.0:
            raise ValueError("minimum_singular_value must be non-negative")
        object.__setattr__(self, "minimum_singular_value", singular)
        if self.objective_policy != _OBJECTIVE_POLICY:
            raise ValueError(f"objective_policy must be exactly {_OBJECTIVE_POLICY!r}")
        if self.residual_policy != _RESIDUAL_POLICY:
            raise ValueError(f"residual_policy must be exactly {_RESIDUAL_POLICY!r}")
        for name in (
            "objective_increase_tolerance",
            "normalized_residual_increase_tolerance",
            "replay_relative_tolerance",
            "replay_absolute_tolerance",
        ):
            tolerance = _require_finite_float(getattr(self, name), name)
            if tolerance < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, tolerance)
        if self.objective_increase_tolerance > _MAX_OBJECTIVE_INCREASE_TOLERANCE:
            raise ValueError(
                "objective_increase_tolerance exceeds the registered evidence maximum "
                f"{_MAX_OBJECTIVE_INCREASE_TOLERANCE}"
            )
        if self.normalized_residual_increase_tolerance > _MAX_NORMALIZED_RESIDUAL_INCREASE_TOLERANCE:
            raise ValueError(
                "normalized_residual_increase_tolerance exceeds the registered evidence maximum "
                f"{_MAX_NORMALIZED_RESIDUAL_INCREASE_TOLERANCE}"
            )
        maximum_replay_relative = max(value[0] for value in _REPLAY_TOLERANCES.values())
        maximum_replay_absolute = max(value[1] for value in _REPLAY_TOLERANCES.values())
        if self.replay_relative_tolerance > maximum_replay_relative:
            raise ValueError("replay_relative_tolerance exceeds the registered evidence maximum")
        if self.replay_absolute_tolerance > maximum_replay_absolute:
            raise ValueError("replay_absolute_tolerance exceeds the registered evidence maximum")
        if self.initializer_policy != _INITIALIZER_POLICY:
            raise ValueError(f"initializer_policy must be exactly {_INITIALIZER_POLICY!r}")
        if not _require_bool(self.require_finite, "require_finite"):
            raise ValueError("v5 safeguards must require finite states")
        if not _require_bool(self.exact_dirichlet, "exact_dirichlet"):
            raise ValueError("v5 safeguards must require exact Dirichlet constraints")
        if self.invalid_state_policy != "reject":
            raise ValueError("v5 invalid_state_policy must be 'reject'")
        if self.fallback != _FALLBACK_POLICY:
            raise ValueError("v5 checkpoint fallback must be exactly 'none'")

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SafeguardContract:
        """Reconstruct and validate a safeguard contract."""
        _strict_keys(
            value,
            {
                "minimum_determinant",
                "minimum_singular_value",
                "objective_policy",
                "residual_policy",
                "objective_increase_tolerance",
                "normalized_residual_increase_tolerance",
                "replay_relative_tolerance",
                "replay_absolute_tolerance",
                "initializer_policy",
                "require_finite",
                "exact_dirichlet",
                "invalid_state_policy",
                "fallback",
            },
            "safeguards",
        )
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class ProposalSafeguardContract:
    """Authenticated fixed-candidate inference globalization semantics."""

    candidate_step_fractions: tuple[float, ...]
    policy: str = "fixed-constrained-backtracking-v1"
    interpolation_policy: str = "current-to-projected-position-segment"
    selection_policy: str = "first-admissible-positive-else-zero"
    zero_policy: str = "exact-no-op"
    candidate_state_policy: str = "same-prepared-state-selected-successor"

    def __post_init__(self) -> None:
        for name in (
            "policy",
            "interpolation_policy",
            "selection_policy",
            "zero_policy",
            "candidate_state_policy",
        ):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"proposal safeguard {name} must be a built-in string")
        try:
            config = ProposalSafeguardConfig(
                candidate_step_fractions=self.candidate_step_fractions,
                policy=self.policy,
                interpolation_policy=self.interpolation_policy,
                selection_policy=self.selection_policy,
                zero_policy=self.zero_policy,
                candidate_state_policy=self.candidate_state_policy,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("proposal safeguard does not match the registered core policy") from error
        object.__setattr__(self, "candidate_step_fractions", config.candidate_step_fractions)

    @classmethod
    def from_config(cls, config: ProposalSafeguardConfig) -> ProposalSafeguardContract:
        """Bind an exact registered runtime proposal-safeguard config."""
        if type(config) is not ProposalSafeguardConfig:
            raise TypeError("config must be the exact ProposalSafeguardConfig type")
        return cls(**dataclasses.asdict(config))

    def as_config(self) -> ProposalSafeguardConfig:
        """Reconstruct the exact registered runtime configuration."""
        return ProposalSafeguardConfig(
            candidate_step_fractions=self.candidate_step_fractions,
            policy=self.policy,
            interpolation_policy=self.interpolation_policy,
            selection_policy=self.selection_policy,
            zero_policy=self.zero_policy,
            candidate_state_policy=self.candidate_state_policy,
        )

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return {
            "policy": self.policy,
            "candidate_step_fractions": list(self.candidate_step_fractions),
            "interpolation_policy": self.interpolation_policy,
            "selection_policy": self.selection_policy,
            "zero_policy": self.zero_policy,
            "candidate_state_policy": self.candidate_state_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProposalSafeguardContract:
        """Reconstruct and validate a proposal-safeguard contract."""
        _strict_keys(
            value,
            {
                "policy",
                "candidate_step_fractions",
                "interpolation_policy",
                "selection_policy",
                "zero_policy",
                "candidate_state_policy",
            },
            "proposal safeguard",
        )
        fractions = value["candidate_step_fractions"]
        if not isinstance(fractions, list):
            raise TypeError("proposal safeguard candidate_step_fractions must be a JSON list")
        result = cls(
            candidate_step_fractions=tuple(fractions),
            policy=value["policy"],
            interpolation_policy=value["interpolation_policy"],
            selection_policy=value["selection_policy"],
            zero_policy=value["zero_policy"],
            candidate_state_policy=value["candidate_state_policy"],
        )
        if result.as_dict() != value:
            raise ValueError("proposal safeguard is not in canonical serialized form")
        return result


@dataclasses.dataclass(frozen=True)
class OptimizerContract:
    """Optimizer policy metadata, not proof that an opaque state can resume."""

    kind: str
    hyperparameters_json: str
    hyperparameters_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _require_string(self.kind, "optimizer kind")
        hyperparameters = _decoded_canonical_json(self.hyperparameters_json, "optimizer hyperparameters")
        if not isinstance(hyperparameters, Mapping):
            raise ValueError("optimizer hyperparameters must decode to a mapping")
        object.__setattr__(self, "hyperparameters_sha256", canonical_json_sha256(hyperparameters))

    @classmethod
    def build(cls, kind: str, hyperparameters: Mapping[str, object]) -> OptimizerContract:
        """Build an immutable optimizer contract."""
        return cls(kind=kind, hyperparameters_json=_canonical_json_text(hyperparameters))

    @property
    def hyperparameters(self) -> dict[str, object]:
        """Return a fresh copy of the authenticated hyperparameters."""
        decoded = _decoded_canonical_json(self.hyperparameters_json, "optimizer hyperparameters")
        if not isinstance(decoded, dict):
            raise RuntimeError("validated optimizer hyperparameters changed type")
        return decoded

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return {
            "kind": self.kind,
            "hyperparameters": self.hyperparameters,
            "hyperparameters_sha256": self.hyperparameters_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OptimizerContract:
        """Reconstruct and verify an optimizer contract."""
        _strict_keys(value, {"kind", "hyperparameters", "hyperparameters_sha256"}, "optimizer")
        hyperparameters = value["hyperparameters"]
        if not isinstance(hyperparameters, Mapping):
            raise ValueError("optimizer hyperparameters must be a mapping")
        result = cls.build(value["kind"], hyperparameters)
        if result.hyperparameters_sha256 != value["hyperparameters_sha256"]:
            raise ValueError("optimizer hyperparameter SHA-256 verification failed")
        return result


@dataclasses.dataclass(frozen=True)
class ParentLineage:
    """Root or externally pinned continuation lineage."""

    kind: str
    parent_checkpoint_payload_sha256: str | None
    parent_learned_state_sha256: str | None
    parent_completed_updates: int | None

    def __post_init__(self) -> None:
        if self.kind == "root":
            if any(
                value is not None
                for value in (
                    self.parent_checkpoint_payload_sha256,
                    self.parent_learned_state_sha256,
                    self.parent_completed_updates,
                )
            ):
                raise ValueError("root lineage must not name a parent")
            return
        if self.kind != "continuation":
            raise ValueError("parent lineage kind must be 'root' or 'continuation'")
        _require_sha256(self.parent_checkpoint_payload_sha256, "parent checkpoint payload sha256")
        _require_sha256(self.parent_learned_state_sha256, "parent learned state sha256")
        _require_nonnegative_integer(self.parent_completed_updates, "parent completed_updates")

    @classmethod
    def root(cls) -> ParentLineage:
        """Return an explicit no-parent lineage."""
        return cls("root", None, None, None)

    @classmethod
    def continuation(
        cls,
        *,
        parent_checkpoint_payload_sha256: str,
        parent_learned_state_sha256: str,
        parent_completed_updates: int,
    ) -> ParentLineage:
        """Return externally pinned continuation lineage metadata."""
        return cls(
            "continuation",
            parent_checkpoint_payload_sha256,
            parent_learned_state_sha256,
            parent_completed_updates,
        )

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ParentLineage:
        """Reconstruct and validate lineage metadata."""
        _strict_keys(
            value,
            {
                "kind",
                "parent_checkpoint_payload_sha256",
                "parent_learned_state_sha256",
                "parent_completed_updates",
            },
            "parent lineage",
        )
        return cls(**dict(value))


def _verify_split_manifest(value: object) -> SplitManifest:
    if type(value) is not SplitManifest:
        raise ValueError("training_split must be a canonical SplitManifest")
    payload = value.as_dict()
    declared = payload.pop("manifest_sha256")
    if canonical_json_sha256(payload) != declared or declared != value.manifest_sha256:
        raise ValueError("training SplitManifest SHA-256 verification failed")
    return value


def _verify_sampling_schedule(
    value: object,
    *,
    manifest_sha256: str,
    expected_steps: int,
) -> SamplingSchedule:
    if type(value) is not SamplingSchedule:
        raise ValueError("sampling_schedule must be a canonical SamplingSchedule")
    payload = value.as_dict()
    declared = payload.pop("schedule_sha256")
    if canonical_json_sha256(payload) != declared or declared != value.schedule_sha256:
        raise ValueError("training SamplingSchedule SHA-256 verification failed")
    if value.manifest_sha256 != manifest_sha256 or value.role is not DatasetRole.TRAIN:
        raise ValueError("training SamplingSchedule must bind the checkpoint train split")
    if value.steps != expected_steps:
        raise ValueError("training SamplingSchedule steps must equal the ordered stage-plan endpoint")
    return value


def _graph_config_payload(graph_config_json: str) -> dict[str, object]:
    graph = _decoded_canonical_json(graph_config_json, "graph config")
    if not isinstance(graph, dict):
        raise ValueError("graph config must decode to a mapping")
    return graph


@dataclasses.dataclass(frozen=True)
class V5SolverContract:
    """Immutable architecture, data, training, and fixed-work solver contract."""

    graph_config_json: str
    learned_parameter_dtype: str
    training_dataset_kind: str
    training_dataset_sha256: str
    sampling_schedule_sha256: str
    sampling_steps: int
    sampling_batch_size: int
    sampling_seed: int
    stages: tuple[TrainingStage, ...]
    trained_iterations: int
    inference_iterations: int
    residual: ResidualContract
    representation: RepresentationContract
    projection: ProjectionContract
    constraint: ConstraintContract
    corrector: CorrectorContract
    safeguards: SafeguardContract
    optimizer: OptimizerContract
    physical_timestep_source: str
    rng_algorithm: str
    batch_stream_contract: str
    solver_contract_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        for name, expected_type in (
            ("residual", ResidualContract),
            ("representation", RepresentationContract),
            ("projection", ProjectionContract),
            ("constraint", ConstraintContract),
            ("corrector", CorrectorContract),
            ("safeguards", SafeguardContract),
            ("optimizer", OptimizerContract),
        ):
            if type(getattr(self, name)) is not expected_type:
                raise TypeError(f"{name} must be a canonical {expected_type.__name__}")
        graph = _canonical_v5_graph_config(_graph_config_payload(self.graph_config_json))
        if self.learned_parameter_dtype not in _V5_PARAMETER_DTYPES:
            raise ValueError("learned_parameter_dtype must be registered for architecture v5")
        learned_dtype = _V5_PARAMETER_DTYPES[self.learned_parameter_dtype]
        for name, value in (
            ("graph max_hencky_update", graph["max_hencky_update"]),
            ("graph max_rotation_update", graph["max_rotation_update"]),
            ("graph legacy dt", graph["dt"]),
            ("minimum_principal_stretch", self.representation.minimum_principal_stretch),
            ("representation max_hencky_update", self.representation.max_hencky_update),
            ("representation max_rotation_update", self.representation.max_rotation_update),
        ):
            materialized = torch.as_tensor(value, dtype=learned_dtype)
            if not torch.isfinite(materialized) or materialized <= 0.0:
                raise ValueError(f"{name} must remain finite and positive in learned_parameter_dtype")
        expected_replay_tolerances = _REPLAY_TOLERANCES[self.learned_parameter_dtype]
        observed_replay_tolerances = (
            self.safeguards.replay_relative_tolerance,
            self.safeguards.replay_absolute_tolerance,
        )
        if observed_replay_tolerances != expected_replay_tolerances:
            raise ValueError(
                "safeguard replay tolerances must exactly match the registered learned-parameter dtype policy"
            )
        if graph["max_hencky_update"] != self.representation.max_hencky_update:
            raise ValueError("graph and representation max_hencky_update differ")
        if graph["max_rotation_update"] != self.representation.max_rotation_update:
            raise ValueError("graph and representation max_rotation_update differ")

        if self.training_dataset_kind != "pss-v5-split-manifest-sha256":
            raise ValueError("training dataset identity must be a pss-v5 SplitManifest SHA-256")
        _require_sha256(self.training_dataset_sha256, "training dataset sha256")
        _require_sha256(self.sampling_schedule_sha256, "training sampling schedule sha256")
        _require_positive_integer(self.sampling_steps, "sampling_steps")
        _require_positive_integer(self.sampling_batch_size, "sampling_batch_size")
        _require_nonnegative_integer(self.sampling_seed, "sampling_seed")
        stages = tuple(self.stages)
        if not stages or any(not isinstance(stage, TrainingStage) for stage in stages):
            raise ValueError("stages must contain canonical TrainingStage values")
        if stages[0].start_update != 0:
            raise ValueError("the ordered training stage plan must begin at update zero")
        if any(left.end_update != right.start_update for left, right in itertools.pairwise(stages)):
            raise ValueError("the ordered training stage plan must be contiguous")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("training stage names must be unique")
        if tuple(stage.name for stage in stages) != ("representation", "physics"):
            raise ValueError("ordered training stage plan must be exactly representation then physics")
        if stages[-1].end_update != self.sampling_steps:
            raise ValueError("ordered training stage plan must cover the exact sampling schedule")
        representation_stage = stages[0]
        assert representation_stage.label_config is not None
        assert representation_stage.representation_loss_config is not None
        if (
            representation_stage.label_config.minimum_principal_stretch != self.representation.minimum_principal_stretch
            or representation_stage.label_config.max_hencky_update != self.representation.max_hencky_update
            or representation_stage.label_config.max_rotation_update != self.representation.max_rotation_update
            or representation_stage.representation_loss_config.max_hencky_update
            != self.representation.max_hencky_update
            or representation_stage.representation_loss_config.max_rotation_update
            != self.representation.max_rotation_update
        ):
            raise ValueError("training label/loss representation differs from the solver representation")
        object.__setattr__(self, "stages", stages)
        _require_positive_integer(self.trained_iterations, "trained_iterations")
        _require_positive_integer(self.inference_iterations, "inference_iterations")
        if self.physical_timestep_source != _PHYSICAL_TIMESTEP_SOURCE:
            raise ValueError(f"physical_timestep_source must be exactly {_PHYSICAL_TIMESTEP_SOURCE!r}")
        if self.rng_algorithm != _RNG_ALGORITHM:
            raise ValueError(f"rng_algorithm must be exactly {_RNG_ALGORITHM!r}")
        if self.batch_stream_contract != _BATCH_STREAM_CONTRACT:
            raise ValueError(f"batch_stream_contract must be exactly {_BATCH_STREAM_CONTRACT!r}")
        object.__setattr__(self, "solver_contract_sha256", canonical_json_sha256(self._payload()))

    @classmethod
    def build(
        cls,
        *,
        graph_config: Mapping[str, object],
        learned_parameter_dtype: str,
        training_split: object,
        sampling_schedule: object,
        stages: Sequence[TrainingStage],
        trained_iterations: int,
        inference_iterations: int,
        residual: ResidualContract,
        representation: RepresentationContract,
        projection: ProjectionContract,
        constraint: ConstraintContract,
        corrector: CorrectorContract,
        safeguards: SafeguardContract,
        optimizer: OptimizerContract,
        physical_timestep_source: str,
        rng_algorithm: str,
        batch_stream_contract: str,
    ) -> V5SolverContract:
        """Build the canonical contract from a split object or JSON payload."""
        manifest = _verify_split_manifest(training_split)
        canonical_stages = tuple(stages)
        if not canonical_stages:
            raise ValueError("stages must not be empty")
        schedule = _verify_sampling_schedule(
            sampling_schedule,
            manifest_sha256=manifest.manifest_sha256,
            expected_steps=canonical_stages[-1].end_update,
        )
        return cls(
            graph_config_json=_canonical_json_text(_canonical_v5_graph_config(graph_config)),
            learned_parameter_dtype=learned_parameter_dtype,
            training_dataset_kind="pss-v5-split-manifest-sha256",
            training_dataset_sha256=manifest.manifest_sha256,
            sampling_schedule_sha256=schedule.schedule_sha256,
            sampling_steps=schedule.steps,
            sampling_batch_size=schedule.batch_size,
            sampling_seed=schedule.seed,
            stages=canonical_stages,
            trained_iterations=trained_iterations,
            inference_iterations=inference_iterations,
            residual=residual,
            representation=representation,
            projection=projection,
            constraint=constraint,
            corrector=corrector,
            safeguards=safeguards,
            optimizer=optimizer,
            physical_timestep_source=physical_timestep_source,
            rng_algorithm=rng_algorithm,
            batch_stream_contract=batch_stream_contract,
        )

    @property
    def graph_config(self) -> dict[str, object]:
        """Return a fresh copy of the authenticated graph configuration."""
        return _graph_config_payload(self.graph_config_json)

    def _work(self, iterations: int) -> dict[str, int]:
        return {
            "predictor_passes": iterations,
            "global_compatibility_projections": iterations,
            "maximum_projection_iterations": iterations * self.projection.max_iterations,
            "common_residual_evaluations": iterations + 1 + self.corrector.residual_operator_calls,
            "common_objective_evaluations": iterations + 1,
            "state_validity_evaluations": iterations + 1,
            "physical_step_authentications": 2 * iterations + 3,
            "common_objective_authentications": 2 * iterations + 3,
            "constraint_preparations": iterations,
            "constraint_applications": iterations,
            "corrector_iterations": self.corrector.iterations,
            "corrector_residual_operator_calls": self.corrector.residual_operator_calls,
            "corrector_preconditioner_calls": self.corrector.preconditioner_calls,
            "corrector_line_search_candidates": self.corrector.line_search_candidates,
        }

    @property
    def trained_work(self) -> dict[str, int]:
        """Return exact/upper-bound work authenticated for training unrolls."""
        return self._work(self.trained_iterations)

    @property
    def inference_work(self) -> dict[str, int]:
        """Return exact/upper-bound work authenticated for inference."""
        return self._work(self.inference_iterations)

    def _payload(self) -> dict[str, object]:
        graph = self.graph_config
        return {
            "schema_version": _SCHEMA_VERSION,
            "contract": _SOLVER_CONTRACT,
            "model_semantics": _MODEL_SEMANTICS,
            "graph": {
                "architecture_version": 5,
                "config": graph,
                "config_sha256": canonical_json_sha256(graph),
                "learned_parameter_dtype": self.learned_parameter_dtype,
                "legacy_dt_policy": _LEGACY_GRAPH_DT_POLICY,
            },
            "training_dataset": {
                "kind": self.training_dataset_kind,
                "sha256": self.training_dataset_sha256,
            },
            "sampling_schedule": {
                "schedule_sha256": self.sampling_schedule_sha256,
                "steps": self.sampling_steps,
                "batch_size": self.sampling_batch_size,
                "seed": self.sampling_seed,
            },
            "ordered_stage_plan": [stage.as_dict() for stage in self.stages],
            "trained_iterations": self.trained_iterations,
            "inference_iterations": self.inference_iterations,
            "residual": self.residual.as_dict(),
            "representation": self.representation.as_dict(),
            "projection": self.projection.as_dict(),
            "constraint": self.constraint.as_dict(),
            "corrector": self.corrector.as_dict(),
            "safeguards": self.safeguards.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "physical_timestep": {
                "source": self.physical_timestep_source,
                "legacy_graph_dt_policy": _LEGACY_GRAPH_DT_POLICY,
            },
            "rng_algorithm": self.rng_algorithm,
            "batch_stream_contract": self.batch_stream_contract,
            "trained_work": self.trained_work,
            "inference_work": self.inference_work,
        }

    def as_dict(self) -> dict[str, object]:
        """Return the self-checking canonical solver contract."""
        result = self._payload()
        result["solver_contract_sha256"] = self.solver_contract_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> V5SolverContract:
        """Reconstruct and fully verify a serialized solver contract."""
        canonical = _jsonable(value)
        if not isinstance(canonical, dict):
            raise ValueError("solver contract must be a mapping")
        digest = canonical.pop("solver_contract_sha256", None)
        if digest != canonical_json_sha256(canonical):
            raise ValueError("solver-contract SHA-256 verification failed")
        expected = {
            "schema_version",
            "contract",
            "model_semantics",
            "graph",
            "training_dataset",
            "sampling_schedule",
            "ordered_stage_plan",
            "trained_iterations",
            "inference_iterations",
            "residual",
            "representation",
            "projection",
            "constraint",
            "corrector",
            "safeguards",
            "optimizer",
            "physical_timestep",
            "rng_algorithm",
            "batch_stream_contract",
            "trained_work",
            "inference_work",
        }
        _strict_keys(canonical, expected, "solver contract")
        if canonical["schema_version"] != _SCHEMA_VERSION or canonical["contract"] != _SOLVER_CONTRACT:
            raise ValueError("solver contract is not schema v5")
        if canonical["model_semantics"] != _MODEL_SEMANTICS:
            raise ValueError("solver contract does not describe the v5 learned critical path")
        graph = canonical["graph"]
        dataset = canonical["training_dataset"]
        sampling = canonical["sampling_schedule"]
        timestep = canonical["physical_timestep"]
        if (
            not isinstance(graph, Mapping)
            or not isinstance(dataset, Mapping)
            or not isinstance(sampling, Mapping)
            or not isinstance(timestep, Mapping)
        ):
            raise ValueError("solver contract has malformed graph, dataset, or timestep metadata")
        _strict_keys(
            graph,
            {
                "architecture_version",
                "config",
                "config_sha256",
                "learned_parameter_dtype",
                "legacy_dt_policy",
            },
            "solver graph",
        )
        if graph["architecture_version"] != 5 or graph["legacy_dt_policy"] != _LEGACY_GRAPH_DT_POLICY:
            raise ValueError("solver graph is not canonical architecture v5")
        if not isinstance(graph["config"], Mapping):
            raise ValueError("solver graph config must be a mapping")
        if graph["config_sha256"] != canonical_json_sha256(graph["config"]):
            raise ValueError("solver graph-config SHA-256 verification failed")
        _strict_keys(dataset, {"kind", "sha256"}, "training dataset identity")
        _strict_keys(sampling, {"schedule_sha256", "steps", "batch_size", "seed"}, "sampling schedule identity")
        _strict_keys(timestep, {"source", "legacy_graph_dt_policy"}, "physical timestep")
        if timestep["legacy_graph_dt_policy"] != _LEGACY_GRAPH_DT_POLICY:
            raise ValueError("physical timestep legacy graph-dt policy changed")
        stages = canonical["ordered_stage_plan"]
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
            raise ValueError("ordered_stage_plan must be a sequence")
        for name in ("residual", "representation", "projection", "constraint", "corrector", "safeguards", "optimizer"):
            if not isinstance(canonical[name], Mapping):
                raise ValueError(f"solver contract {name} must be a mapping")
        result = cls(
            graph_config_json=_canonical_json_text(graph["config"]),
            learned_parameter_dtype=graph["learned_parameter_dtype"],
            training_dataset_kind=dataset["kind"],
            training_dataset_sha256=dataset["sha256"],
            sampling_schedule_sha256=sampling["schedule_sha256"],
            sampling_steps=sampling["steps"],
            sampling_batch_size=sampling["batch_size"],
            sampling_seed=sampling["seed"],
            stages=tuple(TrainingStage.from_dict(stage) for stage in stages),
            trained_iterations=canonical["trained_iterations"],
            inference_iterations=canonical["inference_iterations"],
            residual=ResidualContract.from_dict(canonical["residual"]),
            representation=RepresentationContract.from_dict(canonical["representation"]),
            projection=ProjectionContract.from_dict(canonical["projection"]),
            constraint=ConstraintContract.from_dict(canonical["constraint"]),
            corrector=CorrectorContract.from_dict(canonical["corrector"]),
            safeguards=SafeguardContract.from_dict(canonical["safeguards"]),
            optimizer=OptimizerContract.from_dict(canonical["optimizer"]),
            physical_timestep_source=timestep["source"],
            rng_algorithm=canonical["rng_algorithm"],
            batch_stream_contract=canonical["batch_stream_contract"],
        )
        if result.as_dict() != value:
            raise ValueError("solver contract is not in canonical serialized form")
        return result


def _validate_v6_contract_execution_dtype(
    proposal: ProposalSafeguardContract,
    safeguards: SafeguardContract,
    execution_dtype: str,
) -> None:
    """Reject schema-6 scalars that change meaning in the execution dtype."""
    dtype = _V5_PARAMETER_DTYPES[execution_dtype]
    reference = torch.empty((), dtype=dtype)
    for name in (
        "minimum_determinant",
        "minimum_singular_value",
        "objective_increase_tolerance",
        "normalized_residual_increase_tolerance",
    ):
        python_value = getattr(safeguards, name)
        materialized = reference.new_tensor(python_value)
        if not bool(torch.isfinite(materialized).item()):
            raise ValueError(f"{name} must remain finite in execution dtype {dtype}")
        if python_value > 0.0 and not bool((materialized > 0.0).item()):
            raise ValueError(f"{name} must remain positive in execution dtype {dtype}")
    fractions = reference.new_tensor(proposal.candidate_step_fractions)
    if not bool(torch.isfinite(fractions).all().item()):
        raise ValueError(f"candidate_step_fractions must remain finite in execution dtype {dtype}")
    if not bool((fractions[0] == reference.new_tensor(1.0)).item()) or not bool(
        (fractions[-1] == reference.new_tensor(0.0)).item()
    ):
        raise ValueError(f"candidate_step_fractions endpoints changed in execution dtype {dtype}")
    if not bool((fractions[:-1] > fractions[1:]).all().item()):
        raise ValueError(
            f"candidate_step_fractions must remain unique and strictly descending in execution dtype {dtype}"
        )


@dataclasses.dataclass(frozen=True)
class V6SolverContract:
    """Architecture-v5 training plus fixed-candidate inference contract."""

    graph_config_json: str
    learned_parameter_dtype: str
    training_dataset_kind: str
    training_dataset_sha256: str
    sampling_schedule_sha256: str
    sampling_steps: int
    sampling_batch_size: int
    sampling_seed: int
    stages: tuple[TrainingStage, ...]
    trained_iterations: int
    inference_iterations: int
    residual: ResidualContract
    representation: RepresentationContract
    projection: ProjectionContract
    constraint: ConstraintContract
    corrector: CorrectorContract
    safeguards: SafeguardContract
    optimizer: OptimizerContract
    physical_timestep_source: str
    rng_algorithm: str
    batch_stream_contract: str
    training_proposal_policy: str
    inference_proposal_safeguard: ProposalSafeguardContract
    solver_contract_sha256: str = dataclasses.field(init=False)
    _direct_contract: V5SolverContract = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.training_proposal_policy) is not str or self.training_proposal_policy != (
            _TRAINING_PROPOSAL_POLICY
        ):
            raise ValueError(f"training_proposal_policy must be exactly {_TRAINING_PROPOSAL_POLICY!r}")
        if type(self.inference_proposal_safeguard) is not ProposalSafeguardContract:
            raise TypeError("inference_proposal_safeguard must be a canonical ProposalSafeguardContract")
        direct = V5SolverContract(
            graph_config_json=self.graph_config_json,
            learned_parameter_dtype=self.learned_parameter_dtype,
            training_dataset_kind=self.training_dataset_kind,
            training_dataset_sha256=self.training_dataset_sha256,
            sampling_schedule_sha256=self.sampling_schedule_sha256,
            sampling_steps=self.sampling_steps,
            sampling_batch_size=self.sampling_batch_size,
            sampling_seed=self.sampling_seed,
            stages=self.stages,
            trained_iterations=self.trained_iterations,
            inference_iterations=self.inference_iterations,
            residual=self.residual,
            representation=self.representation,
            projection=self.projection,
            constraint=self.constraint,
            corrector=self.corrector,
            safeguards=self.safeguards,
            optimizer=self.optimizer,
            physical_timestep_source=self.physical_timestep_source,
            rng_algorithm=self.rng_algorithm,
            batch_stream_contract=self.batch_stream_contract,
        )
        _validate_v6_contract_execution_dtype(
            self.inference_proposal_safeguard,
            direct.safeguards,
            direct.projection.execution_dtype,
        )
        object.__setattr__(self, "stages", direct.stages)
        object.__setattr__(self, "_direct_contract", direct)
        object.__setattr__(self, "solver_contract_sha256", canonical_json_sha256(self._payload()))

    @classmethod
    def _from_direct(
        cls,
        direct: V5SolverContract,
        *,
        training_proposal_policy: str,
        inference_proposal_safeguard: ProposalSafeguardContract,
    ) -> V6SolverContract:
        fields = {
            field.name: getattr(direct, field.name) for field in dataclasses.fields(V5SolverContract) if field.init
        }
        return cls(
            **fields,
            training_proposal_policy=training_proposal_policy,
            inference_proposal_safeguard=inference_proposal_safeguard,
        )

    @classmethod
    def build(
        cls,
        *,
        graph_config: Mapping[str, object],
        learned_parameter_dtype: str,
        training_split: object,
        sampling_schedule: object,
        stages: Sequence[TrainingStage],
        trained_iterations: int,
        inference_iterations: int,
        residual: ResidualContract,
        representation: RepresentationContract,
        projection: ProjectionContract,
        constraint: ConstraintContract,
        corrector: CorrectorContract,
        safeguards: SafeguardContract,
        optimizer: OptimizerContract,
        physical_timestep_source: str,
        rng_algorithm: str,
        batch_stream_contract: str,
        training_proposal_policy: str,
        inference_proposal_safeguard: ProposalSafeguardContract,
    ) -> V6SolverContract:
        """Build a canonical schema-6 contract from verified v5 foundations."""
        direct = V5SolverContract.build(
            graph_config=graph_config,
            learned_parameter_dtype=learned_parameter_dtype,
            training_split=training_split,
            sampling_schedule=sampling_schedule,
            stages=stages,
            trained_iterations=trained_iterations,
            inference_iterations=inference_iterations,
            residual=residual,
            representation=representation,
            projection=projection,
            constraint=constraint,
            corrector=corrector,
            safeguards=safeguards,
            optimizer=optimizer,
            physical_timestep_source=physical_timestep_source,
            rng_algorithm=rng_algorithm,
            batch_stream_contract=batch_stream_contract,
        )
        return cls._from_direct(
            direct,
            training_proposal_policy=training_proposal_policy,
            inference_proposal_safeguard=inference_proposal_safeguard,
        )

    @property
    def graph_config(self) -> dict[str, object]:
        """Return a fresh copy of the authenticated graph configuration."""
        return self._direct_contract.graph_config

    @property
    def trained_work(self) -> dict[str, int]:
        """Return unchanged direct-record work for differentiable training."""
        return self._direct_contract.trained_work

    @property
    def inference_work(self) -> dict[str, int]:
        """Return exact fixed-candidate work for inference."""
        iterations = self.inference_iterations
        candidates = len(self.inference_proposal_safeguard.candidate_step_fractions)
        work = self._direct_contract._work(iterations)
        work.update(
            {
                "common_residual_evaluations": iterations * candidates + 1,
                "common_objective_evaluations": iterations * candidates + 1,
                "state_validity_evaluations": iterations * candidates + 1,
                "physical_step_authentications": iterations * (candidates + 1) + 3,
                "common_objective_authentications": iterations * (candidates + 1) + 3,
                "constraint_applications": iterations * candidates,
            }
        )
        return work

    def _payload(self) -> dict[str, object]:
        payload = self._direct_contract._payload()
        payload.update(
            {
                "schema_version": _SCHEMA_V6_VERSION,
                "contract": _SOLVER_V6_CONTRACT,
                "model_semantics": _MODEL_V6_SEMANTICS,
                "training_proposal_policy": self.training_proposal_policy,
                "inference_proposal_safeguard": self.inference_proposal_safeguard.as_dict(),
                "trained_work": self.trained_work,
                "inference_work": self.inference_work,
            }
        )
        return payload

    def as_dict(self) -> dict[str, object]:
        """Return the self-checking canonical schema-6 solver contract."""
        result = self._payload()
        result["solver_contract_sha256"] = self.solver_contract_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> V6SolverContract:
        """Reconstruct and fully verify a schema-6 solver contract."""
        canonical = _jsonable(value)
        if not isinstance(canonical, dict):
            raise ValueError("schema-6 solver contract must be a mapping")
        digest = canonical.pop("solver_contract_sha256", None)
        if digest != canonical_json_sha256(canonical):
            raise ValueError("schema-6 solver-contract SHA-256 verification failed")
        expected = {
            "schema_version",
            "contract",
            "model_semantics",
            "graph",
            "training_dataset",
            "sampling_schedule",
            "ordered_stage_plan",
            "trained_iterations",
            "inference_iterations",
            "residual",
            "representation",
            "projection",
            "constraint",
            "corrector",
            "safeguards",
            "optimizer",
            "physical_timestep",
            "rng_algorithm",
            "batch_stream_contract",
            "training_proposal_policy",
            "inference_proposal_safeguard",
            "trained_work",
            "inference_work",
        }
        _strict_keys(canonical, expected, "schema-6 solver contract")
        if canonical["schema_version"] != _SCHEMA_V6_VERSION or canonical["contract"] != _SOLVER_V6_CONTRACT:
            raise ValueError("solver contract is not schema v6")
        if canonical["model_semantics"] != _MODEL_V6_SEMANTICS:
            raise ValueError("schema-6 solver contract does not bind fixed-candidate inference")
        proposal_payload = canonical["inference_proposal_safeguard"]
        if not isinstance(proposal_payload, Mapping):
            raise ValueError("schema-6 inference proposal safeguard must be a mapping")
        proposal = ProposalSafeguardContract.from_dict(proposal_payload)

        direct_payload = dict(canonical)
        training_policy = direct_payload.pop("training_proposal_policy")
        direct_payload.pop("inference_proposal_safeguard")
        direct_payload.update(
            {
                "schema_version": _SCHEMA_VERSION,
                "contract": _SOLVER_CONTRACT,
                "model_semantics": _MODEL_SEMANTICS,
            }
        )
        iterations = direct_payload["inference_iterations"]
        projection = direct_payload["projection"]
        corrector = direct_payload["corrector"]
        if not isinstance(projection, Mapping) or not isinstance(corrector, Mapping):
            raise ValueError("schema-6 solver work inputs are malformed")
        iterations = _require_positive_integer(iterations, "schema-6 inference_iterations")
        maximum_projection_iterations = _require_nonnegative_integer(
            projection.get("max_iterations"), "schema-6 projection max_iterations"
        )
        corrector_iterations = _require_nonnegative_integer(
            corrector.get("iterations"), "schema-6 corrector iterations"
        )
        corrector_residual_calls = _require_nonnegative_integer(
            corrector.get("residual_operator_calls"), "schema-6 corrector residual_operator_calls"
        )
        corrector_preconditioner_calls = _require_nonnegative_integer(
            corrector.get("preconditioner_calls"), "schema-6 corrector preconditioner_calls"
        )
        corrector_line_search_candidates = _require_nonnegative_integer(
            corrector.get("line_search_candidates"), "schema-6 corrector line_search_candidates"
        )
        direct_payload["inference_work"] = {
            "predictor_passes": iterations,
            "global_compatibility_projections": iterations,
            "maximum_projection_iterations": iterations * maximum_projection_iterations,
            "common_residual_evaluations": iterations + 1 + corrector_residual_calls,
            "common_objective_evaluations": iterations + 1,
            "state_validity_evaluations": iterations + 1,
            "physical_step_authentications": 2 * iterations + 3,
            "common_objective_authentications": 2 * iterations + 3,
            "constraint_preparations": iterations,
            "constraint_applications": iterations,
            "corrector_iterations": corrector_iterations,
            "corrector_residual_operator_calls": corrector_residual_calls,
            "corrector_preconditioner_calls": corrector_preconditioner_calls,
            "corrector_line_search_candidates": corrector_line_search_candidates,
        }
        direct_payload["solver_contract_sha256"] = canonical_json_sha256(direct_payload)
        direct = V5SolverContract.from_dict(direct_payload)
        result = cls._from_direct(
            direct,
            training_proposal_policy=training_policy,
            inference_proposal_safeguard=proposal,
        )
        if result.as_dict() != value:
            raise ValueError("schema-6 solver contract is not in canonical serialized form")
        return result


@dataclasses.dataclass(frozen=True)
class VerifiedV5Checkpoint:
    """Self-contained verification result with no evaluation assumptions."""

    checkpoint_payload_sha256: str
    learned_state_sha256: str
    optimizer_state_sha256: str
    rng_state_sha256: str
    batch_stream_sha256: str
    completed_updates: int
    solver_contract: V5SolverContract
    parent_lineage: ParentLineage


def _checkpoint_payload_record(checkpoint: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "contract": _CHECKPOINT_CONTRACT,
        "learned_state_sha256": checkpoint["learned_state_sha256"],
        "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
        "rng_state_sha256": checkpoint["rng_state_sha256"],
        "batch_stream_sha256": checkpoint["batch_stream_sha256"],
        "metadata_sha256": checkpoint["metadata_sha256"],
    }


def build_v5_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    *,
    solver_contract: V5SolverContract,
    optimizer_state: object,
    rng_state: object,
    batch_stream: SamplingSchedule,
    completed_updates: int,
    parent_lineage: ParentLineage,
) -> dict[str, object]:
    """Build a CPU-resident, self-authenticating architecture-v5 checkpoint."""
    if not isinstance(solver_contract, V5SolverContract):
        raise TypeError("solver_contract must be a canonical V5SolverContract")
    if not isinstance(parent_lineage, ParentLineage):
        raise TypeError("parent_lineage must be a canonical ParentLineage")
    _validate_v5_learned_state(
        state_dict,
        solver_contract.graph_config,
        solver_contract.learned_parameter_dtype,
    )
    schedule = _verify_sampling_schedule(
        batch_stream,
        manifest_sha256=solver_contract.training_dataset_sha256,
        expected_steps=solver_contract.sampling_steps,
    )
    if schedule.schedule_sha256 != solver_contract.sampling_schedule_sha256:
        raise ValueError("batch stream differs from the solver contract's training SamplingSchedule")
    _require_nonnegative_integer(completed_updates, "completed_updates")
    if completed_updates > solver_contract.sampling_steps:
        raise ValueError("completed_updates exceeds the authenticated training schedule")
    if parent_lineage.kind == "continuation" and parent_lineage.parent_completed_updates >= completed_updates:
        raise ValueError("continuation parent must precede completed_updates")
    learned_state = {name: tensor.detach().cpu().clone() for name, tensor in sorted(state_dict.items())}
    learned_digest = learned_state_sha256(learned_state)
    optimizer = _clone_state_tree(optimizer_state)
    rng = _clone_state_tree(rng_state)
    batches = schedule.as_dict()
    optimizer_digest = state_tree_sha256(optimizer)
    rng_digest = state_tree_sha256(rng)
    batch_digest = state_tree_sha256(batches)
    continuation_snapshot = {
        "resume_capability": False,
        "semantics": "integrity-only-not-a-resume-proof",
        "optimizer": {
            "contract": solver_contract.optimizer.as_dict(),
            "state_sha256": optimizer_digest,
        },
        "rng": {
            "algorithm": solver_contract.rng_algorithm,
            "state_sha256": rng_digest,
        },
        "batch_stream": {
            "contract": solver_contract.batch_stream_contract,
            "state_sha256": batch_digest,
        },
        "parent_lineage": parent_lineage.as_dict(),
        "completed_updates": completed_updates,
        "next_batch_index": completed_updates,
    }
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _CHECKPOINT_CONTRACT,
        "solver_contract": solver_contract.as_dict(),
        "solver_contract_sha256": solver_contract.solver_contract_sha256,
        "continuation_snapshot": continuation_snapshot,
    }
    checkpoint: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _CHECKPOINT_CONTRACT,
        "state_dict": learned_state,
        "learned_state_sha256": learned_digest,
        "optimizer_state": optimizer,
        "optimizer_state_sha256": optimizer_digest,
        "rng_state": rng,
        "rng_state_sha256": rng_digest,
        "batch_stream": batches,
        "batch_stream_sha256": batch_digest,
        "metadata": metadata,
        "metadata_sha256": canonical_json_sha256(metadata),
    }
    checkpoint["checkpoint_payload_sha256"] = canonical_json_sha256(_checkpoint_payload_record(checkpoint))
    verify_v5_checkpoint(checkpoint)
    return checkpoint


def verify_v5_checkpoint(checkpoint: Mapping[str, object]) -> VerifiedV5Checkpoint:
    """Verify only checkpoint-internal v5 identity, semantics, and hashes.

    No history, topology, or evaluation state is accepted here.  Call
    :func:`build_v5_evaluation_binding` to bind this verified artifact to a
    separately authenticated held-out trajectory.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("schema-v5 checkpoint must be a mapping")
    expected_keys = {
        "schema_version",
        "contract",
        "state_dict",
        "learned_state_sha256",
        "optimizer_state",
        "optimizer_state_sha256",
        "rng_state",
        "rng_state_sha256",
        "batch_stream",
        "batch_stream_sha256",
        "metadata",
        "metadata_sha256",
        "checkpoint_payload_sha256",
    }
    _strict_keys(checkpoint, expected_keys, "schema-v5 checkpoint")
    if (checkpoint.get("schema_version"), checkpoint.get("contract")) != (
        _SCHEMA_VERSION,
        _CHECKPOINT_CONTRACT,
    ):
        raise ValueError("checkpoint does not have the exact schema-v5 checkpoint identity")

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("schema-v5 checkpoint state_dict must be a mapping")
    if learned_state_sha256(state_dict) != checkpoint["learned_state_sha256"]:
        raise ValueError("checkpoint learned-state SHA-256 verification failed")
    if state_tree_sha256(checkpoint["optimizer_state"]) != checkpoint["optimizer_state_sha256"]:
        raise ValueError("checkpoint optimizer-state SHA-256 verification failed")
    if state_tree_sha256(checkpoint["rng_state"]) != checkpoint["rng_state_sha256"]:
        raise ValueError("checkpoint RNG-state SHA-256 verification failed")
    if state_tree_sha256(checkpoint["batch_stream"]) != checkpoint["batch_stream_sha256"]:
        raise ValueError("checkpoint batch-stream SHA-256 verification failed")

    metadata = checkpoint["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("schema-v5 checkpoint metadata must be a mapping")
    _strict_keys(
        metadata,
        {
            "schema_version",
            "contract",
            "solver_contract",
            "solver_contract_sha256",
            "continuation_snapshot",
        },
        "schema-v5 checkpoint metadata",
    )
    if (metadata.get("schema_version"), metadata.get("contract")) != (
        _SCHEMA_VERSION,
        _CHECKPOINT_CONTRACT,
    ):
        raise ValueError("checkpoint and metadata schema-v5 identities disagree")
    solver_payload = metadata["solver_contract"]
    if not isinstance(solver_payload, Mapping):
        raise ValueError("checkpoint solver contract must be a mapping")
    solver_contract = V5SolverContract.from_dict(solver_payload)
    if metadata["solver_contract_sha256"] != solver_contract.solver_contract_sha256:
        raise ValueError("checkpoint solver-contract SHA-256 copies disagree")
    _validate_v5_learned_state(
        state_dict,
        solver_contract.graph_config,
        solver_contract.learned_parameter_dtype,
    )

    continuation_snapshot = metadata["continuation_snapshot"]
    if not isinstance(continuation_snapshot, Mapping):
        raise ValueError("checkpoint continuation snapshot metadata must be a mapping")
    _strict_keys(
        continuation_snapshot,
        {
            "resume_capability",
            "semantics",
            "optimizer",
            "rng",
            "batch_stream",
            "parent_lineage",
            "completed_updates",
            "next_batch_index",
        },
        "continuation snapshot metadata",
    )
    if continuation_snapshot["resume_capability"] is not False or continuation_snapshot["semantics"] != (
        "integrity-only-not-a-resume-proof"
    ):
        raise ValueError("foundation checkpoint must explicitly disclaim exact-resume capability")
    optimizer = continuation_snapshot["optimizer"]
    rng = continuation_snapshot["rng"]
    batch_stream = continuation_snapshot["batch_stream"]
    parent = continuation_snapshot["parent_lineage"]
    if not all(isinstance(item, Mapping) for item in (optimizer, rng, batch_stream, parent)):
        raise ValueError("checkpoint continuation component metadata must be mappings")
    _strict_keys(optimizer, {"contract", "state_sha256"}, "optimizer continuation metadata")
    _strict_keys(rng, {"algorithm", "state_sha256"}, "RNG continuation metadata")
    _strict_keys(batch_stream, {"contract", "state_sha256"}, "batch-stream continuation metadata")
    if optimizer["contract"] != solver_contract.optimizer.as_dict():
        raise ValueError("checkpoint optimizer metadata differs from its solver contract")
    if optimizer["state_sha256"] != checkpoint["optimizer_state_sha256"]:
        raise ValueError("checkpoint optimizer-state SHA-256 copies disagree")
    if rng != {
        "algorithm": solver_contract.rng_algorithm,
        "state_sha256": checkpoint["rng_state_sha256"],
    }:
        raise ValueError("checkpoint RNG metadata differs from its solver contract or state")
    if batch_stream != {
        "contract": solver_contract.batch_stream_contract,
        "state_sha256": checkpoint["batch_stream_sha256"],
    }:
        raise ValueError("checkpoint batch-stream metadata differs from its solver contract or state")
    parent_lineage = ParentLineage.from_dict(parent)
    completed_updates = _require_nonnegative_integer(
        continuation_snapshot["completed_updates"],
        "completed_updates",
    )
    if continuation_snapshot["next_batch_index"] != completed_updates:
        raise ValueError("checkpoint next_batch_index differs from completed_updates")
    if completed_updates > solver_contract.sampling_steps:
        raise ValueError("checkpoint completed_updates exceeds the training schedule")
    if parent_lineage.kind == "continuation" and parent_lineage.parent_completed_updates >= completed_updates:
        raise ValueError("checkpoint continuation parent does not precede completed_updates")

    schedule_payload = checkpoint["batch_stream"]
    if not isinstance(schedule_payload, Mapping):
        raise ValueError("checkpoint batch stream must be a SamplingSchedule payload")
    if schedule_payload.get("schedule_sha256") != solver_contract.sampling_schedule_sha256:
        raise ValueError("checkpoint batch stream differs from the authenticated SamplingSchedule")
    if schedule_payload.get("manifest_sha256") != solver_contract.training_dataset_sha256:
        raise ValueError("checkpoint batch stream differs from the authenticated SplitManifest")
    if (
        schedule_payload.get("contract") != solver_contract.batch_stream_contract
        or schedule_payload.get("role") != "train"
        or schedule_payload.get("steps") != solver_contract.sampling_steps
        or schedule_payload.get("batch_size") != solver_contract.sampling_batch_size
        or schedule_payload.get("seed") != solver_contract.sampling_seed
    ):
        raise ValueError("checkpoint batch stream settings differ from the solver contract")

    if canonical_json_sha256(metadata) != checkpoint["metadata_sha256"]:
        raise ValueError("checkpoint metadata SHA-256 verification failed")
    payload_digest = canonical_json_sha256(_checkpoint_payload_record(checkpoint))
    if payload_digest != checkpoint["checkpoint_payload_sha256"]:
        raise ValueError("checkpoint payload SHA-256 verification failed")
    return VerifiedV5Checkpoint(
        checkpoint_payload_sha256=payload_digest,
        learned_state_sha256=checkpoint["learned_state_sha256"],
        optimizer_state_sha256=checkpoint["optimizer_state_sha256"],
        rng_state_sha256=checkpoint["rng_state_sha256"],
        batch_stream_sha256=checkpoint["batch_stream_sha256"],
        completed_updates=completed_updates,
        solver_contract=solver_contract,
        parent_lineage=parent_lineage,
    )


@dataclasses.dataclass(frozen=True)
class VerifiedV6Checkpoint:
    """Self-contained schema-6 verification result."""

    checkpoint_payload_sha256: str
    learned_state_sha256: str
    optimizer_state_sha256: str
    rng_state_sha256: str
    batch_stream_sha256: str
    completed_updates: int
    solver_contract: V6SolverContract
    parent_lineage: ParentLineage


def _checkpoint_v6_payload_record(checkpoint: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_V6_VERSION,
        "contract": _CHECKPOINT_V6_CONTRACT,
        "learned_state_sha256": checkpoint["learned_state_sha256"],
        "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
        "rng_state_sha256": checkpoint["rng_state_sha256"],
        "batch_stream_sha256": checkpoint["batch_stream_sha256"],
        "metadata_sha256": checkpoint["metadata_sha256"],
    }


def build_v6_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    *,
    solver_contract: V6SolverContract,
    optimizer_state: object,
    rng_state: object,
    batch_stream: SamplingSchedule,
    completed_updates: int,
    parent_lineage: ParentLineage,
) -> dict[str, object]:
    """Build a schema-6 checkpoint for fixed-candidate v5 inference."""
    if type(solver_contract) is not V6SolverContract:
        raise TypeError("solver_contract must be a canonical V6SolverContract")
    if not isinstance(parent_lineage, ParentLineage):
        raise TypeError("parent_lineage must be a canonical ParentLineage")
    _validate_v5_learned_state(
        state_dict,
        solver_contract.graph_config,
        solver_contract.learned_parameter_dtype,
    )
    schedule = _verify_sampling_schedule(
        batch_stream,
        manifest_sha256=solver_contract.training_dataset_sha256,
        expected_steps=solver_contract.sampling_steps,
    )
    if schedule.schedule_sha256 != solver_contract.sampling_schedule_sha256:
        raise ValueError("batch stream differs from the schema-6 solver contract's training SamplingSchedule")
    _require_nonnegative_integer(completed_updates, "completed_updates")
    if completed_updates > solver_contract.sampling_steps:
        raise ValueError("completed_updates exceeds the authenticated training schedule")
    if parent_lineage.kind == "continuation" and parent_lineage.parent_completed_updates >= completed_updates:
        raise ValueError("continuation parent must precede completed_updates")
    learned_state = {name: tensor.detach().cpu().clone() for name, tensor in sorted(state_dict.items())}
    learned_digest = learned_state_sha256(learned_state)
    optimizer = _clone_state_tree(optimizer_state)
    rng = _clone_state_tree(rng_state)
    batches = schedule.as_dict()
    optimizer_digest = state_tree_sha256(optimizer)
    rng_digest = state_tree_sha256(rng)
    batch_digest = state_tree_sha256(batches)
    continuation_snapshot = {
        "resume_capability": False,
        "semantics": "integrity-only-not-a-resume-proof",
        "optimizer": {
            "contract": solver_contract.optimizer.as_dict(),
            "state_sha256": optimizer_digest,
        },
        "rng": {
            "algorithm": solver_contract.rng_algorithm,
            "state_sha256": rng_digest,
        },
        "batch_stream": {
            "contract": solver_contract.batch_stream_contract,
            "state_sha256": batch_digest,
        },
        "parent_lineage": parent_lineage.as_dict(),
        "completed_updates": completed_updates,
        "next_batch_index": completed_updates,
    }
    metadata = {
        "schema_version": _SCHEMA_V6_VERSION,
        "contract": _CHECKPOINT_V6_CONTRACT,
        "solver_contract": solver_contract.as_dict(),
        "solver_contract_sha256": solver_contract.solver_contract_sha256,
        "continuation_snapshot": continuation_snapshot,
    }
    checkpoint: dict[str, object] = {
        "schema_version": _SCHEMA_V6_VERSION,
        "contract": _CHECKPOINT_V6_CONTRACT,
        "state_dict": learned_state,
        "learned_state_sha256": learned_digest,
        "optimizer_state": optimizer,
        "optimizer_state_sha256": optimizer_digest,
        "rng_state": rng,
        "rng_state_sha256": rng_digest,
        "batch_stream": batches,
        "batch_stream_sha256": batch_digest,
        "metadata": metadata,
        "metadata_sha256": canonical_json_sha256(metadata),
    }
    checkpoint["checkpoint_payload_sha256"] = canonical_json_sha256(_checkpoint_v6_payload_record(checkpoint))
    verify_v6_checkpoint(checkpoint)
    return checkpoint


def verify_v6_checkpoint(checkpoint: Mapping[str, object]) -> VerifiedV6Checkpoint:
    """Verify only schema-6 checkpoint identity, semantics, and hashes."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("schema-v6 checkpoint must be a mapping")
    expected_keys = {
        "schema_version",
        "contract",
        "state_dict",
        "learned_state_sha256",
        "optimizer_state",
        "optimizer_state_sha256",
        "rng_state",
        "rng_state_sha256",
        "batch_stream",
        "batch_stream_sha256",
        "metadata",
        "metadata_sha256",
        "checkpoint_payload_sha256",
    }
    _strict_keys(checkpoint, expected_keys, "schema-v6 checkpoint")
    if (checkpoint.get("schema_version"), checkpoint.get("contract")) != (
        _SCHEMA_V6_VERSION,
        _CHECKPOINT_V6_CONTRACT,
    ):
        raise ValueError("checkpoint does not have the exact schema-v6 checkpoint identity")

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("schema-v6 checkpoint state_dict must be a mapping")
    if learned_state_sha256(state_dict) != checkpoint["learned_state_sha256"]:
        raise ValueError("checkpoint learned-state SHA-256 verification failed")
    if state_tree_sha256(checkpoint["optimizer_state"]) != checkpoint["optimizer_state_sha256"]:
        raise ValueError("checkpoint optimizer-state SHA-256 verification failed")
    if state_tree_sha256(checkpoint["rng_state"]) != checkpoint["rng_state_sha256"]:
        raise ValueError("checkpoint RNG-state SHA-256 verification failed")
    if state_tree_sha256(checkpoint["batch_stream"]) != checkpoint["batch_stream_sha256"]:
        raise ValueError("checkpoint batch-stream SHA-256 verification failed")

    metadata = checkpoint["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("schema-v6 checkpoint metadata must be a mapping")
    _strict_keys(
        metadata,
        {
            "schema_version",
            "contract",
            "solver_contract",
            "solver_contract_sha256",
            "continuation_snapshot",
        },
        "schema-v6 checkpoint metadata",
    )
    if (metadata.get("schema_version"), metadata.get("contract")) != (
        _SCHEMA_V6_VERSION,
        _CHECKPOINT_V6_CONTRACT,
    ):
        raise ValueError("checkpoint and metadata schema-v6 identities disagree")
    solver_payload = metadata["solver_contract"]
    if not isinstance(solver_payload, Mapping):
        raise ValueError("checkpoint schema-6 solver contract must be a mapping")
    solver_contract = V6SolverContract.from_dict(solver_payload)
    if metadata["solver_contract_sha256"] != solver_contract.solver_contract_sha256:
        raise ValueError("checkpoint solver-contract SHA-256 copies disagree")
    _validate_v5_learned_state(
        state_dict,
        solver_contract.graph_config,
        solver_contract.learned_parameter_dtype,
    )

    continuation_snapshot = metadata["continuation_snapshot"]
    if not isinstance(continuation_snapshot, Mapping):
        raise ValueError("checkpoint continuation snapshot metadata must be a mapping")
    _strict_keys(
        continuation_snapshot,
        {
            "resume_capability",
            "semantics",
            "optimizer",
            "rng",
            "batch_stream",
            "parent_lineage",
            "completed_updates",
            "next_batch_index",
        },
        "continuation snapshot metadata",
    )
    if continuation_snapshot["resume_capability"] is not False or continuation_snapshot["semantics"] != (
        "integrity-only-not-a-resume-proof"
    ):
        raise ValueError("foundation checkpoint must explicitly disclaim exact-resume capability")
    optimizer = continuation_snapshot["optimizer"]
    rng = continuation_snapshot["rng"]
    batch_stream = continuation_snapshot["batch_stream"]
    parent = continuation_snapshot["parent_lineage"]
    if not all(isinstance(item, Mapping) for item in (optimizer, rng, batch_stream, parent)):
        raise ValueError("checkpoint continuation component metadata must be mappings")
    _strict_keys(optimizer, {"contract", "state_sha256"}, "optimizer continuation metadata")
    _strict_keys(rng, {"algorithm", "state_sha256"}, "RNG continuation metadata")
    _strict_keys(batch_stream, {"contract", "state_sha256"}, "batch-stream continuation metadata")
    if optimizer["contract"] != solver_contract.optimizer.as_dict():
        raise ValueError("checkpoint optimizer metadata differs from its solver contract")
    if optimizer["state_sha256"] != checkpoint["optimizer_state_sha256"]:
        raise ValueError("checkpoint optimizer-state SHA-256 copies disagree")
    if rng != {
        "algorithm": solver_contract.rng_algorithm,
        "state_sha256": checkpoint["rng_state_sha256"],
    }:
        raise ValueError("checkpoint RNG metadata differs from its solver contract or state")
    if batch_stream != {
        "contract": solver_contract.batch_stream_contract,
        "state_sha256": checkpoint["batch_stream_sha256"],
    }:
        raise ValueError("checkpoint batch-stream metadata differs from its solver contract or state")
    parent_lineage = ParentLineage.from_dict(parent)
    completed_updates = _require_nonnegative_integer(
        continuation_snapshot["completed_updates"],
        "completed_updates",
    )
    if continuation_snapshot["next_batch_index"] != completed_updates:
        raise ValueError("checkpoint next_batch_index differs from completed_updates")
    if completed_updates > solver_contract.sampling_steps:
        raise ValueError("checkpoint completed_updates exceeds the training schedule")
    if parent_lineage.kind == "continuation" and parent_lineage.parent_completed_updates >= completed_updates:
        raise ValueError("checkpoint continuation parent does not precede completed_updates")

    schedule_payload = checkpoint["batch_stream"]
    if not isinstance(schedule_payload, Mapping):
        raise ValueError("checkpoint batch stream must be a SamplingSchedule payload")
    if schedule_payload.get("schedule_sha256") != solver_contract.sampling_schedule_sha256:
        raise ValueError("checkpoint batch stream differs from the authenticated SamplingSchedule")
    if schedule_payload.get("manifest_sha256") != solver_contract.training_dataset_sha256:
        raise ValueError("checkpoint batch stream differs from the authenticated SplitManifest")
    if (
        schedule_payload.get("contract") != solver_contract.batch_stream_contract
        or schedule_payload.get("role") != "train"
        or schedule_payload.get("steps") != solver_contract.sampling_steps
        or schedule_payload.get("batch_size") != solver_contract.sampling_batch_size
        or schedule_payload.get("seed") != solver_contract.sampling_seed
    ):
        raise ValueError("checkpoint batch stream settings differ from the solver contract")

    if canonical_json_sha256(metadata) != checkpoint["metadata_sha256"]:
        raise ValueError("checkpoint metadata SHA-256 verification failed")
    payload_digest = canonical_json_sha256(_checkpoint_v6_payload_record(checkpoint))
    if payload_digest != checkpoint["checkpoint_payload_sha256"]:
        raise ValueError("checkpoint payload SHA-256 verification failed")
    return VerifiedV6Checkpoint(
        checkpoint_payload_sha256=payload_digest,
        learned_state_sha256=checkpoint["learned_state_sha256"],
        optimizer_state_sha256=checkpoint["optimizer_state_sha256"],
        rng_state_sha256=checkpoint["rng_state_sha256"],
        batch_stream_sha256=checkpoint["batch_stream_sha256"],
        completed_updates=completed_updates,
        solver_contract=solver_contract,
        parent_lineage=parent_lineage,
    )


def _verified_held_out_trajectory(value: object) -> dict[str, object]:
    if type(value) is not TrajectoryRecord:
        raise ValueError("held-out trajectory must be a canonical TrajectoryRecord")
    payload = value.as_dict()
    canonical = _jsonable(payload)
    if not isinstance(canonical, dict):
        raise RuntimeError("canonical held-out trajectory changed type")
    trajectory_digest = canonical.get("trajectory_sha256")
    _require_sha256(trajectory_digest, "held-out trajectory_sha256")
    body = dict(canonical)
    body.pop("trajectory_sha256")
    if canonical_json_sha256(body) != trajectory_digest:
        raise ValueError("held-out trajectory SHA-256 verification failed")
    if body.get("contract") != _TRAJECTORY_CONTRACT:
        raise ValueError("held-out trajectory uses an unsupported record contract")
    _require_string(body.get("trajectory_id"), "held-out trajectory_id")
    _require_sha256(body.get("topology_sha256"), "held-out topology_sha256")
    _require_sha256(body.get("operator_geometry_sha256"), "held-out operator_geometry_sha256")
    samples = body.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
        raise ValueError("held-out trajectory must contain sample records")
    sample_keys: list[tuple[int, str]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("held-out trajectory has a malformed sample record")
        sample_digest = sample.get("sample_sha256")
        _require_sha256(sample_digest, "held-out sample_sha256")
        sample_body = dict(sample)
        sample_body.pop("sample_sha256")
        if canonical_json_sha256(sample_body) != sample_digest:
            raise ValueError("held-out sample SHA-256 verification failed")
        sample_id = _require_string(sample.get("sample_id"), "held-out sample_id")
        ordinal = _require_nonnegative_integer(sample.get("ordinal"), "held-out sample ordinal")
        sample_keys.append((ordinal, sample_id))
    if len(set(sample_keys)) != len(sample_keys) or sample_keys != sorted(sample_keys):
        raise ValueError("held-out sample records must be unique and canonically ordered")
    return canonical


def _verified_evaluation_access(
    ledger: object,
    *,
    manifest: SplitManifest,
    trajectory: TrajectoryRecord,
    role: DatasetRole,
) -> tuple[str, tuple[str, ...]]:
    if type(ledger) is not DataAccessLedger:
        raise ValueError("evaluation requires a canonical DataAccessLedger")
    if ledger.manifest.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("evaluation DataAccessLedger differs from the frozen SplitManifest")
    payload = ledger.as_dict()
    declared = payload.pop("ledger_sha256")
    if canonical_json_sha256(payload) != declared or declared != ledger.ledger_sha256:
        raise ValueError("evaluation DataAccessLedger SHA-256 verification failed")
    expected_purpose = (
        DataAccessPurpose.CONFIRMATION_EVALUATION
        if role is DatasetRole.CONFIRMATION
        else DataAccessPurpose.MODEL_SELECTION
    )
    matching = [
        access
        for access in ledger.accesses
        if access.trajectory_id == trajectory.trajectory_id
        and access.scope is DataAccessScope.PAYLOAD
        and access.purpose is expected_purpose
    ]
    if not matching:
        raise ValueError("evaluation DataAccessLedger has no role-correct held-out payload access")
    required_context_payloads = {"physical_step", "common_objective"}
    if not required_context_payloads.issubset(matching[-1].payload_names):
        raise ValueError("evaluation payload access must include physical_step and common_objective")
    if role is DatasetRole.CONFIRMATION and not ledger.confirmation_payload_released:
        raise ValueError("confirmation payload was not released on the supplied access-ledger branch")
    return ledger.ledger_sha256, matching[-1].payload_names


def _require_reconstructed_tensor(
    name: str,
    observed: object,
    expected: torch.Tensor,
    *,
    epsilon_multiplier: float = 1024.0,
) -> None:
    """Compare deterministic reconstructed state with a fixed roundoff bound."""
    if not isinstance(observed, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if (
        observed.layout != expected.layout
        or observed.shape != expected.shape
        or observed.dtype != expected.dtype
        or observed.device != expected.device
    ):
        raise ValueError(f"{name} has incompatible tensor metadata")
    if observed.layout == torch.sparse_csr:
        if not torch.equal(observed.crow_indices(), expected.crow_indices()) or not torch.equal(
            observed.col_indices(), expected.col_indices()
        ):
            raise ValueError(f"{name} has a noncanonical sparse structure")
        _require_reconstructed_tensor(
            f"{name}.values",
            observed.values(),
            expected.values(),
            epsilon_multiplier=epsilon_multiplier,
        )
        return
    if observed.layout != torch.strided:
        raise ValueError(f"{name} uses an unsupported tensor layout")
    if not observed.is_floating_point():
        if not torch.equal(observed, expected):
            raise ValueError(f"{name} differs from its canonical reconstruction")
        return
    if not torch.isfinite(observed).all() or not torch.isfinite(expected).all():
        raise ValueError(f"{name} must be finite")
    epsilon = torch.finfo(expected.dtype).eps
    # Keep reconstruction error scale-relative.  A unit absolute floor would
    # authenticate arbitrarily large relative changes on small physical
    # meshes/operators.  Canonical zeros are exact; nonzero entries receive a
    # fixed ULP-scale relative allowance for independent NumPy/Torch inversion
    # and factor-product roundoff.
    bound = epsilon_multiplier * epsilon * expected.abs()
    if not bool(((observed - expected).abs() <= bound).all()):
        raise ValueError(f"{name} differs from its canonical reconstruction")


def _require_contribution_scaled_tensor(
    name: str,
    observed: object,
    expected: torch.Tensor,
    contribution_scale: torch.Tensor,
) -> None:
    """Compare an assembled operator using its local absolute contributions."""
    if not isinstance(observed, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if (
        observed.layout != expected.layout
        or contribution_scale.layout != expected.layout
        or observed.shape != expected.shape
        or contribution_scale.shape != expected.shape
        or observed.dtype != expected.dtype
        or contribution_scale.dtype != expected.dtype
        or observed.device != expected.device
        or contribution_scale.device != expected.device
    ):
        raise ValueError(f"{name} has incompatible tensor metadata")
    if observed.layout == torch.sparse_csr:
        for candidate in (observed, contribution_scale):
            if not torch.equal(candidate.crow_indices(), expected.crow_indices()) or not torch.equal(
                candidate.col_indices(), expected.col_indices()
            ):
                raise ValueError(f"{name} has a noncanonical sparse structure")
        _require_contribution_scaled_tensor(
            f"{name}.values",
            observed.values(),
            expected.values(),
            contribution_scale.values(),
        )
        return
    if observed.layout != torch.strided or not observed.is_floating_point():
        raise ValueError(f"{name} must use a strided floating tensor")
    if (
        not torch.isfinite(observed).all()
        or not torch.isfinite(expected).all()
        or not torch.isfinite(contribution_scale).all()
        or (contribution_scale < 0.0).any()
    ):
        raise ValueError(f"{name} or its contribution scale is invalid")
    epsilon = torch.finfo(expected.dtype).eps
    local_scale = torch.maximum(expected.abs(), contribution_scale)
    bound = 4096.0 * epsilon * local_scale
    if not bool(((observed - expected).abs() <= bound).all()):
        raise ValueError(f"{name} differs from its canonical contribution-scaled reconstruction")


def _require_cholesky_product(
    name: str,
    factor: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Verify ``factor @ factor.T`` with a local dot-product error bound."""
    if (
        factor.layout != torch.strided
        or factor.ndim != 2
        or factor.shape[0] != factor.shape[1]
        or expected.layout != torch.strided
        or expected.shape != factor.shape
        or expected.dtype != factor.dtype
        or expected.device != factor.device
        or not factor.is_floating_point()
    ):
        raise ValueError(f"{name} has incompatible tensor metadata")
    if not torch.isfinite(factor).all() or not torch.isfinite(expected).all():
        raise ValueError(f"{name} must be finite")
    product = factor @ factor.transpose(0, 1)
    absolute_product = factor.abs() @ factor.abs().transpose(0, 1)
    epsilon = torch.finfo(factor.dtype).eps
    dot_roundoff = factor.shape[1] * epsilon
    if dot_roundoff >= 0.5:
        raise ValueError(f"{name} dimension exceeds the registered roundoff model")
    gamma = dot_roundoff / (1.0 - dot_roundoff)
    # A structural zero can acquire a cancellation residual in the factor
    # product even though the directly assembled operator is exactly zero.
    # Scale only by the local absolute dot product—never by a dimensionless
    # unit floor—so the allowance shrinks with the physical operator.
    bound = 32.0 * gamma * absolute_product
    if not bool(((product - expected).abs() <= bound).all()):
        raise ValueError(f"{name} differs from the directly verified compatibility operator")


def _canonical_sparse_reduced_operator(
    stiffness: torch.Tensor,
    tets: torch.Tensor,
    free: torch.Tensor,
    pinned: torch.Tensor,
    n_vertices: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reassemble the registered zero-regularization Jacobi projection."""
    free_index = torch.full((n_vertices,), -1, dtype=torch.int64, device=tets.device)
    free_index[free] = torch.arange(free.numel(), dtype=torch.int64, device=tets.device)
    pinned_index = torch.full((n_vertices,), -1, dtype=torch.int64, device=tets.device)
    pinned_index[pinned] = torch.arange(pinned.numel(), dtype=torch.int64, device=tets.device)
    rows = tets[:, :, None].expand(-1, -1, 4)
    columns = tets[:, None, :].expand(-1, 4, -1)
    free_rows = free_index[rows]
    free_columns = free_index[columns]
    pinned_columns = pinned_index[columns]

    free_free = (free_rows >= 0) & (free_columns >= 0)
    ff = torch.sparse_coo_tensor(
        torch.stack((free_rows[free_free], free_columns[free_free])),
        stiffness[free_free],
        size=(free.numel(), free.numel()),
        dtype=stiffness.dtype,
        device=stiffness.device,
    ).coalesce()
    ff_contribution_scale = torch.sparse_coo_tensor(
        torch.stack((free_rows[free_free], free_columns[free_free])),
        stiffness[free_free].abs(),
        size=(free.numel(), free.numel()),
        dtype=stiffness.dtype,
        device=stiffness.device,
    ).coalesce()
    ff_indices = ff.indices()
    diagonal_mask = ff_indices[0] == ff_indices[1]
    diagonal = torch.zeros(free.numel(), dtype=stiffness.dtype, device=stiffness.device)
    diagonal.index_add_(0, ff_indices[0, diagonal_mask], ff.values()[diagonal_mask])
    if not torch.isfinite(diagonal).all() or (diagonal <= 0.0).any():
        raise ValueError("canonical sparse compatibility operator has an invalid diagonal")

    free_pinned = (free_rows >= 0) & (pinned_columns >= 0)
    fp = torch.sparse_coo_tensor(
        torch.stack((free_rows[free_pinned], pinned_columns[free_pinned])),
        stiffness[free_pinned],
        size=(free.numel(), pinned.numel()),
        dtype=stiffness.dtype,
        device=stiffness.device,
    ).coalesce()
    fp_contribution_scale = torch.sparse_coo_tensor(
        torch.stack((free_rows[free_pinned], pinned_columns[free_pinned])),
        stiffness[free_pinned].abs(),
        size=(free.numel(), pinned.numel()),
        dtype=stiffness.dtype,
        device=stiffness.device,
    ).coalesce()
    return (
        ff.to_sparse_csr(),
        fp.to_sparse_csr(),
        diagonal.reciprocal(),
        ff_contribution_scale.to_sparse_csr(),
        fp_contribution_scale.to_sparse_csr(),
    )


def _verify_projection_operator(state: SolverState, contract: ProjectionContract) -> None:
    """Rebuild geometry and the compatibility operator without trusting its hash."""
    if state.projection_backend != contract.backend or state.tikhonov != 0.0:
        raise ValueError("evaluation projection backend or regularization differs from the solver contract")
    if state.operator_geometry_policy != contract.operator_geometry_policy:
        raise ValueError("evaluation operator-geometry policy differs from the solver contract")
    validate_authenticated_operator_geometry(state)
    rest = state.rest_q
    source_rest = state.source_rest_q
    if rest.layout != torch.strided or str(rest.dtype) != contract.execution_dtype:
        raise ValueError("evaluation projection rest positions differ from the contracted execution dtype")
    if (
        source_rest.layout != torch.strided
        or source_rest.dtype != torch.float64
        or source_rest.device != rest.device
        or source_rest.shape != rest.shape
        or not torch.isfinite(source_rest).all()
        or not torch.equal(rest, source_rest.to(dtype=rest.dtype))
    ):
        raise ValueError("evaluation projection runtime rest positions differ from canonical source_rest_q")
    if rest.shape != (state.n_verts, 3) or not torch.isfinite(rest).all():
        raise ValueError("evaluation projection rest positions are invalid")
    if (
        state.tets.dtype != torch.int64
        or state.pinned.dtype != torch.int64
        or state.free.dtype != torch.int64
        or state.tets.device != rest.device
        or state.pinned.device != rest.device
        or state.free.device != rest.device
        or state.tets.shape != (state.n_tets, 4)
    ):
        raise ValueError("evaluation projection index tensors are invalid")
    if (
        (state.tets < 0).any()
        or (state.tets >= state.n_verts).any()
        or (state.pinned < 0).any()
        or (state.pinned >= state.n_verts).any()
        or torch.unique(state.pinned).numel() != state.pinned.numel()
    ):
        raise ValueError("evaluation projection connectivity or pins are invalid")
    mask = torch.ones(state.n_verts, dtype=torch.bool, device=rest.device)
    mask[state.pinned] = False
    expected_free = torch.where(mask)[0]
    if not torch.equal(state.free, expected_free):
        raise ValueError("evaluation projection free-vertex ordering is noncanonical")

    expected_dm_inv = state.source_tet_poses.to(dtype=rest.dtype)
    if not torch.equal(state.Dm_inv, expected_dm_inv):
        raise ValueError("evaluation projection Dm_inv differs from the exact source-pose promotion")
    expected_j = torch.zeros(state.n_tets, 4, 3, dtype=rest.dtype, device=rest.device)
    expected_j[:, 1:, :] = expected_dm_inv
    expected_j[:, 0, :] = -expected_dm_inv.sum(dim=1)
    expected_volume = 1.0 / (6.0 * torch.linalg.det(expected_dm_inv))
    _require_reconstructed_tensor("evaluation projection state J", state.J, expected_j, epsilon_multiplier=1024.0)
    _require_reconstructed_tensor(
        "evaluation projection state volume", state.w, expected_volume, epsilon_multiplier=1024.0
    )

    stiffness = torch.einsum("tac,tbc->tab", expected_j, expected_j) * expected_volume[:, None, None]
    if contract.backend == "dense":
        if state.L is None or state.L_ff_chol is None:
            raise ValueError("dense compatibility projection is missing its operator or factor")
        if (
            state.L_ff_sparse is not None
            or state.L_ff_inverse_diagonal is not None
            or state.multigrid_hierarchy is not None
        ):
            raise ValueError("dense compatibility projection contains sparse-only state")
        expected_l = torch.zeros(state.n_verts, state.n_verts, dtype=rest.dtype, device=rest.device)
        contribution_scale_l = torch.zeros_like(expected_l)
        rows = state.tets[:, :, None].expand(-1, -1, 4)
        columns = state.tets[:, None, :].expand(-1, 4, -1)
        expected_l.index_put_((rows.reshape(-1), columns.reshape(-1)), stiffness.reshape(-1), accumulate=True)
        contribution_scale_l.index_put_(
            (rows.reshape(-1), columns.reshape(-1)), stiffness.abs().reshape(-1), accumulate=True
        )
        expected_fp = expected_l[state.free][:, state.pinned]
        _require_contribution_scaled_tensor(
            "evaluation projection state dense compatibility operator",
            state.L,
            expected_l,
            contribution_scale_l,
        )
        _require_contribution_scaled_tensor(
            "evaluation projection state dense boundary operator",
            state.L_fp,
            expected_fp,
            contribution_scale_l[state.free][:, state.pinned],
        )
        if not torch.equal(state.L_ff_chol, torch.tril(state.L_ff_chol)) or (state.L_ff_chol.diagonal() <= 0.0).any():
            raise ValueError("evaluation dense compatibility factor is not canonical lower Cholesky form")
        _require_cholesky_product(
            "evaluation projection state dense factored operator",
            state.L_ff_chol,
            state.L[state.free][:, state.free],
        )
        return

    if (
        state.L is not None
        or state.L_ff_chol is not None
        or state.L_ff_sparse is None
        or state.L_ff_inverse_diagonal is None
        or state.multigrid_hierarchy is not None
    ):
        raise ValueError("sparse Jacobi compatibility projection has noncanonical backend state")
    if (
        state.pcg_relative_tolerance != contract.relative_tolerance
        or state.pcg_absolute_tolerance != contract.absolute_tolerance
        or state.pcg_max_iterations != contract.max_iterations
        or state.pcg_raise_on_nonconvergence != contract.raise_on_nonconvergence
        or state.pcg_preconditioner != contract.preconditioner
    ):
        raise ValueError("evaluation sparse projection policy differs from the solver contract")
    materialized_tolerance = torch.as_tensor(contract.relative_tolerance, dtype=rest.dtype, device=rest.device)
    if not torch.isfinite(materialized_tolerance) or materialized_tolerance <= 0.0:
        raise ValueError("evaluation sparse relative tolerance is not representable in the projection dtype")
    expected_ff, expected_fp, expected_inverse_diagonal, ff_contribution_scale, fp_contribution_scale = (
        _canonical_sparse_reduced_operator(
            stiffness,
            state.tets,
            state.free,
            state.pinned,
            state.n_verts,
        )
    )
    _require_contribution_scaled_tensor(
        "evaluation projection state sparse compatibility operator",
        state.L_ff_sparse,
        expected_ff,
        ff_contribution_scale,
    )
    _require_contribution_scaled_tensor(
        "evaluation projection state sparse boundary operator",
        state.L_fp,
        expected_fp,
        fp_contribution_scale,
    )
    _require_reconstructed_tensor(
        "evaluation projection state sparse Jacobi preconditioner",
        state.L_ff_inverse_diagonal,
        expected_inverse_diagonal,
        epsilon_multiplier=4096.0,
    )


def _verify_predictor_static_buffers(
    predictor: StretchPredictor,
    projection_state: SolverState,
    contract: V5SolverContract,
) -> None:
    """Rebuild every graph preprocessing buffer from materialized topology."""
    if type(predictor.model) is not PrincipalStretchGraphTransformer:
        raise ValueError("evaluation predictor model must be the exact PrincipalStretchGraphTransformer type")
    parameter = next(predictor.model.parameters())
    expected_dtype = _V5_PARAMETER_DTYPES[contract.learned_parameter_dtype]
    if parameter.dtype != expected_dtype:
        raise ValueError("evaluation predictor parameter dtype differs from the checkpoint")
    rest = projection_state.source_rest_q.detach().cpu().numpy().astype(np.float64, copy=True)
    tets = projection_state.tets.detach().cpu().numpy().astype(np.int64, copy=True)
    config = GraphTransformerConfig(**contract.graph_config)
    try:
        hierarchy = build_hierarchy(tets, rest, n_levels=config.n_levels, target=config.cluster_size)
        with torch.random.fork_rng(devices=[]):
            expected_model = PrincipalStretchGraphTransformer(
                hierarchy,
                tets,
                rest.shape[0],
                config,
                rest_q=rest,
            ).to(device=parameter.device, dtype=expected_dtype)
    except (RuntimeError, ValueError) as error:
        raise ValueError("evaluation predictor graph preprocessing cannot be canonically rebuilt") from error
    if predictor.model.n_levels != expected_model.n_levels:
        raise ValueError("evaluation predictor hierarchy depth differs from its canonical reconstruction")
    observed_modules = dict(predictor.model.named_modules())
    expected_modules = dict(expected_model.named_modules())
    if set(observed_modules) != set(expected_modules) or any(
        type(observed_modules[name]) is not type(expected_modules[name]) for name in expected_modules
    ):
        raise ValueError("evaluation predictor module tree differs from the canonical architecture-v5 tree")
    observed_buffers = dict(predictor.model.named_buffers(recurse=True))
    expected_buffers = dict(expected_model.named_buffers(recurse=True))
    if set(observed_buffers) != set(expected_buffers):
        raise ValueError("evaluation predictor static buffer schema is noncanonical")
    for name in sorted(expected_buffers):
        observed = observed_buffers[name]
        expected = expected_buffers[name]
        if (
            observed.layout != torch.strided
            or observed.shape != expected.shape
            or observed.dtype != expected.dtype
            or observed.device != expected.device
            or not torch.equal(observed, expected)
        ):
            raise ValueError(
                f"evaluation predictor static graph buffer {name} differs from its canonical reconstruction"
            )


def _verify_predictor_execution_surface(predictor: StretchPredictor) -> None:
    """Reject hooks and instance method overrides that can bypass learned heads."""
    if type(predictor) is not StretchPredictor or type(predictor.model) is not PrincipalStretchGraphTransformer:
        raise ValueError("evaluation predictor and model must use the exact registered architecture-v5 types")
    hook_registries = (
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )

    def reject_instance_overrides(module: torch.nn.Module, label: str) -> None:
        for attribute_name, value in vars(module).items():
            if callable(value) and callable(getattr(type(module), attribute_name, None)):
                raise ValueError(
                    f"evaluation predictor module {label} overrides method {attribute_name!r} on the instance"
                )

    reject_instance_overrides(predictor, "<predictor>")
    for name, module in predictor.named_modules():
        if module.training:
            label = name or "<predictor>"
            raise ValueError(f"evaluation predictor module {label} must be in evaluation mode")
        for registry_name in hook_registries:
            registry = getattr(module, registry_name, None)
            if registry:
                label = name or "<predictor>"
                raise ValueError(f"evaluation predictor module {label} has an active {registry_name} hook")
        reject_instance_overrides(module, name or "<predictor>")

    global_hook_names = (
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
        "_global_buffer_registration_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_pre_hooks",
        "_global_forward_pre_hooks_with_kwargs",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
    )
    torch_module = torch.nn.modules.module
    if any(getattr(torch_module, name, None) for name in global_hook_names):
        raise ValueError("evaluation predictor cannot run while global PyTorch module hooks are active")


@dataclasses.dataclass(frozen=True)
class EvaluationSampleSelection:
    """One selected held-out sample identity."""

    sample_id: str
    sample_sha256: str
    ordinal: int
    dt_seconds: float
    dt_float64_bits: str
    physical_step_sha256: str
    physical_integration_policy: str
    source_integration_evidence_sha256: str | None
    common_objective_sha256: str
    operator_geometry_sha256: str

    def __post_init__(self) -> None:
        _require_string(self.sample_id, "evaluation sample_id")
        _require_sha256(self.sample_sha256, "evaluation sample_sha256")
        _require_nonnegative_integer(self.ordinal, "evaluation sample ordinal")
        dt = _require_finite_float(self.dt_seconds, "evaluation sample dt_seconds", strictly_positive=True)
        expected_bits = f"0x{struct.unpack('<Q', struct.pack('<d', dt))[0]:016x}"
        if self.dt_float64_bits != expected_bits:
            raise ValueError("evaluation sample dt_float64_bits differs from dt_seconds")
        _require_sha256(self.physical_step_sha256, "evaluation sample physical_step_sha256")
        if type(self.physical_integration_policy) is not str or self.physical_integration_policy not in (
            "algebraic-float64-position-history-loads-v1",
            "solver-vbd-staged-float32-v1",
        ):
            raise ValueError("evaluation sample physical integration policy is not registered")
        if self.physical_integration_policy == "algebraic-float64-position-history-loads-v1":
            if self.source_integration_evidence_sha256 is not None:
                raise ValueError("algebraic evaluation sample must not name source integration evidence")
        else:
            if type(self.source_integration_evidence_sha256) is not str:
                raise TypeError("evaluation sample source integration evidence sha256 must be canonical text")
            _require_sha256(
                self.source_integration_evidence_sha256,
                "evaluation sample source integration evidence sha256",
            )
        _require_sha256(self.common_objective_sha256, "evaluation sample common_objective_sha256")
        _require_sha256(self.operator_geometry_sha256, "evaluation sample operator_geometry_sha256")

    def as_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EvaluationSampleSelection:
        """Reconstruct and validate one selected sample."""
        _strict_keys(
            value,
            {
                "sample_id",
                "sample_sha256",
                "ordinal",
                "dt_seconds",
                "dt_float64_bits",
                "physical_step_sha256",
                "physical_integration_policy",
                "source_integration_evidence_sha256",
                "common_objective_sha256",
                "operator_geometry_sha256",
            },
            "evaluation sample selection",
        )
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class V5EvaluationBinding:
    """Independent binding from one checkpoint to held-out runtime inputs."""

    checkpoint_payload_sha256: str
    solver_contract_sha256: str
    split_manifest_sha256: str
    data_access_ledger_sha256: str
    accessed_payload_names: tuple[str, ...]
    split_role: str
    held_out_trajectory_id: str
    held_out_trajectory_sha256: str
    held_out_topology_sha256: str
    held_out_operator_geometry_sha256: str
    projection_state_sha256: str
    static_graph_sha256: str
    selected_samples: tuple[EvaluationSampleSelection, ...]
    physical_dt_seconds: float
    residual_scale: float
    physical_timestep_source: str = _PHYSICAL_TIMESTEP_SOURCE
    selection_sha256: str = dataclasses.field(init=False)
    evaluation_binding_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.checkpoint_payload_sha256, "evaluation checkpoint payload sha256")
        _require_sha256(self.solver_contract_sha256, "evaluation solver contract sha256")
        _require_sha256(self.split_manifest_sha256, "evaluation split manifest sha256")
        _require_sha256(self.data_access_ledger_sha256, "evaluation data-access ledger sha256")
        payload_names = tuple(self.accessed_payload_names)
        if (
            not payload_names
            or len(set(payload_names)) != len(payload_names)
            or payload_names != tuple(sorted(payload_names))
        ):
            raise ValueError("evaluation accessed_payload_names must be non-empty, unique, and sorted")
        object.__setattr__(self, "accessed_payload_names", payload_names)
        if self.split_role not in ("validation", "confirmation"):
            raise ValueError("held-out split role must be 'validation' or 'confirmation'")
        _require_string(self.held_out_trajectory_id, "held-out trajectory id")
        _require_sha256(self.held_out_trajectory_sha256, "held-out trajectory sha256")
        _require_sha256(self.held_out_topology_sha256, "held-out topology sha256")
        _require_sha256(self.held_out_operator_geometry_sha256, "held-out operator-geometry sha256")
        _require_sha256(self.projection_state_sha256, "evaluation projection_state_sha256")
        _require_sha256(self.static_graph_sha256, "evaluation static_graph_sha256")
        selected = tuple(self.selected_samples)
        if not selected or any(not isinstance(sample, EvaluationSampleSelection) for sample in selected):
            raise ValueError("evaluation selection must contain canonical held-out samples")
        if len({sample.sample_id for sample in selected}) != len(selected):
            raise ValueError("evaluation selected sample ids must be unique")
        if any(sample.operator_geometry_sha256 != self.held_out_operator_geometry_sha256 for sample in selected):
            raise ValueError("evaluation sample operator geometry differs from its held-out trajectory")
        object.__setattr__(self, "selected_samples", selected)
        object.__setattr__(
            self,
            "physical_dt_seconds",
            _require_finite_float(self.physical_dt_seconds, "physical_dt_seconds", strictly_positive=True),
        )
        if any(sample.dt_seconds != self.physical_dt_seconds for sample in selected):
            raise ValueError("evaluation physical_dt_seconds differs from a selected sample")
        object.__setattr__(
            self,
            "residual_scale",
            _require_finite_float(self.residual_scale, "residual_scale", strictly_positive=True),
        )
        if self.physical_timestep_source != _PHYSICAL_TIMESTEP_SOURCE:
            raise ValueError(f"physical_timestep_source must be exactly {_PHYSICAL_TIMESTEP_SOURCE!r}")
        selection_digest = canonical_json_sha256([sample.as_dict() for sample in selected])
        object.__setattr__(self, "selection_sha256", selection_digest)
        object.__setattr__(self, "evaluation_binding_sha256", canonical_json_sha256(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": _EVALUATION_SCHEMA_VERSION,
            "contract": _EVALUATION_CONTRACT,
            "checkpoint": {
                "schema_version": _SCHEMA_VERSION,
                "contract": _CHECKPOINT_CONTRACT,
                "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
                "solver_contract_sha256": self.solver_contract_sha256,
            },
            "held_out": {
                "split_manifest_sha256": self.split_manifest_sha256,
                "data_access_ledger_sha256": self.data_access_ledger_sha256,
                "accessed_payload_names": list(self.accessed_payload_names),
                "split_role": self.split_role,
                "trajectory_id": self.held_out_trajectory_id,
                "trajectory_sha256": self.held_out_trajectory_sha256,
                "topology_sha256": self.held_out_topology_sha256,
                "operator_geometry_sha256": self.held_out_operator_geometry_sha256,
                "projection_state_sha256": self.projection_state_sha256,
                "static_graph_sha256": self.static_graph_sha256,
            },
            "selection": [sample.as_dict() for sample in self.selected_samples],
            "selection_sha256": self.selection_sha256,
            "physical_timestep": {
                "source": self.physical_timestep_source,
                "dt_seconds": self.physical_dt_seconds,
                "legacy_graph_dt_policy": _LEGACY_GRAPH_DT_POLICY,
            },
            "residual_scale": {
                "source": _RESIDUAL_SCALE_SOURCE,
                "value": self.residual_scale,
            },
        }

    def as_dict(self) -> dict[str, object]:
        """Return the self-checking evaluation binding."""
        result = self._payload()
        result["evaluation_binding_sha256"] = self.evaluation_binding_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> V5EvaluationBinding:
        """Reconstruct and self-verify a serialized evaluation binding."""
        canonical = _jsonable(value)
        if not isinstance(canonical, dict):
            raise ValueError("evaluation binding must be a mapping")
        digest = canonical.pop("evaluation_binding_sha256", None)
        if digest != canonical_json_sha256(canonical):
            raise ValueError("evaluation-binding SHA-256 verification failed")
        _strict_keys(
            canonical,
            {
                "schema_version",
                "contract",
                "checkpoint",
                "held_out",
                "selection",
                "selection_sha256",
                "physical_timestep",
                "residual_scale",
            },
            "evaluation binding",
        )
        if canonical["schema_version"] != _EVALUATION_SCHEMA_VERSION or canonical["contract"] != _EVALUATION_CONTRACT:
            raise ValueError("unsupported evaluation binding identity")
        checkpoint = canonical["checkpoint"]
        held_out = canonical["held_out"]
        timestep = canonical["physical_timestep"]
        residual_scale = canonical["residual_scale"]
        selection = canonical["selection"]
        if (
            not isinstance(checkpoint, Mapping)
            or not isinstance(held_out, Mapping)
            or not isinstance(timestep, Mapping)
            or not isinstance(residual_scale, Mapping)
        ):
            raise ValueError("evaluation binding has malformed identity metadata")
        if not isinstance(selection, Sequence) or isinstance(selection, (str, bytes)):
            raise ValueError("evaluation selection must be a sequence")
        _strict_keys(
            checkpoint,
            {"schema_version", "contract", "checkpoint_payload_sha256", "solver_contract_sha256"},
            "evaluation checkpoint identity",
        )
        if checkpoint["schema_version"] != _SCHEMA_VERSION or checkpoint["contract"] != _CHECKPOINT_CONTRACT:
            raise ValueError("evaluation binding does not name a schema-v5 checkpoint")
        _strict_keys(
            held_out,
            {
                "split_manifest_sha256",
                "data_access_ledger_sha256",
                "accessed_payload_names",
                "split_role",
                "trajectory_id",
                "trajectory_sha256",
                "topology_sha256",
                "operator_geometry_sha256",
                "projection_state_sha256",
                "static_graph_sha256",
            },
            "held-out identity",
        )
        _strict_keys(
            timestep,
            {"source", "dt_seconds", "legacy_graph_dt_policy"},
            "evaluation physical timestep",
        )
        if timestep["legacy_graph_dt_policy"] != _LEGACY_GRAPH_DT_POLICY:
            raise ValueError("evaluation legacy graph-dt policy changed")
        _strict_keys(residual_scale, {"source", "value"}, "evaluation residual scale")
        if residual_scale["source"] != _RESIDUAL_SCALE_SOURCE:
            raise ValueError("evaluation residual-scale source changed")
        result = cls(
            checkpoint_payload_sha256=checkpoint["checkpoint_payload_sha256"],
            solver_contract_sha256=checkpoint["solver_contract_sha256"],
            split_manifest_sha256=held_out["split_manifest_sha256"],
            data_access_ledger_sha256=held_out["data_access_ledger_sha256"],
            accessed_payload_names=tuple(held_out["accessed_payload_names"]),
            split_role=held_out["split_role"],
            held_out_trajectory_id=held_out["trajectory_id"],
            held_out_trajectory_sha256=held_out["trajectory_sha256"],
            held_out_topology_sha256=held_out["topology_sha256"],
            held_out_operator_geometry_sha256=held_out["operator_geometry_sha256"],
            projection_state_sha256=held_out["projection_state_sha256"],
            static_graph_sha256=held_out["static_graph_sha256"],
            selected_samples=tuple(EvaluationSampleSelection.from_dict(sample) for sample in selection),
            physical_dt_seconds=timestep["dt_seconds"],
            residual_scale=residual_scale["value"],
            physical_timestep_source=timestep["source"],
        )
        if canonical["selection_sha256"] != result.selection_sha256:
            raise ValueError("evaluation selection SHA-256 verification failed")
        if result.as_dict() != value:
            raise ValueError("evaluation binding is not in canonical serialized form")
        return result


@dataclasses.dataclass(frozen=True)
class V6EvaluationBinding(V5EvaluationBinding):
    """Evaluation-binding revision naming one schema-6 checkpoint."""

    def _payload(self) -> dict[str, object]:
        payload = V5EvaluationBinding._payload(self)
        payload["schema_version"] = _EVALUATION_V6_SCHEMA_VERSION
        payload["contract"] = _EVALUATION_V6_CONTRACT
        payload["checkpoint"] = {
            "schema_version": _SCHEMA_V6_VERSION,
            "contract": _CHECKPOINT_V6_CONTRACT,
            "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
            "solver_contract_sha256": self.solver_contract_sha256,
        }
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> V6EvaluationBinding:
        """Reconstruct and self-verify a schema-6 evaluation binding."""
        canonical = _jsonable(value)
        if not isinstance(canonical, dict):
            raise ValueError("schema-6 evaluation binding must be a mapping")
        digest = canonical.pop("evaluation_binding_sha256", None)
        if digest != canonical_json_sha256(canonical):
            raise ValueError("schema-6 evaluation-binding SHA-256 verification failed")
        _strict_keys(
            canonical,
            {
                "schema_version",
                "contract",
                "checkpoint",
                "held_out",
                "selection",
                "selection_sha256",
                "physical_timestep",
                "residual_scale",
            },
            "schema-6 evaluation binding",
        )
        if (
            canonical["schema_version"] != _EVALUATION_V6_SCHEMA_VERSION
            or canonical["contract"] != _EVALUATION_V6_CONTRACT
        ):
            raise ValueError("unsupported schema-6 evaluation binding identity")
        checkpoint = canonical["checkpoint"]
        held_out = canonical["held_out"]
        timestep = canonical["physical_timestep"]
        residual_scale = canonical["residual_scale"]
        selection = canonical["selection"]
        if (
            not isinstance(checkpoint, Mapping)
            or not isinstance(held_out, Mapping)
            or not isinstance(timestep, Mapping)
            or not isinstance(residual_scale, Mapping)
        ):
            raise ValueError("schema-6 evaluation binding has malformed identity metadata")
        if not isinstance(selection, Sequence) or isinstance(selection, (str, bytes)):
            raise ValueError("evaluation selection must be a sequence")
        _strict_keys(
            checkpoint,
            {"schema_version", "contract", "checkpoint_payload_sha256", "solver_contract_sha256"},
            "schema-6 evaluation checkpoint identity",
        )
        if checkpoint["schema_version"] != _SCHEMA_V6_VERSION or checkpoint["contract"] != _CHECKPOINT_V6_CONTRACT:
            raise ValueError("evaluation binding does not name a schema-v6 checkpoint")
        _strict_keys(
            held_out,
            {
                "split_manifest_sha256",
                "data_access_ledger_sha256",
                "accessed_payload_names",
                "split_role",
                "trajectory_id",
                "trajectory_sha256",
                "topology_sha256",
                "operator_geometry_sha256",
                "projection_state_sha256",
                "static_graph_sha256",
            },
            "held-out identity",
        )
        _strict_keys(
            timestep,
            {"source", "dt_seconds", "legacy_graph_dt_policy"},
            "evaluation physical timestep",
        )
        if timestep["legacy_graph_dt_policy"] != _LEGACY_GRAPH_DT_POLICY:
            raise ValueError("evaluation legacy graph-dt policy changed")
        _strict_keys(residual_scale, {"source", "value"}, "evaluation residual scale")
        if residual_scale["source"] != _RESIDUAL_SCALE_SOURCE:
            raise ValueError("evaluation residual-scale source changed")
        result = cls(
            checkpoint_payload_sha256=checkpoint["checkpoint_payload_sha256"],
            solver_contract_sha256=checkpoint["solver_contract_sha256"],
            split_manifest_sha256=held_out["split_manifest_sha256"],
            data_access_ledger_sha256=held_out["data_access_ledger_sha256"],
            accessed_payload_names=tuple(held_out["accessed_payload_names"]),
            split_role=held_out["split_role"],
            held_out_trajectory_id=held_out["trajectory_id"],
            held_out_trajectory_sha256=held_out["trajectory_sha256"],
            held_out_topology_sha256=held_out["topology_sha256"],
            held_out_operator_geometry_sha256=held_out["operator_geometry_sha256"],
            projection_state_sha256=held_out["projection_state_sha256"],
            static_graph_sha256=held_out["static_graph_sha256"],
            selected_samples=tuple(EvaluationSampleSelection.from_dict(sample) for sample in selection),
            physical_dt_seconds=timestep["dt_seconds"],
            residual_scale=residual_scale["value"],
            physical_timestep_source=timestep["source"],
        )
        if canonical["selection_sha256"] != result.selection_sha256:
            raise ValueError("evaluation selection SHA-256 verification failed")
        if result.as_dict() != value:
            raise ValueError("schema-6 evaluation binding is not in canonical serialized form")
        return result


def build_v5_evaluation_binding(
    checkpoint: Mapping[str, object],
    *,
    held_out_trajectory: TrajectoryRecord,
    split_manifest: SplitManifest,
    access_ledger: DataAccessLedger,
    projection_state: SolverState,
    predictor: StretchPredictor,
    selected_sample_ids: Sequence[str],
    physical_dt_seconds: float,
    residual_scale: float,
) -> V5EvaluationBinding:
    """Bind a verified v5 artifact to independent held-out runtime inputs."""
    verified = verify_v5_checkpoint(checkpoint)
    manifest = _verify_split_manifest(split_manifest)
    if manifest.manifest_sha256 != verified.solver_contract.training_dataset_sha256:
        raise ValueError("evaluation SplitManifest differs from the checkpoint's frozen split")
    if type(held_out_trajectory) is not TrajectoryRecord:
        raise ValueError("held_out_trajectory must be a canonical TrajectoryRecord")
    try:
        role = manifest.role_for_trajectory(held_out_trajectory.trajectory_id)
        frozen_trajectory = manifest.trajectory(held_out_trajectory.trajectory_id)
    except ValueError as error:
        raise ValueError("held-out trajectory does not belong to the checkpoint SplitManifest") from error
    if role not in (DatasetRole.VALIDATION, DatasetRole.CONFIRMATION):
        raise ValueError("held-out split role must resolve to validation or confirmation")
    if frozen_trajectory.as_dict() != held_out_trajectory.as_dict():
        raise ValueError("held-out trajectory differs from its frozen SplitManifest record")
    if type(projection_state) is not SolverState:
        raise ValueError("evaluation projection_state must be a SolverState")
    verify_trajectory_topology(
        held_out_trajectory,
        projection_state.source_rest_q.detach().cpu().numpy(),
        projection_state.tets.detach().cpu().numpy(),
    )
    _verify_projection_operator(projection_state, verified.solver_contract.projection)
    actual_projection_sha256 = projection_state_sha256(projection_state)
    if projection_state.projection_state_sha256 != actual_projection_sha256:
        raise ValueError("evaluation projection state differs from its authenticated identity")
    if projection_state.static_mesh_sha256 != held_out_trajectory.topology_sha256:
        raise ValueError("evaluation projection static mesh differs from the held-out topology")
    if projection_state.operator_geometry_sha256 != held_out_trajectory.operator_geometry_sha256:
        raise ValueError("evaluation projection operator geometry differs from the held-out trajectory")
    if type(predictor) is not StretchPredictor:
        raise ValueError("evaluation predictor must be an architecture-v5 StretchPredictor")
    if type(predictor.model) is not PrincipalStretchGraphTransformer:
        raise ValueError("evaluation predictor model must be the exact PrincipalStretchGraphTransformer type")
    if predictor_architecture_version(predictor) != 5:
        raise ValueError("evaluation predictor must be an architecture-v5 StretchPredictor")
    _verify_predictor_execution_surface(predictor)
    if predictor.checkpoint_config().get("graph_transformer") != verified.solver_contract.graph_config:
        raise ValueError("evaluation predictor graph config differs from the checkpoint")
    if learned_state_sha256(predictor.model.state_dict()) != verified.learned_state_sha256:
        raise ValueError("evaluation predictor learned-state SHA-256 differs from the checkpoint")
    if getattr(predictor.model, "static_mesh_sha256", None) != held_out_trajectory.topology_sha256:
        raise ValueError("evaluation predictor static mesh differs from the held-out topology")
    _verify_predictor_static_buffers(predictor, projection_state, verified.solver_contract)
    actual_static_graph_sha256 = predictor.model.compute_static_graph_sha256()
    if getattr(predictor.model, "static_graph_sha256", None) != actual_static_graph_sha256:
        raise ValueError("evaluation predictor static graph differs from its authenticated identity")
    ledger_sha256, payload_names = _verified_evaluation_access(
        access_ledger,
        manifest=manifest,
        trajectory=frozen_trajectory,
        role=role,
    )
    trajectory = _verified_held_out_trajectory(held_out_trajectory)
    identifiers = tuple(selected_sample_ids)
    if not identifiers:
        raise ValueError("held-out evaluation selection must not be empty")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("held-out selected_sample_ids must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("held-out selected_sample_ids must be unique")
    samples = {sample["sample_id"]: sample for sample in trajectory["samples"]}
    missing = [identifier for identifier in identifiers if identifier not in samples]
    if missing:
        raise ValueError(f"held-out selection names unknown sample ids {missing}")
    selected = tuple(
        EvaluationSampleSelection(
            sample_id=identifier,
            sample_sha256=samples[identifier]["sample_sha256"],
            ordinal=samples[identifier]["ordinal"],
            dt_seconds=samples[identifier]["dt_seconds"],
            dt_float64_bits=samples[identifier]["dt_float64_bits"],
            physical_step_sha256=samples[identifier]["physical_step_sha256"],
            physical_integration_policy=samples[identifier]["physical_integration_policy"],
            source_integration_evidence_sha256=samples[identifier]["source_integration_evidence_sha256"],
            common_objective_sha256=samples[identifier]["common_objective_sha256"],
            operator_geometry_sha256=samples[identifier]["operator_geometry_sha256"],
        )
        for identifier in identifiers
    )
    return V5EvaluationBinding(
        checkpoint_payload_sha256=verified.checkpoint_payload_sha256,
        solver_contract_sha256=verified.solver_contract.solver_contract_sha256,
        split_manifest_sha256=manifest.manifest_sha256,
        data_access_ledger_sha256=ledger_sha256,
        accessed_payload_names=payload_names,
        split_role=role.value,
        held_out_trajectory_id=trajectory["trajectory_id"],
        held_out_trajectory_sha256=trajectory["trajectory_sha256"],
        held_out_topology_sha256=trajectory["topology_sha256"],
        held_out_operator_geometry_sha256=trajectory["operator_geometry_sha256"],
        projection_state_sha256=actual_projection_sha256,
        static_graph_sha256=actual_static_graph_sha256,
        selected_samples=selected,
        physical_dt_seconds=physical_dt_seconds,
        residual_scale=residual_scale,
        physical_timestep_source=verified.solver_contract.physical_timestep_source,
    )


def verify_v5_evaluation_binding(
    binding: V5EvaluationBinding | Mapping[str, object],
    *,
    checkpoint: Mapping[str, object],
    held_out_trajectory: TrajectoryRecord,
    split_manifest: SplitManifest,
    access_ledger: DataAccessLedger,
    projection_state: SolverState,
    predictor: StretchPredictor,
) -> V5EvaluationBinding:
    """Verify a binding against both independently authenticated inputs."""
    if isinstance(binding, V5EvaluationBinding):
        canonical_binding = V5EvaluationBinding.from_dict(binding.as_dict())
    elif isinstance(binding, Mapping):
        canonical_binding = V5EvaluationBinding.from_dict(binding)
    else:
        raise TypeError("binding must be a V5EvaluationBinding or mapping")
    rebuilt = build_v5_evaluation_binding(
        checkpoint,
        held_out_trajectory=held_out_trajectory,
        split_manifest=split_manifest,
        access_ledger=access_ledger,
        projection_state=projection_state,
        predictor=predictor,
        selected_sample_ids=tuple(sample.sample_id for sample in canonical_binding.selected_samples),
        physical_dt_seconds=canonical_binding.physical_dt_seconds,
        residual_scale=canonical_binding.residual_scale,
    )
    if rebuilt != canonical_binding:
        raise ValueError("evaluation binding does not match the checkpoint or held-out selection")
    return canonical_binding


def build_v6_evaluation_binding(
    checkpoint: Mapping[str, object],
    *,
    held_out_trajectory: TrajectoryRecord,
    split_manifest: SplitManifest,
    access_ledger: DataAccessLedger,
    projection_state: SolverState,
    predictor: StretchPredictor,
    selected_sample_ids: Sequence[str],
    physical_dt_seconds: float,
    residual_scale: float,
) -> V6EvaluationBinding:
    """Bind a verified schema-6 artifact to held-out runtime inputs."""
    verified = verify_v6_checkpoint(checkpoint)
    manifest = _verify_split_manifest(split_manifest)
    if manifest.manifest_sha256 != verified.solver_contract.training_dataset_sha256:
        raise ValueError("evaluation SplitManifest differs from the checkpoint's frozen split")
    if type(held_out_trajectory) is not TrajectoryRecord:
        raise ValueError("held_out_trajectory must be a canonical TrajectoryRecord")
    try:
        role = manifest.role_for_trajectory(held_out_trajectory.trajectory_id)
        frozen_trajectory = manifest.trajectory(held_out_trajectory.trajectory_id)
    except ValueError as error:
        raise ValueError("held-out trajectory does not belong to the checkpoint SplitManifest") from error
    if role not in (DatasetRole.VALIDATION, DatasetRole.CONFIRMATION):
        raise ValueError("held-out split role must resolve to validation or confirmation")
    if frozen_trajectory.as_dict() != held_out_trajectory.as_dict():
        raise ValueError("held-out trajectory differs from its frozen SplitManifest record")
    if type(projection_state) is not SolverState:
        raise ValueError("evaluation projection_state must be a SolverState")
    verify_trajectory_topology(
        held_out_trajectory,
        projection_state.source_rest_q.detach().cpu().numpy(),
        projection_state.tets.detach().cpu().numpy(),
    )
    _verify_projection_operator(projection_state, verified.solver_contract.projection)
    actual_projection_sha256 = projection_state_sha256(projection_state)
    if projection_state.projection_state_sha256 != actual_projection_sha256:
        raise ValueError("evaluation projection state differs from its authenticated identity")
    if projection_state.static_mesh_sha256 != held_out_trajectory.topology_sha256:
        raise ValueError("evaluation projection static mesh differs from the held-out topology")
    if projection_state.operator_geometry_sha256 != held_out_trajectory.operator_geometry_sha256:
        raise ValueError("evaluation projection operator geometry differs from the held-out trajectory")
    if type(predictor) is not StretchPredictor:
        raise ValueError("evaluation predictor must be an architecture-v5 StretchPredictor")
    if type(predictor.model) is not PrincipalStretchGraphTransformer:
        raise ValueError("evaluation predictor model must be the exact PrincipalStretchGraphTransformer type")
    if predictor_architecture_version(predictor) != 5:
        raise ValueError("evaluation predictor must be an architecture-v5 StretchPredictor")
    _verify_predictor_execution_surface(predictor)
    if predictor.checkpoint_config().get("graph_transformer") != verified.solver_contract.graph_config:
        raise ValueError("evaluation predictor graph config differs from the checkpoint")
    if learned_state_sha256(predictor.model.state_dict()) != verified.learned_state_sha256:
        raise ValueError("evaluation predictor learned-state SHA-256 differs from the checkpoint")
    if getattr(predictor.model, "static_mesh_sha256", None) != held_out_trajectory.topology_sha256:
        raise ValueError("evaluation predictor static mesh differs from the held-out topology")
    _verify_predictor_static_buffers(predictor, projection_state, verified.solver_contract._direct_contract)
    actual_static_graph_sha256 = predictor.model.compute_static_graph_sha256()
    if getattr(predictor.model, "static_graph_sha256", None) != actual_static_graph_sha256:
        raise ValueError("evaluation predictor static graph differs from its authenticated identity")
    ledger_sha256, payload_names = _verified_evaluation_access(
        access_ledger,
        manifest=manifest,
        trajectory=frozen_trajectory,
        role=role,
    )
    trajectory = _verified_held_out_trajectory(held_out_trajectory)
    identifiers = tuple(selected_sample_ids)
    if not identifiers:
        raise ValueError("held-out evaluation selection must not be empty")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("held-out selected_sample_ids must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("held-out selected_sample_ids must be unique")
    samples = {sample["sample_id"]: sample for sample in trajectory["samples"]}
    missing = [identifier for identifier in identifiers if identifier not in samples]
    if missing:
        raise ValueError(f"held-out selection names unknown sample ids {missing}")
    selected = tuple(
        EvaluationSampleSelection(
            sample_id=identifier,
            sample_sha256=samples[identifier]["sample_sha256"],
            ordinal=samples[identifier]["ordinal"],
            dt_seconds=samples[identifier]["dt_seconds"],
            dt_float64_bits=samples[identifier]["dt_float64_bits"],
            physical_step_sha256=samples[identifier]["physical_step_sha256"],
            physical_integration_policy=samples[identifier]["physical_integration_policy"],
            source_integration_evidence_sha256=samples[identifier]["source_integration_evidence_sha256"],
            common_objective_sha256=samples[identifier]["common_objective_sha256"],
            operator_geometry_sha256=samples[identifier]["operator_geometry_sha256"],
        )
        for identifier in identifiers
    )
    return V6EvaluationBinding(
        checkpoint_payload_sha256=verified.checkpoint_payload_sha256,
        solver_contract_sha256=verified.solver_contract.solver_contract_sha256,
        split_manifest_sha256=manifest.manifest_sha256,
        data_access_ledger_sha256=ledger_sha256,
        accessed_payload_names=payload_names,
        split_role=role.value,
        held_out_trajectory_id=trajectory["trajectory_id"],
        held_out_trajectory_sha256=trajectory["trajectory_sha256"],
        held_out_topology_sha256=trajectory["topology_sha256"],
        held_out_operator_geometry_sha256=trajectory["operator_geometry_sha256"],
        projection_state_sha256=actual_projection_sha256,
        static_graph_sha256=actual_static_graph_sha256,
        selected_samples=selected,
        physical_dt_seconds=physical_dt_seconds,
        residual_scale=residual_scale,
        physical_timestep_source=verified.solver_contract.physical_timestep_source,
    )


def verify_v6_evaluation_binding(
    binding: V6EvaluationBinding | Mapping[str, object],
    *,
    checkpoint: Mapping[str, object],
    held_out_trajectory: TrajectoryRecord,
    split_manifest: SplitManifest,
    access_ledger: DataAccessLedger,
    projection_state: SolverState,
    predictor: StretchPredictor,
) -> V6EvaluationBinding:
    """Verify a schema-6 binding against its independent inputs."""
    if type(binding) is V6EvaluationBinding:
        canonical_binding = V6EvaluationBinding.from_dict(binding.as_dict())
    elif isinstance(binding, Mapping):
        canonical_binding = V6EvaluationBinding.from_dict(binding)
    else:
        raise TypeError("binding must be a V6EvaluationBinding or mapping")
    rebuilt = build_v6_evaluation_binding(
        checkpoint,
        held_out_trajectory=held_out_trajectory,
        split_manifest=split_manifest,
        access_ledger=access_ledger,
        projection_state=projection_state,
        predictor=predictor,
        selected_sample_ids=tuple(sample.sample_id for sample in canonical_binding.selected_samples),
        physical_dt_seconds=canonical_binding.physical_dt_seconds,
        residual_scale=canonical_binding.residual_scale,
    )
    if rebuilt != canonical_binding:
        raise ValueError("schema-6 evaluation binding does not match the checkpoint or held-out selection")
    return canonical_binding


@dataclasses.dataclass(frozen=True)
class VerifiedV5Runtime:
    """Integrity/runtime replay evidence, explicitly not contribution evidence."""

    claim_scope: str
    checkpoint_payload_sha256: str
    evaluation_binding_sha256: str
    learned_state_sha256: str
    held_out_topology_sha256: str
    operator_geometry_sha256: str
    projection_state_sha256: str
    static_graph_sha256: str
    physical_integration_policy: str
    source_integration_evidence_sha256: str | None
    physical_step_sha256: str
    common_objective_sha256: str
    iterations: int
    constraint_descriptor_sha256: str
    projection_iterations: int
    projection_matrix_vector_products: int
    projection_preconditioner_applications: int
    projection_factor_solves: int


@dataclasses.dataclass(frozen=True)
class VerifiedV6Runtime(VerifiedV5Runtime):
    """Independent fixed-candidate replay evidence for schema 6."""

    proposal_accepted_iterations: int
    zero_step_iterations: int
    learned_contribution_retained_iterations: int


def _require_same_tensor(
    name: str,
    observed: object,
    expected: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if not isinstance(observed, torch.Tensor):
        raise ValueError(f"runtime {name} must be a torch.Tensor")
    matching = (
        observed.shape == expected.shape and observed.device == expected.device and observed.dtype == expected.dtype
    )
    if matching and (observed.is_floating_point() or observed.is_complex()):
        matching = bool(
            torch.allclose(
                observed,
                expected,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
    elif matching:
        matching = torch.equal(observed, expected)
    if not matching:
        raise ValueError(f"runtime {name} differs from the replayed learned solve")


def _runtime_state_metrics(
    objective: CommonObjectiveContext,
    projection_state: SolverState,
    positions: torch.Tensor,
    *,
    detach_residual: bool,
) -> dict[str, torch.Tensor]:
    raw_residual = common_objective_residual(objective, positions, detach=detach_residual)
    normalized_residual = raw_residual / objective.residual_scale
    deformation_gradient = compute_F(positions, projection_state.tets, projection_state.J)
    determinant = torch.linalg.det(deformation_gradient).amin(dim=-1)
    minimum_singular_value = torch.linalg.svdvals(deformation_gradient).amin(dim=(-2, -1))
    return {
        "normalized_residual": normalized_residual,
        "raw_residual_norm": torch.linalg.vector_norm(raw_residual.flatten(start_dim=-2), dim=-1),
        "normalized_residual_norm": torch.linalg.vector_norm(
            normalized_residual.flatten(start_dim=-2),
            dim=-1,
        ),
        "objective": common_objective_components(objective, positions)["total"],
        "minimum_determinant": determinant,
        "minimum_singular_value": minimum_singular_value,
    }


def _require_finite_v6_state_metrics(scope: str, metrics: Mapping[str, torch.Tensor]) -> None:
    """Fail closed on finite residual entries whose vector norms overflow."""
    for name, value in metrics.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"runtime {scope} state metric {name} is non-finite")


def _require_canonical_v6_work(value: object) -> IterativeSolverWork:
    """Require exact primitive types for every schema-6 work field."""
    if type(value) is not IterativeSolverWork:
        raise ValueError("runtime result work must be an IterativeSolverWork")
    integer_fields = (
        "predictor_passes",
        "projection_calls",
        "residual_evaluations",
        "objective_evaluations",
        "state_validity_evaluations",
        "constraint_preparations",
        "constraint_applications",
        "physical_step_authentications",
        "common_objective_authentications",
    )
    nullable_integer_fields = (
        "projection_iterations",
        "projection_matrix_vector_products",
        "projection_preconditioner_applications",
        "projection_factor_solves",
    )
    if (
        any(type(getattr(value, name)) is not int for name in integer_fields)
        or type(value.projection_backend) is not str
        or type(value.projection_diagnostics_recorded) is not bool
        or any(
            item is not None and type(item) is not int
            for item in (getattr(value, name) for name in nullable_integer_fields)
        )
    ):
        raise ValueError("runtime result work field has a non-canonical primitive type")
    return value


def _require_canonical_v6_projection_diagnostics(value: object) -> ProjectionDiagnostics:
    """Require exact primitive types for every projection diagnostic field."""
    if type(value) is not ProjectionDiagnostics:
        raise ValueError("runtime trace is missing canonical projection diagnostics")
    integer_fields = (
        "iterations",
        "rhs_count",
        "converged_rhs",
        "matrix_vector_products",
        "preconditioner_applications",
        "factor_solves",
        "hierarchy_levels",
        "preconditioner_matrix_vector_products",
    )
    float_fields = (
        "rhs_norm_max",
        "initial_residual_norm_max",
        "residual_norm_max",
        "relative_residual_max",
    )
    if (
        type(value.backend) is not str
        or type(value.converged) is not bool
        or type(value.breakdown) is not bool
        or any(type(getattr(value, name)) is not int for name in integer_fields)
        or any(type(getattr(value, name)) is not float for name in float_fields)
        or (value.relative_tolerance is not None and type(value.relative_tolerance) is not float)
        or (value.absolute_tolerance is not None and type(value.absolute_tolerance) is not float)
        or (value.preconditioner is not None and type(value.preconditioner) is not str)
    ):
        raise ValueError("runtime projection diagnostic field has a non-canonical primitive type")
    return value


def _v6_canonical_primitive_tree_equal(observed: object, expected: object) -> bool:
    """Compare JSON-like evidence without Python bool/numeric aliases."""
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        if len(observed) != len(expected) or any(type(key) is not str for key in observed):
            return False
        return all(
            key in observed and _v6_canonical_primitive_tree_equal(observed[key], value)
            for key, value in expected.items()
        )
    if type(expected) in (tuple, list):
        return len(observed) == len(expected) and all(
            _v6_canonical_primitive_tree_equal(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _projection_residual_metrics(
    state: SolverState,
    target_f: torch.Tensor,
    pinned_targets: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    contrib = torch.einsum("tdc,tac->tad", target_f, state.J) * state.w[:, None, None]
    rhs = torch.zeros(state.n_verts, 3, dtype=state.rest_q.dtype, device=state.rest_q.device)
    rhs.index_add_(0, state.tets.reshape(-1), contrib.reshape(-1, 3))
    if state.projection_backend == "dense":
        boundary_rhs = torch.einsum("fp,pd->fd", state.L_fp, pinned_targets)
        if state.L_ff_chol is None:
            raise ValueError("runtime dense projection is missing its factor")
        applied = state.L_ff_chol @ (state.L_ff_chol.transpose(0, 1) @ positions[state.free])
    else:
        if state.L_ff_sparse is None:
            raise ValueError("runtime sparse projection is missing its reduced operator")
        boundary_rhs = torch.sparse.mm(state.L_fp, pinned_targets)
        applied = torch.sparse.mm(state.L_ff_sparse, positions[state.free])
    reduced_rhs = rhs[state.free] - boundary_rhs
    residual_norm = torch.linalg.vector_norm(applied - reduced_rhs, dim=0)
    rhs_norm = torch.linalg.vector_norm(reduced_rhs, dim=0)
    return residual_norm, rhs_norm


def _require_same_v6_candidate_tensor(
    name: str,
    observed: object,
    expected: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    """Compare candidate replay tensors, including matching rejected NaNs."""
    if not isinstance(observed, torch.Tensor):
        raise ValueError(f"runtime {name} must be a torch.Tensor")
    matching = (
        observed.shape == expected.shape and observed.device == expected.device and observed.dtype == expected.dtype
    )
    if matching and (observed.is_floating_point() or observed.is_complex()):
        matching = bool(
            torch.allclose(
                observed,
                expected,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                equal_nan=True,
            )
        )
        if matching and observed.is_floating_point():
            matching_zero = (observed == 0.0) & (expected == 0.0)
            matching = not bool((matching_zero & (torch.signbit(observed) != torch.signbit(expected))).any().item())
    elif matching:
        matching = torch.equal(observed, expected)
    if not matching:
        raise ValueError(f"runtime {name} differs from the independently replayed candidate")


def _require_exact_v6_trace_tensor(name: str, observed: object, expected: torch.Tensor) -> None:
    """Require bitwise internal consistency between stored schema-6 tensors."""
    if not isinstance(observed, torch.Tensor):
        raise ValueError(f"runtime {name} must be a torch.Tensor")
    if (
        observed.shape != expected.shape
        or observed.device != expected.device
        or observed.dtype != expected.dtype
        or not _tensor_bytes_equal(observed, expected)
    ):
        raise ValueError(f"runtime {name} differs from the exact internal trace evidence")


def _runtime_candidate_metrics(
    *,
    objective: CommonObjectiveContext,
    projection_state: SolverState,
    config: IterativeSolverConfig,
    current: torch.Tensor,
    projected_positions: torch.Tensor,
    constrained_positions: torch.Tensor,
    pinned_targets: torch.Tensor,
    step_fraction: float,
    objective_before: torch.Tensor,
    normalized_residual_norm_before: torch.Tensor,
) -> dict[str, object]:
    """Independently reproduce all core candidate metrics and gates."""
    positions = constrained_positions
    nan = positions.new_full((), torch.nan)
    positions_finite = bool(torch.isfinite(positions).all().item())
    exact_pins = torch.equal(positions[projection_state.pinned], pinned_targets)
    minimum_determinant = nan
    minimum_singular_value = nan
    determinant_valid = False
    singular_value_valid = False
    if positions_finite:
        deformation_gradient = compute_F(positions, projection_state.tets, projection_state.J)
        determinant = torch.linalg.det(deformation_gradient)
        if bool(torch.isfinite(determinant).all().item()):
            minimum_determinant = determinant.amin()
            determinant_valid = bool((determinant > config.minimum_determinant).all().item())
        if bool(torch.isfinite(deformation_gradient).all().item()):
            try:
                singular_values = torch.linalg.svdvals(deformation_gradient)
            except RuntimeError:
                singular_values = None
            if singular_values is not None and bool(torch.isfinite(singular_values).all().item()):
                minimum_singular_value = singular_values.amin()
                singular_value_valid = bool((singular_values > config.minimum_singular_value).all().item())
    state_valid = positions_finite and exact_pins and determinant_valid and singular_value_valid
    raw_residual = _common_objective_residual_trusted(
        objective,
        positions,
        detach=config.detach_residual_features,
    )
    normalized_residual = raw_residual / objective.residual_scale
    raw_residual_norm = torch.linalg.vector_norm(raw_residual.flatten(start_dim=-2), dim=-1)
    normalized_residual_norm = torch.linalg.vector_norm(normalized_residual.flatten(start_dim=-2), dim=-1)
    objective_value = _common_objective_components_trusted(objective, positions)["total"]
    objective_finite = bool(torch.isfinite(objective_value).all().item())
    residual_finite = (
        bool(torch.isfinite(normalized_residual).all().item())
        and bool(torch.isfinite(raw_residual_norm).all().item())
        and bool(torch.isfinite(normalized_residual_norm).all().item())
    )
    objective_nonincreasing = objective_finite and bool(
        (objective_value <= objective_before + objective_value.new_tensor(config.objective_increase_tolerance))
        .all()
        .item()
    )
    residual_nonincreasing = residual_finite and bool(
        (
            normalized_residual_norm
            <= normalized_residual_norm_before
            + normalized_residual_norm.new_tensor(config.normalized_residual_increase_tolerance)
        )
        .all()
        .item()
    )
    zero_step_unchanged = _tensor_bytes_equal(positions, current) if step_fraction == 0.0 else None
    free = torch.ones(projection_state.n_verts, dtype=torch.bool, device=current.device)
    free[projection_state.pinned] = False
    full_displacement = (projected_positions - current)[free].reshape(-1)
    constrained_displacement = (positions - current)[free].reshape(-1)
    full_displacement_finite = bool(torch.isfinite(full_displacement).all().item())
    constrained_displacement_finite = bool(torch.isfinite(constrained_displacement).all().item())
    if (
        not full_displacement_finite
        or not constrained_displacement_finite
        or torch.equal(full_displacement, torch.zeros_like(full_displacement))
    ):
        displacement_retention = None
    else:
        retention_numerator = torch.dot(constrained_displacement, full_displacement)
        retention_denominator = torch.dot(full_displacement, full_displacement)
        if (
            not bool(torch.isfinite(retention_numerator).item())
            or not bool(torch.isfinite(retention_denominator).item())
            or not bool((retention_denominator > 0.0).item())
        ):
            displacement_retention = None
        else:
            candidate_retention = retention_numerator / retention_denominator
            displacement_retention = candidate_retention if bool(torch.isfinite(candidate_retention).item()) else None
    learned_contribution_retained = (
        step_fraction > 0.0
        and displacement_retention is not None
        and bool(torch.isfinite(displacement_retention).item())
        and bool((displacement_retention > 0.0).item())
    )
    rejection_reasons: list[str] = []
    if not positions_finite:
        rejection_reasons.append("non-finite-positions")
    if not exact_pins:
        rejection_reasons.append("changed-exact-pins")
    if not determinant_valid:
        rejection_reasons.append("determinant-bound")
    if not singular_value_valid:
        rejection_reasons.append("singular-value-bound")
    if not objective_finite:
        rejection_reasons.append("non-finite-objective")
    if not residual_finite:
        rejection_reasons.append("non-finite-residual")
    if not objective_nonincreasing:
        rejection_reasons.append("objective-increase")
    if not residual_nonincreasing:
        rejection_reasons.append("residual-increase")
    if zero_step_unchanged is False:
        rejection_reasons.append("zero-step-moved")
    admissible = state_valid and objective_nonincreasing and residual_nonincreasing and zero_step_unchanged is not False
    return {
        "normalized_residual": normalized_residual,
        "raw_residual_norm": raw_residual_norm,
        "normalized_residual_norm": normalized_residual_norm,
        "objective": objective_value,
        "minimum_determinant": minimum_determinant,
        "minimum_singular_value": minimum_singular_value,
        "positions_finite": positions_finite,
        "exact_pins": exact_pins,
        "determinant_valid": determinant_valid,
        "singular_value_valid": singular_value_valid,
        "objective_finite": objective_finite,
        "residual_finite": residual_finite,
        "state_valid": state_valid,
        "objective_nonincreasing": objective_nonincreasing,
        "residual_nonincreasing": residual_nonincreasing,
        "zero_step_unchanged": zero_step_unchanged,
        "displacement_retention": displacement_retention,
        "learned_contribution_retained": learned_contribution_retained,
        "admissible": admissible,
        "rejection_reasons": tuple(rejection_reasons),
    }


def verify_v5_runtime_compatibility(
    checkpoint: Mapping[str, object],
    *,
    evaluation_binding: V5EvaluationBinding | Mapping[str, object],
    held_out_trajectory: TrajectoryRecord,
    split_manifest: SplitManifest,
    access_ledger: DataAccessLedger,
    predictor: object,
    solver_config: object,
    projection_state: object,
    constraint: object,
    objective: object,
    physical_step: object,
    result: object,
) -> VerifiedV5Runtime:
    """Authenticate one development solve replay without claiming learned value."""
    verified = verify_v5_checkpoint(checkpoint)
    binding = verify_v5_evaluation_binding(
        evaluation_binding,
        checkpoint=checkpoint,
        held_out_trajectory=held_out_trajectory,
        split_manifest=split_manifest,
        access_ledger=access_ledger,
        projection_state=projection_state,
        predictor=predictor,
    )
    if len(binding.selected_samples) != 1:
        raise ValueError("one runtime solve must bind exactly one held-out sample")
    selected_sample = binding.selected_samples[0]
    contract = verified.solver_contract

    def require_same(name: str, observed: object, expected: torch.Tensor) -> None:
        _require_same_tensor(
            name,
            observed,
            expected,
            relative_tolerance=contract.safeguards.replay_relative_tolerance,
            absolute_tolerance=contract.safeguards.replay_absolute_tolerance,
        )

    if type(predictor) is not StretchPredictor or predictor_architecture_version(predictor) != 5:
        raise ValueError("runtime predictor must be an architecture-v5 StretchPredictor")
    if predictor.model.training:
        raise ValueError("runtime replay verification requires predictor evaluation mode")
    predictor_config = predictor.checkpoint_config()
    if predictor_config.get("kind") != "graph-transformer" or predictor_config.get("residual") is not True:
        raise ValueError("runtime predictor is not the residual graph-transformer route")
    if predictor_config.get("graph_transformer") != contract.graph_config:
        raise ValueError("runtime predictor graph config differs from the checkpoint contract")
    if str(next(predictor.model.parameters()).dtype) != contract.learned_parameter_dtype:
        raise ValueError("runtime predictor parameter dtype differs from the checkpoint contract")
    runtime_state_sha256 = learned_state_sha256(predictor.model.state_dict())
    if runtime_state_sha256 != verified.learned_state_sha256:
        raise ValueError("runtime predictor learned-state SHA-256 differs from the checkpoint")

    if type(solver_config) is not IterativeSolverConfig:
        raise ValueError("runtime solver_config must be IterativeSolverConfig")
    if solver_config.iterations != contract.inference_iterations:
        raise ValueError("runtime solver iteration count differs from inference K")
    if solver_config.detach_residual_features != contract.residual.detach_features:
        raise ValueError("runtime residual detach policy differs from the checkpoint")
    safeguard_fields = (
        "minimum_determinant",
        "minimum_singular_value",
        "objective_policy",
        "residual_policy",
        "objective_increase_tolerance",
        "normalized_residual_increase_tolerance",
        "initializer_policy",
    )
    if any(getattr(solver_config, name) != getattr(contract.safeguards, name) for name in safeguard_fields):
        raise ValueError("runtime safeguard policy differs from the checkpoint")
    if solver_config.return_projection_diagnostics != contract.projection.require_runtime_diagnostics:
        raise ValueError("runtime projection-diagnostics policy differs from the checkpoint")
    if solver_config.head_mode != "learned" or solver_config.head_permutation is not None:
        raise ValueError("runtime replay verification requires the learned, unpermuted heads")

    if type(projection_state) is not SolverState:
        raise ValueError("runtime projection_state must be a SolverState")
    if projection_state.static_mesh_sha256 != binding.held_out_topology_sha256:
        raise ValueError("runtime static mesh differs from the held-out topology")
    actual_operator_sha256 = validate_authenticated_operator_geometry(projection_state)
    if actual_operator_sha256 != binding.held_out_operator_geometry_sha256:
        raise ValueError("runtime operator geometry differs from the held-out evaluation binding")
    if getattr(predictor.model, "static_mesh_sha256", None) != binding.held_out_topology_sha256:
        raise ValueError("runtime predictor static mesh differs from the held-out topology")
    actual_projection_sha256 = projection_state_sha256(projection_state)
    if (
        projection_state.projection_state_sha256 != actual_projection_sha256
        or actual_projection_sha256 != binding.projection_state_sha256
    ):
        raise ValueError("runtime projection state differs from its evaluation binding")
    actual_static_graph_sha256 = predictor.model.compute_static_graph_sha256()
    if (
        getattr(predictor.model, "static_graph_sha256", None) != actual_static_graph_sha256
        or actual_static_graph_sha256 != binding.static_graph_sha256
    ):
        raise ValueError("runtime predictor static graph differs from its evaluation binding")
    projection = contract.projection
    if projection_state.tikhonov != 0.0 or projection_state.projection_backend != projection.backend:
        raise ValueError("runtime projection backend or regularization differs from the checkpoint")
    if projection.backend == "sparse_pcg" and (
        projection_state.pcg_relative_tolerance != projection.relative_tolerance
        or projection_state.pcg_absolute_tolerance != projection.absolute_tolerance
        or projection_state.pcg_max_iterations != projection.max_iterations
        or projection_state.pcg_raise_on_nonconvergence != projection.raise_on_nonconvergence
        or projection_state.pcg_preconditioner != projection.preconditioner
    ):
        raise ValueError("runtime sparse projection policy differs from the checkpoint")

    if type(constraint) is not IdentityConstraintHook:
        raise ValueError("runtime constraint must be the exact registered IdentityConstraintHook")
    descriptor = _jsonable(constraint.descriptor())
    if descriptor != contract.constraint.descriptor:
        raise ValueError("runtime constraint descriptor differs from the checkpoint")
    descriptor_sha256 = canonical_json_sha256(descriptor)
    if descriptor_sha256 != contract.constraint.descriptor_sha256:
        raise ValueError("runtime constraint descriptor SHA-256 differs from the checkpoint")

    if type(objective) is not CommonObjectiveContext:
        raise ValueError("runtime objective must be a CommonObjectiveContext")
    objective.validate_immutable()
    if objective.common_objective_sha256 != selected_sample.common_objective_sha256:
        raise ValueError("runtime common objective differs from the selected sample")
    if objective.dt != binding.physical_dt_seconds:
        raise ValueError("runtime common-objective dt differs from the evaluation binding")
    if objective.residual_scale != binding.residual_scale:
        raise ValueError("runtime common-objective residual scale differs from the evaluation binding")

    if type(physical_step) is not PhysicalStepContext:
        raise ValueError("runtime physical_step must be a PhysicalStepContext")
    physical_step.validate_immutable()
    if physical_step.physical_step_sha256 != selected_sample.physical_step_sha256:
        raise ValueError("runtime physical step differs from the selected sample")
    if physical_step.integration_policy != selected_sample.physical_integration_policy:
        raise ValueError("runtime physical integration policy differs from the selected sample")
    source_evidence_sha256 = (
        None if physical_step.source_evidence is None else physical_step.source_evidence.evidence_sha256
    )
    if source_evidence_sha256 != selected_sample.source_integration_evidence_sha256:
        raise ValueError("runtime source integration evidence differs from the selected sample")
    validate_physical_objective_integration(projection_state, objective, physical_step)

    if type(result) is not IterativeSolverResult:
        raise ValueError("runtime result must be an IterativeSolverResult")
    expected_work = contract.inference_work
    observed_work = {
        "predictor_passes": result.work.predictor_passes,
        "global_compatibility_projections": result.work.projection_calls,
        "common_residual_evaluations": result.work.residual_evaluations,
        "common_objective_evaluations": result.work.objective_evaluations,
        "state_validity_evaluations": result.work.state_validity_evaluations,
        "physical_step_authentications": result.work.physical_step_authentications,
        "common_objective_authentications": result.work.common_objective_authentications,
        "constraint_preparations": result.work.constraint_preparations,
        "constraint_applications": result.work.constraint_applications,
    }
    for name, observed in observed_work.items():
        if observed != expected_work[name]:
            raise ValueError(f"runtime observed work {name} differs from the checkpoint")
    if len(result.trace) != contract.inference_iterations:
        raise ValueError("runtime trace length differs from inference K")
    if result.constraint_descriptor != descriptor or result.constraint_descriptor_sha256 != descriptor_sha256:
        raise ValueError("runtime result constraint descriptor differs from the invoked hook")
    if result.constraint_registration != "registered-identity-development":
        raise ValueError("runtime result does not carry the registered identity-constraint scope")
    if result.head_mode != "learned" or result.head_permutation is not None:
        raise ValueError("runtime result did not use the learned, unpermuted heads")
    if (
        type(result.physical_integration_policy) is not str
        or result.physical_integration_policy != physical_step.integration_policy
    ):
        raise ValueError("runtime result physical integration policy differs from the invoked solve")
    if (
        result.source_integration_evidence_sha256 is not None
        and type(result.source_integration_evidence_sha256) is not str
    ) or result.source_integration_evidence_sha256 != source_evidence_sha256:
        raise ValueError("runtime result source integration evidence differs from the invoked solve")
    if result.physical_step_sha256 != physical_step.physical_step_sha256:
        raise ValueError("runtime result physical-step identity differs from the invoked solve")
    if result.common_objective_sha256 != objective.common_objective_sha256:
        raise ValueError("runtime result common-objective identity differs from the invoked solve")
    if result.operator_geometry_sha256 != actual_operator_sha256:
        raise ValueError("runtime result operator-geometry identity differs from the invoked solve")
    if result.projection_state_sha256 != actual_projection_sha256:
        raise ValueError("runtime result projection-state identity differs from the invoked solve")
    if result.static_graph_sha256 != actual_static_graph_sha256:
        raise ValueError("runtime result static-graph identity differs from the invoked solve")
    if result.work.projection_backend != projection.backend or not result.work.projection_diagnostics_recorded:
        raise ValueError("runtime result lacks required projection diagnostics")

    x_current = physical_step.x_current
    if x_current.ndim != 2:
        raise ValueError("held-out runtime replay verification is one unbatched sample at a time")
    if (
        result.positions.shape != x_current.shape
        or result.positions.device != objective.device
        or result.positions.dtype != objective.dtype
    ):
        raise ValueError("runtime result position shape, device, or dtype differs from the bound sample")
    targets = physical_step.pinned_targets
    if targets.shape != (projection_state.pinned.numel(), 3):
        raise ValueError("runtime physical-step pinned targets have the wrong unbatched shape")
    if not torch.isfinite(x_current).all() or not torch.equal(x_current[projection_state.pinned], targets):
        raise ValueError("runtime persistence initializer is non-finite or violates exact pinned targets")
    initial = _runtime_state_metrics(
        objective,
        projection_state,
        x_current,
        detach_residual=solver_config.detach_residual_features,
    )
    if (initial["minimum_determinant"] <= contract.safeguards.minimum_determinant).any() or (
        initial["minimum_singular_value"] <= contract.safeguards.minimum_singular_value
    ).any():
        raise ValueError("runtime persistence initializer violates a state-validity safeguard")

    network_force = physical_step.force.clone()
    network_force[projection_state.pinned] = 0.0
    previous_positions = x_current
    projection_diagnostics: list[ProjectionDiagnostics] = []
    for iteration, trace in enumerate(result.trace):
        if trace.iteration != iteration or trace.iteration_fraction != iteration / max(len(result.trace) - 1, 1):
            raise ValueError("runtime trace iteration order or fraction differs from fixed K")
        require_same(f"trace[{iteration}].positions_before", trace.positions_before, previous_positions)
        before = _runtime_state_metrics(
            objective,
            projection_state,
            previous_positions,
            detach_residual=solver_config.detach_residual_features,
        )
        for name in (
            "normalized_residual_before",
            "raw_residual_norm_before",
            "normalized_residual_norm_before",
            "objective_before",
            "minimum_determinant_before",
            "minimum_singular_value_before",
        ):
            metric_name = name.removesuffix("_before")
            require_same(f"trace[{iteration}].{name}", getattr(trace, name), before[metric_name])

        zeros_normal = torch.zeros_like(previous_positions)
        zeros_slack = torch.zeros(
            previous_positions.shape[:-1],
            dtype=previous_positions.dtype,
            device=previous_positions.device,
        )
        replay_target, replay_delta_h, replay_omega = predictor.predict_principal_stretch_update(
            projection_state,
            physical_step.x_current,
            physical_step.x_previous,
            previous_positions,
            network_force,
            physical_step.gravity,
            physical_step.mu,
            physical_step.lam,
            physical_step.pin,
            before["normalized_residual"],
            zeros_normal,
            zeros_slack,
            iteration_fraction=trace.iteration_fraction,
            physical_dt=objective.dt,
            head_mode="learned",
        )
        require_same(f"trace[{iteration}].delta_h", trace.delta_h, replay_delta_h)
        require_same(f"trace[{iteration}].omega", trace.omega, replay_omega)
        require_same(
            f"trace[{iteration}].target_deformation_gradient",
            trace.target_deformation_gradient,
            replay_target,
        )

        projection_kwargs: dict[str, object] = {}
        if projection.backend == "sparse_pcg":
            projection_kwargs["initial_positions"] = previous_positions
        replay_projection = project_deformation_gradient(
            projection_state,
            trace.target_deformation_gradient,
            targets,
            return_diagnostics=True,
            **projection_kwargs,
        )
        if not isinstance(replay_projection, tuple):
            raise RuntimeError("diagnostic projection replay did not return diagnostics")
        replay_proposed, replay_diagnostics = replay_projection
        require_same(f"trace[{iteration}].proposed_positions", trace.proposed_positions, replay_proposed)
        if type(trace.projection_diagnostics) is not ProjectionDiagnostics:
            raise ValueError("runtime trace is missing canonical projection diagnostics")
        diagnostics = trace.projection_diagnostics
        if (
            not diagnostics.converged
            or diagnostics.breakdown
            or diagnostics.backend != projection.backend
            or diagnostics.iterations < 0
            or diagnostics.iterations > projection.max_iterations
            or diagnostics.relative_tolerance != projection.relative_tolerance
            or diagnostics.absolute_tolerance != projection.absolute_tolerance
            or diagnostics.preconditioner != (None if projection.backend == "dense" else projection.preconditioner)
            or diagnostics.rhs_count != 3
            or diagnostics.converged_rhs != diagnostics.rhs_count
            or min(
                diagnostics.matrix_vector_products,
                diagnostics.preconditioner_applications,
                diagnostics.factor_solves,
                diagnostics.preconditioner_matrix_vector_products,
            )
            < 0
        ):
            raise ValueError("runtime projection diagnostics violate the checkpoint contract")
        diagnostic_scalars = (
            diagnostics.rhs_norm_max,
            diagnostics.initial_residual_norm_max,
            diagnostics.residual_norm_max,
            diagnostics.relative_residual_max,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in diagnostic_scalars):
            raise ValueError("runtime projection diagnostics contain an invalid residual metric")
        diagnostic_scalar_names = {
            "rhs_norm_max",
            "initial_residual_norm_max",
            "residual_norm_max",
            "relative_residual_max",
        }
        for field in dataclasses.fields(ProjectionDiagnostics):
            if field.name in diagnostic_scalar_names:
                if not math.isclose(
                    getattr(diagnostics, field.name),
                    getattr(replay_diagnostics, field.name),
                    rel_tol=contract.safeguards.replay_relative_tolerance,
                    abs_tol=contract.safeguards.replay_absolute_tolerance,
                ):
                    raise ValueError(f"runtime projection diagnostic {field.name} differs from the exact work replay")
            elif getattr(diagnostics, field.name) != getattr(replay_diagnostics, field.name):
                raise ValueError(f"runtime projection diagnostic {field.name} differs from the exact work replay")
        residual_norm, rhs_norm = _projection_residual_metrics(
            projection_state,
            trace.target_deformation_gradient,
            targets,
            trace.proposed_positions,
        )
        if projection.backend == "sparse_pcg":
            threshold = projection.absolute_tolerance + projection.relative_tolerance * rhs_norm
        else:
            threshold = (
                contract.safeguards.replay_absolute_tolerance + contract.safeguards.replay_relative_tolerance * rhs_norm
            )
        replay_slack = (
            contract.safeguards.replay_absolute_tolerance + contract.safeguards.replay_relative_tolerance * rhs_norm
        )
        if (residual_norm > threshold + replay_slack).any():
            raise ValueError("runtime projected proposal violates the authenticated normal-equation tolerance")
        rhs_norm_max = float(rhs_norm.detach().max())
        residual_norm_max = float(residual_norm.detach().max())
        relative = torch.where(rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm)
        relative_residual_max = float(relative.detach().max())
        for name, observed, recomputed in (
            ("rhs_norm_max", diagnostics.rhs_norm_max, rhs_norm_max),
            ("residual_norm_max", diagnostics.residual_norm_max, residual_norm_max),
            ("relative_residual_max", diagnostics.relative_residual_max, relative_residual_max),
        ):
            if not math.isclose(
                observed,
                recomputed,
                rel_tol=contract.safeguards.replay_relative_tolerance,
                abs_tol=contract.safeguards.replay_absolute_tolerance,
            ):
                raise ValueError(f"runtime projection diagnostic {name} differs from the observed proposal")
        projection_diagnostics.append(diagnostics)

        expected_committed = trace.proposed_positions.index_copy(-2, projection_state.pinned, targets)
        require_same(f"trace[{iteration}].positions", trace.positions, expected_committed)
        if not torch.equal(trace.proposed_positions[projection_state.pinned], targets) or not torch.equal(
            trace.positions,
            trace.proposed_positions,
        ):
            raise ValueError("runtime identity constraint did not commit the exact projected proposal and pins")
        if trace.constraint_prepare_diagnostics != {"refreshes": 0} or trace.constraint_diagnostics != {
            "truncation_calls": 0,
            "minimum_fraction": 1.0,
        }:
            raise ValueError("runtime identity-constraint diagnostics are not canonical")
        after = _runtime_state_metrics(
            objective,
            projection_state,
            trace.positions,
            detach_residual=solver_config.detach_residual_features,
        )
        for name in (
            "residual_after",
            "raw_residual_norm_after",
            "normalized_residual_norm_after",
            "objective_after",
            "minimum_determinant_after",
            "minimum_singular_value_after",
        ):
            metric_name = "normalized_residual" if name == "residual_after" else name.removesuffix("_after")
            require_same(f"trace[{iteration}].{name}", getattr(trace, name), after[metric_name])
        if (after["minimum_determinant"] <= contract.safeguards.minimum_determinant).any() or (
            after["minimum_singular_value"] <= contract.safeguards.minimum_singular_value
        ).any():
            raise ValueError("runtime trace violates the determinant or singular-value safeguard")
        if (after["objective"] > before["objective"] + contract.safeguards.objective_increase_tolerance).any() or (
            after["normalized_residual_norm"]
            > before["normalized_residual_norm"] + contract.safeguards.normalized_residual_increase_tolerance
        ).any():
            raise ValueError("runtime trace violates the active nonincrease safeguard")
        previous_positions = trace.positions

    final = _runtime_state_metrics(
        objective,
        projection_state,
        previous_positions,
        detach_residual=solver_config.detach_residual_features,
    )
    require_same("result.positions", result.positions, previous_positions)
    if not torch.equal(result.positions, previous_positions):
        raise ValueError("runtime final positions do not equal the last committed trace state")
    for name in (
        "normalized_residual",
        "objective",
        "raw_residual_norm",
        "normalized_residual_norm",
        "minimum_determinant",
        "minimum_singular_value",
    ):
        require_same(f"result.{name}", getattr(result, name), final[name])
    aggregate = {
        "projection_iterations": sum(item.iterations for item in projection_diagnostics),
        "projection_matrix_vector_products": sum(item.matrix_vector_products for item in projection_diagnostics),
        "projection_preconditioner_applications": sum(
            item.preconditioner_applications for item in projection_diagnostics
        ),
        "projection_factor_solves": sum(item.factor_solves for item in projection_diagnostics),
    }
    if any(getattr(result.work, name) != count for name, count in aggregate.items()):
        raise ValueError("runtime aggregate projection work differs from its per-iteration diagnostics")
    _verify_predictor_execution_surface(predictor)
    return VerifiedV5Runtime(
        claim_scope=_RUNTIME_CLAIM_SCOPE,
        checkpoint_payload_sha256=verified.checkpoint_payload_sha256,
        evaluation_binding_sha256=binding.evaluation_binding_sha256,
        learned_state_sha256=runtime_state_sha256,
        held_out_topology_sha256=binding.held_out_topology_sha256,
        operator_geometry_sha256=actual_operator_sha256,
        projection_state_sha256=actual_projection_sha256,
        static_graph_sha256=actual_static_graph_sha256,
        physical_integration_policy=physical_step.integration_policy,
        source_integration_evidence_sha256=source_evidence_sha256,
        physical_step_sha256=physical_step.physical_step_sha256,
        common_objective_sha256=objective.common_objective_sha256,
        iterations=solver_config.iterations,
        constraint_descriptor_sha256=descriptor_sha256,
        **aggregate,
    )


def verify_v6_runtime_compatibility(
    checkpoint: Mapping[str, object],
    *,
    evaluation_binding: V6EvaluationBinding | Mapping[str, object],
    held_out_trajectory: TrajectoryRecord,
    split_manifest: SplitManifest,
    access_ledger: DataAccessLedger,
    predictor: object,
    solver_config: object,
    projection_state: object,
    constraint: object,
    objective: object,
    physical_step: object,
    result: object,
) -> VerifiedV6Runtime:
    """Authenticate every fixed candidate and deterministic selection."""
    verified = verify_v6_checkpoint(checkpoint)
    binding = verify_v6_evaluation_binding(
        evaluation_binding,
        checkpoint=checkpoint,
        held_out_trajectory=held_out_trajectory,
        split_manifest=split_manifest,
        access_ledger=access_ledger,
        projection_state=projection_state,
        predictor=predictor,
    )
    if len(binding.selected_samples) != 1:
        raise ValueError("one runtime solve must bind exactly one held-out sample")
    selected_sample = binding.selected_samples[0]
    contract = verified.solver_contract

    def require_same(name: str, observed: object, expected: torch.Tensor) -> None:
        _require_same_tensor(
            name,
            observed,
            expected,
            relative_tolerance=contract.safeguards.replay_relative_tolerance,
            absolute_tolerance=contract.safeguards.replay_absolute_tolerance,
        )

    def require_same_candidate(name: str, observed: object, expected: torch.Tensor) -> None:
        _require_same_v6_candidate_tensor(
            name,
            observed,
            expected,
            relative_tolerance=contract.safeguards.replay_relative_tolerance,
            absolute_tolerance=contract.safeguards.replay_absolute_tolerance,
        )

    if type(predictor) is not StretchPredictor or predictor_architecture_version(predictor) != 5:
        raise ValueError("runtime predictor must be an architecture-v5 StretchPredictor")
    if predictor.model.training:
        raise ValueError("runtime replay verification requires predictor evaluation mode")
    predictor_config = predictor.checkpoint_config()
    if predictor_config.get("kind") != "graph-transformer" or predictor_config.get("residual") is not True:
        raise ValueError("runtime predictor is not the residual graph-transformer route")
    if predictor_config.get("graph_transformer") != contract.graph_config:
        raise ValueError("runtime predictor graph config differs from the checkpoint contract")
    if str(next(predictor.model.parameters()).dtype) != contract.learned_parameter_dtype:
        raise ValueError("runtime predictor parameter dtype differs from the checkpoint contract")
    runtime_state_sha256 = learned_state_sha256(predictor.model.state_dict())
    if runtime_state_sha256 != verified.learned_state_sha256:
        raise ValueError("runtime predictor learned-state SHA-256 differs from the checkpoint")

    if type(solver_config) is not IterativeSolverConfig:
        raise ValueError("runtime solver_config must be IterativeSolverConfig")
    if solver_config.iterations != contract.inference_iterations:
        raise ValueError("runtime solver iteration count differs from inference K")
    if solver_config.detach_residual_features != contract.residual.detach_features:
        raise ValueError("runtime residual detach policy differs from the checkpoint")
    safeguard_fields = (
        "minimum_determinant",
        "minimum_singular_value",
        "objective_policy",
        "residual_policy",
        "objective_increase_tolerance",
        "normalized_residual_increase_tolerance",
        "initializer_policy",
    )
    if any(getattr(solver_config, name) != getattr(contract.safeguards, name) for name in safeguard_fields):
        raise ValueError("runtime safeguard policy differs from the checkpoint")
    if solver_config.return_projection_diagnostics != contract.projection.require_runtime_diagnostics:
        raise ValueError("runtime projection-diagnostics policy differs from the checkpoint")
    if solver_config.head_mode != "learned" or solver_config.head_permutation is not None:
        raise ValueError("runtime replay verification requires the learned, unpermuted heads")
    solver_config.validate()
    if type(solver_config.proposal_safeguard) is not ProposalSafeguardConfig:
        raise ValueError("schema-6 runtime requires the exact ProposalSafeguardConfig type")
    runtime_proposal = ProposalSafeguardContract.from_config(solver_config.proposal_safeguard)
    if runtime_proposal != contract.inference_proposal_safeguard:
        raise ValueError("runtime proposal safeguard differs from the schema-6 solver contract")
    candidate_fractions = runtime_proposal.candidate_step_fractions

    if type(projection_state) is not SolverState:
        raise ValueError("runtime projection_state must be a SolverState")
    if projection_state.static_mesh_sha256 != binding.held_out_topology_sha256:
        raise ValueError("runtime static mesh differs from the held-out topology")
    actual_operator_sha256 = validate_authenticated_operator_geometry(projection_state)
    if actual_operator_sha256 != binding.held_out_operator_geometry_sha256:
        raise ValueError("runtime operator geometry differs from the held-out evaluation binding")
    if getattr(predictor.model, "static_mesh_sha256", None) != binding.held_out_topology_sha256:
        raise ValueError("runtime predictor static mesh differs from the held-out topology")
    actual_projection_sha256 = projection_state_sha256(projection_state)
    if (
        projection_state.projection_state_sha256 != actual_projection_sha256
        or actual_projection_sha256 != binding.projection_state_sha256
    ):
        raise ValueError("runtime projection state differs from its evaluation binding")
    actual_static_graph_sha256 = predictor.model.compute_static_graph_sha256()
    if (
        getattr(predictor.model, "static_graph_sha256", None) != actual_static_graph_sha256
        or actual_static_graph_sha256 != binding.static_graph_sha256
    ):
        raise ValueError("runtime predictor static graph differs from its evaluation binding")
    projection = contract.projection
    if projection_state.tikhonov != 0.0 or projection_state.projection_backend != projection.backend:
        raise ValueError("runtime projection backend or regularization differs from the checkpoint")
    if projection.backend == "sparse_pcg" and (
        projection_state.pcg_relative_tolerance != projection.relative_tolerance
        or projection_state.pcg_absolute_tolerance != projection.absolute_tolerance
        or projection_state.pcg_max_iterations != projection.max_iterations
        or projection_state.pcg_raise_on_nonconvergence != projection.raise_on_nonconvergence
        or projection_state.pcg_preconditioner != projection.preconditioner
    ):
        raise ValueError("runtime sparse projection policy differs from the checkpoint")

    if type(constraint) is not IdentityConstraintHook:
        raise ValueError("runtime constraint must be the exact registered IdentityConstraintHook")
    descriptor = _jsonable(constraint.descriptor())
    if descriptor != contract.constraint.descriptor:
        raise ValueError("runtime constraint descriptor differs from the checkpoint")
    descriptor_sha256 = canonical_json_sha256(descriptor)
    if descriptor_sha256 != contract.constraint.descriptor_sha256:
        raise ValueError("runtime constraint descriptor SHA-256 differs from the checkpoint")

    if type(objective) is not CommonObjectiveContext:
        raise ValueError("runtime objective must be a CommonObjectiveContext")
    objective.validate_immutable()
    if objective.common_objective_sha256 != selected_sample.common_objective_sha256:
        raise ValueError("runtime common objective differs from the selected sample")
    if objective.dt != binding.physical_dt_seconds:
        raise ValueError("runtime common-objective dt differs from the evaluation binding")
    if objective.residual_scale != binding.residual_scale:
        raise ValueError("runtime common-objective residual scale differs from the evaluation binding")

    if type(physical_step) is not PhysicalStepContext:
        raise ValueError("runtime physical_step must be a PhysicalStepContext")
    physical_step.validate_immutable()
    if physical_step.physical_step_sha256 != selected_sample.physical_step_sha256:
        raise ValueError("runtime physical step differs from the selected sample")
    if physical_step.integration_policy != selected_sample.physical_integration_policy:
        raise ValueError("runtime physical integration policy differs from the selected sample")
    source_evidence_sha256 = (
        None if physical_step.source_evidence is None else physical_step.source_evidence.evidence_sha256
    )
    if source_evidence_sha256 != selected_sample.source_integration_evidence_sha256:
        raise ValueError("runtime source integration evidence differs from the selected sample")
    validate_physical_objective_integration(projection_state, objective, physical_step)
    _validate_config_execution_dtype(solver_config, physical_step.x_current)

    if type(result) is not IterativeSolverResult:
        raise ValueError("runtime result must be an IterativeSolverResult")
    work = _require_canonical_v6_work(result.work)
    expected_work = contract.inference_work
    observed_work = {
        "predictor_passes": work.predictor_passes,
        "global_compatibility_projections": work.projection_calls,
        "common_residual_evaluations": work.residual_evaluations,
        "common_objective_evaluations": work.objective_evaluations,
        "state_validity_evaluations": work.state_validity_evaluations,
        "physical_step_authentications": work.physical_step_authentications,
        "common_objective_authentications": work.common_objective_authentications,
        "constraint_preparations": work.constraint_preparations,
        "constraint_applications": work.constraint_applications,
    }
    for name, observed in observed_work.items():
        if type(observed) is not int or observed != expected_work[name]:
            raise ValueError(f"runtime observed work {name} differs from the checkpoint")
    if type(result.trace) is not tuple or len(result.trace) != contract.inference_iterations:
        raise ValueError("runtime trace length differs from inference K")
    if (
        not _v6_canonical_primitive_tree_equal(result.constraint_descriptor, descriptor)
        or type(result.constraint_descriptor_sha256) is not str
        or result.constraint_descriptor_sha256 != descriptor_sha256
    ):
        raise ValueError("runtime result constraint descriptor differs from the invoked hook")
    if (
        type(result.constraint_registration) is not str
        or result.constraint_registration != "registered-identity-development"
    ):
        raise ValueError("runtime result does not carry the registered identity-constraint scope")
    if type(result.head_mode) is not str or result.head_mode != "learned" or result.head_permutation is not None:
        raise ValueError("runtime result did not use the learned, unpermuted heads")
    if (
        type(result.physical_integration_policy) is not str
        or result.physical_integration_policy != physical_step.integration_policy
    ):
        raise ValueError("runtime result physical integration policy differs from the invoked solve")
    if (
        result.source_integration_evidence_sha256 is not None
        and type(result.source_integration_evidence_sha256) is not str
    ) or result.source_integration_evidence_sha256 != source_evidence_sha256:
        raise ValueError("runtime result source integration evidence differs from the invoked solve")
    if (
        type(result.physical_step_sha256) is not str
        or result.physical_step_sha256 != physical_step.physical_step_sha256
    ):
        raise ValueError("runtime result physical-step identity differs from the invoked solve")
    if (
        type(result.common_objective_sha256) is not str
        or result.common_objective_sha256 != objective.common_objective_sha256
    ):
        raise ValueError("runtime result common-objective identity differs from the invoked solve")
    if type(result.operator_geometry_sha256) is not str or result.operator_geometry_sha256 != actual_operator_sha256:
        raise ValueError("runtime result operator-geometry identity differs from the invoked solve")
    if type(result.projection_state_sha256) is not str or result.projection_state_sha256 != actual_projection_sha256:
        raise ValueError("runtime result projection-state identity differs from the invoked solve")
    if type(result.static_graph_sha256) is not str or result.static_graph_sha256 != actual_static_graph_sha256:
        raise ValueError("runtime result static-graph identity differs from the invoked solve")
    if work.projection_backend != projection.backend or work.projection_diagnostics_recorded is not True:
        raise ValueError("runtime result lacks required projection diagnostics")

    x_current = physical_step.x_current
    if x_current.ndim != 2:
        raise ValueError("schema-6 runtime replay is one unbatched sample at a time")
    if (
        result.positions.shape != x_current.shape
        or result.positions.device != objective.device
        or result.positions.dtype != objective.dtype
    ):
        raise ValueError("runtime result position shape, device, or dtype differs from the bound sample")
    targets = physical_step.pinned_targets
    if targets.shape != (projection_state.pinned.numel(), 3):
        raise ValueError("runtime physical-step pinned targets have the wrong unbatched shape")
    if not torch.isfinite(x_current).all() or not torch.equal(x_current[projection_state.pinned], targets):
        raise ValueError("runtime persistence initializer is non-finite or violates exact pinned targets")
    initial = _runtime_state_metrics(
        objective,
        projection_state,
        x_current,
        detach_residual=solver_config.detach_residual_features,
    )
    _require_finite_v6_state_metrics("initial", initial)
    if (initial["minimum_determinant"] <= contract.safeguards.minimum_determinant).any() or (
        initial["minimum_singular_value"] <= contract.safeguards.minimum_singular_value
    ).any():
        raise ValueError("runtime persistence initializer violates a state-validity safeguard")

    network_force = physical_step.force.clone()
    network_force[projection_state.pinned] = 0.0
    previous_positions = x_current
    projection_diagnostics: list[ProjectionDiagnostics] = []
    replay_constraint = IdentityConstraintHook()
    constraint_state = replay_constraint.begin_step(
        previous_positions.clone(),
        projection_state.pinned.clone(),
        targets.clone(),
    )
    accepted_count = 0
    zero_count = 0
    retained_count = 0
    for iteration, trace in enumerate(result.trace):
        if type(trace) is not IterativeSolverIteration:
            raise ValueError("runtime trace entries must be exact IterativeSolverIteration values")
        iteration_fraction = iteration / max(len(result.trace) - 1, 1)
        if (
            type(trace.iteration) is not int
            or trace.iteration != iteration
            or type(trace.iteration_fraction) is not float
            or trace.iteration_fraction != (iteration_fraction)
        ):
            raise ValueError("runtime trace iteration order or fraction has a non-canonical primitive type or value")
        iteration_positions = previous_positions.clone()
        internal_positions_before = x_current if iteration == 0 else result.trace[iteration - 1].positions
        _require_exact_v6_trace_tensor(
            f"trace[{iteration}].positions_before internal chain",
            trace.positions_before,
            internal_positions_before,
        )
        if iteration > 0:
            previous_trace = result.trace[iteration - 1]
            for before_name, after_name in (
                ("normalized_residual_before", "residual_after"),
                ("raw_residual_norm_before", "raw_residual_norm_after"),
                ("normalized_residual_norm_before", "normalized_residual_norm_after"),
                ("objective_before", "objective_after"),
                ("minimum_determinant_before", "minimum_determinant_after"),
                ("minimum_singular_value_before", "minimum_singular_value_after"),
            ):
                _require_exact_v6_trace_tensor(
                    f"trace[{iteration}].{before_name} internal chain",
                    getattr(trace, before_name),
                    getattr(previous_trace, after_name),
                )
        require_same(f"trace[{iteration}].positions_before", trace.positions_before, iteration_positions)
        before = _runtime_state_metrics(
            objective,
            projection_state,
            iteration_positions,
            detach_residual=solver_config.detach_residual_features,
        )
        _require_finite_v6_state_metrics(f"trace[{iteration}] before", before)
        for name in (
            "normalized_residual_before",
            "raw_residual_norm_before",
            "normalized_residual_norm_before",
            "objective_before",
            "minimum_determinant_before",
            "minimum_singular_value_before",
        ):
            metric_name = name.removesuffix("_before")
            require_same(f"trace[{iteration}].{name}", getattr(trace, name), before[metric_name])

        observation = replay_constraint.prepare_iteration(constraint_state, iteration, iteration_positions.clone())
        if type(observation) is not ConstraintObservation:
            raise RuntimeError("canonical identity prepare replay changed type")
        constraint_state = observation.state
        require_same(f"trace[{iteration}].constraint.normal", observation.normal, torch.zeros_like(iteration_positions))
        require_same(
            f"trace[{iteration}].constraint.normalized_slack",
            observation.normalized_slack,
            torch.zeros(
                iteration_positions.shape[:-1],
                dtype=iteration_positions.dtype,
                device=iteration_positions.device,
            ),
        )
        canonical_prepare_diagnostics = {"refreshes": 0}
        if not _v6_canonical_primitive_tree_equal(
            trace.constraint_prepare_diagnostics, canonical_prepare_diagnostics
        ) or not _v6_canonical_primitive_tree_equal(observation.diagnostics, canonical_prepare_diagnostics):
            raise ValueError(
                "runtime identity constraint preparation diagnostics have a non-canonical primitive type or value"
            )

        replay_target, replay_delta_h, replay_omega = predictor.predict_principal_stretch_update(
            projection_state,
            physical_step.x_current,
            physical_step.x_previous,
            iteration_positions,
            network_force,
            physical_step.gravity,
            physical_step.mu,
            physical_step.lam,
            physical_step.pin,
            before["normalized_residual"],
            observation.normal,
            observation.normalized_slack,
            iteration_fraction=iteration_fraction,
            physical_dt=objective.dt,
            head_mode="learned",
        )
        require_same(f"trace[{iteration}].delta_h", trace.delta_h, replay_delta_h)
        require_same(f"trace[{iteration}].omega", trace.omega, replay_omega)
        require_same(
            f"trace[{iteration}].target_deformation_gradient",
            trace.target_deformation_gradient,
            replay_target,
        )

        projection_kwargs: dict[str, object] = {}
        if projection.backend == "sparse_pcg":
            projection_kwargs["initial_positions"] = iteration_positions
        replay_projection = project_deformation_gradient(
            projection_state,
            replay_target,
            targets,
            return_diagnostics=True,
            **projection_kwargs,
        )
        if not isinstance(replay_projection, tuple):
            raise RuntimeError("diagnostic projection replay did not return diagnostics")
        replay_proposed, replay_diagnostics = replay_projection
        require_same_candidate(f"trace[{iteration}].proposed_positions", trace.proposed_positions, replay_proposed)
        diagnostics = _require_canonical_v6_projection_diagnostics(trace.projection_diagnostics)
        if (
            not diagnostics.converged
            or diagnostics.breakdown
            or diagnostics.backend != projection.backend
            or diagnostics.iterations < 0
            or diagnostics.iterations > projection.max_iterations
            or diagnostics.relative_tolerance != projection.relative_tolerance
            or diagnostics.absolute_tolerance != projection.absolute_tolerance
            or diagnostics.preconditioner != (None if projection.backend == "dense" else projection.preconditioner)
            or diagnostics.rhs_count != 3
            or diagnostics.converged_rhs != diagnostics.rhs_count
            or min(
                diagnostics.matrix_vector_products,
                diagnostics.preconditioner_applications,
                diagnostics.factor_solves,
                diagnostics.preconditioner_matrix_vector_products,
            )
            < 0
        ):
            raise ValueError("runtime projection diagnostics violate the checkpoint contract")
        diagnostic_scalars = (
            diagnostics.rhs_norm_max,
            diagnostics.initial_residual_norm_max,
            diagnostics.residual_norm_max,
            diagnostics.relative_residual_max,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in diagnostic_scalars):
            raise ValueError("runtime projection diagnostics contain an invalid residual metric")
        diagnostic_scalar_names = {
            "rhs_norm_max",
            "initial_residual_norm_max",
            "residual_norm_max",
            "relative_residual_max",
        }
        for field in dataclasses.fields(ProjectionDiagnostics):
            if field.name in diagnostic_scalar_names:
                if not math.isclose(
                    getattr(diagnostics, field.name),
                    getattr(replay_diagnostics, field.name),
                    rel_tol=contract.safeguards.replay_relative_tolerance,
                    abs_tol=contract.safeguards.replay_absolute_tolerance,
                ):
                    raise ValueError(f"runtime projection diagnostic {field.name} differs from the exact work replay")
            elif getattr(diagnostics, field.name) != getattr(replay_diagnostics, field.name):
                raise ValueError(f"runtime projection diagnostic {field.name} differs from the exact work replay")
        residual_norm, rhs_norm = _projection_residual_metrics(
            projection_state,
            replay_target,
            targets,
            replay_proposed,
        )
        if projection.backend == "sparse_pcg":
            threshold = projection.absolute_tolerance + projection.relative_tolerance * rhs_norm
        else:
            threshold = (
                contract.safeguards.replay_absolute_tolerance + contract.safeguards.replay_relative_tolerance * rhs_norm
            )
        replay_slack = (
            contract.safeguards.replay_absolute_tolerance + contract.safeguards.replay_relative_tolerance * rhs_norm
        )
        if (residual_norm > threshold + replay_slack).any():
            raise ValueError("runtime projected proposal violates the authenticated normal-equation tolerance")
        rhs_norm_max = float(rhs_norm.detach().max())
        residual_norm_max = float(residual_norm.detach().max())
        relative = torch.where(rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm)
        relative_residual_max = float(relative.detach().max())
        for name, observed, recomputed in (
            ("rhs_norm_max", diagnostics.rhs_norm_max, rhs_norm_max),
            ("residual_norm_max", diagnostics.residual_norm_max, residual_norm_max),
            ("relative_residual_max", diagnostics.relative_residual_max, relative_residual_max),
        ):
            if not math.isclose(
                observed,
                recomputed,
                rel_tol=contract.safeguards.replay_relative_tolerance,
                abs_tol=contract.safeguards.replay_absolute_tolerance,
            ):
                raise ValueError(f"runtime projection diagnostic {name} differs from the observed proposal")
        projection_diagnostics.append(diagnostics)

        candidates = trace.candidate_evaluations
        if type(candidates) is not tuple or len(candidates) != len(candidate_fractions):
            raise ValueError("runtime trace does not contain the exact fixed candidate schedule")
        stored_positions_before = trace.positions_before
        stored_proposed = trace.proposed_positions
        projected_displacement = stored_proposed - stored_positions_before
        applications: list[ConstraintApplication] = []
        candidate_metrics: list[dict[str, object]] = []
        for candidate_index, (step_fraction, candidate) in enumerate(zip(candidate_fractions, candidates, strict=True)):
            if type(candidate) is not CandidateEvaluation:
                raise ValueError("runtime candidate must be an exact CandidateEvaluation")
            if type(candidate.candidate_index) is not int or candidate.candidate_index != candidate_index:
                raise ValueError("runtime candidate order differs from the authenticated schedule")
            if type(candidate.step_fraction) is not float or candidate.step_fraction != step_fraction:
                raise ValueError("runtime candidate fraction differs from the authenticated schedule")
            if step_fraction == 1.0:
                candidate_positions = stored_proposed.clone()
            elif step_fraction == 0.0:
                candidate_positions = stored_positions_before.clone()
            else:
                candidate_positions = (
                    stored_positions_before + stored_positions_before.new_tensor(step_fraction) * projected_displacement
                )
            _require_exact_v6_trace_tensor(
                f"trace[{iteration}].candidate[{candidate_index}].exact interpolated candidate",
                candidate.candidate_positions,
                candidate_positions,
            )
            application = replay_constraint.constrain_candidate(
                constraint_state,
                iteration,
                candidate_index,
                step_fraction,
                stored_positions_before.clone(),
                candidate_positions.clone(),
                projection_state.pinned.clone(),
                targets.clone(),
            )
            if type(application) is not ConstraintApplication:
                raise RuntimeError("canonical identity candidate replay changed type")
            expected_constrained = candidate_positions.index_copy(-2, projection_state.pinned, targets)
            if not _tensor_bytes_equal(application.positions, expected_constrained):
                raise ValueError("runtime identity candidate replay changed the constrained position")
            _require_exact_v6_trace_tensor(
                f"trace[{iteration}].candidate[{candidate_index}].constrained_positions "
                "exact identity-constrained candidate",
                candidate.constrained_positions,
                expected_constrained,
            )
            metrics = _runtime_candidate_metrics(
                objective=objective,
                projection_state=projection_state,
                config=solver_config,
                current=stored_positions_before,
                projected_positions=stored_proposed,
                constrained_positions=expected_constrained,
                pinned_targets=targets,
                step_fraction=step_fraction,
                objective_before=trace.objective_before,
                normalized_residual_norm_before=trace.normalized_residual_norm_before,
            )
            for name in (
                "normalized_residual",
                "raw_residual_norm",
                "normalized_residual_norm",
                "objective",
                "minimum_determinant",
                "minimum_singular_value",
            ):
                require_same_candidate(
                    f"trace[{iteration}].candidate[{candidate_index}].{name}",
                    getattr(candidate, name),
                    metrics[name],
                )
            for name in (
                "positions_finite",
                "exact_pins",
                "determinant_valid",
                "singular_value_valid",
                "objective_finite",
                "residual_finite",
                "state_valid",
                "objective_nonincreasing",
                "residual_nonincreasing",
                "learned_contribution_retained",
                "admissible",
            ):
                observed = getattr(candidate, name)
                if type(observed) is not bool or observed is not metrics[name]:
                    raise ValueError(f"runtime candidate gate {name} differs from the independent replay")
            expected_zero_unchanged = metrics["zero_step_unchanged"]
            if candidate.zero_step_unchanged is not expected_zero_unchanged or (
                candidate.zero_step_unchanged is not None and type(candidate.zero_step_unchanged) is not bool
            ):
                raise ValueError("runtime candidate zero-step gate differs from the independent replay")
            expected_retention = metrics["displacement_retention"]
            if expected_retention is None:
                if candidate.displacement_retention is not None:
                    raise ValueError("runtime candidate displacement retention must be absent")
            else:
                require_same_candidate(
                    f"trace[{iteration}].candidate[{candidate_index}].displacement_retention",
                    candidate.displacement_retention,
                    expected_retention,
                )
            if not _v6_canonical_primitive_tree_equal(candidate.rejection_reasons, metrics["rejection_reasons"]):
                raise ValueError(
                    "runtime candidate rejection reasons have a non-canonical primitive type or differ from replay"
                )
            canonical_candidate_diagnostics = {"truncation_calls": 0, "minimum_fraction": 1.0}
            if not _v6_canonical_primitive_tree_equal(
                candidate.constraint_diagnostics, canonical_candidate_diagnostics
            ) or not _v6_canonical_primitive_tree_equal(application.diagnostics, canonical_candidate_diagnostics):
                raise ValueError("runtime identity candidate diagnostics have a non-canonical primitive type or value")
            applications.append(application)
            candidate_metrics.append(metrics)

        zero_candidate = candidates[-1]
        if zero_candidate.zero_step_unchanged is not True or not _tensor_bytes_equal(
            zero_candidate.constrained_positions, iteration_positions
        ):
            raise ValueError("runtime zero candidate is not the authenticated exact no-op")
        selected_index = next(
            (candidate.candidate_index for candidate in candidates[:-1] if candidate.admissible),
            len(candidates) - 1,
        )
        if selected_index == len(candidates) - 1 and not zero_candidate.admissible:
            raise ValueError("runtime zero candidate is not admissible")
        selected_candidate = candidates[selected_index]
        selected_metrics = candidate_metrics[selected_index]
        selected_application = applications[selected_index]
        selected_fraction = candidate_fractions[selected_index]
        proposal_accepted = selected_fraction > 0.0
        learned_retained = selected_metrics["learned_contribution_retained"]
        expected_retention = selected_metrics["displacement_retention"]
        if selected_index == len(candidates) - 1:
            selection_reason = "no-admissible-positive"
        elif expected_retention is None:
            selection_reason = "first-admissible-positive-candidate-zero-projected-displacement"
        elif learned_retained:
            selection_reason = "first-admissible-positive-candidate"
        else:
            selection_reason = "first-admissible-positive-candidate-no-learned-displacement"
        if type(trace.selected_candidate_index) is not int or trace.selected_candidate_index != selected_index:
            raise ValueError("runtime selected candidate index differs from deterministic selection")
        if type(trace.selected_step_fraction) is not float or trace.selected_step_fraction != selected_fraction:
            raise ValueError("runtime selected step fraction differs from deterministic selection")
        if type(trace.proposal_accepted) is not bool or trace.proposal_accepted is not proposal_accepted:
            raise ValueError("runtime proposal acceptance differs from deterministic selection")
        if type(trace.learned_contribution_retained) is not bool or trace.learned_contribution_retained is not (
            learned_retained
        ):
            raise ValueError("runtime learned-contribution retention differs from candidate replay")
        if selected_candidate.displacement_retention is None:
            if trace.learned_displacement_retention is not None:
                raise ValueError("runtime selected displacement retention must be absent")
        else:
            _require_exact_v6_trace_tensor(
                f"trace[{iteration}].learned_displacement_retention internal selection",
                trace.learned_displacement_retention,
                selected_candidate.displacement_retention,
            )
        if type(trace.selection_reason) is not str or trace.selection_reason != selection_reason:
            raise ValueError("runtime selection reason differs from deterministic selection")
        if not _v6_canonical_primitive_tree_equal(trace.constraint_diagnostics, selected_application.diagnostics):
            raise ValueError(
                "runtime selected constraint diagnostics have a non-canonical primitive type or differ from selection"
            )
        _require_exact_v6_trace_tensor(
            f"trace[{iteration}].positions internal selection",
            trace.positions,
            selected_candidate.constrained_positions,
        )
        for name in (
            "residual_after",
            "raw_residual_norm_after",
            "normalized_residual_norm_after",
            "objective_after",
            "minimum_determinant_after",
            "minimum_singular_value_after",
        ):
            metric_name = "normalized_residual" if name == "residual_after" else name.removesuffix("_after")
            _require_exact_v6_trace_tensor(
                f"trace[{iteration}].{name} internal selection",
                getattr(trace, name),
                getattr(selected_candidate, metric_name),
            )
        constraint_state = selected_application.state
        previous_positions = selected_candidate.constrained_positions.clone()
        accepted_count += int(proposal_accepted)
        zero_count += int(selected_fraction == 0.0)
        retained_count += int(learned_retained)

    final = _runtime_state_metrics(
        objective,
        projection_state,
        previous_positions,
        detach_residual=solver_config.detach_residual_features,
    )
    _require_finite_v6_state_metrics("final", final)
    final_trace = result.trace[-1]
    _require_exact_v6_trace_tensor("result.positions internal chain", result.positions, final_trace.positions)
    for result_name, trace_name in (
        ("normalized_residual", "residual_after"),
        ("raw_residual_norm", "raw_residual_norm_after"),
        ("normalized_residual_norm", "normalized_residual_norm_after"),
        ("objective", "objective_after"),
        ("minimum_determinant", "minimum_determinant_after"),
        ("minimum_singular_value", "minimum_singular_value_after"),
    ):
        _require_exact_v6_trace_tensor(
            f"result.{result_name} internal chain",
            getattr(result, result_name),
            getattr(final_trace, trace_name),
        )
    require_same("result.positions", result.positions, previous_positions)
    if not _tensor_bytes_equal(result.positions, previous_positions):
        raise ValueError("runtime final positions do not equal the last committed trace state")
    for name in (
        "normalized_residual",
        "objective",
        "raw_residual_norm",
        "normalized_residual_norm",
        "minimum_determinant",
        "minimum_singular_value",
    ):
        require_same(f"result.{name}", getattr(result, name), final[name])
    for name, expected in (
        ("proposal_accepted_iterations", accepted_count),
        ("zero_step_iterations", zero_count),
        ("learned_contribution_retained_iterations", retained_count),
    ):
        observed = getattr(result, name)
        if type(observed) is not int or observed != expected:
            raise ValueError(f"runtime result aggregate {name} differs from the candidate trace")
    aggregate = {
        "projection_iterations": sum(item.iterations for item in projection_diagnostics),
        "projection_matrix_vector_products": sum(item.matrix_vector_products for item in projection_diagnostics),
        "projection_preconditioner_applications": sum(
            item.preconditioner_applications for item in projection_diagnostics
        ),
        "projection_factor_solves": sum(item.factor_solves for item in projection_diagnostics),
    }
    if any(getattr(result.work, name) != count for name, count in aggregate.items()):
        raise ValueError("runtime aggregate projection work differs from its per-iteration diagnostics")
    _verify_predictor_execution_surface(predictor)
    return VerifiedV6Runtime(
        claim_scope=_RUNTIME_CLAIM_SCOPE,
        checkpoint_payload_sha256=verified.checkpoint_payload_sha256,
        evaluation_binding_sha256=binding.evaluation_binding_sha256,
        learned_state_sha256=runtime_state_sha256,
        held_out_topology_sha256=binding.held_out_topology_sha256,
        operator_geometry_sha256=actual_operator_sha256,
        projection_state_sha256=actual_projection_sha256,
        static_graph_sha256=actual_static_graph_sha256,
        physical_integration_policy=physical_step.integration_policy,
        source_integration_evidence_sha256=source_evidence_sha256,
        physical_step_sha256=physical_step.physical_step_sha256,
        common_objective_sha256=objective.common_objective_sha256,
        iterations=solver_config.iterations,
        constraint_descriptor_sha256=descriptor_sha256,
        proposal_accepted_iterations=accepted_count,
        zero_step_iterations=zero_count,
        learned_contribution_retained_iterations=retained_count,
        **aggregate,
    )
