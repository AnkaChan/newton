# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Provenance-bound full-gradient training on audited PR transitions.

This module is the common-objective counterpart to the legacy trajectory
trainer.  Every sample is an accepted :class:`HistoryTransition` from
``pr_scene_history``.  The graph transformer observes the post-callback state
``A_k.q`` and the exact float32 reconstruction of the preceding position,

``x_previous = float32(C_k.q - float32(C_k.qd * dt32))``.

Pin indices and targets are transition-local, so moving Dirichlet data are not
collapsed to a rest-pose constant.  Architectures v3 and v4 predict a full
deformation-gradient field, which is decoded by exactly one weighted global
projection.  The default dense Cholesky projection is the only backend this
initial trainer accepts; a checkpoint records that choice explicitly.

The compatibility-default position loss is a dimensionless mass-weighted
free-vertex error,

``sum_i m_i ||x_i - x_i_ref||^2 / (sum_i m_i * ell^2)``,

where ``ell`` is the static RMS rest-edge length. An optional decoded
deformation-gradient term is the rest-volume-weighted mean squared component
error. An explicit alternative supervises the raw full-gradient target before
projection and normalizes each transition by its zero-head observed-to-
reference error. Every choice and normalization is checkpoint-authenticated.

Checkpoint tensor state and JSON metadata have separate SHA-256 digests.  The
metadata binds the history manifest, complete static bundle, selected
transition hashes and prefixes, exact model configuration, realized hierarchy
depth, seeds, loss contract, and decoder work.  A selected transition set is
never described as a complete history.

This remains research infrastructure rather than a public Newton API.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from . import torch_solver
from .graph_transformer import GraphTransformerConfig
from .pr_scene_history import HistoryTransition, PRHistoryStaticBundle, PRSceneHistory
from .predictor import (
    StretchPredictor,
    build_stretch_predictor,
    checkpoint_predictor_config,
    load_stretch_predictor_state,
    predictor_decoder_work,
)
from .solver_benchmark import build_common_problem, common_objective_manifest, evaluate_common_state

_SCHEMA_VERSION = 3
_CHECKPOINT_CONTRACT = "pr2901-history-v3-checkpoint-v3"
_EVALUATION_CONTRACT = "pr2901-history-v3-evaluation-v3"
_MILESTONE_SCHEMA_VERSION = 4
_MILESTONE_CHECKPOINT_CONTRACT = "pr2901-history-v3-milestone-checkpoint-v4"
_RAW_F_EVALUATION_CONTRACT = "pr2901-history-v3-raw-f-evaluation-v1"
_LEGACY_CHECKPOINT_IDENTITIES = ((1, "pr2901-history-v3-checkpoint-v1"),)
_SUPPORTED_PROJECTION_BACKEND = "dense_cholesky"
_DECODED_LOSS_MODE = "decoded-position-deformation"
_NORMALIZED_RAW_F_LOSS_MODE = "normalized-raw-deformation-gradient"
_LOSS_MODES = (_DECODED_LOSS_MODE, _NORMALIZED_RAW_F_LOSS_MODE)
_PHASE_BALANCED_SAMPLING_CONTRACT = "pr2901-substep-phase-balanced-epochs-v1"
_OPTIMIZER_PARAMETER_BINDING_CONTRACT = "pr2901-adamw-ordered-model-parameters-v1"
_EXTERNAL_PARENT_LINEAGE_CONTRACT = "pr2901-externally-pinned-milestone-parent-v1"
_EXTERNAL_PARENT_LINEAGE_SCOPE = (
    "child records component hashes from the supplied verified parent; standalone child verification "
    "cannot prove parent availability or contents, so the experiment must externally pin parent file "
    "and payload SHA-256"
)
_DIAGNOSTIC_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _jsonable(value: object) -> object:
    """Return a finite, canonical-JSON-compatible copy."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("authenticated JSON metadata must be finite")
        return value
    raise TypeError(f"unsupported authenticated metadata type {type(value).__name__}")


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=canonical_dtype)


def _array_digest(value: np.ndarray) -> str:
    array = _canonical_array(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU bytes."""
    records: list[dict[str, object]] = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state_dict entry {name!r} is not a tensor")
        array = tensor.detach().cpu().contiguous().numpy()
        records.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": _array_digest(array),
            }
        )
    return _canonical_digest(records)


def _state_tree_record(value: object) -> object:
    """Return a canonical digest record for nested optimizer state."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_digest(array),
        }
    if isinstance(value, Mapping):
        items: list[dict[str, object]] = []
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise TypeError(f"unsupported optimizer-state key type {type(key).__name__}")
            key_record = {"type": type(key).__name__, "value": key}
            items.append({"key": key_record, "value": _state_tree_record(item)})
        items.sort(key=lambda item: json.dumps(item["key"], sort_keys=True, separators=(",", ":")))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_state_tree_record(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_state_tree_record(item) for item in value]}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer state must contain only finite floats")
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported optimizer-state value type {type(value).__name__}")


def _state_tree_digest(value: object) -> str:
    """Hash a nested tensor/scalar state without relying on pickle bytes."""
    return _canonical_digest(_state_tree_record(value))


def _clone_state_tree(value: object) -> object:
    """Clone nested optimizer state onto CPU for an immutable checkpoint."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
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
    raise TypeError(f"unsupported optimizer-state value type {type(value).__name__}")


def _seed_everything(seed: int) -> np.random.Generator:
    """Seed NumPy and Torch generators and return the sampling generator."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _source_provenance() -> dict[str, str | None]:
    """Return HEAD and a content digest for every dirty source path."""
    repository = pathlib.Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "-z"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    dirty_sha256 = None
    if status.returncode == 0 and status.stdout:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        digest = hashlib.sha256()
        digest.update(status.stdout)
        digest.update(diff.stdout)
        if untracked.returncode == 0:
            for relative_bytes in sorted(item for item in untracked.stdout.split(b"\0") if item):
                path = repository / relative_bytes.decode("utf-8", errors="surrogateescape")
                digest.update(relative_bytes)
                digest.update(path.read_bytes())
        dirty_sha256 = digest.hexdigest()
    return {
        "newton_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty_tree_sha256": dirty_sha256,
    }


_SOURCE_PROVENANCE_AT_IMPORT = _source_provenance()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _characteristic_length(rest_q: np.ndarray, tets: np.ndarray) -> float:
    """Return the RMS length over all six rest edges of every tet."""
    corners = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    squared = [
        np.sum((rest_q[tets[:, a]].astype(np.float64) - rest_q[tets[:, b]].astype(np.float64)) ** 2, axis=1)
        for a, b in corners
    ]
    length = float(np.sqrt(np.mean(np.concatenate(squared))))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("static mesh has no finite positive characteristic edge length")
    return length


def _pin_flag(pinned_indices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pinned = np.zeros(int(tets.max()) + 1, dtype=bool)
    pinned[pinned_indices] = True
    return pinned[tets].any(axis=1).astype(np.float32)


def _float32_previous(transition: HistoryTransition) -> np.ndarray:
    dt32 = np.float32(transition.dt_seconds)
    displacement = (transition.input_state.qd * dt32).astype(np.float32)
    return (transition.input_state.q - displacement).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class PRV3TrainingConfig:
    """Optimization and explicit loss settings authenticated per checkpoint."""

    steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    position_loss_weight: float = 1.0
    deformation_gradient_loss_weight: float = 0.0
    loss_mode: str = _DECODED_LOSS_MODE
    raw_deformation_gradient_floor: float = 1.0e-8
    gradient_clip_norm: float = 5.0
    seed: int = 0
    log_every: int = 50
    projection_backend: str = _SUPPORTED_PROJECTION_BACKEND

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.log_every, bool) or not isinstance(self.log_every, int) or self.log_every < 1:
            raise ValueError("log_every must be a positive integer")
        for name in (
            "learning_rate",
            "position_loss_weight",
            "deformation_gradient_loss_weight",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.loss_mode, str) or self.loss_mode not in _LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {_LOSS_MODES}")
        if (
            isinstance(self.raw_deformation_gradient_floor, bool)
            or not math.isfinite(self.raw_deformation_gradient_floor)
            or self.raw_deformation_gradient_floor <= 0.0
        ):
            raise ValueError("raw_deformation_gradient_floor must be finite and positive")
        if self.loss_mode == _DECODED_LOSS_MODE:
            if self.position_loss_weight == 0.0 and self.deformation_gradient_loss_weight == 0.0:
                raise ValueError("decoded loss mode requires at least one positive loss weight")
        elif self.position_loss_weight != 0.0 or self.deformation_gradient_loss_weight != 0.0:
            raise ValueError(
                "normalized raw deformation-gradient mode requires position_loss_weight=0 and "
                "deformation_gradient_loss_weight=0"
            )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.projection_backend != _SUPPORTED_PROJECTION_BACKEND:
            raise ValueError(
                f"unsupported projection_backend {self.projection_backend!r}; "
                f"this trainer currently requires {_SUPPORTED_PROJECTION_BACKEND!r}"
            )


@dataclasses.dataclass(frozen=True)
class PRV4MilestoneConfig:
    """Authenticated sampling and restart settings for v4 training."""

    milestone_updates: tuple[int, ...] = (8_000, 15_000)
    sampling_contract: str = _PHASE_BALANCED_SAMPLING_CONTRACT
    track_parameter_update_norm: bool = False

    def __post_init__(self) -> None:
        milestones = tuple(self.milestone_updates)
        if not milestones:
            raise ValueError("milestone_updates must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in milestones):
            raise ValueError("milestone_updates must contain positive integers")
        if tuple(sorted(set(milestones))) != milestones:
            raise ValueError("milestone_updates must be strictly increasing and unique")
        if self.sampling_contract != _PHASE_BALANCED_SAMPLING_CONTRACT:
            raise ValueError(f"sampling_contract must be {_PHASE_BALANCED_SAMPLING_CONTRACT!r}")
        if not isinstance(self.track_parameter_update_norm, bool):
            raise ValueError("track_parameter_update_norm must be a bool")
        object.__setattr__(self, "milestone_updates", milestones)


@dataclasses.dataclass(frozen=True)
class _PreparedSample:
    transition: HistoryTransition
    x_current: np.ndarray
    x_previous: np.ndarray
    pinned_indices: np.ndarray
    pin_targets: np.ndarray
    reference_positions: np.ndarray
    observed_F: np.ndarray
    reference_F: np.ndarray
    raw_deformation_gradient_observed_loss: float
    pin_signature: tuple[int, ...]


@dataclasses.dataclass
class _PreparedDataset:
    history: PRSceneHistory
    static: PRHistoryStaticBundle
    samples: tuple[_PreparedSample, ...]
    characteristic_length_m: float
    selection_record: dict[str, object]


@dataclasses.dataclass(frozen=True)
class _RawTargetSupervision:
    """Immutable device-resident tensors reused by every sampled step."""

    reference_F: torch.Tensor
    normalizer: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _ResidentPredictionInput:
    """One sample's device-resident inference tensors."""

    state: torch_solver.SolverState
    x_current: torch.Tensor
    x_previous: torch.Tensor
    force: torch.Tensor
    gravity: torch.Tensor
    mu: torch.Tensor
    lam: torch.Tensor
    pin: torch.Tensor
    pinned_targets: torch.Tensor


@dataclasses.dataclass
class PRV3TrainingResult:
    """In-memory result of one authenticated training run."""

    predictor: StretchPredictor
    checkpoint: dict[str, object]


@dataclasses.dataclass(frozen=True)
class _PhaseBalancedSchedule:
    """Immutable complete batch stream plus its authenticated contract."""

    batches: np.ndarray
    record: dict[str, object]


@dataclasses.dataclass
class PRV4MilestoneTrainingResult:
    """Result of one new or resumed milestone-training process."""

    predictor: StretchPredictor
    checkpoints: dict[int, dict[str, object]]
    checkpoint_paths: dict[int, pathlib.Path]
    completed_updates: int


def _validate_static_bundle(history: PRSceneHistory) -> PRHistoryStaticBundle:
    static = history.static_bundle
    manifest = history.manifest
    expected = {
        "manifest_sha256": manifest.manifest_sha256,
        "base_physical_sha256": manifest.base_physical_sha256,
        "topology_sha256": manifest.topology_sha256,
        "material_sha256": manifest.material_sha256,
    }
    for name, value in expected.items():
        if getattr(static, name) != value:
            raise ValueError(f"history static bundle has inconsistent {name}")
    if static.as_dict().get("static_sha256") != static.static_sha256:
        raise ValueError("history static bundle record is not self-consistent")
    return static


def _prepare_dataset(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
) -> _PreparedDataset:
    """Validate and content-address one same-topology transition selection."""
    static = _validate_static_bundle(history)
    transitions = tuple(transitions)
    if not transitions:
        raise ValueError("at least one accepted history transition is required")
    hashes = [transition.transition_sha256 for transition in transitions]
    if len(hashes) != len(set(hashes)):
        raise ValueError("training transition selection contains duplicates")

    static_tets = torch.tensor(static.tet_indices, dtype=torch.int64)
    static_dm_inverse = torch.tensor(static.tet_poses, dtype=torch.float64)
    static_J = torch_solver._build_J(static_dm_inverse)
    static_volume = 1.0 / (6.0 * torch.linalg.det(static_dm_inverse))
    raw_F_denominator = 9.0 * static_volume.sum()

    samples: list[_PreparedSample] = []
    transition_records: list[dict[str, object]] = []
    for transition in transitions:
        if not isinstance(transition, HistoryTransition):
            raise TypeError("every sample must be a HistoryTransition")
        if transition.manifest_sha256 != history.manifest.manifest_sha256:
            raise ValueError("transition belongs to a different history manifest")
        if transition.topology_sha256 != static.topology_sha256:
            raise ValueError("multi-topology training is not yet supported")
        if transition.material_sha256 != static.material_sha256:
            raise ValueError("transition material does not match the static bundle")
        if transition.dt_seconds != history.manifest.dt_seconds:
            raise ValueError("transition dt does not exactly match the history manifest")
        if transition.reference_record.get("accepted") is not True:
            raise ValueError("training requires an explicitly accepted dense-Newton reference")

        model_inputs = transition.model_inputs()
        expected_keys = (
            "x_current",
            "x_previous",
            "pinned_indices",
            "pin_targets",
            "inertial_target",
        )
        if tuple(model_inputs) != expected_keys:
            raise ValueError("transition model-input schema is not the expected v1 order")
        if not np.array_equal(model_inputs["x_current"], transition.applied_state.q):
            raise ValueError("transition model x_current is not the post-callback state A_k.q")
        if not np.array_equal(model_inputs["x_previous"], _float32_previous(transition)):
            raise ValueError("transition model x_previous violates the float32 velocity contract")
        if not np.array_equal(model_inputs["pinned_indices"], transition.applied_state.pinned_indices):
            raise ValueError("transition model pinned_indices do not match A_k")
        if not np.array_equal(model_inputs["pin_targets"], transition.applied_state.pin_targets):
            raise ValueError("transition model pin_targets do not match A_k")
        if not np.array_equal(model_inputs["inertial_target"], transition.inertial_target):
            raise ValueError("transition model inertial_target is inconsistent")

        scene = history.build_atomic_scene(transition.input_state, transition.applied_state)
        problem = build_common_problem(scene)
        objective = common_objective_manifest(scene, problem)
        if scene.manifest()["scene_sha256"] != transition.scene_sha256:
            raise ValueError("transition scene hash does not match reconstructed common scene")
        if objective["objective_instance_sha256"] != transition.objective_instance_sha256:
            raise ValueError("transition objective hash does not match reconstructed common objective")

        pinned_indices = model_inputs["pinned_indices"]
        observed = torch.tensor(model_inputs["x_current"], dtype=torch.float64)
        reference = torch.tensor(transition.reference_positions, dtype=torch.float64)
        observed_F_tensor = torch_solver.compute_F(observed, static_tets, static_J)
        reference_F_tensor = torch_solver.compute_F(reference, static_tets, static_J)
        raw_observed_loss = float(
            (static_volume[:, None, None] * (observed_F_tensor - reference_F_tensor).square()).sum() / raw_F_denominator
        )
        if not math.isfinite(raw_observed_loss) or raw_observed_loss < 0.0:
            raise ValueError("raw deformation-gradient normalizer must be finite and non-negative")
        observed_F = np.ascontiguousarray(observed_F_tensor.numpy())
        reference_F = np.ascontiguousarray(reference_F_tensor.numpy())
        observed_F.setflags(write=False)
        reference_F.setflags(write=False)
        samples.append(
            _PreparedSample(
                transition=transition,
                x_current=model_inputs["x_current"],
                x_previous=model_inputs["x_previous"],
                pinned_indices=pinned_indices,
                pin_targets=model_inputs["pin_targets"],
                reference_positions=transition.reference_positions,
                observed_F=observed_F,
                reference_F=reference_F,
                raw_deformation_gradient_observed_loss=raw_observed_loss,
                pin_signature=tuple(int(index) for index in pinned_indices),
            )
        )
        training_record = transition.training_record()
        transition_records.append(
            {
                "transition_sha256": transition.transition_sha256,
                "training_record_sha256": _canonical_digest(training_record),
                "input_prefix_sha256": transition.input_prefix_sha256,
                "input_state_sha256": transition.input_state_sha256,
                "coordinate": transition.coordinate.as_dict(),
                "scene_sha256": transition.scene_sha256,
                "objective_instance_sha256": transition.objective_instance_sha256,
                "pin_signature": [int(index) for index in pinned_indices],
                "pin_signature_sha256": _array_digest(pinned_indices),
                "pin_count": int(pinned_indices.size),
                "observed_F_sha256": _array_digest(observed_F),
                "reference_F_sha256": _array_digest(reference_F),
                "raw_deformation_gradient_observed_loss": raw_observed_loss,
            }
        )

    selection_record: dict[str, object] = {
        "contract": "pr2901-history-selected-transition-set-v2",
        "provenance_scope": "selected-content-addressed-transitions-not-a-complete-history-claim",
        "history_manifest_sha256": history.manifest.manifest_sha256,
        "static_sha256": static.static_sha256,
        "transitions": transition_records,
    }
    selection_record["selection_sha256"] = _canonical_digest(selection_record)
    return _PreparedDataset(
        history=history,
        static=static,
        samples=tuple(samples),
        characteristic_length_m=_characteristic_length(static.rest_q, static.tet_indices),
        selection_record=selection_record,
    )


def _build_phase_balanced_schedule_from_records(
    transition_records: Sequence[Mapping[str, object]],
    *,
    substeps_per_frame: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> _PhaseBalancedSchedule:
    """Build exact shuffled epochs interleaved by atomic-substep phase.

    Every epoch presents every transition exactly once. Equal-sized substep
    groups are independently shuffled and then interleaved in increasing
    substep order. Because a batch contains an integer number of complete
    phase cycles, every update receives the same number of samples from every
    phase. Epoch-boundary prefixes are arranged to prevent duplicate samples
    inside a batch that straddles two epochs.
    """
    integer_values = {
        "substeps_per_frame": substeps_per_frame,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
    }
    for name, value in integer_values.items():
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    records = tuple(transition_records)
    sample_count = len(records)
    if sample_count < 1:
        raise ValueError("phase-balanced sampling requires at least one transition")
    if batch_size > sample_count:
        raise ValueError("phase-balanced sampling requires batch_size <= sample_count")
    if batch_size % substeps_per_frame != 0:
        raise ValueError("batch_size must be divisible by substeps_per_frame")
    total_presentations = steps * batch_size
    if total_presentations % sample_count != 0:
        raise ValueError("total sampled presentations must be divisible by the sample count")

    phases: list[int] = []
    transition_hashes: list[str] = []
    phase_groups: dict[int, list[int]] = {phase: [] for phase in range(substeps_per_frame)}
    for index, record in enumerate(records):
        coordinate = record.get("coordinate")
        transition_sha256 = record.get("transition_sha256")
        if not isinstance(coordinate, Mapping):
            raise ValueError("training transition record is missing its atomic coordinate")
        phase = coordinate.get("substep")
        if isinstance(phase, bool) or not isinstance(phase, int) or not 0 <= phase < substeps_per_frame:
            raise ValueError("training transition has an invalid atomic-substep phase")
        if not _is_sha256(transition_sha256):
            raise ValueError("training transition has an invalid transition SHA-256")
        phases.append(phase)
        transition_hashes.append(transition_sha256)
        phase_groups[phase].append(index)

    group_sizes = {len(indices) for indices in phase_groups.values()}
    if len(group_sizes) != 1 or 0 in group_sizes:
        raise ValueError("every atomic-substep phase must contain the same positive sample count")
    group_size = next(iter(group_sizes))
    samples_per_phase_per_batch = batch_size // substeps_per_frame
    if samples_per_phase_per_batch > group_size:
        raise ValueError("a phase-balanced batch cannot repeat a sample within one update")

    epoch_count = total_presentations // sample_count
    rng = np.random.default_rng(seed)
    stream: list[int] = []
    previous_permutations: dict[int, list[int]] | None = None
    for _epoch in range(epoch_count):
        tail_count = len(stream) % batch_size
        if tail_count % substeps_per_frame != 0:
            raise RuntimeError("internal phase stream lost cycle alignment")
        tail_cycles = tail_count // substeps_per_frame
        prefix_cycles = 0 if tail_count == 0 else samples_per_phase_per_batch - tail_cycles
        permutations: dict[int, list[int]] = {}
        for phase in range(substeps_per_frame):
            permutation = [int(index) for index in rng.permutation(phase_groups[phase])]
            if prefix_cycles and previous_permutations is not None:
                forbidden = set(previous_permutations[phase][-tail_cycles:])
                allowed = [index for index in permutation if index not in forbidden]
                blocked = [index for index in permutation if index in forbidden]
                if len(allowed) < prefix_cycles:
                    raise RuntimeError("cannot construct a duplicate-free epoch boundary")
                permutation = allowed + blocked
            permutations[phase] = permutation
        for offset in range(group_size):
            for phase in range(substeps_per_frame):
                stream.append(permutations[phase][offset])
        previous_permutations = permutations

    batches = np.asarray(stream, dtype=np.int64).reshape(steps, batch_size)
    expected_phase_count = samples_per_phase_per_batch
    for batch in batches:
        if len({int(index) for index in batch}) != batch_size:
            raise RuntimeError("phase-balanced sampler produced a duplicate within one update")
        counts = np.bincount(np.asarray([phases[int(index)] for index in batch]), minlength=substeps_per_frame)
        if not np.array_equal(counts, np.full(substeps_per_frame, expected_phase_count, dtype=np.int64)):
            raise RuntimeError("phase-balanced sampler produced an imbalanced update")
    exposures = np.bincount(batches.reshape(-1), minlength=sample_count)
    expected_exposure = total_presentations // sample_count
    if not np.array_equal(exposures, np.full(sample_count, expected_exposure, dtype=np.int64)):
        raise RuntimeError("complete phase-balanced schedule does not expose every sample equally")
    batches = np.ascontiguousarray(batches)
    batches.setflags(write=False)
    record: dict[str, object] = {
        "contract": _PHASE_BALANCED_SAMPLING_CONTRACT,
        "phase_source": "HistoryTransition.coordinate.substep",
        "generator": "numpy.random.Generator(PCG64)",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "sample_count": sample_count,
        "substeps_per_frame": substeps_per_frame,
        "samples_per_phase": group_size,
        "samples_per_phase_per_batch": samples_per_phase_per_batch,
        "epoch_count": epoch_count,
        "total_presentations": total_presentations,
        "without_replacement_within_batch": True,
        "equal_final_exposure": True,
        "phase_by_sample_index": phases,
        "transition_sha256": transition_hashes,
        "batch_stream_sha256": _array_digest(batches),
        "exposure_by_sample_index": [int(value) for value in exposures],
        "exposure_by_transition_sha256": {
            transition_hash: int(exposures[index]) for index, transition_hash in enumerate(transition_hashes)
        },
    }
    record["sampling_sha256"] = _canonical_digest(record)
    return _PhaseBalancedSchedule(batches=batches, record=record)


def _build_phase_balanced_schedule(
    dataset: _PreparedDataset,
    train_config: PRV3TrainingConfig,
) -> _PhaseBalancedSchedule:
    records = dataset.selection_record["transitions"]
    if not isinstance(records, Sequence):
        raise RuntimeError("prepared transition selection is malformed")
    return _build_phase_balanced_schedule_from_records(
        records,
        substeps_per_frame=dataset.history.manifest.substeps_per_frame,
        steps=train_config.steps,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )


def _validate_equal_milestone_exposure(
    schedule: _PhaseBalancedSchedule,
    milestone_config: PRV4MilestoneConfig,
) -> None:
    """Require every serialized restart boundary to expose all samples equally."""
    sample_count = int(schedule.record["sample_count"])
    batch_size = int(schedule.record["batch_size"])
    for update in milestone_config.milestone_updates:
        if update * batch_size % sample_count != 0:
            raise ValueError(f"milestone update {update} does not give equal exposure to all {sample_count} samples")
        prefix = _prefix_exposure_record(schedule, update)
        if prefix["equal_exposure_at_this_milestone"] is not True:
            raise RuntimeError("milestone exposure divisibility did not produce exact equality")


def _validate_graph_config(config: GraphTransformerConfig, dt: float) -> None:
    if config.architecture_version not in (3, 4):
        raise ValueError("PR history common training requires graph-transformer architecture version 3 or 4")
    if float(np.float32(config.dt)) != dt or config.dt != dt:
        raise ValueError(
            "graph-transformer dt must exactly equal the transition float32 dt; "
            f"got config.dt={config.dt!r}, transition.dt={dt!r}"
        )


def _decoder_work(projection_backend: str) -> dict[str, object]:
    if projection_backend != _SUPPORTED_PROJECTION_BACKEND:
        raise ValueError(f"unsupported projection backend {projection_backend!r}")
    # Use the shared work contract, then make the backend and factorization
    # explicit rather than inferring Cholesky from "one global solve".
    work = {
        "schema_version": 1,
        "target": "full-deformation-gradient",
        "decoder": "weighted-global-projection",
        "predictor_passes": 1,
        "global_triangular_solves": 1,
        "local_polar_sweeps": 0,
        "projection_backend": projection_backend,
        "factorization": "dense-prefactored-cholesky",
        "linear_solve_rhs": "three-coordinate-columns-per-sample",
    }
    return work


def _build_predictor_and_solvers(
    dataset: _PreparedDataset,
    graph_config: GraphTransformerConfig,
    device: torch.device,
) -> tuple[StretchPredictor, dict[tuple[int, ...], torch_solver.SolverState]]:
    static = dataset.static
    predictor = build_stretch_predictor(
        "graph-transformer",
        np.array(static.rest_q, copy=True),
        np.array(static.tet_indices, copy=True),
        device,
        torch.float32,
        residual=True,
        graph_config=graph_config,
    )
    shared_work = predictor_decoder_work(predictor, solver_iterations=1, blocks=1)
    if shared_work["target"] != "full-deformation-gradient" or shared_work["global_triangular_solves"] != 1:
        raise RuntimeError("shared predictor work contract is not the full-gradient one-shot projection")

    return predictor, _build_solvers(dataset, device)


def _build_solvers(
    dataset: _PreparedDataset,
    device: torch.device,
) -> dict[tuple[int, ...], torch_solver.SolverState]:
    """Build one dense projection factorization per dynamic pin signature."""
    static = dataset.static

    solvers: dict[tuple[int, ...], torch_solver.SolverState] = {}
    for sample in dataset.samples:
        if sample.pin_signature not in solvers:
            solvers[sample.pin_signature] = torch_solver.build_solver(
                np.array(static.rest_q, copy=True),
                np.array(static.tet_indices, copy=True),
                np.array(static.tet_poses, copy=True),
                np.array(sample.pinned_indices, copy=True),
                device=device,
                dtype=torch.float64,
            )
    return solvers


def _prepare_raw_target_supervision(
    dataset: _PreparedDataset,
    device: torch.device,
    floor: float,
) -> dict[str, _RawTargetSupervision]:
    """Upload authenticated raw-target references and scales exactly once."""
    supervision: dict[str, _RawTargetSupervision] = {}
    for sample in dataset.samples:
        key = sample.transition.transition_sha256
        reference_F = torch.tensor(sample.reference_F, dtype=torch.float64, device=device)
        normalizer = torch.tensor(
            max(sample.raw_deformation_gradient_observed_loss, floor),
            dtype=torch.float64,
            device=device,
        )
        supervision[key] = _RawTargetSupervision(reference_F=reference_F, normalizer=normalizer)
    return supervision


def _prepare_resident_prediction_input(
    solvers: Mapping[tuple[int, ...], torch_solver.SolverState],
    static: PRHistoryStaticBundle,
    sample: _PreparedSample,
    device: torch.device,
) -> _ResidentPredictionInput:
    """Upload one sample outside the hot resident inference interval."""
    return _ResidentPredictionInput(
        state=solvers[sample.pin_signature],
        x_current=torch.tensor(sample.x_current[None], dtype=torch.float64, device=device),
        x_previous=torch.tensor(sample.x_previous[None], dtype=torch.float64, device=device),
        force=torch.tensor(static.external_force[None], dtype=torch.float64, device=device),
        gravity=torch.tensor(static.gravity, dtype=torch.float64, device=device),
        mu=torch.tensor(static.tet_materials[:, 0], dtype=torch.float32, device=device),
        lam=torch.tensor(static.tet_materials[:, 1], dtype=torch.float32, device=device),
        pin=torch.tensor(_pin_flag(sample.pinned_indices, static.tet_indices), dtype=torch.float32, device=device),
        pinned_targets=torch.tensor(sample.pin_targets[None], dtype=torch.float64, device=device),
    )


def _resident_prediction(predictor: StretchPredictor, inputs: _ResidentPredictionInput) -> torch.Tensor:
    """Run one predictor pass and projection from resident device tensors."""
    target_F = predictor.predict_deformation_gradient(
        inputs.state,
        inputs.x_current,
        inputs.x_previous,
        inputs.force,
        inputs.gravity,
        inputs.mu,
        inputs.lam,
        inputs.pin,
    )
    return torch_solver.project_deformation_gradient(inputs.state, target_F, inputs.pinned_targets)[0]


def _grouped_prediction_with_targets(
    predictor: StretchPredictor,
    solvers: Mapping[tuple[int, ...], torch_solver.SolverState],
    static: PRHistoryStaticBundle,
    samples: Sequence[_PreparedSample],
    device: torch.device,
    *,
    decode: bool = True,
) -> tuple[list[torch.Tensor] | None, list[torch.Tensor]]:
    """Predict raw target fields and optionally decode positions in input order."""
    groups: dict[tuple[int, ...], list[tuple[int, _PreparedSample]]] = defaultdict(list)
    for output_index, sample in enumerate(samples):
        groups[sample.pin_signature].append((output_index, sample))

    output: list[torch.Tensor | None] | None = [None] * len(samples) if decode else None
    target_output: list[torch.Tensor | None] = [None] * len(samples)
    force_one = torch.tensor(static.external_force, dtype=torch.float64, device=device)
    gravity = torch.tensor(static.gravity, dtype=torch.float64, device=device)
    mu = torch.tensor(static.tet_materials[:, 0], dtype=torch.float32, device=device)
    lam = torch.tensor(static.tet_materials[:, 1], dtype=torch.float32, device=device)

    for signature, indexed_samples in groups.items():
        state = solvers[signature]
        group_samples = [sample for _index, sample in indexed_samples]
        x_current = torch.as_tensor(
            np.stack([sample.x_current for sample in group_samples]),
            dtype=torch.float64,
            device=device,
        )
        x_previous = torch.as_tensor(
            np.stack([sample.x_previous for sample in group_samples]),
            dtype=torch.float64,
            device=device,
        )
        force = force_one[None].expand(len(group_samples), -1, -1)
        pin = torch.as_tensor(
            _pin_flag(group_samples[0].pinned_indices, static.tet_indices),
            dtype=torch.float32,
            device=device,
        )
        target_F = predictor.predict_deformation_gradient(
            state,
            x_current,
            x_previous,
            force,
            gravity,
            mu,
            lam,
            pin,
        )
        predicted = None
        if decode:
            pinned_targets = torch.as_tensor(
                np.stack([sample.pin_targets for sample in group_samples]),
                dtype=torch.float64,
                device=device,
            )
            predicted = torch_solver.project_deformation_gradient(state, target_F, pinned_targets)
        for local_index, (output_index, _sample) in enumerate(indexed_samples):
            if output is not None and predicted is not None:
                output[output_index] = predicted[local_index]
            target_output[output_index] = target_F[local_index]

    if (output is not None and any(value is None for value in output)) or any(value is None for value in target_output):
        raise RuntimeError("internal grouped prediction did not fill every sample")
    return (
        None if output is None else [value for value in output if value is not None],
        [value for value in target_output if value is not None],
    )


def _grouped_prediction(
    predictor: StretchPredictor,
    solvers: Mapping[tuple[int, ...], torch_solver.SolverState],
    static: PRHistoryStaticBundle,
    samples: Sequence[_PreparedSample],
    device: torch.device,
) -> list[torch.Tensor]:
    """Predict decoded positions while preserving the established interface."""
    predictions, _target_fields = _grouped_prediction_with_targets(predictor, solvers, static, samples, device)
    if predictions is None:
        raise RuntimeError("decoded grouped prediction unexpectedly omitted positions")
    return predictions


def _sample_loss(
    prediction: torch.Tensor,
    sample: _PreparedSample,
    state: torch_solver.SolverState,
    static: PRHistoryStaticBundle,
    characteristic_length_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.tensor(sample.reference_positions, dtype=torch.float64, device=prediction.device)
    mass = torch.tensor(static.mass, dtype=torch.float64, device=prediction.device)
    difference = prediction[state.free] - reference[state.free]
    free_mass = mass[state.free]
    position_loss = (free_mass[:, None] * difference.square()).sum()
    position_loss = position_loss / (free_mass.sum() * characteristic_length_m**2)

    predicted_F = torch_solver.compute_F(prediction, state.tets, state.J)
    reference_F = torch_solver.compute_F(reference, state.tets, state.J)
    deformation_loss = (state.w[:, None, None] * (predicted_F - reference_F).square()).sum()
    deformation_loss = deformation_loss / (9.0 * state.w.sum())
    return position_loss, deformation_loss


def _normalized_raw_deformation_gradient_loss(
    target_F: torch.Tensor,
    state: torch_solver.SolverState,
    supervision: _RawTargetSupervision,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise the pre-projection full-gradient target with a local scale.

    The numerator is the rest-volume-weighted mean squared component error to
    the accepted reference deformation gradient. Its normalizer is the same
    error between the observed post-callback state and that reference, floored
    by ``floor``. On a healthy floor-inactive state, a zero-head full-gradient
    model starts at normalized loss one. The v3 reconstruction can introduce
    spectral roundoff, while v4 returns the observed gradient directly. All
    kinematic tensors remain float64 even though neural features are float32.
    """
    target_F = target_F.to(dtype=torch.float64)
    weight = state.w[:, None, None]
    denominator = 9.0 * state.w.sum()
    raw_loss = (weight * (target_F - supervision.reference_F).square()).sum() / denominator
    return raw_loss / supervision.normalizer, raw_loss, supervision.normalizer


def _loss_contract(
    train_config: PRV3TrainingConfig,
    characteristic_length_m: float,
) -> dict[str, object]:
    """Return the exact authenticated loss semantics for schema v3."""
    return {
        "active_mode": train_config.loss_mode,
        "default_mode": _DECODED_LOSS_MODE,
        "default_compatibility": "preserve decoded position/deformation supervision unless explicitly opted in",
        "position": "sum_free(m_i*||x_i-x_ref_i||^2)/(sum_free(m_i)*rms_rest_edge^2)",
        "decoded_deformation_gradient": "sum_t(V_t*||F_t(x_projected)-F_ref_t||_F^2)/(9*sum_t(V_t))",
        "normalized_raw_deformation_gradient": (
            "raw(F_target,F_ref)/max(raw(F_observed,F_ref),raw_deformation_gradient_floor), "
            "raw(A,B)=sum_t(V_t*||A_t-B_t||_F^2)/(9*sum_t(V_t))"
        ),
        "raw_deformation_gradient_floor": train_config.raw_deformation_gradient_floor,
        "raw_deformation_gradient_precision": "float64 target/reference/observed fields and rest-volume reduction",
        "characteristic_length_m": characteristic_length_m,
        "pins": "transition-local exact Dirichlet indices and targets excluded from position loss",
        "common_objective_role": "independent evaluation gate, not a term in either authenticated training mode",
    }


def _training_work_contract(
    train_config: PRV3TrainingConfig,
    available_pin_signature_count: int,
) -> dict[str, object]:
    """Describe grouped calls exactly, including mixed pin signatures."""
    if (
        isinstance(available_pin_signature_count, bool)
        or not isinstance(available_pin_signature_count, int)
        or available_pin_signature_count < 1
    ):
        raise ValueError("available_pin_signature_count must be a positive integer")
    decoded_mode = train_config.loss_mode == _DECODED_LOSS_MODE
    grouped_count = "distinct_pin_signature_count_in_sampled_batch"
    maximum_group_count = min(train_config.batch_size, available_pin_signature_count)
    return {
        "grouping": "one predictor call per distinct pin signature represented in the sampled batch",
        "available_pin_signature_count": available_pin_signature_count,
        "predictor_passes_per_step": {
            "count": grouped_count,
            "minimum": 1,
            "maximum": maximum_group_count,
            "maximum_scope": "exact upper bound for the authenticated training selection",
        },
        "global_triangular_solves_per_step": {
            "count": grouped_count if decoded_mode else 0,
            "minimum": 1 if decoded_mode else 0,
            "maximum": maximum_group_count if decoded_mode else 0,
            "maximum_scope": "exact upper bound for the authenticated training selection",
        },
        "decoded_position_loss_evaluated": decoded_mode,
        "raw_target_deformation_gradient_loss_evaluated": not decoded_mode,
        "setup_scope": (
            "one dense projection factorization per available pin signature is built before the timed training loop, "
            "including in raw-target mode"
        ),
    }


def _adamw_contract(train_config: PRV3TrainingConfig) -> dict[str, object]:
    """Return the explicit AdamW settings used by both training paths."""
    return {
        "optimizer": "torch.optim.AdamW",
        "learning_rate": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }


def _milestone_seed_contract(train_config: PRV3TrainingConfig) -> dict[str, object]:
    """Return the exact seed and post-initialization stochasticity contract."""
    return {
        "numpy_generator": "PCG64",
        "numpy_generator_seed": train_config.seed,
        "torch_manual_seed": train_config.seed,
        "torch_cuda_manual_seed_all": train_config.seed,
        "post_initialization_stochastic_layers": "none; dropout is exactly zero",
        "resume_sampling": "reconstruct complete authenticated batch stream and continue at next_batch_index",
    }


def _optimizer_parameter_binding(
    predictor: StretchPredictor,
    state_dict: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, object],
) -> dict[str, object]:
    """Bind serialized AdamW parameter IDs to ordered learned tensor names."""
    param_groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    if (
        not isinstance(param_groups, Sequence)
        or isinstance(param_groups, (str, bytes))
        or len(param_groups) != 1
        or not isinstance(param_groups[0], Mapping)
        or not isinstance(state, Mapping)
    ):
        raise ValueError("AdamW state must contain exactly one parameter group and a state mapping")
    parameter_ids = param_groups[0].get("params")
    if not isinstance(parameter_ids, Sequence) or isinstance(parameter_ids, (str, bytes)):
        raise ValueError("AdamW parameter group has no ordered parameter IDs")
    parameter_ids = list(parameter_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in parameter_ids):
        raise ValueError("AdamW parameter IDs must be integers")
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("AdamW parameter IDs must be unique")
    named_parameters = list(predictor.model.named_parameters())
    if len(named_parameters) != len(parameter_ids):
        raise ValueError("AdamW parameter count differs from ordered model parameters")
    if set(state) != set(parameter_ids):
        raise ValueError("AdamW state keys do not exactly cover the ordered parameter group")

    parameters: list[dict[str, object]] = []
    for parameter_id, (name, parameter) in zip(parameter_ids, named_parameters, strict=True):
        model_tensor = state_dict.get(name)
        if not isinstance(model_tensor, torch.Tensor):
            raise ValueError(f"model state_dict is missing ordered parameter {name!r}")
        if model_tensor.shape != parameter.shape or model_tensor.dtype != parameter.dtype:
            raise ValueError(f"model state_dict parameter {name!r} differs from the live model")
        parameters.append(
            {
                "parameter_id": parameter_id,
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "model_tensor_sha256": _array_digest(model_tensor.detach().cpu().contiguous().numpy()),
            }
        )
    if list(state_dict) != [record["name"] for record in parameters]:
        raise ValueError("schema-v4 model state_dict must contain exactly the learned parameter tensors")
    binding: dict[str, object] = {
        "contract": _OPTIMIZER_PARAMETER_BINDING_CONTRACT,
        "ordered_parameters": parameters,
    }
    binding["binding_sha256"] = _canonical_digest(binding)
    return binding


def _external_parent_lineage(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Describe a resume parent whose file/payload must be pinned externally."""
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("resume parent is missing metadata")
    progress = metadata.get("training_progress")
    sampling = metadata.get("sampling")
    selection = metadata.get("transition_selection")
    if not isinstance(progress, Mapping) or not isinstance(sampling, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("resume parent is missing progress, sampling, or selection provenance")
    return {
        "contract": _EXTERNAL_PARENT_LINEAGE_CONTRACT,
        "verification_scope": _EXTERNAL_PARENT_LINEAGE_SCOPE,
        "parent_checkpoint_payload_sha256": checkpoint.get("checkpoint_payload_sha256"),
        "parent_completed_updates": progress.get("completed_updates"),
        "parent_training_progress_sha256": _canonical_digest(progress),
        "parent_state_dict_sha256": checkpoint.get("state_dict_sha256"),
        "parent_optimizer_state_sha256": checkpoint.get("optimizer_state_sha256"),
        "parent_rng_state_sha256": checkpoint.get("rng_state_sha256"),
        "parent_diagnostic_series_sha256": checkpoint.get("diagnostic_series_sha256"),
        "parent_restart_state_sha256": checkpoint.get("restart_state_sha256"),
        "parent_metadata_sha256": checkpoint.get("metadata_sha256"),
        "parent_sampling_sha256": sampling.get("sampling_sha256"),
        "parent_training_selection_sha256": selection.get("selection_sha256"),
    }


def _restart_reproducibility_contract(device_type: str) -> dict[str, object]:
    """Describe the device-specific, deliberately bounded restart claim."""
    if device_type == "cpu":
        return {
            "mode": "bit-exact-single-thread-cpu",
            "explicit_state_restored": [
                "complete learned model state",
                "AdamW state",
                "Torch CPU RNG state",
                "authenticated sampler progress",
                "all-step diagnostic series",
            ],
            "required_torch_cpu_thread_count": 1,
            "comparison": "state_dict and AdamW digests must match uninterrupted continuation exactly",
        }
    if device_type == "cuda":
        return {
            "mode": "authenticated-stateful-cuda-tolerance-repeatable",
            "explicit_state_restored": [
                "complete learned model state",
                "AdamW state",
                "Torch CPU and CUDA RNG states",
                "authenticated sampler progress",
                "all-step diagnostic series",
            ],
            "non_bit_exact_cause": (
                "hierarchy aggregation uses CUDA index-add/atomic reductions whose order is not bit-exact"
            ),
            "claim_boundary": (
                "checkpoint state and next batch are exact; post-resume learned tensors require an "
                "explicit experiment-calibrated tolerance and are not claimed bitwise equal"
            ),
        }
    raise ValueError(f"unsupported restart device type {device_type!r}")


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return an exact linear-interpolation quantile from sorted values."""
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty series")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _series_summary(values: Sequence[float], *, clipping_threshold: float | None = None) -> dict[str, object]:
    """Summarize every finite non-negative observation without subsampling."""
    copied = [float(value) for value in values]
    if not copied:
        raise ValueError("diagnostic series must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in copied):
        raise ValueError("diagnostic series must contain finite non-negative values")
    ordered = sorted(copied)
    summary: dict[str, object] = {
        "count": len(copied),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": statistics.fmean(copied),
        "quantile_method": "linear interpolation at p*(n-1)",
        "quantiles": {
            format(probability, ".12g"): _linear_quantile(ordered, probability) for probability in _DIAGNOSTIC_QUANTILES
        },
    }
    if clipping_threshold is not None:
        if not math.isfinite(clipping_threshold) or clipping_threshold < 0.0:
            raise ValueError("clipping threshold must be finite and non-negative")
        clipped_count = sum(value > clipping_threshold for value in copied)
        summary.update(
            {
                "clipping_threshold": clipping_threshold,
                "clipped_count": clipped_count,
                "clipped_fraction": clipped_count / len(copied),
                "clipped_definition": "gradient_norm_before_clipping > clipping_threshold",
            }
        )
    return summary


def _diagnostic_record(
    gradient_norms: Sequence[float],
    parameter_update_norms: Sequence[float] | None,
    train_config: PRV3TrainingConfig,
) -> dict[str, object]:
    """Return recomputable all-update optimization diagnostics."""
    return {
        "scope": "every completed optimizer update, without log-interval subsampling",
        "gradient_norm_before_clipping": _series_summary(
            gradient_norms,
            clipping_threshold=train_config.gradient_clip_norm,
        ),
        "parameter_update_norm": (None if parameter_update_norms is None else _series_summary(parameter_update_norms)),
    }


def _prefix_exposure_record(
    schedule: _PhaseBalancedSchedule,
    completed_updates: int,
) -> dict[str, object]:
    """Record exact sample exposure through one restart boundary."""
    batches = schedule.batches[:completed_updates]
    sample_count = int(schedule.record["sample_count"])
    exposures = np.bincount(batches.reshape(-1), minlength=sample_count)
    transition_hashes = schedule.record["transition_sha256"]
    if not isinstance(transition_hashes, Sequence):
        raise RuntimeError("phase-balanced schedule transition hashes are malformed")
    return {
        "completed_updates": completed_updates,
        "completed_presentations": int(batches.size),
        "exposure_by_sample_index": [int(value) for value in exposures],
        "exposure_by_transition_sha256": {
            str(transition_hash): int(exposures[index]) for index, transition_hash in enumerate(transition_hashes)
        },
        "minimum_exposure": int(exposures.min()),
        "maximum_exposure": int(exposures.max()),
        "equal_exposure_at_this_milestone": bool(np.all(exposures == exposures[0])),
    }


def train_pr_history_v3(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    *,
    graph_config: GraphTransformerConfig | None = None,
    training_config: PRV3TrainingConfig | None = None,
    device: torch.device | str = "cpu",
    output_path: pathlib.Path | str | None = None,
) -> PRV3TrainingResult:
    """Train architecture v3 or v4 on authenticated PR transitions.

    Args:
        history: Exact PR schedule and static common-objective bundle.
        transitions: One or more accepted same-topology transition samples.
        graph_config: Exact v3 or v4 graph configuration. When omitted, the v3
            compatibility default is used with the canonical float32 timestep.
        training_config: Optimizer, seed, losses, and projection backend.
        device: Torch device for predictor and dense projection.
        output_path: Optional checkpoint path written with :func:`torch.save`.

    Returns:
        Predictor plus authenticated in-memory checkpoint.
    """
    output = None if output_path is None else pathlib.Path(output_path)
    if output is not None and output.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint {output}")
    source_provenance_start = _source_provenance()
    if source_provenance_start != _SOURCE_PROVENANCE_AT_IMPORT:
        raise RuntimeError(
            "Newton source changed after this trainer module was imported; "
            "restart from one settled source tree before training"
        )
    dataset = _prepare_dataset(history, transitions)
    train_config = PRV3TrainingConfig() if training_config is None else training_config
    dt = history.manifest.dt_seconds
    model_config = GraphTransformerConfig(dt=dt) if graph_config is None else graph_config
    _validate_graph_config(model_config, dt)
    device = torch.device(device)
    rng = _seed_everything(train_config.seed)

    predictor, solvers = _build_predictor_and_solvers(dataset, model_config, device)
    decoded_mode = train_config.loss_mode == _DECODED_LOSS_MODE
    raw_supervision = (
        {}
        if decoded_mode
        else _prepare_raw_target_supervision(
            dataset,
            device,
            train_config.raw_deformation_gradient_floor,
        )
    )
    predictor.train()
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    decoder_work = _decoder_work(train_config.projection_backend)
    available_pin_signature_count = len({sample.pin_signature for sample in dataset.samples})

    log: list[dict[str, object]] = []
    start = time.perf_counter()
    sample_count = len(dataset.samples)
    for step in range(train_config.steps):
        selected_indices = rng.choice(
            sample_count,
            size=train_config.batch_size,
            replace=train_config.batch_size > sample_count,
        )
        selected = [dataset.samples[int(index)] for index in selected_indices]
        optimizer.zero_grad(set_to_none=True)
        predictions, target_fields = _grouped_prediction_with_targets(
            predictor,
            solvers,
            dataset.static,
            selected,
            device,
            decode=decoded_mode,
        )
        position_loss = torch.zeros((), dtype=torch.float64, device=device)
        deformation_loss = torch.zeros((), dtype=torch.float64, device=device)
        normalized_raw_F_loss = torch.zeros((), dtype=torch.float64, device=device)
        raw_F_loss = torch.zeros((), dtype=torch.float64, device=device)
        raw_F_normalizers: list[torch.Tensor] = []
        for sample_index, (target_F, sample) in enumerate(zip(target_fields, selected, strict=True)):
            state = solvers[sample.pin_signature]
            if predictions is not None:
                sample_position, sample_deformation = _sample_loss(
                    predictions[sample_index],
                    sample,
                    state,
                    dataset.static,
                    dataset.characteristic_length_m,
                )
                position_loss = position_loss + sample_position
                deformation_loss = deformation_loss + sample_deformation
            if not decoded_mode:
                sample_normalized_raw_F, sample_raw_F, sample_raw_F_normalizer = (
                    _normalized_raw_deformation_gradient_loss(
                        target_F,
                        state,
                        raw_supervision[sample.transition.transition_sha256],
                    )
                )
                normalized_raw_F_loss = normalized_raw_F_loss + sample_normalized_raw_F
                raw_F_loss = raw_F_loss + sample_raw_F
                raw_F_normalizers.append(sample_raw_F_normalizer)
        position_loss = position_loss / len(selected)
        deformation_loss = deformation_loss / len(selected)
        normalized_raw_F_loss = normalized_raw_F_loss / len(selected)
        raw_F_loss = raw_F_loss / len(selected)
        if decoded_mode:
            loss = (
                train_config.position_loss_weight * position_loss
                + train_config.deformation_gradient_loss_weight * deformation_loss
            )
        else:
            loss = normalized_raw_F_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), train_config.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite predictor gradient at step {step}")
        optimizer.step()

        if step % train_config.log_every == 0 or step + 1 == train_config.steps:
            log_entry: dict[str, object] = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_mode": train_config.loss_mode,
                "gradient_norm_before_clipping": float(gradient_norm.detach()),
                "sample_indices": [int(index) for index in selected_indices],
                "transition_sha256": [sample.transition.transition_sha256 for sample in selected],
            }
            if decoded_mode:
                log_entry.update(
                    {
                        "normalized_position_loss": float(position_loss.detach()),
                        "volume_weighted_deformation_gradient_loss": float(deformation_loss.detach()),
                    }
                )
            else:
                log_entry.update(
                    {
                        "normalized_raw_deformation_gradient_loss": float(normalized_raw_F_loss.detach()),
                        "raw_target_deformation_gradient_loss": float(raw_F_loss.detach()),
                        "raw_deformation_gradient_normalizers": [
                            float(normalizer.detach()) for normalizer in raw_F_normalizers
                        ],
                    }
                )
            log.append(log_entry)

    _synchronize(device)
    train_seconds = time.perf_counter() - start
    source_provenance_end = _source_provenance()
    if source_provenance_end != source_provenance_start:
        raise RuntimeError("Newton source changed during training; refusing to authenticate the result")
    predictor.eval()
    state_dict = {name: value.detach().cpu().clone() for name, value in predictor.model.state_dict().items()}
    state_dict_sha256 = _state_dict_digest(state_dict)
    predictor_config = predictor.checkpoint_config()
    realized_levels = int(predictor.model.n_levels)
    metadata: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _CHECKPOINT_CONTRACT,
        "history_manifest": history.manifest.as_dict(),
        "static_bundle": dataset.static.as_dict(),
        "transition_selection": dataset.selection_record,
        "predictor_config": predictor_config,
        "training_realized_hierarchy_levels": realized_levels,
        "decoder_work": decoder_work,
        "training_work": _training_work_contract(train_config, available_pin_signature_count),
        "training_config": dataclasses.asdict(train_config),
        "seed_contract": {
            "numpy_generator": "PCG64",
            "numpy_generator_seed": train_config.seed,
            "torch_manual_seed": train_config.seed,
            "torch_cuda_manual_seed_all": train_config.seed,
        },
        "loss_contract": _loss_contract(train_config, dataset.characteristic_length_m),
        "training_log": log,
        "runtime": {
            "train_seconds": train_seconds,
            "device_type": device.type,
            "parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        },
        "source_provenance": source_provenance_end,
        "source_execution_binding": {
            "module_import": _SOURCE_PROVENANCE_AT_IMPORT,
            "training_start": source_provenance_start,
            "training_end": source_provenance_end,
            "stable": True,
        },
        "software": {"torch_version": str(torch.__version__), "numpy_version": str(np.__version__)},
    }
    if device.type == "cuda":
        metadata["runtime"]["device_name"] = torch.cuda.get_device_name(device)
    metadata_sha256 = _canonical_digest(metadata)
    checkpoint: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _CHECKPOINT_CONTRACT,
        "state_dict": state_dict,
        "state_dict_sha256": state_dict_sha256,
        "metadata": metadata,
        "metadata_sha256": metadata_sha256,
        "predictor_config": predictor_config,
        "training_realized_hierarchy_levels": realized_levels,
        "decoder_work": decoder_work,
    }
    checkpoint["checkpoint_payload_sha256"] = _canonical_digest(
        {
            "contract": _CHECKPOINT_CONTRACT,
            "state_dict_sha256": state_dict_sha256,
            "metadata_sha256": metadata_sha256,
        }
    )
    if _source_provenance() != source_provenance_end:
        raise RuntimeError("Newton source changed while serializing training metadata; refusing to publish it")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also closes the race between the early check and
        # a different process publishing the same experiment path.
        with output.open("xb") as checkpoint_file:
            torch.save(checkpoint, checkpoint_file)
    return PRV3TrainingResult(predictor=predictor, checkpoint=checkpoint)


def _validate_milestone_training_scope(
    graph_config: GraphTransformerConfig,
    train_config: PRV3TrainingConfig,
    milestone_config: PRV4MilestoneConfig,
) -> None:
    """Reject any configuration outside the controlled milestone-v4 scope."""
    if graph_config.architecture_version != 4:
        raise ValueError("milestone training requires graph-transformer architecture version 4")
    if graph_config.dropout != 0.0:
        raise ValueError("milestone training requires dropout=0 for controlled restart semantics")
    if train_config.loss_mode != _NORMALIZED_RAW_F_LOSS_MODE:
        raise ValueError("milestone training requires normalized raw deformation-gradient loss")
    if milestone_config.milestone_updates[-1] != train_config.steps:
        raise ValueError("the final milestone update must exactly equal training_config.steps")
    if any(update > train_config.steps for update in milestone_config.milestone_updates):
        raise ValueError("milestone update exceeds training_config.steps")


def _raw_f_training_batch_loss(
    predictor: StretchPredictor,
    solvers: Mapping[tuple[int, ...], torch_solver.SolverState],
    dataset: _PreparedDataset,
    selected: Sequence[_PreparedSample],
    raw_supervision: Mapping[str, _RawTargetSupervision],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Evaluate one raw-F batch without invoking the global projection."""
    predictions, target_fields = _grouped_prediction_with_targets(
        predictor,
        solvers,
        dataset.static,
        selected,
        device,
        decode=False,
    )
    if predictions is not None:
        raise RuntimeError("raw-F milestone training unexpectedly decoded positions")
    normalized_loss = torch.zeros((), dtype=torch.float64, device=device)
    raw_loss = torch.zeros((), dtype=torch.float64, device=device)
    normalizers: list[torch.Tensor] = []
    for target_F, sample in zip(target_fields, selected, strict=True):
        sample_normalized, sample_raw, sample_normalizer = _normalized_raw_deformation_gradient_loss(
            target_F,
            solvers[sample.pin_signature],
            raw_supervision[sample.transition.transition_sha256],
        )
        normalized_loss = normalized_loss + sample_normalized
        raw_loss = raw_loss + sample_raw
        normalizers.append(sample_normalizer)
    return normalized_loss / len(selected), raw_loss / len(selected), normalizers


def _milestone_checkpoint_path(directory: pathlib.Path, completed_updates: int) -> pathlib.Path:
    return directory / f"checkpoint-update-{completed_updates:08d}.pt"


def _build_milestone_checkpoint(
    *,
    predictor: StretchPredictor,
    optimizer: torch.optim.Optimizer,
    history: PRSceneHistory,
    dataset: _PreparedDataset,
    train_config: PRV3TrainingConfig,
    milestone_config: PRV4MilestoneConfig,
    schedule: _PhaseBalancedSchedule,
    completed_updates: int,
    training_log: Sequence[Mapping[str, object]],
    gradient_norms: Sequence[float],
    parameter_update_norms: Sequence[float] | None,
    source_provenance: Mapping[str, str | None],
    device: torch.device,
    process_start_update: int,
    process_train_seconds: float,
    external_parent_lineage: Mapping[str, object] | None,
) -> dict[str, object]:
    """Freeze one inference and restart state at an optimizer milestone."""
    predictor_config = predictor.checkpoint_config()
    realized_levels = int(predictor.model.n_levels)
    state_dict = {name: value.detach().cpu().clone() for name, value in predictor.model.state_dict().items()}
    state_dict_sha256 = _state_dict_digest(state_dict)
    optimizer_state_value = _clone_state_tree(optimizer.state_dict())
    if not isinstance(optimizer_state_value, dict):
        raise RuntimeError("cloned optimizer state is not a dictionary")
    optimizer_state_sha256 = _state_tree_digest(optimizer_state_value)
    optimizer_parameter_binding = _optimizer_parameter_binding(
        predictor,
        state_dict,
        optimizer_state_value,
    )
    rng_state: dict[str, object] = {
        "device_type": device.type,
        "torch_cpu_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "torch_device_rng_state": (
            torch.cuda.get_rng_state(device).detach().cpu().clone() if device.type == "cuda" else None
        ),
    }
    rng_state_sha256 = _state_tree_digest(rng_state)
    diagnostic_series: dict[str, object] = {
        "gradient_norm_before_clipping": [float(value) for value in gradient_norms],
        "parameter_update_norm": (
            None if parameter_update_norms is None else [float(value) for value in parameter_update_norms]
        ),
    }
    diagnostic_series_sha256 = _canonical_digest(diagnostic_series)
    progress = {
        "target_updates": train_config.steps,
        "completed_updates": completed_updates,
        "next_batch_index": completed_updates,
        "is_final": completed_updates == train_config.steps,
        "external_parent_lineage": (None if external_parent_lineage is None else dict(external_parent_lineage)),
        "sampling_prefix": _prefix_exposure_record(schedule, completed_updates),
    }
    restart_state = {
        "completed_updates": completed_updates,
        "next_batch_index": completed_updates,
        "sampling_sha256": schedule.record["sampling_sha256"],
        "batch_stream_sha256": schedule.record["batch_stream_sha256"],
        "optimizer_state_sha256": optimizer_state_sha256,
        "rng_state_sha256": rng_state_sha256,
        "diagnostic_series_sha256": diagnostic_series_sha256,
    }
    restart_state_sha256 = _canonical_digest(restart_state)
    metadata: dict[str, object] = {
        "schema_version": _MILESTONE_SCHEMA_VERSION,
        "contract": _MILESTONE_CHECKPOINT_CONTRACT,
        "history_manifest": history.manifest.as_dict(),
        "static_bundle": dataset.static.as_dict(),
        "transition_selection": dataset.selection_record,
        "predictor_config": predictor_config,
        "training_realized_hierarchy_levels": realized_levels,
        "decoder_work": _decoder_work(train_config.projection_backend),
        "training_work": _training_work_contract(
            train_config,
            len({sample.pin_signature for sample in dataset.samples}),
        ),
        "training_config": dataclasses.asdict(train_config),
        "milestone_config": dataclasses.asdict(milestone_config),
        "optimizer_contract": _adamw_contract(train_config),
        "optimizer_parameter_binding": optimizer_parameter_binding,
        "sampling": schedule.record,
        "training_progress": progress,
        "restart_reproducibility": _restart_reproducibility_contract(device.type),
        "seed_contract": _milestone_seed_contract(train_config),
        "loss_contract": _loss_contract(train_config, dataset.characteristic_length_m),
        "training_log": [dict(entry) for entry in training_log],
        "optimization_diagnostics": _diagnostic_record(
            gradient_norms,
            parameter_update_norms,
            train_config,
        ),
        "runtime": {
            "process_start_update": process_start_update,
            "process_completed_update": completed_updates,
            "process_wall_seconds_including_prior_milestone_io": process_train_seconds,
            "wall_time_scope": (
                "optimizer work plus any earlier checkpoint verification and serialization in this process; "
                "not a pure training-throughput measurement"
            ),
            "device_type": device.type,
            "torch_cpu_thread_count": torch.get_num_threads(),
            "parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        },
        "source_provenance": dict(source_provenance),
        "source_execution_binding": {
            "module_import": dict(source_provenance),
            "training_start": dict(source_provenance),
            "training_end": dict(source_provenance),
            "stable": True,
        },
        "software": {"torch_version": str(torch.__version__), "numpy_version": str(np.__version__)},
    }
    if device.type == "cuda":
        metadata["runtime"]["device_name"] = torch.cuda.get_device_name(device)
    metadata_sha256 = _canonical_digest(metadata)
    checkpoint: dict[str, object] = {
        "schema_version": _MILESTONE_SCHEMA_VERSION,
        "contract": _MILESTONE_CHECKPOINT_CONTRACT,
        "state_dict": state_dict,
        "state_dict_sha256": state_dict_sha256,
        "optimizer_state": optimizer_state_value,
        "optimizer_state_sha256": optimizer_state_sha256,
        "rng_state": rng_state,
        "rng_state_sha256": rng_state_sha256,
        "diagnostic_series": diagnostic_series,
        "diagnostic_series_sha256": diagnostic_series_sha256,
        "restart_state": restart_state,
        "restart_state_sha256": restart_state_sha256,
        "metadata": metadata,
        "metadata_sha256": metadata_sha256,
        "predictor_config": predictor_config,
        "training_realized_hierarchy_levels": realized_levels,
        "decoder_work": metadata["decoder_work"],
    }
    checkpoint["checkpoint_payload_sha256"] = _canonical_digest(
        {
            "contract": _MILESTONE_CHECKPOINT_CONTRACT,
            "state_dict_sha256": state_dict_sha256,
            "optimizer_state_sha256": optimizer_state_sha256,
            "rng_state_sha256": rng_state_sha256,
            "diagnostic_series_sha256": diagnostic_series_sha256,
            "restart_state_sha256": restart_state_sha256,
            "metadata_sha256": metadata_sha256,
        }
    )
    return checkpoint


def train_pr_history_v4_milestones(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    *,
    graph_config: GraphTransformerConfig,
    training_config: PRV3TrainingConfig,
    milestone_config: PRV4MilestoneConfig | None = None,
    device: torch.device | str = "cpu",
    output_directory: pathlib.Path | str,
    resume_from: Mapping[str, object] | pathlib.Path | str | None = None,
    stop_after_update: int | None = None,
) -> PRV4MilestoneTrainingResult:
    """Train v4 raw-F models with exact batches and restartable milestones.

    This is a separate schema-v4 path. The legacy :func:`train_pr_history_v3`
    implementation and its schema-v3 serialization remain unchanged.

    Args:
        history: Exact PR schedule and static common-objective bundle.
        transitions: Accepted same-topology training transitions.
        graph_config: Architecture-v4 graph configuration with zero dropout.
        training_config: Raw-F AdamW settings and total update count.
        milestone_config: Immutable checkpoint and diagnostic settings.
        device: Torch device used for training.
        output_directory: Directory receiving exclusive milestone files.
        resume_from: Optional earlier schema-v4 milestone checkpoint.
        stop_after_update: Optional configured milestone at which this process
            returns, used to exercise or perform a planned restart.

    Returns:
        Predictor and every checkpoint emitted by this process.
    """
    source_provenance_start = _source_provenance()
    if source_provenance_start != _SOURCE_PROVENANCE_AT_IMPORT:
        raise RuntimeError(
            "Newton source changed after this trainer module was imported; "
            "restart from one settled source tree before training"
        )
    dataset = _prepare_dataset(history, transitions)
    _validate_graph_config(graph_config, history.manifest.dt_seconds)
    milestone = PRV4MilestoneConfig() if milestone_config is None else milestone_config
    _validate_milestone_training_scope(graph_config, training_config, milestone)
    schedule = _build_phase_balanced_schedule(dataset, training_config)
    _validate_equal_milestone_exposure(schedule, milestone)
    device = torch.device(device)
    if device.type == "cpu" and torch.get_num_threads() != 1:
        raise RuntimeError("exact CPU milestone restart requires torch.set_num_threads(1) before training")
    _seed_everything(training_config.seed)
    predictor, solvers = _build_predictor_and_solvers(dataset, graph_config, device)
    raw_supervision = _prepare_raw_target_supervision(
        dataset,
        device,
        training_config.raw_deformation_gradient_floor,
    )
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    )

    checkpoint_directory = pathlib.Path(output_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    start_update = 0
    external_parent_lineage: dict[str, object] | None = None
    training_log: list[dict[str, object]] = []
    gradient_norms: list[float] = []
    parameter_update_norms: list[float] | None = [] if milestone.track_parameter_update_norm else None
    if resume_from is not None:
        if isinstance(resume_from, Mapping):
            resume_checkpoint = dict(resume_from)
        else:
            resume_checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
            if not isinstance(resume_checkpoint, dict):
                raise ValueError("resume checkpoint file did not contain a dictionary")
        _verify_checkpoint(resume_checkpoint, dataset)
        if (
            resume_checkpoint.get("schema_version"),
            resume_checkpoint.get("contract"),
        ) != (_MILESTONE_SCHEMA_VERSION, _MILESTONE_CHECKPOINT_CONTRACT):
            raise ValueError("resume checkpoint is not a schema-v4 milestone checkpoint")
        resume_metadata = resume_checkpoint["metadata"]
        if resume_metadata["transition_selection"] != dataset.selection_record:
            raise ValueError("resume checkpoint training selection does not exactly match this run")
        if resume_metadata["predictor_config"] != {
            "kind": "graph-transformer",
            "residual": True,
            "graph_transformer": dataclasses.asdict(graph_config),
        }:
            raise ValueError("resume checkpoint predictor configuration does not exactly match this run")
        if resume_metadata["training_config"] != dataclasses.asdict(training_config):
            raise ValueError("resume checkpoint training configuration does not exactly match this run")
        if resume_metadata["milestone_config"] != dataclasses.asdict(milestone):
            raise ValueError("resume checkpoint milestone configuration does not exactly match this run")
        if resume_metadata["sampling"] != schedule.record:
            raise ValueError("resume checkpoint sampling stream does not exactly match this run")
        if resume_metadata["source_provenance"] != source_provenance_start:
            raise ValueError("resume checkpoint source provenance does not exactly match this run")
        start_update = int(resume_metadata["training_progress"]["completed_updates"])
        if start_update not in milestone.milestone_updates or start_update >= training_config.steps:
            raise ValueError("resume checkpoint is not an eligible non-final configured milestone")
        # Resume is bound to the exact same authenticated static selection.
        # Strict loading restores every learned tensor; the dataset rebuild
        # separately authenticates the nonpersistent topology/hierarchy state.
        predictor.model.load_state_dict(resume_checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        rng_state = resume_checkpoint["rng_state"]
        if rng_state["device_type"] != device.type:
            raise ValueError("resume checkpoint RNG device type does not exactly match this run")
        torch.set_rng_state(rng_state["torch_cpu_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng_state["torch_device_rng_state"], device)
        training_log = [dict(entry) for entry in resume_metadata["training_log"]]
        series = resume_checkpoint["diagnostic_series"]
        gradient_norms = [float(value) for value in series["gradient_norm_before_clipping"]]
        update_values = series["parameter_update_norm"]
        parameter_update_norms = None if update_values is None else [float(value) for value in update_values]
        external_parent_lineage = _external_parent_lineage(resume_checkpoint)

    if stop_after_update is None:
        stop_update = training_config.steps
    else:
        if isinstance(stop_after_update, bool) or not isinstance(stop_after_update, int):
            raise ValueError("stop_after_update must be an integer")
        stop_update = stop_after_update
    if stop_update not in milestone.milestone_updates:
        raise ValueError("stop_after_update must be one of the configured milestones")
    if stop_update <= start_update:
        raise ValueError("stop_after_update must lie after the resume checkpoint")
    for update in milestone.milestone_updates:
        path = _milestone_checkpoint_path(checkpoint_directory, update)
        if start_update < update <= stop_update and path.exists():
            raise FileExistsError(f"refusing to overwrite existing checkpoint {path}")

    predictor.train()
    emitted: dict[int, dict[str, object]] = {}
    emitted_paths: dict[int, pathlib.Path] = {}
    process_start = time.perf_counter()
    trainable_parameters = tuple(parameter for parameter in predictor.parameters() if parameter.requires_grad)
    milestone_set = set(milestone.milestone_updates)
    for step in range(start_update, stop_update):
        selected_indices = schedule.batches[step]
        selected = [dataset.samples[int(index)] for index in selected_indices]
        optimizer.zero_grad(set_to_none=True)
        normalized_raw_F_loss, raw_F_loss, raw_F_normalizers = _raw_f_training_batch_loss(
            predictor,
            solvers,
            dataset,
            selected,
            raw_supervision,
            device,
        )
        if not torch.isfinite(normalized_raw_F_loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        normalized_raw_F_loss.backward()
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            training_config.gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm_tensor):
            raise RuntimeError(f"non-finite predictor gradient at step {step}")
        gradient_norm = float(gradient_norm_tensor.detach())
        gradient_norms.append(gradient_norm)
        before_parameters = (
            [parameter.detach().clone() for parameter in trainable_parameters]
            if parameter_update_norms is not None
            else None
        )
        optimizer.step()
        if parameter_update_norms is not None:
            if before_parameters is None:
                raise RuntimeError("parameter-update tracking lost its pre-update snapshot")
            squared_update = torch.zeros((), dtype=torch.float64, device=device)
            with torch.no_grad():
                for parameter, before in zip(trainable_parameters, before_parameters, strict=True):
                    squared_update = squared_update + (parameter.detach() - before).to(torch.float64).square().sum()
            update_norm = float(torch.sqrt(squared_update).detach())
            if not math.isfinite(update_norm):
                raise RuntimeError(f"non-finite parameter update at step {step}")
            parameter_update_norms.append(update_norm)

        completed_updates = step + 1
        if step % training_config.log_every == 0 or completed_updates in milestone_set:
            training_log.append(
                {
                    "step": step,
                    "completed_updates": completed_updates,
                    "loss": float(normalized_raw_F_loss.detach()),
                    "loss_mode": training_config.loss_mode,
                    "gradient_norm_before_clipping": gradient_norm,
                    "sample_indices": [int(index) for index in selected_indices],
                    "transition_sha256": [sample.transition.transition_sha256 for sample in selected],
                    "normalized_raw_deformation_gradient_loss": float(normalized_raw_F_loss.detach()),
                    "raw_target_deformation_gradient_loss": float(raw_F_loss.detach()),
                    "raw_deformation_gradient_normalizers": [
                        float(normalizer.detach()) for normalizer in raw_F_normalizers
                    ],
                }
            )

        if completed_updates in milestone_set:
            _synchronize(device)
            process_train_seconds = time.perf_counter() - process_start
            source_provenance_now = _source_provenance()
            if source_provenance_now != source_provenance_start:
                raise RuntimeError("Newton source changed during training; refusing to authenticate the result")
            checkpoint = _build_milestone_checkpoint(
                predictor=predictor,
                optimizer=optimizer,
                history=history,
                dataset=dataset,
                train_config=training_config,
                milestone_config=milestone,
                schedule=schedule,
                completed_updates=completed_updates,
                training_log=training_log,
                gradient_norms=gradient_norms,
                parameter_update_norms=parameter_update_norms,
                source_provenance=source_provenance_now,
                device=device,
                process_start_update=start_update,
                process_train_seconds=process_train_seconds,
                external_parent_lineage=external_parent_lineage,
            )
            _verify_checkpoint(checkpoint, dataset)
            if _source_provenance() != source_provenance_start:
                raise RuntimeError("Newton source changed while serializing a milestone checkpoint")
            path = _milestone_checkpoint_path(checkpoint_directory, completed_updates)
            with path.open("xb") as checkpoint_file:
                torch.save(checkpoint, checkpoint_file)
            emitted[completed_updates] = checkpoint
            emitted_paths[completed_updates] = path

    predictor.eval()
    if _source_provenance() != source_provenance_start:
        raise RuntimeError("Newton source changed before milestone training completed")
    return PRV4MilestoneTrainingResult(
        predictor=predictor,
        checkpoints=emitted,
        checkpoint_paths=emitted_paths,
        completed_updates=stop_update,
    )


def _verify_schema_v3_training_contract(
    metadata: Mapping[str, object],
    dataset: _PreparedDataset,
) -> None:
    """Fail closed on the schema-v3 training semantics, not only their hash."""
    training_config_value = metadata.get("training_config")
    if not isinstance(training_config_value, Mapping):
        raise ValueError("schema-v3 checkpoint is missing training_config")
    if "loss_mode" not in training_config_value or "raw_deformation_gradient_floor" not in training_config_value:
        raise ValueError("schema-v3 checkpoint training_config is missing required loss semantics")
    try:
        train_config = PRV3TrainingConfig(**dict(training_config_value))
    except (TypeError, ValueError) as error:
        raise ValueError("schema-v3 checkpoint has an invalid training_config") from error
    if _jsonable(dataclasses.asdict(train_config)) != _jsonable(training_config_value):
        raise ValueError("schema-v3 checkpoint training_config is not canonical")

    expected_loss_contract = _loss_contract(train_config, dataset.characteristic_length_m)
    if _jsonable(metadata.get("loss_contract")) != _jsonable(expected_loss_contract):
        raise ValueError("schema-v3 checkpoint loss_contract disagrees with training_config")

    selection = metadata.get("transition_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("schema-v3 checkpoint is missing its training transition selection")
    if selection.get("contract") != "pr2901-history-selected-transition-set-v2":
        raise ValueError("schema-v3 checkpoint has an unsupported training transition-selection contract")
    if selection.get("provenance_scope") != "selected-content-addressed-transitions-not-a-complete-history-claim":
        raise ValueError("schema-v3 checkpoint has an invalid training transition-selection scope")
    history_manifest = metadata.get("history_manifest")
    static_bundle = metadata.get("static_bundle")
    if not isinstance(history_manifest, Mapping) or not isinstance(static_bundle, Mapping):
        raise ValueError("schema-v3 checkpoint is missing its history or static identity")
    if selection.get("history_manifest_sha256") != history_manifest.get("manifest_sha256"):
        raise ValueError("schema-v3 checkpoint training selection has the wrong history identity")
    if selection.get("static_sha256") != static_bundle.get("static_sha256"):
        raise ValueError("schema-v3 checkpoint training selection has the wrong static identity")
    selection_without_digest = dict(selection)
    selection_sha256 = selection_without_digest.pop("selection_sha256", None)
    if selection_sha256 != _canonical_digest(selection_without_digest):
        raise ValueError("schema-v3 checkpoint training transition selection is not self-consistent")
    transition_records = selection.get("transitions")
    if not isinstance(transition_records, Sequence) or isinstance(transition_records, (str, bytes)):
        raise ValueError("schema-v3 checkpoint training transition records are missing")
    if not transition_records:
        raise ValueError("schema-v3 checkpoint has an empty training transition selection")
    static_arrays = static_bundle.get("arrays")
    if not isinstance(static_arrays, Mapping) or not isinstance(static_arrays.get("rest_q"), Mapping):
        raise ValueError("schema-v3 checkpoint static vertex record is missing")
    rest_shape = static_arrays["rest_q"].get("shape")
    if not isinstance(rest_shape, Sequence) or list(rest_shape)[1:] != [3]:
        raise ValueError("schema-v3 checkpoint static vertex shape is invalid")
    vertex_count = rest_shape[0]
    if isinstance(vertex_count, bool) or not isinstance(vertex_count, int) or vertex_count < 1:
        raise ValueError("schema-v3 checkpoint static vertex count is invalid")
    pin_signatures: set[str] = set()
    transition_hashes: set[str] = set()
    for record in transition_records:
        if not isinstance(record, Mapping):
            raise ValueError("schema-v3 checkpoint has a malformed training transition record")
        digest_names = (
            "transition_sha256",
            "training_record_sha256",
            "input_prefix_sha256",
            "input_state_sha256",
            "scene_sha256",
            "objective_instance_sha256",
            "pin_signature_sha256",
            "observed_F_sha256",
            "reference_F_sha256",
        )
        required = (
            *digest_names,
            "pin_signature",
            "pin_count",
            "raw_deformation_gradient_observed_loss",
        )
        if any(name not in record for name in required):
            raise ValueError("schema-v3 checkpoint training transition record is missing loss provenance")
        for digest_name in digest_names:
            digest = record[digest_name]
            if not _is_sha256(digest):
                raise ValueError(f"schema-v3 checkpoint has an invalid {digest_name}")
        transition_hash = record["transition_sha256"]
        if transition_hash in transition_hashes:
            raise ValueError("schema-v3 checkpoint training transition selection contains duplicates")
        transition_hashes.add(transition_hash)
        pin_signature = record["pin_signature_sha256"]
        pin_indices = record["pin_signature"]
        if not isinstance(pin_indices, Sequence) or isinstance(pin_indices, (str, bytes)):
            raise ValueError("schema-v3 checkpoint has a malformed pin signature")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in pin_indices):
            raise ValueError("schema-v3 checkpoint pin signature contains a non-integer index")
        if list(pin_indices) != sorted(set(pin_indices)):
            raise ValueError("schema-v3 checkpoint pin signature is not sorted and unique")
        if pin_indices and (pin_indices[0] < 0 or pin_indices[-1] >= vertex_count):
            raise ValueError("schema-v3 checkpoint pin signature is outside the static vertex range")
        pin_count = record["pin_count"]
        if isinstance(pin_count, bool) or not isinstance(pin_count, int) or pin_count < 0:
            raise ValueError("schema-v3 checkpoint has an invalid training pin count")
        if pin_count != len(pin_indices):
            raise ValueError("schema-v3 checkpoint training pin count disagrees with its signature")
        pin_array = np.asarray(pin_indices, dtype=np.int64)
        if _array_digest(pin_array) != pin_signature:
            raise ValueError("schema-v3 checkpoint training pin-signature digest is inconsistent")
        raw_observed_loss = record["raw_deformation_gradient_observed_loss"]
        if (
            isinstance(raw_observed_loss, bool)
            or not isinstance(raw_observed_loss, (int, float))
            or not math.isfinite(raw_observed_loss)
        ):
            raise ValueError("schema-v3 checkpoint has an invalid raw deformation-gradient normalizer")
        if raw_observed_loss < 0.0:
            raise ValueError("schema-v3 checkpoint has a negative raw deformation-gradient normalizer")
        pin_signatures.add(pin_signature)
    expected_training_work = _training_work_contract(train_config, len(pin_signatures))
    if _jsonable(metadata.get("training_work")) != _jsonable(expected_training_work):
        raise ValueError("schema-v3 checkpoint training_work disagrees with loss mode or batch grouping")
    source_provenance = metadata.get("source_provenance")
    source_binding = metadata.get("source_execution_binding")
    if not isinstance(source_provenance, Mapping) or not isinstance(source_binding, Mapping):
        raise ValueError("schema-v3 checkpoint is missing its source execution binding")
    expected_source_binding = {
        "module_import": source_provenance,
        "training_start": source_provenance,
        "training_end": source_provenance,
        "stable": True,
    }
    if _jsonable(source_binding) != _jsonable(expected_source_binding):
        raise ValueError("schema-v3 checkpoint source changed while its training process was live")

    training_log = metadata.get("training_log")
    if not isinstance(training_log, Sequence) or isinstance(training_log, (str, bytes)) or not training_log:
        raise ValueError("schema-v3 checkpoint has no authenticated training log")
    decoded_mode = train_config.loss_mode == _DECODED_LOSS_MODE
    for entry in training_log:
        if not isinstance(entry, Mapping) or entry.get("loss_mode") != train_config.loss_mode:
            raise ValueError("schema-v3 checkpoint training log disagrees with its loss mode")
        decoded_fields = ("normalized_position_loss", "volume_weighted_deformation_gradient_loss")
        raw_fields = (
            "normalized_raw_deformation_gradient_loss",
            "raw_target_deformation_gradient_loss",
            "raw_deformation_gradient_normalizers",
        )
        if decoded_mode and (
            any(name not in entry for name in decoded_fields) or any(name in entry for name in raw_fields)
        ):
            raise ValueError("schema-v3 checkpoint decoded training log has inconsistent loss fields")
        if not decoded_mode and (
            any(name in entry for name in decoded_fields) or any(name not in entry for name in raw_fields)
        ):
            raise ValueError("schema-v3 checkpoint raw-target training log has inconsistent loss fields")


def _verify_schema_v4_milestone_contract(
    checkpoint: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Fail closed on restart, sampling, and all-step diagnostic semantics."""
    training_config_value = metadata.get("training_config")
    milestone_config_value = metadata.get("milestone_config")
    predictor_config = metadata.get("predictor_config")
    if not isinstance(training_config_value, Mapping) or not isinstance(milestone_config_value, Mapping):
        raise ValueError("schema-v4 checkpoint is missing training or milestone configuration")
    if not isinstance(predictor_config, Mapping) or not isinstance(predictor_config.get("graph_transformer"), Mapping):
        raise ValueError("schema-v4 checkpoint is missing graph-transformer configuration")
    try:
        train_config = PRV3TrainingConfig(**dict(training_config_value))
        milestone_config = PRV4MilestoneConfig(**dict(milestone_config_value))
        graph_config = GraphTransformerConfig(**dict(predictor_config["graph_transformer"]))
    except (TypeError, ValueError) as error:
        raise ValueError("schema-v4 checkpoint has an invalid authenticated configuration") from error
    if _jsonable(dataclasses.asdict(milestone_config)) != _jsonable(milestone_config_value):
        raise ValueError("schema-v4 checkpoint milestone configuration is not canonical")
    _validate_milestone_training_scope(graph_config, train_config, milestone_config)
    if metadata.get("optimizer_contract") != _adamw_contract(train_config):
        raise ValueError("schema-v4 checkpoint optimizer contract disagrees with training configuration")
    if metadata.get("seed_contract") != _milestone_seed_contract(train_config):
        raise ValueError("schema-v4 checkpoint seed contract disagrees with training configuration")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device_type") not in ("cpu", "cuda"):
        raise ValueError("schema-v4 checkpoint has an invalid runtime device type")
    device_type = str(runtime["device_type"])
    if metadata.get("restart_reproducibility") != _restart_reproducibility_contract(device_type):
        raise ValueError("schema-v4 checkpoint restart reproducibility contract is inconsistent")
    if device_type == "cpu" and runtime.get("torch_cpu_thread_count") != 1:
        raise ValueError("schema-v4 CPU checkpoint was not trained with exactly one Torch thread")

    selection = metadata.get("transition_selection")
    history_manifest = metadata.get("history_manifest")
    if not isinstance(selection, Mapping) or not isinstance(history_manifest, Mapping):
        raise ValueError("schema-v4 checkpoint is missing sampling provenance")
    transition_records = selection.get("transitions")
    substeps_per_frame = history_manifest.get("substeps_per_frame")
    if not isinstance(transition_records, Sequence) or isinstance(transition_records, (str, bytes)):
        raise ValueError("schema-v4 checkpoint training selection is malformed")
    if isinstance(substeps_per_frame, bool) or not isinstance(substeps_per_frame, int):
        raise ValueError("schema-v4 checkpoint history has an invalid substep count")
    expected_schedule = _build_phase_balanced_schedule_from_records(
        transition_records,
        substeps_per_frame=substeps_per_frame,
        steps=train_config.steps,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )
    _validate_equal_milestone_exposure(expected_schedule, milestone_config)
    if _jsonable(metadata.get("sampling")) != _jsonable(expected_schedule.record):
        raise ValueError("schema-v4 checkpoint sampling record is not exactly reproducible")

    progress = metadata.get("training_progress")
    if not isinstance(progress, Mapping):
        raise ValueError("schema-v4 checkpoint is missing training progress")
    completed_updates = progress.get("completed_updates")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates not in milestone_config.milestone_updates
    ):
        raise ValueError("schema-v4 checkpoint completed update is not a configured milestone")
    expected_progress = {
        "target_updates": train_config.steps,
        "completed_updates": completed_updates,
        "next_batch_index": completed_updates,
        "is_final": completed_updates == train_config.steps,
        "external_parent_lineage": progress.get("external_parent_lineage"),
        "sampling_prefix": _prefix_exposure_record(expected_schedule, completed_updates),
    }
    parent_lineage = expected_progress["external_parent_lineage"]
    process_start_update = runtime.get("process_start_update")
    if parent_lineage is None:
        if process_start_update != 0:
            raise ValueError("schema-v4 checkpoint has no external parent for a nonzero process start")
    else:
        if not isinstance(parent_lineage, Mapping):
            raise ValueError("schema-v4 checkpoint has a malformed external parent lineage")
        expected_lineage_keys = {
            "contract",
            "verification_scope",
            "parent_checkpoint_payload_sha256",
            "parent_completed_updates",
            "parent_training_progress_sha256",
            "parent_state_dict_sha256",
            "parent_optimizer_state_sha256",
            "parent_rng_state_sha256",
            "parent_diagnostic_series_sha256",
            "parent_restart_state_sha256",
            "parent_metadata_sha256",
            "parent_sampling_sha256",
            "parent_training_selection_sha256",
        }
        if set(parent_lineage) != expected_lineage_keys:
            raise ValueError("schema-v4 checkpoint external parent lineage has unexpected fields")
        if parent_lineage.get("contract") != _EXTERNAL_PARENT_LINEAGE_CONTRACT:
            raise ValueError("schema-v4 checkpoint has an unsupported external parent-lineage contract")
        if parent_lineage.get("verification_scope") != _EXTERNAL_PARENT_LINEAGE_SCOPE:
            raise ValueError("schema-v4 checkpoint overstates external parent-lineage verification")
        digest_names = (
            "parent_checkpoint_payload_sha256",
            "parent_training_progress_sha256",
            "parent_state_dict_sha256",
            "parent_optimizer_state_sha256",
            "parent_rng_state_sha256",
            "parent_diagnostic_series_sha256",
            "parent_restart_state_sha256",
            "parent_metadata_sha256",
            "parent_sampling_sha256",
            "parent_training_selection_sha256",
        )
        if any(not _is_sha256(parent_lineage.get(name)) for name in digest_names):
            raise ValueError("schema-v4 checkpoint external parent lineage has an invalid digest")
        parent_completed = parent_lineage.get("parent_completed_updates")
        if (
            isinstance(parent_completed, bool)
            or not isinstance(parent_completed, int)
            or parent_completed not in milestone_config.milestone_updates
            or parent_completed >= completed_updates
            or parent_completed != process_start_update
        ):
            raise ValueError("schema-v4 checkpoint external parent progress is inconsistent")
        if parent_lineage.get("parent_sampling_sha256") != expected_schedule.record["sampling_sha256"]:
            raise ValueError("schema-v4 checkpoint external parent used another sampling stream")
        if parent_lineage.get("parent_training_selection_sha256") != selection.get("selection_sha256"):
            raise ValueError("schema-v4 checkpoint external parent used another training selection")
    if _jsonable(progress) != _jsonable(expected_progress):
        raise ValueError("schema-v4 checkpoint training progress is inconsistent")

    optimizer_state = checkpoint.get("optimizer_state")
    optimizer_state_sha256 = checkpoint.get("optimizer_state_sha256")
    if not isinstance(optimizer_state, Mapping) or not _is_sha256(optimizer_state_sha256):
        raise ValueError("schema-v4 checkpoint is missing optimizer restart state")
    if _state_tree_digest(optimizer_state) != optimizer_state_sha256:
        raise ValueError("schema-v4 checkpoint optimizer-state SHA-256 verification failed")
    param_groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    if (
        not isinstance(param_groups, Sequence)
        or isinstance(param_groups, (str, bytes))
        or len(param_groups) != 1
        or not isinstance(param_groups[0], Mapping)
        or not isinstance(state, Mapping)
        or not state
    ):
        raise ValueError("schema-v4 checkpoint AdamW state is malformed")
    group = param_groups[0]
    optimizer_contract = _adamw_contract(train_config)
    expected_group_values = {
        "lr": optimizer_contract["learning_rate"],
        "weight_decay": optimizer_contract["weight_decay"],
        "betas": tuple(optimizer_contract["betas"]),
        "eps": optimizer_contract["eps"],
        "amsgrad": optimizer_contract["amsgrad"],
        "maximize": optimizer_contract["maximize"],
        "foreach": optimizer_contract["foreach"],
        "capturable": optimizer_contract["capturable"],
        "differentiable": optimizer_contract["differentiable"],
        "fused": optimizer_contract["fused"],
        "decoupled_weight_decay": optimizer_contract["decoupled_weight_decay"],
    }
    for name, expected in expected_group_values.items():
        if group.get(name) != expected:
            raise ValueError(f"schema-v4 checkpoint AdamW {name} differs from its contract")

    binding = metadata.get("optimizer_parameter_binding")
    if not isinstance(binding, Mapping) or binding.get("contract") != _OPTIMIZER_PARAMETER_BINDING_CONTRACT:
        raise ValueError("schema-v4 checkpoint is missing its optimizer parameter binding")
    binding_without_digest = dict(binding)
    binding_sha256 = binding_without_digest.pop("binding_sha256", None)
    if binding_sha256 != _canonical_digest(binding_without_digest):
        raise ValueError("schema-v4 checkpoint optimizer parameter binding is not self-consistent")
    ordered_parameters = binding.get("ordered_parameters")
    parameter_ids = group.get("params")
    if (
        not isinstance(ordered_parameters, Sequence)
        or isinstance(ordered_parameters, (str, bytes))
        or not isinstance(parameter_ids, Sequence)
        or isinstance(parameter_ids, (str, bytes))
        or not ordered_parameters
    ):
        raise ValueError("schema-v4 checkpoint optimizer parameter binding is malformed")
    bound_ids: list[int] = []
    bound_names: list[str] = []
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("schema-v4 checkpoint has no model state for optimizer binding")
    for record in ordered_parameters:
        if not isinstance(record, Mapping) or set(record) != {
            "parameter_id",
            "name",
            "shape",
            "dtype",
            "model_tensor_sha256",
        }:
            raise ValueError("schema-v4 checkpoint has a malformed optimizer parameter record")
        parameter_id = record["parameter_id"]
        name = record["name"]
        if isinstance(parameter_id, bool) or not isinstance(parameter_id, int) or not isinstance(name, str):
            raise ValueError("schema-v4 checkpoint optimizer parameter identity is malformed")
        tensor = state_dict.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("schema-v4 checkpoint optimizer binding names a missing model tensor")
        if record["shape"] != list(tensor.shape) or record["dtype"] != str(tensor.dtype):
            raise ValueError("schema-v4 checkpoint optimizer binding shape or dtype changed")
        if record["model_tensor_sha256"] != _array_digest(tensor.detach().cpu().contiguous().numpy()):
            raise ValueError("schema-v4 checkpoint optimizer binding model-tensor digest changed")
        bound_ids.append(parameter_id)
        bound_names.append(name)
    if set(group) != {*expected_group_values, "params"}:
        raise ValueError("schema-v4 checkpoint AdamW parameter group has unexpected fields")
    if list(parameter_ids) != bound_ids or len(bound_ids) != len(set(bound_ids)):
        raise ValueError("schema-v4 checkpoint optimizer parameter order differs from its binding")
    if len(bound_names) != len(set(bound_names)) or bound_names != list(state_dict):
        raise ValueError("schema-v4 checkpoint optimizer binding does not cover the model state exactly")
    if set(state) != set(bound_ids):
        raise ValueError("schema-v4 checkpoint AdamW state keys do not exactly cover bound parameters")
    parameter_by_id = dict(zip(bound_ids, (state_dict[name] for name in bound_names), strict=True))
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, Mapping) or set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("schema-v4 checkpoint has malformed per-parameter AdamW state")
        step_value = parameter_state["step"]
        if isinstance(step_value, torch.Tensor):
            if step_value.numel() != 1:
                raise ValueError("schema-v4 checkpoint AdamW step is not scalar")
            step_value = float(step_value.detach().cpu())
        if step_value != float(completed_updates):
            raise ValueError("schema-v4 checkpoint AdamW step disagrees with training progress")
        parameter_tensor = parameter_by_id[parameter_id]
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter_tensor.shape
                or moment.dtype != parameter_tensor.dtype
            ):
                raise ValueError("schema-v4 checkpoint AdamW moment shape or dtype differs from its parameter")

    rng_state = checkpoint.get("rng_state")
    rng_state_sha256 = checkpoint.get("rng_state_sha256")
    if not isinstance(rng_state, Mapping) or not _is_sha256(rng_state_sha256):
        raise ValueError("schema-v4 checkpoint is missing Torch RNG restart state")
    if _state_tree_digest(rng_state) != rng_state_sha256:
        raise ValueError("schema-v4 checkpoint RNG-state SHA-256 verification failed")
    cpu_rng_state = rng_state.get("torch_cpu_rng_state")
    device_rng_state = rng_state.get("torch_device_rng_state")
    if not isinstance(cpu_rng_state, torch.Tensor) or cpu_rng_state.dtype != torch.uint8 or cpu_rng_state.ndim != 1:
        raise ValueError("schema-v4 checkpoint CPU RNG state is malformed")
    if rng_state.get("device_type") != device_type:
        raise ValueError("schema-v4 checkpoint RNG and runtime device types disagree")
    if rng_state.get("device_type") == "cpu":
        if device_rng_state is not None:
            raise ValueError("schema-v4 CPU checkpoint unexpectedly has a device RNG state")
    elif rng_state.get("device_type") == "cuda":
        if not isinstance(device_rng_state, torch.Tensor) or device_rng_state.dtype != torch.uint8:
            raise ValueError("schema-v4 CUDA checkpoint has a malformed device RNG state")
    else:
        raise ValueError("schema-v4 checkpoint has an unsupported RNG device type")

    diagnostic_series = checkpoint.get("diagnostic_series")
    diagnostic_series_sha256 = checkpoint.get("diagnostic_series_sha256")
    if not isinstance(diagnostic_series, Mapping) or not _is_sha256(diagnostic_series_sha256):
        raise ValueError("schema-v4 checkpoint is missing all-step diagnostic series")
    if _canonical_digest(diagnostic_series) != diagnostic_series_sha256:
        raise ValueError("schema-v4 checkpoint diagnostic-series SHA-256 verification failed")
    gradient_values = diagnostic_series.get("gradient_norm_before_clipping")
    update_values = diagnostic_series.get("parameter_update_norm")
    if (
        not isinstance(gradient_values, Sequence)
        or isinstance(gradient_values, (str, bytes))
        or len(gradient_values) != completed_updates
    ):
        raise ValueError("schema-v4 checkpoint gradient series does not cover every update")
    if milestone_config.track_parameter_update_norm:
        if (
            not isinstance(update_values, Sequence)
            or isinstance(update_values, (str, bytes))
            or len(update_values) != completed_updates
        ):
            raise ValueError("schema-v4 checkpoint parameter-update series does not cover every update")
    elif update_values is not None:
        raise ValueError("schema-v4 checkpoint has unrequested parameter-update diagnostics")
    expected_diagnostics = _diagnostic_record(
        gradient_values,
        update_values if isinstance(update_values, Sequence) else None,
        train_config,
    )
    if _jsonable(metadata.get("optimization_diagnostics")) != _jsonable(expected_diagnostics):
        raise ValueError("schema-v4 checkpoint optimization diagnostics disagree with all-step series")

    restart_state = checkpoint.get("restart_state")
    restart_state_sha256 = checkpoint.get("restart_state_sha256")
    expected_restart = {
        "completed_updates": completed_updates,
        "next_batch_index": completed_updates,
        "sampling_sha256": expected_schedule.record["sampling_sha256"],
        "batch_stream_sha256": expected_schedule.record["batch_stream_sha256"],
        "optimizer_state_sha256": optimizer_state_sha256,
        "rng_state_sha256": rng_state_sha256,
        "diagnostic_series_sha256": diagnostic_series_sha256,
    }
    if not isinstance(restart_state, Mapping) or restart_state != expected_restart:
        raise ValueError("schema-v4 checkpoint restart state is inconsistent")
    if restart_state_sha256 != _canonical_digest(expected_restart):
        raise ValueError("schema-v4 checkpoint restart-state SHA-256 verification failed")

    training_log = metadata.get("training_log")
    if not isinstance(training_log, Sequence) or isinstance(training_log, (str, bytes)) or not training_log:
        raise ValueError("schema-v4 checkpoint has no authenticated training log")
    expected_logged_steps = {
        step
        for step in range(completed_updates)
        if step % train_config.log_every == 0 or step + 1 in milestone_config.milestone_updates
    }
    actual_logged_steps: list[int] = []
    for entry in training_log:
        if not isinstance(entry, Mapping):
            raise ValueError("schema-v4 checkpoint has a malformed training-log entry")
        step = entry.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("schema-v4 checkpoint training log has a non-integer step")
        actual_logged_steps.append(step)
        if entry.get("completed_updates") != step + 1:
            raise ValueError("schema-v4 checkpoint training log has inconsistent completed updates")
        expected_indices = [int(index) for index in expected_schedule.batches[step]]
        if entry.get("sample_indices") != expected_indices:
            raise ValueError("schema-v4 checkpoint training log disagrees with the sampling stream")
        transition_hashes = expected_schedule.record["transition_sha256"]
        expected_hashes = [transition_hashes[index] for index in expected_indices]
        if entry.get("transition_sha256") != expected_hashes:
            raise ValueError("schema-v4 checkpoint training-log hashes disagree with the sampling stream")
    if actual_logged_steps != sorted(expected_logged_steps):
        raise ValueError("schema-v4 checkpoint training log does not cover the exact required steps")


def _verify_checkpoint(
    checkpoint: Mapping[str, object],
    dataset: _PreparedDataset,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    identity = (checkpoint.get("schema_version"), checkpoint.get("contract"))
    supported_identities = (
        (_SCHEMA_VERSION, _CHECKPOINT_CONTRACT),
        (_MILESTONE_SCHEMA_VERSION, _MILESTONE_CHECKPOINT_CONTRACT),
        *_LEGACY_CHECKPOINT_IDENTITIES,
    )
    if identity not in supported_identities:
        raise ValueError("unsupported PR history v3 checkpoint schema")
    metadata = checkpoint.get("metadata")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint is missing metadata or state_dict")
    if (metadata.get("schema_version"), metadata.get("contract")) != identity:
        raise ValueError("checkpoint and metadata schema identities disagree")
    if _canonical_digest(metadata) != checkpoint.get("metadata_sha256"):
        raise ValueError("checkpoint metadata SHA-256 verification failed")
    tensor_state: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint state_dict must map strings to tensors")
        tensor_state[name] = value
    if _state_dict_digest(tensor_state) != checkpoint.get("state_dict_sha256"):
        raise ValueError("checkpoint state_dict SHA-256 verification failed")
    payload_record = {
        "contract": identity[1],
        "state_dict_sha256": checkpoint["state_dict_sha256"],
        "metadata_sha256": checkpoint["metadata_sha256"],
    }
    if identity == (_MILESTONE_SCHEMA_VERSION, _MILESTONE_CHECKPOINT_CONTRACT):
        payload_record.update(
            {
                "optimizer_state_sha256": checkpoint.get("optimizer_state_sha256"),
                "rng_state_sha256": checkpoint.get("rng_state_sha256"),
                "diagnostic_series_sha256": checkpoint.get("diagnostic_series_sha256"),
                "restart_state_sha256": checkpoint.get("restart_state_sha256"),
            }
        )
    expected_payload = _canonical_digest(payload_record)
    if expected_payload != checkpoint.get("checkpoint_payload_sha256"):
        raise ValueError("checkpoint payload SHA-256 verification failed")

    if _jsonable(metadata.get("history_manifest")) != _jsonable(dataset.history.manifest.as_dict()):
        raise ValueError("checkpoint history manifest does not match evaluation history")
    if _jsonable(metadata.get("static_bundle")) != _jsonable(dataset.static.as_dict()):
        raise ValueError("checkpoint static bundle does not match evaluation history")
    if identity in (
        (_SCHEMA_VERSION, _CHECKPOINT_CONTRACT),
        (_MILESTONE_SCHEMA_VERSION, _MILESTONE_CHECKPOINT_CONTRACT),
    ):
        _verify_schema_v3_training_contract(metadata, dataset)
    if identity == (_MILESTONE_SCHEMA_VERSION, _MILESTONE_CHECKPOINT_CONTRACT):
        _verify_schema_v4_milestone_contract(checkpoint, metadata)
    predictor_config = metadata.get("predictor_config")
    if not isinstance(predictor_config, Mapping):
        raise ValueError("checkpoint predictor_config is missing")
    if checkpoint.get("predictor_config") != predictor_config:
        raise ValueError("checkpoint predictor config copies disagree")
    predictor_config_copy = checkpoint_predictor_config(
        {
            "predictor_config": dict(predictor_config),
            "state_dict": tensor_state,
        }
    )
    graph = predictor_config_copy.get("graph_transformer")
    if predictor_config_copy.get("kind") != "graph-transformer" or not isinstance(graph, Mapping):
        raise ValueError("checkpoint is not a graph-transformer checkpoint")
    if graph.get("architecture_version") not in (3, 4):
        raise ValueError("checkpoint is not architecture version 3 or 4")
    if graph.get("dt") != dataset.history.manifest.dt_seconds:
        raise ValueError("checkpoint graph timestep does not exactly match evaluation transitions")
    if checkpoint.get("decoder_work") != metadata.get("decoder_work"):
        raise ValueError("checkpoint decoder work copies disagree")
    if metadata.get("decoder_work") != _decoder_work(_SUPPORTED_PROJECTION_BACKEND):
        raise ValueError("checkpoint decoder work is not the supported dense one-shot projection")
    if checkpoint.get("training_realized_hierarchy_levels") != metadata.get("training_realized_hierarchy_levels"):
        raise ValueError("checkpoint realized hierarchy depth copies disagree")
    return predictor_config_copy, tensor_state


def load_pr_history_v3_checkpoint(
    checkpoint_or_path: Mapping[str, object] | pathlib.Path | str,
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    *,
    device: torch.device | str = "cpu",
) -> tuple[StretchPredictor, _PreparedDataset, dict[str, object]]:
    """Verify and load a v3/v4 checkpoint against an exact history dataset."""
    dataset = _prepare_dataset(history, transitions)
    device = torch.device(device)
    if isinstance(checkpoint_or_path, Mapping):
        checkpoint = dict(checkpoint_or_path)
    else:
        checkpoint = torch.load(checkpoint_or_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint file did not contain a dictionary")
    predictor_config, state_dict = _verify_checkpoint(checkpoint, dataset)
    graph_config = GraphTransformerConfig(**dict(predictor_config["graph_transformer"]))
    _validate_graph_config(graph_config, history.manifest.dt_seconds)
    predictor, _solvers = _build_predictor_and_solvers(dataset, graph_config, device)
    load_stretch_predictor_state(predictor, {"state_dict": state_dict})
    predictor.eval()
    realized = int(predictor.model.n_levels)
    if realized != checkpoint["training_realized_hierarchy_levels"]:
        raise ValueError(
            "evaluation hierarchy realizes a different depth from training; "
            f"evaluation={realized}, training={checkpoint['training_realized_hierarchy_levels']}"
        )
    return predictor, dataset, checkpoint


def _geometric_mean_nonnegative(values: Sequence[float]) -> float:
    """Return the geometric mean, defining any zero-valued set as zero."""
    copied = [float(value) for value in values]
    if not copied:
        raise ValueError("geometric mean requires at least one value")
    if any(not math.isfinite(value) or value < 0.0 for value in copied):
        raise ValueError("geometric mean requires finite non-negative values")
    if any(value == 0.0 for value in copied):
        return 0.0
    result = math.exp(statistics.fmean(math.log(value) for value in copied))
    if not math.isfinite(result):
        raise ValueError("geometric mean is not finite")
    return result


def evaluate_pr_history_v4_raw_f(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    checkpoint_or_path: Mapping[str, object] | pathlib.Path | str,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Evaluate per-transition normalized raw-F loss without projection.

    The report uses the exact training numerator and transition-local
    normalizer. It deliberately never calls the global position projection,
    so it diagnoses the learned full-gradient target independently of decoded
    common-objective quality.
    """
    source_provenance_start = _source_provenance()
    if source_provenance_start != _SOURCE_PROVENANCE_AT_IMPORT:
        raise RuntimeError(
            "Newton source changed after this trainer module was imported; "
            "restart from one settled source tree before raw-F evaluation"
        )
    device = torch.device(device)
    predictor, dataset, checkpoint = load_pr_history_v3_checkpoint(
        checkpoint_or_path,
        history,
        transitions,
        device=device,
    )
    if (
        checkpoint.get("schema_version"),
        checkpoint.get("contract"),
    ) != (_MILESTONE_SCHEMA_VERSION, _MILESTONE_CHECKPOINT_CONTRACT):
        raise ValueError("raw-F evaluation requires a schema-v4 milestone checkpoint")
    checkpoint_source = checkpoint["metadata"].get("source_provenance")
    if checkpoint_source != source_provenance_start:
        raise ValueError("raw-F evaluation source does not exactly match checkpoint training source")
    training_config_value = checkpoint["metadata"].get("training_config")
    if not isinstance(training_config_value, Mapping):
        raise ValueError("raw-F checkpoint has no training configuration")
    train_config = PRV3TrainingConfig(**dict(training_config_value))
    predictor_config = checkpoint["predictor_config"]
    graph_config_value = predictor_config.get("graph_transformer")
    if not isinstance(graph_config_value, Mapping):
        raise ValueError("raw-F checkpoint has no graph-transformer configuration")
    graph_config = GraphTransformerConfig(**dict(graph_config_value))
    if graph_config.architecture_version != 4 or train_config.loss_mode != _NORMALIZED_RAW_F_LOSS_MODE:
        raise ValueError("raw-F evaluation requires an architecture-v4 raw-F-trained checkpoint")

    solvers = _build_solvers(dataset, device)
    supervision = _prepare_raw_target_supervision(
        dataset,
        device,
        train_config.raw_deformation_gradient_floor,
    )
    sample_records: list[dict[str, object]] = []
    with torch.no_grad():
        predictions, target_fields = _grouped_prediction_with_targets(
            predictor,
            solvers,
            dataset.static,
            dataset.samples,
            device,
            decode=False,
        )
        if predictions is not None:
            raise RuntimeError("raw-F evaluation unexpectedly decoded positions")
        for target_F, sample in zip(target_fields, dataset.samples, strict=True):
            normalized, raw_loss, normalizer = _normalized_raw_deformation_gradient_loss(
                target_F,
                solvers[sample.pin_signature],
                supervision[sample.transition.transition_sha256],
            )
            target_array = np.ascontiguousarray(target_F.to(dtype=torch.float64).detach().cpu().numpy())
            normalized_value = float(normalized.detach())
            raw_value = float(raw_loss.detach())
            normalizer_value = float(normalizer.detach())
            values = (normalized_value, raw_value, normalizer_value)
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise RuntimeError("raw-F evaluation produced an invalid loss")
            sample_records.append(
                {
                    "transition_sha256": sample.transition.transition_sha256,
                    "coordinate": sample.transition.coordinate.as_dict(),
                    "scene_sha256": sample.transition.scene_sha256,
                    "objective_instance_sha256": sample.transition.objective_instance_sha256,
                    "predicted_F_sha256": _array_digest(target_array),
                    "observed_F_sha256": _array_digest(sample.observed_F),
                    "reference_F_sha256": _array_digest(sample.reference_F),
                    "raw_target_deformation_gradient_loss": raw_value,
                    "raw_deformation_gradient_observed_loss": sample.raw_deformation_gradient_observed_loss,
                    "raw_deformation_gradient_normalizer": normalizer_value,
                    "normalized_raw_deformation_gradient_loss": normalized_value,
                    "normalizer_floor_active": (
                        sample.raw_deformation_gradient_observed_loss < train_config.raw_deformation_gradient_floor
                    ),
                }
            )

    normalized_values = [float(record["normalized_raw_deformation_gradient_loss"]) for record in sample_records]
    deterministic: dict[str, object] = {
        "schema_version": 1,
        "contract": _RAW_F_EVALUATION_CONTRACT,
        "checkpoint_payload_sha256": checkpoint["checkpoint_payload_sha256"],
        "checkpoint_identity": {
            "schema_version": checkpoint["schema_version"],
            "contract": checkpoint["contract"],
        },
        "checkpoint_training_selection_sha256": checkpoint["metadata"]["transition_selection"].get("selection_sha256"),
        "history_manifest_sha256": history.manifest.manifest_sha256,
        "static_sha256": dataset.static.static_sha256,
        "evaluation_selection": dataset.selection_record,
        "training_evaluation_selection_match": (
            checkpoint["metadata"]["transition_selection"].get("selection_sha256")
            == dataset.selection_record["selection_sha256"]
        ),
        "loss_contract": _loss_contract(train_config, dataset.characteristic_length_m),
        "projection_calls": 0,
        "source_provenance": source_provenance_start,
        "samples": sample_records,
        "summary": {
            "sample_count": len(sample_records),
            "mean_normalized_raw_deformation_gradient_loss": statistics.fmean(normalized_values),
            "geometric_mean_normalized_raw_deformation_gradient_loss": _geometric_mean_nonnegative(normalized_values),
            "maximum_normalized_raw_deformation_gradient_loss": max(normalized_values),
        },
    }
    if _source_provenance() != source_provenance_start:
        raise RuntimeError("Newton source changed during raw-F evaluation")
    deterministic["evaluation_sha256"] = _canonical_digest(deterministic)
    return deterministic


def evaluate_pr_history_v3(
    history: PRSceneHistory,
    transitions: Sequence[HistoryTransition],
    checkpoint_or_path: Mapping[str, object] | pathlib.Path | str,
    *,
    device: torch.device | str = "cpu",
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, object]:
    """Evaluate v3/v4 positions with the independent common-objective scorer.

    The requested transitions may be held-out samples, but they must share the
    exact authenticated history manifest, static mesh, material, and timestep
    with the checkpoint. Primary inference timings use device-resident inputs
    and cover one predictor pass plus one projection. Adapter end-to-end
    timings retain host-to-device tensor preparation as a separate series.
    Common-objective scoring is never included in either interval.
    """
    if isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    device = torch.device(device)
    predictor, dataset, checkpoint = load_pr_history_v3_checkpoint(
        checkpoint_or_path,
        history,
        transitions,
        device=device,
    )
    solvers = _build_solvers(dataset, device)
    # CUDA graph aggregation uses atomic index additions.  On the audited PR
    # mesh their ordering changes positions at roughly 1e-10 m, so retain a
    # strict one-nanometre ceiling while CPU inference remains bit-exact.
    repeat_tolerance_m = 0.0 if device.type == "cpu" else 1.0e-9

    sample_records: list[dict[str, object]] = []
    timing_records: list[dict[str, object]] = []
    with torch.no_grad():
        for sample in dataset.samples:
            _synchronize(device)
            preparation_start = time.perf_counter()
            resident_inputs = _prepare_resident_prediction_input(solvers, dataset.static, sample, device)
            _synchronize(device)
            preparation_seconds = time.perf_counter() - preparation_start
            for _ in range(warmup):
                _resident_prediction(predictor, resident_inputs)
            _synchronize(device)

            durations: list[float] = []
            adapter_call_durations: list[float] = []
            repeat_positions: list[np.ndarray] = []
            adapter_positions: list[np.ndarray] = []
            for _ in range(repeats):
                _synchronize(device)
                start = time.perf_counter()
                candidate = _resident_prediction(predictor, resident_inputs)
                _synchronize(device)
                durations.append(time.perf_counter() - start)
                # Device-to-host transfer is deliberately outside the solver
                # interval.  It provides an independent repeat comparison and
                # the representative array consumed by the CPU evaluator.
                repeat_positions.append(candidate.detach().cpu().numpy())
            for _ in range(repeats):
                _synchronize(device)
                adapter_start = time.perf_counter()
                adapter_candidate = _grouped_prediction(predictor, solvers, dataset.static, [sample], device)[0]
                _synchronize(device)
                adapter_call_durations.append(time.perf_counter() - adapter_start)
                adapter_positions.append(adapter_candidate.detach().cpu().numpy())
            if not repeat_positions or not adapter_positions:
                raise RuntimeError("evaluation produced no prediction")
            prediction = repeat_positions[0]
            repeat_max_discrepancy_m = max(
                float(np.max(np.abs(candidate - prediction))) for candidate in repeat_positions
            )
            if repeat_max_discrepancy_m > repeat_tolerance_m:
                raise RuntimeError(
                    "repeat inference outputs exceeded the device-aware discrepancy tolerance: "
                    f"observed={repeat_max_discrepancy_m:.3e} m, tolerance={repeat_tolerance_m:.3e} m"
                )
            adapter_resident_max_discrepancy_m = max(
                float(np.max(np.abs(candidate - prediction))) for candidate in adapter_positions
            )
            if adapter_resident_max_discrepancy_m > repeat_tolerance_m:
                raise RuntimeError(
                    "adapter and resident inference outputs exceeded the device-aware discrepancy tolerance: "
                    f"observed={adapter_resident_max_discrepancy_m:.3e} m, "
                    f"tolerance={repeat_tolerance_m:.3e} m"
                )

            # The common evaluator intentionally differentiates its scalar
            # objective even though network inference above is no-grad.
            with torch.enable_grad():
                scene = history.build_atomic_scene(sample.transition.input_state, sample.transition.applied_state)
                problem = build_common_problem(scene)
                objective = common_objective_manifest(scene, problem)
                if objective["objective_instance_sha256"] != sample.transition.objective_instance_sha256:
                    raise ValueError("evaluation transition objective hash changed after loading")
                metric_start = time.perf_counter()
                metrics = evaluate_common_state(
                    problem,
                    prediction,
                    reference_positions=sample.reference_positions,
                )
                metric_seconds = time.perf_counter() - metric_start
            sample_records.append(
                {
                    "transition_sha256": sample.transition.transition_sha256,
                    "coordinate": sample.transition.coordinate.as_dict(),
                    "scene_sha256": sample.transition.scene_sha256,
                    "objective_instance_sha256": sample.transition.objective_instance_sha256,
                    "reference_position_sha256": _array_digest(sample.reference_positions),
                    "metrics": metrics.as_dict(),
                    "decoder_work": checkpoint["decoder_work"],
                }
            )
            timing_records.append(
                {
                    "transition_sha256": sample.transition.transition_sha256,
                    "warmup_runs": warmup,
                    "repeat_seconds": durations,
                    "median_inference_seconds": statistics.median(durations),
                    "minimum_inference_seconds": min(durations),
                    "input_preparation_seconds": preparation_seconds,
                    "adapter_call_repeat_seconds": adapter_call_durations,
                    "median_adapter_call_seconds": statistics.median(adapter_call_durations),
                    "minimum_adapter_call_seconds": min(adapter_call_durations),
                    "adapter_resident_max_discrepancy_m": adapter_resident_max_discrepancy_m,
                    "repeat_max_discrepancy_m": repeat_max_discrepancy_m,
                    "repeat_discrepancy_tolerance_m": repeat_tolerance_m,
                    "common_evaluator_seconds": metric_seconds,
                    "timing_temperature": "warmed" if warmup > 0 else "unwarmed first resident repeat",
                    "inference_scope": (
                        "device-resident predictor pass plus one dense global projection; "
                        + ("explicit warmup completed" if warmup > 0 else "no explicit warmup requested")
                    ),
                    "adapter_call_scope": (
                        "per-call NumPy stacking, tensor upload, predictor pass, and one dense global projection; "
                        "excludes model/factor setup, common-objective scoring, and device-to-host result transfer"
                    ),
                    "adapter_timing_temperature": "runs after the resident timing loop has exercised the model",
                }
            )

    free_errors = [float(record["metrics"]["free_rms_error_m"]) for record in sample_records]
    residuals = [float(record["metrics"]["relative_residual"]) for record in sample_records]
    deterministic: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "contract": _EVALUATION_CONTRACT,
        "checkpoint_payload_sha256": checkpoint["checkpoint_payload_sha256"],
        "checkpoint_identity": {
            "schema_version": checkpoint["schema_version"],
            "contract": checkpoint["contract"],
        },
        "checkpoint_training_selection_sha256": checkpoint["metadata"]["transition_selection"].get("selection_sha256"),
        "history_manifest_sha256": history.manifest.manifest_sha256,
        "static_sha256": dataset.static.static_sha256,
        "evaluation_selection": dataset.selection_record,
        "training_evaluation_selection_match": (
            checkpoint["metadata"]["transition_selection"].get("selection_sha256")
            == dataset.selection_record["selection_sha256"]
        ),
        "predictor_config": checkpoint["predictor_config"],
        "training_realized_hierarchy_levels": checkpoint["training_realized_hierarchy_levels"],
        "decoder_work": checkpoint["decoder_work"],
        "samples": sample_records,
        "summary": {
            "sample_count": len(sample_records),
            "mean_free_rms_error_m": statistics.fmean(free_errors),
            "maximum_free_rms_error_m": max(free_errors),
            "mean_relative_residual": statistics.fmean(residuals),
            "maximum_relative_residual": max(residuals),
        },
    }
    deterministic["evaluation_sha256"] = _canonical_digest(deterministic)
    timing = {
        "samples": timing_records,
        "timing_scope": "excluded from deterministic evaluation_sha256",
    }
    timing["timing_sha256"] = _canonical_digest(timing)
    return {"deterministic": deterministic, "timing": timing}
