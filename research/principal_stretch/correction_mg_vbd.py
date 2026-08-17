# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU quality integration for multiplicative MG-VBD correction.

This module joins three already independent research components without
weakening their contracts:

* fresh scalar-CPU VBD K1 provides the nonlinear initializer;
* one static rest-tangent multigrid hierarchy preconditions every Krylov solve;
* each accepted outer correction rebuilds the current matrix-free
  stable-Neo-Hookean Gauss--Newton operator.

Fresh VBD K4 is a comparator, never a continuation of K1.  An accepted dense
reference may be supplied by an authenticated history transition; standalone
scenes instead run and validate a fresh dense Newton reference.  All states
are scored independently with the common float64 evaluator.  Timings are
returned in a separate diagnostic-only record and are not performance
evidence for the eventual captured GPU solver.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import time
import types
from collections.abc import Mapping

import numpy as np
import torch

from .correction_gpu import (
    MatrixFreeCorrectionConfig,
    MatrixFreeCorrectionResult,
    MatrixFreeStableNHOperator,
    solve_matrix_free_correction,
)
from .correction_multigrid import (
    StaticMultigridHierarchy,
    VCycleWorkRecord,
    apply_v_cycle,
    build_stable_nh_rest_multigrid,
)
from .solver_benchmark import (
    CommonStateMetrics,
    NewtonRunResult,
    TetBenchmarkScene,
    VBDRunResult,
    _vbd_run_digest,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
    run_newton,
    run_vbd,
)

_QUALITY_CONTRACT = "pss-multiplicative-mg-vbd-cpu-quality-v1"
_TIMING_CONTRACT = "pss-multiplicative-mg-vbd-cpu-diagnostic-timing-v1"
_OBJECTIVE_ROUNDOFF_FACTOR = 8.0


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return types.MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_copy(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    try:
        copied = json.loads(json.dumps(_thaw_json(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error
    return _freeze_json(copied)  # type: ignore[return-value]


def _array_digest(value: np.ndarray) -> str:
    """Hash array dtype, shape, and canonical little-endian contents."""
    array = np.asarray(value)
    dtype = array.dtype if array.dtype.byteorder == "|" else array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(json.dumps(canonical.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _readonly_positions(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    owned = np.array(value, dtype=np.float64, order="C", copy=True)
    if owned.ndim != 2 or owned.shape[1] != 3 or not np.isfinite(owned).all():
        raise ValueError(f"{name} must be a finite array with shape (V, 3)")
    return np.frombuffer(owned.tobytes(order="C"), dtype=np.float64).reshape(owned.shape)


def _readonly_indices(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"{name} must contain integers")
        value = value.detach().cpu().numpy()
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    owned = np.array(source, dtype=np.int64, order="C", copy=True)
    return np.frombuffer(owned.tobytes(order="C"), dtype=np.int64)


def _validate_sha256(value: str, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_finite_metrics(metrics: CommonStateMetrics, *, require_reference_errors: bool) -> None:
    if type(metrics) is not CommonStateMetrics:
        raise TypeError("metrics must be an exact CommonStateMetrics")
    required = (
        metrics.objective,
        metrics.inertia,
        metrics.elastic,
        metrics.gradient_norm,
        metrics.relative_residual,
        metrics.determinant_min,
        metrics.determinant_max,
        metrics.inverted_tet_fraction,
        metrics.minimum_singular_value,
        metrics.max_pin_error_m,
    )
    if any(not math.isfinite(value) for value in required):
        raise ValueError("common metrics must be finite")
    if require_reference_errors:
        errors = (metrics.free_rms_error_m, metrics.mass_weighted_rms_error_m)
        if any(value is None or not math.isfinite(value) or value < 0.0 for value in errors):
            raise ValueError("reference-scored metrics require finite non-negative RMS errors")
    if (
        metrics.gradient_norm < 0.0
        or metrics.relative_residual < 0.0
        or metrics.inverted_tet_fraction < 0.0
        or metrics.inverted_tet_fraction > 1.0
        or metrics.minimum_singular_value < 0.0
        or metrics.max_pin_error_m < 0.0
    ):
        raise ValueError("common metrics contain an invalid non-negative quantity")
    _validate_sha256(metrics.position_sha256, "metric position_sha256")


def _same_float64_measurement(left: float, right: float) -> bool:
    guard = 128.0 * np.finfo(np.float64).eps * max(1.0, abs(left), abs(right))
    return abs(left - right) <= guard


@dataclasses.dataclass(frozen=True)
class MGVBDCorrectionConfig:
    """Fixed multiplicative outer work and static hierarchy settings."""

    outer_corrections: int = 3
    correction: MatrixFreeCorrectionConfig = dataclasses.field(
        default_factory=lambda: MatrixFreeCorrectionConfig(pcg_iterations=4)
    )
    mode_kind: str = "rigid"
    target_aggregate_size: int = 4
    minimum_aggregate_size: int = 3
    coarse_node_limit: int = 4
    maximum_levels: int = 8
    pre_smooth_steps: int = 1
    post_smooth_steps: int = 1
    smoother_safety: float = 0.9

    def validate(self) -> None:
        """Validate the fixed numerical schedule."""
        integer_fields = (
            "outer_corrections",
            "target_aggregate_size",
            "minimum_aggregate_size",
            "coarse_node_limit",
            "maximum_levels",
            "pre_smooth_steps",
            "post_smooth_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be an integer")
        if self.outer_corrections < 1:
            raise ValueError("outer_corrections must be positive")
        if type(self.correction) is not MatrixFreeCorrectionConfig:
            raise TypeError("correction must be an exact MatrixFreeCorrectionConfig")
        self.correction.validate()
        if self.mode_kind not in ("rigid", "translation"):
            raise ValueError("mode_kind must be 'rigid' or 'translation'")
        if self.target_aggregate_size < 2:
            raise ValueError("target_aggregate_size must be at least two")
        if not 2 <= self.minimum_aggregate_size <= self.target_aggregate_size:
            raise ValueError("minimum_aggregate_size must lie in [2, target_aggregate_size]")
        if self.mode_kind == "rigid" and self.minimum_aggregate_size < 3:
            raise ValueError("rigid enrichment requires minimum_aggregate_size >= 3")
        if self.coarse_node_limit < 1 or self.maximum_levels < 2:
            raise ValueError("coarse_node_limit must be positive and maximum_levels at least two")
        if self.pre_smooth_steps < 1 or self.post_smooth_steps != self.pre_smooth_steps:
            raise ValueError("symmetric V-cycles require equal positive smoothing counts")
        if not math.isfinite(self.smoother_safety) or not 0.0 < self.smoother_safety < 1.0:
            raise ValueError("smoother_safety must lie in (0, 1)")

    def deterministic_record(self) -> dict[str, object]:
        """Serialize the numerical policy without diagnostic settings."""
        self.validate()
        return {
            "outer_corrections": int(self.outer_corrections),
            "correction": self.correction.deterministic_record(),
            "mode_kind": self.mode_kind,
            "target_aggregate_size": int(self.target_aggregate_size),
            "minimum_aggregate_size": int(self.minimum_aggregate_size),
            "coarse_node_limit": int(self.coarse_node_limit),
            "maximum_levels": int(self.maximum_levels),
            "pre_smooth_steps": int(self.pre_smooth_steps),
            "post_smooth_steps": int(self.post_smooth_steps),
            "smoother_safety": float(self.smoother_safety),
            "outer_policy": "relinearize-current-A-after-each-accepted-correction",
            "hierarchy_policy": "reuse-one-static-rest-A0-hierarchy",
            "failure_policy": "stop-at-first-rejection-and-retain-last-accepted-state",
        }


@dataclasses.dataclass(frozen=True)
class VBDStateEvidence:
    """Timing-free identity for one independently restarted VBD state."""

    role: str
    iterations: int
    scene_sha256: str
    objective_instance_sha256: str
    physical_state_sha256: str
    iterate_zero_sha256: str
    position_sha256: str
    velocity_sha256: str
    color_group_count: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.role not in ("vbd-k1", "vbd-k4"):
            raise ValueError("VBD evidence has an unknown role")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, numbers.Integral):
            raise ValueError("VBD iterations must be a positive integer")
        if self.iterations < 1 or self.color_group_count < 1:
            raise ValueError("VBD iteration and color counts must be positive")
        for name in (
            "scene_sha256",
            "objective_instance_sha256",
            "physical_state_sha256",
            "iterate_zero_sha256",
            "position_sha256",
            "velocity_sha256",
            "evidence_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        payload = self.deterministic_record()
        payload.pop("evidence_sha256")
        if _canonical_digest(payload) != self.evidence_sha256:
            raise ValueError("VBD evidence does not match its SHA-256")

    def deterministic_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "execution": "fresh-scalar-cpu-run_vbd",
            "iterations": self.iterations,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "physical_state_sha256": self.physical_state_sha256,
            "iterate_zero_sha256": self.iterate_zero_sha256,
            "position_sha256": self.position_sha256,
            "velocity_sha256": self.velocity_sha256,
            "requested_tile_solve": False,
            "effective_tile_solve": False,
            "device": "cpu",
            "color_group_count": self.color_group_count,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclasses.dataclass(frozen=True)
class ReferenceEvidence:
    """Accepted reference identity, independent of timing data."""

    provenance: str
    scene_sha256: str
    objective_instance_sha256: str
    position_sha256: str
    source_record_sha256: str
    source_record: Mapping[str, object] = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.provenance) is not str or not self.provenance:
            raise ValueError("reference provenance must be a non-empty exact string")
        for name in ("scene_sha256", "objective_instance_sha256", "position_sha256", "source_record_sha256"):
            _validate_sha256(getattr(self, name), name)
        record = _canonical_json_copy(self.source_record, "reference source record")
        if _canonical_digest(_thaw_json(record)) != self.source_record_sha256:
            raise ValueError("reference source record does not match its SHA-256")
        required = {
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "position_sha256": self.position_sha256,
            "accepted": True,
        }
        for name, expected in required.items():
            if record.get(name) != expected:
                raise ValueError(f"reference source record does not bind {name}")
        allowed_methods = {
            "dense-cpu-newton-float64",
            "dense-cpu-newton-float64-with-strict-residual-polish",
            "dense-cpu-newton-float64-with-alternate-residual-verification",
        }
        if record.get("method") not in allowed_methods:
            raise ValueError("reference source record has an unsupported method")
        if record.get("failures") not in (None, ()):
            raise ValueError("accepted reference source record contains failures")
        object.__setattr__(self, "source_record", record)

    def deterministic_record(self) -> dict[str, object]:
        return {
            "provenance": self.provenance,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "position_sha256": self.position_sha256,
            "source_record_sha256": self.source_record_sha256,
            "source_record": _thaw_json(self.source_record),
        }


@dataclasses.dataclass(frozen=True)
class HierarchyEvidence:
    """Content and storage identity for the one reused rest hierarchy."""

    hierarchy_sha256: str
    solver_contract: str
    mode_kind: str
    target_aggregate_size: int
    minimum_aggregate_size: int
    coarse_node_limit: int
    maximum_levels: int
    pre_smooth_steps: int
    post_smooth_steps: int
    smoother_safety: float
    level_shapes: tuple[tuple[int, int], ...]
    storage_sha256: str
    static_model_sha256: str
    total_bytes: int
    preconditioner_identity: str

    def __post_init__(self) -> None:
        _validate_sha256(self.hierarchy_sha256, "hierarchy_sha256")
        _validate_sha256(self.storage_sha256, "storage_sha256")
        _validate_sha256(self.static_model_sha256, "static_model_sha256")
        if type(self.solver_contract) is not str or not self.solver_contract:
            raise ValueError("hierarchy solver_contract must be non-empty")
        if type(self.preconditioner_identity) is not str or not self.preconditioner_identity:
            raise ValueError("hierarchy preconditioner_identity must be non-empty")
        expected_identity = f"{self.solver_contract}:rest-a0-vcycle:{self.hierarchy_sha256}"
        if self.preconditioner_identity != expected_identity:
            raise ValueError("preconditioner identity does not bind the hierarchy")
        if self.mode_kind not in ("rigid", "translation"):
            raise ValueError("hierarchy mode_kind is invalid")
        if not self.level_shapes or any(rows < 1 or block_size < 1 for rows, block_size in self.level_shapes):
            raise ValueError("hierarchy level_shapes must contain positive dimensions")
        if self.total_bytes < 1:
            raise ValueError("hierarchy storage must be positive")

    def deterministic_record(self) -> dict[str, object]:
        return {
            "hierarchy_sha256": self.hierarchy_sha256,
            "solver_contract": self.solver_contract,
            "mode_kind": self.mode_kind,
            "target_aggregate_size": self.target_aggregate_size,
            "minimum_aggregate_size": self.minimum_aggregate_size,
            "coarse_node_limit": self.coarse_node_limit,
            "maximum_levels": self.maximum_levels,
            "pre_smooth_steps": self.pre_smooth_steps,
            "post_smooth_steps": self.post_smooth_steps,
            "smoother_safety": self.smoother_safety,
            "level_shapes": [list(shape) for shape in self.level_shapes],
            "storage_sha256": self.storage_sha256,
            "static_model_sha256": self.static_model_sha256,
            "total_bytes": self.total_bytes,
            "preconditioner_identity": self.preconditioner_identity,
            "static_reuse": True,
        }


@dataclasses.dataclass(frozen=True)
class MetricRatios:
    """Exact candidate/comparator arithmetic for one common-objective pair."""

    comparator_role: str
    candidate_position_sha256: str
    comparator_position_sha256: str
    objective_magnitude_ratio: float
    objective_delta: float
    objective_roundoff_guard: float
    residual_ratio: float
    free_rms_ratio: float
    mass_weighted_rms_ratio: float

    def __post_init__(self) -> None:
        if type(self.comparator_role) is not str or not self.comparator_role:
            raise ValueError("comparator_role must be a non-empty exact string")
        _validate_sha256(self.candidate_position_sha256, "candidate_position_sha256")
        _validate_sha256(self.comparator_position_sha256, "comparator_position_sha256")
        for field in dataclasses.fields(self):
            if field.name.endswith("ratio") or field.name in ("objective_delta", "objective_roundoff_guard"):
                value = getattr(self, field.name)
                if type(value) is not float or not math.isfinite(value):
                    raise ValueError(f"{field.name} must be a finite exact float")
        if (
            self.objective_magnitude_ratio < 0.0
            or self.objective_roundoff_guard < 0.0
            or self.residual_ratio < 0.0
            or self.free_rms_ratio < 0.0
            or self.mass_weighted_rms_ratio < 0.0
        ):
            raise ValueError("metric ratios and objective guard must be non-negative")

    @property
    def objective_no_worse(self) -> bool:
        return self.objective_delta <= self.objective_roundoff_guard

    @property
    def residual_no_worse(self) -> bool:
        return self.residual_ratio <= 1.0

    @property
    def free_rms_no_worse(self) -> bool:
        return self.free_rms_ratio <= 1.0

    @property
    def mass_weighted_rms_no_worse(self) -> bool:
        return self.mass_weighted_rms_ratio <= 1.0

    def deterministic_record(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "objective_no_worse": self.objective_no_worse,
            "residual_no_worse": self.residual_no_worse,
            "free_rms_no_worse": self.free_rms_no_worse,
            "mass_weighted_rms_no_worse": self.mass_weighted_rms_no_worse,
        }


@dataclasses.dataclass(frozen=True)
class MGOuterCorrectionEvidence:
    """One relinearized correction and every V-cycle it consumed."""

    outer_index: int
    start_position_sha256: str
    end_position_sha256: str
    result: MatrixFreeCorrectionResult
    start_metrics: CommonStateMetrics
    metrics: CommonStateMetrics
    v_cycle_work: tuple[VCycleWorkRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.outer_index, bool) or not isinstance(self.outer_index, numbers.Integral):
            raise ValueError("outer_index must be a non-negative integer")
        if self.outer_index < 0:
            raise ValueError("outer_index must be a non-negative integer")
        _validate_sha256(self.start_position_sha256, "outer start_position_sha256")
        _validate_sha256(self.end_position_sha256, "outer end_position_sha256")
        if type(self.result) is not MatrixFreeCorrectionResult:
            raise TypeError("outer result must be an exact MatrixFreeCorrectionResult")
        _require_finite_metrics(self.start_metrics, require_reference_errors=True)
        _require_finite_metrics(self.metrics, require_reference_errors=True)
        if self.start_position_sha256 != self.start_metrics.position_sha256:
            raise ValueError("outer start metrics do not bind its start state")
        result_sha256 = _array_digest(self.result.x)
        if self.end_position_sha256 != self.metrics.position_sha256 or self.end_position_sha256 != result_sha256:
            raise ValueError("outer endpoint identities disagree")
        work = tuple(self.v_cycle_work)
        if any(type(item) is not VCycleWorkRecord for item in work):
            raise TypeError("outer V-cycle work must contain exact VCycleWorkRecord instances")
        object.__setattr__(self, "v_cycle_work", work)
        expected_applications = 0 if self.result.pcg is None else self.result.pcg.work.preconditioner_applications
        if len(work) != expected_applications:
            raise ValueError("outer V-cycle records do not match PCG preconditioner applications")
        if self.result.used_fallback and self.end_position_sha256 != self.start_position_sha256:
            raise ValueError("fallback outer correction did not preserve its exact start state")
        scalar_pairs = (
            (self.result.initial_objective, self.start_metrics.objective, "initial objective"),
            (self.result.initial_gradient_norm, self.start_metrics.gradient_norm, "initial gradient"),
            (self.result.initial_minimum_determinant, self.start_metrics.determinant_min, "initial determinant"),
            (self.result.final_objective, self.metrics.objective, "final objective"),
            (self.result.final_gradient_norm, self.metrics.gradient_norm, "final gradient"),
            (self.result.final_minimum_determinant, self.metrics.determinant_min, "final determinant"),
        )
        for measured, independent, label in scalar_pairs:
            if not _same_float64_measurement(measured, independent):
                raise ValueError(f"outer correction {label} disagrees with independent common metrics")

    @property
    def exact_work_completed(self) -> bool:
        pcg = self.result.pcg
        return bool(
            self.result.accepted
            and pcg is not None
            and pcg.consumed_exact_iteration_count
            and len(self.v_cycle_work) == pcg.work.preconditioner_applications
        )

    def deterministic_record(self) -> dict[str, object]:
        return {
            "outer_index": self.outer_index,
            "start_position_sha256": self.start_position_sha256,
            "end_position_sha256": self.end_position_sha256,
            "correction": self.result.deterministic_record(),
            "start_metrics": self.start_metrics.as_dict(),
            "metrics": self.metrics.as_dict(),
            "v_cycle_work": [dataclasses.asdict(work) for work in self.v_cycle_work],
            "exact_work_completed": self.exact_work_completed,
        }


@dataclasses.dataclass(frozen=True)
class MGVBDPromotionGate:
    """Promotion gate for the final multiplicative state against fresh K4."""

    versus_k1: MetricRatios
    versus_k4: MetricRatios
    exact_pins: bool
    inversion_free: bool
    all_outer_work_completed: bool
    fallback_used: bool

    def __post_init__(self) -> None:
        if type(self.versus_k1) is not MetricRatios or type(self.versus_k4) is not MetricRatios:
            raise TypeError("promotion comparisons must be exact MetricRatios")
        for name in ("exact_pins", "inversion_free", "all_outer_work_completed", "fallback_used"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact bool")

    @property
    def passed(self) -> bool:
        comparator = self.versus_k4
        return bool(
            self.exact_pins
            and self.inversion_free
            and self.all_outer_work_completed
            and not self.fallback_used
            and comparator.objective_no_worse
            and comparator.residual_no_worse
            and comparator.free_rms_no_worse
            and comparator.mass_weighted_rms_no_worse
        )

    def deterministic_record(self) -> dict[str, object]:
        return {
            "versus_k1": self.versus_k1.deterministic_record(),
            "versus_k4": self.versus_k4.deterministic_record(),
            "exact_pins": self.exact_pins,
            "inversion_free": self.inversion_free,
            "all_outer_work_completed": self.all_outer_work_completed,
            "fallback_used": self.fallback_used,
            "passed": self.passed,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class MGVBDQualityResult:
    """Timing-free positions, metrics, identities, and fixed-work evidence."""

    scene_sha256: str
    objective_instance_sha256: str
    config: MGVBDCorrectionConfig
    reference: ReferenceEvidence
    hierarchy: HierarchyEvidence
    vbd_k1: VBDStateEvidence
    vbd_k4: VBDStateEvidence
    dt: float
    residual_scale: float
    pinned_indices: np.ndarray = dataclasses.field(repr=False, compare=False)
    x_current: np.ndarray = dataclasses.field(repr=False, compare=False)
    reference_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    k1_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    k1_velocities: np.ndarray = dataclasses.field(repr=False, compare=False)
    k4_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    final_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    final_velocities: np.ndarray = dataclasses.field(repr=False, compare=False)
    reference_metrics: CommonStateMetrics
    k1_metrics: CommonStateMetrics
    k4_metrics: CommonStateMetrics
    outer_corrections: tuple[MGOuterCorrectionEvidence, ...]
    final_metrics: CommonStateMetrics
    gate: MGVBDPromotionGate
    quality_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        for name in ("scene_sha256", "objective_instance_sha256"):
            _validate_sha256(getattr(self, name), name)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("quality dt must be finite and positive")
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("quality residual_scale must be finite and positive")
        pinned = _readonly_indices(self.pinned_indices, "pinned_indices")
        object.__setattr__(self, "pinned_indices", pinned)
        arrays = {}
        for name in (
            "x_current",
            "reference_positions",
            "k1_positions",
            "k1_velocities",
            "k4_positions",
            "final_positions",
            "final_velocities",
        ):
            arrays[name] = _readonly_positions(getattr(self, name), name)
            object.__setattr__(self, name, arrays[name])
        shape = arrays["reference_positions"].shape
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("all quality states must have the reference position shape")
        if pinned.size and (pinned.min() < 0 or pinned.max() >= shape[0] or np.unique(pinned).size != pinned.size):
            raise ValueError("pinned_indices must contain unique in-range vertices")
        for metrics in (
            self.reference_metrics,
            self.k1_metrics,
            self.k4_metrics,
            self.final_metrics,
        ):
            _require_finite_metrics(metrics, require_reference_errors=True)
        expected_hashes = {
            "reference": _array_digest(arrays["reference_positions"]),
            "k1": _array_digest(arrays["k1_positions"]),
            "k4": _array_digest(arrays["k4_positions"]),
            "final": _array_digest(arrays["final_positions"]),
        }
        measured_hashes = {
            "reference": self.reference_metrics.position_sha256,
            "k1": self.k1_metrics.position_sha256,
            "k4": self.k4_metrics.position_sha256,
            "final": self.final_metrics.position_sha256,
        }
        if expected_hashes != measured_hashes:
            raise ValueError("quality metrics do not bind the stored raw states")
        if self.reference.position_sha256 != expected_hashes["reference"]:
            raise ValueError("reference evidence does not bind reference_positions")
        if (
            self.reference.scene_sha256 != self.scene_sha256
            or self.reference.objective_instance_sha256 != self.objective_instance_sha256
        ):
            raise ValueError("reference evidence belongs to another scene/objective")
        validated_reference = _supplied_reference_evidence(
            arrays["reference_positions"],
            self.reference.source_record,
            self.reference.source_record_sha256,
            scene_sha256=self.scene_sha256,
            objective_instance_sha256=self.objective_instance_sha256,
            metrics=self.reference_metrics,
            residual_scale=self.residual_scale,
        )
        if validated_reference.source_record_sha256 != self.reference.source_record_sha256:
            raise ValueError("reference evidence failed independent source-record validation")
        if self.vbd_k1.position_sha256 != expected_hashes["k1"]:
            raise ValueError("K1 evidence does not bind k1_positions")
        if self.vbd_k4.position_sha256 != expected_hashes["k4"]:
            raise ValueError("K4 evidence does not bind k4_positions")
        for evidence, label in ((self.vbd_k1, "K1"), (self.vbd_k4, "K4")):
            if evidence.scene_sha256 != self.scene_sha256:
                raise ValueError(f"{label} evidence belongs to another scene")
            if evidence.objective_instance_sha256 != self.objective_instance_sha256:
                raise ValueError(f"{label} evidence belongs to another objective")
        if self.vbd_k1.role != "vbd-k1" or self.vbd_k1.iterations != 1:
            raise ValueError("K1 evidence changed role or iteration budget")
        if self.vbd_k4.role != "vbd-k4" or self.vbd_k4.iterations != 4:
            raise ValueError("K4 evidence changed role or iteration budget")
        if self.vbd_k1.physical_state_sha256 != _array_digest(arrays["x_current"]):
            raise ValueError("K1 evidence does not bind x_current")
        if self.vbd_k1.physical_state_sha256 != self.vbd_k4.physical_state_sha256:
            raise ValueError("K1 and K4 physical-state identities disagree")
        if self.vbd_k1.iterate_zero_sha256 != self.vbd_k4.iterate_zero_sha256:
            raise ValueError("K1 and K4 iterate-zero identities disagree")
        if self.vbd_k1.velocity_sha256 != _array_digest(arrays["k1_velocities"]):
            raise ValueError("K1 evidence does not bind k1_velocities")
        hierarchy_settings = (
            (self.hierarchy.mode_kind, self.config.mode_kind),
            (self.hierarchy.target_aggregate_size, self.config.target_aggregate_size),
            (self.hierarchy.minimum_aggregate_size, self.config.minimum_aggregate_size),
            (self.hierarchy.coarse_node_limit, self.config.coarse_node_limit),
            (self.hierarchy.maximum_levels, self.config.maximum_levels),
            (self.hierarchy.pre_smooth_steps, self.config.pre_smooth_steps),
            (self.hierarchy.post_smooth_steps, self.config.post_smooth_steps),
            (self.hierarchy.smoother_safety, self.config.smoother_safety),
        )
        if any(actual != expected for actual, expected in hierarchy_settings):
            raise ValueError("hierarchy evidence does not bind the configured hierarchy policy")
        outer = tuple(self.outer_corrections)
        object.__setattr__(self, "outer_corrections", outer)
        if not outer or len(outer) > self.config.outer_corrections:
            raise ValueError("outer correction evidence has an invalid length")
        if tuple(item.outer_index for item in outer) != tuple(range(len(outer))):
            raise ValueError("outer correction indices must be contiguous from zero")
        expected_start = expected_hashes["k1"]
        expected_start_metrics = self.k1_metrics
        for index, item in enumerate(outer):
            if item.start_position_sha256 != expected_start:
                raise ValueError(f"outer correction {index} does not start at the previous endpoint")
            if item.start_metrics != expected_start_metrics:
                raise ValueError(f"outer correction {index} does not retain the previous endpoint metrics")
            if item.result.config != self.config.correction:
                raise ValueError(f"outer correction {index} changed the fixed correction config")
            if item.result.preconditioner_identity != self.hierarchy.preconditioner_identity:
                raise ValueError(f"outer correction {index} changed the preconditioner identity")
            if any(work.hierarchy_sha256 != self.hierarchy.hierarchy_sha256 for work in item.v_cycle_work):
                raise ValueError(f"outer correction {index} used another hierarchy")
            if not item.result.accepted and index != len(outer) - 1:
                raise ValueError("a rejected outer correction must terminate the multiplicative sequence")
            expected_start = item.end_position_sha256
            expected_start_metrics = item.metrics
        if outer[-1].end_position_sha256 != expected_hashes["final"]:
            raise ValueError("final state does not match the last outer correction")
        accepted_count = sum(item.result.accepted for item in outer)
        if accepted_count:
            expected_velocity = (arrays["final_positions"] - arrays["x_current"]) / self.dt
            if pinned.size:
                expected_velocity[pinned] = 0.0
            if not np.array_equal(arrays["final_velocities"], expected_velocity):
                raise ValueError("accepted correction velocity does not match the one-shot update policy")
        elif not np.array_equal(arrays["final_velocities"], arrays["k1_velocities"]):
            raise ValueError("zero-accept fallback must preserve exact K1 velocities")
        expected_gate = MGVBDPromotionGate(
            versus_k1=_metric_ratios(self.final_metrics, self.k1_metrics, "fresh-vbd-k1"),
            versus_k4=_metric_ratios(self.final_metrics, self.k4_metrics, "fresh-vbd-k4"),
            exact_pins=self.final_metrics.max_pin_error_m == 0.0,
            inversion_free=bool(
                self.final_metrics.inverted_tet_fraction == 0.0 and self.final_metrics.determinant_min > 0.0
            ),
            all_outer_work_completed=bool(
                len(outer) == self.config.outer_corrections and all(item.exact_work_completed for item in outer)
            ),
            fallback_used=any(item.result.used_fallback for item in outer),
        )
        if self.gate != expected_gate:
            raise ValueError("promotion gate does not match independently recomputed metrics/work")
        object.__setattr__(self, "quality_sha256", _canonical_digest(self._payload()))

    @property
    def accepted_outer_correction_count(self) -> int:
        return sum(item.result.accepted for item in self.outer_corrections)

    @property
    def total_v_cycle_count(self) -> int:
        return sum(len(item.v_cycle_work) for item in self.outer_corrections)

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _QUALITY_CONTRACT,
            "performance_evidence": False,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "config": self.config.deterministic_record(),
            "reference": self.reference.deterministic_record(),
            "hierarchy": self.hierarchy.deterministic_record(),
            "vbd_k1": self.vbd_k1.deterministic_record(),
            "vbd_k4": self.vbd_k4.deterministic_record(),
            "dt_seconds": self.dt,
            "residual_scale_newtons": self.residual_scale,
            "pinned_indices_sha256": _array_digest(self.pinned_indices),
            "x_current_sha256": _array_digest(self.x_current),
            "reference_metrics": self.reference_metrics.as_dict(),
            "k1_metrics": self.k1_metrics.as_dict(),
            "k4_metrics": self.k4_metrics.as_dict(),
            "outer_corrections": [item.deterministic_record() for item in self.outer_corrections],
            "final_metrics": self.final_metrics.as_dict(),
            "final_velocity_sha256": _array_digest(self.final_velocities),
            "velocity_policy": "update-once-from-x_n-after-last-accepted-correction; exact-K1-on-zero-accept",
            "accepted_outer_correction_count": self.accepted_outer_correction_count,
            "total_v_cycle_count": self.total_v_cycle_count,
            "gate": self.gate.deterministic_record(),
        }

    def deterministic_record(self) -> dict[str, object]:
        """Return the content-addressed quality record."""
        payload = self._payload()
        payload["quality_sha256"] = self.quality_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class MGVBDDiagnosticTiming:
    """CPU diagnostic timings kept outside the quality identity."""

    quality_sha256: str
    problem_build_seconds: float
    reference_source_seconds: float
    reference_evaluation_seconds: float
    hierarchy_build_seconds: float
    outer_correction_seconds: tuple[float, ...]
    outer_evaluation_seconds: tuple[float, ...]
    k1_evaluation_seconds: float
    k4_evaluation_seconds: float
    vbd_k1_run_sha256: str
    vbd_k4_run_sha256: str
    vbd_k1_setup_seconds: float
    vbd_k4_setup_seconds: float
    vbd_k1_warmup_seconds: float
    vbd_k4_warmup_seconds: float
    vbd_k1_repeat_seconds: tuple[float, ...]
    vbd_k4_repeat_seconds: tuple[float, ...]
    vbd_k1_transfer_seconds: tuple[float, ...]
    vbd_k4_transfer_seconds: tuple[float, ...]
    newton_run_sha256: str | None
    newton_warmup_seconds: float | None
    newton_repeat_seconds: tuple[float, ...] | None

    def deterministic_record(self) -> dict[str, object]:
        """Serialize non-performance CPU diagnostic timings."""
        payload = {
            "contract": _TIMING_CONTRACT,
            "performance_evidence": False,
            "measurement_provenance": "eager-scalar-cpu-diagnostic-only",
            **dataclasses.asdict(self),
        }
        payload["timing_sha256"] = _canonical_digest(payload)
        return payload


@dataclasses.dataclass(frozen=True)
class MGVBDRunResult:
    """One quality result and its deliberately separate diagnostic timing."""

    quality: MGVBDQualityResult
    timing: MGVBDDiagnosticTiming

    def __post_init__(self) -> None:
        if self.timing.quality_sha256 != self.quality.quality_sha256:
            raise ValueError("diagnostic timing belongs to another quality result")


def _vbd_evidence(run: VBDRunResult, role: str) -> VBDStateEvidence:
    payload = {
        "role": role,
        "execution": "fresh-scalar-cpu-run_vbd",
        "iterations": run.iterations,
        "scene_sha256": run.scene_sha256,
        "objective_instance_sha256": run.objective_instance_sha256,
        "physical_state_sha256": run.physical_state_sha256,
        "iterate_zero_sha256": run.iterate_zero_sha256,
        "position_sha256": _array_digest(run.positions),
        "velocity_sha256": _array_digest(run.velocities),
        "requested_tile_solve": False,
        "effective_tile_solve": False,
        "device": "cpu",
        "color_group_count": run.color_group_count,
    }
    return VBDStateEvidence(
        role=role,
        iterations=run.iterations,
        scene_sha256=run.scene_sha256,
        objective_instance_sha256=run.objective_instance_sha256,
        physical_state_sha256=run.physical_state_sha256,
        iterate_zero_sha256=run.iterate_zero_sha256,
        position_sha256=payload["position_sha256"],
        velocity_sha256=payload["velocity_sha256"],
        color_group_count=run.color_group_count,
        evidence_sha256=_canonical_digest(payload),
    )


def _validate_vbd_run(
    run: VBDRunResult,
    *,
    role: str,
    iterations: int,
    scene_sha256: str,
    objective_instance_sha256: str,
    physical_state_sha256: str,
    iterate_zero_sha256: str,
    repeats: int,
) -> VBDStateEvidence:
    if type(run) is not VBDRunResult:
        raise TypeError(f"{role} must be an exact VBDRunResult")
    expected = {
        "iterations": iterations,
        "scene_sha256": scene_sha256,
        "objective_instance_sha256": objective_instance_sha256,
        "physical_state_sha256": physical_state_sha256,
        "iterate_zero_sha256": iterate_zero_sha256,
    }
    for name, value in expected.items():
        if getattr(run, name) != value:
            raise ValueError(f"fresh {role} run changed {name}")
    if run.requested_tile_solve or run.effective_tile_solve or run.device != "cpu":
        raise ValueError(f"fresh {role} evidence must use scalar CPU VBD")
    if len(run.repeat_seconds) != repeats or len(run.transfer_seconds) != repeats:
        raise ValueError(f"fresh {role} run did not execute the requested repeats")
    if run.result_state_sha256 != _array_digest(run.positions):
        raise ValueError(f"fresh {role} result-state identity is invalid")
    if run.run_sha256 != _vbd_run_digest(run):
        raise ValueError(f"fresh {role} run digest is invalid")
    return _vbd_evidence(run, role)


def _newton_reference_evidence(run: NewtonRunResult, positions: np.ndarray) -> ReferenceEvidence:
    if not run.reference_accepted:
        raise ValueError(f"fresh dense Newton candidate failed reference gates: {run.reference_failures}")
    record = {
        "contract": "fresh-dense-newton-accepted-reference-v1",
        "method": "dense-cpu-newton-float64",
        "config": dataclasses.asdict(run.config),
        "scene_sha256": run.scene_sha256,
        "objective_instance_sha256": run.objective_instance_sha256,
        "accepted": run.reference_accepted,
        "failures": list(run.reference_failures),
        "native_converged": run.result.converged,
        "native_reason": run.result.reason,
        "accepted_iterations": run.result.accepted_iterations,
        "final_objective": run.result.final_objective,
        "final_gradient_norm": run.result.final_gradient_norm,
        "final_relative_residual": run.result.final_relative_residual,
        "verification_converged": run.verification_converged,
        "verification_reason": run.verification_reason,
        "verification_displacement_relative": run.verification_displacement_relative,
        "alternate_start_converged": run.alternate_start_converged,
        "alternate_start_reason": run.alternate_start_reason,
        "alternate_start_displacement_relative": run.alternate_start_displacement_relative,
        "position_sha256": _array_digest(positions),
    }
    return ReferenceEvidence(
        provenance="fresh-dense-newton",
        scene_sha256=run.scene_sha256,
        objective_instance_sha256=run.objective_instance_sha256,
        position_sha256=record["position_sha256"],
        source_record_sha256=_canonical_digest(record),
        source_record=record,
    )


def _supplied_reference_evidence(
    positions: np.ndarray,
    source_record: Mapping[str, object],
    expected_source_record_sha256: str,
    *,
    scene_sha256: str,
    objective_instance_sha256: str,
    metrics: CommonStateMetrics,
    residual_scale: float,
) -> ReferenceEvidence:
    _validate_sha256(expected_source_record_sha256, "expected_reference_record_sha256")
    record = _canonical_json_copy(source_record, "supplied reference record")
    thawed = _thaw_json(record)
    if _canonical_digest(thawed) != expected_source_record_sha256:
        raise ValueError("supplied reference record does not match its expected SHA-256")
    required = {
        "accepted": True,
        "scene_sha256": scene_sha256,
        "objective_instance_sha256": objective_instance_sha256,
        "position_sha256": _array_digest(positions),
    }
    for name, expected in required.items():
        if record.get(name) != expected:
            raise ValueError(f"supplied reference record changed {name}")
    method = record.get("method")
    allowed_methods = {
        "dense-cpu-newton-float64",
        "dense-cpu-newton-float64-with-strict-residual-polish",
        "dense-cpu-newton-float64-with-alternate-residual-verification",
    }
    if method not in allowed_methods:
        raise ValueError("supplied reference record has an unsupported method")
    method_policy = {
        "dense-cpu-newton-float64-with-strict-residual-polish": (
            "residual_polish_policy",
            "strict-reference-residual-newton-three-start-v1",
        ),
        "dense-cpu-newton-float64-with-alternate-residual-verification": (
            "alternate_residual_policy",
            "alternate-start-only-residual-verification-v1",
        ),
    }.get(method)
    if method_policy is not None and record.get(method_policy[0]) != method_policy[1]:
        raise ValueError(f"supplied reference record changed {method_policy[0]}")
    failures = record.get("failures")
    if failures not in (None, ()):
        raise ValueError("supplied accepted reference record must not contain failures")
    for name, expected in (
        ("final_objective", metrics.objective),
        ("final_gradient_norm", metrics.gradient_norm),
        ("final_relative_residual", metrics.relative_residual),
    ):
        value = record.get(name)
        if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"supplied reference record has an invalid {name}")
        if not _same_float64_measurement(float(value), expected):
            raise ValueError(f"supplied reference record changed {name}")
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("supplied reference record is missing its Newton config")
    expected_config = {
        "max_iterations": 50,
        "gradient_absolute_tolerance": 1.0e-10,
        "gradient_relative_tolerance": 1.0e-10,
        "armijo": 1.0e-4,
        "backtrack": 0.5,
        "max_line_search_steps": 30,
        "minimum_eigenvalue_relative": 1.0e-9,
        "regularization_growth": 10.0,
        "max_regularization_attempts": 12,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise ValueError(f"supplied reference record changed Newton config {name}")
    if config.get("step_relative_tolerance") not in (0.0, 1.0e-14):
        raise ValueError("supplied reference record changed Newton step tolerance")
    residual_limit = max(1.0e-10, 1.0e-10 * residual_scale)
    if metrics.gradient_norm > residual_limit:
        raise ValueError("supplied reference exceeds the independent gradient gate")
    if record.get("verification_converged") is not True or record.get("verification_reason") != "gradient":
        raise ValueError("supplied reference lacks converged independent verification")
    for name, limit in (
        ("verification_displacement_relative", 1.0e-12),
        ("alternate_start_displacement_relative", 1.0e-9),
    ):
        value = record.get(name)
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= limit
        ):
            raise ValueError(f"supplied reference exceeds its {name} gate")
    return ReferenceEvidence(
        provenance=f"externally-pinned-supplied-reference:{method}",
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        position_sha256=required["position_sha256"],
        source_record_sha256=expected_source_record_sha256,
        source_record=record,
    )


def _hierarchy_evidence(
    hierarchy: StaticMultigridHierarchy,
    identity: str,
) -> HierarchyEvidence:
    return HierarchyEvidence(
        hierarchy_sha256=hierarchy.content_sha256,
        solver_contract=hierarchy.solver_contract,
        mode_kind=hierarchy.mode_kind,
        target_aggregate_size=hierarchy.target_aggregate_size,
        minimum_aggregate_size=hierarchy.minimum_aggregate_size,
        coarse_node_limit=hierarchy.coarse_node_limit,
        maximum_levels=hierarchy.maximum_levels,
        pre_smooth_steps=hierarchy.pre_smooth_steps,
        post_smooth_steps=hierarchy.post_smooth_steps,
        smoother_safety=hierarchy.smoother_safety,
        level_shapes=tuple((level.matrix.block_row_count, level.matrix.block_size) for level in hierarchy.levels),
        storage_sha256=hierarchy.storage.content_sha256,
        static_model_sha256=hierarchy.static_model_sha256,
        total_bytes=hierarchy.storage.total_bytes,
        preconditioner_identity=identity,
    )


def _metric_ratios(
    candidate: CommonStateMetrics,
    comparator: CommonStateMetrics,
    comparator_role: str,
) -> MetricRatios:
    _require_finite_metrics(candidate, require_reference_errors=True)
    _require_finite_metrics(comparator, require_reference_errors=True)
    if candidate.free_rms_error_m is None or comparator.free_rms_error_m is None:
        raise ValueError("free RMS metrics are required")
    if candidate.mass_weighted_rms_error_m is None or comparator.mass_weighted_rms_error_m is None:
        raise ValueError("mass-weighted RMS metrics are required")
    tiny = np.finfo(np.float64).tiny
    objective_denominator = max(abs(comparator.objective), tiny)
    residual_denominator = max(comparator.relative_residual, tiny)
    free_rms_denominator = max(comparator.free_rms_error_m, tiny)
    mass_rms_denominator = max(comparator.mass_weighted_rms_error_m, tiny)
    objective_guard = float(
        _OBJECTIVE_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * max(1.0, abs(candidate.objective), abs(comparator.objective))
    )
    values = (
        abs(candidate.objective) / objective_denominator,
        candidate.objective - comparator.objective,
        objective_guard,
        candidate.relative_residual / residual_denominator,
        candidate.free_rms_error_m / free_rms_denominator,
        candidate.mass_weighted_rms_error_m / mass_rms_denominator,
    )
    if not np.isfinite(values).all():
        raise ValueError("metric comparison ratios overflowed")
    return MetricRatios(
        comparator_role=comparator_role,
        candidate_position_sha256=candidate.position_sha256,
        comparator_position_sha256=comparator.position_sha256,
        objective_magnitude_ratio=float(values[0]),
        objective_delta=float(values[1]),
        objective_roundoff_guard=float(values[2]),
        residual_ratio=float(values[3]),
        free_rms_ratio=float(values[4]),
        mass_weighted_rms_ratio=float(values[5]),
    )


def _build_hierarchy(
    operator: MatrixFreeStableNHOperator,
    rest_positions: np.ndarray,
    config: MGVBDCorrectionConfig,
) -> StaticMultigridHierarchy:
    return build_stable_nh_rest_multigrid(
        operator,
        rest_positions,
        mode_kind=config.mode_kind,
        target_aggregate_size=config.target_aggregate_size,
        minimum_aggregate_size=config.minimum_aggregate_size,
        coarse_node_limit=config.coarse_node_limit,
        maximum_levels=config.maximum_levels,
        pre_smooth_steps=config.pre_smooth_steps,
        post_smooth_steps=config.post_smooth_steps,
        smoother_safety=config.smoother_safety,
    )


@dataclasses.dataclass
class _RecordingVCyclePreconditioner:
    """Apply one hierarchy while retaining each immutable work record."""

    hierarchy: StaticMultigridHierarchy
    work: list[VCycleWorkRecord] = dataclasses.field(default_factory=list)

    def __call__(self, residual: np.ndarray) -> np.ndarray:
        result = apply_v_cycle(self.hierarchy, residual)
        self.work.append(result.work)
        return result.correction


def run_multiplicative_mg_vbd(
    scene: TetBenchmarkScene,
    *,
    reference_positions: np.ndarray | torch.Tensor | None = None,
    reference_record: Mapping[str, object] | None = None,
    expected_reference_record_sha256: str | None = None,
    config: MGVBDCorrectionConfig | None = None,
    vbd_warmup: bool = False,
    vbd_repeats: int = 1,
    newton_warmup: bool = False,
    newton_repeats: int = 1,
) -> MGVBDRunResult:
    """Run fresh K1, multiplicative static-MG corrections, and fresh K4.

    Args:
        scene: Exact common-objective tetrahedral benchmark scene.
        reference_positions: Optional accepted dense reference positions [m].
            When omitted, a fresh independently verified dense Newton reference
            is run. Supplying positions is intended for verified PR history
            transitions whose accepted reference was built upstream;
            ``reference_record`` and its externally pinned digest are then
            mandatory and its numerical acceptance gates are re-evaluated.
        reference_record: Upstream accepted-reference record bound to the
            scene, objective, position hash, method, and accepted status.
        expected_reference_record_sha256: Externally pinned digest of
            ``reference_record``.
        config: Fixed outer-correction and hierarchy policy.
        vbd_warmup: Run one untimed VBD warmup per fresh K1/K4 call.
        vbd_repeats: Diagnostic timing repeats per fresh VBD call.
        newton_warmup: Run one untimed dense-Newton reference warmup.
        newton_repeats: Diagnostic repeats for a generated reference.

    Returns:
        Timing-free quality evidence plus a separate CPU diagnostic record.
    """
    if type(scene) is not TetBenchmarkScene:
        raise TypeError("scene must be an exact TetBenchmarkScene")
    cfg = MGVBDCorrectionConfig() if config is None else config
    if type(cfg) is not MGVBDCorrectionConfig:
        raise TypeError("config must be an exact MGVBDCorrectionConfig")
    cfg.validate()
    for name, value in (("vbd_repeats", vbd_repeats), ("newton_repeats", newton_repeats)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(vbd_warmup) is not bool or type(newton_warmup) is not bool:
        raise ValueError("warmup settings must be booleans")

    start = time.perf_counter()
    problem = build_common_problem(scene)
    problem_build_seconds = time.perf_counter() - start
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_instance_sha256 = str(common_objective_manifest(scene, problem)["objective_instance_sha256"])
    physical_state_sha256 = _array_digest(scene.x_current)
    iterate_zero = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets).detach().numpy()
    iterate_zero_sha256 = _array_digest(iterate_zero)

    newton_run = None
    reference_start = time.perf_counter()
    if reference_positions is None:
        if reference_record is not None or expected_reference_record_sha256 is not None:
            raise ValueError("reference record inputs require reference_positions")
        newton_run = run_newton(
            scene,
            problem,
            warmup=newton_warmup,
            repeats=int(newton_repeats),
        )
        if newton_run.scene_sha256 != scene_sha256:
            raise ValueError("fresh dense Newton reference belongs to another scene")
        if newton_run.objective_instance_sha256 != objective_instance_sha256:
            raise ValueError("fresh dense Newton reference belongs to another objective")
        reference = _readonly_positions(newton_run.result.x, "fresh dense Newton reference")
        reference_evidence = _newton_reference_evidence(newton_run, reference)
    else:
        if reference_record is None or expected_reference_record_sha256 is None:
            raise ValueError("supplied reference positions require a verified record and expected digest")
        reference = _readonly_positions(reference_positions, "supplied accepted reference")
        reference_evidence = None
    reference_source_seconds = time.perf_counter() - reference_start
    if reference.shape != scene.rest_q.shape:
        raise ValueError("reference_positions do not match the scene")
    evaluation_start = time.perf_counter()
    reference_metrics = evaluate_common_state(problem, reference, reference_positions=reference)
    reference_evaluation_seconds = time.perf_counter() - evaluation_start
    _require_finite_metrics(reference_metrics, require_reference_errors=True)
    if (
        reference_metrics.max_pin_error_m != 0.0
        or reference_metrics.inverted_tet_fraction != 0.0
        or reference_metrics.determinant_min <= 0.0
    ):
        raise ValueError("accepted reference must be exactly pinned and inversion-free")
    if reference_evidence is None:
        assert reference_record is not None and expected_reference_record_sha256 is not None
        reference_evidence = _supplied_reference_evidence(
            reference,
            reference_record,
            expected_reference_record_sha256,
            scene_sha256=scene_sha256,
            objective_instance_sha256=objective_instance_sha256,
            metrics=reference_metrics,
            residual_scale=problem.residual_scale,
        )

    # Deliberately separate calls. K4 never continues the K1 state.
    vbd_k1_run = run_vbd(
        scene,
        1,
        device="cpu",
        tile_solve=False,
        warmup=vbd_warmup,
        repeats=int(vbd_repeats),
    )
    vbd_k4_run = run_vbd(
        scene,
        4,
        device="cpu",
        tile_solve=False,
        warmup=vbd_warmup,
        repeats=int(vbd_repeats),
    )
    vbd_k1 = _validate_vbd_run(
        vbd_k1_run,
        role="vbd-k1",
        iterations=1,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        physical_state_sha256=physical_state_sha256,
        iterate_zero_sha256=iterate_zero_sha256,
        repeats=int(vbd_repeats),
    )
    vbd_k4 = _validate_vbd_run(
        vbd_k4_run,
        role="vbd-k4",
        iterations=4,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        physical_state_sha256=physical_state_sha256,
        iterate_zero_sha256=iterate_zero_sha256,
        repeats=int(vbd_repeats),
    )
    if (
        vbd_k1.physical_state_sha256 != vbd_k4.physical_state_sha256
        or vbd_k1.iterate_zero_sha256 != vbd_k4.iterate_zero_sha256
    ):
        raise ValueError("fresh K1 and K4 runs did not restart the same objective state")

    evaluation_start = time.perf_counter()
    k1_metrics = evaluate_common_state(problem, vbd_k1_run.positions, reference_positions=reference)
    k1_evaluation_seconds = time.perf_counter() - evaluation_start
    evaluation_start = time.perf_counter()
    k4_metrics = evaluate_common_state(problem, vbd_k4_run.positions, reference_positions=reference)
    k4_evaluation_seconds = time.perf_counter() - evaluation_start
    _require_finite_metrics(k1_metrics, require_reference_errors=True)
    _require_finite_metrics(k4_metrics, require_reference_errors=True)

    initial_operator = MatrixFreeStableNHOperator.from_problem(problem, vbd_k1_run.positions)
    hierarchy_start = time.perf_counter()
    hierarchy = _build_hierarchy(initial_operator, scene.rest_q, cfg)
    hierarchy_build_seconds = time.perf_counter() - hierarchy_start
    preconditioner_identity = f"{hierarchy.solver_contract}:rest-a0-vcycle:{hierarchy.content_sha256}"
    hierarchy_record = _hierarchy_evidence(hierarchy, preconditioner_identity)

    current = np.array(vbd_k1_run.positions, dtype=np.float64, copy=True)
    current_metrics = k1_metrics
    outer_evidence = []
    correction_seconds = []
    outer_evaluation_seconds = []
    for outer_index in range(cfg.outer_corrections):
        start_position_sha256 = _array_digest(current)
        preconditioner = _RecordingVCyclePreconditioner(hierarchy)

        correction_start = time.perf_counter()
        correction = solve_matrix_free_correction(
            problem,
            current,
            cfg.correction,
            preconditioner=preconditioner,
            preconditioner_identity=preconditioner_identity,
        )
        correction_seconds.append(time.perf_counter() - correction_start)
        if correction.preconditioner_identity != preconditioner_identity:
            raise RuntimeError("matrix-free correction changed the MG preconditioner identity")
        if correction.pcg is not None and len(preconditioner.work) != correction.pcg.work.preconditioner_applications:
            raise RuntimeError("retained V-cycle count does not match PCG preconditioner work")
        if any(work.hierarchy_sha256 != hierarchy.content_sha256 for work in preconditioner.work):
            raise RuntimeError("a correction used a different multigrid hierarchy")
        if correction.used_fallback and not np.array_equal(correction.x, current):
            raise RuntimeError("rejected correction did not return its exact input state")
        evaluation_start = time.perf_counter()
        metrics = evaluate_common_state(problem, correction.x, reference_positions=reference)
        outer_evaluation_seconds.append(time.perf_counter() - evaluation_start)
        evidence = MGOuterCorrectionEvidence(
            outer_index=outer_index,
            start_position_sha256=start_position_sha256,
            end_position_sha256=metrics.position_sha256,
            result=correction,
            start_metrics=current_metrics,
            metrics=metrics,
            v_cycle_work=tuple(preconditioner.work),
        )
        outer_evidence.append(evidence)
        current = np.array(correction.x, dtype=np.float64, copy=True)
        current_metrics = metrics
        if not correction.accepted:
            break

    final_metrics = outer_evidence[-1].metrics
    all_outer_work_completed = bool(
        len(outer_evidence) == cfg.outer_corrections and all(item.exact_work_completed for item in outer_evidence)
    )
    fallback_used = any(item.result.used_fallback for item in outer_evidence)
    exact_pins = final_metrics.max_pin_error_m == 0.0
    inversion_free = bool(final_metrics.inverted_tet_fraction == 0.0 and final_metrics.determinant_min > 0.0)
    gate = MGVBDPromotionGate(
        versus_k1=_metric_ratios(final_metrics, k1_metrics, "fresh-vbd-k1"),
        versus_k4=_metric_ratios(final_metrics, k4_metrics, "fresh-vbd-k4"),
        exact_pins=exact_pins,
        inversion_free=inversion_free,
        all_outer_work_completed=all_outer_work_completed,
        fallback_used=fallback_used,
    )

    if any(item.result.accepted for item in outer_evidence):
        final_velocities = (current - scene.x_current) / scene.dt
        final_velocities[scene.pinned_indices] = 0.0
    else:
        final_velocities = np.array(vbd_k1_run.velocities, dtype=np.float64, copy=True)

    quality = MGVBDQualityResult(
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_instance_sha256,
        config=cfg,
        reference=reference_evidence,
        hierarchy=hierarchy_record,
        vbd_k1=vbd_k1,
        vbd_k4=vbd_k4,
        dt=scene.dt,
        residual_scale=problem.residual_scale,
        pinned_indices=scene.pinned_indices,
        x_current=scene.x_current,
        reference_positions=reference,
        k1_positions=vbd_k1_run.positions,
        k1_velocities=vbd_k1_run.velocities,
        k4_positions=vbd_k4_run.positions,
        final_positions=current,
        final_velocities=final_velocities,
        reference_metrics=reference_metrics,
        k1_metrics=k1_metrics,
        k4_metrics=k4_metrics,
        outer_corrections=tuple(outer_evidence),
        final_metrics=final_metrics,
        gate=gate,
    )
    timing = MGVBDDiagnosticTiming(
        quality_sha256=quality.quality_sha256,
        problem_build_seconds=problem_build_seconds,
        reference_source_seconds=reference_source_seconds,
        reference_evaluation_seconds=reference_evaluation_seconds,
        hierarchy_build_seconds=hierarchy_build_seconds,
        outer_correction_seconds=tuple(correction_seconds),
        outer_evaluation_seconds=tuple(outer_evaluation_seconds),
        k1_evaluation_seconds=k1_evaluation_seconds,
        k4_evaluation_seconds=k4_evaluation_seconds,
        vbd_k1_run_sha256=vbd_k1_run.run_sha256,
        vbd_k4_run_sha256=vbd_k4_run.run_sha256,
        vbd_k1_setup_seconds=vbd_k1_run.setup_seconds,
        vbd_k4_setup_seconds=vbd_k4_run.setup_seconds,
        vbd_k1_warmup_seconds=vbd_k1_run.warmup_seconds,
        vbd_k4_warmup_seconds=vbd_k4_run.warmup_seconds,
        vbd_k1_repeat_seconds=vbd_k1_run.repeat_seconds,
        vbd_k4_repeat_seconds=vbd_k4_run.repeat_seconds,
        vbd_k1_transfer_seconds=vbd_k1_run.transfer_seconds,
        vbd_k4_transfer_seconds=vbd_k4_run.transfer_seconds,
        newton_run_sha256=None if newton_run is None else newton_run.run_sha256,
        newton_warmup_seconds=None if newton_run is None else newton_run.warmup_seconds,
        newton_repeat_seconds=None if newton_run is None else newton_run.repeat_seconds,
    )
    return MGVBDRunResult(quality=quality, timing=timing)
