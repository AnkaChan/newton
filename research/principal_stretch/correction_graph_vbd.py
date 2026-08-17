# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Direct multiplicative graph correction for the CPU MG-VBD oracle.

The fixed candidate in this module starts from a fresh scalar-CPU VBD K1
state and performs four nonlinear corrections.  Every correction rebuilds the
current stable-Neo-Hookean Gauss--Newton operator ``A(x)`` and applies the
static rest-tangent V-cycle ``B`` exactly twice through

``d = B b + B (b - A(x) B b)``, where ``b = -gradient(x)``.

This is the stationary, spectral-free multiplicative graph solver itself; it
does not wrap the V-cycle in PCG.  The rest hierarchy is built once and reused
unchanged.  Each accepted endpoint passes the same alpha-one strict Armijo and
exact cubic segment-determinant safeguards as the matrix-free correction
oracle.  Fresh VBD K4 and dense Newton runs are comparators only.

All quality evidence is immutable and content addressed.  Raw stationary
vectors are retained so the current-operator residual recurrence can be
independently replayed, while eager CPU timings live in a separate
diagnostic-only record and are never performance evidence.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import time

import numpy as np

from .correction_gpu import MatrixFreeStableNHOperator, minimum_determinant_on_segment
from .correction_mg_vbd import (
    HierarchyEvidence,
    MGVBDPromotionGate,
    ReferenceEvidence,
    VBDStateEvidence,
    _array_digest,
    _canonical_digest,
    _hierarchy_evidence,
    _metric_ratios,
    _newton_reference_evidence,
    _readonly_indices,
    _readonly_positions,
    _require_finite_metrics,
    _same_float64_measurement,
    _supplied_reference_evidence,
    _validate_sha256,
    _validate_vbd_run,
)
from .correction_multigrid import (
    BlockJacobiSmoother,
    HierarchyStorage,
    StaticBlockMatrix,
    StaticMultigridHierarchy,
    StaticMultigridLevel,
    TentativeProlongation,
    VCycleWorkRecord,
    _hash_parts,
    apply_v_cycle,
    build_stable_nh_rest_multigrid,
    stable_nh_static_model_digest,
)
from .solver_benchmark import (
    CommonStateMetrics,
    TetBenchmarkScene,
    build_common_problem,
    common_objective_manifest,
    evaluate_common_state,
    run_newton,
    run_vbd,
)

_QUALITY_CONTRACT = "pss-direct-multiplicative-graph-vbd-cpu-quality-v1"
_CORRECTION_CONTRACT = "pss-two-vcycle-stationary-current-a-correction-v1"
_TIMING_CONTRACT = "pss-direct-multiplicative-graph-vbd-cpu-diagnostic-timing-v1"


def _readonly_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    owned = np.array(value, dtype=np.float64, order="C", copy=True).reshape(-1)
    if owned.shape != (size,) or not np.isfinite(owned).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return np.frombuffer(owned.tobytes(order="C"), dtype=np.float64)


def _operator_sha256(operator: MatrixFreeStableNHOperator) -> str:
    """Bind one current tangent to every physical and geometric input."""
    if type(operator) is not MatrixFreeStableNHOperator:
        raise TypeError("operator must be an exact MatrixFreeStableNHOperator")
    return _canonical_digest(
        {
            "contract": "matrix-free-stable-nh-current-operator-content-v1",
            "positions_sha256": _array_digest(operator.positions),
            "tets_sha256": _array_digest(operator.tets),
            "shape_gradients_sha256": _array_digest(operator.shape_gradients),
            "volumes_sha256": _array_digest(operator.volumes),
            "mass_sha256": _array_digest(operator.mass),
            "mu_sha256": _array_digest(operator.mu),
            "lambda_sha256": _array_digest(operator.lam),
            "inertial_target_sha256": _array_digest(operator.inertial_target),
            "pinned_sha256": _array_digest(operator.pinned),
            "free_sha256": _array_digest(operator.free),
            "pin_targets_sha256": _array_digest(operator.pin_targets),
            "dt_seconds": operator.dt,
        }
    )


def _require_frozen_hierarchy_array(value: np.ndarray, name: str) -> None:
    if type(value) is not np.ndarray or value.flags["W"]:
        raise ValueError(f"hierarchy retained content {name} must be an immutable exact NumPy array")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"hierarchy retained content {name} must be finite")


def _same_nested_content(left: object, right: object) -> bool:
    """Compare every nested dataclass scalar and array without using hashes."""
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if dataclasses.is_dataclass(left):
        return all(
            _same_nested_content(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )
    if isinstance(left, tuple):
        assert isinstance(right, tuple)
        return len(left) == len(right) and all(
            _same_nested_content(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_static_hierarchy_content(hierarchy: StaticMultigridHierarchy) -> None:
    """Recompute every nested hierarchy digest from raw retained arrays."""
    if type(hierarchy) is not StaticMultigridHierarchy:
        raise TypeError("hierarchy must be an exact StaticMultigridHierarchy")
    if type(hierarchy.levels) is not tuple:
        raise ValueError("hierarchy retained content levels must be an exact immutable tuple")
    for name, value in (
        ("free_vertices", hierarchy.free_vertices),
        ("rest_positions", hierarchy.rest_positions),
        ("free_masses", hierarchy.free_masses),
        ("coarse_cholesky", hierarchy.coarse_cholesky),
    ):
        _require_frozen_hierarchy_array(value, name)

    for level_index, level in enumerate(hierarchy.levels):
        if type(level) is not StaticMultigridLevel or type(level.matrix) is not StaticBlockMatrix:
            raise ValueError("hierarchy retained content has invalid level or matrix types")
        matrix = level.matrix
        for name, value in (
            (f"level_{level_index}.matrix.row_offsets", matrix.row_offsets),
            (f"level_{level_index}.matrix.column_indices", matrix.column_indices),
            (f"level_{level_index}.matrix.values", matrix.values),
            (f"level_{level_index}.node_ids", level.node_ids),
            (f"level_{level_index}.enrichment", level.enrichment),
        ):
            _require_frozen_hierarchy_array(value, name)
        matrix_sha256 = _hash_parts(
            "static-block-matrix-v1",
            (
                ("block_row_count", matrix.block_row_count),
                ("block_size", matrix.block_size),
                ("row_offsets", matrix.row_offsets),
                ("column_indices", matrix.column_indices),
                ("values", matrix.values),
            ),
        )
        if matrix.content_sha256 != matrix_sha256:
            raise ValueError(f"hierarchy retained content level {level_index} matrix hash is stale")
        if level.aggregate is not None:
            _require_frozen_hierarchy_array(level.aggregate, f"level_{level_index}.aggregate")

        prolongation_sha256 = None
        if level.prolongation is not None:
            prolongation = level.prolongation
            if type(prolongation) is not TentativeProlongation:
                raise ValueError("hierarchy retained content has an invalid prolongation type")
            _require_frozen_hierarchy_array(
                prolongation.aggregate,
                f"level_{level_index}.prolongation.aggregate",
            )
            _require_frozen_hierarchy_array(
                prolongation.blocks,
                f"level_{level_index}.prolongation.blocks",
            )
            prolongation_sha256 = _hash_parts(
                "tentative-prolongation-v1",
                (
                    ("aggregate", prolongation.aggregate),
                    ("blocks", prolongation.blocks),
                    ("coarse_node_count", prolongation.coarse_node_count),
                ),
            )
            if prolongation.content_sha256 != prolongation_sha256:
                raise ValueError(f"hierarchy retained content level {level_index} prolongation hash is stale")

        smoother_sha256 = None
        if level.smoother is not None:
            smoother = level.smoother
            if type(smoother) is not BlockJacobiSmoother:
                raise ValueError("hierarchy retained content has an invalid smoother type")
            _require_frozen_hierarchy_array(
                smoother.inverse_diagonal,
                f"level_{level_index}.smoother.inverse_diagonal",
            )
            smoother_sha256 = _hash_parts(
                "block-jacobi-smoother-v1",
                (
                    ("inverse_diagonal", smoother.inverse_diagonal),
                    ("omega", smoother.omega),
                    ("normalized_spectral_upper_bound", smoother.normalized_spectral_upper_bound),
                ),
            )
            if smoother.content_sha256 != smoother_sha256:
                raise ValueError(f"hierarchy retained content level {level_index} smoother hash is stale")

        level_sha256 = _hash_parts(
            "static-multigrid-level-v1",
            (
                ("matrix_sha256", matrix_sha256),
                ("node_ids", level.node_ids),
                ("enrichment", level.enrichment),
                ("aggregate", level.aggregate),
                ("prolongation_sha256", prolongation_sha256),
                ("smoother_sha256", smoother_sha256),
            ),
        )
        if level.content_sha256 != level_sha256:
            raise ValueError(f"hierarchy retained content level {level_index} hash is stale")

    storage = hierarchy.storage
    if type(storage) is not HierarchyStorage:
        raise ValueError("hierarchy retained content has an invalid storage type")
    storage_parts = tuple(
        (field.name, getattr(storage, field.name))
        for field in dataclasses.fields(storage)
        if field.name != "content_sha256"
    )
    storage_sha256 = _hash_parts("hierarchy-storage-v1", storage_parts)
    if storage.content_sha256 != storage_sha256:
        raise ValueError("hierarchy retained content storage hash is stale")

    hierarchy_sha256 = _hash_parts(
        "static-multigrid-hierarchy-v1",
        (
            ("free_vertices", hierarchy.free_vertices),
            ("rest_positions", hierarchy.rest_positions),
            ("free_masses", hierarchy.free_masses),
            ("solver_contract", hierarchy.solver_contract),
            ("mode_kind", hierarchy.mode_kind),
            ("target_aggregate_size", hierarchy.target_aggregate_size),
            ("minimum_aggregate_size", hierarchy.minimum_aggregate_size),
            ("coarse_node_limit", hierarchy.coarse_node_limit),
            ("maximum_levels", hierarchy.maximum_levels),
            ("pre_smooth_steps", hierarchy.pre_smooth_steps),
            ("post_smooth_steps", hierarchy.post_smooth_steps),
            ("smoother_safety", hierarchy.smoother_safety),
            ("static_model_sha256", hierarchy.static_model_sha256),
            ("coarse_cholesky", hierarchy.coarse_cholesky),
            ("storage_sha256", storage_sha256),
            *((f"level_{index}_sha256", level.content_sha256) for index, level in enumerate(hierarchy.levels)),
        ),
    )
    if hierarchy.content_sha256 != hierarchy_sha256:
        raise ValueError("hierarchy retained content top-level hash is stale")


def _canonical_static_hierarchy(
    operator: MatrixFreeStableNHOperator,
    hierarchy: StaticMultigridHierarchy,
    rest_positions: np.ndarray,
    config: DirectGraphVBDConfig,
) -> tuple[np.ndarray, StaticMultigridHierarchy]:
    """Validate caller content against and return a canonical rebuilt A0 hierarchy."""
    if type(hierarchy) is not StaticMultigridHierarchy:
        raise TypeError("hierarchy must be an exact StaticMultigridHierarchy")
    config.validate()
    rest = _readonly_positions(rest_positions, "rest_positions")
    if rest.shape != operator.positions.shape:
        raise ValueError("rest_positions must match the current operator")
    expected_static_model_sha256 = stable_nh_static_model_digest(operator, rest)
    if hierarchy.static_model_sha256 != expected_static_model_sha256:
        raise ValueError("hierarchy static model does not match the current problem")
    if not np.array_equal(hierarchy.free_vertices, operator.free):
        raise ValueError("hierarchy free-vertex ordering does not match the current problem")
    if not np.array_equal(hierarchy.rest_positions, rest[operator.free]):
        raise ValueError("hierarchy rest geometry does not match the current problem")
    if not np.array_equal(hierarchy.free_masses, operator.mass[operator.free]):
        raise ValueError("hierarchy free masses do not match the current problem")
    hierarchy_settings = (
        (hierarchy.mode_kind, config.mode_kind),
        (hierarchy.target_aggregate_size, config.target_aggregate_size),
        (hierarchy.minimum_aggregate_size, config.minimum_aggregate_size),
        (hierarchy.coarse_node_limit, config.coarse_node_limit),
        (hierarchy.maximum_levels, config.maximum_levels),
        (hierarchy.pre_smooth_steps, config.pre_smooth_steps),
        (hierarchy.post_smooth_steps, config.post_smooth_steps),
        (hierarchy.smoother_safety, config.smoother_safety),
    )
    if any(actual != expected for actual, expected in hierarchy_settings):
        raise ValueError("hierarchy settings do not match the direct graph config")
    _validate_static_hierarchy_content(hierarchy)
    expected_hierarchy = build_stable_nh_rest_multigrid(
        operator,
        rest,
        mode_kind=config.mode_kind,
        target_aggregate_size=config.target_aggregate_size,
        minimum_aggregate_size=config.minimum_aggregate_size,
        coarse_node_limit=config.coarse_node_limit,
        maximum_levels=config.maximum_levels,
        pre_smooth_steps=config.pre_smooth_steps,
        post_smooth_steps=config.post_smooth_steps,
        smoother_safety=config.smoother_safety,
    )
    if not _same_nested_content(hierarchy, expected_hierarchy):
        raise ValueError("hierarchy retained content does not match the deterministic A0 rebuild")
    return rest, expected_hierarchy


def _same_metrics(left: CommonStateMetrics, right: CommonStateMetrics) -> bool:
    """Compare independently repeated common-objective measurements."""
    if type(left) is not CommonStateMetrics or type(right) is not CommonStateMetrics:
        return False
    if left.position_sha256 != right.position_sha256:
        return False
    for name in (
        "objective",
        "inertia",
        "elastic",
        "relative_residual",
        "determinant_min",
        "determinant_max",
        "inverted_tet_fraction",
        "minimum_singular_value",
        "max_pin_error_m",
        "free_rms_error_m",
        "mass_weighted_rms_error_m",
    ):
        a = getattr(left, name)
        b = getattr(right, name)
        if a is None or b is None:
            if a is not b:
                return False
        elif not _same_float64_measurement(a, b):
            return False
    return _same_float64_measurement(left.gradient_norm, right.gradient_norm)


def _same_cross_backend_gradient(left: float, right: float) -> bool:
    """Bound NumPy scatter versus Torch autograd cancellation roundoff."""
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    scale = max(1.0, abs(left), abs(right))
    delta = abs(left - right)
    return bool(delta <= 1024.0 * np.finfo(np.float64).eps * scale and delta / scale <= 1.0e-9)


@dataclasses.dataclass(frozen=True)
class DirectGraphVBDConfig:
    """Frozen four-by-two stationary graph correction policy."""

    outer_corrections: int = 4
    stationary_v_cycles: int = 2
    alpha: float = 1.0
    minimum_determinant: float = 0.0
    armijo: float = 1.0e-4
    mode_kind: str = "rigid"
    target_aggregate_size: int = 4
    minimum_aggregate_size: int = 3
    coarse_node_limit: int = 4
    maximum_levels: int = 8
    pre_smooth_steps: int = 1
    post_smooth_steps: int = 1
    smoother_safety: float = 0.9

    def validate(self) -> None:
        """Reject any change to the frozen candidate or invalid hierarchy."""
        integer_fields = (
            "outer_corrections",
            "stationary_v_cycles",
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
        if self.outer_corrections != 4 or self.stationary_v_cycles != 2:
            raise ValueError("the direct graph candidate is frozen at four outer corrections and two V-cycles")
        if self.alpha != 1.0:
            raise ValueError("the direct graph candidate is frozen at alpha=1")
        if not math.isfinite(self.minimum_determinant) or self.minimum_determinant < 0.0:
            raise ValueError("minimum_determinant must be finite and non-negative")
        if not math.isfinite(self.armijo) or not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must lie in (0, 1)")
        if self.mode_kind != "rigid":
            raise ValueError("the direct graph candidate requires rigid rest-space enrichment")
        if self.target_aggregate_size < 2:
            raise ValueError("target_aggregate_size must be at least two")
        if not 3 <= self.minimum_aggregate_size <= self.target_aggregate_size:
            raise ValueError("minimum_aggregate_size must lie in [3, target_aggregate_size]")
        if self.coarse_node_limit < 1 or self.maximum_levels < 2:
            raise ValueError("coarse_node_limit must be positive and maximum_levels at least two")
        if self.pre_smooth_steps < 1 or self.post_smooth_steps != self.pre_smooth_steps:
            raise ValueError("symmetric V-cycles require equal positive smoothing counts")
        if not math.isfinite(self.smoother_safety) or not 0.0 < self.smoother_safety < 1.0:
            raise ValueError("smoother_safety must lie in (0, 1)")

    def deterministic_record(self) -> dict[str, object]:
        """Serialize the timing-free fixed numerical contract."""
        self.validate()
        return {
            **dataclasses.asdict(self),
            "contract": _CORRECTION_CONTRACT,
            "linear_formula": "d=B*b+B*(b-A_current*B*b)",
            "operator_policy": "rebuild-current-A-after-each-accepted-correction",
            "hierarchy_policy": "one-static-rigid-rest-A0-hierarchy",
            "safeguard_policy": "alpha-one-strict-armijo-and-exact-cubic-segment-determinant",
            "failure_policy": "stop-at-first-rejection-and-retain-last-accepted-state",
            "performance_evidence": False,
        }


_CORRECTION_REASONS = (
    "accepted",
    "non_descent",
    "candidate_nonfinite",
    "segment_inversion",
    "objective_increase",
)


@dataclasses.dataclass(frozen=True, eq=False)
class TwoVCycleStationaryResult:
    """One replayable current-A stationary correction with exactly two cycles."""

    config: DirectGraphVBDConfig
    operator: MatrixFreeStableNHOperator = dataclasses.field(repr=False, compare=False)
    hierarchy: StaticMultigridHierarchy = dataclasses.field(repr=False, compare=False)
    rest_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    rhs: np.ndarray = dataclasses.field(repr=False, compare=False)
    first_correction: np.ndarray = dataclasses.field(repr=False, compare=False)
    residual_after_first: np.ndarray = dataclasses.field(repr=False, compare=False)
    second_correction: np.ndarray = dataclasses.field(repr=False, compare=False)
    direction: np.ndarray = dataclasses.field(repr=False, compare=False)
    true_residual: np.ndarray = dataclasses.field(repr=False, compare=False)
    v_cycle_work: tuple[VCycleWorkRecord, VCycleWorkRecord]
    candidate_positions: np.ndarray | None = dataclasses.field(repr=False, compare=False)
    x: np.ndarray = dataclasses.field(repr=False, compare=False)
    accepted: bool
    reason: str
    initial_objective: float
    candidate_objective: float | None
    final_objective: float
    initial_gradient_norm: float
    candidate_gradient_norm: float | None
    final_gradient_norm: float
    initial_minimum_determinant: float
    candidate_minimum_determinant: float | None
    final_minimum_determinant: float
    directional_derivative: float
    segment_minimum_determinant: float | None
    segment_minimum_fraction: float | None
    operator_sha256: str = dataclasses.field(init=False)
    evidence_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        if type(self.operator) is not MatrixFreeStableNHOperator:
            raise TypeError("operator must be an exact MatrixFreeStableNHOperator")
        rest_positions, hierarchy = _canonical_static_hierarchy(
            self.operator,
            self.hierarchy,
            self.rest_positions,
            self.config,
        )
        object.__setattr__(self, "rest_positions", rest_positions)
        object.__setattr__(self, "hierarchy", hierarchy)
        size = self.operator.n_free_dofs
        arrays = {}
        for name in (
            "rhs",
            "first_correction",
            "residual_after_first",
            "second_correction",
            "direction",
            "true_residual",
        ):
            arrays[name] = _readonly_vector(getattr(self, name), size, name)
            object.__setattr__(self, name, arrays[name])
        x = _readonly_positions(self.x, "x")
        object.__setattr__(self, "x", x)
        candidate = None
        if self.candidate_positions is not None:
            candidate = _readonly_positions(self.candidate_positions, "candidate_positions")
            object.__setattr__(self, "candidate_positions", candidate)
        if x.shape != self.operator.positions.shape or (candidate is not None and candidate.shape != x.shape):
            raise ValueError("correction positions have the wrong shape")
        if self.reason not in _CORRECTION_REASONS or self.accepted != (self.reason == "accepted"):
            raise ValueError("correction status is inconsistent")
        if self.hierarchy.levels[0].matrix.scalar_size != size:
            raise ValueError("hierarchy does not match the current operator")
        try:
            v_cycle_work = tuple(self.v_cycle_work)
        except TypeError as error:
            raise ValueError("v_cycle_work must be an iterable of work records") from error
        object.__setattr__(self, "v_cycle_work", v_cycle_work)
        if len(v_cycle_work) != 2 or any(type(item) is not VCycleWorkRecord for item in v_cycle_work):
            raise ValueError("stationary correction must retain exactly two V-cycle work records")

        operator_sha256 = _operator_sha256(self.operator)
        object.__setattr__(self, "operator_sha256", operator_sha256)
        expected_rhs = -self.operator.gradient_free()
        if not np.array_equal(arrays["rhs"], expected_rhs):
            raise ValueError("rhs is not the negative current nonlinear gradient")
        first = apply_v_cycle(self.hierarchy, arrays["rhs"])
        if not np.array_equal(arrays["first_correction"], first.correction) or v_cycle_work[0] != first.work:
            raise ValueError("first V-cycle evidence does not replay")
        expected_residual = arrays["rhs"] - self.operator.apply_free(arrays["first_correction"])
        if not np.array_equal(arrays["residual_after_first"], expected_residual):
            raise ValueError("current-A residual update after the first V-cycle is invalid")
        second = apply_v_cycle(self.hierarchy, arrays["residual_after_first"])
        if not np.array_equal(arrays["second_correction"], second.correction) or v_cycle_work[1] != second.work:
            raise ValueError("second V-cycle evidence does not replay")
        expected_direction = arrays["first_correction"] + arrays["second_correction"]
        if not np.array_equal(arrays["direction"], expected_direction):
            raise ValueError("stationary direction does not equal B*b+B*(b-A*B*b)")
        expected_true_residual = arrays["rhs"] - self.operator.apply_free(arrays["direction"])
        if not np.array_equal(arrays["true_residual"], expected_true_residual):
            raise ValueError("stored true residual is not b-A_current*d")

        initial_objective = self.operator.objective()
        initial_gradient_norm = float(np.linalg.norm(self.operator.gradient_free()))
        initial_determinant = self.operator.minimum_determinant
        directional_derivative = float(np.dot(self.operator.gradient_free(), arrays["direction"]))
        for measured, expected, name in (
            (self.initial_objective, initial_objective, "initial objective"),
            (self.initial_gradient_norm, initial_gradient_norm, "initial gradient"),
            (self.initial_minimum_determinant, initial_determinant, "initial determinant"),
            (self.directional_derivative, directional_derivative, "directional derivative"),
        ):
            if not _same_float64_measurement(measured, expected):
                raise ValueError(f"correction {name} is invalid")

        expected_candidate_free = self.operator.positions[self.operator.free].reshape(-1) + arrays["direction"]
        if not np.isfinite(expected_candidate_free).all():
            expected_reason = "candidate_nonfinite" if directional_derivative < 0.0 else "non_descent"
            if candidate is not None:
                raise ValueError("non-finite candidate unexpectedly retained positions")
            candidate_operator = None
        elif directional_derivative >= 0.0:
            expected_reason = "non_descent"
            if candidate is not None:
                raise ValueError("non-descent correction unexpectedly evaluated a candidate")
            candidate_operator = None
        else:
            expected_candidate = self.operator.positions_from_free(expected_candidate_free)
            if candidate is None or not np.array_equal(candidate, expected_candidate):
                raise ValueError("candidate positions do not match the stationary direction")
            try:
                candidate_operator = MatrixFreeStableNHOperator(
                    positions=candidate,
                    tets=self.operator.tets,
                    shape_gradients=self.operator.shape_gradients,
                    volumes=self.operator.volumes,
                    mass=self.operator.mass,
                    mu=self.operator.mu,
                    lam=self.operator.lam,
                    inertial_target=self.operator.inertial_target,
                    pinned=self.operator.pinned,
                    free=self.operator.free,
                    pin_targets=self.operator.pin_targets,
                    dt=self.operator.dt,
                )
            except (FloatingPointError, OverflowError, ValueError):
                candidate_operator = None
                expected_reason = "candidate_nonfinite"
            else:
                candidate_objective = candidate_operator.objective()
                candidate_gradient_norm = float(np.linalg.norm(candidate_operator.gradient_free()))
                if not math.isfinite(candidate_objective) or not math.isfinite(candidate_gradient_norm):
                    expected_reason = "candidate_nonfinite"
                else:
                    segment = minimum_determinant_on_segment(self.operator, candidate_operator)
                    if segment.determinant <= self.config.minimum_determinant:
                        expected_reason = "segment_inversion"
                    else:
                        armijo_limit = initial_objective + self.config.armijo * directional_derivative
                        expected_reason = (
                            "accepted"
                            if candidate_objective < initial_objective and candidate_objective <= armijo_limit
                            else "objective_increase"
                        )
                    scalar_pairs = (
                        (self.candidate_objective, candidate_objective, "candidate objective"),
                        (self.candidate_gradient_norm, candidate_gradient_norm, "candidate gradient"),
                        (
                            self.candidate_minimum_determinant,
                            candidate_operator.minimum_determinant,
                            "candidate determinant",
                        ),
                        (self.segment_minimum_determinant, segment.determinant, "segment determinant"),
                        (self.segment_minimum_fraction, segment.fraction, "segment fraction"),
                    )
                    for measured, expected, name in scalar_pairs:
                        if measured is None or not _same_float64_measurement(measured, expected):
                            raise ValueError(f"correction {name} is invalid")

        if self.reason != expected_reason:
            raise ValueError(f"correction reason {self.reason!r} does not match replayed {expected_reason!r}")
        expected_x = candidate if self.accepted else self.operator.positions
        if expected_x is None or not np.array_equal(x, expected_x):
            raise ValueError("correction endpoint violates fail-closed acceptance")
        expected_final = (
            (self.candidate_objective, self.candidate_gradient_norm, self.candidate_minimum_determinant)
            if self.accepted
            else (initial_objective, initial_gradient_norm, initial_determinant)
        )
        for measured, expected, name in zip(
            (self.final_objective, self.final_gradient_norm, self.final_minimum_determinant),
            expected_final,
            ("final objective", "final gradient", "final determinant"),
            strict=True,
        ):
            if expected is None or not _same_float64_measurement(measured, expected):
                raise ValueError(f"correction {name} is invalid")
        if self.operator.pinned.size and not np.array_equal(x[self.operator.pinned], self.operator.pin_targets):
            raise ValueError("correction endpoint changed an exact pin")
        object.__setattr__(self, "evidence_sha256", _canonical_digest(self._payload()))

    @property
    def exact_work_completed(self) -> bool:
        """Whether both stationary V-cycles and both current-A products exist."""
        return len(self.v_cycle_work) == 2

    @property
    def true_residual_norm(self) -> float:
        """Norm of the independently replayed current-A residual."""
        return float(np.linalg.norm(self.true_residual))

    def _payload(self) -> dict[str, object]:
        return {
            "contract": _CORRECTION_CONTRACT,
            "config": self.config.deterministic_record(),
            "operator_sha256": self.operator_sha256,
            "hierarchy_sha256": self.hierarchy.content_sha256,
            "rest_positions_sha256": _array_digest(self.rest_positions),
            "static_model_sha256": self.hierarchy.static_model_sha256,
            "start_position_sha256": _array_digest(self.operator.positions),
            "rhs_sha256": _array_digest(self.rhs),
            "first_correction_sha256": _array_digest(self.first_correction),
            "residual_after_first_sha256": _array_digest(self.residual_after_first),
            "second_correction_sha256": _array_digest(self.second_correction),
            "direction_sha256": _array_digest(self.direction),
            "true_residual_sha256": _array_digest(self.true_residual),
            "true_residual_norm": self.true_residual_norm,
            "candidate_position_sha256": None
            if self.candidate_positions is None
            else _array_digest(self.candidate_positions),
            "end_position_sha256": _array_digest(self.x),
            "v_cycle_work": [dataclasses.asdict(item) for item in self.v_cycle_work],
            "work": {
                "v_cycle_applications": 2,
                "current_operator_applications": 2,
                "nonlinear_operator_builds": 1 + int(self.candidate_positions is not None),
                "exact_work_completed": self.exact_work_completed,
            },
            "evidence_validation": {
                "hierarchy_policy": "independent-deterministic-rebuild-and-raw-deep-compare",
                "classification": "validation-only-excluded-from-algebraic-work",
            },
            "accepted": self.accepted,
            "reason": self.reason,
            "alpha": self.config.alpha,
            "initial_objective": self.initial_objective,
            "candidate_objective": self.candidate_objective,
            "final_objective": self.final_objective,
            "initial_gradient_norm": self.initial_gradient_norm,
            "candidate_gradient_norm": self.candidate_gradient_norm,
            "final_gradient_norm": self.final_gradient_norm,
            "initial_minimum_determinant": self.initial_minimum_determinant,
            "candidate_minimum_determinant": self.candidate_minimum_determinant,
            "final_minimum_determinant": self.final_minimum_determinant,
            "directional_derivative": self.directional_derivative,
            "segment_minimum_determinant": self.segment_minimum_determinant,
            "segment_minimum_fraction": self.segment_minimum_fraction,
            "performance_evidence": False,
        }

    def deterministic_record(self) -> dict[str, object]:
        """Return the timing-free content-addressed correction record."""
        payload = self._payload()
        payload["evidence_sha256"] = self.evidence_sha256
        return payload


def solve_two_vcycle_stationary_correction(
    problem,
    x_initial: np.ndarray,
    hierarchy: StaticMultigridHierarchy,
    config: DirectGraphVBDConfig | None = None,
) -> TwoVCycleStationaryResult:
    """Attempt one alpha-one correction using the fixed two-cycle formula."""
    cfg = DirectGraphVBDConfig() if config is None else config
    if type(cfg) is not DirectGraphVBDConfig:
        raise TypeError("config must be an exact DirectGraphVBDConfig")
    cfg.validate()
    if type(hierarchy) is not StaticMultigridHierarchy:
        raise TypeError("hierarchy must be an exact StaticMultigridHierarchy")
    operator = MatrixFreeStableNHOperator.from_problem(problem, x_initial)
    rest_positions, hierarchy = _canonical_static_hierarchy(
        operator,
        hierarchy,
        problem.rest_q.detach().cpu().numpy(),
        cfg,
    )
    gradient = operator.gradient_free()
    rhs = -gradient
    first = apply_v_cycle(hierarchy, rhs)
    residual_after_first = rhs - operator.apply_free(first.correction)
    second = apply_v_cycle(hierarchy, residual_after_first)
    direction = first.correction + second.correction
    true_residual = rhs - operator.apply_free(direction)
    directional_derivative = float(np.dot(gradient, direction))
    initial_objective = operator.objective()
    initial_gradient_norm = float(np.linalg.norm(gradient))
    initial_determinant = operator.minimum_determinant

    candidate_positions = None
    candidate_objective = None
    candidate_gradient_norm = None
    candidate_determinant = None
    segment_determinant = None
    segment_fraction = None
    candidate_free = operator.positions[operator.free].reshape(-1) + direction
    if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
        reason = "non_descent"
    elif not np.isfinite(candidate_free).all():
        reason = "candidate_nonfinite"
    else:
        candidate_positions = operator.positions_from_free(candidate_free)
        try:
            candidate = MatrixFreeStableNHOperator.from_problem(problem, candidate_positions)
            candidate_objective = candidate.objective()
            candidate_gradient_norm = float(np.linalg.norm(candidate.gradient_free()))
            candidate_determinant = candidate.minimum_determinant
            segment = minimum_determinant_on_segment(operator, candidate)
            segment_determinant = segment.determinant
            segment_fraction = segment.fraction
        except (FloatingPointError, OverflowError, ValueError):
            candidate_objective = None
            candidate_gradient_norm = None
            candidate_determinant = None
            segment_determinant = None
            segment_fraction = None
            reason = "candidate_nonfinite"
        else:
            if not math.isfinite(candidate_objective) or not math.isfinite(candidate_gradient_norm):
                reason = "candidate_nonfinite"
            elif segment_determinant <= cfg.minimum_determinant:
                reason = "segment_inversion"
            else:
                armijo_limit = initial_objective + cfg.armijo * directional_derivative
                reason = (
                    "accepted"
                    if candidate_objective < initial_objective and candidate_objective <= armijo_limit
                    else "objective_increase"
                )
    accepted = reason == "accepted"
    return TwoVCycleStationaryResult(
        config=cfg,
        operator=operator,
        hierarchy=hierarchy,
        rest_positions=rest_positions,
        rhs=rhs,
        first_correction=first.correction,
        residual_after_first=residual_after_first,
        second_correction=second.correction,
        direction=direction,
        true_residual=true_residual,
        v_cycle_work=(first.work, second.work),
        candidate_positions=candidate_positions,
        x=candidate_positions if accepted else operator.positions,
        accepted=accepted,
        reason=reason,
        initial_objective=initial_objective,
        candidate_objective=candidate_objective,
        final_objective=candidate_objective if accepted else initial_objective,
        initial_gradient_norm=initial_gradient_norm,
        candidate_gradient_norm=candidate_gradient_norm,
        final_gradient_norm=candidate_gradient_norm if accepted else initial_gradient_norm,
        initial_minimum_determinant=initial_determinant,
        candidate_minimum_determinant=candidate_determinant,
        final_minimum_determinant=candidate_determinant if accepted else initial_determinant,
        directional_derivative=directional_derivative,
        segment_minimum_determinant=segment_determinant,
        segment_minimum_fraction=segment_fraction,
    )


@dataclasses.dataclass(frozen=True)
class DirectGraphOuterEvidence:
    """One nonlinear outer endpoint and its independent common metrics."""

    outer_index: int
    start_metrics: CommonStateMetrics
    metrics: CommonStateMetrics
    correction: TwoVCycleStationaryResult

    def __post_init__(self) -> None:
        if isinstance(self.outer_index, bool) or not isinstance(self.outer_index, numbers.Integral):
            raise ValueError("outer_index must be an integer")
        if self.outer_index < 0:
            raise ValueError("outer_index must be non-negative")
        if type(self.correction) is not TwoVCycleStationaryResult:
            raise TypeError("correction must be an exact TwoVCycleStationaryResult")
        _require_finite_metrics(self.start_metrics, require_reference_errors=True)
        _require_finite_metrics(self.metrics, require_reference_errors=True)
        if self.start_metrics.position_sha256 != _array_digest(self.correction.operator.positions):
            raise ValueError("outer start metrics do not bind the current operator")
        if self.metrics.position_sha256 != _array_digest(self.correction.x):
            raise ValueError("outer endpoint metrics do not bind the correction")
        scalar_pairs = (
            (self.correction.initial_objective, self.start_metrics.objective, "initial objective"),
            (self.correction.initial_minimum_determinant, self.start_metrics.determinant_min, "initial determinant"),
            (self.correction.final_objective, self.metrics.objective, "final objective"),
            (self.correction.final_minimum_determinant, self.metrics.determinant_min, "final determinant"),
        )
        for measured, independent, name in scalar_pairs:
            if not _same_float64_measurement(measured, independent):
                raise ValueError(f"outer {name} disagrees with common metrics")
        for measured, independent, name in (
            (self.correction.initial_gradient_norm, self.start_metrics.gradient_norm, "initial gradient"),
            (self.correction.final_gradient_norm, self.metrics.gradient_norm, "final gradient"),
        ):
            if not _same_cross_backend_gradient(measured, independent):
                raise ValueError(f"outer {name} disagrees with common metrics")

    @property
    def exact_work_completed(self) -> bool:
        return self.correction.accepted and self.correction.exact_work_completed

    def deterministic_record(self) -> dict[str, object]:
        return {
            "outer_index": self.outer_index,
            "start_metrics": self.start_metrics.as_dict(),
            "metrics": self.metrics.as_dict(),
            "correction": self.correction.deterministic_record(),
            "exact_work_completed": self.exact_work_completed,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class DirectGraphVBDQualityResult:
    """Self-validating timing-free result for the fixed four-by-two solver."""

    scene: TetBenchmarkScene = dataclasses.field(repr=False, compare=False)
    hierarchy_object: StaticMultigridHierarchy = dataclasses.field(repr=False, compare=False)
    config: DirectGraphVBDConfig
    scene_sha256: str
    objective_instance_sha256: str
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
    k4_velocities: np.ndarray = dataclasses.field(repr=False, compare=False)
    final_positions: np.ndarray = dataclasses.field(repr=False, compare=False)
    final_velocities: np.ndarray = dataclasses.field(repr=False, compare=False)
    reference_metrics: CommonStateMetrics
    k1_metrics: CommonStateMetrics
    k4_metrics: CommonStateMetrics
    outer_corrections: tuple[DirectGraphOuterEvidence, ...]
    final_metrics: CommonStateMetrics
    gate: MGVBDPromotionGate
    quality_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        if type(self.scene) is not TetBenchmarkScene:
            raise TypeError("scene must be an exact TetBenchmarkScene")
        if type(self.hierarchy_object) is not StaticMultigridHierarchy:
            raise TypeError("hierarchy_object must be an exact StaticMultigridHierarchy")
        for name in ("scene_sha256", "objective_instance_sha256"):
            _validate_sha256(getattr(self, name), name)
        if not math.isfinite(self.dt) or self.dt <= 0.0 or self.dt != self.scene.dt:
            raise ValueError("quality dt must equal the scene dt")
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        pinned = _readonly_indices(self.pinned_indices, "pinned_indices")
        object.__setattr__(self, "pinned_indices", pinned)
        arrays = {}
        for name in (
            "x_current",
            "reference_positions",
            "k1_positions",
            "k1_velocities",
            "k4_positions",
            "k4_velocities",
            "final_positions",
            "final_velocities",
        ):
            arrays[name] = _readonly_positions(getattr(self, name), name)
            object.__setattr__(self, name, arrays[name])
        shape = self.scene.rest_q.shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("quality states must match the scene shape")
        if not np.array_equal(pinned, self.scene.pinned_indices):
            raise ValueError("quality pinned_indices changed the scene")
        if not np.array_equal(arrays["x_current"], self.scene.x_current):
            raise ValueError("quality x_current changed the physical state")

        problem = build_common_problem(self.scene)
        expected_scene_sha256 = str(self.scene.manifest()["scene_sha256"])
        expected_objective_sha256 = str(common_objective_manifest(self.scene, problem)["objective_instance_sha256"])
        if self.scene_sha256 != expected_scene_sha256 or self.objective_instance_sha256 != expected_objective_sha256:
            raise ValueError("quality scene/objective identities do not match the supplied scene")
        if self.residual_scale != problem.residual_scale:
            raise ValueError("quality residual_scale changed the common objective")

        independently_scored = (
            evaluate_common_state(
                problem, arrays["reference_positions"], reference_positions=arrays["reference_positions"]
            ),
            evaluate_common_state(problem, arrays["k1_positions"], reference_positions=arrays["reference_positions"]),
            evaluate_common_state(problem, arrays["k4_positions"], reference_positions=arrays["reference_positions"]),
            evaluate_common_state(
                problem, arrays["final_positions"], reference_positions=arrays["reference_positions"]
            ),
        )
        supplied_metrics = (self.reference_metrics, self.k1_metrics, self.k4_metrics, self.final_metrics)
        for supplied, independent, role in zip(
            supplied_metrics,
            independently_scored,
            ("reference", "K1", "K4", "final"),
            strict=True,
        ):
            _require_finite_metrics(supplied, require_reference_errors=True)
            if not _same_metrics(supplied, independent):
                raise ValueError(f"{role} metrics do not match independent common evaluation")
        if (
            self.reference_metrics.max_pin_error_m != 0.0
            or self.reference_metrics.inverted_tet_fraction != 0.0
            or self.reference_metrics.determinant_min <= 0.0
        ):
            raise ValueError("accepted reference must be exactly pinned and inversion-free")

        hashes = {
            "reference": _array_digest(arrays["reference_positions"]),
            "k1": _array_digest(arrays["k1_positions"]),
            "k4": _array_digest(arrays["k4_positions"]),
            "final": _array_digest(arrays["final_positions"]),
        }
        if self.reference.position_sha256 != hashes["reference"]:
            raise ValueError("reference evidence does not bind reference_positions")
        if (
            self.reference.scene_sha256 != self.scene_sha256
            or self.reference.objective_instance_sha256 != self.objective_instance_sha256
        ):
            raise ValueError("reference evidence belongs to another scene/objective")
        if self.reference.provenance != "fresh-dense-newton":
            raise ValueError("reference evidence must come from the fresh dense Newton run")
        if (
            self.reference.source_record.get("native_converged") is not True
            or self.reference.source_record.get("native_reason") != "gradient"
        ):
            raise ValueError("fresh reference lacks converged native Newton evidence")
        validated_reference = _supplied_reference_evidence(
            arrays["reference_positions"],
            self.reference.source_record,
            self.reference.source_record_sha256,
            scene_sha256=self.scene_sha256,
            objective_instance_sha256=self.objective_instance_sha256,
            metrics=self.reference_metrics,
            residual_scale=self.residual_scale,
        )
        validated_binding = validated_reference.deterministic_record()
        supplied_binding = self.reference.deterministic_record()
        # Fresh and supplied evidence intentionally use different provenance
        # labels.  Every content binding returned by the common reference
        # validator must otherwise agree exactly.
        validated_binding.pop("provenance")
        supplied_binding.pop("provenance")
        if validated_binding != supplied_binding:
            raise ValueError("reference evidence failed exact accepted-reference binding")
        if self.vbd_k1.position_sha256 != hashes["k1"] or self.vbd_k4.position_sha256 != hashes["k4"]:
            raise ValueError("VBD evidence does not bind K1/K4 positions")
        if self.vbd_k1.velocity_sha256 != _array_digest(arrays["k1_velocities"]):
            raise ValueError("K1 evidence does not bind K1 velocity")
        if self.vbd_k4.velocity_sha256 != _array_digest(arrays["k4_velocities"]):
            raise ValueError("K4 evidence does not bind K4 velocity")
        if self.vbd_k1.role != "vbd-k1" or self.vbd_k1.iterations != 1:
            raise ValueError("K1 evidence changed the fresh one-iteration role")
        if self.vbd_k4.role != "vbd-k4" or self.vbd_k4.iterations != 4:
            raise ValueError("K4 evidence changed the fresh four-iteration role")
        for evidence in (self.vbd_k1, self.vbd_k4):
            if (
                evidence.scene_sha256 != self.scene_sha256
                or evidence.objective_instance_sha256 != self.objective_instance_sha256
            ):
                raise ValueError("VBD evidence belongs to another scene/objective")
        expected_color_group_count = int(self.scene.color_group_offsets.size - 1)
        if self.vbd_k1.color_group_count != expected_color_group_count:
            raise ValueError("K1 color-group count does not match the scene")
        if self.vbd_k4.color_group_count != expected_color_group_count:
            raise ValueError("K4 color-group count does not match the scene")
        if self.vbd_k1.physical_state_sha256 != self.vbd_k4.physical_state_sha256:
            raise ValueError("fresh K1 and K4 physical-state identities disagree")
        if self.vbd_k1.physical_state_sha256 != _array_digest(arrays["x_current"]):
            raise ValueError("fresh VBD evidence does not bind x_current")
        if self.vbd_k1.iterate_zero_sha256 != self.vbd_k4.iterate_zero_sha256:
            raise ValueError("fresh K1 and K4 iterate-zero identities disagree")
        iterate_zero = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets).detach().numpy()
        if self.vbd_k1.iterate_zero_sha256 != _array_digest(iterate_zero):
            raise ValueError("fresh VBD evidence does not bind the common iterate zero")

        initial_operator = MatrixFreeStableNHOperator.from_problem(problem, arrays["k1_positions"])
        _rest_positions, canonical_hierarchy = _canonical_static_hierarchy(
            initial_operator,
            self.hierarchy_object,
            self.scene.rest_q,
            self.config,
        )
        object.__setattr__(self, "hierarchy_object", canonical_hierarchy)
        hierarchy_identity = (
            f"{canonical_hierarchy.solver_contract}:rest-a0-vcycle:{canonical_hierarchy.content_sha256}"
        )
        expected_hierarchy = _hierarchy_evidence(canonical_hierarchy, hierarchy_identity)
        if self.hierarchy != expected_hierarchy:
            raise ValueError("hierarchy evidence does not bind the retained hierarchy")
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
            raise ValueError("hierarchy evidence changed the configured policy")

        outer = tuple(self.outer_corrections)
        object.__setattr__(self, "outer_corrections", outer)
        if not outer or len(outer) > self.config.outer_corrections:
            raise ValueError("outer correction evidence has an invalid length")
        if tuple(item.outer_index for item in outer) != tuple(range(len(outer))):
            raise ValueError("outer correction indices must be contiguous")
        expected_start_hash = hashes["k1"]
        expected_start_metrics = self.k1_metrics
        for index, item in enumerate(outer):
            if type(item) is not DirectGraphOuterEvidence:
                raise TypeError("outer evidence must contain exact DirectGraphOuterEvidence instances")
            if (
                item.start_metrics.position_sha256 != expected_start_hash
                or item.start_metrics != expected_start_metrics
            ):
                raise ValueError(f"outer correction {index} does not continue the previous endpoint")
            expected_operator = MatrixFreeStableNHOperator.from_problem(problem, item.correction.operator.positions)
            if item.correction.operator_sha256 != _operator_sha256(expected_operator):
                raise ValueError(f"outer correction {index} did not rebuild the current operator")
            if item.correction.hierarchy.content_sha256 != self.hierarchy_object.content_sha256:
                raise ValueError(f"outer correction {index} changed the static hierarchy")
            if item.correction.config != self.config:
                raise ValueError(f"outer correction {index} changed the fixed config")
            if not item.correction.accepted and index != len(outer) - 1:
                raise ValueError("a rejected correction must terminate the outer sequence")
            expected_start_hash = item.metrics.position_sha256
            expected_start_metrics = item.metrics
        if outer[-1].metrics.position_sha256 != hashes["final"] or outer[-1].metrics != self.final_metrics:
            raise ValueError("final state does not match the last outer endpoint")

        accepted_count = sum(item.correction.accepted for item in outer)
        if accepted_count:
            expected_velocity = (arrays["final_positions"] - arrays["x_current"]) / self.dt
            expected_velocity[pinned] = 0.0
        else:
            expected_velocity = arrays["k1_velocities"]
        if not np.array_equal(arrays["final_velocities"], expected_velocity):
            raise ValueError("final velocity violates the one-shot BDF1 policy")
        if pinned.size and not np.array_equal(arrays["final_positions"][pinned], self.scene.pin_targets):
            raise ValueError("final positions changed exact pins")

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
            fallback_used=any(not item.correction.accepted for item in outer),
        )
        if self.gate != expected_gate:
            raise ValueError("promotion gate does not match independently recomputed states and work")
        object.__setattr__(self, "quality_sha256", _canonical_digest(self._payload()))

    @property
    def accepted_outer_correction_count(self) -> int:
        return sum(item.correction.accepted for item in self.outer_corrections)

    @property
    def total_v_cycle_count(self) -> int:
        return sum(len(item.correction.v_cycle_work) for item in self.outer_corrections)

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
            "k1_velocity_sha256": _array_digest(self.k1_velocities),
            "k4_velocity_sha256": _array_digest(self.k4_velocities),
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
            "velocity_policy": "BDF1-once-from-x_n-after-last-accepted-correction; exact-K1-on-zero-accept",
            "accepted_outer_correction_count": self.accepted_outer_correction_count,
            "total_v_cycle_count": self.total_v_cycle_count,
            "gate": self.gate.deterministic_record(),
        }

    def deterministic_record(self) -> dict[str, object]:
        """Return the complete content-addressed quality record."""
        payload = self._payload()
        payload["quality_sha256"] = self.quality_sha256
        return payload


@dataclasses.dataclass(frozen=True)
class DirectGraphVBDDiagnosticTiming:
    """Eager CPU diagnostics deliberately excluded from quality identity."""

    quality_sha256: str
    problem_build_seconds: float
    reference_run_sha256: str
    reference_seconds: float
    vbd_k1_run_sha256: str
    vbd_k4_run_sha256: str
    vbd_k1_seconds: float
    vbd_k4_seconds: float
    hierarchy_build_seconds: float
    correction_seconds: tuple[float, ...]
    evaluation_seconds: tuple[float, ...]

    def deterministic_record(self) -> dict[str, object]:
        payload = {
            "contract": _TIMING_CONTRACT,
            "performance_evidence": False,
            "measurement_provenance": "eager-scalar-cpu-diagnostic-only",
            "correction_seconds_scope": "includes-evidence-only-hierarchy-validation-rebuilds",
            **dataclasses.asdict(self),
        }
        payload["timing_sha256"] = _canonical_digest(payload)
        return payload


@dataclasses.dataclass(frozen=True)
class DirectGraphVBDRunResult:
    """Timing-free quality plus separate diagnostic-only CPU timings."""

    quality: DirectGraphVBDQualityResult
    timing: DirectGraphVBDDiagnosticTiming

    def __post_init__(self) -> None:
        if self.timing.quality_sha256 != self.quality.quality_sha256:
            raise ValueError("diagnostic timing belongs to another quality result")


def _build_hierarchy(
    operator: MatrixFreeStableNHOperator,
    scene: TetBenchmarkScene,
    config: DirectGraphVBDConfig,
) -> StaticMultigridHierarchy:
    return build_stable_nh_rest_multigrid(
        operator,
        scene.rest_q,
        mode_kind=config.mode_kind,
        target_aggregate_size=config.target_aggregate_size,
        minimum_aggregate_size=config.minimum_aggregate_size,
        coarse_node_limit=config.coarse_node_limit,
        maximum_levels=config.maximum_levels,
        pre_smooth_steps=config.pre_smooth_steps,
        post_smooth_steps=config.post_smooth_steps,
        smoother_safety=config.smoother_safety,
    )


def run_direct_graph_vbd(
    scene: TetBenchmarkScene,
    *,
    config: DirectGraphVBDConfig | None = None,
    vbd_warmup: bool = False,
    vbd_repeats: int = 1,
    newton_warmup: bool = False,
    newton_repeats: int = 1,
) -> DirectGraphVBDRunResult:
    """Run fresh K1, the fixed direct graph solver, fresh K4, and Newton."""
    if type(scene) is not TetBenchmarkScene:
        raise TypeError("scene must be an exact TetBenchmarkScene")
    cfg = DirectGraphVBDConfig() if config is None else config
    if type(cfg) is not DirectGraphVBDConfig:
        raise TypeError("config must be an exact DirectGraphVBDConfig")
    cfg.validate()
    for name, value in (("vbd_repeats", vbd_repeats), ("newton_repeats", newton_repeats)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(vbd_warmup) is not bool or type(newton_warmup) is not bool:
        raise ValueError("warmup settings must be booleans")

    start = time.perf_counter()
    problem = build_common_problem(scene)
    problem_seconds = time.perf_counter() - start
    scene_sha256 = str(scene.manifest()["scene_sha256"])
    objective_sha256 = str(common_objective_manifest(scene, problem)["objective_instance_sha256"])
    physical_state_sha256 = _array_digest(scene.x_current)
    iterate_zero = problem.inertial_target.index_copy(0, problem.pinned, problem.pin_targets).detach().numpy()
    iterate_zero_sha256 = _array_digest(iterate_zero)

    start = time.perf_counter()
    newton_run = run_newton(scene, problem, warmup=newton_warmup, repeats=int(newton_repeats))
    reference_seconds = time.perf_counter() - start
    if newton_run.scene_sha256 != scene_sha256 or newton_run.objective_instance_sha256 != objective_sha256:
        raise ValueError("fresh Newton reference belongs to another scene/objective")
    reference_positions = _readonly_positions(newton_run.result.x, "reference_positions")
    reference_evidence = _newton_reference_evidence(newton_run, reference_positions)

    start = time.perf_counter()
    k1_run = run_vbd(
        scene,
        1,
        device="cpu",
        tile_solve=False,
        warmup=vbd_warmup,
        repeats=int(vbd_repeats),
    )
    k1_seconds = time.perf_counter() - start
    start = time.perf_counter()
    k4_run = run_vbd(
        scene,
        4,
        device="cpu",
        tile_solve=False,
        warmup=vbd_warmup,
        repeats=int(vbd_repeats),
    )
    k4_seconds = time.perf_counter() - start
    k1_evidence = _validate_vbd_run(
        k1_run,
        role="vbd-k1",
        iterations=1,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_sha256,
        physical_state_sha256=physical_state_sha256,
        iterate_zero_sha256=iterate_zero_sha256,
        repeats=int(vbd_repeats),
    )
    k4_evidence = _validate_vbd_run(
        k4_run,
        role="vbd-k4",
        iterations=4,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_sha256,
        physical_state_sha256=physical_state_sha256,
        iterate_zero_sha256=iterate_zero_sha256,
        repeats=int(vbd_repeats),
    )
    if (
        k1_evidence.physical_state_sha256 != k4_evidence.physical_state_sha256
        or k1_evidence.iterate_zero_sha256 != k4_evidence.iterate_zero_sha256
    ):
        raise ValueError("fresh K1 and K4 did not restart the same objective state")

    reference_metrics = evaluate_common_state(problem, reference_positions, reference_positions=reference_positions)
    k1_metrics = evaluate_common_state(problem, k1_run.positions, reference_positions=reference_positions)
    k4_metrics = evaluate_common_state(problem, k4_run.positions, reference_positions=reference_positions)
    for metrics in (reference_metrics, k1_metrics, k4_metrics):
        _require_finite_metrics(metrics, require_reference_errors=True)

    initial_operator = MatrixFreeStableNHOperator.from_problem(problem, k1_run.positions)
    start = time.perf_counter()
    hierarchy = _build_hierarchy(initial_operator, scene, cfg)
    hierarchy_seconds = time.perf_counter() - start
    hierarchy_identity = f"{hierarchy.solver_contract}:rest-a0-vcycle:{hierarchy.content_sha256}"
    hierarchy_evidence = _hierarchy_evidence(hierarchy, hierarchy_identity)

    current = np.array(k1_run.positions, dtype=np.float64, copy=True)
    current_metrics = k1_metrics
    outer_evidence = []
    correction_seconds = []
    evaluation_seconds = []
    for outer_index in range(cfg.outer_corrections):
        start = time.perf_counter()
        correction = solve_two_vcycle_stationary_correction(problem, current, hierarchy, cfg)
        correction_seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        metrics = evaluate_common_state(problem, correction.x, reference_positions=reference_positions)
        evaluation_seconds.append(time.perf_counter() - start)
        outer_evidence.append(
            DirectGraphOuterEvidence(
                outer_index=outer_index,
                start_metrics=current_metrics,
                metrics=metrics,
                correction=correction,
            )
        )
        current = np.array(correction.x, dtype=np.float64, copy=True)
        current_metrics = metrics
        if not correction.accepted:
            break

    final_metrics = outer_evidence[-1].metrics
    gate = MGVBDPromotionGate(
        versus_k1=_metric_ratios(final_metrics, k1_metrics, "fresh-vbd-k1"),
        versus_k4=_metric_ratios(final_metrics, k4_metrics, "fresh-vbd-k4"),
        exact_pins=final_metrics.max_pin_error_m == 0.0,
        inversion_free=bool(final_metrics.inverted_tet_fraction == 0.0 and final_metrics.determinant_min > 0.0),
        all_outer_work_completed=bool(
            len(outer_evidence) == cfg.outer_corrections and all(item.exact_work_completed for item in outer_evidence)
        ),
        fallback_used=any(not item.correction.accepted for item in outer_evidence),
    )
    if any(item.correction.accepted for item in outer_evidence):
        final_velocities = (current - scene.x_current) / scene.dt
        final_velocities[scene.pinned_indices] = 0.0
    else:
        final_velocities = np.array(k1_run.velocities, dtype=np.float64, copy=True)

    quality = DirectGraphVBDQualityResult(
        scene=scene,
        hierarchy_object=hierarchy,
        config=cfg,
        scene_sha256=scene_sha256,
        objective_instance_sha256=objective_sha256,
        reference=reference_evidence,
        hierarchy=hierarchy_evidence,
        vbd_k1=k1_evidence,
        vbd_k4=k4_evidence,
        dt=scene.dt,
        residual_scale=problem.residual_scale,
        pinned_indices=scene.pinned_indices,
        x_current=scene.x_current,
        reference_positions=reference_positions,
        k1_positions=k1_run.positions,
        k1_velocities=k1_run.velocities,
        k4_positions=k4_run.positions,
        k4_velocities=k4_run.velocities,
        final_positions=current,
        final_velocities=final_velocities,
        reference_metrics=reference_metrics,
        k1_metrics=k1_metrics,
        k4_metrics=k4_metrics,
        outer_corrections=tuple(outer_evidence),
        final_metrics=final_metrics,
        gate=gate,
    )
    timing = DirectGraphVBDDiagnosticTiming(
        quality_sha256=quality.quality_sha256,
        problem_build_seconds=problem_seconds,
        reference_run_sha256=newton_run.run_sha256,
        reference_seconds=reference_seconds,
        vbd_k1_run_sha256=k1_run.run_sha256,
        vbd_k4_run_sha256=k4_run.run_sha256,
        vbd_k1_seconds=k1_seconds,
        vbd_k4_seconds=k4_seconds,
        hierarchy_build_seconds=hierarchy_seconds,
        correction_seconds=tuple(correction_seconds),
        evaluation_seconds=tuple(evaluation_seconds),
    )
    return DirectGraphVBDRunResult(quality=quality, timing=timing)
