# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""First fully captured multiplicative MG-VBD research composition.

The CUDA graph scheduled here starts from a pristine public ``SolverVBD`` K1
substep, converts its float32 endpoint to the common float64 objective, and
then schedules exactly three current-operator PCG4 corrections.  Every PCG
uses the same spectral-free static-rest multigrid hierarchy.  Candidate
selection stays on device and fails closed on PCG work shortfall, nonfinite
data, non-descent, exact cubic segment inversion, or a strict Armijo failure.

This is an intentionally narrow research harness.  It supports the
contact-free particle scenes used by the principal-stretch suite and reports
timing only as a diagnostic, not as production performance evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import statistics

import numpy as np
import warp as wp

from .captured_vbd_baseline import CapturedPublicVBDBaseline
from .correction_gpu import MatrixFreeStableNHOperator
from .correction_gpu_warp import WarpFixedPCGWorkspace, WarpMatrixFreeStableNHOperator
from .correction_mg_vbd import MGVBDCorrectionConfig
from .correction_multigrid import build_stable_nh_rest_multigrid
from .correction_multigrid_warp import (
    WarpStaticMultigridHierarchy,
    WarpStaticMultigridPreconditioner,
)
from .solver_benchmark import TetBenchmarkScene, build_common_problem, common_objective_manifest

CONTRACT_ID = "captured-multiplicative-mg-vbd-v1"
OUTER_CORRECTIONS = 3
PCG_ITERATIONS = 4

_PCG_COMPLETED = 1
_REASON_PENDING = 0
_REASON_ACCEPTED = 1
_REASON_MASKED = 2
_REASON_PCG = 3
_REASON_NONFINITE = 4
_REASON_NON_DESCENT = 5
_REASON_SEGMENT_INVERSION = 6
_REASON_OBJECTIVE = 7

REASON_NAMES = (
    "pending",
    "accepted",
    "masked-after-rejection",
    "pcg-failure-or-work-shortfall",
    "candidate-nonfinite",
    "non-descent",
    "segment-inversion",
    "objective-increase",
)


def _immutable_float64(value: np.ndarray, *, name: str) -> np.ndarray:
    """Copy a finite float64 array into immutable bytes-backed storage."""
    owned = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(owned).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _array_digest(value: np.ndarray) -> str:
    """Hash an array together with its canonical dtype and shape."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    """Hash finite canonical JSON."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    """Require a canonical lowercase SHA-256 string."""
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


@wp.func
def _outer(left: wp.vec3d, right: wp.vec3d) -> wp.mat33d:
    return wp.mat33d(
        left[0] * right[0],
        left[0] * right[1],
        left[0] * right[2],
        left[1] * right[0],
        left[1] * right[1],
        left[1] * right[2],
        left[2] * right[0],
        left[2] * right[1],
        left[2] * right[2],
    )


@wp.func
def _zero_matrix() -> wp.mat33d:
    return wp.mat33d(
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )


@wp.func
def _cofactor(matrix: wp.mat33d) -> wp.mat33d:
    return wp.mat33d(
        matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1],
        matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2],
        matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0],
        matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2],
        matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0],
        matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1],
        matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1],
        matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2],
        matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0],
    )


@wp.func
def _determinant(matrix: wp.mat33d) -> wp.float64:
    return (
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


@wp.func
def _double_dot(left: wp.mat33d, right: wp.mat33d) -> wp.float64:
    value = wp.float64(0.0)
    for row in range(3):
        for column in range(3):
            value += left[row, column] * right[row, column]
    return value


@wp.func
def _finite_vec(value: wp.vec3d) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _cubic_value(
    c0: wp.float64,
    c1: wp.float64,
    c2: wp.float64,
    c3: wp.float64,
    s: wp.float64,
) -> wp.float64:
    return ((c3 * s + c2) * s + c1) * s + c0


@wp.func
def _segment_minimum(c0: wp.float64, c1: wp.float64, c2: wp.float64, c3: wp.float64) -> wp.float64:
    """Evaluate endpoints and every real interior root of the cubic derivative."""
    best = wp.min(c0, c0 + c1 + c2 + c3)
    scale = wp.max(wp.abs(c1), wp.max(wp.abs(wp.float64(2.0) * c2), wp.abs(wp.float64(3.0) * c3)))
    epsilon = wp.float64(2.220446049250313e-16)
    if scale > wp.float64(0.0):
        c = c1 / scale
        b = wp.float64(2.0) * c2 / scale
        a = wp.float64(3.0) * c3 / scale
        if wp.abs(a) <= wp.float64(32.0) * epsilon:
            if wp.abs(b) > wp.float64(32.0) * epsilon:
                root = -c / b
                if root > wp.float64(0.0) and root < wp.float64(1.0):
                    best = wp.min(best, _cubic_value(c0, c1, c2, c3, root))
        else:
            discriminant = b * b - wp.float64(4.0) * a * c
            tolerance = (
                wp.float64(64.0)
                * epsilon
                * wp.max(wp.float64(1.0), wp.max(wp.abs(b * b), wp.abs(wp.float64(4.0) * a * c)))
            )
            if discriminant >= -tolerance:
                square_root = wp.sqrt(wp.max(wp.float64(0.0), discriminant))
                signed_root = square_root
                if b < wp.float64(0.0):
                    signed_root = -square_root
                q = -wp.float64(0.5) * (b + signed_root)
                if q == wp.float64(0.0):
                    root = -b / (wp.float64(2.0) * a)
                    if root > wp.float64(0.0) and root < wp.float64(1.0):
                        best = wp.min(best, _cubic_value(c0, c1, c2, c3, root))
                else:
                    root = q / a
                    if root > wp.float64(0.0) and root < wp.float64(1.0):
                        best = wp.min(best, _cubic_value(c0, c1, c2, c3, root))
                    root = c / q
                    if root > wp.float64(0.0) and root < wp.float64(1.0):
                        best = wp.min(best, _cubic_value(c0, c1, c2, c3, root))
    return best


@wp.kernel(enable_backward=False)
def _initialize_from_k1(
    k1_positions: wp.array[wp.vec3],
    canonical_positions: wp.array[wp.vec3d],
    vertex_to_free: wp.array[int],
    current: wp.array[wp.vec3d],
    candidate: wp.array[wp.vec3d],
    active: wp.array[int],
    accepted: wp.array[int],
    reasons: wp.array[int],
):
    vertex = wp.tid()
    source = k1_positions[vertex]
    value = wp.vec3d(wp.float64(source[0]), wp.float64(source[1]), wp.float64(source[2]))
    if vertex_to_free[vertex] < 0:
        value = canonical_positions[vertex]
    current[vertex] = value
    candidate[vertex] = value
    if vertex == 0:
        active[0] = 1
        for outer_index in range(OUTER_CORRECTIONS):
            accepted[outer_index] = 0
            reasons[outer_index] = _REASON_PENDING


@wp.kernel(enable_backward=False)
def _mask_rhs(active: wp.array[int], rhs: wp.array[wp.vec3d]):
    index = wp.tid()
    if active[0] == 0:
        rhs[index] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.kernel(enable_backward=False)
def _copy_positions(source: wp.array[wp.vec3d], destination: wp.array[wp.vec3d]):
    destination[wp.tid()] = source[wp.tid()]


@wp.kernel(enable_backward=False)
def _build_candidate(
    current: wp.array[wp.vec3d],
    vertex_to_free: wp.array[int],
    solution: wp.array[wp.vec3d],
    pcg_status: wp.array[int],
    pcg_completed: wp.array[int],
    active: wp.array[int],
    candidate: wp.array[wp.vec3d],
):
    vertex = wp.tid()
    value = current[vertex]
    free_index = vertex_to_free[vertex]
    if active[0] != 0 and pcg_status[0] == _PCG_COMPLETED and pcg_completed[0] == PCG_ITERATIONS:
        if free_index >= 0:
            proposed = value + solution[free_index]
            publishable = wp.vec3(
                wp.float32(proposed[0]),
                wp.float32(proposed[1]),
                wp.float32(proposed[2]),
            )
            value = wp.vec3d(
                wp.float64(publishable[0]),
                wp.float64(publishable[1]),
                wp.float64(publishable[2]),
            )
    candidate[vertex] = value


@wp.kernel(enable_backward=False)
def _vertex_gate_terms(
    current: wp.array[wp.vec3d],
    candidate: wp.array[wp.vec3d],
    inertial_target: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    inverse_dt_squared: wp.float64,
    current_inertia: wp.array[wp.float64],
    candidate_inertia: wp.array[wp.float64],
    vertex_finite: wp.array[int],
):
    vertex = wp.tid()
    start_delta = current[vertex] - inertial_target[vertex]
    end_delta = candidate[vertex] - inertial_target[vertex]
    current_inertia[vertex] = wp.float64(0.5) * inverse_dt_squared * mass[vertex] * wp.dot(start_delta, start_delta)
    candidate_inertia[vertex] = wp.float64(0.5) * inverse_dt_squared * mass[vertex] * wp.dot(end_delta, end_delta)
    vertex_finite[vertex] = int(_finite_vec(current[vertex]) and _finite_vec(candidate[vertex]))


@wp.kernel(enable_backward=False)
def _directional_terms(
    rhs: wp.array[wp.vec3d],
    free_vertices: wp.array[int],
    current: wp.array[wp.vec3d],
    candidate: wp.array[wp.vec3d],
    terms: wp.array[wp.float64],
):
    index = wp.tid()
    vertex = free_vertices[index]
    terms[index] = -wp.dot(rhs[index], candidate[vertex] - current[vertex])


@wp.kernel(enable_backward=False)
def _tet_gate_terms(
    current_deformation: wp.array[wp.mat33d],
    current_cofactor: wp.array[wp.mat33d],
    current_determinant: wp.array[wp.float64],
    candidate: wp.array[wp.vec3d],
    tets: wp.array[int],
    shape_gradients: wp.array[wp.vec3d],
    volumes: wp.array[wp.float64],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    current_elastic: wp.array[wp.float64],
    candidate_elastic: wp.array[wp.float64],
    candidate_determinants: wp.array[wp.float64],
    segment_minima: wp.array[wp.float64],
    tet_finite: wp.array[int],
):
    tet = wp.tid()
    end_deformation = _zero_matrix()
    for corner in range(4):
        entry = 4 * tet + corner
        end_deformation += _outer(candidate[tets[entry]], shape_gradients[entry])
    end_determinant = _determinant(end_deformation)
    start_deformation = current_deformation[tet]
    delta_deformation = end_deformation - start_deformation
    c0 = current_determinant[tet]
    c1 = _double_dot(current_cofactor[tet], delta_deformation)
    c2 = _double_dot(_cofactor(delta_deformation), start_deformation)
    c3 = _determinant(delta_deformation)
    segment_minimum = _segment_minimum(c0, c1, c2, c3)
    alpha = wp.float64(1.0) + mu[tet] / wp.max(lam[tet], wp.float64(1.0e-6))
    start_density = wp.float64(0.5) * mu[tet] * (
        _double_dot(start_deformation, start_deformation) - wp.float64(3.0)
    ) + wp.float64(0.5) * lam[tet] * (c0 - alpha) * (c0 - alpha)
    end_density = wp.float64(0.5) * mu[tet] * (
        _double_dot(end_deformation, end_deformation) - wp.float64(3.0)
    ) + wp.float64(0.5) * lam[tet] * (end_determinant - alpha) * (end_determinant - alpha)
    current_elastic[tet] = volumes[tet] * start_density
    candidate_elastic[tet] = volumes[tet] * end_density
    candidate_determinants[tet] = end_determinant
    segment_minima[tet] = segment_minimum
    tet_finite[tet] = int(
        wp.isfinite(start_density)
        and wp.isfinite(end_density)
        and wp.isfinite(end_determinant)
        and wp.isfinite(segment_minimum)
    )


@wp.kernel(enable_backward=False)
def _finalize_gate(
    outer_index: int,
    pcg_status: wp.array[int],
    pcg_completed: wp.array[int],
    current_inertia: wp.array[wp.float64],
    candidate_inertia: wp.array[wp.float64],
    current_elastic: wp.array[wp.float64],
    candidate_elastic: wp.array[wp.float64],
    directional_terms: wp.array[wp.float64],
    candidate_determinants: wp.array[wp.float64],
    segment_minima: wp.array[wp.float64],
    vertex_finite: wp.array[int],
    tet_finite: wp.array[int],
    minimum_determinant: wp.float64,
    armijo: wp.float64,
    active: wp.array[int],
    accepted: wp.array[int],
    reasons: wp.array[int],
    initial_objective: wp.array[wp.float64],
    candidate_objective: wp.array[wp.float64],
    directional_derivative: wp.array[wp.float64],
    minimum_segment_determinant: wp.array[wp.float64],
):
    if wp.tid() != 0:
        return
    accepted[outer_index] = 0
    initial_objective[outer_index] = wp.float64(0.0)
    candidate_objective[outer_index] = wp.float64(0.0)
    directional_derivative[outer_index] = wp.float64(0.0)
    minimum_segment_determinant[outer_index] = wp.float64(0.0)
    if active[0] == 0:
        reasons[outer_index] = _REASON_MASKED
        return
    if pcg_status[0] != _PCG_COMPLETED or pcg_completed[0] != PCG_ITERATIONS:
        reasons[outer_index] = _REASON_PCG
        active[0] = 0
        return
    start_objective = wp.float64(0.0)
    end_objective = wp.float64(0.0)
    derivative = wp.float64(0.0)
    all_finite = bool(True)
    for vertex in range(current_inertia.shape[0]):
        start_objective += current_inertia[vertex]
        end_objective += candidate_inertia[vertex]
        all_finite = all_finite and vertex_finite[vertex] != 0
    for index in range(directional_terms.shape[0]):
        derivative += directional_terms[index]
    minimum_segment = wp.float64(1.0e300)
    minimum_candidate = wp.float64(1.0e300)
    for tet in range(current_elastic.shape[0]):
        start_objective += current_elastic[tet]
        end_objective += candidate_elastic[tet]
        minimum_segment = wp.min(minimum_segment, segment_minima[tet])
        minimum_candidate = wp.min(minimum_candidate, candidate_determinants[tet])
        all_finite = all_finite and tet_finite[tet] != 0
    initial_objective[outer_index] = start_objective
    candidate_objective[outer_index] = end_objective
    directional_derivative[outer_index] = derivative
    minimum_segment_determinant[outer_index] = minimum_segment
    if (
        not all_finite
        or not wp.isfinite(start_objective)
        or not wp.isfinite(end_objective)
        or not wp.isfinite(derivative)
    ):
        reasons[outer_index] = _REASON_NONFINITE
        active[0] = 0
    elif derivative >= wp.float64(0.0):
        reasons[outer_index] = _REASON_NON_DESCENT
        active[0] = 0
    elif minimum_candidate <= minimum_determinant or minimum_segment <= minimum_determinant:
        reasons[outer_index] = _REASON_SEGMENT_INVERSION
        active[0] = 0
    elif not (end_objective < start_objective and end_objective <= start_objective + armijo * derivative):
        reasons[outer_index] = _REASON_OBJECTIVE
        active[0] = 0
    else:
        accepted[outer_index] = 1
        reasons[outer_index] = _REASON_ACCEPTED


@wp.kernel(enable_backward=False)
def _commit_candidate(
    outer_index: int, candidate: wp.array[wp.vec3d], accepted: wp.array[int], current: wp.array[wp.vec3d]
):
    vertex = wp.tid()
    if accepted[outer_index] != 0:
        current[vertex] = candidate[vertex]


@wp.kernel(enable_backward=False)
def _write_endpoint(
    current: wp.array[wp.vec3d],
    x_current: wp.array[wp.vec3d],
    vertex_to_free: wp.array[int],
    inverse_dt: wp.float64,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    vertex = wp.tid()
    value = current[vertex]
    position = wp.vec3(wp.float32(value[0]), wp.float32(value[1]), wp.float32(value[2]))
    rounded_value = wp.vec3d(
        wp.float64(position[0]),
        wp.float64(position[1]),
        wp.float64(position[2]),
    )
    velocity = (rounded_value - x_current[vertex]) * inverse_dt
    if vertex_to_free[vertex] < 0:
        velocity = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    positions[vertex] = position
    velocities[vertex] = wp.vec3(wp.float32(velocity[0]), wp.float32(velocity[1]), wp.float32(velocity[2]))


@dataclasses.dataclass(frozen=True, eq=False)
class CapturedMGVBDEndpoint:
    """Synchronized endpoint and device-side safeguard evidence."""

    scene_sha256: str
    objective_instance_sha256: str
    static_hierarchy_sha256: str
    config_sha256: str
    armijo: float
    minimum_determinant: float
    positions: np.ndarray
    velocities: np.ndarray
    accepted: tuple[bool, ...]
    reasons: tuple[str, ...]
    initial_objectives: tuple[float, ...]
    candidate_objectives: tuple[float, ...]
    directional_derivatives: tuple[float, ...]
    segment_minimum_determinants: tuple[float, ...]
    outer_start_positions: tuple[np.ndarray, ...]
    outer_candidate_positions: tuple[np.ndarray, ...]
    pcg_statuses: tuple[int, ...]
    pcg_completed_iterations: tuple[int, ...]
    graph_replay: bool
    position_sha256: str = dataclasses.field(init=False)
    velocity_sha256: str = dataclasses.field(init=False)
    outer_start_position_sha256s: tuple[str, ...] = dataclasses.field(init=False)
    outer_candidate_position_sha256s: tuple[str, ...] = dataclasses.field(init=False)
    current_operator_sha256s: tuple[str, ...] = dataclasses.field(init=False)
    endpoint_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        sequence_fields = (
            "accepted",
            "reasons",
            "initial_objectives",
            "candidate_objectives",
            "directional_derivatives",
            "segment_minimum_determinants",
            "outer_start_positions",
            "outer_candidate_positions",
            "pcg_statuses",
            "pcg_completed_iterations",
        )
        if any(type(getattr(self, name)) is not tuple for name in sequence_fields):
            raise ValueError("captured endpoint sequence fields must be exact tuples")
        scene_sha256 = _require_sha256(self.scene_sha256, name="scene_sha256")
        objective_sha256 = _require_sha256(self.objective_instance_sha256, name="objective_instance_sha256")
        hierarchy_sha256 = _require_sha256(self.static_hierarchy_sha256, name="static_hierarchy_sha256")
        config_sha256 = _require_sha256(self.config_sha256, name="config_sha256")
        if type(self.armijo) is not float or not math.isfinite(self.armijo) or not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must be a built-in float in (0, 1)")
        if (
            type(self.minimum_determinant) is not float
            or not math.isfinite(self.minimum_determinant)
            or self.minimum_determinant < 0.0
        ):
            raise ValueError("minimum_determinant must be a non-negative built-in float")
        positions = _immutable_float64(self.positions, name="positions")
        velocities = _immutable_float64(self.velocities, name="velocities")
        if positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape:
            raise ValueError("captured positions and velocities must have matching shape (V, 3)")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        outer_starts = tuple(
            _immutable_float64(value, name=f"outer_start_positions[{index}]")
            for index, value in enumerate(self.outer_start_positions)
        )
        outer_candidates = tuple(
            _immutable_float64(value, name=f"outer_candidate_positions[{index}]")
            for index, value in enumerate(self.outer_candidate_positions)
        )
        if any(value.shape != positions.shape for value in outer_starts):
            raise ValueError("every retained outer start must match the endpoint position shape")
        if any(value.shape != positions.shape for value in outer_candidates):
            raise ValueError("every retained outer candidate must match the endpoint position shape")
        object.__setattr__(self, "outer_start_positions", outer_starts)
        object.__setattr__(self, "outer_candidate_positions", outer_candidates)
        lengths = (
            len(self.accepted),
            len(self.reasons),
            len(self.initial_objectives),
            len(self.candidate_objectives),
            len(self.directional_derivatives),
            len(self.segment_minimum_determinants),
            len(self.outer_start_positions),
            len(self.outer_candidate_positions),
            len(self.pcg_statuses),
            len(self.pcg_completed_iterations),
        )
        if any(length != OUTER_CORRECTIONS for length in lengths):
            raise ValueError("captured MG-VBD endpoint must retain all outer slots")
        if type(self.graph_replay) is not bool:
            raise ValueError("graph_replay must be a bool")
        if any(type(value) is not bool for value in self.accepted):
            raise ValueError("accepted entries must be bools")
        if any(type(value) is not str or value not in REASON_NAMES for value in self.reasons):
            raise ValueError("reason entries must use the fixed reason vocabulary")
        if any(
            accepted != (reason == "accepted") for accepted, reason in zip(self.accepted, self.reasons, strict=True)
        ):
            raise ValueError("accepted flags and reason entries disagree")
        numeric_rows = (
            self.initial_objectives,
            self.candidate_objectives,
            self.directional_derivatives,
            self.segment_minimum_determinants,
        )
        if any(type(value) is not float or not math.isfinite(value) for row in numeric_rows for value in row):
            raise ValueError("outer gate scalar evidence must contain finite built-in floats")
        if any(type(value) is not int for value in self.pcg_statuses + self.pcg_completed_iterations):
            raise ValueError("PCG status and iteration evidence must contain built-in ints")
        for name, array in (
            ("positions", positions),
            ("velocities", velocities),
            *((f"outer_start_positions[{index}]", value) for index, value in enumerate(outer_starts)),
            *((f"outer_candidate_positions[{index}]", value) for index, value in enumerate(outer_candidates)),
        ):
            if not np.array_equal(array, array.astype(np.float32).astype(np.float64)):
                raise ValueError(f"{name} must be exactly representable by the publishable float32 state")

        inactive = False
        for index in range(OUTER_CORRECTIONS):
            accepted = self.accepted[index]
            reason = self.reasons[index]
            if inactive:
                if accepted or reason != "masked-after-rejection":
                    raise ValueError("outer evidence after the first rejection must be masked")
                if any(
                    value != 0.0
                    for value in (
                        self.initial_objectives[index],
                        self.candidate_objectives[index],
                        self.directional_derivatives[index],
                        self.segment_minimum_determinants[index],
                    )
                ):
                    raise ValueError("masked outer evidence must contain zero gate scalars")
                continue
            if accepted:
                if self.pcg_statuses[index] != _PCG_COMPLETED or self.pcg_completed_iterations[index] != PCG_ITERATIONS:
                    raise ValueError("accepted outer evidence must bind completed PCG4 work")
                initial = self.initial_objectives[index]
                candidate = self.candidate_objectives[index]
                derivative = self.directional_derivatives[index]
                if not (
                    derivative < 0.0
                    and candidate < initial
                    and candidate <= initial + self.armijo * derivative
                    and self.segment_minimum_determinants[index] > self.minimum_determinant
                ):
                    raise ValueError("accepted outer evidence violates its bound numerical gate")
                committed = outer_starts[index + 1] if index + 1 < OUTER_CORRECTIONS else positions
                if not np.array_equal(outer_candidates[index], committed):
                    raise ValueError("accepted outer candidate does not match the next committed state")
            else:
                if reason == "masked-after-rejection":
                    raise ValueError("the first rejected outer cannot already be masked")
                if not np.array_equal(positions, outer_starts[index]):
                    raise ValueError("a rejected correction must preserve its exact input state")
                inactive = True

        position_sha256 = _array_digest(positions)
        velocity_sha256 = _array_digest(velocities)
        start_sha256s = tuple(_array_digest(value) for value in outer_starts)
        candidate_sha256s = tuple(_array_digest(value) for value in outer_candidates)
        operator_sha256s = tuple(
            _canonical_digest(
                {
                    "contract": "captured-mg-vbd-current-operator-v1",
                    "objective_instance_sha256": objective_sha256,
                    "position_sha256": start_sha256,
                }
            )
            for start_sha256 in start_sha256s
        )
        object.__setattr__(self, "position_sha256", position_sha256)
        object.__setattr__(self, "velocity_sha256", velocity_sha256)
        object.__setattr__(self, "outer_start_position_sha256s", start_sha256s)
        object.__setattr__(self, "outer_candidate_position_sha256s", candidate_sha256s)
        object.__setattr__(self, "current_operator_sha256s", operator_sha256s)
        object.__setattr__(
            self,
            "endpoint_sha256",
            _canonical_digest(
                {
                    "contract": CONTRACT_ID,
                    "scene_sha256": scene_sha256,
                    "objective_instance_sha256": objective_sha256,
                    "static_hierarchy_sha256": hierarchy_sha256,
                    "config_sha256": config_sha256,
                    "armijo": self.armijo,
                    "minimum_determinant": self.minimum_determinant,
                    "position_sha256": position_sha256,
                    "velocity_sha256": velocity_sha256,
                    "accepted": list(self.accepted),
                    "reasons": list(self.reasons),
                    "initial_objectives": list(self.initial_objectives),
                    "candidate_objectives": list(self.candidate_objectives),
                    "directional_derivatives": list(self.directional_derivatives),
                    "segment_minimum_determinants": list(self.segment_minimum_determinants),
                    "outer_start_position_sha256s": list(start_sha256s),
                    "outer_candidate_position_sha256s": list(candidate_sha256s),
                    "current_operator_sha256s": list(operator_sha256s),
                    "pcg_statuses": list(self.pcg_statuses),
                    "pcg_completed_iterations": list(self.pcg_completed_iterations),
                    "graph_replay": self.graph_replay,
                }
            ),
        )

    def deterministic_record(self) -> dict[str, object]:
        """Serialize content identities and retained gate evidence."""
        return {
            "contract": CONTRACT_ID,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "static_hierarchy_sha256": self.static_hierarchy_sha256,
            "config_sha256": self.config_sha256,
            "armijo": self.armijo,
            "minimum_determinant": self.minimum_determinant,
            "position_sha256": self.position_sha256,
            "velocity_sha256": self.velocity_sha256,
            "accepted": list(self.accepted),
            "reasons": list(self.reasons),
            "initial_objectives": list(self.initial_objectives),
            "candidate_objectives": list(self.candidate_objectives),
            "directional_derivatives": list(self.directional_derivatives),
            "segment_minimum_determinants": list(self.segment_minimum_determinants),
            "outer_start_position_sha256s": list(self.outer_start_position_sha256s),
            "outer_candidate_position_sha256s": list(self.outer_candidate_position_sha256s),
            "current_operator_sha256s": list(self.current_operator_sha256s),
            "pcg_statuses": list(self.pcg_statuses),
            "pcg_completed_iterations": list(self.pcg_completed_iterations),
            "graph_replay": self.graph_replay,
            "endpoint_sha256": self.endpoint_sha256,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class CapturedMGVBDTiming:
    """Paired CUDA-event timings for integrated MG-VBD versus pristine K4."""

    pair_orders: tuple[str, ...]
    mg_seconds: tuple[float, ...]
    k4_seconds: tuple[float, ...]
    warmup_replays: int
    random_seed: int
    device: str
    setup_included: bool = False
    transfers_included: bool = False
    integrated_mg: bool = True
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_orders", tuple(self.pair_orders))
        object.__setattr__(self, "mg_seconds", tuple(self.mg_seconds))
        object.__setattr__(self, "k4_seconds", tuple(self.k4_seconds))
        count = len(self.pair_orders)
        if count < 2 or count % 2 != 0 or len(self.mg_seconds) != count or len(self.k4_seconds) != count:
            raise ValueError("paired timing arrays must have the same positive even length")
        if any(order not in ("AB", "BA") for order in self.pair_orders):
            raise ValueError("pair orders must use AB/BA labels")
        if self.pair_orders.count("AB") != self.pair_orders.count("BA"):
            raise ValueError("paired timing must contain equal AB and BA counts")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.mg_seconds + self.k4_seconds):
            raise ValueError("CUDA-event timings must be finite and positive")
        if type(self.warmup_replays) is not int or self.warmup_replays < 1:
            raise ValueError("warmup_replays must be a positive built-in int")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be a built-in int")
        if type(self.device) is not str or not self.device:
            raise ValueError("device must be a non-empty built-in string")
        if any(
            type(value) is not bool
            for value in (self.setup_included, self.transfers_included, self.integrated_mg, self.performance_evidence)
        ):
            raise ValueError("timing policy fields must be built-in bools")
        if self.setup_included or self.transfers_included or not self.integrated_mg or self.performance_evidence:
            raise ValueError("integrated timing must exclude setup/transfers and remain diagnostic-only")

    @property
    def mg_median_seconds(self) -> float:
        """Median captured integrated MG-VBD time [s]."""
        return statistics.median(self.mg_seconds)

    @property
    def k4_median_seconds(self) -> float:
        """Median captured pristine K4 time [s]."""
        return statistics.median(self.k4_seconds)

    def deterministic_record(self) -> dict[str, object]:
        """Serialize the paired diagnostic timing result."""
        return {
            "pair_orders": list(self.pair_orders),
            "mg_seconds": list(self.mg_seconds),
            "k4_seconds": list(self.k4_seconds),
            "warmup_replays": self.warmup_replays,
            "random_seed": self.random_seed,
            "device": self.device,
            "setup_included": self.setup_included,
            "transfers_included": self.transfers_included,
            "integrated_mg": self.integrated_mg,
            "performance_evidence": self.performance_evidence,
        }


class CapturedMultiplicativeMGVBD:
    """Persistent captured K1 + three-current-A/static-A0-MG-PCG4 lane."""

    def __init__(
        self,
        scene: TetBenchmarkScene,
        *,
        device: str = "cuda:0",
        config: MGVBDCorrectionConfig | None = None,
        tile_solve: bool = False,
    ):
        if type(scene) is not TetBenchmarkScene:
            raise TypeError("scene must be an exact TetBenchmarkScene")
        self.config = MGVBDCorrectionConfig() if config is None else config
        if type(self.config) is not MGVBDCorrectionConfig:
            raise TypeError("config must be an exact MGVBDCorrectionConfig")
        self.config.validate()
        self.config_sha256 = _canonical_digest(self.config.deterministic_record())
        if self.config.outer_corrections != OUTER_CORRECTIONS:
            raise ValueError("captured composition requires exactly three outer corrections")
        if self.config.correction.pcg_iterations != PCG_ITERATIONS or self.config.correction.alpha != 1.0:
            raise ValueError("captured composition requires fixed-alpha=1 PCG4")
        self.scene = scene
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError("the first fully captured MG-VBD composition requires CUDA")
        self.baseline = CapturedPublicVBDBaseline(scene, device=str(self.device), tile_solve=tile_solve)

        # One uncaptured deterministic K1 endpoint supplies construction-time
        # geometry only. Every actual run starts from the graph's pristine K1.
        k1 = self.baseline.run(1, graph_replay=False)
        self.problem = build_common_problem(scene)
        self.scene_sha256 = _require_sha256(scene.manifest()["scene_sha256"], name="scene_sha256")
        self.objective_instance_sha256 = _require_sha256(
            common_objective_manifest(scene, self.problem)["objective_instance_sha256"],
            name="objective_instance_sha256",
        )
        oracle = MatrixFreeStableNHOperator.from_problem(self.problem, k1.positions)
        hierarchy = build_stable_nh_rest_multigrid(
            oracle,
            scene.rest_q,
            mode_kind=self.config.mode_kind,
            target_aggregate_size=self.config.target_aggregate_size,
            minimum_aggregate_size=self.config.minimum_aggregate_size,
            coarse_node_limit=self.config.coarse_node_limit,
            maximum_levels=self.config.maximum_levels,
            pre_smooth_steps=self.config.pre_smooth_steps,
            post_smooth_steps=self.config.post_smooth_steps,
            smoother_safety=self.config.smoother_safety,
        )
        self.operator = WarpMatrixFreeStableNHOperator.from_oracle(oracle, device=str(self.device))
        self.device_hierarchy = WarpStaticMultigridHierarchy.from_hierarchy(hierarchy, device=str(self.device))
        self.preconditioner = WarpStaticMultigridPreconditioner(self.device_hierarchy)
        self.pcg = tuple(
            WarpFixedPCGWorkspace(self.operator, PCG_ITERATIONS, device_preconditioner=self.preconditioner)
            for _ in range(OUTER_CORRECTIONS)
        )
        n_vertices = scene.n_vertices
        n_tets = scene.n_tets
        n_free = int(oracle.free.size)
        self.canonical_positions = wp.array(oracle.positions, dtype=wp.vec3d, device=self.device)
        self.x_current = wp.array(scene.x_current, dtype=wp.vec3d, device=self.device)
        self.candidate = wp.empty(n_vertices, dtype=wp.vec3d, device=self.device)
        self.final_positions = wp.empty(n_vertices, dtype=wp.vec3, device=self.device)
        self.final_velocities = wp.empty(n_vertices, dtype=wp.vec3, device=self.device)
        self.active = wp.empty(1, dtype=wp.int32, device=self.device)
        self.accepted = wp.empty(OUTER_CORRECTIONS, dtype=wp.int32, device=self.device)
        self.reasons = wp.empty(OUTER_CORRECTIONS, dtype=wp.int32, device=self.device)
        self.current_inertia = wp.empty(n_vertices, dtype=wp.float64, device=self.device)
        self.candidate_inertia = wp.empty(n_vertices, dtype=wp.float64, device=self.device)
        self.vertex_finite = wp.empty(n_vertices, dtype=wp.int32, device=self.device)
        self.current_elastic = wp.empty(n_tets, dtype=wp.float64, device=self.device)
        self.candidate_elastic = wp.empty(n_tets, dtype=wp.float64, device=self.device)
        self.candidate_determinants = wp.empty(n_tets, dtype=wp.float64, device=self.device)
        self.segment_minima = wp.empty(n_tets, dtype=wp.float64, device=self.device)
        self.tet_finite = wp.empty(n_tets, dtype=wp.int32, device=self.device)
        self.directional_terms = wp.empty(n_free, dtype=wp.float64, device=self.device)
        self.outer_start_positions = tuple(
            wp.empty(n_vertices, dtype=wp.vec3d, device=self.device) for _ in range(OUTER_CORRECTIONS)
        )
        self.outer_candidate_positions = tuple(
            wp.empty(n_vertices, dtype=wp.vec3d, device=self.device) for _ in range(OUTER_CORRECTIONS)
        )
        self.initial_objectives = wp.empty(OUTER_CORRECTIONS, dtype=wp.float64, device=self.device)
        self.candidate_objectives = wp.empty(OUTER_CORRECTIONS, dtype=wp.float64, device=self.device)
        self.directional_derivatives = wp.empty(OUTER_CORRECTIONS, dtype=wp.float64, device=self.device)
        self.minimum_segment_determinants = wp.empty(OUTER_CORRECTIONS, dtype=wp.float64, device=self.device)
        self.graph: object | None = None
        self.k4_graph: object | None = None

    def _enqueue_integrated(self) -> None:
        lane = self.baseline._lane(1)
        self.baseline._enqueue_reset_and_step(lane)
        wp.launch(
            _initialize_from_k1,
            dim=self.scene.n_vertices,
            inputs=[
                lane.state_out.particle_q,
                self.canonical_positions,
                self.operator.vertex_to_free,
                self.operator.positions,
                self.candidate,
                self.active,
                self.accepted,
                self.reasons,
            ],
            device=self.device,
        )
        for outer_index, pcg in enumerate(self.pcg):
            wp.launch(
                _copy_positions,
                dim=self.scene.n_vertices,
                inputs=[self.operator.positions, self.outer_start_positions[outer_index]],
                device=self.device,
            )
            self.operator.launch_refresh_geometry()
            self.operator.launch_gradient(pcg.rhs, scale=-1.0)
            wp.launch(_mask_rhs, dim=self.operator.n_free, inputs=[self.active, pcg.rhs], device=self.device)
            pcg.launch()
            wp.launch(
                _build_candidate,
                dim=self.scene.n_vertices,
                inputs=[
                    self.operator.positions,
                    self.operator.vertex_to_free,
                    pcg.solution,
                    pcg.state_status,
                    pcg.completed_iterations,
                    self.active,
                    self.candidate,
                ],
                device=self.device,
            )
            wp.launch(
                _copy_positions,
                dim=self.scene.n_vertices,
                inputs=[self.candidate, self.outer_candidate_positions[outer_index]],
                device=self.device,
            )
            wp.launch(
                _vertex_gate_terms,
                dim=self.scene.n_vertices,
                inputs=[
                    self.operator.positions,
                    self.candidate,
                    self.operator.inertial_target,
                    self.operator.mass,
                    self.operator.inverse_dt_squared,
                    self.current_inertia,
                    self.candidate_inertia,
                    self.vertex_finite,
                ],
                device=self.device,
            )
            wp.launch(
                _directional_terms,
                dim=self.operator.n_free,
                inputs=[
                    pcg.rhs,
                    self.operator.free,
                    self.operator.positions,
                    self.candidate,
                    self.directional_terms,
                ],
                device=self.device,
            )
            wp.launch(
                _tet_gate_terms,
                dim=self.scene.n_tets,
                inputs=[
                    self.operator.deformation_gradients,
                    self.operator.cofactors,
                    self.operator.determinants,
                    self.candidate,
                    self.operator.tets,
                    self.operator.shape_gradients,
                    self.operator.volumes,
                    self.operator.mu,
                    self.operator.lam,
                    self.current_elastic,
                    self.candidate_elastic,
                    self.candidate_determinants,
                    self.segment_minima,
                    self.tet_finite,
                ],
                device=self.device,
            )
            wp.launch(
                _finalize_gate,
                dim=1,
                inputs=[
                    outer_index,
                    pcg.state_status,
                    pcg.completed_iterations,
                    self.current_inertia,
                    self.candidate_inertia,
                    self.current_elastic,
                    self.candidate_elastic,
                    self.directional_terms,
                    self.candidate_determinants,
                    self.segment_minima,
                    self.vertex_finite,
                    self.tet_finite,
                    self.config.correction.minimum_determinant,
                    self.config.correction.armijo,
                    self.active,
                    self.accepted,
                    self.reasons,
                    self.initial_objectives,
                    self.candidate_objectives,
                    self.directional_derivatives,
                    self.minimum_segment_determinants,
                ],
                device=self.device,
            )
            wp.launch(
                _commit_candidate,
                dim=self.scene.n_vertices,
                inputs=[outer_index, self.candidate, self.accepted, self.operator.positions],
                device=self.device,
            )
        wp.launch(
            _write_endpoint,
            dim=self.scene.n_vertices,
            inputs=[
                self.operator.positions,
                self.x_current,
                self.operator.vertex_to_free,
                1.0 / self.scene.dt,
                self.final_positions,
                self.final_velocities,
            ],
            device=self.device,
        )

    def capture_graphs(self, *, warmup_replays: int = 1) -> None:
        """Capture separate integrated MG-VBD and pristine K4 graphs."""
        if isinstance(warmup_replays, bool) or not isinstance(warmup_replays, numbers.Integral) or warmup_replays < 1:
            raise ValueError("warmup_replays must be a positive integer")
        for _ in range(int(warmup_replays)):
            self._enqueue_integrated()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self._enqueue_integrated()
        self.graph = capture.graph
        k4_lane = self.baseline._lane(4)
        for _ in range(int(warmup_replays)):
            self.baseline._enqueue_reset_and_step(k4_lane)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self.baseline._enqueue_reset_and_step(k4_lane)
        self.k4_graph = capture.graph

    def run(self, *, graph_replay: bool = True) -> CapturedMGVBDEndpoint:
        """Execute and synchronize one integrated lane."""
        if graph_replay:
            if self.graph is None:
                raise RuntimeError("capture_graphs() must complete before graph replay")
            wp.capture_launch(self.graph)
        else:
            self._enqueue_integrated()
        return self.record(graph_replay=graph_replay)

    def run_k4(self, *, graph_replay: bool = True):
        """Execute the separate pristine K4 comparator lane."""
        lane = self.baseline._lane(4)
        if graph_replay:
            if self.k4_graph is None:
                raise RuntimeError("capture_graphs() must complete before graph replay")
            wp.capture_launch(self.k4_graph)
        else:
            self.baseline._enqueue_reset_and_step(lane)
        lane.completed_launches += 1
        return self.baseline.record(4, graph_replay=graph_replay)

    def benchmark_paired(
        self,
        *,
        pair_count: int = 10,
        warmup_replays: int = 5,
        random_seed: int = 20260817,
    ) -> CapturedMGVBDTiming:
        """Measure captured integrated MG-VBD and pristine K4 in balanced AB/BA order."""
        if self.graph is None or self.k4_graph is None:
            raise RuntimeError("capture_graphs() must complete before timing")
        for name, value, minimum in (("pair_count", pair_count, 2), ("warmup_replays", warmup_replays, 1)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if int(pair_count) % 2 != 0:
            raise ValueError("pair_count must be even for balanced AB/BA timing")
        if isinstance(random_seed, bool) or not isinstance(random_seed, numbers.Integral):
            raise ValueError("random_seed must be an integer")

        for _ in range(int(warmup_replays)):
            wp.capture_launch(self.graph)
            wp.capture_launch(self.k4_graph)
        wp.synchronize_device(self.device)

        orders = ["AB" if index % 2 == 0 else "BA" for index in range(int(pair_count))]
        np.random.default_rng(int(random_seed)).shuffle(orders)
        mg_events = [
            (wp.Event(self.device, enable_timing=True), wp.Event(self.device, enable_timing=True)) for _ in orders
        ]
        k4_events = [
            (wp.Event(self.device, enable_timing=True), wp.Event(self.device, enable_timing=True)) for _ in orders
        ]
        for pair_index, order in enumerate(orders):
            if order == "AB":
                begin, end = mg_events[pair_index]
                wp.record_event(begin)
                wp.capture_launch(self.graph)
                wp.record_event(end)
                begin, end = k4_events[pair_index]
                wp.record_event(begin)
                wp.capture_launch(self.k4_graph)
                wp.record_event(end)
            else:
                begin, end = k4_events[pair_index]
                wp.record_event(begin)
                wp.capture_launch(self.k4_graph)
                wp.record_event(end)
                begin, end = mg_events[pair_index]
                wp.record_event(begin)
                wp.capture_launch(self.graph)
                wp.record_event(end)
        wp.synchronize_event(mg_events[-1][1])
        wp.synchronize_event(k4_events[-1][1])
        return CapturedMGVBDTiming(
            pair_orders=tuple(orders),
            mg_seconds=tuple(
                float(wp.get_event_elapsed_time(begin, end, synchronize=False)) * 1.0e-3 for begin, end in mg_events
            ),
            k4_seconds=tuple(
                float(wp.get_event_elapsed_time(begin, end, synchronize=False)) * 1.0e-3 for begin, end in k4_events
            ),
            warmup_replays=int(warmup_replays),
            random_seed=int(random_seed),
            device=str(self.device),
        )

    def record(self, *, graph_replay: bool) -> CapturedMGVBDEndpoint:
        """Synchronously materialize endpoint and gate evidence."""
        positions = np.asarray(self.final_positions.numpy(), dtype=np.float32).astype(np.float64)
        velocities = np.asarray(self.final_velocities.numpy(), dtype=np.float32).astype(np.float64)
        accepted = tuple(bool(value) for value in self.accepted.numpy())
        reason_codes = tuple(int(value) for value in self.reasons.numpy())
        if any(code < 0 or code >= len(REASON_NAMES) for code in reason_codes):
            raise RuntimeError("device returned an invalid MG-VBD gate reason")
        return CapturedMGVBDEndpoint(
            scene_sha256=self.scene_sha256,
            objective_instance_sha256=self.objective_instance_sha256,
            static_hierarchy_sha256=self.device_hierarchy.hierarchy_sha256,
            config_sha256=self.config_sha256,
            armijo=float(self.config.correction.armijo),
            minimum_determinant=float(self.config.correction.minimum_determinant),
            positions=positions,
            velocities=velocities,
            accepted=accepted,
            reasons=tuple(REASON_NAMES[code] for code in reason_codes),
            initial_objectives=tuple(float(value) for value in self.initial_objectives.numpy()),
            candidate_objectives=tuple(float(value) for value in self.candidate_objectives.numpy()),
            directional_derivatives=tuple(float(value) for value in self.directional_derivatives.numpy()),
            segment_minimum_determinants=tuple(float(value) for value in self.minimum_segment_determinants.numpy()),
            outer_start_positions=tuple(
                np.asarray(value.numpy(), dtype=np.float64) for value in self.outer_start_positions
            ),
            outer_candidate_positions=tuple(
                np.asarray(value.numpy(), dtype=np.float64) for value in self.outer_candidate_positions
            ),
            pcg_statuses=tuple(int(pcg.state_status.numpy()[0]) for pcg in self.pcg),
            pcg_completed_iterations=tuple(int(pcg.completed_iterations.numpy()[0]) for pcg in self.pcg),
            graph_replay=graph_replay,
        )

    def poison(self, *, seed: int) -> None:
        """Poison public endpoint and visible correction state before replay."""
        rng = np.random.default_rng(seed)
        self.baseline.poison_lane(1, seed=seed + 1)
        self.operator.positions.assign(rng.normal(size=(self.scene.n_vertices, 3)))
        self.candidate.assign(rng.normal(size=(self.scene.n_vertices, 3)))
        self.final_positions.assign(rng.normal(size=(self.scene.n_vertices, 3)).astype(np.float32))
        self.final_velocities.assign(rng.normal(size=(self.scene.n_vertices, 3)).astype(np.float32))
        for value in self.outer_start_positions:
            value.assign(rng.normal(size=(self.scene.n_vertices, 3)))
        for value in self.outer_candidate_positions:
            value.assign(rng.normal(size=(self.scene.n_vertices, 3)))
        self.active.assign(np.array([0], dtype=np.int32))
        self.accepted.assign(np.full(OUTER_CORRECTIONS, -1, dtype=np.int32))
        self.reasons.assign(np.full(OUTER_CORRECTIONS, -1, dtype=np.int32))
        self.initial_objectives.assign(np.full(OUTER_CORRECTIONS, np.nan, dtype=np.float64))
        self.candidate_objectives.assign(np.full(OUTER_CORRECTIONS, np.nan, dtype=np.float64))
        self.directional_derivatives.assign(np.full(OUTER_CORRECTIONS, np.nan, dtype=np.float64))
        self.minimum_segment_determinants.assign(np.full(OUTER_CORRECTIONS, np.nan, dtype=np.float64))

    def deterministic_record(self) -> dict[str, object]:
        """Return the fixed schedule and scope without timing data."""
        return {
            "contract_id": CONTRACT_ID,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "config": self.config.deterministic_record(),
            "config_sha256": self.config_sha256,
            "device": str(self.device),
            "vbd_iterations": 1,
            "outer_corrections": OUTER_CORRECTIONS,
            "pcg_iterations_per_outer": PCG_ITERATIONS,
            "v_cycles": OUTER_CORRECTIONS * PCG_ITERATIONS,
            "current_operator_refreshes": OUTER_CORRECTIONS,
            "static_hierarchy_sha256": self.device_hierarchy.hierarchy_sha256,
            "alpha": self.config.correction.alpha,
            "armijo": self.config.correction.armijo,
            "minimum_determinant": self.config.correction.minimum_determinant,
            "gate_execution": "device-side-strict-armijo-exact-cubic-segment-fail-closed",
            "rejection_mask": "sticky-after-first-rejection",
            "final_velocity": "BDF1-from-physical-x-current-exact-zero-pins",
            "separate_k4_graph": True,
            "performance_evidence": False,
        }
