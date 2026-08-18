# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Captured direct multiplicative-graph VBD research composition.

This CUDA-only harness starts from a pristine public ``SolverVBD`` K1
substep and schedules four nonlinear corrections.  At every outer state it
refreshes the current stable-Neo-Hookean Gauss--Newton operator ``A(x)`` and
applies one immutable spectral-free rest hierarchy ``B`` through

``d = B b + B (b - A(x) B b)``, where ``b = -gradient(x)``.

There is no PCG or other Krylov iteration in this path.  Each proposed state
is rounded to the float32 state that Newton would publish before its actual
step, objective, strict Armijo condition, and exact cubic segment determinant
are evaluated.  The first rejected proposal disables every later proposal,
while all fixed graph work still executes.  This module is deliberately a
contact-free research harness and its paired CUDA timing is diagnostic only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import statistics
from collections.abc import Iterable
from typing import Any

import numpy as np
import warp as wp

from newton.solvers import SolverVBD

from .captured_mg_vbd import (
    _commit_candidate,
    _copy_positions,
    _directional_terms,
    _tet_gate_terms,
    _vertex_gate_terms,
    _write_endpoint,
)
from .captured_vbd_baseline import (
    CONTRACT_ID as VBD_BASELINE_CONTRACT_ID,
)
from .captured_vbd_baseline import (
    CapturedPublicVBDBaseline,
    CapturedVBDEndpoint,
    _public_model_sha256,
)
from .captured_vbd_baseline import (
    _array_sha256 as _vbd_array_sha256,
)
from .captured_vbd_baseline import (
    _minimum_determinant as _vbd_minimum_determinant,
)
from .captured_vbd_baseline import (
    _named_arrays_sha256 as _vbd_named_arrays_sha256,
)
from .correction_gpu import MatrixFreeStableNHOperator, minimum_determinant_on_segment
from .correction_gpu_warp import WarpMatrixFreeStableNHOperator, WarpMatrixFreeWorkspace
from .correction_graph_vbd import (
    DirectGraphVBDConfig,
    _canonical_static_hierarchy,
    _operator_sha256,
)
from .correction_multigrid import (
    StaticMultigridHierarchy,
    VCycleWorkRecord,
    apply_v_cycle,
    build_stable_nh_rest_multigrid,
)
from .correction_multigrid_warp import (
    CONTRACT_ID as V_CYCLE_CONTRACT_ID,
)
from .correction_multigrid_warp import (
    KERNEL_VERSION as V_CYCLE_KERNEL_VERSION,
)
from .correction_multigrid_warp import (
    MAX_COARSE_SCALAR_SIZE,
    WarpStaticMultigridHierarchy,
    WarpVCycleRecord,
    WarpVCycleWorkspace,
)
from .solver_benchmark import TetBenchmarkScene, build_common_problem, common_objective_manifest

CONTRACT_ID = "captured-direct-multiplicative-graph-vbd-v1"
OUTER_CORRECTIONS = 4
V_CYCLES_PER_OUTER = 2

_REASON_PENDING = 0
_REASON_ACCEPTED = 1
_REASON_MASKED = 2
_REASON_NONFINITE = 3
_REASON_NON_DESCENT = 4
_REASON_SEGMENT_INVERSION = 5
_REASON_OBJECTIVE = 6

_SOLVER_SCRATCH_ARRAY_NAMES = frozenset(
    {
        "body_inv_inertia_effective",
        "body_inv_mass_effective",
        "inertia",
        "particle_displacements",
        "particle_forces",
        "particle_hessians",
        "particle_q_prev",
        "pos_prev_collision_detection",
        "truncation_ts",
    }
)

_PARTICLE_ADJACENCY_ARRAY_NAMES = (
    "v_adj_faces",
    "v_adj_faces_offsets",
    "v_adj_edges",
    "v_adj_edges_offsets",
    "v_adj_springs",
    "v_adj_springs_offsets",
    "v_adj_tets",
    "v_adj_tets_offsets",
)

_PARTICLE_SCRATCH_ARRAY_SPECS = (
    ("particle_q_prev", wp.vec3),
    ("inertia", wp.vec3),
    ("particle_displacements", wp.vec3),
    ("pos_prev_collision_detection", wp.vec3),
    ("truncation_ts", wp.float32),
    ("particle_forces", wp.vec3),
    ("particle_hessians", wp.mat33),
)

_ARRAY_DESCRIPTOR_FIELD_NAMES = ("data", "grad", "shape", "strides", "ndim")

REASON_NAMES = (
    "pending",
    "accepted",
    "masked-after-rejection",
    "candidate-nonfinite",
    "non-descent",
    "segment-inversion",
    "objective-increase",
)

_VALIDATION_TOKEN = object()


def _immutable_float64(value: np.ndarray, *, name: str) -> np.ndarray:
    """Copy a finite float64 array into immutable bytes-backed storage."""
    owned = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(owned).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _immutable_int64(value: np.ndarray, *, name: str) -> np.ndarray:
    """Copy a one-dimensional index array into immutable bytes-backed storage."""
    owned = np.array(value, dtype=np.int64, order="C", copy=True)
    if owned.ndim != 1 or (owned.size and np.any(owned < 0)):
        raise ValueError(f"{name} must be a one-dimensional non-negative index array")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype)


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


def _hash_parts(tag: str, parts: Iterable[tuple[str, object]]) -> str:
    """Reproduce the length-delimited device V-cycle vector hash schema."""
    digest = hashlib.sha256()

    def add_bytes(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    add_bytes(tag.encode("utf-8"))
    for name, value in parts:
        add_bytes(name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            add_bytes(b"array")
            add_bytes(value.dtype.str.encode("ascii"))
            add_bytes(repr(value.shape).encode("ascii"))
            add_bytes(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, bool):
            add_bytes(b"bool")
            add_bytes(b"1" if value else b"0")
        elif isinstance(value, int):
            add_bytes(b"int")
            add_bytes(str(value).encode("ascii"))
        elif isinstance(value, str):
            add_bytes(b"str")
            add_bytes(value.encode("utf-8"))
        else:
            raise TypeError(f"unsupported hash part {name!r}: {type(value).__name__}")
    return digest.hexdigest()


def _array_container_items(value: object, *, prefix: str) -> tuple[tuple[str, wp.array[Any]], ...]:
    """Recursively enumerate Warp arrays in built-in attribute containers."""
    if isinstance(value, wp.array):
        return ((prefix, value),)
    if isinstance(value, dict):
        return tuple(
            item
            for key in sorted(value, key=repr)
            for item in _array_container_items(value[key], prefix=f"{prefix}[{key!r}]")
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            item
            for index, child in enumerate(value)
            for item in _array_container_items(child, prefix=f"{prefix}[{index}]")
        )
    return ()


def _attribute_array_items(value: object, *, prefix: str) -> tuple[tuple[str, wp.array[Any]], ...]:
    """Enumerate every direct/container Warp array attribute of one public object."""
    try:
        attributes = vars(value)
    except TypeError:
        return ()
    return tuple(
        item
        for name, child in sorted(attributes.items())
        for item in _array_container_items(child, prefix=f"{prefix}.{name}")
    )


def _array_pointer(value: wp.array[Any]) -> int:
    """Return a canonical pointer value, including Warp's null empty arrays."""
    return 0 if value.ptr is None else int(value.ptr)


def _validate_descriptor_matches_array(
    descriptor: object,
    value: wp.array[Any],
    *,
    name: str,
    expected_type: type,
    expected_fields: tuple[tuple[str, type], ...],
) -> tuple[object, ...]:
    """Require one cached Warp C descriptor to exactly describe its array."""
    fields = getattr(type(descriptor), "_fields_", ())
    if (
        type(descriptor) is not expected_type
        or fields != expected_fields
        or tuple(field[0] for field in fields) != _ARRAY_DESCRIPTOR_FIELD_NAMES
    ):
        raise RuntimeError(f"{name} cached C descriptor type or field layout changed")
    if isinstance(value.ndim, bool) or not isinstance(value.ndim, numbers.Integral) or not 0 <= value.ndim <= 4:
        raise RuntimeError(f"{name} has an unsupported array rank")
    expected_shape = tuple(int(component) for component in value.shape) + (0,) * (4 - value.ndim)
    expected_strides = tuple(int(component) for component in value.strides) + (0,) * (4 - value.ndim)
    gradient = getattr(value, "grad", None)
    expected = (
        _array_pointer(value),
        0 if gradient is None else _array_pointer(gradient),
        int(value.ndim),
        expected_shape,
        expected_strides,
    )
    try:
        actual = (
            int(descriptor.data),
            int(descriptor.grad),
            int(descriptor.ndim),
            tuple(int(descriptor.shape[index]) for index in range(4)),
            tuple(int(descriptor.strides[index]) for index in range(4)),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"{name} cached C descriptor is malformed") from error
    if actual != expected:
        raise RuntimeError(f"{name} cached C descriptor does not match its array")
    return actual


def _canonical_array_descriptor(
    value: wp.array[Any],
    *,
    name: str,
    expected_type: type,
    expected_fields: tuple[tuple[str, type], ...],
) -> tuple[object, ...]:
    """Validate and record the C descriptor used for one direct array argument."""
    descriptor = value.__ctype__()
    if descriptor is not value.ctype:
        raise RuntimeError(f"{name} cached C descriptor object changed")
    values = _validate_descriptor_matches_array(
        descriptor,
        value,
        name=name,
        expected_type=expected_type,
        expected_fields=expected_fields,
    )
    return (id(descriptor), type(descriptor), type(descriptor)._fields_, values)


def _v_cycle_rhs_sha256(value: np.ndarray) -> str:
    """Hash one flattened V-cycle RHS using its registered work schema."""
    column = np.asarray(value, dtype=np.float64).reshape(-1, 1)
    return _hash_parts("v-cycle-rhs-v1", (("rhs", column),))


def _v_cycle_result_sha256(value: np.ndarray) -> str:
    """Hash one flattened V-cycle output using its registered work schema."""
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return _hash_parts("v-cycle-correction-v1", (("correction", flat),))


def _require_exact_array(actual: np.ndarray, expected: np.ndarray, *, name: str) -> None:
    """Require identical dtype, shape, and values for persistent inputs."""
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
        raise RuntimeError(f"persistent {name} changed after captured direct graph construction")


def _require_close_array(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
    relative_tolerance: float = 8.0e-12,
    absolute_tolerance: float = 8.0e-13,
) -> None:
    """Require the independently replayed CPU/GPU result within a tight bound."""
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.allclose(
        left,
        right,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        raise ValueError(f"{name} does not match the canonical CPU replay")


def _require_close_scalar(
    actual: float,
    expected: float,
    *,
    name: str,
    relative_tolerance: float = 8.0e-12,
    absolute_tolerance: float = 8.0e-13,
) -> None:
    """Require one finite independently replayed scalar within a tight bound."""
    if not math.isclose(actual, expected, rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
        raise ValueError(f"{name} does not match the canonical CPU replay")


def _validate_public_vbd_endpoint(
    endpoint: CapturedVBDEndpoint,
    reference: CapturedVBDEndpoint,
    scene: TetBenchmarkScene,
    *,
    iterations: int,
    label: str,
    device: str,
    graph_replay: bool,
) -> None:
    """Validate one complete public VBD endpoint against a fresh-bound oracle."""
    if type(endpoint) is not CapturedVBDEndpoint or type(reference) is not CapturedVBDEndpoint:
        raise TypeError(f"{label} validation requires exact CapturedVBDEndpoint values")
    if endpoint.contract_id != VBD_BASELINE_CONTRACT_ID or endpoint.iterations != iterations:
        raise ValueError(f"captured direct graph {label} provenance is invalid")
    if endpoint.device != device or endpoint.graph_replay != graph_replay:
        raise ValueError(f"captured direct graph {label} execution provenance is invalid")
    if endpoint.pristine_state_sha256 != reference.pristine_state_sha256:
        raise ValueError(f"captured direct graph {label} pristine-state identity changed")
    if not np.array_equal(endpoint.positions, reference.positions) or not np.array_equal(
        endpoint.velocities,
        reference.velocities,
    ):
        raise ValueError(f"captured direct graph {label} differs from its fresh construction reference")
    positions_fp32 = endpoint.positions.astype(np.float32)
    for actual, expected, name in (
        (endpoint.position_sha256, _vbd_array_sha256(endpoint.positions), f"{label} position hash"),
        (endpoint.position_fp32_sha256, _vbd_array_sha256(positions_fp32), f"{label} float32 position hash"),
        (endpoint.velocity_sha256, _vbd_array_sha256(endpoint.velocities), f"{label} velocity hash"),
        (
            endpoint.endpoint_sha256,
            _vbd_named_arrays_sha256(
                "captured-public-vbd-endpoint-v1",
                (("positions", endpoint.positions), ("velocities", endpoint.velocities)),
            ),
            f"{label} endpoint hash",
        ),
    ):
        if actual != expected:
            raise ValueError(f"{name} is stale")
    pin_error = (
        float(np.max(np.linalg.norm(endpoint.positions[scene.pinned_indices] - scene.pin_targets, axis=1)))
        if scene.pinned_indices.size
        else 0.0
    )
    if endpoint.max_pin_error_m != pin_error or pin_error != 0.0:
        raise ValueError(f"captured direct graph {label} changed an exact pin")
    if endpoint.minimum_determinant != _vbd_minimum_determinant(scene, endpoint.positions):
        raise ValueError(f"captured direct graph {label} minimum determinant is stale")
    if (
        not endpoint.research_only
        or not endpoint.diagnostic_baseline
        or endpoint.integrated_mg
        or endpoint.performance_evidence
    ):
        raise ValueError(f"captured direct graph {label} carries invalid baseline policy")


def _validate_k1_endpoint(
    endpoint: CapturedVBDEndpoint,
    reference: CapturedVBDEndpoint,
    scene: TetBenchmarkScene,
    *,
    device: str,
    graph_replay: bool,
) -> None:
    """Validate the complete public K1 endpoint and its immutable provenance."""
    _validate_public_vbd_endpoint(
        endpoint,
        reference,
        scene,
        iterations=1,
        label="K1",
        device=device,
        graph_replay=graph_replay,
    )


def _validate_k4_endpoint(
    endpoint: CapturedVBDEndpoint,
    reference: CapturedVBDEndpoint,
    scene: TetBenchmarkScene,
    *,
    device: str,
    graph_replay: bool,
) -> None:
    """Validate the public K4 comparator against its fresh construction oracle."""
    _validate_public_vbd_endpoint(
        endpoint,
        reference,
        scene,
        iterations=4,
        label="K4",
        device=device,
        graph_replay=graph_replay,
    )


def _validate_v_cycle_record(
    record: WarpVCycleRecord,
    *,
    rhs: np.ndarray,
    output: np.ndarray,
    canonical_output: np.ndarray,
    canonical_work: object,
    hierarchy_sha256: str,
    warp_snapshot_sha256: str,
    scheduled_kernel_launches: int,
    capture_replay: bool,
    name: str,
) -> None:
    """Recompute one device V-cycle's output, exact work, and nested hashes."""
    if type(record) is not WarpVCycleRecord:
        raise TypeError(f"{name} must be an exact WarpVCycleRecord")
    if type(record.work) is not VCycleWorkRecord:
        raise TypeError(f"{name} work must be an exact VCycleWorkRecord")
    if (
        record.contract_id != V_CYCLE_CONTRACT_ID
        or record.kernel_version != V_CYCLE_KERNEL_VERSION
        or not record.research_only
        or record.performance_evidence
    ):
        raise ValueError(f"{name} has invalid V-cycle policy provenance")
    if record.capture_replay != capture_replay:
        raise ValueError(f"{name} capture provenance disagrees")
    if record.scheduled_kernel_launches != scheduled_kernel_launches:
        raise ValueError(f"{name} scheduled launch count is stale")
    if not np.array_equal(record.correction.reshape(output.shape), output):
        raise ValueError(f"{name} retained correction does not exactly bind its output vector")
    _require_close_array(output, canonical_output.reshape(output.shape), name=f"{name} canonical correction")

    work = record.work
    expected_structural = {
        "hierarchy_sha256": hierarchy_sha256,
        "rhs_count": canonical_work.rhs_count,
        "level_visits": canonical_work.level_visits,
        "matrix_block_products": canonical_work.matrix_block_products,
        "smoother_block_solves": canonical_work.smoother_block_solves,
        "restriction_block_products": canonical_work.restriction_block_products,
        "prolongation_block_products": canonical_work.prolongation_block_products,
        "coarsest_factor_solves": canonical_work.coarsest_factor_solves,
    }
    for field_name, expected in expected_structural.items():
        actual = getattr(work, field_name)
        if actual != expected:
            raise ValueError(f"{name} {field_name} does not match canonical fixed work")
    for field_name in (
        "rhs_count",
        "matrix_block_products",
        "smoother_block_solves",
        "restriction_block_products",
        "prolongation_block_products",
        "coarsest_factor_solves",
    ):
        value = getattr(work, field_name)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f"{name} {field_name} must be a non-negative integer")
    if type(work.level_visits) is not tuple or any(
        isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0 for value in work.level_visits
    ):
        raise ValueError(f"{name} level_visits must contain non-negative integers")
    expected_rhs_sha256 = _v_cycle_rhs_sha256(rhs)
    expected_result_sha256 = _v_cycle_result_sha256(output)
    if work.rhs_sha256 != expected_rhs_sha256 or work.result_sha256 != expected_result_sha256:
        raise ValueError(f"{name} hashes do not bind the retained vectors")
    work_sha256 = _hash_parts(
        "v-cycle-work-record-v1",
        (
            ("hierarchy_sha256", hierarchy_sha256),
            ("rhs_sha256", expected_rhs_sha256),
            ("result_sha256", expected_result_sha256),
            ("rhs_count", work.rhs_count),
            ("level_visits", np.asarray(work.level_visits, dtype=np.int64)),
            ("matrix_block_products", work.matrix_block_products),
            ("smoother_block_solves", work.smoother_block_solves),
            ("restriction_block_products", work.restriction_block_products),
            ("prolongation_block_products", work.prolongation_block_products),
            ("coarsest_factor_solves", work.coarsest_factor_solves),
        ),
    )
    if work.content_sha256 != work_sha256:
        raise ValueError(f"{name} work hash is stale")
    record_sha256 = _hash_parts(
        "warp-v-cycle-result-v1",
        (
            ("snapshot_sha256", warp_snapshot_sha256),
            ("work_sha256", work_sha256),
            ("scheduled_kernel_launches", scheduled_kernel_launches),
            ("capture_replay", capture_replay),
        ),
    )
    if record.content_sha256 != record_sha256:
        raise ValueError(f"{name} record hash is stale")


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutionReceipt:
    """One solver-issued, registered, single-consumption launch receipt."""

    issuer: object = dataclasses.field(repr=False, compare=False)
    serial: int
    graph_replay: bool
    graph_object_identity: tuple[int, int] | None

    def __post_init__(self) -> None:
        if type(self.serial) is not int or self.serial < 1:
            raise ValueError("execution receipt serial must be a positive built-in int")
        if type(self.graph_replay) is not bool:
            raise ValueError("execution receipt graph_replay must be a bool")
        if self.graph_replay:
            if (
                type(self.graph_object_identity) is not tuple
                or len(self.graph_object_identity) != 2
                or any(type(value) is not int or value <= 0 for value in self.graph_object_identity)
            ):
                raise ValueError("captured execution receipt requires exact graph object identities")
        elif self.graph_object_identity is not None:
            raise ValueError("uncaptured execution receipt cannot claim graph object identities")


@dataclasses.dataclass(frozen=True, slots=True)
class _OuterSlotBinding:
    """One solver-issued binding for a specific executed outer-work slot."""

    token: object
    outer_index: int
    start_position_sha256: str
    current_operator_sha256: str
    rhs_sha256: str
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        if self.token is not _VALIDATION_TOKEN:
            raise ValueError("outer slot binding is solver-private")
        if type(self.outer_index) is not int or not 0 <= self.outer_index < OUTER_CORRECTIONS:
            raise ValueError("outer slot binding index is outside the fixed schedule")
        for name in ("start_position_sha256", "current_operator_sha256", "rhs_sha256"):
            _require_sha256(getattr(self, name), name=name)
        if type(self.accepted) is not bool:
            raise ValueError("outer slot binding accepted must be a bool")
        if type(self.reason) is not str or self.reason not in REASON_NAMES:
            raise ValueError("outer slot binding reason is outside the fixed vocabulary")
        if self.accepted != (self.reason == "accepted"):
            raise ValueError("outer slot binding status is inconsistent")


@dataclasses.dataclass(frozen=True, slots=True)
class _EndpointValidationContext:
    """Private canonical inputs required to issue a content-bound endpoint."""

    token: object
    issuer: object = dataclasses.field(repr=False, compare=False)
    execution_serial: int
    graph_replay: bool
    scene: TetBenchmarkScene
    config: DirectGraphVBDConfig
    construction_k1: CapturedVBDEndpoint
    execution_k1: CapturedVBDEndpoint
    hierarchy: StaticMultigridHierarchy
    persistent_device_sha256: str
    warp_snapshot_sha256: str
    graph_identity_sha256: str
    v_cycle_kernel_launches: int
    device: str
    outer_slots: tuple[_OuterSlotBinding, ...]

    def __post_init__(self) -> None:
        if self.token is not _VALIDATION_TOKEN:
            raise ValueError("endpoint validation context is solver-private")
        if type(self.execution_serial) is not int or self.execution_serial < 1:
            raise ValueError("endpoint validation execution_serial must be a positive built-in int")
        if type(self.graph_replay) is not bool:
            raise ValueError("endpoint validation graph_replay must be a bool")
        if type(self.scene) is not TetBenchmarkScene or type(self.config) is not DirectGraphVBDConfig:
            raise TypeError("endpoint validation context has invalid canonical inputs")
        if type(self.hierarchy) is not StaticMultigridHierarchy:
            raise TypeError("endpoint validation context has an invalid hierarchy")
        for name in ("persistent_device_sha256", "warp_snapshot_sha256", "graph_identity_sha256"):
            _require_sha256(getattr(self, name), name=name)
        if type(self.v_cycle_kernel_launches) is not int or self.v_cycle_kernel_launches < 1:
            raise ValueError("v_cycle_kernel_launches must be a positive built-in int")
        if (
            type(self.outer_slots) is not tuple
            or len(self.outer_slots) != OUTER_CORRECTIONS
            or any(type(slot) is not _OuterSlotBinding for slot in self.outer_slots)
            or any(slot.outer_index != index for index, slot in enumerate(self.outer_slots))
        ):
            raise ValueError("endpoint validation context requires four exact ordered outer-slot bindings")


def _require_issued_validation_context(
    context: _EndpointValidationContext,
    *,
    validate_raw_sources: bool,
) -> None:
    """Require the exact live context object registered by its owning solver."""
    if type(context) is not _EndpointValidationContext or context.token is not _VALIDATION_TOKEN:
        raise ValueError("validation context is not a live solver-issued capability")
    issuer_type = globals().get("CapturedDirectGraphVBD")
    if issuer_type is None or type(context.issuer) is not issuer_type:
        raise ValueError("validation context has no exact owning solver")
    context.issuer._validate_issued_context(context, validate_raw_sources=validate_raw_sources)


def _require_issued_outer_slot(context: _EndpointValidationContext, slot: _OuterSlotBinding) -> None:
    """Require one exact slot object registered with its owning execution context."""
    if type(slot) is not _OuterSlotBinding or slot.token is not _VALIDATION_TOKEN:
        raise ValueError("outer slot is not a live solver-issued capability")
    _require_issued_validation_context(context, validate_raw_sources=False)
    context.issuer._validate_issued_outer_slot(context, slot)


@wp.func
def _finite_vec(value: wp.vec3d) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.kernel(enable_backward=False)
def _initialize_from_k1(
    k1_positions: wp.array[wp.vec3],
    canonical_positions: wp.array[wp.vec3d],
    vertex_to_free: wp.array[int],
    current: wp.array[wp.vec3d],
    candidate: wp.array[wp.vec3d],
    proposal_finite: wp.array[int],
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
    proposal_finite[vertex] = 1
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
def _subtract_vectors(
    left: wp.array[wp.vec3d],
    right: wp.array[wp.vec3d],
    output: wp.array[wp.vec3d],
):
    index = wp.tid()
    output[index] = left[index] - right[index]


@wp.kernel(enable_backward=False)
def _add_vectors(
    left: wp.array[wp.vec3d],
    right: wp.array[wp.vec3d],
    output: wp.array[wp.vec3d],
):
    index = wp.tid()
    output[index] = left[index] + right[index]


@wp.kernel(enable_backward=False)
def _build_candidate(
    current: wp.array[wp.vec3d],
    vertex_to_free: wp.array[int],
    direction: wp.array[wp.vec3d],
    active: wp.array[int],
    candidate: wp.array[wp.vec3d],
    proposal_finite: wp.array[int],
):
    vertex = wp.tid()
    value = current[vertex]
    valid = bool(True)
    free_index = vertex_to_free[vertex]
    if active[0] != 0 and free_index >= 0:
        proposed = value + direction[free_index]
        valid = _finite_vec(proposed)
        if valid:
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
    proposal_finite[vertex] = int(valid)


@wp.kernel(enable_backward=False)
def _finalize_gate(
    outer_index: int,
    current_inertia: wp.array[wp.float64],
    candidate_inertia: wp.array[wp.float64],
    current_elastic: wp.array[wp.float64],
    candidate_elastic: wp.array[wp.float64],
    directional_terms: wp.array[wp.float64],
    candidate_determinants: wp.array[wp.float64],
    segment_minima: wp.array[wp.float64],
    proposal_finite: wp.array[int],
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

    start_objective = wp.float64(0.0)
    end_objective = wp.float64(0.0)
    derivative = wp.float64(0.0)
    all_finite = bool(True)
    for vertex in range(current_inertia.shape[0]):
        start_objective += current_inertia[vertex]
        end_objective += candidate_inertia[vertex]
        all_finite = all_finite and proposal_finite[vertex] != 0 and vertex_finite[vertex] != 0
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

    if (
        not all_finite
        or not wp.isfinite(start_objective)
        or not wp.isfinite(end_objective)
        or not wp.isfinite(derivative)
        or not wp.isfinite(minimum_segment)
    ):
        reasons[outer_index] = _REASON_NONFINITE
        active[0] = 0
        return

    initial_objective[outer_index] = start_objective
    candidate_objective[outer_index] = end_objective
    directional_derivative[outer_index] = derivative
    minimum_segment_determinant[outer_index] = minimum_segment
    if derivative >= wp.float64(0.0):
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


@dataclasses.dataclass(frozen=True, eq=False)
class CapturedGraphVBDOuterWork:
    """Immutable current-operator and two-V-cycle evidence for one outer."""

    outer_index: int
    start_position_sha256: str
    current_operator_sha256: str
    static_hierarchy_sha256: str
    rhs: np.ndarray
    first_correction: np.ndarray
    operator_product_after_first: np.ndarray
    residual_after_first: np.ndarray
    second_correction: np.ndarray
    direction: np.ndarray
    v_cycles: tuple[WarpVCycleRecord, WarpVCycleRecord]
    capture_replay: bool
    linear_kernel_launches: int
    persistent_device_sha256: str
    accepted: bool
    reason: str
    _validation_context: _EndpointValidationContext = dataclasses.field(repr=False, compare=False)
    _validation_slot: _OuterSlotBinding = dataclasses.field(repr=False, compare=False)
    _validation_operator: MatrixFreeStableNHOperator = dataclasses.field(repr=False, compare=False)
    content_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        _require_issued_validation_context(self._validation_context, validate_raw_sources=False)
        _require_issued_outer_slot(self._validation_context, self._validation_slot)
        if type(self._validation_operator) is not MatrixFreeStableNHOperator:
            raise TypeError("outer work requires an exact canonical current operator")
        if type(self.outer_index) is not int or not 0 <= self.outer_index < OUTER_CORRECTIONS:
            raise ValueError("outer_index is outside the fixed four-correction schedule")
        slot = self._validation_slot
        if self.outer_index != slot.outer_index:
            raise ValueError("outer work index does not match its exact solver-issued schedule slot")
        start_position_sha256 = _require_sha256(self.start_position_sha256, name="start_position_sha256")
        operator_sha256 = _require_sha256(self.current_operator_sha256, name="current_operator_sha256")
        hierarchy_sha256 = _require_sha256(self.static_hierarchy_sha256, name="static_hierarchy_sha256")
        persistent_device_sha256 = _require_sha256(
            self.persistent_device_sha256,
            name="persistent_device_sha256",
        )
        if start_position_sha256 != slot.start_position_sha256 or start_position_sha256 != _array_digest(
            self._validation_operator.positions
        ):
            raise ValueError("outer work start does not match its exact solver-issued schedule slot")
        if operator_sha256 != slot.current_operator_sha256 or operator_sha256 != _operator_sha256(
            self._validation_operator
        ):
            raise ValueError("outer work current operator identity is not canonical")
        if hierarchy_sha256 != self._validation_context.hierarchy.content_sha256:
            raise ValueError("outer work hierarchy identity is not canonical")
        if persistent_device_sha256 != self._validation_context.persistent_device_sha256:
            raise ValueError("outer work persistent device identity is stale")
        if type(self.v_cycles) is not tuple or len(self.v_cycles) != V_CYCLES_PER_OUTER:
            raise ValueError("outer work must retain exactly two V-cycle records")
        if any(type(record) is not WarpVCycleRecord for record in self.v_cycles):
            raise TypeError("v_cycles must contain exact WarpVCycleRecord values")
        if type(self.capture_replay) is not bool:
            raise ValueError("capture_replay must be a bool")
        if self.capture_replay != self._validation_context.graph_replay:
            raise ValueError("outer work capture provenance does not match its execution context")
        if type(self.accepted) is not bool or type(self.reason) is not str or self.reason not in REASON_NAMES:
            raise ValueError("outer work status is outside the fixed gate vocabulary")
        if self.accepted != (self.reason == "accepted"):
            raise ValueError("outer work accepted flag and reason disagree")
        if self.accepted != slot.accepted or self.reason != slot.reason:
            raise ValueError("outer work status does not match its exact solver-issued schedule slot")

        arrays: dict[str, np.ndarray] = {}
        shape: tuple[int, ...] | None = None
        for name in (
            "rhs",
            "first_correction",
            "operator_product_after_first",
            "residual_after_first",
            "second_correction",
            "direction",
        ):
            array = _immutable_float64(getattr(self, name), name=name)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"{name} must have shape (free_vertex_count, 3)")
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("all outer linear vectors must have the same shape")
            arrays[name] = array
            object.__setattr__(self, name, array)

        rhs_sha256 = _array_digest(arrays["rhs"])
        if rhs_sha256 != slot.rhs_sha256:
            raise ValueError("outer work RHS does not match its exact solver-issued schedule slot")
        expected_rhs = -self._validation_operator.gradient_free().reshape(arrays["rhs"].shape)
        if self.reason == "masked-after-rejection":
            if np.any(arrays["rhs"] != 0.0):
                raise ValueError("masked outer work RHS must be exactly zero")
        else:
            _require_close_array(arrays["rhs"], expected_rhs, name="negative current gradient RHS")

        if not np.array_equal(arrays["residual_after_first"], arrays["rhs"] - arrays["operator_product_after_first"]):
            raise ValueError("retained first residual is not b-A_current*z1")
        if not np.array_equal(arrays["direction"], arrays["first_correction"] + arrays["second_correction"]):
            raise ValueError("retained direction is not z1+z2")

        expected_product = self._validation_operator.apply_free(arrays["first_correction"].reshape(-1)).reshape(
            arrays["operator_product_after_first"].shape
        )
        _require_close_array(
            arrays["operator_product_after_first"],
            expected_product,
            name="current operator product after first V-cycle",
        )
        first = apply_v_cycle(self._validation_context.hierarchy, arrays["rhs"].reshape(-1))
        second = apply_v_cycle(self._validation_context.hierarchy, arrays["residual_after_first"].reshape(-1))
        for cycle_index, (record, rhs, output, canonical) in enumerate(
            zip(
                self.v_cycles,
                (arrays["rhs"], arrays["residual_after_first"]),
                (arrays["first_correction"], arrays["second_correction"]),
                (first, second),
                strict=True,
            )
        ):
            _validate_v_cycle_record(
                record,
                rhs=rhs,
                output=output,
                canonical_output=canonical.correction,
                canonical_work=canonical.work,
                hierarchy_sha256=hierarchy_sha256,
                warp_snapshot_sha256=self._validation_context.warp_snapshot_sha256,
                scheduled_kernel_launches=self._validation_context.v_cycle_kernel_launches,
                capture_replay=self.capture_replay,
                name=f"V-cycle {cycle_index}",
            )

        expected_launches = 7 + sum(record.scheduled_kernel_launches for record in self.v_cycles)
        if type(self.linear_kernel_launches) is not int or self.linear_kernel_launches != expected_launches:
            raise ValueError("linear_kernel_launches does not match the fixed direct schedule")
        payload = {
            "contract": "captured-direct-graph-vbd-outer-work-v1",
            "outer_index": self.outer_index,
            "start_position_sha256": start_position_sha256,
            "current_operator_sha256": operator_sha256,
            "static_hierarchy_sha256": hierarchy_sha256,
            "rhs_sha256": rhs_sha256,
            "first_correction_sha256": _array_digest(arrays["first_correction"]),
            "operator_product_after_first_sha256": _array_digest(arrays["operator_product_after_first"]),
            "residual_after_first_sha256": _array_digest(arrays["residual_after_first"]),
            "second_correction_sha256": _array_digest(arrays["second_correction"]),
            "direction_sha256": _array_digest(arrays["direction"]),
            "persistent_device_sha256": persistent_device_sha256,
            "v_cycle_content_sha256s": [record.content_sha256 for record in self.v_cycles],
            "linear_kernel_launches": self.linear_kernel_launches,
            "capture_replay": self.capture_replay,
            "accepted": self.accepted,
            "reason": self.reason,
        }
        object.__setattr__(self, "content_sha256", _canonical_digest(payload))

    @property
    def exact_work_completed(self) -> bool:
        """Whether the complete two-cycle fixed schedule is retained."""
        return bool(
            len(self.v_cycles) == V_CYCLES_PER_OUTER
            and all(record.work.rhs_count == 1 and record.work.coarsest_factor_solves == 1 for record in self.v_cycles)
        )

    def deterministic_record(self) -> dict[str, object]:
        """Serialize hashes and exact work without duplicating raw vectors."""
        return {
            "contract": "captured-direct-graph-vbd-outer-work-v1",
            "outer_index": self.outer_index,
            "start_position_sha256": self.start_position_sha256,
            "current_operator_sha256": self.current_operator_sha256,
            "static_hierarchy_sha256": self.static_hierarchy_sha256,
            "rhs_sha256": _array_digest(self.rhs),
            "first_correction_sha256": _array_digest(self.first_correction),
            "operator_product_after_first_sha256": _array_digest(self.operator_product_after_first),
            "residual_after_first_sha256": _array_digest(self.residual_after_first),
            "second_correction_sha256": _array_digest(self.second_correction),
            "direction_sha256": _array_digest(self.direction),
            "persistent_device_sha256": self.persistent_device_sha256,
            "v_cycles": [record.deterministic_record() for record in self.v_cycles],
            "linear_kernel_launches": self.linear_kernel_launches,
            "exact_work_completed": self.exact_work_completed,
            "capture_replay": self.capture_replay,
            "accepted": self.accepted,
            "reason": self.reason,
            "content_sha256": self.content_sha256,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class CapturedGraphVBDEndpoint:
    """Synchronized endpoint, exact gate evidence, and fixed graph work."""

    scene_sha256: str
    objective_instance_sha256: str
    static_hierarchy_sha256: str
    config_sha256: str
    k1_endpoint_sha256: str
    k1_position_sha256: str
    k1_velocity_sha256: str
    k1_pristine_state_sha256: str
    persistent_device_sha256: str
    graph_identity_sha256: str
    armijo: float
    minimum_determinant: float
    free_vertices: np.ndarray
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
    outer_work: tuple[CapturedGraphVBDOuterWork, ...]
    graph_replay: bool
    _validation_context: _EndpointValidationContext = dataclasses.field(repr=False, compare=False)
    position_sha256: str = dataclasses.field(init=False)
    velocity_sha256: str = dataclasses.field(init=False)
    outer_start_position_sha256s: tuple[str, ...] = dataclasses.field(init=False)
    outer_candidate_position_sha256s: tuple[str, ...] = dataclasses.field(init=False)
    current_operator_sha256s: tuple[str, ...] = dataclasses.field(init=False)
    correction_kernel_launches: int = dataclasses.field(init=False)
    endpoint_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        context = self._validation_context
        _require_issued_validation_context(context, validate_raw_sources=True)
        sequence_fields = (
            "accepted",
            "reasons",
            "initial_objectives",
            "candidate_objectives",
            "directional_derivatives",
            "segment_minimum_determinants",
            "outer_start_positions",
            "outer_candidate_positions",
            "outer_work",
        )
        if any(type(getattr(self, name)) is not tuple for name in sequence_fields):
            raise ValueError("captured endpoint sequence fields must be exact tuples")
        scene_sha256 = _require_sha256(self.scene_sha256, name="scene_sha256")
        objective_sha256 = _require_sha256(self.objective_instance_sha256, name="objective_instance_sha256")
        hierarchy_sha256 = _require_sha256(self.static_hierarchy_sha256, name="static_hierarchy_sha256")
        config_sha256 = _require_sha256(self.config_sha256, name="config_sha256")
        k1_endpoint_sha256 = _require_sha256(self.k1_endpoint_sha256, name="k1_endpoint_sha256")
        k1_position_sha256 = _require_sha256(self.k1_position_sha256, name="k1_position_sha256")
        k1_velocity_sha256 = _require_sha256(self.k1_velocity_sha256, name="k1_velocity_sha256")
        k1_pristine_sha256 = _require_sha256(
            self.k1_pristine_state_sha256,
            name="k1_pristine_state_sha256",
        )
        persistent_device_sha256 = _require_sha256(
            self.persistent_device_sha256,
            name="persistent_device_sha256",
        )
        graph_identity_sha256 = _require_sha256(self.graph_identity_sha256, name="graph_identity_sha256")
        context.config.validate()
        expected_scene_sha256 = _require_sha256(context.scene.manifest()["scene_sha256"], name="scene_sha256")
        if scene_sha256 != expected_scene_sha256:
            raise ValueError("scene_sha256 does not bind the canonical retained scene")
        expected_config_sha256 = _canonical_digest(context.config.deterministic_record())
        if config_sha256 != expected_config_sha256:
            raise ValueError("config_sha256 does not bind the exact captured configuration")
        if self.armijo != float(context.config.armijo) or self.minimum_determinant != float(
            context.config.minimum_determinant
        ):
            raise ValueError("endpoint gate parameters do not match the exact captured configuration")
        if persistent_device_sha256 != context.persistent_device_sha256:
            raise ValueError("endpoint persistent device identity is stale")
        if graph_identity_sha256 != context.graph_identity_sha256:
            raise ValueError("endpoint graph identity is stale")
        if type(self.armijo) is not float or not math.isfinite(self.armijo) or not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must be a built-in float in (0, 1)")
        if (
            type(self.minimum_determinant) is not float
            or not math.isfinite(self.minimum_determinant)
            or self.minimum_determinant < 0.0
        ):
            raise ValueError("minimum_determinant must be a non-negative built-in float")
        if type(self.graph_replay) is not bool:
            raise ValueError("graph_replay must be a bool")
        if self.graph_replay != context.graph_replay:
            raise ValueError("endpoint graph provenance does not match its execution context")

        _validate_k1_endpoint(
            context.execution_k1,
            context.construction_k1,
            context.scene,
            device=context.device,
            graph_replay=self.graph_replay,
        )
        k1 = context.execution_k1
        if (
            k1_endpoint_sha256 != k1.endpoint_sha256
            or k1_position_sha256 != k1.position_sha256
            or k1_velocity_sha256 != k1.velocity_sha256
            or k1_pristine_sha256 != k1.pristine_state_sha256
        ):
            raise ValueError("endpoint K1 hashes do not bind the validated public K1 execution")
        problem = build_common_problem(context.scene)
        expected_objective_sha256 = _require_sha256(
            common_objective_manifest(context.scene, problem)["objective_instance_sha256"],
            name="objective_instance_sha256",
        )
        if objective_sha256 != expected_objective_sha256:
            raise ValueError("objective identity does not bind the canonical retained scene")
        k1_operator = MatrixFreeStableNHOperator.from_problem(problem, k1.positions)
        _, canonical_hierarchy = _canonical_static_hierarchy(
            k1_operator,
            context.hierarchy,
            context.scene.rest_q,
            context.config,
        )
        if hierarchy_sha256 != canonical_hierarchy.content_sha256:
            raise ValueError("static hierarchy identity does not bind the canonical A0 rebuild")

        free = _immutable_int64(self.free_vertices, name="free_vertices")
        positions = _immutable_float64(self.positions, name="positions")
        velocities = _immutable_float64(self.velocities, name="velocities")
        if positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape:
            raise ValueError("captured positions and velocities must have matching shape (V, 3)")
        if free.size and (free.max() >= positions.shape[0] or np.unique(free).size != free.size):
            raise ValueError("free_vertices contains duplicates or out-of-range entries")
        canonical_free = np.asarray(problem.free.detach().cpu().numpy(), dtype=np.int64)
        if not np.array_equal(free, canonical_free):
            raise ValueError("free_vertices does not exactly match the canonical problem ordering")
        if positions.shape != k1.positions.shape:
            raise ValueError("captured endpoint shape does not match the canonical scene")
        object.__setattr__(self, "free_vertices", free)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)

        starts = tuple(
            _immutable_float64(value, name=f"outer_start_positions[{index}]")
            for index, value in enumerate(self.outer_start_positions)
        )
        candidates = tuple(
            _immutable_float64(value, name=f"outer_candidate_positions[{index}]")
            for index, value in enumerate(self.outer_candidate_positions)
        )
        if any(value.shape != positions.shape for value in starts + candidates):
            raise ValueError("every outer position array must match the endpoint shape")
        object.__setattr__(self, "outer_start_positions", starts)
        object.__setattr__(self, "outer_candidate_positions", candidates)
        lengths = tuple(len(getattr(self, name)) for name in sequence_fields)
        if any(length != OUTER_CORRECTIONS for length in lengths):
            raise ValueError("captured graph-VBD endpoint must retain all four outer slots")
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
        if any(type(work) is not CapturedGraphVBDOuterWork for work in self.outer_work):
            raise TypeError("outer_work must contain exact CapturedGraphVBDOuterWork values")

        for name, array in (
            ("positions", positions),
            ("velocities", velocities),
            *((f"outer_start_positions[{index}]", value) for index, value in enumerate(starts)),
            *((f"outer_candidate_positions[{index}]", value) for index, value in enumerate(candidates)),
        ):
            if not np.array_equal(array, array.astype(np.float32).astype(np.float64)):
                raise ValueError(f"{name} must be exactly representable by the publishable float32 state")

        start_hashes = tuple(_array_digest(value) for value in starts)
        operator_sha256s: list[str] = []
        current = np.array(k1.positions, dtype=np.float64, copy=True)
        active = True
        for index, work in enumerate(self.outer_work):
            if not np.array_equal(starts[index], current):
                raise ValueError("outer starts do not form one continuous fail-closed state sequence")
            operator = MatrixFreeStableNHOperator.from_problem(problem, current)
            operator_sha256 = _operator_sha256(operator)
            operator_sha256s.append(operator_sha256)
            if work.outer_index != index:
                raise ValueError("outer work indices must cover the fixed schedule in order")
            if work._validation_context is not context:
                raise ValueError("outer work belongs to another endpoint validation context")
            if work._validation_slot is not context.outer_slots[index]:
                raise ValueError("outer work does not retain its exact solver-issued schedule slot")
            if work.start_position_sha256 != start_hashes[index]:
                raise ValueError("outer work does not bind its current start position")
            if work.accepted != self.accepted[index] or work.reason != self.reasons[index]:
                raise ValueError("outer work status does not bind the endpoint gate status")
            if work.current_operator_sha256 != operator_sha256:
                raise ValueError("outer work does not bind its current position/operator")
            if _operator_sha256(work._validation_operator) != operator_sha256 or not np.array_equal(
                work._validation_operator.positions,
                current,
            ):
                raise ValueError("outer work private operator does not match the canonical current operator")
            if work.static_hierarchy_sha256 != hierarchy_sha256:
                raise ValueError("outer work changed the static rest hierarchy")
            if work.persistent_device_sha256 != persistent_device_sha256:
                raise ValueError("outer work changed the persistent device snapshot")
            if work.capture_replay != self.graph_replay or not work.exact_work_completed:
                raise ValueError("outer work did not retain the exact capture schedule")
            if work.rhs.shape != (free.size, 3):
                raise ValueError("outer work does not match the endpoint free-vertex set")

            expected_rhs = -operator.gradient_free().reshape(-1, 3)
            if not active:
                expected_rhs.fill(0.0)
            _require_close_array(work.rhs, expected_rhs, name=f"outer {index} negative current gradient")
            if not active:
                if self.accepted[index] or self.reasons[index] != "masked-after-rejection":
                    raise ValueError("outer evidence after the first rejection must be masked")
                if any(row[index] != 0.0 for row in numeric_rows):
                    raise ValueError("masked outer evidence must contain zero gate scalars")
                if not np.array_equal(candidates[index], current):
                    raise ValueError("masked outer candidate must preserve the rejected state")
                for name in (
                    "rhs",
                    "first_correction",
                    "operator_product_after_first",
                    "residual_after_first",
                    "second_correction",
                    "direction",
                ):
                    if np.any(getattr(work, name) != 0.0):
                        raise ValueError(f"masked outer {name} must be exactly zero")
                continue

            with np.errstate(over="ignore", invalid="ignore"):
                proposed_free = current[free] + work.direction
            finite_rows = np.isfinite(proposed_free).all(axis=1)
            expected_candidate = current.copy()
            expected_candidate[free[finite_rows]] = proposed_free[finite_rows].astype(np.float32).astype(np.float64)
            if not np.array_equal(candidates[index], expected_candidate):
                raise ValueError("outer candidate is not the exact float32-publishable current+d state")

            if not bool(np.all(finite_rows)):
                expected_reason = "candidate-nonfinite"
                expected_scalars = (0.0, 0.0, 0.0, 0.0)
            else:
                candidate_operator = MatrixFreeStableNHOperator.from_problem(problem, expected_candidate)
                initial_objective = operator.objective()
                candidate_objective = candidate_operator.objective()
                actual_step = expected_candidate[free] - current[free]
                derivative = float(np.vdot(operator.gradient_free(), actual_step.reshape(-1)))
                segment = minimum_determinant_on_segment(operator, candidate_operator).determinant
                expected_scalars = (initial_objective, candidate_objective, derivative, segment)
                if not all(math.isfinite(value) for value in expected_scalars):
                    expected_reason = "candidate-nonfinite"
                    expected_scalars = (0.0, 0.0, 0.0, 0.0)
                elif derivative >= 0.0:
                    expected_reason = "non-descent"
                elif (
                    candidate_operator.minimum_determinant <= self.minimum_determinant
                    or segment <= self.minimum_determinant
                ):
                    expected_reason = "segment-inversion"
                elif not (
                    candidate_objective < initial_objective
                    and candidate_objective <= initial_objective + self.armijo * derivative
                ):
                    expected_reason = "objective-increase"
                else:
                    expected_reason = "accepted"

            if self.reasons[index] != expected_reason or self.accepted[index] != (expected_reason == "accepted"):
                raise ValueError("outer gate status does not match the canonical published-state replay")
            for row, expected, name in zip(
                numeric_rows,
                expected_scalars,
                ("initial objective", "candidate objective", "directional derivative", "segment determinant"),
                strict=True,
            ):
                _require_close_scalar(
                    row[index],
                    expected,
                    name=f"outer {index} {name}",
                    relative_tolerance=5.0e-11 if name == "segment determinant" else 8.0e-12,
                    absolute_tolerance=5.0e-12 if name == "segment determinant" else 8.0e-13,
                )
            if expected_reason == "accepted":
                current = expected_candidate
            else:
                active = False

        operator_sha256s_tuple = tuple(operator_sha256s)
        if not np.array_equal(positions, current):
            raise ValueError("captured endpoint is not the final canonical accepted state")
        for name, state in (
            ("endpoint", positions),
            *((f"outer start {index}", value) for index, value in enumerate(starts)),
            *((f"outer candidate {index}", value) for index, value in enumerate(candidates)),
        ):
            if context.scene.pinned_indices.size and not np.array_equal(
                state[context.scene.pinned_indices],
                context.scene.pin_targets,
            ):
                raise ValueError(f"{name} changed an exact pin")
        expected_velocities = (
            ((positions - context.scene.x_current) * np.float64(1.0 / context.scene.dt))
            .astype(np.float32)
            .astype(np.float64)
        )
        expected_velocities[context.scene.pinned_indices] = 0.0
        if not np.array_equal(velocities, expected_velocities):
            raise ValueError("captured velocities are not exact published BDF1 velocities")

        position_sha256 = _array_digest(positions)
        velocity_sha256 = _array_digest(velocities)
        candidate_hashes = tuple(_array_digest(value) for value in candidates)
        correction_launches = 2 + sum(work.linear_kernel_launches + 8 for work in self.outer_work)
        object.__setattr__(self, "position_sha256", position_sha256)
        object.__setattr__(self, "velocity_sha256", velocity_sha256)
        object.__setattr__(self, "outer_start_position_sha256s", start_hashes)
        object.__setattr__(self, "outer_candidate_position_sha256s", candidate_hashes)
        object.__setattr__(self, "current_operator_sha256s", operator_sha256s_tuple)
        object.__setattr__(self, "correction_kernel_launches", correction_launches)
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
                    "k1_endpoint_sha256": k1_endpoint_sha256,
                    "k1_position_sha256": k1_position_sha256,
                    "k1_velocity_sha256": k1_velocity_sha256,
                    "k1_pristine_state_sha256": k1_pristine_sha256,
                    "persistent_device_sha256": persistent_device_sha256,
                    "graph_identity_sha256": graph_identity_sha256,
                    "armijo": self.armijo,
                    "minimum_determinant": self.minimum_determinant,
                    "free_vertices_sha256": _array_digest(free),
                    "position_sha256": position_sha256,
                    "velocity_sha256": velocity_sha256,
                    "accepted": list(self.accepted),
                    "reasons": list(self.reasons),
                    "initial_objectives": list(self.initial_objectives),
                    "candidate_objectives": list(self.candidate_objectives),
                    "directional_derivatives": list(self.directional_derivatives),
                    "segment_minimum_determinants": list(self.segment_minimum_determinants),
                    "outer_start_position_sha256s": list(start_hashes),
                    "outer_candidate_position_sha256s": list(candidate_hashes),
                    "outer_work_sha256s": [work.content_sha256 for work in self.outer_work],
                    "correction_kernel_launches": correction_launches,
                    "graph_replay": self.graph_replay,
                }
            ),
        )

    @property
    def total_v_cycle_count(self) -> int:
        """Exact retained V-cycle count."""
        return sum(len(work.v_cycles) for work in self.outer_work)

    @property
    def exact_work_completed(self) -> bool:
        """Whether all four two-cycle outer slots retain fixed work."""
        return bool(
            len(self.outer_work) == OUTER_CORRECTIONS
            and self.total_v_cycle_count == OUTER_CORRECTIONS * V_CYCLES_PER_OUTER
            and all(work.exact_work_completed for work in self.outer_work)
        )

    def deterministic_record(self) -> dict[str, object]:
        """Serialize content identities, exact work, and gate evidence."""
        return {
            "contract": CONTRACT_ID,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "static_hierarchy_sha256": self.static_hierarchy_sha256,
            "config_sha256": self.config_sha256,
            "k1_endpoint_sha256": self.k1_endpoint_sha256,
            "k1_position_sha256": self.k1_position_sha256,
            "k1_velocity_sha256": self.k1_velocity_sha256,
            "k1_pristine_state_sha256": self.k1_pristine_state_sha256,
            "persistent_device_sha256": self.persistent_device_sha256,
            "graph_identity_sha256": self.graph_identity_sha256,
            "armijo": self.armijo,
            "minimum_determinant": self.minimum_determinant,
            "free_vertices_sha256": _array_digest(self.free_vertices),
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
            "outer_work": [work.deterministic_record() for work in self.outer_work],
            "total_v_cycle_count": self.total_v_cycle_count,
            "correction_kernel_launches": self.correction_kernel_launches,
            "exact_work_completed": self.exact_work_completed,
            "graph_replay": self.graph_replay,
            "endpoint_sha256": self.endpoint_sha256,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class CapturedGraphVBDTiming:
    """Paired CUDA-event timings for direct graph VBD versus pristine K4."""

    pair_orders: tuple[str, ...]
    graph_seconds: tuple[float, ...]
    k4_seconds: tuple[float, ...]
    warmup_replays: int
    random_seed: int
    device: str
    contract_id: str
    scene_sha256: str
    objective_instance_sha256: str
    config_sha256: str
    static_hierarchy_sha256: str
    persistent_device_sha256: str
    graph_identity_sha256: str
    k4_graph_identity_sha256: str
    comparator_contract_id: str
    setup_included: bool = False
    transfers_included: bool = False
    integrated_direct_graph: bool = True
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_orders", tuple(self.pair_orders))
        object.__setattr__(self, "graph_seconds", tuple(self.graph_seconds))
        object.__setattr__(self, "k4_seconds", tuple(self.k4_seconds))
        count = len(self.pair_orders)
        if count < 2 or count % 2 != 0 or len(self.graph_seconds) != count or len(self.k4_seconds) != count:
            raise ValueError("paired timing arrays must have the same positive even length")
        if any(order not in ("AB", "BA") for order in self.pair_orders):
            raise ValueError("pair orders must use AB/BA labels")
        if self.pair_orders.count("AB") != self.pair_orders.count("BA"):
            raise ValueError("paired timing must contain equal AB and BA counts")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.graph_seconds + self.k4_seconds):
            raise ValueError("CUDA-event timings must be finite and positive")
        if type(self.warmup_replays) is not int or self.warmup_replays < 1:
            raise ValueError("warmup_replays must be a positive built-in int")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be a built-in int")
        if type(self.device) is not str or not self.device:
            raise ValueError("device must be a non-empty built-in string")
        if self.contract_id != CONTRACT_ID or self.comparator_contract_id != VBD_BASELINE_CONTRACT_ID:
            raise ValueError("timing contract identities are invalid")
        for name in (
            "scene_sha256",
            "objective_instance_sha256",
            "config_sha256",
            "static_hierarchy_sha256",
            "persistent_device_sha256",
            "graph_identity_sha256",
            "k4_graph_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        policy = (self.setup_included, self.transfers_included, self.integrated_direct_graph, self.performance_evidence)
        if any(type(value) is not bool for value in policy):
            raise ValueError("timing policy fields must be built-in bools")
        if (
            self.setup_included
            or self.transfers_included
            or not self.integrated_direct_graph
            or self.performance_evidence
        ):
            raise ValueError("integrated timing must exclude setup/transfers and remain diagnostic-only")

    @property
    def graph_median_seconds(self) -> float:
        """Median captured direct graph VBD time [s]."""
        return statistics.median(self.graph_seconds)

    @property
    def k4_median_seconds(self) -> float:
        """Median captured pristine K4 time [s]."""
        return statistics.median(self.k4_seconds)

    def deterministic_record(self) -> dict[str, object]:
        """Serialize the paired diagnostic timing result."""
        return {
            "contract_id": self.contract_id,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "config_sha256": self.config_sha256,
            "static_hierarchy_sha256": self.static_hierarchy_sha256,
            "persistent_device_sha256": self.persistent_device_sha256,
            "graph_identity_sha256": self.graph_identity_sha256,
            "k4_graph_identity_sha256": self.k4_graph_identity_sha256,
            "comparator_contract_id": self.comparator_contract_id,
            "pair_orders": list(self.pair_orders),
            "graph_seconds": list(self.graph_seconds),
            "k4_seconds": list(self.k4_seconds),
            "warmup_replays": self.warmup_replays,
            "random_seed": self.random_seed,
            "device": self.device,
            "setup_included": self.setup_included,
            "transfers_included": self.transfers_included,
            "integrated_direct_graph": self.integrated_direct_graph,
            "performance_evidence": self.performance_evidence,
        }


def _canonical_free_incidence(tets: np.ndarray, free: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact sorted free-vertex incidence uploaded by the Warp operator."""
    lookup = {int(vertex): index for index, vertex in enumerate(free)}
    rows: list[list[tuple[int, int]]] = [[] for _ in range(free.size)]
    for tet_index, tet in enumerate(tets):
        for corner, vertex in enumerate(tet):
            free_index = lookup.get(int(vertex))
            if free_index is not None:
                rows[free_index].append((tet_index, corner))
    offsets = np.zeros(free.size + 1, dtype=np.int32)
    entries: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        row.sort()
        entries.extend(row)
        offsets[index + 1] = len(entries)
    incidence_tets = np.asarray([entry[0] for entry in entries], dtype=np.int32)
    incidence_corners = np.asarray([entry[1] for entry in entries], dtype=np.int32)
    return offsets, incidence_tets, incidence_corners


def _validate_device_operator_inputs(
    device_operator: WarpMatrixFreeStableNHOperator,
    oracle: MatrixFreeStableNHOperator,
    canonical_positions: wp.array[wp.vec3d],
    x_current: wp.array[wp.vec3d],
    scene: TetBenchmarkScene,
) -> str:
    """Deep-compare every persistent Warp operator input against the CPU oracle."""
    if type(device_operator) is not WarpMatrixFreeStableNHOperator:
        raise TypeError("device operator must be an exact WarpMatrixFreeStableNHOperator")
    expected_tets = np.asarray(oracle.tets, dtype=np.int32).reshape(-1)
    expected_free = np.asarray(oracle.free, dtype=np.int32)
    vertex_to_free = np.full(oracle.n_vertices, -1, dtype=np.int32)
    vertex_to_free[expected_free] = np.arange(expected_free.size, dtype=np.int32)
    incidence_offsets, incidence_tets, incidence_corners = _canonical_free_incidence(
        np.asarray(oracle.tets, dtype=np.int64),
        np.asarray(oracle.free, dtype=np.int64),
    )
    arrays = (
        ("operator.tets", device_operator.tets.numpy(), expected_tets),
        (
            "operator.shape_gradients",
            device_operator.shape_gradients.numpy(),
            np.asarray(oracle.shape_gradients, dtype=np.float64).reshape(-1, 3),
        ),
        ("operator.volumes", device_operator.volumes.numpy(), np.asarray(oracle.volumes, dtype=np.float64)),
        ("operator.mass", device_operator.mass.numpy(), np.asarray(oracle.mass, dtype=np.float64)),
        ("operator.mu", device_operator.mu.numpy(), np.asarray(oracle.mu, dtype=np.float64)),
        ("operator.lam", device_operator.lam.numpy(), np.asarray(oracle.lam, dtype=np.float64)),
        (
            "operator.inertial_target",
            device_operator.inertial_target.numpy(),
            np.asarray(oracle.inertial_target, dtype=np.float64),
        ),
        ("operator.free", device_operator.free.numpy(), expected_free),
        ("operator.vertex_to_free", device_operator.vertex_to_free.numpy(), vertex_to_free),
        ("operator.incidence_offsets", device_operator.incidence_offsets.numpy(), incidence_offsets),
        ("operator.incidence_tets", device_operator.incidence_tets.numpy(), incidence_tets),
        ("operator.incidence_corners", device_operator.incidence_corners.numpy(), incidence_corners),
        ("canonical_positions", canonical_positions.numpy(), np.asarray(oracle.positions, dtype=np.float64)),
        ("x_current", x_current.numpy(), np.asarray(scene.x_current, dtype=np.float64)),
    )
    parts: list[tuple[str, object]] = []
    for name, actual_value, expected_value in arrays:
        actual = np.asarray(actual_value)
        expected = np.asarray(expected_value)
        _require_exact_array(actual, expected, name=name)
        parts.append((name, actual))
    for name, actual, expected in (
        ("operator.free_host", device_operator.free_host, expected_free),
        ("operator.vertex_to_free_host", device_operator.vertex_to_free_host, vertex_to_free),
        ("operator.incidence_offsets_host", device_operator.incidence_offsets_host, incidence_offsets),
        ("operator.incidence_tets_host", device_operator.incidence_tets_host, incidence_tets),
        ("operator.incidence_corners_host", device_operator.incidence_corners_host, incidence_corners),
    ):
        _require_exact_array(np.asarray(actual), np.asarray(expected), name=name)
        parts.append((name, np.asarray(actual)))
    scalar_checks = (
        ("n_vertices", device_operator.n_vertices, oracle.n_vertices),
        ("n_tets", device_operator.n_tets, int(oracle.tets.shape[0])),
        ("n_free", device_operator.n_free, int(oracle.free.size)),
        ("n_free_dofs", device_operator.n_free_dofs, oracle.n_free_dofs),
    )
    for name, actual, expected in scalar_checks:
        if type(actual) is not int or actual != expected:
            raise RuntimeError(f"persistent operator {name} changed after construction")
        parts.append((f"operator.{name}", actual))
    if device_operator.dt != oracle.dt or device_operator.inverse_dt_squared != 1.0 / (oracle.dt * oracle.dt):
        raise RuntimeError("persistent operator timestep scalars changed after construction")
    parts.extend(
        (
            ("operator.dt", np.asarray(device_operator.dt, dtype=np.float64)),
            ("operator.inverse_dt_squared", np.asarray(device_operator.inverse_dt_squared, dtype=np.float64)),
        )
    )
    return _hash_parts("captured-direct-graph-vbd-device-operator-inputs-v1", parts)


def _validate_device_hierarchy_inputs(
    device_hierarchy: WarpStaticMultigridHierarchy,
    hierarchy: StaticMultigridHierarchy,
) -> str:
    """Deep-compare every uploaded hierarchy scalar and array to canonical CPU data."""
    if type(device_hierarchy) is not WarpStaticMultigridHierarchy:
        raise TypeError("device hierarchy must be an exact WarpStaticMultigridHierarchy")
    expected_warp_snapshot_sha256 = _hash_parts(
        "warp-static-multigrid-snapshot-v1",
        (
            ("hierarchy_sha256", hierarchy.content_sha256),
            ("kernel_version", V_CYCLE_KERNEL_VERSION),
            ("coarse_scalar_bound", MAX_COARSE_SCALAR_SIZE),
        ),
    )
    if (
        device_hierarchy.hierarchy_sha256 != hierarchy.content_sha256
        or device_hierarchy.solver_contract != hierarchy.solver_contract
        or device_hierarchy.static_model_sha256 != hierarchy.static_model_sha256
        or device_hierarchy.pre_smooth_steps != hierarchy.pre_smooth_steps
        or device_hierarchy.post_smooth_steps != hierarchy.post_smooth_steps
        or device_hierarchy.n_free != hierarchy.levels[0].matrix.block_row_count
        or device_hierarchy.n_free_dofs != hierarchy.levels[0].matrix.scalar_size
        or len(device_hierarchy.levels) != len(hierarchy.levels)
        or device_hierarchy.device_snapshot_sha256 != expected_warp_snapshot_sha256
    ):
        raise RuntimeError("persistent device hierarchy metadata changed after construction")
    expected_free = np.asarray(hierarchy.free_vertices, dtype=np.int32)
    _require_exact_array(device_hierarchy.free_vertices_host, expected_free, name="hierarchy.free_vertices_host")
    parts: list[tuple[str, object]] = [
        ("hierarchy_sha256", hierarchy.content_sha256),
        ("solver_contract", hierarchy.solver_contract),
        ("static_model_sha256", "none" if hierarchy.static_model_sha256 is None else hierarchy.static_model_sha256),
        ("pre_smooth_steps", hierarchy.pre_smooth_steps),
        ("post_smooth_steps", hierarchy.post_smooth_steps),
        ("free_vertices_host", expected_free),
        ("warp_snapshot_sha256", expected_warp_snapshot_sha256),
    ]
    for level_index, (device_level, cpu_level) in enumerate(
        zip(device_hierarchy.levels, hierarchy.levels, strict=True)
    ):
        matrix = cpu_level.matrix
        scalar_checks = (
            ("block_row_count", device_level.block_row_count, matrix.block_row_count),
            ("block_size", device_level.block_size, matrix.block_size),
            ("scalar_size", device_level.scalar_size, matrix.scalar_size),
            ("stored_block_count", device_level.stored_block_count, matrix.stored_block_count),
        )
        for name, actual, expected in scalar_checks:
            if type(actual) is not int or actual != expected:
                raise RuntimeError(f"persistent hierarchy level {level_index} {name} changed")
            parts.append((f"level_{level_index}.{name}", actual))
        level_arrays = (
            ("row_offsets", device_level.row_offsets, np.asarray(matrix.row_offsets, dtype=np.int32)),
            (
                "column_indices",
                device_level.column_indices,
                np.asarray(matrix.column_indices, dtype=np.int32),
            ),
            ("matrix_values", device_level.matrix_values, np.asarray(matrix.values, dtype=np.float64).reshape(-1)),
        )
        for name, device_array, expected in level_arrays:
            actual = np.asarray(device_array.numpy())
            _require_exact_array(actual, expected, name=f"hierarchy.level_{level_index}.{name}")
            parts.append((f"level_{level_index}.{name}", actual))

        is_coarsest = level_index == len(hierarchy.levels) - 1
        if is_coarsest:
            optional = (
                device_level.inverse_diagonal,
                device_level.omega,
                device_level.aggregate,
                device_level.prolongation_blocks,
                device_level.member_offsets,
                device_level.member_fine_nodes,
                device_level.coarse_node_count,
                device_level.coarse_block_size,
            )
            if any(value is not None for value in optional):
                raise RuntimeError("persistent coarsest device hierarchy level gained transfer data")
            continue
        if cpu_level.smoother is None or cpu_level.prolongation is None:
            raise RuntimeError("canonical noncoarse hierarchy level is incomplete")
        if any(
            value is None
            for value in (
                device_level.inverse_diagonal,
                device_level.omega,
                device_level.aggregate,
                device_level.prolongation_blocks,
                device_level.member_offsets,
                device_level.member_fine_nodes,
                device_level.coarse_node_count,
                device_level.coarse_block_size,
            )
        ):
            raise RuntimeError("persistent noncoarse device hierarchy level lost transfer data")
        aggregate = np.asarray(cpu_level.prolongation.aggregate, dtype=np.int32)
        coarse_count = cpu_level.prolongation.coarse_node_count
        counts = np.bincount(aggregate, minlength=coarse_count)
        member_offsets = np.zeros(coarse_count + 1, dtype=np.int32)
        member_offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
        member_nodes = np.concatenate(
            [np.flatnonzero(aggregate == aggregate_id) for aggregate_id in range(coarse_count)]
        ).astype(np.int32, copy=False)
        optional_arrays = (
            (
                "inverse_diagonal",
                device_level.inverse_diagonal,
                np.asarray(cpu_level.smoother.inverse_diagonal, dtype=np.float64).reshape(-1),
            ),
            ("aggregate", device_level.aggregate, aggregate),
            (
                "prolongation_blocks",
                device_level.prolongation_blocks,
                np.asarray(cpu_level.prolongation.blocks, dtype=np.float64).reshape(-1),
            ),
            ("member_offsets", device_level.member_offsets, member_offsets),
            ("member_fine_nodes", device_level.member_fine_nodes, member_nodes),
        )
        for name, device_array, expected in optional_arrays:
            assert device_array is not None
            actual = np.asarray(device_array.numpy())
            _require_exact_array(actual, expected, name=f"hierarchy.level_{level_index}.{name}")
            parts.append((f"level_{level_index}.{name}", actual))
        if (
            device_level.omega != float(cpu_level.smoother.omega)
            or device_level.coarse_node_count != coarse_count
            or device_level.coarse_block_size != cpu_level.prolongation.coarse_block_size
        ):
            raise RuntimeError(f"persistent hierarchy level {level_index} transfer scalars changed")
        parts.extend(
            (
                (f"level_{level_index}.omega", np.asarray(device_level.omega, dtype=np.float64)),
                (f"level_{level_index}.coarse_node_count", int(device_level.coarse_node_count)),
                (f"level_{level_index}.coarse_block_size", int(device_level.coarse_block_size)),
            )
        )
    actual_factor = np.asarray(device_hierarchy.coarse_cholesky.numpy())
    expected_factor = np.asarray(hierarchy.coarse_cholesky, dtype=np.float64).reshape(-1)
    _require_exact_array(actual_factor, expected_factor, name="hierarchy.coarse_cholesky")
    parts.append(("coarse_cholesky", actual_factor))
    return _hash_parts("captured-direct-graph-vbd-device-hierarchy-inputs-v1", parts)


@dataclasses.dataclass(slots=True)
class _OuterWorkspace:
    rhs: wp.array[wp.vec3d]
    first_correction: wp.array[wp.vec3d]
    operator_product_after_first: wp.array[wp.vec3d]
    residual_after_first: wp.array[wp.vec3d]
    second_correction: wp.array[wp.vec3d]
    direction: wp.array[wp.vec3d]
    first_cycle: WarpVCycleWorkspace
    second_cycle: WarpVCycleWorkspace
    operator_apply: WarpMatrixFreeWorkspace


class CapturedDirectGraphVBD:
    """Persistent captured K1 plus fixed four-by-two direct graph lane."""

    def __init__(
        self,
        scene: TetBenchmarkScene,
        *,
        device: str = "cuda:0",
        config: DirectGraphVBDConfig | None = None,
        tile_solve: bool = False,
    ):
        if type(scene) is not TetBenchmarkScene:
            raise TypeError("scene must be an exact TetBenchmarkScene")
        self.config = DirectGraphVBDConfig() if config is None else config
        if type(self.config) is not DirectGraphVBDConfig:
            raise TypeError("config must be an exact DirectGraphVBDConfig")
        self.config.validate()
        if self.config.outer_corrections != OUTER_CORRECTIONS:
            raise ValueError("captured direct graph VBD requires exactly four outer corrections")
        if self.config.stationary_v_cycles != V_CYCLES_PER_OUTER or self.config.alpha != 1.0:
            raise ValueError("captured direct graph VBD requires exactly two stationary V-cycles at alpha=1")
        if self.config.coarse_node_limit != 4:
            raise ValueError("captured direct graph VBD freezes coarse_node_limit=4")
        self.config_sha256 = _canonical_digest(self.config.deterministic_record())
        self.scene = scene
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError("the captured direct graph VBD composition requires CUDA")
        self.baseline = CapturedPublicVBDBaseline(scene, device=str(self.device), tile_solve=tile_solve)
        reference_adjacency = self.baseline._lane(1).solver.particle_adjacency
        reference_adjacency_fields = dict(getattr(type(reference_adjacency._ctype), "_fields_", ()))
        self._array_descriptor_type_bound = reference_adjacency_fields["v_adj_tets"]
        self._array_descriptor_fields_bound = getattr(self._array_descriptor_type_bound, "_fields_", ())
        self._solver_lane_contract_sha256_bound = self._validate_solver_lane_contract()

        # This eager construction endpoint is used only to bind static arrays;
        # every execution below begins with a graph-scheduled pristine K1.
        k1 = self.baseline.run(1, graph_replay=False)
        self._construction_k1 = k1
        construction_k4, _fresh_k4 = self.baseline.validate_against_run_vbd(4, graph_replay=False)
        self._construction_k4 = construction_k4
        self.problem = build_common_problem(scene)
        self.scene_sha256 = _require_sha256(scene.manifest()["scene_sha256"], name="scene_sha256")
        self.objective_instance_sha256 = _require_sha256(
            common_objective_manifest(scene, self.problem)["objective_instance_sha256"],
            name="objective_instance_sha256",
        )
        oracle = MatrixFreeStableNHOperator.from_problem(self.problem, k1.positions)
        self._construction_operator = oracle
        self.hierarchy = build_stable_nh_rest_multigrid(
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
        self.device_hierarchy = WarpStaticMultigridHierarchy.from_hierarchy(self.hierarchy, device=str(self.device))

        n_vertices = scene.n_vertices
        n_tets = scene.n_tets
        n_free = int(oracle.free.size)
        self.workspaces = tuple(
            _OuterWorkspace(
                rhs=wp.empty(n_free, dtype=wp.vec3d, device=self.device),
                first_correction=wp.empty(n_free, dtype=wp.vec3d, device=self.device),
                operator_product_after_first=wp.empty(n_free, dtype=wp.vec3d, device=self.device),
                residual_after_first=wp.empty(n_free, dtype=wp.vec3d, device=self.device),
                second_correction=wp.empty(n_free, dtype=wp.vec3d, device=self.device),
                direction=wp.empty(n_free, dtype=wp.vec3d, device=self.device),
                first_cycle=self.device_hierarchy.create_workspace(),
                second_cycle=self.device_hierarchy.create_workspace(),
                operator_apply=self.operator.create_apply_workspace(),
            )
            for _ in range(OUTER_CORRECTIONS)
        )
        self.canonical_positions = wp.array(oracle.positions, dtype=wp.vec3d, device=self.device)
        self.x_current = wp.array(scene.x_current, dtype=wp.vec3d, device=self.device)
        self.candidate = wp.empty(n_vertices, dtype=wp.vec3d, device=self.device)
        self.proposal_finite = wp.empty(n_vertices, dtype=wp.int32, device=self.device)
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
        self._execution_serial = 0
        self._issued_execution_receipts: dict[int, _ExecutionReceipt] = {}
        self._issued_validation_contexts: dict[int, tuple[object, ...]] = {}
        self._capture_generation = 0
        self._captured_graph_object_identity: tuple[int, int] | None = None
        self._solver_graph_owner_identity_bound = self._solver_graph_owner_identity()
        self._solver_scalar_sha256_bound = self._solver_scalar_sha256()
        self._solver_static_array_sha256_bound = self._solver_static_array_sha256()
        self._persistent_array_identity = self._persistent_array_signatures()
        self._construction_content_identity = (
            self.scene_sha256,
            self.objective_instance_sha256,
            self.config_sha256,
            self.hierarchy.content_sha256,
            self.baseline.model_sha256,
            self.baseline.pristine_state_sha256,
            self._construction_k1.endpoint_sha256,
            self._construction_k1.position_sha256,
            self._construction_k1.velocity_sha256,
            self._construction_k4.endpoint_sha256,
            self._construction_k4.position_sha256,
            self._construction_k4.velocity_sha256,
        )
        self._persistent_device_sha256 = self._validate_persistent_sources(require_bound=False)
        self._construction_persistent_device_sha256 = self._persistent_device_sha256
        self._uncaptured_graph_identity_sha256 = self._graph_identity(captured=False, comparator=False)
        self.graph_identity_sha256 = self._graph_identity(captured=True, comparator=False)
        self.k4_graph_identity_sha256 = self._graph_identity(captured=True, comparator=True)

    @property
    def linear_kernel_launches_per_outer(self) -> int:
        """Exact direct linear-solver launches in each fixed outer slot."""
        return 7 + 2 * self.device_hierarchy.scheduled_kernel_launches

    @property
    def correction_kernel_launches(self) -> int:
        """Exact correction launches, excluding the public K1 graph prefix."""
        return 2 + OUTER_CORRECTIONS * (self.linear_kernel_launches_per_outer + 8)

    def _graph_identity(self, *, captured: bool, comparator: bool) -> str:
        """Bind one fixed launch schedule to all immutable execution inputs."""
        graph = self.k4_graph if comparator else self.graph
        return _canonical_digest(
            {
                "contract": "captured-direct-graph-vbd-graph-identity-v1",
                "solver_contract": CONTRACT_ID,
                "comparator_contract": VBD_BASELINE_CONTRACT_ID if comparator else None,
                "scene_sha256": self.scene_sha256,
                "objective_instance_sha256": self.objective_instance_sha256,
                "config_sha256": self.config_sha256,
                "static_hierarchy_sha256": self.hierarchy.content_sha256,
                "persistent_device_sha256": self._persistent_device_sha256,
                "captured": captured,
                "capture_generation": self._capture_generation if captured else 0,
                "graph_object_identity": id(graph) if captured and graph is not None else None,
                "lane": "public-k4" if comparator else "public-k1-plus-direct-four-by-two",
                "correction_kernel_launches": 0 if comparator else self.correction_kernel_launches,
            }
        )

    def _consume_execution_receipt(self, receipt: object) -> _ExecutionReceipt:
        """Consume exactly one registered receipt; copied or stale receipts fail closed."""
        if type(receipt) is not _ExecutionReceipt:
            raise RuntimeError("recording requires an exact solver-issued execution receipt")
        registered = self._issued_execution_receipts.pop(id(receipt), None)
        if registered is not receipt or receipt.issuer is not self:
            raise RuntimeError("execution receipt is forged, stale, or already consumed")
        if receipt.serial != self._execution_serial:
            raise RuntimeError("execution receipt is not the latest monotonic solver launch")
        expected_graph_objects = self._captured_graph_object_identity if receipt.graph_replay else None
        if receipt.graph_object_identity != expected_graph_objects:
            raise RuntimeError("execution receipt graph objects do not match the actual launch")
        return receipt

    def _validate_issued_context(
        self,
        context: _EndpointValidationContext,
        *,
        validate_raw_sources: bool,
    ) -> None:
        """Validate one exact registered execution context against its live owner."""
        receipt = self._issued_validation_contexts.get(id(context))
        if receipt is None or receipt[0] is not context:
            raise ValueError("validation context is not the exact live solver-issued object")
        (
            _registered_context,
            execution_serial,
            graph_replay,
            execution_k1,
            persistent_device_sha256,
            graph_identity_sha256,
            graph_object_identity,
            outer_slots,
            outer_slot_records,
        ) = receipt
        if (
            context.issuer is not self
            or context.execution_serial != execution_serial
            or context.graph_replay != graph_replay
            or context.scene is not self.scene
            or context.config is not self.config
            or context.construction_k1 is not self._construction_k1
            or context.execution_k1 is not execution_k1
            or context.hierarchy is not self.hierarchy
            or context.persistent_device_sha256 != persistent_device_sha256
            or context.graph_identity_sha256 != graph_identity_sha256
            or context.device != str(self.device)
            or context.outer_slots is not outer_slots
            or tuple(
                (
                    slot.outer_index,
                    slot.start_position_sha256,
                    slot.current_operator_sha256,
                    slot.rhs_sha256,
                    slot.accepted,
                    slot.reason,
                )
                for slot in context.outer_slots
            )
            != outer_slot_records
        ):
            raise ValueError("validation context fields do not match their solver-issued receipt")
        expected_graph_objects = self._captured_graph_object_identity if graph_replay else None
        if graph_object_identity != expected_graph_objects:
            raise ValueError("validation context graph objects are stale")
        expected_graph_identity = self._graph_identity(captured=graph_replay, comparator=False)
        if context.graph_identity_sha256 != expected_graph_identity:
            raise ValueError("validation context graph identity is not canonically derived")
        expected_warp_snapshot_sha256 = _hash_parts(
            "warp-static-multigrid-snapshot-v1",
            (
                ("hierarchy_sha256", self.hierarchy.content_sha256),
                ("kernel_version", V_CYCLE_KERNEL_VERSION),
                ("coarse_scalar_bound", MAX_COARSE_SCALAR_SIZE),
            ),
        )
        noncoarse = len(self.hierarchy.levels) - 1
        expected_v_cycle_kernel_launches = 3 + noncoarse * (
            5 + 3 * (self.hierarchy.pre_smooth_steps + self.hierarchy.post_smooth_steps)
        )
        if context.warp_snapshot_sha256 != expected_warp_snapshot_sha256:
            raise ValueError("validation context Warp snapshot identity is not canonical")
        if context.v_cycle_kernel_launches != expected_v_cycle_kernel_launches:
            raise ValueError("validation context V-cycle launch count is not canonical")
        if validate_raw_sources:
            current_persistent_sha256 = self._validate_persistent_sources()
            if current_persistent_sha256 != context.persistent_device_sha256:
                raise ValueError("validation context persistent inputs are no longer live")

    def _validate_issued_outer_slot(
        self,
        context: _EndpointValidationContext,
        slot: _OuterSlotBinding,
    ) -> None:
        """Bind standalone outer work to one exact registered execution slot."""
        receipt = self._issued_validation_contexts.get(id(context))
        if receipt is None or receipt[0] is not context:
            raise ValueError("outer slot has no live owning execution context")
        outer_slots = receipt[7]
        if (
            type(slot.outer_index) is not int
            or not 0 <= slot.outer_index < OUTER_CORRECTIONS
            or outer_slots[slot.outer_index] is not slot
            or context.outer_slots[slot.outer_index] is not slot
        ):
            raise ValueError("outer work does not use its exact solver-issued schedule slot")

    def _solver_graph_array_items(self) -> tuple[tuple[str, wp.array[Any]], ...]:
        """Enumerate all lane-solver, adjacency, control, model, and state arrays."""
        arrays: list[tuple[str, wp.array[Any]]] = []
        arrays.extend(_attribute_array_items(self.baseline.model, prefix="baseline.model"))
        arrays.extend(_attribute_array_items(self.baseline.control, prefix="baseline.control"))
        for iterations in (1, 4):
            lane = self.baseline._lane(iterations)
            arrays.extend(_attribute_array_items(lane.solver, prefix=f"lane_{iterations}.solver"))
            arrays.extend(
                _attribute_array_items(
                    lane.solver.particle_adjacency,
                    prefix=f"lane_{iterations}.solver.particle_adjacency",
                )
            )
            arrays.extend(_attribute_array_items(lane.state_in, prefix=f"lane_{iterations}.state_in"))
            arrays.extend(_attribute_array_items(lane.state_out, prefix=f"lane_{iterations}.state_out"))
        return tuple(sorted(arrays, key=lambda item: item[0]))

    @staticmethod
    def _validate_particle_adjacency_descriptor(adjacency: object, *, iterations: int) -> str:
        """Bind every cached adjacency struct field to its public Warp array."""
        descriptor = getattr(adjacency, "_ctype", None)
        try:
            regenerated = adjacency.__ctype__()
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(f"public K{iterations} adjacency cached C struct is malformed") from error
        if descriptor is None or regenerated is not descriptor:
            raise RuntimeError(f"public K{iterations} adjacency cached C struct object changed")
        fields = getattr(type(descriptor), "_fields_", ())
        if tuple(field[0] for field in fields) != _PARTICLE_ADJACENCY_ARRAY_NAMES:
            raise RuntimeError(f"public K{iterations} adjacency cached C struct fields changed")

        records: list[dict[str, object]] = []
        for name, field_type in fields:
            field_descriptor = getattr(descriptor, name, None)
            if type(field_descriptor) is not field_type:
                raise RuntimeError(f"public K{iterations} adjacency cached C field {name} type changed")
            array = getattr(adjacency, name, None)
            if not isinstance(array, wp.array):
                raise RuntimeError(f"public K{iterations} adjacency field {name} lost its Warp array")
            data, gradient, ndim, shape, strides = _validate_descriptor_matches_array(
                field_descriptor,
                array,
                name=f"public K{iterations} adjacency._ctype.{name}",
                expected_type=field_type,
                expected_fields=getattr(field_type, "_fields_", ()),
            )
            records.append(
                {
                    "name": name,
                    "data": data,
                    "gradient": gradient,
                    "ndim": ndim,
                    "shape": list(shape),
                    "strides": list(strides),
                    "dtype": str(array.dtype),
                    "device": str(array.device),
                }
            )
        return _canonical_digest(
            {
                "contract": "captured-direct-graph-vbd-adjacency-ctype-v1",
                "iterations": iterations,
                "fields": records,
            }
        )

    def _validate_solver_lane_contract(self) -> str:
        """Validate the exact public K1/K4 solvers and their graph topology."""
        if type(self.baseline) is not CapturedPublicVBDBaseline:
            raise RuntimeError("captured VBD baseline object type changed")
        if type(self.baseline.tile_solve) is not bool:
            raise RuntimeError("captured VBD tile-solve policy must remain a bool")
        model = self.baseline.model
        if wp.get_device(self.baseline.device) != self.device or wp.get_device(model.device) != self.device:
            raise RuntimeError("captured VBD baseline/model device changed")
        expected_counts = {
            "edges": (int(model.edge_count), 4),
            "faces": (int(model.tri_count), 6),
            "springs": (int(model.spring_count), 2),
            "tets": (int(model.tet_count), 8),
        }
        if (
            int(model.particle_count) != self.scene.n_vertices
            or int(model.tri_count) != self.scene.n_triangles
            or int(model.tet_count) != self.scene.n_tets
            or any(count < 0 for count, _ in expected_counts.values())
        ):
            raise RuntimeError("captured VBD model topology counts changed")

        color_groups = tuple(model.particle_color_groups)
        color_vertices = (
            np.concatenate([np.asarray(group.numpy(), dtype=np.int64) for group in color_groups])
            if color_groups
            else np.empty(0, dtype=np.int64)
        )
        if color_vertices.size != self.scene.n_vertices or not np.array_equal(
            np.sort(color_vertices), np.arange(self.scene.n_vertices, dtype=np.int64)
        ):
            raise RuntimeError("captured VBD color groups no longer partition the particles")

        reference_adjacency: dict[str, np.ndarray] = {}
        lane_records: list[dict[str, object]] = []
        for iterations in (1, 4):
            lane = self.baseline._lane(iterations)
            solver = lane.solver
            if type(solver) is not SolverVBD:
                raise RuntimeError(f"public K{iterations} solver type changed")
            if (
                type(lane.iterations) is not int
                or lane.iterations != iterations
                or type(solver.iterations) is not int
                or solver.iterations != iterations
            ):
                raise RuntimeError(f"public K{iterations} solver iteration schedule changed")
            if solver.model is not model or wp.get_device(solver.device) != self.device:
                raise RuntimeError(f"public K{iterations} solver model/device binding changed")
            if solver.particle_enable_self_contact is not False:
                raise RuntimeError(f"public K{iterations} solver enabled unsupported self-contact")
            if solver.use_particle_tile_solve is not self.baseline.tile_solve:
                raise RuntimeError(f"public K{iterations} solver tile policy changed")

            for state_name, state in (("state_in", lane.state_in), ("state_out", lane.state_out)):
                for field_name in ("particle_q", "particle_qd", "particle_f"):
                    array = getattr(state, field_name, None)
                    if (
                        not isinstance(array, wp.array)
                        or tuple(array.shape) != (self.scene.n_vertices,)
                        or array.dtype is not wp.vec3
                        or wp.get_device(array.device) != self.device
                    ):
                        raise RuntimeError(f"public K{iterations} {state_name}.{field_name} topology/device changed")

            for name, dtype in _PARTICLE_SCRATCH_ARRAY_SPECS:
                array = getattr(solver, name, None)
                if (
                    not isinstance(array, wp.array)
                    or tuple(array.shape) != (self.scene.n_vertices,)
                    or array.dtype is not dtype
                    or wp.get_device(array.device) != self.device
                ):
                    raise RuntimeError(f"public K{iterations} solver scratch {name} changed")
            particle_q_rest = getattr(solver, "particle_q_rest", None)
            if (
                not isinstance(particle_q_rest, wp.array)
                or tuple(particle_q_rest.shape) != (self.scene.n_vertices,)
                or particle_q_rest.dtype is not wp.vec3
                or wp.get_device(particle_q_rest.device) != self.device
            ):
                raise RuntimeError(f"public K{iterations} solver rest-state topology/device changed")

            adjacency = solver.particle_adjacency
            adjacency_descriptor_sha256 = self._validate_particle_adjacency_descriptor(
                adjacency,
                iterations=iterations,
            )
            for name in _PARTICLE_ADJACENCY_ARRAY_NAMES:
                array = getattr(adjacency, name, None)
                topology = name.removeprefix("v_adj_").removesuffix("_offsets")
                topology_count, packed_width = expected_counts[topology]
                expected_size = (
                    self.scene.n_vertices + 1
                    if name.endswith("_offsets") and topology_count > 0
                    else packed_width * topology_count
                    if not name.endswith("_offsets")
                    else 0
                )
                if (
                    not isinstance(array, wp.array)
                    or tuple(array.shape) != (expected_size,)
                    or array.dtype is not wp.int32
                    or wp.get_device(array.device) != self.device
                ):
                    raise RuntimeError(f"public K{iterations} adjacency {name} topology/device changed")
                host = np.asarray(array.numpy(), dtype=np.int32)
                if (
                    name.endswith("_offsets")
                    and host.size
                    and (
                        int(host[0]) != 0 or int(host[-1]) != packed_width * topology_count or np.any(np.diff(host) < 0)
                    )
                ):
                    raise RuntimeError(f"public K{iterations} adjacency {name} offsets are invalid")
                if iterations == 1:
                    reference_adjacency[name] = host.copy()
                elif not np.array_equal(host, reference_adjacency[name]):
                    raise RuntimeError(f"public K4 adjacency {name} differs from public K1")

            lane_records.append(
                {
                    "iterations": iterations,
                    "solver_type": f"{type(solver).__module__}.{type(solver).__qualname__}",
                    "device": str(solver.device),
                    "self_contact": solver.particle_enable_self_contact,
                    "tile_solve": solver.use_particle_tile_solve,
                    "adjacency_descriptor_sha256": adjacency_descriptor_sha256,
                }
            )
        return _canonical_digest(
            {
                "contract": "captured-direct-graph-vbd-public-lanes-v1",
                "device": str(self.device),
                "requested_tile_solve": self.baseline.tile_solve,
                "particle_count": self.scene.n_vertices,
                "triangle_count": self.scene.n_triangles,
                "tetrahedron_count": self.scene.n_tets,
                "topology_counts": {name: count for name, (count, _width) in expected_counts.items()},
                "lanes": lane_records,
            }
        )

    def _solver_graph_owner_identity(self) -> tuple[tuple[str, int], ...]:
        """Bind every public owner object used to resolve captured lane allocations."""
        owners: list[tuple[str, object]] = [
            ("baseline", self.baseline),
            ("baseline._lanes", self.baseline._lanes),
            ("baseline.model", self.baseline.model),
            ("baseline.control", self.baseline.control),
            ("baseline.pristine_input", self.baseline.pristine_input),
            ("baseline.pristine_output", self.baseline.pristine_output),
        ]
        for iterations in (1, 4):
            lane = self.baseline._lane(iterations)
            adjacency = lane.solver.particle_adjacency
            owners.extend(
                (
                    (f"lane_{iterations}", lane),
                    (f"lane_{iterations}.solver", lane.solver),
                    (f"lane_{iterations}.solver.model", lane.solver.model),
                    (f"lane_{iterations}.solver.particle_adjacency", adjacency),
                    (f"lane_{iterations}.solver.particle_adjacency._ctype", adjacency._ctype),
                    (f"lane_{iterations}.state_in", lane.state_in),
                    (f"lane_{iterations}.state_out", lane.state_out),
                )
            )
        return tuple((name, id(value)) for name, value in owners)

    @staticmethod
    def _primitive_attribute_record(value: object, *, prefix: str) -> list[dict[str, object]]:
        """Serialize direct primitive settings that control graph launch behavior."""
        records: list[dict[str, object]] = []
        for name, child in sorted(vars(value).items()):
            if child is None or type(child) in (bool, int, float, str):
                if type(child) is float and not math.isfinite(child):
                    raise RuntimeError(f"persistent solver scalar {prefix}.{name} became nonfinite")
                records.append(
                    {
                        "name": f"{prefix}.{name}",
                        "type": type(child).__name__,
                        "value": child,
                    }
                )
        return records

    def _solver_scalar_sha256(self) -> str:
        """Hash public scalar settings consumed when either VBD lane is enqueued."""
        records = self._primitive_attribute_record(self.baseline.model, prefix="baseline.model")
        records.extend(self._primitive_attribute_record(self.baseline.control, prefix="baseline.control"))
        for iterations in (1, 4):
            records.extend(
                self._primitive_attribute_record(
                    self.baseline._lane(iterations).solver,
                    prefix=f"lane_{iterations}.solver",
                )
            )
        return _canonical_digest({"contract": "captured-direct-graph-vbd-solver-scalars-v1", "fields": records})

    def _solver_static_array_sha256(self) -> str:
        """Hash static adjacency, rest-state, and control arrays for both lanes."""
        arrays: list[tuple[str, np.ndarray]] = []
        for name, array in _attribute_array_items(self.baseline.control, prefix="baseline.control"):
            arrays.append((name, np.asarray(array.numpy())))
        for iterations in (1, 4):
            solver = self.baseline._lane(iterations).solver
            for name, value in sorted(vars(solver).items()):
                if isinstance(value, wp.array) and name not in _SOLVER_SCRATCH_ARRAY_NAMES:
                    arrays.append((f"lane_{iterations}.solver.{name}", np.asarray(value.numpy())))
            arrays.extend(
                (name, np.asarray(array.numpy()))
                for name, array in _attribute_array_items(
                    solver.particle_adjacency,
                    prefix=f"lane_{iterations}.solver.particle_adjacency",
                )
            )
        return _hash_parts("captured-direct-graph-vbd-solver-static-arrays-v1", tuple(arrays))

    def _persistent_input_arrays(self) -> tuple[tuple[str, wp.array[Any]], ...]:
        """Return every array whose allocation is embedded in either captured graph."""
        model = self.baseline.model
        model_arrays = (
            ("model.particle_q", model.particle_q),
            ("model.particle_qd", model.particle_qd),
            ("model.particle_mass", model.particle_mass),
            ("model.particle_inv_mass", model.particle_inv_mass),
            ("model.particle_flags", model.particle_flags),
            ("model.tet_indices", model.tet_indices),
            ("model.tet_poses", model.tet_poses),
            ("model.tet_materials", model.tet_materials),
            ("model.tri_indices", model.tri_indices),
            ("model.tri_poses", model.tri_poses),
            ("model.tri_materials", model.tri_materials),
            ("model.tri_areas", model.tri_areas),
            ("model.gravity", model.gravity),
            *(
                (f"model.particle_color_group_{index}", value)
                for index, value in enumerate(model.particle_color_groups)
            ),
        )
        pristine_arrays = tuple(
            (f"pristine.{state_name}.{field_name}", getattr(state, field_name))
            for state_name, state in (
                ("input", self.baseline.pristine_input),
                ("output", self.baseline.pristine_output),
            )
            for field_name in ("particle_q", "particle_qd", "particle_f")
        )
        lane_arrays = tuple(
            (f"lane_{iterations}.{state_name}.{field_name}", getattr(state, field_name))
            for iterations in (1, 4)
            for state_name, state in (
                ("state_in", self.baseline._lane(iterations).state_in),
                ("state_out", self.baseline._lane(iterations).state_out),
            )
            for field_name in ("particle_q", "particle_qd", "particle_f")
        )
        solver_graph_arrays = self._solver_graph_array_items()
        operator_arrays = tuple(
            (f"operator.{name}", getattr(self.operator, name))
            for name in (
                "tets",
                "shape_gradients",
                "volumes",
                "mass",
                "mu",
                "lam",
                "inertial_target",
                "free",
                "vertex_to_free",
                "incidence_offsets",
                "incidence_tets",
                "incidence_corners",
            )
        )
        hierarchy_arrays: list[tuple[str, wp.array[Any]]] = []
        for level_index, level in enumerate(self.device_hierarchy.levels):
            for name in (
                "row_offsets",
                "column_indices",
                "matrix_values",
                "inverse_diagonal",
                "aggregate",
                "prolongation_blocks",
                "member_offsets",
                "member_fine_nodes",
            ):
                value = getattr(level, name)
                if value is not None:
                    hierarchy_arrays.append((f"hierarchy.level_{level_index}.{name}", value))
        hierarchy_arrays.append(("hierarchy.coarse_cholesky", self.device_hierarchy.coarse_cholesky))
        correction_arrays = (
            ("operator.positions", self.operator.positions),
            ("operator.deformation_gradients", self.operator.deformation_gradients),
            ("operator.cofactors", self.operator.cofactors),
            ("operator.determinants", self.operator.determinants),
            ("operator.first_piola", self.operator.first_piola),
            ("candidate", self.candidate),
            ("proposal_finite", self.proposal_finite),
            ("final_positions", self.final_positions),
            ("final_velocities", self.final_velocities),
            ("active", self.active),
            ("accepted", self.accepted),
            ("reasons", self.reasons),
            ("current_inertia", self.current_inertia),
            ("candidate_inertia", self.candidate_inertia),
            ("vertex_finite", self.vertex_finite),
            ("current_elastic", self.current_elastic),
            ("candidate_elastic", self.candidate_elastic),
            ("candidate_determinants", self.candidate_determinants),
            ("segment_minima", self.segment_minima),
            ("tet_finite", self.tet_finite),
            ("directional_terms", self.directional_terms),
            ("initial_objectives", self.initial_objectives),
            ("candidate_objectives", self.candidate_objectives),
            ("directional_derivatives", self.directional_derivatives),
            ("minimum_segment_determinants", self.minimum_segment_determinants),
            *((f"outer_start_positions_{index}", value) for index, value in enumerate(self.outer_start_positions)),
            *(
                (f"outer_candidate_positions_{index}", value)
                for index, value in enumerate(self.outer_candidate_positions)
            ),
        )
        workspace_arrays: list[tuple[str, wp.array[Any]]] = []
        for outer_index, workspace in enumerate(self.workspaces):
            for name in (
                "rhs",
                "first_correction",
                "operator_product_after_first",
                "residual_after_first",
                "second_correction",
                "direction",
            ):
                workspace_arrays.append((f"workspace_{outer_index}.{name}", getattr(workspace, name)))
            workspace_arrays.append(
                (f"workspace_{outer_index}.operator_apply.delta_piola", workspace.operator_apply.delta_piola)
            )
            for cycle_name, cycle in (("first_cycle", workspace.first_cycle), ("second_cycle", workspace.second_cycle)):
                workspace_arrays.extend(
                    (
                        (f"workspace_{outer_index}.{cycle_name}.rhs", cycle.rhs),
                        (f"workspace_{outer_index}.{cycle_name}.correction", cycle.correction),
                        (f"workspace_{outer_index}.{cycle_name}.coarse_intermediate", cycle.coarse_intermediate),
                    )
                )
                for sequence_name in ("level_rhs", "level_correction", "level_product", "level_residual"):
                    workspace_arrays.extend(
                        (f"workspace_{outer_index}.{cycle_name}.{sequence_name}_{level_index}", value)
                        for level_index, value in enumerate(getattr(cycle, sequence_name))
                    )
        return tuple(
            (name, value)
            for name, value in (
                *model_arrays,
                *pristine_arrays,
                *lane_arrays,
                *solver_graph_arrays,
                *operator_arrays,
                ("canonical_positions", self.canonical_positions),
                ("x_current", self.x_current),
                *hierarchy_arrays,
                *correction_arrays,
                *workspace_arrays,
            )
            if value is not None
        )

    def _persistent_array_signatures(self) -> tuple[tuple[str, object], ...]:
        """Bind allocation identity, null-safe pointer, and launch-relevant metadata."""
        return tuple(
            (
                name,
                (
                    id(value),
                    _array_pointer(value),
                    tuple(value.shape),
                    str(value.dtype),
                    str(value.device),
                    _canonical_array_descriptor(
                        value,
                        name=name,
                        expected_type=self._array_descriptor_type_bound,
                        expected_fields=self._array_descriptor_fields_bound,
                    ),
                ),
            )
            for name, value in self._persistent_input_arrays()
        )

    def _validate_persistent_sources(self, *, require_bound: bool = True) -> str:
        """Synchronously reject any stale model, reset, operator, or hierarchy input."""
        current_content_identity = (
            self.scene_sha256,
            self.objective_instance_sha256,
            self.config_sha256,
            self.hierarchy.content_sha256,
            self.baseline.model_sha256,
            self.baseline.pristine_state_sha256,
            self._construction_k1.endpoint_sha256,
            self._construction_k1.position_sha256,
            self._construction_k1.velocity_sha256,
            self._construction_k4.endpoint_sha256,
            self._construction_k4.position_sha256,
            self._construction_k4.velocity_sha256,
        )
        if current_content_identity != self._construction_content_identity:
            raise RuntimeError("captured construction content identity labels changed")
        if hasattr(self, "_persistent_array_identity") and (
            self._persistent_array_signatures() != self._persistent_array_identity
        ):
            raise RuntimeError("a persistent captured input array allocation or pointer changed")
        lane_contract_sha256 = self._validate_solver_lane_contract()
        if lane_contract_sha256 != self._solver_lane_contract_sha256_bound:
            raise RuntimeError("public K1/K4 SolverVBD lane schedule changed")
        if self._solver_graph_owner_identity() != self._solver_graph_owner_identity_bound:
            raise RuntimeError("persistent SolverVBD lane/control/state owner object changed")
        solver_scalar_sha256 = self._solver_scalar_sha256()
        if solver_scalar_sha256 != self._solver_scalar_sha256_bound:
            raise RuntimeError("persistent SolverVBD lane scalar configuration changed")
        solver_static_array_sha256 = self._solver_static_array_sha256()
        if solver_static_array_sha256 != self._solver_static_array_sha256_bound:
            raise RuntimeError("persistent SolverVBD static input or adjacency content changed")
        if self._captured_graph_object_identity is not None and (
            self.graph is None
            or self.k4_graph is None
            or (id(self.graph), id(self.k4_graph)) != self._captured_graph_object_identity
        ):
            raise RuntimeError("a captured graph object changed after capture")
        if type(self.scene) is not TetBenchmarkScene or type(self.config) is not DirectGraphVBDConfig:
            raise RuntimeError("captured direct graph canonical scene/config objects changed")
        self.config.validate()
        scene_sha256 = _require_sha256(self.scene.manifest()["scene_sha256"], name="scene_sha256")
        config_sha256 = _canonical_digest(self.config.deterministic_record())
        if scene_sha256 != self.scene_sha256 or config_sha256 != self.config_sha256:
            raise RuntimeError("captured direct graph scene or configuration identity changed")
        if self.baseline.scene_sha256 != self.scene_sha256:
            raise RuntimeError("captured direct graph public baseline belongs to another scene")
        model_sha256 = _public_model_sha256(self.baseline.model)
        if model_sha256 != self.baseline.model_sha256:
            raise RuntimeError("public static model changed after captured direct graph construction")
        pristine_sha256 = self.baseline._record_pristine_state_sha256()
        if pristine_sha256 != self.baseline.pristine_state_sha256:
            raise RuntimeError("persistent pristine input state was mutated")
        _validate_k1_endpoint(
            self._construction_k1,
            self._construction_k1,
            self.scene,
            device=str(self.device),
            graph_replay=False,
        )
        _validate_k4_endpoint(
            self._construction_k4,
            self._construction_k4,
            self.scene,
            device=str(self.device),
            graph_replay=False,
        )

        canonical_problem = build_common_problem(self.scene)
        objective_sha256 = _require_sha256(
            common_objective_manifest(self.scene, canonical_problem)["objective_instance_sha256"],
            name="objective_instance_sha256",
        )
        retained_objective_sha256 = _require_sha256(
            common_objective_manifest(self.scene, self.problem)["objective_instance_sha256"],
            name="retained_objective_instance_sha256",
        )
        if objective_sha256 != self.objective_instance_sha256 or retained_objective_sha256 != objective_sha256:
            raise RuntimeError("captured direct graph retained common problem changed")
        canonical_operator = MatrixFreeStableNHOperator.from_problem(canonical_problem, self._construction_k1.positions)
        if _operator_sha256(canonical_operator) != _operator_sha256(self._construction_operator):
            raise RuntimeError("captured direct graph construction operator changed")
        _, canonical_hierarchy = _canonical_static_hierarchy(
            canonical_operator,
            self.hierarchy,
            self.scene.rest_q,
            self.config,
        )
        operator_inputs_sha256 = _validate_device_operator_inputs(
            self.operator,
            canonical_operator,
            self.canonical_positions,
            self.x_current,
            self.scene,
        )
        hierarchy_inputs_sha256 = _validate_device_hierarchy_inputs(self.device_hierarchy, canonical_hierarchy)
        persistent_sha256 = _hash_parts(
            "captured-direct-graph-vbd-persistent-inputs-v1",
            (
                ("scene_sha256", scene_sha256),
                ("objective_instance_sha256", objective_sha256),
                ("config_sha256", config_sha256),
                ("public_model_sha256", model_sha256),
                ("pristine_state_sha256", pristine_sha256),
                ("construction_k1_endpoint_sha256", self._construction_k1.endpoint_sha256),
                ("construction_k4_endpoint_sha256", self._construction_k4.endpoint_sha256),
                ("solver_lane_contract_sha256", lane_contract_sha256),
                ("solver_scalar_sha256", solver_scalar_sha256),
                ("solver_static_array_sha256", solver_static_array_sha256),
                ("construction_operator_sha256", _operator_sha256(canonical_operator)),
                ("operator_inputs_sha256", operator_inputs_sha256),
                ("hierarchy_inputs_sha256", hierarchy_inputs_sha256),
            ),
        )
        if require_bound and (
            persistent_sha256 != self._persistent_device_sha256
            or self._persistent_device_sha256 != self._construction_persistent_device_sha256
        ):
            raise RuntimeError("captured direct graph persistent input digest changed")
        if require_bound:
            expected_uncaptured_identity = self._graph_identity(captured=False, comparator=False)
            expected_graph_identity = self._graph_identity(captured=True, comparator=False)
            expected_k4_identity = self._graph_identity(captured=True, comparator=True)
            if (
                self._uncaptured_graph_identity_sha256 != expected_uncaptured_identity
                or self.graph_identity_sha256 != expected_graph_identity
                or self.k4_graph_identity_sha256 != expected_k4_identity
            ):
                raise RuntimeError("captured graph identity label is stale or was mutated")
        return persistent_sha256

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
                self.proposal_finite,
                self.active,
                self.accepted,
                self.reasons,
            ],
            device=self.device,
        )
        for outer_index, workspace in enumerate(self.workspaces):
            wp.launch(
                _copy_positions,
                dim=self.scene.n_vertices,
                inputs=[self.operator.positions, self.outer_start_positions[outer_index]],
                device=self.device,
            )
            self.operator.launch_refresh_geometry()
            self.operator.launch_gradient(workspace.rhs, scale=-1.0)
            wp.launch(_mask_rhs, dim=self.operator.n_free, inputs=[self.active, workspace.rhs], device=self.device)
            self.device_hierarchy.launch_apply(
                workspace.rhs,
                workspace.first_correction,
                workspace.first_cycle,
            )
            self.operator.launch_apply(
                workspace.first_correction,
                workspace.operator_product_after_first,
                workspace.operator_apply,
            )
            wp.launch(
                _subtract_vectors,
                dim=self.operator.n_free,
                inputs=[workspace.rhs, workspace.operator_product_after_first, workspace.residual_after_first],
                device=self.device,
            )
            self.device_hierarchy.launch_apply(
                workspace.residual_after_first,
                workspace.second_correction,
                workspace.second_cycle,
            )
            wp.launch(
                _add_vectors,
                dim=self.operator.n_free,
                inputs=[workspace.first_correction, workspace.second_correction, workspace.direction],
                device=self.device,
            )
            wp.launch(
                _build_candidate,
                dim=self.scene.n_vertices,
                inputs=[
                    self.operator.positions,
                    self.operator.vertex_to_free,
                    workspace.direction,
                    self.active,
                    self.candidate,
                    self.proposal_finite,
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
                    workspace.rhs,
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
                    self.current_inertia,
                    self.candidate_inertia,
                    self.current_elastic,
                    self.candidate_elastic,
                    self.directional_terms,
                    self.candidate_determinants,
                    self.segment_minima,
                    self.proposal_finite,
                    self.vertex_finite,
                    self.tet_finite,
                    self.config.minimum_determinant,
                    self.config.armijo,
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
        """Capture separate integrated direct-graph and pristine K4 graphs."""
        if isinstance(warmup_replays, bool) or not isinstance(warmup_replays, numbers.Integral) or warmup_replays < 1:
            raise ValueError("warmup_replays must be a positive integer")
        self._validate_persistent_sources()
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
        self._captured_graph_object_identity = (id(self.graph), id(self.k4_graph))
        self._capture_generation += 1
        self.graph_identity_sha256 = self._graph_identity(captured=True, comparator=False)
        self.k4_graph_identity_sha256 = self._graph_identity(captured=True, comparator=True)
        self._validate_persistent_sources()

    def run(self, *, graph_replay: bool = True) -> CapturedGraphVBDEndpoint:
        """Execute and synchronize one integrated direct graph lane."""
        if type(graph_replay) is not bool:
            raise ValueError("graph_replay must be a bool")
        self._validate_persistent_sources()
        if graph_replay:
            if self.graph is None:
                raise RuntimeError("capture_graphs() must complete before graph replay")
            wp.capture_launch(self.graph)
        else:
            self._enqueue_integrated()
        self.baseline._lane(1).completed_launches += 1
        self._execution_serial += 1
        receipt = _ExecutionReceipt(
            issuer=self,
            serial=self._execution_serial,
            graph_replay=graph_replay,
            graph_object_identity=self._captured_graph_object_identity if graph_replay else None,
        )
        self._issued_execution_receipts[id(receipt)] = receipt
        return self._record(execution_receipt=receipt)

    def run_k4(self, *, graph_replay: bool = True):
        """Execute the separate pristine K4 comparator lane."""
        if type(graph_replay) is not bool:
            raise ValueError("graph_replay must be a bool")
        self._validate_persistent_sources()
        lane = self.baseline._lane(4)
        if graph_replay:
            if self.k4_graph is None:
                raise RuntimeError("capture_graphs() must complete before graph replay")
            wp.capture_launch(self.k4_graph)
        else:
            self.baseline._enqueue_reset_and_step(lane)
        lane.completed_launches += 1
        endpoint = self.baseline.record(4, graph_replay=graph_replay)
        _validate_k4_endpoint(
            endpoint,
            self._construction_k4,
            self.scene,
            device=str(self.device),
            graph_replay=graph_replay,
        )
        self._validate_persistent_sources()
        return endpoint

    def benchmark_paired(
        self,
        *,
        pair_count: int = 10,
        warmup_replays: int = 5,
        random_seed: int = 20260817,
    ) -> CapturedGraphVBDTiming:
        """Measure captured direct graph VBD and K4 in balanced AB/BA order."""
        if self.graph is None or self.k4_graph is None:
            raise RuntimeError("capture_graphs() must complete before timing")
        for name, value, minimum in (("pair_count", pair_count, 2), ("warmup_replays", warmup_replays, 1)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if int(pair_count) % 2 != 0:
            raise ValueError("pair_count must be even for balanced AB/BA timing")
        if isinstance(random_seed, bool) or not isinstance(random_seed, numbers.Integral):
            raise ValueError("random_seed must be an integer")
        persistent_device_sha256 = self._validate_persistent_sources()
        graph_identity_sha256 = self._graph_identity(captured=True, comparator=False)
        k4_graph_identity_sha256 = self._graph_identity(captured=True, comparator=True)

        for _ in range(int(warmup_replays)):
            wp.capture_launch(self.graph)
            wp.capture_launch(self.k4_graph)
        wp.synchronize_device(self.device)

        orders = ["AB" if index % 2 == 0 else "BA" for index in range(int(pair_count))]
        np.random.default_rng(int(random_seed)).shuffle(orders)
        graph_events = [
            (wp.Event(self.device, enable_timing=True), wp.Event(self.device, enable_timing=True)) for _ in orders
        ]
        k4_events = [
            (wp.Event(self.device, enable_timing=True), wp.Event(self.device, enable_timing=True)) for _ in orders
        ]
        for pair_index, order in enumerate(orders):
            if order == "AB":
                begin, end = graph_events[pair_index]
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
                begin, end = graph_events[pair_index]
                wp.record_event(begin)
                wp.capture_launch(self.graph)
                wp.record_event(end)
        wp.synchronize_event(graph_events[-1][1])
        wp.synchronize_event(k4_events[-1][1])
        if self._validate_persistent_sources() != persistent_device_sha256:
            raise RuntimeError("persistent captured inputs changed during paired timing")
        return CapturedGraphVBDTiming(
            pair_orders=tuple(orders),
            graph_seconds=tuple(
                float(wp.get_event_elapsed_time(begin, end, synchronize=False)) * 1.0e-3 for begin, end in graph_events
            ),
            k4_seconds=tuple(
                float(wp.get_event_elapsed_time(begin, end, synchronize=False)) * 1.0e-3 for begin, end in k4_events
            ),
            warmup_replays=int(warmup_replays),
            random_seed=int(random_seed),
            device=str(self.device),
            contract_id=CONTRACT_ID,
            scene_sha256=self.scene_sha256,
            objective_instance_sha256=self.objective_instance_sha256,
            config_sha256=self.config_sha256,
            static_hierarchy_sha256=self.hierarchy.content_sha256,
            persistent_device_sha256=persistent_device_sha256,
            graph_identity_sha256=graph_identity_sha256,
            k4_graph_identity_sha256=k4_graph_identity_sha256,
            comparator_contract_id=VBD_BASELINE_CONTRACT_ID,
        )

    def _record(self, *, execution_receipt: object) -> CapturedGraphVBDEndpoint:
        """Consume one solver-private execution and materialize validated evidence."""
        receipt = self._consume_execution_receipt(execution_receipt)
        graph_replay = receipt.graph_replay
        execution_k1 = self.baseline.record(1, graph_replay=graph_replay)
        persistent_device_sha256 = self._validate_persistent_sources()
        graph_identity_sha256 = self._graph_identity(captured=graph_replay, comparator=False)
        warp_snapshot_sha256 = _hash_parts(
            "warp-static-multigrid-snapshot-v1",
            (
                ("hierarchy_sha256", self.hierarchy.content_sha256),
                ("kernel_version", V_CYCLE_KERNEL_VERSION),
                ("coarse_scalar_bound", MAX_COARSE_SCALAR_SIZE),
            ),
        )
        noncoarse = len(self.hierarchy.levels) - 1
        v_cycle_kernel_launches = 3 + noncoarse * (
            5 + 3 * (self.hierarchy.pre_smooth_steps + self.hierarchy.post_smooth_steps)
        )
        positions = np.asarray(self.final_positions.numpy(), dtype=np.float32).astype(np.float64)
        velocities = np.asarray(self.final_velocities.numpy(), dtype=np.float32).astype(np.float64)
        starts = tuple(np.asarray(value.numpy(), dtype=np.float64) for value in self.outer_start_positions)
        candidates = tuple(np.asarray(value.numpy(), dtype=np.float64) for value in self.outer_candidate_positions)
        rhs_values = tuple(np.asarray(workspace.rhs.numpy(), dtype=np.float64) for workspace in self.workspaces)
        accepted = tuple(bool(value) for value in self.accepted.numpy())
        reason_codes = tuple(int(value) for value in self.reasons.numpy())
        if any(code < 0 or code >= len(REASON_NAMES) for code in reason_codes):
            raise RuntimeError("device returned an invalid direct graph VBD gate reason")
        reasons = tuple(REASON_NAMES[code] for code in reason_codes)
        active = True
        for outer_index, (was_accepted, reason) in enumerate(zip(accepted, reasons, strict=True)):
            if active:
                if reason in ("pending", "masked-after-rejection"):
                    raise RuntimeError(f"device returned an invalid active gate status at outer {outer_index}")
                active = was_accepted
            elif was_accepted or reason != "masked-after-rejection":
                raise RuntimeError(f"device returned a non-sticky masked status at outer {outer_index}")
        canonical_problem = build_common_problem(self.scene)
        canonical_operators = tuple(
            MatrixFreeStableNHOperator.from_problem(canonical_problem, start) for start in starts
        )
        outer_slots = tuple(
            _OuterSlotBinding(
                token=_VALIDATION_TOKEN,
                outer_index=outer_index,
                start_position_sha256=_array_digest(start),
                current_operator_sha256=_operator_sha256(operator),
                rhs_sha256=_array_digest(rhs),
                accepted=accepted[outer_index],
                reason=reasons[outer_index],
            )
            for outer_index, (start, operator, rhs) in enumerate(
                zip(starts, canonical_operators, rhs_values, strict=True)
            )
        )
        context = _EndpointValidationContext(
            token=_VALIDATION_TOKEN,
            issuer=self,
            execution_serial=receipt.serial,
            graph_replay=graph_replay,
            scene=self.scene,
            config=self.config,
            construction_k1=self._construction_k1,
            execution_k1=execution_k1,
            hierarchy=self.hierarchy,
            persistent_device_sha256=persistent_device_sha256,
            warp_snapshot_sha256=warp_snapshot_sha256,
            graph_identity_sha256=graph_identity_sha256,
            v_cycle_kernel_launches=v_cycle_kernel_launches,
            device=str(self.device),
            outer_slots=outer_slots,
        )
        outer_slot_records = tuple(
            (
                slot.outer_index,
                slot.start_position_sha256,
                slot.current_operator_sha256,
                slot.rhs_sha256,
                slot.accepted,
                slot.reason,
            )
            for slot in outer_slots
        )
        self._issued_validation_contexts[id(context)] = (
            context,
            receipt.serial,
            graph_replay,
            execution_k1,
            persistent_device_sha256,
            graph_identity_sha256,
            receipt.graph_object_identity,
            outer_slots,
            outer_slot_records,
        )
        outer_work = []
        for outer_index, (start, workspace, canonical_operator, rhs, slot) in enumerate(
            zip(starts, self.workspaces, canonical_operators, rhs_values, outer_slots, strict=True)
        ):
            operator_sha256 = _operator_sha256(canonical_operator)
            outer_work.append(
                CapturedGraphVBDOuterWork(
                    outer_index=outer_index,
                    start_position_sha256=_array_digest(start),
                    current_operator_sha256=operator_sha256,
                    static_hierarchy_sha256=self.device_hierarchy.hierarchy_sha256,
                    rhs=rhs,
                    first_correction=np.asarray(workspace.first_correction.numpy(), dtype=np.float64),
                    operator_product_after_first=np.asarray(
                        workspace.operator_product_after_first.numpy(), dtype=np.float64
                    ),
                    residual_after_first=np.asarray(workspace.residual_after_first.numpy(), dtype=np.float64),
                    second_correction=np.asarray(workspace.second_correction.numpy(), dtype=np.float64),
                    direction=np.asarray(workspace.direction.numpy(), dtype=np.float64),
                    v_cycles=(
                        workspace.first_cycle.record_internal_application(capture_replay=graph_replay),
                        workspace.second_cycle.record_internal_application(capture_replay=graph_replay),
                    ),
                    capture_replay=graph_replay,
                    linear_kernel_launches=self.linear_kernel_launches_per_outer,
                    persistent_device_sha256=persistent_device_sha256,
                    accepted=accepted[outer_index],
                    reason=reasons[outer_index],
                    _validation_context=context,
                    _validation_slot=slot,
                    _validation_operator=canonical_operator,
                )
            )
        return CapturedGraphVBDEndpoint(
            scene_sha256=self.scene_sha256,
            objective_instance_sha256=self.objective_instance_sha256,
            static_hierarchy_sha256=self.device_hierarchy.hierarchy_sha256,
            config_sha256=self.config_sha256,
            k1_endpoint_sha256=execution_k1.endpoint_sha256,
            k1_position_sha256=execution_k1.position_sha256,
            k1_velocity_sha256=execution_k1.velocity_sha256,
            k1_pristine_state_sha256=execution_k1.pristine_state_sha256,
            persistent_device_sha256=persistent_device_sha256,
            graph_identity_sha256=graph_identity_sha256,
            armijo=float(self.config.armijo),
            minimum_determinant=float(self.config.minimum_determinant),
            free_vertices=np.asarray(self.operator.free_host, dtype=np.int64),
            positions=positions,
            velocities=velocities,
            accepted=accepted,
            reasons=reasons,
            initial_objectives=tuple(float(value) for value in self.initial_objectives.numpy()),
            candidate_objectives=tuple(float(value) for value in self.candidate_objectives.numpy()),
            directional_derivatives=tuple(float(value) for value in self.directional_derivatives.numpy()),
            segment_minimum_determinants=tuple(float(value) for value in self.minimum_segment_determinants.numpy()),
            outer_start_positions=starts,
            outer_candidate_positions=candidates,
            outer_work=tuple(outer_work),
            graph_replay=graph_replay,
            _validation_context=context,
        )

    @staticmethod
    def _poison_array(array: wp.array[Any], rng: np.random.Generator) -> None:
        """Assign finite random data with the exact Warp array shape/dtype."""
        shape = tuple(array.shape)
        if array.dtype in (wp.int32, wp.int64):
            array.assign(rng.integers(-17, 18, size=shape, dtype=np.int32))
        elif array.dtype in (wp.mat33, wp.mat33d):
            array.assign(rng.normal(size=(*shape, 3, 3)))
        elif array.dtype in (wp.vec3, wp.vec3d):
            array.assign(rng.normal(size=(*shape, 3)))
        else:
            array.assign(rng.normal(size=shape))

    def poison(self, *, seed: int) -> None:
        """Poison every mutable visible correction buffer before replay."""
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError("seed must be an integer")
        rng = np.random.default_rng(int(seed))
        self.baseline.poison_lane(1, seed=int(seed) + 1)
        for array in (
            self.operator.positions,
            self.operator.deformation_gradients,
            self.operator.cofactors,
            self.operator.determinants,
            self.operator.first_piola,
            self.candidate,
            self.proposal_finite,
            self.final_positions,
            self.final_velocities,
            self.active,
            self.accepted,
            self.reasons,
            self.current_inertia,
            self.candidate_inertia,
            self.vertex_finite,
            self.current_elastic,
            self.candidate_elastic,
            self.candidate_determinants,
            self.segment_minima,
            self.tet_finite,
            self.directional_terms,
            self.initial_objectives,
            self.candidate_objectives,
            self.directional_derivatives,
            self.minimum_segment_determinants,
            *self.outer_start_positions,
            *self.outer_candidate_positions,
        ):
            self._poison_array(array, rng)
        for workspace in self.workspaces:
            for array in (
                workspace.rhs,
                workspace.first_correction,
                workspace.operator_product_after_first,
                workspace.residual_after_first,
                workspace.second_correction,
                workspace.direction,
                workspace.operator_apply.delta_piola,
                *workspace.first_cycle.level_rhs,
                *workspace.first_cycle.level_correction,
                *workspace.first_cycle.level_product,
                *workspace.first_cycle.level_residual,
                workspace.first_cycle.coarse_intermediate,
                *workspace.second_cycle.level_rhs,
                *workspace.second_cycle.level_correction,
                *workspace.second_cycle.level_product,
                *workspace.second_cycle.level_residual,
                workspace.second_cycle.coarse_intermediate,
            ):
                self._poison_array(array, rng)

    def deterministic_record(self) -> dict[str, object]:
        """Return the fixed graph schedule and scope without timing data."""
        persistent_device_sha256 = self._validate_persistent_sources()
        graph_identity_sha256 = self._graph_identity(captured=True, comparator=False)
        k4_graph_identity_sha256 = self._graph_identity(captured=True, comparator=True)
        return {
            "contract_id": CONTRACT_ID,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "config": self.config.deterministic_record(),
            "config_sha256": self.config_sha256,
            "construction_k1_endpoint_sha256": self._construction_k1.endpoint_sha256,
            "construction_k1_position_sha256": self._construction_k1.position_sha256,
            "construction_k1_velocity_sha256": self._construction_k1.velocity_sha256,
            "construction_k1_pristine_state_sha256": self._construction_k1.pristine_state_sha256,
            "construction_k4_endpoint_sha256": self._construction_k4.endpoint_sha256,
            "construction_k4_position_sha256": self._construction_k4.position_sha256,
            "construction_k4_velocity_sha256": self._construction_k4.velocity_sha256,
            "solver_lane_contract_sha256": self._solver_lane_contract_sha256_bound,
            "solver_scalar_sha256": self._solver_scalar_sha256_bound,
            "solver_static_array_sha256": self._solver_static_array_sha256_bound,
            "persistent_device_sha256": persistent_device_sha256,
            "graph_identity_sha256": graph_identity_sha256,
            "k4_graph_identity_sha256": k4_graph_identity_sha256,
            "device": str(self.device),
            "vbd_iterations": 1,
            "outer_corrections": OUTER_CORRECTIONS,
            "stationary_v_cycles_per_outer": V_CYCLES_PER_OUTER,
            "v_cycles": OUTER_CORRECTIONS * V_CYCLES_PER_OUTER,
            "krylov_iterations": 0,
            "linear_formula": "d=B*b+B*(b-A_current*B*b)",
            "current_operator_refreshes": OUTER_CORRECTIONS,
            "current_operator_applications": OUTER_CORRECTIONS,
            "static_hierarchy_sha256": self.device_hierarchy.hierarchy_sha256,
            "hierarchy_level_count": len(self.device_hierarchy.levels),
            "hierarchy_level_scalar_sizes": [level.scalar_size for level in self.device_hierarchy.levels],
            "v_cycle_kernel_launches": self.device_hierarchy.scheduled_kernel_launches,
            "linear_kernel_launches_per_outer": self.linear_kernel_launches_per_outer,
            "correction_kernel_launches_excluding_public_k1": self.correction_kernel_launches,
            "alpha": self.config.alpha,
            "armijo": self.config.armijo,
            "minimum_determinant": self.config.minimum_determinant,
            "gate_execution": "device-side-published-fp32-strict-armijo-exact-cubic-segment-fail-closed",
            "rejection_mask": "sticky-after-first-rejection-with-fixed-work",
            "final_velocity": "BDF1-from-physical-x-current-exact-zero-pins",
            "separate_k4_graph": True,
            "performance_evidence": False,
        }
