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

import ctypes
import dataclasses
import hashlib
import json
import math
import numbers
import statistics
import weakref
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np
import warp as wp

from newton.solvers import SolverVBD

from .captured_mg_vbd import (
    _commit_candidate,
    _tet_gate_terms,
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
from .correction_gpu_warp import (
    FUSED_GATHER_KERNEL_VERSION,
    SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
    WarpMatrixFreeStableNHOperator,
    WarpMatrixFreeWorkspace,
)
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
    KERNEL_VERSION as SOURCE_V_CYCLE_KERNEL_VERSION,
)
from .correction_multigrid_warp import (
    MAX_COARSE_SCALAR_SIZE,
    WarpStaticMultigridHierarchy,
)
from .correction_multigrid_warp_scalar_fused import (
    _CORE_RECORD_TOKEN,
    WarpScalarFusedStaticMultigridHierarchy,
    WarpScalarFusedVCyclePhysicalWork,
    WarpScalarFusedVCycleRecord,
    WarpScalarFusedVCycleWorkspace,
)
from .correction_multigrid_warp_scalar_fused import (
    CONTRACT_ID as V_CYCLE_CONTRACT_ID,
)
from .correction_multigrid_warp_scalar_fused import (
    EXTERNAL_SHARED_PUBLICATION_ROUTE as V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
)
from .correction_multigrid_warp_scalar_fused import (
    KERNEL_VERSION as V_CYCLE_KERNEL_VERSION,
)
from .correction_multigrid_warp_scalar_fused import (
    PUBLICATION_VERSION as V_CYCLE_PUBLICATION_VERSION,
)
from .correction_multigrid_warp_scalar_fused import (
    SCHEDULE_VERSION as V_CYCLE_SCHEDULE_VERSION,
)
from .correction_multigrid_warp_scalar_fused import (
    STANDALONE_PUBLICATION_ROUTE as V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
)
from .solver_benchmark import TetBenchmarkScene, build_common_problem, common_objective_manifest

CONTRACT_ID = "captured-direct-multiplicative-graph-vbd-v1"
OUTER_CORRECTIONS = 4
V_CYCLES_PER_OUTER = 2
OUTER_KERNEL_VERSION = "captured-direct-graph-vbd-four-warp-exact-finalize-outer-v5"
OUTER_SCHEDULE_VERSION = "captured-direct-graph-vbd-outer-schedule-v6"
FIRST_CYCLE_PUBLICATION_ROLE = "current-a-apply-free-row-owner-scalar-to-vec3"
SECOND_CYCLE_PUBLICATION_ROLE = "vertex-owner-scalar-to-vec3"
FINALIZE_GATE_ROUTE = "cuda-one-block-four-warp-ordered-fp64-v1"
FINALIZE_GATE_BLOCK_DIM = 128
FINALIZE_GATE_OWNER_THREADS = (0, 32, 64, 96)
FINALIZE_GATE_OWNER_ROLES = (
    "ordered-objective-pair",
    "ordered-directional-derivative",
    "ordered-determinant-minima-pair",
    "ordered-finite-flags",
)
FINALIZE_GATE_COLLECTIVE_VERSION = "shared-tile-vec2d-float64-vec2d-int32-broadcasts-v1"

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


def _derive_outer_schedule_sha256(
    kernel_version: str,
    fused_gather_kernel_version: str,
    scalar_direction_apply_kernel_version: str,
    v_cycle_publication_version: str,
    v_cycle_standalone_publication_route: str,
    v_cycle_external_shared_publication_route: str,
    first_cycle_publication_role: str,
    second_cycle_publication_role: str,
    finalize_gate_route: str,
    finalize_gate_block_dim: int,
    finalize_gate_owner_threads: tuple[int, int, int, int],
    finalize_gate_owner_roles: tuple[str, str, str, str],
    finalize_gate_collective_version: str,
    schedule_version: str,
) -> str:
    """Bind fused formulas and the exact ordered four-warp gate."""
    return _canonical_digest(
        {
            "contract": "captured-direct-graph-vbd-outer-schedule-v6",
            "kernel_version": kernel_version,
            "fused_gather_kernel_version": fused_gather_kernel_version,
            "scalar_direction_apply_kernel_version": scalar_direction_apply_kernel_version,
            "v_cycle_publication_version": v_cycle_publication_version,
            "v_cycle_standalone_publication_route": v_cycle_standalone_publication_route,
            "v_cycle_external_shared_publication_route": v_cycle_external_shared_publication_route,
            "first_cycle_publication_role": first_cycle_publication_role,
            "second_cycle_publication_role": second_cycle_publication_role,
            "finalize_gate_route": finalize_gate_route,
            "finalize_gate_block_dim": finalize_gate_block_dim,
            "finalize_gate_owner_threads": list(finalize_gate_owner_threads),
            "finalize_gate_owner_roles": list(finalize_gate_owner_roles),
            "finalize_gate_collective_version": finalize_gate_collective_version,
            "finalize_gate_recurrences": [
                "owner-0-interleaved-start-end-current-inertia-then-current-elastic",
                "owner-32-directional-terms",
                "owner-64-interleaved-segment-and-candidate-minima",
                "owner-96-proposal-and-vertex-finite-then-tet-finite",
            ],
            "finalize_gate_collective_layout": [
                "vec2d-objectives",
                "float64-derivative",
                "vec2d-minima",
                "int32-finite",
            ],
            "finalize_gate_synchronization": "four-unconditional-shared-tile-from-thread-block-broadcasts",
            "finalize_gate_reduction_policy": "no-domain-partials-no-tree-no-atomic",
            "finalize_gate_owner_executable_binding": "literal-thread-ids-0-32-64-96-no-mutable-duplicate-globals",
            "schedule_version": schedule_version,
            "fused_gather_operations": [
                "gradient-and-final-store-active-mask",
                "matrix-free-product-and-rhs-minus-product",
            ],
            "ownership": "one-thread-per-scene-vertex-with-unique-free-index-owner",
            "first_cycle_publication_operations": [
                "tet-domain-reads-immutable-first-cycle-final-scalar",
                "free-row-owner-publishes-first-correction-and-retains-current-a-product-residual",
            ],
            "fused_vertex_operations": [
                "outer_start=current",
                "second_correction[free]=second_cycle_final_scalar[3*free:3*free+3]",
                "direction[free]=first_correction[free]+second_correction[free]",
                "candidate=current-or-fp32(current+direction)-published-as-fp64",
                "outer_candidate=candidate",
                "current_candidate_inertia-and-vertex-finite",
                "directional[free]=-dot(rhs[free],candidate-current)",
            ],
            "candidate_policy": "pinned-or-inactive-or-nonfinite-proposal-keeps-current",
            "post_fusion_order": ["tet-gate-terms", "four-warp-exact-finalize-gate", "commit-candidate"],
            "linear_prefix_kernel_launches_per_outer": "4+2*core_v_cycle_launches",
            "fused_gather_kernel_launches_per_outer": 2,
            "fused_vertex_kernel_launches_per_outer": 1,
            "finalize_gate_kernel_launches_per_outer": 1,
            "retained_linear_work_launches_per_outer": "5+2*core_v_cycle_launches",
            "remaining_gate_commit_launches_per_outer": 3,
        }
    )


def _require_finalize_gate_evidence(
    route: object,
    block_dim: object,
    owner_threads: object,
    owner_roles: object,
    collective_version: object,
) -> None:
    """Require the exact built-in four-warp gate schedule values."""
    if type(route) is not str or route != FINALIZE_GATE_ROUTE:
        raise ValueError("finalize gate route is not canonical")
    if type(block_dim) is not int or block_dim != FINALIZE_GATE_BLOCK_DIM:
        raise ValueError("finalize gate block dimension is not canonical")
    if (
        type(owner_threads) is not tuple
        or owner_threads != FINALIZE_GATE_OWNER_THREADS
        or any(type(value) is not int for value in owner_threads)
    ):
        raise ValueError("finalize gate owner threads are not canonical")
    if (
        type(owner_roles) is not tuple
        or owner_roles != FINALIZE_GATE_OWNER_ROLES
        or any(type(value) is not str for value in owner_roles)
    ):
        raise ValueError("finalize gate owner roles are not canonical")
    if type(collective_version) is not str or collective_version != FINALIZE_GATE_COLLECTIVE_VERSION:
        raise ValueError("finalize gate collective version is not canonical")


OUTER_SCHEDULE_SHA256 = _derive_outer_schedule_sha256(
    OUTER_KERNEL_VERSION,
    FUSED_GATHER_KERNEL_VERSION,
    SCALAR_DIRECTION_APPLY_KERNEL_VERSION,
    V_CYCLE_PUBLICATION_VERSION,
    V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
    V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
    FIRST_CYCLE_PUBLICATION_ROLE,
    SECOND_CYCLE_PUBLICATION_ROLE,
    FINALIZE_GATE_ROUTE,
    FINALIZE_GATE_BLOCK_DIM,
    FINALIZE_GATE_OWNER_THREADS,
    FINALIZE_GATE_OWNER_ROLES,
    FINALIZE_GATE_COLLECTIVE_VERSION,
    OUTER_SCHEDULE_VERSION,
)
print(f"[kernels] captured graph VBD outer version: {OUTER_KERNEL_VERSION}")


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
    fields = tuple(getattr(type(descriptor), "_fields_", ()))
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
    return (id(descriptor), type(descriptor), tuple(type(descriptor)._fields_), values)


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
    record: WarpScalarFusedVCycleRecord,
    *,
    rhs: np.ndarray,
    output: np.ndarray,
    canonical_output: np.ndarray,
    canonical_work: VCycleWorkRecord,
    hierarchy_sha256: str,
    schedule_sha256: str,
    static_device_content_sha256: str,
    device_snapshot_sha256: str,
    scheduled_kernel_launches: int,
    core_kernel_launches: int,
    publication_kernel_launches: int,
    publication_version: str,
    publication_route: str,
    root_ingress_zero_start_fusions: int,
    hierarchy: StaticMultigridHierarchy,
    capture_replay: bool,
    name: str,
) -> None:
    """Recompute one scalar-fused V-cycle's algebraic and physical evidence."""
    if type(record) is not WarpScalarFusedVCycleRecord:
        raise TypeError(f"{name} must be an exact WarpScalarFusedVCycleRecord")
    if type(record.work) is not VCycleWorkRecord:
        raise TypeError(f"{name} work must be an exact VCycleWorkRecord")
    if type(record.physical_work) is not WarpScalarFusedVCyclePhysicalWork:
        raise TypeError(f"{name} physical work must be exact scalar-fused evidence")
    if (
        type(record.contract_id) is not str
        or record.contract_id != V_CYCLE_CONTRACT_ID
        or type(record.kernel_version) is not str
        or record.kernel_version != V_CYCLE_KERNEL_VERSION
        or type(record.schedule_version) is not str
        or record.schedule_version != V_CYCLE_SCHEDULE_VERSION
        or record.research_only is not True
        or record.performance_evidence is not False
    ):
        raise ValueError(f"{name} has invalid V-cycle policy provenance")
    if type(record.capture_replay) is not bool:
        raise TypeError(f"{name} capture provenance must be a built-in bool")
    if record.capture_replay != capture_replay:
        raise ValueError(f"{name} capture provenance disagrees")
    if type(record.scheduled_kernel_launches) is not int:
        raise TypeError(f"{name} scheduled launch count must be a built-in int")
    if record.scheduled_kernel_launches != scheduled_kernel_launches:
        raise ValueError(f"{name} scheduled launch count is stale")
    if (
        record.schedule_sha256 != schedule_sha256
        or record.static_device_content_sha256 != static_device_content_sha256
        or record.device_snapshot_sha256 != device_snapshot_sha256
    ):
        raise ValueError(f"{name} scalar-fused schedule or static snapshot identity is stale")
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
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} {field_name} must be a non-negative built-in integer")
    if type(work.level_visits) is not tuple or any(type(value) is not int or value < 0 for value in work.level_visits):
        raise ValueError(f"{name} level_visits must contain non-negative built-in integers")
    expected_rhs_sha256 = _v_cycle_rhs_sha256(rhs)
    expected_result_sha256 = _v_cycle_result_sha256(output)
    if _v_cycle_result_sha256(record.correction) != expected_result_sha256:
        raise ValueError(f"{name} retained correction bytes do not exactly bind its output vector")
    for field_name in ("hierarchy_sha256", "rhs_sha256", "result_sha256", "content_sha256"):
        _require_sha256(getattr(work, field_name), name=f"{name}.{field_name}")
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

    noncoarse_levels = hierarchy.levels[:-1]
    expected_elided_products = sum(level.matrix.stored_block_count for level in noncoarse_levels)
    expected_zero_start_solves = sum(level.matrix.block_row_count for level in noncoarse_levels)
    expected_matrix_launches = len(noncoarse_levels) * (hierarchy.pre_smooth_steps + hierarchy.post_smooth_steps)
    expected_root_fusions = int(bool(noncoarse_levels))
    if type(root_ingress_zero_start_fusions) is not int or root_ingress_zero_start_fusions != expected_root_fusions:
        raise ValueError(f"{name} root ingress fusion count is stale")
    physical = record.physical_work
    expected_physical = {
        "hierarchy_sha256": hierarchy_sha256,
        "schedule_sha256": schedule_sha256,
        "rhs_sha256": expected_rhs_sha256,
        "result_sha256": expected_result_sha256,
        "matrix_block_products_executed": canonical_work.matrix_block_products - expected_elided_products,
        "matrix_block_products_elided_zero_start": expected_elided_products,
        "zero_start_block_solves": expected_zero_start_solves,
        "root_ingress_zero_start_fusions": root_ingress_zero_start_fusions,
        "out_of_place_jacobi_block_solves": canonical_work.smoother_block_solves - expected_zero_start_solves,
        "matrix_kernel_launches": expected_matrix_launches,
        "jacobi_kernel_launches": expected_matrix_launches,
        "core_kernel_launches": core_kernel_launches,
        "publication_kernel_launches": publication_kernel_launches,
        "publication_version": publication_version,
        "publication_route": publication_route,
        "scheduled_kernel_launches": scheduled_kernel_launches,
    }
    for field_name, expected in expected_physical.items():
        actual = getattr(physical, field_name)
        if actual != expected:
            raise ValueError(f"{name} physical {field_name} does not match the canonical scalar schedule")
    for field_name in (
        "matrix_block_products_executed",
        "matrix_block_products_elided_zero_start",
        "zero_start_block_solves",
        "root_ingress_zero_start_fusions",
        "out_of_place_jacobi_block_solves",
        "matrix_kernel_launches",
        "jacobi_kernel_launches",
        "core_kernel_launches",
        "publication_kernel_launches",
        "scheduled_kernel_launches",
    ):
        value = getattr(physical, field_name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} physical {field_name} must be a non-negative built-in integer")
    if physical.matrix_block_products_executed + physical.matrix_block_products_elided_zero_start != (
        canonical_work.matrix_block_products
    ):
        raise ValueError(f"{name} physical executed and elided work does not recover canonical matrix work")
    if (
        type(physical.publication_version) is not str
        or type(physical.publication_route) is not str
        or physical.scheduled_kernel_launches != physical.core_kernel_launches + physical.publication_kernel_launches
    ):
        raise ValueError(f"{name} physical publication route or launch accounting is invalid")
    physical_sha256 = _hash_parts(
        "warp-scalar-fused-v-cycle-physical-work-v4",
        tuple(expected_physical.items()),
    )
    if physical.content_sha256 != physical_sha256:
        raise ValueError(f"{name} physical work hash is stale")
    record_sha256 = _hash_parts(
        "warp-scalar-fused-v-cycle-result-v4",
        (
            ("device_snapshot_sha256", device_snapshot_sha256),
            ("static_device_content_sha256", static_device_content_sha256),
            ("schedule_sha256", schedule_sha256),
            ("work_sha256", work_sha256),
            ("physical_work_sha256", physical_sha256),
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
    capture_binding: _CaptureGraphOwnerBinding | None = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.serial) is not int or self.serial < 1:
            raise ValueError("execution receipt serial must be a positive built-in int")
        if type(self.graph_replay) is not bool:
            raise ValueError("execution receipt graph_replay must be a bool")
        if self.graph_replay:
            if type(self.capture_binding) is not _CaptureGraphOwnerBinding:
                raise ValueError("captured execution receipt requires one exact capture binding")
        elif self.capture_binding is not None:
            raise ValueError("uncaptured execution receipt cannot claim a capture binding")


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
    v_cycle_schedule_sha256: str
    v_cycle_core_schedule_sha256: str
    v_cycle_static_device_content_sha256: str
    v_cycle_device_snapshot_sha256: str
    v_cycle_core_device_snapshot_sha256: str
    graph_identity_sha256: str
    fused_gather_kernel_version: str
    scalar_direction_apply_kernel_version: str
    v_cycle_publication_version: str
    v_cycle_standalone_publication_route: str
    v_cycle_external_shared_publication_route: str
    first_cycle_publication_role: str
    second_cycle_publication_role: str
    outer_kernel_version: str
    outer_schedule_version: str
    outer_schedule_sha256: str
    finalize_gate_route: str
    finalize_gate_block_dim: int
    finalize_gate_owner_threads: tuple[int, int, int, int]
    finalize_gate_owner_roles: tuple[str, str, str, str]
    finalize_gate_collective_version: str
    capture_binding: _CaptureGraphOwnerBinding | None = dataclasses.field(repr=False, compare=False)
    v_cycle_kernel_launches: int
    v_cycle_core_kernel_launches: int
    v_cycle_root_ingress_zero_start_fusions: int
    device: str
    outer_slots: tuple[_OuterSlotBinding, ...]

    def __post_init__(self) -> None:
        if self.token is not _VALIDATION_TOKEN:
            raise ValueError("endpoint validation context is solver-private")
        if type(self.execution_serial) is not int or self.execution_serial < 1:
            raise ValueError("endpoint validation execution_serial must be a positive built-in int")
        if type(self.graph_replay) is not bool:
            raise ValueError("endpoint validation graph_replay must be a bool")
        if self.graph_replay:
            if type(self.capture_binding) is not _CaptureGraphOwnerBinding:
                raise ValueError("captured endpoint validation requires one exact capture binding")
        elif self.capture_binding is not None:
            raise ValueError("uncaptured endpoint validation cannot claim a capture binding")
        if type(self.scene) is not TetBenchmarkScene or type(self.config) is not DirectGraphVBDConfig:
            raise TypeError("endpoint validation context has invalid canonical inputs")
        if type(self.hierarchy) is not StaticMultigridHierarchy:
            raise TypeError("endpoint validation context has an invalid hierarchy")
        for name in (
            "persistent_device_sha256",
            "v_cycle_schedule_sha256",
            "v_cycle_core_schedule_sha256",
            "v_cycle_static_device_content_sha256",
            "v_cycle_device_snapshot_sha256",
            "v_cycle_core_device_snapshot_sha256",
            "graph_identity_sha256",
            "outer_schedule_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_finalize_gate_evidence(
            self.finalize_gate_route,
            self.finalize_gate_block_dim,
            self.finalize_gate_owner_threads,
            self.finalize_gate_owner_roles,
            self.finalize_gate_collective_version,
        )
        if (
            type(self.fused_gather_kernel_version) is not str
            or self.fused_gather_kernel_version != FUSED_GATHER_KERNEL_VERSION
            or type(self.scalar_direction_apply_kernel_version) is not str
            or self.scalar_direction_apply_kernel_version != SCALAR_DIRECTION_APPLY_KERNEL_VERSION
            or type(self.v_cycle_publication_version) is not str
            or self.v_cycle_publication_version != V_CYCLE_PUBLICATION_VERSION
            or type(self.v_cycle_standalone_publication_route) is not str
            or self.v_cycle_standalone_publication_route != V_CYCLE_STANDALONE_PUBLICATION_ROUTE
            or type(self.v_cycle_external_shared_publication_route) is not str
            or self.v_cycle_external_shared_publication_route != V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE
            or type(self.first_cycle_publication_role) is not str
            or self.first_cycle_publication_role != FIRST_CYCLE_PUBLICATION_ROLE
            or type(self.second_cycle_publication_role) is not str
            or self.second_cycle_publication_role != SECOND_CYCLE_PUBLICATION_ROLE
            or type(self.outer_kernel_version) is not str
            or type(self.outer_schedule_version) is not str
            or self.finalize_gate_route != FINALIZE_GATE_ROUTE
            or self.finalize_gate_block_dim != FINALIZE_GATE_BLOCK_DIM
            or self.finalize_gate_owner_threads != FINALIZE_GATE_OWNER_THREADS
            or self.finalize_gate_owner_roles != FINALIZE_GATE_OWNER_ROLES
            or self.finalize_gate_collective_version != FINALIZE_GATE_COLLECTIVE_VERSION
            or self.outer_schedule_sha256
            != _derive_outer_schedule_sha256(
                self.outer_kernel_version,
                self.fused_gather_kernel_version,
                self.scalar_direction_apply_kernel_version,
                self.v_cycle_publication_version,
                self.v_cycle_standalone_publication_route,
                self.v_cycle_external_shared_publication_route,
                self.first_cycle_publication_role,
                self.second_cycle_publication_role,
                self.finalize_gate_route,
                self.finalize_gate_block_dim,
                self.finalize_gate_owner_threads,
                self.finalize_gate_owner_roles,
                self.finalize_gate_collective_version,
                self.outer_schedule_version,
            )
        ):
            raise ValueError("endpoint validation outer kernel schedule is not canonical")
        if type(self.v_cycle_kernel_launches) is not int or self.v_cycle_kernel_launches < 1:
            raise ValueError("v_cycle_kernel_launches must be a positive built-in int")
        if (
            type(self.v_cycle_core_kernel_launches) is not int
            or self.v_cycle_core_kernel_launches < 1
            or self.v_cycle_kernel_launches != self.v_cycle_core_kernel_launches + 1
        ):
            raise ValueError("V-cycle full/core launch counts must differ by one publication kernel")
        if type(
            self.v_cycle_root_ingress_zero_start_fusions
        ) is not int or self.v_cycle_root_ingress_zero_start_fusions not in (0, 1):
            raise ValueError("v_cycle_root_ingress_zero_start_fusions must be zero or one")
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
def _fused_vertex_outer_terms(
    current: wp.array[wp.vec3d],
    vertex_to_free: wp.array[int],
    first_correction: wp.array[wp.vec3d],
    second_cycle_final_scalar: wp.array[wp.float64],
    second_correction: wp.array[wp.vec3d],
    rhs: wp.array[wp.vec3d],
    direction: wp.array[wp.vec3d],
    active: wp.array[int],
    inertial_target: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    inverse_dt_squared: wp.float64,
    outer_start: wp.array[wp.vec3d],
    candidate: wp.array[wp.vec3d],
    outer_candidate: wp.array[wp.vec3d],
    proposal_finite: wp.array[int],
    current_inertia: wp.array[wp.float64],
    candidate_inertia: wp.array[wp.float64],
    vertex_finite: wp.array[int],
    directional_terms: wp.array[wp.float64],
):
    vertex = wp.tid()
    value = current[vertex]
    outer_start[vertex] = value
    valid = bool(True)
    free_index = vertex_to_free[vertex]
    if free_index >= 0:
        scalar_offset = 3 * free_index
        second_value = wp.vec3d(
            second_cycle_final_scalar[scalar_offset],
            second_cycle_final_scalar[scalar_offset + 1],
            second_cycle_final_scalar[scalar_offset + 2],
        )
        second_correction[free_index] = second_value
        direction_value = first_correction[free_index] + second_value
        direction[free_index] = direction_value
        if active[0] != 0:
            proposed = value + direction_value
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
        directional_terms[free_index] = -wp.dot(rhs[free_index], value - current[vertex])
    candidate[vertex] = value
    outer_candidate[vertex] = value
    proposal_finite[vertex] = int(valid)
    start_delta = current[vertex] - inertial_target[vertex]
    end_delta = value - inertial_target[vertex]
    current_inertia[vertex] = (
        wp.float64(0.5)
        * inverse_dt_squared
        * mass[vertex]
        * wp.dot(
            start_delta,
            start_delta,
        )
    )
    candidate_inertia[vertex] = (
        wp.float64(0.5)
        * inverse_dt_squared
        * mass[vertex]
        * wp.dot(
            end_delta,
            end_delta,
        )
    )
    vertex_finite[vertex] = int(_finite_vec(current[vertex]) and _finite_vec(value))


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
    lane = wp.tid()
    if lane == 0:
        accepted[outer_index] = 0
        initial_objective[outer_index] = wp.float64(0.0)
        candidate_objective[outer_index] = wp.float64(0.0)
        directional_derivative[outer_index] = wp.float64(0.0)
        minimum_segment_determinant[outer_index] = wp.float64(0.0)

    gate_active = active[0]
    start_objective = wp.float64(0.0)
    end_objective = wp.float64(0.0)
    derivative = wp.float64(0.0)
    minimum_segment = wp.float64(1.0e300)
    minimum_candidate = wp.float64(1.0e300)
    finite_value = int(1)

    if gate_active != 0:
        if lane == 0:
            for vertex in range(current_inertia.shape[0]):
                start_objective += current_inertia[vertex]
                end_objective += candidate_inertia[vertex]
            for tet in range(current_elastic.shape[0]):
                start_objective += current_elastic[tet]
                end_objective += candidate_elastic[tet]
        if lane == 32:
            for index in range(directional_terms.shape[0]):
                derivative += directional_terms[index]
        if lane == 64:
            for tet in range(current_elastic.shape[0]):
                minimum_segment = wp.min(minimum_segment, segment_minima[tet])
                minimum_candidate = wp.min(minimum_candidate, candidate_determinants[tet])
        if lane == 96:
            all_finite = bool(True)
            for vertex in range(current_inertia.shape[0]):
                all_finite = all_finite and proposal_finite[vertex] != 0 and vertex_finite[vertex] != 0
            for tet in range(current_elastic.shape[0]):
                all_finite = all_finite and tet_finite[tet] != 0
            finite_value = int(all_finite)

    objective_tile = wp.tile_from_thread(
        shape=(1,),
        value=wp.vec2d(start_objective, end_objective),
        thread_idx=0,
        storage="shared",
    )
    derivative_tile = wp.tile_from_thread(
        shape=(1,),
        value=derivative,
        thread_idx=32,
        storage="shared",
    )
    minima_tile = wp.tile_from_thread(
        shape=(1,),
        value=wp.vec2d(minimum_segment, minimum_candidate),
        thread_idx=64,
        storage="shared",
    )
    finite_tile = wp.tile_from_thread(
        shape=(1,),
        value=finite_value,
        thread_idx=96,
        storage="shared",
    )
    objective_pair = wp.tile_extract(objective_tile, 0)
    derivative = wp.tile_extract(derivative_tile, 0)
    minima = wp.tile_extract(minima_tile, 0)
    finite_value = wp.tile_extract(finite_tile, 0)

    if lane != 0:
        return
    if gate_active == 0:
        reasons[outer_index] = _REASON_MASKED
        return

    start_objective = objective_pair[0]
    end_objective = objective_pair[1]
    minimum_segment = minima[0]
    minimum_candidate = minima[1]
    all_finite = finite_value != 0

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
    v_cycles: tuple[WarpScalarFusedVCycleRecord, WarpScalarFusedVCycleRecord]
    capture_replay: bool
    linear_kernel_launches: int
    persistent_device_sha256: str
    fused_gather_kernel_version: str
    scalar_direction_apply_kernel_version: str
    v_cycle_publication_version: str
    first_cycle_publication_route: str
    second_cycle_publication_route: str
    outer_kernel_version: str
    outer_schedule_version: str
    outer_schedule_sha256: str
    finalize_gate_route: str
    finalize_gate_block_dim: int
    finalize_gate_owner_threads: tuple[int, int, int, int]
    finalize_gate_owner_roles: tuple[str, str, str, str]
    finalize_gate_collective_version: str
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
        _require_finalize_gate_evidence(
            self.finalize_gate_route,
            self.finalize_gate_block_dim,
            self.finalize_gate_owner_threads,
            self.finalize_gate_owner_roles,
            self.finalize_gate_collective_version,
        )
        if (
            type(self.fused_gather_kernel_version) is not str
            or self.fused_gather_kernel_version != self._validation_context.fused_gather_kernel_version
            or type(self.scalar_direction_apply_kernel_version) is not str
            or self.scalar_direction_apply_kernel_version
            != self._validation_context.scalar_direction_apply_kernel_version
            or type(self.v_cycle_publication_version) is not str
            or self.v_cycle_publication_version != self._validation_context.v_cycle_publication_version
            or type(self.first_cycle_publication_route) is not str
            or self.first_cycle_publication_route != self._validation_context.first_cycle_publication_role
            or type(self.second_cycle_publication_route) is not str
            or self.second_cycle_publication_route != self._validation_context.second_cycle_publication_role
            or type(self.outer_kernel_version) is not str
            or self.outer_kernel_version != self._validation_context.outer_kernel_version
            or type(self.outer_schedule_version) is not str
            or self.outer_schedule_version != self._validation_context.outer_schedule_version
            or self.outer_schedule_sha256 != self._validation_context.outer_schedule_sha256
            or self.finalize_gate_route != self._validation_context.finalize_gate_route
            or self.finalize_gate_block_dim != self._validation_context.finalize_gate_block_dim
            or self.finalize_gate_owner_threads != self._validation_context.finalize_gate_owner_threads
            or self.finalize_gate_owner_roles != self._validation_context.finalize_gate_owner_roles
            or self.finalize_gate_collective_version != self._validation_context.finalize_gate_collective_version
            or self.outer_schedule_sha256
            != _derive_outer_schedule_sha256(
                self.outer_kernel_version,
                self.fused_gather_kernel_version,
                self.scalar_direction_apply_kernel_version,
                self.v_cycle_publication_version,
                self._validation_context.v_cycle_standalone_publication_route,
                self._validation_context.v_cycle_external_shared_publication_route,
                self.first_cycle_publication_route,
                self.second_cycle_publication_route,
                self.finalize_gate_route,
                self.finalize_gate_block_dim,
                self.finalize_gate_owner_threads,
                self.finalize_gate_owner_roles,
                self.finalize_gate_collective_version,
                self.outer_schedule_version,
            )
        ):
            raise ValueError("outer work kernel schedule identity is not canonical")
        _require_sha256(self.outer_schedule_sha256, name="outer_schedule_sha256")
        if type(self.v_cycles) is not tuple or len(self.v_cycles) != V_CYCLES_PER_OUTER:
            raise ValueError("outer work must retain exactly two V-cycle records")
        if any(type(record) is not WarpScalarFusedVCycleRecord for record in self.v_cycles):
            raise TypeError("v_cycles must contain exact WarpScalarFusedVCycleRecord values")
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
        for cycle_index, (
            record,
            rhs,
            output,
            canonical,
            schedule_sha256,
            device_snapshot_sha256,
            scheduled_kernel_launches,
            publication_kernel_launches,
            publication_route,
        ) in enumerate(
            zip(
                self.v_cycles,
                (arrays["rhs"], arrays["residual_after_first"]),
                (arrays["first_correction"], arrays["second_correction"]),
                (first, second),
                (self._validation_context.v_cycle_core_schedule_sha256,) * V_CYCLES_PER_OUTER,
                (self._validation_context.v_cycle_core_device_snapshot_sha256,) * V_CYCLES_PER_OUTER,
                (self._validation_context.v_cycle_core_kernel_launches,) * V_CYCLES_PER_OUTER,
                (0,) * V_CYCLES_PER_OUTER,
                (self._validation_context.v_cycle_external_shared_publication_route,) * V_CYCLES_PER_OUTER,
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
                schedule_sha256=schedule_sha256,
                static_device_content_sha256=self._validation_context.v_cycle_static_device_content_sha256,
                device_snapshot_sha256=device_snapshot_sha256,
                scheduled_kernel_launches=scheduled_kernel_launches,
                core_kernel_launches=self._validation_context.v_cycle_core_kernel_launches,
                publication_kernel_launches=publication_kernel_launches,
                publication_version=self.v_cycle_publication_version,
                publication_route=publication_route,
                root_ingress_zero_start_fusions=(self._validation_context.v_cycle_root_ingress_zero_start_fusions),
                hierarchy=self._validation_context.hierarchy,
                capture_replay=self.capture_replay,
                name=f"V-cycle {cycle_index}",
            )

        expected_launches = 5 + sum(record.scheduled_kernel_launches for record in self.v_cycles)
        if type(self.linear_kernel_launches) is not int or self.linear_kernel_launches != expected_launches:
            raise ValueError("linear_kernel_launches does not match the fixed direct schedule")
        payload = {
            "contract": "captured-direct-graph-vbd-outer-work-v4",
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
            "fused_gather_kernel_version": self.fused_gather_kernel_version,
            "scalar_direction_apply_kernel_version": self.scalar_direction_apply_kernel_version,
            "v_cycle_publication_version": self.v_cycle_publication_version,
            "first_cycle_publication_route": self.first_cycle_publication_route,
            "second_cycle_publication_route": self.second_cycle_publication_route,
            "outer_kernel_version": self.outer_kernel_version,
            "outer_schedule_version": self.outer_schedule_version,
            "outer_schedule_sha256": self.outer_schedule_sha256,
            "finalize_gate_route": self.finalize_gate_route,
            "finalize_gate_block_dim": self.finalize_gate_block_dim,
            "finalize_gate_owner_threads": list(self.finalize_gate_owner_threads),
            "finalize_gate_owner_roles": list(self.finalize_gate_owner_roles),
            "finalize_gate_collective_version": self.finalize_gate_collective_version,
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
        _require_issued_validation_context(self._validation_context, validate_raw_sources=False)
        _require_issued_outer_slot(self._validation_context, self._validation_slot)
        validated = dataclasses.replace(self)
        if validated.content_sha256 != self.content_sha256:
            raise ValueError("outer work content hash is not canonical at serialization")
        return {
            "contract": "captured-direct-graph-vbd-outer-work-v4",
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
            "fused_gather_kernel_version": self.fused_gather_kernel_version,
            "scalar_direction_apply_kernel_version": self.scalar_direction_apply_kernel_version,
            "v_cycle_publication_version": self.v_cycle_publication_version,
            "first_cycle_publication_route": self.first_cycle_publication_route,
            "second_cycle_publication_route": self.second_cycle_publication_route,
            "outer_kernel_version": self.outer_kernel_version,
            "outer_schedule_version": self.outer_schedule_version,
            "outer_schedule_sha256": self.outer_schedule_sha256,
            "finalize_gate_route": self.finalize_gate_route,
            "finalize_gate_block_dim": self.finalize_gate_block_dim,
            "finalize_gate_owner_threads": list(self.finalize_gate_owner_threads),
            "finalize_gate_owner_roles": list(self.finalize_gate_owner_roles),
            "finalize_gate_collective_version": self.finalize_gate_collective_version,
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
    fused_gather_kernel_version: str
    scalar_direction_apply_kernel_version: str
    v_cycle_publication_version: str
    v_cycle_standalone_publication_route: str
    v_cycle_external_shared_publication_route: str
    first_cycle_publication_role: str
    second_cycle_publication_role: str
    outer_kernel_version: str
    outer_schedule_version: str
    outer_schedule_sha256: str
    finalize_gate_route: str
    finalize_gate_block_dim: int
    finalize_gate_owner_threads: tuple[int, int, int, int]
    finalize_gate_owner_roles: tuple[str, str, str, str]
    finalize_gate_collective_version: str
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
        _require_finalize_gate_evidence(
            self.finalize_gate_route,
            self.finalize_gate_block_dim,
            self.finalize_gate_owner_threads,
            self.finalize_gate_owner_roles,
            self.finalize_gate_collective_version,
        )
        if (
            type(self.fused_gather_kernel_version) is not str
            or self.fused_gather_kernel_version != context.fused_gather_kernel_version
            or type(self.scalar_direction_apply_kernel_version) is not str
            or self.scalar_direction_apply_kernel_version != context.scalar_direction_apply_kernel_version
            or type(self.v_cycle_publication_version) is not str
            or self.v_cycle_publication_version != context.v_cycle_publication_version
            or type(self.v_cycle_standalone_publication_route) is not str
            or self.v_cycle_standalone_publication_route != context.v_cycle_standalone_publication_route
            or type(self.v_cycle_external_shared_publication_route) is not str
            or self.v_cycle_external_shared_publication_route != context.v_cycle_external_shared_publication_route
            or type(self.first_cycle_publication_role) is not str
            or self.first_cycle_publication_role != context.first_cycle_publication_role
            or type(self.second_cycle_publication_role) is not str
            or self.second_cycle_publication_role != context.second_cycle_publication_role
            or type(self.outer_kernel_version) is not str
            or self.outer_kernel_version != context.outer_kernel_version
            or type(self.outer_schedule_version) is not str
            or self.outer_schedule_version != context.outer_schedule_version
            or self.outer_schedule_sha256 != context.outer_schedule_sha256
            or self.finalize_gate_route != context.finalize_gate_route
            or self.finalize_gate_block_dim != context.finalize_gate_block_dim
            or self.finalize_gate_owner_threads != context.finalize_gate_owner_threads
            or self.finalize_gate_owner_roles != context.finalize_gate_owner_roles
            or self.finalize_gate_collective_version != context.finalize_gate_collective_version
            or self.outer_schedule_sha256
            != _derive_outer_schedule_sha256(
                self.outer_kernel_version,
                self.fused_gather_kernel_version,
                self.scalar_direction_apply_kernel_version,
                self.v_cycle_publication_version,
                self.v_cycle_standalone_publication_route,
                self.v_cycle_external_shared_publication_route,
                self.first_cycle_publication_role,
                self.second_cycle_publication_role,
                self.finalize_gate_route,
                self.finalize_gate_block_dim,
                self.finalize_gate_owner_threads,
                self.finalize_gate_owner_roles,
                self.finalize_gate_collective_version,
                self.outer_schedule_version,
            )
        ):
            raise ValueError("endpoint outer kernel schedule identity is not canonical")
        _require_sha256(self.outer_schedule_sha256, name="outer_schedule_sha256")
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
            if (
                work.scalar_direction_apply_kernel_version != self.scalar_direction_apply_kernel_version
                or work.v_cycle_publication_version != self.v_cycle_publication_version
                or work.first_cycle_publication_route != self.first_cycle_publication_role
                or work.second_cycle_publication_route != self.second_cycle_publication_role
            ):
                raise ValueError("outer work changed the scalar publication routes")
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
        correction_launches = 2 + sum(work.linear_kernel_launches + 3 for work in self.outer_work)
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
                    "fused_gather_kernel_version": self.fused_gather_kernel_version,
                    "v_cycle_publication_version": self.v_cycle_publication_version,
                    "v_cycle_standalone_publication_route": self.v_cycle_standalone_publication_route,
                    "v_cycle_external_shared_publication_route": self.v_cycle_external_shared_publication_route,
                    "outer_kernel_version": self.outer_kernel_version,
                    "outer_schedule_version": self.outer_schedule_version,
                    "outer_schedule_sha256": self.outer_schedule_sha256,
                    "finalize_gate_route": self.finalize_gate_route,
                    "finalize_gate_block_dim": self.finalize_gate_block_dim,
                    "finalize_gate_owner_threads": list(self.finalize_gate_owner_threads),
                    "finalize_gate_owner_roles": list(self.finalize_gate_owner_roles),
                    "finalize_gate_collective_version": self.finalize_gate_collective_version,
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
                    "scalar_direction_apply_kernel_version": self.scalar_direction_apply_kernel_version,
                    "first_cycle_publication_role": self.first_cycle_publication_role,
                    "second_cycle_publication_role": self.second_cycle_publication_role,
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
        _require_issued_validation_context(self._validation_context, validate_raw_sources=True)
        validated = dataclasses.replace(self)
        if validated.endpoint_sha256 != self.endpoint_sha256:
            raise ValueError("endpoint content hash is not canonical at serialization")
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
            "fused_gather_kernel_version": self.fused_gather_kernel_version,
            "scalar_direction_apply_kernel_version": self.scalar_direction_apply_kernel_version,
            "v_cycle_publication_version": self.v_cycle_publication_version,
            "v_cycle_standalone_publication_route": self.v_cycle_standalone_publication_route,
            "v_cycle_external_shared_publication_route": self.v_cycle_external_shared_publication_route,
            "first_cycle_publication_role": self.first_cycle_publication_role,
            "second_cycle_publication_role": self.second_cycle_publication_role,
            "first_cycle_kernel_launches": self._validation_context.v_cycle_core_kernel_launches,
            "second_cycle_kernel_launches": self._validation_context.v_cycle_core_kernel_launches,
            "first_cycle_publication_kernel_launches": 0,
            "second_cycle_publication_kernel_launches": 0,
            "outer_kernel_version": self.outer_kernel_version,
            "outer_schedule_version": self.outer_schedule_version,
            "outer_schedule_sha256": self.outer_schedule_sha256,
            "finalize_gate_route": self.finalize_gate_route,
            "finalize_gate_block_dim": self.finalize_gate_block_dim,
            "finalize_gate_owner_threads": list(self.finalize_gate_owner_threads),
            "finalize_gate_owner_roles": list(self.finalize_gate_owner_roles),
            "finalize_gate_collective_version": self.finalize_gate_collective_version,
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
    """Schema-validated paired CUDA-event diagnostics versus pristine K4.

    Timing samples are intentionally not solver-issued or content-authenticated
    execution evidence. They remain diagnostic-only even after schema validation.
    """

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
    fused_gather_kernel_version: str
    scalar_direction_apply_kernel_version: str
    v_cycle_kernel_version: str
    v_cycle_schedule_version: str
    v_cycle_schedule_sha256: str
    v_cycle_core_schedule_sha256: str
    v_cycle_publication_version: str
    v_cycle_standalone_publication_route: str
    v_cycle_external_shared_publication_route: str
    first_cycle_publication_role: str
    second_cycle_publication_role: str
    v_cycle_kernel_launches: int
    v_cycle_core_kernel_launches: int
    outer_kernel_version: str
    outer_schedule_version: str
    outer_schedule_sha256: str
    finalize_gate_route: str
    finalize_gate_block_dim: int
    finalize_gate_owner_threads: tuple[int, int, int, int]
    finalize_gate_owner_roles: tuple[str, str, str, str]
    finalize_gate_collective_version: str
    correction_kernel_launches: int
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
        _require_finalize_gate_evidence(
            self.finalize_gate_route,
            self.finalize_gate_block_dim,
            self.finalize_gate_owner_threads,
            self.finalize_gate_owner_roles,
            self.finalize_gate_collective_version,
        )
        if (
            type(self.fused_gather_kernel_version) is not str
            or self.fused_gather_kernel_version != FUSED_GATHER_KERNEL_VERSION
            or type(self.scalar_direction_apply_kernel_version) is not str
            or self.scalar_direction_apply_kernel_version != SCALAR_DIRECTION_APPLY_KERNEL_VERSION
            or type(self.v_cycle_kernel_version) is not str
            or self.v_cycle_kernel_version != V_CYCLE_KERNEL_VERSION
            or type(self.v_cycle_schedule_version) is not str
            or self.v_cycle_schedule_version != V_CYCLE_SCHEDULE_VERSION
            or type(self.v_cycle_publication_version) is not str
            or self.v_cycle_publication_version != V_CYCLE_PUBLICATION_VERSION
            or type(self.v_cycle_standalone_publication_route) is not str
            or self.v_cycle_standalone_publication_route != V_CYCLE_STANDALONE_PUBLICATION_ROUTE
            or type(self.v_cycle_external_shared_publication_route) is not str
            or self.v_cycle_external_shared_publication_route != V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE
            or type(self.first_cycle_publication_role) is not str
            or self.first_cycle_publication_role != FIRST_CYCLE_PUBLICATION_ROLE
            or type(self.second_cycle_publication_role) is not str
            or self.second_cycle_publication_role != SECOND_CYCLE_PUBLICATION_ROLE
            or type(self.outer_kernel_version) is not str
            or self.outer_kernel_version != OUTER_KERNEL_VERSION
            or type(self.outer_schedule_version) is not str
            or self.outer_schedule_version != OUTER_SCHEDULE_VERSION
            or self.finalize_gate_route != FINALIZE_GATE_ROUTE
            or self.finalize_gate_block_dim != FINALIZE_GATE_BLOCK_DIM
            or self.finalize_gate_owner_threads != FINALIZE_GATE_OWNER_THREADS
            or self.finalize_gate_owner_roles != FINALIZE_GATE_OWNER_ROLES
            or self.finalize_gate_collective_version != FINALIZE_GATE_COLLECTIVE_VERSION
            or self.outer_schedule_sha256
            != _derive_outer_schedule_sha256(
                self.outer_kernel_version,
                self.fused_gather_kernel_version,
                self.scalar_direction_apply_kernel_version,
                self.v_cycle_publication_version,
                self.v_cycle_standalone_publication_route,
                self.v_cycle_external_shared_publication_route,
                self.first_cycle_publication_role,
                self.second_cycle_publication_role,
                self.finalize_gate_route,
                self.finalize_gate_block_dim,
                self.finalize_gate_owner_threads,
                self.finalize_gate_owner_roles,
                self.finalize_gate_collective_version,
                self.outer_schedule_version,
            )
        ):
            raise ValueError("timing kernel and publication schedule identity is invalid")
        for name in (
            "scene_sha256",
            "objective_instance_sha256",
            "config_sha256",
            "static_hierarchy_sha256",
            "persistent_device_sha256",
            "graph_identity_sha256",
            "k4_graph_identity_sha256",
            "v_cycle_schedule_sha256",
            "v_cycle_core_schedule_sha256",
            "outer_schedule_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if (
            type(self.v_cycle_kernel_launches) is not int
            or type(self.v_cycle_core_kernel_launches) is not int
            or self.v_cycle_kernel_launches != self.v_cycle_core_kernel_launches + 1
            or type(self.correction_kernel_launches) is not int
            or self.correction_kernel_launches != 2 + OUTER_CORRECTIONS * (8 + 2 * self.v_cycle_core_kernel_launches)
        ):
            raise ValueError("timing launch counts do not match the fixed dual-core captured schedule")
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
        validated = dataclasses.replace(self)
        return statistics.median(validated.graph_seconds)

    @property
    def k4_median_seconds(self) -> float:
        """Median captured pristine K4 time [s]."""
        validated = dataclasses.replace(self)
        return statistics.median(validated.k4_seconds)

    def deterministic_record(self) -> dict[str, object]:
        """Serialize schema-validated, unauthenticated diagnostic timing."""
        validated = dataclasses.replace(self)
        return {
            "contract_id": validated.contract_id,
            "scene_sha256": validated.scene_sha256,
            "objective_instance_sha256": validated.objective_instance_sha256,
            "config_sha256": validated.config_sha256,
            "static_hierarchy_sha256": validated.static_hierarchy_sha256,
            "persistent_device_sha256": validated.persistent_device_sha256,
            "graph_identity_sha256": validated.graph_identity_sha256,
            "k4_graph_identity_sha256": validated.k4_graph_identity_sha256,
            "comparator_contract_id": validated.comparator_contract_id,
            "fused_gather_kernel_version": validated.fused_gather_kernel_version,
            "scalar_direction_apply_kernel_version": validated.scalar_direction_apply_kernel_version,
            "v_cycle_kernel_version": validated.v_cycle_kernel_version,
            "v_cycle_schedule_version": validated.v_cycle_schedule_version,
            "v_cycle_schedule_sha256": validated.v_cycle_schedule_sha256,
            "v_cycle_core_schedule_sha256": validated.v_cycle_core_schedule_sha256,
            "v_cycle_publication_version": validated.v_cycle_publication_version,
            "v_cycle_standalone_publication_route": validated.v_cycle_standalone_publication_route,
            "v_cycle_external_shared_publication_route": validated.v_cycle_external_shared_publication_route,
            "first_cycle_publication_role": validated.first_cycle_publication_role,
            "second_cycle_publication_role": validated.second_cycle_publication_role,
            "v_cycle_kernel_launches": validated.v_cycle_kernel_launches,
            "v_cycle_core_kernel_launches": validated.v_cycle_core_kernel_launches,
            "first_cycle_kernel_launches": validated.v_cycle_core_kernel_launches,
            "second_cycle_kernel_launches": validated.v_cycle_core_kernel_launches,
            "first_cycle_publication_kernel_launches": 0,
            "second_cycle_publication_kernel_launches": 0,
            "outer_kernel_version": validated.outer_kernel_version,
            "outer_schedule_version": validated.outer_schedule_version,
            "outer_schedule_sha256": validated.outer_schedule_sha256,
            "finalize_gate_route": validated.finalize_gate_route,
            "finalize_gate_block_dim": validated.finalize_gate_block_dim,
            "finalize_gate_owner_threads": list(validated.finalize_gate_owner_threads),
            "finalize_gate_owner_roles": list(validated.finalize_gate_owner_roles),
            "finalize_gate_collective_version": validated.finalize_gate_collective_version,
            "correction_kernel_launches": validated.correction_kernel_launches,
            "pair_orders": list(validated.pair_orders),
            "graph_seconds": list(validated.graph_seconds),
            "k4_seconds": list(validated.k4_seconds),
            "warmup_replays": validated.warmup_replays,
            "random_seed": validated.random_seed,
            "device": validated.device,
            "setup_included": validated.setup_included,
            "transfers_included": validated.transfers_included,
            "integrated_direct_graph": validated.integrated_direct_graph,
            "performance_evidence": validated.performance_evidence,
            "measurement_authentication": "schema-validated-not-content-authenticated-v1",
            "solver_issued_authentication": False,
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


@dataclasses.dataclass(frozen=True, slots=True)
class _ScalarFusedHierarchyEvidence:
    """Canonical scalar-fused schedule, static-content, and snapshot identities."""

    source_device_snapshot_sha256: str
    schedule_sha256: str
    core_schedule_sha256: str
    static_device_content_sha256: str
    device_snapshot_sha256: str
    core_device_snapshot_sha256: str
    scheduled_kernel_launches: int
    core_kernel_launches: int
    publication_version: str
    standalone_publication_route: str
    external_shared_publication_route: str
    root_ingress_zero_start_fusions: int


def _canonical_scalar_static_arrays(hierarchy: StaticMultigridHierarchy) -> tuple[np.ndarray, ...]:
    """Return scalar-wrapper static arrays in its exact construction order."""
    arrays: list[np.ndarray] = [np.asarray(hierarchy.coarse_cholesky, dtype=np.float64).reshape(-1)]
    for level_index, level in enumerate(hierarchy.levels):
        matrix = level.matrix
        arrays.extend(
            (
                np.asarray(matrix.row_offsets, dtype=np.int32),
                np.asarray(matrix.column_indices, dtype=np.int32),
                np.asarray(matrix.values, dtype=np.float64).reshape(-1),
            )
        )
        if level_index == len(hierarchy.levels) - 1:
            continue
        if level.smoother is None or level.prolongation is None:
            raise RuntimeError("canonical noncoarse hierarchy level is incomplete")
        aggregate = np.asarray(level.prolongation.aggregate, dtype=np.int32)
        coarse_count = level.prolongation.coarse_node_count
        counts = np.bincount(aggregate, minlength=coarse_count)
        member_offsets = np.zeros(coarse_count + 1, dtype=np.int32)
        member_offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
        member_nodes = np.concatenate(
            [np.flatnonzero(aggregate == aggregate_id) for aggregate_id in range(coarse_count)]
        ).astype(np.int32, copy=False)
        arrays.extend(
            (
                np.asarray(level.smoother.inverse_diagonal, dtype=np.float64).reshape(-1),
                aggregate,
                np.asarray(level.prolongation.blocks, dtype=np.float64).reshape(-1),
                member_offsets,
                member_nodes,
            )
        )
    return tuple(arrays)


def _source_scalar_static_arrays(device_hierarchy: WarpStaticMultigridHierarchy) -> tuple[wp.array[Any], ...]:
    """Enumerate the source hierarchy arrays consumed by the scalar wrapper."""
    arrays: list[wp.array[Any]] = [device_hierarchy.coarse_cholesky]
    for level in device_hierarchy.levels:
        arrays.extend((level.row_offsets, level.column_indices, level.matrix_values))
        for value in (
            level.inverse_diagonal,
            level.aggregate,
            level.prolongation_blocks,
            level.member_offsets,
            level.member_fine_nodes,
        ):
            if value is not None:
                arrays.append(value)
    return tuple(arrays)


def _canonical_scalar_fused_evidence(hierarchy: StaticMultigridHierarchy) -> _ScalarFusedHierarchyEvidence:
    """Derive scalar schedule/static/snapshot hashes without trusting wrapper labels."""
    source_snapshot_sha256 = _hash_parts(
        "warp-static-multigrid-snapshot-v1",
        (
            ("hierarchy_sha256", hierarchy.content_sha256),
            ("kernel_version", SOURCE_V_CYCLE_KERNEL_VERSION),
            ("coarse_scalar_bound", MAX_COARSE_SCALAR_SIZE),
        ),
    )
    static_parts: list[tuple[str, object]] = [("hierarchy_sha256", hierarchy.content_sha256)]
    for index, array in enumerate(_canonical_scalar_static_arrays(hierarchy)):
        static_parts.append((f"static_array_{index}", array))
    static_device_content_sha256 = _hash_parts(
        "warp-scalar-fused-static-device-content-v1",
        tuple(static_parts),
    )
    level_shapes: list[int] = []
    transfer_paths: list[int] = []
    for level_index, level in enumerate(hierarchy.levels):
        level_shapes.extend(
            (
                level.matrix.block_row_count,
                level.matrix.block_size,
                level.matrix.stored_block_count,
            )
        )
        if level_index != len(hierarchy.levels) - 1:
            if level.prolongation is None:
                raise RuntimeError("canonical noncoarse hierarchy level is missing prolongation")
            transfer_paths.extend((level.matrix.block_size, level.prolongation.coarse_block_size))
    noncoarse = len(hierarchy.levels) - 1
    root_ingress_zero_start_fusions = int(noncoarse > 0)
    core_kernel_launches = (
        2
        + noncoarse * (2 + 2 * hierarchy.pre_smooth_steps + 2 * hierarchy.post_smooth_steps)
        - root_ingress_zero_start_fusions
    )
    scheduled_kernel_launches = core_kernel_launches + 1
    common_schedule_parts = (
        ("source_device_snapshot_sha256", source_snapshot_sha256),
        ("kernel_version", V_CYCLE_KERNEL_VERSION),
        ("schedule_version", V_CYCLE_SCHEDULE_VERSION),
        ("owner_parallelism", "one-owner-per-scalar-row"),
        ("pre_smooth_steps", hierarchy.pre_smooth_steps),
        ("post_smooth_steps", hierarchy.post_smooth_steps),
        ("level_shapes", np.asarray(level_shapes, dtype=np.int64)),
        ("transfer_block_paths", np.asarray(transfer_paths, dtype=np.int64)),
        (
            "root_ingress_route",
            "fused-vec3d-scalar-zero-start" if root_ingress_zero_start_fusions else "standalone-vec3d-scalar",
        ),
        ("root_ingress_zero_start_fusions", root_ingress_zero_start_fusions),
        ("noncoarse_result_buffer", "B"),
        ("coarsest_result_buffer", "A"),
        ("core_kernel_launches", core_kernel_launches),
        ("publication_version", V_CYCLE_PUBLICATION_VERSION),
    )
    core_schedule_sha256 = _hash_parts(
        "warp-scalar-fused-v-cycle-core-schedule-v2",
        (
            *common_schedule_parts,
            ("publication_route", V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE),
            ("publication_kernel_launches", 0),
            ("scheduled_kernel_launches", core_kernel_launches),
        ),
    )
    schedule_sha256 = _hash_parts(
        "warp-scalar-fused-v-cycle-schedule-v4",
        (
            *common_schedule_parts,
            ("publication_route", V_CYCLE_STANDALONE_PUBLICATION_ROUTE),
            ("publication_kernel_launches", 1),
            ("scheduled_kernel_launches", scheduled_kernel_launches),
        ),
    )
    core_device_snapshot_sha256 = _hash_parts(
        "warp-scalar-fused-static-multigrid-core-snapshot-v2",
        (
            ("source_device_snapshot_sha256", source_snapshot_sha256),
            ("static_device_content_sha256", static_device_content_sha256),
            ("core_schedule_sha256", core_schedule_sha256),
        ),
    )
    device_snapshot_sha256 = _hash_parts(
        "warp-scalar-fused-static-multigrid-snapshot-v4",
        (
            ("source_device_snapshot_sha256", source_snapshot_sha256),
            ("static_device_content_sha256", static_device_content_sha256),
            ("schedule_sha256", schedule_sha256),
        ),
    )
    return _ScalarFusedHierarchyEvidence(
        source_device_snapshot_sha256=source_snapshot_sha256,
        schedule_sha256=schedule_sha256,
        core_schedule_sha256=core_schedule_sha256,
        static_device_content_sha256=static_device_content_sha256,
        device_snapshot_sha256=device_snapshot_sha256,
        core_device_snapshot_sha256=core_device_snapshot_sha256,
        scheduled_kernel_launches=scheduled_kernel_launches,
        core_kernel_launches=core_kernel_launches,
        publication_version=V_CYCLE_PUBLICATION_VERSION,
        standalone_publication_route=V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
        external_shared_publication_route=V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
        root_ingress_zero_start_fusions=root_ingress_zero_start_fusions,
    )


def _validate_scalar_fused_hierarchy_inputs(
    scalar_hierarchy: WarpScalarFusedStaticMultigridHierarchy,
    source_hierarchy: WarpStaticMultigridHierarchy,
    hierarchy: StaticMultigridHierarchy,
    *,
    validate_device_content: bool,
) -> _ScalarFusedHierarchyEvidence:
    """Validate the exact wrapper topology and independently derived evidence."""
    if type(scalar_hierarchy) is not WarpScalarFusedStaticMultigridHierarchy:
        raise TypeError("scalar hierarchy must be an exact WarpScalarFusedStaticMultigridHierarchy")
    if type(source_hierarchy) is not WarpStaticMultigridHierarchy:
        raise TypeError("source hierarchy must be an exact WarpStaticMultigridHierarchy")
    expected = _canonical_scalar_fused_evidence(hierarchy)
    source_arrays = _source_scalar_static_arrays(source_hierarchy)
    expected_arrays = _canonical_scalar_static_arrays(hierarchy)
    expected_level_signature: list[int | float | None] = []
    actual_source_level_signature: list[int | float | None] = []
    for level in hierarchy.levels:
        expected_level_signature.extend(
            (
                level.matrix.block_row_count,
                level.matrix.block_size,
                level.matrix.scalar_size,
                level.matrix.stored_block_count,
                None if level.smoother is None else float(level.smoother.omega),
                None if level.prolongation is None else level.prolongation.coarse_node_count,
                None if level.prolongation is None else level.prolongation.coarse_block_size,
            )
        )
    for level in source_hierarchy.levels:
        actual_source_level_signature.extend(
            (
                level.block_row_count,
                level.block_size,
                level.scalar_size,
                level.stored_block_count,
                level.omega,
                level.coarse_node_count,
                level.coarse_block_size,
            )
        )
    _require_exact_array(
        source_hierarchy.free_vertices_host,
        np.asarray(hierarchy.free_vertices, dtype=np.int32),
        name="scalar_hierarchy.source_free_vertices_host",
    )
    if (
        source_hierarchy.hierarchy_sha256 != hierarchy.content_sha256
        or source_hierarchy.solver_contract != hierarchy.solver_contract
        or source_hierarchy.static_model_sha256 != hierarchy.static_model_sha256
        or source_hierarchy.pre_smooth_steps != hierarchy.pre_smooth_steps
        or source_hierarchy.post_smooth_steps != hierarchy.post_smooth_steps
        or source_hierarchy.n_free != hierarchy.levels[0].matrix.block_row_count
        or source_hierarchy.n_free_dofs != hierarchy.levels[0].matrix.scalar_size
        or source_hierarchy.device_snapshot_sha256 != expected.source_device_snapshot_sha256
        or len(source_hierarchy.levels) != len(hierarchy.levels)
        or tuple(actual_source_level_signature) != tuple(expected_level_signature)
        or scalar_hierarchy.source_hierarchy is not source_hierarchy
        or scalar_hierarchy._source_identity != id(source_hierarchy)
        or scalar_hierarchy.device != source_hierarchy.device
        or scalar_hierarchy.hierarchy_sha256 != hierarchy.content_sha256
        or scalar_hierarchy.solver_contract != hierarchy.solver_contract
        or scalar_hierarchy.static_model_sha256 != hierarchy.static_model_sha256
        or scalar_hierarchy.free_vertices_host is not source_hierarchy.free_vertices_host
        or scalar_hierarchy.pre_smooth_steps != hierarchy.pre_smooth_steps
        or scalar_hierarchy.post_smooth_steps != hierarchy.post_smooth_steps
        or scalar_hierarchy.n_free != hierarchy.levels[0].matrix.block_row_count
        or scalar_hierarchy.n_free_dofs != hierarchy.levels[0].matrix.scalar_size
        or scalar_hierarchy.levels is not source_hierarchy.levels
        or scalar_hierarchy.coarse_cholesky is not source_hierarchy.coarse_cholesky
        or scalar_hierarchy.source_device_snapshot_sha256 != expected.source_device_snapshot_sha256
        or scalar_hierarchy.schedule_sha256 != expected.schedule_sha256
        or scalar_hierarchy.core_schedule_sha256 != expected.core_schedule_sha256
        or scalar_hierarchy.static_device_content_sha256 != expected.static_device_content_sha256
        or scalar_hierarchy.device_snapshot_sha256 != expected.device_snapshot_sha256
        or scalar_hierarchy.core_device_snapshot_sha256 != expected.core_device_snapshot_sha256
        or scalar_hierarchy.scheduled_kernel_launches != expected.scheduled_kernel_launches
        or scalar_hierarchy.core_kernel_launches != expected.core_kernel_launches
        or scalar_hierarchy._static_level_signature != tuple(expected_level_signature)
        or len(scalar_hierarchy._static_array_objects) != len(source_arrays)
        or any(
            actual is not expected
            for actual, expected in zip(scalar_hierarchy._static_array_objects, source_arrays, strict=True)
        )
        or scalar_hierarchy._static_array_pointers != tuple(int(array.ptr) for array in source_arrays)
    ):
        raise RuntimeError("persistent scalar-fused hierarchy schedule or identity changed")
    if len(source_arrays) != len(expected_arrays):
        raise RuntimeError("persistent scalar-fused hierarchy static-array topology changed")
    if validate_device_content:
        static_parts: list[tuple[str, object]] = [("hierarchy_sha256", hierarchy.content_sha256)]
        for index, (device_array, canonical_array) in enumerate(zip(source_arrays, expected_arrays, strict=True)):
            actual = np.asarray(device_array.numpy())
            _require_exact_array(actual, canonical_array, name=f"scalar_hierarchy.static_array_{index}")
            static_parts.append((f"static_array_{index}", actual))
        actual_static_sha256 = _hash_parts(
            "warp-scalar-fused-static-device-content-v1",
            tuple(static_parts),
        )
        if actual_static_sha256 != expected.static_device_content_sha256:
            raise RuntimeError("persistent scalar-fused static device content digest changed")
    return expected


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
            ("kernel_version", SOURCE_V_CYCLE_KERNEL_VERSION),
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
    first_cycle: WarpScalarFusedVCycleWorkspace
    second_cycle: WarpScalarFusedVCycleWorkspace
    operator_apply: WarpMatrixFreeWorkspace


class _ScalarCycleWorkspaceOwnerBinding(NamedTuple):
    """Construction-time identity binding for one scalar-fused workspace."""

    workspace: WarpScalarFusedVCycleWorkspace
    hierarchy: WarpScalarFusedStaticMultigridHierarchy
    rhs: wp.array[wp.vec3d]
    correction: wp.array[wp.vec3d]
    coarse_intermediate: wp.array[wp.float64]
    final_scalar_correction: wp.array[wp.float64]
    final_scalar_pointer: int
    level_rhs: tuple[wp.array[wp.float64], ...]
    level_correction: tuple[wp.array[wp.float64], ...]
    level_correction_alt: tuple[wp.array[wp.float64], ...]
    level_residual: tuple[wp.array[wp.float64], ...]
    persistent_arrays: tuple[wp.array[Any], ...]
    persistent_pointers: tuple[int, ...]


class _OuterWorkspaceOwnerBinding(NamedTuple):
    """Construction-time identity binding for one direct outer workspace."""

    workspace: _OuterWorkspace
    rhs: wp.array[wp.vec3d]
    first_correction: wp.array[wp.vec3d]
    operator_product_after_first: wp.array[wp.vec3d]
    residual_after_first: wp.array[wp.vec3d]
    second_correction: wp.array[wp.vec3d]
    direction: wp.array[wp.vec3d]
    first_cycle: _ScalarCycleWorkspaceOwnerBinding
    second_cycle: _ScalarCycleWorkspaceOwnerBinding
    operator_apply: WarpMatrixFreeWorkspace
    operator_apply_delta_piola: wp.array[wp.mat33d]


class _PublicVBDOwnerBinding(NamedTuple):
    """Construction-time public SolverVBD lane and color owners."""

    baseline: CapturedPublicVBDBaseline
    lanes: dict[int, object]
    lane_keys: tuple[int, int]
    k1_lane: object
    k4_lane: object
    model: object
    control: object
    pristine_input: object
    pristine_output: object
    particle_color_groups: list[wp.array[wp.int32]]
    particle_color_group_arrays: tuple[wp.array[wp.int32], ...]
    k1_solver: SolverVBD
    k4_solver: SolverVBD
    k1_state_in: object
    k1_state_out: object
    k4_state_in: object
    k4_state_out: object


class _DirectCorrectionOwnerBinding(NamedTuple):
    """Construction-time identity binding for every direct graph buffer."""

    canonical_positions: wp.array[wp.vec3d]
    x_current: wp.array[wp.vec3d]
    candidate: wp.array[wp.vec3d]
    proposal_finite: wp.array[wp.int32]
    final_positions: wp.array[wp.vec3]
    final_velocities: wp.array[wp.vec3]
    active: wp.array[wp.int32]
    accepted: wp.array[wp.int32]
    reasons: wp.array[wp.int32]
    current_inertia: wp.array[wp.float64]
    candidate_inertia: wp.array[wp.float64]
    vertex_finite: wp.array[wp.int32]
    current_elastic: wp.array[wp.float64]
    candidate_elastic: wp.array[wp.float64]
    candidate_determinants: wp.array[wp.float64]
    segment_minima: wp.array[wp.float64]
    tet_finite: wp.array[wp.int32]
    directional_terms: wp.array[wp.float64]
    outer_start_positions: tuple[wp.array[wp.vec3d], ...]
    outer_candidate_positions: tuple[wp.array[wp.vec3d], ...]
    initial_objectives: wp.array[wp.float64]
    candidate_objectives: wp.array[wp.float64]
    directional_derivatives: wp.array[wp.float64]
    minimum_segment_determinants: wp.array[wp.float64]


class _DeviceHierarchyLevelOwnerBinding(NamedTuple):
    """Construction-time owner graph for one shared device hierarchy level."""

    level: object
    row_offsets: wp.array[wp.int32]
    column_indices: wp.array[wp.int32]
    matrix_values: wp.array[wp.float64]
    inverse_diagonal: wp.array[wp.float64] | None
    aggregate: wp.array[wp.int32] | None
    prolongation_blocks: wp.array[wp.float64] | None
    member_offsets: wp.array[wp.int32] | None
    member_fine_nodes: wp.array[wp.int32] | None


class _DeviceHierarchyOwnerBinding(NamedTuple):
    """Exact source/scalar hierarchy containers and shared level owners."""

    source: WarpStaticMultigridHierarchy
    scalar: WarpScalarFusedStaticMultigridHierarchy
    source_levels: tuple[object, ...]
    scalar_levels: tuple[object, ...]
    levels: tuple[_DeviceHierarchyLevelOwnerBinding, ...]
    coarse_cholesky: wp.array[wp.float64]


class _PersistentArrayOwnerBinding(NamedTuple):
    """Exact ordered owner/signature set for every captured-graph array."""

    arrays: tuple[tuple[str, wp.array[Any]], ...]
    signatures: tuple[tuple[str, object], ...]


class _NativeGraphOwnerBinding(NamedTuple):
    """Exact Warp Graph wrapper attributes and native launch handles."""

    graph: object
    attributes: dict[str, object]
    attribute_names: tuple[str, ...]
    device: object
    capture_id: int
    module_execs: set[object]
    module_exec_members: tuple[object, ...]
    graph_exec: ctypes.c_void_p
    graph_exec_value: int
    graph_handle: ctypes.c_void_p
    graph_handle_value: int


class _ReplayStreamOwnerBinding(NamedTuple):
    """Exact dedicated replay stream and native device/stream handles."""

    stream: object
    attributes: dict[str, object]
    attribute_names: tuple[str, ...]
    device: object
    device_context: int
    cuda_stream: int
    owner: bool
    cached_event: object | None


class _CaptureGraphOwnerBinding(NamedTuple):
    """Exact immutable owner and identity claims for one capture generation."""

    graph: object
    k4_graph: object
    graph_type: type
    graph_native: _NativeGraphOwnerBinding
    k4_graph_native: _NativeGraphOwnerBinding
    replay_stream: _ReplayStreamOwnerBinding
    generation: int
    object_identity: tuple[int, int]
    graph_identity_sha256: str
    k4_graph_identity_sha256: str


class _ConstructionClaimsOwnerBinding(NamedTuple):
    """Construction-derived anchors independent of mutable solver labels."""

    array_descriptor_type: object
    array_descriptor_fields: tuple[object, ...]
    fused_gather_kernel_version: str
    scalar_direction_apply_kernel_version: str
    scalar_fused_evidence: _ScalarFusedHierarchyEvidence
    first_cycle_publication_role: str
    second_cycle_publication_role: str
    outer_kernel_version: str
    outer_schedule_version: str
    outer_schedule_sha256: str
    finalize_gate_route: str
    finalize_gate_block_dim: int
    finalize_gate_owner_threads: tuple[int, int, int, int]
    finalize_gate_owner_roles: tuple[str, str, str, str]
    finalize_gate_collective_version: str
    solver_lane_contract_sha256: str
    hierarchy_owner_identity: tuple[int, int]
    solver_graph_owner_identity: tuple[tuple[str, int], ...]
    solver_scalar_sha256: str
    solver_static_array_sha256: str
    content_identity: tuple[object, ...]
    persistent_device_sha256: str | None
    uncaptured_graph_identity_sha256: str | None


class _WorkspaceOwnerBinding(NamedTuple):
    """Private exact construction owner graph for every captured resource."""

    scene: TetBenchmarkScene
    config: DirectGraphVBDConfig
    warp_device: object
    replay_stream: _ReplayStreamOwnerBinding
    public_vbd: _PublicVBDOwnerBinding
    hierarchy: StaticMultigridHierarchy
    device: _DeviceHierarchyOwnerBinding
    operator: WarpMatrixFreeStableNHOperator
    problem: object
    construction_operator: MatrixFreeStableNHOperator
    construction_k1: CapturedVBDEndpoint
    construction_k4: CapturedVBDEndpoint
    direct: _DirectCorrectionOwnerBinding
    workspaces: tuple[_OuterWorkspace, ...]
    outer: tuple[_OuterWorkspaceOwnerBinding, ...]
    persistent: _PersistentArrayOwnerBinding | None
    claims: _ConstructionClaimsOwnerBinding
    capture: _CaptureGraphOwnerBinding | None


def _make_workspace_owner_registry():
    """Create a non-replaceable weak construction-time owner registry."""
    registry: weakref.WeakKeyDictionary[object, _WorkspaceOwnerBinding] = weakref.WeakKeyDictionary()

    def register(owner: object, binding: _WorkspaceOwnerBinding) -> None:
        if owner in registry:
            raise RuntimeError("captured correction workspace owners are already registered")
        registry[owner] = binding

    def lookup(owner: object) -> _WorkspaceOwnerBinding:
        binding = registry.get(owner)
        if type(binding) is not _WorkspaceOwnerBinding:
            raise RuntimeError("captured correction workspace owners are not registered")
        return binding

    def replace(
        owner: object,
        expected: _WorkspaceOwnerBinding,
        replacement: _WorkspaceOwnerBinding,
    ) -> None:
        try:
            registered = registry[owner]
        except KeyError as error:
            raise RuntimeError("captured correction workspace owners are not registered") from error
        if registered is not expected:
            raise RuntimeError("captured correction workspace owner generation changed concurrently")
        if type(replacement) is not _WorkspaceOwnerBinding:
            raise RuntimeError("replacement captured correction workspace owners have an invalid exact type")
        registry[owner] = replacement

    return register, lookup, replace


_register_workspace_owners, _lookup_workspace_owners, _replace_workspace_owners = _make_workspace_owner_registry()
del _make_workspace_owner_registry


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
        self._replay_stream = wp.Stream(self.device)
        self.baseline = CapturedPublicVBDBaseline(scene, device=str(self.device), tile_solve=tile_solve)
        reference_adjacency = self.baseline._lane(1).solver.particle_adjacency
        reference_adjacency_fields = dict(getattr(type(reference_adjacency._ctype), "_fields_", ()))
        self._array_descriptor_type_bound = reference_adjacency_fields["v_adj_tets"]
        self._array_descriptor_fields_bound = tuple(getattr(self._array_descriptor_type_bound, "_fields_", ()))
        self._fused_gather_kernel_version_bound = FUSED_GATHER_KERNEL_VERSION
        self._scalar_direction_apply_kernel_version_bound = SCALAR_DIRECTION_APPLY_KERNEL_VERSION
        self._first_cycle_publication_role_bound = FIRST_CYCLE_PUBLICATION_ROLE
        self._second_cycle_publication_role_bound = SECOND_CYCLE_PUBLICATION_ROLE
        self._outer_kernel_version_bound = OUTER_KERNEL_VERSION
        self._outer_schedule_version_bound = OUTER_SCHEDULE_VERSION
        self._outer_schedule_sha256_bound = OUTER_SCHEDULE_SHA256
        self._finalize_gate_route_bound = FINALIZE_GATE_ROUTE
        self._finalize_gate_block_dim_bound = FINALIZE_GATE_BLOCK_DIM
        self._finalize_gate_owner_threads_bound = FINALIZE_GATE_OWNER_THREADS
        self._finalize_gate_owner_roles_bound = FINALIZE_GATE_OWNER_ROLES
        self._finalize_gate_collective_version_bound = FINALIZE_GATE_COLLECTIVE_VERSION
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
        self.source_device_hierarchy = WarpStaticMultigridHierarchy.from_hierarchy(
            self.hierarchy,
            device=str(self.device),
        )
        self.device_hierarchy = WarpScalarFusedStaticMultigridHierarchy.from_device_hierarchy(
            self.source_device_hierarchy
        )
        self._hierarchy_owner_identity_bound = (id(self.source_device_hierarchy), id(self.device_hierarchy))

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
        owner_binding = self._capture_workspace_owner_binding()
        persistent = owner_binding.persistent
        if type(persistent) is not _PersistentArrayOwnerBinding:
            raise RuntimeError("captured construction did not bind the persistent array owner graph")
        claims = owner_binding.claims
        self._hierarchy_owner_identity_bound = claims.hierarchy_owner_identity
        self._solver_lane_contract_sha256_bound = claims.solver_lane_contract_sha256
        self._solver_graph_owner_identity_bound = claims.solver_graph_owner_identity
        self._solver_scalar_sha256_bound = claims.solver_scalar_sha256
        self._solver_static_array_sha256_bound = claims.solver_static_array_sha256
        self._persistent_array_identity = persistent.signatures
        self._construction_content_identity = claims.content_identity
        persistent_device_sha256 = self._validate_persistent_sources(
            require_bound=False,
            owner_binding=owner_binding,
        )
        owner_binding = owner_binding._replace(
            claims=claims._replace(persistent_device_sha256=persistent_device_sha256)
        )
        uncaptured_graph_identity_sha256 = self._derive_graph_identity(
            captured=False,
            comparator=False,
            owner_binding=owner_binding,
            graph=None,
            generation=0,
        )
        owner_binding = owner_binding._replace(
            claims=owner_binding.claims._replace(
                uncaptured_graph_identity_sha256=uncaptured_graph_identity_sha256,
            )
        )
        self._persistent_device_sha256 = persistent_device_sha256
        self._construction_persistent_device_sha256 = persistent_device_sha256
        self._uncaptured_graph_identity_sha256 = uncaptured_graph_identity_sha256
        self.graph_identity_sha256 = self._derive_graph_identity(
            captured=True,
            comparator=False,
            owner_binding=owner_binding,
            graph=None,
            generation=0,
        )
        self.k4_graph_identity_sha256 = self._derive_graph_identity(
            captured=True,
            comparator=True,
            owner_binding=owner_binding,
            graph=None,
            generation=0,
        )
        self._validate_persistent_sources(owner_binding=owner_binding)
        self._register_construction_owner_binding(owner_binding)

    @property
    def linear_kernel_launches_per_outer(self) -> int:
        """Retained linear work launches, including the fused direction owner."""
        return self.linear_prefix_kernel_launches_per_outer + 1

    @property
    def linear_prefix_kernel_launches_per_outer(self) -> int:
        """Pure current-operator and two-V-cycle launches before vertex fusion."""
        return 4 + 2 * self.device_hierarchy.core_kernel_launches

    @property
    def correction_kernel_launches(self) -> int:
        """Exact correction launches, excluding the public K1 graph prefix."""
        return 2 + OUTER_CORRECTIONS * self.outer_kernel_launches_per_outer

    @property
    def outer_kernel_launches_per_outer(self) -> int:
        """Exact linear, fused vertex, tet, four-warp gate, and commit launches."""
        return self.linear_kernel_launches_per_outer + 3

    def _register_construction_owner_binding(self, binding: _WorkspaceOwnerBinding) -> None:
        """Register one exact construction binding before any public boundary."""
        if type(binding) is not _WorkspaceOwnerBinding:
            raise RuntimeError("captured construction owner binding has an invalid exact type")
        _register_workspace_owners(self, binding)

    def _capture_public_vbd_owner_binding(self) -> _PublicVBDOwnerBinding:
        """Capture exact public K1/K4 lane and particle-color containers."""
        baseline = self.baseline
        lanes = baseline._lanes
        if type(lanes) is not dict or tuple(lanes) != (1, 4):
            raise RuntimeError("public VBD lanes must be the exact built-in K1/K4 dictionary")
        k1_lane = lanes[1]
        k4_lane = lanes[4]
        groups = baseline.model.particle_color_groups
        if type(groups) is not list or not groups:
            raise RuntimeError("public VBD particle color groups must be one nonempty built-in list")
        group_arrays = tuple(groups)
        if any(not isinstance(group, wp.array) or group.dtype is not wp.int32 for group in group_arrays):
            raise RuntimeError("public VBD particle color groups must contain int32 Warp arrays")
        return _PublicVBDOwnerBinding(
            baseline=baseline,
            lanes=lanes,
            lane_keys=(1, 4),
            k1_lane=k1_lane,
            k4_lane=k4_lane,
            model=baseline.model,
            control=baseline.control,
            pristine_input=baseline.pristine_input,
            pristine_output=baseline.pristine_output,
            particle_color_groups=groups,
            particle_color_group_arrays=group_arrays,
            k1_solver=k1_lane.solver,
            k4_solver=k4_lane.solver,
            k1_state_in=k1_lane.state_in,
            k1_state_out=k1_lane.state_out,
            k4_state_in=k4_lane.state_in,
            k4_state_out=k4_lane.state_out,
        )

    def _capture_device_hierarchy_owner_binding(self) -> _DeviceHierarchyOwnerBinding:
        """Capture exact hierarchy tuples, levels, and level-array owners."""
        source = self.source_device_hierarchy
        scalar = self.device_hierarchy
        source_levels = source.levels
        scalar_levels = scalar.levels
        if (
            type(source_levels) is not tuple
            or type(scalar_levels) is not tuple
            or source_levels is not scalar_levels
            or not source_levels
        ):
            raise RuntimeError("source/scalar device hierarchy levels must share one exact nonempty tuple")
        levels = tuple(
            _DeviceHierarchyLevelOwnerBinding(
                level=level,
                row_offsets=level.row_offsets,
                column_indices=level.column_indices,
                matrix_values=level.matrix_values,
                inverse_diagonal=level.inverse_diagonal,
                aggregate=level.aggregate,
                prolongation_blocks=level.prolongation_blocks,
                member_offsets=level.member_offsets,
                member_fine_nodes=level.member_fine_nodes,
            )
            for level in source_levels
        )
        return _DeviceHierarchyOwnerBinding(
            source=source,
            scalar=scalar,
            source_levels=source_levels,
            scalar_levels=scalar_levels,
            levels=levels,
            coarse_cholesky=source.coarse_cholesky,
        )

    def _capture_direct_correction_owner_binding(self) -> _DirectCorrectionOwnerBinding:
        """Capture every direct graph array and both fixed four-slot tuples."""
        if (
            type(self.outer_start_positions) is not tuple
            or type(self.outer_candidate_positions) is not tuple
            or len(self.outer_start_positions) != OUTER_CORRECTIONS
            or len(self.outer_candidate_positions) != OUTER_CORRECTIONS
        ):
            raise RuntimeError("direct outer position buffers must be exact four-slot tuples")
        return _DirectCorrectionOwnerBinding(
            canonical_positions=self.canonical_positions,
            x_current=self.x_current,
            candidate=self.candidate,
            proposal_finite=self.proposal_finite,
            final_positions=self.final_positions,
            final_velocities=self.final_velocities,
            active=self.active,
            accepted=self.accepted,
            reasons=self.reasons,
            current_inertia=self.current_inertia,
            candidate_inertia=self.candidate_inertia,
            vertex_finite=self.vertex_finite,
            current_elastic=self.current_elastic,
            candidate_elastic=self.candidate_elastic,
            candidate_determinants=self.candidate_determinants,
            segment_minima=self.segment_minima,
            tet_finite=self.tet_finite,
            directional_terms=self.directional_terms,
            outer_start_positions=self.outer_start_positions,
            outer_candidate_positions=self.outer_candidate_positions,
            initial_objectives=self.initial_objectives,
            candidate_objectives=self.candidate_objectives,
            directional_derivatives=self.directional_derivatives,
            minimum_segment_determinants=self.minimum_segment_determinants,
        )

    def _construction_content_identity_record(self) -> tuple[object, ...]:
        """Return construction labels that must remain canonical cached claims."""
        return (
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
            self._fused_gather_kernel_version_bound,
            self._scalar_direction_apply_kernel_version_bound,
            V_CYCLE_KERNEL_VERSION,
            V_CYCLE_SCHEDULE_VERSION,
            V_CYCLE_PUBLICATION_VERSION,
            V_CYCLE_STANDALONE_PUBLICATION_ROUTE,
            V_CYCLE_EXTERNAL_SHARED_PUBLICATION_ROUTE,
            self._first_cycle_publication_role_bound,
            self._second_cycle_publication_role_bound,
            self._outer_kernel_version_bound,
            self._outer_schedule_version_bound,
            self._outer_schedule_sha256_bound,
            self._finalize_gate_route_bound,
            self._finalize_gate_block_dim_bound,
            self._finalize_gate_owner_threads_bound,
            self._finalize_gate_owner_roles_bound,
            self._finalize_gate_collective_version_bound,
        )

    def _capture_construction_claims(self) -> _ConstructionClaimsOwnerBinding:
        """Compute immutable construction anchors before registry publication."""
        return _ConstructionClaimsOwnerBinding(
            array_descriptor_type=self._array_descriptor_type_bound,
            array_descriptor_fields=self._array_descriptor_fields_bound,
            fused_gather_kernel_version=self._fused_gather_kernel_version_bound,
            scalar_direction_apply_kernel_version=self._scalar_direction_apply_kernel_version_bound,
            scalar_fused_evidence=_canonical_scalar_fused_evidence(self.hierarchy),
            first_cycle_publication_role=self._first_cycle_publication_role_bound,
            second_cycle_publication_role=self._second_cycle_publication_role_bound,
            outer_kernel_version=self._outer_kernel_version_bound,
            outer_schedule_version=self._outer_schedule_version_bound,
            outer_schedule_sha256=self._outer_schedule_sha256_bound,
            finalize_gate_route=self._finalize_gate_route_bound,
            finalize_gate_block_dim=self._finalize_gate_block_dim_bound,
            finalize_gate_owner_threads=self._finalize_gate_owner_threads_bound,
            finalize_gate_owner_roles=self._finalize_gate_owner_roles_bound,
            finalize_gate_collective_version=self._finalize_gate_collective_version_bound,
            solver_lane_contract_sha256=self._solver_lane_contract_sha256_bound,
            hierarchy_owner_identity=(id(self.source_device_hierarchy), id(self.device_hierarchy)),
            solver_graph_owner_identity=self._solver_graph_owner_identity(),
            solver_scalar_sha256=self._solver_scalar_sha256(),
            solver_static_array_sha256=self._solver_static_array_sha256(),
            content_identity=self._construction_content_identity_record(),
            persistent_device_sha256=None,
            uncaptured_graph_identity_sha256=None,
        )

    @staticmethod
    def _capture_cycle_workspace_owner_binding(
        workspace: WarpScalarFusedVCycleWorkspace,
        hierarchy: WarpScalarFusedStaticMultigridHierarchy,
    ) -> _ScalarCycleWorkspaceOwnerBinding:
        """Capture exact scalar-workspace objects and tuple containers once."""
        if type(workspace) is not WarpScalarFusedVCycleWorkspace or workspace.hierarchy is not hierarchy:
            raise RuntimeError("scalar-fused workspace construction owner is invalid")
        for name in (
            "level_rhs",
            "level_correction",
            "level_correction_alt",
            "level_residual",
            "_persistent_arrays",
            "_persistent_pointers",
        ):
            if type(getattr(workspace, name)) is not tuple:
                raise RuntimeError(f"scalar-fused workspace {name} must be an exact tuple")
        return _ScalarCycleWorkspaceOwnerBinding(
            workspace=workspace,
            hierarchy=hierarchy,
            rhs=workspace.rhs,
            correction=workspace.correction,
            coarse_intermediate=workspace.coarse_intermediate,
            final_scalar_correction=workspace.final_scalar_correction,
            final_scalar_pointer=int(workspace.final_scalar_correction.ptr),
            level_rhs=workspace.level_rhs,
            level_correction=workspace.level_correction,
            level_correction_alt=workspace.level_correction_alt,
            level_residual=workspace.level_residual,
            persistent_arrays=workspace._persistent_arrays,
            persistent_pointers=workspace._persistent_pointers,
        )

    def _capture_replay_stream_owner_binding(self) -> _ReplayStreamOwnerBinding:
        """Capture the dedicated replay stream and native device handles."""
        stream = self._replay_stream
        attributes = vars(stream)
        expected_names = ("cuda_stream", "owner", "_cached_event", "device")
        device_context = getattr(self.device, "_context", None)
        if (
            type(stream) is not wp.Stream
            or type(attributes) is not dict
            or tuple(attributes) != expected_names
            or attributes["device"] is not self.device
            or type(device_context) is not int
            or device_context <= 0
            or type(attributes["cuda_stream"]) is not int
            or attributes["cuda_stream"] <= 0
            or type(attributes["owner"]) is not bool
            or attributes["owner"] is not True
            or attributes["_cached_event"] is not None
        ):
            raise RuntimeError("dedicated captured replay stream has invalid native owners")
        return _ReplayStreamOwnerBinding(
            stream=stream,
            attributes=attributes,
            attribute_names=expected_names,
            device=self.device,
            device_context=device_context,
            cuda_stream=attributes["cuda_stream"],
            owner=attributes["owner"],
            cached_event=attributes["_cached_event"],
        )

    def _capture_workspace_owner_binding(self) -> _WorkspaceOwnerBinding:
        """Build the exact construction-time public, hierarchy, and work owners."""
        if type(self.operator) is not WarpMatrixFreeStableNHOperator:
            raise RuntimeError("matrix-free operator construction owner is invalid")
        if type(self.workspaces) is not tuple or len(self.workspaces) != OUTER_CORRECTIONS:
            raise RuntimeError("captured correction workspaces must be one exact four-slot tuple")
        outer_bindings: list[_OuterWorkspaceOwnerBinding] = []
        for workspace in self.workspaces:
            if type(workspace) is not _OuterWorkspace:
                raise RuntimeError("captured correction outer workspace has an invalid exact type")
            if type(workspace.operator_apply) is not WarpMatrixFreeWorkspace:
                raise RuntimeError("matrix-free apply workspace has an invalid exact type")
            outer_binding = _OuterWorkspaceOwnerBinding(
                workspace=workspace,
                rhs=workspace.rhs,
                first_correction=workspace.first_correction,
                operator_product_after_first=workspace.operator_product_after_first,
                residual_after_first=workspace.residual_after_first,
                second_correction=workspace.second_correction,
                direction=workspace.direction,
                first_cycle=self._capture_cycle_workspace_owner_binding(
                    workspace.first_cycle,
                    self.device_hierarchy,
                ),
                second_cycle=self._capture_cycle_workspace_owner_binding(
                    workspace.second_cycle,
                    self.device_hierarchy,
                ),
                operator_apply=workspace.operator_apply,
                operator_apply_delta_piola=workspace.operator_apply.delta_piola,
            )
            self._validate_external_publication_aliases(outer_binding, name="construction outer")
            outer_bindings.append(outer_binding)
        binding = _WorkspaceOwnerBinding(
            scene=self.scene,
            config=self.config,
            warp_device=self.device,
            replay_stream=self._capture_replay_stream_owner_binding(),
            public_vbd=self._capture_public_vbd_owner_binding(),
            hierarchy=self.hierarchy,
            device=self._capture_device_hierarchy_owner_binding(),
            operator=self.operator,
            problem=self.problem,
            construction_operator=self._construction_operator,
            construction_k1=self._construction_k1,
            construction_k4=self._construction_k4,
            direct=self._capture_direct_correction_owner_binding(),
            workspaces=self.workspaces,
            outer=tuple(outer_bindings),
            persistent=None,
            claims=self._capture_construction_claims(),
            capture=None,
        )
        persistent_arrays = self._persistent_input_arrays(binding)
        persistent_signatures = self._persistent_array_signatures_from_items(
            persistent_arrays,
            descriptor_type=binding.claims.array_descriptor_type,
            descriptor_fields=binding.claims.array_descriptor_fields,
        )
        return binding._replace(
            persistent=_PersistentArrayOwnerBinding(
                arrays=persistent_arrays,
                signatures=persistent_signatures,
            )
        )

    def _validate_external_publication_aliases(
        self,
        binding: _OuterWorkspaceOwnerBinding,
        *,
        name: str,
    ) -> None:
        """Reject every source/output overlap in both shared publications."""
        retained = (
            ("rhs", binding.rhs),
            ("first_correction", binding.first_correction),
            ("operator_product_after_first", binding.operator_product_after_first),
            ("residual_after_first", binding.residual_after_first),
            ("second_correction", binding.second_correction),
            ("direction", binding.direction),
        )
        second_internal = tuple(
            (f"second_cycle.persistent[{index}]", value)
            for index, value in enumerate(binding.second_cycle.persistent_arrays)
        )
        first_internal = tuple(
            (f"first_cycle.persistent[{index}]", value)
            for index, value in enumerate(binding.first_cycle.persistent_arrays)
        )
        hierarchy = binding.second_cycle.hierarchy
        delta_piola = binding.operator_apply_delta_piola
        for cycle_name, cycle in (("first", binding.first_cycle), ("second", binding.second_cycle)):
            final_scalar = cycle.final_scalar_correction
            if final_scalar is not cycle.workspace.final_scalar_correction:
                raise RuntimeError(f"{name} {cycle_name}-cycle final scalar owner changed")
            if int(final_scalar.ptr) != cycle.final_scalar_pointer:
                raise RuntimeError(f"{name} {cycle_name}-cycle final scalar pointer changed")
            if not any(final_scalar is value for value in cycle.persistent_arrays):
                raise RuntimeError(f"{name} {cycle_name}-cycle final scalar is outside its persistent owner tuple")
            for field_name, value in (*retained, ("operator_apply.delta_piola", delta_piola)):
                if hierarchy._arrays_overlap(final_scalar, value):
                    raise RuntimeError(f"{name} {cycle_name}-cycle final scalar aliases {field_name}")

        publications = (
            ("first_correction", binding.first_correction),
            ("second_correction", binding.second_correction),
        )
        for output_name, output in publications:
            for field_name, value in (
                *retained,
                *first_internal,
                *second_internal,
                ("operator_apply.delta_piola", delta_piola),
            ):
                if value is not output and hierarchy._arrays_overlap(output, value):
                    raise RuntimeError(f"{name} {output_name} aliases {field_name}")

        for left_name, left, other_internal in (
            ("first", binding.first_cycle.final_scalar_correction, second_internal),
            ("second", binding.second_cycle.final_scalar_correction, first_internal),
        ):
            for field_name, value in other_internal:
                if hierarchy._arrays_overlap(left, value):
                    raise RuntimeError(f"{name} {left_name}-cycle final scalar aliases {field_name}")

    def _validate_cycle_workspace_owner_binding(
        self,
        workspace: object,
        binding: _ScalarCycleWorkspaceOwnerBinding,
        *,
        name: str,
    ) -> None:
        """Require one exact registered scalar owner, arrays, and containers."""
        if (
            type(workspace) is not WarpScalarFusedVCycleWorkspace
            or workspace is not binding.workspace
            or workspace.hierarchy is not binding.hierarchy
            or workspace.hierarchy is not self.device_hierarchy
            or type(binding.final_scalar_pointer) is not int
            or int(binding.final_scalar_correction.ptr) != binding.final_scalar_pointer
        ):
            raise RuntimeError(f"{name} scalar-fused workspace owner object changed")
        array_bindings = (
            ("rhs", binding.rhs),
            ("correction", binding.correction),
            ("coarse_intermediate", binding.coarse_intermediate),
            ("final_scalar_correction", binding.final_scalar_correction),
        )
        for field_name, expected in array_bindings:
            if getattr(workspace, field_name) is not expected:
                raise RuntimeError(f"{name} scalar-fused workspace {field_name} owner changed")
        container_bindings = (
            ("level_rhs", binding.level_rhs),
            ("level_correction", binding.level_correction),
            ("level_correction_alt", binding.level_correction_alt),
            ("level_residual", binding.level_residual),
            ("_persistent_arrays", binding.persistent_arrays),
            ("_persistent_pointers", binding.persistent_pointers),
        )
        for field_name, expected in container_bindings:
            actual = getattr(workspace, field_name)
            if type(actual) is not tuple or actual is not expected:
                raise RuntimeError(f"{name} scalar-fused workspace {field_name} container changed")
        self.device_hierarchy._validate_workspace(workspace)

    def _validate_public_vbd_owner_binding(self, binding: _PublicVBDOwnerBinding) -> None:
        """Require exact built-in lane/color containers and all nested owners."""
        baseline = self.baseline
        if type(binding) is not _PublicVBDOwnerBinding or baseline is not binding.baseline:
            raise RuntimeError("persistent public VBD baseline owner object changed")
        if (
            type(baseline._lanes) is not dict
            or baseline._lanes is not binding.lanes
            or tuple(baseline._lanes) != binding.lane_keys
            or baseline._lanes.get(1) is not binding.k1_lane
            or baseline._lanes.get(4) is not binding.k4_lane
        ):
            raise RuntimeError("persistent public VBD K1/K4 lane dictionary changed")
        if (
            baseline.model is not binding.model
            or baseline.control is not binding.control
            or baseline.pristine_input is not binding.pristine_input
            or baseline.pristine_output is not binding.pristine_output
        ):
            raise RuntimeError("persistent public VBD model/control/pristine owner changed")
        lane_fields = (
            (binding.k1_lane, 1, binding.k1_solver, binding.k1_state_in, binding.k1_state_out),
            (binding.k4_lane, 4, binding.k4_solver, binding.k4_state_in, binding.k4_state_out),
        )
        for lane, iterations, solver, state_in, state_out in lane_fields:
            if (
                type(lane.iterations) is not int
                or lane.iterations != iterations
                or lane.solver is not solver
                or lane.state_in is not state_in
                or lane.state_out is not state_out
                or type(solver) is not SolverVBD
                or solver.model is not binding.model
            ):
                raise RuntimeError(f"persistent public K{iterations} lane nested owner changed")
        groups = binding.model.particle_color_groups
        if (
            type(groups) is not list
            or groups is not binding.particle_color_groups
            or len(groups) != len(binding.particle_color_group_arrays)
            or any(
                actual is not expected
                for actual, expected in zip(groups, binding.particle_color_group_arrays, strict=True)
            )
        ):
            raise RuntimeError("persistent public VBD particle color-group list or order changed")

    def _validate_device_hierarchy_owner_binding(self, binding: _DeviceHierarchyOwnerBinding) -> None:
        """Require exact source/scalar level tuples, level objects, and arrays."""
        if (
            type(binding) is not _DeviceHierarchyOwnerBinding
            or self.source_device_hierarchy is not binding.source
            or self.device_hierarchy is not binding.scalar
            or binding.scalar.source_hierarchy is not binding.source
        ):
            raise RuntimeError("persistent scalar-fused hierarchy owner object changed")
        source_levels = binding.source.levels
        scalar_levels = binding.scalar.levels
        if (
            type(source_levels) is not tuple
            or type(scalar_levels) is not tuple
            or source_levels is not binding.source_levels
            or scalar_levels is not binding.scalar_levels
            or source_levels is not scalar_levels
            or len(source_levels) != len(binding.levels)
        ):
            raise RuntimeError("persistent source/scalar hierarchy levels tuple container changed")
        for level_index, (level, expected) in enumerate(zip(source_levels, binding.levels, strict=True)):
            if level is not expected.level:
                raise RuntimeError(f"persistent hierarchy level {level_index} owner object changed")
            for field_name in (
                "row_offsets",
                "column_indices",
                "matrix_values",
                "inverse_diagonal",
                "aggregate",
                "prolongation_blocks",
                "member_offsets",
                "member_fine_nodes",
            ):
                if getattr(level, field_name) is not getattr(expected, field_name):
                    raise RuntimeError(f"persistent hierarchy level {level_index} {field_name} owner changed")
        if (
            binding.source.coarse_cholesky is not binding.coarse_cholesky
            or binding.scalar.coarse_cholesky is not binding.coarse_cholesky
        ):
            raise RuntimeError("persistent hierarchy coarse Cholesky owner changed")

    def _validate_direct_correction_owner_binding(self, binding: _DirectCorrectionOwnerBinding) -> None:
        """Require all direct arrays and fixed outer containers from construction."""
        if type(binding) is not _DirectCorrectionOwnerBinding:
            raise RuntimeError("persistent direct correction owner binding changed")
        for field_name in _DirectCorrectionOwnerBinding._fields:
            if field_name in ("outer_start_positions", "outer_candidate_positions"):
                continue
            if getattr(self, field_name) is not getattr(binding, field_name):
                raise RuntimeError(f"persistent direct correction {field_name} owner changed")
        for field_name in ("outer_start_positions", "outer_candidate_positions"):
            actual = getattr(self, field_name)
            expected = getattr(binding, field_name)
            if (
                type(actual) is not tuple
                or actual is not expected
                or len(actual) != OUTER_CORRECTIONS
                or any(current is not bound for current, bound in zip(actual, expected, strict=True))
            ):
                raise RuntimeError(f"persistent direct correction {field_name} tuple container or order changed")

    def _validate_construction_claims_owner_binding(self, binding: _WorkspaceOwnerBinding) -> None:
        """Require construction objects and cached labels to match private claims."""
        claims = binding.claims
        if type(claims) is not _ConstructionClaimsOwnerBinding:
            raise RuntimeError("persistent construction claims owner binding changed")
        try:
            _require_finalize_gate_evidence(
                claims.finalize_gate_route,
                claims.finalize_gate_block_dim,
                claims.finalize_gate_owner_threads,
                claims.finalize_gate_owner_roles,
                claims.finalize_gate_collective_version,
            )
        except ValueError as exc:
            raise RuntimeError("finalize gate construction claims changed") from exc
        if (
            self.problem is not binding.problem
            or self._construction_operator is not binding.construction_operator
            or self._construction_k1 is not binding.construction_k1
            or self._construction_k4 is not binding.construction_k4
            or self.hierarchy is not binding.hierarchy
        ):
            raise RuntimeError("persistent construction problem, operator, endpoint, or hierarchy owner changed")
        if (
            not isinstance(claims.array_descriptor_type, type)
            or type(claims.array_descriptor_fields) is not tuple
            or tuple(getattr(claims.array_descriptor_type, "_fields_", ())) != claims.array_descriptor_fields
            or tuple(field[0] for field in claims.array_descriptor_fields) != _ARRAY_DESCRIPTOR_FIELD_NAMES
            or self._array_descriptor_type_bound is not claims.array_descriptor_type
            or self._array_descriptor_fields_bound is not claims.array_descriptor_fields
        ):
            raise RuntimeError("persistent array descriptor construction claim changed")
        if (
            type(claims.fused_gather_kernel_version) is not str
            or claims.fused_gather_kernel_version != FUSED_GATHER_KERNEL_VERSION
            or self._fused_gather_kernel_version_bound != claims.fused_gather_kernel_version
            or type(claims.scalar_direction_apply_kernel_version) is not str
            or claims.scalar_direction_apply_kernel_version != SCALAR_DIRECTION_APPLY_KERNEL_VERSION
            or self._scalar_direction_apply_kernel_version_bound != claims.scalar_direction_apply_kernel_version
            or type(claims.scalar_fused_evidence) is not _ScalarFusedHierarchyEvidence
            or claims.scalar_fused_evidence != _canonical_scalar_fused_evidence(binding.hierarchy)
            or claims.first_cycle_publication_role != FIRST_CYCLE_PUBLICATION_ROLE
            or self._first_cycle_publication_role_bound != claims.first_cycle_publication_role
            or claims.second_cycle_publication_role != SECOND_CYCLE_PUBLICATION_ROLE
            or self._second_cycle_publication_role_bound != claims.second_cycle_publication_role
            or type(claims.outer_kernel_version) is not str
            or claims.outer_kernel_version != OUTER_KERNEL_VERSION
            or type(claims.outer_schedule_version) is not str
            or claims.outer_schedule_version != OUTER_SCHEDULE_VERSION
            or claims.outer_schedule_sha256 != OUTER_SCHEDULE_SHA256
            or claims.finalize_gate_route != FINALIZE_GATE_ROUTE
            or claims.finalize_gate_block_dim != FINALIZE_GATE_BLOCK_DIM
            or claims.finalize_gate_owner_threads != FINALIZE_GATE_OWNER_THREADS
            or claims.finalize_gate_owner_roles != FINALIZE_GATE_OWNER_ROLES
            or claims.finalize_gate_collective_version != FINALIZE_GATE_COLLECTIVE_VERSION
            or claims.outer_schedule_sha256
            != _derive_outer_schedule_sha256(
                claims.outer_kernel_version,
                claims.fused_gather_kernel_version,
                claims.scalar_direction_apply_kernel_version,
                claims.scalar_fused_evidence.publication_version,
                claims.scalar_fused_evidence.standalone_publication_route,
                claims.scalar_fused_evidence.external_shared_publication_route,
                claims.first_cycle_publication_role,
                claims.second_cycle_publication_role,
                claims.finalize_gate_route,
                claims.finalize_gate_block_dim,
                claims.finalize_gate_owner_threads,
                claims.finalize_gate_owner_roles,
                claims.finalize_gate_collective_version,
                claims.outer_schedule_version,
            )
            or self._outer_kernel_version_bound != claims.outer_kernel_version
            or self._outer_schedule_version_bound != claims.outer_schedule_version
            or self._outer_schedule_sha256_bound != claims.outer_schedule_sha256
            or self._finalize_gate_route_bound != claims.finalize_gate_route
            or self._finalize_gate_block_dim_bound != claims.finalize_gate_block_dim
            or self._finalize_gate_owner_threads_bound is not claims.finalize_gate_owner_threads
            or self._finalize_gate_owner_roles_bound is not claims.finalize_gate_owner_roles
            or self._finalize_gate_collective_version_bound != claims.finalize_gate_collective_version
        ):
            raise RuntimeError("outer kernel or schedule construction claim changed")
        _require_sha256(claims.outer_schedule_sha256, name="construction claims outer_schedule_sha256")
        if (
            type(claims.hierarchy_owner_identity) is not tuple
            or len(claims.hierarchy_owner_identity) != 2
            or type(claims.solver_graph_owner_identity) is not tuple
            or any(
                type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or type(item[1]) is not int
                for item in claims.solver_graph_owner_identity
            )
            or type(claims.content_identity) is not tuple
        ):
            raise RuntimeError("persistent construction identity claim schema changed")
        for name in (
            "solver_lane_contract_sha256",
            "solver_scalar_sha256",
            "solver_static_array_sha256",
        ):
            _require_sha256(getattr(claims, name), name=f"construction claims {name}")
        if claims.persistent_device_sha256 is not None:
            _require_sha256(claims.persistent_device_sha256, name="construction claims persistent_device_sha256")
        if claims.uncaptured_graph_identity_sha256 is not None:
            _require_sha256(
                claims.uncaptured_graph_identity_sha256,
                name="construction claims uncaptured_graph_identity_sha256",
            )
        if (
            self._hierarchy_owner_identity_bound is not claims.hierarchy_owner_identity
            or self._solver_graph_owner_identity_bound is not claims.solver_graph_owner_identity
            or self._construction_content_identity is not claims.content_identity
            or self._solver_lane_contract_sha256_bound != claims.solver_lane_contract_sha256
            or self._solver_scalar_sha256_bound != claims.solver_scalar_sha256
            or self._solver_static_array_sha256_bound != claims.solver_static_array_sha256
        ):
            raise RuntimeError("a mutable cached construction identity claim changed")
        if claims.persistent_device_sha256 is not None and (
            not hasattr(self, "_persistent_device_sha256")
            or not hasattr(self, "_construction_persistent_device_sha256")
            or self._persistent_device_sha256 != claims.persistent_device_sha256
            or self._construction_persistent_device_sha256 != claims.persistent_device_sha256
        ):
            raise RuntimeError("a mutable cached persistent device identity claim changed")
        if claims.uncaptured_graph_identity_sha256 is not None and (
            not hasattr(self, "_uncaptured_graph_identity_sha256")
            or self._uncaptured_graph_identity_sha256 != claims.uncaptured_graph_identity_sha256
        ):
            raise RuntimeError("the mutable cached uncaptured graph identity label claim changed")

    def _validate_replay_stream_owner_binding(self, binding: _WorkspaceOwnerBinding) -> None:
        """Require the exact dedicated replay stream and native handle values."""
        replay = binding.replay_stream
        if (
            type(replay) is not _ReplayStreamOwnerBinding
            or type(replay.stream) is not wp.Stream
            or self._replay_stream is not replay.stream
            or self.device is not binding.warp_device
            or replay.device is not binding.warp_device
            or getattr(binding.warp_device, "_context", None) != replay.device_context
            or type(getattr(binding.warp_device, "_context", None)) is not int
        ):
            raise RuntimeError("dedicated captured replay device or context owner changed")
        try:
            attributes = vars(replay.stream)
        except TypeError as error:
            raise RuntimeError("dedicated captured replay stream attribute owner changed") from error
        if (
            type(attributes) is not dict
            or attributes is not replay.attributes
            or tuple(attributes) != replay.attribute_names
            or attributes["device"] is not replay.device
            or attributes["cuda_stream"] != replay.cuda_stream
            or type(attributes["cuda_stream"]) is not int
            or attributes["owner"] is not replay.owner
            or type(attributes["owner"]) is not bool
            or attributes["_cached_event"] is not replay.cached_event
        ):
            raise RuntimeError("dedicated captured replay stream owner or native handle changed")

    @staticmethod
    def _capture_native_graph_owner_binding(graph: object, device: object) -> _NativeGraphOwnerBinding:
        """Capture every mutable Warp Graph attribute after eager exec creation."""
        try:
            attributes = vars(graph)
        except TypeError as error:
            raise RuntimeError("captured Warp graph has no exact mutable attribute dictionary") from error
        expected_names = ("device", "capture_id", "module_execs", "graph_exec", "graph")
        if type(attributes) is not dict or tuple(attributes) != expected_names:
            raise RuntimeError("captured Warp graph attribute inventory changed")
        capture_id = attributes["capture_id"]
        module_execs = attributes["module_execs"]
        graph_exec = attributes["graph_exec"]
        graph_handle = attributes["graph"]
        if (
            attributes["device"] is not device
            or type(capture_id) is not int
            or type(module_execs) is not set
            or type(graph_exec) is not ctypes.c_void_p
            or type(graph_exec.value) is not int
            or graph_exec.value <= 0
            or type(graph_handle) is not ctypes.c_void_p
            or type(graph_handle.value) is not int
            or graph_handle.value <= 0
        ):
            raise RuntimeError("captured Warp graph native launch state is incomplete")
        return _NativeGraphOwnerBinding(
            graph=graph,
            attributes=attributes,
            attribute_names=expected_names,
            device=device,
            capture_id=capture_id,
            module_execs=module_execs,
            module_exec_members=tuple(sorted(module_execs, key=id)),
            graph_exec=graph_exec,
            graph_exec_value=int(graph_exec.value),
            graph_handle=graph_handle,
            graph_handle_value=int(graph_handle.value),
        )

    def _validate_native_graph_owner_binding(
        self,
        graph: object,
        native: _NativeGraphOwnerBinding,
        binding: _WorkspaceOwnerBinding,
        *,
        name: str,
    ) -> None:
        """Require one exact Warp Graph wrapper and every native launch field."""
        if type(native) is not _NativeGraphOwnerBinding or graph is not native.graph:
            raise RuntimeError(f"{name} native graph owner binding changed")
        try:
            attributes = vars(graph)
        except TypeError as error:
            raise RuntimeError(f"{name} graph attribute dictionary changed") from error
        if (
            type(attributes) is not dict
            or attributes is not native.attributes
            or tuple(attributes) != native.attribute_names
            or attributes["device"] is not native.device
            or native.device is not binding.warp_device
            or attributes["capture_id"] != native.capture_id
            or type(attributes["capture_id"]) is not int
        ):
            raise RuntimeError(f"{name} graph device, capture ID, or attribute owner changed")
        if attributes["module_execs"] is not native.module_execs or type(attributes["module_execs"]) is not set:
            raise RuntimeError(f"{name} graph retained module-exec set changed")
        module_exec_members = tuple(sorted(attributes["module_execs"], key=id))
        if len(module_exec_members) != len(native.module_exec_members) or any(
            current is not expected
            for current, expected in zip(module_exec_members, native.module_exec_members, strict=True)
        ):
            raise RuntimeError(f"{name} graph retained module-exec set changed")
        graph_exec = attributes["graph_exec"]
        if (
            graph_exec is not native.graph_exec
            or type(graph_exec) is not ctypes.c_void_p
            or type(graph_exec.value) is not int
            or int(graph_exec.value) != native.graph_exec_value
        ):
            raise RuntimeError(f"{name} native graph_exec owner or handle value changed")
        graph_handle = attributes["graph"]
        if (
            graph_handle is not native.graph_handle
            or type(graph_handle) is not ctypes.c_void_p
            or type(graph_handle.value) is not int
            or int(graph_handle.value) != native.graph_handle_value
        ):
            raise RuntimeError(f"{name} native graph owner or handle value changed")

    def _validate_capture_graph_owner_binding(self, binding: _WorkspaceOwnerBinding) -> None:
        """Require the active graph pair and identity labels from one generation."""
        capture = binding.capture
        claims = binding.claims
        if capture is None:
            if (
                self.graph is not None
                or self.k4_graph is not None
                or self._capture_generation != 0
                or self._captured_graph_object_identity is not None
            ):
                raise RuntimeError("uncaptured graph facade claims a captured graph generation")
            if claims.persistent_device_sha256 is None or claims.uncaptured_graph_identity_sha256 is None:
                return
            expected_graph_identity = self._derive_graph_identity(
                captured=True,
                comparator=False,
                owner_binding=binding,
                graph=None,
                generation=0,
            )
            expected_k4_identity = self._derive_graph_identity(
                captured=True,
                comparator=True,
                owner_binding=binding,
                graph=None,
                generation=0,
            )
            if (
                self.graph_identity_sha256 != expected_graph_identity
                or self.k4_graph_identity_sha256 != expected_k4_identity
            ):
                raise RuntimeError("uncaptured graph identity label changed")
            return
        if (
            type(capture) is not _CaptureGraphOwnerBinding
            or not isinstance(capture.graph_type, type)
            or capture.graph is None
            or capture.k4_graph is None
            or capture.graph is capture.k4_graph
            or type(capture.graph) is not capture.graph_type
            or type(capture.k4_graph) is not capture.graph_type
            or capture.replay_stream is not binding.replay_stream
            or type(capture.generation) is not int
            or capture.generation < 1
            or type(capture.object_identity) is not tuple
            or capture.object_identity != (id(capture.graph), id(capture.k4_graph))
        ):
            raise RuntimeError("captured graph owner binding or comparator relation changed")
        self._validate_native_graph_owner_binding(
            capture.graph,
            capture.graph_native,
            binding,
            name="captured integrated graph",
        )
        self._validate_native_graph_owner_binding(
            capture.k4_graph,
            capture.k4_graph_native,
            binding,
            name="captured K4 graph",
        )
        if (
            capture.graph_native.graph_exec_value == capture.k4_graph_native.graph_exec_value
            or capture.graph_native.graph_handle_value == capture.k4_graph_native.graph_handle_value
        ):
            raise RuntimeError("captured integrated and K4 native graph handles alias")
        _require_sha256(capture.graph_identity_sha256, name="captured graph identity")
        _require_sha256(capture.k4_graph_identity_sha256, name="captured K4 graph identity")
        expected_graph_identity = self._derive_graph_identity(
            captured=True,
            comparator=False,
            owner_binding=binding,
            graph=capture.graph,
            generation=capture.generation,
        )
        expected_k4_identity = self._derive_graph_identity(
            captured=True,
            comparator=True,
            owner_binding=binding,
            graph=capture.k4_graph,
            generation=capture.generation,
        )
        if (
            capture.graph_identity_sha256 != expected_graph_identity
            or capture.k4_graph_identity_sha256 != expected_k4_identity
            or capture.graph_identity_sha256 == capture.k4_graph_identity_sha256
        ):
            raise RuntimeError("captured graph identity claims do not bind the exact graph pair")
        if (
            self.graph is not capture.graph
            or self.k4_graph is not capture.k4_graph
            or self._capture_generation != capture.generation
            or self._captured_graph_object_identity is not capture.object_identity
            or self.graph_identity_sha256 != capture.graph_identity_sha256
            or self.k4_graph_identity_sha256 != capture.k4_graph_identity_sha256
        ):
            raise RuntimeError("captured graph object facade or generation changed; an identity label may be stale")

    def _validate_persistent_array_owner_binding(self, binding: _WorkspaceOwnerBinding) -> None:
        """Require every live graph array to be its construction-owned allocation."""
        persistent = binding.persistent
        if (
            type(persistent) is not _PersistentArrayOwnerBinding
            or type(persistent.arrays) is not tuple
            or type(persistent.signatures) is not tuple
            or len(persistent.arrays) != len(persistent.signatures)
        ):
            raise RuntimeError("persistent captured array owner binding changed")
        live_arrays = self._persistent_input_arrays(binding)
        if type(live_arrays) is not tuple or len(live_arrays) != len(persistent.arrays):
            raise RuntimeError("persistent captured array owner inventory changed")
        for live_item, bound_item, signature_item in zip(
            live_arrays,
            persistent.arrays,
            persistent.signatures,
            strict=True,
        ):
            if (
                type(live_item) is not tuple
                or type(bound_item) is not tuple
                or type(signature_item) is not tuple
                or len(live_item) != 2
                or len(bound_item) != 2
                or len(signature_item) != 2
                or type(bound_item[0]) is not str
                or live_item[0] != bound_item[0]
                or signature_item[0] != bound_item[0]
            ):
                raise RuntimeError("persistent captured array owner name or order changed")
            if live_item[1] is not bound_item[1]:
                raise RuntimeError(f"persistent array owner {bound_item[0]} changed allocation or pointer")
        live_signatures = self._persistent_array_signatures_from_items(
            live_arrays,
            descriptor_type=binding.claims.array_descriptor_type,
            descriptor_fields=binding.claims.array_descriptor_fields,
        )
        if live_signatures != persistent.signatures:
            raise RuntimeError("a persistent captured input array descriptor or pointer changed")
        if (
            not hasattr(self, "_persistent_array_identity")
            or self._persistent_array_identity is not persistent.signatures
        ):
            raise RuntimeError("the mutable persistent array identity cache changed")

    def _validate_workspace_owner_bindings(self, binding: _WorkspaceOwnerBinding) -> None:
        """Fail closed on every resource-owner or container identity change."""
        if type(binding) is not _WorkspaceOwnerBinding:
            raise RuntimeError("captured correction construction owners are not registered")
        if self.scene is not binding.scene or self.config is not binding.config:
            raise RuntimeError("persistent captured scene or configuration identity owner object changed")
        self._validate_construction_claims_owner_binding(binding)
        self._validate_replay_stream_owner_binding(binding)
        self._validate_capture_graph_owner_binding(binding)
        self._validate_public_vbd_owner_binding(binding.public_vbd)
        self._validate_device_hierarchy_owner_binding(binding.device)
        self._validate_direct_correction_owner_binding(binding.direct)
        if type(self.operator) is not WarpMatrixFreeStableNHOperator or self.operator is not binding.operator:
            raise RuntimeError("persistent matrix-free operator owner object changed")
        if (
            type(self.workspaces) is not tuple
            or self.workspaces is not binding.workspaces
            or len(self.workspaces) != OUTER_CORRECTIONS
            or type(binding.outer) is not tuple
            or len(binding.outer) != OUTER_CORRECTIONS
        ):
            raise RuntimeError("persistent captured correction workspace tuple container changed")
        for outer_index, (workspace, outer_binding) in enumerate(zip(self.workspaces, binding.outer, strict=True)):
            name = f"outer {outer_index}"
            if type(workspace) is not _OuterWorkspace or workspace is not outer_binding.workspace:
                raise RuntimeError(f"persistent {name} workspace owner object changed")
            direct_arrays = (
                ("rhs", outer_binding.rhs),
                ("first_correction", outer_binding.first_correction),
                ("operator_product_after_first", outer_binding.operator_product_after_first),
                ("residual_after_first", outer_binding.residual_after_first),
                ("second_correction", outer_binding.second_correction),
                ("direction", outer_binding.direction),
            )
            for field_name, expected in direct_arrays:
                if getattr(workspace, field_name) is not expected:
                    raise RuntimeError(f"persistent {name} workspace {field_name} owner changed")
            self._validate_cycle_workspace_owner_binding(
                workspace.first_cycle,
                outer_binding.first_cycle,
                name=f"persistent {name} first cycle",
            )
            self._validate_cycle_workspace_owner_binding(
                workspace.second_cycle,
                outer_binding.second_cycle,
                name=f"persistent {name} second cycle",
            )
            self._validate_external_publication_aliases(outer_binding, name=f"persistent {name}")
            operator_apply = workspace.operator_apply
            if (
                type(operator_apply) is not WarpMatrixFreeWorkspace
                or operator_apply is not outer_binding.operator_apply
                or operator_apply._operator_identity != id(self.operator)
                or operator_apply.delta_piola is not outer_binding.operator_apply_delta_piola
            ):
                raise RuntimeError(f"persistent {name} matrix-free apply workspace owner changed")
        self._validate_persistent_array_owner_binding(binding)

    def _workspace_owner_identity_sha256(self, binding: _WorkspaceOwnerBinding | None = None) -> str:
        """Hash the registered Python owner/container graph for provenance."""
        if binding is None:
            binding = _lookup_workspace_owners(self)
        self._validate_workspace_owner_bindings(binding)
        persistent = binding.persistent
        if type(persistent) is not _PersistentArrayOwnerBinding:
            raise RuntimeError("persistent captured array owner binding changed")

        def cycle_record(cycle: _ScalarCycleWorkspaceOwnerBinding) -> dict[str, object]:
            return {
                "workspace_object": id(cycle.workspace),
                "hierarchy_object": id(cycle.hierarchy),
                "rhs_object": id(cycle.rhs),
                "correction_object": id(cycle.correction),
                "coarse_intermediate_object": id(cycle.coarse_intermediate),
                "final_scalar_correction_object": id(cycle.final_scalar_correction),
                "final_scalar_correction_pointer": cycle.final_scalar_pointer,
                "level_rhs_container": id(cycle.level_rhs),
                "level_correction_container": id(cycle.level_correction),
                "level_correction_alt_container": id(cycle.level_correction_alt),
                "level_residual_container": id(cycle.level_residual),
                "persistent_arrays_container": id(cycle.persistent_arrays),
                "persistent_pointers_container": id(cycle.persistent_pointers),
                "level_rhs_objects": [id(value) for value in cycle.level_rhs],
                "level_correction_objects": [id(value) for value in cycle.level_correction],
                "level_correction_alt_objects": [id(value) for value in cycle.level_correction_alt],
                "level_residual_objects": [id(value) for value in cycle.level_residual],
            }

        return _canonical_digest(
            {
                "contract": "captured-direct-graph-vbd-workspace-owner-identity-v6",
                "scene_object": id(binding.scene),
                "config_object": id(binding.config),
                "warp_device_object": id(binding.warp_device),
                "warp_device_context": binding.replay_stream.device_context,
                "replay_stream_object": id(binding.replay_stream.stream),
                "replay_stream_attributes": id(binding.replay_stream.attributes),
                "replay_stream_native_handle": binding.replay_stream.cuda_stream,
                "problem_object": id(binding.problem),
                "construction_operator_object": id(binding.construction_operator),
                "construction_k1_object": id(binding.construction_k1),
                "construction_k4_object": id(binding.construction_k4),
                "hierarchy_object": id(binding.hierarchy),
                "baseline_object": id(binding.public_vbd.baseline),
                "lanes_container": id(binding.public_vbd.lanes),
                "lane_objects": [id(binding.public_vbd.k1_lane), id(binding.public_vbd.k4_lane)],
                "particle_color_groups_container": id(binding.public_vbd.particle_color_groups),
                "particle_color_group_objects": [id(value) for value in binding.public_vbd.particle_color_group_arrays],
                "source_hierarchy_object": id(binding.device.source),
                "scalar_hierarchy_object": id(binding.device.scalar),
                "source_levels_container": id(binding.device.source_levels),
                "scalar_levels_container": id(binding.device.scalar_levels),
                "hierarchy_level_objects": [id(value.level) for value in binding.device.levels],
                "operator_object": id(binding.operator),
                "direct_arrays": {
                    name: id(getattr(binding.direct, name))
                    for name in _DirectCorrectionOwnerBinding._fields
                    if name not in ("outer_start_positions", "outer_candidate_positions")
                },
                "outer_start_positions_container": id(binding.direct.outer_start_positions),
                "outer_start_position_objects": [id(value) for value in binding.direct.outer_start_positions],
                "outer_candidate_positions_container": id(binding.direct.outer_candidate_positions),
                "outer_candidate_position_objects": [id(value) for value in binding.direct.outer_candidate_positions],
                "workspaces_container": id(binding.workspaces),
                "persistent_owner_object": id(persistent),
                "persistent_arrays_container": id(persistent.arrays),
                "persistent_signatures_container": id(persistent.signatures),
                "persistent_array_owners": [[name, id(value)] for name, value in persistent.arrays],
                "array_descriptor_type_object": id(binding.claims.array_descriptor_type),
                "array_descriptor_fields": [
                    [name, id(field_type)] for name, field_type in binding.claims.array_descriptor_fields
                ],
                "fused_gather_kernel_version": binding.claims.fused_gather_kernel_version,
                "scalar_direction_apply_kernel_version": (binding.claims.scalar_direction_apply_kernel_version),
                "first_cycle_publication_role": binding.claims.first_cycle_publication_role,
                "second_cycle_publication_role": binding.claims.second_cycle_publication_role,
                "scalar_fused_evidence": {
                    field.name: getattr(binding.claims.scalar_fused_evidence, field.name)
                    for field in dataclasses.fields(binding.claims.scalar_fused_evidence)
                },
                "outer_kernel_version": binding.claims.outer_kernel_version,
                "outer_schedule_version": binding.claims.outer_schedule_version,
                "outer_schedule_sha256": binding.claims.outer_schedule_sha256,
                "finalize_gate_route": binding.claims.finalize_gate_route,
                "finalize_gate_block_dim": binding.claims.finalize_gate_block_dim,
                "finalize_gate_owner_threads": list(binding.claims.finalize_gate_owner_threads),
                "finalize_gate_owner_roles": list(binding.claims.finalize_gate_owner_roles),
                "finalize_gate_collective_version": binding.claims.finalize_gate_collective_version,
                "solver_lane_contract_sha256": binding.claims.solver_lane_contract_sha256,
                "hierarchy_owner_identity": list(binding.claims.hierarchy_owner_identity),
                "solver_graph_owner_identity": [list(item) for item in binding.claims.solver_graph_owner_identity],
                "solver_scalar_sha256": binding.claims.solver_scalar_sha256,
                "solver_static_array_sha256": binding.claims.solver_static_array_sha256,
                "construction_content_identity": list(binding.claims.content_identity),
                "outer": [
                    {
                        "workspace_object": id(outer.workspace),
                        "rhs_object": id(outer.rhs),
                        "first_correction_object": id(outer.first_correction),
                        "operator_product_after_first_object": id(outer.operator_product_after_first),
                        "residual_after_first_object": id(outer.residual_after_first),
                        "second_correction_object": id(outer.second_correction),
                        "direction_object": id(outer.direction),
                        "first_cycle": cycle_record(outer.first_cycle),
                        "second_cycle": cycle_record(outer.second_cycle),
                        "operator_apply_object": id(outer.operator_apply),
                        "operator_apply_delta_piola_object": id(outer.operator_apply_delta_piola),
                    }
                    for outer in binding.outer
                ],
            }
        )

    def _derive_graph_identity(
        self,
        *,
        captured: bool,
        comparator: bool,
        owner_binding: _WorkspaceOwnerBinding,
        graph: object | None,
        generation: int,
    ) -> str:
        """Derive one graph identity from explicit construction and capture owners."""
        claims = owner_binding.claims
        if claims.persistent_device_sha256 is None or len(claims.content_identity) < 4:
            raise RuntimeError("graph identity requires finalized construction claims")
        scene_sha256, objective_sha256, config_sha256, hierarchy_sha256 = claims.content_identity[:4]
        scalar_evidence = _canonical_scalar_fused_evidence(owner_binding.hierarchy)
        v_cycle_kernel_launches = scalar_evidence.scheduled_kernel_launches
        v_cycle_core_kernel_launches = scalar_evidence.core_kernel_launches
        linear_prefix_kernel_launches = 4 + 2 * v_cycle_core_kernel_launches
        retained_linear_work_launches = linear_prefix_kernel_launches + 1
        outer_kernel_launches = retained_linear_work_launches + 3
        correction_kernel_launches = 2 + OUTER_CORRECTIONS * outer_kernel_launches
        return _canonical_digest(
            {
                "contract": "captured-direct-graph-vbd-graph-identity-v7",
                "solver_contract": CONTRACT_ID,
                "comparator_contract": VBD_BASELINE_CONTRACT_ID if comparator else None,
                "scene_sha256": scene_sha256,
                "objective_instance_sha256": objective_sha256,
                "config_sha256": config_sha256,
                "static_hierarchy_sha256": hierarchy_sha256,
                "persistent_device_sha256": claims.persistent_device_sha256,
                "captured": captured,
                "capture_generation": generation if captured else 0,
                "graph_object_identity": id(graph) if captured and graph is not None else None,
                "lane": "public-k4" if comparator else "public-k1-plus-direct-four-by-two",
                "fused_gather_kernel_version": claims.fused_gather_kernel_version,
                "scalar_direction_apply_kernel_version": claims.scalar_direction_apply_kernel_version,
                "first_cycle_publication_role": claims.first_cycle_publication_role,
                "second_cycle_publication_role": claims.second_cycle_publication_role,
                "outer_kernel_version": claims.outer_kernel_version,
                "outer_schedule_version": claims.outer_schedule_version,
                "outer_schedule_sha256": claims.outer_schedule_sha256,
                "finalize_gate_route": claims.finalize_gate_route,
                "finalize_gate_block_dim": claims.finalize_gate_block_dim,
                "finalize_gate_owner_threads": list(claims.finalize_gate_owner_threads),
                "finalize_gate_owner_roles": list(claims.finalize_gate_owner_roles),
                "finalize_gate_collective_version": claims.finalize_gate_collective_version,
                "v_cycle_kernel_version": V_CYCLE_KERNEL_VERSION,
                "v_cycle_schedule_version": V_CYCLE_SCHEDULE_VERSION,
                "v_cycle_schedule_sha256": scalar_evidence.schedule_sha256,
                "v_cycle_core_schedule_sha256": scalar_evidence.core_schedule_sha256,
                "v_cycle_publication_version": scalar_evidence.publication_version,
                "v_cycle_standalone_publication_route": scalar_evidence.standalone_publication_route,
                "v_cycle_external_shared_publication_route": scalar_evidence.external_shared_publication_route,
                "v_cycle_root_ingress_zero_start_fusions": (scalar_evidence.root_ingress_zero_start_fusions),
                "v_cycle_kernel_launches": v_cycle_kernel_launches,
                "v_cycle_core_kernel_launches": v_cycle_core_kernel_launches,
                "first_cycle_kernel_launches": v_cycle_core_kernel_launches,
                "second_cycle_kernel_launches": v_cycle_core_kernel_launches,
                "first_cycle_publication_kernel_launches": 0,
                "second_cycle_publication_kernel_launches": 0,
                "linear_prefix_kernel_launches_per_outer": 0 if comparator else linear_prefix_kernel_launches,
                "fused_gather_kernel_launches_per_outer": 0 if comparator else 2,
                "fused_vertex_kernel_launches_per_outer": 0 if comparator else 1,
                "outer_kernel_launches_per_outer": 0 if comparator else outer_kernel_launches,
                "correction_kernel_launches": 0 if comparator else correction_kernel_launches,
            }
        )

    def _graph_identity(
        self,
        *,
        captured: bool,
        comparator: bool,
        owner_binding: _WorkspaceOwnerBinding | None = None,
    ) -> str:
        """Bind one fixed launch schedule to registered graph owners."""
        if owner_binding is None:
            owner_binding = _lookup_workspace_owners(self)
        capture = owner_binding.capture
        if captured and capture is not None:
            graph = capture.k4_graph if comparator else capture.graph
            generation = capture.generation
        else:
            graph = None
            generation = 0
        return self._derive_graph_identity(
            captured=captured,
            comparator=comparator,
            owner_binding=owner_binding,
            graph=graph,
            generation=generation,
        )

    def _bound_graph_launch_owners(
        self,
        owner_binding: _WorkspaceOwnerBinding,
        *,
        comparator: bool,
    ) -> tuple[object, object]:
        """Return exact validated graph and stream owners immediately before launch."""
        self._validate_replay_stream_owner_binding(owner_binding)
        self._validate_capture_graph_owner_binding(owner_binding)
        capture = owner_binding.capture
        if capture is None:
            raise RuntimeError("capture_graphs() must complete before graph replay")
        return (capture.k4_graph if comparator else capture.graph, capture.replay_stream.stream)

    def _consume_execution_receipt(
        self,
        receipt: object,
        owner_binding: _WorkspaceOwnerBinding,
    ) -> _ExecutionReceipt:
        """Consume exactly one registered receipt; copied or stale receipts fail closed."""
        if type(receipt) is not _ExecutionReceipt:
            raise RuntimeError("recording requires an exact solver-issued execution receipt")
        registered = self._issued_execution_receipts.pop(id(receipt), None)
        if registered is not receipt or receipt.issuer is not self:
            raise RuntimeError("execution receipt is forged, stale, or already consumed")
        if receipt.serial != self._execution_serial:
            raise RuntimeError("execution receipt is not the latest monotonic solver launch")
        expected_capture = owner_binding.capture if receipt.graph_replay else None
        if receipt.capture_binding is not expected_capture:
            raise RuntimeError("execution receipt capture binding does not match the actual launch")
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
            capture_binding,
            outer_slots,
            outer_slot_records,
            owner_binding,
        ) = receipt
        self._validate_capture_graph_owner_binding(owner_binding)
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
            or context.fused_gather_kernel_version != owner_binding.claims.fused_gather_kernel_version
            or context.scalar_direction_apply_kernel_version
            != owner_binding.claims.scalar_direction_apply_kernel_version
            or context.first_cycle_publication_role != owner_binding.claims.first_cycle_publication_role
            or context.second_cycle_publication_role != owner_binding.claims.second_cycle_publication_role
            or context.outer_kernel_version != owner_binding.claims.outer_kernel_version
            or context.outer_schedule_version != owner_binding.claims.outer_schedule_version
            or context.outer_schedule_sha256 != owner_binding.claims.outer_schedule_sha256
            or context.finalize_gate_route != owner_binding.claims.finalize_gate_route
            or context.finalize_gate_block_dim != owner_binding.claims.finalize_gate_block_dim
            or context.finalize_gate_owner_threads != owner_binding.claims.finalize_gate_owner_threads
            or context.finalize_gate_owner_roles != owner_binding.claims.finalize_gate_owner_roles
            or context.finalize_gate_collective_version != owner_binding.claims.finalize_gate_collective_version
            or context.capture_binding is not capture_binding
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
        expected_capture = owner_binding.capture if graph_replay else None
        if capture_binding is not expected_capture:
            raise ValueError("validation context capture generation is stale")
        expected_graph_identity = self._graph_identity(
            captured=graph_replay,
            comparator=False,
            owner_binding=owner_binding,
        )
        if context.graph_identity_sha256 != expected_graph_identity:
            raise ValueError("validation context graph identity is not canonically derived")
        scalar_evidence = _validate_scalar_fused_hierarchy_inputs(
            owner_binding.device.scalar,
            owner_binding.device.source,
            owner_binding.hierarchy,
            validate_device_content=False,
        )
        if context.v_cycle_schedule_sha256 != scalar_evidence.schedule_sha256:
            raise ValueError("validation context scalar-fused schedule identity is not canonical")
        if context.v_cycle_core_schedule_sha256 != scalar_evidence.core_schedule_sha256:
            raise ValueError("validation context scalar-fused core schedule identity is not canonical")
        if context.v_cycle_static_device_content_sha256 != scalar_evidence.static_device_content_sha256:
            raise ValueError("validation context scalar-fused static content identity is not canonical")
        if context.v_cycle_device_snapshot_sha256 != scalar_evidence.device_snapshot_sha256:
            raise ValueError("validation context scalar-fused device snapshot identity is not canonical")
        if context.v_cycle_core_device_snapshot_sha256 != scalar_evidence.core_device_snapshot_sha256:
            raise ValueError("validation context scalar-fused core device snapshot identity is not canonical")
        if context.v_cycle_kernel_launches != scalar_evidence.scheduled_kernel_launches:
            raise ValueError("validation context V-cycle launch count is not canonical")
        if context.v_cycle_core_kernel_launches != scalar_evidence.core_kernel_launches:
            raise ValueError("validation context V-cycle core launch count is not canonical")
        if (
            context.v_cycle_publication_version != scalar_evidence.publication_version
            or context.v_cycle_standalone_publication_route != scalar_evidence.standalone_publication_route
            or context.v_cycle_external_shared_publication_route != scalar_evidence.external_shared_publication_route
        ):
            raise ValueError("validation context V-cycle publication provenance is not canonical")
        if context.v_cycle_root_ingress_zero_start_fusions != scalar_evidence.root_ingress_zero_start_fusions:
            raise ValueError("validation context V-cycle root ingress fusion count is not canonical")
        if validate_raw_sources:
            if type(owner_binding) is not _WorkspaceOwnerBinding:
                raise ValueError("validation context lost its exact construction owner binding")
            current_persistent_sha256 = self._validate_persistent_sources(owner_binding=owner_binding)
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

    def _solver_graph_array_items(
        self,
        public: _PublicVBDOwnerBinding | None = None,
    ) -> tuple[tuple[str, wp.array[Any]], ...]:
        """Enumerate all lane-solver, adjacency, control, model, and state arrays."""
        if public is None:
            public = _lookup_workspace_owners(self).public_vbd
        arrays: list[tuple[str, wp.array[Any]]] = []
        arrays.extend(_attribute_array_items(public.model, prefix="baseline.model"))
        arrays.extend(_attribute_array_items(public.control, prefix="baseline.control"))
        for iterations, lane in ((1, public.k1_lane), (4, public.k4_lane)):
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

    def _persistent_input_arrays(
        self,
        binding: _WorkspaceOwnerBinding | None = None,
    ) -> tuple[tuple[str, wp.array[Any]], ...]:
        """Return every array whose allocation is embedded in either captured graph."""
        if binding is None:
            binding = _lookup_workspace_owners(self)
        public = binding.public_vbd
        direct = binding.direct
        model = public.model
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
                for index, value in enumerate(public.particle_color_group_arrays)
            ),
        )
        pristine_arrays = tuple(
            (f"pristine.{state_name}.{field_name}", getattr(state, field_name))
            for state_name, state in (
                ("input", public.pristine_input),
                ("output", public.pristine_output),
            )
            for field_name in ("particle_q", "particle_qd", "particle_f")
        )
        lane_arrays = tuple(
            (f"lane_{iterations}.{state_name}.{field_name}", getattr(state, field_name))
            for iterations, lane in ((1, public.k1_lane), (4, public.k4_lane))
            for state_name, state in (
                ("state_in", lane.state_in),
                ("state_out", lane.state_out),
            )
            for field_name in ("particle_q", "particle_qd", "particle_f")
        )
        solver_graph_arrays = self._solver_graph_array_items(public)
        operator_arrays = tuple(
            (f"operator.{name}", getattr(binding.operator, name))
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
        for level_index, level in enumerate(binding.device.levels):
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
        hierarchy_arrays.append(("hierarchy.coarse_cholesky", binding.device.coarse_cholesky))
        correction_arrays = (
            ("operator.positions", binding.operator.positions),
            ("operator.deformation_gradients", binding.operator.deformation_gradients),
            ("operator.cofactors", binding.operator.cofactors),
            ("operator.determinants", binding.operator.determinants),
            ("operator.first_piola", binding.operator.first_piola),
            ("candidate", direct.candidate),
            ("proposal_finite", direct.proposal_finite),
            ("final_positions", direct.final_positions),
            ("final_velocities", direct.final_velocities),
            ("active", direct.active),
            ("accepted", direct.accepted),
            ("reasons", direct.reasons),
            ("current_inertia", direct.current_inertia),
            ("candidate_inertia", direct.candidate_inertia),
            ("vertex_finite", direct.vertex_finite),
            ("current_elastic", direct.current_elastic),
            ("candidate_elastic", direct.candidate_elastic),
            ("candidate_determinants", direct.candidate_determinants),
            ("segment_minima", direct.segment_minima),
            ("tet_finite", direct.tet_finite),
            ("directional_terms", direct.directional_terms),
            ("initial_objectives", direct.initial_objectives),
            ("candidate_objectives", direct.candidate_objectives),
            ("directional_derivatives", direct.directional_derivatives),
            ("minimum_segment_determinants", direct.minimum_segment_determinants),
            *((f"outer_start_positions_{index}", value) for index, value in enumerate(direct.outer_start_positions)),
            *(
                (f"outer_candidate_positions_{index}", value)
                for index, value in enumerate(direct.outer_candidate_positions)
            ),
        )
        workspace_arrays: list[tuple[str, wp.array[Any]]] = []
        for outer_index, workspace in enumerate(binding.outer):
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
                (f"workspace_{outer_index}.operator_apply.delta_piola", workspace.operator_apply_delta_piola)
            )
            for cycle_name, cycle_binding in (
                ("first_cycle", workspace.first_cycle),
                ("second_cycle", workspace.second_cycle),
            ):
                workspace_arrays.extend(
                    (
                        (f"workspace_{outer_index}.{cycle_name}.rhs", cycle_binding.rhs),
                        (f"workspace_{outer_index}.{cycle_name}.correction", cycle_binding.correction),
                        (
                            f"workspace_{outer_index}.{cycle_name}.coarse_intermediate",
                            cycle_binding.coarse_intermediate,
                        ),
                    )
                )
                for sequence_name in ("level_rhs", "level_correction", "level_correction_alt", "level_residual"):
                    workspace_arrays.extend(
                        (f"workspace_{outer_index}.{cycle_name}.{sequence_name}_{level_index}", value)
                        for level_index, value in enumerate(getattr(cycle_binding, sequence_name))
                    )
        return tuple(
            (name, value)
            for name, value in (
                *model_arrays,
                *pristine_arrays,
                *lane_arrays,
                *solver_graph_arrays,
                *operator_arrays,
                ("canonical_positions", direct.canonical_positions),
                ("x_current", direct.x_current),
                *hierarchy_arrays,
                *correction_arrays,
                *workspace_arrays,
            )
            if value is not None
        )

    @staticmethod
    def _persistent_array_signatures_from_items(
        arrays: tuple[tuple[str, wp.array[Any]], ...],
        *,
        descriptor_type: object,
        descriptor_fields: tuple[object, ...],
    ) -> tuple[tuple[str, object], ...]:
        """Describe exact array objects, pointers, metadata, and cached C views."""
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
                        expected_type=descriptor_type,
                        expected_fields=descriptor_fields,
                    ),
                ),
            )
            for name, value in arrays
        )

    def _persistent_array_signatures(
        self,
        binding: _WorkspaceOwnerBinding | None = None,
    ) -> tuple[tuple[str, object], ...]:
        """Resolve and describe every live facade array for diagnostics."""
        if binding is None:
            binding = _lookup_workspace_owners(self)
        return self._persistent_array_signatures_from_items(
            self._persistent_input_arrays(binding),
            descriptor_type=binding.claims.array_descriptor_type,
            descriptor_fields=binding.claims.array_descriptor_fields,
        )

    def _validate_persistent_sources(
        self,
        *,
        require_bound: bool = True,
        owner_binding: _WorkspaceOwnerBinding | None = None,
    ) -> str:
        """Synchronously reject any stale model, reset, operator, or hierarchy input."""
        if owner_binding is None:
            owner_binding = _lookup_workspace_owners(self)
        self._validate_workspace_owner_bindings(owner_binding)
        workspace_owner_identity_sha256 = self._workspace_owner_identity_sha256(owner_binding)
        claims = owner_binding.claims
        current_content_identity = self._construction_content_identity_record()
        if current_content_identity != claims.content_identity:
            raise RuntimeError("captured construction content identity labels changed")
        if (id(owner_binding.device.source), id(owner_binding.device.scalar)) != claims.hierarchy_owner_identity:
            raise RuntimeError("persistent scalar-fused hierarchy owner object changed")
        lane_contract_sha256 = self._validate_solver_lane_contract()
        if lane_contract_sha256 != claims.solver_lane_contract_sha256:
            raise RuntimeError("public K1/K4 SolverVBD lane schedule changed")
        if self._solver_graph_owner_identity() != claims.solver_graph_owner_identity:
            raise RuntimeError("persistent SolverVBD lane/control/state owner object changed")
        solver_scalar_sha256 = self._solver_scalar_sha256()
        if solver_scalar_sha256 != claims.solver_scalar_sha256:
            raise RuntimeError("persistent SolverVBD lane scalar configuration changed")
        solver_static_array_sha256 = self._solver_static_array_sha256()
        if solver_static_array_sha256 != claims.solver_static_array_sha256:
            raise RuntimeError("persistent SolverVBD static input or adjacency content changed")
        scene = owner_binding.scene
        config = owner_binding.config
        public = owner_binding.public_vbd
        if type(scene) is not TetBenchmarkScene or type(config) is not DirectGraphVBDConfig:
            raise RuntimeError("captured direct graph canonical scene/config objects changed")
        config.validate()
        scene_sha256 = _require_sha256(scene.manifest()["scene_sha256"], name="scene_sha256")
        config_sha256 = _canonical_digest(config.deterministic_record())
        if scene_sha256 != self.scene_sha256 or config_sha256 != self.config_sha256:
            raise RuntimeError("captured direct graph scene or configuration identity changed")
        if public.baseline.scene_sha256 != self.scene_sha256:
            raise RuntimeError("captured direct graph public baseline belongs to another scene")
        model_sha256 = _public_model_sha256(public.model)
        if model_sha256 != public.baseline.model_sha256:
            raise RuntimeError("public static model changed after captured direct graph construction")
        pristine_sha256 = public.baseline._record_pristine_state_sha256()
        if pristine_sha256 != public.baseline.pristine_state_sha256:
            raise RuntimeError("persistent pristine input state was mutated")
        _validate_k1_endpoint(
            owner_binding.construction_k1,
            owner_binding.construction_k1,
            scene,
            device=str(self.device),
            graph_replay=False,
        )
        _validate_k4_endpoint(
            owner_binding.construction_k4,
            owner_binding.construction_k4,
            scene,
            device=str(self.device),
            graph_replay=False,
        )

        canonical_problem = build_common_problem(scene)
        objective_sha256 = _require_sha256(
            common_objective_manifest(scene, canonical_problem)["objective_instance_sha256"],
            name="objective_instance_sha256",
        )
        retained_objective_sha256 = _require_sha256(
            common_objective_manifest(scene, owner_binding.problem)["objective_instance_sha256"],
            name="retained_objective_instance_sha256",
        )
        if objective_sha256 != self.objective_instance_sha256 or retained_objective_sha256 != objective_sha256:
            raise RuntimeError("captured direct graph retained common problem changed")
        canonical_operator = MatrixFreeStableNHOperator.from_problem(
            canonical_problem,
            owner_binding.construction_k1.positions,
        )
        if _operator_sha256(canonical_operator) != _operator_sha256(owner_binding.construction_operator):
            raise RuntimeError("captured direct graph construction operator changed")
        _, canonical_hierarchy = _canonical_static_hierarchy(
            canonical_operator,
            owner_binding.hierarchy,
            scene.rest_q,
            config,
        )
        operator_inputs_sha256 = _validate_device_operator_inputs(
            owner_binding.operator,
            canonical_operator,
            owner_binding.direct.canonical_positions,
            owner_binding.direct.x_current,
            scene,
        )
        hierarchy_inputs_sha256 = _validate_device_hierarchy_inputs(
            owner_binding.device.source,
            canonical_hierarchy,
        )
        scalar_evidence = _validate_scalar_fused_hierarchy_inputs(
            owner_binding.device.scalar,
            owner_binding.device.source,
            canonical_hierarchy,
            validate_device_content=True,
        )
        if scalar_evidence != claims.scalar_fused_evidence:
            raise RuntimeError("captured scalar-fused construction evidence changed")
        persistent_sha256 = _hash_parts(
            "captured-direct-graph-vbd-persistent-inputs-v5",
            (
                ("scene_sha256", scene_sha256),
                ("objective_instance_sha256", objective_sha256),
                ("config_sha256", config_sha256),
                ("public_model_sha256", model_sha256),
                ("pristine_state_sha256", pristine_sha256),
                ("construction_k1_endpoint_sha256", owner_binding.construction_k1.endpoint_sha256),
                ("construction_k4_endpoint_sha256", owner_binding.construction_k4.endpoint_sha256),
                ("solver_lane_contract_sha256", lane_contract_sha256),
                ("solver_scalar_sha256", solver_scalar_sha256),
                ("solver_static_array_sha256", solver_static_array_sha256),
                ("fused_gather_kernel_version", claims.fused_gather_kernel_version),
                ("scalar_direction_apply_kernel_version", claims.scalar_direction_apply_kernel_version),
                ("first_cycle_publication_role", claims.first_cycle_publication_role),
                ("second_cycle_publication_role", claims.second_cycle_publication_role),
                ("outer_kernel_version", claims.outer_kernel_version),
                ("outer_schedule_version", claims.outer_schedule_version),
                ("outer_schedule_sha256", claims.outer_schedule_sha256),
                ("finalize_gate_route", claims.finalize_gate_route),
                ("finalize_gate_block_dim", claims.finalize_gate_block_dim),
                *(
                    (f"finalize_gate_owner_thread_{index}", value)
                    for index, value in enumerate(claims.finalize_gate_owner_threads)
                ),
                *(
                    (f"finalize_gate_owner_role_{index}", value)
                    for index, value in enumerate(claims.finalize_gate_owner_roles)
                ),
                ("finalize_gate_collective_version", claims.finalize_gate_collective_version),
                ("workspace_owner_identity_sha256", workspace_owner_identity_sha256),
                ("construction_operator_sha256", _operator_sha256(canonical_operator)),
                ("operator_inputs_sha256", operator_inputs_sha256),
                ("hierarchy_inputs_sha256", hierarchy_inputs_sha256),
                ("scalar_v_cycle_schedule_sha256", scalar_evidence.schedule_sha256),
                ("scalar_v_cycle_core_schedule_sha256", scalar_evidence.core_schedule_sha256),
                ("scalar_static_device_content_sha256", scalar_evidence.static_device_content_sha256),
                ("scalar_device_snapshot_sha256", scalar_evidence.device_snapshot_sha256),
                ("scalar_core_device_snapshot_sha256", scalar_evidence.core_device_snapshot_sha256),
                ("scalar_v_cycle_kernel_launches", scalar_evidence.scheduled_kernel_launches),
                ("scalar_v_cycle_core_kernel_launches", scalar_evidence.core_kernel_launches),
                ("scalar_v_cycle_publication_version", scalar_evidence.publication_version),
                ("scalar_v_cycle_standalone_publication_route", scalar_evidence.standalone_publication_route),
                (
                    "scalar_v_cycle_external_shared_publication_route",
                    scalar_evidence.external_shared_publication_route,
                ),
            ),
        )
        if require_bound and persistent_sha256 != claims.persistent_device_sha256:
            raise RuntimeError("captured direct graph persistent input digest changed")
        if require_bound:
            expected_uncaptured_identity = self._graph_identity(
                captured=False,
                comparator=False,
                owner_binding=owner_binding,
            )
            expected_graph_identity = self._graph_identity(
                captured=True,
                comparator=False,
                owner_binding=owner_binding,
            )
            expected_k4_identity = self._graph_identity(
                captured=True,
                comparator=True,
                owner_binding=owner_binding,
            )
            if (
                expected_uncaptured_identity != claims.uncaptured_graph_identity_sha256
                or self._uncaptured_graph_identity_sha256 != expected_uncaptured_identity
                or self.graph_identity_sha256 != expected_graph_identity
                or self.k4_graph_identity_sha256 != expected_k4_identity
            ):
                raise RuntimeError("captured graph identity label is stale or was mutated")
        return persistent_sha256

    def _enqueue_integrated(self, owner_binding: _WorkspaceOwnerBinding) -> None:
        """Enqueue one fixed direct schedule from one validated owner graph."""
        self._validate_workspace_owner_bindings(owner_binding)
        public = owner_binding.public_vbd
        direct = owner_binding.direct
        scene = owner_binding.scene
        operator = owner_binding.operator
        device_hierarchy = owner_binding.device.scalar
        lane = public.k1_lane
        public.baseline._enqueue_reset_and_step(lane)
        wp.launch(
            _initialize_from_k1,
            dim=scene.n_vertices,
            inputs=[
                lane.state_out.particle_q,
                direct.canonical_positions,
                operator.vertex_to_free,
                operator.positions,
                direct.candidate,
                direct.proposal_finite,
                direct.active,
                direct.accepted,
                direct.reasons,
            ],
            device=self.device,
        )
        for outer_index, workspace in enumerate(owner_binding.outer):
            operator.launch_refresh_geometry()
            operator.launch_gradient_masked(workspace.rhs, direct.active, scale=-1.0)
            device_hierarchy.launch_apply_core(
                workspace.rhs,
                workspace.first_cycle.workspace,
            )
            operator.launch_apply_residual_scalar_direction(
                workspace.first_cycle.final_scalar_correction,
                workspace.first_correction,
                workspace.rhs,
                workspace.operator_product_after_first,
                workspace.residual_after_first,
                workspace.operator_apply,
            )
            device_hierarchy.launch_apply_core(
                workspace.residual_after_first,
                workspace.second_cycle.workspace,
            )
            wp.launch(
                _fused_vertex_outer_terms,
                dim=scene.n_vertices,
                inputs=[
                    operator.positions,
                    operator.vertex_to_free,
                    workspace.first_correction,
                    workspace.second_cycle.final_scalar_correction,
                    workspace.second_correction,
                    workspace.rhs,
                    workspace.direction,
                    direct.active,
                    operator.inertial_target,
                    operator.mass,
                    operator.inverse_dt_squared,
                    direct.outer_start_positions[outer_index],
                    direct.candidate,
                    direct.outer_candidate_positions[outer_index],
                    direct.proposal_finite,
                    direct.current_inertia,
                    direct.candidate_inertia,
                    direct.vertex_finite,
                    direct.directional_terms,
                ],
                device=self.device,
            )
            wp.launch(
                _tet_gate_terms,
                dim=scene.n_tets,
                inputs=[
                    operator.deformation_gradients,
                    operator.cofactors,
                    operator.determinants,
                    direct.candidate,
                    operator.tets,
                    operator.shape_gradients,
                    operator.volumes,
                    operator.mu,
                    operator.lam,
                    direct.current_elastic,
                    direct.candidate_elastic,
                    direct.candidate_determinants,
                    direct.segment_minima,
                    direct.tet_finite,
                ],
                device=self.device,
            )
            wp.launch(
                _finalize_gate,
                dim=FINALIZE_GATE_BLOCK_DIM,
                inputs=[
                    outer_index,
                    direct.current_inertia,
                    direct.candidate_inertia,
                    direct.current_elastic,
                    direct.candidate_elastic,
                    direct.directional_terms,
                    direct.candidate_determinants,
                    direct.segment_minima,
                    direct.proposal_finite,
                    direct.vertex_finite,
                    direct.tet_finite,
                    owner_binding.config.minimum_determinant,
                    owner_binding.config.armijo,
                    direct.active,
                    direct.accepted,
                    direct.reasons,
                    direct.initial_objectives,
                    direct.candidate_objectives,
                    direct.directional_derivatives,
                    direct.minimum_segment_determinants,
                ],
                block_dim=FINALIZE_GATE_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                _commit_candidate,
                dim=scene.n_vertices,
                inputs=[outer_index, direct.candidate, direct.accepted, operator.positions],
                device=self.device,
            )
        wp.launch(
            _write_endpoint,
            dim=scene.n_vertices,
            inputs=[
                operator.positions,
                direct.x_current,
                operator.vertex_to_free,
                1.0 / scene.dt,
                direct.final_positions,
                direct.final_velocities,
            ],
            device=self.device,
        )

    def capture_graphs(self, *, warmup_replays: int = 1) -> None:
        """Capture separate integrated direct-graph and pristine K4 graphs."""
        if isinstance(warmup_replays, bool) or not isinstance(warmup_replays, numbers.Integral) or warmup_replays < 1:
            raise ValueError("warmup_replays must be a positive integer")
        owner_binding = _lookup_workspace_owners(self)
        self._validate_persistent_sources(owner_binding=owner_binding)
        for _ in range(int(warmup_replays)):
            self._enqueue_integrated(owner_binding)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self._enqueue_integrated(owner_binding)
        integrated_graph = capture.graph

        public = owner_binding.public_vbd
        k4_lane = public.k4_lane
        for _ in range(int(warmup_replays)):
            self._validate_workspace_owner_bindings(owner_binding)
            public.baseline._enqueue_reset_and_step(k4_lane)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self._validate_workspace_owner_bindings(owner_binding)
            public.baseline._enqueue_reset_and_step(k4_lane)
        k4_graph = capture.graph
        replay_stream = owner_binding.replay_stream.stream
        self._validate_replay_stream_owner_binding(owner_binding)
        wp.capture_launch(integrated_graph, stream=replay_stream)
        wp.capture_launch(k4_graph, stream=replay_stream)
        wp.synchronize_stream(replay_stream)
        graph_native = self._capture_native_graph_owner_binding(integrated_graph, owner_binding.warp_device)
        k4_graph_native = self._capture_native_graph_owner_binding(k4_graph, owner_binding.warp_device)
        generation = 1 if owner_binding.capture is None else owner_binding.capture.generation + 1
        object_identity = (id(integrated_graph), id(k4_graph))
        graph_identity_sha256 = self._derive_graph_identity(
            captured=True,
            comparator=False,
            owner_binding=owner_binding,
            graph=integrated_graph,
            generation=generation,
        )
        k4_graph_identity_sha256 = self._derive_graph_identity(
            captured=True,
            comparator=True,
            owner_binding=owner_binding,
            graph=k4_graph,
            generation=generation,
        )
        capture_binding = _CaptureGraphOwnerBinding(
            graph=integrated_graph,
            k4_graph=k4_graph,
            graph_type=type(integrated_graph),
            graph_native=graph_native,
            k4_graph_native=k4_graph_native,
            replay_stream=owner_binding.replay_stream,
            generation=generation,
            object_identity=object_identity,
            graph_identity_sha256=graph_identity_sha256,
            k4_graph_identity_sha256=k4_graph_identity_sha256,
        )
        replacement_binding = owner_binding._replace(capture=capture_binding)
        old_facade = (
            self.graph,
            self.k4_graph,
            self._captured_graph_object_identity,
            self._capture_generation,
            self.graph_identity_sha256,
            self.k4_graph_identity_sha256,
        )
        try:
            self.graph = integrated_graph
            self.k4_graph = k4_graph
            self._captured_graph_object_identity = object_identity
            self._capture_generation = generation
            self.graph_identity_sha256 = graph_identity_sha256
            self.k4_graph_identity_sha256 = k4_graph_identity_sha256
            self._validate_persistent_sources(owner_binding=replacement_binding)
            _replace_workspace_owners(self, owner_binding, replacement_binding)
        except Exception:
            (
                self.graph,
                self.k4_graph,
                self._captured_graph_object_identity,
                self._capture_generation,
                self.graph_identity_sha256,
                self.k4_graph_identity_sha256,
            ) = old_facade
            raise

    def run(self, *, graph_replay: bool = True) -> CapturedGraphVBDEndpoint:
        """Execute and synchronize one integrated direct graph lane."""
        if type(graph_replay) is not bool:
            raise ValueError("graph_replay must be a bool")
        owner_binding = _lookup_workspace_owners(self)
        self._validate_persistent_sources(owner_binding=owner_binding)
        capture_binding = owner_binding.capture
        if graph_replay:
            if capture_binding is None:
                raise RuntimeError("capture_graphs() must complete before graph replay")
            graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=False)
            wp.capture_launch(graph, stream=replay_stream)
        else:
            self._enqueue_integrated(owner_binding)
        owner_binding.public_vbd.k1_lane.completed_launches += 1
        self._execution_serial += 1
        receipt = _ExecutionReceipt(
            issuer=self,
            serial=self._execution_serial,
            graph_replay=graph_replay,
            capture_binding=capture_binding if graph_replay else None,
        )
        self._issued_execution_receipts[id(receipt)] = receipt
        return self._record(execution_receipt=receipt, owner_binding=owner_binding)

    def run_k4(self, *, graph_replay: bool = True):
        """Execute the separate pristine K4 comparator lane."""
        if type(graph_replay) is not bool:
            raise ValueError("graph_replay must be a bool")
        owner_binding = _lookup_workspace_owners(self)
        self._validate_persistent_sources(owner_binding=owner_binding)
        public = owner_binding.public_vbd
        lane = public.k4_lane
        capture_binding = owner_binding.capture
        if graph_replay:
            if capture_binding is None:
                raise RuntimeError("capture_graphs() must complete before graph replay")
            graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=True)
            wp.capture_launch(graph, stream=replay_stream)
        else:
            self._validate_workspace_owner_bindings(owner_binding)
            public.baseline._enqueue_reset_and_step(lane)
        lane.completed_launches += 1
        endpoint = public.baseline.record(4, graph_replay=graph_replay)
        _validate_k4_endpoint(
            endpoint,
            self._construction_k4,
            self.scene,
            device=str(self.device),
            graph_replay=graph_replay,
        )
        self._validate_persistent_sources(owner_binding=owner_binding)
        return endpoint

    def benchmark_paired(
        self,
        *,
        pair_count: int = 10,
        warmup_replays: int = 5,
        random_seed: int = 20260817,
    ) -> CapturedGraphVBDTiming:
        """Measure captured direct graph VBD and K4 in balanced AB/BA order."""
        for name, value, minimum in (("pair_count", pair_count, 2), ("warmup_replays", warmup_replays, 1)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if int(pair_count) % 2 != 0:
            raise ValueError("pair_count must be even for balanced AB/BA timing")
        if isinstance(random_seed, bool) or not isinstance(random_seed, numbers.Integral):
            raise ValueError("random_seed must be an integer")
        owner_binding = _lookup_workspace_owners(self)
        persistent_device_sha256 = self._validate_persistent_sources(owner_binding=owner_binding)
        capture_binding = owner_binding.capture
        if capture_binding is None:
            raise RuntimeError("capture_graphs() must complete before timing")
        graph_identity_sha256 = self._graph_identity(
            captured=True,
            comparator=False,
            owner_binding=owner_binding,
        )
        k4_graph_identity_sha256 = self._graph_identity(
            captured=True,
            comparator=True,
            owner_binding=owner_binding,
        )

        for _ in range(int(warmup_replays)):
            graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=False)
            wp.capture_launch(graph, stream=replay_stream)
            graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=True)
            wp.capture_launch(graph, stream=replay_stream)
        wp.synchronize_stream(capture_binding.replay_stream.stream)

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
                graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=False)
                begin, end = graph_events[pair_index]
                replay_stream.record_event(begin)
                wp.capture_launch(graph, stream=replay_stream)
                replay_stream.record_event(end)
                graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=True)
                begin, end = k4_events[pair_index]
                replay_stream.record_event(begin)
                wp.capture_launch(graph, stream=replay_stream)
                replay_stream.record_event(end)
            else:
                graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=True)
                begin, end = k4_events[pair_index]
                replay_stream.record_event(begin)
                wp.capture_launch(graph, stream=replay_stream)
                replay_stream.record_event(end)
                graph, replay_stream = self._bound_graph_launch_owners(owner_binding, comparator=False)
                begin, end = graph_events[pair_index]
                replay_stream.record_event(begin)
                wp.capture_launch(graph, stream=replay_stream)
                replay_stream.record_event(end)
        wp.synchronize_event(graph_events[-1][1])
        wp.synchronize_event(k4_events[-1][1])
        if self._validate_persistent_sources(owner_binding=owner_binding) != persistent_device_sha256:
            raise RuntimeError("persistent captured inputs changed during paired timing")
        scalar_evidence = owner_binding.claims.scalar_fused_evidence
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
            fused_gather_kernel_version=owner_binding.claims.fused_gather_kernel_version,
            scalar_direction_apply_kernel_version=owner_binding.claims.scalar_direction_apply_kernel_version,
            v_cycle_kernel_version=V_CYCLE_KERNEL_VERSION,
            v_cycle_schedule_version=V_CYCLE_SCHEDULE_VERSION,
            v_cycle_schedule_sha256=scalar_evidence.schedule_sha256,
            v_cycle_core_schedule_sha256=scalar_evidence.core_schedule_sha256,
            v_cycle_publication_version=scalar_evidence.publication_version,
            v_cycle_standalone_publication_route=scalar_evidence.standalone_publication_route,
            v_cycle_external_shared_publication_route=scalar_evidence.external_shared_publication_route,
            first_cycle_publication_role=owner_binding.claims.first_cycle_publication_role,
            second_cycle_publication_role=owner_binding.claims.second_cycle_publication_role,
            v_cycle_kernel_launches=scalar_evidence.scheduled_kernel_launches,
            v_cycle_core_kernel_launches=scalar_evidence.core_kernel_launches,
            outer_kernel_version=owner_binding.claims.outer_kernel_version,
            outer_schedule_version=owner_binding.claims.outer_schedule_version,
            outer_schedule_sha256=owner_binding.claims.outer_schedule_sha256,
            finalize_gate_route=owner_binding.claims.finalize_gate_route,
            finalize_gate_block_dim=owner_binding.claims.finalize_gate_block_dim,
            finalize_gate_owner_threads=owner_binding.claims.finalize_gate_owner_threads,
            finalize_gate_owner_roles=owner_binding.claims.finalize_gate_owner_roles,
            finalize_gate_collective_version=owner_binding.claims.finalize_gate_collective_version,
            correction_kernel_launches=self.correction_kernel_launches,
        )

    def _record(
        self,
        *,
        execution_receipt: object,
        owner_binding: _WorkspaceOwnerBinding | None = None,
    ) -> CapturedGraphVBDEndpoint:
        """Consume one solver-private execution and materialize validated evidence."""
        if owner_binding is None:
            owner_binding = _lookup_workspace_owners(self)
        receipt = self._consume_execution_receipt(execution_receipt, owner_binding)
        graph_replay = receipt.graph_replay
        self._validate_workspace_owner_bindings(owner_binding)
        public = owner_binding.public_vbd
        direct = owner_binding.direct
        execution_k1 = public.baseline.record(1, graph_replay=graph_replay)
        persistent_device_sha256 = self._validate_persistent_sources(owner_binding=owner_binding)
        graph_identity_sha256 = self._graph_identity(
            captured=graph_replay,
            comparator=False,
            owner_binding=owner_binding,
        )
        scalar_evidence = _validate_scalar_fused_hierarchy_inputs(
            owner_binding.device.scalar,
            owner_binding.device.source,
            owner_binding.hierarchy,
            validate_device_content=False,
        )
        positions = np.asarray(direct.final_positions.numpy(), dtype=np.float32).astype(np.float64)
        velocities = np.asarray(direct.final_velocities.numpy(), dtype=np.float32).astype(np.float64)
        starts = tuple(np.asarray(value.numpy(), dtype=np.float64) for value in direct.outer_start_positions)
        candidates = tuple(np.asarray(value.numpy(), dtype=np.float64) for value in direct.outer_candidate_positions)
        rhs_values = tuple(np.asarray(workspace.rhs.numpy(), dtype=np.float64) for workspace in owner_binding.outer)
        accepted = tuple(bool(value) for value in direct.accepted.numpy())
        reason_codes = tuple(int(value) for value in direct.reasons.numpy())
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
        canonical_problem = build_common_problem(owner_binding.scene)
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
            scene=owner_binding.scene,
            config=owner_binding.config,
            construction_k1=self._construction_k1,
            execution_k1=execution_k1,
            hierarchy=owner_binding.hierarchy,
            persistent_device_sha256=persistent_device_sha256,
            v_cycle_schedule_sha256=scalar_evidence.schedule_sha256,
            v_cycle_core_schedule_sha256=scalar_evidence.core_schedule_sha256,
            v_cycle_static_device_content_sha256=scalar_evidence.static_device_content_sha256,
            v_cycle_device_snapshot_sha256=scalar_evidence.device_snapshot_sha256,
            v_cycle_core_device_snapshot_sha256=scalar_evidence.core_device_snapshot_sha256,
            graph_identity_sha256=graph_identity_sha256,
            fused_gather_kernel_version=owner_binding.claims.fused_gather_kernel_version,
            scalar_direction_apply_kernel_version=owner_binding.claims.scalar_direction_apply_kernel_version,
            v_cycle_publication_version=scalar_evidence.publication_version,
            v_cycle_standalone_publication_route=scalar_evidence.standalone_publication_route,
            v_cycle_external_shared_publication_route=scalar_evidence.external_shared_publication_route,
            first_cycle_publication_role=owner_binding.claims.first_cycle_publication_role,
            second_cycle_publication_role=owner_binding.claims.second_cycle_publication_role,
            outer_kernel_version=owner_binding.claims.outer_kernel_version,
            outer_schedule_version=owner_binding.claims.outer_schedule_version,
            outer_schedule_sha256=owner_binding.claims.outer_schedule_sha256,
            finalize_gate_route=owner_binding.claims.finalize_gate_route,
            finalize_gate_block_dim=owner_binding.claims.finalize_gate_block_dim,
            finalize_gate_owner_threads=owner_binding.claims.finalize_gate_owner_threads,
            finalize_gate_owner_roles=owner_binding.claims.finalize_gate_owner_roles,
            finalize_gate_collective_version=owner_binding.claims.finalize_gate_collective_version,
            capture_binding=receipt.capture_binding,
            v_cycle_kernel_launches=scalar_evidence.scheduled_kernel_launches,
            v_cycle_core_kernel_launches=scalar_evidence.core_kernel_launches,
            v_cycle_root_ingress_zero_start_fusions=(scalar_evidence.root_ingress_zero_start_fusions),
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
            receipt.capture_binding,
            outer_slots,
            outer_slot_records,
            owner_binding,
        )
        outer_work = []
        for outer_index, (start, workspace, canonical_operator, rhs, slot) in enumerate(
            zip(starts, owner_binding.outer, canonical_operators, rhs_values, outer_slots, strict=True)
        ):
            operator_sha256 = _operator_sha256(canonical_operator)
            outer_work.append(
                CapturedGraphVBDOuterWork(
                    outer_index=outer_index,
                    start_position_sha256=_array_digest(start),
                    current_operator_sha256=operator_sha256,
                    static_hierarchy_sha256=owner_binding.device.scalar.hierarchy_sha256,
                    rhs=rhs,
                    first_correction=np.asarray(workspace.first_correction.numpy(), dtype=np.float64),
                    operator_product_after_first=np.asarray(
                        workspace.operator_product_after_first.numpy(), dtype=np.float64
                    ),
                    residual_after_first=np.asarray(workspace.residual_after_first.numpy(), dtype=np.float64),
                    second_correction=np.asarray(workspace.second_correction.numpy(), dtype=np.float64),
                    direction=np.asarray(workspace.direction.numpy(), dtype=np.float64),
                    v_cycles=(
                        workspace.first_cycle.workspace.record_core_application(
                            token=_CORE_RECORD_TOKEN,
                            capture_replay=graph_replay,
                        ),
                        workspace.second_cycle.workspace.record_core_application(
                            token=_CORE_RECORD_TOKEN,
                            capture_replay=graph_replay,
                        ),
                    ),
                    capture_replay=graph_replay,
                    linear_kernel_launches=(5 + 2 * owner_binding.device.scalar.core_kernel_launches),
                    persistent_device_sha256=persistent_device_sha256,
                    fused_gather_kernel_version=owner_binding.claims.fused_gather_kernel_version,
                    scalar_direction_apply_kernel_version=(owner_binding.claims.scalar_direction_apply_kernel_version),
                    v_cycle_publication_version=scalar_evidence.publication_version,
                    first_cycle_publication_route=owner_binding.claims.first_cycle_publication_role,
                    second_cycle_publication_route=owner_binding.claims.second_cycle_publication_role,
                    outer_kernel_version=owner_binding.claims.outer_kernel_version,
                    outer_schedule_version=owner_binding.claims.outer_schedule_version,
                    outer_schedule_sha256=owner_binding.claims.outer_schedule_sha256,
                    finalize_gate_route=owner_binding.claims.finalize_gate_route,
                    finalize_gate_block_dim=owner_binding.claims.finalize_gate_block_dim,
                    finalize_gate_owner_threads=owner_binding.claims.finalize_gate_owner_threads,
                    finalize_gate_owner_roles=owner_binding.claims.finalize_gate_owner_roles,
                    finalize_gate_collective_version=owner_binding.claims.finalize_gate_collective_version,
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
            static_hierarchy_sha256=owner_binding.device.scalar.hierarchy_sha256,
            config_sha256=self.config_sha256,
            k1_endpoint_sha256=execution_k1.endpoint_sha256,
            k1_position_sha256=execution_k1.position_sha256,
            k1_velocity_sha256=execution_k1.velocity_sha256,
            k1_pristine_state_sha256=execution_k1.pristine_state_sha256,
            persistent_device_sha256=persistent_device_sha256,
            graph_identity_sha256=graph_identity_sha256,
            fused_gather_kernel_version=owner_binding.claims.fused_gather_kernel_version,
            scalar_direction_apply_kernel_version=owner_binding.claims.scalar_direction_apply_kernel_version,
            v_cycle_publication_version=scalar_evidence.publication_version,
            v_cycle_standalone_publication_route=scalar_evidence.standalone_publication_route,
            v_cycle_external_shared_publication_route=scalar_evidence.external_shared_publication_route,
            first_cycle_publication_role=owner_binding.claims.first_cycle_publication_role,
            second_cycle_publication_role=owner_binding.claims.second_cycle_publication_role,
            outer_kernel_version=owner_binding.claims.outer_kernel_version,
            outer_schedule_version=owner_binding.claims.outer_schedule_version,
            outer_schedule_sha256=owner_binding.claims.outer_schedule_sha256,
            finalize_gate_route=owner_binding.claims.finalize_gate_route,
            finalize_gate_block_dim=owner_binding.claims.finalize_gate_block_dim,
            finalize_gate_owner_threads=owner_binding.claims.finalize_gate_owner_threads,
            finalize_gate_owner_roles=owner_binding.claims.finalize_gate_owner_roles,
            finalize_gate_collective_version=owner_binding.claims.finalize_gate_collective_version,
            armijo=float(owner_binding.config.armijo),
            minimum_determinant=float(owner_binding.config.minimum_determinant),
            free_vertices=np.asarray(owner_binding.operator.free_host, dtype=np.int64),
            positions=positions,
            velocities=velocities,
            accepted=accepted,
            reasons=reasons,
            initial_objectives=tuple(float(value) for value in direct.initial_objectives.numpy()),
            candidate_objectives=tuple(float(value) for value in direct.candidate_objectives.numpy()),
            directional_derivatives=tuple(float(value) for value in direct.directional_derivatives.numpy()),
            segment_minimum_determinants=tuple(float(value) for value in direct.minimum_segment_determinants.numpy()),
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
        owner_binding = _lookup_workspace_owners(self)
        self._validate_workspace_owner_bindings(owner_binding)
        direct = owner_binding.direct
        operator = owner_binding.operator
        rng = np.random.default_rng(int(seed))
        owner_binding.public_vbd.baseline.poison_lane(1, seed=int(seed) + 1)
        for array in (
            operator.positions,
            operator.deformation_gradients,
            operator.cofactors,
            operator.determinants,
            operator.first_piola,
            direct.candidate,
            direct.proposal_finite,
            direct.final_positions,
            direct.final_velocities,
            direct.active,
            direct.accepted,
            direct.reasons,
            direct.current_inertia,
            direct.candidate_inertia,
            direct.vertex_finite,
            direct.current_elastic,
            direct.candidate_elastic,
            direct.candidate_determinants,
            direct.segment_minima,
            direct.tet_finite,
            direct.directional_terms,
            direct.initial_objectives,
            direct.candidate_objectives,
            direct.directional_derivatives,
            direct.minimum_segment_determinants,
            *direct.outer_start_positions,
            *direct.outer_candidate_positions,
        ):
            self._poison_array(array, rng)
        for workspace in owner_binding.outer:
            first_cycle = workspace.first_cycle
            second_cycle = workspace.second_cycle
            for array in (
                workspace.rhs,
                workspace.first_correction,
                workspace.operator_product_after_first,
                workspace.residual_after_first,
                workspace.second_correction,
                workspace.direction,
                workspace.operator_apply_delta_piola,
                *first_cycle.level_rhs,
                *first_cycle.level_correction,
                *first_cycle.level_correction_alt,
                *first_cycle.level_residual,
                first_cycle.coarse_intermediate,
                *second_cycle.level_rhs,
                *second_cycle.level_correction,
                *second_cycle.level_correction_alt,
                *second_cycle.level_residual,
                second_cycle.coarse_intermediate,
            ):
                self._poison_array(array, rng)

    def deterministic_record(self) -> dict[str, object]:
        """Return the fixed graph schedule and scope without timing data."""
        owner_binding = _lookup_workspace_owners(self)
        persistent_device_sha256 = self._validate_persistent_sources(owner_binding=owner_binding)
        graph_identity_sha256 = self._graph_identity(
            captured=True,
            comparator=False,
            owner_binding=owner_binding,
        )
        k4_graph_identity_sha256 = self._graph_identity(
            captured=True,
            comparator=True,
            owner_binding=owner_binding,
        )
        scalar_evidence = _validate_scalar_fused_hierarchy_inputs(
            owner_binding.device.scalar,
            owner_binding.device.source,
            owner_binding.hierarchy,
            validate_device_content=False,
        )
        hierarchy = owner_binding.hierarchy
        config = owner_binding.config
        noncoarse_levels = hierarchy.levels[:-1]
        canonical_matrix_products = sum(
            level.matrix.stored_block_count * (hierarchy.pre_smooth_steps + 1 + hierarchy.post_smooth_steps)
            for level in noncoarse_levels
        )
        smoother_solves = sum(
            level.matrix.block_row_count * (hierarchy.pre_smooth_steps + hierarchy.post_smooth_steps)
            for level in noncoarse_levels
        )
        elided_matrix_products = sum(level.matrix.stored_block_count for level in noncoarse_levels)
        zero_start_solves = sum(level.matrix.block_row_count for level in noncoarse_levels)
        matrix_launches = len(noncoarse_levels) * (hierarchy.pre_smooth_steps + hierarchy.post_smooth_steps)
        linear_prefix_kernel_launches = 4 + 2 * scalar_evidence.core_kernel_launches
        linear_kernel_launches = linear_prefix_kernel_launches + 1
        outer_kernel_launches = linear_kernel_launches + 3
        correction_kernel_launches = 2 + OUTER_CORRECTIONS * outer_kernel_launches
        return {
            "contract_id": CONTRACT_ID,
            "scene_sha256": self.scene_sha256,
            "objective_instance_sha256": self.objective_instance_sha256,
            "config": config.deterministic_record(),
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
            "workspace_owner_identity_sha256": self._workspace_owner_identity_sha256(owner_binding),
            "persistent_device_sha256": persistent_device_sha256,
            "graph_identity_schema": "captured-direct-graph-vbd-graph-identity-v7",
            "graph_identity_sha256": graph_identity_sha256,
            "k4_graph_identity_sha256": k4_graph_identity_sha256,
            "fused_gather_kernel_version": owner_binding.claims.fused_gather_kernel_version,
            "scalar_direction_apply_kernel_version": (owner_binding.claims.scalar_direction_apply_kernel_version),
            "first_cycle_publication_role": owner_binding.claims.first_cycle_publication_role,
            "second_cycle_publication_role": owner_binding.claims.second_cycle_publication_role,
            "outer_kernel_version": owner_binding.claims.outer_kernel_version,
            "outer_schedule_version": owner_binding.claims.outer_schedule_version,
            "outer_schedule_sha256": owner_binding.claims.outer_schedule_sha256,
            "finalize_gate_route": owner_binding.claims.finalize_gate_route,
            "finalize_gate_block_dim": owner_binding.claims.finalize_gate_block_dim,
            "finalize_gate_owner_threads": list(owner_binding.claims.finalize_gate_owner_threads),
            "finalize_gate_owner_roles": list(owner_binding.claims.finalize_gate_owner_roles),
            "finalize_gate_collective_version": owner_binding.claims.finalize_gate_collective_version,
            "device": str(self.device),
            "vbd_iterations": 1,
            "outer_corrections": OUTER_CORRECTIONS,
            "stationary_v_cycles_per_outer": V_CYCLES_PER_OUTER,
            "v_cycles": OUTER_CORRECTIONS * V_CYCLES_PER_OUTER,
            "krylov_iterations": 0,
            "linear_formula": "d=B*b+B*(b-A_current*B*b)",
            "current_operator_refreshes": OUTER_CORRECTIONS,
            "current_operator_applications": OUTER_CORRECTIONS,
            "static_hierarchy_sha256": owner_binding.device.scalar.hierarchy_sha256,
            "hierarchy_level_count": len(owner_binding.device.scalar_levels),
            "hierarchy_level_scalar_sizes": [level.scalar_size for level in owner_binding.device.scalar_levels],
            "v_cycle_contract_id": V_CYCLE_CONTRACT_ID,
            "v_cycle_kernel_version": V_CYCLE_KERNEL_VERSION,
            "v_cycle_schedule_version": V_CYCLE_SCHEDULE_VERSION,
            "v_cycle_schedule_sha256": scalar_evidence.schedule_sha256,
            "v_cycle_core_schedule_sha256": scalar_evidence.core_schedule_sha256,
            "v_cycle_publication_version": scalar_evidence.publication_version,
            "v_cycle_standalone_publication_route": scalar_evidence.standalone_publication_route,
            "v_cycle_external_shared_publication_route": scalar_evidence.external_shared_publication_route,
            "v_cycle_static_device_content_sha256": scalar_evidence.static_device_content_sha256,
            "v_cycle_device_snapshot_sha256": scalar_evidence.device_snapshot_sha256,
            "v_cycle_core_device_snapshot_sha256": scalar_evidence.core_device_snapshot_sha256,
            "v_cycle_kernel_launches": owner_binding.device.scalar.scheduled_kernel_launches,
            "v_cycle_core_kernel_launches": owner_binding.device.scalar.core_kernel_launches,
            "first_cycle_kernel_launches": owner_binding.device.scalar.core_kernel_launches,
            "second_cycle_kernel_launches": owner_binding.device.scalar.core_kernel_launches,
            "v_cycle_first_publication_kernel_launches": 0,
            "v_cycle_second_publication_kernel_launches": 0,
            "v_cycle_root_ingress_zero_start_fusions": (scalar_evidence.root_ingress_zero_start_fusions),
            "v_cycle_canonical_matrix_block_products": canonical_matrix_products,
            "v_cycle_matrix_block_products_executed": canonical_matrix_products - elided_matrix_products,
            "v_cycle_matrix_block_products_elided_zero_start": elided_matrix_products,
            "v_cycle_zero_start_block_solves": zero_start_solves,
            "v_cycle_out_of_place_jacobi_block_solves": smoother_solves - zero_start_solves,
            "v_cycle_matrix_kernel_launches": matrix_launches,
            "v_cycle_jacobi_kernel_launches": matrix_launches,
            "linear_prefix_kernel_launches_per_outer": linear_prefix_kernel_launches,
            "fused_gather_kernel_launches_per_outer": 2,
            "fused_vertex_kernel_launches_per_outer": 1,
            "finalize_gate_kernel_launches_per_outer": 1,
            "linear_kernel_launches_per_outer": linear_kernel_launches,
            "outer_kernel_launches_per_outer": outer_kernel_launches,
            "correction_kernel_launches_excluding_public_k1": correction_kernel_launches,
            "alpha": config.alpha,
            "armijo": config.armijo,
            "minimum_determinant": config.minimum_determinant,
            "gate_execution": "device-side-published-fp32-strict-armijo-exact-cubic-segment-fail-closed",
            "rejection_mask": "sticky-after-first-rejection-with-fixed-work",
            "final_velocity": "BDF1-from-physical-x-current-exact-zero-pins",
            "separate_k4_graph": True,
            "performance_evidence": False,
        }
