# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Launch-fused device V-cycle over an immutable static Warp hierarchy.

The committed :mod:`correction_multigrid_warp` hierarchy remains the source
of every persistent operator, transfer, smoother, and coarsest-factor array.
This wrapper changes only the fixed launch schedule.  A noncoarsest level
uses one block-row-owner launch for each Jacobi sweep, one fused matrix-vector
product/residual launch, deterministic restriction, recursive coarse work,
and deterministic prolongation.  Jacobi updates ping-pong between two
buffers, so off-diagonal reads never race writes from another block row.

For ``n`` noncoarsest levels, ``p`` pre-sweeps, and ``q`` post-sweeps, one
application schedules exactly ``3 + n * (3 + p + q)`` kernels.  The launch
path allocates no arrays, performs no synchronization or device readback, and
is CUDA-graph capturable after warm-up.  Diagnostic records separately retain
the canonical V-cycle algebra and the physically executed matrix work; the
zero-start first sweep elides one matrix traversal per noncoarsest level.
Host metadata, topology, and device-array pointers are checked before launch.
Mutating the contents of a shared static device array cannot be detected
without the prohibited readback; the wrapped hierarchy therefore remains an
explicit immutable-input contract, and nonfinite corruption fails at record.
This is research-only evidence, not a performance claim.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import warp as wp

from .correction_gpu_warp import WarpDevicePreconditioner, WarpDevicePreconditionerApplication
from .correction_multigrid import StaticMultigridHierarchy, VCycleWorkRecord
from .correction_multigrid_warp import (
    WarpStaticMultigridHierarchy,
    _copy_scalar_to_vec3,
    _copy_vec3_to_scalar,
    _hash_parts,
    _immutable_array,
    _solve_coarsest_cholesky,
)

KERNEL_VERSION = "mg-vbd-warp-static-v-cycle-fused-v1"
CONTRACT_ID = "spectral-free-multiplicative-graph-vbd-warp-static-fused-v1"
SCHEDULE_VERSION = "zero-jacobi-residual-restrict-recurse-prolong-ping-pong-v1"
SUPPORTED_BLOCK_SIZES = (3, 6)

# Keep the kernel marker visible in test output so a stale Warp cache cannot
# be mistaken for this launch-fusion milestone.
print(f"[kernels] version: {KERNEL_VERSION}")


class _Vec6d(wp.types.vector(length=6, dtype=wp.float64)):
    pass


def _require_sha256(value: object, *, name: str) -> str:
    """Require one canonical lowercase SHA-256 string."""
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _work_sha256(work: VCycleWorkRecord) -> str:
    """Recompute the canonical CPU V-cycle work identity."""
    return _hash_parts(
        "v-cycle-work-record-v1",
        (
            ("hierarchy_sha256", work.hierarchy_sha256),
            ("rhs_sha256", work.rhs_sha256),
            ("result_sha256", work.result_sha256),
            ("rhs_count", work.rhs_count),
            ("level_visits", _immutable_array(work.level_visits, np.int64)),
            ("matrix_block_products", work.matrix_block_products),
            ("smoother_block_solves", work.smoother_block_solves),
            ("restriction_block_products", work.restriction_block_products),
            ("prolongation_block_products", work.prolongation_block_products),
            ("coarsest_factor_solves", work.coarsest_factor_solves),
        ),
    )


@wp.kernel(enable_backward=False)
def _zero_start_block_jacobi_3(
    rhs: wp.array[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    omega: wp.float64,
    output: wp.array[wp.float64],
):
    block_row = wp.tid()
    vector_base = block_row * 3
    block_base = block_row * 9
    for local_row in range(3):
        value = wp.float64(0.0)
        for local_column in range(3):
            value += inverse_diagonal[block_base + local_row * 3 + local_column] * rhs[vector_base + local_column]
        output[vector_base + local_row] = omega * value


@wp.kernel(enable_backward=False)
def _zero_start_block_jacobi_6(
    rhs: wp.array[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    omega: wp.float64,
    output: wp.array[wp.float64],
):
    block_row = wp.tid()
    vector_base = block_row * 6
    block_base = block_row * 36
    for local_row in range(6):
        value = wp.float64(0.0)
        for local_column in range(6):
            value += inverse_diagonal[block_base + local_row * 6 + local_column] * rhs[vector_base + local_column]
        output[vector_base + local_row] = omega * value


@wp.kernel(enable_backward=False)
def _block_csr_residual_3(
    row_offsets: wp.array[wp.int32],
    column_indices: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    vector: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    block_row = wp.tid()
    output_base = block_row * 3
    for local_row in range(3):
        product = wp.float64(0.0)
        for entry in range(row_offsets[block_row], row_offsets[block_row + 1]):
            vector_base = column_indices[entry] * 3
            value_base = (entry * 3 + local_row) * 3
            for local_column in range(3):
                product += values[value_base + local_column] * vector[vector_base + local_column]
        residual[output_base + local_row] = rhs[output_base + local_row] - product


@wp.kernel(enable_backward=False)
def _block_csr_residual_6(
    row_offsets: wp.array[wp.int32],
    column_indices: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    vector: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    block_row = wp.tid()
    output_base = block_row * 6
    for local_row in range(6):
        product = wp.float64(0.0)
        for entry in range(row_offsets[block_row], row_offsets[block_row + 1]):
            vector_base = column_indices[entry] * 6
            value_base = (entry * 6 + local_row) * 6
            for local_column in range(6):
                product += values[value_base + local_column] * vector[vector_base + local_column]
        residual[output_base + local_row] = rhs[output_base + local_row] - product


@wp.kernel(enable_backward=False)
def _fused_block_jacobi_3(
    row_offsets: wp.array[wp.int32],
    column_indices: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    omega: wp.float64,
    current: wp.array[wp.float64],
    output: wp.array[wp.float64],
):
    block_row = wp.tid()
    vector_base = block_row * 3
    block_base = block_row * 9
    residual = wp.vec3d(wp.float64(0.0))
    for local_row in range(3):
        product = wp.float64(0.0)
        for entry in range(row_offsets[block_row], row_offsets[block_row + 1]):
            input_base = column_indices[entry] * 3
            value_base = (entry * 3 + local_row) * 3
            for local_column in range(3):
                product += values[value_base + local_column] * current[input_base + local_column]
        residual[local_row] = rhs[vector_base + local_row] - product
    for local_row in range(3):
        update = wp.float64(0.0)
        for local_column in range(3):
            update += inverse_diagonal[block_base + local_row * 3 + local_column] * residual[local_column]
        output[vector_base + local_row] = current[vector_base + local_row] + omega * update


@wp.kernel(enable_backward=False)
def _fused_block_jacobi_6(
    row_offsets: wp.array[wp.int32],
    column_indices: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    omega: wp.float64,
    current: wp.array[wp.float64],
    output: wp.array[wp.float64],
):
    block_row = wp.tid()
    vector_base = block_row * 6
    block_base = block_row * 36
    residual = _Vec6d(wp.float64(0.0))
    for local_row in range(6):
        product = wp.float64(0.0)
        for entry in range(row_offsets[block_row], row_offsets[block_row + 1]):
            input_base = column_indices[entry] * 6
            value_base = (entry * 6 + local_row) * 6
            for local_column in range(6):
                product += values[value_base + local_column] * current[input_base + local_column]
        residual[local_row] = rhs[vector_base + local_row] - product
    for local_row in range(6):
        update = wp.float64(0.0)
        for local_column in range(6):
            update += inverse_diagonal[block_base + local_row * 6 + local_column] * residual[local_column]
        output[vector_base + local_row] = current[vector_base + local_row] + omega * update


@wp.kernel(enable_backward=False)
def _restrict_owned_3_3(
    member_offsets: wp.array[wp.int32],
    member_fine_nodes: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
):
    coarse_node = wp.tid()
    for coarse_local in range(3):
        result = wp.float64(0.0)
        for cursor in range(member_offsets[coarse_node], member_offsets[coarse_node + 1]):
            fine_base = member_fine_nodes[cursor] * 3
            for fine_local in range(3):
                result += (
                    prolongation_blocks[(fine_base + fine_local) * 3 + coarse_local]
                    * fine_value[fine_base + fine_local]
                )
        coarse_value[coarse_node * 3 + coarse_local] = result


@wp.kernel(enable_backward=False)
def _restrict_owned_3_6(
    member_offsets: wp.array[wp.int32],
    member_fine_nodes: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
):
    coarse_node = wp.tid()
    for coarse_local in range(6):
        result = wp.float64(0.0)
        for cursor in range(member_offsets[coarse_node], member_offsets[coarse_node + 1]):
            fine_base = member_fine_nodes[cursor] * 3
            for fine_local in range(3):
                result += (
                    prolongation_blocks[(fine_base + fine_local) * 6 + coarse_local]
                    * fine_value[fine_base + fine_local]
                )
        coarse_value[coarse_node * 6 + coarse_local] = result


@wp.kernel(enable_backward=False)
def _restrict_owned_6_3(
    member_offsets: wp.array[wp.int32],
    member_fine_nodes: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
):
    coarse_node = wp.tid()
    for coarse_local in range(3):
        result = wp.float64(0.0)
        for cursor in range(member_offsets[coarse_node], member_offsets[coarse_node + 1]):
            fine_base = member_fine_nodes[cursor] * 6
            for fine_local in range(6):
                result += (
                    prolongation_blocks[(fine_base + fine_local) * 3 + coarse_local]
                    * fine_value[fine_base + fine_local]
                )
        coarse_value[coarse_node * 3 + coarse_local] = result


@wp.kernel(enable_backward=False)
def _restrict_owned_6_6(
    member_offsets: wp.array[wp.int32],
    member_fine_nodes: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
):
    coarse_node = wp.tid()
    for coarse_local in range(6):
        result = wp.float64(0.0)
        for cursor in range(member_offsets[coarse_node], member_offsets[coarse_node + 1]):
            fine_base = member_fine_nodes[cursor] * 6
            for fine_local in range(6):
                result += (
                    prolongation_blocks[(fine_base + fine_local) * 6 + coarse_local]
                    * fine_value[fine_base + fine_local]
                )
        coarse_value[coarse_node * 6 + coarse_local] = result


@wp.kernel(enable_backward=False)
def _prolong_add_owned_3_3(
    aggregate: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
):
    fine_node = wp.tid()
    fine_base = fine_node * 3
    coarse_base = aggregate[fine_node] * 3
    for fine_local in range(3):
        result = wp.float64(0.0)
        for coarse_local in range(3):
            result += (
                prolongation_blocks[(fine_base + fine_local) * 3 + coarse_local]
                * coarse_value[coarse_base + coarse_local]
            )
        fine_value[fine_base + fine_local] += result


@wp.kernel(enable_backward=False)
def _prolong_add_owned_3_6(
    aggregate: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
):
    fine_node = wp.tid()
    fine_base = fine_node * 3
    coarse_base = aggregate[fine_node] * 6
    for fine_local in range(3):
        result = wp.float64(0.0)
        for coarse_local in range(6):
            result += (
                prolongation_blocks[(fine_base + fine_local) * 6 + coarse_local]
                * coarse_value[coarse_base + coarse_local]
            )
        fine_value[fine_base + fine_local] += result


@wp.kernel(enable_backward=False)
def _prolong_add_owned_6_3(
    aggregate: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
):
    fine_node = wp.tid()
    fine_base = fine_node * 6
    coarse_base = aggregate[fine_node] * 3
    for fine_local in range(6):
        result = wp.float64(0.0)
        for coarse_local in range(3):
            result += (
                prolongation_blocks[(fine_base + fine_local) * 3 + coarse_local]
                * coarse_value[coarse_base + coarse_local]
            )
        fine_value[fine_base + fine_local] += result


@wp.kernel(enable_backward=False)
def _prolong_add_owned_6_6(
    aggregate: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
):
    fine_node = wp.tid()
    fine_base = fine_node * 6
    coarse_base = aggregate[fine_node] * 6
    for fine_local in range(6):
        result = wp.float64(0.0)
        for coarse_local in range(6):
            result += (
                prolongation_blocks[(fine_base + fine_local) * 6 + coarse_local]
                * coarse_value[coarse_base + coarse_local]
            )
        fine_value[fine_base + fine_local] += result


_ZERO_KERNELS = {3: _zero_start_block_jacobi_3, 6: _zero_start_block_jacobi_6}
_RESIDUAL_KERNELS = {3: _block_csr_residual_3, 6: _block_csr_residual_6}
_SMOOTH_KERNELS = {3: _fused_block_jacobi_3, 6: _fused_block_jacobi_6}
_RESTRICT_KERNELS = {
    (3, 3): _restrict_owned_3_3,
    (3, 6): _restrict_owned_3_6,
    (6, 3): _restrict_owned_6_3,
    (6, 6): _restrict_owned_6_6,
}
_PROLONG_KERNELS = {
    (3, 3): _prolong_add_owned_3_3,
    (3, 6): _prolong_add_owned_3_6,
    (6, 3): _prolong_add_owned_6_3,
    (6, 6): _prolong_add_owned_6_6,
}


@dataclasses.dataclass(frozen=True, slots=True)
class _LevelLaunchRoute:
    """Prevalidated specialized kernels for one noncoarsest level."""

    zero: object
    residual: object
    smooth: object
    restrict: object
    prolong: object


@dataclasses.dataclass(frozen=True, slots=True)
class WarpFusedVCyclePhysicalWork:
    """Immutable physical-work evidence for one fused V-cycle schedule."""

    hierarchy_sha256: str
    schedule_sha256: str
    rhs_sha256: str
    result_sha256: str
    matrix_block_products_executed: int
    matrix_block_products_elided_zero_start: int
    zero_start_block_solves: int
    fused_smoother_block_solves: int
    matrix_kernel_launches: int
    scheduled_kernel_launches: int
    content_sha256: str

    def __post_init__(self) -> None:
        for name in ("hierarchy_sha256", "schedule_sha256", "rhs_sha256", "result_sha256", "content_sha256"):
            _require_sha256(getattr(self, name), name=name)
        integer_fields = (
            "matrix_block_products_executed",
            "matrix_block_products_elided_zero_start",
            "zero_start_block_solves",
            "fused_smoother_block_solves",
            "matrix_kernel_launches",
            "scheduled_kernel_launches",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in integer_fields):
            raise ValueError("physical-work counts must be non-negative built-in integers")
        if self.scheduled_kernel_launches < 3:
            raise ValueError("a fused V-cycle must schedule at least input, coarse, and output kernels")
        expected = _hash_parts(
            "warp-fused-v-cycle-physical-work-v1",
            tuple(
                (field.name, getattr(self, field.name))
                for field in dataclasses.fields(self)
                if field.name != "content_sha256"
            ),
        )
        if self.content_sha256 != expected:
            raise ValueError("physical-work content_sha256 does not bind its exact fields")


@dataclasses.dataclass(frozen=True, slots=True)
class WarpFusedVCycleRecord:
    """Synchronized immutable algebraic and physical evidence for one cycle."""

    correction: np.ndarray
    work: VCycleWorkRecord
    physical_work: WarpFusedVCyclePhysicalWork
    scheduled_kernel_launches: int
    capture_replay: bool
    schedule_sha256: str
    static_device_content_sha256: str
    device_snapshot_sha256: str
    content_sha256: str
    contract_id: str = CONTRACT_ID
    kernel_version: str = KERNEL_VERSION
    research_only: bool = True
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        correction = _immutable_array(self.correction, np.float64).reshape(-1)
        if correction.size == 0 or not np.isfinite(correction).all():
            raise ValueError("correction must be a finite non-empty vector")
        object.__setattr__(self, "correction", correction)
        if type(self.work) is not VCycleWorkRecord or _work_sha256(self.work) != self.work.content_sha256:
            raise ValueError("work must be an exact untampered VCycleWorkRecord")
        if type(self.physical_work) is not WarpFusedVCyclePhysicalWork:
            raise TypeError("physical_work must be a WarpFusedVCyclePhysicalWork")
        for name in (
            "schedule_sha256",
            "static_device_content_sha256",
            "device_snapshot_sha256",
            "content_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if type(self.scheduled_kernel_launches) is not int or self.scheduled_kernel_launches < 3:
            raise ValueError("scheduled_kernel_launches must be a built-in integer of at least three")
        if type(self.capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        if self.contract_id != CONTRACT_ID or self.kernel_version != KERNEL_VERSION:
            raise ValueError("record contract or kernel version is stale")
        if not self.research_only or self.performance_evidence:
            raise ValueError("this research primitive cannot claim performance evidence")
        result_sha256 = _hash_parts("v-cycle-correction-v1", (("correction", correction),))
        if result_sha256 != self.work.result_sha256:
            raise ValueError("correction bytes do not match the retained result hash")
        physical = self.physical_work
        if (
            physical.hierarchy_sha256 != self.work.hierarchy_sha256
            or physical.schedule_sha256 != self.schedule_sha256
            or physical.rhs_sha256 != self.work.rhs_sha256
            or physical.result_sha256 != self.work.result_sha256
            or physical.scheduled_kernel_launches != self.scheduled_kernel_launches
        ):
            raise ValueError("physical work does not bind the same hierarchy, schedule, input, and output")
        if (
            physical.matrix_block_products_executed + physical.matrix_block_products_elided_zero_start
            != self.work.matrix_block_products
        ):
            raise ValueError("physical and elided matrix work do not recover the canonical V-cycle algebra")
        expected = _hash_parts(
            "warp-fused-v-cycle-result-v1",
            (
                ("device_snapshot_sha256", self.device_snapshot_sha256),
                ("static_device_content_sha256", self.static_device_content_sha256),
                ("schedule_sha256", self.schedule_sha256),
                ("work_sha256", self.work.content_sha256),
                ("physical_work_sha256", physical.content_sha256),
                ("scheduled_kernel_launches", self.scheduled_kernel_launches),
                ("capture_replay", self.capture_replay),
            ),
        )
        if self.content_sha256 != expected:
            raise ValueError("content_sha256 does not bind the complete fused V-cycle record")

    def deterministic_record(self) -> dict[str, object]:
        """Return finite JSON-shaped identity and exact-work evidence."""
        return {
            "contract_id": self.contract_id,
            "kernel_version": self.kernel_version,
            "schedule_version": SCHEDULE_VERSION,
            "schedule_sha256": self.schedule_sha256,
            "static_device_content_sha256": self.static_device_content_sha256,
            "device_snapshot_sha256": self.device_snapshot_sha256,
            "research_only": self.research_only,
            "performance_evidence": self.performance_evidence,
            "capture_replay": self.capture_replay,
            "hierarchy_sha256": self.work.hierarchy_sha256,
            "rhs_sha256": self.work.rhs_sha256,
            "result_sha256": self.work.result_sha256,
            "rhs_count": self.work.rhs_count,
            "level_visits": list(self.work.level_visits),
            "matrix_block_products": self.work.matrix_block_products,
            "smoother_block_solves": self.work.smoother_block_solves,
            "restriction_block_products": self.work.restriction_block_products,
            "prolongation_block_products": self.work.prolongation_block_products,
            "coarsest_factor_solves": self.work.coarsest_factor_solves,
            "matrix_block_products_executed": self.physical_work.matrix_block_products_executed,
            "matrix_block_products_elided_zero_start": self.physical_work.matrix_block_products_elided_zero_start,
            "zero_start_block_solves": self.physical_work.zero_start_block_solves,
            "fused_smoother_block_solves": self.physical_work.fused_smoother_block_solves,
            "matrix_kernel_launches": self.physical_work.matrix_kernel_launches,
            "scheduled_kernel_launches": self.scheduled_kernel_launches,
            "work_sha256": self.work.content_sha256,
            "physical_work_sha256": self.physical_work.content_sha256,
            "content_sha256": self.content_sha256,
        }


class WarpFusedStaticMultigridHierarchy:
    """Launch-fused wrapper around one committed static Warp hierarchy."""

    __slots__ = (
        "_coarse_cholesky",
        "_device",
        "_device_snapshot_sha256",
        "_free_vertices_host",
        "_hierarchy_sha256",
        "_launch_routes",
        "_levels",
        "_n_free",
        "_n_free_dofs",
        "_post_smooth_steps",
        "_pre_smooth_steps",
        "_schedule_sha256",
        "_solver_contract",
        "_source_device_snapshot_sha256",
        "_source_hierarchy",
        "_source_identity",
        "_static_array_objects",
        "_static_array_pointers",
        "_static_device_content_sha256",
        "_static_level_signature",
        "_static_model_sha256",
    )

    def __init__(self, hierarchy: WarpStaticMultigridHierarchy):
        if type(hierarchy) is not WarpStaticMultigridHierarchy:
            raise TypeError("hierarchy must be an exact WarpStaticMultigridHierarchy")
        if (
            type(hierarchy.pre_smooth_steps) is not int
            or type(hierarchy.post_smooth_steps) is not int
            or hierarchy.pre_smooth_steps < 1
            or hierarchy.post_smooth_steps != hierarchy.pre_smooth_steps
        ):
            raise ValueError("the wrapped hierarchy must contain equal positive pre/post smoothing counts")
        routes: list[_LevelLaunchRoute] = []
        shape_rows: list[int] = []
        transfer_paths: list[int] = []
        for level_index, level in enumerate(hierarchy.levels):
            if level.block_size not in SUPPORTED_BLOCK_SIZES:
                raise ValueError(f"level {level_index} block size must be one of {SUPPORTED_BLOCK_SIZES}")
            shape_rows.extend((level.block_row_count, level.block_size, level.stored_block_count))
            if level_index == len(hierarchy.levels) - 1:
                continue
            coarse_block_size = level.coarse_block_size
            if coarse_block_size not in SUPPORTED_BLOCK_SIZES:
                raise ValueError(f"level {level_index} coarse block size must be one of {SUPPORTED_BLOCK_SIZES}")
            if hierarchy.levels[level_index + 1].block_size != coarse_block_size:
                raise ValueError(f"level {level_index} transfer block size is inconsistent with the next level")
            transfer_key = (level.block_size, coarse_block_size)
            transfer_paths.extend(transfer_key)
            routes.append(
                _LevelLaunchRoute(
                    zero=_ZERO_KERNELS[level.block_size],
                    residual=_RESIDUAL_KERNELS[level.block_size],
                    smooth=_SMOOTH_KERNELS[level.block_size],
                    restrict=_RESTRICT_KERNELS[transfer_key],
                    prolong=_PROLONG_KERNELS[transfer_key],
                )
            )
        self._source_hierarchy = hierarchy
        self._source_identity = id(hierarchy)
        self._source_device_snapshot_sha256 = hierarchy.device_snapshot_sha256
        self._device = hierarchy.device
        self._hierarchy_sha256 = hierarchy.hierarchy_sha256
        self._solver_contract = hierarchy.solver_contract
        self._static_model_sha256 = hierarchy.static_model_sha256
        self._free_vertices_host = hierarchy.free_vertices_host
        self._pre_smooth_steps = hierarchy.pre_smooth_steps
        self._post_smooth_steps = hierarchy.post_smooth_steps
        self._n_free = hierarchy.n_free
        self._n_free_dofs = hierarchy.n_free_dofs
        self._levels = hierarchy.levels
        self._coarse_cholesky = hierarchy.coarse_cholesky
        self._static_array_objects = self._current_static_arrays()
        self._static_array_pointers = tuple(int(array.ptr) for array in self._static_array_objects)
        self._static_level_signature = self._current_level_signature()
        self._static_device_content_sha256 = self._read_static_device_content_sha256()
        self._launch_routes = tuple(routes)
        self._schedule_sha256 = _hash_parts(
            "warp-fused-v-cycle-schedule-v1",
            (
                ("source_device_snapshot_sha256", hierarchy.device_snapshot_sha256),
                ("kernel_version", KERNEL_VERSION),
                ("schedule_version", SCHEDULE_VERSION),
                ("pre_smooth_steps", hierarchy.pre_smooth_steps),
                ("post_smooth_steps", hierarchy.post_smooth_steps),
                ("level_shapes", _immutable_array(shape_rows, np.int64)),
                ("transfer_block_paths", _immutable_array(transfer_paths, np.int64)),
                ("noncoarse_result_buffer", "alternate"),
                ("coarsest_result_buffer", "primary"),
                (
                    "scheduled_kernel_launches",
                    3 + len(routes) * (3 + hierarchy.pre_smooth_steps + hierarchy.post_smooth_steps),
                ),
            ),
        )
        self._device_snapshot_sha256 = _hash_parts(
            "warp-fused-static-multigrid-snapshot-v1",
            (
                ("source_device_snapshot_sha256", hierarchy.device_snapshot_sha256),
                ("static_device_content_sha256", self._static_device_content_sha256),
                ("schedule_sha256", self._schedule_sha256),
            ),
        )

    @classmethod
    def from_hierarchy(
        cls,
        hierarchy: StaticMultigridHierarchy,
        *,
        device: str = "cpu",
    ) -> WarpFusedStaticMultigridHierarchy:
        """Upload a CPU hierarchy with the committed path and wrap it."""
        source = WarpStaticMultigridHierarchy.from_hierarchy(hierarchy, device=device)
        return cls(source)

    @classmethod
    def from_device_hierarchy(
        cls,
        hierarchy: WarpStaticMultigridHierarchy,
    ) -> WarpFusedStaticMultigridHierarchy:
        """Wrap an already-uploaded committed static Warp hierarchy."""
        return cls(hierarchy)

    @property
    def source_hierarchy(self) -> WarpStaticMultigridHierarchy:
        """Committed hierarchy that owns every persistent operator array."""
        return self._source_hierarchy

    @property
    def device(self):
        return self._device

    @property
    def hierarchy_sha256(self) -> str:
        return self._hierarchy_sha256

    @property
    def solver_contract(self) -> str:
        return self._solver_contract

    @property
    def static_model_sha256(self) -> str | None:
        return self._static_model_sha256

    @property
    def free_vertices_host(self) -> np.ndarray:
        return self._free_vertices_host

    @property
    def pre_smooth_steps(self) -> int:
        return self._pre_smooth_steps

    @property
    def post_smooth_steps(self) -> int:
        return self._post_smooth_steps

    @property
    def n_free(self) -> int:
        return self._n_free

    @property
    def n_free_dofs(self) -> int:
        return self._n_free_dofs

    @property
    def levels(self):
        return self._levels

    @property
    def coarse_cholesky(self) -> wp.array:
        return self._coarse_cholesky

    @property
    def source_device_snapshot_sha256(self) -> str:
        return self._source_device_snapshot_sha256

    @property
    def schedule_sha256(self) -> str:
        return self._schedule_sha256

    @property
    def device_snapshot_sha256(self) -> str:
        return self._device_snapshot_sha256

    @property
    def static_device_content_sha256(self) -> str:
        """Construction-time digest of every shared static device array."""
        return self._static_device_content_sha256

    @property
    def scheduled_kernel_launches(self) -> int:
        """Exact fixed launch count for one fused V-cycle."""
        noncoarse = len(self.levels) - 1
        return 3 + noncoarse * (3 + self.pre_smooth_steps + self.post_smooth_steps)

    def _current_static_arrays(self) -> tuple[wp.array, ...]:
        arrays = [self._source_hierarchy.coarse_cholesky]
        for level in self._source_hierarchy.levels:
            arrays.extend((level.row_offsets, level.column_indices, level.matrix_values))
            for optional in (
                level.inverse_diagonal,
                level.aggregate,
                level.prolongation_blocks,
                level.member_offsets,
                level.member_fine_nodes,
            ):
                if optional is not None:
                    arrays.append(optional)
        return tuple(arrays)

    def _current_level_signature(self) -> tuple[int | float | None, ...]:
        signature: list[int | float | None] = []
        for level in self._source_hierarchy.levels:
            signature.extend(
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
        return tuple(signature)

    def _read_static_device_content_sha256(self) -> str:
        """Synchronously hash shared static arrays at a diagnostic boundary."""
        parts: list[tuple[str, object]] = [("hierarchy_sha256", self._hierarchy_sha256)]
        for index, array in enumerate(self._static_array_objects):
            host = np.asarray(array.numpy())
            if host.dtype.kind in "fc" and not np.isfinite(host).all():
                raise FloatingPointError("static device hierarchy arrays must remain finite")
            parts.append((f"static_array_{index}", host))
        return _hash_parts("warp-fused-static-device-content-v1", parts)

    def _validate_static_device_content(self) -> None:
        """Fail closed on finite or nonfinite static-array mutation at record."""
        if self._read_static_device_content_sha256() != self._static_device_content_sha256:
            raise RuntimeError("shared static device hierarchy content changed after construction")

    def create_workspace(self) -> WarpFusedVCycleWorkspace:
        """Allocate every reusable input, output, and ping-pong buffer."""
        return WarpFusedVCycleWorkspace(self)

    def _validate_source(self) -> None:
        hierarchy = self._source_hierarchy
        static_arrays = self._current_static_arrays()
        if (
            id(hierarchy) != self._source_identity
            or hierarchy.device_snapshot_sha256 != self._source_device_snapshot_sha256
            or hierarchy.device != self._device
            or hierarchy.hierarchy_sha256 != self._hierarchy_sha256
            or hierarchy.solver_contract != self._solver_contract
            or hierarchy.static_model_sha256 != self._static_model_sha256
            or hierarchy.free_vertices_host is not self._free_vertices_host
            or hierarchy.pre_smooth_steps != self._pre_smooth_steps
            or hierarchy.post_smooth_steps != self._post_smooth_steps
            or hierarchy.n_free != self._n_free
            or hierarchy.n_free_dofs != self._n_free_dofs
            or hierarchy.levels is not self._levels
            or hierarchy.coarse_cholesky is not self._coarse_cholesky
            or len(self._levels) != len(self._launch_routes) + 1
            or self._current_level_signature() != self._static_level_signature
            or len(static_arrays) != len(self._static_array_objects)
            or any(
                actual is not expected or int(actual.ptr) != pointer
                for actual, expected, pointer in zip(
                    static_arrays,
                    self._static_array_objects,
                    self._static_array_pointers,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("the wrapped static Warp hierarchy identity changed")

    def _validate_fine_vector(self, vector: wp.array[wp.vec3d], *, name: str) -> None:
        if vector.device != self.device or vector.dtype != wp.vec3d or vector.shape != (self.n_free,):
            raise ValueError(f"{name} must be a vec3d array of shape ({self.n_free},) on {self.device}")

    def _validate_workspace(self, workspace: WarpFusedVCycleWorkspace) -> None:
        if type(workspace) is not WarpFusedVCycleWorkspace or workspace.hierarchy is not self:
            raise ValueError("workspace belongs to a different fused device hierarchy")
        if (
            workspace._hierarchy_identity != id(self)
            or workspace._hierarchy_sha256 != self.hierarchy_sha256
            or workspace._schedule_sha256 != self.schedule_sha256
            or workspace._device_snapshot_sha256 != self.device_snapshot_sha256
        ):
            raise RuntimeError("workspace identity or schedule binding changed")
        workspace._validate_persistent_arrays()

    def launch_apply(
        self,
        rhs: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: WarpFusedVCycleWorkspace,
    ) -> None:
        """Launch one allocation-free fixed-shape symmetric V-cycle."""
        self._validate_source()
        self._validate_fine_vector(rhs, name="rhs")
        self._validate_fine_vector(output, name="output")
        self._validate_workspace(workspace)
        wp.launch(_copy_vec3_to_scalar, dim=self.n_free, inputs=[rhs, workspace.level_rhs[0]], device=self.device)
        self._launch_level(0, workspace)
        wp.launch(
            _copy_scalar_to_vec3,
            dim=self.n_free,
            inputs=[workspace._final_level_correction(0), output],
            device=self.device,
        )

    def _launch_smooth(
        self,
        level_index: int,
        rhs: wp.array,
        current: wp.array,
        output: wp.array,
    ) -> None:
        level = self.levels[level_index]
        if current.ptr == output.ptr:
            raise RuntimeError("fused block-Jacobi requires distinct input and output buffers")
        if level.inverse_diagonal is None or level.omega is None:
            raise RuntimeError(f"device level {level_index} is missing its smoother")
        wp.launch(
            self._launch_routes[level_index].smooth,
            dim=level.block_row_count,
            inputs=[
                level.row_offsets,
                level.column_indices,
                level.matrix_values,
                rhs,
                level.inverse_diagonal,
                level.omega,
                current,
                output,
            ],
            device=self.device,
        )

    def _launch_level(self, level_index: int, workspace: WarpFusedVCycleWorkspace) -> None:
        level = self.levels[level_index]
        rhs = workspace.level_rhs[level_index]
        primary = workspace.level_correction[level_index]
        if level_index == len(self.levels) - 1:
            wp.launch(
                _solve_coarsest_cholesky,
                dim=1,
                inputs=[self.coarse_cholesky, level.scalar_size, rhs, workspace.coarse_intermediate, primary],
                device=self.device,
            )
            return
        if (
            level.inverse_diagonal is None
            or level.omega is None
            or level.aggregate is None
            or level.prolongation_blocks is None
            or level.member_offsets is None
            or level.member_fine_nodes is None
            or level.coarse_node_count is None
        ):
            raise RuntimeError(f"device level {level_index} is missing non-coarsest arrays")

        route = self._launch_routes[level_index]
        alternate = workspace.level_correction_alt[level_index]
        wp.launch(
            route.zero,
            dim=level.block_row_count,
            inputs=[rhs, level.inverse_diagonal, level.omega, primary],
            device=self.device,
        )
        active = primary
        inactive = alternate
        for _ in range(1, self.pre_smooth_steps):
            self._launch_smooth(level_index, rhs, active, inactive)
            active, inactive = inactive, active

        wp.launch(
            route.residual,
            dim=level.block_row_count,
            inputs=[
                level.row_offsets,
                level.column_indices,
                level.matrix_values,
                rhs,
                active,
                workspace.level_residual[level_index],
            ],
            device=self.device,
        )
        wp.launch(
            route.restrict,
            dim=level.coarse_node_count,
            inputs=[
                level.member_offsets,
                level.member_fine_nodes,
                level.prolongation_blocks,
                workspace.level_residual[level_index],
                workspace.level_rhs[level_index + 1],
            ],
            device=self.device,
        )
        self._launch_level(level_index + 1, workspace)
        wp.launch(
            route.prolong,
            dim=level.block_row_count,
            inputs=[
                level.aggregate,
                level.prolongation_blocks,
                workspace._final_level_correction(level_index + 1),
                active,
            ],
            device=self.device,
        )
        for _ in range(self.post_smooth_steps):
            self._launch_smooth(level_index, rhs, active, inactive)
            active, inactive = inactive, active
        if active.ptr != alternate.ptr:
            raise RuntimeError("symmetric fused schedule did not finish in its fixed alternate buffer")


class WarpFusedVCycleWorkspace:
    """Persistent buffers for one fused V-cycle application."""

    __slots__ = (
        "_device_snapshot_sha256",
        "_hierarchy_identity",
        "_hierarchy_sha256",
        "_persistent_arrays",
        "_persistent_pointers",
        "_schedule_sha256",
        "coarse_intermediate",
        "correction",
        "hierarchy",
        "level_correction",
        "level_correction_alt",
        "level_residual",
        "level_rhs",
        "rhs",
    )

    def __init__(self, hierarchy: WarpFusedStaticMultigridHierarchy):
        if type(hierarchy) is not WarpFusedStaticMultigridHierarchy:
            raise TypeError("hierarchy must be an exact WarpFusedStaticMultigridHierarchy")
        self.hierarchy = hierarchy
        self._hierarchy_identity = id(hierarchy)
        self._hierarchy_sha256 = hierarchy.hierarchy_sha256
        self._schedule_sha256 = hierarchy.schedule_sha256
        self._device_snapshot_sha256 = hierarchy.device_snapshot_sha256
        self.rhs = wp.empty(hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        self.correction = wp.empty(hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        self.level_rhs = tuple(
            wp.empty(level.scalar_size, dtype=wp.float64, device=hierarchy.device) for level in hierarchy.levels
        )
        self.level_correction = tuple(
            wp.empty(level.scalar_size, dtype=wp.float64, device=hierarchy.device) for level in hierarchy.levels
        )
        self.level_correction_alt = tuple(
            wp.empty(level.scalar_size, dtype=wp.float64, device=hierarchy.device) for level in hierarchy.levels[:-1]
        )
        self.level_residual = tuple(
            wp.empty(level.scalar_size, dtype=wp.float64, device=hierarchy.device) for level in hierarchy.levels[:-1]
        )
        self.coarse_intermediate = wp.empty(
            hierarchy.levels[-1].scalar_size,
            dtype=wp.float64,
            device=hierarchy.device,
        )
        self._persistent_arrays = self._current_arrays()
        self._persistent_pointers = tuple(int(array.ptr) for array in self._persistent_arrays)

    def _current_arrays(self) -> tuple[wp.array, ...]:
        arrays = (
            (self.rhs, self.correction, self.coarse_intermediate),
            self.level_rhs,
            self.level_correction,
            self.level_correction_alt,
            self.level_residual,
        )
        return tuple(array for group in arrays for array in group)

    def _validate_persistent_arrays(self) -> None:
        current = self._current_arrays()
        if len(current) != len(self._persistent_arrays) or any(
            actual is not expected or int(actual.ptr) != pointer
            for actual, expected, pointer in zip(
                current,
                self._persistent_arrays,
                self._persistent_pointers,
                strict=True,
            )
        ):
            raise RuntimeError("workspace persistent array pointers or topology changed")
        for primary, alternate in zip(self.level_correction[:-1], self.level_correction_alt, strict=True):
            if primary.ptr == alternate.ptr:
                raise RuntimeError("workspace correction ping-pong buffers alias")

    def _final_level_correction(self, level_index: int) -> wp.array:
        if not 0 <= level_index < len(self.level_correction):
            raise IndexError("level_index is outside the fixed workspace")
        if level_index == len(self.level_correction) - 1:
            return self.level_correction[level_index]
        return self.level_correction_alt[level_index]

    @property
    def scheduled_kernel_launches(self) -> int:
        """Exact launch count for one fixed fused V-cycle."""
        return self.hierarchy.scheduled_kernel_launches

    def set_rhs(self, rhs: np.ndarray | Sequence[float]) -> None:
        """Copy one finite host RHS into the persistent fine input buffer."""
        values = np.asarray(rhs, dtype=np.float64)
        allowed = ((self.hierarchy.n_free, 3), (self.hierarchy.n_free_dofs,))
        if values.shape not in allowed:
            raise ValueError(
                f"rhs must have shape ({self.hierarchy.n_free}, 3) or "
                f"({self.hierarchy.n_free_dofs},), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("rhs must contain only finite values")
        self.rhs.assign(values.reshape(-1, 3))

    def launch(self) -> None:
        """Launch the complete allocation-free fused V-cycle schedule."""
        self.hierarchy.launch_apply(self.rhs, self.correction, self)

    def record(self, *, capture_replay: bool = False) -> WarpFusedVCycleRecord:
        """Synchronously materialize immutable result and work evidence."""
        if type(capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.rhs.numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self.correction.numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(rhs, correction, capture_replay=capture_replay)

    def record_internal_application(self, *, capture_replay: bool = False) -> WarpFusedVCycleRecord:
        """Record an external-array apply retained in this workspace's levels."""
        if type(capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.level_rhs[0].numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self._final_level_correction(0).numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(rhs, correction, capture_replay=capture_replay)

    def _record_host_vectors(
        self,
        rhs: np.ndarray,
        correction: np.ndarray,
        *,
        capture_replay: bool,
    ) -> WarpFusedVCycleRecord:
        """Build one fail-closed immutable record after explicit synchronization."""
        self.hierarchy._validate_source()
        self.hierarchy._validate_workspace(self)
        self.hierarchy._validate_static_device_content()
        rhs = np.asarray(rhs, dtype=np.float64).reshape(self.hierarchy.n_free_dofs, 1)
        correction = np.asarray(correction, dtype=np.float64).reshape(-1)
        if not np.isfinite(rhs).all() or not np.isfinite(correction).all():
            raise FloatingPointError("fused V-cycle input and correction must remain finite")
        rhs_frozen = _immutable_array(rhs, np.float64)
        correction_frozen = _immutable_array(correction, np.float64)
        rhs_sha256 = _hash_parts("v-cycle-rhs-v1", (("rhs", rhs_frozen),))
        result_sha256 = _hash_parts("v-cycle-correction-v1", (("correction", correction_frozen),))
        level_visits = tuple(1 for _ in self.hierarchy.levels)
        matrix_products = sum(
            level.stored_block_count * (self.hierarchy.pre_smooth_steps + 1 + self.hierarchy.post_smooth_steps)
            for level in self.hierarchy.levels[:-1]
        )
        smoother_solves = sum(
            level.block_row_count * (self.hierarchy.pre_smooth_steps + self.hierarchy.post_smooth_steps)
            for level in self.hierarchy.levels[:-1]
        )
        restriction_products = sum(level.block_row_count for level in self.hierarchy.levels[:-1])
        record_parts = (
            ("hierarchy_sha256", self.hierarchy.hierarchy_sha256),
            ("rhs_sha256", rhs_sha256),
            ("result_sha256", result_sha256),
            ("rhs_count", 1),
            ("level_visits", _immutable_array(level_visits, np.int64)),
            ("matrix_block_products", matrix_products),
            ("smoother_block_solves", smoother_solves),
            ("restriction_block_products", restriction_products),
            ("prolongation_block_products", restriction_products),
            ("coarsest_factor_solves", 1),
        )
        work_sha256 = _hash_parts("v-cycle-work-record-v1", record_parts)
        work = VCycleWorkRecord(
            hierarchy_sha256=self.hierarchy.hierarchy_sha256,
            rhs_sha256=rhs_sha256,
            result_sha256=result_sha256,
            rhs_count=1,
            level_visits=level_visits,
            matrix_block_products=matrix_products,
            smoother_block_solves=smoother_solves,
            restriction_block_products=restriction_products,
            prolongation_block_products=restriction_products,
            coarsest_factor_solves=1,
            content_sha256=work_sha256,
        )
        elided_matrix_products = sum(level.stored_block_count for level in self.hierarchy.levels[:-1])
        zero_start_solves = sum(level.block_row_count for level in self.hierarchy.levels[:-1])
        fused_smoother_solves = smoother_solves - zero_start_solves
        matrix_kernel_launches = (len(self.hierarchy.levels) - 1) * (
            self.hierarchy.pre_smooth_steps + self.hierarchy.post_smooth_steps
        )
        physical_parts = (
            ("hierarchy_sha256", self.hierarchy.hierarchy_sha256),
            ("schedule_sha256", self.hierarchy.schedule_sha256),
            ("rhs_sha256", rhs_sha256),
            ("result_sha256", result_sha256),
            ("matrix_block_products_executed", matrix_products - elided_matrix_products),
            ("matrix_block_products_elided_zero_start", elided_matrix_products),
            ("zero_start_block_solves", zero_start_solves),
            ("fused_smoother_block_solves", fused_smoother_solves),
            ("matrix_kernel_launches", matrix_kernel_launches),
            ("scheduled_kernel_launches", self.scheduled_kernel_launches),
        )
        physical_sha256 = _hash_parts("warp-fused-v-cycle-physical-work-v1", physical_parts)
        physical_work = WarpFusedVCyclePhysicalWork(
            hierarchy_sha256=self.hierarchy.hierarchy_sha256,
            schedule_sha256=self.hierarchy.schedule_sha256,
            rhs_sha256=rhs_sha256,
            result_sha256=result_sha256,
            matrix_block_products_executed=matrix_products - elided_matrix_products,
            matrix_block_products_elided_zero_start=elided_matrix_products,
            zero_start_block_solves=zero_start_solves,
            fused_smoother_block_solves=fused_smoother_solves,
            matrix_kernel_launches=matrix_kernel_launches,
            scheduled_kernel_launches=self.scheduled_kernel_launches,
            content_sha256=physical_sha256,
        )
        content_sha256 = _hash_parts(
            "warp-fused-v-cycle-result-v1",
            (
                ("device_snapshot_sha256", self.hierarchy.device_snapshot_sha256),
                ("static_device_content_sha256", self.hierarchy.static_device_content_sha256),
                ("schedule_sha256", self.hierarchy.schedule_sha256),
                ("work_sha256", work_sha256),
                ("physical_work_sha256", physical_sha256),
                ("scheduled_kernel_launches", self.scheduled_kernel_launches),
                ("capture_replay", capture_replay),
            ),
        )
        return WarpFusedVCycleRecord(
            correction=correction_frozen,
            work=work,
            physical_work=physical_work,
            scheduled_kernel_launches=self.scheduled_kernel_launches,
            capture_replay=capture_replay,
            schedule_sha256=self.hierarchy.schedule_sha256,
            static_device_content_sha256=self.hierarchy.static_device_content_sha256,
            device_snapshot_sha256=self.hierarchy.device_snapshot_sha256,
            content_sha256=content_sha256,
        )


class WarpFusedStaticMultigridPreconditioner(WarpDevicePreconditioner):
    """Typed PCG boundary for the launch-fused static hierarchy wrapper."""

    def __init__(self, hierarchy: WarpFusedStaticMultigridHierarchy):
        if type(hierarchy) is not WarpFusedStaticMultigridHierarchy:
            raise TypeError("hierarchy must be an exact WarpFusedStaticMultigridHierarchy")
        self.hierarchy = hierarchy
        self.device = hierarchy.device
        self.vector_count = hierarchy.n_free
        self.free_vertices_host = hierarchy.free_vertices_host
        self.static_preconditioner_sha256 = hierarchy.hierarchy_sha256
        self.device_snapshot_sha256 = hierarchy.device_snapshot_sha256
        self.preconditioner_identity = (
            f"static-mg-v-cycle-warp-fused-v1:{hierarchy.hierarchy_sha256}:{hierarchy.schedule_sha256}"
        )
        self.application_kernel_launches = hierarchy.scheduled_kernel_launches

    def create_application_workspace(self) -> WarpFusedVCycleWorkspace:
        """Allocate one independently retained fused V-cycle workspace."""
        return self.hierarchy.create_workspace()

    def launch_apply(
        self,
        rhs: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: object,
    ) -> None:
        """Enqueue one fused V-cycle without synchronization or allocation."""
        if type(workspace) is not WarpFusedVCycleWorkspace:
            raise TypeError("workspace must be an exact WarpFusedVCycleWorkspace")
        self.hierarchy.launch_apply(rhs, output, workspace)

    def record_application(
        self,
        application_index: int,
        workspace: object,
        *,
        capture_replay: bool,
    ) -> WarpDevicePreconditionerApplication:
        """Synchronously retain canonical algebraic work for one application."""
        if type(workspace) is not WarpFusedVCycleWorkspace:
            raise TypeError("workspace must be an exact WarpFusedVCycleWorkspace")
        self.hierarchy._validate_workspace(workspace)
        result = workspace.record_internal_application(capture_replay=capture_replay)
        work = result.work
        return WarpDevicePreconditionerApplication(
            application_index=application_index,
            preconditioner_identity=self.preconditioner_identity,
            static_preconditioner_sha256=self.static_preconditioner_sha256,
            device_snapshot_sha256=self.device_snapshot_sha256,
            input_sha256=work.rhs_sha256,
            output_sha256=work.result_sha256,
            algebraic_work_sha256=work.content_sha256,
            rhs_count=work.rhs_count,
            level_visits=work.level_visits,
            matrix_block_products=work.matrix_block_products,
            smoother_block_solves=work.smoother_block_solves,
            restriction_block_products=work.restriction_block_products,
            prolongation_block_products=work.prolongation_block_products,
            coarsest_factor_solves=work.coarsest_factor_solves,
            scheduled_kernel_launches=result.scheduled_kernel_launches,
            output_finite=True,
            capture_replay=capture_replay,
        )
