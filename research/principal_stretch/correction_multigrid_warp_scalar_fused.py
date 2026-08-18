# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Scalar-row launch-fused device V-cycle over a static Warp hierarchy.

The committed :mod:`correction_multigrid_warp` hierarchy remains the owner of
all operator, transfer, smoother, and coarsest-factor arrays.  This wrapper
retains one GPU owner per scalar row while removing avoidable launches.  The
fine-level vec3d ingress and first zero-start Jacobi sweep share one
scalar-row-owner launch.  Deeper zero-start sweeps still use one launch, and
every later sweep uses a scalar-row CSR residual followed by an out-of-place
scalar-row Jacobi update.  Corrections ping-pong between fixed A/B buffers so
no sweep reads values that another thread is overwriting.

For ``n`` noncoarsest levels, ``p`` pre-sweeps, ``q`` post-sweeps, and a
construction-bound CUDA terminal-fusion indicator ``f``, the scalar core
schedules exactly
``2 + n * (2 + 2*p + 2*q) - int(n > 0) - 2*f`` kernels.  The one-block CUDA
route makes ``f == 1`` by replacing the deepest restriction, ordered coarse
solve, and deepest prolongation launches with one launch.  CPU, oversized,
unsupported, and coarsest-only hierarchies retain the exact legacy route with
``f == 0``.
The standalone public adapter adds one scalar-to-vec3 publication kernel.
With the required symmetric ``p == q >= 1``, every noncoarsest result finishes
in B and the coarsest result finishes in A.  The launch path allocates no
device arrays, performs no synchronization or device readback, and is
CUDA-graph capturable after warm-up.  Records retain both canonical V-cycle
work and physical work, including the one algebraic ``A*0`` traversal elided
at each noncoarsest level.  This is research-only diagnostic evidence, not a
performance claim.
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
    _prolong_add_owned_rows,
    _restrict_owned_rows,
    _solve_coarsest_cholesky,
)

KERNEL_VERSION = "mg-vbd-warp-static-v-cycle-scalar-fused-v5"
CONTRACT_ID = "spectral-free-multiplicative-graph-vbd-warp-static-scalar-fused-v1"
SCHEDULE_VERSION = "scalar-core-terminal-fused-seeded-root-publication-routes-v7"
PUBLICATION_VERSION = "scalar-fused-v-cycle-publication-routes-v2"
STANDALONE_PUBLICATION_ROUTE = "standalone-scalar-to-vec3-kernel"
EXTERNAL_SHARED_PUBLICATION_ROUTE = "external-shared-owner-scalar-to-vec3"
ROOT_INGRESS_INTERNAL_ROUTE = "internal-fused-vec3d-scalar-zero-start"
ROOT_INGRESS_EXTERNAL_SHARED_ROUTE = "external-shared-producer-scalar-zero-start"
ROOT_INGRESS_COARSE_COPY_ROUTE = "coarsest-copy-vec3d-to-scalar"
PHYSICAL_EXECUTION_AUTHENTICATION = "schema-validated-not-launch-authenticated-v1"
TERMINAL_FUSION_VERSION = "terminal-restrict-ordered-cholesky-prolong-v1"
TERMINAL_FUSION_CUDA_ROUTE = "cuda-one-block-terminal-restrict-ordered-cholesky-prolong-v1"
TERMINAL_FUSION_CPU_FALLBACK_ROUTE = "cpu-legacy-three-launch-terminal-restrict-coarse-prolong-v1"
TERMINAL_FUSION_OVERSIZE_FALLBACK_ROUTE = "cuda-oversize-legacy-three-launch-terminal-v1"
TERMINAL_FUSION_UNSUPPORTED_FALLBACK_ROUTE = "cuda-unsupported-legacy-three-launch-terminal-v1"
TERMINAL_FUSION_COARSEST_ONLY_ROUTE = "coarsest-only-copy-and-ordered-solve-v1"
_SCHEMA_ROUTE_STANDALONE = "standalone"
_SCHEMA_ROUTE_CORE = "core"
_SCHEMA_ROUTE_SEEDED_CORE = "seeded-core"
SUPPORTED_BLOCK_SIZES = (3, 6)

# Keep a visible marker so CUDA test output identifies the exact compiled path.
print(f"[kernels] version: {KERNEL_VERSION}")


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


def _schema_route_claim(hierarchy: object, route: str) -> dict[str, object]:
    """Derive one internally coherent route claim for diagnostic serialization."""
    if route == _SCHEMA_ROUTE_STANDALONE:
        schedule_sha256 = hierarchy.schedule_sha256
        device_snapshot_sha256 = hierarchy.device_snapshot_sha256
        core_kernel_launches = hierarchy.core_kernel_launches
        publication_kernel_launches = 1
        publication_route = STANDALONE_PUBLICATION_ROUTE
        seeded_root = False
    elif route == _SCHEMA_ROUTE_CORE:
        schedule_sha256 = hierarchy.core_schedule_sha256
        device_snapshot_sha256 = hierarchy.core_device_snapshot_sha256
        core_kernel_launches = hierarchy.core_kernel_launches
        publication_kernel_launches = 0
        publication_route = EXTERNAL_SHARED_PUBLICATION_ROUTE
        seeded_root = False
    elif route == _SCHEMA_ROUTE_SEEDED_CORE:
        schedule_sha256 = hierarchy.seeded_core_schedule_sha256
        device_snapshot_sha256 = hierarchy.seeded_core_device_snapshot_sha256
        core_kernel_launches = hierarchy.seeded_core_kernel_launches
        publication_kernel_launches = 0
        publication_route = EXTERNAL_SHARED_PUBLICATION_ROUTE
        seeded_root = hierarchy.supports_seeded_root_zero_start
    else:
        raise ValueError("schema route is outside the fixed scalar-fused schedule")
    if hierarchy.supports_seeded_root_zero_start:
        root_ingress_route = ROOT_INGRESS_EXTERNAL_SHARED_ROUTE if seeded_root else ROOT_INGRESS_INTERNAL_ROUTE
        root_ingress_kernel_launches = 0 if seeded_root else 1
    else:
        root_ingress_route = ROOT_INGRESS_COARSE_COPY_ROUTE
        root_ingress_kernel_launches = 1
    return {
        "schedule_sha256": schedule_sha256,
        "device_snapshot_sha256": device_snapshot_sha256,
        "core_kernel_launches": core_kernel_launches,
        "publication_kernel_launches": publication_kernel_launches,
        "publication_route": publication_route,
        "root_ingress_route": root_ingress_route,
        "root_ingress_kernel_launches": root_ingress_kernel_launches,
        "scheduled_kernel_launches": core_kernel_launches + publication_kernel_launches,
    }


@wp.kernel(enable_backward=False)
def _zero_start_scalar_jacobi(
    rhs: wp.array[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    block_size: int,
    omega: wp.float64,
    output: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    block_row = scalar_row // block_size
    local_row = scalar_row - block_row * block_size
    block_base = block_row * block_size * block_size + local_row * block_size
    rhs_base = block_row * block_size
    value = wp.float64(0.0)
    for local_column in range(block_size):
        value += inverse_diagonal[block_base + local_column] * rhs[rhs_base + local_column]
    output[scalar_row] = omega * value


@wp.kernel(enable_backward=False)
def _fused_root_ingress_zero_start_scalar_jacobi(
    external_rhs: wp.array[wp.vec3d],
    inverse_diagonal: wp.array[wp.float64],
    omega: wp.float64,
    scalar_rhs: wp.array[wp.float64],
    output: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    block_row = scalar_row // 3
    local_row = scalar_row - block_row * 3
    block_base = block_row * 9 + local_row * 3
    rhs_value = external_rhs[block_row]
    scalar_rhs[scalar_row] = rhs_value[local_row]
    value = wp.float64(0.0)
    for local_column in range(3):
        value += inverse_diagonal[block_base + local_column] * rhs_value[local_column]
    output[scalar_row] = omega * value


@wp.kernel(enable_backward=False)
def _scalar_csr_residual(
    row_offsets: wp.array[wp.int32],
    column_indices: wp.array[wp.int32],
    values: wp.array[wp.float64],
    block_size: int,
    rhs: wp.array[wp.float64],
    vector: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    block_row = scalar_row // block_size
    local_row = scalar_row - block_row * block_size
    product = wp.float64(0.0)
    for entry in range(row_offsets[block_row], row_offsets[block_row + 1]):
        block_column = column_indices[entry]
        value_base = (entry * block_size + local_row) * block_size
        vector_base = block_column * block_size
        for local_column in range(block_size):
            product += values[value_base + local_column] * vector[vector_base + local_column]
    residual[scalar_row] = rhs[scalar_row] - product


@wp.kernel(enable_backward=False)
def _out_of_place_scalar_jacobi(
    residual: wp.array[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    block_size: int,
    omega: wp.float64,
    current: wp.array[wp.float64],
    output: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    block_row = scalar_row // block_size
    local_row = scalar_row - block_row * block_size
    block_base = block_row * block_size * block_size + local_row * block_size
    residual_base = block_row * block_size
    value = wp.float64(0.0)
    for local_column in range(block_size):
        value += inverse_diagonal[block_base + local_column] * residual[residual_base + local_column]
    output[scalar_row] = current[scalar_row] + omega * value


@wp.kernel(enable_backward=False)
def _terminal_restrict_ordered_solve_prolong(
    member_offsets: wp.array[wp.int32],
    member_fine_nodes: wp.array[wp.int32],
    aggregate: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    fine_block_size: int,
    coarse_block_size: int,
    fine_scalar_size: int,
    fine_residual: wp.array[wp.float64],
    coarse_scalar_size: int,
    coarse_rhs: wp.array[wp.float64],
    lower: wp.array[wp.float64],
    intermediate: wp.array[wp.float64],
    coarse_solution: wp.array[wp.float64],
    fine_correction: wp.array[wp.float64],
):
    """Fuse the terminal transfer pair around one exactly ordered solve."""
    scalar_row = wp.tid()
    if scalar_row < coarse_scalar_size:
        coarse_node = scalar_row // coarse_block_size
        coarse_local = scalar_row - coarse_node * coarse_block_size
        value = wp.float64(0.0)
        for cursor in range(member_offsets[coarse_node], member_offsets[coarse_node + 1]):
            fine_node = member_fine_nodes[cursor]
            fine_base = fine_node * fine_block_size
            for fine_local in range(fine_block_size):
                block_entry = (fine_base + fine_local) * coarse_block_size + coarse_local
                value += prolongation_blocks[block_entry] * fine_residual[fine_base + fine_local]
        coarse_rhs[scalar_row] = value

    restriction_done = int(0)
    if scalar_row == 0:
        restriction_done = 1
    restriction_tile = wp.tile_from_thread(
        shape=(1,),
        value=restriction_done,
        thread_idx=0,
        storage="shared",
    )
    restriction_done = wp.tile_extract(restriction_tile, 0)

    solve_done = int(0)
    if scalar_row == 0 and restriction_done != 0:
        for row in range(coarse_scalar_size):
            value = coarse_rhs[row]
            for column in range(row):
                value -= lower[row * coarse_scalar_size + column] * intermediate[column]
            intermediate[row] = value / lower[row * coarse_scalar_size + row]
        cursor = int(0)
        while cursor < coarse_scalar_size:
            row = coarse_scalar_size - cursor - 1
            value = intermediate[row]
            for column in range(row + 1, coarse_scalar_size):
                value -= lower[column * coarse_scalar_size + row] * coarse_solution[column]
            coarse_solution[row] = value / lower[row * coarse_scalar_size + row]
            cursor += 1
        solve_done = 1

    solve_tile = wp.tile_from_thread(
        shape=(1,),
        value=solve_done,
        thread_idx=0,
        storage="shared",
    )
    solve_done = wp.tile_extract(solve_tile, 0)

    if scalar_row < fine_scalar_size and solve_done != 0:
        fine_node = scalar_row // fine_block_size
        coarse_base = aggregate[fine_node] * coarse_block_size
        block_base = scalar_row * coarse_block_size
        value = wp.float64(0.0)
        for coarse_local in range(coarse_block_size):
            value += prolongation_blocks[block_base + coarse_local] * coarse_solution[coarse_base + coarse_local]
        fine_correction[scalar_row] += value


@dataclasses.dataclass(frozen=True, slots=True)
class WarpScalarFusedVCyclePhysicalWork:
    """Immutable physical-work evidence for one scalar-fused schedule."""

    hierarchy_sha256: str
    schedule_sha256: str
    rhs_sha256: str
    result_sha256: str
    matrix_block_products_executed: int
    matrix_block_products_elided_zero_start: int
    zero_start_block_solves: int
    noncoarse_level_count: int
    terminal_fusion_kernel_launches: int
    terminal_level_index: int
    terminal_block_dim: int
    terminal_collective_count: int
    terminal_owner_thread: int
    terminal_fusion_version: str
    terminal_fusion_route: str
    root_ingress_zero_start_fusions: int
    root_ingress_route: str
    root_ingress_kernel_launches: int
    out_of_place_jacobi_block_solves: int
    matrix_kernel_launches: int
    jacobi_kernel_launches: int
    core_kernel_launches: int
    publication_kernel_launches: int
    publication_version: str
    publication_route: str
    scheduled_kernel_launches: int
    content_sha256: str
    physical_execution_authentication: str = PHYSICAL_EXECUTION_AUTHENTICATION
    solver_issued_authentication: bool = False
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        for name in ("hierarchy_sha256", "schedule_sha256", "rhs_sha256", "result_sha256", "content_sha256"):
            _require_sha256(getattr(self, name), name=name)
        integer_fields = (
            "matrix_block_products_executed",
            "matrix_block_products_elided_zero_start",
            "zero_start_block_solves",
            "noncoarse_level_count",
            "terminal_fusion_kernel_launches",
            "terminal_block_dim",
            "terminal_collective_count",
            "root_ingress_zero_start_fusions",
            "root_ingress_kernel_launches",
            "out_of_place_jacobi_block_solves",
            "matrix_kernel_launches",
            "jacobi_kernel_launches",
            "core_kernel_launches",
            "publication_kernel_launches",
            "scheduled_kernel_launches",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in integer_fields):
            raise ValueError("physical-work counts must be non-negative built-in integers")
        if self.core_kernel_launches < 2 or self.scheduled_kernel_launches < 2:
            raise ValueError("a scalar-fused V-cycle core must schedule at least input and coarse kernels")
        noncoarse = self.noncoarse_level_count
        if type(self.terminal_level_index) is not int or type(self.terminal_owner_thread) is not int:
            raise TypeError("terminal level and owner fields must be built-in integers")
        if type(self.terminal_fusion_version) is not str or self.terminal_fusion_version != TERMINAL_FUSION_VERSION:
            raise ValueError("terminal fusion version is stale")
        if type(self.terminal_fusion_route) is not str:
            raise TypeError("terminal_fusion_route must be a built-in string")
        if self.terminal_fusion_route == TERMINAL_FUSION_CUDA_ROUTE:
            if (
                noncoarse < 1
                or self.terminal_fusion_kernel_launches != 1
                or self.terminal_level_index != noncoarse - 1
                or self.terminal_block_dim < 32
                or self.terminal_block_dim > 1024
                or self.terminal_block_dim % 32 != 0
                or self.terminal_collective_count != 2
                or self.terminal_owner_thread != 0
            ):
                raise ValueError("CUDA terminal fusion route metadata is inconsistent")
        elif self.terminal_fusion_route in (
            TERMINAL_FUSION_CPU_FALLBACK_ROUTE,
            TERMINAL_FUSION_OVERSIZE_FALLBACK_ROUTE,
            TERMINAL_FUSION_UNSUPPORTED_FALLBACK_ROUTE,
        ):
            if (
                noncoarse < 1
                or self.terminal_fusion_kernel_launches != 0
                or self.terminal_level_index != noncoarse - 1
                or self.terminal_block_dim != 0
                or self.terminal_collective_count != 0
                or self.terminal_owner_thread != -1
            ):
                raise ValueError("legacy terminal fallback route metadata is inconsistent")
        elif self.terminal_fusion_route == TERMINAL_FUSION_COARSEST_ONLY_ROUTE:
            if (
                noncoarse != 0
                or self.terminal_fusion_kernel_launches != 0
                or self.terminal_level_index != -1
                or self.terminal_block_dim != 0
                or self.terminal_collective_count != 0
                or self.terminal_owner_thread != -1
            ):
                raise ValueError("coarsest-only terminal route metadata is inconsistent")
        else:
            raise ValueError("terminal_fusion_route is outside the fixed physical schedule")
        expected_root_fusions = int(noncoarse > 0)
        if self.root_ingress_zero_start_fusions != expected_root_fusions:
            raise ValueError("root ingress fusion count does not match the retained topology")
        if self.matrix_kernel_launches != self.jacobi_kernel_launches:
            raise ValueError("matrix and logical Jacobi launch counts must match")
        if noncoarse == 0:
            if self.matrix_kernel_launches != 0:
                raise ValueError("a coarsest-only topology cannot claim smoother launches")
        elif self.matrix_kernel_launches < 2 * noncoarse or self.matrix_kernel_launches % (2 * noncoarse) != 0:
            raise ValueError("symmetric positive smoothing counts do not match the retained topology")
        if type(self.root_ingress_route) is not str:
            raise TypeError("root_ingress_route must be a built-in string")
        if self.root_ingress_route == ROOT_INGRESS_INTERNAL_ROUTE:
            if noncoarse == 0:
                raise ValueError("internal root ingress requires a noncoarsest root level")
            expected_root_launches = 1
            expected_core_launches = (
                self.matrix_kernel_launches
                + self.jacobi_kernel_launches
                + 2 * noncoarse
                + 1
                - 2 * self.terminal_fusion_kernel_launches
            )
        elif self.root_ingress_route == ROOT_INGRESS_EXTERNAL_SHARED_ROUTE:
            if noncoarse == 0:
                raise ValueError("external root ingress requires a noncoarsest root level")
            expected_root_launches = 0
            expected_core_launches = (
                self.matrix_kernel_launches
                + self.jacobi_kernel_launches
                + 2 * noncoarse
                - 2 * self.terminal_fusion_kernel_launches
            )
        elif self.root_ingress_route == ROOT_INGRESS_COARSE_COPY_ROUTE:
            if noncoarse != 0:
                raise ValueError("coarsest-copy root ingress requires a coarsest-only topology")
            expected_root_launches = 1
            expected_core_launches = 2
        else:
            raise ValueError("root_ingress_route is outside the fixed physical schedule")
        if (
            self.root_ingress_kernel_launches != expected_root_launches
            or self.core_kernel_launches != expected_core_launches
        ):
            raise ValueError("root ingress route, topology, and physical core launch counts disagree")
        if type(self.publication_version) is not str or self.publication_version != PUBLICATION_VERSION:
            raise ValueError("publication_version is stale")
        if type(self.publication_route) is not str:
            raise TypeError("publication_route must be a built-in string")
        if self.publication_route == STANDALONE_PUBLICATION_ROUTE:
            if self.root_ingress_route == ROOT_INGRESS_EXTERNAL_SHARED_ROUTE:
                raise ValueError("standalone publication cannot claim an externally seeded root")
            expected_publication_launches = 1
        elif self.publication_route == EXTERNAL_SHARED_PUBLICATION_ROUTE:
            expected_publication_launches = 0
        else:
            raise ValueError("publication_route is outside the fixed physical schedule")
        if (
            self.publication_kernel_launches != expected_publication_launches
            or self.scheduled_kernel_launches != self.core_kernel_launches + expected_publication_launches
        ):
            raise ValueError("physical publication route and launch counts disagree")
        if (
            self.physical_execution_authentication != PHYSICAL_EXECUTION_AUTHENTICATION
            or self.solver_issued_authentication is not False
            or self.performance_evidence is not False
        ):
            raise ValueError("physical-work evidence must remain schema-validated and unauthenticated")
        expected = _hash_parts(
            "warp-scalar-fused-v-cycle-physical-work-v9",
            tuple(
                (field.name, getattr(self, field.name))
                for field in dataclasses.fields(self)
                if field.name != "content_sha256"
            ),
        )
        if self.content_sha256 != expected:
            raise ValueError("physical-work content_sha256 does not bind its exact fields")


@dataclasses.dataclass(frozen=True, slots=True)
class WarpScalarFusedVCycleRecord:
    """Synchronized immutable algebraic and physical evidence for one cycle."""

    correction: np.ndarray
    work: VCycleWorkRecord
    physical_work: WarpScalarFusedVCyclePhysicalWork
    scheduled_kernel_launches: int
    capture_replay: bool
    schedule_sha256: str
    static_device_content_sha256: str
    device_snapshot_sha256: str
    standalone_schedule_sha256: str
    core_schedule_sha256: str
    seeded_core_schedule_sha256: str
    standalone_device_snapshot_sha256: str
    core_device_snapshot_sha256: str
    seeded_core_device_snapshot_sha256: str
    content_sha256: str
    contract_id: str = CONTRACT_ID
    kernel_version: str = KERNEL_VERSION
    schedule_version: str = SCHEDULE_VERSION
    research_only: bool = True
    physical_execution_authentication: str = PHYSICAL_EXECUTION_AUTHENTICATION
    solver_issued_authentication: bool = False
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        correction = _immutable_array(self.correction, np.float64).reshape(-1)
        if correction.size == 0 or not np.isfinite(correction).all():
            raise ValueError("correction must be a finite non-empty vector")
        object.__setattr__(self, "correction", correction)
        if type(self.work) is not VCycleWorkRecord or _work_sha256(self.work) != self.work.content_sha256:
            raise ValueError("work must be an exact untampered VCycleWorkRecord")
        if type(self.physical_work) is not WarpScalarFusedVCyclePhysicalWork:
            raise TypeError("physical_work must be a WarpScalarFusedVCyclePhysicalWork")
        for name in (
            "schedule_sha256",
            "static_device_content_sha256",
            "device_snapshot_sha256",
            "standalone_schedule_sha256",
            "core_schedule_sha256",
            "seeded_core_schedule_sha256",
            "standalone_device_snapshot_sha256",
            "core_device_snapshot_sha256",
            "seeded_core_device_snapshot_sha256",
            "content_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if type(self.scheduled_kernel_launches) is not int or self.scheduled_kernel_launches < 2:
            raise ValueError("scheduled_kernel_launches must be a built-in integer of at least two")
        if type(self.capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        if (
            self.contract_id != CONTRACT_ID
            or self.kernel_version != KERNEL_VERSION
            or self.schedule_version != SCHEDULE_VERSION
        ):
            raise ValueError("record contract, kernel version, or schedule version is stale")
        if (
            self.research_only is not True
            or self.physical_execution_authentication != PHYSICAL_EXECUTION_AUTHENTICATION
            or self.solver_issued_authentication is not False
            or self.performance_evidence is not False
        ):
            raise ValueError("this research primitive must remain schema-validated and unauthenticated")
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
            or physical.physical_execution_authentication != self.physical_execution_authentication
            or physical.solver_issued_authentication is not self.solver_issued_authentication
            or physical.performance_evidence is not self.performance_evidence
        ):
            raise ValueError("physical work does not bind the same hierarchy, schedule, input, output, and policy")
        if (
            physical.matrix_block_products_executed + physical.matrix_block_products_elided_zero_start
            != self.work.matrix_block_products
        ):
            raise ValueError("physical and elided matrix work do not recover the canonical V-cycle algebra")
        if physical.root_ingress_zero_start_fusions != int(len(self.work.level_visits) > 1):
            raise ValueError("physical work has the wrong root ingress fusion count")
        noncoarse = len(self.work.level_visits) - 1
        if physical.noncoarse_level_count != noncoarse:
            raise ValueError("physical work topology does not match the canonical level visits")
        if physical.publication_route == STANDALONE_PUBLICATION_ROUTE:
            expected_schedule = self.standalone_schedule_sha256
            expected_snapshot = self.standalone_device_snapshot_sha256
        elif physical.root_ingress_route == ROOT_INGRESS_EXTERNAL_SHARED_ROUTE:
            expected_schedule = self.seeded_core_schedule_sha256
            expected_snapshot = self.seeded_core_device_snapshot_sha256
        else:
            expected_schedule = self.core_schedule_sha256
            expected_snapshot = self.core_device_snapshot_sha256
        if self.schedule_sha256 != expected_schedule or self.device_snapshot_sha256 != expected_snapshot:
            raise ValueError("selected schedule and device snapshot do not match the physical route")
        if self.standalone_schedule_sha256 == self.core_schedule_sha256:
            raise ValueError("standalone and core schedule identities must remain distinct")
        if self.standalone_device_snapshot_sha256 == self.core_device_snapshot_sha256:
            raise ValueError("standalone and core device snapshot identities must remain distinct")
        if noncoarse > 0:
            if self.seeded_core_schedule_sha256 in (self.standalone_schedule_sha256, self.core_schedule_sha256):
                raise ValueError("seeded-core schedule identity must remain route-specific")
            if self.seeded_core_device_snapshot_sha256 in (
                self.standalone_device_snapshot_sha256,
                self.core_device_snapshot_sha256,
            ):
                raise ValueError("seeded-core device snapshot identity must remain route-specific")
        elif (
            self.seeded_core_schedule_sha256 != self.core_schedule_sha256
            or self.seeded_core_device_snapshot_sha256 != self.core_device_snapshot_sha256
        ):
            raise ValueError("coarsest-only seeded-core bindings must equal the unchanged core fallback")
        expected = _hash_parts(
            "warp-scalar-fused-v-cycle-result-v9",
            (
                ("contract_id", self.contract_id),
                ("kernel_version", self.kernel_version),
                ("schedule_version", self.schedule_version),
                ("device_snapshot_sha256", self.device_snapshot_sha256),
                ("static_device_content_sha256", self.static_device_content_sha256),
                ("schedule_sha256", self.schedule_sha256),
                ("standalone_schedule_sha256", self.standalone_schedule_sha256),
                ("core_schedule_sha256", self.core_schedule_sha256),
                ("seeded_core_schedule_sha256", self.seeded_core_schedule_sha256),
                ("standalone_device_snapshot_sha256", self.standalone_device_snapshot_sha256),
                ("core_device_snapshot_sha256", self.core_device_snapshot_sha256),
                ("seeded_core_device_snapshot_sha256", self.seeded_core_device_snapshot_sha256),
                ("work_sha256", self.work.content_sha256),
                ("physical_work_sha256", physical.content_sha256),
                ("scheduled_kernel_launches", self.scheduled_kernel_launches),
                ("capture_replay", self.capture_replay),
                ("research_only", self.research_only),
                ("physical_execution_authentication", self.physical_execution_authentication),
                ("solver_issued_authentication", self.solver_issued_authentication),
                ("performance_evidence", self.performance_evidence),
            ),
        )
        if self.content_sha256 != expected:
            raise ValueError("content_sha256 does not bind the complete scalar-fused V-cycle record")

    def deterministic_record(self) -> dict[str, object]:
        """Return finite JSON-shaped, schema-validated diagnostic evidence."""
        physical = dataclasses.replace(self.physical_work)
        record = dataclasses.replace(self, physical_work=physical)
        return {
            "contract_id": record.contract_id,
            "kernel_version": record.kernel_version,
            "schedule_version": record.schedule_version,
            "schedule_sha256": record.schedule_sha256,
            "static_device_content_sha256": record.static_device_content_sha256,
            "device_snapshot_sha256": record.device_snapshot_sha256,
            "standalone_schedule_sha256": record.standalone_schedule_sha256,
            "core_schedule_sha256": record.core_schedule_sha256,
            "seeded_core_schedule_sha256": record.seeded_core_schedule_sha256,
            "standalone_device_snapshot_sha256": record.standalone_device_snapshot_sha256,
            "core_device_snapshot_sha256": record.core_device_snapshot_sha256,
            "seeded_core_device_snapshot_sha256": record.seeded_core_device_snapshot_sha256,
            "research_only": record.research_only,
            "physical_execution_authentication": record.physical_execution_authentication,
            "solver_issued_authentication": record.solver_issued_authentication,
            "performance_evidence": record.performance_evidence,
            "capture_replay": record.capture_replay,
            "hierarchy_sha256": record.work.hierarchy_sha256,
            "rhs_sha256": record.work.rhs_sha256,
            "result_sha256": record.work.result_sha256,
            "rhs_count": record.work.rhs_count,
            "level_visits": list(record.work.level_visits),
            "matrix_block_products": record.work.matrix_block_products,
            "smoother_block_solves": record.work.smoother_block_solves,
            "restriction_block_products": record.work.restriction_block_products,
            "prolongation_block_products": record.work.prolongation_block_products,
            "coarsest_factor_solves": record.work.coarsest_factor_solves,
            "matrix_block_products_executed": physical.matrix_block_products_executed,
            "matrix_block_products_elided_zero_start": physical.matrix_block_products_elided_zero_start,
            "zero_start_block_solves": physical.zero_start_block_solves,
            "noncoarse_level_count": physical.noncoarse_level_count,
            "terminal_fusion_kernel_launches": physical.terminal_fusion_kernel_launches,
            "terminal_level_index": physical.terminal_level_index,
            "terminal_block_dim": physical.terminal_block_dim,
            "terminal_collective_count": physical.terminal_collective_count,
            "terminal_owner_thread": physical.terminal_owner_thread,
            "terminal_fusion_version": physical.terminal_fusion_version,
            "terminal_fusion_route": physical.terminal_fusion_route,
            "root_ingress_zero_start_fusions": physical.root_ingress_zero_start_fusions,
            "root_ingress_route": physical.root_ingress_route,
            "root_ingress_kernel_launches": physical.root_ingress_kernel_launches,
            "out_of_place_jacobi_block_solves": physical.out_of_place_jacobi_block_solves,
            "matrix_kernel_launches": physical.matrix_kernel_launches,
            "jacobi_kernel_launches": physical.jacobi_kernel_launches,
            "core_kernel_launches": physical.core_kernel_launches,
            "publication_kernel_launches": physical.publication_kernel_launches,
            "publication_version": physical.publication_version,
            "publication_route": physical.publication_route,
            "scheduled_kernel_launches": record.scheduled_kernel_launches,
            "work_sha256": record.work.content_sha256,
            "physical_work_sha256": physical.content_sha256,
            "content_sha256": record.content_sha256,
        }


class WarpScalarFusedStaticMultigridHierarchy:
    """Scalar-row launch-fused wrapper around one committed hierarchy."""

    __slots__ = (
        "_coarse_cholesky",
        "_core_device_snapshot_sha256",
        "_core_schedule_sha256",
        "_device",
        "_device_snapshot_sha256",
        "_free_vertices_host",
        "_hierarchy_sha256",
        "_levels",
        "_n_free",
        "_n_free_dofs",
        "_post_smooth_steps",
        "_pre_smooth_steps",
        "_schedule_sha256",
        "_seeded_core_device_snapshot_sha256",
        "_seeded_core_schedule_sha256",
        "_solver_contract",
        "_source_device_snapshot_sha256",
        "_source_hierarchy",
        "_source_identity",
        "_static_array_objects",
        "_static_array_pointers",
        "_static_device_content_sha256",
        "_static_level_signature",
        "_static_model_sha256",
        "_terminal_block_dim",
        "_terminal_collective_count",
        "_terminal_fusion_kernel_launches",
        "_terminal_fusion_route",
        "_terminal_level_index",
        "_terminal_owner_thread",
        "_terminal_route_signature",
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
        root = hierarchy.levels[0]
        if (
            root.block_size != 3
            or root.block_row_count != hierarchy.n_free
            or root.scalar_size != hierarchy.n_free_dofs
        ):
            raise ValueError("the root level must contain exactly one 3-vector block per free vertex")
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
            transfer_paths.extend((level.block_size, coarse_block_size))
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
        terminal_metadata = self._derive_terminal_fusion_metadata()
        (
            self._terminal_fusion_route,
            self._terminal_fusion_kernel_launches,
            self._terminal_level_index,
            self._terminal_block_dim,
            self._terminal_collective_count,
            self._terminal_owner_thread,
        ) = terminal_metadata
        self._terminal_route_signature = terminal_metadata
        self._static_array_objects = self._current_static_arrays()
        self._static_array_pointers = tuple(int(array.ptr) for array in self._static_array_objects)
        self._static_level_signature = self._current_level_signature()
        self._static_device_content_sha256 = self._read_static_device_content_sha256()
        root_ingress_fused = int(len(hierarchy.levels) > 1)
        root_ingress_route = ROOT_INGRESS_INTERNAL_ROUTE if root_ingress_fused else ROOT_INGRESS_COARSE_COPY_ROUTE
        common_schedule_parts = (
            ("source_device_snapshot_sha256", hierarchy.device_snapshot_sha256),
            ("kernel_version", KERNEL_VERSION),
            ("schedule_version", SCHEDULE_VERSION),
            ("owner_parallelism", "one-owner-per-scalar-row"),
            ("pre_smooth_steps", hierarchy.pre_smooth_steps),
            ("post_smooth_steps", hierarchy.post_smooth_steps),
            ("level_shapes", _immutable_array(shape_rows, np.int64)),
            ("transfer_block_paths", _immutable_array(transfer_paths, np.int64)),
            ("root_ingress_route", root_ingress_route),
            ("root_ingress_zero_start_fusions", root_ingress_fused),
            ("root_ingress_kernel_launches", 1),
            ("noncoarse_result_buffer", "B"),
            ("coarsest_result_buffer", "A"),
            ("terminal_fusion_version", TERMINAL_FUSION_VERSION),
            ("terminal_fusion_route", self.terminal_fusion_route),
            ("terminal_fusion_kernel_launches", self.terminal_fusion_kernel_launches),
            ("terminal_level_index", self.terminal_level_index),
            ("terminal_block_dim", self.terminal_block_dim),
            ("terminal_collective_count", self.terminal_collective_count),
            ("terminal_owner_thread", self.terminal_owner_thread),
            ("core_kernel_launches", self.core_kernel_launches),
            ("publication_version", PUBLICATION_VERSION),
        )
        self._core_schedule_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-core-schedule-v5",
            (
                *common_schedule_parts,
                ("publication_route", EXTERNAL_SHARED_PUBLICATION_ROUTE),
                ("publication_kernel_launches", 0),
                ("scheduled_kernel_launches", self.core_kernel_launches),
            ),
        )
        self._schedule_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-schedule-v7",
            (
                *common_schedule_parts,
                ("publication_route", STANDALONE_PUBLICATION_ROUTE),
                ("publication_kernel_launches", 1),
                ("scheduled_kernel_launches", self.scheduled_kernel_launches),
            ),
        )
        self._core_device_snapshot_sha256 = _hash_parts(
            "warp-scalar-fused-static-multigrid-core-snapshot-v5",
            (
                ("source_device_snapshot_sha256", hierarchy.device_snapshot_sha256),
                ("static_device_content_sha256", self._static_device_content_sha256),
                ("core_schedule_sha256", self._core_schedule_sha256),
            ),
        )
        self._device_snapshot_sha256 = _hash_parts(
            "warp-scalar-fused-static-multigrid-snapshot-v7",
            (
                ("source_device_snapshot_sha256", hierarchy.device_snapshot_sha256),
                ("static_device_content_sha256", self._static_device_content_sha256),
                ("schedule_sha256", self._schedule_sha256),
            ),
        )
        if self.supports_seeded_root_zero_start:
            seeded_common_schedule_parts = tuple(
                (name, value)
                for name, value in common_schedule_parts
                if name not in ("root_ingress_route", "root_ingress_kernel_launches", "core_kernel_launches")
            )
            seeded_common_schedule_parts += (
                ("root_ingress_route", ROOT_INGRESS_EXTERNAL_SHARED_ROUTE),
                ("root_ingress_kernel_launches", 0),
                ("core_kernel_launches", self.seeded_core_kernel_launches),
            )
            self._seeded_core_schedule_sha256 = _hash_parts(
                "warp-scalar-fused-v-cycle-seeded-core-schedule-v3",
                (
                    *seeded_common_schedule_parts,
                    ("publication_route", EXTERNAL_SHARED_PUBLICATION_ROUTE),
                    ("publication_kernel_launches", 0),
                    ("scheduled_kernel_launches", self.seeded_core_kernel_launches),
                ),
            )
            self._seeded_core_device_snapshot_sha256 = _hash_parts(
                "warp-scalar-fused-static-multigrid-seeded-core-snapshot-v3",
                (
                    ("source_device_snapshot_sha256", hierarchy.device_snapshot_sha256),
                    ("static_device_content_sha256", self._static_device_content_sha256),
                    ("seeded_core_schedule_sha256", self._seeded_core_schedule_sha256),
                ),
            )
        else:
            self._seeded_core_schedule_sha256 = self._core_schedule_sha256
            self._seeded_core_device_snapshot_sha256 = self._core_device_snapshot_sha256

    def _derive_terminal_fusion_metadata(self) -> tuple[str, int, int, int, int, int]:
        """Derive the immutable terminal route from exact device/topology facts."""
        if len(self._levels) == 1:
            return (TERMINAL_FUSION_COARSEST_ONLY_ROUTE, 0, -1, 0, 0, -1)
        terminal_index = len(self._levels) - 2
        terminal = self._levels[terminal_index]
        coarse = self._levels[-1]
        topology_supported = (
            terminal.member_offsets is not None
            and terminal.member_fine_nodes is not None
            and terminal.aggregate is not None
            and terminal.prolongation_blocks is not None
            and terminal.coarse_block_size is not None
            and terminal.coarse_node_count is not None
            and terminal.coarse_node_count == coarse.block_row_count
            and terminal.coarse_block_size == coarse.block_size
            and terminal.scalar_size > 0
            and coarse.scalar_size > 0
            and int(self._coarse_cholesky.size) == coarse.scalar_size * coarse.scalar_size
        )
        if not topology_supported:
            return (TERMINAL_FUSION_UNSUPPORTED_FALLBACK_ROUTE, 0, terminal_index, 0, 0, -1)
        if not self._device.is_cuda:
            return (TERMINAL_FUSION_CPU_FALLBACK_ROUTE, 0, terminal_index, 0, 0, -1)
        required_threads = max(terminal.scalar_size, coarse.scalar_size)
        block_dim = ((required_threads + 31) // 32) * 32
        if block_dim > 1024:
            return (TERMINAL_FUSION_OVERSIZE_FALLBACK_ROUTE, 0, terminal_index, 0, 0, -1)
        return (TERMINAL_FUSION_CUDA_ROUTE, 1, terminal_index, block_dim, 2, 0)

    @classmethod
    def from_hierarchy(
        cls,
        hierarchy: StaticMultigridHierarchy,
        *,
        device: str = "cpu",
    ) -> WarpScalarFusedStaticMultigridHierarchy:
        """Upload a CPU hierarchy with the committed path and wrap it."""
        source = WarpStaticMultigridHierarchy.from_hierarchy(hierarchy, device=device)
        return cls(source)

    @classmethod
    def from_device_hierarchy(
        cls,
        hierarchy: WarpStaticMultigridHierarchy,
    ) -> WarpScalarFusedStaticMultigridHierarchy:
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
    def core_schedule_sha256(self) -> str:
        return self._core_schedule_sha256

    @property
    def device_snapshot_sha256(self) -> str:
        return self._device_snapshot_sha256

    @property
    def core_device_snapshot_sha256(self) -> str:
        return self._core_device_snapshot_sha256

    @property
    def seeded_core_schedule_sha256(self) -> str:
        """Schedule identity for the externally seeded root tail or fallback."""
        return self._seeded_core_schedule_sha256

    @property
    def seeded_core_device_snapshot_sha256(self) -> str:
        """Static snapshot bound to the externally seeded root tail."""
        return self._seeded_core_device_snapshot_sha256

    @property
    def terminal_fusion_route(self) -> str:
        """Construction-bound terminal execution route."""
        return self._terminal_fusion_route

    @property
    def terminal_fusion_kernel_launches(self) -> int:
        """Number of one-block terminal fusion launches per V-cycle."""
        return self._terminal_fusion_kernel_launches

    @property
    def terminal_level_index(self) -> int:
        """Deepest noncoarsest level, or ``-1`` for coarsest-only."""
        return self._terminal_level_index

    @property
    def terminal_block_dim(self) -> int:
        """CUDA block dimension for terminal fusion, or zero on fallback."""
        return self._terminal_block_dim

    @property
    def terminal_collective_count(self) -> int:
        """Number of unconditional tile broadcast synchronization collectives."""
        return self._terminal_collective_count

    @property
    def terminal_owner_thread(self) -> int:
        """Ordered coarse-solve owner thread, or ``-1`` on fallback."""
        return self._terminal_owner_thread

    @property
    def supports_terminal_fusion(self) -> bool:
        """Whether this exact hierarchy uses the CUDA one-block terminal route."""
        return self._terminal_fusion_kernel_launches == 1

    @property
    def static_device_content_sha256(self) -> str:
        """Construction-time digest of every shared static device array."""
        return self._static_device_content_sha256

    @property
    def scheduled_kernel_launches(self) -> int:
        """Exact fixed launch count for one standalone scalar-fused V-cycle."""
        return self.core_kernel_launches + 1

    @property
    def core_kernel_launches(self) -> int:
        """Exact fixed launch count before scalar-to-vec3 publication."""
        noncoarse = len(self.levels) - 1
        return (
            2
            + noncoarse * (2 + 2 * self.pre_smooth_steps + 2 * self.post_smooth_steps)
            - int(noncoarse > 0)
            - 2 * self.terminal_fusion_kernel_launches
        )

    @property
    def supports_seeded_root_zero_start(self) -> bool:
        """Whether an external 3x3 producer can own the root zero-start sweep."""
        return len(self.levels) > 1

    @property
    def seeded_core_kernel_launches(self) -> int:
        """Physical tail launches after an external root seed, with fallback."""
        return self.core_kernel_launches - int(self.supports_seeded_root_zero_start)

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
        return _hash_parts("warp-scalar-fused-static-device-content-v1", parts)

    def _validate_static_device_content(self) -> None:
        """Fail closed on finite or nonfinite static-array mutation at record."""
        if self._read_static_device_content_sha256() != self._static_device_content_sha256:
            raise RuntimeError("shared static device hierarchy content changed after construction")

    def create_workspace(self) -> WarpScalarFusedVCycleWorkspace:
        """Allocate every reusable input, output, residual, and A/B buffer."""
        return WarpScalarFusedVCycleWorkspace(self)

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
            or self._current_level_signature() != self._static_level_signature
            or self._derive_terminal_fusion_metadata() != self._terminal_route_signature
            or (
                self._terminal_fusion_route,
                self._terminal_fusion_kernel_launches,
                self._terminal_level_index,
                self._terminal_block_dim,
                self._terminal_collective_count,
                self._terminal_owner_thread,
            )
            != self._terminal_route_signature
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

    def _validate_external_fine_vector(
        self,
        vector: object,
        *,
        name: str,
        writable: bool,
    ) -> wp.array:
        """Validate one naturally aligned external vec3 view.

        External solver inputs retain the historical support for zero-stride,
        gapped, and reversed readable views.  Writable views additionally must
        not overlap themselves.  Construction-owned arrays use the stricter
        exact-contiguous validator below.
        """
        if type(vector) is not wp.array:
            raise TypeError(f"{name} must be an exact Warp array")
        if (
            vector.device != self.device
            or vector.dtype is not wp.vec3d
            or vector.ndim != 1
            or vector.shape != (self.n_free,)
            or vector.size != self.n_free
        ):
            raise ValueError(f"{name} has the wrong device, dtype, shape, or size")
        if vector.ptr is None or int(vector.ptr) == 0:
            raise ValueError(f"{name} must have non-null storage")
        element_size = wp.types.type_size_in_bytes(wp.vec3d)
        alignment = min(element_size, 8)
        stride = int(vector.strides[0])
        if int(vector.ptr) % alignment != 0 or stride % alignment != 0:
            raise ValueError(f"{name} storage and stride must be naturally aligned")
        if writable and vector.size > 1 and abs(stride) < element_size:
            raise ValueError(f"{name} writable elements must not overlap")
        return vector

    def _validate_readable_fine_vector(self, vector: object, *, name: str) -> wp.array:
        """Validate one external read-only vec3 view."""
        return self._validate_external_fine_vector(vector, name=name, writable=False)

    def _validate_writable_fine_vector(self, vector: object, *, name: str) -> wp.array:
        """Validate one external writable vec3 view."""
        return self._validate_external_fine_vector(vector, name=name, writable=True)

    @staticmethod
    def _array_memory_span(array: wp.array) -> tuple[int, int]:
        """Return a conservative byte interval for one validated 1-D array."""
        if array.ptr is None:
            return (0, 0)
        if array.size == 0:
            return (int(array.ptr), int(array.ptr))
        final_offset = (array.size - 1) * int(array.strides[0])
        element_size = wp.types.type_size_in_bytes(array.dtype)
        start = int(array.ptr) + min(0, final_offset)
        return (start, int(array.ptr) + max(0, final_offset) + element_size)

    @staticmethod
    def _memory_spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    @classmethod
    def _arrays_overlap(cls, left: wp.array, right: wp.array) -> bool:
        return cls._memory_spans_overlap(cls._array_memory_span(left), cls._array_memory_span(right))

    def _validate_exact_1d_array(
        self,
        array: object,
        *,
        name: str,
        dtype: type,
        size: int,
    ) -> wp.array:
        """Require one naturally aligned contiguous non-null device vector."""
        if type(array) is not wp.array:
            raise TypeError(f"{name} must be an exact Warp array")
        if (
            array.device != self.device
            or array.dtype is not dtype
            or array.ndim != 1
            or array.shape != (size,)
            or array.size != size
        ):
            raise ValueError(f"{name} has the wrong device, dtype, shape, or size")
        element_size = wp.types.type_size_in_bytes(dtype)
        if not array.is_contiguous or array.strides != (element_size,):
            raise ValueError(f"{name} must be a contiguous one-dimensional array")
        if array.ptr is None or int(array.ptr) == 0:
            raise ValueError(f"{name} must have non-null storage")
        alignment = min(element_size, 8)
        if int(array.ptr) % alignment != 0:
            raise ValueError(f"{name} storage is not naturally aligned")
        return array

    def _static_array_specs(self) -> tuple[tuple[str, wp.array, type, int], ...]:
        """Return every immutable source array with its exact launch layout."""
        coarse = self.levels[-1]
        specs: list[tuple[str, wp.array, type, int]] = [
            ("coarse_cholesky", self.coarse_cholesky, wp.float64, coarse.scalar_size * coarse.scalar_size)
        ]
        for level_index, level in enumerate(self.levels):
            specs.extend(
                (
                    (f"level[{level_index}].row_offsets", level.row_offsets, wp.int32, level.block_row_count + 1),
                    (
                        f"level[{level_index}].column_indices",
                        level.column_indices,
                        wp.int32,
                        level.stored_block_count,
                    ),
                    (
                        f"level[{level_index}].matrix_values",
                        level.matrix_values,
                        wp.float64,
                        level.stored_block_count * level.block_size * level.block_size,
                    ),
                )
            )
            if level_index == len(self.levels) - 1:
                continue
            if (
                level.inverse_diagonal is None
                or level.aggregate is None
                or level.prolongation_blocks is None
                or level.member_offsets is None
                or level.member_fine_nodes is None
                or level.coarse_node_count is None
                or level.coarse_block_size is None
            ):
                raise RuntimeError(f"device level {level_index} is missing non-coarsest arrays")
            specs.extend(
                (
                    (
                        f"level[{level_index}].inverse_diagonal",
                        level.inverse_diagonal,
                        wp.float64,
                        level.block_row_count * level.block_size * level.block_size,
                    ),
                    (f"level[{level_index}].aggregate", level.aggregate, wp.int32, level.block_row_count),
                    (
                        f"level[{level_index}].prolongation_blocks",
                        level.prolongation_blocks,
                        wp.float64,
                        level.scalar_size * level.coarse_block_size,
                    ),
                    (
                        f"level[{level_index}].member_offsets",
                        level.member_offsets,
                        wp.int32,
                        level.coarse_node_count + 1,
                    ),
                    (
                        f"level[{level_index}].member_fine_nodes",
                        level.member_fine_nodes,
                        wp.int32,
                        level.block_row_count,
                    ),
                )
            )
        return tuple(specs)

    def _workspace_array_specs(
        self,
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> tuple[tuple[str, wp.array, type, int], ...]:
        """Return every mutable persistent array with its exact layout."""
        specs: list[tuple[str, wp.array, type, int]] = [
            ("workspace.rhs", workspace.rhs, wp.vec3d, self.n_free),
            ("workspace.correction", workspace.correction, wp.vec3d, self.n_free),
            (
                "workspace.coarse_intermediate",
                workspace.coarse_intermediate,
                wp.float64,
                self.levels[-1].scalar_size,
            ),
        ]
        for level_index, level in enumerate(self.levels):
            specs.extend(
                (
                    (f"level_rhs[{level_index}]", workspace.level_rhs[level_index], wp.float64, level.scalar_size),
                    (
                        f"primary correction[{level_index}]",
                        workspace.level_correction[level_index],
                        wp.float64,
                        level.scalar_size,
                    ),
                )
            )
            if level_index != len(self.levels) - 1:
                specs.extend(
                    (
                        (
                            f"alternate correction[{level_index}]",
                            workspace.level_correction_alt[level_index],
                            wp.float64,
                            level.scalar_size,
                        ),
                        (
                            f"level_residual[{level_index}]",
                            workspace.level_residual[level_index],
                            wp.float64,
                            level.scalar_size,
                        ),
                    )
                )
        return tuple(specs)

    def _validate_all_array_layouts_and_aliases(self, workspace: WarpScalarFusedVCycleWorkspace) -> None:
        """Reject every persistent layout or source/output overlap pre-launch."""
        static_specs = self._static_array_specs()
        workspace_specs = self._workspace_array_specs(workspace)
        for name, array, dtype, size in (*static_specs, *workspace_specs):
            self._validate_exact_1d_array(array, name=name, dtype=dtype, size=size)
        for specs, kind in ((static_specs, "static arrays"), (workspace_specs, "workspace arrays")):
            for left_index, (left_name, left, _left_dtype, _left_size) in enumerate(specs):
                for right_name, right, _right_dtype, _right_size in specs[left_index + 1 :]:
                    if self._arrays_overlap(left, right):
                        raise ValueError(f"{kind} {left_name} and {right_name} must not alias")
        for static_name, static, _static_dtype, _static_size in static_specs:
            for workspace_name, workspace_array, _workspace_dtype, _workspace_size in workspace_specs:
                if self._arrays_overlap(static, workspace_array):
                    raise ValueError(f"static {static_name} and mutable {workspace_name} must not alias")

    def _terminal_active_correction(self, workspace: WarpScalarFusedVCycleWorkspace) -> wp.array:
        """Return the deepest active pre-smoothed correction buffer."""
        if self.terminal_level_index < 0:
            raise RuntimeError("a coarsest-only hierarchy has no terminal transfer level")
        if self.pre_smooth_steps % 2 == 1:
            return workspace.level_correction[self.terminal_level_index]
        return workspace.level_correction_alt[self.terminal_level_index]

    def _validate_terminal_fusion_preflight(self, workspace: WarpScalarFusedVCycleWorkspace) -> None:
        """Validate all ten same-kernel arrays before the first fused launch."""
        if not self.supports_terminal_fusion:
            return
        terminal = self.levels[self.terminal_level_index]
        coarse = self.levels[-1]
        if (
            terminal.member_offsets is None
            or terminal.member_fine_nodes is None
            or terminal.aggregate is None
            or terminal.prolongation_blocks is None
            or terminal.coarse_node_count is None
            or terminal.coarse_block_size is None
        ):
            raise RuntimeError("terminal fusion arrays changed after route construction")
        specs = (
            ("terminal.member_offsets", terminal.member_offsets, wp.int32, terminal.coarse_node_count + 1),
            ("terminal.member_fine_nodes", terminal.member_fine_nodes, wp.int32, terminal.block_row_count),
            ("terminal.aggregate", terminal.aggregate, wp.int32, terminal.block_row_count),
            (
                "terminal.prolongation_blocks",
                terminal.prolongation_blocks,
                wp.float64,
                terminal.scalar_size * terminal.coarse_block_size,
            ),
            (
                "terminal.fine_residual",
                workspace.level_residual[self.terminal_level_index],
                wp.float64,
                terminal.scalar_size,
            ),
            ("terminal.coarse_rhs", workspace.level_rhs[-1], wp.float64, coarse.scalar_size),
            (
                "terminal.coarse_cholesky",
                self.coarse_cholesky,
                wp.float64,
                coarse.scalar_size * coarse.scalar_size,
            ),
            ("terminal.coarse_intermediate", workspace.coarse_intermediate, wp.float64, coarse.scalar_size),
            ("terminal.coarse_solution", workspace.level_correction[-1], wp.float64, coarse.scalar_size),
            (
                "terminal.fine_correction",
                self._terminal_active_correction(workspace),
                wp.float64,
                terminal.scalar_size,
            ),
        )
        for name, array, dtype, size in specs:
            self._validate_exact_1d_array(array, name=name, dtype=dtype, size=size)
        for left_index, (left_name, left, _left_dtype, _left_size) in enumerate(specs):
            for right_name, right, _right_dtype, _right_size in specs[left_index + 1 :]:
                if self._arrays_overlap(left, right):
                    raise ValueError(f"terminal fused arrays {left_name} and {right_name} must not alias")

    def _validate_core_launch_aliases(
        self,
        rhs: wp.array[wp.vec3d],
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> None:
        root_rhs = workspace.level_rhs[0]
        root_primary = workspace.level_correction[0]
        root_final = workspace._final_level_correction(0)
        root_outputs = [("root primary correction", root_primary)]
        if root_final is not root_primary:
            root_outputs.append(("root final correction", root_final))
        if self._arrays_overlap(rhs, root_rhs):
            raise ValueError("rhs and root level_rhs must not alias")
        for internal_name, internal in root_outputs:
            if self._arrays_overlap(rhs, internal):
                raise ValueError(f"rhs and {internal_name} must not alias")
        named_root_buffers = (("root level_rhs", root_rhs), *root_outputs)
        for left_index, (left_name, left) in enumerate(named_root_buffers):
            for right_name, right in named_root_buffers[left_index + 1 :]:
                if self._arrays_overlap(left, right):
                    raise ValueError(f"{left_name} and {right_name} must not alias")
        for internal_name, internal, _dtype, _size in self._workspace_array_specs(workspace):
            if internal is not rhs and self._arrays_overlap(rhs, internal):
                raise ValueError(f"rhs and {internal_name} must not alias")
        for static_name, static, _dtype, _size in self._static_array_specs():
            if self._arrays_overlap(rhs, static):
                raise ValueError(f"rhs and static {static_name} must not alias")

    def _validate_publication_aliases(
        self,
        output: wp.array[wp.vec3d],
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> None:
        root_rhs = workspace.level_rhs[0]
        root_primary = workspace.level_correction[0]
        root_final = workspace._final_level_correction(0)
        root_outputs = [root_primary]
        if root_final is not root_primary:
            root_outputs.append(root_final)
        for internal_name, internal in (
            ("root level_rhs", root_rhs),
            *(("root correction", value) for value in root_outputs),
        ):
            if self._arrays_overlap(output, internal):
                raise ValueError(f"output and {internal_name} must not alias")
        for internal_name, internal, _dtype, _size in self._workspace_array_specs(workspace):
            if internal is not output and self._arrays_overlap(output, internal):
                raise ValueError(f"output and {internal_name} must not alias")
        for static_name, static, _dtype, _size in self._static_array_specs():
            if self._arrays_overlap(output, static):
                raise ValueError(f"output and static {static_name} must not alias")

    def _validate_workspace(self, workspace: WarpScalarFusedVCycleWorkspace) -> None:
        if type(workspace) is not WarpScalarFusedVCycleWorkspace or workspace.hierarchy is not self:
            raise ValueError("workspace belongs to a different scalar-fused device hierarchy")
        if (
            workspace._hierarchy_identity != id(self)
            or workspace._hierarchy_sha256 != self.hierarchy_sha256
            or workspace._schedule_sha256 != self.schedule_sha256
            or workspace._core_schedule_sha256 != self.core_schedule_sha256
            or workspace._seeded_core_schedule_sha256 != self.seeded_core_schedule_sha256
            or workspace._device_snapshot_sha256 != self.device_snapshot_sha256
            or workspace._core_device_snapshot_sha256 != self.core_device_snapshot_sha256
            or workspace._seeded_core_device_snapshot_sha256 != self.seeded_core_device_snapshot_sha256
        ):
            raise RuntimeError("workspace identity or schedule binding changed")
        workspace._validate_persistent_arrays()
        self._validate_all_array_layouts_and_aliases(workspace)
        self._validate_terminal_fusion_preflight(workspace)

    def launch_apply(
        self,
        rhs: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> None:
        """Launch one allocation-free fixed-shape symmetric V-cycle.

        ``rhs`` and ``output`` may overlap because the complete ingress launch
        precedes the final egress launch on the same device stream.
        """
        self._validate_source()
        self._validate_readable_fine_vector(rhs, name="rhs")
        self._validate_writable_fine_vector(output, name="output")
        self._validate_workspace(workspace)
        self._validate_core_launch_aliases(rhs, workspace)
        self._validate_publication_aliases(output, workspace)
        self._launch_apply_core(rhs, workspace)
        wp.launch(
            _copy_scalar_to_vec3,
            dim=self.n_free,
            inputs=[workspace._final_level_correction(0), output],
            device=self.device,
        )

    def launch_apply_core(
        self,
        rhs: wp.array[wp.vec3d],
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> None:
        """Launch the fixed scalar core without a vec3 publication adapter."""
        self._validate_source()
        self._validate_readable_fine_vector(rhs, name="rhs")
        self._validate_workspace(workspace)
        self._validate_core_launch_aliases(rhs, workspace)
        self._launch_apply_core(rhs, workspace)

    def root_zero_start_seed_parameters(
        self,
        rhs: wp.array[wp.vec3d],
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> tuple[wp.array[wp.float64], float, wp.array[wp.float64], wp.array[wp.float64]] | None:
        """Validate one core and expose its exact external root-seed buffers.

        A coarsest-only hierarchy returns ``None`` and must use the unchanged
        vec3d-to-scalar copy plus coarse solve fallback.
        """
        self._validate_source()
        self._validate_readable_fine_vector(rhs, name="rhs")
        self._validate_workspace(workspace)
        self._validate_core_launch_aliases(rhs, workspace)
        if not self.supports_seeded_root_zero_start:
            return None
        root = self.levels[0]
        if root.inverse_diagonal is None or root.omega is None:
            raise RuntimeError("device root level is missing its smoother")
        return (
            root.inverse_diagonal,
            float(root.omega),
            workspace.level_rhs[0],
            workspace.level_correction[0],
        )

    def launch_apply_core_seeded_root(
        self,
        rhs: wp.array[wp.vec3d],
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> None:
        """Launch the validated seeded-root tail or coarsest-only fallback."""
        seed = self.root_zero_start_seed_parameters(rhs, workspace)
        if seed is None:
            self._launch_apply_core(rhs, workspace)
        else:
            self._launch_level(0, workspace, root_zero_start_complete=True)

    def _launch_apply_core(
        self,
        rhs: wp.array[wp.vec3d],
        workspace: WarpScalarFusedVCycleWorkspace,
    ) -> None:
        """Enqueue the already-validated fixed scalar core."""
        root = self.levels[0]
        if len(self.levels) > 1:
            if root.inverse_diagonal is None or root.omega is None:
                raise RuntimeError("device root level is missing its smoother")
            wp.launch(
                _fused_root_ingress_zero_start_scalar_jacobi,
                dim=root.scalar_size,
                inputs=[
                    rhs,
                    root.inverse_diagonal,
                    root.omega,
                    workspace.level_rhs[0],
                    workspace.level_correction[0],
                ],
                device=self.device,
            )
            self._launch_level(0, workspace, root_zero_start_complete=True)
        else:
            wp.launch(_copy_vec3_to_scalar, dim=self.n_free, inputs=[rhs, workspace.level_rhs[0]], device=self.device)
            self._launch_level(0, workspace)

    def _launch_residual(
        self,
        level_index: int,
        rhs: wp.array,
        current: wp.array,
        residual: wp.array,
    ) -> None:
        level = self.levels[level_index]
        wp.launch(
            _scalar_csr_residual,
            dim=level.scalar_size,
            inputs=[
                level.row_offsets,
                level.column_indices,
                level.matrix_values,
                level.block_size,
                rhs,
                current,
                residual,
            ],
            device=self.device,
        )

    def _launch_jacobi(
        self,
        level_index: int,
        residual: wp.array,
        current: wp.array,
        output: wp.array,
    ) -> None:
        level = self.levels[level_index]
        if current.ptr == output.ptr:
            raise RuntimeError("scalar Jacobi requires distinct input and output buffers")
        if level.inverse_diagonal is None or level.omega is None:
            raise RuntimeError(f"device level {level_index} is missing its smoother")
        wp.launch(
            _out_of_place_scalar_jacobi,
            dim=level.scalar_size,
            inputs=[
                residual,
                level.inverse_diagonal,
                level.block_size,
                level.omega,
                current,
                output,
            ],
            device=self.device,
        )

    def _launch_level(
        self,
        level_index: int,
        workspace: WarpScalarFusedVCycleWorkspace,
        *,
        root_zero_start_complete: bool = False,
    ) -> None:
        level = self.levels[level_index]
        rhs = workspace.level_rhs[level_index]
        primary = workspace.level_correction[level_index]
        if root_zero_start_complete and level_index != 0:
            raise RuntimeError("only the root level can arrive with its zero-start sweep complete")
        if level_index == len(self.levels) - 1:
            if root_zero_start_complete:
                raise RuntimeError("a coarsest-only hierarchy cannot use the fused root ingress")
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
            or level.coarse_block_size is None
        ):
            raise RuntimeError(f"device level {level_index} is missing non-coarsest arrays")

        alternate = workspace.level_correction_alt[level_index]
        residual = workspace.level_residual[level_index]
        if not root_zero_start_complete:
            wp.launch(
                _zero_start_scalar_jacobi,
                dim=level.scalar_size,
                inputs=[rhs, level.inverse_diagonal, level.block_size, level.omega, primary],
                device=self.device,
            )
        active = primary
        inactive = alternate
        for _ in range(1, self.pre_smooth_steps):
            self._launch_residual(level_index, rhs, active, residual)
            self._launch_jacobi(level_index, residual, active, inactive)
            active, inactive = inactive, active

        self._launch_residual(level_index, rhs, active, residual)
        if self.supports_terminal_fusion and level_index == self.terminal_level_index:
            coarse = self.levels[-1]
            wp.launch(
                _terminal_restrict_ordered_solve_prolong,
                dim=self.terminal_block_dim,
                block_dim=self.terminal_block_dim,
                inputs=[
                    level.member_offsets,
                    level.member_fine_nodes,
                    level.aggregate,
                    level.prolongation_blocks,
                    level.block_size,
                    level.coarse_block_size,
                    level.scalar_size,
                    residual,
                    coarse.scalar_size,
                    workspace.level_rhs[-1],
                    self.coarse_cholesky,
                    workspace.coarse_intermediate,
                    workspace.level_correction[-1],
                    active,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _restrict_owned_rows,
                dim=self.levels[level_index + 1].scalar_size,
                inputs=[
                    level.member_offsets,
                    level.member_fine_nodes,
                    level.prolongation_blocks,
                    level.block_size,
                    level.coarse_block_size,
                    residual,
                    workspace.level_rhs[level_index + 1],
                ],
                device=self.device,
            )
            self._launch_level(level_index + 1, workspace)
            wp.launch(
                _prolong_add_owned_rows,
                dim=level.scalar_size,
                inputs=[
                    level.aggregate,
                    level.prolongation_blocks,
                    level.block_size,
                    level.coarse_block_size,
                    workspace._final_level_correction(level_index + 1),
                    active,
                ],
                device=self.device,
            )
        for _ in range(self.post_smooth_steps):
            self._launch_residual(level_index, rhs, active, residual)
            self._launch_jacobi(level_index, residual, active, inactive)
            active, inactive = inactive, active
        if active.ptr != alternate.ptr:
            raise RuntimeError("symmetric scalar-fused schedule did not finish in its fixed B buffer")


class WarpScalarFusedVCycleWorkspace:
    """Persistent buffers for one scalar-row fused V-cycle application."""

    __slots__ = (
        "_core_device_snapshot_sha256",
        "_core_schedule_sha256",
        "_device_snapshot_sha256",
        "_hierarchy_identity",
        "_hierarchy_sha256",
        "_persistent_arrays",
        "_persistent_pointers",
        "_schedule_sha256",
        "_seeded_core_device_snapshot_sha256",
        "_seeded_core_schedule_sha256",
        "coarse_intermediate",
        "correction",
        "hierarchy",
        "level_correction",
        "level_correction_alt",
        "level_residual",
        "level_rhs",
        "rhs",
    )

    def __init__(self, hierarchy: WarpScalarFusedStaticMultigridHierarchy):
        if type(hierarchy) is not WarpScalarFusedStaticMultigridHierarchy:
            raise TypeError("hierarchy must be an exact WarpScalarFusedStaticMultigridHierarchy")
        self.hierarchy = hierarchy
        self._hierarchy_identity = id(hierarchy)
        self._hierarchy_sha256 = hierarchy.hierarchy_sha256
        self._schedule_sha256 = hierarchy.schedule_sha256
        self._core_schedule_sha256 = hierarchy.core_schedule_sha256
        self._seeded_core_schedule_sha256 = hierarchy.seeded_core_schedule_sha256
        self._device_snapshot_sha256 = hierarchy.device_snapshot_sha256
        self._core_device_snapshot_sha256 = hierarchy.core_device_snapshot_sha256
        self._seeded_core_device_snapshot_sha256 = hierarchy.seeded_core_device_snapshot_sha256
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
                raise RuntimeError("workspace correction A/B buffers alias")

    def _final_level_correction(self, level_index: int) -> wp.array:
        if not 0 <= level_index < len(self.level_correction):
            raise IndexError("level_index is outside the fixed workspace")
        if level_index == len(self.level_correction) - 1:
            return self.level_correction[level_index]
        return self.level_correction_alt[level_index]

    @property
    def scheduled_kernel_launches(self) -> int:
        """Exact launch count for one standalone scalar-fused V-cycle."""
        return self.hierarchy.scheduled_kernel_launches

    @property
    def core_kernel_launches(self) -> int:
        """Exact launch count before scalar-to-vec3 publication."""
        return self.hierarchy.core_kernel_launches

    @property
    def seeded_core_kernel_launches(self) -> int:
        """Exact root-seeded tail count, including coarsest fallback."""
        return self.hierarchy.seeded_core_kernel_launches

    @property
    def final_scalar_correction(self) -> wp.array[wp.float64]:
        """Persistent root scalar result used by either publication route."""
        return self._final_level_correction(0)

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
        """Launch the complete allocation-free scalar-fused schedule."""
        self.hierarchy.launch_apply(self.rhs, self.correction, self)

    def launch_core(self) -> None:
        """Launch only the scalar core, leaving vec3 publication external."""
        self.hierarchy.launch_apply_core(self.rhs, self)

    def launch_seeded_core(self) -> None:
        """Launch the externally seeded tail or the coarsest-only fallback."""
        self.hierarchy.launch_apply_core_seeded_root(self.rhs, self)

    def record(
        self,
        *,
        capture_replay: bool = False,
    ) -> WarpScalarFusedVCycleRecord:
        """Synchronously materialize immutable result and work evidence."""
        if type(capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.rhs.numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self.correction.numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(
            rhs,
            correction,
            capture_replay=capture_replay,
            schema_route=_SCHEMA_ROUTE_STANDALONE,
        )

    def record_internal_application(
        self,
        *,
        capture_replay: bool = False,
    ) -> WarpScalarFusedVCycleRecord:
        """Record a full standalone apply retained in this workspace's levels."""
        if type(capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.level_rhs[0].numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self._final_level_correction(0).numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(
            rhs,
            correction,
            capture_replay=capture_replay,
            schema_route=_SCHEMA_ROUTE_STANDALONE,
        )

    def record_core_application(
        self,
        *,
        capture_replay: bool = False,
    ) -> WarpScalarFusedVCycleRecord:
        """Serialize a core-route schema claim for the retained level buffers."""
        if type(capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.level_rhs[0].numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self._final_level_correction(0).numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(
            rhs,
            correction,
            capture_replay=capture_replay,
            schema_route=_SCHEMA_ROUTE_CORE,
        )

    def record_seeded_core_application(
        self,
        *,
        capture_replay: bool = False,
    ) -> WarpScalarFusedVCycleRecord:
        """Serialize a seeded-core schema claim for the retained buffers."""
        if type(capture_replay) is not bool:
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.level_rhs[0].numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self._final_level_correction(0).numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(
            rhs,
            correction,
            capture_replay=capture_replay,
            schema_route=_SCHEMA_ROUTE_SEEDED_CORE,
        )

    def _record_host_vectors(
        self,
        rhs: np.ndarray,
        correction: np.ndarray,
        *,
        capture_replay: bool,
        schema_route: str,
    ) -> WarpScalarFusedVCycleRecord:
        """Build one fail-closed immutable schema record after synchronization."""
        self.hierarchy._validate_source()
        self.hierarchy._validate_workspace(self)
        self.hierarchy._validate_static_device_content()
        rhs = np.asarray(rhs, dtype=np.float64).reshape(self.hierarchy.n_free_dofs, 1)
        correction = np.asarray(correction, dtype=np.float64).reshape(-1)
        if not np.isfinite(rhs).all() or not np.isfinite(correction).all():
            raise FloatingPointError("scalar-fused V-cycle input and correction must remain finite")
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
        noncoarse_level_count = len(self.hierarchy.levels) - 1
        root_ingress_zero_start_fusions = int(noncoarse_level_count > 0)
        matrix_kernel_launches = noncoarse_level_count * (
            self.hierarchy.pre_smooth_steps + self.hierarchy.post_smooth_steps
        )
        schema_claim = _schema_route_claim(self.hierarchy, schema_route)
        publication_kernel_launches = schema_claim["publication_kernel_launches"]
        publication_route = schema_claim["publication_route"]
        schedule_sha256 = schema_claim["schedule_sha256"]
        device_snapshot_sha256 = schema_claim["device_snapshot_sha256"]
        core_kernel_launches = schema_claim["core_kernel_launches"]
        root_ingress_route = schema_claim["root_ingress_route"]
        root_ingress_kernel_launches = schema_claim["root_ingress_kernel_launches"]
        scheduled_kernel_launches = schema_claim["scheduled_kernel_launches"]
        physical_parts = (
            ("hierarchy_sha256", self.hierarchy.hierarchy_sha256),
            ("schedule_sha256", schedule_sha256),
            ("rhs_sha256", rhs_sha256),
            ("result_sha256", result_sha256),
            ("matrix_block_products_executed", matrix_products - elided_matrix_products),
            ("matrix_block_products_elided_zero_start", elided_matrix_products),
            ("zero_start_block_solves", zero_start_solves),
            ("noncoarse_level_count", noncoarse_level_count),
            ("terminal_fusion_kernel_launches", self.hierarchy.terminal_fusion_kernel_launches),
            ("terminal_level_index", self.hierarchy.terminal_level_index),
            ("terminal_block_dim", self.hierarchy.terminal_block_dim),
            ("terminal_collective_count", self.hierarchy.terminal_collective_count),
            ("terminal_owner_thread", self.hierarchy.terminal_owner_thread),
            ("terminal_fusion_version", TERMINAL_FUSION_VERSION),
            ("terminal_fusion_route", self.hierarchy.terminal_fusion_route),
            ("root_ingress_zero_start_fusions", root_ingress_zero_start_fusions),
            ("root_ingress_route", root_ingress_route),
            ("root_ingress_kernel_launches", root_ingress_kernel_launches),
            ("out_of_place_jacobi_block_solves", smoother_solves - zero_start_solves),
            ("matrix_kernel_launches", matrix_kernel_launches),
            ("jacobi_kernel_launches", matrix_kernel_launches),
            ("core_kernel_launches", core_kernel_launches),
            ("publication_kernel_launches", publication_kernel_launches),
            ("publication_version", PUBLICATION_VERSION),
            ("publication_route", publication_route),
            ("scheduled_kernel_launches", scheduled_kernel_launches),
            ("physical_execution_authentication", PHYSICAL_EXECUTION_AUTHENTICATION),
            ("solver_issued_authentication", False),
            ("performance_evidence", False),
        )
        physical_sha256 = _hash_parts("warp-scalar-fused-v-cycle-physical-work-v9", physical_parts)
        physical_work = WarpScalarFusedVCyclePhysicalWork(
            hierarchy_sha256=self.hierarchy.hierarchy_sha256,
            schedule_sha256=schedule_sha256,
            rhs_sha256=rhs_sha256,
            result_sha256=result_sha256,
            matrix_block_products_executed=matrix_products - elided_matrix_products,
            matrix_block_products_elided_zero_start=elided_matrix_products,
            zero_start_block_solves=zero_start_solves,
            noncoarse_level_count=noncoarse_level_count,
            terminal_fusion_kernel_launches=self.hierarchy.terminal_fusion_kernel_launches,
            terminal_level_index=self.hierarchy.terminal_level_index,
            terminal_block_dim=self.hierarchy.terminal_block_dim,
            terminal_collective_count=self.hierarchy.terminal_collective_count,
            terminal_owner_thread=self.hierarchy.terminal_owner_thread,
            terminal_fusion_version=TERMINAL_FUSION_VERSION,
            terminal_fusion_route=self.hierarchy.terminal_fusion_route,
            root_ingress_zero_start_fusions=root_ingress_zero_start_fusions,
            root_ingress_route=root_ingress_route,
            root_ingress_kernel_launches=root_ingress_kernel_launches,
            out_of_place_jacobi_block_solves=smoother_solves - zero_start_solves,
            matrix_kernel_launches=matrix_kernel_launches,
            jacobi_kernel_launches=matrix_kernel_launches,
            core_kernel_launches=core_kernel_launches,
            publication_kernel_launches=publication_kernel_launches,
            publication_version=PUBLICATION_VERSION,
            publication_route=publication_route,
            scheduled_kernel_launches=scheduled_kernel_launches,
            content_sha256=physical_sha256,
        )
        content_sha256 = _hash_parts(
            "warp-scalar-fused-v-cycle-result-v9",
            (
                ("contract_id", CONTRACT_ID),
                ("kernel_version", KERNEL_VERSION),
                ("schedule_version", SCHEDULE_VERSION),
                ("device_snapshot_sha256", device_snapshot_sha256),
                ("static_device_content_sha256", self.hierarchy.static_device_content_sha256),
                ("schedule_sha256", schedule_sha256),
                ("standalone_schedule_sha256", self.hierarchy.schedule_sha256),
                ("core_schedule_sha256", self.hierarchy.core_schedule_sha256),
                ("seeded_core_schedule_sha256", self.hierarchy.seeded_core_schedule_sha256),
                ("standalone_device_snapshot_sha256", self.hierarchy.device_snapshot_sha256),
                ("core_device_snapshot_sha256", self.hierarchy.core_device_snapshot_sha256),
                ("seeded_core_device_snapshot_sha256", self.hierarchy.seeded_core_device_snapshot_sha256),
                ("work_sha256", work_sha256),
                ("physical_work_sha256", physical_sha256),
                ("scheduled_kernel_launches", scheduled_kernel_launches),
                ("capture_replay", capture_replay),
                ("research_only", True),
                ("physical_execution_authentication", PHYSICAL_EXECUTION_AUTHENTICATION),
                ("solver_issued_authentication", False),
                ("performance_evidence", False),
            ),
        )
        return WarpScalarFusedVCycleRecord(
            correction=correction_frozen,
            work=work,
            physical_work=physical_work,
            scheduled_kernel_launches=scheduled_kernel_launches,
            capture_replay=capture_replay,
            schedule_sha256=schedule_sha256,
            static_device_content_sha256=self.hierarchy.static_device_content_sha256,
            device_snapshot_sha256=device_snapshot_sha256,
            standalone_schedule_sha256=self.hierarchy.schedule_sha256,
            core_schedule_sha256=self.hierarchy.core_schedule_sha256,
            seeded_core_schedule_sha256=self.hierarchy.seeded_core_schedule_sha256,
            standalone_device_snapshot_sha256=self.hierarchy.device_snapshot_sha256,
            core_device_snapshot_sha256=self.hierarchy.core_device_snapshot_sha256,
            seeded_core_device_snapshot_sha256=self.hierarchy.seeded_core_device_snapshot_sha256,
            content_sha256=content_sha256,
        )


class WarpScalarFusedStaticMultigridPreconditioner(WarpDevicePreconditioner):
    """Typed PCG boundary for the scalar-row fused hierarchy wrapper."""

    def __init__(self, hierarchy: WarpScalarFusedStaticMultigridHierarchy):
        if type(hierarchy) is not WarpScalarFusedStaticMultigridHierarchy:
            raise TypeError("hierarchy must be an exact WarpScalarFusedStaticMultigridHierarchy")
        self.hierarchy = hierarchy
        self.device = hierarchy.device
        self.vector_count = hierarchy.n_free
        self.free_vertices_host = hierarchy.free_vertices_host
        self.static_preconditioner_sha256 = hierarchy.hierarchy_sha256
        self.device_snapshot_sha256 = hierarchy.device_snapshot_sha256
        self.preconditioner_identity = (
            f"static-mg-v-cycle-warp-scalar-fused-v1:{hierarchy.hierarchy_sha256}:{hierarchy.schedule_sha256}"
        )
        self.application_kernel_launches = hierarchy.scheduled_kernel_launches

    def create_application_workspace(self) -> WarpScalarFusedVCycleWorkspace:
        """Allocate one independently retained scalar-fused workspace."""
        return self.hierarchy.create_workspace()

    def launch_apply(
        self,
        rhs: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: object,
    ) -> None:
        """Enqueue one scalar-fused V-cycle without synchronization/allocation."""
        if type(workspace) is not WarpScalarFusedVCycleWorkspace:
            raise TypeError("workspace must be an exact WarpScalarFusedVCycleWorkspace")
        self.hierarchy.launch_apply(rhs, output, workspace)

    def record_application(
        self,
        application_index: int,
        workspace: object,
        *,
        capture_replay: bool,
    ) -> WarpDevicePreconditionerApplication:
        """Synchronously retain canonical algebraic work for one application."""
        if type(workspace) is not WarpScalarFusedVCycleWorkspace:
            raise TypeError("workspace must be an exact WarpScalarFusedVCycleWorkspace")
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
