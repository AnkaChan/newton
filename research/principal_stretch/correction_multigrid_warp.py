# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Device-resident symmetric V-cycle for static MG-VBD research.

This module uploads :class:`StaticMultigridHierarchy` without changing its
algebra.  All matrices remain block CSR, each transfer retains one block per
fine graph node, and only the explicitly bounded coarsest Cholesky factor is
dense.  Every kernel has one owner thread per output scalar (except the
single-thread coarsest triangular solve), so restriction and sparse products
use no floating-point atomics and have a deterministic accumulation order.

The workspace owns every temporary array needed by one V-cycle.  Its launch
path performs no allocation, synchronization, or device-to-host transfer and
is suitable for CUDA graph capture after warm-up.  Diagnostic records are an
explicit synchronization point and bind the result and exact algebraic work
to the immutable CPU hierarchy SHA.  This is a research primitive, not an
integrated :class:`newton.solvers.SolverVBD` implementation or performance
claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import warp as wp

from .correction_gpu_warp import WarpDevicePreconditioner, WarpDevicePreconditionerApplication
from .correction_multigrid import (
    SPECTRAL_FREE_CONTRACT,
    StaticMultigridHierarchy,
    VCycleWorkRecord,
)

KERNEL_VERSION = "mg-vbd-warp-static-v-cycle-v1"
CONTRACT_ID = "spectral-free-multiplicative-graph-vbd-warp-static-v1"
MAX_COARSE_SCALAR_SIZE = 96
"""Hard upper bound for the serial device-side coarsest solve."""


def _immutable_array(value: np.ndarray | Iterable[float], dtype: np.dtype | type) -> np.ndarray:
    """Return a C-contiguous array backed by immutable bytes."""
    source = np.array(value, dtype=dtype, order="C", copy=True)
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)


def _hash_parts(tag: str, parts: Iterable[tuple[str, object]]) -> str:
    """Hash typed, length-delimited fields using the CPU V-cycle schema."""
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
        elif value is None:
            add_bytes(b"none")
        else:
            raise TypeError(f"unsupported hash part {name!r}: {type(value).__name__}")
    return digest.hexdigest()


def _as_int32(value: np.ndarray, *, name: str) -> np.ndarray:
    """Narrow a non-negative index array only when every entry is exact."""
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if array.size and (np.any(array < 0) or np.any(array > np.iinfo(np.int32).max)):
        raise ValueError(f"{name} is outside the supported int32 range")
    return np.asarray(array, dtype=np.int32)


@wp.kernel(enable_backward=False)
def _copy_vec3_to_scalar(source: wp.array[wp.vec3d], destination: wp.array[wp.float64]):
    node = wp.tid()
    value = source[node]
    destination[3 * node] = value[0]
    destination[3 * node + 1] = value[1]
    destination[3 * node + 2] = value[2]


@wp.kernel(enable_backward=False)
def _copy_scalar_to_vec3(source: wp.array[wp.float64], destination: wp.array[wp.vec3d]):
    node = wp.tid()
    destination[node] = wp.vec3d(source[3 * node], source[3 * node + 1], source[3 * node + 2])


@wp.kernel(enable_backward=False)
def _clear_scalar(vector: wp.array[wp.float64]):
    row = wp.tid()
    vector[row] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _block_csr_matvec(
    row_offsets: wp.array[wp.int32],
    column_indices: wp.array[wp.int32],
    values: wp.array[wp.float64],
    block_size: int,
    vector: wp.array[wp.float64],
    output: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    block_row = scalar_row // block_size
    local_row = scalar_row - block_row * block_size
    result = wp.float64(0.0)
    for entry in range(row_offsets[block_row], row_offsets[block_row + 1]):
        block_column = column_indices[entry]
        value_base = (entry * block_size + local_row) * block_size
        vector_base = block_column * block_size
        for local_column in range(block_size):
            result += values[value_base + local_column] * vector[vector_base + local_column]
    output[scalar_row] = result


@wp.kernel(enable_backward=False)
def _subtract_product(
    rhs: wp.array[wp.float64],
    product: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    row = wp.tid()
    residual[row] = rhs[row] - product[row]


@wp.kernel(enable_backward=False)
def _jacobi_add(
    residual: wp.array[wp.float64],
    inverse_diagonal: wp.array[wp.float64],
    block_size: int,
    omega: wp.float64,
    correction: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    block_row = scalar_row // block_size
    local_row = scalar_row - block_row * block_size
    block_base = block_row * block_size * block_size + local_row * block_size
    residual_base = block_row * block_size
    value = wp.float64(0.0)
    for local_column in range(block_size):
        value += inverse_diagonal[block_base + local_column] * residual[residual_base + local_column]
    correction[scalar_row] += omega * value


@wp.kernel(enable_backward=False)
def _restrict_owned_rows(
    member_offsets: wp.array[wp.int32],
    member_fine_nodes: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    fine_block_size: int,
    coarse_block_size: int,
    fine_value: wp.array[wp.float64],
    coarse_value: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    coarse_node = scalar_row // coarse_block_size
    coarse_local = scalar_row - coarse_node * coarse_block_size
    result = wp.float64(0.0)
    for cursor in range(member_offsets[coarse_node], member_offsets[coarse_node + 1]):
        fine_node = member_fine_nodes[cursor]
        fine_base = fine_node * fine_block_size
        for fine_local in range(fine_block_size):
            block_entry = (fine_base + fine_local) * coarse_block_size + coarse_local
            result += prolongation_blocks[block_entry] * fine_value[fine_base + fine_local]
    coarse_value[scalar_row] = result


@wp.kernel(enable_backward=False)
def _prolong_add_owned_rows(
    aggregate: wp.array[wp.int32],
    prolongation_blocks: wp.array[wp.float64],
    fine_block_size: int,
    coarse_block_size: int,
    coarse_value: wp.array[wp.float64],
    fine_value: wp.array[wp.float64],
):
    scalar_row = wp.tid()
    fine_node = scalar_row // fine_block_size
    coarse_base = aggregate[fine_node] * coarse_block_size
    block_base = scalar_row * coarse_block_size
    result = wp.float64(0.0)
    for coarse_local in range(coarse_block_size):
        result += prolongation_blocks[block_base + coarse_local] * coarse_value[coarse_base + coarse_local]
    fine_value[scalar_row] += result


@wp.kernel(enable_backward=False)
def _solve_coarsest_cholesky(
    lower: wp.array[wp.float64],
    scalar_size: int,
    rhs: wp.array[wp.float64],
    intermediate: wp.array[wp.float64],
    solution: wp.array[wp.float64],
):
    # The host rejects scalar_size > MAX_COARSE_SCALAR_SIZE.  One owner thread
    # preserves exactly ordered forward/back substitution without atomics.
    if wp.tid() == 0:
        for row in range(scalar_size):
            value = rhs[row]
            for column in range(row):
                value -= lower[row * scalar_size + column] * intermediate[column]
            intermediate[row] = value / lower[row * scalar_size + row]
        cursor = int(0)
        while cursor < scalar_size:
            row = scalar_size - cursor - 1
            value = intermediate[row]
            for column in range(row + 1, scalar_size):
                value -= lower[column * scalar_size + row] * solution[column]
            solution[row] = value / lower[row * scalar_size + row]
            cursor += 1


@dataclasses.dataclass(frozen=True, slots=True)
class _WarpMultigridLevel:
    """Persistent device arrays for one immutable CPU hierarchy level."""

    block_row_count: int
    block_size: int
    scalar_size: int
    stored_block_count: int
    row_offsets: wp.array
    column_indices: wp.array
    matrix_values: wp.array
    inverse_diagonal: wp.array | None
    omega: float | None
    aggregate: wp.array | None
    prolongation_blocks: wp.array | None
    member_offsets: wp.array | None
    member_fine_nodes: wp.array | None
    coarse_node_count: int | None
    coarse_block_size: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class WarpVCycleRecord:
    """Synchronized immutable diagnostic record for one device V-cycle."""

    correction: np.ndarray
    work: VCycleWorkRecord
    scheduled_kernel_launches: int
    capture_replay: bool
    content_sha256: str
    contract_id: str = CONTRACT_ID
    kernel_version: str = KERNEL_VERSION
    research_only: bool = True
    performance_evidence: bool = False

    def __post_init__(self) -> None:
        correction = _immutable_array(self.correction, np.float64)
        object.__setattr__(self, "correction", correction)
        if self.scheduled_kernel_launches < 1:
            raise ValueError("scheduled_kernel_launches must be positive")
        if not isinstance(self.capture_replay, bool):
            raise TypeError("capture_replay must be a bool")
        if not self.research_only or self.performance_evidence:
            raise ValueError("this research primitive cannot claim performance evidence")

    def deterministic_record(self) -> dict[str, object]:
        """Return finite JSON-shaped identity and exact-work evidence."""
        return {
            "contract_id": self.contract_id,
            "kernel_version": self.kernel_version,
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
            "scheduled_kernel_launches": self.scheduled_kernel_launches,
            "work_sha256": self.work.content_sha256,
            "content_sha256": self.content_sha256,
        }


class WarpStaticMultigridHierarchy:
    """Persistent Warp snapshot of one static spectral-free MG hierarchy."""

    def __init__(self, hierarchy: StaticMultigridHierarchy, *, device: str = "cpu"):
        if not isinstance(hierarchy, StaticMultigridHierarchy):
            raise TypeError("hierarchy must be a StaticMultigridHierarchy")
        if hierarchy.solver_contract != SPECTRAL_FREE_CONTRACT:
            raise ValueError("hierarchy does not implement the registered spectral-free contract")
        if not hierarchy.levels or hierarchy.levels[0].matrix.block_size != 3:
            raise ValueError("the fine hierarchy level must contain three displacement DOFs per node")
        coarse_size = hierarchy.levels[-1].matrix.scalar_size
        if coarse_size > MAX_COARSE_SCALAR_SIZE:
            raise ValueError(f"coarsest scalar size {coarse_size} exceeds fixed bound {MAX_COARSE_SCALAR_SIZE}")
        self.device = wp.get_device(device)
        self.hierarchy_sha256 = hierarchy.content_sha256
        self.solver_contract = hierarchy.solver_contract
        self.static_model_sha256 = hierarchy.static_model_sha256
        self.free_vertices_host = _immutable_array(
            _as_int32(hierarchy.free_vertices, name="free_vertices"),
            np.int32,
        )
        self.pre_smooth_steps = hierarchy.pre_smooth_steps
        self.post_smooth_steps = hierarchy.post_smooth_steps
        self.n_free = hierarchy.levels[0].matrix.block_row_count
        self.n_free_dofs = hierarchy.levels[0].matrix.scalar_size
        levels: list[_WarpMultigridLevel] = []

        for level_index, cpu_level in enumerate(hierarchy.levels):
            matrix = cpu_level.matrix
            offsets = _as_int32(matrix.row_offsets, name=f"level {level_index} row_offsets")
            columns = _as_int32(matrix.column_indices, name=f"level {level_index} column_indices")
            matrix_values = np.asarray(matrix.values, dtype=np.float64).reshape(-1)
            is_coarsest = level_index == len(hierarchy.levels) - 1
            if is_coarsest:
                inverse_diagonal = None
                omega = None
                aggregate = None
                prolongation_blocks = None
                member_offsets = None
                member_fine_nodes = None
                coarse_node_count = None
                coarse_block_size = None
            else:
                if cpu_level.smoother is None or cpu_level.prolongation is None:
                    raise ValueError(f"non-coarsest level {level_index} is missing its smoother or transfer")
                prolongation = cpu_level.prolongation
                next_matrix = hierarchy.levels[level_index + 1].matrix
                if (
                    prolongation.fine_node_count != matrix.block_row_count
                    or prolongation.coarse_node_count != next_matrix.block_row_count
                    or prolongation.fine_block_size != matrix.block_size
                    or prolongation.coarse_block_size != next_matrix.block_size
                ):
                    raise ValueError(f"level {level_index} transfer is inconsistent with its adjacent matrices")
                aggregate_host = _as_int32(prolongation.aggregate, name=f"level {level_index} aggregate")
                coarse_node_count = prolongation.coarse_node_count
                coarse_block_size = prolongation.coarse_block_size
                counts = np.bincount(aggregate_host, minlength=coarse_node_count)
                reverse_offsets = np.zeros(coarse_node_count + 1, dtype=np.int32)
                reverse_offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
                reverse_nodes = np.concatenate(
                    [np.flatnonzero(aggregate_host == aggregate_id) for aggregate_id in range(coarse_node_count)]
                ).astype(np.int32, copy=False)
                inverse_diagonal = wp.array(
                    np.asarray(cpu_level.smoother.inverse_diagonal, dtype=np.float64).reshape(-1),
                    dtype=wp.float64,
                    device=self.device,
                )
                omega = float(cpu_level.smoother.omega)
                aggregate = wp.array(aggregate_host, dtype=wp.int32, device=self.device)
                prolongation_blocks = wp.array(
                    np.asarray(prolongation.blocks, dtype=np.float64).reshape(-1),
                    dtype=wp.float64,
                    device=self.device,
                )
                member_offsets = wp.array(reverse_offsets, dtype=wp.int32, device=self.device)
                member_fine_nodes = wp.array(reverse_nodes, dtype=wp.int32, device=self.device)
            levels.append(
                _WarpMultigridLevel(
                    block_row_count=matrix.block_row_count,
                    block_size=matrix.block_size,
                    scalar_size=matrix.scalar_size,
                    stored_block_count=matrix.stored_block_count,
                    row_offsets=wp.array(offsets, dtype=wp.int32, device=self.device),
                    column_indices=wp.array(columns, dtype=wp.int32, device=self.device),
                    matrix_values=wp.array(matrix_values, dtype=wp.float64, device=self.device),
                    inverse_diagonal=inverse_diagonal,
                    omega=omega,
                    aggregate=aggregate,
                    prolongation_blocks=prolongation_blocks,
                    member_offsets=member_offsets,
                    member_fine_nodes=member_fine_nodes,
                    coarse_node_count=coarse_node_count,
                    coarse_block_size=coarse_block_size,
                )
            )
        self.levels = tuple(levels)
        factor = np.asarray(hierarchy.coarse_cholesky, dtype=np.float64)
        if factor.shape != (coarse_size, coarse_size):
            raise ValueError("coarse Cholesky shape does not match the coarsest matrix")
        self.coarse_cholesky = wp.array(factor.reshape(-1), dtype=wp.float64, device=self.device)
        self.device_snapshot_sha256 = _hash_parts(
            "warp-static-multigrid-snapshot-v1",
            (
                ("hierarchy_sha256", self.hierarchy_sha256),
                ("kernel_version", KERNEL_VERSION),
                ("coarse_scalar_bound", MAX_COARSE_SCALAR_SIZE),
            ),
        )

    @classmethod
    def from_hierarchy(
        cls,
        hierarchy: StaticMultigridHierarchy,
        *,
        device: str = "cpu",
    ) -> WarpStaticMultigridHierarchy:
        """Upload an immutable CPU hierarchy to one Warp device."""
        return cls(hierarchy, device=device)

    def create_workspace(self) -> WarpVCycleWorkspace:
        """Allocate all reusable temporaries required by one V-cycle."""
        return WarpVCycleWorkspace(self)

    @property
    def scheduled_kernel_launches(self) -> int:
        """Exact fixed launch count for one V-cycle application."""
        noncoarse = len(self.levels) - 1
        smooth_steps = self.pre_smooth_steps + self.post_smooth_steps
        return 3 + noncoarse * (5 + 3 * smooth_steps)

    def _validate_fine_vector(self, vector: wp.array[wp.vec3d], *, name: str) -> None:
        if vector.device != self.device or vector.dtype != wp.vec3d or vector.shape != (self.n_free,):
            raise ValueError(f"{name} must be a vec3d array of shape ({self.n_free},) on {self.device}")

    def launch_apply(
        self,
        rhs: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: WarpVCycleWorkspace,
    ) -> None:
        """Launch one allocation-free fixed-shape symmetric V-cycle."""
        self._validate_fine_vector(rhs, name="rhs")
        self._validate_fine_vector(output, name="output")
        if not isinstance(workspace, WarpVCycleWorkspace) or workspace._hierarchy_identity != id(self):
            raise ValueError("workspace belongs to a different device hierarchy")
        wp.launch(
            _copy_vec3_to_scalar,
            dim=self.n_free,
            inputs=[rhs, workspace.level_rhs[0]],
            device=self.device,
        )
        self._launch_level(0, workspace)
        wp.launch(
            _copy_scalar_to_vec3,
            dim=self.n_free,
            inputs=[workspace.level_correction[0], output],
            device=self.device,
        )

    def _launch_smooth(self, level_index: int, workspace: WarpVCycleWorkspace) -> None:
        """Launch one fixed block-Jacobi correction sweep."""
        level = self.levels[level_index]
        if level.inverse_diagonal is None or level.omega is None:
            raise RuntimeError(f"device level {level_index} is missing its smoother")
        wp.launch(
            _block_csr_matvec,
            dim=level.scalar_size,
            inputs=[
                level.row_offsets,
                level.column_indices,
                level.matrix_values,
                level.block_size,
                workspace.level_correction[level_index],
                workspace.level_product[level_index],
            ],
            device=self.device,
        )
        wp.launch(
            _subtract_product,
            dim=level.scalar_size,
            inputs=[
                workspace.level_rhs[level_index],
                workspace.level_product[level_index],
                workspace.level_residual[level_index],
            ],
            device=self.device,
        )
        wp.launch(
            _jacobi_add,
            dim=level.scalar_size,
            inputs=[
                workspace.level_residual[level_index],
                level.inverse_diagonal,
                level.block_size,
                level.omega,
                workspace.level_correction[level_index],
            ],
            device=self.device,
        )

    def _launch_level(self, level_index: int, workspace: WarpVCycleWorkspace) -> None:
        level = self.levels[level_index]
        rhs = workspace.level_rhs[level_index]
        correction = workspace.level_correction[level_index]
        if level_index == len(self.levels) - 1:
            wp.launch(
                _solve_coarsest_cholesky,
                dim=1,
                inputs=[
                    self.coarse_cholesky,
                    level.scalar_size,
                    rhs,
                    workspace.coarse_intermediate,
                    correction,
                ],
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

        wp.launch(_clear_scalar, dim=level.scalar_size, inputs=[correction], device=self.device)

        for _ in range(self.pre_smooth_steps):
            self._launch_smooth(level_index, workspace)
        wp.launch(
            _block_csr_matvec,
            dim=level.scalar_size,
            inputs=[
                level.row_offsets,
                level.column_indices,
                level.matrix_values,
                level.block_size,
                correction,
                workspace.level_product[level_index],
            ],
            device=self.device,
        )
        wp.launch(
            _subtract_product,
            dim=level.scalar_size,
            inputs=[rhs, workspace.level_product[level_index], workspace.level_residual[level_index]],
            device=self.device,
        )
        wp.launch(
            _restrict_owned_rows,
            dim=self.levels[level_index + 1].scalar_size,
            inputs=[
                level.member_offsets,
                level.member_fine_nodes,
                level.prolongation_blocks,
                level.block_size,
                level.coarse_block_size,
                workspace.level_residual[level_index],
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
                workspace.level_correction[level_index + 1],
                correction,
            ],
            device=self.device,
        )
        for _ in range(self.post_smooth_steps):
            self._launch_smooth(level_index, workspace)


class WarpVCycleWorkspace:
    """Persistent input, output, and temporary buffers for one V-cycle."""

    def __init__(self, hierarchy: WarpStaticMultigridHierarchy):
        if not isinstance(hierarchy, WarpStaticMultigridHierarchy):
            raise TypeError("hierarchy must be a WarpStaticMultigridHierarchy")
        self.hierarchy = hierarchy
        self._hierarchy_identity = id(hierarchy)
        self.rhs = wp.empty(hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        self.correction = wp.empty(hierarchy.n_free, dtype=wp.vec3d, device=hierarchy.device)
        self.level_rhs = tuple(
            wp.empty(level.scalar_size, dtype=wp.float64, device=hierarchy.device) for level in hierarchy.levels
        )
        self.level_correction = tuple(
            wp.empty(level.scalar_size, dtype=wp.float64, device=hierarchy.device) for level in hierarchy.levels
        )
        self.level_product = tuple(
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

    @property
    def scheduled_kernel_launches(self) -> int:
        """Exact launch count for one fixed V-cycle application."""
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
        """Launch the complete allocation-free V-cycle schedule."""
        self.hierarchy.launch_apply(self.rhs, self.correction, self)

    def record(self, *, capture_replay: bool = False) -> WarpVCycleRecord:
        """Synchronously materialize immutable result and work evidence."""
        if not isinstance(capture_replay, bool):
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.rhs.numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self.correction.numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(rhs, correction, capture_replay=capture_replay)

    def record_internal_application(self, *, capture_replay: bool = False) -> WarpVCycleRecord:
        """Record an external-array apply retained in this workspace's levels."""
        if not isinstance(capture_replay, bool):
            raise TypeError("capture_replay must be a bool")
        rhs = np.asarray(self.level_rhs[0].numpy(), dtype=np.float64).reshape(-1)
        correction = np.asarray(self.level_correction[0].numpy(), dtype=np.float64).reshape(-1)
        return self._record_host_vectors(rhs, correction, capture_replay=capture_replay)

    def _record_host_vectors(
        self,
        rhs: np.ndarray,
        correction: np.ndarray,
        *,
        capture_replay: bool,
    ) -> WarpVCycleRecord:
        """Build one immutable record from synchronized flat host vectors."""
        rhs = np.asarray(rhs, dtype=np.float64).reshape(self.hierarchy.n_free_dofs, 1)
        correction = np.asarray(correction, dtype=np.float64).reshape(-1)
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
        prolongation_products = restriction_products
        record_parts = (
            ("hierarchy_sha256", self.hierarchy.hierarchy_sha256),
            ("rhs_sha256", rhs_sha256),
            ("result_sha256", result_sha256),
            ("rhs_count", 1),
            ("level_visits", _immutable_array(level_visits, np.int64)),
            ("matrix_block_products", matrix_products),
            ("smoother_block_solves", smoother_solves),
            ("restriction_block_products", restriction_products),
            ("prolongation_block_products", prolongation_products),
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
            prolongation_block_products=prolongation_products,
            coarsest_factor_solves=1,
            content_sha256=work_sha256,
        )
        content_sha256 = _hash_parts(
            "warp-v-cycle-result-v1",
            (
                ("snapshot_sha256", self.hierarchy.device_snapshot_sha256),
                ("work_sha256", work_sha256),
                ("scheduled_kernel_launches", self.scheduled_kernel_launches),
                ("capture_replay", capture_replay),
            ),
        )
        return WarpVCycleRecord(
            correction=correction_frozen,
            work=work,
            scheduled_kernel_launches=self.scheduled_kernel_launches,
            capture_replay=capture_replay,
            content_sha256=content_sha256,
        )


class WarpStaticMultigridPreconditioner(WarpDevicePreconditioner):
    """Typed PCG boundary for one immutable static Warp MG hierarchy."""

    def __init__(self, hierarchy: WarpStaticMultigridHierarchy):
        if not isinstance(hierarchy, WarpStaticMultigridHierarchy):
            raise TypeError("hierarchy must be a WarpStaticMultigridHierarchy")
        self.hierarchy = hierarchy
        self.device = hierarchy.device
        self.vector_count = hierarchy.n_free
        self.free_vertices_host = hierarchy.free_vertices_host
        self.static_preconditioner_sha256 = hierarchy.hierarchy_sha256
        self.device_snapshot_sha256 = hierarchy.device_snapshot_sha256
        self.preconditioner_identity = f"static-mg-v-cycle-warp-v1:{hierarchy.hierarchy_sha256}"
        self.application_kernel_launches = hierarchy.scheduled_kernel_launches

    def create_application_workspace(self) -> WarpVCycleWorkspace:
        """Allocate one independently retained V-cycle workspace."""
        return self.hierarchy.create_workspace()

    def launch_apply(
        self,
        rhs: wp.array[wp.vec3d],
        output: wp.array[wp.vec3d],
        workspace: object,
    ) -> None:
        """Enqueue one V-cycle without host synchronization or allocation."""
        if not isinstance(workspace, WarpVCycleWorkspace):
            raise TypeError("workspace must be a WarpVCycleWorkspace")
        if workspace._hierarchy_identity != id(self.hierarchy):
            raise ValueError("workspace belongs to a different static multigrid preconditioner")
        self.hierarchy.launch_apply(rhs, output, workspace)

    def record_application(
        self,
        application_index: int,
        workspace: object,
        *,
        capture_replay: bool,
    ) -> WarpDevicePreconditionerApplication:
        """Synchronously retain one V-cycle's hashes and exact algebraic work."""
        if not isinstance(workspace, WarpVCycleWorkspace):
            raise TypeError("workspace must be a WarpVCycleWorkspace")
        if workspace._hierarchy_identity != id(self.hierarchy):
            raise ValueError("workspace belongs to a different static multigrid preconditioner")
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
            output_finite=bool(np.isfinite(result.correction).all()),
            capture_replay=capture_replay,
        )
