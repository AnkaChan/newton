# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Deterministic static block multigrid for residual-correction research.

This module is a NumPy quality ceiling for the multiplicative graph VBD
corrector.  It deliberately separates the hierarchy policy from the eventual
matrix-free GPU implementation: callers supply one assembled fine-level SPD
block matrix, rest positions, and the vertex IDs represented by its blocks.
The retained hierarchy is block sparse and has no stored dense matrices.

The default coarse enrichment contains the three translations and three
infinitesimal rotations evaluated in rest space.  The rotational columns are
*empirical coarse features*, not asserted null modes of the stable-Neo-Hookean
Gauss--Newton tangent: its shear term can give a skew deformation positive
energy.  ``mode_kind="translation"`` is provided for the required ablation.

Every operation that affects the hierarchy is deterministic.  Aggregation
uses graph strength followed by original vertex-ID tie breaks, local bases use
deterministic modified Gram--Schmidt with sign normalization and coordinate
rank completion, and sparse rows are stored in increasing column order.
Arrays are copied, made read-only, and covered by content hashes.  V-cycle and
PCG calls return immutable work records so quality evidence can bind both the
operator and the exact amount of algebraic work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import math
import numbers
from collections.abc import Iterable, Mapping

import numpy as np

from .correction_gpu import MatrixFreeStableNHOperator

SPECTRAL_FREE_CONTRACT = "spectral-free-multiplicative-graph-vbd-static-v1"
"""Content-record contract for the static MG-VBD research hierarchy."""


def _frozen_array(value: np.ndarray | Iterable[float], dtype: np.dtype | type) -> np.ndarray:
    """Return a C-contiguous array backed by immutable ``bytes``."""
    source = np.array(value, dtype=dtype, order="C", copy=True)
    result = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)
    return result


def _hash_parts(tag: str, parts: Iterable[tuple[str, object]]) -> str:
    """Hash typed, length-delimited scalar and array content."""
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
        elif isinstance(value, float):
            add_bytes(b"float64")
            add_bytes(np.float64(value).tobytes())
        elif isinstance(value, str):
            add_bytes(b"str")
            add_bytes(value.encode("utf-8"))
        elif value is None:
            add_bytes(b"none")
        else:
            raise TypeError(f"unsupported hash part {name!r}: {type(value).__name__}")
    return digest.hexdigest()


def _as_rhs(value: np.ndarray, size: int, *, name: str) -> tuple[np.ndarray, bool]:
    """Canonicalize one or several right-hand sides to shape ``(size, K)``."""
    rhs = np.asarray(value, dtype=np.float64)
    was_vector = rhs.ndim == 1
    if was_vector:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != size or rhs.shape[1] == 0:
        raise ValueError(f"{name} must have shape ({size},) or ({size}, K), got {rhs.shape}")
    if not np.isfinite(rhs).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(rhs, dtype=np.float64, order="C", copy=True), was_vector


def _scaled_norm(value: np.ndarray, *, name: str) -> float:
    """Compute a finite Euclidean norm without squaring large values."""
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains a non-finite value")
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    if scale == 0.0:
        return 0.0
    normalized = array / scale
    with np.errstate(over="ignore", invalid="ignore"):
        result = scale * math.sqrt(float(normalized @ normalized))
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} norm is not representable in float64")
    return result


def _finite_dot(left: np.ndarray, right: np.ndarray, *, name: str) -> float:
    """Compute a scaled dot product and reject unrepresentable results."""
    left_scale = float(np.max(np.abs(left))) if left.size else 0.0
    right_scale = float(np.max(np.abs(right))) if right.size else 0.0
    if not math.isfinite(left_scale) or not math.isfinite(right_scale):
        raise FloatingPointError(f"{name} input is non-finite")
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    normalized_dot = float((left / left_scale) @ (right / right_scale))
    with np.errstate(over="ignore", invalid="ignore"):
        result = (normalized_dot * left_scale) * right_scale
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is not representable in float64")
    return result


def _require_finite(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``value`` or fail closed when arithmetic produced NaN/Inf."""
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} produced a non-finite value")
    return value


def _finite_scale(value: float, scale: float, *, name: str) -> float:
    """Multiply two non-negative scalars and reject overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = value * scale
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is not representable in float64")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class StaticBlockMatrix:
    """Immutable square block-CSR matrix."""

    block_row_count: int
    block_size: int
    row_offsets: np.ndarray
    column_indices: np.ndarray
    values: np.ndarray
    content_sha256: str

    @property
    def scalar_size(self) -> int:
        """Number of scalar rows and columns."""
        return self.block_row_count * self.block_size

    @property
    def stored_block_count(self) -> int:
        """Number of explicitly stored blocks, including diagonal blocks."""
        return int(self.column_indices.size)

    @classmethod
    def from_dense(
        cls,
        matrix: np.ndarray,
        *,
        block_size: int = 3,
        zero_tolerance: float = 0.0,
    ) -> StaticBlockMatrix:
        """Create a canonical block-CSR matrix from a dense SPD matrix.

        Args:
            matrix: Dense scalar matrix, shape ``(block_size*N, block_size*N)``.
            block_size: Degrees of freedom represented by one graph node.
            zero_tolerance: Drop an off-diagonal block only when every entry is
                no larger than this absolute threshold. Diagonal blocks are
                always retained.
        """
        dense = np.asarray(matrix, dtype=np.float64)
        if dense.ndim == 4:
            if dense.shape[0] != dense.shape[1] or dense.shape[2:] != (block_size, block_size):
                raise ValueError("four-dimensional matrix must have shape (N, N, block_size, block_size)")
            dense = dense.transpose(0, 2, 1, 3).reshape(dense.shape[0] * block_size, -1)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1] or dense.shape[0] == 0:
            raise ValueError("matrix must be a non-empty square matrix")
        if block_size <= 0 or dense.shape[0] % block_size:
            raise ValueError("matrix scalar size must be divisible by block_size")
        if not math.isfinite(zero_tolerance) or zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be finite and non-negative")
        if not np.isfinite(dense).all():
            raise ValueError("matrix must contain only finite values")
        scale = max(1.0, float(np.max(np.abs(dense))))
        if not np.allclose(dense, dense.T, rtol=0.0, atol=64.0 * np.finfo(np.float64).eps * scale):
            raise ValueError("matrix must be symmetric")
        dense = 0.5 * (dense + dense.T)
        try:
            np.linalg.cholesky(dense)
        except np.linalg.LinAlgError as error:
            raise ValueError("matrix must be positive definite") from error

        block_count = dense.shape[0] // block_size
        blocks: dict[tuple[int, int], np.ndarray] = {}
        for row in range(block_count):
            row_slice = slice(row * block_size, (row + 1) * block_size)
            for column in range(block_count):
                column_slice = slice(column * block_size, (column + 1) * block_size)
                block = dense[row_slice, column_slice]
                if row == column or float(np.max(np.abs(block))) > zero_tolerance:
                    blocks[(row, column)] = block
        return _block_matrix_from_mapping(block_count, block_size, blocks)

    @classmethod
    def from_block_entries(
        cls,
        block_row_count: int,
        entries: Iterable[tuple[int, int, np.ndarray]],
        *,
        block_size: int = 3,
        symmetry_tolerance: float = 1.0e-12,
    ) -> StaticBlockMatrix:
        """Assemble block CSR directly without forming a dense fine matrix.

        Duplicate entries are accumulated in input order, which supports
        deterministic assembly from element-local tangent blocks or from a
        matrix-free operator's frozen block-diagonal and edge-block export.
        Both directions of every nonzero off-diagonal block must be supplied.
        The caller retains responsibility for the global SPD contract; this
        constructor verifies finite values, symmetric block structure, and
        positive-definite diagonal blocks without materializing the matrix.

        Args:
            block_row_count: Number of graph nodes.
            entries: ``(row, column, block)`` entries. Repeated locations are
                summed deterministically.
            block_size: Scalar degrees of freedom per graph node.
            symmetry_tolerance: Relative/absolute tolerance for paired blocks.
        """
        if block_row_count <= 0 or block_size <= 0:
            raise ValueError("block_row_count and block_size must be positive")
        if not math.isfinite(symmetry_tolerance) or symmetry_tolerance < 0.0:
            raise ValueError("symmetry_tolerance must be finite and non-negative")
        accumulated: dict[tuple[int, int], np.ndarray] = {}
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError(f"entry {entry_index} must be a (row, column, block) tuple")
            row, column, raw_block = entry
            if (
                not isinstance(row, int)
                or isinstance(row, bool)
                or not isinstance(column, int)
                or isinstance(column, bool)
            ):
                raise ValueError(f"entry {entry_index} row and column must be integers")
            if not 0 <= row < block_row_count or not 0 <= column < block_row_count:
                raise ValueError(f"entry {entry_index} index is outside the matrix")
            block = np.asarray(raw_block, dtype=np.float64)
            if block.shape != (block_size, block_size) or not np.isfinite(block).all():
                raise ValueError(f"entry {entry_index} must contain a finite ({block_size}, {block_size}) block")
            key = (row, column)
            if key in accumulated:
                accumulated[key] += block
            else:
                accumulated[key] = np.array(block, dtype=np.float64, copy=True)
        for row in range(block_row_count):
            if (row, row) not in accumulated:
                raise ValueError(f"entries are missing diagonal block {row}")
            diagonal = accumulated[(row, row)]
            diagonal_scale = max(1.0, float(np.max(np.abs(diagonal))))
            if not np.allclose(
                diagonal,
                diagonal.T,
                rtol=symmetry_tolerance,
                atol=symmetry_tolerance * diagonal_scale,
            ):
                raise ValueError(f"diagonal block {row} is not symmetric")
            diagonal = 0.5 * (diagonal + diagonal.T)
            try:
                np.linalg.cholesky(diagonal)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"diagonal block {row} is not positive definite") from error
            accumulated[(row, row)] = diagonal
        for (row, column), block in tuple(accumulated.items()):
            if row >= column:
                continue
            reverse = accumulated.get((column, row))
            if reverse is None:
                raise ValueError(f"entries are missing transpose block ({column}, {row})")
            scale = max(1.0, float(np.max(np.abs(block))), float(np.max(np.abs(reverse))))
            if not np.allclose(
                block,
                reverse.T,
                rtol=symmetry_tolerance,
                atol=symmetry_tolerance * scale,
            ):
                raise ValueError(f"blocks ({row}, {column}) and ({column}, {row}) are not transposes")
            canonical = 0.5 * (block + reverse.T)
            accumulated[(row, column)] = canonical
            accumulated[(column, row)] = canonical.T
        for row, column in accumulated:
            if row != column and (column, row) not in accumulated:
                raise ValueError(f"entries are missing transpose block ({column}, {row})")
        return _block_matrix_from_mapping(block_row_count, block_size, accumulated)

    def diagonal_blocks(self) -> np.ndarray:
        """Return a writable copy of the diagonal blocks."""
        diagonal = np.empty((self.block_row_count, self.block_size, self.block_size), dtype=np.float64)
        for row in range(self.block_row_count):
            begin = int(self.row_offsets[row])
            end = int(self.row_offsets[row + 1])
            columns = self.column_indices[begin:end]
            hits = np.flatnonzero(columns == row)
            if hits.size != 1:
                raise RuntimeError(f"matrix row {row} does not contain exactly one diagonal block")
            diagonal[row] = self.values[begin + int(hits[0])]
        return diagonal

    def matmul(self, value: np.ndarray) -> np.ndarray:
        """Multiply by one vector or a matrix of column vectors."""
        rhs, was_vector = _as_rhs(value, self.scalar_size, name="value")
        width = rhs.shape[1]
        shaped = rhs.reshape(self.block_row_count, self.block_size, width)
        result = np.zeros_like(shaped)
        for row in range(self.block_row_count):
            for entry in range(int(self.row_offsets[row]), int(self.row_offsets[row + 1])):
                column = int(self.column_indices[entry])
                result[row] += self.values[entry] @ shaped[column]
        result = result.reshape(self.scalar_size, width)
        return result[:, 0] if was_vector else result

    def to_dense(self) -> np.ndarray:
        """Materialize a dense copy for validation or a small coarsest solve."""
        dense = np.zeros((self.scalar_size, self.scalar_size), dtype=np.float64)
        size = self.block_size
        for row in range(self.block_row_count):
            row_slice = slice(row * size, (row + 1) * size)
            for entry in range(int(self.row_offsets[row]), int(self.row_offsets[row + 1])):
                column = int(self.column_indices[entry])
                column_slice = slice(column * size, (column + 1) * size)
                dense[row_slice, column_slice] = self.values[entry]
        return dense


def _block_matrix_from_mapping(
    block_row_count: int,
    block_size: int,
    blocks: Mapping[tuple[int, int], np.ndarray],
) -> StaticBlockMatrix:
    """Canonicalize a complete symmetric block mapping."""
    row_buckets: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(block_row_count)]
    for (row, column), block in blocks.items():
        if not 0 <= row < block_row_count:
            raise ValueError("block row is outside the matrix")
        row_buckets[row].append((column, block))

    row_offsets = [0]
    column_indices: list[int] = []
    values: list[np.ndarray] = []
    for row in range(block_row_count):
        row_items = sorted(row_buckets[row], key=lambda item: item[0])
        if not row_items or all(column != row for column, _block in row_items):
            raise ValueError(f"block mapping is missing diagonal block {row}")
        for column, block in row_items:
            if not 0 <= column < block_row_count:
                raise ValueError("block column is outside the matrix")
            block_array = np.asarray(block, dtype=np.float64)
            if block_array.shape != (block_size, block_size) or not np.isfinite(block_array).all():
                raise ValueError("block values must be finite square block_size arrays")
            column_indices.append(column)
            values.append(np.array(block_array, copy=True))
        row_offsets.append(len(column_indices))

    offsets_array = _frozen_array(row_offsets, np.int64)
    columns_array = _frozen_array(column_indices, np.int64)
    values_array = _frozen_array(values, np.float64)
    content_sha256 = _hash_parts(
        "static-block-matrix-v1",
        (
            ("block_row_count", block_row_count),
            ("block_size", block_size),
            ("row_offsets", offsets_array),
            ("column_indices", columns_array),
            ("values", values_array),
        ),
    )
    return StaticBlockMatrix(
        block_row_count=block_row_count,
        block_size=block_size,
        row_offsets=offsets_array,
        column_indices=columns_array,
        values=values_array,
        content_sha256=content_sha256,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TentativeProlongation:
    """One-block-per-fine-node tentative prolongation."""

    aggregate: np.ndarray
    blocks: np.ndarray
    coarse_node_count: int
    content_sha256: str

    @property
    def fine_node_count(self) -> int:
        return int(self.aggregate.size)

    @property
    def fine_block_size(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def coarse_block_size(self) -> int:
        return int(self.blocks.shape[2])

    @property
    def fine_scalar_size(self) -> int:
        return self.fine_node_count * self.fine_block_size

    @property
    def coarse_scalar_size(self) -> int:
        return self.coarse_node_count * self.coarse_block_size

    def prolong(self, value: np.ndarray) -> np.ndarray:
        """Apply ``P`` to one or several coarse vectors."""
        rhs, was_vector = _as_rhs(value, self.coarse_scalar_size, name="coarse value")
        width = rhs.shape[1]
        coarse = rhs.reshape(self.coarse_node_count, self.coarse_block_size, width)
        fine = np.empty((self.fine_node_count, self.fine_block_size, width), dtype=np.float64)
        for node in range(self.fine_node_count):
            fine[node] = self.blocks[node] @ coarse[int(self.aggregate[node])]
        fine = fine.reshape(self.fine_scalar_size, width)
        return fine[:, 0] if was_vector else fine

    def restrict(self, value: np.ndarray) -> np.ndarray:
        """Apply ``P.T`` to one or several fine vectors."""
        rhs, was_vector = _as_rhs(value, self.fine_scalar_size, name="fine value")
        width = rhs.shape[1]
        fine = rhs.reshape(self.fine_node_count, self.fine_block_size, width)
        coarse = np.zeros((self.coarse_node_count, self.coarse_block_size, width), dtype=np.float64)
        for node in range(self.fine_node_count):
            coarse[int(self.aggregate[node])] += self.blocks[node].T @ fine[node]
        coarse = coarse.reshape(self.coarse_scalar_size, width)
        return coarse[:, 0] if was_vector else coarse

    def to_dense(self) -> np.ndarray:
        """Materialize ``P`` for validation."""
        dense = np.zeros((self.fine_scalar_size, self.coarse_scalar_size), dtype=np.float64)
        for node in range(self.fine_node_count):
            row = slice(node * self.fine_block_size, (node + 1) * self.fine_block_size)
            aggregate = int(self.aggregate[node])
            column = slice(aggregate * self.coarse_block_size, (aggregate + 1) * self.coarse_block_size)
            dense[row, column] = self.blocks[node]
        return dense


@dataclasses.dataclass(frozen=True, slots=True)
class BlockJacobiSmoother:
    """Conservatively damped block-Jacobi smoother."""

    inverse_diagonal: np.ndarray
    omega: float
    normalized_spectral_upper_bound: float
    content_sha256: str

    def apply(self, value: np.ndarray) -> np.ndarray:
        """Apply ``omega * D^-1`` to one or several residual vectors."""
        block_count, block_size, _ = self.inverse_diagonal.shape
        rhs, was_vector = _as_rhs(value, block_count * block_size, name="residual")
        width = rhs.shape[1]
        shaped = rhs.reshape(block_count, block_size, width)
        result = self.omega * np.einsum("nij,njk->nik", self.inverse_diagonal, shaped, optimize=False)
        result = result.reshape(block_count * block_size, width)
        return result[:, 0] if was_vector else result


@dataclasses.dataclass(frozen=True, slots=True)
class StaticMultigridLevel:
    """One matrix level and, except at the coarsest level, its transfer."""

    matrix: StaticBlockMatrix
    node_ids: np.ndarray
    enrichment: np.ndarray
    aggregate: np.ndarray | None
    prolongation: TentativeProlongation | None
    smoother: BlockJacobiSmoother | None
    content_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class HierarchyStorage:
    """Exact retained NumPy-array storage for a static hierarchy.

    ``dense_matrix_scalar_count_excluding_coarse_factor`` excludes the
    explicitly bounded coarsest Cholesky factor, whose payload is reported by
    ``factor_scalar_count``. No unbounded dense operator is retained.
    """

    fine_node_count: int
    fine_undirected_edge_count: int
    level_count: int
    matrix_block_count: int
    prolongation_block_count: int
    smoother_block_count: int
    matrix_scalar_count: int
    prolongation_scalar_count: int
    smoother_scalar_count: int
    enrichment_scalar_count: int
    factor_scalar_count: int
    geometry_scalar_count: int
    index_scalar_count: int
    dense_matrix_scalar_count_excluding_coarse_factor: int
    total_scalar_count: int
    total_bytes: int
    content_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class StaticMultigridHierarchy:
    """Immutable static SPD block hierarchy."""

    levels: tuple[StaticMultigridLevel, ...]
    free_vertices: np.ndarray
    rest_positions: np.ndarray
    free_masses: np.ndarray
    solver_contract: str
    mode_kind: str
    target_aggregate_size: int
    minimum_aggregate_size: int
    coarse_node_limit: int
    maximum_levels: int
    pre_smooth_steps: int
    post_smooth_steps: int
    smoother_safety: float
    static_model_sha256: str | None
    coarse_cholesky: np.ndarray
    storage: HierarchyStorage
    content_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class VCycleWorkRecord:
    """Immutable accounting for one V-cycle application."""

    hierarchy_sha256: str
    rhs_sha256: str
    result_sha256: str
    rhs_count: int
    level_visits: tuple[int, ...]
    matrix_block_products: int
    smoother_block_solves: int
    restriction_block_products: int
    prolongation_block_products: int
    coarsest_factor_solves: int
    content_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class VCycleResult:
    """A read-only V-cycle result and its bound work record."""

    correction: np.ndarray
    work: VCycleWorkRecord
    content_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class PCGSolveResult:
    """Deterministic PCG result used for hierarchy ablations."""

    solution: np.ndarray
    residual_norms: tuple[float, ...]
    iteration_count: int
    converged: bool
    matrix_sha256: str
    preconditioner_sha256: str
    rhs_sha256: str
    relative_tolerance: float
    maximum_iterations: int
    operator_applications: int
    preconditioner_applications: int
    inner_products: int
    vector_updates: int
    true_residual_norm: float
    true_relative_residual: float
    content_sha256: str


def rigid_enrichment(rest_positions: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Return translations and infinitesimal rotations, shape ``(3*N, 6)``.

    The columns are coarse enrichment features. They are not claimed exact
    null modes of a constitutive tangent.
    """
    rest = np.asarray(rest_positions, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3 or rest.shape[0] == 0 or not np.isfinite(rest).all():
        raise ValueError("rest_positions must be a finite non-empty (N, 3) array")
    if masses is None:
        weights = np.ones(rest.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(masses, dtype=np.float64)
        if weights.shape != (rest.shape[0],) or not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("masses must contain one finite positive value per rest position")
    total_mass = float(np.sum(weights))
    centroid = np.sum(weights[:, None] * rest, axis=0, keepdims=True) / total_mass
    relative = rest - centroid
    characteristic_length = math.sqrt(float(np.sum(weights * np.sum(relative * relative, axis=1)) / total_mass))
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        characteristic_length = 1.0
    relative /= characteristic_length
    modes = np.zeros((3 * rest.shape[0], 6), dtype=np.float64)
    for node, (x, y, z) in enumerate(relative):
        rows = slice(3 * node, 3 * node + 3)
        modes[rows, :3] = np.eye(3)
        modes[rows, 3:] = np.array(
            (
                (0.0, z, -y),
                (-z, 0.0, x),
                (y, -x, 0.0),
            ),
            dtype=np.float64,
        )
    return modes


def translation_enrichment(node_count: int) -> np.ndarray:
    """Return three blockwise-constant translation columns."""
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    return np.tile(np.eye(3, dtype=np.float64), (node_count, 1))


def _assemble_stable_nh_blocks(
    operator: MatrixFreeStableNHOperator,
    cofactors: np.ndarray,
) -> StaticBlockMatrix:
    """Assemble GN blocks for explicitly supplied frozen cofactors."""
    if not isinstance(operator, MatrixFreeStableNHOperator):
        raise TypeError("operator must be a MatrixFreeStableNHOperator")
    if cofactors.shape != (operator.tets.shape[0], 3, 3) or not np.isfinite(cofactors).all():
        raise ValueError("cofactors must contain one finite 3x3 matrix per tet")
    free_lookup = np.full(operator.n_vertices, -1, dtype=np.int64)
    free_lookup[operator.free] = np.arange(operator.free.size, dtype=np.int64)
    identity = np.eye(3, dtype=np.float64)

    def block_entries() -> Iterable[tuple[int, int, np.ndarray]]:
        for free_index, vertex in enumerate(operator.free):
            yield (
                free_index,
                free_index,
                (operator.mass[int(vertex)] / (operator.dt * operator.dt)) * identity,
            )
        for tet_index, tet in enumerate(operator.tets):
            cofactor = cofactors[tet_index]
            for local_row, vertex_row in enumerate(tet):
                free_row = int(free_lookup[int(vertex_row)])
                if free_row < 0:
                    continue
                shape_row = operator.shape_gradients[tet_index, local_row]
                cofactor_row = cofactor @ shape_row
                for local_column, vertex_column in enumerate(tet):
                    free_column = int(free_lookup[int(vertex_column)])
                    if free_column < 0:
                        continue
                    shape_column = operator.shape_gradients[tet_index, local_column]
                    cofactor_column = cofactor @ shape_column
                    block = operator.volumes[tet_index] * (
                        operator.mu[tet_index] * float(shape_row @ shape_column) * identity
                        + operator.lam[tet_index] * np.outer(cofactor_row, cofactor_column)
                    )
                    yield free_row, free_column, block

    return StaticBlockMatrix.from_block_entries(int(operator.free.size), block_entries())


def assemble_current_stable_nh_block_matrix(operator: MatrixFreeStableNHOperator) -> StaticBlockMatrix:
    """Assemble the current-tangent GN oracle matched by ``apply_free``.

    This helper exists for operator validation. Static MG setup must instead
    use :func:`assemble_stable_nh_rest_block_matrix`.
    """
    return _assemble_stable_nh_blocks(operator, operator.cofactors)


def assemble_stable_nh_rest_block_matrix(
    operator: MatrixFreeStableNHOperator,
    rest_positions: np.ndarray,
) -> StaticBlockMatrix:
    """Assemble the once-per-model/dt rest tangent ``A0`` with ``C = I``.

    The supplied rest geometry is independently checked against the
    operator's shape gradients. Current positions and cofactors are ignored,
    making the result invariant over correction states with identical static
    model fields.
    """
    if not isinstance(operator, MatrixFreeStableNHOperator):
        raise TypeError("operator must be a MatrixFreeStableNHOperator")
    rest = np.asarray(rest_positions, dtype=np.float64)
    if rest.shape != operator.positions.shape or not np.isfinite(rest).all():
        raise ValueError(f"rest_positions must be a finite array with shape {operator.positions.shape}")
    deformation_gradients = np.einsum(
        "tac,tad->tdc",
        operator.shape_gradients,
        rest[operator.tets],
        optimize=False,
    )
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), deformation_gradients.shape)
    # Scene rest data can originate in float32 before the research snapshot;
    # this tolerance accepts that representation noise while rejecting a
    # materially inconsistent rest/J pair.
    if not np.allclose(deformation_gradients, identity, rtol=0.0, atol=5.0e-7):
        error = float(np.max(np.abs(deformation_gradients - identity)))
        raise ValueError(f"rest_positions and shape_gradients must produce F=I (max error {error:.3e})")
    return _assemble_stable_nh_blocks(operator, identity)


def stable_nh_static_model_digest(
    operator: MatrixFreeStableNHOperator,
    rest_positions: np.ndarray,
) -> str:
    """Hash every static model input that defines the rest hierarchy."""
    if not isinstance(operator, MatrixFreeStableNHOperator):
        raise TypeError("operator must be a MatrixFreeStableNHOperator")
    rest = np.asarray(rest_positions, dtype=np.float64)
    if rest.shape != operator.positions.shape or not np.isfinite(rest).all():
        raise ValueError(f"rest_positions must be a finite array with shape {operator.positions.shape}")
    return _hash_parts(
        "stable-nh-static-model-v1",
        (
            ("tets", operator.tets),
            ("shape_gradients", operator.shape_gradients),
            ("volumes", operator.volumes),
            ("mass", operator.mass),
            ("mu", operator.mu),
            ("lam", operator.lam),
            ("dt", operator.dt),
            ("pinned", operator.pinned),
            ("free", operator.free),
            ("rest_positions", _frozen_array(rest, np.float64)),
        ),
    )


def _matrix_adjacency(matrix: StaticBlockMatrix) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Return adjacency with diagonally normalized coupling strengths."""
    diagonal = matrix.diagonal_blocks()
    inverse_lower = np.empty_like(diagonal)
    identity = np.eye(matrix.block_size, dtype=np.float64)
    for node, block in enumerate(diagonal):
        try:
            lower = np.linalg.cholesky(0.5 * (block + block.T))
        except np.linalg.LinAlgError as error:
            raise ValueError(f"diagonal block {node} is not positive definite") from error
        inverse_lower[node] = np.linalg.solve(lower, identity)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(matrix.block_row_count)]
    for row in range(matrix.block_row_count):
        for entry in range(int(matrix.row_offsets[row]), int(matrix.row_offsets[row + 1])):
            column = int(matrix.column_indices[entry])
            if column == row:
                continue
            normalized = inverse_lower[row] @ matrix.values[entry] @ inverse_lower[column].T
            strength = float(np.linalg.norm(normalized, ord="fro"))
            if strength > 0.0:
                adjacency[row].append((column, strength))
        adjacency[row].sort(key=lambda item: item[0])
    return tuple(tuple(row) for row in adjacency)


def _is_noncollinear(rest: np.ndarray, members: list[int]) -> bool:
    """Whether member rest positions span at least a plane."""
    if len(members) < 3:
        return False
    points = rest[np.asarray(members, dtype=np.int64)]
    origin = points[0]
    offsets = points - origin
    distance_squared = np.sum(offsets * offsets, axis=1)
    second = int(np.argmax(distance_squared))
    largest_distance_squared = float(distance_squared[second])
    if largest_distance_squared == 0.0:
        return False
    edge = offsets[second]
    largest_cross_norm = float(np.max(np.linalg.norm(np.cross(edge, offsets), axis=1)))
    return largest_cross_norm > 256.0 * np.finfo(np.float64).eps * largest_distance_squared


def _connected_aggregate(
    matrix: StaticBlockMatrix,
    node_ids: np.ndarray,
    *,
    target_size: int,
    minimum_size: int,
    first_level_rest: np.ndarray | None,
) -> np.ndarray:
    """Build connected strength-greedy aggregates with ID tie breaks."""
    node_count = matrix.block_row_count
    adjacency = _matrix_adjacency(matrix)
    unassigned = np.ones(node_count, dtype=bool)
    remaining = node_count
    ordered_nodes = sorted(range(node_count), key=lambda node: (int(node_ids[node]), node))
    seed_cursor = 0
    groups: list[list[int]] = []
    while remaining:
        while not unassigned[ordered_nodes[seed_cursor]]:
            seed_cursor += 1
        seed = ordered_nodes[seed_cursor]
        members = [seed]
        unassigned[seed] = False
        remaining -= 1
        while len(members) < target_size:
            frontier: dict[int, float] = {}
            for member in members:
                for neighbor, strength in adjacency[member]:
                    if unassigned[neighbor]:
                        frontier[neighbor] = frontier.get(neighbor, 0.0) + strength
            if not frontier:
                break
            selected = min(frontier, key=lambda node: (-frontier[node], int(node_ids[node]), node))
            members.append(selected)
            unassigned[selected] = False
            remaining -= 1
        groups.append(members)

    owner = np.empty(node_count, dtype=np.int64)
    for group_index, members in enumerate(groups):
        owner[members] = group_index
    group_keys = [min((int(node_ids[node]), node) for node in members) for members in groups]

    def group_valid(members: list[int]) -> bool:
        if len(members) < minimum_size:
            return False
        return first_level_rest is None or _is_noncollinear(first_level_rest, members)

    valid = [group_valid(members) for members in groups]
    invalid_heap = [(group_keys[index], index) for index in range(len(groups)) if not valid[index]]
    heapq.heapify(invalid_heap)
    while invalid_heap:
        _key, source = heapq.heappop(invalid_heap)
        if not groups[source] or valid[source]:
            continue
        crossing: dict[int, float] = {}
        for member in groups[source]:
            for neighbor, strength in adjacency[member]:
                target = int(owner[neighbor])
                if target != source:
                    crossing[target] = crossing.get(target, 0.0) + strength
        if not crossing:
            requirement = "three non-collinear vertices" if first_level_rest is not None else f"{minimum_size} nodes"
            ids = sorted(int(node_ids[node]) for node in groups[source])
            raise ValueError(f"graph component {ids} cannot form an aggregate with {requirement}")
        target = min(
            crossing,
            key=lambda index: (
                -crossing[index],
                group_keys[index],
                index,
            ),
        )
        groups[target].extend(groups[source])
        owner[groups[source]] = target
        group_keys[target] = min(group_keys[target], group_keys[source])
        groups[source] = []
        if not valid[target]:
            valid[target] = group_valid(groups[target])
        if not valid[target]:
            heapq.heappush(invalid_heap, (group_keys[target], target))

    groups = [
        sorted(members, key=lambda node: (int(node_ids[node]), node))
        for _key, members in sorted(zip(group_keys, groups, strict=True))
        if members
    ]
    aggregate = np.empty(node_count, dtype=np.int64)
    for aggregate_id, members in enumerate(groups):
        aggregate[members] = aggregate_id
    return aggregate


def _members_by_aggregate(aggregate: np.ndarray, aggregate_count: int) -> tuple[np.ndarray, ...]:
    """Bucket node ordinals by aggregate in one linear traversal."""
    buckets: list[list[int]] = [[] for _ in range(aggregate_count)]
    for node, aggregate_id in enumerate(aggregate):
        buckets[int(aggregate_id)].append(node)
    return tuple(np.asarray(members, dtype=np.int64) for members in buckets)


def _orthonormal_completion(candidates: np.ndarray, column_count: int) -> np.ndarray:
    """Deterministic MGS basis with coordinate completion and fixed signs."""
    row_count = candidates.shape[0]
    if column_count > row_count:
        raise ValueError("aggregate has fewer scalar rows than enrichment columns")
    scale = max(1.0, float(np.linalg.norm(candidates, ord="fro")))
    threshold = 512.0 * np.finfo(np.float64).eps * scale
    basis: list[np.ndarray] = []

    def try_add(raw: np.ndarray) -> None:
        if len(basis) >= column_count:
            return
        vector = np.array(raw, dtype=np.float64, copy=True)
        for _ in range(2):
            for existing in basis:
                vector -= existing * float(existing @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= threshold:
            return
        vector /= norm
        nonzero = np.flatnonzero(np.abs(vector) > threshold)
        if nonzero.size and vector[int(nonzero[0])] < 0.0:
            vector *= -1.0
        basis.append(vector)

    for column in range(candidates.shape[1]):
        try_add(candidates[:, column])
    for coordinate in range(row_count):
        unit = np.zeros(row_count, dtype=np.float64)
        unit[coordinate] = 1.0
        try_add(unit)
    if len(basis) != column_count:
        raise RuntimeError("deterministic rank completion failed")
    result = np.column_stack(basis)
    error = float(np.max(np.abs(result.T @ result - np.eye(column_count))))
    if error > 2.0e-12:
        raise RuntimeError(f"local basis lost orthogonality ({error:.3e})")
    return result


def _build_prolongation(
    aggregate: np.ndarray,
    enrichment: np.ndarray,
    fine_block_size: int,
) -> tuple[TentativeProlongation, np.ndarray]:
    """Build local tentative bases and exactly restrict the enrichment."""
    node_count = aggregate.size
    coarse_node_count = int(aggregate.max()) + 1
    coarse_block_size = enrichment.shape[1]
    blocks = np.zeros((node_count, fine_block_size, coarse_block_size), dtype=np.float64)
    for members in _members_by_aggregate(aggregate, coarse_node_count):
        local_rows = np.concatenate(
            [np.arange(node * fine_block_size, (node + 1) * fine_block_size, dtype=np.int64) for node in members]
        )
        local_basis = _orthonormal_completion(enrichment[local_rows], coarse_block_size)
        for local_node, node in enumerate(members):
            row = slice(local_node * fine_block_size, (local_node + 1) * fine_block_size)
            blocks[int(node)] = local_basis[row]
    aggregate_frozen = _frozen_array(aggregate, np.int64)
    blocks_frozen = _frozen_array(blocks, np.float64)
    content_sha256 = _hash_parts(
        "tentative-prolongation-v1",
        (
            ("aggregate", aggregate_frozen),
            ("blocks", blocks_frozen),
            ("coarse_node_count", coarse_node_count),
        ),
    )
    prolongation = TentativeProlongation(
        aggregate=aggregate_frozen,
        blocks=blocks_frozen,
        coarse_node_count=coarse_node_count,
        content_sha256=content_sha256,
    )
    coarse_enrichment = prolongation.restrict(enrichment)
    reproduced = prolongation.prolong(coarse_enrichment)
    scale = max(1.0, float(np.max(np.abs(enrichment))))
    if not np.allclose(reproduced, enrichment, rtol=0.0, atol=2.0e-12 * scale):
        raise RuntimeError("tentative prolongation does not reproduce the supplied enrichment")
    return prolongation, coarse_enrichment


def _galerkin(matrix: StaticBlockMatrix, prolongation: TentativeProlongation) -> StaticBlockMatrix:
    """Form a deterministic sparse Galerkin product ``P.T @ A @ P``."""
    coarse_count = prolongation.coarse_node_count
    coarse_size = prolongation.coarse_block_size
    accumulated: dict[tuple[int, int], np.ndarray] = {}
    for fine_row in range(matrix.block_row_count):
        coarse_row = int(prolongation.aggregate[fine_row])
        left = prolongation.blocks[fine_row].T
        for entry in range(int(matrix.row_offsets[fine_row]), int(matrix.row_offsets[fine_row + 1])):
            fine_column = int(matrix.column_indices[entry])
            coarse_column = int(prolongation.aggregate[fine_column])
            contribution = left @ matrix.values[entry] @ prolongation.blocks[fine_column]
            key = (coarse_row, coarse_column)
            if key in accumulated:
                accumulated[key] += contribution
            else:
                accumulated[key] = np.array(contribution, copy=True)

    canonical: dict[tuple[int, int], np.ndarray] = {}
    for row, column in sorted(accumulated):
        if row > column:
            continue
        forward = accumulated[(row, column)]
        if row == column:
            canonical[(row, row)] = 0.5 * (forward + forward.T)
            continue
        reverse = accumulated.get((column, row))
        if reverse is None:
            raise RuntimeError(f"Galerkin matrix is missing transpose block ({column}, {row})")
        block = 0.5 * (forward + reverse.T)
        canonical[(row, column)] = block
        canonical[(column, row)] = block.T
    for row in range(coarse_count):
        if (row, row) not in canonical:
            raise RuntimeError(f"Galerkin matrix is missing diagonal block {row}")
    coarse = _block_matrix_from_mapping(coarse_count, coarse_size, canonical)
    return coarse


def _build_smoother(matrix: StaticBlockMatrix, safety: float) -> BlockJacobiSmoother:
    """Build ``omega D^-1`` below the proven symmetric-smoothing limit."""
    if not math.isfinite(safety) or not 0.0 < safety < 1.0:
        raise ValueError("smoother_safety must be strictly between zero and one")
    diagonal = matrix.diagonal_blocks()
    inverse = np.empty_like(diagonal)
    inverse_lower = np.empty_like(diagonal)
    identity = np.eye(matrix.block_size, dtype=np.float64)
    for node, block in enumerate(diagonal):
        try:
            lower = np.linalg.cholesky(0.5 * (block + block.T))
        except np.linalg.LinAlgError as error:
            raise ValueError(f"diagonal block {node} is not positive definite") from error
        inverse_lower[node] = np.linalg.solve(lower, identity)
        inverse[node] = inverse_lower[node].T @ inverse_lower[node]

    row_bounds = np.zeros(matrix.block_row_count, dtype=np.float64)
    for row in range(matrix.block_row_count):
        for entry in range(int(matrix.row_offsets[row]), int(matrix.row_offsets[row + 1])):
            column = int(matrix.column_indices[entry])
            normalized = inverse_lower[row] @ matrix.values[entry] @ inverse_lower[column].T
            # Frobenius upper-bounds the block spectral norm, so its row sum
            # is conservative without an eigendecomposition.
            row_bounds[row] += float(np.linalg.norm(normalized, ord="fro"))
    upper_bound = float(np.max(row_bounds))
    if not math.isfinite(upper_bound) or upper_bound <= 0.0:
        raise RuntimeError("invalid normalized block-Jacobi spectral bound")
    # If rho bounds lambda_max(D^-1/2 A D^-1/2), then the symmetric
    # pre/post-smoothing contribution 2S-SAS is positive definite whenever
    # omega*rho < 2 for S=omega*D^-1.  ``safety`` is the registered fraction
    # of that full interval.  The classical 2/3 cap only decreases omega.
    omega = min(2.0 / 3.0, 2.0 * safety / upper_bound)
    inverse_frozen = _frozen_array(inverse, np.float64)
    content_sha256 = _hash_parts(
        "block-jacobi-smoother-v1",
        (
            ("inverse_diagonal", inverse_frozen),
            ("omega", omega),
            ("normalized_spectral_upper_bound", upper_bound),
        ),
    )
    return BlockJacobiSmoother(
        inverse_diagonal=inverse_frozen,
        omega=omega,
        normalized_spectral_upper_bound=upper_bound,
        content_sha256=content_sha256,
    )


def _make_level(
    matrix: StaticBlockMatrix,
    node_ids: np.ndarray,
    enrichment: np.ndarray,
    aggregate: np.ndarray | None,
    prolongation: TentativeProlongation | None,
    smoother: BlockJacobiSmoother | None,
) -> StaticMultigridLevel:
    node_ids_frozen = _frozen_array(node_ids, np.int64)
    enrichment_frozen = _frozen_array(enrichment, np.float64)
    aggregate_frozen = None if aggregate is None else _frozen_array(aggregate, np.int64)
    content_sha256 = _hash_parts(
        "static-multigrid-level-v1",
        (
            ("matrix_sha256", matrix.content_sha256),
            ("node_ids", node_ids_frozen),
            ("enrichment", enrichment_frozen),
            ("aggregate", aggregate_frozen),
            ("prolongation_sha256", None if prolongation is None else prolongation.content_sha256),
            ("smoother_sha256", None if smoother is None else smoother.content_sha256),
        ),
    )
    return StaticMultigridLevel(
        matrix=matrix,
        node_ids=node_ids_frozen,
        enrichment=enrichment_frozen,
        aggregate=aggregate_frozen,
        prolongation=prolongation,
        smoother=smoother,
        content_sha256=content_sha256,
    )


def _count_undirected_edges(matrix: StaticBlockMatrix) -> int:
    count = 0
    for row in range(matrix.block_row_count):
        columns = matrix.column_indices[int(matrix.row_offsets[row]) : int(matrix.row_offsets[row + 1])]
        count += int(np.count_nonzero(columns > row))
    return count


def _storage_accounting(
    levels: tuple[StaticMultigridLevel, ...],
    coarse_cholesky: np.ndarray,
    free_vertices: np.ndarray,
    rest_positions: np.ndarray,
    free_masses: np.ndarray,
) -> HierarchyStorage:
    root = levels[0].matrix
    matrix_blocks = sum(level.matrix.stored_block_count for level in levels)
    prolongation_blocks = sum(
        0 if level.prolongation is None else level.prolongation.fine_node_count for level in levels
    )
    smoother_blocks = sum(0 if level.smoother is None else level.matrix.block_row_count for level in levels)
    matrix_scalars = sum(level.matrix.values.size for level in levels)
    prolongation_scalars = sum(0 if level.prolongation is None else level.prolongation.blocks.size for level in levels)
    smoother_scalars = sum(0 if level.smoother is None else level.smoother.inverse_diagonal.size for level in levels)
    enrichment_scalars = sum(level.enrichment.size for level in levels)
    factor_scalars = int(coarse_cholesky.size)
    geometry_scalars = int(rest_positions.size + free_masses.size)
    index_scalars = sum(level.matrix.row_offsets.size + level.matrix.column_indices.size for level in levels)
    index_scalars += sum(level.node_ids.size for level in levels)
    index_scalars += sum(0 if level.aggregate is None else level.aggregate.size for level in levels)
    index_scalars += sum(0 if level.prolongation is None else level.prolongation.aggregate.size for level in levels)
    index_scalars += int(free_vertices.size)
    total_scalars = (
        matrix_scalars
        + prolongation_scalars
        + smoother_scalars
        + enrichment_scalars
        + factor_scalars
        + geometry_scalars
        + index_scalars
    )
    retained_arrays: list[np.ndarray] = [coarse_cholesky, free_vertices, rest_positions, free_masses]
    for level in levels:
        retained_arrays.extend(
            (
                level.matrix.row_offsets,
                level.matrix.column_indices,
                level.matrix.values,
                level.node_ids,
                level.enrichment,
            )
        )
        if level.aggregate is not None:
            retained_arrays.append(level.aggregate)
        if level.prolongation is not None:
            retained_arrays.extend((level.prolongation.aggregate, level.prolongation.blocks))
        if level.smoother is not None:
            retained_arrays.append(level.smoother.inverse_diagonal)
    total_bytes = sum(array.nbytes for array in retained_arrays)
    if total_bytes != total_scalars * np.dtype(np.float64).itemsize:
        raise RuntimeError("hierarchy storage accounting does not match retained array payloads")
    parts = (
        ("fine_node_count", root.block_row_count),
        ("fine_undirected_edge_count", _count_undirected_edges(root)),
        ("level_count", len(levels)),
        ("matrix_block_count", matrix_blocks),
        ("prolongation_block_count", prolongation_blocks),
        ("smoother_block_count", smoother_blocks),
        ("matrix_scalar_count", matrix_scalars),
        ("prolongation_scalar_count", prolongation_scalars),
        ("smoother_scalar_count", smoother_scalars),
        ("enrichment_scalar_count", enrichment_scalars),
        ("factor_scalar_count", factor_scalars),
        ("geometry_scalar_count", geometry_scalars),
        ("index_scalar_count", index_scalars),
        ("dense_matrix_scalar_count_excluding_coarse_factor", 0),
        ("total_scalar_count", total_scalars),
        ("total_bytes", total_bytes),
    )
    content_sha256 = _hash_parts("hierarchy-storage-v1", parts)
    return HierarchyStorage(
        fine_node_count=root.block_row_count,
        fine_undirected_edge_count=_count_undirected_edges(root),
        level_count=len(levels),
        matrix_block_count=matrix_blocks,
        prolongation_block_count=prolongation_blocks,
        smoother_block_count=smoother_blocks,
        matrix_scalar_count=matrix_scalars,
        prolongation_scalar_count=prolongation_scalars,
        smoother_scalar_count=smoother_scalars,
        enrichment_scalar_count=enrichment_scalars,
        factor_scalar_count=factor_scalars,
        geometry_scalar_count=geometry_scalars,
        index_scalar_count=index_scalars,
        dense_matrix_scalar_count_excluding_coarse_factor=0,
        total_scalar_count=total_scalars,
        total_bytes=total_bytes,
        content_sha256=content_sha256,
    )


def build_static_multigrid(
    fine_matrix: StaticBlockMatrix | np.ndarray,
    rest_positions: np.ndarray,
    free_vertices: np.ndarray,
    *,
    free_masses: np.ndarray | None = None,
    mode_kind: str = "rigid",
    target_aggregate_size: int = 4,
    minimum_aggregate_size: int = 3,
    coarse_node_limit: int = 4,
    maximum_levels: int = 8,
    pre_smooth_steps: int = 1,
    post_smooth_steps: int = 1,
    smoother_safety: float = 0.9,
    static_model_sha256: str | None = None,
) -> StaticMultigridHierarchy:
    """Build a deterministic static Galerkin hierarchy.

    Args:
        fine_matrix: Fine SPD block matrix with three displacement DOFs per
            free vertex.
        rest_positions: Rest positions for all vertices [m], shape ``(V, 3)``.
        free_vertices: Vertex IDs, in the same order as fine matrix blocks.
        free_masses: Optional positive masses in fine-block order. Unit masses
            are used when omitted.
        mode_kind: ``"rigid"`` for translations plus rotations, or
            ``"translation"`` for the ablation coarse space.
        target_aggregate_size: Strength-greedy target number of graph nodes.
        minimum_aggregate_size: Minimum first-level vertex count. Rigid
            enrichment requires at least three.
        coarse_node_limit: Stop aggregation at or below this graph-node count.
        maximum_levels: Maximum number of retained matrix levels.
        pre_smooth_steps: Damped block-Jacobi sweeps before coarse correction.
        post_smooth_steps: Matching sweeps after coarse correction.
        smoother_safety: Fraction of the full SPD damping interval established
            by the block-Frobenius row bound.
        static_model_sha256: Optional digest of the static operator/model
            inputs used to assemble ``fine_matrix``.
    """
    if mode_kind not in ("rigid", "translation"):
        raise ValueError("mode_kind must be 'rigid' or 'translation'")
    if target_aggregate_size < 2:
        raise ValueError("target_aggregate_size must be at least two")
    if minimum_aggregate_size < 2 or minimum_aggregate_size > target_aggregate_size:
        raise ValueError("minimum_aggregate_size must be in [2, target_aggregate_size]")
    if mode_kind == "rigid" and minimum_aggregate_size < 3:
        raise ValueError("rigid enrichment requires minimum_aggregate_size >= 3")
    if coarse_node_limit < 1 or maximum_levels < 2:
        raise ValueError("coarse_node_limit must be positive and maximum_levels must be at least two")
    if pre_smooth_steps < 1 or post_smooth_steps != pre_smooth_steps:
        raise ValueError("symmetric V-cycles require equal positive pre_smooth_steps and post_smooth_steps")
    if static_model_sha256 is not None:
        if (
            not isinstance(static_model_sha256, str)
            or len(static_model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in static_model_sha256)
        ):
            raise ValueError("static_model_sha256 must be a lowercase SHA-256 digest")

    rest = np.asarray(rest_positions, dtype=np.float64)
    vertices = np.asarray(free_vertices)
    if rest.ndim != 2 or rest.shape[1] != 3 or rest.shape[0] == 0 or not np.isfinite(rest).all():
        raise ValueError("rest_positions must be a finite non-empty (V, 3) array")
    if vertices.ndim != 1 or vertices.size == 0 or vertices.dtype.kind not in "iu":
        raise ValueError("free_vertices must be a non-empty one-dimensional integer array")
    vertices = np.asarray(vertices, dtype=np.int64)
    if np.any(vertices < 0) or np.any(vertices >= rest.shape[0]) or np.unique(vertices).size != vertices.size:
        raise ValueError("free_vertices must contain unique valid rest-position indices")
    matrix = fine_matrix if isinstance(fine_matrix, StaticBlockMatrix) else StaticBlockMatrix.from_dense(fine_matrix)
    if matrix.block_size != 3 or matrix.block_row_count != vertices.size:
        raise ValueError("fine_matrix must contain one 3x3 block per free vertex")

    if free_masses is None:
        masses = np.ones(vertices.size, dtype=np.float64)
    else:
        masses = np.asarray(free_masses, dtype=np.float64)
        if masses.shape != (vertices.size,) or not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("free_masses must contain one finite positive value per free vertex")
    rest_free = np.array(rest[vertices], dtype=np.float64, copy=True)
    enrichment = rigid_enrichment(rest_free, masses) if mode_kind == "rigid" else translation_enrichment(vertices.size)
    current_matrix = matrix
    current_ids = np.array(vertices, dtype=np.int64, copy=True)
    current_enrichment = enrichment
    levels: list[StaticMultigridLevel] = []

    for level_index in range(maximum_levels - 1):
        if current_matrix.block_row_count <= coarse_node_limit:
            break
        aggregate = _connected_aggregate(
            current_matrix,
            current_ids,
            target_size=target_aggregate_size,
            minimum_size=minimum_aggregate_size,
            first_level_rest=rest_free if level_index == 0 and mode_kind == "rigid" else None,
        )
        coarse_count = int(aggregate.max()) + 1
        if coarse_count >= current_matrix.block_row_count:
            raise ValueError("aggregation did not reduce the graph")
        coarse_block_size = current_enrichment.shape[1]
        if coarse_count * coarse_block_size >= current_matrix.scalar_size:
            raise ValueError("coarse enrichment does not strictly reduce scalar degrees of freedom")
        prolongation, coarse_enrichment = _build_prolongation(
            aggregate,
            current_enrichment,
            current_matrix.block_size,
        )
        smoother = _build_smoother(current_matrix, smoother_safety)
        levels.append(
            _make_level(
                current_matrix,
                current_ids,
                current_enrichment,
                aggregate,
                prolongation,
                smoother,
            )
        )
        coarse_matrix = _galerkin(current_matrix, prolongation)
        coarse_ids = np.empty(coarse_count, dtype=np.int64)
        for aggregate_id, members in enumerate(_members_by_aggregate(aggregate, coarse_count)):
            coarse_ids[aggregate_id] = min(current_ids[members])
        current_matrix = coarse_matrix
        current_ids = coarse_ids
        current_enrichment = coarse_enrichment

    if current_matrix.block_row_count > coarse_node_limit:
        raise ValueError(
            f"maximum_levels={maximum_levels} exhausted at {current_matrix.block_row_count} nodes, "
            f"above coarse_node_limit={coarse_node_limit}"
        )
    levels.append(_make_level(current_matrix, current_ids, current_enrichment, None, None, None))
    if len(levels) < 2:
        raise ValueError("hierarchy requires at least one strict coarsening")
    coarse_cholesky = _frozen_array(np.linalg.cholesky(current_matrix.to_dense()), np.float64)
    levels_tuple = tuple(levels)
    vertices_frozen = _frozen_array(vertices, np.int64)
    rest_frozen = _frozen_array(rest_free, np.float64)
    masses_frozen = _frozen_array(masses, np.float64)
    storage = _storage_accounting(levels_tuple, coarse_cholesky, vertices_frozen, rest_frozen, masses_frozen)
    content_sha256 = _hash_parts(
        "static-multigrid-hierarchy-v1",
        (
            ("free_vertices", vertices_frozen),
            ("rest_positions", rest_frozen),
            ("free_masses", masses_frozen),
            ("solver_contract", SPECTRAL_FREE_CONTRACT),
            ("mode_kind", mode_kind),
            ("target_aggregate_size", target_aggregate_size),
            ("minimum_aggregate_size", minimum_aggregate_size),
            ("coarse_node_limit", coarse_node_limit),
            ("maximum_levels", maximum_levels),
            ("pre_smooth_steps", pre_smooth_steps),
            ("post_smooth_steps", post_smooth_steps),
            ("smoother_safety", smoother_safety),
            ("static_model_sha256", static_model_sha256),
            ("coarse_cholesky", coarse_cholesky),
            ("storage_sha256", storage.content_sha256),
            *((f"level_{index}_sha256", level.content_sha256) for index, level in enumerate(levels_tuple)),
        ),
    )
    return StaticMultigridHierarchy(
        levels=levels_tuple,
        free_vertices=vertices_frozen,
        rest_positions=rest_frozen,
        free_masses=masses_frozen,
        solver_contract=SPECTRAL_FREE_CONTRACT,
        mode_kind=mode_kind,
        target_aggregate_size=target_aggregate_size,
        minimum_aggregate_size=minimum_aggregate_size,
        coarse_node_limit=coarse_node_limit,
        maximum_levels=maximum_levels,
        pre_smooth_steps=pre_smooth_steps,
        post_smooth_steps=post_smooth_steps,
        smoother_safety=smoother_safety,
        static_model_sha256=static_model_sha256,
        coarse_cholesky=coarse_cholesky,
        storage=storage,
        content_sha256=content_sha256,
    )


def build_stable_nh_rest_multigrid(
    operator: MatrixFreeStableNHOperator,
    rest_positions: np.ndarray,
    *,
    mode_kind: str = "rigid",
    target_aggregate_size: int = 4,
    minimum_aggregate_size: int = 3,
    coarse_node_limit: int = 4,
    maximum_levels: int = 8,
    pre_smooth_steps: int = 1,
    post_smooth_steps: int = 1,
    smoother_safety: float = 0.9,
) -> StaticMultigridHierarchy:
    """Build the spectral-free MG hierarchy from the stable-NH rest tangent.

    This is the production-shaped research entrypoint: fine blocks are
    assembled directly from mass and tet-local Gauss--Newton terms, never by
    dense materialization or unit-vector probing.

    Args:
        operator: Frozen matrix-free stable-Neo-Hookean operator.
        rest_positions: Rest positions for all full vertices [m].
        mode_kind: ``"rigid"`` or the ``"translation"`` ablation.
        target_aggregate_size: Strength-greedy target graph-node count.
        minimum_aggregate_size: Minimum nodes per aggregate.
        coarse_node_limit: Coarsest target graph-node count.
        maximum_levels: Maximum retained matrix levels.
        pre_smooth_steps: Block-Jacobi pre-smoothing sweeps.
        post_smooth_steps: Matching post-smoothing sweeps.
        smoother_safety: Fraction of the full SPD damping interval established
            by the Cholesky-whitened block-Frobenius row bound.
    """
    matrix = assemble_stable_nh_rest_block_matrix(operator, rest_positions)
    model_sha256 = stable_nh_static_model_digest(operator, rest_positions)
    return build_static_multigrid(
        matrix,
        rest_positions,
        operator.free,
        free_masses=operator.mass[operator.free],
        mode_kind=mode_kind,
        target_aggregate_size=target_aggregate_size,
        minimum_aggregate_size=minimum_aggregate_size,
        coarse_node_limit=coarse_node_limit,
        maximum_levels=maximum_levels,
        pre_smooth_steps=pre_smooth_steps,
        post_smooth_steps=post_smooth_steps,
        smoother_safety=smoother_safety,
        static_model_sha256=model_sha256,
    )


@dataclasses.dataclass(slots=True)
class _MutableWork:
    level_visits: list[int]
    matrix_block_products: int = 0
    smoother_block_solves: int = 0
    restriction_block_products: int = 0
    prolongation_block_products: int = 0
    coarsest_factor_solves: int = 0


def _v_cycle_recursive(
    hierarchy: StaticMultigridHierarchy,
    level_index: int,
    rhs: np.ndarray,
    work: _MutableWork,
) -> np.ndarray:
    level = hierarchy.levels[level_index]
    width = rhs.shape[1]
    work.level_visits[level_index] += 1
    if level_index == len(hierarchy.levels) - 1:
        intermediate = np.linalg.solve(hierarchy.coarse_cholesky, rhs)
        work.coarsest_factor_solves += width
        return np.linalg.solve(hierarchy.coarse_cholesky.T, intermediate)

    if level.smoother is None or level.prolongation is None:
        raise RuntimeError("non-coarsest level is missing smoother or prolongation")
    correction = np.zeros_like(rhs)

    def smooth() -> None:
        nonlocal correction
        residual = rhs - level.matrix.matmul(correction)
        work.matrix_block_products += level.matrix.stored_block_count * width
        correction += level.smoother.apply(residual)
        work.smoother_block_solves += level.matrix.block_row_count * width

    for _ in range(hierarchy.pre_smooth_steps):
        smooth()
    residual = rhs - level.matrix.matmul(correction)
    work.matrix_block_products += level.matrix.stored_block_count * width
    coarse_rhs = level.prolongation.restrict(residual)
    work.restriction_block_products += level.prolongation.fine_node_count * width
    coarse_correction = _v_cycle_recursive(hierarchy, level_index + 1, coarse_rhs, work)
    correction += level.prolongation.prolong(coarse_correction)
    work.prolongation_block_products += level.prolongation.fine_node_count * width
    for _ in range(hierarchy.post_smooth_steps):
        smooth()
    return correction


def apply_v_cycle(hierarchy: StaticMultigridHierarchy, rhs: np.ndarray) -> VCycleResult:
    """Apply one symmetric V-cycle and return immutable work evidence."""
    if not isinstance(hierarchy, StaticMultigridHierarchy):
        raise TypeError("hierarchy must be a StaticMultigridHierarchy")
    canonical_rhs, was_vector = _as_rhs(rhs, hierarchy.levels[0].matrix.scalar_size, name="rhs")
    work = _MutableWork(level_visits=[0] * len(hierarchy.levels))
    correction = _v_cycle_recursive(hierarchy, 0, canonical_rhs, work)
    correction_output = correction[:, 0] if was_vector else correction
    correction_frozen = _frozen_array(correction_output, np.float64)
    rhs_frozen = _frozen_array(canonical_rhs, np.float64)
    rhs_sha256 = _hash_parts("v-cycle-rhs-v1", (("rhs", rhs_frozen),))
    result_sha256 = _hash_parts("v-cycle-correction-v1", (("correction", correction_frozen),))
    record_parts = (
        ("hierarchy_sha256", hierarchy.content_sha256),
        ("rhs_sha256", rhs_sha256),
        ("result_sha256", result_sha256),
        ("rhs_count", canonical_rhs.shape[1]),
        ("level_visits", _frozen_array(work.level_visits, np.int64)),
        ("matrix_block_products", work.matrix_block_products),
        ("smoother_block_solves", work.smoother_block_solves),
        ("restriction_block_products", work.restriction_block_products),
        ("prolongation_block_products", work.prolongation_block_products),
        ("coarsest_factor_solves", work.coarsest_factor_solves),
    )
    record_sha256 = _hash_parts("v-cycle-work-record-v1", record_parts)
    record = VCycleWorkRecord(
        hierarchy_sha256=hierarchy.content_sha256,
        rhs_sha256=rhs_sha256,
        result_sha256=result_sha256,
        rhs_count=canonical_rhs.shape[1],
        level_visits=tuple(work.level_visits),
        matrix_block_products=work.matrix_block_products,
        smoother_block_solves=work.smoother_block_solves,
        restriction_block_products=work.restriction_block_products,
        prolongation_block_products=work.prolongation_block_products,
        coarsest_factor_solves=work.coarsest_factor_solves,
        content_sha256=record_sha256,
    )
    result_hash = _hash_parts(
        "v-cycle-result-v1",
        (("correction_sha256", result_sha256), ("work_sha256", record_sha256)),
    )
    return VCycleResult(correction=correction_frozen, work=record, content_sha256=result_hash)


def solve_pcg(
    matrix: StaticBlockMatrix,
    rhs: np.ndarray,
    *,
    hierarchy: StaticMultigridHierarchy | None = None,
    jacobi: BlockJacobiSmoother | None = None,
    relative_tolerance: float = 1.0e-10,
    maximum_iterations: int = 256,
) -> PCGSolveResult:
    """Solve one SPD system with either V-cycle or block-Jacobi PCG.

    Exactly one of ``hierarchy`` and ``jacobi`` must be supplied.
    """
    if (hierarchy is None) == (jacobi is None):
        raise ValueError("supply exactly one of hierarchy and jacobi")
    if hierarchy is not None and hierarchy.levels[0].matrix.content_sha256 != matrix.content_sha256:
        raise ValueError("hierarchy root matrix does not match matrix")
    if not math.isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be strictly between zero and one")
    if isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, numbers.Integral):
        raise ValueError("maximum_iterations must be an exact positive integer")
    maximum_iterations = int(maximum_iterations)
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be an exact positive integer")
    canonical_rhs, was_vector = _as_rhs(rhs, matrix.scalar_size, name="rhs")
    if not was_vector:
        raise ValueError("solve_pcg accepts exactly one right-hand side")
    vector = canonical_rhs[:, 0]
    rhs_norm = _scaled_norm(vector, name="rhs")
    rhs_scale = float(np.max(np.abs(vector)))
    working_vector = vector.copy() if rhs_scale == 0.0 else vector / rhs_scale
    working_rhs_norm = _scaled_norm(working_vector, name="scaled rhs")
    inner_products = 1
    operator_applications = 0
    preconditioner_applications = 0
    vector_updates = 0
    solution = np.zeros_like(working_vector)
    residual = working_vector.copy()
    residual_norms = [rhs_norm]
    preconditioner_sha256 = hierarchy.content_sha256 if hierarchy is not None else jacobi.content_sha256
    if rhs_norm == 0.0:
        converged = True
        iterations = 0
    else:
        preconditioned = (
            apply_v_cycle(hierarchy, residual).correction if hierarchy is not None else jacobi.apply(residual)
        )
        preconditioned = _require_finite(np.asarray(preconditioned), name="initial preconditioner")
        preconditioner_applications += 1
        direction = np.array(preconditioned, copy=True)
        residual_product = _finite_dot(residual, preconditioned, name="initial preconditioned residual product")
        inner_products += 1
        if residual_product <= 0.0:
            raise RuntimeError("preconditioner is not positive on the initial residual")
        converged = False
        iterations = 0
        for iteration in range(1, maximum_iterations + 1):
            matrix_direction = _require_finite(matrix.matmul(direction), name="matrix application")
            operator_applications += 1
            denominator = _finite_dot(direction, matrix_direction, name="PCG curvature")
            inner_products += 1
            if denominator <= 0.0:
                raise RuntimeError("matrix lost positive definiteness during PCG")
            alpha = residual_product / denominator
            if not math.isfinite(alpha):
                raise FloatingPointError("PCG alpha is non-finite")
            with np.errstate(over="ignore", invalid="ignore"):
                solution = solution + alpha * direction
                residual = residual - alpha * matrix_direction
            _require_finite(solution, name="PCG solution update")
            _require_finite(residual, name="PCG residual update")
            vector_updates += 2
            working_norm = _scaled_norm(residual, name="PCG recursive residual")
            norm = _finite_scale(working_norm, rhs_scale, name="PCG recursive residual norm")
            inner_products += 1
            residual_norms.append(norm)
            iterations = iteration
            if working_norm <= relative_tolerance * working_rhs_norm:
                converged = True
                break
            new_preconditioned = (
                apply_v_cycle(hierarchy, residual).correction if hierarchy is not None else jacobi.apply(residual)
            )
            new_preconditioned = _require_finite(np.asarray(new_preconditioned), name="PCG preconditioner")
            preconditioner_applications += 1
            new_product = _finite_dot(residual, new_preconditioned, name="preconditioned residual product")
            inner_products += 1
            if new_product <= 0.0:
                raise RuntimeError("preconditioner lost positivity during PCG")
            beta = new_product / residual_product
            if not math.isfinite(beta):
                raise FloatingPointError("PCG beta is non-finite")
            with np.errstate(over="ignore", invalid="ignore"):
                direction = new_preconditioned + beta * direction
            _require_finite(direction, name="PCG direction update")
            vector_updates += 1
            preconditioned = new_preconditioned
            residual_product = new_product
    with np.errstate(over="ignore", invalid="ignore"):
        solution = solution if rhs_scale == 0.0 else solution * rhs_scale
    _require_finite(solution, name="rescaled PCG solution")
    matrix_solution = _require_finite(matrix.matmul(solution), name="true-residual matrix application")
    with np.errstate(over="ignore", invalid="ignore"):
        true_residual = vector - matrix_solution
    _require_finite(true_residual, name="true residual")
    operator_applications += 1
    true_residual_norm = _scaled_norm(true_residual, name="true residual")
    inner_products += 1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        true_relative_residual = true_residual_norm / rhs_norm if rhs_norm > 0.0 else 0.0
    if not math.isfinite(true_relative_residual):
        raise FloatingPointError("true relative residual is not representable in float64")
    converged = converged and true_relative_residual <= relative_tolerance
    solution_frozen = _frozen_array(solution, np.float64)
    rhs_sha256 = _hash_parts("pcg-rhs-v1", (("rhs", _frozen_array(vector, np.float64)),))
    content_sha256 = _hash_parts(
        "pcg-solve-result-v1",
        (
            ("solution", solution_frozen),
            ("residual_norms", _frozen_array(residual_norms, np.float64)),
            ("iteration_count", iterations),
            ("converged", converged),
            ("matrix_sha256", matrix.content_sha256),
            ("preconditioner_sha256", preconditioner_sha256),
            ("rhs_sha256", rhs_sha256),
            ("relative_tolerance", relative_tolerance),
            ("maximum_iterations", maximum_iterations),
            ("operator_applications", operator_applications),
            ("preconditioner_applications", preconditioner_applications),
            ("inner_products", inner_products),
            ("vector_updates", vector_updates),
            ("true_residual_norm", true_residual_norm),
            ("true_relative_residual", true_relative_residual),
        ),
    )
    return PCGSolveResult(
        solution=solution_frozen,
        residual_norms=tuple(residual_norms),
        iteration_count=iterations,
        converged=converged,
        matrix_sha256=matrix.content_sha256,
        preconditioner_sha256=preconditioner_sha256,
        rhs_sha256=rhs_sha256,
        relative_tolerance=relative_tolerance,
        maximum_iterations=maximum_iterations,
        operator_applications=operator_applications,
        preconditioner_applications=preconditioner_applications,
        inner_products=inner_products,
        vector_updates=vector_updates,
        true_residual_norm=true_residual_norm,
        true_relative_residual=true_relative_residual,
        content_sha256=content_sha256,
    )


def build_block_jacobi(matrix: StaticBlockMatrix, *, safety: float = 0.9) -> BlockJacobiSmoother:
    """Build the deterministic block-Jacobi PCG ablation preconditioner."""
    return _build_smoother(matrix, safety)
