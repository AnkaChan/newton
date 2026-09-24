# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental incremental hexahedral fusion with a CPU sparse-solve bridge.

Eight Gauss samples per cell fit a single target increment to shared-corner
displacements. This is a reconstruction layer, not the physical implicit-Euler
objective. CPU and CUDA tensors retain their device and precision; a cached
CPU sparse factorization supplies first-order forward and adjoint solves.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch  # noqa: TID253 -- This opt-in autograd layer explicitly requires Torch.

from .hex_energy import hex_gauss_quadrature
from .pardiso import PardisoFactor

if TYPE_CHECKING:
    from .data import VoxelGridData

__all__ = ["HexFusion"]


def _columns(values: np.ndarray) -> np.ndarray:
    """Pack batch and world coordinates as sparse-solve right-hand sides."""
    return np.asfortranarray(values.transpose(1, 0, 2).reshape(values.shape[1], values.shape[0] * values.shape[2]))


def _batch(columns: np.ndarray, batch_count: int) -> np.ndarray:
    """Restore batch-first world vectors after a sparse multiplication."""
    return columns.reshape(columns.shape[0], batch_count, 3).transpose(1, 0, 2)


class _FusionSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fusion, base_positions, world_axis_increments, fixed_positions):
        device = base_positions.device
        fixed, free = fusion._indices(device)
        increments = world_axis_increments.detach().cpu().contiguous().numpy()
        batch_count = base_positions.shape[0]
        target_rows = increments.swapaxes(-1, -2).reshape(batch_count, 3 * fusion.cell_count, 3)
        # The solve needs only targets and boundary displacement, not all base
        # coordinates. Keep the full position array on its original device.
        fixed_delta = (fixed_positions - base_positions[:, fixed]).detach().cpu().contiguous().numpy()
        rhs = fusion._target_operator @ _columns(target_rows) - fusion._fixed_coupling @ _columns(fixed_delta)
        free_delta = fusion._solve(rhs)
        delta = torch.from_numpy(np.ascontiguousarray(_batch(free_delta, batch_count))).to(device=device)
        result = base_positions.clone()
        result[:, free] += delta
        # Assign the supplied values directly rather than adding a cancellation.
        result[:, fixed] = fixed_positions
        ctx.fusion = fusion
        ctx.batch_count = batch_count
        ctx.device = device
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient_output):
        fusion = ctx.fusion
        batch_count = ctx.batch_count
        fixed, free = fusion._indices(ctx.device)
        free_gradient = gradient_output[:, free].detach().cpu().contiguous().numpy()
        adjoint = fusion._solve(_columns(free_gradient), transpose=True)
        boundary_transfer = _batch(fusion._fixed_coupling.T @ adjoint, batch_count)
        target_rows = _batch(fusion._target_operator.T @ adjoint, batch_count)
        gradient_targets = target_rows.reshape(batch_count, fusion.cell_count, 3, 3).swapaxes(-1, -2)
        boundary_transfer = torch.from_numpy(np.ascontiguousarray(boundary_transfer)).to(device=ctx.device)
        gradient_base = gradient_output.clone()
        gradient_base[:, fixed] = boundary_transfer
        gradient_fixed = gradient_output[:, fixed] - boundary_transfer
        return (
            None,
            gradient_base,
            torch.from_numpy(np.ascontiguousarray(gradient_targets)).to(device=ctx.device),
            gradient_fixed,
        )


class HexFusion:
    """Fuse cell gradient increments using fixed full-quadrature hex operators.

    Experimental: fixed cubic rest cells and first-order autograd. Inputs and
    outputs may reside on CPU or CUDA. Sparse factorization and solves remain
    on CPU in the requested precision; the bridge transfers targets, boundary
    displacements, free-corner cotangents, and solved increments/adjoints.
    The baseline requires a connected material grid and at least one fixed
    vertex. An unclamped translation gauge is not supplied. Topology, weights,
    and the sparse factorization remain fixed after construction and are not
    differentiated. Gradients through targets, base positions, and prescribed
    positions remain available across the device transfers. The CPU bridge
    performs synchronous transfers and does not support CUDA graph capture.

    The minimized objective is the cell-weighted average of
    ``||grad(delta_position) - world_axis_increment||^2`` over all eight Gauss
    points. Fixed displacements are prescribed positions minus base positions.
    The result adds the solved increment to the base, so a zero target exactly
    preserves a warped base when its fixed positions are already satisfied.
    A unique fit does not imply that one target per cell can express every
    possible corner update. Non-inversion and physical descent are not enforced.

    Args:
        rest: Cubic material grid with shared rest corners [m] and cell topology.
        fixed_indices: Unique fixed corner IDs in the order used by
            ``fixed_positions``. At least one ID is required.
        cell_weights: Optional fixed positive weights, shape [C]. Physical
            reconstruction weights should include stiffness [J/m^3] times rest
            cell volume [m^3]. Quadrature averages are included internally;
            volume must not be multiplied a second time. Defaults to unit
            weights for uniformly weighted fitting. Weights are not learnable.
        dtype: Working tensor precision. Float32 uses single-precision sparse
            operators, factorization, forward solves, and adjoint solves.
            Float64 is also supported for reference checks.

    Raises:
        ValueError: If topology, weights, constraints, or precision are invalid.
    """

    def __init__(
        self,
        rest: VoxelGridData,
        fixed_indices,
        *,
        cell_weights=None,
        dtype: torch.dtype = torch.float32,
    ):
        from scipy import sparse

        if dtype not in (torch.float32, torch.float64):
            raise ValueError("HexFusion supports only torch.float32 and torch.float64")
        self.dtype = dtype
        self._numpy_dtype = np.dtype(np.float32 if dtype == torch.float32 else np.float64)
        positions = np.asarray(rest.corner_rest_positions)
        cells = np.asarray(rest.cell_corner_indices)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("rest corner positions must have shape [P, 3] with P > 0")
        if cells.ndim != 2 or cells.shape[1] != 8 or len(cells) == 0 or cells.dtype.kind not in "iu":
            raise ValueError("rest cell corners must have integer shape [C, 8] with C > 0")
        self.corner_count = len(positions)
        self.cell_count = len(cells)
        if np.any(cells < 0) or np.any(cells >= self.corner_count):
            raise ValueError("rest cell corner IDs are out of range")
        if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
            raise ValueError("each rest hexahedron must have eight distinct corners")
        if not math.isfinite(rest.cell_size) or rest.cell_size <= 0:
            raise ValueError("rest cell size must be positive and finite")

        if isinstance(fixed_indices, torch.Tensor):
            if fixed_indices.device.type != "cpu":
                raise ValueError("fixed_indices must be on the CPU")
            fixed_indices = fixed_indices.detach().numpy()
        fixed = np.asarray(fixed_indices)
        if fixed.ndim != 1 or fixed.size == 0 or fixed.dtype.kind not in "iu":
            raise ValueError(
                "fixed_indices must contain at least one integer corner ID; unclamped fusion is unsupported"
            )
        if np.any(fixed < 0) or np.any(fixed >= self.corner_count) or len(np.unique(fixed)) != len(fixed):
            raise ValueError("fixed_indices must contain unique in-range corner IDs")
        self._fixed = fixed.astype(np.int64, copy=True)
        free_mask = np.ones(self.corner_count, dtype=bool)
        free_mask[self._fixed] = False
        self._free = np.flatnonzero(free_mask)
        self._device_indices = {}

        if isinstance(cell_weights, torch.Tensor):
            if cell_weights.requires_grad or cell_weights.device.type != "cpu":
                raise ValueError("cell_weights must be fixed CPU values without gradients")
            cell_weights = cell_weights.detach().numpy()
        weights = (
            np.ones(self.cell_count, dtype=self._numpy_dtype)
            if cell_weights is None
            else np.asarray(cell_weights, dtype=self._numpy_dtype)
        )
        if weights.shape != (self.cell_count,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("cell_weights must have shape [C] with finite positive values")

        quadrature = hex_gauss_quadrature(rest.cell_size, dtype=self._numpy_dtype)
        gradients = np.asarray(quadrature.shape_gradients, dtype=self._numpy_dtype)
        quadrature_weights = np.asarray(quadrature.weights, dtype=self._numpy_dtype)
        quadrature_weights = quadrature_weights / quadrature_weights.sum(dtype=self._numpy_dtype)
        row_count = self.cell_count * 8 * 3
        rows = np.repeat(np.arange(row_count), 8)
        columns = np.broadcast_to(cells[:, None, None, :], (self.cell_count, 8, 3, 8)).reshape(-1)
        values = np.broadcast_to(gradients.transpose(0, 2, 1)[None], (self.cell_count, 8, 3, 8)).reshape(-1)
        gradient_operator = sparse.coo_matrix(
            (values, (rows, columns)), shape=(row_count, self.corner_count), dtype=self._numpy_dtype
        ).tocsr()
        row_weights = np.repeat((weights[:, None] * quadrature_weights[None]).reshape(-1), 3)
        stiffness = (gradient_operator.T @ gradient_operator.multiply(row_weights[:, None])).tocsc()
        target_columns = np.broadcast_to(
            np.arange(self.cell_count * 3).reshape(self.cell_count, 1, 3), (self.cell_count, 8, 3)
        ).reshape(-1)
        weighted_repeat = sparse.coo_matrix(
            (row_weights, (np.arange(row_count), target_columns)),
            shape=(row_count, self.cell_count * 3),
            dtype=self._numpy_dtype,
        ).tocsr()
        target_operator = (gradient_operator.T @ weighted_repeat).tocsr()
        self._target_operator = target_operator[self._free].tocsr()
        self._fixed_coupling = stiffness[self._free][:, self._fixed].tocsr()
        self._free_stiffness = stiffness[self._free][:, self._free].tocsc()
        self._factor = None
        if len(self._free):
            try:
                self._factor = PardisoFactor(self._free_stiffness)
            except RuntimeError as error:
                raise ValueError(f"fusion factorization failed: {error}") from error
            if self._factor.dtype != self._numpy_dtype:
                raise RuntimeError("sparse factorization changed the requested working precision")

    @property
    def fixed_indices(self) -> torch.Tensor:
        """Return a copy of the fixed corner IDs in prescribed-position order."""
        return torch.from_numpy(self._fixed.copy())

    @property
    def free_indices(self) -> torch.Tensor:
        """Return a copy of the free corner IDs in reconstruction order."""
        return torch.from_numpy(self._free.copy())

    def _solve(self, right_hand_side: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        if self._factor is None:
            return np.zeros_like(right_hand_side)
        return self._factor.solve(np.asfortranarray(right_hand_side), transpose=transpose)

    def _indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache private index tensors without exposing mutable operator topology."""
        if device not in self._device_indices:
            self._device_indices[device] = (
                torch.tensor(self._fixed, dtype=torch.long, device=device),
                torch.tensor(self._free, dtype=torch.long, device=device),
            )
        return self._device_indices[device]

    def fuse(
        self,
        base_positions: torch.Tensor,
        world_axis_increments: torch.Tensor,
        fixed_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return a differentiable compatible update of shared corner positions.

        Args:
            base_positions: Current world corner positions [m], shape [B, P, 3].
            world_axis_increments: Dimensionless world deformation-gradient
                increments, shape [B, C, 3, 3], with material axes in columns.
                Any frame detachment must happen before constructing this input.
            fixed_positions: Prescribed world positions [m], shape [B, K, 3],
                following ``fixed_indices`` order. These values always win.

        Returns:
            World positions [m], shape [B, P, 3], in the inputs' common device
            and chosen precision. First-order gradients are available for all
            three inputs; the fixed CPU factorization is not differentiated.

        Raises:
            TypeError: If an input is not a tensor in the chosen precision.
            ValueError: If inputs have differing/unsupported devices or invalid
                shapes. CPU and CUDA tensors are supported.
        """
        tensors = (
            ("base_positions", base_positions),
            ("world_axis_increments", world_axis_increments),
            ("fixed_positions", fixed_positions),
        )
        for name, tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != self.dtype:
                raise TypeError(f"{name} must be a tensor with dtype {self.dtype}")
            if tensor.device != base_positions.device:
                raise ValueError("all fusion inputs must use the same device")
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError("fusion inputs must use CPU or CUDA devices")
        if (
            base_positions.ndim != 3
            or base_positions.shape[1:] != (self.corner_count, 3)
            or base_positions.shape[0] == 0
        ):
            raise ValueError("base_positions must have shape [B, P, 3] with B > 0")
        batch_count = base_positions.shape[0]
        if world_axis_increments.shape != (batch_count, self.cell_count, 3, 3):
            raise ValueError("world_axis_increments must have shape [B, C, 3, 3]")
        if fixed_positions.shape != (batch_count, len(self._fixed), 3):
            raise ValueError("fixed_positions must have shape [B, K, 3]")
        return _FusionSolve.apply(self, base_positions, world_axis_increments, fixed_positions)
