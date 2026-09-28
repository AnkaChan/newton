# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental incremental hexahedral fusion with a CPU sparse-solve bridge.

Eight Gauss samples per cell fit a per-cell target increment to shared-corner
displacements. The target is either one 3x3 gradient increment copied to all
eight Gauss points (three affine modes) or seven mode vectors that also fix
the bilinear and trilinear warping of the cell, so the per-cell fit is exact
and only shared-corner agreement remains. This is a reconstruction layer, not
the physical implicit-Euler objective. CPU and CUDA tensors retain their
device and precision; a cached CPU sparse factorization supplies first-order
forward and adjoint solves.
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

# Material-axis pairs (b, c) of the bilinear warping modes xi^b xi^c, in the
# shared seven-vector order w_12, w_13, w_23 (zero-based axes).
_WARPING_PAIRS = ((0, 1), (0, 2), (1, 2))


def _mode_gradient_directions(points: np.ndarray, target_modes: int) -> np.ndarray:
    """Return the dimensionless mode gradient directions ``g_m(xi_q)``.

    Mode ``m`` contributes ``v_m g_m(xi_q)^T`` to the deformation gradient at
    the reference point ``xi_q``: ``g_a = e_a`` for the three affine modes and,
    for seven modes, ``(xi^c e_b + xi^b e_c)`` for the bilinear pairs
    ``(1, 2), (1, 3), (2, 3)`` followed by ``(xi^2 xi^3, xi^1 xi^3, xi^1 xi^2)``
    for the trilinear mode.

    Args:
        points: Reference Gauss points in [-1, 1]^3, shape [Q, 3].
        target_modes: 3 or 7.

    Returns:
        Directions with shape [Q, target_modes, 3], indexed [point, mode, axis].
    """
    directions = np.zeros((len(points), target_modes, 3), dtype=points.dtype)
    directions[:, :3] = np.eye(3, dtype=points.dtype)
    if target_modes == 7:
        for column, (first, second) in enumerate(_WARPING_PAIRS, start=3):
            directions[:, column, first] = points[:, second]
            directions[:, column, second] = points[:, first]
        directions[:, 6, 0] = points[:, 1] * points[:, 2]
        directions[:, 6, 1] = points[:, 0] * points[:, 2]
        directions[:, 6, 2] = points[:, 0] * points[:, 1]
    return directions


def _columns(values: np.ndarray) -> np.ndarray:
    """Pack batch and world coordinates as sparse-solve right-hand sides."""
    return np.asfortranarray(values.transpose(1, 0, 2).reshape(values.shape[1], values.shape[0] * values.shape[2]))


def _batch(columns: np.ndarray, batch_count: int) -> np.ndarray:
    """Restore batch-first world vectors after a sparse multiplication."""
    return columns.reshape(columns.shape[0], batch_count, 3).transpose(1, 0, 2)


class _FusionSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fusion, base_positions, world_targets, fixed_positions):
        device = base_positions.device
        fixed, free = fusion._indices(device)
        targets = world_targets.detach().cpu().contiguous().numpy()
        batch_count = base_positions.shape[0]
        # Rows run over (cell, mode) with world components as right-hand sides.
        target_rows = targets.swapaxes(-1, -2).reshape(batch_count, fusion.target_modes * fusion.cell_count, 3)
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
        adjoint, gradient_targets = fusion._adjoint_targets(gradient_output[:, free])
        boundary_transfer = _batch(fusion._fixed_coupling.T @ adjoint, batch_count)
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
    ``project_gradient`` exposes the same transpose solve as a detached
    operator that maps free-corner position gradients to target gradients.

    The minimized objective is the cell-weighted average over all eight Gauss
    points of ``||grad(delta_position)(xi_q) - Delta F_{c,q}||^2``, where the
    Gauss-point target ``Delta F_{c,q}`` is assembled from the cell's target
    vectors. Fixed displacements are prescribed positions minus base positions.
    The result adds the solved increment to the base, so a zero target exactly
    preserves a warped base when its fixed positions are already satisfied.
    With three modes a unique fit does not imply that one 3x3 target per cell
    can express every possible corner update; with seven modes every
    single-cell corner update is expressible and only shared-corner agreement
    remains. Non-inversion and physical descent are not enforced.

    Target modes. Cell corners carry reference coordinates ``xi_k`` in
    ``{-1, +1}^3`` in the z-fast corner order of ``rest.cell_corner_indices``
    (local corner ``k = 4x + 2y + z`` has ``xi_k = (2x - 1, 2y - 1, 2z - 1)``).
    The eight scalar corner modes ``1, xi^1, xi^2, xi^3, xi^1 xi^2, xi^1 xi^3,
    xi^2 xi^3, xi^1 xi^2 xi^3`` span the trilinear space exactly, and with
    mode coefficients ``c_m = (1/8) sum_k e_m(xi_k) x_k`` the target vectors
    are ``v_m = (2 / h) c_m`` for ``m = 1..7``: the three columns of the centre
    deformation gradient ``a_1, a_2, a_3`` followed by the warping vectors
    ``w_12, w_13, w_23, w_123``. Targets are packed as ``[..., 3, target_modes]``
    with ``v_m`` in columns, so ``[..., :, :3]`` is the centre 3x3 increment
    and ``target_modes=3`` is exactly the affine-only representation. The
    deformation gradient implied at a reference point is
    ``Delta F(xi) = sum_m v_m g_m(xi)^T`` with the dimensionless directions
    ``g_a = e_a``, ``g_bc = xi^c e_b + xi^b e_c`` and
    ``g_123 = (xi^2 xi^3, xi^1 xi^3, xi^1 xi^2)``, independent of ``h``.

    Operator construction. ``G`` is the sparse Gauss-point gradient operator
    whose row ``(c, q, i)`` holds ``dN_k/dX_i(xi_q)`` for the eight corners
    ``k`` of cell ``c``, so ``G d`` stacks material-axis column ``i`` of the
    trilinear displacement gradient at every Gauss point; the world component
    is the right-hand-side column. With the row weights ``w_c wq_q`` (cell
    weight times normalised quadrature weight) the stiffness is
    ``K = G^T W G``; unknowns are corners and ``K`` does not depend on the
    mode count. Targets enter through the sparse mode operator ``M`` with one
    column per ``(c, m)`` and entries ``M[(c, q, i), (c, m)] = w_c wq_q
    g_m,i(xi_q)``; for three modes ``M`` is the weighted copy of the single
    3x3 increment to all eight Gauss points. The free-corner normal equations
    read ``K_ff d_f = B v - K_fp d_p`` with the target operator
    ``B = (G^T M)_f``, and ``project_gradient`` applies the adjoint
    ``B^T K_ff^{-T}``. Only ``B`` changes with ``target_modes``; the system
    size, sparsity pattern and cached factor are identical.

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
        target_modes: Number of target vectors per cell, 3 (affine only,
            the default) or 7 (affine plus bilinear and trilinear warping).

    Raises:
        ValueError: If topology, weights, constraints, precision, or the mode
            count are invalid.
    """

    def __init__(
        self,
        rest: VoxelGridData,
        fixed_indices,
        *,
        cell_weights=None,
        dtype: torch.dtype = torch.float32,
        target_modes: int = 3,
    ):
        from scipy import sparse

        if dtype not in (torch.float32, torch.float64):
            raise ValueError("HexFusion supports only torch.float32 and torch.float64")
        if isinstance(target_modes, bool) or target_modes not in (3, 7):
            raise ValueError("target_modes must be 3 (affine) or 7 (affine plus warping)")
        self.target_modes = int(target_modes)
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
        # Mode operator M: row (c, q, i) couples to column (c, m) with weight
        # w_c wq_q g_m,i(xi_q). Structural zeros of the directions are dropped so
        # the three-mode operator is the weighted repeat of one 3x3 per cell.
        modes = self.target_modes
        directions = _mode_gradient_directions(np.asarray(quadrature.points, dtype=self._numpy_dtype), modes)
        entry_shape = (self.cell_count, 8, 3, modes)
        mode_rows = np.repeat(np.arange(row_count), modes)
        mode_columns = np.broadcast_to(
            (np.arange(self.cell_count) * modes)[:, None, None, None] + np.arange(modes)[None, None, None, :],
            entry_shape,
        ).reshape(-1)
        mode_values = (
            weights[:, None, None, None] * quadrature_weights[None, :, None, None] * directions.transpose(0, 2, 1)[None]
        ).reshape(-1)
        mode_mask = np.broadcast_to(directions.transpose(0, 2, 1)[None] != 0, entry_shape).reshape(-1)
        mode_operator = sparse.coo_matrix(
            (mode_values[mode_mask], (mode_rows[mode_mask], mode_columns[mode_mask])),
            shape=(row_count, self.cell_count * modes),
            dtype=self._numpy_dtype,
        ).tocsr()
        target_operator = (gradient_operator.T @ mode_operator).tocsr()
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

    def _adjoint_targets(self, free_gradient: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Pull free-corner cotangents back to target vectors through the transpose solve.

        Args:
            free_gradient: Free-corner position cotangents, shape [B, F, 3], in
                ``free_indices`` order, on any device. Autograd history is
                discarded.

        Returns:
            The packed adjoint columns ``K_ff^{-T} g_free`` with shape [F, 3B]
            and the target gradients ``unpack(B^T K_ff^{-T} g_free)`` with
            shape [B, C, 3, target_modes] (mode vectors in columns), both as
            CPU NumPy arrays in the working precision.
        """
        batch_count = free_gradient.shape[0]
        columns = _columns(free_gradient.detach().cpu().contiguous().numpy())
        adjoint = self._solve(columns, transpose=True)
        target_rows = _batch(self._target_operator.T @ adjoint, batch_count)
        return adjoint, target_rows.reshape(batch_count, self.cell_count, self.target_modes, 3).swapaxes(-1, -2)

    def fuse(
        self,
        base_positions: torch.Tensor,
        world_targets: torch.Tensor,
        fixed_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return a differentiable compatible update of shared corner positions.

        Args:
            base_positions: Current world corner positions [m], shape [B, P, 3].
            world_targets: Dimensionless world target vectors, shape
                [B, C, 3, target_modes], with the mode vectors in columns: the
                three deformation-gradient axis increments followed, for seven
                modes, by the warping increments ``w_12, w_13, w_23, w_123``.
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
            ("world_targets", world_targets),
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
        if world_targets.shape != (batch_count, self.cell_count, 3, self.target_modes):
            raise ValueError(f"world_targets must have shape [B, C, 3, {self.target_modes}]")
        if fixed_positions.shape != (batch_count, len(self._fixed), 3):
            raise ValueError("fixed_positions must have shape [B, K, 3]")
        return _FusionSolve.apply(self, base_positions, world_targets, fixed_positions)

    def project_gradient(self, position_gradient: torch.Tensor) -> torch.Tensor:
        """Return target gradients ``unpack(B^T K_ff^{-T} g_free)``.

        This is the adjoint of the target-to-position map behind ``fuse`` for
        a frozen base and frozen prescribed positions: for every target ``D``,
        ``<project_gradient(g), D>`` equals
        ``<g_free, fuse(base, D, fixed) - fuse(base, 0, fixed)>``. It runs the
        cached factor's transpose solve through the same code path as the
        autograd backward of ``fuse``, so it matches
        ``autograd.grad(<g, fuse(base, D, fixed)>, D)`` for any ``D``. With
        seven modes the transpose solve is followed by the mode adjoint, so
        the warping columns hold the gradient projected onto the warping modes.
        No extra stiffness or volume factor is applied.

        Args:
            position_gradient: World position gradient of the physical
                objective [N], shape [B, P, 3], on CPU or CUDA. Only free-corner
                rows are read; fixed rows are ignored. Autograd history is
                discarded.

        Returns:
            Detached target gradients [J], shape [B, C, 3, target_modes], on
            the input device in the working precision, in the same world
            layout as ``world_targets`` in ``fuse`` (mode vectors in columns).

        Raises:
            TypeError: If the input is not a tensor in the chosen precision.
            ValueError: If the shape is not [B, P, 3] with B > 0, the device is
                neither CPU nor CUDA, or the free rows are not finite.
        """
        if not isinstance(position_gradient, torch.Tensor) or position_gradient.dtype != self.dtype:
            raise TypeError(f"position_gradient must be a tensor with dtype {self.dtype}")
        if position_gradient.device.type not in ("cpu", "cuda"):
            raise ValueError("position_gradient must use a CPU or CUDA device")
        if (
            position_gradient.ndim != 3
            or position_gradient.shape[1:] != (self.corner_count, 3)
            or position_gradient.shape[0] == 0
        ):
            raise ValueError("position_gradient must have shape [B, P, 3] with B > 0")
        _, free = self._indices(position_gradient.device)
        _, gradient_targets = self._adjoint_targets(position_gradient[:, free])
        return torch.from_numpy(np.ascontiguousarray(gradient_targets)).to(device=position_gradient.device)
