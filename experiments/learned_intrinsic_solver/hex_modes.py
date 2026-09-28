# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Seven-mode trilinear basis for per-cell deformation targets.

Experimental. This module is the single source of truth for the mode basis
shared by the learned optimizer's 21-value per-cell targets, the fusion
assembly and the state blocks. It holds pure, differentiable Torch functions
with no module state; callers decide what enters autograd.

Basis. Corner local coordinates ``xi_k`` lie in ``{-1, +1}^3`` in the z-fast
corner order of :attr:`~experiments.learned_intrinsic_solver.data.VoxelGridData.cell_corner_indices`
(local corner ``k = 4 x + 2 y + z`` with bits ``x, y, z`` has
``xi_k = (2 x - 1, 2 y - 1, 2 z - 1)``). The eight scalar modes are

``e_0 = 1``, ``e_1..3 = xi^a`` (affine), ``e_4 = xi^1 xi^2``,
``e_5 = xi^1 xi^3``, ``e_6 = xi^2 xi^3`` (bilinear warping) and
``e_7 = xi^1 xi^2 xi^3`` (trilinear warping).

The matrix ``E[k, m] = e_m(xi_k)`` (:func:`mode_matrix`) has orthogonal
``+-1`` columns with ``|column|^2 = 8``, and the trilinear shape functions are
``N_k(xi) = (1/8) sum_m E[k, m] e_m(xi)``. For world corner positions ``x_k``
the mode coefficients are ``c_m = (1/8) sum_k E[k, m] x_k``, so the
interpolated position is ``x(xi) = sum_m c_m e_m(xi)`` and the deformation
gradient at any local point is ``F(xi) = sum_m c_m (grad_X e_m)(xi)^T`` with
``grad_X = (2 / h) grad_xi`` for rest cell size ``h``.

Targets. The seven target vectors are ``v_m = (2 / h) c_m`` for ``m = 1..7``
(dimensionless like ``F``): ``v_1..3`` are the columns ``a_1, a_2, a_3`` of the
centre deformation gradient and ``v_4..7`` the warping vectors ``w_12, w_13,
w_23, w_123``. With the dimensionless directions ``d_m(xi) = grad_xi e_m(xi)``,

``F(xi) = sum_m v_m d_m(xi)^T = sum_a a_a e_a^T + sum_{b<c} w_bc (xi^c e_b +
xi^b e_c)^T + w_123 (xi^2 xi^3, xi^1 xi^3, xi^1 xi^2)^T``,

independent of ``h``. Packing convention everywhere: a ``[..., 3, 7]`` tensor
whose columns are ``v_1..v_7`` in that order, so ``[..., :, :3]`` is exactly the
centre ``F`` of :func:`~experiments.learned_intrinsic_solver.features.center_deformation`.
The local representation in a cell frame ``R`` is ``R^T V``.
"""

from __future__ import annotations

import math
from numbers import Real

import torch  # noqa: TID253 -- This opt-in experimental module explicitly requires Torch.

__all__ = [
    "MODE_COUNT",
    "TARGET_DIM",
    "corner_local_coordinates",
    "gauss_point_deformation",
    "mode_gradient_directions",
    "mode_matrix",
    "mode_vectors",
    "project_gauss_gradients",
]

MODE_COUNT = 7
"""Number of target vectors per cell: three axes plus four warping vectors."""

TARGET_DIM = 3 * MODE_COUNT
"""Number of scalar per-cell target values, ``3 * MODE_COUNT``."""


def _require_tensor(name: str, value) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_positive_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _require_local_points(xi, *, like: torch.Tensor) -> None:
    """Check a floating [Q, 3] point tensor sharing the dtype and device of ``like``."""
    _require_tensor("xi", xi)
    if xi.ndim != 2 or xi.shape[-1] != 3:
        raise ValueError("xi must have shape [Q, 3]")
    if xi.dtype != like.dtype:
        raise TypeError("xi must share the dtype of the mode tensors")
    if xi.device != like.device:
        raise ValueError("xi must be on the device of the mode tensors")


def _mode_values(xi: torch.Tensor) -> torch.Tensor:
    """Evaluate the eight scalar modes ``e_0..e_7`` at local points, shape [..., 8]."""
    x, y, z = xi.unbind(-1)
    return torch.stack((torch.ones_like(x), x, y, z, x * y, x * z, y * z, x * y * z), dim=-1)


def _reference_gradient_directions(xi: torch.Tensor) -> torch.Tensor:
    """Return the dimensionless directions ``d_m(xi) = grad_xi e_m(xi)`` for ``m = 1..7``, shape [Q, 7, 3]."""
    x, y, z = xi.unbind(-1)
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    rows = (
        (one, zero, zero),
        (zero, one, zero),
        (zero, zero, one),
        (y, x, zero),
        (z, zero, x),
        (zero, z, y),
        (y * z, x * z, x * y),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def corner_local_coordinates(
    *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return the corner local coordinates ``xi_k`` in z-fast order.

    Local corner ``k = 4 x + 2 y + z`` (bits ``x, y, z``) maps to
    ``xi_k = (2 x - 1, 2 y - 1, 2 z - 1)``, the order of
    :attr:`~experiments.learned_intrinsic_solver.data.VoxelGridData.cell_corner_indices`
    and of :func:`~experiments.learned_intrinsic_solver.hex_energy.hex_gauss_quadrature`.

    Args:
        dtype: Floating dtype of the result; the Torch default dtype when None.
        device: Device of the result; the CPU when None.

    Returns:
        Corner coordinates in ``{-1, +1}^3``, shape [8, 3].
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    signs = [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    return torch.tensor(signs, dtype=dtype, device=device)


def mode_matrix(*, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Return the mode matrix ``E[k, m] = e_m(xi_k)``.

    Rows follow the z-fast corner order of :func:`corner_local_coordinates`;
    columns are the modes ``e_0..e_7`` in the module order (constant, three
    affine, three bilinear ``xi^1 xi^2, xi^1 xi^3, xi^2 xi^3``, one trilinear).
    The columns are orthogonal with ``E^T E = 8 I``.

    Args:
        dtype: Floating dtype of the result; the Torch default dtype when None.
        device: Device of the result; the CPU when None.

    Returns:
        Mode matrix with ``+-1`` entries, shape [8, 8].
    """
    return _mode_values(corner_local_coordinates(dtype=dtype, device=device))


def mode_gradient_directions(xi: torch.Tensor, cell_size) -> torch.Tensor:
    """Return the material gradients ``grad_X e_m(xi)`` of the seven non-constant modes.

    With ``grad_X = (2 / h) grad_xi`` for rest cell size ``h``, index ``m - 1``
    of the middle axis holds mode ``m = 1..7`` in the target order (``xi^1,
    xi^2, xi^3, xi^1 xi^2, xi^1 xi^3, xi^2 xi^3, xi^1 xi^2 xi^3``):

    ``grad_X xi^a = (2 / h) e_a``,
    ``grad_X (xi^b xi^c) = (2 / h) (xi^c e_b + xi^b e_c)``,
    ``grad_X (xi^1 xi^2 xi^3) = (2 / h) (xi^2 xi^3, xi^1 xi^3, xi^1 xi^2)``.

    The deformation gradient follows from the mode coefficients ``c_m`` as
    ``F(xi_q) = sum_m c_m D[q, m - 1]^T = (h / 2) sum_m v_m D[q, m - 1]^T``
    where ``v_m = (2 / h) c_m`` are the target vectors of :func:`mode_vectors`;
    :func:`gauss_point_deformation` evaluates this ``h``-free form directly.
    The trilinear shape gradients of
    :func:`~experiments.learned_intrinsic_solver.hex_energy.hex_gauss_quadrature`
    are ``G[q, k, :] = (1/8) sum_{m>=1} E[k, m] D[q, m - 1, :]``.

    Args:
        xi: Local points in the reference cube ``[-1, 1]^3``, floating, shape [Q, 3].
        cell_size: Positive rest cell edge length ``h`` [m].

    Returns:
        Material gradients [1/m], shape [Q, 7, 3], in the dtype and device of ``xi``.

    Raises:
        TypeError: If ``xi`` is not a floating tensor.
        ValueError: If ``xi`` is not [Q, 3] or ``cell_size`` is not positive and finite.
    """
    _require_tensor("xi", xi)
    if not xi.is_floating_point():
        raise TypeError("xi must have a floating dtype")
    if xi.ndim != 2 or xi.shape[-1] != 3:
        raise ValueError("xi must have shape [Q, 3]")
    scale = 2.0 / _require_positive_scalar("cell_size", cell_size)
    return scale * _reference_gradient_directions(xi)


def mode_vectors(positions: torch.Tensor, cell_corner_indices: torch.Tensor, cell_size) -> torch.Tensor:
    """Compute the seven target vectors ``v_1..v_7`` of every cell from shared corners.

    ``v_m = (2 / h) c_m = (1 / (4 h)) sum_k E[k, m] (x_k - x_0)``; subtracting
    corner zero is exact because every non-constant mode column sums to zero
    and keeps the float32 sum well conditioned. The first three columns are the
    centre deformation gradient of
    :func:`~experiments.learned_intrinsic_solver.features.center_deformation`
    (same corners, same ``signs / (4 h)`` weights); the last four are the
    bilinear and trilinear warping vectors, zero for an affinely deformed cell.
    Rotating the positions rotates every column; translating them changes
    nothing. Differentiable in ``positions``.

    Args:
        positions: World corner positions [m], floating, shape [B, P, 3].
        cell_corner_indices: Long corner IDs per cell in z-fast order, shape [C, 8],
            on the device of ``positions``.
        cell_size: Positive rest cell edge length ``h`` [m].

    Returns:
        Target vectors, dimensionless, shape [B, C, 3, 7], columns ``a_1, a_2,
        a_3, w_12, w_13, w_23, w_123``, in the dtype and device of ``positions``.

    Raises:
        TypeError: If an input is not a tensor or has an incompatible dtype.
        ValueError: If shapes or devices are incompatible or ``cell_size`` is invalid.
    """
    _require_tensor("positions", positions)
    _require_tensor("cell_corner_indices", cell_corner_indices)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B, P, 3]")
    if not positions.is_floating_point():
        raise TypeError("positions must have a floating dtype")
    if cell_corner_indices.dtype != torch.long:
        raise TypeError("cell_corner_indices must have dtype torch.long")
    if cell_corner_indices.ndim != 2 or cell_corner_indices.shape[-1] != 8:
        raise ValueError("cell_corner_indices must have shape [C, 8]")
    if cell_corner_indices.device != positions.device:
        raise ValueError("cell_corner_indices must be on the device of positions")
    size = _require_positive_scalar("cell_size", cell_size)
    weights = mode_matrix(dtype=positions.dtype, device=positions.device)[:, 1:] / (4.0 * size)
    corners = positions[:, cell_corner_indices]
    return torch.einsum("bcki,km->bcim", corners - corners[:, :, :1], weights)


def gauss_point_deformation(vectors: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Evaluate the deformation gradient ``F(xi_q)`` from the seven target vectors.

    ``F(xi_q) = sum_m v_m d_m(xi_q)^T`` with the dimensionless directions
    ``d_m = grad_xi e_m``, independent of the cell size. With the eight points
    of :func:`~experiments.learned_intrinsic_solver.hex_energy.hex_gauss_quadrature`
    this equals the Gauss-point deformation that
    :class:`~experiments.learned_intrinsic_solver.hex_energy.HexImplicitEulerLoss`
    forms from the same corners. At ``xi = 0`` the result is the centre ``F``,
    the first three columns of ``vectors``. Linear in ``vectors`` and
    differentiable in both inputs.

    Args:
        vectors: Target vectors in the packing of :func:`mode_vectors`, shape [..., 3, 7];
            typically [B, C, 3, 7].
        xi: Local evaluation points, shape [Q, 3], sharing the dtype and device of ``vectors``.

    Returns:
        Deformation gradients, shape [..., Q, 3, 3]; [B, C, Q, 3, 3] for cell batches.

    Raises:
        TypeError: If an input is not a tensor or the dtypes differ.
        ValueError: If the shapes or devices are incompatible.
    """
    _require_tensor("vectors", vectors)
    if not vectors.is_floating_point():
        raise TypeError("vectors must have a floating dtype")
    if vectors.ndim < 2 or vectors.shape[-2:] != (3, MODE_COUNT):
        raise ValueError(f"vectors must have shape [..., 3, {MODE_COUNT}]")
    _require_local_points(xi, like=vectors)
    return torch.einsum("...im,qmj->...qij", vectors, _reference_gradient_directions(xi))


def project_gauss_gradients(gradients: torch.Tensor, xi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Project weighted Gauss-point gradients onto the seven modes.

    Returns the exact linear adjoint of :func:`gauss_point_deformation` under
    the weighted inner product: the derivative with respect to ``v_m`` of
    ``sum_q w_q <G_q, F(xi_q)>``, that is ``sum_q w_q G_q d_m(xi_q)`` per
    mode, so that ``sum_q w_q <G_q, gauss_point_deformation(V)_q> = <project(G), V>``
    for every ``V``. Pass the quadrature weights (including the rest Jacobian
    when the gradients are energy densities) to obtain the mode-space gradient
    of a quadrature sum. Linear in ``gradients`` and differentiable.

    Args:
        gradients: Per-point gradients with respect to ``F``, shape [..., Q, 3, 3];
            typically [B, C, Q, 3, 3].
        xi: Local points, shape [Q, 3], sharing the dtype and device of ``gradients``.
        weights: Per-point weights, shape [Q], sharing the dtype and device of ``gradients``.

    Returns:
        Mode-space gradients in the packing of :func:`mode_vectors`, shape [..., 3, 7].

    Raises:
        TypeError: If an input is not a tensor or the dtypes differ.
        ValueError: If the shapes or devices are incompatible.
    """
    _require_tensor("gradients", gradients)
    if not gradients.is_floating_point():
        raise TypeError("gradients must have a floating dtype")
    if gradients.ndim < 3 or gradients.shape[-2:] != (3, 3):
        raise ValueError("gradients must have shape [..., Q, 3, 3]")
    _require_local_points(xi, like=gradients)
    _require_tensor("weights", weights)
    if weights.shape != xi.shape[:1] or gradients.shape[-3] != xi.shape[0]:
        raise ValueError("weights must have shape [Q] matching xi and the gradients point axis")
    if weights.dtype != gradients.dtype:
        raise TypeError("weights must share the dtype of gradients")
    if weights.device != gradients.device:
        raise ValueError("weights must be on the device of gradients")
    return torch.einsum("q,...qij,qmj->...im", weights, gradients, _reference_gradient_directions(xi))
