# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Revised nine-value input schema for the learned hexahedral optimizer.

This module is the single source of truth for the schema-3 node input layout
and holds the pure feature functions that the mixed-material step composes.
Every spatial matrix block is a 3x3 matrix expressed in the receiving cell's
current proper frame ``R`` (world columns) and flattened row-major; see
:func:`pack_state_features` for the exact packing order. Current axes
``A = R^T F`` are not part of the state vector: the network prepends them
separately, so the complete node input has ``9 + STATE_FEATURE_DIM`` values.

Normalization follows the LeCO convention (plan: "Gradient inputs and
normalization"): the current and previous axis gradients share the current
object's gradient RMS, the previous achieved axis update uses its own RMS,
RMS values are floored at :data:`RMS_FLOOR`, normalized components are clipped
to ``+/-CLIP`` in the receiving local frame, and the natural log of the current
RMS is a separate scalar input. ``history_valid`` distinguishes missing
optimizer history from a measured zero.

The functions hold no module state and never detach their inputs; callers
decide what enters autograd.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "BOUNDARY_DIM",
    "CLIP",
    "CONDITIONING_CHANNELS",
    "CONDITIONING_DIM",
    "EDGE_FEATURE_DIM",
    "FEATURE_SCHEMA_VERSION",
    "MATRIX_BLOCKS",
    "MATRIX_FEATURE_DIM",
    "RMS_FLOOR",
    "SCALAR_FEATURES",
    "STATE_FEATURE_DIM",
    "center_deformation",
    "conditioning_channels",
    "pack_state_features",
    "rms_normalize",
    "to_local",
]

MATRIX_BLOCKS = (
    "inertial_axis_offset",
    "physical_axis_change",
    "current_axis_gradient",
    "previous_axis_gradient",
    "previous_axis_update",
)
"""Nine-value matrix blocks in packing order; each is ``R^T M`` flattened row-major."""

MATRIX_FEATURE_DIM = 9 * len(MATRIX_BLOCKS)
"""Width of the five matrix blocks (45)."""

BOUNDARY_DIM = 14
"""Six exposed-face flags followed by eight fixed-corner flags in z-fast corner order."""

SCALAR_FEATURES = ("log_gradient_rms", "history_valid")
"""Trailing per-cell scalars: natural log of the object gradient RMS and the history flag."""

STATE_FEATURE_DIM = MATRIX_FEATURE_DIM + BOUNDARY_DIM + len(SCALAR_FEATURES)
"""Total state width (61): 45 matrix components, 14 boundary flags, 2 scalars."""

CONDITIONING_CHANNELS = (
    "log1p_lame_lambda",
    "log1p_lame_mu",
    "log_density",
    "log_cell_size",
    "log_time_step",
    "log1p_damping",
)
"""Per-object conditioning channels in order; see :func:`conditioning_channels`."""

CONDITIONING_DIM = len(CONDITIONING_CHANNELS)
"""Number of conditioning channels (6)."""

EDGE_FEATURE_DIM = 24
"""Directed-edge descriptor width from :func:`network_geometry.build_edge_features`."""

FEATURE_SCHEMA_VERSION = 3
"""Schema version stored in checkpoints; legacy 38/86 schemas are incompatible."""

RMS_FLOOR = 1e-12
"""Lower bound on every input RMS before division."""

CLIP = 10.0
"""Symmetric clip applied to normalized local-frame components."""

_REFERENCE_LAME = 1e5
_REFERENCE_DENSITY = 1000.0
_REFERENCE_CELL_SIZE = 0.025
_REFERENCE_TIME_STEP = 1.0 / 60.0
_HISTORY_BLOCKS = ("previous_axis_gradient", "previous_axis_update")


def _require_tensor(name: str, value) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_cell_matrices(name: str, value, *, like: torch.Tensor | None = None) -> None:
    """Check a floating [B, C, 3, 3] tensor, optionally matching a reference tensor."""
    _require_tensor(name, value)
    if value.ndim != 4 or value.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have shape [B, C, 3, 3]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if like is not None:
        if value.shape != like.shape:
            raise ValueError(f"{name} must share the [B, C, 3, 3] shape of the other matrix blocks")
        if value.dtype != like.dtype:
            raise TypeError(f"{name} must share the dtype of the other matrix blocks")
        if value.device != like.device:
            raise ValueError(f"{name} must be on the same device as the other matrix blocks")


def _per_object(name: str, value, batch_count: int) -> torch.Tensor:
    """Return a tensor with one entry per object reshaped to [B]."""
    _require_tensor(name, value)
    if value.numel() != batch_count:
        raise ValueError(f"{name} must hold exactly one value per object (B = {batch_count})")
    return value.reshape(batch_count)


def _require_positive_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def center_deformation(
    positions: torch.Tensor, cell_corner_indices: torch.Tensor, center_gradients: torch.Tensor
) -> torch.Tensor:
    """Compute the cell-center deformation gradient F from shared corner positions.

    Uses the convention of the mixed-material step: ``F_ij = sum_k (x_k - x_0)_i
    G_kj`` over the eight z-fast corners with the rest-cube center shape
    gradients ``G = signs / (4 h)`` [1/m]. Columns of F are the transformed
    material unit directions; they are not normalized, and inverted or collapsed
    cells yield a finite F with nonpositive determinant. Subtracting corner zero
    is exact because the gradients sum to zero and keeps the float32 sum well
    conditioned. Differentiable in ``positions``.

    Args:
        positions: World corner positions [m], shape [B, P, 3].
        cell_corner_indices: Long corner IDs per cell in z-fast order, shape [C, 8].
        center_gradients: Center shape gradients [1/m], shape [8, 3], sharing the
            dtype and device of positions.

    Returns:
        Deformation gradients, shape [B, C, 3, 3].

    Raises:
        TypeError: If an input is not a tensor or has an incompatible dtype.
        ValueError: If shapes or devices are incompatible.
    """
    import torch

    _require_tensor("positions", positions)
    _require_tensor("cell_corner_indices", cell_corner_indices)
    _require_tensor("center_gradients", center_gradients)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B, P, 3]")
    if not positions.is_floating_point():
        raise TypeError("positions must have a floating dtype")
    if cell_corner_indices.dtype != torch.long:
        raise TypeError("cell_corner_indices must have dtype torch.long")
    if cell_corner_indices.ndim != 2 or cell_corner_indices.shape[-1] != 8:
        raise ValueError("cell_corner_indices must have shape [C, 8]")
    if center_gradients.shape != (8, 3):
        raise ValueError("center_gradients must have shape [8, 3]")
    if center_gradients.dtype != positions.dtype:
        raise TypeError("center_gradients must share the dtype of positions")
    if cell_corner_indices.device != positions.device or center_gradients.device != positions.device:
        raise ValueError("cell_corner_indices and center_gradients must be on the device of positions")

    corners = positions[:, cell_corner_indices]
    return torch.einsum("bcki,kj->bcij", corners - corners[:, :, :1], center_gradients)


def to_local(frames: torch.Tensor, world_matrices: torch.Tensor) -> torch.Tensor:
    """Express world 3x3 matrices in each cell's frame as ``R^T @ M``.

    Frames store proper rotations with world columns. Axis differences, axis
    gradients, and achieved axis updates all transform as ``R^T M`` under a
    frame change, so one function serves every matrix block. Nothing is
    detached here; callers freeze frames before use.

    Args:
        frames: Local-to-world rotations, shape [..., 3, 3].
        world_matrices: World matrices with axes in columns, same shape as frames.

    Returns:
        Local matrices ``R^T M`` with the shape of the inputs.

    Raises:
        TypeError: If an input is not a floating tensor or the dtypes differ.
        ValueError: If the shapes are not equal trailing-(3, 3) shapes.
    """
    _require_tensor("frames", frames)
    _require_tensor("world_matrices", world_matrices)
    if frames.ndim < 2 or frames.shape[-2:] != (3, 3):
        raise ValueError("frames must have shape [..., 3, 3]")
    if world_matrices.shape != frames.shape:
        raise ValueError("world_matrices must share the shape of frames")
    if not frames.is_floating_point() or world_matrices.dtype != frames.dtype:
        raise TypeError("frames and world_matrices must share one floating dtype")
    if world_matrices.device != frames.device:
        raise ValueError("world_matrices must be on the same device as frames")
    return frames.transpose(-1, -2) @ world_matrices


def rms_normalize(values: torch.Tensor, *, rms: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a per-object matrix field by a floored RMS and clip the result.

    The RMS is the root mean square over every cell matrix component of one
    object, ``sqrt(mean_{c,i,j} values^2)``, floored at :data:`RMS_FLOOR` so a
    measured zero stays finite: zero input yields zero output and an RMS equal
    to the floor. Pass ``rms`` to reuse another field's statistics, as the
    previous gradient does with the current gradient RMS; the supplied RMS is
    floored again and returned as the second element. Clipping to
    ``+/-CLIP`` acts on the components as given, so pass local-frame values.

    Args:
        values: Local-frame matrix field, shape [B, C, 3, 3].
        rms: Optional per-object RMS with one value per object (any shape with B
            elements, e.g. [B] or [B, 1, 1, 1]); computed from ``values`` when None.

    Returns:
        ``(normalized, rms)`` with normalized shaped like ``values`` and clipped to
        ``[-CLIP, CLIP]``, and rms shaped [B, 1, 1, 1] in the dtype of ``values``.

    Raises:
        TypeError: If values is not a floating tensor.
        ValueError: If values is not [B, C, 3, 3] or rms does not hold B values.
    """
    _require_cell_matrices("values", values)
    batch_count = values.shape[0]
    if rms is None:
        scale = values.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    else:
        scale = _per_object("rms", rms, batch_count).to(dtype=values.dtype, device=values.device)
        scale = scale.reshape(batch_count, 1, 1, 1)
    scale = scale.clamp_min(RMS_FLOOR)
    return (values / scale).clamp(-CLIP, CLIP), scale


def pack_state_features(
    *,
    inertial_axis_offset: torch.Tensor,
    physical_axis_change: torch.Tensor,
    current_axis_gradient: torch.Tensor,
    previous_axis_gradient: torch.Tensor,
    previous_axis_update: torch.Tensor,
    boundary_features: torch.Tensor,
    log_gradient_rms: torch.Tensor,
    history_valid: torch.Tensor,
) -> torch.Tensor:
    """Pack the schema-3 per-cell state vector of width :data:`STATE_FEATURE_DIM`.

    Packing order along the last axis: ``inertial_axis_offset`` (9, row-major
    3x3), ``physical_axis_change`` (9), ``current_axis_gradient`` (9),
    ``previous_axis_gradient`` (9), ``previous_axis_update`` (9), exposed-face
    flags (6), fixed-corner flags (8), ``log_gradient_rms`` (1, broadcast per
    cell), ``history_valid`` (1, exactly 1.0 or 0.0). All matrix blocks must
    already be expressed in the receiving local frame and normalized by the
    caller. Objects whose ``history_valid`` is false have both history blocks
    written as zeros regardless of the tensors supplied, so a missing history is
    always distinguishable from a measured zero.

    Args:
        inertial_axis_offset: ``R^T (F_target - F_current)``, shape [B, C, 3, 3].
        physical_axis_change: ``R^T (F_current - F_previous)``, shape [B, C, 3, 3].
        current_axis_gradient: Normalized current local axis gradient, shape [B, C, 3, 3].
        previous_axis_gradient: Normalized previous gradient in the current frame, shape [B, C, 3, 3].
        previous_axis_update: Normalized previous achieved update in the current frame, shape [B, C, 3, 3].
        boundary_features: Boundary flags, shape [C, 14] shared across objects or [B, C, 14].
        log_gradient_rms: Natural log of the current gradient RMS, one value per object.
        history_valid: One flag per object; boolean, or numeric where nonzero means valid.

    Returns:
        State features, shape [B, C, STATE_FEATURE_DIM], in the dtype of the matrix blocks.

    Raises:
        TypeError: If an input is not a tensor or dtypes are incompatible.
        ValueError: If shapes or devices are incompatible.
    """
    import torch

    blocks = {
        "inertial_axis_offset": inertial_axis_offset,
        "physical_axis_change": physical_axis_change,
        "current_axis_gradient": current_axis_gradient,
        "previous_axis_gradient": previous_axis_gradient,
        "previous_axis_update": previous_axis_update,
    }
    reference = inertial_axis_offset
    _require_cell_matrices("inertial_axis_offset", reference)
    for name in MATRIX_BLOCKS[1:]:
        _require_cell_matrices(name, blocks[name], like=reference)
    batch_count, cell_count = reference.shape[:2]

    _require_tensor("boundary_features", boundary_features)
    if boundary_features.shape == (cell_count, BOUNDARY_DIM):
        boundary = boundary_features[None].expand(batch_count, -1, -1)
    elif boundary_features.shape == (batch_count, cell_count, BOUNDARY_DIM):
        boundary = boundary_features
    else:
        raise ValueError(f"boundary_features must have shape [C, {BOUNDARY_DIM}] or [B, C, {BOUNDARY_DIM}]")
    if boundary.dtype != reference.dtype:
        raise TypeError("boundary_features must share the dtype of the matrix blocks")
    if boundary.device != reference.device:
        raise ValueError("boundary_features must be on the same device as the matrix blocks")

    log_rms = _per_object("log_gradient_rms", log_gradient_rms, batch_count)
    if not log_rms.is_floating_point():
        raise TypeError("log_gradient_rms must have a floating dtype")
    log_rms = log_rms.to(dtype=reference.dtype, device=reference.device)
    valid = _per_object("history_valid", history_valid, batch_count).to(reference.device)
    valid = valid if valid.dtype == torch.bool else valid != 0

    mask = valid[:, None, None, None]
    columns = []
    for name in MATRIX_BLOCKS:
        block = blocks[name]
        if name in _HISTORY_BLOCKS:
            block = torch.where(mask, block, torch.zeros_like(block))
        columns.append(block.flatten(-2))
    columns.append(boundary)
    columns.append(log_rms[:, None, None].expand(-1, cell_count, 1))
    columns.append(valid.to(reference.dtype)[:, None, None].expand(-1, cell_count, 1))
    return torch.cat(columns, dim=-1)


def conditioning_channels(
    lame_lambda: torch.Tensor,
    lame_mu: torch.Tensor,
    density: torch.Tensor,
    damping: torch.Tensor,
    cell_size: float,
    time_step: float,
) -> torch.Tensor:
    """Build the six per-object conditioning channels, shape [B, 6].

    Channel order follows :data:`CONDITIONING_CHANNELS`::

        log1p(lambda / 1e5), log1p(mu / 1e5), log(rho / 1000),
        log(h / 0.025), log(dt * 60), log1p(eta / (mu * dt))

    The viscosity channel is dimensionless and does not rescale the physical
    eta. Material tensors are not value-checked here (contexts validate them);
    nonpositive entries would produce nonfinite channels visibly. Callers expand
    the result to cells as needed.

    Args:
        lame_lambda: First Lamé parameter [Pa], shape [B].
        lame_mu: Shear modulus [Pa], shape [B].
        density: Mass density [kg/m^3], shape [B].
        damping: Viscosity eta [Pa s], shape [B].
        cell_size: Positive finite rest voxel edge length [m].
        time_step: Positive finite physical time step [s].

    Returns:
        Conditioning channels, shape [B, 6], in the dtype and device of lame_mu.

    Raises:
        TypeError: If a material input is not a floating tensor or dtypes differ.
        ValueError: If shapes, devices, or the scalar arguments are invalid.
    """
    import torch

    materials = (("lame_lambda", lame_lambda), ("lame_mu", lame_mu), ("density", density), ("damping", damping))
    for name, tensor in materials:
        _require_tensor(name, tensor)
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape [B]")
        if not tensor.is_floating_point() or tensor.dtype != lame_mu.dtype:
            raise TypeError(f"{name} must share the floating dtype of lame_mu")
        if tensor.shape != lame_mu.shape:
            raise ValueError(f"{name} must share the [B] shape of lame_mu")
        if tensor.device != lame_mu.device:
            raise ValueError(f"{name} must be on the same device as lame_mu")
    size = _require_positive_scalar("cell_size", cell_size)
    step = _require_positive_scalar("time_step", time_step)

    channels = (
        (lame_lambda / _REFERENCE_LAME).log1p(),
        (lame_mu / _REFERENCE_LAME).log1p(),
        (density / _REFERENCE_DENSITY).log(),
        torch.full_like(lame_mu, math.log(size / _REFERENCE_CELL_SIZE)),
        torch.full_like(lame_mu, math.log(step / _REFERENCE_TIME_STEP)),
        (damping / (lame_mu * step)).log1p(),
    )
    return torch.stack(channels, dim=-1)
