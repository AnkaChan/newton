# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Node input schema for the learned hexahedral optimizer.

This module is the single source of truth for the node input layout and holds
the pure feature functions that the mixed-material step composes. Every
spatial block is a ``3 x m`` matrix ``V`` whose ``m >= 1`` columns are world
target vectors of one cell: for ``m = 3`` the columns of the centre
deformation gradient ``F`` (the axes ``a_1, a_2, a_3``); for ``m = 7``
(:data:`TARGET_MODES_DEFAULT`) those three axes followed by the four warping
vectors ``w_12, w_13, w_23, w_123`` of the trilinear cell map, so ``V[:, :3]``
is always the centre ``F``. Each block is expressed in the receiving cell's
current proper frame ``R`` (world columns) as ``R^T V`` and flattened
row-major; see :func:`pack_state_features` for the exact packing order. The
current local targets ``R^T V`` are not part of the state vector: the network
prepends them separately, so the node input has ``target_dim(m) +
state_feature_dim(m)`` values (``9 + STATE_FEATURE_DIM`` for the legacy
three-mode schema), plus
:data:`CONTACT_FEATURE_DIM` pooled contact channels that the network produces
itself from :data:`CONTACT_TOKEN_DIM`-wide contact tokens when contact tokens
are enabled (``contact_features.build_contact_tokens`` builds the tokens).
The per-object conditioning (:func:`conditioning_channels`) consists of
dimensionless groups only: the Lamé ratio, the inertia-to-stiffness ratio
``rho h^2 / (mu dt^2)``, the gravity drop per step in cell units ``|g| dt^2 /
h``, the viscosity ratio ``eta / (mu dt)`` and the three contact ratios of
:func:`contact_ratios`. Two scenes related by the cell-normalisation law of
``notes/ideas/idea-normalize-cells.md`` therefore receive identical
conditioning; contact-free objects carry zeros in the contact channels.

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
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "BOUNDARY_DIM",
    "CLIP",
    "CONDITIONING_CHANNELS",
    "CONDITIONING_DIM",
    "CONTACT_FEATURE_DIM",
    "CONTACT_TOKEN_DIM",
    "EDGE_FEATURE_DIM",
    "FEATURE_SCHEMA_VERSION",
    "MATRIX_BLOCKS",
    "MATRIX_FEATURE_DIM",
    "RMS_FLOOR",
    "SCALAR_FEATURES",
    "STATE_FEATURE_DIM",
    "TARGET_MODES_DEFAULT",
    "center_deformation",
    "conditioning_channels",
    "contact_ratios",
    "pack_state_features",
    "rms_normalize",
    "state_feature_dim",
    "target_dim",
    "to_local",
]

MATRIX_BLOCKS = (
    "inertial_axis_offset",
    "physical_axis_change",
    "current_axis_gradient",
    "previous_axis_gradient",
    "previous_axis_update",
)
"""Matrix blocks in packing order; each is a ``3 x m`` world block ``V`` stored as ``R^T V`` flattened row-major."""

BOUNDARY_DIM = 14
"""Six exposed-face flags followed by eight fixed-corner flags in z-fast corner order."""

SCALAR_FEATURES = ("log_gradient_rms", "history_valid")
"""Trailing per-cell scalars: natural log of the object gradient RMS and the history flag."""

TARGET_MODES_DEFAULT = 7
"""Target vectors per cell in the seven-mode schema: three axes plus the warping vectors ``w_12, w_13, w_23, w_123``."""


def _require_modes(modes) -> int:
    if isinstance(modes, bool) or not isinstance(modes, Integral) or modes < 1:
        raise ValueError("modes must be a positive integer")
    return int(modes)


def target_dim(modes: int) -> int:
    """Return the per-cell target width ``3 * modes`` for ``modes`` world target vectors.

    Args:
        modes: Number of target vectors per cell: 3 for the centre axes alone,
            :data:`TARGET_MODES_DEFAULT` with the four warping vectors.

    Returns:
        Number of scalar targets per cell (9 for 3 modes, 21 for 7).

    Raises:
        ValueError: If modes is not a positive integer.
    """
    return 3 * _require_modes(modes)


def state_feature_dim(modes: int) -> int:
    """Return the state width of :func:`pack_state_features` for ``modes`` target vectors per cell.

    Each of the five :data:`MATRIX_BLOCKS` packs ``target_dim(modes)`` values,
    followed by :data:`BOUNDARY_DIM` boundary flags and the
    :data:`SCALAR_FEATURES`.

    Args:
        modes: Number of target vectors per cell.

    Returns:
        State width (61 for 3 modes, 121 for 7).

    Raises:
        ValueError: If modes is not a positive integer.
    """
    return len(MATRIX_BLOCKS) * target_dim(modes) + BOUNDARY_DIM + len(SCALAR_FEATURES)


MATRIX_FEATURE_DIM = len(MATRIX_BLOCKS) * target_dim(3)
"""Width of the five matrix blocks in the legacy three-mode schema (45)."""

STATE_FEATURE_DIM = state_feature_dim(3)
"""Legacy three-mode state width (61): 45 matrix components, 14 boundary flags, 2 scalars."""

CONDITIONING_CHANNELS = (
    "log1p_lame_ratio",
    "log_inertia_ratio",
    "log1p_gravity_ratio",
    "log1p_viscosity_ratio",
    "log1p_contact_kappa",
    "contact_beta",
    "contact_mu",
)
"""Per-object dimensionless conditioning channels in order; see :func:`conditioning_channels`."""

CONDITIONING_DIM = len(CONDITIONING_CHANNELS)
"""Number of conditioning channels (7): four material/scale groups and three contact ratios."""

CONTACT_TOKEN_DIM = 19
"""Channels per contact token; layout in :func:`contact_features.build_contact_tokens`."""

CONTACT_FEATURE_DIM = 17
"""Pooled contact channels per cell appended to the node input: 16 learned plus ``count / M``."""

EDGE_FEATURE_DIM = 24
"""Directed-edge descriptor width from :func:`network_geometry.build_edge_features`."""

FEATURE_SCHEMA_VERSION = 6
"""Schema version stored in checkpoints.

Schema 6: :data:`TARGET_MODES_DEFAULT` (seven) target vectors per cell, so the
per-cell targets carry ``target_dim(7) = 21`` values and the five
:data:`MATRIX_BLOCKS` are 21-value blocks in the same seven-vector basis
(``state_feature_dim(7) = 121``; node input ``21 + 121`` plus the contact
channels), and gravity is a per-context quantity of the mixed step (its
``log1p_gravity_ratio`` channel, inertial prediction and rigid predictor read
the context's gravity). The conditioning channels, the LeCO normalisation and
the history unit ``mu h^3`` are those of schema 5. Schema 5 (three modes,
nine-value blocks, ``state_feature_dim(3) = 61``, one gravity per step) has a
different output head and node input width, so its checkpoints are not
loadable; schema 4 and older were already incompatible. Steps built with
``target_modes = 3`` still produce the schema-5 layout for ablations and
legacy tests.
"""

RMS_FLOOR = 1e-12
"""Lower bound on every input RMS before division."""

CLIP = 10.0
"""Symmetric clip applied to normalized local-frame components."""

_HISTORY_BLOCKS = ("previous_axis_gradient", "previous_axis_update")


def _require_tensor(name: str, value) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_cell_matrices(name: str, value, *, like: torch.Tensor | None = None) -> None:
    """Check a floating [B, C, 3, m] tensor with m >= 1, optionally matching a reference tensor."""
    _require_tensor(name, value)
    if value.ndim != 4 or value.shape[-2] != 3 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have shape [B, C, 3, m] with m >= 1")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if like is not None:
        if value.shape != like.shape:
            raise ValueError(f"{name} must share the [B, C, 3, m] shape of the other matrix blocks")
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


def _gravity_magnitude(gravity) -> float:
    """Return ``|g|`` from a nonnegative finite magnitude or a finite three-vector (sequence or tensor)."""
    import torch

    if isinstance(gravity, Real) and not isinstance(gravity, bool):
        if not math.isfinite(gravity) or gravity < 0:
            raise ValueError("gravity must be a nonnegative finite magnitude or a finite three-vector")
        return float(gravity)
    try:
        vector = torch.as_tensor(gravity, dtype=torch.float64).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("gravity must be a nonnegative finite magnitude or a finite three-vector") from error
    if vector.shape != (3,) or not torch.isfinite(vector).all():
        raise ValueError("gravity must be a nonnegative finite magnitude or a finite three-vector")
    return float(torch.linalg.vector_norm(vector))


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
    """Express world target vectors in each cell's frame as ``R^T @ V``.

    Frames store proper rotations with world columns. ``V`` holds ``m >= 1``
    world vectors in its columns: a deformation gradient, axis difference,
    gradient or update for ``m = 3``, or the axes and warping vectors of the
    seven-mode basis for ``m = 7``. Every column transforms as ``R^T v`` under
    a frame change, so one function serves every block and
    ``to_local(R, V)[..., :k]`` equals ``to_local(R, V[..., :k])``. Nothing is
    detached here; callers freeze frames before use.

    Args:
        frames: Local-to-world rotations, shape [..., 3, 3].
        world_matrices: World vectors in columns, shape [..., 3, m] with the
            leading dimensions of frames and any ``m >= 1``.

    Returns:
        Local matrices ``R^T V``, shape [..., 3, m].

    Raises:
        TypeError: If an input is not a floating tensor or the dtypes differ.
        ValueError: If frames is not [..., 3, 3], world_matrices is not
            [..., 3, m] with the leading shape of frames and ``m >= 1``, or
            the devices differ.
    """
    _require_tensor("frames", frames)
    _require_tensor("world_matrices", world_matrices)
    if frames.ndim < 2 or frames.shape[-2:] != (3, 3):
        raise ValueError("frames must have shape [..., 3, 3]")
    if (
        world_matrices.ndim != frames.ndim
        or world_matrices.shape[:-1] != frames.shape[:-1]
        or world_matrices.shape[-1] < 1
    ):
        raise ValueError("world_matrices must have shape [..., 3, m] with m >= 1 and the leading shape of frames")
    if not frames.is_floating_point() or world_matrices.dtype != frames.dtype:
        raise TypeError("frames and world_matrices must share one floating dtype")
    if world_matrices.device != frames.device:
        raise ValueError("world_matrices must be on the same device as frames")
    return frames.transpose(-1, -2) @ world_matrices


def rms_normalize(values: torch.Tensor, *, rms: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a per-object matrix field by a floored RMS and clip the result.

    The RMS is the root mean square over every cell matrix component of one
    object, ``sqrt(mean_{c,i,j} values^2)`` across all ``3 m`` components of
    every cell, floored at :data:`RMS_FLOOR` so a
    measured zero stays finite: zero input yields zero output and an RMS equal
    to the floor. Pass ``rms`` to reuse another field's statistics, as the
    previous gradient does with the current gradient RMS; the supplied RMS is
    floored again and returned as the second element. Clipping to
    ``+/-CLIP`` acts on the components as given, so pass local-frame values.

    Args:
        values: Local-frame matrix field, shape [B, C, 3, m] with ``m >= 1``.
        rms: Optional per-object RMS with one value per object (any shape with B
            elements, e.g. [B] or [B, 1, 1, 1]); computed from ``values`` when None.

    Returns:
        ``(normalized, rms)`` with normalized shaped like ``values`` and clipped to
        ``[-CLIP, CLIP]``, and rms shaped [B, 1, 1, 1] in the dtype of ``values``.

    Raises:
        TypeError: If values is not a floating tensor.
        ValueError: If values is not [B, C, 3, m] or rms does not hold B values.
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
    """Pack the per-cell state vector of width ``state_feature_dim(m)``.

    The five matrix blocks are ``[B, C, 3, m]`` local-frame blocks ``R^T V``
    with one common ``m >= 1`` (3 for the legacy axes-only schema, 7 with the
    warping vectors). Packing order along the last axis:
    ``inertial_axis_offset`` (``3 m``, row-major so component ``[i, j]`` lands
    at ``m i + j``), ``physical_axis_change`` (``3 m``),
    ``current_axis_gradient`` (``3 m``), ``previous_axis_gradient`` (``3 m``),
    ``previous_axis_update`` (``3 m``), exposed-face flags (6), fixed-corner
    flags (8), ``log_gradient_rms`` (1, broadcast per cell), ``history_valid``
    (1, exactly 1.0 or 0.0). For ``m = 3`` this is the schema-5 layout of
    width :data:`STATE_FEATURE_DIM`. All matrix blocks must already be
    expressed in the receiving local frame and normalized by the caller.
    Objects whose ``history_valid`` is false have both history blocks written
    as zeros regardless of the tensors supplied, so a missing history is always
    distinguishable from a measured zero.

    Args:
        inertial_axis_offset: ``R^T (V_target - V_current)``, shape [B, C, 3, m].
        physical_axis_change: ``R^T (V_current - V_previous)``, shape [B, C, 3, m].
        current_axis_gradient: Normalized current local target gradient, shape [B, C, 3, m].
        previous_axis_gradient: Normalized previous gradient in the current frame, shape [B, C, 3, m].
        previous_axis_update: Normalized previous achieved update in the current frame, shape [B, C, 3, m].
        boundary_features: Boundary flags, shape [C, 14] shared across objects or [B, C, 14].
        log_gradient_rms: Natural log of the current gradient RMS, one value per object.
        history_valid: One flag per object; boolean, or numeric where nonzero means valid.

    Returns:
        State features, shape [B, C, state_feature_dim(m)], in the dtype of the matrix blocks.

    Raises:
        TypeError: If an input is not a tensor or dtypes are incompatible.
        ValueError: If shapes or devices are incompatible, including blocks whose ``m`` differ.
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


def _check_per_object_tensors(reference_name: str, reference: torch.Tensor, others) -> None:
    """Require every named tensor to be a [B] floating tensor matching ``reference``."""
    for name, tensor in ((reference_name, reference), *others):
        _require_tensor(name, tensor)
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape [B]")
        if not tensor.is_floating_point() or tensor.dtype != reference.dtype:
            raise TypeError(f"{name} must share the floating dtype of {reference_name}")
        if tensor.shape != reference.shape:
            raise ValueError(f"{name} must share the [B] shape of {reference_name}")
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on the same device as {reference_name}")


def contact_ratios(
    lame_lambda: torch.Tensor,
    lame_mu: torch.Tensor,
    contact_ke: torch.Tensor,
    contact_kd: torch.Tensor,
    contact_mu: torch.Tensor,
    cell_size: float,
    time_step: float,
) -> torch.Tensor:
    """Return the dimensionless contact ratios ``(kappa, beta, mu)`` per object, shape [B, 3].

    ``kappa = ke / (E h)`` with the Young's modulus ``E = mu (3 lambda + 2 mu) /
    (lambda + mu)`` of the Lamé pair, ``beta = kd / (ke dt)`` (exactly zero
    where ``ke = 0``) and ``mu`` the friction coefficient unchanged. The
    conditioning channel ``log1p_contact_kappa`` and the contact tokens both
    derive from these ratios, so they share one definition. Material tensors
    are not value-checked (contexts validate them).

    Args:
        lame_lambda: First Lamé parameter [Pa], shape [B].
        lame_mu: Shear modulus [Pa], shape [B].
        contact_ke: Contact stiffness [N/m], shape [B].
        contact_kd: Contact damping [N s/m], shape [B].
        contact_mu: Friction coefficient, shape [B].
        cell_size: Positive finite rest voxel edge length [m].
        time_step: Positive finite physical time step [s].

    Returns:
        ``(kappa, beta, mu)`` stacked along the last axis, shape [B, 3], in the
        dtype and device of lame_mu.

    Raises:
        TypeError: If an input is not a floating tensor or dtypes differ.
        ValueError: If shapes, devices, or the scalar arguments are invalid.
    """
    import torch

    _check_per_object_tensors(
        "lame_mu",
        lame_mu,
        (
            ("lame_lambda", lame_lambda),
            ("contact_ke", contact_ke),
            ("contact_kd", contact_kd),
            ("contact_mu", contact_mu),
        ),
    )
    size = _require_positive_scalar("cell_size", cell_size)
    step = _require_positive_scalar("time_step", time_step)
    youngs_modulus = lame_mu * (3 * lame_lambda + 2 * lame_mu) / (lame_lambda + lame_mu)
    kappa = contact_ke / (youngs_modulus * size)
    stiff = contact_ke > 0
    # Divide by one where ke = 0 so the unselected branch stays finite.
    denominator = torch.where(stiff, contact_ke * step, torch.ones_like(contact_ke))
    beta = torch.where(stiff, contact_kd / denominator, torch.zeros_like(contact_kd))
    return torch.stack((kappa, beta, contact_mu), dim=-1)


def conditioning_channels(
    lame_lambda: torch.Tensor,
    lame_mu: torch.Tensor,
    density: torch.Tensor,
    damping: torch.Tensor,
    cell_size: float,
    time_step: float,
    gravity,
    *,
    contact_ke: torch.Tensor | None = None,
    contact_kd: torch.Tensor | None = None,
    contact_mu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the seven dimensionless per-object conditioning channels, shape [B, 7].

    Channel order follows :data:`CONDITIONING_CHANNELS`::

        log1p(lambda / mu), log(rho h^2 / (mu dt^2)), log1p(|g| dt^2 / h),
        log1p(eta / (mu dt)), log1p(ke / (E h)), kd / (ke dt), mu_contact

    with the Young's modulus ``E = mu (3 lambda + 2 mu) / (lambda + mu)``; see
    :func:`contact_ratios`. Every channel is a dimensionless group of the
    cell-normalisation law (``notes/ideas/idea-normalize-cells.md``): a scene
    and its normalised copy (``h' = dt' = mu' = 1`` with ``rho' = rho h^2 / (mu
    dt^2)``, ``g' = g dt^2 / h``, ``eta' = eta / (mu dt)``, ``ke' = ke / (mu
    h)``, ``kd' = kd / (mu h dt)``) produce identical channels, so no absolute
    reference scale enters the network. ``log1p`` is used wherever the ratio
    may legitimately be zero (``lambda = 0``, zero gravity, zero viscosity,
    no contact); the inertia ratio is strictly positive and uses ``log``. A
    contact argument left at None means zeros for every object, so
    contact-free objects produce ``log1p(0) = 0``, ``beta = 0`` and ``mu = 0``
    in the last three channels. Material tensors are not value-checked here
    (contexts validate them); nonpositive entries would produce nonfinite
    channels visibly. Callers expand the result to cells as needed.

    Args:
        lame_lambda: First Lamé parameter [Pa], shape [B].
        lame_mu: Shear modulus [Pa], shape [B].
        density: Mass density [kg/m^3], shape [B].
        damping: Viscosity eta [Pa s], shape [B].
        cell_size: Positive finite rest voxel edge length [m].
        time_step: Positive finite physical time step [s].
        gravity: Gravitational acceleration [m/s^2] as a nonnegative finite
            magnitude or a finite three-vector (sequence or tensor); only
            ``|g|`` enters the channel.
        contact_ke: Contact stiffness [N/m], shape [B]; None means zeros.
        contact_kd: Contact damping [N s/m], shape [B]; None means zeros.
        contact_mu: Contact friction coefficient, shape [B]; None means zeros.

    Returns:
        Conditioning channels, shape [B, 7], in the dtype and device of lame_mu.

    Raises:
        TypeError: If a material input is not a floating tensor or dtypes differ.
        ValueError: If shapes, devices, or the scalar arguments are invalid.
    """
    import torch

    _check_per_object_tensors(
        "lame_mu", lame_mu, (("lame_lambda", lame_lambda), ("density", density), ("damping", damping))
    )
    size = _require_positive_scalar("cell_size", cell_size)
    step = _require_positive_scalar("time_step", time_step)
    magnitude = _gravity_magnitude(gravity)
    contact = [torch.zeros_like(lame_mu) if value is None else value for value in (contact_ke, contact_kd, contact_mu)]
    ratios = contact_ratios(lame_lambda, lame_mu, *contact, size, step)

    channels = (
        (lame_lambda / lame_mu).log1p(),
        (density * size**2 / (lame_mu * step**2)).log(),
        torch.full_like(lame_mu, math.log1p(magnitude * step**2 / size)),
        (damping / (lame_mu * step)).log1p(),
        ratios[:, 0].log1p(),
        ratios[:, 1],
        ratios[:, 2],
    )
    return torch.stack(channels, dim=-1)
