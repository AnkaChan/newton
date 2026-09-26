# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared per-object network input assembly for the revised nine-value schema.

The single-material :class:`.solver_step.LearnedHexSolverStep` and the
mixed-material :class:`.mixed_physics.MixedHexSolverStep` both compose their
network inputs through :func:`assemble_inputs`, so one object queried through
either path receives identical frames, state features, edge descriptors and
conditioning. The assembly performs, in order: the cell-center deformation
``F`` and its closest proper rotation frames with the clamped-face tie-break
(:mod:`.frames`), the differentiable local axes ``A = R^T F``, the inertial
axis offset and physical axis change blocks, the detached position gradient of
the complete physical objective projected through the fusion adjoint, the LeCO
normalization of the current and previous gradients and the previous achieved
update (:mod:`.features`), the state packing, the directed edge descriptors
(:mod:`.network_geometry`) and the conditioning pass-through.

Frames, the gradient feature and the history are detached here; local axes and
the two deformation-difference blocks remain differentiable. This module also
owns the shared interface tuples: :class:`LearnedHexInputs`,
:class:`LearnedHexStepOutput` and :class:`OptimizerHistory`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor  # noqa: TID253

from .features import center_deformation, pack_state_features, rms_normalize, to_local
from .frames import closest_proper_rotations, reference_rotation
from .hex_energy import HexLossTerms
from .network_geometry import build_edge_features

__all__ = [
    "LearnedHexInputs",
    "LearnedHexStepOutput",
    "OptimizerHistory",
    "assemble_inputs",
    "check_history",
    "objective_gradient",
]


class LearnedHexInputs(NamedTuple):
    """Experimental packed geometry and features for one network evaluation.

    The trailing fields carry detached diagnostics of the revised nine-value
    schema: the world axis gradient feature before normalization [J], the
    position gradient of the physical objective [N] with prescribed rows
    zeroed, and the frame tie-break mask. ``tie_mask`` is None when the caller
    replayed supplied frames instead of decomposing the deformation.
    """

    frames: Tensor
    local_axes: Tensor
    state_features: Tensor
    edge_features: dict[int, Tensor]
    conditioning: Tensor
    axis_gradient_world: Tensor | None = None
    position_gradient: Tensor | None = None
    tie_mask: Tensor | None = None


class LearnedHexStepOutput(NamedTuple):
    """Experimental shared corners [m], raw local targets, frames, and energy [J].

    ``step_size`` is the per-cell dimensionless step, shape [B, C]. The trailing
    fields are detached diagnostics of the revised schema: the current query's
    world axis gradient feature [J], the achieved world change of the center
    deformation produced by this fused update, the Euclidean norm of the
    free-corner position gradient at the pre-update candidate [N], and the
    frame tie-break mask. Positions and loss describe the fused update; no
    acceptance scaling or geometry backtracking is applied.
    """

    positions: Tensor
    local_target_axes: Tensor
    axis_correction: Tensor
    step_size: Tensor
    frames: Tensor
    loss: HexLossTerms
    axis_gradient_world: Tensor | None = None
    achieved_axis_update_world: Tensor | None = None
    force_residual_norm: Tensor | None = None
    tie_mask: Tensor | None = None


class OptimizerHistory(NamedTuple):
    """Detached optimizer history from the previous learned query.

    World matrix coordinates keep the history independent of the frames, which
    are recomputed for every candidate; the step expresses both blocks in the
    current frame when it consumes them. Objects whose flag is False receive
    zero history blocks and ``history_valid = 0`` regardless of the tensors.

    Attributes:
        axis_gradient_world: Previous query's world axis gradient feature [J],
            shape [B, C, 3, 3].
        axis_update_world: Previous achieved world change of the center
            deformation (fused minus pre-update candidate), shape [B, C, 3, 3].
        valid: One boolean per object, shape [B].
    """

    axis_gradient_world: Tensor
    axis_update_world: Tensor
    valid: Tensor


def check_history(
    history: OptimizerHistory | None, *, batch: int, cell_count: int, dtype: torch.dtype, device: torch.device
) -> OptimizerHistory | None:
    """Validate and detach optimizer history for a batch of ``batch`` objects.

    Args:
        history: ``OptimizerHistory`` or None (no history for the whole batch).
        batch: Number of objects B in the query.
        cell_count: Number of cells C of the canonical grid.
        dtype: Working dtype of the step (float32 in production).
        device: Module device that the history tensors must already use.

    Returns:
        A detached ``OptimizerHistory`` with ``valid`` moved to ``device``, or None.

    Raises:
        TypeError: If ``history`` is neither None nor an ``OptimizerHistory``.
        ValueError: If a block is not a finite ``[B, C, 3, 3]`` tensor of the
            working dtype on ``device`` or ``valid`` is not a ``[B]`` bool tensor.
    """
    if history is None:
        return None
    if not isinstance(history, OptimizerHistory):
        raise TypeError("history must be an OptimizerHistory or None")
    expected = (batch, cell_count, 3, 3)
    dtype_name = str(dtype).removeprefix("torch.")
    for name in ("axis_gradient_world", "axis_update_world"):
        value = getattr(history, name)
        if not isinstance(value, Tensor) or value.shape != expected:
            raise ValueError(f"history.{name} must have shape [B, C, 3, 3] matching positions")
        if value.dtype != dtype or value.device != device:
            raise ValueError(f"history.{name} must use {dtype_name} on the module device")
        if not torch.isfinite(value).all():
            raise ValueError(f"history.{name} must be finite")
    valid = history.valid
    if not isinstance(valid, Tensor) or valid.shape != (batch,) or valid.dtype != torch.bool:
        raise ValueError("history.valid must be a boolean tensor with one flag per object")
    return OptimizerHistory(history.axis_gradient_world.detach(), history.axis_update_world.detach(), valid.to(device))


def objective_gradient(
    positions: Tensor,
    inertial_prediction: Tensor,
    previous_positions: Tensor,
    energy_total: Callable[[Tensor, Tensor, Tensor], Tensor],
    fixed_indices: Tensor,
) -> Tensor:
    """Return the detached position gradient [N] of the physical objective with zeroed pins.

    The gradient is evaluated under ``torch.enable_grad()`` at the detached
    candidate with the inertial prediction and physical-step start held fixed,
    so callers may run the whole query under ``torch.no_grad()``.

    Args:
        positions: Candidate world corners [m], shape [B, P, 3].
        inertial_prediction: Unchanged physical Y [m], same shape.
        previous_positions: Physical-step start [m], same shape.
        energy_total: Callable ``(candidate, Y, X_prev) -> total energy [B]``.
        fixed_indices: Long indices of prescribed corners whose rows are zeroed.

    Returns:
        Detached gradient, shape [B, P, 3], in the dtype/device of positions.
    """
    with torch.enable_grad():
        candidate = positions.detach().requires_grad_(True)
        total = energy_total(candidate, inertial_prediction.detach(), previous_positions.detach())
        gradient = torch.autograd.grad(total.sum(), candidate)[0]
    gradient = gradient.detach()
    gradient[:, fixed_indices] = 0
    return gradient


def assemble_inputs(
    step,
    positions: Tensor,
    inertial_prediction: Tensor,
    previous_positions: Tensor,
    *,
    energy_total: Callable[[Tensor, Tensor, Tensor], Tensor],
    project_gradient: Callable[[Tensor], Tensor],
    conditioning: Tensor,
    history: OptimizerHistory | None = None,
    frames: Tensor | None = None,
) -> LearnedHexInputs:
    """Compose the schema-3 network inputs for a batch of same-grid objects.

    ``step`` supplies the canonical-grid buffers and the network; it must expose
    ``cell_corner_indices`` [C, 8], ``center_gradients`` [8, 3],
    ``rest_centers`` [C, 3], ``cell_size``, ``reference_corners`` (long [3] or
    empty), ``fixed_indices`` [K], ``boundary_features`` [C, 14] and
    ``network`` (for ``hops`` and ``neighborhood``). Inputs are assumed already
    validated (finite, matching shapes, module dtype and device).

    State packing (see :func:`.features.pack_state_features`): ``R^T (F_Y - F)``,
    ``R^T (F - F_prev)``, ``clip(R^T G / rms)``, ``clip(R^T G_prev / rms)``,
    ``clip(R^T U_prev / rms_own)``, boundary flags, ``log rms`` and the history
    flag, where ``G`` is the fusion-adjoint projection of the zero-pinned
    position gradient and ``rms`` its per-object RMS floored at
    :data:`.features.RMS_FLOOR`.

    Args:
        step: Module exposing the buffers and network listed above.
        positions: Candidate world corners [m], shape [B, P, 3].
        inertial_prediction: Unchanged physical Y [m], same shape.
        previous_positions: Physical-step start [m], same shape.
        energy_total: Callable ``(candidate, Y, X_prev) -> total energy [B]``
            of the complete physical objective for this batch.
        project_gradient: Callable mapping the zero-pinned position gradient
            [B, P, 3] to world axis-increment gradients [B, C, 3, 3] through
            the fusion adjoint of each object.
        conditioning: Prepared per-cell conditioning channels [B, C, K].
        history: Validated detached history or None (zero blocks, flag 0).
        frames: Optional validated, detached proper rotations [B, C, 3, 3] to
            replay instead of decomposing ``F``; ``tie_mask`` is then None.

    Returns:
        ``LearnedHexInputs`` with detached frames, differentiable local axes,
        packed state, edge and conditioning features, the detached world axis
        gradient, the zero-pinned position gradient and the tie mask.
    """
    cells, gradients = step.cell_corner_indices, step.center_gradients
    deformation = center_deformation(positions, cells, gradients)
    tie_mask = None
    if frames is None:
        reference = None
        if step.reference_corners.numel():
            reference = reference_rotation(positions, step.reference_corners)
        result = closest_proper_rotations(deformation, reference)
        frames, tie_mask = result.frames, result.tie_mask
    axes = to_local(frames, deformation)
    target_deformation = center_deformation(inertial_prediction, cells, gradients)
    previous_deformation = center_deformation(previous_positions, cells, gradients)
    position_gradient = objective_gradient(
        positions, inertial_prediction, previous_positions, energy_total, step.fixed_indices
    )
    world_gradient = project_gradient(position_gradient)
    current_gradient, gradient_rms = rms_normalize(to_local(frames, world_gradient))
    if history is None:
        previous_gradient = torch.zeros_like(current_gradient)
        previous_update = torch.zeros_like(current_gradient)
        history_valid = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    else:
        previous_gradient, _ = rms_normalize(to_local(frames, history.axis_gradient_world), rms=gradient_rms)
        previous_update, _ = rms_normalize(to_local(frames, history.axis_update_world))
        history_valid = history.valid
    state = pack_state_features(
        inertial_axis_offset=to_local(frames, target_deformation - deformation),
        physical_axis_change=to_local(frames, deformation - previous_deformation),
        current_axis_gradient=current_gradient,
        previous_axis_gradient=previous_gradient,
        previous_axis_update=previous_update,
        boundary_features=step.boundary_features,
        log_gradient_rms=gradient_rms.log(),
        history_valid=history_valid,
    )
    corners = positions[:, cells]
    edges = {
        hop: build_edge_features(
            step.rest_centers, corners.mean(-2), frames, axes, step.cell_size, *step.network.neighborhood(hop)
        )
        for hop in set(step.network.hops)
    }
    return LearnedHexInputs(
        frames,
        axes,
        state,
        edges,
        conditioning,
        axis_gradient_world=world_gradient,
        position_gradient=position_gradient,
        tie_mask=tie_mask,
    )
