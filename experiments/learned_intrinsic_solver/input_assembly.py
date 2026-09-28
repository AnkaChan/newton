# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared per-object network input assembly for the mode-vector schema.

The single-material :class:`.solver_step.LearnedHexSolverStep` and the
mixed-material :class:`.mixed_physics.MixedHexSolverStep` both compose their
network inputs through :func:`assemble_inputs`. Every cell is described by
``m`` world target vectors packed as the columns of a ``[3, m]`` matrix ``V``
(:func:`target_vectors`): the three columns of the centre deformation gradient
``F`` for ``m = 3`` (the legacy nine-value schema) or those three axes followed
by the four warping vectors ``w_12, w_13, w_23, w_123`` of
:mod:`.hex_modes` for ``m = 7``; ``m`` is the step's ``target_modes``. The
assembly is unit-agnostic: it works in whatever length units the caller's
positions use (SI for the single-material step, the cell units ``h = dt = mu =
1`` for the mixed step) and in whatever unit ``project_gradient`` returns.
Both steps return the projected gradient in the energy unit ``S h^3`` (``S``
the shear modulus, ``h`` the cell size), so every input, the scalar
``log_gradient_rms`` included, is dimensionless and the two paths agree on
frames, matrix blocks, the log RMS, edge descriptors and conditioning. The
assembly performs, in order: the target vectors ``V`` and the closest proper
rotation frames of their centre ``F = V[..., :3]`` with the clamped-face
tie-break (:mod:`.frames`), the differentiable local axes ``R^T V``, the
inertial axis offset and physical axis change blocks (also ``[B, C, 3, m]``),
the detached position gradient of the complete physical objective projected
through the fusion adjoint onto the ``m`` modes, the LeCO normalization of the
current and previous gradients and the previous achieved update
(:mod:`.features`), the state packing, the directed edge descriptors
(:mod:`.network_geometry`, which read the affine part of the local axes) and
the conditioning pass-through.

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
from .hex_modes import mode_vectors
from .network_geometry import build_edge_features

__all__ = [
    "LearnedHexInputs",
    "LearnedHexStepOutput",
    "OptimizerHistory",
    "assemble_inputs",
    "check_history",
    "objective_gradient",
    "target_vectors",
]


class LearnedHexInputs(NamedTuple):
    """Experimental packed geometry and features for one network evaluation.

    ``local_axes`` are the current local target vectors ``R^T V``, shape
    [B, C, 3, m] with ``m`` the step's ``target_modes`` (columns 0..2 are the
    centre axes). The trailing fields carry detached diagnostics: the world
    target gradient feature before normalization, shape [B, C, 3, m], in units
    of ``S h^3`` (``S`` the shear modulus, ``h`` the cell size) for both steps,
    the position gradient of the SI objective with prescribed rows zeroed [N],
    and the frame tie-break mask. ``tie_mask`` is None when the caller
    replayed supplied frames instead of decomposing the deformation.
    ``contact_tokens`` [B, C, M, CONTACT_TOKEN_DIM] and ``contact_mask``
    [B, C, M] are the detached per-cell contact tokens built by
    :func:`.contact_features.build_contact_tokens`; both are None when the
    network does not consume contact tokens.
    """

    frames: Tensor
    local_axes: Tensor
    state_features: Tensor
    edge_features: dict[int, Tensor]
    conditioning: Tensor
    axis_gradient_world: Tensor | None = None
    position_gradient: Tensor | None = None
    tie_mask: Tensor | None = None
    contact_tokens: Tensor | None = None
    contact_mask: Tensor | None = None


class LearnedHexStepOutput(NamedTuple):
    """Experimental shared corners [m], raw local targets, frames, and energy [J].

    ``local_target_axes`` and ``axis_correction`` are the network outputs,
    shape [B, C, 3, m]; ``step_size`` is the per-cell dimensionless step,
    shape [B, C]. The trailing fields are detached diagnostics: the current
    query's world target gradient feature [B, C, 3, m] (units of ``S h^3``
    with ``S`` the shear modulus, for both steps), the achieved world change
    of the ``m`` target vectors produced by this fused update [B, C, 3, m]
    (fused minus pre-update candidate, dimensionless), the Euclidean norm of
    the free-corner position gradient at the pre-update candidate [N], and the
    frame tie-break mask. Positions and loss describe the fused update; no
    acceptance scaling or geometry backtracking is applied. ``contact_energy``
    [J] and ``contact_max_penetration`` (deepest penetration over the frozen
    contact pairs at the fused positions, in units of the sample radius r) are
    detached per-object diagnostics, shape [B], both zero when the batch has
    no contact pairs and None for steps without contact handling.
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
    contact_energy: Tensor | None = None
    contact_max_penetration: Tensor | None = None


class OptimizerHistory(NamedTuple):
    """Detached optimizer history from the previous learned query.

    World matrix coordinates keep the history independent of the frames, which
    are recomputed for every candidate; the step expresses both blocks in the
    current frame when it consumes them. Objects whose flag is False receive
    zero history blocks and ``history_valid = 0`` regardless of the tensors.
    Both blocks share the step's mode count ``m`` (``target_modes``).

    Attributes:
        axis_gradient_world: Previous query's world target gradient feature in
            the units the step returned it (``S h^3`` for both steps), shape
            [B, C, 3, m]. The step RMS-normalises it against the current
            gradient, so only consistency within an object matters.
        axis_update_world: Previous achieved world change of the ``m`` target
            vectors (fused minus pre-update candidate), shape [B, C, 3, m].
        valid: One boolean per object, shape [B].
    """

    axis_gradient_world: Tensor
    axis_update_world: Tensor
    valid: Tensor


def check_history(
    history: OptimizerHistory | None,
    *,
    batch: int,
    cell_count: int,
    dtype: torch.dtype,
    device: torch.device,
    modes: int = 3,
) -> OptimizerHistory | None:
    """Validate and detach optimizer history for a batch of ``batch`` objects.

    Args:
        history: ``OptimizerHistory`` or None (no history for the whole batch).
        batch: Number of objects B in the query.
        cell_count: Number of cells C of the canonical grid.
        dtype: Working dtype of the step (float32 in production).
        device: Module device that the history tensors must already use.
        modes: Target vectors per cell ``m`` of the consuming step (3 or 7).

    Returns:
        A detached ``OptimizerHistory`` with ``valid`` moved to ``device``, or None.

    Raises:
        TypeError: If ``history`` is neither None nor an ``OptimizerHistory``.
        ValueError: If a block is not a finite ``[B, C, 3, m]`` tensor of the
            working dtype on ``device`` or ``valid`` is not a ``[B]`` bool tensor.
    """
    if history is None:
        return None
    if not isinstance(history, OptimizerHistory):
        raise TypeError("history must be an OptimizerHistory or None")
    expected = (batch, cell_count, 3, modes)
    dtype_name = str(dtype).removeprefix("torch.")
    for name in ("axis_gradient_world", "axis_update_world"):
        value = getattr(history, name)
        if not isinstance(value, Tensor) or value.shape != expected:
            raise ValueError(f"history.{name} must have shape [B, C, 3, {modes}] matching positions")
        if value.dtype != dtype or value.device != device:
            raise ValueError(f"history.{name} must use {dtype_name} on the module device")
        if not torch.isfinite(value).all():
            raise ValueError(f"history.{name} must be finite")
    valid = history.valid
    if not isinstance(valid, Tensor) or valid.shape != (batch,) or valid.dtype != torch.bool:
        raise ValueError("history.valid must be a boolean tensor with one flag per object")
    return OptimizerHistory(history.axis_gradient_world.detach(), history.axis_update_world.detach(), valid.to(device))


def target_vectors(step, positions: Tensor) -> Tensor:
    """Return the world target vectors ``V`` of every cell, shape [B, C, 3, m].

    ``m = step.target_modes``. For three modes this is exactly
    :func:`.features.center_deformation` with the step's ``center_gradients``
    buffer, so the legacy affine-only path is reproduced bit for bit; for seven
    modes it is :func:`.hex_modes.mode_vectors` on the step's grid (``cell_size``
    in the step's length unit), whose first three columns equal the centre
    ``F`` and whose last four are the warping vectors. Differentiable in
    ``positions``.

    Args:
        step: Module exposing ``target_modes``, ``cell_corner_indices`` [C, 8],
            ``center_gradients`` [8, 3] and ``cell_size`` (see :func:`assemble_inputs`).
        positions: World corner positions, shape [B, P, 3], in the step's length unit.

    Returns:
        Target vectors with the mode vectors in columns, shape [B, C, 3, m].
    """
    if step.target_modes == 3:
        return center_deformation(positions, step.cell_corner_indices, step.center_gradients)
    return mode_vectors(positions, step.cell_corner_indices, step.cell_size)


def objective_gradient(
    positions: Tensor,
    inertial_prediction: Tensor,
    previous_positions: Tensor,
    energy_total: Callable[[Tensor, Tensor, Tensor], Tensor],
    fixed_indices: Tensor,
) -> Tensor:
    """Return the detached position gradient of the objective with zeroed pins.

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
    contact_tokens: Tensor | None = None,
    contact_mask: Tensor | None = None,
) -> LearnedHexInputs:
    """Compose the network inputs for a batch of same-grid objects.

    ``step`` supplies the canonical-grid buffers and the network; it must expose
    ``target_modes`` (3 or 7), ``cell_corner_indices`` [C, 8],
    ``center_gradients`` [8, 3], ``rest_centers`` [C, 3], ``cell_size``,
    ``reference_corners`` (long [3] or empty), ``fixed_indices`` [K],
    ``boundary_features`` [C, 14] and ``network`` (for ``hops`` and
    ``neighborhood``; its ``target_modes`` must equal the step's). Inputs are
    assumed already validated (finite, matching shapes, module dtype and device).

    With ``V`` the target vectors of :func:`target_vectors` (``[B, C, 3, m]``),
    the frames are the closest proper rotations of the centre ``V[..., :3]``
    and the state packing (see :func:`.features.pack_state_features`) is
    ``R^T (V_Y - V)``, ``R^T (V - V_prev)``, ``clip(R^T G / rms)``,
    ``clip(R^T G_prev / rms)``, ``clip(R^T U_prev / rms_own)``, boundary
    flags, ``log rms`` and the history flag, where ``G`` is the fusion-adjoint
    projection of the zero-pinned position gradient onto the ``m`` modes and
    ``rms`` its per-object RMS over all ``3 m`` components per cell, floored at
    :data:`.features.RMS_FLOOR`. For ``m = 3`` every quantity is the one of
    the nine-value schema.

    Args:
        step: Module exposing the buffers and network listed above.
        positions: Candidate world corners [m], shape [B, P, 3].
        inertial_prediction: Unchanged physical Y [m], same shape.
        previous_positions: Physical-step start [m], same shape.
        energy_total: Callable ``(candidate, Y, X_prev) -> total energy [B]``
            of the complete physical objective for this batch.
        project_gradient: Callable mapping the zero-pinned position gradient
            [B, P, 3] to world target-increment gradients [B, C, 3, m] through
            the fusion adjoint of each object.
        conditioning: Prepared per-cell conditioning channels [B, C, K].
        history: Validated detached history or None (zero blocks, flag 0).
        frames: Optional validated, detached proper rotations [B, C, 3, 3] to
            replay instead of decomposing ``F``; ``tie_mask`` is then None.
        contact_tokens: Optional prepared contact tokens [B, C, M, CONTACT_TOKEN_DIM]
            passed through unchanged; the assembly does not build them.
        contact_mask: Boolean valid-token flags [B, C, M] accompanying
            ``contact_tokens``; both or neither must be given.

    Returns:
        ``LearnedHexInputs`` with detached frames, differentiable local axes
        [B, C, 3, m], packed state [B, C, state_feature_dim(m)], edge and
        conditioning features, the detached world target gradient [B, C, 3, m],
        the zero-pinned position gradient, the tie mask and the contact tokens
        as supplied.

    Raises:
        ValueError: If exactly one of ``contact_tokens`` and ``contact_mask`` is given.
    """
    if (contact_tokens is None) != (contact_mask is None):
        raise ValueError("contact_tokens and contact_mask must be given together")
    cells = step.cell_corner_indices
    vectors = target_vectors(step, positions)
    tie_mask = None
    if frames is None:
        reference = None
        if step.reference_corners.numel():
            reference = reference_rotation(positions, step.reference_corners)
        result = closest_proper_rotations(vectors[..., :3], reference)
        frames, tie_mask = result.frames, result.tie_mask
    axes = to_local(frames, vectors)
    target_vectors_y = target_vectors(step, inertial_prediction)
    previous_vectors = target_vectors(step, previous_positions)
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
        inertial_axis_offset=to_local(frames, target_vectors_y - vectors),
        physical_axis_change=to_local(frames, vectors - previous_vectors),
        current_axis_gradient=current_gradient,
        previous_axis_gradient=previous_gradient,
        previous_axis_update=previous_update,
        boundary_features=step.boundary_features,
        log_gradient_rms=gradient_rms.log(),
        history_valid=history_valid,
    )
    corners = positions[:, cells]
    # The edge descriptors transport the affine part of the neighbour axes (the centre F in the local frame).
    affine_axes = axes[..., :3]
    edges = {
        hop: build_edge_features(
            step.rest_centers, corners.mean(-2), frames, affine_axes, step.cell_size, *step.network.neighborhood(hop)
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
        contact_tokens=contact_tokens,
        contact_mask=contact_mask,
    )
