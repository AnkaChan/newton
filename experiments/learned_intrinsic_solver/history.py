# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental optimizer-history plumbing shared by the mixed trainer and validator.

The learned optimizer consumes two history blocks from the preceding learned
query: the world target gradient feature and the achieved world change of the
per-cell target vectors. Both live in trajectory payloads as detached world
matrices of shape ``[C, 3, m]`` with ``m`` the step's ``target_modes`` (3 for
the legacy affine-only schema, 7 with the warping vectors), are expressed in
the current frame only when the step consumes them, are carried across
physical timestep boundaries, and are cleared only when a trajectory resets.
Preparing a new candidate never writes them. Training and validation must use
the same functions so held-out queries receive the same history the network
was trained with. The mode count is never stored separately: it is read from
the tensors themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING

from .mixed_physics import OptimizerHistory

if TYPE_CHECKING:
    import torch

__all__ = ["HISTORY_KEYS", "batch_history", "carry_history", "empty_history", "store_history"]

HISTORY_KEYS = ("history_axis_gradient_world", "history_axis_update_world", "history_valid")
"""Payload keys: world target gradient [C,3,m], achieved world target update [C,3,m], validity flag."""

_LEGACY_MODES = 3
"""Mode count assumed when no payload of a batch carries a history block (the affine-only schema)."""


def _require_modes(modes) -> int:
    if isinstance(modes, bool) or not isinstance(modes, int) or modes < 1:
        raise ValueError("modes must be a positive integer")
    return int(modes)


def empty_history(cell_count: int, modes: int = _LEGACY_MODES) -> dict:
    """Return payload entries for a trajectory without optimizer history.

    Args:
        cell_count: Number of cells C in the canonical grid.
        modes: Target vectors per cell ``m`` of the consuming step (3 or 7).

    Returns:
        Zero float32 CPU blocks of shape [C, 3, m] and ``history_valid=False``.
    """
    import torch

    if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count < 1:
        raise ValueError("cell_count must be a positive integer")
    modes = _require_modes(modes)
    return {
        "history_axis_gradient_world": torch.zeros(cell_count, 3, modes, dtype=torch.float32),
        "history_axis_update_world": torch.zeros(cell_count, 3, modes, dtype=torch.float32),
        "history_valid": False,
    }


def _infer_modes(payloads: list[Mapping], cell_count: int) -> int:
    """Return ``m`` from the first stored block of shape [C, 3, m]; the legacy 3 when no payload stores one."""
    import torch

    for payload in payloads:
        for name in HISTORY_KEYS[:2]:
            value = payload.get(name)
            if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[:2] == (cell_count, 3):
                return int(value.shape[2])
    return _LEGACY_MODES


def _blocks(payload: Mapping, cell_count: int, device, modes: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    import torch

    valid = bool(payload.get("history_valid", False))
    if not valid:
        zeros = torch.zeros(cell_count, 3, modes, dtype=torch.float32, device=device)
        return zeros, zeros, False
    blocks = []
    for name in HISTORY_KEYS[:2]:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (cell_count, 3, modes):
            raise ValueError(f"{name} must be a [C, 3, {modes}] tensor when history_valid is true")
        blocks.append(value.detach().to(device=device, dtype=torch.float32))
    return blocks[0], blocks[1], True


def batch_history(
    payloads: Iterable[Mapping], device, *, cell_count: int, modes: int | None = None
) -> OptimizerHistory:
    """Stack payload history into one detached batch on ``device``.

    Payloads without history entries, or with ``history_valid`` false, count
    as missing history: zero blocks and a false flag, which the step turns
    into zero input blocks and ``history_valid = 0``. Every stored block of
    the batch must share one mode count ``m``.

    Args:
        payloads: Trajectory payload mappings in batch order.
        device: Target device of the stacked tensors.
        cell_count: Number of cells C; validates stored block shapes.
        modes: Target vectors per cell ``m``; None reads it from the first
            payload that stores a block (valid or not) and falls back to the
            legacy 3 when none does.

    Returns:
        ``OptimizerHistory`` with float32 blocks [B, C, 3, m] and a bool flag [B].

    Raises:
        ValueError: If ``payloads`` is empty, ``modes`` is not a positive
            integer, or a valid payload stores a block of another shape.
    """
    import torch

    payloads = list(payloads)
    if not payloads:
        raise ValueError("payloads must not be empty")
    modes = _infer_modes(payloads, cell_count) if modes is None else _require_modes(modes)
    gradients, updates, flags = [], [], []
    for payload in payloads:
        gradient, update, valid = _blocks(payload, cell_count, device, modes)
        gradients.append(gradient)
        updates.append(update)
        flags.append(valid)
    return OptimizerHistory(
        torch.stack(gradients), torch.stack(updates), torch.tensor(flags, dtype=torch.bool, device=device)
    )


def store_history(payloads: Iterable[MutableMapping], result) -> None:
    """Write this query's detached history into each payload after a learned update.

    The stored gradient is the world target gradient that served as the query's
    current-gradient input before RMS normalization, in the producing step's
    energy unit ``S h^3`` (both solver steps use it, see
    ``LearnedHexSolverStep.energy_unit``); the stored update is the achieved
    world change of the ``m`` target vectors produced by the fused update.

    Args:
        payloads: Payload mappings in the batch order of ``result``.
        result: ``LearnedHexStepOutput`` with ``axis_gradient_world`` and
            ``achieved_axis_update_world`` of shape [B, C, 3, m].
    """
    payloads = list(payloads)
    gradient = getattr(result, "axis_gradient_world", None)
    update = getattr(result, "achieved_axis_update_world", None)
    if gradient is None or update is None:
        raise ValueError("result must carry axis_gradient_world and achieved_axis_update_world")
    if (
        gradient.shape != update.shape
        or gradient.ndim != 4
        or gradient.shape[2] != 3
        or gradient.shape[0] != len(payloads)
    ):
        raise ValueError("history tensors must have shape [B, C, 3, m] with one entry per payload")
    for index, payload in enumerate(payloads):
        payload["history_axis_gradient_world"] = gradient[index].detach()
        payload["history_axis_update_world"] = update[index].detach()
        payload["history_valid"] = True


def carry_history(source: Mapping, target: MutableMapping) -> None:
    """Copy optimizer history unchanged across a physical timestep boundary.

    A missing source entry is left missing, which ``batch_history`` treats as
    no history. Initializer motion of the new candidate never enters history.
    """
    for name in HISTORY_KEYS:
        if name in source:
            target[name] = source[name]
