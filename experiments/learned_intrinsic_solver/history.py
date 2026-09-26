# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental optimizer-history plumbing shared by the mixed trainer and validator.

The learned optimizer consumes two history blocks from the preceding learned
query: the world axis gradient feature and the achieved world change of the
center deformation. Both live in trajectory payloads as detached world
matrices, are expressed in the current frame only when the step consumes them,
are carried across physical timestep boundaries, and are cleared only when a
trajectory resets. Preparing a new candidate never writes them. Training and
validation must use the same functions so held-out queries receive the same
history the network was trained with.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING

from .mixed_physics import OptimizerHistory

if TYPE_CHECKING:
    import torch

__all__ = ["HISTORY_KEYS", "batch_history", "carry_history", "empty_history", "store_history"]

HISTORY_KEYS = ("history_axis_gradient_world", "history_axis_update_world", "history_valid")
"""Payload keys: world axis gradient [C,3,3], achieved world axis update [C,3,3], validity flag."""


def empty_history(cell_count: int) -> dict:
    """Return payload entries for a trajectory without optimizer history.

    Args:
        cell_count: Number of cells C in the canonical grid.

    Returns:
        Zero float32 CPU blocks of shape [C, 3, 3] and ``history_valid=False``.
    """
    import torch

    if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count < 1:
        raise ValueError("cell_count must be a positive integer")
    return {
        "history_axis_gradient_world": torch.zeros(cell_count, 3, 3, dtype=torch.float32),
        "history_axis_update_world": torch.zeros(cell_count, 3, 3, dtype=torch.float32),
        "history_valid": False,
    }


def _blocks(payload: Mapping, cell_count: int, device) -> tuple[torch.Tensor, torch.Tensor, bool]:
    import torch

    valid = bool(payload.get("history_valid", False))
    if not valid:
        zeros = torch.zeros(cell_count, 3, 3, dtype=torch.float32, device=device)
        return zeros, zeros, False
    blocks = []
    for name in HISTORY_KEYS[:2]:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (cell_count, 3, 3):
            raise ValueError(f"{name} must be a [C, 3, 3] tensor when history_valid is true")
        blocks.append(value.detach().to(device=device, dtype=torch.float32))
    return blocks[0], blocks[1], True


def batch_history(payloads: Iterable[Mapping], device, *, cell_count: int) -> OptimizerHistory:
    """Stack payload history into one detached batch on ``device``.

    Payloads without history entries, or with ``history_valid`` false, count
    as missing history: zero blocks and a false flag, which the step turns
    into zero input blocks and ``history_valid = 0``.

    Args:
        payloads: Trajectory payload mappings in batch order.
        device: Target device of the stacked tensors.
        cell_count: Number of cells C; validates stored block shapes.

    Returns:
        ``OptimizerHistory`` with float32 blocks [B, C, 3, 3] and a bool flag [B].
    """
    import torch

    payloads = list(payloads)
    if not payloads:
        raise ValueError("payloads must not be empty")
    gradients, updates, flags = [], [], []
    for payload in payloads:
        gradient, update, valid = _blocks(payload, cell_count, device)
        gradients.append(gradient)
        updates.append(update)
        flags.append(valid)
    return OptimizerHistory(
        torch.stack(gradients), torch.stack(updates), torch.tensor(flags, dtype=torch.bool, device=device)
    )


def store_history(payloads: Iterable[MutableMapping], result) -> None:
    """Write this query's detached history into each payload after a learned update.

    The stored gradient is the un-normalized world axis gradient that served as
    the query's current-gradient input; the stored update is the achieved world
    change of the center deformation produced by the fused update.

    Args:
        payloads: Payload mappings in the batch order of ``result``.
        result: ``LearnedHexStepOutput`` with ``axis_gradient_world`` and
            ``achieved_axis_update_world`` of shape [B, C, 3, 3].
    """
    payloads = list(payloads)
    gradient = getattr(result, "axis_gradient_world", None)
    update = getattr(result, "achieved_axis_update_world", None)
    if gradient is None or update is None:
        raise ValueError("result must carry axis_gradient_world and achieved_axis_update_world")
    if gradient.shape != update.shape or gradient.ndim != 4 or gradient.shape[0] != len(payloads):
        raise ValueError("history tensors must have shape [B, C, 3, 3] with one entry per payload")
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
