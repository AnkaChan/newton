# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental schema-4 contact tokens for the per-cell contact encoder.

Section 6 of ``notes/contact-design-20260927.md``: every frozen contact pair
(surface sample, static partner) becomes one :data:`.features.CONTACT_TOKEN_DIM`
token expressed in the frame of the cell that owns the sample. Tokens are
grouped per owning cell, capped at ``tokens_per_cell`` slots and padded with
zero rows and a False mask, ready for
:class:`.contact_network.ContactEncoder`. Everything here is detached: the
tokens describe the current candidate to the network and never carry
gradients, just like the frames and the gradient feature.

Token layout along the last axis (all dimensionless):

======  ==========================================================
index   value
======  ==========================================================
0-2     contact point in the owner-cell frame, ``R_i^T (x_s - c_i) / h``
3-5     partner point in the owner-cell frame, ``R_i^T (p - c_i) / h``
6-8     partner normal in the owner-cell frame, ``R_i^T n``
9       ``gap / r`` with ``gap = (x_s - p) . n``
10      approach rate ``-(n . (x_s - x_s0)) / r`` over the current step
11      ``min(r_p / r, 10)`` lateral partner radius
12-14   ``log1p(kappa)``, ``beta``, ``mu`` of the object (see :func:`.features.contact_ratios`)
15-17   kind one-hot: plane, point, self
18      self flag, always 0 in version 1
======  ==========================================================
"""

from __future__ import annotations

import math
from numbers import Real

import torch  # noqa: TID253 -- Explicit opt-in PyTorch implementation.
from torch import Tensor  # noqa: TID253

from .features import CONTACT_TOKEN_DIM

__all__ = ["RADIUS_RATIO_CAP", "build_contact_tokens"]

RADIUS_RATIO_CAP = 10.0
"""Upper bound of the ``r_p / r`` token channel; the plane's large radius saturates here."""

_KIND_COUNT = 3


def _require_tensor(name: str, value) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(value.shape)}")


def _require_floating(name: str, value: Tensor, like: Tensor) -> None:
    if value.dtype != like.dtype or value.device != like.device:
        raise TypeError(f"{name} must share the floating dtype and device of frames")


def _require_integer(name: str, value: Tensor, like: Tensor) -> None:
    if value.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"{name} must be an int64 or int32 tensor")
    if value.device != like.device:
        raise ValueError(f"{name} must be on the device of frames")


def _positive_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _gather_rows(values: Tensor, index: Tensor) -> Tensor:
    """Gather ``values[b, index[b, q]]`` for trailing shapes of any rank, result [B, Q, ...]."""
    batch = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[batch, index]


def build_contact_tokens(
    *,
    frames: Tensor,
    cell_centers: Tensor,
    cell_size: float,
    radius: float,
    face_cell_index: Tensor,
    sample_positions: Tensor,
    sample_start_positions: Tensor,
    sample_index: Tensor,
    kind: Tensor,
    partner_point: Tensor,
    partner_normal: Tensor,
    partner_radius: Tensor,
    pair_mask: Tensor,
    contact_kappa: Tensor,
    contact_beta: Tensor,
    contact_mu: Tensor,
    tokens_per_cell: int,
) -> tuple[Tensor, Tensor]:
    """Group the frozen contact pairs of a batch into padded per-cell tokens.

    The owner of a pair is the cell of its surface sample,
    ``face_cell_index[sample_index]``. Within a cell the first
    ``tokens_per_cell`` valid pairs in pair order fill the slots in that order
    and later pairs are dropped, so the result is deterministic and stable
    under padding. Slots without a pair hold zeros and a False mask. The
    computation is vectorized over all pairs (a stable sort by owner followed
    by a cumulative per-owner count) and runs under ``torch.no_grad``: both
    outputs are detached.

    Args:
        frames: Detached proper cell frames with world columns, shape [B, C, 3, 3].
        cell_centers: Current cell centers [m], mean of the eight corners, shape [B, C, 3].
        cell_size: Positive rest cell edge length h [m].
        radius: Positive sample radius r [m].
        face_cell_index: Owning cell of every surface sample, int64, shape [S].
        sample_positions: Current surface sample positions [m], shape [B, S, 3].
        sample_start_positions: Step-start sample positions [m], shape [B, S, 3].
        sample_index: Sample of every pair, integer, shape [B, Q]; ignored where masked.
        kind: Partner kind of every pair (0 plane, 1 point, 2 self), integer, shape [B, Q].
        partner_point: Partner point p [m], shape [B, Q, 3]; ignored where masked.
        partner_normal: Unit partner normal n, shape [B, Q, 3]; ignored where masked.
        partner_radius: Lateral partner radius r_p [m], shape [B, Q]; ignored where masked.
        pair_mask: True for valid pairs, bool, shape [B, Q].
        contact_kappa: Dimensionless contact stiffness ratio ``ke / (E h)`` per object, shape [B].
        contact_beta: Dimensionless contact damping ratio ``kd / (ke dt)`` per object, shape [B].
        contact_mu: Friction coefficient per object, shape [B].
        tokens_per_cell: Slot count M per cell, at least 1.

    Returns:
        ``(tokens, mask)`` with tokens of shape [B, C, M, CONTACT_TOKEN_DIM] in
        the dtype of ``frames`` and a boolean mask of shape [B, C, M]. With
        ``Q = 0`` or no valid pair both are all zeros and all False.

    Raises:
        TypeError: If an input is not a tensor of the expected dtype family.
        ValueError: If shapes or devices are inconsistent, a scalar is not
            positive and finite, a valid pair indexes a sample outside ``[0, S)``
            or has a kind outside ``{0, 1, 2}``, or ``face_cell_index`` leaves ``[0, C)``.
    """
    _require_tensor("frames", frames)
    if frames.ndim != 4 or frames.shape[-2:] != (3, 3) or not frames.is_floating_point():
        raise ValueError("frames must be a floating tensor of shape [B, C, 3, 3]")
    batch, cells = frames.shape[:2]
    _require_tensor("cell_centers", cell_centers)
    _require_shape("cell_centers", cell_centers, (batch, cells, 3))
    _require_floating("cell_centers", cell_centers, frames)
    size = _positive_scalar("cell_size", cell_size)
    sample_radius = _positive_scalar("radius", radius)
    if isinstance(tokens_per_cell, bool) or not isinstance(tokens_per_cell, int) or tokens_per_cell < 1:
        raise ValueError("tokens_per_cell must be an integer >= 1")

    _require_tensor("face_cell_index", face_cell_index)
    if face_cell_index.ndim != 1 or face_cell_index.dtype != torch.int64:
        raise TypeError("face_cell_index must be an int64 tensor of shape [S]")
    if face_cell_index.device != frames.device:
        raise ValueError("face_cell_index must be on the device of frames")
    sample_count = face_cell_index.shape[0]
    if sample_count and (face_cell_index.min() < 0 or face_cell_index.max() >= cells):
        raise ValueError("face_cell_index must lie in [0, C)")
    for name, value in (("sample_positions", sample_positions), ("sample_start_positions", sample_start_positions)):
        _require_tensor(name, value)
        _require_shape(name, value, (batch, sample_count, 3))
        _require_floating(name, value, frames)

    _require_tensor("sample_index", sample_index)
    if sample_index.ndim != 2 or sample_index.shape[0] != batch:
        raise ValueError("sample_index must have shape [B, Q]")
    _require_integer("sample_index", sample_index, frames)
    pairs = sample_index.shape[1]
    _require_tensor("kind", kind)
    _require_shape("kind", kind, (batch, pairs))
    _require_integer("kind", kind, frames)
    for name, value in (("partner_point", partner_point), ("partner_normal", partner_normal)):
        _require_tensor(name, value)
        _require_shape(name, value, (batch, pairs, 3))
        _require_floating(name, value, frames)
    _require_tensor("partner_radius", partner_radius)
    _require_shape("partner_radius", partner_radius, (batch, pairs))
    _require_floating("partner_radius", partner_radius, frames)
    _require_tensor("pair_mask", pair_mask)
    _require_shape("pair_mask", pair_mask, (batch, pairs))
    if pair_mask.dtype != torch.bool or pair_mask.device != frames.device:
        raise TypeError("pair_mask must be a boolean tensor on the device of frames")
    for name, value in (("contact_kappa", contact_kappa), ("contact_beta", contact_beta), ("contact_mu", contact_mu)):
        _require_tensor(name, value)
        _require_shape(name, value, (batch,))
        _require_floating(name, value, frames)

    with torch.no_grad():
        tokens = frames.new_zeros((batch, cells, tokens_per_cell, CONTACT_TOKEN_DIM))
        mask = torch.zeros((batch, cells, tokens_per_cell), dtype=torch.bool, device=frames.device)
        if pairs == 0 or not bool(pair_mask.any()):
            return tokens, mask

        valid = pair_mask
        index = sample_index.to(torch.int64)
        if bool((valid & ((index < 0) | (index >= sample_count))).any()):
            raise ValueError("sample_index of valid pairs must lie in [0, S)")
        kinds = kind.to(torch.int64)
        if bool((valid & ((kinds < 0) | (kinds >= _KIND_COUNT))).any()):
            raise ValueError("kind of valid pairs must be 0 (plane), 1 (point) or 2 (self)")
        # Masked rows are replaced by finite placeholders so no NaN can leak through the gathers.
        index = torch.where(valid, index, torch.zeros_like(index))
        kinds = torch.where(valid, kinds, torch.zeros_like(kinds))
        row = valid[..., None]
        point = torch.where(row, partner_point, torch.zeros_like(partner_point))
        normal = torch.where(row, partner_normal, torch.zeros_like(partner_normal))
        lateral = torch.where(valid, partner_radius, torch.zeros_like(partner_radius))

        owner = face_cell_index[index]
        current = _gather_rows(sample_positions, index)
        start = _gather_rows(sample_start_positions, index)
        centers = _gather_rows(cell_centers, owner)
        rotation = _gather_rows(frames, owner)

        def local(vectors: Tensor) -> Tensor:
            # R^T v for every pair: (R^T v)_i = sum_j R_ji v_j.
            return torch.einsum("bqji,bqj->bqi", rotation, vectors)

        gap = ((current - point) * normal).sum(-1)
        approach = -((current - start) * normal).sum(-1) / sample_radius
        radius_ratio = (lateral / sample_radius).clamp(max=RADIUS_RATIO_CAP)
        coefficients = torch.stack((contact_kappa.log1p(), contact_beta, contact_mu), dim=-1)
        token = torch.cat(
            (
                local(current - centers) / size,
                local(point - centers) / size,
                local(normal),
                gap[..., None] / sample_radius,
                approach[..., None],
                radius_ratio[..., None],
                coefficients[:, None, :].expand(batch, pairs, 3),
                torch.nn.functional.one_hot(kinds, _KIND_COUNT).to(frames.dtype),
                frames.new_zeros((batch, pairs, 1)),
            ),
            dim=-1,
        )

        # Stable sort by owner keeps pair order inside each cell; masked pairs sort last.
        key = torch.where(valid, owner, torch.full_like(owner, cells))
        sorted_key, order = torch.sort(key, dim=1, stable=True)
        position = torch.arange(pairs, device=frames.device)[None].expand(batch, pairs)
        run_start = torch.ones_like(sorted_key, dtype=torch.bool)
        run_start[:, 1:] = sorted_key[:, 1:] != sorted_key[:, :-1]
        first_of_run = torch.cummax(torch.where(run_start, position, torch.zeros_like(position)), dim=1).values
        slot = position - first_of_run
        keep = (sorted_key < cells) & (slot < tokens_per_cell)
        batch_index = torch.arange(batch, device=frames.device)[:, None].expand(batch, pairs)[keep]
        cell_index = sorted_key[keep]
        slot_index = slot[keep]
        tokens[batch_index, cell_index, slot_index] = _gather_rows(token, order)[keep]
        mask[batch_index, cell_index, slot_index] = True
    return tokens, mask
