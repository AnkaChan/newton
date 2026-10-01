# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rotation-invariant inputs of the network (design spec 5.3): 142 node values (the contact encoder appends 17),
24 edge values on the flat CSR edge list (plus the 21 sender modes per edge for the "pair" edge module), 7
conditioning channels, contact tokens."""

from __future__ import annotations

import torch

from . import contact as _contact
from .hex import MODE_COUNT, mat3
from .structs import Features
from .units import conditioning

Tensor = torch.Tensor
RMS_FLOOR = 1e-12
CLIP = 10.0
NODE_WIDTH = 142
CONTACT_WIDTH = 17
EDGE_WIDTH = 24
SENDER_WIDTH = 3 * MODE_COUNT  # 21: one cell's modes packed, the per-edge sender block of the "pair" edge module
TOKEN_WIDTH = 19


def seg_rms(v: Tensor, seg: Tensor, count: int) -> Tensor:
    """Per-object RMS over all components of v [C,7,3] -> [O], floored."""
    sq = torch.zeros(count, dtype=v.dtype, device=v.device).index_add(0, seg, (v * v).sum((1, 2)))
    # cells per object as a masked count, not an index_add of a constant: inductor (torch 2.11, max-autotune)
    # leaves the padded lanes of that scatter unmasked and adds 16 where there are 12 cells
    n = 21.0 * (seg[None, :] == torch.arange(count, device=seg.device)[:, None]).sum(1).to(v.dtype)
    return (sq / n.clamp_min(1.0)).sqrt().clamp_min(RMS_FLOOR)


def to_frame(Rt: Tensor, v: Tensor) -> Tensor:
    """7 world vectors per cell into the frozen frame: Rt [C,3,3], v [C,7,3] -> [C,7,3]."""
    return torch.einsum("cab,cvb->cva", Rt, v)


def pack(v: Tensor) -> Tensor:
    """[C,7,3] -> 21 values in the old layout: [3,7] row-major, value index 7 i + m (component i, mode m)."""
    return v.transpose(1, 2).reshape(v.shape[0], 21)


def unpack(vals: Tensor) -> Tensor:
    """Inverse of pack: [C,21] -> [C,7,3]."""
    return vals.view(-1, 3, MODE_COUNT).transpose(1, 2)


def node_features(
    R: Tensor,
    m_c: Tensor,
    m_Y: Tensor,
    m_prev: Tensor,
    g_m: Tensor,
    hist_grad: Tensor,
    hist_update: Tensor,
    hist_valid: Tensor,
    flags: Tensor,
    seg: Tensor,
    O: int,
) -> Tensor:
    """The 142 per-cell values (per-cell elementwise chain; static shapes, so a caller may torch.compile it)."""
    Rt = R.transpose(1, 2)
    dtype = m_c.dtype
    rms = seg_rms(g_m, seg, O)
    rms_c = rms[seg][:, None, None]
    hist_ok = hist_valid.to(dtype)[seg][:, None, None]
    rms_u = seg_rms(hist_update, seg, O)[seg][:, None, None]
    return torch.cat(
        [
            pack(to_frame(Rt, m_c)),
            pack(to_frame(Rt, m_Y - m_c)),
            pack(to_frame(Rt, m_c - m_prev)),
            pack((to_frame(Rt, g_m) / rms_c).clamp(-CLIP, CLIP)),
            pack((to_frame(Rt, hist_grad) / rms_c * hist_ok).clamp(-CLIP, CLIP)),
            pack((to_frame(Rt, hist_update) / rms_u * hist_ok).clamp(-CLIP, CLIP)),
            flags,
            torch.log(rms)[seg][:, None],
            hist_valid.to(dtype)[seg][:, None],
        ],
        -1,
    )


def edge_features(
    edges: Tensor, edge_rest: Tensor, R: Tensor, F_c: Tensor, center: Tensor, m_c: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """The 24 per-edge values (per-edge elementwise chain; static shapes, so a caller may torch.compile it), and
    with the modes `m_c` [C,7,3] the sender block of the "pair" edge module: the sender's 21 modes in the receiver's
    frame, pack(R_i^T m_j) [E,21] (None without `m_c`). Same chain, so the compiled version is one kernel either way;
    the rotation is the elementwise 3x3 product of `hex.mat3`, not a cuBLAS batched matmul."""
    src, dst = edges
    Rt_dst = R.transpose(1, 2)[dst]
    edge_attr = torch.cat(
        [
            edge_rest,
            torch.einsum("eab,eb->ea", Rt_dst, center[src] - center[dst]),
            mat3(Rt_dst, R[src]).flatten(1),
            mat3(Rt_dst, F_c[src]).flatten(1),
        ],
        -1,
    )
    if m_c is None:
        return edge_attr, None
    sender = (Rt_dst[:, None, :, :] * m_c[src][:, :, None, :]).sum(-1)  # [E,7,3]: sum_b Rt[a,b] m_j[v,b]
    return edge_attr, pack(sender)


def features(
    batch,
    x: Tensor,
    R: Tensor,
    m_c: Tensor,
    F_c: Tensor,
    g_m: Tensor,
    node_fn=node_features,
    edge_fn=edge_features,
    sender: bool = False,
) -> Features:
    """x [N,3] candidate (may carry a graph; features are detached by the caller where required).
    `node_fn` / `edge_fn`: the two chains above, or compiled versions of them. `sender`: also build the per-edge
    sender block `edge_sender` (the "pair" edge module reads it; the a02 module does not)."""
    node = node_fn(
        R,
        m_c,
        batch.m_Y,
        batch.m_prev,
        g_m,
        batch.hist_grad,
        batch.hist_update,
        batch.hist_valid,
        batch.flags,
        batch.cell_obj,
        batch.O,
    )
    center = x[batch.cells].mean(1)  # [C,3]
    edge_attr, edge_sender = edge_fn(batch.edges, batch.edge_rest, R, F_c, center, m_c if sender else None)
    pairs = batch.pairs
    tokens = (
        _contact.contact_tokens(batch, x, R)
        if pairs.count > 0
        else torch.zeros(0, TOKEN_WIDTH, dtype=x.dtype, device=x.device)
    )
    padded = pairs.padded and pairs.count > 0
    return Features(
        node=node,
        edge_attr=edge_attr,
        cond=conditioning(batch.material),
        tokens=tokens,
        token_offsets=pairs.token_offsets,
        token_valid=pairs.valid if padded else None,
        token_cell=pairs.cell if padded else None,
        token_attn=(pairs.attn_pairs, pairs.attn_offsets) if padded else None,
        edge_sender=edge_sender,
    )
