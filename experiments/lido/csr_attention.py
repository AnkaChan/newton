# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSR softmax attention as Warp kernels behind a torch.autograd.Function.

Same semantics as the torch ``network.csr_attention``: edges [2,E] = (src, dst) sorted by dst with CSR offsets
[R+1]; score_e = q[dst_e] . k[src_e] / sqrt(D) + bias_e; per row and head the weights are the softmax over the
row's edges; out[i,h] = sum_e w_e (v[src_e,h] + add_e). Empty rows give zero. With an edge code [E,Dc] (one vector
per edge, shared by the heads) the kernels also return its weighted sum agg[i,h] = sum_e w_e code_e [R,H,Dc]: the
layer's edge values U e_ij are linear in e_ij, so sum_e w_e U e_e = U agg and the [E,H*D] value tensor is never
built. The score does not depend on the code; only the aggregation does.

Thread layout: LANES = D / C consecutive threads share one (row, head) and each owns a C-wide chunk of the D
axis (C = 4 when D is a multiple of 4), so the per-edge value loads of a thread group are one contiguous
D-vector. The score dot product is computed redundantly by every lane of the group (the loads broadcast). The
code axis is split into Lc = Dc / Cc chunks of Cc values (Cc = chunk_width(Dc)) that the LANES threads of a group
take in turns (lane, lane + LANES, ...), so Dc need not be a multiple of LANES.

Forward: pass 1 computes the scores, stores them in an [E,H] scratch and takes the row max; pass 2 accumulates
den = sum exp(s - m) and the weighted value sum; with a code, one more pass per owned code chunk accumulates its
weighted sum (the exp is recomputed; the score reads hit the cache). The scores, m and den are kept for the backward.
Backward, kernel 1 (per (row, head, lane)): pass 1 recovers w_e = exp(s_e - m) / den and dw_e = grad_out . val_e
(+ dagg[i,h] . code_e over the lane's code chunks), accumulates S = sum_e w_e dw_e and stores w_e, dw_e per edge;
pass 2 turns dw_e into ds_e = w_e (dw_e - S) (this is dbias), writes dadd_e = w_e grad_out and accumulates
dq = sum_e ds_e k[src_e] / sqrt(D). Kernel 2 (per (node, head, lane)) walks the edges grouped by source (a torch
sort of src per backward) and sums dk_j = sum ds_e q[dst_e] / sqrt(D) and dv_j = sum w_e grad_out[dst_e] without
atomics. Kernel 3 (per (row, code chunk)) writes dcode_e = sum_h w_e[h] dagg[i,h] for the row's edges.
D and Cc are compile-time constants: the kernels are generated per (D, Cc) (closure capture) and cached.
"""

import math

import torch
import warp as wp

Tensor = torch.Tensor

_kernels: dict[tuple[int, int], tuple] = {}
CUSTOM_OP = True  # without autograd, go through the custom op (torch.compile traces it); False: always the Function


def chunk_width(D: int) -> int:
    for C in (4, 2, 1):
        if D % C == 0:
            return C
    return 1


def kernels(D: int, Cc: int = 4) -> tuple:
    """Kernels (score, forward, weights, backward_rows, backward_sources, backward_code) and the chunk types
    (vecC, vecCc) for head width D and code chunk width Cc."""
    if (D, Cc) in _kernels:
        return _kernels[D, Cc]
    C = chunk_width(D)
    vecC = wp.types.vector(length=C, dtype=wp.float32)
    vecCc = wp.types.vector(length=Cc, dtype=wp.float32)

    @wp.kernel
    def score_kernel(
        q: wp.array2d[vecC],  # [R,H*LANES]
        k: wp.array2d[vecC],  # [R,H*LANES]
        src: wp.array[wp.int64],
        offsets: wp.array[wp.int64],
        bias: wp.array2d[wp.float32],  # [E,H] or None
        has_bias: int,
        H: int,
        LANES: int,
        scale: float,
        score: wp.array2d[wp.float32],  # [E,H], zero-filled; receives the lane partials
    ):
        tid = wp.tid()
        g = tid // LANES
        lane = tid - g * LANES
        i = g // H
        h = g - i * H
        hl = h * LANES + lane
        qi = q[i, hl]
        for e in range(int(offsets[i]), int(offsets[i + 1])):
            s = wp.dot(qi, k[int(src[e]), hl]) / scale
            if has_bias != 0 and lane == 0:
                s = s + bias[e, h]
            wp.atomic_add(score, e, h, s)

    @wp.kernel
    def forward_kernel(
        v: wp.array2d[vecC],  # [R,H*LANES]
        src: wp.array[wp.int64],
        offsets: wp.array[wp.int64],
        add: wp.array2d[vecC],  # [E,H*LANES] or None
        has_add: int,
        code: wp.array2d[vecCc],  # [E,Lc] or None
        has_code: int,
        Lc: int,
        H: int,
        LANES: int,
        score: wp.array2d[wp.float32],  # [E,H]
        out: wp.array2d[vecC],  # [R,H*LANES]
        agg: wp.array2d[vecCc],  # [R,H*Lc] or None
        row_max: wp.array2d[wp.float32],  # [R,H]
        row_den: wp.array2d[wp.float32],  # [R,H]
    ):
        tid = wp.tid()
        g = tid // LANES
        lane = tid - g * LANES
        i = g // H
        h = g - i * H
        hl = h * LANES + lane
        start = int(offsets[i])
        end = int(offsets[i + 1])
        if end <= start:
            out[i, hl] = vecC()
            if has_code != 0:
                for l in range(lane, Lc, LANES):
                    agg[i, h * Lc + l] = vecCc()
            if lane == 0:
                row_max[i, h] = 0.0
                row_den[i, h] = 1.0
            return
        m = float(-1e30)
        for e in range(start, end):
            m = wp.max(m, score[e, h])
        den = float(0.0)
        acc = vecC()
        for e in range(start, end):
            w = wp.exp(score[e, h] - m)
            den = den + w
            val = v[int(src[e]), hl]
            if has_add != 0:
                val = val + add[e, hl]
            acc = acc + w * val
        inv_den = 1.0 / den
        out[i, hl] = acc * inv_den
        if has_code != 0:
            for l in range(lane, Lc, LANES):
                acc_c = vecCc()
                for e in range(start, end):
                    acc_c = acc_c + wp.exp(score[e, h] - m) * code[e, l]
                agg[i, h * Lc + l] = acc_c * inv_den
        if lane == 0:
            row_max[i, h] = m
            row_den[i, h] = den

    @wp.kernel
    def weights_kernel(
        v: wp.array2d[vecC],  # [R,H*LANES]
        src: wp.array[wp.int64],
        offsets: wp.array[wp.int64],
        add: wp.array2d[vecC],  # [E,H*LANES] or None
        has_add: int,
        code: wp.array2d[vecCc],  # [E,Lc] or None
        has_code: int,
        Lc: int,
        H: int,
        LANES: int,
        score: wp.array2d[wp.float32],  # [E,H]
        row_max: wp.array2d[wp.float32],  # [R,H]
        row_den: wp.array2d[wp.float32],  # [R,H]
        grad_out: wp.array2d[vecC],  # [R,H*LANES]
        dagg: wp.array2d[vecCc],  # [R,H*Lc] or None
        w_buf: wp.array2d[wp.float32],  # [E,H] out: softmax weights
        dw_buf: wp.array2d[wp.float32],  # [E,H], zero-filled; receives the lane partials of grad_out . val
    ):
        tid = wp.tid()
        g = tid // LANES
        lane = tid - g * LANES
        i = g // H
        h = g - i * H
        hl = h * LANES + lane
        start = int(offsets[i])
        end = int(offsets[i + 1])
        if end <= start:
            return
        gi = grad_out[i, hl]
        m = row_max[i, h]
        inv_den = 1.0 / row_den[i, h]
        for e in range(start, end):
            val = v[int(src[e]), hl]
            if has_add != 0:
                val = val + add[e, hl]
            dw = wp.dot(gi, val)
            if has_code != 0:
                for l in range(lane, Lc, LANES):
                    dw = dw + wp.dot(dagg[i, h * Lc + l], code[e, l])
            wp.atomic_add(dw_buf, e, h, dw)
            if lane == 0:
                w_buf[e, h] = wp.exp(score[e, h] - m) * inv_den

    @wp.kernel
    def backward_rows_kernel(
        k: wp.array2d[vecC],  # [R,H*LANES]
        src: wp.array[wp.int64],
        offsets: wp.array[wp.int64],
        H: int,
        LANES: int,
        scale: float,
        grad_out: wp.array2d[vecC],  # [R,H*LANES]
        w_buf: wp.array2d[wp.float32],  # [E,H]
        ds_buf: wp.array2d[wp.float32],  # [E,H] in: dw_e; out: ds_e = w_e (dw_e - S) (= dbias)
        dq: wp.array2d[vecC],  # [R,H*LANES]
        dadd: wp.array2d[vecC],  # [E,H*LANES] or None
        want_dadd: int,
    ):
        tid = wp.tid()
        g = tid // LANES
        lane = tid - g * LANES
        i = g // H
        h = g - i * H
        hl = h * LANES + lane
        start = int(offsets[i])
        end = int(offsets[i + 1])
        if end <= start:
            dq[i, hl] = vecC()
            return
        S = float(0.0)
        for e in range(start, end):
            S = S + w_buf[e, h] * ds_buf[e, h]
        gi = grad_out[i, hl]
        dqi = vecC()
        for e in range(start, end):
            w = w_buf[e, h]
            ds = w * (ds_buf[e, h] - S)
            if lane == 0:
                ds_buf[e, h] = ds
            if want_dadd != 0:
                dadd[e, hl] = w * gi
            dqi = dqi + (ds / scale) * k[int(src[e]), hl]
        dq[i, hl] = dqi

    @wp.kernel
    def backward_sources_kernel(
        perm: wp.array[wp.int64],  # edge ids sorted by src
        perm_dst: wp.array[wp.int64],  # dst[perm]
        src_offsets: wp.array[wp.int64],  # [R+1] into perm
        q: wp.array2d[vecC],  # [R,H*LANES]
        grad_out_c: wp.array2d[vecC],  # [R,H*LANES]
        w_buf: wp.array2d[wp.float32],  # [E,H]
        ds_buf: wp.array2d[wp.float32],  # [E,H]
        H: int,
        LANES: int,
        scale: float,
        dk: wp.array2d[vecC],  # [R,H*LANES]
        dv: wp.array2d[vecC],  # [R,H*LANES]
    ):
        tid = wp.tid()
        g = tid // LANES
        lane = tid - g * LANES
        j = g // H
        h = g - j * H
        hl = h * LANES + lane
        dkj = vecC()
        dvj = vecC()
        for p in range(int(src_offsets[j]), int(src_offsets[j + 1])):
            e = int(perm[p])
            i = int(perm_dst[p])
            dkj = dkj + (ds_buf[e, h] / scale) * q[i, hl]
            dvj = dvj + w_buf[e, h] * grad_out_c[i, hl]
        dk[j, hl] = dkj
        dv[j, hl] = dvj

    @wp.kernel
    def backward_code_kernel(
        offsets: wp.array[wp.int64],
        H: int,
        Lc: int,
        w_buf: wp.array2d[wp.float32],  # [E,H]
        dagg: wp.array2d[vecCc],  # [R,H*Lc]
        dcode: wp.array2d[vecCc],  # [E,Lc]
    ):
        tid = wp.tid()
        i = tid // Lc
        l = tid - i * Lc
        for e in range(int(offsets[i]), int(offsets[i + 1])):
            acc = vecCc()
            for h in range(H):
                acc = acc + w_buf[e, h] * dagg[i, h * Lc + l]
            dcode[e, l] = acc

    _kernels[D, Cc] = (
        score_kernel,
        forward_kernel,
        weights_kernel,
        backward_rows_kernel,
        backward_sources_kernel,
        backward_code_kernel,
        vecC,
        vecCc,
    )
    return _kernels[D, Cc]


def _launch(kernel, dim: int, inputs: list, device: torch.device) -> None:
    if device.type == "cuda":
        wp.launch(kernel, dim=dim, inputs=inputs, stream=wp.stream_from_torch(torch.cuda.current_stream(device)))
    else:
        wp.launch(kernel, dim=dim, inputs=inputs, device="cpu")


def _chunks(t: Tensor | None, vec, C: int) -> wp.array | None:
    """[N,H,D] -> array2d [N,H*D/C] of C-vectors, or [N,Dc] -> [N,Dc/C] (a descriptor, no wp.array object). Rows
    may be strided (the q, k, v unbind views of one [N,3,H,D] projection): the descriptor carries the row stride,
    so no copy."""
    if t is None:
        return None
    N = t.shape[0]
    inner = math.prod(t.shape[1:])
    if t.stride(-1) != 1 or (t.dim() == 3 and t.stride(1) != t.shape[2]):  # each row must be one contiguous block
        t = t.contiguous()
    return wp.from_torch(t.view(N, inner // C, C), dtype=vec, requires_grad=False, return_ctype=True)


def _scalars(t: Tensor | None, dtype) -> wp.array | None:
    return None if t is None else wp.from_torch(t, dtype=dtype, requires_grad=False, return_ctype=True)


def _code_layout(edge_code: Tensor | None) -> tuple[int, int, int]:
    """(Dc, Cc, Lc) of the edge code; (0, 4, 0) without one (the same kernel set as a 4-aligned code)."""
    if edge_code is None:
        return 0, 4, 0
    Dc = edge_code.shape[1]
    Cc = chunk_width(Dc)
    return Dc, Cc, Dc // Cc


def attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edges: Tensor,
    offsets: Tensor,
    score_bias: Tensor | None,
    val_add: Tensor | None,
    edge_code: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The two forward launches; returns (out, agg, score, row_max, row_den). q, k, v may be row-strided views.
    agg is the [R,H,Dc] weighted code sum ([R,H,0] without a code)."""
    wp.init()
    R, H, D = q.shape
    E = edges.shape[1]
    edges, offsets = edges.contiguous(), offsets.contiguous()
    score_bias = None if score_bias is None else score_bias.contiguous()
    val_add = None if val_add is None else val_add.contiguous()
    edge_code = None if edge_code is None else edge_code.contiguous()
    Dc, Cc, Lc = _code_layout(edge_code)
    score_kernel, forward_kernel, _, _, _, _, vecC, vecCc = kernels(D, Cc)
    C = chunk_width(D)
    LANES = D // C
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)  # contiguous, as the fake below promises
    agg = torch.empty(R, H, Dc, dtype=q.dtype, device=q.device)
    score = torch.zeros(E, H, dtype=q.dtype, device=q.device)
    row_max = torch.empty(R, H, dtype=q.dtype, device=q.device)
    row_den = torch.empty(R, H, dtype=q.dtype, device=q.device)
    if R > 0:
        src, dev = _scalars(edges[0], wp.int64), q.device
        offs, sc = _scalars(offsets, wp.int64), _scalars(score, wp.float32)
        _launch(
            score_kernel,
            R * H * LANES,
            [
                _chunks(q, vecC, C),
                _chunks(k, vecC, C),
                src,
                offs,
                _scalars(score_bias, wp.float32),
                int(score_bias is not None),
                H,
                LANES,
                math.sqrt(D),
                sc,
            ],
            dev,
        )
        _launch(
            forward_kernel,
            R * H * LANES,
            [
                _chunks(v, vecC, C),
                src,
                offs,
                _chunks(val_add, vecC, C),
                int(val_add is not None),
                _chunks(edge_code, vecCc, Cc),
                int(edge_code is not None),
                Lc,
                H,
                LANES,
                sc,
                _chunks(out, vecC, C),
                _chunks(agg if edge_code is not None else None, vecCc, Cc),
                _scalars(row_max, wp.float32),
                _scalars(row_den, wp.float32),
            ],
            dev,
        )
    return out, agg, score, row_max, row_den


@torch.library.custom_op("lido::csr_attention", mutates_args=(), device_types="cuda")
def csr_attention_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edges: Tensor,
    offsets: Tensor,
    score_bias: Tensor | None,
    val_add: Tensor | None,
    edge_code: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Forward only, as an opaque op: torch.compile traces through it (no graph break) and CUDA graphs record
    its launches. The inference path of ``csr_attention_warp``. Returns (out [R,H,D], agg [R,H,Dc])."""
    out, agg, _, _, _ = attention_forward(q, k, v, edges, offsets, score_bias, val_add, edge_code)
    return out, agg


@csr_attention_op.register_fake
def _(q, k, v, edges, offsets, score_bias, val_add, edge_code):
    # contiguous whatever q's strides are (q is often an unbind view): inductor reads the outputs with these strides
    R, H, _ = q.shape
    Dc = 0 if edge_code is None else edge_code.shape[1]
    return torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty(R, H, Dc, dtype=q.dtype, device=q.device)


class CSRAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        edges: Tensor,
        offsets: Tensor,
        score_bias: Tensor | None,
        val_add: Tensor | None,
        edge_code: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        out, agg, score, row_max, row_den = attention_forward(q, k, v, edges, offsets, score_bias, val_add, edge_code)
        ctx.save_for_backward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            edges.contiguous(),
            offsets.contiguous(),
            None if val_add is None else val_add.contiguous(),
            None if edge_code is None else edge_code.contiguous(),
            score,
            row_max,
            row_den,
        )
        return out, agg

    @staticmethod
    def backward(ctx, grad_out: Tensor, dagg: Tensor | None):
        q, k, v, edges, offsets, val_add, edge_code, score, row_max, row_den = ctx.saved_tensors
        R, H, D = q.shape
        E = edges.shape[1]
        _, Cc, Lc = _code_layout(edge_code)
        _, _, weights_kernel, rows_kernel, sources_kernel, code_kernel, vecC, vecCc = kernels(D, Cc)
        C = chunk_width(D)
        LANES = D // C
        needs = ctx.needs_input_grad
        want_dadd = val_add is not None and needs[6]
        has_code = edge_code is not None and dagg is not None  # no agg gradient: the code term vanishes
        want_dcode = has_code and needs[7]
        grad_out = grad_out.contiguous()
        dagg = dagg.contiguous() if has_code else None
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        w_buf = torch.empty(E, H, dtype=q.dtype, device=q.device)
        ds_buf = torch.zeros(E, H, dtype=q.dtype, device=q.device)
        dadd = torch.empty_like(val_add) if want_dadd else None
        dcode = torch.empty_like(edge_code) if want_dcode else None
        if R > 0:
            src, dev = _scalars(edges[0], wp.int64), q.device
            offs, grad_out_c = _scalars(offsets, wp.int64), _chunks(grad_out, vecC, C)
            wb, dsb = _scalars(w_buf, wp.float32), _scalars(ds_buf, wp.float32)
            dagg_c = _chunks(dagg, vecCc, Cc)
            _launch(
                weights_kernel,
                R * H * LANES,
                [
                    _chunks(v, vecC, C),
                    src,
                    offs,
                    _chunks(val_add, vecC, C),
                    int(val_add is not None),
                    _chunks(edge_code if has_code else None, vecCc, Cc),
                    int(has_code),
                    Lc,
                    H,
                    LANES,
                    _scalars(score, wp.float32),
                    _scalars(row_max, wp.float32),
                    _scalars(row_den, wp.float32),
                    grad_out_c,
                    dagg_c,
                    wb,
                    dsb,
                ],
                dev,
            )
            if want_dcode:
                _launch(code_kernel, R * Lc, [offs, H, Lc, wb, dagg_c, _chunks(dcode, vecCc, Cc)], dev)
            _launch(
                rows_kernel,
                R * H * LANES,
                [
                    _chunks(k, vecC, C),
                    src,
                    offs,
                    H,
                    LANES,
                    math.sqrt(D),
                    grad_out_c,
                    wb,
                    dsb,
                    _chunks(dq, vecC, C),
                    _chunks(dadd, vecC, C),
                    int(want_dadd),
                ],
                dev,
            )
            if needs[1] or needs[2]:
                sorted_src, perm = torch.sort(edges[0])
                src_offsets = torch.searchsorted(sorted_src, torch.arange(R + 1, device=dev))
                _launch(
                    sources_kernel,
                    R * H * LANES,
                    [
                        _scalars(perm, wp.int64),
                        _scalars(edges[1][perm], wp.int64),
                        _scalars(src_offsets, wp.int64),
                        _chunks(q, vecC, C),
                        grad_out_c,
                        wb,
                        dsb,
                        H,
                        LANES,
                        math.sqrt(D),
                        _chunks(dk, vecC, C),
                        _chunks(dv, vecC, C),
                    ],
                    dev,
                )
        return (
            dq if needs[0] else None,
            dk if needs[1] else None,
            dv if needs[2] else None,
            None,
            None,
            ds_buf if needs[5] else None,
            dadd,
            dcode,
        )


def csr_attention_warp(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edges: Tensor,
    offsets: Tensor,
    score_bias: Tensor | None = None,
    val_add: Tensor | None = None,
    edge_code: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Warp CSR attention with the signature of ``network.csr_attention``; float32 [R,H,D] -> [R,H,D], and with
    an edge code [E,Dc] the pair (out, agg [R,H,Dc]).

    With autograd on (training) this is the autograd Function, run eagerly inside torch.compile'd code (graph
    break); without (inference) the custom op, which compiles through. Both launch on the calling torch stream.
    """
    if torch.is_grad_enabled() or not CUSTOM_OP:
        out, agg = _csr_attention_train(q, k, v, edges, offsets, score_bias, val_add, edge_code)
    else:
        out, agg = torch.ops.lido.csr_attention(q, k, v, edges, offsets, score_bias, val_add, edge_code)
    return out if edge_code is None else (out, agg)


@torch._dynamo.disable
def _csr_attention_train(q, k, v, edges, offsets, score_bias, val_add, edge_code) -> tuple[Tensor, Tensor]:
    return CSRAttention.apply(q, k, v, edges, offsets, score_bias, val_add, edge_code)
