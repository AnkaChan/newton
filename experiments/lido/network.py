# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One radius-one graph transformer block with FiLM conditioning, the edge module and the contact encoder
(design spec 5). Attention runs over flat edge lists sorted by destination with CSR offsets.

Two edge modules produce the per-edge code e_ij that the layer turns into the score bias and the edge values
(`edge_module` in TrainConfig): "a02", the edge encoder on the 24 geometry values followed by the A02 update from
the hidden states inside the layer (the trained checkpoints), and "pair" (design record, section 10), one MLP
on the pair's current configuration in the receiver's frame, `PairEdge` below.

Parameter names and shapes of the a02 network mirror the v4 network so that its checkpoints load directly
(grid buffers aside).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .csr_attention import csr_attention_warp
from .features import CONTACT_WIDTH, EDGE_WIDTH, NODE_WIDTH, SENDER_WIDTH, TOKEN_WIDTH, unpack
from .hex import MODE_COUNT
from .structs import Features, NetOutput

Tensor = torch.Tensor

USE_WARP = (
    True  # CUDA tensors go through the Warp kernels of csr_attention.py; the torch path below stays the reference
)
EDGE_MODULES = ("a02", "pair")


def zero_init(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def mlp2(n_in: int, n_hidden: int, n_out: int, zero_last: bool = False) -> nn.Sequential:
    """Two linears with a SiLU between (indices 0 and 2, as in the v4 state dict)."""
    last = nn.Linear(n_hidden, n_out)
    if zero_last:
        zero_init(last)
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.SiLU(), last)


def gather(x: Tensor, idx: Tensor) -> Tensor:
    """x[idx] along dim 0 with an index_add backward (advanced indexing backward sorts; this is 3x faster)."""
    return x.index_select(0, idx)


def csr_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edges: Tensor,
    offsets: Tensor,
    score_bias: Tensor | None = None,
    val_add: Tensor | None = None,
    edge_code: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Softmax attention over a flat edge list [2,E] = (src, dst) sorted by dst with CSR offsets [R+1].

    q, k, v [R,H,D] -> messages [R,H,D]. This is the one attention primitive of the package (cell graph and
    contact tokens). On CUDA (and with USE_WARP) it dispatches to the Warp kernels; the torch body is the reference.
    With `edge_code` [E,Dc] (one vector per edge, the same for every head) it also returns the code aggregated with
    the attention weights, [R,H,Dc]: the layer's edge values are a shared linear map of the code, applied after the
    aggregation instead of per edge.
    """
    if USE_WARP and q.is_cuda:
        return csr_attention_warp(q, k, v, edges, offsets, score_bias, val_add, edge_code)
    src, dst = edges
    D = q.shape[-1]
    score = (gather(q, dst) * gather(k, src)).sum(-1) / math.sqrt(D)  # [E,H]
    val = gather(v, src)
    if score_bias is not None:
        score = score + score_bias
    if val_add is not None:
        val = val + val_add
    lengths = offsets[1:] - offsets[:-1]
    m = torch.segment_reduce(score, "max", lengths=lengths, initial=-1e30)
    w = (score - m.repeat_interleave(lengths, 0)).exp()
    den = torch.segment_reduce(w, "sum", lengths=lengths, initial=0.0).clamp_min(1e-12)
    w = w / den.repeat_interleave(lengths, 0)
    out = torch.segment_reduce(val * w[..., None], "sum", lengths=lengths, initial=0.0)
    if edge_code is None:
        return out
    agg = torch.segment_reduce(w[:, :, None] * edge_code[:, None, :], "sum", lengths=lengths, initial=0.0)
    return out, agg


def within_range_pairs(offsets: Tensor) -> tuple[Tensor, Tensor]:
    """All (src, dst) token pairs inside each cell's contiguous range, sorted by dst, with CSR offsets [Q+1]."""
    lengths = offsets[1:] - offsets[:-1]
    Q = int(offsets[-1])
    dev = offsets.device
    if Q == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev)
    tok_cell = torch.repeat_interleave(torch.arange(lengths.numel(), device=dev), lengths)
    n_tok = lengths[tok_cell]
    pair_offsets = torch.zeros(Q + 1, dtype=torch.long, device=dev)
    pair_offsets[1:] = n_tok.cumsum(0)
    total = int(pair_offsets[-1])
    dst = torch.repeat_interleave(torch.arange(Q, device=dev), n_tok)
    src = offsets[tok_cell][dst] + torch.arange(total, device=dev) - pair_offsets[dst]
    return torch.stack([src, dst]), pair_offsets


class ContactEncoder(nn.Module):
    """tokens [Q,19] sorted by cell with token_offsets [C+1] -> [C,17]: 16 pooled channels (zero-init) + count / M."""

    def __init__(self, width: int = 64, heads: int = 2, tokens_per_cell: int = 24):
        super().__init__()
        self.heads = heads
        self.token_encoder = mlp2(TOKEN_WIDTH, width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out_projection = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = mlp2(width, 4 * width, width)
        self.pool_projection = zero_init(nn.Linear(2 * width, CONTACT_WIDTH - 1))
        self.scale = float(tokens_per_cell)

    def forward(
        self,
        tokens: Tensor,
        offsets: Tensor,
        valid: Tensor | None = None,
        cell: Tensor | None = None,
        attn: tuple | None = None,
    ) -> Tensor:
        """Compacted tokens: `offsets` is the per-cell CSR of the Q rows. Padded tokens (capacity layout, design
        spec 1b "CUDA graphs"): `valid` [Q] marks the real rows, `cell` [Q] is every row's owning cell and `attn`
        holds the precomputed within-cell (pairs, pair_offsets); padded rows get a -1e30 attention score as
        sources and are excluded from the mean / max pooling and the count, so the output equals the compacted
        one. Nothing on that path synchronises with the host (the rollout query is CUDA-graph captured)."""
        C = offsets.numel() - 1
        lengths = offsets[1:] - offsets[:-1]
        if tokens.shape[0] == 0:
            # no pairs in this batch: keep every parameter in the graph (DDP requires a gradient for each one)
            unused = sum(p.sum() for p in self.parameters()) * 0.0
            return torch.zeros(C, CONTACT_WIDTH, dtype=tokens.dtype, device=tokens.device) + unused
        t = self.token_encoder(tokens)
        Q = t.shape[0]
        if valid is None:
            pairs, pair_offsets = within_range_pairs(offsets)
            bias = None
        else:
            pairs, pair_offsets = attn if attn is not None else within_range_pairs(offsets)
            bias = torch.where(valid[pairs[0]], 0.0, -1e30).to(t.dtype)[:, None].expand(-1, self.heads)
        q, k, v = self.qkv(self.attention_norm(t)).view(Q, 3, self.heads, -1).unbind(1)
        t = t + self.out_projection(csr_attention(q, k, v, pairs, pair_offsets, bias).reshape(Q, -1))
        t = t + self.ffn(self.ffn_norm(t))
        if valid is None:
            count = lengths.to(t.dtype)
            mean = torch.segment_reduce(t, "mean", lengths=lengths, initial=0.0)
            mx = torch.segment_reduce(t, "max", lengths=lengths, initial=0.0)
        else:
            W = t.shape[1]
            vf = valid.to(t.dtype)
            # in-place scatters into fresh zeros: the out-of-place forms clone the zeros first (one copy each)
            count = torch.zeros(C, dtype=t.dtype, device=t.device).index_add_(0, cell, vf)
            mean = torch.zeros(C, W, dtype=t.dtype, device=t.device).index_add_(0, cell, t * vf[:, None])
            mean = mean / count.clamp_min(1.0)[:, None]
            mx = torch.zeros(C, W, dtype=t.dtype, device=t.device).scatter_reduce_(
                0, cell[:, None].expand(-1, W), torch.where(valid[:, None], t, float("-inf")), "amax"
            )  # include_self: max(0, max over the valid tokens), as segment_reduce with initial 0
        empty = (count == 0)[:, None]
        pooled = self.pool_projection(torch.cat([mean, mx], -1)).masked_fill_(empty, 0.0)
        return torch.cat([pooled, (count / self.scale)[:, None]], -1)


class PairEdge(nn.Module):
    """The "pair" edge module: e_ij from the pair's current configuration alone, everything in the receiver's frozen
    frame R_i. Input [E,66] = geometry g_ij (24, `edge_attr`) | receiver modes m_i (21, the first node values) |
    sender modes R_i^T m_j (21, `edge_sender`) -> Linear -> SiLU -> Linear = e_ij [E,hidden]; code width = hidden
    width. No hidden states enter, no residual, standard init (the output heads carry the zero-init). The first
    linear's receiver block W_r m_i is the same on every edge into i: applied once per cell and gathered to the
    edges; the geometry and sender blocks run per edge (two accumulating GEMMs, no [E,66] concatenation).
    Weight columns of `mlp[0]`: [0,24) geometry, [24,45) receiver, [45,66) sender."""

    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = mlp2(EDGE_WIDTH + 2 * SENDER_WIDTH, hidden, hidden)

    def forward(self, edge_attr: Tensor, m_i: Tensor, sender: Tensor, dst: Tensor) -> Tensor:
        lin0 = self.mlp[0]
        W = lin0.weight
        g, r = EDGE_WIDTH, EDGE_WIDTH + SENDER_WIDTH
        per_cell = torch.addmm(lin0.bias, m_i, W[:, g:r].t())  # [C,hidden]: the receiver block, once per cell
        h0 = torch.addmm(gather(per_cell, dst), edge_attr, W[:, :g].t())
        h0 = torch.addmm(h0, sender, W[:, r:].t())
        return self.mlp[2](self.mlp[1](h0))


class Layer(nn.Module):
    """Pre-LayerNorm transformer layer with FiLM on both branches, edge bias/values and the A02 edge update."""

    def __init__(self, width: int, heads: int, edge_hidden: int, edge_network: bool):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_bias = nn.Linear(edge_hidden, heads)
        self.edge_val = nn.Linear(edge_hidden, width)
        self.out_projection = nn.Linear(width, width)
        self.ffn = mlp2(width, 4 * width, width)
        self.film = zero_init(nn.Linear(width, 4 * width))
        self.edge_update = mlp2(2 * width + edge_hidden, width, edge_hidden, zero_last=True) if edge_network else None

    def forward(self, x: Tensor, e: Tensor, edges: Tensor, offsets: Tensor, cond: Tensor) -> Tensor:
        g1, b1, g2, b2 = self.film(cond).chunk(4, -1)  # cond [C,W] per cell
        n = self.attention_norm(x) * (1 + g1) + b1
        C = x.shape[0]
        q, k, v = self.qkv(n).view(C, 3, self.heads, -1).unbind(1)
        src, dst = edges
        if self.edge_update is not None:
            # edge_update.0 on [n_dst, n_src, e] without materialising the [E, 2W + 96] concatenation:
            # W [h, 2W+96] @ cat(...) = (n @ W_dst^T)[dst] + (n @ W_src^T)[src] + e @ W_e^T + b
            lin0 = self.edge_update[0]
            W = x.shape[1]
            per_cell = n @ torch.cat([lin0.weight[:, :W], lin0.weight[:, W : 2 * W]], 0).t()  # [C, 2h]
            h0 = gather(per_cell[:, : lin0.out_features], dst) + gather(per_cell[:, lin0.out_features :], src)
            h0 = h0 + torch.addmm(lin0.bias, e, lin0.weight[:, 2 * W :].t())
            e = e + self.edge_update[2](self.edge_update[1](h0))
        # edge values: sum_j w_ij (v_j + U e_ij) = sum_j w_ij v_j + U sum_j w_ij e_ij per head, U = edge_val (linear,
        # shared by the edges): the kernel aggregates the 96-value code with the attention weights and the per-head
        # slice of U is applied once per cell ([H,C,96] @ [H,96,D]); the bias enters once (the weights sum to 1; no
        # row is empty, every cell has its self edge). Same result, without the [E, W] edge-value tensor.
        msg, agg = csr_attention(q, k, v, edges, offsets, self.edge_bias(e), edge_code=e)  # [C,H,D], [C,H,96]
        H, D = self.heads, q.shape[-1]
        U = self.edge_val.weight.view(H, D, -1)
        msg = msg + torch.bmm(agg.transpose(0, 1), U.transpose(1, 2)).transpose(0, 1) + self.edge_val.bias.view(H, D)
        x = x + self.out_projection(msg.reshape(C, -1))
        n = self.ffn_norm(x) * (1 + g2) + b2
        return x + self.ffn(n)


class Net(nn.Module):
    def __init__(
        self,
        width: int = 192,
        heads: int = 6,
        edge_hidden: int = 96,
        contact_width: int = 64,
        edge_network: bool = True,
        max_step: float = 0.05,
        tokens_per_cell: int = 24,
        cond_width: int = 7,
        edge_module: str = "a02",
    ):
        super().__init__()
        if edge_module not in EDGE_MODULES:
            raise ValueError(f"edge_module must be one of {EDGE_MODULES}, got {edge_module!r}")
        self.max_step = max_step
        self.edge_module = edge_module
        self.contact_encoder = ContactEncoder(contact_width, 2, tokens_per_cell)
        self.node_encoder = mlp2(NODE_WIDTH + CONTACT_WIDTH, width, width)
        # "a02": geometry encoder here, the A02 update (edge_network) inside the layer; "pair": PairEdge, no update
        self.edge_encoder = mlp2(EDGE_WIDTH, edge_hidden, edge_hidden) if edge_module == "a02" else None
        self.pair_edge = PairEdge(edge_hidden) if edge_module == "pair" else None
        self.condition_encoder = mlp2(cond_width, width, width)
        self.layers = nn.ModuleList([Layer(width, heads, edge_hidden, edge_network and edge_module == "a02")])
        self.output_norm = nn.LayerNorm(width)
        self.correction_head = zero_init(nn.Linear(width, 3 * MODE_COUNT))
        self.step_head = zero_init(nn.Linear(width, 1))
        self._compiled_layers = None
        self._compiled_edge_encoder = None  # one-element list: a bare OptimizedModule would register as a submodule

    def edge_code(self, f: Features, edges: Tensor) -> Tensor:
        """e_ij [E, edge_hidden] of the configured edge module (the layer adds the A02 update for "a02")."""
        fn = (self._compiled_edge_encoder or [self.edge_encoder or self.pair_edge])[0]
        if self.edge_module == "a02":
            return fn(f.edge_attr)
        if f.edge_sender is None:
            raise ValueError('the "pair" edge module needs Features.edge_sender (features(..., sender=True))')
        return fn(f.edge_attr, f.node[:, :SENDER_WIDTH], f.edge_sender, edges[1])

    def forward(self, f: Features, edges: Tensor, edge_offsets: Tensor, cell_obj: Tensor) -> NetOutput:
        prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")  # TF32 matmuls inside the network only; physics stays full fp32
        try:
            return self._forward(f, edges, edge_offsets, cell_obj)
        finally:
            torch.set_float32_matmul_precision(prev)

    def _forward(self, f: Features, edges: Tensor, edge_offsets: Tensor, cell_obj: Tensor) -> NetOutput:
        contact = self.contact_encoder(f.tokens, f.token_offsets, f.token_valid, f.token_cell, f.token_attn)
        x = self.node_encoder(torch.cat([f.node, contact], -1))
        e = self.edge_code(f, edges)
        cond = self.condition_encoder(f.cond)[cell_obj]
        for layer in getattr(self, "_compiled_layers", None) or self.layers:
            x = layer(x, e, edges, edge_offsets, cond)
        y = self.output_norm(x)
        raw = self.correction_head(y)
        corr = raw / torch.sqrt(1.0 + (raw * raw).sum(-1, keepdim=True))  # bounded 21-vector, old layout [3,7]
        step = self.max_step * torch.sigmoid(self.step_head(y)).squeeze(-1)
        return NetOutput(corr=unpack(corr), step=step)

    def compile_layers(self, edge_encoder: bool = False, fullgraph: bool = False, **kwargs) -> Net:
        """torch.compile the cell-graph layers (fuses the per-edge elementwise chains; shapes are static per grid),
        and with `edge_encoder` the edge MLP too (its SiLU on [E, 96] fuses into the GEMM; the rollout path).
        The "pair" edge module is always compiled with the layers: it is that network's per-edge chain (the a02
        counterpart, the A02 update, lives inside the compiled layer).
        `fullgraph` demands one graph per layer: possible without autograd, where the Warp attention is a custom op.

        The compiled callables live outside the module tree so state_dict keys and checkpoints are unchanged."""
        self._compiled_layers = [
            torch.compile(layer, dynamic=False, fullgraph=fullgraph, **kwargs) for layer in self.layers
        ]
        if edge_encoder or self.edge_module == "pair":
            module = self.edge_encoder if self.edge_module == "a02" else self.pair_edge
            self._compiled_edge_encoder = [torch.compile(module, dynamic=False, **kwargs)]
        return self

    @staticmethod
    def from_config(cfg) -> Net:
        return Net(
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.edge_hidden_dim,
            cfg.contact_hidden_dim,
            cfg.edge_network,
            cfg.max_step_size,
            cfg.contact_tokens_per_cell,
            edge_module=cfg.edge_module,
        )
