# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical stretch predictor: level features, edge-conditioned message
passing, ancestor context, per-level Hencky residual heads, exp/log composition.

Levels
------
Level 0 is the tets themselves: no message passing, the existing 28-dim
per-tet features from :func:`model.build_features` feed head 0 directly.
Levels 1..L are the coarse clusters of a topological :class:`Hierarchy`,
each running ``mp_rounds`` rounds of edge-conditioned message passing on
its quotient graph.

Per-frame cluster kinematics (F5), volume-weighted through ``pool_mean``:

    c_A = (sum v_e c_e) / (sum v_e),   F_A = (sum v_e F_e) / (sum v_e),
    R_A = polar(F_A),

with per-tet ``F`` computed by the same ``einsum("tac,...tad->...tdc")``
convention as :func:`torch_solver.compute_S_from_x` and ``c`` the mean of
the tet's 4 current vertex positions.  Edge features (F6, receiver ``a``,
sender ``b``, ``l0_ab`` the rest-centroid distance):

    e_ab = [ R_a^T (c_b - c_a) / l0_ab,  |c_b - c_a| / l0_ab - 1,
             axial(LogSO3(R_a^T R_b)) ]                          (7 dims)

One MP round (F7), with static normalized quotient-edge weights
``wn_ab = W_ab / sum_b W_ab``:

    m_ab = MLP_edge([h_a, h_b, e_ab]);   h_a' = MLP_node([h_a, sum_b wn_ab m_ab])

Ancestor context (F8) of a node at level ``l`` is the concatenation of the
post-MP hidden states of its ancestor chain at levels ``l+1 .. L`` (pure
gathers along precomputed composed assignment maps).  Heads (F9) are
zero-initialised in the last layer:

    dH_l = delta_l * tanh(MLP_head_l([h'_l, z_l])),   delta_0 = 0.6, delta_{l>=1} = 0.3

and the composition (F11) is

    S* = sym_exp( H_t + dH_0 + sum_{l>=1} Prolong_l(dH_l) ),

where ``Prolong_l`` chains :func:`hierarchy.prolong` from level ``l`` down
to the tets and ``H_t = sym_log(spd_floor(S_t))``.  Every ``sym_log`` input
in this module goes through :func:`spd_log.spd_floor` first (GT data
contains transiently inverted tets with a negative stretch eigenvalue).

Coarse node features (fp32, exactly 31 dims, in this order)
-----------------------------------------------------------
    Hbar_t (6 sym components, sym_to_vec order), Hbar_prev (6),
    gravity/10 (3), f_ext_mean/30 (3), f_ext_sum/30 (3), mu/1e5 (1),
    lam/1e5 (1), pin_fraction (1), edge-weight-weighted mean of the
    neighbors' Hbar_t (6), log(vol_A / mean(vol at this level)) (1)

``Hbar`` fields are ``pool_mean`` of the tet-level ``H = sym_log(spd_floor(S))``;
per-tet ``f_ext`` is the mean over the tet's 4 vertices of the per-vertex
force (as in ``build_features``), pooled with ``pool_mean`` for the mean
channel and ``pool_sum`` for the sum channel; ``pin_fraction`` is the
``pool_mean`` of the per-tet pin flag.  The gravity/10, mu/1e5, lam/1e5 and
pin-flag channels are read from the corresponding ``feat28`` columns (they
are not otherwise available through the forward signature); gravity is a
global constant, the scalars are pooled with ``pool_mean``.

Layout and precision (documented decisions)
-------------------------------------------
- **Public API is batch-first, internals are node-first.**  Inputs are
  ``(V, 3)`` / ``(B, V, 3)`` etc. with an optional leading batch dim; the
  output matches.  Internally every per-node field is transposed to
  ``(nodes, B, ...)`` with a batch dim always present, because
  ``pool_mean`` / ``pool_sum`` / ``prolong`` infer the node dimension from
  ``shape[0]`` — with nodes first the inference can never be ambiguous,
  no matter how ``B`` compares to any level's node count.
- **Geometry in the input dtype, network in fp32.**  ``x_t`` / ``x_prev``
  / ``S_t`` arrive fp64 from the trainer; kinematics pooling, polar
  rotations and matrix logs run in that dtype for robustness, and the
  results are cast to fp32 where they enter the network (features, edge
  features, and the ``H_t`` composition term).  ``S*`` is returned fp32.
- Buffers are registered so ``.to(device)`` moves them; index buffers stay
  int64 (``nn.Module`` dtype casts only touch floating-point buffers).
- Each level and each MP round has its own edge/node MLPs (no weight
  sharing); F7 is implemented literally, with no residual shortcut around
  ``MLP_node``.
- Padded adjacency entries get an identity relative rotation and are
  aggregated with weight 0; a level with no quotient edges (a single
  cluster) yields a zero message aggregate and still works.
- No ``detach`` anywhere: gradients flow from ``S*`` back into ``x_t`` /
  ``x_prev`` / ``S_t`` during K-step rollouts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .hierarchy import Hierarchy, pool_mean, pool_sum, prolong
from .model import sym_to_vec, vec_to_sym
from .polar import polar_rotation
from .spd_log import so3_log_axial, spd_floor, sym_exp, sym_log
from .torch_solver import SolverState, compute_S_from_x

COARSE_FEAT_DIM = 31
EDGE_FEAT_DIM = 7

# Column layout of the 31-dim coarse node features (see module docstring).
F_EXT_MEAN_SLICE = slice(15, 18)
F_EXT_SUM_SLICE = slice(18, 21)

# Column layout of feat28 (see model.build_features).
_FEAT28_GRAVITY = slice(12, 15)
_FEAT28_SCALARS = slice(18, 21)  # mu/1e5, lam/1e5, pin_flag

_F_SCALE = 30.0  # same body-force normalisation as build_features


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


class HierStretchNet(nn.Module):
    """Hierarchical stretch predictor (see module docstring for the math)."""

    def __init__(
        self,
        hierarchy: Hierarchy,
        in_dim: int = 28,
        hidden: int = 64,
        mp_rounds: int = 2,
        delta_fine: float = 0.6,
        delta_coarse: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.mp_rounds = mp_rounds
        self.n_levels = len(hierarchy.levels)  # L
        self._deltas = [delta_fine] + [delta_coarse] * self.n_levels

        # --- static hierarchy buffers (int64 indices, fp32 weights) ---
        child_vol = hierarchy.tet_vol
        for level, lev in enumerate(hierarchy.levels, start=1):
            valid = lev.adj >= 0
            w = lev.w_adj * valid
            row_sum = w.sum(axis=1, keepdims=True)
            wn = w / np.where(row_sum > 0.0, row_sum, 1.0)
            nbr_c0 = lev.c0[np.clip(lev.adj, 0, None)]
            l0 = np.linalg.norm(nbr_c0 - lev.c0[:, None, :], axis=-1)
            l0 = np.where(valid, np.maximum(l0, 1e-12), 1.0)
            log_vol = np.log(lev.vol / lev.vol.mean())

            buf = self.register_buffer
            buf(f"assign_{level}", torch.as_tensor(lev.assign, dtype=torch.int64))
            buf(f"child_vol_{level}", torch.as_tensor(child_vol, dtype=torch.float32))
            buf(f"adj_{level}", torch.as_tensor(lev.adj, dtype=torch.int64))
            buf(f"wn_{level}", torch.as_tensor(wn, dtype=torch.float32))
            buf(f"l0_{level}", torch.as_tensor(l0, dtype=torch.float32))
            buf(f"log_vol_{level}", torch.as_tensor(log_vol, dtype=torch.float32))
            buf(f"pou_idx_{level}", torch.as_tensor(lev.pou_idx, dtype=torch.int64))
            buf(f"pou_w_{level}", torch.as_tensor(lev.pou_w, dtype=torch.float32))
            child_vol = lev.vol

        # Composed ancestor maps: ancestor_{l}_{m}[i] = ancestor at level m of level-l node i.
        assigns = [lev.assign for lev in hierarchy.levels]
        for src in range(self.n_levels):
            chain = None
            for dst in range(src + 1, self.n_levels + 1):
                chain = assigns[dst - 1] if chain is None else assigns[dst - 1][chain]
                self.register_buffer(f"ancestor_{src}_{dst}", torch.as_tensor(chain, dtype=torch.int64))

        # --- modules: per-level encoder + per-level per-round MP MLPs ---
        self.encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(COARSE_FEAT_DIM, hidden), nn.SiLU()) for _ in range(self.n_levels)]
        )
        self.edge_mlps = nn.ModuleList(
            [
                nn.ModuleList([_mlp(2 * hidden + EDGE_FEAT_DIM, hidden, hidden) for _ in range(mp_rounds)])
                for _ in range(self.n_levels)
            ]
        )
        self.node_mlps = nn.ModuleList(
            [nn.ModuleList([_mlp(2 * hidden, hidden, hidden) for _ in range(mp_rounds)]) for _ in range(self.n_levels)]
        )

        # Heads (F9): head 0 sees [feat28, z_0]; head l >= 1 sees [h'_l, z_l].
        # z_l concatenates the hidden states of the L - l ancestors above.
        head_in = [in_dim + self.n_levels * hidden]
        head_in += [hidden + (self.n_levels - level) * hidden for level in range(1, self.n_levels + 1)]
        self.heads = nn.ModuleList([_mlp(d, hidden, 6) for d in head_in])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _buf(self, name: str, level: int) -> torch.Tensor:
        return getattr(self, f"{name}_{level}")

    def _prepare(self, state: SolverState, x_t, x_prev, f_ext, feat28, S_t) -> dict:
        """Batch canonicalisation, tet-level fields, pooled per-level fields.

        Returns node-first tensors (``(nodes, B, ...)``); see module docstring.
        """
        if feat28.shape[-1] != self.in_dim:
            raise ValueError(f"feat28 has {feat28.shape[-1]} channels, expected {self.in_dim}")
        batched = x_t.dim() == 3
        if not batched:
            x_t, x_prev, f_ext = x_t[None], x_prev[None], f_ext[None]
            feat28, S_t = feat28[None], S_t[None]

        # --- tet-level geometry in the input dtype (fp64 from the trainer) ---
        x_tet = x_t[:, state.tets]  # (B, T, 4, 3)
        F = torch.einsum("tac,btad->btdc", state.J, x_tet)  # (B, T, 3, 3)
        c = x_tet.mean(dim=2)  # (B, T, 3)
        S_prev = compute_S_from_x(state, x_prev)  # (B, T, 3, 3)
        H_t = sym_log(spd_floor(S_t))
        H_prev = sym_log(spd_floor(S_prev))
        f_tet = f_ext[:, state.tets].mean(dim=2)  # (B, T, 3)

        # --- node-first tet fields ---
        feat28_nf = feat28.transpose(0, 1).to(torch.float32)  # (T, B, 28)
        fields = {
            "H": sym_to_vec(H_t).transpose(0, 1),  # (T, B, 6)
            "H_prev": sym_to_vec(H_prev).transpose(0, 1),
            "F": F.transpose(0, 1),  # (T, B, 3, 3)
            "c": c.transpose(0, 1),  # (T, B, 3)
            "f_mean": f_tet.transpose(0, 1),  # (T, B, 3)
            "f_sum": f_tet.transpose(0, 1),
            "scalars": feat28_nf[..., _FEAT28_SCALARS],  # (T, B, 3) mu, lam, pin
        }
        gravity = feat28_nf[0, :, _FEAT28_GRAVITY]  # (B, 3): global constant, already /10

        # --- pool every field bottom-up; assemble the 31-dim features ---
        n_batch = x_t.shape[0]
        level_feats: list[torch.Tensor] = []
        level_kin: list[tuple[torch.Tensor, torch.Tensor]] = []
        for level in range(1, self.n_levels + 1):
            assign = self._buf("assign", level)
            child_vol = self._buf("child_vol", level)
            pooled = {}
            for key, value in fields.items():
                if key == "f_sum":
                    pooled[key] = pool_sum(value, assign)
                else:
                    pooled[key] = pool_mean(value, assign, child_vol)
            fields = pooled

            adj = self._buf("adj", level)
            wn = self._buf("wn", level)
            n_nodes = adj.shape[0]
            H_bar32 = fields["H"].to(torch.float32)
            neighbor_h = (H_bar32[adj.clamp(min=0)] * wn[:, :, None, None]).sum(dim=1)  # (N, B, 6)
            feat = torch.cat(
                [
                    H_bar32,
                    fields["H_prev"].to(torch.float32),
                    gravity[None].expand(n_nodes, n_batch, 3),
                    (fields["f_mean"] / _F_SCALE).to(torch.float32),
                    (fields["f_sum"] / _F_SCALE).to(torch.float32),
                    fields["scalars"],
                    neighbor_h,
                    self._buf("log_vol", level)[:, None, None].expand(n_nodes, n_batch, 1),
                ],
                dim=-1,
            )
            level_feats.append(feat)  # (N_l, B, 31) fp32
            level_kin.append((fields["c"], polar_rotation(fields["F"])))  # input dtype

        return {
            "batched": batched,
            "feat28_nf": feat28_nf,
            "H_t32": H_t.to(torch.float32).transpose(0, 1),  # (T, B, 3, 3)
            "level_feats": level_feats,
            "level_kin": level_kin,
        }

    def _edge_features(self, level: int, c: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """F6 edge features on the level's padded quotient adjacency, fp32.

        Args:
            level: coarse level index (1-based).
            c: cluster centroids, ``(N, B, 3)``, input dtype.
            R: cluster rotations, ``(N, B, 3, 3)``, input dtype.

        Returns:
            ``(N, K, B, 7)`` fp32; padded slots hold finite values and are
            aggregated with weight zero downstream.
        """
        adj = self._buf("adj", level)
        valid = adj >= 0
        idx = adj.clamp(min=0)
        d = c[idx] - c[:, None]  # (N, K, B, 3)
        l0 = self._buf("l0", level)[:, :, None, None]  # (N, K, 1, 1)
        rel = torch.einsum("nbji,nkbj->nkbi", R, d) / l0  # R_a^T (c_b - c_a) / l0
        stretch = d.norm(dim=-1, keepdim=True) / l0 - 1.0  # (N, K, B, 1)
        R_rel = torch.einsum("nbji,nkbjl->nkbil", R, R[idx])  # R_a^T R_b
        eye = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R_rel)
        R_rel = torch.where(valid[:, :, None, None, None], R_rel, eye)
        return torch.cat([rel, stretch, so3_log_axial(R_rel)], dim=-1).to(torch.float32)

    def level_features(self, state: SolverState, x_t, x_prev, f_ext, feat28, S_t) -> list[torch.Tensor]:
        """The 31-dim coarse node features per level (public, for tests/diagnostics).

        Returns:
            List over levels 1..L of ``(N_l, 31)`` fp32 tensors — or
            ``(B, N_l, 31)`` when the inputs carry a leading batch dim.
        """
        prep = self._prepare(state, x_t, x_prev, f_ext, feat28, S_t)
        feats = [f.transpose(0, 1) for f in prep["level_feats"]]
        return feats if prep["batched"] else [f[0] for f in feats]

    def forward(self, state: SolverState, x_t, x_prev, f_ext, feat28, S_t) -> torch.Tensor:
        """Predict the target stretch field ``S*``.

        Args:
            state: decoder solver state (``tets`` and ``J`` are used).
            x_t: current positions ``(V, 3)`` or ``(B, V, 3)``.
            x_prev: previous positions, same shape as ``x_t``.
            f_ext: per-vertex external forces [N], same shape as ``x_t``.
            feat28: tet-level features from ``build_features``, ``(..., T, 28)``.
            S_t: current stretches from ``compute_S_from_x``, ``(..., T, 3, 3)``.

        Returns:
            ``S*`` of shape ``(..., T, 3, 3)``, fp32, SPD.
        """
        prep = self._prepare(state, x_t, x_prev, f_ext, feat28, S_t)

        # A4: coarse-to-fine — MP + head per level, ancestor context from above.
        h_levels: list[torch.Tensor | None] = [None] * (self.n_levels + 1)
        dh_levels: list[torch.Tensor | None] = [None] * (self.n_levels + 1)
        for level in range(self.n_levels, 0, -1):
            c, R = prep["level_kin"][level - 1]
            edge = self._edge_features(level, c, R)  # (N, K, B, 7)
            adj = self._buf("adj", level)
            idx = adj.clamp(min=0)
            wn = self._buf("wn", level)[:, :, None, None]

            h = self.encoders[level - 1](prep["level_feats"][level - 1])  # (N, B, hidden)
            for edge_mlp, node_mlp in zip(self.edge_mlps[level - 1], self.node_mlps[level - 1], strict=True):
                h_recv = h[:, None].expand(-1, adj.shape[1], -1, -1)  # (N, K, B, hidden)
                m = edge_mlp(torch.cat([h_recv, h[idx], edge], dim=-1))
                h = node_mlp(torch.cat([h, (wn * m).sum(dim=1)], dim=-1))
            h_levels[level] = h

            context = [h_levels[m][self._buf(f"ancestor_{level}", m)] for m in range(level + 1, self.n_levels + 1)]
            head_in = torch.cat([h, *context], dim=-1)
            dh_levels[level] = self._deltas[level] * torch.tanh(self.heads[level](head_in))

        context = [h_levels[m][self._buf("ancestor_0", m)] for m in range(1, self.n_levels + 1)]
        head_in = torch.cat([prep["feat28_nf"], *context], dim=-1)
        dh_levels[0] = self._deltas[0] * torch.tanh(self.heads[0](head_in))  # (T, B, 6)

        # F11: prolong every coarse residual down to the tets, compose in log space.
        acc = dh_levels[0]
        for level in range(1, self.n_levels + 1):
            y = dh_levels[level]
            for down in range(level, 0, -1):
                y = prolong(y, self._buf("pou_idx", down), self._buf("pou_w", down))
            acc = acc + y
        S_star = sym_exp(prep["H_t32"] + vec_to_sym(acc))  # (T, B, 3, 3) fp32
        S_star = S_star.transpose(0, 1)
        return S_star if prep["batched"] else S_star[0]
