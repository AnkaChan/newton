# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-tet stretch predictor.

Inputs (world frame, no per-tet rotation in this proof-of-concept):
    S(t)        6 sym components
    S(t-dt)     6
    gravity     3
    f_ext_tet   3   (mean over the tet's 4 vertices)
    mu, lam     2
    pin_flag    1
    S_neigh     6   (mean S over face-adjacent neighbors)
    n_neigh     1   (normalised to [0,1] by 4)
    --
    total      28

Output: 6 floats forming a symmetric S^* = I + tanh(delta) * scale, where
    S^*[0,0] += d0,  S^*[1,1] += d1,  S^*[2,2] += d2,
    S^*[0,1] = S^*[1,0] += d3,  ...  S^*[1,2] = S^*[2,1] += d5
The tanh bound prevents the network from emitting extreme stretches that
break the polar decomposition in the decoder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


_SYM_IDX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def sym_to_vec(S: torch.Tensor) -> torch.Tensor:
    """Flatten symmetric (..., 3, 3) -> (..., 6) using upper-triangle order."""
    out = []
    for a, b in _SYM_IDX:
        out.append(S[..., a, b])
    return torch.stack(out, dim=-1)


def vec_to_sym(v: torch.Tensor) -> torch.Tensor:
    """Inverse of sym_to_vec: (..., 6) -> (..., 3, 3) symmetric."""
    *batch, _ = v.shape
    M = torch.zeros(*batch, 3, 3, dtype=v.dtype, device=v.device)
    for k, (a, b) in enumerate(_SYM_IDX):
        M[..., a, b] = v[..., k]
        if a != b:
            M[..., b, a] = v[..., k]
    return M


def build_face_adjacency(tets: np.ndarray) -> np.ndarray:
    """For each tet, list up to 4 face-neighbour tet ids (or -1).

    Two tets are face-adjacent if they share exactly 3 vertices.
    """
    n_tets = tets.shape[0]
    # Map each face (sorted 3-tuple of vertex ids) to list of tet ids.
    face_map: dict[tuple, list[int]] = {}
    for t in range(n_tets):
        v = tets[t]
        for skip in range(4):
            face = tuple(sorted(int(v[k]) for k in range(4) if k != skip))
            face_map.setdefault(face, []).append(t)

    adj = -np.ones((n_tets, 4), dtype=np.int64)
    next_slot = np.zeros(n_tets, dtype=np.int32)
    for face, ts in face_map.items():
        if len(ts) != 2:
            continue
        a, b = ts
        adj[a, next_slot[a]] = b
        next_slot[a] += 1
        adj[b, next_slot[b]] = a
        next_slot[b] += 1
    return adj


class StretchNet(nn.Module):
    """3-layer MLP per tet. No graph layers (neighbor info injected via mean pool)."""

    def __init__(self, in_dim: int = 28, hidden: int = 64, max_delta: float = 0.6):
        super().__init__()
        self.max_delta = max_delta
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 6),
        )
        # Zero-init last layer so initial output is identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (T, in_dim) -> S* (T, 3, 3)."""
        delta = self.max_delta * torch.tanh(self.net(feat))  # (T, 6)
        S_star = vec_to_sym(delta)
        eye = torch.eye(3, dtype=feat.dtype, device=feat.device).expand_as(S_star)
        return eye + S_star


def build_features(
    S_t: torch.Tensor,            # (T, 3, 3)
    S_prev: torch.Tensor,         # (T, 3, 3)
    gravity: torch.Tensor,        # (3,)
    f_ext_vert: torch.Tensor,     # (V, 3)
    mu: torch.Tensor,             # (T,)
    lam: torch.Tensor,            # (T,)
    pin_flag: torch.Tensor,       # (T,) 0/1
    tets: torch.Tensor,           # (T, 4)
    face_adj: torch.Tensor,       # (T, 4) int64, -1 padding
) -> torch.Tensor:
    T = S_t.shape[0]
    # Per-tet mean external force.
    f_ext_tet = f_ext_vert[tets].mean(dim=1)  # (T, 3)

    # Per-tet neighbor S aggregation.
    valid = face_adj >= 0  # (T, 4)
    idx = face_adj.clamp(min=0)
    S_neigh_all = S_t[idx]  # (T, 4, 3, 3)
    S_neigh_all = S_neigh_all * valid[:, :, None, None].to(S_t.dtype)
    n_neigh = valid.sum(dim=1).to(S_t.dtype).clamp(min=1.0)  # (T,)
    S_neigh_mean = S_neigh_all.sum(dim=1) / n_neigh[:, None, None]

    # Rough scale normalisation so all input groups are O(1).
    G_SCALE = 10.0     # gravity magnitude ~9.8
    F_SCALE = 30.0     # body forces sampled up to ~50 N
    MAT_SCALE = 1e5    # Lame parameters
    # Centre S around identity so deviation is the signal.
    eye3 = torch.eye(3, dtype=S_t.dtype, device=S_t.device)
    S_t_c = S_t - eye3
    S_prev_c = S_prev - eye3
    S_n_c = S_neigh_mean - eye3

    feats = [
        sym_to_vec(S_t_c),                          # 6
        sym_to_vec(S_prev_c),                       # 6
        (gravity / G_SCALE).expand(T, -1),          # 3
        f_ext_tet / F_SCALE,                        # 3
        mu[:, None] / MAT_SCALE,                    # 1
        lam[:, None] / MAT_SCALE,                   # 1
        pin_flag[:, None],                          # 1
        sym_to_vec(S_n_c),                          # 6
        (n_neigh[:, None] / 4.0),                   # 1
    ]
    return torch.cat(feats, dim=-1)  # (T, 28)
