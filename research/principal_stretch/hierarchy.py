# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Topological tet-graph hierarchy: coarsening, quotient graphs, operators.

Builds a multi-level partition of a tet mesh purely from **mesh topology**
(face adjacency), never from Euclidean proximity — so geometrically close
but topologically distant material (the two arms of a U-bar) is never
merged into one cluster.

Construction (offline, deterministic NumPy)
-------------------------------------------
Level 0 is the tet graph: two tets are adjacent iff they share a face, and
the edge strength is the rest-space area of that shared face (F1),

    w_ab = area(shared face of tets a, b) = 0.5 |cross(v1 - v0, v2 - v0)|.

Each level is coarsened by greedy topological aggregation (A1): repeatedly
seed at the unassigned node with the fewest unassigned neighbors (tie:
lowest index) and grow the cluster one node at a time along graph edges,
always taking the frontier node with the largest total edge weight into the
cluster (tie: lowest index), until ``target`` members are reached or the
frontier is exhausted.  A post-pass merges every cluster smaller than
``target / 2`` into the adjacent cluster with the largest total crossing
weight (tie: lowest cluster id; clusters whose whole graph component is
exhausted have no adjacent cluster and are kept as-is).  Clusters are
connected by construction — growth and merging only ever follow edges.

The quotient graph (A2) accumulates child edge weights across cluster
boundaries (F2),

    W_AB = sum of w_ab over all child edges (a, b) with a in A, b in B,

and stores padded neighbor / weight arrays sorted by weight descending
(tie: lowest neighbor id).  Cluster rest volumes are member sums; cluster
rest centroids are volume-weighted member-centroid means.

Prolongation weights (F10) are a static partition of unity in rest space:
a child node e with parent A0 blends over C(e) = {A0} union
quotient-neighbors(A0) with normalized Gaussian weights

    omega_eA = exp(-|c_e^0 - c_A^0|^2 / sigma_l^2) / (sum over C(e)),

where ``sigma_l`` is the mean rest-centroid distance over the level's
quotient edges (1.0 when the level has no edges, in which case the PoU
degenerates to piecewise-constant injection).

Operators (torch, differentiable)
---------------------------------
``pool_mean`` (F3, volume-weighted intensive average), ``pool_sum`` (F4,
extensive sum) and ``prolong`` (F10 applied one level down) are plain
``index_add`` / gather arithmetic, differentiable with respect to their
value inputs, and accept arbitrary trailing shapes with an optional leading
batch dimension: values of shape ``(N, ...)`` or ``(B, N, ...)``.  When the
leading two sizes are ambiguous (``B == N``) the input is treated as
unbatched.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class Level:
    """One coarsening level: the map from child nodes into this level's clusters."""

    assign: np.ndarray
    """(N_child,) int64 — child-node -> this level's cluster id."""
    adj: np.ndarray
    """(N, K) int64, -1 padded — this level's quotient adjacency."""
    w_adj: np.ndarray
    """(N, K) float64 — matching edge weights (F2), sorted descending per row."""
    vol: np.ndarray
    """(N,) float64 — cluster rest volumes."""
    c0: np.ndarray
    """(N, 3) float64 — cluster rest centroids (volume-weighted)."""
    pou_idx: np.ndarray
    """(N_child, P) int64 — PoU candidate cluster ids, -1 padded; parent first."""
    pou_w: np.ndarray
    """(N_child, P) float64 — PoU weights (F10); rows sum to 1, 0 on padding."""


@dataclasses.dataclass
class Hierarchy:
    """Topological tet-graph hierarchy: level-0 tet graph plus coarsening levels."""

    levels: list[Level]
    """levels[0] maps tets -> level-1 clusters, levels[1] maps those -> level-2, ..."""
    tet_adj: np.ndarray
    """(T, 4) int64 — level-0 face adjacency, -1 padded."""
    tet_w_adj: np.ndarray
    """(T, 4) float64 — shared-face rest areas (F1)."""
    tet_vol: np.ndarray
    """(T,) float64 — tet rest volumes, |det(Ds)| / 6."""
    tet_c0: np.ndarray
    """(T, 3) float64 — tet rest centroids (mean of the 4 rest vertices)."""


def _build_tet_adjacency(tets: np.ndarray, rest_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Face adjacency (like model.build_face_adjacency) plus shared-face areas."""
    n_tets = tets.shape[0]
    face_map: dict[tuple[int, int, int], list[int]] = {}
    for t in range(n_tets):
        v = tets[t]
        for skip in range(4):
            face = tuple(sorted(int(v[k]) for k in range(4) if k != skip))
            face_map.setdefault(face, []).append(t)

    adj = np.full((n_tets, 4), -1, dtype=np.int64)
    w_adj = np.zeros((n_tets, 4), dtype=np.float64)
    next_slot = np.zeros(n_tets, dtype=np.int32)
    for face, ts in face_map.items():
        if len(ts) != 2:
            continue
        a, b = ts
        p0, p1, p2 = rest_q[face[0]], rest_q[face[1]], rest_q[face[2]]
        area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        adj[a, next_slot[a]] = b
        w_adj[a, next_slot[a]] = area
        next_slot[a] += 1
        adj[b, next_slot[b]] = a
        w_adj[b, next_slot[b]] = area
        next_slot[b] += 1
    return adj, w_adj


def _aggregate(adj: np.ndarray, w_adj: np.ndarray, target: int) -> np.ndarray:
    """Greedy topological aggregation (A1). Returns (N,) int64 cluster ids.

    Deterministic: seed = unassigned node with the fewest unassigned
    neighbors (tie: lowest index); growth = frontier node with the largest
    total edge weight into the cluster (tie: lowest index); post-pass
    processes small clusters in ascending id order, re-checking sizes so a
    cluster that already grew past ``target / 2`` by absorbing an earlier
    merge is left alone, and merges into the adjacent cluster with the
    largest total crossing weight (tie: lowest cluster id).
    """
    n_nodes, n_slots = adj.shape
    valid = adj >= 0
    neighbor = np.clip(adj, 0, None)
    assign = np.full(n_nodes, -1, dtype=np.int64)
    unassigned = np.ones(n_nodes, dtype=bool)
    n_clusters = 0

    while unassigned.any():
        counts = (valid & unassigned[neighbor]).sum(axis=1)
        counts = np.where(unassigned, counts, n_nodes + n_slots + 1)
        seed = int(np.argmin(counts))  # argmin takes the lowest index on ties
        members = [seed]
        assign[seed] = n_clusters
        unassigned[seed] = False
        while len(members) < target:
            weight_into: dict[int, float] = {}
            for m in members:
                for slot in range(n_slots):
                    b = int(adj[m, slot])
                    if b >= 0 and unassigned[b]:
                        weight_into[b] = weight_into.get(b, 0.0) + float(w_adj[m, slot])
            if not weight_into:
                break
            best = min(weight_into.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            members.append(best)
            assign[best] = n_clusters
            unassigned[best] = False
        n_clusters += 1

    # Post-pass: merge undersized clusters into their strongest neighbor.
    sizes = np.bincount(assign, minlength=n_clusters)
    for cluster in range(n_clusters):
        if sizes[cluster] == 0 or sizes[cluster] >= target / 2:
            continue
        members = np.nonzero(assign == cluster)[0]
        crossing: dict[int, float] = {}
        for a in members:
            for slot in range(n_slots):
                b = int(adj[a, slot])
                if b >= 0 and assign[b] != cluster:
                    other = int(assign[b])
                    crossing[other] = crossing.get(other, 0.0) + float(w_adj[a, slot])
        if not crossing:
            continue  # whole graph component exhausted — keep the small cluster
        absorber = min(crossing.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        assign[members] = absorber
        sizes[absorber] += sizes[cluster]
        sizes[cluster] = 0

    # Compact relabel, preserving emission order of the surviving clusters.
    surviving = np.unique(assign)
    remap = np.zeros(int(surviving.max()) + 1, dtype=np.int64)
    remap[surviving] = np.arange(surviving.shape[0], dtype=np.int64)
    return remap[assign]


def _quotient_graph(assign: np.ndarray, adj: np.ndarray, w_adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quotient graph (A2): accumulate crossing weights (F2), pad, sort.

    Rows are sorted by weight descending (tie: lowest neighbor id).  The pad
    width is ``max(1, max degree)`` so downstream code never sees zero-width
    arrays.
    """
    n_clusters = int(assign.max()) + 1
    rows, slots = np.nonzero(adj >= 0)
    a_cluster = assign[rows]
    b_cluster = assign[adj[rows, slots]]
    weights = w_adj[rows, slots]
    crossing = a_cluster != b_cluster
    a_cluster, b_cluster, weights = a_cluster[crossing], b_cluster[crossing], weights[crossing]

    # Each undirected child edge appears in both directed adjacency slots, so
    # accumulating directed entries into W[A, B] counts it exactly once per
    # direction — symmetric by construction.
    key = a_cluster * n_clusters + b_cluster
    unique_key, inverse = np.unique(key, return_inverse=True)
    weight_sum = np.bincount(inverse, weights=weights)
    unique_a = unique_key // n_clusters
    unique_b = unique_key % n_clusters

    degree = np.bincount(unique_a, minlength=n_clusters)
    width = max(1, int(degree.max())) if degree.size else 1
    q_adj = np.full((n_clusters, width), -1, dtype=np.int64)
    q_w = np.zeros((n_clusters, width), dtype=np.float64)
    order = np.lexsort((unique_b, -weight_sum, unique_a))
    next_slot = np.zeros(n_clusters, dtype=np.int64)
    for e in order:
        a = int(unique_a[e])
        q_adj[a, next_slot[a]] = unique_b[e]
        q_w[a, next_slot[a]] = weight_sum[e]
        next_slot[a] += 1
    return q_adj, q_w


def _build_pou(
    assign: np.ndarray, child_c0: np.ndarray, q_adj: np.ndarray, q_c0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Partition-of-unity prolongation rows (F10), rest-space, precomputed."""
    rows, slots = np.nonzero(q_adj >= 0)
    neighbors = q_adj[rows, slots]
    once = rows < neighbors  # each undirected quotient edge once
    if once.any():
        sigma = float(np.linalg.norm(q_c0[rows[once]] - q_c0[neighbors[once]], axis=1).mean())
    else:
        sigma = 1.0
    if sigma <= 0.0:
        sigma = 1.0

    candidates = np.concatenate([assign[:, None], q_adj[assign]], axis=1)  # (N_child, P)
    valid = candidates >= 0
    diff = child_c0[:, None, :] - q_c0[np.clip(candidates, 0, None)]
    omega = np.exp(-(diff**2).sum(axis=-1) / sigma**2) * valid
    omega = omega / omega.sum(axis=1, keepdims=True)  # parent always valid -> sum > 0
    pou_idx = np.where(valid, candidates, -1).astype(np.int64)
    return pou_idx, omega


def build_hierarchy(tets: np.ndarray, rest_q: np.ndarray, n_levels: int = 3, target: int = 8) -> Hierarchy:
    """Build an ``n_levels``-deep topological hierarchy over a tet mesh.

    Args:
        tets: (T, 4) tet vertex indices.
        rest_q: (V, 3) rest vertex positions [m].
        n_levels: number of coarsening levels to build.
        target: cluster size the aggregation aims for at every level.
    """
    tets = np.asarray(tets, dtype=np.int64)
    rest_q = np.asarray(rest_q, dtype=np.float64)
    tet_adj, tet_w_adj = _build_tet_adjacency(tets, rest_q)
    corners = rest_q[tets]  # (T, 4, 3)
    edge_matrix = np.stack(
        [
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 0],
            corners[:, 3] - corners[:, 0],
        ],
        axis=-1,
    )
    tet_vol = np.abs(np.linalg.det(edge_matrix)) / 6.0
    tet_c0 = corners.mean(axis=1)

    levels: list[Level] = []
    adj, w_adj, vol, c0 = tet_adj, tet_w_adj, tet_vol, tet_c0
    for _ in range(n_levels):
        assign = _aggregate(adj, w_adj, target)
        q_adj, q_w = _quotient_graph(assign, adj, w_adj)
        n_clusters = q_adj.shape[0]
        # One root per connected component is the natural stopping point.
        # Once aggregation cannot reduce the graph, more levels would only
        # repeat an identity assignment and add no receptive field.
        if n_clusters == adj.shape[0]:
            break
        q_vol = np.bincount(assign, weights=vol, minlength=n_clusters)
        q_c0 = (
            np.stack(
                [np.bincount(assign, weights=vol * c0[:, d], minlength=n_clusters) for d in range(3)],
                axis=1,
            )
            / q_vol[:, None]
        )
        pou_idx, pou_w = _build_pou(assign, c0, q_adj, q_c0)
        levels.append(Level(assign=assign, adj=q_adj, w_adj=q_w, vol=q_vol, c0=q_c0, pou_idx=pou_idx, pou_w=pou_w))
        adj, w_adj, vol, c0 = q_adj, q_w, q_vol, q_c0
    return Hierarchy(levels=levels, tet_adj=tet_adj, tet_w_adj=tet_w_adj, tet_vol=tet_vol, tet_c0=tet_c0)


def _node_dim(x: torch.Tensor, n_nodes: int) -> int:
    """0 for (N, ...) values, 1 for (B, N, ...); ambiguity resolves to 0."""
    if x.shape[0] == n_nodes:
        return 0
    if x.dim() >= 2 and x.shape[1] == n_nodes:
        return 1
    raise ValueError(f"values of shape {tuple(x.shape)} do not match {n_nodes} nodes at dim 0 or 1")


def _segment_sum(x: torch.Tensor, assign: torch.Tensor, n_out: int, node_dim: int) -> torch.Tensor:
    out_shape = list(x.shape)
    out_shape[node_dim] = n_out
    return x.new_zeros(out_shape).index_add(node_dim, assign, x)


def pool_mean(x: torch.Tensor, assign: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    """Volume-weighted intensive pooling (F3): x_A = sum(v_e x_e) / sum(v_e).

    Args:
        x: child values, shape (N, ...) or (B, N, ...); any trailing shape.
        assign: (N,) child -> cluster ids.
        vol: (N,) child rest volumes (the weights).
    """
    assign = assign.long()
    n_nodes = assign.shape[0]
    node_dim = _node_dim(x, n_nodes)
    n_out = int(assign.max().item()) + 1
    v = vol.to(dtype=x.dtype, device=x.device)
    v_shape = [1] * x.dim()
    v_shape[node_dim] = n_nodes
    numerator = _segment_sum(x * v.reshape(v_shape), assign, n_out, node_dim)
    denominator = v.new_zeros(n_out).index_add(0, assign, v)
    d_shape = [1] * x.dim()
    d_shape[node_dim] = n_out
    return numerator / denominator.reshape(d_shape)


def pool_sum(x: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    """Extensive pooling (F4): xsum_A = sum of x_e over the members of A.

    Args:
        x: child values, shape (N, ...) or (B, N, ...); any trailing shape.
        assign: (N,) child -> cluster ids.
    """
    assign = assign.long()
    n_nodes = assign.shape[0]
    node_dim = _node_dim(x, n_nodes)
    n_out = int(assign.max().item()) + 1
    return _segment_sum(x, assign, n_out, node_dim)


def prolong(y: torch.Tensor, pou_idx: torch.Tensor, pou_w: torch.Tensor) -> torch.Tensor:
    """PoU prolongation (F10), one level down: out_e = sum_A omega_eA y_A.

    Args:
        y: cluster values, shape (M, ...) or (B, M, ...); any trailing shape.
        pou_idx: (N_child, P) candidate cluster ids, -1 padded.
        pou_w: (N_child, P) PoU weights; rows sum to 1, 0 on padding.
    """
    pou_idx = pou_idx.long()
    n_out = int(pou_idx.max().item()) + 1  # every cluster is some child's parent
    node_dim = _node_dim(y, n_out)
    index = pou_idx.clamp(min=0)
    weights = (pou_w * (pou_idx >= 0)).to(dtype=y.dtype, device=y.device)
    if node_dim == 0:
        gathered = y[index]  # (N_child, P, ...)
        w_shape = (*index.shape, *([1] * (gathered.dim() - 2)))
        return (gathered * weights.reshape(w_shape)).sum(dim=1)
    gathered = y[:, index]  # (B, N_child, P, ...)
    w_shape = (1, *index.shape, *([1] * (gathered.dim() - 3)))
    return (gathered * weights.reshape(w_shape)).sum(dim=2)
