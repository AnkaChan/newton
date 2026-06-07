# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side precompute for the multi-resolution (cluster-affine) coarse step in VBD.

Groups triangles into clusters on the triangle dual graph (farthest-point-sampling seeds +
geodesic multi-source Dijkstra flood-fill), then builds the data the coarse cluster solve needs:
each cluster carries a 12-DOF affine increment ``dq=(dA, dt)`` prolonged as ``dx_i = dA r_i + dt``.

The coarse Hessian is the Galerkin reduced Hessian assembled from per-element-corner pairs whose
two vertices are both members of a cluster (interior element -> all 9 pairs = the full F-pullback;
seam element -> the pairs within the neighbour cluster). This module produces, per cluster (CSR):
the free-vertex members + their rest offsets, the element-corner-pair entries, and a cluster
colouring (clusters adjacent iff they share a free vertex) so each colour is vertex-disjoint and the
coarse sweep needs no atomics.

Stdlib/NumPy only (no SciPy), to stay within Newton's dependency budget. All outputs are NumPy
arrays; the solver uploads them to the device. Validated against the NumPy reference prototype.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# triangle dual graph + geodesic clustering
# --------------------------------------------------------------------------- #
def build_dual_graph(verts: np.ndarray, faces: np.ndarray):
    """Triangles adjacent iff they share an edge; edge weight = centroid distance.

    Returns (adjacency list of (tri, weight), triangle centroids).
    """
    cent = verts[faces].mean(axis=1)
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for t, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_tris.setdefault((u, v) if u < v else (v, u), []).append(t)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(len(faces))]
    for tris in edge_to_tris.values():
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                t1, t2 = tris[i], tris[j]
                w = float(np.linalg.norm(cent[t1] - cent[t2]))
                adj[t1].append((t2, w))
                adj[t2].append((t1, w))
    return adj, cent


def multi_source_dijkstra(n: int, adj, sources):
    """Geodesic distance + nearest-source label (index into ``sources``)."""
    dist = np.full(n, np.inf)
    label = np.full(n, -1, dtype=np.int64)
    pq: list[tuple[float, int, int]] = []
    for si, s in enumerate(sources):
        if dist[s] > 0.0:
            dist[s] = 0.0
            label[s] = si
            heapq.heappush(pq, (0.0, int(s), si))
    while pq:
        d, u, lu = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            ndist = d + w
            if ndist < dist[v]:
                dist[v] = ndist
                label[v] = lu
                heapq.heappush(pq, (ndist, v, lu))
    return dist, label


def fps_seeds(n: int, adj, k: int, start: int = 0):
    """Farthest-point sampling on the geodesic metric (covers disconnected components)."""
    seeds = [int(start)]
    dist, _ = multi_source_dijkstra(n, adj, seeds)
    while len(seeds) < k:
        finite = np.where(np.isfinite(dist), dist, -np.inf)
        nxt = int(np.argmax(finite))
        if not np.isfinite(dist[nxt]):  # remaining vertices are unreachable islands
            unreached = np.where(~np.isfinite(dist))[0]
            if len(unreached) == 0:
                break
            nxt = int(unreached[0])
        seeds.append(nxt)
        d_new, _ = multi_source_dijkstra(n, adj, [nxt])
        dist = np.minimum(dist, d_new)
    return seeds


def greedy_coloring(cadj, k: int):
    """Largest-first greedy graph coloring. Returns color[k]."""
    order = sorted(range(k), key=lambda c: -len(cadj[c]))
    color = np.full(k, -1, dtype=np.int64)
    for c in order:
        used = {int(color[nb]) for nb in cadj[c] if color[nb] >= 0}
        col = 0
        while col in used:
            col += 1
        color[c] = col
    return color


def tri_corner_coeffs(tri_poses: np.ndarray) -> np.ndarray:
    """Per-(triangle, corner) dF/dx scalar coefficients (m,3,2) from the rest pose Dm_inv (m,2,2).

    Matches Newton's membrane F definition (particle_vbd_kernels.py): with DmInv = tri_poses,
    f0 = x01·DmInv00 + x02·DmInv10, f1 = x01·DmInv01 + x02·DmInv11. So corner 1 (=x1) has
    coeffs (DmInv00, DmInv01), corner 2 (=x2) has (DmInv10, DmInv11), corner 0 (=x0) is −(c1+c2).
    Corner k's vec(F) Jacobian is then J_k = [coeff[k,0]·I3 ; coeff[k,1]·I3] (6×3).
    """
    m = tri_poses.shape[0]
    coeff = np.zeros((m, 3, 2))
    coeff[:, 1, 0], coeff[:, 1, 1] = tri_poses[:, 0, 0], tri_poses[:, 0, 1]
    coeff[:, 2, 0], coeff[:, 2, 1] = tri_poses[:, 1, 0], tri_poses[:, 1, 1]
    coeff[:, 0, 0] = -(coeff[:, 1, 0] + coeff[:, 2, 0])
    coeff[:, 0, 1] = -(coeff[:, 1, 1] + coeff[:, 2, 1])
    return coeff


# --------------------------------------------------------------------------- #
# the full cluster system
# --------------------------------------------------------------------------- #
@dataclass
class ClusterData:
    """Device-ready (NumPy) arrays for the coarse cluster solve. CSR = offsets + flat values.

    Members are FREE vertices only (fixed / zero-mass vertices are dropped — the affine never
    moves them). ``ent_*`` are the Galerkin element-corner-pair gather entries; ``ent_rk``/``ent_rl``
    index into the flat ``clu_vert`` membership (to fetch r_k, r_l for P_k, P_l).
    """

    num_clusters: int
    elem_label: np.ndarray  # (num_tris,) tri -> cluster
    # cluster -> free-vertex members (CSR), with rest offsets r_i = X_i - centroid_c
    clu_vert_offsets: np.ndarray  # (num_clusters+1,)
    clu_vert: np.ndarray  # (M,) vertex ids
    clu_vert_r: np.ndarray  # (M,3) rest offsets
    # cluster -> Galerkin element-corner-pair entries (CSR)
    clu_ent_offsets: np.ndarray  # (num_clusters+1,)
    ent_tri: np.ndarray  # (E,) triangle id
    ent_k: np.ndarray  # (E,) corner k (0..2)
    ent_l: np.ndarray  # (E,) corner l (0..2)
    ent_rk: np.ndarray  # (E,) row into clu_vert for corner k's vertex
    ent_rl: np.ndarray  # (E,) row into clu_vert for corner l's vertex
    # cluster -> Galerkin dihedral-bending entries (CSR). Each entry is one bending edge owning
    # >=1 free stencil vertex of cluster c; bend_r0..3 are rows into clu_vert for the edge's four
    # stencil vertices (-1 if that vertex is not a free member of c).
    bend_offsets: np.ndarray  # (num_clusters+1,)
    bend_edge: np.ndarray  # (B,) bending-edge id
    bend_r0: np.ndarray  # (B,) row into clu_vert for stencil vert 0 (-1 if absent)
    bend_r1: np.ndarray  # (B,)
    bend_r2: np.ndarray  # (B,)
    bend_r3: np.ndarray  # (B,)
    # cluster coloring (CSR): clusters in a colour are vertex-disjoint
    color_offsets: np.ndarray  # (num_colors+1,)
    color_clusters: np.ndarray  # (num_clusters,)
    tri_coeff: np.ndarray  # (num_tris,3,2) per-corner dF/dx coefficients
    # stats
    num_colors: int
    avg_overlap: float
    num_boundary_verts: int
    num_disconnected: int


def build_cluster_system(
    verts: np.ndarray,
    faces: np.ndarray,
    free_mask: np.ndarray,
    tri_poses: np.ndarray,
    edges: np.ndarray | None = None,
    target_elems_per_cluster: int = 32,
    cluster_count: int | None = None,
    start: int = 0,
) -> ClusterData:
    """Build the cluster system from the rest mesh.

    Args:
        verts: rest positions, shape [n, 3].
        faces: triangle vertex indices, shape [m, 3] int.
        free_mask: shape [n] bool, True for solvable (free) vertices; fixed/zero-mass excluded.
        tri_poses: rest pose Dm_inv per triangle, shape [m, 2, 2] (Newton's model.tri_poses).
        target_elems_per_cluster: desired triangles per cluster (used if cluster_count is None).
        cluster_count: explicit number of clusters (overrides target_elems_per_cluster).
        start: FPS start seed (triangle index).
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    free_mask = np.asarray(free_mask, dtype=bool)
    m = len(faces)

    k = cluster_count if cluster_count is not None else max(1, m // max(1, target_elems_per_cluster))
    k = min(k, m)

    adj, _cent = build_dual_graph(verts, faces)
    seeds = fps_seeds(m, adj, k, start=start)
    _, e_label = multi_source_dijkstra(m, adj, seeds)
    if np.any(e_label < 0):  # disconnected tris -> nearest seed (Euclidean)
        tri_cent = verts[faces].mean(1)
        sc = tri_cent[seeds]
        for t in np.where(e_label < 0)[0]:
            e_label[t] = int(np.argmin(np.linalg.norm(sc - tri_cent[t], axis=1)))
    e_label = e_label.astype(np.int64)
    num_clusters = int(e_label.max()) + 1

    # (vertex, cluster) membership from element labels; drop non-free (pinned/fixed) vertices
    vc = np.unique(np.stack([faces.ravel(), np.repeat(e_label, 3)], axis=1), axis=0)
    free_vc = vc[free_mask[vc[:, 0]]]
    m_vid = free_vc[:, 0].astype(np.int64)
    m_clu = free_vc[:, 1].astype(np.int64)

    # cluster centroid over ALL touched verts (incl. fixed) for a stable rest frame
    cl_sum = np.zeros((num_clusters, 3))
    cl_cnt = np.zeros(num_clusters)
    np.add.at(cl_sum, vc[:, 1], verts[vc[:, 0]])
    np.add.at(cl_cnt, vc[:, 1], 1.0)
    centroid = cl_sum / np.maximum(cl_cnt[:, None], 1.0)
    m_r = verts[m_vid] - centroid[m_clu]

    # sort membership by cluster -> CSR (clu_vert / clu_vert_r), and a (vid,clu)->row lookup
    order = np.lexsort((m_vid, m_clu))
    m_vid, m_clu, m_r = m_vid[order], m_clu[order], m_r[order]
    clu_vert_offsets = np.zeros(num_clusters + 1, np.int64)
    np.add.at(clu_vert_offsets, m_clu + 1, 1)
    clu_vert_offsets = np.cumsum(clu_vert_offsets)
    row_of: dict[tuple[int, int], int] = {(int(v), int(c)): r for r, (v, c) in enumerate(zip(m_vid, m_clu))}
    clusters_of_vertex: dict[int, list[int]] = {}
    for v, c in zip(m_vid, m_clu):
        clusters_of_vertex.setdefault(int(v), []).append(int(c))

    # Galerkin element-corner-pair entries: for each triangle, each ordered corner pair (k,l) whose
    # vertices are both free members of a common cluster c -> an entry assembling P_k^T(J_k^T HF J_l)P_l.
    ent_c, ent_tri, ent_k, ent_l, ent_rk, ent_rl = [], [], [], [], [], []
    for e in range(m):
        vs = (int(faces[e, 0]), int(faces[e, 1]), int(faces[e, 2]))
        cs = [clusters_of_vertex.get(v, ()) for v in vs]
        for kk in range(3):
            for ll in range(3):
                for c in set(cs[kk]) & set(cs[ll]):
                    ent_c.append(c)
                    ent_tri.append(e)
                    ent_k.append(kk)
                    ent_l.append(ll)
                    ent_rk.append(row_of[(vs[kk], c)])
                    ent_rl.append(row_of[(vs[ll], c)])
    ent_c = np.asarray(ent_c, np.int64)
    ent_tri = np.asarray(ent_tri, np.int64)
    ent_k = np.asarray(ent_k, np.int64)
    ent_l = np.asarray(ent_l, np.int64)
    ent_rk = np.asarray(ent_rk, np.int64)
    ent_rl = np.asarray(ent_rl, np.int64)
    eorder = np.argsort(ent_c, kind="stable")
    ent_c, ent_tri, ent_k, ent_l, ent_rk, ent_rl = (
        ent_c[eorder],
        ent_tri[eorder],
        ent_k[eorder],
        ent_l[eorder],
        ent_rk[eorder],
        ent_rl[eorder],
    )
    clu_ent_offsets = np.zeros(num_clusters + 1, np.int64)
    np.add.at(clu_ent_offsets, ent_c + 1, 1)
    clu_ent_offsets = np.cumsum(clu_ent_offsets)

    # Galerkin dihedral-bending entries: for each bending edge (4 stencil verts vi0..vi3, with
    # (vi0,vi1) the opposite/wing tips and (vi2,vi3) the shared edge), each cluster c owning >=1
    # free stencil vertex -> a rank-1 entry k * G_c G_c^T with G_c = sum_{k in c} P_k^T dtheta/dx_k.
    bend_c, bend_edge, bend_rows = [], [], []
    if edges is not None and len(edges):
        edges = np.asarray(edges, np.int64).reshape(-1, 4)
        for e in range(len(edges)):
            st = edges[e]
            if st[0] < 0 or st[1] < 0:  # boundary edge -> no bending
                continue
            cand: set[int] = set()
            for vk in st:
                if vk >= 0:
                    cand.update(clusters_of_vertex.get(int(vk), ()))
            for c in cand:
                rows = [row_of.get((int(vk), c), -1) if vk >= 0 else -1 for vk in st]
                if any(r >= 0 for r in rows):
                    bend_c.append(c)
                    bend_edge.append(e)
                    bend_rows.append(rows)
    bend_c = np.asarray(bend_c, np.int64)
    bend_edge = np.asarray(bend_edge, np.int64)
    bend_rows = np.asarray(bend_rows, np.int64).reshape(-1, 4) if len(bend_edge) else np.zeros((0, 4), np.int64)
    if len(bend_edge):
        border = np.argsort(bend_c, kind="stable")
        bend_c, bend_edge, bend_rows = bend_c[border], bend_edge[border], bend_rows[border]
    bend_offsets = np.zeros(num_clusters + 1, np.int64)
    np.add.at(bend_offsets, bend_c + 1, 1)
    bend_offsets = np.cumsum(bend_offsets)

    # cluster coloring: clusters adjacent iff they share a free vertex
    cadj = [set() for _ in range(num_clusters)]
    for v, cl_list in clusters_of_vertex.items():
        for i in range(len(cl_list)):
            for j in range(i + 1, len(cl_list)):
                cadj[cl_list[i]].add(cl_list[j])
                cadj[cl_list[j]].add(cl_list[i])
    color = greedy_coloring(cadj, num_clusters)
    num_colors = int(color.max()) + 1 if num_clusters else 0
    corder = np.argsort(color, kind="stable")
    color_clusters = corder.astype(np.int64)
    color_offsets = np.zeros(num_colors + 1, np.int64)
    np.add.at(color_offsets, color[corder] + 1, 1)
    color_offsets = np.cumsum(color_offsets)

    # validation / stats
    bad_color = any(color[c] == color[nb] for c in range(num_clusters) for nb in cadj[c])
    if bad_color:
        raise ValueError("cluster coloring invalid: adjacent clusters share a colour")
    vcount = np.bincount(m_vid, minlength=len(verts))
    num_boundary = int((vcount > 1).sum())
    avg_overlap = float(vcount[vcount > 0].mean()) if num_boundary else 1.0
    num_disconnected = _count_disconnected(adj, e_label, num_clusters)

    return ClusterData(
        num_clusters=num_clusters,
        elem_label=e_label,
        clu_vert_offsets=clu_vert_offsets,
        clu_vert=m_vid,
        clu_vert_r=m_r.astype(np.float32),
        clu_ent_offsets=clu_ent_offsets,
        ent_tri=ent_tri,
        ent_k=ent_k.astype(np.int32),
        ent_l=ent_l.astype(np.int32),
        ent_rk=ent_rk,
        ent_rl=ent_rl,
        bend_offsets=bend_offsets,
        bend_edge=bend_edge.astype(np.int32),
        bend_r0=bend_rows[:, 0].astype(np.int32),
        bend_r1=bend_rows[:, 1].astype(np.int32),
        bend_r2=bend_rows[:, 2].astype(np.int32),
        bend_r3=bend_rows[:, 3].astype(np.int32),
        color_offsets=color_offsets,
        color_clusters=color_clusters,
        tri_coeff=tri_corner_coeffs(tri_poses).astype(np.float32),
        num_colors=num_colors,
        avg_overlap=avg_overlap,
        num_boundary_verts=num_boundary,
        num_disconnected=num_disconnected,
    )


def _count_disconnected(adj, label: np.ndarray, k: int) -> int:
    """How many clusters are NOT a single connected dual-graph patch (want 0)."""
    bad = 0
    for c in range(k):
        members = np.where(label == c)[0]
        if len(members) == 0:
            continue
        mset = set(int(x) for x in members)
        seen = {int(members[0])}
        stack = [int(members[0])]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if v in mset and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != len(mset):
            bad += 1
    return bad
