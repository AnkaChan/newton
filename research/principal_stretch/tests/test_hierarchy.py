# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the topological tet-graph hierarchy (construction + operators).

Construction is validated on the real training meshes (toy: 640 tets, 4k:
17280 tets — only the small ``tet_indices`` / ``rest_q`` members are read
from the npz files) and on a synthetic mesh of two disjoint boxes built
inline.  The two-box mesh miniaturises the U-bar property: clusters and
quotient graphs must never bridge disconnected components, because the
hierarchy is built on mesh topology, not Euclidean proximity.

The torch operators (``pool_mean``, ``pool_sum``, ``prolong``) are checked
against explicit python loops — values and gradients — on random data with
and without a leading batch dimension.
"""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path

import numpy as np
import torch

from research.principal_stretch.hierarchy import (
    build_hierarchy,
    pool_mean,
    pool_sum,
    prolong,
)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_TARGET = 8


def _load_mesh(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only tet_indices and rest_q (np.load is lazy — cheap)."""
    with np.load(_DATA_DIR / name) as data:
        return np.asarray(data["tet_indices"]), np.asarray(data["rest_q"])


def _box_mesh(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box of nx*ny*nz unit cells, 6 tets per cell (Kuhn).

    The Kuhn subdivision splits every cube into the 6 tets spanned by the
    monotone lattice paths from corner (0,0,0) to (1,1,1); it induces the
    same two-triangle split on every cube face, so adjacent cells are
    face-adjacent in the tet graph and the box is connected.
    """

    def vertex_id(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    verts = np.array(
        [[i, j, k] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)],
        dtype=np.float64,
    )
    tets = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        for perm in itertools.permutations(range(3)):
            corner = np.array([i, j, k])
            path = [corner]
            for axis in perm:
                corner = corner.copy()
                corner[axis] += 1
                path.append(corner)
            tets.append([vertex_id(*p) for p in path])
    return np.array(tets, dtype=np.int64), verts


def _two_box_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint 4x2x2 boxes in one tets array (two graph components)."""
    tets_1, verts_1 = _box_mesh(4, 2, 2)
    tets_2 = tets_1 + verts_1.shape[0]
    verts_2 = verts_1 + np.array([100.0, 0.0, 0.0])
    return np.concatenate([tets_1, tets_2]), np.concatenate([verts_1, verts_2])


def _components(adj: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected-component labels over a padded adjacency array."""
    n = adj.shape[0]
    comp = np.full(n, -1, dtype=np.int64)
    n_comp = 0
    for start in range(n):
        if comp[start] >= 0:
            continue
        comp[start] = n_comp
        stack = [start]
        while stack:
            a = stack.pop()
            for raw in adj[a]:
                b = int(raw)
                if b >= 0 and comp[b] < 0:
                    comp[b] = n_comp
                    stack.append(b)
        n_comp += 1
    return comp, n_comp


_CACHE: dict[str, object] = {}


def _toy_hierarchy():
    if "toy" not in _CACHE:
        tets, rest_q = _load_mesh("train.npz")
        _CACHE["toy"] = (build_hierarchy(tets, rest_q, n_levels=3, target=_TARGET), tets)
    return _CACHE["toy"]


def _two_box_hierarchy():
    if "two_box" not in _CACHE:
        tets, rest_q = _two_box_mesh()
        _CACHE["two_box"] = (build_hierarchy(tets, rest_q, n_levels=3, target=_TARGET), tets)
    return _CACHE["two_box"]


class TestHierarchy(unittest.TestCase):
    def test_partition(self):
        """Every node assigned exactly once; sizes in [target/2, 2*target]."""
        hier, tets = _toy_hierarchy()
        n_child = tets.shape[0]
        for level_index, level in enumerate(hier.levels):
            self.assertEqual(level.assign.shape, (n_child,))
            self.assertEqual(level.assign.dtype, np.int64)
            n_clusters = level.adj.shape[0]
            self.assertTrue((level.assign >= 0).all())
            self.assertTrue((level.assign < n_clusters).all())
            sizes = np.bincount(level.assign, minlength=n_clusters)
            self.assertTrue((sizes > 0).all(), f"level {level_index + 1}: empty cluster")
            self.assertGreaterEqual(sizes.min(), _TARGET // 2, f"level {level_index + 1}: sizes {sorted(sizes)}")
            self.assertLessEqual(sizes.max(), 2 * _TARGET, f"level {level_index + 1}: sizes {sorted(sizes)}")
            n_child = n_clusters

    def test_cluster_connectivity(self):
        """BFS inside each cluster over child adjacency reaches all members."""
        hier, _tets = _toy_hierarchy()
        child_adj = hier.tet_adj
        for level_index, level in enumerate(hier.levels):
            n_clusters = level.adj.shape[0]
            for cluster in range(n_clusters):
                members = {int(v) for v in np.nonzero(level.assign == cluster)[0]}
                first = min(members)
                seen = {first}
                stack = [first]
                while stack:
                    a = stack.pop()
                    for raw in child_adj[a]:
                        b = int(raw)
                        if b >= 0 and b in members and b not in seen:
                            seen.add(b)
                            stack.append(b)
                self.assertEqual(
                    seen,
                    members,
                    f"level {level_index + 1} cluster {cluster} is not connected",
                )
            child_adj = level.adj

    def test_topology_respected(self):
        """Two disjoint boxes: no cluster spans both, quotients stay disconnected."""
        hier, _tets = _two_box_hierarchy()
        comp, n_comp = _components(hier.tet_adj)
        self.assertEqual(n_comp, 2)  # sanity: the mesh really has two components
        child_comp = comp
        for level_index, level in enumerate(hier.levels):
            n_clusters = level.adj.shape[0]
            cluster_comp = np.full(n_clusters, -1, dtype=np.int64)
            for child, cluster in enumerate(level.assign):
                if cluster_comp[cluster] < 0:
                    cluster_comp[cluster] = child_comp[child]
                else:
                    self.assertEqual(
                        cluster_comp[cluster],
                        child_comp[child],
                        f"level {level_index + 1} cluster {cluster} spans both boxes",
                    )
            # Both boxes must survive as separate clusters at every level.
            self.assertEqual(set(cluster_comp.tolist()), {0, 1})
            for a in range(n_clusters):
                for b in level.adj[a]:
                    if b >= 0:
                        self.assertEqual(
                            cluster_comp[a],
                            cluster_comp[int(b)],
                            f"level {level_index + 1}: quotient edge bridges the boxes",
                        )
            child_comp = cluster_comp

    def test_quotient_symmetric(self):
        """A in adj[B] iff B in adj[A]; weights equal."""
        hier, _tets = _toy_hierarchy()
        graphs = [(hier.tet_adj, hier.tet_w_adj)]
        graphs += [(level.adj, level.w_adj) for level in hier.levels]
        for graph_index, (adj, w_adj) in enumerate(graphs):
            edges: dict[tuple[int, int], float] = {}
            for a in range(adj.shape[0]):
                for slot in range(adj.shape[1]):
                    b = int(adj[a, slot])
                    if b < 0:
                        continue
                    self.assertNotEqual(a, b, f"graph {graph_index}: self loop at {a}")
                    self.assertNotIn((a, b), edges, f"graph {graph_index}: duplicate neighbor {b} of {a}")
                    edges[(a, b)] = float(w_adj[a, slot])
                    self.assertGreater(w_adj[a, slot], 0.0)
            for (a, b), weight in edges.items():
                self.assertIn((b, a), edges, f"graph {graph_index}: edge ({a},{b}) one-sided")
                self.assertAlmostEqual(
                    weight / edges[(b, a)],
                    1.0,
                    places=12,
                    msg=f"graph {graph_index}: asymmetric weight on ({a},{b})",
                )

    def test_pou_rows_sum_to_one(self):
        for name, (hier, _tets) in (
            ("toy", _toy_hierarchy()),
            ("two_box", _two_box_hierarchy()),
        ):
            for level_index, level in enumerate(hier.levels):
                tag = f"{name} level {level_index + 1}"
                n_clusters = level.adj.shape[0]
                self.assertEqual(level.pou_idx.shape, level.pou_w.shape)
                self.assertEqual(level.pou_idx.shape[0], level.assign.shape[0])
                valid = level.pou_idx >= 0
                self.assertTrue((level.pou_idx[valid] < n_clusters).all(), tag)
                self.assertTrue((level.pou_w >= 0.0).all(), tag)
                self.assertTrue((level.pou_w[~valid] == 0.0).all(), f"{tag}: pad weight != 0")
                np.testing.assert_allclose(level.pou_w.sum(axis=1), 1.0, atol=1e-12, err_msg=tag)
                # The parent cluster is always a candidate with positive weight.
                for child, parent in enumerate(level.assign):
                    row = level.pou_idx[child]
                    (where,) = np.nonzero(row == parent)
                    self.assertEqual(len(where), 1, f"{tag}: parent missing for child {child}")
                    self.assertGreater(level.pou_w[child, where[0]], 0.0, tag)

    def test_pool_matches_loop(self):
        """pool_mean / pool_sum / prolong vs explicit python loops (values + grads)."""
        n_child, n_clusters = 40, 6
        rng = np.random.default_rng(0)
        assign_np = rng.permutation(np.arange(n_child) % n_clusters).astype(np.int64)
        assign = torch.from_numpy(assign_np)
        generator = torch.Generator().manual_seed(0)
        vol = torch.rand(n_child, dtype=torch.float64, generator=generator) + 0.1

        def loop_pool(x, node_dim, weighted):
            columns = []
            for cluster in range(n_clusters):
                members = [e for e in range(n_child) if assign_np[e] == cluster]
                take = (lambda e: x[e]) if node_dim == 0 else (lambda e: x[:, e])
                if weighted:
                    numerator = sum(vol[e] * take(e) for e in members)
                    columns.append(numerator / sum(vol[e] for e in members))
                else:
                    columns.append(sum(take(e) for e in members))
            return torch.stack(columns, dim=node_dim)

        for shape in [(n_child,), (n_child, 3), (n_child, 3, 3), (2, n_child, 3), (3, n_child, 3, 3)]:
            node_dim = 0 if shape[0] == n_child else 1
            for op, weighted in ((pool_mean, True), (pool_sum, False)):
                x = torch.randn(*shape, dtype=torch.float64, generator=generator)
                x = x.requires_grad_(True)
                out = op(x, assign, vol) if weighted else op(x, assign)
                ref = loop_pool(x, node_dim, weighted)
                self.assertEqual(out.shape, ref.shape, f"{shape} weighted={weighted}")
                self.assertTrue(torch.allclose(out, ref, atol=1e-12), f"{shape} weighted={weighted}")
                grad_out = torch.randn(out.shape, dtype=torch.float64, generator=generator)
                (grad,) = torch.autograd.grad(out, x, grad_out, retain_graph=False)
                (grad_ref,) = torch.autograd.grad(ref, x, grad_out)
                self.assertTrue(torch.allclose(grad, grad_ref, atol=1e-12), f"grad {shape} weighted={weighted}")
                self.assertGreater(grad.abs().max().item(), 0.0)

        # prolong: random PoU rows with -1 padding, rows sum to 1.
        n_candidates = 3
        pou_idx_np = rng.integers(0, n_clusters, size=(n_child, n_candidates))
        pou_idx_np[:, 0] = assign_np  # parent first; guarantees every cluster id occurs
        pou_idx_np[rng.random((n_child, n_candidates)) < 0.3] = -1
        pou_idx_np[:, 0] = np.abs(pou_idx_np[:, 0])  # keep at least one valid entry per row
        weights_np = rng.random((n_child, n_candidates)) * (pou_idx_np >= 0)
        weights_np /= weights_np.sum(axis=1, keepdims=True)
        pou_idx = torch.from_numpy(pou_idx_np.astype(np.int64))
        pou_w = torch.from_numpy(weights_np)

        def loop_prolong(y, node_dim):
            rows = []
            for child in range(n_child):
                acc = None
                for slot in range(n_candidates):
                    cluster = int(pou_idx_np[child, slot])
                    if cluster < 0:
                        continue
                    take = y[cluster] if node_dim == 0 else y[:, cluster]
                    term = pou_w[child, slot] * take
                    acc = term if acc is None else acc + term
                rows.append(acc)
            return torch.stack(rows, dim=node_dim)

        for shape in [(n_clusters,), (n_clusters, 3, 3), (2, n_clusters, 3)]:
            node_dim = 0 if shape[0] == n_clusters else 1
            y = torch.randn(*shape, dtype=torch.float64, generator=generator)
            y = y.requires_grad_(True)
            out = prolong(y, pou_idx, pou_w)
            ref = loop_prolong(y, node_dim)
            self.assertEqual(out.shape, ref.shape, f"prolong {shape}")
            self.assertTrue(torch.allclose(out, ref, atol=1e-12), f"prolong {shape}")
            grad_out = torch.randn(out.shape, dtype=torch.float64, generator=generator)
            (grad,) = torch.autograd.grad(out, y, grad_out)
            (grad_ref,) = torch.autograd.grad(ref, y, grad_out)
            self.assertTrue(torch.allclose(grad, grad_ref, atol=1e-12), f"prolong grad {shape}")
            self.assertGreater(grad.abs().max().item(), 0.0)

    def test_level_sizes_4k(self):
        """17280 tets -> [1500, 3000] -> [180, 400] -> [20, 60]."""
        tets, rest_q = _load_mesh("train_4k.npz")
        self.assertEqual(tets.shape[0], 17280)
        hier = build_hierarchy(tets, rest_q, n_levels=3, target=_TARGET)
        bounds = [(1500, 3000), (180, 400), (20, 60)]
        for level_index, (level, (low, high)) in enumerate(zip(hier.levels, bounds, strict=True)):
            n_clusters = level.adj.shape[0]
            self.assertGreaterEqual(n_clusters, low, f"level {level_index + 1}: {n_clusters}")
            self.assertLessEqual(n_clusters, high, f"level {level_index + 1}: {n_clusters}")


if __name__ == "__main__":
    unittest.main()
