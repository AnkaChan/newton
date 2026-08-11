# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the hierarchical stretch predictor (``HierStretchNet``).

Fixtures are built on the real toy mesh topology (``data/val.npz``, 640
tets, 225 vertices), loading only the small members plus two frames of
positions and one frame of external forces (``np.load`` is lazy per key).
Everything runs on CPU.

Design notes baked into the tests:

- The default fixture hierarchy is 3 levels deep (cluster counts 76 / 9 / 1
  on this mesh).  ``test_grad_reaches_all_levels`` instead uses a 2-level
  hierarchy (76 / 9): the 3-level hierarchy's coarsest level is a single
  cluster with no quotient edges, so every message it produces is
  aggregated with weight zero and its edge MLP is *structurally*
  unreachable by gradients — asserting a nonzero gradient there would be
  wrong, not strict.  The 3-level fixture still covers the no-edge code
  path in the other tests (it must run without NaN or error).
- A freshly constructed net has zero-initialised head output layers, so its
  forward is exactly the round-trip of ``S_t`` — which would make the
  SE(3), batching and gradient tests vacuous.  Those tests re-initialise
  every head's last layer with small seeded random values first.  For the
  gradient test this is required (zero head output layers multiply every
  gradient path into the MP stacks: ``tanh'(0) = 1`` keeps the heads
  themselves trainable from zero, but MP parameters only receive gradient
  once the head output layer is nonzero); for the SE(3) and batching tests
  it makes the full feature -> MP -> head path affect the compared output.
- The poked vertex in ``test_poke_visible_at_coarse`` is chosen with
  exactly 4 incident tets, so the per-tet vertex-mean forces sum back to
  exactly the injected 50 N and "total injected force / 30" is literal.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from research.principal_stretch import torch_solver as ts
from research.principal_stretch.hier_model import (
    F_EXT_MEAN_SLICE,
    F_EXT_SUM_SLICE,
    HierStretchNet,
)
from research.principal_stretch.hierarchy import build_hierarchy
from research.principal_stretch.model import build_face_adjacency, build_features
from research.principal_stretch.torch_solver import compute_S_from_x

_DATA = Path(__file__).resolve().parents[3] / "data" / "val.npz"
_FIX: dict = {}


def _fixture() -> dict:
    """Load the toy mesh + two frames once, build solver state and hierarchies."""
    if _FIX:
        return _FIX
    with np.load(_DATA) as d:
        tets = np.asarray(d["tet_indices"], dtype=np.int64)
        rest_q = np.asarray(d["rest_q"], dtype=np.float64)
        tet_poses = np.asarray(d["tet_poses"], dtype=np.float64)
        pinned = np.asarray(d["pinned_indices"], dtype=np.int64)
        mu = np.asarray(d["mu_per_tet"], dtype=np.float32)
        lam = np.asarray(d["lam_per_tet"], dtype=np.float32)
        gravity = np.asarray(d["gravity"], dtype=np.float64)
        x2 = np.asarray(d["x"][:2], dtype=np.float64)  # two frames only
        f1 = np.asarray(d["f_ext"][:1], dtype=np.float64)  # one frame only

    solver = ts.build_solver(rest_q, tets, tet_poses, pinned, device=torch.device("cpu"), dtype=torch.float64)

    pin_set = {int(v) for v in pinned}
    pin_flag = torch.tensor(
        [float(any(int(v) in pin_set for v in row)) for row in tets],
        dtype=torch.float32,
    )

    # An unpinned vertex with exactly 4 incident tets (see module docstring).
    incidence = np.zeros(rest_q.shape[0], dtype=np.int64)
    np.add.at(incidence, tets.reshape(-1), 1)
    candidates = [v for v in range(rest_q.shape[0]) if incidence[v] == 4 and v not in pin_set]
    assert candidates, "toy mesh should contain an unpinned vertex with exactly 4 incident tets"

    _FIX.update(
        tets_np=tets,
        solver=solver,
        face_adj=torch.as_tensor(build_face_adjacency(tets), dtype=torch.int64),
        pin_flag=pin_flag,
        mu32=torch.as_tensor(mu),
        lam32=torch.as_tensor(lam),
        gravity32=torch.as_tensor(gravity, dtype=torch.float32),
        x_prev=torch.as_tensor(x2[0]),
        x_t=torch.as_tensor(x2[1]),
        f_ext=torch.as_tensor(f1[0]),
        poke_vertex=int(candidates[0]),
        hier3=build_hierarchy(tets, rest_q, n_levels=3, target=8),
        hier2=build_hierarchy(tets, rest_q, n_levels=2, target=8),
    )
    return _FIX


def _make_inputs(fix: dict, x_t: torch.Tensor, x_prev: torch.Tensor, f_ext: torch.Tensor):
    """Rebuild (feat28, S_t) from positions exactly the way the trainer does."""
    solver = fix["solver"]
    S_prev = compute_S_from_x(solver, x_prev)
    S_t = compute_S_from_x(solver, x_t)
    feat28 = build_features(
        S_t.to(torch.float32),
        S_prev.to(torch.float32),
        fix["gravity32"],
        f_ext.to(torch.float32),
        fix["mu32"],
        fix["lam32"],
        fix["pin_flag"],
        solver.tets,
        fix["face_adj"],
    )
    return feat28, S_t


def _randomize_head_outputs(net: HierStretchNet, std: float, seed: int) -> None:
    """Test-only: replace every head's zero-initialised last layer with noise."""
    g = torch.Generator().manual_seed(seed)
    for head in net.heads:
        last = head[-1]
        with torch.no_grad():
            last.weight.copy_(std * torch.randn(last.weight.shape, generator=g, dtype=last.weight.dtype))
            last.bias.copy_(std * torch.randn(last.bias.shape, generator=g, dtype=last.bias.dtype))


def _random_rigid(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A random world rotation Q (det +1) and translation t, fp64."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, dtype=torch.float64, generator=g)
    Q, _ = torch.linalg.qr(A)
    if torch.linalg.det(Q).item() < 0.0:
        Q = Q.clone()
        Q[:, 0] = -Q[:, 0]
    t = torch.randn(3, dtype=torch.float64, generator=g)
    return Q, t


class TestHierModel(unittest.TestCase):
    def test_zero_init_identity(self):
        # Every head is zero-initialised, so the forward pass reduces to
        # sym_exp(sym_log(spd_floor(S_t))) — the round-trip of S_t.
        fix = _fixture()
        net = HierStretchNet(fix["hier3"])
        feat28, S_t = _make_inputs(fix, fix["x_t"], fix["x_prev"], fix["f_ext"])
        out = net(fix["solver"], fix["x_t"], fix["x_prev"], fix["f_ext"], feat28, S_t)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (fix["solver"].n_tets, 3, 3))
        self.assertLess((out - S_t.to(torch.float32)).abs().max().item(), 1e-5)

    def test_se3_invariance(self):
        # One random world rotation + translation applied to (x_t, x_prev);
        # feat28 and S_t rebuilt from the transformed positions the same way
        # the trainer does.  Zero external force, and gravity is NOT rotated
        # (it is the same constant channel in both runs), so the only
        # frame-dependence candidates are geometric.
        fix = _fixture()
        net = HierStretchNet(fix["hier3"])
        _randomize_head_outputs(net, std=0.05, seed=11)
        f_zero = torch.zeros_like(fix["f_ext"])
        Q, t = _random_rigid(seed=7)

        outs = []
        for transform in (False, True):
            x_t, x_prev = fix["x_t"], fix["x_prev"]
            if transform:
                x_t = x_t @ Q.T + t
                x_prev = x_prev @ Q.T + t
            feat28, S_t = _make_inputs(fix, x_t, x_prev, f_zero)
            outs.append(net(fix["solver"], x_t, x_prev, f_zero, feat28, S_t))
        self.assertLess((outs[0] - outs[1]).abs().max().item(), 1e-4)

    def test_poke_visible_at_coarse(self):
        # f_ext zero except one +50 N z on a single unpinned vertex with
        # exactly 4 incident tets (so the tet-level vertex-mean forces sum
        # back to exactly the injected 50 N).
        fix = _fixture()
        net = HierStretchNet(fix["hier3"])
        vid = fix["poke_vertex"]
        f = torch.zeros_like(fix["f_ext"])
        f[vid, 2] = 50.0
        feat28, S_t = _make_inputs(fix, fix["x_t"], fix["x_prev"], f)
        feats = net.level_features(fix["solver"], fix["x_t"], fix["x_prev"], f, feat28, S_t)
        self.assertEqual(len(feats), 3)

        total = f[fix["solver"].tets].mean(dim=-2).sum(dim=0)  # tet-level total
        self.assertLess((total - torch.tensor([0.0, 0.0, 50.0], dtype=total.dtype)).abs().max().item(), 1e-10)
        for level, lf in enumerate(feats, start=1):
            level_total = lf[:, F_EXT_SUM_SLICE].sum(dim=0)
            self.assertLess(
                (level_total - total.to(torch.float32) / 30.0).abs().max().item(),
                1e-4,
                f"f_ext_sum not conserved at level {level}",
            )

        # Coarsest-level ancestor of a poked tet: the extensive sum channel
        # must dominate the volume-diluted mean channel by > 100x.
        poked_tets = np.nonzero((fix["tets_np"] == vid).any(axis=1))[0]
        node = int(poked_tets[0])
        for lev in fix["hier3"].levels:
            node = int(lev.assign[node])
        sum_mag = feats[-1][node, F_EXT_SUM_SLICE].norm().item()
        mean_mag = feats[-1][node, F_EXT_MEAN_SLICE].norm().item()
        self.assertGreater(sum_mag, 100.0 * mean_mag)

    def test_grad_reaches_all_levels(self):
        # 2-level hierarchy: see the module docstring — the 3-level toy
        # hierarchy's coarsest level has no quotient edges, so its edge MLP
        # cannot receive gradient by construction.
        fix = _fixture()
        net = HierStretchNet(fix["hier2"])
        # Required per the task brief: with zero-initialised head output
        # layers d(loss)/d(MP params) is exactly zero (the head output layer
        # weight multiplies every gradient path into h'), even though
        # tanh'(0) = 1 lets the heads themselves train from zero.
        # Re-initialise the head output layers with small random values so
        # gradient propagates into the MP stacks (test-only).
        _randomize_head_outputs(net, std=1e-3, seed=3)

        feat28, S_t = _make_inputs(fix, fix["x_t"], fix["x_prev"], fix["f_ext"])
        out = net(fix["solver"], fix["x_t"], fix["x_prev"], fix["f_ext"], feat28, S_t)
        out.sum().backward()  # decoded-position proxy loss per the task brief

        for i, head in enumerate(net.heads):
            g = head[-1].weight.grad
            self.assertIsNotNone(g, f"head {i} last layer has no grad")
            self.assertGreater(g.norm().item(), 0.0, f"head {i} last-layer grad is zero")
        levels = zip(net.encoders, net.edge_mlps, net.node_mlps, strict=True)
        for level, (encoder, edge_rounds, node_rounds) in enumerate(levels, start=1):
            g = encoder[0].weight.grad
            self.assertIsNotNone(g, f"encoder level {level} has no grad")
            self.assertGreater(g.norm().item(), 0.0, f"encoder level {level} first-layer grad is zero")
            for r, mlp in enumerate(edge_rounds):
                g = mlp[0].weight.grad
                self.assertIsNotNone(g, f"edge MLP level {level} round {r} has no grad")
                self.assertGreater(g.norm().item(), 0.0, f"edge MLP level {level} round {r} grad is zero")
            for r, mlp in enumerate(node_rounds):
                g = mlp[0].weight.grad
                self.assertIsNotNone(g, f"node MLP level {level} round {r} has no grad")
                self.assertGreater(g.norm().item(), 0.0, f"node MLP level {level} round {r} grad is zero")

    def test_batched(self):
        # B = 3 stacked frames (the two loaded frames repeated with small
        # noise): the batched forward must equal a per-sample loop.
        fix = _fixture()
        net = HierStretchNet(fix["hier3"])
        _randomize_head_outputs(net, std=0.05, seed=5)

        g = torch.Generator().manual_seed(21)
        B = 3
        noise = lambda shape: 1e-4 * torch.randn(*shape, dtype=torch.float64, generator=g)  # noqa: E731
        x_t = fix["x_t"][None].expand(B, -1, -1) + noise((B, *fix["x_t"].shape))
        x_prev = fix["x_prev"][None].expand(B, -1, -1) + noise((B, *fix["x_prev"].shape))
        f_ext = fix["f_ext"][None].expand(B, -1, -1).contiguous()
        feat28, S_t = _make_inputs(fix, x_t, x_prev, f_ext)

        out_b = net(fix["solver"], x_t, x_prev, f_ext, feat28, S_t)
        self.assertEqual(tuple(out_b.shape), (B, fix["solver"].n_tets, 3, 3))
        for b in range(B):
            out_s = net(fix["solver"], x_t[b], x_prev[b], f_ext[b], feat28[b], S_t[b])
            self.assertLess((out_b[b] - out_s).abs().max().item(), 1e-5, f"sample {b}")


if __name__ == "__main__":
    unittest.main()
