# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The "pair" edge module (TrainConfig.edge_module = "pair", design record section 10): widths, the untrained net as
a no-op, the sender block of the features, equivariance of the fused update and the loss, gradients to every
parameter, the per-edge multiply-add budget, the a02 path left bitwise as it was, and the pair module inside the
captured and the compiled inference paths."""

import os
import unittest

import torch
from torch.profiler import ProfilerActivity, profile

from experiments.lido import physics
from experiments.lido.config import TrainConfig
from experiments.lido.features import EDGE_WIDTH, NODE_WIDTH, SENDER_WIDTH, TOKEN_WIDTH, features, pack
from experiments.lido.frames import frames
from experiments.lido.fusion import Fusion
from experiments.lido.network import Net
from experiments.lido.step import Step
from experiments.lido.structs import Features
from experiments.lido.tests import test_capture as capture_tests  # module imports: no duplicate collection
from experiments.lido.tests import test_parity as parity_tests
from experiments.lido.tests.test_step import make, rotate_batch
from experiments.lido.validation import local_objective

PAIR = {"edge_module": "pair"}
PAIR_SMALL_CFG = TrainConfig(**capture_tests.SMALL, edge_module="pair")
V4 = "generated/training_v4_20260928/checkpoints/best_validation.pt"
GEMMS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm")


def random_features(C, O, Q, per_cell=3, sender=True, seed=0):
    """Random Features on a ring graph with `per_cell` directed edges into every cell (itself and its neighbours)."""
    torch.manual_seed(seed)
    E = per_cell * C
    cell_obj = torch.arange(C) % O
    dst = torch.arange(C).repeat_interleave(per_cell)
    src = (dst + (torch.arange(per_cell) - per_cell // 2).repeat(C)) % C
    edges = torch.stack([src, dst])
    offsets = torch.arange(0, E + 1, per_cell)
    token_offsets = torch.zeros(C + 1, dtype=torch.long)
    token_offsets[1:] = torch.bincount(torch.randint(0, C, (Q,)), minlength=C).cumsum(0)
    f = Features(
        node=torch.randn(C, NODE_WIDTH),
        edge_attr=torch.randn(E, EDGE_WIDTH),
        cond=torch.randn(O, 7),
        tokens=torch.randn(Q, TOKEN_WIDTH),
        token_offsets=token_offsets,
        edge_sender=torch.randn(E, SENDER_WIDTH) if sender else None,
    )
    return f, edges, offsets, cell_obj


def randomized(net: Net, scale=0.1) -> Net:
    with torch.no_grad():
        for p in net.parameters():
            p.add_(scale * torch.randn_like(p))
    return net


def gemm_macs(fn) -> int:
    """Multiply-adds of the GEMMs (mm, addmm, bmm, baddbmm) in one call of `fn`, from the profiler's flop count."""
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        fn()
    return sum(e.flops for e in prof.key_averages() if e.key in GEMMS) // 2


class TestPairEdge(unittest.TestCase):
    def test_widths(self):
        net = Net(**PAIR)
        self.assertEqual(net.edge_module, "pair")
        self.assertIsNone(net.edge_encoder)
        self.assertIsNone(net.layers[0].edge_update)
        self.assertEqual(net.pair_edge.mlp[0].weight.shape, (96, 66))  # geometry 24 | receiver 21 | sender 21
        self.assertEqual(net.pair_edge.mlp[2].weight.shape, (96, 96))  # code width = hidden width
        self.assertEqual(net.layers[0].edge_bias.weight.shape, (6, 96))
        self.assertEqual(net.layers[0].edge_val.weight.shape, (192, 96))
        keys = list(net.state_dict())
        self.assertFalse(any(k.startswith("edge_encoder") or ".edge_update." in k for k in keys))
        self.assertEqual(
            [k for k in keys if k.startswith("pair_edge")],
            [f"pair_edge.mlp.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")],
        )
        # one knob: cfg.edge_hidden_dim sets the hidden and the code width
        small = Net.from_config(TrainConfig(edge_module="pair", edge_hidden_dim=48, hidden_dim=96, num_heads=3))
        self.assertEqual(small.pair_edge.mlp[0].weight.shape, (48, 66))
        self.assertEqual(small.pair_edge.mlp[2].weight.shape, (48, 48))
        self.assertEqual(small.layers[0].edge_bias.weight.shape, (3, 48))
        self.assertEqual(small.layers[0].edge_val.weight.shape, (96, 48))
        with self.assertRaises(ValueError):
            Net(edge_module="a03")
        with self.assertRaises(KeyError):
            TrainConfig.from_dict({"edge_modules": "pair"})

    def test_a02_is_the_current_network(self):
        """The default edge module builds the same parameters, in the same order, with the same values."""
        self.assertEqual(
            TrainConfig().edge_module, "pair"
        )  # default for new training since 2026-10-02; Net() keeps a02 for the trained checkpoints
        self.assertEqual(TrainConfig.from_dict({"edge_network": True, "edge_hidden_dim": 96}).edge_module, "pair")
        torch.manual_seed(3)
        a = Net()
        torch.manual_seed(3)
        b = Net.from_config(TrainConfig.from_dict({"edge_module": "a02"}))
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(list(sa), list(sb))
        self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))
        self.assertIn("edge_encoder.0.weight", sa)
        self.assertIn("layers.0.edge_update.0.weight", sa)
        self.assertEqual(sa["edge_encoder.0.weight"].shape, (96, 24))
        self.assertEqual(sa["layers.0.edge_update.0.weight"].shape, (192, 480))
        self.assertEqual(len(sa), 58)  # the v4 checkpoint's 60 entries minus its two neighbour-table buffers
        if os.path.exists(V4):
            state = torch.load(V4, map_location="cpu", weights_only=False)["network_state"]
            self.assertEqual(sorted(k for k in state if not k.startswith("neighbor_")), sorted(sa))
        self.assertFalse(Step(a).sender)
        self.assertTrue(Step(Net(**PAIR)).sender)

    def test_untrained_net_is_a_no_op(self):
        net = Net(**PAIR)
        f, edges, offsets, cell_obj = random_features(12, 2, 7)
        out = net(f, edges, offsets, cell_obj)
        self.assertEqual(out.corr.shape, (12, 7, 3))
        self.assertEqual(out.corr.abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(out.step, torch.full((12,), 0.025)))
        # through the step: the candidate stays where it is
        _, b, step, _ = make(randomize=False, net_kwargs=PAIR)
        out = step.query(b)
        self.assertLess((out.cand_after - b.x).abs().max().item(), 1e-12)
        self.assertLess((out.E_after - out.E_before).abs().max().item(), 1e-6)

    def test_sender_required(self):
        net = Net(**PAIR)
        f, edges, offsets, cell_obj = random_features(6, 1, 0, sender=False)
        with self.assertRaises(ValueError):
            net(f, edges, offsets, cell_obj)

    def test_sender_block_of_the_features(self):
        """edge_sender = pack(R_i^T m_j) per directed edge: the receiver's own node values on the self edges, its
        first nine entries the R_i^T F_j block of the geometry, and absent (features unchanged) without `sender`."""
        _, _, _, X, Y, X_prev = parity_tests.si_state()
        g, b = parity_tests.new_batch(X, Y, X_prev)
        _, b.gX = physics.energy_and_grad(b, b.x)
        m_c, F_c = physics.modes_and_center(b.x, b)
        R = frames(F_c.float(), b.R_ref[b.cell_obj].float()).double()
        g_m = Fusion().project_gradient(b, b.gX)
        f0 = features(b, b.x, R, m_c, F_c, g_m)
        f = features(b, b.x, R, m_c, F_c, g_m, sender=True)
        self.assertIsNone(f0.edge_sender)
        self.assertTrue(torch.equal(f0.edge_attr, f.edge_attr))
        src, dst = b.edges
        E = src.numel()
        self.assertEqual(f.edge_sender.shape, (E, SENDER_WIDTH))
        ref = pack(torch.einsum("eab,evb->eva", R.transpose(1, 2)[dst], m_c[src]))
        self.assertLess((f.edge_sender - ref).abs().max().item(), 1e-14)
        self_edge = src == dst
        self.assertEqual(int(self_edge.sum()), g.C)
        self.assertLess((f.edge_sender[self_edge] - f.node[dst[self_edge], :SENDER_WIDTH]).abs().max().item(), 1e-14)
        axes = f.edge_sender.view(E, 3, 7)[:, :, :3]  # pack layout 7 a + v: component a of mode v
        self.assertLess((axes - f.edge_attr[:, 15:].view(E, 3, 3)).abs().max().item(), 1e-14)

    def test_equivariance(self):
        """As test_step.test_equivariance, with the pair edge module (float64 physics, float32 frames)."""
        _, b, step, _ = make(dtype=torch.float64, net_kwargs=PAIR)
        out = step.query(b)
        self.assertGreater((out.cand_after - b.x).abs().max().item(), 1e-6)  # the randomised net moves the candidate
        loss = local_objective(out.E_after, out.E_before, b.material.floor)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        q = q * torch.sign(torch.det(q))
        t = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
        _, b2, step2, _ = make(dtype=torch.float64, net_kwargs=PAIR)
        step2.net.load_state_dict(step.net.state_dict())
        rotate_batch(b2, q, t)
        b2.material.g = b2.material.g @ q.t()
        step2.prepare(b2, torch.ones(2, dtype=torch.bool), None)
        b2.x = b.x @ q.t() + t
        b2.E, b2.gX = physics.energy_and_grad(b2, b2.x)
        self.assertLess((b2.E - b.E).abs().max().item(), 1e-9)
        out2 = step2.query(b2)
        d1 = (out.cand_after - b.x) @ q.t()
        d2 = out2.cand_after - b2.x
        self.assertLess((d1 - d2).abs().max().item(), 1e-5 * max(1.0, d1.abs().max().item()))
        loss2 = local_objective(out2.E_after, out2.E_before, b2.material.floor)
        self.assertLess((loss - loss2).abs().max().item(), 1e-5)

    def test_gradients_reach_every_parameter(self):
        for Q in (9, 0):  # with contact tokens and without (DDP needs a gradient for every parameter)
            net = randomized(Net(**PAIR))
            f, edges, offsets, cell_obj = random_features(10, 2, Q)
            out = net(f, edges, offsets, cell_obj)
            self.assertLess(out.corr.flatten(1).norm(dim=-1).max().item(), 1.0)
            (out.corr.sum() + out.step.sum()).backward()
            for name, p in net.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertTrue(all(p.grad.abs().max() > 0 for p in net.pair_edge.parameters()))
        # through the step and the loss
        _, b, step, _ = make(net_kwargs=PAIR)
        out = step.query(b)
        local_objective(out.E_after, out.E_before, b.material.floor).mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in step.net.pair_edge.parameters()))

    def test_per_edge_multiply_adds_within_budget(self):
        """Multiply-adds per directed edge, hidden 96, six heads of width 32.

        GEMMs, per edge: geometry block 24 x 96 = 2304, sender block 21 x 96 = 2016, second linear 96 x 96 = 9216,
          edge bias 96 x 6 = 576: 14112. Measured here as the GEMM flops the profiler attributes to a forward at two
          edge counts (same cells, same tokens), difference over the extra edges.
        Attention, per edge (segment reductions, counted by hand): score 6 x 32 = 192, value 6 x 32 = 192, code
          aggregation 6 x 96 = 576: 960.
        Per cell, shared by its (at most) 27 edges: receiver block 21 x 96 = 2016, edge values U on the aggregated
          code 6 x 96 x 32 = 18432: 20448 per cell, 758 per edge.
        Total 14112 + 960 + 758 = 15830 <= 16000. (a02 for comparison: encoder 24 x 96 + 96 x 96 = 11520, A02 update
        96 x 192 + 192 x 96 = 36864 with its node halves folded per cell, bias 576, attention 960: about 49.9k.)"""
        net = randomized(Net(**PAIR))
        C, O, Q = 40, 2, 11
        macs = {}
        for per_cell in (3, 7):
            inputs = random_features(C, O, Q, per_cell)
            with torch.no_grad():
                macs[per_cell] = gemm_macs(lambda inputs=inputs: net(*inputs))
        per_edge_gemm = (macs[7] - macs[3]) // (4 * C)
        hidden, heads, width = 96, 6, 192
        self.assertEqual(per_edge_gemm, (EDGE_WIDTH + SENDER_WIDTH + hidden) * hidden + hidden * heads)  # 14112
        attention = 2 * width + heads * hidden  # 960
        per_cell = SENDER_WIDTH * hidden + heads * hidden * (width // heads)  # 20448
        total = per_edge_gemm + attention + -(-per_cell // 27)
        self.assertLessEqual(total, 16000, f"{total} multiply-adds per edge")
        self.assertEqual(total, 15830)


@unittest.skipUnless(capture_tests.HAS_CUDA, "cuda")
class TestPairCapturedQuery(capture_tests.TestCapturedQuery):
    """CUDA-graph replay of the query with the pair edge module against eager queries (within 1e-5)."""

    cfg = PAIR_SMALL_CFG


@unittest.skipUnless(capture_tests.HAS_CUDA, "cuda")
class TestPairCompiledInference(capture_tests.TestCompiledInference):
    """The compiled rollout chains (layer, pair edge module, feature chains with the sender block) against eager."""

    cfg = PAIR_SMALL_CFG


if __name__ == "__main__":
    unittest.main()
