# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Network: CSR attention against dense softmax, token pair ranges, zero-init no-op, the aggregated edge values
against the per-edge form, v4 state-dict compatibility."""

import os
import unittest
import unittest.mock

import torch

from experiments.lido import network
from experiments.lido.features import EDGE_WIDTH, NODE_WIDTH, TOKEN_WIDTH
from experiments.lido.network import Layer, Net, csr_attention, gather, within_range_pairs
from experiments.lido.structs import Features

V4 = "generated/training_v4_20260928/checkpoints/best_validation.pt"


def per_edge_layer_forward(self: Layer, x, e, edges, offsets, cond):
    """Layer.forward with the edge values added per edge, val_ij = v_j + edge_val(e_ij) (the form before the
    aggregated edge code; the edge_update.0 concatenation materialised). The reference for the equivalence test."""
    g1, b1, g2, b2 = self.film(cond).chunk(4, -1)
    n = self.attention_norm(x) * (1 + g1) + b1
    C = x.shape[0]
    q, k, v = self.qkv(n).view(C, 3, self.heads, -1).unbind(1)
    src, dst = edges
    if self.edge_update is not None:
        e = e + self.edge_update(torch.cat([gather(n, dst), gather(n, src), e], -1))
    msg = csr_attention(q, k, v, edges, offsets, self.edge_bias(e), self.edge_val(e).view(-1, self.heads, q.shape[-1]))
    x = x + self.out_projection(msg.reshape(C, -1))
    n = self.ffn_norm(x) * (1 + g2) + b2
    return x + self.ffn(n)


class TestNetwork(unittest.TestCase):
    def test_csr_attention_matches_dense(self):
        torch.manual_seed(0)
        R, H, D = 5, 4, 8
        q, k, v = (torch.randn(R, H, D, dtype=torch.float64) for _ in range(3))
        dst = torch.arange(R).repeat_interleave(R)
        src = torch.arange(R).repeat(R)
        edges = torch.stack([src, dst])
        offsets = torch.arange(0, R * R + 1, R)
        bias = torch.randn(R * R, H, dtype=torch.float64)
        add = torch.randn(R * R, H, D, dtype=torch.float64)
        code = torch.randn(R * R, 6, dtype=torch.float64)
        out = csr_attention(q, k, v, edges, offsets, bias, add)
        s = torch.einsum("ihd,jhd->hij", q, k) / D**0.5 + bias.view(R, R, H).permute(2, 0, 1)
        w = s.softmax(-1)  # [H,i,j]
        vals = v[None].expand(R, R, H, D) + add.view(R, R, H, D)  # [i,j,H,D]
        ref = torch.einsum("hij,ijhd->ihd", w, vals)
        self.assertLess((out - ref).abs().max().item(), 1e-12)
        out2, agg = csr_attention(q, k, v, edges, offsets, bias, add, code)
        self.assertTrue(torch.equal(out2, out))
        self.assertLess((agg - torch.einsum("hij,ijc->ihc", w, code.view(R, R, 6))).abs().max().item(), 1e-12)

    def test_within_range_pairs(self):
        offsets = torch.tensor([0, 0, 3, 3, 5])
        pairs, po = within_range_pairs(offsets)
        self.assertEqual(po.tolist(), [0, 3, 6, 9, 11, 13])
        self.assertEqual(pairs[1].tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4])
        self.assertEqual(pairs[0].tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4])
        pairs, po = within_range_pairs(torch.tensor([0, 0, 0]))
        self.assertEqual(pairs.shape, (2, 0))

    def _features(self, C, O, Q):
        node = torch.randn(C, NODE_WIDTH)
        cell_obj = torch.arange(C) % O
        dst = torch.arange(C).repeat_interleave(3)
        src = (dst + torch.tensor([0, 1, -1]).repeat(C)) % C
        edges = torch.stack([src, dst])
        offsets = torch.arange(0, 3 * C + 1, 3)
        token_offsets = torch.zeros(C + 1, dtype=torch.long)
        token_offsets[1:] = torch.bincount(torch.randint(0, C, (Q,)), minlength=C).cumsum(0)
        f = Features(
            node=node,
            edge_attr=torch.randn(3 * C, EDGE_WIDTH),
            cond=torch.randn(O, 7),
            tokens=torch.randn(Q, TOKEN_WIDTH),
            token_offsets=token_offsets,
        )
        return f, edges, offsets, cell_obj

    def test_zero_init_is_a_no_op(self):
        net = Net()
        f, edges, offsets, cell_obj = self._features(12, 2, 7)
        out = net(f, edges, offsets, cell_obj)
        self.assertEqual(out.corr.shape, (12, 7, 3))
        self.assertEqual(out.step.shape, (12,))
        self.assertEqual(out.corr.abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(out.step, torch.full((12,), 0.025)))

    def test_contact_free_batch(self):
        net = Net()
        f, edges, offsets, cell_obj = self._features(6, 1, 0)
        out = net(f, edges, offsets, cell_obj)
        self.assertTrue(torch.isfinite(out.corr).all())
        (out.corr.sum() + out.step.sum()).backward()
        # every parameter takes part in the graph even without contact tokens (DDP needs that)
        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_bounded_correction_and_gradients(self):
        net = Net()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1 * torch.randn_like(p))
        f, edges, offsets, cell_obj = self._features(10, 2, 5)
        out = net(f, edges, offsets, cell_obj)
        self.assertLess(out.corr.flatten(1).norm(dim=-1).max().item(), 1.0)
        self.assertTrue(((out.step > 0) & (out.step < 0.05)).all())
        (out.corr.sum() + out.step.sum()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def _equivalence_inputs(self, device):
        torch.manual_seed(0)
        net = Net().to(device)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1 * torch.randn_like(p))
        f, edges, offsets, cell_obj = self._features(64, 2, 40)
        f = Features(
            node=f.node.to(device),
            edge_attr=f.edge_attr.to(device),
            cond=f.cond.to(device),
            tokens=f.tokens.to(device),
            token_offsets=f.token_offsets.to(device),
        )
        return net, f, edges.to(device), offsets.to(device), cell_obj.to(device)

    def _run(self, net, f, edges, offsets, cell_obj):
        net.zero_grad()
        out = net(f, edges, offsets, cell_obj)
        (out.corr.sum() + out.step.sum()).backward()
        return [out.corr.detach(), out.step.detach()] + [p.grad.clone() for p in net.parameters()]

    def _assert_aggregated_matches_per_edge(self, device):
        """The aggregated edge values (U applied to sum_j w_ij e_ij per cell) against the per-edge form
        (sum_j w_ij U e_ij): outputs within 1e-5 and parameter gradients within 1e-4 relative, matmuls in
        full float32 (the TF32 setting of Net.forward is patched out: TF32 rounding is 1e-3)."""
        net, f, edges, offsets, cell_obj = self._equivalence_inputs(device)
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            with unittest.mock.patch("torch.set_float32_matmul_precision"):
                new = self._run(net, f, edges, offsets, cell_obj)
                with unittest.mock.patch.object(Layer, "forward", per_edge_layer_forward):
                    old = self._run(net, f, edges, offsets, cell_obj)
        finally:
            torch.set_float32_matmul_precision(precision)
        self.assertGreater(new[0].abs().max().item(), 0.0)
        names = ["corr", "step"] + [n for n, _ in net.named_parameters()]
        for name, a, b in zip(names, new, old, strict=True):
            tol = 1e-5 if name in ("corr", "step") else 1e-4
            # floor: analytic zeros (the edge_bias bias gradient: the score gradients of a row sum to zero)
            self.assertLess((a - b).abs().max().item(), tol * b.abs().max().item() + 1e-6, name)

    def test_aggregated_edge_values_match_per_edge_form(self):
        self._assert_aggregated_matches_per_edge("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_aggregated_edge_values_match_per_edge_form_on_cuda(self):
        self._assert_aggregated_matches_per_edge("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_warp_attention_matches_torch_path_on_cuda(self):
        """Net forward + backward through the Warp attention against the torch path. The matmuls run in full fp32
        here (the TF32 setting of Net.forward is patched out): with TF32 the float32-rounding-level attention
        differences flip input quantisation and show up as 1e-4 relative differences downstream."""
        torch.manual_seed(0)
        net = Net().cuda()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1 * torch.randn_like(p))
        f, edges, offsets, cell_obj = self._features(64, 2, 40)
        f = Features(
            node=f.node.cuda(),
            edge_attr=f.edge_attr.cuda(),
            cond=f.cond.cuda(),
            tokens=f.tokens.cuda(),
            token_offsets=f.token_offsets.cuda(),
        )
        edges, offsets, cell_obj = edges.cuda(), offsets.cuda(), cell_obj.cuda()
        results = {}
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            with unittest.mock.patch("torch.set_float32_matmul_precision"):
                for use_warp in (False, True):
                    network.USE_WARP = use_warp
                    net.zero_grad()
                    out = net(f, edges, offsets, cell_obj)
                    (out.corr.sum() + out.step.sum()).backward()
                    results[use_warp] = [out.corr.detach(), out.step.detach()] + [
                        p.grad.clone() for p in net.parameters()
                    ]
        finally:
            network.USE_WARP = True
            torch.set_float32_matmul_precision(precision)
        for a, b in zip(results[True], results[False], strict=True):
            self.assertLess((a - b).abs().max().item(), 1e-4 * b.abs().max().item() + 1e-6)  # floor: analytic zeros

    @unittest.skipUnless(os.path.exists(V4), "v4 checkpoint")
    def test_v4_state_dict_loads(self):
        state = torch.load(V4, map_location="cpu", weights_only=False)["network_state"]
        state = {k: v for k, v in state.items() if not k.startswith("neighbor_")}
        net = Net()
        missing, unexpected = net.load_state_dict(state, strict=False)
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])


if __name__ == "__main__":
    unittest.main()
