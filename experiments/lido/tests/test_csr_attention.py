# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Warp CSR attention against the torch reference: forward, gradients (float32 and float64 references), the
aggregated edge code, empty rows and graphs, and a printed benchmark at the training size (R=64000, 27 edges per
row, H=6, D=32)."""

import time
import unittest

import torch

from experiments.lido import network
from experiments.lido.csr_attention import csr_attention_warp

CUDA = torch.cuda.is_available()
DEV = "cuda"
CODE_WIDTH = {8: 10, 16: 96, 32: 96}  # edge code width per head width; 10: a code chunk width of 2 (not 4-aligned)


def torch_reference(q, k, v, edges, offsets, bias=None, add=None, code=None):
    """The torch body of network.csr_attention regardless of the dispatch flag."""
    prev = network.USE_WARP
    network.USE_WARP = False
    try:
        return network.csr_attention(q, k, v, edges, offsets, bias, add, code)
    finally:
        network.USE_WARP = prev


def random_graph(R, gen, max_len=40):
    """Random row lengths in 0..max_len with every seventh row empty; src uniform. Returns (edges, offsets) on DEV."""
    lengths = torch.randint(0, max_len + 1, (R,), generator=gen)
    lengths[::7] = 0
    offsets = torch.zeros(R + 1, dtype=torch.long)
    offsets[1:] = lengths.cumsum(0)
    E = int(offsets[-1])
    dst = torch.arange(R).repeat_interleave(lengths)
    src = torch.randint(0, R, (E,), generator=gen)
    return torch.stack([src, dst]).to(DEV), offsets.to(DEV)


def inputs(R, H, D, E, gen, use_bias, use_add, Dc=0):
    """q, k, v, bias, add and an edge code [E,Dc] (None for Dc = 0)."""
    q, k, v = (torch.randn(R, H, D, generator=gen).to(DEV).requires_grad_() for _ in range(3))
    bias = torch.randn(E, H, generator=gen).to(DEV).requires_grad_() if use_bias else None
    add = torch.randn(E, H, D, generator=gen).to(DEV).requires_grad_() if use_add else None
    code = torch.randn(E, Dc, generator=gen).to(DEV).requires_grad_() if Dc else None
    return q, k, v, bias, add, code


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


@unittest.skipUnless(CUDA, "cuda")
class TestCSRAttentionWarp(unittest.TestCase):
    R, H = 300, 3

    def cases(self):
        """(D, bias?, add?, code?, args): every combination; the code width 96 is the network's (10 for D = 8)."""
        gen = torch.Generator().manual_seed(0)
        for D in (8, 16, 32):
            for use_bias in (False, True):
                for use_add in (False, True):
                    for use_code in (False, True):
                        edges, offsets = random_graph(self.R, gen)
                        Dc = CODE_WIDTH[D] if use_code else 0
                        q, k, v, bias, add, code = inputs(self.R, self.H, D, edges.shape[1], gen, use_bias, use_add, Dc)
                        yield D, use_bias, use_add, use_code, (q, k, v, edges, offsets, bias, add, code)

    @staticmethod
    def outputs(fn, args):
        """(out, agg or None) of either implementation."""
        res = fn(*args)
        return res if args[7] is not None else (res, None)

    def test_forward_matches_torch(self):
        for D, use_bias, use_add, use_code, args in self.cases():
            with self.subTest(D=D, bias=use_bias, add=use_add, code=use_code):
                out, agg = self.outputs(csr_attention_warp, args)
                ref, ref_agg = self.outputs(torch_reference, args)
                self.assertEqual(out.shape, ref.shape)
                self.assertLess((out - ref).abs().max().item(), 1e-5)
                empty = (args[4][1:] == args[4][:-1]).nonzero().flatten()
                self.assertGreater(empty.numel(), 0)
                self.assertEqual(out[empty].abs().max().item(), 0.0)
                if use_code:
                    self.assertEqual(agg.shape, ref_agg.shape)
                    self.assertEqual(agg.shape[2], args[7].shape[1])
                    self.assertLess((agg - ref_agg).abs().max().item(), 1e-5)
                    self.assertEqual(agg[empty].abs().max().item(), 0.0)

    def test_gradients_match_torch(self):
        gen = torch.Generator().manual_seed(1)
        for D, use_bias, use_add, use_code, args in self.cases():
            with self.subTest(D=D, bias=use_bias, add=use_add, code=use_code):
                q, k, v, _, _, bias, add, code = args
                leaves = [t for t in (q, k, v, bias, add, code) if t is not None]
                grad_out = torch.randn(q.shape, generator=gen).to(DEV)
                dagg = torch.randn(q.shape[0], q.shape[1], code.shape[1], generator=gen).to(DEV) if use_code else None

                def grads(fn, args, grad_out, dagg):
                    out, agg = self.outputs(fn, args)
                    outs, douts = ((out, agg), (grad_out, dagg)) if agg is not None else ((out,), (grad_out,))
                    return torch.autograd.grad(outs, [t for t in args if t is not None and t.requires_grad], douts)

                g_warp = grads(csr_attention_warp, args, grad_out, dagg)
                g_ref = grads(torch_reference, args, grad_out, dagg)
                args64 = [
                    None
                    if t is None
                    else t.detach().double().requires_grad_(t.requires_grad)
                    if t.is_floating_point()
                    else t
                    for t in args
                ]
                g_ref64 = grads(torch_reference, args64, grad_out.double(), None if dagg is None else dagg.double())
                names = "qkv" + "b" * use_bias + "a" * use_add + "c" * use_code
                self.assertEqual(len(names), len(leaves))
                for name, gw, gr, gr64 in zip(names, g_warp, g_ref, g_ref64, strict=True):
                    self.assertLess(rel_err(gw, gr), 1e-4, name)
                    self.assertLess(rel_err(gw.double(), gr64), 1e-4, name)

    def test_code_gradient_through_the_scores_only(self):
        """With only the message output used, the code still reaches q, k and the bias through the softmax (the
        aggregation term of dw vanishes with a zero agg gradient): the gradients equal the ones without a code."""
        gen = torch.Generator().manual_seed(3)
        edges, offsets = random_graph(80, gen)
        q, k, v, bias, _, code = inputs(80, 2, 16, edges.shape[1], gen, True, False, 96)
        grad_out = torch.randn(q.shape, generator=gen).to(DEV)
        out, _ = csr_attention_warp(q, k, v, edges, offsets, bias, None, code)
        g_code = torch.autograd.grad(out, (q, k, v, bias, code), grad_out)
        g_plain = torch.autograd.grad(csr_attention_warp(q, k, v, edges, offsets, bias), (q, k, v, bias), grad_out)
        for a, b in zip(g_code[:4], g_plain, strict=True):
            self.assertLess(rel_err(a, b), 1e-5)
        self.assertEqual(g_code[4].abs().max().item(), 0.0)

    def test_unneeded_gradients_are_skipped(self):
        gen = torch.Generator().manual_seed(2)
        edges, offsets = random_graph(50, gen)
        q, k, v, bias, add, code = inputs(50, 2, 8, edges.shape[1], gen, True, True, 12)
        k, bias, code = k.detach(), bias.detach(), code.detach()
        out, agg = csr_attention_warp(q, k, v, edges, offsets, bias, add, code)
        (out.sum() + agg.sum()).backward()
        self.assertIsNone(k.grad)
        self.assertIsNone(bias.grad)
        self.assertIsNone(code.grad)
        q2, v2, add2 = (t.detach().requires_grad_() for t in (q, v, add))
        out2, agg2 = torch_reference(q2, k, v2, edges, offsets, bias, add2, code)
        (out2.sum() + agg2.sum()).backward()
        for a, b in ((q, q2), (v, v2), (add, add2)):
            self.assertLess(rel_err(a.grad, b.grad), 1e-4)

    def test_empty_graph(self):
        R, H, D = 20, 2, 16
        edges = torch.zeros(2, 0, dtype=torch.long, device=DEV)
        offsets = torch.zeros(R + 1, dtype=torch.long, device=DEV)
        q, k, v = (torch.randn(R, H, D, device=DEV, requires_grad=True) for _ in range(3))
        out = csr_attention_warp(q, k, v, edges, offsets)
        self.assertEqual(out.abs().max().item(), 0.0)
        out.sum().backward()
        self.assertEqual(q.grad.abs().max().item() + k.grad.abs().max().item() + v.grad.abs().max().item(), 0.0)
        out = csr_attention_warp(q[:0], k[:0], v[:0], edges, offsets[:1])
        self.assertEqual(out.shape, (0, H, D))

    def test_benchmark(self):
        """Prints forward and forward+backward times of the warp and torch paths at the training size."""
        R, per_row, H, D = 64000, 27, 6, 32
        gen = torch.Generator().manual_seed(0)
        E = R * per_row
        dst = torch.arange(R).repeat_interleave(per_row)
        src = torch.randint(0, R, (E,), generator=gen)
        edges = torch.stack([src, dst]).to(DEV)
        offsets = torch.arange(0, E + 1, per_row).to(DEV)
        q, k, v, bias, add, code = inputs(R, H, D, E, gen, True, True, 96)
        grad_out = torch.randn(R, H, D, device=DEV)
        dagg = torch.randn(R, H, 96, device=DEV)

        def timeit(fn, f, n=10):
            for _ in range(3):
                fn(f)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                fn(f)
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / n * 1e3

        def fwd(f):
            with torch.no_grad():
                f(q, k, v, edges, offsets, bias, add)

        def fwd_bwd(f):
            torch.autograd.grad(f(q, k, v, edges, offsets, bias, add), (q, k, v, bias, add), grad_out)

        def fwd_code(f):
            with torch.no_grad():
                f(q, k, v, edges, offsets, bias, None, code)

        def fwd_bwd_code(f):
            torch.autograd.grad(f(q, k, v, edges, offsets, bias, None, code), (q, k, v, bias, code), (grad_out, dagg))

        for name, f in (("torch", torch_reference), ("warp", csr_attention_warp)):
            print(
                f"\ncsr_attention {name}: R={R} E={E} H={H} D={D}  fwd {timeit(fwd, f):.2f} ms  "
                f"fwd+bwd {timeit(fwd_bwd, f):.2f} ms",
                flush=True,
            )
        # the network's form (bias + 96-value code; the torch reference would build the [E,H,96] product)
        print(
            f"csr_attention warp, bias + code 96: fwd {timeit(fwd_code, csr_attention_warp):.2f} ms  "
            f"fwd+bwd {timeit(fwd_bwd_code, csr_attention_warp):.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
