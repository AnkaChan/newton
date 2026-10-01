# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The float32 frames kernel against a float64 SVD closest proper rotation (design spec section 8, Frames)."""

import time
import unittest

import torch

from experiments.lido.frames import TIE_EPS, frames

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
N = 4096


def reference(F):
    """Closest proper rotation U diag(1, 1, det(U V^T)) V^T in float64; also returns the singular values."""
    U, S, Vh = torch.linalg.svd(F.double())
    D = torch.diag_embed(
        torch.stack([torch.ones_like(S[:, 0]), torch.ones_like(S[:, 0]), torch.linalg.det(U @ Vh)], -1)
    )
    return U @ D @ Vh, S


def rotations(n, gen):
    Q, _ = torch.linalg.qr(torch.randn(n, 3, 3, generator=gen, dtype=torch.float64))
    Q[:, :, 2] *= torch.linalg.det(Q).sign()[:, None]
    return Q


def cells(s, gen, inverted=False):
    """F = U diag(s) V^T with random proper rotations U, V; the inverted variant flips one column of U."""
    U, V = rotations(s.shape[0], gen), rotations(s.shape[0], gen)
    if inverted:
        U[:, :, 2] *= -1
    return (U @ torch.diag_embed(s) @ V.transpose(-1, -2)).float()


def log_uniform(shape, lo, hi, gen):
    return lo * (hi / lo) ** torch.rand(shape, generator=gen, dtype=torch.float64)


class TestFrames(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)

    def check(self, F, tol, R_ref=None):
        R_ref = rotations(F.shape[0], self.gen).float() if R_ref is None else R_ref
        R_exp, _ = reference(F)
        for dev in DEVICES:
            with self.subTest(device=dev):
                R = frames(F.to(dev), R_ref.to(dev))
                self.assertEqual(R.dtype, torch.float32)
                self.assertFalse(R.requires_grad)
                err = (R.cpu().double() - R_exp).abs().amax((-1, -2)).max().item()
                self.assertLess(err, tol, f"max error {err:.2e} on {dev}")

    def test_random_well_conditioned(self):
        self.check(cells(log_uniform((N, 3), 0.5, 2.0, self.gen), self.gen), 1e-5)

    def test_near_identity(self):
        F = torch.eye(3, dtype=torch.float64) + 1e-3 * torch.randn(N, 3, 3, generator=self.gen, dtype=torch.float64)
        self.check(F.float(), 1e-5)

    def test_strongly_deformed(self):
        s0 = log_uniform((N, 1), 0.5, 2.0, self.gen)  # ratios s0 / s2 up to 1e3, s1 + s2 >= 0.1 s0
        s = s0 * torch.cat(
            [
                torch.ones(N, 1, dtype=torch.float64),
                log_uniform((N, 1), 0.1, 1.0, self.gen),
                log_uniform((N, 1), 1e-3, 1.0, self.gen),
            ],
            1,
        )
        self.check(cells(s, self.gen), 1e-5)

    def test_inverted(self):
        s0 = log_uniform((N, 1), 1.0, 2.0, self.gen)  # s1 - s2 >= 0.1 s0: not a tie
        u = torch.rand(N, 2, generator=self.gen, dtype=torch.float64)
        s = s0 * torch.cat([torch.ones(N, 1, dtype=torch.float64), 0.5 + 0.4 * u[:, :1], 0.1 + 0.3 * u[:, 1:]], 1)
        self.check(cells(s, self.gen, inverted=True), 5e-4)

    def test_tie(self):
        """Two equal smallest singular values with det < 0: a proper rotation, optimal for F, close to that of F + eps R_ref."""
        scale = torch.tensor([1.0, 1.0, 3.0, 0.5] * 64, dtype=torch.float64)
        s = scale[:, None] * torch.tensor([1.0, 0.3, 0.3], dtype=torch.float64)
        F = cells(s, self.gen, inverted=True)
        F[::4] = torch.diag(torch.tensor([1.0, 0.3, -0.3])) * scale[::4, None, None].float()  # plain diagonal variants
        R_ref = rotations(F.shape[0], self.gen).float()
        _, S = reference(F)
        eps = TIE_EPS * S[:, 0].clamp_min(1.0)
        R_exp, S_eps = reference(F.double() + eps[:, None, None] * R_ref.double())
        # Where eps R_ref hardly splits s1 and s2 the reference itself is ill-conditioned (the pick within the family
        # of closest rotations rests on second-order terms), so the comparison is made where the tie is resolved.
        resolved = S_eps[:, 1] - S_eps[:, 2] > 0.2 * eps
        self.assertGreater(resolved.sum().item(), 0.7 * F.shape[0])
        for dev in DEVICES:
            with self.subTest(device=dev):
                R = frames(F.to(dev), R_ref.to(dev)).cpu().double()
                orth = (R.transpose(-1, -2) @ R - torch.eye(3, dtype=torch.float64)).abs().max().item()
                self.assertLess(orth, 1e-5)
                self.assertLess((torch.linalg.det(R) - 1.0).abs().max().item(), 1e-5)
                shortfall = S[:, 0] + S[:, 1] - S[:, 2] - torch.einsum("cij,cij->c", R, F.double())  # tr(R^T F) optimal
                self.assertLess(shortfall.max().item(), 1e-5)
                err = (R - R_exp).abs().amax((-1, -2))[resolved].max().item()
                self.assertLess(err, 5e-3, f"max error {err:.2e} on {dev}")

    def test_equivariance(self):
        s = log_uniform((2 * N, 3), 0.5, 2.0, self.gen)
        F = torch.cat([cells(s[:N], self.gen), cells(s[N:], self.gen, inverted=True)])
        _, S = reference(F)
        F = F[S[:, 1] - S[:, 2] > 0.1]  # keep clear of ties
        R_ref = rotations(F.shape[0], self.gen).float()
        Q = rotations(1, self.gen).float()
        for dev in DEVICES:
            with self.subTest(device=dev):
                F_d, R_ref_d, Q_d = F.to(dev), R_ref.to(dev), Q.to(dev)
                lhs = frames(Q_d @ F_d, Q_d @ R_ref_d)
                rhs = Q_d @ frames(F_d, R_ref_d)
                self.assertLess((lhs - rhs).abs().max().item(), 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "timing needs cuda")
    def test_timing(self):
        F = torch.randn(65536, 3, 3, device="cuda")
        R_ref = rotations(65536, self.gen).float().cuda()
        frames(F, R_ref)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            frames(F, R_ref)
        torch.cuda.synchronize()
        print(f"\nframes: {(time.perf_counter() - t0) / 20 * 1e3:.3f} ms per 65536 cells on cuda")
