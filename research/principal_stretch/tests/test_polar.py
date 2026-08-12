# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the analytic 3x3 polar decomposition.

The forward must match the reflection-corrected SVD polar factor, and — the
point of the exercise — the backward must match finite differences across the
whole deformation range, not just at ``S = I`` where the previous
``torch_solver.polar_R`` happened to be exact.
"""

from __future__ import annotations

import unittest

import torch

from research.principal_stretch.polar import _svd_polar, polar_rotation


def random_rotation(n, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, 3, 3, dtype=dtype, generator=g)
    Q, _ = torch.linalg.qr(A)
    det = torch.linalg.det(Q)
    Q[:, :, 0] = Q[:, :, 0] * torch.sign(det)[:, None]
    return Q


def random_spd(n, dtype, seed, dev):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(n, 3, 3, dtype=dtype, generator=g)
    Ssym = 0.5 * (B + B.transpose(-1, -2))
    Ssym = Ssym / Ssym.flatten(-2).norm(dim=-1)[:, None, None]
    return torch.eye(3, dtype=dtype).expand(n, 3, 3) + dev * Ssym


class TestPolarRotation(unittest.TestCase):
    def test_matches_svd_polar(self):
        for dev in (0.0, 0.05, 0.2, 0.5, 0.75):
            R_true = random_rotation(64, torch.float64, seed=1)
            S = random_spd(64, torch.float64, seed=2, dev=dev)
            M = R_true @ S
            R = polar_rotation(M)
            self.assertLess((R - _svd_polar(M)).abs().max().item(), 1e-11, f"dev={dev}")

    def test_extreme_anisotropy(self):
        """Default iteration count must hold at deformations Phase 1 actually saw.

        Phase 1's forward run reached ``det(F) = 0.21``; this is harsher.
        """
        S = torch.diag(torch.tensor([2.5, 1.0, 0.15], dtype=torch.float64)).expand(64, 3, 3)
        M = random_rotation(64, torch.float64, seed=20) @ S
        self.assertLess((polar_rotation(M) - _svd_polar(M)).abs().max().item(), 1e-12)

    def test_orthogonal_and_proper(self):
        R_true = random_rotation(64, torch.float64, seed=3)
        M = R_true @ random_spd(64, torch.float64, seed=4, dev=0.6)
        R = polar_rotation(M)
        eye = torch.eye(3, dtype=torch.float64).expand_as(R)
        self.assertLess((R @ R.transpose(-1, -2) - eye).abs().max().item(), 1e-12)
        self.assertLess((torch.linalg.det(R) - 1.0).abs().max().item(), 1e-12)

    def test_recovers_symmetric_positive_stretch(self):
        R_true = random_rotation(64, torch.float64, seed=5)
        S_true = random_spd(64, torch.float64, seed=6, dev=0.4)
        M = R_true @ S_true
        R = polar_rotation(M)
        S = R.transpose(-1, -2) @ M
        self.assertLess((S - S_true).abs().max().item(), 1e-11)
        self.assertLess((S - S.transpose(-1, -2)).abs().max().item(), 1e-12)

    def test_gradient_matches_finite_differences(self):
        """The regression this module exists for.

        ``torch_solver.polar_R`` is exact only at S = I and drifts to 22%
        relative Jacobian error at 50% stretch.  This must stay near round-off
        across the whole range.
        """
        eps = 1e-6
        for dev in (0.0, 0.05, 0.2, 0.5):
            M = (random_rotation(1, torch.float64, seed=7) @ random_spd(1, torch.float64, seed=8, dev=dev)).clone()

            rows = []
            for i in range(3):
                for j in range(3):
                    Mv = M.clone().requires_grad_(True)
                    R = polar_rotation(Mv)
                    (g,) = torch.autograd.grad(R[0, i, j], Mv)
                    rows.append(g[0].reshape(-1))
            J_auto = torch.stack(rows)

            cols = []
            for a in range(3):
                for b in range(3):
                    Mp = M.clone()
                    Mp[0, a, b] += eps
                    Mm = M.clone()
                    Mm[0, a, b] -= eps
                    cols.append(((_svd_polar(Mp) - _svd_polar(Mm)) / (2 * eps))[0].reshape(-1))
            J_fd = torch.stack(cols).T

            rel = ((J_auto - J_fd).norm() / J_fd.norm()).item()
            self.assertLess(rel, 1e-6, f"dev={dev}: relative Jacobian error {rel:.3e}")

    def test_gradcheck(self):
        M = (random_rotation(4, torch.float64, seed=9) @ random_spd(4, torch.float64, seed=10, dev=0.3)).requires_grad_(
            True
        )
        self.assertTrue(torch.autograd.gradcheck(polar_rotation, (M,), eps=1e-6, atol=1e-6))

    def test_batch_shape_preserved(self):
        M = random_rotation(12, torch.float64, seed=11) @ random_spd(12, torch.float64, seed=12, dev=0.2)
        M2 = M.reshape(3, 4, 3, 3)
        self.assertEqual(polar_rotation(M2).shape, (3, 4, 3, 3))
        self.assertLess((polar_rotation(M2).reshape(12, 3, 3) - polar_rotation(M)).abs().max().item(), 1e-14)

    def test_float32(self):
        R_true = random_rotation(64, torch.float32, seed=13)
        M = R_true @ random_spd(64, torch.float32, seed=14, dev=0.3)
        R = polar_rotation(M)
        self.assertLess((R - _svd_polar(M)).abs().max().item(), 1e-5)

    def test_bad_branches_have_finite_stopped_gradient(self):
        good = random_rotation(1, torch.float64, seed=15) @ random_spd(1, torch.float64, seed=16, dev=0.2)
        inverted = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))[None]
        rank_one = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))[None]
        near_rank_one = torch.diag(torch.tensor([1.0e-14, 1.0e-14, 1.0], dtype=torch.float64))[None]
        M = torch.cat([good, inverted, rank_one, near_rank_one]).requires_grad_(True)
        generator = torch.Generator().manual_seed(17)
        grad_out = torch.randn(4, 3, 3, dtype=torch.float64, generator=generator)
        (gradient,) = torch.autograd.grad(polar_rotation(M), M, grad_outputs=grad_out)

        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[0].norm().item(), 0.0)
        self.assertEqual(gradient[1:].abs().max().item(), 0.0)

        good_only = good.clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(polar_rotation, (good_only,), eps=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
