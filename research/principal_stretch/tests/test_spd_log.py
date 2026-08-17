# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the batched SPD matrix log/exp and the SO(3) log.

Forward values are checked from first principles (round trips, identity,
Taylor limits, known axis-angle constructions) and against
``scipy.linalg.logm`` when scipy is importable.  The analytic
Daleckii-Krein backward is checked with ``torch.autograd.gradcheck`` and,
at repeated eigenvalues -- where plain ``eigh`` autograd divides by the
zero eigen-gaps -- for finiteness and against the closed-form isotropic
Jacobian.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from research.principal_stretch.spd_log import so3_log_axial, spd_floor, sym_exp, sym_log

try:
    import scipy.linalg

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def random_rotation(n, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, 3, 3, dtype=dtype, generator=g)
    Q, _ = torch.linalg.qr(A)
    det = torch.linalg.det(Q)
    Q[:, :, 0] = Q[:, :, 0] * torch.sign(det)[:, None]
    return Q


def random_spd(n, dtype, seed, lam_lo=0.5, lam_hi=2.0):
    """Random SPD batch with eigenvalues uniform in ``[lam_lo, lam_hi]``."""
    g = torch.Generator().manual_seed(seed)
    lam = lam_lo + (lam_hi - lam_lo) * torch.rand(n, 3, dtype=dtype, generator=g)
    Q = random_rotation(n, dtype, seed + 1000)
    return Q @ torch.diag_embed(lam) @ Q.transpose(-1, -2)


def rotation_from_axis_angle(axis, angle, dtype=torch.float64):
    """Rodrigues rotation about (unnormalized) ``axis`` by ``angle`` [rad]."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=dtype)
    return torch.eye(3, dtype=dtype) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


class TestSpdLog(unittest.TestCase):
    def test_round_trip(self):
        S = random_spd(64, torch.float64, seed=1)
        Q = random_rotation(16, torch.float64, seed=2)
        lam = torch.tensor([2.55, 1.0, 0.15], dtype=torch.float64).expand(16, 3)  # 17:1 anisotropy
        S = torch.cat([S, Q @ torch.diag_embed(lam) @ Q.transpose(-1, -2)])

        self.assertLess((sym_exp(sym_log(S)) - S).abs().max().item(), 1e-12)
        H = sym_log(S)
        self.assertLess((sym_log(sym_exp(H)) - H).abs().max().item(), 1e-12)

        # arbitrary leading batch dims are preserved
        self.assertEqual(sym_log(S[:8].reshape(2, 4, 3, 3)).shape, (2, 4, 3, 3))

    def test_identity(self):
        eye = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
        self.assertLess(sym_log(eye).abs().max().item(), 1e-14)
        self.assertLess((sym_exp(torch.zeros(4, 3, 3, dtype=torch.float64)) - eye).abs().max().item(), 1e-14)

    def test_repeated_eigenvalues(self):
        # Exactly isotropic: log(s I) = log(s) I and the Jacobian is Id/s on
        # the symmetric part, so grad of sum(out) w.r.t. S is ones/s.
        s = 1.7
        S = (s * torch.eye(3, dtype=torch.float64)).expand(4, 3, 3).clone().requires_grad_(True)
        out = sym_log(S)
        ref = math.log(s) * torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
        self.assertLess((out - ref).abs().max().item(), 1e-14)
        (g,) = torch.autograd.grad(out.sum(), S)
        self.assertTrue(torch.isfinite(g).all())
        self.assertLess((g - torch.ones_like(g) / s).abs().max().item(), 1e-12)

        # Near-isotropic, eigenvalue gaps 0.5e-9 / 1e-9 -- at and below the
        # fp64 close-branch threshold, so all pairs take the f'(mid) branch:
        # log(S) = S - I to first order, and the gradient must stay finite
        # where eigh autograd would blow up.
        lam = torch.tensor([1.0, 1.0 + 0.5e-9, 1.0 + 1e-9], dtype=torch.float64).expand(8, 3)
        Q = random_rotation(8, torch.float64, seed=3)
        S = (Q @ torch.diag_embed(lam) @ Q.transpose(-1, -2)).requires_grad_(True)
        out = sym_log(S)
        eye = torch.eye(3, dtype=torch.float64)
        # log(S) = S - I to O(1e-18); 5e-15 allows eigh round-off, while a
        # wrong branch or eigenvector handling would show up at ~1e-9.
        self.assertLess((out - (S.detach() - eye)).abs().max().item(), 5e-15)
        (g,) = torch.autograd.grad(out.sum(), S)
        self.assertTrue(torch.isfinite(g).all())
        self.assertLess((g - torch.ones_like(g)).abs().max().item(), 1e-6)

    def test_gradcheck(self):
        S = random_spd(3, torch.float64, seed=4).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(sym_log, (S,), eps=1e-6, atol=1e-6))

        g = torch.Generator().manual_seed(5)
        A = torch.randn(3, 3, 3, dtype=torch.float64, generator=g)
        H = 0.5 * (A + A.transpose(-1, -2))
        H = (0.8 * H / H.flatten(-2).norm(dim=-1)[:, None, None]).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(sym_exp, (H,), eps=1e-6, atol=1e-6))

    @unittest.skipUnless(_HAVE_SCIPY, "scipy is not installed")
    def test_matches_scipy(self):
        S = random_spd(50, torch.float64, seed=6)
        out = sym_log(S).numpy()
        for i in range(50):
            ref = np.real(scipy.linalg.logm(S[i].numpy()))
            self.assertLess(np.abs(out[i] - ref).max(), 1e-12, f"matrix {i}")

    def test_so3_log(self):
        # Known axis-angle constructions across the admissible range,
        # including the theta < 1e-4 Taylor branch.
        for axis, angle in (
            ((1.0, 0.0, 0.0), 0.3),
            ((0.0, 1.0, 0.0), 1.5),
            ((1.0, -2.0, 0.5), 2.9),
            ((0.3, 0.4, -1.2), 1e-6),
        ):
            R = rotation_from_axis_angle(axis, angle)
            n = math.sqrt(sum(c * c for c in axis))
            ref = torch.tensor([c / n * angle for c in axis], dtype=torch.float64)
            w = so3_log_axial(R)
            self.assertLess((w - ref).abs().max().item(), 1e-10, f"axis={axis} angle={angle}")

        # Identity: zero vector and a finite gradient (guards the 0/0 branch
        # from leaking NaN through torch.where).
        R = torch.eye(3, dtype=torch.float64).requires_grad_(True)
        w = so3_log_axial(R)
        self.assertEqual(w.abs().max().item(), 0.0)
        (g,) = torch.autograd.grad(w.sum(), R)
        self.assertTrue(torch.isfinite(g).all())

        # Plain-autograd gradient at moderate angles, batched.
        R = torch.stack(
            [
                rotation_from_axis_angle((1.0, 0.5, -0.3), 1.2),
                rotation_from_axis_angle((0.0, 1.0, 4.0), 0.4),
            ]
        ).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(so3_log_axial, (R,), eps=1e-6, atol=1e-6))

        # Adjacent-tet relative rotations near pi are out of physical range.
        with self.assertRaises(ValueError):
            so3_log_axial(rotation_from_axis_angle((1.0, 0.0, 0.0), 3.05))

    def test_float32(self):
        S = random_spd(64, torch.float32, seed=8)
        self.assertLess((sym_exp(sym_log(S)) - S).abs().max().item(), 1e-5)
        self.assertLess((sym_log(S) - sym_log(S.double()).float()).abs().max().item(), 1e-5)

        R32 = rotation_from_axis_angle((1.0, 2.0, 0.3), 1.1, dtype=torch.float32)
        R64 = rotation_from_axis_angle((1.0, 2.0, 0.3), 1.1, dtype=torch.float64)
        self.assertLess((so3_log_axial(R32) - so3_log_axial(R64).float()).abs().max().item(), 1e-5)

    def test_so3_log_float32_identity_and_small_angle_have_finite_backward(self):
        rotations = torch.stack(
            (
                torch.eye(3, dtype=torch.float32),
                rotation_from_axis_angle((1.0, -2.0, 0.5), 1.0e-6, dtype=torch.float32),
            )
        ).requires_grad_(True)

        axial = so3_log_axial(rotations)
        self.assertTrue(torch.isfinite(axial).all())
        torch.testing.assert_close(axial[0], torch.zeros(3, dtype=torch.float32), rtol=0.0, atol=0.0)
        expected_axis = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
        expected = 1.0e-6 * expected_axis / torch.linalg.vector_norm(expected_axis)
        torch.testing.assert_close(axial[1], expected, rtol=1.0e-5, atol=1.0e-9)

        (gradient,) = torch.autograd.grad(axial.sum(), rotations)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_float32_backward_near_rest(self):
        # sym_exp backward at H ~= 0 with eigenvalue gaps 1e-8..1e-6: the
        # eigenvalues are densely representable, but exp(lam) ~= 1 is stored
        # at ulp 1.19e-7 in fp32, so a divided difference taken there returns
        # G = 0 (gap 1e-8: both exponentials round to exactly 1.0f) or spikes
        # to |G| ~ 1e2.  The dtype-dependent close branch (eps = 1e-4 in
        # fp32) must keep the gradient at the fp64 reference.
        for gap in (1e-8, 1e-7, 1e-6):
            lam = torch.tensor([0.0, gap, 2.0 * gap], dtype=torch.float64).expand(8, 3)
            Q = random_rotation(8, torch.float64, seed=9)
            H64 = Q @ torch.diag_embed(lam) @ Q.transpose(-1, -2)

            g = torch.Generator().manual_seed(10)
            grad_out = torch.randn(8, 3, 3, dtype=torch.float64, generator=g)

            h64 = H64.clone().requires_grad_(True)
            (g64,) = torch.autograd.grad(sym_exp(h64), h64, grad_outputs=grad_out)
            h32 = H64.float().clone().requires_grad_(True)
            (g32,) = torch.autograd.grad(sym_exp(h32), h32, grad_outputs=grad_out.float())

            self.assertTrue(torch.isfinite(g32).all(), f"gap={gap}")
            rel = ((g32.double() - g64).norm() / g64.norm()).item()
            self.assertLess(rel, 1e-3, f"gap={gap}: relative grad error {rel:.3e}")
            # no silently zeroed components, no spikes
            ratio = (g32.double().abs().max() / g64.abs().max()).item()
            self.assertGreater(ratio, 0.5, f"gap={gap}: max-norm ratio {ratio:.3e}")
            self.assertLess(ratio, 2.0, f"gap={gap}: max-norm ratio {ratio:.3e}")


class TestSpdFloor(unittest.TestCase):
    def test_negative_eigenvalue(self):
        lam = torch.tensor([-0.3, 0.8, 1.2], dtype=torch.float64).expand(8, 3)
        Q = random_rotation(8, torch.float64, seed=12)
        S = (Q @ torch.diag_embed(lam) @ Q.transpose(-1, -2)).requires_grad_(True)

        out = spd_floor(S)
        eigs = torch.linalg.eigvalsh(out.detach())
        self.assertLess((eigs[:, 0] - 0.05).abs().max().item(), 1e-12)
        self.assertLess((eigs[:, 2] - 1.2).abs().max().item(), 1e-12)
        (g,) = torch.autograd.grad(sym_log(out).sum(), S)
        self.assertTrue(torch.isfinite(g).all())

    def test_identity_away_from_floor(self):
        S = random_spd(32, torch.float64, seed=13)
        self.assertLess((spd_floor(S) - S).abs().max().item(), 1e-12)

    def test_exact_gradient_straddles_floor(self):
        floor = 0.05
        S = torch.diag(torch.tensor([0.049, 0.051, 0.2], dtype=torch.float64)).requires_grad_(True)
        grad_out = torch.zeros_like(S)
        grad_out[0, 1] = 1.0
        grad_out[1, 0] = 1.0
        (gradient,) = torch.autograd.grad(spd_floor(S, floor), S, grad_outputs=grad_out)

        # The off-diagonal Daleckii-Krein coefficient crosses the clamp kink:
        # (clamp(.051)-clamp(.049)) / (.051-.049) = 0.5.  Substituting the
        # derivative at the midpoint incorrectly gives either zero or one.
        self.assertAlmostEqual(gradient[0, 1].item(), 0.5, places=12)
        self.assertAlmostEqual(gradient[1, 0].item(), 0.5, places=12)
        self.assertTrue(torch.autograd.gradcheck(lambda value: spd_floor(value, floor), (S,), eps=1e-6, atol=1e-6))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLargeBatchCuda(unittest.TestCase):
    def test_large_batch_eigh(self):
        # cusolverDnXsyevBatched rejects flattened batches beyond ~31.6k fp64
        # 3x3 matrices (CUSOLVER_STATUS_INVALID_VALUE); sym_log must chunk.
        # 40k matrices with a multi-dim leading batch exercises the chunked
        # reshape path; values are checked against the (unchunked) CPU result.
        S = random_spd(40000, torch.float64, seed=15).reshape(5, 8000, 3, 3)
        H_cpu = sym_log(S)
        H_gpu = sym_log(S.cuda())
        self.assertEqual(H_gpu.shape, (5, 8000, 3, 3))
        self.assertLess((H_gpu.cpu() - H_cpu).abs().max().item(), 1e-10)

        # gradient flows through every chunk
        S_g = S.cuda().requires_grad_(True)
        (g,) = torch.autograd.grad(sym_log(S_g).sum(), S_g)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().amax(dim=(-1, -2)).min().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
