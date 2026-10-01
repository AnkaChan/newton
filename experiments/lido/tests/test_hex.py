# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit-hex identities: partition of unity, mode maps against the shape-function gradients (derivation 8.4)."""

import unittest

import torch

from experiments.lido import hex as hx


class TestHex(unittest.TestCase):
    def test_partition_of_unity(self):
        self.assertLess(hx.GQ.sum(1).abs().max().item(), 1e-14)

    def test_gauss_gradients_from_modes(self):
        coef = hx._mode_coefficients(hx.XI_GAUSS)
        recon = torch.einsum("qva,vk->qka", coef, hx.P_MODES)
        self.assertLess((recon - hx.GQ).abs().max().item(), 1e-14)

    def test_modes_to_gauss_matches_direct(self):
        hc = hx.HexConstants.get("cpu", torch.float64)
        x = torch.randn(6, 8, 3, dtype=torch.float64)
        m = hx.modes(x, hc)
        self.assertLess((hx.gauss_deformation(x, hc) - hx.modes_to_gauss(m, hc)).abs().max().item(), 1e-13)
        Fc = torch.einsum("ckr,ka->cra", x, hx.shape_gradients(torch.zeros(1, 3))[0])
        self.assertLess((Fc - hx.center_deformation(m)).abs().max().item(), 1e-13)

    def test_rest_cell_identity(self):
        hc = hx.HexConstants.get("cpu", torch.float64)
        rest = hx.CORNER_OFFSETS.to(torch.float64)[None]
        F = hx.gauss_deformation(rest, hc)
        self.assertLess((F - torch.eye(3, dtype=torch.float64)).abs().max().item(), 1e-14)

    def test_gauss_to_modes_is_adjoint(self):
        hc = hx.HexConstants.get("cpu", torch.float64)
        dm = torch.randn(4, 7, 3, dtype=torch.float64)
        dF = torch.randn(4, 8, 3, 3, dtype=torch.float64)
        lhs = (hx.modes_to_gauss(dm, hc) * dF).sum()
        rhs = (dm * hx.gauss_to_modes(dF, hc)).sum()
        self.assertAlmostEqual(lhs.item(), rhs.item(), places=10)

    def test_face_corners_outward(self):
        pts = hx.CORNER_OFFSETS.to(torch.float64)
        for f in range(6):
            p = pts[hx.FACE_CORNERS[f]]
            n = torch.linalg.cross(p[1] - p[0], p[2] - p[1])
            self.assertGreater((n @ hx.FACE_NORMALS[f]).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
