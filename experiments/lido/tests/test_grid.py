# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from experiments.lido.grid import Grid, reference_rotation


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.g = Grid.build((2, 3, 4))

    def test_counts(self):
        g = self.g
        self.assertEqual((g.P, g.C), (60, 24))
        self.assertEqual(g.pinned.numel(), 12)
        self.assertEqual(g.exposed.sum().item(), 52)
        self.assertEqual(g.S, 52)

    def test_edges_sorted_csr_symmetric(self):
        g = self.g
        self.assertTrue((g.edges[1].diff() >= 0).all())
        counts = torch.bincount(g.edges[1], minlength=g.C)
        self.assertTrue((g.edge_offsets[1:] - g.edge_offsets[:-1] == counts).all())
        self.assertEqual((g.edges[0] == g.edges[1]).sum().item(), g.C)
        s = set(map(tuple, g.edges.t().tolist()))
        self.assertTrue(all((b, a) in s for a, b in s))
        self.assertTrue((g.edge_rest.abs() <= 1).all())

    def test_mass_lumping(self):
        self.assertAlmostEqual(self.g.mass.sum().item(), float(self.g.C))
        self.assertEqual(self.g.mass.min().item(), 0.125)

    def test_samples_on_faces(self):
        g = self.g
        for i in range(g.S):
            f = g.samples.face[i].item()
            pts = g.rest[g.samples.corners[i]]
            self.assertTrue((pts[:, f // 2] == pts[0, f // 2]).all())

    def test_reference_rotation(self):
        g = self.g
        R0 = reference_rotation(g.rest, g.ref_corners)
        self.assertLess((R0.t() @ R0 - torch.eye(3)).abs().max().item(), 1e-6)
        self.assertGreater(torch.det(R0).item(), 0.99)
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        q = q * torch.sign(torch.det(q))
        x = g.rest @ q.t() + 0.3
        self.assertLess((reference_rotation(x, g.ref_corners) - q @ R0).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
