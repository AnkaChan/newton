# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grid.from_voxels: equality with Grid.build on full occupancy, a hand-countable carved shape, pins, reference corners."""

import dataclasses
import unittest

import torch

from experiments.lido import hex as hx
from experiments.lido.batch import Batch
from experiments.lido.fusion import Fusion
from experiments.lido.grid import Grid


def carved(nx: int, ny: int, nz: int) -> torch.Tensor:
    """The carving rule of the solver tests: remove a central cylinder along z of radius nx / 3 and the corner block
    x > 0.7 nx & y > 0.7 ny (voxel centres)."""
    x, y, _ = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij")
    cx, cy = (x + 0.5) - nx / 2, (y + 0.5) - ny / 2
    occ = torch.ones(nx, ny, nz, dtype=torch.bool)
    occ &= ~((cx**2 + cy**2) <= (nx / 3) ** 2)
    occ &= ~((x + 0.5 > 0.7 * nx) & (y + 0.5 > 0.7 * ny))
    return occ


def holed_box() -> torch.Tensor:
    """6x6x8 box with a 2x2 through-hole along z at x, y in {2, 3} and the corner block x, y in {4, 5}, z >= 4 removed.

    By hand: cells 288 - 32 - 16 = 240; lattice corners 7*7*9 = 441 minus the hole axis (3, 3, z) (9 corners) minus
    the block interior x, y in {5, 6}, z in 5..8 (16 corners) = 416; z-min pins 49 - 1 = 48.
    """
    occ = torch.ones(6, 6, 8, dtype=torch.bool)
    occ[2:4, 2:4, :] = False
    occ[4:6, 4:6, 4:] = False
    return occ


def cell_at(g: Grid, ijk) -> int:
    hit = (g.voxel_index == torch.tensor(ijk)).all(1).nonzero()
    assert hit.numel() == 1, ijk
    return int(hit)


class TestVoxelGridEqualsBuild(unittest.TestCase):
    def test_full_occupancy_equals_build_field_by_field(self):
        a = Grid.build((2, 3, 4))
        b = Grid.from_voxels(torch.ones(2, 3, 4, dtype=torch.bool))
        for f in dataclasses.fields(Grid):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if f.name == "key":
                self.assertEqual(va, (2, 3, 4, "zmin_face"))
                self.assertEqual((vb[0], vb[2]), ("voxel", "zmin_face"))
            elif f.name == "samples":
                for n in ("cell", "face", "corners"):
                    self.assertTrue(torch.equal(getattr(va, n), getattr(vb, n)), n)
            elif isinstance(va, torch.Tensor):
                self.assertEqual(va.dtype, vb.dtype, f.name)
                self.assertTrue(torch.equal(va, vb), f.name)
            else:
                self.assertEqual(va, vb, f.name)
        self.assertEqual((a.kind, b.kind), ("box", "voxel"))
        self.assertEqual((a.pins, b.pins), ("zmin_face", "zmin_face"))

    def test_key_is_a_content_hash(self):
        occ = holed_box()
        self.assertEqual(Grid.from_voxels(occ).key, Grid.from_voxels(occ.clone()).key)
        other = occ.clone()
        other[0, 0, 7] = False
        self.assertNotEqual(Grid.from_voxels(occ).key, Grid.from_voxels(other).key)
        self.assertNotEqual(Grid.from_voxels(occ).key, Grid.from_voxels(occ, None).key)


class TestCarvedShape(unittest.TestCase):
    def setUp(self):
        self.occ = holed_box()
        self.g = Grid.from_voxels(self.occ)

    def test_counts_by_hand(self):
        g = self.g
        self.assertEqual((g.C, g.P), (240, 416))
        self.assertEqual((g.pinned.numel(), g.Pf), (48, 368))
        self.assertEqual(g.cell_counts, (6, 6, 8))
        self.assertTrue(torch.equal(g.voxel_index, self.occ.nonzero()))
        # corners in lattice order, every corner touched by a cell, the hole axis and the block interior absent
        lin = (g.corner_lattice[:, 0] * 7 + g.corner_lattice[:, 1]) * 9 + g.corner_lattice[:, 2]
        self.assertTrue((lin.diff() > 0).all())
        self.assertTrue((g.mass > 0).all())
        self.assertAlmostEqual(g.mass.sum().item(), 240.0)
        self.assertEqual(g.mass.min().item(), 0.125)
        present = set(map(tuple, g.corner_lattice.tolist()))
        for z in range(9):
            self.assertNotIn((3, 3, z), present)
        for x in (5, 6):
            for y in (5, 6):
                for z in range(5, 9):
                    self.assertNotIn((x, y, z), present)
                self.assertIn((x, y, 4), present)
        self.assertTrue(torch.equal(g.rest, g.corner_lattice.to(torch.float32)))
        # cells reference their own lattice corners
        corners = g.corner_lattice[g.cells]  # [C,8,3]
        self.assertTrue(torch.equal(corners, g.voxel_index[:, None, :] + hx.CORNER_OFFSETS[None]))

    def test_edges_join_present_cells_sorted_csr(self):
        g = self.g
        self.assertTrue(((g.edges >= 0) & (g.edges < g.C)).all())
        d = g.voxel_index[g.edges[0]] - g.voxel_index[g.edges[1]]
        self.assertTrue((d.abs().amax(1) <= 1).all())
        self.assertTrue(torch.equal(g.edge_rest, d.to(torch.float32)))
        self.assertTrue((g.edges[1].diff() >= 0).all())
        counts = torch.bincount(g.edges[1], minlength=g.C)
        self.assertTrue(torch.equal(g.edge_offsets[1:] - g.edge_offsets[:-1], counts))
        self.assertEqual(int(g.edge_offsets[-1]), g.E)
        self.assertEqual(int((g.edges[0] == g.edges[1]).sum()), g.C)
        s = set(map(tuple, g.edges.t().tolist()))
        self.assertTrue(all((b, a) in s for a, b in s))
        # every interior cell of the 6x6 section reaches the hole: (1,1,1) misses (2,2,0..2), (2,1,3) misses (2|3,2,2..4)
        self.assertEqual(int(counts[cell_at(g, (1, 1, 1))]), 27 - 3)
        self.assertEqual(int(counts[cell_at(g, (2, 1, 3))]), 27 - 6)
        self.assertEqual(int(counts[cell_at(g, (0, 0, 0))]), 8)

    def test_exposed_faces_hand_picked(self):
        g = self.g
        expect = {
            (0, 0, 0): [0, 2, 4],  # -x, -y, -z
            (1, 1, 1): [],
            (2, 1, 3): [3],  # +y neighbour (2,2,3) is in the hole
            (4, 4, 3): [5],  # +z neighbour (4,4,4) is in the removed block
            (3, 5, 7): [1, 3, 5],  # +x neighbour (4,5,7) removed, +y and +z outside
            (5, 5, 3): [1, 3, 5],
        }
        for ijk, faces in expect.items():
            got = g.exposed[cell_at(g, ijk)].nonzero().flatten().tolist()
            self.assertEqual(got, faces, ijk)
        self.assertEqual(int(g.exposed.sum()), g.S)

    def test_samples_on_exposed_faces(self):
        g = self.g
        for i in range(g.S):
            c, f = int(g.samples.cell[i]), int(g.samples.face[i])
            self.assertTrue(bool(g.exposed[c, f]))
            pts = g.corner_lattice[g.samples.corners[i]]
            axis = f // 2
            self.assertTrue((pts[:, axis] == g.voxel_index[c, axis] + (f % 2)).all())
            self.assertTrue(torch.equal(g.samples.corners[i], g.cells[c][hx.FACE_CORNERS[f]]))
        order = g.samples.cell * 6 + g.samples.face
        self.assertTrue((order.diff() > 0).all())

    def test_fixed_flags(self):
        g = self.g
        self.assertTrue(torch.equal(g.fixed_flags, g.pinned_mask[g.cells]))
        self.assertTrue((g.corner_lattice[g.pinned][:, 2] == 0).all())
        self.assertTrue((g.corner_lattice[g.free][:, 2] > 0).all())


class TestPins(unittest.TestCase):
    def test_mask_restricted_to_present_corners(self):
        occ = holed_box()
        mask = torch.zeros(7, 7, 9, dtype=torch.bool)
        mask[0] = True  # the x = 0 plane: 63 corners, all present
        mask[3, 3, :] = True  # the hole axis: absent corners, must not count
        g = Grid.from_voxels(occ, mask)
        self.assertEqual(g.pinned.numel(), 63)
        self.assertTrue((g.corner_lattice[g.pinned][:, 0] == 0).all())
        self.assertEqual(g.pins, "mask")
        self.assertEqual(g.key[2], "mask")
        self.assertTrue(torch.equal(g.fixed_flags, g.pinned_mask[g.cells]))
        other = mask.clone()
        other[6, 0, 8] = True
        self.assertNotEqual(Grid.from_voxels(occ, other).key, g.key)
        with self.assertRaises(ValueError):
            Grid.from_voxels(occ, torch.zeros(6, 6, 8, dtype=torch.bool))

    def test_none_and_missing_base(self):
        occ = holed_box()
        g = Grid.from_voxels(occ, None)
        self.assertEqual((g.pinned.numel(), g.Pf, g.pins), (0, g.P, "none"))
        self.assertEqual(Grid.from_voxels(occ, "none").key, g.key)
        floating = occ.clone()
        floating[:, :, 0] = False
        with self.assertRaises(ValueError):
            Grid.from_voxels(floating, "zmin_face")
        self.assertEqual(Grid.from_voxels(floating, None).C, 240 - 36 + 4)

    def test_device_argument(self):
        g = Grid.from_voxels(holed_box(), "zmin_face", "cpu")
        self.assertEqual(g.device, torch.device("cpu"))
        self.assertEqual(g.samples.cell.device, torch.device("cpu"))


class TestReferenceCorners(unittest.TestCase):
    def test_zmin_rule_matches_build(self):
        g = Grid.from_voxels(holed_box())
        ref = g.corner_lattice[g.ref_corners].tolist()
        self.assertEqual(ref, [[0, 0, 0], [6, 6, 0], [0, 6, 0]])

    def test_general_rule_on_a_mask(self):
        occ = holed_box()
        gen = torch.Generator().manual_seed(3)
        mask = torch.rand(7, 7, 9, generator=gen) < 0.1
        g = Grid.from_voxels(occ, mask)
        lat = g.corner_lattice.to(torch.float64)
        pinned = g.pinned
        p0, p1, p2 = g.ref_corners.tolist()
        self.assertEqual(p0, int(pinned[0]))
        d = lat[pinned] - lat[p0]
        self.assertEqual(p1, int(pinned[d.norm(dim=1).argmax()]))
        cross = torch.linalg.cross((lat[p1] - lat[p0]).expand_as(d), d).norm(dim=1)
        self.assertEqual(p2, int(pinned[cross.argmax()]))
        self.assertGreater(float(cross.max()), 0)

    def test_collinear_pins_fall_back_with_a_note(self):
        occ = holed_box()
        mask = torch.zeros(7, 7, 9, dtype=torch.bool)
        mask[0, 0, :] = True  # one lattice line
        with self.assertWarns(UserWarning):
            g = Grid.from_voxels(occ, mask)
        self.assertEqual(g.pinned.numel(), 9)
        self.assertEqual(g.corner_lattice[g.ref_corners].tolist(), [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mask2 = torch.zeros(7, 7, 9, dtype=torch.bool)
        mask2[0, 0, 0] = mask2[6, 6, 8] = True
        with self.assertWarns(UserWarning):
            Grid.from_voxels(occ, mask2)


class TestBatchWithVoxelGrids(unittest.TestCase):
    def test_batch_groups_by_key_and_fusion_runs(self):
        gv = Grid.from_voxels(holed_box())
        gb = Grid.build((2, 2, 3))
        batch = Batch.build([gv, gb, gv], "cpu", torch.float64)
        self.assertEqual((batch.O, batch.N, batch.C), (3, 2 * gv.P + gb.P, 2 * gv.C + gb.C))
        self.assertEqual(len(batch.groups), 2)
        self.assertEqual([grp.objects.tolist() for grp in batch.groups], [[0, 2], [1]])
        self.assertTrue(
            torch.equal(batch.edges[:, batch.edge_off[1] : batch.edge_off[2]], gb.edges + batch.cell_off[1])
        )
        hc = hx.HexConstants.get("cpu", torch.float64)
        dm = torch.randn(batch.C, 7, 3, dtype=torch.float64)
        d = Fusion().fuse(batch, hx.modes_to_gauss(dm, hc))
        self.assertEqual(tuple(d.shape), (batch.N, 3))
        self.assertTrue(torch.isfinite(d).all())
        self.assertTrue((d[batch.pinned] == 0).all())
        # the two copies of the voxel grid solve independently: object 2 matches a single-object batch
        single = Batch.build([gv], "cpu", torch.float64)
        cs, ce = int(batch.cell_off[2]), int(batch.cell_off[3])
        d1 = Fusion().fuse(single, hx.modes_to_gauss(dm[cs:ce], hc))
        self.assertLess((d[int(batch.corner_off[2]) : int(batch.corner_off[3])] - d1).abs().max().item(), 1e-10)


if __name__ == "__main__":
    unittest.main()
