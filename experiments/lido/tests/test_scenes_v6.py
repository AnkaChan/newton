# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""v6 scenes (Anka, 2026-10-02): the per-run shape library (determinism, ids, cell range), voxel bodies in the scene
generator (library draws, the occupied cell count, JSON round trip, realisation into a Batch with finite energies,
pinned corners on the clamped lattice face), the well (four vertical quads outside every body up to the well
height, the off switch), the box path reproducing the recorded v5 scenes bit for bit, and a SceneRunner epoch of
small voxel scenes on the CPU with the batched fusion."""

import dataclasses
import json
import math
import unittest

import numpy as np
import torch

from experiments.lido import contact, physics, shapes
from experiments.lido import hex as hx
from experiments.lido import scenes_v5 as S
from experiments.lido.augment import Augmenter
from experiments.lido.batch import Batch
from experiments.lido.config import TrainConfig
from experiments.lido.fusion import Fusion
from experiments.lido.grid import FACE_PINS, Grid, GridCache, face_pin, face_pin_mask, lattice_face_mask
from experiments.lido.network import Net
from experiments.lido.runner import SceneRunner, scene_batch
from experiments.lido.step import Step
from experiments.lido.tests.test_scene_runner import MASTER as RUNNER_MASTER
from experiments.lido.tests.test_scene_runner import scene_cfg
from experiments.lido.tests.test_scenes_v5 import CFG as V5_CFG
from experiments.lido.tests.test_scenes_v5 import REFERENCE, body_aabbs

MASTER = 73
LIBRARY = {"shape_library_size": 12, "shape_library_seed": 7, "shape_cell_range": (27, 300)}  # small: quick tests
CFG = TrainConfig(scene_cells=3000, body_shapes="voxel", **LIBRARY)  # the well on (the config default)


def library():
    return shapes.shape_library(
        LIBRARY["shape_library_size"], LIBRARY["shape_library_seed"], LIBRARY["shape_cell_range"]
    )


def well_slice(scene: S.SceneV5) -> slice:
    """Where the well quads sit in `static_faces`: after the regular faces, before the pinned contact faces."""
    n_well, n_pinned = scene.placement["well_faces"], scene.placement["pinned_contact_faces"]
    return slice(len(scene.static_faces) - n_well - n_pinned, len(scene.static_faces) - n_pinned)


def quad_normal(corners: np.ndarray) -> np.ndarray:
    n = np.cross(corners[2] - corners[0], corners[3] - corners[1])
    return n / np.linalg.norm(n)


class TestShapeLibrary(unittest.TestCase):
    def test_determinism_ids_and_cell_range(self):
        lib = library()
        self.assertIs(lib, library())  # built once per process
        self.assertEqual((len(lib), lib.key), (12, (12, 7, (27, 300))))
        fresh = shapes.ShapeLibrary(12, 7, (27, 300))
        lo, hi = LIBRARY["shape_cell_range"]
        carved = 0
        for i in range(12):
            occ = lib[i]
            self.assertEqual(occ.dtype, np.bool_)
            self.assertEqual(occ.ndim, 3)
            self.assertTrue(np.array_equal(occ, fresh[i]), i)  # the same seed gives the same shape with the same id
            cells = int(occ.sum())
            self.assertEqual(int(lib.cells[i]), cells)
            self.assertTrue(lo <= cells <= hi, (i, cells))
            self.assertEqual(int(shapes.largest_component(occ).sum()), cells)  # face-connected
            for axis in range(3):  # trimmed: every face of the bounding lattice is touched
                self.assertTrue(np.take(occ, 0, axis=axis).any() and np.take(occ, -1, axis=axis).any())
            carved += int(cells < math.prod(occ.shape))
        self.assertGreater(carved, 6)  # mostly not boxes
        other = shapes.ShapeLibrary(12, 8, (27, 300))
        self.assertFalse(
            all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(lib.shapes, other.shapes, strict=True))
        )
        self.assertIsNot(shapes.shape_library(12, 7, (27, 400)), lib)  # another cell range: another library
        with self.assertRaises(ValueError):
            shapes.ShapeLibrary(0, 7, (27, 300))

    def test_an_emptied_draw_is_a_failed_try(self):
        """The smoothing of `refine` can prune a thin coarse shape away entirely (the default library's seed 2026
        hit it at index 484 and `trim` failed on the empty occupancy, 2026-10-02): the try is drawn again."""
        self.assertEqual(shapes.trim(np.zeros((2, 3, 2), dtype=bool)).shape, (0, 0, 0))
        self.assertEqual(int(shapes.largest_component(np.zeros((2, 2, 2), dtype=bool)).sum()), 0)
        for seed in (1079, 2642):  # the first draw of these seeds empties out
            occ = shapes.sample_voxel_shape(np.random.default_rng(seed))
            self.assertTrue(27 <= int(occ.sum()) <= 1728)
            self.assertEqual(int(shapes.largest_component(occ).sum()), int(occ.sum()))

    def test_face_counts(self):
        bar = np.ones((2, 1, 1), dtype=bool)
        self.assertEqual(shapes.face_counts(bar).tolist(), [1, 1, 2, 2, 2, 2])
        box = np.ones((3, 4, 5), dtype=bool)
        self.assertEqual(shapes.face_counts(box).tolist(), [20, 20, 15, 15, 12, 12])
        # an L: brute force over the voxels and the six directions
        occ = np.zeros((3, 3, 2), dtype=bool)
        occ[:, 0, :] = True
        occ[0, :, :] = True
        counts = np.zeros(6, dtype=int)
        for v in np.argwhere(occ):
            for f, d in enumerate(shapes.FACE_NEIGHBOURS):
                q = v + d
                if not all(0 <= q[a] < occ.shape[a] for a in range(3)) or not occ[tuple(q)]:
                    counts[f] += 1
        self.assertEqual(shapes.face_counts(occ).tolist(), counts.tolist())
        self.assertEqual(int(shapes.face_counts(occ).max()), 6)  # the -x, +x and -y sides: 6 voxel faces each
        self.assertEqual(int(occ.sum()), 10)


class TestVoxelScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = [S.sample_scene(MASTER, seed, CFG) for seed in range(4)]
        cls.lib = library()
        cls.grids = GridCache("cpu")
        cls.aug = Augmenter("cpu")

    def test_bodies_are_library_shapes(self):
        lib = self.lib
        used = set()
        for scene in self.scenes:
            self.assertEqual(scene.shape_library, {"size": 12, "seed": 7, "cell_range": (27, 300)})
            for b in scene.bodies:
                self.assertIsInstance(b.shape, int)
                self.assertTrue(0 <= b.shape < len(lib))
                used.add(b.shape)
                self.assertEqual(b.cell_counts, tuple(int(v) for v in lib[b.shape].shape))  # the bounding lattice
                self.assertEqual(b.voxels, int(lib[b.shape].sum()))
                self.assertEqual(b.cells, b.voxels)  # the occupied count, not the box
                self.assertLess(b.cells, math.prod(b.cell_counts) + 1)
                self.assertEqual(len(b.cell_counts), 3)
                self.assertAlmostEqual(sum(v * v for v in b.quaternion), 1.0, places=12)
            self.assertEqual(scene.cells, sum(int(lib[b.shape].sum()) for b in scene.bodies))
            self.assertGreaterEqual(scene.cells, CFG.scene_cells)  # the budget counts occupied voxels
            self.assertLess(scene.cells - scene.bodies[-1].cells, CFG.scene_cells)
            summary = S.scene_summary(scene)
            self.assertEqual(summary["voxel_bodies"], len(scene.bodies))
            self.assertEqual(summary["well_faces"], 4)
            self.assertEqual(summary["cells"], scene.cells)
            json.dumps(summary)
        self.assertGreater(len(used), 8)  # uniform over the library
        self.assertTrue(any(b.pinned for sc in self.scenes for b in sc.bodies))
        self.assertTrue(any(b.resting for sc in self.scenes for b in sc.bodies))
        # deterministic and distinct across seeds
        self.assertEqual(dataclasses.asdict(S.sample_scene(MASTER, 0, CFG)), dataclasses.asdict(self.scenes[0]))
        self.assertNotEqual([b.shape for b in self.scenes[0].bodies], [b.shape for b in self.scenes[1].bodies])
        with self.assertRaises(ValueError):
            S.sample_scene(MASTER, 0, dataclasses.replace(CFG, body_shapes="sphere"))

    def test_json_round_trip(self):
        scene = self.scenes[0]
        back = S.SceneV5.from_dict(json.loads(json.dumps(dataclasses.asdict(scene))))
        self.assertEqual(back, scene)
        self.assertEqual(back.shape_library, scene.shape_library)
        self.assertEqual([b.shape for b in back.bodies], [b.shape for b in scene.bodies])
        # a record written before the shapes existed reads as boxes
        d = dataclasses.asdict(scene)
        d.pop("shape_library")
        for b in d["bodies"]:
            b.pop("shape")
            b.pop("voxels")
        boxes = S.SceneV5.from_dict(d)
        self.assertEqual(boxes.shape_library, {})
        self.assertTrue(all(b.shape is None and b.voxels is None for b in boxes.bodies))
        self.assertEqual(boxes.cells, sum(math.prod(b.cell_counts) for b in scene.bodies))

    def test_realise_batch_cells_and_energies(self):
        scene = self.scenes[0]
        gs, X, V, material, cs = S.realise(scene, self.grids, self.aug, "cpu", torch.float64)
        self.assertTrue(torch.isfinite(X).all() and torch.isfinite(V).all())
        for g, b in zip(gs, scene.bodies, strict=True):
            self.assertEqual(g.kind, "voxel")
            self.assertEqual(g.cell_counts, b.cell_counts)
            self.assertEqual(g.C, b.voxels)
            self.assertTrue(torch.equal(g.voxel_index, torch.from_numpy(self.lib[b.shape]).nonzero()))
            self.assertEqual(g.pinned.numel() > 0, b.pinned)
            self.assertEqual(g.pins, "mask" if b.pinned else "none")
        self.assertIs(
            gs[0], self.grids.get_voxel(scene.bodies[0].shape, self.lib[scene.bodies[0].shape], scene.bodies[0].pins)
        )
        batch = Batch.build(gs, "cpu", torch.float64)
        self.assertEqual(batch.C, scene.cells)  # the sum of the occupied voxels
        self.assertEqual(batch.C, sum(int(self.lib[b.shape].sum()) for b in scene.bodies))
        self.assertEqual(batch.N, sum(g.P for g in gs))
        self.assertEqual(material.count, len(gs))
        self.assertEqual(tuple(cs.faces.shape), (len(scene.static_faces), 4, 3))
        # the energies of the realised batch after scene_batch and the step's preparation
        b = scene_batch(scene, self.grids, self.aug, "cpu")
        self.assertEqual(b.C, scene.cells)
        step = Step(Net.from_config(scene_cfg()), Fusion(), self.aug)
        step.prepare(b, b.active, None)
        self.assertTrue(torch.isfinite(b.E).all() and torch.isfinite(b.gX).all())
        self.assertGreater(b.pairs.count, 0)
        # the energy pass at X against Y = X + V + g (the v5 smoke test's convention): finite and positive
        b.Y = b.X + b.V + b.material.g[b.corner_obj]
        E, gX = physics.energy_and_grad(b, b.X)
        self.assertEqual(tuple(E.shape), (len(gs),))
        self.assertTrue(torch.isfinite(E).all() and torch.isfinite(gX).all())
        self.assertTrue((E > 0).all())
        # the rigid pose turns about the lattice centre: the realised corners of a rigid scene are the lattice at
        # the pose, inside the bounding lattice's box (a shape's corners are a subset of the box's)
        rigid = dataclasses.replace(
            scene, bodies=[dataclasses.replace(x, perturbation_scale=0.0) for x in scene.bodies]
        )
        gs, X, _, _, _ = S.realise(rigid, self.grids, self.aug, "cpu", torch.float64)
        lo, hi = body_aabbs(rigid)
        off = 0
        for g, body, lo_b, hi_b in zip(gs, rigid.bodies, lo, hi, strict=True):
            Xb = X[off : off + g.P]
            off += g.P
            self.assertLess((Xb - S.rigid_pose(body, g.rest, scene.h)).abs().max().item(), 1e-12)
            self.assertTrue((Xb.min(0).values.numpy() * scene.h >= lo_b - 1e-9).all())
            self.assertTrue((Xb.max(0).values.numpy() * scene.h <= hi_b + 1e-9).all())
        # the elastic energy of the rigid pose is zero on voxel grids as well
        b = Batch.build(gs, "cpu", torch.float64)
        b.material, b.scene = S.realise(rigid, self.grids, self.aug, "cpu", torch.float64)[3:]
        b.X = X
        F = hx.gauss_deformation(X[b.cells], b.hc)
        b.C_prev = hx.mat3_tn(F, F)
        E_el, _ = physics.elastic_damping(b, X)
        self.assertLess(E_el.abs().max().item(), 1e-12)

    def test_pinned_voxel_bodies_on_the_lattice_face(self):
        scene = S.sample_scene(MASTER, 1, CFG, mix=S.SceneMix(1.0, 0.0))
        self.assertTrue(all(b.pinned for b in scene.bodies))
        self.assertEqual({b.pins for b in scene.bodies} - set(FACE_PINS), set())
        gs, X, V, _, _ = S.realise(scene, self.grids, self.aug, "cpu", torch.float64)
        off = 0
        faces_seen = set()
        for g, b in zip(gs, scene.bodies, strict=True):
            Xb, Vb = X[off : off + g.P], V[off : off + g.P]
            off += g.P
            axis, side = face_pin(b.pins)
            value = 0 if side == "min" else b.cell_counts[axis]
            on_face = g.corner_lattice[:, axis] == value
            self.assertTrue(torch.equal(g.pinned_mask, on_face), b.pins)  # exactly the present corners of the face
            self.assertGreaterEqual(g.pinned.numel(), 4)
            self.assertTrue((g.corner_lattice[g.free][:, axis] != value).all())
            self.assertTrue(torch.equal(g.fixed_flags, g.pinned_mask[g.cells]))
            pose = S.rigid_pose(b, g.rest, scene.h)
            self.assertLess((Xb[g.pinned] - pose[g.pinned]).abs().max().item(), 1e-12)
            self.assertEqual(Vb[g.pinned].abs().max().item(), 0.0)
            self.assertEqual(b.velocity, (0.0, 0.0, 0.0))
            faces_seen.add(b.pins)
        self.assertGreater(len(faces_seen), 3)
        # the face mask of a full lattice is the box grid's pin mask
        for cc in ((2, 3, 4), (3, 2, 2)):
            lattice = Grid.build(cc, "none").rest.to(torch.int64)
            for pins in FACE_PINS:
                mask = lattice_face_mask(cc, pins)
                self.assertEqual(tuple(mask.shape), tuple(n + 1 for n in cc))
                self.assertTrue(torch.equal(mask.reshape(-1), face_pin_mask(lattice, cc, pins)))
        # a cache serves one library: a hit from another occupancy raises
        other = GridCache("cpu")
        other.get_voxel(0, self.lib[0], "none")
        with self.assertRaises(ValueError):
            other.get_voxel(0, self.lib[1], "none")

    def test_well(self):
        h = CFG.cell_size
        margin = CFG.well_margin_cells * h
        for scene in self.scenes:
            self.assertEqual(scene.placement["well_faces"], 4)
            sl = well_slice(scene)
            well = [np.asarray(f) for f in scene.static_faces[sl]]
            self.assertEqual(len(well), 4)
            expected = S.well_faces(scene.placement["footprint"], margin, scene.plane_height, CFG.well_height_m)
            for c, e in zip(well, expected, strict=True):
                self.assertTrue(np.allclose(c, e))
            half = 0.5 * scene.placement["footprint"] + margin
            lo, hi = body_aabbs(scene)
            inward = []
            for c in well:
                # vertical: one horizontal coordinate constant at +-half, the normal horizontal
                const = [a for a in (0, 2) if np.ptp(c[:, a]) < 1e-12]
                self.assertEqual(len(const), 1)
                axis = const[0]
                self.assertAlmostEqual(abs(c[0, axis]), half, places=12)
                n = quad_normal(c)
                self.assertLess(abs(n[1]), 1e-12)
                inward.append(float(n[axis] * -np.sign(c[0, axis])))
                # from the ground to the well height, spanning the grown footprint
                self.assertAlmostEqual(c[:, 1].min(), scene.plane_height, places=12)
                self.assertAlmostEqual(c[:, 1].max(), CFG.well_height_m, places=12)
                other = 2 - axis
                self.assertAlmostEqual(c[:, other].min(), -half, places=12)
                self.assertAlmostEqual(c[:, other].max(), half, places=12)
                self.assertLess(np.abs(c[2] - (c[1] + c[3] - c[0])).max(), 1e-12)  # a planar rectangle
                # outside every body's box, at least the margin away
                self.assertFalse(S.quad_boxes_overlap(c, lo, hi).any())
                self.assertGreaterEqual(half - np.abs(np.concatenate([lo[:, axis], hi[:, axis]])).max(), margin - 1e-9)
            self.assertTrue(all(v > 0.999 for v in inward))  # the normals point into the well
            # the pinned contact faces stay last
            self.assertEqual(
                len(scene.placement["pinned_contact_face_owners"]), scene.placement["pinned_contact_faces"]
            )
        # realised: the walls are in the contact scene, and no wall pairs with a body at rest at step 0
        scene = self.scenes[0]
        gs, X, V, material, cs = S.realise(scene, self.grids, self.aug, "cpu", torch.float64)
        self.assertEqual(cs.faces.shape[0], len(scene.static_faces))
        sl = well_slice(scene)
        self.assertTrue(
            torch.allclose(cs.faces[sl] * scene.h, torch.tensor(scene.static_faces[sl], dtype=torch.float64))
        )
        b = Batch.build(gs, "cpu", torch.float64)
        b.material, b.scene, b.body_contact = material, cs, True
        b.X, b.V, b.X_prev, b.x = X.clone(), torch.zeros_like(V), X.clone(), X.clone()
        p = contact.detect(b, b.X, b.V)
        static = p.partner_body == contact.PARTNER_STATIC
        wall = static & (p.partner_face >= sl.start) & (p.partner_face < sl.stop)
        self.assertEqual(int(wall.sum()), 0)
        # the off switch: the same scene without the four quads and the record
        off = S.sample_scene(MASTER, 0, dataclasses.replace(CFG, world_well=False))
        self.assertNotIn("well_faces", off.placement)
        self.assertEqual(S.scene_summary(off)["well_faces"], 0)
        self.assertEqual(off.bodies, scene.bodies)
        self.assertEqual((off.contact, off.drift, off.shape_library), (scene.contact, scene.drift, scene.shape_library))
        self.assertEqual(off.static_faces, scene.static_faces[: sl.start] + scene.static_faces[sl.stop :])
        self.assertEqual(off.placement, {k: v for k, v in scene.placement.items() if k != "well_faces"})
        # the geometry helper on its own
        faces = S.well_faces(2.0, 0.5, 0.0, 1.5)
        self.assertEqual(len(faces), 4)
        for c, (axis, sign) in zip(faces, ((0, 1), (0, -1), (2, 1), (2, -1)), strict=True):
            self.assertTrue(np.allclose(c[:, axis], sign * 1.5))
            self.assertTrue(np.allclose(quad_normal(c), -sign * np.eye(3)[axis]))

    def test_box_shapes_reproduce_the_recorded_scenes(self):
        ref = json.loads(REFERENCE.read_text())
        still = dataclasses.replace(
            V5_CFG,
            body_shapes="box",
            world_well=False,
            resting_body_fraction=0.0,
            static_face_count_range=(0, 0),
            scene_wave_speed_min=0.0,
            pinned_contact_face_fraction=0.0,
            poissons_ratio_range=(0.2, 0.49),
        )  # the reference config of test_scenes_v5 with the v6 features off
        self.assertFalse(V5_CFG.world_well)
        checked = 0
        for key, d in ref["scenes"].items():
            master, seed, *validation = key.split("_")
            sc = S.sample_scene(int(master), int(seed), still, validation=bool(validation))
            self.assertEqual(sc.shape_library, {})
            self.assertTrue(all(b.shape is None and b.voxels is None for b in sc.bodies))
            self.assertEqual(sc.cells, sum(math.prod(b.cell_counts) for b in sc.bodies))
            stripped = {
                k: v
                for k, v in sc.placement.items()
                if k != "resting_on_bodies" and "static_face" not in k and "pinned_contact" not in k
            }
            sc = dataclasses.replace(sc, placement=stripped)
            self.assertEqual(dataclasses.asdict(sc), dataclasses.asdict(S.SceneV5.from_dict(d)), key)
            checked += 1
        self.assertGreater(checked, 0)
        # the box path with the full v5 configuration and the well: the bodies and the draws are the v5 ones
        v5 = S.sample_scene(MASTER, 0, V5_CFG)
        with_well = S.sample_scene(MASTER, 0, dataclasses.replace(V5_CFG, world_well=True))
        self.assertEqual(with_well.bodies, v5.bodies)
        self.assertEqual((with_well.contact, with_well.drift), (v5.contact, v5.drift))
        self.assertEqual(with_well.placement["well_faces"], 4)
        self.assertEqual(len(with_well.static_faces), len(v5.static_faces) + 4)
        self.assertEqual(S.scene_summary(v5)["voxel_bodies"], 0)
        # the box grids stay the box grids
        grids = GridCache("cpu")
        gs, *_ = S.realise(v5, grids, Augmenter("cpu"), "cpu")
        self.assertTrue(all(g.kind == "box" for g in gs))
        self.assertTrue(all(len(k) == 4 for k in grids.grids))


class TestSceneRunnerV6(unittest.TestCase):
    def test_epoch_of_voxel_scenes_on_the_cpu(self):
        cfg = scene_cfg(
            body_shapes="voxel",
            shape_library_size=8,
            shape_library_seed=7,
            shape_cell_range=(27, 200),
            world_well=True,
            static_face_count_range=(0, 2),
        )
        grids, aug = GridCache("cpu"), Augmenter("cpu")
        fusion = Fusion(batched=True)
        step = Step(Net.from_config(cfg), fusion, aug)
        runner = SceneRunner(cfg, step, aug, grids, 0, 1, "cpu", RUNNER_MASTER)
        runner.start_epoch(1)
        b = runner.batch
        self.assertTrue(all(g.kind == "voxel" for g in b.grids))
        self.assertEqual(b.C, runner.scene.cells)
        self.assertGreaterEqual(b.scene.faces.shape[0], 4)
        self.assertTrue(torch.isfinite(b.E).all())
        served = 0
        batched = set()
        for _ in range(runner.U):
            self.assertIsNotNone(runner.job)
            self.assertEqual(runner.batch.C, runner.scene.cells)
            out = step.query(runner.batch)
            served += 1
            cache = runner.batch.fusion_cache or {}
            batched.update(type(v).__name__ for v in cache.values())
            runner.commit(out)
        self.assertEqual(served, runner.U)
        self.assertEqual(runner.loaded_jobs, cfg.scene_count)
        self.assertEqual(runner.failures, [])
        self.assertIsNone(runner.job)
        summary = runner.epoch_summary()
        self.assertEqual(summary["scenes"], cfg.scene_count)
        for s in summary["scenes_served"]:
            self.assertEqual(s["voxel_bodies"], s["bodies"])
            self.assertEqual(s["well_faces"], 4)
            self.assertGreaterEqual(s["static_faces"], 4)
        # the batched Kron chain needs box grids: on voxel grids the fusion takes the per-group path (recorded by
        # the batch's cache as False); the epoch runs either way
        self.assertTrue(batched <= {"bool"}, batched)


if __name__ == "__main__":
    unittest.main()
