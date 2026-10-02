# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""v5 scene generator (design spec section 11): seed streams, cell budget, placement rules, drift and materials,
pinned bodies, resting bodies, JSON round trip, realisation in normalised units and an energy pass on the realised
batch."""

import collections
import dataclasses
import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.lido import contact, physics, scenes
from experiments.lido import hex as hx
from experiments.lido import scenes_v5 as S
from experiments.lido.augment import Augmenter
from experiments.lido.batch import Batch
from experiments.lido.config import TrainConfig
from experiments.lido.grid import FACE_PINS, GridCache

MASTER = 73
CFG = TrainConfig(scene_cells=6000)  # about 14 bodies: every rule is exercised, the tests stay quick
REFERENCE = Path(__file__).parent / "reference" / "scenes_v5_6000_cells.json"  # recorded before the resting start


def box_top(x: np.ndarray, z: np.ndarray, body: S.BodySpec, h: float, inflate: float = 0.0) -> np.ndarray:
    """Height (m) of the rigid box's upper surface (inflated by `inflate` m on every side) over the vertical lines
    through (x, z) [n] by an inside test and bisection (independent of the generator's slab clipping); NaN where the
    line misses the box."""
    R = S.rotation_matrix(body.quaternion)
    c = np.asarray(body.position)
    half = 0.5 * h * np.asarray(body.cell_counts, dtype=float) + inflate
    reach = float(np.abs(R) @ half @ np.ones(3))
    ys = np.linspace(c[1] - reach, c[1] + reach, 801)
    pts = np.stack(
        [
            np.broadcast_to(x[:, None], (x.size, ys.size)),
            np.broadcast_to(ys[None], (x.size, ys.size)),
            np.broadcast_to(z[:, None], (x.size, ys.size)),
        ],
        -1,
    )
    local = (pts - c) @ R  # R^T (p - c) per point
    inside = (np.abs(local) <= half + 1e-12).all(-1)  # [n, ys]
    hit = inside.any(1)
    top = np.full(x.size, np.nan)
    for i in np.nonzero(hit)[0]:
        j = int(np.nonzero(inside[i])[0].max())  # the highest inside ladder point: bisect to the surface above it
        lo, hi = ys[j], ys[min(j + 1, ys.size - 1)]
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if (np.abs(R.T @ (np.array([x[i], mid, z[i]]) - c)) <= half + 1e-12).all():
                lo = mid
            else:
                hi = mid
        top[i] = lo
    return top


def bottom_quad(body: S.BodySpec, h: float) -> np.ndarray:
    """The (x, z) corners [4,2] of a flat body's bottom face, in order around the face."""
    corners = S.box_corners(body.position, S.rotation_matrix(body.quaternion), 0.5 * h * np.asarray(body.cell_counts))
    bottom = corners[np.argsort(corners[:, 1])[:4]][:, [0, 2]]
    centre = bottom.mean(0)
    return bottom[np.argsort(np.arctan2(bottom[:, 1] - centre[1], bottom[:, 0] - centre[0]))]


def inside_quad(p: np.ndarray, quad: np.ndarray) -> bool:
    """Point [2] inside the convex quad [4,2] (cross products with every edge share a sign)."""
    signs = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        signs.append((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    return all(v >= -1e-12 for v in signs) or all(v <= 1e-12 for v in signs)


def footprint_samples(body: S.BodySpec, h: float, n: int = 40, edge: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """(x, z) samples over a flat body's bottom face: a grid, the corners and dense points along the edges."""
    bottom = bottom_quad(body, h)
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    grid = (1 - u[..., None]) * ((1 - v[..., None]) * bottom[0] + v[..., None] * bottom[1]) + u[..., None] * (
        (1 - v[..., None]) * bottom[3] + v[..., None] * bottom[2]
    )
    t = np.linspace(0, 1, edge)[:, None]
    edges = np.concatenate([(1 - t) * bottom[k] + t * bottom[(k + 1) % 4] for k in range(4)])
    pts = np.concatenate([grid.reshape(-1, 2), edges, bottom])
    return pts[:, 0], pts[:, 1]


def resting_support_distance(scene: S.SceneV5, a: int, b: int, inflate: float = 0.0) -> float:
    """Smallest vertical distance (m) from resting body a's bottom plane to pinned body b's upper surface (box
    inflated by `inflate`) over a's footprint; inf when b is not under the footprint."""
    h = scene.h
    A, B = scene.bodies[a], scene.bodies[b]
    x, z = footprint_samples(A, h)
    # the support point may be a corner of B inside the footprint: sample just inside those corners as well
    cb = S.box_corners(B.position, S.rotation_matrix(B.quaternion), 0.5 * h * np.asarray(B.cell_counts) + inflate)
    cb = cb + 1e-7 * (np.asarray(B.position)[None] - cb)
    quad = bottom_quad(A, h)
    cb = np.array([c for c in cb if inside_quad(c[[0, 2]], quad)]).reshape(-1, 3)
    x, z = np.concatenate([x, cb[:, 0]]), np.concatenate([z, cb[:, 2]])
    top = box_top(x, z, B, h, inflate)
    if not np.isfinite(top).any():
        return math.inf
    bottom_y = A.position[1] - S.half_extents(A.cell_counts, A.quaternion, h)[1]
    return float(np.nanmin(bottom_y - top))


def body_aabbs(scene: S.SceneV5) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding boxes [n,3] (m) of the rigidly placed bodies."""
    ext = np.stack([S.half_extents(b.cell_counts, b.quaternion, scene.h) for b in scene.bodies])
    pos = np.array([b.position for b in scene.bodies])
    return pos - ext, pos + ext


def pairwise_separation(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Largest axis gap between the boxes of every pair [n,n] (negative when the boxes overlap); inf on the diagonal."""
    n = lo.shape[0]
    sep = np.maximum(lo[:, None, :] - hi[None, :, :], lo[None, :, :] - hi[:, None, :]).max(-1)
    sep[np.arange(n), np.arange(n)] = np.inf
    return sep


def rigid(scene: S.SceneV5) -> S.SceneV5:
    """The scene with the deformation fields switched off."""
    return dataclasses.replace(scene, bodies=[dataclasses.replace(b, perturbation_scale=0.0) for b in scene.bodies])


def support_pairs(scene: S.SceneV5) -> np.ndarray:
    """[n,n] bool (symmetric): pairs of a resting body and a pinned body placed before it whose (x, z) bounding boxes
    overlap (the only pairs the grown-box rule does not separate: the resting body may sit on the pinned one)."""
    lo, hi = body_aabbs(scene)
    xz = ((lo[:, None, [0, 2]] <= hi[None, :, [0, 2]]) & (lo[None, :, [0, 2]] <= hi[:, None, [0, 2]])).all(-1)
    resting = np.array([b.resting for b in scene.bodies])
    pinned = np.array([b.pinned for b in scene.bodies])
    idx = np.arange(len(scene.bodies))
    before = idx[None, :] < idx[:, None]  # [a, j]: j placed before a
    on = resting[:, None] & pinned[None, :] & before
    return xz & (on | on.T)


class TestSampleScene(unittest.TestCase):
    def setUp(self):
        self.scene = S.sample_scene(MASTER, 0, CFG)

    def test_deterministic_and_distinct_across_seeds(self):
        again = S.sample_scene(MASTER, 0, CFG)
        self.assertEqual(dataclasses.asdict(again), dataclasses.asdict(self.scene))
        self.assertNotEqual(dataclasses.asdict(S.sample_scene(MASTER, 1, CFG)), dataclasses.asdict(self.scene))
        self.assertNotEqual(dataclasses.asdict(S.sample_scene(MASTER + 1, 0, CFG)), dataclasses.asdict(self.scene))
        held = S.held_out_scene(MASTER, 0, CFG)
        self.assertTrue(held.validation)
        self.assertFalse(self.scene.validation)
        self.assertNotEqual([b.cell_counts for b in held.bodies], [b.cell_counts for b in self.scene.bodies])
        self.assertEqual(dataclasses.asdict(held), dataclasses.asdict(S.sample_scene(MASTER, 0, CFG, validation=True)))

    def test_cell_budget_and_sides(self):
        lo, hi = CFG.body_sides
        for seed in range(4):
            scene = S.sample_scene(MASTER, seed, CFG)
            total = scene.cells
            self.assertGreaterEqual(total, CFG.scene_cells)
            self.assertLess(total - scene.bodies[-1].cells, CFG.scene_cells)  # the last body alone exceeds it
            for b in scene.bodies:
                self.assertEqual(len(b.cell_counts), 3)
                self.assertTrue(all(isinstance(v, int) and lo <= v <= hi for v in b.cell_counts))
                self.assertEqual(b.cells, math.prod(b.cell_counts))
        self.assertGreater(len(self.scene.bodies), 5)

    def test_placement(self):
        h = self.scene.h
        for seed in range(4):
            scene = S.sample_scene(MASTER, seed, CFG)
            lo, hi = body_aabbs(scene)
            sep = pairwise_separation(lo, hi)
            free_pairs = ~support_pairs(scene)
            self.assertGreaterEqual(sep[free_pairs].min(), CFG.placement_gap_cells[0] * h - 1e-9)  # at least the gap
            lowest = lo[:, 1] / h  # cells above the plane at y = 0
            self.assertEqual(scene.plane_height, 0.0)
            resting = np.array([b.resting for b in scene.bodies])
            if not resting[0]:
                self.assertTrue(S.FIRST_BODY_CELLS[0] - 1e-9 <= lowest[0] <= S.FIRST_BODY_CELLS[1] + 1e-9)
            self.assertTrue((lowest[1:][~resting[1:]] >= S.GROUND_CELLS - 1e-9).all())
            self.assertTrue((lowest[resting] >= S.RESTING_GAP_CELLS - 1e-9).all())  # on the ground or higher
            self.assertTrue((lowest[~resting] <= CFG.placement_height / h + 1e-9).all())
            pinned = np.array([b.pinned for b in scene.bodies])
            top = max(hi[pinned, 1].max() / h if pinned.any() else 0.0, 0.0)  # a resting body may sit on a pinned one
            clearance = max(S.field_clearance(b.perturbation_scale, b.strength, h) / h for b in scene.bodies)
            self.assertTrue((lowest[resting] <= top + S.RESTING_GAP_CELLS + clearance + 1e-9).all())
            for b in scene.bodies:
                self.assertAlmostEqual(sum(v * v for v in b.quaternion), 1.0, places=12)
                self.assertTrue(all(isinstance(v, float) for v in (*b.position, *b.quaternion, *b.velocity)))
            footprint = scene.placement["footprint"]
            self.assertTrue(
                (lo[:, [0, 2]] >= -footprint / 2 - 1e-9).all() and (hi[:, [0, 2]] <= footprint / 2 + 1e-9).all()
            )
            self.assertEqual(scene.placement["draws"] * scene.placement["acceptance_rate"], len(scene.bodies))

    def test_drift_and_velocities(self):
        lo, hi = CFG.drift_speed_range
        for seed in range(6):
            scene = S.sample_scene(MASTER, seed, CFG)
            drift = np.asarray(scene.drift)
            speed = np.linalg.norm(drift)
            self.assertTrue(lo <= speed <= hi)
            noise = np.array([np.linalg.norm(np.asarray(b.velocity) - drift) for b in scene.bodies])
            self.assertTrue((noise <= speed + 1e-9).all())
            self.assertGreater(noise.max(), 0.0)
        # the drift direction is horizontal-biased: over many scenes |d_y| is well below the isotropic mean 1/2
        dy = [
            abs(S.sample_scene(MASTER, s, CFG).drift[1]) / np.linalg.norm(S.sample_scene(MASTER, s, CFG).drift)
            for s in range(24)
        ]
        self.assertLess(float(np.mean(dy)), 0.35)

    def test_materials_and_contact(self):
        c = self.scene.contact
        for b in self.scene.bodies:
            m = b.material
            self.assertTrue(CFG.youngs_modulus_range[0] <= m["E"] <= CFG.youngs_modulus_range[1])
            self.assertTrue(CFG.poissons_ratio_range[0] <= m["nu"] <= CFG.poissons_ratio_range[1])
            self.assertTrue(CFG.density_range[0] <= m["rho"] <= CFG.density_range[1])
            self.assertTrue(CFG.damping_range[0] <= m["eta"] <= CFG.damping_range[1])
            self.assertTrue(CFG.strength_range[0] <= b.strength <= CFG.strength_range[1])
            self.assertTrue(CFG.velocity_dt_range[0] <= b.velocity_dt <= CFG.velocity_dt_range[1])
            self.assertTrue(CFG.perturbation_scale_range[0] <= b.perturbation_scale <= CFG.perturbation_scale_range[1])
            self.assertTrue(0 <= b.seed < 2**63)
        self.assertTrue(CFG.contact_kappa_range[0] <= c["kappa_drawn"] <= CFG.contact_kappa_range[1])
        self.assertEqual(c["kappa"], max(c["kappa_drawn"], c["kappa_floor"]))
        self.assertEqual(c["floor_bound"], c["kappa_floor"] > c["kappa_drawn"])
        self.assertTrue(CFG.contact_beta_range[0] <= c["beta"] <= CFG.contact_beta_range[1])
        self.assertTrue(CFG.contact_mu_range[0] <= c["mu_f"] <= CFG.contact_mu_range[1])
        self.assertEqual(self.scene.gravity, tuple(CFG.gravity))
        # the floor is the contact note's rule maximised over the bodies
        h, g = self.scene.h, 9.81
        floors = []
        for b in self.scene.bodies:
            nx, ny, nz = b.cell_counts
            ke_floor = scenes.contact_stiffness_floor(
                b.material["rho"], g, h, b.cells, max(ny * nz, nx * nz, nx * ny), CFG.contact_static_penetration_max
            )
            floors.append(ke_floor / (b.material["E"] * h))
        self.assertAlmostEqual(c["kappa_floor"], max(floors), places=9)
        heavy = TrainConfig(
            scene_cells=6000,
            contact_kappa_range=(10.0, 10.1),
            density_range=(1e4, 1e4),
            youngs_modulus_range=(1e3, 1e3),
        )
        bound = S.sample_scene(MASTER, 0, heavy)
        self.assertTrue(bound.contact["floor_bound"])
        self.assertGreater(bound.contact["kappa"], 10.1)

    def test_pinned_bodies(self):
        # the draw: fraction 0.25 per body from the pin stream, the clamped face uniform over the six
        bodies = [b for seed in range(20) for b in S.sample_scene(MASTER, seed, CFG).bodies]
        self.assertGreaterEqual(len(bodies), 200)
        pinned = [b for b in bodies if b.pinned]
        self.assertAlmostEqual(len(pinned) / len(bodies), CFG.pinned_body_fraction, delta=0.1)
        faces = collections.Counter(b.pins for b in pinned)
        self.assertEqual(set(faces), set(FACE_PINS))
        self.assertGreater(min(faces.values()), 0)
        for b in bodies:
            self.assertEqual(b.pinned, b.pins != "none")
            if b.pinned:
                self.assertEqual(b.velocity, (0.0, 0.0, 0.0))  # held at its pose: no rigid velocity
            elif not b.resting:
                self.assertGreater(float(np.linalg.norm(b.velocity)), 0.0)
        # the pin stream disturbs no other draw: fraction 0 gives the all-free scene with the same bodies (resting
        # bodies off: which bodies rest depends on which are free)
        still_cfg = dataclasses.replace(CFG, resting_body_fraction=0.0)
        free_cfg = dataclasses.replace(still_cfg, pinned_body_fraction=0.0)
        for seed in range(3):
            scene, free = S.sample_scene(MASTER, seed, still_cfg), S.sample_scene(MASTER, seed, free_cfg)
            self.assertTrue(all(not b.pinned for b in free.bodies))
            self.assertEqual((scene.contact, scene.placement, scene.drift), (free.contact, free.placement, free.drift))
            for a, b in zip(scene.bodies, free.bodies, strict=True):
                self.assertEqual(
                    dataclasses.replace(a, pins="none", velocity=b.velocity if a.pinned else a.velocity), b
                )
        every = S.sample_scene(MASTER, 0, dataclasses.replace(CFG, pinned_body_fraction=1.0))
        self.assertTrue(all(b.pinned for b in every.bodies))
        self.assertFalse(any(b.resting for b in every.bodies))  # only free bodies rest

    def test_resting_bodies(self):
        h = CFG.cell_size
        scenes = [S.sample_scene(MASTER, seed, CFG) for seed in range(20)]
        free = [b for sc in scenes for b in sc.bodies if not b.pinned]
        resting = [b for b in free if b.resting]
        self.assertGreaterEqual(len(free), 200)
        self.assertAlmostEqual(len(resting) / len(free), CFG.resting_body_fraction, delta=0.1)
        on_bodies = sum(sc.placement["resting_on_bodies"] for sc in scenes)
        self.assertGreater(on_bodies, 0)
        self.assertLess(on_bodies, len(resting))  # some rest on the ground, some on pinned bodies
        faces = collections.Counter()
        for sc in scenes:
            for b in sc.bodies:
                self.assertEqual(b.resting, (not b.pinned) and b.resting)
                if not b.resting:
                    continue
                self.assertEqual(b.velocity, (0.0, 0.0, 0.0))  # at rest: no rigid velocity
                self.assertEqual(b.perturbation_scale, 0.0)  # and no deformation field
                # flat on one of its faces: the world's vertical is a lattice axis of the body
                down = S.rotation_matrix(b.quaternion).T @ np.array([0.0, -1.0, 0.0])
                axis = int(np.argmax(np.abs(down)))
                self.assertAlmostEqual(abs(float(down[axis])), 1.0, places=12)
                faces[2 * axis + (1 if down[axis] > 0 else 0)] += 1
        self.assertEqual(set(faces), set(range(6)))  # every face comes down
        for sc in scenes:
            lo, hi = body_aabbs(sc)
            lowest = lo[:, 1]
            pairs = support_pairs(sc)
            for a, A in enumerate(sc.bodies):
                if not A.resting:
                    continue
                # the pinned bodies under the footprint (not those hanging entirely above the resting body)
                supports = [j for j, B in enumerate(sc.bodies) if pairs[a, j] and lo[j, 1] < hi[a, 1]]
                room = []  # vertical distance to the support's box inflated by its deformation clearance
                for j in supports:
                    B = sc.bodies[j]
                    clear = S.field_clearance(B.perturbation_scale, B.strength, h)
                    dist = resting_support_distance(sc, a, j, clear)
                    if math.isfinite(dist):
                        self.assertGreaterEqual(dist / h, S.RESTING_GAP_CELLS - 1e-6)  # no overlap, clearance kept
                        room.append(dist)
                if not room:  # on the ground: the bottom face at the sample radius above the plane
                    self.assertAlmostEqual(lowest[a] / h, S.RESTING_GAP_CELLS, delta=1e-9)
                    continue
                # on a pinned body: the bottom face at the sample radius above the closest approach along -y to the
                # inflated box; the footprint samples miss the exact support point by at most their spacing
                self.assertLessEqual(min(room) / h, S.RESTING_GAP_CELLS + 0.05)
                self.assertGreater(lowest[a] / h, S.RESTING_GAP_CELLS)
        # fraction 0 reproduces the scenes recorded before the resting start (the fifth stream disturbs no draw;
        # the static faces of the sixth stream off as well, their statistics stripped from the placement record)
        ref = json.loads(REFERENCE.read_text())
        still = dataclasses.replace(CFG, resting_body_fraction=0.0, static_face_count_range=(0, 0))
        for key, d in ref["scenes"].items():
            master, seed, *validation = key.split("_")
            sc = S.sample_scene(int(master), int(seed), still, validation=bool(validation))
            self.assertFalse(any(b.resting for b in sc.bodies))
            self.assertEqual(sc.static_faces, [])
            stripped = {k: v for k, v in sc.placement.items() if k != "resting_on_bodies" and "static_face" not in k}
            sc = dataclasses.replace(sc, placement=stripped)
            self.assertEqual(sc, S.SceneV5.from_dict(d), key)
        # the face-down pose: R takes the face's outward normal to -y, the yaw turns about the vertical
        for face in range(6):
            R = S.rotation_matrix(S.face_down_quaternion(face, 0.7))
            self.assertTrue(np.allclose(R @ np.array(S.FACE_NORMALS[face], dtype=float), [0.0, -1.0, 0.0]))
            self.assertTrue(np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0))
        # the exact support against the bisection over dense samples of a footprint under tilted boxes
        rng = np.random.default_rng(3)
        foot = np.array([[-0.08, -0.06], [0.1, -0.07], [0.09, 0.05], [-0.07, 0.06]])
        u, v = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60), indexing="ij")
        grid = (1 - u[..., None]) * ((1 - v[..., None]) * foot[0] + v[..., None] * foot[1]) + u[..., None] * (
            (1 - v[..., None]) * foot[3] + v[..., None] * foot[2]
        )
        t = np.linspace(0, 1, 500)[:, None]
        samples = np.concatenate([grid.reshape(-1, 2)] + [(1 - t) * foot[k] + t * foot[(k + 1) % 4] for k in range(4)])
        checked = 0
        for _ in range(30):
            B = S.BodySpec(
                (4, 3, 5), {}, tuple(rng.normal(size=3) * 0.06), S.random_quaternion(rng), (0, 0, 0), 0, 0, 0, 0
            )
            top = S.support_height(foot, B.position, S.rotation_matrix(B.quaternion), 0.5 * h * np.array(B.cell_counts))
            bis = box_top(samples[:, 0], samples[:, 1], B, h)
            if top is None:
                self.assertFalse(np.isfinite(bis).any())
                continue
            checked += 1
            self.assertGreaterEqual(top, np.nanmax(bis) - 1e-9)  # at least every sampled height
            self.assertLessEqual(top, np.nanmax(bis) + 0.05 * h)  # and no more than the sampling resolution above
        self.assertGreater(checked, 5)

    def test_json_round_trip(self):
        self.assertTrue(any(b.pinned for b in self.scene.bodies))
        self.assertTrue(any(b.resting for b in self.scene.bodies))
        text = json.dumps(dataclasses.asdict(self.scene))
        back = S.SceneV5.from_dict(json.loads(text))
        self.assertEqual(back, self.scene)
        self.assertEqual(dataclasses.asdict(back), dataclasses.asdict(self.scene))
        self.assertEqual([b.pins for b in back.bodies], [b.pins for b in self.scene.bodies])
        self.assertEqual([b.resting for b in back.bodies], [b.resting for b in self.scene.bodies])
        # a record written before the pins and the resting start existed reads as all free and moving
        d = dataclasses.asdict(self.scene)
        for b in d["bodies"]:
            b.pop("pins")
            b.pop("resting")
        self.assertTrue(all(b.pins == "none" and not b.resting for b in S.SceneV5.from_dict(d).bodies))

    def test_summary(self):
        s = S.scene_summary(self.scene)
        self.assertEqual(s["bodies"], len(self.scene.bodies))
        self.assertEqual(s["pinned_bodies"], sum(1 for b in self.scene.bodies if b.pinned))
        self.assertGreater(s["pinned_bodies"], 0)
        self.assertEqual(s["resting_bodies"], sum(1 for b in self.scene.bodies if b.resting))
        self.assertEqual(s["resting_on_bodies"], self.scene.placement["resting_on_bodies"])
        self.assertEqual(s["cells"], self.scene.cells)
        self.assertEqual(s["kappa"], self.scene.contact["kappa"])
        self.assertAlmostEqual(s["drift_speed"], float(np.linalg.norm(self.scene.drift)))
        self.assertIn("placement_acceptance", s)
        json.dumps(s)


class TestRealise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grids = GridCache("cpu")
        cls.aug = Augmenter("cpu")
        cls.scene = S.sample_scene(MASTER, 0, CFG)

    def test_shapes_grids_material_scene(self):
        gs, X, V, material, cs = S.realise(self.scene, self.grids, self.aug, "cpu")
        O = len(self.scene.bodies)
        self.assertEqual(len(gs), O)
        N = sum(g.P for g in gs)
        self.assertEqual(tuple(X.shape), (N, 3))
        self.assertEqual(tuple(V.shape), (N, 3))
        self.assertEqual(X.dtype, torch.float32)
        self.assertTrue(torch.isfinite(X).all() and torch.isfinite(V).all())
        self.assertTrue(any(b.pinned for b in self.scene.bodies) and any(not b.pinned for b in self.scene.bodies))
        for g, b in zip(gs, self.scene.bodies, strict=True):
            self.assertEqual(g.cell_counts, b.cell_counts)
            self.assertEqual(g.pins, b.pins)
            self.assertEqual(g.pinned.numel() > 0, b.pinned)
        self.assertEqual(material.count, O)
        self.assertEqual(material.si[0]["E"], self.scene.bodies[0].material["E"])
        self.assertTrue(torch.allclose(material.kappa, torch.full((O,), self.scene.contact["kappa"])))
        self.assertTrue(
            torch.allclose(material.g, torch.tensor([[0.0, -9.81 * self.scene.dt**2 / self.scene.h, 0.0]]).expand(O, 3))
        )
        self.assertTrue(cs.plane_present.all())
        self.assertTrue((cs.plane_d == 0).all())
        self.assertTrue(torch.equal(cs.plane_n, torch.tensor([[0.0, 1.0, 0.0]]).expand(O, 3)))
        self.assertEqual(cs.points.shape[0], 0)
        self.assertTrue(torch.equal(cs.point_offsets, torch.zeros(O + 1, dtype=torch.int64)))
        # deterministic
        _, X2, V2, _, _ = S.realise(self.scene, self.grids, self.aug, "cpu")
        self.assertTrue(torch.equal(X, X2) and torch.equal(V, V2))

    def test_pinned_bodies_realise_at_the_pose_with_zero_velocity(self):
        gs, X, V, _, _ = S.realise(self.scene, self.grids, self.aug, "cpu", torch.float64)
        off = 0
        seen = 0
        for g, b in zip(gs, self.scene.bodies, strict=True):
            Xb, Vb = X[off : off + g.P], V[off : off + g.P]
            off += g.P
            if not b.pinned:
                continue
            seen += 1
            pose = S.rigid_pose(b, g.rest, self.scene.h)
            # the clamped face sits exactly at the rigid pose, the rest of the body carries the deformation field
            self.assertLess((Xb[g.pinned] - pose[g.pinned]).abs().max().item(), 1e-12)
            self.assertGreater((Xb[g.free] - pose[g.free]).abs().max().item(), 0.0)
            self.assertEqual(Vb[g.pinned].abs().max().item(), 0.0)
            # the deformation velocity field remains, the rigid velocity is zero
            self.assertGreater(Vb[g.free].abs().max().item(), 0.0)
            # a spec with a rigid velocity on a pinned body is ignored by realise
            moved = dataclasses.replace(self.scene, bodies=[dataclasses.replace(b, velocity=(1.0, 2.0, 3.0))])
            _, _, V1, _, _ = S.realise(moved, self.grids, self.aug, "cpu", torch.float64)
            self.assertTrue(torch.equal(V1, Vb))
        self.assertGreater(seen, 0)

    def test_rigid_pose_velocity_and_no_overlap(self):
        scene = rigid(self.scene)
        gs, X, V, _, _ = S.realise(scene, self.grids, self.aug, "cpu", torch.float64)
        h, dt = scene.h, scene.dt
        lo_r, hi_r = body_aabbs(scene)
        off = 0
        lo, hi = [], []
        for g, b, lo_b, hi_b in zip(gs, scene.bodies, lo_r, hi_r, strict=True):
            Xb, Vb = X[off : off + g.P], V[off : off + g.P]
            off += g.P
            self.assertLess((Xb - S.rigid_pose(b, g.rest, h)).abs().max().item(), 1e-12)
            self.assertLess((Vb - torch.tensor(b.velocity, dtype=torch.float64) * dt / h).abs().max().item(), 1e-12)
            lo.append(Xb.min(0).values.numpy() * h)
            hi.append(Xb.max(0).values.numpy() * h)
            self.assertTrue(np.allclose(lo[-1], lo_b, atol=1e-9) and np.allclose(hi[-1], hi_b, atol=1e-9))
        lo, hi = np.stack(lo), np.stack(hi)
        free_pairs = ~support_pairs(scene)
        self.assertGreaterEqual(pairwise_separation(lo, hi)[free_pairs].min(), CFG.placement_gap_cells[0] * h - 1e-9)
        lowest = lo[:, 1] / h
        resting = np.array([b.resting for b in scene.bodies])
        self.assertTrue(resting.any())
        if not resting[0]:
            self.assertTrue(S.FIRST_BODY_CELLS[0] - 1e-9 <= lowest[0] <= S.FIRST_BODY_CELLS[1] + 1e-9)
        self.assertTrue((lowest[1:][~resting[1:]] >= S.GROUND_CELLS - 1e-9).all())
        self.assertTrue((lowest[resting] >= S.RESTING_GAP_CELLS - 1e-9).all())
        # with the deformation fields: the boxes stay at least gap - 2 max|displacement| apart and the bodies above the plane
        _, Xd, _, _, _ = S.realise(self.scene, self.grids, self.aug, "cpu", torch.float64)
        disp = (Xd - X).norm(dim=-1).max().item() * h
        self.assertGreater(disp, 0.0)
        off, lo, hi = 0, [], []
        for g in gs:
            lo.append(Xd[off : off + g.P].min(0).values.numpy() * h)
            hi.append(Xd[off : off + g.P].max(0).values.numpy() * h)
            off += g.P
        self.assertGreaterEqual(
            pairwise_separation(np.stack(lo), np.stack(hi))[free_pairs].min(),
            CFG.placement_gap_cells[0] * h - 2 * disp - 1e-9,
        )
        self.assertGreater(Xd[:, 1].min().item(), -S.GROUND_CELLS)
        # resting bodies carry no field: they realise at the rigid pose with zero velocity
        gs2, Xd2, Vd2, _, _ = S.realise(self.scene, self.grids, self.aug, "cpu", torch.float64)
        off = 0
        for g, b in zip(gs2, self.scene.bodies, strict=True):
            if b.resting:
                self.assertLess((Xd2[off : off + g.P] - S.rigid_pose(b, g.rest, h)).abs().max().item(), 1e-12)
                self.assertEqual(Vd2[off : off + g.P].abs().max().item(), 0.0)
            off += g.P

    def test_resting_contact_at_step_zero(self):
        """Detection at X finds the plane pairs of every body resting on the ground with zero penetration (gap = r),
        and no pair of a resting body penetrates its support."""
        cfg = TrainConfig(scene_cells=3000)
        found_ground = found_body = 0
        for seed in range(4):
            scene = S.sample_scene(MASTER, seed, cfg)
            gs, X, V, material, cs = S.realise(scene, self.grids, self.aug, "cpu", torch.float64)
            b = Batch.build(gs, "cpu", torch.float64)
            b.material, b.scene, b.body_contact = material, cs, True
            b.X, b.V, b.X_prev, b.x = X.clone(), V.clone(), X.clone(), X.clone()
            b.pairs = p = contact.detect(b, X, V)
            _, _, _, gap, r_total, _ = contact._geometry(b, X, p)
            pen = r_total - gap  # penetration depth of every pair at X
            lo, _ = body_aabbs(scene)
            for o, body in enumerate(scene.bodies):
                if not body.resting:
                    continue
                own = p.obj == o
                partner = p.partner_body == o
                self.assertLessEqual(float(pen[own | partner].max()) if bool((own | partner).any()) else 0.0, 1e-6)
                if abs(lo[o, 1] / scene.h - S.RESTING_GAP_CELLS) < 1e-9:  # on the ground
                    plane = own & (p.kind == 0)
                    self.assertGreater(int(plane.sum()), 0, f"seed {seed} body {o}: no plane pair at step 0")
                    self.assertLess(pen[plane].abs().max().item(), 1e-6)  # gap = r exactly: d = 0
                    found_ground += 1
                else:
                    found_body += int(bool(((own | partner) & (p.kind == contact.KIND_BODY)).any()))
        self.assertGreater(found_ground, 0)

    def test_batch_energy_smoke(self):
        cfg = TrainConfig(scene_cells=2000)
        scene = S.sample_scene(MASTER, 0, cfg)
        gs, X, V, material, cs = S.realise(scene, self.grids, self.aug, "cpu", torch.float64)
        b = Batch.build(gs, "cpu", torch.float64)
        b.material, b.scene = material, cs
        b.X, b.V, b.X_prev, b.x = X.clone(), V.clone(), X.clone(), X.clone()
        b.Y = X + V + material.g[b.corner_obj]
        F = hx.gauss_deformation(X[b.cells], b.hc)
        b.C_prev = hx.mat3_tn(F, F)
        b.pairs = contact.detect(b, X, V)
        self.assertTrue(b.any_free)
        self.assertTrue((b.pairs.kind == 0).all())  # the ground plane is the only static partner
        E, gX = physics.energy_and_grad(b, X)
        self.assertEqual(tuple(E.shape), (len(gs),))
        self.assertEqual(tuple(gX.shape), tuple(X.shape))
        self.assertTrue(torch.isfinite(E).all() and torch.isfinite(gX).all())
        self.assertTrue((E > 0).all())  # inertia against Y = X + V + g
        # the rigid pose carries no elastic or damping energy: the rotation and translation are exact
        _, X_r, _, _, _ = S.realise(rigid(scene), self.grids, self.aug, "cpu", torch.float64)
        b.X = X_r
        F = hx.gauss_deformation(X_r[b.cells], b.hc)
        b.C_prev = hx.mat3_tn(F, F)
        E_el, E_d = physics.elastic_damping(b, X_r)
        self.assertLess(E_el.abs().max().item(), 1e-12)
        self.assertEqual(E_d.abs().max().item(), 0.0)


class TestConfigKeys(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.scene_mode, "body")
        self.assertEqual(cfg.scene_cells, 64000)
        self.assertEqual(cfg.body_sides, (3, 12))
        self.assertEqual(cfg.drift_speed_range, (0.1, 0.5))
        self.assertEqual(cfg.placement_height, 1.0)
        self.assertEqual(cfg.placement_gap_cells, (1, 3))
        self.assertEqual(cfg.pinned_body_fraction, 0.25)
        self.assertEqual(cfg.resting_body_fraction, 0.3)
        self.assertEqual((cfg.static_face_count_range, cfg.static_face_size_cells), ((0, 8), (2, 10)))
        self.assertEqual((cfg.scene_count, cfg.validation_scene_count, cfg.validation_full_scene_count), (64, 8, 2))
        d = cfg.to_dict()
        d.update({"scene_mode": "v5", "body_sides": [4, 8], "scene_cells": 1000, "pinned_body_fraction": 0.5})
        back = TrainConfig.from_dict(json.loads(json.dumps(d)))
        self.assertEqual((back.scene_mode, back.body_sides, back.scene_cells), ("v5", (4, 8), 1000))
        self.assertEqual(back.pinned_body_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()


class TestSceneCurriculum(unittest.TestCase):
    """The scene curriculum (Anka, 2026-10-02): pinned bodies against fixed geometry first, free bodies later."""

    def test_mix_schedule(self):
        cfg = dataclasses.replace(CFG, scene_curriculum=True, scene_curriculum_epochs=(4, 20))
        final = S.SceneMix(CFG.pinned_body_fraction, CFG.resting_body_fraction)
        self.assertEqual(S.scene_mix(cfg), final)  # no epoch: the goal
        self.assertEqual(S.scene_mix(cfg, None), final)
        for epoch in (1, 4):
            self.assertEqual(S.scene_mix(cfg, epoch), S.SceneMix(1.0, 0.0))
        mid = S.scene_mix(cfg, 12)  # halfway
        self.assertAlmostEqual(mid.pinned_fraction, 1.0 - 0.5 * (1.0 - CFG.pinned_body_fraction))
        self.assertAlmostEqual(mid.resting_fraction, 0.5 * CFG.resting_body_fraction)
        for epoch in (20, 21, 48):
            self.assertEqual(S.scene_mix(cfg, epoch), final)
        pins = [S.scene_mix(cfg, e).pinned_fraction for e in range(1, 25)]
        self.assertEqual(pins, sorted(pins, reverse=True))
        off = dataclasses.replace(CFG, scene_curriculum=False)
        self.assertEqual(S.scene_mix(off, 1), final)

    def test_mix_overrides_the_fractions_on_the_same_streams(self):
        every = S.sample_scene(MASTER, 0, CFG, mix=S.SceneMix(1.0, 0.0))
        self.assertTrue(all(b.pinned for b in every.bodies))
        self.assertFalse(any(b.resting for b in every.bodies))
        self.assertTrue(all(b.velocity == (0.0, 0.0, 0.0) for b in every.bodies))
        default = S.sample_scene(MASTER, 0, CFG)
        same = S.sample_scene(MASTER, 0, CFG, mix=S.scene_mix(CFG))
        self.assertEqual(dataclasses.asdict(same), dataclasses.asdict(default))
        # the body draws are shared: a higher pinned fraction pins a superset, the sizes and materials do not change
        self.assertEqual([b.cell_counts for b in every.bodies], [b.cell_counts for b in default.bodies])
        self.assertEqual([b.material for b in every.bodies], [b.material for b in default.bodies])
        half = S.sample_scene(MASTER, 0, CFG, mix=S.SceneMix(0.6, 0.15))
        pinned_default = {i for i, b in enumerate(default.bodies) if b.pinned}
        pinned_half = {i for i, b in enumerate(half.bodies) if b.pinned}
        self.assertTrue(pinned_default <= pinned_half)
        self.assertGreater(len(pinned_half), len(pinned_default))
        self.assertEqual(len(every.static_faces), len(default.static_faces))

    def test_epoch_scene_seed(self):
        seeds = {S.epoch_scene_seed(e, i) for e in range(1, 4) for i in range(3)}
        self.assertEqual(len(seeds), 9)
        self.assertTrue(all(s >= S.EPOCH_SEED_STRIDE for s in seeds))  # never a test/reference scene seed
        a = S.sample_scene(MASTER, S.epoch_scene_seed(1, 0), CFG)
        b = S.sample_scene(MASTER, S.epoch_scene_seed(2, 0), CFG)
        self.assertNotEqual([x.cell_counts for x in a.bodies], [x.cell_counts for x in b.bodies])
        with self.assertRaises(ValueError):
            S.epoch_scene_seed(1, S.EPOCH_SEED_STRIDE)
        with self.assertRaises(ValueError):
            S.epoch_scene_seed(1, -1)
