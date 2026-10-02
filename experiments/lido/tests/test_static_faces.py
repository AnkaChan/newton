# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static colliding faces in v5 scenes (design spec section 11, Anka 2026-10-02): the generator's face draws
(statistics, determinism, placement in the column above the ground and clear of every body's box, JSON round trip,
count 0 reproducing the recorded scenes), detection of a free box above and below a static quad (two-sided normal
towards the box, closest points on the finite quad, no lateral rule, competition for the M_PAIR slots), tokens
with the radius channel at its cap, the penetration fold, the contact gradient by finite differences, a tilted
static face reproducing the tilted plane step for step, a box dropped onto a tilted face (rests or slides, never
falls through), the Warp kernel and the fused pass against torch without any partner scatter, capacity against
compaction, the captured query against eager, and the scene runner's epoch with static faces on the CPU."""

import dataclasses
import json
import math
import unittest

import numpy as np
import torch

from experiments.lido import contact, physics
from experiments.lido import hex as hx
from experiments.lido import scenes_v5 as S
from experiments.lido import validation as V
from experiments.lido.augment import Augmenter
from experiments.lido.batch import Batch
from experiments.lido.capture import CapturedQuery
from experiments.lido.config import TrainConfig
from experiments.lido.frames import frames
from experiments.lido.fusion import Fusion
from experiments.lido.grid import Grid, GridCache
from experiments.lido.network import Net
from experiments.lido.runner import pair_counts, scene_batch
from experiments.lido.step import Step
from experiments.lido.structs import Material
from experiments.lido.tests.test_body_contact import MATERIAL_FIELDS, SI, SMALL_CFG, world_scene
from experiments.lido.tests.test_capture import FullFloat32, small_net
from experiments.lido.tests.test_contact_kernel import rel_err, torch_paths
from experiments.lido.tests.test_scene_runner import MASTER as RUNNER_MASTER
from experiments.lido.tests.test_scene_runner import make_runner_cpu, scene_cfg
from experiments.lido.tests.test_scenes_v5 import REFERENCE, body_aabbs
from experiments.lido.units import material_from_si

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
MASTER = 73
CFG = TrainConfig(
    scene_cells=6000, pinned_contact_face_fraction=0.0
)  # the regular faces alone (TestPinnedContactFaces covers the rest)
R = contact.R_SAMPLE
UP = (0.0, 1.0, 0.0)


def quad(centre, normal=UP, sides=(10.0, 10.0), angle=0.0) -> np.ndarray:
    """Corners [4,3] (cells) of a static quad."""
    return S.static_face_corners(centre, normal, angle, sides)


def quad_points(corners: np.ndarray, n: int = 60) -> np.ndarray:
    """A dense bilinear grid of points [n*n,3] on the quad."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    u, v = u[..., None], v[..., None]
    pts = (1 - u) * ((1 - v) * corners[0] + v * corners[1]) + u * ((1 - v) * corners[3] + v * corners[2])
    return pts.reshape(-1, 3)


def rotation_z(angle: float, dtype=torch.float64, device="cpu") -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)


def boxes_with_faces(
    faces,
    sides=((3, 3, 3),),
    offsets=((0.0, 0.0, 0.0),),
    plane_d=None,
    dtype=torch.float64,
    device="cpu",
    kappa=10.0,
    beta=0.3,
    mu_f=0.4,
    rotation=None,
    noise=0.0,
    seed=0,
) -> Batch:
    """Free boxes (rest lattice at the origin plus `offsets`) in one world frame with the static quads `faces`
    (list of [4,3] corner arrays, cells), the y-plane at `plane_d` when given; `rotation` [3,3] turns the whole
    world (corners and faces) about the origin; V = 0, X_prev = x = Y = X."""
    device = torch.device(device)
    gen = torch.Generator().manual_seed(seed)
    grids = [Grid.build(s, pins="none", device=device) for s in sides]
    b = Batch.build(grids, device, dtype)
    b.material = Material.cat(
        [
            material_from_si(cell_count=g.C, sample_count=g.S, kappa=kappa, beta=beta, mu_f=mu_f, device=device, **SI)
            for g in grids
        ]
    )
    for f in MATERIAL_FIELDS:
        setattr(b.material, f, getattr(b.material, f).to(dtype))
    b.scene = world_scene(len(grids), plane_d, dtype, device)
    f = torch.tensor(np.stack(faces), dtype=dtype, device=device) if faces else torch.zeros(0, 4, 3, dtype=dtype)
    b.body_contact = True
    X = b.rest.to(dtype).clone()
    for o, off in enumerate(offsets):
        X[b.corner_obj == o] += torch.tensor(off, dtype=dtype, device=device)
    X += noise * torch.randn(b.N, 3, generator=gen, dtype=dtype).to(device)
    if rotation is not None:
        Rm = rotation.to(device=device, dtype=dtype)
        X = X @ Rm.T
        f = f @ Rm.T
    b.scene.faces = f.to(device)
    b.X, b.V = X, torch.zeros_like(X)
    b.X_prev, b.x, b.Y = X.clone(), X.clone(), X.clone()
    return b


def static_rows(pairs):
    return (pairs.partner_body == contact.PARTNER_STATIC) & pairs.valid


class TestSampling(unittest.TestCase):
    """The sixth seed stream: 0-8 quads of 2-10 cells per side, random orientation, in the column above the ground
    and clear of the bodies' grown boxes; count 0 reproduces the scenes recorded before."""

    @classmethod
    def setUpClass(cls):
        cls.scenes = [S.sample_scene(MASTER, seed, CFG) for seed in range(24)]

    def test_statistics_and_determinism(self):
        lo_n, hi_n = CFG.static_face_count_range
        lo_s, hi_s = CFG.static_face_size_cells
        counts = [len(sc.static_faces) for sc in self.scenes]
        self.assertTrue(all(lo_n <= c <= hi_n for c in counts))
        self.assertGreater(max(counts), 4)
        self.assertAlmostEqual(float(np.mean(counts)), 0.5 * (lo_n + hi_n), delta=1.5)
        normals = []
        for sc in self.scenes:
            h = sc.h
            self.assertEqual(sc.placement["static_faces_dropped"], 0)
            if sc.static_faces:
                self.assertGreater(sc.placement["static_face_acceptance"], 0.0)
            else:
                self.assertEqual(sc.placement["static_face_draws"], 0)
            for f in sc.static_faces:
                c = np.asarray(f)
                self.assertEqual(c.shape, (4, 3))
                e1, e2 = c[1] - c[0], c[3] - c[0]
                self.assertTrue(lo_s - 1e-9 <= np.linalg.norm(e1) / h <= hi_s + 1e-9)
                self.assertTrue(lo_s - 1e-9 <= np.linalg.norm(e2) / h <= hi_s + 1e-9)
                self.assertLess(abs(float(e1 @ e2)), 1e-9)  # a rectangle ...
                self.assertLess(np.abs(c[2] - (c[1] + c[3] - c[0])).max(), 1e-9)  # ... and planar
                normals.append(np.cross(c[2] - c[0], c[3] - c[1]) / np.linalg.norm(np.cross(c[2] - c[0], c[3] - c[1])))
        normals = np.stack(normals)
        self.assertGreater(len(normals), 50)
        self.assertGreater(float(np.abs(normals[:, 1]).std()), 0.2)  # random orientations, not a preferred axis
        self.assertGreater(float(np.abs(normals).mean(0).min()), 0.3)
        # deterministic, distinct across seeds and from the validation stream
        again = S.sample_scene(MASTER, 0, CFG)
        self.assertEqual(dataclasses.asdict(again), dataclasses.asdict(self.scenes[0]))
        self.assertNotEqual(self.scenes[0].static_faces, self.scenes[1].static_faces)
        held = S.held_out_scene(MASTER, 0, CFG)
        self.assertNotEqual(held.static_faces, self.scenes[0].static_faces)

    def test_faces_in_the_column_above_the_ground_and_outside_the_bodies(self):
        for sc in self.scenes[:12]:
            h = sc.h
            lo, hi = body_aabbs(sc)
            grown = np.maximum(
                S.FACE_GROWTH_CELLS * h, [S.field_clearance(b.perturbation_scale, b.strength, h) for b in sc.bodies]
            )
            lo_g, hi_g = lo - grown[:, None], hi + grown[:, None]
            half = 0.5 * sc.placement["footprint"]
            for f in sc.static_faces:
                c = np.asarray(f)
                centre = c.mean(0)
                self.assertTrue(abs(centre[0]) <= half + 1e-9 and abs(centre[2]) <= half + 1e-9)
                self.assertTrue(0.0 <= centre[1] <= CFG.placement_height + 1e-9)
                self.assertGreaterEqual(c[:, 1].min(), 0.0)  # no corner below the ground
                self.assertFalse(S.quad_boxes_overlap(c, lo_g, hi_g).any())
                # independent check: no point of the quad inside any grown box
                pts = quad_points(c)
                inside = ((pts[:, None, :] >= lo_g[None] - 1e-12) & (pts[:, None, :] <= hi_g[None] + 1e-12)).all(-1)
                self.assertFalse(inside.any())
            # the bodies' boxes themselves are at least a cell from every face: the first contact needs motion
        # the separating-axis test against brute force on random quads and boxes
        rng = np.random.default_rng(1)
        for _ in range(200):
            c = quad(rng.normal(size=3) * 0.1, rng.normal(size=3), rng.uniform(0.05, 0.3, size=2), rng.uniform(0, 6.3))
            lo = rng.normal(size=(4, 3)) * 0.1
            hi = lo + rng.uniform(0.02, 0.3, size=(4, 3))
            sat = S.quad_boxes_overlap(c, lo, hi)
            pts = quad_points(c, 150)
            for j in range(4):
                inside = ((pts >= lo[j] - 1e-9) & (pts <= hi[j] + 1e-9)).all(1).any()
                near = ((pts >= lo[j] - 2e-3) & (pts <= hi[j] + 2e-3)).all(1).any()
                if inside:
                    self.assertTrue(sat[j])
                if sat[j]:
                    self.assertTrue(near)  # an overlap the sampling misses lies within its resolution
            self.assertEqual(S.quad_box_overlap(c, lo[0], hi[0]), bool(sat[0]))

    def test_count_zero_reproduces_the_recorded_scenes_and_the_body_draws(self):
        ref = json.loads(REFERENCE.read_text())
        still = dataclasses.replace(
            CFG, resting_body_fraction=0.0, static_face_count_range=(0, 0), scene_wave_speed_min=0.0
        )
        for key, d in ref["scenes"].items():
            master, seed, *validation = key.split("_")
            sc = S.sample_scene(int(master), int(seed), still, validation=bool(validation))
            self.assertEqual(sc.static_faces, [])
            stripped = {
                k: v
                for k, v in sc.placement.items()
                if k != "resting_on_bodies" and "static_face" not in k and "pinned_contact" not in k
            }
            self.assertEqual(dataclasses.replace(sc, placement=stripped), S.SceneV5.from_dict(d), key)
        # the face stream disturbs no other draw: bodies, contact, drift and the body placement are the same
        for seed in range(3):
            with_faces, without = (
                self.scenes[seed],
                S.sample_scene(MASTER, seed, dataclasses.replace(CFG, static_face_count_range=(0, 0))),
            )
            self.assertEqual(with_faces.bodies, without.bodies)
            self.assertEqual((with_faces.contact, with_faces.drift), (without.contact, without.drift))
            self.assertEqual(
                {k: v for k, v in with_faces.placement.items() if "static_face" not in k},
                {k: v for k, v in without.placement.items() if "static_face" not in k},
            )
            self.assertEqual(without.static_faces, [])
            self.assertEqual(without.placement["static_face_draws"], 0)

    def test_json_round_trip_summary_and_realise(self):
        sc = next(s for s in self.scenes if len(s.static_faces) >= 3)
        back = S.SceneV5.from_dict(json.loads(json.dumps(dataclasses.asdict(sc))))
        self.assertEqual(back, sc)
        self.assertEqual(back.static_faces, sc.static_faces)
        d = dataclasses.asdict(sc)
        d.pop("static_faces")  # a record written before the static faces existed
        self.assertEqual(S.SceneV5.from_dict(d).static_faces, [])
        summary = S.scene_summary(sc)
        self.assertEqual(summary["static_faces"], len(sc.static_faces))
        self.assertEqual(summary["static_face_acceptance"], sc.placement["static_face_acceptance"])
        json.dumps(summary)
        small = S.sample_scene(
            MASTER, 0, TrainConfig(scene_cells=2000, static_face_count_range=(3, 5), pinned_contact_face_fraction=0.0)
        )
        gs, X, V, material, cs = S.realise(small, GridCache("cpu"), Augmenter("cpu"), "cpu", torch.float64)
        self.assertEqual(tuple(cs.faces.shape), (len(small.static_faces), 4, 3))
        self.assertEqual(cs.faces.dtype, torch.float64)
        expected = torch.tensor(small.static_faces, dtype=torch.float64) / small.h
        self.assertTrue(torch.allclose(cs.faces, expected))
        self.assertTrue(cs.plane_present.all())
        # float32 by default and the faces at least one cell from every body: no static pair at step 0 at rest
        _, _, _, _, cs32 = S.realise(small, GridCache("cpu"), Augmenter("cpu"), "cpu")
        self.assertEqual(cs32.faces.dtype, torch.float32)
        b = Batch.build(gs, "cpu", torch.float64)
        b.material, b.scene, b.body_contact = material, cs, True
        b.X, b.V, b.X_prev, b.x = X.clone(), torch.zeros_like(V), X.clone(), X.clone()
        p = contact.detect(b, b.X, b.V)
        self.assertEqual(int(static_rows(p).sum()), 0)


class TestDetection(unittest.TestCase):
    """A free 3x3x3 box over (and under) static quads on the CPU in float64."""

    def test_box_above_and_below_a_slab(self):
        for order in (1, -1):  # both corner orders: the two-sided normal points at the box either way
            face = quad((1.5, 0.0, 1.5))[::order].copy()
            b = boxes_with_faces([face], offsets=((0.0, 0.4, 0.0),))  # bottom samples at gap 0.4 < r
            p = contact.detect(b, b.X, b.V)
            self.assertEqual(p.count, 9)
            self.assertTrue((p.kind == contact.KIND_STATIC).all())
            self.assertTrue((p.partner_body == contact.PARTNER_STATIC).all())
            self.assertTrue((p.partner_face == 0).all())
            self.assertTrue((p.radius == contact.STATIC_RADIUS).all())
            self.assertTrue((b.sample_face[p.sample] == 2).all())  # the bottom faces
            self.assertTrue(torch.allclose(p.partner_normal, torch.tensor([UP], dtype=torch.float64)))
            xs = contact.sample_positions(b, b.X)
            self.assertTrue(torch.allclose(p.anchor, xs[p.sample]))
            gap = ((xs[p.sample] - p.partner_point) * p.partner_normal).sum(-1)
            self.assertTrue(torch.allclose(gap, torch.full((9,), 0.4, dtype=torch.float64)))
            foot = xs[p.sample] * torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
            self.assertTrue(torch.allclose(p.partner_point, foot))  # the closest point is the foot point
            self.assertTrue((p.cell[1:] >= p.cell[:-1]).all())
            self.assertTrue(
                torch.equal(p.token_offsets[1:] - p.token_offsets[:-1], torch.bincount(p.cell, minlength=b.C))
            )
            # the box under the slab: its top samples pair and see the normal pointing down at them
            b = boxes_with_faces([face], offsets=((0.0, -3.4, 0.0),))
            p = contact.detect(b, b.X, b.V)
            self.assertEqual(p.count, 9)
            self.assertTrue((b.sample_face[p.sample] == 3).all())
            self.assertTrue(torch.allclose(p.partner_normal, -torch.tensor([UP], dtype=torch.float64)))
            gap = ((contact.sample_positions(b, b.X)[p.sample] - p.partner_point) * p.partner_normal).sum(-1)
            self.assertTrue(torch.allclose(gap, torch.full((9,), 0.4, dtype=torch.float64)))
        # out of reach: nothing; a velocity towards the slab extends the margin
        b = boxes_with_faces([quad((1.5, 0.0, 1.5))], offsets=((0.0, 1.2, 0.0),))
        self.assertEqual(contact.detect(b, b.X, b.V).count, 0)
        b.V[:, 1] = -0.5
        self.assertEqual(contact.detect(b, b.X, b.V).count, 9)
        b.V[:, 1] = 0.5  # the margin is a speed
        self.assertEqual(contact.detect(b, b.X, b.V).count, 9)
        # penetration within the step: the pairs found at X (gap 0.4) hold their normal while the candidate sinks
        # 0.7 through the slab's plane (gap -0.3), as for the plane
        b = boxes_with_faces([quad((1.5, 0.0, 1.5))], offsets=((0.0, 0.4, 0.0),))
        b.pairs = p = contact.detect(b, b.X, b.V)
        x = b.X - torch.tensor([0.0, 0.7, 0.0], dtype=torch.float64)
        _, _, _, gap, _, _ = contact._geometry(b, x, p)
        self.assertTrue(torch.allclose(gap, torch.full((9,), -0.3, dtype=torch.float64)))
        self.assertAlmostEqual(contact.penetration(b, x).item(), 0.8 / R, places=12)
        self.assertAlmostEqual(contact.kind_penetration(b, x)[contact.KIND_STATIC].item(), 0.8 / R, places=12)
        # a sample that has crossed the quad's plane by the time of a detection sees the face from the other side
        # with its own normal pointing away: a thin face has no interior, so it is no candidate (unlike the plane,
        # whose one side is the inside); the box's other samples hold it
        b = boxes_with_faces([quad((1.5, 0.0, 1.5))], offsets=((0.0, -0.3, 0.0),))
        self.assertEqual(contact.detect(b, b.X, b.V).count, 0)

    def test_tilted_quad_and_its_edges(self):
        Rm = rotation_z(0.35)
        face = quad((1.5, 0.0, 1.5), sides=(6.0, 6.0))
        b = boxes_with_faces([face], offsets=((0.0, 0.4, 0.0),), rotation=Rm)
        p = contact.detect(b, b.X, b.V)
        rows = b.sample_face[p.sample] == 2
        self.assertEqual(int(rows.sum()), 9)  # the bottom samples (plus grazing side samples from the rounding)
        n = (Rm @ torch.tensor(UP, dtype=torch.float64))[None]
        self.assertTrue(torch.allclose(p.partner_normal[rows], n.expand(9, 3)))
        xs = contact.sample_positions(b, b.X)
        gap = ((xs[p.sample] - p.partner_point) * p.partner_normal).sum(-1)
        self.assertTrue(torch.allclose(gap[rows], torch.full((9,), 0.4, dtype=torch.float64)))
        self.assertTrue((gap >= 0.4 - 1e-9).all())  # the grazing side samples lie higher
        # the closest points lie on the quad: the brute-force distance over a dense grid is no smaller
        fq = b.scene.faces[0].numpy()
        pts = torch.tensor(quad_points(fq, 200), dtype=torch.float64)
        brute = (xs[p.sample][:, None, :] - pts[None]).norm(dim=-1).min(1).values
        d = (xs[p.sample] - p.partner_point).norm(dim=-1)
        self.assertTrue((d <= brute + 1e-9).all())
        self.assertLess((brute - d).max().item(), 0.05)
        # overhanging the edge: the closest point sits on the quad's edge, the gap stays the plane distance (no
        # lateral rule), and a sample too far beyond the edge is out of reach
        face = quad((1.5, 0.0, 1.5), sides=(10.0, 4.0))  # x in [-0.5, 3.5] (the second side runs along x)
        b = boxes_with_faces([face], offsets=((3.3, 0.4, 0.0),))  # the box's x in [3.3, 6.3]
        p = contact.detect(b, b.X, b.V)
        xs = contact.sample_positions(b, b.X)
        bottom = b.sample_face[p.sample] == 2
        self.assertEqual(int(bottom.sum()), 3)  # the row of bottom samples at x = 3.8: 0.3 beyond the edge
        self.assertTrue(torch.allclose(p.partner_point[bottom][:, 0], torch.full((3,), 3.5, dtype=torch.float64)))
        gap = ((xs[p.sample] - p.partner_point) * p.partner_normal).sum(-1)
        self.assertTrue(torch.allclose(gap[bottom], torch.full((3,), 0.4, dtype=torch.float64)))
        d = (xs[p.sample] - p.partner_point).norm(dim=-1)
        self.assertTrue(torch.allclose(d[bottom], torch.full((3,), math.hypot(0.3, 0.4), dtype=torch.float64)))
        b = boxes_with_faces([face], offsets=((4.2, 0.4, 0.0),))  # 1.2 beyond the edge: distance - r > margin
        self.assertEqual(contact.detect(b, b.X, b.V).count, 0)

    def test_slots_shared_with_plane_and_bodies(self):
        """Box A on the plane under a slab with box B beside it: plane, static face and body pairs compete for the
        slots; the capacity layout has 1 + min(M_PAIR, 1 + O - 1) columns."""
        slab = quad((1.5, 3.4, 1.5))
        b = boxes_with_faces(
            [slab], sides=((3, 3, 3), (3, 3, 3)), offsets=((0.0, 0.0, 0.0), (3.4, 0.0, 0.0)), plane_d=-0.4
        )
        b.pairs = p = contact.detect(b, b.X, b.V)
        counts = pair_counts(b)
        self.assertEqual(counts["point"], 0)
        self.assertEqual(counts["plane"], 18)  # both bottoms
        self.assertEqual(counts["static"], 18)  # both tops under the slab
        self.assertEqual(counts["body"], 18)  # A's +x face against B's -x face, both directions
        self.assertEqual(counts["total"], 54)
        self.assertEqual(int(static_rows(p).sum()), 18)
        same = p.sample[1:] == p.sample[:-1]
        self.assertTrue((p.kind[1:][same] >= p.kind[:-1][same]).all())  # plane, static, body within a sample
        cap = contact.detect(b, b.X, b.V, capacity=True)
        self.assertEqual(cap.count, b.S * (1 + min(contact.M_PAIR, 1 + b.O - 1)))
        self.assertEqual(int(cap.valid.sum()), p.count)
        for name in ("sample", "cell", "obj", "partner_point", "partner_normal", "kind", "radius", "anchor"):
            self.assertTrue(torch.equal(getattr(cap, name)[cap.valid], getattr(p, name)), name)
        for name in ("partner_body", "partner_face"):
            self.assertTrue(torch.equal(getattr(cap, name)[cap.valid], getattr(p, name)), name)

    def test_tokens_and_the_penetration_fold(self):
        face = quad((1.5, 0.0, 1.5), normal=(0.3, 1.0, -0.2), sides=(8.0, 8.0), angle=0.4)
        b = boxes_with_faces([face], offsets=((0.0, 0.6, 0.0),), plane_d=-0.3, noise=0.02)
        b.pairs = contact.detect(b, b.X, b.V)
        rows = static_rows(b.pairs)
        self.assertGreater(int(rows.sum()), 0)
        self.assertGreater(int(((b.pairs.kind == 0) & b.pairs.valid).sum()), 0)
        Rf = torch.eye(3, dtype=torch.float64).expand(b.C, 3, 3)
        t = contact.contact_tokens(b, b.X, Rf)
        self.assertEqual(t.shape, (b.pairs.count, contact.TOKEN_DIM))
        self.assertTrue((t[rows, 15:18].argmax(1) == contact.KIND_STATIC).all())
        self.assertTrue((t[rows, 11] == contact.RADIUS_CHANNEL_CAP).all())  # r_p / r at its cap
        self.assertTrue((t[~rows, 11] == 1.0).all())  # the plane
        centre = b.X[b.cells[b.pairs.cell]].mean(1)
        _, p, n, gap, _, _ = contact._geometry(b, b.X, b.pairs)
        self.assertTrue(torch.allclose(t[:, 3:6], p - centre))
        self.assertTrue(torch.allclose(t[:, 6:9], n))
        self.assertTrue(torch.allclose(t[:, 9], gap / R))
        self.assertTrue(torch.isfinite(t).all())
        # penetration by kind: the static slot carries the faces; the validation folds it into plane_penetration_r
        nq = np.cross(face[2] - face[0], face[3] - face[1])
        x = b.X - 0.5 * torch.tensor(nq / np.linalg.norm(nq), dtype=torch.float64)  # pushed into the face
        by_kind = contact.kind_penetration(b, x)
        self.assertGreater(by_kind[contact.KIND_STATIC].item(), 0.0)
        self.assertEqual(by_kind[contact.KIND_BODY].item(), 0.0)
        b.x = x
        metrics = V.scene_metrics(b)
        self.assertEqual(metrics["plane_penetration_r"], max(by_kind[0].item(), by_kind[1].item()))
        self.assertEqual(metrics["contact_pairs"]["static"], int(rows.sum()))
        self.assertEqual(metrics["contact_pairs"]["point"], 0)
        # a static face alone (no plane) still shows in the folded key
        b.scene.plane_present.fill_(False)
        b.pairs = contact.detect(b, b.X, b.V)
        self.assertTrue(static_rows(b.pairs).all())
        self.assertGreater(V.scene_metrics(b)["plane_penetration_r"], 0.0)


class TestGradientAndEquivalence(unittest.TestCase):
    def test_finite_differences_with_a_static_partner(self):
        """Autograd against central differences (friction load frozen) with respect to the body's corners, with
        damping and friction active on a tilted face; the static face receives nothing (no other body exists, and
        the owner's force equals minus the gradient sum)."""
        gen = torch.Generator().manual_seed(4)
        for angle, normal in ((0.0, UP), (0.3, (0.2, 1.0, 0.1)), (1.1, (-0.4, 1.0, 0.3))):
            face = quad((1.5, 0.0, 1.5), normal=normal, sides=(12.0, 12.0), angle=angle)
            n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
            b = boxes_with_faces([face], offsets=(tuple(0.4 * n),), beta=0.5, mu_f=0.6, noise=0.02, seed=1)
            b.pairs = contact.detect(b, b.X, b.V)
            self.assertGreaterEqual(int(static_rows(b.pairs).sum()), 9)
            push = torch.tensor(-0.15 * n + 0.05 * np.cross(n, [0.0, 0.0, 1.0]), dtype=torch.float64)
            x = (b.X + push).requires_grad_(True)
            E = contact.contact_energy(b, x)
            self.assertGreater(E.item(), 0.0)
            (g,) = torch.autograd.grad(E.sum(), x)
            with torch.no_grad():
                _, _, _, gap, r_total, _ = contact._geometry(b, x, b.pairs)
                load = b.material.ke[b.pairs.obj] * torch.relu(r_total - gap)
                E_n, E_d, E_f = contact.pair_energies(b, x, b.pairs, load)
                self.assertTrue((E_n > 0).any() and (E_d > 0).any() and (E_f > 0).any())
                frozen = lambda xx, b=b, load=load: sum(contact.pair_energies(b, xx, b.pairs, load)).sum()  # noqa: E731
                for _ in range(3):
                    e = torch.randn(b.N, 3, dtype=torch.float64, generator=gen)
                    fd = (frozen(x + 1e-6 * e) - frozen(x - 1e-6 * e)) / 2e-6
                    self.assertAlmostEqual(fd.item(), (g * e).sum().item(), delta=1e-7 * max(1.0, abs(fd.item())))
            F = contact.contact_force(b, x.detach())
            self.assertTrue(torch.allclose(F[0], -g.sum(0)))
            ke_sum, H = contact.active_stiffness(b, x.detach())
            active = (r_total - gap > 0) & b.pairs.valid
            self.assertAlmostEqual(ke_sum[0].item(), int(active.sum()) * b.material.ke[0].item(), places=9)
            self.assertTrue(torch.allclose(H, H.transpose(-1, -2)))

    def test_tilted_face_reproduces_the_tilted_plane(self):
        """The same box on the same tilted surface, once a static quad and once the plane (plane_n, plane_d): the
        energy, gradient, centroid update and 20 steps of the zero-init solver agree to round-off (the pair law is
        the same; only the token's partner point and radius channel differ, which the zero-init network ignores)."""
        Rm = rotation_z(math.radians(20))
        n = Rm @ torch.tensor(UP, dtype=torch.float64)
        face = quad((1.5, 0.0, 1.5), sides=(16.0, 16.0))
        runs = []
        for static in (True, False):
            b = boxes_with_faces(
                [face] if static else [], offsets=((0.0, 0.4, 0.0),), plane_d=None, rotation=Rm, mu_f=0.6, kappa=20.0
            )
            if not static:
                b.scene.plane_n[:] = n
                b.scene.plane_d[:] = 0.0
                b.scene.plane_present[:] = True
            step = Step(Net.from_config(SMALL_CFG).double().eval(), Fusion(), translation="implicit_contact")
            sel = torch.ones(1, dtype=torch.bool)
            step.prepare(b, sel)
            kinds = b.pairs.partner_body if static else b.pairs.kind
            self.assertTrue((kinds == (contact.PARTNER_STATIC if static else 0)).all())
            # the bottom samples are loaded at gap 0.4 (grazing side rows may differ by rounding; they carry nothing)
            gap = contact._geometry(b, b.X, b.pairs)[3]
            self.assertEqual(int((gap < R).sum()), 9)
            self.assertGreater(b.E.item(), 0.0)
            xs, es = [], [b.E.clone()]
            for _ in range(20):
                for _ in range(2):
                    step.commit(b, step.query(b))
                step.advance(b, sel)
                xs.append(b.X.clone())
                es.append(b.E.clone())
            runs.append((torch.stack(xs), torch.stack(es)))
        (X_s, E_s), (X_p, E_p) = runs
        self.assertGreater((X_s[-1] - X_s[0]).abs().max().item(), 0.1)  # the box moved (slid along the surface)
        self.assertLess((X_s - X_p).abs().max().item(), 1e-9)
        self.assertLess((E_s - E_p).abs().max().item(), 1e-9 * max(1.0, E_p.abs().max().item()))


class TestDrop(unittest.TestCase):
    """A free 3x3x3 box dropped from 1.1 cells onto a 15-degree static ramp, zero-init network, implicit-contact
    centroid update, 2 queries per step in float64: it lands with a bounded penetration and never falls through;
    the plane far below never pairs. Without friction it slides down the ramp as a whole and keeps its bottom face
    exactly one sample radius over the quad for 60 steps. With friction the base sticks while the free corners
    integrate freely (the zero-init network has no elastic response, the spec's note on pinned bodies), so the
    body shears and tumbles down-slope; 30 steps, the robust checks only. The tilted-plane equivalence test shows
    the same motion on the plane: it is the untrained solver's, not the static face's."""

    def run_drop(self, mu_f: float, steps: int):
        tilt = math.radians(15)
        n = np.array([-math.sin(tilt), math.cos(tilt), 0.0])
        face = quad((1.5, 0.0, 1.5), normal=n, sides=(30.0, 30.0))
        Rm = rotation_z(tilt)
        b = boxes_with_faces([], offsets=((0.0, 0.0, 0.0),), plane_d=-12.0, kappa=20.0, beta=0.3, mu_f=mu_f)
        c = torch.tensor([1.5, 0.0, 1.5], dtype=torch.float64)
        X = (
            (b.X - c) @ Rm.T + c + 1.1 * torch.tensor(n, dtype=torch.float64)
        )  # the box tilted with the ramp, 1.1 cells up
        b.X, b.X_prev, b.x, b.Y = X, X.clone(), X.clone(), X.clone()
        b.scene.faces = torch.tensor(face, dtype=torch.float64)[None]
        step = Step(Net.from_config(SMALL_CFG).double().eval(), Fusion(), translation="implicit_contact")
        sel = torch.ones(1, dtype=torch.bool)
        step.prepare(b, sel)
        self.assertEqual(b.pairs.count, 0)  # 1.1 cells up: distance - r = 0.6 > the margin r at rest
        nt, p0 = torch.tensor(n, dtype=torch.float64), torch.tensor(face[0], dtype=torch.float64)
        pens, dist, static, centroid = [], [], [], []
        for _ in range(steps):
            for _ in range(2):
                out = step.query(b)
                self.assertTrue(torch.isfinite(out.E_after).all())
                step.commit(b, out)
            step.advance(b, sel)
            self.assertTrue(torch.isfinite(b.X).all() and torch.isfinite(b.E).all())
            pens.append(contact.kind_penetration(b, b.X)[contact.KIND_STATIC].item())
            dist.append(((b.X - p0) @ nt).min().item())  # the lowest corner over the face's plane
            static.append(int(static_rows(b.pairs).sum()))
            centroid.append(((physics.centroid(b, b.X)[0] - p0) @ nt).item())
            self.assertEqual(int(((b.pairs.kind == 0) & b.pairs.valid).sum()), 0)
        return pens, dist, static, centroid

    def test_slides_without_falling_through(self):
        pens, dist, static, centroid = self.run_drop(0.0, 60)
        self.assertLess(next(i for i, v in enumerate(static) if v > 0), 10)  # free fall brings it within reach
        self.assertGreater(static[-1], 0)  # and it is still on the face at the end
        self.assertLess(max(pens), 0.1)  # penetration bounded (0.09 r at the landing, the static value 7e-4 r after)
        self.assertGreater(min(dist), 0.45)  # no corner ever reaches the face's plane (r = 0.5 over it)
        self.assertLess(abs(centroid[-1] - (1.5 + R)), 1e-3)  # resting height: half the box plus r
        self.assertLess(abs(centroid[-1] - centroid[-2]), 1e-4)  # settled in the normal direction

    def test_sticks_and_shears_without_falling_through(self):
        pens, dist, static, _ = self.run_drop(0.8, 30)
        self.assertLess(next(i for i, v in enumerate(static) if v > 0), 10)
        self.assertGreater(static[-1], 0)
        self.assertLess(max(pens), 0.1)
        self.assertGreater(min(dist), 0.0)  # no corner through the face's plane


def kernel_batch(seed: int, variant: str, device=CUDA) -> Batch:
    """A free box over a tilted static quad with the plane below, CUDA float32, capacity pairs; the candidate
    selects the regime as `test_body_contact.kernel_batch`."""
    gen = torch.Generator().manual_seed(seed)
    face = quad((1.5, 0.0, 1.5), normal=(0.25, 1.0, -0.15), sides=(10.0, 10.0), angle=0.3 * seed)
    n = np.array([0.25, 1.0, -0.15]) / np.linalg.norm([0.25, 1.0, -0.15])
    b = boxes_with_faces(
        [face],
        offsets=(tuple(0.6 * n),),
        plane_d=-0.6,
        dtype=torch.float32,
        device=device,
        beta=0.5,
        mu_f=0.6,
        noise=0.05,
        seed=seed,
    )
    b.V = 0.1 * torch.randn(b.N, 3, generator=gen).to(device)
    noise = torch.randn(b.N, 3, generator=gen).to(device)
    push = -torch.tensor(n, dtype=torch.float32, device=device)
    x = b.X.clone()
    if variant == "slip":
        x = x + 0.02 * noise + 0.3 * torch.tensor([1.0, 0.0, 0.5], device=device) + 0.3 * push
    elif variant == "band":
        x = x + 3e-4 * noise + 0.25 * push
    elif variant == "approach":
        x = x + 0.01 * noise + 0.4 * push
    b.x = x
    b.pairs = contact.detect(b, b.X, b.V, capacity=True)
    return b


@unittest.skipUnless(HAS_CUDA, "needs cuda")
class TestKernel(unittest.TestCase):
    """The Warp pair kernel, the fused pass and the geometry kernel against the torch path with static-face rows;
    no partner gradient for a static face."""

    def test_regimes_match_torch(self):
        for seed in range(2):
            for variant in ("slip", "band", "approach", "rest"):
                b = kernel_batch(seed, variant)
                tag = f"seed {seed} {variant}"
                self.assertGreater(int(static_rows(b.pairs).sum()), 0, tag)
                x = b.x.clone().requires_grad_(True)
                E_w = contact.contact_energy(b, x)
                (g_w,) = torch.autograd.grad(E_w.sum(), x)
                with torch_paths(contact_only=True):
                    E_t = contact.contact_energy(b, x)
                    (g_t,) = torch.autograd.grad(E_t.sum(), x)
                self.assertGreater(E_t.abs().max().item(), 0.0, tag)
                self.assertLess(rel_err(E_w, E_t), 1e-5, f"{tag}: energy")
                self.assertLess(rel_err(g_w, g_t), 1e-4, f"{tag}: gradient")
                self.assertTrue(physics.fused_pass_applies(b, b.x))
                E, gX = physics.energy_and_grad(b, b.x)
                with torch_paths():
                    E_ref, g_ref = physics.energy_and_grad(b, b.x)
                self.assertLess(rel_err(E, E_ref), 1e-5, f"{tag}: fused energy")
                self.assertLess(rel_err(gX, g_ref), 1e-4, f"{tag}: fused gradient")

    def test_geometry_kernel_and_tokens(self):
        for seed, variant in ((0, "slip"), (1, "approach")):
            b = kernel_batch(seed, variant)
            Rf = frames(hx.center_deformation(hx.modes(b.x[b.cells], b.hc)), b.R_ref[b.cell_obj])
            geo_w = contact._geometry(b, b.x, b.pairs)
            tok_w = contact.contact_tokens(b, b.x, Rf)
            ke_w, H_w = contact.active_stiffness(b, b.x)
            pen_w = contact.penetration(b, b.x)
            with torch_paths(contact_only=True):
                geo_t = contact._geometry(b, b.x, b.pairs)
                tok_t = contact.contact_tokens(b, b.x, Rf)
                ke_t, H_t = contact.active_stiffness(b, b.x)
                pen_t = contact.penetration(b, b.x)
            valid = b.pairs.valid
            rows = static_rows(b.pairs)
            self.assertGreater(int(rows.sum()), 0)
            for name, gw, gt in zip(("xs", "p", "n", "gap", "r", "delta"), geo_w, geo_t, strict=True):
                self.assertLess(rel_err(gw[valid], gt[valid]), 1e-5, name)
                if name != "r":
                    self.assertEqual(gw[~valid].abs().max().item(), 0.0, name)
            self.assertLess(rel_err(tok_w[valid], tok_t[valid]), 1e-5)
            self.assertTrue((tok_w[rows, 11] == contact.RADIUS_CHANNEL_CAP).all())
            self.assertTrue((tok_w[rows, 15:18].argmax(1) == contact.KIND_STATIC).all())
            self.assertLess(rel_err(ke_w, ke_t), 1e-5)
            self.assertLess(rel_err(H_w, H_t), 1e-4)
            self.assertLess(rel_err(pen_w, pen_t), 1e-5)

    def test_no_partner_scatter_for_static_faces(self):
        """The kernel's partner-corner gradients are exactly zero on static rows, and a second body that touches
        nothing receives no gradient from the first body's static pairs (torch and Warp)."""
        from experiments.lido import contact_kernel

        b = kernel_batch(1, "slip")
        b.pairs.valid &= b.pairs.partner_body == contact.PARTNER_STATIC  # static rows only
        Q = b.pairs.count
        energy = torch.zeros(b.O, device=CUDA)
        grad_pair = torch.empty(Q, 3, device=CUDA)
        grad_partner = torch.full((Q, 4, 3), float("nan"), device=CUDA)
        contact_kernel.launch_contact(b, b.x.contiguous(), b.pairs, R, energy, grad_pair, None, grad_partner)
        self.assertGreater(energy.item(), 0.0)
        self.assertEqual(grad_partner.abs().max().item(), 0.0)
        self.assertGreater(grad_pair.abs().max().item(), 0.0)
        # two bodies: B far away, A on the face: B's corners get nothing
        face = quad((1.5, 0.0, 1.5), sides=(10.0, 10.0))
        b2 = boxes_with_faces(
            [face],
            sides=((3, 3, 3), (2, 2, 2)),
            offsets=((0.0, 0.3, 0.0), (20.0, 5.0, 0.0)),
            dtype=torch.float32,
            device=CUDA,
        )
        b2.pairs = contact.detect(b2, b2.X, b2.V, capacity=True)
        self.assertEqual(int(static_rows(b2.pairs).sum()), 9)
        self.assertEqual(int(b2.pairs.valid.sum()), 9)
        for use_torch in (False, True):
            with torch_paths(contact_only=use_torch):
                x = b2.X.clone().requires_grad_(True)
                (g,) = torch.autograd.grad(contact.contact_energy(b2, x).sum(), x)
            self.assertEqual(g[b2.corner_obj == 1].abs().max().item(), 0.0)
            self.assertGreater(g[b2.corner_obj == 0].abs().max().item(), 0.0)


class TestCapacity(unittest.TestCase):
    def test_same_pairs_same_values(self):
        for seed in range(3):
            face = quad((1.5, 0.0, 1.5), normal=(0.2, 1.0, 0.1), sides=(9.0, 9.0), angle=0.5 * seed)
            n = np.array([0.2, 1.0, 0.1]) / np.linalg.norm([0.2, 1.0, 0.1])
            b = boxes_with_faces(
                [face],
                sides=((3, 3, 3), (3, 3, 3)),
                offsets=(tuple(0.4 * n), tuple(0.4 * n + np.array([0.3, 3.3, -0.2]))),
                plane_d=-0.4,
                noise=0.05,
                seed=seed,
            )
            b.V = 0.1 * torch.randn(b.N, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
            compact = contact.detect(b, b.X, b.V)
            cap = contact.detect(b, b.X, b.V, capacity=True)
            self.assertEqual(cap.count, b.S * (1 + min(contact.M_PAIR, 1 + b.O - 1)))
            self.assertEqual(int(cap.valid.sum()), compact.count)
            self.assertGreater(int(static_rows(compact).sum()), 0)
            self.assertGreater(int((compact.kind == contact.KIND_BODY).sum()), 0)
            for name in (
                "sample",
                "cell",
                "obj",
                "partner_point",
                "partner_normal",
                "kind",
                "radius",
                "anchor",
                "partner_body",
                "partner_face",
            ):
                self.assertTrue(torch.equal(getattr(cap, name)[cap.valid], getattr(compact, name)), name)
            x = (b.X + 0.02 * torch.randn(b.N, 3, dtype=torch.float64)).requires_grad_(True)
            x.data -= 0.1 * torch.tensor(n, dtype=torch.float64)
            Rf = frames(hx.center_deformation(hx.modes(x.detach()[b.cells], b.hc)).float(), b.R_ref[b.cell_obj].float())
            out = []
            for pairs in (compact, cap):
                b.pairs = pairs
                E = contact.contact_energy(b, x)
                (g,) = torch.autograd.grad(E.sum(), x)
                out.append(
                    (E, g, contact.penetration(b, x.detach()), contact.contact_tokens(b, x.detach(), Rf.double()))
                )
            (E1, g1, pen1, tok1), (E2, g2, pen2, tok2) = out
            self.assertGreater(E1.sum().item(), 0.0)
            self.assertLess((E1 - E2).abs().max().item(), 1e-12 * E1.abs().max().item())
            self.assertLess((g1 - g2).abs().max().item(), 1e-12 * g1.abs().max().item())
            self.assertTrue(torch.equal(pen1, pen2))
            self.assertTrue(torch.equal(tok2[cap.valid], tok1))


@unittest.skipUnless(HAS_CUDA, "cuda")
class TestCapturedQueryStaticFaces(FullFloat32):
    """The captured query (capacity pairs, Warp kernels) replays the eager compacted query on a box over a tilted
    static face with the plane below, implicit-contact centroid update; 5e-5 as the body-contact capture test."""

    def states(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        net = small_net(gen, CUDA, SMALL_CFG)
        pair = []
        face = quad((1.5, 0.0, 1.5), normal=(0.2, 1.0, -0.1), sides=(12.0, 12.0), angle=0.4)
        n = np.array([0.2, 1.0, -0.1]) / np.linalg.norm([0.2, 1.0, -0.1])
        for capacity in (False, True):
            b = boxes_with_faces(
                [face],
                offsets=(tuple(0.3 * n),),
                plane_d=-0.6,
                dtype=torch.float32,
                device=CUDA,
                noise=0.03,
                kappa=5.0,
                seed=seed,
            )
            gen_b = torch.Generator().manual_seed(seed + 100)
            b.V = 0.1 * torch.randn(b.N, 3, generator=gen_b).to(CUDA)
            b.X_prev = b.X - b.V
            step = Step(net, Fusion(), pair_capacity=capacity, translation="implicit_contact")
            step.prepare(b, torch.ones(1, dtype=torch.bool, device=CUDA))
            pair.append((b, step))
        return pair

    def assert_same(self, b1, b2, tag, tol=5e-5):
        for name in ("x", "E", "gX", "hist_grad", "hist_update", "picard_constant"):
            a, c = getattr(b1, name), getattr(b2, name)
            err = (a - c).abs().max().item()
            self.assertLess(err, tol * max(1.0, c.abs().max().item()), f"{tag}: {name} differs by {err:.3e}")

    def test_three_queries_and_an_advance(self):
        (b1, s1), (b2, s2) = self.states()
        self.assertGreater(int(static_rows(b2.pairs).sum()), 0)
        self.assertGreater(int(((b2.pairs.kind == 0) & b2.pairs.valid).sum()), 0)
        sel = torch.ones(1, dtype=torch.bool, device=CUDA)
        with torch.no_grad():
            cq = CapturedQuery(s2, b2)
            torch.cuda.synchronize()
            self.assert_same(b1, b2, "before")
            for k in range(3):
                s1.commit(b1, s1.query(b1))
                cq.replay()
                torch.cuda.synchronize()
                self.assert_same(b1, b2, f"query {k}")
            s1.advance(b1, sel)
            s2.advance(b2, sel)
            cq.sync()
            self.assertIs(b2.pairs, cq.pairs)
            self.assertGreater(int(static_rows(b2.pairs).sum()), 0)
            self.assert_same(b1, b2, "after advance")
            s1.commit(b1, s1.query(b1))
            cq.replay()
            torch.cuda.synchronize()
            self.assert_same(b1, b2, "query after advance")


class TestSceneRunner(unittest.TestCase):
    """An epoch of small scenes with 4-8 static faces each on the CPU, the pair counts and validation records with
    the `static` key, and a face injected under a resting body pairing at step 0."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = scene_cfg(static_face_count_range=(4, 8))

    def test_epoch_with_static_faces(self):
        cfg = self.cfg
        runner, step = make_runner_cpu(cfg)
        runner.start_epoch(1)
        self.assertGreaterEqual(runner.batch.scene.faces.shape[0], 4)
        self.assertIsNotNone(runner.batch.static_mesh)
        faces_seen = []
        for _ in range(runner.U):
            self.assertIsNotNone(runner.job)
            faces_seen.append(int(runner.batch.scene.faces.shape[0]))
            self.assertEqual(faces_seen[-1], len(runner.scene.static_faces))
            runner.commit(step.query(runner.batch))
        self.assertTrue(all(4 <= f <= 8 for f in faces_seen))
        self.assertEqual(runner.loaded_jobs, cfg.scene_count)
        self.assertEqual(runner.failures, [])
        self.assertTrue(all("static" in p for p in runner.pair_history))
        summary = runner.epoch_summary()
        self.assertGreaterEqual(summary["static_faces_mean"], 4.0)
        self.assertIn("static_pairs_mean", summary)
        self.assertTrue(all(s["static_faces"] >= 4 for s in summary["scenes_served"]))
        records = V.validate_cheap_v5(step, cfg, runner.grids, runner.aug, "cpu", RUNNER_MASTER)
        for r in records:
            self.assertTrue(r["survived"])
            self.assertIn("static", r["contact_pairs"])
            self.assertGreaterEqual(r["scene"]["static_faces"], 4)
        full, _ = V.validate_full_horizon_v5(step, cfg, runner.grids, runner.aug, "cpu", RUNNER_MASTER, 1, 2)
        self.assertTrue(all(r["survived"] for r in full))
        self.assertTrue(math.isfinite(full[0]["momentum_drift"]) and full[0]["momentum_drift"] < 1e-3)

    def test_injected_face_under_a_resting_body(self):
        cfg = self.cfg
        grids, aug = GridCache("cpu"), Augmenter("cpu")
        step = Step(Net.from_config(cfg), Fusion(), aug)
        scene = None
        for seed in range(32):
            sc = S.sample_scene(RUNNER_MASTER, seed, cfg, validation=True)
            lo, _ = body_aabbs(sc)
            ground = [
                o for o, b in enumerate(sc.bodies) if b.resting and abs(lo[o, 1] / sc.h - S.RESTING_GAP_CELLS) < 1e-9
            ]
            if ground:
                scene, o = sc, ground[0]
                break
        self.assertIsNotNone(scene, "no scene with a body resting on the ground in 32 seeds")
        body = scene.bodies[o]
        h = scene.h
        # a slab 0.1 cells above the ground under the resting body (its bottom samples sit at r over the ground)
        slab = quad((body.position[0] / h, 0.1, body.position[2] / h), sides=(14.0, 14.0)) * h
        injected = dataclasses.replace(
            scene, static_faces=[*scene.static_faces, [tuple(float(v) for v in c) for c in slab]]
        )
        b = scene_batch(injected, grids, aug, "cpu")
        self.assertEqual(b.scene.faces.shape[0], len(scene.static_faces) + 1)
        step.prepare(b, b.active, None)
        p = b.pairs
        own_static = static_rows(p) & (p.obj == o)
        self.assertGreater(int(own_static.sum()), 0)
        self.assertTrue((p.partner_face[own_static] == len(scene.static_faces)).all())
        _, _, _, gap, _, _ = contact._geometry(b, b.X, p)
        # every sample of the face it lies on pairs with the slab at gap 0.4 (the lowest side row grazes at 0.9)
        down = S.rotation_matrix(body.quaternion).T @ np.array([0.0, -1.0, 0.0])
        axis = int(np.argmax(np.abs(down)))
        n_bottom = int(np.prod([c for i, c in enumerate(body.cell_counts) if i != axis]))
        bottom = own_static & (gap < R)
        self.assertEqual(int(bottom.sum()), n_bottom)
        self.assertTrue(torch.allclose(gap[bottom], torch.full((n_bottom,), 0.4, dtype=gap.dtype), atol=1e-5))
        self.assertGreaterEqual(gap[own_static & ~bottom].min().item(), 0.9 - 1e-5)
        counts = pair_counts(b)
        self.assertEqual(counts["static"], int(static_rows(p).sum()))
        self.assertGreater(counts["plane"], 0)
        metrics = V.scene_metrics(b)
        self.assertEqual(metrics["contact_pairs"], counts)
        by_kind = contact.kind_penetration(b, b.x)
        self.assertEqual(metrics["plane_penetration_r"], max(by_kind[0].item(), by_kind[1].item()))
        self.assertGreaterEqual(metrics["plane_penetration_r"], (R - 0.4) / R - 1e-4)  # at least the slab's d = r - gap
        # the contact-free copy drops the faces
        free = V.contact_free_copy(injected, 8)
        self.assertEqual(free.static_faces, [])
        self.assertEqual(scene_batch(free, grids, aug, "cpu", plane=False).scene.faces.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
