# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""v5 scenes (design spec section 11; Anka's note `notes/v5-free-motion-with-contact.md`): one scene per batch,
cuboids of random size, material, pose and velocity above a ground plane, all in ONE world frame; most bodies are
free, a fraction is clamped at one face and held at its initial pose (anchors and hanging beams).

`sample_scene` draws a scene in SI from numpy seed streams: a pure function of (master_seed, scene_seed, validation)
like `jobs.sample_scene_spec`, with the validation flag selecting a separate stream so held-out scene indices never
collide with training scenes. `realise` converts it to the solver's normalised units (h = dt = 1): box grids with
the body's pins ("none" or the clamped face), X = the rotated and translated rest lattice plus the multiscale
deformation field (pinned corners exactly at the rigid pose), V = the rigid velocity in cells per step plus the
deformation velocity field (zero on pinned rows, rigid velocity zero for a pinned body), one Material per body
(gravity along -y, the scene's contact constants) and a ContactScene with the ground plane for every body, no
static points and the scene's static faces (`ContactScene.faces`, cell units). All objects share the world frame with origin zero, so plane_d = plane_height / h is the same
number for every object.

Draws (per scene stream): drift direction (standard normal with the vertical component scaled by
`DRIFT_VERTICAL_SCALE`, normalised), drift speed U(drift_speed_range), kappa log-uniform over contact_kappa_range,
beta, mu_f. Per body (body stream, until the cell total reaches `scene_cells`; the last body may exceed it): sides
U{body_sides} per axis independently, E, nu, rho, eta, perturbation_scale, strength, velocity_dt with the ranges and
draw order of `jobs.sample_scene_spec` (no gravity draw: the scene's gravity is `cfg.gravity`), a velocity noise of
magnitude U(0, drift speed) in a uniform random direction added to the drift, and the deformation-field seed.
The contact stiffness ratio is floored by the load rule of the contact note (section 7, amendment 2026-09-29) taken
over every body: kappa >= m_b g / (n_face_b d_max E_b h), which the heaviest and softest bodies decide.

Pinned bodies (Anka, 2026-10-02; pin stream): every body draws `pinned` with probability `cfg.pinned_body_fraction`
and one of its six faces (`grid.FACE_PINS`, uniform) from a fourth seed stream, so the body, placement and scene
draws are those of the all-free generator (fraction 0 reproduces it exactly). A pinned body is placed like any other
(random pose above the ground) and simply stays there: its clamped face is held at the initial pose by the solver
(`Step.prepare` / `advance` keep pinned rows at X with zero velocity), its rigid velocity is zero (`BodySpec.velocity`),
and the fusion takes the Dirichlet solve of its pinned grid while the free bodies take the translation-free solve
with the centroid update; free and pinned grids never share a fusion group.

Resting bodies (Anka, 2026-10-02; fifth seed stream): a fraction `cfg.resting_body_fraction` of the FREE bodies
starts at rest in contact. Every body draws (u, face, yaw) from the resting stream whatever the fraction (so the other
streams are untouched and fraction 0 reproduces the earlier scenes exactly); a free body with u < fraction rests. A
body at rest lies flat on one of its six faces (`face`, uniform) with a random yaw about the vertical (its orientation
is random within what a body at rest allows: a cuboid balanced on a corner is not at rest), carries no rigid velocity
and no deformation field (`perturbation_scale` 0: the field's ~0.4-cell RMS would put its bottom samples half a cell
into the support, far beyond the static penetration of ~1e-3 cells, and the body would jump), and sits with its
bottom face at the contact gap zero of the sample-sphere model: the face centres' spheres of radius r = h/2 touch the
support (gap = r, penetration d = r - gap = 0; `RESTING_GAP_CELLS`), so detection finds the pairs at step 0 and the
body settles by the static penetration only. The support is the ground, or, when a pinned body placed before it lies
under the candidate footprint, that body's upper surface at the closest approach along -y (`support_height`: the exact
maximum of the tilted box's upper envelope over the flat body's footprint; the bounding boxes only select the
candidates). The pinned body counts as its box inflated on every side by a clearance of `CLEARANCE_SIGMAS` times the
RMS of its deformation field (its free corners carry the field, which the SI placement cannot see: without the
clearance the deformed support penetrated the resting body by up to 2 RMS, from below and from the side where a
tilted box rises past the resting body's edge). The grown-bounding-box rule still separates the body from everything
that is not its support; against the inflated boxes of its supports a separating-axis test verifies that the boxes do
not intersect (`boxes_overlap`).

Static faces (Anka, 2026-10-02; sixth seed stream): `static_face_count` in U{cfg.static_face_count_range} planar
quads per scene (walls, ramps, slabs) as fixed contact partners: side lengths U(cfg.static_face_size_cells) cells,
a uniform random unit normal (normalised standard normal) and an in-plane rotation U(0, 2 pi), the centre uniform
over the placement column's footprint and between the ground and `placement_height`. The faces are placed AFTER all
bodies (the body draws and the body placement are those of the generator without faces, so a face count of 0
reproduces the earlier scenes exactly, and no body is ever placed after a face): a candidate is rejected while any
corner lies below the ground or the quad intersects any body's initial axis-aligned bounding box grown by one cell
(or by the body's deformation clearance when that is larger; `quad_box_overlap`, a separating-axis test); after
`MAX_FACE_DRAWS` rejected positions the face is dropped (`placement["static_faces_dropped"]`). The corners are
stored in metres in order around the quad (`SceneV5.static_faces`; the quad normal is the unit diagonal cross
product (c2 - c0) x (c3 - c1), the convention of the bodies' faces), and `realise` hands them to
`ContactScene.faces` in cell units, shared by every body of the scene.

Placement (placement stream): per body a uniform random quaternion (Shoemake) and a gap U{placement_gap_cells}
cells, then bodies are placed one at a time in a square column of side `footprint` (m) centred on the origin.
The body's lowest point is uniform between 2 cells and `placement_height` above the plane (the FIRST body: between
1 and 2 cells, so at least one body reaches the ground within a few steps), the horizontal position uniform with
the body's axis-aligned bounding box inside the column. A candidate is rejected while its bounding box grown by
the body's gap on every side overlaps a placed body's bounding box, so two bodies' boxes are at least the gap
apart. The footprint is derived from the bodies: the grown bounding-box volumes sum to `PLACEMENT_FILL` of the
column over the vertical band the boxes can occupy; when a body is not placed within `MAX_PLACEMENT_DRAWS` draws the
footprint grows by `FOOTPRINT_GROWTH` and the draws restart (counted in `scene.placement`). Measured on the default
configuration (64000 cells, 135-166 bodies, 16 scenes, 40 ms per scene): footprint 6.4-7.1 m, 17-31 % of the position
draws accepted (mean 22 %), one footprint growth in 2 of the 16 scenes; nearest bounding-box separation median 3.4
cells (minimum the gap), bodies' lowest points 2-20 cells above the ground, the first body's 1-2 cells. The bodies
(1.0 m^3 of material) land in a single layer on a 40-50 m^2 floor; the fill constant trades acceptance for density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .augment import DISPLACEMENT_SCALE
from .grid import FACE_PINS
from .jobs import _log_uniform
from .scenes import PLANE_NORMAL, contact_stiffness_floor
from .structs import ContactScene, Material
from .units import material_from_si, reference_modulus

Tensor = torch.Tensor

STREAM_TAG = 5  # last SeedSequence key: v5 scene streams never coincide with the body-mode streams of jobs.py
EPOCH_SEED_STRIDE = 1 << 20  # training scene seed = epoch x stride + scene index: fresh scenes every epoch


@dataclass(frozen=True)
class SceneMix:
    """The scene composition of an epoch (scene curriculum, Anka 2026-10-02): the probability that a body is
    pinned and the fraction of the free bodies placed at rest."""

    pinned_fraction: float
    resting_fraction: float


def scene_mix(cfg, epoch: int | None = None) -> SceneMix:
    """The mix of `epoch` (1-based): the config's `pinned_body_fraction` and `resting_body_fraction` when the
    curriculum is off or `epoch` is None (held-out scenes of the goal); otherwise every body pinned and nothing
    resting through `scene_curriculum_epochs[0]`, a linear ramp to the config's values at `scene_curriculum_epochs[1]`
    and the config's values from then on."""
    target = SceneMix(float(cfg.pinned_body_fraction), float(cfg.resting_body_fraction))
    if epoch is None or not getattr(cfg, "scene_curriculum", False):
        return target
    e0, e1 = (int(v) for v in cfg.scene_curriculum_epochs)
    p = 1.0 if epoch >= e1 else 0.0 if epoch <= e0 else (epoch - e0) / (e1 - e0)
    return SceneMix(1.0 - (1.0 - target.pinned_fraction) * p, target.resting_fraction * p)


def epoch_scene_seed(epoch: int, index: int) -> int:
    """The training scene seed of scene `index` in `epoch`: distinct scenes every epoch (seeds below
    EPOCH_SEED_STRIDE are the fixed scenes of the tests and of the reference file)."""
    if not 0 <= index < EPOCH_SEED_STRIDE:
        raise ValueError(f"scene index {index} is outside [0, {EPOCH_SEED_STRIDE})")
    return int(epoch) * EPOCH_SEED_STRIDE + int(index)


DRIFT_VERTICAL_SCALE = 0.25  # the drift direction's y component is scaled by this before normalisation
PLACEMENT_FILL = 0.6  # grown bounding-box volume over the column volume that sets the footprint
MAX_PLACEMENT_DRAWS = 200  # position draws per body before the footprint grows
FOOTPRINT_GROWTH = 1.1
GROUND_CELLS = 2.0  # lowest point of a body at least this many cells above the plane (the first body: 1-2 cells)
FIRST_BODY_CELLS = (1.0, 2.0)
RESTING_GAP_CELLS = 0.5  # a resting body's bottom face sits this far above its support: the sample radius r = h / 2
CLEARANCE_SIGMAS = 3.0  # a pinned support's deformation field (RMS) times this clears the resting body above it
MAX_FACE_DRAWS = 200  # position draws per static face before it is dropped
MAX_PINNED_FACE_DRAWS = 8  # side and gap draws per pinned contact face before it is dropped
PINNED_FACE_GAP = (0.2, 0.8)  # gap of a pinned contact face from the body's face, in units of the field clearance
PINNED_FACE_SCALE = (0.8, 1.5)  # sides of a pinned contact face over the sides of the body's face
PINNED_FACE_OFFSET = 0.25  # in-plane offset of a pinned contact face, in units of the body's face sides
FACE_GROWTH_CELLS = 1.0  # a static face keeps this many cells (at least) from every body's initial bounding box
FACE_NORMALS = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)  # local outward normals, grid order
_GEOM_TOL = 1e-9  # m: tolerance of the support and overlap geometry


@dataclass
class BodySpec:
    """One cuboid of a v5 scene (SI)."""

    cell_counts: tuple  # (nx, ny, nz) cells
    material: dict  # E, nu, rho, eta
    position: tuple  # [3] m: world position of the rest lattice's centre
    quaternion: tuple  # [4] (x, y, z, w) unit quaternion; world = R(q) body
    velocity: tuple  # [3] m/s rigid velocity (drift + body noise)
    strength: float
    velocity_dt: float
    perturbation_scale: float
    seed: int  # deformation-field seed
    pins: str = "none"  # "none" (free body) or the clamped lattice face, one of `grid.FACE_PINS` (held at the pose)
    resting: bool = False  # free body starting at rest on the ground or on a pinned body (flat, no velocity, no field)

    @property
    def cells(self) -> int:
        return int(math.prod(self.cell_counts))

    @property
    def pinned(self) -> bool:
        return self.pins != "none"


@dataclass
class SceneV5:
    """A v5 scene in SI; JSON-serialisable through `dataclasses.asdict`, rebuilt by `SceneV5.from_dict`."""

    seed: int
    validation: bool
    h: float
    dt: float
    gravity: tuple
    bodies: list
    plane_height: float  # m: the ground is the y = plane_height plane (0: the world's y = 0)
    drift: tuple  # [3] m/s scene-wide drift velocity
    contact: dict  # kappa (floored), kappa_drawn, kappa_floor, floor_bound, beta, mu_f, friction_epsilon, floor_scale
    placement: dict = field(default_factory=dict)  # footprint, draws, acceptance_rate, footprint_growths, static_face_*
    static_faces: list = field(default_factory=list)  # [F][4][3] m: corners of the static quads, in order around each

    @property
    def cells(self) -> int:
        return sum(b.cells for b in self.bodies)

    @staticmethod
    def from_dict(d: dict) -> SceneV5:
        bodies = [
            BodySpec(
                cell_counts=tuple(int(v) for v in b["cell_counts"]),
                material=dict(b["material"]),
                position=tuple(float(v) for v in b["position"]),
                quaternion=tuple(float(v) for v in b["quaternion"]),
                velocity=tuple(float(v) for v in b["velocity"]),
                strength=float(b["strength"]),
                velocity_dt=float(b["velocity_dt"]),
                perturbation_scale=float(b["perturbation_scale"]),
                seed=int(b["seed"]),
                pins=str(b.get("pins", "none")),
                resting=bool(b.get("resting", False)),
            )
            for b in d["bodies"]
        ]
        return SceneV5(
            seed=int(d["seed"]),
            validation=bool(d["validation"]),
            h=float(d["h"]),
            dt=float(d["dt"]),
            gravity=tuple(float(v) for v in d["gravity"]),
            bodies=bodies,
            plane_height=float(d["plane_height"]),
            drift=tuple(float(v) for v in d["drift"]),
            contact=dict(d["contact"]),
            placement=dict(d.get("placement", {})),
            static_faces=[[tuple(float(v) for v in c) for c in f] for f in d.get("static_faces", [])],
        )


# ---------------------------------------------------------------------------------------------------- geometry
def random_quaternion(rng: np.random.Generator) -> tuple:
    """Uniform random unit quaternion (x, y, z, w) (Shoemake's method, three uniforms)."""
    u1, u2, u3 = rng.random(3)
    a, b = math.sqrt(1.0 - u1), math.sqrt(u1)
    return (
        a * math.sin(2 * math.pi * u2),
        a * math.cos(2 * math.pi * u2),
        b * math.sin(2 * math.pi * u3),
        b * math.cos(2 * math.pi * u3),
    )


def rotation_matrix(q) -> np.ndarray:
    """Rotation matrix [3,3] of the unit quaternion (x, y, z, w): world = R body."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def half_extents(cell_counts, q, h: float) -> np.ndarray:
    """Half extents [3] (m) of the axis-aligned bounding box of the rotated box, centred on the lattice centre."""
    R = rotation_matrix(q)
    return 0.5 * h * np.abs(R) @ np.asarray(cell_counts, dtype=float)


def quaternion_from_matrix(R: np.ndarray) -> tuple:
    """Unit quaternion (x, y, z, w) of a rotation matrix (the largest component computed first, the rest from it)."""
    t = float(np.trace(R))
    if t > 0:
        w = math.sqrt(1.0 + t) / 2
        x, y, z = (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = s / 4
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = v
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def face_down_quaternion(face: int, yaw: float) -> tuple:
    """The pose of a body lying flat on its lattice face `face` (0..5: -x, +x, -y, +y, -z, +z) with a yaw about the
    vertical: R = R_y(yaw) R_0 with R_0 the rotation taking the face's outward normal to -y."""
    n = np.asarray(FACE_NORMALS[face], dtype=float)
    down = np.array([0.0, -1.0, 0.0])
    c = float(n @ down)
    if c > 1 - 1e-12:
        R0 = np.eye(3)
    elif c < -1 + 1e-12:
        R0 = np.diag([1.0, -1.0, -1.0])  # the +y face: half a turn about x
    else:
        v = np.cross(n, down)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R0 = np.eye(3) + vx + vx @ vx / (1 + c)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return quaternion_from_matrix(Ry @ R0)


def box_corners(centre, R: np.ndarray, half_sides) -> np.ndarray:
    """The 8 corners [8,3] (m) of a box with centre, rotation (world = R body) and half sides (m)."""
    signs = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
    return np.asarray(centre, dtype=float)[None] + (signs * np.asarray(half_sides, dtype=float)) @ R.T


_BOX_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1)  # 12 corner pairs


def _upper_envelope(x: float, z: float, centre, R: np.ndarray, half_sides) -> float | None:
    """Height of the box's upper surface over the vertical line through (x, z) (None when the line misses it):
    the line in the box frame is a + t b, clipped against the three slabs |.| <= half side."""
    a = R.T @ (np.array([x, 0.0, z]) - centre)
    b = R.T @ np.array([0.0, 1.0, 0.0])
    t_lo, t_hi = -math.inf, math.inf
    for i in range(3):
        if abs(b[i]) < 1e-15:
            if abs(a[i]) > half_sides[i] + _GEOM_TOL:
                return None
            continue
        t1, t2 = (-half_sides[i] - a[i]) / b[i], (half_sides[i] - a[i]) / b[i]
        t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
    return t_hi if t_hi >= t_lo - _GEOM_TOL else None


def _inside_convex(p: np.ndarray, poly: np.ndarray) -> bool:
    """Point [2] inside (or on the boundary of) the convex polygon poly [m,2] given in order."""
    side = 0.0
    for i in range(poly.shape[0]):
        a, b = poly[i], poly[(i + 1) % poly.shape[0]]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) <= _GEOM_TOL * (1.0 + np.abs(b - a).sum()):
            continue
        if side == 0.0:
            side = cross
        elif side * cross < 0:
            return False
    return True


def _segment_crossing(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Intersection point [2] of the segments p-q and a-b, None when they do not cross."""
    d1, d2 = q - p, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-15:
        return None
    r = a - p
    t = (r[0] * d2[1] - r[1] * d2[0]) / den
    u = (r[0] * d1[1] - r[1] * d1[0]) / den
    if -_GEOM_TOL <= t <= 1 + _GEOM_TOL and -_GEOM_TOL <= u <= 1 + _GEOM_TOL:
        return p + t * d1
    return None


def support_height(footprint: np.ndarray, centre, R: np.ndarray, half_sides) -> float | None:
    """Closest approach along -y of a flat-bottomed body onto a box: the maximum (m) of the box's upper surface over
    the body's footprint (a convex quad [4,2] in (x, z), in order), None when the footprint and the box's projection
    do not meet. The upper envelope of a convex box is concave, so the maximum over the convex overlap region sits
    at a vertex of the arrangement: a box corner inside the footprint, a footprint corner over the box, or a crossing
    of a projected box edge with a footprint edge."""
    corners = box_corners(centre, R, half_sides)
    pts = [c[[0, 2]] for c in corners if _inside_convex(c[[0, 2]], footprint)]
    pts += list(footprint)
    for i, j in _BOX_EDGES:
        for k in range(4):
            x = _segment_crossing(corners[i][[0, 2]], corners[j][[0, 2]], footprint[k], footprint[(k + 1) % 4])
            if x is not None:
                pts.append(x)
    heights = [_upper_envelope(float(p[0]), float(p[1]), np.asarray(centre, dtype=float), R, half_sides) for p in pts]
    heights = [v for v in heights if v is not None]
    return max(heights) if heights else None


def boxes_overlap(c1, R1: np.ndarray, s1, c2, R2: np.ndarray, s2) -> bool:
    """Two oriented boxes intersect (separating-axis test over the 15 candidate axes; touching counts as free)."""
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            a = np.cross(R1[:, i], R2[:, j])
            n = float(np.linalg.norm(a))
            if n > 1e-12:
                axes.append(a / n)
    d = np.asarray(c2, dtype=float) - np.asarray(c1, dtype=float)
    for a in axes:
        r1 = float(np.abs(R1.T @ a) @ np.asarray(s1, dtype=float))
        r2 = float(np.abs(R2.T @ a) @ np.asarray(s2, dtype=float))
        if abs(float(d @ a)) >= r1 + r2 - _GEOM_TOL:
            return False
    return True


def static_face_corners(centre, normal, angle: float, sides) -> np.ndarray:
    """Corners [4,3] (m) of a planar quad with the given centre, unit normal n, in-plane rotation and side lengths
    (a, b), in order around the quad: c0 = centre - a/2 u - b/2 v, c1 = + a/2 u - b/2 v, c2 = + a/2 u + b/2 v,
    c3 = - a/2 u + b/2 v with u, v an orthonormal in-plane frame, v = n x u, so that the diagonal cross product
    (c2 - c0) x (c3 - c1) = 2 a b n points along the given normal (the convention of `contact.quad_normals`)."""
    n = _unit(np.asarray(normal, dtype=float))
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u0 = _unit(np.cross(n, helper))
    v0 = np.cross(n, u0)
    u = math.cos(angle) * u0 + math.sin(angle) * v0
    v = np.cross(n, u)
    a, b = 0.5 * float(sides[0]), 0.5 * float(sides[1])
    c = np.asarray(centre, dtype=float)
    return np.stack([c - a * u - b * v, c + a * u - b * v, c + a * u + b * v, c - a * u + b * v])


def quad_boxes_overlap(corners: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[n] bool: the planar quad `corners` [4,3] intersects each axis-aligned box [lo[j], hi[j]] ([n,3] each);
    separating-axis test over the three box axes, the quad normal and the cross products of the box axes with the
    two quad edge directions (touching counts as free)."""
    lo, hi = np.asarray(lo, dtype=float).reshape(-1, 3), np.asarray(hi, dtype=float).reshape(-1, 3)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)  # [n,3]
    corners = np.asarray(corners, dtype=float)
    e1, e2 = corners[1] - corners[0], corners[3] - corners[0]
    axes = [np.eye(3)[i] for i in range(3)] + [np.cross(e1, e2)]
    for i in range(3):
        for e in (e1, e2):
            axes.append(np.cross(np.eye(3)[i], e))
    axes = np.stack(axes)  # [A,3]
    norm = np.linalg.norm(axes, axis=1)
    axes = axes[norm > 1e-12] / norm[norm > 1e-12, None]
    proj = corners @ axes.T  # [4,A]
    centre_proj = centre @ axes.T  # [n,A]
    r_box = np.abs(axes) @ half.T  # [A,n]
    q_min, q_max = proj.min(0)[None, :] - centre_proj, proj.max(0)[None, :] - centre_proj  # [n,A]
    separated = (q_max <= -r_box.T + _GEOM_TOL) | (q_min >= r_box.T - _GEOM_TOL)
    return ~separated.any(axis=1)


def quad_box_overlap(corners: np.ndarray, lo, hi) -> bool:
    """`quad_boxes_overlap` for one box."""
    return bool(quad_boxes_overlap(corners, np.asarray(lo)[None], np.asarray(hi)[None])[0])


def _place_pinned_faces(rng: np.random.Generator, cfg, h: float, drawn, centres, quats, pins, clearance, lo, hi):
    """Static faces within the deformation reach of pinned bodies (Anka, 2026-10-02: "pinned + artificial collision"
    as in the v4 campaign, whose plane stood where the swinging beam would hit it; a pinned body neither drifts nor
    falls, so the regular faces, kept a cell plus the clearance away, never touch it). Every pinned body draws one
    uniform number (so the fraction moves no other body's face) and, below `cfg.pinned_contact_face_fraction`, one
    quad parallel to one of its five unpinned faces at a gap of U(PINNED_FACE_GAP) x its field clearance (three
    sigma of the deformation field) from that face, covering it with sides U(PINNED_FACE_SCALE) x the face's sides
    and an in-plane offset of U(-PINNED_FACE_OFFSET, PINNED_FACE_OFFSET) x the sides; a quad below the ground or
    intersecting another body's grown box is redrawn (side, gap, size, offset) up to MAX_PINNED_FACE_DRAWS times,
    then dropped. Returns the corner arrays [4,3] (m, the order of `static_face_corners`), the owner index of each
    and the statistics."""
    p = float(getattr(cfg, "pinned_contact_face_fraction", 0.0) or 0.0)
    faces, owners, dropped, candidates = [], [], 0, 0
    index = np.arange(len(drawn))
    for i, (body, pin) in enumerate(zip(drawn, pins, strict=True)):
        u = float(rng.random())
        if pin == "none" or u >= p:
            continue
        candidates += 1
        R = rotation_matrix(quats[i])
        half = 0.5 * h * np.asarray(body["cell_counts"], dtype=float)
        c = np.asarray(centres[i], dtype=float)
        others = index != i
        placed = False
        for _attempt in range(MAX_PINNED_FACE_DRAWS):
            k = int(rng.integers(0, len(FACE_NORMALS)))
            if FACE_PINS[k] == pin:
                continue
            axis = k // 2
            n = R @ np.asarray(FACE_NORMALS[k], dtype=float)
            ia, ib = [j for j in range(3) if j != axis]
            gap = float(rng.uniform(*PINNED_FACE_GAP)) * float(clearance[i])
            scale = rng.uniform(*PINNED_FACE_SCALE, size=2)
            offset = rng.uniform(-PINNED_FACE_OFFSET, PINNED_FACE_OFFSET, size=2)
            u_vec, v_vec = R[:, ia], R[:, ib]
            centre = (
                c + (half[axis] + gap) * n + 2.0 * offset[0] * half[ia] * u_vec + 2.0 * offset[1] * half[ib] * v_vec
            )
            a, b = scale[0] * half[ia], scale[1] * half[ib]
            corners = np.stack(
                [
                    centre - a * u_vec - b * v_vec,
                    centre + a * u_vec - b * v_vec,
                    centre + a * u_vec + b * v_vec,
                    centre - a * u_vec + b * v_vec,
                ]
            )
            if corners[:, 1].min() < 0.0:
                continue
            if others.any() and bool(quad_boxes_overlap(corners, lo[others], hi[others]).any()):
                continue
            faces.append(corners)
            owners.append(i)
            placed = True
            break
        dropped += int(not placed)
    stats = {
        "pinned_contact_faces": len(faces),
        "pinned_contact_faces_dropped": int(dropped),
        "pinned_contact_candidates": int(candidates),
        "pinned_contact_face_owners": owners,
    }
    return faces, stats


def _place_faces(rng: np.random.Generator, cfg, h: float, footprint: float, lo: np.ndarray, hi: np.ndarray):
    """The scene's static faces (module docstring): a list of corner arrays [4,3] (m) and the statistics. `lo`,
    `hi` [n,3] are the bodies' bounding boxes already grown by the clearance a face must keep."""
    lo_n, hi_n = (int(v) for v in cfg.static_face_count_range)
    size_lo, size_hi = (float(v) for v in cfg.static_face_size_cells)
    count = int(rng.integers(lo_n, hi_n + 1))
    faces, draws, dropped = [], 0, 0
    half = 0.5 * footprint
    for _ in range(count):
        sides = rng.uniform(size_lo, size_hi, size=2) * h
        normal = _unit(rng.normal(size=3))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        placed = False
        for _attempt in range(MAX_FACE_DRAWS):
            draws += 1
            u = rng.random(3)
            centre = np.array([(2.0 * u[0] - 1.0) * half, u[1] * cfg.placement_height, (2.0 * u[2] - 1.0) * half])
            corners = static_face_corners(centre, normal, angle, sides)
            if corners[:, 1].min() < 0.0:  # a corner below the ground
                continue
            if lo.shape[0] > 0 and bool(quad_boxes_overlap(corners, lo, hi).any()):
                continue
            faces.append(corners)
            placed = True
            break
        dropped += int(not placed)
    stats = {
        "static_face_draws": int(draws),
        "static_face_acceptance": float(len(faces) / max(draws, 1)),
        "static_faces_dropped": int(dropped),
    }
    return faces, stats


def rigid_pose(body: BodySpec, rest: Tensor, h: float) -> Tensor:
    """The rotated and translated rest lattice [P,3] in cell units (float64): R(q) (rest - centre) + position / h."""
    R = torch.tensor(rotation_matrix(body.quaternion), dtype=torch.float64, device=rest.device)
    centre = torch.tensor(body.cell_counts, dtype=torch.float64, device=rest.device) / 2
    p = torch.tensor(body.position, dtype=torch.float64, device=rest.device) / h
    return (rest.to(torch.float64) - centre) @ R.T + p


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-300)


def _place(
    rng: np.random.Generator,
    extents: np.ndarray,
    gaps: np.ndarray,
    h: float,
    cfg,
    resting: list | None = None,
    pinned: list | None = None,
    rotations: list | None = None,
    half_sides: np.ndarray | None = None,
    clearance: np.ndarray | None = None,
) -> tuple[list, dict]:
    """Body centres [n][3] (m) by rejection sampling of grown bounding boxes (module docstring) and the statistics.

    `resting` [n] bool selects the bodies placed at rest (module docstring): their height follows from the support,
    the ground or the upper surface of the pinned bodies (`pinned` [n] bool) already placed under the footprint
    (`rotations` [n] [3,3] and `half_sides` [n,3] m describe the boxes, `clearance` [n] m inflates a pinned body's
    box on every side by the room its deformation field needs); the grown-box rule applies to every other placed
    body and `boxes_overlap` verifies the supports. Without `resting` the rule is the earlier one exactly."""
    n = extents.shape[0]
    ground = GROUND_CELLS * h
    if cfg.placement_height <= ground:
        raise ValueError(f"placement_height {cfg.placement_height} m must exceed {GROUND_CELLS} cells = {ground} m")
    grown = extents + gaps[:, None]
    band = (cfg.placement_height - ground) + 2.0 * float(extents[:, 1].mean())
    footprint = max(
        math.sqrt(float(np.prod(2.0 * grown, axis=1).sum()) / (PLACEMENT_FILL * band)),
        float((2.0 * grown[:, [0, 2]]).max()),
    )
    resting = [False] * n if resting is None else list(resting)
    lo_placed, hi_placed = np.zeros((0, 3)), np.zeros((0, 3))
    centres = []
    supports: list[list] = []  # per placed body: the indices of the pinned bodies it rests on
    draws = growths = 0
    for b in range(n):
        e, g = extents[b], float(gaps[b])
        y_min, y_max = (FIRST_BODY_CELLS[0] * h, FIRST_BODY_CELLS[1] * h) if b == 0 else (ground, cfg.placement_height)
        attempts = 0
        while True:
            draws += 1
            attempts += 1
            u = rng.random(3)
            half = 0.5 * footprint
            cx = (2.0 * u[0] - 1.0) * max(half - e[0] - g, 0.0)
            cz = (2.0 * u[2] - 1.0) * max(half - e[2] - g, 0.0)
            on = []
            if resting[b]:
                # flat on the support: the ground or the highest upper surface of a pinned body under the footprint
                corners = box_corners((cx, 0.0, cz), rotations[b], half_sides[b])
                bottom = corners[np.argsort(corners[:, 1])[:4]][:, [0, 2]]
                order = np.argsort(np.arctan2(bottom[:, 1] - bottom[:, 1].mean(), bottom[:, 0] - bottom[:, 0].mean()))
                foot = bottom[order]
                support = 0.0
                for j in range(b):
                    if not pinned[j]:
                        continue
                    if lo_placed[j, 0] > cx + e[0] or hi_placed[j, 0] < cx - e[0]:
                        continue
                    if lo_placed[j, 2] > cz + e[2] or hi_placed[j, 2] < cz - e[2]:
                        continue
                    top = support_height(foot, centres[j], rotations[j], half_sides[j] + clearance[j])
                    if top is not None:
                        on.append((j, top))
                        support = max(support, top)
                cy = support + RESTING_GAP_CELLS * h + e[1]
                on = [j for j, _ in on]
            else:
                cy = y_min + u[1] * (y_max - y_min) + e[1]
            c = np.array([cx, cy, cz])
            lo, hi = c - e - g, c + e + g
            overlap = (lo < hi_placed) & (lo_placed < hi)
            if on:
                overlap[on] = False  # the supports are checked exactly below
            if not bool(overlap.all(axis=1).any()) and not any(
                boxes_overlap(c, rotations[b], half_sides[b], centres[j], rotations[j], half_sides[j] + clearance[j])
                for j in on
            ):
                break
            if attempts >= MAX_PLACEMENT_DRAWS:
                footprint *= FOOTPRINT_GROWTH
                growths += 1
                attempts = 0
        centres.append(tuple(float(v) for v in c))
        supports.append(on)
        lo_placed = np.vstack([lo_placed, c - e])
        hi_placed = np.vstack([hi_placed, c + e])
    stats = {
        "footprint": float(footprint),
        "draws": int(draws),
        "acceptance_rate": float(n / max(draws, 1)),
        "footprint_growths": int(growths),
        "resting_on_bodies": int(sum(1 for on in supports if on)),
    }
    return centres, stats


def field_clearance(perturbation_scale: float, strength: float, h: float) -> float:
    """Room (m) a body's deformation field needs: `CLEARANCE_SIGMAS` times the field's RMS displacement
    (`Augmenter.initial_states`: perturbation_scale strength DISPLACEMENT_SCALE cells)."""
    return CLEARANCE_SIGMAS * DISPLACEMENT_SCALE * float(perturbation_scale) * float(strength) * h


def _box_n_face(cell_counts) -> int:
    """Largest count of exposed face samples sharing one face index of a full box (the load floor's n_face)."""
    nx, ny, nz = cell_counts
    return max(ny * nz, nx * nz, nx * ny)


# ---------------------------------------------------------------------------------------------------- sampling
def material_band(rng: np.random.Generator, cfg) -> dict | None:
    """The scene's material band (config `scene_wave_speed_min`, `scene_wave_speed_band`, `scene_density_band`;
    2026-10-02): the density interval [rho_lo, rho_hi] (log-uniform lower edge over the config's range, at most the
    density band wide, its heaviest material fast enough at E_max) and the squared wave speed interval [c2_lo, c2_hi]
    (log-uniform lower edge over what keeps E = c2 rho inside the config's range for every density of the interval,
    at most the wave-speed band squared wide, at least c_min squared). None when c_min is 0 (independent draws)."""
    c_min = float(getattr(cfg, "scene_wave_speed_min", 0.0) or 0.0)
    if c_min <= 0.0:
        return None
    E_lo, E_hi = (float(v) for v in cfg.youngs_modulus_range)
    rho_lo, rho_hi = (float(v) for v in cfg.density_range)
    c2_min = c_min * c_min
    if E_hi / rho_lo < c2_min:
        raise ValueError(
            f"scene_wave_speed_min {c_min} m/s needs E / rho >= {c2_min}, the ranges allow {E_hi / rho_lo}"
        )
    rho_band = max(1.0, float(cfg.scene_density_band))
    if E_hi / E_lo < rho_band:
        raise ValueError(f"scene_density_band {rho_band} exceeds the Young's modulus ratio {E_hi / E_lo}")
    r_lo = _log_uniform(rng, rho_lo, max(rho_lo, min(rho_hi / rho_band, E_hi / c2_min)))
    r_hi = min(rho_hi, r_lo * rho_band, E_hi / c2_min)
    c2_floor, c2_ceiling = max(c2_min, E_lo / r_lo), E_hi / r_hi  # E = c2 rho stays in range for rho in [r_lo, r_hi]
    band2 = max(1.0, float(cfg.scene_wave_speed_band)) ** 2
    c2_lo = _log_uniform(rng, c2_floor, max(c2_floor, c2_ceiling / band2))
    c2_hi = min(c2_ceiling, c2_lo * band2)
    return {"rho": (r_lo, r_hi), "c2": (c2_lo, c2_hi)}


def draw_material(rng: np.random.Generator, cfg, band: dict | None) -> dict:
    """One body's material (SI): independent log-uniform E and rho without a band, otherwise rho and the squared
    wave speed log-uniform in the scene's band and E = c2 rho (within the config's range by construction)."""
    if band is None:
        E = _log_uniform(rng, *cfg.youngs_modulus_range)
        nu = float(rng.uniform(*cfg.poissons_ratio_range))
        rho = _log_uniform(rng, *cfg.density_range)
    else:
        c2 = _log_uniform(rng, *band["c2"])
        nu = float(rng.uniform(*cfg.poissons_ratio_range))
        rho = _log_uniform(rng, *band["rho"])
        E = c2 * rho
    return {"E": E, "nu": nu, "rho": rho, "eta": _log_uniform(rng, *cfg.damping_range)}


def sample_scene(
    master_seed: int, scene_seed: int, cfg, validation: bool = False, mix: SceneMix | None = None
) -> SceneV5:
    """One v5 scene (SI): a pure function of (master_seed, scene_seed, validation), the config and the mix (module
    docstring); `mix` (default: the config's fractions) overrides `pinned_body_fraction` and `resting_body_fraction`
    on the same seed streams, so the same seed with a higher pinned fraction pins a superset of the bodies."""
    mix = scene_mix(cfg) if mix is None else mix
    ss = np.random.SeedSequence([master_seed, scene_seed, 1 if validation else 0, STREAM_TAG])
    rng_scene, rng_bodies, rng_place, rng_pins, rng_rest, rng_faces, rng_pinned_faces = (
        np.random.default_rng(s) for s in ss.spawn(7)
    )  # the first six children of a SeedSequence do not depend on how many are spawned
    h, dt = float(cfg.cell_size), float(cfg.time_step)
    gravity = tuple(float(v) for v in cfg.gravity)
    g_mag = math.sqrt(sum(v * v for v in gravity))

    # scene-wide draws
    direction = rng_scene.normal(size=3)
    direction[1] *= DRIFT_VERTICAL_SCALE
    direction = _unit(direction)
    speed = float(rng_scene.uniform(*cfg.drift_speed_range))
    drift = speed * direction
    kappa_drawn = _log_uniform(rng_scene, *cfg.contact_kappa_range)
    beta = float(rng_scene.uniform(*cfg.contact_beta_range))
    mu_f = float(rng_scene.uniform(*cfg.contact_mu_range))
    band = material_band(
        rng_scene, cfg
    )  # drawn after the contact constants: the scene stream of a band-less config is unchanged

    # bodies until the cell budget is reached (the last body may exceed it)
    lo, hi = (int(v) for v in cfg.body_sides)
    drawn: list[dict] = []
    total = 0
    while total < int(cfg.scene_cells):
        sides = tuple(int(v) for v in rng_bodies.integers(lo, hi + 1, size=3))
        material = draw_material(rng_bodies, cfg, band)
        perturbation_scale = float(rng_bodies.uniform(*cfg.perturbation_scale_range))
        strength = float(rng_bodies.uniform(*cfg.strength_range))
        velocity_dt = float(rng_bodies.uniform(*cfg.velocity_dt_range))
        noise = float(rng_bodies.uniform(0.0, speed)) * _unit(rng_bodies.normal(size=3))
        seed = int(rng_bodies.integers(0, 2**63 - 1))
        drawn.append(
            {
                "cell_counts": sides,
                "material": material,
                "velocity": tuple(float(v) for v in drift + noise),
                "strength": strength,
                "velocity_dt": velocity_dt,
                "perturbation_scale": perturbation_scale,
                "seed": seed,
            }
        )
        total += math.prod(sides)

    # orientation and gap per body (placement stream)
    gap_lo, gap_hi = (int(v) for v in cfg.placement_gap_cells)
    quats, gaps = [], []
    for _ in drawn:
        quats.append(random_quaternion(rng_place))
        gaps.append(int(rng_place.integers(gap_lo, gap_hi + 1)) * h)
    # pins (own stream, one draw pair per body whatever the fraction): a pinned body is held at its pose, so its
    # rigid velocity is zero; the deformation fields stay
    pins = []
    for _ in drawn:
        u, face = float(rng_pins.random()), int(rng_pins.integers(0, len(FACE_PINS)))
        pins.append(FACE_PINS[face] if u < float(mix.pinned_fraction) else "none")
    # resting bodies (own stream, one draw triple per body whatever the fraction): flat on a face with a random yaw
    resting = []
    for pin in pins:
        u, face, yaw = float(rng_rest.random()), int(rng_rest.integers(0, 6)), float(rng_rest.uniform(0.0, 2 * math.pi))
        rest = pin == "none" and u < float(mix.resting_fraction)
        resting.append(rest)
        if rest:
            quats[-len(pins) + len(resting) - 1] = face_down_quaternion(face, yaw)
    any_resting = any(resting)
    extents = np.stack([half_extents(b["cell_counts"], q, h) for b, q in zip(drawn, quats, strict=True)])
    rotations = [rotation_matrix(q) for q in quats] if any_resting else None
    half_sides = 0.5 * h * np.array([b["cell_counts"] for b in drawn], dtype=float) if any_resting else None
    clearance = np.array([field_clearance(b["perturbation_scale"], b["strength"], h) for b in drawn])
    centres, placement = _place(
        rng_place,
        extents,
        np.asarray(gaps),
        h,
        cfg,
        resting=resting if any_resting else None,
        pinned=[pin != "none" for pin in pins],
        rotations=rotations,
        half_sides=half_sides,
        clearance=clearance,
    )
    # static faces after every body (own stream): rejected against the bodies' boxes grown by at least one cell
    grown = np.maximum(FACE_GROWTH_CELLS * h, clearance)[:, None]
    centres_np = np.asarray(centres, dtype=float)
    lo_grown, hi_grown = centres_np - extents - grown, centres_np + extents + grown
    static_faces, face_stats = _place_faces(rng_faces, cfg, h, placement["footprint"], lo_grown, hi_grown)
    # pinned contact faces (seventh stream): within the deformation reach of pinned bodies, after the regular faces
    pinned_faces, pinned_stats = _place_pinned_faces(
        rng_pinned_faces, cfg, h, drawn, centres_np, quats, pins, clearance, lo_grown, hi_grown
    )
    static_faces = list(static_faces) + list(pinned_faces)
    placement = {**placement, **face_stats, **pinned_stats}
    if band is not None:
        placement["material_band"] = {k: [float(v) for v in band[k]] for k in ("rho", "c2")}
    still = {"velocity": (0.0, 0.0, 0.0)}
    bodies = [
        BodySpec(
            position=c,
            quaternion=q,
            pins=pin,
            resting=rest,
            **({**b, **still, "perturbation_scale": 0.0} if rest else {**b, **still} if pin != "none" else b),
        )
        for b, q, c, pin, rest in zip(drawn, quats, centres, pins, resting, strict=True)
    ]

    # contact stiffness ratio with the load floor over every body
    kappa_floor = 0.0
    for b in bodies:
        m = b.material
        ke_floor = contact_stiffness_floor(
            m["rho"], g_mag, h, b.cells, _box_n_face(b.cell_counts), cfg.contact_static_penetration_max
        )
        kappa_floor = max(kappa_floor, ke_floor / (m["E"] * h))
    contact = {
        "kappa": float(max(kappa_drawn, kappa_floor)),
        "kappa_drawn": float(kappa_drawn),
        "kappa_floor": float(kappa_floor),
        "floor_bound": bool(kappa_floor > kappa_drawn),
        "beta": beta,
        "mu_f": mu_f,
        "friction_epsilon": float(cfg.contact_friction_epsilon),
        "floor_scale": float(cfg.energy_floor_scale),
        "plane_normal": list(PLANE_NORMAL),
    }
    return SceneV5(
        seed=int(scene_seed),
        validation=bool(validation),
        h=h,
        dt=dt,
        gravity=gravity,
        bodies=bodies,
        plane_height=0.0,
        drift=tuple(float(v) for v in drift),
        contact=contact,
        placement=placement,
        static_faces=[[tuple(float(v) for v in c) for c in f] for f in static_faces],
    )


def held_out_scene(master_seed: int, index: int, cfg, mix: SceneMix | None = None) -> SceneV5:
    """Validation scene `index`: the separate stream of `sample_scene(..., validation=True)`, by default with the
    config's final mix (the goal); the cheap validation passes the epoch's mix."""
    return sample_scene(master_seed, index, cfg, validation=True, mix=mix)


def scene_summary(scene: SceneV5) -> dict:
    """The run record's view of a scene."""
    drift = np.asarray(scene.drift)
    return {
        "seed": scene.seed,
        "validation": scene.validation,
        "bodies": len(scene.bodies),
        "pinned_bodies": sum(1 for b in scene.bodies if b.pinned),
        "resting_bodies": sum(1 for b in scene.bodies if b.resting),
        "resting_on_bodies": scene.placement.get("resting_on_bodies", 0),
        "static_faces": len(scene.static_faces),
        "pinned_contact_faces": scene.placement.get("pinned_contact_faces", 0),
        "cells": scene.cells,
        "drift": [float(v) for v in drift],
        "drift_speed": float(np.linalg.norm(drift)),
        "kappa": scene.contact["kappa"],
        "kappa_drawn": scene.contact["kappa_drawn"],
        "floor_bound": scene.contact["floor_bound"],
        "beta": scene.contact["beta"],
        "mu_f": scene.contact["mu_f"],
        "plane_height": scene.plane_height,
        "footprint": scene.placement.get("footprint"),
        "placement_acceptance": scene.placement.get("acceptance_rate"),
        "material_band": scene.placement.get("material_band"),
        "footprint_growths": scene.placement.get("footprint_growths"),
        "static_face_acceptance": scene.placement.get("static_face_acceptance"),
    }


# ---------------------------------------------------------------------------------------------------- realise
def realise(scene: SceneV5, grids, aug, device, dtype=torch.float32, physical_floor: bool = True):
    """(grids, X [N,3], V [N,3], Material, ContactScene) of the scene in normalised units on `device`.

    `grids` is a GridCache and `aug` an Augmenter on the same device (the deformation fields are drawn there, one
    generator per body seeded by `BodySpec.seed`). X = R(q) (rest - centre + deformation) + position / h in cells,
    V = R(q) deformation velocity + velocity dt / h in cells per step. A pinned body (`BodySpec.pins`) takes the
    grid with that face clamped: its pinned corners sit exactly at the rigid pose (the deformation field is zero
    there) with zero velocity, and its rigid velocity is zero whatever the spec says. `dtype` other than float32
    builds the state and the material tensors in that precision (CPU float64 tests). All bodies share one unit of energy, mu_ref h^3
    with mu_ref the geometric mean of their shear moduli (`material_from_si(mu_ref=...)`, section 11: the reaction
    of a body pair on the partner and the coupled centroid update are then consistent across stiffnesses).
    """
    device = torch.device(device)
    h, dt = scene.h, scene.dt
    gs = [grids.get(b.cell_counts, b.pins) for b in scene.bodies]
    gens = [torch.Generator(device=aug.device).manual_seed(b.seed) for b in scene.bodies]
    Xa, Va = aug.initial_states(gs, scene.bodies, gens)
    X_parts, V_parts, materials = [], [], []
    c = scene.contact
    mu_ref = reference_modulus([b.material for b in scene.bodies])  # one unit of energy for the whole scene
    for g, b, X_b, V_b in zip(gs, scene.bodies, Xa, Va, strict=True):
        R = torch.tensor(rotation_matrix(b.quaternion), dtype=torch.float64, device=device)
        centre = torch.tensor(b.cell_counts, dtype=torch.float64, device=device) / 2
        p = torch.tensor(b.position, dtype=torch.float64, device=device) / h
        v = torch.tensor((0.0, 0.0, 0.0) if b.pinned else b.velocity, dtype=torch.float64, device=device) * dt / h
        X_parts.append((X_b.to(device, torch.float64) - centre) @ R.T + p)
        V_parts.append(V_b.to(device, torch.float64) @ R.T + v)
        materials.append(
            material_from_si(
                E=b.material["E"],
                nu=b.material["nu"],
                rho=b.material["rho"],
                eta=b.material["eta"],
                gravity=scene.gravity,
                h=h,
                dt=dt,
                cell_count=g.C,
                sample_count=g.S,
                kappa=float(c["kappa"]),
                beta=float(c["beta"]),
                mu_f=float(c["mu_f"]),
                friction_epsilon=float(c.get("friction_epsilon", 0.01)),
                floor_scale=float(c.get("floor_scale", 1.0)),
                physical_floor=physical_floor,
                device=device,
                mu_ref=mu_ref,
                dtype=dtype,
            )
        )
    X = torch.cat(X_parts).to(dtype)
    V = torch.cat(V_parts).to(dtype)
    material = Material.cat(materials)
    O = len(gs)
    z = lambda *shape, dt=dtype: torch.zeros(*shape, dtype=dt, device=device)  # noqa: E731
    contact_scene = ContactScene(
        plane_n=torch.tensor([c.get("plane_normal", PLANE_NORMAL)], dtype=dtype, device=device).expand(O, 3).clone(),
        plane_d=torch.full((O,), scene.plane_height / h, dtype=dtype, device=device),
        plane_present=torch.ones(O, dtype=torch.bool, device=device),
        points=z(0, 3),
        normals=z(0, 3),
        radii=z(0),
        point_offsets=z(O + 1, dt=torch.int64),
        faces=torch.tensor(scene.static_faces, dtype=torch.float64, device=device).reshape(-1, 4, 3).to(dtype) / h,
    )
    return gs, X, V, material, contact_scene
