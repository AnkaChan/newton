# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""v5 scenes (design spec section 11; Anka's note `notes/v5-free-motion-with-contact.md`): one scene per batch,
free cuboids of random size, material, pose and velocity above a ground plane, all in ONE world frame.

`sample_scene` draws a scene in SI from numpy seed streams: a pure function of (master_seed, scene_seed, validation)
like `jobs.sample_scene_spec`, with the validation flag selecting a separate stream so held-out scene indices never
collide with training scenes. `realise` converts it to the solver's normalised units (h = dt = 1): box grids
without pins, X = the rotated and translated rest lattice plus the multiscale deformation field, V = the rigid
velocity in cells per step plus the deformation velocity field, one Material per body (gravity along -y, the scene's
contact constants) and a ContactScene with the ground plane for every body and no static points. All objects
share the world frame with origin zero, so plane_d = plane_height / h is the same number for every object.

Draws (per scene stream): drift direction (standard normal with the vertical component scaled by
`DRIFT_VERTICAL_SCALE`, normalised), drift speed U(drift_speed_range), kappa log-uniform over contact_kappa_range,
beta, mu_f. Per body (body stream, until the cell total reaches `scene_cells`; the last body may exceed it): sides
U{body_sides} per axis independently, E, nu, rho, eta, perturbation_scale, strength, velocity_dt with the ranges and
draw order of `jobs.sample_scene_spec` (no gravity draw: the scene's gravity is `cfg.gravity`), a velocity noise of
magnitude U(0, drift speed) in a uniform random direction added to the drift, and the deformation-field seed.
The contact stiffness ratio is floored by the load rule of the contact note (section 7, amendment 2026-09-29) taken
over every body: kappa >= m_b g / (n_face_b d_max E_b h), which the heaviest and softest bodies decide.

Placement (placement stream): per body a uniform random quaternion (Shoemake) and a gap U{placement_gap_cells}
cells, then bodies are placed one at a time in a square column of side `footprint` (m) centred on the origin.
The body's lowest point is uniform between 2 cells and `placement_height` above the plane (the FIRST body: between
1 and 2 cells, so at least one body reaches the ground within a few steps), the horizontal position uniform with
the body's axis-aligned bounding box inside the column. A candidate is rejected while its bounding box grown by
the body's gap on every side overlaps a placed body's bounding box, so two bodies' boxes are at least the gap
apart. The footprint is derived from the bodies: the grown bounding-box volumes sum to `PLACEMENT_FILL` of the
column over the vertical band the boxes can occupy; when a body is not placed within `MAX_PLACEMENT_DRAWS` draws the
footprint grows by `FOOTPRINT_GROWTH` and the draws restart (counted in `scene.placement`). Measured on the default
configuration (64000 cells, about 150 bodies, 16 scenes): footprint 4.6-5.1 m, acceptance 68-74 % of the position
draws, no footprint growth.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .jobs import _log_uniform
from .scenes import PLANE_NORMAL, contact_stiffness_floor
from .structs import _MATERIAL_TENSORS, ContactScene, Material
from .units import material_from_si

Tensor = torch.Tensor

STREAM_TAG = 5  # last SeedSequence key: v5 scene streams never coincide with the body-mode streams of jobs.py
DRIFT_VERTICAL_SCALE = 0.25  # the drift direction's y component is scaled by this before normalisation
PLACEMENT_FILL = 0.3  # grown bounding-box volume over the column volume that sets the footprint
MAX_PLACEMENT_DRAWS = 200  # position draws per body before the footprint grows
FOOTPRINT_GROWTH = 1.1
GROUND_CELLS = 2.0  # lowest point of a body at least this many cells above the plane (the first body: 1-2 cells)
FIRST_BODY_CELLS = (1.0, 2.0)


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

    @property
    def cells(self) -> int:
        return int(math.prod(self.cell_counts))


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
    placement: dict = field(default_factory=dict)  # footprint, draws, acceptance_rate, footprint_growths

    @property
    def cells(self) -> int:
        return sum(b.cells for b in self.bodies)

    @staticmethod
    def from_dict(d: dict) -> SceneV5:
        t = lambda v: tuple(v)  # noqa: E731
        bodies = [
            BodySpec(
                cell_counts=t(int(v) for v in b["cell_counts"]),
                material=dict(b["material"]),
                position=t(float(v) for v in b["position"]),
                quaternion=t(float(v) for v in b["quaternion"]),
                velocity=t(float(v) for v in b["velocity"]),
                strength=float(b["strength"]),
                velocity_dt=float(b["velocity_dt"]),
                perturbation_scale=float(b["perturbation_scale"]),
                seed=int(b["seed"]),
            )
            for b in d["bodies"]
        ]
        return SceneV5(
            seed=int(d["seed"]),
            validation=bool(d["validation"]),
            h=float(d["h"]),
            dt=float(d["dt"]),
            gravity=t(float(v) for v in d["gravity"]),
            bodies=bodies,
            plane_height=float(d["plane_height"]),
            drift=t(float(v) for v in d["drift"]),
            contact=dict(d["contact"]),
            placement=dict(d.get("placement", {})),
        )


# ---------------------------------------------------------------------------------------------------- geometry
def random_quaternion(rng: np.random.Generator) -> tuple:
    """Uniform random unit quaternion (x, y, z, w) (Shoemake's method, three uniforms)."""
    u1, u2, u3 = rng.random(3)
    a, b = math.sqrt(1.0 - u1), math.sqrt(u1)
    return (a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2), b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3))


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


def rigid_pose(body: BodySpec, rest: Tensor, h: float) -> Tensor:
    """The rotated and translated rest lattice [P,3] in cell units (float64): R(q) (rest - centre) + position / h."""
    R = torch.tensor(rotation_matrix(body.quaternion), dtype=torch.float64, device=rest.device)
    centre = torch.tensor(body.cell_counts, dtype=torch.float64, device=rest.device) / 2
    p = torch.tensor(body.position, dtype=torch.float64, device=rest.device) / h
    return (rest.to(torch.float64) - centre) @ R.T + p


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-300)


def _place(rng: np.random.Generator, extents: np.ndarray, gaps: np.ndarray, h: float, cfg) -> tuple[list, dict]:
    """Body centres [n][3] (m) by rejection sampling of grown bounding boxes (module docstring) and the statistics."""
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
    lo_placed, hi_placed = np.zeros((0, 3)), np.zeros((0, 3))
    centres = []
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
            cy = y_min + u[1] * (y_max - y_min) + e[1]
            c = np.array([cx, cy, cz])
            lo, hi = c - e - g, c + e + g
            if not bool(((lo < hi_placed) & (lo_placed < hi)).all(axis=1).any()):
                break
            if attempts >= MAX_PLACEMENT_DRAWS:
                footprint *= FOOTPRINT_GROWTH
                growths += 1
                attempts = 0
        centres.append(tuple(float(v) for v in c))
        lo_placed = np.vstack([lo_placed, c - e])
        hi_placed = np.vstack([hi_placed, c + e])
    stats = {
        "footprint": float(footprint),
        "draws": int(draws),
        "acceptance_rate": float(n / max(draws, 1)),
        "footprint_growths": int(growths),
    }
    return centres, stats


def _box_n_face(cell_counts) -> int:
    """Largest count of exposed face samples sharing one face index of a full box (the load floor's n_face)."""
    nx, ny, nz = cell_counts
    return max(ny * nz, nx * nz, nx * ny)


# ---------------------------------------------------------------------------------------------------- sampling
def sample_scene(master_seed: int, scene_seed: int, cfg, validation: bool = False) -> SceneV5:
    """One v5 scene (SI): a pure function of (master_seed, scene_seed, validation) and the config (module docstring)."""
    ss = np.random.SeedSequence([master_seed, scene_seed, 1 if validation else 0, STREAM_TAG])
    rng_scene, rng_bodies, rng_place = (np.random.default_rng(s) for s in ss.spawn(3))
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

    # bodies until the cell budget is reached (the last body may exceed it)
    lo, hi = (int(v) for v in cfg.body_sides)
    drawn: list[dict] = []
    total = 0
    while total < int(cfg.scene_cells):
        sides = tuple(int(v) for v in rng_bodies.integers(lo, hi + 1, size=3))
        material = {
            "E": _log_uniform(rng_bodies, *cfg.youngs_modulus_range),
            "nu": float(rng_bodies.uniform(*cfg.poissons_ratio_range)),
            "rho": _log_uniform(rng_bodies, *cfg.density_range),
            "eta": _log_uniform(rng_bodies, *cfg.damping_range),
        }
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

    # orientation and gap per body, then the positions
    gap_lo, gap_hi = (int(v) for v in cfg.placement_gap_cells)
    quats, gaps = [], []
    for _ in drawn:
        quats.append(random_quaternion(rng_place))
        gaps.append(int(rng_place.integers(gap_lo, gap_hi + 1)) * h)
    extents = np.stack([half_extents(b["cell_counts"], q, h) for b, q in zip(drawn, quats, strict=True)])
    centres, placement = _place(rng_place, extents, np.asarray(gaps), h, cfg)
    bodies = [
        BodySpec(position=c, quaternion=q, **b) for b, q, c in zip(drawn, quats, centres, strict=True)
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
    )


def held_out_scene(master_seed: int, index: int, cfg) -> SceneV5:
    """Validation scene `index`: the separate stream of `sample_scene(..., validation=True)`."""
    return sample_scene(master_seed, index, cfg, validation=True)


def scene_summary(scene: SceneV5) -> dict:
    """The run record's view of a scene."""
    drift = np.asarray(scene.drift)
    return {
        "seed": scene.seed,
        "validation": scene.validation,
        "bodies": len(scene.bodies),
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
        "footprint_growths": scene.placement.get("footprint_growths"),
    }


# ---------------------------------------------------------------------------------------------------- realise
def realise(scene: SceneV5, grids, aug, device, dtype=torch.float32):
    """(grids, X [N,3], V [N,3], Material, ContactScene) of the scene in normalised units on `device`.

    `grids` is a GridCache and `aug` an Augmenter on the same device (the deformation fields are drawn there, one
    generator per body seeded by `BodySpec.seed`). X = R(q) (rest - centre + deformation) + position / h in cells,
    V = R(q) deformation velocity + velocity dt / h in cells per step. `dtype` other than float32 casts the state
    and the material tensors (CPU float64 tests).
    """
    device = torch.device(device)
    h, dt = scene.h, scene.dt
    gs = [grids.get(b.cell_counts, "none") for b in scene.bodies]
    gens = [torch.Generator(device=aug.device).manual_seed(b.seed) for b in scene.bodies]
    Xa, Va = aug.initial_states(gs, scene.bodies, gens)
    X_parts, V_parts, materials = [], [], []
    c = scene.contact
    for g, b, X_b, V_b in zip(gs, scene.bodies, Xa, Va, strict=True):
        R = torch.tensor(rotation_matrix(b.quaternion), dtype=torch.float64, device=device)
        centre = torch.tensor(b.cell_counts, dtype=torch.float64, device=device) / 2
        p = torch.tensor(b.position, dtype=torch.float64, device=device) / h
        v = torch.tensor(b.velocity, dtype=torch.float64, device=device) * dt / h
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
                device=device,
            )
        )
    X = torch.cat(X_parts).to(dtype)
    V = torch.cat(V_parts).to(dtype)
    material = Material.cat(materials)
    if dtype != torch.float32:
        for f in _MATERIAL_TENSORS:
            setattr(material, f, getattr(material, f).to(dtype))
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
    )
    return gs, X, V, material, contact_scene


def to_json_dict(scene: SceneV5) -> dict:
    """`dataclasses.asdict` (tuples become lists under json)."""
    return dataclasses.asdict(scene)
