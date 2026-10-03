# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validation on held-out states (design spec 7.3): cheap (one physical step, K queries) and full horizon
(K queries x H steps). Frozen weights, inertial candidates, SI units in the records.

v5 scenes (`validate_cheap_v5`, `validate_full_horizon_v5`, design spec section 11): held-out scenes from
`scenes_v5.held_out_scene`; a record is one scene with the per-body fields aggregated over its bodies in SI
(residual_n: mean of the bodies' free-corner residual norms, energy_joule: sum, penetration_r: max,
inverted_cells: sum), plus `interbody_penetration_r` (max over the body pairs of relu(r - gap) / r),
`plane_penetration_r` (the same over the STATIC partners: the plane, the discs and the scene's static faces; the key
keeps its name for the dashboard), `contact_pairs` counts by kind (static faces included), and in the full horizon the
`momentum_drift` of a contact-free copy of the first scene (`momentum_drift_check`: plane removed, bodies spread
far apart; |sum m_i v_i(t) - sum m_i v_i(0) - t M g| / |M g t| over the FREE bodies after the horizon, SI; a pinned
body hands its momentum to its pins and is left out of the sums).
"""

from __future__ import annotations

import dataclasses
import gc
import math
import time

import numpy as np
import torch

from . import contact, physics, scenes, scenes_v5
from .batch import Batch
from .jobs import sample_scene_spec
from .runner import material_for, pair_counts, scene_batch, seeded_generator
from .structs import Material
from .units import energy_scale, force_scale

Tensor = torch.Tensor


def local_objective(
    E_after: Tensor, E_before: Tensor, floor: Tensor, increase_weight: float = 1.0, bounded: bool = True
) -> Tensor:
    """asinh(E_after / scale) + w * penalty(increase / scale), scale = max(|E_before|, floor) detached.

    `bounded` (Anka, 2026-10-02): the increase penalty is asinh(relu(increase / scale)) instead of the linear relu, so a
    single body whose energy jumps by orders of magnitude cannot dominate the batch gradient; both forms share the
    zero point and the sign."""
    scale = torch.maximum(E_before.abs(), floor).detach()
    increase = torch.relu((E_after - E_before.detach()) / scale)
    if bounded:
        increase = torch.asinh(increase)
    return torch.asinh(E_after / scale) + increase_weight * increase


def _held_out_batch(step, cfg, aug, grid, seeds: list, device, master_seed: int):
    specs = [sample_scene_spec(master_seed, s, cfg, validation=True) for s in seeds]
    batch = Batch.build([grid] * len(seeds), device)
    batch.material = Material.cat([material_for(s, grid, cfg, device) for s in specs])
    batch.scene = scenes.scenes_for_objects([s.contact or {} for s in specs], [cfg.cell_size] * len(seeds), device)
    gens = [seeded_generator(device, master_seed, s, 1, 101) for s in seeds]
    Xs, Vs = aug.initial_states([grid] * len(seeds), specs, gens)
    batch.X, batch.V = torch.cat(Xs), torch.cat(Vs)
    batch.X_prev, batch.x = batch.X.clone(), batch.X.clone()
    sel = torch.ones(len(seeds), dtype=torch.bool, device=device)
    step.prepare(batch, sel, gens)  # candidates: the training rule (50 % inertial, 50 % perturbed), seeded per state
    return batch, specs, sel


def _to_list(t: Tensor) -> list:
    return [float(v) for v in t.detach().cpu().tolist()]


@torch.no_grad()
def validate_cheap(step, cfg, aug, grid, device, master_seed: int) -> list:
    """Per-sample records for report.summarize_cheap_validation."""
    out_records = []
    K = cfg.validation_iterations
    seeds = list(range(cfg.validation_count))
    for start in range(0, len(seeds), cfg.batch_size):
        chunk = seeds[start : start + cfg.batch_size]
        batch, specs, _sel = _held_out_batch(step, cfg, aug, grid, chunk, device, master_seed)
        es, fs = energy_scale(batch.material), force_scale(batch.material)
        residual = [_to_list(physics.residual(batch.gX, batch) * fs)]
        energy = [_to_list(batch.E * es)]
        pen = [_to_list(contact.penetration(batch, batch.x))]
        inv = [_to_list(physics.inverted_cells(physics.modes_and_center(batch.x, batch)[1], batch))]
        scale = torch.maximum(batch.E.abs(), batch.material.floor) * es
        for _ in range(K):
            out = step.query(batch)
            step.commit(batch, out)
            residual.append(_to_list(physics.residual(batch.gX, batch) * fs))
            energy.append(_to_list(batch.E * es))
            pen.append(_to_list(contact.penetration(batch, batch.x)))
            inv.append(_to_list(physics.inverted_cells(physics.modes_and_center(batch.x, batch)[1], batch)))
        for o, seed in enumerate(chunk):
            r = [row[o] for row in residual]
            e = [row[o] for row in energy]
            survived = all(map(_finite, r)) and all(map(_finite, e))
            out_records.append(
                {
                    "seed": seed,
                    "residual_n": r,
                    "energy_joule": e,
                    "penetration_r": [row[o] for row in pen],
                    "inverted_cells": [row[o] for row in inv],
                    "survived": survived,
                    "failed_first_update": not (_finite(r[1]) and _finite(e[1])),
                    "scale_joule": float(scale[o]),
                    "contact": bool(specs[o].contact),
                }
            )
    return out_records


@torch.no_grad()
def validate_full_horizon(
    step, cfg, aug, grid, device, master_seed: int, K: int | None = None, H: int | None = None
) -> tuple[list, float]:
    """Per-sample physical records for report.summarize_full_horizon; returns (samples, seconds).

    K and H default to the config values; the trainer passes the growth stage's caps (K = min(K_max, validation_full_iterations),
    H = H_max) as the previous campaigns did."""
    t0 = time.perf_counter()
    K = K or cfg.validation_full_iterations
    H = H or cfg.validation_full_steps
    seeds = list(range(cfg.validation_full_count))
    records = []
    for start in range(0, len(seeds), cfg.batch_size):
        chunk = seeds[start : start + cfg.batch_size]
        batch, _specs, _sel = _held_out_batch(step, cfg, aug, grid, chunk, device, master_seed)
        es, fs = energy_scale(batch.material), force_scale(batch.material)
        rows = [[] for _ in chunk]
        alive = torch.ones(len(chunk), dtype=torch.bool, device=device)
        for _ in range(H):
            for _ in range(K):
                out = step.query(batch)
                step.commit(batch, out)
            res = physics.residual(batch.gX, batch) * fs
            en = batch.E * es
            pen = contact.penetration(batch, batch.x)
            inv = physics.inverted_cells(physics.modes_and_center(batch.x, batch)[1], batch)
            alive &= torch.isfinite(res) & torch.isfinite(en)
            for o in range(len(chunk)):
                rows[o].append(
                    {
                        "residual_n": float(res[o]),
                        "energy_joule": float(en[o]),
                        "penetration_r": float(pen[o]),
                        "inverted_cells": float(inv[o]),
                    }
                )
            if not bool(alive.any()):
                break
            step.advance(batch, alive, None)
            batch.x = torch.where(torch.isfinite(batch.x), batch.x, batch.X)  # keep dead objects finite
        for o, seed in enumerate(chunk):
            records.append(
                {"seed": seed, "physical_records": rows[o], "survived": bool(alive[o]) and len(rows[o]) == H}
            )
    return records, time.perf_counter() - t0


def _finite(v: float) -> bool:
    return v == v and abs(v) != float("inf")


# ------------------------------------------------------------------------------------------------ v5 scenes
SCENE_CURVES = (
    "residual_n",
    "energy_joule",
    "penetration_r",
    "inverted_cells",
    "interbody_penetration_r",
    "plane_penetration_r",
)


def _held_out_scene_batch(step, cfg, grids, aug, scene, device, master_seed: int, plane: bool = True):
    batch = scene_batch(scene, grids, aug, device, plane=plane, physical_floor=cfg.physical_floor)
    gens = seeded_generator(device, master_seed, scene.seed, 1, 101)  # one candidate-noise stream per scene
    step.prepare(batch, batch.active, gens)  # candidates: the training rule (50 % inertial, 50 % perturbed)
    return batch


def _release(batch) -> None:
    """Return a finished validation batch's memory (cuDSS factor, Warp meshes, cached allocator blocks) before the
    next scene is built (2026-10-02: the v6 validation ran rank 0 out of device memory)."""
    batch.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scene_metrics(batch) -> dict:
    """The scene's metrics at the batch's candidate, aggregated over its bodies in SI (module docstring)."""
    es, fs = energy_scale(batch.material), force_scale(batch.material)
    res = physics.residual(batch.gX, batch) * fs
    en = batch.E * es
    pen = contact.penetration(batch, batch.x)
    inv = physics.inverted_cells(physics.modes_and_center(batch.x, batch)[1], batch)
    by_kind = contact.kind_penetration(batch, batch.x)
    finite = bool(torch.isfinite(res).all()) and bool(torch.isfinite(en).all())
    return {
        "residual_n": float(res.mean()),
        "energy_joule": float(en.sum()),
        "penetration_r": float(pen.max()),
        "inverted_cells": float(inv.sum()),
        "interbody_penetration_r": float(by_kind[contact.KIND_BODY]),
        "plane_penetration_r": float(max(by_kind[0], by_kind[contact.KIND_STATIC])),  # plane, discs, static faces
        "contact_pairs": pair_counts(batch),
        "finite": finite,
    }


@torch.no_grad()
def validate_cheap_v5(step, cfg, grids, aug, device, master_seed: int, epoch: int | None = None) -> list:
    """Per-scene records (`cfg.validation_scene_count` held-out scenes, K = `cfg.validation_iterations` queries of
    one physical step) for report.summarize_cheap_validation; the curves hold the value before each query and after
    the last, `contact_pairs` the step's detection. The held-out scenes follow the curriculum mix of `epoch`
    (`scenes_v5.scene_mix`; the config's final mix when None), unlike the full-horizon scenes that keep the goal mix."""
    records = []
    K = cfg.validation_iterations
    mix = scenes_v5.scene_mix(cfg, epoch)
    for index in range(cfg.validation_scene_count):
        scene = scenes_v5.held_out_scene(master_seed, index, cfg, mix=mix)
        batch = _held_out_scene_batch(step, cfg, grids, aug, scene, device, master_seed)
        es = energy_scale(batch.material)
        scale = float((torch.maximum(batch.E.abs(), batch.material.floor) * es).sum())
        curves = [scene_metrics(batch)]
        for _ in range(K):
            out = step.query(batch)
            step.commit(batch, out)
            curves.append(scene_metrics(batch))
        records.append(
            {
                "seed": index,
                "scene": scenes_v5.scene_summary(scene),
                **{key: [c[key] for c in curves] for key in SCENE_CURVES},
                "contact_pairs": curves[0]["contact_pairs"],
                "survived": all(c["finite"] for c in curves),
                "failed_first_update": not curves[1]["finite"],
                "scale_joule": scale,
                "contact": True,
            }
        )
        _release(batch)
    return records


@torch.no_grad()
def validate_full_horizon_v5(
    step, cfg, grids, aug, device, master_seed: int, K: int | None = None, H: int | None = None
) -> tuple[list, float]:
    """Per-scene physical records (`cfg.validation_full_scene_count` held-out scenes, K queries x H steps) for
    report.summarize_full_horizon; returns (samples, seconds). K and H default to the config values; the trainer
    passes the growth stage's caps. The first scene's record carries `momentum_drift` (`momentum_drift_check`)."""
    t0 = time.perf_counter()
    K = K or cfg.validation_full_iterations
    H = H or cfg.validation_full_steps
    records = []
    for index in range(cfg.validation_full_scene_count):
        scene = scenes_v5.held_out_scene(master_seed, index, cfg)
        batch = _held_out_scene_batch(step, cfg, grids, aug, scene, device, master_seed)
        rows = []
        alive = True
        for _ in range(H):
            for _ in range(K):
                out = step.query(batch)
                step.commit(batch, out)
            m = scene_metrics(batch)
            alive = m.pop("finite")
            bound = float(getattr(cfg, "reset_penetration_r", 0.0) or 0.0)
            if alive and bound > 0.0 and not m["penetration_r"] <= bound:
                # a collapsed pile (2026-10-03): the same bound as the training guard ends the rollout; such a scene
                # is not a survivor, and its detection over thousands of interpenetrating bodies ran rank 0 out of
                # memory in the v6 run (45 GB of candidate tensors)
                alive = False
                m["collapsed"] = True
            rows.append(m)
            if not alive:
                break
            step.advance(batch, batch.active, None)
        drift = momentum_drift_check(step, cfg, grids, aug, device, master_seed, K, H, scene) if index == 0 else None
        records.append(
            {
                "seed": index,
                "scene": scenes_v5.scene_summary(scene),
                "physical_records": rows,
                "survived": alive and len(rows) == H,
                "momentum_drift": drift,
            }
        )
        _release(batch)
    return records, time.perf_counter() - t0


def contact_free_copy(scene: scenes_v5.SceneV5, H: int) -> scenes_v5.SceneV5:
    """The scene with its bodies on a square grid in (x, z) centred on the origin, so far apart that no two can touch
    within H steps (bounding-box reach plus the rigid travel plus a margin of 4 cells; the grid keeps the float32
    positions small: 159 bodies in a row would reach 95 m), without its static faces; the plane is removed by the
    caller (`scene_batch`)."""
    h, dt = scene.h, scene.dt
    reach = max(float(scenes_v5.half_extents(b.cell_counts, b.quaternion, h).max()) for b in scene.bodies)
    speed = max(float(np.linalg.norm(b.velocity)) for b in scene.bodies)
    spacing = 2.0 * (reach + speed * H * dt) + 4.0 * h
    n = math.ceil(math.sqrt(len(scene.bodies)))
    bodies = [
        dataclasses.replace(
            b, position=((i % n - (n - 1) / 2) * spacing, float(b.position[1]), (i // n - (n - 1) / 2) * spacing)
        )
        for i, b in enumerate(scene.bodies)
    ]
    return dataclasses.replace(scene, bodies=bodies, static_faces=[])


def _corner_mass_si(batch) -> torch.Tensor:
    """Corner masses [N] (kg) of the batch's free bodies, zero on the corners of pinned bodies."""
    si = batch.material.si
    rho = torch.tensor([d["rho"] for d in si], dtype=torch.float64, device=batch.mass.device)
    h = torch.tensor([d["h"] for d in si], dtype=torch.float64, device=batch.mass.device)
    o = batch.corner_obj
    return (rho[o] * batch.mass.to(torch.float64) * h[o] ** 3).masked_fill(~batch.free_objects[o], 0.0)


def momentum_si(batch, V: torch.Tensor) -> torch.Tensor:
    """Total linear momentum [3] (kg m/s) of the batch's free bodies for the velocities V (cells per step)."""
    si = batch.material.si
    h = torch.tensor([d["h"] for d in si], dtype=torch.float64, device=V.device)
    dt = torch.tensor([d["dt"] for d in si], dtype=torch.float64, device=V.device)
    o = batch.corner_obj
    return (_corner_mass_si(batch)[:, None] * V.to(torch.float64) * (h / dt)[o, None]).sum(0)


def total_mass_si(batch) -> float:
    """Total mass (kg) of the batch's free bodies."""
    return float(_corner_mass_si(batch).sum())


@torch.no_grad()
def momentum_drift_check(step, cfg, grids, aug, device, master_seed: int, K: int, H: int, scene=None) -> float:
    """|p(t) - p(0) - t M g| / |M g t| over the free bodies after H steps of K queries on the contact-free copy of
    the held-out scene (`contact_free_copy`, no plane): the solver's momentum drift without contact (free fall of
    every free body's centroid with the shape update in the fusion's null space; body-body detection stays on and
    must find no pair; pinned bodies hang from their pins and are left out). SI; NaN when the state leaves the
    finite range or the scene has no free body."""
    scene = scene if scene is not None else scenes_v5.held_out_scene(master_seed, 0, cfg)
    free = contact_free_copy(scene, H)
    batch = _held_out_scene_batch(step, cfg, grids, aug, free, device, master_seed, plane=False)
    p0 = momentum_si(batch, batch.V)
    g = torch.tensor([float(v) for v in scene.gravity], dtype=torch.float64, device=device)
    t = H * scene.dt
    for _ in range(H):
        for _ in range(K):
            out = step.query(batch)
            step.commit(batch, out)
        if not bool(torch.isfinite(batch.E).all()):
            _release(batch)
            return float("nan")
        step.advance(batch, batch.active, None)
    M = total_mass_si(batch)
    if M == 0.0:
        _release(batch)
        return float("nan")  # no free body: nothing falls freely
    expected = t * M * g
    drift = (momentum_si(batch, batch.V) - p0 - expected).norm() / expected.norm().clamp_min(1e-300)
    value = float(drift)
    _release(batch)
    return value if math.isfinite(value) else float("nan")
