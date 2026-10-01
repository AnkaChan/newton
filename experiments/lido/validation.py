# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validation on held-out states (design spec 7.3): cheap (one physical step, K queries) and full horizon
(K queries x H steps). Frozen weights, inertial candidates, SI units in the records."""

from __future__ import annotations

import time

import torch

from . import contact, physics, scenes
from .batch import Batch
from .jobs import sample_scene_spec
from .runner import material_for, seeded_generator
from .structs import Material
from .units import energy_scale, force_scale

Tensor = torch.Tensor


def local_objective(E_after: Tensor, E_before: Tensor, floor: Tensor, increase_weight: float = 1.0) -> Tensor:
    scale = torch.maximum(E_before.abs(), floor).detach()
    return torch.asinh(E_after / scale) + increase_weight * torch.relu((E_after - E_before.detach()) / scale)


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
