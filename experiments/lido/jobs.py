# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-state regime (epoch-regime note of 2026-09-27, design spec 6.3): jobs, growth timetable, LPT assignment
and the per-state scene specification. Pure functions on numpy seed streams; identical on every rank."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import torch

from .grid import GridCache
from .structs import Job, SceneSpec
from .units import lame

_grids = GridCache("cpu")


def growth_stage(epoch: int, cfg) -> tuple[int, int, int]:
    """(stage_index, K_max, H_max) for a 1-based epoch; the timetable advances every `growth_stage_epochs`."""
    stage = min((epoch - 1) // cfg.growth_stage_epochs, len(cfg.growth_stages) - 1)
    K_max, H_max = cfg.growth_stages[stage]
    return stage, int(K_max), int(H_max)


def job_count(cfg) -> int:
    """Jobs per epoch: `state_count` states in the body regime, `scene_count` scenes in the v5 scene regime."""
    return int(cfg.scene_count if cfg.scene_mode == "v5" else cfg.state_count)


def sample_epoch_jobs(master_seed: int, epoch: int, cfg) -> list[Job]:
    """One Job per state: H ~ U{1..H_max}, K uniform over the powers of two <= K_max with K H <= budget_cap.

    v5 scene regime (`cfg.scene_mode == "v5"`, design spec section 11): a fixed state is a scene, `job_count` =
    `scene_count` jobs with seed = scene index and the same K, H draws; `assign(jobs, world, 1)` (one scene per
    rank at a time) gives U = the rank's sum of K H. Updates per rank per epoch for the v4 growth table with 64
    scenes (expectation; the LPT loads of seed 73 over 4 ranks are within 1 %):

        stage  K_max  H_max   E[sum K H]   per rank of 4   per rank of 1
          0      1      8         288            72             288
          1      2     16         816           204             816
          2      4     32        2464           616            2464
          3      8     64        7800          1950            7800
          4     16    128       25594          6398           25594
          5     32    128       30066          7516           30066

    (E[K H] per job = mean over H of the mean of the admissible k H; the budget cap 2048 binds from stage 4.)
    """
    _, K_max, H_max = growth_stage(epoch, cfg)
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, epoch, 7331]))
    jobs = []
    for state in range(job_count(cfg)):
        H = int(rng.integers(1, H_max + 1))
        ks = [int(k) for k in cfg.iteration_counts if k & (k - 1) == 0 and k <= K_max and k * H <= cfg.budget_cap]
        jobs.append(Job(seed=state, K=int(rng.choice(ks)), H=H))
    return jobs


def assign(jobs: list, world_size: int, B: int) -> tuple[list, int]:
    """LPT by K H: longest job first onto the least loaded rank. U = max(ceil(max load / B), longest K H)."""
    queues = [[] for _ in range(world_size)]
    load = [0] * world_size
    for job in sorted(jobs, key=lambda j: -j.K * j.H):
        r = load.index(min(load))
        queues[r].append(job)
        load[r] += job.K * job.H
    if not jobs:
        return queues, 0
    longest = max(j.K * j.H for j in jobs)
    U = max(math.ceil(max(load) / B), longest)
    return queues, U


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _contact_spec(gen: torch.Generator, cfg, material_si: dict, grid) -> dict:
    if importlib.util.find_spec(".scenes", __package__) is None:
        return {}
    from . import scenes

    return scenes.sample_contact_spec(gen, cfg, material_si, grid)


def sample_scene_spec(master_seed: int, seed: int, cfg, validation: bool = False) -> SceneSpec:
    """The scene of one state (SI): a pure function of (master_seed, seed, validation), independent of the epoch.

    Stream `SeedSequence([master_seed, seed, 1 if validation else 0])`, so held-out validation seeds 0..count-1 never
    collide with training seeds. The contact dict (JSON-serialisable) comes from `scenes.sample_contact_spec` when
    `cfg.contact` is set; it is `{}` when contact is off or the scenes module is absent.
    """
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, seed, 1 if validation else 0]))
    E = _log_uniform(rng, *cfg.youngs_modulus_range)
    nu = float(rng.uniform(*cfg.poissons_ratio_range))
    rho = _log_uniform(rng, *cfg.density_range)
    eta = _log_uniform(rng, *cfg.damping_range)
    g_mag = _log_uniform(rng, *cfg.gravity_magnitude_range)
    g_dir = np.asarray(cfg.gravity, dtype=float)
    gravity = tuple(float(v) for v in g_mag * g_dir / np.linalg.norm(g_dir))
    perturbation_scale = float(rng.uniform(*cfg.perturbation_scale_range))
    strength = float(rng.uniform(*cfg.strength_range))
    velocity_dt = float(rng.uniform(*cfg.velocity_dt_range))
    cell_counts = tuple(int(v) for v in cfg.cell_counts)
    h, dt = float(cfg.cell_size), float(cfg.time_step)
    contact = {}
    if cfg.contact:
        mu, lam = lame(E, nu)
        material_si = {
            "E": E,
            "nu": nu,
            "rho": rho,
            "eta": eta,
            "gravity": gravity,
            "h": h,
            "dt": dt,
            "mu": mu,
            "lam": lam,
        }
        gen = torch.Generator().manual_seed(int(rng.integers(0, 2**63 - 1)))
        contact = _contact_spec(gen, cfg, material_si, _grids.get(cell_counts, cfg.pins))
    return SceneSpec(
        seed=int(seed),
        cell_counts=cell_counts,
        h=h,
        dt=dt,
        pins=cfg.pins,
        E=E,
        nu=nu,
        rho=rho,
        eta=eta,
        gravity=gravity,
        perturbation_scale=perturbation_scale,
        strength=strength,
        velocity_dt=velocity_dt,
        contact=contact,
    )


def sample_scene_specs(jobs: list, master_seed: int, epoch: int, cfg) -> dict:
    """seed -> SceneSpec for every state of the epoch. `epoch` is accepted for the call site's symmetry with
    `sample_epoch_jobs` but does not enter: a fixed state keeps its scene across epochs."""
    return {job.seed: sample_scene_spec(master_seed, job.seed, cfg) for job in jobs}
