# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Job runner (design spec 6): B GPU-resident slots per rank pull (seed, K, H) jobs from the rank's queue.
Slots own fixed row ranges of one persistent flat Batch; commit / advance / load are batch tensor ops."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from . import scenes
from .batch import Batch
from .grid import GridCache
from .jobs import assign, sample_epoch_jobs, sample_scene_spec, sample_scene_specs
from .structs import FailureRecord, Job, Material, SceneSpec
from .units import material_from_si

Tensor = torch.Tensor


def material_for(spec: SceneSpec, grid, cfg, device) -> Material:
    c = spec.contact or {}
    return material_from_si(
        E=spec.E,
        nu=spec.nu,
        rho=spec.rho,
        eta=spec.eta,
        gravity=spec.gravity,
        h=spec.h,
        dt=spec.dt,
        cell_count=grid.C,
        sample_count=grid.S,
        kappa=float(c.get("kappa", 0.0)),
        beta=float(c.get("beta", 0.0)),
        mu_f=float(c.get("mu_f", 0.0)),
        friction_epsilon=cfg.contact_friction_epsilon,
        floor_scale=cfg.energy_floor_scale,
        device=device,
    )


def seeded_generator(device, *keys: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(np.random.SeedSequence(list(keys)).generate_state(1, dtype=np.uint64)[0] >> 1))
    return gen


class JobRunner:
    def __init__(self, cfg, step, aug, grids: GridCache, rank: int, world: int, device, master_seed: int):
        self.cfg, self.step, self.aug, self.grids = cfg, step, aug, grids
        self.rank, self.world, self.device, self.master_seed = rank, world, torch.device(device), master_seed
        self.B = cfg.batch_size
        self.grid = grids.get(cfg.cell_counts, cfg.pins)
        self.batch = Batch.build([self.grid] * self.B, self.device)
        default = sample_scene_spec(master_seed, 0, cfg)
        self.batch.material = Material.cat([material_for(default, self.grid, cfg, self.device) for _ in range(self.B)])
        self.slot_contact = [{} for _ in range(self.B)]
        self.gens = [seeded_generator(self.device, master_seed, rank, i) for i in range(self.B)]
        self.k = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.h = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.K = torch.ones(self.B, dtype=torch.long, device=self.device)
        self.H = torch.ones(self.B, dtype=torch.long, device=self.device)
        self.jobs: list = [None] * self.B
        self.failures: list = []
        self.resets = 0
        self.loaded_jobs = 0
        self.contact_scenes = 0
        self.queue: deque = deque()
        self.U = 0
        self.epoch = 0
        self.update = 0
        self.specs: dict = {}

    # ------------------------------------------------------------------ epoch
    def start_epoch(self, epoch: int) -> None:
        jobs = sample_epoch_jobs(self.master_seed, epoch, self.cfg)
        queues, self.U = assign(jobs, self.world, self.B)
        self.queue = deque(queues[self.rank])
        self.specs = sample_scene_specs(jobs, self.master_seed, epoch, self.cfg)
        self.epoch, self.update = epoch, 0
        self.failures, self.resets, self.loaded_jobs, self.contact_scenes = [], 0, 0, 0
        self.batch.active.fill_(True)
        self.load(list(range(self.B)))

    # ------------------------------------------------------------------- load
    def load(self, slots: list) -> None:
        b = self.batch
        busy = []
        for i in slots:
            job = self.queue.popleft() if self.queue else None
            self.jobs[i] = job
            if job is None:
                b.active[i] = False
            else:
                busy.append((i, job))
        if not busy:
            return
        idx = torch.tensor([i for i, _ in busy], device=self.device)
        specs = [self.specs[j.seed] for _, j in busy]
        gens = [seeded_generator(self.device, self.master_seed, j.seed, self.epoch, 11) for _, j in busy]
        for (i, j), _gen in zip(busy, gens, strict=True):
            self.gens[i] = seeded_generator(self.device, self.master_seed, j.seed, self.epoch, 13)
        Xs, Vs = self.aug.initial_states([self.grid] * len(busy), specs, gens)
        sel = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        sel[idx] = True
        rows = sel[b.corner_obj]
        X = torch.cat(Xs)
        V = torch.cat(Vs)
        b.X[rows] = X
        b.V[rows] = V
        b.X_prev[rows] = X
        b.x[rows] = X
        cell_rows = sel[b.cell_obj]
        b.hist_grad[cell_rows] = 0
        b.hist_update[cell_rows] = 0
        b.hist_valid[idx] = False
        b.active[idx] = True
        b.material.write(idx, Material.cat([material_for(s, self.grid, self.cfg, self.device) for s in specs]))
        for (i, _), s in zip(busy, specs, strict=True):
            self.slot_contact[i] = s.contact or {}
            self.contact_scenes += int(
                bool(s.contact) and (s.contact.get("plane_present") or len(s.contact.get("points", [])) > 0)
            )
        b.scene = scenes.scenes_for_objects(self.slot_contact, [self.cfg.cell_size] * self.B, self.device)
        self.K[idx] = torch.tensor([j.K for _, j in busy], device=self.device)
        self.H[idx] = torch.tensor([j.H for _, j in busy], device=self.device)
        self.k[idx] = 0
        self.h[idx] = 0
        self.loaded_jobs += len(busy)
        self.step.prepare(b, sel, self.gens)

    # ----------------------------------------------------------------- commit
    def commit(self, out) -> None:
        b = self.batch
        self.step.commit(b, out)
        self.update += 1
        self.k += b.active.long()
        finite = torch.isfinite(b.E)
        bad = (~finite) & b.active
        adv = (self.k >= self.K) & b.active & ~bad
        if bool(adv.any()):
            self.step.advance(b, adv, self.gens)
            self.k[adv] = 0
            self.h[adv] += 1
        done = ((self.h >= self.H) & b.active) | bad
        if bool(done.any()):
            for i in bad.nonzero().flatten().tolist():
                j = self.jobs[i] or Job(-1, 0, 0)
                self.failures.append(
                    FailureRecord(
                        j.seed, j.K, j.H, int(self.k[i]), int(self.h[i]), "non_finite", self.epoch, self.update
                    )
                )
                self.resets += 1
            self.load(done.nonzero().flatten().tolist())

    @property
    def active_count(self) -> int:
        return int(self.batch.active.sum())
