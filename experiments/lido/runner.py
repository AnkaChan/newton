# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Job runner (design spec 6): B GPU-resident slots per rank pull (seed, K, H) jobs from the rank's queue.
Slots own fixed row ranges of one persistent flat Batch; commit / advance / load are batch tensor ops.

`SceneRunner` (v5, design spec section 11): the rank holds ONE scene at a time, a fresh Batch of free bodies in a
shared world frame with body-body contact; its K H queries are served one per update, then the next scene loads.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from . import contact, scenes, scenes_v5
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


# ------------------------------------------------------------------------------------------------ v5 scenes
def scene_batch(scene: scenes_v5.SceneV5, grids: GridCache, aug, device, plane: bool = True) -> Batch:
    """A fresh Batch of the realised scene (normalised units, one world frame, origin zero, body contact on), its
    state at step start (X_prev = x = X) and no step-constant tier yet (`Step.prepare` builds it). `plane=False`
    removes the ground plane (the validation's contact-free check)."""
    gs, X, V, material, contact_scene = scenes_v5.realise(scene, grids, aug, device)
    b = Batch.build(gs, device)
    b.body_contact = True
    b.material, b.scene = material, contact_scene
    if not plane:
        b.scene.plane_present.fill_(False)
    b.X, b.V = X, V
    b.X_prev, b.x = X.clone(), X.clone()
    return b


PAIR_KINDS = ("total", "plane", "point", "static", "body")


def pair_counts(batch) -> dict:
    """Valid pairs of the batch's current step by kind: {"total", "plane", "point", "static", "body"} (static = the
    scene's static faces, `partner_body == contact.PARTNER_STATIC`, which share kind 1 with the discs; one host
    sync)."""
    p = batch.pairs
    if p is None or p.count == 0:
        return dict.fromkeys(PAIR_KINDS, 0)
    kind = p.kind[p.valid]
    counts = torch.bincount(kind, minlength=3).tolist()
    static = int((p.partner_body[p.valid] == contact.PARTNER_STATIC).sum())
    point = int(counts[1]) - static
    return {
        "total": int(sum(counts)),
        "plane": int(counts[0]),
        "point": point,
        "static": static,
        "body": int(counts[2]),
    }


class SceneRunner:
    """v5 regime: one scene per rank at a time (design spec section 11).

    `start_epoch` draws `cfg.scene_count` jobs (seed = scene index; K, H from the growth table) and assigns them
    over the ranks by LPT with one slot per rank, so U = the rank's sum of K H (`jobs.sample_epoch_jobs`). `load`
    samples and realises the next job's scene into a fresh Batch (`scene_batch`) and runs `Step.prepare` on all
    bodies with one candidate-noise generator for the scene (seeded by master, scene, epoch; the batched draw of
    `Step.prepare`); every update serves one query of the scene (`commit`: k += 1; k == K: `Step.advance` on all bodies,
    h += 1; h == H: load the next scene). A non-finite energy of any body fails the scene: a FailureRecord with
    the scene's (seed, K, H, k, h) and the next scene loads. When the rank's queue is empty the last batch stays
    with `active` all False (the trainer's loss mask) and a finite state (`_idle`), so a rank with less load idles
    through the common U with a finite loss and zero gradients.
    """

    def __init__(self, cfg, step, aug, grids: GridCache, rank: int, world: int, device, master_seed: int):
        self.cfg, self.step, self.aug, self.grids = cfg, step, aug, grids
        self.rank, self.world, self.device, self.master_seed = rank, world, torch.device(device), master_seed
        self.batch: Batch | None = None
        self.scene: scenes_v5.SceneV5 | None = None
        self.job: Job | None = None
        self.gens: torch.Generator | None = None  # the scene's candidate-noise stream
        self.k = 0
        self.h = 0
        self.queue: deque = deque()
        self.U = 0
        self.epoch = 0
        self.update = 0
        self.failures: list = []
        self.resets = 0
        self.loaded_jobs = 0
        self.contact_scenes = 0
        self.queries = 0  # body queries served this epoch (bodies of the scene per update)
        self.idle_updates = 0  # updates after the rank's queue ran dry
        self.pairs = dict.fromkeys(PAIR_KINDS, 0)  # valid pairs of the current step
        self.pair_history: list = []  # pair counts at every detection of the epoch
        self.scene_summaries: list = []  # scenes_v5.scene_summary of every loaded scene plus its job

    # ------------------------------------------------------------------ epoch
    def start_epoch(self, epoch: int) -> None:
        jobs = sample_epoch_jobs(self.master_seed, epoch, self.cfg)
        if len(jobs) < self.world:
            raise ValueError(
                f"scene_count {len(jobs)} is below the world size {self.world}: a rank would have no scene"
            )
        queues, self.U = assign(jobs, self.world, 1)
        self.queue = deque(queues[self.rank])
        self.epoch, self.update = epoch, 0
        self.failures, self.resets, self.loaded_jobs, self.contact_scenes = [], 0, 0, 0
        self.queries, self.idle_updates = 0, 0
        self.pair_history, self.scene_summaries = [], []
        self.load()

    # ------------------------------------------------------------------- load
    def load(self) -> None:
        self.job = self.queue.popleft() if self.queue else None
        if self.job is None:
            if self.batch is not None:
                self._idle(self.batch)
            return
        job = self.job
        scene = scenes_v5.sample_scene(self.master_seed, job.seed, self.cfg)
        b = scene_batch(scene, self.grids, self.aug, self.device)
        # one candidate-noise generator per scene (Step.prepare draws every grid group in one call)
        self.gens = seeded_generator(self.device, self.master_seed, job.seed, self.epoch, 13)
        self.step.prepare(b, b.active, self.gens)
        self.batch, self.scene = b, scene
        self.k = self.h = 0
        self.loaded_jobs += 1
        self.contact_scenes += int(bool(b.scene.plane_present.any()))
        self.scene_summaries.append({**scenes_v5.scene_summary(scene), "K": job.K, "H": job.H})
        self._record_pairs()

    @staticmethod
    @torch.no_grad()
    def _idle(b: Batch) -> None:
        """The rank's batch while its queue is empty: `active` all False (the trainer's loss mask) and a FINITE state
        for the idle queries. A scene that failed with a non-finite energy must not stay as it is: the trainer's loss
        on the masked rows is 0 x NaN = NaN (review finding 2, 2026-10-02), and so are the gradients that reach the
        all-reduce under DDP. The candidate goes back to the step-start positions (finite: a failure is caught before
        `advance`), the energies and gradients to zero, the history to invalid."""
        b.active.fill_(False)
        X = torch.where(torch.isfinite(b.X), b.X, b.rest)  # X is finite unless the state was never finite
        b.X.copy_(X)
        b.x.copy_(X)
        b.E.zero_()
        b.gX.zero_()
        b.hist_grad.zero_()
        b.hist_update.zero_()
        b.hist_valid.fill_(False)
        b.picard_constant.zero_()

    def _record_pairs(self) -> None:
        self.pairs = pair_counts(self.batch)
        self.pair_history.append(self.pairs)

    # ----------------------------------------------------------------- commit
    def commit(self, out) -> None:
        b = self.batch
        self.step.commit(b, out)
        self.update += 1
        if self.job is None:
            self.idle_updates += 1
            return
        self.queries += b.O
        self.k += 1
        if not bool(torch.isfinite(b.E).all()):
            j = self.job
            self.failures.append(FailureRecord(j.seed, j.K, j.H, self.k, self.h, "non_finite", self.epoch, self.update))
            self.resets += 1
            self.load()
            return
        if self.k >= self.job.K:
            self.step.advance(b, b.active, self.gens)
            self.k = 0
            self.h += 1
            self._record_pairs()
        if self.h >= self.job.H:
            self.load()

    @property
    def active_count(self) -> int:
        return self.batch.O if self.job is not None else 0

    def epoch_summary(self) -> dict:
        """The epoch's scene statistics for the run record."""
        hist = self.pair_history
        n = max(1, len(hist))
        return {
            "scenes": self.loaded_jobs,
            "bodies_mean": float(np.mean([s["bodies"] for s in self.scene_summaries])) if self.scene_summaries else 0.0,
            "pinned_bodies_mean": float(np.mean([s["pinned_bodies"] for s in self.scene_summaries]))
            if self.scene_summaries
            else 0.0,
            "resting_bodies_mean": float(np.mean([s.get("resting_bodies", 0) for s in self.scene_summaries]))
            if self.scene_summaries
            else 0.0,
            "static_faces_mean": float(np.mean([s.get("static_faces", 0) for s in self.scene_summaries]))
            if self.scene_summaries
            else 0.0,
            "cells_mean": float(np.mean([s["cells"] for s in self.scene_summaries])) if self.scene_summaries else 0.0,
            "body_queries": self.queries,
            "idle_updates": self.idle_updates,
            "detections": len(hist),
            "pairs_mean": sum(p["total"] for p in hist) / n,
            "body_pairs_mean": sum(p["body"] for p in hist) / n,
            "body_pairs_max": max((p["body"] for p in hist), default=0),
            "plane_pairs_mean": sum(p["plane"] for p in hist) / n,
            "static_pairs_mean": sum(p.get("static", 0) for p in hist) / n,
            "steps_with_body_pairs": sum(1 for p in hist if p["body"] > 0) / n,
            "scenes_served": self.scene_summaries,
        }


def make_runner(cfg, step, aug, grids: GridCache, rank: int, world: int, device, master_seed: int):
    """The runner of `cfg.scene_mode`: `JobRunner` ("body") or `SceneRunner` ("v5")."""
    if cfg.scene_mode == "v5":
        return SceneRunner(cfg, step, aug, grids, rank, world, device, master_seed)
    if cfg.scene_mode == "body":
        return JobRunner(cfg, step, aug, grids, rank, world, device, master_seed)
    raise ValueError(f"scene_mode must be 'body' or 'v5', got {cfg.scene_mode!r}")
