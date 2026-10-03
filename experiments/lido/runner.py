# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Job runner (design spec 6): B GPU-resident slots per rank pull (seed, K, H) jobs from the rank's queue.
Slots own fixed row ranges of one persistent flat Batch; commit / advance / load are batch tensor ops.

`SceneRunner` (v5, design spec section 11): the rank holds ONE scene at a time, a fresh Batch of free bodies in a
shared world frame with body-body contact; its K H queries are served one per update, then the next scene loads.
"""

from __future__ import annotations

import gc
import time
from collections import deque

import numpy as np
import torch

from . import contact, scenes, scenes_v5
from .batch import Batch
from .grid import GridCache
from .jobs import assign, growth_stage, sample_epoch_jobs, sample_scene_spec, sample_scene_specs
from .structs import FailureRecord, Job, Material, SceneSpec
from .units import material_from_si

Tensor = torch.Tensor


PRE_ROLL_BACKTRACK = 15  # a truncated pre-roll restores the state this many steps before the guard tripped (2026-10-03)


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
        physical_floor=cfg.physical_floor,
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
        finite = torch.isfinite(b.E) & (
            b.E <= self.cfg.blowup_energy_factor * b.material.floor
        )  # diverged states count as failures
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
def scene_batch(
    scene: scenes_v5.SceneV5, grids: GridCache, aug, device, plane: bool = True, physical_floor: bool = True
) -> Batch:
    """A fresh Batch of the realised scene (normalised units, one world frame, origin zero, body contact on), its
    state at step start (X_prev = x = X) and no step-constant tier yet (`Step.prepare` builds it). `plane=False`
    removes the ground plane (the validation's contact-free check)."""
    gs, X, V, material, contact_scene = scenes_v5.realise(scene, grids, aug, device, physical_floor=physical_floor)
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
    the scene's (seed, K, H, k, h) and the next scene loads; so does a contact penetration deeper than
    `cfg.reset_penetration_r` (`_guard`). When the rank's queue is empty the last batch stays with `active` all False
    (the trainer's loss mask) and a finite state (`_idle`), so a rank with less load idles through the common U with
    a finite loss and zero gradients.

    Pre-roll (Anka, 2026-10-02; `cfg.pre_roll_max_steps`, `cfg.pre_roll_queries`). Once the growth table has
    reached its last stage, every loaded scene first runs n ~ U{0..pre_roll_max_steps} physical steps of
    `pre_roll_queries` queries each, inference only (`_pre_roll`: no gradient, no loss, no optimizer step, the
    network in eval mode for these queries and back in train mode after), so the training window of K x H steps
    starts from a state the solver reached itself rather than from the placement. The same guards apply: a scene
    that blows up or penetrates deeply during its pre-roll fails with kind "pre_roll_" + the guard's kind and the
    next scene loads. n is the first draw of the scene's generator stream (master seed, scene, epoch), so it is
    reproducible; while the pre-roll is off nothing is drawn and the stream is the one of the earlier stages. The
    pre-roll runs inside `load` on every rank independently and leaves U alone (the rank's training updates stay
    its sum of K H), so DDP ranks keep the same number of updates; its forward passes run without gradients, which
    under DDP means no collective. Per scene the summary carries "pre_roll" {steps run, drawn, seconds};
    `epoch_summary` adds pre_roll_steps_mean and pre_roll_seconds.
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
        self.pair_history: list = []  # pair counts at every detection of the epoch (the training window's)
        self.scene_summaries: list = []  # scenes_v5.scene_summary of every loaded scene plus its job and pre-roll
        self.pre_roll_max = 0  # the epoch's pre-roll bound: cfg.pre_roll_max_steps at the last growth stage, else 0
        self.pre_roll_seconds = 0.0  # wall time of the epoch's pre-rolls

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
        self.mix = scenes_v5.scene_mix(self.cfg, epoch)
        stage, _, _ = growth_stage(epoch, self.cfg)  # the pre-roll runs at the growth table's last stage only
        self.pre_roll_max = int(self.cfg.pre_roll_max_steps) if stage == len(self.cfg.growth_stages) - 1 else 0
        self.failures, self.resets, self.loaded_jobs, self.contact_scenes = [], 0, 0, 0
        self.queries, self.idle_updates, self.pre_roll_seconds = 0, 0, 0.0
        self.pre_roll_truncated = 0
        self.refills = 0
        self.pair_history, self.scene_summaries = [], []
        self.pre_rolled: dict = {}
        if self.pre_roll_max > 0:
            self._pre_roll_epoch()
        self.load()

    def _pre_roll_epoch(self) -> None:
        """Pre-roll every scene of the rank's queue up front, before the epoch's first training update
        (2026-10-03): the ranks then pre-roll concurrently with no collective between them, where a pre-roll inside
        `load` stalled the other ranks at every gradient all-reduce until it was over, so the pre-rolls were in
        effect serialised across the ranks (the v6 run's updates fell seven-fold). Each scene is realised, prepared
        and pre-rolled with the epoch-start weights by `_pre_roll`; its settled state (X, V, X_prev) is kept on the
        host and restored by `load`, which prepares it again (the query history starts afresh there, which the
        inline pre-roll kept). A scene that fails its pre-roll is recorded (kind "pre_roll_" + the guard's kind) and
        dropped from the queue."""
        kept: deque = deque()
        for job in list(self.queue):
            scene = scenes_v5.sample_scene(
                self.master_seed, scenes_v5.epoch_scene_seed(self.epoch, job.seed), self.cfg, mix=self.mix
            )
            b = scene_batch(scene, self.grids, self.aug, self.device, physical_floor=self.cfg.physical_floor)
            self.gens = seeded_generator(self.device, self.master_seed, job.seed, self.epoch, 13)
            n = self._pre_roll_length()
            self.step.prepare(b, b.active, self.gens)
            kind, k, h, seconds = self._pre_roll(b, n)
            self.pre_roll_seconds += seconds
            if kind is not None and not kind.startswith("truncated_"):
                self.failures.append(
                    FailureRecord(job.seed, job.K, job.H, k, h, "pre_roll_" + kind, self.epoch, self.update)
                )
                self.resets += 1
            else:
                self.pre_rolled[job.seed] = {
                    "X": b.X.detach().to("cpu", copy=True),
                    "V": b.V.detach().to("cpu", copy=True),
                    "X_prev": b.X_prev.detach().to("cpu", copy=True),
                    "steps": h,
                    "drawn": n,
                    "seconds": seconds,
                    "truncated": kind[len("truncated_") :] if kind else None,
                }
                kept.append(job)
            b.release()
            del b
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.queue = kept

    # ------------------------------------------------------------------- load
    def load(self) -> None:
        """The next scene of the rank's queue, pre-rolled when the pre-roll is on; a scene that fails during its
        pre-roll is recorded (kind "pre_roll_" + the guard's kind) and the next one loads. An empty queue leaves the
        last batch idle (`_idle`)."""
        while True:
            self.job = self.queue.popleft() if self.queue else None
            if self.job is None and getattr(self.cfg, "refill_scenes", False) and self.update < self.U:
                self.job = self._refill_job()  # the queue ran dry before the rank's update budget: a fresh scene
            if self.job is None:
                if self.batch is not None:
                    self._idle(self.batch)
                return
            job = self.job
            if self.batch is not None:  # return the previous scene's memory before building the next one
                self.batch.release()
                self.batch = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            # fresh scenes every epoch, composed by the epoch's curriculum mix (scenes_v5.scene_mix)
            scene = scenes_v5.sample_scene(
                self.master_seed, scenes_v5.epoch_scene_seed(self.epoch, job.seed), self.cfg, mix=self.mix
            )
            b = scene_batch(scene, self.grids, self.aug, self.device, physical_floor=self.cfg.physical_floor)
            # one candidate-noise generator per scene (Step.prepare draws every grid group in one call); the
            # pre-roll length is the stream's first draw
            self.gens = seeded_generator(self.device, self.master_seed, job.seed, self.epoch, 13)
            n = self._pre_roll_length()
            state = self.pre_rolled.pop(job.seed, None)  # settled by _pre_roll_epoch: restore, then prepare
            if state is not None:
                b.X.copy_(state["X"].to(self.device))
                b.V.copy_(state["V"].to(self.device))
                b.X_prev.copy_(state["X_prev"].to(self.device))
                b.x.copy_(b.X)
            self.step.prepare(b, b.active, self.gens)
            self.batch, self.scene = b, scene
            self.k = self.h = 0
            self.loaded_jobs += 1
            self.contact_scenes += int(bool(b.scene.plane_present.any()))
            summary = {**scenes_v5.scene_summary(scene), "K": job.K, "H": job.H}
            summary["refill"] = job.seed >= int(self.cfg.scene_count)  # drawn by _refill_job, not an epoch scene
            self.scene_summaries.append(summary)
            truncated = None
            if state is not None:
                kind, k, h, seconds, n = None, 0, state["steps"], state["seconds"], state["drawn"]
                truncated = state.get("truncated")
            else:
                kind, k, h, seconds = self._pre_roll(b, n)
                self.pre_roll_seconds += seconds
                if kind is not None and kind.startswith("truncated_"):
                    truncated, kind = kind[len("truncated_") :], None
            summary["pre_roll"] = {"steps": h, "drawn": n, "seconds": seconds, "truncated": truncated}
            if kind is not None:
                self.failures.append(
                    FailureRecord(job.seed, job.K, job.H, k, h, "pre_roll_" + kind, self.epoch, self.update)
                )
                self.resets += 1
                continue
            self._record_pairs()
            return

    def _refill_job(self) -> Job:
        """A fresh job for a rank whose queue ran dry before its update budget (2026-10-03: guard replacements
        forfeit a scene's remaining K x H, so a rank could idle for most of an epoch): K and H drawn as
        `jobs.sample_epoch_jobs` draws them for this epoch, the scene seed unique per rank and refill and above the
        epoch's scene indices; refills are pre-rolled like the queued scenes were not (no up-front state), i.e.
        they start from the generator's state."""
        self.refills += 1
        _, k_max, h_max = growth_stage(self.epoch, self.cfg)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.master_seed, self.epoch, self.rank, self.refills, 9173])
        )
        H = int(rng.integers(1, h_max + 1))
        ks = [
            int(k)
            for k in self.cfg.iteration_counts
            if k & (k - 1) == 0 and k <= k_max and k * H <= self.cfg.budget_cap
        ]
        seed = int(self.cfg.scene_count) + self.rank * 4096 + self.refills
        return Job(seed=seed, K=int(rng.choice(ks)), H=H)

    def _pre_roll_length(self) -> int:
        """n ~ U{0..pre_roll_max} from the scene's generator stream (its first draw, before the candidate noise of
        `Step.prepare`); 0 without a draw while the pre-roll is off. One host synchronisation per scene."""
        if self.pre_roll_max <= 0:
            return 0
        return int(torch.randint(0, self.pre_roll_max + 1, (), generator=self.gens, device=self.device))

    def _pre_roll(self, b: Batch, n: int) -> tuple[str | None, int, int, float]:
        """n inference-only physical steps of `cfg.pre_roll_queries` queries each on the freshly prepared scene
        (Anka, 2026-10-02): under `torch.no_grad`, the network in eval mode for these queries if it was training
        (restored after), no loss, no optimizer step; every query passes `_guard`. Returns (failure kind or None,
        queries into the failing step, steps completed, wall seconds); (None, 0, 0, 0.0) for n = 0."""
        if n <= 0:
            return None, 0, 0, 0.0
        t0 = time.perf_counter()
        net = self.step.net
        was_training = net.training
        if was_training:
            net.eval()
        try:
            with torch.no_grad():
                recent: deque = deque(maxlen=PRE_ROLL_BACKTRACK)  # step-start states of the last steps, known sane
                for h in range(n):
                    recent.append((h, b.X.clone(), b.V.clone(), b.X_prev.clone()))
                    for k in range(self.cfg.pre_roll_queries):
                        self.step.commit(b, self.step.query(b))
                        kind = self._guard(b)
                        if kind is not None:
                            # 2026-10-03: the pre-roll is truncated instead of dropping the scene (dropping most of a
                            # rank's scenes left it idle for two thirds of epoch 11), and PRE_ROLL_BACKTRACK steps
                            # before the violation rather than at the last step (a state on the verge of the bound
                            # tripped the training guard at once, which forfeited the scene's budget): the training
                            # window starts from the restored state and the training guard takes it from there
                            h0, X0, V0, Xp0 = recent[0]
                            b.X.copy_(X0)
                            b.V.copy_(V0)
                            b.X_prev.copy_(Xp0)
                            b.x.copy_(b.X)
                            self.step.prepare(b, b.active, self.gens)
                            self.pre_roll_truncated += 1
                            return "truncated_" + kind, k + 1, h0, time.perf_counter() - t0
                    self.step.advance(b, b.active, self.gens)
        finally:
            if was_training:
                net.train()
        return None, 0, n, time.perf_counter() - t0

    def _guard(self, b: Batch) -> str | None:
        """The failure kind of the batch's committed query, or None: "non_finite" when a body's energy is not
        finite or exceeds `cfg.blowup_energy_factor` x its loss floor (a diverged state), "penetration" when the
        deepest penetration over the step's frozen pairs exceeds `cfg.reset_penetration_r` sample radii (NaN counts
        as deep; 0 = off). Up to two host synchronisations."""
        if not bool((torch.isfinite(b.E) & (b.E <= self.cfg.blowup_energy_factor * b.material.floor)).all()):
            return "non_finite"
        if self.cfg.reset_penetration_r > 0.0 and b.pairs is not None and b.pairs.count > 0:
            deepest = float(contact.penetration(b, b.x).max())
            if not deepest <= self.cfg.reset_penetration_r:
                return "penetration"
        return None

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
        kind = self._guard(b)
        if kind is not None:
            j = self.job
            self.failures.append(FailureRecord(j.seed, j.K, j.H, self.k, self.h, kind, self.epoch, self.update))
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
            "pinned_fraction": self.mix.pinned_fraction,
            "resting_fraction": self.mix.resting_fraction,
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
            "pre_roll_steps_mean": float(np.mean([s["pre_roll"]["steps"] for s in self.scene_summaries]))
            if self.scene_summaries
            else 0.0,
            "pre_roll_seconds": self.pre_roll_seconds,
            "pre_roll_truncated": self.pre_roll_truncated,
            "refill_scenes": self.refills,
            "scenes_served": self.scene_summaries,
        }


def make_runner(cfg, step, aug, grids: GridCache, rank: int, world: int, device, master_seed: int):
    """The runner of `cfg.scene_mode`: `JobRunner` ("body") or `SceneRunner` ("v5")."""
    if cfg.scene_mode == "v5":
        return SceneRunner(cfg, step, aug, grids, rank, world, device, master_seed)
    if cfg.scene_mode == "body":
        return JobRunner(cfg, step, aug, grids, rank, world, device, master_seed)
    raise ValueError(f"scene_mode must be 'body' or 'v5', got {cfg.scene_mode!r}")
