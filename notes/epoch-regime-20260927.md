# Fixed-state epoch regime for LIDO training — decided 2026-09-27

Anka's proposal (2026-09-27, evening), replacing the open-ended trajectory pool.

## Decisions

| Topic | Decision |
|---|---|
| Training states | A fixed set of 2048 initial states (seeded augmenter + contact scene per state), reused every epoch. |
| Epoch | At the start of each epoch, sample (K, H) for every state; run every state through its K x H queries (each query is one Adam update, intermediate states included, as before); the epoch ends when all states are done. |
| Budget cap | K x H <= 1024 (= 8 x 128): long horizons run few iterations, short horizons may run many. |
| Sampling | Per state per epoch: H uniform over the integers 1..128 (every trajectory starts at step 1 and runs H steps); K uniform over {1, 2, 4, 8, 16, 32} capped at 1024 / H. Mean about 300 queries per state, about 610k queries per epoch, about 1.9 h per epoch on four L40s at current throughput. (Anka, 2026-09-27: 'H should start from 1 as well'.) |
| Growth (Anka, 2026-09-27: 'both start from 1 and grow') | Fixed timetable, advancing every 2 epochs, no validation gating: (K max, H max) = (1, 8) -> (2, 16) -> (4, 32) -> (8, 64) -> (16, 128) -> (32, 128), then fixed. Within a stage H is uniform over 1..H max and K uniform over the powers of two up to K max, capped at 1024 / H. Early stages have few queries per state (about 9k, 26k, 80k, 250k, 600k, 610k queries per epoch), so the ramp takes about 6 h before the full regime. |
| Initialisation | Run 3 resumes the network and optimizer from run 2's best_validation.pt (epoch 56, 15.0 N); the pool state of run 2 is discarded. |
| Validation | Every epoch (cheap: 64 states x 32 iterations + 8-step check; full horizon: 8 states, K = 8, H = 128). Validation states stay a disjoint fixed seed set. |
| Learning rate | Cosine by epoch, 1e-4 -> 2.5e-5 over the new max_epochs (about 60 for a 5-day run). |
| Distributed | States are assigned to ranks with balanced sum of K x H; the tail is padded so all ranks perform the same number of updates. |

## Why

The pool spent many queries at K = 16..32 with H = 128 (4096 queries per trajectory), far
from the K = 8 inference regime, while the long-horizon full-horizon check kept ending at
~270 J / 16 r / 2 inverted. Long sequences at few iterations train exactly the compounding
behaviour we roll out. A fixed state set makes epochs comparable and reproducible.

## Status

Run 2 (generated/training_v3_contact_20260927_r2) stopped 2026-09-27 ~23:50 UTC at
epoch 62 (best 13.1 N at epoch 60) to free the GPUs; run 3 to be launched from its best checkpoint once the
trainer change lands.

## Implementation record — 2026-09-28

Landed uncommitted in worktree `learned-instrinic-solver` (branch `ankac/learned-instrinic-solver`,
on top of `123a6be9`); nothing was launched.

**What landed**

- `trajectory_pool.py`: job-list mode. `ActiveTrajectoryPool.set_jobs(jobs)` (pool built with
  `initialize=False`) replaces the reset stream by an explicit list of `(seed, K, H)` jobs, each run
  exactly once with its own budgets; `remaining_queries` / `exhausted` let the trainer assert that
  exactly `U x B` queries were served. Capacity is the preferred live count; the tail may exceed it.
  A pool that never calls `set_jobs` is unchanged (its 15 existing tests pass unmodified).
- `train_mixed.py`: `MixedTrainConfig` gains `regime` (`pool` | `fixed_states`), `state_count`,
  `budget_cap`, `growth_stage_epochs`, `growth_stages` (validated: K_max a power of two <= 32,
  H_max <= 128, non-decreasing, positive counts); pure functions `fixed_state_stage`,
  `sample_epoch_jobs(config, epoch, master_seed)` (stream `SeedSequence([master_seed, epoch, 7331])`)
  and `assign_jobs(jobs, world_size, batch_size) -> JobAssignment` (LPT by K x H, own-seed fillers,
  common `U`); the trainer loop runs `U` full updates per epoch in the fixed regime, takes available
  K/H for the full-horizon check from the growth stage, records
  `row["regime"] = {name, stage, k_max, h_max, queries, filler_queries, updates}` and
  `row["curriculum"] = None`, and includes the stage in progress heartbeats and the verbose line.
  Weights-only start: `run_training(..., resume=path, resume_weights_only=True)` and
  `--resume-weights-only`; only the 13 architecture fields must match, `report["initialized_from"]`
  records checkpoint path, sha256, completed epochs/updates, best selection and every differing field.
- `launch_training.py`: `--resume-weights-only` (checkpoint may live in another run; output must be
  fresh; `launcher.json` records `resume_weights_only`; the launcher log prints the origin line).
- `mixed_report.py`: `write_progress(..., regime=...)` adds a `regime` block to `progress.json`;
  `epochs.csv` gains `regime_stage`, `regime_k_max`, `regime_h_max`, `regime_updates` (blank for pool
  rows). Nothing in the report ever read `curriculum`, so `None` needs no further tolerance.
- `generated/training_v3_fixed_config.json` (contact config + fixed regime, 60 epochs, validation
  64 x 32 every epoch, full horizon 8 states / K 8 every 5 epochs, cosine 1e-4 -> 2.5e-5, no early
  stopping), `generated/start_training_v3_fixed.sh` (weights-only from run 2's `best_validation.pt`
  into `generated/training_v3_fixed_20260928`, log `generated/training_v3_fixed_launcher.log`) and
  `generated/resume_training_v3_fixed.sh <checkpoint>`.
- Tests: `tests/test_fixed_state_regime.py` (16 tests: sampling determinism/ranges/stages, assignment
  balance/purity/fillers, pool job mode incl. a randomized full-batch property, CPU end-to-end run on
  grid (2,2,3), fixed-regime resume, weights-only start, config validation, trainer CLI and launcher).

**Campaign-scale check** (`sample_epoch_jobs` + `assign_jobs` on the fixed config, 4 ranks, B = 16):
stage 0 (epoch 1) 9.3k queries, U = 146; stage 1 25.7k, U = 401; stage 2 78.8k, U = 1231;
stage 3 252k, U = 3945; stage 4 572k, U = 8941; stage 5 617-627k, U ≈ 9650-9800. Rank loads differ
by at most one query; fillers are 0-16 queries per rank per epoch.

**Deviations from the contract, and why**

1. `U = max(ceil(max_rank_queries / B), longest K x H)` instead of the ceiling alone. A trajectory
   receives at most one query per update, so an epoch with `U` smaller than its longest job cannot
   end in full batches whatever the scheduler does. In the campaign `U` is 146-9800 against a longest
   job of at most 1024, so the term never binds; it matters for tiny configurations (tests).
2. Job-mode dispatch is the usual FIFO rotation except that "critical" trajectories (remaining
   queries equal to the remaining update count) are served first and critical unstarted jobs start at
   once, exceeding the capacity if needed. Plain FIFO strands long jobs at the tail even when the
   total is a multiple of B (B = 2, jobs 8,1,1,1,1,1,1,1,1: FIFO ends with one record holding 4
   queries). Invariant kept by the rule: with `n` updates left every trajectory needs <= n queries,
   at most B of them exactly n, so the last update is full. Bulk-of-epoch behaviour and CPU overlap
   are unchanged.
3. Checkpoints of fixed-state runs store `rank_states[r]["pool"] = None` and no contexts (all are
   retired at the epoch end); `state_dict` raises in job mode; an ordinary resume simply starts the
   next epoch's job list, which is a pure function of `(seed, epoch)`.
4. In the fixed regime `_TrajectoryFactory` gives every reset a unique context key
   (`train-{rank}-{seed}-{n}`) because a filler job may run a seed concurrently with that seed's main
   job; the pool regime keeps its keys.
5. Pool-regime epoch rows are byte-for-byte as before (no `regime` key); `regime["queries"]` counts the
   sampled `K x H` without fillers while `row["query_count"] = U x B x world_size` includes them.
6. Early stopping in the fixed regime (`_allow_early_stop_fixed`) is permitted only after
   `plateau_min_final_stage_epochs` epochs in the final growth stage (the curriculum analogue); the
   campaign disables it anyway.
7. The launcher accepts `--resume-weights-only` for `--pipeline mixed` only. No `source.json` is
   written: that manifest is produced by hand at launch time.
8. `world_size` is not compared for a weights-only start (the network does not care).

**Verification** (CPU only, `CUDA_VISIBLE_DEVICES=""`)

- Baseline before the change (shared worktree, HEAD `123a6be9`): `Ran 574 tests in 141.676s — OK (skipped=5)`.
- Existing pool-regime modules after the change (`test_train_mixed`, `test_trajectory_pool`,
  `test_launch_training`, `test_mixed_report`, `test_training_cli`, `test_mixed_training_failures`):
  `Ran 72 tests in 68.542s — OK`, unmodified.
- New module `tests/test_fixed_state_regime.py`: `Ran 16 tests in 14.754s` (all pass; one fixture was
  corrected during development).
- Whole suite, run on an export of HEAD plus the five changed/new files (see the warning below):
  `Ran 590 tests in 152.891s — OK (skipped=5)`. `uvx ruff format --check` / `uvx ruff check` pass on
  every touched file.

**Warning — concurrent edit in the shared worktree.** At 00:19 UTC another workflow rewrote
`experiments/learned_intrinsic_solver/features.py` (seven dimensionless conditioning channels,
`conditioning_channels(..., gravity)`, `CONDITIONING_DIM` 9 -> 7) without yet updating its callers
(`mixed_physics.py`, `solver_step.py`). In the worktree itself every mixed-trainer test, including the
untouched pool-regime smoke test, currently fails with `TypeError: conditioning_channels() missing 1
required positional argument: 'gravity'`; this is independent of the regime change, which is why the
whole-suite figure above was measured on a HEAD export. Two consequences for run 3: (a) the suite in the
worktree is only green once that edit is completed or set aside; (b) if the new conditioning schema
lands, run 2's `best_validation.pt` (schema 4, nine channels) no longer fits the network's FiLM input
and the planned weights-only start of run 3 is rejected on `feature_schema_version` / parameter shapes.
Anka decides the order: start run 3 from the run-2 weights under the current schema, or retrain the
conditioning from scratch under the new one.
Measured in the shared worktree at 00:27 UTC, while that edit was still landing (`mixed_physics.py`,
`network.py`, `solver_step.py` changed at 00:25): `Ran 590 tests in 116.602s — FAILED (failures=18,
errors=89, skipped=5)`; 87 of the 107 are the `gravity` TypeError, the rest are conditioning-width and
normalisation assertions of the same schema change. None involve the regime code paths beyond their
shared use of the physics step.

### Review fixes — 2026-09-28 (post-review, uncommitted in the same worktree)

- **Stationary epoch mixture.** `assign_jobs(..., shuffle_seed=(master_seed, epoch))` permutes every
  rank's sampled jobs with `SeedSequence([master_seed, epoch, rank, 7332])` after the LPT assignment
  and before the fillers; the trainer passes the seed, so the pool no longer runs an epoch from the
  longest budgets down to the shortest. `shuffle_seed=None` keeps the LPT order (pure function; the
  counts, `U` and fillers are unchanged by the permutation).
- **Per-rank fillers.** `row["regime"]["filler_queries"]` is now the per-rank list (was rank 0 only).
- **Coordinated exhaustion guard.** The end-of-epoch "queries unserved" check goes through
  `_all_ranks_ok` like every other loop failure.
- **Heartbeats.** `progress.json` carries the `regime` block in the `initializing`, `complete` and
  `failed` phases too (stage, K_max, H_max; counts once the jobs are assigned).
- **Weights-only shape check.** Before hashing the checkpoint or touching the output, the saved
  `network_state` tensor shapes are compared with the freshly built network; a mismatch the
  architecture fields do not cover (for example a conditioning-width change without a schema bump)
  raises the same "requires the checkpoint architecture" `ValueError`.
- **Dashboard.** `index.html` shows the fixed-state status line (states, cap, growth stage, sampled
  and filler queries, updates per rank), the growth timetable instead of the curriculum gate, and the
  weights-only origin (`report["initialized_from"]`); pool-regime pages are unchanged. The four
  `regime_*` columns of `epochs.csv` moved to the end of the table.
- `generated/start_training_v3_fixed.sh` appends to its launcher log and documents that a failed
  start leaves the output directory behind.
- Not changed (design questions for Anka): the candidate stream `SeedSequence([master_seed, seed,
  physical_age, 911])` repeats per state every epoch (fillers duplicate the seed's first query);
  `report["updates"]` grows one row per update (~590k rows over 60 epochs).
- **Launch blocker unchanged:** the worktree's uncommitted 7-channel conditioning edit
  (`features.py`, schema still 4) makes run 2's `best_validation.pt` (9 channels) unloadable; with the
  shape check the start now fails cleanly with the architecture `ValueError` instead of a raw size
  mismatch, but run 3 cannot start from the run-2 weights until that edit is reverted, stashed or
  landed with a schema bump and a retrain.
