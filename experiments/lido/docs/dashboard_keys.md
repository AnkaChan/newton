# LIDO training dashboard: data keys

Every key path that the dashboard pipeline reads from a training run directory's
`report.json`, `progress.json` and `epochs.csv`, with type, unit, meaning and
whether the reader tolerates its absence.

Cross-checked on 2026-09-30 against `generated/training_v4_20260928/` (25 completed
epochs, `fixed_states` regime, `status = running`) and against the currently served
copy in `~/.bun/install/global/node_modules/kanna-code/dist/client/artifacts/learned-intrinsic-training-v2/`.

## 1. Where the code lives

| Role | File | What it does with the three files |
| --- | --- | --- |
| Producer | `experiments/learned_intrinsic_solver/train_mixed.py` | Rank 0 writes `report.json` (atomic) and, via `mixed_report.write_progress`, `progress.json`. On an exception it writes `failure.json` and sets `status = "failed"`. Also calls `write_mixed_report` locally, which is why the run directory contains its own `index.html`, CSVs and SVGs. |
| Renderer | `experiments/learned_intrinsic_solver/mixed_report.py` (`write_mixed_report`, `write_progress`) | Turns a report dict into `index.html`, `epochs.csv`, `updates.csv`, four SVGs, and re-emits `report.json` / `progress.json`. This is the only code that reads metric keys for display. |
| Publisher | `experiments/learned_intrinsic_solver/publish_mixed_report.py` (`prepare_public_report`, `main`) | Loop (default every 45 s, allowed 10–300 s): reads `run/report.json`, `run/progress.json`, `run/failure.json`; augments the report; renders into a temp dir with `write_mixed_report`; mirrors that dir to `<kanna_dist>/artifacts/<slug>/` (default slug `learned-intrinsic-training-v2`) using `copy_source` from `~/.codex/skills/publish-artifact/scripts/publish_artifact.py`. |
| Monitor | `experiments/learned_intrinsic_solver/training_monitor.py` (`check_training_health`, `_dashboard`) | Read-only watchdog. Reads `run/progress.json`, `run/report.json`, `run/failure.json`, `run/checkpoints/latest.pt` (mtime only) and the published `report.json` over HTTP. |
| Served page | `.../artifacts/learned-intrinsic-training-v2/index.html` | Static HTML produced by `write_mixed_report`. It contains **no JavaScript**: it reloads itself with `<meta http-equiv="refresh" content="30">`, embeds three SVGs inline, loads `loss_curve.svg` with `<img>`, and links to `report.json`, `progress.json`, `epochs.csv`, `updates.csv` and the SVGs for download. Nothing in the browser parses JSON or CSV. |

## 2. Files the publisher reads, transforms and copies

Read from the private run directory (never modified):

| File | Required | Use |
| --- | --- | --- |
| `report.json` | optional | If missing, a skeleton report is synthesised (`format = mixed_pool_v2`, `config = {max_epochs: 500, validation_iterations: 100}`, zero counters, empty `epochs`/`updates`, `status = "initializing"` when the tmux session exists, else `"preparing"`). |
| `progress.json` | optional | Embedded verbatim as `report["progress"]` (`{}` when missing). |
| `failure.json` | optional | Embedded as `report["failure"]`; forces `status = "failed"`. |

Written to the staging directory, then mirrored **as a whole directory** by `copy_source`
(`copytree` into `.<slug>.tmp.<pid>`, rename old dir to `.<slug>.old.<pid>`, rename tmp into
place, delete old; the staging dir must contain `index.html`):

| Published file | Origin |
| --- | --- |
| `index.html` | rendered page |
| `report.json` | run `report.json` **plus** publisher-added keys `progress`, `publication`, optionally `failure`, with `status` possibly rewritten and `updated_at` set to the run report's `updated_at` (fallback: mtime of run `report.json`; skeleton case: the string `"Waiting for first epoch metrics"`). All other content is passed through unchanged, including the multi-MB `epochs[].validation.samples[]` arrays. |
| `progress.json` | run `progress.json` re-serialised (`{}` when the run file is missing) |
| `epochs.csv` | derived from `report["epochs"]` (columns in section 5) |
| `updates.csv` | derived from `report["updates"]` |
| `loss_curve.svg` | matplotlib 3x2 epoch panel |
| `validation_curve.svg`, `residual_curve.svg`, `penetration_curve.svg` | hand-built SVG line charts of the latest cheap validation |

Never copied: `checkpoints/`, `logs*/`, `failure.json` as a file, trajectory state.

## 3. `report.json`

Legend for the "Req" column:
**hard** = unguarded access, absence raises; **opt** = read with `.get()`/`_lookup()` and a default or
placeholder; **pass** = not read by any reader, only carried through into the published copy.
"Where" abbreviations: R = renderer (`write_mixed_report`), P = publisher, M = monitor, C = `epochs.csv`.

### 3.1 Top level

| Key path | Type | Unit | Meaning | Req | Where |
| --- | --- | --- | --- | --- | --- |
| `format` | str | – | schema tag, `"mixed_pool_v2"` | pass (written by P skeleton only) | – |
| `status` | str | – | training status; see section 6 for the value set | **hard** in P (`report["status"]` at the interrupted check and in the log line); opt in R (default `"preparing"`), M | R, P, M |
| `updated_at` | str, ISO-8601 UTC | – | time the trainer last wrote the report; shown as "Epoch metrics" | opt (P falls back to file mtime) | R, P |
| `completed_epochs` | int | epochs | count of finished epochs; headline "N / max" | **hard** in P log line; opt in R (default 0), M fallback for `epoch` | R, P, M |
| `completed_updates` | int | Adam updates | total optimizer steps; fallback when `progress.completed_updates` is missing | opt (default 0) | R, P, M |
| `world_size` | int | GPUs | number of ranks; batch text `world_size * batch_size` | opt (default 1) | R |
| `parameter_count` | int | – | model parameters | pass | – |
| `config` | dict | – | full training configuration; dumped verbatim in the "Configuration" details block | opt (default `{}`) | R |
| `epochs` | list[dict] | – | one row per completed epoch; drives every plot and `epochs.csv` | opt (default `[]`) | R, M |
| `updates` | list[dict] | – | rolling window (last `config.updates_history_limit` = 8192) of per-update rows; source of `updates.csv` and `progress.latest_batch_loss` | opt (default `[]`) | R, `write_progress` |
| `best_selection` | dict | – | current best-checkpoint record | opt (default `{}`) | R |
| `best_selection_history` | list[dict] | – | selection resets after budget changes; only the last entry is shown | opt | R |
| `configuration_changes` | list[dict] | – | fields changed at checkpoint resume (`field`, `previous`, `current`, `effective_from_epoch`, `completed_updates`, `source`) | pass | – |
| `initialized_from` | dict | – | weights-only initialisation origin (absent in the example run) | opt (renders nothing when absent) | R |
| `progress` | dict | – | **publisher-added**: copy of `progress.json` (section 4) | opt (default `{}`) | R, M (remote) |
| `publication.updated_at` | str, ISO-8601 UTC | – | **publisher-added**: mirror heartbeat; shown as "Page published" | opt in R (`"Local report"`); required by M dashboard check | R, M |
| `publication.training_session_running` | bool or null | – | **publisher-added**: result of `tmux has-session -t =<training-session>` | pass | – |
| `failure` | dict | – | **publisher-added** (or trainer-written into `failure.json`): `{"error": str, ...}`; rendered as the red "Training failure" block | opt | R, M (`failure.error`) |

### 3.2 `config.*` keys the renderer reads

| Key path | Type | Unit | Meaning | Req / default |
| --- | --- | --- | --- | --- |
| `config.max_epochs` | int | epochs | denominator of "completed epochs"; also copied to `progress.max_epochs` | opt, 500 |
| `config.validation_iterations` | int | iterations | fallback label for the latest cheap validation (K) | opt, 100 |
| `config.validation_interval` | int | epochs | adds "Validation runs every N epochs" if > 1 | opt, 1 |
| `config.validation_count` | int | states | "N fixed validation states" | opt, fallback `validation.sample_count`, then "—" |
| `config.validation_full_iterations` | int | iterations | K cap of the full-horizon check | opt |
| `config.validation_full_interval` | int | epochs | "every N epochs" for the full-horizon check | opt, "—" |
| `config.selection_source` | `"cheap"` or `"full_horizon"` | – | which summary selects the best checkpoint | opt, fallback `best_selection.source`, then `"cheap"` |
| `config.batch_size` | int | trajectories/GPU | batch text | opt (text omitted) |
| `config.damping_range` | [float, float] | Pa·s | absolute viscosity range text | opt |
| `config.regime` | `"pool"` or `"fixed_states"` | – | picks budget/schedule wording | opt |
| `config.state_count` | int | states | fixed-state regime text | opt, "—" |
| `config.budget_cap` | int | K x H | fixed-state budget cap | opt, "—" |
| `config.growth_stages` | list[[int,int]] | (K_max, H_max) | fixed-state timetable text | opt, "—" |
| `config.growth_stage_epochs` | int | epochs | stage advance period | opt, "—" |
| `config.queries_per_epoch` | int | queries | pool-regime budget text | opt, "—" |
| `config.stage_descent_rate` | float | fraction | pool-regime curriculum gate text | opt, "Unavailable" |
| `config.stage_max_epochs` | int | epochs | pool-regime hard cap text | opt, "No hard cap" |
| `config.energy_floor_scale` | float | – | `c` in the loss-formula text | opt, 1.0 |
| `config.energy_increase_weight` | float | – | `λ` in the loss-formula text | opt, 1.0 |

All 74 config keys present in the example are dumped verbatim; the remainder are not otherwise read.

### 3.3 `epochs[]` rows

| Key path | Type | Unit | Meaning | Req | Where |
| --- | --- | --- | --- | --- | --- |
| `epochs[].epoch` | int | – | epoch number, x axis of every panel | **hard** (`row["epoch"]` in `_epoch_plot`) | R, C |
| `epochs[].loss` | float | – (LeCO objective) | mean training objective over queries | opt (gap) | R, C |
| `epochs[].query_count` | int | queries | queries processed in the epoch | opt | C |
| `epochs[].seconds` | float | s | wall time of the epoch; M uses the last three for its checkpoint-staleness threshold | opt (M default 0) | C, M |
| `epochs[].mean_force_residual_n` | float | N | mean free-corner force residual during training | opt | C |
| `epochs[].step_size_mean` / `_min` / `_max` | float | model length units (capped by `config.max_step_size`) | learned step length statistics | opt | C |
| `epochs[].tie_cell_count` | int | cells | tie cells in the epoch | opt | C |
| `epochs[].gradient_norm_mean` / `_max` | float | – | global gradient norm before clipping; twin axis in the LR panel (only drawn if any row has a finite mean) | opt | R, C |
| `epochs[].learning_rate` | float | – | learning rate after the epoch's scheduler step | opt (gap) | R |
| `epochs[].contact_scene_fraction` | float | fraction 0–1 | rank-0 trajectories with a contact plane; latest row only | opt ("not recorded" text) | R, C |
| `epochs[].contact_realized_fraction` | float | fraction 0–1 | trajectories with at least one detected pair; latest row | opt | R, C |
| `epochs[].contact_max_penetration_r` | float | multiples of sample radius r | deepest training penetration; latest row | opt | R, C |
| `epochs[].available_K` | list[int] | iterations | allowed inner-iteration counts; fallback for `progress.available_K` | opt, `[1]` | R |
| `epochs[].available_H` | list[int] | steps | allowed physical-step counts; fallback for `progress.available_H` | opt, `[8]` | R |
| `epochs[].regime` | dict | – | fixed-state regime block (absent in pool regime); fallback for `progress.regime` | opt | R, C |
| `epochs[].regime.name` | str | – | `"fixed_states"` selects the fixed-state wording | opt | R |
| `epochs[].regime.stage` | int | – | growth stage | opt | R, C |
| `epochs[].regime.k_max` / `.h_max` | int | iterations / steps | stage caps | opt | R, C |
| `epochs[].regime.queries` | int | queries | sampled queries this epoch | opt | R |
| `epochs[].regime.filler_queries` | list[int] or int | queries | per-rank fillers; summed if a list | opt | R |
| `epochs[].regime.updates` | int | updates/rank | Adam updates per rank | opt | R, C |
| `epochs[].validation` | dict or null | – | cheap validation summary; `null` = skipped epoch (gap in plots, blank CSV cells). The most recent non-null row feeds the "Latest validation" section. | opt | R, C |
| `epochs[].validation.mean_normalized_loss` | float | – | first-update objective on fixed seeds | opt | R |
| `epochs[].validation.descent_rate` | float | fraction 0–1 | share of validation queries whose energy fell after one update (plotted x100) | opt | R |
| `epochs[].validation.mean_before_joule` / `.mean_after_joule` | float | J | mean physical energy before/after the first update | opt | R |
| `epochs[].validation.selection.metric` | float | N | cheap selection metric (mean final residual) | opt | R, C (`selection_metric`) |
| `epochs[].validation.selection.eligible` | bool | – | all trajectories survived | opt | R, C (`selection_eligible`) |
| `epochs[].validation.physical_survivors` | int | samples | survivors of the physical rollout | opt | R, C |
| `epochs[].validation.sample_count` | int | samples | validation seeds; survivor-percentage denominator | opt | R, C |
| `epochs[].validation.failed_count` | int | samples | all validation failures | opt, "Not evaluated" | R |
| `epochs[].validation.first_update_failed_count` | int | samples | failures at the first update | opt, "Not evaluated" | R |
| `epochs[].validation.failures` | list | – | failure records; dumped in a details block when non-empty | opt | R |
| `epochs[].validation.relative_energy[]` | list[dict] | – | per-iteration curve for `validation_curve.svg` | opt | R |
| `epochs[].validation.relative_energy[].iteration` | int | iterations | x value; last entry labels the table | opt (skipped if non-finite) | R |
| `epochs[].validation.relative_energy[].mean` / `.median` / `.max` | float | ratio Eᵢ/E₀ | aggregated relative energy | opt | R |
| `epochs[].validation.relative_energy[].near_zero_count` | int | samples | last entry only: near-zero initial energies | opt, 0 | R |
| `epochs[].validation.force_residual[]` | list[dict] | – | per-iteration curve for `residual_curve.svg`; the last entry's `iteration` labels the section | opt | R |
| `epochs[].validation.force_residual[].iteration` | int | iterations | x value | opt | R |
| `epochs[].validation.force_residual[].mean` / `.median` / `.max` | float | N | free-corner force residual norm | opt | R |
| `epochs[].validation.penetration[]` | list[dict] | – | per-iteration curve for `penetration_curve.svg` | opt | R, C |
| `epochs[].validation.penetration[].iteration` | int | iterations | x value | opt | R |
| `epochs[].validation.penetration[].mean` / `.max` | float | multiples of r | deepest penetration; last `.max` becomes CSV `validation_final_max_penetration_r` | opt | R, C |
| `epochs[].full_horizon_validation` | dict or null | – | held-out full-horizon check; most recent non-null row is shown | opt ("not evaluated yet") | R, C |
| `epochs[].full_horizon_validation.iterations` | int | iterations (K) | learned iterations per physical step | opt, "—" | R |
| `epochs[].full_horizon_validation.physical_steps` | int | steps (H) | physical horizon | opt, "—" | R |
| `epochs[].full_horizon_validation.physical_survivors` | int | samples | survivors | opt | R |
| `epochs[].full_horizon_validation.sample_count` | int | samples | held-out seeds | opt | R |
| `epochs[].full_horizon_validation.seconds` | float | s | wall time | opt | R |
| `epochs[].full_horizon_validation.final_free_force_residual_norm_n.mean` / `.median` / `.max` | float | N | residual after the last physical step; `.mean` is the square marker in the selection panel | opt | R |
| `epochs[].full_horizon_validation.final_energy_joule.mean` | float | J | final energy | opt | R |
| `epochs[].full_horizon_validation.final_max_penetration_r.mean` / `.max` | float | multiples of r | final deepest penetration | opt | R |
| `epochs[].full_horizon_validation.selection.metric` | float | N | full-horizon selection metric | opt | R, C (`full_horizon_selection_metric`) |
| `epochs[].full_horizon_validation.selection.eligible` | bool | – | eligibility | opt | R, C (`full_horizon_selection_eligible`) |

Present in the example but **pass-through only** (not read by any reader): `epochs[].candidate_modes`,
`rank_0_budgets`, `rank_0_inner_ages`, `rank_0_physical_ages`, `rank_0_material_ranges`,
`rank_0_perturbation_scale`, `rank_0_timings`, `rank_0_pool`, `rank_0_peak_cuda_bytes`,
`rank_diagnostics[]`, `curriculum` (null in the example), `allow_early_stop`,
`validation.samples[]` (64 per epoch, each with 9-point curves and 8 `physical_records`; this is
the bulk of the file size), `validation.optimization_failed_count`, `validation.physical_failed_count`,
`validation.inversion[]`, `validation.final_inverted_sample_count`, `validation.selection.aggregation`,
`validation.selection.survival_required`, `validation.near_zero`, `validation.physical_steps`,
`validation.physical_iterations`, `validation.physical_curves[]`, `validation.final_physical_residual`,
`validation.displacement_rms_mean`, `validation.seconds`, `full_horizon_validation.failed_count`,
`full_horizon_validation.final_physical_residual`, `full_horizon_validation.final_inverted_sample_count`,
`full_horizon_validation.physical_curves[]`, `full_horizon_validation.samples[]`,
`relative_energy[].mean_energy_joule` / `.valid_count` / `.failed_count` / `.relative_sample_count`,
`force_residual[].valid_count` / `.failed_count`, `penetration[].valid_count`.

### 3.4 `best_selection` and `best_selection_history`

| Key path | Type | Unit | Meaning | Req |
| --- | --- | --- | --- | --- |
| `best_selection.metric` | float | N | best selection metric; non-finite → "No eligible epoch has been selected yet." | opt |
| `best_selection.epoch` | int | – | epoch of the best checkpoint | opt |
| `best_selection.source` | str | – | `"cheap"` or `"full_horizon"`; the full-horizon detail text is only added when both this and the configured source are `full_horizon` | opt |
| `best_selection.iterations` | int | K | budget of the selecting full-horizon check | opt, "—" |
| `best_selection.physical_steps` | int | H | budget | opt, "—" |
| `best_selection.final_energy_joule.mean` | float | J | final energy of the best record | opt |
| `best_selection.final_max_penetration_r.mean` / `.max` | float | multiples of r | final penetration of the best record | opt |
| `best_selection_history[-1].reset_at_epoch` | int | – | epoch after which selection restarted (only the last entry is read) | opt, "—" |
| `best_selection_history[-1].reason` | str | – | e.g. `"full-horizon budget changed"` | opt |
| `best_selection_history[-1].record.metric` | float | N | superseded record's metric | opt |
| `best_selection_history[-1].record.epoch` | int | – | superseded record's epoch | opt, "—" |

Other fields (`best_selection.aggregation`, `.completed_updates`, `.sample_count`, `.physical_survivors`,
`.final_energy_joule.median/max`, `.final_max_penetration_r.median`, `best_selection_history[].iterations`,
`.physical_steps`, the rest of `.record`) are pass-through.

### 3.5 `initialized_from` (optional block)

`initialized_from.checkpoint` (str path, last three components shown), `initialized_from.completed_epochs` (int),
`initialized_from.best_selection.metric` (float, N), `initialized_from.best_selection.epoch` (int). All `opt`.

### 3.6 `updates[]` (source of `updates.csv`)

Each row: `update` (int), `epoch` (int), `loss` (float, objective), `before_joule`, `after_joule` (float, J),
`mean_force_residual_n` (float, N), `step_size_mean` / `_min` / `_max` (float), `tie_cell_count` (int),
`gradient_norm` (float), `contact_max_penetration_r` (float, r), `contact_pair_mean` (float, pairs).
Only `updates[-1].loss` is read elsewhere (→ `progress.latest_batch_loss`). All `opt`
(`DictWriter(extrasaction="ignore")`, missing cells left blank).

## 4. `progress.json`

Written atomically by `mixed_report.write_progress` on rank 0 at phase changes and during the epoch
(`train_mixed.py` calls it with `phase` = `initializing`, `training`, `validation`, `complete`, `failed`).

| Key | Type | Unit | Meaning | Req | Where |
| --- | --- | --- | --- | --- | --- |
| `updated_at` | str, ISO-8601 UTC | – | heartbeat time; shown as "Training heartbeat" | opt, "Waiting" | R |
| `status` | str | – | copy of `report.status` at write time (default `"running"`) | opt; M prefers it over `report.status` | M |
| `phase` | str | – | `initializing` / `training` / `validation` / `complete` / `failed`; page headline unless a terminal `report.status` overrides it | opt, falls back to `status` | R, M |
| `epoch` | int | – | epoch in progress (may exceed `completed_epochs` by one); "Epoch in progress:" | opt, "Not started" | R, M |
| `max_epochs` | int | epochs | copy of `config.max_epochs` | pass (R uses `config.max_epochs`) | – |
| `completed_epochs` | int | epochs | copy of `report.completed_epochs` | pass (R uses the report; M uses `epoch`) | – |
| `completed_updates` | int | updates | headline Adam-update count, preferred over `report.completed_updates`; compared local vs published by M | opt | R, P (log), M |
| `latest_batch_loss` | float or null | – | `updates[-1].loss` | pass | – |
| `available_K` | list[int] | iterations | headline "K = [...]" | opt, fallback latest epoch row, then `[1]` | R, M |
| `available_H` | list[int] | steps | headline "H = [...]" | opt, fallback latest epoch row, then `[8]` | R, M |
| `regime` | dict | – | present only in the `fixed_states` regime: `name`, `stage`, `k_max`, `h_max`, `queries`, `filler_queries` (list[int]), `updates`; preferred over `epochs[-1].regime` for the budget line | opt | R |

Both example files (`generated/training_v4_20260928/progress.json` and the served 2026-09-27 copy) contain
exactly these keys; the served copy predates the `regime` block.

## 5. `epochs.csv` columns (in order)

Produced by `write_mixed_report` from `report["epochs"]`; extra row keys are ignored, missing values are
blank. Booleans are written as `True`/`False`.

| # | Column | Source key |
| --- | --- | --- |
| 1 | `epoch` | `epochs[].epoch` |
| 2 | `loss` | `epochs[].loss` |
| 3 | `query_count` | `epochs[].query_count` |
| 4 | `seconds` | `epochs[].seconds` |
| 5 | `mean_force_residual_n` | `epochs[].mean_force_residual_n` |
| 6 | `step_size_mean` | `epochs[].step_size_mean` |
| 7 | `step_size_min` | `epochs[].step_size_min` |
| 8 | `step_size_max` | `epochs[].step_size_max` |
| 9 | `tie_cell_count` | `epochs[].tie_cell_count` |
| 10 | `gradient_norm_mean` | `epochs[].gradient_norm_mean` |
| 11 | `gradient_norm_max` | `epochs[].gradient_norm_max` |
| 12 | `selection_metric` | `epochs[].validation.selection.metric` |
| 13 | `selection_eligible` | `epochs[].validation.selection.eligible` |
| 14 | `physical_survivors` | `epochs[].validation.physical_survivors` |
| 15 | `sample_count` | `epochs[].validation.sample_count` |
| 16 | `contact_scene_fraction` | `epochs[].contact_scene_fraction` |
| 17 | `contact_realized_fraction` | `epochs[].contact_realized_fraction` |
| 18 | `contact_max_penetration_r` | `epochs[].contact_max_penetration_r` |
| 19 | `validation_final_max_penetration_r` | `epochs[].validation.penetration[-1].max` |
| 20 | `regime_stage` | `epochs[].regime.stage` |
| 21 | `regime_k_max` | `epochs[].regime.k_max` |
| 22 | `regime_h_max` | `epochs[].regime.h_max` |
| 23 | `regime_updates` | `epochs[].regime.updates` |
| 24 | `full_horizon_selection_metric` | `epochs[].full_horizon_validation.selection.metric` |
| 25 | `full_horizon_selection_eligible` | `epochs[].full_horizon_validation.selection.eligible` |

Columns 20–25 were appended after the original 19 so positional readers of older tables keep working.
`generated/training_v4_20260928/epochs.csv` has all 25 columns. The served 2026-09-27 copy has only 13
(no `gradient_norm_*`, contact, regime or full-horizon columns): it was rendered by an older
`mixed_report.py` for the LIDO-v2 run, not by the current code. Nothing reads `epochs.csv` back; it is a
download-only artifact.

`updates.csv` columns: `update, epoch, loss, before_joule, after_joule, mean_force_residual_n,
step_size_mean, step_size_min, step_size_max, tie_cell_count, gradient_norm, contact_max_penetration_r,
contact_pair_mean`.

## 6. Running / stopped state and staleness

### Status values

- Trainer (`train_mixed.py`): `"running"` while training; on each epoch
  `report.update(status=decision["status"])` from the schedule controller
  (`training_schedule.py`), whose only values are `"running"`, `"plateau_converged"`, `"stalled"`
  and `"epoch_limit"` (with `config.early_stopping = false`, as in the example run, only
  `epoch_limit` ends a run); `"failed"` on exception (with `failure.json`). `write_progress` copies
  `report.status` into `progress.status`. `early_stopped` is recognised by the renderer but not
  emitted by the current controller; `complete` is a `progress.phase`, never a `status`.
- Publisher adds: `"initializing"` (no `report.json` yet, tmux session exists), `"preparing"` (neither),
  `"failed"` (a `failure.json` exists), `"interrupted"` (see below).
- Renderer: the headline is `progress.phase`, but if `report.status` is one of
  `failed`, `interrupted`, `epoch_limit`, `early_stopped`, `plateau_converged`, `stalled` the status
  string replaces the phase. Any other status leaves the trainer's phase visible.
- Monitor: `status = progress.status` else `report.status` else `"unknown"`; `"paused"` → health
  `paused`; any of `complete`, `completed`, `plateau_converged`, `stalled`, `epoch_limit`, `max_epochs`,
  `finished` → `completed`; `failed`/`interrupted` → an issue; otherwise `healthy` (then `attention`
  if any issue is raised).

### How "running vs stopped" is decided

1. Every publisher tick runs `tmux has-session -t =<training-session>` (default session name
   `learned-intrinsic-v2-damping`, override with `--training-session`). The boolean is recorded as
   `publication.training_session_running` and remembered in `seen_training` (set once the session has
   ever been observed).
2. If `seen_training` is true, the session is now absent, and `report.status` is still
   `running`/`initializing`/`preparing`, the publisher rewrites `status = "interrupted"` and adds
   `failure = {"error": "Training session stopped before reporting completion. ..."}`. This is the
   only automatic "stopped" detection; it is inferred from the tmux session, not from file ages.
3. A `failure.json` in the run directory always wins: `status = "failed"` and its content is shown in
   the red failure block.

### Staleness

The page itself computes nothing; it reloads every 30 s and prints three timestamps in the footer so a
reader can compare them by eye:

- "Training heartbeat: `progress.updated_at`" — last `write_progress` call (trainer liveness).
- "Epoch metrics: `report.updated_at`" — last time the trainer saved `report.json` (advances once per epoch
  plus checkpoint writes; expected to lag the heartbeat by up to an epoch, ~5 min in the example run).
- "Page published: `publication.updated_at`" — last publisher tick (mirror liveness; expected within
  `--interval`, 45 s by default).

`training_monitor.py` is the only component that turns these into alerts (all thresholds in seconds):

- `progress.json` **file mtime** age > `--stale-seconds` (default 2700) → "heartbeat is stale or missing".
- `checkpoints/latest.pt` mtime age > `max(stale_seconds, 3 * max(epochs[-3:].seconds))` → "checkpoint is
  stale or missing".
- Published `report.json` (`--dashboard-url`, default
  `https://ankachen.com/artifacts/learned-intrinsic-training-v2/report.json`): `publication.updated_at`
  missing or older than 180 s → "Dashboard check failed".
- Published `progress.completed_updates` < local `progress.completed_updates` while the local
  `progress.json` mtime is older than 180 s → "Dashboard check failed" (mirror is behind).
- Live worker count/ranks are checked from `/proc`, not from the JSON files.

## 7. Cross-check notes (2026-09-30)

- `generated/training_v4_20260928/report.json` (41.9 MB): 13 top-level keys `format, config, world_size,
  parameter_count, completed_updates, completed_epochs, epochs[25], updates[8192], best_selection,
  status, best_selection_history[5], configuration_changes[3], updated_at`. Every renderer-read path in
  section 3 is present in the epoch rows; `validation` and `full_horizon_validation` are dicts in all 25
  rows (no skipped epochs), `curriculum` is null, `initialized_from` and `failure` are absent, and there
  is no `failure.json`.
- `generated/training_v4_20260928/progress.json`: all 11 keys of section 4 including `regime`
  (`fixed_states`, stage 5, K_max 32, H_max 128).
- Served copy (`.../learned-intrinsic-training-v2/`, dated 2026-09-27): `report.json` (624 MB) carries the
  publisher-added `progress` and `publication` keys (`training_session_running: true`) but was rendered
  for the earlier LIDO-v2 run by an older `mixed_report.py`: no `penetration_curve.svg`, 13-column
  `epochs.csv`, no `regime` in `progress.json`. The published `report.json` is large because
  `epochs[].validation.samples[]` is passed through verbatim.
