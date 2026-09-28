# LIDO v3 implementation record — 2026-09-27

Built autonomously overnight at Anka's request ("implement v3, including the new
edge vector, collision, learning rate and other decisions we made, then start
training"). Architecture stays at ONE transformer block, `hops = (1,)`, hidden 128,
four heads, per-cell step bounded by `max_step_size = 0.05` (Anka: do not try
deeper layouts yet).

## What v3 adds over the schema-3 code (commit 98d4580c)

| Item | Where | Decision source |
|---|---|---|
| A02 state-dependent edge network: `e'_ij = e_ij + MLP([h_i, h_j, e_ij])`, zero-initialized last layer, applied inside the block before `edge_bias`/`edge_val` | `network.py` (`edge_network=True`) | `notes/ablation-list.md` A02 |
| Contact v1: exposed-face samples, ground plane + static points, Newton contact law (quadratic penalty, gap-rate damping, IPC-smoothed friction with constant normal force), detection once per physical step with velocity margin, contact tokens → `ContactEncoder` → 17 channels appended to the node input, three dimensionless conditioning channels | `contact_geometry.py`, `contact_scene.py`, `contact_energy.py`, `contact_features.py`, `contact_network.py`, `features.py` (schema 4), `mixed_physics.py`, `train_mixed.py`, `mixed_validation.py`, `mixed_report.py`, `simulate_mixed.py`, `render_learned.py` | `notes/contact-design-20260927.md` |
| LeCO-style optimizer: AdamW (weight decay 1e-6), global gradient-norm clipping 1.0 (norm logged), cosine 1e-4 → 2.5e-5 over the 500-epoch cap, no early stopping | `train_mixed.py` (commit 98d4580c) | Anka's choice 2026-09-27 |

Parameter count of the campaign network: 432,606 trainable (schema-3 baseline 323,214;
edge update +49,344; contact encoder +57,488; wider node/condition encoders for the
extra inputs). Peak GPU memory at batch 16 per rank: about 10 GB (was 5.2 GB).

## Deviations from the written spec (all deliberate, recorded here)

1. **Conditioning and token contact scalars are dimensionless.** The spec wrote
   `log(ke h / E)`, `log(1 + kd)`; those are not dimensionless. Implemented:
   `log1p(kappa)` with `kappa = ke / (E h)`, `beta = kd / (ke dt)` (0 when `ke = 0`),
   and `mu`. Young's modulus `E = mu (3 lambda + 2 mu) / (lambda + mu)`.
2. **Damping and friction are gated on active penetration** (`d > 0`), matching
   Newton's caller `_eval_body_particle_contact`, not the literal spec formula. This
   makes the energy zero (with zero gradient) whenever no pair penetrates.
3. **`STATE_FEATURE_DIM` stays 61.** The 17 contact channels (16 pooled + count/M)
   are produced inside the network by `ContactEncoder` and concatenated to the node
   input (9 + 61 + 17). The spec's "78" is the same total, packaged differently.
4. **Static points live in a shell around the body** (rest bounding box grown by one
   cell excluded) and the disk is one-sided with thickness `r` (`-r <= gap`).
   The original box put points inside the beam; 95 % of default scenes started with
   25 r penetrations and kJ contact energies. Amendment written into the spec.
5. **Two scene metrics instead of one**: `contact_scene_fraction` (plane or points
   present, about 0.997 under the sampler) and `contact_realized_fraction`
   (trajectories with at least one detected pair in the epoch).
6. **Validation records** deepest penetration (units of `r`) and contact energy per
   iteration and physical step, not yet the mean penetration over active pairs or
   the contact share of the residual (spec §8, deferred).
7. **Renderer switched to a Y-up viewer** with the camera above the floor on the +x
   side; the floor is a 0.75-opacity quad, static points are spheres with a normal tick.

Provisional ranges (spec §7) implemented unchanged: plane probability 0.8, plane
height `[-0.35, -0.02]` m below the rest y-minimum, 0–64 points, `r_p` in
`[0.5 h, 2 h]`, `kappa` log-uniform `[0.1, 10]`, `beta` uniform `[0, 1]`, `mu`
uniform `[0, 1]`, `M_pair = 4`, `M_cell = 24`, `friction_epsilon = 1e-2`.

Second amendment (2026-09-27, after a read-only diagnostic of the running
campaign; spec §3.3, §4 and §7 amended in place):

- **Why.** The deepest pairs at the initial candidate were 100 % static-point
  pairs with grazing normals (median `|cos|` 0.29), capped at `2 r` by the
  one-sided rule; the one-cell clearance kept the *point* outside the body but
  the disk *plane* still cut the body in 43 % of default scenes, 21 of 220
  penetrating validation seeds had their deepest pair on the fully clamped -z
  face, and L-BFGS on the true objective left the same penetration (data, not
  solver). The floor never penetrated in 512 validation seeds.
- **Sampler rejection loop** (`contact_scene.sample_contact_partners`,
  `_draw_static_points`): points come from a spawned child of the scene seed so
  `kappa`, `beta`, `mu`, the plane flag, the plane height and the count keep
  their draws; a point is redrawn while its disk would be a detection candidate
  of any rest surface sample in the band `-r <= gap < r + h` (`r = 0.5 h`,
  shared helper `_point_candidates` with `detect_contacts`), while its normal
  does not oppose the nearest rest face within 60° (`POINT_NORMAL_MAX_ANGLE`;
  eight normal draws, then a new position), and the box starts at
  `z = z_min + h` behind the clamped face; 1000 position draws raise
  `ValueError`. Realized (200 canonical seeds): scenes with a rest candidate
  61 % → 0 %, mean points 32.3 unchanged, 35 % of position draws rejected,
  11 ms per scene.
- **Normal filter in detection** (`detect_contacts(..., sample_normals=)`):
  plane and point candidates require `n_partner . n_face < 0`;
  `MixedHexSolverStep.prepare` passes `sample_normals` of the step-start
  positions. Consequence visible in the tests: a floor within the band of a
  cell's side faces now pairs with its bottom face only (4 instead of 12 pairs
  on the (2, 1, 2) test grid).
- **Plane height default** `MixedTrainConfig.contact_plane_height_range =
  (-0.15, -0.005)` m and the same in `generated/training_v3_contact_config.json`
  and `generated/smoke_v3_contact_config.json`; the sampler default keeps
  `(-0.35, -0.02)`.
- **Tests.** `test_contact_scene`: 200 canonical seeds with zero rest
  candidates in the `r + h` band, cone and `z >= h` checks, and the coefficient
  stream reproduced from the documented `SeedSequence` order; the superseded
  "shallow touches expected" rest-shape test is removed; `sample_normals`
  drops side and top faces for the plane and a side face for a point while
  keeping the bottom face, and filtered candidates free their `M_pair` slots;
  placement failure raises. `test_mixed_physics`: `prepare()` keeps the four
  bottom faces of a floor 0.03 m below the grid and drops the eight side faces,
  a disk beside the +x face pointing away yields no pairs and facing the body
  yields exactly the two +x faces. `test_train_mixed`: new default.

## Validation

- Unit tests: 550 pass on CPU (`python -m unittest discover -s experiments/learned_intrinsic_solver/tests -t .`),
  including a Warp oracle test of the Torch contact gradient against Newton's
  `_compute_body_particle_contact_force`, finite-difference checks, detector and
  scene-range tests, zero-init identity tests for both new network paths, a
  two-epoch CPU training smoke with a floor, and resume with the scene intact.
- One-epoch GPU smoke run on four L40s: see the campaign section below.

## Campaign

- Run directory `generated/training_v3_contact_20260927`, config
  `generated/training_v3_contact_config.json`, start script
  `generated/start_training_v3_contact.sh`, resume script
  `generated/resume_training_v3_contact.sh <checkpoint>`.
- tmux `LIDO-v3` (training) and `LIDO-v3-web` (publisher); dashboard slug
  `learned-intrinsic-training-v3`.

## Implementation record — normalised physics, schema 5 (2026-09-28)

Landed uncommitted in worktree `learned-instrinic-solver` on top of `14dc83c1` (the fixed-state regime),
together with the trainer changes recorded in `notes/epoch-regime-20260927.md`.

- **Units.** `MixedHexSolverStep` evaluates the objective, the fusion and the network inputs in cell
  units (`h = 1`, `mu = 1`, `dt = 1`) behind an unchanged SI API: material `lambda' = lambda/mu`,
  `rho' = rho h^2/(mu dt^2)`, `eta' = eta/(mu dt)`, contact `ke' = ke/(mu h)`, `kd' = kd/(mu h dt)`,
  `friction_epsilon' = friction_epsilon dt/h`; energies scale back by `mu h^3` (total rebuilt as the exact
  float32 sum of the parts), the force residual and `position_gradient` by `mu h^2` (newtons), fused
  positions by `h` with prescribed rows copied bit-exactly. Measured against a float64 SI reference:
  elastic 2.1e-6, inertia 4.8e-7, damping 1.6e-6, contact 2.5e-7, total 3.5e-7 relative; two scenes
  related by the law give identical inputs to float32 rounding, energy x8 and residual x4 exactly.
- **Conditioning (schema 5).** `features.CONDITIONING_CHANNELS = (log1p_lame_ratio, log_inertia_ratio,
  log1p_gravity_ratio, log1p_viscosity_ratio, log1p_contact_kappa, contact_beta, contact_mu)`,
  `CONDITIONING_DIM = 7`; `conditioning_channels(..., gravity)` takes the SI gravity (magnitude or
  vector, only `|g|` enters). The SI reference constants of schema 4 are gone. `log_gradient_rms` and
  the RMS-normalised gradient blocks, `axis_gradient_world` and the optimizer history are expressed in
  `mu h^3`; `LearnedHexSolverStep` (deployment path) divides its SI projected gradient by
  `energy_unit = mean(lame_mu) h^3` so both steps feed a network identical inputs (test_solver_step
  cross-check compares every state column; `test_normalised_physics` checks the single step at `h` and
  `2h`). `FEATURE_SCHEMA_VERSION = 5`; schema-4 checkpoints, pools and their joule histories are rejected
  for a full resume (`MixedTrainConfig.from_checkpoint_config`, both steps' schema check, which names
  61/9 as legacy).
- **Shared factor.** One `HexFusion` on the unit grid with unit weights is built in the constructor and
  shared by every context (`register_context` builds no factor: 17.7 -> 5.7 ms per registration on
  (4,4,40), the rest is the rigid predictor). The unit lattice is generated exactly
  (`generate_cuboid(cell_counts, 1.0, origin/h)`), so far origins pass the canonical check.
- **Partial load (run 3 start).** `train_mixed.migrate_weights_only_checkpoint` implements the
  weights-only start from run 2's schema-4 `best_validation.pt`: every state-dict tensor whose name and
  shape match is copied (58 tensors, 647,326 elements incl. the two neighbour buffers);
  `condition_encoder.0.weight` ([128, 9] -> [128, 7]) is the only shape mismatch and the layer is redrawn
  with weight std `CONDITION_ENCODER_REINIT_STD = 0.02` and zero bias (both tensors recorded under
  `report["initialized_from"]["reinitialized_parameters"]`, with `source_feature_schema_version = 4`,
  `shape_mismatches` and the copied counts); the AdamW state loads for all 58 parameters with the moments
  of the two re-initialised tensors zeroed at the new shape and their step counters (7680) kept. Any
  other shape mismatch, and any other schema pair, keeps the strict "requires the checkpoint
  architecture" rejection; a full resume across schemas is rejected as legacy. Why the small scale: the
  loaded FiLM layers are trained (no longer zero-initialised), so a default first layer (std about 0.22)
  would drive them with random codes; with std 0.02 the code starts near the encoder's second-layer
  bias and the dependence on the new channels is relearned. Consequence of keeping the step counters:
  Adam's bias correction is inactive for the two fresh tensors, so their first updates are up to about
  6 x lr (the standard warm-start behaviour) until their moments fill in.
- **Offline dry check (CPU, no training, 2026-09-28)** with `generated/training_v3_fixed_config.json`
  (schema 5, `updates_history_limit` 8192): architecture fields all match; config differences are
  max_epochs 500 -> 60, regime pool -> fixed_states, validation_count 512 -> 64, validation_iterations
  100 -> 32, validation_full_count 16 -> 8, validation_full_iterations None -> 8, feature_schema_version
  4 -> 5; re-initialised `['condition_encoder.0.bias', 'condition_encoder.0.weight']`, 58 tensors copied
  bit-identically, redrawn weight std 0.0197; strict `load_state_dict` and `AdamW.load_state_dict` succeed
  and one in-memory step with synthetic gradients runs (a [128, 9] moment would have failed there).
  Parameter count 432,350 (run 2: 432,606; the 256 lost weights are the two removed input columns).
- **Consumers.** `mixed_validation` (SI residual via `step.energy`), `simulate_mixed`,
  `mixed_training_reference` (7-wide network from `config.conditioning_dim`; schema-4 artefacts are
  rejected explicitly), `evaluate_rollout`/`replay_rollout`/`unrolled_solver` (LearnedHexSolverStep, history
  in the step's unit) needed no code change; `history.store_history` documents the unit. Run-2 artefacts
  are replayed with a checkout of `14dc83c1` or earlier.
- **Tests.** Whole suite on CPU after the change: see the epoch-regime note for the final `Ran` line.
  The nine schema-4 pins (`test_train_mixed`, `test_damping_training`, `test_damping_solver`,
  `test_mixed_damping`) now address channels by `features.CONDITIONING_CHANNELS.index(...)`, convert the
  joule reference by `mu h^3` where the gradient feature is compared, and compare the float32 mixed energy
  with a float64 SI reference at rtol 1e-5 (measured 3e-6).
