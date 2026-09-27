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
