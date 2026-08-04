# Current Implementation of the Principal-Stretch Network (as of 2026-08-04)

Everything lives in `research/principal_stretch/`. This documents what the
code *actually does today* on branch `ankac/principal-stretch-dev`, so it can
be reviewed before we build anything new. Line references are to this branch.

## 1. Big picture

The method splits one dynamics step into a **learned local prediction** and a
**deterministic global reconstruction**:

```
                 x_{t-1}, x_t, f_ext                    (world-frame state)
                          |
              [compute_S_from_x]  torch_solver.py:150
                          |
        S_{t-1}, S_t   per-tet symmetric stretch (rotation removed)
                          |
              [build_features]    model.py:117          28 floats per tet
                          |
              [StretchNet MLP]    model.py:79           shared weights, per tet
                          |
        S*  = predicted target stretch for t+dt         (6 DOF per tet)
                          |
              [solve = local-global decoder]  torch_solver.py:185
              warm-started at  2 x_t - x_{t-1}
                          |
                       x_{t+dt}
```

- The network never sees or emits positions; it maps *stretch state* to
  *stretch prediction*, per tet, with shared weights.
- All rotation handling is delegated to the decoder. This is the point of the
  representation: S is invariant to rigid motion, so the network's job is
  SE(3)-invariant by construction.
- The decoder is differentiable end-to-end (exact gradients, see §3), so
  training supervises **decoded positions**, not S directly. §8 shows why
  that matters more than we originally understood.

## 2. The representation

For each tet, `F = Ds @ Dm_inv` (deformation gradient), polar-decomposed as
`F = R S` with `R in SO(3)` and `S` symmetric positive-definite. The network
predicts the 6 independent components of a target `S*` per tet.

- `compute_S_from_x` (`torch_solver.py:150`) produces `S = sym(R^T F)` from
  positions. Batched over a leading dim.
- `S` is fed to the network *centred at identity* (`S - I`), so "rest" is 0.

## 3. The decoder (`torch_solver.py`, `polar.py`)

An ARAP-style alternating minimisation of

```
E(x, R) = sum_e  w_e/2 * || F_e(x) - R_e S*_e ||_F^2      w_e = rest volume
```

subject to pinned vertices held at their targets. Given target stretches S*
it recovers positions:

- **Local step** (`solve`, `torch_solver.py:227`): per tet,
  `R = polar(F S*^T)` — the optimal rotation for fixed x.
- **Global step** (`torch_solver.py:232`): positions solve the linear system
  `L x = rhs(R S*)`, where `L = sum_e w_e J_e J_e^T` is the **rest-mesh
  Laplacian** — constant, so it is partitioned into free/pinned blocks and
  Cholesky-factored **once** at `build_solver` (`torch_solver.py:92-132`).
  Every iteration afterwards is one pre-factored triangular solve.

Properties that shape everything downstream:

- **Convergence is linear at ~0.98/iteration** (amplitude, not reach). 500
  iterations to converge fully; at any practical budget (6-20 iters) the
  output is dominated by the warm start. This was the single largest error
  source in the May Phase-2 experiments (90% of it).
- **The warm start is therefore part of the method**:
  `inertial_predictor` (`torch_solver.py:168`) = `2 x_t - x_{t-1}` with pins
  restored. Measured 10x on the decoder floor vs warm-starting at `x_t`.
- **One global solve propagates information across the whole mesh** — the
  system is elliptic, like an implicit-integration step. The 0.98/iter rate
  is about error *amplitude*, not about how far information travels. What the
  global step cannot do is change S*: it least-squares-fits positions to
  whatever per-tet targets the network emitted (see notes/01).
- `solve` is **batched**: a leading batch dim shares the single Cholesky
  factor, so a whole training batch (and all rollout steps of the eval set)
  decode in one triangular solve per iteration. Decoder runs fp64; the
  network runs fp32.

### Polar decomposition with exact gradients (`polar.py`)

`polar_rotation` is a custom autograd Function:

- Forward (`polar.py:82`): Higham's scaled Newton iteration
  `R <- (gamma R + gamma^-1 R^-T)/2`, 5-6 iterations to fp64 round-off even
  at 17:1 anisotropy; SVD fallback when `det(M) <= 0`.
- Backward (`polar.py:130`): exact, via the Sylvester identity. With
  `A = skew(R^T grad_R)`, solve `(tr(S) I - S) b = axial(A)`, then
  `grad_M = 2 R [b]_x` (up to symmetrisation) — closed-form 3x3 solve, no
  iteration in the backward pass.
- The previous implementation routed gradients through an SVD surrogate that
  is exact only at `S = I`; it reached 22-27% Jacobian error at 50% stretch
  (measured in `diag_polar_grad.py`). Unit tests incl. `gradcheck` and
  finite-difference Jacobians: `tests/test_polar.py` (8 tests).

## 4. The network (`model.py`)

`StretchNet` (`model.py:79`) is deliberately minimal — a **3-layer MLP
(28 -> 64 -> 64 -> 6, SiLU)** applied independently to every tet with shared
weights. There is **no message passing**; the only non-local input is a mean
over face-adjacent neighbours (below).

Input features, 28 per tet (`build_features`, `model.py:117`):

| group | dims | content |
|---|---|---|
| S(t) - I | 6 | current stretch, identity-centred |
| S(t-dt) - I | 6 | previous stretch (velocity in S-space) |
| gravity / 10 | 3 | constant per dataset |
| f_ext_tet / 30 | 3 | mean external force over the tet's 4 vertices |
| mu, lam / 1e5 | 2 | Lame parameters (uniform on current articles) |
| pin_flag | 1 | 1 if any of the tet's vertices is pinned |
| mean_neighbour(S) - I | 6 | mean S over face-adjacent tets (<= 4) |
| n_neighbours / 4 | 1 | boundary indicator |

Output: 6 numbers, `delta = 0.6 * tanh(raw)`, assembled into a symmetric
matrix. Two parameterisations:

- absolute: `S* = I + delta` (the network regresses the whole deformation);
- **residual** (`--residual`, what all current checkpoints use):
  `S* = S_t + delta` — with the zero-initialised last layer, training starts
  from "stretch unchanged", the correct prior for one dynamics step.

**Receptive field**: the features give each tet its own state + a 1-ring
mean. Everything farther away is invisible to the predictor — this is the
issue analysed in notes/01. (The `--blocks` option in §5 widens the
*composite* receptive field; the per-block network stays local.)

## 5. Training (`train.py`)

Self-supervised on recorded VBD trajectories (§6). One training step:

1. Sample a batch of windows `(x_{t-1}, x_t)` with `K` ground-truth frames of
   room (`build_windows`, `train.py:48`).
2. Optionally perturb the *input* states with Gaussian noise
   (`--noise-std`, `train.py:166`) — targets stay clean, S is recomputed
   from the noisy positions. This is the MGN trick: training visits the
   off-manifold states that autoregressive rollout inevitably produces, and
   the model learns to contract them.
3. Roll out `K` steps *through the differentiable decoder*
   (`train.py:184-247`): predict S*, decode `x_{t+1}`, recompute S from the
   decoded positions, feed forward. Gradients flow through the whole chain.
4. Loss per step (`--loss pos`): mass-weighted squared position error against
   the reference frame, optionally + `--phys-weight` x the backward-Euler
   incremental potential (`potentials.py:19`) as an off-manifold regulariser.
   (`--loss phys` alone = pure self-supervised; kept as an ablation. Note its
   physics is one BE step at dt=1/60 while the data is 10 VBD substeps at
   dt/600 — a known objective mismatch, review §2.3.)
5. `K` follows a **curriculum** (`--curriculum-frac`): linear 1 -> `--max-rollout`
   over the first half of training, then flat. Long-K training is what makes
   rollouts stable (§8).

`--blocks N` (`train.py:199-207`): N alternating network->decoder passes per
step, decoder iterations split evenly, S recomputed from the decoded
positions between passes. Each pass's global solve hands the network a
mesh-wide view of the previous pass's prediction — the PoissonNet pattern.

Speed: batched trainer does ~0.22 s/step at K=4 on the toy article (was ~2 s
in the per-sample loop it replaced).

## 6. Data generation (`gen_train_data.py`)

- Article: hanging soft grid (`add_soft_grid`), one face pinned. Toy =
  8x4x4 cells (225 verts / 640 tets / 0.8 m); the 4k article = 24x12x12
  (4225 verts / 17280 tets / 2.4 m) via `--dim-x/y/z`.
- Reference simulator: **SolverVBD, 10 substeps x 10 iterations @ 60 fps**,
  StVK material, no self-contact.
- Per trajectory: random constant body force (uniform over the mesh,
  magnitude 0-25 N) **plus a point poke** — one random unpinned vertex gets
  up to 50 N for the first 2-10 frames. The poke is the localized
  disturbance; the body force is visible to every tet through its features,
  the poke is not (notes/01).
- Recorded per frame: `x`, `f_ext`, `F`, `S`, plus topology/material/pins
  once. Toy: 200 train + 30 val trajectories x 20 frames.
  Files: `data/{train,val}.npz` (toy, in the reference worktree),
  `data/{train,val}_4k.npz` (4k, here, gitignored — 3.9 GB).

## 7. Evaluation harnesses

- `eval_singlestep.py` — teacher-forced: feed GT `(x_{t-1}, x_t)`, predict
  one step, per-vertex error. Isolates model quality from rollout drift.
- `rollout.py` — autoregressive from the first two frames of each val
  trajectory, replaying the recorded `f_ext` schedule; reports
  mean-over-trajectory and final-frame error. Reads the checkpoint's own
  config (residual/warm/blocks) so eval always matches training.
- `bench_pareto.py` — the decisive instrument. Re-simulates the val force
  schedules with VBD at a (substeps x iterations) grid **and** rolls out the
  net at several decoder budgets; reports (ms/frame, rollout error) pairs.
  Sanity anchor: VBD at the data-generation budget (10x10) replays to
  exactly 0 error. Timing is wall-clock per frame on the same GPU.
- `diag_*.py` — the takeover-review diagnostics (decoder convergence, polar
  gradient error, floor decomposition, trivial baselines) plus
  `diag_knn_floor.py` (new, §8).

## 8. Empirical state (what we know, with pointers)

Full reports live in
`/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/PrincipalStrecchSolver/`.

1. **May's Phase-2 error was 90% decoder truncation**, not model error
   (takeover review, 2026-07-28). Fixed by warm start + batching + exact
   polar gradients + residual parameterisation + position loss.
2. **Toy single-step is now 2.1 mm** (was 11.7 mm), vs a 1.0 mm decoder
   floor with oracle S (round 1 doc).
3. **Single-step accuracy does not determine rollout error.** 2.1 mm and
   13 mm single-step models both land at ~0.065 m over 18 frames. Rollout is
   stability-bound; the winning recipe is input noise 3 mm + potential
   regulariser 1e-4 + K=8 curriculum -> 0.027 m mean / 0.046 m final, error
   growth saturating instead of compounding (round 2 doc).
4. **Phase B (beat VBD on wall-clock) failed its pre-registered kill
   criteria on the 4k article** (pareto4k doc): the net's best point
   (0.074 m @ 100 ms) is matched by VBD's cheapest (0.0785 m @ 1.14 ms).
   Error is model-bound (4 -> 20 decoder iters: 0.089 -> 0.074), not
   decoder-bound. VBD's accuracy-per-ms scales better with mesh size.
5. **`--blocks 2` improves both single-step and rollout ~20% at matched
   decoder budget** (7.47 -> 5.97 mm; 0.0336 -> 0.0264 m). blocks=3 keeps
   improving single-step but regresses rollout. Real mechanism, ~1.2x-class
   lever (day report §4).
6. **kNN conditional-variance test** (`diag_knn_floor.py`, 2026-08-04, raw
   tables in the appendix below): on the toy article the accuracy-trained
   checkpoint decodes at 2.12 mm — **2x better than the kNN estimate of the
   feature-information floor (4.2 mm at k=5)**, and 3.4x better than
   persistence (7.1 mm). On the **4k article** the picture changes: the only
   existing checkpoint (the stability recipe) lands **exactly on the kNN
   floor estimate** (8.17 vs 8.42 mm decoded; persistence 21.5 mm, oracle
   1.53 mm). Consequences: (a) kNN is not a tight floor estimator — on the
   toy the net extracts 2x more than kNN does, so kNN alone cannot *prove*
   feature exhaustion; (b) the accuracy-trained 4k control (pos_4k,
   recipe-matched to the toy's) resolves the question **gradedly**: it
   reaches 5.73 mm — still 1.45x below the kNN estimate, so the features
   are not exhausted at 4k either, but the margin over kNN shrank (2x ->
   1.45x) and the gap over the decoder oracle grew (2.1x -> 3.7x) from toy
   to 4k. The information ceiling is not a wall we have hit; it is a wall
   that closes in with mesh diameter, exactly the trend notes/01 predicts;
   (c) S-space
   Frobenius error is a **misleading metric**: the stability nets are
   *worse* than persistence in S-norm while decoding 2.6-3.4x better, and
   decoded kNN error *worsens* with k (8.4 -> 11.9 mm, k=1 -> 20) while
   S-space error improves — the decoder weighs S directions very unevenly,
   and training through the decoder exploits exactly that. Any future
   scheme that supervises S directly with an L2 loss is set up to fail.
7. **Same-tet kNN adds nothing** (toy): giving the predictor the tet's
   identity (equivalent to arbitrary positional features: distance to the
   pinned end, etc.) does not improve on the global pool. The unpredictable
   part of the next stretch is not explained by *where* the tet is — it is
   explained by far-field *state* the features do not carry.

## 9. Known limitations

- **Local receptive field / ill-posed per-tet prediction** — notes/01. The
  poke in the data is invisible to distant tets until the stretch wave
  reaches their 1-ring; with blocks=1 the predictor cannot react earlier in
  principle. How much average error this costs at which mesh scale is
  exactly what the kNN test + a hierarchy prototype are meant to settle.
- **No contact, no self-collision, single article topology, uniform
  material.** All experiments so far are one hanging cuboid family.
- **Decoder cost at scale**: fp64 dense Cholesky is fine at 225-4225 verts
  and latency-bound on GPU (fp32 no faster at toy size), but dense
  factorisation stops scaling long before production meshes; a sparse/Warp
  port is unwritten.
- **Rollout stability is empirical**, bought with noise + regulariser +
  curriculum; there is no stability guarantee, and single-step gains
  routinely fail to transfer (blocks=3).
- Gravity is constant across the current datasets, so its 3 feature dims
  are dead inputs on these articles.

## 10. File map

| file | role |
|---|---|
| `torch_solver.py` | decoder: `build_solver` (Laplacian + Cholesky), `solve` (local-global), `compute_S_from_x`, `inertial_predictor` |
| `polar.py` | batched 3x3 polar rotation, Newton forward / exact Sylvester backward |
| `model.py` | `StretchNet` MLP, `build_features`, `build_face_adjacency` |
| `potentials.py` | StVK energy + batched backward-Euler incremental potential |
| `train.py` | batched trainer: pos/phys loss, noise, curriculum, blocks |
| `gen_train_data.py` | VBD trajectory generator (body force + poke) |
| `eval_singlestep.py` / `rollout.py` | teacher-forced / autoregressive eval |
| `bench_pareto.py` | accuracy-vs-wall-clock, VBD grid vs net |
| `diag_knn_floor.py` | kNN conditional-variance test of the feature set |
| `diag_polar_grad.py`, `diag_decoder_conv.py`, `diag_floors.py`, `diag_baselines.py` | takeover-review diagnostics |
| `tests/test_polar.py` | polar unit tests (gradcheck, FD Jacobians, anisotropy) |
| `recover_local_global.py`, `run_*.py`, `recorder.py`, `kernels.py` | Phase-1 machinery (stretch recording, recovery demos) |

## 11. How to run

```bash
cd <worktree>  # this repo
source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh <name>

# data (toy; use --dim-x 24 --dim-y 12 --dim-z 12 for the 4k article)
uv run python -m research.principal_stretch.gen_train_data --out data/train.npz
uv run python -m research.principal_stretch.gen_train_data --out data/val.npz --num-trajs 30 --seed 1

# train the current best recipe (toy combo)
uv run python -m research.principal_stretch.train --train data/train.npz \
  --out checkpoints/combo.pt --loss pos --residual --warm inertial \
  --noise-std 3e-3 --phys-weight 1e-4 --max-rollout 8 --steps 4000

# evaluate
uv run python -m research.principal_stretch.eval_singlestep --ckpt checkpoints/combo.pt --data data/val.npz
uv run python -m research.principal_stretch.rollout --ckpt checkpoints/combo.pt --data data/val.npz
uv run python -m research.principal_stretch.bench_pareto --data data/val.npz --ckpt checkpoints/combo.pt

# feature-information floor
uv run python -m research.principal_stretch.diag_knn_floor \
  --data-train data/train.npz --data-val data/val.npz --ckpt checkpoints/combo.pt
```

## Appendix: kNN conditional-variance raw results (2026-08-04)

Decoded single-step position error, per-vertex mean (solver iters 10,
inertial warm — the eval_singlestep protocol). "kNN" predicts each tet's
next stretch as the mean of its k nearest cross-trajectory training samples
in z-scored 28-dim feature space; it upper-bounds what any per-tet function
of these features can achieve, but is not tight (see toy: net beats it 2x).

| predictor | toy (225 v) | 4k (4225 v) |
|---|---|---|
| oracle S_gt (decoder floor) | 1.01 mm | 1.53 mm |
| persistence S* = S_t | 7.14 mm | 21.5 mm |
| net, accuracy recipe (pos-only) | **2.12 mm** | **5.73 mm** |
| net, stability recipe (combo) | 8.06 mm | **8.17 mm** |
| kNN global k=1 | 4.34 mm | 8.42 mm |
| kNN global k=5 | 4.17 mm | 9.71 mm |
| kNN global k=20 | 4.75 mm | 11.9 mm |
| kNN same-tet k=5 | 5.14 mm | (skipped) |

S-space Frobenius error `||S_pred - S_gt||`, mean per tet — note the
inversions vs the decoded table (net worse than persistence here, k=20
best here but worst decoded):

| predictor | toy | 4k |
|---|---|---|
| persistence | 1.19e-2 | 1.11e-2 |
| net (accuracy / stability) | 1.73e-2 / 7.21e-2 | 1.22e-2 / 1.77e-2 |
| kNN global k=1 / 5 / 20 | 1.18 / 1.03 / 1.11e-2 | 1.06 / 0.82 / 0.82e-2 |

Match quality: median 1-NN z-distance per dim 0.081 (toy) / 0.087 (4k) —
comparable, so the toy-vs-4k comparison is fair. Pools: 2.3M samples (toy,
full train set) / 4M subsampled of 62M (4k). Queries: full val (toy) /
180 frames = 3.1M samples (4k).
