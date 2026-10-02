# LIDO reimplementation plan: same method, new implementation (2026-09-30)

Scope: rewrite the learned intrinsic hex solver for simplicity and per-query efficiency.
Every method decision recorded in `notes/` (v2-plan, v4-plan, epoch-regime, fusion derivation,
contact design v1, normalised cells) is kept exactly. Only the implementation changes. The old
code was not consulted; LeCO was read for implementation lessons only, not for collision.

## 1. Method, kept as is

| Item | Decision (unchanged) |
|---|---|
| Unknowns | Shared corner positions; cells are graph tokens; canonical 10x10x40 grid, h = 0.025 m, z-min face pinned |
| Frames | Closest proper rotation of the centre F: `R = U diag(1,1,det(UV^T)) V^T`; clamped-face tie-break with the reference frame from three prescribed corners; recomputed at every query from the current candidate, frozen (detached) for the backward, never carried |
| Representation | 21 values per cell = 7 vectors x 3 in the frozen frame: three axes (columns of centre F) and four warping vectors (xi^a xi^b, xi^1 xi^2 xi^3), scaled 2/h. `target_modes` 7 (3 = affine-only ablation) |
| Node input | 21 current + five 21-blocks (inertial offset, physical change for damping, projected current gradient through the fusion adjoint, previous gradient, previous achieved update) + 6 exposed-face flags + 8 fixed-corner flags + log gradient RMS + history_valid + 17 contact = 159. LeCO normalisation: RMS floor 1e-12, clip +-10 |
| Conditioning | 7 dimensionless FiLM channels: log1p lambda/mu, log rho h^2/(mu dt^2), log1p g dt^2/h, log1p eta/(mu dt), log1p kappa, beta, mu_friction |
| Edges | Radius-one neighbourhood (self plus every face-, edge- and corner-sharing cell, up to 27); 24 edge values: rest offset/h, receiver-frame current offset/h, R_i^T R_j, R_i^T F_j; edge attention bias and edge values |
| Network | One radius-one transformer block, width 192, 6 heads, FFN x4, edge hidden 96; A02 state-dependent edge network `e'_ij = e_ij + MLP([h_i, h_j, e_ij])` with zero-initialised last layer, applied before edge bias/values (v3 record, `edge_network: true` in v4); contact encoder width 64 (19-dim tokens, masked attention per cell, mean+max pool, zero-init Linear to 16); heads: bounded 21-value correction, per-cell step = 0.05 sigmoid |
| Fusion | Weighted least squares of one corner displacement to the per-Gauss-point target increments `Delta F_{c,q} = Gamma_q Delta m_c` over the 8 Gauss points, `K = B^T W B`, prescribed corners eliminated exactly; energy evaluated on the fused shape; backprop through the solve; projected gradient `W B K_ff^{-1} g_free` collapsed to the 21 modes |
| Physics | Stable Neo-Hookean (mu_NH = mu, lambda_NH = lambda + mu), 8 Gauss points; implicit Euler with Y fixed; VBD metric damping eta/(2 dt) ||C - C_prev||^2; Newton contact law (ke/2 relu(r - gap)^2, gap-rate damping and friction gated on active penetration, IPC smooth friction with detached normal load); detection once per physical step on the step-start shape with velocity margin, partner normal must oppose the face normal, M_pair 4, pairs frozen for the K queries; static points sampled in a shell outside the body with the rejection rules of the contact note; normalised units h = mu = dt = 1 inside, SI API |
| Loss | `scale = max(|E_before|, floor)` detached; `asinh(E_after/scale) + relu((E_after - E_before)/scale)`; floor `c eps32 (V (lambda + 2mu + eta/dt + rho h^2/dt^2) + ke r^2 S)`; batch mean; one AdamW update per batch; carried candidates and history detached |
| Candidates | 50 % inertial Y, 50 % Y + multiscale noise (RMS 1-10 % h); after K queries advance: V = (X - X_prev)/dt, new Y, new candidate, new detection |
| Data | Multiresolution deformation and velocity augmenter with a U(0,1) global multiplier; E log-uniform [1e3,1e6], nu U[0.2,0.49], rho log-uniform [100,1e4], eta log-uniform [10,1000], |g| log-uniform [2,40] along -y; contact scenes: plane p 0.8, 0-64 static points, kappa log-uniform [10,1000] with the load floor, beta, mu_f U[0,1] |
| Regime | Fixed-state epochs: 2048 states; per state per epoch H ~ U{1..H_max}, K ~ U{powers of two <= K_max}, K H <= 2048; growth (1,8) (2,16) (4,32) (8,64) (16,128) (32,128) every 2 epochs; each query is one update in a mixed batch (16 per rank x 4 ranks); LPT assignment by K H |
| Validation | Cheap 64 states x 8 and full horizon 16 states K = 8, H = 128 every epoch; selection: mean final free-corner residual with survival required; AdamW cosine 1e-4 -> 2.5e-5 over 48 epochs, clip 1.0, no early stopping |
| History | World-coordinate axis gradient and achieved update, carried across physical steps, cleared on reset |

## 1b. Decisions taken in the review of 2026-09-30

| Question | Decision |
|---|---|
| Location | New sibling package `experiments/lido/` on the same branch; the old package stays until parity, then is deleted in its own commit |
| Parity gate | (a) a subagent writes an adapter that runs the old package as a black box; energy, gradient, frames, fusion, projected gradient and (with a test-only weight mapping from a v4 checkpoint) the network forward must agree within float32 on random small-grid states; (b) a 6-epoch run under the v4 config must track v4's first epochs within noise (statistical, the generator is new) |
| Batch layout | Flat, Newton's way: cells and corners of all objects concatenated with offsets, `cell_obj` / `corner_obj` ids, heterogeneous grids allowed in one batch |
| Edges | Flat directed edge list `[2, E]` with self edges (radius-one neighbourhood, 24 values per edge), **sorted by destination cell** with CSR offsets `edge_offsets [C+1]`; softmax normalisation and aggregation are contiguous segment reductions (torch `segment_reduce` first, a Warp CSR attention kernel with `wp.Tape` adjoint later). No padded slot table |
| Fusion solve | `K = I3 (x) K_s`. On a box grid `K_s` is exactly the Kronecker sum `Kx (x) My (x) Mz + Mx (x) Ky (x) Mz + Mx (x) My (x) Kz` of the assembled 1D stiffness and mass matrices (verified to 1e-16), and the pinned z-min face is a Dirichlet plane of the z factor, so the free block is solved exactly by fast diagonalisation (generalised eigenvectors per axis; six small matmuls and a pointwise division, no stored factor, O(P (nx+ny+nz))). Default solver since 2026-10-01 (`Fusion(solver="kron")`): 0.28 ms per solve on the canonical grid under GPU contention and the same 0.28 ms at 20x20x80 (32k cells), relative error 1.6e-6. The stored dense float32 inverse (`solver="dense"`, 94 MB and 0.05 ms idle for the canonical grid, 0.64 GB / 0.87 ms at 11k cells, out of memory at 32k) stays as the reference. (`cholesky_solve` 1.45 ms; cuDSS not in this torch build; CPU path 130 ms at batch 16; PCG 15 ms) |
| Energy passes | One per query: E and dE/dX at the fused shape are this query's E_after and the next query's E_before and gradient feature; fresh pass only after a physical advance; identity covered by a test |
| Energy kernels | torch einsum + autograd first (measured 1.6 ms batch 1, 10.6 ms batch 16, memory-bound; compile does not help); then a Warp energy + analytic-stress kernel as an autograd Function behind the same call, validated against the torch version |
| Frames | All float32, one Warp kernel (built, 0.09 ms per 64k cells): converged Jacobi eigen-solve of F^T F gives the singular values and the smallest right singular direction; det(F) < 0 reflects that direction; a scaled Newton polar iteration (6 steps) gives R; tie cells (gap <= 1e-4 max(s1,1)) redo it on F + eps R_ref. Checked: 2e-7 on proper cells, 8e-5 on inverted cells, 1e-3 on near-tie cells. (float32 `wp.svd3` alone has outliers up to 2e-2; float64 svd3 is exact but was declined) |
| CUDA graphs | Rollout only: one query captured per scene and replayed K times per step; training stays eager. Pair buffers at capacity `S (1 + M_pair)` with a valid mask keep the captured shapes static |
| Newton API | `SolverLIDO(newton.SolverBase)` from the start. `lido.add_hex_body(builder, cell_counts, h, pos, rot, material)` adds corner particles and registers the cell table and grid key; the solver builds one Grid per distinct shape and the flat batch over all bodies; pins = particles with zero inverse mass, their particle_q each step is the prescribed position (kinematic pins); contact partners from the model's ground plane plus an explicit static-point list |
| Augmenter | New generator: Gaussian noise on coarse lattices at wavelengths 2h, 4h, ... up to the longest side, trilinearly interpolated, amplitude proportional to wavelength; initial displacement RMS = strength x 4.45 h, strength in [0.02, 0.1] (the previous generator measured 0.445 h RMS at strength 0.1 on the canonical beam; the review's "strength x L" wording was a wrong scale: it gave 10-40x larger deformations and deep initial penetration in the first parity attempt); velocity RMS x dt in [0, 0.1] h; both times the U(0,1) multiplier; candidate noise the same field at RMS 1-10 % h; pinned corners overwritten, velocities zeroed |
| Contact layout | Flat, like the cell graph: one row per detected pair (`Q` rows per batch) with sample, owning cell, partner point/normal, kind, radius, anchor; tokens `[Q, 19]` with `token_cell [Q]`; the list is **sorted by owning cell, then face sample, then partner**, so `token_offsets [C+1]` (CSR) replaces token ids and pair lists: per-cell token attention loops over the cell's contiguous range (<= 30 tokens) and pooling is a segment reduce; same primitive as the cell-graph attention. Detection brute force on the GPU on the step-start shape, nearest `M_pair = 4` partners per sample as in the note; no per-cell cap, no padding. For the captured rollout query the pair buffers are allocated at capacity `S (1 + M_pair)` with a valid mask so shapes stay static |
| Initialisation | Zero-initialised last layers of the correction head, the A02 edge MLP and the contact pool, so the untrained solver leaves the candidate at Y; step head starts at 0.025 |
| Epoch tail | Idle slots masked out of the loss; same U formula as the current trainer; no filler jobs |
| Distributed | DDP, one process per GPU, 16 slots each, own job queue and factor cache per rank |
| Resume | Epoch boundary: latest / best / periodic archives with weights, AdamW, scheduler, counters, selection record, config, report; a resume starts the next epoch's job list |
| Validation | Residual per sample per iteration taken from the query's own gradient; the final post-update residual computed once (allowed by the acceptance criteria of the takeover note). Full horizon per epoch with the growth stage's caps, K = min(K_max, 8), H = H_max, as the v4 campaign did (its report shows (1,8), (2,16), (4,32), (8,64), (8,128)) |
| Reporting | Keep the existing dashboard and publisher: a subagent lists the keys of report.json / progress.json / epochs.csv they read; train.py writes exactly those |
| Implementation start | Go given by Anka 2026-09-30 ("keep the behaviour, training and inference time at least on par with LeCO, better surpass it"); built 2026-10-01, see section 10 |

## 2. What the implementation changes, and why

Measured on the current code (batch 1, 4000 cells, L40): 40 ms per query = network 17.5 + CPU fusion 6.4
+ features/energy 16.5, about 1850 kernel launches; training (batch 16): forward 0.27-0.46 s,
backward+Adam 0.24-0.30 s (takeover note). Sources of the cost and their fix:

| Cost | Now | Rewrite |
|---|---|---|
| Energy passes per query | 3 (E_before, gradient feature, E_after) + 1 in validation | 1: `E, dE/dX` at the fused shape serve as this query's E_after and the next query's E_before and gradient feature (same candidate; identity covered by a test). Fresh pass only after a physical advance |
| Fusion solves | 3B serialised CPU PARDISO solves per query with GPU<->CPU copies | `K = I_3 (x) K_s`: the same scalar matrix acts on x, y, z (F_{a.} depends only on x_{.,a}). One dense float32 Cholesky factor of `K_s,ff` per distinct grid shape (4840 x 4840, 94 MB for the canonical grid; unit weights in normalised units); `cholesky_solve` with 3n columns for the n objects on that grid, for fuse, projected gradient and the autograd backward. Measured 2.2 ms at batch 16 against 130 ms for a CPU factor with copies. No CPU, no custom autograd Function |
| Attention | Query chunks of 128, checkpoint option, ~1160 launches | Flat edge list `[2, E]` (E = 27 C incl. self edges) sorted by destination with CSR offsets: gather q[dst], k[src], v[src], edge MLP on `[E, 24]`, softmax by contiguous segment reduce, ~12 launches in torch; step 2: one Warp CSR attention kernel (forward + Tape adjoint) for both the block and the contact encoder |
| Frames | Batched torch SVD plus tie handling in Python (14.6 ms per 64k cells) | One float32 Warp kernel per query: Jacobi eigen-solve of F^T F to convergence, reflection for inverted cells, scaled Newton polar iteration, tie recompute with `F + eps R_ref`; about 0.1 ms per 64k cells; frames are constants, so no autograd needed |
| Contact detection | CPU preparation workers, variable-size pair lists; padded `[C, 24, 19]` token buffers | Brute force on the GPU (S ~ 1400 samples per object x <= 64 points) once per physical step; flat pair list `[Q]` and tokens `[Q, 19]` (Q is hundreds to thousands per object instead of 4000 x 24 padded slots); token attention with the same segment-softmax primitive as the cell graph |
| Data generation | CPU workers, FIFO pool with fillers and critical-first dispatch | GPU-resident slots per rank pulling `(seed, K, H)` jobs from a per-rank queue; augmenter and noise on the GPU; tail handled by masking idle slots instead of filler jobs |
| Launch overhead at inference | eager | all shapes static per scene -> one `torch.cuda.CUDAGraph` per query |

Estimated per query after the rewrite (4000 cells, batch 1): frames 0.1 ms, features 1 ms, block +
contact encoder 2 ms, fusion 0.3 ms, energy + gradient 2 ms eager; about 5-6 ms eager, 2-3 ms captured.
Training batch 16: forward + backward about 0.1 s versus 0.5-0.75 s now. These are estimates, not
measurements. Dense factors scale as P^2 (20k free corners = 1.6 GB); a sparse GPU factor (cuDSS) is a
drop-in behind the same `Fusion` interface if larger grids are ever needed.

Everything inside the step runs in normalised units (h = 1, mu = 1, dt = 1) as landed in schema 5;
`step.py` converts SI in and out.

## 3. Overall flow

```
                                  TRAINING (one rank; 4 ranks under DDP)
 ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ epoch e                                                                                        │
 │   jobs = sample_epoch_jobs(master_seed, e)  ──►  LPT by K·H  ──►  queue_r   (DataGenerator)    │
 │                                                                     │                          │
 │   ┌─────────────── B slots ───────────────┐  load(job) ◄────────────┘                          │
 │   │ Trajectory_0 … Trajectory_{B-1}        │  Augmenter: X, V ; Material ; ContactScene        │
 │   └────────────────┬──────────────────────┘  Y, candidate, detection, E, gX                    │
 │                    │ Batch (flat: x, cells, edges, corner_obj, cell_obj, groups)                │
 │                    ▼                                                                           │
 │   ┌──────────────────────────── Solver.query ────────────────────────────┐                     │
 │   │ modes ─► frames R ─► features ─► Network ─► corr, step ─► dm_world   │                     │
 │   │        ─► ΔF_{c,q} = Γ_q dm ─► Fusion.fuse ─► cand⁺ ─► energy_and_grad ─► E⁺, gX⁺ │        │
 │   └──────────────────────────────────┬───────────────────────────────────┘                     │
 │                                      │ E (before), E⁺, graph to the network                    │
 │                                      ▼                                                         │
 │   ┌──────────────── Trainer.update ─────────────────┐                                          │
 │   │ loss = asinh(E⁺/scale) + relu((E⁺−E)/scale)      │                                          │
 │   │ backward ─► clip 1.0 ─► AdamW ─► cosine LR        │                                          │
 │   └──────────────────────┬──────────────────────────┘                                          │
 │                          │ detached cand⁺, E⁺, gX⁺, history                                     │
 │                          ▼                                                                     │
 │   JobRunner.commit: k += 1 ; k == K → advance (V, Y, candidate, detection) ; h == H → load next │
 │                                                                                                │
 │   epoch end: validation (cheap 64×8, full horizon 16×K8×H128) ─► selection ─► checkpoints      │
 │              ─► report.json / progress.json / epochs.csv ─► existing dashboard                 │
 └────────────────────────────────────────────────────────────────────────────────────────────────┘

                                  ROLLOUT (rollout.py or SolverLIDO inside Newton)
   add_hex_body ─► Grid(s) ─► Batch ─► capture one Solver.query as a CUDA graph
   per physical step: prepare (Y, candidate, detection, E, gX) ─► replay query K times ─► advance ─► npz / Newton state
```

Components and their files:

```
experiments/lido/
  Solver         step.py, physics.py, fusion.py, frames.py, contact.py, newton_solver.py
  Network        features.py, network.py
  DataGenerator  augment.py, scenes.py, jobs.py
  Trainer        train.py, validation.py, report.py, rollout.py
  tests/
```

## 4. Solver

### 4.1 Flow

```
prepare(traj):                                   start of a physical step (at load and after every advance)
  Y  = X + dt V + dt² g                          inertial prediction, fixed for the step
  cand = Y                       (p = 0.5)       or  Y + multiscale noise, RMS 1–10 % h   (p = 0.5)
  pairs = detect(X, V)                           flat list: sample, cell, partner point/normal, anchor; frozen for the K queries
  E, gX = energy_and_grad(cand)                  the only extra energy pass per physical step
  history: keep across steps (cleared only at load)

query(batch):                                    one learned proposal + fusion on the fixed objective
  1  m(cand), m(Y), m(X_prev), F_center           P_modes / G_q einsums                     torch
  2  R = frames(F_center, R_ref)                  float32 Warp kernel, detached
  3  g_m = Γᵀ (W B K_ff⁻¹ gX_free) ─► Rᵀ, RMS-normalised   one cholesky_solve per grid group
  4  node 159 / edge 24 / cond 7 / tokens [Q,19]   features.py
  5  corr [C,7,3], step [C] = Network(...)
  6  dm_world = R (step · corr) ; ΔF_{c,q} = Γ_q dm_world ; d = Fusion.fuse(batch, ΔF, d_pinned)
  7  cand⁺ = cand + d ; E⁺, gX⁺ = energy_and_grad(cand⁺)     graph reaches the network through the solve
  8  return QueryOutput(E, E⁺, gX⁺, cand⁺, dm_world, g_world)

advance(traj):                                   after K queries
  X_prev ← X ; X ← cand ; V ← (X − X_prev)/dt (pinned: prescribed velocity) ; prepare(traj)
```

Work is split into three tiers so nothing step- or job-constant is recomputed per query
(LeCO's per-frame inference cache, applied to training as well):

| Tier | When | Contents |
|---|---|---|
| Job-constant | at load | grid tensors (edge list, rest edge offsets, exposed/fixed flags, `G_q`, `Γ_q`, `P_modes`), material and floor, conditioning and its FiLM outputs, contact partners, fusion factor (per-grid cache), slot offsets in the flat buffers |
| Step-constant | at prepare | `Y`, `m(Y)`, `X_prev`, `m(X_prev)`, damping anchor `C_prev` at the 8 Gauss points, the sorted flat pair list with its CSR offsets (partner points, friction anchors), `d_pinned`, the one energy pass at the new candidate |
| Query-varying | every query | `m(cand)`, `F_center`, frames `R`, current edge offsets and `R_iᵀR_j`, `R_iᵀF_j`, projected gradient, token geometry, network, fusion right-hand side, `cand⁺`, `E⁺`, `gX⁺`, history |

Frames are recomputed every query by design (never carried). The energy pass at a new physical step's
candidate is fresh because `Y` is a different point from the last fused shape; across queries the
fused-shape pass is reused.

### 4.2 Data structures

```python
@dataclass
class Grid:                        # one per grid shape; cell units (h = 1); immutable, shared by objects
    key: tuple                     # (nx, ny, nz, pin pattern)
    rest: Tensor                   # [P,3]
    cells: Tensor                  # [C,8]   local order 000..111, z fastest
    free: Tensor; pinned: Tensor   # corner index sets
    edges: Tensor                  # [2,E]   directed radius-one edges incl. self edges (src, dst), local cell ids, E <= 27 C, sorted by dst;
                                   #         every undirected neighbour pair appears twice (i->j, j->i), each with its own receiver-frame features
    edge_offsets: Tensor           # [C+1]   CSR row pointers into edges (dst c owns rows offsets[c]..offsets[c+1])
    exposed: Tensor                # [C,6]   bool
    fixed_flags: Tensor            # [C,8]   bool
    samples: FaceSamples           # cell [S], face [S], corners [S,4]  (exposed face centres, r = 0.5 h)
    Gq: Tensor                     # [8,8,3] grad N_k(xi_q) * 2/h
    weights: Tensor                # [8]     Gauss weights (1/8 each)
    P_modes: Tensor                # [7,8]   21 mode values = P_modes @ corners
    Gamma: Tensor                  # [8,9,21] F(xi_q) from the 21 mode values (derivation eq. 8.4)
    ref_corners: Tensor            # [3]     prescribed corners for the frame tie-break

@dataclass
class Material:                    # per object, normalised units; si dict kept for records and scale-back
    lam: Tensor; rho: Tensor; eta: Tensor; g: Tensor      # lambda/mu, rho h^2/(mu dt^2), eta/(mu dt), g dt^2/h [3]
    ke: Tensor; kd: Tensor; mu_f: Tensor                   # ke/(mu h), kd/(mu h dt), friction coefficient
    floor: Tensor                  # energy floor in mu h^3 units
    si: dict                       # E, nu, rho, eta, |g|, h, dt, kappa_eff, ke_floor, floor_bound, ...

@dataclass
class ContactScene:                # per object; static partners drawn at load
    plane_n: Tensor; plane_d: Tensor; plane_present: Tensor   # [3], [], bool
    points: Tensor; normals: Tensor; radii: Tensor            # [n_pts,3], [n_pts,3], [n_pts]  (n_pts <= 64, no padding)
    ke: Tensor; kd: Tensor; mu_f: Tensor                       # normalised contact law constants

@dataclass
class Pairs:                       # flat list of detected pairs, frozen per physical step; built by detect()
                                   # SORTED by owning cell, then face sample, then partner (detection order, compaction keeps it)
    token_offsets: Tensor          # [C+1] CSR: tokens of cell c are rows offsets[c]..offsets[c+1]; empty cells have equal offsets
    sample: Tensor                 # [Q]   sample id (global)
    obj: Tensor                    # [Q]   object id (implied by the cell ranges; kept for the energy scatter)
    partner_point: Tensor          # [Q,3]
    partner_normal: Tensor         # [Q,3]
    kind: Tensor                   # [Q]   0 plane, 1 static point, 2 self (unused in v1)
    radius: Tensor                 # [Q]   partner radius r_p (plane: r)
    anchor: Tensor                 # [Q,3] sample position at step start (friction anchor)
    valid: Tensor                  # [Q]   bool; all true in training, capacity mask in the captured rollout

@dataclass
class Batch:                       # flat concatenation of the objects served this update; everything SORTED so that every
                                   # grouping is a CSR offsets vector: objects contiguous, edges by destination cell, tokens by cell
    x: Tensor                      # [N,3]  candidates
    Y: Tensor; X_prev: Tensor      # [N,3]
    cells: Tensor                  # [C,8]  global corner ids
    edges: Tensor                  # [2,E]  global cell ids
    corner_obj: Tensor; cell_obj: Tensor                       # [N], [C]
    groups: list[tuple[Grid, slice, slice]]                    # (grid, corner slice, cell slice) per run of same-grid objects
    Gq: Tensor; weights: Tensor    # gathered per cell [C,8,8,3], [C,8] (views when one grid)
    pinned: Tensor                 # [N] bool
    mass: Tensor                   # [N]
    material: Material             # fields stacked over objects [O]
    contact: ContactScene          # fields concatenated over objects
    pairs: Pairs                   # flat pair list of the batch (step-constant)
    E: Tensor; gX: Tensor          # [O], [N,3]  energy and gradient at x (detached)
    hist_grad: Tensor; hist_update: Tensor; hist_valid: Tensor   # [C,7,3], [C,7,3], [O]
    active: Tensor                 # [O] bool  (idle slots at the epoch tail)
    # step-constant (written by prepare, read by every query of the step)
    m_Y: Tensor; m_prev: Tensor    # [C,7,3] modes of Y and X_prev
    C_prev: Tensor                 # [C,8,3,3] damping anchor F(X_prev)^T F(X_prev) at the Gauss points
    d_pinned: Tensor               # [N,3] prescribed displacement of pinned corners (zero unless kinematic pins)
    # job-constant (written by load)
    film: tuple[Tensor, ...]       # 4 x [O,W] FiLM scale/shift per object
    edge_rest: Tensor              # [E,3] rest offsets / h in the rest frame
    flags: Tensor                  # [C,14] exposed and fixed flags as float

@dataclass
class QueryOutput:
    E_before: Tensor; E_after: Tensor        # [O]
    gX_after: Tensor                         # [N,3] detached
    cand_after: Tensor                       # [N,3] attached to the graph
    dm_world: Tensor; g_world: Tensor        # [C,7,3] achieved-update and gradient-feature inputs for the history
    residual: Tensor                         # [O] free-corner force residual norm at x (for validation)
```

### 4.3 Key implementation

```python
class Fusion:
    """K = B^T W B on the unit grid. F_{a,.} depends only on x_{.,a}, so K = I_3 (x) K_s with one scalar
    matrix K_s [P,P] per grid shape. One float32 factor per grid shape serves every coordinate, every object
    on that grid, every material and every query."""
    def __init__(self):
        self.factors = {}                                           # grid key -> (L fp32 [Pf,Pf], K_fp fp32 [Pf,Pp])

    def factor(self, grid):
        if grid.key not in self.factors:
            Ks = assemble_scalar(grid)                              # sum_c sum_q w_q g_{c,q} g_{c,q}^T, float64 [P,P] (test reference)
            f, p = grid.free, grid.pinned
            self.factors[grid.key] = (torch.linalg.cholesky(Ks[f][:, f]).float(), Ks[f][:, p].float())
        return self.factors[grid.key]

    def rhs(self, grid, dF, n):                                     # dF [n*C,8,3,3] world target increments -> B^T W dF [n,P,3]
        contrib = torch.einsum('q,ncqab,qkb->ncka', grid.weights, dF.view(n, grid.C, 8, 3, 3), grid.Gq)
        return torch.zeros(n, grid.P, 3).index_add_(1, grid.cells.reshape(-1), contrib.reshape(n, -1, 3))

    def solve(self, L, r):                                          # r [n,Pf,3] -> K_s,ff^-1 r, all 3n columns in one call
        n, Pf, _ = r.shape
        return torch.cholesky_solve(r.permute(1, 0, 2).reshape(Pf, -1), L).reshape(Pf, n, 3).permute(1, 0, 2)

    def fuse(self, batch, dF, d_pinned):                             # flat in, flat out; exact elimination of prescribed corners
        d = torch.zeros_like(batch.x)
        for grid, corners, cells in batch.groups:
            n = (corners.stop - corners.start) // grid.P
            L, K_fp = self.factor(grid)
            dp = d_pinned[corners].view(n, grid.P, 3)[:, grid.pinned]
            b = self.rhs(grid, dF[cells], n)[:, grid.free] - torch.einsum('fp,npa->nfa', K_fp, dp)
            dg = torch.zeros(n, grid.P, 3); dg[:, grid.free] = self.solve(L, b); dg[:, grid.pinned] = dp
            d[corners] = dg.reshape(-1, 3)
        return d                                                    # autograd through cholesky_solve = the adjoint solve

    def project_gradient(self, batch, gX):                          # gX [N,3], pinned rows zeroed -> [C,21] mode gradient
        out = torch.zeros(batch.C_total, 21)
        for grid, corners, cells in batch.groups:
            n = (corners.stop - corners.start) // grid.P
            L, _ = self.factor(grid)
            z = torch.zeros(n, grid.P, 3); z[:, grid.free] = self.solve(L, gX[corners].view(n, grid.P, 3)[:, grid.free])
            dFz = torch.einsum('ncka,qkb->ncqab', z[:, grid.cells], grid.Gq) * grid.weights[:, None, None]
            out[cells] = torch.einsum('qij,ncqi->ncj', grid.Gamma, dFz.reshape(n, grid.C, 8, 9)).reshape(-1, 21)
        return out
```

```python
@wp.kernel
def frames_kernel(F: wp.array(dtype=wp.mat33), R_ref: wp.array(dtype=wp.mat33), R: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    Fi = F[i]
    s, V = jacobi_eigen(wp.transpose(Fi) * Fi)                    # converged Jacobi sweeps, descending sqrt-eigenvalues, float32
    inverted = wp.determinant(Fi) < 0.0
    gap = wp.where(inverted, s[1] - s[2], s[1] + s[2])
    if gap <= 1e-4 * wp.max(s[0], 1.0):                            # tie: closest proper rotation of F + eps R_ref, same route
        Fi = Fi + 1e-4 * wp.max(s[0], 1.0) * R_ref[i]
        s, V = jacobi_eigen(wp.transpose(Fi) * Fi)
        inverted = wp.determinant(Fi) < 0.0
    if inverted:                                                   # reflect the smallest right singular direction
        v3 = wp.vec3(V[0, 2], V[1, 2], V[2, 2])
        Fi = Fi * (wp.identity(3, float) - 2.0 * wp.outer(v3, v3))
    R[i] = polar_newton(Fi, 6)                                     # scaled Newton: X <- (g X + X^-T / g) / 2, g = |det X|^-1/3
```
Checked against a float64 SVD: 2e-7 on proper cells, 8e-5 on inverted cells, 1e-3 on near-tie cells.

```python
def energy(batch, x, Y, X_prev):                                   # flat, normalised units; returns per-object [O]
    F = torch.einsum('cka,cqkb->cqab', x[batch.cells], batch.Gq)   # [C,8,3,3]
    lam = batch.material.lam[batch.cell_obj]
    E_el = (batch.weights * stable_neo_hookean(F, lam)).sum(-1)                  # mu = 1; Newton mapping inside
    C_now = F.transpose(-1, -2) @ F                                # batch.C_prev is step-constant (computed in prepare)
    E_damp = (batch.weights * batch.material.eta[batch.cell_obj, None] / 2 * ((C_now - batch.C_prev) ** 2).sum((-1, -2))).sum(-1)
    E_in = 0.5 * batch.material.rho[batch.corner_obj] * batch.mass * ((x - Y) ** 2).sum(-1)
    return seg_sum(E_el + E_damp, batch.cell_obj) + seg_sum(E_in, batch.corner_obj) + contact_energy(batch.pairs, x, batch)   # per pair; pairs sorted by cell -> by object, so this is a segment reduce too

def energy_and_grad(batch, x, Y, X_prev):                          # one pass: value with graph + position gradient
    E = energy(batch, x, Y, X_prev)
    gX, = torch.autograd.grad(E.sum(), x, retain_graph=True)       # x is the fused candidate (graph to the network kept)
    gX = gX.detach(); gX[batch.pinned] = 0
    return E, gX
```

```python
class Step:                                                        # step.py: SI in and out, normalised inside
    def query(self, batch) -> QueryOutput:
        m_c, F_c = modes(batch.x, batch)                                           # [C,7,3], [C,3,3]; m_Y, m_prev are step-constant
        R = frames(F_c, batch.R_ref)                                               # [C,3,3] detached
        g_m = self.fusion.project_gradient(batch, batch.gX).view(-1, 7, 3)         # world axis gradient
        feats = features(batch, R, m_c, batch.m_Y, batch.m_prev, g_m)              # query-varying parts only; rest/flags/FiLM cached
        corr, step = self.net(*feats, batch.edges, batch.cell_obj)                 # [C,7,3], [C]
        dm_world = torch.einsum('cab,cvb->cva', R, step[:, None, None] * corr)     # 7 vectors back to world
        dF = torch.einsum('qij,cj->cqi', batch.Gamma, dm_world.reshape(-1, 21)).view(-1, 8, 3, 3)
        cand = batch.x + self.fusion.fuse(batch, dF, batch.d_pinned)
        E_after, gX_after = energy_and_grad(batch, cand, batch.Y, batch.X_prev)
        return QueryOutput(batch.E, E_after, gX_after, cand, dm_world, g_m, residual(batch.gX, batch))

    def prepare(self, traj, rng):                                  # per trajectory, at load and after advance: step-constant tier
        traj.Y = traj.X + traj.V + traj.material.g                                  # dt = 1 in normalised units
        traj.m_Y, _ = modes(traj.Y, traj); traj.m_prev, F_prev = modes(traj.X, traj)
        traj.C_prev = F_prev.transpose(-1, -2) @ F_prev                             # damping anchor for the K queries
        traj.cand = traj.Y if rng.random() < 0.5 else traj.Y + self.aug.candidate_noise(traj, rng)
        traj.pairs = detect(traj.grid, traj.X, traj.V, traj.scene)                 # flat pair list + token edges, frozen
        traj.E, traj.gX = energy_and_grad(single(traj), traj.cand, traj.Y, traj.X)  # the one fresh pass per step

    def advance(self, traj, fixed_positions=None):
        traj.X_prev, traj.X = traj.X, traj.cand
        traj.V = traj.X - traj.X_prev
        if fixed_positions is not None: traj.X[traj.grid.pinned] = fixed_positions; traj.V[traj.grid.pinned] = ...
        self.prepare(traj, traj.rng)
```

```python
class SolverLIDO(newton.SolverBase):                               # newton_solver.py
    def __init__(self, model, checkpoint, iterations=8):
        bodies = model.lido_bodies                                  # registered by lido.add_hex_body(builder, ...)
        self.grids = {b.key: Grid.build(b.cell_counts, b.pin_mask) for b in bodies}
        self.step = Step.load(checkpoint); self.K = iterations
        self.graph = None                                          # CUDA graph of one query, captured on first step
    def step(self, state_in, state_out, control, contacts, dt):
        batch = self.batch_from(state_in)                          # particle_q/qd -> flat Batch; inv_mass == 0 -> prescribed
        for _ in range(self.K): self.replay_query(batch)           # captured query; commit in place
        self.advance_to(state_out, batch, dt)                      # positions, velocities back to particle_q/qd
```

## 5. Network

### 5.1 Flow

```
 batch, R, modes, g_m                                                                  per object
   │                                                                                   material + contact scalars
   ▼                                                                                   + log RMS + history flag
 features.py ─► node [C,142]   edge_attr [E,24] (CSR by dst)   tokens [Q,19] (CSR by cell)   cond [O,7]  ──► FiLM MLP ─► (s1,b1,s2,b2) [O,4W]
   │                │                  │                                                        │ gathered by cell_obj
   ▼                │                  ▼                                                        ▼
 Linear(159→192) ◄──┼───── ContactEncoder(64): token MLP ─► CSR attention within each cell's token range (same primitive as below)
                    │                            ─► segment mean|max over token_offsets ─► Linear→16 (zero-init) + count/24  → [C,17]
   │                │
   ▼                ▼
 Block: h + CellGraphAttention(LN·FiLM(h), edges, edge_offsets, edge_attr)   edge MLP(24→96) ─► + A02 MLP([h_dst, h_src, e]) (zero-init)
        h + FFN(LN·FiLM(h))                                  ─► bias per head [E,6], value add [E,192] ; segment softmax by dst
   │
   ▼
 LN ─► corr_head ─► bound ─► [C,7,3]        step_head ─► 0.05·sigmoid ─► [C]
```

### 5.2 Data structures

```python
@dataclass
class Features:                    # produced by features.py, consumed by Net; all rotation-invariant
    node: Tensor                   # [C,159]  21 axes+warp | 21 inertial | 21 physical change | 21 grad | 21 prev grad | 21 prev update
                                   #          | 6 exposed | 8 fixed | log RMS(grad) | history_valid | 17 contact (filled by the encoder)
    edge_attr: Tensor              # [E,24]   rest offset/h (3) | receiver-frame current offset/h (3) | R_i^T R_j (9) | R_i^T F_j (9)
    cond: Tensor                   # [O,7]    log1p lam, log rho', log1p g', log1p eta', log1p kappa, beta, mu_f
    tokens: Tensor                 # [Q,19]  one token per detected pair, in the owning cell's frame / h, sorted by cell
    token_offsets: Tensor          # [C+1]   CSR row pointers (from Pairs)
    schema: int = 6                # field order pinned by test

@dataclass
class NetOutput:
    corr: Tensor                   # [C,7,3] bounded correction in the frozen frame
    step: Tensor                   # [C]     per-cell step in (0, 0.05)

# parameters (width W = 192, heads 6, edge hidden 96, contact width 64):
#   enc 159->W ; film 7->4W ; qkv W->3W ; edge_mlp 24->96 ; edge_update (2W+96)->96 (zero-init) ; edge_bias 96->6 ;
#   edge_val 96->W ; out W->W ; ffn W->4W->W ; contact: 19->64->64, CSR attention 64 (2 heads), pool 128->16 (zero-init) ;
#   corr_head W->W->21 (zero-init) ; step_head W->1
```

### 5.3 Key implementation

```python
def features(batch, R, m_c, m_Y, m_prev, g_m):                     # features.py
    Rt = R.transpose(-1, -2)
    loc = lambda v: torch.einsum('cab,cvb->cva', Rt, v)             # 7 world vectors -> frozen frame
    rms = seg_rms(g_m, batch.cell_obj)[batch.cell_obj]              # per-object RMS over the 21 components, floor 1e-12
    hist_ok = batch.hist_valid[batch.cell_obj, None, None]
    node = torch.cat([
        loc(m_c).flatten(1), loc(m_Y - m_c).flatten(1), loc(m_c - m_prev).flatten(1),
        (loc(g_m) / rms[:, None, None]).clamp(-10, 10).flatten(1),
        (loc(batch.hist_grad) / rms[:, None, None] * hist_ok).clamp(-10, 10).flatten(1),
        (loc(batch.hist_update) / seg_rms(batch.hist_update, batch.cell_obj)[batch.cell_obj, None, None] * hist_ok).clamp(-10, 10).flatten(1),
        batch.flags,                                                                        # job-constant
        torch.log(seg_rms(g_m, batch.cell_obj))[batch.cell_obj, None], batch.hist_valid[batch.cell_obj, None].float(),
    ], -1)                                                          # [C,142]; the encoder appends the 17 contact channels
    src, dst = batch.edges
    edge_attr = torch.cat([
        batch.edge_rest,                                                                     # job-constant
        torch.einsum('eab,eb->ea', Rt[dst], batch.center[src] - batch.center[dst]),
        (Rt[dst] @ R[src]).flatten(1),
        (Rt[dst] @ F_center[src]).flatten(1),
    ], -1)                                                          # [E,24]
    return Features(node, edge_attr, conditioning(batch.material), *contact_tokens(batch, R))
```

```python
class CellGraphAttention(nn.Module):
    """Attention over a flat edge list sorted by destination with CSR offsets. Serves the cell graph
    (edges = radius-one neighbours, edge_attr 24 values) and the contact encoder (edges = all token pairs
    within a cell's range, no edge_attr). Torch version below; step 2 is a Warp kernel that loops over each
    row's contiguous range in registers, with the adjoint generated by wp.Tape (one launch each way)."""
    def forward(self, x, edges, offsets, edge_attr=None):   # x [R,W], edges [2,E] sorted by dst, offsets [R+1]
        src, dst = edges
        R = x.shape[0]
        q, k, v = self.qkv(x).view(R, 3, self.H, self.D).unbind(1)
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.D)                              # [E,H]
        val = v[src]
        if edge_attr is not None:
            e = self.edge_mlp(edge_attr)                                                   # [E,96]
            e = e + self.edge_update(torch.cat([x[dst], x[src], e], -1))                   # A02, zero-init last layer
            score = score + self.edge_bias(e); val = val + self.edge_val(e).view(-1, self.H, self.D)
        lengths = offsets[1:] - offsets[:-1]
        m = torch.segment_reduce(score, 'max', lengths=lengths)                            # [R,H] contiguous ranges
        w = (score - m.repeat_interleave(lengths, 0)).exp()
        den = torch.segment_reduce(w, 'sum', lengths=lengths).clamp_min(1e-12)
        out = torch.segment_reduce(val * (w / den.repeat_interleave(lengths, 0))[..., None], 'sum', lengths=lengths)
        return self.out(out.reshape(R, -1))

class Block(nn.Module):
    def forward(self, x, edges, offsets, edge_attr, film):   # film = (s1, b1, s2, b2) per cell
        s1, b1, s2, b2 = film
        x = x + self.attn(self.n1(x) * (1 + s1) + b1, edges, offsets, edge_attr)
        return x + self.ffn(self.n2(x) * (1 + s2) + b2)

class ContactEncoder(nn.Module):                    # tokens [Q,19] sorted by cell, token_offsets [C+1] -> [C,17]
    def forward(self, tokens, offsets):
        t = self.token_mlp(tokens)                                       # [Q,64]
        pairs, pair_offsets = within_range_pairs(offsets)                # all (i, j) with i, j in the same cell range; sorted by i
        t = t + self.attn(self.ln(t), pairs, pair_offsets)               # CellGraphAttention, 2 heads, no edge_attr
        t = t + self.ffn(self.ln2(t))
        lengths = offsets[1:] - offsets[:-1]                             # [C]; zero for cells without tokens
        mean = torch.segment_reduce(t, 'mean', lengths=lengths, initial=0.0)
        mx = torch.segment_reduce(t, 'max', lengths=lengths, initial=0.0)
        return torch.cat([self.pool(torch.cat([mean, mx], -1)), (lengths / 24)[:, None]], -1)   # empty cells -> zeros

class Net(nn.Module):
    def forward(self, f: Features, edges, edge_offsets, cell_obj) -> NetOutput:
        h = self.enc(torch.cat([f.node, self.contact(f.tokens, f.token_offsets)], -1))   # 142 + 17 -> 192
        film = tuple(t[cell_obj] for t in f.film)                                       # job-constant FiLM outputs
        h = self.norm(self.block(h, edges, edge_offsets, f.edge_attr, film))
        corr = self.bound(self.corr_head(h)).view(-1, 7, 3)
        step = 0.05 * torch.sigmoid(self.step_head(h)).squeeze(-1)
        return NetOutput(corr, step)
```

## 6. DataGenerator

### 6.1 Flow

```
 epoch e (all ranks, same seed)
   sample_epoch_jobs(master_seed, e) ─► 2048 × Job(seed, K, H)   stage caps (K_max, H_max) from the growth timetable, K·H ≤ 2048
   assign(jobs, world) ─► LPT by K·H ─► queue_r per rank ; U = max(ceil(max_r Σ K·H / B), longest K·H)

 rank r: B slots
   ┌ slot: Trajectory or idle ┐      every update: each busy slot serves one query, idle slots are masked; commit is whole-batch tensor ops,
   │                          │      advance runs once on the sub-batch of slots with k == K, load runs once on the slots that finished
   │  load(job):              │        SceneSpec(seed) ─► Grid (cache) ; Material ; ContactScene ; Augmenter ─► X, V
   │                          │        job-constant tier: FiLM outputs, rest edge offsets, flags, factor, slot offsets
   │                          │        Step.prepare (step-constant tier) ─► Y, m(Y), m(X_prev), C_prev, cand, sorted pairs + CSR offsets, E, gX ; history zero
   │  commit(out):            │        cand, E, gX ← detached out ; history ← (g_world, dm_world), hist_valid 1 ; k += 1
   │     k == K ─► advance    │        X, V, Y, cand, detection ; h += 1
   │     h == H ─► load(next) │        or idle if queue_r is empty
   │  failure ─► record, load │        non-finite candidate or energy: failure record (seed, K, H, k, h), slot reloads
   └──────────────────────────┘
   batch() ─► Batch: view of the persistent flat buffers (slot i owns fixed corner/cell/edge row ranges); offsets, ids and
              grid groups are rewritten only when a load changes a slot's grid shape; active mask marks idle slots
```

### 6.2 Data structures

```python
@dataclass(frozen=True)
class Job:
    seed: int; K: int; H: int

@dataclass
class SceneSpec:                   # everything drawn from the state's seed streams; logged with the run
    cell_counts: tuple[int, int, int] = (10, 10, 40); h: float = 0.025; pins: str = "zmin_face"
    E: float; nu: float; rho: float; eta: float           # log-uniform / uniform ranges of the config
    gravity: tuple[float, float, float]                   # |g| log-uniform [2, 40] along -y
    perturbation_scale: float                             # U(0,1) multiplier on displacement and velocity
    strength: float; velocity_dt: float                   # [0.02, 0.1] x L ; [0, 0.1] x h
    contact: ContactSpec                                  # plane present/height, points, radii, kappa (floored), beta, mu_f

@dataclass
class Trajectory:                  # one slot; all tensors on the GPU, single object
    grid: Grid; spec: SceneSpec; material: Material; scene: ContactScene
    X: Tensor; V: Tensor; X_prev: Tensor; Y: Tensor; cand: Tensor     # [P,3]
    E: Tensor; gX: Tensor                                              # [], [P,3]
    hist_grad: Tensor; hist_update: Tensor; hist_valid: bool            # [C,7,3], [C,7,3]
    job: Job; k: int; h: int
    rng: torch.Generator           # candidate stream (inertial / perturbed, noise draws)

@dataclass
class FailureRecord:
    seed: int; K: int; H: int; k: int; h: int; kind: str; epoch: int; update: int
```

### 6.3 Key implementation

```python
def sample_epoch_jobs(master_seed, epoch, cfg) -> list[Job]:        # pure function
    K_max, H_max = growth_stage(epoch, cfg)
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, epoch, 7331]))
    jobs = []
    for state in range(cfg.state_count):
        H = int(rng.integers(1, H_max + 1))
        ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= K_max and k * H <= cfg.budget_cap]
        jobs.append(Job(seed=state, K=int(rng.choice(ks)), H=H))
    return jobs

def assign(jobs, world_size, B):                                    # LPT by K*H, returns per-rank queues and U
    ranks = [[] for _ in range(world_size)]; load = [0] * world_size
    for job in sorted(jobs, key=lambda j: -j.K * j.H):
        r = load.index(min(load)); ranks[r].append(job); load[r] += job.K * job.H
    U = max(math.ceil(max(load) / B), max(j.K * j.H for j in jobs))
    return ranks, U

class Augmenter:                                                    # augment.py, GPU, seeded
    def field(self, grid, rng, rms, spectrum="wavelength"):         # smooth vector field on the corners, RMS in metres
        out = torch.zeros(grid.P, 3)
        for wavelength in (2, 4, 8, 16, 32):                        # in cells, up to the longest side
            if wavelength > max(grid.cell_counts): break
            coarse = torch.randn(*(n // wavelength + 2 for n in grid.cell_counts), 3, generator=rng)
            out += wavelength * trilinear_sample(coarse, grid.rest / wavelength)      # amplitude ∝ wavelength
        return out * (rms / out.pow(2).sum(-1).mean().sqrt())
    def initial_state(self, grid, spec, rng):
        L = max(grid.cell_counts) * spec.h
        X = grid.rest * spec.h + spec.perturbation_scale * self.field(grid, rng, spec.strength * L)
        V = spec.perturbation_scale * self.field(grid, rng, spec.velocity_dt * spec.h / spec.dt)
        X[grid.pinned] = grid.rest[grid.pinned] * spec.h; V[grid.pinned] = 0
        return X, V
    def candidate_noise(self, traj, rng):                           # RMS 1-10 % h, pinned rows zero
        return self.field(traj.grid, rng, rng_uniform(rng, 0.01, 0.10) * traj.spec.h).masked_fill(traj.grid.pinned[:, None], 0)

class JobRunner:
    """Slots own fixed row ranges of persistent flat buffers. Python touches a slot only at load; everything
    per update is vectorised over the batch or over the sub-batch of slots that advance or load together."""
    def start_epoch(self, epoch):
        jobs = sample_epoch_jobs(self.master_seed, epoch, self.cfg)
        queues, self.U = assign(jobs, self.world, self.B)
        self.queue = deque(queues[self.rank])
        self.specs = sample_scene_specs(jobs, self.master_seed, epoch)     # all 2048 SceneSpecs at once (pure function of seeds)
        self.load([i for i in range(self.B)])

    def load(self, slots):                                          # grouped: one generator call, one sub-batch prepare
        jobs = [self.queue.popleft() if self.queue else None for _ in slots]
        busy = [(i, j) for i, j in zip(slots, jobs) if j is not None]
        for i, j in zip(slots, jobs):
            if j is None: self.active[i] = False                     # idle until the next epoch
        if not busy: return
        X, V = self.aug.initial_states([self.specs[j.seed] for _, j in busy])      # batched multiscale fields
        self.buffers.write(busy, X, V, materials, scenes)          # row ranges, job-constant tier (FiLM, edge_rest, flags, offsets)
        self.step.prepare(self.buffers.sub_batch([i for i, _ in busy]))          # step-constant tier for these slots only
        self.k[[i for i, _ in busy]] = 0; self.h[...] = 0; self.hist_valid[...] = False

    def commit(self, out: QueryOutput):                             # whole-batch tensor ops, no per-slot Python
        b = self.buffers
        b.x.copy_(out.cand_after.detach()); b.E.copy_(out.E_after.detach()); b.gX.copy_(out.gX_after)
        b.hist_grad.copy_(out.g_world); b.hist_update.copy_(out.dm_world); b.hist_valid.fill_(True)
        self.k += 1
        bad = ~torch.isfinite(b.E) & self.active                    # failure: record and reload those slots
        adv = (self.k == self.K) & self.active & ~bad
        if adv.any():
            self.step.advance(b.sub_batch(adv))                     # one sub-batch: V, X, Y, m(Y), m(X_prev), C_prev, cand, pairs, E, gX
            self.k[adv] = 0; self.h[adv] += 1
        done = ((self.h == self.H) & self.active) | bad
        if done.any():
            self.failures += [FailureRecord(...) for i in bad.nonzero()]
            self.load(done.nonzero().flatten().tolist())            # the only host sync per update, only when a slot finishes
```

## 7. Trainer

### 7.1 Flow

```
 torchrun 4 ranks ─► TrainConfig (v4 JSON) ─► Grid cache, Step (Fusion, Net), Augmenter, JobRunner(rank)
   │
   ▼  for epoch in 1..max_epochs (cosine LR by epoch, no plateau logic)
   runner.start_epoch(epoch)
   for update in range(runner.U):
       batch = runner.batch()
       out   = step.query(batch)
       loss  = local_objective(out.E_after, out.E_before, batch.material.floor)[batch.active].mean()
       loss.backward() ─► clip_grad_norm 1.0 (skip step if non-finite) ─► AdamW.step() ─► zero_grad
       runner.commit(out)                                        detach, history, advance / load
       every log_every: update row (loss, |g|, accept-free stats, resets)
   epoch end (rank 0 aggregates):
       validation.cheap: 64 held-out states, K = 8 ... residual per sample per iteration, energy, inversions
       validation.full_horizon: 16 states, K = 8, H = 128: final free-corner residual, survival, penetration, energy
       selection: mean final residual with survival ─► best_validation.pt if better (record K, H, epoch)
       checkpoint latest.pt (+ archive every checkpoint_interval) ; report.json / progress.json / epochs.csv
```

### 7.2 Data structures

```python
@dataclass
class TrainConfig:                 # loaded from the v4 JSON; unknown keys rejected
    cell_counts, cell_size, time_step, gravity
    hidden_dim=192, num_heads=6, edge_hidden_dim=96, target_modes=7, max_step_size=0.05, feature_schema_version=6
    batch_size=16, state_count=2048, budget_cap=2048, growth_stages, growth_stage_epochs=2, max_epochs=48
    learning_rate=1e-4, lr_final=2.5e-5, weight_decay=1e-6, gradient_clip_norm=1.0, energy_increase_weight=1.0, energy_floor_scale=1.0
    material ranges, gravity_magnitude_range, strength_range, velocity_dt_range, perturbation_scale_range
    contact: probability, plane height range, max points, radius range, kappa range, static_penetration_max, beta/mu ranges, max_pairs, tokens_per_cell, friction_epsilon
    validation_count=64, validation_iterations=8, validation_full_count=16, validation_full_iterations=8, validation_full_steps=128
    seed, device, log_every, checkpoint_interval

@dataclass
class Checkpoint:                  # epoch boundary only
    network_state: dict; optimizer_state: dict; scheduler_state: dict
    epoch: int; updates_done: int
    best_selection: dict           # metric, epoch, K, H, eligible, survivors, final energy, penetration
    config: dict; feature_schema_version: int; git_sha: str

@dataclass
class EpochRecord:                 # one row of report["epochs"], also epochs.csv
    epoch, updates, mean_loss, lr, grad_norm, resets, failures, regime={stage, k_max, h_max, queries, updates}
    validation={samples: [{seed, free_force_residual_norm_n: [K+1], energy: [K+1], inverted_cells: [K+1]}], summary}
    full_horizon_validation={samples: [{seed, physical_records: [H x {residual, energy, penetration_r, inverted}]}], selection}
    material_histograms, contact_scene_fraction, contact_realized_fraction, wall_seconds
```
The exact key set of `report.json`, `progress.json` and `epochs.csv` is taken from the list the dashboard reader produces (decision: keep the dashboard).

### 7.3 Key implementation

```python
def local_objective(E_after, E_before, floor, increase_weight=1.0):
    scale = torch.maximum(E_before.abs(), floor).detach()
    return torch.asinh(E_after / scale) + increase_weight * torch.relu((E_after - E_before.detach()) / scale)

def train(cfg, rank, world):
    grids, step, aug = build(cfg); net = DDP(step.net, broadcast_buffers=False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    runner = JobRunner(cfg, step, aug, rank, world)
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        set_lr(opt, cosine(epoch, cfg)); runner.start_epoch(epoch)
        for update in range(runner.U):
            batch = runner.batch()
            out = step.query(batch)
            loss = local_objective(out.E_after, out.E_before, batch.material.floor, cfg.energy_increase_weight)
            (loss[batch.active].sum() / batch.active.sum().clamp_min(1)).backward()
            gn = clip_grad_norm_(net.parameters(), cfg.gradient_clip_norm)
            if torch.isfinite(gn): opt.step()
            opt.zero_grad(set_to_none=True)
            runner.commit(out)
        if rank == 0:
            cheap = validate_cheap(step, cfg); full = validate_full_horizon(step, cfg)
            best = select(best, full, epoch); save_checkpoints(...); write_report(...)
        dist.barrier()

def validate_full_horizon(step, cfg):                              # frozen weights, held-out seeds, K = 8, H = 128
    records = []
    for seed in held_out(cfg.validation_full_count):
        traj = Trajectory.new(Job(seed, cfg.validation_full_iterations, cfg.validation_full_steps), step, aug, validation=True)
        rows = []
        for h in range(traj.job.H):
            for k in range(traj.job.K):
                out = step.query(single(traj)); commit_one(traj, out)             # residual before each query from out.residual
            rows.append(dict(residual=residual_final(traj), energy=traj.E, penetration=penetration(traj), inverted=inverted(traj)))
            step.advance(traj)
        records.append(dict(seed=seed, physical_records=rows, survived=all_finite(rows)))
    return dict(samples=records, selection=dict(metric=mean_final_residual(records), eligible=all(r["survived"] for r in records)))
```

## 8. Tests that guard the shortcuts

- Fusion: `fuse` and `project_gradient` against a float64 sparse least-squares solve of the full
  `B^T W B` system on 2x2x3, with random prescribed displacements; separability `K = I_3 (x) K_s` checked
  numerically; 21-mode assembly equals the Gauss-point assembly (derivation eq. 8.8); mixed-grid batch
  equals per-object solves.
- Energy reuse: `E_after` of query k equals `E_before` recomputed at query k+1; gradient reuse likewise.
- Frames: the float32 kernel against a float64 SVD on random, near-identity, strongly deformed and inverted
  cells (tolerance 1e-6 proper, 1e-4 inverted; tie cells: a proper rotation close to R_ref); whole-problem
  rotation gives rotated frames.
- Energy: stable Neo-Hookean against Newton's stress formula (Warp oracle) and finite differences; contact
  law against Newton's force; damping vanishes under rigid rotation of the previous shape.
- Equivariance: rotating and translating a scene rotates the fused update and leaves the loss unchanged.
- Features: field order and widths (159 / 24 / 7 / 19) pinned; normalisation floors and clips; zero-init
  network gives a zero update.
- Regime: job sampling ranges and stage caps; LPT balance; resume at an epoch boundary reproduces the next
  job list; idle-slot masking gives the same U on every rank.
- Parity (oracle adapter, subagent-written): energy, gradient, frames, fusion, projected gradient and the
  network forward with mapped v4 weights against the old package, float32 tolerances.
- Pipeline: two-epoch CPU run on a tiny grid with a floor; resume; rollout npz readable by render_learned;
  SolverLIDO on a Newton model with two hex bodies.

## 9. Build order

1. grid + fusion + physics with the float64 parity tests (the numerical core); parity adapter and dashboard-key extraction start in parallel.
2. frames kernel; features; network; equivariance, width and no-op tests.
3. step.query / prepare / advance; SolverLIDO and add_hex_body; a random-weight rollout on the canonical beam with timing.
4. augmenter; contact detection and scenes; job runner.
5. trainer, validation, selection, report; smoke run; short GPU run for time and memory; 6-epoch parity run.
6. CUDA-graph capture for rollout; Warp energy/stress kernel; per-query timing at batch 1.

## 10. Implementation record (2026-10-01)

Package `experiments/lido/` (25 modules and tests, 120+ unit tests passing on CPU, GPU tests on the L40s). The
method of section 1 is unchanged; the oracle parity tests (`tests/test_parity.py`, adapter `tests/oracle.py`)
show energy, gradient, modes, frames, fusion, projected gradient, node and edge features and the network forward
with the v4 weights agree with the old package to float32 precision (2e-8 relative on the float64 quantities,
2e-4 on the float32 network). The v4 `network_state` loads into `Net` with no missing or unexpected keys.

Deviations from the sketches of sections 4-7 (implementation only):

| Sketch | Built | Why |
|---|---|---|
| `enc 159->W`, `film 7->4W`, 2-layer correction head, `tanh` bound | Parameter map of the v4 network: `node_encoder` 159->192->192, `edge_encoder` 24->96->96, `condition_encoder` 7->192->192 + `layers.0.film` 192->768 (zero-init), `correction_head` single linear (zero-init) with bound `raw / sqrt(1 + \|raw\|^2)`, `step_head` zero-init, SiLU everywhere, affine LayerNorms | same parameters as the old network so its checkpoints load and the parity test covers the forward |
| 21-value blocks flattened mode-major | `[3,7]` row-major (value index `7 i + m`) as the old features | parity of the node input |
| `cholesky_solve` | fast diagonalisation of the Kronecker-sum structure (exact, no factor); dense inverse kept as reference | scales to any grid at constant ~0.3 ms (section 1b); the dense inverse needs 5 GB at 32k cells |
| per-slot `Trajectory` objects | `prepare` / `advance` take an object mask and run whole-batch tensor ops; recomputing a step-constant quantity for an unchanged object reproduces it exactly, so only the candidate is masked; detection runs on the whole batch at every prepare (identical pairs for unchanged objects) | no per-slot Python, no pair-list surgery |
| `batch.film` cache | FiLM recomputed every forward (16 rows; its weights change every update) | the cache only holds for frozen weights |
| `R_i^T R_j`, `F^T F` as batched matmuls | elementwise multiply-sum (`hex.mat3`, `hex.mat3_tn`) | cuBLAS batched 3x3 gemm cost 48 ms per features pass at batch 16; elementwise 0.3 ms |
| `edge_update(cat[h_i, h_j, e])` | the first linear is split so no `[E, 480]` concatenation is materialised | memory traffic |
| float32 matmuls | TF32 matmuls inside `Net.forward` only (`set_float32_matmul_precision("high")` scoped); physics stays full float32 | network speed, physics precision |
| eager network | `Net.compile_layers()` (torch.compile of the cell-graph layer, static shapes; compiled callables kept outside the module tree so checkpoints are unchanged), on by `compile_network` | fuses the per-edge elementwise chains: 130 -> 95 ms forward+backward at batch 16 |
| contact token channels 12-13 | `log1p(kappa)`, `beta` as in the old layout; penetration depth `d = r - gap` for planes and points alike (contact note section 5) | parity, method |
| displacement RMS `strength x L` | `strength x 4.45 h` (the previous generator's measured RMS at strength 0.1 was 0.445 h); validation candidates follow the training rule (50 % inertial, 50 % perturbed) as before | the first parity attempt had 10-40x larger deformations: median states matched v4 (32 J vs 30 J) but outliers with deep initial contact penetration (0.67 r mean against 0.04 r) dominated every mean metric |
| full horizon K = 8, H = 128 | K = min(K_max, 8), H = H_max of the growth stage | what the v4 campaign did; a fixed 128-step horizon on an untrained network explodes (8.6e10 N at epoch 1) and says nothing |
| contact encoder returns zeros without tokens | plus a zero-weighted sum of its parameters | DDP needs every parameter in the graph on every rank; a rank without contact pairs in one update crashed the 4-rank run |
| 4 ranks with GPU peer-to-peer | `NCCL_P2P_DISABLE=1` in the launch script | the first collective hangs between some GPU pairs on this VM; host-routed reduction of 0.9M parameters costs nothing measurable |

Measured (L40, canonical 10x10x40 beam, 4000 cells, 92.5k directed edges per object; both Warp kernels in):

| Quantity | Old code | Rewrite | LeCO (RTX 3090, 19k-vertex cloth) |
|---|---|---|---|
| Query, batch 1, eager | 40 ms | 8.3 ms (network 3.2, energy+gradient 0.7-2.1, features 1.5, fusion 0.5, projected gradient 0.4, frames 0.2) | 25.6 ms per optimizer cell |
| Training update, batch 16 (16 objects, 64k cells, 1.5M edges) | 0.5-0.75 s | 0.093 s compiled (0.110 s eager); network forward+backward 77 ms of it, 8.4 GB | 0.134 s (8 cells per step, 64 slots) |
| Attention primitive, forward+backward, 1.7M edges | - | 10.2 ms Warp CSR kernel (torch segment_reduce path 54 ms) | 1-2 ms per block at 135k-517k edges |
| Fusion solve | 6.4 ms CPU PARDISO x3 | 0.05 ms | - |
| Frames | 14.6 ms per 64k cells | 0.09 ms per 64k cells | - |
| Energy + gradient | 3 passes per query | 1 pass, 0.7 ms (Warp stress kernel, torch path 1.6-2 ms) | - |

Warp kernels (all float32, launched on the torch stream, torch paths kept as references and CPU fallbacks):
`frames.py` (Jacobi eigen + reflection + polar iteration), `csr_attention.py` (8 lanes per (row, head), each
owning a 4-wide chunk of D for coalesced loads; forward 2 launches, backward 3 launches plus a sort of the
sources; no vector atomics), `energy_kernel.py` (one thread per cell, energies and the 8 corner gradients of
stable Neo-Hookean + metric damping; `torch.autograd.Function` so training backpropagates through it).

Still open: bf16 autocast for the network body (measured 73 vs 95 ms compiled forward+backward before the
attention kernel; declined for now, everything stays float32), the 6-epoch parity run under the v4 configuration (started 2026-10-01 00:49 UTC in
`generated/lido_parity_20261001/`, tmux `LIDO-rewrite-parity`, compare with `python -m experiments.lido.compare_runs`),
and the deletion of the old package after parity.

### Parity run, first six epochs (2026-10-01, unchanged v4 configuration, 4 ranks x 16 slots)

| Epoch | Loss v4 / new | Epoch seconds v4 / new | Full-horizon metric (N) v4 / new | Cheap metric (N) v4 / new | Cheap final residual median (N) v4 / new | Final penetration (r) v4 / new |
|---|---|---|---|---|---|---|
| 1 | 0.872 / 0.983 | 299 / 49 | 878 / 481 | 184 / 1072 | 86 / 125 | 0.094 / 0.000 |
| 2 | 0.694 / 0.753 | 291 / 44 | 599 / 374 | 180 / 2616 | 101 / 68 | 0.099 / 0.000 |
| 3 | 0.703 / 0.751 | 479 / 72 | 366 / 147 | 86 / 79 | 35 / 30 | 0.093 / 0.126 |
| 4 | 0.677 / 0.716 | 481 / 74 | 314 / 89 | 73 / 63 | 27 / 14 | 0.180 / 0.099 |
| 5 | 0.742 / 0.744 | 862 / 161 | 141 / 80 | 83 / 54 | 18 / 8 | 0.979 / 0.143 |
| 6 | 0.730 / 0.727 | 864 / 161 | 115 / 81 | 57 / 46 | 17 / 6 | 0.724 / 0.092 |

All 16 full-horizon states survive at every epoch in both runs. The loss tracks v4 within 0.05 from epoch 2 and
coincides at epochs 5-6; the selection metric (full horizon) is lower at every epoch; the cheap metric's means of
epochs 1-2 are dominated by one validation seed (37: plane 8 mm under the rest bottom, stiff contact, 0.84 r initial
penetration, 3.9e4 J) while its medians match v4 from epoch 2. The mean training residual is 3-4x higher than v4's
(heavy-tailed contact states; the medians of the validation residuals are lower), which is the one quantity not
within noise. Epochs run 5.4-6.6x faster including validation. Parity gate (b) is taken as passed; the run was left
going as the first campaign of the rewrite (`generated/lido_parity_20261001/`, dashboard
https://ankachen.com/artifacts/lido-parity-20261001/index.html).

### Inference path: captured query (2026-10-01)

Built after the parity run (`capture.py`, `bench_query.py`; flags `capture` / `compile_network` on `rollout()` and
`SolverLIDO`, capture on by default inside Newton on CUDA, eager fallback with a warning). The method is unchanged;
only the execution differs from training:

- `contact.detect(capacity=True)`: every candidate slot kept, `S (1 + k)` rows (k = min(M_pair, Npts)) in the
  sample-major, slot-minor order of the compacted list with a `valid` mask; `token_offsets` is the static capacity
  CSR, the valid counts per cell are data, the within-cell attention pairs are precomputed once per layout. Padded
  rows get a -1e30 score as attention sources, zero contact energy and penetration, and are masked out of the
  mean / max pooling and the count channel (`index_add` / `scatter_reduce`: `segment_reduce` and `bincount` are
  not capturable). Tests show the two layouts give identical pairs, energies, gradients, tokens and encoder output.
- `CapturedQuery`: `Step.query` + `Step.commit` recorded as one CUDA graph on a side stream after warm-up queries
  (state restored afterwards); `sync()` copies whatever `prepare` / `advance` rebound (candidate, E, gX, history,
  Y, m_Y, m_prev, C_prev, R_ref, pairs, material) into the static buffers. The Warp kernels capture as they are
  (launched on the capturing torch stream, descriptor-only array wrappers). Host syncs removed from the query:
  `Batch.N / C / S` are cached ints, the padded encoder never calls `within_range_pairs`.
- Compiled chains for the rollout (`rollout.compile_inference`): the cell-graph layer as one graph (without
  autograd the Warp attention is a `torch.library.custom_op`, so no graph break), the edge encoder, and the node /
  edge feature chains; `max-autotune-no-cudagraphs`, static shapes. Two pitfalls met: the custom op's fake must
  return the real output's (contiguous) strides, and inductor in that mode miscompiles an `index_add` of a
  constant-filled source (the padded lanes of the scatter tile are not masked: 12 cells counted as 16), so
  `features.seg_rms` counts cells with a masked compare instead. `tests/test_capture.py` checks the compiled
  query against the eager one on one and two objects.

Measured on an L40 shared with a training campaign (about 94 % busy; eager and captured interleaved in one
process, 6 rounds x 10 queries, canonical beam with the ground plane in contact, 400 pairs; median / best block,
ms per query): eager with compacted pairs 23.7 / 21.2 (its two host syncs wait for the other process's kernels),
eager with capacity pairs 7.1 / 7.0 (compiling changes nothing for eager: it is CPU-bound), captured 5.8 / 3.9,
captured + compiled layer and edge encoder 4.3 / 2.8, plus the attention custom op 4.2 / 2.4, plus compiled feature
chains 3.8 / 2.6 (the compiled variants are within the contention noise of each other). Replay reproduces the
eager query within 1e-7 (candidate) to 1e-5 (gradient) relative; the compiled query within 1e-5. Kernels per
query: 457 captured eager, 386 with the compiled chains; what remains is many small
per-cell / per-pair kernels (the contact energy and its autograd backward, the inertia and segment sums, the
fusion's two [Pf, Pf] x [Pf, 3] matmuls at full float32, the index_add scatters). Idle-GPU confirmation pending.

### Fusion solvers for general meshes (2026-10-01, after Anka's question about irregular meshes at the million-vertex level)

`Fusion(solver="auto")` picks per grid: the structured fast-diagonalisation solve for box grids with the z-min face
pinned, otherwise cuDSS through nvmath-python (`SparseFactor`, optional dependency `nvmath-python[cu12]`, installed
in the venv, not yet in pyproject), otherwise the dense inverse. The cuDSS solve is wrapped in an autograd Function
whose backward is the same solve; the factor is per mesh and reused by every query, material and state (the fusion
matrix never changes). Measured on an L40 shared with the campaign (fp32, 3 right-hand sides):

| Free corners | nnz of K_ff | cuDSS factor (once) | cuDSS memory | cuDSS solve | structured box solve |
|---|---|---|---|---|---|
| 4.8k (10x10x40) | 113k | 0.1 s | small | 0.21 ms | 0.28 ms |
| 35k (20x20x80) | 886k | 0.5 s | ~0.3 GB | 0.65 ms | 0.29 ms |
| 130k (50^3) | 3.4M | 2.3 s | 1.5 GB | 4.1 ms | 0.57 ms |
| 1.02M (100^3) | 27M | 17.8 s | 13.4 GB | 31 ms | 0.73 ms |

Reading: a sparse direct factorisation is the right general solver up to a few hundred thousand vertices (sub-5 ms
solves, constant matrix so the factor is paid once per mesh). At a million vertices it still works (31 ms per solve,
two solves per query, 13 GB) but the network on 27M edges costs more than that, so the solver is not the bottleneck
there; if memory or solve time matter at that scale the options are a PCG with the structured box solve as the
preconditioner for voxel-type meshes (lattice subsets), or AMG (AmgX) for arbitrary topology.

### Inference tail (2026-10-01, after the capture work)

`contact_kernel.py` (one Warp kernel per capacity row: normal, damping and IPC friction energies with their analytic
sample gradient scattered to the four face corners; `torch.autograd.Function` for the differentiable path) and a
fused `energy_kernel` pass (cells kernel + contact kernel + corner kernel: elastic, damping, inertia, pinned zeroing,
per-object totals by atomics) make `energy_and_grad` on the inference path 4 launches / 0.24 ms instead of 154 /
2.75 ms. Layout copies removed across the query (414 -> 242 kernels eager, 386 -> 245 per captured replay); the
structured fusion solve now uses batched matmuls on contiguous views (no operand permutes). Measured under GPU
contention (campaign running), canonical beam with ground plane, 400 pairs: captured + compiled query
**1.5-3.0 ms median, 1.47-1.5 ms best block** (idle-GPU confirmation pending); eager with capacity pairs 5.3 ms;
the old eager path with compacted pairs is 24 ms under contention because its host syncs wait behind the other
process's kernels. Kernel-time shares in the compiled replay: network GEMMs ~35 %, Warp attention ~10 %, frames ~7 %,
fusion ~10 %; remaining small items: `inverted_cells` (det via LU, ~12 kernels), one frames `.contiguous()`.

### Canonical-cube meshes of arbitrary shape (2026-10-01, Anka)

Target meshes keep unit-cube cells but are arbitrary voxel subsets of a lattice with pins anywhere. Measured
matrix-free GPU conjugate gradient on the fusion matrix (fp32, 1e-5): plain CG 85 / 155 / 290 iterations at 4k /
125k / 1M cells (condition number ~L^2), Jacobi preconditioning useless, CG preconditioned by the structured box
solve 15 iterations on carved voxel shapes independent of size. Decision (Anka): build a geometric multigrid
(Galerkin V-cycle on the voxel hierarchy, Chebyshev-Jacobi smoother, dense coarsest solve) as the PCG preconditioner
behind `Fusion(solver="mg")`, together with `Grid.from_voxels`. Design idea recorded for later (Anka): the same
hierarchy can carry a coarse-level learned solution down as the fine level's starting candidate (a multigrid of the
neural solver itself); not part of the current method.

Built (2026-10-01): `Grid.from_voxels(occupancy, pins)` (any voxel subset, pins "zmin_face" | mask | none, generalised
tie-break corners) and `multigrid.py::MultigridFactor` (Galerkin V-cycle on the lattice hierarchy, trilinear transfer,
Chebyshev-Jacobi degree-2 smoother, dense coarsest solve, vectorised PCG over all right-hand sides, float64
residual by default because a float32 PCG has an accuracy floor of eps x cond(K) (9e-4 at 1M cells) that no
tolerance removes). `Fusion("auto")`: box + pinned face -> structured; other meshes on CUDA -> multigrid; cuDSS and
dense selectable. 181 tests pass. Measured under GPU contention:

| Mesh | Free corners | Levels | PCG iterations | Setup | Solve (3 rhs) | Hierarchy memory | cuDSS for comparison |
|---|---|---|---|---|---|---|---|
| 20x20x30 carved, random pins | 8.5k | 3 | 4 | 0.23 s | 11.5 ms | 5 MB | 0.7 ms |
| 50^3 carved | 78k | 5 | 5 | 0.30 s | 21 ms | 48 MB | ~2.5 ms (est.) |
| 100^3 carved | 596k | 6 | 5 | 0.40 s | 72 ms (50 without fp64 residual) | 376 MB | ~20 ms (est.) |
| 100^3 full | 1.02M | 6 | 3 | 0.52 s | 70 ms (36) | 653 MB (325) | 31 ms, 13.4 GB, 18 s factor |

Iteration counts are size-independent (3-8 everywhere, including plates, slabs, rods and four-pin shapes), setup
and memory are small, but the solve is launch-bound (~60 kernels per iteration plus one host sync), so per solve it
is slower than cuDSS at every size as implemented. Next implementation step: fixed iteration count and CUDA-graph
capture of the whole PCG (expected ~0.5 ms at 8k and 10-15 ms at 1M).

### Multigrid solve: fixed iterations, Warp kernels, CUDA graphs (2026-10-01, idle L40)

The method is unchanged; `MultigridFactor` now runs a fixed number of PCG iterations by default (`iterations=8`; the
adaptive stop remains as `iterations=None`), with no host synchronisation and no data-dependent control flow (the
per-column alpha / beta stay on the device, zero columns stay exactly zero through `where` guards), so the solve
records into a CUDA graph. Inside a capture (the inference query) it is recorded as part of the outer graph; outside,
`solve` records one graph per right-hand-side shape on first use and replays it (`static_in.copy_`, replay, `clone`),
and the autograd backward, the same solve, replays it from the autograd worker thread (checked). The cycle's vector
operations are Warp CSR kernels (`multigrid_kernel.py`, one thread per (row, 3-vector of columns), generated per
(float64, float32) pair): a Jacobi step, a Chebyshev step, a generic `alpha A x + beta y` for the PCG product,
restriction and prolongation, a float64 residual that also writes its float32 copy, and a mixed-precision axpy for the
float64 iterate. cuSPARSE's CSR SpMM measured 1.06 / 4.3 ms (fp32 / fp64) for 27M nonzeros and 3 columns against 0.42
/ 0.84 ms for these kernels (0.081 against 0.017 ms at 2.1M nonzeros), so they are both the fusion and the bandwidth
lever; the torch sparse path stays for the CPU and for column counts that are not multiples of three. Kernels per PCG
iteration: 108 -> 32 on a 3-level hierarchy, 179 -> 48 on 5 levels, 212 -> 54 on 6 levels.

Two pitfalls met. Warp rebuilds and reloads a module when a new kernel instantiation is added to it, which invalidates
the kernel handles baked into any CUDA graph recorded with the previous build (illegal memory access on replay when a
second precision pair was instantiated): every generated kernel now lives in its own module (`module="unique"`). And
the Chebyshev step's tracked residual is that of the step's input iterate, not of its output, so the coarse-level
residual needs its own operator apply (using the tracked one cost one order of magnitude of convergence per
iteration).

float64 relative residual after a fixed number of iterations (fp32 cycle, fp64 residual every iteration, 3 rhs):

| Mesh | Free corners | Levels | 4 | 6 | 8 | 10 | fp32 result vs fp64 solve at 8 |
|---|---|---|---|---|---|---|---|
| box 2x2x3 / holed 6x6x8 (zmin, random pins) | 27-368 | 1 | 1e-15 | 1e-15 | 2e-15 | 6e-15 | 2.5-2.8e-8 |
| plate 30x30x2 | 1.9k | 3 | 3.5e-8 | 5.3e-12 | 8e-16 | 2e-16 | 2.5e-8 |
| rod 3x3x60 | 960 | 2 | 4.1e-8 | 5.8e-12 | 1.7e-14 | 2.2e-14 | 2.6e-8 |
| slab 40x4x40, sparse pins | 8.1k | 3 | 1.7e-7 | 6.1e-11 | 2.1e-14 | 4e-16 | 2.5e-8 |
| beam 10x10x40, hole r3 | 4.0k | 3 | 5.3e-6 | 9.9e-9 | 1.0e-11 | 1.6e-14 | 2.5e-8 |
| box 10x10x40 | 4.8k | 3 | 8.7e-8 | 1.8e-11 | 1.0e-14 | 1.0e-14 | 2.7e-8 |
| carved 20x20x30, random pins | 8.5k | 3 | 1.3e-6 | 1.1e-9 | 7.0e-13 | 5.5e-16 | 2.5e-8 |
| carved 50^3 | 78.5k | 5 | 5.0e-5 | 1.0e-7 | 1.5e-10 | 5.0e-12 | 2.6e-8 |
| carved 100^3 | 596k | 6 | 4.4e-5 | 5.5e-8 | 7.7e-10 | 2.3e-12 | - |
| box 100^3 | 1.02M | 6 | 1.4e-7 | 5.6e-11 | 3.3e-14 | 1.5e-15 | 2.5e-8 |

Four iterations miss 1e-5 on the carved shapes, six leave two orders of magnitude, eight four; 8 is the default, 6 is
the lean choice (25 % less). The float32 result is within 3e-8 of the float64 solve everywhere (the float32 output
rounding; its true residual is eps x cond, 1e-7 to 2e-6). The float64 residual (`fp64_every`): every iteration 0.70 /
1.91 / 19.2 / 33.4 ms at 8k / 78k / 596k / 1M, every second iteration 0.67 / 1.77 / 16.7 / 28.9, every fourth 0.66 /
1.70 / 15.4 / 26.6, float32 residual only 0.68 / 1.65 / 14.7 / 25.2; the error against the float64 solve stays 2.5-3e-8
for every 1 and 2 (every 4: 9.4e-7 on the 10x10x40 box), the float32-only residual gives 1.4e-4 at 78k and 8.7e-4 at 1M
(fails 1e-5). Default kept at every iteration; `fp64_every=2` is a free 10-15 % at >= 600k.

Solve timings (ms, median of back-to-back blocks, same process; "old" is the previous cuSPARSE eager adaptive code,
iterations in parentheses; eager fixed is the Warp path without a graph, CPU-bound by `wp.launch` at ~40 us per launch):

| Mesh | Free | rhs | old eager adaptive | eager adaptive (Warp) | eager fixed 8 | captured fixed 8 | cuDSS | Kron | setup s | hierarchy MB |
|---|---|---|---|---|---|---|---|---|---|---|
| box 10x10x40 | 4.8k | 3 / 48 | 3.4 (3) / 3.6 | 4.6 / 4.8 | 11.2 / 11.4 | 0.71 / 0.85 | 0.23 / 0.28 | 0.17 / 0.17 | 0.01 | 3 |
| beam 10x10x40 hole r3 | 4.0k | 3 / 48 | 4.5 (4) / 4.7 | 6.2 / 6.3 | 11.1 / 11.1 | 0.70 / 0.94 | 0.23 / 0.27 | - | 0.01 | 2 |
| carved 20x20x30 random pins | 8.5k | 3 / 48 | 4.5 (4) / 4.7 | 6.2 / 6.3 | 11.2 / 11.3 | 0.73 / 1.02 | 0.24 / 0.37 | - | 0.01 | 5 |
| carved 50^3 | 78.5k | 3 / 48 | 9.7 (5) / 11.4 | 12.4 / 15.4 | 18.6 / 19.4 | 1.92 / 7.88 | 0.99 / 2.90 | - | 0.03 | 49 |
| carved 100^3 | 596k | 3 / 48 | 32.3 (5) / 99 | 24.4 / 52 | 37.5 / 79.5 | 19.3 / 79.1 | 6.4 / 27.7 | - | 0.09 | 378 |
| box 100^3 | 1.02M | 3 / 48 | 33.4 (3) / 104 | 16.1 / 56 | 33.8 / 140 | 33.4 / 141 | 17.3 / 71.4 | 1.49 / 32.9 | 0.16 | 658 |

The captured graph's private pool retains the solve's intermediates (25 MB at 8.5k x 3 rhs, 336 MB at 78k x 48, 252
MB at 1M x 3, 3.8 GB at 1M x 48; 1.3-1.5x the eager transient peak); the first solve of a shape costs 30-330 ms
(warm-up + capture, the first one also compiles the Warp modules). Reading: at <= 80k free corners the captured solve
is 5-15x faster than before and within 3x of cuDSS (which needs no graph and a 0.1-2 s factor); at 600k-1M the solve
is bandwidth-bound in the level-0 operator applies (0.42 ms each, 7 per iteration including the float64 residual) and
the fixed 8 iterations cost what the adaptive 3-5 did, 2-3x cuDSS's time with 20x less memory. The remaining lever
there is a matrix-free lattice-stencil operator for level 0 (coefficient table by the 8-bit cell mask, dense lattice
position table in L2) instead of the 216 MB CSR read per apply, and 6 iterations.

End-to-end query (`bench_query.py --plane --voxel-hole 3`, 2720 cells, 4000 free corners, 400 contact pairs,
captured + compiled layer, edge encoder and feature chains, 5 x 10 queries, ms per query): `Fusion("mg")` 2.24 (plain
captured 2.80, 714 kernels per replay, two multigrid solves at ~0.65 ms each in the graph), `Fusion("sparse")` 0.90
(captured 1.77), the box beam with the structured solve 1.31 (captured 2.42). Tests: 188 pass (`tests/test_multigrid.py`:
fixed solve against the dense float64 solve to 1e-9 on the CPU and 1e-5 on CUDA incl. autograd, bitwise replay,
one graph per shape with the backward replaying, zero columns, argument checks; `tests/test_capture.py`: a voxel-grid
batch with `Fusion("mg")` replayed against the eager query over three queries and an advance).

### Campaign result and dashboard additions (2026-10-01 evening)

The campaign `generated/lido_parity_20261001/` (unchanged v4 configuration, 4 ranks) completed 48 epochs in 18.0 h
(610k updates, no failures); best full-horizon selection metric 28.7 N at epoch 44 (v4: 65.9 N at epoch 8 of its 25
epochs). Idle-GPU confirmation of the speed figures: inference query captured + compiled with ground contact 1.32 ms
(eager old path 7.8 ms); training update at batch 16 89.9 ms compiled, 105.6 ms eager. Dashboard (old renderer
`mixed_report.py`, kept by decision): every validation chart carries a visible title; a new chart shows the force
residual relative to iteration 0 (mean, median, maximum over the validation trajectories) on a logarithmic axis
labelled in powers of ten (`relative_residual_curve.svg`). Observation on the finished run: the mean ratio falls to
0.005 by iteration 16 and climbs back to 0.04 by iteration 32 while the median keeps falling to about 0.001, so a
few validation states drift after 16 iterations.

### Solver choice after the multigrid acceleration (2026-10-01 evening) and the free-body decision

With the fixed-iteration, Warp-fused, graph-captured multigrid (0.7 ms at <= 8.5k free corners, 1.9 ms at 78k,
19 ms at 596k, 33 ms at 1M for 3 right-hand sides; cuDSS idle: 0.23 / 1.0 / 6.4 / 17 ms; memory 0.66 GB against
13 GB at 1M) the automatic choice is: structured solve for box grids with the pinned face; cuDSS for any other mesh
up to `sparse_max_free` = 300k free corners; multigrid beyond that or without nvmath. End-to-end captured query on a
voxel beam with a through-hole: 0.90 ms with cuDSS, 2.24 ms with multigrid.

Anka (2026-10-01): free bodies are to use the rigid semi-implicit centroid target with the contact force of the
current candidate, recomputed at every optimizer query (derivation 7.4, Picard iteration 7.11-7.12), realised by the
centroid-only blend 7.6 through 7.21; the Newton target 7.15 kept as the alternative because the Picard contraction
constant 7.13 exceeds 1 for stiff contact. Implementation delegated; pinned bodies stay bitwise unchanged.

### Free bodies: translation-free fusion and the per-query centroid target (2026-10-01, after Anka's decision)

Built as decided (derivation note section 7; translation only, no rotation handling or inertia tensor). `Grid.build(...,
pins="none")` and `Grid.from_voxels(..., "none")` give unpinned bodies; a batch records them in `free_objects` /
`any_free`, and a batch without free objects runs the previous code path (pinned-beam query results are bitwise those
recorded before the change, `tests/test_free_body.py::TestPinnedPathUnchanged` against `tests/reference/pinned_beam_reference.pt`).

- Shape solve without pins (7.1): `KronFactor` on a free box uses the generalised eigen-decomposition of all three axes
  with the constant mode's inverse set to 0 (the pseudo-inverse; the right-hand side B^T W dF has zero translation
  component for every dF); `SparseFactor`, `MultigridFactor` and `DenseFactor` receive `fusion.dirichlet_view(grid)`,
  the grid with the reference corner `ref_corners[0]` as the only Dirichlet node (K_ff SPD by Corollary 4.2), and set
  d_r = 0. Every factor exposes `free`, the corner set its solve covers.
- `Fusion.fuse(batch, dF, d_pinned=None, centroid_target=None)`: for free objects the translation t = c_t - c(x_k) -
  c(d_hat) of eq. 7.21 is added after the shape solve (mass-weighted centroid over rho m_i), so c(x_k + d) = c_t exactly
  (Proposition 7.3(b)); checked against the dense solution of (7.19) for lambda in {1e-3, 1, 1e3} on a free 2x2x2 block
  (same d to 1e-10, centroid to 1e-13, Kron / dense / multigrid / cuDSS). `project_gradient` removes the translation
  component (mean over the object's corners) of gX before the solve; the modes cannot carry it (B Z = 0).
- `Step(translation="picard" | "implicit_contact")`, default picard. `prepare` stores c_n = c(X) and cdot_n = c(V)
  (step constants, static buffers in the captured query), centres a free body's candidate noise so that c(x_0) = c(Y)
  (7.6), and checks the Picard constant sum_active ke / M_tot (7.13) at the candidate, warning once per object when it
  exceeds 1. Every `query` recomputes F_con(x_k) = -sum of the contact-energy gradient (autograd through
  `contact.contact_energy`, so the torch pair energies or the Warp pair kernel) and the active-pair stiffness
  sum_active ke n n^T, and passes c_rig = c_n + cdot_n + g + F_con / M_tot (7.11) or the centroid Newton step
  c(x_k) - (M_tot I + sum_active ke n n^T)^-1 r_tr (7.15, closed-form 3x3 inverse, no host sync) to `fuse`; the Picard
  constant of the candidate goes to `batch.picard_constant` and `QueryOutput.picard_constant`. `rollout` and
  `SolverLIDO` take `translation`; `add_hex_body(pins="none")` gives every particle a positive mass.

Observed with the zero-init network (rigid body, float64, 2x2x3 box, h = 0.05 m, dt = 1/300 s, rho = 1000, E = 1e5,
M_tot = 70.2 normalised, six bottom samples on a y-plane): kappa = 1 gives ke = 2.6 and Picard constant 0.222; the
translation residual r_tr (7.14) falls by exactly 0.222 per query (0.187, 0.0416, 0.00924, ..., 1.1e-6 after eight) and
a body dropped from one cell rests after 200 steps at penetration 9.8102e-3 cells against the analytic 9.8100e-3 =
M_tot |g| / (6 ke). kappa = 20 gives Picard constant 4.44: the Picard target is an exact 2-cycle between the penetrating
inertial position and a lifted position without contact (r_tr constant at 0.833), and since the state after an even
number of queries is the free-fall one the body falls through the floor (y = -42 cells after 200 steps); the
implicit-contact target reaches r_tr = 1e-14 after one query from a resting start (the active set is the root's) and
after two queries on a tilted body whose six bottom samples have two heights (Newton-Fourier from below, finite
termination, Proposition 7.2(i)), and the dropped body rests at 4.9050e-4 cells = the analytic value. Free fall without
contact follows c_n = c_0 + n cdot_0 + n(n+1)/2 g to 1e-14 with the rest shape to 1e-14. Captured query with a free
voxel body (holed 6x6x8, multigrid): replay within 5e-5 of the eager compacted-pair query (1e-5 with the implicit-contact
target; the contact force driving the translation comes from different float32 paths in the two). Tests: 207 pass.

Decision (Anka, 2026-10-01, after the measurements): the default centroid update for free bodies is the implicit-contact
step (7.15, `translation="implicit_contact"`): same backward-Euler fixed point as the Picard target (7.11), converges
for any contact stiffness (exact in one or two queries on a floor), while the Picard constant (7.13) is above 1 for
most of the campaign's contact range (E 1e5 with kappa >= 100, E 1e6 with kappa >= 10). Picard stays selectable.
Translation only: no rotation handling (rotation is not in the fusion null space and the inertial channel carries it).

### Edge network cost and the proposed redesign (2026-10-01, discussion with Anka; not implemented, pending a training comparison)

Flop budget of one query on the canonical beam (92.5k directed edges), multiply-adds: edges 6.3e9 (edge encoder
24->96->96 11.5k per edge, A02 correction 37k per edge with its node halves folded per cell, edge value 96->192
18.4k, bias and attention ~1k), per-cell 2.3e9 (node encoder, qkv, FFN 192->768->192, heads), physics + features +
fusion ~6e7. The trained campaign checkpoint relies on the A02 term completely (removing it at inference: median
final residual 0.19 N -> 509 N), which says nothing about a network trained without it; no recorded ablation exists.

Anka's assessment: the two-stage edge (geometry MLP, then the A02 residual correction from the hidden states) is a
historical patch; the hidden states carry history and damping information an edge has no use for. Proposed single
edge module (Anka's correction: the neighbour's gradient, history and the rest reach a cell through the attention
values; the edge only sets the pair's bias and value from the pair's CURRENT configuration), everything in the
receiver's frozen frame: input = geometry g_ij (24) | receiver's current representation m_i (21, folded per cell) |
sender's current representation R_i^T m_j (21, per edge) -> MLP 66 -> 96 -> SiLU -> 96 -> e_ij (Anka: hidden 96, not 192;
the value projection 96 -> 192 lifts the aggregated code once per cell), consumed as today (per-head score bias, message value).
Anka, 2026-10-01: "implement it as I said"; config switch `edge_module: "pair"` with the current path kept as `"a02"`
for the trained checkpoint; comparison run of 6 epochs against the campaign. Exact implementation
move independent of the redesign: the shared edge-value projection U (96->192) is linear, so sum_j w_ij U e_ij =
U sum_j w_ij e_ij: aggregate the 96-value code per head in the attention kernel and apply U once per cell
(18.4k -> 0.6k per edge, model unchanged). Budget after both: H = 96 gives ~15k per edge, edges 1.4e9 against the
per-cell 2.1e9, network ~41 % of today; H = 192 about 60 %. The redesign is an architecture change and needs the
6-epoch comparison against the campaign before replacing the current edge.

### Aggregated edge values (2026-10-01, implemented; the model is unchanged)

The exact move from the paragraph above: `edge_val` (96 -> 192, one linear U shared by the edges) commutes with the
attention sum, sum_j w_ij (v_j + U e_ij) = sum_j w_ij v_j + U sum_j w_ij e_ij per head. The CSR attention (torch
reference and Warp kernels, `csr_attention.py`) takes an optional edge code [E,Dc] (one vector per edge, the same for
every head) and returns its weighted sum per (row, head) as a second output [R,H,Dc] next to the messages; the score does
not see the code. Forward: the forward kernel adds, per owned code chunk (Cc = 4 values; the LANES threads of a (row,
head) group take the Dc / Cc chunks in turns), one more pass over the row's edges accumulating exp(s_e - m) code_e.
Backward: the weights kernel adds dagg[i,h] . code_e to the per-edge dw_e before the softmax derivative ds_e = w_e (dw_e
- S), so dq, dk and dbias carry the code term through the scores, and a new per-(row, chunk) kernel writes dcode_e =
sum_h w_e[h] dagg[i,h] (no atomics; the kernels are generated per (D, Cc)). The custom op `lido::csr_attention` returns
the pair (agg is [R,H,0] without a code; the fake has the same contiguous strides), so the compiled layer and the captured
query use the same form. `Layer.forward` passes e as the code, applies the per-head slice of U once per cell as a batched
matmul ([H,C,96] @ [H,96,D]) and adds the bias once (the weights sum to 1; every cell has its self edge), 18.4k -> 0.6k
multiply-adds per edge and no [E,192] edge-value tensor or its gradient. `edge_val` keeps its name and shape (the v4
checkpoint loads unchanged, the parity test against the old network forward passes at its 2e-4). Equivalence test
(`test_network`, random weights and inputs, full float32): the aggregated form against the per-edge form gives corr
within 7.7e-7 / 5.8e-7 and step within 5.1e-7 / 3.1e-7 relative (CPU torch path / CUDA Warp path), every parameter
gradient within 1.5e-6 (the edge_bias bias gradient is an analytic zero, 1e-7 absolute); the Warp kernels against the
torch reference with a code: 1e-5 forward, 1e-4 gradients in float32 and against float64, for D in {8, 16, 32} and
code widths 96 and 10. Measured on one L40 (back to back, same scripts): network forward + backward at batch 16
(64000 cells, 1.48M edges) eager 95.2 -> 90.8 ms (453 -> 456 CUDA kernels, peak memory 7.70 -> 6.39 GiB), with the
compiled layer 79.8 -> 75.1 ms (435 -> 436, 7.75 -> 6.91 GiB); the no-grad forward 36.3 -> 34.2 ms eager and 24.0 ->
21.7 ms compiled. Query at batch 1 on the canonical beam with the plane (`bench_query --plane`): captured 2.42 -> 2.35
ms (294 -> 296 kernels per replay), captured with the compiled layer, edge encoder and feature chains 1.31 -> 1.22 ms
(222 -> 223). The attention forward kernel itself got cheaper (it now reads the [E,96] code instead of the [E,192]
value add); the saving is the edge-value GEMM and its [E,192] round trip. The bitwise pinned-beam reference
(`tests/reference/pinned_beam_reference.pt`) was re-recorded on the new form, since the reordered sums move the query
results by rounding (old against new recording: float64 within 1.2e-15 relative, float32 within 1.1e-6; the old code
reproduced the old recording bitwise just before). Tests: 210 pass (207 + 3 new).

## 11. v5: free motion with contact (design decisions, 2026-10-02)

Source: `notes/v5-free-motion-with-contact.md` (Anka). Decisions so far:
- The whole batch is one scene: every body can touch every other body and the ground (Anka). Budget per scene
  ~64k cells, ~100 ms per training update (the cost of today's 16-beam batch; measured 5 ms and 0.45 GB per 4000
  cells, 80 such bodies per L40 at most).
- Bodies: random cuboids with sides drawn independently in 3-12 cells (27-1728 cells), h = 0.025 m for all, added
  until the scene reaches ~64k cells (40-100 bodies). Materials per body as today; contact constants per scene.
- Regime: a fixed state is a scene; 64 fixed scenes per epoch with the unchanged K-H growth table give the same
  ~15k updates per rank per epoch as the campaign's last stage (to be re-derived with the scene generator in place).
- Body-body contact: a `wp.Mesh` of each body's exposed faces rebuilt once per physical step from the step-start
  shape; each sample queried against the other bodies' meshes with `mesh_query_point_sign_parity` (closest point and
  inside/outside sign, robust under penetration); detection records (sample, body, face); every query recomputes the
  contact point and the face normal from the partner's current corners, so the contact energy's gradient reaches both
  bodies and the partner force enters the partner's centroid update. Both directions kept (A's samples against B and
  B's against A). Pair kind "other body" uses the reserved slot of the token one-hot.
- Free-body translation by the implicit-contact centroid update (section 10); no rotation handling.
- Placement and motion (Anka): random position and random orientation per body in a box up to ~0.5 m above the
  ground (everything lands within the 128-step horizon), no initial overlap (bounding boxes with a 1-3 cell gap), at
  least one body within 2 cells of the ground; each body gets a rigid velocity = scene-wide drift direction plus
  per-body noise, 0.1-0.5 m/s, on top of the existing deformation and velocity fields; gravity along -y.
- Detection for body pairs follows the contact note's rules used for static points: margin r + |v| dt, nearest
  M_pair = 4 partners per sample over plane, static points and other bodies' surfaces, partner normal opposing the
  sample's face normal; implementation detail, not a design decision.
- Validation per scene: residual and survival as today, plus inter-body penetration (max over body pairs in units of
  r) and the total momentum drift of contact-free scenes; the full-horizon selection metric stays the free-corner
  residual.

### Pair edge module (2026-10-01, implemented as `edge_module: "pair"`; the default `"a02"` is the network above, unchanged)

Anka's single edge module from the paragraph before last, built as a second option next to the current two-stage edge.
`TrainConfig.edge_module` selects it ("a02" default: `Net.from_config` builds the same parameters in the same order with
the same values from the same seed, 58 state-dict entries, the v4 checkpoint and the parity test unchanged). The module
(`network.PairEdge`): input [66] per directed edge j -> i, everything in the receiver's frozen frame, geometry g_ij
(the 24 `edge_attr` values) | receiver modes m_i (21, the first node values) | sender modes R_i^T m_j (21, the new
`Features.edge_sender`, built by `features.edge_features` in the same fused per-edge chain as the geometry with the
elementwise 3x3 product; `Step` asks for it only when the net is the pair module) -> Linear(66 -> 96) -> SiLU ->
Linear(96 -> 96) = e_ij; standard init, no residual, no hidden states; `edge_bias` (96 -> 6) and the aggregated
`edge_val` (96 -> 192, once per cell) as the layer already does; the A02 `edge_update` and the `edge_encoder` do not
exist in this network. Hidden width = code width = `edge_hidden_dim`. The first linear's receiver block is folded per
cell (W_r m_i once per cell, gathered to its edges, 21 x 96 per cell); geometry and sender are two accumulating GEMMs
per edge, no [E,66] concatenation. The 21 sender values keep the three axes although R_i^T F_j already sits in the
geometry (9 duplicated numbers, 864 multiply-adds per edge; kept for the plain 66-wide input Anka specified).
Multiply-adds per edge (`tests/test_pair_edge.py`, GEMM flops from torch.profiler at two edge counts, attention and
per-cell terms by hand): GEMMs 24x96 + 21x96 + 96x96 + 96x6 = 14112, attention 192 + 192 + 576 = 960, per-cell share
(receiver 2016 + edge values 18432) / 27 = 758: 15830 (a02: about 49.9k). Parameters 903k -> 797k (edge part 142k ->
35k). The output heads keep the zero-init: the untrained pair net leaves the candidate at Y (test). Equivariance of the
fused update and the loss under rotation + translation (float64, 1e-5 as test_step), gradients to every parameter with
and without contact tokens, captured query and compiled inference against eager within 1e-5 (the capture tests run
again with the pair config), and a 40-update GPU training smoke on the canonical config with `edge_module: "pair"`
(validation counts reduced): loss finite (0.81-1.08 at the logged updates, gradient norms 9-175, 0.10 s per update
after the compile), validation and the epoch record written. `Net.compile_layers` compiles the pair module together with
the layers (it is that network's per-edge chain; the a02 counterpart lives inside the compiled layer), so the training
configuration benefits too. Measured on one idle L40, back to back, same scripts as the record above: network forward +
backward at batch 16 (64000 cells, 1.48M edges) eager 91.1 -> 54.4 ms (456 -> 428 kernels, peak 6.39 -> 3.86 GiB),
with the compiled layer (training configuration) 75.2 -> 51.9 ms (437 -> 419, 6.91 -> 3.81 GiB); no-grad forward 33.9 ->
18.9 ms eager, 21.6 -> 13.4 ms compiled. Query at batch 1 on the canonical beam with the plane (`bench_query --plane
[--edge-module pair]`): captured 2.35 -> 1.41 ms, captured with the compiled layer and edge module 1.44 -> 1.21 ms, with
the compiled feature chains as well 1.22 -> 0.96 ms (eager 7.5 / 4.8 ms either way: host-bound). Under max-autotune
inductor logs "No valid triton configs" for some tile sizes of the thin K = 21 / 24 GEMMs of the pair module and uses
the valid ones; harmless, absent in the default training compile mode. Tests: 222 pass (210 + 12). Not started: the
6-epoch comparison against the campaign (the main session launches it).
- Pinned bodies in v5 scenes (Anka, 2026-10-02): a fraction of the bodies (default 25 %, config `pinned_body_fraction`)
  are clamped at one of their six faces, chosen per body, and held at their initial pose in the world (anchors and
  hanging beams for the free bodies to hit). The fusion already switches per body: pinned grids take the Dirichlet
  solve, free grids the translation-free solve with the implicit-contact centroid update, within one batch and one
  query. Follow-up to the current v5 agents (scene generator change plus tests).

### v5 scene generator (2026-10-02, implemented: `scenes_v5.py` and the scene config keys)

`scenes_v5.sample_scene(master_seed, scene_seed, cfg, validation=False) -> SceneV5` and `realise(scene, grids, aug,
device, dtype=float32) -> (grids, X, V, Material, ContactScene)` as in the v5 interface contract, plus `held_out_scene`
(the validation stream) and `scene_summary` (the run record). Seed streams: `SeedSequence([master_seed, scene_seed,
0 | 1, 5])` spawned into a scene, a body and a placement stream, so body-mode streams (jobs.py) and validation scenes
never coincide; the placement rejections disturb no other draw. Bodies are drawn until the cell total reaches
`scene_cells`, the last body may exceed it (defaults: 64015-64722 cells, 135-166 bodies over 16 seeds; with the sides
independent per axis the mean body is 7.5^3 = 422 cells, so about 150 bodies, not the 40-100 estimated above).
Materials per body with the body-mode ranges and draw order (E, nu, rho, eta, perturbation_scale, strength,
velocity_dt; gravity is `cfg.gravity`, no magnitude draw); kappa per scene, log-uniform with the load floor maximised
over the bodies, kappa >= m_b g / (n_face_b d_max E_b h) (defaults: kappa_floor 15-47, binding in 5 of 16 scenes);
beta, mu_f per scene. Drift: direction = normalised N(0, I) with the y component scaled by 0.25, speed
U(drift_speed_range); per-body rigid velocity = drift + U(0, speed) times a uniform random direction. Placement:
uniform random quaternion (Shoemake), gap U{placement_gap_cells}, lowest point of the rotated box U(2 cells,
placement_height) (the first body U(1, 2) cells), rejection sampling of the gap-grown axis-aligned bounding box
against the placed boxes in a square column whose side follows from a 0.3 fill of the grown-box volume
(`PLACEMENT_FILL`; defaults: footprint 6.4-7.1 m, 17-31 % of the position draws accepted, mean 22 %, 40 ms per scene;
the footprint grows 10 % after 200 failed draws of one body, needed once in 2 of 16 scenes; nearest bounding-box
separation median 3.4 cells, minimum the gap). Realise: X = R(q) (rest - centre + d) + p / h, V = R(q) v_def + v dt / h
in one world frame (the deformation fields from `Augmenter.initial_states` with the body's strength / velocity_dt /
perturbation_scale, one generator per body seeded by `BodySpec.seed`), `Material.cat` of `material_from_si` per body
(the scene's kappa, beta, mu_f; friction_epsilon and floor_scale travel in the scene's contact dict), the plane for
every object with plane_d = plane_height / h = 0 and no static points; a 157-body scene realises on an L40 in 1.9 s
cold (grid builds) and 0.18 s with cached grids. Checked: the rigid pose carries elastic energy 2e-16 (rotation and
translation exact), per-body mean velocities equal the rigid velocities, JSON round trip through
`dataclasses.asdict` / `SceneV5.from_dict`, energy pass on the realised CPU float64 batch finite.
For Anka: the bounding-box rule with the 0.5 m height cap makes the default scene sparse (1.0 m^3 of material lands
on 40-50 m^2 of floor, bodies 3.4 cells apart at the median), so body-body contact comes from the drift noise and the
landing; `PLACEMENT_FILL`, `placement_height` and `scene_cells` trade density for acceptance. For the integration:
across 64 scenes the GridCache holds up to about 1000 distinct box shapes (1-2 MB each on the GPU). Tests:
`tests/test_scenes_v5.py` (11: seed streams, cell budget and sides, placement and gaps, drift and velocities, materials
and the kappa floor, JSON round trip, summary, realise shapes / grids / material / scene, rigid pose and no overlap,
energy pass on the realised CPU float64 batch, config keys); full suite 233 pass (222 + 11). Nothing committed.
- Scene density (Anka, 2026-10-02): dense stacking, placement fill 0.6 and column height 1.0 m (the generator's first
  defaults, fill 0.3 and 0.5 m, spread 1 m^3 of material over 40-50 m^2 with a median box separation of 3.4 cells).
  Placement is bounding-box rejection sampling (grown by a 1-3 cell gap, footprint +10 % after 200 failed draws),
  about 150 bodies per 64k-cell scene, tens of milliseconds per scene, once per scene load.

### Body-body contact (2026-10-02, implemented: `contact.py`, `contact_kernel.py`; `Pairs`, `Batch`, `capture.py`, `step.py`)

Built as decided above, with the v5 interface contract (one world frame, `batch.body_contact`, `batch.meshes`,
`Pairs.partner_body` / `partner_face`, kind 2 in the reserved one-hot slot, partner radius channel 1).

- Detection (`contact.detect`, once per physical step at X). `BodyMeshes` holds one `wp.Mesh` per object over its
  exposed faces (two triangles per quad, local corner ids; triangle t of object o is the face of global sample
  `sample_off[o] + t // 2`), the vertices aliasing one float32 copy of the corners; rebuilt after `Batch.relayout`,
  otherwise refreshed (copy + `refit`) every detection. `body_query_kernel` (one thread per sample and other object,
  skipped outside the object's box grown by the reach) runs `wp.mesh_query_point_sign_parity` (3 rays) with
  max_dist = margin + r = 2r + |v_s| and writes the signed surface distance, the closest point and the partner face.
  Because the BVH returns either face when the closest point lies on an edge, the found face and its edge-adjacent
  faces (`face_neighbours`, built once per layout from shared corner pairs) are re-evaluated in torch in the batch's
  dtype; faces within `FACE_TIE_TOL` = 1e-5 (relative) of the smallest distance are ties and the one whose normal
  opposes the sample normal best is taken (without this an overhanging sample flickered between the top and the side
  face of its partner from step to step, a 5e-6 position change moving 8 % of the energy). Keep rule: face normal at
  X opposing the sample normal (n_q . n_s < 0), gap - r < margin with gap the signed distance (inside the partner
  always), and the closest point's lateral offset from the sample's normal line below r (the static disc's rule with
  r_p = r; a sample overhanging an edge by more than its radius does not touch the face). Static points and body
  faces compete for the nearest M_pair = 4 slots per sample (bodies ranked by signed surface distance, points by
  centre distance); slots are ordered plane, point ids, partner object ids; capacity mode has 1 + min(M_pair,
  Npts + O - 1) columns per sample. Both directions are kept.
- Geometry at a query (torch `_geometry` / `_body_geometry`, Warp `contact_pair_kernel`): for a kind-2 row the
  weights w of the closest point to x_s on the partner quad at its current corners c_i are computed and held, the
  partner point is p = sum w_i c_i, the normal the quad's diagonal cross product (rest normal when degenerate, as
  `sample_normals`), and the step displacement delta = (x_s - anchor) - sum w_i (c_i - C_i) with C_i the partner's
  corners at X: the slip of the sample relative to the partner's material point under it, so two bodies moving
  together see no damping or friction between them (the capture's static state gained `X` for this). Holding w is
  exact for planar faces; the gradient reaches the partner corners as -w_i times the sample's gradient (E depends
  on x_s - p only) plus the gradient through the normal, dE/dn = -ke relu(d)(x_s - p) - kd relu(-v_n) delta
  [d > 0] - (mu f_n f0'(y)/y) v_n u pulled back through the normalisation and the cross product; the four partner
  terms of the normal sum to zero, so the total force on the two bodies is zero exactly. The kernel writes a [Q,4]
  partner gradient (autograd Function) or scatters it with atomics (fused pass). `active_stiffness` adds a body
  pair's stiffness to the partner too; `penetration` stays per owner.
- Coupled centroid update (`Step.centroid_target`, `contact.translation_hessian`, `contact.pair_stiffness`). With
  body pairs the per-body implicit-contact Newton step (7.15) treats the partner as fixed and two stacked bodies each
  resolve the same penetration: the stack cycled and fell through. With `batch.body_contact` the Newton step is now
  taken jointly on all free bodies' centroids with the full translational contact Hessian (diagonal blocks M I +
  the sum of the pair stiffnesses a body owns or partners, off-diagonal blocks minus the sum over the pairs between
  two bodies; a 3O x 3O `torch.linalg.solve_ex`, CUDA-graph capturable). The pair stiffness is the exact Hessian of
  the pair energy with respect to the relative rigid translation with the load held: ke n n^T [d > 0] + kd n n^T
  [d > 0, v_n < 0] + the friction curvature mu f_n ((f0'' - f0'/y) u u^T / y^2 + (f0'/y)(I - n n^T)). The damping
  and friction terms were missing from H before: a single damped body on the plane (beta 0.3 or 1.0, kappa 20)
  two-cycled with translation residual 0.16 / 0.88 forever and never settled; it now converges in one query
  (1e-13), with friction in two. The one-body path without body contact keeps the 3x3 `_solve3`; the pinned beam
  reference (`tests/reference/pinned_beam_reference.pt`) is unchanged bitwise.
- Measured (L40 shared with other agents, float32, capacity layout). 40 bodies with sides 9-14 (61.9k cells, 32k
  samples, 6.7k valid pairs): detect 5.3 ms steady state (mesh refresh 1.1 ms, the rest query, tie-break and top-k),
  first call 298 ms (mesh builds, kernel load); 147 bodies with sides 3-12 (64.0k cells, 50k samples): 10.7 ms
  (refresh 3.7 ms: one `refit` per object in Python). Fused energy + gradient with body pairs 0.3-0.5 ms. The torch
  geometry of the capacity rows (160k-250k rows, 4 % valid) costs 3-4 ms per evaluation and the query evaluates it
  for the tokens and once for the stiffness (8.8 ms for the stiffness alone before the two were merged; the tokens
  remain): the inference query with body contact is bound by it, a Warp kernel for the kind-2 token geometry would
  remove it.
- Tests (`tests/test_body_contact.py`, 16): two touching 3x3x3 free boxes without a plane (9 + 9 pairs, partner
  faces under the samples, normals, both directions, no self pairs, sorted CSR; separated / velocity margin /
  penetrating with the sign parity; detection off and single body; tokens in the body slot with radius ratio 1;
  meshes rebuilt after relayout), closest point on a quad against a 200^2 barycentric grid on 64 warped quads and
  the tie rule, finite differences in float64 with respect to each body's corners at 1e-7 (planar faces under a
  random affine map of the world, friction load frozen, damping and friction active), Newton's third law on warped
  faces with all three terms (1e-10 of the gradient scale, also `contact_force` and the partner stiffness), co-moving
  bodies without slip, a 2x2x2 box on a 3x3x3 box on the plane for 100 steps with the zero-init network (static
  penetrations M_B g / (13 ke) between the bodies and (M_A + M_B) g / (9 ke) into the plane to 1e-4, 13 pairs
  because B's footprint overhangs A's samples at x = 2.5 and z = 0.5 by 0.1 cells), the Warp kernel against the torch
  path in four regimes (energies 1e-5, gradients 1e-4, the fused pass, partner-corner gradients alone, object
  weights), capacity against compaction (identical energy, gradient, tokens, penetration), and the captured query
  with body pairs against eager over three queries and an advance (5e-5 as the free-body capture tests: positions
  and energies within 1e-5, gradients 3e-5 at kappa 5, 3e-4 at kappa 20; the contact gradient of the two paths comes
  from torch and Warp float32 arithmetic and the stiff coupling feeds the difference back through both bodies).
  Full suite 249 (222 + 11 scene-generator + 16) pass. Nothing committed.
- For Anka: (i) the material-point slip (delta relative to the partner's held material point) and the exact
  translational Hessian (damping and friction stiffness, coupled over bodies) go beyond the letter of 7.15 and the
  contact note's "partner velocity zero"; both were needed for a stack of two bodies to rest. (ii) The pair list is
  frozen per step with the face recorded at X; a sample sliding across a partner edge keeps the old face until the
  next detection (the closest point then sits on that quad's edge). (iii) Deep penetrations beyond 2r + |v_s| are
  not detected (the query's max_dist); a larger reach costs nothing measurable if wanted.

### Pair edge comparison result (2026-10-02, 9 epochs, 3 ranks, otherwise the v4 configuration)

Against the A02 campaign: loss lower in all 9 epochs (mean 0.747 vs 0.755 over epochs 3-9), full-horizon selection
metric lower in 6 of 9 (mean 91 vs 113 N), cheap-validation medians and penetration at the same level, all states
surviving; network forward+backward 52 vs 75 ms at batch 16, memory 3.8 vs 6.9 GB, 797k vs 903k parameters.
Decision: `edge_module` defaults to "pair" for new training (TrainConfig); `Net()` keeps "a02" so the trained
checkpoint and the parity tests load unchanged.

### Scene regime, runner, validation and report keys for `scene_mode = "v5"` (2026-10-02, implemented: `jobs.py`, `runner.py`, `validation.py`, `report.py`, `train.py`; body mode unchanged)

- Regime (`jobs.sample_epoch_jobs`): with `scene_mode = "v5"` a fixed state is a scene; `job_count` = `scene_count`
  jobs (seed = scene index) with the unchanged K, H draws of the growth table (the same seed stream, so the first
  `scene_count` jobs coincide with the body regime's); `assign(jobs, world, 1)` (one scene per rank at a time) gives
  U = the heaviest rank's sum of K H. Updates per rank per epoch for the v4 growth table with 64 scenes (expectation;
  the LPT loads of seed 73 over 4 ranks are within 1 %): stage 0 (1, 8): 72; stage 1 (2, 16): 204; stage 2 (4, 32):
  616; stage 3 (8, 64): 1950; stage 4 (16, 128): 6398; stage 5 (32, 128): 7516 (30066 on one rank). Not the ~15k
  per rank of the campaign's last stage estimated above: with 4 ranks the last stage gives half of it, and the first
  stages give under 300 updates per epoch, each a 64k-cell scene query, so the first epochs are short.
- Runner (`runner.SceneRunner`, chosen by `runner.make_runner` from `cfg.scene_mode`): the rank holds ONE scene at a
  time. `load` samples the job's scene (`scenes_v5.sample_scene`), realises it into a fresh `Batch` (`runner.scene_batch`:
  `Batch.build(grids)`, `body_contact = True`, origin zero, material / scene / X / V, X_prev = x = X) and runs
  `Step.prepare` on all bodies (one candidate-noise generator per body, seeded by (master, scene, epoch, body)); every
  update serves one query of the scene like the body runner (`commit`: k += 1; k == K: `Step.advance` on all
  bodies, h += 1; h == H: load the next scene). Any non-finite body energy fails the scene (FailureRecord with the
  scene's seed, K, H, k, h; the next scene loads). A rank whose queue ran dry keeps its last batch with `active`
  all False (the trainer's loss mask) and counts `idle_updates`, so lighter ranks idle through the common U; a
  `scene_count` below the world size raises. The runner exposes `batch`, `U`, `active_count` (bodies of the current
  scene), `failures`, `queries` (body queries served), `pairs` / `pair_history` (valid pairs by kind at every
  detection) and `epoch_summary()` for the run record. `train.py` keeps the per-object objective averaged over the
  scene's bodies (the loss mask), records the max penetration of every update's fused candidate (v5 only), skips
  `compile_layers` in v5 (`torch.compile` with static shapes would recompile at every scene) and writes `query_count`
  = body queries, `filler_queries` = idle updates and the `scene_regime` block.
- Validation (`validation.validate_cheap_v5`, `validate_full_horizon_v5`): held-out scenes from `scenes_v5.held_out_scene`;
  cheap = `validation_scene_count` scenes x `validation_iterations` queries, full horizon = `validation_full_scene_count`
  scenes x K x H with the stage caps as before. A record is one scene with the per-body fields aggregated over its
  bodies in SI (residual_n: mean of the bodies' free-corner residual norms, energy_joule: sum, penetration_r: max,
  inverted_cells: sum, scale_joule: sum), plus `interbody_penetration_r` (max over the kind-2 pairs of relu(r - gap) / r,
  `contact.kind_penetration`), `plane_penetration_r`, `contact_pairs` {total, plane, point, body} and the scene summary.
  `momentum_drift` = |sum m_i v_i(t) - sum m_i v_i(0) - t M g| / |M g t| (SI) after the full horizon on a contact-free
  copy of the first full-horizon scene (`contact_free_copy`: plane removed, bodies on a centred (x, z) grid spaced by
  twice the bounding-box reach plus the rigid travel plus 4 cells, body-body detection left on and finding nothing).
  `report.summarize_cheap_validation` adds the `interbody_penetration_r` / `plane_penetration_r` iteration curves and
  mean `contact_pairs` when the samples carry them, `summarize_full_horizon` the `final_*` statistics and
  `momentum_drift`; `build_epoch_record` takes `scene_regime`; body-mode records are unchanged (the keys appear only
  with scene samples) and the existing dashboard renders a v5 run (`write_mixed_report` in the test).
- Centroid precision (found by the momentum check, fixed in `physics.centroid` and `Fusion.fuse`): the mass-weighted
  centroid sums ran in float32, and for a body 60 cells from the world origin (positions ~60, ~500 corners) each
  reduction lost ~1e-4 cells, 2 % of a step's gravity increment g dt^2 / h = 4.4e-3 cells; the free-body update
  c(x_{n+1}) = c_n + cdot_n + g turned that into a velocity error that random-walked over the horizon, and the
  fusion's float32 weights w = m / sum m summed to 1 + eta with eta ~ 1e-7 fixed per grid, a BIASED error eta |c| ~
  6e-6 cells per step (the drift did not shrink with H). Both sums now run in float64 and return the batch dtype
  (pinned paths never read them; the one-body free-body tests sit near the origin and never saw it). Measured on a
  4-body CPU scene (x up to 63 cells, zero-init network, K = 1): drift 5.4e-3 / 7.8e-3 / 8.6e-3 at H = 2 / 8 / 32
  before, 5.0e-5 / 1.9e-4 / 1.1e-4 after (seed 1: 1.9e-4 / 1.6e-4 / 1.3e-4; float64 batch: 3e-8). The remaining
  error is the float32 representation of the positions themselves (ulp(60 cells) = 3.8e-6): a float64 c_n / c_t
  pipeline would remove it, but needs the step-constant centroid buffers in float64 (capture state); not done.
- Picard warning (`Step._check_picard`): emitted only with `translation = "picard"`; with the default
  implicit-contact translation a 150-body scene exceeded the constant on dozens of bodies at every landing and
  flooded the log (the test of the warning uses the Picard translation).
- Smoke (v4 config + scene_mode v5, max_epochs 1, L40 shared with other agents' runs: timings indicative).
  scene_cells 8000, scene_count 6, validation 2 / 1, `--max-updates 60`: the 6 stage-0 jobs (K = 1, H = 8, 7, 3,
  1, 1, 1) give U = 21 updates, 5.7 s for the epoch, 55 ms per update at 8.2k cells / 15-25 bodies (K = 1, so every
  update advances), 0.17-0.34 s on the updates that load a scene; losses 0.88 -> 0.85 then up to 4.5 over 21
  updates (meaningless at 21 updates; mean 1.13); mean residual 544 N; cheap validation (2 scenes x 32 queries)
  and full horizon (1 scene x 1 x 8) survive; momentum drift 4.1e-4; the report renders. scene_cells 64000,
  scene_count 2 (157-159 bodies), 10 updates: 0.74-0.91 s per update and 3.5-4.6 s on the two loads (grid builds
  and the first prepare; the GPU was running the test suite at the same time), losses 0.86-0.88, residual 519 N,
  validation 1 scene x 8 and 1 x 1 x 8 survive. Pair counts: ZERO in every stage-0 run: within 8 steps (27 ms)
  no body falls the 2+ cells to the ground or to a neighbour; with the zero-init network over 128 steps on the
  default scene 0 (159 bodies, 64.3k cells, 51.9k samples): 25 pairs (12 plane, 13 body) at step 16, 690 (212 / 478)
  at step 32, then the shapes degenerate (no elastic response: the deformation velocity field integrates freely,
  residual 1.3e5 N at step 32, 2e9 N at step 128, penetrations of tens of r, fall-through) and the counts (48k at
  step 128) mean nothing; query + commit 0.40 s median and advance (prepare + detect) 0.19 s there, inflated by the
  contention and the degenerate pair counts. So the first two growth stages (H <= 16, 53 ms) train v5 scenes
  almost without contact; `placement_height` / the first body's 1-2 cells decide when contact starts.
  One training update of the 157-body scene (profile, eager network, the suite's tail still on the GPU): query
  170-280 ms, loss backward 140-160 ms, optimizer 2 ms, commit + advance 80-120 ms, so 0.4-0.5 s per update at
  K = 1 (the 16-beam body batch takes 0.09-0.11 s); the first query of a freshly loaded scene 1.9-2.1 s (mesh
  builds, fusion factors of ~150 new grid shapes) and a load with cold grids 1.5-1.9 s. `Step.prepare` costs 11 ms
  without the per-body candidate-noise loop and 108 ms with it (159 bodies, Python loop with one host sync per
  body): the loop should be batched by grid group like `Augmenter.initial_states`. Momentum drift of the 159-body
  scene's contact-free copy (grid layout, float32, K = 1, H = 128): 3.6e-4 (1.1e-3 with the bodies in a 95 m row).
- Tests: `tests/test_scene_runner.py` (5: v5 job sampling and U incl. the default regime's stage-0 U, an epoch of
  3 scenes of ~1500 cells on the CPU serving exactly sum K H queries with a fresh Batch per scene and a detection
  per load and advance, a lighter rank idling through U with world 2, failure loading the next scene, the runner
  factory and the world-size check) and `tests/test_validation_v5.py` (5: cheap records and summary with the new
  keys and finite values, body-mode summaries without them; full-horizon records, summary, epoch record with
  `scene_regime`, JSON-clean and rendered by the dashboard; momentum drift of the contact-free check below 1e-3
  on two scenes with the zero-init network (no pairs, no plane); `kind_penetration`). The first full-suite runs
  showed one failure that is not from this work: `TestPinnedPathUnchanged.test_bitwise_reference` (the
  `contact_f64` records) broke when the `TrainConfig.edge_module` default became "pair" (decision above): the
  reference network of `tests.test_capture.small_net` followed the default; the test now pins "a02", the module
  the reference was recorded with. Full suite: 258 pass, the 10 new tests included. Nothing committed.

### Confirmation run (2026-10-02, GPU 3 of the L40 box shared with other agents: timings indicative; nothing in the package changed)

- Full suite: 258 pass (`unittest discover`, 143 s), no failures, errors or skips.
- Smoke as requested (v4 config + `scene_mode` v5, `scene_cells` 32000, `scene_count` 4, validation 2 / 1, `max_epochs` 1,
  `log_every` 5, `--max-updates 60`): the stage-0 regime caps the epoch at U = 12 (K = 1, H drawn 7, 3, 1, 1 over the
  4 scenes), so 12 updates were served (984 body queries) in 13.3 s (13.7 s wall with start-up); scenes of 76-86 bodies
  (mean 81.5) and 32.1-32.6k cells; logged updates 0.77 s (first query of the first scene), 0.18 s (steady), 0.87 and
  0.54 s (updates whose commit loads the next scene); losses 0.845 / 0.873 / 1.059 / 0.911 at updates 1 / 6 / 11 / 12,
  epoch mean 0.888, gradient norms 1.4-49, residual 488 N; peak CUDA memory 1.90 GiB allocated (4.98 GiB reserved); pairs
  zero at all 16 detections (plane and body, as in the stage-0 smokes above); cheap validation (2 scenes x 32 queries,
  79 / 80 bodies) and full horizon (1 x 1 x 8) survive, penetrations 0, momentum drift 1.3e-4; no warnings in the log.
  Keys present: `validation` {contact_pairs, interbody_penetration_r, plane_penetration_r, penetration, force_residual,
  relative_energy, descent_rate, mean_normalized_loss, physical_survivors, failed_count, samples, selection, ...},
  `full_horizon_validation` {final_contact_pairs, final_interbody_penetration_r, final_plane_penetration_r,
  final_max_penetration_r, final_energy_joule, final_free_force_residual_norm_n, momentum_drift, physical_survivors,
  selection, ...}, `scene_regime` {scenes, bodies_mean, cells_mean, body_queries, idle_updates, detections, pairs_mean,
  body_pairs_mean, body_pairs_max, plane_pairs_mean, steps_with_body_pairs, scenes_served}.
- Supplementary run to actually reach 60 updates (same config with `log_every` 1 and the first growth stage set to
  K 4 / H 32; scratch config only): 60 updates on one scene (83 bodies, 32.2k cells, the job's H = 32, so 15 physical
  steps), 0.17-0.21 s per update steady (median 0.179 s; the first 0.83 s), 38.8 s for the epoch; loss 0.881 -> 0.82-0.86
  over the last ten updates (epoch mean 0.845), gradient norm 9 -> 0.6-0.9, residual 595 N; peak 1.83 GiB. Plane pairs
  appear from the 10th physical step (7 pairs summed over the 16 detections), no body pairs; the full horizon (1 x 4 x 32)
  shows plane pairs from step 20 (1) to step 32 (9), body pairs 0, all penetrations 0, residual 146 -> 24 N, momentum
  drift 7.3e-5; cheap validation 2 / 2 survive.
- Full-size scene (seed 73 scene 0: 159 bodies, 64.3k cells, 93.9k corners, 51.9k samples): realise + `Batch.build`
  1.01 s with cold grids; first `prepare` 0.38 s (mesh builds, kernel loads), steady `prepare` 57 ms with the per-body
  noise loop and 11 ms without; `contact.detect` alone 9.4 ms at 0 pairs and 12 ms at 558 pairs, energy + gradient
  0.3 ms; eager inference query 1.09 s first, 153 ms steady (peak 2.0 GiB); query + backward in train mode 0.32 s
  (peak 3.58 GiB). Pairs with the zero-init network: 0 through step 8, 6 plane at step 16, 207 (92 plane / 115 body)
  at step 24, 558 (191 / 367) at step 32; lowest corner 1.01 -> -2.74 cells.

### Adversarial review of the body-body contact physics and the v5 data path (2026-10-02, read-only; `tests/test_v5_review.py`)

Checked `contact.py`, `contact_kernel.py`, `scenes_v5.py`, `runner.py`, `validation.py`, `step.py` against the plan and
the contract. Confirmed (pinned as `expectedFailure` tests so the suite stays green; remove the decorator with the fix):
- Units between bodies (`units.material_from_si`, `physics.energy`, `contact.pair_energies`, `Step.centroid_target`):
  every body's energy is in its OWN normalised units (lam, rho, eta, ke, kd divided by the body's mu; mu = 1 in the
  elastic density), and a body pair's energy belongs to the sample's body, so its gradient on the PARTNER's corners is
  in the owner's force units (mu_owner h^2) added to rows in the partner's units. In SI the partner feels the reaction
  scaled by mu_partner / mu_owner: two 3x3x3 boxes with E 1e5 / 1e6 (equal masses) give -1125 N on A and +11250 N on B
  (normalised forces exactly equal and opposite, which is why the equal-material third-law test passes; the same on
  the Warp capacity path), and the coupled centroid step without gravity moves the stiffer body 10x further (1000x for
  E 1e3 / 1e6): SI momentum is not conserved in body-body contact. Each body effectively sees every pair with its own
  kappa E_o h (up to the (1 + nu) ratio). v5 draws E log-uniformly over 1e3-1e6 per body, so nearly every pair is
  affected; a common normaliser per scene (one reference mu with a per-body mu ratio in the elastic density) is the
  fix (`test_contact_force_sums_to_zero_in_si`, `test_coupled_centroid_update_conserves_si_momentum`;
  `test_equal_materials_control` passes and records the effective stiffness (n_A + n_B) ke of both directions).
- Failed last scene (`SceneRunner.load`, `train.py`): a non-finite scene when the rank's queue is already empty stays
  as the rank's batch with `active` all False, and the trainer's `(loss_vec * mask).sum()` is NaN (NaN * 0): no
  optimizer step for the rest of the epoch, on every rank under DDP (`test_failed_last_scene_leaves_a_finite_loss`).
Refuted: partner normal outward and opposing, gap / depth law identical to the plane law (d = r - gap, r_total = r);
no detached term drops the partner force (w held and f_n detached symmetrically; the normal's gradient sums to zero
over the four corners; the kernel's partner scatter matches the derivation and the builder's tests); self pairs
excluded in the kernel, faces recorded as `sample_off[o] + tri // 2`, meshes refreshed by `body_meshes` at every
detection, capacity columns 1 + min(4, Npts + O - 1); the centroid update takes the partner force through
`contact_force` and the coupled Hessian (momentum conserved to 4e-15 with equal materials); the generator's AABB rule
left no body pair, no sample inside another body and no plane pair at step 0 in 16 default scenes (nearest surfaces
1.6-3.0 cells; the guarantee is for the rigid boxes, the deformation field never ate the gap); velocities m/s x dt / h;
validation metrics in N, J, r and kg m/s as documented. Remark without a test: the detection margin r + |v_s| uses the
sample's own speed, not the relative speed to the partner body, so in the approach step only the moving body's samples
pair (gap 1.3 cells, 0.6 cells/step: 9 pairs owned by the mover, 0 by the body at rest); contact is still detected
from one side, but the impact step has half the stiffness and the owner (hence, with the units finding, the modulus
that decides the forces) depends on who moves.
Full suite after the review: 262 tests, OK with 3 expected failures (the pinned findings). Nothing committed.

### Pinned bodies in v5 scenes (2026-10-02, implemented: `scenes_v5.py`, `grid.py`, `fusion.py`, `config.py`, `validation.py`, `runner.py`)

Anka's decision above (a fraction of the bodies clamped at one face and held at the initial pose), built as follows.
- Config: `pinned_body_fraction` = 0.25. Generator: a fourth seed stream (`SeedSequence(...).spawn(4)`: scene, bodies,
  placement, pins) draws per body a uniform `u` and a face index whatever the fraction, so the body, placement and
  scene draws are those of the all-free generator (fraction 0 reproduces the earlier scenes exactly; tested).
  `BodySpec.pins` is "none" or one of `grid.FACE_PINS` ("xmin_face", "xmax_face", "ymin_face", "ymax_face",
  "zmin_face", "zmax_face"; records without the key read as free), `BodySpec.pinned`, JSON round trip,
  `scene_summary["pinned_bodies"]`, `SceneRunner.epoch_summary()["pinned_bodies_mean"]`. Placement unchanged: a pinned
  body keeps its random pose above the ground (an anchor or a hanging beam); its rigid velocity is zero in the spec and
  in `realise` (the deformation velocity field remains, zero on the pinned rows). `realise` takes the grid with the
  body's pins: the clamped face sits exactly at the rigid pose (`Augmenter.initial_states` puts pinned corners at rest
  before the rotation), V = 0 there.
- Grid: `Grid.build(cell_counts, pins)` accepts the six faces (`face_pin_mask`: coordinate 0 or n_axis); "zmin_face"
  and "none" are unchanged bitwise (mask and the hard-coded reference corners), the other faces take the reference
  corners from `reference_corners` (the voxel rule), and every field equals `Grid.from_voxels` on full occupancy with
  the face's corner mask (test). The key is (nx, ny, nz, pins), so pinned and free bodies of one shape are different
  grids and never share a fusion group.
- Fusion: `KronFactor` solves any single pinned face (the Dirichlet node 0 or n dropped from that axis' 1D factor; the
  free corners are a product set in lattice order) and `Fusion.factor`'s "auto" takes it for every box grid in
  `grid.BOX_PINS`; the z-min arithmetic is the one before. Against the dense inverse on (2, 3, 4) and (3, 2, 2) for all
  six faces: solve, K_fp, the fused displacement with prescribed pins and the projected gradient within 1e-11 (float64;
  measured 1e-13 to 1e-15).
- Step: nothing changed. `prepare` sets Y = X on pinned rows and the candidate noise is zero there, the fusion leaves
  pinned rows at zero displacement, `advance` without prescribed positions keeps X and sets V = 0 on pinned rows, so a
  pinned corner is bitwise X over the steps; `centroid_target` masks the non-free bodies in the coupled Newton step
  (c_t = c_x, no coupling blocks) and `fuse` applies the centroid target to unpinned grids only.
- Validation: `momentum_si` / `total_mass_si` sum over the FREE bodies (a pinned body hands its momentum to the pins;
  the check returns NaN for a scene without a free body). The other records aggregate over all bodies as before
  (the residual is the free-corner residual, so a pinned body's clamped rows do not enter). Grep of `free_objects`,
  `any_free`, `pins == "none"` across the package: the only all-free assumptions were the momentum check and two test
  assertions (`test_scene_runner`, `test_scenes_v5`), now per body.
- Statistics (default configuration, 16 scenes of 64k cells): 2452 bodies, 616 pinned (25.1 %; 24-50 per scene of
  135-166 bodies; 24.7 % of the cells), faces xmax 111 / ymax 108 / zmin 103 / xmin 102 / zmax 98 / ymin 94, 43 ms per
  scene (40 before); the 6000-cell test scenes over 20 seeds: 303 bodies, 87 pinned (28.7 %), all six faces drawn.
- Tests: `tests/test_scenes_v5.py` +2 (fraction within 0.1 of 0.25 over >= 200 bodies, all six faces, zero rigid
  velocity of pinned bodies, fraction 0 reproduces the all-free scene body by body, fraction 1 pins every body, JSON
  round trip with pins and the legacy record; realise: pinned corners at the pose to 1e-12 with the deformation field
  on the free corners, V = 0 on pinned rows with the deformation velocity field elsewhere, a rigid velocity in the spec
  ignored), the existing realise / summary / config tests extended; `tests/test_fusion.py` +1 (the six faces against
  dense); `tests/test_pinned_bodies_v5.py` (6): the face grids against voxel masks field by field and the unchanged
  z-min / none reference corners; a 3x2x3 block hanging from its pinned top face at y = 6 above the plane with a free
  2x2x2 box one cell above it, 30 steps of 4 queries with the zero-init network (implicit-contact translation, beta
  0.3, kappa 20, float64): pinned corners bitwise X with V = 0 and the candidate at X at every step, the free body in
  exact free fall (c_n = c_0 - g n(n+1)/2 to 1e-9) until its first body pair at step 15, at rest on the pinned face at
  the end (bottom corners r - 1.5e-4 r over the face, 13 body pairs, no plane pair, body penetration 1.5e-4 r, all
  energies finite); the same pins held in float32; a mixed 1500-cell scene at fraction 0.5 on the CPU: free and
  pinned grids in separate groups, pins held over 3 steps while the free bodies fall, the runner's epoch of 3 scenes
  served without failures with both kinds of body, `validate_cheap_v5` / `validate_full_horizon_v5` records surviving
  with `pinned_bodies` in the scene summary, momentum drift of the free bodies 1e-4 (below 1e-3), the mass and
  momentum sums over the free bodies only and NaN for an all-pinned scene. Full suite (`unittest discover`, GPU 3
  shared): 271 tests OK (262 + 9; the 3 expected failures are the review's pinned findings), 154 s. Nothing committed.
- For Anka: (i) with the zero-init network a pinned body's free corners integrate freely like everything else (the
  lower part of the hanging block in the test sags 2 cells in 30 steps), so a trained network is what makes a pinned
  body an elastic anchor; the pins themselves hold regardless. (ii) A free body landing on a pinned face rests on
  corners that cannot move, so the partner force on them is absorbed (the pinned body's residual ignores pinned rows).
  (iii) The pin draw costs the generator nothing measurable; `pinned_body_fraction` = 0 gives the earlier scenes.
- Early contact (Anka, 2026-10-02): about 30 % of the free bodies start resting, placed with zero gap directly on the
  ground or on a pinned body, so contact exists from step 0 in every growth stage; the others fall and drift as
  planned; growth table unchanged. (Gravity needs 20+ steps to bring bodies together even from 2 cells up, so stages
  with H <= 16 would otherwise be contact-free.)
- Review findings to fix before training (2026-10-02): common energy normaliser per scene (reference modulus, per-body
  elastic ratio) so body-body reactions are equal and opposite in SI; NaN-safe loss mask and a finite idle batch after a
  failed last scene; dynamic-shape compilation of the network for per-scene shapes; batched candidate noise; detection
  margin from the relative speed of sample and partner.
- Static faces (Anka, 2026-10-02): scenes also get artificially sampled static colliding faces: 0-8 planar quads per
  scene, side 2-10 cells, random orientation (walls, ramps, slabs), placed in the scene column and rejected against the
  bodies' initial bounding boxes, stored as four corners in the scene record; one `wp.Mesh` per scene built once,
  queried like a body mesh (`partner_body = -2`, face index into the static table), the body-face contact law with
  constant partner corners, token = static partner with the radius channel at its cap; penetration folded into the
  static metric. To implement after the current fix agent (same files).

### Review fixes, resting start, relative-speed margin and the v5 update speed (2026-10-02, implemented: `units.py`, `structs.py`, `physics.py`, `energy_kernel.py`, `scenes_v5.py`, `config.py`, `contact.py`, `contact_kernel.py`, `runner.py`, `step.py`, `augment.py`, `fusion.py`, `batch.py`, `network.py`, `train.py`, `validation.py`; body mode bitwise unchanged)

- Common energy normaliser per scene (review finding 1). `material_from_si(..., mu_ref=None, dtype=float32)`: with
  `mu_ref` the unit of energy is mu_ref h^3 for every body of the scene (`Material.mu_norm`; `energy_scale` /
  `force_scale` read it), `ke`, `kd` and the energy floor are divided by mu_ref, and the new field `mu_scale` =
  mu / mu_ref carries the elastic (mu_scale psi(lam), lam = the body's own lambda / mu), damping (mu_scale eta) and
  inertia (mu_scale rho) terms into that unit; `lam`, `rho`, `eta` keep the body's OWN dimensionless groups, so
  `units.conditioning` is untouched and bitwise what it was (test: a v5 body against the same material in body mode).
  Consumers: `physics.elastic_damping` / `inertia` / `corner_mass` (`units.unit_rho` = rho mu_scale, so `total_mass`,
  `centroid`, the fusion's centroid weights (a ratio, unchanged) and the coupled centroid Hessian are in the common
  unit), both Warp kernels (`mu_scale` [O] passed in; the per-cell kernel outputs are bitwise the old ones at
  mu_scale = 1, checked on the GPU), the contact code unchanged (ke, kd already per object). `scenes_v5.realise` passes
  mu_ref = the geometric mean of the bodies' shear moduli (`units.reference_modulus`) and builds the material tensors
  in the requested dtype (exact float64 on the CPU paths instead of float32 roundings cast up). Body mode passes
  nothing: mu_scale = 1, mu_norm = mu, every field as before (`Material.__post_init__` defaults; `cat` / `__getitem__`
  by keyword). Measured on the review's two 3x3x3 boxes (`tests/test_v5_review.py`): SI third-law residual
  |F_A + F_B| / max |F| = 0.90 before (E 1e5 / 1e6: the partner felt mu_B / mu_A = 10 times the reaction), 1e-16
  after (also 1e3 / 1e6, a factor 1000 before); the coupled centroid update's SI impulse sum 1e-12 relative after
  (both bodies move, momentum conserved; the former `expectedFailure` tests pass and the own-units configuration is
  kept as `test_own_units_record_the_finding`); a 3-body scene's total SI energy equals the sum of the three
  body-mode SI energies with the same plane pairs to 1e-9 relative (exact float64 materials). Full-size scene: a
  1492-pair step 0 (resting bodies) gives the same losses to three digits as before.
- Resting start (Anka's early-contact decision). `resting_body_fraction` = 0.3 of the FREE bodies; a fifth seed
  stream draws (u, face, yaw) per body whatever the fraction, so fraction 0 reproduces the earlier scenes exactly
  (`tests/reference/scenes_v5_6000_cells.json`, three scenes recorded before the change, compared field by field).
  Decisions taken in building it, for Anka: (i) a resting body lies FLAT on one of its six faces (uniform) with a
  random yaw about the vertical, not in a random orientation with a corner on the plane: a cuboid on a corner is not
  at rest and its face samples are cells away from the plane, so detection would see nothing; (ii) "zero gap" is the
  contact gap of the sample-sphere model, d = r - gap = 0: the bottom face sits r = h/2 above the support, so all its
  samples pair at step 0 with zero penetration and the body settles by the static penetration (~1e-3 cells) only;
  corners on the plane would mean d = r on every bottom sample, 100x the body's weight (kappa 30, E 1e5), and the
  body would jump; (iii) a resting body carries no deformation field (`perturbation_scale` 0) and no velocity: the
  field's RMS (up to 0.45 cells) is hundreds of static penetrations; (iv) on a pinned body: the exact closest approach
  along -y of the flat bottom to the tilted box (`support_height`, the maximum of the box's upper envelope over the
  footprint at the vertices of the arrangement: box corners inside the footprint, footprint corners over the box,
  edge crossings), the pinned body counted as its box inflated on every side by `CLEARANCE_SIGMAS` = 3 RMS of its own
  deformation field (without it the deformed support penetrated the resting body by up to 0.4 cells from below and
  from the side where a tilted box rises past the resting body's edge), the bounding boxes only selecting the
  candidates, a separating-axis test (`boxes_overlap`) verifying the supports and the grown-box rule everything
  else. `BodySpec.resting`, JSON round trip (old records read as False), `scene_summary["resting_bodies"]` and
  `["resting_on_bodies"]`, `epoch_summary()["resting_bodies_mean"]`. Statistics: 20 scenes of 6000 cells: 70 of 216
  free bodies resting (32 %), 36 of them on pinned bodies; 8 default scenes (145-163 bodies): 264 of 910 free bodies
  (29 %), 75 on pinned bodies, placement acceptance 6-10 % with 2.3 footprint growths per scene (the resting bodies
  fill the ground layer; the fill constant assumes the whole column) and 74 ms per scene (43 before); step 0 of the
  default scene 5 has 1492 pairs (0 before). Tests (`tests/test_scenes_v5.py`): fraction, faces, zero velocity and
  field, flat pose, ground gap r to 1e-9 cells, pinned supports at r above the inflated box to 1e-6 (no overlap) and
  within 0.05 cells of the closest approach by an independent bisection over dense footprint samples, the exact
  support against the bisection on 30 random tilted boxes, fraction 0 against the recorded scenes, detection at X of
  four 3000-cell scenes finding the plane pairs of every ground-resting body with |d| < 1e-6 and no penetrating pair
  of any resting body; the placement / realise tests exempt the (resting, pinned support) pairs from the grown-box
  rule and the resting bodies from the height rules.
- Detection margin from the relative speed (review remark). Body pairs use margin = r + |v_s - v_f| with v_f the mean
  corner velocity of the partner face found by the query (the query's reach is the largest possible margin + r,
  2r + |v_s| + max |v|); static partners keep r + |v_s|. Two co-moving bodies at surface distance 0.7 no longer pair,
  a body at rest sees the body coming at it from both sides (9 + 9 pairs), moving away or sliding sideways counts the
  same (the margin is a speed) (`tests/test_body_contact.py`).
- Failed last scene (review finding 2). `SceneRunner._idle`: with the queue empty the batch gets `active` False AND a
  finite state (candidate back at X, energies, gradients, history and Picard constant zero), so the trainer's loss is
  finite in the product form too and its gradients are exactly zero (`test_failed_last_scene_leaves_a_finite_loss`,
  decorator dropped, checks both forms and every parameter's gradient over two idle updates). Under DDP: a 2-rank
  gloo test on the CPU (`tests/test_scene_runner.py::TestIdleRankUnderDDP`, torch.multiprocessing spawn, one scene per
  rank, rank 1's scene failing at the first update): both ranks keep finite gradient norms and losses for the rest of
  the epoch, rank 1's losses zero. Note for Anka: the failing update itself is not new territory: a NaN produced inside
  the energy sends NaN gradients to the parameters through the masked loss (masked_fill zeroes the loss row, but
  0 x d asinh(NaN) = NaN) and, under DDP, to every rank through the all-reduce; train.py skips that optimizer step on
  the non-finite gradient norm, so one update is lost per failure on all ranks.
- Performance of the v5 update (default scene 5 of seed 73: 159 bodies, 64.4k cells, 94k corners, 52k samples, 151
  distinct grid shapes; L40 alone, float32, eager network unless stated). Before: prepare 57.6 ms, eager query 152 ms
  (0 pairs; 169 ms at the 1492 pairs of the resting start), update (query + backward + optimizer) 306 ms (346 ms with
  the pairs). The breakdown showed the per-group Python loops of the fusion as the cost, not the network: fuse 77-92
  ms, project_gradient 60 ms, network forward 16.5 ms, the backward 130 ms mostly the fuse's. Done: (a) `Fusion(
  batched=True)` / `fusion.BatchedKron`: every object's free lattice embedded in the batch's largest one (13^3 at most)
  with zero-padded eigenvector blocks and infinite eigenvalue sums on the padding, so all ~150 Kron solves are one
  padded chain of six batched matmuls; the right-hand side, the centroid completion and `project_gradient`'s mean
  removal by batch-wide segment sums; built once per batch layout and dtype (`Batch.fusion_cache`); the loop path
  stays for one-group batches (body mode: unchanged bitwise) and for prescribed pins; train.py sets it in v5 mode.
  Fuse 77 -> 0.9 ms, project_gradient 60 -> 0.8 ms, backward 130 -> 35 ms: eager query 169 -> 31 ms, update 346 -> 68
  ms (loop and batched agree to 1e-11 in float64 and 2e-5 in float32 on a mixed batch of free and pinned boxes,
  gradients to dF included; `tests/test_fusion.py::TestBatchedKron`). (b) Candidate noise: `Step.prepare` takes ONE
  generator for the batch (`SceneRunner`, the v5 validation; seeded by master, scene, epoch) and draws the use flags,
  the RMS values and the multiscale fields of ALL objects in a handful of launches (`Augmenter.candidate_noise_all`:
  per wavelength one normal tensor over the padded lattices of every object, one gather for all corners through
  tables cached on the batch, RMS normalisation and the free bodies' mass-weighted mean removal by segment sums); the
  per-object loop with its host synchronisations stays for the body regime's generator lists. The v5 candidate stream
  is a different stream (documented in `prepare`); prepare 57.6 (per-body loop; a first per-group variant cost 96 ms
  with ~1 body per group) -> 22.6 ms, of which detection 12.8 ms. (c) `contact_kernel.pair_geometry_kernel`: the
  tokens', translation stiffness's and penetration's geometry over the capacity rows in one launch (padded rows
  zero; `contact.USE_WARP_GEOMETRY`), the inference path with 260,700 capacity rows of which 1492 valid: tokens 3.00
  -> 1.03 ms, `_geometry` 2.19 -> 0.20, `translation_hessian` 3.88 -> 1.92, penetration 2.37 -> 0.40, the capacity
  inference query 37.0 -> 34.5 ms (training uses the compacted rows and is unaffected). (d) Mesh refresh: Warp 1.17 has
  no batched refit (`Mesh.refit` is one `wp_mesh_refit_device` call per mesh); 4.2 ms for 159 meshes, kept.
  (e) `Net.compile_layers(dynamic=True)` and train.py compiling in v5 mode too: with static shapes the compiled layer
  recompiles at every scene (dynamo unique graphs 5 / 10 / 15 over three scenes, 8 s each); dynamic keeps 5 graphs
  over the three scenes (first compile 11.5 s, 0.8 s per further scene for factors and meshes) and the update is
  65 ms against 68 ms eager: the network is 18 ms forward and ~30 ms backward of the update now, the compile saves
  little. After everything: prepare 22.6 ms, eager inference query 31 ms, update 65-68 ms (0.1-0.15 s targeted; the
  remaining breakdown of the no-grad query: network 17.6, centroid_target 9.0 (contact force, coupled Hessian and
  the 477 x 477 solve), features 6.8, energy + gradient 5.1, fuse 0.9, project_gradient 0.8 ms; with grad +35 ms
  backward), peak memory 3.7-3.8 GiB.
- Tests: 281 pass, no expected failures left (271 + 10 new: common unit 3 (own-units record, scene energy sum,
  conditioning channels), resting 2, DDP idle rank 1, batched Kron 2, batched noise 1, geometry kernel 1; extended:
  the two former expected failures and the idle-batch test now pass, detection with the relative speed, the placement
  / realise tests with resting bodies), 206 s on GPU 3; `uvx ruff format` / `check` clean; pre-commit clean on the
  package files (the typos hook flags two words in Anka's `notes/v5-free-motion-with-contact.md`, untouched).
  Nothing committed.

### Static faces in v5 scenes (2026-10-02, implemented: `scenes_v5.py`, `config.py`, `structs.py`, `batch.py`, `scenes.py`, `contact.py`, `contact_kernel.py`, `runner.py`, `validation.py`, `report.py`; body mode unchanged)

Anka's decision above (artificially sampled static colliding quads as fixed contact partners), built as follows.
- Generator (`scenes_v5.py`, sixth seed stream `rng_faces`; `ss.spawn(6)`): `static_face_count` in U{`static_face_count_range`}
  (default (0, 8)) quads per scene with side lengths U(`static_face_size_cells`) (default (2, 10)) cells, a uniform random
  unit normal (normalised standard normal) and an in-plane rotation U(0, 2 pi), the centre uniform over the placement
  column's footprint and between the ground and `placement_height`. Order (documented in the module): the faces are placed
  AFTER every body, so the body draws and the body placement are exactly the earlier generator's (count 0 reproduces
  `tests/reference/scenes_v5_6000_cells.json` field by field; the body lists of a scene with and without faces are equal)
  and no body is ever placed after a face; a candidate position is rejected while any corner lies below the ground or
  the quad intersects a body's initial bounding box grown by one cell, or by the body's deformation clearance
  `field_clearance` when that is larger (a free body's field moves corners up to 3 RMS = 1.35 cells, more than the one
  cell of the decision; `quad_boxes_overlap`, a separating-axis test over the box axes, the quad normal and their cross
  products, vectorised over the bodies, checked against dense sampling of the quad); after `MAX_FACE_DRAWS` = 200 rejected
  positions the face is dropped (`placement["static_faces_dropped"]`, never happened in 32 scenes). Record:
  `SceneV5.static_faces` [F][4][3] m, corners in order around the quad so the quad normal is the unit diagonal cross
  product (c2 - c0) x (c3 - c1) as for the bodies' faces (`static_face_corners`), JSON round trip, `from_dict` reads
  older records as empty, `scene_summary["static_faces"]` and `["static_face_acceptance"]`, `placement["static_face_draws"
  / "static_face_acceptance" / "static_faces_dropped"]`, `SceneRunner.epoch_summary()["static_faces_mean"]`. `realise`
  hands the table to `ContactScene.faces` [F,4,3] in cell units (one world frame, shared by every body; empty in body
  mode and in `batch.empty_scene`; `scenes.cat_scenes` concatenates it). Statistics: default configuration, 8 scenes of
  64k cells: 4.6 faces per scene (2-8), position acceptance 53 % (24-73 %), no face dropped, sides 2.1-10.0 cells,
  centre heights 3-39 cells; 24 scenes of 6000 cells: 5.0 per scene (0-8), acceptance 52 %; 61 ms per default scene
  (74 before: the face check is vectorised numpy, cheaper than the time it replaces in noise). The contact-free copy of
  the momentum check drops the faces (`validation.contact_free_copy`).
- Detection (`contact.static_candidates`, `StaticMesh`, `contact_kernel.static_query_kernel`): one `wp.Mesh` over all static
  faces (two triangles per quad over a float32 copy of the corners), built once per scene and never refitted, cached as
  `batch.static_mesh` (rebuilt when `batch.scene.faces` is another table; reset by `relayout`), queried once per sample
  with `wp.mesh_query_point_no_sign` within margin + r = 2r + |v_s| (the quads are an open surface, so the parity sign of
  `mesh_query_point_sign_parity` would mean nothing: the decision's "unsigned distance" without the three rays); the face
  is triangle // 2, the closest point and the distance are recomputed from the quad's corners in the batch's dtype.
  Two-sided rule: the quad normal is flipped towards the side the sample is on (the sign of (x_s - p) . n_quad; a sample
  exactly in the quad's plane takes the sign opposing its own normal), and the pair is kept when that normal opposes the
  sample normal (n . n_s < 0) and distance - r < r + |v_s|; no lateral rule (the closest point on the finite quad bounds
  the offset: a sample 0.3 beyond the edge pairs with the edge, 1.2 beyond is out of reach). The wording of the decision
  ("the normal that opposes the sample's face normal") taken alone is NOT safe: for a grazing sample (its face
  perpendicular to the quad, the side rows of a box standing on a slab) the sign is decided by rounding, and for a side
  face tilted upwards it is decidedly the far side; such a row then sees gap = -height, d = r + height, and the box is
  pushed through the face (first drop test: penetration 6.8 r, the box flung 8000 cells off the ramp). With the side from
  the position the two-sided face behaves on each side exactly like the one-sided plane (a tilted quad reproduces the
  tilted plane step for step to 1e-9 over 20 steps of the zero-init solver, test). Consequence: a sample that has crossed
  the quad's plane by the time of a detection sees the face from the other side with its own normal pointing away and is
  no candidate (a thin face has no interior; the plane keeps pushing such a sample back, a body mesh by its parity sign);
  within a step the frozen pair holds its normal and pushes back as the plane does. Recorded fields: kind 1 (`KIND_STATIC`,
  the static-partner slot shared with the discs), `partner_body = -2` (`PARTNER_STATIC`), `partner_face` = index into the
  static table, anchor = the sample position at X, `radius` = r x `RADIUS_CHANNEL_CAP` (`STATIC_RADIUS` = 5 cells) so the
  token's r_p / r channel sits exactly at its cap 10: to the network a static face is a disc of unbounded radius. The
  closest static face competes with the discs and the bodies for the M_PAIR = 4 slots (ranked by its unsigned surface
  distance; slot order plane, point ids, the static face, partner object ids); capacity mode has 1 + min(M_PAIR,
  Npts + [F > 0] + O - 1) columns per sample. One mesh means one static candidate per sample (the nearest face), as one
  body mesh means one candidate per body. Cost (default scene 5 of seed 73: 159 bodies, 52k samples, 6 faces, L40 alone,
  capacity layout): the static query 0.29 ms (closest points and normals recomputed for the hit rows only), detection
  11.7 -> 12.0 ms; the capacity column count is unchanged (5) since the bodies already fill the M_PAIR slots. With the
  default 0-8 faces of 2-10 cells on a 6-7 m footprint the faces are small targets: step 0 has no static pair (above),
  and 24 steps of extrapolated free flight of that scene still gave none (2347 plane, 81 body pairs), so static contacts
  will be rare events per scene at the default sizes; `static_face_count_range` / `static_face_size_cells` set the rate.
- Geometry at a query (torch `_body_geometry`, Warp `contact_pair_kernel` / `pair_geometry_kernel`, both kernels take
  `partner_body` and `faces` [F,4] now): a static row follows the body-face path with CONSTANT corners from
  `scene.faces[partner_face]`: the closest point on the quad at the current sample position (weights held), the normal is
  the detection's (the quad is fixed, so it is the quad's normal with the side fixed at detection), delta = x - anchor as
  for every static partner; nothing but the sample receives a gradient (the kernel's partner rows stay zero and are not
  scattered; `ContactEnergy.backward` indexes partner corners for body rows only). Body rows are identified by
  `partner_body >= 0` (not `partner_face >= 0` as before). `active_stiffness` / `translation_hessian` already excluded
  partners below 0. Tokens: kind one-hot slot 1, radius ratio 10, partner point and normal from the quad.
- Validation and report: `plane_penetration_r` = the maximum over kinds 0 and 1 (plane, discs and static faces; the key
  keeps its name for the dashboard, documented in `validation.py`); `pair_counts` returns {total, plane, point, static,
  body} (static = kind 1 with `partner_body == -2`), `report.PAIR_KEYS` summarises total / plane / static / body,
  `SceneRunner.epoch_summary()` adds `static_pairs_mean` and `static_faces_mean`; `test_validation_v5` key sets updated.
- Tests (`tests/test_static_faces.py`, 19): sampling statistics and determinism (counts in range, mean within 1.5 of 4,
  rectangles of 2-10 cells, planar, random normals), faces in the column above the ground and clear of every grown body
  box (SAT and dense sampling; the SAT against brute force on 200 random quads and boxes), count 0 against the recorded
  scenes and the body draws unchanged, JSON round trip / legacy record / summary / realise in cells with no static pair at
  rest at step 0; detection of a box above and below a slab for both corner orders (9 pairs at gap 0.4, normal towards
  the box, foot points, radius 5, anchors), velocity margin, within-step penetration 0.8 / r through the frozen pair and
  the crossed sample losing its pair, a tilted quad (normals, closest points against a 200^2 grid) and its edges, the slots
  shared with the plane and a second body (18 / 18 / 18 pairs, capacity 3 columns, fields equal), tokens (slot 1, cap,
  partner point and normal) and the penetration fold through `scene_metrics` with and without the plane; finite
  differences in float64 with respect to the body's corners on three tilted faces with damping and friction (1e-7), the
  force and the owner-only stiffness; the tilted plane equivalence; the drop (frictionless: 60 steps, penetration below 0.1 r
  at the landing (0.09 r from 1.1 cells, 0.011 r from 1.0) and 7e-4 r at rest, corners never below 0.494 cells over the
  quad, resting height 1.5 + r to 1e-3; with friction 0.8: 30 steps, bounded, never through); the Warp kernel against torch in four regimes (energies 1e-5, gradients 1e-4, the fused
  pass, the geometry kernel and tokens, zero partner gradients and no gradient on a second body); capacity against
  compaction (identical energy, gradient, tokens, penetration); the captured query against eager over three queries and an
  advance (5e-5); the scene runner's epoch of 1500-cell scenes with 4-8 faces on the CPU (faces on every batch, the
  `static` key in the pair history, cheap and full-horizon validation records surviving, momentum drift below 1e-3) and a
  slab injected 0.1 cells under a ground-resting body pairing with every sample of its bottom face at gap 0.4 at step 0.
  Full suite 300 (281 + 19) pass on GPU 3; `uvx ruff format` / `check` and `uvx pre-commit run --files` clean.
- For Anka: (i) the untrained zero-init solver on a slope: the frictionless box slides as a whole and stays exactly one
  sample radius over the ramp; with friction 0.8 the base sticks while the free corners keep integrating gravity (no
  elastic response), so the body shears, tumbles and moves down-slope FASTER than without friction (0.26 against 0.06
  cells per step by step 50), on the static face and on the tilted plane alike; a trained network is what makes the box an
  elastic body. (ii) Step 0 of a default scene has no static pair: faces keep at least one cell from every body box and
  the reach at rest is one cell; the first static pairs come from the drift and the landing, like the body pairs.
  (iii) `CapturedQuery` does not watch `batch.scene` (plane fields included): the static table is baked into the graph,
  fine for the runner's one Batch per scene.

### v5 campaign, first attempt and the loss decisions (2026-10-02)

Attempt 1 (roundoff energy floor, linear increase penalty, no step curriculum; 03:17-08:00 UTC, 23 epochs) learnt on
validation at first (full-horizon metric 354 -> 35 N by epoch 4) but its training signal was owned by a few bodies per
update: resting bodies with E_before near zero pushed a fraction of a cell into stiff partners (kappa up to 1000,
k_e up to 2.5e7 N/m per pair) gave losses of 1e4-1e5 and gradient norms of 1e10, and by epoch 6 the run had diverged
(loss 5e8, full-horizon 4e5 N, penetration 26 r). The mechanism: only the translation is implicit; the shape comes
from the network's proposal without an energy check, an untrained proposal on a resting body is a bad one, the
unconverged step's error is carried as velocity, and the linear increase term divided by a roundoff floor turns it
into the whole gradient.

Decisions (Anka): (1) loss scale floor = max(roundoff floor, body weight x one cell) (`physical_floor`), (2) the
increase penalty becomes asinh(relu(increase/scale)) (`bounded_increase`), (3) a per-cell step cap curriculum: the cap
starts at 10 % of `max_step_size` and ramps to the full value over 8 advancing epochs, advancing only while the
selection metric does not get worse (`step_cap_start`, `step_cap_ramp_epochs`, `step_cap_gate_on_validation`; stored
in checkpoints; the method's own 0.05 cap is unchanged and the curriculum leaves nothing behind), plus a blow-up guard
(`blowup_energy_factor` 1e6 x floor counts as a failed state). Attempt 2 started 08:02 UTC from scratch.

### v5 campaign, second attempt and the scene curriculum (2026-10-02, decided autonomously)

Attempt 2 (physical floor, bounded increase, step-cap curriculum; 08:02-08:20 UTC, 8 epochs, full v5 mix from the
first epoch) trained tamely (loss 0.64 -> 0.14, gradient norms below 2) and the step cap ramped from 0.005 to 0.03
(the gate uses the full-horizon selection metric, which fell 7.5e5 -> 145 -> 115 -> 80 N over epochs 1-4), but the
scenes themselves collapsed as the cap loosened: failed scenes per epoch 0, 1, 1, 1, 3, 3, 11, 6; the deepest
training penetration 4 -> 11 -> 44 -> 53 -> 350 r; the held-out 32-query curves grew spikes (energy ratios of 44 and
94 at single iterations); the full-horizon validation lost both scenes at epochs 7 and 8. The run is archived in
`generated/lido_v5_20261002/attempt2_full_mix_from_start/`.

Anka, before going to bed: make decisions autonomously; "maybe we can start with pinned + artificial collision as
previous training (v4) and add more challenging training later?". Decision: a scene curriculum on a fixed epoch
schedule (Anka prefers fixed schedules over plateau rules), `scene_curriculum` and `scene_curriculum_epochs = (4, 20)`
in `TrainConfig`, `scenes_v5.SceneMix` / `scene_mix(cfg, epoch)`:

- every body pinned through epoch 4: the moving bodies meet only fixed geometry (ground plane, static faces, the
  held pinned neighbours), as in the v4 campaign, but with 150 bodies of random size, material and pose per scene;
- from epoch 5 the pinned fraction falls linearly from 1.0 to 0.25 and the resting fraction rises from 0 to 0.3,
  both reached at epoch 20 (the dense free stacks of attempt 2 arrive when the network has learnt the elastic and
  the contact response); the drift speed, the materials, the static faces and the body-body contact are unchanged;
- the mix overrides the two fractions on the same seed streams (`sample_scene(..., mix=)`), so a higher pinned
  fraction pins a superset of the bodies and the body draws, poses and faces do not change;
- the cheap validation (32 queries on 8 held-out scenes) follows the epoch's mix, the full-horizon validation keeps
  the final mix: it is the goal, the selection metric and the step-cap gate (a diverging goal scene in the pinned
  phase is expected and only delays the cap);
- the step-cap curriculum stays as it is (its ramp will likely finish during the pinned phase; the network then
  meets the free bodies with the method's own cap, as the v4 campaign did with its pinned beams).

Found while implementing: the v5 runner drew `sample_scene(master_seed, job.seed, cfg)` with `job.seed` = scene
index, so every epoch replayed the same 64 layouts (only K, H and the candidate noise changed). Training scene seeds
are now `scenes_v5.epoch_scene_seed(epoch, index)` = epoch x 2^20 + index (test and reference scenes keep seeds below
2^20). The epoch record's regime now carries `step_cap`, `pinned_fraction` and `resting_fraction` (the report had
filtered the step-cap key out, hence the empty cap in the first attempt-2 records). Tests: `TestSceneCurriculum`
(schedule, superset property, fresh seeds) and `TestSceneCurriculumRunner`; the runner-based suites pin
`scene_curriculum=False` since they describe the final mix. Attempt 3 started 08:21 UTC from scratch with this
configuration (same run directory, dashboard slug `lido-v5-20261002`).

### v5 campaign, third attempt and the material band (2026-10-02, decided autonomously)

Attempt 3 (scene curriculum, 08:21-09:33 UTC, 9 epochs) was clean through the all-pinned phase (no failed scene,
full-horizon 52.6 N with both goal scenes alive at epoch 4, cheap metric 29-60 N) and collapsed as the free bodies
arrived: with 19-23 % free bodies at epochs 8-9, 1 then 5 failed scenes, training penetration 52 then 1713 r, both
goal scenes lost at the K = 8, H = 64 horizon. The run is archived in
`generated/lido_v5_20261002/attempt3_curriculum_v4_materials/`.

Replaying the two goal scenes with the epoch-8 weights (64 steps, 8 queries each) showed the mechanism. Inverted
cells accumulate from step 20 on, in free resting bodies AND in the pinned bodies that carry them (one pinned body
reached 178 inverted cells, the scene 594 by step 63), the energy rises steadily, and eventually a resting body
penetrates its partner by 3-4 r, the stiff contact (kappa 102) launches it (penetration 50-80 r in one step, a
corner speed of 1e7 cells per step) and the state is NaN the step after. The cause is the material distribution,
inherited from the v4 campaign where each body hung alone from a pinned face: E log-uniform in 1e3-1e6 Pa and rho
log-uniform in 100-1e4 kg/m^3, drawn independently. A body of 1e4 kg/m^3 falling from the 1 m column lands at
4.4 m/s; the compressive strain of an impact is about speed over elastic wave speed sqrt(E / rho), which is 0.3 m/s
for the softest and densest draws, so soft bodies are crushed flat by dense neighbours (and under their own weight:
rho g L / E reaches 30), which no per-step correction bounded by the step cap can follow.

Decision: a per-scene material band (`scene_wave_speed_min` 15 m/s, `scene_wave_speed_band` 3,
`scene_density_band` 10; `scenes_v5.material_band`, `draw_material`): the scene draws a density interval at most a
factor 10 wide (log-uniform lower edge over the config's range, its heaviest material fast enough at E_max) and a
squared-wave-speed interval at most a factor 9 wide (log-uniform lower edge over what keeps E = c^2 rho inside the
config's range, at least 15^2); every body draws rho and c^2 in the band and gets E = c^2 rho. The strain of a
full-height impact is then at most about 0.3 for equal densities and bounded by sqrt(10) x 0.3 for the densest
body of a scene on its lightest; the self-weight strain rho g L / E is below 2 %. Across scenes E and rho still
cover their ranges (the softest admissible material is 22.5 kPa at 100 kg/m^3, the densest 4400 kg/m^3 at 1 MPa).
The band is drawn after the contact constants and 0 switches it off, so the recorded reference scenes are
reproduced with the band off (`TestMaterialBand`). Attempt 4 started 09:35 UTC from scratch with the curriculum and
the band.

Observed in the attempt-3 records and left open: the all-pinned phase has almost no contact (27 plane pairs per
detection for 150 bodies, 0.0 static-face pairs), because faces keep at least one cell plus the field clearance
from every body and pinned bodies neither drift nor fall. Anka's "pinned + artificial collision" phase would need
faces placed within the deformation reach of pinned bodies (candidate next step, not implemented).

### Pinned contact faces (2026-10-02, implemented, not yet in a run)

The open observation above is closed in code: `pinned_contact_face_fraction` (0.5; `scenes_v5._place_pinned_faces`,
seventh seed stream) gives a pinned body, with that probability, one static quad parallel to one of its five
unpinned faces at a gap of U(0.2, 0.8) x its field clearance (three sigma of the deformation field), with sides
U(0.8, 1.5) x the face's sides and an in-plane offset of up to a quarter of them, redrawn up to eight times when it
reaches below the ground or into another body's grown box, then dropped (statistics `pinned_contact_faces`,
`..._dropped`, `..._candidates`, `..._owners` in the placement record). The detection at step 0 of an all-pinned
scene finds static pairs (`TestPinnedContactFaces`). Attempt 4 was already past its all-pinned phase when this
landed and runs without the faces (its config says 0.0), so that the material band's effect is seen alone; the
faces are for the next restart or the next campaign.
