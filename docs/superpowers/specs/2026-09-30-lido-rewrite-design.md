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
