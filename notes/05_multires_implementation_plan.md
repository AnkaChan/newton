# Multi-Res Smoke Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the hierarchical S-residual predictor (design `notes/04`, decisions D7.1–D7.12) and run the stage-0 smoke test: audits 0a/0b, then one PoC training run on the 4k article, judged against the 4.5 mm signal bar.

**Architecture:** A deterministic tet-graph hierarchy (greedy face-adjacency aggregation, quotient graphs) feeds per-level shared-weight networks (edge-conditioned message passing + ancestor context); per-level Hencky residuals compose as `S* = exp(H_t + sum_l P_l dH_l)`; the existing decoder, trainer, and eval harnesses are unchanged.

**Tech Stack:** PyTorch (fp32 net / fp64 decoder), NumPy for offline hierarchy construction, existing `research/principal_stretch` package.

## Global Constraints

- All work in `research/principal_stretch/` + `notes/` only; no engine files (D6.1).
- Tests use **unittest**, never pytest (repo rule); run via `uv run python -m research.principal_stretch.tests.<name>`.
- Lint before every commit: `uvx pre-commit run --files <files>`; hooks auto-reformat — re-read files after.
- Commit identity `AnkaChen <ankac@nvidia.com>`; imperative subjects; commit at every task boundary.
- GPU work: `source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh <name>` first; long runs in tmux (never background Bash >10 min); sync Cursor tmux profiles on create/kill.
- Never supervise S (or H) with plain L2 (D5.3) — all quality metrics are decoded positions.
- No positional encodings / per-node weights / per-article learned structure (D5.4, D7.9).
- Decoder (`torch_solver.py`), loss, warm start, curriculum: **do not modify** (except the audited `train.py --hier` switch).
- Datasets: reuse `data/{train,val}.npz` (toy) and `data/{train,val}_4k.npz`. No new data generation in this plan.

---

# Part I — Concepts & Terminology

| term | meaning |
|---|---|
| fine graph `G_0` | nodes = tets of the article; edges = face adjacency (two tets sharing exactly 3 vertices); this is today's `face_adj` |
| cluster (aggregate) | a *connected* set of ~8 level-l nodes, formed by greedy growth along graph edges only — never through space |
| assignment `a_l` | array mapping each level-l node to its cluster id at level l+1; `a_l(e)` is e's **parent** |
| ancestor chain | parents of parents: `a_1(e), a_2(a_1(e)), ...` — one node per coarse level for every fine tet |
| quotient graph `G_{l+1}` | nodes = clusters; edge A–B iff any member of A is adjacent to any member of B |
| level | one (graph, features, network) stratum; level 0 = tets, level 3 = ~34 nodes on the 4k article |
| restriction (pooling) | computing a cluster's features from its members' (volume-weighted means / sums) |
| prolongation `P_l` | mapping a coarse per-cluster field to a fine per-tet field by partition-of-unity blending |
| partition of unity (PoU) | per-tet blend weights over {own cluster + its *adjacent* clusters} that sum to 1 — smooths the piecewise-constant field without ever crossing a topological gap |
| intensive / extensive | intensive = concentration-like, pools by mean (stretch, material, pin fraction); extensive = total-like, pools by sum (force) — a poke must not be diluted away (D7.4) |
| Hencky strain `H` | `H = log(S)`, matrix log of the SPD right-stretch; residuals live and add here; `exp` maps back, guaranteeing SPD (D7.11) |
| log-Euclidean mean | the mean of SPD tensors computed as `exp(mean(log(S_i)))` — the geometrically sound average; equals pooling H linearly |
| polar frame `R_e` | rotation from `polar(F_e)`; edge geometry is expressed in the *receiving* node's polar frame for SE(3) invariance (D7.5) |
| edge features `e_ab` | 7 per-edge numbers: relative centroid offset in the receiver's frame, edge stretch, relative rotation (axis-angle) — gives MP the bending information per-node S cannot carry |
| message passing (MP) | edge-conditioned neighbor aggregation, 2 rounds per coarse level; weights shared across a level's nodes (D7.6) |
| ancestor context `z_e` | concatenated post-MP hidden states of a node's ancestor chain — the far field, pre-digested per scale, as *input* (D7.10) |
| per-level head | small MLP mapping (hidden, context) to a 6-component Hencky residual `dH_l`, tanh-bounded, zero-initialized |
| composition | `S* = exp(H_t + dH_0 + sum_l P_l dH_l)` — one fine S\* field for the unchanged decoder |
| flat net | the current 28-dim per-tet MLP; the baseline to beat (4k single-step 5.73 mm) |
| decoder oracle / floor | decode ground-truth S with the standard protocol: 1.53 mm at 4k — the best any predictor could do |
| kNN feature audit (0a) | cross-trajectory kNN regression floor, re-estimated per candidate feature set, before training (D7.10) |
| composition oracle audit (0b) | reconstruct GT stretch fields through the hierarchy with each composition rule, score decoded positions (D7.11) |
| signal bar | PoC success threshold: 4k single-step <= 4.5 mm (>= 20% below flat) (D7.12) |

# Part II — Method & Formulas

The method wraps a multi-scale predictor around the *unchanged* ARAP
decoder. One sentence of design intent: the flat per-tet network fails
because a tet cannot see the far field (D5.2), so we give every tet a
pre-digested view of the whole body — built along the material, summarized
per scale, injected as input, and returned as per-scale stretch corrections
that compose into a single SPD field the decoder already knows how to
consume. Each subsection below is one design element, with the exact
formulas the implementation is bound to. Every element answers a specific
measured failure, cited inline.

All tensors are material-frame; `v_e` = rest volume, `c_e` = current
centroid, `c_e^0` = rest centroid of node e; `A, B` denote clusters.

## II.1 Structure: a hierarchy built along the material (F1, F2)

Two parts of an object that touch in space but not through material must
never share a coarse node — otherwise the network learns spatial shortcuts
that break exactly on articles like the U-bar (notes/01, D7.2). So the
hierarchy is grown on the tet **face-adjacency graph**, and the strength of
a connection is how much material actually joins the two cells:

**F1 — coarsening edge strength (level 0):** `w_ab = area(shared face of tets a, b)`; higher levels: F2 output.

Clusters aggregate ~8 strongly-connected nodes (algorithm A1); the coarse
level's own connectivity is inherited, not re-derived — cluster A talks to
cluster B exactly as much as their members touch:

**F2 — quotient edge weight:** `W_AB = sum of w_ab over all edges (a, b) with a in A, b in B`.

Applying this recursively gives 17280 -> ~2160 -> ~270 -> ~34 nodes on the
4k article: by the third level, two message-passing rounds see the entire
body.

## II.2 State at every scale: pooling and cluster kinematics (F3, F4, F5)

A coarse node must summarize its members' state without destroying the
signals the far field needs. Two physically different kinds of quantity
pool differently. Intensive quantities (stretch, material, pin fraction)
are concentration-like — a cluster's state is its members' volume-weighted
mean; for stretch this mean is taken in log space (the log-Euclidean mean,
the geometrically sound average of SPD tensors — II.5):

**F3 — intensive pooling (log-Euclidean for stretch):**
```
H_A = ( sum_{e in A} v_e * H_e ) / ( sum_{e in A} v_e )        same rule for mu, lam, pin fraction
```

Forces are extensive — totals, not concentrations. A 50 N poke on one
vertex, mean-pooled over 512 tets, dilutes to numerical noise at exactly
the level meant to see it (D7.4). The sum channel preserves it at every
scale:

**F4 — extensive pooling:** `fsum_A = sum_{e in A} f_e` (kept alongside the mean channel); `v_A = sum v_e`; feature uses `log(v_A / mean_cluster_volume)`.

Coarse levels also need geometry — where the cluster is and how it is
oriented — for the edge features of II.3:

**F5 — cluster kinematics (per frame):**
```
c_A = ( sum v_e c_e ) / ( sum v_e )        F_A = ( sum v_e F_e ) / ( sum v_e )        R_A = polar(F_A)
```

## II.3 Communication: edge geometry and message passing (F6, F7)

Per-node S is rotation-free by design — which means it is blind to
*relative rotation between neighbors*: two adjacent tets in a bending beam
can each be unstretched while strongly rotated against each other, so pure
bending is invisible to every per-node feature (D7.5). Edges carry exactly
that missing information. To keep the whole pipeline SE(3)-invariant, every
edge vector is expressed in the *receiving* node's polar frame — rotate the
world and R_a co-rotates, leaving the features unchanged:

**F6 — edge features (receiver a, sender b; l0_ab = |c_b^0 - c_a^0|):**
```
e_ab = [ R_a^T (c_b - c_a) / l0_ab ,          3   offset in receiver's frame
         |c_b - c_a| / l0_ab - 1 ,             1   edge stretch
         axial( LogSO3( R_a^T R_b ) ) ]        3   relative rotation (bending)
```

Messages are aggregated with the material-connection weights from F2, so a
neighbor's influence scales with how much face area joins it:

**F7 — message passing (one round; normalized weights `wn_ab = W_ab / sum_b W_ab`):**
```
m_ab = MLP_edge([ h_a , h_b , e_ab ])
h_a' = MLP_node([ h_a , sum_b wn_ab * m_ab ])
```

Two rounds per coarse level suffice because reach comes from the hierarchy
(II.1), not from MP depth — that is the whole point of multi-res.

## II.4 Prediction: ancestor context and per-level heads (F8, F9)

Weights are shared across a level's nodes (D7.6), so each level's map
`features -> output` must be well-posed: identical inputs must never
require different outputs. Pooling alone sends information *up*; without a
downward path the fine level would still face the one-to-many problem that
broke the flat net. The fix (D7.10, Anka) is to hand every node the far
field as *input* — the post-MP hidden states of its ancestor chain, one
small vector per scale, fetched by pure gathers:

**F8 — ancestor context of node e at level l:** `z_e = concat( h'_{parent(e)}, h'_{grandparent(e)}, ... )` — post-MP hidden states, gathered (no extra graph work).

Each level then predicts a *correction at its own scale*. Zero-initialized
last layers make the untrained model predict "stretch unchanged" — the
correct prior for one dynamics step (D2.5) — and the tanh bound keeps any
single level from emitting extreme residuals:

**F9 — per-level head (zero-init last layer):**
```
dH_l = delta_l * tanh( MLP_head_l([ h'_l , z_l ]) )      delta_0 = 0.6, delta_{l>=1} = 0.3
```

## II.5 Composition: one SPD stretch field from many scales (F10, F11)

Stretches do not add (Anka, D7.11): composition of finite stretches is
multiplicative, and naive addition can leave the SPD cone. Residuals
therefore live in **log space** (Hencky strain), where composition is
addition, the small-strain limit is exact, commuting strains compose
exactly, and `exp` guarantees a legal SPD output no matter how the level
outputs stack. Rotations never enter: right-stretch tensors share the one
material basis, and all rotation handling stays in the decoder. The
composition audit (Task 3) measured the remaining non-commuting error at
< 0.5% of decoded error at our strains — negligible.

Coarse corrections reach the tets through a partition-of-unity blend over
{own cluster + *adjacent* clusters} — smooth instead of blocky (the
multires-vbd lesson), and incapable of bleeding across a topological gap
because adjacency is material adjacency:

**F10 — PoU prolongation (static, rest-space, precomputed):** for tet e with parent A0 and candidates `C(e) = {A0} union neighbors(A0)`:
```
omega_eA = exp( -|c_e^0 - c_A^0|^2 / sigma_l^2 ) / (normalizer over C(e)),    sigma_l = mean cluster diameter at level l
(P_l dH_l)(e) = sum_{A in C(e)} omega_eA * dH_l[A]        for l >= 2, prolong level-by-level down to tets
```

**F11 — composition:** `S*_e = exp( H_t,e + dH_0,e + sum_{l>=1} (P_l dH_l)(e) )`.

The decoder consumes this fine S\* field exactly as it consumed the flat
net's — the entire method is upstream of an unchanged solver (§3.6).

## II.6 Numerics: exact derivatives for the new maps (F12, F13)

The composition sits inside the training loss's gradient path, so its
derivatives must be exact — approximate gradients through the
representation are what sank the original method (D1.4). The log/exp of
symmetric matrices gets the closed-form Daleckii–Krein backward:

**F12 — SPD log/exp with exact backward (Daleckii–Krein):** for symmetric `M = U diag(lam) U^T` and `f in {log, exp}`:
```
f(M) = U diag(f(lam)) U^T
backward:  grad_M = U ( G .* (U^T grad_out U) ) U^T
           G_ij = ( f(lam_i) - f(lam_j) ) / ( lam_i - lam_j )   if |lam_i - lam_j| > eps(dtype)
           G_ii = f'(lam_i);  for |lam_i - lam_j| <= eps(dtype) use f'((lam_i+lam_j)/2)
           eps(float64) = 1e-9;  eps(float32) = 1e-4   [amended after Task-1 review: with
           eps = 1e-9 the fp32 exp numerator quantizes at ulp ~1.2e-7, zeroing or spiking
           gradient components in the near-rest regime; f'(mid) error is O(gap^2/24),
           safe at 1e-4]
```

The relative-rotation edge feature needs a guarded SO(3) log — exact where
physical, loud where not:

**F13 — SO(3) log (axis-angle), with small-angle guard:**
```
theta = arccos( clamp( (tr(R) - 1)/2 , -1, 1 ) )
axial(LogSO3(R)) = theta / (2 sin theta) * [R32-R23, R13-R31, R21-R12]
theta < 1e-4:  factor -> 1/2 (Taylor);  theta near pi is out of range for adjacent tets — assert, don't handle
[amended after Task 7: so3_log_axial gains keyword-only saturate=False; the ONE sanctioned
 saturate=True call site is hier_model's edge features, where untrained-net K=4 transients
 exceed 3 rad (theta clamps to 3.0, output bounded in [0,3], gradient bounded; fail-loud
 default everywhere else). Stage-1 work item: per-level saturation-rate telemetry — persistent
 nonzero rates at coarse levels at convergence mean real near-pi cluster rotations and require
 a pi-safe featurization (e.g. 6D rotation) instead of the clamp]
```

## II.7 Evaluation: everything is scored through the decoder (F14)

S-space norms anti-correlate with decoded quality (D5.3) — the decoder's
pullback metric on stretch space is extremely anisotropic — so every
quality number in this plan, including both audits, is a decoded position
error:

**F14 — audit scores (both audits, per D5.3):** per-vertex mean of `| decode(S*_candidate) - x_gt |`, decoded with the standard protocol (10 iterations, inertial warm start), against the recorded frames.

# Part III — Algorithms

**A1 — greedy topological aggregation (offline, per level):**
```
input:  adjacency lists adj, edge weights w, target size m = 8
unassigned = all nodes
while unassigned not empty:
    seed = unassigned node with the fewest unassigned neighbors (tie: lowest index)
    C = {seed}
    while |C| < m:
        frontier = unassigned nodes adjacent to C
        if frontier empty: break
        add argmax over frontier of (total edge weight into C)   (tie: lowest index)
    emit C
post-pass: every cluster with |C| < m/2 is merged into the adjacent
           cluster with the largest total crossing weight
guarantee: clusters are connected by construction (growth only via edges)
```

**A2 — quotient graph:** iterate all level-l edges; for `a_l(a) != a_l(b)` accumulate `W[a_l(a), a_l(b)] += w_ab`; emit padded neighbor/weight arrays sorted by weight descending.

**A3 — per-frame hierarchy state (bottom-up):**
```
level 0: F, R, S, H, centroids from x (existing compute_S_from_x machinery); feat28 as today
for l = 1..L:
    pool per F3/F4/F5 using precomputed index_add scatter over assign
    R_A = polar(F_A); build e_AB per quotient edge per F6
```

**A4 — forward pass (coarse-to-fine so ancestor context exists):**
```
pool all levels (A3)
for l = L down to 1:
    z_l  = gather(h'_{l+1..L}) along ancestor chain      (empty for l = L)
    h'_l = MP_l x2 ( [feat_l, z_l], quotient graph l )
    dH_l = head_l([h'_l, z_l])
z_0  = gather(h'_{1..L})
dH_0 = head_0([feat28, z_0])
S*   = exp( H_t + dH_0 + sum_l P_l dH_l )     -> existing solve()
```

**A5 — training:** unchanged trainer loop (`train.py`); `--hier` swaps the flat net's forward for A4. Noise, curriculum, K-rollout, loss all as today. PoC recipe: pos + residual + inertial, K=4, 4000 steps, batch 8.

**A6 — composition oracle audit (0b, no training):**
```
for each val frame: H_gt = log(S_gt at t+1)
telescoping decomposition:  r_L = pool_L(H_gt);  for l < L:  r_l = pool_l(H_gt) - P_{l+1}(r_{l+1} accumulated)
reconstructions:
    log rule:     S_hat = exp( sum_l P_l(r_l) )
    linear rule:  same telescope computed on (S_gt - I) instead of H_gt; S_hat = I + sum
score per F14 at truncation depths (coarsest-only, +level2, +level1, full)
```
Reading: the gap between rules at each depth is the composition error; the
gap between full-depth and oracle is the hierarchy's representation loss.

**A7 — kNN feature audit (0a):** existing `diag_knn_floor.py` protocol with feature arms per query tet: (i) base 28; (ii) + own edge features (mean over incident edges); (iii) + oracle ancestor context (pooled *feature* vectors of the ancestor chain, from GT state — no network); floors must order (i) > (ii) > (iii), and (iii) should approach the oracle.

# Part IV — Implementation Tasks

File map (all under `research/principal_stretch/`):

| file | responsibility |
|---|---|
| `spd_log.py` (new) | batched symmetric 3x3 `sym_log` / `sym_exp` with F12 backward; `so3_log_axial` (F13) |
| `hierarchy.py` (new) | A1/A2 construction; `Hierarchy` container; pooling + prolongation operators (F3/F4/F10) |
| `hier_model.py` (new) | level features + edge features (F6), MP (F7), ancestor context (F8), heads (F9), composition (F11); class `HierStretchNet` |
| `diag_composition.py` (new) | audit 0b (A6) |
| `diag_knn_floor.py` (modify) | `--feature-arm` option (A7) |
| `train.py`, `eval_singlestep.py`, `rollout.py` (modify) | `--hier` switch, checkpoint metadata |
| `tests/test_spd_log.py`, `tests/test_hierarchy.py`, `tests/test_hier_model.py` (new) | unittest suites |

### Task 1: `spd_log.py` — batched SPD log/exp + SO(3) log

**Files:** Create `research/principal_stretch/spd_log.py`, `research/principal_stretch/tests/test_spd_log.py`

**Interfaces — Produces:**
```python
def sym_log(S: torch.Tensor) -> torch.Tensor   # (..., 3, 3) SPD -> (..., 3, 3) symmetric, differentiable
def sym_exp(H: torch.Tensor) -> torch.Tensor   # inverse map, differentiable
def so3_log_axial(R: torch.Tensor) -> torch.Tensor  # (..., 3, 3) rotation -> (..., 3) axis-angle vector
```

- [ ] **Step 1: failing tests** — `tests/test_spd_log.py` (unittest), covering:

```python
class TestSpdLog(unittest.TestCase):
    def test_round_trip(self):        # sym_exp(sym_log(S)) == S to 1e-12 fp64, random SPD batch incl. anisotropy 17:1
    def test_identity(self):          # sym_log(I) == 0, sym_exp(0) == I
    def test_repeated_eigenvalues(self):  # isotropic S = s*I and near-isotropic (lam spread 1e-9): finite grads, correct values
    def test_gradcheck(self):         # torch.autograd.gradcheck on both maps, fp64, small batch
    def test_matches_scipy(self):     # values vs scipy.linalg.logm on 50 random SPD matrices
    def test_so3_log(self):           # axial vs known axis-angle constructions; small-angle (1e-6) Taylor branch; gradcheck
```

- [ ] **Step 2:** run, verify all fail (module missing).
- [ ] **Step 3:** implement — forward `torch.linalg.eigh` + `f(lam)`; backward as `torch.autograd.Function` with the F12 divided-difference matrix (`eps = 1e-9` threshold); `so3_log_axial` per F13 with the theta<1e-4 branch and an assert for theta > 3.0.
- [ ] **Step 4:** run tests → all pass (fp64; fp32 tolerances 1e-5 in a separate case).
- [ ] **Step 5:** lint, commit `"Add batched SPD log/exp and SO(3) log with exact backward"`.

### Task 2: `hierarchy.py` — construction + operators

**Files:** Create `research/principal_stretch/hierarchy.py`, `research/principal_stretch/tests/test_hierarchy.py`

**Interfaces — Produces:**
```python
@dataclasses.dataclass
class Hierarchy:
    levels: list[Level]     # per level l>=1: assign (N_{l-1},) int64, adj (N_l,K) int64 (-1 pad),
                            # w_adj (N_l,K) float, vol (N_l,), c0 (N_l,3) rest centroids,
                            # pou_idx (N_{l-1},P) int64, pou_w (N_{l-1},P) float  (P_l operator rows)

def build_hierarchy(tets: np.ndarray, rest_q: np.ndarray, n_levels: int = 3, target: int = 8) -> Hierarchy
def pool_mean(x: torch.Tensor, assign: torch.Tensor, vol: torch.Tensor) -> torch.Tensor   # F3 (works on any trailing shape)
def pool_sum(x: torch.Tensor, assign: torch.Tensor) -> torch.Tensor                        # F4
def prolong(y: torch.Tensor, pou_idx: torch.Tensor, pou_w: torch.Tensor) -> torch.Tensor   # F10, one level down
```

- [ ] **Step 1: failing tests**:

```python
class TestHierarchy(unittest.TestCase):
    def test_partition(self):          # every node assigned exactly once; cluster sizes in [target/2, 2*target] after post-pass
    def test_cluster_connectivity(self):   # BFS inside each cluster over level-(l-1) adjacency reaches all members
    def test_topology_respected(self):     # two disjoint 4x2x2 boxes in one mesh: no cluster (any level) spans both;
                                           # quotient graphs stay disconnected  <- the U-bar property, miniaturized
    def test_quotient_symmetric(self):     # A in adj[B] iff B in adj[A]; weights equal
    def test_pou_rows_sum_to_one(self)
    def test_pool_matches_loop(self)       # pool_mean/pool_sum vs explicit python loop on random data
    def test_level_sizes_4k(self)          # 17280 -> within [1500,3000] -> [180,400] -> [20,60]
```

- [ ] **Step 2:** run, verify fail. **Step 3:** implement A1/A2 in NumPy (deterministic: documented tie-breaks), operators in torch (`index_add_` / gather). **Step 4:** tests pass (toy mesh + synthetic two-box mesh built inline with `build_model`-independent numpy tets). **Step 5:** lint, commit `"Add topological tet-graph hierarchy with pooling and PoU prolongation"`.

### Task 3: `diag_composition.py` — oracle audit 0b

**Files:** Create `research/principal_stretch/diag_composition.py`

**Interfaces — Consumes:** `Hierarchy`, `sym_log`/`sym_exp`, `ts.solve`, dataset npz. **Produces:** printed table + JSON (`artifacts_fix/composition_audit_{toy,4k}.json`): rows = truncation depth, columns = rule (linear / log), values = F14 decoded error; plus the oracle row.

- [ ] **Step 1:** implement A6 (numpy/torch, no autograd needed).
- [ ] **Step 2:** run on toy + 4k val (GPU claim; ~minutes).
- [ ] **Step 3:** record numbers into the report and `notes/00` D7.11 evidence cell. Gate: **log rule <= linear rule at every depth**, full-depth log within 2x of oracle — else stop, the decomposition is lossier than the ceiling we chase; re-examine before training.
- [ ] **Step 4:** lint, commit `"Add composition oracle audit (linear vs log-space reconstruction)"`.

### Task 4: `hier_model.py` — features, MP, heads, composition

**Files:** Create `research/principal_stretch/hier_model.py`, `research/principal_stretch/tests/test_hier_model.py`

**Interfaces — Consumes:** Task 1 + 2 exports, `build_features`/`compute_S_from_x`/`polar_rotation`. **Produces:**
```python
class HierStretchNet(nn.Module):
    def __init__(self, hierarchy: Hierarchy, in_dim: int = 28, hidden: int = 64,
                 mp_rounds: int = 2, delta_fine: float = 0.6, delta_coarse: float = 0.3): ...
    def forward(self, state: SolverState, x_t, x_prev, f_ext, feat28, S_t) -> torch.Tensor  # S* (..., T, 3, 3)
```
(fine-level `feat28` and `S_t` come from the existing pipeline so the trainer integration stays thin; the module computes F/R/c pooling internally per A3/A4.)

- [ ] **Step 1: failing tests**:

```python
class TestHierModel(unittest.TestCase):
    def test_zero_init_identity(self):   # freshly built net: forward returns S_t exactly (exp(log S_t)), tol 1e-5 fp32
    def test_se3_invariance(self):       # apply random world rotation+translation to (x_t, x_prev): S* identical to 1e-4
    def test_poke_visible_at_coarse(self):  # single-vertex force: sum-channel at every level nonzero where ancestors of poked tet are; mean-channel near zero at level 3
    def test_grad_reaches_all_levels(self): # decoded-position loss backward: every head_l has nonzero grad
    def test_batched(self):              # leading batch dim (B=3) matches per-sample loop to 1e-6
```

- [ ] **Step 2:** run, fail. **Step 3:** implement (A3 pooled state, F6 edge features, F7 MP x2 per coarse level, F8 gathers, F9 heads, F11 composition via Task-1 ops). **Amendment (audit-0b finding):** GT data contains transiently inverted tets (S with a negative eigenvalue, no real log — ~3-10% of frames contain at least one). Every `sym_log` input goes through an SPD floor first: `spd_floor(S, lam_min=0.05)` — eigh, clamp eigenvalues to >= lam_min, reconstruct; add to `spd_log.py` with a round-trip + gradient test. The floor is inactive on healthy tets (eigenvalues ~O(1)). **Step 4:** tests pass. **Step 5:** lint, commit `"Add hierarchical stretch predictor (edge-MP, ancestor context, Hencky composition)"`.

### Task 5: audit 0a — kNN feature arms

**Files:** Modify `research/principal_stretch/diag_knn_floor.py` (add `--feature-arm {base,edge,ancestor}`)

- [ ] **Step 1:** implement arms per A7 (edge features per F6 averaged over incident edges; ancestor arm = concat pooled level-1..3 *feature* vectors along the ancestor chain, oracle-computed from GT state via Task-2 pooling).
- [ ] **Step 2:** run all three arms on toy and 4k (same pool/query protocol as before, `--no-same-tet` on 4k).
- [ ] **Step 3:** record: floors must order base > edge > ancestor at 4k, with the ancestor arm moving decisively toward the 1.53 mm oracle. Gate per D7.10: if the ancestor arm does not drop the floor, the feature set cannot carry the far field — fix features before Task 6.
- [ ] **Step 4:** lint, commit `"Add feature-arm variants to the kNN floor audit"`.

### Task 6: trainer + eval integration

**Files:** Modify `research/principal_stretch/train.py` (flag `--hier`, construct `Hierarchy` from the dataset topology, instantiate `HierStretchNet`, save hierarchy config in the checkpoint), `eval_singlestep.py` and `rollout.py` (rebuild `HierStretchNet` when `ckpt["args"]["hier"]`).

- [ ] **Step 1:** overfit gate — 20 toy frames, 500 steps, `--hier`: final windowed loss < 1% of its initial value (gradients flow through exp/log and all levels end to end). **Protocol (pinned after Task-6 execution):** `--solver-iters 100` (at the default 10 the oracle-S floor is already 3.4% of initial — the bar is unreachable for any predictor) and a two-stage lr (400 steps @ 3e-3, then 100 @ 2e-4 via `--init-ckpt`); the load-bearing evidence is the free-dH control (no net, direct dH optimization -> 3e-7), which isolates gradient health from optimizer noise.
- [ ] **Step 2:** toy parity gate — full toy training, accuracy recipe (`--loss pos --residual --warm inertial --max-rollout 4 --steps 4000`): `eval_singlestep` <= 2.5 mm (flat achieves 2.1; toy has little headroom — correctness gate, not signal gate). Log per-level `|dH_l|` means every 50 steps (the dead-level detector from the failure playbook).
- [ ] **Step 3:** lint, commit `"Wire hierarchical predictor into trainer and eval harnesses"`.

### Task 7: the 4k PoC (stage 0c)

- [ ] **Step 1:** launch in tmux (`pss-hier-poc`, GPU claim, profile sync): `--hier` on `data/train_4k.npz`, accuracy recipe as Task 6, `--batch 8`; tee to `AI-Logs/.../2026-08-XX-hier-poc.log`; Monitor with step/error/Traceback filter; ~2–3 h.
- [ ] **Step 2:** evaluate: `eval_singlestep --data data/val_4k.npz` + the standard anchor table (flat 5.73 / kNN 8.32 / oracle 1.53 mm); record an 18-frame rollout for information only (non-gating, Law 1).
- [ ] **Step 3: decision per D7.12** — <= 4.5 mm: write the report, update `notes/00` (D7.x statuses -> ACTIVE), proceed to stage-1 controls. Worse, with alive levels and passed audits: exactly one debug loop (per-level magnitudes, audit re-check, level dropout if coarse levels dead), then stop and write up honestly.
- [ ] **Step 4:** report to `AI-Logs/Newton/tasks/PrincipalStrecchSolver/2026-08-XX-hier-poc-results.md`; commit code repo + AI-Docs; update the decision ledger.

## Self-review notes

- Spec coverage: notes/04 §2 (0a→Task 5, 0b→Task 3, 0c→Task 7), §3.1–3.6 → Tasks 2/4, §5 playbook → Task 6 dead-level logging + Task 7 step 3, §6 file list → file map above. U-bar/attention/controls are deliberately out of scope (stage >= 1).
- Type consistency: `Hierarchy`/`pool_mean`/`prolong`/`sym_log`/`sym_exp` signatures used in Tasks 3–6 match Task 1/2 definitions.
- No placeholder steps; every gate has a number.
