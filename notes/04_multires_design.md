# Multi-Res + Topology Phase — Design (2026-08-11)

Decisions D7.1–D7.12 in `00_decision_log.md`; brainstormed with Anka
2026-08-10/11. Strategy per D7.12: **smoke test first** — build the favored
configuration, prove any variant beats the flat net, and only then spend on
controls, ablations, and the topology article.

## 1. Hypothesis and goal

The flat per-tet predictor is information-limited: its 28-dim local view
cannot determine the next stretch when the correct answer depends on
far-field state (notes/01). Measured: the net's margin over the kNN floor
estimate shrinks 2.0x (toy) -> 1.45x (4k), its gap over the decoder oracle
grows 2.1x -> 3.7x (D5.2). **Goal: break this ceiling** — 4k single-step
error moves from 5.73 mm toward the 1.53 mm oracle, and the degradation
trend flattens.

Phase-level pre-registered numbers (evaluated only after the PoC earns the
investment): success = 4k single-step <= 3.0 mm AND kNN margin >= 2x;
kill = full stage-1 model < 15% better than flat at matched budget.

## 2. Smoke test (stage 0 — the next action)

Order of operations, cheapest signal first:

- **0a. kNN feature audit** (minutes, no training): re-run
  `diag_knn_floor.py` with the hierarchical feature set — pooled ancestor
  context + edge features appended to each tet's vector. The estimated
  floor must drop toward the oracle; if it does not, the features are wrong
  and we stop before training (D7.10).
- **0b. Composition oracle audit** (minutes, numpy): restrict GT stretch
  fields to each level (log-Euclidean mean), reconstruct with linear /
  log / sandwich composition, score **decoded positions** (never S-norms,
  D5.3). Confirms log-space (D7.11) and quantifies its ceiling.
- **0c. Train the PoC**: full favored configuration (Section 3), accuracy
  recipe (pos + residual + inertial, K=4 — single-step is the metric, so
  no stability recipe), existing `data/train_4k.npz`, ~2 h.

**Signal bar:** teacher-forced 4k single-step >= 20% below the flat net
(<= 4.5 mm vs 5.73). On success -> stage 1 rigor. On failure -> exactly one
debug loop (per-level output magnitude logging — dead coarse levels are the
known joint-training failure — plus re-check audits), then rethink rather
than tune.

Sanity path before 4k: overfit 20 toy frames (loss -> ~0 confirms gradients
flow through exp/log and all levels), then toy single-step (should at least
match flat 2.1 mm; toy has little headroom — it is a correctness gate, not
a signal gate).

## 3. Architecture (the favored configuration)

### 3.1 Hierarchy structure (built once per mesh, deterministic, no learned parameters)

Greedy aggregation on the **tet face-adjacency graph**, edge strength =
shared-face area, clusters grown only through material edges (~8:1 per
level). 4k: 17280 -> ~2160 -> ~270 -> ~34 nodes. Level l+1 connectivity =
quotient graph (clusters adjacent iff any members are), edge weight = total
crossing face area.

```
levels[l] = { assign (N_l,), adj (N_l,K), w_adj (N_l,K), vol (N_l,) }
```

Clusters are connected-through-material **by construction** — the topology
thesis (D7.2) is embodied in the data structure, not enforced by a loss.

### 3.2 Node inputs

Level 0: today's 28-dim vector, unchanged. Levels 1+: same layout pooled
from children — intensive quantities (H, mu/lam, pin-fraction) by
volume-weighted **log-Euclidean/linear mean**, forces by mean **and sum**
(a 50 N poke mean-pooled over 512 tets dilutes to nothing; the sum channel
keeps it visible, D7.4) — plus log rest volume.

**Ancestor context (D7.10):** every node additionally receives the post-MP
hidden states of its ancestor chain (3 gathers), so each level's
input-to-output map is well-posed on its own — the fine level *sees* the
far field pre-digested per scale instead of hoping the output sum divides
labour.

### 3.3 Edge features and message passing

Per-edge, per-frame, expressed in the receiving node's polar frame
(SE(3)-invariant; R comes free from `compute_S_from_x`):

```
e_ab = [ R_a^T (c_b - c_a) / l0,  |c_b - c_a|/l0 - 1,  axisangle(R_a^T R_b) ]   (7 dims)
```

The relative-rotation channel fixes the bending blind spot: per-node S
cannot see neighbours rotating against each other (D7.5). MP is the
standard edge-conditioned form, 2 rounds per coarse level (graphs are
tiny); level 0 keeps its existing 1-ring feature (no extra fine-level MP in
the PoC). Cluster centroid = volume-weighted member mean; cluster rotation
= polar of volume-pooled F.

### 3.4 Output composition (D7.11)

```
S* = exp( H_t + dH_0 + sum_{l>=1} Prolong_l(dH_l) ),   H_t = log(S_t)
```

- Per-level heads output dH_l (6 sym components, tanh-bounded, zero-init ->
  training starts at exactly S* = S_t).
- exp/log of symmetric 3x3 via a new `spd_log.py` module (eigh forward,
  Daleckii–Krein backward — same engineering class as `polar.py`, with
  gradcheck + FD tests before first use).
- SPD by construction; rotations never enter (material-frame tensors, one
  shared basis).
- Prolongation: partition-of-unity blend over {own cluster + adjacent
  clusters} — adjacency is topological, so no bleed across gaps like the
  U-bar's.

### 3.5 Weights (D7.6)

Shared across all nodes within a level (never per-node — D5.4); separate
small nets per level for the PoC (~50k params total). Log-volume in the
features is the hook for the shared-across-levels transfer variant later.

### 3.6 What does not change

Decoder, loss-through-decoder, warm start, curriculum, data, eval
harnesses. The hierarchy is strictly upstream of the same fine S\* field.

## 4. Stages after the PoC (contingent on signal)

| stage | question | pre-registered criterion |
|---|---|---|
| 1 rigor | is it the hierarchy? | beat (i) param-matched flat net always; (ii) 24-round deep-MP flat GNN once at the end. Phase success/kill numbers from Section 1 apply here |
| 2 attention | does content-based routing beat fixed pooling? | swap coarse-level MP for a small cluster-token transformer (optional fine->coarse cross-attention; no full-graph attention, no positional encodings — D5.4). Adopt iff > 10% over stage 1 at matched budget |
| 3 topology | does connectivity-respecting coarsening matter? | U-bar article; identical code, topological vs spatial clustering; spatial must measurably fail on it (this is the notes/01 thesis test) |
| 4 rollout | does the ceiling win survive autoregression? | stability recipe + optional blocks composition; 18-frame rollout on 4k. Single-step gains do not auto-transfer (Law 1) — gated separately |

Stage-1 metric discipline: decisions on single-step + kNN margin only;
rollout recorded but non-gating until stage 4.

## 5. Failure playbook

- Audit 0a shows no floor drop -> feature construction wrong (pooling,
  frames); fix before any training.
- PoC trains but coarse levels go dead (|dH_l| ~ 0 for l>=1) -> level
  dropout (randomly zero a level's output during training), the
  pre-identified fix for joint-training free-riding.
- PoC single-step improves < 20% with alive levels and passing audits ->
  the information-routing hypothesis fails as implemented; write it up,
  reconsider multigrid-in-the-loop (Approach 3) whose information carrier
  is decoded state rather than pooled features.
- exp/log numerics unstable near repeated eigenvalues -> Daleckii–Krein
  with thresholded divided differences (known technique); worst case,
  PoC falls back to linear composition (D7.3) to get signal while the
  module is hardened — composition error is second-order, the ceiling
  signal should not hinge on it.

## 6. New files

```
research/principal_stretch/hierarchy.py      coarsening, quotient graphs, pooling/prolongation operators
research/principal_stretch/spd_log.py        batched symmetric 3x3 log/exp with exact backward + tests
research/principal_stretch/hier_model.py     edge-MP, ancestor context, per-level heads, composition
train.py --hier flag                          reuses the existing trainer end to end
diag_knn_floor.py --feature-set               the 0a audit arms
diag_composition.py                           the 0b oracle audit
```
