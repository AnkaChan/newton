# Principal-Stretch Solver — Decision Log & Ablation Registry

The single ledger for this project. Every technical decision, why it was
made, the ablation/experiment that justifies it, and its current status.
Append-only: new decisions get new entries; overturned ones get their status
flipped, never deleted. Detailed reports live in
`AI-Logs/Newton/tasks/PrincipalStrecchSolver/` (AI-Docs repo); design and
review docs live here in `notes/`.

Status legend: **ACTIVE** (in force) · **KILLED** (tried, measured, rejected)
· **SUPERSEDED** (replaced by a later decision) · **PLANNED** (decided, not
yet implemented).

Doc map: `00` this ledger · `01` Anka's critique of the flat per-tet net ·
`02` implementation overview · `03` code-level review (+ interactive HTML,
comment sidecar) · `04` multi-res design (next).

---

## Era 0 — Original method (May 2026, pre-takeover)

| id | decision | rationale / evidence | status |
|----|----------|----------------------|--------|
| D0.1 | **Representation: per-tet symmetric stretch S** (6 DOF), rotation handled outside the network | S = sym(RᵀF) is invariant to rigid motion, so the predictor's job is SE(3)-invariant by construction | **ACTIVE** — the load-bearing idea |
| D0.2 | **Decoder: ARAP-with-target-stretch** local-global, E = Σ wₑ‖Fₑ − RₑS\*ₑ‖² | Global step = one pre-factored Poisson solve; all non-convexity quarantined in per-tet polar rotations | **ACTIVE** |
| D0.3 | Phase-2 training: 6 unrolled decoder iters, physics (incremental-potential) loss, absolute S\* = I + Δ, SVD-surrogate polar gradient | — | **KILLED** by the takeover review: every one of these four choices was individually wrong (D1.x) |

**Phase-1 result** (still valid): with *oracle* stretches and enough
iterations the decoder reconstructs trajectories to high accuracy — the
representation itself is sound.

## Era 1 — Takeover review: diagnose before repairing (2026-07-28)

Report: `2026-07-28-takeover-review.md`.

| id | decision / finding | evidence | status |
|----|--------------------|----------|--------|
| D1.1 | **Decoder truncation was ~90% of the May error** — local-global contracts at ~0.98/iter; 6 iters vs ~500 to converge | `diag_decoder_conv.py`, `diag_floors.py` error decomposition | finding — drove D2.1/D2.2 |
| D1.2 | **Constant-velocity baseline beat the trained net 5.6×** | `diag_baselines.py` | finding — established the bar |
| D1.3 | **Physics-loss objective mismatch**: loss = 1 backward-Euler step @ dt=1/60; data = 10 VBD substeps @ dt/10 → ~3.67 cm rollout ceiling built into the objective | analytic + measured in review §2.3 | finding — drove D2.4 |
| D1.4 | **SVD-surrogate polar gradient is biased**: exact only at S=I, 22–27% Jacobian error at 50% stretch | `diag_polar_grad.py` FD comparison | finding — drove D2.3 |
| D1.5 | Repair the pipeline before doing any new research | Phase-B/C decisions would be meaningless on a broken baseline | **ACTIVE** principle |

## Era 2 — Repairs, round 1 (2026-08-03)

Report: `2026-08-03-toyfix-round1-results.md`. Baseline repro: 11.7 mm
single-step (toy, teacher-forced, 10 decoder iters).

| id | decision | ablation → effect | status |
|----|----------|-------------------|--------|
| D2.1 | **Inertial warm start** x₀ = 2xₜ − xₜ₋₁ (pins restored) | decoder floor with oracle S: 1.05e-2 → **1.02e-3 m** (10×) at identical cost | **ACTIVE** — part of the method, not an optimisation |
| D2.2 | **Batched decoder + trainer** (leading batch dim everywhere, shared Cholesky) | ~2 s/step → **0.22 s/step** at K=4; enabled every later experiment | **ACTIVE** |
| D2.3 | **Analytic polar** (Higham scaled-Newton forward, exact Sylvester backward) | gradcheck + FD Jacobians to 1e-9; 5 iters to fp64 round-off at 17:1 anisotropy; ~4.5× faster than batched SVD | **ACTIVE** — `polar.py`, 8 unit tests |
| D2.4 | **Position-supervised loss** (mass-weighted, through the decoder); physics loss demoted to optional regulariser | ~4× single-step vs phys-loss arm | **ACTIVE** |
| D2.5 | **Residual parameterisation** S\* = Sₜ + Δ, zero-init head ("stretch unchanged" prior) | ~1.7× vs absolute S\* = I + Δ | **ACTIVE** |
| — | Round-1 combo (pos + residual + inertial) | single-step 11.7 → **2.1 mm** (decoder floor 1.0 mm) | |

**Law 1 (load-bearing since):** single-step accuracy does **not** move
rollout — 2.1 mm and 13 mm single-step models both land at ~0.065 m over 18
frames. Rollout is autoregressive-stability-bound.

## Era 3 — Rollout stability, round 2 (2026-08-03)

Report: `2026-08-03-round2-results.md`. Metric: 18-frame autoregressive
rollout, toy.

| id | decision | ablation → effect | status |
|----|----------|-------------------|--------|
| D3.1 | **MGN-style input noise** (σ = 3 mm on positions, S recomputed from noisy state, targets clean) | best single lever; 2 mm / 5 mm arms both worse than 3 mm | **ACTIVE** |
| D3.2 | **Incremental-potential regulariser** at weight 1e-4 on top of position loss | helps alone; main value in the combo | **ACTIVE** |
| D3.3 | **K=8 curriculum** (linear 1→8 over first half of training) | helps alone; largest single lever ≈ 2.3× | **ACTIVE** |
| — | **Combo is super-additive**: rollout mean 0.062 → **0.027 m**, final frame 0.137 → **0.046 m**; error growth quadratic → saturating | the standing "stability recipe" for all rollout evals | |

## Era 4 — Phase B: substep amortisation, killed by pre-registration (2026-08-03)

Plan with kill criteria: `2026-08-03-phaseB-plan.md` §5. Result:
`2026-08-03-pareto4k-results.md` + `pareto_4k.json`.

| id | decision | evidence | status |
|----|----------|----------|--------|
| D4.1 | **Pre-register kill criteria before scaling up** | both criteria tripped; no sunk-cost negotiation | **ACTIVE** principle |
| D4.2 | 4k article (24×12×12, 4225 verts / 17280 tets) as the scale testbed — speed claims meaningless at 225 verts (launch-overhead-bound) | VBD 1-substep cost ≈ 2.6 ms at toy = kernel launches | **ACTIVE** |
| D4.3 | **Phase B (beat VBD on wall-clock) is dead** | VBD's *cheapest* point (1.14 ms, err 0.0785 m) matches the net's *best* (100 ms, 0.0739 m); error is model-bound (decoder 4→20 iters: 0.089→0.074); net needs ~15× accuracy to sit on the VBD curve at plausible cost | **KILLED** |
| D4.4 | **Blocks: alternating net↔decoder passes** (PoissonNet pattern) | blocks 1→2→3: single-step 7.47→5.97→4.68 mm (monotone); rollout 0.0336→**0.0264**→0.0338 m — 2 blocks −20% both, 3 regresses rollout | **ACTIVE** option, ~1.2×-class lever |
| D4.5 | fp32 vs fp64 decoder at toy scale: identical wall-clock (latency-bound, not FLOP-bound) → keep fp64 | timing test 2026-08-03 | **ACTIVE** |

**Law 2:** the net's *relative* accuracy scales fine (3.1% of body length at
4k vs 3.4% toy); what changes with mesh size is VBD's accuracy-per-ms.
Speed-vs-VBD is the wrong axis for this method.

## Era 5 — The information-ceiling question (2026-08-04)

Report: `2026-08-04-knn-floor-results.md`. Instrument: `diag_knn_floor.py`
(kNN conditional-variance test: cross-trajectory kNN regression on the
28-dim features upper-bounds what any per-tet predictor can achieve).

| id | decision / finding | evidence (decoded single-step, per-vertex mean) | status |
|----|--------------------|-------------------------------------------------|--------|
| D5.1 | **Toy: features not exhausted** — hypothesis notes/01 does not bind at 8-cell diameter | oracle 1.01 / net 2.12 / kNN 4.17 / persistence 7.14 mm — net beats the kNN estimate 2× | finding |
| D5.2 | **4k: ceiling closes in with diameter, not yet hit** | oracle 1.53 / accuracy-net 5.73 / kNN 8.32 / persistence 21.5 mm — margin over kNN shrank 2.0×→1.45×, gap over oracle grew 2.1×→3.7× | finding — the target of the multi-res phase |
| D5.3 | **Never supervise S with plain L2** — S-space Frobenius error anti-correlates with decoded quality (stability nets *worse* than persistence in S-norm yet decode 3× better; kNN k=1→20 improves S-norm, *worsens* decode 8.4→11.9 mm) | same experiment, both scales | **ACTIVE** constraint on all future designs |
| D5.4 | **Positional identity is worthless** — same-tet kNN (= arbitrary position features) no better than global pool; the missing information is far-field *state* | toy same-tet k=5: 5.14 vs global 4.17 mm | **ACTIVE** constraint (rules out positional encodings as the fix) |
| D5.5 | `cholesky_solve` batch-fold fix (batched RHS vs unbatched factor materialised the factor per batch element: 22 GB OOM at 180×4225) | batched == per-sample to 1.5e-15; Warp reference test passes | **ACTIVE**, commit `0c3465e3` |

## Era 6 — Direction & housekeeping decisions (2026-08-05 → 08-10)

| id | decision | rationale | status |
|----|----------|-----------|--------|
| D6.1 | Research code **stays inside the newton repo** (offered standalone-repo extraction; declined) | Anka's call 2026-08-05; boundaries: zero engine-file changes beyond the one pyproject lint line, all work in `research/` + `notes/` | **ACTIVE** |
| D6.2 | **Canonical workspace** = worktree `pss-takeover`, branch `ankac/principal-stretch-dev`; old branches/worktree kept as archives, never worked in | Anka's call 2026-08-09; all data consolidated under canonical `data/` | **ACTIVE** |
| D6.3 | **Free-floating global rot/trans will be solved by body momentum** (direction C mechanics): momentum integration is exact for the rigid part regardless of net error; decoder nullspace is rank-3 (translations only — rotations are not in L's nullspace with frozen R), fixed via mass-weighted constraints, still one pre-factored solve | Anka's call 2026-08-09 | **PLANNED** (after multi-res phase) |
| D6.4 | Review flow: implementation docs in `notes/`, interactive HTML walkthrough with per-line comment threads (JSON sidecar = the channel; file-handle / server / offline modes), excerpt lines verified against source at build time | notes/02, notes/03, `build_review_html.py`; generalised as the `code-walkthrough-html` skill | **ACTIVE** process |

## Era 7 — Multi-res + topology phase (brainstormed 2026-08-10/11)

Goal (pre-registered per D4.1): **break the information ceiling** — 4k
single-step well below the flat net's 5.73 mm toward the 1.53 mm decoder
floor, and the toy→4k degradation trend (2.0×→1.45× margin over kNN;
2.1×→3.7× over oracle) flattens. Design doc: `notes/04` (next).

| id | decision | rationale / planned ablation | status |
|----|----------|------------------------------|--------|
| D7.1 | **Approach: hierarchical S-residual predictor, decoder untouched** (over full graph-U-Net latents, and over multigrid-in-the-decoder — those remain the escalation path) | minimum-delta controlled test of the information hypothesis; every piece mesh-independent | **PLANNED** |
| D7.2 | **Hierarchy = greedy aggregation on the tet face-adjacency graph** (~8:1/level; edge strength = shared-face area), coarse levels = quotient graphs; clusters connected-through-material by construction | topology constraint (notes/01); BSMS lesson: topology-blind/spatial coarsening measured worse than flat | **PLANNED** |
| D7.3 | Composition by *linear addition* in S-space | first-order only; can leave the SPD cone; Anka: stretches do not add and composition needs a frame | **SUPERSEDED** by D7.11 |
| D7.4 | **Pooling split: intensive quantities mean-pooled (volume-weighted), forces also SUM-pooled** | a 50 N poke mean-pooled over 512 tets dilutes to nothing at exactly the level meant to see it | **PLANNED** |
| D7.5 | **Edge-feature message passing in the receiving node's polar frame**: [Rₐᵀ(c_b−cₐ)/ℓ₀, edge stretch, axisangle(RₐᵀR_b)] — fixes the bending blind spot (per-node S cannot see relative rotation between neighbours) while preserving SE(3) invariance | ablation lever: zero the relative-rotation channel → separates "bending visibility" from "hierarchical reach" | **PLANNED** |
| D7.6 | **Weights shared within a level** (never per-node — D5.4); **separate small nets per level** for stage 1, shared-across-levels with scale conditioning as the transfer upgrade | sample efficiency, mesh independence; per-level fits differing feature statistics | **PLANNED** |
| D7.7 | **Staging for attribution**: stage 1 = hierarchy with plain edge-MP; stage 2 = cluster-token attention ablation (small transformer over coarse-level nodes only, optional fine→coarse cross-attention; no full-graph attention, no positional encodings — D5.4) | one variable per stage; MGN-Transformer placement | **PLANNED** |
| D7.8 | **U-bar (or slit-plate) article** for the decisive topology ablation: identical hierarchy code, topological vs spatial clustering — spatial merges the arms and should visibly fail | on a plain cuboid the distinction is untestable | **PLANNED** |
| D7.9 | Cross-topology generalization is the **long-term** goal: no per-article learned structure anywhere, but training stays single-asset for now | Anka's call 2026-08-10 | **ACTIVE** constraint |
| D7.10 | **Per-level shared weights require per-level well-posedness** (Anka, 2026-08-11): (a) every node's input includes **top-down ancestor context** — the post-MP hidden states of its ancestor chain — so the fine level sees the far field pre-digested per scale, instead of hoping the output sum divides labour; (b) every candidate feature set is **audited with the kNN conditional-variance instrument before training** — the estimated floor must drop toward the oracle when a feature is added, else the feature is wrong | `diag_knn_floor.py` re-run per feature set: (i) current 28-dim, (ii) +edge features, (iii) +ancestor context — minutes each, no training | **PLANNED** |
| D7.11 | **Log-space (Hencky) residual composition**: S\* = exp(Hₜ + ΔH₀ + Σ Prolongₗ(ΔHₗ)) with Hₜ = log(Sₜ); restriction uses the **log-Euclidean mean**; rotations never enter (material-frame right-stretch tensors share one basis; rotation is the decoder's job). SPD by construction; exact for commuting/volumetric strains; O([H_c,H_f]) (second-order) vs the exact sandwich C = U_c U_f² U_c, which is held in reserve | **oracle decomposition audit** (pre-training, minutes): restrict GT stretch fields per level, reconstruct with linear / log / sandwich rules, score decoded positions — decides if the BCH error matters at our ~10–20% strains. **Result 2026-08-11** (`diag_composition.py`, F14 decoded mean): log ≈ linear within 0.5% at every truncation depth on toy AND 4k (log better at +level2/+level1, 0.4% worse at coarsest-only — the literal "log ≤ linear everywhere" gate misses by that hair; BCH error immaterial, rule choice is not the bottleneck); full depth exact for both, decode = oracle to ~3e-15 m. 4k: coarsest-only 4.61e-2, +level2 1.96e-2, +level1 6.51e-3, full/oracle 1.33e-3 m — truncation is the lossy part (coarsest-only = 35× oracle), so the ΔH₀ head is mandatory. **Caveat**: GT contains inverted tets (S eig < 0, log undefined) in 16/540 toy and 19/180 4k val frames (excluded from the audit) — the training pipeline needs an SPD floor or sample exclusion | **ACTIVE** (controller ruling 2026-08-11: rules tie within 0.5% at all depths, literal gate missed by 0.4% at coarsest-only — adopted for the SPD guarantee; pending Anka sign-off) |
| D7.12 | **Smoke test before rigor** (Anka, 2026-08-11): build the intuitively-favored full configuration (hierarchy + edge-MP + ancestor context + log composition) and run one PoC on the existing 4k data; control baselines, U-bar, attention ablations only *after* some variant shows signal. PoC signal bar: single-step ≥ 20% below the flat net (≤ 4.5 mm vs 5.73). One debug loop on failure (per-level output magnitudes + audits), then rethink | "baseline is only meaningful if we prove any variant works" | **EXECUTED — SUCCESS 2026-08-12**: hier 4k single-step **4.21 mm** vs bar 4.5 (flat 5.73, oracle 1.53, −26.5%); all levels alive; toy 1.51 mm beats flat 2.12; rollout −14–19% (informational, still stability-bound). Report: `2026-08-11-hier-poc-results.md` |
| D7.13 | **kNN-on-concatenation invalidated as a feature-audit instrument beyond ~30 dims** (audit 0a executed, gate FAIL adjudicated 2026-08-11): floors rose with every feature superset (4k: 8.37 -> 10.11 -> 15.78 mm for 28 -> 35 -> 128 dims) — impossible for a true conditional-variance floor, so the rise is estimator degradation (1-NN z-distance/dim 0.087 -> 0.161); the base "floor" already sat above the trained flat net (8.37 vs 5.73 mm). Far-field information content of the pooled coarse fields stands established by the audit-0b telescope (6.5 mm vs 21.5 mm persistence). Ruling: proceed to the PoC per D7.12; future feature audits use oracle telescopes or small trained probes. Pending Anka sign-off | task-5 report (all 6 runs); composition audit | **ACTIVE** ruling |

---

## Era 7 outcome — the smoke test passed (2026-08-12)

D7.1–D7.11 are implemented and now **ACTIVE** (Tasks 1–6 of `notes/05`, all
review-gated; per-task evidence in the reports). The information ceiling on
the 4k article is broken: 5.73 -> 4.21 mm single-step at matched training
recipe, with every hierarchy level contributing. Known deviations recorded
en route: F12 eps made dtype-dependent (Task-1 review), spd_floor added for
inverted GT tets (audit 0b), kNN-concat audits invalidated as an instrument
(D7.13), `so3_log_axial(saturate=True)` on the edge-feature path only
(untrained-net K=4 transients exceed the 3 rad guard; plan amendment
pending review). Next: stage 1 (controls: param-matched flat, deep-MP
flat; anchor re-pinning; multi-seed), then the reordered stage 2 per the
GNN survey (feature-space attention before cluster tokens), stage 3 U-bar.

## Standing constraints (the short list every new idea must pass)

1. Supervise through the decoder; never plain L2 on S (D5.3).
2. No positional identity — encodings, per-node weights, learned per-article
   structure all ruled out (D5.4, D7.9).
3. Hierarchies and attention must respect material connectivity; nothing
   couples through space (D7.2).
4. Rollout claims require the stability recipe (D3.x) and 18-frame
   autoregressive evaluation — single-step gains do not transfer (Law 1).
5. Scale claims require the 4k article; toy-scale wall-clock is
   launch-overhead-bound (D4.2).
6. Pre-register success/kill criteria before scaling any experiment (D4.1).
