# ALM-DAT algorithm conformance review

Independent review of the `alm_dat` self-contact path of `SolverVBD` against the
authoritative algorithm in `review/algorithm-reference.md`.

| | |
|---|---|
| Reviewer | Claude Fable 5.1, fresh context, local tmux agent |
| Reviewed snapshot | `986b7c921a3a4f9d0b7dd17af1edf475666b622e` (branch `ankac/alm-dat-fable-review`, identical in `newton/` to `ankac/alm-dat-solver` at that commit) |
| Implementation commits | `6cbff0ae`, `9eb4ce20`, `f4fe933d`, `fb9828e2`, `86d8bddc`, `09e9521f`, `5fb01828`, `986b7c92` |
| Source-head drift at handoff | source worktree is at `11702e64` = snapshot + one commit (`notes/alm-dat-implementation-review/*` only). `git diff 986b7c92..11702e64 -- newton/` is empty, so every finding applies to the current source head as well. |
| Files reviewed | `newton/_src/solvers/vbd/particle_dat_kernels.py` (all 1084 lines), the `alm_dat` changes in `newton/_src/solvers/vbd/solver_vbd.py`, the detector storage in `newton/_src/geometry/tri_mesh_collision.py` and `newton/_src/geometry/kernels.py`, the DAT helpers in `newton/_src/solvers/vbd/rigid_vbd_kernels.py`, the forward-step and velocity kernels in `newton/_src/solvers/vbd/particle_vbd_kernels.py`, the three new test modules, the changelog fragment |
| Not changed | nothing outside `review/` was edited; no commits, pushes or PR actions |

Line references below are `file:line` in the snapshot. Abbreviations: K = `particle_dat_kernels.py`,
S = `solver_vbd.py`, G = `geometry/kernels.py`, T = `geometry/tri_mesh_collision.py`,
R = `rigid_vbd_kernels.py`, V = `particle_vbd_kernels.py`. "A side" is the pseudocode's A
(triangle, or edge `e2`; the implementation's negative side), "B side" is the pseudocode's B
(vertex, or edge `e1`; the implementation's positive side).

## 1. Verdict

The driver and kernels implement the reference algorithm's control flow faithfully. The
three states stay distinct (`x_start` = `pos_prev_collision_detection`, `x_trial` =
`state.particle_q`, `x_anchor` = `_dat_anchor`), truncation never writes the trial, both
stage loops run fixed counts, the output is the anchor and the velocity is
`(x_anchor - x_start) / dt`. The plane placement rule (`SlidePlanes`) matches a literal
float64 transcription of the pseudocode on 2200 randomized vertex-triangle and edge-edge
cases including one-sided, nobody-approaches, thin-gap, pinned and near-threshold inputs,
and a brute-force grid confirms both pick the maximin offset. The truncation preserves
every stored plane across repeated rounds, through both plane-build call forms, under the
motion budget. All 15 CPU-runnable tests of the three new test modules pass.

There is one confirmed, low-impact ordering mismatch (multipliers are updated before the
conditional plane update instead of after). Everything else that differs from the
pseudocode is either a mathematically equivalent rewrite, a conservative tolerance, or a
choice the pseudocode leaves unspecified. Four of those unspecified choices change the
physical meaning of the method enough that they need the author's explicit confirmation:
the ALM row never reads the stored plane point and uses the plane normal only as a sign
reference; the plane margin `m` is hardwired to zero and floored at the float32 separation
band; the normal refresh interval is hardwired to one; and the two-argument plane
construction form of the pseudocode is always realized with the trial-aware placement.

Fit verdict under `REVIEW_GUIDELINES.rst`: **Fits if narrowed.** The outcome belongs in
`SolverVBD` and is an opt-in experimental mode with the default path unchanged, but the
public surface should be narrowed before stabilization (see section 6). This agrees with
the author's own review in the source worktree, which I read only after freezing the
findings below.

## 2. Findings, severity ordered

Classification tags: **MISMATCH** = differs from an explicit statement of the pseudocode;
**UNSPECIFIED** = the pseudocode does not fix the choice and the implementation picked one
that changes behavior; **EQUIVALENT** = different formula, same result; **HARMLESS** =
differs but cannot change the certified result; **GAP** = a hole in the algorithm itself
rather than in its implementation; **QUESTION** = needs the author's answer.

### F1. P1. UNSPECIFIED / QUESTION. The ALM row does not use the stored plane; it uses the closest-point witness gap, shifted by `hard_gap`, along the closest-point direction

- Where: `K:619-655` (vertex-triangle row geometry), `K:659-701` (edge-edge), `K:742-751`
  and `K:783-792` (force), `K:835-845` and `K:855-865` (multiplier update). `vt_point` /
  `ee_point` are read only by the truncation kernel (`K:425`, `K:446`); the stored normal
  enters the row only to fix the sign of the closest-point direction (`K:646-647`,
  `K:689-690`).
- Pseudocode: `x_trial = ALM_VBDSweep(x_trial, p, n, planes_lambdas)` and
  `UpdateALMMultipliers(x_trial, contacts)` (lines 27 and 33) pass the stored planes to the
  sweep. The pseudocode never defines the row's constraint function, so this is not a
  literal mismatch, but the data flow it writes down (p and n into the sweep) is not what
  runs.
- Mechanism: the row's gap is `|x_v - witness| - hard_gap` with `witness` the closest point
  on the counterpart (`K:638-648`), the cushion spans `hard_gap .. R` with `R =
  particle_self_contact_margin` (`S:4050-4054`), and the wall sits at `hard_gap` above the
  counterpart surface. The division plane `(p, n)` is a separate safety net that only the
  truncation enforces.
- Impact: the "hard wall" and the "division plane" are two different surfaces along two
  different axes (closest-point direction versus certified separator). The commit message
  of `5fb01828` records this as a deliberate fix for lateral drift when the certified axis
  is tilted. It is a sound choice, but it is a design decision the pseudocode does not
  contain. The author's review (section 5 and R5) documents the consequence that the wall
  is only guaranteed to be above the plane for one-sided approach.
- Evidence: static trace of every read of `vt_point`/`ee_point` in `K`; `review/checks/check_alm_row_reference.py`
  confirms the row's closed form (released / cushion / wall) against a brute-force slack
  minimization, worst force error 1.5e-6 over 400 samples, stiffness equals minus the
  numerical slope in every regime.
- Confidence: high that this is what the code does; whether it is intended is the author's call.

### F2. P1. UNSPECIFIED / QUESTION. The plane margin `m` is zero, floored at the float32 separation band

- Where: `S:995` (`self._dat_margin = 0.0`), passed at `S:3386` and `S:3407`; `K:102`
  (`margin_eff = min(max(margin, eps), max(0, 0.5 * gap))`); `eps` from
  `dat_separation_epsilon` (`R:6949-6951`), about 1e-6 m at metre scale.
- Pseudocode: `SlidePlanes` keeps "both primitives separated with margin m" (lines 59-60)
  and uses `m` in `b_ij` and `tau` (lines 68-70). The value of `m` is never stated.
- Mechanism: with `m` equal to the band, the certified anchor only guarantees "no
  crossing"; the contact thickness is carried entirely by the compliant row (cushion onset
  at `R`, wall at `hard_gap`, F1). The driver trace recorded `margin = 0.0` on every build
  and `margin_eff` exactly `1.0e-6` on every valid plane of the two-layer scenes.
- Impact: if `m` was meant to be the self-contact thickness (as in the plane-only design
  where planes enforce separation), the implementation differs materially. If `m` was meant
  as a numerical band, it matches. Also note that `hard_gap` (default `R/4`, `S:985-988`)
  adds a third distance the pseudocode does not have, so "margin" now means three things:
  the plane band (about 1e-6 m), the wall height `hard_gap`, and the cushion onset `R`.
- Related: the knob is not configurable from the constructor. Acknowledged by the author's
  review as R7.

### F3. P2. MISMATCH (confirmed, low impact). Multiplier update runs before the conditional plane update within an iteration

- Where: `S:4161-4176` (`update_alm_contact_multipliers` at the trial) precedes
  `S:4178-4181` (interval test and `_alm_dat_truncate`). Verified by the call trace in
  `review/checks/results/check_driver_trace.txt`: `... update_lambda -> truncate -> commit -> build` per iteration.
- Pseudocode lines 28-33: truncate, re-anchor, recompute planes, then `UpdateALMMultipliers`.
- Impact: because the multiplier update never reads `p` and uses `n` only as a sign (F1),
  the order only matters through the `valid` flags and the sign normals. A pair that becomes
  valid during this iteration's rebuild receives its first dual update one iteration later,
  and a pair whose certified axis flips sign during the rebuild uses the previous sign for
  this update. Both effects are bounded by one iteration. The author's review (section 6)
  notes the swap as intentional so the dual sees the normal the sweep used.

### F4. P2. UNSPECIFIED. `NORMAL_UPDATE_INTERVAL` is hardwired to 1; the slide-only plane update is never used by the solver

- Where: every call passes `refresh_normals=True` (`S:3408`, `S:3412`, `S:4181`, `S:4681`).
  The kernel's slide-only mode (`K:278`, `K:333`: `refresh_normals == 1 or valid == 0`) is
  reachable only from tests.
- Pseudocode lines 43-47: slide `p` every round, re-pick normals every
  `NORMAL_UPDATE_INTERVAL` rounds including round 0.
- Equivalence: with interval 1 the pseudocode also recomputes `(p, n)` every round and the
  implementation is exactly that (the slide result would be overwritten by the recompute).
  With the parameter fixed, the "occasionally" intent is not exercised. In stage 1 the
  pseudocode's two-argument recompute (line 31) also re-picks normals, so stage 1 matches.
- Note: the separator search prefers the recomputed closest-point axis and otherwise the
  largest certified gap, with the stored normal as one candidate (`R:7095-7104`,
  `R:7114-7141`, `R:7184-7236`), so normals can switch between candidate axes from round to
  round. Each new axis is certified at the anchor, so safety is unaffected. Acknowledged by
  the author's review as R9.

### F5. P2. UNSPECIFIED / QUESTION. The pseudocode's two-argument `ComputeContactConstraintsAndNormals(contacts, x)` does not say where `p` goes; the implementation always uses the trial-aware placement

- Where: one kernel, `trial` always supplied: `S:3401-3410` (at `x_start` with `trial = y`),
  `S:3379-3389` (after every truncation with the current trial), `K:293-310`, `K:345-363`.
- Pseudocode: lines 11, 18 and 31 call the two-argument form; line 47 calls the
  three-argument form with `x_trial`. Only `SlidePlanes` (lines 52-75) is specified for
  placing `p`, and it needs a trial.
- Impact: if the two-argument form was meant to place `p` at the midpoint (the
  `SlidePlanes` outcome with zero remaining displacement) or at a support point, stage 1
  planes would sit differently from what runs now. If it was meant as shorthand for
  `SlidePlanes` with the current trial, the implementation matches exactly.

### F6. P2. QUESTION. How the vertex-oriented history description extends to edge-edge

- Where: detection stores `(e_index, colliding_edge)` from both edges' threads
  (`G:1617-1623`, record at `G:1750`); rows are sorted by counterpart index (`G:1835-1862`,
  enabled at `S:1132` and `S:1138`); planes and multipliers are built only for the
  canonical record `e1 < e2` (`K:265`, `K:312-313`); matching merges each edge's previous
  and current sorted rows by counterpart index (`K:870-891`).
- Pseudocode line 22: "arranged by vertex side's id, each vertex's contact is sorted by
  triangles 3 indices".
- Reading taken by the implementation: identity = (row-owner element, counterpart element
  index). For VT that is (vertex, triangle index). For EE that is (owner edge, counterpart
  edge index), the pair appears in both edges' rows, and only the canonical direction owns
  state. Sorting by the triangle (or edge) index is equivalent to sorting by its vertex
  indices only because topology is static, which holds here.
- Inert state: the mirrored EE slot receives a carried multiplier from the previous mirrored
  slot (`K:884-891`) but is never updated or read (`K:739`, `K:832` gate on `valid`); it
  stays zero forever. `particle_contact_alm_multipliers()` (`S:3428-3441`) therefore
  returns zeros for mirrored slots. Harmless, but worth a comment. Under overflow the kept
  prefix may hold only the mirrored record, so that pair has no plane; the coverage flag
  (`K:1068-1084`, `S:3395`, `S:3414-3426`) reports exactly this case. The author's review
  lists this as R6.

### F7. P2. GAP (algorithm precondition) with probe. A pair without a strict separator at `x_start` gets no plane and no ALM force for the whole step

- Where: `_certify_primitive_pair_separator` fails closed at zero gap (`R:6991-6995`);
  the force and multiplier kernels skip `valid == 0` slots (`K:739`, `K:780`, `K:832`,
  `K:852`); such a slot's multiplier is neither updated nor zeroed, so a stale carried value
  survives to the snapshot (`K:895-912`).
- Pseudocode line 6 assumes "x_start: penetration free initial state". The implementation's
  precondition is stronger: strictly separated along a certifiable axis.
- Probe (`review/checks/probe_touching_start.py`, results in `review/checks/results/probe_touching_start.txt`):
  a free 4x4 patch over a fixed 8x8 plate.

  | start configuration | alm_dat lowest z after 1 / 10 / 40 steps | penalty_dat lowest z after 1 / 10 / 40 steps |
  |---|---|---|
  | 5 mm above (control) | 1.0177 / 1.0194 / 1.0197 | 1.0150 / 1.0198 / 1.0198 |
  | coplanar, gap 0 | 0.9872 / 0.8725 / 0.4900 | 0.9845 / 0.8219 / 0.3552 |
  | 2 mm interpenetrating | 0.9852 / 0.8705 / 0.4880 | 0.9825 / 0.8199 / 0.3532 |

  Both paths fall through on a touching or interpenetrating start, so this is a property of
  the DAT framework's strict-separation requirement and not an `alm_dat` regression. On the
  coplanar start 926 planes were still certified after step 1; they are in-plane separators
  (edge-edge and offset vertex-triangle pairs) and cannot stop vertical motion.
- Impact: the algorithm has no recovery mechanism for pairs that start a step without a
  separator, and `alm_dat` additionally withholds the compliant force from them. The
  pseudocode should state the strict precondition, or the design should say what happens to
  uncertified pairs. Reported separately from conformance as requested.

### F8. P2. EQUIVALENT and HARMLESS differences (no action needed, listed for completeness)

- `p` update: pseudocode slides the stored point, `p + (b - n.p) n` (line 75). The kernel
  writes `origin + b n` with `origin` the A side's first vertex (`K:92`, `K:178`). Same
  plane, different point on it; every consumer uses only `n.(x - p)` (`R:7293`). EQUIVALENT.
  Heights are measured relative to `origin` for float32 accuracy, as the rigid-DAT code does.
- Extra candidate in the both-sides branch: the kernel seeds the argmax with `b_lo`
  (`K:138-151`) before scanning the clamped `b_ij` candidates (`K:152-177`). The minimum
  remaining-motion fraction is a concave piecewise-linear function of `b` whose maximum on
  `[b_lo, b_hi]` is always attained by some clamped `b_ij` (the increasing A-side lines and
  decreasing B-side lines either cross inside the interval, at a `b_ij`, or the maximum is
  at an endpoint reached by clamping). So the seed never wins against the pseudocode's
  candidate set except on exact ties, where both give the same `b`. Verified numerically
  (section 4). EQUIVALENT.
- Approach test: `rate > eps` (`K:54`, `K:59`, `K:108`, `K:112`, `K:154`, `K:158`) instead
  of `max(0, rate) > 0`. A vertex approaching by less than about 1e-6 m per step is treated
  as static. My float64 reference needed the same guard: with a literal `> 0` test, float
  noise of 1e-17 in the normal rate produced `tau` values of 1e+10 and wrong branch
  selection. The implementation's threshold is the robust reading. HARMLESS, and arguably
  a necessary clarification of the pseudocode.
- Conservative truncation: `planar_truncation_t` scales the crossing fraction by
  `gamma = dat_conservative_bound_relaxation = 0.85` (`R:7306-7308`, `S:3368`), and the
  commit clamps the per-particle total displacement from `x_start` by the triangle
  inequality (`K:504-509`). Both only shorten steps and keep the anchor on the segment
  toward the trial, so separation is preserved and the invariant is unaffected. HARMLESS.
  Per round a vertex heading into a plane closes 85 percent of its remaining normal distance.
- Lambda restore before the initial truncation (`S:3411` before `S:3412`) while the
  pseudocode restores after (lines 16-23). Nothing between them reads multipliers. HARMLESS.
- `m` shrink reading: the kernel applies the shrunk `margin_eff` in `b_lo`, `b_hi`, `b_ij`
  and `tau` alike (`K:102-104`, `K:160`, `K:56`, `K:61`). The pseudocode's comment on line 60
  could also be read as shrinking only the bounds; that reading would make `tau` negative
  for thin pairs, so the kernel's reading is the sensible one. My reference uses the same reading.
- Pinned or zero-mobility particles: `forward_step` gives them zero displacement
  (`V:1573-1577`), so they never approach and never move; the row's inertia metric is
  floored at `K` (`K:546`) so a fully pinned pair still has a finite `rho`. HARMLESS.
- Empty contact set: every kernel loops to the stored count (`K:260`, `K:317`), launch size
  is at least 1 (`K:216`), matching skips empty element sets (`K:1023-1025`). HARMLESS.

### F9. P2. Standards and surface (brief; overlaps the author's R4)

- The `.. experimental::` block (`S:485-488`) names three of the six new parameters;
  `particle_contact_alm_rho_scale`, `particle_contact_alm_rho_floor_ratio` and
  `particle_contact_alm_hard_gap` are not marked. `particle_dat_coverage_ok()` and
  `particle_contact_alm_multipliers()` are new public methods without an experimental marker.
- `particle_self_contact_method` is a string selector for a closed set; `CODING_GUIDELINES.rst`
  prefers an enum for new closed categories.
- Changelog fragment present and categorized; tests use `unittest` with imperative
  docstrings; `wp.array[...]` annotations throughout; no new dependencies.

### Not re-verified here

The author's review reports a demonstrated per-world `reset()` defect (R1: all worlds'
multipliers are cleared, `S:2861-2868`) and a design decision on the `rho` floor (R2). Both
lie outside the pseudocode's scope and I did not probe them. I note that the reset branch
does clear every world unconditionally while the elasticity-ALM reset one line above honours
the mask, which supports R1.

## 3. Conformance matrix

Status key: OK = conforms; EQ = equivalent rewrite; TOL = conservative tolerance; UNSPEC =
pseudocode leaves it open, implementation chose; MIS = mismatch; see F-numbers for detail.

### 3.1 Contact storage, grouping, sort order, history identity

| Pseudocode | Implementation | Status |
|---|---|---|
| line 7: compact global buffer arranged by the vertex side's id | shared pair scratch filled by one thread per vertex, records `(vertex, triangle)` (`G:1565-1583`, `G:1572`); exact CSR rows per vertex rebuilt after detection (`T:904-947`), capacity = budget x element count (`T:342-343`, `T:348`) | OK |
| line 7 for EE (not stated) | records `(edge, colliding_edge)` from both edges' threads (`G:1617-1623`, `G:1750`); rows per edge; canonical `e1 < e2` owns the plane (`K:265`, `K:312-313`) | UNSPEC, F6 |
| line 10: `p`, `n` have the same layout as contacts | plane arrays sized to row capacity, indexed by row slot (`K:182-212`); multipliers likewise (`K:583-615`) | OK |
| line 22: each vertex's contacts sorted by the triangle's indices | insertion sort by counterpart index per row (`G:1835-1862`), enabled on both construction paths (`S:1132`, `S:1138`) | EQ (static topology) |
| line 21-23: restore lambda from history | two-pointer merge of previous and current sorted rows, zero for unmatched (`K:870-891`, `S:3411`); snapshot at step end (`K:895-912`, `S:4683`) | OK |
| new contacts | matched as unmatched, lambda 0 (`K:888-891`) | OK |
| lost contacts | not in the current rows, dropped at snapshot | OK |
| inactive slots (valid 0) | plane not built, no force, no dual update, carried lambda kept (`K:739`, `K:780`, `K:832`, `K:852`) | UNSPEC, F7 |
| slots re-assigned by detection | both valid arrays zeroed at step start (`S:3399-3400`) | OK |
| capacity overflow | detector keeps a prefix and sets overflow flags (`G:1290-1293`, `G:1580-1583`); coverage flag recorded per step (`K:1068-1084`, `S:3395`), host accessor (`S:3414-3426`); growth re-creates planes and ALM state, history dropped once (`S:1237-1241`) | OK (reported, not silent) |

### 3.2 States and initialization (pseudocode lines 6-23)

| Pseudocode | Implementation | Status |
|---|---|---|
| `x_start` penetration free | `pos_prev_collision_detection` assigned at the PRE_INIT detection on `state_in` (`S:3083` via `S:3086-3107`, `S:4821-4833`, `S:2686`); also `particle_q_prev` (`V:1571`) | OK; precondition is strict separation, F7 |
| `contacts = InitialSphericalQuery(x_start, motion_budgets)` | query radius `margin + gap` (`S:1098`); schedule forced to PRE_INIT for `alm_dat` (`S:4756-4759`, validated `S:4878-4882`); budget `0.5 * gamma * radius` (`S:3344-3346`) | OK |
| line 11: `p, n = Compute(contacts, x_start)` | `build_contact_planes(anchor = x_start, trial = y, refresh_normals=True)` (`S:3401-3410`) | OK / UNSPEC form, F5 |
| line 14: `x_anchor = x_start` | `_dat_anchor.assign(pos_prev_collision_detection)` (`S:3396`) | OK |
| line 15: `x_trial = y` | `_alm_dat_materialize_trial` writes `x_start + v_new dt` into `state_in.particle_q` (`S:3397`, `S:3348-3357`, `V:1583`), no budget clamp (`max_displacement = inf`) | OK |
| line 16-17: initial truncation, anchor update | `_alm_dat_truncate(state_in.particle_q)` (`S:3412`): per-vertex atomic-min fraction (`K:401-456`), commit with budget from `x_start` (`K:493-509`, `S:3371-3378`); writes `_dat_anchor` only | OK |
| line 18: `p, n = Compute(contacts, x_anchor)` | rebuild at the new anchor with refreshed normals (`S:3379-3389`) | OK / UNSPEC form, F5 |
| line 20-23: lambda init and restore | `match_contact_history` (`S:3411`) | OK (order before truncation, harmless, F8) |
| `x_start`, `x_trial`, `x_anchor` distinct | three arrays; the trace asserted identities and that the trial is bit-identical across every truncate / commit / build | OK |

### 3.3 ALM iteration stage (lines 26-33)

| Pseudocode | Implementation | Status |
|---|---|---|
| fixed `FIXED_ALM_ITERATIONS` | `for iter_num in range(self.iterations)` (`S:2700-2710`) | OK |
| `ALM_VBDSweep(x_trial, p, n, lambdas)` | per color: ALM contact scatter at the current trial (`S:4041-4059`, `K:705-803`), VBD solve, materialize trial (`S:4155-4156`) | OK structurally; row ignores `p`, uses `n` as sign, F1 |
| `update_contact_planes()` condition | `interval > 0 and iteration % interval == 0` (`S:4178-4181`), default 1 | OK |
| truncate, anchor, recompute planes at the penetration-free anchor | `_alm_dat_truncate(state_in.particle_q, refresh_normals=True)` (`S:4181`) | OK |
| `UpdateALMMultipliers(x_trial, contacts)` | `update_alm_contact_multipliers(state_in.particle_q)` (`S:4163-4176`), `lambda <- transmitted force` (`K:845`, `K:865`), nonnegative (`K:578-579`) | OK value and pose |
| order: planes then multipliers | multipliers then planes | MIS (low impact), F3 |
| multiplier sign and indexing | compressive, one scalar per canonical slot, pushes +direction on B and minus barycentric weights on A (`K:762-773`, `K:795-802`) | OK |
| plane history within the step | normals re-searched with the stored normal as hint, stored plane kept when no separator is found (`K:276-291`, `K:331-344`) | UNSPEC (robustness guard), preserves invariant |

### 3.4 Final truncation stage and output (lines 37-50)

| Pseudocode | Implementation | Status |
|---|---|---|
| fixed `FIXED_TRUNCATION_ITERATIONS` | `particle_dat_truncation_iterations` rounds (`S:4677-4681`), at least 1 enforced (`S:989-993`) | OK |
| `x_trial` immutable | `state_out.particle_q` passed read-only to truncate and build; trace confirmed unchanged | OK |
| truncate before sliding | `_alm_dat_truncate`: truncate, commit, then build (`S:3359-3389`) | OK |
| `SlidePlanes` from the truncated anchor and remaining displacement | `dat_plane_offset(n, anchor positions, trial - anchor)` (`K:293-310`, `K:345-363`) | OK |
| normals every `NORMAL_UPDATE_INTERVAL`, including round 0, at the anchor | every round, at the anchor (`S:4681`) | OK for interval 1, UNSPEC, F4 |
| `x_out = x_anchor` | `state_out.particle_q.assign(self._dat_anchor)` (`S:4682`) | OK |
| `v_out = (x_out - x_start) / dt` | `update_velocity(dt, particle_q_prev, ...)` (`S:4685-4689`, `V:1587-1589`); trace confirmed `particle_q_prev == pos_prev_collision_detection == x_start` | OK |

### 3.5 SlidePlanes (lines 52-75) against `dat_plane_offset` (`K:66-178`) and `_dat_plane_room` (`K:36-62`)

| Pseudocode | Implementation | Status |
|---|---|---|
| orientation: `n` from A toward B, unit | separators normalize (`R:6964-6966`) and orient toward the B primitive (`R:7074`, `R:7155`); kept normals stay unit; truncation uses `+n` on B and `-n` on A (`K:429`, `K:433`, `K:448`, `K:453`) | OK |
| `h_i = n . x_anchor_i` for every vertex of the pair | heights relative to `origin = A[0]` (`K:92-98`); VT 3 + 1 (`K:358-359`), EE 2 + 2 (`K:305-306`) | EQ (origin shift) |
| `a_i = max(0, +n.r_i)` on A, `max(0, -n.r_i)` on B, `r = x_trial - x_anchor` | `K:108`, `K:112`, `K:153`, `K:157`; displacements `trial - anchor` (`K:297-304`, `K:349-357`) | OK |
| approaching = `a_i > 0` | `rate > eps` | TOL, F8 |
| `b_lo = max_A h + m`, `b_hi = min_B h - m` | `K:103-104` | OK |
| `m` shrunk to half the gap when thinner than `2m` | `margin_eff = min(max(m, eps), max(0, gap/2))` (`K:102`) | OK plus eps floor; `m = 0` passed, F2 |
| nobody approaches: midpoint | `K:115` | OK |
| only B approaches: `b_lo` | `K:119-120` | OK |
| only A approaches: `b_hi` | `K:117-118` | OK |
| both: pairs (3 for VT, 4 for EE) | nested loops over approaching A x B (`K:152-159`) | OK |
| `b_ij = (a_j (h_i + m) + a_i (h_j - m)) / (a_i + a_j)` | `K:160` with `margin_eff` | OK |
| clamp to `[b_lo, b_hi]` | `K:161` | OK |
| score = min over all approaching vertices of `tau` | `_dat_plane_room` over all approaching A and B vertices (`K:52-61`) | OK |
| `b` = argmax score | `K:175-177`, seeded with `b_lo` (`K:138-151`) | EQ, F8 |
| `p = p + (b - n.p) n` | `origin + b n` (`K:178`) | EQ, F8 |
| degeneracy: zero denominators | guarded by `rate > eps`, `a_i + a_j >= 2 eps` | OK |
| degeneracy: gap <= 0 (kept plane or rounding) | `margin_eff = 0`, finite output; truncation then blocks further crossing (`R:7297-7300`); verified by probe | OK (guard preserves no-crossing) |

### 3.6 Normal construction call forms, truncation safety, thickness semantics

- One kernel serves both pseudocode forms (`K:233-365`). Inputs: anchor, trial, triangle and
  edge indices, collision rows, stored normal and valid flag. Outputs: normal (when refreshed
  and found), point, effective margin, valid flag. `refresh_normals=True` re-searches the
  separator with the stored normal as a hint; `refresh_normals=False` keeps the normal and
  only re-places `p`. The trial is always needed because `p` is always placed with
  `SlidePlanes` (F5).
- Truncation preserves separation: at build time every A vertex satisfies
  `n.(x - p) <= -margin_eff` and every B vertex `n.(x - p) >= margin_eff` by construction of
  `[b_lo, b_hi]`; `planar_truncation_t` keeps each vertex's straight segment in its closed
  half-space (`R:7277-7308`); half-spaces are convex so independent per-vertex fractions are
  jointly safe; the commit clamp shortens steps along the same segment. Hence the committed
  anchor never crosses a stored plane, and a plane strictly separating the convex hulls keeps
  the primitives disjoint. Confirmed numerically: worst violation of the band after 40 commits
  across 8 seeds was 6.5e-8 m (float32 rounding inside the 1e-6 band, still on the correct
  side of the plane), and 6.2e-8 m across 12 solver steps in the driver trace.
- Thickness semantics: three distances now exist (plane band about 1e-6 m, wall at
  `hard_gap`, cushion onset at `R`), versus the pseudocode's single `m` (F1, F2). The legacy
  per-color DAT path also places planes with no thickness (5 percent of the gap plus the
  band, `R:7241-7273`), so the plane semantics are consistent with existing Newton DAT.

## 4. Tests and independent checks

All runs on this VM's CPU with `CUDA_VISIBLE_DEVICES=""` and a private Warp kernel cache per
process, because every GPU was occupied by another agent's processes during the review.
Environment: Python 3.10.12, Warp 1.17.0, NumPy 2.5.0, `uv sync --extra dev` in this
worktree. Scripts and raw outputs are under `review/checks/` and `review/checks/results/`.

| Run | What it checks | Result |
|---|---|---|
| `python -m unittest newton.tests.test_particle_dat_kernels newton.tests.test_particle_contact_alm newton.tests.test_solver_vbd_alm_dat -v` | the three new test modules | 16 tests: 15 ok, 1 skipped (`test_alm_dat_step_is_graph_capturable` needs CUDA); 37 s |
| `check_slide_planes_reference.py` | literal float64 `SlidePlanes` versus `dat_plane_offset`: 2 kinds x 4 approach patterns x {plain, thin, pinned} x 60 cases, plus 80 near-threshold cases, a boundary-maximum case, zero and negative gap degeneracies, and the project's worked example; brute-force grid (20001 points) validates the maximin for both | all categories pass; offsets agree to 5e-6 m, `margin_eff` to 1e-6 m; both attain the brute-force maximum |
| `check_alm_row_reference.py` | brute-force slack minimization versus `alm_contact_row` over 400 random (gap, lambda) samples; stiffness versus numerical slope | worst force error 1.5e-6; all stiffnesses match their regime |
| `check_truncation_invariant.py` | 8 seeds of two rotated cloth layers, random trials up to 6 cm, 5 truncate-commit-build rounds alternating `refresh_normals` True/False; plane invariant, sigma range, budget, monotone remaining displacement, immutable trial, kept-plane zero-gap case | pass; worst band violation 6.5e-8 m; budget respected (0.01275 m) |
| `check_driver_trace.py` | monkeypatched kernel launchers in a 12-step `alm_dat` run (3 iterations, interval 1, 2 final rounds, 3 colors): call sequence versus pseudocode, array identities, trial immutability across truncation, multiplier update pose = `x_start + displacement`, anchor within budget, planes hold at every commit and build, output = anchor, velocity = `(x_out - x_start)/dt`, detection pose = step-start pose | pass; sequence `record_coverage, build, match_history, truncate, commit, build, [accumulate x3, update_lambda, truncate, commit, build] x3, [truncate, commit, build] x2, snapshot_history` |
| `probe_touching_start.py` | F7: coplanar and interpenetrating starts, `alm_dat` versus `penalty_dat` | both paths fall through; control rests at z = 1.0197 |

Probe corrections made during the review (recorded for honesty): my first versions of the
SlidePlanes reference used the literal `a_i > 0` test and a case generator that moved
"receding" B vertices toward the plane, the ALM-row probe used an over-strict release
threshold for regime bookkeeping, and the driver probe compared against a stale trial
snapshot across a state swap. All three were my errors, not implementation defects; the
fixed scripts are the ones in `review/checks/`.

Limits: no CUDA execution (graph-capture test skipped; no float32 GPU atomics path
exercised); no large scenes; no multi-world or solver-owned-pipeline probes; no timing.

## 5. Remaining uncertainties and questions for the author

1. F1: Is it intended that the ALM row never reads the stored plane point and measures its
   gap from the closest-point witness, with the wall at `hard_gap` above the counterpart
   surface? If yes, the algorithm reference should say so, since `p, n` are passed to the
   sweep in the pseudocode.
2. F2: What value is `m` meant to have in `SlidePlanes`? Zero plus the separation band (as
   implemented), the self-contact thickness `R`, or something else?
3. F5: What should the two-argument `ComputeContactConstraintsAndNormals(contacts, x)` do
   for `p`? The implementation always uses the trial-aware placement.
4. F4: Is `NORMAL_UPDATE_INTERVAL = 1` acceptable for now, or should the slide-only cadence
   be wired and exposed?
5. F6: Please confirm the edge-edge history identity (owner edge, counterpart edge index,
   canonical `e1 < e2` owns the state) is the intended extension of the vertex-oriented
   description.
6. F7: Should the algorithm state the strict-separation precondition explicitly, and should
   uncertified pairs receive any force on `alm_dat`?
7. F3: Keep the swapped order (multipliers before plane update) or match the pseudocode?

## 6. Review axes (per REVIEW_GUIDELINES.rst)

- Fit: **Fits if narrowed.** Core solver capability, right owner, opt-in experimental, default
  unchanged, with regression tests. Narrow the public contract before stabilization: mark all
  six parameters and the two accessor methods experimental or move the accessors behind a
  private seam, and consider an enum for the selector. Runtime-only solver configuration,
  not USD-authorable; no schema disposition needed beyond saying so.
- Requirements: the pseudocode's structure is implemented and evidenced; the semantic
  choices in F1, F2, F4, F5 are not derivable from the pseudocode and need agreement; F3 is a
  confirmed low-impact ordering mismatch; F7 is a precondition the algorithm should state.
- Standards: F9; otherwise compliant (Google docstrings with units, keyword-only options,
  `wp.array[...]` annotations, unittest with docstrings, Towncrier fragment, SPDX 2026).

## Appendix: review artifacts

```
review/algorithm-reference.md                 user's authoritative algorithm (unchanged)
review/task-prompt.md                         the review task (unchanged)
review/alm-dat-conformance-review.md          this report
review/checks/check_slide_planes_reference.py
review/checks/check_alm_row_reference.py
review/checks/check_truncation_invariant.py
review/checks/check_driver_trace.py
review/checks/probe_touching_start.py
review/checks/results/*.txt                   raw outputs, each stamped with the snapshot and command
```

## Addendum: work after the review on this worktree (2026-10-05, same session)

Not part of the snapshot review above; recorded here because it changes the `alm_dat`
row on branch `ankac/alm-dat-fable-review` (uncommitted, for cherry-picking).

- **Coulomb friction added to the ALM row** per `notes/alm-contact-design/design.md` section 4:
  one world-space tangential multiplier per canonical slot, projected into the current
  tangent plane, advanced by `rho_t` times the slip `(x - x_start)` of the positive witness
  relative to the negative one, clamped to the disk of radius `mu * p` with `p` the row's
  transmitted normal force, updated once per sweep after the normal update, carried across
  steps by the same sorted-row matching (`K: alm_friction_row`, `_alm_contact_response`,
  `_match_rows_kernel`, `_snapshot_rows_kernel`). `rho_t` is the inertia metric times
  `particle_contact_alm_rho_scale` without the stiffness floor: with the floor (1000 N/m in the
  tests) every loaded row saturated its disk after about 11 microns of slip and acted as kinetic
  friction with stale directions, and a patch on a 10 degree incline crept 8.6 mm uphill in 2 s;
  at the inertia scale it holds at its 1.0 mm settling offset for the whole run.
- **Gap-rate damping added** (plain formula, no multiplier): `-kd * closing / dt` with
  `closing = gap - min(gap_start, margin)`, that is only the closing motion inside the margin is
  damped, so the force is zero before contact and continuous at the margin. Two earlier variants
  were rejected by the landing diagnostics: switching the force on at the margin made the local
  solve chatter between the inside and outside of the margin (a strongly damped row's Newton step
  lands outside the margin, where every force vanishes, and falls back), and fading the damping in
  over the detection band made a strongly damped vertex hover above the margin before contact.
- Tests: `test_friction_row_clamps_to_the_coulomb_disk`, `test_friction_row_jacobian_matches_finite_differences`,
  `test_contact_damping_reduces_rebound` (3 cm drop: kd 30 cuts the rebound from 0.055 to 0.005 m/s
  and the cushion penetration from 2.5 to 1.1 mm), `test_patch_sticks_on_an_incline_with_friction`
  (mu 0.5 holds, mu 0 slides 1.09 m). All 20 tests of the three ALM-DAT modules pass on CPU.

Two pre-existing behaviors found while debugging, both outside the friction and damping scope:

- **Path-length motion budget blocks in-step corrections after a fast landing.** The initial
  truncation moves the anchor to the inertial target first (pseudocode lines 16 to 17); the commit
  kernel then clamps `used + |step| <= budget` (`K:_commit_truncated_displacement_kernel`), so once the
  inertial move has spent the budget, the ALM sweep's upward corrections are rejected and the committed
  state is the clipped inertial target regardless of contact forces or damping. Visible whenever the
  closing speed exceeds `0.5 * gamma * query_radius / dt` (0.765 m/s in the tests): 5 and 10 cm drops
  land identically with kd 0, 30 and 300. A clamp on the resulting displacement, `|anchor + a step - x_start| <= budget`
  with the largest admissible `a`, would admit corrections that move the anchor back toward `x_start`.
  The author's review lists the conservative clamp as R13 (P2); the impact consequence makes it P1.
- **Lateral kick on deep margin penetration.** When a vertex lands deep inside the margin, the rows
  of neighboring triangles (witness on their shared edge) push along tilted closest-point directions;
  a frictionless vertex picked up 0.15 m/s of lateral velocity from a 0.76 m/s vertical landing, slid
  across the 20 cm fixed patch at constant speed and dropped off its edge at step 44, which first looked
  like a resting-contact failure. Friction removes the drift; the penalty path has the same row
  structure. Worth a dedicated test once the budget clamp is revisited.

### Folded stack (2026-10-06): the ALM-DAT path does not settle on initially crushed geometry

Scene: `example_vbd_rest_on_contact --fold` (four sheets folded in half, fold gap 0.3 cm, 0.8 cm
apart, pinned cloth table, margin 0.2 cm, gap 0.1 cm, 10 substeps, 5 iterations, mu 0.5). The fold
follows the original `grid_on_table.py`: every vertex right of the sheet's world-x mid-line is
reflected over it and lifted. The sheets are rotated by up to 0.1 rad, so the crease cuts across
mesh columns and the triangles it passes through start crushed (a vertex 0.05 cm right of the line
lands 0.05 cm left of it, 0.3 cm up).

Observed (`notes/vbd-rest-on-contact/output/fold_*.json`, `fold_probe_*.json`):

- `penalty_dat` rests (sheet means 0.270 / 0.626 / 0.908 / 1.328 cm, speeds below 10 cm/s after 0.1 s).
- `alm_dat` never settles: the largest sheet speed sits on the motion-budget cap (0.1275 cm per
  substep = 76.5 cm/s) for all 300 frames; the two middle sheets are squeezed out of the stack
  sideways within 0.1 s, crumple, and leave the table edge at 0.41 s. The detector coverage flag
  stays true; the first vertex below the table surface (0.41 s, sheet 2) is 15.8 cm from the
  centre, past the 15 cm table edge, so the sheets left the table rather than crossing it
  (`fold_probe_tight.json`, `first_below`). Sheet-to-sheet crossing while being squeezed out was
  not checked. As far as checked this is a dynamics failure, not a certificate failure.
- Per-substep probe: at the first substep about 70 vertices exceed 40 cm/s on both paths, all
  within 1.5 cm of the crease (elastic snap-back of the crushed triangles). `penalty_dat` peaks at
  129 cm/s at substep 2 and is quiet by 0.1 s; `alm_dat` is capped at 76.5 cm/s and the fast set
  spreads from the crease to the whole sheet within 0.03 s.
- Controls: a clean fold along the sheet's own middle mesh column (`--fold-clean`, same
  perturbations, no crushed triangles) settles on both paths, sheet means within 0.15 cm of each
  other over the last 2 s (`foldc_*.json`; a few vertices keep jittering at 21 to 27 cm/s peak on
  both paths, as on the crushed fold's vanilla run); the unfolded stack rests on both paths
  (section 2 of the showcase page).

Cause (verified 2026-10-06 with `fold_probe.py`, files `fold_probe_mult_*.json`, `fold_probe_softdual.json`,
`fold_probe_gap04.json`, `fold_probe_sub40.json`): the motion budget locks fast vertices at the cap
speed.

- Budget per substep = `0.5 * gamma * query_radius` = 0.5 x 0.85 x 0.3 cm = 0.1275 cm; at dt = 1/600 s
  that is a speed cap of 76.5 cm/s (`_alm_dat_budget`).
- Substep 1: the crushed crease moves 52 vertices further than one budget. `_commit_truncated_displacement_kernel`
  clips each to the budget, so their velocity is exactly 76.5 cm/s.
- Next substep: the inertial target of such a vertex is exactly one budget from `x_start`. The step-start
  truncation (`_alm_dat_initialize`) spends the whole budget on it; the commit rule `used + |step| <= budget`
  then scales every later correction to zero. Elastic force, contact row, damping and gravity cannot
  change the vertex's path within the substep; only the plane truncation (`sigma`) can shorten it. This is
  the path-length budget finding above (author's R13), now with its worst consequence.
- Default run: 50 to 77 vertices clamped in every one of 180 probed substeps; the largest sheet speed is
  76.50 cm/s in all 300 rendered frames.
- Same constant, different use on the penalty path (verified: `_resolve_self_contact_schedule` returns
  `PRE_POST_INIT` for `penalty_dat` and `PRE_INIT` for `alm_dat`; `_initialize_particles` clamps the
  inertial step with `_penetration_free_truncation`; `_refresh_collision_sets` resets the reference at
  the post-init detection; `apply_truncation_ts` rescales the whole displacement vector). The penalty
  path therefore allows 0.1275 cm of inertia from `x_start` plus 0.1275 cm of corrections from the
  inertial pose (153 cm/s per substep) and always admits a correction that points back against the
  inertial motion. ALM-DAT detects once per substep, so one budget covers inertia and corrections, and
  its path-length rule rejects backward corrections once the budget is spent. The penalty path relaxes
  the crease at up to 129 cm/s, under its limit, and is quiet after 0.1 s.
- Tests. Detection gap 0.4 cm (budget 0.255 cm, cap 153 cm/s): 15 vertices clamped at substep 1, none
  afterwards, stack settles. 40 substeps (cap 306 cm/s): 7 clamped vertex-substeps in the first three
  substeps, none afterwards, stack settles. Multipliers zeroed after every substep: unchanged failure
  (cross-substep multiplier memory is not the cause). ALM metric 100x softer without floor: still fails
  (sheet order swaps at 0.185 s) but no clamped vertices remain by 0.3 s, so the within-substep
  multiplier overshoot amplifies the loop but does not start it. Clean fold: no vertex ever clamped.
- Inference (labelled): after a stored plane releases a locked vertex, the still-crushed crease asks for
  more than one budget again and the vertex locks again in the new direction; the loop injects 76.5 cm/s
  into the stack at every substep until the middle sheets are squeezed out.

Mitigation to test first: replace the path-length clamp by a displacement clamp relative to `x_start`
(`|anchor + a * step - x_start| <= budget` with the largest admissible `a`), so corrections that move a
capped vertex back toward `x_start` are admitted and forces can stop a locked vertex. Add a regression
test with a pre-stressed crease. Showcase page section 5: https://ankachen.com/report/vbd-rest-on-contact/.
