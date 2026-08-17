# Cloth Franka self-contact optimization log

## 2026-08-14: scope correction and baseline

- Verified branch `ankac/cloth-franka-perf` starts at `2ba7ffd6` and imports
  Newton from this worktree with the required AnkaChen git identity.
- Read the inherited profiling archive and discovered that its cloth workload
  was IsaacLab `Isaac-Lift-Cloth-Franka`, whose VBD configuration leaves
  particle self-contact disabled. Its soft face/edge results are therefore not
  a valid baseline for this task.
- Selected the native `newton.examples cloth_franka` path and the intended
  30-frame `FastExampleClothManipulation` ASV timed body. The convenience
  `run_benchmark()` helper is invalid for this stateful benchmark because its
  unmeasured warm-up exhausts `ViewerNull`; added a fresh-process harness that
  constructs/captures outside timing and runs the 30 frames once.
- Confirmed the native workload has 6,436 particles, 12,736 triangles, 19,174
  edges, five color groups, ten simulation substeps per frame, five VBD
  iterations, and self-contact enabled.

Five fresh-process baseline measurements on one dynamically claimed NVIDIA L40
were:

| Sample | ms/frame | FPS |
|---:|---:|---:|
| 1 | 16.7383 | 59.7433 |
| 2 | 16.8194 | 59.4550 |
| 3 | 16.7941 | 59.5446 |
| 4 | 16.8156 | 59.4685 |
| 5 | 16.7783 | 59.6009 |

Median baseline: **16.7941 ms/frame**. The final frame contained about 1.7k
stored vertex-triangle and 10.5k stored directed edge-edge pairs, with no row
overflow. CUDA atomic ordering makes exact trajectories/contact counts vary
slightly across fresh processes, so correctness needs synthetic equivalence
tests plus distribution/tolerance-based trajectory comparison.

The 30-frame Nsight Systems graph-node trace contains 430.3 ms of summed CUDA
kernel time. Dominant self-contact kernels were:

| Kernel | Total ms | Per frame ms | Kernel share |
|---|---:|---:|---:|
| Force/Hessian accumulation | 87.691 | 2.923 | 20.4% |
| Edge-edge detection | 51.920 | 1.731 | 12.1% |
| Planar truncation by collision | 56.652 | 1.888 | 13.2% |
| Vertex-triangle detection | 23.596 | 0.787 | 5.5% |
| BVH refit kernels | 11.280 | 0.376 | 2.6% |

The requested detection plus force/Hessian subset therefore costs at least
5.441 ms/frame before AABB/refit support work; truncation adds another 1.888
ms/frame. Force accumulation launches 250 times/frame and truncation launches
260 times/frame. Both scan each row's full capacity rather than its clamped
active count, although the frame-30 active contact set occupies only about
2.6% of the fixed 486,456-slot VT+EE capacity.

First retained-candidate hypothesis: clamp every consumer to the stored row
count, update tests to make stale tails explicitly unspecified, then remove the
full buffer sentinel clears. This attacks both the largest self-contact kernel
and truncation while preserving directed pair and overflow semantics.

## 2026-08-15: milestone 1, count-bounded consumers

Implemented clamped row-count traversal in all segmented self-contact buffer
consumers: force/Hessian accumulation, planar truncation, proxy-force harvest,
and the dormant combined accumulation kernel. Removed the VT and EE sentinel
tail clears only after every consumer was migrated. Updated detector validators
to treat tails as unspecified.

Added synthetic regression coverage for zero, partial, exact-capacity, and raw
overflow counts plus changing device counts under CUDA graph replay. Fail-first
evidence against the old loop contract showed that a zero-count row with valid
stale tail records incorrectly produced nonzero forces. The new tests passed on
CPU, CUDA, and CUDA graph replay. Focused VT and EE detector tests also passed
on CUDA.

Warm-cache candidate timings settled at 16.2905, 16.1422, 16.0043, and 15.9153
ms/frame (median 16.0732 ms/frame), a provisional **1.0449x end-to-end gain**
over the 16.7941 ms/frame baseline. The component trace confirmed:

| Component | Baseline ms / 30 frames | Candidate ms | Speedup |
|---|---:|---:|---:|
| Force/Hessian accumulation | 87.691 | 71.200 | **1.232x** |
| Planar truncation | 56.652 | 50.124 | **1.130x** |
| Core traversal + force/Hessian | 163.207 | 147.599 | **1.106x** |

This is a meaningful retained gain but does not meet the 1.30x component or
10% end-to-end target. Committed as `b3d4b82b` (`Skip inactive self-contact
buffer tails`) and continued to detection traversal optimization.

The table above is the three dominant kernels only (VT traversal, EE
traversal, and force accumulation). A follow-up SQLite audit classified the
full detector support path as the traversals plus triangle/edge AABB updates,
BVH refits and per-frame rebuild kernels, group-root updates, and self-contact
buffer fills. On that scope, milestone 1 improved detector plus force from
184.577 to 167.790 ms per 30 frames (**1.100x**). This full subtotal is the
primary completion metric; the core subtotal remains useful for explaining
hot-kernel changes.

## 2026-08-17: milestone 2, reject current misses before rest geometry

Moved the unchanged strict current-distance predicate ahead of the
rest/reference closest-point calculation in both VT and EE traversal. Hoisted
per-owner filtering bounds, reference positions, and EE buffer metadata out of
the BVH candidate loops. This preserves emitted contacts, raw/stored counts,
minimum-distance values, and overflow semantics while avoiding a second exact
narrow-phase evaluation for broad-phase candidates outside the current contact
margin.

The brute-force VT/EE comparison tests, rest-distance filtering test, and CUDA
graph-capture detector test all passed on the claimed L40 after clearing and
rebuilding the Warp cache. The source marker
`self_contact_detection_narrowphase_v1` appeared in every run.

Five warm fresh-process measurements were 15.5165, 15.5839, 15.5510, 15.7434,
and 15.6013 ms/frame (median **15.5839 ms/frame**). Relative to the pinned
16.7941 ms/frame baseline, the retained milestones now give a provisional
**1.0777x end-to-end speedup** (7.21% lower latency).

The 30-frame graph-node trace showed:

| Component | Baseline ms | Milestone 1 ms | Milestone 2 ms | Baseline speedup |
|---|---:|---:|---:|---:|
| VT traversal | 23.596 | 23.628 | 21.108 | **1.118x** |
| EE traversal | 51.920 | 52.771 | 43.075 | **1.205x** |
| Force/Hessian | 87.691 | 71.200 | 71.147 | **1.233x** |
| Core traversal + force | 163.207 | 147.599 | 135.330 | **1.206x** |
| Full detector + force | 184.577 | 167.790 | 155.522 | **1.187x** |
| Extended pipeline with truncation | 253.044 | 229.738 | 217.354 | **1.164x** |

The detector change is a repeatable win, but the primary component speedup is
still below 1.30x and end-to-end latency is still above the 10% reduction gate.
The next retained-candidate hypothesis specializes force evaluation to produce
only the current color's endpoint force/Hessian rather than materializing two
EE or four VT output pairs on every accepted contact.

### Rejected experiment: single-output force evaluators

Refactored the otherwise-unused per-vertex EE/VT evaluators to match the
multi-output arithmetic and strict distance bounds, then changed the live
accumulator to evaluate only endpoints in the current color. Exact synthetic
oracles covered directed EE, VT, shared-color stencils, damping, friction, zero
distance, the exact radius, and a representable just-inside distance. All seven
qualified CPU/CUDA/graph buffer tests and both relative-gap damping tests
passed.

Performance nevertheless regressed. After one post-cache-clear sample was
discarded, four warm fresh processes measured 16.8048, 16.4228, 16.4580, and
16.4427 ms/frame (median **16.4504 ms/frame**) versus 15.5839 ms/frame before
the experiment, about **5.6% slower**. The likely cause is that a query vertex
can share a color with a triangle vertex: the single-output design then repeats
closest-point, normal, friction, and damping work, and that cost exceeds the
saved output scaling/register pressure. Reverted the uncommitted experiment;
the multi-output implementation remains the retained path.

## 2026-08-17: milestone 3, divergence-oriented detector blocks

Swept detector block sizes 8, 16, 32, and 64 by deferring example graph
capture in the benchmark harness and setting the detector launch geometry
before capture. An initial sweep was invalidated after discovering an unrelated
Warp ASV process using physical GPU 1. Re-sourced the GPU claim, selected idle
physical GPU 2, acquired an exclusive flock for the persistent shell, warmed
all four cached variants, and reran 16 fresh processes in the interleaved order
`16 8 32 64 32 64 8 16 64 32 16 8 8 16 64 32`.

| Block size | Samples (ms/frame) | Median |
|---:|---|---:|
| 8 | 15.5114, 15.4208, 15.3898, 15.6252 | **15.4661** |
| 16 | 15.6899, 16.1092, 15.9451, 15.7030 | 15.8240 |
| 32 | 16.2113, 16.4235, 16.2526, 16.4266 | 16.3381 |
| 64 | 16.1913, 16.2185, 16.3518, 16.3357 | 16.2771 |

Although size 8 leaves most lanes in each hardware warp inactive, smaller
query groups reduce BVH traversal divergence enough to win. Paired 30-frame
Nsight traces on the isolated GPU confirmed that this is a detector win, not
wall-time noise:

| Component | Block 16 ms | Block 8 ms | Speedup |
|---|---:|---:|---:|
| VT traversal | 21.332 | 18.000 | **1.185x** |
| EE traversal | 42.796 | 38.457 | **1.113x** |
| Full detector | 84.467 | 76.762 | **1.100x** |
| Full detector + force | 155.964 | 148.138 | **1.053x** |

Changed the internal detector default from 16 to 8 threads. Relative to the
original pinned trace, the full detector plus force subtotal is now 184.577 to
148.138 ms per 30 frames (**1.246x**), and the historical three-hot-kernel
subtotal is 163.207 to 127.833 ms (**1.277x**). The isolated short-run median
is provisionally **1.086x** faster than the original 16.7941 ms/frame baseline.
Both completion thresholds remain just out of reach, so force-row color
hoisting and launch specialization continue next.

## 2026-08-17: milestone 4, skip inactive-color force rows

Hoisted the EE row-owner vertex IDs and colors out of the stored-pair loop,
then skipped rows whose owner does not belong to the active solver color
after the active-lane count gate. Hoisted the VT query color out of its pair
loop as well. The retained change preserves the force arithmetic, exact
contact semantics, and atomic accumulation order while avoiding pair-detail
loads and evaluations for irrelevant color rows.

After a fresh Warp cache build, the three focused self-contact buffer tests
covering CPU, CUDA, and CUDA graph replay all passed. Six fresh-process
end-to-end runs used the locked, isolated physical GPU 2. The first run was
the post-build cold sample at 27.226991 ms/frame and was excluded. The five
warm samples were 15.484293, 15.484597, 15.380343, 15.583087, and 15.576674
ms/frame, with a median of **15.484597 ms/frame**. That is only 0.12% slower
than the isolated block-8 median of 15.466077 ms/frame and is effectively
flat at the observed fresh-process noise level.

Paired 30-frame Nsight traces on the same isolated GPU showed the intended
force-kernel improvement without shifting detector cost:

| Component | Block 8 ms | Owner-row guard ms | Speedup |
|---|---:|---:|---:|
| Force/Hessian accumulation | 71.376309 | 69.893818 | **1.021211x** |
| Full detector | 76.761952 | 76.808356 | 0.999396x |
| Full detector + force | 148.138261 | 146.702174 | **1.009789x** |
| Extended pipeline with truncation | 209.792558 | 208.187407 | **1.007710x** |
| All graph kernels | 380.785617 | 379.121869 | **1.004388x** |

Relative to the original pinned trace, the cumulative primary full-detector
plus force subtotal is now 184.576841 to 146.702174 ms per 30 frames,
or **1.25817x** faster. The owner-row guard is retained as milestone 4: its
component gain is small but measurable, and end-to-end behavior remains flat.
The raw Nsight reports remain untracked because their process metadata is
secret-bearing; only sanitized aggregate results are recorded here.

## 2026-08-17: milestone 5, squared-distance VT rejection

Added exact squared-distance prefilters to the VT current and reference
narrow phases. Candidates that can still satisfy the original predicates
continue through the same square root and strict comparison, so stored
Euclidean minimum distances and conservative-bound inputs are unchanged.

After a fresh Warp cache build, the CPU and CUDA brute-force VT comparisons,
CUDA rest-distance filtering test, and CUDA graph-capture detector test all
passed (four tests, 65.936 seconds). Six fresh-process end-to-end runs used the
locked physical GPU 2 and the same harness SHA as milestone 4. The first
post-build sample, 26.941237 ms/frame, was excluded. The five warm samples
were 15.261878, 15.367143, 15.230597, 15.241908, and 15.306419 ms/frame,
with a median of **15.261878 ms/frame**.

The cumulative 30-frame trace relative to the original pinned baseline was:

| Component | Baseline ms | Milestone 5 ms | Speedup |
|---|---:|---:|---:|
| VT traversal | 23.595861 | 17.885711 | **1.31926x** |
| EE traversal | 51.919703 | 38.497472 | **1.34865x** |
| Full detector | 96.885593 | 76.683465 | **1.26345x** |
| Force/Hessian accumulation | 87.691248 | 69.700537 | **1.25811x** |
| Full detector + force | 184.576841 | 146.384002 | **1.26091x** |
| Core traversal + force | 163.206812 | 126.083720 | **1.29443x** |
| Extended pipeline with truncation | 253.043636 | 207.994419 | **1.21659x** |

The full graph-kernel total improved by **1.11516x** cumulatively. Compared
directly with milestone 4, the component gain is small (146.702174 to
146.384002 ms), but the warm end-to-end batch moved in the expected direction
without a traced component regression. The prefilter is retained and committed
as `798499c3`; a final interleaved baseline/candidate run remains required to
separate the cumulative gain from run-order and clock effects. Raw Nsight
reports remain untracked because their metadata is secret-bearing.

### Rejected experiment: predicate-local contact normalization

Moved EE and VT normal-vector division inside the existing strict
`0 < distance < radius` branches so rejected buffered pairs would avoid the
division. The three CPU/CUDA/graph segmented-buffer tests and both CPU/CUDA
relative-gap damping tests passed after rebuilding the Warp cache.

The five warm end-to-end samples were 15.212404, 15.252769, 16.074830,
15.336895, and 15.383125 ms/frame (median **15.336895 ms/frame**), worse than
milestone 5. A direct 30-frame trace confirmed a regression: force/Hessian
accumulation increased from 69.700537 to 70.469770 ms (**0.98908x**), while
full detector plus force increased from 146.384002 to 147.060156 ms. The
change was reverted without a commit before proceeding to masked endpoint
materialization.

### Rejected experiment: masked endpoint materialization

Added dynamic endpoint masks to the existing two-output EE and four-output VT
evaluators, leaving geometry and friction evaluation shared while skipping
scaling and damping blocks for stencil vertices outside the active color. The
selected arithmetic and caller atomic order were unchanged. Focused exact
CPU evaluator oracles covered distinct, shared, and all-same color masks,
nonzero damping/friction, zero distance, the exact radius, and a representable
just-inside distance. The full CPU/CUDA/graph segmented-buffer suite and both
relative-gap damping tests passed.

Performance regressed because the fixed return tuple remained at 128
registers/thread and the dynamic predicates added overhead. Five warm samples
were 15.315308, 15.319103, 15.353884, 15.445255, and 15.333179 ms/frame
(median **15.333179 ms/frame**). A direct trace measured force/Hessian at
70.686334 ms per 30 frames versus 69.700537 ms for milestone 5
(**0.98605x**), and full detector plus force at 147.293009 versus
146.384002 ms. The production and test changes were fully reverted without a
commit. The next experiment targets the historical one-block-per-SM launch
cap instead of adding work to the 128-register kernel.

## 2026-08-17: milestone 6, remove the force launch cap

Audited the live force launch and found that `max_blocks=SM count` limited the
76,696-logical-thread kernel to 142 physical 256-thread blocks. Nsight reports
128 registers/thread and no spills, so the L40 can resident two such blocks
per SM; the historical cap forced one block per SM and only 16.7% hardware
occupancy. Warp's grid-stride loop, deterministic record allocation, and
logical sort keys do not require this physical-grid cap.

After warming all variants, ran 15 fresh processes in interleaved capped,
2x-SM, and uncapped order on locked physical GPU 2:

| Launch | Samples (ms/frame) | Median |
|---|---|---:|
| SM cap | 15.233272, 15.164231, 15.373290, 15.446286, 15.307761 | 15.307761 |
| 2x-SM cap | 15.044321, 14.991361, 15.016258, 15.138934, 15.104884 | 15.044321 |
| Uncapped | 14.901847, 14.827233, 14.879277, 14.784452, 15.003917 | **14.879277** |

Uncapped is **1.02880x** faster than the capped control end to end in the
same interleaved batch. Removing the cap in production and running six more
fresh processes without any harness override produced 15.038452, 14.881197,
15.030876, 14.825241, 14.866989, and 14.871585 ms/frame (median
**14.876391 ms/frame**).

Paired 30-frame component traces confirmed that the gain is concentrated in
the intended force kernel:

| Component | Original baseline ms | Uncapped milestone ms | Speedup |
|---|---:|---:|---:|
| Force/Hessian accumulation | 87.691248 | 56.737749 | **1.54555x** |
| Full detector | 96.885593 | 76.650089 | **1.26400x** |
| Full detector + force | 184.576841 | 133.387838 | **1.38376x** |
| Core traversal + force | 163.206812 | 113.078285 | **1.44331x** |
| Extended pipeline with truncation | 253.043636 | 194.908449 | **1.29827x** |

The cumulative graph-kernel total improved by **1.15513x**. Relative to the
original 16.794136 ms/frame median, the production no-override median is a
provisional **1.12891x end-to-end speedup** (11.42% lower latency). Both
predeclared completion gates are now exceeded before the final two-worktree
ABBA run. The CPU/CUDA/graph segmented-buffer tests and CUDA VBD deterministic
rollout passed with the production marker
`self_contact_uncapped_force_launch_v1`. Final baseline/candidate isolation,
sanitized paired profiles, and the broader regression suite remain pending.

## 2026-08-17: final isolated ABBA and completion

Created clean sibling worktrees at baseline `2ba7ffd6` and profiled candidate
`c84f62fb`, verified both tracked trees were clean, and ran them from the same
frozen harness and environment with separate Warp caches on exclusively locked
physical GPU 2. The later commit-message hygiene pass produced final equivalent
source commit `a4c6a1a2`; both candidate commits have tree
`1b8af7981a41b378d800e673e43b445c1814d1cd`. After one excluded warm-up process
per variant, the final suite used eight alternating four-process blocks: ABBA
for odd blocks and BAAB for even blocks. All 32 measured processes were
included.

| Variant | Processes | Median ms/frame | Mean ms/frame | CV |
|---|---:|---:|---:|---:|
| Baseline | 16 | 16.874108 | 16.868155 | 0.458% |
| Candidate | 16 | 14.869220 | 14.859687 | 0.577% |

The primary balanced-block geometric speedup is **1.135169x**, with a 95%
whole-block bootstrap interval of **[1.132199x, 1.138467x]** from 200,000
resamples. All eight blocks favored the candidate; their ratios ranged from
1.129558x to 1.143837x. This is 13.52% higher throughput and corresponds to
11.91% lower frame time, exceeding the predeclared 10% end-to-end gate.

Final source-isolated Nsight captures used the same frozen harness and source
worktrees. The sanitizer structurally classified all expected graph nodes and
reported:

| Component | Baseline ms / 30 frames | Candidate ms | Speedup |
|---|---:|---:|---:|
| VT traversal | 23.886073 | 17.946966 | **1.33093x** |
| EE traversal | 53.021116 | 38.521513 | **1.37640x** |
| Full detector | 98.431936 | 76.801801 | **1.28164x** |
| Force/Hessian accumulation | 87.847182 | 56.366150 | **1.55851x** |
| **Full detector + force/Hessian** | **186.279118** | **133.167951** | **1.39883x** |
| Extended self-contact pipeline | 255.488768 | 193.834376 | **1.31808x** |
| All graph kernels | 426.319122 | 364.857619 | **1.16845x** |

The primary self-contact component gate of 1.30x is exceeded. Raw Nsight
reports and SQLite exports remain untracked because they embed the complete
process environment; only allowlisted provenance, checksums, and sanitized
aggregates are retained.

No completed timing process was excluded, no VT or EE row overflow occurred,
and source/configuration validation produced no warnings. Fresh-process
particle hashes vary within both variants because the existing CUDA path uses
nondeterministic floating-point atomics; rigid hashes were stable and the CUDA
VBD deterministic-mode rollout passed. The final focused suite additionally
passed CPU/CUDA brute-force detector comparisons, filtering and graph replay,
segmented-buffer CPU/CUDA/graph cases, relative-gap damping, and a VBD cloth
collision rollout.

Both predeclared completion thresholds are satisfied after six retained,
individually committed improvements. The complete methodology, rejected
experiments, exact metrics, and evidence-handling policy are summarized in
`profiling/CLOTH_SELF_CONTACT_REPORT.md`.

### Final packaging gate

Re-ran the exact post-format working tree with an empty, isolated Warp cache.
All 13 focused CPU/CUDA/graph tests passed in 185.242 seconds and printed the
expected production markers. A first cache-cold 30-frame benchmark process also
completed; its 26.493 ms/frame cold result is excluded from performance evidence
by the established warm-process protocol.

All pre-commit hooks pass when restricted to every task-owned changed file.
The mandated repository-wide `pre-commit run -a` was invoked as well; it is
blocked by inherited, unrelated import-lint failures and typo matches in older
profiling scripts, CSV exports, and process logs. The hook made no remaining
finding in the cloth self-contact task files.
