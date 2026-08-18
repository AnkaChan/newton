# Cloth Franka self-contact performance report

- Dates: 2026-08-17 through 2026-08-18
- Baseline: `2ba7ffd608648b485ce145173985e9556523af4c`
- Profiled candidate: `c84f62fb00d71ee000faf06476610d37bbd7751c`
- Final equivalent source commit: `a4c6a1a2`
- Detector block-size follow-up: `647a1f1a`
- Radius-query AABB control: `7134ddfa`
- Radius-query candidate: `2feb9748`
- Radius-query Warp: measured `4491567b`, final equivalent `67621f80`
- Current-main comparison: upstream `6d3fdf6f`, candidate `29fa8719`
- Branch: `ankac/cloth-franka-perf`
- Device: NVIDIA L40 (isolated by the Newton GPU claim for each suite)

## Result

The original `2ba7ffd6` baseline-to-`a4c6a1a2` optimization exceeded both
predeclared completion gates.

| Metric | Baseline | Candidate | Result | Target |
|---|---:|---:|---:|---:|
| Full self-contact detector + force/Hessian, 30 frames | 186.279 ms | 133.168 ms | **1.399x** | >=1.30x |
| End-to-end balanced-block throughput | 16.874 ms/frame median | 14.869 ms/frame median | **1.135x** | >=1.10x |

The component trace is 28.51% lower in summed CUDA-kernel time. The primary
end-to-end estimator is 13.52% higher throughput, corresponding to 11.91%
lower frame time. All eight counterbalanced blocks favored the candidate; the
95% block-bootstrap confidence interval is **[1.132x, 1.138x]**.

The later radius-query follow-up is a separate incremental comparison on top
of that optimized implementation. It holds the final Warp build constant and
compares padded AABB queries with VT sphere and EE capsule queries. That
follow-up improves the frozen production traversal pair by **1.061696x**, the
traced detector by **1.039724x**, and end-to-end throughput by **1.005657x**,
95% CI **[1.002605x, 1.008975x]**. It does not replace or compound the
original baseline-to-candidate result.

## Direct current upstream/main comparison

A separate user-requested ABBA suite directly compared official
`upstream/main` `6d3fdf6f7885378677d9b69899aad1ee5bd6c667` with final candidate
`29fa8719a5c5a1277b1d1fdde3d68090bd7d08b9`:

| Variant | Processes | Median ms/frame | Mean ms/frame | CV |
|---|---:|---:|---:|---:|
| Current upstream/main | 16 | 17.272598 | 17.314414 | 0.592% |
| Final candidate | 16 | 14.494030 | 14.500801 | 0.673% |

The primary balanced-block throughput speedup is **1.194037x** (+19.4037%),
with a 200,000-resample complete-block 95% confidence interval of
**[1.189707x, 1.198300x]**. All eight blocks favored the candidate. The median
frame latency is 16.09% lower (17.272598 to 14.494030 ms/frame). All 32 included
30-frame processes completed, all observations were retained, the analyzer
emitted no warnings, and no contact row overflowed.

This is an aggregate current-main-versus-feature result, not an isolated
radius-query or single-change ablation. The branches diverge at `fd8d9d4e`;
main and the candidate have 80 and 27 unique commits respectively. The target
`example_cloth_franka.py` is byte-identical in both trees (Git blob `e41e136e`,
SHA-256 `16b2c9d3...383c`), as is `unisex_shirt.usd` (SHA-256
`9eb7f161...4c5`). The Franka asset cache used by each process has the same 64
non-Git files and 80,323,199 bytes; their normalized content fingerprint is
`26b1bf34...6758`. Current main resolves a newer asset-repository commit, but
the target Franka sparse-checkout contents are byte-identical.

Both variants used the exact same final custom Warp `67621f80`, tree
`4dcf21d7`, and native binaries (`warp.so` SHA-256 `d071cd89...a43` and
`warp-clang.so` SHA-256 `5a436646...691`). This controls Warp but means the
result is not a stock-main dependency comparison. Each Newton tree retained its
production launch policy without overrides: main resolved VT16/EE16 and capped
force/Hessian accumulation at the L40's 142 SM blocks; the candidate resolved
VT4/EE8 and used the natural uncapped force grid. Both variants also used the
candidate worktree's common frozen virtual environment while `PYTHONPATH` and
Newton/Warp caches were isolated per variant. This controls Python dependencies
but does not compare each branch's own lockfile-derived environment.

The suite is `/tmp/cloth-franka-main-vs-final-20260818`. Primary evidence
SHA-256 values are `f7851419...b31` (`manifest.json`),
`59ab5160...77c` (`analysis.json`), `d5a67ce4...9db` (`summary.json`), and
`6caef88a...d83` (`runs.csv`). The frozen runner, benchmark, and analyzer hashes
are `14cde073...0f1`, `b5c2f79f...cae`, and `5cc0253b...c27` respectively.

One 30-frame Nsight Systems graph-node trace per variant localizes the aggregate
gain under the same pinned-source, shared-Warp, common-environment,
isolated-cache, and production-launch controls:

| Component | Current main ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| Vertex-triangle traversal | 21.804819 | 13.299248 | **1.639553x** |
| Edge-edge traversal | 49.115971 | 36.738971 | **1.336890x** |
| Full detector | 92.370216 | 70.223996 | **1.315365x** |
| Force/Hessian accumulation | 85.274545 | 54.052334 | **1.577629x** |
| **Full detector + force/Hessian** | **177.644761** | **124.276330** | **1.429434x** |
| Planar truncation | 55.001940 | 47.425850 | **1.159746x** |
| Extended self-contact pipeline | 244.488994 | 183.536550 | **1.332100x** |
| All captured graph kernels | 430.530160 | 353.734332 | **1.217100x** |

Both traces contain 30 frame graphs, 63,870 graph kernels, all expected
component launch counts, and no analyzer warnings. The two fresh CUDA processes
followed different trajectories and ended with different contact totals because
of the solver's unordered floating-point atomics. These sums localize work but
are not a paired statistical estimate, and profiler wall time is invalid for
end-to-end comparison; the 32-process ABBA result remains authoritative.

Nsight itself emitted two diagnostic warnings per trace: the 12.8 driver is
newer than this Nsight 2024.5 build, so it used its CUDA 12.6 tracing libraries,
and a generic warning said that not all CUDA events might have been collected.
There were no severity errors. All 30 frame markers, total graph-kernel counts,
and expected per-component/per-frame launch counts are present, so there is no
evidence that an expected workload kernel was omitted; the diagnostics remain
a trace-level caveat.

Trace evidence is under `/tmp/cloth-franka-main-vs-final-nsys-20260818`.
`trace-analysis.json` has SHA-256 `9e71cbef...b90`; the fail-closed trace runner
has SHA-256 `8e5f0a23...045`. Raw main/candidate report hashes are
`e30c6d60...a0b`/`3535c281...ff7`, SQLite hashes are
`b6e3186e...4bb`/`2b7c0c0d...8e4`, and benchmark-result hashes are
`493d4beb...14b`/`4d7aadb3...450`.

## Workload and measurement

The inherited IsaacLab cloth traces were not used as the baseline because that
configuration leaves VBD particle self-contact disabled. Measurements use the
native `example_cloth_franka` simulation exercised by the repository's
`FastExampleClothManipulation` benchmark shape:

- one 6,436-particle shirt with 12,736 triangles and 19,174 edges;
- ten simulation substeps per 60 Hz frame and five VBD iterations per substep;
- self-contact radius and margin 0.2, with 16 VT and 20 EE records per row;
- CUDA graph capture, no visualizer, and 30 timed frames per fresh process.

The final end-to-end comparison used clean baseline and candidate worktrees,
separate Warp caches, one excluded warm-up process per variant, then eight
alternating four-process blocks. Odd blocks used baseline/candidate/candidate/
baseline order and even blocks reversed it. No completed process was excluded.
The geometric mean of the eight balanced block ratios is the primary speedup.
The confidence interval resampled whole four-process blocks 200,000 times.

| Variant | Processes | Median ms/frame | Mean ms/frame | CV |
|---|---:|---:|---:|---:|
| Baseline | 16 | 16.874108 | 16.868155 | 0.458% |
| Candidate | 16 | 14.869220 | 14.859687 | 0.577% |

Block speedups were 1.14384x, 1.13117x, 1.13620x, 1.13066x, 1.13969x,
1.12956x, 1.13521x, and 1.13511x.

## Profile breakdown

Separate process-isolated Nsight Systems captures used the same frozen harness,
source worktrees, workload, and GPU as the end-to-end suite. Durations below
are sums of CUDA graph-node kernel durations over 30 frames.

| Component | Baseline ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| Vertex-triangle traversal | 23.886 | 17.947 | **1.331x** |
| Edge-edge traversal | 53.021 | 38.522 | **1.376x** |
| Full detector support | 98.432 | 76.802 | **1.282x** |
| Force/Hessian accumulation | 87.847 | 56.366 | **1.559x** |
| **Full detector + force/Hessian** | **186.279** | **133.168** | **1.399x** |
| Planar truncation | 57.410 | 48.846 | **1.175x** |
| Extended self-contact pipeline | 255.489 | 193.834 | **1.318x** |
| All captured graph kernels | 426.319 | 364.858 | **1.168x** |

The full detector subtotal includes VT/EE traversal, triangle and edge AABB
updates, BVH refits and per-frame rebuild support, group-root updates, and
self-contact buffer initialization. This is the scope-faithful completion
metric; the narrower traversal-plus-force subtotal improved 1.460x.
The profiler's instrumented wall times are intentionally not used as
end-to-end evidence. Each trace is one fresh process and inherits the solver's
normal atomic-order trajectory variation, so the kernel totals localize the
gain while the 32-process ABBA suite supports the workload-level claim.

## Adaptive AABB detector block-size follow-up

After completing the baseline-to-candidate comparison above, a frozen-state
follow-up swept independent VT and EE launch sizes. The native cloth state was
replicated into 1, 2, 4, 8, and 16 independent collision worlds so the launch
geometry could be varied without changing per-world contacts. All graph
specializations were warmed before the balanced event-timing schedule.

| World copies | Best VT block: us | Best EE block: us | Best common block: us | Best split: us |
|---:|---:|---:|---:|---:|
| 1 | 4: 63.602 | 8: 172.789 | 8: 251.976 | 4/8: 236.391 |
| 2 | 8: 80.453 | 16: 245.180 | 16: 342.536 | 8/16: 325.633 |
| 4 | 12: 95.529 | 16: 387.354 | 16: 490.023 | 12/16: 482.883 |
| 8 | 24: 155.968 | 24: 663.185 | 24: 819.153 | 24/24: 819.153 |
| 16 | 16: 275.755 | 64: 1173.246 | 64: 1464.803 | 16/64: 1449.002 |

These are tested-grid optima for the synthetic replicated workloads, not a
general prescription for those mesh sizes. In particular, the K=16 EE 32/64
difference and K=8 VT 24/64 difference were only 0.106% and 0.093%.

Every tested specialization produced bitwise-identical live count arrays,
canonical stored pairs, owner minimum distances, and resize flags. Each
replicated world retained exactly 1,739 VT and 10,652 EE pairs, targets stayed
within their world, and no row overflowed. Triangle-owned reverse VT buffers
were not included in the snapshot because `record_triangle_contacting_vertices`
is disabled for this workload; they are not live solver outputs here.

The then-profiled AABB BVH query is a scalar per-thread traversal with no warp
vote, shuffle, or barrier. Its 32-entry stack is laid out per CUDA thread in
shared memory, so sub-warp blocks are correct but still consume a physical
warp. The native L40 trace reported 106 VT registers/thread and reduced static
shared memory from 1,056 bytes at block 8 to 528 bytes at block 4. VT block 4
launches 1,609 CTAs instead of 805, giving the irregular traversal more
independent work; EE already launches 2,397 CTAs at block 8.

Correct CUDA graph-node captures contained 300 VT calls per variant. VT time
fell from 17.855490 ms at block 8 to 14.583589 ms at block 4 over 30 frames,
an additional **1.224355x** traversal speedup. EE remained at block 8 and was
effectively unchanged. The earlier captures made without graph-node tracing
contained no detector kernels and were discarded.

The authoritative end-to-end follow-up used 32 fresh processes in eight
alternating BCCB/CBBC blocks, comparing common 8/8 against split VT4/EE8. The
balanced-block geometric speedup was **1.006617x**, with a 95% whole-block
bootstrap interval of **[1.003078x, 1.010002x]**. This is an incremental
comparison on top of the optimized candidate and does not replace the original
1.135x baseline-to-candidate result.

After committing the production policy, a smaller clean-tree confirmation ran
four BCCB/CBBC blocks at `647a1f1a`, with warm-up processes excluded:

| Variant | Processes | Median ms/frame | Mean ms/frame | CV |
|---|---:|---:|---:|---:|
| Explicit common VT8/EE8 | 8 | 14.834293 | 14.826974 | 0.604% |
| Production adaptive VT4/EE8 | 8 | 14.688145 | 14.735277 | 0.562% |

All four blocks favored production, for a geometric speedup of **1.006221x**
and 0.618% lower latency; its 100,000-resample whole-block interval was
**[1.004225x, 1.008446x]**. All processes used the same clean source and
harness hashes, the control resolved to VT8/EE8 through an explicit override,
production resolved to VT4/EE8 without an override, and no row overflowed.
This confirms wiring at the committed source; the larger 32-process result
remains the authoritative estimate.

Commit `647a1f1a` retains the default maximum of eight threads but halves VT or
EE independently until an uncapped CUDA launch supplies at least eight CTAs per
SM. It therefore resolves cloth Franka on the 142-SM L40 to VT4/EE8 while
preserving explicit settings, the compatibility attribute, and CPU block 8.
The added `unittest` covers every halving boundary, the L40 cloth resolution,
the CPU default, and explicit-setting preservation; the frozen graph sweep
provides live CUDA output-equivalence coverage.

## Sphere/capsule radius-query follow-up

The follow-up replaces the padded point AABB VT query with
`bvh_query_sphere`, and the padded segment AABB EE query with
`bvh_query_capsule`. The EE query passes its unnormalized end-minus-start
direction with `max_dist=1`; a degenerate edge uses a unit direction with
`max_dist=0`, reducing it to a radius point query. Newton enables this path
only when Warp publicly exposes the radius-query types, so the supported Warp
1.16 configuration retains the AABB implementation. The public types prove API
availability, not the presence of later traversal safety fixes; final evidence
therefore pins the exact Warp commit and native-library hashes below.

This experiment compares clean Newton AABB control `7134ddfa` with clean
radius-query candidate `2feb9748`. Both use the exact same measured fixed Warp
build, launch policy, assets, harness, and frozen cloth state; the only
production difference is AABB versus sphere/capsule broad phase.

Exact query-volume accounting was captured earlier on Newton `b963b1b1` and
the requested Warp merge `7c66e2f6`, using the same frozen cloth state. It
shows that the radius volumes are strict subsets of the padded AABBs:

| Query | AABB candidates | Radius candidates | Reduction |
|---|---:|---:|---:|
| Vertex-triangle | 54,718 | 50,724 | 3,994 (7.2992%) |
| Edge-edge | 344,838 | 302,261 | 42,577 (12.3470%) |

Neither radius query introduced an extra candidate. These counts document the
geometric pruning; all timing and live-output claims below come from fresh
reruns on measured Warp `4491567b`. Commit-message hygiene later rewrote that
commit to final `67621f80` without changing its tree, source, or binaries.

The authoritative frozen sweep used 2,000 graph warmups, 1,000 launches per
sample, nine samples, and blocks 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128,
256, and 512. The independently tuned and production-policy results were:

| Scope | AABB | Sphere/capsule | Speedup |
|---|---:|---:|---:|
| VT independent optimum | 55.644257 us, block 4 | 55.431679 us, block 4 | **1.003835x** |
| EE independent optimum | 178.495941 us, block 8 | 165.102463 us, block 8 | **1.081122x** |
| Each variant's best same-block VT+EE policy | 248.563911 us, block 8 | 229.901154 us, block 8 | **1.081177x** |
| Production VT4 + EE8 | 234.140198 us | 220.534142 us | **1.061696x** |

The production split saves 13.606056 us per collision substep, or 0.136061 ms
over the ten substeps in one frame. Every AABB and radius specialization at all
14 block sizes produced bitwise-identical final counts, canonical active
pairs, owner minimum distances, and resize flags: 1,739 VT and 10,652 EE
contacts, with no overflow. The sweep therefore both re-profiles the launch
sizes through block 512 and verifies the complete live detector outputs.

The final end-to-end suite used one excluded warm process per variant followed
by 32 included fresh processes in eight alternating BCCB/CBBC blocks. It used
the production VT4/EE8 policy without a launch override:

| Variant | Processes | Median ms/frame | Mean ms/frame | CV |
|---|---:|---:|---:|---:|
| AABB control | 16 | 14.505825 | 14.519181 | 0.673% |
| Sphere/capsule | 16 | 14.423435 | 14.437344 | 0.450% |

The balanced-block geometric speedup is **1.005657x**, with a 200,000-resample
whole-block 95% interval of **[1.002605x, 1.008975x]**. Seven of eight blocks
favored the radius queries. This is 0.5657% higher throughput, corresponding to
about 0.5625% lower latency: small, but statistically resolved by the declared
estimator. All 32 measured processes completed, the analyzer emitted no
warnings, and no contact row overflowed.

One source-isolated Nsight trace per variant localizes the detector effect.
Each trace contains the expected 30 frame graphs and identical kernel counts:

| Component | AABB ms | Sphere/capsule ms | Speedup |
|---|---:|---:|---:|
| Vertex-triangle traversal | 13.145353 | 13.105166 | **1.003067x** |
| Edge-edge traversal | 39.118766 | 36.442143 | **1.073449x** |
| Full detector | 72.515473 | 69.744900 | **1.039724x** |
| Force/Hessian accumulation | 53.561322 | 53.617444 | 0.998953x |
| Full detector + force/Hessian | 126.076795 | 123.362344 | **1.022004x** |
| Extended self-contact pipeline | 183.707334 | 181.713332 | **1.010973x** |
| All captured graph kernels | 353.940709 | 351.709760 | **1.006343x** |

The full detector subtotal includes the traversals, AABB updates, refits,
rebuild support, group-root updates, and buffer initialization. The EE gain is
consistent with the frozen sweep; the unrelated force and truncation shifts
reflect that fresh CUDA processes do not follow bitwise-identical cloth
trajectories or contact histories because of unordered floating-point atomics.
The traces are therefore localization evidence only. Their
profiler-instrumented wall times are invalid for end-to-end comparison; the
32-process balanced suite above is authoritative for that scope.

### Radius-query release safety

The requested Warp merge initially made EE capsule traversal a large
regression. Its stackless skip-link implementation took 415.9061 us at its
block-2 optimum versus 175.0496 us for AABB at block 12. The unusual two-thread
optimum was a latency-hiding symptom of the dependent skip-link traversal, not
a correctness requirement. A compact right-first, test-on-pop capsule stack
recovered the performance, then two fail-first release checks exposed separate
safety defects:

- a 33-deep host SAH adversary crashed the original 32-entry capsule stack;
- a direct 64-entry per-thread stack fixed the depth case but made the default
  block-256 CUDA module require 66,560 bytes of static shared memory, above the
  L40 toolchain's 49,152-byte limit.

Intermediate Warp `91a49d1a` made capsule block 256 safe, but generic volume
queries still allocated 32 slots per thread at blocks 512 and 1024 and could
again exceed static shared-memory limits. It is historical diagnostic evidence,
not the final build.

Measured Warp `4491567b`, final equivalent `67621f80`, bounds both volume and
capsule stacks. CPU and CUDA blocks up to 64 retain 64 slots. Blocks 128, 256,
512, and 1024 use 32, 16, 8, and 4 slots per query respectively, switching an
overflowing subtree to skip-link continuation. Each query type consumes at
most 16 KiB of shared stack, and a mixed volume/capsule kernel uses at most
36 KiB including other static state.

The focused Warp suite passes sphere/capsule semantics, maximum-depth SAH and
grouped LBVH cases, packed leaves, decreasing `max_dist`, exact-once
enumeration, canaries, and sequential mixed query types through block 1024.
The cloth detector kernels themselves cannot launch at block 1024 because
their roughly 100 registers per thread exceed a separate per-block register
limit; the production kernels are output-exact through the highest launchable
sweep size, block 512. Final Newton tests pass collision 4/4, segmented buffers
3/3, damping, deterministic rollout, cloth collision, and the Warp 1.16 CPU
fallback.

The final fixed Warp provenance is:

- measured commit `4491567b4fa0d59e3d2578dbfb8a0849e1c67fc2`;
- final commit after message-only history rewrite
  `67621f8074b045673c2b72db5f2c8ce5e9b2cbc6`;
- tree `4dcf21d7714f3fd481d4eb0fd3eb3f4565e408c6`;
- source SHA-256
  `5a91716353e5814287d7ae5908dae0768aff75632a0dc5f7a894b13db15772de`
  over 598 files and 13,936,215 bytes;
- `warp.so` SHA-256
  `d071cd89a45f66bb2784dcd0e942852211fde3f6ec68fcffdf131d48e211ca43`;
- `warp-clang.so` SHA-256
  `5a4366461d9acf30d598354448b5894f88d2c30ddb1a9c76155034cb869fe691`.

## Retained changes

1. `b3d4b82b` — bound all segmented-buffer consumers by each row's clamped
   active count, make tails explicitly unspecified, and remove full VT/EE
   sentinel clears.
2. `00735cc8` — reject candidates outside the current contact radius before
   evaluating reference/rest geometry and hoist owner-invariant metadata.
3. `193ddf11` — reduce VT/EE traversal blocks from 16 to 8 threads to reduce
   divergence in the small-query BVH traversals.
4. `6a9bc1c2` — skip force rows whose owner vertices do not participate in the
   current solver color and hoist row metadata behind the active-count gate.
5. `798499c3` — use exact squared-distance VT prefilters before the original
   square root and strict acceptance tests.
6. `a4c6a1a2` — remove the historical one-block-per-SM force-launch cap. The
   128-register kernel can resident two 256-thread blocks per L40 SM, so the
   natural grid doubles active warps without changing logical indexing.
7. `d01d72af` — add the frozen detector benchmark and exact live-output checks
   across independent VT/EE launch sizes and replicated collision worlds.
8. `b5fb5264` — support independent VT and EE launch overrides in the
   end-to-end cloth Franka benchmark.
9. `647a1f1a` — resolve default CUDA VT and EE block sizes independently from
   the primitive count and SM count while preserving explicit and CPU sizes.
10. `b963b1b1` — use sphere and capsule BVH queries for VT and EE broad phase
    when the installed Warp supports them.
11. `02d5c217` — add a provenance-checked diagnostic that separates capsule
    traversal cost from avoided EE narrow-phase work.
12. `2feb9748` — gate radius-query dispatch on public Warp API availability;
    the exact fixed Warp pin above supplies the separate safety provenance.

Three measured force experiments were rejected and fully reverted: repeated
single-output stencil evaluation (about 5.6% slower end to end), predicate-local
normalization (force kernel 0.989x), and masked fixed-tuple outputs (force
kernel 0.986x). Their evidence remains in the optimization log.

## Correctness

The buffer-contract regression test fails against the old implementation when
a zero-count row contains valid stale tail records, then passes with the new
count-bounded consumers. It covers zero, partial, exact-capacity, and overflow
raw counts, all affected consumers, and changing counts under CUDA graph replay.

The final candidate passed:

- CPU and CUDA brute-force VT and EE detector comparisons;
- CUDA rest/filter behavior and detector graph-capture replay;
- CPU, CUDA, and CUDA-graph segmented-buffer tests;
- CPU and CUDA relative-gap self-contact damping tests;
- the CUDA VBD deterministic rollout; and
- the VBD cloth-collision rollout.

The subsequent radius-query candidate passed the same Newton detector,
segmented-buffer, damping, deterministic-rollout, and cloth-collision coverage
against the final Warp build. Warp's focused CPU and CUDA query suites also
cover finite and decreasing distance bounds, degenerate and boundary capsules,
grouped roots, packed leaves, and adversarial maximum-depth trees through
generic block 1024. The frozen 14-block comparison adds bitwise live-output
equivalence against the AABB control through launchable production block 512.

After the final formatter and commit-message hygiene pass, the exact working
tree was retested with a new Warp cache: all 13 focused CPU/CUDA/graph tests
passed in 185.242 seconds and all required source-version markers appeared. A
first cache-cold benchmark process also completed successfully; as specified by
the timing protocol, its cold result was not included in performance claims.

Fresh CUDA processes are not bitwise trajectory-deterministic because the
existing solver uses unordered floating-point atomics. Accordingly, state
hashes vary within both variants and are retained only as diagnostics. Rigid
state hashes were stable, contact distributions remained in the same regime,
no VT or EE row overflow occurred in any of the 32 final runs, and the explicit
deterministic-mode test passed.

## Reproduction and evidence handling

`benchmark_cloth_franka_self_contact.py` records source, workload, device,
state, contact, timing, and launch provenance. The companion analyzer validates
frozen tool hashes, live source hashes and git trees, every run's configuration,
alternating block structure, result checksums, and trace kernel counts before
computing results. Sanitized per-process timings and trace component totals are
committed under `profiling/cloth_self_contact_results/`. The credential-safe
radius-query aggregate is
`profiling/cloth_franka_radius_query_results.json`; it records the final source,
native-library, frozen-sweep, ABBA, and trace metrics without the raw process
environment.

Raw `.nsys-rep` and SQLite exports are intentionally not committed: Nsight's
metadata includes the complete launch environment and can contain session
credentials. The analyzer allowlists only CUDA, Python, UV, Newton, and Warp
provenance before producing the sanitized totals in this report. The raw trace
SHA-256 values for the original baseline-to-candidate pair are
`5b6d6331e8fe2c447113287f33b688f8b916c84a2bb68c0a5bb027988e58bca2`
and `3790ce83dbdf36949fb7ad27858d526953e72a1dcc54272ec721f61c6ad3ddf2`.
The corresponding values for the later AABB and radius-query traces are
`ca39f75f814b57c0e85f21997c3096db75df1c252d3c908cad9accc26c84def6`
and `8b9603d1980f0514f4b0e1f6f1bbb615df442055e80a6f13a6582689680cc59d`.

The final hygiene pass reworded the six local commit messages to the required
72-column format after measurement. Profiled commit `c84f62fb` and final source
commit `a4c6a1a2` both resolve to tree
`1b8af7981a41b378d800e673e43b445c1814d1cd`; no measured source changed.
The report commit subsequently applied formatter-only layout changes to one
detector call and two tests; the fresh-cache final-tree gate above covers them.

All pre-commit hooks pass on every task-owned changed file. The required
repository-wide `pre-commit run -a` was also invoked; it remains blocked by
pre-existing lint and typo findings in unrelated inherited profiling scripts,
CSV exports, and logs. No such finding remains in this task's files.
