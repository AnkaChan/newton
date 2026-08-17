# Cloth Franka self-contact performance report

- Date: 2026-08-17
- Baseline: `2ba7ffd608648b485ce145173985e9556523af4c`
- Profiled candidate: `c84f62fb00d71ee000faf06476610d37bbd7751c`
- Final equivalent source commit: `a4c6a1a2`
- Branch: `ankac/cloth-franka-perf`
- Device: NVIDIA L40, CUDA device 2 (isolated by the Newton GPU claim)

## Result

Both predeclared completion gates were exceeded.

| Metric | Baseline | Candidate | Result | Target |
|---|---:|---:|---:|---:|
| Full self-contact detector + force/Hessian, 30 frames | 186.279 ms | 133.168 ms | **1.399x** | >=1.30x |
| End-to-end balanced-block throughput | 16.874 ms/frame median | 14.869 ms/frame median | **1.135x** | >=1.10x |

The component trace is 28.51% lower in summed CUDA-kernel time. The primary
end-to-end estimator is 13.52% higher throughput, corresponding to 11.91%
lower frame time. All eight counterbalanced blocks favored the candidate; the
95% block-bootstrap confidence interval is **[1.132x, 1.138x]**.

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
committed under `profiling/cloth_self_contact_results/`.

Raw `.nsys-rep` and SQLite exports are intentionally not committed: Nsight's
metadata includes the complete launch environment and can contain session
credentials. The analyzer allowlists only CUDA, Python, UV, Newton, and Warp
provenance before producing the sanitized totals in this report. The raw trace
SHA-256 values are `5b6d6331e8fe2c447113287f33b688f8b916c84a2bb68c0a5bb027988e58bca2`
for the baseline and `3790ce83dbdf36949fb7ad27858d526953e72a1dcc54272ec721f61c6ad3ddf2`
for the candidate.

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
