# VBD self-contact scheduling and bound initialization, 2026-09-14

The final 32-process comparison measures **1.046259x end-to-end throughput**
(+4.63%; 4.42% less time per frame), with a 95% whole-block bootstrap interval of
**[1.04137x, 1.05104x]**. Median frame time changes from
**14.541074 ms** to **13.917116 ms**. All eight balanced blocks favor the
candidate.

This measures the combined scheduling and bound-initialization changes against
the original takeover control `ce314e2a`. The candidate is `58ed2eb7`; both use
the same custom Warp `67621f80` and native binaries on an NVIDIA L40. The
[scheduling-only report](CLOTH_SELF_CONTACT_SCHEDULING_REPORT.md) records the
earlier 1.027224x result. The two experiments have separate process samples;
their speedup ratios do not isolate the incremental contribution of fusion.

Ordinary CUDA assigns eight threads per contact row, with 128-thread force
blocks and 64-thread truncation blocks. The force pass now also initializes
every particle's truncation bound, allowing the following truncation pass to
omit its separate fill. Initialization and direct truncation calls still fill
their bounds, and computed final minima remain available. CPU, deterministic
CUDA, and proxy-force harvesting retain their existing execution paths.

## Measurements

Eight alternating ABBA/BAAB blocks contain 32 fresh processes of 30 frames each.
One cache warmup per variant is excluded, and every planned timed observation
is retained. Baseline process 12 is flagged by the modified-Z outlier diagnostic
and remains included in every estimate. Source bytes, commits, custom
Warp/native binaries, assets, benchmark, analyzer, GPU identity, physics, and
launch settings are pinned.
The [sanitized results](cloth_self_contact_results/20260914-fused.json) retain
all process timings, confidence-interval settings, resource measurements,
diagnostics, and evidence hashes. No final-frame contact-buffer rows overflowed
in any timed run.

Matched Nsight Systems traces attribute **7,800 → 300 truncation-bound fills**,
removing exactly **7,500 graph kernel launches**. The retained 300 fills cover
initialization. Structural attribution identifies fills immediately before
planar truncation, following a forward step or elasticity solve; generic fills
elsewhere are excluded. Measured graph counts are 63,870 → 56,370.

| Component, 30 frames | Control ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| Force/Hessian | 54.277625 | 47.769994 | 1.1362x |
| Truncation-bound fill | 6.636528 | 0.248354 | 26.7220x |
| Force/Hessian + bound fill | 60.914153 | 48.018348 | 1.2686x |
| Planar truncation | 47.429432 | 43.877567 | 1.0809x |
| Extended self-contact pipeline | 183.684620 | 173.370219 | 1.0595x |
| Extended pipeline + bound fill | 190.321148 | 173.618573 | 1.0962x |
| All frame-graph kernels | 355.084431 | 338.557238 | 1.0488x |

The analyzer's extended-pipeline definition is unchanged and excludes
truncation-bound fills; the additional sum is labeled explicitly. These single
profiled captures locate the savings. Unordered atomic accumulation can change
trajectories and contact ordering, so they do not establish an exact causal
decomposition. Profiled wall-clock timings are excluded from the end-to-end
estimate.

## Compiled code and validation

The existing four- and eight-thread force kernels retain their normalized PTX
instruction streams. Fusion adds 18 integer, control, and memory instructions,
bringing the force kernel from 1,699 to **1,717 static instructions**. All
**694 floating arithmetic instructions** retain their operands and order, and
the **72 atomic sites** are preserved. The fused kernel also matches the
successful direct prototype's full normalized instruction stream. Actual
driver-JIT allocation remains **128 registers per thread**, with zero local
memory and 512 bytes of static shared memory at block size 128.

The deliberate force-body copy preserves those arithmetic contractions.
Factoring it into a shared Warp function inlined successfully but changed
floating-point contraction choices, so that implementation was rejected.
Only the fused variant initializes bounds; the original kernel signatures and
shared module-option handling are retained.

All **108 broader regression tests** and **seven focused scheduling/reset
tests** pass. Tests cover poisoned and empty rows, live-count changes, CUDA
graph replay, nontrivial retained minima, direct calls, nonzero proxy forces,
and interleaved explicit/inherited determinism settings. A separate comparison
with actual pre-change kernels passes **450 stored-bit array checks** on CPU
and both deterministic CUDA modes, including captured replay. Removing the
reset produces **15 numerical failures** across five count phases and three
colors, demonstrating that the regression test detects the missing behavior.

Changed-file pre-commit checks pass. The repository-wide invocation still
reports inherited issues in unrelated profiling sources and archived CSV
hashes; its unrelated edits were inspected and reversed.

Earlier Nsight Compute attempts could not reserve a driver profiling resource
and produced no hardware-counter report. Their saved metadata records commands,
tool versions, source/native-library hashes, GPU identity, and diagnostic logs.
Achieved occupancy, warp stalls, and memory traffic therefore remain
unmeasured. Register allocation comes from successful live driver queries and
Systems traces, rather than offline assembler estimates.

Raw evidence remains under the ignored
`profiling/cloth_self_contact/20260914/integrated/` directory. The committed
aggregate uses explicit field selection and excludes process environments and
raw profiler metadata.
