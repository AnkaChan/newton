# VBD self-contact scheduling, 2026-09-14

The 32-process comparison measures **1.027224x end-to-end throughput**
(+2.72%), with a 95% whole-block bootstrap interval of
**[1.02450x, 1.03041x]**. Median frame time falls from **14.540967 ms** to
**14.161186 ms**. All eight balanced blocks favor the candidate. The matched
Systems trace measures a **1.0635x** improvement in the extended self-contact
pipeline.

The retained change assigns eight CUDA threads to each collision row for
ordinary, nondeterministic self-contact force evaluation and planar truncation.
Force evaluation uses 128-thread blocks and truncation uses 64-thread blocks.
CPU and deterministic CUDA retain their original four-thread, 256-thread-block
execution. Proxy force harvesting also retains its original traversal.

The control is the takeover commit `ce314e2a`; the measured candidate is
`7a156e66`. Both use the same custom Warp commit `67621f80` and native binaries
on an NVIDIA L40. This comparison isolates the new Newton scheduling change
from the earlier Newton/Warp optimizations.

## Measurements

The final end-to-end estimate comes from 32 fresh processes in eight alternating
ABBA/BAAB blocks, with 30 frames per process and one excluded cache warmup per
variant. The suite pins source bytes, commits, native libraries, benchmark,
analyzer, assets, GPU identity, and solver settings. All completed timed
observations are included.

The [sanitized aggregate](cloth_self_contact_results/20260914-scheduling.json)
records all 32 timings, source/native-library pins, trace totals, actual JIT
resources, semantic diagnostics, and evidence hashes. No contact buffer
overflows in the timed runs. These results compare against the already
optimized takeover code; they are separate from the historical combined
Newton/Warp endpoint comparison.

The matched Nsight Systems traces contain 63,870 kernels in their 30 frame
graphs and 65,612 kernels overall in each variant. The structural analyzer
reports no warnings. Summed frame-graph durations are:

| Component | Control ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| Force/Hessian | 54.487480 | 47.336315 | 1.1511x |
| Planar truncation | 47.295959 | 43.873965 | 1.0780x |
| Detector | 70.131754 | 69.756201 | 1.0054x |
| Detector + force/Hessian | 124.619234 | 117.092516 | 1.0643x |
| Extended self-contact pipeline | 183.853941 | 172.881038 | 1.0635x |
| All frame-graph kernels | 355.418067 | 344.321797 | 1.0322x |

These traces localize the improvement. Unordered floating-point atomics produce
different trajectories even between fresh runs of the same variant, so trace
differences do not provide an exact causal decomposition. Profiled wall-clock
time is excluded from the end-to-end estimate.
The extended-pipeline definition is unchanged from the existing analyzer and
does not include the separate truncation-buffer fills.

## Compiled code and behavior

The compiled four-thread kernels have the same normalized PTX instruction
streams as the control. The eight-thread variants change 11 integer scheduling
operands per kernel; floating-point arithmetic, atomic instructions, instruction
counts, and opcode histograms are unchanged. The force and truncation kernels
still have 1,699 and 1,318 static instruction sites respectively. Actual driver-JIT
register counts remain 128 and 102 per thread. PTX virtual registers and offline
assembler register counts are not used as hardware occupancy measurements.

The tests verify exact individual force/Hessian contributions, contact
multiplicity, truncation minima, stale tails, capacities above eight, overflow,
degeneracies, friction, damping, and changing counts in CUDA graphs. Integration
tests execute contact and nonzero proxy-force launches while interleaving six
explicit/inherited determinism configurations. Explicit and inherited
deterministic rollouts agree bitwise. A separate comparison against the actual
pre-change kernel module passes 450 bitwise checks on CPU and both deterministic
CUDA modes, including captured replay and switching between kernel variants.

Mutation checks fail when the eight-thread kernel receives the old undersized
grid, and when interleaved solvers fail to restore module options. All five new
scheduling tests and nine existing buffer/determinism checks pass. All 100 tests
in the VBD module pass across the initial run and the rerun of 14 checks after
restoring a missing `math` import; the missing-import error also reproduces on
the unchanged control. The import repair is a separate commit, `7eba9b6a`.

Task-file pre-commit hooks pass. The required repository-wide invocation still
reports inherited findings in unrelated profiling scripts and archived CSV
hashes. Its edits to those unrelated files were inspected and reversed.

## Other experiments and profiler limitation

Active-row compaction slowed force evaluation by about 6% on frozen inputs and
about 0.8% in an end-to-end pilot. Truncation compaction also lost its small
kernel gain once refresh cost was included. Geometry caching required careful
floating-point equivalence checks and retained too little benefit to justify
the additional buffers and refresh path. These experiments were not retained.

Nsight Compute did not produce a hardware-counter report. A compatible version
was installed and tried on two GPUs, including a launch-statistics-only request;
each attempt failed because a driver profiling resource was unavailable. The
resource owner was not identified, and no host monitoring configuration was
changed. Consequently, achieved occupancy, warp stalls, and memory traffic are
not measured here. The saved reproduction can be rerun after the host resource
is released.

Raw profiles and logs remain in the ignored
`profiling/cloth_self_contact/20260914/` directory. Raw Nsight metadata can include
the full process environment; only allowlisted aggregates belong in committed
reports.
