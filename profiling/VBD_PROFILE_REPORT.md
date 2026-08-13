# VBD solver performance profile

Date: 2026-08-12

## Executive result

The measured bottleneck is not proxy coupling or the rigid solve. It is the
soft-contact pipeline plus deformable elasticity:

- soft-contact generation,
- particle-side contact force/Hessian accumulation, and
- elasticity

account for **76.5916%** of summed CUDA kernel duration for the volume task and
**85.2061%** for cloth in the selected traced environment step.

The strongest structural issue is work amplification from the soft-contact
capacity. Its three main particle-contact phases traverse the full capacity once
for initialization, once per iteration for dual updates, and once per particle
color per iteration for force/Hessian accumulation. Contact-list construction
and proxy-force harvesting add two more full-capacity passes per physics
advance. At 10 iterations, one environment step therefore launches 2.141
billion capacity-indexed threads for the profiled volume topology and 2.188
billion for cloth. The cloth run stored only about 396 thousand active contacts,
or 3.11% of its 12.72 million-record capacity, in a separate counter sample.

Static compilation also identifies a second concrete target. The face and edge
soft-contact generation kernels compile at 255 registers per thread and spill
heavily. They use 256-thread blocks, so each block nominally requests 65,280
registers, essentially the device's reported 65,536-register per-block limit.
Measured hardware-counter data is still required to determine the resulting
occupancy, issue efficiency, memory traffic, and divergence.

The current pinned software stack does **not** reproduce Mike's older 9k/5.5k
Newton training figures on this RTX 4090. Trainer-reported steady throughput was
17.9k steps/s for volume and 12.0k steps/s for cloth. This is not a new
Newton-vs-PhysX comparison: a local PhysX control could not run because Isaac
Sim is not installed in this isolated environment.

The optimized Newton candidate materially improves the profiled cloth task
without changing its simulation settings. A four-block ABBA suite with eight
fresh processes per variant measured a **1.5948x paired median speedup
(+59.48%)**, from a 15,838.790 FPS baseline median to a 25,223.074 FPS candidate
median. The 95% paired process-bootstrap confidence interval is
**[1.5582x, 1.6018x]**. All completed processes are included. This controlled
result supersedes the larger gain seen in a shorter exploratory run.

The final fixed-topology volume ABBA suite, which includes the retained
tetrahedron-only elasticity specialization and 128-thread gather launch,
measured a **1.3392x paired median speedup (+33.92%)**, from a 24,610.636 FPS
baseline median to a 32,949.169 FPS candidate median. Its 95% paired
process-bootstrap confidence interval is **[1.3325x, 1.3654x]**. Together, the
controlled suites establish a 59.48% cloth improvement and a 33.92% volume
improvement on this machine. The matched Nsight Systems traces measure summed
CUDA-kernel duration, not wall time: cloth falls by 51.9658% in the selected
environment step and fully integrated frozen volume falls by 54.8220%
(2.2135x less kernel time). These traces confirm the intended contact and
elasticity work reductions without implying the same factor for end-to-end
throughput.

## Revisions and configuration

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, AD102, compute capability 8.9, WDDM display mode |
| NVIDIA driver | 591.86; driver API reports CUDA 13.1 |
| CUDA toolkit used by Warp/PTXAS | 12.9.86 |
| PyTorch | 2.11.0+cu128 |
| Warp | 1.16.0 |
| Nsight Systems | 2025.3.1 |
| Nsight Compute | 2025.2.1 |
| IsaacLab | PR 6998 head `529a94b38cef419eec10488d5cb2e6aade3e4ec4` |
| Newton baseline | release-1.5 pin `284af96bb563bf68536f070f508ea3561336ee73`, version 1.5.0rc2 |
| Newton baseline checkout | `D:\Code\Graphics\newton-working-copies\codex-vbd-baseline` |
| Optimized Newton checkout | `D:\Code\Graphics\newton-working-copies\codex-vbd-profile`, same pin plus the isolated source changes described below |
| IsaacLab checkout | `D:\Code\Graphics\newton-working-copies\codex-isaaclab-vbd-profile` |
| Environments | 1024 |
| Simulation dt | 1/120 s |
| Action decimation | 4 physics advances per environment step |
| Newton substeps | 1, set explicitly with `env.sim.physics.num_substeps=1` |
| VBD iterations | 10 |
| Seed | 42 |
| Visualizer | none |

The checked-in Newton presets on this PR still specify two substeps. The
one-substep override is therefore required for the comparison Mike described.
One environment step contains four physics advances because decimation is four;
that must not be confused with Newton substeps.

The Newton import was verified to resolve to the isolated Newton checkout. The
installed package metadata also records the same Newton commit as the source
checkout used to classify kernels.

### Single-GPU experiment sizing policy

Exploratory experiments may use fewer than 1024 environments when required to
fit this machine's single GPU or to shorten iteration time. Every recorded
finding must state the exact environment count, topology dimensions, contact
capacity or active-contact count when relevant, and whether the result is a
kernel microbenchmark, correctness run, or end-to-end environment benchmark.
Results from different environment counts are not compared as throughput A/B
measurements. Promising changes are rerun with matched baseline and candidate
sizes; final task-level claims use 1024 environments when they fit. If a result
cannot be reproduced at that scale, it is labeled scale-limited rather than
generalized.

In this report, the 1-environment trajectories are correctness checks, focused
synthetic workloads are kernel experiments, and the controlled ABBA and Nsight
task results are matched 1024-environment measurements. This distinction is
preserved in the artifact names and accompanying findings.

## End-to-end measurements

The solver-focused benchmark used random actions, 50 warm-up environment steps,
and 200 measured environment steps. These numbers include environment-manager
work but exclude policy inference and learning.

| Task | Mean FPS | Mean env-step time | FPS standard deviation | GPU utilization | GPU memory |
|---|---:|---:|---:|---:|---:|
| Volume, 10 iterations | 23,954.9 | 42.7469 ms | 3,643.0 | 76.44% | 4.47 GB |
| Cloth, 10 iterations | 14,684.2 | 69.7349 ms | 1,644.3 | 83.69% | 5.22 GB |

The 10-iteration RSL-RL training smoke tests used ten trainer iterations. The
mean of trainer iterations 2 through 9 was:

| Task | Trainer-reported steps/s |
|---|---:|
| Volume | 17,921.8 |
| Cloth | 11,996.8 |

These short runs establish current local behavior; they are not statistically
controlled comparisons to Mike's machine or to PhysX. The attempted local PhysX
control stopped before task construction because this isolated environment does
not contain Isaac Sim/Omniverse Kit.

### Final controlled cloth ABBA result

The final comparison used four ABBA blocks (`baseline, candidate, candidate,
baseline`), 16 fresh processes, 100 warm-up steps and 500 measured steps per
process. Each variant therefore has eight process-level samples. The harness
enforced the same 1024-environment, 10-iteration, one-substep, seed-42
configuration, checked immutable source hashes, required at most 20% GPU
utilization before launch, and retained every completed process. Ratios pair
nearest chronological baseline and candidate processes within each block;
100,000 paired process-bootstrap resamples provide the interval.

| Cloth metric | Baseline | Candidate | Comparison |
|---|---:|---:|---:|
| Median FPS | 15,838.790 | 25,223.074 | Ratio of medians 1.5925x |
| Mean FPS | 15,870.645 | 25,111.577 | - |
| Process coefficient of variation | 0.4940% | 1.5755% | - |
| Paired median speedup | - | - | **1.5948x (+59.48%)** |
| 95% paired bootstrap CI | - | - | **[1.5582x, 1.6018x]** |

The eight observed pair ratios span 1.5333x to 1.6098x. No process was removed
as an outlier. The complete summary, machine-readable analysis, manifest, run
table, and per-process logs/results are linked below:

- [Human-readable ABBA analysis](final_abba/cloth_final/analysis.md)
- [Machine-readable ABBA analysis](final_abba/cloth_final/analysis.json)
- [Locked suite manifest](final_abba/cloth_final/manifest.json)
- [Per-process result table](final_abba/cloth_final/runs.csv)
- [Per-process artifacts](final_abba/cloth_final/runs/)

The shorter final-candidate development run measured 26,546.773 FPS versus the
original short-run baseline of 14,684.174 FPS (1.8079x). It used only 30 warm-up
and 100 measured steps in one WDDM process, so it is retained as optimization
progress evidence, not the final performance claim. Reported GPU memory was
5.32 GB versus 5.22 GB in the original short cloth baseline, approximately
0.10 GB higher in those two samples. Its result is under
[`profiling/optimization/gather/cloth_rep01`](optimization/gather/cloth_rep01/).

### Final controlled volume ABBA result

The volume suite used the same four-block ABBA protocol, eight valid fresh
processes per variant, 100 warm-up steps, 500 measured steps, and 100,000 paired
bootstrap resamples. Every process replayed the same frozen PyTetWild cache,
eliminating the topology variation that confounded the earlier volume
ablations.

| Volume metric | Baseline | Candidate | Comparison |
|---|---:|---:|---:|
| Median FPS | 24,610.636 | 32,949.169 | Ratio of medians 1.3388x |
| Mean FPS | 24,592.984 | 33,073.756 | - |
| Process coefficient of variation | 1.0685% | 1.0021% | - |
| Paired median speedup | - | - | **1.3392x (+33.92%)** |
| 95% paired bootstrap CI | - | - | **[1.3325x, 1.3654x]** |

The exact paired median ratio is `1.339155641016597`; the exact interval is
`[1.332469070388701, 1.3653807102297684]`, and the ratio of variant medians is
`1.3388182319692077`. The eight observed pair ratios span 1.3298x to 1.3660x.
All 16 valid measured results were retained. Candidate ordinal 14 has a
modified-z diagnostic flag, but the prespecified policy does not remove flagged
samples.

Baseline ordinal 8 terminated twice during native OpenUSD scene startup with
Windows access violation `0xC0000005`, before warm-up or measurement began.
Attempt 03 completed. No measured result existed to discard; the analysis still
contains the full intended eight valid results per variant. Both failed
startups and the successful retry remain in the per-process artifact directory
for auditability.

The suite manifest locks both the cache and its sidecar manifest by SHA-256.
The replayed cache contains 44 vertices and 75 tetrahedra; its output arrays and
PyTetWild parameters are recorded in the sidecar. Relevant artifacts are:

- [Human-readable volume ABBA analysis](final_abba/volume_integrated_final/analysis.md)
- [Machine-readable volume ABBA analysis](final_abba/volume_integrated_final/analysis.json)
- [Locked volume suite manifest](final_abba/volume_integrated_final/manifest.json)
- [Volume per-process result table](final_abba/volume_integrated_final/runs.csv)
- [Volume per-process artifacts](final_abba/volume_integrated_final/runs/)
- [Frozen tetrahedralization cache](validation/volume_tet_cache.npz)
- [Frozen-cache evidence manifest](validation/volume_tet_cache.npz.manifest.json)
- [First failed startup](final_abba/volume_integrated_final/runs/008_baseline/)
- [Second failed startup](final_abba/volume_integrated_final/runs/008_baseline_attempt_02/)
- [Successful attempt 03](final_abba/volume_integrated_final/runs/008_baseline_attempt_03/)

As a non-statistical post-integration screen, 50 warm-up and 200 measured steps
reported 25,327.588 mean FPS for cloth and 36,557.255 mean FPS for frozen
volume. These single WDDM-process numbers are smoke evidence only and do not
replace the ABBA results:

- [Cloth integrated screen](final_integrated_screen/cloth/benchmark_runtime_Isaac-Lift-Cloth-Franka_2026-08-12_17-07-33-687483_3cbe8a01_schema.json)
- [Frozen-volume integrated screen](final_integrated_screen/volume/benchmark_runtime_Isaac-Lift-Soft-Franka_2026-08-12_17-10-19-249494_0473700a_schema.json)

## Nsight Systems component breakdown

The first table preserves the original baseline diagnosis. Its cloth topology
matches the final candidate comparison. Its volume process produced 48
particles and is historical only; the fair volume before/after table below uses
the separately captured frozen topology with exactly 44 particles and 75
tetrahedra on both sides.

The table below uses one selected `ManagerBasedRLEnv.step` NVTX occurrence from
each trace. The denominator is the exact sum of CUDA kernel execution durations
whose execution overlaps that range. It is **not** environment-step wall time:
it excludes CUDA memory operations, CPU work, and launch gaps and can sum work
from overlapping streams.

The selected occurrences have 28.468174 ms (volume) and 53.786981 ms (cloth) of
summed kernel duration. The corresponding original NVTX wall durations are
42.356933 ms and 67.288093 ms. Nearby occurrences 3 through 8 show mean summed
kernel durations of 28.658240 ms for volume (0.955% coefficient of variation)
and 54.014259 ms for cloth (0.650% coefficient of variation), so occurrence 5
is representative of this short capture. Masked reset/reconciliation-path
kernels contribute 0.314565 ms for volume and 0.315266 ms for cloth and remain
in the remainder bucket. Those kernels launch in every captured occurrence;
because they are masked full-view launches, their presence does not mean an
environment actually reset.

| Non-overlapping kernel category | Volume ms | Volume % | Cloth ms | Cloth % |
|---|---:|---:|---:|---:|
| Soft-contact generation | 7.688321 | 27.0067% | 16.628597 | 30.9157% |
| Particle-side contact force/Hessian accumulation | 9.099804 | 31.9648% | 15.536071 | 28.8844% |
| Elasticity | 5.016094 | 17.6200% | 13.665141 | 25.4060% |
| Dual updates and truncation | 1.131896 | 3.9760% | 1.836833 | 3.4150% |
| Rigid VBD module kernels, excluding dual updates | 2.011773 | 7.0667% | 2.245457 | 4.1747% |
| Explicit proxy/coupled exchange kernels | 0.248793 | 0.8739% | 0.523572 | 0.9734% |
| Remainder | 3.271493 | 11.4918% | 3.351310 | 6.2307% |
| **Total** | **28.468174** | **100%** | **53.786981** | **100%** |

The category totals use exact nanoseconds. Displayed percentages are rounded,
so their printed sum differs from 100% by 0.0001 percentage point.

The rigid-module bucket is source based. Its three main iterative kernels take
1.764159 ms for volume and 1.853063 ms for cloth; contact-list construction,
initialization, history, and finalization account for the balance. The proxy
bucket includes explicitly attributable mapping, exchange, reconciliation, and
VBD-coupling kernels. It excludes generic copies/zeroing that cannot be assigned
from the kernel-summary report and excludes masked coupled reset-path kernels.
Even including every named coupled-module reset/reconciliation kernel would put
this bucket at only 1.9789% for volume and 1.5596% for cloth.

### Top individual kernels

| Task | Kernel | Time | Launches | Share of summed kernel duration |
|---|---|---:|---:|---:|
| Volume | `accumulate_particle_body_contact_force_and_hessian` | 9.099804 ms | 240 | 31.9648% |
| Volume | `create_soft_face_contacts` | 6.198782 ms | 4 | 21.7744% |
| Volume | `solve_elasticity_tile` | 5.016094 ms | 240 | 17.6200% |
| Cloth | `accumulate_particle_body_contact_force_and_hessian` | 15.536071 ms | 120 | 28.8844% |
| Cloth | `solve_elasticity_tile` | 13.665141 ms | 120 | 25.4060% |
| Cloth | `create_soft_face_contacts` | 13.428472 ms | 4 | 24.9660% |

The top three individual kernels account for 71.3593% of the volume total and
79.2565% of the cloth total. These are the first Nsight Compute targets once
counter access is enabled.

Each selected environment step contains four launches of the contact-generation
kernels, confirming four physics advances. With ten VBD iterations, the volume
task has six particle colors, so contact accumulation and elasticity each launch
`4 * 10 * 6 = 240` times. Cloth has three colors and therefore launches each
`4 * 10 * 3 = 120` times. Dividing the whole kernel sums by four would give
7.117044 ms and 13.446745 ms, but those are only amortized accounting values:
the environment-step range also contains observation, reward, action, and
masked reset-path work that runs once per environment step. They are not
measured physics-advance durations or wall latencies.

### Final optimized-candidate Nsight Systems breakdown

All final captures use Nsight Systems with CUDA graph tracing set to `node` and
contain 12 `ManagerBasedRLEnv.step` NVTX ranges. As in the baseline diagnosis,
the tables select zero-based occurrence 5 and sum CUDA kernel execution durations
overlapping that range. These sums are not wall time and exclude CPU work,
memory operations, and launch gaps.

#### Cloth: identical topology

The original cloth baseline and final candidate have the same 81-particle,
128-triangle, 208-edge topology. Summed kernel duration falls from
53.786981 ms to 25.836130 ms, a reduction of **27.950851 ms (51.9658%)** or
**2.0819x** less kernel time.

| Non-overlapping kernel category | Baseline ms | Baseline % | Candidate ms | Candidate % | Duration change |
|---|---:|---:|---:|---:|---:|
| Soft-contact generation | 16.628597 | 30.9157% | 6.288744 | 24.3409% | -62.1812% |
| Particle scatter vs. gather plus adjacency | 15.536071 | 28.8844% | 6.016787 | 23.2883% | -61.2721% |
| Elasticity | 13.665141 | 25.4060% | 6.681845 | 25.8624% | -51.1030% |
| Dual updates and truncation | 1.836833 | 3.4150% | 0.842108 | 3.2594% | -54.1544% |
| Rigid VBD module, excluding dual updates | 2.245457 | 4.1747% | 2.185581 | 8.4594% | -2.6665% |
| Explicit proxy/coupled exchange | 0.523572 | 0.9734% | 0.478767 | 1.8531% | -8.5576% |
| Remainder | 3.351310 | 6.2307% | 3.342298 | 12.9365% | -0.2689% |
| **Total** | **53.786981** | **100%** | **25.836130** | **100%** | **-51.9658%** |

The candidate's top individual kernels are the packed two-particle elasticity
kernel at 6.681845 ms over 120 launches (25.8624%), the per-particle gather at
5.935530 ms over 120 launches (22.9738%), and face-contact generation at
4.630646 ms over four launches (17.9231%). Together they account for 66.7593%
of candidate summed kernel duration. Adjacency construction contributes the
remaining 0.081257 ms in the combined particle gather-plus-adjacency category.

The valid capture is attempt 02; attempt 01 failed before producing NVTX step
ranges and contributes no timing data. Artifacts:

- [Cloth candidate Nsight report](final_nsys/cloth/nsys_cloth_candidate_attempt02.nsys-rep)
- [Cloth candidate SQLite export](final_nsys/cloth/nsys_cloth_candidate_attempt02.sqlite)
- [Selected cloth range kernel table](final_nsys/cloth/cloth_candidate_step5_cuda_gpu_kern_sum_nvtx=isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv.step@IsaacLab-Env-5.csv)
- [Cloth candidate runtime result](final_nsys/cloth/runtime_attempt02/benchmark_runtime_Isaac-Lift-Cloth-Franka_2026-08-12_14-53-41-898058_5fdd5a8e.json)

#### Volume: fully integrated frozen identical topology

The fair volume delta compares the frozen baseline under
[`final_nsys/volume_baseline_frozen`](final_nsys/volume_baseline_frozen/) with
the fully integrated candidate under
[`final_nsys/volume_integrated_final/attempt02`](final_nsys/volume_integrated_final/attempt02/).
Both replay the same 44-particle, 82-boundary-triangle, 123-edge,
75-tetrahedron cache. The old 48-particle volume trace above remains useful for
diagnosis but is not used for this delta.

Summed kernel duration falls from 25.943873 ms to 11.720930 ms, a reduction of
**14.222943 ms (54.8220%)** or **2.2135x** less kernel time. Relative to the
previous candidate trace, the integrated tet-only/gather changes reduce the sum
from 12.898006 ms by another **9.1260% (1.1004x)**.

| Non-overlapping kernel category | Baseline ms | Baseline % | Candidate ms | Candidate % | Duration change |
|---|---:|---:|---:|---:|---:|
| Soft-contact generation | 7.223824 | 27.8440% | 0.828506 | 7.0686% | -88.5309% |
| Particle scatter vs. gather plus adjacency | 7.762728 | 29.9212% | 0.921091 | 7.8585% | -88.1344% |
| Elasticity | 4.593046 | 17.7038% | 3.751959 | 32.0108% | -18.3122% |
| Dual updates and truncation | 1.078233 | 4.1560% | 0.580039 | 4.9487% | -46.2047% |
| Rigid VBD module, excluding dual updates | 1.975219 | 7.6134% | 2.120100 | 18.0882% | +7.3349% |
| Explicit proxy/coupled exchange | 0.241784 | 0.9320% | 0.255232 | 2.1776% | +5.5620% |
| Remainder | 3.069039 | 11.8295% | 3.264003 | 27.8476% | +6.3526% |
| **Total** | **25.943873** | **100%** | **11.720930** | **100%** | **-54.8220%** |

The tet-only kernel itself takes 3.751959 ms over 240 launches, 22.7693% less
than the prior candidate's 4.858117 ms generic elasticity bucket and 18.3122%
less than the frozen baseline. The 5.6-7.3% apparent increases in otherwise
unmodified rigid, proxy, and remainder buckets are not proven regressions from a
single WDDM trace. The contact and elasticity reductions dominate the total.

Occurrence 5 contains 2,651 kernels, all fully inside the NVTX range. Its wall
span is 31.702579 ms versus 39.229840 ms in the baseline (-19.1876%), but it is
19.4621% longer than the prior candidate's 26.537772 ms even though summed
kernel work is lower. That disagreement is direct evidence that this
single-trace WDDM wall span is noisy; the controlled ABBA suite is the
authoritative end-to-end result.

- [Frozen volume baseline Nsight report](final_nsys/volume_baseline_frozen/nsys_volume_baseline_frozen.nsys-rep)
- [Frozen volume baseline SQLite export](final_nsys/volume_baseline_frozen/nsys_volume_baseline_frozen.sqlite)
- [Selected frozen baseline range](final_nsys/volume_baseline_frozen/volume_baseline_frozen_step5_cuda_gpu_kern_sum_nvtx=isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv.step@IsaacLab-Env-5.csv)
- [Integrated volume Nsight report](final_nsys/volume_integrated_final/attempt02/nsys_volume_integrated_final_attempt02.nsys-rep)
- [Integrated volume SQLite export](final_nsys/volume_integrated_final/attempt02/nsys_volume_integrated_final_attempt02.sqlite)
- [Selected integrated volume range](final_nsys/volume_integrated_final/attempt02/volume_integrated_final_step5_cuda_gpu_kern_sum_nvtx=isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv.step@IsaacLab-Env-5.csv)
- [Integrated volume runtime result](final_nsys/volume_integrated_final/attempt02/runtime/benchmark_runtime_Isaac-Lift-Soft-Franka_2026-08-12_17-50-53-205384_90ca2dbf.json)

Attempt 02 is authoritative. Attempt 01 contains only empty directories from a
command-line syntax error; the target never started. The captured candidate
source hash is
`4b57c2e39f34bd7ee12d5431ce4374b48c785d2fcd2e24d429830d00ce256da0`
and its tracked diff SHA-256 is
`345e0a80c00b2f40d9024b4a5d4980757cf2141f0814ce18bcac11ee2c26abbc`.

## Profile-tied workload dimensions

The following dimensions were recovered from actual Nsight launch grids, rather
than copied from a separate construction process.

| Task | Particles/env | Boundary tris/env | Boundary edges/env | Particle candidates | Edge candidates | Face candidates | Total capacity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Volume | 48 | 88 | 132 | 4,177,920 | 1,892,352 | 1,261,568 | 7,331,840 |
| Cloth | 81 | 128 | 208 | 7,216,128 | 3,407,872 | 2,097,152 | 12,721,152 |

Profile-tied color launch sizes are:

- Volume: 8,447, 8,448, 8,448, 8,448, 7,681, and 7,680 particles.
- Cloth: 25,600, 40,960, and 16,384 particles.

The volume tetrahedralizer, PyTetWild, produced different valid topologies in
separate processes even with the same IsaacLab seed. The table therefore reports
only dimensions recoverable from this exact trace and deliberately omits the
volume tetrahedron count.

For iteration count `I`, color count `C`, and contact capacity `M`, contact
initialization, per-iteration dual updates, and per-color force/Hessian
accumulation perform capacity-indexed work proportional to:

```text
M * (1 + I + I*C)
```

The VBD contact-list build adds one pass, and proxy-force harvesting adds one
more. The complete steady application path observed in the trace is therefore:

```text
M * (3 + I + I*C)
```

With `I = 10`:

| Task | Main three-phase passes | All observed steady passes | All capacity threads/physics advance | All capacity threads/environment step |
|---|---:|---:|---:|---:|
| Volume, six colors | 71 | 73 | 535,224,320 | 2,140,897,280 |
| Cloth, three colors | 41 | 43 | 547,009,536 | 2,188,038,144 |

Contact generation is separate from those capacity scans. It enumerates another
29,327,360 candidate feature-shape pairs per volume environment step and
50,884,608 per cloth environment step in the profiled runs.

A separate cloth counter run sampled 395,910 to 396,109 stored contacts, with a
median of 396,030. That is 3.11% of the 12,721,152-record capacity. A separate
volume startup run sampled zero active contacts during its first eight steps,
but its nondeterministic mesh topology differs from the profiled process; it is
evidence that empty full-capacity scans can occur, not a label for the selected
trace occurrence.

## Static CUDA resource inspection

Because Nsight Compute counters are permission-blocked, the exact Warp-generated
PTX for the profiled kernels was assembled with CUDA 12.9 `ptxas -v`, including
spill and local-memory warnings. This is static compiler information, not a
replacement for measured occupancy or memory-traffic counters.

| Kernel | Version | Threads/block | Registers/thread | Stack frame/thread | Spill stores/thread | Spill loads/thread |
|---|---|---:|---:|---:|---:|---:|
| `create_soft_face_contacts` | Baseline | 256 | 255 | 544 B | 576 B | 572 B |
| `create_soft_face_contacts` | Candidate | 256 | 255 | 536 B | 568 B | 564 B |
| `create_soft_edge_contacts` | Baseline | 256 | 255 | 288 B | 284 B | 284 B |
| `create_soft_edge_contacts` | Candidate | 256 | 255 | 280 B | 280 B | 280 B |
| `create_soft_contacts` (vertex) | Baseline | 256 | 113 | 256 B | 0 B | 0 B |
| Legacy contact scatter | Baseline/candidate fallback | 256 | 70 | 0 B | 0 B | 0 B |
| Active-contact adjacency builder | Candidate | 256 | 20 | 0 B | 0 B | 0 B |
| Per-particle contact gather | Candidate | 128 | 89 | 16 B | 0 B | 0 B |
| Active-prefix dual update | Candidate | 256 | 40 | 0 B | 0 B | 0 B |
| Legacy `solve_elasticity_tile` | Baseline/mixed-material path | 16 | 128 | 0 B | 0 B | 0 B |
| Tetrahedron-only tiled elasticity | Candidate eligible volume path | 16 | 128 | 0 B | 0 B | 0 B |
| Two-particle cloth elasticity | Candidate cloth path | 32 | 128 | 0 B | 0 B | 0 B |
| Scalar `solve_elasticity` | Measured separately | topology dependent | 128 | 0 B | 0 B | 0 B |

The candidate shortened some live ranges, but the face and edge kernels remain
at 255 registers and still spill heavily. Their small static stack/spill
reductions do not resolve the original register-pressure hypothesis. Nsight
Compute must confirm achieved occupancy, eligible warps, issue stalls,
L1/L2/DRAM traffic, branch efficiency, and source-correlated hot instructions
before a more invasive generation-kernel rewrite.

The legacy tiled elasticity kernel launches 16-thread blocks, half a hardware
warp, uses one barrier, and reserves 64 B of shared memory. The retained cloth
path packs two independent 16-lane particles into a 32-thread warp while
preserving each half-warp's reduction order. It remains at 128 registers, but
uses no barriers and 128 B of shared memory per 32-thread block. The
active-contact adjacency builder and gather do not spill. These are static
compiler facts, not achieved-occupancy measurements.

The tetrahedron-only kernel has the same 128-register, zero-spill, one-barrier
profile as the legacy tiled kernel, but its constant-argument footprint falls
from 2,064 B to 1,616 B. Approximate PTX instruction count falls from 3,430 to
1,768, global-load instructions from 140 to 70, and branch instructions from 82
to 48 because inactive triangle and edge adjacency paths are absent. ABBA and
Nsight manifests lock the source snapshots used for dynamic measurements;
Warp module hashes are configuration-specific and are not used as cross-run
identifiers here.

## Controlled ablations

Every ablation intentionally changed one solver setting, used the same
1024-environment, one-substep, seed-42 random-action benchmark, and was restored
afterward. The cloth topology is deterministic. The volume mesh was rebuilt in
each process by nondeterministic PyTetWild, so the volume deltas are confounded
by small topology changes and must be treated as directional rather than clean
single-variable comparisons. All temporary ablation settings were restored.
The retained optimized source changes described below are separate from those
configuration experiments.

| Configuration | Volume FPS | Delta | Cloth FPS | Delta |
|---|---:|---:|---:|---:|
| Baseline: 10 iterations, tiled elasticity, full capacity | 23,954.9 | - | 14,684.2 | - |
| 2 VBD iterations | 34,159.6 | +42.6% | 23,129.8 | +57.5% |
| Scalar elasticity | 19,933.8 | -16.8% | 16,193.2 | +10.3% |
| Contact-record cap 1,048,576 | 27,347.5 | +14.2% | 17,055.6 | +16.2% |

Interpretation:

- Reducing iterations is helpful but cannot approach a fivefold speedup because
  contact generation and other fixed work remain. No policy quality or physical
  fidelity evaluation was performed for the two-iteration setting.
- Tiled elasticity is clearly beneficial for the tetrahedral volume workload,
  while scalar elasticity is faster for this cloth topology. A single default
  is leaving cloth performance on the table.
- The smaller contact-record capacity produced no overflow warning during these
  200-step runs and improved both tasks. It is not a production-safe cap: a
  full training run needs per-world high-water marks, explicit overflow counts,
  and correctness checks. Candidate replay arrays remain full-sized, so this
  ablation does not fix collision-generation enumeration.

The large deltas are useful directional evidence, but these are single WDDM
runs on an active display GPU, not confidence intervals from repeated isolated
trials. The benchmark JSON schema records the preset and one-substep override,
but not the temporary iteration, scalar-solve, or contact-cap setting. The
ablation directory names and `profiling/ablation_manifest.json` preserve that
mapping explicitly.

## Implemented optimization candidate

The retained candidate changes internal Newton implementation only. The
1024-environment task definitions, one substep, 10 VBD iterations, dt,
decimation, contact capacity, material parameters, and tiled-solve setting are
unchanged.

### Active-contact iteration and atomics-free gather

After each contact refresh, a CUDA graph-safe worker grid scans only the
clamped active prefix (`min(raw contact count, capacity)`) and builds a linked
per-particle adjacency. The list supports vertex, edge, and face contacts,
including repeated particle corners. It is built once per refresh rather than
once per color or VBD iteration.

During a colored solve, one thread per particle traverses that particle's
active incidences, calls the same contact evaluators, accumulates force and
Hessian locally, and writes each result once. This replaces repeated
capacity-wide scatter launches and their scalar atomics. The dual update also
uses a fixed-size grid-stride worker launch over the active prefix. Buffer
growth retains the existing CUDA-capture guard.

The gather alone launches at 128 threads per block; adjacency construction and
dual updates remain at 256. A prewarmed 16-cell microbenchmark of unchanged
gather code measured 0.170057 ms at 128 threads versus 0.178894 ms at 256,
**1.0520x (+5.20%)**, with bitwise-identical outputs.

This path is enabled only for CUDA with
`DeterministicMode.NOT_GUARANTEED`. CPU execution and deterministic modes keep
the legacy scatter implementation. Contact capacity, evaluator formulas, and
solver iteration count are unchanged. Linked-list insertion uses CUDA atomics,
so the non-guaranteed mode retains the allowed run-to-run atomic-order
variability instead of silently changing the deterministic contract.

### Full-warp cloth elasticity

For CUDA tiled models with no tetrahedra, a 32-thread block now solves two
independent particles. Native half-warp XOR reductions preserve the legacy
16-lane butterfly tree inside each half warp, so the second particle fills the
otherwise idle half of the hardware warp without changing the reduction order.
No scratch arrays are added.

### Tetrahedron-only volume elasticity

At solver construction, an eligible CUDA tiled model selects a 16-thread
tetrahedron-only elasticity kernel when tetrahedra are present and all triangle
Lame coefficients and edge-bending stiffnesses are inactive. The specialization
uses the same tetrahedral evaluator, adjacency order, tile reduction, and solve
arithmetic as the legacy kernel, but does not traverse inactive triangle and
edge adjacency. Mixed or active triangle/edge material models retain the legacy
path. CPU execution is unchanged.

Newton documents model data as static/non-time-varying. Eligibility is therefore
computed once, consistently with other solver construction choices; a caller
that changes triangle or edge stiffness must rebuild `SolverVBD`. A direct
52-particle high-valence fixture with 17 adjacent tetrahedra, shuffled indices,
inactive particles, and `uint32` adjacency matched the legacy output
bitwise. At 1,024 frozen-topology replicas, focused six-color CUDA-event timings
were 0.304323 -> 0.220613 ms (**1.3794x**) and, on a rerun, 0.232430 ->
0.180065 ms (**1.2908x**). The final end-to-end ABBA result above is the
authoritative workload-level effect.

### Contact-generation work reduction

The texture-SDF path now has exact value-only and gradient-only helpers, and
the soft-contact search avoids full evaluations whose values are discarded.
Its golden-section pass computes only the gamma quantity needed for the search,
then performs the full evaluation at the selected point. Arithmetic and texture
quantization of retained outputs are preserved.

Edge and face generation apply a conservative world-AABB early reject before
the narrow phase. The expansion includes soft margin, maximum particle radius,
and negative shape-gap contribution; planes bypass the test and strict
comparisons retain touching boundaries. CUDA candidate pairs are also grouped
stably by shape so shape data has better locality. CPU candidate order remains
unchanged, and contact capacity plus replay identifiers retain their existing
semantics.

## Rejected or reverted experiments

- Splitting triangle elasticity into two launches reduced the triangle kernel
  to 72 registers and was numerically exact, but launch and scratch-buffer cost
  reduced combined cloth throughput to about 18.4k FPS. It was reverted.
- An earlier two-particle implementation built from a 2-D Warp tile compiled at
  167 registers and reduced cloth throughput to about 10.2k FPS. It was
  replaced by the retained half-warp native reductions.
- Static box/sphere evaluator specialization preserved contact identities and
  counts but changed payload values by as much as approximately `2.0558e-4`.
  It was rejected because the requested optimization must preserve behavior.
- Reducing VBD iterations or contact capacity improved the ablation benchmark,
  but both alter application/solver configuration and are not part of the
  candidate.
- Dynamic AABB compaction was not added. The retained in-kernel conservative
  reject had clear evidence and a smaller correctness surface; further
  compaction should wait for hardware-counter and updated trace evidence.
- A later stable edge/face candidate-compaction prototype remained bitwise
  exact. Compacting only faces projected a 1.239x reset and 1.336x contact-rich
  speedup for the combined cloth edge-plus-face pass, but the same compact face
  operation regressed the exact frozen-volume pass to 0.674x. A task-specific
  dispatch heuristic was rejected.
- Fusing the linked gather into packed cloth elasticity was bitwise exact and
  passed changing-count graph replay, but measured 0.560879 ms versus
  0.334007 ms for the separate gather-plus-elasticity sequence: **1.679x
  slower**. During contact traversal only two lanes per warp do useful work,
  overwhelming the saved launches and global round trip.
- A CSR half-warp gather and a cached-particle-state gather both lost to the
  retained linked one-thread-per-particle gather; the cached version was 4.35%
  slower. Neither entered production dispatch.
- Four-particle cloth packing at a 64-thread block was neutral to slower than
  the retained two-particle warp. A two-particle volume elasticity prototype
  moved from about +1.9% to about -2% on repetition, and a rigid-solve
  four-thread-to-one-thread experiment was neutral. All were rejected as noise
  or regressions.

## Correctness and targeted validation

The final one-environment cloth comparison replays the same 20-step action tape
through the fully integrated candidate and reports `pass: true`. Its topology
is exactly 81 particles, 128 triangles, and 208 edges. It contains 526 contacts
per measured step and 10,520 saved contact keys in total, so it exercises the
active-contact path throughout. Raw counts, offsets, duplicate counts, and
contact keys match exactly. Particle, body, and joint positions and velocities,
observations, rewards, and all saved contact payload fields have
`max_abs = 0.0`; actions, terminations, and truncations are exact. See the
[integrated cloth comparison](validation/cloth_1env_20step_integrated_comparison.json)
and its [candidate trajectory](validation/cloth_1env_20step_candidate_integrated/).

The frozen volume-topology tooling records and replays 44 particles, 82
boundary triangles, 123 edges, and 75 tetrahedra with topology hash
`5c9478730b4460fc6db2582824293091645bd28e2d03fe414cc4abc2d2a97bd1`.
Its [record/replay comparison](validation/volume_1env_1step_record_replay_comparison.json)
passes exactly. That artifact validates the topology-freezing mechanism itself;
both saved sides use the clean baseline checkout, so it is not presented as a
baseline-versus-optimized volume-equivalence result.

The fully integrated frozen-volume baseline-versus-candidate trajectory extends
that check to 20 environment steps. Its
[comparison artifact](validation/volume_1env_20step_integrated_frozen_comparison.json)
reports `pass: true`: topology is exactly 44 particles, 82 boundary triangles,
123 edges, and 75 tetrahedra; every saved numeric field has `max_abs = 0.0`, and
the action, termination, truncation, contact-count, and contact-key fields are
exact. This particular window remains contact-free, so it establishes exact
volume integration and tetrahedron-only elasticity equivalence but does not
independently exercise the optimized active-contact path. The contact-rich cloth
comparison supplies that complementary evidence.

After all integrations and cleanup, the final canonical regression run passed
**83/83** with zero failures, errors, or skips:

- `TestVBDFullSurfaceContact`: 16/16 in 122.714 s,
- `TestFullSurfaceSoftContact`: 52/52 in 59.960 s,
- isolated CUDA VBD determinism: 1/1 in 18.355 s, and
- focused VBD solver batch: 14/14 in 58.325 s.

Aggregate unittest time was 259.354 s and wall time was 274.397 s. The process
imported Newton from the isolated canonical candidate, emitted only two expected
`joint_target_q` deprecation warnings, and left HEAD plus the tracked source/diff
hashes unchanged. Focused coverage includes tetrahedron-only eligibility and
direct high-valence equivalence, gather legacy equivalence and changing-count
graph replay, production `SolverVBD.step()` graph capture/replay with zero and
active contact counts, deterministic fallback, soft reset then step for cloth
and tetrahedra, and isolated VBD determinism. Pre-commit passed every configured
check, including Ruff lint/format, `uv-lock`, typos, and Warp-array syntax.

The final independent static review found no actionable blocker. It confirmed
that tet-only eligibility matches the existing material guards, mixed/active
triangle or edge models retain the legacy kernel, 128 threads applies only to
the gather, CPU and deterministic fallbacks remain intact, and no dead formal or
prototype residue remains. `git diff --check`, `py_compile`, and Ruff passed;
the retained Newton diff is nine files, 2,154 insertions and 193 deletions.

## Baseline diagnosis and optimization order

1. **Compact active contacts before iterative VBD work.** Build a compact active
   list or per-particle/per-color contact adjacency after collision generation.
   Launch dual and force/Hessian work over active records rather than
   `soft_contact_max` for every color and iteration. Preserve deterministic
   ordering where required and support CUDA graph capture with device-side
   counts or bounded per-world buffers.

2. **Add capacity/overflow telemetry first.** Record total and per-world active
   contact high-water marks, stored/contact-capacity ratio, and overflow counts
   through full training. This converts the promising 1,048,576-cap result into
   a defensible sizing policy and makes regressions observable.

3. **Specialize face and edge contact generation.** Split code paths or shorten
   live ranges so shape-specific state is not simultaneously live. Reassemble
   the exact PTX and then use Nsight Compute to verify lower registers, spills,
   local-memory traffic, and stall reasons. The face+edge kernels alone account
   for 24.90% of volume and 28.82% of cloth summed kernel duration.

4. **Continue topology-specific elasticity work.** The retained candidate now
   packs two cloth particles per warp and removes inactive triangle/edge work
   from eligible tetrahedral models. Hardware counters should guide any further
   split of membrane, bending, and volume paths; four-particle cloth and
   two-particle volume prototypes were not repeatable wins.

5. **Reduce candidate generation.** Active-contact compaction does not remove
   the 27-31% contact-generation floor. Add stronger world-local spatial pruning
   or narrower collidable proxy-shape lists so vertex/edge/face SDF kernels do
   not scan every compatible feature-shape candidate.

6. **Do not optimize proxy exchange first.** Explicit proxy/coupled exchange
   kernels are below 1% of summed kernel duration in both selected ranges.
   Rigid VBD module work is also secondary at 4-7%.

The retained candidate implements the core of items 1, 4, and 5 and removes
discarded evaluator work from item 3. It does not introduce a smaller capacity
or a capacity-sizing policy, and the face/edge kernels remain register-bound
static candidates. Item 2 and hardware-counter-guided generation work are the
main unfinished follow-ups.

Kernel shares prioritize investigation; they are not end-to-end speedup
predictions. A PhysX trace with the same workload is still needed before
claiming that any one Newton hotspot explains the complete backend gap.

## Nsight Compute compatibility and blocker

The installed versions are compatible:

- Nsight Compute 2025.2.1 recognizes Windows and the AD102/sm_89 GPU.
- Nsight Systems 2025.3.1 supports the installed CUDA 12.9 toolkit.
- Warp initializes and CUDA 12.9 `deviceQuery` and `vectorAdd` both pass.

The final Nsight Compute access probe was repeated after the optimized
candidate and still connects to `vectorAdd`, which completes successfully, but
NCU exits with:

```text
ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

No software update is needed. Enable counter access in NVIDIA Control Panel:

```text
Desktop / Developer -> Manage GPU Performance Counters
                     -> Allow access to all users
```

Alternatively, run the capture from an elevated process. After changing the
setting, restart the profiling process and first rerun this low-cost check:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1\target\windows-desktop-win7-x64\ncu.exe' `
  --set basic --launch-count 1 `
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\demo_suite\vectorAdd.exe'
```

Once permission works, capture one representative launch each of:

1. the candidate per-particle contact gather, compared with the legacy
   `accumulate_particle_body_contact_force_and_hessian` scatter
2. `create_soft_face_contacts`
3. the two-particle cloth elasticity kernel, compared with legacy
   `solve_elasticity_tile`
4. `create_soft_edge_contacts`
5. the active-contact adjacency builder and active-prefix dual update

Collect Speed of Light, occupancy, memory-workload, scheduler/warp-state, and
source-counter sections. Compare face/edge results against any split-kernel
prototype, and use the gather/elasticity comparison to verify that the measured
gain comes from reduced work and full-warp utilization rather than a hidden
memory or scheduler bottleneck.

## Reproduction

PowerShell environment and exploratory candidate command:

```powershell
$env:PYTHONPATH = 'D:\Code\Graphics\newton-working-copies\codex-vbd-profile'
$uv = 'C:\Users\ankac\AppData\Local\anaconda3\Scripts\uv.exe'
$lab = 'D:\Code\Graphics\newton-working-copies\codex-isaaclab-vbd-profile'

Set-Location $lab
& $uv run --no-sync python scripts/benchmarks/runtime.py `
  --task Isaac-Lift-Soft-Franka `
  --num_envs 1024 --num_steps 200 --warmup_steps 50 --seed 42 `
  --visualizer none --output_path profiling/volume_10it_1substep `
  presets=newton_mjwarp_vbd_proxy env.sim.physics.num_substeps=1
```

Change the task to `Isaac-Lift-Cloth-Franka` and the output directory for cloth.
Before every profiling run, verify the Newton source actually imported:

```powershell
& $uv run --no-sync python -c `
  "import newton; print(newton.__file__)"
```

The path must resolve under
`D:\Code\Graphics\newton-working-copies\codex-vbd-profile`.

The controlled suites are reproduced by the checked-in profiling harness,
which selects and hashes the clean baseline and optimized candidate checkouts:

```powershell
$python = 'D:\Code\Graphics\newton-working-copies\codex-isaaclab-vbd-profile\.venv\Scripts\python.exe'
& $python profiling\run_vbd_abba.py `
  --workload cloth `
  --output-root profiling\final_abba\cloth_final `
  --execute
```

Replace `cloth` with `volume` and use
`profiling\final_abba\volume_integrated_final` as the output root to reproduce
the final volume suite. That mode automatically replays and verifies the frozen
tetrahedralization cache and its evidence manifest.

The exact Nsight Systems reports, SQLite exports, filtered CSVs, benchmark JSON,
and workload-counter JSON files are stored in the sibling `profiling` task
directories. `nsys_volume_discovery.nsys-rep` and
`nsys_cloth_discovery.nsys-rep` are the original diagnostic traces. The final
candidate and frozen-volume comparison reports are linked directly from the
final breakdown above.

## Repository integrity

The official Newton checkout was not modified. It remains at
`15da174d3946279436f94c5bc650eeca423bff5a` on
`ankac/water-tight-rigid-soft-sdf-demos` with no tracked or staged diff caused by
this work. Untracked files already present in that official checkout were left
alone.

The optimized Newton checkout is intentionally dirty on branch
`codex/vbd-profile`: its tracked diff contains the retained implementation,
tests, and changelog entry. Its untracked `.warp-cache` is generated compiler
output. The clean comparison checkout remains detached at
`284af96bb563bf68536f070f508ea3561336ee73`.

The isolated IsaacLab checkout's four apparent tracked modifications are only
LF/CRLF worktree normalization; their canonical hashes equal their HEAD blobs.
The profiling harnesses and artifacts are untracked by design. Both task
configs are restored to ten VBD iterations, and every reported comparison
overrides Newton to one substep explicitly.
