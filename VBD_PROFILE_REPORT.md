# VBD solver performance profile

Date: 2026-08-12

## Summary

This branch contains behavior-preserving implementation optimizations for the
Newton VBD solver. Simulation and application parameters were not tuned: the
matched task measurements use 10 VBD iterations, one Newton substep, `dt=1/120`,
action decimation 4, seed 42, and 1,024 environments.

The original bottleneck was the combination of soft-contact generation,
particle-side contact force/Hessian accumulation, and deformable elasticity.
Those categories accounted for 76.59% of summed CUDA kernel duration in the
profiled volume step and 85.21% in cloth. Explicit proxy/coupled exchange was
below 1% and was not the primary target.

| Workload | Summed CUDA kernel speedup | End-to-end environment speedup |
|---|---:|---:|
| Cloth | 2.0819x | 1.5948x (+59.48%) |
| Frozen-topology volume | 2.2135x | 1.3392x (+33.92%) |

The CUDA values are sums of kernel execution durations inside one selected
environment-step NVTX range. They exclude CPU work, memory operations, launch
gaps, and may sum overlapping streams; they are not interchangeable with wall
time. The controlled process-level ABBA measurements are the authoritative
end-to-end results.

## Revisions and workload

- GPU: NVIDIA GeForce RTX 4090, AD102/sm_89, WDDM display mode
- Driver: 591.86; driver API reports CUDA 13.1
- CUDA toolkit: 12.9.86
- Warp: 1.16.0
- Nsight Systems: 2025.3.1
- Nsight Compute: 2025.2.1
- Newton baseline: `284af96bb563bf68536f070f508ea3561336ee73`
- IsaacLab: PR 6998 head `529a94b38cef419eec10488d5cb2e6aade3e4ec4`
- Tasks: `Isaac-Lift-Soft-Franka` and `Isaac-Lift-Cloth-Franka`
- Environments: 1,024 for the final task-level benchmarks and traces
- VBD iterations: 10
- Newton substeps: 1
- Simulation timestep: 1/120 s
- Action decimation: 4 physics advances per environment step
- Seed: 42
- Visualizer: disabled

The checked-in IsaacLab Newton task presets used for this study still specify
two substeps, so every matched run explicitly overrides the value to one.

## Controlled results

Each final ABBA suite used four `baseline, candidate, candidate, baseline`
blocks: eight fresh processes per variant, 100 warm-up steps, and 500 measured
steps per process. No measured result was discarded. Confidence intervals use
100,000 paired process-level bootstrap resamples.

| Cloth metric | Baseline | Candidate | Comparison |
|---|---:|---:|---:|
| Median environment FPS | 15,838.790 | 25,223.074 | 1.5948x |
| Paired median improvement | - | - | +59.48% |
| 95% paired confidence interval | - | - | [1.5582x, 1.6018x] |

| Frozen-volume metric | Baseline | Candidate | Comparison |
|---|---:|---:|---:|
| Median environment FPS | 24,610.636 | 32,949.169 | 1.3392x |
| Paired median improvement | - | - | +33.92% |
| 95% paired confidence interval | - | - | [1.3325x, 1.3654x] |

The volume task normally tetrahedralizes procedurally through PyTetWild, whose
topology varies between processes. The controlled comparison therefore replays
one frozen mesh with 44 particles, 75 tetrahedra, 82 boundary triangles, and
123 edges for both variants.

The current local stack did not reproduce the historical 9k/5.5k Newton
training rates that motivated the investigation. It also did not provide a
working matched PhysX control, so these results characterize and improve the
pinned Newton workload; they do not claim a new Newton-versus-PhysX ratio.

## Final CUDA component breakdown

### Cloth

| Category | Baseline | Candidate | Reduction |
|---|---:|---:|---:|
| Soft-contact generation | 16.628597 ms | 6.288744 ms | 62.18% |
| Particle contact processing | 15.536071 ms | 6.016787 ms | 61.27% |
| Elasticity | 13.665141 ms | 6.681845 ms | 51.10% |
| Duals and truncation | 1.836833 ms | 0.842108 ms | 54.15% |
| Rigid VBD excluding duals | 2.245457 ms | 2.185581 ms | 2.67% |
| Explicit proxy/coupled exchange | 0.523572 ms | 0.478767 ms | 8.56% |
| Remainder | 3.351310 ms | 3.342298 ms | 0.27% |
| **Total kernel sum** | **53.786981 ms** | **25.836130 ms** | **51.97%** |

### Frozen-topology volume

| Category | Baseline | Candidate | Reduction |
|---|---:|---:|---:|
| Soft-contact generation | 7.223824 ms | 0.828506 ms | 88.53% |
| Gather plus active adjacency | 7.762728 ms | 0.921091 ms | 88.13% |
| Elasticity | 4.593046 ms | 3.751959 ms | 18.31% |
| Duals and truncation | 1.078233 ms | 0.580039 ms | 46.20% |
| Rigid VBD excluding duals | 1.975219 ms | 2.120100 ms | -7.33% |
| Explicit proxy/coupled exchange | 0.241784 ms | 0.255232 ms | -5.56% |
| Remainder | 3.069039 ms | 3.264003 ms | -6.35% |
| **Total kernel sum** | **25.943873 ms** | **11.720930 ms** | **54.82%** |

The small increases in untouched volume categories are consistent with
between-capture clock and WDDM noise. The optimized categories account for the
material reduction. The final volume range contains 2,651 kernel launches and
has a 31.702579 ms NVTX wall span.

## Retained implementation

### Active contacts and atomics-free particle gather

After each contact refresh, a graph-safe CUDA worker grid scans the clamped
active prefix and builds linked per-particle adjacency. During each colored VBD
solve, one thread per particle traverses its active incidences, evaluates the
same contact law, accumulates force and Hessian locally, and writes each result
once. This replaces repeated capacity-wide scatter passes and their scalar
atomics. The dual update likewise uses a fixed-size grid-stride launch over the
active prefix.

The gather uses 128 threads per block, while adjacency construction and dual
updates remain at 256. A fully warmed unchanged-kernel microbenchmark measured
0.170057 ms at 128 threads versus 0.178894 ms at 256, a 5.20% gather gain.

This path is enabled only for CUDA with
`DeterministicMode.NOT_GUARANTEED`. CPU execution and deterministic modes keep
the legacy scatter implementation. Contact capacity and solver iterations are
unchanged.

### Cloth elasticity

For CUDA tiled models without tetrahedra, one 32-thread warp solves two
independent particles. Half-warp XOR reductions preserve the original 16-lane
butterfly order in each half while filling both halves of the hardware warp.
No scratch arrays are added.

### Tetrahedron-only elasticity

An eligible CUDA tiled model selects a tetrahedron-only kernel when tetrahedra
are present and all triangle Lame coefficients and edge-bending stiffnesses are
inactive. The specialization retains the same tetrahedral evaluator, adjacency
order, reduction, and solve arithmetic, but skips inactive triangle and edge
adjacency. Mixed-material models and CPU execution keep the legacy kernel.

The direct high-valence fixture and the 1,024-copy frozen workload matched the
legacy output bitwise. Focused six-color timings measured 1.29-1.38x for this
kernel. Both versions compile at 128 registers with no spills; the specialized
PTX approximately halves the static instruction count and global loads.

### Contact generation

Texture SDF evaluation now has exact value-only and gradient-only helpers, and
the search avoids full evaluations whose discarded results were previously
computed. Edge and face generation add a conservative world-AABB early reject.
Planes bypass the reject, touching boundaries are retained, and the expansion
accounts for the soft margin, particle radius, and negative shape gap. CUDA
candidate pairs are grouped stably by shape for locality while CPU order and
replay-identifier semantics remain unchanged.

## Correctness and validation

The final canonical test pass completed 83/83 tests with no failures, errors,
or skips:

- `TestVBDFullSurfaceContact`: 16/16
- `TestFullSurfaceSoftContact`: 52/52
- isolated CUDA VBD determinism: 1/1
- focused VBD solver regressions: 14/14

Focused coverage includes active-prefix boundaries and graph replay with a
changing device-side count, gather-versus-legacy equivalence, production
`SolverVBD.step()` capture/replay, deterministic fallback, tetrahedron-only
eligibility and bitwise equivalence, and reset/step coverage for cloth and
tetrahedra. The configured pre-commit suite also passed, including Ruff lint
and format, `uv-lock`, typos, and Warp array syntax.

One-environment, 20-step baseline/candidate trajectory comparisons replayed
identical actions:

- Cloth: 81 particles, 128 triangles, 208 edges, and 526 contacts each step.
  Contact keys/counts were exact and every saved numeric field had
  `max_abs = 0.0`.
- Frozen volume: 44 particles, 75 tetrahedra, 82 boundary triangles, and 123
  edges. Every saved numeric field had `max_abs = 0.0`; this short window was
  contact-free, while the cloth comparison exercises the contact path.

## Experiment sizing and reporting policy

Exploratory experiments may use fewer environments when required to fit a
single GPU or shorten iteration time. Every finding records the exact
environment count, topology, contact capacity or active count when relevant,
and whether it is a microbenchmark, correctness check, or end-to-end run.
Different environment counts are not used for throughput A/B comparisons.
Promising changes are rerun with matched baseline/candidate sizes, and final
task-level claims use 1,024 environments when that workload fits. A result that
cannot be reproduced at final scale is labeled scale-limited.

## Rejected experiments

The following prototypes were measured and removed rather than being hidden in
the retained implementation:

- a split cloth elasticity path reduced register pressure but lost to launch
  and scratch-buffer overhead;
- static shape specialization changed contact payload floating-point values;
- stable candidate compaction helped cloth faces but regressed frozen volume;
- fusing gather into cloth elasticity was bitwise exact but 1.679x slower;
- CSR and cached-state gathers were slower than the linked scalar gather;
- four-particle cloth and two-particle volume packing were neutral or noisy;
- changing VBD iteration count or contact capacity improved throughput but was
  rejected because it changes solver/application configuration.

## Nsight Compute status

Nsight Compute 2025.2.1 is compatible with this Windows AD102/sm_89 system,
CUDA toolkit 12.9, and the installed driver. The CUDA sample itself passes, but
counter collection remains blocked by `ERR_NVGPUCTRPERM`. Enable NVIDIA Control
Panel's developer setting to allow GPU performance counters for all users, or
run the capture elevated. No Nsight software update is required.

## Reproduction outline

Use the IsaacLab PR-head environment with this Newton checkout on
`PYTHONPATH`, then run the task benchmark with the Newton preset and explicit
one-substep override:

```powershell
$env:PYTHONPATH = '<path-to-this-newton-checkout>'
$uv = '<path-to-uv.exe>'

& $uv run --no-sync python scripts/benchmarks/runtime.py `
  --task Isaac-Lift-Soft-Franka `
  --num_envs 1024 --num_steps 200 --warmup_steps 50 --seed 42 `
  --visualizer none `
  presets=newton_mjwarp_vbd_proxy env.sim.physics.num_substeps=1
```

Change the task to `Isaac-Lift-Cloth-Franka` for cloth. Before each run, verify
that `import newton; print(newton.__file__)` resolves to the intended baseline
or candidate checkout. Final statistical acceptance should use fresh-process
ABBA ordering, matched frozen volume topology, and process-level rather than
per-step confidence intervals.

Raw Nsight reports, SQLite exports, trajectory arrays, process logs, and the
reproduction harnesses are archived under `profiling/`. The measurements and
hashes were audited against the unchanged source snapshot before this report
was prepared.
