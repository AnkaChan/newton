# VBD optimization code changes

Date: 2026-08-13

## Purpose and scope

This report explains the implementation changes between the pinned Newton
baseline `284af96bb563bf68536f070f508ea3561336ee73` and optimization commit
`ac9d62310d32537fdc6c7a677105ad223bbd9776`.

The objective was to improve VBD performance without tuning the application or
changing the solver's mathematical configuration. The final task measurements
therefore keep all of the following fixed:

- 10 VBD iterations,
- one Newton substep,
- simulation timestep `1/120 s`,
- action decimation 4,
- the original contact capacity,
- the original material parameters and particle coloring, and
- 1,024 environments for the final workload benchmarks.

The allowed behavioral exception is the ordering variability already inherent
in nondeterministic CUDA atomics. CPU execution and deterministic CUDA modes
retain the legacy contact-scatter path.

The detailed measurements and raw artifacts are described in
[VBD_PROFILE_REPORT.md](VBD_PROFILE_REPORT.md). This document focuses on what
changed in the code, why each change is valid, and how it can improve
performance.

## Executive summary

The baseline spent most of its CUDA kernel time in three areas:

1. enumerating soft feature/shape contact candidates,
2. repeatedly scanning the entire allocated soft-contact capacity during VBD
   iterations, and
3. solving deformable elasticity with inefficient or unnecessary work for the
   task topology.

The retained implementation attacks each area directly:

| Change group | Main performance mechanism | Measured component result |
|---|---|---:|
| Contact generation | Reject impossible feature/shape pairs before expensive SDF searches; avoid computing unused SDF outputs; improve shape-data locality | 2.64x cloth and 8.72x volume for the combined generation bucket |
| Particle contact processing | Build active per-particle adjacency once, gather only active incidences, and remove repeated capacity scans and scalar atomics | 2.58x cloth and 8.43x volume for the particle-processing bucket |
| Dual update | Traverse the clamped active prefix with a fixed graph-safe worker grid | 2.18x cloth and 1.86x volume for the combined dual/truncation bucket |
| Cloth elasticity | Use both halves of a hardware warp while preserving each 16-lane reduction tree | 2.05x for the cloth elasticity bucket |
| Volume elasticity | Compile a tetrahedron-only kernel when triangle and edge stiffness are inactive | 1.295x versus the previous generic candidate elasticity kernel |
| Gather launch geometry | Use 128 rather than 256 threads for the high-register gather kernel | 1.052x in the isolated gather microbenchmark |

These figures are not additive. The categories interact, and the controlled
end-to-end result includes CPU work, launch gaps, observations, rewards, rigid
solving, and other environment work:

| Workload | Summed CUDA kernel speedup | Controlled end-to-end speedup |
|---|---:|---:|
| Cloth | 2.0819x | 1.5948x (+59.48%) |
| Frozen-topology volume | 2.2135x | 1.3392x (+33.92%) |

## Why the baseline was expensive

### Capacity-wide iterative work

`soft_contact_max` is a storage capacity, not the number of active contacts.
For the profiled 1,024-environment tasks it was 7.33 million records for the
original volume trace and 12.72 million for cloth. A separate cloth sample
contained about 396 thousand stored contacts, only 3.11% of capacity.

The legacy particle-contact path launched over the full capacity for:

- initialization once per collision refresh,
- dual updates once per VBD iteration,
- force/Hessian accumulation once per particle color per iteration,
- contact-list construction, and
- proxy-force harvesting.

For contact capacity `M`, iteration count `I`, and color count `C`, the observed
steady path performs capacity-indexed work proportional to:

```text
M * (3 + I + I*C)
```

At 10 iterations this corresponds to 2.141 billion capacity-indexed threads per
environment step in the profiled six-color volume topology and 2.188 billion
for the three-color cloth topology. Most of those threads only checked a count
or stale tail record and returned.

### Expensive full-surface contact search

Edge and face contact generation enumerated every world-compatible
feature/shape candidate. The profiled environment step examined about 29.3
million volume and 50.9 million cloth candidates across its four physics
advances.

The edge/face kernels also compiled at the architectural register ceiling of
255 registers per thread and spilled to local memory. A thread rejected by a
cheap geometric condition still paid the occupancy cost of that large kernel.
Inside accepted search paths, helper functions produced SDF values and
gradients even when the caller consumed only one of those outputs.

### Elasticity utilization and inactive work

The legacy tiled solver uses 16 threads per particle. On CUDA that occupies
only half of a 32-lane hardware warp. The cloth task has no tetrahedra, so two
independent particles can safely occupy the two half-warps.

The volume task has active tetrahedral material but zero triangle membrane and
edge-bending stiffness. The generic elasticity kernel still carries the
triangle and edge arguments, branches, adjacency traversal, and generated code
needed for mixed models.

## File-by-file change map

| File | Responsibility of the change |
|---|---|
| [`sdf_texture.py`](../newton/_src/geometry/sdf_texture.py) | Add exact value-only and gradient-only texture-SDF sampling helpers. |
| [`soft_contacts_sdf.py`](../newton/_src/geometry/soft_contacts_sdf.py) | Split SDF outputs by use, remove discarded line-search evaluations, add conservative AABB rejection, and defer expensive accepted-contact work. |
| [`collide.py`](../newton/_src/sim/collide.py) | Stably group CUDA candidate pairs by shape and pass current rigid-shape AABBs into full-surface generation. |
| [`particle_vbd_kernels.py`](../newton/_src/solvers/vbd/particle_vbd_kernels.py) | Add active-contact adjacency/gather kernels and specialized cloth and tetrahedral elasticity kernels. |
| [`rigid_vbd_kernels.py`](../newton/_src/solvers/vbd/rigid_vbd_kernels.py) | Convert body-particle dual updates to an active-prefix grid-stride traversal. |
| [`solver_vbd.py`](../newton/_src/solvers/vbd/solver_vbd.py) | Allocate and refresh adjacency, choose graph-safe launch dimensions, dispatch optimized kernels, and retain safe fallbacks. |
| [`test_collision_pipeline.py`](../newton/tests/test_collision_pipeline.py) | Add exact SDF-split, AABB boundary/multiset, world-pair ordering, and graph-replay coverage. |
| [`test_solver_vbd.py`](../newton/tests/test_solver_vbd.py) | Add active-prefix, gather, graph replay, deterministic fallback, and specialized elasticity equivalence tests. |
| [`CHANGELOG.md`](../CHANGELOG.md) | Record the user-visible performance improvement. |

All new runtime symbols are internal; no public Newton API was added or
removed.

## 1. Output-specific texture-SDF sampling

### Code change

[`sdf_texture.py`](../newton/_src/geometry/sdf_texture.py) adds:

- `texture_sample_sdf_value_only()`, and
- `texture_sample_sdf_grad_only()`.

Both helpers reuse the same cell lookup, corner reads, sparse-subgrid
dequantization, interpolation weights, scale handling, and extrapolation
conventions as `texture_sample_sdf_grad()`. They omit only calculations for an
output the caller does not consume.

[`soft_contacts_sdf.py`](../newton/_src/geometry/soft_contacts_sdf.py) adds the
corresponding shape-dispatch helpers:

- `eval_shape_sdf_lower()` for the conservative scalar distance used by
  culling and one-dimensional search, and
- `eval_shape_sdf_grad()` for the gradient used by the face optimizer's
  Frank-Wolfe direction.

The existing full evaluator remains the source of final accepted contact
payloads.

### Why it is valid

The omitted return values had no caller use and therefore no effect on the
selected parameter, final contact, or stored payload. The texture value-only
path intentionally follows the full gradient sampler's per-corner
dequantization order rather than substituting a superficially similar sampler
with different floating-point rounding.

Analytic and texture tests compare the split helpers against the original full
evaluator bit-for-bit, including nonuniform and mirrored texture scale.

### Why it is faster

SDF search calls its evaluator many times per candidate. Avoiding gradient
arithmetic during scalar search shortens live ranges and removes vector math;
avoiding value interpolation during gradient-only Frank-Wolfe steps removes
work in the other direction. This matters most in the face kernel, where the
search is nested.

Static compilation showed only a small stack/spill reduction because the
overall dynamic-shape kernel remains at 255 registers. The principal gain is
less executed arithmetic and fewer simultaneously live intermediate values,
not a new occupancy tier.

## 2. Remove discarded line-search evaluations

### Code change

[`soft_contacts_sdf.py`](../newton/_src/geometry/soft_contacts_sdf.py) factors
the golden-section search into `optimize_edge_sdf_gamma()`. It returns the final
interpolation parameter directly. `optimize_edge_sdf()` uses that parameter and
performs one full SDF evaluation only for its externally returned final result.
The face optimizer calls the gamma-only helper because it previously discarded
the line search's position, distance, and gradient.

Within `optimize_face_sdf()`, Frank-Wolfe iterations use the gradient-only
helper, while the final selected point still uses the original full evaluator.

### Why it is valid

The returned gamma is the same `0.5 * (lo + hi)` produced by the original fixed
iteration search. Search bounds, comparison order, interpolation expressions,
iteration counts, and the final contact evaluation are preserved.

### Why it is faster

The face optimizer no longer performs a discarded full SDF query at the end of
every internal edge line search. It also avoids producing unused value/gradient
tuples throughout the search. This removes repeated work from every surviving
face candidate without changing the number of search iterations.

## 3. Conservative feature-AABB rejection

### Code change

[`soft_contacts_sdf.py`](../newton/_src/geometry/soft_contacts_sdf.py) adds an
early world-space AABB test for edge and face candidates. The feature bound is
expanded by:

```text
soft margin + maximum feature particle radius + max(0, -shape gap)
```

and compared against the rigid shape AABB already maintained by the collision
pipeline. Strict `<` and `>` comparisons retain touching bounds. Infinite
planes bypass the test. Supplying no AABB arrays disables the optimization for
isolated/reference callers.

[`collide.py`](../newton/_src/sim/collide.py) passes the current narrow-phase
`shape_aabb_lower` and `shape_aabb_upper` arrays to the edge/face launcher.

### Why it is valid

The expansion is conservative relative to the contact threshold and the shape
gap already included in the rigid AABB. A candidate is rejected only when the
expanded feature bound is strictly disjoint. Planes are excluded because their
finite stored bounds cannot represent an infinite contact surface.

Boundary tests cover exact touching, separation, negative gap restoration,
plane bypass, and empty-array fallback. Random mixed-shape tests compare the
complete order-independent contact multiset, including floating-point payload
bits, with the AABB path enabled and disabled.

### Why it is faster

The check consists of feature min/max calculation and six scalar comparisons.
Rejected candidates avoid shape-frame inversion, transformations, SDF
dispatch, iterative edge/face optimization, and contact emission. It is
especially effective in replicated volume scenes where millions of candidates
are geometrically far from their shapes.

The code also defers shape-frame and accepted-contact-only work until after the
rejection, shortening expensive live ranges for the common miss path.

## 4. Stable CUDA pair grouping by shape

### Code change

[`collide.py`](../newton/_src/sim/collide.py) extends
`_world_compatible_pairs()` with `group_by_shape`. CUDA particle, edge, and face
candidate arrays are stably sorted by shape index after the existing world and
capability filtering. CPU ordering remains unchanged.

The original candidate identifier remains associated with each pair, so
backward replay and contact-record mapping retain their semantics.

### Why it is valid

The candidate set is unchanged; only its processing order changes on the
nondeterministic CUDA path. Stable sorting preserves relative order within each
shape group. Tests compare the resulting candidate sets against brute-force
world compatibility, assert CUDA shape grouping, and exercise graph replay.

### Why it is faster

Neighboring threads are more likely to read the same shape type, transform,
scale, SDF descriptor, and AABB. This improves cache locality and branch
coherence in the dynamic shape dispatcher. Grouping alone does not reduce the
number of candidates, so its benefit is included in the combined contact
generation measurement rather than claimed independently.

## 5. Active-contact adjacency

### Code change

[`particle_vbd_kernels.py`](../newton/_src/solvers/vbd/particle_vbd_kernels.py)
adds `build_particle_body_contact_adjacency_active()`. It reads the device-side
contact count, clamps it to capacity, and traverses the compact active prefix
with a fixed worker grid and device-side stride.

For every active record it inserts one adjacency node per valid particle
corner:

- particle contact: one node,
- edge contact: two nodes, and
- face contact: three nodes.

The representation is:

- `head[particle]`: first linked node for a particle,
- `next[3 * contact_capacity]`: next node, and
- an encoded node index that identifies contact record and corner.

Repeated corners remain repeated incidences, matching the legacy scatter
semantics. Raw contact counts greater than capacity are always clamped before
array access.

[`solver_vbd.py`](../newton/_src/solvers/vbd/solver_vbd.py) allocates these
buffers only for the CUDA nondeterministic gather path, clears `head`, and
rebuilds the list once when contacts refresh. Buffer resizing follows the
existing capture-safety rules.

### Why it is valid

Newton's contact arrays already store valid records as a compact prefix. The
builder changes the index used to visit those records; it does not compact,
filter, or alter them. Particle records remain self-described by
`(particle, -1, -1)`, and edge/face records retain their original corner and
barycentric data.

Tests cover counts at zero, worker-boundary values, exact capacity, and raw
overflow; particle/edge/face mixtures; repeated corners; changing device-side
counts under one captured graph; and full `SolverVBD.step()` dispatch.

### Why it is faster

The list is built once per contact refresh, while force/Hessian evaluation runs
once per color per VBD iteration. Paying one active-prefix pass avoids dozens
of capacity-wide passes. The builder compiles at 20 registers with no spills,
so the indexing stage is cheap relative to the contact evaluator.

The tradeoff is additional CUDA memory: one `int32` head per particle and up to
three `int32` links per contact-capacity slot. That is approximately 153 MB for
the 12.72-million-capacity cloth workload and 88 MB for the 7.33-million
profiled volume capacity. These buffers are not allocated for CPU or
deterministic fallback execution.

## 6. Atomics-free per-particle contact gather

### Code change

`gather_particle_body_contact_force_and_hessian()` in
[`particle_vbd_kernels.py`](../newton/_src/solvers/vbd/particle_vbd_kernels.py)
launches one thread per particle in the current color. The thread walks that
particle's active adjacency, invokes the same particle or edge/face contact
evaluator, accumulates its force and Hessian locally, and writes one force plus
one Hessian result.

It writes a zero result for an empty list at the beginning of the color's
force/Hessian phase, preventing stale values before later spring and
self-contact kernels add their contributions.

The production gather launch uses 128 threads per block. The adjacency builder
and dual kernels remain at 256.

### Why it is valid

Each legacy contact scatter contributed the same evaluator result to every
valid particle corner. The gather visits the identical incidence set but sums
by destination particle. Floating-point arrival order may differ from a CUDA
atomic scatter, which is within the explicitly nondeterministic mode's existing
contract.

Deterministic modes retain the original one-record-per-thread scatter because
the linked-list insertion order is atomic and because a dynamic loop would
require a different deterministic atomic-record budget. CPU also retains the
legacy path.

Direct legacy/gather tests compare force and Hessian results within the
predeclared atomic-order tolerance, and full 20-step cloth trajectories match
exactly for the tested action tape. The deterministic regression confirms that
the legacy route remains selected.

### Why it is faster

The baseline launched over `soft_contact_max` once per color and iteration,
even when only a small active prefix existed. It also issued atomic additions
for three force components and nine Hessian components per participating
corner. The gather:

- launches over colored particles rather than contact capacity,
- follows only active incidences,
- accumulates in registers,
- removes inter-thread force/Hessian atomics, and
- writes 12 scalar outputs once per particle.

The retained gather compiles at 89 registers, a 16-byte stack, and no spills.
Reducing its block size from 256 to 128 improved a fully warmed isolated
workload by 5.20%, consistent with better residency for a high-register kernel.

The combined adjacency-plus-gather category fell from 15.536071 ms to
6.016787 ms for cloth and from 7.762728 ms to 0.921091 ms for frozen volume.

## 7. Active-prefix body-particle dual update

### Code change

`update_duals_body_particle_contacts()` in
[`rigid_vbd_kernels.py`](../newton/_src/solvers/vbd/rigid_vbd_kernels.py) now
accepts a contact stride. Each worker reads the clamped active count and handles
indices `tid`, `tid + stride`, and so on until it reaches the count.

[`solver_vbd.py`](../newton/_src/solvers/vbd/solver_vbd.py) computes a fixed
graph-safe worker dimension bounded by capacity and available CUDA parallelism.
The device-side count can change across graph replays without host readback or
recapture.

### Why it is valid

Each dual record is updated independently, so changing which physical thread
processes a record does not change write ordering or introduce atomics. The
kernel preserves the particle-versus-edge/face interpretation, maximum feature
radius, contact evaluation, and penalty update. Raw overflow is clamped to
capacity.

Boundary and captured-graph tests change the device count through zero, worker
boundaries, capacity, and overflow while checking exact active-row results and
an unchanged stale tail.

### Why it is faster

The baseline launched one thread for every capacity slot on every VBD
iteration. The new fixed worker grid exits after the active prefix. It removes
millions of tail checks while remaining graph-capturable. The kernel compiles at
40 registers with no spills.

The measured dual/truncation category fell by 54.15% for cloth and 46.20% for
volume. Because the category also contains truncation kernels, these percentages
are evidence for the integrated phase rather than an isolated dual-only timing.

## 8. Two-particle cloth elasticity

### Code change

[`particle_vbd_kernels.py`](../newton/_src/solvers/vbd/particle_vbd_kernels.py)
adds a native half-warp XOR reduction and
`solve_elasticity_tile_two_particles()`. One 32-thread CUDA block is divided
into two independent 16-lane groups, each solving one cloth particle.

Each half-warp performs the same lane reductions in the same `8, 4, 2, 1`
order as the legacy 16-thread tiled kernel. The solver selects this path only
for tiled CUDA models with no tetrahedra.

### Why it is valid

Particles remain independent within one color. No particle changes its
adjacency traversal or Gauss-Seidel color boundary. The reduction tree and
lane-zero arithmetic order are preserved inside each half-warp. Odd color sizes
use an inactive second slot safely.

Tests include odd particle counts, a high-valence fan with more than one
16-element adjacency batch, shuffled indices, inactive particles, exact
`uint32` comparison, and captured execution.

### Why it is faster

The legacy block occupies only half of a hardware warp. Packing two particles
fills the full warp without adding a scratch buffer or another launch. The
specialized kernel remains at 128 registers and zero spills but removes the
legacy block barrier. Cloth elasticity kernel time fell from 13.665141 ms to
6.681845 ms, a 51.10% reduction or 2.05x speedup.

## 9. Tetrahedron-only elasticity

### Code change

[`solver_vbd.py`](../newton/_src/solvers/vbd/solver_vbd.py) adds
`_is_tet_only_elasticity_model()`. At solver construction it selects
`solve_elasticity_tile_tet_only()` when:

- CUDA tiled elasticity is enabled,
- at least one tetrahedron exists,
- every triangle has inactive membrane coefficients, and
- every edge has inactive bending stiffness.

The specialized kernel in
[`particle_vbd_kernels.py`](../newton/_src/solvers/vbd/particle_vbd_kernels.py)
retains the original tetrahedral evaluator, adjacency order, 16-lane reduction,
Hessian solve, and displacement update. It omits triangle/edge inputs and code.
Mixed or active triangle/edge models use the legacy kernel.

### Why it is valid

The removed branches would contribute exactly zero under the same material
guards used by the generic kernel. Newton documents `Model` data as static
solver input, so eligibility is computed once like other construction-time
choices. A caller that mutates triangle or edge stiffness must rebuild
`SolverVBD`.

Tests cover active and inactive material combinations, damping-only cases, a
52-particle odd-size fixture, 17 tetrahedra adjacent to one vertex, shuffled
indices, inactive particles, and exact bitwise output comparison. The frozen
1,024-replica workload also matched bitwise.

### Why it is faster

For the frozen volume mesh, the generic kernel traversed 738 inactive
triangle/edge adjacency records per environment alongside 300 active
tetrahedral incidences. Removing those paths:

- reduces approximate PTX instructions from 3,430 to 1,768,
- reduces static global-load instructions from 140 to 70,
- reduces branches from 82 to 48, and
- reduces constant arguments from 2,064 B to 1,616 B.

Both kernels remain at 128 registers, zero spills, and one barrier, so the gain
comes from less executed and fetched work rather than higher register-limited
occupancy.

Focused six-color timings measured 1.29-1.38x. In the integrated Nsight trace,
elasticity fell from the previous candidate's 4.858117 ms to 3.751959 ms, a
22.77% reduction or 1.295x speedup.

## 10. Solver dispatch, lifetime, and CUDA graph safety

### Code change

[`solver_vbd.py`](../newton/_src/solvers/vbd/solver_vbd.py) ties the new kernels
together:

- `_SOFT_CONTACT_BLOCK_DIM = 256` and a bounded worker-count policy define
  fixed active-prefix launch grids.
- `_PARTICLE_CONTACT_GATHER_BLOCK_DIM = 128` applies only to the gather.
- `_use_particle_contact_gather` is true only for CUDA and
  `DeterministicMode.NOT_GUARANTEED`.
- Contact adjacency buffers are allocated or resized with the existing
  capture-allocation guard.
- Adjacency is rebuilt on the first use and whenever contacts refresh, then
  reused through the VBD color/iteration loop.
- Cloth, tetrahedron-only, and generic elasticity paths are selected from
  static model properties.
- Active-prefix dual launch dimensions are used at both body-particle dual
  update call sites.

### Why it is valid

CUDA launch dimensions remain host-static during graph capture. Dynamic contact
counts are read only on the device, so replay can change from zero contacts to
active contacts without synchronizing or freezing an old count. Capacity
growth retains the existing rule that allocation is forbidden during capture.

The optimized and fallback paths share the same contact/material buffers and
evaluators. The dispatch changes scheduling and removes provably inactive work;
it does not alter timestep, iteration count, convergence policy, contact
capacity, or material values.

### Why it is faster

The solver amortizes the adjacency build across every particle color and VBD
iteration, selects launch geometries appropriate to each kernel's register
footprint, and prevents host synchronization from becoming a new bottleneck.
Static dispatch also keeps the specialized elasticity kernels free of runtime
branches that would preserve the generic register/code footprint.

## 11. Tests and correctness gates

### Collision tests

[`test_collision_pipeline.py`](../newton/tests/test_collision_pipeline.py) adds
coverage for:

- exact analytic value/gradient split results,
- exact texture-SDF split results under nonuniform and mirrored scale,
- AABB touching, separation, negative gap, plane bypass, and disabled fallback,
- exact contact multisets with AABB rejection enabled/disabled,
- multi-world candidate compatibility and CUDA shape grouping, and
- CUDA graph capture/replay.

### VBD tests

[`test_solver_vbd.py`](../newton/tests/test_solver_vbd.py) adds coverage for:

- gather versus legacy scatter on mixed particle/edge/face records,
- repeated edge/face corners,
- zero, boundary, capacity, and overflow counts,
- changing contact counts inside a captured graph,
- real `SolverVBD.step()` dispatch and adjacency refresh,
- deterministic fallback,
- active-prefix dual exactness and stale-tail preservation,
- two-particle cloth elasticity equivalence,
- tetrahedron-only eligibility and bitwise solve equivalence, and
- cloth/tetrahedron reset and step behavior.

The final canonical suite passed 83/83 targeted tests with zero failures,
errors, or skips. One-environment 20-step trajectory comparisons reported
`max_abs = 0.0` for every saved numeric field in contact-rich cloth and frozen
volume. The volume window was contact-free, while cloth exercised 526 contacts
per step.

## 12. Changelog and documentation

[`CHANGELOG.md`](../CHANGELOG.md) records that `SolverVBD` soft-contact
processing and cloth elasticity are accelerated without changing settings or
contact capacity.

No runtime behavior comes from the Markdown reports or profiling archive. They
preserve the configuration, source hashes, benchmark statistics, selected
Nsight ranges, validation outputs, and rejected experiments needed to audit the
performance claims.

## Performance attribution and limitations

The following distinctions are important when interpreting the numbers:

- The generation result combines SDF output splitting, discarded-evaluation
  removal, AABB rejection, deferred accepted-contact work, and CUDA shape
  grouping. Those subchanges were not independently timed in the final task
  trace.
- The particle-processing result compares legacy capacity-wide scatter against
  the integrated adjacency-plus-gather path. It includes adjacency construction
  overhead.
- The dual/truncation result is a source-based category, not a dual-only
  microbenchmark.
- CUDA kernel sums are not wall time. They exclude CPU work, CUDA memory
  operations, launch gaps, and may sum overlapping streams.
- End-to-end ABBA results are the authoritative task-level gains.
- Nsight Compute hardware counters could not be collected because this machine
  still reports `ERR_NVGPUCTRPERM`. Static PTXAS resources and Nsight Systems
  timings were used instead.

## Rejected alternatives

Several plausible changes were measured and deliberately excluded:

- splitting cloth elasticity into multiple launches reduced one kernel's
  registers but lost to launch and scratch traffic;
- shape-specialized contact kernels changed floating-point payloads;
- candidate compaction helped cloth faces but regressed frozen volume;
- fusing gather with cloth elasticity was bitwise exact but 1.679x slower;
- CSR and cached-state gathers lost to the linked scalar gather;
- four-particle cloth and two-particle volume packing were neutral or unstable;
- reducing iterations or contact capacity improved speed but changed solver or
  application configuration.

Rejecting these variants is part of the implementation rationale: the retained
branch contains only changes with a repeatable benefit and an acceptable
correctness surface.

## Result

The code changes reduce the work performed by VBD rather than lowering its
quality settings. The two profiled workloads show more than 2x reduction in
summed CUDA kernel duration, while controlled environment throughput improves
by 59.48% for cloth and 33.92% for frozen volume. Deterministic and CPU fallback
behavior is retained, graph replay remains dynamic-count safe, and the final
tests and exact trajectory comparisons pass.
