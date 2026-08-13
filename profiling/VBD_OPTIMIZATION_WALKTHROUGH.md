# VBD soft-contact optimization walkthrough — `codex/vbd-profile`

This walkthrough explains the code changes of the `codex/vbd-profile` branch
(commit `ac9d6231` on top of the release-1.5 pin `284af96b`) and — using the
independent validation run of 2026-08-13 — attributes exactly **where the
acceleration comes from**. All speedups below were re-measured on this
machine (L40, warp 1.16.0) on the reports' own IsaacLab workloads:
`Isaac-Lift-Cloth-Franka` and `Isaac-Lift-Soft-Franka`, 1024 envs, 10 VBD
iterations, one substep. Validation details:
`$AI_LOGS/Newton/tasks/vbd-profile-validate/2026-08-13-report.md`.

## The big picture

The baseline solver had three structural cost sinks:

- **Capacity-wide iterative scans.** `soft_contact_max` is a *storage
  capacity* (12.7M records for 1024 cloth envs), while only ~396k contacts
  (3%) are active. The legacy path launched one thread per capacity slot for
  contact initialization, for every dual update (once per iteration), and for
  force/Hessian accumulation (once per particle color per iteration) — about
  2.2 billion mostly-idle threads per env step.
- **Brute-force contact generation.** The edge/face SDF kernels enumerate
  every world-compatible (feature, shape) candidate — 51M per cloth env step —
  in a 255-register spilling kernel, with candidates ordered feature-major so
  adjacent threads touch different shapes.
- **Half-empty warps in elasticity.** The tiled elasticity kernel uses a
  16-thread block per particle: half of every 32-lane hardware warp idle. The
  volume task additionally paid for triangle/edge material branches that are
  compiled in but always inactive (its cloth-style stiffnesses are zero).

The branch attacks each sink without changing solver settings, capacity, or
results (trajectories match the baseline bitwise). Measured end-to-end:
cloth **1.664x**, volume **1.317x**; summed CUDA kernel time: cloth
**1.94x**, volume **2.24x**.

**Where each change lives in one physics advance.** An env step is 4
physics advances (action decimation). Each advance runs collision detection
once, then 10 VBD iterations over the particle colors:

```python
# one physics advance (x4 per env step)
CollisionPipeline.collide(state, contacts)
#   candidate pairs -> edge/face SDF kernels -> contact records        <- §2 (candidate ordering), §3 (AABB early-out), §8 (SDF query internals)
#   results land in contacts.soft_contact_* (fixed 12.7M-slot buffer)
SolverVBD.step(...)
#   contact refresh, once per advance                                  <- §1a (build per-particle contact lists)
#   for iteration in range(10):
#       update contact penalty stiffness (AVBD dual update)            <- §6 (walk active prefix, not the whole buffer)
#       for color in particle colors (3 cloth / 6 volume):
#           sum rigid-contact forces onto this color's particles       <- §1b (gather via the lists), §7 (its block size)
#           elasticity solve for this color's particles                <- §4 (cloth: 2 particles/warp), §5 (volume: tet-only kernel)
```
(the numbers refer to the sections below)

Where the kernel-time acceleration actually comes from, measured by
reverting each change individually. All numbers are summed CUDA kernel time
**per env step** (one step = 4 physics advances x 10 VBD iterations). The
baseline step costs **50.98 ms** of kernel time on cloth and **27.49 ms** on
volume; the branch brings that to **26.32 ms** (-24.66, -48.4%) and
**12.30 ms** (-15.19, -55.3%). Each row shows the step-time saving that
disappears when that one change is reverted, and that saving as a share of
the baseline step:

| Change | Cloth saving (of 50.98 ms/step) | Volume saving (of 27.49 ms/step) | Verdict vs. report |
|---|---|---|---|
| §2 Detection: candidate list sorted by shape | 9.85 ms = 19.3% of step | 0.37 ms = 1.3% | Confirmed — undersold by the report |
| §1 Contact-force summation via per-particle lists | 9.11 ms = 17.9% of step | 7.17 ms = 26.1% | Confirmed (largest volume win) |
| §4 Cloth elasticity: two particles per warp | 6.44 ms = 12.6% of step | n/a | Confirmed (2.01x vs claim 2.05x) |
| §3 Detection: bounding-box early-out | ~0 | 2.34 ms = 8.5% | Confirmed for volume; inert on cloth |
| §5 Tet elasticity: kernel without cloth code | n/a | 1.11 ms = 4.0% | Confirmed (1.30x vs claim 1.295x) |
| §6 Contact penalty update over real contacts | 0.97 ms = 1.9% | 0.57 ms = 2.1% | Confirmed (small) |
| §7 Contact-force kernel at 128-thread blocks | 0.47 ms = 0.9% | small | Confirmed (~1.08x of its bucket) |
| §8 SDF value-only/gradient-only queries | ~0 (0.008 ms) | ~0 (0.010 ms) | **No measurable effect** |
| **Total step reduction (branch vs base)** | **24.66 ms = 48.4%** | **15.19 ms = 55.3%** | |

Two reading notes. First, one-at-a-time savings sum to more than the total
(cloth rows add to 26.8 ms vs 24.7 actual) because the changes interact —
see finding R6. Second, these are *kernel-time* shares: the measured
wall-clock env step also contains CPU work and launch gaps that the branch
does not shrink (baseline cloth wall step is 57.6 ms vs 50.98 ms of kernel
time), which is why 48-55% kernel savings become 1.66x / 1.32x end-to-end
rather than 2x.

## 1. Contact-force summation in the VBD loop: per-particle contact lists replace full-buffer scans

**Where this lives / what this part does.** After collision detection has
filled `contacts.soft_contact_*` with "this cloth particle/edge/triangle
touches that rigid shape" records, every VBD iteration needs - for each
particle of the color being solved - the total contact force and Hessian
acting on it. That total feeds the per-particle 3x3 solve in the elasticity
kernel. Old kernel: `accumulate_particle_body_contact_force_and_hessian`;
new kernels: `build_particle_body_contact_adjacency_active` (runs once per
advance) + `gather_particle_body_contact_force_and_hessian` (runs per
color/iteration), both in `newton/_src/solvers/vbd/particle_vbd_kernels.py`
and dispatched from `_solve_particle_iteration`
(`newton/_src/solvers/vbd/solver_vbd.py:2705`). This was the single biggest
cost in the whole step on both workloads.

**Old behavior.** The contact buffer has 12.7 million slots (worst-case
capacity for 1024 cloth envs), but only ~400k hold real contacts. Every
solver iteration, for every particle color, the solver launched *one thread
per slot* — 12.7M threads — and each thread checked "is my slot a real
contact? does it belong to a particle of this color?" Almost all said no and
exited. The few real ones computed the contact force and pushed it onto
their particle with atomic adds (12 atomics per contact corner). For cloth
that is 12.7M threads x 3 colors x 10 iterations x 4 advances per env step,
mostly doing nothing.

**New behavior.** Once per physics advance, right after collision detection,
one small kernel walks just the ~400k *real* contacts and builds a lookup:
for each particle, a linked list of "here are the contacts that touch you."
Then each iteration/color launches only one thread per particle; the thread
follows its own list, sums its contact forces in registers, and writes the
result once. No scanning of empty slots, no atomic adds. The list is built
4 times per env step and used 120 times — that amortization is the payoff
(the build costs 21 us; each scan it replaces cost ~3.3 ms).

```python
@wp.kernel
def build_particle_body_contact_adjacency_active(
    body_particle_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_count: wp.array[int],
    body_particle_contact_max: int,
    contact_stride: int,
    particle_contact_head: wp.array[int],
    particle_contact_next: wp.array[int],
):
    """Build linked per-particle incidence lists over the compact active contact prefix."""
    contact_index = wp.tid()
    count = min(body_particle_contact_max, body_particle_contact_count[0])

    while contact_index < count:
        corners = body_particle_contact_indices[contact_index]

        # A -1 in slot one identifies the single-particle record. Match the scatter path's
        # self-description rule exactly instead of interpreting any malformed trailing slot.
        corner_count = 1
        if corners[1] >= 0:
            corner_count = 3

        for corner in range(corner_count):
            particle_index = corners[corner]
            if particle_index >= 0:
                node = 3 * contact_index + corner
                previous = wp.atomic_exch(particle_contact_head, particle_index, node)
                particle_contact_next[node] = previous

        contact_index += contact_stride
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:2452-2482`)

Subtle points: the launch dimension is a *fixed* worker count and the kernel
grid-strides to the device-side `count` — so a captured CUDA graph stays
valid when the contact count changes between replays (no host readback). The
node encoding `3 * contact + corner` lets one `int32` identify both the
record and which of its up-to-3 particle corners this incidence is. Repeated
corners stay repeated, matching legacy scatter semantics exactly.

The consumer runs once per color per iteration — one thread per *colored
particle*, walking only that particle's incidences and writing its force and
Hessian exactly once (no atomics on the output):

```python
    """Gather all body-contact contributions for one colored particle without output atomics."""
    particle_index = particle_ids_in_color[wp.tid()]
    force = wp.vec3(0.0)
    hessian = wp.mat33(0.0)
    node = particle_contact_head[particle_index]

    while node >= 0:
        contact_index = node // 3
        corner = node - 3 * contact_index
        corners = body_particle_contact_indices[contact_index]
        contact_ke = body_particle_contact_penalty_k[contact_index]
        contact_kd = body_particle_contact_material_kd[contact_index]
        contact_mu = body_particle_contact_material_mu[contact_index]
...
    # This is the first force/Hessian phase for the color. Writing even an empty list clears any
    # stale value and leaves later spring/self-contact kernels free to add their contributions.
    particle_forces[particle_index] = force
    particle_hessians[particle_index] = hessian
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:2516-2528, 2586-2589`)

The contact evaluators (`_eval_body_particle_contact`, `_eval_soft_ef_contact`)
are the *same functions* the legacy scatter calls — only the iteration order
and the accumulation destination change. That is why this path is restricted
to `DeterministicMode.NOT_GUARANTEED` on CUDA
(`newton/_src/solvers/vbd/solver_vbd.py:460`): floating-point arrival order
differs from atomic scatter, which that mode already permits. CPU and
deterministic runs keep the legacy scatter.

Measured: the particle-contact bucket drops 805.0 -> 344.0 ms on cloth
(2.59x; report claimed 2.58x) and 481.5 -> 51.5 ms on volume (9.36x; report
8.43x). The adjacency build itself is noise: 20.9 us per launch on cloth,
~0.08 ms per env step, 0.3% of the step (20 registers, no spills).

## 2. Contact detection, candidate ordering: sort the (feature, shape) test list by shape

**Where this lives / what this part does.** Contact detection tests every
world-compatible (cloth feature, rigid shape) pair - "does edge 4711 touch
box 12?" - with one GPU thread per pair. The list of pairs to test is built
once on the host at pipeline construction (`_world_compatible_pairs` in
`newton/_src/sim/collide.py:509`) and consumed every advance by the
detection kernels. This change only reorders that list.

**Old behavior.** The candidate list for contact generation ("test this
cloth edge against this rigid shape") was ordered feature-first. So 32
neighboring GPU threads each worked on a *different* shape — different
transforms, different SDF data, different shape-type branches — thrashing
the cache and diverging at every dispatch.

**New behavior.** The same candidate list is stably sorted by shape before
upload (a two-line host-side change, CUDA only). Neighboring threads now
work on the *same* shape: same data in cache, same branch taken. Nothing
about the candidates themselves changes — only their order.

```python
    def _pairs(f_idx: np.ndarray, s_idx: np.ndarray):
        # ``shape_ok`` (optional, indexed by shape) drops pairs whose shape cannot participate -- e.g.
        # full-surface edge/face excludes shapes without a usable SDF, which fall back to per-particle.
        if shape_ok is not None and len(s_idx):
            keep = shape_ok[s_idx.astype(np.intp)]
            f_idx, s_idx = f_idx[keep], s_idx[keep]
        if group_by_shape and len(s_idx):
            order = np.argsort(s_idx, kind="stable")
            f_idx, s_idx = f_idx[order], s_idx[order]
        stacked = np.column_stack((f_idx, s_idx)).astype(np.int32) if len(f_idx) else np.empty((0, 2), np.int32)
        return wp.array(stacked, dtype=wp.vec2i, device=device)
```
(`newton/_src/sim/collide.py:509-520`)

The consumers are `create_soft_edge_contacts` and
`create_soft_face_contacts` — kernels at the 255-register ceiling that
spill to local memory, where every avoided memory stall and divergent
branch matters most.

Measured by reverting just this: cloth generation 366.8 -> 958.0 ms — i.e.
grouping alone is a **2.61x** on the cloth generation bucket, essentially
the *entire* cloth generation gain. On volume it is 1.48x (46.5 -> 68.7 ms)
because the AABB reject (§3) already removes most work there. The report
made no isolated claim for this change ("included in the combined
generation measurement") — that undersells it.

## 3. Contact detection, early-out: bounding-box test before the SDF search

**Where this lives / what this part does.** Inside the detection kernels
(`create_soft_edge_contacts`, `create_soft_face_contacts` in
`newton/_src/geometry/soft_contacts_sdf.py`), each thread decides whether
its (feature, shape) pair actually makes contact by running an iterative
signed-distance-field search. This change adds a cheap screen at the top of
the thread, before any of that machinery starts.

**Old behavior.** Every edge/face candidate pair — even a cloth edge 30
meters away from the shape in a different replicated env — went through the
full treatment: invert the shape's frame, transform the feature into shape
space, and run the iterative SDF search, inside a kernel already at the
255-register ceiling.

**New behavior.** Before any of that, a cheap box-overlap test: does the
feature's bounding box (grown by the contact margin, particle radius, and
shape gap) overlap the shape's world AABB that the narrow phase already
maintains? If not — six comparisons and the thread exits. Far-away
candidates, which are almost all of them in a 1024-env replicated scene,
never touch the expensive path.

```python
def _soft_feature_aabb_misses_analytic_shape(
    shape_index: wp.int32,
    shape_gap: wp.array[float],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    feature_lower: wp.vec3,
    feature_upper: wp.vec3,
    margin: float,
    radius: float,
) -> bool:
    """Analytic non-plane variant of :func:`_soft_feature_aabb_misses_shape`."""
    if shape_aabb_lower.shape[0] == 0:
        return False

    gap = shape_gap[shape_index]
    expansion = margin + radius + wp.max(0.0, -gap)
    expansion_vec = wp.vec3(expansion, expansion, expansion)
    feature_lower = feature_lower - expansion_vec
    feature_upper = feature_upper + expansion_vec
    rigid_lower = shape_aabb_lower[shape_index]
    rigid_upper = shape_aabb_upper[shape_index]
    return (
        feature_upper[0] < rigid_lower[0]
        or feature_upper[1] < rigid_lower[1]
        or feature_upper[2] < rigid_lower[2]
        or feature_lower[0] > rigid_upper[0]
        or feature_lower[1] > rigid_upper[1]
        or feature_lower[2] > rigid_upper[2]
    )
```
(`newton/_src/geometry/soft_contacts_sdf.py:375-403`)

Strict `<`/`>` keep touching boundaries; planes bypass the test (their
stored AABB cannot represent an infinite surface, see
`newton/_src/geometry/soft_contacts_sdf.py:313`); an empty AABB array
disables the optimization for isolated kernel tests. The face kernel runs
the test before computing shape frames (`newton/_src/geometry/soft_contacts_sdf.py:518`),
so a rejected candidate pays only the six comparisons.

Measured: volume generation 46.5 -> 186.8 ms without it — a **4.02x** on
that bucket by itself, the dominant volume generation win (replicated
far-apart envs mean almost all candidates are misses). On cloth it changes
nothing (366.8 vs 359.6 ms, within noise): the cloth hangs directly over
its shapes, so features are rarely rejectable. Both halves match the
report's own narrative ("especially effective in replicated volume
scenes").

## 4. Cloth elasticity solve: two particles per 32-thread warp

**Where this lives / what this part does.** The elasticity solve is the
core of a VBD iteration: for each particle of the current color, sum the
stretching/bending forces of all triangles and hinge edges around it, add
the contact force from §1, and solve a 3x3 system for the particle's new
position. Kernel: `solve_elasticity_tile` -> new
`solve_elasticity_tile_two_particles`
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:2837`), selected for
tet-free models at `newton/_src/solvers/vbd/solver_vbd.py:2796`.

**Old behavior.** Each particle's elastic solve used a 16-thread team
(`TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 16`), but the GPU schedules threads
in fixed groups of 32 (a warp) — so half of every warp sat idle for the
whole kernel.

**New behavior.** For models with no tetrahedra, two independent particles
share one 32-thread warp, each on its own 16-thread half. The previously
idle half of the hardware now does useful work; nothing else changes:

```python
    """Solve two triangle-mesh particles per warp using independent 16-lane reductions."""
    tid = wp.tid()
    particle_slot = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_slot_valid = particle_slot < particle_ids_in_color.shape[0]
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:2859-2863`)

The trick that keeps results bitwise identical is the reduction. A generic
`wp.tile_reduce` over 32 lanes would change summation order; instead a
native half-warp XOR butterfly reproduces the legacy 16-lane tree exactly,
independently in each half warp:

```python
@wp.func_native(
    """
    #if defined(__CUDA_ARCH__)
    float r = v;
    r += __shfl_xor_sync(0xffffffffu, r, 8, 16);
    r += __shfl_xor_sync(0xffffffffu, r, 4, 16);
    r += __shfl_xor_sync(0xffffffffu, r, 2, 16);
    r += __shfl_xor_sync(0xffffffffu, r, 1, 16);
    return r;
    #else
    return v;
    #endif
    """
)
def _warp_half_reduce_sum(v: wp.float32) -> wp.float32: ...
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:57-71`)

```python
    # The legacy 16-thread tile reduction uses the same 8,4,2,1 butterfly tree
    # for lane zero. Scalar shuffles retain that arithmetic order while allowing
    # the two half-warps to solve independent particles in one full warp.
    f_total = wp.vec3(
        _warp_half_reduce_sum(f[0]),
        _warp_half_reduce_sum(f[1]),
        _warp_half_reduce_sum(f[2]),
    )
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:2929-2936`)

The `width=16` argument of `__shfl_xor_sync` partitions the warp into two
independent 16-lane groups, and offsets 8,4,2,1 replay the same tree the
16-thread tile used — same addends, same order, same rounding. It also
removes the block barrier the legacy kernel needed.

Measured: cloth elasticity 768.7 -> 381.8 ms, **2.01x** (report claimed
2.05x) — the second-largest cloth win. Doubling occupancy of a
compute-dense 128-register kernel halves its time almost exactly.

## 5. Soft-body (tet) elasticity solve: a kernel without the unused cloth code

**Where this lives / what this part does.** Same elasticity solve as §4,
but for the volumetric soft-body task: each particle sums the forces of its
surrounding tetrahedra. The generic kernel also contains the cloth code
paths (triangle membrane + edge bending) because one kernel served every
model type. New kernel: `solve_elasticity_tile_tet_only`
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:2757`), selected at
construction when the model has tets and zero cloth-style stiffness.

**Old behavior.** The generic elasticity kernel always carried the code for
cloth-style triangle stretching and edge bending, and walked those adjacency
lists per particle — even for the soft-body task, whose triangle/edge
stiffnesses are all zero. It computed guaranteed zeros every iteration.

**New behavior.** At solver construction, if the model has tetrahedra and no
active triangle/edge material, a slimmed kernel containing only the
tetrahedral code is selected. Half the instructions and loads, identical
answers. Eligibility is decided once, from static model data:

```python
def _is_tet_only_elasticity_model(model: Model) -> bool:
    """Return whether the model's active element materials are tetrahedral only."""
    if model.tet_count == 0:
        return False

    if model.tri_count > 0:
        tri_materials = model.tri_materials.numpy()
        if np.any((tri_materials[:, 0] > 0.0) | (tri_materials[:, 1] > 0.0)):
            return False

    if model.edge_count > 0:
        edge_bending_properties = model.edge_bending_properties.numpy()
        if np.any(edge_bending_properties[:, 0] > 0.0):
            return False

    return True
```
(`newton/_src/solvers/vbd/solver_vbd.py:102-117`)

`solve_elasticity_tile_tet_only`
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:2757`) is the legacy
kernel minus the triangle/edge paths: same tetrahedral evaluator, adjacency
order, 16-lane reduction, and Hessian solve, so outputs are bitwise
identical. Halving the code and loads (PTX 3430 -> 1768 instructions per the
report) speeds up the *executed* work even though registers stay at 128.
The removed branches would contribute exactly zero under the same material
guards the generic kernel applies per element — a caller who later turns
membrane stiffness on must rebuild `SolverVBD`, which the docstring states.

Measured: volume elasticity 219.1 -> 285.6 ms when reverted — **1.30x**,
matching the claimed 1.295x almost exactly.

## 6. Contact penalty-stiffness update: walk only the real contacts

**Where this lives / what this part does.** The solver's contact model
(AVBD) ramps each contact's penalty stiffness up over the iterations while
it stays penetrating - one small independent update per contact record,
once per VBD iteration. Kernel: `update_duals_body_particle_contacts`
(`newton/_src/solvers/vbd/rigid_vbd_kernels.py:4874`), launched from
`solver_vbd.py` at both call sites (internal-rigid and external-rigid
paths).

**Old behavior.** Same capacity problem as §1, in a smaller consumer: to
update the penalty stiffness of the ~400k real contacts, the solver launched
one thread per 12.7M-slot capacity, every iteration; almost every thread
checked the count and exited.

**New behavior.** A fixed-size crew of ~70k threads walks only the
real-contact prefix, each handling a few contacts in a loop. The fixed
launch size is what keeps captured CUDA graphs valid — the count can change
between replays, only how far the threads walk changes:

```python
    idx = wp.tid()
    count = min(body_particle_contact_max, body_particle_contact_count[0])
    while idx < count:
        corners = soft_contact_indices[idx]
        shape_idx = body_particle_contact_shape[idx]
```
(`newton/_src/solvers/vbd/rigid_vbd_kernels.py:4900-4904`)

```python
    def _active_soft_contact_launch_dim(self, soft_contact_max: int) -> int:
        """Return a bounded launch size for grid-stride traversal of active soft contacts."""
        if soft_contact_max <= 0 or self.model.particle_count <= 0:
            return 0

        parallelism = self.model.particle_count
        if self.device.is_cuda:
            parallelism = max(
                parallelism,
                self.device.sm_count * _SOFT_CONTACT_BLOCKS_PER_SM * _SOFT_CONTACT_BLOCK_DIM,
            )
        return min(soft_contact_max, parallelism)
```
(`newton/_src/solvers/vbd/solver_vbd.py:1210-1221`)

Each record's update is independent, so re-assigning records to threads
changes nothing numerically. Measured bucket (duals + truncation): cloth
100.6 -> 39.6 ms (2.54x), volume 63.6 -> 29.3 ms (2.17x). Real, but only
~2-4% of total kernel time — the same structural fix as §1 applied to a much
smaller consumer.

## 7. Kernel launch sizes: 128-thread blocks for the contact-force kernel

**Where this lives / what this part does.** Not an algorithm change - just
how many threads per block each of the new kernels from §1/§6 is launched
with (`newton/_src/solvers/vbd/solver_vbd.py:97-99`), and how the solver
picks which elasticity kernel to run.

**Old behavior.** All soft-contact kernels ran at Warp's default 256 threads
per block.

**New behavior.** The per-particle gather — a register-hungry kernel (89
registers/thread) — runs at 128 threads per block so more blocks fit on
each SM; the lighter adjacency build and dual updates stay at 256.

The per-particle gather runs at 128 threads per block
(`newton/_src/solvers/vbd/solver_vbd.py:99`, used at
`newton/_src/solvers/vbd/solver_vbd.py:2709`); adjacency build and dual
updates stay at 256 (`newton/_src/solvers/vbd/solver_vbd.py:97`). The gather
compiles at 89 registers — at 256 threads/block that risks poorer SM
residency; at 128 it fits better. Measured: reverting to 256 costs 28.2 ms
on the cloth bucket (~1.08x), consistent with the report's 5.2%
microbenchmark. The elasticity dispatch picks the specialized kernels from
static model properties:

```python
            if self.use_particle_tile_solve:
                particle_count_in_color = self.model.particle_color_groups[color].size
                if self.model.tet_count == 0:
                    elasticity_dim = (particle_count_in_color + 1) // 2 * (2 * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE)
                    wp.launch(
                        kernel=solve_elasticity_tile_two_particles,
                        dim=elasticity_dim,
...
                elif self._use_tet_only_tile_solve:
                    elasticity_dim = particle_count_in_color * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
                    wp.launch(
```
(`newton/_src/solvers/vbd/solver_vbd.py:2794-2800, 2827-2829`)

Mixed cloth+tet models fall through to the untouched legacy kernel, so the
specializations are pure additions.

## 8. SDF distance queries inside contact detection: value-only / gradient-only variants (measured: no effect)

**Where this lives / what this part does.** Deepest layer of the detection
kernels from §3: the iterative search evaluates the shape's signed
distance function many times per candidate (golden-section search along
edges, Frank-Wolfe over triangles, in
`newton/_src/geometry/soft_contacts_sdf.py`). Each evaluation used to
return distance value, adjusted value, and gradient together.

**Old behavior.** Every SDF query inside the edge/face contact searches
computed both the distance value and the gradient, even when the caller
consumed only one of them; the internal golden-section line search also
finished with a full evaluation whose result was thrown away.

**New behavior.** Output-specific evaluators — `eval_shape_sdf_lower`
(`newton/_src/geometry/soft_contacts_sdf.py:142`) for scalar-only queries
and `eval_shape_sdf_grad` (`newton/_src/geometry/soft_contacts_sdf.py:156`)
for Frank-Wolfe directions — plus a golden-section search that returns only
the parameter and skips the discarded final evaluation:

```python
    inv_phi = float(0.6180339887498949)  # 1 / golden ratio
    lo = float(0.0)
    hi = float(1.0)
    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    fc = eval_shape_sdf_lower(geo, scale, (1.0 - c) * p + c * q, shape_sdf_index, texture_sdf_table)
    fd = eval_shape_sdf_lower(geo, scale, (1.0 - d) * p + d * q, shape_sdf_index, texture_sdf_table)
```
(`newton/_src/geometry/soft_contacts_sdf.py:224-230`)

The construction is careful (the texture value-only sampler replays the
gradient sampler's exact dequantization order so accepted contacts stay
bit-identical, `newton/_src/geometry/sdf_texture.py:1147`), and the tests
compare the split helpers bit-for-bit. But reverting all of it — full
`eval_shape_sdf` everywhere, restored discarded line-search evaluations —
changes neither workload measurably (generation bucket ±0.2%). The likely
reason: both tasks' full-surface shapes are analytic primitives whose SDF
evaluations are a handful of ALU ops; the kernels' cost is dominated by
memory behavior (fixed by grouping, §2) and candidate count (fixed by AABB
rejection, §3), and the compiler already removes much of the dead-output
arithmetic within the 255-register kernel. The change is correctness-safe
and might matter for texture-SDF-heavy (mesh shape) scenes, but on these
benchmarks it carries no measured weight.

## Findings

- **R1 — The acceleration decomposes into four load-bearing changes.**
  (Section numbers as in this document.)
  Cloth: gather (+546.6 ms), shape grouping (+591.2 ms), two-particle
  elasticity (+386.5 ms), gather block size (+28.2 ms), duals (+58.0 ms) —
  against a total cloth saving of 1479.6 ms/60 steps. Volume: gather
  (+430.3 ms), AABB rejection (+140.3 ms), tet-only elasticity (+66.5 ms),
  duals (+34.4 ms), grouping (+22.2 ms) — against a total of 911.6 ms. The
  per-change reverts over-sum the total because of interactions (R6).
- **R2 — The candidate-list sort (§2) is undersold by the report.** It carries no isolated
  claim there, yet it is the single largest cloth-generation contributor
  (2.61x of the bucket alone). Anyone porting a subset of this branch should
  treat `group_by_shape` (`newton/_src/sim/collide.py:515`) as a first-class
  optimization, not a locality footnote.
- **R3 — The SDF query splitting (§8 here; §1-2 of the report) shows zero
  measurable gain** on both shipped workloads (cloth +0.5 ms, volume
  +0.6 ms — noise). It is bitwise-safe and well-tested, but its ~200 lines of
  parallel evaluator code are not load-bearing for the claimed speedups.
- **R4 — The bounding-box early-out (§3) is volume-only in practice.** 4.02x of the volume
  generation bucket; exactly nothing on cloth (features there always overlap
  their shapes). Fine as shipped — the test costs six comparisons.
- **R5 — Test regression: the branch dropped `import math` from
  `newton/tests/test_solver_vbd.py`** while `math.pi` is still used
  (`newton/tests/test_solver_vbd.py:3195`), so 14 pre-existing tests
  (capsule friction, damping, cable) now ERROR with `NameError` on both CPU
  and CUDA. The report's "83/83 targeted tests" was a curated subset that
  missed this. One-line fix; must land before merge.
- **R6 — The candidate sort (§2) and the contact-force gather (§1) interact.** With grouping kept but the gather
  reverted, the legacy scatter is *slower than baseline* (890.6 vs 805.0 ms):
  shape-grouped candidate order makes neighboring contact records hit the
  same particles, raising atomic contention in the old scatter. The changes
  are a package; per-change savings do not add linearly.
- **R7 — "~2x" needs the kernel-time qualifier.** Summed CUDA kernel time
  drops 1.94x (cloth) / 2.24x (volume), but end-to-end env throughput is
  1.664x / 1.317x — CPU work, launch gaps, and env-manager overhead do not
  shrink. The reports state this correctly; PR messaging should too.

## Correctness evidence

| Check | Result |
|---|---|
| 20-step, 1-env, seed-42 trajectory, cloth (526 contacts/step) | `pass: true`, every numeric field `max_abs = 0.0` (bitwise) |
| Same, frozen-topology volume (contact-free window) | `pass: true`, bitwise — covers tet-only elasticity, not the contact path |
| New optimization tests (gather/adjacency/graph replay/tet-only/two-particle/deterministic fallback) | pass |
| Full `test_solver_vbd.py` | 132/146 pass; 14 ERROR from the dropped `import math` (R5), all pre-existing tests untouched by the branch |
| Deterministic mode / CPU | legacy scatter + legacy kernels retained (verified by dispatch flags and fallback test) |

The gather/adjacency path is exercised for real by the cloth comparison;
the volume trajectory window contains no soft contacts, the same limitation
the report itself discloses.
