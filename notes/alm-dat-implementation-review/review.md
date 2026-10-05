# ALM-DAT self-contact in SolverVBD: implementation review

Reviewed range: `4996f98f..986b7c92` on `ankac/alm-dat-solver` (6 commits, 2 new
modules/tests, 2142 insertions), base = PR 4222 head e6ee0e2f plus the 35
elasticity-ALM commits. Stated intent: Anka's pseudo code and the ALM-DAT
design notes (task folder `AI-Logs/Newton/tasks/alm-dat/`), narrowed by the
decisions taken on 2026-10-02..04 (vertex-witness force row, canonical
edge-edge record, planes placed from the committed pose). Reviewer: the author
of the change, applying `REVIEW_GUIDELINES.rst` adversarially; every finding
below carries its evidence, and probes that were run are named.

## 1. Big picture

Old behavior. With `particle_enable_self_contact=True`, SolverVBD detects
vertex-triangle and edge-edge pairs once per step, applies a log-barrier
penalty force (plus friction and damping) along the closest-point direction,
and after every color of every iteration rebuilds a division plane per pair
from the detection pose plus the accumulated displacement and clips each
particle's displacement against it (`_penetration_free_truncation`,
`newton/_src/solvers/vbd/solver_vbd.py:3191`). Contact forces have no memory;
a loaded contact needs a stiff barrier to hold.

New behavior. `particle_self_contact_method="alm_dat"` (default stays
`"penalty_dat"`, that path is byte-for-byte unchanged) keeps one certified
anchor pose per step and one stored division plane per detected pair. The VBD
iterate moves freely (it may penetrate), a compliant ALM row per pair pushes
along the closest-point direction with a multiplier that is carried across
steps by matching the detector's sorted rows, and the accumulated motion is
truncated against the stored planes after every N iterations and twice before
the step commits. The committed state is the anchor, so every stored pair stays
separated by at least the DAT separation band; a loaded contact rests on an ALM
hard wall a thickness (`hard_gap`, default margin / 4) above the plane with
`lambda` equal to the load, so a soft `K` no longer needs a barrier blow-up.

Verdict (details in section 9 and 11): **Fits if narrowed.** Opt-in,
experimental, default untouched, 730 tests green; but one demonstrated defect
(per-world reset wipes all worlds' contact history), one design decision that
contradicts the agreed formula sheet (rho floored at K) and needs Anka's
sign-off, an undocumented semantic (friction and damping ignored on this path),
and a public surface of six parameters plus two methods of which only three
carry the experimental marker.

## One step through the pipeline (where each section lives)

```
SolverVBD.step()                                            once per step
  PRE_INIT detection (forced for alm_dat) .................................. §7
    -> sorted CSR rows, global_pair_counts; pos_prev_collision_detection = x_start
  INITIALIZE  (_initialize_particles -> _alm_dat_initialize)
    forward_step: inertia target, displacement = v dt
    record_contact_coverage (overflow flags -> device int) ................. §8
    anchor = x_start; trial = x_start + displacement ........................ §7
    clear slot flags; build_contact_planes at anchor (certified separators) . §2, §3
    match_contact_history (previous sorted rows -> lambda) ................. §6
    truncate -> commit anchor -> rebuild planes  (one round) ............... §4
  ITERATE  (x iterations)
    for each color:
      elasticity + rigid-soft contact forces (unchanged)
      accumulate_alm_contact_force_and_hessian (one thread per record) ..... §5
      local 3x3 / tiled solves (unchanged; accumulate displacements)
      materialize trial = x_start + displacement (no truncation) ........... §7
    update_alm_contact_multipliers at the trial pose ........................ §6
    every particle_dat_plane_update_interval iterations:
      truncate -> commit anchor -> rebuild planes ........................... §4
  FINALIZE  (_finalize_particles)
    particle_dat_truncation_iterations x (truncate -> commit -> rebuild) .... §4
    particle_q = anchor; snapshot_contact_history ........................... §6
    velocities = (anchor - q_prev) / dt (unchanged)
  between steps (host, not captured): check_and_grow_self_contact_buffers
    -> particle_self_contact_info re-creates planes + ALM state ............. §8
```

## 2. Stored division planes: one certified plane per detector slot

Where this lives / what this part does. `newton/_src/solvers/vbd/particle_dat_kernels.py:233-365`,
launched by `build_contact_planes` from `_alm_dat_initialize` and after every
truncation. One thread per stored record of the detector's CSR rows (vertex
rows then edge rows), strided like the existing per-pair kernels.

Old behavior. The old truncation kernel rebuilt a plane from scratch on every
call (after every color), with no memory between calls, and processed both
stored directions of every edge-edge pair.

New behavior. The plane is state: normal, point, effective margin and a
validity flag per slot (`ParticleContactDatPlanes`,
`newton/_src/solvers/vbd/particle_dat_kernels.py:182-196`). Normals come from
the rigid-DAT certified separators (`find_vertex_triangle_separator`,
`find_edge_edge_separator`), which test the closest-point axis, the face or
cross axis and the edge-support axes and keep the largest certified gap. Only
the canonical edge-edge record (`e1 < e2`) owns a plane; the mirrored record is
marked invalid. Within one detection a pair that has become thinner than the
separation band keeps its stored plane instead of losing it.

```python
            valid = int(planes.ee_valid[i])
            n = planes.ee_normal[i]
            if refresh_normals == 1 or valid == 0:
                hint = wp.vec3(0.0)
                if valid == 1:
                    hint = n
                # n points from e2 (negative) toward e1 (positive)
                found, n_found, _s0, _s1, _gap, _candidate = find_edge_edge_separator(
                    x_e1_a, x_e1_b, x_e2_a, x_e2_b, hint, separation_eps
                )
                if found:
                    n = n_found
                    valid = int(1)
                # not found: a pair thinner than the separation band keeps its stored
                # plane, which still separates the anchor because the committed motion
                # never crossed it; a slot without a stored plane stays unconstrained
            if valid == 1:
                positive = wp.mat33(0.0)
                positive[0] = x_e1_a
                positive[1] = x_e1_b
                positive_disp = wp.mat33(0.0)
                positive_disp[0] = trial[e1_a] - x_e1_a
                positive_disp[1] = trial[e1_b] - x_e1_b
                negative = wp.mat33(0.0)
                negative[0] = x_e2_a
                negative[1] = x_e2_b
                negative_disp = wp.mat33(0.0)
                negative_disp[0] = trial[e2_a] - x_e2_a
                negative_disp[1] = trial[e2_b] - x_e2_b
                point, _room, margin_eff = dat_plane_offset(
                    n, negative, negative_disp, 2, positive, positive_disp, 2, margin, separation_eps
                )
                planes.ee_normal[i] = n
                planes.ee_point[i] = point
                planes.ee_margin[i] = margin_eff
            planes.ee_valid[i] = valid
        else:
            planes.ee_valid[i] = 0
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:276-313`)

Why it is written this way. Slots are re-assigned by every detection, so the
"keep the stored plane" branch is only safe within one detection; the solver
therefore zeroes both validity arrays before the first build of a step
(`newton/_src/solvers/vbd/solver_vbd.py:3399-3401`). The "keep" branch exists
because a pinched pair converges toward the separation band over repeated gamma
rounds and the separator search then fails; dropping the plane there is how a
strip crossed a plate during development (commit 09e9521f). The invariant that
makes keeping safe: the committed motion never crosses a stored plane, so the
anchor is still on the right side of it.

Subtle points. The stored normal is only a hint to the separator; when a
different certified axis has a larger gap, the normal changes between rounds and
the force row's gap changes definition with it (lambda is carried as a scalar).
The `refresh_normals=0` branch (slide only) is never selected by the solver
(see R9).

## 3. Plane placement: the maximin offset along the normal

Where this lives / what this part does. `dat_plane_offset`
(`newton/_src/solvers/vbd/particle_dat_kernels.py:66-178`), called from the
plane builder with the committed positions and the remaining displacement
`trial - anchor`. Pure function, no memory.

Old behavior. The paper's proportional split between the two closest-point
witnesses, clamped so each side keeps 5 percent of the gap
(`place_dat_division_plane` in the rigid DAT module).

New behavior. Every primitive vertex gets a "time to plane" line; the plane is
placed where the slowest approaching vertex on each side gets the same fraction
of its remaining motion. One witness per side reduces to the paper's rule. A
side that does not approach gets no room. Each side keeps at least the
separation band and at most half the gap.

```python
    origin = negative_pos[0]
    negative_max = float(-_DAT_INFINITE_ROOM)
    for i in range(negative_count):
        negative_max = wp.max(negative_max, wp.dot(n, negative_pos[i] - origin))
    positive_min = float(_DAT_INFINITE_ROOM)
    for j in range(positive_count):
        positive_min = wp.min(positive_min, wp.dot(n, positive_pos[j] - origin))
    gap = positive_min - negative_max
    # reserve at least the separation band on each side (never place the plane on a
    # primitive); fall back to the midpoint when the pair is thinner than the band
    margin_eff = wp.min(wp.max(margin, eps), wp.max(0.0, 0.5 * gap))
    b_lo = negative_max + margin_eff
    b_hi = positive_min - margin_eff
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:92-104`)

```python
        for i in range(negative_count):
            rate_i = wp.dot(n, negative_disp[i])
            if rate_i > eps:
                height_i = wp.dot(n, negative_pos[i] - origin)
                for j in range(positive_count):
                    rate_j = -wp.dot(n, positive_disp[j])
                    if rate_j > eps:
                        height_j = wp.dot(n, positive_pos[j] - origin)
                        b_ij = (rate_j * (height_i + margin_eff) + rate_i * (height_j - margin_eff)) / (rate_i + rate_j)
                        b_ij = wp.clamp(b_ij, b_lo, b_hi)
                        room_ij = _dat_plane_room(
                            b_ij,
                            n,
                            origin,
                            negative_pos,
                            negative_disp,
                            negative_count,
                            positive_pos,
                            positive_disp,
                            positive_count,
                            margin_eff,
                            eps,
                        )
                        if room_ij > best_room:
                            best_room = room_ij
                            best_b = b_ij
    return origin + best_b * n, best_room, margin_eff
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:152-178`)

Why it is written this way. The balanced fraction is a concave piecewise-linear
function of the offset, so its maximum is at a crossing of one negative-side
line with one positive-side line; with at most 3 and 2 vertices per side that
is at most 3 or 4 candidates, each scored by the true minimum over all lines.
Heights are measured from a primitive vertex (`origin`) for float32 precision
at metre-scale coordinates, as the rigid-DAT separators do.

Subtle points. `margin` is passed as 0 by the solver (R7), so `margin_eff` is
the separation band (about 1e-6 m at unit scale) except for pairs thinner than
twice that. The maximin uses the full remaining displacement of a vertex that
other planes may cap further; that is the coupling the repeated rounds in
section 4 recover.

## 4. Truncation against stored planes and the committed anchor

Where this lives / what this part does. Three per-call launches in
`SolverVBD._alm_dat_truncate` (`newton/_src/solvers/vbd/solver_vbd.py:3359-3389`):
`truncate_against_contact_planes` (per record, atomic minimum of per-vertex
fractions), `commit_truncated_displacement` (per particle), then
`build_contact_planes` at the new anchor. Called once at initialization, after
every `particle_dat_plane_update_interval` iterations, and
`particle_dat_truncation_iterations` times in finalize.

Old behavior. After every color, planes were rebuilt and the whole accumulated
displacement from the detection pose was clipped; the committed positions were
the clipped iterate, and the solve continued from the clipped positions.

New behavior. The anchor advances by a fraction `sigma_i` of the remaining
displacement toward the trial, never past a stored plane (`planar_truncation_t`
keeps `dot(n, x - p) >= margin_eff`), with the total displacement from the
detection pose capped by the motion budget. The trial is left untouched; the
solve keeps working on it.

```python
    while i < vt_count:
        if planes.vt_valid[i] == 1:
            v = collision_info.vertex_colliding_triangles[2 * i]
            tri = collision_info.vertex_colliding_triangles[2 * i + 1]
            n = planes.vt_normal[i]
            p = planes.vt_point[i]
            m = planes.vt_margin[i]
            t = planar_truncation_t(anchor[v], trial[v] - anchor[v], n, p, gamma, m)
            if t < 1.0:
                wp.atomic_min(sigma, v, t)
            for k in range(3):
                u = tri_indices[tri, k]
                t = planar_truncation_t(anchor[u], trial[u] - anchor[u], -n, p, gamma, m)
                if t < 1.0:
                    wp.atomic_min(sigma, u, t)
        i += stride
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:441-456`)

```python
    i = wp.tid()
    step = sigma[i] * (trial[i] - anchor[i])
    step_length = wp.length(step)
    used = wp.length(anchor[i] - x_start[i])
    if used + step_length > budget and step_length > 0.0:
        step = step * (wp.max(budget - used, 0.0) / step_length)
    anchor_out[i] = anchor[i] + step
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:503-509`)

Why it is written this way. Independent per-vertex fractions are jointly safe
because half-spaces are convex and the motion is linear; the only coupling is
through plane placement, which the next round re-does with the remaining
displacement. The budget is measured from `x_start`, not from the anchor,
because the detection's query reach is what certifies that no undetected pair
can come into contact.

Subtle points. The budget clamp uses the triangle inequality, so a particle
near its budget can be stopped short even when the exact clamp would allow more
(R13). `sigma` is the solver's `truncation_ts` array, which the per-color
materialization also fills with ones; the two uses never overlap in time.

## 5. The slack-gap ALM contact row

Where this lives / what this part does. `alm_contact_row`
(`newton/_src/solvers/vbd/particle_dat_kernels.py:550-579`), the row geometry
helpers (`newton/_src/solvers/vbd/particle_dat_kernels.py:619-701`) and the
per-color scatter kernel (`newton/_src/solvers/vbd/particle_dat_kernels.py:705-803`),
launched instead of the penalty kernel from
`newton/_src/solvers/vbd/solver_vbd.py:4041-4059`.

Old behavior. Force magnitude from a log barrier active below half the margin
and a quadratic blend above it, applied along the closest-point direction, plus
Coulomb friction and gap-rate damping; the barrier's stiffness grows without
bound toward the surface.

New behavior. One slack variable per pair carries a linear cushion of stiffness
`K = soft_contact_ke` over the band `hard_gap .. margin`; past the hard gap the
row is a wall whose stiffness is `rho`, independent of `K`. The multiplier is
the force memory. Friction and damping are not part of the row (R3).

```python
    if rho <= 0.0:
        return float(0.0), float(0.0)
    z_free = gap - lam / rho
    if z_free >= radius:
        return float(0.0), float(0.0)
    z0 = (stiffness * radius - lam + rho * gap) / (stiffness + rho)
    if z0 >= 0.0:
        s = stiffness / (stiffness + rho)
        k_eff = stiffness * rho / (stiffness + rho)
        return wp.max(s * lam + k_eff * (radius - gap), 0.0), k_eff
    return wp.max(lam - rho * gap, 0.0), rho
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:569-579`)

```python
    _closest, bary, _feature = triangle_closest_point(a, b, c, pos[v])
    witness = bary[0] * a + bary[1] * b + bary[2] * c
    diff = pos[v] - witness
    dis = wp.length(diff)
    direction = n
    gap = float(0.0)
    if dis > _CONTACT_DIRECTION_EPS:
        direction = diff / dis
        if wp.dot(direction, n) < 0.0:
            direction = -direction
        gap = wp.dot(direction, diff)
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:638-648`)

```python
            force, k = alm_contact_row(
                gap - hard_gap,
                alm.vt_lambda[i],
                stiffness,
                radius - hard_gap,
                contact_alm_rho(rho, stiffness, rho_scale, rho_floor_ratio),
            )
            if force > 0.0 or k > 0.0:
                nn = wp.outer(n, n)
                if particle_colors[v] == current_color:
                    wp.atomic_add(particle_forces, v, force * n)
                    wp.atomic_add(particle_hessians, v, k * nn)
                for j in range(3):
                    u = tri_indices[tri, j]
                    if particle_colors[u] == current_color:
                        wp.atomic_add(particle_forces, u, -bary[j] * force * n)
                        wp.atomic_add(particle_hessians, u, bary[j] * bary[j] * k * nn)
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:786-802`)

Why it is written this way. The force acts along the closest-point direction,
as the penalty path does, and the stored plane normal only fixes its sign: a
vertex that crossed the triangle reports a negative gap instead of a flipped
normal. This replaced an earlier version that pushed along the plane normal
itself; certified separators favour the largest-gap axis, which for laterally
adjacent triangles is tilted or horizontal, and a frictionless resting patch
drifted sideways (commit 5fb01828). The regime switch is continuous in force
and at `C = (lambda - K R) / rho`, not at the plane, so a loaded contact rests in
the band pushed by `lambda`.

Subtle points. `rho` is recomputed from the current barycentric weights at
every evaluation and floored at `K` (`contact_alm_rho`,
`newton/_src/solvers/vbd/particle_dat_kernels.py:537-546`); see R2 for why the
floor exists and why it needs Anka's agreement. With a 1 kg vertex at 240 Hz
the inertia metric is 57600 N/m, so there the row is penalty-like inside a
sweep regardless of the floor, and only the wall term `lambda - rho * gap`
distinguishes it from the old force.

## 6. Multiplier update and history matching

Where this lives / what this part does. `_update_alm_contact_multipliers_kernel`
(`newton/_src/solvers/vbd/particle_dat_kernels.py:807-866`), launched once per
iteration after the color sweep at the trial pose
(`newton/_src/solvers/vbd/solver_vbd.py:4161-4182`); `_match_rows_kernel` and
`_snapshot_rows_kernel` (`newton/_src/solvers/vbd/particle_dat_kernels.py:870-912`),
one thread per vertex or edge, at step start and step end.

Old behavior. No contact memory; every step and every iteration started from
the geometric penalty.

New behavior. `lambda <- transmitted force at the trial pose` once per sweep,
so the multiplier learns from the attempt even though the committed state never
penetrates. At step end the sorted rows and multipliers are copied; at the next
step start each element merges its previous row with its new row in one pass
(both sorted by counterpart index) and carries matched multipliers, zero for new
pairs.

```python
    element = wp.tid()
    start = offsets[element]
    end = offsets[element + 1]
    p = prev_offsets[element]
    p_end = prev_offsets[element + 1]
    for slot in range(start, end):
        counterpart = values[2 * slot + 1]
        while p < p_end and prev_counterpart[p] < counterpart:
            p += 1
        carried = float(0.0)
        if p < p_end and prev_counterpart[p] == counterpart:
            carried = prev_lambda[p]
        lam[slot] = carried
```
(`newton/_src/solvers/vbd/particle_dat_kernels.py:879-891`)

```python
        if self._alm_dat:
            # the dual eats the attempt: multipliers learn from the untruncated iterate
            update_alm_contact_multipliers(
                model,
                dt,
                state_in.particle_q,
                self.trimesh_collision_info,
                self._dat_planes,
                self._dat_alm_state,
                stiffness=self.model.soft_contact_ke,
                radius=self.particle_self_contact_margin,
                rho_scale=self.particle_contact_alm_rho_scale,
                rho_floor_ratio=self.particle_contact_alm_rho_floor_ratio,
                hard_gap=self.particle_contact_alm_hard_gap,
                parallel_eps=self._self_contact_edge_edge_parallel_epsilon,
                launch_size=self.particle_self_contact_evaluation_kernel_launch_size,
            )
            self._alm_dat_iteration += 1
            interval = self.particle_dat_plane_update_interval
            if interval > 0 and self._alm_dat_iteration % interval == 0:
                self._alm_dat_truncate(state_in.particle_q, refresh_normals=True)
```
(`newton/_src/solvers/vbd/solver_vbd.py:4161-4182`)

Why it is written this way. Matching requires sorted rows, which is why the
detector's `sort_contact_rows` is enabled on both construction paths
(`newton/_src/solvers/vbd/solver_vbd.py:1138`). The dual update runs before the
plane refresh of the same iteration so the dual sees the same normal the
sweep's forces used; Anka's pseudo code places it after the refresh, and either
order keeps the attempt-fed property.

Subtle points. The non-canonical edge-edge slot also receives a matched value
but no kernel ever reads or updates it. A capacity change re-creates the ALM
state with zeroed history, so one step after growth starts cold (documented in
code, not in the user docstring).

## 7. Solver integration: schedule, materialization, cadence, finalize

Where this lives / what this part does. Constructor validation
(`newton/_src/solvers/vbd/solver_vbd.py:966-996`), schedule default and
validation (`newton/_src/solvers/vbd/solver_vbd.py:4756-4759`,
`newton/_src/solvers/vbd/solver_vbd.py:4878-4882`), the three private step
helpers (`newton/_src/solvers/vbd/solver_vbd.py:3348-3412`), the per-color
branch (`newton/_src/solvers/vbd/solver_vbd.py:4155-4158`) and finalize
(`newton/_src/solvers/vbd/solver_vbd.py:4677-4684`).

Old behavior. `AUTO` for the self-contact slot meant `PRE_POST_INIT`: detect at
step start and again right after initialization, resetting the motion reference
each time. Positions were materialized by the per-color truncation pass.

New behavior. `alm_dat` resolves `AUTO` to `PRE_INIT` and rejects any other
schedule, because a post-init or in-iteration detection would reset the motion
reference at a possibly penetrating trial pose. After each color the trial is
materialized as `x_start + displacement` without truncation; the committed
state is the anchor written in finalize.

```python
    def _alm_dat_materialize_trial(self, state: State) -> None:
        """Write the untruncated iterate ``x_start + displacement`` into ``state.particle_q``."""
        self.truncation_ts.fill_(1.0)
        wp.launch(
            kernel=apply_truncation_ts,
            dim=self.model.particle_count,
            inputs=[self.pos_prev_collision_detection, self.particle_displacements, self.truncation_ts, wp.inf],
            outputs=[self.particle_displacements, state.particle_q],
            device=self.device,
        )
```
(`newton/_src/solvers/vbd/solver_vbd.py:3348-3357`)

```python
    def _alm_dat_initialize(self, state_in: State) -> None:
        """Start the step's certified anchor at the detection pose and truncate the inertial target."""
        self._alm_dat_iteration = 0
        # dropped detection records would leave pairs without planes: record it
        record_contact_coverage(self._self_contact_info_source, self._dat_coverage, self.device)
        self._dat_anchor.assign(self.pos_prev_collision_detection)
        self._alm_dat_materialize_trial(state_in)
        # the detection re-assigned the row slots: stored planes belong to last step's pairs
        self._dat_planes.vt_valid.zero_()
        self._dat_planes.ee_valid.zero_()
        build_contact_planes(
            self.model,
            self._dat_anchor,
            state_in.particle_q,
            self.trimesh_collision_info,
            self._dat_planes,
            margin=self._dat_margin,
            refresh_normals=True,
            launch_size=self.particle_self_contact_evaluation_kernel_launch_size,
        )
        match_contact_history(self._self_contact_info_source, self._dat_alm_state, self.device)
        self._alm_dat_truncate(state_in.particle_q, refresh_normals=True)
```
(`newton/_src/solvers/vbd/solver_vbd.py:3391-3412`)

```python
        if self._alm_dat:
            # final truncation rounds against the stored planes; the committed
            # state is the certified anchor, never the raw iterate
            for _ in range(self.particle_dat_truncation_iterations):
                self._alm_dat_truncate(state_out.particle_q, refresh_normals=True)
            state_out.particle_q.assign(self._dat_anchor)
            snapshot_contact_history(self._self_contact_info_source, self._dat_alm_state, self.device)
```
(`newton/_src/solvers/vbd/solver_vbd.py:4677-4683`)

Why it is written this way. The solve kernels only accumulate displacements;
positions must be materialized for the next color to see them, which the old
path did inside its truncation pass. Requiring at least one final truncation
round is what makes the committed state certified; with zero rounds the output
would be the truncated inertial target (that misconfiguration is rejected at
construction).

Subtle points. Every call to `_alm_dat_truncate` passes `refresh_normals=True`,
so the stored normal is re-searched at every refresh; the "slide only" cadence
from the design is not wired (R9). `_alm_dat_materialize_trial` fills a
ones-array per color only to feed `apply_truncation_ts`; a dedicated kernel
would save one fill per color (R10).

## 8. Coverage flag, storage growth, reset

Where this lives / what this part does. `record_contact_coverage`
(`newton/_src/solvers/vbd/particle_dat_kernels.py:1068-1084`) at step start,
`particle_dat_coverage_ok()` (`newton/_src/solvers/vbd/solver_vbd.py:3414-3426`),
storage re-creation in `particle_self_contact_info()`
(`newton/_src/solvers/vbd/solver_vbd.py:1237-1241`) and the reset branch
(`newton/_src/solvers/vbd/solver_vbd.py:2862-2868`).

Old behavior. Overflow of the shared pair buffer dropped records silently for
the penalty force too, but neighbouring penalty forces usually masked the gap.

New behavior. A dropped record means a pair without a plane and therefore
without a certificate, so the detector's overflow flags are copied into a
device integer every step (graph-safe) and exposed through a host accessor;
tests show a pinched strip crossing a plate at the default budgets and
recovering after `check_and_grow_self_contact_buffers()`.

```python
        reset_particle_elasticity_alm(model, world_mask, self._particle_elasticity_alm_state)
        if self._alm_dat and self._dat_alm_state is not None:
            # contact multipliers are rebuilt from matching every step; a reset clears
            # the whole history (all worlds) rather than tracking per-world rows
            self._dat_alm_state.vt_lambda.zero_()
            self._dat_alm_state.ee_lambda.zero_()
            self._dat_alm_state.prev_vt_offsets.zero_()
            self._dat_alm_state.prev_ee_offsets.zero_()
```
(`newton/_src/solvers/vbd/solver_vbd.py:2861-2868`)

Why it is written this way. The flag is written on the device so captured
graphs keep it current; only the accessor synchronizes. The reset shortcut was
a deliberate simplification; its cross-world effect is R1.

## 9. Findings

- **R1 — Per-world reset clears every world's contact multipliers.** `reset(state, world_mask)` zeroes `vt_lambda`, `ee_lambda` and both previous-offset arrays unconditionally (`newton/_src/solvers/vbd/solver_vbd.py:2862-2868`), while `reset_particle_elasticity_alm` on the line above honours the mask. Probe: two worlds of a patch resting on a plate, 60 steps, `reset(world_mask=[False, True])`: multiplier sum 7.57 before, 0.00 after; world 0 restarts cold for one step (its contact re-converges within the step, so the observable effect is a transient, not a penetration). The public `reset` docstring promises selected-world semantics. Priority P1 (behavioral, documented only in a code comment). Evidence: probe (e) of this review. A per-world clear needs the particle world of each row's element, which the element index already gives.
- **R2 — Rho floored at K contradicts the agreed contact policy and needs sign-off.** The formula sheet (`alm_formulas.md` section 6) fixes the contact-normal rho at the plain inertia scale with no stiffness floor; `contact_alm_rho` (`newton/_src/solvers/vbd/particle_dat_kernels.py:537-546`) floors it at `soft_contact_ke` by default. Evidence for the floor: a 6x6 patch resting on a level frictionless plate drifted 9 cm in 4 s at the plain scale (rho 72 N/m against K 1000) and 4 to 6 mm, same as the penalty path, at floor 1 or scale 10 and above; the mechanism is the stale multiplier transmitted through a 67 N/m Hessian pumping lateral oscillation in the coupled Gauss-Seidel sweep. The converged force is unchanged, but the fixed-point speed and the in-sweep character are not what the sheet specified, the 9K bilateral floor and adaptive growth were not compared, and the single-vertex resting test passes with and without the floor. Priority P1 (requirements: do not silently reinterpret; this is a design decision for Anka, not a defect).
- **R3 — Friction and damping are silently ignored on `alm_dat`.** `soft_contact_mu` and `soft_contact_kd` are model-level parameters that the penalty kernel honours and the ALM row does not; the parameter docstring (`newton/_src/solvers/vbd/solver_vbd.py:474-485`) does not say so (only the changelog fragment does). A user switching methods on an existing scene loses friction without any warning. Priority P1 (undocumented semantic change; the behavior itself is the agreed phase scope).
- **R4 — Public surface is wider than its experimental marker.** Six new constructor parameters, but the `.. experimental::` block names three; `particle_dat_coverage_ok()` and `particle_contact_alm_multipliers()` are new public methods on a public class with no experimental marking and no API-reference consideration; `particle_self_contact_method` is a string selector where the guidelines prefer an enum for a closed category (`CODING_GUIDELINES.rst`, "Prefer enums for closed categories"). Priority P1 for the marker (compatibility contract), P2 for the enum and for whether the two methods belong on the public class at all.
- **R5 — The "wall above the safety plane" claim holds only for one-sided approach.** The row's gap is measured from the closest point, not from the plane; the plane is placed by the maximin rule between the primitives. When only one side approaches, the plane sits at the other side's surface plus the band and the wall (hard gap from the surface) is above it. When both sides approach, the plane can sit farther from the lower primitive than the hard gap, so a contact pushed to the wall still reaches the plane and the scalar truncation scales its whole displacement, which is the damping mechanism the hard gap was added to remove (heavy incline: 13 percent of the analytic slide before, 67 percent after). Probe (a), two free layers settled on each other for 1 s, showed zero truncated vertices at rest, so the exposure is dynamic (mutual approach), not static. Priority P1 as a question: the docstring overstates, and a mutual-approach scene with truncation statistics is the missing evidence.
- **R6 — Canonical edge-edge records depend on both directions being stored.** The detector records each edge-edge pair from both edges' threads, and the plane builder keeps only `e1 < e2`. Under overflow the kept prefix may contain only the mirrored record, so that pair has no plane; coverage is already reported false in that case, and `match_contact_history` writes multipliers into mirrored slots that nothing reads. Priority P2 (consistent with the coverage contract; worth a comment and a test).
- **R7 — Truncation margin is hardcoded to zero.** `self._dat_margin = 0.0` (`newton/_src/solvers/vbd/solver_vbd.py:995`) means the design's thickness margin for the safety planes is never applied; the certificate is "no crossing plus the separation band", exactly what the old path guarantees. Judgment call, but the knob is dead and the comment is the only record. Priority P2.
- **R8 — Cost is measured only on toy scenes.** 1.3 to 1.5x the penalty path per frame on 300 to 900-vertex scenes (plane build, sorted rows, per-color materialization, dual update, two final rounds). No demo-scale (cloth_twist, rollers) timing exists. Priority P2 (evidence request).
- **R9 — The slide-only plane refresh is dead code.** `build_contact_planes(refresh_normals=False)` exists and is tested at the kernel level, but every solver call passes `True`, so normals are re-searched at every refresh and the "SlidePlanes" cadence of the design is not exercised. Priority P2.
- **R10 — Per-color materialization pays a redundant fill.** `_alm_dat_materialize_trial` fills `truncation_ts` with ones before every color only to reuse `apply_truncation_ts`; a two-array add kernel would do. Priority P2.
- **R11 — Test coverage gaps.** No test for the solver-owned pipeline path (probe (b) passes), multi-world (probe (e) fails per R1), determinism, `particle_dat_plane_update_interval=0` (probe (d): no penetration, 4.2 mm minimum height on the drop), `reset()` (probe (c) passes), or edge-edge planes under overflow. Priority P2.
- **R12 — Mutable detector shared with the pipeline.** With a solver-owned pipeline the solver sets `sort_contact_rows = True` on the pipeline's detector (`newton/_src/solvers/vbd/solver_vbd.py:1138`); harmless for other consumers (rows are a permutation) but it is a side effect on an object the solver does not own. Priority P2.
- **R13 — Conservative budget clamp.** The commit kernel scales by `(budget - used) / |step|`, a triangle-inequality bound; a particle whose anchor is already near the budget can be stopped short of a step that the exact clamp would allow. Only matters at the motion budget (half the detection slack per step). Priority P2.

## 10. Test coverage

| test | file | what it demonstrates |
|---|---|---|
| plane offset on the worked example, one-sided approach, thin pair | `newton/tests/test_particle_dat_kernels.py` | the maximin closed form (b = 1.4375), band edges, margin shrink |
| planes certify every stored pair; mirrored edge records invalid | `newton/tests/test_particle_dat_kernels.py` | real detector rows, both families |
| truncate plus commit keeps every vertex on its side under random displacement | `newton/tests/test_particle_dat_kernels.py` | includes the negative control that the trial does cross planes |
| method validation; drop never penetrates (alm and penalty); graph capture replay | `newton/tests/test_solver_vbd_alm_dat.py` | solver path on cpu and cuda |
| pinched strip never crosses with adequate budgets and reports coverage; tiny budgets report lost coverage and recover after growth | `newton/tests/test_solver_vbd_alm_dat.py` | the overflow defect found by the comparison script |
| cushion law at its fixed point, hard wall, release | `newton/tests/test_particle_contact_alm.py` | `alm_contact_row` regimes and continuity |
| resting vertex at R - m g / K with lambda = m g and no sinking over 2 s | `newton/tests/test_particle_contact_alm.py` | warm start works (a cold start rests 2.7 mm lower) |
| overloaded vertex rests at the hard gap with lambda = weight | `newton/tests/test_particle_contact_alm.py` | hard gap semantics |
| resting patch's lateral drift stops growing | `newton/tests/test_particle_contact_alm.py` | force direction and rho floor |
| relied on implicitly, untested here | | pipeline path, multi-world, determinism, interval 0, reset, edge-edge overflow, demo-scale cost |

## 11. Verdict by axis

Fit: **Fits if narrowed.** The outcome (non-penetrating self-contact with a
compliant force that holds load without a stiffness blow-up) is a Newton
solver concern and is implemented as an opt-in experimental mode on the
existing solver with the default path untouched. Narrowing wanted before this
becomes a public contract: settle R2 against the formula sheet; collapse
`rho_scale` and `rho_floor_ratio` into one documented policy; decide whether
the two diagnostic methods belong on `SolverVBD` or behind a private seam;
mark every new parameter experimental.

Requirements: the agreed phases 1 to 3 (planes, switch, ALM row with matching)
are implemented and evidenced; R1 is a demonstrated defect, R3 and R5 are
documentation versus behavior gaps, R2 is an unagreed design change. Friction,
damping, rho growth, the room-based normal choice and rigid-soft coexistence
remain open by agreement.

Standards: R4 (experimental marker, enum, public methods); Towncrier fragment
present and categorized; tests use unittest with imperative docstrings; no new
dependencies; `wp.array[...]` annotations throughout; 730 tests green on cpu
and two CUDA devices after the last commit.
