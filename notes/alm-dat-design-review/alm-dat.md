# ALM-DAT self-contact design, as implemented

- Purpose: describe the implemented `alm_dat` self-contact path of `SolverVBD` as design chapters, one per component, so the data structures and the pseudo code can be reviewed line by line before the code is touched again.
- Inputs: `AnkaChan/newton` branch `ankac/alm-dat-design-review` at `119d9ba90` (the review worktree's solver code on one revision); Anka's step pseudocode of 2026-10-14 (`inputs/algorithm-reference.md`); the design note of 2026-09-14 (`inputs/design-note.md`); the conformance review of 2026-10-06 (`inputs/conformance-review.md`).
- Pseudo code convention: a call tree of function names. Every name exists today and is cited; nothing in this document is new. Comments stand on their own line or after #. Control flow (for, if, else, return) is written out.
- Terms: `x_start` is the detection pose of the step; the `trial` is the untruncated VBD iterate; the `anchor` is the certified pose that only moves through plane truncation; a `slot` is one record of the detector's CSR rows; `P`, `E`, `T` are the particle, edge and triangle counts; `C_vt`, `C_ee` are the row capacities of the two pair families.
- Status: chapters 1 to 3 drafted 2026-11-05 from the code. Nothing changes until the review closes.

## 1. The step scheme: detection pose, trial, anchor and commit

### 1.1 Intent

The host pattern is the `penalty_dat` path. One detection per checkpoint resets the DAT reference `pos_prev_collision_detection` and zeroes `particle_displacements` (`solver_vbd.py:3122-3128`). After the inertial step and after every colour, `_penetration_free_truncation` computes a fraction per vertex from the division planes of the current positions (`solver_vbd.py:3263-3279`, kernel `particle_vbd_kernels.py:1988`), then scales the accumulated displacement by that fraction and clamps its length to `0.5 gamma (margin + gap)` (`solver_vbd.py:3253-3257`, `particle_vbd_kernels.py:2121-2140`). The iterate itself is therefore penetration-free at every colour, and the contact force is a penalty on the witness gap (`solver_vbd.py:4110-4134`, kernel `particle_vbd_kernels.py:1593`).

The `alm_dat` path keeps the detection reference and the displacement bookkeeping, but never truncates the iterate. The iterate is the trial: `x_start + displacement`, written by `_alm_dat_materialize_trial` with fraction one and no clamp (`solver_vbd.py:3392-3401`). A second position array, the anchor, starts at `x_start` (`solver_vbd.py:3441`) and moves toward the trial only through `_alm_dat_truncate`: plane truncation, then the motion budget, then a rebuild of the planes at the new anchor (`solver_vbd.py:3403-3434`). The contact rows read the trial and the multipliers learn from the trial after every sweep (`solver_vbd.py:4086-4109`, `solver_vbd.py:4211-4231`). The step commits the anchor, never the trial (`solver_vbd.py:4731-4737`).

### 1.2 Old and new

| | `penalty_dat` today | `alm_dat` |
|---|---|---|
| Soft self-contact schedule | default `PRE_POST_INIT`, two detections per substep (`solver_vbd.py:4814-4816`) | forced `PRE_INIT`, one detection per substep (`solver_vbd.py:4810-4813`, check `solver_vbd.py:4932-4935`) |
| What is truncated | the iterate, after the inertial step and after every colour (`solver_vbd.py:3385-3386`, `solver_vbd.py:4207-4208`) | the anchor only; the trial is free (`solver_vbd.py:3383-3384`, `solver_vbd.py:4205-4206`) |
| Truncation cadence | every colour | after the inertial step, every `particle_dat_plane_update_interval` sweeps, and `particle_dat_truncation_iterations` final rounds (`solver_vbd.py:3457`, `solver_vbd.py:4232-4235`, `solver_vbd.py:4734-4735`) |
| Division planes | recomputed per pair from the current positions inside the truncation kernel (`particle_vbd_kernels.py:1988`), not stored | one stored plane per slot, built at the anchor, re-placed at every truncation (`particle_dat_kernels.py:182-197`, chapter 2) |
| Motion budget | magnitude clamp of the accumulated displacement at every truncation (`particle_vbd_kernels.py:2133-2136`) | applied once per commit on the anchor's displacement from `x_start`; path-length or distance rule (`particle_dat_kernels.py:517-546`) |
| Contact force | penalty of the witness gap (`particle_vbd_kernels.py:1593`) | compliant ALM row with slack, hard wall, damping and Coulomb friction, read at the trial (chapter 3) |
| Contact memory | none | one normal and one tangential multiplier per slot, matched across steps through sorted rows (`particle_dat_kernels.py:757-779`) |
| Committed state | the truncated iterate | the anchor (`solver_vbd.py:4736`) |
| Coverage flag | none | `_dat_coverage` from the detector's overflow flags (`particle_dat_kernels.py:1404-1421`) |
| Row order | detection order | sorted within each row by the detector (`solver_vbd.py:1174`, `solver_vbd.py:1180`) |

### 1.3 Argument list

```
SolverVBD(model, *,                                       # keyword only (solver_vbd.py:326)
    particle_enable_self_contact: bool = False,           # required True for alm_dat (solver_vbd.py:993-994)
    particle_self_contact_margin: float | None = None,    # cushion width R and thickness [m]; None = 0.2 (solver_vbd.py:849)
    particle_self_contact_gap: float | None = None,       # detection gap beyond the margin [m]; None = 0.0 (solver_vbd.py:850)
    dat_conservative_bound_relaxation: float = 0.85,      # gamma, shared with every DAT truncation (solver_vbd.py:330)
    particle_vertex_contact_buffer_size: int = 8,         # average vt records per vertex; C_vt = P x this (solver_vbd.py:338)
    particle_edge_contact_buffer_size: int = 16,          # average ee records per edge; C_ee = E x this (solver_vbd.py:339)
    particle_self_contact_method: str = "penalty_dat",    # "alm_dat" selects this path (solver_vbd.py:346)
    particle_dat_plane_update_interval: int = 1,          # truncate and rebuild every this many sweeps; 0 = never mid-step (solver_vbd.py:347)
    particle_dat_truncation_iterations: int = 2,          # final truncation rounds; at least 1 (solver_vbd.py:348, 1026-1030)
    particle_dat_budget_rule: str = "path_length",        # or "distance" (solver_vbd.py:349, 1031-1035)
    particle_contact_alm_rho_scale: float = 1.0,          # multiplies the inertia metric rho (solver_vbd.py:350)
    particle_contact_alm_rho_floor_ratio: float = 1.0,    # rho >= ratio x soft_contact_ke; 0 = no floor (solver_vbd.py:351)
    particle_contact_alm_hard_gap: float | None = None,   # wall height above the plane [m]; None = 0.25 x margin (solver_vbd.py:1011-1012)
    particle_contact_alm_transition_band: float = 0.0,    # smooth stiffness band above the wall [m]; 0 = piecewise law (solver_vbd.py:353)
    particle_contact_alm_wall_stiffness_ratio: float = 9.0,  # wall stiffness = ratio x rho, with a band (solver_vbd.py:354)
    particle_topological_contact_filter_threshold: int = 2,  # rings excluded from detection (solver_vbd.py:355)
    particle_rest_shape_contact_exclusion_radius: float = 0.0,  # rest-shape exclusion [m] (solver_vbd.py:356)
    rigid_soft_enable_dat: bool = False,                  # must stay False with alm_dat (solver_vbd.py:995-998)
)
```

Resolution order of the geometry scalars: `particle_self_contact_margin` and `particle_self_contact_gap`, then the legacy `particle_self_contact_radius`, then the defaults 0.2 and 0.0 (`solver_vbd.py:813-850`). The detection reach is `margin + gap` (`solver_vbd.py:1140`). The motion budget is `0.5 gamma (margin + gap)`, half the reach (`solver_vbd.py:3388-3390`). The hard gap must lie in `[0, margin)` and the transition band must fit the cushion `margin - hard_gap` (`solver_vbd.py:1013-1023`).

Material scalars come from the model, not from arguments: `soft_contact_ke` is the cushion stiffness `K`, `soft_contact_kd` the gap-rate damping and `soft_contact_mu` the friction coefficient (`solver_vbd.py:4096`, `solver_vbd.py:4103-4104`).

Not arguments: the plane margin `_dat_margin` is the constant 0 (`solver_vbd.py:1037`); the per-pair separation band comes from `dat_separation_epsilon` of the coordinate scale (`particle_dat_kernels.py:275`, `rigid_vbd_kernels.py:6949-6951`); the detector's row sorting is forced on for this path (`solver_vbd.py:1174`, `solver_vbd.py:1180`); the launch size of the per-slot kernels is `min(max(C_vt, C_ee), sm_count x 2048)` (`solver_vbd.py:1206-1226`).

### 1.4 Input types

The component consumes the detector's result struct `TriMeshCollisionInfo` (`tri_mesh_collision.py:36`). The members read by this path:

| Member | dtype, shape | Filled by | Meaning |
|---|---|---|---|
| `global_pair_counts` | int32 `[4]` | detection (`tri_mesh_collision.py:67`) | `[vt demand, vt overflow, ee demand, ee overflow]`; the two flags feed `_dat_coverage` (`kernels.py:1291`, `kernels.py:1293`) |
| `vertex_colliding_triangles` | int32 `[2 C_vt]` flat | CSR build (`tri_mesh_collision.py:71`) | interleaved `(vertex, triangle)` per slot; slot `i` is the plane and multiplier index |
| `vertex_colliding_triangles_offsets` | int32 `[P + 1]` | CSR build (`tri_mesh_collision.py:73`) | row offsets per vertex; the last entry is the stored vt count |
| `edge_colliding_edges` | int32 `[2 C_ee]` flat | CSR build (`tri_mesh_collision.py:90`) | interleaved `(edge, edge)` per slot, both directions stored |
| `edge_colliding_edges_offsets` | int32 `[E + 1]` | CSR build (`tri_mesh_collision.py:92`) | row offsets per edge; the last entry is the stored ee count |

The rows are sorted ascending by counterpart index when `sort_contact_rows` is on (`tri_mesh_collision.py:391`, `tri_mesh_collision.py:935-947`, kernel `kernels.py:1835`); the history matching of chapter 3 needs this order. The struct lives in `Contacts.soft_self_contact_data` when the solver owns a pipeline (`contacts.py:437`), else in the standalone detector (`solver_vbd.py:1254-1258`). The solver keeps a device-side copy of the struct in a one-element array and refreshes it when the storage generation changes (`solver_vbd.py:1261-1278`, `tri_mesh_collision.py:493`, `tri_mesh_collision.py:1009`).

### 1.5 Data structures on the solver

| Attribute | dtype, shape | Meaning |
|---|---|---|
| `pos_prev_collision_detection` | vec3 `[P]` | `x_start`: the detection pose of the step; reset at every detection (`solver_vbd.py:1202`, `solver_vbd.py:3127`) |
| `particle_displacements` | vec3 `[P]` | trial minus `x_start`: inertial step plus every local solve (`solver_vbd.py:1203`, `solver_vbd.py:3377`, `solver_vbd.py:4200-4202`) |
| `particle_q_prev` | vec3 `[P]` | positions at the step start, for velocities and the row's relative displacement (`solver_vbd.py:1125`, `particle_vbd_kernels.py:1571`) |
| `inertia` | vec3 `[P]` | inertial target of the step (`solver_vbd.py:1128`, `particle_vbd_kernels.py:1556-1585`) |
| `_dat_anchor` | vec3 `[P]` | the certified anchor; starts at `x_start`, moves only through commits (`solver_vbd.py:1177`, `solver_vbd.py:3441`, `solver_vbd.py:3416-3424`) |
| `truncation_ts` | float `[P]` | per-vertex fraction of `trial - anchor` admitted by the planes, atomic minimum over the vertex's slots (`solver_vbd.py:1204`, `particle_dat_kernels.py:401-457`) |
| `_dat_coverage` | int32 `[1]` | 1 when the last detection stored every pair, else 0 (`solver_vbd.py:1178`, `particle_dat_kernels.py:1404-1421`) |
| `trimesh_collision_info` | `TriMeshCollisionInfo` `[1]` | device copy of the result struct read by every per-slot kernel (`solver_vbd.py:1271-1277`) |
| `_dat_planes` | `ParticleContactDatPlanes` | stored planes per slot (chapter 2); re-created when the row capacity changes (`solver_vbd.py:1279-1283`) |
| `_dat_alm_state` | `ParticleContactAlmState` | multipliers per slot and last step's rows (chapter 3); re-created with the planes (`solver_vbd.py:1283`) |
| `_alm_dat_iteration` | int, host | sweeps since the step start, for the plane-update cadence (`solver_vbd.py:3438`, `solver_vbd.py:4232-4235`) |
| `particle_self_contact_evaluation_kernel_launch_size` | int, host | grid of the per-slot kernels; the kernels stride over the stored slots (`solver_vbd.py:1206-1226`, `particle_dat_kernels.py:262-263`) |

Conventions reused from the host: the detection schedule is the solver-base slot mechanism, and `alm_dat` only fixes the default and the check (`solver_vbd.py:4798-4819`, `solver_vbd.py:4919-4945`). Storage growth stays the user's call between steps; `check_and_grow_self_contact_buffers` grows the detector storage and refreshes the solver's view, which drops the multiplier history once (`solver_vbd.py:1288-1332`, `solver_vbd.py:1279-1283`). The step is graph-capturable; the view refresh is not and raises during capture (`solver_vbd.py:1264-1270`).

### 1.6 Pseudo code

```
step(state_in, state_out, control, contacts, dt)                       # solver_vbd.py:2677
  -> _validate_dat_collision_schedules        # alm_dat: SOFT_SELF_CONTACT must be PRE_INIT
  -> _pre_initialization_detection(state_in, dt)
     -> _refresh_collision_sets
        -> _collision_detection_penetration_free(state_in)   # detector: vt and ee rows within margin + gap, sorted
        -> particle_self_contact_info                        # refresh the device copy if the storage changed
        -> _reset_dat_references                             # x_start = particle_q, displacement = 0
  -> _initialize_particles(state_in, state_out, contacts, dt)
     -> forward_step                           # particle_q_prev = x_start, displacement = inertial step
     -> _alm_dat_initialize(state_in)
        -> record_contact_coverage             # _dat_coverage from the overflow flags
        anchor = x_start
        -> _alm_dat_materialize_trial          # trial = x_start + displacement
        vt_valid = ee_valid = 0                # slots were re-assigned by the detection
        -> build_contact_planes(anchor, trial, refresh_normals = True)
        -> match_contact_history               # multipliers from last step's sorted rows, zero otherwise
        -> _alm_dat_truncate(trial)            # the inertial step is the first commit
  for k in range(iterations):
    -> _solve_particle_iteration(state_in, state_out, contacts, dt)
       for color in particle_color_groups:
         -> gather_particle_body_contact_force_and_hessian    # rigid-soft penalty, unchanged
         -> accumulate_alm_contact_force_and_hessian(color, trial, particle_q_prev)   # chapter 3
         -> local VBD solve of the color       # writes displacement
         -> _alm_dat_materialize_trial         # trial = x_start + displacement, no truncation
       -> update_alm_contact_multipliers(trial)   # once per sweep, after all colors
       _alm_dat_iteration += 1
       if plane_update_interval > 0 and _alm_dat_iteration % plane_update_interval == 0:
         -> _alm_dat_truncate(trial)
       state_out.particle_q = trial
  -> _finalize_particles(state_out, dt)
     for r in range(truncation_iterations):
       -> _alm_dat_truncate(state_out.particle_q)
     state_out.particle_q = anchor             # the committed state is the anchor
     -> snapshot_contact_history               # rows and multipliers for the next step
     -> update_velocity                        # (anchor - particle_q_prev) / dt

_alm_dat_truncate(trial, refresh_normals)                                 # solver_vbd.py:3403
  -> truncate_against_contact_planes(anchor, trial, planes, gamma) -> sigma   # chapter 2
  -> commit_truncated_displacement(x_start, anchor, trial, sigma, budget, rule)   # anchor moves
  -> build_contact_planes(anchor, trial, refresh_normals)                       # planes at the new anchor
```

All names exist today: `step` `solver_vbd.py:2677-2753`, `_alm_dat_initialize` `solver_vbd.py:3436-3457`, `_solve_particle_iteration` `solver_vbd.py:4000-4236`, `_finalize_particles` `solver_vbd.py:4725-4744`, `_alm_dat_truncate` `solver_vbd.py:3403-3434`. The local VBD solve of a colour is the tiled elasticity kernel or `solve_particle_iteration_kernel`, unchanged by this path.

### 1.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| Detections per substep | none; `alm_dat` forces one (`solver_vbd.py:4810-4813`) | one detection: planes certified for the whole substep, speed cap `0.5 gamma (margin + gap) / dt`; two detections as in `penalty_dat`: twice the cap, but the stored planes would need a second certified anchor | free-fall and drop cases of the collision-fidelity benchmark |
| Budget rule | `particle_dat_budget_rule` | `path_length`: distance travelled plus step within the budget, a vertex at the cap freezes; `distance`: end point within the budget sphere, steps back are admitted | collision-fidelity benchmark run of 2026-11-04 (`notes/collision-benchmark-alm-dat` on the review worktree) |
| Budget scope | none | every vertex (today, `particle_dat_kernels.py:549-575` launches over `P`); or only vertices with a stored plane | none |
| Rigid-soft DAT with `alm_dat` | none; refused (`solver_vbd.py:995-998`) | off: rigid-soft contact is penalty only; on: would need rigid-soft planes in the final truncation | none |
| Plane update cadence | `particle_dat_plane_update_interval` | 1: anchor follows every sweep; larger: cheaper, planes staler; 0: final rounds only | none |
| Final rounds | `particle_dat_truncation_iterations` | more rounds let the anchor approach the trial through re-placed planes | none |
| Storage growth | user call `check_and_grow_self_contact_buffers` | between steps, drops the multiplier history; or automatic, which would break graph capture | none |

### 1.8 Open questions for the reviewer

1. Should `distance` become the default budget rule, or stay an option until the benchmark run is read (fork "Budget rule")?
2. Should the budget apply only to vertices that own a stored plane, or to every vertex as today (fork "Budget scope")? The second keeps the certificate for pairs that enter the reach during the step.
3. Is one detection per substep the intended final design, or should the path support a second certified anchor after the inertial step (fork "Detections per substep")?
4. Should coverage loss (`_dat_coverage == 0`) stay a flag, or should the step grow the storage and re-detect on its own outside capture (fork "Storage growth")?

## 2. Plane placement and truncation

### 2.1 Intent

The host pattern is the rigid-soft DAT of the rigid path. It certifies a separating direction per pair with `find_vertex_triangle_separator` and `find_edge_edge_separator` (`rigid_vbd_kernels.py:7066`, `rigid_vbd_kernels.py:7147`), keeps a representable half-band `dat_separation_epsilon` around the plane (`rigid_vbd_kernels.py:6949-6951`), and caps a straight vertex trajectory at a fraction `gamma` of the way to the plane with `planar_truncation_t` (`rigid_vbd_kernels.py:7277-7308`). The `penalty_dat` soft path recomputes its planes from the current positions in every truncation and keeps nothing (`particle_vbd_kernels.py:1988`).

The `alm_dat` path uses the same three helpers, but stores one plane per slot. `build_contact_planes` finds or keeps the normal at the anchor and places the offset with `dat_plane_offset`, which reserves room for the approaching side and balances two approaching sides so that the slowest vertex on each side can take the same fraction of its remaining motion (`particle_dat_kernels.py:233-366`, `particle_dat_kernels.py:66-179`). `truncate_against_contact_planes` takes the atomic minimum of the plane fractions over every slot of a vertex (`particle_dat_kernels.py:401-457`). `commit_truncated_displacement` moves the anchor by that fraction of `trial - anchor` and applies the motion budget (`particle_dat_kernels.py:517-546`).

### 2.2 Old and new

| | `penalty_dat` today | `alm_dat` |
|---|---|---|
| Plane lifetime | one truncation call | from one truncation to the next; cleared only by a detection (`solver_vbd.py:3444-3445`) |
| Plane pose | current, truncated iterate | the anchor; normals certified there (`particle_dat_kernels.py:279-292`, `particle_dat_kernels.py:334-345`) |
| Offset rule | fixed proportional split in the truncation kernel (`particle_vbd_kernels.py:1988`) | `dat_plane_offset`: one-sided room when one side approaches, balanced slowest-vertex fractions when both do, midpoint when none does (`particle_dat_kernels.py:107-179`) |
| Thickness margin per side | the separation band | `max(margin, eps)` capped at half the current gap; the solver passes `margin = 0` (`particle_dat_kernels.py:103`, `solver_vbd.py:1037`) |
| Edge-edge records | both stored directions truncate | canonical direction `e1 < e2` only; the mirror slot is marked invalid (`particle_dat_kernels.py:266`, `particle_dat_kernels.py:313-314`) |
| Failed separator | pair skipped | a stored plane is kept, a slot without one stays unconstrained (`particle_dat_kernels.py:290-292`) |
| Budget | clamp of the accumulated displacement length (`particle_vbd_kernels.py:2133-2136`) | on the anchor's offset from `x_start` at the commit, path-length or distance rule (`particle_dat_kernels.py:493-546`) |

### 2.3 Argument list

```
build_contact_planes(model, anchor, trial, collision_info_array, planes, *,   # particle_dat_kernels.py:369
    margin: float,                # thickness reserved per side [m]; the solver passes 0.0 (solver_vbd.py:3431)
    refresh_normals: bool,        # True: search a certified normal with the stored one as hint; False: keep it
    launch_size: int | None,      # grid; None = _plane_launch_size (particle_dat_kernels.py:216-220)
)
truncate_against_contact_planes(model, anchor, trial, collision_info_array, planes, *,   # particle_dat_kernels.py:460
    gamma: float,                 # fraction of the way to the plane, dat_conservative_bound_relaxation
    sigma: wp.array[float],       # out, [P], filled with 1 then atomic-minimized
    launch_size: int | None,
)
commit_truncated_displacement(x_start, anchor, trial, sigma, *,              # particle_dat_kernels.py:549
    budget: float,                # 0.5 gamma (margin + gap) [m] (solver_vbd.py:3388-3390)
    anchor_out: wp.array[wp.vec3],  # [P]; the solver passes the anchor itself (solver_vbd.py:3422)
    distance_rule: bool = False,  # True selects the distance rule (solver_vbd.py:3423)
)
```

The solver calls the three in this order inside `_alm_dat_truncate`, always with `refresh_normals = True` (`solver_vbd.py:3403-3434`, `solver_vbd.py:3457`, `solver_vbd.py:4235`, `solver_vbd.py:4735`). Not arguments: the separation band per pair, `dat_separation_epsilon` of the largest coordinate of the four points (`particle_dat_kernels.py:223-230`, `particle_dat_kernels.py:275`, `particle_dat_kernels.py:330`).

### 2.4 Input types

| Member of `ParticleContactDatPlanes` | dtype, shape | Detector counterpart | Meaning |
|---|---|---|---|
| `vt_normal` | vec3 `[C_vt]` | slot of `vertex_colliding_triangles` (`tri_mesh_collision.py:71`) | unit normal from the triangle (negative side) toward the vertex (positive side) (`particle_dat_kernels.py:338`) |
| `vt_point` | vec3 `[C_vt]` | same slot | a point on the plane, `origin + b n` (`particle_dat_kernels.py:179`) |
| `vt_margin` | float `[C_vt]` | same slot | thickness reserved per side at this placement, `margin_eff` (`particle_dat_kernels.py:103`) |
| `vt_valid` | int32 `[C_vt]` | same slot | 1 when the slot owns a certified plane (`particle_dat_kernels.py:184-188`) |
| `ee_normal`, `ee_point`, `ee_margin`, `ee_valid` | as above, `[C_ee]` | slot of `edge_colliding_edges` (`tri_mesh_collision.py:90`) | normal from `e2` (negative) toward `e1` (positive); valid only for `e1 < e2` (`particle_dat_kernels.py:283`, `particle_dat_kernels.py:313-314`) |

Constructor: `create_particle_contact_dat_planes` sizes every array to the row capacity of its family, at least one (`particle_dat_kernels.py:200-213`). All arrays start at zero, so a fresh struct has no valid plane.

### 2.5 Data structures on the solver

The planes above and `truncation_ts` (chapter 1) are the only storage. Per pair, the kernels derive transient values: the separation band `separation_eps` (`particle_dat_kernels.py:275`, `particle_dat_kernels.py:330`); the positive and negative vertex sets as `mat33` with their remaining displacements `trial - anchor` (`particle_dat_kernels.py:294-305`, `particle_dat_kernels.py:347-358`); heights along `n`, the gap `positive_min - negative_max`, and the admissible offset interval `[b_lo, b_hi]` (`particle_dat_kernels.py:93-105`).

### 2.6 Pseudo code

```
_build_contact_planes_kernel(anchor, trial, ..., margin, refresh_normals, planes)   # particle_dat_kernels.py:233
  for slot in stored ee slots (strided):
    if e1 < e2:                                      # canonical direction only
      eps = -> dat_separation_epsilon(-> _coordinate_scale)
      if refresh_normals or not ee_valid[slot]:
        found, n = -> find_edge_edge_separator(anchor points, hint = stored n if valid)
        if found: ee_valid[slot] = 1                 # else: keep the stored plane, or stay unconstrained
      if ee_valid[slot]:
        point, room, margin_eff = -> dat_plane_offset(n, e2 points, e2 remaining, e1 points, e1 remaining, margin, eps)
        store n, point, margin_eff
    else:
      ee_valid[slot] = 0
  for slot in stored vt slots (strided):
    same with -> find_vertex_triangle_separator, triangle = negative side, vertex = positive side

dat_plane_offset(n, negative, negative_disp, positive, positive_disp, margin, eps)   # particle_dat_kernels.py:66
  gap = min height of the positive side - max height of the negative side
  margin_eff = min(max(margin, eps), gap / 2)
  b_lo, b_hi = negative max + margin_eff, positive min - margin_eff
  if nobody approaches:            b = midpoint(b_lo, b_hi)
  elif only the negative side:     b = b_hi                    # give it all the room
  elif only the positive side:     b = b_lo
  else:
    for each approaching (i on negative, j on positive):
      b_ij = offset where the two slowest fractions are equal, clamped to [b_lo, b_hi]
      keep the b_ij with the largest -> _dat_plane_room
  return origin + b n, room, margin_eff

_truncate_against_contact_planes_kernel(anchor, trial, ..., planes, gamma, sigma)   # particle_dat_kernels.py:401
  for slot in stored ee slots, then vt slots (strided):
    if valid:
      for each vertex of the positive side:  t = -> planar_truncation_t(anchor, trial - anchor, +n, p, gamma, m); atomic_min(sigma)
      for each vertex of the negative side:  t = -> planar_truncation_t(anchor, trial - anchor, -n, p, gamma, m); atomic_min(sigma)

_commit_truncated_displacement_kernel(x_start, anchor, trial, sigma, budget, rule)   # particle_dat_kernels.py:517
  step = sigma * (trial - anchor); offset = anchor - x_start
  if rule is distance: step *= -> budget_sphere_fraction(offset, step, budget)      # end point inside the sphere
  else: if |offset| + |step| > budget: step *= max(budget - |offset|, 0) / |step|  # path length
  anchor_out = anchor + step
```

All names exist today: `_build_contact_planes_kernel` `particle_dat_kernels.py:233-366`, `dat_plane_offset` `particle_dat_kernels.py:66-179`, `_dat_plane_room` `particle_dat_kernels.py:36-63`, `_truncate_against_contact_planes_kernel` `particle_dat_kernels.py:401-457`, `_commit_truncated_displacement_kernel` `particle_dat_kernels.py:517-546`, `budget_sphere_fraction` `particle_dat_kernels.py:493-514`, `planar_truncation_t` `rigid_vbd_kernels.py:7277-7308`.

### 2.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| Normal refresh cadence | `refresh_normals` of `build_contact_planes`, always True from the solver | refresh at every commit (today): normals follow the closest-point direction at the anchor; keep normals within a step and slide offsets only, as Anka's reference keeps them between `NORMAL_UPDATE_INTERVAL` rounds (`inputs/algorithm-reference.md`) | none |
| Plane margin | `margin` of `build_contact_planes`, solver passes 0 | 0: the certificate is "no crossing", the row's hard gap keeps loads off the plane; the cloth thickness: planes keep surfaces apart on their own | none |
| Failed separator at the anchor | none | keep the stored plane (today); or invalidate the slot | none |
| Plane offset rule | none | balanced slowest-vertex fractions (today, the DAT proportional split generalized); midpoint always | none |
| Edge-edge direction | none | canonical `e1 < e2` only (today); both directions with a shared plane | none |

### 2.8 Open questions for the reviewer

1. Is the always-on normal refresh the intended behaviour, or should the solver keep the normals between the final rounds as the reference pseudocode does (fork "Normal refresh cadence")?
2. Should the planes carry the cloth thickness as their margin, now that the row has its own hard gap (fork "Plane margin")?
3. When a stored plane cannot be re-certified at the anchor, is keeping it the right rule, given that the anchor never crossed it (fork "Failed separator")?

## 3. The ALM contact row, its multipliers and their history

### 3.1 Intent

The host pattern is the penalty self-contact row of `penalty_dat`, `accumulate_self_contact_force_and_hessian` (`particle_vbd_kernels.py:1593`): one thread per stored slot, closest-point witness, barycentric weights, and the force and Hessian scattered to the vertices of the current colour with atomic adds. The `alm_dat` row keeps that structure: the same slots, the same witness geometry, the same per-colour scatter with the same weights (`particle_dat_kernels.py:1022-1044`, `particle_dat_kernels.py:1075-1085`).

What changes is the force law. Each slot carries a normal multiplier and a tangential multiplier (`particle_dat_kernels.py:757-779`). The normal row is a compliant cushion of stiffness `K = soft_contact_ke` and width `R = margin - hard_gap` in front of a hard wall at `hard_gap` above the stored plane, solved through one slack with the ALM metric `rho` (`particle_dat_kernels.py:614-644`, `particle_dat_kernels.py:908-951`). Gap-rate damping and Coulomb friction add to it (`particle_dat_kernels.py:734-754`, `particle_dat_kernels.py:708-731`). The multipliers are replaced by the row's transmitted force at the trial pose once per sweep (`particle_dat_kernels.py:1089-1169`, `solver_vbd.py:4211-4231`) and carried to the next step by matching the sorted rows (`particle_dat_kernels.py:1172-1223`).

### 3.2 Old and new

| | `penalty_dat` today | `alm_dat` |
|---|---|---|
| Normal force | penalty `K (R - gap)` on the witness gap (`particle_vbd_kernels.py:1593`) | slack-gap ALM row: released, cushion `s lam + k_eff (R - gap)`, or wall `lam - rho gap` (`particle_dat_kernels.py:634-644`) |
| Memory | none | `lam` and `lambda_t` per slot (`particle_dat_kernels.py:768-771`) |
| Contact direction | closest-point direction | closest-point direction, sign fixed by the stored plane normal (`particle_dat_kernels.py:834-840`) |
| Where the force is read | the truncated iterate | the trial (`solver_vbd.py:4091`) |
| Multiplier update | none | once per sweep at the trial, after all colours (`solver_vbd.py:4211-4231`) |
| Friction | penalty friction with `friction_epsilon` | tangential multiplier capped by `mu x transmitted`, metric at the inertia scale (`particle_dat_kernels.py:945-950`) |
| Damping | `soft_contact_kd` penalty damping | gap-rate damping of the closing motion inside the margin (`particle_dat_kernels.py:734-754`) |
| Across steps | none | sorted-row matching (`particle_dat_kernels.py:1332-1365`, `particle_dat_kernels.py:1368-1401`) |

### 3.3 Argument list

```
accumulate_alm_contact_force_and_hessian(model, dt, color, pos, pos_prev, collision_info_array, planes, alm, *,   # particle_dat_kernels.py:1226
    stiffness: float,             # K = soft_contact_ke [N/m] (solver_vbd.py:4096)
    radius: float,                # R_total = particle_self_contact_margin [m] (solver_vbd.py:4097)
    rho_scale: float,             # multiplies the inertia metric (solver_vbd.py:4098)
    rho_floor_ratio: float,       # rho >= ratio x K (solver_vbd.py:4099)
    hard_gap: float,              # wall height above the plane [m]; the row sees gap - hard_gap and R_total - hard_gap (particle_dat_kernels.py:935-937)
    transition_band: float = 0.0, # smooth stiffness band [m]
    wall_stiffness_ratio: float = 9.0,  # wall stiffness = ratio x rho
    damping: float,               # soft_contact_kd (solver_vbd.py:4103)
    friction_mu: float,           # soft_contact_mu (solver_vbd.py:4104)
    parallel_eps: float,          # edge-edge parallel epsilon (solver_vbd.py:4105)
    launch_size: int,
    particle_forces, particle_hessians,   # out, [P] vec3 and [P] mat33, atomic adds
)
update_alm_contact_multipliers(model, dt, pos, pos_prev, collision_info_array, planes, alm, *,   # particle_dat_kernels.py:1283
    stiffness, radius, rho_scale, rho_floor_ratio, hard_gap, transition_band, wall_stiffness_ratio,
    friction_mu, parallel_eps, launch_size,   # same meanings; writes alm in place
)
```

The metric of a slot is `rho = max(rho_inertia x rho_scale, rho_floor_ratio x K)` with `rho_inertia = 1 / (dt^2 x mobility)` and the mobility the barycentric sum of inverse masses (`particle_dat_kernels.py:578-588`, `particle_dat_kernels.py:843-848`, `particle_dat_kernels.py:896-904`). The friction metric is `rho_inertia x rho_scale` without the floor (`particle_dat_kernels.py:946-950`). Not arguments: the witness weights come from `triangle_closest_point` and `wp.closest_point_edge_edge` at the trial pose (`particle_dat_kernels.py:830`, `particle_dat_kernels.py:879`).

### 3.4 Input types

| Member of `ParticleContactAlmState` | dtype, shape | Detector counterpart | Meaning |
|---|---|---|---|
| `vt_lambda`, `ee_lambda` | float `[C_vt]`, `[C_ee]` | slot of each family | normal multiplier, the compressive force carried by the slot [N] (`particle_dat_kernels.py:768-769`) |
| `vt_lambda_t`, `ee_lambda_t` | vec3 `[C_vt]`, `[C_ee]` | slot | tangential multiplier in world space, the friction force on the negative side [N] (`particle_dat_kernels.py:770-771`) |
| `prev_vt_offsets`, `prev_ee_offsets` | int32 `[P + 1]`, `[E + 1]` | `*_offsets` of last step (`tri_mesh_collision.py:73`, `tri_mesh_collision.py:92`) | last step's row offsets (`particle_dat_kernels.py:772`, `particle_dat_kernels.py:776`) |
| `prev_vt_counterpart`, `prev_ee_counterpart` | int32 `[C_vt]`, `[C_ee]` | second index of each pair | last step's counterpart per slot, ascending within a row (`particle_dat_kernels.py:773`, `particle_dat_kernels.py:777`) |
| `prev_vt_lambda`, `prev_ee_lambda`, `prev_vt_lambda_t`, `prev_ee_lambda_t` | as the live multipliers | slot of last step | last step's multipliers (`particle_dat_kernels.py:774-779`) |

Constructor: `create_particle_contact_alm_state` sizes the slot arrays to the row capacities and the offset arrays to `P + 1` and `E + 1`, all zero (`particle_dat_kernels.py:782-799`). The matching needs the rows sorted ascending by counterpart on both time levels (`particle_dat_kernels.py:1183`, `tri_mesh_collision.py:935-947`).

### 3.5 Data structures on the solver

The struct above is the only storage of this component. Per slot, the kernels derive transient values at the trial pose: the contact direction `n`, the signed witness gap, the weights (`bary` or `s, t`), `rho_inertia` and the relative displacement `rel` of the positive witness against the negative witness since the step start, using the current weights on both time levels (`particle_dat_kernels.py:802-849`, `particle_dat_kernels.py:852-905`). The host reads the live multipliers through `particle_contact_alm_multipliers` (`solver_vbd.py:3473-3487`).

### 3.6 Pseudo code

```
_accumulate_alm_contact_force_and_hessian_kernel(dt, color, pos = trial, pos_prev, ..., planes, alm, ...)   # particle_dat_kernels.py:954
  for slot in stored ee slots, then vt slots (strided):
    if valid[slot]:
      n, gap, weights, rho_inertia, rel = -> _edge_edge_row_geometry | _vertex_triangle_row_geometry(slot, pos, pos_prev, n_stored)
      F_n, k_n, f_t, J_t, transmitted = -> _alm_contact_response(gap, rel, n, lam[slot], lambda_t[slot], rho_inertia, ...)
      if F_n > 0 or k_n > 0 or f_t != 0:
        for each vertex v of the pair with signed weight w:
          if color[v] == color:
            atomic_add(particle_forces[v],   w F_n n - w f_t)
            atomic_add(particle_hessians[v], w^2 (k_n n n^T + J_t))

_alm_contact_response(gap, rel, n, lam, lambda_t, rho_inertia, dt, K, R_total, ..., hard_gap, band, wall_ratio, kd, mu)   # particle_dat_kernels.py:908
  rho = -> contact_alm_rho(rho_inertia, K, rho_scale, floor_ratio)
  transmitted, k = -> alm_contact_row_smooth(gap - hard_gap, lam, K, R_total - hard_gap, rho, band, wall_ratio)
     if band <= 0: return -> alm_contact_row                   # piecewise law
     if gap' - lam / rho >= R': return 0, 0                    # released
     if cushion force at z = 0 < lam - rho gap': return max(lam - rho gap', 0), rho    # wall
     z = root of F(z) = lam + rho (z - gap') by clamped Newton  # cushion
     return F(z), K(z) rho / (K(z) + rho)
  F_damp, k_damp = -> contact_damping_row(gap, n . rel, kd, R_total, dt)   # closing motion inside the margin only
  if mu > 0 and transmitted > 0:
    f_t, J_t = -> alm_friction_row(lambda_t, rho_inertia rho_scale, rel, n, mu transmitted)   # disk clamp
  return transmitted + F_damp, k + k_damp, f_t, J_t, transmitted

_update_alm_contact_multipliers_kernel(dt, pos = trial, pos_prev, ..., planes, ..., alm)   # particle_dat_kernels.py:1089
  for slot in stored ee slots, then vt slots (strided):
    if valid[slot]:
      geometry as above at the trial
      lam[slot]      = transmitted force of -> alm_contact_row_smooth      # the dual eats the attempt
      lambda_t[slot] = clamped trial of -> alm_friction_row with budget mu lam[slot]

match_contact_history(info, alm)                                 # particle_dat_kernels.py:1332, after each detection
  for family in (vt, ee): -> _match_rows_kernel per element      # merge sorted rows; equal counterpart carries lam, lambda_t; else 0
snapshot_contact_history(info, alm)                              # particle_dat_kernels.py:1368, at the step end
  for family in (vt, ee): -> _snapshot_rows_kernel per element   # copy offsets, counterparts and multipliers
```

All names exist today: `_alm_contact_response` `particle_dat_kernels.py:908-951`, `alm_contact_row` `particle_dat_kernels.py:614-644`, `alm_contact_row_smooth` `particle_dat_kernels.py:669-705`, `cushion_force` `particle_dat_kernels.py:647-666`, `contact_damping_row` `particle_dat_kernels.py:734-754`, `alm_friction_row` `particle_dat_kernels.py:708-731`, `_match_rows_kernel` `particle_dat_kernels.py:1172-1199`, `_snapshot_rows_kernel` `particle_dat_kernels.py:1202-1223`.

### 3.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| ALM metric floor | `particle_contact_alm_rho_floor_ratio` | 1 (today): in-sweep response at least half penalty-like, converged force unchanged; 0: plain inertia scale, slower and prone to lateral pumping (`particle_dat_kernels.py:580-587`) | cloth bag and stacked sheets of the showcase (review worktree, 2026-10-05) |
| Wall position | `particle_contact_alm_hard_gap` | a quarter of the margin (today): loads rest on the wall, the plane stays a safety net; 0: wall on the plane | none |
| Compliant-to-hard transition | `particle_contact_alm_transition_band`, `particle_contact_alm_wall_stiffness_ratio` | 0 (today): piecewise law with a stiffness jump at the wall; a band: smoothstep stiffness rise, scalar Newton slack solve | single-vertex and patch probes of 2026-10-14 (review worktree) |
| Friction metric | none | inertia scale without the floor (today): the disk saturates after microns of slip; with the floor: stiffer stick | none |
| Damping law | none | gap-rate dashpot anchored at the margin crossing (today); penalty-style damping of the full rate | none |
| Multiplier update cadence | none | once per sweep after all colours (today); per colour | none |
| History key | none | sorted rows merged by counterpart index (today); hashed pair keys | none |

### 3.8 Open questions for the reviewer

1. Should the smooth transition become the default row (fork "Compliant-to-hard transition"), or stay off until the benchmark is read?
2. Is the quarter-margin default of the wall position the intended value, or should the wall sit on the plane with the thickness carried by the plane margin (fork "Wall position", chapter 2 fork "Plane margin")?
3. Should the friction metric get the same floor as the normal metric (fork "Friction metric")?
4. Is once-per-sweep the intended multiplier update cadence, given that the reference pseudocode updates after every sweep as well (`inputs/algorithm-reference.md`)?
