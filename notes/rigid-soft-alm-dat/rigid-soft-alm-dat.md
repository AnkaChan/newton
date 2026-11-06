# Rigid-soft contact under ALM-DAT: design

- Purpose: one sentence. Give the rigid-soft contact rows (a soft particle, edge or face against a rigid shape) the scheme the soft self-contact has under `alm_dat`: a stored division plane per record, an ALM row with a multiplier that reads the trial, the host's support metric, and a particle anchor that commits against both plane families. One chapter per component, written before the code.
- Inputs: `newton` at `cfd6e07b5` on `ankac/alm-dat-main` (today's main, the #4222 self-contact storage, ALM-DAT with decisions D1 to D16); `inputs/alm-dat-self-contact-design.md` and `inputs/alm-dat-self-contact-decisions.md` (the self-contact design and its decisions); `inputs/algorithm-reference.md` (Anka's reference pseudo code of 2026-10-14).
- Pseudo code convention: structured statements with real function names, per the skill's `pseudo-code-style.md`. Names that exist today are written as they are; names that do not exist yet carry (new) the first time. Comments stand on their own line or after #.
- Status: chapter 1 drafted 2026-11-06. Nothing implemented.

## 1. The step scheme: rigid-soft rows beside the self-contact rows

### 1.1 Intent

Today a rigid-soft record is a penalty: a per-contact penalty stiffness ramped every iteration and re-seeded at every detection, with damping and projected friction, while the rigid-soft DAT rebuilds a division plane from scratch at every truncation. The `alm_dat` self-contact path has a different contract: certified anchor, stored planes, multiplier rows at the trial, support metric. The two cannot run together (`solver_vbd.py:1218`). This chapter puts the rigid-soft records on the `alm_dat` contract so a garment on a gripper gets the same compliant-ALM contact as cloth on cloth, and the refusal goes.

### 1.2 Old and new

| | rigid-soft today (`penalty_dat` or penalty with rigid-soft DAT) | rigid-soft under `alm_dat` |
|---|---|---|
| Force law | penalty of the record's gap with a per-contact stiffness `body_particle_contact_penalty_k`, seeded at `min(k_start, averaged ke)` and ramped by `beta x penetration` every iteration up to the averaged material stiffness (`rigid_vbd_kernels.py:2140-2175`, `rigid_vbd_kernels.py:5045-5085`, `rigid_vbd_kernels.py:7333-7405`); kd damping on closing; projected isotropic friction | the ALM row of the self-contact path: cushion `K` within the margin, hard wall `hard_gap` above the stored plane, multiplier updated once per sweep at the trial (`particle_dat_kernels.py:726-779`, `particle_dat_kernels.py:1021-1060`); gap-rate damping (`particle_dat_kernels.py:846-867`); friction as a capped tangential multiplier (`particle_dat_kernels.py:820-843`) |
| Gap | `dot(n, particle - body point) - (particle radius + shape margin)` with the record normal `n` (`rigid_vbd_kernels.py:2177-2257`); edge and face records use the barycentric soft point (`rigid_vbd_kernels.py:2259`) | witness gap along the stored plane normal: the barycentric soft point against the rigid primitive's support along `n`, minus particle radius and shape margin; negative once crossed |
| Division plane | rebuilt inside every truncation from the reference configuration, both sides, one thread per record (`rigid_vbd_kernels.py:8252-8500`); separator from `find_primitive_pair_separator` (`rigid_vbd_kernels.py:7916`), offset from `place_dat_division_plane` (`rigid_vbd_kernels.py:8001`) | one stored plane per record (`RigidSoftDatPlanes`, chapter 2), built at the particle anchor and the current body pose, re-placed at every commit like the self-contact planes (`particle_dat_kernels.py:369`) |
| What is truncated | particles in place, bodies in place after every body colour (`solver_vbd.py:3839`, `solver_vbd.py:5489`, `solver_vbd.py:4655`) | the particle anchor against both plane families in `_alm_dat_truncate` (`solver_vbd.py:4095`); bodies in place as today, against the stored planes (fork "Body side") |
| Schedule | RIGID and SOFT_SELF_CONTACT slots must match (`solver_vbd.py:6074-6100`); `alm_dat` refuses rigid-soft DAT (`solver_vbd.py:1218`) | the refusal goes; both slots detect together; the second detection of D3 re-detects both families at the anchor (`solver_vbd.py:4162`) |
| Penalty metric | the ramped penalty stiffness (`rigid_vbd_kernels.py:7333`) | the rigid contact rule (D15): the particle support along `n` (`particle_dat_kernels.py:566`) in series with the body's Delassus scale along `n` (`rigid_vbd_kernels.py:1205`, `rigid_vbd_kernels.py:1262`, `rigid_vbd_kernels.py:1276`); friction from the inertial supports in series |
| Multiplier memory | none: the penalty is re-seeded at every detection (`solver_vbd.py:4440`) | carried across detections by matching the record key (fork "Multiplier history") |
| Body side | the same penalty force and torque accumulated per body (`rigid_vbd_kernels.py:5945`) | the reaction of the same row: `-F n` and its torque at the record's body point, from the shared multiplier and gap |
| Masked rows (#4227) | no force, geometry kept (`soft_contacts_mesh.py:1593`) | no row, plane kept: the truncation still certifies the pair |
| Analytic SDF records | plane at the stored rigid surface point and record normal (`rigid_vbd_kernels.py:8252` docstring) | the same, re-placed with both sides' motion |
| Kinematic and world shapes | not truncated, Delassus zero | unchanged; rho from the particle support alone |

### 1.3 Argument list

```
SolverVBD(model, *,
    particle_self_contact_method: str = "penalty_dat",        # "alm_dat" selects the self-contact path (solver_vbd.py:499)
    rigid_soft_enable_dat: bool = False,                      # planes and truncation for rigid-soft records (solver_vbd.py:537)
    rigid_soft_contact_method: str = "penalty",               # (new) "alm": ALM rows on the rigid-soft records; requires rigid_soft_enable_dat and particle_self_contact_method="alm_dat"
    rigid_soft_dat_use_interval_arithmetic: bool = False,     # unchanged (solver_vbd.py:538)
    particle_contact_alm_hard_gap: float | None = None,       # shared row law [m]; None = 0.25 x margin (solver_vbd.py:503)
    particle_contact_alm_transition_band: float = 0.0,        # shared (solver_vbd.py:504)
    particle_contact_alm_wall_stiffness_ratio: float = 9.0,   # shared (solver_vbd.py:505)
    dat_conservative_bound_relaxation: float = 0.85,          # shared budget factor (solver_vbd.py:486)
)
```

The soft-side radius of a record is `particle_radius + shape_margin` as in the penalty law (`rigid_vbd_kernels.py:2218-2221`); the row's margin `R` is the pipeline's soft contact gap, the same reach the detection used.

### 1.4 Input types

One record of `Contacts`, length `soft_contact_max`, one index space (`contacts.py:400-458`):

| Member | dtype and shape | Host counterpart | Meaning |
|---|---|---|---|
| `soft_contact_count` | int32 `[1]` | `contacts.py:400` | number of live records |
| `soft_contact_indices` | vec3i `[C]` | `contacts.py:410` | soft particle ids, -1 padded: particle `(p,-1,-1)`, edge `(v0,v1,-1)`, face `(v0,v1,v2)` |
| `soft_contact_barycentric` | vec3 `[C]` | `contacts.py:448` | weights of the soft point on those particles |
| `soft_contact_shape` | int32 `[C]` | `contacts.py:450` | rigid shape; `shape_body` gives the body, -1 for world |
| `soft_contact_rigid_indices` | vec3i `[C]` | `contacts.py:416` | rigid mesh primitive corners, -1 padded; `(-1,-1,-1)` for analytic and recovery records |
| `soft_contact_body_pos` | vec3 `[C]` | `contacts.py:451` | contact point on the raw shape surface, body frame |
| `soft_contact_normal` | vec3 `[C]` | `contacts.py:458` | detection normal, rigid toward soft |
| `soft_contact_force_mask` | bool `[C]` or None | `soft_contacts_mesh.py:1593` | rows that carry no force |

Body state read by the rows: `body_q`, `body_q_prev_collision_detection` (`solver_vbd.py:3636`), `body_com`, `body_inv_mass`, `body_inv_inertia`, `shape_body`, `shape_transform`, `shape_scale`, `shape_source_ptr`, `shape_margin`, as `apply_rigid_soft_truncation` takes them (`rigid_vbd_kernels.py:8252-8275`).

### 1.5 Data structures on the solver

| Attribute | dtype, shape | Meaning |
|---|---|---|
| `_rigid_soft_planes` (new) | `RigidSoftDatPlanes`: normal vec3 `[C]`, point vec3 `[C]`, margin float `[C]`, valid int32 `[C]` | stored plane per record, world frame, rebuilt at every commit (chapter 2) |
| `_rigid_soft_alm_state` (new) | `RigidSoftAlmState`: lambda float `[C]`, lambda_t vec3 `[C]`; key int32 `[C, 7]`, prev_key, prev_lambda, prev_lambda_t only in the matched variant | multipliers per record; the key arrays exist only if fork "Multiplier history" picks matching |
| `_dat_anchor` | vec3 `[P]` | the certified anchor, shared with the self-contact rows (`solver_vbd.py:4095-4125`) |
| `_dat_material_hessian` | mat33 `[P]` | the particles' material Hessian for the support metric (`particle_dat_kernels.py:667`) |
| `truncation_ts`, `body_truncation_ts` | float `[P]`, float `[B]` | per-vertex and per-body fractions, atomic minimum over records (`solver_vbd.py:1442`, `solver_vbd.py:3632`) |
| `_rigid_dat_particle_max_displacement`, `_rigid_dat_body_max_displacement` | float, float `[B]` | the rigid-soft budget, half the minimum rigid-soft query radius (`solver_vbd.py:3645-3670`) |

### 1.6 Pseudo code

```
step(state_in, state_out, control, contacts, dt):                        # solver_vbd.py:3203
    _validate_dat_collision_schedules()               # RIGID and SOFT_SELF_CONTACT slots equal; the alm_dat refusal of rigid DAT is removed (solver_vbd.py:1218, 6074)
    _pre_initialization_detection(state_in, dt)       # rigid-soft records and self-contact rows; x_start and body_q_ref are set (solver_vbd.py:3261, 3636)
    _initialize_rigid_bodies(state_in, control, contacts, dt, update_rigid)   # forward step; bodies truncated in place against the stored planes (solver_vbd.py:4532, 4655)
    _initialize_particles(state_in, state_out, contacts, dt)                  # forward step; _alm_dat_initialize (solver_vbd.py:4127)
    rigid_due, soft_due = _collision_detection_due((PRE_POST_INIT,))          # solver_vbd.py:3285
    if _alm_dat and soft_due:
        _alm_dat_redetect(state_in, state_out, contacts, dt, rigid_due = rigid_due)   # both families re-detected at the anchor; planes and multipliers re-based (solver_vbd.py:3291, 4162)
    for k in range(iterations):
        _solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)    # bodies feel the row reactions; truncated in place per colour (solver_vbd.py:5205, 5489)
        _solve_particle_iteration(state_in, state_out, contacts, dt)              # particles feel both row families at the trial (solver_vbd.py:4759)
    _finalize_particles(state_out, dt)                # final rounds against both plane families; the anchor is committed (solver_vbd.py:5881)

_alm_dat_initialize(state_in, dt):                                        # solver_vbd.py:4127
    compute_particle_material_hessian(model, dt, x_start, particle_q_prev)        # D15 metric, once per step (particle_dat_kernels.py:667)
    anchor = x_start
    trial = _alm_dat_materialize_trial(state_in)
    build_contact_planes(anchor, trial, refresh_normals = True)                   # self-contact planes (particle_dat_kernels.py:369)
    build_rigid_soft_planes(anchor, trial, body_q_ref, body_q) (new)              # one plane per record: separator at the reference, offset for both sides' remaining motion (rigid_vbd_kernels.py:7916, 8001)
    match_contact_history()                                                       # self-contact multipliers (particle_dat_kernels.py:1426)
    match_rigid_soft_history() (new)                                              # rigid-soft multipliers by record key, else zero
    _alm_dat_truncate(trial, refresh_normals = True)

_alm_dat_truncate(trial, refresh_normals):                                # solver_vbd.py:4095
    sigma = truncate_against_contact_planes(anchor, trial, planes, gamma)                    # self-contact family (particle_dat_kernels.py:459)
    sigma = truncate_against_rigid_soft_planes(anchor, trial, rigid_soft_planes, gamma, sigma) (new)   # soft side of every record, atomic minimum into the same sigma
    anchor = commit_truncated_displacement(x_start, anchor, trial, sigma, budget)          # budget = min(self-contact, rigid-soft) (particle_dat_kernels.py:541, solver_vbd.py:3667)
    build_contact_planes(anchor, trial, refresh_normals)
    build_rigid_soft_planes(anchor, trial, body_q_ref, body_q)

_solve_particle_iteration(state_in, state_out, contacts, dt):            # solver_vbd.py:4759
    for color in particle_color_groups:
        accumulate_rigid_soft_alm_force_and_hessian(color, trial, body_q, body_q_prev) (new)          # replaces gather_particle_body_contact_force_and_hessian (solver_vbd.py:4837)
        accumulate_alm_contact_force_and_hessian(color, trial, particle_q_prev)                       # self rows (particle_dat_kernels.py:1324)
        _tiled_elasticity_kernel(color)                                                               # or solve_elasticity: the local solve writes displacement (solver_vbd.py:4959)
        trial = _alm_dat_materialize_trial(state_in)
    update_alm_contact_multipliers(trial)                                                             # once per sweep (particle_dat_kernels.py:1379)
    update_rigid_soft_multipliers(trial, body_q) (new)
    _alm_dat_iteration += 1
    if plane_update_interval > 0 and _alm_dat_iteration % plane_update_interval == 0:
        _alm_dat_truncate(trial, refresh_normals = True)

_solve_rigid_body_iteration(state_in, state_out, control, contacts, dt): # solver_vbd.py:5205; update_duals_body_particle_contacts (solver_vbd.py:5239) is not launched under alm
    for color in body_color_groups:
        accumulate_rigid_soft_alm_reaction_per_body(color, trial, body_q) (new)   # replaces accumulate_body_particle_contacts_per_body (solver_vbd.py:5308): -F n and torque at the record's body point, same multiplier and gap
        _solve_rigid_body_kernel(color)                                           # unchanged (solver_vbd.py:5405)
        _rigid_penetration_free_truncation(state_in, contacts)                    # bodies in place against the stored planes; curved trajectories (solver_vbd.py:3839, rigid_vbd_kernels.py:8134)

rigid_soft_row(record, trial, body_q, lam, lambda_t): (new)              # mirrors _alm_contact_response (particle_dat_kernels.py:1021)
    n, p, m = rigid_soft_planes[record]
    soft_point = barycentric point of the record's particles at the trial          # soft_contact_indices, soft_contact_barycentric (contacts.py:410, 448)
    rigid_support = support of the rigid primitive along n at body_q               # soft_contact_rigid_indices, else soft_contact_body_pos (contacts.py:416, 451)
    gap = dot(n, soft_point - rigid_support) - particle_radius - shape_margin
    rho = _series_scale(particle support along n, body Delassus scale along n)     # D15 and the rigid rule (particle_dat_kernels.py:566, rigid_vbd_kernels.py:1205, 1262)
    F, k = alm_contact_row_smooth(gap - hard_gap, lam, K, R - hard_gap, rho, band, wall_ratio)   # (particle_dat_kernels.py:781)
    f_t, J_t = alm_friction_row(lambda_t, rho_t, rel, n, mu * F)                               # rho_t from the inertial supports in series (particle_dat_kernels.py:820)
    return F, k, f_t, J_t
```

Names that exist today: `step` `solver_vbd.py:3203-3300`, `_collision_detection_due` `solver_vbd.py:6020-6027`, `_initialize_rigid_bodies` `solver_vbd.py:4532-4533`, `_initialize_particles` `solver_vbd.py:4038-4039`, `_pre_initialization_detection` `solver_vbd.py:6029-6030`, `_alm_dat_initialize` `solver_vbd.py:4127-4160`, `_alm_dat_truncate` `solver_vbd.py:4095-4125`, `_alm_dat_redetect` `solver_vbd.py:4162-4205`, `_solve_particle_iteration` `solver_vbd.py:4759-4760`, `_solve_rigid_body_iteration` `solver_vbd.py:5205-5210`, `_rigid_penetration_free_truncation` `solver_vbd.py:3839-3840`, `_finalize_particles` `solver_vbd.py:5881-5882`, `compute_particle_material_hessian` `particle_dat_kernels.py:667-668`, `build_contact_planes` `particle_dat_kernels.py:369-370`, `truncate_against_contact_planes` `particle_dat_kernels.py:459-460`, `commit_truncated_displacement` `particle_dat_kernels.py:541-542`, `accumulate_alm_contact_force_and_hessian` `particle_dat_kernels.py:1324-1325`, `update_alm_contact_multipliers` `particle_dat_kernels.py:1379-1380`, `match_contact_history` `particle_dat_kernels.py:1426-1427`, `alm_contact_row_smooth` `particle_dat_kernels.py:781-782`, `alm_friction_row` `particle_dat_kernels.py:820-821`, `_series_scale` `rigid_vbd_kernels.py:1262-1263`, `_contact_delassus_w` `rigid_vbd_kernels.py:1205-1206`, `find_primitive_pair_separator` `rigid_vbd_kernels.py:7916-7917`, `place_dat_division_plane` `rigid_vbd_kernels.py:8001-8002`, `rigid_trajectory_truncation_t` `rigid_vbd_kernels.py:8134-8135`. New names: `build_rigid_soft_planes`, `truncate_against_rigid_soft_planes`, `accumulate_rigid_soft_alm_force_and_hessian`, `accumulate_rigid_soft_alm_reaction_per_body`, `update_rigid_soft_multipliers`, `match_rigid_soft_history`, `rigid_soft_row`, `RigidSoftDatPlanes`, `RigidSoftAlmState`, `rigid_soft_contact_method`.

### 1.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| Body side | none in this chapter | bodies truncated in place after every body colour, as the rigid-soft DAT does today (`solver_vbd.py:5489`); or a body anchor and trial pair like the particles, committed once per sweep | none; the in-place form first |
| Method switch | `rigid_soft_contact_method` (new) | separate switch, default `"penalty"`, as the host keeps one flag per contact family (`rigid_compliant_alm` `solver_vbd.py:510`, `particle_self_contact_method` `solver_vbd.py:499`); or implied by `particle_self_contact_method="alm_dat"` whenever rigid-soft DAT is on | none |
| Multiplier history | none | re-seeded at every detection as the penalty is today (`solver_vbd.py:4440`); matched by the record key `(soft indices, shape, rigid indices)` like the sorted self-contact rows (`particle_dat_kernels.py:1426`); or the body-body contact history mechanism (`rigid_contact_history`, `solver_vbd.py:520`) | garments on grippers of the collision-fidelity benchmark |
| Rigid support in the gap | none | the rigid primitive's support along `n` for mesh rows (#4936, `contacts.py:416`); the stored body point for analytic and recovery records; or the stored point for every record | none |
| Friction metric | none | inertial supports in series (the D15 rule for the soft rows); or the body's tangent Delassus scale alone | incline and towel cases |
| Masked rows | none | plane kept, no row (proposed); or no plane either | none |

### 1.8 Open questions for the reviewer

1. Body side: in place as today, or a body anchor and trial pair (fork "Body side")?
2. One switch for both families, or the separate `rigid_soft_contact_method` (fork "Method switch")?
3. Should rigid-soft multipliers persist across detections, and by which key (fork "Multiplier history")?
4. World and kinematic shapes have Delassus zero: rho from the particle support alone, as `_series_scale` returns for an absent side, acceptable?
5. Masked rows keep their plane and carry no row: is that the intended reading of the force mask?
