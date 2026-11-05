# Suture simulation in SolverVBD: design

- Purpose: one chapter per component of the needle-and-thread suturing feature of `SolverVBD`, written before the code: the tunnel point row (chapter 1), the needle state machine with the contact gate (chapter 2), and the authoring data with the demo (chapter 3).
- Inputs: `AnkaChan/newton` branch `ankac/vbd-suture` at `84242814c` (the `upstream/main` of 2026-11-05 that the study read); the study `inputs/prior-work.md`; the options `inputs/options.md`; the language `inputs/CONTEXT.md`; the decisions `decisions.md` with ADR 0001 and ADR 0002 in `inputs/`.
- Pseudo code convention: structured statements with real function names. Names that exist today are written as they are and cited; names that do not exist yet carry (new) the first time. Only the body of a function and the body of a control statement (if, elif, else, for, while) are indented. Comments stand on their own line or after #.
- Terms: `inputs/CONTEXT.md` defines them. A tunnel point is one barycentric sample of a tunnel; its row is a hard bilateral constraint in the two directions transverse to the rod tangent with capped ALM friction along the tangent; the rod curve is the needle polyline followed by the thread polyline, one segment per rigid body; the needle group is the set of needle and thread shapes that share one gate bit. `N` is the suture needle count, `S` the rod segment count, `T` the tunnel capacity per needle, `Q` the point capacity per tunnel, `P` the particle count, `B` the body count.
- Status: chapters 1 to 3 drafted 2026-11-05 from the interview decisions D1 to D13. Per R4 the implementation starts from these chapters; review comments are applied to the code and the chapters together.

## 1. The tunnel point row

### 1.1 Intent

The host pattern is the particle-body attachment. Once per step `step_particle_body_attachment_lambda_rho` sets the ALM metric of every attachment from the two endpoints' support and retains the multiplier (`particle_vbd_kernels.py:1964-2068`). In every particle color `accumulate_body_particle_attachment_force_and_hessian` adds the row's force and 3x3 Hessian to the colored particle (`particle_vbd_kernels.py:2070-2113`), computed by `evaluate_body_particle_attachment_particle_force_hessian`: residual `C = x_particle - anchor`, force `-(k_eff C + lambda_eff)`, Hessian `k_eff I`, where infinite stiffness gives `k_eff = rho` and `lambda_eff = lambda` (`rigid_vbd_kernels.py:1734-1764`, `rigid_vbd_kernels.py:160-173`). In every body color `accumulate_body_particle_attachments_per_body` adds the opposite force as a wrench with the three 6x6 blocks (`rigid_vbd_kernels.py:5866-5943`, `rigid_vbd_kernels.py:1637`). After both passes `update_duals_particle_body_attachments` advances the multiplier by the relaxed ascent (`rigid_vbd_kernels.py:1767-1786`, launched at `solver_vbd.py:3006`). The second host pattern is the friction of body-body contact: the tangential force `rho_t (slip + c0) + lambda_t` projected onto the Coulomb disk of radius `mu f_n`, with Hessian `rho_t` on the tangent plane inside the cone and a conservative slip metric outside it (`rigid_vbd_kernels.py:2033-2057`), and the dual step that projects `lambda_t + rho_t slip` onto the disk of radius `mu lambda_n` (`rigid_vbd_kernels.py:1121-1147`).

The tunnel point row is that attachment row with four changes. The soft endpoint is a barycentric point of one tet, so the force and Hessian scatter to four vertices with weights `w_i` and `w_i^2`. The rigid endpoint is the closest place on the rod curve, found again every iteration and fixed within it. The residual keeps only the two components transverse to the rod tangent. Along the tangent the row applies the capped ALM friction, with the transverse row force in the role of `f_n`, the transverse multiplier in the role of `lambda_n`, and the cap raised by the grip force per unit length times the point's spacing (D3).

### 1.2 Old and new

| | Attachment today (file:line) | Tunnel point row |
|---|---|---|
| Soft endpoint | one particle (`model.py:1284`) | one tet and four barycentric weights, fixed at creation (chapter 2) |
| Rigid endpoint | body-local point (`model.py:1286`) | the closest place on the rod curve: a segment and a parameter `u`, refreshed every iteration from the cached place |
| Residual | three components `x - anchor` (`rigid_vbd_kernels.py:1753`) | `P (x - p)` with `P = I - t t^T`, two effective components; `t` is the segment tangent, frozen within the iteration |
| Stiffness | per attachment, `inf` is the hard row (`builder.py:11264-11265`) | per point, `inf` by default; a soft wall is a value change (D2) |
| Friction | none | capped ALM friction along `t`; cap `mu (|f_perp| + g ds)` (D3) |
| Multipliers | one vec3, retained by `rigid_avbd_gamma` (`solver_vbd.py:471`, `particle_vbd_kernels.py:2009`) | one vec3 in the normal plane plus one scalar along `t`, both retained the same way |
| Metric `rho` | support of one particle, reduced by its material Hessian, plus the body (`particle_vbd_kernels.py:2015-2066`) | support of four weighted vertices plus the body's Delassus term in the normal plane; `rho_t` by `_contact_auto_tangent_rho` (`rigid_vbd_kernels.py:233-239`) |
| Where forces land | `particle_forces`, `particle_hessians` (`particle_vbd_kernels.py:2111-2112`); body accumulators (`rigid_vbd_kernels.py:5938-5942`) | the same arrays, same kernels' position in the color loops (`solver_vbd.py:4280`, `solver_vbd.py:4740-4745`) |
| Coloring | none needed (`solver_vbd.py:4280-4527`) | none needed: the four vertices of a tet already have distinct colors (`graph_coloring.py:202`) |

### 1.3 Argument list

```
SolverVBD(model, *,                                            # keyword only (solver_vbd.py:439)
    suture_tunnel_update_interval: int = 1,                    # substeps between needle-state updates; 1 = every substep (D8, chapter 2)
    suture_point_spacing: float | None = None,                 # ds_tunnel [m]; None = clamp(0.5 x mean tet edge length, 0.5 mm, 2 mm) (D8)
    suture_gate_slack: float = 1.0e-3,                         # added to r_gate [m]: needle radius + shape margin + soft_contact_gap + particle radius + slack (D7)
    suture_armed: bool = False,                                # True: the first tip contact punctures; False: force and angle test (D6)
    suture_closest_search_window: int = 1,                     # rod segments searched on each side of the cached segment per refresh (D14)
)
```

Row material comes from the model, not from arguments: the per-point stiffness array is filled with `inf` at creation; the tissue parameters are the per-particle custom attributes `vbd.suture_grip` [N/m], `vbd.suture_mu`, `vbd.suture_puncture_force` [N] and `vbd.suture_puncture_cos` of chapter 3, read at a point as the barycentric average over its tet's vertices. The retention factor is the existing `rigid_avbd_gamma` (`solver_vbd.py:471`), the friction smoothing length the existing `friction_epsilon` times `dt` (`rigid_vbd_kernels.py:2052`).

Not arguments: `rho` and `rho_t` (automatic from the supports, as every compliant row), the tangent (from the rod segment), the gate masks (chapter 2), the point capacities (authored per needle, chapter 3).

### 1.4 Input types

The row consumes the rod curve tables of chapter 3 and the tunnel point tables below. No new Python type is introduced; the rod curve is four model arrays.

| Member | dtype, shape | Attachment counterpart | Meaning |
|---|---|---|---|
| `suture_segment_body` | int32 `[S]` | `attachment_particle_body_body` (`model.py:1282`) | body that moves the segment; all needle segments name the needle body |
| `suture_segment_local_a` | vec3 `[S]` | `attachment_particle_body_body_point` (`model.py:1286`) | segment start in the body frame [m], toward the tip |
| `suture_segment_local_b` | vec3 `[S]` | same | segment end in the body frame [m], toward the free end |
| `suture_needle_segment_start` | int32 `[N + 1]` | none | segment range of needle `n`: tip first, then the thread from the swage to the free end |

### 1.5 Data structures on the solver

Capacity per needle is `T x Q` points; the flat point index is `(n T + t) Q + q`. Every array below has length `N T Q` unless stated.

| Attribute | dtype, shape | Meaning |
|---|---|---|
| `_suture_point_tet` | int32 `[N T Q]` | tet that holds the point; -1 when the slot is free |
| `_suture_point_bary` | vec4 `[N T Q]` | barycentric weights in that tet, fixed at creation |
| `_suture_point_active` | int32 `[N T Q]` | 1 while some part of the rod passes through the point (rule 5) |
| `_suture_point_segment` | int32 `[N T Q]` | flat rod segment of the closest place, this iteration |
| `_suture_point_u` | float `[N T Q]` | parameter of the closest place on that segment, in `[0, 1]` |
| `_suture_point_tangent` | vec3 `[N T Q]` | world unit tangent of that segment, from `a` to `b`; frozen within the iteration |
| `_suture_point_segment_start`, `_suture_point_u_start` | int32, float `[N T Q]` | the rod place at the step start; the friction slip is measured against this material point, as the surface anchors of body-body contact (`rigid_vbd_kernels.py:2009-2012`) |
| `_suture_point_lambda` | vec3 `[N T Q]` | transverse multiplier, world frame, kept in the normal plane |
| `_suture_point_lambda_t` | float `[N T Q]` | tangential multiplier along `tangent` |
| `_suture_point_rho`, `_suture_point_rho_t` | float `[N T Q]` | transverse and tangential ALM metrics, set once per step |
| `_suture_point_ds` | float `[N T Q]` | tunnel length the point represents [m]: half the gap to each neighbor, one-sided at the ends |
| `_suture_point_stiffness` | float `[N T Q]` | row stiffness; `inf` by default (D2) |

Conventions reused from the host: the particle inverse mass of an inactive or zero-mass vertex counts as zero support (`particle_vbd_kernels.py:2012-2013`); the body side of a kinematic body has zero inverse mass and receives no force (`rigid_vbd_kernels.py:5893-5894`); multipliers of a point whose row is off are zero (`particle_vbd_kernels.py:2005-2008`).

### 1.6 Pseudo code

```
step(state_in, state_out, control, contacts, dt):                        # solver_vbd.py:2898
    _initialize_rigid_bodies(...); _initialize_particles(...)            # solver_vbd.py:2975-2976, unchanged
    _initialize_particle_body_attachments(state_in, dt)                  # solver_vbd.py:2977
    _initialize_suture_points(state_in, dt) (new)                        # once per step: retention, rho, step-start place
    for iter_num in range(iterations):                                   # solver_vbd.py:2993
        refresh_suture_points(state_in) (new)                            # closest place and tangent for this iteration
        _solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)   # solver_vbd.py:3002; per body color: accumulate_suture_rows_per_body (new) next to accumulate_body_particle_attachments_per_body (solver_vbd.py:4745)
        _solve_particle_iteration(state_in, state_out, contacts, dt)     # solver_vbd.py:3003; per particle color: accumulate_suture_row_force_and_hessian (new) next to the attachment kernel (solver_vbd.py:4313-4315)
        update_duals_suture_points(state_in) (new)                       # next to update_duals_particle_body_attachments (solver_vbd.py:3006)
    _update_suture_needles(state_in, contacts, dt) (new)                 # chapter 2, every suture_tunnel_update_interval substeps

_initialize_suture_points(state_in, dt):                                 # new; one thread per point
    for point in all points:
        if point is free or not active:
            lambda = 0; lambda_t = 0; continue
        lambda = rigid_avbd_gamma * lambda; lambda_t = rigid_avbd_gamma * lambda_t          # retention as particle_vbd_kernels.py:2009
        segment_start = segment; u_start = u                                                  # the rod material point the slip is measured from
        w_soft = sum over the four vertices of bary_i^2 x (inverse mass reduced by the material Hessian)   # particle_vbd_kernels.py:2015-2066 per vertex (D15)
        n1, n2 = orthonormal_basis(tangent)                                                   # math/__init__.py:242
        w_body = max(_contact_delassus_w(body, place, n1), _contact_delassus_w(body, place, n2))   # rigid_vbd_kernels.py:1205
        rho = _attachment_auto_rho(1 / (dt^2 (w_soft + w_body)), stiffness)                   # rigid_vbd_kernels.py:214
        w_body_t = _contact_delassus_w(body, place, tangent)
        rho_t = _contact_auto_tangent_rho(1 / (dt^2 (w_soft + w_body_t)), rho, body_structural_k[body])   # rigid_vbd_kernels.py:233

refresh_suture_points(state):                                            # new; one thread per point
    for point in active points:
        x = sum of bary_i x particle_q[tet vertex i]
        best = none
        for segment in [cached segment - window, cached segment + window] within the needle's range:
            place, u = closest point of x on the world segment (a, b)
            if |x - place| < best distance:
                best = segment, u
        if best segment is the last segment of the needle and u >= 1:
            active = 0; continue                                         # rule 5: the free end passed the point
        segment, u = best
        tangent = normalize(b - a) of that segment in world

_evaluate_suture_row(point, particle_q, particle_q_prev, body_q, body_q_prev, dt):   # new; shared by both sides
    x = barycentric point at particle_q; x_prev = barycentric point at particle_q_prev
    p = transform_point(body_q[body], lerp(local_a, local_b, u)); t = tangent
    P = I - t t^T
    C = P (x - p)
    k_eff, lambda_eff = _material_force_terms(rho, stiffness, lambda, 1)           # rigid_vbd_kernels.py:160; inf gives (rho, lambda)
    f_perp = k_eff * C + lambda_eff; H_perp = k_eff * P                              # on the rod; the tissue gets the opposite
    p0 = place (segment_start, u_start) at body_q; p0_prev = same place at body_q_prev
    slip = dot(t, (x - x_prev) - (p0 - p0_prev))                                     # tissue minus rod along the tangent since the step start
    cap = mu * (|f_perp| + grip * ds)                                                # D3: transverse force plus grip
    f_t = rho_t * slip + lambda_t
    if |f_t| > cap:
        H_t = cap * _regularized_coulomb_scale(|slip|, friction_epsilon * dt) * t t^T   # rigid_vbd_kernels.py:1107; sliding metric as 2052-2054
        f_t = cap * sign(f_t)                                                            # _project_coulomb_tangent in one dimension (rigid_vbd_kernels.py:1097)
    else:
        H_t = rho_t * t t^T
    return f_perp + f_t * t, H_perp + H_t

accumulate_suture_row_force_and_hessian(color, ...):                     # new; one thread per point; particle side
    for point in active points:
        f, H = _evaluate_suture_row(...)
        for i in range(4):
            v = tet vertex i
            if particle_colors[v] == color:
                particle_forces[v] += -bary_i * f                        # atomic, as particle_vbd_kernels.py:2111
                particle_hessians[v] += bary_i^2 * H                     # atomic

accumulate_suture_rows_per_body(color_group, ...):                       # new; one thread per body of the color
    for body in color_group:
        if body_inv_mass[body] <= 0: continue                            # kinematic bodies take no force (rigid_vbd_kernels.py:5893)
        for point in active points of every needle:
            if suture_segment_body[segment] != body: continue
            f, H = _evaluate_suture_row(...)
            torque, H_al, H_aa = _point_force_to_body_torque_and_hessian(p - com, f, H)   # rigid_vbd_kernels.py:1637
            body_forces[body] += f; body_torques[body] += torque; body_hessian_ll += H; body_hessian_al += H_al; body_hessian_aa += H_aa

update_duals_suture_points(...):                                         # new; one thread per point, after both passes
    for point in active points:
        C, slip computed as in _evaluate_suture_row at the current positions
        lambda = P * _alm_relaxed_ascent(lambda, C, stiffness, rho)      # rigid_vbd_kernels.py:145; bilateral, kept in the normal plane
        cap = mu * (|lambda| + grip * ds)
        lambda_t = clamp(lambda_t + rho_t * slip, -cap, cap)             # the one-dimensional form of _compliant_contact_dual_step (rigid_vbd_kernels.py:1121)
```

Names that exist today: `step` `solver_vbd.py:2898-3049`, `_initialize_particle_body_attachments` `solver_vbd.py:2732-2769`, `_solve_rigid_body_iteration` `solver_vbd.py:4676-5081`, `_solve_particle_iteration` `solver_vbd.py:4280-4525`, `orthonormal_basis` `math/__init__.py:242`, `_contact_delassus_w` `rigid_vbd_kernels.py:1205-1228`, `_attachment_auto_rho` `rigid_vbd_kernels.py:214-218`, `_contact_auto_tangent_rho` `rigid_vbd_kernels.py:233-239`, `_material_force_terms` `rigid_vbd_kernels.py:160-173`, `_regularized_coulomb_scale` `rigid_vbd_kernels.py:1107-1118`, `_project_coulomb_tangent` `rigid_vbd_kernels.py:1097-1104`, `_point_force_to_body_torque_and_hessian` `rigid_vbd_kernels.py:1637-1650`, `_alm_relaxed_ascent` `rigid_vbd_kernels.py:145-157`, `body_structural_k` `solver_vbd.py:2417`. New names: `_initialize_suture_points`, `refresh_suture_points`, `_evaluate_suture_row`, `accumulate_suture_row_force_and_hessian`, `accumulate_suture_rows_per_body`, `update_duals_suture_points`, `_update_suture_needles`, all in a new file `newton/_src/solvers/vbd/suture_kernels.py` except the solver methods.

### 1.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| Row form, closed 2026-11-05 (D2, Anka, chat) | per-point `stiffness`, `inf` default | hard bilateral line (chosen); finite stiffness = soft wall by value; unilateral tube deferred to a later `suture_tube_radius` | none |
| Friction law, closed 2026-11-05 (D3, Anka, chat) | none | the capped ALM contact law, one tangent direction (chosen); the IPC smooth law of soft contacts | none |
| Grip in the cap, closed 2026-11-05 (D3) | `vbd.suture_grip`, default 1 N/m | `g > 0`: hold at zero side load; `g = 0`: the unchanged contact law | demo: thread drift with `g = 0` against `g = 1 N/m` |
| Closest-place search window | `suture_closest_search_window` | 1: cheapest, assumes the rod moves under one segment per iteration; 2 or more: robust to faster sliding at more distance tests | demo at high pull speed: count of points that lose their place with window 1 |
| Transverse `rho` particle support | none | four material-reduced inverse masses (as attachments); plain inverse masses (cheaper, larger `rho`) | row convergence test: iterations to a 0.1 mm residual |
| Grip reference | none | `g` per unit length (chosen for the first milestone); a pressure times the local rod radius, so the needle feels more grip than the thread | none |
| Tangent within an iteration | none | frozen tangent (chosen, as the constant normal of `compute_friction`, `particle_vbd_kernels.py:1375-1377`); re-evaluated per color | none |

### 1.8 Open questions for the reviewer

1. Search window: is one segment on each side enough, given that the thread slides under one segment per iteration at demo speeds (fork "Closest-place search window", D14)?
2. Should the particle support of `rho` use the material-reduced inverse masses as attachments do, or plain inverse masses (fork "Transverse `rho` particle support", D15)?
3. Should the grip be authored per unit length or as a pressure times the local rod radius (fork "Grip reference", D16)?
4. Should tunnel point multipliers decay with `rigid_avbd_gamma` like attachments, or persist undecayed because the rows are long-lived?

## 2. Needle state, puncture, tunnel points and the contact gate

### 2.1 Intent

The host patterns are four. The contact force export evaluates one wrench per rigid-soft record at the final configuration with `compute_body_particle_contact_forces` (`rigid_vbd_kernels.py:6140`, launched by `_export_body_particle_contact_forces`, `solver_vbd.py:5151-5199`); the tip force test reuses the same per-record normal force. The soft-contact record filter `_soft_contact_force_enabled` (`soft_contacts_mesh.py:1593-1636`) with its bound data `_SoftContactFilterData` (`soft_contacts_mesh.py:1522-1547`, bound by `_make_soft_contact_filter_data`, `soft_contacts_mesh.py:1549-1591`) is called by every consumer of rigid-soft records: the particle gather and scatter kernels (`particle_vbd_kernels.py:2526`, `particle_vbd_kernels.py:2655`, `particle_vbd_kernels.py:2691`), the per-body accumulation (`rigid_vbd_kernels.py:6047`), the force export (`rigid_vbd_kernels.py:6182`), the body-particle dual update (`rigid_vbd_kernels.py:7359`) and the coupling harvest (`vbd_coupling_kernels.py:208`, `vbd_coupling_kernels.py:366`); only the rigid-soft truncation keeps raw records on purpose (`rigid_vbd_kernels.py:8252-8277`). The vertex-to-tet adjacency `v_adj_tets` of `MeshAdjacencyData` gives the tets around a particle (`mesh.py:297-298`, `mesh.py:335-343`). Kinematic bodies are caller-moved and skipped by the rigid pass (`rigid_vbd_kernels.py:4103-4106`).

The needle state machine runs on the device once every `suture_tunnel_update_interval` substeps, after the last iteration (D8). It sums the tip shape's contact normal force over the stored records, decides the puncture (D6), walks the tet mesh from the tip's last tet to the tet that contains the tip now, appends or removes tunnel points by the spacing rule, places the exit point when the walk leaves through a boundary face, and finishes a tunnel when no point holds rod anymore (rules 1 to 6). A per-particle kernel then recomputes the gate bits of every needle group from the distance to the tunnel polylines (rule 7), and the record filter plus the truncation kernel ignore every record whose particle and shape share a bit (D7, ADR 0002).

### 2.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Record filter | live particle activity, resolved sign, feature-region test (`soft_contacts_mesh.py:1593-1636`) | the same, preceded by the gate test on the record's corners and shape |
| Rigid-soft truncation | every stored record builds a plane (`rigid_vbd_kernels.py:8252-8277`) | gated records build no plane |
| Needle tip against tissue | penalty contact plus DAT (`geometry/kernels.py:1184`, `rigid_vbd_kernels.py:2177`) | contact until the puncture; afterwards the tip's tet is gated and the tunnel points hold the needle |
| Tet containment of a point | none | a tet walk over face neighbors, one thread per needle |
| Tunnel data | none | per needle `T` tunnels of `Q` points, fixed capacity, created on the device |
| Gate | none; `ParticleFlags` has ACTIVE and PROXY only (`flags.py:8-22`), `COLLIDE_PARTICLES` is per shape (`flags.py:36`) | `particle_contact_gate` and `shape_contact_gate` bit masks, one bit per needle |

### 2.3 Argument list

The constructor arguments are those of 1.3 (`suture_tunnel_update_interval`, `suture_point_spacing`, `suture_gate_slack`, `suture_armed`). The filter data gains two members:

```
_SoftContactFilterData (soft_contacts_mesh.py:1522)
    particle_contact_gate: wp.array[int32]        # [P], bit per needle group; empty array = no gate (new)
    shape_contact_gate: wp.array[int32]           # [shape_count], bit per needle group; empty array = no gate (new)
```

Not arguments: the contact normal convention (the SDF gradient of the shape at the particle, pointing from the shape toward the particle, `geometry/kernels.py:1289-1291`); the containment tolerance of the tet walk, a constant `1e-6` in barycentric units; the walk step limit, a constant 32.

### 2.4 Input types

| Member of `Contacts` | dtype, shape | Filled by | Meaning |
|---|---|---|---|
| `soft_contact_count` | int32 `[1]` | detection (`contacts.py:397`) | stored record count |
| `soft_contact_indices` | vec3i `[soft_contact_max]` | detection (`contacts.py:407`) | -1 padded soft corners: particle, edge or face record |
| `soft_contact_barycentric` | vec3 `[soft_contact_max]` | detection (`contacts.py:445`) | weights on the corners; `(1, 0, 0)` for a particle |
| `soft_contact_shape` | int32 `[soft_contact_max]` | detection (`contacts.py:447`) | the shape of the record; the tip shape identifies tip records |
| `soft_contact_body_pos`, `soft_contact_normal` | vec3 `[soft_contact_max]` | detection (`contacts.py:448`, `contacts.py:455`) | surface point in the body frame and the normal |
| `soft_contact_max` | int | pipeline (`contacts.py:526`) | capacity; the tip force kernel strides over it |

### 2.5 Data structures on the solver and the model

| Attribute | dtype, shape | Meaning |
|---|---|---|
| `_suture_needle_state` | int32 `[N]` | 0 OUTSIDE, 1 PIERCING, 2 THROUGH |
| `_suture_needle_tunnel` | int32 `[N]` | index `t` of the current tunnel, -1 when OUTSIDE |
| `_suture_tunnel_count` | int32 `[N]` | tunnels opened so far; at most `T` |
| `_suture_tunnel_state` | int32 `[N T]` | 0 unused, 1 PIERCING, 2 THROUGH, 3 FINISHED |
| `_suture_tunnel_point_count` | int32 `[N T]` | points of the tunnel |
| `_suture_needle_tip_tet` | int32 `[N]` | tet that held the tip at the last update |
| `_suture_needle_last_tip` | vec3 `[N]` | tip position at the last update [m] |
| `_suture_needle_travel` | float `[N]` | tip travel inside the tissue since the last point [m] |
| `_suture_needle_tip_force` | float `[N]` | sum of tip record normal forces of the substep [N]; atomic |
| `_suture_needle_tip_record` | int32 `[N]` | record index of the largest tip normal force; its corner particle starts the walk, its normal is the surface normal |
| `_suture_substep` | int32 `[1]` | substep counter for the update interval; graph-safe because it lives on the device |
| `_suture_tet_neighbors` | int32 `[tet_count, 4]` | neighbor tet across the face opposite vertex `i`; -1 on the boundary; built once at construction from `tet_indices` (`model.py:1331`) |
| `particle_contact_gate` (model) | int32 `[P]` | bit per needle group; recomputed every update |
| `shape_contact_gate` (model) | int32 `[shape_count]` | bit per needle group; set by `add_suture_needle` (chapter 3) |

Conventions reused: the tet of a particle is reached through `v_adj_tets` and `v_adj_tets_offsets` (`mesh.py:297-298`); a kinematic body's pose is read, never written (`rigid_vbd_kernels.py:4103-4106`); the tissue parameters of a point are the barycentric average of the per-particle attributes of its tet (chapter 3).

### 2.6 Pseudo code

```
_update_suture_needles(state, contacts, dt):                             # new; at the end of the last iteration
    _suture_substep += 1
    if _suture_substep % suture_tunnel_update_interval != 0: return
    accumulate_suture_tip_force(contacts, state)
    update_suture_needle_state(state)
    update_suture_gate(state)

accumulate_suture_tip_force(contacts, state):                            # new; one thread per record
    for record in range(min(soft_contact_count, soft_contact_max)):
        n = needle whose tip shape is soft_contact_shape[record]; if none: continue
        if not _soft_contact_force_enabled(record, filter): continue     # soft_contacts_mesh.py:1593; gated records do not count
        f, H = _eval_body_particle_contact(...)                           # rigid_vbd_kernels.py:2177, the particle branch; the edge and face branch uses _eval_soft_ef_contact (2259)
        f_n = max(dot(f, soft_contact_normal[record]), 0)
        _suture_needle_tip_force[n] += f_n                               # atomic
        if f_n is the largest seen for n: _suture_needle_tip_record[n] = record   # atomic max on the force, then the record

update_suture_needle_state(state):                                       # new; one thread per needle
    for n in needles:
        tip = transform_point(body_q[suture_needle_body[n]], suture_needle_tip_local[n])
        t = _suture_needle_tunnel[n]
        if _suture_needle_state[n] == OUTSIDE:
            record = _suture_needle_tip_record[n]
            if suture_armed: punctured = _suture_needle_tip_force[n] > 0
            else:
                particle = first corner of soft_contact_indices[record]
                punctured = _suture_needle_tip_force[n] >= suture_puncture_force[particle] and dot(tip_direction, soft_contact_normal[record]) >= suture_puncture_cos[particle]
            if punctured and _suture_tunnel_count[n] < T:
                tet = among v_adj_tets of particle, the tet with the largest smallest barycentric weight of tip   # mesh.py:335-343
                t = _suture_tunnel_count[n]; _suture_tunnel_count[n] += 1; _suture_tunnel_state[n, t] = PIERCING
                append point (tet, clamped bary of tip, ds = 0); stiffness = inf             # rule 1: entry point
                _suture_needle_state[n] = PIERCING; _suture_needle_tip_tet[n] = tet; _suture_needle_travel[n] = 0
        elif _suture_needle_state[n] == PIERCING:
            tet = _suture_needle_tip_tet[n]; bary = barycentric(tip, tet); steps = 0
            while min(bary) < -eps and steps < 32:
                face = index of the smallest bary; neighbor = _suture_tet_neighbors[tet, face]
                if neighbor < 0:
                    append point (tet, clamped bary); _suture_tunnel_state[n, t] = THROUGH; _suture_needle_state[n] = THROUGH   # rule 4: exit point
                    break
                tet = neighbor; bary = barycentric(tip, tet); steps += 1
            if _suture_needle_state[n] == PIERCING:
                _suture_needle_tip_tet[n] = tet
                _suture_needle_travel[n] += |tip - _suture_needle_last_tip[n]|
                if _suture_needle_travel[n] >= ds_tunnel:
                    append point (tet, bary); _suture_needle_travel[n] = 0                              # rule 2
                last = last point of tunnel t
                if last is not the entry point and its place is the tip end (segment 0, u = 0) and |x_last - tip| > ds_tunnel:
                    remove last                                                                          # rule 3
            recompute ds of the last two points                                                          # half gaps
        elif _suture_needle_state[n] == THROUGH:
            if no point of tunnel t is active:
                _suture_tunnel_state[n, t] = FINISHED; _suture_needle_state[n] = OUTSIDE; _suture_needle_tunnel[n] = -1   # rule 6
        _suture_needle_last_tip[n] = tip; _suture_needle_tip_force[n] = 0; _suture_needle_tip_record[n] = -1

update_suture_gate(state):                                               # new; one thread per particle
    for p in particles:
        bits = 0
        for n in needles:
            d = distance from particle_q[p] to the polyline of every tunnel of n in state PIERCING or THROUGH, plus the segment last point to tip while PIERCING
            if d < r_gate[n]: bits |= suture_needle_group[n]
        particle_contact_gate[p] = bits                                  # rule 7: from scratch every update

_soft_contact_force_enabled(record, data):                               # soft_contacts_mesh.py:1593; new first statements
    if data.particle_contact_gate has entries:
        shape_bits = data.shape_contact_gate[data.contact_shapes[record]]
        for k in range(3):
            corner = data.soft_contact_indices[record][k]
            if corner >= 0 and (data.particle_contact_gate[corner] & shape_bits) != 0: return False
    ... existing tests ...

apply_rigid_soft_truncation(...):                                        # rigid_vbd_kernels.py:8252; the same gate test before the plane is built
```

Names that exist today: `_eval_body_particle_contact` `rigid_vbd_kernels.py:2177-2256`, `_eval_soft_ef_contact` `rigid_vbd_kernels.py:2259-2366`, `_soft_contact_force_enabled` `soft_contacts_mesh.py:1593-1636`, `apply_rigid_soft_truncation` `rigid_vbd_kernels.py:8252-8500`, `get_vertex_num_adjacent_tets` `mesh.py:335-338`, `get_vertex_adjacent_tet_id_order` `mesh.py:340-343`. New names: `_update_suture_needles`, `accumulate_suture_tip_force`, `update_suture_needle_state`, `update_suture_gate`, the two gate members of `_SoftContactFilterData`.

### 2.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| Gate placement, closed 2026-11-05 (D7, Anka, chat) | none | at consumption in the record filter plus the truncation kernel (chosen, ADR 0002); at detection inside `create_soft_contacts` | none |
| Gate set, closed 2026-11-05 (D7) | `suture_gate_slack` | distance to the tunnel polyline (chosen); vertices of tets with a point plus one ring | none |
| Puncture trigger, closed 2026-11-05 (D6) | `suture_armed` | force and angle test (chosen default); armed first contact | demo test mode uses the force test; unit tests use both |
| Update cadence, closed 2026-11-05 (D8) | `suture_tunnel_update_interval` | 1: every substep (default); larger: cheaper, gate slack must cover the tip travel between updates | demo: interval 1 against 4 at the same holder speed |
| Walk start | none | the tets around the strongest tip record's particle (chosen); a global search over all tets | none |
| Tip force source | none | re-evaluate the tip records at the end of the substep (chosen); accumulate during the per-body contact pass | none |
| Gate of a finished tunnel | none | gate off at once (chosen); keep the gate for a few updates | none |

### 2.8 Open questions for the reviewer

1. With full-surface rigid-soft contact the tip record can be an edge or face record; starting the walk from its first corner is the proposed rule. Is that acceptable, or should the walk start from the corner with the largest barycentric weight?
2. Containment tolerance `1e-6` and walk limit 32: fine as constants, or constructor arguments?
3. When the tip retreats out of the entry face before any exit, should the tunnel close (rule 3 removes every point down to the entry point; the proposal then keeps the entry point and the PIERCING state until the tip re-enters or the needle leaves the gate radius, at which point the tunnel is FINISHED)?

## 3. Authoring, model data and the demo

### 3.1 Intent

The host pattern for the builder method is `add_attachment_particle_body` (`builder.py:11237-11320`): validate the indices and the coefficients, append to the builder lists (`builder.py:2071-2085`), convert them to model arrays in `finalize` (`builder.py:15918-15924`), expose them as model fields (`model.py:1282-1294`) with a count (`model.py:1814`). The thread comes from `add_rod` (`builder.py:10481`) over a `Rod` (`rod.py:112`): one capsule body per segment with the body origin at the segment midpoint and the centerline endpoints at local `(0, 0, -half_height)` and `(0, 0, half_height)`, joined by ROD joints (`builder.py:7006`). The holder is a kinematic body joined to the needle by `add_joint_d6` (`builder.py:6952`) with a `JointDofConfig` per axis carrying `target_ke` and `target_kd` (`builder.py:1314`); VBD treats drives as bilateral rows on the free axes (`vbd.rst:534-536`). Solver-specific per-entity parameters are declared by `SolverVBD.register_custom_attributes` in the `vbd` namespace (`solver_vbd.py:2531-2616`, `builder.py:2646`).

`add_suture_needle` records one suture needle the same way: it validates the needle body, the tip shape and the thread bodies; it builds the rod curve as a segment list, tip first; it sets the gate bit of the needle group on every needle and thread shape; it appends the capacities; `finalize` converts the lists to the arrays of 3.5; `SolverVBD` allocates the point and needle state of chapters 1 and 2 from the capacities. The demo builds two tissue cuboids, the needle with its holder, the thread with its clamped end, and drives the holder through two bites and a pull.

### 3.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Authoring a needle | rigid body plus shapes, no suture semantics (`builder.py:5952`, `builder.py:9299`) | `add_suture_needle(needle_body, tip_shape, centerline, thread_bodies, ...)` |
| Rod curve | none | `suture_segment_*` arrays, one segment per body, tip first |
| Thread | `add_rod` capsule chain (`builder.py:10481`) | unchanged; its bodies are listed in `add_suture_needle` |
| Gate masks | none | `shape_contact_gate` set by the builder; `particle_contact_gate` owned by the solver update (chapter 2) |
| Tissue parameters | none | per-particle custom attributes `vbd.suture_grip`, `vbd.suture_mu`, `vbd.suture_puncture_force`, `vbd.suture_puncture_cos` with defaults 1.0, 0.3, 0.5, 0.5 |
| Example | none | `newton/examples/vbd/example_vbd_suture.py` with test mode checks |

### 3.3 Argument list

```
ModelBuilder.add_suture_needle(
    needle_body: int,                                 # dynamic rigid body that carries the needle shapes
    tip_shape: int,                                   # shape on needle_body whose contacts decide the puncture
    *,
    centerline: Sequence[Vec3],                       # needle centerline in the body frame, tip first [m]; a polyline, an arc in the demo (R3)
    thread_bodies: Sequence[int] | None = None,       # capsule bodies of the thread from the swage to the free end; None = no thread
    max_tunnels: int = 4,                             # tunnel capacity T of this needle
    max_points_per_tunnel: int = 256,                 # point capacity Q per tunnel
    label: str | None = None,
    custom_attributes: dict[str, Any] | None = None,  # SUTURE_NEEDLE frequency attributes, forwarded as add_attachment_particle_body does (builder.py:11246)
) -> int                                              # suture needle index
```

The order mirrors `add_attachment_particle_body`: the two indices first, then keyword-only parameters. Resolution of the thread segments: for each thread body, the segment endpoints are the capsule centerline ends `(0, 0, -half_height)` and `(0, 0, half_height)` of its first capsule shape (`builder.py:10481`, note on `body_frame_origin = "com"`); the end nearer the previous segment's end in the initial world pose becomes `a`. Needle segments: consecutive centerline points, all on `needle_body`; `suture_needle_tip_local = centerline[0]`. Tissue parameters: declared with their defaults by `register_custom_attributes`; a scene sets them per particle through the custom value API (`builder.py:2955`, `builder.py:3050`) or fills the model array after `finalize`.

Not arguments: the gate bit (assigned as `1 << needle index`), the segment rest lengths (computed), the tissue parameters (per particle, so two tissues can differ, D10), the holder (a scene construction with `add_joint_d6`, D4).

### 3.4 Input types

No new Python type. The centerline is a sequence of `Vec3` like `Rod.points` (`rod.py:124-127`). The thread bodies are the first element of the pair returned by `add_rod` (`builder.py:10562-10563`).

### 3.5 Data structures on the model

| Attribute | dtype, shape | Meaning |
|---|---|---|
| `suture_needle_count` | int | `N` |
| `suture_needle_body` | int32 `[N]` | needle body |
| `suture_needle_tip_shape` | int32 `[N]` | tip shape |
| `suture_needle_tip_local` | vec3 `[N]` | tip in the body frame [m], `centerline[0]` |
| `suture_needle_group` | int32 `[N]` | gate bit `1 << n` |
| `suture_needle_max_tunnels`, `suture_needle_max_points` | int32 `[N]` | `T`, `Q` |
| `suture_needle_segment_start` | int32 `[N + 1]` | segment range per needle |
| `suture_segment_body` | int32 `[S]` | body of the segment |
| `suture_segment_local_a`, `suture_segment_local_b` | vec3 `[S]` | endpoints in the body frame [m]; `a` toward the tip |
| `suture_segment_rest_length` | float `[S]` | `|b - a|` [m] |
| `shape_contact_gate` | int32 `[shape_count]` | gate bits of the shapes; 0 for every shape outside a needle group |
| `particle_contact_gate` | int32 `[P]` | gate bits of the particles; zero at `finalize`, written by the solver (chapter 2) |
| `vbd.suture_grip` | float32 `[P]` | grip force per unit length [N/m], default 1.0 |
| `vbd.suture_mu` | float32 `[P]` | tunnel friction coefficient, default 0.3 |
| `vbd.suture_puncture_force` | float32 `[P]` | puncture force [N], default 0.5 |
| `vbd.suture_puncture_cos` | float32 `[P]` | cosine of the puncture angle, default 0.5 (60 degrees) |

Conventions reused: a kinematic body is a body with zero inverse mass whose pose the scene writes (`rigid_vbd_kernels.py:4103-4106`); joints to the needle filter collisions between parent and child shapes by default (`builder.py:6978`); `ModelBuilder.color()` colors bodies by joint connectivity, so the needle, the holder and the thread get body colors without change (`builder.py:13124`); custom attributes in the `vbd` namespace follow `register_custom_attributes` (`solver_vbd.py:2557-2566`).

### 3.6 Pseudo code

```
add_suture_needle(needle_body, tip_shape, *, centerline, thread_bodies, max_tunnels, max_points_per_tunnel, label, custom_attributes):   # new; builder.py, next to add_attachment_particle_body (11237)
    if needle_body or tip_shape out of range, or shape_body[tip_shape] != needle_body: raise IndexError
    if len(centerline) < 2 or a point is not finite: raise ValueError
    n = len(suture_needle_body)
    append needle_body, tip_shape, centerline[0], 1 << n, max_tunnels, max_points_per_tunnel to the needle lists
    for i in range(len(centerline) - 1):
        append segment (needle_body, centerline[i], centerline[i + 1])                        # tip first
    previous_end = world position of centerline[-1]
    for body in thread_bodies:
        shape = first capsule shape of body; h = its half height
        a, b = (0, 0, -h), (0, 0, h) in the body frame
        if world(a) is farther from previous_end than world(b): a, b = b, a                   # orient from the swage to the free end
        append segment (body, a, b); previous_end = world(b)
    for shape in shapes of needle_body and of every thread body:
        shape_contact_gate[shape] |= 1 << n
    return n

finalize(device):                                                                           # builder.py:14757
    ... existing ...
    suture_needle_* lists to model arrays; suture_segment_* lists to model arrays           # as builder.py:15918-15924
    model.shape_contact_gate = array of the shape bits; model.particle_contact_gate = zeros of P

Example.__init__(viewer, args):                                                              # new example, skeleton as example_vbd_dat_rigid_soft.py:122
    builder = newton.ModelBuilder()
    SolverVBD.register_custom_attributes(builder)                                            # solver_vbd.py:2531
    add the table as a static box
    for side in (left, right):
        add_soft_mesh(cuboid 40 x 30 x 12 mm, tet edge 3 mm, density 1000, E about 100 kPa, nu 0.45)   # builder.py:12441; 4 mm gap between the pieces
    holder = add_body(kinematic: mass 0)                                                     # builder.py:5952
    needle = add_body(); 12 add_shape_capsule along the arc plus one tip capsule             # builder.py:9299; arc radius 12 mm, wire radius 0.4 mm
    add_joint_d6(holder, needle, three linear and three angular JointDofConfig with target_ke, target_kd)   # builder.py:6952
    thread_bodies, thread_joints = add_rod(Rod(points along the needle tail direction, radius 0.15 mm), ...)   # builder.py:10481; 150 mm, 2.5 mm segments
    add_joint_rod(needle, thread_bodies[0], parent_xform with +Z along the arc tangent at the tail)   # builder.py:7006; the swage joint
    clamp = add_body(kinematic); add_joint_fixed(clamp, thread_bodies[-1])                 # the clamped end
    add_suture_needle(needle, tip_shape, centerline = arc points tip first, thread_bodies = thread_bodies)
    builder.color(); model = builder.finalize()                                              # builder.py:13124
    solver = SolverVBD(model, iterations = 10, rigid_soft_enable_dat = True, ...)

Example.step():                                                                              # new
    for substep in range(10):
        holder pose = scripted: approach, rotate about the arc center through block A and block B, withdraw, pull   # holder is kinematic
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        swap states

Example.test_final():                                                                        # new, as example_vbd_dat_rigid_soft.py:387
    gap between the facing block faces < 0.5 mm; largest active row residual < 0.2 mm; no penetration outside the gate set; exactly four punctures
```

Names that exist today: `add_attachment_particle_body` `builder.py:11237-11320`, `finalize` `builder.py:14757`, `add_soft_mesh` `builder.py:12441`, `add_body` `builder.py:5952`, `add_shape_capsule` `builder.py:9299`, `add_joint_d6` `builder.py:6952`, `add_rod` `builder.py:10481`, `add_joint_rod` `builder.py:7006`, `register_custom_attributes` `solver_vbd.py:2531`, `color` `builder.py:13124`, the example skeleton `example_vbd_dat_rigid_soft.py:122-412`. New names: `add_suture_needle`, the `suture_*` model arrays, `shape_contact_gate`, `particle_contact_gate`, the `vbd.suture_*` attributes, `example_vbd_suture.py`.

### 3.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| Needle driving, closed 2026-11-05 (D4, Anka, chat) | scene | kinematic holder with six D6 drives (chosen); kinematic needle; grasped needle | demo: drive stiffness sweep once the demo runs |
| Thread model, closed 2026-11-05 (D5) | none | `add_rod` capsule chain with a ROD swage joint (chosen); particle chain; particle Cosserat rod | none |
| Tissue parameter scope, closed 2026-11-05 (D10) | none | per particle (chosen); per needle scalars | none |
| Thread segment source | none | the first capsule shape of each thread body (chosen); an explicit polyline per body | none |
| Needle collision shapes | scene | capsules along the arc (chosen); one mesh shape with a mesh SDF | none |
| Clamped end | scene | static clamp body with a FIXED joint (chosen); a second kinematic puller | none |
| Capacity defaults | `max_tunnels`, `max_points_per_tunnel` | 4 and 256 (chosen for the demo: two bites, about 60 points per bite at 1.5 mm spacing) | none |

### 3.8 Open questions for the reviewer

1. Should the tissue parameters be set through the builder's custom value API per particle, or is filling the model array per soft mesh after `finalize` acceptable for the first demo?
2. Demo material: Young's modulus about 100 kPa with Poisson 0.45 and a 4 mm gap; raise or lower before the recording?
3. Holder drive stiffness and damping defaults for the demo: proposed `1e4 N/m` and `1e2 N s/m` linear, `10 N m/rad` and `0.1 N m s/rad` angular; acceptable as a start?
