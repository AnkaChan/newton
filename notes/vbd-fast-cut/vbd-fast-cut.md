# Fast cut in SolverVBD: design

- Purpose: one chapter per component of a cut feature for tetrahedral tissue in `SolverVBD`, written before the code.
  Chapter 1: authoring and model data. Chapter 2: the blade force. Chapter 3: the cut faces. Chapter 4: the split and the start gap. Chapter 5: the surface rows. Chapter 6: solver setup. Chapter 7: the stage-1 scene.
  Two lists close the document: limits, and checks with ablations.
  D11 asks for stated limits, and one ablation entry serves several forks, so both are lists for the whole document.
  Chapters 2 to 6 name section N.5 "Data structures on the solver", because their arrays live on the solver.
  Chapters 2, 3, 5, 6 and 7 are shorter than the 150 lines of the chapter template.
  Chapters 3 to 6 put check results, word definitions or consequences under the table or the code block of a section.
- Inputs: `newton-physics/newton` main at `2134db9cd` (paths without a prefix);
  the suture branch `ankac/vbd-suture-dev` at `70de8cf35` (paths with the prefix `vbd-suture/`);
  the study `inputs/prior-work.md`; the words `inputs/CONTEXT.md`;
  the decisions `decisions.md` (D1 to D14, Anka, 2026-11-10, terminal).
  `inputs/options.md` is the state before D4 to D8: its blade without contact, its blade plate and its armed stroke are replaced.
- Pseudo code convention: structured statements with real function names.
  Names that exist today are written as they are and cited. Names that do not exist yet carry (new).
  Only the body of a function and the body of a control statement are indented.
  The blocks of 4.6 and 5.6 have about 75 and 60 lines, more than the 40 of the style rule: each keeps one kernel together with its helper functions.
- Requirements: R1 fast, graph-capturable, pre-allocated data, about one cut per frame. R2 full self contact between the new cut surfaces. R3 no model rebuild and no solver rebuild per cut. R4 stage 1: cut, then stitch.
- Status: version 4, 2026-11-10 (PT), after three rounds of critics.
  This document was written before the code.
  An implementation now exists on the local branch `ankac/vbd-fast-cut-impl` (worktree `/home/horde/Code/Graphics/newton-working-copies/vbd-fast-cut-impl`, on top of the suture branch at `70de8cf35`, not pushed).
  It was written after Anka's order to get the demo ready.
  The list of its deviations from this document is in `impl-notes.md`, next to this document.
  Three host-only checks ran with Anka's word (D12): Python on the host, no solver, no GPU.
  Checks 2 and 3 also ran cases that D12 does not list: a tissue drift of 0.8 mm, and two second meshes that are not tissue meshes. Their numbers are marked (extra).
  One simulation run, P0, ran with Anka's word (D14): notes in `checks/p0/p0_notes.md`, an independent review in `checks/p0/p0_review.md` (verdict: Q1 trust, Q2 trust with limits).
  P0 opens fork F3.4: how the cut goes on below the first tet row. Its three arms were measured with the implementation on the stage-1 scene (3.3). A number from those runs is marked "implementation run".
  A sentence marked "inference" is my reasoning. A number marked "critic" comes from a geometry script of a critic, not from the three checks.

The design in ten lines:

1. The cut follows existing tet faces. No tet is added, removed or reshaped (D1).
2. A cut vertex keeps its particle on one side. Each other side gets a copy: a spare particle that exists from the start (D1).
3. The blade is a dynamic rigid body with a round edge. It keeps full contact with the tissue at all times and has no gate (D4, D5, D7).
4. When the blade force reaches the cut force of the tissue, every uncut face that has a pressed vertex and whose two tets lie on different sides of the blade plane is cut: one tet row (D5, D6).
5. The wound is then a V: one start gap wide at the split vertices, closed at the cut front.
   Run P0 measured what the blade does there (`checks/p0/p0_notes.md` 6.1 to 6.3). The blade does not enter the V. It rests on the two wound lips, which stay 1.015 to 1.016 mm apart from 0.5 N to 5.0 N of blade force.
   No cut-front vertex comes within the contact range (smallest distance 3.29 mm), and all blade force sits on the 22 lip vertices.
   So the rule of line 4 would not cut the second tet row in that scene (inference of the probe author from the measurement).
   Fork F3.4 holds three ways to go on. Anka has not chosen one (D14: keep the rule, and measure it first).
6. Right after a split the original and its copy move one start gap apart, each into its own material. Without that the two wound faces sit at distance zero, where this solver gives no contact force and no truncation plane (D2).
7. Copies live in a second color group per original color (D9). Unused spares sit on a parking lattice away from tissue and tools (D10).
8. Surface triangles and edges are derived rows: kernels rewrite them from the tet corners.
9. All cut kernels are the last call of `SolverVBD.step`, with fixed launch sizes, no readback and no allocation. The captured frame rebuilds the BVH once per frame (D8).
10. Existing kernel code changes in one place: one test and one argument in the suture puncture kernel (D11 h).

## 1. Authoring: capacity, copy colors, parking lattice, blade

### 1.1 Intent

The pattern in Newton is `add_suture_needle` (`vbd-suture/newton/_src/sim/builder.py:11372-11381`).
It checks its arguments, appends one row per needle to builder lists that carry merge metadata (`vbd-suture/newton/_src/sim/builder.py:809-822`), and returns the needle index.
`finalize` uploads the lists (`vbd-suture/newton/_src/sim/builder.py:16134`).
The solver constructor allocates every array of the feature once (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3447-3461`).
The tissue values are per-particle attributes in the `vbd` namespace (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:143-150`).
Two more facts are reused. `add_soft_grid` gives a tet body skin triangles and edges with zero stiffness, for collision only (`newton/_src/sim/builder.py:12462-12466`).
The suture example builds a second set of color groups with `set_coloring` after `color()` (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:257-263`).

The cut does the same.
`add_cut_blade` (new) declares the blade body and its edge shape.
`add_cut_capacity` (new) appends the spare particles, the spare surface rows, the parking particles and one idle tet, and one copy color group per original color.
The tissue value is the per-particle attribute `vbd.cut_force` (new).
After `finalize` no array size changes.

### 1.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Entry point | `add_suture_needle` (`vbd-suture/newton/_src/sim/builder.py:11372`) | `add_cut_blade`, `add_cut_capacity` |
| Tissue value | `vbd.suture_puncture_force` and four more (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:143-150`) | `vbd.cut_force` [N], default infinite. A particle with an infinite value is never pressed, and no face that holds it is cut. A contact record adds no blade force when all its particles have an infinite value |
| Particles of a tet body | all in use (`newton/_src/sim/builder.py:12473-12493`) | plus spare particles and parking particles: not ACTIVE, mass 0, tissue radius |
| Color groups | built by `color()` (`newton/_src/sim/builder.py:13252`) | C original groups plus C copy color groups; group C + c holds the spares for vertices of color c |
| Skin rows | one triangle row per open tet face, zero stiffness by default (`newton/_src/sim/builder.py:12543-12559`, `newton/_src/sim/builder.py:12404-12408`) | plus spare triangle rows and spare edge rows, each on its own parking particle |
| Tets | all with stiffness (`newton/_src/sim/builder.py:12506-12507`) | plus one idle tet with zero stiffness on four parking particles |
| Blade | none | one dynamic body with a capsule edge shape and a cutting direction |

### 1.3 Argument list

```
ModelBuilder.add_cut_blade(
    blade_body: int,                     # dynamic rigid body that carries the blade shapes
    edge_shape: int,                     # capsule shape of blade_body: the blade edge
    *,
    cut_direction: Vec3,                 # in the body frame, from the blade body out through the blade edge
    label: str | None = None,
) -> int                                 # blade index

ModelBuilder.add_cut_capacity(
    *,                                   # keyword only; the last builder call before finalize()
    max_vertices: int = 32,              # spare particles per copy color [count]
    max_faces: int = 128,                # cut faces in the whole run; two spare triangle rows each
    max_edges: int | None = None,        # spare edge rows; None = 3 * max_faces
    park_position: Vec3,                 # world position of the first parking place [m]
    park_spacing: float,                 # distance between two parking places [m]
    particle_radius: float,              # radius of every spare and parking particle [m]
    copy_colors: bool = True,            # False: spares sit in the original color groups (fork F1.1)
) -> None
```

`add_cut_blade` follows `add_suture_needle(needle_body, tip_shape, *, ...)`: the body, then the shape whose contacts decide the event.
Two deviations from that helper.
The capacity is one call for the model and not per blade, because spare particles and spare rows belong to the tissue.
`particle_radius` has no default, because the builder default radius would change the suture gate radius and the rigid-soft motion budget,
which the solver reads once from all particles (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3569`, `vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3420-3442`).

Blade geometry (new), in world space at the time of a cut update:

- a and b are the two cap centres of the edge capsule: the shape transform applied to (0, 0, -half_height) and (0, 0, +half_height), then the body pose. The capsule axis is the Z axis of the shape frame (`newton/_src/sim/builder.py:9395-9412`).
- e = normalize(b - a) is the direction of the blade edge. The blade edge axis is the segment a b.
- d is the cutting direction: `cut_direction` rotated by the body pose. `add_cut_blade` removes its part along the capsule axis and normalizes it.
- The blade plane goes through a and has the normal n = cross(d, e). A point x is on side B when dot(n, x - a) >= 0, else on side A.

Rules of the two builder calls. Each raises on a miss.

1. `edge_shape` is a capsule of `blade_body` and belongs to no other blade.
2. `cut_direction` is finite and not parallel to the capsule axis.
3. `color()` or `set_coloring()` ran before `add_cut_capacity`, and `add_cut_capacity` runs once.
4. After `add_cut_capacity`, `color()` and `set_coloring()` raise.
5. A builder with cut capacity is neither the source nor the target of `add_builder()` or `replicate()`.
   `add_builder` joins the color groups of two builders in pairs, sorted by size (`newton/_src/sim/builder.py:4196-4211`, `newton/_src/sim/graph_coloring.py:385-413`), so a merge would mix original groups and copy color groups.
   It would also drop the capacity values (stage 1).

Not arguments: the cut force (a per-particle attribute, as the suture values); the start gap, the update interval and the switches (solver arguments, chapters 2 to 6); the color count C (read from the color groups).

### 1.4 Input types

None. The cut adds no input type.

### 1.5 Data structures on the model

Blade, frequency `cut_blade` (new), B rows. These three builder lists carry merge metadata as the `suture_needle_*` lists do (`vbd-suture/newton/_src/sim/builder.py:809-822`).

| Attribute (new) | dtype, shape | Meaning |
|---|---|---|
| `cut_blade_body` | int32 [B] | body index (references BODY); filled by `add_cut_blade` |
| `cut_blade_edge_shape` | int32 [B] | capsule shape whose contact records give the blade force (references SHAPE); filled by `add_cut_blade` |
| `cut_blade_direction` | vec3 [B] | d in the body frame, unit, perpendicular to the capsule axis; filled by `add_cut_blade` |

Capacity, filled by `add_cut_capacity`. C = number of original colors, V = `max_vertices`, F = `max_faces`, E = `max_edges`.
The capacity values are builder scalars and NumPy arrays, not builder lists: a builder list without merge metadata makes every `add_builder` call raise (`newton/_src/sim/builder.py:4323-4325`).

| Attribute (new) | dtype, shape | Meaning |
|---|---|---|
| `cut_color_count` | int | C; the copy color of color c is C + c |
| `cut_spare_start` | int32 [C] | first spare particle of the pool of copy color C + c |
| `cut_spare_count` | int | V, the length of each pool |
| `cut_tri_spare_start`, `cut_tri_spare_count` | int | first spare triangle row; 2 F rows |
| `cut_edge_spare_start`, `cut_edge_spare_count` | int | first spare edge row; E rows |
| `cut_row_park_particle` | int32 [2 F + E] | the parking particle of each spare row |
| `cut_idle_tet` | int | the tet with zero stiffness that unused tet records point at |
| `cut_park_lower`, `cut_park_upper` | vec3 | the box of the parking lattice [m] |
| `vbd.cut_force` | float32 [P] | cut force of the tissue at this particle [N]; infinite = not cuttable. Registered as the suture attributes |

What a spare looks like before use, with the Newton convention it reuses:

- Spare particle: no ACTIVE flag and mass 0, so the forward step, the plain solve and the tiled solve skip it
  (`newton/_src/solvers/vbd/particle_vbd_kernels.py:1488-1492`, `newton/_src/solvers/vbd/particle_vbd_kernels.py:3046-3048`, `newton/_src/solvers/vbd/particle_vbd_kernels.py:2852-2861`).
  It is in its copy color group from the start.
- Parking particle: the same, and it never becomes a copy.
- Spare triangle row: its parking particle three times. Spare edge row: `(-1, -1, p, p)`.
  Both act as the point p in detection (`inputs/prior-work.md` 2.4).
  The triangle row is appended to the seven triangle lists directly, because `add_triangle` drops a triangle with zero area (`newton/_src/sim/builder.py:11569-11581`).
  The stiffness of a spare row is always zero.
  A spare triangle row takes the color and the opacity of the triangle rows that exist at the call, so the model keeps one triangle appearance (limit 19).
- Idle tet: four parking particles of its own, positive rest volume, k_mu = k_lambda = 0.
  The elasticity kernels skip such a tet before any other read (`newton/_src/solvers/vbd/particle_vbd_kernels.py:2004`).
- Parking lattice (new rule): one place per spare particle and per spare row, C V + 2 F + E places.
  S = ceil(cube root of that number).
  Place number k sits at `park_position` + `park_spacing` x (k mod S, (k div S) mod S, k div S^2).
  The four idle tet particles sit at the lattice points (S, 0, 0), (S + 1, 0, 0), (S, 1, 0), (S, 0, 1).

Known price of the parking lattice (D10): the solver tests the start state only (rule 10 of 6.3).
A tool or tissue that enters the box later meets parked particles and spare rows; self contact, and the edge and face contact records of a box or capsule, do not read the ACTIVE flag (`inputs/prior-work.md` 2.4).
D10 asks for a lattice inside the scene bounds. No rule tests that (fork F1.4).

Stage-1 numbers (block of 28 x 10 x 4 cells of 3 mm).
Host-only check 1 (`checks/check1_color_count.txt`, Warp 1.19.0, default `color()`): C = 4 with group sizes 410, 402, 390, 393.
The 55 vertices of the cut plane (the grid plane x = 42 mm) have 18, 10, 12 and 15 vertices per color. No interior x plane has more than 18 per color.
The cut needs 80 cut faces, 160 triangle rows and 240 edge rows (hand count: 106 inner wound edges with two fans each, 28 wound edges on the skin with one new fan each; chapter 5).
With V = 32, F = 128, E = 384 the lattice has 768 places, S = 10, and with a spacing of 3 mm it is a box of 33 mm.

### 1.6 Pseudo code

```
park_place(k):                                                           # (new); the lattice rule of 1.5
    S = ceil(cube root of the number of places)
    return park_position + park_spacing * (k mod S, (k div S) mod S, k div (S * S))

add_cut_blade(blade_body, edge_shape, *, cut_direction, label):          # (new); as add_suture_needle, vbd-suture builder.py:11372
    if edge_shape is not a capsule of blade_body or it belongs to a blade:
        raise an error
    d = cut_direction without its part along the capsule axis
    if d is zero or not finite:
        raise an error
    append blade_body, edge_shape, normalize(d) to the blade lists
    return blade index

add_cut_capacity(*, max_vertices, max_faces, max_edges, park_position, park_spacing, particle_radius, copy_colors):   # (new)
    if particle_color_groups is empty or cut capacity exists:
        raise an error
    if max_edges is None:
        max_edges = 3 * max_faces
    C = number of color groups
    groups = the C groups of particle_color_groups
    if copy_colors:                                                      # F1.1
        append C empty groups to groups
    place = 0
    for color in range(C):
        for i in range(max_vertices):
            s = add_particle(pos = park_place(place), vel = 0, mass = 0, radius = particle_radius, flags = 0)   # builder.py:11250
            place += 1
            if i == 0:
                cut_spare_start[color] = s
            if copy_colors:                                              # F1.1
                append s to groups[C + color]
            else:
                append s to groups[color]
    last = index of the last group                                       # a defined color for particles that never move
    cut_tri_spare_start = tri_count
    for row in range(2 * max_faces):
        p = add_particle(pos = park_place(place), vel = 0, mass = 0, radius = particle_radius, flags = 0)
        place += 1
        append the triangle row (p, p, p) with zero stiffness to the seven triangle lists     # builder.py:11575-11581
        cut_row_park_particle[row] = p
        append p to groups[last]
    cut_edge_spare_start = edge_count
    for row in range(max_edges):
        p = add_particle(pos = park_place(place), vel = 0, mass = 0, radius = particle_radius, flags = 0)
        place += 1
        add_edge(-1, -1, p, p, edge_ke = 0, edge_kd = 0)                 # builder.py:11776
        cut_row_park_particle[2 * max_faces + row] = p
        append p to groups[last]
    for k in range(4):
        idle[k] = add_particle(pos = idle tet place k of 1.5, vel = 0, mass = 0, radius = particle_radius, flags = 0)
        append idle[k] to groups[last]
    volume = add_tetrahedron(idle[0], idle[1], idle[2], idle[3], k_mu = 0, k_lambda = 0, k_damp = 0)   # builder.py:11711; returns the rest volume
    if volume <= 0:
        raise an error                                                   # the builder only warns and drops such a tet
    cut_idle_tet = tet_count - 1
    set_coloring(groups)                                                 # builder.py:13238
    store C, max_vertices, the two row counts and the lattice box

finalize(device):                                                        # builder.py:14885
    run the existing finalize                                            # unchanged
    upload the blade lists and the capacity values as the model attributes of 1.5
```

Names that exist today: `add_particle` (`newton/_src/sim/builder.py:11250`),
`add_edge` (`newton/_src/sim/builder.py:11776`),
`add_tetrahedron` (`newton/_src/sim/builder.py:11711`),
`set_coloring` (`newton/_src/sim/builder.py:13238`),
`finalize` (`newton/_src/sim/builder.py:14885`).
New names: `park_place`, `add_cut_blade`, `add_cut_capacity`, the attributes of 1.5.

### 1.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| F1.1 Color of a copy | `copy_colors` | True: copy color C + c (D9). False: the color of the original, as the VBD paper allows; expected to jitter by suture decision I31 (`inputs/prior-work.md` 1.8) | A1 |
| F1.2 Empty copy colors | `cut_skip_empty_copy_colors` (6.3) | off: every color pass runs from the first frame (D9). on: `wp.capture_if` skips a copy color with no active copy (block 6.6) | A2 |
| F1.3 Spare rows per parking place | none (code change) | one row per place (proposed, no new constant). Several rows per place: fewer particles; rows on one place are detection candidates of each other (`inputs/prior-work.md` 2.4) | A8c |
| F1.4 Distance of the parking lattice | `park_position` | inside the scene bounds (D10). Any distance | A8c |
| F1.5 Unused rows | closed | parking lattice, no kernel change (D10, Anka, 2026-11-10, terminal) | none |
| F1.6 Scope of the capacity | none (code change) | one capacity for the model (proposed). One per soft body | none |

### 1.8 Open questions for the reviewer

1. F1.6: is one capacity for the model enough, or do you want pools and row ranges per soft body?
2. Rules 4 and 5 of 1.3: after `add_cut_capacity`, `color()`, `set_coloring()` and a merge raise. Acceptable for stage 1?
3. Row "Tissue value" of 1.2: is an infinite default of `vbd.cut_force` the right way to mark tissue that must not be cut?

## 2. The blade force

### 2.1 Intent

The pattern in Newton is the suture puncture.
`begin_suture_update` writes a device "due" flag (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:710`).
`accumulate_suture_tip_force` runs one thread per rigid-soft contact record, computes the normal force again with the solver's law, sums it per needle and keeps the strongest contact record through an atomic maximum (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:720-797`).
`update_suture_needle_state` compares the sum with the tissue value at the strongest contact record and tests the angle between the tip direction and the contact normal; in armed mode any positive force is enough (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:848-863`).

The blade force is that sum for the blade edge shape, taken along the cutting direction.
When it reaches the cut force of the tissue, the update is a cut advance: chapter 3 then cuts one tet row.
The force test repeats in every cut update, so the next tet row is cut when the blade has gone down and the force is reached again (D5).

### 2.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Event | puncture: a tunnel opens once (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:848-897`) | cut advance: one tet row, as often as the force test passes |
| Force | sum of the normal force of the tip contact records (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:720-797`) | blade force: sum over the blade edge contact records of normal force x positive part of (normal . d) |
| Angle test | tip direction against the contact normal (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:860-863`) | none; the positive part along d does that job (fork F2.2) |
| Tissue value | `vbd.suture_puncture_force` at the strongest contact record (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:848-863`) | `vbd.cut_force` at the strongest contact record (D6) |
| Armed mode | the first tip contact punctures (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:525`) | the first blade edge contact with force cuts |
| Tool against uncut tissue | contact gate near the tunnel (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:1014-1066`) | no gate (D5; fork F3.4 has an arm with a gate for the wound lips). Rigid-soft truncation and full-surface contact are required (rule 11 of 6.3) |
| Call site | before `_finalize_particles`, on `state_in` (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3141-3142`) | last call of `step`, on `state_out`: the cut edits positions after the velocity update |

### 2.3 Argument list

```
SolverVBD(
    ...,
    cut_armed: bool = False,             # True: the first contact with force cuts; False: force test (as suture_armed)
    cut_update_interval: int = 1,        # substeps between two cut updates (as suture_tunnel_update_interval)
    cut_force_measure: str = "total",    # "total" [N] or "per_length" [N/m] (fork F2.1)
    cut_force_records: str = "all",      # "all", "cone" or "front" (fork F2.2)
)
```

Bound of the blade force (inference, hand calculation).
With the truncation on, one contact record gives at most stiffness x (particle radius + shape margin): 0.25 N with the suture scene values.
A `vbd.cut_force` above the sum of these bounds is never reached, and the tissue holds the blade.
The host can read `_cut_blade_force`.

Not arguments: the cut force (a tissue attribute, 1.5).

### 2.4 Input types

None.

### 2.5 Data structures on the solver

B = blades (1 in stage 1).

| Attribute (new) | dtype, shape | Meaning |
|---|---|---|
| `_cut_substep`, `_cut_due`, `_cut_update_number` | int32 [1] each | substep counter; 1 in a substep with a cut update; number of the current cut update. Written by `begin_cut_update` |
| `_cut_shape_blade` | int32 [shape_count] | blade index of a blade edge shape, else -1. Host, at construction |
| `_cut_blade_force` | float32 [B] | blade force [N], this cut update |
| `_cut_blade_record_key` | int64 [B] | strongest blade edge contact record, same key as `tip_record_key`; 0 = none |
| `_cut_contact_count` | int32 [B] | contact records of the blade edge shape, this cut update |
| `_cut_advance` | int32 [B] | 1 when the force test passed in this cut update |

A stroke (new) is the time during which the blade edge shape has at least one contact record.
While the blade edge is inside a wound or rests on its lips, tissue is within the detection range, so the stroke goes on (inference; in run P0 the blade edge shape has 218 contact records with force while it rests on the lips, `checks/p0/p0_notes.md` 6.2).

The normal of a contact record points from the shape to the tissue, and no array stores the normal force.
`accumulate_cut_blade_force` computes it again as its template does, with the same inputs (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:763-797`).
It keeps the early returns of its template before the force:
the contact record is below the record count, its shape is not negative, the shape belongs to a blade (a needle in the template), the contact filter allows the force, and the first particle index is not negative (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:750-762`).
A particle index of -1 in the second or third place is skipped (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:772-784`).

### 2.6 Pseudo code

```
step(state_in, state_out, control, contacts, dt):                        # solver_vbd.py:2920
    run the existing step up to the suture update                        # unchanged
    _update_suture_needles(state_in, contacts)                           # suture branch; reads the mesh before the cut
    _finalize_particles(state_out, dt)                                   # called at solver_vbd.py:3069
    _save_tendon_history(state_out)                                      # solver_vbd.py:3070
    if the model has a blade:
        _update_cut_blades(state_out)                                    # (new) last call

_update_cut_blades(state):                                               # (new); kernels only, fixed launch sizes
    contacts = the contact records of the solver-owned pipeline          # rule 11 of 6.3
    begin_cut_update()                                                   # dim 1
    accumulate_cut_blade_force(contacts, state)                          # dim soft_contact_max
    decide_cut_advance(contacts)                                         # dim B
    mark_pressed_vertices(state)                                         # chapter 3, dim P
    mark_cut_faces(state)                                                # chapter 3, dim 4 T
    split_cut_vertices(state)                                            # chapter 4, dim P
    compute_start_gap_offset(state)                                      # chapter 4, dim P
    apply_start_gap_offset(state)                                        # chapter 4, dim P
    take_surface_rows()                                                  # chapter 5, dim 4 T
    refresh_triangle_rows()                                              # chapter 5, dim tri_count
    refresh_edge_rows()                                                  # chapter 5, dim edge_count

begin_cut_update():                                                      # (new) one thread; as begin_suture_update
    _cut_substep += 1
    _cut_due = 1 if _cut_substep is a multiple of cut_update_interval else 0             # F2.3
    if _cut_due == 1:
        _cut_update_number += 1
        for blade in blades:
            _cut_blade_force[blade] = 0
            _cut_blade_record_key[blade] = 0
            _cut_contact_count[blade] = 0

accumulate_cut_blade_force(contacts, state):                             # (new) one thread per contact record; as accumulate_suture_tip_force
    for record in the rigid-soft contact records:
        blade = _cut_shape_blade[shape of record]
        if _cut_due == 0 or blade < 0:
            continue
        _cut_contact_count[blade] += 1                                   # atomic; the stroke goes on
        if every particle of the record has an infinite cut_force:
            continue
        f = normal force of the record * max(dot(normal of the record, d of blade), 0)     # F2.2: which contact records count
        if f <= 0:
            continue
        _cut_blade_force[blade] += f                                     # atomic
        _cut_blade_record_key[blade] = max(_cut_blade_record_key[blade], key of (f, record))   # atomic

decide_cut_advance(contacts):                                            # (new) one thread per blade
    for blade in blades:
        _cut_advance[blade] = 0
        if _cut_due == 0 or _cut_blade_record_key[blade] == 0:
            continue                                                     # no contact record with force
        p = among the particles of the strongest contact record that have a finite cut_force, the one with the largest contact weight
        if cut_armed or _cut_blade_force[blade] >= cut_force[p]:         # F7.3: cut_armed; F2.1: per_length divides the force by the contact length
            _cut_advance[blade] = 1
```

Names that exist today: `step` (`newton/_src/solvers/vbd/solver_vbd.py:2920`),
`_finalize_particles` (defined at `newton/_src/solvers/vbd/solver_vbd.py:5382`),
`_save_tendon_history` (called at `newton/_src/solvers/vbd/solver_vbd.py:3070`, defined at `newton/_src/solvers/tendon_state.py:191`),
`_update_suture_needles` (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3765`),
`begin_suture_update` and `accumulate_suture_tip_force` as templates (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:710-797`).
New names: every function marked (new), the arrays of 2.5.

### 2.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| F2.1 Force measure | `cut_force_measure` | total force on the blade edge [N] (D6). Force per length of contact [N/m]: needs a rule for the contact length and a lower bound | A3 |
| F2.2 Contact records in the blade force | `cut_force_records` | all contact records of the edge shape, positive part along d (D14 keeps this; run P0: below the first tet row all of this force comes from the wound lips). Only contact records inside a cone around d, with a per-particle cosine as `vbd.suture_puncture_cos`. Only contact records with a particle on the cut front | P0, A5 |
| F2.3 Update interval | `cut_update_interval` | 1 substep (proposed, as the suture). Larger values | A4 |
| F2.4 Blade contact | closed | full contact, no gate, force-driven advance (D4, D5, Anka, 2026-11-10, terminal) | none |

D14 in `decisions.md` names F2.2 and F3.2 by their version 2 numbers, F2.5 and F2.6.

### 2.8 Open questions for the reviewer

1. F2.2: the blade force counts only the positive part of normal . d. Is that the reading of D6 you want, or do you want the full normal force as for the needle tip?
2. Rule 11 of 6.3: stage 1 allows one blade. Do you agree to leave two blades to a later stage?

Closed in the terminal (2026-11-10, D14): the rule of F2.2 stays as written, and run P0 measures it first.
Run P0 ran. Its result opens fork F3.4 (question 3.8.3).

## 3. The cut faces

### 3.1 Intent

The pattern in Newton is the growth of a suture tunnel: after the event, a kernel walks the tet mesh through a tet-neighbor table and marks what the tool reached (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:905-957`).
The table is built once on the host; an entry below zero is a boundary (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:50-75`).

A cut advance marks faces in the same table.
A vertex is pressed when it is within the contact range of the blade edge.
A face is cut when it has a pressed vertex and its two tets have different side labels.
A cut face gets a negative entry on both sides, so the suture walk exits there (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:908-911`).

### 3.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Tet neighbors | private suture table, -1 on the skin (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3532-3537`) | one shared table; a cut face holds -(2 + other tet), so the cut can still name the other tet |
| Which tissue the tool reached | particles within the gate radius of a tunnel (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:1014-1066`) | pressed vertices: cuttable ACTIVE particles within the contact range of the blade edge axis |
| Side of the tool | none | side label of a tet: the side of the blade plane where its centroid is, kept for the stroke |
| Interior faces | not stored (`newton/_src/sim/builder.py:12495-12504`) | named by (tet, face), with a stamp and a side for cut faces |

### 3.3 Argument list

```
SolverVBD(
    ...,
    cut_contact_range: str | float = "detection",   # "detection", "force" or a value [m] (fork F3.2)
    cut_keep_side_labels: bool = True,   # False: side labels are computed again at every cut update (fork F3.1)
)
```

Contact range of the blade edge (new, a host constant per blade). Three distances from the blade edge axis:

- Force range: edge radius + shape margin + particle radius. Inside it a contact record gives force. Stage 1: 0.45 + 0 + 0.5 = 0.95 mm (shape margin: the default 0, `vbd-suture/newton/_src/sim/builder.py:1063`).
- Detection range: the force range plus the pipeline `soft_contact_gap`. Stage 1: 0.95 + 1.0 = 1.95 mm (`soft_contact_gap` of the suture example, `vbd-suture/newton/examples/vbd/example_vbd_suture.py:128`).
- Contact range: the proposed value is the detection range, as the suture gate radius without its slack (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3570-3577`). The particle radius is that of the cuttable particles.

A cut advance cuts one tet row only when the cells are larger than `soft_contact_gap` (inference, hand calculation): with smaller cells the vertex row below a pressed row is inside the contact range too.
Stage 1: 3 mm against 1 mm.

What the rule does (host-only check 2: two independent scripts, 340 of 340 requested numbers equal, `checks/check2_3_compare.md`).
The check replays the rule on the rest shape and takes the force test as passed at every replay step.
A replay step is one blade position of the replay.
The check has no contact records. It ends a stroke at a replay step with no pressed vertex (`checks/check2_3_compare.md`, section 5, item 3), so it clears the side labels at least as often as the solver will (inference).
A tet is cut free when its part has fewer than 1 percent of all tets.
Tissue drift (extra): the tissue moves along the blade plane normal during the replay, from 0 at the first replay step to 0.8 mm at the last.
The two second meshes (extra) are mesh a, a block of 12 x 6 x 4 cells with moved inner vertices, meshed again by Delaunay (455 vertices, 1982 tets), and mesh b, an fTetWild mesh of a slab (2056 vertices, 9438 tets). Neither is tissue.

| Blade | Cut faces | Split vertices | Parts | Largest distance from the blade plane to a cut face | Lead of the cut |
|---|---|---|---|---|---|
| on the grid plane | 80 | 55 | 2800 + 2800 tets | 0 | 4.95 mm |
| a quarter cell off | 80 | 55 | 2800 + 2800 | 0.75 mm (the cut is flat, on the grid plane) | 4.70 mm |
| half a cell off | 80 | 55 | 2880 + 2720 | 1.5 mm | 4.20 mm |
| tilted 15 degrees | 96 to 100 | 65 to 67 | 2800 + 2800 | 2.1 to 2.3 mm | 4.95 to 5.23 mm |

- In all runs of the table the body separates into two parts, and no tet is cut free.
- The lead is the largest distance, along d, from the blade edge axis to a particle of a face cut in that replay step: the contact range plus one tet.
- (extra) With a tissue drift of 0.8 mm during the stroke and side labels that are not kept, the rule cuts a second surface: 14 tets cut free at 10 degrees. With kept labels: none.
- (extra) On mesh a and mesh b the body separates in all 6 runs; the lead is 5.4 to 7.6 mm.

Run P0 (measured, `checks/p0/p0_notes.md` 6.1 to 6.4; review `checks/p0/p0_review.md`).
The probe scene is a held block of 12 x 10 x 4 cells with the top tet row of the cut plane already split and one start gap open.
The blade does not enter the V. It rests on the two wound lips.
The lips stay 1.015 to 1.016 mm apart from 0.5 N to 5.0 N of force along d. The blade edge with the particle radius needs 1.9 mm.
No cut-front vertex comes within the contact range: the smallest distance to the blade edge axis is 3.29 mm.
The force along d reaches 0.5 N at 0.73 mm of holder travel and 1.0 N at 1.17 mm.
Every contact record with force has its contact point on one of the 22 lip vertices.
The first 0.2 to 0.33 s after the start gap shake, with particle speeds up to 439 mm/s.
So the replay above does not hold below the first tet row: with the rule of 3.1 only the lips are pressed, and their faces on the blade plane are already cut (inference of the probe author; the probe has no cut code).

The three arms of fork F3.4 were measured in implementation runs on the stage-1 scene: block of 28 x 10 x 4 cells, cut force 0.5 N, holder speed 4 mm/s, 10 substeps, 20 iterations, graph capture.
The logs are `impl/logs/d1_gate_cut.log`, `impl/logs/d3_wound_cut.log`, `impl/logs/d3_design_cut.log` and `impl/logs/d2_gate_full.log` in the task folder.
A wound particle is a particle of a face that was cut in the current stroke, or a copy of one.

| Arm | Switch | Rule | Measured (implementation run) | Against the decisions |
|---|---|---|---|---|
| edge | none: the rule of 3.1 as designed | a vertex is pressed only within the contact range of the blade edge axis | The first tet row is cut at 0.6 s. Then the blade rests on the lips and pushes the block down, and the blade force rises to 5.8 N. Later tet rows are cut unevenly as the tissue deforms (20, 22, 36, 39, 42, 44, 46, 50 cut faces over 3.5 s), with particle speeds up to 757 mm/s. All 80 faces are cut only when the holder is at the end of its travel | as decided (D4, D5) |
| wound | `cut_press_wound = True` | a wound particle inside the span of the blade edge counts as pressed too; no gate | All 80 faces are cut within about 0.3 s of the first force test that passes: one tet row per cut update. Then the blade wedges down through the open wound with 1.1 to 3.2 N. Largest particle speed: 118 mm/s, when the blade leaves the block | keeps D4 and D5; the cut runs far ahead of the blade edge |
| gate | `cut_gate_wound = True` | the blade has no contact with the wound lips (split vertices and copies). It uses the contact gate of the suture branch with one more bit. Contact with uncut tissue stays as it is | One tet row at a time, as the blade edge reaches the cut front: 20, 40, 60 and 80 cut faces at about 0.55, 1.3, 2.0 and 2.8 s. Blade force at most the cut force of 0.5 N before each tet row. Largest particle speed 16 mm/s. Largest blade penetration into tissue outside the wound 0.008 mm. The full demo (cut, 4 mm pull, three stitches, release) passes its final test: 80 cut faces, parts of 2800 and 2800 tets, 18, 10, 12 and 15 copies per pool, 160 triangle rows, 240 edge rows, no overflow, six through tunnels, wound gap 1.0 mm before the release | departs from D5 ("no gate") for the wound lips only |

No arm is decided.
The demo of the implementation uses the gate arm, because it is the only arm in which the cut follows the blade edge.
That is the implementer's choice for the demo, not a decision.

Not arguments: the blade geometry (1.3).

### 3.4 Input types

None.

### 3.5 Data structures on the solver

T = tets, P = particles.

| Attribute (new) | dtype, shape | Meaning |
|---|---|---|
| `_cut_contact_range` | float32 [B] | contact range of the blade edge [m]. Host, at construction |
| `_cut_particle_pressed` | int32 [P] | 1 for a pressed vertex, this cut update |
| `_tet_neighbors` | int32 [T, 4] | tet across the face opposite corner i; -1 skin; -(2 + other) cut. Host, at construction; `mark_cut_faces` writes cut faces. One array for cut and suture |
| `_cut_tet_side` | int32 [T] | side label of a tet for the current stroke: -1 side A, +1 side B, 0 none |
| `_cut_face_side` | int32 [T, 4] | side of this tet at a cut face |
| `_cut_face_stamp` | int32 [T, 4] | number of the cut update that cut this face, -1 if not cut |
| `_cut_particle_dirty` | int32 [P] | 1 for a particle of a new cut face that the split has not examined |

### 3.6 Pseudo code

```
mark_pressed_vertices(state):                                            # (new) one thread per particle; stage 1: blade 0
    for p in all particles:
        _cut_particle_pressed[p] = 0
        if _cut_due == 0 or p is not ACTIVE or cut_force[p] is infinite:
            continue
        if distance from particle_q[p] to the blade edge axis <= _cut_contact_range[0]:        # F3.2, F3.3; F3.4: the wound arm also presses wound particles inside the span
            _cut_particle_pressed[p] = 1

mark_cut_faces(state):                                                   # (new) one thread per (tet, face); stage 1: blade 0
    for tet, face in all tet faces:
        if _cut_due == 0:
            continue
        if _cut_contact_count[0] == 0:
            _cut_tet_side[tet] = 0                                       # the stroke ended: clear the side labels
            continue
        other = _tet_neighbors[tet, face]
        if other < tet:
            continue                                                     # skin, already cut, or the thread of the other tet handles this face
        if _cut_advance[0] == 0 or no particle of the face is pressed:
            continue
        if a particle of the face has an infinite cut_force:
            continue
        side_tet = side_label(tet)
        side_other = side_label(other)
        _cut_tet_side[tet] = side_tet
        _cut_tet_side[other] = side_other
        if side_tet == side_other:
            continue
        face_other = the face of other whose neighbor is tet
        _tet_neighbors[tet, face] = -(2 + other)
        _tet_neighbors[other, face_other] = -(2 + tet)
        _cut_face_side[tet, face] = side_tet
        _cut_face_side[other, face_other] = side_other
        _cut_face_stamp[tet, face] = _cut_update_number
        _cut_face_stamp[other, face_other] = _cut_update_number
        for p in the three particles of the face:
            _cut_particle_dirty[p] = 1

side_label(tet):                                                         # (new) function
    if cut_keep_side_labels and _cut_tet_side[tet] != 0:                 # F3.1
        return _cut_tet_side[tet]
    if dot(n, centroid of tet - a) >= 0:                                 # n and a of blade 0, 1.3
        return +1                                                        # side B
    return -1                                                            # side A
```

Names that exist today: none in this block; the tet-neighbor table is built by `build_tet_neighbors` (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:50`). New names: `mark_pressed_vertices`, `mark_cut_faces`, `side_label`, the arrays of 3.5.

### 3.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| F3.1 Side labels | `cut_keep_side_labels` | kept for the stroke (proposed; host-only check 2, extra: without it a drift of 0.8 mm cuts tets free). Computed again at every cut update | host-only check 2 |
| F3.2 Contact range | `cut_contact_range` | detection range (proposed). Force range: a pressed vertex then always carries force, and the wound must open wider before the next tet row (inference, not run). A value in metres | P0, A5 |
| F3.3 Pressed vertex | none (code change) | by distance to the blade edge axis (proposed). The particles of the blade edge contact records | A5 |
| F3.4 Advance below the first tet row | `cut_press_wound`, `cut_gate_wound` (implementation; both False is the edge arm) | edge: the rule of 3.1 as designed. wound: wound particles inside the span of the blade edge count as pressed. gate: the blade has no contact with the wound lips; the gate acts in the contact filter of the suture branch (`vbd-suture/newton/_src/geometry/soft_contacts_mesh.py:1612-1621`), which no block of this document shows. Not decided; the measured arms are in 3.3 | P0; implementation runs of 3.3 |

### 3.8 Open questions for the reviewer

1. F3.2: with the detection range the cut leads the blade edge axis by about 5 mm in stage 1 (table of 3.3). Acceptable, or do you want the force range?
2. Row "Side of the tool" of 3.2: a stroke ends when the blade edge has no contact record, and the side labels are cleared then. Is that the right end of a stroke, or should the host end it?
3. F3.4: run P0 shows that the rule of 3.1 does not go on below the first tet row as designed, because the blade edge rests on the wound lips.
   Which arm do you want?
   Edge: as decided, but uneven and with up to 5.8 N.
   Wound: no gate, but the cut runs far ahead of the blade edge.
   Gate: the cut follows the blade edge, but the wound lips have no contact with the blade, against D5.
   The numbers are in 3.3.

## 4. The split and the start gap

### 4.1 Intent

Newton has no pattern for an index edit on the device (`inputs/prior-work.md` 2.1).
The method is the split by copied vertices that the Vertex Block Descent paper names for tearing: only vertex indices change, and the elastic coloring stays valid (Chen, Liu, Yang, Yuksel 2024, section 3.9).
Two code facts allow it.
The elasticity solve reads `tet_indices`, the rest matrices, the masses and the flags from the model at every launch (`newton/_src/solvers/vbd/solver_vbd.py:4517-4541`).
It trusts the vertex-to-tet table without a check (`newton/_src/solvers/vbd/particle_vbd_kernels.py:1999-2016`).

So the split edits contents only.
For each particle of a new cut face it finds the tet groups of that vertex, gives each group but one a copy, re-points the tet corners, moves the tet records and shares the mass.
Then each new particle moves half a start gap into its own material.

### 4.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Vertex-to-tet table | exact size, built at `finalize`, one struct shared by solver and collision (`newton/_src/utils/mesh.py:280-298`, `newton/_src/solvers/vbd/solver_vbd.py:1232-1236`) | the solver holds its own struct with two new tet arrays that have Q idle tet records per spare particle; the other fields still point at the model arrays |
| Unused tet record | does not exist | `(cut_idle_tet, 0)`; skipped by the plain and the tiled solve (`newton/_src/solvers/vbd/particle_vbd_kernels.py:2004`, `newton/_src/solvers/vbd/particle_vbd_kernels.py:2949`) |
| Mass of a particle | set at build time (`newton/_src/sim/builder.py:12471-12493`) | a copy takes the share that fits the rest volume of its tets (D11 d) |
| Position of a wound face | none | the original and its copy start one start gap apart (D2) |

Words. A cut vertex is a vertex on a cut face.
A split vertex is a cut vertex that has a copy.
The tet groups of a vertex are the sets of its tets that are joined through uncut faces.
The original stays on side A when it can (new rule).

### 4.3 Argument list

```
SolverVBD(
    ...,
    cut_start_gap: float | None = None,  # distance between the two wound faces right after a split [m]; None = particle_self_contact_margin
    cut_gap_direction: str = "largest_least_opening",   # "normal_sum", or "normal_sum_first" (fork F4.2)
    cut_start_gap_in_rest: bool = False, # True: the rest shape moves too (fork F4.3; not designed yet)
    cut_mass_rule: str = "rest_volume",  # or "full_copy" (fork F4.4)
    cut_take_mode: str = "atomic",       # or "ordered" (fork F4.5)
)
```

Size of the gap (inference: hand calculation from `newton/_src/solvers/vbd/particle_vbd_kernels.py:866-882`, with about 8 pairs per vertex).
With the suture scene values a gap of one margin (1 mm) gives no contact force at the split.
A gap of 0.5 mm gives about 2 N per vertex, and 0.1 mm about 10 N.
A gap below 0.02 mm can freeze vertices in the truncation (`inputs/prior-work.md` 2.7).
Each new particle moves by half a gap, so a tet at the wound whose height is below about one gap can start inverted.
The constructor warns when `cut_start_gap` is larger than the smallest height of a tet with a cuttable particle (rule 12 of 6.3).

Direction of the gap. Each new particle moves into its own material, away from every cut face at it.
The opening of a direction u at a cut face is -dot(u, m), with m the outward unit normal of the tet face.
Two rules were compared on the rest shape with a gap of 1 mm (host-only check 3, `checks/check2_3_compare.md`).
A pair is one cut face and one of its split vertices.
The opening of a pair is the distance that the two particles of that vertex move apart along the face normal, in mm for a gap of 1 mm; a crossed pair has an opening of zero or less.

| Case | Normal sum: crossed pairs | Largest least opening: crossed pairs | Largest least opening: smallest opening | Inverted tets |
|---|---|---|---|---|
| grid, blade on a grid plane, a quarter cell off, or tilted 3 degrees | 0 | 0 | 1.0 mm | 0 |
| grid, half a cell off | 0 | 0 | 0.58 mm | 0 |
| grid, tilted 10 degrees | 0 | 0 | 0.58 to 0.81 mm | 0 |
| grid, tilted 15 degrees | 16 to 20 | 0 | 0.34 mm | 0 |
| (extra) mesh a and mesh b, 6 cuts | 0 to 20 per cut | 0 to 3 per cut | -0.07 mm | up to 6 and up to 5 |

- Normal sum: u is opposite to the area-weighted sum of the outward normals of the cut faces of the particle.
- Largest least opening (proposed): u is the unit direction that makes the smallest opening over those faces as large as possible.
- (extra) On mesh b 2 particles of 190 have no direction that opens all their faces. Tets lower than about 1 mm invert at a gap of 1 mm with both rules.
- On a flat cut the smallest volume ratio of a tet after the gap is 0.833 (host-only check 3).

Not arguments: the copy colors and the pools (builder, 1.3).

### 4.4 Input types

None.

### 4.5 Data structures on the solver

Q = the largest number of tets at one vertex of the model, measured on the host at construction. For `add_soft_grid` it is 32: eight cells meet at a vertex, and a vertex of the centre tets holds four tets in each (`newton/_src/sim/builder.py:12529-12541`).

| Attribute (new unless marked) | dtype, shape | Meaning |
|---|---|---|
| `particle_adjacency` (exists) | struct | a new struct instance of the solver; `v_adj_tets` and `v_adj_tets_offsets` are new arrays, the other fields point at the arrays of `model.soft_mesh_adjacency_device` |
| `v_adj_tets` | int32 [2 x number of tet records] | tet record k of particle p is the pair at offset[p] + 2 k: (tet, corner); a spare has Q tet records `(cut_idle_tet, 0)` |
| `v_adj_tets_offsets` | int32 [P + 1] | offsets in int32 values, as today |
| `_cut_tet_record_group` | int32 [number of tet records] | scratch of the group search; a thread writes the tet records of its own vertex only |
| `_cut_pool_taken` | int32 [C] | spares taken from each pool |
| `_cut_overflow` | int32 [C + 2] | failed takes: one counter per pool, then one for triangle rows, then one for edge rows; the last one also counts fans that do not end within Q tets |
| `_cut_unsafe` | int32 [1] | new particles with no direction that opens all their cut faces |
| `_cut_particle_attached` | int32 [P] | 1 for a particle that an attachment names. Host, at construction |
| `_cut_particle_new` | int32 [P] | 1 for a particle that was split in this cut update: a new copy, or an original that gave tets to a copy |
| `_cut_offset` | vec3 [P] | start gap offset of a new particle |

The color of a particle is `model.particle_colors`.
The five suture attributes are `vbd.suture_grip`, `vbd.suture_mu`, `vbd.suture_puncture_force`, `vbd.suture_puncture_cos` and `vbd.suture_stiffness` (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:143-150`).
The normals of a cut face come from `particle_q` with the corner order of `_TET_FACES` (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:47`); a face with zero area is left out.

Write rule (inference). A thread of `split_cut_vertices` writes only the tet corners that hold its own vertex, the tet records of its own vertex and of the spares it took.
It reads only those tet records, `_tet_neighbors`, `_cut_face_side`, the rest matrices and the attributes of its own vertex.
Positions move in a later launch.
A tet group moves as a whole, so two tets joined through an uncut face always hold the same particle at each particle of that face.

### 4.6 Pseudo code

```
split_cut_vertices(state):                                               # (new) one thread per particle
    for v in all particles:
        if _cut_due == 0 or _cut_particle_dirty[v] == 0:
            continue
        _cut_particle_dirty[v] = 0
        if v is not ACTIVE or its mass is 0 or _cut_particle_attached[v] == 1:
            continue                                                     # a pinned or attached vertex stays shared
        for k in the tet records of v without the idle tet:
            _cut_tet_record_group[k] = k
        for sweep in range(Q):
            for k, (tet, corner) in the tet records of v without the idle tet:
                for face in the three faces of tet that hold v:
                    other = _tet_neighbors[tet, face]
                    if other < 0:
                        continue
                    j = the tet record of v whose tet is other           # a scan of the tet records of v; no read of tet_indices
                    _cut_tet_record_group[k] = min(_cut_tet_record_group[k], _cut_tet_record_group[j])
        groups = the sets of tet records with equal _cut_tet_record_group
        if there is one group:
            continue                                                     # cut front
        keep = the tet group with the most cut faces that hold v and have _cut_face_side == -1      # side A; tie: the tet group with the lowest tet id
        M = particle_mass[v]
        W = rest volume of the tets of v without the idle tet
        moved = 0
        pool = particle_colors[v] if particle_colors[v] < C else particle_colors[v] - C      # a copy takes from its own pool
        for g in the groups other than keep:
            s = take the next spare of pool                              # atomic add on _cut_pool_taken; F4.5
            if s is past the last spare of pool:
                _cut_overflow[pool] += 1                                 # atomic; g stays on v
                continue
            for tet, corner in g:
                tet_indices[tet, corner] = s
                write (tet, corner) into the next free tet record of s
                write (cut_idle_tet, 0) over that tet record of v
            particle_mass[s] = M * rest volume of g / W                  # F4.4: full_copy writes M
            particle_inv_mass[s] = 1 / particle_mass[s]
            copy particle_q, particle_qd, model.particle_q (the rest position), vbd.cut_force and the five suture attributes of 4.5 from v to s
            particle_flags[s] = particle_flags[v]
            _cut_particle_new[s] = 1
            moved += rest volume of g
        if moved > 0:
            particle_mass[v] = M * (W - moved) / W
            particle_inv_mass[v] = 1 / particle_mass[v]
            _cut_particle_new[v] = 1

compute_start_gap_offset(state):                                         # (new) one thread per particle; writes _cut_offset and _cut_unsafe
    for p in all particles:
        if _cut_particle_new[p] == 0:
            continue
        faces = the cut faces of the tets of p that hold p and whose other tet does not hold p
        if cut_gap_direction is "largest_least_opening":                 # F4.2
            u = largest_least_opening(faces)
        else:
            u = -normalize(sum of the area-weighted outward normals of faces)
            if cut_gap_direction is "normal_sum_first" and the smallest opening of u over faces is not positive:
                u = largest_least_opening(faces)                         # only where the normal sum does not open every face
        if faces is empty or the smallest opening of u over faces is not positive:
            _cut_offset[p] = 0
            _cut_unsafe[0] += 1                                          # atomic; p starts at distance zero on some faces (F4.6)
        else:
            _cut_offset[p] = 0.5 * cut_start_gap * u                         # F4.1

largest_least_opening(faces):                                            # (new) function; fixed cost: at most 3 Q faces, three nested loops
    best = no direction
    for each set of one, two or three faces of faces:                    # m = outward unit normal of a face
        for u in the candidates of the set:                              # one face: -m_i; two: +-normalize(m_i + m_j); three: +-normalize(cross(m_i - m_j, m_i - m_k))
            if the smallest opening of u over faces is larger than that of best:
                best = u
    return best

apply_start_gap_offset(state):                                           # (new) one thread per particle
    for p in all particles:
        if _cut_particle_new[p] == 1:
            particle_q[p] += _cut_offset[p]
            _cut_particle_new[p] = 0
```

Names that exist today: `tet_indices`, `particle_mass`, `particle_inv_mass`, `particle_flags`, `particle_q`, `particle_qd` (model and state arrays).
New names: the four functions and the arrays of 4.5.
The candidates of `largest_least_opening` are those of `direction_r2` in the second script of host-only check 3 (`checks/check2_3_impl_b.py` in the task folder).

### 4.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| F4.1 Start gap | `cut_start_gap` | one margin (D2). Half a margin. A tenth of a margin. A fraction of the smallest tet height | A6 |
| F4.2 Gap direction | `cut_gap_direction` | largest least opening (proposed; host-only check 3). Normal sum. Normal sum first, the other rule only where it leaves a crossed pair | A6 |
| F4.3 Gap in the rest shape | `cut_start_gap_in_rest` | off (D2). on: the rest shape moves too; no strain, a thin slice of material is lost. Not designed yet: it rewrites rest matrices that 6.5 does not save | A6 |
| F4.4 Mass of a copy | `cut_mass_rule` | share by rest volume (D11 d). Full copy | A7 |
| F4.5 Take of spares | `cut_take_mode` | atomic counters (D11 g). One device thread in order: the same ids in every run | A8d |
| F4.6 Particle with no opening direction | none (code change) | split, count in `_cut_unsafe`, start at distance zero (proposed). Do not split the vertex: needs the direction before the split, one more launch | none; question 4.8.2 (host-only check 3, extra: 2 of 190 particles on mesh b) |

### 4.8 Open questions for the reviewer

1. Line `s = take the next spare of pool` of 4.6 and limit 9: a vertex with three or more tet groups takes two or more spares from one pool, so copies of one color can lie on facing wound faces.
   Host-only check 3 (extra) found such vertices on both second meshes (2 of 48 split vertices on mesh a, 4 of 93 on mesh b), none on the grid.
   A copy that is split again gives the same state (inference).
   Accept as a stage-1 limit?
2. F4.6: split and count, or keep such a vertex shared?
3. Table 4.2, first row: the model's own vertex-to-tet table stays as built. Is a solver-owned table acceptable, or must the model table be the padded one?

## 5. The surface rows

### 5.1 Intent

The pattern in Newton is the skin of a tet body.
`add_soft_grid` adds one triangle row per open tet face and the skin edges of those triangles, all with zero stiffness (`newton/_src/sim/builder.py:12543-12565`).
The self-contact detector reads `tri_indices` and `edge_indices` at every refit and detection (`newton/_src/geometry/tri_mesh_collision.py:905-911`, `newton/_src/geometry/tri_mesh_collision.py:916-922`, `newton/_src/geometry/tri_mesh_collision.py:947-978`).
Full-surface rigid-soft contact holds every row in its pair lists from construction and copies the particle ids at detection time (`inputs/prior-work.md` 2.11).

So a wound face needs rows, not new code.
Each side of a new cut face takes one spare triangle row.
Each fan at a wound edge takes one spare edge row.
Every row with a source is then rewritten from the tet corners, so skin rows follow a split with no special case.

### 5.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Triangle rows | written by the builder, never changed (`newton/_src/sim/builder.py:12543-12559`) | derived: a row has a source (tet, three local corners) and is rewritten from `tet_indices` |
| Edge rows | written by the builder (`newton/_src/sim/builder.py:12562-12565`) | derived: a row has a source (tet, two local corners) |
| Wound faces | none | one spare triangle row per side of a cut face |
| Wound edges | none | one spare edge row per fan (D11 c) |

A fan is the run of tets around one mesh edge that are joined through uncut faces.
Each of the two ends of a fan is a cut face or a skin face.

### 5.3 Argument list

```
SolverVBD(
    ...,
    cut_surface_edges: bool = True,      # False: wound faces get no edge rows (fork F5.1)
)
```

Not arguments: the row capacities (builder, 1.3).

### 5.4 Input types

None.

### 5.5 Data structures on the solver

| Attribute (new) | dtype, shape | Meaning |
|---|---|---|
| `_cut_tri_source` | int32 [tri_count, 4] | (tet, three local corners in row order), or tet = -1. Host for skin rows; `take_surface_rows` for spare rows |
| `_cut_edge_source` | int32 [edge_count, 3] | (tet, two local corners in row order), or tet = -1. Host for skin rows; `take_surface_rows` for spare rows |
| `_cut_tet_face_row` | int32 [T, 4] | the triangle row whose source is this tet face, or -1 |
| `_cut_tet_edge_row` | int32 [T, 6] | the edge row whose source is this tet edge, or -1 |
| `_cut_tri_taken`, `_cut_edge_taken` | int32 [1] each | spare rows taken |

The sources of the skin rows are found on the host at construction, by matching vertex triples and pairs to tet faces and tet edges, as `build_tet_neighbors` matches faces (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:50-75`).
The source of a skin edge row is the tet with the lowest id that holds the edge.
Local faces and edges are numbered as `_TET_FACES` (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:47`) and `_TET_EDGE_PAIRS` (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:153`).

### 5.6 Pseudo code

```
take_surface_rows():                                                     # (new) one thread per (tet, face)
    for tet, face in all tet faces:
        if _cut_due == 0 or _cut_face_stamp[tet, face] != _cut_update_number or _cut_tet_face_row[tet, face] >= 0:
            continue
        row = take the next spare triangle row                           # atomic add on _cut_tri_taken; F4.5
        if row is past the last spare triangle row:
            _cut_overflow[C] += 1                                        # atomic; the edges below are still handled
        else:
            _cut_tri_source[row] = (tet, _TET_FACES[face])
            _cut_tet_face_row[tet, face] = row
        if not cut_surface_edges:                                        # F5.1
            continue
        for edge in the three edges of the face:
            end_tet, end_face, has_row, ended = walk_fan(tet, face, edge)
            if not ended:
                _cut_overflow[C + 1] += 1                                # atomic
                continue
            if has_row:
                continue                                                 # the fan has its edge row
            if _cut_face_stamp[end_tet, end_face] == _cut_update_number and (end_tet, end_face) < (tet, face):
                continue                                                 # the thread of the other end takes the row
            erow = take the next spare edge row                          # atomic add on _cut_edge_taken; F4.5
            if erow is past the last spare edge row:
                _cut_overflow[C + 1] += 1                                # atomic
                continue
            _cut_edge_source[erow] = (tet, _TET_EDGE_PAIRS[edge])
            _cut_tet_edge_row[tet, edge] = erow

walk_fan(tet, face, edge):                                               # (new) function; turns around edge, away from face
    has_row = _cut_tet_edge_row[tet, edge] >= 0
    for count in range(Q):                                               # a fan has at most Q tets
        face = the other face of tet that holds edge
        other = _tet_neighbors[tet, face]
        if other < 0:
            return tet, face, has_row, True                              # the fan ends at a cut face or a skin face
        face = the face of other whose neighbor is tet
        edge = the edge of other with the same two particles
        tet = other
        if _cut_tet_edge_row[tet, edge] >= 0:
            has_row = True
    return tet, face, has_row, False

refresh_triangle_rows():                                                 # (new) one thread per triangle row
    for row in all triangle rows:
        tet, c0, c1, c2 = _cut_tri_source[row]
        if _cut_due == 0 or tet < 0:
            continue
        ids = (tet_indices[tet, c0], tet_indices[tet, c1], tet_indices[tet, c2])
        other, face_other = the tet and the face across the face of this row if that face is cut, else -1     # from _tet_neighbors
        if other >= 0 and other < tet and _cut_tet_face_row[other, face_other] >= 0 and that face holds the same three ids:
            tri_indices[row] = the parking particle of the row, three times      # no particle of the face is split yet: the row of the other tet is enough
        else:
            tri_indices[row] = ids

refresh_edge_rows():                                                     # (new) one thread per edge row
    for row in all edge rows:
        tet, c0, c1 = _cut_edge_source[row]
        if _cut_due == 0 or tet < 0:
            continue
        edge_indices[row, 2] = tet_indices[tet, c0]
        edge_indices[row, 3] = tet_indices[tet, c1]
```

Names that exist today: `tet_indices`, `tri_indices`, `edge_indices` (model arrays); `_TET_FACES`, `_TET_EDGE_PAIRS` (suture branch, cited in 5.5). New names: the four functions and the arrays of 5.5.

### 5.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| F5.1 Wound edges | `cut_surface_edges` | on: one edge row per fan (D11 c). off: vertex-triangle contact only | A7 |
| F5.2 Equal edge rows | none (code change) | accept (proposed): at a wound edge whose two ends are both unsplit, two fans hold the same edge and the edge-edge force doubles (inference). Other option: one of the two rows goes back to its parking particle | A7 |
| F5.3 Viewer | none (code change) | `_log_triangles` passes `dynamic=True` when the model has cut capacity: one changed line (proposed). The example logs its own mesh | none |

### 5.8 Open questions for the reviewer

1. F5.2: is a doubled edge-edge force at a wound edge with two unsplit ends acceptable for stage 1?
2. F5.3: is one changed line in `viewer.py` acceptable?

## 6. Solver setup: settings, reset, what the suture gets

### 6.1 Intent

The pattern in Newton is `_init_suture_system`.
The solver constructor checks the model, builds every table and scratch array of the feature once from NumPy arrays of the model, and the step path only launches kernels (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3447-3461`).
It raises for more than one world (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3477-3478`).

`_init_cut_system` (new) does the same for the cut.
It checks the settings that a cut body needs, builds the shared tet-neighbor table, the solver-owned vertex-to-tet table and the row sources, and saves the arrays that `reset` restores.

### 6.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Topological contact filter | n-ring lists built in Python at construction (`newton/_src/geometry/tri_mesh_collision.py:786-819`), default threshold 2 (`newton/_src/solvers/vbd/solver_vbd.py:460`) | threshold 1 on a model with cut capacity: no list is built, and only the test for a shared vertex on the live index arrays stays (D11 a) |
| Rest-shape exclusion | default radius 0 (`newton/_src/solvers/vbd/solver_vbd.py:461`); the test is at `newton/_src/geometry/kernels.py:1650-1660` | must stay 0: the wound faces coincide in the rest shape |
| `particle_edge_parallel_epsilon` | default 1e-5 (`newton/_src/solvers/vbd/solver_vbd.py:458`), compared with the squared edge length in Warp, so edges of 3.16 mm or less act as points (`newton/_src/geometry/kernels.py:1797`) | cut scenes set a value below the squared shortest tet edge (D11 b); a warning otherwise |
| Rigid-soft contact | full-surface contact and truncation are off by default (`newton/_src/sim/collide.py:1661`, `newton/_src/solvers/vbd/solver_vbd.py:493`) | a model with a blade needs a solver-owned pipeline with full-surface contact, and the truncation on (D5) |
| Suture entry tet | best tet of the tet records of the contact particle (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:865-876`) | the loop skips `cut_idle_tet`: one test, one kernel argument, one launch input (D11 h) |
| Suture neighbor table | built in `_init_suture_system` (`vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3532-3537`) | built once in `_init_cut_system`, which runs first; the suture launch passes `_tet_neighbors` |
| `reset` | copies `model.particle_q` into the state for the selected worlds (`newton/_src/solvers/vbd/solver_vbd.py:3204-3218`) | a full particle reset first restores the saved arrays (D11 i) |
| Warp deterministic mode | forwarded to the kernel modules (`newton/_src/solvers/vbd/solver_vbd.py:495`, `newton/_src/solvers/vbd/solver_vbd.py:1037-1102`) | an error at construction when the model has cut capacity (D11 g) |
| Viewer | uploads triangle indices once (`newton/_src/viewer/viewer.py:3927-3943`) | fork F5.3. Stage-1 limit: one triangle appearance (one color and opacity for all triangles of the model); with more the viewer copies the indices (`newton/_src/viewer/viewer.py:3909-3919`) |

What the suture gets from the cut (inference from `inputs/prior-work.md` 1.5; the implementation run of the full demo with the gate arm made six through tunnels across the cut, 3.3):

- A cut face is a boundary in the shared tet-neighbor table, so the tet walk exits there.
- A copy has correct tet records, the suture attributes and the tissue radius, so a puncture can start at a wound face.
- Wound faces are triangle rows of particles, already in the pair lists of full-surface contact.
- A tunnel point stores a tet and four weights and reads `tet_indices` live, so it survives a split.

### 6.3 Argument list

```
SolverVBD(
    ...,
    cut_skip_empty_copy_colors: bool = False,   # True: skip a copy color with no active copy through wp.capture_if (fork F1.2)
)
```

The other cut arguments are in 2.3, 3.3, 4.3 and 5.3: `cut_armed`, `cut_update_interval`, `cut_force_measure`, `cut_force_records`, `cut_contact_range`, `cut_keep_side_labels`, `cut_start_gap`, `cut_gap_direction`, `cut_start_gap_in_rest`, `cut_mass_rule`, `cut_take_mode`, `cut_surface_edges`.
The switches exist for the ablations; a later round can drop the arms that lose.

Rules that `_init_cut_system` checks. Rules 1 to 10 and 12 apply to a model with cut capacity; rule 11 to a model with a blade. Each raises on a miss, except rules 9 and 12.

1. One world; every spare, parking particle and cuttable particle is in it.
2. No triangle and no edge has stiffness (`_is_tet_only_elasticity_model`, `newton/_src/solvers/vbd/solver_vbd.py:167`).
3. No Warp deterministic mode (constructor argument or `wp.config`).
4. `model.vbd.cut_force` exists.
5. The copy color layout is as `add_cut_capacity` made it.
6. `particle_enable_self_contact` is True; `particle_self_contact_margin` and `particle_self_contact_gap` are both given; the smallest distance between two particles on the parking lattice (spare particles and parking particles), read from `model.particle_q`, is larger than their sum.
7. `particle_topological_contact_filter_threshold` <= 1; `particle_rest_shape_contact_exclusion_radius` is 0; both external filter maps are None.
8. The radius of every spare and parking particle equals the radius of every cuttable particle.
9. Warning: `particle_edge_parallel_epsilon` is not below the squared shortest edge of a tet with a cuttable particle.
10. At the start, no ACTIVE particle is within self-contact margin plus self-contact gap of a spare particle or a parking particle, and no shape is within its rigid-soft detection distance of one.
11. A model with a blade has cut capacity, exactly one blade (stage 1), and a solver-owned collision pipeline with `enable_rigid_soft_full_surface_contact`.
    `rigid_soft_enable_dat` is True and `integrate_with_external_rigid_solver` is False (`newton/_src/solvers/vbd/solver_vbd.py:448`).
    The blade body is dynamic.
12. Warning: `cut_start_gap` is larger than the smallest height of a tet with a cuttable particle.

Not arguments: the capacities and the parking lattice (builder, 1.3).

### 6.4 Input types

None.

### 6.5 Data structures on the solver

| Attribute (new) | dtype, shape | Meaning |
|---|---|---|
| `_cut_saved_*` | same as the source | saved at construction: `tet_indices`, the two solver-owned tet arrays, `particle_mass`, `particle_inv_mass`, `particle_flags`, `model.particle_q`, `tri_indices`, `edge_indices`, `_tet_neighbors`, the two source tables, `_cut_tet_face_row` and `_cut_tet_edge_row` of 5.5 |

Not saved: `vbd.cut_force` and the suture attributes of spares (a stale value on a spare is never read; inference).
Not saved: the suture state (the `reset` of the suture branch does not touch it, `vbd-suture/newton/_src/solvers/vbd/solver_vbd.py:3146`).

The capacities of the existing contact buffers stay as they are: 32 pairs per vertex and 64 per edge (`inputs/prior-work.md` 2.5).
Critic: two closed flat wound faces need at most 12 pairs per vertex and 21 pairs per edge (`panel/critique-contact.md` in the task folder, finding 6).
The host reads the overflow flags and `_cut_overflow` after the cut, outside the graph.

### 6.6 Pseudo code

```
_init_cut_system(model, margin, gap, deterministic, the two external filter maps, the cut arguments of 2.3, 3.3, 4.3, 5.3 and 6.3):   # (new) in SolverVBD.__init__, before _init_suture_system; all values as the caller gave them
    if the model has a suture needle or cut capacity:
        _tet_neighbors = build_tet_neighbors(model.tet_indices)          # suture_kernels.py:50
    if the model has a blade and no cut capacity:
        raise an error                                                   # rule 11
    if the model has no cut capacity:
        return
    check the rules of 6.3                                               # raises
    Q = largest number of tets at one vertex
    particle_adjacency = a new struct: the model's tet records, plus Q tet records (cut_idle_tet, 0) for each spare particle
    allocate the arrays of 2.5, 3.5, 4.5 and 5.5
    fill _cut_face_stamp, _cut_shape_blade, _cut_tet_face_row, _cut_tet_edge_row and the tet column of the two source tables with -1
    find the source of every skin triangle row and skin edge row; write the two source tables, _cut_tet_face_row and _cut_tet_edge_row
    for p in the particles that an attachment names:
        _cut_particle_attached[p] = 1
    for blade in blades:
        _cut_shape_blade[cut_blade_edge_shape[blade]] = blade
        _cut_contact_range[blade] = edge radius + shape margin + soft_contact_gap + radius of the cuttable particles      # F3.2
    save the arrays of 6.5

reset(state, world_mask, flags):                                         # solver_vbd.py:3073
    run the existing reset up to the particle launch                     # unchanged; its checks can raise (solver_vbd.py:3157-3158, 3191-3195)
    if the model has cut capacity:
        if world_mask is not None:
            raise an error                                               # the cut is undone for the whole model only
        if flags select the particle positions and state.particle_q is not None:     # the test of solver_vbd.py:3190
            copy every _cut_saved_* array back
            fill _cut_face_stamp with -1
            zero the counters, the side labels, the dirty flags and the new flags
    run the existing reset from the particle launch on                   # unchanged; reset_particle_state, solver_vbd.py:3205

_solve_particle_iteration(state_in, state_out, contacts, dt):            # solver_vbd.py:4302
    for color in range(number of color groups):                          # solver_vbd.py:4334
        if cut_skip_empty_copy_colors and color >= C and _cut_pool_taken[color - C] == 0:   # F1.2; a device test through wp.capture_if
            continue
        run the existing loop body                                       # unchanged

update_suture_needle_state(its arguments of today, idle_tet):            # suture_kernels.py:801; one thread per needle; one new argument, -1 without cut capacity
    for needle in needles:
        run the existing body up to the entry search                     # unchanged
        for tet in the tet records of the contact particle:              # suture_kernels.py:868
            if tet == idle_tet:
                continue                                                 # (new)
            run the existing loop body                                   # unchanged
        run the existing body after the entry search                     # unchanged
```

Names that exist today: `build_tet_neighbors` (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:50`),
`reset` (`newton/_src/solvers/vbd/solver_vbd.py:3073`),
`_solve_particle_iteration` (`newton/_src/solvers/vbd/solver_vbd.py:4302`),
`update_suture_needle_state` (`vbd-suture/newton/_src/solvers/vbd/suture_kernels.py:801`).
New names: `_init_cut_system`, the arrays of 6.5.

### 6.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| F6.1 BVH rebuild | closed | in every captured frame (D8, Anka, 2026-11-10, terminal) | A8b for its cost |
| F6.2 Contact filter | `particle_topological_contact_filter_threshold` (exists) | 1 (D11 a). Lists rewritten on the device | A9 |
| F6.3 Short-edge threshold | `particle_edge_parallel_epsilon` (exists) | below the squared shortest tet edge (D11 b). 1e-5, the value of the suture demo | A9 |
| F6.4 Base branch | none | the suture branch merged with main (proposed). Main only, without the stitch | A10 |

### 6.8 Open questions for the reviewer

1. F6.4: does the cut build on the suture branch merged with main?
2. F6.4: the merge brings the two-sided contact damping of #5182 from main, default on; the suture branch does not have it (`inputs/prior-work.md`, note on the two bases). Which damping value does stage 1 use?
3. Rule 2 of 6.3 forbids stiff skin triangles on a model with cut capacity. Acceptable for stage 1?

## 7. The stage-1 scene

### 7.1 Intent

The pattern in Newton is the suture example (`vbd-suture/newton/examples/vbd/example_vbd_suture.py`).
It builds the tissue with `add_soft_grid` and holds it with grippers through particle attachments.
It drives the needle as a dynamic body on a kinematic holder through a D6 joint with drives (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:440-477`).
It captures one frame of 10 substeps (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:615-627`) and writes the kinematic poses on the host once per frame (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:592-611`).
Its captured frame has no BVH rebuild; `example_cloth_twist.py` calls `rebuild_bvh` once per frame before its substeps (`newton/examples/cloth/example_cloth_twist.py:240-243`).

The stage-1 scene is that example with one block and a blade before the needle: cut through, pull apart, stitch, release (D3).
Its pass tests are in the list "Checks and ablations".
The implementation has this scene as `newton/examples/vbd/example_vbd_cut_suture.py` on its branch. It passes its final test with the gate arm of F3.4 (implementation run, 3.3).

### 7.2 Old and new

| | Today (file:line) | New |
|---|---|---|
| Wound | two blocks 4 mm apart (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:359-378`) | one block; the blade cuts it through on a grid plane |
| Colors of the wound sides | split per block in the example (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:257-263`) | copy colors from `add_cut_capacity` |
| Tools | needle on a holder, grippers (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:440-477`) | plus a blade on a holder of its own |
| Captured frame | substeps only (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:615-627`) | `rebuild_bvh`, then the substeps (D8) |
| Wound before the stitch | 4 mm from the start (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:49-66`) | the grippers pull the two parts 4 mm apart after the cut, more than the gate radius of 2.9 mm (`inputs/prior-work.md` 1.6) |

### 7.3 Argument list

The scene values, as the parameter table of the suture example (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:49-66`). A value marked (first guess) is not calibrated.

```
block:        28 x 10 x 4 cells of 3 mm (84 x 30 x 12 mm), add_soft_grid, tissue values of the suture example
cut plane:    the grid plane x = 42 mm from the block corner
grippers:     attachments on every particle 12 mm or more from the cut plane, as the suture example
blade:        one capsule, radius 0.45 mm (first guess), half height 17 mm, along y; density 7800 kg/m^3 (0.17 g); cut_direction -z
blade holder: a kinematic body without shapes; D6 joint to the blade, six driven axes, gains of the needle holder
holder path:  down at 4 mm/s (first guess) from 2 mm above the block to 3 mm below it, then up
needle:       needle and thread wait away from the cut plane until the pull is done
pull:         each gripper moves 2 mm away from the cut plane in 0.5 s
stitches:     three, as the suture example; then one gripper lets go
capacity:     max_vertices 32, max_faces 128, max_edges 384, park_spacing 3 mm, particle_radius 0.5 mm
parking:      a lattice box of 33 mm, 0.1 m to the side of the block (first guess)
tissue:       vbd.cut_force = 0.5 N (first guess) on the block particles
solver:       the suture example values, with rigid_soft_enable_dat True, full-surface contact True,
              filter threshold 1, particle_edge_parallel_epsilon 1e-7, cut_start_gap = margin = 1 mm
frame:        60 frames per second, 10 substeps, 20 iterations
```

Below the block the blade edge has no contact record (inference: 3 mm is more than the detection range of 1.95 mm; not run), so the stroke ends there; the way up through the closed wound is a second stroke with new side labels (limit 20).
With the capsule of 7.6 the direction of the blade edge is +y, and the blade plane normal n = cross(d, e) is +x (hand calculation). So side B is the part with larger x, and the originals stay on the part with smaller x.

Not arguments: the stitch values, which stay those of the suture example.

### 7.4 Input types

None.

### 7.5 Data structures on the model

None beyond chapters 1 to 6.
With the values of 7.3 the model has 1595 tissue particles, 128 spare particles and 644 parking particles; 5600 tets and the idle tet; 1728 skin triangle rows and 256 spare ones; 2592 skin edge rows and 384 spare ones.

### 7.6 Pseudo code

```
Example.__init__(viewer, args):                                          # (new); as example_vbd_suture.py
    SolverVBD.register_custom_attributes(builder)                        # solver_vbd.py:2553; adds vbd.cut_force
    add_soft_grid(the block of 7.3)                                      # builder.py:12384
    add the grippers, the needle, its holder and the thread              # as the suture example
    blade = add_body()
    edge = add_shape_capsule(blade, xform = a rotation of -90 degrees about body x (the capsule axis is then body +y), radius = 0.45 mm, half_height = 17 mm)   # builder.py:9395
    blade_holder = add_body()                                            # kinematic, no shapes; as the needle holder
    add a D6 joint with six drives from blade_holder to blade            # as example_vbd_suture.py:440-477
    add_cut_blade(blade, edge, cut_direction = (0, 0, -1))
    color()                                                              # builder.py:13252
    add_cut_capacity(max_vertices = 32, max_faces = 128, park_position = the parking place of 7.3, park_spacing = 3 mm, particle_radius = 0.5 mm)
    model = finalize()                                                   # builder.py:14885
    set vbd.cut_force of the block particles
    solver = SolverVBD(model, collision_pipeline = a pipeline with full-surface contact, the solver values of 7.3)

Example.simulate():                                                      # (new); the captured frame; as example_vbd_suture.py:615, with rebuild_bvh first
    solver.rebuild_bvh(state_0)                                          # solver_vbd.py:5612; once per frame (D8)
    for substep in range(sim_substeps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, sim_dt)             # ends with _update_cut_blades
        state_0, state_1 = state_1, state_0

Example.step():                                                          # (new); host, once per frame; as example_vbd_suture.py:629
    _set_kinematics(sim_time)                                            # as example_vbd_suture.py:592; poses of the blade holder, the needle holder and the grippers
    launch the captured frame
    if the phase "cut" just ended:
        read _cut_overflow, _cut_unsafe and the self-contact overflow flags      # outside the graph
```

Names that exist today: `register_custom_attributes` (`newton/_src/solvers/vbd/solver_vbd.py:2553`),
`add_soft_grid` (`newton/_src/sim/builder.py:12384`),
`add_body` (`newton/_src/sim/builder.py:5989`),
`add_shape_capsule` (`newton/_src/sim/builder.py:9395`),
`color` (`newton/_src/sim/builder.py:13252`),
`finalize` (`newton/_src/sim/builder.py:14885`),
`rebuild_bvh` (`newton/_src/solvers/vbd/solver_vbd.py:5612`),
`clear_forces` (`newton/_src/sim/state.py:179`),
`_set_kinematics` (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:592`).
New names: `Example.__init__`, `Example.simulate`, `Example.step`, `add_cut_blade`, `add_cut_capacity`.

### 7.7 Forks

| Fork | Switch | Options | Ablation entry |
|---|---|---|---|
| F7.1 Needle across a narrow wound | `suture_gate_slack` (exists) | pull beyond the gate radius (D3). Smaller slack. Accept a late second puncture. A gate with a side test | A11 |
| F7.2 Blade shape | none | one capsule (proposed for stage 1). A capsule edge with a box body above it | none |
| F7.3 Force test or armed mode in the tests | `cut_armed` | force test in the demo, armed in the small capture test (proposed). Force test in both | none |

### 7.8 Open questions for the reviewer

1. Rows "blade" and "tissue" of 7.3: do you have better first values than a cut force of 0.5 N and an edge radius of 0.45 mm?
2. F7.2: is a blade that is one capsule (a wire) enough for stage 1?
3. Row "holder path" of 7.3: the holder moves at a set speed and the joint drive presses the blade. Do you want a set force instead?

## Limits

Stage 1 has these limits. Each one is a decision (D), a proposal with its question, or a consequence of the method.

1. Tools are boxes and capsules; skin triangles and edges have zero stiffness; one world; no cut after the first puncture (D11 j). Nothing checks the tool shapes or the order of cut and puncture.
2. One blade; two blades need one set of side labels per blade (question 2.8.2).
3. The cut follows tet faces. It is flat while the blade plane stays less than a quarter cell (0.75 mm) from a grid plane; at exactly a quarter cell the result hangs on a tie; elsewhere the wound faces are stepped by up to one cell (host-only check 2).
4. In a stepped wound the straight blade edge meets the steps of the wound faces. It must push them aside to go down, and it may stop there. Not run (A12).
5. The cut leads the blade edge axis by about the contact range plus one tet: about 5 mm in stage 1 (host-only check 2).
   This holds only when the blade edge reaches the cut front.
   Run P0 measured that it does not: the blade edge rests on the wound lips, 3.29 mm or more from every cut-front vertex (`checks/p0/p0_notes.md` 6.1). Fork F3.4.
6. The cut force is compared with a total force, which depends on the contact length of the blade edge (D6).
   Below the first tet row the blade force comes from the wound lips (measured, run P0): all of it sits on the 22 lip vertices, and it reaches 0.5 N at 0.73 mm and 1.0 N at 1.17 mm of holder travel (`checks/p0/p0_notes.md` 6.1 and 6.2).
   The start gap shakes the wound for 0.2 to 0.33 s, with particle speeds up to 439 mm/s (measured, `checks/p0/p0_notes.md` 6.4).
7. A pinned or attached vertex is not split and stays shared (D11 e).
8. Capacities hold for the whole run. When a pool is empty the vertex stays shared and a counter grows (D11 f).
9. A vertex with more than one copy has copies of one color, also on facing wound faces (question 4.8.1).
10. A particle with no opening direction starts at distance zero on some faces (F4.6). A tet lower than about one start gap can start inverted (4.3).
11. The copy colors double the color passes from the first frame (D9).
12. Nothing keeps tissue or tools out of the parking lattice after the start (D10).
13. Edges shorter than the square root of `particle_edge_parallel_epsilon` act as points in edge-edge contact (D11 b).
14. No Warp deterministic mode; the ids of copies differ from run to run (D11 g).
15. `reset` undoes the cut for the whole model only, together with the particle positions; it also undoes host edits of mass, flags and `model.particle_q` made after construction (6.6).
16. Inside a frame the BVH is only refit. It is rebuilt at the start of the next frame, so a cut in the middle of a frame meets a tree that was built before it (D8).
17. Filter threshold 1 and exclusion radius 0 hold for every soft body of the model (D11 a).
18. One particle radius for all cuttable tissue; no `color()` and no merge after `add_cut_capacity` (1.3).
19. One triangle appearance on the model, or the wound does not show in the viewer (6.2).
20. The way up of the blade through the closed wound is a second stroke. Not run (A15).
21. The captured frame has an even substep count, so the two states are back in place after a replay (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:624`).
22. Interior tissue has no surface rows, so below the skin the blade meets particles only. A blade that is off the particle planes by more than the force range meets no resistance below the skin (inference from the contact code, not run).

The design covers the free cut: a stepped cut, a partial cut, a vertex with three or more tet groups and a second stroke.
Stage 1 tests none of them with a stitch.
A stepped cut and a partial cut each get a small test, cut and contact only (D3).

## Checks and ablations

Host-only checks that ran (D12), with their files:

1. Colors: C = 4 for the stage-1 block; numbers in 1.5 (`checks/check1_color_count.txt`).
2. Cut rule replay: numbers in 3.3; the (extra) drift runs support F3.1 (`checks/check2_3_compare.md`).
3. Start gap direction: numbers in 4.3 (same file).

Simulation runs that ran:

1. P0 (D14): the wedge probe on main, with no cut code. Numbers in 3.3 and limits 5 and 6 (`checks/p0/p0_notes.md`; review `checks/p0/p0_review.md`, verdict: Q1 trust, Q2 trust with limits).
2. Implementation runs of the three arms of F3.4 on the stage-1 scene, after Anka's order to get the demo ready. Numbers in 3.3 (logs in `impl/logs/` of the task folder).

Pass tests of the stage-1 scene (D3):

1. The state is finite.
2. Exactly the 80 faces of the cut plane are cut. The pools gave 18, 10, 12 and 15 copies.
3. The tet mesh has two parts, joined by no particle. The sum of `particle_mass` is unchanged.
4. `_cut_overflow`, `_cut_unsafe` and the self-contact overflow flags are zero.
5. For every particle that is on no cut face, the penetration of the blade edge (edge radius + particle radius - distance from the particle to the blade edge axis) stays below 0.5 mm, the bound of the needle test (`vbd-suture/newton/examples/vbd/example_vbd_suture.py:801`) (D5).
6. From the cut to the start of the pull, at every tenth frame: each original and each copy lies on the side of the nearest triangle of the facing wound face that the outward normal of that triangle points to.
7. Two through tunnels per stitch; the stitch tests of the suture example hold.
8. Capture test on a small armed scene.
   One captured step is replayed across one cut advance.
   The plain run does the same step.
   The positions, read through the tet corners, agree within 1e-6 m.
   `_tet_neighbors` and `_cut_pool_taken` are equal.
   The test reads through the tet corners because the ids of copies can differ (D11 g).

Two small tests (D3), cut and contact only:

- Stepped cut: the blade turned 15 degrees about the vertical axis (yaw; 96 cut faces in host-only check 2).
  Until A12 shows that the blade edge can go through all tet rows, the test ends after the first cut advance.
  Pass until then: one part; the vertices of the cut front are not split; pass tests 1, 4 and 6.
  Pass after A12: two parts; pass tests 1, 3, 4 and 6.
- Partial cut: the holder stops after two tet rows. Pass: one part; the vertices of the cut front are not split; pass tests 1, 4 and 6.

Runs. P0 ran (D14).
The others need the implementation and Anka's word.
The implementation exists now (status line); of these entries only the three arms of F3.4 ran with it, on the stage-1 scene (3.3).

| Entry | Run | Decides |
|---|---|---|
| P0 (ran) | A block built with the top tet row of the cut plane already split and one start gap open; a capsule on a holder drive is lowered into it; the contact force is read per contact record. Result: the blade edge rests on the two wound lips (3.3) | F2.2, F3.2, F3.4, limits 5 and 6 |
| A1 | Jitter of the closed wound faces and time per iteration: copy colors against the color of the original | F1.1 |
| A2 | Time per iteration before the first cut, with and without the skip of empty copy colors | F1.2 |
| A3 | Blade force at a cut advance against the contact length of the blade edge | F2.1 |
| A4 | Lead of the cut against `cut_update_interval` | F2.3 |
| A5 | Blade force at each tet row, split into wound face contact records and cut front contact records; lead of the cut; three values of `vbd.cut_force` | F2.2, F3.2, F3.3 |
| A6 | Start gap values and directions under a load that presses the wound faces together, on the flat cut and on the stepped cut of the small test: vertices on the wrong side, peak speed after a split, strain energy. The rest shape arm (F4.3) runs after it is designed | F4.1, F4.2, F4.3 |
| A7 | Crossed pairs under the same load with and without wound edge rows; one run with one attached vertex on the cut; mass of each original and each copy for the two mass rules | F4.4, F5.1, F5.2 |
| A8a | Time of the eleven launches of one cut update with no cut, and of the update that cuts the first tet row | R1 |
| A8b | Time of one BVH rebuild for the stage-1 model | F6.1 |
| A8c | Detection time with all spares parked, with the lattice 0.1, 1 and 10 m away, and with several rows per place | F1.3, F1.4 |
| A8d | Atomic take against ordered take: time of one cut update | F4.5 |
| A9 | Rest contact on the uncut body with threshold 1; jitter of closed wound faces for the two short-edge thresholds | F6.2, F6.3 |
| A10 | The suture tests and demo on the merged base, with both damping values | F6.4 |
| A11 | Two-block suture demo with a distance of 1, 2 and 3 mm between the blocks: where the second puncture starts | F7.1 |
| A12 | Stepped wound and straight blade edge: blade force, holder force and smallest tet volume ratio per tet row, blade half a cell off and 15 degrees off | limit 4 |
| A13 | Three runs from one state. Run 1: the cut kernels in the captured frame. Run 2: the same kernels without capture. Run 3: a new model built with the topology after the cut | two risks of the method: in-place edits of model arrays under graph replay, and data built at construction that is stale after a cut |
| A14 | Cut, reset, compare the model arrays with a new model, run again | D11 i |
| A15 | Blade out: blade force, holder force, `_cut_advance` and the number of cut faces per frame while the blade goes up through the closed wound | limit 20 |
