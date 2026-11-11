# Options for the VBD fast-cut design

- Purpose: candidate designs for a cut in tetrahedral tissue that is fast, fits a captured CUDA graph, and gives full self contact between the new faces. Anka chooses; `design.md` follows the choice.
- Date: 2026-11-10, second run. Draft 2, with the corrections of the final check (`panel/final-check-regression.md`). Nothing is implemented. No simulation was run.
- Decision D1 (Anka, 2026-11-10): option 1, the free cut. See `decisions.md`.
- How draft 2 was made: four independent designers each proposed a full design from a different starting point (`panel/idea-*.md`). Seven critics attacked draft 1 against the code (`panel/critique-*.md`): no blocker, 40 major and 36 minor findings. This draft has their corrections.
- Facts: a tag such as [P 2.7] names a paragraph of `prior-work.md`. A tag such as [C contact] names a critique file. Some critics checked geometry and counts with small Python scripts (no Newton, no Warp, no solver). I mark those numbers "geometry check". Every other sentence is a proposal.
- Names that do not exist yet carry (new) the first time.

## Summary

1. All four designers chose the same method: split the mesh along existing tet faces and give the far side copies of the vertices. I recommend it (option 1). It changes no elasticity or contact kernel. It adds one line to one suture kernel (1.10).
2. Option 2 is the same method on a cut sheet that is planned in the mesh. It is simpler and cheaper, and it is level with option 1 on both hard requirements. Option 1 wins only if the cut place must be free at run time.
3. Option 3 (tet copies, the XFEM form) and option 4 (remove tets) are not recommended. Section 8 gives the reasons.
4. The hard part is not the topology edit. It is contact at birth: two faces created at the same place do not collide in this solver [P 2.7]. Every option needs an answer to that.

## Words

| Word | Meaning |
|---|---|
| Blade | The kinematic rigid body that cuts. |
| Blade plate | A flat rectangle fixed in the blade frame. One side of it is the blade edge. |
| Armed | A switch that the host sets. The blade cuts only while it is armed. One armed period is one **stroke**. |
| Cut face | One interior tet face that the cut separates. It becomes two triangles, one per side. |
| Side A, side B | The material on the two sides of the blade plate. The plate normal points from A to B. |
| Wound face | All new triangles of one side. |
| Cut front | The line where the cut ends inside the tissue. Vertices on it are not split. |
| Spare particle | A particle that exists from the start and does nothing. |
| Copy | A spare particle after a cut gave it the place of a vertex for one tet group. |
| Tet group of a vertex | Tets around one vertex that are still joined through uncut faces. |
| Tet list of a vertex | The vertex-to-tet table entries of one vertex. One entry is a **record**. |
| Idle tet | One tet with zero stiffness. Unused records point at it. |
| Copy color | A second color group for each original color. Copies live there. |
| Parking lattice | A set of places away from tissue and tools where unused spares sit. |
| Start gap | The distance between the two wound faces right after the split. |

## 1. Parts that all options share

**1.1 Capacity before capture.** A captured graph fixes every array size and launch size [P 2.9, 2.10]. The builder adds spare particles, spare triangle rows and spare edge rows; the user gives the capacities, as the suture does with `max_tunnels` [P 1.2]. A spare particle has the tissue radius, the tissue world and zero triangle and edge stiffness from the start, because three solver values are read once at construction: the suture gate radius, the rigid-soft truncation budget and the tiled-kernel form [C suture-and-demo, C capture-and-step]. When a pool is empty, the vertex stays shared, nothing is written, and a device counter `cut_overflow` (new) grows. The host reads it after the cut.

**1.2 Parking.** Self contact ignores the ACTIVE flag, and box or capsule tools with full-surface contact ignore it too [P 2.4]. Proposed: unused spares sit on a parking lattice, each place more than one query radius from the next, away from the tissue and from every tool path. An unused triangle row holds one parking particle three times and then acts as that point. Parking particles for rows are never used as copies. The lattice must stay inside the scene bounds (about 1 m for 3 mm elements), or the BVH loses resolution [C contact]. Nothing enforces the keep-out zone; a host check at construction can test it. Fork F1.

**1.3 Where the update runs.** `cut_update(state)` (new) is a public solver method like `rebuild_bvh`. The captured frame calls `cut_update`, then `rebuild_bvh`, then the substeps. Reasons [C capture-and-step]: the host writes the pose of a kinematic blade once per frame; the new rows meet a fresh BVH; the update runs between two steps, so no solver-private array needs a fix and no velocity is created. All of it is kernels with fixed launch sizes: no readback, no allocation. Draft 1 placed the update at the suture call site; that was wrong, because that site is before the velocity update. Note: four shipped examples already rebuild the BVH once per frame inside the captured frame [P 2.9]. Fork F2.

**1.4 What decides a cut.** The earlier work tested whole tets against a plane fixed in the world and missed tissue that moved [P 5.7]. Proposed rule, in the blade frame, with current positions:

- At the first update of a stroke every tet gets a side label: side B if its centroid is on or beyond the plate plane, else side A. The label does not change during the stroke. While the blade is not armed, the labels are cleared.
- A face is cut when its two tets have opposite labels and the segment between their centroids passes through the blade plate.
- The mark is kept for good.

Geometry check on the `add_soft_grid` mesh [C trigger-and-blade]: with the blade on a grid plane the rule selects exactly the two triangles of each cell face, and the body falls into two parts. The rule needs no rest data and no stored blade path. Its only history is the side label. Limits: the blade must not move farther than the depth of its plate between two updates; and after the labels are set, a tet whose centroid drifts across the plate plane before its faces are cut leaves an uncut face. The reserve is the distance of the nearest centroid from the plane: 0.75 mm with the blade on a grid plane, less off it (`panel/final-check-regression.md`). Fork F3 holds the other form (follow the path of the blade edge relative to the material). A force threshold needs blade contact and a rule from a contact force to an interior face; it is not designed.

**1.5 How the blade passes through tissue.** Proposed for stage 1: the blade shapes have no contact with particles (shape flag `COLLIDE_PARTICLES` off). This works today for box and capsule shapes, also with full-surface contact on [C trigger-and-blade]. The blade then cuts where its plate is and pushes nothing. A blade with contact needs its own gate: the suture gate cannot be reused as it is [P 1.6], and a critic lists five things that such a gate needs. Fork F4.

**1.6 Contact filters.** The cut body runs with `particle_topological_contact_filter_threshold = 1` and rest-shape exclusion radius 0. Then the only filter is the live shared-vertex test, which follows every index edit [P 2.6]. Geometry check [C contact]: on the uncut `add_soft_grid` body with 3 mm cells this creates no pair inside the 2 mm query radius at rest. `particle_edge_parallel_epsilon` must be set for the mesh scale (1e-7 or less for edges down to 1 mm), or edges of 3.16 mm and less act as points [P 2.8]. Fork F5.

**1.7 The start gap.** New faces need a positive gap with each vertex on its correct side [P 2.7]. Proposed: after the split, each of the two particles moves by half of `cut_start_gap` (new) into its own material, along the summed outward normals of its own cut faces. With the demo values a gap of one margin (1 mm) gives no contact force at birth. Smaller gaps push: a wound vertex has about 8 pairs, so 0.5 mm gives about 2 N per vertex and 0.1 mm about 10 N. For every value the faces end about one margin apart [P 1.7]. Geometry check [C contact]: on a grid-plane cut all split corners open by the full gap for 0.1 to 1 mm, no vertex is on a wrong side, and no tet inverts (smallest volume ratio 0.833 at 1 mm). Off a grid plane this is not proven: the check found corners that are born crossed for 6 of 12 random blade directions; the smallest such tilt was 14 degrees. Fork F6.

**1.8 Colors.** A copy can take the color of its original and the elastic coloring stays valid [P 4.2]. But then every vertex and its copy face each other in one color pass, which is the state that jittered in the suture work [P 1.8]. Proposed: each original color has one copy color with its own pool of spare particles, built by one builder call after `color()`. No color array changes at run time. Every tet keeps four different colors after any sequence of cuts (proof in [C colors]). For one stroke, the split vertices of wound face B have copy colors and all other vertices keep theirs. Price: each copy color adds one full color pass per iteration, also before the first cut. The suture record gives about 50 percent more time per iteration for a doubled color count [P 1.8]; the number for copy pools is not measured. The color count of the stage-1 block is not known [P 7]; a hand-made coloring with four colors exists for `add_soft_grid` [C colors]. Forks F7 and F8.

**1.9 Contact slots.** The self-contact buffers hold 32 pairs per vertex and 64 per edge by default; overflow is silent [P 2.5]. Geometry check [C contact]: two closed flat wound faces of 3 mm triangles at query radius 2 mm need at most 12 and 21. The sizes are stated capacities of the design, and the host reads the overflow flag after the cut.

**1.10 What the suture needs.**

- The tet-neighbor table is shared. The cut writes a negative value on both sides of a cut face, so the needle walk exits there [P 1.5].
- Every copy has correct records in the vertex-to-tet table. The puncture entry search is the only reader without a stiffness test; it must skip the idle tet (one line in a suture kernel) [C suture-and-demo].
- A copy takes mass, rest position and the five suture attributes of its original.
- Cut faces are triangles of simulation particles, so contact records keep particle ids.
- Where the needle crosses, the wound must be open by more than the gate radius (2.9 mm in the demo) [P 1.6]. A through cut on a flat sheet opens by the jaw travel, so a 4 mm pull leaves 1.1 mm, the same as in the two-block demo. A partial cut does not open that far at the needle depth (hand estimate in [C suture-and-demo]). So stage 1 cuts through the whole section. Below the gate radius the code suggests a late second puncture, not a failure; nobody ran it. Fork F9.
- Stage 1 order: no cut after the first puncture. A particle that an attachment names is not split.

**1.11 Viewer and reset.** The viewer uploads triangle indices once [P 2.12]; the model triangles must be logged as dynamic, which is a small change in viewer code or in the example. `reset` copies `model.particle_q` into the state. The split wrote the rest position of the original into the copy, so a reset would put each copy exactly on its original while the tets stay split. That is distance zero, which this contact cannot handle [P 2.7]. So `reset` must restore the edited arrays from copies taken at construction. This is a required part, not a fork.

## 2. Option 1: split along existing tet faces where the blade passes

**Idea.** The cut surface is the set of tet faces chosen by rule 1.4. A vertex is split when the cut faces around it separate its tets into two or more tet groups. One group keeps the vertex; each other group gets a copy. No tet is added, removed or reshaped. The Vertex Block Descent paper names this method for tearing [P 4.2].

**Data layout.** T = tets, P = particles (spares included).

| Array (new unless marked) | dtype, shape | Filled by |
|---|---|---|
| `tet_neighbors` (today private to the suture) | int32 [T, 4] | solver constructor; cut kernel writes -(2 + other tet) |
| `tet_cut_side` | int8 [T] | `label_tets`: set at the first update of a stroke, cleared while the blade is not armed |
| `cut_update_number` | int32 [1] | first launch of `cut_update` adds 1; a Python number would be fixed at capture |
| `cut_face_side`, `cut_face_stamp` | int8 [T, 4], int32 [T, 4] | cut kernel: side of this tet at this face; update number of the cut (starts at -1) |
| `tet_edge_row` | int32 [T, 6]: the edge row that this tet edge is the source of, or -1 | constructor for skin rows; `claim_surface_rows` for spare rows |
| `particle_dirty`, `particle_new` | int32 [P] | cut kernels |
| `cut_offset` | vec3 [P] | scratch for the start gap |
| spare particles, one pool per copy color | part of the particle arrays | builder call after `color()` |
| `copy_color`, pool start, pool size, pool count | int32 [colors] each | builder; the count by an atomic counter |
| vertex-to-tet table with room for spares | existing arrays, rebuilt once | solver constructor: original lists plus R idle records per spare |
| idle tet | one tet, zero stiffness, four parking particles of its own | builder |
| `tri_source` | int32 [tri_count, 2]: (tet, face) or (-1, -1) | constructor for skin rows; cut kernel for spare rows |
| `edge_source` | int32 [edge_count, 2]: (tet, local edge) or (-1, -1) | constructor for skin rows; cut kernel for spare rows |
| `cut_overflow` | int32 [colors + 2] | cut kernels |
| copies for reset | same shapes as the edited arrays | solver constructor |

R is the largest number of tets at one vertex of the body: 32 for `add_soft_grid` [C split-and-adjacency]. A record of the idle tet is skipped by both elasticity kernels, the attachment kernel and the suture row metric before any other read [P 2.2, C split-and-adjacency].

**Per-cut update.** Nine launches, sizes 1, T, 4 T, P, P, P, 4 T, `tri_count`, `edge_count`. Every thread returns at once when it has nothing to do.

```
cut_update(state):                                    # (new) public solver method; first call of the captured frame, before rebuild_bvh
    advance_update_number()                           # dim 1
    label_tets(state)                                 # dim T
    mark_cut_faces(state)                             # dim 4 T
    split_cut_vertices(state)                         # dim P
    place_new_particles(state)                        # dim P, twice
    claim_surface_rows()                              # dim 4 T
    refresh_surface_rows()                            # dim tri_count, then dim edge_count

label_tets(state):                                    # one thread per tet
    for tet in all tets:
        if the blade is not armed:
            tet_cut_side[tet] = 0
        elif tet_cut_side[tet] == 0:
            tet_cut_side[tet] = side B if the centroid is on or beyond the plate plane, else side A

mark_cut_faces(state):                                # one thread per (tet, face)
    for tet, face in all tet faces:
        other = tet_neighbors[tet, face]
        if the blade is not armed or other < tet:
            continue                                  # skin, already cut, or the thread of the other tet handles this face
        if the side labels of tet and other are equal:
            continue                                  # a label is set once per stroke; a centroid on the plate plane counts as side B
        if the segment between the two centroids does not pass through the blade plate:
            continue
        tet_neighbors[tet, face] = -(2 + other)       # negative: the suture walk exits here; the value still names the other tet
        tet_neighbors[other, face of other] = -(2 + tet)
        cut_face_side[tet, face] = label of tet; cut_face_side[other, face of other] = label of other
        cut_face_stamp of both = number of this update
        particle_dirty[each corner of the face] = 1

split_cut_vertices(state):                            # one thread per particle
    for v in all particles:
        if particle_dirty[v] == 0:
            continue
        particle_dirty[v] = 0
        if v is pinned or named by an attachment:
            continue                                  # the vertex stays shared
        groups = tet groups of v: its records without the idle tet, joined where tet_neighbors >= 0
        if there is one group:
            continue                                  # cut front
        keep = the group with the most side A cut faces at v      # lowest tet id decides a tie
        for g in the other groups:
            s = take the next spare of the pool of copy_color[color of v]     # atomic add first, then test the result
            if s is outside the pool:
                cut_overflow[that color] += 1         # g stays on v; nothing is written
                continue
            for tet, corner in g:
                tet_indices[tet, corner] = s
                write the record (tet, corner) into the tet list of s
                write the idle tet over that record in the tet list of v
            particle_mass[s] = particle_mass[v] x rest volume of g / rest volume of all tets of v; same rule for v; write particle_inv_mass of both
            copy position, velocity, rest position and suture attributes from v to s; set ACTIVE on s
            particle_new[s] = 1; particle_new[v] = 1

place_new_particles(state):                           # one thread per particle; first launch reads positions, second writes them
    for p in all particles with particle_new[p] == 1:
        n = sum of the outward normals of the cut faces of the tets of p
        cut_offset[p] = -0.5 cut_start_gap normalize(n)       # into its own material
    for p in all particles with particle_new[p] == 1:
        particle_q[p] += cut_offset[p]; particle_new[p] = 0

claim_surface_rows():                                 # one thread per (tet, face)
    for tet, face in all tet faces with cut_face_stamp == number of this update:
        tri_source[next spare triangle row] = (tet, face)
        for edge in the three edges of the face:
            walk the tets around the edge, through uncut faces, to the next cut or skin face
            if no tet of that fan has an entry in tet_edge_row for this edge:      # the lower (tet, face) claims when both ends are new
                edge_source[next spare edge row] = (tet, edge); tet_edge_row[tet, edge] = that row

refresh_surface_rows():                               # one thread per triangle row, then one per edge row
    for row in all triangle rows with a source:
        tri_indices[row] = the three corners of that tet face, read from tet_indices
    for row in all edge rows with a source:
        edge_indices[row, 2:4] = the two corners of that tet edge, in a fixed order
```

**Topology without a rebuild.** Tet corners, records, masses and flags are edited in place and read live [P 2.1, 2.2]. Each thread writes only the corners and lists of its own vertex and of the spares it took; a separate launch moves the particles [C split-and-adjacency]. Triangle and edge rows are derived data: one kernel rewrites every row from the tet corners, so skin rows follow a split with no special case. Color arrays never change. BVH item counts never change. Stale after a cut and not maintained: the vertex-to-triangle and vertex-to-edge tables. That is harmless while skin stiffness is zero and tools are boxes or capsules [P 2.11].

**Self contact.** The two triangles of a cut face are ordinary surface triangles in the existing detector; every particle is already a query vertex. Start gap 1.7, filters 1.6, colors 1.8, slots 1.9. Each fan of tets around a wound edge gets one edge row, so edge-edge contact acts too (fork F12). At the cut front the vertices are shared and the gap goes to zero; a copy still meets the facing triangle that holds a front vertex, and that pair is the contact near the front. Closed wound faces rest about one margin apart, as any two faces do [P 1.7].

**Graph-capture plan.** `cut_update` and `rebuild_bvh` are the first two calls of the captured frame. Every launch size is a host constant. The cut kernels live in their own module (new file `cut_kernels.py`). Stage 1 raises an error at construction when the solver runs in a Warp deterministic mode, because the atomic counters compile differently there [C capture-and-step]. The slot order of the counters differs from run to run; it changes ids, not physics, up to round-off. Fork F10.

**Stage-1 numbers** (block of 28 x 10 x 4 cells of 3 mm, blade on the middle grid plane; counts by hand and by geometry check): T = 5600, so 22 400 threads in `mark_cut_faces`. The cut has 80 cut faces and 55 split vertices. It needs 55 copies, 160 triangle rows and 240 edge rows. Proposed capacities: 64 spare particles per copy color, 256 triangle rows, 384 edge rows, about 80 parking particles.

**Limits.**

- The cut follows tet faces. It is flat while the blade plate stays within a quarter cell (0.75 mm) of a grid plane in the material. Outside that band the wound faces are stepped by up to one cell.
- The opening lags the blade by up to one tet.
- The start gap is proven by geometry only for a flat cut (1.7).
- A vertex is split into two in stage 1. A copy that is split again, a vertex with three or more tet groups, and two strokes that cross are defined (copy takes the copy color of its original) but lose the disjoint colors at that vertex.
- A pinned or attached vertex is not split and stays a bridge.
- Tools are boxes and capsules. Skin triangles and edges have zero stiffness. One world.
- Each copy color costs one color pass per iteration from the first frame.

**Risks.** (1) No Newton test edits an index or adjacency array between graph replays [P 2.9]. (2) The start gap on a stepped cut can be born crossed. (3) The edge-row rule is the most intricate kernel. (4) The keep-out zone of the parking lattice is not enforced. (5) The cost of the copy colors is not measured. (6) No run with a contact-free blade exists [P 5.8].

**What stage 1 proves and what it leaves open.** With the blade on a grid plane every vertex has two tet groups, so the demo exercises the same path as option 2. It does not test the group search on a general cut, crossing strokes, or contact on a stepped wound.

## 3. Option 2: release a cut sheet that was planned in the mesh

**Idea.** Before `finalize` the user marks a sheet of tet faces where a cut may happen. The builder prepares everything for it. At run time the blade releases sheet faces. A vertex is split when all sheet faces around it are released.

**Data layout.** As option 1, with fixed slots in place of pools:

| Array (new) | dtype, shape | Filled by |
|---|---|---|
| `sheet_face` | int32 [F, 2]: (tet, face) of side A | builder |
| `sheet_face_rows` | int32 [F, 2]: its two triangle rows | builder |
| `sheet_vertex_copy` | int32 [V]: the one spare of each sheet vertex | builder |
| `sheet_vertex_corners` | int32 [V + 1] offsets and (tet, corner) pairs | builder: the corners that move to the copy |
| `sheet_vertex_open`, `sheet_vertex_total` | int32 [V] | cut kernel counts released faces; builder sets the total |
| edge rows of both sides | fixed rows | builder |

**Per-cut update.** Four kernels, launch sizes F, V, V, then the row refresh. `mark_cut_faces` tests only sheet faces, with its own rule: the centroid segment passes through the blade plate, within a tolerance between plate and sheet (a new constant). `split_cut_vertices` has no group search: when the count reaches the total, it applies the stored list.

**Topology without a rebuild.** As option 1, but every write target is known at build time. No atomic counter: slot use is the same in every run.

**Self contact.** As option 1. The sheet is exact, so a planar sheet gives flat wound faces and the start gap is on proven ground. Colors can be planned: inference, the copies of one flat sheet need about three extra colors, not one per original color.

**Graph-capture plan.** As option 1, with simpler kernels and no atomics.

**Limits.** A cut can happen only on a planned sheet. A blade that is off the sheet by more than the tolerance releases nothing. Two crossing sheets need a rule for the vertices on both.

**Risks.** Risks 1 and 6 of option 1. One of its own: the blade path and the sheet must agree.

**Relation to option 1.** Both give the same wound when the mesh has faces on the blade path. Option 2 has fewer colors, fixed slots and planned edge rows. Option 1 lets the cut place be chosen at run time.

## 4. Option 3: copy each cut tet (XFEM in the phantom-node form)

**Idea.** The cut surface follows the blade through the inside of the tets. Each cut tet becomes two overlapping copies. Each copy owns the material on one side and gets new particles for its far-side corners. The wound faces are an embedded surface: their vertices are points inside tets, not particles [P 3.1 to 3.4].

**Two forms.**

- 3a, embedded surface: contact acts on points given as (tet, weights). This needs a new detection and force path. The earlier work did this and got three index spaces [P 5.6, 3.5].
- 3b, surface particles tied to their tets: the wound faces are triangles of extra particles, each tied to a point of its tet by a new constraint. The existing contact path then applies and contact records keep particle ids. The paper that describes such ties says that springs alone do not give the wanted behavior (`study/verify-literature.md`, claim C3). Not proven.

**Data layout.** Spare tets (one per cut tet), spare particles (up to four per cut tet before sharing), a material share per copy or full copies (a fork: full copies add mass and stiffness [P 3.2]), and the surface: for 3a points, triangles and their own contact buffers; for 3b spare particles, spare rows and tie records.

**Per-cut update.** Classify the corners of each crossed tet. Fill a spare tet row. Re-point far-side corners in both copies. Share the new particles between neighbor copies (the same one-ring group search as option 1). Scale materials and masses. Create the surface points and triangles.

**Topology without a rebuild.** Possible with spare rows, with more arrays than option 1: tet rows, materials, tie records.

**Self contact.** 3a: a second contact system. 3b: the existing one, plus the tie. In both forms the start gap can sit in the weights of the surface points, with no strain (inference).

**Graph-capture plan.** The copy step fits fixed arrays. The contact path of 3a must be written for capture from the start.

**Limits.** The cut front still moves by whole tets [P 3.3]. Overlapping copies break the suture tet walk, which tests whole tets; this holds for both forms [P 3.5].

**Risks.** Material shares near zero: the earlier work saw a mass fraction of 5.64e-14 [P 3.3]. About twice the spare particles of option 1. Changes in the suture kernels. The most new code of all options.

## 5. Option 4: remove the tets that the blade plate crosses

**Idea.** A tet with corners on both sides of the blade plate gets zero stiffness. The open faces of its neighbors get triangle rows. The two sides then share no vertex, so no vertex is split [C fairness-and-completeness].

**Data layout.** `tet_removed` int8 [T] (new); spare triangle and edge rows with source tables as in option 1. No spare particles.

**Per-cut update.** Three kernels, launch sizes T, 4 T, then the row refresh. Mark crossed tets. Write zero into their stiffness [P 2.4] and a negative value into the neighbor table around them. Point their records at the idle tet, or the suture puncture can choose a removed tet. Claim rows for the open faces.

**Topology without a rebuild.** Materials and surface rows only.

**Self contact.** The faces start one tet apart, so the zero-distance problem does not exist. The two faces do not match. There are no copies, so facing vertices share colors: the jitter case of the suture work, with no remedy inside this option [P 1.8].

**Graph-capture plan.** As option 1, with fewer kernels.

**Limits.** Volume is lost: one layer of tets, about 3 mm of the block. The mass of a removed tet has no rule, because tets store no density [P 2.1]. With the blade exactly on a grid plane no tet is crossed and nothing is removed. Closing the wound with stitches leaves strain, because material is missing [P 4.6].

**Risks.** Low for the cut. High for the stitch and for jitter of the closed wound.

## 6. Families that are not options

- **Subdivision** of cut tets: 4 to 17 new tets per cut tet, tiny tets, a new coloring per cut [P 4.5]. It does not fit a fixed capacity.
- **Bonded faces from the start** (cohesive or DiSECt form): both sides have their own vertices before the run and bonds hold them. A cut only switches bonds off, which is ideal for capture. But the bonded faces sit at distance zero from the first frame: no contact force, no truncation plane, and gaps of a few micrometres freeze vertices [P 2.7]. It also needs stiff bonds in a block-descent solver. Option 2 reaches the same end state without bonds.

## 7. An extension of option 1: move vertices onto the blade plate before the split

Before a stroke, vertices near the plate plane move onto it, in the rest shape and in the current shape, so the faces that split later lie on the blade path. The fidelity-first designer worked this out (`panel/idea-fidelity-first.md`). It needs two new constants (how near, how flat a tet may get), it thins the tets next to the wound, which conflicts with the margin rule of 1.6, and where a move is refused the cut stays stepped. The literature reports flat tets at about 5 percent of the cut faces on one test mesh [P 4.4]. Later switch, fork F13.

## 8. Comparison

| | 1 Split along faces | 2 Planned sheet | 3 Tet copies | 4 Remove tets |
|---|---|---|---|---|
| Cut place | free, along mesh faces | planned sheet only | free, through tets | free, one tet layer wide |
| Wound faces | flat on a fitting mesh, else stepped | exact | smooth | stepped, do not match |
| Volume | kept | kept | kept | one layer lost |
| Solver kernels changed | none (one line in a suture kernel) | none (same line) | 3a: new contact path; 3b: new tie; suture walk | none (same line) |
| Start of self contact | start gap; proven by geometry for a flat cut only | start gap on a flat sheet | start gap in the surface weights | one tet apart |
| Closed wound | disjoint colors on the wound faces | the same, fewer colors | as option 1 | shared colors: jitter risk |
| Extra color passes per iteration | one per original color | about three (inference) | one per original color | none |
| Spare capacity for the stage-1 cut | 55 copies, 160 triangle rows, 240 edge rows | the same, fixed slots | about 2 x the copies, plus spare tets | rows only |
| Slot use the same in every run | no (atomic counters) | yes | no | no |
| Suture | table write, correct records | the same | kernel changes | table write; gap of one tet to close |
| Fits fixed arrays (by reading, not tested) | yes | yes | yes for copies; open for contact | yes |
| New code (my estimate) | medium | small | large | small |

## 9. Forks

"Proposed" is my default, not a decision. Each fork with a switch is also an ablation entry.

| Fork | Switch (new) | Options | Decided by |
|---|---|---|---|
| F1 Unused spares | `cut_park_mode` | parking lattice, no kernel change, keep-out zone not enforced (proposed); a skip rule for unused rows in self-contact detection and in the rigid-soft face and edge kernels | measured cost of the skip rule |
| F2 Call site of the update | none | first call of the captured frame (proposed); host code between two graph launches; last call of each step | Anka |
| F3 Cut rule | `cut_rule` | blade plate at one instant with side labels (proposed); path of the blade edge relative to the material | geometry replay, then a run |
| F4 Blade contact | none in stage 1 | no contact (proposed); own gate, later stage | Anka |
| F5 Contact filter | existing argument | threshold 1 (proposed); filter lists rewritten on the device | ablation on rest contact |
| F6 Start gap | `cut_start_gap`, `cut_start_gap_in_rest` | one margin (proposed); half; a tenth; the same gap also in the rest shape (no strain, about 1 percent of material lost) | ablation: crossed vertices, peak speed after the split |
| F7 Color of a copy | `cut_copy_color` | copy colors (proposed); color of the original (expected to jitter); a small set chosen on the device | ablation: jitter against time |
| F8 Copy color passes before the first cut | `cut_gate_copy_colors` | always run (proposed); skip an empty copy color with `wp.capture_if` | measured time |
| F9 Needle across a narrow wound | `suture_gate_slack` exists | pull beyond the gate radius (proposed); smaller gate slack; accept a late second puncture; a gate with a side test | gap sweep in the two-block suture demo |
| F10 Slot hand-out | `cut_slot_mode` | atomic counters (proposed); one device thread in order, same ids every run; fixed slots in a declared region | Anka |
| F11 Mass of a copy | `cut_mass_rule` | share by rest volume (proposed); full copy | ablation |
| F12 Edge-edge contact on wound faces | `cut_surface_edges` | on (proposed); vertex-triangle only | ablation with crossed-face count |
| F13 Smooth cut | `cut_snap` | off (proposed); section 7 | later |

## 10. Recommendation

Option 1, with the stage-1 demo on a grid plane.

- On the two hard requirements options 1 and 2 are level; option 2 is simpler and has fewer colors. What decides for option 1 is a free cut place. My record of the first run says that you chose the free cut ("we need to do it properly", 2026-11-10). Question 1 asks if that stands.
- Both share one data layout. Option 2 is option 1 with the group search done at build time, so the stage-1 demo of option 1 is also a test of option 2.
- Option 4 stays as a baseline for ablations. Option 3 is not recommended: 3a is a second contact system, 3b is unproven, and both break the suture walk.

## 11. Checks and runs that the design needs

Nothing here runs without Anka's word.

Host-only checks (no solver):

1. Color count of the stage-1 block from `builder.color()`, and the colors of the vertices on the cut plane.
2. Geometry replay of rule 1.4 on the stage-1 mesh with smooth random deformations: blade on a grid plane, a quarter and half a cell off, tilted 3, 10 and 15 degrees. Pass: no uncut face between the two sides.
3. Start-gap direction scan on one unstructured tissue mesh: count corners that are born crossed.

Runs:

4. Three runs from one state: cut kernels inside the captured frame; the same kernels without capture; a new model built with the topology after the cut. The first pair tests capture, the second pair tests stale copies.
5. Start gap: the F6 values under a closing load; count vertices on the wrong side, read the peak speed after the split.
6. Colors: jitter of closed wound faces for the F7 rules, with time per iteration before and after the cut, and with the gate of F8.
7. Detection time with all spares parked, against the same scene without spares; one BVH rebuild for the tissue size.
8. Two-block suture demo with gaps of 1, 2 and 3 mm (no cut code needed): does the second puncture come, and where.
9. One stage-1 stroke with the contact-free blade. Pass: two parts, each above 10 percent of the mass, `cut_overflow` zero.

## 12. Questions for Anka

Asked one at a time in the grill.

1. Do your statements from the first run still stand? (cut then stitch; free cut along existing faces; partial cut supported; BVH rebuild on the host)
2. Is a start gap by a position offset acceptable (fork F6)?
3. Blade in stage 1: no contact with tissue (fork F4)?
4. Copy colors at the price of extra color passes (forks F7, F8)?
5. Unused spares: parking lattice, or a skip rule in shared kernels (fork F1)?
6. Stage 1 as a through cut on a grid plane, with a 4 mm pull before the stitch?
7. Which of the checks in section 11 may run?
