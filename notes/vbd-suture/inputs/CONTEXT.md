# Suture simulation in SolverVBD

The language for the needle-and-thread suturing feature of SolverVBD: a curved rigid needle with a thread sews two tetrahedral soft bodies. This context exists so that the design, the code and the discussion with Anka use one word per concept.

## Language

**Tunnel**:
The material-space path that the needle tip has traced through one tissue body, from its entry point to its exit point (or to the current tip position while the tip is still inside).
_Avoid_: channel, hole, needle path

**Virtual tunnel**:
A tunnel represented only by tunnel points and their constraint rows; the tet mesh keeps its topology. Decided 2026-11-05 (ADR 0001).
_Avoid_: real tunnel, carved tunnel (the rejected alternative)

**Tunnel point**:
One sample point of the tunnel, stored as a tet index and four barycentric weights so it moves with the tissue; the tip creates one every tunnel point spacing, and each carries one constraint row that holds the needle or thread where it passes through the point.
_Avoid_: anchor, tunnel anchor, tunnel node, constraint point, sliding point, fixation point (Newton already uses "anchor" for attachment and contact points)

**Tunnel point row**:
The constraint of one tunnel point: a bilateral compliant-ALM row in the two directions transverse to the rod tangent, hard (infinite stiffness) by default, with a per-point stiffness value so a soft wall is a parameter change. Decided 2026-11-05.
_Avoid_: anchor row, tunnel constraint, sliding constraint, tube constraint (the deferred unilateral variant)

**Transverse multiplier**:
The two-component ALM multiplier of a tunnel point row; its magnitude is the normal force that caps the friction of that tunnel point.
_Avoid_: normal force (ambiguous with contact), lambda_n

**Grip force per unit length** (`g`, N/m):
A per-tissue parameter that stands for the radial pre-compression of tissue around a rod thinner than the hole it sits in; it enters only the friction cap of each tunnel point row as `mu * (transverse force + g * tunnel point spacing)`, in the primal projection and in the dual projection alike, and adds no force on its own. Decided 2026-11-05: default small and non-zero, 1 N/m; `g = 0` gives the unchanged body-body law. Meaning: `g = p_grip * 2 pi r_rod`, so 1 N/m on a 0.15 mm radius thread is about 1 kPa of tissue squeeze; the hold force of a thread of in-tissue length `L` at zero side load is `mu * g * L` (0.3 x 1 N/m x 15 mm = 4.5 mN, about 25 times the weight of a 20 cm thread, far below a deliberate pull of 0.5 N or more).
_Avoid_: minimal compression, pre-load, grip model

**Tunnel point spacing** (`ds_k`, m):
The tunnel length one tunnel point represents: half the distance to the previous point plus half the distance to the next, one-sided for the entry and exit points.
_Avoid_: anchor spacing, node spacing, resolution

**Needle**:
One dynamic rigid body whose centerline is a general curve given as a `Rod`-style polyline (a circular arc in the first demo), carrying short capsule shapes along it and one separate tip shape. Decided 2026-11-05: simulated, not kinematic; the shape is not hard-coded, and a later flexible needle is a chain of ROD segments like the thread.
_Avoid_: needle rod, needle chain, tool, arc (the demo shape, not the concept)

**Needle holder**:
A kinematic body with a prescribed pose, joined to the needle by a D6 joint with drives on all six axes, so the needle follows the holder with finite stiffness and the tissue can push back. Decided 2026-11-05.
_Avoid_: virtual coupling, target body, gripper (a grasped needle is a later scene variant)

**Thread**:
The suture thread as a `ModelBuilder.add_rod` rod: capsule rigid bodies joined by ROD joints with stretch, shear, bend and twist. Decided 2026-11-05: no particle thread.
_Avoid_: suture (ambiguous with the whole procedure), string, cable (the Newton example name, same object)

**Swage joint**:
The ROD joint between the needle tail and the first thread segment, with the needle-side joint frame's local +Z along the arc tangent so bending is continuous across it.
_Avoid_: eye, tail joint, attachment (reserved for particle-body attachments)

**Puncture**:
The event that opens a tunnel: at the end of a step the tip shape's summed rigid-soft contact normal force exceeds the puncture force `F_puncture` while the tip direction is within `alpha_max` of the inward surface normal; once opened, the tunnel persists regardless of force. Decided 2026-11-05.
_Avoid_: penetration (reserved for an unwanted contact failure), insertion (the whole phase), pop

**Armed mode**:
A switch that replaces the force test by the first tip contact with the tissue surface, for scripted demos and tests.
_Avoid_: geometric puncture, auto-puncture

**Contact gate**:
The per-pair switch that makes the solver ignore a rigid-soft contact record between a gated tissue particle and a shape of the same needle group, for forces, duals, export and DAT truncation alike; records are still detected and stored. Decided 2026-11-05 (ADR 0002).
_Avoid_: contact filter (the existing geometric feature filter), collision mask, exclusion

**Needle group**:
The set of shapes of one needle and its thread, identified by one bit shared by `shape_contact_gate` on those shapes and `particle_contact_gate` on gated particles.
_Avoid_: tool group, suture group

**Gate radius** (`r_gate`, m):
The distance from the tunnel polyline (through its tunnel points, plus the segment to the current tip), within which tissue particles are gated: needle radius + shape margin + soft contact gap + particle radius + slack for one update interval of tip travel.
_Avoid_: gate set, tunnel tets

**Needle state**:
The per-needle device record that drives the tunnel: phase (OUTSIDE, PIERCING, THROUGH), the tet that contains the tip, the current tunnel and its tunnel point count, and the summed tip contact force of the substep. Decided 2026-11-05: the state machine runs on the device only, with one thread per needle for the sequential parts, so a captured substep graph stays valid and no readback is needed; a constructor argument `tunnel_update_interval` sets how many substeps pass between updates (default 1).
_Avoid_: controller (the SOFA host-side name), puncture manager

**Tunnel point spacing rule**:
A new tunnel point is appended when the tip has travelled `ds_tunnel` inside the tissue since the last point; default `ds_tunnel` is half the mean tet edge length of the tissue, clamped to 0.5 to 2 mm.
_Avoid_: constraintDistance (SOFA name)

**Suture needle**:
The authored unit: one needle body, its tip shape, its centerline points, the thread rod it carries, and the capacities of its tunnels and tunnel points, added with one `ModelBuilder` method. Decided 2026-11-05: the rows, the needle state and the gate are SolverVBD internals in one kernel file, like attachments and contacts; no companion class.
_Avoid_: suture tool, suturing system, suture module

**Tissue suture parameters**:
Per-particle SolverVBD custom attributes that let two tissues differ: grip force per unit length, tunnel friction coefficient, puncture force and puncture angle.
_Avoid_: global suture settings, solver flags (there is no enable flag; a suture needle in the model enables the feature)

**Bite**:
One pass of the needle through one tissue piece, from its puncture to its exit; one tunnel per bite.
_Avoid_: pass, stitch (a stitch is the thread loop left by two bites)

**Clamped end**:
The thread's free end fixed to a static body as the first milestone's stand-in for a knot, so pulling the needle end tightens the stitch.
_Avoid_: knot, fixed end, pinned end

**Rod curve**:
The one continuous centerline made of the needle polyline followed by the thread polyline, stored as segments that each name the body (or two bodies) that move them; a tunnel point constrains its closest point on this curve. The needle may be one body for all its segments or, later, one body per segment.
_Avoid_: needle path, suture path

**Capped ALM friction**:
The solver's compliant-ALM contact friction: the tangential force rho_t times the tangential offset since the step start plus the tangential multiplier, projected onto the Coulomb disk of radius mu times the normal force (`evaluate_rigid_contact_from_collision`), and the multiplier advanced and projected onto the disk of radius mu times the normal multiplier in the dual step (`_compliant_contact_dual_step`), both in `rigid_vbd_kernels.py`; friction carries no material stiffness.
_Avoid_: IPC friction (the smooth law of soft contacts), viscous friction, stick-slip state machine

**Tunnel point friction**:
Capped ALM friction applied to a tunnel point row: one tangent direction instead of a tangent plane, the transverse row force in the role of the normal force, the transverse multiplier in the role of the normal multiplier, and the same rho_t rule (`_contact_auto_tangent_rho`). Decided 2026-11-05: same law, same functions, no separate friction model.
_Avoid_: anchor friction, tunnel friction model, grip model

## Relationships

- A **Tunnel** belongs to exactly one tissue body and has one entry point and at most one exit point
- A **Tunnel** opens with exactly one **Puncture** and owns an ordered list of **Tunnel points**
- A **Tunnel point** has exactly one **Tunnel point row**; the row constrains the **Rod curve** transversally and applies **Capped ALM friction** along the tangent, capped by its **Transverse multiplier**
- One **Rod curve** can pass through several **Tunnels** (two tissue pieces, several stitches)
- The **Needle** follows exactly one **Needle holder**; the **Thread** starts at the needle tail through one **Swage joint**
- The **Rod curve** is the **Needle** polyline followed by the **Thread** polyline
- A **Bite** opens exactly one **Tunnel**; two bites through two tissue pieces plus the **Clamped end** form the demo's stitch
- A **Suture needle** owns one **Needle group**, one **Needle state** and up to a fixed number of **Tunnels**; its rows read the **Tissue suture parameters** of the pierced particles
- A **Contact gate** applies to one **Needle group** and the particles within the **Gate radius** of its **Tunnels**

## Scope of the first milestone (decided 2026-11-05)

In scope: two tissue pieces, one suture needle with one thread in a single world, puncture, tunnel growth and exit through both pieces in one or several bites, sliding with capped ALM friction, pulling the thread to close the gap, all inside a CUDA-graph-capturable substep.

Out of scope: knot tying, tissue tearing and thread cut-through, a visible hole in the simulation mesh, cutting the thread, a flexible needle, several worlds, several needles in one world, a grasped needle (gripper or robot).

## Tunnel lifecycle (decided 2026-11-05)

1. Entry: the puncture places the entry **Tunnel point** on the pierced surface face, inside that face's tet; the **Needle state** becomes PIERCING.
2. Growth: while PIERCING the tunnel ends at the tip; a new point is appended per the **Tunnel point spacing rule**.
3. Retraction while PIERCING: if the tip retreats past the last point by more than `ds_tunnel`, that point is removed; the tunnel never extends beyond the tip before the exit.
4. Exit: when the tet walk crosses a boundary face, the exit point is placed on that face and the state becomes THROUGH; from then on the tunnel is fixed, whatever the needle does.
5. Row on and off: a point's row is on from its creation while some part of the needle or thread passes through the point, and off for good once the thread's free end has passed it; each point caches its last closest place on the rod curve and searches locally from there.
6. Finish: when the last row switches off the state becomes OUTSIDE and the tunnel is finished; a new puncture always opens a new tunnel.
7. Gate: every update, a per-particle kernel recomputes the needle group's gate bit from scratch from the distance to the tunnel polyline and the tip segment.

## Example dialogue

> **Dev:** "When the thread end slides past the last **Tunnel point**, does the **Tunnel** disappear?"
> **Domain expert:** "It is finished. The **Tunnel** stays as data for display, its points have nothing to hold, so their rows are off. A new puncture always opens a new **Tunnel**; we never reuse one."
> **Dev:** "And the friction at a tunnel point is the smooth soft-contact law?"
> **Domain expert:** "No. It is **Capped ALM friction**: the tangential multiplier is projected onto the Coulomb disk each iteration, like body-body contact."

## Flagged ambiguities

- "anchor", "tunnel node" and "constraint point" were all used for the same thing, and "anchor" was not understood in discussion. Resolved: **Tunnel point**.
- The friction cap at zero side load was zero (free sliding). Resolved: the cap adds **Grip force per unit length** times **Tunnel point spacing**; the law and functions stay those of body-body contact.
- "path" was used for the tip trajectory, the tunnel and the needle centerline. Resolved: **Tunnel** for the material path, **Rod curve** for the needle-plus-thread centerline.
