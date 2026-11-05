# Decisions: suture simulation in SolverVBD

## Requirements as stated

- R1 (Anka, 2026-11-05, task prompt): "Sew two pieces of elastic body together with a needle and a thread, simulated with SolverVBD in Newton. A curved needle with a thread (string) attached and following it. Only the needle tip can poke through the body. The rest of the needle and the thread cannot penetrate; they collide normally. After the tip passes through, the entry creates a tunnel through the body. The tunnel can be real (geometry changes) or virtual (constraints that keep the needle and thread on the path). The needle and thread can slide along the tunnel, with friction against the body. They leave the body when the needle pokes through again (exit point). The thread then stays in the tunnel between entry and exit and can be pulled to close the wound."
- R2 (Anka, 2026-11-05, chat): "The first demo is sewing 2 elastic cuboids and tighten the thread to make them close."
- R3 (Anka, 2026-11-05, chat): "note that the needle can be curved in the future" (the centerline is a general curve; a flexible needle is a later ROD chain).
- R4 (Anka, 2026-11-05, chat): "there is a design review using PR skill; use that; once the design is finished, start implementing and create the demo and record it right away for me. dont wait for me"
- R5 (Anka, 2026-10-19 and 2026-11-05, standing rule): pseudo code as structured statements with real names and the control flow written out; every array with dtype and shape.

## Decision table

| ID | Item | Where | Decision |
|---|---|---|---|
| D1 | Tunnel representation | chapter 1 | decided 2026-11-05 (chat, question 1): virtual tunnel of barycentric tunnel points with constraint rows; no mesh change (ADR 0001) |
| D2 | Row form | chapter 1, 1.7 "Row form" | decided 2026-11-05 (question 2): hard bilateral line in the two transverse directions; stiffness stored per point; unilateral tube deferred |
| D3 | Friction law | chapter 1 | decided 2026-11-05 (question 3): the existing capped ALM contact friction, same functions and rho_t rule, one tangent direction; cap = mu (transverse force + g ds); g default 1 N/m |
| D4 | Needle driving | chapter 3 | decided 2026-11-05 (question 4): dynamic needle held by a kinematic holder through a D6 joint with six drives |
| D5 | Thread | chapter 3 | decided 2026-11-05 (question 5): `add_rod` capsule chain with ROD joints; ROD swage joint at the needle tail |
| D6 | Puncture trigger | chapter 2 | decided 2026-11-05 (question 6): tip contact normal force above F_puncture with the tip direction within alpha_max of the inward normal; armed mode as a switch |
| D7 | Contact gate | chapter 2 | decided 2026-11-05 (question 7): at consumption through `_soft_contact_force_enabled` plus `apply_rigid_soft_truncation`; gate set by distance to the tunnel polyline and the tip segment (ADR 0002) |
| D8 | Tunnel point creation | chapter 2 | decided 2026-11-05 (question 8): device only, one thread per needle, `tunnel_update_interval` substeps (default 1); spacing rule half the mean tet edge clamped 0.5 to 2 mm |
| D9 | Lifecycle rules 1 to 7 | chapter 2 | decided 2026-11-05 (question 9): entry, growth, retraction while piercing, exit, row on and off, finish without reuse, gate recompute |
| D10 | Code location and authoring | chapter 3 | decided 2026-11-05 (question 10): inside SolverVBD, one kernel file, `add_suture_needle`, per-tissue custom attributes; single world first |
| D11 | Scope | all | decided 2026-11-05 (question 11): see inputs/CONTEXT.md "Scope of the first milestone" |
| D12 | Demo | chapter 3 | decided 2026-11-05 (question 12): two cuboids with a 4 mm gap, clamped thread end, holder script, defaults and four checks as listed in chapter 3 |
| D13 | Tests | implementation plan | decided 2026-11-05 (question 13): eight tests as listed in the task folder's `decisions.md` |
| D14 | Closest-place search window | chapter 1, 1.7 | pending: one segment on each side of the cached segment per refresh, or a wider window |
| D15 | Transverse rho particle support | chapter 1, 1.7 | pending: material-reduced inverse masses of the four vertices (as attachments do for one particle) or plain inverse masses |
| D16 | Grip reference | chapter 1, 1.7 | pending: g authored per unit length, or a pressure times the local rod radius |

## Decisions taken in review

- 2026-11-05, Anka, chat (questions 1 to 13 of the grill-with-docs interview): all rows D1 to D13 above. The chapters are drafted from these decisions. Per R4 the implementation starts from the drafted chapters without a review round; later review comments are applied to the code and the chapters together.

## Decisions taken during implementation (2026-11-05, Claude, recorded for Anka's review)

| ID | Item | Decision |
| --- | --- | --- |
| I1 | Needle state semantics | The needle state is only "growing" or "not growing"; a tunnel that exited is THROUGH on its own. Reason: in the demo the tip enters block B while block A's tunnel still holds the needle, so a puncture must be allowed while earlier tunnels are fixed. Chapter 2 wrote OUTSIDE, PIERCING, THROUGH per needle; the code and CONTEXT.md now say OUTSIDE, PIERCING per needle and THROUGH per tunnel. |
| I2 | D15 transverse rho | Plain inverse masses of the four vertices (no material reduction) for the first implementation; the row tests converge (residual under 0.1 mm in 20 steps). The material-reduced variant stays a fork. |
| I3 | `custom_attributes` on `add_suture_needle` | Not implemented: it would need a new attribute frequency. The method takes `label` only. |
| I4 | Tunnel point spacing tolerance | A point is appended at travel >= spacing x (1 - 1e-4) because float accumulation of 0.5 mm steps missed 1.5 mm by rounding. |
| I5 | Removal rule 3 | Only forward tip motion (along the tip direction) grows the tunnel; retraction resets the travel and removes the last point once it lies more than one spacing behind the tip. |
| I6 | Never-entered tunnel | A tunnel whose tip never entered the tissue and moved more than spacing + gate radius from its entry point is finished (open question 3 of chapter 2). |
| I7 | Merge metadata | The suture lists use string frequencies "suture_needle" and "suture_segment" like the muscle lists; `suture_needle_segment_start` holds one start per needle in the builder and gets the total appended in `finalize`; `add_builder` rebuilds the group bits and the shape gate. |
| I8 | Needles without tets | Allowed (the gate still works); such a needle can never puncture. |
| I9 | Test helper `suture_set_points` | Creates a fixed (THROUGH) tunnel so the state machine leaves hand-set points alone. |
| I10 | Graph capture test | Both runs warm up with one step before comparison: the solver's first step differs from later ones (contact state initialization), captured and plain runs then agree to the bit. |
| I11 | Demo scale | `builder.rigid_gap = 1 mm`: the default 0.1 m pairs every thread capsule with every other capsule and overflows the per-body rigid contact buffer. |
