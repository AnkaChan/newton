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
