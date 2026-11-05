# Suture simulation with SolverVBD: design options

Date: 2026-11-05. Author: Claude (Fable 5.1) for Anka. Status: draft for discussion. No code was written.

This document gives three candidate designs for the feature in `task-prompt.md`. The facts about Newton come from `prior-work.md`, Part B (file and line citations there). The facts about earlier papers come from `prior-work.md`, Part A, cited as A.x. Statements marked "(my inference)" are my own judgement.

## 0. Shared parts of every option

These parts are the same in all three options. I list them once.

**Needle.** One rigid body. Its centerline is a circular arc given by radius `R_n`, arc angle, and wire radius `r_n`. The body carries 8 to 16 short capsule shapes along the arc (analytic SDF, cheap contact) and one extra small capsule at the tip, so the tip is a separate shape index. A mesh shape is possible but costs a mesh SDF query per particle. The needle is either kinematic (`inv_mass = 0`, pose set each step) or dynamic and held by a drive to a target pose. This is a switch and a later ablation.

**Thread.** `ModelBuilder.add_rod` with a `Rod` centerline: capsule bodies joined by ROD joints (stretch, shear, bend, twist). The first thread segment joins the needle tail with a BALL or FIXED joint. Thread radius `r_t` is smaller than `r_n`. Thread-thread contact (knots) uses existing body-body contact and is out of scope for the first milestone.

**Tissue.** Two tet meshes from `add_soft_mesh`, Neo-Hookean, with self-contact on (the self-contact detector also handles contact between the two pieces, because both are in one particle set).

**Puncture state machine (host side, once per step).** States per needle: `OUTSIDE`, `PIERCING` (tip inside tissue, tunnel grows), `THROUGH` (tip exited, thread follows). The tip position is tracked in material coordinates by a tet walk: find the tet that contains the tip, step to the neighbor across the face with the most negative barycentric weight, stop at a boundary face. An exit through a boundary face is the exit event. Transition rules:

- `OUTSIDE -> PIERCING` when the tip shape presses the tissue with normal contact force above `F_puncture` and the tip direction is within an angle `alpha_max` of the inward surface normal. The contact force comes from the existing rigid-soft contact rows. Precedents: Berkley 2004 multiplies the direction-normal dot product by the force (A.3.1); the SOFA Cosserat needle and InfinyTech3D use a force threshold (A.4.3, A.4.4); Yu 2020 uses a force threshold (A.5.2). A geometric alternative (tip within `eps` of the surface while "armed") is a switch.
- `PIERCING`: every time the tip has advanced by `ds_tunnel` inside the tissue, append a tunnel node.
- `PIERCING -> THROUGH` at the exit event. Append the exit node on the exit face.
- `THROUGH -> OUTSIDE` when the thread end leaves the last tunnel node; the tunnel data stays for display.

**Contact gate (needed by Options 1 and 3, partly by Option 2).** Tissue particles near the tunnel must not collide with the needle shaft or the thread, or contact would fight the tunnel. The SOFA Cosserat needle deactivates its contact constraint after the puncture, and InfinyTech3D has an explicit `enableShaftCollision` switch (A.4.3, A.4.4). Gate set = particles of tets that contain a tunnel node, plus one ring of neighbors, plus the surface ring around the entry and exit faces. Representation: `particle_contact_gate: wp.array[int32]`, one bit per suture tool group; shapes of the needle and thread carry the same group id. Fork (one must be picked, both are cheap):

- (G1, preferred) skip the pair in `create_soft_contacts` (`newton/_src/geometry/kernels.py:1184`) and in the full-surface mesh pass, so no record exists and DAT never sees the pair.
- (G2) keep the record and filter at force time through `_soft_contact_force_enabled` (`soft_contacts_mesh.py:1593`), and add the same skip to `apply_rigid_soft_truncation` (`rigid_vbd_kernels.py:8252`).

G1 keeps the DAT guarantee exact for every pair that is not gated. Both need a small change in shared geometry or solver code.

**DAT.** Stays on for everything outside the gate set. The tip is not gated before the puncture, so the needle cannot sink into tissue below `F_puncture`. After the puncture, the tunnel tets are gated and DAT no longer protects them against the needle group; the tunnel constraint takes over. The tissue still keeps DAT against itself and against other rigid bodies.

## 1. Option 1: virtual tunnel as sliding point-on-curve constraints

The tunnel is a polyline of nodes fixed in the tissue material. Each node is a barycentric point in one tet. Each active node holds the rod (needle arc or thread polyline) at its closest point, in the two directions normal to the rod tangent, and lets it slide along the tangent with friction. This is Lenoir's sliding point (A.2.4), the two-transverse-rows-plus-free-tangent layout of BeamAdapter and Martin 2023 (A.4.1, A.2.3), and the barycentric tunnel of the SOFA Cosserat needle, iMSTK and the VirtaMed patent (A.4.3, A.5.4), moved into VBD's compliant ALM rows. No published VBD version exists (A.5.5).

### 1.1 Data structures

| Name | dtype | Shape | Written by | Read by |
| --- | --- | --- | --- | --- |
| `tunnel_tet` | int32 | `[max_nodes]`, tet index, -1 when free | host at node creation | particle and rigid kernels |
| `tunnel_bary` | vec4 | `[max_nodes]`, barycentric weights in `tunnel_tet` | host at node creation | both kernels (anchor `x_k = sum w_i x_i`) |
| `tunnel_tool` | int32 | `[max_nodes]`, needle index the node belongs to | host | both kernels |
| `tunnel_active` | int32 | `[max_nodes]`, 1 while some rod part passes the node | device per iteration (closest-point search) | both kernels |
| `tunnel_rod_body` | vec2i | `[max_nodes]`, `[0]` body A, `[1]` body B of the rod segment at the closest point (`B = -1` on the needle arc) | device | both kernels |
| `tunnel_rod_u` | float | `[max_nodes]`, parameter of the closest point on that segment, or arc angle on the needle | device | both kernels |
| `tunnel_lambda` | vec3 | `[max_nodes]`, ALM multiplier in the normal plane | dual kernel | both kernels |
| `tunnel_rho` | float | `[max_nodes]`, ALM metric, once per step from both supports | set-up kernel | both kernels |
| `tunnel_slip_prev` | float | `[max_nodes]`, accumulated tangential slip for friction history | finalize | friction term |
| `body_tunnel_offsets`, `body_tunnel_indices` | int32 | CSR body -> nodes, rebuilt at node creation | host | rigid kernel |
| `particle_contact_gate` | int32 | `[particle_count]`, bit per tool group | host at node creation | contact detection (G1) |
| `tunnel_grip_force` | float | scalar or `[max_nodes]`, extra normal force `F_grip` [N] for friction | author | friction term |

Node capacity `max_nodes` is fixed at construction (for example 256 per needle). Node creation is host work between steps, like a cut event in the XFEM companion, but it only writes a few array entries. It can stay outside a CUDA graph, or run as a device kernel later.

### 1.2 Constraint and how it enters the per-vertex solve

For node `k` with anchor `x_k = sum_i w_i x_i` (four tet vertices) and rod point `p(u)` with tangent `t`:

```
P = I - t t^T                      (projector onto the normal plane; t held constant in one iteration)
C = P (x_k - p)                    (2 effective residual components)
f = k_eff C + s lambda             (compliant ALM, k_eff = k rho/(k+rho), s = k/(k+rho))
lambda <- P s (lambda + rho C)      (dual update after both passes, like update_duals_particle_body_attachments)
```

Particle side, per color (new kernel next to `accumulate_body_particle_attachment_force_and_hessian`, `particle_vbd_kernels.py:2070`): vertex `i` of the current color receives force `-w_i f` and Hessian `w_i^2 k_eff P`. The weights follow the XFEM branch's embedded scatter (`particle_contact_kernels.py:52`: force times `w`, Hessian times `w*w`). The other three vertices are frozen in that color pass; they have different colors because they share a tet, so no coloring change is needed.

Rigid side, per body (new kernel next to `accumulate_body_particle_attachments_per_body`, `rigid_vbd_kernels.py:5866`): the needle body receives `+f` at `p` as a wrench with the 6x6 blocks from `_point_force_to_body_torque_and_hessian`. A thread segment splits it: body A gets `(1-u) f`, body B gets `u f`. The CSR `body_tunnel_offsets` lists the nodes per body; it changes when the closest point moves to the next segment, so the device recomputes it or the kernel loops over all active nodes of its tool (a few hundred, cheap).

`rho` per node is set once per step from the particle support (the four weighted supports, reduced by material Hessians like `step_particle_body_attachment_lambda_rho`, `particle_vbd_kernels.py:1964`) plus the rod body support. Infinite `k` gives the hard version, as in AVBD (A.5.5). Finite `k` of a few kN/m gives a soft tunnel wall.

Unilateral tube variant (switch, ablation entry): replace the bilateral line by a tube of radius `r_tube`: `C = P (x_k - p)` active only when `|C| > r_tube`, force along `C/|C|`. This lets a thin thread sit loosely in the hole left by a thicker needle. Default off.

### 1.3 Tip-only puncture

The state machine of Section 0 creates nodes only from the tip trajectory. The shaft and the thread can never create a node; they collide with tissue everywhere except in the gate set. Therefore only the tip pokes through. Switching = changing `particle_contact_gate` bits for the tets the tip has entered, and appending nodes. The SOFA Cosserat needle adds one constraint point per `constraintDistance` of tip travel and removes points on retraction (A.4.3); the same rule applies here.

### 1.4 Friction

At node `k`, tangential relative displacement `d_t = t . ((x_k - x_k_prev) - (p - p_prev))` over the step. Normal force `N = |lambda| + F_grip`. Use `compute_friction` (`particle_vbd_kernels.py:1353`) with `mu_tunnel`, `N`, tangent basis `T = [t]` (one column), and `d_t`. Force `-f_t t` on the tissue anchor (scattered by `w_i`), `+f_t t` on the rod. `F_grip` models tissue that squeezes the thread, so friction exists at zero constraint force. This is the stick-slip-along-the-shaft idea of Chentanez 2009 (A.1.1) and the surface and insertion coefficients `mu_p`, `mu_i` of Perrusi 2021 (A.2.3) in one scalar per node. None of the PBD systems in A.5.4 reports Coulomb friction along the tunnel, so this is where the VBD version adds value.

### 1.5 Failure modes

- Residual drift at few iterations: the thread sits visibly off the line. Mitigation: `rho` from support, more iterations near the tunnel, or hard `k`.
- Gate boundary jitter: a vertex just outside the gate set collides with the shaft, its neighbor inside does not. Mitigation: one-ring margin and a smooth margin on `r_gate`.
- Node spacing: `ds_tunnel` above half a tet edge lets the rod cut corners between nodes. Mitigation: tie `ds_tunnel` to the mean edge length.
- Thread end leaving a node: the closest point hits the thread end, the node must deactivate without a kick. Mitigation: fade `k` with the end distance over one segment.
- Fast kinematic needle: gated pairs have no DAT, so a fast needle drags tissue through other tissue. Mitigation: cap the needle speed per substep to the DAT motion budget.
- False puncture from a sideways press: handled by `alpha_max`; still a tuning risk.
- Rigid-soft DAT and the gate interact in shared code; a mistake there breaks the penetration-free guarantee elsewhere. Needs a regression test on an ungated scene.

### 1.6 Effort

New module (companion class like `VBDTetCutting` or a mixin in `SolverVBD`), three kernels (particle side, rigid side, duals) plus a closest-point kernel and a set-up kernel, the gate change in detection, the host state machine, an example with two tissue blocks, and tests. Rough size: 1,500 to 2,500 lines including tests. Rough time: two to three weeks of agent work with review rounds. Risk: medium. All building blocks exist.

## 2. Option 2: real tunnel by local remeshing or XFEM

The tip carves a channel of radius `r_n` along its path. The channel wall becomes collision surface. Needle and thread then interact with the tissue only through the existing rigid-soft contact, friction, and DAT. No tunnel constraint exists. Precedents are the remeshing family: Chentanez 2009 (A.1.1), Goksel node repositioning and addition (A.1.3), and the surface subdivision at the pierce point of Marshall 2005 (A.3.3).

### 2.1 Two ways to carve

- (2a) Phantom-node XFEM along a tube. The XFEM branch splits tets by a planar sweep (`tet_cutting.py:282`). A tunnel needs a tubular level set `phi = dist(x, tunnel curve) - r_n`, non-planar clipping, and a hole surface inside the tet. The branch's polygon clipping (`xfem_geometry.py:48`) is planar and would need a new tube clipper. The embedded contact kernels of the branch (`particle_contact_kernels.py:120`) would consume the hole surface.
- (2b) Node snapping plus edge splitting, as in Goksel et al. (A.1.3) and Chentanez et al. (A.1.1): move the nearest tissue nodes onto the tip path so the path runs along element edges, then duplicate those nodes to open a zero-width slit that the needle widens by contact. This changes the simulation tets, mass, adjacency, and coloring at every advance, like a cut event.

### 2.2 Data structures

Everything the XFEM companion already owns (cells, DOF map, embedded surface, host rebuild), plus a tube description `tunnel_curve: vec3[max_nodes]` in material coordinates and `tunnel_radius`. The rebuild runs every time the tip advances by one tet.

### 2.3 Constraints in the solve and coloring

No new constraint type. Contact between the hole wall (embedded vertices, or real duplicated nodes in 2b) and the needle capsules uses the existing rigid-soft rows, gathered per color through the embedded scatter. Coloring is rebuilt by the companion at each topology event, so the per-vertex solve is unchanged, but the rebuild is host work and not graph-capturable.

### 2.4 Tip-only puncture

Same state machine. The puncture event starts carving. No gate is needed for the needle, because the hole has the needle radius. A gate is still needed if the thread radius is smaller than the hole and the hole should close around the thread (otherwise the thread rattles).

### 2.5 Friction

Existing rigid-soft contact friction (`_compute_body_particle_contact_force`, `rigid_vbd_kernels.py:2140`) on the wall rows. Real normal forces come from the wall pressing on the needle only when the needle is off-center or the tissue is pre-stressed, so sliding friction is weak unless the hole radius is below `r_n`, which makes the needle push on the wall everywhere and drives thin clipped elements hard.

### 2.6 Failure modes

- Conditioning: a 0.5 mm hole in 4 mm tets gives clipped volume fractions near 1e-2 and sliver surface triangles. VBD's local solves degrade with such elements.
- Cost: a host rebuild each time the tip enters a new tet; not graph-capturable; the XFEM branch reports that its full cut sequence does not pass yet (design.md, 2026-11-04).
- Rigid-soft DAT is rejected with an embedded contact mesh on the branch (`solver_vbd.py:1056` on the branch), so the needle-wall contact has no penetration guarantee without new work.
- Thread rattles in the needle hole; wound closing by thread tension relies on friction at the hole rim and is weaker than with a constraint tunnel.
- Non-planar clipping is new geometry code with many degenerate cases.

### 2.7 Effort

Depends on the XFEM branch landing. Beyond it: tube clipper, tunnel carving per tet, DAT support for embedded rows, tests. Rough size 4,000 to 6,000 lines. Rough time: six to ten weeks, with high uncertainty. Risk: high.

## 3. Option 3: hybrid, constraints for the needle and an embedded Eulerian thread

Use Option 1 for the needle phase. When the needle is through and the thread occupies the tunnel, switch the in-tissue part of the thread to an embedded Eulerian representation: each tunnel node stores the thread arc-length coordinate `s_k` passing through it as one scalar unknown. The thread has no transverse degrees of freedom inside the tissue; it moves with the tissue exactly. The outside parts stay ROD capsule bodies. This follows the Eulerian-on-Lagrangian strands of Sueda et al. 2011 (A.5.5), the "only the tangent component is kept" rule of CRESSim-MPM (A.5.4), and the composite catheter of Lenoir et al. 2006 (A.2.4).

### 3.1 Data structures

Option 1 arrays, plus `tunnel_s: float[max_nodes]` (thread material coordinate at the node), `tunnel_s_prev`, and a variable-length end segment on each side of the tissue whose rest length changes as thread slides in or out.

### 3.2 Constraints in the solve and coloring

Inside: a 1x1 block per tunnel node solves `s_k` from the thread tension difference across the node and 1D friction; it enters the iteration as a new block type between the rigid pass and the particle pass. The tissue vertices receive the thread tension as forces scattered by `w_i` (like Option 1 but without the transverse row). Outside: ROD bodies unchanged, but the capsule at the tissue surface needs a variable rest length, which `add_rod` and the ROD rows do not support today (segment lengths are fixed at construction, `builder.py:10481`). A direct solve along the chain (Deul 2018, Xu and Liu 2018, issue #4794; A.5.1, A.5.3, B.3) would help the stiff in-tissue part.

### 3.3 Tip-only puncture

As Option 1.

### 3.4 Friction

Exact 1D stick-slip on `s_k` with normal force `F_grip` plus the tension-induced normal force from the tunnel curvature. Simplest and most robust friction of the three options.

### 3.5 Failure modes

- Mass and length transfer at the tissue surface needs a variable-length rod segment: new rod machinery in the rigid solver.
- Two representations of one thread; the hand-over when the needle tail enters the tissue is a special case with its own bugs.
- Thread tension is transmitted exactly along the tunnel, so the tissue sees no transverse slack; good for wound closing, wrong for a loose thread in a wide hole.

### 3.6 Effort

Option 1 plus a new block type and variable-length rod segments. Rough size 3,000 to 4,500 lines. Rough time: four to six weeks. Risk: medium to high.

## 4. Comparison

| Criterion | Option 1: virtual tunnel | Option 2: real tunnel | Option 3: hybrid Eulerian thread |
| --- | --- | --- | --- |
| New constraint type | yes, ALM point-on-curve | no | yes, 1D sliding block |
| Topology change | none | per tet entered, host rebuild | none |
| Graph capture | yes, except node creation | no | yes, except node creation |
| Penetration guarantee outside tunnel | kept (gate is explicit) | kept for self-contact; rigid-soft DAT missing on embedded rows | kept |
| Thread in tunnel | follows the line, slides with friction | rattles unless hole < thread | follows exactly, 1D friction |
| Visible hole | no (display only) | yes | no |
| Reuse | attachments, embedded scatter, friction | XFEM companion, embedded contact | Option 1 plus new rod segment type |
| Effort | 2-3 weeks, medium risk | 6-10 weeks, high risk | 4-6 weeks, medium-high risk |

My recommendation (my inference): start with Option 1. It covers every item in the feature list, reuses the most code, and keeps the DAT guarantee explicit through the gate. Option 3 is the natural second step if thread sliding quality matters more than the first demo needs. Option 2 is the right path only if a visible, physically carved hole is a requirement.

## 5. Questions for Anka

1. Is a virtual tunnel acceptable for the first milestone, or is a real hole in the mesh a requirement?
2. Needle driving: kinematic prescribed pose, or a dynamic needle held by a gripper or robot?
3. Scales for the demo: tissue block size, tet edge length, needle radius and arc radius, thread radius, thread segment length.
4. Puncture trigger: contact-force threshold (physical) or armed geometric contact (simple)?
5. Where should the code live: a companion class like `VBDTetCutting`, or inside `SolverVBD` behind a flag?
6. Should I wait for the XFEM branch for anything, or proceed independently from `upstream/main`?
