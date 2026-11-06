# Decisions: rigid-soft contact under ALM-DAT

## Requirements as stated

- R1 (Anka, 2026-11-06, chat): "我接下来想把 rigid-soft contact 改成 ALM DAT，你能不能给我做一个 Design PR，按 design review with PR 的形式".
- R2 (Anka, 2026-11-05, standing rule): every mechanism, rule, constant or default outside her design is a fork in the review first; nothing ships as a default before her decision.
- R3 (Anka, 2026-11-06, D15 of the self-contact review): the penalty metric follows the rigid contact rule, rho from the pair's support, never from the contact stiffness.

## Decision table

| ID | Item | Where | Decision |
|---|---|---|---|
| D1 | Body side: in place or body anchor | chapter 1, fork "Body side", question 1 | decided 2026-11-06: anchor and trial pair, as the soft side |
| D2 | Method switch granularity | chapter 1, fork "Method switch", question 2 | decided 2026-11-06: separate `rigid_soft_contact_method`, values `"penalty"` and `"alm"` |
| D3 | Multiplier history across detections | chapter 1, fork "Multiplier history", question 3 | decided 2026-11-06: matched by the record key |
| D4 | Rigid point in the gap | chapter 1, fork "Rigid point in the gap" | decided 2026-11-06: closest point at the body trial for mesh rows, stored body point moved with the body for analytic rows |
| D5 | Friction metric | chapter 1, fork "Friction metric" | pending |
| D6 | Masked rows | chapter 1, fork "Masked rows", question 5 | decided 2026-11-06: plane kept, no row, as the host |
| D7 | World and kinematic shapes: rho from the particle support alone | chapter 1, question 4 | pending |
| D8 | Body budget fraction rule: exact translation sphere plus linear rotation bound, or bisection on the combined end-point bound | chapter 2 | pending |
| D9 | Matching mechanism for the record key: sorted merge on a packed key as the self-contact rows, or a hash table | chapter 3 | pending |

## Decisions taken in review

### D1, body side (2026-11-06, chat, grill session)

Anka: "follow the soft side design". The body has a detection pose (`body_q_prev_collision_detection`), an anchor
(`_dat_body_anchor`, new) and a trial (`state_in.body_q`), exactly like a particle. Consequences written into
chapter 1: no in-place truncation of either side under alm (`_rigid_penetration_free_truncation` at
`solver_vbd.py:4655` and `solver_vbd.py:5489` does not run); both anchors commit in `_alm_dat_truncate`, the
rigid side along curved trajectories (`rigid_trajectory_truncation_t`); the body commit keeps the host's end-point
bound `|dx| + |angle| x radius <= budget` from the detection pose (`apply_body_truncation_ts`); the final rounds
run before `_finalize_rigid_bodies`, which then reads the body anchor; the second detection runs at both anchors
with `body_q` parked. Rejected: bodies truncated in place as the rigid-soft DAT does today. Opened D8 for chapter 2.

### D3, multiplier history (2026-11-06, chat, grill session)

Anka: "yes, implement match". The rigid-soft multipliers follow the record key (soft indices, shape, rigid indices)
across the mid-step detection and across substeps; unmatched records start at zero. The key is unique within one
detection: the legacy and SDF passes emit at most one record per soft feature and shape (`kernels.py:1354`,
`soft_contacts_sdf.py:527`), the mesh pass adds the rigid corners (`soft_contacts_mesh.py:1208`). Rejected:
re-seeding at every detection (the penalty today), and the body-body `rigid_contact_history` mechanism. Opened D9
(matching mechanism) for chapter 3.

### D2, the switch (2026-11-06, chat, grill session)

Anka first accepted the implied form ("ok"), then revised it minutes later: "changed my mind; give it a separate
switch". Decision: a separate `rigid_soft_contact_method` string, default `"penalty"`, value `"alm"` for the
ALM-DAT rows, as the host keeps one flag per contact family. `"alm"` requires `rigid_soft_enable_dat=True` and
`particle_self_contact_method="alm_dat"`; `"penalty"` with `alm_dat` and rigid DAT stays refused
(`solver_vbd.py:1218`). Rejected: implying the rows from the existing pair of flags.

### D4, the rigid point of the gap (2026-11-06, chat, grill session)

Anka: "ok" to the recommendation. Mesh records recompute the closest-point pair of the soft feature and the rigid
primitive at the trial pair every iteration, as the soft rows do (point-triangle, edge-edge, triangle-point); the gap
is the signed distance along the stored plane normal minus particle radius and shape margin. Analytic records keep the
soft feature point at the detection barycentrics and the stored body point moved with the body trial, as the host's
penalty does (`rigid_vbd_kernels.py:2177`). Rejected: the support point along the normal; the stored point for every
record.

### D6, masked rows (2026-11-06, from the host code, no objection)

`_soft_contact_force_enabled` (`soft_contacts_mesh.py:1593`) keeps all raw records for DAT and selects forces by
particle activity, resolved sign and feature ownership. The chapter keeps that meaning: a masked record keeps its
plane and gets no row.
