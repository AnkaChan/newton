# Decisions: rigid-soft contact under ALM-DAT

## Requirements as stated

- R1 (Anka, 2026-11-06, chat): "我接下来想把 rigid-soft contact 改成 ALM DAT，你能不能给我做一个 Design PR，按 design review with PR 的形式".
- R2 (Anka, 2026-11-05, standing rule): every mechanism, rule, constant or default outside her design is a fork in the review first; nothing ships as a default before her decision.
- R3 (Anka, 2026-11-06, D15 of the self-contact review): the penalty metric follows the rigid contact rule, rho from the pair's support, never from the contact stiffness.

## Decision table

| ID | Item | Where | Decision |
|---|---|---|---|
| D1 | Body side: in place or body anchor | chapter 1, fork "Body side", question 1 | decided 2026-11-06: anchor and trial pair, as the soft side |
| D2 | Method switch granularity | chapter 1, fork "Method switch", question 2 | pending |
| D3 | Multiplier history across detections | chapter 1, fork "Multiplier history", question 3 | pending |
| D4 | Rigid support in the gap | chapter 1, fork "Rigid support in the gap" | pending |
| D5 | Friction metric | chapter 1, fork "Friction metric" | pending |
| D6 | Masked rows | chapter 1, fork "Masked rows", question 5 | pending |
| D7 | World and kinematic shapes: rho from the particle support alone | chapter 1, question 4 | pending |
| D8 | Body budget fraction rule: exact translation sphere plus linear rotation bound, or bisection on the combined end-point bound | chapter 2 | pending |

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
