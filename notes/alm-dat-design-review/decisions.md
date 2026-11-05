# Decisions: ALM-DAT design review

## Requirements as stated

- R1 (Anka, 2026-11-05, chat): "I want to see the pseudo-code and data structure of the ALM_DAT implementation. We have a skill of proposing a design using a PR; I want you to create something like that for the existing; you can leave the code on the side; this is more important."
- R2 (Anka, 2026-10-19, chat, standing rule): pseudo code is a call tree of real function names with the control flow written out, comments on their own line, no prose and no near-real code.
- R3 (Anka, 2026-10-20, chat, standing rule): every array carries its dtype, its shape and the meaning of each dimension.

## Decision table

| ID | Item | Where | Decision |
|---|---|---|---|
| D1 | Budget rule default (`path_length` or `distance`) | chapter 1, fork "Budget rule", question 1 | decided 2026-11-05: `path_length` is wrong and is removed; the end-point clamp is the only rule, no switch |
| D2 | Budget scope (every vertex or vertices with a stored plane) | chapter 1, fork "Budget scope", question 2 | pending |
| D3 | Detections per substep (one certified anchor or two) | chapter 1, fork "Detections per substep", question 3 | decided 2026-11-05: do not force one; `PRE_INIT` default, `PRE_POST_INIT` allowed with a second certified anchor (to implement) |
| D4 | Coverage loss handling (flag or grow and re-detect) | chapter 1, fork "Storage growth", question 4 | pending |
| D5 | Rigid-soft DAT together with `alm_dat` | chapter 1, fork "Rigid-soft DAT" | pending |
| D6 | Normal refresh cadence (every commit or between final rounds) | chapter 2, fork "Normal refresh cadence", question 1 | pending |
| D7 | Plane margin (0 or the cloth thickness) | chapter 2, fork "Plane margin", question 2 | pending |
| D8 | Failed separator at the anchor (keep or invalidate) | chapter 2, fork "Failed separator", question 3 | pending |
| D9 | Smooth transition as the default row | chapter 3, fork "Compliant-to-hard transition", question 1 | pending |
| D10 | Wall position default | chapter 3, fork "Wall position", question 2 | pending |
| D11 | Friction metric floor | chapter 3, fork "Friction metric", question 3 | pending |
| D12 | Multiplier update cadence | chapter 3, fork "Multiplier update cadence", question 4 | pending |

## Decisions taken in review

- 2026-11-05, Anka, chat, on the first draft: "the pseudo code is very unclear; the logic structure is bad. I am not asking you to make it more verbose! I want you to make the logic clear; add to the skill that `->` is not allowed; indent is only used for code blocks in controls such as if and for." Applied in round 1: the three pseudo-code blocks are rewritten as structured statements (one function per block, plain calls and assignments, indentation only for function and control bodies, no arrows). The skill `designing-through-pr-review` carries the rule from now on. Closes no fork.

- 2026-11-05, Anka, chat, on the budget rule (D1): "path_length is wrong; theoretically distance will guarantee not penetrate, and it requires much less records." Background from the same exchange: the budget `0.5 gamma (margin + gap)` is the penalty DAT clamp on the displacement from the detection pose (`apply_truncation_ts`, `particle_vbd_kernels.py:2133-2136`); the first ALM-DAT revision replaced it with the bound `|offset| + |step| <= budget` at each commit (the kwarg called it `path_length`; it kept no extra state, `offset` is the anchor's displacement from the detection pose). That is the triangle-inequality sufficient condition for the end point, strictly tighter than the sphere with nothing in return: it froze a vertex at the cap even for a step back and cut every non-radial step short (lock-in on the collision-fidelity benchmark). The end-point distance alone bounds how far an undetected pair can close, so it is the certificate. Applied: `particle_dat_budget_rule` and the path-length branch are removed in the code (fable-review worktree, uncommitted for cherry-pick; unit test `test_commit_budget_sphere`); chapter 1 and chapter 2 updated in place; fork "Budget rule" and question 1 closed. The benchmark column `alm_dat` stands as the record of the removed rule and is retired in the benchmark copy.

## Round 2 (review 5778252743, submitted 2026-11-05 18:15 UTC on 457bd1d5)

- Line 21, schedule row. Anka: "why forcing it? I dont think anything requires doing it once per step. Of course I agree to switch the default value to PRE_INIT." Answer: nothing in the scheme requires it; the code forces `PRE_INIT` because `_alm_dat_initialize` runs once and a mid-step detection re-assigns the slots. A second detection at the certified anchor after the inertial commit, followed by a re-initialization (`x_start = anchor`, displacement re-based, planes rebuilt, multipliers matched through the sorted rows, budget restarted), gives twice the admissible motion at the same record count. Decision D3: allow `PRE_POST_INIT`, default `PRE_INIT`, to implement. Benchmark evidence (`notes/collision-benchmark-alm-dat`): the 3 mm gap, the other way to raise the cap, multiplied the self-contact record demand by 14 and 19 (vt, ee) on the Franka towel, 12 and 17 on the H1 towel, 2.1 to 2.4 on the T-shirt and the dress, and made the jacket run non-finite.
- Line 23, cadence row. Anka: "particle_dat_truncation_iterations what is this? ... what do you mean by 'final rounds'? shouldn't it truncate at every iteration." Answer: with `particle_dat_plane_update_interval = 1` (default) the anchor is truncated after every sweep, which is stage 1 of `inputs/algorithm-reference.md` with `PLANE_UPDATE_INTERVAL`; not after every colour, because the rows read the trial, not the anchor. `particle_dat_truncation_iterations` is stage 2 of the reference (`FIXED_TRUNCATION_ITERATIONS`): after the last sweep, truncate, re-place the planes at the new anchor, truncate again, so the anchor gets closer to the final trial than one truncation allows. Row and argument comments rewritten with the stage names.
- Line 125, `_alm_dat_materialize_trial`. Anka: "what is this?" Answer: the solver keeps the iterate as `particle_displacements` from `x_start`; this call writes `x_start + displacement` into `state.particle_q` with the penalty path's kernel at `t = 1` and no clamp. It is the trial that the rows and the multiplier update read. Comments rewritten; `state.particle_q` added to the data-structure table. Open: rename it to `_alm_dat_write_trial` at implementation time?
