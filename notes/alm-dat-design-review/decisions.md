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
| D3 | Detections per substep (one certified anchor or two) | chapter 1, fork "Detections per substep", question 3 | pending |
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

- 2026-11-05, Anka, chat, on the budget rule (D1): "path_length is wrong; theoretically distance will guarantee not penetrate, and it requires much less records." Background from the same exchange: the budget `0.5 gamma (margin + gap)` is the penalty DAT clamp on the displacement from the detection pose (`apply_truncation_ts`, `particle_vbd_kernels.py:2133-2136`); the first ALM-DAT revision replaced it with a path-length accounting at the commit, which froze vertices at the cap (lock-in on the collision-fidelity benchmark) and carried the travelled length as extra state. The end-point distance alone bounds how far an undetected pair can close, so it is the certificate. Applied: `particle_dat_budget_rule` and the path-length branch are removed in the code (fable-review worktree, uncommitted for cherry-pick; unit test `test_commit_budget_sphere`); chapter 1 and chapter 2 updated in place; fork "Budget rule" and question 1 closed. The benchmark column `alm_dat` stands as the record of the removed rule and is retired in the benchmark copy.
