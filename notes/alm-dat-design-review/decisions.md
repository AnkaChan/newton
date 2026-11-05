# Decisions: ALM-DAT design review

## Requirements as stated

- R1 (Anka, 2026-11-05, chat): "I want to see the pseudo-code and data structure of the ALM_DAT implementation. We have a skill of proposing a design using a PR; I want you to create something like that for the existing; you can leave the code on the side; this is more important."
- R2 (Anka, 2026-10-19, chat, standing rule): pseudo code is a call tree of real function names with the control flow written out, comments on their own line, no prose and no near-real code.
- R3 (Anka, 2026-10-20, chat, standing rule): every array carries its dtype, its shape and the meaning of each dimension.

## Decision table

| ID | Item | Where | Decision |
|---|---|---|---|
| D1 | Budget rule default (`path_length` or `distance`) | chapter 1, fork "Budget rule", question 1 | pending |
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

None yet.
