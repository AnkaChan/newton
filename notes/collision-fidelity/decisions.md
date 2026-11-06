<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Collision-fidelity overview decisions

## Requirements

The earlier requirements come from Anka's chat. Dates below are record dates
when the original message time is not retained. This review does not reopen them.

| ID | Requirement stated by Anka | Source |
|---|---|---|
| R1 | "no changes to the solver; this benchmark must stand alone" | Chat; recorded in the task design on 2026-11-04. |
| R2 | "keep JItter"; "pentration must be clear about it's rigid soft or self-contact" | Chat; recorded in the task design on 2026-11-04. |
| R3 | "never rely on reference. must be purely on self-measurable criteria"; "except if there are simple analytic solution" (such as free fall) | Earlier chat; original date not retained. |
| R4 | "measure each criteria separately"; "it has to reach all criteria to pass" | Earlier chat; original date not retained. |
| R5 | "should be no more than 10" substeps | Earlier chat; original date not retained. |
| R6 | "you should keep all of them! just replace those cubes with real griper" | Earlier chat; original date not retained. |
| R7 | "write me a overview using the review with pr skill" | Chat, 2026-11-05. |
| R8 | "we dont have to exactly match the asset, as long as the machnism are the same" | Chat, 2026-11-06. |
| R9 | Fix small review issues and "propose the big ones". | Chat, 2026-11-06. |

## Overview review

Pending means the chapter text or proposal has not been approved. It does not cancel R1 to R9.

| ID | Item | Where | Decision |
|---|---|---|---|
| D1 | Runner and scene boundary | overview.md, sections 1.1 to 1.4 | pending |
| D2 | Measurement scope and result meaning | overview.md, section 1.5 | pending |
| D3 | Numerical loop | overview.md, section 1.6 | pending |
| D4 | Positive task checks and acceptance limits | overview.md, sections 1.7 and 1.8 | pending |
| D5 | Full ASV opt-in and runtime budget | overview.md, sections 1.7 and 1.8 | Explicit opt-in approved by Anka, 2026-11-06; implementation and runtime measurement pending. |
| D6 | Lower-is-better ASV status and coverage encoding | overview.md, sections 1.7 and 1.8 | Approved by Anka, 2026-11-06; use clear failure and missing-coverage measures. Implementation pending. |

## Decisions taken in review

On 2026-11-06, R8 clarified the asset requirement and R9 authorized small fixes.
The fixes landed at `32ef532f7ffc62ffa0688826583d72496df4604b`:

- Generated-asset tests check the mechanism, topology, scale, and usable landmarks. Archived hashes remain provenance, not cross-platform acceptance limits.
- Input validation rejects duplicate catalog records, invalid element indices, unknown soft-scene options, noninteger or excessive cable substeps, and durations shorter than one substep.
- ASV reports execution errors and invalid capacity results while preserving numerical-failure tracks. Unsupported cases still skip. Full-workload gating and scalar direction did not change.
- Versioning excludes asset documentation; H1 metadata hashes and notices are checked or corrected. Scene input reports retain boundary notes.
- CLI output ownership and USD overwrite checks are enforced. Missing Git and USD-close failures no longer discard otherwise available results.
- Contact metadata states the numeric cloth/tet comparison. Rod references include stretch and shear; repeated policy application verifies elastic references. The prescribed pad boundary remains fixed through its declared observation endpoint.

These are implementation fixes, not approval of D1 to D6 or evidence of a successful full-hold campaign on the new head.
Chapter 1 describes that code head.
The review branch changes documents only. The upstream implementation PR stays draft.

On 2026-11-06, Anka chose A for full-run cost: full collision-fidelity cases
must require explicit opt-in. An unfiltered ASV run must skip them. All 45
catalog cases remain available, with their full hold durations unchanged.
This settles the opt-in part of D5. No runtime limit or shortened schedule was
approved. The current implementation still includes full cases by default.

On 2026-11-06, Anka chose clear negative measures for ASV: "you can change
success rate to failure rate, etc". This settles D6. ASV 0.6.6 can store
higher-is-better values, but its automatic comparisons assume lower is better.
The upstream [direction-setting request](https://github.com/airspeed-velocity/asv/issues/1617)
is still open. This choice needs no ASV extension.

- Report failure rate as Fail / (Pass + Fail), where a rate is used. Keep counts visible. A zero denominator gives no value.
- Encode per-case Overall failure as Fail = 1 and Pass = 0. Incomplete and N/A stay unavailable, not failures or passes.
- Replace complete-window count with missing-window count: declared minus valid complete windows. Keep the declared count as context, not a quality score. Do not add a second track for the same coverage.
- Replace finite completion with an incomplete-run flag: 0 for a full finite timeline; 1 for a known incomplete or nonfinite timeline. Missing evidence stays unavailable. This flag does not score task success.
- Where evaluation coverage is compared, use the unevaluated fraction: Incomplete / (Pass + Fail + Incomplete). Exclude unsupported and N/A cases from both rate denominators.
- Keep Jitter, RMS speed, and fixed-support penetration unchanged. Unsupported execution still skips; execution errors remain errors.

Implementation is pending while the remaining review points are discussed.
Existing result files remain unchanged. Overall Pass rules are still open.
