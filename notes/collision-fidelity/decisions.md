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

## Overview review

Pending means the chapter text has not been approved. It does not cancel R1 to R7.

| ID | Item | Where | Decision |
|---|---|---|---|
| D1 | Runner and scene boundary | overview.md, sections 1.1 to 1.4 | pending |
| D2 | Measurement scope and result meaning | overview.md, section 1.5 | pending |
| D3 | Numerical loop | overview.md, section 1.6 | pending |
| D4 | Form of future Stability limits | overview.md, sections 1.7 and 1.8 | pending |

## Decisions taken in review

None yet. Chapter 1 describes code at `c63a3a24969f33cd86a7521e8d705664f2d29ed5`.
The review branch changes documents only. The upstream implementation PR stays draft.
