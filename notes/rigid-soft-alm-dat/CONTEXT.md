# Rigid-soft contact under ALM-DAT

The language of the rigid-soft ALM-DAT design (`rigid-soft-alm-dat.md`). It extends the self-contact ALM-DAT
terms to rigid bodies. Settled in the grill session of 2026-11-06.

## Language

**Detection pose**:
The particle positions and body poses at the last collision detection; the budget is centred here.
_Avoid_: reference configuration, pos_prev, x_start (in prose)

**Trial**:
The iterate the local solves write and the rows read; it may penetrate.
_Avoid_: candidate, current pose, iterate

**Anchor**:
The certified state of a particle or a body; it never penetrates, moves only at a commit, and is what the step writes out.
_Avoid_: committed state, safe state, truncated state

**Division plane**:
A stored plane per record or self-contact slot that separates the two sides' anchors with a maximin offset.
_Avoid_: separating plane, DAT plane, witness plane

**Commit**:
Moving an anchor toward its trial, truncated by the division planes and clamped by the budget, then rebuilding the planes.
_Avoid_: truncation (for the anchor move), projection

**Budget**:
The largest end-point motion of any point since the detection pose: half the detection reach times gamma.
_Avoid_: max displacement, cap, path length

**Record**:
One rigid-soft contact entry of `Contacts.soft_contact_*`: soft feature, rigid shape and primitive, body point, normal.
_Avoid_: pair, contact

**Row**:
The ALM constraint of a record or a self-contact slot, with its multiplier and its metric.
_Avoid_: constraint, contact force

**Metric**:
The penalty scale rho of a row, taken from the pair's support along the plane normal.
_Avoid_: penalty stiffness, rho floor, K

## Relationships

- A **Record** has exactly one **Division plane** and one **Row**.
- A **Row** reads the **Trial** of both sides and pushes both.
- A **Commit** moves an **Anchor**; a particle and a body each have one.
- The **Budget** is measured from the **Detection pose**, never from the previous anchor.

## Example dialogue

> **Dev:** "After the body colour, do we truncate the body pose against the planes?"
> **Domain expert:** "No. The body pose is a **Trial**. Only the **Commit** moves the body **Anchor**, and only the **Anchor** is written out."

## Flagged ambiguities

- "truncation" named both the host's in-place pass and the anchor move. Resolved: **Commit** is the anchor move; truncation is only the fraction along anchor to trial.
- "reference configuration" in the host docstrings is the **Detection pose**.
