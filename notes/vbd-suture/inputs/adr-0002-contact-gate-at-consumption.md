---
status: accepted
date: 2026-11-05
---

# Gate needle-tissue contacts in the solver's record filter, not in detection

Tissue particles near a tunnel must not collide with the needle group, or contact fights the tunnel point rows. We decided to detect and store those records as usual and to ignore them on the consumer side: two gate bit masks (per particle, per shape) are bound into `_SoftContactFilterData` and tested at the top of `_soft_contact_force_enabled`, which every VBD rigid-soft consumer already calls (particle gather and scatter, per-body accumulation, contact force export, body-particle duals, coupling harvest), plus the same test in `apply_rigid_soft_truncation`, which bypasses the filter by design.

## Considered options

- Skip the pair inside `create_soft_contacts` and the full-surface mesh pass. Rejected: it changes the shared detection kernels and the `CollisionPipeline` plumbing that every solver uses, for the same result.

## Consequences

- Gated records occupy soft-contact buffer slots (tens to hundreds per needle); negligible against a buffer sized for every particle-shape pair.
- A gated record builds no DAT plane, so the penetration-free guarantee for all other pairs is unchanged.
- Anyone adding a new consumer of rigid-soft records must route it through the filter or re-test the gate.
