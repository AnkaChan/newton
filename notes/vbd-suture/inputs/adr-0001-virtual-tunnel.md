---
status: accepted
date: 2026-11-05
---

# Represent the suture tunnel with tunnel points and constraints, not with mesh changes

Suturing needs a path through the tet mesh that the needle and the thread follow and slide along. We decided to represent that path as a virtual tunnel: barycentric tunnel points created as the tip advances, each carrying one compliant-ALM row that holds the rod curve transversally and lets it slide with capped ALM friction. The tet mesh keeps its topology.

## Considered options

- Real tunnel by XFEM tube cut or node snapping. Rejected for the first milestone: a half-millimetre channel in millimetre tets gives clipped volume fractions near 1e-2, every tip advance needs a host rebuild outside CUDA graph capture, the XFEM branch does not yet pass its own cut sequence, and a thinner thread would rattle in the needle hole.
- Hybrid with an embedded Eulerian thread. Deferred: it needs a variable-length rod segment that the rigid-capsule rod model does not have.

## Consequences

- Tissue particles near the tunnel must be excluded from rigid-soft contact against the needle group, or contact fights the tunnel. This needs a new per-pair contact gate.
- The needle tip must be excluded from Divide-and-Truncate against the tissue it punctures; everywhere else the penetration-free guarantee stays exact.
- The hole is not visible in the simulation mesh; a visible hole is display work.
