# V2 plan: a learned optimizer for deformable hexahedral bodies

Date: 2026-09-24. This document records the current decisions and the proposed
next training campaign. **Training remains stopped. Review and approval of
the training plan are required before launching another campaign.** Implemented
helpers, proposed changes, and remaining integration work are distinguished
below.

## Physical problem and representation

Start with a canonical 10×10×40 grid of cubic hexahedral cells. The cell edge
length is 0.025 m, giving a 0.25×0.25×1 m body with 4,961 shared corners and
4,000 cells. Fix the 121 corners on one end. Use float32, a physical timestep
of 1/300 s, and gravity (0, −9.81, 0) m/s² unless a test explicitly changes it.

The unknown global shape is the set of shared corner positions. From these,
compute each cell's deformation matrix: three dimensionless deformed axes
relative to its rest geometry. Separate a local rotation frame from the axes
expressed in that frame. The axes retain stretching, shearing and signed
orientation; they are not normalized to unit length. Rest size is a separate
input. This is a cell-center affine description, not a complete independent
description of all eight deformed corner positions.

The network proposes changes to those same local axes. Differentiable global
fusion finds compatible shared corner positions while enforcing prescribed
corners. The native semi-implicit rigid prediction guides global pose through
fusion. **It does not replace the inertial target in the implicit-Euler
energy.** Elasticity is evaluated with hexahedral quadrature, not tetrahedra.

## Network baseline

Use context `[1]`: one radius-one graph transformer layer per optimizer
iteration. Each cell has 27 slots including itself and all face-, edge-, and
corner-sharing neighbors. Mask slots outside the grid. There is no context
jumping or fixed geometric attention prior. Keep learned edge attention bias
and edge value contributions, with neighbor geometry expressed in the
receiving cell's frame.

The current input/output layout is:

| Item | Per-cell size | Meaning |
|---|---:|---|
| Local axes | 3×3 | Current dimensionless deformation in the local frame |
| Inertial offsets | 8×3 | Each corner's inertial target minus its candidate position, expressed in the cell frame and divided by rest edge length |
| Exposed faces | 6 | Which faces are on the outer boundary |
| Fixed corners | 8 | Which corners have prescribed positions |
| FiLM conditioning | 5 | Scaled logarithmic lambda, mu, density, rest cell size and timestep |
| Directed edge features | 27×24 | Relative geometry and neighboring deformation for attention |
| Output correction | 3×3 | Change to local axes in the same input frame |

An object-level step-size head scales the corrections. The hidden width is
128, with four attention heads and an edge width of 64. The one-layer model
has 320,142 parameters. It can represent a zero correction. The current
inputs do not yet include the physical energy gradient or optimizer history.

The `[1]` default is implemented in the workspace. Existing checkpoints keep
their explicitly saved architecture. The epoch-198 checkpoint has three
layers and cannot be directly resumed as a one-layer model. **Fresh
initialization is the proposed baseline**; no silent checkpoint truncation.

## Data generation and material

Reuse deterministic multiresolution deformation augmentation on the canonical
rest grid, together with smoothly varying initial velocities. Include both
the inertial candidate and perturbed optimizer candidates. Always preserve
the prescribed corners. Shape, velocity, material and candidate sampling
should use reproducible seeds, with held-out seeds separate from training.

The implemented material sampler draws independently in logarithmic space:

| Quantity | Range |
|---|---:|
| First Lamé parameter, lambda | 1,000–1,000,000 Pa |
| Shear modulus, mu | 1,000–1,000,000 Pa |
| Density | 100–10,000 kg/m³ |

Sample one uniform material per object, keep it fixed along that trajectory,
and restore the exact material during replay. The sampler and replay format
support this; the old production dataset and trainer still use their original
fixed material until the new training pipeline is integrated.

The current physical solver uses one material context per batch. Initially,
group replay minibatches by context and switch physical models between such
batches; arbitrary mixtures of materials in one batch are not implemented.
Across GPU ranks, synchronize learned parameters and gradients while keeping
each rank's physical material and mass buffers local. This needs explicit
verification in the new distributed trainer.

Retain the previous scale of 8,192 fresh training starts and 512 fixed held-out
starts as a proposed starting point. The retained-state archive grows beyond
that seed count as trajectories advance. The old 50/35/10/5 candidate mixture
is historical, not a decided V2 live/replay/rest mixture.

## Consecutive optimizer iterations and physical timesteps

These are different loops. Inside one physical timestep, hold the original
inertial prediction fixed and run K consecutive network-and-fusion updates.
Backpropagate through all K updates. Recompute the local frames as the shape
changes, but detach their decomposition from the gradient graph. Network
weights stay fixed throughout that inner solve and its backward pass.

Across physical timesteps, carry the resulting positions and velocities
forward exactly, then detach both. Train each new timestep independently on
the state produced by the preceding one. A weight update may occur after each
physical solve. The next solve then uses the updated weights. Do not rerun
earlier physical steps after a weight update.

Generate these trajectories live, so they change as the solver learns.
Eight physical steps is a proposed training segment length, not a maximum
trajectory lifetime: some trajectories must continue into later motion and
settling. Mix those continuations with fresh augmented starts and replayed
states. The exact sampling mixture is still a training-plan choice for review.

For the new one-layer model, the proposed curriculum is:

| Stage | Optimizer updates per physical step | Physical steps per segment |
|---|---:|---:|
| Initial | 1 | 2 |
| 2 | 2 | 2 |
| 3 | 4 | 2 |
| 4 | 8 | 4 |
| 5 | 16 | 4 |
| Final | 32 | 8 |

Retain shorter solves after advancing. Use validation descent and trajectory
stability to decide when to advance; numerical thresholds and patience remain
to be selected before launch. Log actual counts and curriculum stage. If a
stage stalls, report it; never silently train only one iteration for the whole
campaign. The reusable K≤32 unroll and per-timestep gradient boundaries are
implemented, but the production epoch trainer is not yet wired to this loop.

## Disk retention and replay — implemented

Keep all collected timestep-start states in per-rank SQLite files. Store
float32 positions and velocities, loads, prescribed positions, source step and
checkpoint/update metadata. Store shared rest geometry, materials, native
masses and timestep in deduplicated physical contexts. Commit each batch
before solving it, including inputs whose subsequent solve fails. Do not
store autograd graphs or treat historical outputs as target labels.

Read only sampled states into memory. Seeded sampling uses a fixed compact
index until explicitly refreshed. Replay reconstructs the physical problem
and runs the current network. A trajectory cannot change material/context
halfway through, and a record cannot be paired with the wrong context.
Changing the replay iteration count or loss policy is allowed explicitly.

There is no automatic eviction. Raw positions plus velocities occupy about
119 kB per full-grid state; one million states need about 119 GB before loads,
contexts and database overhead. Disk usage must be monitored during the future
campaign. Write failures must not silently discard records. Exact sampling
continuation after trainer restart still needs its snapshot and sampler state
linked to the training checkpoint. See [DISK-REPLAY.MD](DISK-REPLAY.MD) for APIs,
usage and backup requirements.

## Loss and settling

The physical objective is the sum of hexahedral Neo-Hookean elastic energy and
the unchanged implicit-Euler inertial energy. The implemented unroll averages
the energy change after every inner update, normalized by the greater of the
initial energy and 1 joule. It adds a penalty for increases relative to the
preceding iterate. Starting energies, normalization scales and comparison
energies are detached. This is self-supervised optimization; no target shape
from a different solver is required.

The old model drifts even from force-free rest: its first update moved corners
by 0.060 mm RMS, and energy increased on all ten diagnostic updates. Reducing
the transformer depth alone does not address this behavior.

Proposed improvements for review are verified resting states and small
perturbations around them, with a penalty for drifting from a verified
equilibrium. Under gravity, equilibrium can be a bent shape; a canonical
straight beam is not automatically at equilibrium. Also consider exposing
the current physical force residual and making the correction vanish as that
residual vanishes. Those input/output changes and the added equilibrium loss
are not yet implemented or finalized. Choose them before fixing the new
training input schema and starting a campaign.

## Inversion — representation supported; solver changes proposed

The three axes can encode negative signed volume. Keep the coordinate frame
right-handed and retain the inversion in the local axes. One proposed frame
construction is:

```python
U, stretches, Vt = svd(F)       # Stretches sorted largest to smallest.
if determinant(U @ Vt) < 0:
    U[:, -1] *= -1
R = U @ Vt
local_axes = R.T @ F           # Use the original, unchanged F.
reconstructed_F = R @ local_axes
```

This preserves the deformation, including inversion, up to numerical
roundoff. Equal stretches or collapse can make the frame ambiguous; a
continuity rule based on the previous frame needs design and testing.

The current frame extractor and logarithmic energy still reject inverted
cells. Proposed work is to preserve the existing Neo-Hookean energy for
healthy positive volume and smoothly continue it below a positive threshold,
with finite derivatives through zero and negative volume. The continuation
and threshold require review and gradient tests. Add near-collapse and mild
inversion recovery examples, checking all eight Gauss-point Jacobians. A
positive cell-center determinant alone does not establish a valid hex.

## Validation, checkpoints and launch criteria

Use frozen network weights for an entire validation trajectory. Reuse held-out
initial shapes, velocities and materials across checkpoints; changing motion
then measures solver improvement. Also retain fixed-state validation queries
so loss comparisons are not obscured by changing live training trajectories.

Report mean, median and maximum relative energy through 100 optimizer
iterations, together with invalid-state counts. Treat near-zero initial
energies separately using absolute energy, displacement and force residuals;
their relative ratios can be misleading. Evaluate physical rollout survival,
speed, resting drift and fixed-corner accuracy. Failed cases must remain
visible rather than disappearing from aggregate curves.

Maintain epoch/update loss curves, curriculum stage, actual K and physical
step counts, data-source counts, material distribution and disk growth. Save
latest/best/periodic resumable checkpoints including optimizer, random states,
trajectory positions/velocities, archive identity and sampling population.
An initial budget of up to 500 epochs is a proposal, with convergence-based
stopping and explicit reporting of plateaus.

After plan approval: finish the trainer/data integration, resolve the chosen
settling and inversion changes, validate gradients, measure batch capacity
with activation checkpointing, and verify the updated four-GPU path. Neural,
feature and energy operations run on GPU; the existing differentiable sparse
fusion uses its CPU solve with an adjoint backward pass. Do not assume that
the previous four-GPU single-iteration run proves this new training loop.

The disk framework, material sampler and rollout adapter passed focused
tests; the complete experiment suite passed 200 tests on CPU/CUDA. These are
verification results, not evidence that a new model has been trained or that
inversion recovery is implemented.
