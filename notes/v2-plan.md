# V2 plan: a learned optimizer for deformable hexahedral bodies

Date: 2026-09-24. This document records the current decisions and the proposed
next training campaign. **Training remains stopped. Review and approval of
the training plan are required before launching another campaign.** Implemented
helpers, proposed changes, and remaining integration work are distinguished
below.

Latest data-policy revision: generate intermediate states on the fly and
reproduce initial conditions with a seeded augmenter reset. Intermediate-state
disk replay is optional diagnostic tooling, not the training baseline.
The latest rollout target is 128 consecutive physical timesteps, with up to
32 optimizer iterations inside each timestep. Add one overall perturbation
multiplier per initial case to cover quieter starting states.

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

Add a global perturbation scale to multiply the whole displacement from rest
and the whole initial velocity field by the same scalar. This keeps the
multiresolution deformation pattern while varying its overall intensity.
For V2, sample the scale uniformly from 0 to 1 on its own seeded random
stream. Smaller values give shapes closer to rest and lower initial speeds.
Use a fixed zero scale for explicit undeformed, zero-velocity cases, and a
fixed scale of one to recover the previous augmentation. Material, gravity,
rest size and timestep are not multiplied. In particular, zero initial
perturbation under gravity is not automatically a static equilibrium.

The augmenter exposes `perturbation_scale_range`: `(0, 1)` selects V2's
variable intensity, `(0, 0)` gives canonical rest with zero velocity, and
`(1, 1)` preserves the previous seed-to-state mapping. The library default
remains `(1, 1)` for compatibility; the future V2 trainer must select `(0, 1)`
explicitly. Record the sampled scalar and its range in initial-state metadata.

The implemented material sampler draws Young's modulus and density independently
in logarithmic space, then derives the Lamé parameters:

| Quantity | Sampling |
|---|---|
| Young's modulus, E | Log-uniform from 1,000–1,000,000 Pa |
| Poisson's ratio, nu | Uniform from 0.2–0.49 |
| Density | Log-uniform from 100–10,000 kg/m³ |

Young's modulus alone is insufficient to determine both Lamé parameters. Use:

```text
mu     = E / (2 * (1 + nu))
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
```

The user selected a varying Poisson's ratio, sampled uniformly from 0.2 to
0.49 independently of Young's modulus and density. This replaces the provisional
fixed ratio of 0.3. The configurable sampler supports `0 <= nu < 0.5`.
The network and physical energy still receive lambda and mu. They are derived
together, not drawn independently. Initial-state
metadata records both forms and the generation version is `initial_state_v3`;
the changed material mapping preserves the shape/velocity and density streams.

Sample one uniform material per object and keep it fixed along that trajectory.
The same reset seed and augmentation configuration reproduce the initial
shape, velocity and material. The old production dataset and trainer still
use their original fixed material until the new training pipeline is integrated.

The current physical solver uses one material context per batch. Initially,
group trajectory minibatches by compatible material and switch physical models
between such batches; arbitrary mixtures of materials in one batch are not
implemented. Independent continuous material draws will usually be unique,
so grouping alone does not provide large batches. Before training, choose
either a material draw shared across each batch or extend the physical energy
and fusion path to handle different materials within a batch.
Across GPU ranks, synchronize learned parameters and gradients while keeping
each rank's physical material and mass buffers local. This needs explicit
verification in the new distributed trainer.

Use the previous scale of 8,192 training starts per logical epoch and 512 fixed
held-out seeds as a proposed starting point. These describe seed schedules,
not a stored collection of intermediate states. The old 50/35/10/5 candidate
mixture is historical, not a decided V2 continuation/reset/rest mixture.

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
The final target is 128 consecutive physical steps per segment. At dt=1/300 s
that covers about 0.427 seconds of physical time; it is not a maximum
trajectory lifetime or a guarantee of settling. Some trajectories must continue
into later motion and settling. Mix those continuations with fresh seeded resets and near-equilibrium
starts. The exact sampling mixture is still a training-plan choice for review.
Do not archive every intermediate state or sample a historical replay buffer
in this baseline.

The gradient boundary is still one physical timestep. Process its loss and
backward pass before advancing; 128 forward physical steps do not require a
128-step gradient graph. The longer horizon increases total computation, not
the intended gradient-window memory. Do not accumulate every window's graph.

For the new one-layer model, the proposed curriculum is:

| Stage | Optimizer updates per physical step | Physical steps per segment |
|---|---:|---:|
| Initial | 1 | 8 |
| 2 | 2 | 16 |
| 3 | 4 | 32 |
| 4 | 8 | 64 |
| 5 | 16 | 128 |
| Final | 32 | 128 |

Retain shorter solves after advancing. Use validation descent and trajectory
stability to decide when to advance; numerical thresholds and patience remain
to be selected before launch. Log actual counts and curriculum stage. If a
stage stalls, report it; never silently train only one iteration for the whole
campaign. The reusable K≤32 unroll and per-timestep gradient boundaries are
implemented, but the production epoch trainer is not yet wired to this loop.

## On-demand generation and reset

`InitialStateAugmenter.reset(seed)` regenerates the compatible shared-corner
shape, smooth velocity field and material from the canonical rest grid.
`reset()` repeats the most recently selected seed. An explicit new seed starts
a different case. Reset never perturbs the last simulated shape; returned
arrays are independent copies. Prescribed corners return to rest and their
velocities are zero. No intermediate-state archive is opened or written.

Use the existing multiresolution deformation generator. Keep the material
random stream separate so adding material sampling does not change the
existing shape/velocity sequence. Preserve float32 output and screen geometry
after conversion, evaluating determinants in float64 on those quantized
coordinates. This is a geometry screen, not a simulation stability proof.
A small configuration/seed record, including the generator
version, identifies an initial case; the seed alone does not identify changes
to grid size, augmentation ranges or implementation.

```python
from experiments.learned_intrinsic_solver.initial_state import InitialStateAugmenter

augmenter = InitialStateAugmenter(
    rest, master_seed=73, time_step=1 / 300, perturbation_scale_range=(0, 1)
)
initial = augmenter.reset(seed=42)  # Also reproduces its global perturbation scale.
# Advance the current physical state with the current network.
# At the next reset, regenerate from rest rather than storing its history.
same_initial = augmenter.reset()
different_initial = augmenter.reset(seed=43)
```

Keep only active trajectory positions and velocities in memory during normal
training. Train on each live step, detach its output, and carry it forward.
Reset selected trajectories when starting a new episode; do not reset every
short training segment or settling states will be underrepresented.

A seed reproduces an initial condition, not a trajectory generated while
network weights were changing. Regeneration with newer weights intentionally
produces newer trajectories. For exact training resume, the proposed checkpoint
includes the small active trajectory pool, in addition to weights, optimizer,
random states and seed schedule. This saves the current state at checkpoint
time, not all intermediate training states. Alternatively, restarting from
seeds is a trajectory reset and must be identified as such.

The existing disk-retention framework remains available for explicit debugging
or selected failure records. It is not invoked by the reset augmenter or the
ordinary physical rollout. No existing artifacts are deleted. Its optional
API is documented in [DISK-REPLAY.MD](DISK-REPLAY.MD).

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
initial shapes, velocities and materials across checkpoints by regenerating
the same held-out seeds; changing motion then measures solver improvement.
Also regenerate fixed initial-state validation queries so loss comparisons
are not obscured by changing live training trajectories.

Report mean, median and maximum relative energy through 100 optimizer
iterations, together with invalid-state counts. Treat near-zero initial
energies separately using absolute energy, displacement and force residuals;
their relative ratios can be misleading. Evaluate physical rollout survival,
speed, resting drift and fixed-corner accuracy. Failed cases must remain
visible rather than disappearing from aggregate curves.

Maintain epoch/update loss curves, curriculum stage, actual K and physical
step counts, continuation/reset counts, perturbation-scale distribution and
material distribution. Save
latest/best/periodic resumable checkpoints including optimizer, random states,
generator configuration/version, seed schedule and active trajectory states.
An initial budget of up to 500 epochs is a proposal, with convergence-based
stopping and explicit reporting of plateaus.

After plan approval: finish the trainer/data integration, resolve the chosen
settling and inversion changes, validate gradients, measure batch capacity
with activation checkpointing, and verify the updated four-GPU path. Neural,
feature and energy operations run on GPU; the existing differentiable sparse
fusion uses its CPU solve with an adjoint backward pass. Do not assume that
the previous four-GPU single-iteration run proves this new training loop.

The complete experiment suite passed 207 tests on CPU/CUDA after adding seeded
reset. After adding the overall perturbation multiplier, all 33 focused data
tests passed, including nine reset tests. They cover reproducibility, mutation
isolation, legacy shape/velocity parity, separate material draws, pins, RNG
isolation, zero/fractional/random multipliers and unrepresentable float32
geometry. One full-grid reset took 0.24 seconds on CPU and repeated exactly.
After deriving Lamé parameters from Young's modulus and Poisson's ratio,
all 36 focused data tests passed, including ten reset tests and seven material
tests. Known conversion values and recorded material provenance are covered.
These are implementation checks, not evidence that a new model has been
trained or that inversion recovery is implemented.
