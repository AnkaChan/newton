# V2 plan: a learned optimizer for deformable hexahedral bodies

Date: 2026-09-24. This document records the retained physical decisions and
the implemented V2 trainer. **No new training campaign has been launched.**
Implementation and bounded verification are authorized; campaign settings and
launch still require review. The new pipeline has passed bounded full-grid
distributed checks; those checks do not establish learned convergence or
long-horizon physical stability.
See [implementation progress](v2-implementation-plan.md) and the current
[decision record](v1-DECISIONS.MD#39-v2-mixed-pool-trainer-implementation--2026-09-24).

Latest data-policy revision: generate intermediate states on the fly and
reproduce initial conditions with a seeded augmenter reset. Intermediate-state
disk replay is optional diagnostic tooling, not the training baseline.
Each initial state independently samples an inner-iteration count K and a
physical trajectory length H from the counts available at its curriculum
stage. Keep those counts until H physical steps finish, then reset and sample
again. The implemented maximum caps are K=32 and H=128. One overall
perturbation multiplier per initial case covers quieter starting states.
The selected trainer advances each sample in a mixed batch by one detached
network-and-fusion update, then makes one Adam update for that batch. Full
connected unrolling remains an opt-in comparison.

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

The network proposes changes to those same local axes. Recompute polar frames
from each current candidate, then freeze their derivatives for that proposal.
Convert the predicted axis increment into world coordinates and fit displacement
gradients at all eight Gauss points, with stiffness times rest-volume weights.
Add the solved displacement to the current shared corners. Pin elimination is
exact; a zero axis increment preserves the current compatible pinned shape,
including its corner warping. Full quadrature constrains the reconstruction;
one axis increment per cell still cannot express every eight-corner update.

The native semi-implicit rigid predictor supplies a once-per-physical-step
rigid-guided initializer through fusion. It is not reapplied as an accumulating
rigid rotation on every inner query. **It does not replace the original free
inertial target Y or the physical implicit-Euler inertia.** Elasticity is
compressible logarithmic Neo-Hookean energy at eight Gauss points per hex,
not a tetrahedral approximation. Positive pin masses remain part of the
physical mass and inertia model. This remains a clamped, contact-free problem.

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
initialization is the V2 baseline**; no silent checkpoint truncation. The
legacy trainer retains explicit saved architectures; the new mixed pipeline
has its own checkpoint format.

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
remains `(1, 1)` for compatibility; the implemented V2 configuration explicitly
selects `(0, 1)`. Initial-state metadata records the sampled scalar and range.

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
shape, velocity and material. The legacy dataset and epoch trainer retain their original material behavior.
The new `train_mixed` pipeline uses these independent per-trajectory materials.

`MixedHexSolverStep` implements batched per-instance material, mass, features
and energy while each object's material remains fixed for its K/H trajectory.
It runs one shared network forward over the mixed fixed-shape batch. Fusion
dispatches to each physical context's cached CPU PARDISO factor; continuously
sampled different matrices require their own factors. Context construction
prepares factors before their first query. Native handles are owned by their
process and remain outside module buffers and serialized tensors.

DDP synchronizes learned parameters and gradients with `broadcast_buffers=False`;
each rank owns its physical contexts and trajectory pool. The launcher exposes
the opt-in `mixed` pipeline alongside the legacy `epochs` pipeline. The new
path requires its own distributed parity, resume and capacity evidence; the
old four-GPU result does not verify this trainer.

A logical epoch defaults to **8,192 optimizer queries globally across all
ranks**, not 8,192 distinct starts or a fixed training seed set. With batch B
per rank and R ranks, an epoch has 8192/(B*R) Adam updates; configuration must
make this quotient integral. New starts form a continuing seeded stream.
Validation defaults to 512 fixed held-out starts partitioned across ranks and
reused across checkpoints. Training maps logical reset seeds to `2*seed` and
validation to `2*seed+1`, with distinct master seeds as well. This disjoint
even/odd namespace is needed because initial shape generation uses the reset
seed directly; changing only the master seed would not separate those shapes.

The candidate mixture is a configurable implementation default: 50% inertial,
35% noisy inertial, 10% previous physical positions and 5% rigid-guided
initialization. These reused historical probabilities are **not an approved
campaign decision**. Initializer perturbations are screened before a learned
query; backoff or fallback is recorded. A failed learned proposal is never
repaired or silently reset through this initializer path.

## Consecutive optimizer iterations and physical timesteps

The [interactive pool illustration](https://ankachen.com/artifacts/learned-intrinsic-solver/pool/index.html)
shows four batch slots fed by twelve active instances. It illustrates the
proposed schedule, not measured training or GPU performance.

Each initial state samples K, the number of optimizer iterations per physical
timestep, and independently samples H, the number of physical timesteps before
reset. Sample both once at trajectory creation. Every physical step of that
trajectory uses its sampled K, and its sampled H does not change midway.
After H completed physical steps, perform a seeded reset and independently
resample both counts for the next trajectory.

Maintain a pool of active trajectories at different inner-iteration and
physical-step indices. Form a mixed batch from that pool. Each selected
sample makes exactly one network-and-fusion proposal, evaluates its local
loss, and carries the resulting positions forward detached. Backpropagate
the mean local loss for the mixed batch, then make one Adam update. Clear
parameter gradients once per mixed batch. Each local backward pass still
differentiates through its network, fusion and energy; local frame
decompositions remain detached.

Weights therefore may change between successive inner iterations of one
physical solve. Hold that sample's original inertial prediction Y fixed for
all K iterations of its physical timestep, even when the shared network
weights change. Do not restart its solve or recompute Y after an Adam update.
After its Kth iteration, commit the new physical positions and velocities,
detach both, and begin its next physical timestep with a newly computed Y
unless H is complete and the trajectory resets.
Count H in completed physical timesteps, not optimizer proposals or batches.

`ActiveTrajectoryPool` implements a bounded pool larger than the training
batch, defaulting to 4B trajectories per rank. A deterministic dispatch FIFO
rotates all active members, including those awaiting preparation. Select the
next B distinct members, make exactly one proposal each, average their local
losses, update Adam once, and append returned members or their replacements
to the tail. This prevents a long-K cohort from monopolizing consecutive
batches while the rest of the pool waits. There is no padding or barrier
requiring every trajectory to finish a common phase.

After K proposals, reconstruct velocity from the **physical-step starting
positions**, `(solved_positions - physical_positions) / dt`, and set prescribed
velocities to zero. If H is not complete, the worker prepares the next step's
new Y and initializer once, clearing the old step's comparison energies.
At H, retire the context and reset with a new unique seed and newly sampled
K/H; do not compute an unused next physical state. CPU reset and advance work
runs in a bounded worker pool. Factorizations persist across inner iterations
and physical timesteps whenever the actual assembled matrix is unchanged.

With capacity 4B, a returned member normally has roughly three batches of
preparation overlap before its next dispatch turn. To make ordering independent
of worker completion timing, dispatch can wait for a pending head even when
later members are ready. This is the explicit fairness/reproducibility
tradeoff. It is not a whole-pool barrier, and it does not guarantee zero GPU
idle time. The CPU PARDISO forward and adjoint also synchronize each proposal.

Release each batch's graph after backward and retain only detached carried
states and diagnostics. There is a gradient boundary after every inner
iteration and every physical timestep. Peak graph memory need not grow with
K or H; total computation and the active state's storage still matter.
Generate intermediate states live with the current network. Do not archive
every state or sample a historical replay buffer in this baseline. At
dt=1/300 s, H=128 covers about 0.427 seconds; completing that trajectory does
not establish that it has settled.

The default available counts at the final stage are K in {1, 2, 4, 8, 16, 32}
and H in {8, 16, 32, 64, 128}. The implementation samples uniformly and
independently from the configured available sets; this sampling policy is a
provisional campaign default. Curriculum stages cap these sets rather than
assign one fixed K/H pair to all samples:

| Stage | Maximum available K | Maximum available H |
|---|---:|---:|
| Initial | 1 | 8 |
| 2 | 2 | 16 |
| 3 | 4 | 32 |
| 4 | 8 | 64 |
| 5 | 16 | 128 |
| Final | 32 | 128 |

For example, the stage with caps K=4 and H=32 permits independent draws from
{1, 2, 4} and {8, 16, 32}. Retain shorter counts as the caps grow. A stage
change affects new resets; active trajectories keep their previously sampled
K and H until completion. `MixedCurriculum` advances only after both a minimum
stage residence and consecutive qualifying validation results. Configurable
implementation defaults are 10 epochs per stage and two consecutive results
with no failures, finite decreasing mean energy, descent rate at least 0.9,
and every evaluated physical trajectory surviving. Qualifying epochs during
the minimum residence count toward patience. These thresholds remain
provisional pending campaign review; they are not evidence that K=32/H=128
is stable. Log counts, ages, stage and qualification status. If a stage stalls,
report it; never silently label a permanent K=1 run the completed curriculum.

The reusable inner solver now detaches carried positions by default. Its
`backward_detached` method and
`PhysicalRollout.windows(backward_each_iteration=True)` implement a different
schedule: keep weights fixed for a whole physical solve, backpropagate each
local loss divided by K immediately, accumulate gradients, and let the caller
make one Adam update afterward. This whole-solve accumulation helper remains
available, but it does not implement the selected mixed-state trainer.

Full connected unrolling remains an opt-in comparison. Connecting groups of
two to four inner updates is a future experiment with no demonstrated
training benefit over the detached baseline. The legacy epoch trainer still uses K=1. The new mixed pipeline implements
the active pool, sampled K/H, up to 128 physical steps per trajectory, and
rank-local pools under DDP. Completion of implementation is separate from
campaign stability and the verification checklist below. Training remains stopped.

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

Keep only active trajectory and optimizer states in memory during normal
training. A sample retains its physical positions and velocities, current
candidate, fixed Y and detached comparison energies for its current physical
step, plus its sampled K/H and progress counters. Advance it by one proposal
when selected into a mixed batch. Reset after its sampled H physical steps;
an Adam update or logging boundary does not itself trigger a reset.

A seed reproduces an initial condition, not a trajectory generated while
network weights were changing. Regeneration with newer weights intentionally
produces newer trajectories. Resumable checkpoints now include every rank's
active physical positions, velocities, candidate, fixed Y, pins, forces,
detached comparison energies, material specifications, K/H and progress,
unique seed cursor, budget RNG, dispatch order and ready/pending membership.
They also save network and Adam state, curriculum and plateau-controller
state, generator configuration/version metadata, and process RNG states.

Checkpointing drains preparation without changing dispatch or queue order,
then snapshots detached independent CPU tensors and all context specifications.
Distributed errors are coordinated before gathering rank states, and the file
is replaced atomically. Resume requires the same rank count, device class and compatible
physical/training configuration. It rebuilds native factors in each process
and restores the prepared queues without invoking extra initial resets.
CPU continuation is bit-for-bit exact in the regression test. Four-GPU runs
preserve discrete pool/RNG state and agree within float32 tolerance, but are
not bit-for-bit identical across runs/device assignments. Differences already
appear before a restart in the repeatability check; no deterministic GPU
reduction mode is forced for this baseline.
Runtime wait/preparation timings are diagnostic and are not deterministic
numerical state. This saves active state at checkpoint time, not every
intermediate training state. Restarting from seeds is a different reset policy.

The existing disk-retention framework remains available for explicit debugging
or selected failure records. It is not invoked by the reset augmenter or the
ordinary physical rollout. No existing artifacts are deleted. Its optional
API is documented in [DISK-REPLAY.MD](DISK-REPLAY.MD).

## Loss and settling

The physical objective is the sum of hexahedral Neo-Hookean elastic energy and
the unchanged implicit-Euler inertial energy. Keep the existing loss: energy
change from the physical solve's initial energy after every inner update,
normalized by the greater of that initial energy and 1 joule, plus the
existing penalty for increases relative to the preceding iterate. Starting
energies, normalization scales and comparison energies remain detached.
The inertial prediction remains unchanged throughout the inner solve.

For the selected mixed-state trainer, average the local losses from the one
proposal per selected sample and update the weights once for that batch.
Do not apply an additional 1/K weight to those mixed-batch local losses.
Whole-solve loss/K accumulation belongs to the helper alternative described
above. The local energy formula and increase-penalty weight are unchanged;
no new physical loss term is introduced.
This is self-supervised optimization; no target shape from a different solver
is required.

The old model drifts even from force-free rest: its first update moved corners
by 0.060 mm RMS, and energy increased on all ten diagnostic updates. Reducing
the transformer depth alone does not address this behavior.

Proposed improvements for review are verified resting states and small
perturbations around them, with a penalty for drifting from a verified
equilibrium. Under gravity, equilibrium can be a bent shape; a canonical
straight beam is not automatically at equilibrium. Also consider exposing
the current physical force residual and making the correction vanish as that
residual vanishes. Those input/output changes and the added equilibrium loss
remain deferred physics and feature research. They are not part of the mixed
trainer implementation, and no equilibrium loss or new feature channel is
silently added. Review them separately before committing to campaign claims
about settling. A zero network axis correction is an exact fusion no-op;
that algebraic property alone does not establish learned resting stability.

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

The implemented validator reports mean, median and maximum relative energy
through 100 optimizer iterations by default, along with invalid-state counts,
absolute energy, displacement and physical survival. Near-zero initial energies
are counted separately and excluded from relative ratios. Physical validation
defaults to eight steps with two optimizer queries per step; this is distinct
from training's sampled K/H. Weights stay frozen for each whole validation
trajectory, and failed cases remain visible instead of disappearing from the
population curves. Optimizer failures invalidate relative population curves
from the failed iteration onward; physical failures are reported separately.
Near-zero cases include absolute energy, displacement and free-corner
force-residual norms. Dedicated resting-drift studies remain a research item;
the current survival check does not substitute for them.

Maintain epoch/update loss curves, curriculum stage, sampled K/H distributions,
actual inner-iteration and physical-step counts, reset counts,
perturbation-scale distribution and material distribution. Save
latest/best/periodic resumable checkpoints including optimizer, random states,
generator configuration/version, seed schedule and active trajectory and
optimizer states, including sampled K/H and progress counters.
The implemented epoch limit defaults to 500, with validation-based
learning-rate reductions and explicit converged/stalled/epoch-limit statuses.
This is a configurable implementation default, not authorization to run a
500-epoch campaign. A training preparation or learned-proposal failure stops
the run and saves a per-rank failure diagnostic with active inputs, contexts,
model and optimizer state; no sample is silently replaced to hide a failure.
The normal live stream is not an intermediate-state archive.

Complete the verification checklist before declaring the new pipeline ready
for campaign review. Settling and inversion changes are deferred, independent
research tasks. Use activation checkpointing within an iteration if needed;
measure connected unrolling separately when comparing it. Neural, feature and
energy operations run on the configured CPU or GPU in float32, with TF32 and
AMP disabled. Differentiable sparse fusion uses cached CPU PARDISO factors
with float32 forward and transpose-adjoint solves.
Factor once per fixed fusion matrix and reuse it across solver iterations and
physical timesteps. Keep float32 for factorization and both solves; the
checkpoint reconstructs each process's native factors from its saved context.
The oneMKL runtime is an optional dependency of this experiment. Missing
PARDISO is an explicit setup error, not a silent fallback to SuperLU. Do not assume that
the previous four-GPU single-iteration run proves this new training loop.

## Historical verification context

The following results predate the mixed trainer and do not verify it.

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


## Current implementation verification checklist

Verified on 2026-09-24. See [commands, artifacts and limitations](v2-implementation-plan.md#verification-evidence).

| Check | Evidence |
|---|---|
| Complete experiment suite, including mixed physics, gradients, frozen frames, fusion, pool lifecycle, detached loss, curriculum, failure retention and CPU resume | 277 tests passed in 55.544 seconds on a claimed L40; CPU and CUDA cases enabled |
| K=32/H=128 scheduler lifecycle | 4,096 proposals, 127 physical advances and one final retirement/reset; callback-only test |
| Four-GPU gradient averaging with heterogeneous materials | Eight distinct objects, batch two per rank; reference Adam weight error at most 2.19e-11, moment relative L2 error 1.48e-7 |
| Mixed inner/physical updates and resume on full 10×10×40 grids | 64 bounded Adam updates with mixed budgets; K=32 completed on every rank; discrete checkpoint state preserved, float32 numerical continuation (not bit-exact GPU repeatability) |
| Full-grid validation curves | Four held-out cases through 100 optimizer iterations, plus separate two-step physical checks; no invalid cases in this bounded check |
| Batch capacity and timing | Batch 16 on one L40: peak 4.76 GiB, approximately 0.62 s forward/backward/Adam excluding setup, input preparation and validation; not a maximum-batch search |
| Candidate probabilities, stage thresholds and campaign budget | Configurable provisional defaults remain subject to campaign review |

No new training campaign has been launched. Bounded verification uses disposable
weights and does not demonstrate convergence, H=128 physical stability,
inversion recovery or resting equilibrium.
