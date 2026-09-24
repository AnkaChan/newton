# V2 plan: a learned optimizer for deformable hexahedral bodies

Date: 2026-09-24. This document records the current decisions and the proposed
next training campaign. **Training remains stopped. Review and approval of
the training plan are required before launching another campaign.** Implemented
helpers, proposed changes, and remaining integration work are distinguished
below.

Latest data-policy revision: generate intermediate states on the fly and
reproduce initial conditions with a seeded augmenter reset. Intermediate-state
disk replay is optional diagnostic tooling, not the training baseline.
Each initial state independently samples an inner-iteration count K and a
physical trajectory length H from the counts available at its curriculum
stage. Keep those counts until H physical steps finish, then reset and sample
again. The eventual proposed caps are K=32 and H=128. Add one overall
perturbation multiplier per initial case to cover quieter starting states.
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

The current `LearnedHexSolverStep` binds one physical context. The selected
mixed-state trainer needs batched per-instance material, mass and energy
inputs while keeping each object's material fixed for its K/H trajectory.
Independent continuous material draws will usually be unique; grouping only
identical materials does not supply the intended mixed batch. Keep one batched
network forward over the fixed-shape grid queries, not Python network calls
per object. Fusion may dispatch by physical operator/context and reuse cached
factors only when the actual assembled matrices match. Prepare required
factors before admitting instances to the ready pool. This heterogeneous
physical-context support remains pending and is not implemented by the
detached-iteration helper.
Across GPU ranks, synchronize learned parameters and gradients while keeping
each rank's physical material and mass buffers local. This needs explicit
verification in the new distributed trainer.

Use the previous scale of 8,192 training starts per logical epoch and 512 fixed
held-out seeds as a proposed starting point. These describe seed schedules,
not a stored collection of intermediate states. The old 50/35/10/5 candidate
mixture is historical, not a decided V2 initial-state mixture.

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

For the proposed implementation, keep a bounded active pool larger than the
training batch: roughly 4B instances per GPU is a starting point for batch
size B. Gather B ready queries, make exactly one proposal per member, average
their local losses, update Adam once, and scatter detached results back to
their instances. Do not pad shorter K/H schedules or wait for every
trajectory to finish a common phase.

After an instance reaches K and still has physical steps remaining, prepare
its next step once, including the new Y, frozen rigid target and initial
candidate. Park instances awaiting
preparation while other ready instances fill the next batch. At H, perform a
seeded reset and independently resample K/H; prefetch initial states so reset
work can overlap other batches. Use a bounded CPU preparation worker pool
and prepare work outside the critical path where possible. Reuse cached
PARDISO factorizations when the actual assembled matrix is the same, with
native handles owned by their process; different matrices require their own
factors. Do not refactor merely because an instance advances physical time.
These are proposed trainer mechanisms, not an implemented scheduling system.
The current CPU PARDISO solve inside
each batch still blocks that path, so this design does not guarantee an
always-busy GPU.

Release each batch's graph after backward and retain only detached carried
states and diagnostics. There is a gradient boundary after every inner
iteration and every physical timestep. Peak graph memory need not grow with
K or H; total computation and the active state's storage still matter.
Generate intermediate states live with the current network. Do not archive
every state or sample a historical replay buffer in this baseline. At
dt=1/300 s, H=128 covers about 0.427 seconds; completing that trajectory does
not establish that it has settled.

The proposed available counts at the final stage are K in {1, 2, 4, 8, 16, 32}
and H in {8, 16, 32, 64, 128}. Uniform independent sampling over the available
sets is a proposal; exact probabilities are not finalized. Curriculum stages
cap these sets rather than assign one fixed K/H pair to all samples:

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
K and H until completion. Use validation descent and trajectory stability to
decide when to advance; thresholds and patience remain to be selected before
launch. Log sampled counts, progress and curriculum stage. If a stage stalls,
report it; never silently train only one iteration for the whole campaign.

The reusable inner solver now detaches carried positions by default. Its
`backward_detached` method and
`PhysicalRollout.windows(backward_each_iteration=True)` implement a different
schedule: keep weights fixed for a whole physical solve, backpropagate each
local loss divided by K immediately, accumulate gradients, and let the caller
make one Adam update afterward. This whole-solve accumulation helper remains
available, but it does not implement the selected mixed-state trainer.

Full connected unrolling remains an opt-in comparison. Connecting groups of
two to four inner updates is a future experiment with no demonstrated
training benefit over the detached baseline. The production epoch trainer
still uses K=1. The mixed active pool, per-trajectory K/H scheduling, live
trajectories up to 128 steps, and updated multi-GPU trainer are not yet
implemented or integrated. Training remains stopped.

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
produces newer trajectories. For exact training resume, the proposed checkpoint
includes the small active trajectory pool and its per-sample K/H, progress,
candidate and fixed Y, in addition to weights, optimizer, random states and
seed schedule. This saves the current state at checkpoint time, not all
intermediate training states. Alternatively, restarting from seeds is a
trajectory reset and must be identified as such.

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

Maintain epoch/update loss curves, curriculum stage, sampled K/H distributions,
actual inner-iteration and physical-step counts, reset counts,
perturbation-scale distribution and material distribution. Save
latest/best/periodic resumable checkpoints including optimizer, random states,
generator configuration/version, seed schedule and active trajectory and
optimizer states, including sampled K/H and progress counters.
An initial budget of up to 500 epochs is a proposal, with convergence-based
stopping and explicit reporting of plateaus.

After plan approval: finish the trainer/data integration, resolve the chosen
settling and inversion changes, validate gradients, measure batch capacity
with the mixed batch's one-proposal graph, and verify the updated four-GPU path.
Use activation checkpointing within an iteration if needed; measure connected
unrolling separately when comparing it. Neural, feature and energy operations
run on GPU; the existing differentiable sparse
fusion uses a cached CPU PARDISO factorization with an adjoint backward pass.
Factor once per fixed fusion matrix and reuse it across solver iterations and
physical timesteps. Keep float32 for factorization and both solves; the
checkpoint reconstructs each process's native factors from its saved context.
The oneMKL runtime is an optional dependency of this experiment. Missing
PARDISO is an explicit setup error, not a silent fallback to SuperLU. Do not assume that
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
