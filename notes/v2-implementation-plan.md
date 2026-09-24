# V2 trainer implementation and verification

Scope: complete the reviewed trainer, retain the physical decisions in
[v2-plan.md](v2-plan.md), verify the implementation, then commit and push the
research branch. **No new training campaign has been launched.** Bounded
verification and provisional implementation defaults are separate from
campaign approval.

## Implemented components

1. `MixedHexSolverStep` supplies one batched learned proposal with independent
   material, density, mass and energy for each object, native rigid initializer
   preparation, and cached process-local PARDISO fusion. Physical contexts
   remain outside DDP buffers; learned parameters and gradients are shared.
2. `ActiveTrajectoryPool` samples K and H once per seeded reset, advances each
   selected member by one proposal, prepares physical boundaries in bounded
   workers, and resets only after H completed steps. Its dispatch FIFO rotates
   every member and replacement fairly; completed-future timing cannot change
   the training order. At 4B capacity there are roughly three other batches of
   preparation overlap. Dispatch may wait for a pending head while later
   members are ready; there is no padding or all-trajectory phase barrier.
3. `train_mixed` makes one Adam update per full mixed batch and detaches carried
   states after backward. It preserves each physical step's original Y and
   starting positions for velocity reconstruction. Local losses retain the
   initial-energy normalization and preceding-iterate increase penalty,
   without additional 1/K scaling.
4. `MixedCurriculum` applies validation-gated caps while retaining shorter K/H
   choices. Active budgets stay fixed. Atomic checkpoints save every rank's
   pool, dispatch/queue order, material specifications and RNG/seed state,
   alongside network, Adam, curriculum and plateau-controller state. Resume
   rebuilds native factors without extra initial resets. Failure diagnostics
   retain failed inputs and model state rather than silently replacing samples.
5. The launcher has an explicit `mixed` pipeline, separate from legacy `epochs`.
   Frozen-weight held-out validation, update/epoch loss reports and resumable
   latest/best/periodic/final checkpoints are integrated. Old three-layer
   checkpoints retain their explicit architectures; new V2 runs start fresh
   with the one-layer `[1]` model and never truncate an old checkpoint.

## Defaults awaiting campaign review

- A logical epoch is 8,192 optimizer queries globally across ranks, not 8,192
  unique starts. Training starts form a continuing seeded stream.
- Validation uses 512 fixed held-out starts. Train and validation augmentation
  seeds occupy disjoint even/odd namespaces in addition to distinct master
  seeds, because the original shape generator uses the reset seed directly.
- The pool contains 4B members per rank, with two preparation workers by
  default. Available K/H values are drawn uniformly and independently.
- The candidate mixture is a configurable 50/35/10/5 split between inertial,
  noisy inertial, previous positions and rigid-guided initialization. Reusing
  these historical probabilities does not turn them into a user-approved
  campaign setting.
- Curriculum defaults require at least 10 epochs in a stage and two
  consecutive qualifying validations: no failures, finite decreasing mean
  energy, descent at least 0.9 and complete physical survival. The stage caps
  remain (1,8), (2,16), (4,32), (8,64), (16,128), (32,128).
- The provisional run limit is 500 epochs with validation-based learning-rate
  reductions and explicit plateau/stalled reporting. No campaign is running.

## Retained physics and deferred work

The implementation preserves original implicit-Euler inertia and Y, physical
step-based velocity, per-query frozen polar frames, exact prescribed corners,
float32 with TF32/AMP off, eight-Gauss-point hex elasticity and displacement-
gradient fusion, exact zero-increment no-op reconstruction, and the cached
PARDISO forward/transpose-adjoint bridge. Material remains independently
sampled E/nu/density with derived Lamé parameters and is constant per trajectory.
No inversion energy continuation, contact, equilibrium loss, line search or
new feature channel is part of this integration. Those remain separate
research and campaign-review questions.

## Verification evidence

Verified 2026-09-24 on NVIDIA L40 GPUs (48 GB advertised), float32 throughout,
TF32/AMP disabled, cached CPU oneMKL PARDISO, and two Torch/MKL/OpenBLAS threads
for the consolidated suite. These are disposable bounded verification runs;
no larger training campaign was started.

- **277 experiment tests passed in 55.544 s.** Command:
  `uv run --no-sync python -m unittest discover -s experiments/learned_intrinsic_solver/tests`
  after an exclusive GPU claim. This includes mixed-material independent
  forward/gradient parity, CPU checkpoint continuation, detached loss, fair
  pool transitions, curriculum gating, pin/geometry checks and failure paths.
  Full log: `generated/verification/mixed-final-tests.log`.
- **Full repository pre-commit checks passed:**
  `uvx --with virtualenv pre-commit run -a`.
  Log: `generated/verification/mixed-precommit.log`.
- **K=32/H=128 lifecycle:** callback-only test executes all 4,096 proposals,
  verifies exactly 127 physical advances and one retirement/reset, and checks
  that no unused next step is prepared. This tests scheduling, not physical
  stability over 128 learned steps.
- **Four-GPU first-update reference:** full 10×10×40 grids, two members per
  rank, eight distinct materials. A single-process concatenated-batch reference
  matches the DDP Adam update: maximum parameter difference 2.183e-11, Adam
  moment relative L2 difference 1.471e-7. All four rank parameter hashes agree.
  CLI: `python -m experiments.learned_intrinsic_solver.mixed_training_reference`
  with `--initial`, `--after`, `--device cuda`, `--output`.
  Artifact: `generated/verification/mixed_gpu_reference.json`.
- **Full-grid mixed progression and resume:** 64 disposable Adam updates,
  with initial K/H budgets 1/8, 2/16, 4/32 and 32/128 on each rank. Every rank
  completes a K=32 physical solve. This final-stage verification fixture uses
  step-size bound 0.001 to test scheduling before convergence; the separate
  first-update/capacity checks use the normal 0.05 bound. It is not a curriculum
  training result. Compare 64 uninterrupted updates with 32 + restore + 32.
  CPU tests are bit-exact. GPU discrete state, materials, seeds, dispatch,
  budgets, counters and RNG states match exactly; floating results do not.
  Maximum weight difference is 1.192e-7, Adam first moment 3.130e-7, positions
  at most 7.153e-7 m and velocities at most 3.577e-5 m/s. Differences begin at
  update two, before restart. Device assignment differs; floating GPU reduction
  order is a plausible, unisolated cause. Numerical audit tolerances were
  chosen after inspecting these diagnostic differences, not pre-registered
  acceptance bounds. Artifact: `generated/verification/mixed_gpu_resume_audit.json`.
- **New validation on four GPUs:** four fixed held-out full-grid cases through
  100 optimizer iterations and independent two-step physical checks. Zero
  optimizer/physical failures. Artifact directory:
  `generated/verification/mixed_validation_gpu/` (JSON, CSV, SVG, HTML and checkpoints).
  Unit tests additionally inject late optimization and physical failures,
  preserving earlier curve points and separate survival counts; near-zero
  energy tests verify absolute energy, displacement and free-force residuals.
- **Batch capacity:** batch 16, pool 64 on one L40, full grid and 320,142 model
  parameters. Peak CUDA allocation 4.761 GiB. One cold measured update:
  input/wait/transfer 0.197 s, forward 0.291 s, backward plus Adam 0.324 s.
  Complete startup/checkpoint/validation run: 34.01 s. These timings are a
  bounded capacity check, not a steady-state throughput benchmark or maximum
  batch search. Artifact: `generated/verification/mixed_capacity_b16/report.json`.

The launcher commands used `--pipeline mixed --workers 4` (or one for the
capacity check), explicit small JSON configurations, fresh verification
output paths and bounded timeouts. Raw artifacts are local ignored outputs;
the source and tests provide reproducible checks.

Implementation is ready for campaign review. CPU fusion and preparation can
still cause GPU idle time. Detached training and valid bounded rollouts do
not establish convergence, H=128 physical stability, inversion recovery or
resting equilibrium. The provisional settings above remain reviewable; no
campaign is running.
