# Takeover note: optimize the learned intrinsic solver implementation

Written 2026-09-26 for an agent working on another machine. Everything referenced
here is in the GitHub repository at the commit named below; nothing depends on
files that exist only on the original machine. The `notes/v2-plan.md` that is
committed at that revision is the superseded pre-revision plan; the design the
code implements is recorded in `notes/implementation-walkthrough.md`, which is
committed, and the constraints below follow it.

## Where the code is

- Repository: `github.com/AnkaChan/newton` (fork of `newton-physics/newton`).
- Branch: `ankac/learned-instrinic-solver`. Start from its head; the implementation
  described below is commit `f4298870` ("Revise learned hex solver to the V2 plan"),
  followed by documentation-only commits.
- Package: `experiments/learned_intrinsic_solver/` (58 modules, about 18.5k lines
  of Python plus 12.5k lines of tests; PyTorch, a CPU PARDISO bridge, and a
  Warp/Newton rigid-pose predictor that is used only to build the initial candidate
  in `MixedHexSolverStep.prepare`). Tests: `experiments/learned_intrinsic_solver/tests/`
  (453 tests).
- Read first: `notes/implementation-walkthrough.md` (or open the generated
  `notes/implementation-walkthrough.html` in a browser). It explains the
  mixed-trainer path (network, energy, damping, frames, fusion, input assembly,
  history, multiscale fields, initial state, trajectories) with verbatim excerpts,
  a pipeline map, findings and a test-coverage table; the launcher, curriculum,
  reporting, rollout and legacy tools are not covered. Regenerate it after code
  changes with
  `python notes/build_walkthrough_html.py --md notes/implementation-walkthrough.md --prefer experiments/learned_intrinsic_solver`.
- Repository conventions: `AGENTS.md` and `CODING_GUIDELINES.rst` at the repo root
  (feature branch, imperative commit messages, `uvx pre-commit run -a` before committing).

## Setup and checks

```bash
uv sync --extra dev --extra torch-cu12            # PyTorch with CUDA
uv run --no-sync python -m unittest discover -s experiments/learned_intrinsic_solver/tests
uvx --with virtualenv pre-commit run -a
```

The fusion solver needs oneMKL PARDISO on the CPU; see
`experiments/learned_intrinsic_solver/requirements-pardiso.txt` and `pardiso.py`.
GPU tests skip without CUDA. Keep `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` when
timing, because the trainer runs four ranks on one host with two CPU threads each.

## What the trainer does per optimizer query (cost model)

One query processes a batch of B objects (B = 16 per GPU rank in the campaign
configuration), each a 10x10x40 hexahedral grid: C = 4000 cells, P = 4961 shared
corners, F = 4840 free corners. Per query, in this order:

1. `train_mixed.run_training` evaluates the energy of the candidate under
   `no_grad` to get `E_before` (`train_mixed.py`, one full [B, C, 8, 3, 3]
   elasticity pass).
2. `input_assembly.assemble_inputs` computes the frames (batched SVD on the GPU),
   then `objective_gradient` runs a second full energy pass with autograd to get
   `dE/dX`, then `HexFusion.project_gradient` performs one transposed sparse solve
   per object on the CPU (`mixed_physics.py`, the loop that calls
   `context.fusion.project_gradient(position_gradient[i : i + 1])`).
3. The network runs once for the batch (`network.py`).
4. `HexFusion.fuse` performs one forward sparse solve per object on the CPU
   (`mixed_physics.py`, the loop that calls `context.fusion.fuse(...)`).
5. The energy of the fused shape is evaluated a third time, with autograd
   (`mixed_physics.py`, `_energy`).
6. `loss.backward()` runs one transposed sparse solve per object on the CPU inside
   `fusion._FusionSolve.backward`, then Adam.

Every CPU solve moves data GPU -> CPU -> GPU synchronously through NumPy
(`fusion.py`). Measured on one NVIDIA L40 with the campaign configuration
(batch 16, full grid, two CPU threads): forward 0.27 to 0.46 s, backward plus Adam
0.24 to 0.30 s, peak 4.8 GiB of GPU memory. A training epoch of 128 updates on
four ranks plus validation took about 500 s in the final curriculum stage; the
100-iteration validation on 512 held-out starts dominates that time because
`mixed_validation._free_force_residual_norms` runs an extra autograd energy pass
at every iteration for every sample, and `_TrajectoryFactory._candidate`
regenerates a multiscale noise field (`multiscale.generate_multiscale`) for every
perturbed candidate on the CPU preparation workers.

## Known inefficiencies worth attacking

- **Three full elasticity passes per query.** `E_before` (no_grad) and the
  gradient-feature pass (`objective_gradient`) evaluate the same energy at the
  same candidate; one autograd pass can return both the value and `dE/dX`. The
  `E_before` of query k+1 also equals the `E_after` of query k when the candidate is
  the previous fused shape (not after a physical `advance`, where the candidate is
  re-initialized).
- **3B serialized CPU sparse solves per query, all against factors that are
  scalar multiples of one matrix.** Each object owns a PARDISO factor because
  `register_context` builds `HexFusion(..., cell_weights=stiffness * h^3)` per
  material (`mixed_physics.py`). For a uniform material every operator in
  `HexFusion` (the stiffness `K_ff`, the target operator and the fixed coupling) is
  `stiffness` times its unit counterpart, so the factor cancels in every output:
  one shared `HexFusion(rest, fixed, cell_weights=None)` reproduces `fuse`,
  `project_gradient` and the backward of every per-material context without any
  right-hand-side scaling (checked in float64 on a 2x2x3 grid, differences about
  1e-16). `fusion._columns` already packs `[F, 3B]` right-hand sides, so all B
  objects could share one factor and one multi-column solve per stage. The design
  note that is committed says contexts with different assembled matrices need
  appropriate factors and not to assume one factor works for all materials, so a
  shared unit factor must be justified by this scale-cancellation identity and
  covered by a test. A GPU sparse solver (cuDSS, CHOLMOD on device, or a
  preconditioned CG warm-started from the previous solve) would remove the
  synchronous transfers entirely.
- **Validation residual pass.** The forward already returns
  `force_residual_norm` for the pre-update candidate; the validator recomputes the
  same float32 gradient with a separate autograd pass at every iteration and only
  accumulates its norm in float64 (`mixed_validation.py`,
  `_free_force_residual_norms`), so the two differ only by the norm's accumulation
  precision. Reusing the forward's gradient for intermediate iterations, or
  batching the residual evaluation, would cut validation time substantially.
- **Multiscale regeneration.** Half of all candidates call `generate_multiscale`
  on a CPU worker (and half of all resets therefore run it twice); its per-cell
  Jacobian screen is discarded afterwards. Caching fields per seed or generating
  on the GPU would relieve the two preparation workers, but the draws must stay
  identical (see the reproducibility constraint below).
- **Attention memory.** `IntrinsicTransformerLayer` chunks queries to bound
  temporaries, but autograd keeps every chunk's gathered keys and values
  (B x N x 27 x 128 each, plus the edge_val output of the same size) and the
  softmax weights (B x N x 27 x 4). A fused or memory-efficient attention over the
  fixed 27-slot neighborhood is a natural target.
- **Damping metric.** When any object in a batch is damped, the metric difference
  is evaluated for the whole batch (`mixed_physics.py`, `_energy`). In the campaign
  configuration every object is damped (`damping_range` (10, 1000) Pa s), so this
  branch is always taken and offers no saving there; it only matters for runs that
  mix damped and undamped contexts.

## What must not change

These are the design decisions the code implements (recorded in the walkthrough);
optimizations must reproduce them exactly or within float32 tolerance:

- Stable Neo-Hookean density with `mu_NH = mu`, `lambda_NH = lambda + mu`, finite
  through collapse and inversion; no rejection or shortening of inverted states.
- Eight-point (2x2x2) Gauss quadrature per cell in both the energy and the fusion
  fit. This is the retained current implementation pending a deferred
  one-versus-eight-point ablation, not a settled decision; do not change it as
  part of an optimization.
- Implicit-Euler objective with the inertial prediction `Y` and the damping anchor
  fixed for the whole physical step; VBD metric damping with all nine entries.
- Fusion is the weighted least-squares fit of one displacement per shared corner
  to the per-cell world targets over the eight Gauss points, with cell weights
  `stiffness * h^3` and normalized quadrature weights, prescribed corners
  eliminated exactly, and the physical energy evaluated on the fused shape. Any
  replacement solver must reproduce that fit, not approximate it differently.
- Frames: the closest proper rotation `U diag(1, 1, det(U V^T)) V^T` with the
  clamped-face tie-break; frames are constants for backpropagation; `A = R^T F`.
  Frames are recomputed from the current candidate at every query and never
  carried or cached across queries; the `frames=` replay argument of
  `assemble_inputs` is for diagnostics only.
- Six nine-value input blocks per cell in the receiving frame, including the
  projected objective gradient `unpack(B^T K_ff^-T g_free)`, LeCO normalization
  (RMS floor 1e-12, clip +/-10, log RMS scalar), history flag; 61 state values, 6
  conditioning channels, 24 edge values; one radius-one transformer block, width
  128, four heads, edge width 64; per-cell step size bounded by 0.05.
- Optimizer history stored in world coordinates, carried across physical steps,
  cleared at reset; the gradient feature and history are detached.
- LeCO per-update loss with the material-aware energy floor; mean over the mixed
  batch; one Adam update per batch; carried candidates detached.
- Candidates: each new candidate is inertial or perturbed inertial with
  probability 0.5 each, drawn from the trajectory's seeded stream; the multiscale
  seed and the 1 to 10 percent RMS scale come from the same stream, so any caching
  or relocation of the noise generation must reproduce those draws exactly.
- Deterministic FIFO dispatch, per-trajectory seeded RNG streams and the
  `mixed_pool_v2` checkpoint contents (pool records, RNG state, history payloads,
  context specs) must be preserved so a run resumes and reproduces the same
  initial states and candidate draws.
- Network, features, energy and fusion run in float32 with TF32 and AMP off.
  Float64 is used deliberately in a few bounded places that must be preserved:
  the tie-cell frame recomputation (`frames.py`), the energy-floor formula
  (`mixed_physics.py`) and diagnostic reductions such as the validation residual
  norms. Float64 fusion exists only for reference checks.
- Legacy 38/86 input schemas stay rejected; checkpoints must never be reshaped.

## Acceptance criteria for an optimization

- The full test suite passes and pre-commit is clean.
- Trainer parity: run a one-update job with the modified trainer
  (`queries_per_epoch = batch_size x world_size`, `max_epochs = 1`, as in
  `tests/test_mixed_training_reference.py`), then
  `python -m experiments.learned_intrinsic_solver.mixed_training_reference --initial <run>/checkpoints/initial.pt --after <run>/checkpoints/latest.pt --output <report.json> [--device cuda]`
  must report `passed: true`. It rebuilds the first global batch in one process,
  repeats the update, and compares parameters and Adam moments against the saved
  checkpoint within its tolerances.
- Validation must still record the free-corner residual norm for every sample at
  iteration 0 and after every update (including the last), keep the mean
  final-iteration residual with survival eligibility as the selection metric, and
  keep failed samples visible; reusing the forward's pre-update residual is
  allowed for intermediate iterations only if the final post-update residual is
  still computed.
- Add a test whenever a shortcut relies on a mathematical identity (for example a
  shared unit factor, or reusing `E_after` as the next `E_before`) that compares
  against the straightforward computation on a small grid in float64.
- Report measured timings for the campaign configuration (batch 16, full grid,
  two CPU threads per rank) before and after, for forward, backward plus Adam,
  validation per epoch and peak GPU memory.
- Keep changes on a feature branch off `ankac/learned-instrinic-solver`; do not
  rebase or force-push that branch.

## Out of scope here

Training campaign results, dashboards and checkpoints are not in the repository.
The legacy single-material pipeline (`train_epochs.py`, `train_smoke.py`,
`unrolled_solver.py`) shares the solver step but is historical; optimize the mixed
trainer path (`train_mixed.py`, `mixed_physics.py`, `input_assembly.py`,
`fusion.py`, `mixed_validation.py`).
