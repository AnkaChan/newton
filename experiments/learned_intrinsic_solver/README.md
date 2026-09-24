# Learned intrinsic solver: test input data

Implements the cuboid and augmentation scope of [M01_DataGeneration.md](../../notes/M01_DataGeneration.md). This is an experimental CPU/NumPy data helper; its API and schema may change.

A node means an occupied voxel **cell**. The cuboid uses cubic voxels with one shared corner array. Generation supplies identity deformation matrices and zero cell velocities. Augmentation supplies independent random deformation and velocity fields per cell.

## Generate a sample

Run from the worktree root:

```bash
uv run --no-sync python -m experiments.learned_intrinsic_solver.data \
  --cells 4 3 2 \
  --cell-size 0.1 \
  --deformation-amplitude 0.1 \
  --velocity-amplitude 0.5 \
  --seed 42 \
  --output experiments/learned_intrinsic_solver/generated/cuboid.npz
```

This creates 24 cells with physical extents 0.4 by 0.3 by 0.2 meters and 60 shared rest corners. Use `--origin X Y Z` to change the minimum rest corner. Both augmentation amplitudes can be zero.

## Python interface

```python
import numpy as np

from experiments.learned_intrinsic_solver.data import augment_grid, generate_cuboid

rest = generate_cuboid((4, 3, 2), cell_size=0.1)
sample = augment_grid(rest, deformation_amplitude=0.1, velocity_amplitude=0.5, seed=42)

with np.load("experiments/learned_intrinsic_solver/generated/cuboid.npz", allow_pickle=False) as saved:
    axes = saved["cell_deformation"]  # (24, 3, 3), material axes are columns
```

Augmentation returns a new sample with independent arrays and leaves its input untouched. It uses a local seeded random generator.

## Data layout

Let `C = nx * ny * nz` and `P = (nx+1) * (ny+1) * (nz+1)`.

| Field | Shape | Meaning |
|---|---|---|
| `cell_counts` | 3 | Cell counts along material x, y, z |
| `cell_size` | scalar | Rest voxel edge length in meters |
| `corner_rest_positions` | `P x 3` | Shared rest geometry in meters |
| `cell_corner_indices` | `C x 8` | Each cell's indices into the shared corners |
| `cell_rest_centers` | `C x 3` | Rest centers in meters |
| `cell_neighbors` | `C x 6` | Face-neighbor cell indices; -1 outside the cuboid |
| `cell_exposed_faces` | `C x 6` | Boolean exposed-face flags |
| `cell_deformation` | `C x 3 x 3` | Dimensionless deformation matrices with axes as columns |
| `cell_velocity` | `C x 3` | Cell velocity vectors in meters per second |

Cells and corners use x/y/z ordering with z varying fastest. Local corner order is `000, 001, 010, 011, 100, 101, 110, 111`. Face order is `-x, +x, -y, +y, -z, +z`. Numeric arrays use float64 and indices use int64. The archive also stores the seed and augmentation amplitudes.

## Augmentation meaning

For each cell, sample a matrix `E` whose entries are uniform in `[-a, a]`, then set `F_new = (I + E) @ F_old`. Require `0 <= a < 1/3`. This bound keeps the incremental deformation nonsingular with positive orientation, preserving that property for valid input matrices. Target axes retain their lengths; they are not normalized to unit vectors.

Velocity receives a uniform additive perturbation in `[-v, v]` **per component**, in the same common coordinate frame. The magnitude of the full three-dimensional perturbation may exceed `v`. These componentwise distributions are not claimed to be rotationally invariant.

The generated deformation matrices are in a common coordinate frame, before any per-cell polar-frame encoding. Independent cell matrices need not correspond to one compatible deformed mesh. Likewise, cell velocities are not shared-corner velocities. The stored corners remain **rest positions**. These samples exercise later encoding and assembly; they are not simulated trajectories, reconstructed current shapes, or ground-truth physical solutions.

Arbitrary-shape voxelization remains future work. The learned solver and GPU
training prototype are described below; they use compatible shared-corner
multiscale shapes rather than these independent cell samples.

## Interactive cell frames

Build the static 10 by 10 by 40 cell inspector with a deterministic random seed:

```bash
uv run --no-sync python -m experiments.learned_intrinsic_solver.cell_frames \
  --seed 0 --deformation-amplitude 0.15
```

Serve `generated/cell_frames/` with a static HTTP server and open `index.html`.
The directory contains all browser assets locally, plus `sample.npz` with the
exact geometry and frame arrays. `--output` changes the destination and
`--cell-size` changes the canonical voxel edge length (default 0.025 meters).
This geometry-only test uses SciPy for the existing corner projection and
does not run a simulation or require a GPU.

The independent augmented cell targets are projected onto shared corners, with
the material z=0 side fixed. The inspector then recomputes each cell's actual
center deformation gradient `F`: opposite material-face centroid differences
divided by the rest edge length. The polar rotation `R` supplies an orthonormal
frame at that cell's deformed center. Its three deformation axes in the local
frame are the columns of `U = R.T @ F`; their lengths retain stretch and their
angles retain shear. One center matrix does not describe all corner warping.

Click a cell to see its frame on the deformed grid and its deformation axes in
the linked view. The separate-window button opens a synchronized detail window;
the embedded view remains available if popups are blocked. Layer controls and
cell coordinates expose interior cells. Cells with inverted or nearly singular
center gradients are flagged and do not display a misleading local frame.

## Multiscale initial shapes

Generate 20 seeded initial shapes with automatically derived control grids:

```bash
uv run --no-sync python -m experiments.learned_intrinsic_solver.multiscale_frames
```

The new artifact is `generated/multiscale_frames/`. Its seed selector covers
seeds 0–19; its contribution selector compares the combined shape with each
coarse, middle, or fine field alone. The displayed cell frames, local axes,
and linked window all update with the actual selected shape. Exact NumPy
downloads contain the canonical corners, current corners, random controls,
per-level displacement fields, and the applied common amplitude scale.

`multiscale.generate_multiscale(rest, seed=...)` produces compatible shared
corners directly. Each level samples independent uniform displacement-vector
components, zeros its entire material z-min control plane, and interpolates
in rest coordinates. The fields are added to the unchanged canonical rest
positions. A level may produce bending, shear, compression, and twisting;
it is not constrained to pure bending. Existing `augment_grid` behavior is
unchanged.

The default hierarchy starts at the actual voxel spacing, doubles target
spacing until no axis has more than four control intervals, and retains up to
three representative levels including the finest and coarsest. Every grid
spans the exact same rest bounds, even for odd voxel counts and shifted
origins. For 10×10×40 voxels this gives **2×2×4, 4×4×11, and 11×11×41 control
points**; these are derived values, not hardcoded grid dimensions.

Default `strength=0.5` provides a total per-component amplitude budget of half
the shortest rest side. Weights proportional to target spacing to power 1.5
split that budget across the retained levels. The default requested bounds
are ±109.59, ±13.70, and ±1.71 mm per component. These are component bounds,
not bounds on a displacement vector's norm. `--strength` changes the budget;
`--seed-start`, `--count`, and `--output` control export.

The generator deterministically halves a common amplitude scale until both
the combined shape and each isolated contribution pass a minimum volume-ratio
margin of 0.2. Checks include Newton's alternating five tetrahedra per voxel,
and trilinear Jacobians at all eight corners, eight two-point Gauss locations,
and the center of every voxel. Effective strengths and halvings are recorded
and shown in the inspector. These finite checks do not prove continuous
global injectivity or exclude distant self-intersections. The output is a
static initial geometry; it does not run VBD or change the physical solver.

## Newton VBD recordings

`vbd_samples.py` runs the seeded augmenter on a canonical 10 by 10 by 40 cuboid,
projects the independent cell fields onto compatible shared initial corners,
fixes the material z=0 end, and records 10 seconds with Newton VBD and ViewerGL.
The rest model remains canonical; only the initial simulation states receive
the projected augmentation. Newton's physical inertia and solver are unchanged.

```bash
source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh learned-intrinsic-vbd-recordings occupy
uv run --no-sync --with imageio --with imageio-ffmpeg python \
  -m experiments.learned_intrinsic_solver.vbd_samples --seed-start 0 --count 20
```

Use `--seed 7` for one deterministic initialization, `--resume` to skip completed
cases, and `--output` to select a destination. The development environment needs
SciPy, Newton's viewer dependencies, and the local `newton_capture` helper from
`$AI_LOGS/Newton/tools`. GPU trajectories are not claimed to be bitwise reproducible.

The generated directory `generated/vbd_10x10x40/` contains the manifest, exact
initial-state archives, videos, posters, and per-case metrics. To verify media
and build its standalone HTML gallery:

```bash
uv run --no-sync --with imageio --with imageio-ffmpeg python \
  -m experiments.learned_intrinsic_solver.verify_vbd_samples \
  experiments/learned_intrinsic_solver/generated/vbd_10x10x40
uv run --no-sync python -m experiments.learned_intrinsic_solver.vbd_gallery \
  experiments/learned_intrinsic_solver/generated/vbd_10x10x40
```

## Shared project webpage folder

Refresh one portable folder containing all five visualization pages:

```bash
uv run --no-sync python -m experiments.learned_intrinsic_solver.build_site
```

This copies the current generated outputs into `generated/webpages/`, with a
project index and the subfolders `vbd/`, `cell-frames/`, `multiscale/`,
`round-trip/`, and `round-trip-100/`. The last page contains the 100-seed float32
round-trip error analysis and selected 3D comparisons. Each copied main page
links back to the project index. Videos,
data, popup windows, and downloads remain inside the bundle. The original
generated folders are preserved, and no symlinks are used. The builder checks
available space and relative HTML links before replacing an earlier bundle.

To publish the refreshed project folder through the existing route:

```bash
uv run --no-sync python /home/horde/.codex/skills/publish-artifact/scripts/publish_artifact.py \
  --source experiments/learned_intrinsic_solver/generated/webpages \
  --slug learned-intrinsic-solver --base-url https://ankachen.com --verify
```

The shared entry point is
[the project visualization index](https://ankachen.com/artifacts/learned-intrinsic-solver/index.html).
The earlier separately published page URLs remain available.

## PyTorch transformer baseline

`network.IntrinsicTransformerLayer` is the reusable `torch.nn.Module` block.
`network.IntrinsicSolverNetwork` stacks **three local blocks, hops [1, 1, 1]**,
with 128 hidden features and four heads by default. Each cell attends to its
26 face/edge/corner neighbors plus itself. Boundary slots remain present and
are masked before softmax. There is no fixed geometric attention prior.

The layer includes learned `edge_bias` and `edge_val`, pre-normalization,
residual attention and SiLU feed-forward branches, and FiLM conditioning.
Query chunking limits temporary gathers while preserving the whole neighbor
softmax. It does not guarantee bounded memory for an entire training rollout.

PyTorch is an existing optional dependency. For a fresh environment, use the
repository's `torch-cu12` or `torch-cu13` extra; no required dependency was added.
The modules and tests run on CPU, and models/buffers can be moved using `.to()`.
Default working tensors and parameters use float32.

The network accepts an explicit, configurable packed feature contract:

| Input | Shape | Meaning |
|---|---|---|
| `local_axes` | `[B, N, 3, 3]` | Current dimensionless axes as matrix columns |
| `state_features` | `[B, N, D]` | Additional normalized inputs; exclude the nine axes already supplied |
| `edge_features[hop]` | `[B, N, S, 24]` | Relative geometry in the receiver's frame |
| `conditioning` | `[B, N, C]` or `[B, C]` | Prepared material/size/timestep scalars |

All batch entries use the same full-cuboid topology and represent separate
objects. `N` is the cell count and the baseline has `S=27`. One possible state
packing is 24 inertial-offset components divided by rest length, six exposed
faces, and eight fixed-corner flags (`D=38`). Optimizer history can add channels.
Normalization and the final physical feature packing remain caller-owned;
log/normalize material parameters rather than feeding raw large stiffnesses.

For already prepared float32 geometry and features:

```python
from experiments.learned_intrinsic_solver.network import IntrinsicSolverNetwork
from experiments.learned_intrinsic_solver.network_geometry import build_edge_features

model = IntrinsicSolverNetwork((10, 10, 40), state_feature_dim=38, conditioning_dim=5)
# rest_centers: [N,3]; centers: [B,N,3]; frames/local_axes: [B,N,3,3].
edges = {
    hop: build_edge_features(
        rest_centers, centers, frames, local_axes, cell_size,
        *model.neighborhood(hop),
    )
    for hop in set(model.hops)
}
result = model(local_axes, state_features, edges, conditioning)
target_axes = result.local_target_axes  # [B,N,3,3], in the input cell frames
```

The geometric helper detaches frame rotations while retaining other geometry
derivatives. Its edge channels are, in order: rest offset / rest edge length
(3), current offset in the receiver frame / rest edge length (3), relative
rotation (9), and neighbor axes in the receiver frame (9). Matrix entries use
row-major flattening, while the axes themselves remain matrix columns.

The network bounds each nine-component correction by
`raw / sqrt(1 + sum(raw**2))`. A pooled object head supplies a shared sigmoid
step in `[0, max_step_size]`, with a configurable default upper bound of 1.
This is a prototype controller, not an energy-descent guarantee. The final
correction head is zero-initialized: initial local targets equal the input
axes. Frame extraction, global reconstruction, physical losses, and time
integration are outside this neural module; unchanged local targets do not
guarantee unchanged reconstructed corners.

## Verify

```bash
uv run --no-sync python -m unittest discover \
  -s experiments/learned_intrinsic_solver/tests -v
```

## Differentiable hex solver step

`solver_step.LearnedHexSolverStep` now connects the baseline network to
shared-corner reconstruction and a physical implicit-Euler objective. Its
working geometry, network, sparse factorization, and backward solves use
**float32**. Torch geometry, network, and energy run on the network's CPU or
CUDA device. The fixed SciPy factorization and forward/adjoint sparse solves
remain on CPU, with differentiable device transfers handled by the custom
fusion function. At least one prescribed corner is required; use the canonical
fixed end for the current experiment.

`hex_energy.HexImplicitEulerLoss` uses **eight-node trilinear hexahedra with
2×2×2 Gauss integration**, not a tetrahedral decomposition. The material is
compressible logarithmic Neo-Hookean, with direct Lamé inputs
`lame_lambda >= 0` and `lame_mu > 0`, both in pascals. Each cell contributes `density * rest_volume / 8`
to each of its eight shared-corner masses. Energy is returned per object in
joules:

```text
Y = previous_positions + dt * previous_velocity + dt² * explicit_acceleration
loss = hex_elastic_energy(X) + 0.5 * sum(mass * squared_length(X - Y)) / dt²
```

Keep `Y` fixed during optimizer iterations within a physical timestep. An
explicit acceleration included in `Y` must not also be counted as a potential.
The optional rigid fusion map does not change `Y`, mass, or this objective.
Contact, damping, acceptance/line search, and physical-rollout training are
not included. Single-update optimizer training is available below.
A nonpositive deformation determinant at any Gauss point raises
an error; the implementation does not clamp or silently repair the candidate.
Positive Gauss samples alone do not prove a cell is valid everywhere.

The integration packs the baseline inputs automatically: local axes (9),
receiver-frame inertial vectors divided by rest size (24), exposed faces (6),
and fixed-corner flags (8). Five FiLM channels are `log1p(lambda/1e5 Pa)`,
`log1p(mu/1e5 Pa)`, `log(density/1000 kg/m³)`, `log(h/0.025 m)`, and
`log(dt/(1/60 s))`. The two material channels remain finite at zero lambda.
The model stores per-cell `lame_lambda`, `lame_mu`, and `density`; the elastic
law uses these Lamé coefficients directly.
Polar frames are extracted using Torch under `no_grad`, recomputed per call.
Optional precomputed frames are detached as well. A Warp implementation remains
future work; other geometry derivatives are preserved.

Fusion uses **increments**. Multiply the predicted local axis change by its
frozen frame to get a world gradient increment. `fusion.HexFusion` finds the
shared-corner displacement whose gradient fits that increment at all eight
Gauss points, with exact prescribed positions. Each cell has weight
`fusion_stiffness * rest_volume`, where
`fusion_stiffness = mu * (3 - mu / (lambda + mu))` preserves the previous
equivalent Young's-modulus weighting. This derived scalar is only for fusion;
it is not a network input or a replacement elastic law. The eight point weights sum to one, so volume
is counted once. Add the solved displacement to the current corners. Zero
network correction therefore preserves even a warped current shape exactly
when its pins are satisfied and no rigid map is applied.

Full quadrature makes the constrained displacement solve unique. It does **not**
make the nine-value cell target able to control every corner-warping mode:
one matrix is still repeated across the eight integration points. This limits
the corrections available to training and can leave an energy floor. The
current shape's unresolved warping is retained by incremental fusion.

The custom Torch backward reuses the cached sparse LU factorization for an
adjoint solve. First-order derivatives are available for cell increments,
base positions, and prescribed positions. Rest geometry, material weights,
and the constraint set are fixed; second derivatives and GPU fusion are not
implemented. Reconstruct the step to change dtype, material, or constraints.

For optional `rigid_delta_rotation=Q` and `rigid_delta_translation=t`, fusion
uses the base `Q @ current_position + t` and carries the learned world axis
increments with `Q`. Prescribed displacements are handled inside the solve;
pins are not overwritten in the base before fitting. In this clamped version
pins determine the final translation, so `t` cancels from the exact fitting
minimizer. A free-body pose convention remains future work.

```python
import numpy as np
import torch
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.multiscale import generate_multiscale
from experiments.learned_intrinsic_solver.hex_energy import make_inertial_prediction
from experiments.learned_intrinsic_solver.solver_step import LearnedHexSolverStep

rest = generate_cuboid((10, 10, 40), cell_size=0.025)
fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
x = torch.tensor(generate_multiscale(rest, seed=0).positions, dtype=torch.float32)[None]
velocity = torch.zeros_like(x)
dt = 1 / 300
Y = make_inertial_prediction(
    x, velocity, dt, explicit_acceleration=torch.tensor([0.0, -9.81, 0.0])
)
step = LearnedHexSolverStep(
    rest, fixed, lame_lambda=288461.53846153844, lame_mu=192307.6923076923,
    density=1000, time_step=dt
)
result = step(x, Y)
result.loss.total.mean().backward()  # physical loss -> fusion -> network
next_candidate = result.positions   # [1, 4961, 3], still float32
```

Run the gradient checks, including a full 4,000-cell forward/backward pass:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python \
  -m experiments.learned_intrinsic_solver.validate_gradients --full-grid
```

The command writes `generated/gradient_validation/report.json`. It reports
central finite differences at three step sizes for energy, fusion targets,
and the complete network-to-energy chain. A separate float64 reference checks
the same float32-quantized inputs and network parameters. Double precision is
used only for reference runs and diagnostic comparisons, not to implement the
working float32 solve. Frame derivatives are deliberately excluded; geometry
finite differences must hold the supplied frames fixed to check this convention.

## Native Newton model and solver

`newton_model.build_newton_hex_model()` now creates a normal `newton.Model`.
Its shared corners are Newton particles, with the same physical lumped hex
masses used by the implicit-Euler loss. Fixed corners retain positive mass;
their ACTIVE flags are cleared and the learned solver enforces them exactly.
Canonical rest geometry, hex connectivity, fixed flags, and per-cell materials
are registered custom model attributes in `model.learned_intrinsic`. This
uses Newton's existing attribute/frequency mechanism and adds no tetrahedra.

`newton_solver.SolverLearnedIntrinsic` derives from the public
`newton.solvers.SolverBase` and implements the normal
`step(state_in, state_out, control, contacts, dt)` interface. Its states and
controls come from the model. This adapter supports one pinned cuboid with a
CPU native Model/State and a CPU or CUDA Torch network in float32. Move the
network to CUDA before constructing the solver to run learned work there.
Native model setup and the frozen rigid predictor remain on CPU; prepared
problem tensors are transferred once to the network device.

The rigid predictor owns a separate scratch Newton Model containing one body.
Each physical timestep it recomputes the mass center, total linear/angular
momentum, and current inertia from the actual corner positions and velocities.
It refreshes both body inertia and inverse inertia, aggregates external corner
forces and their torque, then calls **Newton's public `integrate_bodies()`**
with zero angular damping. The integrator is reused directly. The scratch
body starts at the current mass center with world-aligned axes on each call;
its resulting pose defines the incremental rigid map, not persistent physical
state. Supplied impulses are supported by the standalone predictor and applied
once before integration; contact detection/response is not supplied here.

The physical predictor `Y` is independently built once from the original
corner positions, velocities, external corner forces, and model gravity.
Corner forces in `state_in.particle_f` must exclude gravity. Rigid-guided fusion
initializes the candidate once, before the first network evaluation. All
learned iterations then refine the latest candidate, using the same `Y` and
prescribed positions. After the final update,
Newton `state_out` receives the proposed positions and the single-step velocity
`(final_position - previous_position) / dt`, with stationary pins at zero
velocity. The input state is not modified. Pins override incompatible rigid
motion; the predictor itself does not compute their reaction forces.

```python
import numpy as np
from experiments.learned_intrinsic_solver.data import generate_cuboid
from experiments.learned_intrinsic_solver.multiscale import generate_multiscale
from experiments.learned_intrinsic_solver.newton_model import build_newton_hex_model
from experiments.learned_intrinsic_solver.newton_solver import SolverLearnedIntrinsic

rest = generate_cuboid((10, 10, 40), cell_size=0.025)
fixed = np.flatnonzero(rest.corner_rest_positions[:, 2] == 0)
model = build_newton_hex_model(
    rest, fixed, lame_lambda=288461.53846153844, lame_mu=192307.6923076923,
    density=1000, gravity=(0, -9.81, 0),
)
state_in, state_out = model.state(), model.state()
state_in.particle_q.assign(generate_multiscale(rest, seed=0).positions.astype(np.float32))
state_in.clear_forces()  # add external corner forces here, excluding gravity
solver = SolverLearnedIntrinsic(model, iterations=5)
solver.step(state_in, state_out, model.control(), None, 1 / 300)
solver.last_result.loss.total.mean().backward()
# optimizer operates on solver.network.parameters()
```

For training without writing a Newton output state, use
`result = solver.predict(state_in, dt)`. Both entry points retain the Torch
network/fusion graph in `last_result`. The input Warp arrays are copied before
Torch evaluation, so future state writes cannot overwrite saved inputs.
Committing to Newton states and computing the rigid predictor are detached
boundaries; differentiating whole physical trajectories through Warp State
storage or the rigid predictor remains future work. Frozen frame extraction
still gives the previously chosen approximate gradient across inner updates.

The optimizer interface also separates the **fixed physical problem** from
the **current candidate**. This allows querying a direction at any feasible
candidate, including candidates that did not come from a previous network
update:

```python
problem = solver.prepare_problem(state_in, 1 / 300)
current = solver.initialize_candidate(problem)  # rigid-guided fusion, once
update = solver.propose_update(current, problem)
direction = update.direction  # [1,4961,3], fused world displacement in meters
current = update.positions   # current + direction

# Or unroll five updates from a supplied feasible candidate:
result = solver.solve(problem, initial_positions=current, iterations=5)
result.loss.total.mean().backward()
```

`prepare_problem()` snapshots the original corners, inertial prediction,
stationary pins, and rigid target, and captures the material/dt/fusion context.
Preparing another timestep does not change an existing problem's objective.
`propose_update()` recomputes current axes, polar frames, inertial offsets,
and neighborhood features on every call. It returns both the local axis
proposal and the actual fused corner displacement. Supplied candidates must
be finite float32 `[1,P,3]` tensors on the network device with prescribed corners satisfied
exactly; the hex energy rejects inverted elements. A query does not integrate
the rigid pose again. `solve()` defaults to five updates, shares network
weights across them, and never detaches intermediate candidates. Each update
retains its input, output, frozen frame, local target, and physical loss.

This interface proposes directions; the initial untrained network cannot
guarantee descent. Its default zero correction head produces exact zero
directions, while a randomly perturbed head can increase energy. A negative
physical `gradient dot direction` means local descent; energy decrease after
the finite update is a separate check. There is currently no line search or
fallback direction. Frozen-frame derivative checks replay the same recorded
frame sequence in both perturbed unrolls, consistent with the chosen backward
convention.

Run the complete inference/backward probe with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python \
  -m experiments.learned_intrinsic_solver.optimizer_probe
```

It uses the 10×10×40 float32 grid, five learned updates, three reproducible
deformations, and both the default zero head and a small diagnostic nonzero
head. It writes energy changes, physical directional derivatives, pin errors,
Gauss determinants, fusion residuals, and gradients through every update to
`generated/optimizer_probe/`. The nonzero head exercises parameter paths that
the zero initialization initially blocks; it is not a trained checkpoint.

Model gravity is read each timestep. Changing `dt` refreshes the learned
step's physical/conditioning constants while retaining the same network.
After changing registered materials, masses, or boundary flags, call
`solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)` to rebuild
cached data. Native masses remain authoritative and must agree with hex
density/volume. Material and topology are not learned through this adapter.

The network remains untrained unless a checkpoint is supplied. These are
learned proposals with physical losses, not a converged implicit solve.
Populated contact buffers are rejected explicitly. The earlier CPU fusion,
pinned-body, and per-cell representation limits still apply.

## Single-update GPU training smoke run

The current design and the planned larger training campaign are recorded in
[v1-DECISIONS.MD](../../notes/v1-DECISIONS.MD#22-current-design-and-research-checkpoint).
The epoch-based, multi-GPU campaign is not implemented in this smoke trainer.

`train_smoke.py` performs actual Adam weight updates using the deterministic
multiscale augmenter, direct Lamé materials, and the existing implicit-Euler
loss. It starts with one learned solver iteration per query. The default is
64 weight updates on a 10×10×40 grid, 16 fixed physical training seeds, and
8 disjoint validation seeds. Training candidates vary; validation candidates
stay fixed so the curve compares the same physical objectives throughout.

The candidate mixture is 50% pin-corrected inertial prediction, 35% that
prediction plus multiscale noise, 10% previous physical positions, and 5%
rigid-guided initialization. Original Y never changes when a candidate is
perturbed. Final float32 candidates are screened at Gauss points and cell
centers. Initialization may reduce noise or use the previous valid shape;
an invalid learned output is reported as a failure rather than repaired.

Training minimizes `(E_after - E_before.detach()) / max(E_before.detach(), 1 J)`.
The subtracted energy and denominator are fixed per query, so its gradient
still minimizes the physical energy. Both normalized losses and physical
energies are recorded. The initial zero head yields zero energy change;
negative normalized loss means the learned update reduced energy. A small
configured `max_step_size=0.05` limits early untrained proposals.

```bash
source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh learned-intrinsic-training occupy
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python -u \
  -m experiments.learned_intrinsic_solver.train_smoke \
  --device cuda --cpu-threads 2 --updates 64 \
  --output generated/training/my_new_run
```

Choose a fresh output directory for each new run. The current trainer can
overwrite an existing directory; use `--resume` only for deliberate continuation.

The CUDA path disables TF32 and autocast. The sparse factorization is fixed;
its factorization procedure is not differentiated. A custom adjoint solve
propagates gradients through the fusion right-hand side, base, and prescribed
positions and returns them to the GPU. Transfers synchronize around the CPU
solve; this bridge is not a CUDA-graph implementation.

The output directory contains per-update `training.csv`, fixed-query
`validation.csv`, `loss_curve.png`/`.svg`, a browser report, and checkpoints.
Checkpoints include initial, periodic, best-validation, and final weights,
Adam state, physical samples, sampler state, fixed validation candidates,
configuration, and CPU/selected-device CUDA RNG state. Resume with the same
configuration and `--resume <checkpoint.pt>`; `--updates` specifies the desired
total completed weight updates. CPU is supported for small tests. This is a
bounded training experiment, not evidence of convergence on arbitrary states.

## Four-GPU numerical diagnostic

`distributed_probe.py` tests the full learned step under PyTorch DistributedDataParallel:
one process per GPU, batch 16 per process, 64 distinct physical queries, and three
Adam updates. The fixed queries repeat across updates to allow exact checkpoint
replay and comparison with a serial reference. This is a numerical diagnostic;
it does not implement the planned epoch trainer or establish learned convergence.

The launcher claims each GPU exclusively through the workspace GPU-claim script,
assigns rank metadata, and captures per-rank logs. Each process sees its own GPU
as `cuda:0`; no physical GPU index is hardcoded. The launcher stops the whole
worker group on failure or timeout and rejects existing output directories.

On this VM, default NCCL peer-to-peer transport stalled during startup collectives.
The same four-GPU test completed with `NCCL_P2P_DISABLE=1`; NCCL selected shared
host memory transport. Apply this setting to the invocation, not globally:

```bash
NCCL_P2P_DISABLE=1 uv run --no-sync python -u \
  -m experiments.learned_intrinsic_solver.launch_distributed_probe \
  --workers 4 --output generated/distributed_probe/my_new_run \
  --batch-size 16 --updates 3 --cells 10 10 40
```

The complete `LearnedHexSolverStep` is wrapped in DDP, including feature construction,
fusion, and physical loss evaluation. Equal per-rank mean losses produce the global
batch-mean gradient. Network, frames, energy, and Adam run on CUDA in float32 with
TF32 and AMP disabled. Each process keeps its own fixed CPU SciPy fusion factor.

The probe checks prescribed corners, finite energies and gradients, byte-identical
replica weights after every update, and exact continuation from a serialized
first-update checkpoint. It stores the actual input tensors, original Y, frozen
first-update frames, gradients, weights, Adam state, and hardware identifiers.
Run the independent reference on one exclusively claimed GPU:

```bash
source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh learned-intrinsic-ddp-reference occupy
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync python -u \
  -m experiments.learned_intrinsic_solver.distributed_reference \
  --probe generated/distributed_probe/my_new_run
```

The reference sequentially processes the four saved batches, accumulates their
equally weighted gradients without distributed Torch, and compares the first
gradient and Adam update. It also compares later updates and final checkpoint
inference, and records serial timing. It requires the distributed run and its
checkpoint replay to have passed before reporting numerical success.

To exercise coordinated failure handling, use a fresh output with
`--cells 2 2 3 --batch-size 2 --fail-rank 2 --fail-update 2`. The expected result
is a nonzero exit, all four reports naming the same failure before backward,
only one completed Adam update, and all GPU claims released. This deliberately
failed run must not be used as a passing numerical reference.

### Larger epoch training

The epoch trainer starts with fresh seeded network weights. Each epoch visits
all 8,192 training physical states once and draws a new deterministic candidate
for each visit. The 512 validation candidates remain fixed. Training uses one
learned optimizer iteration, float32 without TF32/AMP, and the unchanged original
implicit Euler objective. With four GPUs and batch 16 per GPU, there are 128
Adam updates per epoch.

```bash
NCCL_P2P_DISABLE=1 uv run --no-sync python -m \
  experiments.learned_intrinsic_solver.launch_training \
  --output generated/training/large_001 --workers 4 --timeout 86400
```

The launcher obtains an exclusive claim for each GPU and supervises all ranks.
The local VM needs `NCCL_P2P_DISABLE=1`; the launcher defaults to this setting
unless the caller explicitly supplies another value. Logs are `logs/rank_N.log`.
Immutable CPU dataset snapshots are `data/rank_N.pt`. Reports are `index.html`,
`report.json`, `epochs.csv`, and `loss_curve.svg`/`.png`. Shared model/Adam
checkpoints live in `checkpoints/`: initial, latest each epoch, best validation,
every ten epochs, final, and diagnostic failure. Rank identities and RNG states
are recorded in each checkpoint; dataset snapshots are not repeated in them.

Resume from the same run's completed epoch checkpoint, preserving world size,
batch size, physics, and sampling settings. The epoch cap, early-stopping flag, and verbosity may
change. Existing logs are retained in numbered directories:

```bash
NCCL_P2P_DISABLE=1 uv run --no-sync python -m \
  experiments.learned_intrinsic_solver.launch_training \
  --output generated/training/large_001 --workers 4 \
  --resume generated/training/large_001/checkpoints/latest.pt --max-epochs 200
```

The initial learning rate is 1e-4. It halves after five validations without an
absolute normalized-loss improvement of at least 1e-4, down to 1e-6. Plateau
stopping requires at least 30 epochs, 15 non-improving validations, and two rate
reductions. The last five validations must each have no invalid outputs, at
least 95% descent, and lower raw mean energy to label this `plateau_converged`;
otherwise it is `stalled`. The 200-epoch cap is a separate `epoch_limit` status.
Invalid learned training outputs stop the campaign and preserve actual failing
inputs; they are never repaired. Validation failures count against all 512 cases.

A bounded full-grid epoch/resume check uses `--train-count 64 --validation-count
16 --batch-size 16 --min-epochs 1 --max-epochs 2`. These reduced-count runs verify
execution and checkpoint replay, not convergence.

The active larger campaign was subsequently extended to **500 total epochs**
with plateau early stopping disabled. Resume this policy using:

```bash
NCCL_P2P_DISABLE=1 uv run --no-sync python -m \
  experiments.learned_intrinsic_solver.launch_training \
  --output generated/training/large_001 --workers 4 --timeout 259200 \
  --resume generated/training/large_001/checkpoints/latest.pt \
  --max-epochs 500 --no-early-stopping
```

This preserves the existing physical dataset and training state. Learning-rate
reductions, validation, checkpoint saving, and invalid-output failure handling
remain active. The early-stopping flag may change on resume along with the
epoch cap and verbosity. The longer supervisor timeout accommodates this run.
