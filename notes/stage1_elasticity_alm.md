# Stage 1: elasticity ALM experiment

The stage 1 implementation adds opt-in compliant ALM for tet pressure, optional
tet matrix stress, spring stretch, and dihedral bending. The initial ALM default
uses tet pressure only, based on the rotation measurement below. Ordinary
`SolverVBD` still defaults to the existing elasticity formulation.

```python
solver = newton.solvers.SolverVBD(model, particle_elasticity_alm=True)

# Explicit full matrix comparison; carries the rotation limitation below.
full = newton.solvers.SolverVBD(
    model,
    particle_elasticity_alm=True,
    particle_elasticity_alm_deviatoric=True,
)
```

`particle_elasticity_alm_rho_scale=1.0` scales the numerical metric without changing
the converged material law. These are experimental, runtime solver settings;
they do not add material authoring fields or a USD schema.

## Implemented flow

1. Construct fixed-size element histories and the scalar/tile specialization.
2. Before prediction, seed pending histories from the incoming pose and compute
   inertia-based metrics for the current timestep. Metrics stay fixed within the
   step; material stresses carry between steps with retention one.
3. Run the existing predictor, contact evaluations, and per-color DAT. Vertex
   solves use effective elastic stiffness and transmitted ALM stress. Triangle
   membrane elasticity and damping retain their existing formulas.
4. After the entire particle color sweep and DAT, update each element dual once.
5. Finalize velocities as before.

For a scalar row with material stiffness `K`, constraint `C`, stress `lambda`, and
numerical metric `rho`, the shared algebra is:

```text
s = K / (K + rho)
k_eff = K*rho / (K + rho)
transmitted_stress = k_eff*C + s*lambda
lambda_next = k_eff*C + s*lambda
```

The implementation evaluates these expressions in forms that avoid overflowing
`K*rho` or `rho*C`. Tet pressure uses `K = authored_lambda + mu`; its strain is
computed as `(det(F)-1) - mu/K` to retain the resting stress offset at large `K`.
Tet stresses are multiplied by rest volume once when assembling forces and
Hessians. Springs retain their geometric curvature, including compression.

## Data and lifecycle

| Family | Persistent state per element |
| --- | --- |
| Tet pressure | Scalar stress, scalar metric, pending initialization bits |
| Optional tet matrix | Additional 3x3 stress and scalar metric |
| Spring | Scalar force, scalar metric, pending flag |
| Hinge | Scalar moment, scalar metric, pending flag |

History storage is 12 bytes per pressure-only tet, 52 bytes per full tet, and
12 bytes per spring or hinge, excluding array metadata. Disabled mode allocates
no element history. Existing element adjacency and vertex scratch are reused.
Prepare, dual-update, and reset passes use fixed buffer addresses and launch
sizes; CUDA graph replay reads the current device reset mask.

`solver.reset(state, world_mask=mask, flags=0)` preserves user-authored positions
and invalidates selected histories. Their next preparation uses any pose edits
made after reset. If an element spans worlds, resetting any incident vertex's
world invalidates the shared element. Fixed and degenerate rows retire and can
reseed when active again.

Reconstruct the solver after changing topology or materials. Repeated-interval
proxy coupling is explicitly rejected because this experiment does not restore
element histories for a repeated solve of the same interval.

## Measurements

The reproducible diagnostic is [stage1_elasticity_alm_benchmark.py](stage1_elasticity_alm_benchmark.py),
with raw output in [stage1_elasticity_alm_results.json](stage1_elasticity_alm_results.json).
Run from this worktree after claiming a GPU:

```bash
WARP_CACHE_PATH="$PWD/.cache/warp-stage1" uv run --no-sync python notes/stage1_elasticity_alm_benchmark.py --iterations 1 5 10 13 23 30 100
```

Setup: NVIDIA L40, Warp 1.17.0, 64 vertices / 135 tets, 0.6 m cube, density
1000 kg/m^3, bottom vertices fixed, downward force of 10 N per top vertex,
`dt=1/30 s`, `mu=10,000 Pa`. No contact or damping. GPU event timings are medians
of five batches of 25 captured resets plus cold implicit steps. This is a small
solver microbenchmark; it is not a production-scene speedup claim.

The residual below evaluates the original material force in float64, including
inertia and applied load, then divides its norm by the applied-force norm. It
does not evaluate only the temporary ALM force. At nearly equal GPU time with
`authored_lambda=1,000,000 Pa`:

| Mode | Iterations | Reset + step | Relative force residual |
| --- | ---: | ---: | ---: |
| Existing elasticity | 13 | 0.397 ms | 0.311 |
| Pressure ALM | 10 | 0.390 ms | 0.194 |
| Existing elasticity | 30 | 0.902 ms | 0.0387 |
| Pressure ALM | 23 | 0.876 ms | 0.0462 |

Pressure ALM helps the shorter solve here; the longer solve favors the existing
method by force residual. At moderate `authored_lambda=10,000 Pa`, the existing
method already converges rapidly and ALM adds overhead. Position errors in the
JSON use a 1000-iteration legacy reference, whose residual is also reported;
that reference is not exact ground truth.

The rotation probe prepares a unit tet, rotates the incoming resting shape by
0.2 rad without clearing history, and performs one iteration at `dt=1/60 s`,
with `mu=10,000 Pa` and `authored_lambda=10,000 Pa`. Pressure-only mode moves a
vertex by at most `3.94e-8 m`; full matrix mode moves one by `0.110 m`. Reseeding
the full mode reduces this to `1.26e-7 m`, but reseeding is not a solution for
ordinary continuous motion. This is why pressure-only is the ALM default.
Full matrix ALM needs an objective history treatment before relying on it in
rotating scenes. Separately, both ALM modes remain motionless in the tested
1000-step axis-aligned rest experiment.

## Validation and next stage

Tests cover loaded tet/spring equilibrium, rotated rest, default rotation
behavior, scalar/tile parity, capture/replay with reset, selected-world history,
inactive and high-stiffness rows, hinge geometry, preserved damping, and legacy
DAT with an ALM tet. Independent finite-difference probes checked the local tet
and spring Hessians at fixed history.

Stage 2 contact ALM is not implemented here. The current contact law, collision
schedule, and predictor/per-color truncation remain in use. The contact design
still calls for an initial distance query with conservative motion bounds;
swept AABB candidates remain a future direction.
