# ALM elasticity and contact — takeover note

Prepared for Anka on 24 September 2026; finalized on 25 September. This note distinguishes implemented code, experimental results, and the latest discussion decisions. The linked reports and source snapshots are included in this package, so they can be read on another machine.

**Current state:** stage 1 elasticity ALM is implemented and opt-in. SVD/stretch elasticity has derivations and standalone probes, but no solver integration. Stage 2 ALM contact and the proposed final iterative truncation are not implemented.

[Download this note as Markdown](handoff.md) · [Download the portable package](alm-handoff-20260924.zip) · [Download the branch changes](branch-changes.patch)

## Start with these links

| Resource | What it contains |
| --- | --- |
| [Elasticity code walkthrough](reports/alm-stage1-code-walkthrough-246fb405-20260914/index.html) | Solver entry points, state, forces, dual updates, reset, capture, and tests. Best first read for an implementer. |
| [Stage 1 implementation notes](source/notes/stage1_elasticity_alm.md.html) | How to enable the experiment, lifecycle, measurements, and limitations. |
| [Elasticity derivation](reports/alm-elasticity-derivation-20260914/index.html) | Original Neo-Hookean energy, the quadratic-volume model in the code, 9 + 1 constraints, hydrostatic ALM, and the proposed stretch formulation. Equations are rendered. |
| [SVD exploration](reports/alm-svd-exploration-20260914/index.html) | Rotation and stretch-crossing experiments, constitutive derivatives, Warp probes, and downloadable results. |
| [Contact design and softmax proof](reports/alm-contact-design-20260914/index.html) | Earlier contact proposal, cushion law, friction, and full-primitive separation proof. **Some design choices are superseded; read the corrections below.** |
| [Original formula document](source/formulas/alm_formulas.md.html) | The original brainstorming input; it does not override later decisions or corrected signs. |

## Workspace and provenance

- Worktree: `/home/horde/Code/Graphics/newton-working-copies/alm-dat-design`.
- Branch: `ankac/alm-dat-design`; implementation/documentation snapshot: `eba5e1a5`.
- Starting refactor snapshot: `0d101f20326d65ea28f8a3d0345ae6c5829898df`.
- Main implementation commit: `246fb405` — experimental particle elasticity ALM.
- Follow-up commits: `3566d7d0` (walkthrough), `90a74d0d` (SVD probes), `bf80c5b8` / `a91cbfea` / `7e6cf67d` (contact documentation), `23600ad6` (elasticity derivation), `eba5e1a5` (softmax proof).

The downloadable patch contains the committed changes from the starting refactor snapshot through `eba5e1a5`. It requires that baseline; the selected source snapshots in this package are for inspection, not a standalone Newton installation. This handoff is additional documentation and is not part of that patch.

Keep changes in the assigned worktree. Treat the supplied `trimesh-csr-refactor` worktree as read-only. On 24 September it was at `fac9a033`, with untracked `notes/`; its newer changes have not been incorporated into this branch. Do not access the terminated agent's `alm-dat-implementation` worktree or its prompts/notes. Preserve unrelated untracked `notes/review-probes/`, `notes/task-prompt.md`, and `notes/warp-cache/`. Do not post or push to GitHub.

Follow [AGENTS.md](source/AGENTS.md.html), the [coding guidelines](source/CODING_GUIDELINES.rst.html), and the newton-workspace skill. Use `uv`; claim a GPU through the workspace script before GPU execution.

## What is implemented

```python
solver = newton.solvers.SolverVBD(
    model,
    particle_elasticity_alm=True,
)
```

Ordinary `SolverVBD` still defaults to ALM disabled. Enabling it adds tetrahedral pressure ALM, spring stretch ALM, and dihedral bending ALM. For tetrahedra, the default ALM mode leaves the mu term on its existing formulation. Setting `particle_elasticity_alm_deviatoric=True` also enables matrix-F history for the mu term; this mode has a known rotation artifact. `particle_elasticity_alm_rho_scale` defaults to 1.0.

Triangle membrane elasticity, damping, contact forces, collision scheduling, and the existing predictor/per-color DAT are unchanged. **The implemented elasticity solver does not yet use the contact/truncation pipeline discussed later.**

| Source | Responsibility |
| --- | --- |
| [solver_vbd.py](source/newton/_src/solvers/vbd/solver_vbd.py.html#L336) | Flags, history allocation, preparation, scalar/tile dispatch, reset, and once-per-sweep dual updates. |
| [particle_alm_kernels.py](source/newton/_src/solvers/vbd/particle_alm_kernels.py.html#L21) | Persistent element state, metrics, initialization, multiplier updates, and selective reset. |
| [particle_vbd_kernels.py](source/newton/_src/solvers/vbd/particle_vbd_kernels.py.html#L173) | ALM tet, spring, and hinge force/Hessian evaluation and solver integration. |
| [rigid_vbd_kernels.py](source/newton/_src/solvers/vbd/rigid_vbd_kernels.py.html) | Existing compliant-ALM coefficient helpers reused by the particle implementation. |
| [Element tests](source/newton/tests/test_particle_alm_kernels.py.html) / [Solver tests](source/newton/tests/test_solver_vbd_alm.py.html) | Constitutive/lifecycle checks and integrated solver behavior, including CUDA capture/replay tests. |

Each step prepares inertia-based penalty metrics and seeds pending histories from the incoming pose. Each complete particle color sweep, including the existing DAT, is followed by one dual update. Histories persist across steps. Reset invalidates selected histories for reseeding; topology/material changes require reconstructing the solver. Repeated-interval proxy coupling is explicitly unsupported.

For a compliant row with material stiffness K, constraint C, multiplier ell, and penalty rho, the implemented algebra is:

```text
s = K / (K + rho)
k_eff = K * rho / (K + rho)
transmitted_stress = k_eff * C + s * ell
ell_next = transmitted_stress
```

The implementation uses numerically safer equivalent expressions. The material stiffness at a converged fixed point is K; this is not a hard constraint driving every strain to zero.

## Elasticity findings and the SVD direction

The code's tet energy uses a quadratic determinant term. With `Kp = lambda_L + mu`, its pressure constraint is `Cp = (J - 1) - mu/Kp`. Kp is the coefficient in this energy, not the physical bulk modulus. The [hydrostatic derivation](reports/alm-elasticity-derivation-20260914/index.html#hydrostatic-term-the-one-scalar-volume-constraint) explains the offset and force. Classical logarithmic Neo-Hookean and this quadratic-volume model are not identical at finite volume changes.

The matrix-F mode stores spatial stress history. At finite iterations, a rigid rotation can make that retained history produce spurious forces. The recorded rotated-rest probe moved a vertex by about 0.110 m in full matrix mode, versus 3.94e-8 m in pressure-only mode. This motivated the default; it is not a claim that pressure-only solves every material convergence problem.

The proposed replacement for the mu history is a symmetric material stretch tensor:

```text
S = sqrt(F^T F + epsilon^2 I)
Lambda_next = k_eff * S + s * Lambda

Solve S X + X S = Lambda
P_mu = k_eff * F + 2 * s * F X
```

This has six independent history components, plus the existing scalar pressure history. SVD can evaluate S and the small Sylvester solve. Simply storing three multipliers by sorted singular-value index can introduce force jumps when stretches cross; the tensor proposal avoids that indexing ambiguity. Its mu energy matches the original energy up to an additive constant. Force consistency at the ALM fixed point does not make the frozen-history Hessian identical to the original material Hessian.

Standalone probes checked force/tangent derivatives, rotational objectivity, stretch crossings, near-collapse behavior, and Warp SVD/capture primitives. Epsilon regularization introduces a stiffness tradeoff near collapse, and the frozen-history Hessian can be indefinite. **No SVD/stretch tet path is integrated into SolverVBD.**

## Latest contact decisions

These decisions supersede conflicting passages in the older contact report:

1. Perform an initial spherical candidate query for each attempted time step, with conservative motion budgets. Swept AABB detection is deferred.
2. Build constraints from a common feasible starting pose. Intermediate ALM positions may penetrate; initially interpenetrating geometry is not automatically supported.
3. Keep contact normals fixed throughout that attempted step. The earlier proposal to update normals every sweep was set aside, partly because changing normals also changes friction.
4. Keep the soft cushion plus hard nonpenetration boundary.
5. Use one contact constraint and the same normal for forces and truncation. A separate force normal and safety normal was explicitly rejected in discussion.
6. After the ALM solve, run an **iterative final DAT truncation stage**. Each vertex has its own fraction `t_i`; there is no scene-wide minimum fraction shared by all vertices. Preserve the original candidate displacement while iterating.
7. The final truncation may update plane offsets/available motion space while n remains fixed. It must preserve feasibility at every accepted refinement. No additional coloring is budgeted; the pipeline must support CUDA capture.
8. If the method performs poorly, reduce the time step. Retrying requires restoring the step-start state and histories and rebuilding candidates; this retry mechanism is not implemented.

The latest constraint discussion used the conservative, **unnormalized** softmax supports:

```text
U =  log(sum_i exp(beta * dot(n, x_Ai))) / beta
L = -log(sum_j exp(-beta * dot(n, x_Bj))) / beta
C = L - U - required_gap
```

With beta > 0, a unit n, nonnegative required_gap, and all vertices of each convex primitive included, nonnegative C certifies separation of every point of the two primitives along n. Vertex positions are used, not displacements. A barycentric witness gap alone does not provide this guarantee. Normalized softmax sums need a conservativeness correction. Initial geometric separation alone also does not guarantee that the softmax C is feasible; its additional margin must be accommodated.

For fixed n and beta and affine vertex trajectories, C is concave in the vertex fractions. Its feasible set is convex, but **the softmax-constrained problem is not a QP**. The quadratic benchmark becomes a QP only when using hard per-vertex support inequalities. At the same required gap, the hard-support QP is a relaxation of the softmax feasible set; those two feasible sets must not be silently interchanged.

## What remains unresolved in contact

**The final iterative truncation algorithm is not settled or implemented.** The names `RefineDATBounds` and `RefineVertexFraction` in earlier chat were pseudocode placeholders, not Newton APIs. In the proposed monotone sufficient-bound scheme, `RefineVertexFraction` was only the per-vertex minimum over incident bounds; it is ordinary clipping and needs no separate optimizer. Arbitrary moving-plane updates can also impose lower fraction bounds. Minimum-only clipping requires a construction that preserves the current fractions as a feasible witness.

An assistant-proposed local refinement allocated each pair's current softmax gap to additional projected motion on its two sides, optionally through a moving plane offset. This supplies a conservative sufficient bound, but it can stall and has no global-optimality guarantee. It was not established as Anka's accepted final algorithm. Do not turn those placeholder names or that proposal into an assumed implementation specification.

The agreed next research tool was a small coupled-contact benchmark with a clearly stated objective, for example minimizing mass-weighted discarded displacement. Compare the local truncation against a fixed-normal hard-support QP and, separately, the actual softmax-constrained convex problem. Monotone improvement of fractions alone does not imply optimality.

Other open items:

- Contact multiplier initialization, matching, reset, and safe persistence across candidate-buffer changes.
- Consistency between normal multipliers learned from penetrating trial poses and motion removed by final truncation. Do not claim standard ALM convergence for that combination without analysis or measurements.
- Friction history: distinguish temporary solver iterates from committed accumulated slip; discarded trial motion must not be recorded as actual physical sliding. Fixed n removes one source of history changes but does not settle this issue.
- Complete candidate coverage under the initial-query budgets, including overflow handling. A smaller time step alone is not a coverage proof.
- Rigid-body rotation trajectories: affine-vertex bounds do not automatically certify rigid rotation paths.
- Softmax force torque with a frozen n. Normal optimization was explored mathematically, but is outside the current fixed-normal baseline.

## How the CSR collision refactor helps

The inherited [tri_mesh_collision.py](source/newton/_src/geometry/tri_mesh_collision.py.html#L41) provides shared vertex-triangle and edge-edge pair arrays, per-source CSR rows, device counts/overflow flags, and capture-compatible CSR rebuilding. Match persistent contacts by primitive IDs within those rows; append slots depend on scheduling and are not stable identities.

ALM self-contact still needs solver-owned history and correspondence across detections. EE output includes both directions, while the existing force evaluation updates the source edge of each record. A canonical physical EE contact pool would need shared history and force contributions to all four endpoints. Preserve the feasible step-start geometry separately from detector references that can change during redetection.

## Evidence, reproduction, and next work

The [recorded benchmark](source/notes/stage1_elasticity_alm_results.json.html) and [benchmark script](source/notes/stage1_elasticity_alm_benchmark.py.html) cover a small 64-vertex, 135-tet test on an NVIDIA L40 with Warp 1.17. At high volumetric stiffness, one comparison gave force residual 0.194 for pressure ALM versus 0.311 for legacy at roughly 0.39 ms. At a longer budget, legacy was better: 0.0387 versus 0.0462 at roughly 0.9 ms. These are historical microbenchmark measurements, not a general speedup claim.

Tests exist for loaded tet/spring equilibrium, rotated rest, reset, scalar/tile parity, capture/replay, inactive/high-stiffness rows, hinge geometry, damping preservation, and integration with legacy DAT. **Solver tests and GPU benchmarks were not rerun while preparing this handoff.** Source locations, packaged links, and report rendering were checked separately.

From the worktree, after claiming a GPU when needed:

```bash
uv run --extra dev -m newton.tests -k test_particle_alm_kernels
uv run --extra dev -m newton.tests -k test_solver_vbd_alm
uv run --no-sync python notes/stage1_elasticity_alm_benchmark.py --iterations 1 5 10 13 23 30 100
uv run --no-sync python notes/svd-exploration/numpy_probe.py
uv run --no-sync python notes/svd-exploration/warp_probe.py --device cpu
```

Some probes write result files in `notes/`; preserve the historical results before rerunning them. GPU execution must use the workspace GPU-claim script. Run `uvx pre-commit run -a` before committing changes.

For elasticity work, start with the walkthrough, then integrate the tensor-stretch proposal only after checking its derivatives and cost in both scalar and tile paths. For contact work, first make the iterative truncation update concrete and benchmark it; do not assume the earlier chat's incomplete pseudocode is a finished design. Anka requested this handoff and pseudocode in English. Use rendered equations rather than raw LaTeX when communicating with Anka.
