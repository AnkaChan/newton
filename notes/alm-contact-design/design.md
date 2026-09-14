# ALM contact with final-only truncation

Discussion draft for the VBD group meeting · 14 September 2026

**Status:** This is the proposed contact stage, not an implemented solver. The
current stage-1 code implements optional elasticity ALM and retains the existing
contact and per-color DAT behavior. This document records the contact decisions
from our discussion, with remaining choices identified explicitly.

## 1. Main idea

Build oriented contact constraints from a known separated configuration. Let
ALM/VBD iterates penetrate while their multipliers accumulate separating force.
After the solve, conservatively truncate the proposed motion before committing
the next state.

```text
Accepted separated pose
  -> initial spherical queries + motion budgets
  -> match contact history and build separation certificates
  -> ALM/VBD sweeps; internal poses may penetrate
  -> final displacement caps
  -> accepted positions, velocities and retained history
```

The force constraint and the final safety certificate have different jobs.
The first drives the solve; the second bounds the entire primitive motion.
Neither finite-iteration ALM convergence nor arbitrary initial interpenetration
is assumed to be solved automatically.

## 2. Contact constraint and barycentric witnesses

For each supported primitive pair, let the normal point from side A to side B:

$$
g(\mathbf{x})=\mathbf{n}^{\mathsf{T}}\left(\sum_{j\in B}\beta_j\mathbf{x}_j-\sum_{i\in A}\alpha_i\mathbf{x}_i\right)-d_{\mathrm{hard}}\geq 0
$$

The weights are nonnegative barycentric coordinates and sum to one on each side.
A vertex has weight one;
an edge has two weights; a triangle has three. `hard_gap` includes the desired
thickness offset. Positive `g` means separation along the chosen orientation;
negative `g` means a violation of this witness constraint.

**Agreed evaluation policy:** recompute barycentrics before contact evaluation,
then treat them as fixed during the local solve. Hold the selected normal fixed
within that solve as well. Its refresh schedule is a separate choice.

For compressive normal force magnitude `p >= 0`:

```text
force_Ai = -p * alpha_i * n
force_Bj = +p * beta_j  * n
```

The two sides receive equal and opposite total forces. Fixed weights simplify
the local derivatives; they do not guarantee smoothness across witness-feature
changes, nor does a frozen normal guarantee angular-momentum consistency when
the current witnesses are no longer aligned with it. These are evaluation
targets for the normal-refresh policy.

**A witness gap is not a full-primitive certificate.** The gap along a normal
that does certify separation of the two convex primitives is:

```text
g_support = min_j(dot(n, x_Bj)) - max_i(dot(n, x_Ai)) - hard_gap
```

A positive barycentric gap can coexist with another primitive vertex crossing
the plane. The final safety pass must therefore constrain all primitive vertices.

## 3. Soft cushion plus hard boundary

Keep the intended pre-contact cushion and a nonpenetration boundary. Introduce a
slack gap `z` with:

```text
psi(z) = 0.5 * K * max(activation_gap - z, 0)^2

enforce:  g = z
          z >= 0
```

At equilibrium, the cushion supplies `K * (activation_gap - g)` for
`0 < g < activation_gap`; it releases outside the activation range. At `g = 0`,
the hard-boundary reaction can exceed `K * activation_gap`. It is not capped at
the cushion force.

ALM allows the temporary signed witness gap `g` to be negative while `z` remains
nonnegative. With positive compressive multiplier `p` and metric `rho > 0`, use
the augmented term `p*(z-g) + 0.5*rho*(z-g)^2`. The local slack solve is explicit:

```text
y = g - p / rho

if y >= activation_gap:
    z = y
else:
    z = max(0, (rho*y + K*activation_gap) / (rho + K))

p_eval = p + rho*(z - g)
```

This yields three regimes:

| Regime | Transmitted normal force | Local gap stiffness |
| --- | --- | --- |
| Released | `p_eval = 0` | `0` |
| Cushion | `p_eval = s*p + k_eff*(activation_gap-g)` | `k_eff` |
| Hard boundary active | `p_eval = p-rho*g` | `rho` |

```text
s = K / (K + rho)
k_eff = K*rho / (K + rho)
```

In hard mode, penetration increases the separating force because `g < 0`.
Use the raw signed gap so separation can also release unsupported history.
The formulas above define the sign convention; they correct the slack-sign
ambiguity in the original formula reference.

Within a color solve, read persistent `p` and evaluate `p_eval` from the current
local geometry. After a complete color sweep, evaluate the same update at the
resulting pose and write `p <- p_eval` once per contact. Do not update an element's
persistent multiplier separately for every visited endpoint.

With frozen barycentrics and normal, the particle Hessian contribution is the
regime's gap stiffness times `weight^2 * n * transpose(n)`. A rigid-body path must
also account for its pose parameterization. The proposed slack solves remain
local; no global linear system is introduced.

## 4. Updating normals after penetration

An unsigned closest-point distance at a penetrated pose can lose which side of
the contact should be considered feasible. Retain the orientation established
from safe geometry.

The proposed refresh uses a pair-local safe surrogate:

1. Keep a previously certified separated configuration for this pair.
2. Conservatively truncate this pair's motion toward its current candidate to
   obtain another separated configuration.
3. Recompute an oriented normal and witness geometry there, and use the selected
   normal for subsequent force evaluations.

This local truncation changes the geometry used to refresh the contact. It does
not truncate the global ALM iterate or make that iterate collision-free. The
pair-local path needs a valid separation test or conservative bound; an endpoint
distance check alone is insufficient.

Keep **force normals** distinct from **planes used for final safety**. A plane
constructed at a later pair-local surrogate need not separate the accepted
starting pose. Only replace a final safety plane after validating it at that
starting pose. Otherwise retain the original certificate.

Open choices: refresh cadence, behavior at degenerate features, limits on normal
rotation, and multiplier transfer when the contact direction changes substantially.

## 5. Initial spherical queries and motion coverage

Start with the existing distance-neighborhood approach rather than swept AABB
candidate detection. This decision requires conservative motion budgets.

For a pair whose two sides can move by at most `B_A` and `B_B`, the initial query
must cover the corresponding possible approach. A sufficient distance threshold
for finding every pair that could enter the activation region is:

```text
query_reach >= hard_gap + activation_gap + B_A + B_B
```

This assumes the initial query is complete within that threshold for the
supported primitive types. Bound motion of every point on each primitive; for
linear simplex motion, bounds on every vertex suffice. Apply final motion caps
so the committed trajectory respects the declared budgets, even if intermediate
ALM iterates travel farther.

The bounds prevent an excluded pair from reaching contact during committed
motion. An overflowing candidate buffer invalidates the certificate; silently
dropping pairs is not acceptable.

Swept AABB detection and solving newly encountered pairs remain future work.

## 6. Final truncation without a global QP or new coloring

Use planes that separate the accepted starting configuration. For a plane
`dot(n,x)=b`, reserve the thickness margin on the two sides. Let `side_i` be `+1`
for vertices assigned to the positive side and `-1` for the negative side.

```text
clearance_i = side_i * (dot(n, start_x_i) - b) - hard_gap/2
approach_i  = max(0, -side_i * dot(n, candidate_dx_i))

if approach_i > 0:
    t_i <= safety_factor * clearance_i / approach_i

t_i = min(1, all incident plane caps, neighborhood motion cap)
accepted_x_i = start_x_i + t_i * candidate_dx_i
```

Each initial clearance must be nonnegative. `0 < safety_factor < 1` preserves a
fraction of existing clearance. Zero approach adds no plane cap; zero clearance
and positive approach stop that vertex. Numerical tolerance must be represented
in the initial margins and feasibility checks.

Every vertex stays on its assigned side throughout its linear motion. Therefore
the convex primitive stays on that side too. The proof applies to independent
vertex fractions; the fractions need not be equal. Combined with candidate
coverage, this gives the intended final safety certificate under the stated
geometric assumptions.

Per-pair kernels can reduce caps with atomic minima into fixed per-vertex arrays.
This needs neither another coloring nor a global QP. It is deliberately
conservative: a side moving away cannot automatically donate its clearance to
the other side. This can reduce sliding progress or cause stalls even when a
less restricted collision-free motion exists.

### Rigid bodies

Use one fraction per body so truncation preserves rigidity. Specify a continuous
rotation path, such as scaling a fixed rotation increment by `t_body`. Bound the
motion of every relevant body point, including rotation. For radius `r` and
rotation-angle magnitude `theta`, `r*theta` bounds rotational path length over
the full increment. Combine it with translation and scale the bound by the body
fraction. Endpoint separation alone does not certify the rotational trajectory.

An optimal coupled choice of fractions could improve progress, but is outside
the first implementation.

Prescribed kinematic geometry must also obey the certified trajectory and motion
budgets. Its commanded motion cannot be silently truncated. If that trajectory
violates coverage or the plane certificate, flag/reject the step or apply an
explicitly defined kinematic-motion policy. This is required for the safety
claim to include moving obstacles.

## 7. Solver flow and proposed data

```text
At construction:
    allocate bounded candidate, history, incidence and cap buffers

At step start:
    validate accepted pose and initialize motion budgets
    query nearby primitive pairs
    canonicalize pair keys and match previous history
    build start-valid safety planes for candidate pairs
    initialize working history; seed or clear unmatched rows

For each configured VBD sweep:
    refresh force geometry according to the chosen policy
    for each existing particle/body color:
        evaluate ALM contact and elasticity
        solve local systems; allow penetrating trial positions
    update each contact multiplier once

At step end:
    compute candidate displacement from accepted starting pose
    reduce all plane and neighborhood caps
    construct accepted positions/body poses
    compute velocities and friction slip from accepted motion
    validate status and publish state plus reconciled history
```

| Buffer / field | Purpose |
| --- | --- |
| Canonical primitive-pair keys, row types, validity mask | Stable matching across detector reorderings; array slot alone is not identity. |
| Primitive indices and endpoint incidence / CSR | Gather and distribute contact contributions using the refactored representation. |
| Barycentric witnesses, force normal, pair-local safe reference | Geometry for force evaluation and oriented normal refresh. |
| Normal multiplier, rho, cushion stiffness and activation gap | Local contact response and history. |
| Optional tangential history | Friction extension; reproject after normal changes and use the updated normal-force budget. |
| Start-valid safety planes and assigned sides | Full-primitive final-motion certificate. |
| Accepted starting pose and candidate displacement | Define the motion being truncated. |
| Per-vertex/body fractions and motion budgets | Atomic-min reduction and neighborhood coverage. |
| Device counts, overflow/status flags, reset generation | Capture-safe bounded execution and invalid-step handling. |
| Committed and working history, or equivalent rollback storage | Avoid publishing history from a rejected step. |

Allocation sizes, kernels and iteration counts must be fixed for CUDA capture.
Device counts and masks guard inactive work; no host readback or allocation
belongs inside replay. Capacity growth can happen outside capture. If capacity
or certificate validation fails, preserve the accepted state and expose a
device-visible invalid-step status.

Rollback must include structural ALM history because elasticity duals also
change during the candidate solve. Preserve reset invalidation when rolling
back: never resurrect history from before a selected-world reset.

## 8. History after truncation and friction

Final clipping changes the pose from which ALM learned its trial multipliers.
Do not silently treat trial history as equilibrated at the accepted pose.
The carry/reconcile policy remains an implementation decision. Compare retained
trial multipliers with an accepted-pose update and measure the resulting
residual, chatter and truncation frequency. A complete rejected step requires
rollback; partial clipping is a separate case.

Compute velocities and tangential slip from accepted motion. A friction extension
should update the normal budget first, reproject tangential history when normals
change, and enforce the Coulomb bound. Its detailed contact-transition policy is
not settled by the normal-contact design.

## 9. Validation and meeting decisions

Before claiming collision safety, test the certificate and candidate coverage
independently of ALM convergence. Include a primitive whose barycentric witness
is separated while another vertex crosses, a pair just outside query range,
moving obstacles, rigid rotation, zero clearance, and candidate overflow.

For the force law, verify release, cushion, wall accumulation, signed-gap
unwinding, and endpoint force balance. Exercise penetrating iterates and normal
refresh across feature changes. Measure accepted and trial residuals separately.

For the complete solver, compare equal GPU time against current VBD/DAT on
resting contact, stacking, sliding, self-contact and large proposed motion.
Report penetration, final truncation fractions, convergence, jitter, stalls,
and capture/replay equivalence including selected-world reset.

**Decisions for the group:** normal-refresh cadence; matching and history
transfer through feature changes; motion-budget/query-radius policy; history
reconciliation after clipping; and whether conservative plane caps provide
enough progress before considering a coupled optimization or swept queries.

## 10. Relation to the elasticity work

The contact and elasticity rows share the ALM/VBD schedule, but their constraints
and feasibility requirements differ. Optional pressure, spring and hinge ALM
are already implemented. Full spatial matrix mu history has a rotation artifact.

The SVD proposal instead stores symmetric material stretch history with
`S = sqrt(transpose(F)*F + epsilon^2*I)`. It preserves the target mu stress
`mu*F` at convergence, with fixed regularization. It has reference and kernel
probes; integrated performance and stability are still open.

[SVD exploration and plots](https://entertainment-atlantic-mats-randy.trycloudflare.com./artifacts/alm-svd-exploration-20260914/index.html)

[Current stage-1 code walkthrough](https://entertainment-atlantic-mats-randy.trycloudflare.com./artifacts/alm-stage1-code-walkthrough-246fb405-20260914/index.html)
