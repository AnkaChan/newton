# Triangle membrane and bending ALM

Enable with `SolverVBD(model, particle_elasticity_alm=True)`. This now includes
triangle membrane stretch and area as well as the existing tetrahedral,
spring, and dihedral terms. The default remains ALM off. Damping, contact,
collision scheduling, and per-color DAT keep their existing formulations.

## Objective triangle rows

For the existing 3x2 membrane deformation gradient `F`, define
`r = ||F||_F`, `J = sqrt(det(F^T F))`, and `K = authored_lambda + mu`.
The solver's Neo-Hookean energy, up to an additive constant, is exactly

```text
E = A0 [ mu/2 * r^2 + K/2 * (J - alpha)^2 ]
alpha = 1 + mu/K
```

We use two scalar compliant ALM rows: `C_stretch = r` and
`C_area = J - alpha`. The stretch target is zero, not `sqrt(2)`; the two
terms balance to give zero force at rest. This preserves the existing
material energy without requiring an SVD. The representation uses only
the invariants needed by this isotropic energy. Both are unchanged by
world-space rigid rotations, so retained scalar histories have no
world-space matrix rotation artifact. This does not implement the proposed
tetrahedral SVD/stretch-tensor replacement.

For either row with stiffness `k`, history `lambda`, and metric `rho`:

```text
s = k/(k+rho)
k_eff = k*rho/(k+rho)
t = k_eff*C + s*lambda
lambda_next = t                 # after the complete color sweep and DAT
```

The triangle first Piola stress is
`P = (t_stretch/r) F + t_area dJ/dF`, multiplied by rest area exactly once
when accumulating vertex forces. At the fixed point, `lambda=k*C`, hence
`P = mu F + [K(J-1)-mu] dJ/dF`, the original material law.

For a vertex whose deformation weights are `(b0,b1)`, let
`a = b0*f0 + b1*f1` and `g = dJ/dx`. The stretch Hessian block is

```text
(t_stretch/r)*(b0^2+b1^2)*I
    + (k_eff_stretch - t_stretch/r)/r^2 * outer(a,a)
```

The second term is essential: it differentiates the normalized invariant
gradient. Area curvature uses the existing per-vertex PSD projection,
with the effective area stiffness and transmitted area stress. Damping
still uses the change in `F^T F`. Zero-norm and collapsed-area rows retire
independently and reseed when valid again. The existing small-area and
small-material denominator guards remain in use.

Each triangle stores two stresses, two scalar metrics, and initialization
bits (20 bytes). Preparation, ascent, and selective world reset operate on
fixed-address arrays and work inside CUDA graphs. No histories are allocated
when ALM is off. Both scalar and CUDA tile solves evaluate the new rows.

## Why bending needed a metric floor

The old bending metric was `rho_inertia = rho_scale/(dt^2 * mobility)`.
On a light centimeter-scale hinge it can be orders of magnitude below
the material moment stiffness `k = edge_ke * rest_length`. Consequently,
`s=k/(k+rho)` approaches one and the multiplier barely adapts during a
short solve. This explains why the bending-only bag comparison softened
so much despite retaining the same converged material law.

A regression fixture with 1 cm edges, 1e-6 kg vertex masses, edge stiffness
200, `dt=1/600`, and a 0.02 rad fold should carry a moment of -0.04.
Previously, ten updates reached only -1.20e-6 (0.003% of that moment).

Triangle and bending rows now follow the existing spring policy:

```text
triangle rho_inertia = rho_scale / (A0 * dt^2 * sum_i(inv_mass_i * |grad C_i|^2))
bending  rho_inertia = rho_scale / (dt^2 * sum_i(inv_mass_i * |grad theta_i|^2))
rho = max(rho_inertia, 9*k)
```

The floor acts after scaling the inertia estimate. For active rows before
float32 saturation, it bounds fixed-pose stress retention by 0.1 per
update and retains at least 90% of material row curvature. It prevents
gross stress lag; it does not guarantee convergence of the coupled
position solve, nor imply a performance advantage over ordinary VBD.
Tetrahedral metrics are unchanged.

## Validation

Tests cover independent finite-difference forces and unclamped Hessians,
original material forces at a fixed point, rigid rotation with retained
history, metric scaling, collapsed-row retirement/reseeding, selected-world
reset, disabled storage, loaded-triangle analytic response, CPU/CUDA tile
agreement, and captured replay with an in-graph reset. The small-hinge
stress test and invalid triangle material checks fail on the preceding code.

All 178 tests in the ALM element, membrane, ALM solver, and general VBD suites
passed on CPU/CUDA. Repository pre-commit checks passed.

## Pinned-bag result

The [local HTML and side-by-side video](alm-bag-wiggle/results-triangle-bending/index.html)
compare ALM off with triangle-plus-bending ALM at revision `60407c40`. Both use
10 substeps and 10 iterations, 360 frames at 60 fps, the same reconstructed
May fixture, and unchanged contact settings. All 36,000 solver steps completed
with finite saved states. The earlier
[bending-only comparison](alm-bag-wiggle/results/index.html) is preserved.

| Triangle stiffness | Mean stretch off | Mean stretch on | Mean bend off (rad) | Mean bend on (rad) |
| --- | --- | --- | --- | --- |
| 1e3 | 10.7540% | 9.9276% | 0.010860 | 0.010257 |
| 1e4 | 1.4536% | 1.3569% | 0.008075 | 0.007596 |
| 1e5 | 0.5618% | 0.5453% | 0.014873 | 0.014526 |
| 1e6 | 0.5454% | 0.5372% | 0.044861 | 0.044611 |
| 1e7 | 0.5578% | 0.5485% | 0.178796 | 0.173909 |

The previous large bending softening is absent. The change relative to ALM off
is modest: mean stretch decreases 1.5–7.7% and mean bend decreases 0.6–5.9%.
The high-stiffness stretch plateau and growth in bend deformation remain.
These are geometric deformation scores, not equilibrium residuals, and this
single equal-iteration sweep does not establish a convergence or speed benefit.
