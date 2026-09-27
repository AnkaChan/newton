I have an idea of normalizing the cells, which is convenient for different scales and potentially multi-res application. 

My questions:
- How does the physical laws scales with the cell? If we scale the size of the cell, 
- Will the physics be scaled linearly? 
- And how do we scale them back? 
- How do we scale those stiffness correspondingly, and how do we scale the results back?
---

## Discussion log — 2026-09-27 (Anka with Claude)

### How the laws scale with the cell size h

Strain does not depend on size: the axes A = R^T F are dimensionless and the
Neo-Hookean density depends only on F and the moduli. Size enters through
volume and time. At fixed strain and material:

| quantity | scales as |
|---|---|
| displacements, penetration depth | h |
| elastic energy per cell | mu h^3 |
| force (stress x face area), so the residual in N | mu h^2 |
| lumped mass | rho h^3 |
| inertia term rho h^3 dx^2 / dt^2 | rho h^5 / dt^2 |
| gravity displacement per step | g dt^2 |
| contact penalty with ke = kappa E h | E h^3 |

### Is the scaling linear?

No. Scaling h alone changes the balance between terms (elastic ~ h^3, inertia
~ h^5, gravity ~ h^0). The physics is scale-similar only when these
dimensionless groups are held fixed (objective divided by mu h^3):

| group | meaning |
|---|---|
| lambda / mu (Poisson ratio) | material shape |
| Lambda = rho h^2 / (mu dt^2) | inertia vs stiffness; (dt / elastic wave time across one cell)^2 |
| Gamma = g dt^2 / h | gravity drop per step in cell units |
| Xi = eta / (mu dt) | viscosity vs stiffness per step |
| kappa = ke / (E h), beta = kd / (ke dt), friction mu | contact (already dimensionless in the conditioning) |
| v dt / h | velocity in cell units |

Shrinking h by s while keeping rho and mu requires dt -> dt s and g -> g / s;
keeping dt requires rho -> rho / s^2 and g -> g s. The current augmentation
over E (3 decades) and rho (2 decades) at one h and one dt is effectively a
5-decade sweep of Lambda.

### Normalised simulation with the same F response

Yes: set h = 1, mu = 1 (or E = 1), dt = 1 and transform everything else:

- density rho_hat = rho h^2 / (mu dt^2)
- gravity g_hat = g dt^2 / h
- viscosity eta_hat = eta / (mu dt)
- initial velocity v_hat = v dt / h
- contact ke_hat = ke / (mu h), kd_hat = kd / (mu h dt) (= beta kappa E / mu), friction unchanged
- lambda_hat = lambda / mu

The normalised objective is the physical one divided by the constant mu h^3,
so the implicit-Euler minimiser and F are identical at every step. Scale back
with x = h x_hat, E = S h^3 E_hat, f = S h^2 f_hat, v = (h / dt) v_hat,
d = h d_hat, ke = kappa E h, kd = beta ke dt. Normalising h and stiffness
alone while leaving rho, g, dt in SI changes Lambda and Gamma and gives a
different beam.

Side benefits: one PARDISO fusion factor for all materials and sizes (weights
become constants), and better float32 conditioning (positions and energies of
order one).

### What already is dimensionless in the current inputs

Axes, inertial offset and physical change blocks, RMS-normalised gradient
blocks, boundary flags, contact tokens (cell-frame positions / h, gap / r),
and the loss ratio E / floor (floor = c eps V (lambda + 2 mu + eta/dt + rho
h^2/dt^2) = S h^3 x sum of groups). Only the conditioning channels carry
absolute log h and log dt today; replace them by the groups above.

### Edge information

The 24-value edge descriptor is already dimensionless on a uniform grid: rest
offset / h (3), current receiver-frame offset / h (3), relative rotation
R_i^T R_j (9), transported neighbour axes R_i^T F_j (9). Changes are needed
only for mixed resolution:

- add log(h_j / h_i) so the receiver knows a coarser or finer neighbour
  (always zero on a uniform grid);
- divide offsets by the receiver's own h_i ("how many of my cells away");
- axes need no change (F is per-cell strain);
- the hop-shell topology assumes a regular grid; a mixed-resolution mesh needs
  a face-adjacency neighbour table with slot padding.

### Implementation sketch (contained change)

Register contexts in normalised units inside the physics step, convert on the
way in and out, replace the log h / log dt conditioning by the dimensionless
groups, and sample Lambda and Gamma (equivalently h and dt) in training for
multi-resolution transfer. Network architecture and training loop unchanged.
Geometry with a fixed absolute scale (40-cell beam length vs receptive field)
does not normalise away.

### Validation by unit tests — 2026-09-27 (tests/test_scaling_invariance.py, 9 tests)

Physical scene (h = 0.025 m, dt = 1/300 s, 16 materials) versus its normalised
copy (h = 1, mu = 1, dt = 1, transformed rho, g, eta, v, contact):

| identity | module | deviation |
|---|---|---|
| Y = h Y' | make_inertial_prediction | 2e-16 |
| elastic, inertia, damping, total = mu h^3 (...)' | HexImplicitEulerLoss (float64) | 5e-14 |
| dE/dX = mu h^2 dE'/dX' | autograd | 2e-14 |
| contact energy = mu h^3 (...)', penetration = h d' | contact_energy (float64) | 2e-16 |
| same detected pairs, partner points = h (...)' | detect_contacts | 1.5e-7 (float32) |
| fused positions = h fused' | HexFusion | 3e-16 |
| energy floor = mu h^3 floor' | MixedHexSolverStep (float32) | 8e-8 |
| 3-step implicit-Euler trajectory with the same plain L-BFGS: X = h X', V = (h/dt) V', F identical, same contact pairs | MixedHexSolverStep (float32) | <= 4e-6 h |
| negative: normalising only h and mu | total energy off by 100 % | as expected |

Corrections to the discussion above found by the tests and reviewers:

1. **friction_epsilon is a velocity.** contact_energy uses eps_u = friction_epsilon * dt
   in metres, so the normalised scene needs friction_epsilon' = friction_epsilon * dt / h
   (1.33e-3 for h = 0.025, dt = 1/300). Left at 1e-2 the contact identity breaks by
   0.9 % for slips inside the band.
2. **log_gradient_rms is not dimensionless.** The state feature packs log of the RMS of
   the projected axis gradient in joules, so it carries log(mu h^3) (measured offset
   -2.8118 = log(mu h^3) to 4e-6). Normalise it by the energy scale (mu h^3 or the
   energy floor) or drop it.
3. **Five of the nine conditioning channels carry absolute scale**, not only log h and
   log dt: log1p(lambda/1e5), log1p(mu/1e5), log(rho/1000), log(h/0.025), log(dt/(1/60)).
   Only log1p(eta/(mu dt)), log1p(kappa), beta and mu_f coincide. Replace the five by
   lambda/mu, Lambda, Gamma, Xi. Gravity is a constructor constant of the step, so a
   normalised context must be built with g' = g dt^2 / h.
4. Minor absolute constants outside the energy: static-point sampling margins 0.10 m
   and 0.35 m (contact_scene), RMS_FLOOR 1e-12 J, degeneracy guards 1e-12 m; the
   plane partner radius 1e9 m reaches only a capped token channel.
5. torch.optim.LBFGS is not scale-covariant on its first iterate, so the two
   minimisations follow different paths and share only the minimiser; in float32 the
   trajectory identity holds to about 1e-6 h after a fixed-step polish.
