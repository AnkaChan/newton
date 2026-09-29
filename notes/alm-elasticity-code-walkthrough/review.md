# ALM elasticity in VBD: code walkthrough

## 1. What this implementation changes

**Where this lives / what this part does.** This walkthrough follows particle elasticity from `SolverVBD` configuration through element history, vertex forces, dual updates, and reset. It describes source snapshot `b5a7ab81` on `ankac/alm-dat-walkthrough`. The triangle implementation and bending metric change entered in `60407c40`; the preceding bending-only experiment is `708c91c4`. All embedded source is from this local snapshot, including the full files behind the clickable references.

**Old behavior.** With ALM off, VBD directly evaluates the authored elastic forces. Before the latest extension, enabling particle ALM covered tetrahedral pressure, springs, and bending, but triangle membrane forces still followed the ordinary path. The old bending metric could make its stress history respond extremely slowly.

**New behavior.** Enabling particle ALM also gives each triangle two scalar stress histories, for membrane stretch and area. Triangle and bending metrics now have a material-based floor. The limiting material law is preserved; the finite-iteration forces and curvature change. The existing contact evaluation, damping, and per-color displacement truncation remain in the solve.

The central implementation is `particle_alm_kernels.py:19` for state and updates, `particle_vbd_kernels.py:588` for membrane force/Hessian evaluation, and `solver_vbd.py:2784` / `solver_vbd.py:3515` for the two integration points.

Notation: `lambda` in history arrays means an ALM stress multiplier, written `y` below. `lambda_material` is the authored Lamé parameter. `K = lambda_material + mu`. `C` is an elastic strain row; it need not be zero at equilibrium. `rho` is a numerical metric. A substep is one call to `solver.step`; an iteration is one complete sweep over vertex colors.

Click a `file.py:line` reference to open the embedded source at that line. Click a source line number to leave an offline draft comment. Use **copy as JSON** and paste the entries into `review.comments.json` beside this page, then say "review completed" in chat. Drafts stay in the browser until exported; saving a draft does not write the sidecar automatically.

## One substep through the pipeline (where each section lives) — §2

```text
READING CONTEXT: current behavior ......................... §1
                this map and execution cadence ........... §2
CONSTRUCT SolverVBD (once)
  enable the mode; allocate element histories ............. §3
  -> fixed-address lambda, rho, pending arrays
SolverVBD.step(dt) (once per substep)
  INITIALIZE
    prepare triangles: geometry, mobility, rho, seeding .... §5
    prepare bends / springs / tetrahedra .................. §7, §8
    -> rho frozen for this substep; retained lambda
    existing collision handling and position prediction ... §9
  ITERATE (solver.iterations times)
    existing rigid-body iteration ......................... §9
    for each particle color:
      shared compliant-ALM coefficients ................... §4
      triangle membrane force and Hessian ................. §6
      bending / spring / tet force and Hessian ............ §7, §8
      scalar or tiled vertex solve; existing DAT .......... §9, §10
    -> positions after the complete color sweep
    update all element multipliers once .................. §9
  FINALIZE: update velocities; retain histories ............ §9
RESET / REPLAY: selective reset and CUDA graph lifecycle ... §10
VERIFY: element derivatives and integrated behavior ....... §11
ASSESS: numerical limits and bag-sweep evidence ............ §12
```

In the bag experiment, one displayed frame contains **10 substeps**, and each substep contains **10 iterations**. Therefore each element gets 100 dual updates per frame, interleaved with position solves, and retains its history over the full 360-frame trajectory. This is not ten position sweeps followed by a single dual update at the end of the frame. The loop is in `solver_vbd.py:2449`; the experiment sets the counts in `run_case.py:110` and `run_case.py:137`.

## 3. Solver construction: opt-in configuration and persistent element state

**Where this lives / what this part does.** `SolverVBD.__init__` creates the ALM state once through `create_particle_elasticity_alm_state` at `solver_vbd.py:824`. Preparation and iteration kernels reuse these arrays throughout the simulation.

**Old behavior.** ALM-off execution has no element-history storage. The earlier ALM state had tetrahedron, spring, and hinge arrays.

**New behavior.** Triangle arrays join the same state object. The main switch enables all supported particle element families. The separate deviatoric flag controls only the optional tetrahedral matrix history; it is not required for triangle ALM.

```python
        particle_enable_tile_solve: bool = True,
        particle_elasticity_alm: bool = False,
        particle_elasticity_alm_deviatoric: bool = False,
        particle_elasticity_alm_rho_scale: float = 1.0,
```
(`newton/_src/solvers/vbd/solver_vbd.py:335-338`)

```python

@wp.struct
class ParticleElasticityAlmState:
    """Solver-owned structural stresses, numerical metrics, and pending reseeds."""

    enabled: int
    deviatoric: int
    rho_scale: float
    tet_lambda_mu: wp.array[wp.mat33]
    tet_lambda_pressure: wp.array[float]
    tet_rho_mu: wp.array[float]
    tet_rho_pressure: wp.array[float]
    tri_lambda_stretch: wp.array[float]
    tri_lambda_area: wp.array[float]
    tri_rho_stretch: wp.array[float]
    tri_rho_area: wp.array[float]
    spring_lambda: wp.array[float]
    spring_rho: wp.array[float]
    bend_lambda: wp.array[float]
    bend_rho: wp.array[float]
    tet_pending: wp.array[int]
    tri_pending: wp.array[int]
    spring_pending: wp.array[int]
    bend_pending: wp.array[int]
```
(`newton/_src/solvers/vbd/particle_alm_kernels.py:19-42`)

Each triangle uses two scalar stresses, two scalar rho values, and one integer pending mask: **20 bytes**, excluding array metadata. Bits 1 and 2 track stretch and area initialization independently. The same storage supports both scalar and CUDA tile solves. Allocation uses zero triangle count when disabled at `particle_alm_kernels.py:568`.

Material validation rejects negative triangle mu, nonfinite coefficients, and nonpositive K unless both coefficients are zero. It also checks that K fits float32. Rebuild the solver after topology or material edits; resetting history is for pose changes, not reconfiguring a compiled material/topology specialization. See `particle_alm_kernels.py:513` and `solver_vbd.py:420`.

## 4. Shared compliant-ALM algebra: eliminate auxiliary strain, then update stress

**Where this lives / what this part does.** Particle kernels reuse `_compliant_alm_coefficients` and `_alm_relaxed_ascent` from `rigid_vbd_kernels.py:111`. Coefficients are evaluated inside force assembly; ascent runs once per element after a complete color sweep.

**Old behavior.** A quadratic material row directly transmits stress `k*C` with material curvature k.

**New behavior.** The row transmits a combination of current strain and retained stress. To derive it, introduce an auxiliary strain z and minimize the augmented energy over z at fixed x and multiplier y:

```text
L(x, z, y) = (k/2)*z^2 + y*(C(x)-z) + (rho/2)*(C(x)-z)^2
z*         = (y + rho*C)/(k + rho)
t          = y + rho*(C-z*)
           = k_eff*C + s*y
s          = k/(k+rho)
k_eff      = k*rho/(k+rho)
y_next     = y + rho*(C(x_new)-z*(x_new)) = t(x_new)
```

Thus this is a finite-compliance stress update. Solving `y_next=y` gives `y=k*C`, so the original material law is recovered at a fixed point. Rest area/volume multiplies the row energy and force once, outside this algebra.

```python
@wp.func
def _compliant_alm_coefficients(material_k: float, rho: float):
    """Return stable ``(s, k_eff, a)`` coefficients for compliant ALM.

    ``s=K/(K+rho)``, ``a=rho/(K+rho)``, ``k_eff=K*a``, branched to preserve
    whichever of ``s`` and ``a`` is small. A nonpositive input retires the row:
    ``a=1``, not the ``rho -> 0`` limit, lets the ascent clear its dual too.
    """
    if material_k <= 0.0 or rho <= 0.0:
        return 0.0, 0.0, 1.0

    if material_k >= rho:
        r = rho / material_k
        s = 1.0 / (1.0 + r)
        return s, rho * s, r * s

    r = material_k / rho
    a = 1.0 / (1.0 + r)
    return r * a, material_k * a, a


@wp.func
def _alm_relaxed_ascent(lam: Any, R: Any, material_k: float, rho: float) -> Any:
    """Advance one compliant-ALM dual: ``lam <- s*(lam + rho*R)``.

    Both branches evaluate that expression without forming ``rho*R``, which
    can overflow at large rho. The increment form preserves small ``a`` when
    ``rho <= K``; the distributed form preserves small ``s`` otherwise.
    Instantiated for scalar and ``wp.vec3`` duals.
    """
    s, k_eff, a = _compliant_alm_coefficients(material_k, rho)
    if material_k >= rho:
        return lam + (k_eff * R - a * lam)
    return s * lam + k_eff * R

```
(`newton/_src/solvers/vbd/rigid_vbd_kernels.py:110-144`)

The branches compute ratios instead of forming `k*rho` or `rho*C`, which can overflow. For `rho <= k`, the increment form preserves a small relaxation coefficient. Nonpositive stiffness or rho retires a row: force coefficients become zero and ascent clears its multiplier. `particle_alm_coefficients` and `particle_alm_ascent` are thin wrappers at `particle_alm_kernels.py:45`.

## 5. Before prediction: prepare triangle geometry, rho, and initial stresses

**Where this lives / what this part does.** `_initialize_particles` calls the preparation dispatcher before the predictor at `solver_vbd.py:2784`. `_prepare_triangles` launches one thread per triangle. It recomputes metrics once per substep from the incoming pose; it seeds history only when the corresponding pending bit is set.

**Old behavior.** Triangles needed no ALM preparation. Bending used only an inertia-derived metric.

**New behavior.** Triangle stretch and area get separate geometry-dependent metrics, with the same material-floor policy now used for bending and already used for springs.

The 3x2 deformation gradient is `F=[f0,f1]`. The two scalar invariants are `r=norm(F)` and `J=sqrt(det(F^T F))`. Both are unchanged under a world-space rigid rotation. The returned g0 and g1 are area derivatives with respect to the two columns of F.

```python
@wp.func
def _bounded_cloth_rho(inertia: wp.float64, material_k: float) -> float:
    # As for springs, before float32 saturation retain at least 90% of row
    # curvature and reduce a fixed-pose stress error by at least 90% per update.
    return _bounded_rho(wp.max(inertia, wp.float64(9.0) * wp.float64(material_k)))


@wp.func
def particle_alm_triangle_geometry(f0: wp.vec3, f1: wp.vec3):
    """Return objective norm/area invariants and area gradients for F=[f0,f1]."""
    a = wp.dot(f0, f0)
    b = wp.dot(f1, f1)
    c = wp.dot(f0, f1)
    norm = wp.sqrt(a + b)
    area = wp.sqrt(wp.max(a * b - c * c, 1.0e-20))
    return norm, area, (b * f0 - c * f1) / area, (a * f1 - c * f0) / area
```
(`newton/_src/solvers/vbd/particle_alm_kernels.py:93-108`)

For triangle row C, the inertia estimate is `rho_scale / (A0*dt^2*sum(inv_mass*dot(grad C,grad C)))`. Pinned/inactive vertices contribute zero mobility through `particle_alm_kernels.py:81`. The implemented metric is `max(rho_inertia, 9*k)`, clamped to finite float32 range. The material floor is applied **after** rho scaling; decreasing `rho_scale` cannot lower rho below that floor.

```python
    for vertex in range(3):
        w = -(pose[0] + pose[1])
        if vertex > 0:
            w = pose[vertex - 1]
        gn = (w[0] * f0 + w[1] * f1) / wp.max(norm, 1.0e-10)
        ga = w[0] * g0 + w[1] * g1
        mobility = _particle_mobility(indices[face, vertex], inv_mass, flags)
        mobility_norm += mobility * wp.dot(gn, gn)
        mobility_area += mobility * wp.dot(ga, ga)
    pending = state.tri_pending[face]
    denominator = wp.float64(areas[face]) * wp.float64(dt) * wp.float64(dt)
    state.tri_rho_stretch[face] = 0.0
    if mu > 0.0 and norm > 1.0e-10 and areas[face] > 0.0 and mobility_norm > 0.0:
        state.tri_rho_stretch[face] = _bounded_cloth_rho(
            wp.float64(state.rho_scale) / (denominator * wp.float64(mobility_norm)), mu
        )
        if (pending & 1) != 0:
            state.tri_lambda_stretch[face] = mu * norm
        pending = pending & ~1
    else:
        state.tri_lambda_stretch[face] = 0.0
        pending = pending | 1
    state.tri_rho_area[face] = 0.0
    if area_k > 0.0 and area > 1.0e-10 and areas[face] > 0.0 and mobility_area > 0.0:
        state.tri_rho_area[face] = _bounded_cloth_rho(
            wp.float64(state.rho_scale) / (denominator * wp.float64(mobility_area)), area_k
        )
        if (pending & 2) != 0:
            state.tri_lambda_area[face] = area_k * (area - 1.0) - area_k * (mu / wp.max(area_k, 1.0e-6))
        pending = pending & ~2
    else:
        state.tri_lambda_area[face] = 0.0
        pending = pending | 2
    state.tri_pending[face] = pending
```
(`newton/_src/solvers/vbd/particle_alm_kernels.py:139-172`)

Stretch seeds to `mu*r`; area seeds to `K*(J-1)-mu` subject to the existing small-K guard. At rest, the area multiplier is negative, not zero. It balances the positive stretch contribution. Pending bits prevent reseeding every substep, which would discard temporal history. Zero norm, collapsed area, or zero mobility retire the corresponding row independently.

## 6. Inside each vertex solve: objective triangle forces and curvature

**Where this lives / what this part does.** `evaluate_neo_hookean_membrane_force_hessian_alm` in `particle_vbd_kernels.py:588` runs for each incident triangle during the vertex's color solve. It returns one vertex force and one 3x3 Hessian block.

**Old behavior.** The membrane first Piola stress is `P = mu*F + K*(J-alpha)*dJ/dF`, where `alpha=1+mu/K`.

**New behavior.** Write exactly the same energy, up to a constant, as two quadratic scalar rows. The stretch row has target zero; changing it to `r-sqrt(2)` would change the material model.

```text
E = A0 * [ (mu/2)*r^2 + (K/2)*(J-alpha)^2 ]
C_stretch = r             k_stretch = mu
C_area    = J-alpha       k_area    = K
P_ALM = (t_stretch/r)*F + t_area*dJ/dF
```

```python

    # First Piola-Kirchhoff stress: P = mu*F + lambda*(J_s - alpha)*[g0, g1]
    s = lmbd_nh * (J_s - alpha)
    mu_eval = mu_nh
    norm_curvature = float(0.0)
    if elasticity_alm.enabled != 0:
        # The exact energy is quadratic in ||F||_F and J_s-alpha. Scalar
        # invariant histories rotate with the current geometry, unlike a
        # stored world-space 3x2 stress. The stretch target remains zero.
        norm_squared = wp.max(f0_dot_f0 + f1_dot_f1, 1.0e-20)
        norm = wp.sqrt(norm_squared)
        scale_mu, k_mu, _ = particle_alm_coefficients(mu_nh, elasticity_alm.tri_rho_stretch[face])
        stress_mu = k_mu * norm + scale_mu * elasticity_alm.tri_lambda_stretch[face]
        mu_eval = stress_mu / norm
        norm_curvature = (k_mu - mu_eval) / norm_squared
        scale_area, k_area, _ = particle_alm_coefficients(lmbd_nh, elasticity_alm.tri_rho_area[face])
        s = k_area * ((J_s - 1.0) - mu_nh / lmbd_safe) + scale_area * elasticity_alm.tri_lambda_area[face]
        lmbd_nh = k_area
    P_col0 = mu_eval * f0 + s * g0
    P_col1 = mu_eval * f1 + s * g1

```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:645-665`)

History stores only scalar invariant stresses. Rotation enters through the current F and area gradients, so no world-space 3x2 stress matrix is carried into a rotated pose. This is an objective invariant formulation; it does not integrate the separately proposed tetrahedral SVD/stretch-tensor method.

The force is `-A0*dE_density/dx`. Curvature must also differentiate `F/r`. If a vertex's deformation weights are b0 and b1, the stretch block is:

```text
a = b0*f0 + b1*f1
H_stretch = (t_stretch/r)*(b0^2+b1^2)*I
          + (k_eff_stretch-t_stretch/r)/r^2 * outer(a,a)
```

```python
    I33 = wp.identity(n=3, dtype=float)
    hessian = I_coeff * I33 + c1 * wp.outer(dJ_dx, dJ_dx) - r * wp.outer(w, w)
    if elasticity_alm.enabled != 0:
        norm_gradient_numerator = df0_dx * f0 + df1_dx * f1
        hessian += norm_curvature * wp.outer(norm_gradient_numerator, norm_gradient_numerator)
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:698-702`)

The area block retains the existing per-vertex PSD projection at `particle_vbd_kernels.py:682`. Consequently the returned block is a solver approximation wherever that projection clamps curvature. Objective damping still uses changes in `F^T F` at `particle_vbd_kernels.py:704`, and rest area multiplies both force and Hessian once at `particle_vbd_kernels.py:732`.

## 7. Bending and springs: scalar moment or tension, with bounded stress lag

**Where this lives / what this part does.** `_prepare_bends` computes metrics once per substep at `particle_alm_kernels.py:379`. The bending force function runs per incident hinge in the color solve. `_update_bends` advances one scalar moment history per edge after the sweep.

**Old behavior.** Bending ALM already existed. Its inertia-only rho could be tiny compared with `k=edge_ke*rest_length`. Then `s=k/(k+rho)` was nearly one, so its moment barely adapted to changing shape within ten iterations.

**New behavior.** The metric now has the same `9*k` floor as springs. Bending retains its original signed-angle row `C=theta-theta_rest`, its original rest-length weighting, and its existing damping.

```python
    material_k = properties[edge, 0] * rest_length[edge]
    if material_k > 0.0 and valid != 0 and mobility > 0.0:
        state.bend_rho[edge] = _bounded_cloth_rho(
            wp.float64(state.rho_scale) / (wp.float64(dt) * wp.float64(dt) * wp.float64(mobility)), material_k
        )
        if state.bend_pending[edge] != 0:
            state.bend_lambda[edge] = material_k * (theta - rest_angle[edge])
        state.bend_pending[edge] = 0
    else:
        state.bend_lambda[edge] = 0.0
```
(`newton/_src/solvers/vbd/particle_alm_kernels.py:407-416`)

```python
    k = stiffness * edge_rest_length[bending_index]
    if elasticity_alm.enabled != 0:
        scale, k, _ = particle_alm_coefficients(k, elasticity_alm.bend_rho[bending_index])
        dE_dtheta = k * (theta - edge_rest_angle[bending_index]) + scale * elasticity_alm.bend_lambda[bending_index]
    else:
        dE_dtheta = k * (theta - edge_rest_angle[bending_index])
...
    # Select the derivative for the current vertex without branching
    dtheta_dx = dtheta_dx0 * mask0 + dtheta_dx1 * mask1 + dtheta_dx2 * mask2 + dtheta_dx3 * mask3

    # Compute elastic force and hessian
    bending_force = -dE_dtheta * dtheta_dx
    bending_hessian = k * wp.outer(dtheta_dx, dtheta_dx)
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:855-860,904-909`)

The bending Hessian is the existing outer-product approximation `k_eff*grad(theta)*grad(theta)^T`. It does not include the full `t*Hessian(theta)` geometric term. The elastic angle difference remains the raw signed difference; only the damping increment is wrapped at `particle_vbd_kernels.py:940`.

The floor guarantees `s <= 0.1` and `k_eff >= 0.9*k` before float32 saturation. At a fixed pose, the stress error therefore shrinks by at least a factor of ten per update. For example, with `k=100`, `rho=900`, `C=0.02`, and `y0=0`, the first three updates give `1.8`, `1.98`, and `1.998`, approaching the authored stress 2. This is a statement about stress at a fixed pose, not a convergence guarantee for the coupled position solve.

The regression fixture is a 1 cm hinge with 1e-6 kg vertex masses, `edge_ke=200`, `dt=1/600`, and a 0.02 rad fold. Its expected moment is -0.04. The previous metric reached approximately -1.20e-6 after ten updates, only 0.003% of that moment. The test at `test_particle_alm_kernels.py:50` checks the corrected response.

Springs use `C=length-rest_length`, with the same shared coefficient algebra, scalar tension history, and an existing `9*k` floor at `particle_alm_kernels.py:350`. Their force and geometric curvature remain in `particle_vbd_kernels.py:1779`. There are no springs in the bag experiment.

## 8. Tetrahedra: pressure by default, optional matrix history with a known artifact

**Where this lives / what this part does.** `_prepare_tets` at `particle_alm_kernels.py:253` seeds pressure and optional matrix history. `evaluate_volumetric_neo_hookean_force_and_hessian_alm` evaluates them during the same vertex solves as triangles.

**Old behavior.** Ordinary tet elasticity directly evaluates both the mu and pressure stresses.

**New behavior.** The default particle-ALM mode replaces pressure with a scalar row `C=(det(F)-1)-mu/K`. The mu term remains ordinary unless `particle_elasticity_alm_deviatoric=True` also enables a full 3x3 history for F. Tet metrics remain inertia-derived; the new cloth floor does not apply to them.

```python
    if elasticity_alm.enabled != 0:
        pressure_scale, pressure_effective, _ = particle_alm_coefficients(
            lmbd_nh, elasticity_alm.tet_rho_pressure[tet_id]
        )
        # Keep the rest-pressure offset when 1 + mu / K rounds to 1.
        pressure = (
            pressure_effective * (J - 1.0)
            - pressure_effective * (mu_nh / lmbd_safe)
            + pressure_scale * elasticity_alm.tet_lambda_pressure[tet_id]
        )
        P_mu = mu_nh * F
        if elasticity_alm.deviatoric != 0:
            mu_scale, mu_effective, _ = particle_alm_coefficients(mu_nh, elasticity_alm.tet_rho_mu[tet_id])
            P_mu = mu_effective * F + mu_scale * elasticity_alm.tet_lambda_mu[tet_id]
```
(`newton/_src/solvers/vbd/particle_vbd_kernels.py:244-257`)

Subtracting the pressure offset as `(J-1)-mu/K` preserves it when `1+mu/K` would round to 1. Rest volume is applied once in element assembly. With optional matrix history, the transmitted mu stress contains a retained world-space matrix. At finite iterations that matrix can lag a rigid rotation and create transient forces. The default keeps this mode off. The SVD/stretch replacement remains a derivation and standalone exploration, not an integrated solver path.

## 9. One VBD iteration: solve colors, truncate each color, then update all duals

**Where this lives / what this part does.** `SolverVBD.step` runs the iteration loop at `solver_vbd.py:2449`. `_solve_particle_iteration` evaluates contacts, accumulates element contributions, and processes vertex colors at `solver_vbd.py:3359`.

**Old behavior.** The vertex solve minimizes its local inertia, elastic, damping, and contact contributions. The existing displacement truncation limits each color's proposed movement.

**New behavior.** The vertex force/Hessian assembly reads fixed element multipliers for the entire sweep. A separate pass updates those multipliers only after all colors and their truncation have completed. There is no additional inner nonlinear loop per element and no new collision schedule.

```python
    def _initialize_particles(self, state_in: State, state_out: State, dt: float):
        """Initialize particle positions for the VBD iteration."""
        model = self.model

        # Early exit if no particles
        if model.particle_count == 0:
            return

        prepare_particle_elasticity_alm(model, state_in.particle_q, dt, self._particle_elasticity_alm_state)
```
(`newton/_src/solvers/vbd/solver_vbd.py:2776-2784`)

The preparation call happens before prediction. Contact handling follows the solver's existing configured schedule; enabling elasticity ALM does not force one initial collision query or freeze contact normals.

```python
            self._penetration_free_truncation(state_in.particle_q)

        update_particle_elasticity_alm(model, state_in.particle_q, self._particle_elasticity_alm_state)
        wp.copy(state_out.particle_q, state_in.particle_q)
```
(`newton/_src/solvers/vbd/solver_vbd.py:3513-3516`)

The indentation matters: `_penetration_free_truncation` is inside the color loop, while `update_particle_elasticity_alm` is outside it. Therefore dual ascent sees the actual positions after truncation, rather than proposals that were rejected or shortened.

```python
    face = wp.tid()
    mu = materials[face, 0]
    area_k = mu + materials[face, 1]
    f0, f1 = _triangle_deformation(face, pos, indices, poses[face])
    norm, area, _g0, _g1 = particle_alm_triangle_geometry(f0, f1)
    if norm > 1.0e-10:
        state.tri_lambda_stretch[face] = particle_alm_ascent(
            state.tri_lambda_stretch[face], norm, mu, state.tri_rho_stretch[face]
        )
    else:
        state.tri_lambda_stretch[face] = 0.0
        state.tri_pending[face] = state.tri_pending[face] | 1
    if area > 1.0e-10 and area_k > 0.0:
        state.tri_lambda_area[face] = particle_alm_ascent(
            state.tri_lambda_area[face], (area - 1.0) - mu / wp.max(area_k, 1.0e-6), area_k, state.tri_rho_area[face]
        )
    else:
        state.tri_lambda_area[face] = 0.0
        state.tri_pending[face] = state.tri_pending[face] | 2


```
(`newton/_src/solvers/vbd/particle_alm_kernels.py:183-203`)

Triangle updates reevaluate the current invariants and use the same compliant stress recurrence as the primal force. Invalid rows clear history and mark themselves for reseeding. The dispatcher at `particle_alm_kernels.py:664` similarly launches tetrahedron, spring, and hinge updates. After all iterations, `solver_vbd.py:2468` finalizes particle velocities. History remains available for the next substep.

## 10. Reset, scalar/tile execution, and CUDA graph replay

**Where this lives / what this part does.** `SolverVBD.reset` delegates to history invalidation at `solver_vbd.py:2632`. CPU scalar and GPU tile kernels receive the same state object. Their selection is made at construction in `solver_vbd.py:926`.

**Old behavior.** The earlier ALM implementation already had fixed-address histories, selective world reset, and capture-safe preparation/update passes for the original element families.

**New behavior.** Triangle storage and pending bits participate in the same lifecycle. The scalar membrane call at `particle_vbd_kernels.py:2800` and the tiled call at `particle_vbd_kernels.py:2561` both reach the new ALM evaluator. The ALM-off wrapper supplies an empty disabled state.

```python
    if element < state.tri_pending.shape[0]:
        selected = bool(False)
        for vertex in range(3):
            selected = selected or _reset_world_selected(
                particle_world[tri_indices[element, vertex]], world_mask, reset_all, world_count
            )
        if selected:
            state.tri_pending[element] = 3
            state.tri_lambda_stretch[element] = 0.0
            state.tri_lambda_area[element] = 0.0
            state.tri_rho_stretch[element] = 0.0
            state.tri_rho_area[element] = 0.0
```
(`newton/_src/solvers/vbd/particle_alm_kernels.py:463-474`)

If any incident vertex belongs to a selected world, reset invalidates the shared triangle. It clears stresses and rho, and sets both pending bits. Preparation then seeds from the next incoming pose. This keeps buffer addresses stable for captured replay. Use `solver.reset(state, world_mask=mask, flags=0)` when preserving a user-authored pose while invalidating its solver history.

The captured triangle test compares eager execution with CUDA graph replay, including an in-graph reset, at `test_solver_vbd_alm.py:214`. Repeated-interval proxy coupling is explicitly rejected at `solver_vbd.py:1367`: this implementation does not restore all retained particle stresses when replaying the same physical interval through that coupling protocol.

## 11. Tests: what is verified and what remains an assumption

**Where this lives / what this part does.** Element tests inspect stress algebra and derivatives directly. Solver tests exercise loaded geometry, state lifetime, scalar/tile agreement, and CUDA capture. These tests are offline validation; they do not run as part of a simulation step.

**Old behavior.** The ALM tests focused on tetrahedra, springs, and hinges.

**New behavior.** The extension adds an independent triangle energy reference, objective rotation checks, lifecycle tests, loaded-triangle response, and captured triangle replay. The new small-hinge test detects the earlier bending stress lag.

| Test / evidence | What it checks | Scope limit |
| --- | --- | --- |
| `test_particle_alm_membrane.py:60` | Force and unclamped Hessian against finite differences of independently written energy | Area stress chosen where the PSD clamp is inactive |
| `test_particle_alm_membrane.py:100` | Authored fixed-point force; retained-history force and Hessian rotate correctly | Isolated rest/deformed triangle |
| `test_particle_alm_membrane.py:124` | dt scaling, frozen metric, collapsed-row clearing and reseeding | Element-level geometry |
| `test_particle_alm_membrane.py:146` | Selected worlds, global elements, stable pointers, disabled arrays | Triangle state lifecycle |
| `test_particle_alm_kernels.py:50` | Moment recovery on a light centimeter-scale hinge | Fixed-pose stress, not moving-cloth convergence |
| `test_solver_vbd_alm.py:112` | Loaded triangle matches an analytic response at mu=1e3 and 1e6 | Pinned, noncollapsed triangle with 40 iterations |
| `test_solver_vbd_alm.py:129` | Rest pose remains stable after a rotation with retained history | Selected rotations and material values |
| `test_solver_vbd_alm.py:214` | Eager/captured triangle steps and in-graph reset agree | CUDA replay |
| `test_solver_vbd_alm.py:240` | Scalar/tile agreement on a mixed element model | Short integration comparison |
| `test_solver_vbd_alm.py:299` | Existing self-contact/DAT runs with elasticity ALM | No new ALM contact implementation |

The independent derivative reference explicitly includes both eliminated scalar-row energies:

```python
        def energy(q):
            f = np.column_stack((q[1] - q[0], q[2] - q[0]))
            norm = np.linalg.norm(f)
            c_area = np.linalg.norm(np.cross(f[:, 0], f[:, 1])) - 1.4
            return 0.5 * (
                0.5 * (84.0 / 19.0) * norm**2
                + (108.0 / 19.0) * norm
                + 0.5 * (330.0 / 41.0) * c_area**2
                + (1200.0 / 41.0) * c_area
            )
```
(`newton/tests/test_particle_alm_membrane.py:71-80`)

The recorded validation for implementation commit `60407c40` ran **178 tests** across the ALM element, membrane, ALM solver, and general VBD modules on CPU/CUDA; all passed. This is the previous implementation run, not a new test run performed to generate this walkthrough. The record is `notes/alm-bag-wiggle/results-triangle-bending/validation.json`. This document's checks validate source excerpts, line links, JavaScript, navigation, and offline comment behavior.

## 12. Review findings: numerical choices, unfinished work, and the bag result

**Where this lives / what this part does.** These observations connect the preceding code paths to their numerical consequences. They are implementation limits and review points, not a claim that new bugs were found while writing this document.

- **R1 — The rho floor makes this a conservative ALM variant.** `particle_alm_kernels.py:94` enforces rho at least nine times material row stiffness for triangles and bending. This bounds stress lag but retains at least 90% of material row curvature before float32 saturation. It does not strongly remove stiffness contrast. Reducing `rho_scale` below the floor has no effect on those rows, and no rho continuation or per-iteration adaptation is implemented.
- **R2 — Elasticity ALM does not implement the proposed contact solve.** The actual loop still calls DAT per color at `solver_vbd.py:3513` and follows existing collision scheduling at `solver_vbd.py:3346`. The agreed future contact direction is one initial query, fixed normals during ALM, and a final iterative per-vertex DAT stage. That contact design and its detailed truncation algorithm are not implemented here.
- **R3 — The triangle history is objective; optional tet matrix history still has its rotation limitation.** Scalar norm/area histories enter the current geometry at `particle_vbd_kernels.py:650`. The optional tet path retains a world-space matrix at `particle_vbd_kernels.py:257`. The triangle change does not repair or replace that path.
- **R4 — Correct force does not imply an exact global Hessian.** Triangle area curvature retains its PSD projection at `particle_vbd_kernels.py:682`; bending uses an outer-product block at `particle_vbd_kernels.py:909`. VBD solves local vertex blocks. These choices must be accounted for when reasoning about convergence from the material energy alone.
- **R5 — The bag sweep measures deformation, not residual convergence or speed.** At 10 substeps and 10 iterations, the new triangle-plus-bending mode removes the previous large bending softening. Mean stretch decreases 1.5–7.7% and mean bending decreases 0.6–5.9% relative to ALM off. High-stiffness stretch still plateaus and bending deformation still grows. The missing May scene was reconstructed, the sweep has a single seed, and equal iteration counts do not establish an equal-time speed benefit.

| tri_ke | Mean stretch off | Mean stretch on | Mean bending off (rad) | Mean bending on (rad) |
| --- | --- | --- | --- | --- |
| 1e3 | 10.7540% | 9.9276% | 0.010860 | 0.010257 |
| 1e4 | 1.4536% | 1.3569% | 0.008075 | 0.007596 |
| 1e5 | 0.5618% | 0.5453% | 0.014873 | 0.014526 |
| 1e6 | 0.5454% | 0.5372% | 0.044861 | 0.044611 |
| 1e7 | 0.5578% | 0.5485% | 0.178796 | 0.173909 |

The video, raw per-frame data, settings, and validation remain in `notes/alm-bag-wiggle/results-triangle-bending/index.html`. The earlier bending-only report remains in `notes/alm-bag-wiggle/results/index.html`. The branch's older walkthroughs are historical snapshots; this page includes the triangle extension and updated bending rho policy.
