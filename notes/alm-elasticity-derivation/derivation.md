# Neo-Hookean → 9 + 1 rows → SVD stretch

<p class="lead">The derivation of our elasticity ALM, starting from the material energy and ending at the proposed stretch constraint.</p>

<div class="route"><span>Neo-Hookean energy</span><b>→</b><span>9 matrix + 1 volume rows</span><b>→</b><span>3 principal stretches + 1 volume row</span><b>→</b><span>6 tensor components + 1 volume row<br><small>recommended history representation</small></span></div>

**What is exact?** Rewriting the current solver’s energy in matrix, singular-value, or stretch-tensor form is exact, up to constants. The first step below—from classical logarithmic Neo-Hookean to the current solver’s quadratic volume model—is an approximation near rest and a continuation to inverted configurations.

This page derives the proposal. The production stage-1 code still uses pressure-only ALM by default when elasticity ALM is enabled; its optional full mode uses matrix history. The SVD/stretch mode remains an exploration. [Code walkthrough](../alm-stage1-code-walkthrough-246fb405-20260914/index.html) · [SVD experiments](../alm-svd-exploration-20260914/index.html)

## Hydrostatic term: the one scalar volume constraint

**This is the +1 in both 9 + 1 and the SVD formulation.** It depends only on signed volume J = det F. SVD changes the stretch rows; this row stays the same.

The current material’s volume-dependent energy density is:

<!-- eq:hydro-energy|The hydrostatic energy density is minus mu times J minus one plus K p over two times J minus one squared. -->
$$
\psi_p(J)=-\mu(J-1)+\frac{K_p}{2}(J-1)^2,\qquad K_p=\lambda_L+\mu
$$

Complete the square to obtain one scalar residual:

<!-- eq:hydro-constraint|The hydrostatic constraint is C p equals J minus alpha, where alpha equals one plus mu over K p. -->
$$
C_p=J-\alpha,\qquad\alpha=1+\frac{\mu}{K_p},\qquad\psi_p=\frac{K_p}{2}C_p^2-\frac{\mu^2}{2K_p}
$$

The constant has no effect on forces. For a tet, multiply this density by rest volume V₀. **Do not replace this row with J − 1 while dropping the −μ term:** that would remove the stress that balances the μ stretch term at rest.

### Hydrostatic ALM, explicitly

Introduce a scalar material variable z<sub>p</sub> and enforce **C<sub>p</sub> = z<sub>p</sub>**. With scalar multiplier ℓ<sub>p</sub> and penalty ρ<sub>p</sub>, use:

<!-- eq:hydro-lagrangian|Hydrostatic augmented Lagrangian is K p over two z p squared plus ell p times C p minus z p plus rho p over two times C p minus z p squared. -->
$$
\mathcal{L}_p=\frac{K_p}{2}z_p^2+\ell_p(C_p-z_p)+\frac{\rho_p}{2}(C_p-z_p)^2
$$

Minimizing over z<sub>p</sub> gives its exact local solution:

<!-- eq:hydro-slack|Optimal hydrostatic material variable is ell p plus rho p C p divided by K p plus rho p. -->
$$
z_p^*=\frac{\ell_p+\rho_p C_p}{K_p+\rho_p},\qquad s_p=\frac{K_p}{K_p+\rho_p},\qquad k_p=\frac{K_p\rho_p}{K_p+\rho_p}
$$

The resulting scalar stress coefficient during the position solve, and the multiplier update after a full sweep, are:

<!-- eq:hydro-alm-update|Hydrostatic stress coefficient is k p C p plus s p ell p. Next multiplier equals this coefficient evaluated after the sweep. -->
$$
p_{\mathrm{eval}}=k_p C_p+s_p\ell_p,\qquad\ell_p^+=\ell_p+\rho_p(C_p-z_p^*)=k_p C_p+s_p\ell_p
$$

The first Piola stress contribution is p<sub>eval</sub> cof F. Converting it to Cauchy stress shows why it is hydrostatic:

<!-- eq:hydro-stress|First Piola hydrostatic stress is p eval cofactor F. For invertible F, its Cauchy stress is p eval identity. -->
$$
\mathbf{P}_p=p_{\mathrm{eval}}\operatorname{cof}\mathbf{F},\qquad\boldsymbol{\sigma}_p=\frac{1}{J}\mathbf{P}_p\mathbf{F}^T=p_{\mathrm{eval}}\mathbf{I}\quad(J\ne0)
$$

Here positive p<sub>eval</sub> means tensile mean stress; conventional compression-positive pressure has the opposite sign. The cofactor form remains evaluable at J = 0 even though Cauchy stress does not.

At a converged multiplier, the original volume response is recovered:

<!-- eq:hydro-fixed-point|At equilibrium the hydrostatic coefficient equals K p C p, or K p times J minus one minus mu. At rest it equals minus mu. -->
$$
p^*=\ell_p^*=K_p C_p=K_p(J-1)-\mu,\qquad J=1\ \Longrightarrow\ p^*=-\mu
$$

Thus rest has **−μI from the hydrostatic term and +μI from the stretch term**, giving zero total stress. This is a finite-stiffness elastic equality; its multiplier can have either sign. It has no nonnegative contact-pressure clamp.

## 1. Start with the original material energy

For one tetrahedron, **F** is its 3 × 3 deformation gradient, **J** is its signed volume ratio, and **V₀** is its rest volume. Energy density is energy per rest volume. We omit damping and inertia while deriving the elastic terms.

<!-- eq:geometry|F equals current edge matrix times inverse rest edge matrix; J is determinant F; tet energy equals rest volume times energy density. -->
$$
\mathbf{F}=\mathbf{D}_s\mathbf{D}_m^{-1},\qquad J=\det\mathbf{F},\qquad E_{\mathrm{tet}}=V_0\,\psi(\mathbf{F})
$$

One common compressible Neo-Hookean model is:

<!-- eq:classical|Classical logarithmic Neo-Hookean density: mu over two times squared Frobenius norm F minus three, minus mu log J, plus lambda L over two times log J squared. -->
$$
\psi_{\log}=\frac{\mu}{2}(\|\mathbf{F}\|_F^2-3)-\mu\log J+\frac{\lambda_L}{2}(\log J)^2
$$

Here **μ** is the shear modulus and **λ<sub>L</sub>** is the first Lamé parameter. The subscript distinguishes the material parameter from ALM multipliers. This logarithmic energy requires J > 0. [Smith, de Goes & Kim (2018), Eq. 5](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf#page=3).

### The quadratic volume model used by our code

Let **q = J − 1** be the volume change from rest. Expanding only the logarithmic volume contribution to second order gives:

<!-- eq:taylor|Near J equals one, the logarithmic volume energy is minus mu q plus one half lambda L plus mu times q squared, plus terms of order q cubed. -->
$$
-\mu\log(1+q)+\frac{\lambda_L}{2}\log^2(1+q)=-\mu q+\frac{\lambda_L+\mu}{2}q^2+O(q^3)
$$

Define the volume-row stiffness **K<sub>p</sub> = λ<sub>L</sub> + μ** and retain that quadratic polynomial:

<!-- eq:code-energy|The code density is mu over two times squared norm F minus three, minus mu times J minus one, plus K p over two times J minus one squared. -->
$$
\psi=\frac{\mu}{2}(\|\mathbf{F}\|_F^2-3)-\mu(J-1)+\frac{K_p}{2}(J-1)^2
$$

This is the stable quadratic-determinant model implemented in this worktree. It matches the classical model’s linear elastic response near rest, but gives a different material response at finite volume change. It stays finite at J = 0 and J < 0. It corresponds to the paper’s **Eq. 13 with the reparameterization discussed in §3.4**; the paper’s final Eq. 14 adds another term that our code does not use. [Original derivation](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf#page=4).

<div class="note">K<sub>p</sub> is the stiffness of this volume row. It is <strong>not</strong> the physical bulk modulus, which is λ<sub>L</sub> + 2μ/3 in three dimensions. The derivation assumes μ > 0 and K<sub>p</sub> > 0; zero-stiffness rows should be disabled.</div>

## 2. Complete the square: obtain 9 + 1 scalar rows

The linear volume term can be absorbed into a shifted volume target. Set:

<!-- eq:alpha|Alpha equals one plus mu over K p; C mu is vectorized F in nine dimensions; C p equals J minus alpha. -->
$$
\alpha=1+\frac{\mu}{K_p},\qquad \mathbf{C}_\mu=\operatorname{vec}(\mathbf{F})\in\mathbb{R}^9,\qquad C_p=J-\alpha
$$

Expanding the square on the right verifies the equality:

<!-- eq:complete-square|The volume part equals K p over two times J minus alpha squared, minus mu squared over two K p. -->
$$
-\mu(J-1)+\frac{K_p}{2}(J-1)^2=\frac{K_p}{2}(J-\alpha)^2-\frac{\mu^2}{2K_p}
$$

Therefore:

<!-- eq:nine-plus-one|Density equals mu over two times squared norm of the nine mu rows, plus K p over two times the pressure row squared, minus a constant. -->
$$
\psi=\frac{\mu}{2}\|\mathbf{C}_\mu\|^2+\frac{K_p}{2}C_p^2-\left(\frac{3\mu}{2}+\frac{\mu^2}{2K_p}\right)
$$

**This is the 9 + 1 formulation:** nine entries of F describe the stretch energy, and one determinant residual describes the volume energy. These are ten scalar energy rows, not ten independent geometric degrees of freedom: J is already determined by F.

### Why the targets are F = 0 and J = α, rather than F = I and J = 1

Each energy term has its own preferred state. The μ term alone wants to shrink F. The shifted volume term counteracts that shrinkage. Their **combined** stress vanishes at rest:

<!-- eq:physical-stress|Physical first Piola stress equals mu F plus K p times J minus alpha times cofactor F. At identity, the mu and pressure stresses cancel. -->
$$
\mathbf{P}=\mu\mathbf{F}+K_p(J-\alpha)\operatorname{cof}\mathbf{F},\qquad \mathbf{P}(\mathbf{I})=\mu\mathbf{I}-\mu\mathbf{I}=\mathbf{0}
$$

Replacing the μ row with F − I, or later with singular values minus one, would change the material law. Also, ALM here will **not** impose F = 0 as a hard constraint. The next step explains what the equality constraint actually means.

<details><summary>Which term is deviatoric, and which is hydrostatic?</summary>
<p>The μ part is the stretch term; the determinant part is the volume/pressure term. The μ part is often called “deviatoric” in solver code, but it is not a strictly volume-preserving energy: uniform scaling changes its value too.</p>
<p>For invertible F, converting the pressure contribution to Cauchy stress gives pI, with p = K<sub>p</sub>(J − α), so that contribution is purely hydrostatic. The μ contribution is μFFᵀ/J and generally has both shear and hydrostatic parts. Their cancellation at rest is essential.</p>
</details>

## 3. Turn each energy row into a compliant ALM constraint

Take any row vector **C(x)** with material stiffness **K**. It can be one scalar volume row or the whole μ block. Introduce an auxiliary material variable **z**, and require it to equal the value computed from vertex positions **x**:

<!-- eq:lift|Minimize K over two times norm z squared, subject to z equals C of x. -->
$$
\min_{\mathbf{x},\mathbf{z}}\;\frac{K}{2}\|\mathbf{z}\|^2\qquad\mathrm{subject\ to}\qquad\mathbf{C}(\mathbf{x})-\mathbf{z}=\mathbf{0}
$$

Substituting z = C(x) recovers the original row energy exactly. In a simulation, this row sits alongside all other elastic rows and the inertial objective. **The equality ties geometry to a deformable material variable. It does not require C(x) = 0.**

Using multiplier **ℓ** and penalty **ρ > 0**, the augmented Lagrangian density is:

<!-- eq:lagrangian|Augmented Lagrangian is K over two norm z squared plus multiplier dotted with C minus z plus rho over two norm C minus z squared. -->
$$
\mathcal{L}=\frac{K}{2}\|\mathbf{z}\|^2+\boldsymbol{\ell}\cdot(\mathbf{C}-\mathbf{z})+\frac{\rho}{2}\|\mathbf{C}-\mathbf{z}\|^2
$$

For fixed positions and multiplier, solve for z analytically:

<!-- eq:eliminate-z|The derivative with respect to z is zero when K plus rho times z equals multiplier plus rho C. -->
$$
\frac{\partial\mathcal{L}}{\partial\mathbf{z}}=(K+\rho)\mathbf{z}-\boldsymbol{\ell}-\rho\mathbf{C}=\mathbf{0}
$$

<!-- eq:z-star|Optimal z equals multiplier plus rho C over K plus rho. Define s as K over K plus rho and effective stiffness as K rho over K plus rho. -->
$$
\mathbf{z}^*=\frac{\boldsymbol{\ell}+\rho\mathbf{C}}{K+\rho},\qquad s=\frac{K}{K+\rho},\qquad k_{\mathrm{eff}}=\frac{K\rho}{K+\rho}
$$

Substitute z* back into the Lagrangian. This produces the objective used during the position solve, with the multiplier held fixed:

<!-- eq:reduced|Reduced density is effective stiffness over two norm C squared plus s multiplier dotted C minus norm multiplier squared over two K plus rho. -->
$$
\mathcal{L}_{\mathrm{red}}=\frac{k_{\mathrm{eff}}}{2}\|\mathbf{C}\|^2+s\,\boldsymbol{\ell}\cdot\mathbf{C}-\frac{\|\boldsymbol{\ell}\|^2}{2(K+\rho)}
$$

The last term is constant with respect to vertex positions. Differentiating the other two terms gives an effective row stress **t**. The dual update has exactly the same expression:

<!-- eq:dual|Effective row stress and next multiplier both equal s times the old multiplier plus effective stiffness times C. -->
$$
\mathbf{t}=k_{\mathrm{eff}}\mathbf{C}+s\boldsymbol{\ell},\qquad\boldsymbol{\ell}^+=\boldsymbol{\ell}+\rho(\mathbf{C}-\mathbf{z}^*)=s\boldsymbol{\ell}+k_{\mathrm{eff}}\mathbf{C}
$$

In our VBD scheme, evaluate the force using the current C and frozen history throughout a full color sweep; update the stored multiplier after that sweep. Eliminate z during each evaluation. Holding an old z fixed instead would produce a different primal curvature, ρ.

### Why this recovers the original material

At a fixed point of the multiplier update:

<!-- eq:fixed-point|A fixed multiplier equals s times itself plus effective stiffness times C, which implies multiplier equals K C and effective stress equals K C. -->
$$
\boldsymbol{\ell}^*=s\boldsymbol{\ell}^*+k_{\mathrm{eff}}\mathbf{C}\quad\Longrightarrow\quad\boldsymbol{\ell}^*=K\mathbf{C},\qquad\mathbf{t}^*=K\mathbf{C}
$$

Thus the row has the original material stress at a fixed point, while its temporary solve can use a smaller stiffness. Finite iterations produce a different response; fixed-point consistency alone does not prove convergence.

## 4. Apply that derivation to the current 9 + 1 scheme

Use a matrix multiplier **Λ<sub>F</sub>** for the nine μ rows, and scalar multiplier **ℓ<sub>p</sub>** for the pressure row. Compute separate s and k coefficients using K = μ or K = K<sub>p</sub>, respectively.

<!-- eq:matrix-alm|Current matrix ALM stress is effective mu stiffness times F plus s mu times Lambda F, plus the effective pressure times cofactor F. -->
$$
\mathbf{P}=k_\mu\mathbf{F}+s_\mu\boldsymbol{\Lambda}_F+\left[k_p(J-\alpha)+s_p\ell_p\right]\operatorname{cof}\mathbf{F}
$$

<!-- eq:matrix-updates|After the sweep, update Lambda F from F and update the pressure multiplier from J minus alpha. -->
$$
\boldsymbol{\Lambda}_F^+=s_\mu\boldsymbol{\Lambda}_F+k_\mu\mathbf{F},\qquad\ell_p^+=s_p\ell_p+k_p(J-\alpha)
$$

At equilibrium Λ<sub>F</sub> = μF and ℓ<sub>p</sub> = K<sub>p</sub>(J − α), so the original stress is recovered. All multipliers and penalties here use energy-density units; multiply the complete material contribution by V₀ when assembling tet energy and vertex forces.

### The reason to explore SVD: old matrix history remembers rotation

Start at rest with Λ<sub>F</sub> = μI and ℓ<sub>p</sub> = −μ. Rotate the tet to F = R while keeping history fixed for the next primal solve. J is still one. The pressure contribution becomes −μR, but the matrix history term still points along I:

<!-- eq:rotation-artifact|At rotated rest with old matrix history, total stress is s mu times mu times identity minus R, which is generally nonzero. -->
$$
\mathbf{P}(\mathbf{R})=k_\mu\mathbf{R}+s_\mu\mu\mathbf{I}-\mu\mathbf{R}=s_\mu\mu(\mathbf{I}-\mathbf{R})
$$

That is an artificial force under a rigid rotation at finite ALM iteration count. The material energy itself is rotation-invariant; the temporary matrix-history objective is not. We want a history variable that does not change when the body rotates.

## 5. Use SVD: 9 stretch entries become 3 principal stretches

Factor F using an ordinary SVD:

<!-- eq:svd|F equals U times diagonal singular values times V transpose; its squared Frobenius norm is the sum of the three squared singular values. -->
$$
\mathbf{F}=\mathbf{U}\operatorname{diag}(\sigma_1,\sigma_2,\sigma_3)\mathbf{V}^T,\qquad\|\mathbf{F}\|_F^2=\sum_{i=1}^3\sigma_i^2
$$

Left and right multiplication by orthogonal matrices preserves the Frobenius norm. Therefore the same target energy can use only three μ rows:

<!-- eq:three-plus-one|SVD mu constraint is the vector of three singular values; pressure constraint remains determinant F minus alpha. -->
$$
\mathbf{C}_\mu^{\mathrm{svd}}=(\sigma_1,\sigma_2,\sigma_3)^T,\qquad C_p=\det\mathbf{F}-\alpha
$$

<!-- eq:svd-energy|Density is mu over two times the sum of squared singular values plus K p over two times determinant F minus alpha squared plus the previous constant. -->
$$
\psi=\frac{\mu}{2}\sum_{i=1}^3\sigma_i^2+\frac{K_p}{2}(\det\mathbf{F}-\alpha)^2+\mathrm{constant}
$$

This is the exact **3 + 1 energy representation**. The singular values themselves are the rows, **not σ − 1**. Keep the pressure row’s signed determinant: an ordinary SVD has nonnegative singular values, whose product is |J|. Warp can use a signed final singular value; evaluating det F directly avoids relying on that convention.

Away from repeated or zero singular values, the μ stress with three retained scalar multipliers is:

<!-- eq:scalar-svd-stress|The scalar SVD ALM stress is U times diagonal effective stiffness sigma i plus s mu multiplier i times V transpose. -->
$$
\mathbf{P}_\mu=\mathbf{U}\operatorname{diag}(k_\mu\sigma_i+s_\mu\ell_i)\mathbf{V}^T,\qquad\ell_i^+=s_\mu\ell_i+k_\mu\sigma_i
$$

At equilibrium ℓᵢ = μσᵢ, giving P<sub>μ</sub> = μF. It removes the rotated-rest problem when the scalar histories agree.

### Why three sorted histories are not our final recommendation

Suppose two principal stretches cross. “Largest singular value” switches from one material direction to another. If those sorted slots carry different multipliers, their stored stresses switch directions too. The resulting force can jump even though the target material energy is smooth. Equal singular values also admit arbitrary basis choices.

The remedy we explored is to retain the **whole symmetric stretch-history tensor in material coordinates**. Its eigenvalues still come from SVD, but its history does not attach to an arbitrary sorted slot.

## 6. SVD supplies a stretch tensor: the recommended 6 + 1 rows

The right stretch tensor is the positive-semidefinite square root of FᵀF. To make its derivative well-defined at collapsed configurations too, use a fixed positive regularization **ε**:

<!-- eq:stretch|Regularized right stretch is the square root of F transpose F plus epsilon squared identity; SVD evaluates it as V diagonal r V transpose. -->
$$
\mathbf{S}_\varepsilon=(\mathbf{F}^T\mathbf{F}+\varepsilon^2\mathbf{I})^{1/2}=\mathbf{V}\operatorname{diag}(r_i)\mathbf{V}^T
$$

<!-- eq:regularized-values|Each r i is square root of sigma i squared plus epsilon squared; the squared norm of S epsilon equals squared norm of F plus three epsilon squared. -->
$$
r_i=\sqrt{\sigma_i^2+\varepsilon^2},\qquad\|\mathbf{S}_\varepsilon\|_F^2=\|\mathbf{F}\|_F^2+3\varepsilon^2
$$

The second equality follows by taking the trace of S<sub>ε</sub>². **Fixed ε adds only a constant to the target μ energy**, so its physical force stays exactly μF. This remains true for inverted F; it does not depend on the sign convention of the SVD.

A symmetric 3 × 3 matrix has six independent components. To turn its Frobenius norm into the ordinary norm of a six-vector, give the off-diagonal entries a √2 weight:

<!-- eq:mandel|The six mu rows are S11, S22, S33, square root two S12, square root two S13, and square root two S23. -->
$$
\mathbf{C}_\mu^{S}=(S_{11},S_{22},S_{33},\sqrt{2}S_{12},\sqrt{2}S_{13},\sqrt{2}S_{23})^T
$$

Here S abbreviates S<sub>ε</sub>. The weights count each symmetric off-diagonal pair twice, as the full matrix norm does. A full symmetric `mat33` using Frobenius inner products already does this correctly.

<!-- eq:six-plus-one|The proposed constraints are the six Mandel components of regularized S and one determinant residual; their energy equals the old energy plus a constant. -->
$$
\frac{\mu}{2}\|\mathbf{C}_\mu^S\|^2+\frac{K_p}{2}C_p^2=\psi+\mathrm{constant},\qquad C_p=J-\alpha
$$

This is **6 + 1 independent stored tensor components**, rather than the 3 + 1 scalar-slot representation. We retain material directions in the history. A prototype can reuse the existing nine-float matrix allocation while enforcing symmetry.

Under any superposed rigid rotation Q, (QF)ᵀ(QF) = FᵀF. Consequently S is unchanged. Reordering the SVD columns also leaves the reconstructed S unchanged.

## 7. Derive the force of the stretch constraint

Use a symmetric material multiplier **Λ**. Applying the same compliant ALM elimination from Section 3 gives:

<!-- eq:stretch-reduced|The stretch reduced density is k mu over two times squared norm S plus s mu times the Frobenius inner product of Lambda and S plus a position-independent constant. -->
$$
\mathcal{L}_\mu^{\mathrm{red}}=\frac{k_\mu}{2}\|\mathbf{S}\|_F^2+s_\mu\boldsymbol{\Lambda}:\mathbf{S}+\mathrm{constant}
$$

<!-- eq:stretch-update|Seed Lambda as mu S. After each full sweep, update it as s mu Lambda plus k mu S. -->
$$
\boldsymbol{\Lambda}_{\mathrm{seed}}=\mu\mathbf{S},\qquad\boldsymbol{\Lambda}^+=s_\mu\boldsymbol{\Lambda}+k_\mu\mathbf{S}
$$

The colon means the sum of corresponding matrix-entry products. The first energy term differentiates directly to k<sub>μ</sub>F because ‖S‖² and ‖F‖² differ only by a constant. The history term needs the derivative of S.

### Differentiate the square-root identity

Differentiate S² = FᵀF + ε²I, holding ε fixed:

<!-- eq:stretch-differential|S times differential S plus differential S times S equals F transpose differential F plus differential F transpose F. -->
$$
\mathbf{S}\,d\mathbf{S}+d\mathbf{S}\,\mathbf{S}=\mathbf{F}^T d\mathbf{F}+d\mathbf{F}^T\mathbf{F}
$$

Introduce the symmetric matrix **X** satisfying a local Sylvester equation:

<!-- eq:sylvester|S X plus X S equals Lambda. -->
$$
\mathbf{S}\mathbf{X}+\mathbf{X}\mathbf{S}=\boldsymbol{\Lambda}
$$

Then move the differential through the Frobenius product:

<!-- eq:chain-rule|Lambda colon differential S equals X colon S differential S plus differential S S, which equals two F X colon differential F. -->
$$
\boldsymbol{\Lambda}:d\mathbf{S}=\mathbf{X}:(\mathbf{S}\,d\mathbf{S}+d\mathbf{S}\,\mathbf{S})=2\mathbf{F}\mathbf{X}:d\mathbf{F}
$$

So the exact first Piola stress for the frozen-history stretch objective is:

<!-- eq:stretch-stress|Stretch ALM stress equals effective mu stiffness times F plus two s mu times F X. -->
$$
\mathbf{P}_\mu=k_\mu\mathbf{F}+2s_\mu\mathbf{F}\mathbf{X}
$$

**X does not require a global solve.** In the SVD basis, the Sylvester equation is just entrywise division:

<!-- eq:local-solve|Transform Lambda into the V basis, divide each entry by r i plus r j, and transform back to obtain X. -->
$$
\widehat{\boldsymbol{\Lambda}}=\mathbf{V}^T\boldsymbol{\Lambda}\mathbf{V},\qquad\widehat{X}_{ij}=\frac{\widehat{\Lambda}_{ij}}{r_i+r_j},\qquad\mathbf{X}=\mathbf{V}\widehat{\mathbf{X}}\mathbf{V}^T
$$

The denominators are **sums**, so repeated positive stretches cause no singularity. Positive ε also protects zero stretches. The force derives from S itself and is independent of the arbitrary SVD basis. The square-root differential is standard; the ALM substitution and stress above are our derivation. [Bisson & Pennec, stretch differential](https://comptes-rendus.academie-sciences.fr/mathematique/item/10.5802/crmath.692.pdf).

### Verify the original μ force is recovered

At an ALM fixed point, Λ = μS. Substituting into the Sylvester equation gives X = μI/2:

<!-- eq:stretch-fixed-point|At equilibrium Lambda equals mu S, X equals mu identity over two, and P mu equals k mu plus s mu times mu times F, which equals mu F. -->
$$
\boldsymbol{\Lambda}^*=\mu\mathbf{S}\quad\Longrightarrow\quad\mathbf{X}^*=\frac{\mu}{2}\mathbf{I}\quad\Longrightarrow\quad\mathbf{P}_\mu^*=(k_\mu+s_\mu\mu)\mathbf{F}=\mu\mathbf{F}
$$

With the pressure row included, the complete stress is:

<!-- eq:full-stretch-stress|Full stress is k mu F plus two s mu F X plus k p times J minus alpha plus s p pressure multiplier times cofactor F. -->
$$
\mathbf{P}=k_\mu\mathbf{F}+2s_\mu\mathbf{F}\mathbf{X}+\left[k_p(J-\alpha)+s_p\ell_p\right]\operatorname{cof}\mathbf{F}
$$

At equilibrium this is precisely the stress from Section 2. For a rigidly rotated rest state with equilibrium history, it gives zero total stress. With arbitrary fixed material history, it remains objective: rotating F rotates P in the same way. This does not remove physical torques caused by real deformation.

## 8. What changes in the solve, and what stays the same

| Representation | μ rows / history | Target μ energy | Finite-iteration issue |
|---|---|---|---|
| Matrix ALM | 9 entries of F | Original | Retained spatial history produces the rotated-rest artifact. |
| Scalar SVD ALM | 3 singular values | Original | Unequal sorted histories can jump between material directions. |
| Proposed stretch ALM | 6 independent entries of S<sub>ε</sub> | Original plus a constant | Smooth objective history; additional local derivative work and stiff behavior near collapse need testing. |

All three use the same scalar pressure row C<sub>p</sub> = J − α. None introduces a new global linear system or a new coloring. The proposed mode computes SVD and local matrix operations per tet evaluation and retains the existing sweep/dual-update schedule.

**Same target material does not mean the same temporary Hessian.** During a primal solve Λ is fixed. Do not substitute Λ = μS(F) before taking derivatives: that would incorrectly let the history move with a trial vertex displacement.

<details><summary>Exact local tangent for implementation</summary>

<p>For a trial deformation-gradient change A, differentiate S and X with the multiplier fixed:</p>

<!-- eq:tangent-stretch|S H plus H S equals F transpose A plus A transpose F. -->
$$
\mathbf{S}\mathbf{H}+\mathbf{H}\mathbf{S}=\mathbf{F}^T\mathbf{A}+\mathbf{A}^T\mathbf{F},\qquad\mathbf{H}=D\mathbf{S}[\mathbf{A}]
$$

<!-- eq:tangent-x|S Z plus Z S equals minus H X plus X H. -->
$$
\mathbf{S}\mathbf{Z}+\mathbf{Z}\mathbf{S}=-(\mathbf{H}\mathbf{X}+\mathbf{X}\mathbf{H}),\qquad\mathbf{Z}=D\mathbf{X}[\mathbf{A}]
$$

<!-- eq:tangent-stress|Derivative of P mu in direction A is k mu A plus two s mu times A X plus F Z. -->
$$
D\mathbf{P}_\mu[\mathbf{A}]=k_\mu\mathbf{A}+2s_\mu(\mathbf{A}\mathbf{X}+\mathbf{F}\mathbf{Z})
$$

<p>Both Sylvester solves reuse the current SVD basis and the same positive denominators. For vertex a, let w<sub>a</sub> be its rest shape-function gradient. A trial displacement u gives A = uw<sub>a</sub>ᵀ. The force is −V₀Pw<sub>a</sub>, and the elastic energy Hessian action is V₀DP[A]w<sub>a</sub>.</p>
<p>The exact frozen-history block can be indefinite, and small ε can make it stiff near collapse. The prototype therefore needs a suitable local-block stabilization and finite-iteration simulation checks. SVD alone does not settle those choices.</p>

</details>

## Source and scope

This derivation follows the stage-1 code in `newton/_src/solvers/vbd/particle_vbd_kernels.py`, especially `evaluate_volumetric_neo_hookean_force_and_hessian_alm`, and the multiplier lifecycle in `particle_alm_kernels.py`, at implementation commit `246fb405`. The document uses density conventions consistently and omits damping, inertial terms, and unrelated contact constraints. It describes the material equations with numerical denominator guards inactive; the current code floors the denominator in the pressure offset at 10⁻⁶, which changes that offset for exceptionally small positive Kₚ.

The coefficient conversion and classical/stable material distinction are grounded in [Smith, de Goes & Kim, *Stable Neo-Hookean Flesh Simulation* (2018), Eqs. 5, 13 and §3.4](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf). The ALM lifting, history comparison, and regularized stretch proposal are derived here. Existing numerical evidence is in the [SVD exploration report](../alm-svd-exploration-20260914/index.html).
