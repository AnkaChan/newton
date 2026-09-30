# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Derive triangle, tet, and hinge ALM with K_eff and s; build an offline HTML page."""

from pathlib import Path

from build import card, eq, page


def section(number, anchor, title, content):
    return card(number, title, content).replace("<section>", f'<section id="{anchor}">', 1)


sections = [
    section(
        1,
        "shared",
        "The spring derivation applies to each energy coordinate",
        "<p>Start with one scalar coordinate <i>C(x)</i> whose energy is <i>w k C²/2</i>. "
        "<i>k</i> is its material stiffness and <i>w</i> is a constant element weight. "
        "The weight will be rest area for a triangle, rest volume for a tet, and one for a hinge.</p>"
        + eq(
            r"E_i=\frac{w k_i}{2}C_i(x)^2,\qquad K_{\mathrm{eff},i}=\frac{k_i\rho_i}{k_i+\rho_i},\qquad s_i=\frac{k_i}{k_i+\rho_i}",
            True,
        )
        + eq(r"K_{\mathrm{eff},i}=\rho_i s_i=k_i(1-s_i)")
        + "<p>Copy the coordinate into an auxiliary <i>zᵢ</i> and require <i>Cᵢ = zᵢ</i>. "
        "With the same weight multiplying all three terms, the augmented energy is:</p>"
        + eq(r"\mathcal L_i=w\left[\frac{k_i}{2}z_i^2+\lambda_i(C_i-z_i)+\frac{\rho_i}{2}(C_i-z_i)^2\right]")
        + "<p>Minimize over the auxiliary and express the answer using the two coefficients:</p>"
        + eq(r"(k_i+\rho_i)z_i=\lambda_i+\rho_i C_i")
        + eq(r"z_i^*=\frac{K_{\mathrm{eff},i}}{k_i}C_i+\frac{s_i}{k_i}\lambda_i", True)
        + "<p>Substitute it back into the augmented energy:</p>"
        + eq(
            r"\widehat{\mathcal L}_i=w\left[\frac{K_{\mathrm{eff},i}}{2}C_i^2+s_i\lambda_i C_i-\frac{s_i}{2k_i}\lambda_i^2\right]"
        )
        + "<p>While positions are solved, the multiplier and penalty are fixed. The last term is "
        "therefore constant with respect to positions.</p>"
        + '<p class="note">The multipliers here use the same normalization as the code: the rest-area '
        "or rest-volume weight sits outside the entire row. Do not multiply the stored multiplier by that weight again.</p>",
    ),
    section(
        2,
        "updates",
        "Derive force, curvature, and multiplier update once",
        "<p>Differentiate the reduced energy. For vertex <i>v</i>, define only the usual geometry "
        "gradient <i>gᵢᵥ = ∇<sub>xᵥ</sub>Cᵢ</i>. The elastic force is:</p>"
        + eq(r"\mathbf f_{iv}=-w\left(K_{\mathrm{eff},i}C_i+s_i\lambda_i\right)\mathbf g_{iv}", True)
        + "<p>Differentiating once more gives the exact diagonal vertex block of the energy Hessian:</p>"
        + eq(
            r"H_{iv}=w\left[K_{\mathrm{eff},i}\mathbf g_{iv}\mathbf g_{iv}^T+\left(K_{\mathrm{eff},i}C_i+s_i\lambda_i\right)\nabla_{x_v}^2 C_i\right]"
        )
        + "<p>The second term comes from nonlinear geometry. A nonlinear coordinate cannot generally "
        "be handled by replacing material stiffness with K_eff everywhere. The implementation-specific "
        "Hessian approximations are identified below.</p>"
        + "<p>After the position sweep, use the accepted coordinate <i>Cᵢ(xⁿ⁺¹)</i> in the multiplier update. "
        "Expand the mismatch explicitly:</p>"
        + eq(r"C_i-z_i^*=C_i-\left[\frac{K_{\mathrm{eff},i}}{k_i}C_i+\frac{s_i}{k_i}\lambda_i^n\right]")
        + eq(r"C_i-z_i^*=s_i C_i-\frac{s_i}{k_i}\lambda_i^n")
        + eq(r"\lambda_i^{n+1}=\lambda_i^n+\rho_i(C_i-z_i^*)=K_{\mathrm{eff},i}C_i(x^{n+1})+s_i\lambda_i^n", True)
        + "<p>Since <i>K_eff,i = kᵢ(1 &minus; sᵢ)</i>, this is also a weighted update toward the current material response:</p>"
        + eq(r"\lambda_i^{n+1}=s_i\lambda_i^n+(1-s_i)k_i C_i(x^{n+1})")
        + "<p>At a fixed point, <i>λᵢ = kᵢ Cᵢ</i> and the original force is recovered. These identities "
        "also hold component by component for a vector or matrix coordinate, using one shared pair of coefficients for that block.</p>",
    ),
    section(
        3,
        "triangle-energy",
        "Triangle elasticity: identify the two scalar coordinates",
        "<p><b>Implemented.</b> A triangle has a 3 by 2 deformation gradient. Let Dₘ be its 2 by 2 "
        "rest-edge matrix expressed in the rest plane:</p>"
        + eq(r"F=[x_1-x_0,\;x_2-x_0]D_m^{-1},\qquad r=\|F\|_F,\qquad J=\sqrt{\det(F^TF)}")
        + "<p><i>J</i> is the area ratio. Use the material constants:</p>"
        + eq(r"K=\lambda_{\mathrm{mat}}+\mu,\qquad \alpha=1+\frac{\mu}{K}")
        + "<p>The stable Neo-Hookean membrane energy used by this branch is:</p>"
        + eq(r"E_{\triangle}=A_0\left[\frac{\mu}{2}(r^2-2)-\mu(J-1)+\frac{K}{2}(J-1)^2\right]")
        + "<p>Complete the square in the area term:</p>"
        + eq(r"-\mu(J-1)+\frac K2(J-1)^2=\frac K2(J-\alpha)^2-\frac{\mu^2}{2K}")
        + "<p>This gives the two coordinates and their stiffnesses:</p>"
        + eq(r"C_r=r,\quad k_r=\mu;\qquad C_a=J-\alpha,\quad k_a=K", True)
        + eq(r"E_{\triangle}=A_0\left[\frac{\mu}{2}C_r^2+\frac K2 C_a^2\right]+\mathrm{constant}")
        + "<p>The two multipliers are <i>λᵣ</i> and <i>λₐ</i>. The weight in the common derivation is "
        "<i>w = A₀</i>. The norm coordinate is <i>r</i> itself: subtracting √2 would change the material energy.</p>",
    ),
    section(
        4,
        "triangle-alm",
        "Triangle elasticity: eliminate the auxiliaries and obtain forces",
        "<p>Apply the scalar derivation separately to stretch and area:</p>"
        + eq(r"K_{\mathrm{eff},r}=\frac{\mu\rho_r}{\mu+\rho_r},\quad s_r=\frac{\mu}{\mu+\rho_r}")
        + eq(r"K_{\mathrm{eff},a}=\frac{K\rho_a}{K+\rho_a},\quad s_a=\frac{K}{K+\rho_a}")
        + eq(
            r"z_r^*=\frac{K_{\mathrm{eff},r}}{\mu}r+\frac{s_r}{\mu}\lambda_r,\qquad z_a^*=\frac{K_{\mathrm{eff},a}}{K}C_a+\frac{s_a}{K}\lambda_a"
        )
        + "<p>After eliminating both auxiliaries, the position-dependent part of the triangle energy is:</p>"
        + eq(
            r"\widehat E_{\triangle}=A_0\left[\frac{K_{\mathrm{eff},r}}2 r^2+s_r\lambda_r r+\frac{K_{\mathrm{eff},a}}2 C_a^2+s_a\lambda_a C_a\right]+\mathrm{constant}",
            True,
        )
        + "<p>P is the surface first Piola&ndash;Kirchhoff stress: the derivative of this reduced ALM energy per rest area with respect to F. "
        "Use the two geometry derivatives:</p>"
        + eq(r"\frac{\partial r}{\partial F}=\frac F r,\qquad G=\frac{\partial J}{\partial F}=JF(F^TF)^{-1}")
        + "<p>The latter identity assumes a nondegenerate triangle; the code evaluates the equivalent "
        "column formula without taking an inverse. The chain rule gives:</p>"
        + eq(
            r"P=\left(K_{\mathrm{eff},r}r+s_r\lambda_r\right)\frac F r+\left(K_{\mathrm{eff},a}C_a+s_a\lambda_a\right)G",
            True,
        )
        + "<p>To turn this 3 by 2 matrix into a vertex force, let b₁ and b₂ be the transposed rows "
        "of Dₘ⁻¹ and b₀ = &minus;b₁ &minus;b₂. Moving vertex v changes F by δxᵥ bᵥᵀ, so:</p>"
        + eq(r"\mathbf f_v=-A_0 P\mathbf b_v")
        + "<p>After the complete vertex sweep, update the two scalar multipliers:</p>"
        + eq(r"\lambda_r^{n+1}=K_{\mathrm{eff},r}r(F^{n+1})+s_r\lambda_r^n", True)
        + eq(r"\lambda_a^{n+1}=K_{\mathrm{eff},a}C_a(F^{n+1})+s_a\lambda_a^n", True)
        + "<p>At a fixed point, λᵣ = μr and λₐ = K Cₐ. Therefore P = μF + K Cₐ G, the original material law. "
        "At rest the stretch and area stresses cancel. The stored scalar histories remain unchanged under rigid rotation.</p>"
        + "<details><summary>Exact stretch curvature and the implemented Hessian</summary><p>For a flattened F, let n = vec(F)/r. "
        "The exact stretch Hessian per rest area, with λᵣ fixed, is:</p>"
        + eq(r"H_{F,r}=K_{\mathrm{eff},r}nn^T+\frac{K_{\mathrm{eff},r}r+s_r\lambda_r}{r}(I-nn^T)")
        + "<p>Only the radial coefficient is K_eff,r. The transverse coefficient contains the multiplier. "
        "At a constitutive fixed point it equals μ. The code keeps this norm-curvature term. "
        "Its area contribution uses the existing positive-semidefinite projection of each vertex block.</p></details>",
    ),
    section(
        5,
        "tet-energy",
        "Tetrahedral elasticity: scalar stretch and pressure coordinates",
        "<p><b>Implemented as the default when particle ALM is enabled.</b> Like the triangle, a tet "
        "uses one scalar norm-stretch multiplier and one scalar pressure multiplier. Its deformation "
        "gradient is 3 by 3, and its volume ratio is signed:</p>"
        + eq(r"F=[x_1-x_0,\;x_2-x_0,\;x_3-x_0]D_m^{-1},\qquad r=\|F\|_F,\qquad J=\det F")
        + "<p>Use K = λ<sub>mat</sub> + μ and &alpha; = 1 + μ/K. The original stable quadratic-volume energy is:</p>"
        + eq(r"E_{\mathrm{tet}}=V_0\left[\frac{\mu}{2}(r^2-3)-\mu(J-1)+\frac K2(J-1)^2\right]")
        + "<p>Complete the same square as for the triangle:</p>"
        + eq(r"E_{\mathrm{tet}}=V_0\left[\frac\mu2r^2+\frac K2C_p^2\right]+\mathrm{constant},\qquad C_p=J-\alpha")
        + eq(r"C_r=r,\quad k_r=\mu;\qquad C_p=\det(F)-\alpha,\quad k_p=K", True)
        + "<p>The scalar histories are λᵣ and λₚ. The weight in the common derivation is w = V₀. "
        "As with triangles, the norm coordinate has no rest-norm subtraction. At rest r = √3, "
        "λᵣ = μ√3 and λₚ = &minus;μ; their contributions cancel in the total stress.</p>"
        + "<p>Introduce one scalar auxiliary per coordinate:</p>"
        + eq(r"\mathcal L_r=V_0\left[\frac\mu2z_r^2+\lambda_r(r-z_r)+\frac{\rho_r}{2}(r-z_r)^2\right]")
        + eq(r"\mathcal L_p=V_0\left[\frac K2z_p^2+\lambda_p(C_p-z_p)+\frac{\rho_p}{2}(C_p-z_p)^2\right]")
        + "<p>K is the pressure-row stiffness, not the three-dimensional bulk modulus. The scalar norm "
        "implementation is distinct from the separately explored SVD/stretch-tensor method.</p>",
    ),
    section(
        6,
        "tet-alm",
        "Tetrahedral elasticity: scalar ALM forces, Hessian, and updates",
        "<p>Use the same scalar coefficient definitions as for triangle stretch and area:</p>"
        + eq(r"K_{\mathrm{eff},r}=\frac{\mu\rho_r}{\mu+\rho_r},\quad s_r=\frac\mu{\mu+\rho_r}")
        + eq(r"K_{\mathrm{eff},p}=\frac{K\rho_p}{K+\rho_p},\quad s_p=\frac K{K+\rho_p}")
        + "<p>Eliminate the two scalar auxiliaries:</p>"
        + eq(
            r"z_r^*=\frac{K_{\mathrm{eff},r}}\mu r+\frac{s_r}\mu\lambda_r,\qquad z_p^*=\frac{K_{\mathrm{eff},p}}K C_p+\frac{s_p}K\lambda_p"
        )
        + eq(
            r"\widehat E_{\mathrm{tet}}=V_0\left[\frac{K_{\mathrm{eff},r}}2r^2+s_r\lambda_r r+\frac{K_{\mathrm{eff},p}}2C_p^2+s_p\lambda_p C_p\right]+\mathrm{constant}"
        )
        + "<p>P is the first Piola&ndash;Kirchhoff stress of the reduced ALM energy: its derivative per rest "
        "volume with respect to F. The geometry derivatives are:</p>"
        + eq(r"\frac{\partial r}{\partial F}=\frac F r,\qquad \frac{\partial J}{\partial F}=\operatorname{cof}(F)")
        + eq(
            r"P=\left(K_{\mathrm{eff},r}r+s_r\lambda_r\right)\frac F r+\left(K_{\mathrm{eff},p}C_p+s_p\lambda_p\right)\operatorname{cof}(F)",
            True,
        )
        + "<p>As for triangles, let b₁, b₂, b₃ be transposed rows of Dₘ⁻¹ and b₀ their negative sum. Then:</p>"
        + eq(r"\mathbf f_v=-V_0P\mathbf b_v")
        + "<p>The two scalar updates at the accepted positions are:</p>"
        + eq(r"\lambda_r^{n+1}=K_{\mathrm{eff},r}r(F^{n+1})+s_r\lambda_r^n", True)
        + eq(r"\lambda_p^{n+1}=K_{\mathrm{eff},p}C_p(F^{n+1})+s_p\lambda_p^n", True)
        + "<p>At a fixed point λᵣ = μr and λₚ = K Cₚ, recovering P = μF + K Cₚ cof(F). "
        "Both histories are unchanged by a rigid rotation; the force directions follow the current F.</p>"
        + '<p class="note"><b>The norm geometric Hessian is included.</b> K_eff,r is not the entire '
        "positional stiffness. The multiplier contributes to curvature through the second derivative of r.</p>"
        + "<details><summary>The exact tet vertex block used by the scalar split</summary>"
        + eq(r"g_{rv}=\frac{F\mathbf b_v}{r},\qquad g_{pv}=\operatorname{cof}(F)\mathbf b_v")
        + eq(r"\nabla_{x_v}^2r=\frac{\|\mathbf b_v\|^2}{r}I-\frac{(F\mathbf b_v)(F\mathbf b_v)^T}{r^3}")
        + eq(
            r"H_v=V_0\left[K_{\mathrm{eff},r}g_{rv}g_{rv}^T+\left(K_{\mathrm{eff},r}r+s_r\lambda_r\right)\nabla_{x_v}^2r+K_{\mathrm{eff},p}g_{pv}g_{pv}^T\right]"
        )
        + "<p>The determinant is affine in one vertex position when all other vertices are fixed, "
        "so its geometric second derivative contributes zero to this diagonal block. This cancellation "
        "does not apply to the norm row, and determinant cross-vertex terms are not all zero.</p></details>",
    ),
    section(
        7,
        "tet-norm",
        "Tet options and the historical nine-component formulation",
        "<p>Particle ALM is opt-in. With it enabled, the default tet mode now uses scalar norm stretch "
        "and pressure. The existing experimental flag can explicitly select pressure-only behavior:</p>"
        + "<pre><code>SolverVBD(model, particle_elasticity_alm=True)\n\n# Explicit comparison: ordinary stretch plus ALM pressure\nSolverVBD(model, particle_elasticity_alm=True,\n          particle_elasticity_alm_deviatoric=False)</code></pre>"
        + "<p>In pressure-only mode, only λₚ is updated. The stretch part stays ordinary:</p>"
        + eq(r"P=\mu F+\left(K_{\mathrm{eff},p}C_p+s_p\lambda_p\right)\operatorname{cof}(F)")
        + '<p class="note">The experimental <code>particle_elasticity_alm_deviatoric</code> flag now '
        "defaults to True and means scalar norm stretch. The earlier True mode stored a 3 by 3 matrix; "
        "that implementation has been replaced. The term is not strictly volume-preserving despite the flag's historical name.</p>"
        + "<details><summary>Historical matrix formulation: comparison only</summary>"
        "<p>The previous implementation treated F itself as nine scalar coordinates and retained "
        "a 3 by 3 multiplier Λ_F. With k_F = μ and the corresponding K_eff,F and s_F, its formulas were:</p>"
        + eq(r"Z_F^*=\frac{K_{\mathrm{eff},F}}\mu F+\frac{s_F}\mu\Lambda_F")
        + eq(r"P=K_{\mathrm{eff},F}F+s_F\Lambda_F+\left(K_{\mathrm{eff},p}C_p+s_p\lambda_p\right)\operatorname{cof}(F)")
        + eq(r"\Lambda_F^{n+1}=K_{\mathrm{eff},F}F^{n+1}+s_F\Lambda_F^n")
        + "<p>At a constitutive fixed point the matrix and scalar formulations recover the same "
        "original material. Their iteration Hessians differ. The matrix stretch coordinate is linear "
        "in F, whereas the scalar norm has the geometric Hessian shown above.</p>"
        + "<p>For a rigid rotation R of a rest tet with old matrix history Λ_F = μI and λₚ = &minus;μ, "
        "the historical matrix solve produced:</p>"
        + eq(r"P(R)=s_F\mu(I-R)")
        + "<p>This was the finite-iteration rotation artifact. The scalar history removes that "
        "stored world-space direction. Historical benchmark results still describe the old implementation.</p></details>",
    ),
    section(
        8,
        "bending-energy",
        "Dihedral bending: identify the angle coordinate",
        "<p><b>Implemented.</b> Two adjacent triangles form a hinge. Vertices x₀ and x₁ are opposite "
        "the shared edge (x₂, x₃). Let θ(x) be the signed dihedral angle, θ₀ its rest value, and ℓ₀ the rest edge length.</p>"
        + eq(r"C_b(x)=\theta(x)-\theta_0,\qquad k_b=\kappa_b\ell_0")
        + eq(r"E_b=\frac{k_b}{2}C_b(x)^2", True)
        + "<p>Here κ_b is the authored bending coefficient. The rest length is already included in k_b, "
        "so the common row weight is w = 1. Introduce one scalar auxiliary z_b and one multiplier λ_b:</p>"
        + eq(r"\mathcal L_b=\frac{k_b}{2}z_b^2+\lambda_b(C_b-z_b)+\frac{\rho_b}{2}(C_b-z_b)^2")
        + "<p>The coordinate is the signed angle difference used by the elastic kernel. The implementation "
        "wraps the angle increment used for damping, but does not wrap this elastic rest-angle difference.</p>"
        + "<details><summary>How the signed angle is defined</summary>"
        + eq(r"e=x_3-x_2,\quad n_0=(x_2-x_0)\times(x_3-x_0),\quad n_1=(x_3-x_1)\times(x_2-x_1)")
        + eq(r"\theta=\operatorname{atan2}\left[(\hat n_0\times\hat n_1)\cdot\hat e,\;\hat n_0\cdot\hat n_1\right]")
        + "<p>A hat denotes normalization. The derivation assumes nondegenerate faces and a continuous "
        "local angle branch; the implementation guards degenerate geometry.</p></details>",
    ),
    section(
        9,
        "bending-alm",
        "Dihedral bending: eliminate z and obtain the four vertex forces",
        "<p>Define the bending coefficients and eliminate the auxiliary:</p>"
        + eq(r"K_{\mathrm{eff},b}=\frac{k_b\rho_b}{k_b+\rho_b},\qquad s_b=\frac{k_b}{k_b+\rho_b}")
        + eq(r"z_b^*=\frac{K_{\mathrm{eff},b}}{k_b}C_b+\frac{s_b}{k_b}\lambda_b")
        + "<p>The reduced bending energy and its derivative with respect to the angle are:</p>"
        + eq(
            r"\widehat E_b=\frac{K_{\mathrm{eff},b}}2(\theta-\theta_0)^2+s_b\lambda_b(\theta-\theta_0)+\mathrm{constant}"
        )
        + eq(r"\frac{\partial\widehat E_b}{\partial\theta}=K_{\mathrm{eff},b}(\theta-\theta_0)+s_b\lambda_b", True)
        + "<p>Each of the four vertex forces follows from the angle gradient:</p>"
        + eq(r"\mathbf f_v=-\left[K_{\mathrm{eff},b}(\theta-\theta_0)+s_b\lambda_b\right]\nabla_{x_v}\theta", True)
        + "<p>At the accepted positions, update the one hinge multiplier:</p>"
        + eq(r"\lambda_b^{n+1}=K_{\mathrm{eff},b}\left[\theta(x^{n+1})-\theta_0\right]+s_b\lambda_b^n", True)
        + "<p>At a fixed point λ_b = k_b(θ &minus; θ₀), recovering the original bending force.</p>"
        + "<details><summary>Explicit angle gradients for the four vertices</summary>"
        "<p>Using the unnormalized normals and edge from step 8:</p>"
        + eq(r"g_0=-\frac{\|e\|}{\|n_0\|^2}n_0,\qquad g_1=-\frac{\|e\|}{\|n_1\|^2}n_1")
        + eq(r"g_2=-\frac{[(x_3-x_0)\cdot e]g_0+[(x_3-x_1)\cdot e]g_1}{\|e\|^2},\qquad g_3=-g_0-g_1-g_2")
        + "<p>Here gᵥ = ∇<sub>xᵥ</sub>θ. The code computes equivalent derivatives through normalized "
        "normals and atan2.</p></details>"
        + "<details><summary>Exact curvature versus the implemented bending Hessian</summary>"
        + eq(
            r"H_v=K_{\mathrm{eff},b}g_vg_v^T+\left[K_{\mathrm{eff},b}(\theta-\theta_0)+s_b\lambda_b\right]\nabla_{x_v}^2\theta"
        )
        + "<p>The current solver keeps only the positive-semidefinite outer-product approximation:</p>"
        + eq(r"H_v^{\mathrm{solver}}=K_{\mathrm{eff},b}g_vg_v^T")
        + "<p>The omitted angle-curvature term is part of the exact reduced energy Hessian. "
        "The force and multiplier update above are still the implemented expressions.</p></details>",
    ),
    section(
        10,
        "solve",
        "How these element formulas enter one VBD substep",
        "<p>The total position objective contains inertia, all reduced elastic element energies, "
        "and the existing contact terms. For the particle solve, the inertial target is x̂:</p>"
        + eq(r"\Phi(x)=\sum_{v\;\mathrm{free}}\frac{m_v}{2\Delta t^2}\|x_v-\hat x_v\|^2")
        + "<ol><li>At the incoming pose, compute the penalties. On activation or reset, initialize "
        "each scalar history to kᵢ Cᵢ. The scalar tet stretch history is initialized to μ‖F‖.</li>"
        "<li>Compute K_eff,i and sᵢ from the material stiffness and numerical penalty. Keep penalties "
        "and multipliers fixed during the complete vertex-color sweep.</li>"
        "<li>For each vertex, assemble inertia, incident element forces and Hessian blocks, damping, "
        "and contacts. Take its local VBD update and apply the existing per-color DAT truncation.</li>"
        "<li>Once all colors are finished, evaluate the accepted coordinates and apply "
        "<b>λᵢ ← K_eff,i Cᵢ + sᵢ λᵢ</b> to every active scalar row, using the accepted geometry.</li>"
        "<li>Repeat the sweep and multiplier update for the configured number of iterations. "
        "Retain multiplier history across substeps.</li></ol>"
        + "<p>The current triangle, tet norm-stretch, and bending policy is &rho;ᵢ = max(&rho;<sub>inertia,i</sub>, 9kᵢ). "
        "Tet pressure penalties remain inertia-based without that floor. Damping remains separate "
        "from this elasticity ALM split. Contact ALM is not implemented by these formulas.</p>"
        + '<p class="note">The derivation assumes active, positive-stiffness rows and valid geometry. '
        "The implementation also handles disabled rows, degeneracy, float32 bounds, and reset state. "
        "All auxiliary variables are eliminated analytically.</p>",
    ),
]

prefix = page.split("<header>", 1)[0].replace(
    "Triangle elasticity ALM — readable equations", "Triangle, tet, and bending compliant ALM with K_eff and s"
)
extra_css = "<style>summary{cursor:pointer;font-weight:650;color:#096b71}details{margin-top:20px;padding:16px;background:#f6f8fa;border-radius:10px}nav{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}nav a{padding:5px 12px;background:white;border:1px solid #c4d4dc;border-radius:7px;text-decoration:none}section{scroll-margin-top:20px}li{margin:10px 0}code{overflow-wrap:anywhere}pre{overflow-x:auto;padding:12px;background:#f6f8fa;border-radius:8px}</style>"
header = """<header><div class="eyebrow">Derivation matched to the implementation · local / offline</div>
<h1>Triangle, tet, and bending compliant ALM</h1>
<p>The spring derivation carried through each element energy, using K_eff and s throughout. Every force coefficient is written explicitly; there is no extra force shorthand.</p>
<div class="symbols"><span><b>kᵢ</b> = material row stiffness</span><span><b>K_eff,i</b> = effective stiffness</span><span><b>sᵢ</b> = multiplier weight</span><span><b>λᵢ</b> = scalar multiplier</span></div>
<nav aria-label="Derivation sections"><a href="#shared">Shared derivation</a><a href="#triangle-energy">Triangles</a><a href="#tet-energy">Tets</a><a href="#tet-norm">Tet options / old matrix split</a><a href="#bending-energy">Bending</a><a href="#solve">Solver loop</a></nav>
<p style="font-size:15px">Triangles: two scalar rows implemented. Tets: scalar norm stretch plus pressure by default when ALM is enabled; pressure-only comparison remains available. Bending: one scalar row implemented.</p></header>"""
footer = """<footer><p><a href="spring.html">Spring derivation</a> · <a href="../alm-elasticity-code-walkthrough/review.html">Full code walkthrough</a> · <a href="../alm-bag-wiggle/results-floor-sweep/index.html">Residual experiment</a> · <a href="LICENSE_STIX">STIX font license</a></p>
<p>Source: <a href="../../newton/_src/solvers/vbd/particle_alm_kernels.py">ALM state, preparation, and updates</a>; <a href="../../newton/_src/solvers/vbd/particle_vbd_kernels.py">element forces and Hessians</a>; <a href="../../newton/_src/solvers/vbd/solver_vbd.py">VBD integration</a>.</p>
<p>Equations use the current quadratic area/volume material, not the logarithmic Neo-Hookean energy. Embedded math font; no network required.</p><button onclick="window.print()">Print / save as PDF</button></footer></main></body></html>"""
output = Path(__file__).with_name("elements.html")
output.write_text(prefix + extra_css + header + "\n".join(sections) + footer)
print(output)
