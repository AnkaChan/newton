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
        + "<p>Define P as the derivative of this energy per rest area with respect to F. "
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
        "Tetrahedral elasticity: the implemented matrix and pressure coordinates",
        "<p>For a tet, F is 3 by 3 and J is the <b>signed</b> volume ratio:</p>"
        + eq(r"F=[x_1-x_0,\;x_2-x_0,\;x_3-x_0]D_m^{-1},\qquad J=\det F")
        + "<p>Use the same definitions K = λ<sub>mat</sub> + μ and &alpha; = 1 + μ/K. "
        "The stable quadratic-volume material used by the code is:</p>"
        + eq(r"E_{\mathrm{tet}}=V_0\left[\frac{\mu}{2}(\|F\|_F^2-3)-\mu(J-1)+\frac K2(J-1)^2\right]")
        + "<p>Complete the same square to obtain:</p>"
        + eq(
            r"E_{\mathrm{tet}}=V_0\left[\frac\mu2\|F\|_F^2+\frac K2 C_p^2\right]+\mathrm{constant},\qquad C_p=J-\alpha"
        )
        + "<p><b>The optional full matrix mode</b> treats F itself as a matrix coordinate. "
        "Its auxiliary Z_F and multiplier Λ_F each have nine components. The pressure coordinate "
        "has one scalar auxiliary zₚ and multiplier λₚ:</p>"
        + eq(r"C_F=F,\quad k_F=\mu;\qquad C_p=J-\alpha,\quad k_p=K", True)
        + "<p>The weight is w = V₀. The full augmented energy is the sum of:</p>"
        + eq(r"\mathcal L_F=V_0\left[\frac\mu2\|Z_F\|_F^2+\Lambda_F:(F-Z_F)+\frac{\rho_F}{2}\|F-Z_F\|_F^2\right]")
        + eq(r"\mathcal L_p=V_0\left[\frac K2 z_p^2+\lambda_p(C_p-z_p)+\frac{\rho_p}{2}(C_p-z_p)^2\right]")
        + "<p>The colon is the sum of entrywise products. These are nine matrix multipliers plus one pressure "
        "multiplier, not additional vertex position degrees of freedom. K is a row stiffness, not the three-dimensional bulk modulus.</p>",
    ),
    section(
        6,
        "tet-alm",
        "Tetrahedral elasticity: position solve and multiplier updates",
        "<p>Define one pair of coefficients for the matrix block and another for pressure:</p>"
        + eq(r"K_{\mathrm{eff},F}=\frac{\mu\rho_F}{\mu+\rho_F},\quad s_F=\frac\mu{\mu+\rho_F}")
        + eq(r"K_{\mathrm{eff},p}=\frac{K\rho_p}{K+\rho_p},\quad s_p=\frac K{K+\rho_p}")
        + "<p>Eliminating the auxiliaries gives:</p>"
        + eq(
            r"Z_F^*=\frac{K_{\mathrm{eff},F}}\mu F+\frac{s_F}\mu\Lambda_F,\qquad z_p^*=\frac{K_{\mathrm{eff},p}}K C_p+\frac{s_p}K\lambda_p"
        )
        + eq(
            r"\widehat E_{\mathrm{tet}}=V_0\left[\frac{K_{\mathrm{eff},F}}2\|F\|_F^2+s_F\Lambda_F:F+\frac{K_{\mathrm{eff},p}}2 C_p^2+s_p\lambda_p C_p\right]+\mathrm{constant}"
        )
        + "<p>The determinant derivative is cof(F), the cofactor matrix. Differentiating the reduced energy per rest volume gives:</p>"
        + eq(
            r"P=K_{\mathrm{eff},F}F+s_F\Lambda_F+\left(K_{\mathrm{eff},p}C_p+s_p\lambda_p\right)\operatorname{cof}(F)",
            True,
        )
        + "<p>As for triangles, let b₁, b₂, b₃ be transposed rows of Dₘ⁻¹ and b₀ their negative sum. Then:</p>"
        + eq(r"\mathbf f_v=-V_0P\mathbf b_v")
        + "<p>Update the matrix and scalar histories at the accepted positions:</p>"
        + eq(r"\Lambda_F^{n+1}=K_{\mathrm{eff},F}F^{n+1}+s_F\Lambda_F^n", True)
        + eq(r"\lambda_p^{n+1}=K_{\mathrm{eff},p}C_p(F^{n+1})+s_p\lambda_p^n", True)
        + "<p>At a fixed point Λ_F = μF and λₚ = K Cₚ, giving the original stress μF + K Cₚ cof(F).</p>"
        + '<p class="note"><b>Current default: pressure-only ALM.</b> The μ stretch term stays ordinary. '
        "Use the pressure update above, but use this stress in the position solve:</p>"
        + eq(r"P=\mu F+\left(K_{\mathrm{eff},p}C_p+s_p\lambda_p\right)\operatorname{cof}(F)", True)
        + "<details><summary>The tet vertex Hessian and the matrix-history rotation limitation</summary>"
        "<p>For the full matrix mode, the exact diagonal vertex block is:</p>"
        + eq(
            r"H_v=V_0\left[K_{\mathrm{eff},F}\|\mathbf b_v\|^2I+K_{\mathrm{eff},p}\mathbf g_{pv}\mathbf g_{pv}^T\right],\qquad \mathbf g_{pv}=\operatorname{cof}(F)\mathbf b_v"
        )
        + "<p>For pressure-only mode, replace K_eff,F by μ. The determinant is affine in one vertex "
        "position when the others are fixed, so its second derivative contributes zero to this diagonal block. "
        "Cross-vertex Hessian terms are not all zero.</p>"
        + "<p>The implemented matrix history can retain an old world-space orientation. For a rigid "
        "rotation R of a rest tet with unrotated history Λ_F = μI and λₚ = &minus;μ, the full-mode stress is:</p>"
        + eq(r"P(R)=s_F\mu(I-R)")
        + "<p>This is the known finite-iteration rotation artifact. The original material energy remains "
        "rotation-invariant; the stale matrix-history solve is the source of the artifact.</p></details>",
    ),
    section(
        7,
        "tet-norm",
        "Scalar-norm tet alternative: the same two-row idea as triangles",
        '<p class="note"><b>Derived alternative; not integrated in the current tet solver.</b> '
        "The current triangle implementation uses this type of scalar invariant split. "
        "It is distinct from the separately explored SVD/stretch-tensor proposal.</p>"
        + "<p>Since the tet stretch energy depends only on the squared Frobenius norm, we can instead choose:</p>"
        + eq(r"C_r=r=\|F\|_F,\quad k_r=\mu;\qquad C_p=\det(F)-\alpha,\quad k_p=K")
        + "<p>The scalar auxiliaries are eliminated exactly as for a triangle. The position-dependent energy becomes:</p>"
        + eq(
            r"\widehat E=V_0\left[\frac{K_{\mathrm{eff},r}}2r^2+s_r\lambda_r r+\frac{K_{\mathrm{eff},p}}2C_p^2+s_p\lambda_p C_p\right]+\mathrm{constant}"
        )
        + "<p>Using ∂r/∂F = F/r and ∂J/∂F = cof(F), the stress is:</p>"
        + eq(
            r"P=\left(K_{\mathrm{eff},r}r+s_r\lambda_r\right)\frac F r+\left(K_{\mathrm{eff},p}C_p+s_p\lambda_p\right)\operatorname{cof}(F)",
            True,
        )
        + "<p>The two scalar updates are:</p>"
        + eq(r"\lambda_r^{n+1}=K_{\mathrm{eff},r}r(F^{n+1})+s_r\lambda_r^n")
        + eq(r"\lambda_p^{n+1}=K_{\mathrm{eff},p}C_p(F^{n+1})+s_p\lambda_p^n")
        + "<p>This represents the same original energy and has rotation-invariant scalar histories. "
        "Its finite-iteration curvature differs from the matrix split: the norm introduces the geometric "
        "curvature shown in the triangle section. Exact energy equivalence does not imply identical or faster convergence.</p>",
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
        "each scalar history to kᵢ Cᵢ; initialize matrix history to μF if that mode is enabled.</li>"
        "<li>Compute K_eff,i and sᵢ from the material stiffness and numerical penalty. Keep penalties "
        "and multipliers fixed during the complete vertex-color sweep.</li>"
        "<li>For each vertex, assemble inertia, incident element forces and Hessian blocks, damping, "
        "and contacts. Take its local VBD update and apply the existing per-color DAT truncation.</li>"
        "<li>Once all colors are finished, evaluate the accepted coordinates and apply "
        "<b>λᵢ ← K_eff,i Cᵢ + sᵢ λᵢ</b> to every active scalar row, or the corresponding matrix expression.</li>"
        "<li>Repeat the sweep and multiplier update for the configured number of iterations. "
        "Retain multiplier history across substeps.</li></ol>"
        + "<p>The current triangle and bending policy is &rho;ᵢ = max(&rho;<sub>inertia,i</sub>, 9kᵢ). "
        "Tet penalties are inertia-based and do not use that cloth floor. Damping remains separate "
        "from this elasticity ALM split. Contact ALM is not implemented by these formulas.</p>"
        + '<p class="note">The derivation assumes active, positive-stiffness rows and valid geometry. '
        "The implementation also handles disabled rows, degeneracy, float32 bounds, and reset state. "
        "All auxiliary variables are eliminated analytically.</p>",
    ),
]

prefix = page.split("<header>", 1)[0].replace(
    "Triangle elasticity ALM — readable equations", "Triangle, tet, and bending compliant ALM with K_eff and s"
)
extra_css = "<style>summary{cursor:pointer;font-weight:650;color:#096b71}details{margin-top:20px;padding:16px;background:#f6f8fa;border-radius:10px}nav{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}nav a{padding:5px 12px;background:white;border:1px solid #c4d4dc;border-radius:7px;text-decoration:none}section{scroll-margin-top:20px}li{margin:10px 0}code{overflow-wrap:anywhere}</style>"
header = """<header><div class="eyebrow">Derivation matched to the implementation · local / offline</div>
<h1>Triangle, tet, and bending compliant ALM</h1>
<p>The spring derivation carried through each element energy, using K_eff and s throughout. Every force coefficient is written explicitly; there is no extra force shorthand.</p>
<div class="symbols"><span><b>kᵢ</b> = material row stiffness</span><span><b>K_eff,i</b> = effective stiffness</span><span><b>sᵢ</b> = multiplier weight</span><span><b>λᵢ</b> = scalar multiplier</span><span><b>Λ_F</b> = matrix multiplier</span></div>
<nav aria-label="Derivation sections"><a href="#shared">Shared derivation</a><a href="#triangle-energy">Triangles</a><a href="#tet-energy">Tets</a><a href="#tet-norm">Scalar tet alternative</a><a href="#bending-energy">Bending</a><a href="#solve">Solver loop</a></nav>
<p style="font-size:15px">Triangles: two scalar rows implemented. Tets: pressure-only default, optional 9 + 1 matrix mode; scalar-norm alternative derived separately. Bending: one scalar row implemented.</p></header>"""
footer = """<footer><p><a href="spring.html">Spring derivation</a> · <a href="../alm-elasticity-code-walkthrough/review.html">Full code walkthrough</a> · <a href="../alm-bag-wiggle/results-floor-sweep/index.html">Residual experiment</a> · <a href="LICENSE_STIX">STIX font license</a></p>
<p>Source: <a href="../../newton/_src/solvers/vbd/particle_alm_kernels.py">ALM state, preparation, and updates</a>; <a href="../../newton/_src/solvers/vbd/particle_vbd_kernels.py">element forces and Hessians</a>; <a href="../../newton/_src/solvers/vbd/solver_vbd.py">VBD integration</a>.</p>
<p>Equations use the current quadratic area/volume material, not the logarithmic Neo-Hookean energy. Embedded math font; no network required.</p><button onclick="window.print()">Print / save as PDF</button></footer></main></body></html>"""
output = Path(__file__).with_name("elements.html")
output.write_text(prefix + extra_css + header + "\n".join(sections) + footer)
print(output)
