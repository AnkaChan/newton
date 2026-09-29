# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the spring derivation using K_eff and s throughout."""

from pathlib import Path

from build import card, eq, page

sections = [
    card(
        1,
        "Spring energy and the two ALM coefficients",
        "<p>Let <i>x</i> be the endpoint separation, <i>L₀</i> the rest length, and <i>k &gt; 0</i> "
        "the physical spring stiffness. The geometric extension and original problem are:</p>"
        + eq(r"C(x)=x-L_0,\qquad \min_x\left[\Phi(x)+\frac{k}{2}C(x)^2\right]")
        + "<p><i>Φ(x)</i> contains inertia and any other position-dependent terms. Choose a positive "
        "numerical penalty <i>&rho;</i> and define the two coefficients we will use throughout:</p>"
        + eq(r"K_{\mathrm{eff}}=\frac{k\rho}{k+\rho},\qquad s=\frac{k}{k+\rho}", True)
        + eq(r"K_{\mathrm{eff}}=\rho s=k(1-s)")
        + "<p><b>K_eff</b> has stiffness units. <b>s</b> is dimensionless and lies between zero and one. "
        "These abbreviations will emerge as the position stiffness and multiplier weight when we "
        "eliminate the auxiliary extension below.</p>",
    ),
    card(
        2,
        "Introduce the auxiliary extension and apply ALM",
        "<p>Give the material a separate extension <i>z</i>, while requiring it to equal the extension "
        "measured from the positions:</p>"
        + eq(r"\min_{x,z}\left[\Phi(x)+\frac{k}{2}z^2\right],\qquad C(x)-z=0")
        + "<p>The multiplier <i>λ</i> has force units. Add its equality term and the numerical penalty:</p>"
        + eq(r"\mathcal{L}_{\rho}=\Phi(x)+\frac{k}{2}z^2+\lambda(C(x)-z)+\frac{\rho}{2}(C(x)-z)^2", True)
        + "<p>The equality is between the two copies of extension. It allows the physical spring to "
        "stretch; it does not require <i>C(x) = 0</i>.</p>",
    ),
    card(
        3,
        "Eliminate z to obtain the spring force",
        "<p>With the positions and multiplier fixed, minimize over <i>z</i>:</p>"
        + eq(r"kz-\lambda-\rho(C-z)=0")
        + eq(r"z^*=\frac{\lambda+\rho C}{k+\rho}")
        + "<p>Now use the definitions of <b>K_eff</b> and <b>s</b>:</p>"
        + eq(r"z^*=\frac{K_{\mathrm{eff}}}{k}C+\frac{s}{k}\lambda=(1-s)C+s\frac{\lambda}{k}", True)
        + "<p>The auxiliary extension is a weighted average of the geometric extension "
        "<i>C</i> and the extension <i>λ/k</i> implied by the stored stress.</p>"
        + "<p>The derivative of the spring's material energy with respect to its extension is "
        "<i>kz*</i>. Substituting the expression above gives the force coefficient:</p>"
        + eq(r"kz^*=K_{\mathrm{eff}}C+s\lambda", True)
        + "<p>This coefficient combines the current extension <i>C</i> with the stored multiplier "
        "<i>λ</i>. In step 4 we apply the chain rule to obtain the force on the endpoint. "
        "The implementation evaluates <b>K_eff C + s λ</b> directly; it does not store an "
        "auxiliary <i>z</i> array.</p>",
    ),
    card(
        4,
        "Solve positions using K_eff and s",
        "<p>Substitute the eliminated <i>z*</i> back into the augmented Lagrangian. Written entirely "
        "with the two coefficients, the result is:</p>"
        + eq(
            r"\widehat{\mathcal{L}}_{\rho}(x,\lambda)=\Phi(x)+\frac{K_{\mathrm{eff}}}{2}C(x)^2+s\lambda C(x)-\frac{s}{2k}\lambda^2"
        )
        + "<p>Hold <i>λⁿ</i> fixed during the position solve. The last term is constant with respect "
        "to position, so the primal update is:</p>"
        + eq(
            r"x^{n+1}\approx\operatorname{argmin}_x\left[\Phi(x)+\frac{K_{\mathrm{eff}}}{2}C(x)^2+s\lambda^n C(x)\right]",
            True,
        )
        + "<p>Differentiation gives the spring force:</p>"
        + eq(r"f_{\mathrm{spring}}=-\left(K_{\mathrm{eff}}C+s\lambda^n\right)\frac{dC}{dx}")
        + "<p>For this linear extension, <i>dC/dx = 1</i> and the spring contributes "
        "<b>K_eff</b> to positional curvature. For example, let the inertia term be:</p>"
        + eq(r"\Phi(x)=\frac{h}{2}(x-\hat{x})^2,\qquad h=\frac{m}{\Delta t^2}")
        + "<p>The one-dimensional position solve then has the explicit solution:</p>"
        + eq(r"x^{n+1}=\frac{h\hat{x}+K_{\mathrm{eff}}L_0-s\lambda^n}{h+K_{\mathrm{eff}}}", True)
        + "<p>Here <i>x̂</i> is the inertial target. The actual VBD implementation makes an approximate "
        "position update through a vertex sweep, rather than solving the whole coupled system exactly.</p>",
    ),
    card(
        5,
        "Update the multiplier using the same K_eff and s",
        "<p>At the accepted position, apply ordinary ALM ascent to the equality mismatch:</p>"
        + eq(r"\lambda^{n+1}=\lambda^n+\rho\left[C(x^{n+1})-z^*(x^{n+1},\lambda^n)\right]")
        + "<p>From step 3, the mismatch is <i>s</i> times the difference between the geometric and "
        "stress-implied extensions:</p>"
        + eq(r"C-z^*=s\left(C-\frac{\lambda^n}{k}\right)")
        + "<p>Substitute this and use <i>&rho;s = K_eff</i> and <i>1 &minus; K_eff/k = s</i>:</p>"
        + eq(r"\lambda^{n+1}=\rho s\,C(x^{n+1})+\left(1-\frac{\rho s}{k}\right)\lambda^n")
        + eq(r"\lambda^{n+1}=K_{\mathrm{eff}}C(x^{n+1})+s\lambda^n", True)
        + '<p class="note"><b>The force evaluation and multiplier update use the same expression.</b> '
        "During the position solve, evaluate it at trial positions with the old multiplier fixed. "
        "After the sweep, evaluate it at the accepted positions and store the result as the new multiplier.</p>",
    ),
    card(
        6,
        "Why this is compliant ALM and recovers the original spring",
        "<p>The physical compliance is <i>a = 1/k</i>. In this notation the same coefficients are:</p>"
        + eq(r"s=\frac{1}{1+\rho a},\qquad K_{\mathrm{eff}}=\frac{\rho}{1+\rho a}")
        + "<p>Thus the multiplier update from step 5 is exactly the compliant ALM update:</p>"
        + eq(r"\lambda^{n+1}=K_{\mathrm{eff}}C(x^{n+1})+s\lambda^n=\frac{\rho C(x^{n+1})+\lambda^n}{1+\rho a}", True)
        + "<p>At convergence the multiplier stops changing. Using <i>K_eff = k(1 &minus; s)</i>:</p>"
        + eq(r"(1-s)\lambda=K_{\mathrm{eff}}C")
        + eq(r"\lambda=\frac{K_{\mathrm{eff}}}{1-s}C=kC,\qquad z=C")
        + "<p>The stress used by the position solve therefore becomes:</p>"
        + eq(r"K_{\mathrm{eff}}C+s(kC)=k(1-s)C+skC=kC", True)
        + "<p>The iteration uses <b>K_eff</b>, but the converged spring obeys the original physical "
        "stiffness <b>k</b>. In the zero-compliance limit, the multiplier update becomes hard-constraint ALM.</p>",
    ),
    card(
        7,
        "K_eff and s describe the tradeoff made by rho",
        "<p>Hold the extension fixed temporarily. Subtract the correct material stress <i>kC</i> "
        "from the multiplier update:</p>"
        + eq(r"\lambda^{n+1}-kC=s(\lambda^n-kC)", True)
        + "<p>So <b>s</b> is the fraction of the previous fixed-pose stress error retained by each "
        "update, while <b>K_eff = k(1 &minus; s)</b> is the stiffness seen by the position solve.</p>"
        + '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;text-align:left">'
        "<thead><tr><th>Penalty &rho;</th><th>K_eff</th><th>s</th><th>Stress error retained</th></tr></thead>"
        "<tbody><tr><td>9k</td><td>0.9k</td><td>0.1</td><td>10%</td></tr>"
        "<tr><td>k</td><td>0.5k</td><td>0.5</td><td>50%</td></tr></tbody></table></div>"
        + "<p>A smaller penalty gives a softer position solve and slower multiplier catch-up. "
        "These table values apply when the named floor determines the penalty; a larger inertia-based "
        "penalty gives different coefficients.</p>"
        + '<p class="note">The error factor above assumes fixed positions. It is not a convergence '
        "rate for the full coupled position solve.</p>",
    ),
    card(
        8,
        "Numerical example expressed in K_eff and s",
        "<p>Hold the spring at <i>C = 2 mm</i>. Choose <i>k = &rho; = 1000 N/m</i>, "
        "so the physical spring force should be <b>2 N</b>. The two coefficients are:</p>"
        + eq(r"K_{\mathrm{eff}}=500\;\mathrm{N/m},\qquad s=0.5")
        + "<p>Start with <i>λ⁰ = 0</i>. Each multiplier update is:</p>"
        + eq(r"\lambda^{n+1}=K_{\mathrm{eff}}C+s\lambda^n=1\;\mathrm{N}+0.5\lambda^n", True)
        + "<p>To recover the eliminated auxiliary, divide the force coefficient by the physical stiffness:</p>"
        + eq(r"z^*=\frac{K_{\mathrm{eff}}C+s\lambda^n}{k}=\frac{\lambda^{n+1}}{k}")
        + '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;text-align:left">'
        "<thead><tr><th>Update</th><th>Old λ (N)</th><th>K_eff C (N)</th><th>s λ (N)</th><th>New λ (N)</th><th>z* (mm)</th></tr></thead>"
        "<tbody><tr><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td></tr>"
        "<tr><td>2</td><td>1</td><td>1</td><td>0.5</td><td>1.5</td><td>1.5</td></tr>"
        "<tr><td>3</td><td>1.5</td><td>1</td><td>0.75</td><td>1.75</td><td>1.75</td></tr>"
        "<tr><td>4</td><td>1.75</td><td>1</td><td>0.875</td><td>1.875</td><td>1.875</td></tr>"
        "<tr><td>Limit</td><td>2</td><td>1</td><td>1</td><td>2</td><td>2</td></tr></tbody></table></div>"
        + "<p>The current geometry contributes 1 N through <b>K_eff C</b>. The retained multiplier "
        "contribution <b>s λ</b> builds toward the remaining 1 N. Together they recover the physical "
        "2 N force, and <i>z*</i> approaches the actual 2 mm extension.</p>"
        + '<p class="note">Zero initialization is only for this illustration. Our implementation '
        "initially seeds the multiplier with <i>kC</i> and subsequently retains history. "
        "During a real solve the positions also move.</p>",
    ),
]

prefix = page.split("<header>", 1)[0].replace(
    "Triangle elasticity ALM — readable equations", "Spring compliant ALM derived with K_eff and s"
)
header = """<header><div class="eyebrow">Complete derivation · local / offline</div>
<h1>Spring compliant ALM, derived with K_eff and s</h1>
<p>Follow the same two coefficients through auxiliary elimination, the position solve, the multiplier update, and recovery of the original material law.</p>
<div class="symbols"><span><b>K_eff</b> = effective position stiffness</span><span><b>s</b> = multiplier weight</span>
<span><b>k</b> = physical stiffness</span><span><b>&rho;</b> = numerical penalty</span><span><b>λ</b> = stored multiplier</span></div></header>"""
footer = """<footer><p><a href="index.html">Triangle ALM equation sheet</a> ·
<a href="../alm-elasticity-code-walkthrough/review.html">Code walkthrough</a> ·
<a href="LICENSE_STIX">STIX font license</a></p>
<p>Offline MathML with an embedded math font.</p><button onclick="window.print()">Print / save as PDF</button>
</footer></main></body></html>"""
output = Path(__file__).with_name("spring.html")
output.write_text(prefix + header + "\n".join(sections) + footer)
print(output)
