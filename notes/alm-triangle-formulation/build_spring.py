# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the spring derivation using the offline equation sheet's styles and font."""

from pathlib import Path

from build import card, eq, page

sections = [
    card(
        1,
        "Start with an ordinary elastic spring",
        "<p>Let <i>x</i> be the endpoint separation and <i>L₀</i> the rest length. "
        "The actual extension is <i>C(x)</i>. The physical spring stiffness is <i>k</i>.</p>"
        + eq(r"C(x)=x-L_0,\qquad E_{\mathrm{spring}}(x)=\frac{k}{2}C(x)^2")
        + "<p>Include inertia and any other position-dependent terms in <i>Φ(x)</i>. "
        "The original problem is:</p>"
        + eq(r"\min_x\;\Phi(x)+\frac{k}{2}C(x)^2", True)
        + "<p>For an implicit timestep, an example is "
        "<i>Φ(x) = m(x &minus; x̂)² / (2 Δt²)</i>, where <i>x̂</i> is the inertial target.</p>",
    ),
    card(
        2,
        "Give the material its own copy of the extension",
        "<p>Introduce <i>z</i>, the extension used by the spring's material energy. "
        "Require it to equal the extension measured from the geometry:</p>"
        + eq(r"\min_{x,z}\;\Phi(x)+\frac{k}{2}z^2")
        + eq(r"C(x)-z=0", True)
        + "<p>This is exactly the same physical problem. Substituting <i>z = C(x)</i> "
        "recovers the original energy. It does <b>not</b> require zero spring extension.</p>",
    ),
    card(
        3,
        "Apply ALM to the equality between the two copies",
        "<p>Add a multiplier <i>λ</i> and a positive numerical penalty <i>&rho;</i> "
        "for the mismatch <i>C(x) &minus; z</i>:</p>"
        + eq(r"\mathcal{L}_{\rho}(x,z,\lambda)=\Phi(x)+\frac{k}{2}z^2+\lambda(C(x)-z)+\frac{\rho}{2}(C(x)-z)^2", True)
        + "<p>The physical stiffness <i>k</i> determines the spring energy. "
        "The numerical penalty <i>&rho;</i> controls how strongly an individual solve "
        "tries to make the two extensions agree.</p>"
        + '<p class="note">Units: <i>C</i> and <i>z</i> are lengths; <i>k</i> and <i>&rho;</i> '
        "have units N/m; the multiplier <i>λ</i> has units of force.</p>",
    ),
    card(
        4,
        "Solve for z exactly",
        "<p>For any current position and fixed multiplier, differentiate with respect to <i>z</i>:</p>"
        + eq(r"\frac{\partial\mathcal{L}_{\rho}}{\partial z}=kz-\lambda-\rho(C-z)=0")
        + "<p>Collect the terms containing <i>z</i>:</p>"
        + eq(r"(k+\rho)z=\lambda+\rho C")
        + eq(r"z^*(x,\lambda)=\frac{\lambda+\rho C(x)}{k+\rho}", True)
        + "<p>The material extension is therefore available directly from the geometry and the stored "
        "multiplier. No separate array or iterative solve for <i>z</i> is needed.</p>",
    ),
    card(
        5,
        "Substitute z back into the position problem",
        "<p>Substitution gives the reduced augmented energy:</p>"
        + eq(
            r"\widehat{\mathcal{L}}_{\rho}(x,\lambda)=\Phi(x)+\frac{k\rho}{2(k+\rho)}C(x)^2+\frac{k\lambda}{k+\rho}C(x)-\frac{\lambda^2}{2(k+\rho)}"
        )
        + "<p>The final term is constant with respect to position. Define:</p>"
        + eq(r"\bar{k}=\frac{k\rho}{k+\rho},\qquad s=\frac{k}{k+\rho}")
        + "<p>With the multiplier fixed, the position solve is:</p>"
        + eq(r"x^{n+1}\approx\operatorname{argmin}_x\left[\Phi(x)+\frac{\bar{k}}{2}C(x)^2+s\lambda^n C(x)\right]", True)
        + "<p>The spring stress used during that solve is:</p>"
        + eq(r"t=\bar{k}C+s\lambda^n=kz^*")
        + eq(r"f_{\mathrm{spring}}=-t\frac{dC}{dx}")
        + "<p>For this one-dimensional spring, <i>dC/dx = 1</i>. The spring's contribution "
        "to positional curvature is <i>k̄</i>. The actual VBD implementation makes an "
        "approximate position update through a vertex sweep.</p>",
    ),
    card(
        6,
        "Update the multiplier at the accepted position",
        "<p>Use standard ALM ascent on the equality mismatch:</p>"
        + eq(r"\lambda^{n+1}=\lambda^n+\rho\left[C(x^{n+1})-z^*(x^{n+1},\lambda^n)\right]")
        + "<p>Insert the expression for <i>z*</i> and simplify:</p>"
        + eq(r"\lambda^{n+1}=\lambda^n+\rho\left[C-\frac{\lambda^n+\rho C}{k+\rho}\right]")
        + eq(r"\lambda^{n+1}=\frac{k}{k+\rho}\left(\lambda^n+\rho C(x^{n+1})\right)", True)
        + "<p>Now express it using physical compliance <i>a = 1/k</i>:</p>"
        + eq(r"\lambda^{n+1}=\frac{\lambda^n+\rho C(x^{n+1})}{1+\rho a}", True)
        + "<p>Equivalently:</p>"
        + eq(r"\lambda^{n+1}=\lambda^n+\rho\left[C(x^{n+1})-a\lambda^{n+1}\right]")
        + '<p class="note"><b>This is the compliant ALM update.</b> The denominator comes directly '
        "from the finite spring stiffness. For zero compliance, the update becomes the hard-constraint "
        "ALM update.</p>",
    ),
    card(
        7,
        "Check that the original spring law comes back",
        "<p>At convergence, the multiplier stops changing. The update therefore requires:</p>"
        + eq(r"C-a\lambda=0")
        + eq(r"\lambda=kC,\qquad z=C", True)
        + "<p>Substitute that multiplier into the stress used by the position solve:</p>"
        + eq(r"t=\frac{k\rho}{k+\rho}C+\frac{k}{k+\rho}(kC)=kC")
        + "<p>Thus the converged spring force is the original Hookean force. "
        "The numerical penalty changes the iteration, while the physical stiffness determines "
        "the converged material response.</p>",
    ),
    card(
        8,
        "A numerical example: hold the spring at 2 mm extension",
        "<p>To isolate the multiplier's role, temporarily hold the positions fixed. "
        "Choose <i>k = 1000 N/m</i>, <i>&rho; = 1000 N/m</i>, and start with <i>λ = 0</i>. "
        "The correct physical force is <b>2 N</b>.</p>"
        + eq(
            r"z^*=\frac{\lambda+2\,\mathrm{N}}{2000\,\mathrm{N/m}},\qquad \lambda_{\mathrm{new}}=\frac{\lambda+2\,\mathrm{N}}{2}"
        )
        + '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;text-align:left">'
        "<thead><tr><th>Update</th><th>Old λ (N)</th><th>z* (mm)</th><th>C &minus; z* (mm)</th><th>New λ (N)</th></tr></thead>"
        "<tbody><tr><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td></tr>"
        "<tr><td>2</td><td>1</td><td>1.5</td><td>0.5</td><td>1.5</td></tr>"
        "<tr><td>3</td><td>1.5</td><td>1.75</td><td>0.25</td><td>1.75</td></tr>"
        "<tr><td>4</td><td>1.75</td><td>1.875</td><td>0.125</td><td>1.875</td></tr>"
        "<tr><td>Limit</td><td>2</td><td>2</td><td>0</td><td>2</td></tr></tbody></table></div>"
        + "<p>The mismatch disappears, but the spring's actual extension stays at 2 mm. "
        "The multiplier converges to the force required by the material.</p>"
        + '<p class="note">Zero initialization is only for this illustration. Our implementation '
        "initially seeds the multiplier with <i>kC</i> and subsequently retains history. "
        "During a real solve the positions also move.</p>",
    ),
]

prefix = page.split("<header>", 1)[0].replace(
    "Triangle elasticity ALM — readable equations", "Deriving compliant ALM from a spring"
)
header = """<header><div class="eyebrow">Step-by-step derivation · local / offline</div>
<h1>Compliant ALM, starting from a spring</h1>
<p>Start with the spring energy, introduce a copy of its extension, eliminate that copy, and derive the position and multiplier updates.</p>
<div class="symbols"><span><b>C(x)</b> = geometric extension</span><span><b>z</b> = material extension</span>
<span><b>k</b> = physical stiffness</span><span><b>&rho;</b> = numerical penalty</span><span><b>λ</b> = multiplier / force</span></div></header>"""
footer = """<footer><p><a href="index.html">Triangle ALM equation sheet</a> ·
<a href="../alm-elasticity-code-walkthrough/review.html">Code walkthrough</a> ·
<a href="LICENSE_STIX">STIX font license</a></p>
<p>Offline MathML with an embedded math font.</p><button onclick="window.print()">Print / save as PDF</button>
</footer></main></body></html>"""
output = Path(__file__).with_name("spring.html")
output.write_text(prefix + header + "\n".join(sections) + footer)
print(output)
