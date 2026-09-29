# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build an offline equation sheet: uv run --no-project --with latex2mathml python build.py."""

import base64
from pathlib import Path

from latex2mathml.converter import convert


def eq(tex, emphasis=False):
    cls = "equation emphasis" if emphasis else "equation"
    return f'<div class="{cls}">{convert(tex, display="block")}</div>'


def card(number, title, content):
    return f'<section><div class="number">{number}</div><div><h2>{title}</h2>{content}</div></section>'


parts = [
    card(
        1,
        "Two numbers describe the triangle energy",
        "<p>The deformation gradient <i>F</i> has 3 rows and 2 columns. Its norm measures stretch; "
        "<i>J</i> is the current area divided by rest area.</p>"
        + eq(r"C_s=\|F\|_F,\qquad k_s=\mu")
        + eq(r"C_a=J-\alpha,\qquad k_a=K")
        + eq(r"J=\sqrt{\det(F^T F)}")
        + eq(r"K=\lambda_{\mathrm{mat}}+\mu,\qquad \alpha=1+\frac{\mu}{K}")
        + '<p class="note"><b>Two scalar multipliers:</b> one for stretch, one for area. '
        "The material parameter <i>λ<sub>mat</sub></i> is different from either ALM multiplier.</p>",
    ),
    card(
        2,
        "The original elastic energy",
        eq(r"E=A_0\left(\frac{k_s}{2}C_s^2+\frac{k_a}{2}C_a^2\right)+\mathrm{constant}", True)
        + "<p><i>A₀</i> is the rest area. The stretch coordinate is the norm itself: "
        "we do not subtract √2. At rest, the stretch and area stresses balance.</p>"
        + '<p class="note">These are energy coordinates. ALM enforces <i>Cᵢ = zᵢ</i>, '
        "rather than forcing either coordinate to zero.</p>",
    ),
    card(
        3,
        "Add one scalar auxiliary variable per coordinate",
        "<p>For either row <i>i ∈ {s, a}</i>, introduce an auxiliary <i>zᵢ</i>, "
        "a multiplier <i>λᵢ</i>, and a numerical penalty <i>ρᵢ</i>.</p>"
        + eq(r"\mathcal{L}_i=\frac{k_i}{2}z_i^2+\lambda_i(C_i-z_i)+\frac{\rho_i}{2}(C_i-z_i)^2")
        + eq(r"\mathcal{L}=A_0(\mathcal{L}_s+\mathcal{L}_a)")
        + "<p>Minimize over the auxiliary variable analytically:</p>"
        + eq(r"z_i=\frac{\lambda_i+\rho_i C_i}{k_i+\rho_i}", True)
        + "<p>The implementation substitutes this expression directly; it does not store <i>zᵢ</i>.</p>",
    ),
    card(
        4,
        "Use the resulting stress in the vertex solve",
        "<p>Hold the multipliers fixed during the vertex sweep. At each force evaluation:</p>"
        + eq(r"\bar{k}_i=\frac{k_i\rho_i}{k_i+\rho_i}")
        + eq(r"t_i=\bar{k}_i C_i+\frac{k_i}{k_i+\rho_i}\lambda_i", True)
        + "<p>The two scalar stresses produce the full triangle stress matrix:</p>"
        + eq(r"P=t_s\frac{F}{\|F\|_F}+t_a\frac{\partial J}{\partial F}", True)
        + "<p>For a nondegenerate triangle:</p>"
        + eq(r"\frac{\partial J}{\partial F}=JF(F^TF)^{-1}")
        + "<p>The implementation evaluates this derivative without a matrix inverse. "
        "Vertex forces follow from the chain rule, with the rest-area factor:</p>"
        + eq(r"f_{v,d}=-A_0\,P:\frac{\partial F}{\partial x_{v,d}}"),
    ),
    card(
        5,
        "Update the two multipliers after the full sweep",
        "<p>Once every vertex color and its existing DAT truncation have completed, "
        "use the accepted positions to update both multipliers:</p>"
        + eq(r"\lambda_i^{n+1}=\frac{k_i}{k_i+\rho_i}\left(\lambda_i^n+\rho_i C_i(F^{n+1})\right)", True)
        + "<p>This is the usual multiplier ascent after substituting the eliminated auxiliary:</p>"
        + eq(r"\lambda_i^{n+1}=\lambda_i^n+\rho_i(C_i-z_i)")
        + "<p>At a fixed point, the original material law is recovered:</p>"
        + eq(r"\lambda_i=k_i C_i,\qquad P=\mu F+K(J-\alpha)\frac{\partial J}{\partial F}"),
    ),
    card(
        6,
        "Initialization and the current penalty",
        "<p>On first activation or reset, seed the multiplier with the current material stress. "
        "Otherwise retain it across substeps.</p>"
        + eq(r"\lambda_i\leftarrow k_i C_i")
        + "<p>Compute the numerical penalty at the incoming pose, then keep it fixed throughout the substep:</p>"
        + eq(
            r"\rho_i^{\mathrm{inertia}}=\frac{\gamma}{A_0\,\Delta t^2\displaystyle\sum_{v\;\mathrm{free}}m_v^{-1}\|\nabla_{\mathbf{x}_v}C_i\|^2}"
        )
        + eq(r"\rho_i=\max\left(\rho_i^{\mathrm{inertia}},\,9k_i\right),\qquad \gamma=1", True)
        + "<p>The <b>9k floor</b> is the current implementation. The separate experiment sweeps smaller floors. "
        "Triangle damping remains outside this ALM split. Degeneracy guards and floating-point bounds "
        "are omitted from these equations for readability.</p>",
    ),
]

page = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Triangle elasticity ALM — readable equations</title>
<style>
:root{color-scheme:light;--ink:#182b3a;--accent:#096b71;--muted:#536372}
*{box-sizing:border-box}body{margin:0;background:#f1f5f7;color:var(--ink);font:18px/1.65 system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:44px 24px 72px}header{margin-bottom:34px}
.eyebrow{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:750}
h1{font-size:clamp(30px,5vw,46px);line-height:1.15;letter-spacing:-.035em;margin:12px 0 16px}
header p{max-width:760px;color:var(--muted)}section{display:grid;grid-template-columns:40px minmax(0,1fr);gap:15px;
background:white;border:1px solid #dce5ea;border-radius:16px;padding:28px;margin:20px 0;break-inside:avoid}
.number{background:#e0f2ef;color:var(--accent);height:36px;width:36px;border-radius:50%;text-align:center;font-weight:750}
h2{font-size:23px;line-height:1.3;margin:1px 0 18px}p{margin:13px 0}.note{background:#f6f8fa;border-left:3px solid #8ba3b5;padding:12px 16px}
.equation{font-size:1.43rem;overflow-x:auto;padding:14px 8px;margin:12px 0;max-width:100%}
math{font-family:"STIX Two Math","Cambria Math","Latin Modern Math",math}
.emphasis{border:1px solid #b7dcd5;border-radius:10px;background:#f0faf7;padding:20px 12px}
.symbols{display:flex;gap:10px;flex-wrap:wrap}.symbols span{background:white;border:1px solid #dce5ea;padding:5px 12px;border-radius:8px;font-size:15px}
a{color:var(--accent)}footer{font-size:14px;color:var(--muted);padding:15px 0}button{font:inherit;font-size:14px;background:white;border:1px solid #c4d4dc;border-radius:7px;padding:7px 14px;cursor:pointer}
@media(max-width:650px){main{padding:24px 12px}section{padding:18px 12px;grid-template-columns:minmax(0,1fr);gap:10px}.equation{font-size:1.2rem}h2{font-size:21px}}
@media print{body{background:white;font-size:12pt}main{padding:0;max-width:none}section{border-radius:0;box-shadow:none}.equation{font-size:16pt}button{display:none}}
</style></head><body><main><header><div class="eyebrow">Current triangle implementation · local / offline</div>
<h1>Triangle elasticity ALM</h1><p>Two scalar multipliers per triangle: one for the deformation-gradient norm, one for area. The equations below match the implemented split.</p>
<div class="symbols"><span><b>k</b> = material stiffness</span><span><b>&rho;</b> = numerical penalty</span><span><b>λ</b> = stored multiplier</span><span><b>t</b> = stress used by the solve</span></div>
</header>"""
    + "\n".join(parts)
    + """
<footer>Native MathML: this page needs no internet connection or external scripts.
<p><a href="../alm-elasticity-code-walkthrough/review.html">Open the full code walkthrough</a> ·
<a href="../alm-bag-wiggle/results-floor-sweep/index.html">Open the rho sweep results</a></p>
<button onclick="window.print()">Print / save as PDF</button></footer></main></body></html>
"""
)

font = base64.b64encode(Path(__file__).with_name("STIXGeneral.woff2").read_bytes()).decode()
page = page.replace(
    "<style>",
    "<style>@font-face{font-family:EmbeddedSTIX;src:url(data:font/woff2;base64," + font + ') format("woff2");}',
).replace('font-family:"STIX Two Math",', 'font-family:EmbeddedSTIX,"STIX Two Math",')
page = page.replace("Native MathML:", '<a href="LICENSE_STIX">STIX font license</a> · Native MathML:')
output = Path(__file__).with_name("index.html")
output.write_text(page)
print(output)
