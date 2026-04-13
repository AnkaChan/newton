#!/usr/bin/env python
"""Generate self-contained HTML report for step ratio convergence study.

Embeds all PNG figures as base64 data URIs. Also writes a markdown version.
"""
import base64
import os
import numpy as np

OUT_DIR = os.path.expanduser("~/Desktop/scripts/step_ratio_study")
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_tag(path, alt="", width="100%"):
    return f'<img src="data:image/png;base64,{img_b64(path)}" alt="{alt}" style="max-width:{width}; border-radius:6px;">'


def load_exp2():
    d = np.load(os.path.join(OUT_DIR, "exp2_per_iteration.npz"))
    gammas = d["gammas"]
    rows = []
    for g in gammas:
        r = d[f"residual_{g}"]
        rows.append((g, r[0], r[5], r[-1], r[-1] / r[0]))
    return rows


def load_exp1_cloth():
    d = np.load(os.path.join(OUT_DIR, "exp1_cloth_hanging.npz"))
    gammas = d["gammas"]
    rows = []
    baseline = None
    for g in gammas:
        r = d[f"residual_{g}"]
        final = r[-1]
        if g == 1.0:
            baseline = final
        rows.append((g, final))
    return rows, baseline


def load_exp1_table():
    d = np.load(os.path.join(OUT_DIR, "exp1_grid_on_table.npz"))
    gammas = d["gammas"]
    rows = []
    baseline = None
    for g in gammas:
        r = d[f"residual_{g}"]
        final = r[-1]
        if g == 1.0:
            baseline = final
        rows.append((g, final))
    return rows, baseline


def load_exp3():
    d = np.load(os.path.join(OUT_DIR, "exp3_cloth_sag.npz"))
    gammas = d["gammas"]
    centroid = d["eq_centroid_z"]
    lowest = d["eq_lowest_z"]
    return list(zip(gammas, centroid, lowest))


def generate_html():
    exp1_cloth, cloth_base = load_exp1_cloth()
    exp1_table, table_base = load_exp1_table()
    exp2 = load_exp2()
    exp3 = load_exp3()

    # Fixed edge z for excess sag calculation
    fixed_z = 4.0
    base_sag = fixed_z - exp3[-1][2]  # gamma=1.0 lowest z

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VBD Step Ratio Convergence Study</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    margin: 0; padding: 0;
    background: #f4f6f9; color: #333;
    line-height: 1.6;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 30px 40px 60px; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 12px; font-size: 1.8em; }}
  h2 {{ color: #16213e; margin-top: 45px; border-bottom: 2px solid #ddd; padding-bottom: 6px; font-size: 1.4em; }}
  h3 {{ color: #2c3e50; font-size: 1.15em; }}
  p {{ line-height: 1.7; }}
  .timestamp {{ color: #888; font-size: 0.85em; }}
  .summary {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 28px 32px; border-radius: 12px;
    margin: 24px 0; box-shadow: 0 4px 15px rgba(102,126,234,0.3);
  }}
  .summary h2 {{ color: white; border: none; margin-top: 0; font-size: 1.3em; }}
  .summary ul {{ line-height: 1.8; }}
  .card {{
    background: #fff; padding: 24px 28px; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin: 22px 0;
  }}
  .finding {{
    background: #eef7ee; border-left: 4px solid #4daf4a; padding: 14px 20px;
    margin: 14px 0; border-radius: 0 8px 8px 0;
  }}
  .warning {{
    background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 20px;
    margin: 14px 0; border-radius: 0 8px 8px 0;
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ padding: 10px 14px; border: 1px solid #ddd; text-align: center; }}
  th {{ background: #f0f2f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafbfc; }}
  .fig {{ text-align: center; margin: 20px 0; }}
  .fig img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .fig-caption {{ font-size: 0.9em; color: #666; margin-top: 8px; font-style: italic; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  .eq {{ text-align: center; font-size: 1.15em; margin: 16px 0; font-family: 'Times New Roman', serif; }}
  blockquote {{
    border-left: 4px solid #667eea; padding: 12px 20px; margin: 16px 0;
    background: #f8f9ff; border-radius: 0 8px 8px 0;
  }}
</style>
</head>
<body>
<div class="container">

<h1>VBD Step Ratio (&gamma;) Convergence Study</h1>
<p class="timestamp">Date: 2026-04-13 &mdash; Branch: <code>horde/debug-contact-instability</code> &mdash; Newton Physics Engine</p>

<div class="summary">
<h2>Key Findings</h2>
<ul>
<li><strong>&gamma;=1.0</strong> converges fastest per-iteration (optimal for free cloth, no contact)</li>
<li><strong>&gamma;=0.7&ndash;0.9</strong> outperforms 1.0 over full simulations in contact scenes (dampens cross-color interference)</li>
<li><strong>&gamma;&le;0.5</strong> causes severe convergence penalty (33&ndash;95&times; worse) and measurable softness</li>
<li><strong>Recommendation: &gamma;=0.9 for contact, &gamma;=1.0 for free cloth</strong></li>
</ul>
</div>

<h2>1. Motivation</h2>
<div class="card">
<p>The VBD solver updates particle positions via a block-diagonal Newton step:</p>
<div class="eq">x &larr; x + &gamma; &middot; H<sub>local</sub><sup>&minus;1</sup> f</div>
<p>where H<sub>local</sub> is the per-vertex 3&times;3 Hessian and f is the total force (inertia + elastic + bending + contact). The parameter &gamma; (step ratio) controls how much of the Newton step is applied.</p>
<blockquote>Does reducing &gamma; impede convergence of elasticity when objects are moving fast or stretched heavily, causing them to look softer than they should be?</blockquote>
<p>This report answers that question through three controlled experiments.</p>
</div>

<h2>2. Methodology</h2>
<div class="card">
<h3>Implementation</h3>
<p>A <code>step_ratio</code> parameter was added to <code>SolverVBD</code> and the <code>solve_elasticity</code> / <code>solve_elasticity_tile</code> kernels. The displacement update was changed from <code>x += H_inv * f</code> to <code>x += &gamma; * H_inv * f</code>. A <code>force_norm_sq</code> output array records ||f||<sup>2</sup> per particle during each solve.</p>

<h3>Residual Metric</h3>
<p>The primary metric is the <strong>RMS force residual</strong>: root-mean-square of ||f<sub>i</sub>|| across all particles. At the exact solution this is zero. This metric is step-size-independent (unlike displacement, which trivially shrinks with smaller &gamma;).</p>
<p>For per-iteration convergence (Experiment 2), a <strong>consistent evaluation</strong> technique is used: after each iteration, a non-destructive probe pass with &gamma;=0 evaluates forces at the current positions without modifying them. State (positions, displacements, collision buffers) is saved and restored around this probe.</p>

<h3>Scenes</h3>
<table>
<tr><th>Scene</th><th>Description</th><th>Scale</th></tr>
<tr><td>Cloth Hanging</td><td>32&times;16 grid, one side fixed, no ground</td><td>Meter-scale (cell=0.1m)</td></tr>
<tr><td>Tubes on Table</td><td>20&times;10 grid rolled into tubes, 3 stacked layers on ground</td><td>cm-scale (cell=1cm, gravity=981)</td></tr>
</table>

<h3>Parameters</h3>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Gammas tested</td><td>0.3, 0.5, 0.7, 0.9, 1.0</td></tr>
<tr><td>VBD iterations/substep</td><td>5 (Exp 1, 3) or 50 (Exp 2)</td></tr>
<tr><td>Substeps/frame</td><td>10 at 60 FPS</td></tr>
<tr><td>Damping mode</td><td>Absolute</td></tr>
<tr><td>Tile solve</td><td>Disabled</td></tr>
</table>
</div>

<h2>3. Experiment 1: Residual vs Timestep</h2>
<div class="card">
<p><strong>Question:</strong> How does &gamma; affect the residual that accumulates over a full simulation?</p>

<h3>3.1 Cloth Hanging (no ground)</h3>
<p>120 frames (1200 substeps), cloth falls and swings under gravity.</p>
<div class="fig">
{img_tag(os.path.join(OUT_DIR, "exp1_cloth_hanging.png"), "Residual vs timestep — cloth hanging")}
<div class="fig-caption">Figure 1: RMS force residual over substeps for cloth hanging (no ground). Lower is better.</div>
</div>

<table>
<tr><th>&gamma;</th><th>Final Residual</th><th>vs &gamma;=1.0</th></tr>"""

    for g, final in exp1_cloth:
        if g == 1.0:
            html += f'\n<tr><td><strong>{g}</strong></td><td><strong>{final:.3e}</strong></td><td><strong>baseline</strong></td></tr>'
        else:
            ratio = final / cloth_base
            if ratio > 1.5:
                html += f'\n<tr><td>{g}</td><td>{final:.3e}</td><td><strong>{ratio:.1f}&times; worse</strong></td></tr>'
            else:
                html += f'\n<tr><td>{g}</td><td>{final:.3e}</td><td>{ratio:.2f}&times;</td></tr>'

    html += f"""
</table>
<div class="finding">
<strong>Findings:</strong> &gamma;=0.9 and &gamma;=1.0 produce nearly identical residual trajectories. &gamma;&le;0.7 causes progressively worse residual accumulation. With &gamma;=0.3, the residual is two orders of magnitude higher and <em>grows during the active phase</em> before partially recovering as the cloth settles. The mechanism: under-relaxation prevents full per-substep convergence; the unconverged force residual becomes velocity, feeding into the next substep's inertia term in a positive feedback loop.
</div>

<h3>3.2 Tubes on Table (3 layers, contact)</h3>
<p>60 frames (600 substeps), three cylindrical cloth tubes stacked on a ground plane with self-contact.</p>
<div class="fig">
{img_tag(os.path.join(OUT_DIR, "exp1_grid_on_table.png"), "Residual vs timestep — tubes on table")}
<div class="fig-caption">Figure 2: RMS force residual over substeps for tubes on table (3 layers). Contact makes residuals noisier.</div>
</div>

<table>
<tr><th>&gamma;</th><th>Final Residual</th><th>vs &gamma;=1.0</th></tr>"""

    for g, final in exp1_table:
        if g == 1.0:
            html += f'\n<tr><td><strong>{g}</strong></td><td><strong>{final:.0f}</strong></td><td><strong>baseline</strong></td></tr>'
        else:
            ratio = final / table_base
            if ratio > 1.5:
                html += f'\n<tr><td>{g}</td><td>{final:.0f}</td><td><strong>{ratio:.1f}&times; worse</strong></td></tr>'
            elif ratio < 0.85:
                html += f'\n<tr><td>{g}</td><td>{final:.0f}</td><td>{ratio:.2f}&times; (better)</td></tr>'
            else:
                html += f'\n<tr><td>{g}</td><td>{final:.0f}</td><td>{ratio:.2f}&times;</td></tr>'

    html += f"""
</table>
<div class="finding">
<strong>Findings:</strong> &gamma;=0.7 and &gamma;=0.9 <strong>outperform</strong> the baseline in the contact scene. Their steady-state residuals are lower than &gamma;=1.0. The full Newton step causes cross-color interference to compound across substeps. Mild under-relaxation (&gamma;=0.7&ndash;0.9) dampens this feedback. However, &gamma;&le;0.5 is still substantially worse even in the contact scene.
</div>
</div>

<h2>4. Experiment 2: Per-Iteration Convergence</h2>
<div class="card">
<p><strong>Question:</strong> At a fixed state, how does &gamma; affect the rate at which iterations reduce the residual?</p>

<h3>Warmup State Validation</h3>
<p>The tubes-on-table scene is warmed up for <strong>40 frames (400 substeps)</strong> with &gamma;=1.0 to reach a near-rest contact state. The diagnostic below shows that at substep 400, kinetic energy has dropped to KE&asymp;11 (from a peak of ~24,000) and max velocity is ~5.6 cm/s &mdash; a valid near-rest state with all tubes settled on the table.</p>
<div class="fig">
{img_tag(os.path.join(OUT_DIR, "warmup_diagnostic.png"), "Warmup diagnostic")}
<div class="fig-caption">Figure 3: Warmup diagnostic. Top: kinetic energy and max velocity decay. Bottom: Y-Z side views at various substeps. Substep 400 (frame 40) shows tubes at rest on the ground plane.</div>
</div>

<h3>Per-Iteration Convergence Protocol</h3>
<ol>
<li>Warm up for 400 substeps with &gamma;=1.0 to reach near-rest contact state.</li>
<li>Save the state (positions + velocities).</li>
<li>For each &gamma;: restore saved state, run <strong>one substep with 50 iterations</strong>, recording residual after each iteration via a non-destructive &gamma;=0 probe pass.</li>
</ol>

<div class="fig">
{img_tag(os.path.join(OUT_DIR, "exp2_per_iteration.png"), "Per-iteration convergence")}
<div class="fig-caption">Figure 4: Per-iteration convergence curves (left: absolute residual, right: normalized by initial). All gammas start from the same state.</div>
</div>

<table>
<tr><th>&gamma;</th><th>Initial Residual</th><th>After 5 iters</th><th>After 50 iters</th><th>Overall Ratio</th></tr>"""

    for g, init, at5, at50, ratio in exp2:
        if g == 1.0:
            html += f'\n<tr><td><strong>{g}</strong></td><td><strong>{init:.0f}</strong></td><td><strong>{at5:.1f}</strong></td><td><strong>{at50:.1f}</strong></td><td><strong>{ratio:.4f} (fastest)</strong></td></tr>'
        elif g == 0.3:
            html += f'\n<tr><td>{g}</td><td>{init:.0f}</td><td>{at5:.1f}</td><td>{at50:.1f}</td><td>{ratio:.4f} (slowest)</td></tr>'
        else:
            html += f'\n<tr><td>{g}</td><td>{init:.0f}</td><td>{at5:.1f}</td><td>{at50:.1f}</td><td>{ratio:.4f}</td></tr>'

    html += f"""
</table>
<div class="finding">
<strong>Findings:</strong> Per-iteration convergence monotonically improves with higher &gamma;. The full Newton step (&gamma;=1.0) is optimal for reducing residual within a single substep. The benefit is most pronounced in early iterations (0&ndash;5). By iteration 20&ndash;30, all gammas enter a slow convergence regime where off-diagonal coupling dominates. At the typical budget of 5 iterations, &gamma;=1.0 reaches a lower residual than &gamma;=0.3 by a significant margin.
</div>
<div class="warning">
<strong>Reconciling with cross-substep behavior:</strong> &gamma;=1.0 is best per-iteration but &gamma;=0.7&ndash;0.9 is best over a full simulation in contact scenes. This is because the unconverged residual at the end of each substep becomes velocity entering the next substep's inertia. With &gamma;=1.0, this error has a consistent directional bias from cross-color interference and compounds. With &gamma;=0.9, position updates are 10% smaller, making error less directional and less prone to compounding. The net effect: &gamma;=0.9 produces lower <em>cumulative</em> residual despite higher per-substep residual.
</div>
</div>

<h2>5. Experiment 3: Cloth Sag (Softness Test)</h2>
<div class="card">
<p><strong>Question:</strong> Does &gamma; &lt; 1 make cloth appear softer by under-resolving elastic forces?</p>
<p>A 32&times;16 cloth grid is fixed on one side and hangs freely under gravity (no ground). Run for 600 frames (6000 substeps, 10 seconds). Equilibrium metrics averaged over the final 500 substeps.</p>

<div class="fig">
{img_tag(os.path.join(OUT_DIR, "exp3_cloth_sag.png"), "Cloth sag measurements")}
<div class="fig-caption">Figure 5: Cloth sag measurements. Top: equilibrium bar charts. Bottom: time series showing oscillation damping.</div>
</div>

<table>
<tr><th>&gamma;</th><th>Z-Centroid</th><th>Lowest Z</th><th>Excess sag vs &gamma;=1.0</th></tr>"""

    for g, centroid, lowest in exp3:
        excess = ((fixed_z - lowest) - base_sag) / base_sag * 100
        if g == 1.0:
            html += f'\n<tr><td><strong>{g}</strong></td><td><strong>{centroid:.3f}</strong></td><td><strong>{lowest:.3f}</strong></td><td><strong>baseline</strong></td></tr>'
        else:
            html += f'\n<tr><td>{g}</td><td>{centroid:.3f}</td><td>{lowest:.3f}</td><td>+{excess:.1f}%</td></tr>'

    html += f"""
</table>
<div class="finding">
<strong>Findings:</strong> The softness concern is confirmed for aggressive under-relaxation. &gamma;=0.3 produces ~5% more sag &mdash; elastic forces are systematically under-resolved. <strong>&gamma;=0.9 has negligible softness impact</strong> (&lt;1% excess sag). &gamma;=0.7 shows ~1.7% excess, which may or may not be perceptible. The time-series reveals the mechanism: &gamma;=0.3 cloth oscillates with significantly larger amplitude that barely damps. &gamma;=0.7&ndash;1.0 converge to the same equilibrium shape.
</div>
</div>

<h2>6. Summary and Recommendations</h2>
<div class="card">
<h3>The Trade-off</h3>
<table>
<tr><th></th><th>Per-iteration convergence</th><th>Cross-substep stability</th><th>Softness</th></tr>
<tr><td>&gamma;=1.0</td><td>Best</td><td>Worse (contact divergence)</td><td>None</td></tr>
<tr><td>&gamma;=0.9</td><td>Near-best</td><td>Good</td><td>Negligible</td></tr>
<tr><td>&gamma;=0.7</td><td>Moderate</td><td>Good</td><td>Minor (~1.7%)</td></tr>
<tr><td>&gamma;=0.5</td><td>Poor</td><td>Moderate</td><td>Noticeable (~1.8%)</td></tr>
<tr><td>&gamma;=0.3</td><td>Poor</td><td>Poor</td><td>Significant (~5%)</td></tr>
</table>

<h3>Recommendations</h3>
<ol>
<li><strong>Default to &gamma;=1.0</strong> for scenes without contact or with simple ground contact. No convergence benefit to under-relaxation in these cases.</li>
<li><strong>Use &gamma;=0.9 for contact-heavy scenes</strong> (stacked objects, self-contact, folded garments). Eliminates cross-color divergence while preserving &gt;99% convergence speed with negligible softness impact.</li>
<li><strong>Avoid &gamma; &le; 0.5.</strong> The convergence penalty is severe (33&ndash;95&times; higher residuals in free cloth), softness is measurable, and stability benefit is outweighed by convergence loss even in contact scenes.</li>
<li><strong>&gamma;=0.7 is acceptable</strong> as a conservative fallback, but expect ~6&times; residual increase for fast-moving free cloth.</li>
<li><strong>Consider adaptive &gamma;</strong> as future work: &gamma;=1.0 for particles far from contact, &gamma;=0.9 near contact. Optimal convergence everywhere while stabilizing contact regions.</li>
</ol>
</div>

<h2>7. Experimental Details</h2>
<div class="card">
<h3>Files Modified</h3>
<ul>
<li><code>newton/_src/solvers/vbd/particle_vbd_kernels.py</code> &mdash; Added <code>step_ratio</code> and <code>force_norm_sq_out</code> to solve kernels</li>
<li><code>newton/_src/solvers/vbd/solver_vbd.py</code> &mdash; Added <code>step_ratio</code> attribute, allocated buffer, wired to kernels</li>
</ul>
<h3>Experiment Script</h3>
<p><code>scripts/study_step_ratio.py</code> &mdash; Run via <code>uv run --extra examples python scripts/study_step_ratio.py --exp all</code></p>
<h3>Output</h3>
<p>All raw data (.npz) and plots (.png) saved to <code>~/Desktop/scripts/step_ratio_study/</code></p>
</div>

</div>
</body>
</html>"""
    return html


def generate_markdown():
    exp1_cloth, cloth_base = load_exp1_cloth()
    exp1_table, table_base = load_exp1_table()
    exp2 = load_exp2()
    exp3 = load_exp3()
    fixed_z = 4.0
    base_sag = fixed_z - exp3[-1][2]

    md = """# VBD Step Ratio (gamma) Convergence Study

**Date:** 2026-04-13
**Branch:** `horde/debug-contact-instability`
**Newton Physics Engine -- Vertex Block Descent Solver**

---

## 1. Motivation

The VBD solver updates particle positions via a block-diagonal Newton step:

    x <- x + gamma * H_local^{-1} f

where H_local is the per-vertex 3x3 Hessian and f is the total force (inertia + elastic + bending + contact). The parameter gamma (step ratio) controls how much of the Newton step is applied.

> **Does reducing gamma impede convergence of elasticity when objects are moving fast or stretched heavily, causing them to look softer than they should be?**

---

## 2. Methodology

### Residual metric
RMS force residual: root-mean-square of ||f_i|| across all particles. Step-size-independent.

For per-iteration convergence (Exp 2), a consistent evaluation with gamma=0 probe pass (with full state save/restore) gives clean gradient norm snapshots.

### Scenes

| Scene | Description | Scale |
|-------|-------------|-------|
| Cloth Hanging | 32x16 grid, one side fixed, no ground | Meter-scale (cell=0.1m) |
| Tubes on Table | 20x10 grid tubes, 3 stacked layers on ground | cm-scale (cell=1cm, gravity=981) |

### Parameters
- Gammas: 0.3, 0.5, 0.7, 0.9, 1.0
- VBD iterations/substep: 5 (Exp 1, 3) or 50 (Exp 2)
- Substeps/frame: 10 at 60 FPS
- Damping: Absolute; Tile solve: Disabled

---

## 3. Experiment 1: Residual vs Timestep

### 3.1 Cloth Hanging (no ground)

120 frames (1200 substeps), cloth falls and swings under gravity.

![Residual vs timestep -- cloth hanging](exp1_cloth_hanging.png)

| gamma | Final Residual | vs gamma=1.0 |
|-------|---------------|-------------|
"""
    for g, final in exp1_cloth:
        if g == 1.0:
            md += f"| **{g}** | **{final:.3e}** | **baseline** |\n"
        else:
            ratio = final / cloth_base
            md += f"| {g} | {final:.3e} | {ratio:.1f}x {'worse' if ratio > 1 else 'better'} |\n"

    md += """
**Findings:** gamma=0.9 and gamma=1.0 nearly identical. gamma<=0.7 progressively worse. gamma=0.3 is ~100x worse -- under-relaxation leaves unconverged residual that becomes velocity feeding next substep's inertia.

### 3.2 Tubes on Table (3 layers, contact)

60 frames (600 substeps), three cylindrical tubes stacked on ground with self-contact.

![Residual vs timestep -- tubes on table](exp1_grid_on_table.png)

| gamma | Final Residual | vs gamma=1.0 |
|-------|---------------|-------------|
"""
    for g, final in exp1_table:
        if g == 1.0:
            md += f"| **{g}** | **{final:.0f}** | **baseline** |\n"
        else:
            ratio = final / table_base
            label = "worse" if ratio > 1 else "better"
            md += f"| {g} | {final:.0f} | {ratio:.2f}x ({label}) |\n"

    md += """
**Findings:** gamma=0.7-0.9 outperform baseline in contact scene. Cross-color interference compounds with gamma=1.0. gamma<=0.5 still substantially worse.

---

## 4. Experiment 2: Per-Iteration Convergence

### Warmup State Validation

Warmed up for **400 substeps** with gamma=1.0. At substep 400: KE~11 (from peak ~24,000), max velocity ~5.6 cm/s. Valid near-rest state.

![Warmup diagnostic](warmup_diagnostic.png)

### Protocol

1. Warm up 400 substeps with gamma=1.0
2. Save state (positions + velocities)
3. For each gamma: restore state, run 1 substep with 50 iterations, record residual per iteration via gamma=0 probe

![Per-iteration convergence](exp2_per_iteration.png)

| gamma | Initial | After 5 iters | After 50 iters | Ratio |
|-------|---------|--------------|----------------|-------|
"""
    for g, init, at5, at50, ratio in exp2:
        label = " (fastest)" if g == 1.0 else (" (slowest)" if g == 0.3 else "")
        md += f"| {g} | {init:.0f} | {at5:.1f} | {at50:.1f} | {ratio:.4f}{label} |\n"

    md += """
**Findings:** Per-iteration convergence monotonically improves with higher gamma. gamma=1.0 is optimal within a single substep. Benefit most pronounced in iterations 0-5. After iteration 20-30, all gammas enter slow convergence regime.

**Reconciling:** gamma=1.0 best per-iteration but gamma=0.7-0.9 best cross-substep in contact scenes. Unconverged residual becomes velocity with directional bias from cross-color interference; mild under-relaxation dampens this compounding.

---

## 5. Experiment 3: Cloth Sag (Softness Test)

32x16 cloth, one side fixed, no ground. 600 frames (6000 substeps). Equilibrium averaged over last 500 substeps.

![Cloth sag](exp3_cloth_sag.png)

| gamma | Z-Centroid | Lowest Z | Excess sag |
|-------|-----------|----------|-----------|
"""
    for g, centroid, lowest in exp3:
        excess = ((fixed_z - lowest) - base_sag) / base_sag * 100
        if g == 1.0:
            md += f"| **{g}** | **{centroid:.3f}** | **{lowest:.3f}** | **baseline** |\n"
        else:
            md += f"| {g} | {centroid:.3f} | {lowest:.3f} | +{excess:.1f}% |\n"

    md += """
**Findings:** gamma=0.3 produces ~5% more sag. **gamma=0.9 has negligible softness** (<1%). gamma=0.7-1.0 converge to same equilibrium shape.

---

## 6. Summary

| | Per-iter convergence | Cross-substep stability | Softness |
|-|---------------------|------------------------|----------|
| gamma=1.0 | Best | Worse (contact divergence) | None |
| gamma=0.9 | Near-best | Good | Negligible |
| gamma=0.7 | Moderate | Good | Minor (~1.7%) |
| gamma=0.5 | Poor | Moderate | Noticeable (~1.8%) |
| gamma=0.3 | Poor | Poor | Significant (~5%) |

### Recommendations

1. **Default gamma=1.0** for no-contact scenes
2. **gamma=0.9 for contact-heavy scenes** -- eliminates cross-color divergence, >99% convergence speed, negligible softness
3. **Avoid gamma<=0.5** -- severe convergence penalty, measurable softness
4. **gamma=0.7** acceptable as conservative fallback
5. **Future: adaptive gamma** -- 1.0 far from contact, 0.9 near contact
"""
    return md


if __name__ == "__main__":
    html = generate_html()
    html_path = os.path.join(REPORT_DIR, "step_ratio_convergence_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Written: {html_path}")

    md = generate_markdown()
    md_path = os.path.join(REPORT_DIR, "step_ratio_convergence_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Written: {md_path}")

    # Also copy to output dir
    import shutil
    shutil.copy2(html_path, os.path.join(OUT_DIR, "step_ratio_convergence_report.html"))
    shutil.copy2(md_path, os.path.join(OUT_DIR, "step_ratio_convergence_report.md"))
    print(f"Copied to: {OUT_DIR}")
