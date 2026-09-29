# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001 -- Mathematical symbols and punctuation in HTML.

"""Build an offline interactive report from the matched-state residual sweep."""

import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results-floor-sweep"
LABELS = {
    "off": "ALM off",
    "floor9": "ALM · 9k floor",
    "floor1": "ALM · k floor",
    "floor0p1": "ALM · 0.1k floor",
    "floor0p01": "ALM · 0.01k floor",
    "inertia": "ALM · inertia only",
}
COLORS = ["#273449", "#2866cf", "#13805e", "#b07800", "#cd4a3b", "#924aaa"]


def main():
    cases = json.loads((RESULTS / "cases.json").read_text())
    provenance = json.loads((RESULTS / "provenance.json").read_text())
    assert len(cases) == provenance["cases"] == 60
    data = []
    baselines = {}
    for case in cases:
        stem = f"ke{case['stiffness']}_frame{case['checkpoint_frame']}_{case['mode']}"
        rows = np.loadtxt(RESULTS / f"{stem}.csv", delimiter=",", skiprows=1)
        assert rows.shape == (101, 11) and np.isfinite(rows).all()
        assert np.array_equal(rows[:, 0], np.arange(101))
        item = {
            "k": case["stiffness"],
            "frame": case["checkpoint_frame"],
            "mode": case["mode"],
            "csv": f"{stem}.csv",
            "rho": case["rho"],
            # Six significant figures in the embedded display; CSV preserves full precision.
            "rows": [[float(f"{v:.6g}") for v in row[1:8]] for row in rows],
        }
        data.append(item)
        if case["mode"] == "off":
            baselines[case["stiffness"], case["checkpoint_frame"]] = case
    summaries = {}
    summary_rows = []
    for mode in list(LABELS)[1:]:
        selected = [c for c in cases if c["mode"] == mode]
        ratios = {}
        for iteration, field in ((10, "residual_iteration10_N"), (100, "residual_final_N")):
            ratios[iteration] = np.array(
                [c[field] / baselines[c["stiffness"], c["checkpoint_frame"]][field] for c in selected]
            )
        summaries[mode] = {
            str(n): {
                "median_ratio": float(np.median(r)),
                "min_ratio": float(r.min()),
                "max_ratio": float(r.max()),
                "lower_than_off": int((r < 1.0).sum()),
            }
            for n, r in ratios.items()
        }
        summary_rows.append(
            "<tr><th>"
            + LABELS[mode]
            + "</th>"
            + "".join(
                f"<td>{np.median(r):.3f}×</td><td>{r.min():.3f}–{r.max():.3f}×</td><td>{(r < 1).sum()}/10</td>"
                for r in ratios.values()
            )
            + "</tr>"
        )
    detailed = []
    for key, off in baselines.items():
        selected = [c for c in cases if (c["stiffness"], c["checkpoint_frame"]) == key]
        assert len({c["input_state_sha256"] for c in selected}) == 1
        np.testing.assert_allclose([c["residual_initial_N"] for c in selected], off["residual_initial_N"], rtol=2e-5)
        for iteration, field in ((10, "residual_iteration10_N"), (100, "residual_final_N")):
            detailed.append(
                f"<tr><th>{key[0]:.0e}</th><td>{key[1] / 60:g} s</td><td>{iteration}</td>"
                f"<td>{off[field]:.4g} N</td>"
                + "".join(f"<td>{c[field] / off[field]:.3f}×</td>" for c in selected if c["mode"] != "off")
                + "</tr>"
            )
    stats = {"modes": summaries, "validation": provenance["validation"]}
    (RESULTS / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    floor9 = summaries["floor9"]["10"]
    floor1 = summaries["floor1"]["10"]
    checks = [
        c["instrumented_vs_native_max_position_difference_m"]
        for c in cases
        if "instrumented_vs_native_max_position_difference_m" in c
    ]
    template = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    replacements = {
        "__LABELS__": json.dumps(LABELS),
        "__COLORS__": json.dumps(COLORS),
        "__SUMMARY_ROWS__": "\n".join(summary_rows),
        "__DETAIL_ROWS__": "\n".join(detailed),
        "__PROVENANCE__": html.escape(json.dumps(provenance, indent=2)),
        "__REVISION__": provenance["revision"],
        "__F9_MEDIAN__": f"{100 * (1 - floor9['median_ratio']):.1f}%",
        "__F9_RANGE__": f"{floor9['min_ratio']:.3f}–{floor9['max_ratio']:.3f}×",
        "__F1_MEDIAN__": f"{100 * (1 - floor1['median_ratio']):.1f}%",
        "__F1_RANGE__": f"{floor1['min_ratio']:.3f}–{floor1['max_ratio']:.3f}×",
        "__CHECK_MAX__": f"{max(checks):.3g}",
        "__FD_ERROR__": f"{provenance['validation']['energy_gradient_relative_error']:.3g}",
        "__NATIVE_ERROR__": f"{provenance['validation']['native_force_relative_error']:.3g}",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    assert "__DATA__" not in template
    path = RESULTS / "index.html"
    path.write_text(template)
    assert path.stat().st_size < 500 * 1024
    print(path)
    print(json.dumps(summaries, indent=2))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALM cloth: matched-state residual and rho sweep</title>
<style>
:root{color-scheme:light;--ink:#18283b;--muted:#526477;--line:#dce3e9;--paper:#fff;--bg:#f2f5f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1230px;margin:auto;padding:36px 26px 70px}h1{font-size:clamp(29px,4vw,45px);line-height:1.15;letter-spacing:-1.2px;max-width:1000px}h2{font-size:25px;line-height:1.25;margin-top:0}h3{font-size:18px}p{max-width:1040px}
.eyebrow{font-size:13px;letter-spacing:.13em;text-transform:uppercase;color:#2866cf;font-weight:750}.lede{font-size:21px;max-width:1000px}.muted,small{color:var(--muted)}
section{margin:25px 0;padding:25px;background:var(--paper);border:1px solid var(--line);border-radius:14px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:#eaf1fb;border-radius:10px;padding:18px}.card b{display:block;font-size:29px}.card span{font-size:14px}.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end;margin:20px 0}.controls label{display:flex;flex-direction:column;font-size:14px;font-weight:650}select,input{font:inherit}select{max-width:100%;padding:8px;border:1px solid #9aabba;border-radius:5px;background:white}.plots{display:grid;grid-template-columns:1fr 1fr;gap:20px}.plot{min-width:0}svg{width:100%;height:auto;display:block}.legend{display:flex;gap:8px 16px;flex-wrap:wrap;font-size:13px}.legend span:before{content:'';display:inline-block;width:18px;height:3px;background:var(--c);margin-right:6px;vertical-align:middle}.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#f4f7fa}tbody tr:hover{background:#f1f6fc}code{background:#eef2f6;padding:2px 5px;border-radius:4px;font-size:.9em}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:18px;border-radius:8px}a{color:#195eba}details{margin-top:18px}.note{border-left:4px solid #c58a24;padding:8px 16px;background:#fff9ed}.formula{font-family:ui-monospace,monospace;font-size:15px;overflow-wrap:anywhere}.range{display:flex;gap:16px;align-items:center;margin-top:18px}.range input{width:250px;max-width:55%}.badge{font-size:12px;padding:4px 8px;border-radius:5px;background:#e8edf2}.chart-desc{font-size:13px;color:var(--muted);min-height:40px}@media(max-width:760px){main{padding:20px 12px}.cards,.plots{grid-template-columns:1fr}section{padding:18px}.lede{font-size:18px}th,td{padding:8px}.card b{font-size:25px}}
code{overflow-wrap:anywhere}
</style></head><body><main>
<div class="eyebrow">Pinned bag • controlled one-substep experiment • 29 September 2026</div>
<h1>Does lowering rho make ALM elasticity converge faster?</h1>
<p class="lede">The current <code>9k</code> floor changes the measured force residual only modestly. A <code>k</code> floor is a stronger candidate to investigate; much smaller rho can leave the material stress far behind the current deformation.</p>
<div class="cards"><div class="card"><b>__F9_MEDIAN__ lower</b><span>Median original residual at 10 iterations, current 9k floor versus off. Paired ratios: __F9_RANGE__.</span></div><div class="card"><b>__F1_MEDIAN__ lower</b><span>Median original residual at 10 iterations, k floor versus off. Paired ratios: __F1_RANGE__.</span></div><div class="card"><b>60 solves</b><span>5 stiffnesses × 2 matched states × 6 modes. Each solve contains 100 iterations of one 1/600 s substep.</span></div></div>
<p class="muted">These are force-balance measurements, not visual deformation scores or a timing benchmark. The previous videos used a 9k floor; this experiment changes only the triangle and bending rho floors in an instrumented harness.</p>
<section><h2>Inspect a matched comparison</h2><p>Every curve in a selected case starts with identical particle positions, velocities, rigid transforms and rigid velocities. Iteration 0 is after the native predictor and initial DAT truncation; each subsequent sample follows a complete particle color sweep, its per-color DAT stages and the ALM multiplier update.</p>
<div class="controls"><label>Triangle stiffness<select id="stiffness"><option value="1000">1e3</option><option value="10000">1e4</option><option value="100000">1e5</option><option value="1000000">1e6</option><option value="10000000" selected>1e7</option></select></label><label>Starting checkpoint<select id="frame"><option value="60">1 s · after settling</option><option value="180">3 s · during wiggle</option></select></label><label>Stress-lag panel<select id="mode"><option value="floor9">9k floor</option><option value="floor1">k floor</option><option value="floor0p1">0.1k floor</option><option value="floor0p01">0.01k floor</option><option value="inertia" selected>Inertia only</option></select></label><label>Vertical scale<select id="scale"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></label></div>
<div class="plots"><div class="plot"><h3>Original material force residual</h3><svg id="residual" viewBox="0 0 560 350" role="img" aria-label="Original residual versus iterations for six ALM settings"></svg><div class="legend" id="legend"></div><p class="chart-desc">Lower is better for this diagnostic. The original material forces are reevaluated at each accepted pose; ALM cannot improve this curve merely by temporarily reducing its own stress.</p></div><div class="plot"><h3 id="lag-title">Original versus ALM force balance</h3><svg id="lag" viewBox="0 0 560 350" role="img" aria-label="Original residual, reduced ALM residual and constitutive force gap"></svg><div class="legend"><span style="--c:#273449">Original residual</span><span style="--c:#2866cf">ALM residual</span><span style="--c:#cd4a3b">Constitutive force gap</span></div><p class="chart-desc">The ALM residual uses the reduced ALM force with the just-updated multiplier. The gap is the RMS difference between this force and the authored material force at the same pose. This is a force gap, not a geometric constraint violation.</p></div></div>
<div class="range"><label for="iteration">Read iteration <b id="iteration-label">10</b></label><input id="iteration" type="range" min="0" max="100" value="10"></div>
<div class="table-wrap"><table><thead><tr><th>Mode</th><th>Original RMS (N)</th><th>Ratio to off</th><th>ALM RMS (N)</th><th>Force gap (N)</th><th>Raw data</th></tr></thead><tbody id="case-table"></tbody></table></div>
<details><summary>Actual rho / k ranges for this checkpoint</summary><pre id="rho"></pre></details>
</section>
<section><h2>Paired results across all ten starting states</h2><p>Each ratio divides an ALM residual by the ALM-off residual from the same starting state and iteration. Values below 1 are lower. The median treats each of the ten states equally; it is not a pooled force norm. “Lower” counts any decrease, including small changes.</p><div class="table-wrap"><table><thead><tr><th rowspan="2">Mode</th><th colspan="3">10 iterations · original budget</th><th colspan="3">100 iterations · extended solve</th></tr><tr><th>Median ratio</th><th>Range</th><th>Lower</th><th>Median ratio</th><th>Range</th><th>Lower</th></tr></thead><tbody>__SUMMARY_ROWS__</tbody></table></div>
<p>At the stiffest wiggle checkpoint (ke = 1e7, t = 3 s), the off residual falls from 1,786 N after prediction to 494 N at iteration 10 and 101 N at iteration 100. The k floor gives 375 N and 93.1 N at those two budgets. Similar-looking trajectories therefore do not establish that the original solve was already converged.</p><p>The smallest floors do more than lag behind: their original residual grows severely during the extended solve. At iteration 100, the median ratios are 266× off for 0.01k and 954× off for inertia-only. The 0.1k floor is mixed, including one checkpoint at 9.23× off after 100 iterations.</p>
<details><summary>Show every stiffness / checkpoint pair</summary><div class="table-wrap"><table><thead><tr><th>Triangle ke</th><th>Checkpoint</th><th>Iteration</th><th>Off residual</th><th>9k / off</th><th>k / off</th><th>0.1k / off</th><th>0.01k / off</th><th>Inertia / off</th></tr></thead><tbody>__DETAIL_ROWS__</tbody></table></div></details></section>
<section><h2>Why rho matters here</h2><p>For a quadratic material row with stiffness <i>k</i>, eliminating its auxiliary strain gives an effective scalar curvature and a transmitted stress:</p><p class="formula">k_eff = k rho / (k + rho)<br>t = k_eff C + [k / (k + rho)] y<br>y_next = t</p><p>At a 9k floor, <code>k_eff = 0.9k</code>: the row retains 90% of its stiffness. Floors of k, 0.1k and 0.01k retain 50%, 9.09% and 0.99%, respectively, when the floor dominates the inertia metric. This explains why the current implementation changes the solve only modestly.</p><p>Lower rho also slows how quickly the stress history catches up. Holding C fixed, the constitutive mismatch contracts by <code>k / (k + rho)</code> per dual update: 0.1 at 9k, 0.5 at k, 0.909 at 0.1k and 0.990 at 0.01k. Inertia-only triangle rho can be orders of magnitude smaller than k. Its ALM residual may be small while the original residual remains large.</p><p>There is a second limitation: triangle stretching is represented by the scalar invariant <code>C = ||F||</code>, plus the area row. Its transverse curvature contains <code>t / ||F||</code>, which returns to the original shear stiffness at constitutive consistency. Lowering the scalar-row rho does not remove every stiff deformation mode.</p><p class="note">A lower one-substep residual does not establish a safe new default. This sweep resets histories, samples only two trajectory states per stiffness and changes triangle and bending floors together. The report leaves the production 9k policy unchanged.</p></section>
<section><h2>What was measured, and what was controlled</h2><p class="formula">r_i = f_original_elastic,i + f_damping,i + f_contact,i + m_i (x_hat_i − x_i) / dt²<br>RMS = sqrt( sum over free vertices ||r_i||² / N_free )</p><p>Elasticity, material damping and inertia are evaluated in float64 at the solver’s float32 accepted positions. Contact forces are freshly evaluated using native float32 particle–rigid and cloth self-contact kernels, the current rigid poses and the native contact parameters. Cached per-color forces are not reused. Gravity and the external predictor force enter through x_hat. Pinned vertices are excluded.</p><ul><li>The bag has 1,657 particles, 3,240 triangles, 4,896 edges and 72 pinned rim vertices, with three rigid objects inside. Triangle ke ranges from 1e3 through 1e7; ka = 0.2 ke. Bending ke stays 200. Material damping is unchanged.</li><li>An ALM-off trajectory uses the original 60 fps, 10 substeps per frame, 10 iterations per substep, 1 s settling and prescribed rim wiggle. Complete states are sampled at frame 60 and frame 180. Each candidate receives the next frame’s pin motion and runs only the first substep.</li><li>Every candidate uses a fresh solver, a fresh initial collision query and the same full state. Elastic ALM starts from constitutive stress seeds at the incoming pose; rigid contact history is cold. Native contact refresh, rigid updates and DAT scheduling remain active. This is not the proposed, unimplemented ALM contact design.</li><li>Rho is frozen during the measured substep: <code>rho = max(rho_inertia, floor × k)</code>. Both triangle rows and dihedral bending receive the same floor factor. The inertial formulas, clamping and rho_scale = 1 are otherwise unchanged.</li><li>The measured residual covers particle force balance. It excludes rigid-body residuals and DAT constraint reaction forces; it is not a full coupled KKT residual or a guarantee of collision feasibility. Native contact penalty parameters can evolve during the solve.</li><li>No wall-clock speed comparison is made: extra force evaluations, float64 arithmetic and readback instrumentation distort timing. This also does not replace a retained-history, many-step trajectory comparison.</li></ul></section>
<section><h2>Validation and reproducibility</h2><p>The diagnostic force matched an independently differentiated NumPy energy to relative error <b>__FD_ERROR__</b> and the native float32 elastic-plus-damping force to <b>__NATIVE_ERROR__</b> on a deformed two-triangle, one-hinge fixture. ALM-off and 9k instrumented solves were separately compared against plain native solves; the largest coordinate difference was <b>__CHECK_MAX__ m</b>. All 60 cases were finite, each has 101 samples, and all six modes in every pair have matching initial full-state hashes and original residuals.</p><p>Diagnostic source revision: <code>__REVISION__</code>. The production elasticity implementation is unchanged from <code>60407c40</code>. The bag is the previously documented reconstruction of the May 26 scene; the original untracked wiggle worktree was unavailable.</p><p><a href="../run_floor_sweep.py">Experiment harness</a> · <a href="../cloth_residual.py">Residual and metric kernels</a> · <a href="../validate_cloth_residual.py">Independent validation</a> · <a href="cases.json">Full case metadata</a> · <a href="summary.json">Summary numbers</a> · <a href="provenance.json">Provenance</a> · <a href="../results-triangle-bending/index.html">Earlier trajectory videos</a> · <a href="../../alm-elasticity-code-walkthrough/review.html">Implementation walkthrough</a></p><p>CSV links above preserve full-precision measurements. The HTML embeds six significant figures for offline plots. Binary checkpoint files are retained locally and excluded from Git; they can be regenerated from the harness.</p><pre>source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh alm-floor-sweep compete
export PYTHONPATH="$PWD"
export WARP_CACHE_PATH="$PWD/.cache/warp-alm-tri"
uv run --no-sync python notes/alm-bag-wiggle/run_floor_sweep.py
uv run --no-sync python notes/alm-bag-wiggle/build_floor_report.py</pre><details><summary>Full run provenance</summary><pre>__PROVENANCE__</pre></details></section>
<p class="muted">Local, self-contained report. No external scripts, fonts, telemetry or network requests.</p>
</main><script>
const DATA=__DATA__, LABELS=__LABELS__, COLORS=__COLORS__;
const MODES=Object.keys(LABELS), $=id=>document.getElementById(id);
function num(n){return n===0?'0':n.toPrecision(5)}
function selected(){return DATA.filter(d=>d.k===+$('stiffness').value&&d.frame===+$('frame').value)}
function chart(id,series){
 const svg=$(id), log=$('scale').value==='log', W=560,H=350,L=67,R=18,T=18,B=53,pw=W-L-R,ph=H-T-B;
 const vals=series.flatMap(s=>s.values).filter(v=>Number.isFinite(v)&&(!log||v>0));
 let lo=log?Math.floor(Math.log10(Math.min(...vals))):0,hi=log?Math.ceil(Math.log10(Math.max(...vals))):Math.max(...vals)*1.06;
 if(hi<=lo)hi=lo+1;
 const x=i=>L+i/100*pw,y=v=>T+(hi-(log?Math.log10(v):v))/(hi-lo)*ph;
 let out=`<rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="#fafcff"/>`;
 for(let t=0;t<=5;t++){const v=lo+(hi-lo)*t/5,yy=T+ph-ph*t/5,label=log?Math.pow(10,v):v;out+=`<path d="M${L} ${yy}H${W-R}" stroke="#e0e6ed"/><text x="${L-9}" y="${yy+4}" text-anchor="end" font-size="11" fill="#526477">${label.toExponential(1)}</text>`}
 for(let i=0;i<=100;i+=20){const xx=x(i);out+=`<text x="${xx}" y="${H-B+22}" text-anchor="middle" font-size="12" fill="#526477">${i}</text>`}
 out+=`<path d="M${L} ${T}V${H-B}H${W-R}" stroke="#7a8da2" fill="none"/><text x="${L+pw/2}" y="${H-7}" text-anchor="middle" font-size="12" fill="#526477">Completed particle sweeps</text><text transform="translate(16 ${T+ph/2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#526477">Free-vertex RMS force (N)</text>`;
 for(const s of series){let p='',pen=false;for(let i=0;i<s.values.length;i++){const v=s.values[i];if(!Number.isFinite(v)||(log&&v<=0)){pen=false;continue}p+=`${pen?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)} `;pen=true}out+=`<path d="${p}" stroke="${s.color}" stroke-width="2" fill="none"><title>${s.label}</title></path>`}
 const marker=x(+$('iteration').value);out+=`<path d="M${marker} ${T}V${H-B}" stroke="#7d8793" stroke-dasharray="4 4"/>`;svg.innerHTML=out;
}
function render(){const group=selected(),i=+$('iteration').value,off=group.find(d=>d.mode==='off');
 $('iteration-label').textContent=i;
 chart('residual',group.map(d=>({values:d.rows.map(r=>r[0]),color:COLORS[MODES.indexOf(d.mode)],label:LABELS[d.mode]})));
 const lag=group.find(d=>d.mode===$('mode').value);$('lag-title').textContent=LABELS[lag.mode]+' · stress lag';
 chart('lag',[0,1,2].map((c,j)=>({values:lag.rows.map(r=>r[c]),color:['#273449','#2866cf','#cd4a3b'][j],label:['Original residual','ALM residual','Force gap'][j]})));
 $('legend').innerHTML=MODES.map((m,j)=>`<span style="--c:${COLORS[j]}">${LABELS[m]}</span>`).join('');
 $('case-table').innerHTML=group.map(d=>`<tr><th>${LABELS[d.mode]}</th><td>${num(d.rows[i][0])}</td><td>${(d.rows[i][0]/off.rows[i][0]).toFixed(3)}×</td><td>${num(d.rows[i][1])}</td><td>${num(d.rows[i][2])}</td><td><a href="${d.csv}">CSV</a></td></tr>`).join('');
 $('rho').textContent=JSON.stringify(Object.fromEntries(group.filter(d=>d.mode!=='off').map(d=>[LABELS[d.mode],d.rho])),null,2);
 document.documentElement.dataset.reportReady='true';
}
for(const id of ['stiffness','frame','mode','scale','iteration'])$(id).addEventListener('input',render);
render();
</script></body></html>
"""


if __name__ == "__main__":
    main()
