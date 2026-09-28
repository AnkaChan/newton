# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001
"""Build a local HTML report and standalone plots from the recorded PR runs."""

import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LABELS = {
    "extension": "Gravity-loaded beam extension",
    "stretch": "Uniaxial stretch to 2×",
    "twist": "360° beam twist",
    "compression50": "50% compression and recovery",
    "compression90": "90% compression and recovery",
    "refinement_visual": "Refinement: displayed medium mesh",
    "refinement_coarse": "Refinement: coarse mesh",
    "refinement_medium": "Refinement: medium mesh",
    "refinement_fine": "Refinement: fine mesh",
    "sliver": "Sliver elements, 10:1 aspect ratio",
}
ORDER = list(LABELS)
PARENT = {
    **dict.fromkeys(["compression50", "compression90"], "compression"),
    **dict.fromkeys([name for name in ORDER if name.startswith("refinement")], "refinement"),
}


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = {}
    checks = []
    provenance = json.loads((RESULTS / "provenance.json").read_text())
    instrumentation = json.loads((RESULTS / "instrumentation-validation.json").read_text())
    assert len(instrumentation) == 6 and all(row["positions_velocities_bitwise_equal"] for row in instrumentation)
    for case in ("extension", "stretch", "twist", "compression", "refinement", "sliver"):
        for mode in ("off", "on"):
            record = json.loads((RESULTS / f"{case}-{mode}.json").read_text())
            checks.append(record)
            for run in record["runs"]:
                records[run["name"], mode] = run
    table, sections, comparisons = [], [], []
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    for name in ORDER:
        off, on = records[name, "off"], records[name, "on"]
        for key in ("steps", "dt", "iterations", "vertices", "tets", "initial_model_sha256"):
            assert off[key] == on[key], (name, key)
        data = {
            mode: np.genfromtxt(RESULTS / records[name, mode]["csv"], delimiter=",", names=True)
            for mode in ("off", "on")
        }
        assert np.array_equal(data["off"]["step"], data["on"]["step"])
        mean_ratio = on["residual_rms_mean_N"] / off["residual_rms_mean_N"]
        median_ratio = on["residual_rms_median_N"] / off["residual_rms_median_N"]
        better = float(np.mean(data["on"]["residual_rms_N"] < data["off"]["residual_rms_N"]))
        comparisons.append(
            {
                "name": name,
                "mean_rms_ratio_on_off": mean_ratio,
                "median_rms_ratio_on_off": median_ratio,
                "fraction_steps_lower_rms_on": better,
                "off": off,
                "on": on,
            }
        )
        style = "better" if mean_ratio < 1 else "worse"
        table.append(
            f'<tr><td><a href="#{name}">{LABELS[name]}</a></td><td>{off["steps"]:,}</td><td>{off["iterations"]}</td><td>{off["residual_rms_mean_N"]:.5g}</td><td>{on["residual_rms_mean_N"]:.5g}</td><td class="{style}">{mean_ratio:.3f}×</td></tr>'
        )
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), layout="constrained", sharex=True)
        for mode, color, label in (("off", "#6b5eb1", "ALM off"), ("on", "#087f70", "Pressure ALM on")):
            d = data[mode]
            axes[0].plot(d["step"], d["residual_rms_N"], color=color, lw=0.8, alpha=0.9, label=label)
            axes[1].plot(d["step"], d["relative_to_initial"], color=color, lw=0.8, alpha=0.9, label=label)
        for ax in axes:
            ax.set_yscale("log")
            ax.grid(alpha=0.16, which="both")
        axes[0].set_ylabel("Free-vertex RMS residual [N]")
        axes[0].set_title(f"{LABELS[name]} · {off['iterations']} sweeps / step · dt = {off['dt']:.7f} s", loc="left")
        axes[0].legend(loc="best")
        axes[1].set_ylabel("Final / initial residual L2")
        axes[1].set_xlabel("Consecutive solver.step calls (all substeps included)")
        switch = 1000 if name in ("stretch", "twist") else 750 if name.startswith("compression") else None
        if switch:
            for ax in axes:
                ax.axvline(switch, color="#94632d", ls="--", lw=1, alpha=0.8)
            axes[0].text(
                switch + off["steps"] * 0.012,
                0.98,
                "Ramp ends" if switch == 1000 else "Top released",
                transform=axes[0].get_xaxis_transform(),
                va="top",
                fontsize=9,
            )
        filename = f"{name}.svg"
        fig.savefig(RESULTS / filename, metadata={"Date": None})
        plot_path = RESULTS / filename
        plot_path.write_text("\n".join(line.rstrip() for line in plot_path.read_text().splitlines()) + "\n")
        plt.close(fig)
        stats = []
        for title, key in (
            ("Mean RMS [N]", "residual_rms_mean_N"),
            ("Median RMS [N]", "residual_rms_median_N"),
            ("95th-percentile RMS [N]", "residual_rms_p95_N"),
            ("Final-step RMS [N]", "residual_rms_final_N"),
            ("Median final / initial L2", "relative_to_initial_median"),
            ("Minimum det(F) across steps", "minimum_det_F"),
        ):
            stats.append(f"<tr><td>{title}</td><td>{off[key]:.6g}</td><td>{on[key]:.6g}</td></tr>")
        phase = ""
        if switch:
            phase_rows = []
            for label, selection in (
                ("Driven phase", slice(None, switch)),
                ("After ramp / release", slice(switch, None)),
            ):
                a, b = [data[m]["residual_rms_N"][selection].mean() for m in ("off", "on")]
                phase_rows.append(f"<tr><td>{label}</td><td>{a:.5g}</td><td>{b:.5g}</td><td>{b / a:.3f}×</td></tr>")
            phase = (
                '<h3>Mean RMS by motion phase</h3><div class="table"><table><thead><tr><th>Phase</th><th>Off [N]</th><th>On [N]</th><th>On / off</th></tr></thead><tbody>'
                + "".join(phase_rows)
                + "</tbody></table></div>"
            )
        sections.append(f'''<section id="{name}"><h2>{LABELS[name]}</h2><p>{off["vertices"]:,} vertices · {off["tets"]:,} tetrahedra · {off["steps"]:,} consecutive steps · {off["steps"] * off["dt"]:.3f} simulated seconds.</p>
<figure><img src="{filename}" width="1100" height="640" alt="Per-step residual comparison for {LABELS[name]}"><figcaption>Every step is plotted, with no temporal averaging. Logarithmic vertical axes. Values are measured on each mode’s own evolving trajectory.</figcaption></figure>
<p>The ALM-on mean RMS is <strong>{mean_ratio:.3f}×</strong> the ALM-off mean; ALM on has lower RMS at {better:.1%} of corresponding step indices.</p>
<div class="table"><table><thead><tr><th>Metric</th><th>ALM off</th><th>Pressure ALM on</th></tr></thead><tbody>{"".join(stats)}</tbody></table></div>{phase}
<p class="downloads"><a href="{off["csv"]}" download>ALM-off CSV</a> · <a href="{on["csv"]}" download>ALM-on CSV</a> · <a href="{filename}" download>Standalone SVG plot</a> · <a href="{PARENT.get(name, name)}-on.json">Run details</a></p></section>''')
    check_rows = []
    for case in ("extension", "stretch", "twist", "compression", "refinement", "sliver"):
        cells = []
        for mode in ("off", "on"):
            record = next(r for r in checks if r["case"] == case and r["mode"] == mode)
            status = "PASS" if record["original_checks_passed"] else "FAIL"
            detail = "; ".join(f["message"] for f in record["failures"])
            cells.append(
                f'<td class="{"better" if status == "PASS" else "worse"}">{status}<small>{html.escape(detail)}</small></td>'
            )
        check_rows.append(f"<tr><td>{case}</td>{''.join(cells)}</tr>")
    all_steps = sum(r["steps"] for r in records.values())
    passed = sum(r["original_checks_passed"] for r in checks)
    wins = sum(r["mean_rms_ratio_on_off"] < 1 for r in comparisons)
    head = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PR #2901 · ALM on/off per-step residuals</title>
<style>:root{{--ink:#203440;--muted:#566b74;--line:#d6e1df;--accent:#087f70}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f5;color:var(--ink);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1140px;margin:auto;padding:52px 30px}}h1{{font-size:clamp(32px,4vw,52px);line-height:1.15;letter-spacing:-.035em}}h2{{font-size:28px;line-height:1.25}}h3{{font-size:19px}}a{{color:#076d61;text-underline-offset:3px}}p{{max-width:980px}}.eyebrow{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--accent)}}.lead{{font-size:19px;color:var(--muted)}}.note{{background:#e4f1ed;border-left:4px solid var(--accent);padding:20px 24px;margin:24px 0}}.caution{{background:#fff1d9;border-color:#b38136}}.table{{overflow:auto;background:white;border:1px solid var(--line);border-radius:10px;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;text-align:left}}th,td{{padding:12px 15px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eaf0ee;font-size:12px}}tr:last-child td{{border-bottom:0}}td small{{display:block;color:var(--muted);font-size:12px;max-width:420px}}.better{{color:#07694f;font-weight:650}}.worse{{color:#94521d;font-weight:650}}section{{border-top:1px solid var(--line);padding-top:30px;margin-top:48px;scroll-margin-top:20px}}pre{{padding:20px;background:#18303a;color:#e5eeec;overflow:auto;border-radius:10px;font:13px/1.7 ui-monospace,monospace}}code{{font:0.9em ui-monospace,monospace;overflow-wrap:anywhere}}figure{{margin:25px 0;background:white;border:1px solid var(--line);padding:12px;border-radius:10px}}figure img{{width:100%;height:auto;display:block}}figcaption,.small,.downloads{{font-size:13px;color:var(--muted)}}nav{{display:flex;gap:10px;flex-wrap:wrap}}nav a{{padding:5px 12px;background:white;border:1px solid var(--line);border-radius:16px;font-size:13px}}button{{font:inherit;padding:7px 14px;border:1px solid var(--line);border-radius:7px;background:white;cursor:pointer}}a:focus-visible,button:focus-visible{{outline:3px solid #c4902a;outline-offset:4px}}@media(max-width:650px){{main{{padding:30px 18px}}th,td{{padding:10px}}figure{{padding:3px}}figcaption{{padding:8px}}}}@media print{{body{{background:white;font-size:11pt}}main{{max-width:none;padding:0}}nav,button{{display:none}}figure{{break-inside:avoid}}section{{break-before:page}}h2,h3{{break-after:avoid}}a{{color:inherit}}}}</style></head><body><main>
<div class="eyebrow">Newton / VBD · Recorded 28 September 2026</div><h1>ALM on vs off.<br>Residuals across consecutive timesteps.</h1>
<p class="lead">The six volumetric tests from <a href="https://github.com/newton-physics/newton/pull/2901">PR #2901</a>, including their auxiliary compression and mesh-refinement runs, evaluated on the ALM branch with identical scenario settings.</p>
<nav><a href="#summary">Comparison</a><a href="#checks">Original test checks</a><a href="#method">Residual definition</a><a href="#extension">Per-step plots</a><a href="#reproduce">Reproduce</a></nav>
<div class="note"><strong>{all_steps:,} physical steps across both modes.</strong> ALM on has lower mean free-vertex RMS residual in {wins} of {len(comparisons)} recorded trajectories. {passed} of {len(checks)} mode-specific original test runs pass. Read the per-case results: the scenes use different stiffnesses, damping, resolutions, and iteration budgets.</div>
<section id="summary"><h2>Equal iteration budgets</h2><p>ALM on uses the implemented <strong>pressure-only tet ALM</strong>, with <code>rho_scale=1</code> and retained history. ALM off uses the ordinary material path in the same solver source. The optional full-matrix mode and the unintegrated SVD proposal are not used.</p>
<div class="table"><table><thead><tr><th>Trajectory</th><th>Steps / mode</th><th>Sweeps / step</th><th>Mean RMS off [N]</th><th>Mean RMS on [N]</th><th>On / off</th></tr></thead><tbody>{"".join(table)}</tbody></table></div><p class="small">A ratio below 1 means lower mean residual with ALM. These are comparisons at equal iteration counts, not equal wall time. The mean weights every solver step equally, including the driving and settling phases.</p></section>
<section id="checks"><h2>Original PR checks</h2><p>The unmodified example <code>test_post_step()</code> and <code>test_final()</code> routines ran under both modes. The solver factory also instruments auxiliary runs created inside <code>test_final()</code>, so the 90% compression and three refinement resolutions use the selected ALM mode.</p><div class="table"><table><thead><tr><th>PR test</th><th>ALM off</th><th>ALM on</th></tr></thead><tbody>{"".join(check_rows)}</tbody></table></div><p class="small">The residual comparison is distinct from the PR’s displacement, volume, recovery, and stability checks. In particular, the sliver case passes its stability bounds while retaining about 17 N of mean RMS residual in both modes. Passing that test does not establish convergence of the per-step solve.</p></section>
<section id="method"><h2>What the residual measures</h2><pre>r_i = f_elastic,i(x) + f_damping,i(x, x_start) − m_i (x_i − x_hat,i) / dt²

R_L2  = sqrt(Σ_free ||r_i||²)                         [N]
R_RMS = R_L2 / sqrt(number of free vertices)          [N per free vertex]
R_rel = R_L2 / max(R_L2 at x_start, 10⁻¹² N)</pre>
<p>The elastic force uses the original stable Neo-Hookean law: <code>P = μF + [(λ_L+μ)(det F−1)−μ] cof F</code>. Tet damping uses <code>P_d = (2 k_d/dt) F (FᵀF−F_startᵀF_start)</code>. Rest volume and shape gradients assemble both onto particles. Fixed or zero-mobility vertices are excluded so their constraint reactions are not counted as solve error.</p>
<p><code>x_hat</code> is the native solver’s inertial target, containing incoming velocity, gravity, and external force. Its float32 value and the accepted float32 positions are converted to float64 for residual evaluation. This evaluates the discrete objective the solver actually sees. It also exposes float32 position-resolution limits at small timesteps.</p>
<p>Measurement happens after <em>every</em> solver call, including all substeps. Each mode starts from the same model and follows its own continuous trajectory without resets. Corresponding step indices therefore share the prescribed loading schedule but need not have identical incoming states. This is a rollout comparison, not a same-state single-step contraction experiment.</p>
<div class="note caution"><strong>Interpret the relative residual with its denominator.</strong> Near equilibrium, the initial residual can be small, making R_rel noisy or greater than one. The force RMS in newtons is the primary comparison; the CSVs also expose initial L2, elastic, damping, and inertial force norms. No active contact was observed in any recorded step. Surface triangles and bending edges have zero material coefficients.</div>
<p>Validation: the independent CUDA float64 force calculation matches NumPy, and the finite-difference energy-gradient relative error is {provenance["validation"]["energy_gradient_relative_error"]:.3g}. All source-model fingerprints, timestep sizes, mesh sizes, iteration counts, and recorded step counts match between modes. Instrumented and plain native stepping produce bitwise-identical positions and velocities in six additional checks: extension and stretch for four frames, and compression through release at frame 152, each with ALM off and on. <a href="instrumentation-validation.json">Instrumentation check results</a>.</p></section>
"""
    tail = f"""<section id="reproduce"><h2>Sources and reproduction</h2><p>PR fixtures: <a href="https://github.com/AnkaChan/newton/tree/{provenance["pr_sha"]}"><code>{provenance["pr_sha"][:12]}</code></a>. Solver source: <code>{provenance["solver_revision"]}</code> (the walkthrough commit; solver code matches the reviewed ALM snapshot). Warp {provenance["warp_version"]} on {html.escape(provenance["device"])}.</p>
<pre>source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh alm-pr2901 compete
PYTHONPATH="$PWD" uv run --no-sync python notes/alm-pr2901/run_comparison.py
uv run --no-sync python notes/alm-pr2901/build_report.py</pre>
<p>The runner and residual evaluator live beside this results directory. The six exact PR fixture files are retained under <code>../sources/</code>. Instrumentation is read-only with respect to the simulation state. Instrumented wall durations include residual evaluation and host checks and must not be used as solver speed benchmarks.</p><p><a href="provenance.json">Provenance and residual definition</a> · <a href="comparison.json">Comparison data</a></p><button onclick="window.print()">Print / save PDF</button></section><p class="small">Prepared for Anka · Local HTML report · No external scripts or font dependencies.</p></main></body></html>"""
    (RESULTS / "comparison.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    (RESULTS / "index.html").write_text(head + "".join(sections) + tail)
    print(
        f"Wrote {RESULTS / 'index.html'}; {all_steps} steps; {passed}/12 checks pass; {wins}/10 mean RMS improvements"
    )
    for row in comparisons:
        print(
            f"{row['name']:24s} mean ratio {row['mean_rms_ratio_on_off']:.5f}, median ratio {row['median_rms_ratio_on_off']:.5f}"
        )


if __name__ == "__main__":
    main()
