# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the local video viewer, summary, and combined MP4 after the sweep."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
STIFFNESSES = [1000, 10000, 100000, 1000000, 10000000]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = []
    for stiffness in STIFFNESSES:
        pair = [json.loads((OUT / f"ke{stiffness}_{mode}.json").read_text()) for mode in ("off", "on")]
        assert pair[0]["model_sha256"] == pair[1]["model_sha256"]
        for result in pair:
            assert result["frames"] == 360 and result["solver_steps"] == 3600
            assert result["params"]["cloth_edge_ke"] == 200
            assert result["params"]["cloth_tri_ka"] == stiffness * 0.2
            assert result["max_pin_error_m"] < 1e-6
            assert len(result["rows"]) == 361
            assert np.isfinite([list(row.values()) for row in result["rows"]]).all()
        cases.append(pair)

    segments = OUT / "segments.txt"
    segments.write_text("".join(f"file 'ke{value}_comparison.mp4'\n" for value in STIFFNESSES))
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(segments),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUT / "alm_bag_stiffness_comparison.mp4"),
        ],
        check=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for axis, key, title, unit in zip(
        axes,
        ("stretch_score", "shear_score", "bend_score"),
        ("Stretch", "Triangle angle change", "Bend"),
        ("relative edge change", "rad", "rad"),
        strict=True,
    ):
        for index, mode, color in ((0, "ALM off", "#c46b2d"), (1, "ALM on (bending)", "#008f94")):
            axis.semilogx(
                STIFFNESSES, [pair[index]["summary"][key]["mean"] for pair in cases], "o-", color=color, label=mode
            )
        axis.set(title=title, xlabel="Triangle stiffness tri_ke", ylabel=f"Mean RMS ({unit})")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "mean_deformation.svg")
    plt.close(fig)
    svg = OUT / "mean_deformation.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")

    rows = []
    summaries = []
    for stiffness, (off, on) in zip(STIFFNESSES, cases, strict=True):
        summary = {"stiffness": stiffness}
        values = []
        for key in ("stretch_score", "bend_score"):
            a, b = off["summary"][key]["mean"], on["summary"][key]["mean"]
            summary[key] = {"off": a, "on": b, "on_over_off": b / a}
            scale = 100.0 if key == "stretch_score" else 1.0
            values.extend([f"{a * scale:.4f}", f"{b * scale:.4f}", f"{b / a:.2f}x"])
        rows.append(f"<tr><td>{stiffness:.0e}</td>" + "".join(f"<td>{x}</td>" for x in values) + "</tr>")
        summaries.append(summary)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    buttons = "".join(f'<button onclick="jump({i * 6})">tri_ke {s:.0e}</button>' for i, s in enumerate(STIFFNESSES))
    downloads = "".join(
        f'<tr><td>{s:.0e}</td><td><a href="ke{s}_comparison.mp4">Side-by-side MP4</a></td>'
        f'<td><a href="ke{s}_off.csv">Off CSV</a> · <a href="ke{s}_on.csv">On CSV</a></td>'
        f'<td><a href="ke{s}_off.json">Off metadata</a> · <a href="ke{s}_on.json">On metadata</a></td></tr>'
        for s in STIFFNESSES
    )
    first = cases[0][0]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pinned bag · ALM on / off</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101823;color:#eaf0f7;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:30px 24px}}h1{{font-size:32px;margin:0}}h2{{font-size:22px}}
p{{max-width:1100px}}a{{color:#77dedd}}.muted{{color:#b7c6d8}}.notice{{border-left:4px solid #e7ae64;padding:10px 18px;background:#242a32}}
video{{width:100%;background:#121925;border:1px solid #415165;border-radius:8px}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}
button{{color:#eaf0f7;background:#253647;border:1px solid #617c96;padding:10px 18px;border-radius:6px;cursor:pointer}}
button:hover{{background:#36546b}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:10px;border-bottom:1px solid #364657}}th:first-child,td:first-child{{text-align:left}}
img{{width:100%;background:white;border-radius:8px}}code{{color:#97dbde}}details{{margin-top:24px}}
@media(max-width:600px){{main{{padding:20px 12px}}h1{{font-size:26px}}td,th{{padding:7px;font-size:13px}}}}
</style></head><body><main>
<p class="muted">PINNED BAG RIGIDITY SWEEP · LOCAL COMPARISON</p><h1>ALM off / ALM on</h1>
<p>Five stiffnesses, identical initial models and prescribed motion. Each pair runs for 6 seconds:
1 second settling, then 5 seconds wiggling. Both use <strong>10 substeps &times; 10 iterations</strong> per frame at 60 fps.</p>
<p class="notice"><strong>ALM affects bending only in this scene.</strong> Triangle membrane elasticity is not implemented in the current ALM branch.
Both panels use the same triangle stretch/area model and contact algorithms. The missing May fixture was reconstructed from its archived parent scene
and saved pin-motion notes; this is not an exact replay of the original scene file.</p>
<nav aria-label="Stiffness chapters">{buttons}</nav>
<video id="comparison" src="alm_bag_stiffness_comparison.mp4" poster="ke1000_frame180.png" controls muted playsinline preload="metadata"></video>
<p class="muted">Left: ALM off. Right: ALM on (dihedral bending, rho scale 1.0, history retained).
Fixed camera and synchronized playback. Each stiffness occupies 6 seconds of the 30-second video.</p>
<p><a href="alm_bag_stiffness_comparison.mp4">Open or download the full MP4</a></p>
<h2>Deformation across the stiffness sweep</h2>
<p>With the current ALM bending implementation, mean bend deformation increases at all five stiffnesses.
At <code>tri_ke = 1e5</code>, it rises from {summaries[2]["bend_score"]["off"]:.4f} to {summaries[2]["bend_score"]["on"]:.4f} rad
({summaries[2]["bend_score"]["on_over_off"]:.1f}x), while mean stretch changes from {100 * summaries[2]["stretch_score"]["off"]:.3f}%
to {100 * summaries[2]["stretch_score"]["on"]:.3f}%. The high-stiffness stretch plateau remains.</p>
<p>These are geometric deformation metrics, <strong>not force residuals or convergence errors</strong>.
Reduced stretch alone does not establish a better solution: changes in bending and contact can redistribute deformation.</p>
<img src="mean_deformation.svg" alt="Mean stretch, triangle angle change, and bend, comparing ALM off and on over the five triangle stiffnesses">
<div class="scroll"><table><thead><tr><th>tri_ke</th><th>Stretch off %</th><th>Stretch on %</th><th>On/off</th>
<th>Bend off rad</th><th>Bend on rad</th><th>On/off</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">Arithmetic means over 361 samples, including the initial rest state and the final state at 6 seconds.</p>
<details open><summary>Settings and reconstruction</summary>
<p><code>tri_ke = 1e3, 1e4, 1e5, 1e6, 1e7</code>; <code>tri_ka = 0.2 &times; tri_ke</code>;
<code>edge_ke = 200</code>; triangle damping 0.1; bending damping 0.02; density 0.08; seed 42.</p>
<p>The 0.22 &times; 0.14 &times; 0.32 m box bag has resolution 18, {first["particle_count"]} vertices,
{first["triangle_count"]} triangles, {first["pin_count"]} pinned rim vertices, and three rigid contents (bear mesh, cone, sphere).
There is no ground or gripper. The rim moves along x with amplitude 0.07 m, frequency 0.85 Hz,
and a 0.6-second linear ramp after the first second. Displacement and analytic velocity are sampled once per frame and applied before every substep.</p>
<p>The parent source is <code>{first["parent_revision"][:12]}</code>, with only its USD import moved into the loading function.
The wrapper removes the ground and omits the gripper, then applies the documented pins and schedule.
Solver source is <code>{first["newton_revision"][:12]}</code>; no solver code was changed for this experiment.
Contact uses the existing self-contact/DAT and legacy rigid contact settings from the parent, identically in both modes.
There are no spring or tetrahedral elements in this model.</p>
<p>All 36,000 solver steps completed with finite saved positions. Paired model hashes match, pins stay within 1 micrometer
of the prescribed locations, and ALM bending multipliers become nonzero. This is an equal-iteration comparison; no GPU speed claim is made.</p>
</details>
<h2>Per-stiffness videos and raw data</h2><div class="scroll"><table><thead><tr><th>tri_ke</th><th>Video</th><th>Per-frame data</th><th>Settings / provenance</th></tr></thead>
<tbody>{downloads}</tbody></table></div>
<p><a href="../README.md">Reproduction commands and source notes</a> · <a href="summary.json">Summary JSON</a></p>
</main><script>function jump(t){{const v=document.getElementById('comparison');v.currentTime=t;v.play().catch(()=>{{}});}}</script></body></html>
"""
    (OUT / "index.html").write_text(html)
    print(json.dumps(summaries, indent=2))
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
