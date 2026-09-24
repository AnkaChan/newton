# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a standalone HTML gallery from the VBD recording manifest."""

import argparse
import html
import json
from pathlib import Path


def build_gallery(directory: Path) -> Path:
    """Write a responsive, dependency-free gallery with relative artifact links."""
    manifest = json.loads((directory / "manifest.json").read_text())
    samples = manifest["samples"]
    cards = []
    for sample in samples:
        metrics = sample["metrics"]
        seed = sample["seed"]
        cards.append(f"""<article class="card" id="seed-{seed}">
<div class="card-heading"><h2>Seed {seed:02d}</h2><span class="pass">✓ verified</span></div>
<video controls playsinline preload="none" poster="{html.escape(sample["poster"])}"
 aria-label="Newton VBD simulation for seed {seed}">
<source src="{html.escape(sample["video"])}" type="video/mp4">
<a href="{html.escape(sample["video"])}">Download video</a></video>
<div class="card-body"><p class="facts"><span>10 s · 300 frames</span><span>Clamp drift: {metrics["clamp_drift_max_m"]:.0e} m</span></p>
<p class="facts"><span>Initial min det F: {metrics["initial_tet_min_volume_ratio"]:.3f}</span>
<span>Trajectory min det F: {metrics["trajectory_tet_min_volume_ratio"]:.3f}</span></p>
<div class="downloads"><a href="{sample["video"]}" download>MP4 ↗</a>
<a href="{sample["initial_state"]}" download>Initial state ↗</a>
<a href="metrics/sample_{seed:02d}.json">Metrics ↗</a></div></div></article>""")
    worst_ratio = min(sample["metrics"]["trajectory_tet_min_volume_ratio"] for sample in samples) if samples else 0
    body = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seeded cantilever samples · Newton VBD</title>
<style>
:root{color-scheme:dark;--bg:#111820;--panel:#1b2631;--border:#32424f;--text:#e8f0f6;--muted:#a8bccb;--accent:#74d6ec;--orange:#ffab61}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:56px 32px}a{color:var(--accent);text-underline-offset:4px}a:hover{color:white}
.eyebrow{font:600 12px/1.4 system-ui;text-transform:uppercase;letter-spacing:.18em;color:var(--accent)}h1{font-size:clamp(32px,4vw,56px);line-height:1.1;letter-spacing:-.035em;margin:14px 0 18px}
.intro{max-width:820px;color:var(--muted);font-size:18px;margin:0 0 28px}.summary{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin:30px 0}
.stat{padding:20px 24px;background:var(--panel)}.stat+.stat{border-left:1px solid var(--border)}.stat strong{display:block;font-size:27px;font-weight:600}.stat span{color:var(--muted);font-size:13px}
.method{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin:24px 0}.method summary{cursor:pointer;font-weight:650}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px 40px;margin-top:18px}.method p{margin:0 0 12px}.method h3{margin:0 0 10px;font-size:15px;color:var(--accent)}
.method table{border-collapse:collapse;width:100%;font-size:14px}.method td{border-bottom:1px solid var(--border);padding:6px 0;vertical-align:top}.method td:first-child{color:var(--muted);padding-right:15px;width:48%}code{font-size:.88em;color:var(--accent)}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:36px 0 18px}.toolbar p{margin:0;color:var(--muted);font-size:14px}.controls{display:flex;gap:12px;align-items:center}button,select{font:inherit;font-size:14px;border:1px solid var(--border);border-radius:7px;background:var(--panel);color:var(--text);padding:9px 12px;cursor:pointer}button:hover,select:hover{border-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.card{border:1px solid var(--border);background:var(--panel);border-radius:12px;overflow:hidden;min-width:0}.card-heading{display:flex;justify-content:space-between;align-items:center;padding:14px 20px}.card h2{font-size:17px;margin:0;font-weight:650}.pass{font-size:12px;color:#9cdfb5}.card video{width:100%;height:auto;aspect-ratio:16/9;display:block;background:#252b36;object-fit:contain}.card-body{padding:13px 20px 18px}.facts{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:12px;margin:3px 0}.downloads{display:flex;gap:20px;font-size:13px;margin-top:13px}
.legend{display:flex;flex-wrap:wrap;gap:20px;color:var(--muted);font-size:13px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;background:var(--accent)}.dot.orange{background:var(--orange)}footer{margin:38px 0 0;border-top:1px solid var(--border);padding-top:20px;color:var(--muted);font-size:13px}footer p{margin:7px 0}strong{font-weight:650}.links{display:flex;gap:20px;flex-wrap:wrap}
@media(max-width:720px){main{padding:32px 16px}.summary{grid-template-columns:1fr 1fr}.stat{padding:16px}.stat:nth-child(3){border-left:0;border-top:1px solid var(--border)}.stat:nth-child(4){border-top:1px solid var(--border)}.grid,.method-grid{grid-template-columns:1fr}.toolbar{align-items:flex-start;flex-direction:column}.facts{font-size:11px}.downloads{gap:16px}.method{padding:17px}.card-heading{padding:12px 16px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><main>
<div class="eyebrow">Learned intrinsic solver · baseline data generation</div>
<h1>Seeded cantilever samples</h1>
<p class="intro">Twenty initial conditions from the existing cuboid augmenter, simulated with Newton VBD for ten seconds each. The orange end stays fixed; every clip uses the same camera, material, and solver settings.</p>
<div class="summary"><div class="stat"><strong>__COUNT__ seeds</strong><span>Deterministic initial conditions · 0&ndash;19</span></div>
<div class="stat"><strong>10 &times; 10 &times; 40</strong><span>4,000 voxel cells · canonical rest shape</span></div>
<div class="stat"><strong>20,000 tets</strong><span>4,961 shared particles · 121 clamped</span></div>
<div class="stat"><strong>10 seconds</strong><span>30 fps · 300 frames per video</span></div></div>
<div class="legend"><span><i class="dot"></i>Deforming surface with triangle edges</span><span><i class="dot orange"></i>Fixed material z = 0 end</span><span>Newton ViewerGL · fixed camera</span></div>
<details class="method"><summary>Simulation, augmentation, and reproducibility</summary>
<div class="method-grid"><section><h3>Canonical rest model</h3><table>
<tr><td>Grid / cell edge</td><td>10 &times; 10 &times; 40 / 0.025 m</td></tr><tr><td>Material dimensions</td><td>0.25 &times; 0.25 &times; 1.0 m</td></tr>
<tr><td>Young&rsquo;s modulus / &nu;</td><td>500 kPa / 0.3</td></tr><tr><td>Density / damping</td><td>1,000 kg/m³ / 100 Pa·s</td></tr>
<tr><td>Time step / iterations</td><td>1/300 s / 20 VBD iterations</td></tr><tr><td>Substeps per video frame</td><td>10 (3,000 steps per sample)</td></tr>
<tr><td>Gravity</td><td>(0, 0, &minus;9.81) m/s²</td></tr><tr><td>Contact</td><td>No ground or self-contact</td></tr>
<tr><td>Clamp</td><td>Canonical positions, zero inverse mass and velocity</td></tr></table></section>
<section><h3>Seeded initial conditions</h3>
<p>For each cell, <code>F = I + E</code>, with independent entries <code>Eᵢⱼ &sim; U(&minus;0.15, 0.15)</code>. Cell velocity components are sampled in <code>[&minus;0.75, 0.75] m/s</code>. Every sample calls <code>augment_grid(..., seed=k)</code>.</p>
<p>Independent cell deformations are fitted to shared corners using equal-weight least squares on all 12 edges per cell, with the fixed face held at rest. Shared-corner velocities average incident cell velocities. This projection is used only to initialize the simulation.</p>
<p>The canonical cuboid supplies all rest tetrahedral data. Newton&rsquo;s physical implicit-Euler inertia and VBD solver are unchanged. These are baseline simulations; no learned solver is used.</p>
<p>Repeated seeds reproduce the cell fields and projected initial conditions exactly in the recorded environment. The archives store the exact float32 initial positions and velocities passed to Newton. Bitwise GPU trajectory determinism is not promised.</p></section></div>
<p><strong>Validation:</strong> all __COUNT__ clips contain 300 frames and run for 10 seconds. All sampled states are finite, the clamp drift is zero, and every recorded tet stays positively oriented (lowest volume ratio __RATIO__). No initial deformation needed scaling.</p>
<p>Videos show times 1/30 through 10 seconds. Posters show the initial state at time zero. Per-sample metrics, initial states, and the complete manifest are downloadable.</p>
</details>
<div class="toolbar"><p>Play any sample to inspect its transient response.</p><div class="controls"><button id="pause" type="button">Pause all</button>
<label for="speed">Speed</label><select id="speed"><option value="0.25">0.25&times;</option><option value="0.5">0.5&times;</option><option value="1" selected>1&times;</option><option value="2">2&times;</option></select></div></div>
<div class="grid">__CARDS__</div>
<footer><div class="links"><a href="manifest.json">Full manifest</a><a href="verification.json">Verification report</a><a href="reproduce.md">Reproduce these samples</a></div>
<p>Generated with the existing seeded augmenter, Newton SolverVBD, and Newton ViewerGL. All media and data links are relative; this directory can be copied and opened independently.</p></footer>
<script>const videos=[...document.querySelectorAll('video')];document.querySelector('#pause').addEventListener('click',()=>videos.forEach(v=>v.pause()));document.querySelector('#speed').addEventListener('change',e=>videos.forEach(v=>v.playbackRate=Number(e.target.value)));</script>
</main></body></html>"""
    body = (
        body.replace("__COUNT__", str(len(samples)))
        .replace("__RATIO__", f"{worst_ratio:.3f}")
        .replace("__CARDS__", "\n".join(cards))
    )
    output = directory / "index.html"
    output.write_text(body)
    return output


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build_gallery(args.directory))


if __name__ == "__main__":
    _main()
