# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the source-linked HTML walkthrough from a fixed implementation snapshot."""

import argparse
import ast
import hashlib
import html
import json
import re
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

REVISION = "496a3675f0fed5005d897250691a592b1122de45"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = "experiments/learned_intrinsic_solver/"
SOURCE_MAP = (
    ("network.py", "IntrinsicTransformerLayer", "LayerNorm, FiLM, attention, edge bias/value, and residual MLP."),
    ("network.py", "IntrinsicSolverNetwork", "Input encoders, transformer stack, correction head, and object step."),
    ("network_geometry.py", "build_grid_neighborhood", "Fixed 27-slot topology and masks for missing neighbors."),
    ("network_geometry.py", "build_edge_features", "Transport neighbor geometry into the receiving cell's frame."),
    ("mixed_physics.py", "MixedHexSolverStep", "Material contexts, features, learned query, and physical advance."),
    ("hex_energy.py", "HexImplicitEulerLoss", "Hex quadrature, mass assembly, elasticity, inertia, and damping."),
    ("damping.py", "damping_metric_difference", "Physical-start metric differences and six-value feature packing."),
    ("fusion.py", "_FusionSolve", "Custom autograd forward solve and transposed backward solve."),
    ("fusion.py", "HexFusion", "Full-quadrature incremental fitting and prescribed-corner constraints."),
    ("pardiso.py", "PardisoFactor", "Cached CPU sparse factorization and repeated forward/transpose solves."),
    (
        "rigid_predictor.py",
        "RigidPosePredictor",
        "Reduce the physical shape to rigid state and call Newton integration.",
    ),
    ("data.py", "generate_cuboid", "Canonical rest corners, cells, and material-grid topology."),
    ("multiscale.py", "generate_multiscale", "Automatic control-grid hierarchy and compatible shared-corner fields."),
    ("initial_state.py", "InitialStateAugmenter", "Deterministic reset, shape/velocity sampling, and pinned corners."),
    ("material_sampling.py", "sample_material", "Seeded E, Poisson ratio, density, and absolute viscosity."),
    ("train_mixed.py", "local_objective", "Normalized energy change and uphill penalty for one proposal."),
    ("train_mixed.py", "run_training", "Mixed batches, backward, Adam, detached boundaries, and checkpoints."),
    ("trajectory_pool.py", "ActiveTrajectoryPool", "Blend different iteration and timestep ages in the active pool."),
    ("hex_validity.py", "HexFeasibility", "Optional sampled-geometry acceptance and detached step shortening."),
)


class Links(HTMLParser):
    """Collect local navigation and line anchors for build-time validation."""

    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            if values["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {values['id']}")
            self.ids.add(values["id"])
        for name in ("href", "src"):
            if values.get(name):
                self.links.append(values[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated/code-walkthrough")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read_source(name):
        if not re.fullmatch(r"[a-z_]+\.py", name):
            raise ValueError(f"Unexpected source path: {name}")
        if name not in sources:
            sources[name] = subprocess.check_output(["git", "show", f"{REVISION}:{PREFIX}{name}"], cwd=ROOT, text=True)
        return sources[name]

    def source_lines(name, start, end, *, identifiers=False):
        lines = read_source(name).splitlines()
        if not 1 <= start <= end <= len(lines):
            raise ValueError(f"Invalid source excerpt: {name}:{start}:{end}")
        rendered = []
        for number, line in enumerate(lines[start - 1 : end], start):
            identity = f' id="L{number}"' if identifiers else ""
            target = f"#L{number}" if identifiers else f"source/{name}.html#L{number}"
            window = "" if identifiers else ' target="_blank" rel="noopener"'
            rendered.append(
                f'<span class="code-line"{identity}>'
                f'<a class="line-number" href="{target}"{window}>{number}</a>'
                f'<span class="code-text">{html.escape(line)}</span></span>'
            )
        return "\n".join(rendered)

    def excerpt(match):
        name, start, end = match.group(1), int(match.group(2)), int(match.group(3))
        return (
            f'<details class="code-excerpt" data-code-source="{name}:{start}:{end}">'
            f"<summary>Read the implementation · <code>{name}</code> · lines {start}&ndash;{end}</summary>"
            f'<div class="code-bar"><a href="source/{name}.html#L{start}" target="_blank" rel="noopener">'
            'Complete source ↗</a><button type="button" data-copy>Copy code</button></div>'
            f"<pre><code>{source_lines(name, start, end)}</code></pre></details>"
        )

    page = (HERE / "page.html").read_text()
    for key in ("network", "energy", "frames", "data"):
        fragment = (HERE / f"{key}.html").read_text()
        fragment = re.sub(r"<h2>(0[123]) (.*?)</h2>", r'<h2><span class="chapter">\1</span>\2</h2>', fragment)
        page = page.replace(f"@@{key.upper()}@@", fragment)
    page = page.replace("@@STYLE@@", (HERE / "style.css").read_text())
    page = page.replace("@@SCRIPT@@", (HERE / "main.js").read_text())
    page = page.replace("@@REVISION@@", REVISION[:8])
    page = page.replace("@@CONFIG@@", html.escape((HERE / "campaign-config.json").read_text()))
    page = re.sub(r'<div data-code="([a-z_]+\.py):(\d+):(\d+)"></div>', excerpt, page)

    rows = []
    for name, symbol, description in SOURCE_MAP:
        node = next(node for node in ast.walk(ast.parse(read_source(name))) if getattr(node, "name", None) == symbol)
        rows.append(
            f'<tr><td><a href="source/{name}.html#L{node.lineno}"><code>{symbol}</code></a>'
            f"<br><small>{name}</small></td><td>{html.escape(description)}</td></tr>"
        )
    source_map = (
        '<div class="table-wrap"><table class="source-map"><thead><tr><th>Entry point</th>'
        f"<th>Responsibility</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    page = page.replace("@@SOURCE_MAP@@", source_map)
    page = re.sub(r'<a href="(source/[^\"]+)"(?! target=)', r'<a href="\1" target="_blank" rel="noopener"', page)
    if "@@" in page or "data-code=" in page:
        raise ValueError("Unexpanded page placeholder")
    for name in re.findall(r"source/([a-z_]+\.py)\.html", page):
        read_source(name)
    for name in re.findall(r'source: "([a-z_]+\.py)#L\d+"', page):
        read_source(name)
    (output / "index.html").write_text(page)
    (output / "source").mkdir(exist_ok=True)
    source_style = """
body{margin:0;background:#f5f8f7;color:#18333b;font:15px/1.6 system-ui}
header{padding:24px 28px;border-bottom:1px solid #d8e4e4;background:#fff}
h1{font-size:24px;margin:10px 0}a{color:#086d73}header p{margin:5px 0;font-size:13px}
pre{margin:0;padding:20px 12px;overflow:auto;background:#142c34;color:#d7e7e9;font:13px/1.7 ui-monospace,Consolas,monospace}
.code-line{display:block;min-height:1.7em;width:max-content;min-width:100%;scroll-margin-top:16px}
.code-line:target{background:#315052;outline:1px solid #679f98}.line-number{position:sticky;left:0;display:inline-block;width:48px;text-align:right;padding-right:16px;user-select:none;background:#142c34;color:#89abae;text-decoration:none}
@media(max-width:600px){header{padding:18px}pre{font-size:12px}.line-number{width:38px;padding-right:10px}}
"""
    for name, contents in sources.items():
        github = f"https://github.com/AnkaChan/newton/blob/{REVISION}/{PREFIX}{name}"
        source_page = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">'
            f"<title>{name} · LIDO source</title><style>{source_style}</style></head><body><header>"
            '<a href="../index.html#implementation-map">← Code walkthrough</a>'
            f"<h1>{name}</h1><p>{PREFIX}{name} · snapshot {REVISION[:8]}</p>"
            f'<p><a href="{github}">Pinned GitHub source</a> · <a href="{name}" download>Download Python</a></p>'
            f"</header><pre><code>{source_lines(name, 1, len(contents.splitlines()), identifiers=True)}</code></pre>"
            "</body></html>"
        )
        (output / "source" / f"{name}.html").write_text(source_page)
        (output / "source" / name).write_text(contents)
    manifest = {
        "implementation_revision": REVISION,
        "implementation_root": PREFIX,
        "campaign_config": json.loads((HERE / "campaign-config.json").read_text()),
        "source_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sorted(sources.items())},
        "chapters": ["network", "energy", "frames", "augmentation"],
        "excerpt_count": page.count('class="code-excerpt"'),
    }
    (output / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output / "walkthrough.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "walkthrough.zip":
                archive.write(path, path.relative_to(output))

    documents = {}
    for path in output.rglob("*.html"):
        parsed = Links()
        parsed.feed(path.read_text())
        documents[path.resolve()] = parsed
    checked = 0
    for path, document in documents.items():
        for link in document.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.is_relative_to(output) or not target.exists():
                raise ValueError(f"Broken local link in {path}: {link}")
            if parsed.fragment and target in documents and parsed.fragment not in documents[target].ids:
                raise ValueError(f"Broken anchor in {path}: {link}")
            checked += 1
    print(
        json.dumps(
            {
                "output": str(output),
                "source_files": len(sources),
                "local_links_checked": checked,
                "excerpts": manifest["excerpt_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
