#!/usr/bin/env python3
"""Build a portable, browser-readable snapshot of the ALM handoff.

Run with: uv run --no-sync --with markdown python notes/alm-handoff/build_handoff.py
"""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import zipfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".cache/alm-handoff-artifact"
SNAPSHOT = "eba5e1a5"
BASE = "0d101f20326d65ea28f8a3d0345ae6c5829898df"
REPORTS = {
    "notes/alm-code-walkthrough": "alm-stage1-code-walkthrough-246fb405-20260914",
    "notes/alm-elasticity-derivation": "alm-elasticity-derivation-20260914",
    "notes/svd-exploration": "alm-svd-exploration-20260914",
    "notes/alm-contact-design": "alm-contact-design-20260914",
}
SOURCES = (
    "newton/_src/solvers/vbd/solver_vbd.py",
    "newton/_src/solvers/vbd/particle_alm_kernels.py",
    "newton/_src/solvers/vbd/particle_vbd_kernels.py",
    "newton/_src/solvers/vbd/rigid_vbd_kernels.py",
    "newton/_src/geometry/tri_mesh_collision.py",
    "newton/tests/test_particle_alm_kernels.py",
    "newton/tests/test_solver_vbd_alm.py",
    "newton/tests/test_solver_vbd.py",
    "notes/stage1_elasticity_alm.md",
    "notes/stage1_elasticity_alm_benchmark.py",
    "notes/stage1_elasticity_alm_results.json",
    "AGENTS.md",
    "CODING_GUIDELINES.rst",
    "REVIEW_GUIDELINES.rst",
)
STYLE = """+:root{color-scheme:light;--ink:#18242c;--muted:#50636d;--link:#075f88;--line:#dce5e9}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f7;
color:var(--ink);font:17px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1050px;margin:36px auto;background:white;padding:38px 48px;border:1px solid var(--line);
border-radius:12px}h1{font-size:2.15rem;line-height:1.2;letter-spacing:-.035em;margin-top:12px}
h2{font-size:1.35rem;line-height:1.3;margin-top:2.5rem;padding-top:1.1rem;border-top:1px solid var(--line)}
a{color:var(--link);text-underline-offset:3px;overflow-wrap:anywhere}p,li{overflow-wrap:anywhere}
li{margin:.35rem 0}.eyebrow{font-size:.8rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
pre{background:#132733;color:#e1eef4;padding:18px;border-radius:7px;overflow:auto;font-size:13px;line-height:1.55}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.86em}
:not(pre)>code{background:#edf2f5;padding:.1em .25em;border-radius:3px;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-size:.94rem;margin:1.3rem 0}th,td{text-align:left;
vertical-align:top;padding:12px;border:1px solid var(--line);overflow-wrap:anywhere}th{background:#edf3f6}
blockquote{border-left:4px solid #bb7b16;padding:8px 18px;margin:1rem 0;background:#fff8e9}
.toc{font-size:.92rem;columns:2}.toc ul{padding-left:1.2rem}.toc>ul>li{list-style:none}
.toc>ul>li>ul{padding-left:0}details{padding:14px 18px;background:#f4f7f9;border-radius:7px;margin:1rem 0}
.source-line{display:block;white-space:pre;min-height:1.55em}.source-line:target{background:#435438}
.line-number{display:inline-block;width:5em;padding-right:1em;text-align:right;color:#98b6c7;text-decoration:none}
.source-pre{padding:12px 8px}.source-main{max-width:1500px}.source-header{position:sticky;top:0;background:white;padding:12px 0}
@media(max-width:650px){body{font-size:16px}main{margin:0;padding:24px 17px;border:0;border-radius:0}
h1{font-size:1.8rem}.toc{columns:1}th,td{padding:8px;font-size:.9rem}pre{padding:12px}}
"""


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def page(title: str, body: str, *, source: bool = False) -> str:
    main_class = ' class="source-main"' if source else ""
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body><main{main_class}>{body}</main></body></html>\n"
    )


def write_source(relative: str, content: bytes) -> None:
    target = OUTPUT / "source" / relative
    write(target, content)
    back = "../" * len(Path(relative).parts) + "index.html"
    text = content.decode("utf-8")
    header = (
        f'<div class="source-header"><a href="{back}">Back to takeover note</a> · '
        f'<a href="{html.escape(target.name)}">Download raw file</a>'
        f"<h1>{html.escape(relative)}</h1></div>"
    )
    if target.suffix == ".md":
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    else:
        lines = []
        for index, line in enumerate(text.splitlines(), 1):
            lines.append(
                f'<span class="source-line" id="L{index}">'
                f'<a class="line-number" href="#L{index}">{index}</a>{html.escape(line)}</span>'
            )
        body = '<pre class="source-pre"><code>' + "".join(lines) + "</code></pre>"
    write(target.with_name(target.name + ".html"), page(relative, header + body, source=True))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for source, slug in REPORTS.items():
        paths = git_bytes("ls-tree", "-r", "--name-only", SNAPSHOT, "--", source).decode().splitlines()
        for relative in paths:
            destination = OUTPUT / "reports" / slug / Path(relative).relative_to(source)
            content = git_bytes("show", f"{SNAPSHOT}:{relative}")
            if source == "notes/alm-contact-design" and destination.name == "index.html":
                banner = (
                    '<aside style="padding:18px;background:#fff2cc;color:#3a3014;font:16px/1.5 system-ui">'
                    "Historical design: some choices were superseded. "
                    '<a href="../../index.html#latest-contact-decisions">Read the latest contact decisions</a>'
                    " before implementing.</aside>"
                )
                content = content.replace(b"<body>", ("<body>" + banner).encode(), 1)
            write(destination, content)

    for relative in SOURCES:
        write_source(relative, git_bytes("show", f"{SNAPSHOT}:{relative}"))
    formulas = Path("/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/learn-alm-dat/alm_formulas.md")
    write_source("formulas/alm_formulas.md", formulas.read_bytes())

    text = (Path(__file__).parent / "handoff.md").read_text()
    write(OUTPUT / "handoff.md", text)
    document = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
    body = document.convert(text)
    navigation = f"<details><summary>Contents</summary>{document.toc}</details>"
    body = body.replace("</h1>", "</h1>" + navigation, 1)
    write(
        OUTPUT / "index.html",
        page(
            "ALM elasticity and contact — takeover note",
            '<div class="eyebrow">Newton / VBD · 25 September 2026</div>' + body,
        ),
    )
    write(OUTPUT / "branch-changes.patch", git_bytes("diff", "--binary", f"{BASE}..{SNAPSHOT}", "--"))
    write(OUTPUT / "branch-commits.txt", git_bytes("log", "--format=%H %s", f"{BASE}..{SNAPSHOT}"))
    archive_name = "alm-handoff-20260924.zip"
    files = [
        path
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name not in {archive_name, "manifest.json"}
    ]
    manifest = {
        "snapshot": git_bytes("rev-parse", SNAPSHOT).decode().strip(),
        "baseline": BASE,
        "date": "2026-09-25",
        "files": {str(path.relative_to(OUTPUT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    write(OUTPUT / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(OUTPUT / archive_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, OUTPUT / "manifest.json"]:
            archive.write(path, arcname=str(path.relative_to(OUTPUT)))
    print(f"Built {OUTPUT}")
    print(f"Packaged {len(files) + 1} files; archive {(OUTPUT / archive_name).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
