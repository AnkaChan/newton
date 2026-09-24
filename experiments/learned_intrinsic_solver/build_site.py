# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bundle the experimental project visualizations into one portable site."""

import argparse
import html
import json
import os
import re
import shutil
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

__all__ = ["build_project_site"]

_PAGES = (
    (
        "vbd_10x10x40",
        "vbd",
        "Newton VBD simulations",
        "20 recorded runs",
        "Twenty seeded initial states, each simulated for 10 seconds. Watch recordings and download the initial geometry and metrics.",
    ),
    (
        "cell_frames",
        "cell-frames",
        "Cell frames and deformation axes",
        "First geometry test",
        "Click the deformed grid to see each cell's local frame on the shape and its deformation axes in a linked window.",
    ),
    (
        "multiscale_frames",
        "multiscale",
        "Multiscale initial shapes",
        "20 seeds · 4 scale views",
        "Explore automatically generated control grids. Compare coarse, middle, fine, and combined deformation, then inspect individual cells.",
    ),
    (
        "round_trip",
        "round-trip",
        "Global shape round trip",
        "Encode · reconstruct · compare",
        "Compare the original shared-corner shape with its reconstruction from local cell data, including measured recovery errors.",
    ),
    (
        "round_trip_100",
        "round-trip-100",
        "100 float32 round trips",
        "Error analysis across 100 seeds",
        "Review reconstruction-error distributions, per-sample metrics, and selected original-versus-reconstructed 3D comparisons from the float32 pipeline.",
    ),
)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if value and name in ("href", "src", "poster"):
                self.paths.append(value)


def _validate_links(site: Path) -> int:
    checked = 0
    for page in site.rglob("*.html"):
        parser = _Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for link in parser.paths:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(site.resolve()):
                raise ValueError(f"Link escapes the project folder: {page.relative_to(site)} -> {link}")
            if not target.exists():
                raise ValueError(f"Missing relative asset: {page.relative_to(site)} -> {link}")
            checked += 1
    return checked


def _index() -> str:
    cards = []
    for _, folder, title, badge, description in _PAGES:
        cards.append(
            f'<a class="card" href="{folder}/index.html"><span class="badge">{html.escape(badge)}</span>'
            f'<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p><span class="open">Open page →</span></a>'
        )
    return (
        """<!doctype html>
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learned intrinsic solver · project pages</title><link rel="icon" href="data:,">
<style>
:root{font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#183044;background:#f5f8fa;line-height:1.55}*{box-sizing:border-box}body{margin:0;padding:52px 24px}main{max-width:1040px;margin:auto}header{margin-bottom:32px}.eyebrow{font-size:11px;letter-spacing:.15em;font-weight:700;color:#657987;margin:0 0 12px}h1{font-size:clamp(30px,5vw,46px);letter-spacing:-.045em;line-height:1.15;margin:0 0 16px}header p:last-child{font-size:16px;color:#657987;max-width:640px;margin:0}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{display:flex;flex-direction:column;align-items:flex-start;text-decoration:none;color:inherit;padding:26px;background:#fff;border:1px solid #dbe6ec;border-radius:13px;transition:border-color .15s,transform .15s}.card:hover{border-color:#80b1c7;transform:translateY(-2px)}.card:focus-visible{outline:3px solid #65a6c2;outline-offset:4px}.badge{font-size:11px;font-weight:600;color:#386780;background:#edf4f7;border-radius:5px;padding:4px 8px}h2{font-size:21px;line-height:1.3;letter-spacing:-.02em;margin:17px 0 8px}.card p{color:#657987;font-size:14px;margin:0 0 24px}.open{margin-top:auto;color:#216389;font-size:13px;font-weight:650}footer{color:#657987;font-size:12px;margin-top:24px}footer a{color:#216389}@media(max-width:650px){body{padding:30px 16px}.cards{grid-template-columns:1fr}.card{padding:22px}h2{font-size:20px}header{margin-bottom:24px}}
</style></head><body><main><header><p class="eyebrow">NEWTON · LEARNED INTRINSIC SOLVER</p><h1>Project visualizations</h1><p>Project experiments in one place. Each page includes its own views, controls, and downloadable data.</p></header><section class="cards" aria-label="Project webpages">"""
        + "\n".join(cards)
        + """</section><footer>Canonical 10 &times; 10 &times; 40 grid · <a href="site-manifest.json">Bundle contents</a></footer></main></body></html>
"""
    )


def build_project_site(generated: Path, *, output: Path | None = None) -> dict:
    """Copy the existing experimental artifacts into one portable folder.

    Args:
        generated: Directory containing the original generated artifacts.
        output: Bundle destination; defaults to ``generated / 'webpages'``.

    Returns:
        Manifest containing each bundled page, its file count, and total size.
        Sources are preserved. Build and validate a temporary tree before
        replacing the previous bundle. No symlinks are created or followed.
    """
    generated = Path(generated).resolve()
    output = Path(output).resolve() if output is not None else generated / "webpages"
    sources = [generated / source for source, *_ in _PAGES]
    if output == generated or generated.is_relative_to(output):
        raise ValueError("output must not replace the generated root or its parent")
    for source in sources:
        if source == output or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output must not overlap an original artifact directory")
        if not (source / "index.html").is_file():
            raise FileNotFoundError(f"Missing original artifact: {source / 'index.html'}")
        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError(f"Refusing to copy symlinks from {source}")
    required = sum(path.stat().st_size for source in sources for path in source.rglob("*") if path.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < required + 64 * 1024**2:
        raise OSError("Insufficient space for the complete temporary site bundle")
    manifest = {"pages": [], "total_bytes": 0, "relative_links_checked": 0}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-build-", dir=output.parent) as temporary:
        staging = Path(temporary) / "site"
        staging.mkdir()
        for source_name, folder, title, _, _ in _PAGES:
            destination = staging / folder
            shutil.copytree(generated / source_name, destination)
            for page in destination.rglob("*.html"):
                text = page.read_text(encoding="utf-8")
                if 'rel="icon"' not in text and "rel='icon'" not in text:
                    text = re.sub(r"(<head[^>]*>)", r'\1<link rel="icon" href="data:,">', text, count=1, flags=re.I)
                if page == destination / "index.html":
                    backlink = '<nav data-project-home style="margin:0 auto 18px;max-width:1700px;font:13px system-ui"><a href="../index.html" style="color:#216389;text-decoration:none">← All project pages</a></nav>'
                    text = re.sub(
                        r"(<body[^>]*>)",
                        lambda match, backlink=backlink: match.group(1) + backlink,
                        text,
                        count=1,
                        flags=re.I,
                    )
                page.write_text(text, encoding="utf-8")
            files = [path for path in destination.rglob("*") if path.is_file()]
            size = sum(path.stat().st_size for path in files)
            manifest["pages"].append(
                {
                    "title": title,
                    "path": f"{folder}/index.html",
                    "source": source_name,
                    "files": len(files),
                    "bytes": size,
                }
            )
            manifest["total_bytes"] += size
        (staging / "index.html").write_text(_index(), encoding="utf-8")
        manifest_path = staging / "site-manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        manifest["relative_links_checked"] = _validate_links(staging)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        previous = Path(temporary) / "previous"
        if output.exists():
            os.replace(output, previous)
        try:
            os.replace(staging, output)
        except OSError:
            if previous.exists():
                os.replace(previous, output)
            raise
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=Path(__file__).parent / "generated")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_project_site(args.generated, output=args.output)
    print(json.dumps({"output": str(args.output or args.generated / "webpages"), **result}, indent=2))


if __name__ == "__main__":
    _main()
