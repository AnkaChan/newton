# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the elasticity derivation with embedded-glyph SVG equations.

Run: uv run --no-sync --with markdown python notes/alm-elasticity-derivation/build_html.py
"""

import re
from html import escape
from pathlib import Path

import markdown


def main():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    matplotlib.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "alm-elasticity-derivation"})
    folder = Path(__file__).parent
    source = (folder / "derivation.md").read_text()
    pattern = r"<!-- eq:([a-z-]+)\|([^\n]+) -->\n\$\$\n(.*?)\n\$\$"
    blocks = list(re.finditer(pattern, source, re.S))
    for number, block in reversed(list(enumerate(blocks, start=1))):
        name, alt, formula = block.groups()
        svg = folder / f"{name}.svg"
        math_to_image("$" + formula + "$", svg, prop=FontProperties(size=20), format="svg", color="#19313c")
        svg_text = re.sub(r"<dc:date>.*?</dc:date>", "", svg.read_text())
        svg.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
        width = float(re.search(r'<svg[^>]*width="([\d.]+)pt"', svg_text).group(1)) * 1.12
        html = (
            f'<figure class="equation" id="{name}-formula">'
            f'<div class="math-scroll" tabindex="0" aria-label="Equation {number}">'
            f'<img src="{name}.svg" style="width:{width:.1f}px" alt="{escape(alt, quote=True)}">'
            f"</div><figcaption>({number})</figcaption></figure>"
        )
        source = source[: block.start()] + html + source[block.end() :]
    if "$$" in source:
        raise ValueError("Every display equation needs an eq annotation")
    converter = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "md_in_html"])
    content = converter.convert(source)
    content = content.replace("</h1>", "</h1><details><summary>Contents</summary>" + converter.toc + "</details>", 1)
    style = (folder / "style.css").read_text()
    page = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Neo-Hookean → 9 + 1 → SVD · ALM derivation</title><style>" + style + "</style></head><body><main>"
        '<div class="eyebrow">Newton / VBD · Elasticity derivation · 14 September 2026</div>'
        '<nav><a href="../alm-svd-exploration-20260914/index.html">SVD exploration</a>'
        '<a href="derivation.md">Download Markdown and LaTeX source</a></nav>' + content + "</main></body></html>\n"
    )
    (folder / "index.html").write_text(page)
    print(f"Rendered {len(blocks)} equations into {folder / 'index.html'}")


if __name__ == "__main__":
    main()
