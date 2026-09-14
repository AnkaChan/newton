# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the contact design and its LaTeX formulas without runtime dependencies.

Run: uv run --no-sync --with markdown python notes/alm-contact-design/build_html.py
"""

import re
from html import escape
from pathlib import Path

import markdown

EQUATIONS = (
    ("contact-constraint", "Normal contact constraint: signed barycentric gap C is nonnegative"),
    ("friction-slip", "Tangential slip C t is the tangent projection of relative barycentric displacement"),
    ("friction-projection", "Tangential trial multiplier is projected onto the Coulomb disk"),
    ("friction-forces", "Friction forces on A and B distribute the resistance multiplier with opposite signs"),
)


def main():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    matplotlib.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "alm-contact-equations"})
    folder = Path(__file__).parent
    source = (folder / "design.md").read_text()
    blocks = list(re.finditer(r"\$\$\n(.*?)\n\$\$", source, re.S))
    if len(blocks) != len(EQUATIONS):
        raise ValueError("Update the equation names when adding or removing display equations")
    for block, (name, alt) in zip(reversed(blocks), reversed(EQUATIONS), strict=True):
        svg = folder / f"{name}.svg"
        math_to_image("$" + block.group(1) + "$", svg, prop=FontProperties(size=22), format="svg", color="#19313c")
        svg_text = re.sub(r"<dc:date>.*?</dc:date>", "", svg.read_text())
        svg.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n")
        html = (
            f'<div class="contact-equation" id="{name}-formula">'
            f'<img src="{name}.svg" alt="{escape(alt, quote=True)}"></div>'
        )
        source = source[: block.start()] + html + source[block.end() :]
    converter = markdown.Markdown(extensions=["fenced_code", "tables", "toc"])
    content = converter.convert(source)
    content = content.replace(
        "</h1>", "</h1><details><summary>Jump to a section</summary>" + converter.toc + "</details>", 1
    )
    style = (folder / "style.css").read_text()
    page = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>ALM contact design · VBD</title><style>" + style + "</style></head><body><main>"
        '<div class="eyebrow">Newton / VBD · Contact design</div>'
        '<a class="source" href="design.md">Download Markdown source</a>' + content + "</main></body></html>\n"
    )
    (folder / "index.html").write_text(page)
    print(f"Rendered {len(EQUATIONS)} equations and contact HTML")


if __name__ == "__main__":
    main()
