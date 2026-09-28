# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Convert notes/fusion-integrator-derivation.md to a standalone MathJax HTML page.

Deliberately minimal Markdown: headings, paragraphs, bullet lists, pipe tables,
'> ' theorem blocks, **bold**, *italic*, `code`, and TeX math in $...$ / $$...$$.
Math is extracted before any other processing so that no Markdown rule can touch it.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

SRC = Path(__file__).with_name("fusion-integrator-derivation.md")
OUT = Path(__file__).with_name("fusion-integrator-derivation.html")

MATHJAX_SRC = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"

CSS = """
:root{--ink:#1c2b36;--muted:#5d6f7c;--rule:#d9e2e8;--accent:#1f5f85;--thm:#f3f7fa;--eqbg:transparent}
html{font-size:17px}
body{margin:0;padding:0 1.25rem 5rem;color:var(--ink);background:#fff;
  font-family:Charter,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;line-height:1.6}
main{max-width:60em;margin:0 auto}
header{padding:3rem 0 1.5rem;border-bottom:1px solid var(--rule);margin-bottom:2rem}
header h1{font-size:2rem;line-height:1.2;letter-spacing:-.01em;margin:0 0 .6rem}
header .sub{font-size:1.15rem;color:var(--muted);margin:0 0 1rem}
header .meta{font-size:.92rem;color:var(--muted)}
header .meta code{font-size:.88em}
h2{font-size:1.45rem;margin:2.8rem 0 1rem;padding-top:.4rem;border-top:1px solid var(--rule);letter-spacing:-.01em}
h3{font-size:1.15rem;margin:2rem 0 .7rem}
p{margin:0 0 1rem}
nav.toc{background:var(--thm);border:1px solid var(--rule);border-radius:8px;padding:1rem 1.4rem;margin:0 0 2.2rem}
nav.toc h2{border:0;margin:0 0 .5rem;padding:0;font-size:1.05rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
nav.toc ol{margin:0;padding-left:.2rem;list-style:none}
nav.toc ol ol{padding-left:1.6rem;margin:.15rem 0 .3rem;list-style:none}
nav.toc li{margin:.15rem 0}
nav.toc a{color:var(--accent);text-decoration:none}
nav.toc a:hover{text-decoration:underline}
a{color:var(--accent)}
.eq{margin:.9rem 0 1.1rem;overflow-x:auto;overflow-y:hidden;padding:.15rem 0}
.thm{background:var(--thm);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:.85rem 1.2rem;margin:1.3rem 0}
.thm p{margin:0 0 .6rem}
.thm p:last-child{margin-bottom:0}
.thm .eq{margin:.5rem 0}
ul{padding-left:1.4rem;margin:0 0 1.1rem}
li{margin:.35rem 0}
table{border-collapse:collapse;width:100%;margin:1.2rem 0 1.6rem;font-size:.93rem;line-height:1.45}
th,td{border:1px solid var(--rule);padding:.5rem .65rem;vertical-align:top;text-align:left}
th{background:var(--thm);font-weight:600}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em;background:#f4f6f8;padding:.05em .3em;border-radius:4px}
strong{font-weight:650}
footer{margin-top:4rem;padding-top:1rem;border-top:1px solid var(--rule);font-size:.9rem;color:var(--muted)}
mjx-container[display="true"]{margin:.4em 0 !important}
mjx-container[display="true"]{overflow-x:auto}
@media (max-width:640px){html{font-size:16px}header h1{font-size:1.6rem}h2{font-size:1.3rem}}
"""


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return "sec-" + s


def main() -> None:
    src = SRC.read_text(encoding="utf-8")

    display: list[str] = []
    inline: list[str] = []

    def d_repl(m: re.Match) -> str:
        display.append(m.group(1).strip())
        return f"\x00D{len(display) - 1}\x00"

    def i_repl(m: re.Match) -> str:
        inline.append(m.group(1))
        return f"\x00I{len(inline) - 1}\x00"

    src = re.sub(r"\$\$(.+?)\$\$", d_repl, src, flags=re.S)
    src = re.sub(r"\$([^$\n]+?)\$", i_repl, src)
    if "$" in src:
        bad = [ln for ln in src.splitlines() if "$" in ln]
        sys.exit("unbalanced or stray $ delimiters in:\n" + "\n".join(bad))

    def fmt(text: str) -> str:
        """Inline formatting on math-free text: escape, then bold/italic/code."""
        t = html.escape(text, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
        return t

    def restore(t: str) -> str:
        t = re.sub(r"\x00I(\d+)\x00", lambda m: "$" + html.escape(inline[int(m.group(1))], quote=False) + "$", t)
        t = re.sub(
            r"\x00D(\d+)\x00",
            lambda m: '<div class="eq">$$' + html.escape(display[int(m.group(1))], quote=False) + "$$</div>",
            t,
        )
        return t

    disp_only = re.compile(r"^\s*\x00D\d+\x00\s*$")

    def paragraph(lines: list[str]) -> str:
        """A paragraph; display placeholders standing alone become blocks, not <p>."""
        out: list[str] = []
        buf: list[str] = []

        def flush() -> None:
            if buf:
                out.append("<p>" + fmt(" ".join(buf)) + "</p>")
                buf.clear()

        for ln in lines:
            if disp_only.match(ln):
                flush()
                out.append(ln.strip())
            else:
                buf.append(ln.strip())
        flush()
        return "\n".join(out)

    body: list[str] = []
    toc: list[tuple[int, str, str]] = []
    title = ""
    subtitle = ""
    meta: list[str] = []

    lines = src.splitlines()
    i = 0
    n = len(lines)
    seen_first_h2 = False
    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            title = ln[2:].strip()
            i += 1
            continue
        if ln.startswith("## ") or ln.startswith("### "):
            level = 2 if ln.startswith("## ") else 3
            text = ln[level + 1 :].strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\.?\s+(.*)$", text)
            if m:
                anchor = "sec-" + m.group(1).replace(".", "-")
            else:
                anchor = slug(text)
            toc.append((level, anchor, text))
            body.append(f'<h{level} id="{anchor}">{fmt(text)}</h{level}>')
            seen_first_h2 = True
            i += 1
            continue
        if not seen_first_h2:
            # front matter: subtitle, then meta lines
            if not subtitle:
                subtitle = ln.strip()
            else:
                meta.append(ln.strip())
            i += 1
            continue
        if ln.startswith("> "):
            block: list[str] = []
            while i < n and lines[i].startswith(">"):
                block.append(lines[i][1:].lstrip(" "))
                i += 1
            body.append('<div class="thm">\n' + paragraph(block) + "\n</div>")
            continue
        if ln.startswith("- "):
            items: list[str] = []
            while i < n and lines[i].startswith("- "):
                item = [lines[i][2:]]
                i += 1
                while i < n and lines[i].startswith("  ") and lines[i].strip():
                    item.append(lines[i].strip())
                    i += 1
                items.append("<li>" + fmt(" ".join(item)) + "</li>")
            body.append("<ul>\n" + "\n".join(items) + "\n</ul>")
            continue
        if ln.startswith("|"):
            rows: list[list[str]] = []
            while i < n and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            head, sep, data = rows[0], rows[1], rows[2:]
            assert all(re.fullmatch(r":?-+:?", c) for c in sep), sep
            th = "".join(f"<th>{fmt(c)}</th>" for c in head)
            trs = "\n".join("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in r) + "</tr>" for r in data)
            body.append(f"<table>\n<thead><tr>{th}</tr></thead>\n<tbody>\n{trs}\n</tbody>\n</table>")
            continue
        # paragraph
        para: list[str] = []
        while i < n and lines[i].strip() and not re.match(r"^(#{1,3} |> |- |\|)", lines[i]):
            para.append(lines[i])
            i += 1
        body.append(paragraph(para))

    # table of contents (nested h3 under h2)
    toc_html = ['<nav class="toc" aria-label="Contents"><h2>Contents</h2><ol>']
    open_sub = False
    for level, anchor, text in toc:
        label = fmt(re.sub(r"^\d+(?:\.\d+)?\.?\s+", "", text))
        num = re.match(r"^(\d+(?:\.\d+)?)", text)
        shown = (num.group(1) + "&nbsp; " if num else "") + label
        if level == 2:
            if open_sub:
                toc_html.append("</ol></li>")
                open_sub = False
            else:
                if len(toc_html) > 1:
                    toc_html.append("</li>")
            toc_html.append(f'<li><a href="#{anchor}">{shown}</a>')
        else:
            if not open_sub:
                toc_html.append("<ol>")
                open_sub = True
            toc_html.append(f'<li><a href="#{anchor}">{shown}</a></li>')
    if open_sub:
        toc_html.append("</ol></li>")
    else:
        toc_html.append("</li>")
    toc_html.append("</ol></nav>")

    content = restore("\n".join(body))
    toc_str = restore("\n".join(toc_html))
    head_title = restore(fmt(title))
    head_sub = restore(fmt(subtitle))
    head_meta = restore("".join(f"<p>{fmt(m)}</p>" for m in meta))

    page = f"""<!doctype html>
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="icon" href="data:,">
<style>{CSS}</style>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    tags: 'none'
  }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
}};
</script>
<script defer src="{MATHJAX_SRC}"></script>
</head>
<body>
<main>
<header>
<h1>{head_title}</h1>
<p class="sub">{head_sub}</p>
<div class="meta">{head_meta}</div>
</header>
{toc_str}
{content}
<footer>Mathematics note for the LIDO learned intrinsic solver &middot; source: <code>notes/fusion-integrator-derivation.md</code> &middot; equations rendered by MathJax 3.</footer>
</main>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(display)} display, {len(inline)} inline math segments; {len(toc)} headings")


if __name__ == "__main__":
    main()
