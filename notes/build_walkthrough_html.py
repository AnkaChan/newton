# SPDX-License-Identifier: Apache-2.0
"""Turn an annotated code-walkthrough Markdown into interactive, self-contained HTML.

Portable generator behind the `code-walkthrough-html` skill. Reads a review
Markdown (see SKILL.md for the authoring conventions) plus the live source
files it references, and emits ONE offline HTML file with:

- sidebar TOC with scroll-spy (sections + findings),
- code excerpts carrying *real* source line numbers (multi-range captions and
  `...` elision rows supported),
- every ``path/to/file.py:NN`` mention becomes a click target opening an
  embedded full-source viewer scrolled to (and flashing) that line,
- findings ``**R<n> — title.** body`` rendered as anchored cards,
- hand-rolled Python syntax highlighting in JS — zero network dependencies.

Usage:
    python build_walkthrough_html.py --md docs/review.md [--out review.html]
        [--root <repo-root>] [--embed extra/file.py ...] [--title "..."]

Source files are auto-discovered: every ``*.py:<line>`` reference in the
Markdown is resolved against --root (direct relative path first, then a
unique-basename match via ``git ls-files``). Add --embed for files you want
browsable that the text never references by line.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import subprocess
import sys

FILE_REF = re.compile(r"\b([\w./-]+\.py):(\d+)(?:-(\d+))?\b")
CAPTION = re.compile(r"^\(`([\w./-]+\.py):([\d,\s-]+)`(?:,?\s*(.*?))?\)$")
BUILD_WARNINGS: list[str] = []


def esc(s: str) -> str:
    return html.escape(s, quote=False)


class Resolver:
    """Map file mentions (relpath or basename) to repo-relative paths."""

    def __init__(self, root: pathlib.Path, prefer: list[str] | None = None):
        self.root = root
        self.prefer = prefer or []
        self.resolved: dict[str, str] = {}
        try:
            out = subprocess.run(
                ["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True, check=True
            ).stdout.split()
        except (subprocess.CalledProcessError, FileNotFoundError):
            out = [str(p.relative_to(root)) for p in root.rglob("*.py")]
        self.by_base: dict[str, list[str]] = {}
        for p in out:
            self.by_base.setdefault(pathlib.Path(p).name, []).append(p)

    def resolve(self, name: str) -> str | None:
        if name in self.resolved:
            return self.resolved[name]
        path = None
        if (self.root / name).is_file():
            path = name
        else:
            cands = self.by_base.get(pathlib.Path(name).name, [])
            if len(cands) > 1 and self.prefer:
                preferred = [c for c in cands if any(c.startswith(pref) for pref in self.prefer)]
                if len(preferred) == 1:
                    cands = preferred
            if len(cands) == 1:
                path = cands[0]
            elif len(cands) > 1:
                print(f"warning: ambiguous basename {name!r} ({cands}); use a fuller path or --prefer", file=sys.stderr)
        self.resolved[name] = path
        return path


class Builder:
    def __init__(self, resolver: Resolver):
        self.rz = resolver
        self.used_files: set[str] = set()

    # ---- inline formatting -------------------------------------------------
    def linkify_ref(self, m: re.Match) -> str:
        path = self.rz.resolve(m.group(1))
        if path is None:
            return m.group(0)
        self.used_files.add(path)
        return f'<a class="ref" data-file="{path}" data-line="{m.group(2)}" href="#">{m.group(0)}</a>'

    def _bold(self, s: str) -> str:
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<em>\1</em>", s)
        return FILE_REF.sub(self.linkify_ref, s)

    def inline(self, s: str) -> str:
        s = esc(s)
        out, pos = [], 0
        for m in re.finditer(r"``(.+?)``|`([^`]+)`", s):
            out.append(self._bold(s[pos : m.start()]))
            body = m.group(1) or m.group(2)
            out.append(f"<code>{FILE_REF.sub(self.linkify_ref, body)}</code>")
            pos = m.end()
        out.append(self._bold(s[pos:]))
        return "".join(out)

    # ---- code excerpts -----------------------------------------------------
    @staticmethod
    def parse_ranges(spec: str) -> list[tuple[int, int]]:
        ranges = []
        for raw_part in spec.split(","):
            part = raw_part.strip()
            if "-" in part:
                a, b = part.split("-")
                ranges.append((int(a), int(b)))
            elif part:
                ranges.append((int(part), int(part)))
        return ranges

    def render_code(self, code_lines: list[str], caption: re.Match | None, block_id: int, root: pathlib.Path) -> str:
        rows = []
        ranges: list[tuple[int, int]] = []
        note = ""
        file_path = None
        if caption:
            name, spec, note = caption.group(1), caption.group(2), caption.group(3) or ""
            ranges = self.parse_ranges(spec)
            file_path = self.rz.resolve(name)
            if file_path:
                self.used_files.add(file_path)
        src_lines = None
        if file_path and (root / file_path).is_file():
            src_lines = (root / file_path).read_text().split("\n")
        ri, ln = 0, (ranges[0][0] if ranges else None)
        for raw in code_lines:
            if raw.strip() in ("...", "…") and ranges:
                rows.append('<tr class="elide"><td class="ln">⋮</td><td class="cd">⋮ elided</td></tr>')
                ri += 1
                if ri < len(ranges):
                    ln = ranges[ri][0]
                else:
                    ln = None
                    BUILD_WARNINGS.append(f"{file_path}: elision after range {ranges[-1]} has no next range in the caption")
                continue
            if ln is not None and ri < len(ranges) and ln > ranges[ri][1]:
                BUILD_WARNINGS.append(f"{file_path}:{ln}: excerpt runs past range end {ranges[ri]} — caption ranges are wrong")
            if ln is not None and src_lines is not None:
                actual = src_lines[ln - 1].rstrip() if 0 < ln <= len(src_lines) else "<past end of file>"
                if actual != raw.rstrip():
                    BUILD_WARNINGS.append(
                        f"{file_path}:{ln}: excerpt differs from source\n"
                        f"      excerpt: {raw.rstrip()!r}\n      source:  {actual!r}"
                    )
            n = str(ln) if ln is not None else "·"
            attr = f' data-l="{ln}"' if ln is not None else ""
            rows.append(f'<tr{attr}><td class="ln">{n}</td><td class="cd">{esc(raw)}</td></tr>')
            if ln is not None:
                ln += 1
        head = ""
        if caption and file_path:
            first = ranges[0][0] if ranges else 1
            head = (
                f'<div class="codehead"><a class="ref" data-file="{file_path}" data-line="{first}" href="#">'
                f"{esc(caption.group(1))}:{esc(caption.group(2))}</a>"
                f'<span class="note">{esc(note)}</span>'
                f'<button class="copybtn" data-path="{root}/{file_path}:{first}">copy path</button></div>'
            )
        return (
            f'<figure class="code" id="code{block_id}">{head}'
            f'<table class="pysrc" data-hl="1">{"".join(rows)}</table></figure>'
        )

    # ---- document ----------------------------------------------------------
    def build_body(self, md_text: str, root: pathlib.Path) -> tuple[str, list[tuple[int, str, str]], str]:
        lines = md_text.split("\n")
        out: list[str] = []
        toc: list[tuple[int, str, str]] = []
        title = "Code walkthrough"
        i, code_id, sec_id = 0, 0, 0
        in_findings = False
        while i < len(lines):
            line = lines[i]
            if line.startswith("```"):
                j = i + 1
                code = []
                while j < len(lines) and not lines[j].startswith("```"):
                    code.append(lines[j])
                    j += 1
                j += 1
                cap = None
                k = j
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k < len(lines):
                    m = CAPTION.match(lines[k].strip())
                    if m:
                        cap = m
                        j = k + 1
                code_id += 1
                out.append(self.render_code(code, cap, code_id, root))
                i = j
            elif re.match(r"^#{1,3} ", line):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("# ").strip()
                if level == 1:
                    title = re.sub(r"[*`]", "", text)
                sec_id += 1
                anchor = f"s{sec_id}"
                fm = re.match(r"\*?\*?(R\d+)\b", text)
                if fm:
                    anchor = fm.group(1)
                out.append(f'<h{level} id="{anchor}">{self.inline(text)}</h{level}>')
                if level <= 2:
                    toc.append((level, anchor, re.sub(r"[*`]", "", text)))
                in_findings = "finding" in text.lower()
                i += 1
            elif line.strip() == "---":
                out.append("<hr>")
                i += 1
            elif line.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].startswith("|"):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                    i += 1
                body_rows = [r for r in rows if not all(re.fullmatch(r"-{2,}", c) for c in r)]
                thead = "".join(f"<th>{self.inline(c)}</th>" for c in body_rows[0])
                trs = "".join(
                    "<tr>" + "".join(f"<td>{self.inline(c)}</td>" for c in r) + "</tr>" for r in body_rows[1:]
                )
                out.append(f"<table class='doc'><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>")
            elif line.startswith("- "):
                items = []
                while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")):
                    if lines[i].startswith("- "):
                        items.append(lines[i][2:])
                    else:
                        items[-1] += " " + lines[i].strip()
                    i += 1
                if in_findings:
                    out.append(self.render_findings(items, toc))
                else:
                    out.append("<ul>" + "".join(f"<li>{self.inline(it)}</li>" for it in items) + "</ul>")
            elif line.strip():
                # consume the entry line unconditionally: no structural branch
                # claimed it, and a wrapped line starting with '#'/'- '/'|'
                # (e.g. a PR number "#3995...") must not stall the loop
                para = [line.strip()]
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r"^(#|```|\||- |---$)", lines[i]):
                    para.append(lines[i].strip())
                    i += 1
                out.append(f"<p>{self.inline(' '.join(para))}</p>")
            else:
                i += 1
        return "".join(out), toc, title

    def render_findings(self, items: list[str], toc: list[tuple[int, str, str]]) -> str:
        cards = []
        for it in items:
            fm = re.match(r"\*\*(R\d+)\s*—\s*(.*?)\*\*\s*(.*)", it, re.S)
            if fm:
                rid, card_title, body = fm.groups()
                toc.append((3, rid, f"{rid} {card_title.rstrip('.')}"))
                cards.append(
                    f'<div class="finding" id="{rid}"><div class="fhead"><span class="chip">{rid}</span>'
                    f"<strong>{self.inline(card_title)}</strong></div><p>{self.inline(body)}</p></div>"
                )
            else:
                cards.append(f'<div class="finding"><p>{self.inline(it)}</p></div>')
        return "".join(cards)


def validate(out_path: pathlib.Path) -> list[str]:
    """Structural self-check: balanced tags, script extractable."""
    from html.parser import HTMLParser

    text = out_path.read_text()
    problems: list[str] = []

    class Chk(HTMLParser):
        VOID = {"meta", "br", "hr", "img", "input", "link"}

        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.stack: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag not in self.VOID:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if not self.stack or self.stack[-1] != tag:
                problems.append(f"tag mismatch at {tag!r}")
            else:
                self.stack.pop()

    c = Chk()
    c.feed(text)
    if c.stack:
        problems.append(f"unclosed tags: {c.stack[:5]}")
    if "<script>" not in text:
        problems.append("no script block")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="annotated walkthrough Markdown")
    ap.add_argument("--out", default=None, help="output HTML (default: alongside --md)")
    ap.add_argument("--root", default=None, help="repo root (default: git toplevel of --md)")
    ap.add_argument("--embed", nargs="*", default=[], help="extra source files to embed")
    ap.add_argument("--prefer", nargs="*", default=[], help="path prefixes that win basename ambiguities")
    ap.add_argument("--comments", default=None, help="comments sidecar JSON (default: <md>.comments.json)")
    ap.add_argument("--title", default=None, help="override page title")
    args = ap.parse_args()

    md_path = pathlib.Path(args.md).resolve()
    if args.root:
        root = pathlib.Path(args.root).resolve()
    else:
        try:
            root = pathlib.Path(
                subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=md_path.parent,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            root = md_path.parent.parent
    out_path = pathlib.Path(args.out) if args.out else md_path.with_suffix(".html")

    builder = Builder(Resolver(root, args.prefer))
    body, toc, md_title = builder.build_body(md_path.read_text(), root)
    for extra in args.embed:
        p = builder.rz.resolve(extra)
        if p:
            builder.used_files.add(p)
        else:
            print(f"warning: --embed {extra!r} not found under {root}", file=sys.stderr)

    sources = {p: (root / p).read_text() for p in sorted(builder.used_files)}
    toc_html = "".join(f'<a class="t{lvl}" href="#{anchor}">{esc(text)}</a>' for lvl, anchor, text in toc)
    title = args.title or md_title

    page = TEMPLATE
    page = page.replace("/*TITLE*/", esc(title))
    page = page.replace("/*TOC*/", toc_html)
    page = page.replace("/*BODY*/", body)
    page = page.replace("/*SOURCES*/", json.dumps(sources))
    page = page.replace("/*ROOT*/", str(root))
    comments_path = pathlib.Path(args.comments) if args.comments else md_path.with_suffix(".comments.json")
    comments = json.loads(comments_path.read_text()) if comments_path.exists() else {"inbox": [], "threads": []}
    page = page.replace("/*COMMENTS*/", json.dumps(comments))
    out_path.write_text(page)

    problems = validate(out_path)
    n_code = page.count('<figure class="code"')
    n_ref = page.count('class="ref"')
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB): "
          f"{n_code} excerpts, {n_ref} code links, {len(sources)} sources embedded")
    if problems:
        print("VALIDATION PROBLEMS:", problems, file=sys.stderr)
        return 1
    if BUILD_WARNINGS:
        print(f"{len(BUILD_WARNINGS)} LINE-NUMBER WARNING(S):", file=sys.stderr)
        for w in BUILD_WARNINGS:
            print("  " + w, file=sys.stderr)
        return 1
    return 0


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>/*TITLE*/</title>
<style>
:root{--bg:#0f1117;--panel:#161923;--panel2:#1c2030;--fg:#d6dae3;--dim:#8b93a7;--acc:#6ea8fe;--acc2:#e3b341;
--kw:#ff7b72;--str:#a5d6a7;--com:#8b949e;--num:#d2a8ff;--fn:#79c0ff;--dec:#e3b341;--flash:#2d4f2d;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,'Segoe UI',Roboto,sans-serif}
#side{position:fixed;top:0;left:0;bottom:0;width:270px;overflow-y:auto;background:var(--panel);
padding:18px 14px;border-right:1px solid #262b3b}
#side h1{font-size:15px;margin:0 0 12px;color:var(--acc)}
#side a{display:block;color:var(--dim);text-decoration:none;padding:3px 8px;border-radius:6px;font-size:13px}
#side a.t1{font-weight:600;color:var(--fg);margin-top:8px}
#side a.t2{padding-left:16px}
#side a.t3{padding-left:26px;font-size:12px}
#side a:hover{background:var(--panel2);color:var(--fg)}
#side a.on{background:#22304a;color:var(--acc)}
#main{margin-left:270px;max-width:980px;padding:28px 42px 120px}
h1{font-size:26px;border-bottom:1px solid #262b3b;padding-bottom:10px}
h2{font-size:20px;margin-top:44px;color:var(--acc)}
h3{font-size:16px;margin-top:30px;color:var(--acc2)}
p,li{color:var(--fg)}
code{background:var(--panel2);padding:1px 5px;border-radius:5px;font:13px ui-monospace,'Cascadia Code',Menlo,monospace}
hr{border:none;border-top:1px solid #262b3b;margin:36px 0}
a.ref{color:var(--acc);text-decoration:none;border-bottom:1px dotted var(--acc)}
a.ref:hover{background:#22304a}
figure.code{margin:18px 0;background:var(--panel);border:1px solid #262b3b;border-radius:10px;overflow:hidden}
.codehead{display:flex;align-items:center;gap:10px;padding:7px 12px;background:var(--panel2);
border-bottom:1px solid #262b3b;font:12px ui-monospace,monospace}
.codehead .note{color:var(--dim);font-style:italic}
.copybtn{margin-left:auto;background:#22304a;color:var(--acc);border:none;border-radius:6px;
padding:3px 10px;font:11px ui-monospace,monospace;cursor:pointer}
.copybtn:hover{background:#2c3e63}
table.pysrc{border-collapse:collapse;width:100%;font:12.5px/1.55 ui-monospace,'Cascadia Code',Menlo,monospace}
table.pysrc td{padding:0 12px;white-space:pre}
td.ln{color:#4d5670;text-align:right;user-select:none;width:1%;border-right:1px solid #262b3b;background:#131722}
tr.elide td{color:var(--dim);font-style:italic}
tr.flash td{background:var(--flash)!important}
table.doc{border-collapse:collapse;margin:16px 0;width:100%}
table.doc th,table.doc td{border:1px solid #2a3044;padding:7px 11px;text-align:left;font-size:13.5px}
table.doc th{background:var(--panel2)}
.finding{background:var(--panel);border:1px solid #2a3044;border-left:4px solid var(--acc2);
border-radius:8px;padding:10px 16px;margin:10px 0}
.finding .chip{background:var(--acc2);color:#111;font-weight:700;border-radius:6px;padding:1px 8px;
font-size:12px;margin-right:9px}
.finding p{margin:6px 0 2px;color:var(--dim);font-size:14px}
.finding p strong{color:var(--fg)}
.kw{color:var(--kw)}.str{color:var(--str)}.com{color:var(--com)}.num{color:var(--num)}
.fn{color:var(--fn)}.dec{color:var(--dec)}
td.ln{cursor:pointer}
td.ln:hover{color:var(--acc);background:#1a2233}
tr.crow td{padding:8px 14px;background:#12151f;border-top:1px dashed #2a3044;border-bottom:1px dashed #2a3044;
white-space:normal;font:13.5px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif}
.bubble{max-width:680px;margin:6px 0;padding:8px 12px;border-radius:10px}
.bubble .who{font-size:11px;color:var(--dim);margin-bottom:3px}
.bubble.anka{background:#1e2a45;border:1px solid #2c3e63}
.bubble.claude{background:#1c2517;border:1px solid #2f4322}
.bubble.pending{background:#332b12;border:1px solid #6b5a1e}
textarea.cbox{width:100%;max-width:680px;min-height:64px;background:#0f1320;color:var(--fg);
border:1px solid #3a4360;border-radius:8px;padding:8px;font:13px/1.5 -apple-system,sans-serif}
button.cbtn{background:#22304a;color:var(--acc);border:none;border-radius:6px;padding:4px 12px;
margin:6px 6px 0 0;cursor:pointer;font-size:12px}
button.cbtn:hover{background:#2c3e63}
#ctool{position:fixed;bottom:18px;right:18px;background:var(--panel2);border:1px solid #3a4360;
border-radius:10px;padding:9px 14px;font-size:13px;z-index:60;box-shadow:0 4px 20px #0007}
#ctool .hint{color:var(--dim);font-size:11px;margin-top:3px}
#viewer{position:fixed;top:0;right:-56%;width:56%;height:100%;background:var(--panel);z-index:50;
transition:right .25s ease;border-left:1px solid #313850;display:flex;flex-direction:column;box-shadow:-12px 0 40px #0009}
#viewer.open{right:0}
#vhead{display:flex;align-items:center;gap:12px;padding:10px 16px;background:var(--panel2);border-bottom:1px solid #313850}
#vtitle{font:13px ui-monospace,monospace;color:var(--acc)}
#vclose{margin-left:auto;background:none;border:1px solid #3a4360;color:var(--dim);border-radius:6px;
padding:3px 12px;cursor:pointer}
#vclose:hover{color:var(--fg)}
#vbody{overflow:auto;flex:1}
#toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#22304a;color:var(--acc);
padding:8px 18px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none;z-index:99}
#toast.show{opacity:1}
</style></head><body>
<nav id="side"><h1>/*TITLE*/</h1>/*TOC*/</nav>
<main id="main">/*BODY*/</main>
<div id="viewer"><div id="vhead"><span id="vtitle"></span>
<button class="copybtn" id="vcopy">copy path</button><button id="vclose">close ✕</button></div>
<div id="vbody"></div></div>
<div id="ctool"><strong id="ccount">0</strong> draft comment(s) ·
<button class="cbtn" id="cexport">copy as JSON</button>
<button class="cbtn" id="cconnect" style="display:none">connect comments file</button>
<div class="hint"><span id="clive">offline · drafts stay in this browser</span><br>click any line number to comment · reply inside a thread</div></div>
<div id="toast"></div>
<script>
const SOURCES = /*SOURCES*/;
const ROOT = "/*ROOT*/";
const COMMENTS = /*COMMENTS*/;
const KW = new Set(("def class return if elif else for while in not and or is None True False import from as with "+
"try except finally raise pass break continue lambda yield global nonlocal assert del async await match case").split(" "));
function tokenize(src){
  const re = /("{3}[\\s\\S]*?"{3}|'{3}[\\s\\S]*?'{3}|"(?:\\\\.|[^"\\\\\\n])*"|'(?:\\\\.|[^'\\\\\\n])*'|#[^\\n]*|@\\w[\\w.]*|\\b\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b|\\b[A-Za-z_]\\w*\\b)/g;
  const toks=[]; let last=0, m;
  while((m=re.exec(src))){
    if(m.index>last) toks.push([src.slice(last,m.index),null]);
    const t=m[0]; let cls=null;
    if(t[0]==='#') cls='com';
    else if(t[0]==='"'||t[0]==="'") cls='str';
    else if(t[0]==='@') cls='dec';
    else if(/^\\d/.test(t)) cls='num';
    else if(KW.has(t)) cls='kw';
    else if(src.slice(re.lastIndex).match(/^\\s*\\(/) && !KW.has(t)) cls='fn';
    toks.push([t,cls]); last=re.lastIndex;
  }
  if(last<src.length) toks.push([src.slice(last),null]);
  return toks;
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function highlightToLines(src){
  const lines=[['']];
  for(const [text,cls] of tokenize(src)){
    const parts=text.split('\\n');
    parts.forEach((p,i)=>{
      if(i>0) lines.push(['']);
      if(p) lines[lines.length-1].push(cls?`<span class="${cls}">${esc(p)}</span>`:esc(p));
    });
  }
  return lines.map(chunks=>chunks.join(''));
}
document.querySelectorAll('table.pysrc[data-hl]').forEach(tbl=>{
  const cells=[...tbl.querySelectorAll('td.cd')].filter(td=>!td.parentElement.classList.contains('elide'));
  const src=cells.map(td=>td.textContent).join('\\n');
  const hl=highlightToLines(src);
  cells.forEach((td,i)=>{td.innerHTML=hl[i]||'';});
});
const viewer=document.getElementById('viewer'), vbody=document.getElementById('vbody'),
      vtitle=document.getElementById('vtitle'), vcopy=document.getElementById('vcopy');
const rendered={};
function openFile(path,line){
  if(!(path in SOURCES)){toast('source not embedded: '+path);return;}
  if(!rendered[path]){
    const hl=highlightToLines(SOURCES[path]);
    const rows=hl.map((h,i)=>`<tr id="L-${i+1}"><td class="ln">${i+1}</td><td class="cd">${h}</td></tr>`).join('');
    rendered[path]=`<table class="pysrc">${rows}</table>`;
  }
  vbody.innerHTML=rendered[path];
  curFile=path;
  renderThreads(vbody,path);
  vtitle.textContent=path+(line?':'+line:'');
  vcopy.dataset.path=ROOT+'/'+path+(line?':'+line:'');
  viewer.classList.add('open');
  if(line){
    const tr=vbody.querySelector('#L-'+line);
    if(tr){if(tr.scrollIntoView)tr.scrollIntoView({block:'center'});flash(tr);}
  } else vbody.scrollTop=0;
}
function flash(tr){tr.classList.add('flash');setTimeout(()=>tr.classList.remove('flash'),1600);}
document.getElementById('vclose').onclick=()=>viewer.classList.remove('open');
document.addEventListener('keydown',e=>{if(e.key==='Escape')viewer.classList.remove('open');});
document.addEventListener('click',e=>{
  const ref=e.target.closest('a.ref');
  if(ref){e.preventDefault();openFile(ref.dataset.file,parseInt(ref.dataset.line)||0);return;}
  const cb=e.target.closest('.copybtn');
  if(cb&&cb.dataset.path){copyText(cb.dataset.path);return;}
});
function copyText(text,msg){
  const done=()=>toast(msg||('copied: '+text));
  (navigator.clipboard?navigator.clipboard.writeText(text):Promise.reject())
    .then(done)
    .catch(()=>{try{const ta=document.createElement('textarea');ta.value=text;document.body.appendChild(ta);
      ta.select();document.execCommand('copy');ta.remove();done();}catch(e){toast('copy failed');}});}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2200);}
// ---- offline comment threads ----
let curFile=null;
const DOCKEY='cwt:'+document.title;
const MEMSTORE={};  // fallback when localStorage is unavailable (opaque origins)
function lsGet(k){try{return localStorage.getItem(k)}catch(e){return MEMSTORE[k]||null}}
function lsSet(k,v){try{localStorage.setItem(k,v)}catch(e){MEMSTORE[k]=v}}
function drafts(){try{return JSON.parse(lsGet(DOCKEY)||'[]')}catch(e){return []}}
function setDrafts(p){lsSet(DOCKEY,JSON.stringify(p));refreshTool();}
(function prune(){const baked=new Set();
 (COMMENTS.threads||[]).forEach(t=>(t.messages||[]).forEach(m=>baked.add((m.text||'').trim())));
 (COMMENTS.inbox||[]).forEach(q=>baked.add((q.text||'').trim()));
 setDrafts(drafts().filter(p=>!baked.has(p.text.trim())));})();
function bubble(role,text,ts){
  const who=role==='anka'?'Anka':role==='claude'?'Claude':'draft — not sent yet';
  return `<div class="bubble ${role}"><div class="who">${who}${ts?' · '+ts:''}</div>${esc(text).replace(/\\n/g,'<br>')}</div>`;}
function splitAnchor(a){const ix=a.lastIndexOf(':');return [a.slice(0,ix),a.slice(ix+1)];}
function findRow(scope,line){return scope.querySelector(`tr[data-l="${line}"]`)||scope.querySelector('#L-'+line);}
function renderThreads(scope,file){
  scope.querySelectorAll('tr.crow').forEach(r=>r.remove());
  (COMMENTS.inbox||[]).forEach(q=>{
    const [f,l]=splitAnchor(q.anchor);
    if(f!==file)return;
    const tr=findRow(scope,l);
    if(!tr)return;
    const row=document.createElement('tr');row.className='crow';
    row.innerHTML=`<td colspan="2">${bubble('anka',q.text,(q.ts||'')+' · awaiting Claude')}</td>`;
    tr.after(row);});
  (COMMENTS.threads||[]).forEach(t=>{
    const [f,l]=splitAnchor(t.anchor);
    if(f!==file)return;
    const tr=findRow(scope,l);
    if(!tr)return;
    const msgs=(t.messages||[]).map(m=>bubble(m.role,m.text,m.ts||'')).join('');
    const pend=drafts().map((p,i)=>p.tid===t.id?
      bubble('pending',p.text,'')+`<button class="cbtn cdel" data-i="${i}">delete draft</button>`:'').join('');
    const row=document.createElement('tr');row.className='crow';
    row.innerHTML=`<td colspan="2">${msgs}${pend}`+
      `<button class="cbtn reply" data-tid="${t.id}" data-anchor="${t.anchor}">reply</button></td>`;
    tr.after(row);});
  drafts().forEach((p,i)=>{
    if(p.tid)return;
    const [f,l]=splitAnchor(p.anchor);
    if(f!==file)return;
    const tr=findRow(scope,l);
    if(!tr)return;
    const row=document.createElement('tr');row.className='crow';
    row.innerHTML=`<td colspan="2">${bubble('pending',p.text,'')}`+
      `<button class="cbtn cdel" data-i="${i}">delete draft</button></td>`;
    tr.after(row);});
}
function renderAllExcerpts(){document.querySelectorAll('figure.code').forEach(fig=>{
  const a=fig.querySelector('.codehead a.ref');
  if(a)renderThreads(fig,a.dataset.file);});}
function rerender(){renderAllExcerpts();
  if(viewer.classList.contains('open')&&curFile)renderThreads(vbody,curFile);}
function openBox(tr,anchor,tid){
  closeBox();
  const row=document.createElement('tr');row.className='crow ceditrow';
  row.dataset.anchor=anchor;
  if(tid)row.dataset.tid=tid;
  row.innerHTML=`<td colspan="2"><div class="who" style="font:11px ui-monospace,monospace;color:var(--dim)">`+
    `${anchor}${tid?' · reply to thread '+tid:''}</div>`+
    `<textarea class="cbox" placeholder="leave a comment / question for Claude…"></textarea><br>`+
    `<button class="cbtn csave">save draft</button><button class="cbtn ccancel">cancel</button></td>`;
  tr.after(row);row.querySelector('textarea').focus();}
function closeBox(){document.querySelectorAll('tr.ceditrow').forEach(r=>r.remove());}
function refreshTool(){const el=document.getElementById('ccount');if(el)el.textContent=drafts().length;}
function exportComments(){
  const p=drafts();
  if(!p.length){toast('no draft comments');return;}
  const entries=p.map(c=>{
    const o={anchor:c.anchor};
    if(c.tid)o.thread=c.tid;
    o.text=c.text;o.ts=c.ts;
    return JSON.stringify(o);});
  copyText(entries.join(',\\n')+',',
    'copied '+p.length+' JSON entr'+(p.length>1?'ies':'y')+' — paste into the "inbox" array of the comments file');}
document.addEventListener('click',e=>{
  const ln=e.target.closest('td.ln');
  if(ln){const tr=ln.parentElement;
    if(!tr.classList.contains('crow')&&!tr.classList.contains('elide')){
      let file=null,line=null;
      const fig=tr.closest('figure.code');
      if(fig){const a=fig.querySelector('.codehead a.ref');
        if(a&&tr.dataset.l){file=a.dataset.file;line=tr.dataset.l;}}
      else if(tr.closest('#vbody')){file=curFile;line=(tr.id||'').replace('L-','');}
      if(file&&line)openBox(tr,file+':'+line);}
    return;}
  const rep=e.target.closest('.reply');
  if(rep){openBox(rep.closest('tr'),rep.dataset.anchor,rep.dataset.tid);return;}
  const del=e.target.closest('.cdel');
  if(del){const p=drafts();p.splice(parseInt(del.dataset.i),1);setDrafts(p);rerender();return;}
  const sv=e.target.closest('.csave');
  if(sv){const row=sv.closest('tr');
    const txt=row.querySelector('textarea').value.trim();
    if(!txt){toast('empty comment');return;}
    const entry={anchor:row.dataset.anchor,text:txt};
    if(row.dataset.tid)entry.thread=row.dataset.tid;
    if(mode==='server'){postComment(entry).then(()=>{closeBox();return refreshLive();})
      .then(()=>toast('saved to the comments file'))
      .catch(()=>{setMode('offline');saveDraft(entry);closeBox();});}
    else if(mode==='file'){writeFileEntry(entry).then(()=>{closeBox();toast('saved to the comments file');})
      .catch(()=>{setMode('offline');saveDraft(entry);closeBox();});}
    else{saveDraft(entry);closeBox();}
    return;}
  if(e.target.closest('.ccancel')){closeBox();return;}
  if(e.target.closest('#cexport')){exportComments();return;}
});
function saveDraft(entry){
  const p=drafts();
  p.push({anchor:entry.anchor,tid:entry.thread||null,text:entry.text,
    ts:new Date().toISOString().slice(0,16).replace('T',' ')});
  setDrafts(p);rerender();
  toast('draft saved locally — connect the comments file (bottom right) for auto-save');}
// auto-save modes: 'file' (File System Access API, no server) > 'server' > 'offline'
const API=(location.protocol==='http:'||location.protocol==='https:'?'':'http://127.0.0.1:8321')+'/api';
let mode='offline',lastComments='',fileHandle=null,fileLastMod=0;
function fileBtn(show,label){const b=document.getElementById('cconnect');
  if(!b)return;b.style.display=show?'inline-block':'none';if(label)b.textContent=label;}
function setMode(m){mode=m;const el=document.getElementById('clive');
  const txt={offline:'offline · drafts stay in this browser',
             server:'live · auto-saving via review server',
             file:'live · auto-saving straight to the comments file'}[m];
  if(el){el.textContent=txt;el.style.color=m==='offline'?'':'#7ee787';}
  fileBtn(m==='offline'&&typeof window.showOpenFilePicker==='function');}
function applyComments(d){const str=JSON.stringify(d);
  if(str===lastComments)return;lastComments=str;
  COMMENTS.threads=d.threads||[];COMMENTS.inbox=d.inbox||[];rerender();}
async function flushAll(send){
  const p=drafts();if(!p.length)return;
  for(const c of p){const e={anchor:c.anchor,text:c.text};if(c.tid)e.thread=c.tid;await send(e);}
  setDrafts([]);toast('uploaded '+p.length+' local draft(s) to the comments file');}
// server mode
async function postComment(entry){
  const r=await fetch(API+'/comment',{method:'POST',headers:{'Content-Type':'text/plain'},body:JSON.stringify(entry)});
  if(!r.ok)throw new Error('save failed');}
async function refreshLive(){
  const r=await fetch(API+'/comments',{cache:'no-store'});
  if(!r.ok)throw new Error('bad status');
  applyComments(await r.json());}
async function pollServer(){
  while(mode==='server'){await new Promise(rs=>setTimeout(rs,4000));
    try{await refreshLive();}catch(e){setMode('offline');}}}
// file-handle mode (no server; Chromium File System Access API)
async function idb(op,val){
  try{
    const db=await new Promise((res,rej)=>{const r=indexedDB.open('cwt-handles',1);
      r.onupgradeneeded=()=>r.result.createObjectStore('h');
      r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error);});
    return await new Promise((res,rej)=>{
      const tx=db.transaction('h',op==='get'?'readonly':'readwrite');
      const q=op==='get'?tx.objectStore('h').get(DOCKEY):tx.objectStore('h').put(val,DOCKEY);
      q.onsuccess=()=>res(q.result);q.onerror=()=>rej(q.error);});
  }catch(e){return null;}}
async function refreshFile(){
  const f=await fileHandle.getFile();
  if(f.lastModified===fileLastMod)return;
  fileLastMod=f.lastModified;
  applyComments(JSON.parse(await f.text()));}
async function writeFileEntry(entry){
  const f=await fileHandle.getFile();
  let data;try{data=JSON.parse(await f.text());}catch(e){data={};}
  if(!data.inbox)data.inbox=[];
  if(!data.threads)data.threads=[];
  entry.ts=new Date().toISOString().slice(0,16).replace('T',' ');
  data.inbox.push(entry);
  const w=await fileHandle.createWritable();
  await w.write(JSON.stringify(data,null,1)+'\\n');await w.close();
  fileLastMod=0;await refreshFile();}
async function pollFile(){
  while(mode==='file'){await new Promise(rs=>setTimeout(rs,4000));
    try{await refreshFile();}catch(e){/* transient read failure: keep polling */}}}
async function enterFileMode(){
  setMode('file');await flushAll(writeFileEntry);await refreshFile();pollFile();}
async function connectFile(){
  try{
    const [h]=await window.showOpenFilePicker({
      types:[{description:'comments JSON',accept:{'application/json':['.json']}}]});
    if(h.requestPermission&&(await h.requestPermission({mode:'readwrite'}))!=='granted'){
      toast('write permission denied');return;}
    fileHandle=h;await idb('put',h);await enterFileMode();
  }catch(e){if(e.name!=='AbortError')toast('connect failed: '+e.message);}}
async function resumeFile(){
  if(typeof window.showOpenFilePicker!=='function')return false;
  const h=await idb('get');
  if(!h)return false;
  const q=h.queryPermission?await h.queryPermission({mode:'readwrite'}):'granted';
  if(q==='granted'){fileHandle=h;await enterFileMode();return true;}
  fileBtn(true,'reconnect comments file');
  return false;}
renderAllExcerpts();
refreshTool();
document.getElementById('cconnect').addEventListener('click',connectFile);
(async()=>{
  try{await refreshLive();setMode('server');await flushAll(postComment);await refreshLive();pollServer();}
  catch(e){
    try{if(await resumeFile())return;}catch(e2){}
    setMode('offline');}})();
// ---- scroll-spy ----
const tocLinks=[...document.querySelectorAll('#side a')];
const anchors=tocLinks.map(a=>document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
const obs=new IntersectionObserver(es=>{
  es.forEach(en=>{if(en.isIntersecting){
    tocLinks.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+en.target.id));}});
},{rootMargin:'0px 0px -75% 0px'});
anchors.forEach(a=>obs.observe(a));
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
