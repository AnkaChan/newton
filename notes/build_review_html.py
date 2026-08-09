# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the interactive HTML implementation review from notes/03 + live sources.

Reads ``notes/03_implementation_review.md`` and the core source files, emits a
fully self-contained ``notes/03_implementation_review.html``:

- sidebar TOC with scroll-spy,
- code excerpts with *real* line numbers and a caption linking to the source,
- every ``file.py:NN`` mention anywhere becomes a click target that opens an
  embedded full-file viewer scrolled to (and flashing) that line,
- findings R1-R12 rendered as anchored cards,
- zero network dependencies (hand-rolled Python syntax highlighting in JS).

Usage:  uv run python notes/build_review_html.py
"""

from __future__ import annotations

import html
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = ROOT / "notes"
MD = NOTES / "03_implementation_review.md"
OUT = NOTES / "03_implementation_review.html"

EMBED_FILES = [
    "research/principal_stretch/torch_solver.py",
    "research/principal_stretch/polar.py",
    "research/principal_stretch/model.py",
    "research/principal_stretch/train.py",
    "research/principal_stretch/potentials.py",
    "research/principal_stretch/gen_train_data.py",
    "research/principal_stretch/diag_knn_floor.py",
]

FILE_REF = re.compile(r"\b([\w./]+\.py):(\d+)(?:-(\d+))?\b")
BASENAME_TO_PATH = {pathlib.Path(p).name: p for p in EMBED_FILES}


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def linkify_ref(m: re.Match) -> str:
    name, a, b = m.group(1), m.group(2), m.group(3)
    base = pathlib.Path(name).name
    path = BASENAME_TO_PATH.get(base)
    label = m.group(0)
    if path is None:
        return label
    return f'<a class="ref" data-file="{path}" data-line="{a}" href="#">{label}</a>'


def inline(s: str) -> str:
    """Escape + bold + inline code (with file:line linkification inside code)."""
    s = esc(s)
    out, pos = [], 0
    for m in re.finditer(r"``(.+?)``|`([^`]+)`", s):
        out.append(_bold(s[pos : m.start()]))
        body = m.group(1) or m.group(2)
        out.append(f"<code>{FILE_REF.sub(linkify_ref, body)}</code>")
        pos = m.end()
    out.append(_bold(s[pos:]))
    return "".join(out)


def _bold(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<em>\1</em>", s)
    return FILE_REF.sub(linkify_ref, s)


CAPTION = re.compile(r"^\(`([\w./]+\.py):([\d,\s-]+)`(?:,?\s*(.*?))?\)$")


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ranges.append((int(a), int(b)))
        elif part:
            ranges.append((int(part), int(part)))
    return ranges


def render_code(code_lines: list[str], caption: re.Match | None, block_id: int) -> str:
    """Code block with real line numbers; elision rows advance to the next range."""
    rows = []
    ranges: list[tuple[int, int]] = []
    note = ""
    file_path = None
    if caption:
        name, spec, note = caption.group(1), caption.group(2), caption.group(3) or ""
        ranges = parse_ranges(spec)
        file_path = BASENAME_TO_PATH.get(pathlib.Path(name).name, name)
    ri, ln = 0, (ranges[0][0] if ranges else None)
    for raw in code_lines:
        if raw.strip() in ("...", "…") and ranges:
            rows.append('<tr class="elide"><td class="ln">⋮</td><td class="cd">⋮ elided</td></tr>')
            ri += 1
            ln = ranges[ri][0] if ri < len(ranges) else None
            continue
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
            f'<button class="cpy" data-path="{ROOT}/{file_path}:{first}">copy path</button></div>'
        )
    return f'<figure class="code" id="code{block_id}">{head}<table class="pysrc" data-hl="1">{"".join(rows)}</table></figure>'


def build_body(md_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = md_text.split("\n")
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
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
            out.append(render_code(code, cap, code_id))
            i = j
        elif re.match(r"^#{1,3} ", line):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("# ").strip()
            sec_id += 1
            anchor = f"s{sec_id}"
            fm = re.match(r"\*?\*?(R\d+)\b", text)
            if fm:
                anchor = fm.group(1)
            out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            if level <= 2:
                toc.append((level, anchor, re.sub(r"[*`]", "", text)))
            in_findings = "Review findings" in text
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
            thead = "".join(f"<th>{inline(c)}</th>" for c in body_rows[0])
            trs = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body_rows[1:])
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
                cards = []
                for it in items:
                    fm = re.match(r"\*\*(R\d+)\s*—\s*(.*?)\*\*\s*(.*)", it, re.S)
                    if fm:
                        rid, title, body = fm.groups()
                        toc.append((3, rid, f"{rid} {title.rstrip('.')}"))
                        cards.append(
                            f'<div class="finding" id="{rid}"><div class="fhead"><span class="chip">{rid}</span>'
                            f"<strong>{inline(title)}</strong></div><p>{inline(body)}</p></div>"
                        )
                    else:
                        cards.append(f'<div class="finding"><p>{inline(it)}</p></div>')
                out.append("".join(cards))
            else:
                out.append("<ul>" + "".join(f"<li>{inline(it)}</li>" for it in items) + "</ul>")
        elif line.strip():
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|```|\||- |---$)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
        else:
            i += 1
    return "".join(out), toc


def main() -> None:
    md_text = MD.read_text()
    body, toc = build_body(md_text)
    sources = {p: (ROOT / p).read_text() for p in EMBED_FILES}
    toc_html = "".join(f'<a class="t{lvl}" href="#{anchor}">{esc(text)}</a>' for lvl, anchor, text in toc)
    page = TEMPLATE
    page = page.replace("/*TOC*/", toc_html)
    page = page.replace("/*BODY*/", body)
    page = page.replace("/*SOURCES*/", json.dumps(sources))
    page = page.replace("/*ROOT*/", str(ROOT))
    OUT.write_text(page)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PSS Implementation Review — code walkthrough</title>
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
.cpy{margin-left:auto;background:#22304a;color:var(--acc);border:none;border-radius:6px;
padding:3px 10px;font:11px ui-monospace,monospace;cursor:pointer}
.cpy:hover{background:#2c3e63}
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
<nav id="side"><h1>PSS Implementation Review</h1>/*TOC*/</nav>
<main id="main">/*BODY*/</main>
<div id="viewer"><div id="vhead"><span id="vtitle"></span>
<button class="cpy" id="vcopy">copy path</button><button id="vclose">close ✕</button></div>
<div id="vbody"></div></div>
<div id="toast"></div>
<script>
const SOURCES = /*SOURCES*/;
const ROOT = "/*ROOT*/";
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
  const lines=[['']]; // array of arrays of html chunks
  for(const [text,cls] of tokenize(src)){
    const parts=text.split('\\n');
    parts.forEach((p,i)=>{
      if(i>0) lines.push(['']);
      if(p) lines[lines.length-1].push(cls?`<span class="${cls}">${esc(p)}</span>`:esc(p));
    });
  }
  return lines.map(chunks=>chunks.join(''));
}
// highlight in-page excerpts (cell text is plain — re-render with highlighting)
document.querySelectorAll('table.pysrc[data-hl]').forEach(tbl=>{
  const cells=[...tbl.querySelectorAll('td.cd')].filter(td=>!td.parentElement.classList.contains('elide'));
  const src=cells.map(td=>td.textContent).join('\\n');
  const hl=highlightToLines(src);
  cells.forEach((td,i)=>{td.innerHTML=hl[i]||'';});
});
// viewer
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
  vtitle.textContent=path+(line?':'+line:'');
  vcopy.dataset.path=ROOT+'/'+path+(line?':'+line:'');
  viewer.classList.add('open');
  if(line){
    const tr=vbody.querySelector('#L-'+line);
    if(tr){tr.scrollIntoView({block:'center'});flash(tr);}
  } else vbody.scrollTop=0;
}
function flash(tr){tr.classList.add('flash');setTimeout(()=>tr.classList.remove('flash'),1600);}
document.getElementById('vclose').onclick=()=>viewer.classList.remove('open');
document.addEventListener('keydown',e=>{if(e.key==='Escape')viewer.classList.remove('open');});
document.addEventListener('click',e=>{
  const ref=e.target.closest('a.ref');
  if(ref){e.preventDefault();openFile(ref.dataset.file,parseInt(ref.dataset.line)||0);return;}
  const cpy=e.target.closest('.cpy');
  if(cpy){copy(cpy.dataset.path);return;}
});
function copy(text){
  (navigator.clipboard?navigator.clipboard.writeText(text):Promise.reject())
    .then(()=>toast('copied: '+text))
    .catch(()=>{const ta=document.createElement('textarea');ta.value=text;document.body.appendChild(ta);
      ta.select();document.execCommand('copy');ta.remove();toast('copied: '+text);});
}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2200);}
// scroll-spy
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
    main()
