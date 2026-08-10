# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tiny review server: serve the walkthrough and auto-save comments.

One process, stdlib only, localhost only. The walkthrough page detects it and
switches from browser-local drafts to saving straight into the comments
sidecar; Claude's answers (written to the same file) appear on the page via
polling. Kill it any time — the page falls back to offline drafts.

Run from the worktree:
    python notes/serve_review.py            # http://127.0.0.1:8321/03_implementation_review.html

Endpoints:
    GET  /api/comments   current sidecar JSON (no-cache)
    POST /api/comment    append one inbox entry {"anchor", "text"[, "thread"]}
"""

from __future__ import annotations

import argparse
import datetime
import http.server
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--dir", default=str(pathlib.Path(__file__).resolve().parent), help="static root (default: notes/)")
    ap.add_argument("--comments", default=None, help="sidecar path (default: first *.comments.json in --dir)")
    args = ap.parse_args()

    root = pathlib.Path(args.dir).resolve()
    if args.comments:
        cpath = pathlib.Path(args.comments).resolve()
    else:
        found = sorted(root.glob("*.comments.json"))
        if not found:
            print(f"no *.comments.json under {root}", file=sys.stderr)
            return 1
        cpath = found[0]

    def load() -> dict:
        if cpath.exists():
            data = json.loads(cpath.read_text())
        else:
            data = {}
        data.setdefault("inbox", [])
        data.setdefault("threads", [])
        return data

    def store(data: dict) -> None:
        tmp = cpath.with_name(cpath.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        tmp.replace(cpath)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path == "/api/comments":
                self._json(200, load())
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/api/comment":
                self._json(404, {"ok": False, "error": "unknown endpoint"})
                return
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.loads(raw)
                entry = {"anchor": str(body["anchor"]), "text": str(body["text"])}
                if body.get("thread"):
                    entry["thread"] = str(body["thread"])
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._json(400, {"ok": False, "error": str(e)})
                return
            entry["ts"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            data = load()
            data["inbox"].append(entry)
            store(data)
            print(
                f"NEW comment {entry['anchor']}" + (f" (thread {entry['thread']})" if "thread" in entry else ""),
                flush=True,
            )
            self._json(200, {"ok": True})

        def log_message(self, fmt, *a):  # quiet static-file noise
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"review server: http://127.0.0.1:{args.port}/  comments -> {cpath}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
