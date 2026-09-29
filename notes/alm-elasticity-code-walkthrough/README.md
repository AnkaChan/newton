# ALM elasticity code walkthrough

Open [review.html](review.html) directly in a browser. It is a self-contained
snapshot of the code at `b5a7ab81`, including the triangle extension in
`60407c40`. The earlier walkthroughs remain unchanged.

The page follows the `code-walkthrough-html` skill in AI-Docs. It includes a
pipeline map, source-checked excerpts, full embedded source files, a sidebar,
and offline line comments. The adjacent Markdown is the archival source.

Click a file reference to inspect its source. Click a line number to leave a
draft. Drafts stay in local browser storage until exported with **copy as JSON**.
Paste exported entries into the `inbox` array of `review.comments.json`, then
tell the assistant that review is complete. No server or publication is needed.

The copied generator disables the skill's known-flickering file-handle mode
and adds narrow-screen layout and code scrolling. Complete embedded sources
are gzip-compressed to fit the repository's file-size limit and expanded using
the browser's built-in `DecompressionStream` (verified in Chromium). Opening
the file directly makes no network requests, including no local-server probe.
It preserves the skill's embedded-source and offline-comment format.

Regenerate from the worktree root after editing Markdown, source, or comments:

```bash
uv run --no-sync python notes/alm-elasticity-code-walkthrough/build_walkthrough_html.py \
    --md notes/alm-elasticity-code-walkthrough/review.md
```

The generator checks source excerpts against their declared line ranges and
rejects stale excerpts. Update the recorded snapshot when rebuilding against
new solver code. No simulation was rerun for this documentation task; the
recorded solver and bag results are explicitly identified in the walkthrough.
