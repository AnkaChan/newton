# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental portable reports and live progress for mixed-pool training."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["write_mixed_report", "write_progress"]


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_progress(output, report, *, phase, epoch, available_K=None, available_H=None):
    """Atomically publish a small training heartbeat without copying state.

    Experimental. Call on rank zero during initialization, training, validation
    and termination. This writer does not change the in-memory report.

    Args:
        output: Local training output directory.
        report: Current metrics report, containing config and completed counters.
        phase: Human-readable current phase, such as training or validation.
        epoch: Current epoch number, including an in-progress epoch.
        available_K: Current allowed inner-iteration counts for new resets.
        available_H: Current allowed physical-step counts for new resets.
    """
    latest = (report.get("epochs") or [{}])[-1]
    update = (report.get("updates") or [{}])[-1]
    progress = {
        "updated_at": _utc_now(),
        "status": report.get("status", "running"),
        "phase": phase,
        "epoch": epoch,
        "max_epochs": report.get("config", {}).get("max_epochs", 500),
        "completed_epochs": report.get("completed_epochs", 0),
        "completed_updates": report.get("completed_updates", 0),
        "latest_batch_loss": update.get("loss"),
        "available_K": list(available_K if available_K is not None else latest.get("available_K", [1])),
        "available_H": list(available_H if available_H is not None else latest.get("available_H", [8])),
    }
    _atomic_text(Path(output) / "progress.json", json.dumps(progress, indent=2) + "\n")


def _number(value):
    return f"{value:.6g}" if isinstance(value, (int, float)) and math.isfinite(value) else "Unavailable"


def _plot(title, series, *, xlabel):
    finite = [(x, y) for _, _, points in series for x, y in points if isinstance(y, (int, float)) and math.isfinite(y)]
    if not finite:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 340">'
            f'<title>{html.escape(title)}</title><text x="30" y="60" fill="#526174">'
            "Waiting for completed measurements.</text></svg>"
        )
    xs, ys = zip(*finite, strict=True)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmin == xmax:
        xmin, xmax = min(0, xmin), max(1, xmax)
    padding = max((ymax - ymin) * 0.08, abs(ymax) * 0.025, 1e-8)
    ymin, ymax = ymin - padding, ymax + padding

    def point(x, y):
        return 95 + 875 * (x - xmin) / (xmax - xmin), 270 - 220 * (y - ymin) / (ymax - ymin)

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 340" role="img">',
        f'<title>{html.escape(title)}</title><rect width="1000" height="340" fill="white"/>',
        '<g font-family="system-ui,sans-serif" font-size="13" fill="#526174">',
    ]
    for index in range(5):
        value = ymin + (ymax - ymin) * index / 4
        _, y = point(xmin, value)
        elements.append(
            f'<path d="M95 {y:.2f}H970" stroke="#e2e8f0"/>'
            f'<text x="85" y="{y + 4:.2f}" text-anchor="end">{value:.4g}</text>'
        )
    for value in sorted({xmin, (xmin + xmax) / 2, xmax}):
        x, _ = point(value, ymin)
        elements.append(f'<text x="{x:.2f}" y="294" text-anchor="middle">{value:g}</text>')
    elements.append(f'<text x="535" y="328" text-anchor="middle">{html.escape(xlabel)}</text></g>')
    for name, color, values in series:
        commands = []
        connected = False
        for x, y in values:
            if not isinstance(y, (int, float)) or not math.isfinite(y):
                connected = False
                continue
            px, py = point(x, y)
            commands.append(f"{'L' if connected else 'M'}{px:.3f},{py:.3f}")
            connected = True
            elements.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2" fill="{color}"/>')
        elements.append(
            f'<path aria-label="{html.escape(name)}" d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
    elements.append("</svg>")
    return "".join(elements)


def write_mixed_report(output, report, *, updated_at=None):
    """Write portable epoch metrics, two SVG plots and a page refreshing every 30s.

    Experimental. Only report metrics are written; checkpoints and trajectory
    state are never read. ``updated_at`` denotes the source metrics timestamp,
    while an optional ``publication`` block carries a separate mirror heartbeat.

    Args:
        output: Destination directory for public report artifacts.
        report: Training metrics and optional progress/publication/failure data.
        updated_at: UTC timestamp override for the source metrics, or now.
    """
    output = Path(output)
    report = dict(report, updated_at=updated_at or _utc_now())
    rows = report.get("epochs", [])
    latest = rows[-1] if rows else {}
    progress = report.get("progress", {})
    config = report.get("config", {})
    validation = latest.get("validation", {})
    relative = validation.get("relative_energy", [])
    endpoint = relative[-1] if relative else {}
    validation_iterations = config.get("validation_iterations", 100)
    completed = report.get("completed_epochs", 0)
    maximum = config.get("max_epochs", 500)
    status = report.get("status", "preparing")
    phase = progress.get("phase", status)
    if status in ("failed", "interrupted", "epoch_limit", "early_stopped"):
        phase = status

    def escape(value):
        return html.escape(str(value))

    counts_k = progress.get("available_K", latest.get("available_K", [1]))
    counts_h = progress.get("available_H", latest.get("available_H", [8]))
    loss_plot = _plot(
        "Mean local training loss",
        [("Training loss", "#2563eb", [(r["epoch"], r["loss"]) for r in rows])],
        xlabel="Completed epoch",
    )
    validation_plot = _plot(
        "Validation relative physical energy",
        [
            (name, color, [(r["iteration"], r.get(name)) for r in relative])
            for name, color in (("mean", "#2563eb"), ("median", "#16804a"), ("max", "#c63645"))
        ],
        xlabel="Optimizer iteration",
    )
    _atomic_text(output / "loss_curve.svg", loss_plot)
    _atomic_text(output / "validation_curve.svg", validation_plot)
    for name, data, columns in (
        ("updates", report.get("updates", []), ("update", "epoch", "loss", "before_joule", "after_joule")),
        ("epochs", rows, ("epoch", "loss", "query_count", "seconds")),
    ):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
        _atomic_text(output / f"{name}.csv", buffer.getvalue())
    failure = report.get("failure")
    failure_html = (
        f'<section class="failure"><h2>Training failure</h2><pre>{escape(json.dumps(failure, indent=2))}</pre></section>'
        if failure
        else ""
    )
    failures = validation.get("failures", [])
    if failures:
        failure_html += f"<details><summary>Validation failure details</summary><pre>{escape(json.dumps(failures, indent=2))}</pre></details>"
    eta_range = config.get("damping_range")
    damping_text = (
        f"Absolute viscosity: {escape(eta_range)} Pa·s, sampled once per trajectory."
        if eta_range is not None
        else "Absolute metric viscosity is fixed within each trajectory."
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30"><title>Learned intrinsic solver · live training</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5fa;color:#17243b;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1100px;margin:40px auto;padding:0 20px}}h1{{font-size:30px;line-height:1.2}}h2{{font-size:21px;margin:0 0 10px}}
.eyebrow{{color:#526174;font-size:13px;text-transform:uppercase;letter-spacing:.09em}}.muted{{color:#526174;font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card,section{{background:white;border:1px solid #dfe5ef;border-radius:12px;padding:20px;margin:18px 0}}
.card{{margin:0}}.value{{font-size:26px;font-weight:650}}svg{{display:block;width:100%;height:auto}}.legend span{{margin-right:20px}}a{{color:#185bc3}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}.failure{{border-color:#c63645;background:#fff5f5}}code{{font-size:13px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #e5eaf2;text-align:left}}
</style></head><body><main>
<div class="eyebrow">Live experiment · four-GPU mixed trajectories</div><h1>Learned intrinsic solver training</h1>
<p>Page refreshes every 30 seconds. {damping_text}</p>
<div class="cards"><div class="card"><div class="muted">Status / phase</div><div class="value">{escape(phase.replace("_", " ").capitalize())}</div></div>
<div class="card"><div class="muted">Completed epochs</div><div class="value">{completed} / {maximum}</div></div>
<div class="card"><div class="muted">Adam updates</div><div class="value">{progress.get("completed_updates", report.get("completed_updates", 0))}</div></div>
<div class="card"><div class="muted">Current curriculum choices</div><div>K = {escape(counts_k)}<br>H = {escape(counts_h)}</div></div></div>
<p class="muted">Epoch in progress: {escape(progress.get("epoch", "Not started"))}. Training heartbeat: {escape(progress.get("updated_at", "Waiting"))}.<br>
Epoch metrics: {escape(report["updated_at"])}. Page published: {escape(report.get("publication", {}).get("updated_at", "Local report"))}.</p>
{failure_html}<section><h2>Mean local training loss</h2><p class="muted">Mean over all queried trajectories and ranks in each completed epoch; lower is better. Epochs mix different solver ages and curriculum stages. This is the training objective, not a fixed-seed convergence curve.</p>
{loss_plot}<p class="muted">Local loss = (E_after &minus; E_initial) / max(E_initial, 1 J) + λ · max(E_after &minus; E_previous, 0) / max(E_initial, 1 J). λ = {escape(config.get("energy_increase_weight", 1.0))}. Latest completed epoch: {_number(latest.get("loss"))}.</p></section>
<section><h2>Latest validation: {validation_iterations} optimizer iterations</h2>
<p class="muted">Epoch {escape(latest.get("epoch", "—"))}, {escape(validation.get("sample_count", 0))} fixed validation seeds. Each value is a per-trajectory physical energy ratio Eᵢ / E₀; mean, median and maximum aggregate those ratios. Network weights stay fixed during validation.</p>
<div class="legend"><span style="color:#2563eb">● Mean</span><span style="color:#16804a">● Median</span><span style="color:#c63645">● Maximum</span></div>{validation_plot}
<table><tr><th>At iteration {escape(endpoint.get("iteration", validation_iterations))}</th><th>Mean</th><th>Median</th><th>Maximum</th></tr><tr><td>Relative energy</td><td>{_number(endpoint.get("mean"))}</td><td>{_number(endpoint.get("median"))}</td><td>{_number(endpoint.get("max"))}</td></tr></table>
<p class="muted">Failed validation trajectories: {escape(validation.get("failed_count", "Not evaluated"))}; physical survivors: {escape(validation.get("physical_survivors", "Not evaluated"))}; near-zero initial energies: {escape(endpoint.get("near_zero_count", 0))}. Optimizer failures leave gaps from the failed iteration onward; near-zero initial energies are excluded from relative ratios. Physical-rollout failures are counted separately in the report.</p></section>
<p><a href="report.json">Metrics JSON</a> · <a href="progress.json">Live progress</a> · <a href="epochs.csv">Epoch CSV</a> · <a href="updates.csv">Update CSV</a> · <a href="loss_curve.svg">Training SVG</a> · <a href="validation_curve.svg">Validation SVG</a></p>
</main></body></html>"""
    _atomic_text(output / "report.json", json.dumps(report, indent=2) + "\n")
    if "progress" in report:
        _atomic_text(output / "progress.json", json.dumps(progress, indent=2) + "\n")
    _atomic_text(output / "index.html", page)
