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

_UPDATE_COLUMNS = (
    "update",
    "epoch",
    "loss",
    "before_joule",
    "after_joule",
    "mean_force_residual_n",
    "step_size_mean",
    "step_size_min",
    "step_size_max",
    "tie_cell_count",
)
_EPOCH_COLUMNS = (
    "epoch",
    "loss",
    "query_count",
    "seconds",
    "mean_force_residual_n",
    "step_size_mean",
    "step_size_min",
    "step_size_max",
    "tie_cell_count",
    "selection_metric",
    "selection_eligible",
    "physical_survivors",
    "sample_count",
)


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


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value):
    return f"{value:.6g}" if _finite(value) else "Unavailable"


def _lookup(row, *path):
    """Return a nested value or None when any key is missing or not a mapping."""
    value = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _plot(title, series, *, xlabel):
    """Return one SVG line chart; None or nonfinite values leave gaps in the line."""
    finite = [(x, y) for _, _, points in series for x, y in points if _finite(y)]
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
            if not _finite(y):
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


def _curve_plot(title, curves):
    """Plot mean/median/max per iteration with the shared failed/valid gap handling."""
    return _plot(
        title,
        [
            (name, color, [(r.get("iteration"), r.get(name)) for r in curves if _finite(r.get("iteration"))])
            for name, color in (("mean", "#2563eb"), ("median", "#16804a"), ("max", "#c63645"))
        ],
        xlabel="Optimizer iteration",
    )


def _epoch_plot(rows):
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    def values(*path, scale=1):
        return [(_lookup(row, *path) * scale) if _finite(_lookup(row, *path)) else math.nan for row in rows]

    def ratio(*path, denominator):
        result = []
        for row in rows:
            value, total = _lookup(row, *path), _lookup(row, *denominator)
            result.append(100 * value / total if _finite(value) and _finite(total) and total > 0 else math.nan)
        return result

    with mpl.rc_context({"svg.fonttype": "none", "font.size": 10}):
        figure = Figure(figsize=(12, 10), layout="constrained")
        axes = figure.subplots(3, 2)
        epochs = [row["epoch"] for row in rows]
        axes[0, 0].plot(epochs, values("loss"), ".-", color="#157f94", label="Training objective (mean over queries)")
        axes[0, 0].plot(
            epochs,
            values("validation", "mean_normalized_loss"),
            ".-",
            color="#d26a25",
            label="Validation: first update",
        )
        axes[0, 0].set_ylabel("Per-update LeCO objective")
        axes[0, 0].axhline(0, color="grey", linewidth=0.7)
        axes[0, 0].legend()
        axes[0, 1].plot(epochs, values("validation", "descent_rate", scale=100), ".-", color="#d26a25")
        axes[0, 1].set_ylabel("Validation queries with lower energy (%)")
        axes[0, 1].set_ylim(-2, 102)
        for key, label, color in (
            ("mean_before_joule", "Before first update", "#157f94"),
            ("mean_after_joule", "After first update", "#d26a25"),
        ):
            axes[1, 0].plot(epochs, values("validation", key), ".-", label=label, color=color)
        axes[1, 0].set_ylabel("Validation mean physical energy (J)")
        axes[1, 0].legend()
        metric = values("validation", "selection", "metric")
        axes[1, 1].plot(epochs, metric, ".-", color="#7052a3", label="Cheap validation, final iteration")
        full = values("full_horizon_validation", "final_free_force_residual_norm_n", "mean")
        axes[1, 1].plot(epochs, full, "s", color="#c63645", label="Full horizon, final step")
        axes[1, 1].set_ylabel("Selection metric: mean final force residual (N)")
        if any(_finite(value) and value > 0 for value in metric + full):
            axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[2, 0].plot(
            epochs,
            ratio("validation", "physical_survivors", denominator=("validation", "sample_count")),
            ".-",
            color="#157f94",
            label="Cheap validation",
        )
        axes[2, 0].plot(
            epochs,
            ratio(
                "full_horizon_validation", "physical_survivors", denominator=("full_horizon_validation", "sample_count")
            ),
            "s",
            color="#c63645",
            label="Full horizon",
        )
        axes[2, 0].set_ylabel("Physical survivors (%)")
        axes[2, 0].set_ylim(-2, 102)
        axes[2, 0].legend()
        axes[2, 1].plot(epochs, values("learning_rate"), ".-", color="#7052a3")
        axes[2, 1].set_ylabel("Adam learning rate after epoch")
        axes[2, 1].set_yscale("log")
        for axis in axes.flat:
            axis.set_xlabel("Completed epoch")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.2)
        figure.suptitle("LIDO-v2 — training and validation history")
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg")
    return buffer.getvalue()


def _latest_full_horizon(rows):
    """Return (epoch, summary) of the most recent full-horizon validation, or (None, None)."""
    for row in reversed(rows):
        full = row.get("full_horizon_validation")
        if isinstance(full, dict):
            return row.get("epoch"), full
    return None, None


def write_mixed_report(output, report, *, updated_at=None):
    """Write portable epoch metrics, three SVG plots and a page refreshing every 30s.

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
    validation = latest.get("validation") or {}
    relative = validation.get("relative_energy", [])
    endpoint = relative[-1] if relative else {}
    residual_curve = validation.get("force_residual", [])
    residual_endpoint = residual_curve[-1] if residual_curve else {}
    selection = validation.get("selection") or {}
    best = report.get("best_selection") or {}
    validation_iterations = config.get("validation_iterations", 100)
    completed = report.get("completed_epochs", 0)
    maximum = config.get("max_epochs", 500)
    status = report.get("status", "preparing")
    phase = progress.get("phase", status)
    if status in ("failed", "interrupted", "epoch_limit", "early_stopped", "plateau_converged", "stalled"):
        phase = status

    def escape(value):
        return html.escape(str(value))

    counts_k = progress.get("available_K", latest.get("available_K", [1]))
    counts_h = progress.get("available_H", latest.get("available_H", [8]))
    loss_plot = _epoch_plot(rows)
    validation_plot = _curve_plot("Validation relative physical energy", relative)
    residual_plot = _curve_plot("Validation free-corner force residual (N)", residual_curve)
    _atomic_text(output / "loss_curve.svg", loss_plot)
    _atomic_text(output / "validation_curve.svg", validation_plot)
    _atomic_text(output / "residual_curve.svg", residual_plot)
    epoch_rows = [
        {
            **row,
            "selection_metric": _lookup(row, "validation", "selection", "metric"),
            "selection_eligible": _lookup(row, "validation", "selection", "eligible"),
            "physical_survivors": _lookup(row, "validation", "physical_survivors"),
            "sample_count": _lookup(row, "validation", "sample_count"),
        }
        for row in rows
    ]
    for name, data, columns in (
        ("updates", report.get("updates", []), _UPDATE_COLUMNS),
        ("epochs", epoch_rows, _EPOCH_COLUMNS),
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
    descent = validation.get("descent_rate")
    descent_text = f"{descent:.1%}" if _finite(descent) else "Unavailable"
    gate = config.get("stage_descent_rate")
    gate_text = f"{gate:.0%}" if _finite(gate) else "Unavailable"
    stage_limit = config.get("stage_max_epochs")
    stage_limit_text = f"{stage_limit} epochs per stage" if stage_limit is not None else "No hard cap"
    world_size = report.get("world_size", 1)
    batch_size = config.get("batch_size")
    batch_text = (
        f"{world_size} GPUs, batch {batch_size} per GPU (global {world_size * batch_size})."
        if isinstance(batch_size, int)
        else ""
    )
    survivors = validation.get("physical_survivors")
    sample_count = validation.get("sample_count")
    survival_text = (
        f"{survivors} / {sample_count}"
        if isinstance(survivors, int) and isinstance(sample_count, int)
        else "Not evaluated"
    )
    metric = selection.get("metric")
    metric_text = f"{_number(metric)} N" if _finite(metric) else "Unavailable"
    eligible = selection.get("eligible")
    eligibility_text = (
        "eligible for checkpoint selection"
        if eligible is True
        else "not eligible (a failed or incomplete trajectory)"
        if eligible is False
        else "eligibility not recorded"
    )
    best_text = (
        f"Best so far: {_number(best.get('metric'))} N at epoch {escape(best.get('epoch'))}."
        if _finite(best.get("metric"))
        else "No eligible epoch has been selected yet."
    )
    full_epoch, full = _latest_full_horizon(rows)
    if full:
        final_residual = full.get("final_free_force_residual_norm_n") or {}
        final_energy = full.get("final_energy_joule") or {}
        full_html = f"""<table><tr><th>Epoch</th><th>K</th><th>H</th><th>Survivors</th><th>Final residual mean / median / max (N)</th><th>Final energy mean (J)</th><th>Seconds</th></tr>
<tr><td>{escape(full_epoch)}</td><td>{escape(full.get("iterations", "—"))}</td><td>{escape(full.get("physical_steps", "—"))}</td><td>{escape(full.get("physical_survivors", "—"))} / {escape(full.get("sample_count", "—"))}</td><td>{_number(final_residual.get("mean"))} / {_number(final_residual.get("median"))} / {_number(final_residual.get("max"))}</td><td>{_number(final_energy.get("mean"))}</td><td>{_number(full.get("seconds"))}</td></tr></table>"""
    else:
        full_html = '<p class="muted">No full-horizon validation has completed yet.</p>'
    residual_table = (
        f"<tr><td>Force residual (N)</td><td>{_number(residual_endpoint.get('mean'))}</td><td>{_number(residual_endpoint.get('median'))}</td><td>{_number(residual_endpoint.get('max'))}</td></tr>"
        if residual_endpoint
        else ""
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30"><title>LIDO-v2 · live training</title>
<style>
*{{box-sizing:border-box}}body{{font:17px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#20323b;background:#f8fafb}}
a{{color:#087c91}}img,svg{{display:block;width:100%;height:auto;background:white;border-radius:12px}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}
.status{{background:#e6f2f4;padding:18px;border-radius:10px}}h1{{line-height:1.2}}h2{{font-size:22px}}.muted{{color:#526174;font-size:14px}}
.legend span{{margin-right:20px}}.failure{{padding:18px;border:1px solid #c63645;border-radius:10px;background:#fff5f5}}code{{font-size:13px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #e5eaf2;text-align:left}}details{{margin:18px 0}}summary{{cursor:pointer}}section{{margin:28px 0}}
@media(max-width:600px){{table{{table-layout:fixed;font-size:14px}}td,th{{padding:6px 4px;overflow-wrap:anywhere}}}}
</style></head><body><main>
<a href="/artifacts/learned-intrinsic-solver/index.html">← All solver experiments</a>
<h1>LIDO-v2 — training the deformation optimizer</h1>
<p class="status"><strong>{escape(phase.replace("_", " ").capitalize())}</strong> · {completed} / {maximum} completed epochs · {progress.get("completed_updates", report.get("completed_updates", 0))} Adam updates<br>
{escape(config.get("queries_per_epoch", "—"))} training queries per epoch and {escape(config.get("validation_count", validation.get("sample_count", "—")))} fixed validation states. {escape(batch_text)}<br>
Available solver iterations: K = {escape(counts_k)} · Physical timesteps: H = {escape(counts_h)}<br>
Curriculum descent gate: {gate_text} · Hard cap: {escape(stage_limit_text)}</p>
<p>Selection metric (mean free-corner force residual after {validation_iterations} iterations): {metric_text}, {eligibility_text}. Physical survivors: {survival_text}. {best_text}<br>
Validation energy: {_number(validation.get("mean_before_joule"))} → {_number(validation.get("mean_after_joule"))} J after one update. Descent: {descent_text}; first-update failures: {escape(validation.get("first_update_failed_count", "Not evaluated"))}; all validation failures: {escape(validation.get("failed_count", "Not evaluated"))}.</p>
{failure_html}<img src="loss_curve.svg" alt="Training objective and validation first-update objective, validation descent rate, physical energy, selection metric, physical survivors, and learning rate by epoch">
<p class="muted">Lower objective is better. The training objective includes an uphill penalty; validation shows the first update on fixed seeds with the same form. The selection metric is the mean final free-corner force residual of the cheap validation; squares mark full-horizon checks. The learning rate is recorded after each epoch's scheduler decision.</p>
<details><summary>How the loss curves are computed</summary>
<p>Mean local training loss averages all queried trajectories and ranks in each completed epoch. Epochs mix solver ages, physical timesteps and curriculum stages.</p>
<p>Per-update loss = asinh(E_after / s) + &lambda; · max((E_after &minus; E_before) / s, 0) with s = max(|E_before|, floor), where E_before is the energy immediately before that update and floor = c · 2<sup>&minus;23</sup> · V · (&lambda;<sub>Lam&eacute;</sub> + 2&mu; + &eta;/dt + &rho;h<sup>2</sup>/dt<sup>2</sup>) is the material-aware float32 energy floor (c = {escape(config.get("energy_floor_scale", 1.0))}). The penalty weight is &lambda; = {escape(config.get("energy_increase_weight", 1.0))}.</p>
<p>Validation reports the same per-update loss for the first update of each fixed seed. Missing measurements leave gaps. Training descent rate is not recorded; the descent panel shows validation only.</p></details>
<section><h2>Latest validation: {validation_iterations} optimizer iterations</h2>
<p class="muted">Epoch {escape(latest.get("epoch", "—"))}, {escape(validation.get("sample_count", 0))} fixed validation seeds. Each energy value is a per-trajectory physical energy ratio Eᵢ / E₀; the residual is the Euclidean norm of the free-corner position gradient [N]. Mean, median and maximum aggregate the per-trajectory values. Network weights stay fixed during validation.</p>
<div class="legend"><span style="color:#2563eb">● Mean</span><span style="color:#16804a">● Median</span><span style="color:#c63645">● Maximum</span></div>{validation_plot}
{residual_plot}
<table><tr><th>At iteration {escape(endpoint.get("iteration", validation_iterations))}</th><th>Mean</th><th>Median</th><th>Maximum</th></tr><tr><td>Relative energy</td><td>{_number(endpoint.get("mean"))}</td><td>{_number(endpoint.get("median"))}</td><td>{_number(endpoint.get("max"))}</td></tr>{residual_table}</table>
<p class="muted">Failed validation trajectories: {escape(validation.get("failed_count", "Not evaluated"))}; physical survivors: {survival_text}; near-zero initial energies: {escape(endpoint.get("near_zero_count", 0))}. Optimizer failures leave gaps from the failed iteration onward; near-zero initial energies are excluded from relative ratios but keep their residuals. Physical-rollout failures are counted separately in the report.</p>
<h2>Latest full-horizon validation</h2>
<p class="muted">Held-out seeds run K learned iterations on each of H physical steps at the largest currently available budgets; every {escape(config.get("validation_full_interval", "—"))} epochs and before curriculum advancement.</p>
{full_html}</section>
<p><a href="report.json">Metrics JSON</a> · <a href="progress.json">Live progress</a> · <a href="epochs.csv">Epoch CSV</a> · <a href="updates.csv">Update CSV</a> · <a href="loss_curve.svg">Training SVG</a> · <a href="validation_curve.svg">Validation SVG</a> · <a href="residual_curve.svg">Residual SVG</a></p>
<details><summary>Configuration</summary><p>{damping_text}</p><pre>{escape(json.dumps(config, indent=2))}</pre></details>
<p class="muted">This page refreshes every 30 seconds. Curves update after each completed epoch.<br>
Epoch in progress: {escape(progress.get("epoch", "Not started"))}. Training heartbeat: {escape(progress.get("updated_at", "Waiting"))}.<br>
Epoch metrics: {escape(report["updated_at"])}. Page published: {escape(report.get("publication", {}).get("updated_at", "Local report"))}.</p>
</main></body></html>"""
    _atomic_text(output / "report.json", json.dumps(report, indent=2) + "\n")
    if "progress" in report:
        _atomic_text(output / "progress.json", json.dumps(progress, indent=2) + "\n")
    _atomic_text(output / "index.html", page)
