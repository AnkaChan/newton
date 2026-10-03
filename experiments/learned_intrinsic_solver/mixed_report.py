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
    "gradient_norm",
    "contact_max_penetration_r",
    "contact_pair_mean",
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
    "gradient_norm_mean",
    "gradient_norm_max",
    "selection_metric",
    "selection_eligible",
    "physical_survivors",
    "sample_count",
    "contact_scene_fraction",
    "contact_realized_fraction",
    "contact_max_penetration_r",
    "validation_final_max_penetration_r",
    # Fixed-state regime columns, appended last so positional readers of pool-regime tables are unaffected.
    "regime_stage",
    "regime_k_max",
    "regime_h_max",
    "regime_updates",
    # Full-horizon selection columns (schema 6), appended after the regime columns for the same reason.
    "full_horizon_selection_metric",
    "full_horizon_selection_eligible",
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


def write_progress(output, report, *, phase, epoch, available_K=None, available_H=None, regime=None):
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
        regime: Fixed-state regime block of the current epoch (stage, K_max,
            H_max, updates) or None for the pool regime, which writes no
            ``regime`` entry.
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
    if regime is not None:
        progress["regime"] = dict(regime)
    _atomic_text(Path(output) / "progress.json", json.dumps(progress, indent=2) + "\n")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value):
    return f"{value:.6g}" if _finite(value) else "Unavailable"


def _text(value):
    """Return ``value`` escaped for HTML, with a dash for None."""
    return html.escape(str(value if value is not None else "—"))


def _fixed_state_budget_text(config, regime):
    """Describe a fixed-state epoch for the status line: state set, budget cap, growth stage and its counts.

    ``regime`` is the current heartbeat's or latest epoch row's ``regime``
    block; the stage counts are omitted while the jobs are not yet assigned.
    """
    text = (
        f"Fixed-state regime: {_text(config.get('state_count'))} training states per epoch, "
        f"K x H ≤ {_text(config.get('budget_cap'))}"
    )
    if regime.get("stage") is not None:
        text += (
            f"; growth stage {_text(regime.get('stage'))} "
            f"(K ≤ {_text(regime.get('k_max'))}, H ≤ {_text(regime.get('h_max'))})"
        )
    if regime.get("updates") is not None:
        fillers = regime.get("filler_queries")
        filler_total = sum(fillers) if isinstance(fillers, list) else fillers
        text += (
            f": {_text(regime.get('queries'))} sampled queries + {_text(filler_total)} filler "
            f"in {_text(regime.get('updates'))} updates per rank"
        )
    return text


def _fixed_state_schedule_text(config):
    """Describe the fixed-state growth timetable that replaces the curriculum gate line."""
    stages = config.get("growth_stages") or ()
    timetable = " → ".join(f"({_text(k_max)}, {_text(h_max)})" for k_max, h_max in stages) or "—"
    return (
        f"Growth timetable (K_max, H_max) from stage 0: {timetable}, advancing every "
        f"{_text(config.get('growth_stage_epochs'))} epochs; the final stage persists. "
        "The validation-gated curriculum is not used."
    )


def _origin_text(origin):
    """Describe a weights-only initialization (``report["initialized_from"]``) or return an empty string."""
    if not isinstance(origin, dict):
        return ""
    best = origin.get("best_selection") or {}
    best_text = (
        f", best {_number(best.get('metric'))} N at epoch {_text(best.get('epoch'))}"
        if _finite(best.get("metric"))
        else ""
    )
    checkpoint = "/".join(Path(str(origin.get("checkpoint", "—"))).parts[-3:])
    return (
        f"<br>Initialized weights-only (network and AdamW state) from {_text(checkpoint)} "
        f"after {_text(origin.get('completed_epochs'))} completed epochs{best_text}."
    )


def _lookup(row, *path):
    """Return a nested value or None when any key is missing or not a mapping."""
    value = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _plot(title, series, *, xlabel, log_y=False):
    """Return one SVG line chart; None or nonfinite values leave gaps in the line.

    With ``log_y`` the vertical axis is logarithmic (base 10); non-positive values leave gaps."""
    if log_y:
        series = [
            (name, color, [(x, math.log10(y) if _finite(y) and y > 0 else None) for x, y in points])
            for name, color, points in series
        ]
    finite = [(x, y) for _, _, points in series for x, y in points if _finite(y)]
    if not finite:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 340">'
            f"<title>{html.escape(title)}</title>"
            f'<text x="535" y="26" text-anchor="middle" font-family="system-ui,sans-serif" font-size="16" '
            f'font-weight="600" fill="#1f2937">{html.escape(title)}</text>'
            '<text x="30" y="60" fill="#526174" font-family="system-ui,sans-serif" font-size="13">'
            "Waiting for completed measurements.</text></svg>"
        )
    xs, ys = zip(*finite, strict=True)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmin == xmax:
        xmin, xmax = min(0, xmin), max(1, xmax)
    if log_y:
        # whole decades, gridlines and labels on powers of ten (10^n form)
        ymin, ymax = math.floor(ymin), math.ceil(ymax)
        if ymax <= ymin:
            ymax = ymin + 1
        step = max(1, math.ceil((ymax - ymin) / 8))
        ticks = list(range(int(ymin), int(ymax) + 1, step))
    else:
        padding = max((ymax - ymin) * 0.08, abs(ymax) * 0.025, 1e-8)
        ymin, ymax = ymin - padding, ymax + padding
        ticks = [ymin + (ymax - ymin) * index / 4 for index in range(5)]

    def point(x, y):
        return 95 + 875 * (x - xmin) / (xmax - xmin), 270 - 220 * (y - ymin) / (ymax - ymin)

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 340" role="img">',
        f'<title>{html.escape(title)}</title><rect width="1000" height="340" fill="white"/>',
        f'<text x="535" y="26" text-anchor="middle" font-family="system-ui,sans-serif" font-size="16" '
        f'font-weight="600" fill="#1f2937">{html.escape(title)}</text>',
        '<g font-family="system-ui,sans-serif" font-size="13" fill="#526174">',
    ]
    for value in ticks:
        _, y = point(xmin, value)
        if log_y:
            exponent = f"{int(round(value))}".replace("-", "\u2212")
            label = f'10<tspan dy="-6" font-size="10">{exponent}</tspan>'
        else:
            label = f"{value:.4g}"
        elements.append(
            f'<path d="M95 {y:.2f}H970" stroke="#e2e8f0"/><text x="85" y="{y + 4:.2f}" text-anchor="end">{label}</text>'
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


def _penetration_plot(curve):
    """Plot the mean and maximum deepest penetration (units of r) per iteration of the latest validation."""
    return _plot(
        "Validation deepest contact penetration (units of r)",
        [
            (name, color, [(r.get("iteration"), r.get(name)) for r in curve if _finite(r.get("iteration"))])
            for name, color in (("mean", "#2563eb"), ("max", "#c63645"))
        ],
        xlabel="Optimizer iteration",
    )


def _final_penetration(row):
    """Return the maximum deepest penetration at the last validation iteration, or None."""
    curve = _lookup(row, "validation", "penetration")
    if isinstance(curve, list) and curve and isinstance(curve[-1], dict):
        return curve[-1].get("max")
    return None


def _selection_source(report):
    """Return the summary that selects the best checkpoint: the configured source, else the record's, else cheap."""
    config = report.get("config") or {}
    best = report.get("best_selection") or {}
    source = config.get("selection_source") or best.get("source") or "cheap"
    return source if source in ("cheap", "full_horizon") else "cheap"


def _survival_text(summary):
    """Return ``survivors / samples`` of a validation summary, or a placeholder when it was not evaluated."""
    survivors = summary.get("physical_survivors")
    sample_count = summary.get("sample_count")
    if isinstance(survivors, int) and isinstance(sample_count, int):
        return f"{survivors} / {sample_count}"
    return "Not evaluated"


def _eligibility_text(eligible, *, selecting):
    """Describe a selection record's eligibility; a non-selecting summary is described without the selection role."""
    if eligible is True:
        return "eligible for checkpoint selection" if selecting else "all trajectories survived"
    if eligible is False:
        return "not eligible (a failed or incomplete trajectory)" if selecting else "a failed or incomplete trajectory"
    return "eligibility not recorded"


def _best_text(best, source):
    """Describe the best record; the full-horizon source adds its budget, final energy and deepest penetration."""
    if not _finite(best.get("metric")):
        return "No eligible epoch has been selected yet."
    text = f"Best so far: {_number(best.get('metric'))} N at epoch {html.escape(str(best.get('epoch')))}"
    if source == "full_horizon" and best.get("source") == "full_horizon":
        energy = best.get("final_energy_joule") or {}
        penetration = best.get("final_max_penetration_r") or {}
        text += (
            f" (full-horizon check, K = {html.escape(str(best.get('iterations', '—')))}, "
            f"H = {html.escape(str(best.get('physical_steps', '—')))}; final energy mean {_number(energy.get('mean'))} J; "
            f"deepest penetration mean {_number(penetration.get('mean'))} / max {_number(penetration.get('max'))} r)"
        )
    return text + "."


def _pre_roll_text(config, latest) -> str:
    """One status line on the pre-roll (v6, 2026-10-03, asked for by Anka): the sampling range and the queries per
    step from the config, the latest epoch's mean steps actually run, truncations and refills from its records."""
    max_steps = int(config.get("pre_roll_max_steps", 0) or 0)
    if max_steps <= 0:
        return ""
    dt = float(config.get("time_step", 1.0 / 300.0) or 1.0 / 300.0)
    queries = config.get("pre_roll_queries", "?")
    regime = latest.get("regime") or {}
    scene = latest.get("scene_regime") or {}
    mean = regime.get("pre_roll_steps_mean")
    parts = [
        f"Pre-roll: n ~ U{{0..{max_steps}}} inference-only steps ({max_steps * dt:.1f} s) at {html.escape(str(queries))} queries per step, from the growth table's last stage"
    ]
    if mean is not None:
        parts.append(
            f"latest epoch: mean {float(mean):.0f} steps run, {html.escape(str(scene.get('pre_roll_truncated', 0)))} pre-rolls truncated at the penetration guard, {html.escape(str(scene.get('refill_scenes', 0)))} refill scenes, {float(scene.get('pre_roll_seconds', 0.0)) / 60:.0f} min of pre-roll on rank 0"
        )
    return " · ".join(parts) + "<br>\n"


def _tick_plots(report):
    """Two SVGs of the intra-epoch ticks (train.py: running training loss and a small cheap validation every
    ``tick_updates`` updates) against the rank's cumulative updates, with the epoch-end values as a second series."""
    ticks = report.get("ticks") or []
    rows = report.get("epochs") or []
    cum, epoch_points_loss, epoch_points_metric = 0, [], []
    for row in rows:
        cum += int(row.get("updates") or 0)
        epoch_points_loss.append((cum, row.get("loss")))
        epoch_points_metric.append((cum, ((row.get("validation") or {}).get("selection") or {}).get("metric")))
    tick_loss = [(t.get("updates_done"), t.get("train_loss")) for t in ticks]
    tick_metric = [(t.get("updates_done"), t.get("cheap_metric")) for t in ticks]
    loss_svg = _plot(
        "Training loss within epochs: ticks (running mean since the previous tick) and epoch means",
        [("tick", "#2563eb", tick_loss), ("epoch mean", "#c63645", epoch_points_loss)],
        xlabel="updates per rank",
        log_y=True,
    )
    metric_svg = _plot(
        "Cheap validation metric (N) within epochs: ticks (4 held-out scenes) and epoch-end values (8 scenes)",
        [("tick", "#16804a", tick_metric), ("epoch end", "#c63645", epoch_points_metric)],
        xlabel="updates per rank",
        log_y=True,
    )
    return loss_svg, metric_svg


def _epoch_plot(rows, *, selection_source="cheap"):
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
        full_selects = selection_source == "full_horizon"
        axes[1, 1].plot(
            epochs,
            metric,
            ".-",
            color="#7052a3",
            label="Cheap validation, final iteration" + ("" if full_selects else " (selects)"),
        )
        full = values("full_horizon_validation", "final_free_force_residual_norm_n", "mean")
        axes[1, 1].plot(
            epochs,
            full,
            "s",
            color="#c63645",
            label="Full horizon, final step" + (" (selects)" if full_selects else ""),
        )
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
        axes[2, 1].plot(epochs, values("learning_rate"), ".-", color="#7052a3", label="Learning rate")
        axes[2, 1].set_ylabel("Learning rate after epoch")
        axes[2, 1].set_yscale("log")
        if any(_finite(row.get("gradient_norm_mean")) for row in rows):
            twin = axes[2, 1].twinx()
            twin.plot(epochs, values("gradient_norm_mean"), ".-", color="#d26a25", label="Gradient norm (epoch mean)")
            twin.plot(epochs, values("gradient_norm_max"), "_", color="#d26a25", label="Gradient norm (epoch max)")
            twin.set_ylabel("Global gradient norm before clipping")
            twin.set_yscale("log")
            handles, labels = axes[2, 1].get_legend_handles_labels()
            more, more_labels = twin.get_legend_handles_labels()
            axes[2, 1].legend(handles + more, labels + more_labels, loc="upper right")
        for axis in axes.flat:
            axis.set_xlabel("Completed epoch")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.2)
        figure.suptitle("LIDO — training and validation history")
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg")
    return buffer.getvalue()


def _latest_summary(rows, key):
    """Return (epoch, summary) of the most recent row whose ``key`` is a dict, or (None, None).

    Rows of epochs that skipped validation carry ``None`` and are passed over.
    """
    for row in reversed(rows):
        summary = row.get(key)
        if isinstance(summary, dict):
            return row.get("epoch"), summary
    return None, None


def _relative_residual_plot(residual_curve):
    """Force residual relative to the pre-optimisation value, per iteration, on a log axis.

    For each statistic (mean, median, maximum over the validation trajectories) the curve shows
    statistic(iteration k) / statistic(iteration 0)."""
    base = residual_curve[0] if residual_curve else {}
    series = []
    for key, color, name in (
        ("mean", "#2563eb", "Mean"),
        ("median", "#16804a", "Median"),
        ("max", "#c63645", "Maximum"),
    ):
        reference = base.get(key)
        points = []
        for row in residual_curve:
            value = row.get(key)
            ok = _finite(value) and _finite(reference) and reference > 0 and value > 0
            points.append((row.get("iteration", len(points)), value / reference if ok else None))
        series.append((name, color, points))
    return _plot(
        "Validation force residual relative to iteration 0 (log scale)",
        series,
        xlabel="optimizer iteration",
        log_y=True,
    )


def _per_trajectory_relative_residual_plot(samples, iterations):
    """Each trajectory's residual divided by its own initial residual, aggregated per iteration (log axis).

    Aggregates: median, geometric mean (the mean on the log axis) and maximum over the trajectories.
    Works with either residual key used by the two trainer generations."""
    curves = []
    for sample in samples or []:
        values = sample.get("residual_n") or sample.get("free_force_residual_norm_n") or []
        if len(values) < 2 or not _finite(values[0]) or values[0] <= 0:
            continue
        curves.append([v / values[0] if _finite(v) and v > 0 else None for v in values])
    length = max((len(c) for c in curves), default=0)
    med, geo, mx = [], [], []
    for k in range(length):
        column = [c[k] for c in curves if k < len(c) and c[k] is not None]
        if column:
            med.append((k, sorted(column)[len(column) // 2]))
            geo.append((k, math.exp(sum(math.log(v) for v in column) / len(column))))
            mx.append((k, max(column)))
        else:
            med.append((k, None))
            geo.append((k, None))
            mx.append((k, None))
    return _plot(
        "Validation force residual per trajectory relative to its own start: median, geometric mean, maximum (log scale)",
        [("Median", "#16804a", med), ("Geometric mean", "#2563eb", geo), ("Maximum", "#c63645", mx)],
        xlabel="optimizer iteration",
        log_y=True,
    )


def _board_html() -> str:
    """The 留言板 section (Anka, 2026-10-02): messages are comments on a GitHub gist (env LIDO_BOARD_GIST = gist id);
    the page lists the latest comments through GitHub's public API and links to the gist for posting. Empty when
    the variable is not set, so other dashboards are unchanged."""
    gist = os.environ.get("LIDO_BOARD_GIST", "").strip()
    if not gist:
        return ""
    url = f"https://gist.github.com/{html.escape(gist)}"
    return rf"""<section id="board"><h2>留言板 · message board</h2>
<p class="muted">Leave a message for Claude as a <a href="{url}#comments">comment on this gist</a>. Claude reads the board when it checks the training (about once an hour) and replies there, prefixed "Claude:". Latest comments (newest first):</p>
<div id="board-comments" class="muted">Loading comments…</div>
<script>
fetch("https://api.github.com/gists/{html.escape(gist)}/comments?per_page=100").then(r => r.json()).then(cs => {{
  const box = document.getElementById("board-comments");
  if (!Array.isArray(cs) || cs.length === 0) {{ box.textContent = "No messages yet."; return; }}
  box.className = "";
  box.innerHTML = cs.slice().reverse().map(c => {{
    const claude = /^\s*claude\s*:/i.test(c.body);
    const who = claude ? "Claude" : (c.user && c.user.login) || "?";
    const when = (c.created_at || "").replace("T", " ").replace("Z", " UTC");
    const text = c.body.replace(/^\s*claude\s*:\s*/i, "");
    const esc = t => t.replace(/[&<>]/g, ch => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;"}}[ch]));
    return `<div style="border-left:4px solid ${{claude ? "#2e86ab" : "#e4572e"}};background:${{claude ? "#f2f8fb" : "#fff6f3"}};padding:6px 12px;margin:8px 0;white-space:pre-wrap"><span class="muted">${{esc(who)}} · ${{esc(when)}}</span><br>${{esc(text)}}</div>`;
  }}).join("");
}}).catch(() => {{ document.getElementById("board-comments").textContent = "Comments could not be loaded (GitHub API); open the gist directly."; }});
</script></section>"""


def write_mixed_report(output, report, *, updated_at=None):
    """Write portable epoch metrics, four SVG plots and a page refreshing every 30s.

    Experimental. Only report metrics are written; checkpoints and trajectory
    state are never read. ``updated_at`` denotes the source metrics timestamp,
    while an optional ``publication`` block carries a separate mirror heartbeat.
    Epoch rows whose ``validation`` is None (skipped validation) leave gaps in
    the epoch plots and blank validation cells in ``epochs.csv``; the
    validation section shows the most recent validated epoch.

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
    validation_epoch, validation = _latest_summary(rows, "validation")
    validation = validation or {}
    relative = validation.get("relative_energy", [])
    endpoint = relative[-1] if relative else {}
    residual_curve = validation.get("force_residual", [])
    residual_endpoint = residual_curve[-1] if residual_curve else {}
    penetration_curve = validation.get("penetration", [])
    penetration_endpoint = penetration_curve[-1] if penetration_curve else {}
    selection = validation.get("selection") or {}
    best = report.get("best_selection") or {}
    selection_source = _selection_source(report)
    full_epoch, full = _latest_summary(rows, "full_horizon_validation")
    full = full or {}
    full_selection = full.get("selection") or {}
    # Label the shown validation by its own iteration count: after a resume with a changed budget
    # the latest validated row can predate the configured ``validation_iterations``.
    validation_iterations = residual_endpoint.get(
        "iteration", endpoint.get("iteration", config.get("validation_iterations", 100))
    )
    completed = report.get("completed_epochs", 0)
    maximum = config.get("max_epochs", 500)
    dashboard_title = os.environ.get("LIDO_DASHBOARD_TITLE", "").strip() or "LIDO — training the deformation optimizer"
    epochs_text = (
        f"{completed} completed epochs (open-ended run)"
        if int(maximum) >= 10000
        else f"{completed} / {maximum} completed epochs"
    )
    status = report.get("status", "preparing")
    phase = progress.get("phase", status)
    if status in ("failed", "interrupted", "epoch_limit", "early_stopped", "plateau_converged", "stalled"):
        phase = status

    def escape(value):
        return html.escape(str(value))

    validation_interval = config.get("validation_interval", 1)
    interval_text = (
        f" Validation runs every {escape(validation_interval)} epochs and on the final epoch."
        if isinstance(validation_interval, int) and validation_interval > 1
        else ""
    )
    counts_k = progress.get("available_K", latest.get("available_K", [1]))
    counts_h = progress.get("available_H", latest.get("available_H", [8]))
    pre_roll_text = _pre_roll_text(config, latest)
    loss_plot = _epoch_plot(rows, selection_source=selection_source)
    tick_loss_plot, tick_metric_plot = _tick_plots(report)
    validation_plot = _curve_plot("Validation relative physical energy", relative)
    residual_plot = _curve_plot("Validation free-corner force residual (N)", residual_curve)
    relative_residual_plot = _relative_residual_plot(residual_curve)
    per_trajectory_plot = _per_trajectory_relative_residual_plot(validation.get("samples"), None)
    penetration_plot = _penetration_plot(penetration_curve)
    _atomic_text(output / "loss_curve.svg", loss_plot)
    _atomic_text(output / "tick_loss_curve.svg", tick_loss_plot)
    _atomic_text(output / "tick_metric_curve.svg", tick_metric_plot)
    _atomic_text(output / "validation_curve.svg", validation_plot)
    _atomic_text(output / "residual_curve.svg", residual_plot)
    _atomic_text(output / "relative_residual_curve.svg", relative_residual_plot)
    _atomic_text(output / "per_trajectory_residual_curve.svg", per_trajectory_plot)
    _atomic_text(output / "penetration_curve.svg", penetration_plot)
    epoch_rows = [
        {
            **row,
            "selection_metric": _lookup(row, "validation", "selection", "metric"),
            "selection_eligible": _lookup(row, "validation", "selection", "eligible"),
            "physical_survivors": _lookup(row, "validation", "physical_survivors"),
            "sample_count": _lookup(row, "validation", "sample_count"),
            "validation_final_max_penetration_r": _final_penetration(row),
            "regime_stage": _lookup(row, "regime", "stage"),
            "regime_k_max": _lookup(row, "regime", "k_max"),
            "regime_h_max": _lookup(row, "regime", "h_max"),
            "regime_updates": _lookup(row, "regime", "updates"),
            "full_horizon_selection_metric": _lookup(row, "full_horizon_validation", "selection", "metric"),
            "full_horizon_selection_eligible": _lookup(row, "full_horizon_validation", "selection", "eligible"),
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
    board_html = _board_html()
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
    survival_text = _survival_text(validation)
    metric = selection.get("metric")
    metric_text = f"{_number(metric)} N" if _finite(metric) else "Unavailable"
    cheap_selects = selection_source == "cheap"
    eligibility_text = _eligibility_text(selection.get("eligible"), selecting=cheap_selects)
    best_text = _best_text(best, selection_source)
    history = report.get("best_selection_history") or []
    reset = history[-1] if history and isinstance(history[-1], dict) else None
    if reset:
        previous = reset.get("record") or {}
        best_text += (
            f" Selection restarted after epoch {escape(reset.get('reset_at_epoch', '—'))}"
            f" ({escape(reset.get('reason', 'reason not recorded'))}); "
        ) + (
            f"the earlier record was {_number(previous.get('metric'))} N at epoch {escape(previous.get('epoch', '—'))}."
            if _finite(previous.get("metric"))
            else "there was no earlier eligible record."
        )
    cheap_text = (
        f"{'Selection metric (cheap validation: ' if cheap_selects else 'Cheap validation metric ('}"
        f"mean free-corner force residual after {validation_iterations} iterations): {metric_text}, "
        f"{eligibility_text}. Physical survivors: {survival_text}."
    )
    if cheap_selects:
        selection_text = f"{cheap_text} {best_text}"
        selection_note = (
            "The selection metric is the mean final free-corner force residual of the cheap validation; "
            "squares mark full-horizon checks."
        )
    else:
        full_metric = full_selection.get("metric")
        full_metric_text = f"{_number(full_metric)} N" if _finite(full_metric) else "Unavailable"
        full_budget = (
            f"K = {escape(full.get('iterations', '—'))} learned iterations on each of "
            f"H = {escape(full.get('physical_steps', '—'))} physical steps, epoch {escape(full_epoch)}"
            if full
            else "not evaluated yet"
        )
        selection_text = (
            f"Selection metric (full-horizon check: mean final free-corner force residual after {full_budget}): "
            f"{full_metric_text}, {_eligibility_text(full_selection.get('eligible'), selecting=True)}. "
            f"Full-horizon survivors: {_survival_text(full)}. {best_text}<br>\n{cheap_text}"
        )
        selection_note = (
            "The best checkpoint is selected on the full-horizon check (squares: mean final free-corner force "
            "residual after the last physical step, every sample alive); the cheap validation's final-iteration "
            "residual is shown for comparison."
        )
    full_cap = config.get("validation_full_iterations")
    full_budget_text = (
        f"at the largest currently available H and at K capped at {escape(full_cap)}"
        if isinstance(full_cap, int) and not isinstance(full_cap, bool)
        else "at the largest currently available budgets"
    )
    if full:
        final_residual = full.get("final_free_force_residual_norm_n") or {}
        final_energy = full.get("final_energy_joule") or {}
        final_penetration = full.get("final_max_penetration_r") or {}
        full_html = f"""<table><tr><th>Epoch</th><th>K</th><th>H</th><th>Survivors</th><th>Final residual mean / median / max (N)</th><th>Final energy mean (J)</th><th>Final penetration mean / max (r)</th><th>Seconds</th></tr>
<tr><td>{escape(full_epoch)}</td><td>{escape(full.get("iterations", "—"))}</td><td>{escape(full.get("physical_steps", "—"))}</td><td>{escape(full.get("physical_survivors", "—"))} / {escape(full.get("sample_count", "—"))}</td><td>{_number(final_residual.get("mean"))} / {_number(final_residual.get("median"))} / {_number(final_residual.get("max"))}</td><td>{_number(final_energy.get("mean"))}</td><td>{_number(final_penetration.get("mean"))} / {_number(final_penetration.get("max"))}</td><td>{_number(full.get("seconds"))}</td></tr></table>"""
    else:
        full_html = '<p class="muted">No full-horizon validation has completed yet.</p>'
    residual_table = (
        f"<tr><td>Force residual (N)</td><td>{_number(residual_endpoint.get('mean'))}</td><td>{_number(residual_endpoint.get('median'))}</td><td>{_number(residual_endpoint.get('max'))}</td></tr>"
        if residual_endpoint
        else ""
    )
    if penetration_endpoint:
        residual_table += f"<tr><td>Deepest penetration (r)</td><td>{_number(penetration_endpoint.get('mean'))}</td><td>—</td><td>{_number(penetration_endpoint.get('max'))}</td></tr>"
    scene_fraction = latest.get("contact_scene_fraction")
    realized_fraction = latest.get("contact_realized_fraction")
    training_penetration = latest.get("contact_max_penetration_r")
    realized_text = (
        f"; trajectories that made contact (at least one detected pair): {realized_fraction:.1%}"
        if _finite(realized_fraction)
        else ""
    )
    contact_text = (
        f"Contact scenes among this epoch's rank-0 trajectories: {scene_fraction:.1%}{realized_text}; "
        f"deepest training penetration: {_number(training_penetration)} r."
        if _finite(scene_fraction)
        else "Contact statistics are not recorded for this epoch."
    )
    # The heartbeat carries the in-progress epoch's regime block; a completed row is the fallback.
    regime = progress.get("regime") or latest.get("regime") or {}
    if config.get("regime") == "fixed_states" or regime.get("name") == "fixed_states":
        budget_text = _fixed_state_budget_text(config, regime)
        schedule_text = _fixed_state_schedule_text(config)
    else:
        budget_text = f"{escape(config.get('queries_per_epoch', '—'))} training queries per epoch"
        schedule_text = f"Curriculum descent gate: {gate_text} · Hard cap: {escape(stage_limit_text)}"
    origin_text = _origin_text(report.get("initialized_from"))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30"><title>{escape(dashboard_title)} · live training</title>
<style>
*{{box-sizing:border-box}}body{{font:17px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#20323b;background:#f8fafb}}
a{{color:#087c91}}img,svg{{display:block;width:100%;height:auto;background:white;border-radius:12px}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}
.status{{background:#e6f2f4;padding:18px;border-radius:10px}}h1{{line-height:1.2}}h2{{font-size:22px}}.muted{{color:#526174;font-size:14px}}
.legend span{{margin-right:20px}}.failure{{padding:18px;border:1px solid #c63645;border-radius:10px;background:#fff5f5}}code{{font-size:13px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #e5eaf2;text-align:left}}details{{margin:18px 0}}summary{{cursor:pointer}}section{{margin:28px 0}}
@media(max-width:600px){{table{{table-layout:fixed;font-size:14px}}td,th{{padding:6px 4px;overflow-wrap:anywhere}}}}
</style></head><body><main>
<a href="/artifacts/learned-intrinsic-solver/index.html">← All solver experiments</a>
<h1>{escape(dashboard_title)}</h1>
<p class="status"><strong>{escape(phase.replace("_", " ").capitalize())}</strong> · {epochs_text} · {progress.get("completed_updates", report.get("completed_updates", 0))} Adam updates<br>
{budget_text} and {escape(config.get("validation_count", validation.get("sample_count", "—")))} fixed validation states.{interval_text} {escape(batch_text)}<br>
Available solver iterations: K = {escape(counts_k)} · Physical timesteps: H = {escape(counts_h)}<br>
{pre_roll_text}{schedule_text}{origin_text}</p>
<p>{selection_text}<br>
Validation energy: {_number(validation.get("mean_before_joule"))} → {_number(validation.get("mean_after_joule"))} J after one update. Descent: {descent_text}; first-update failures: {escape(validation.get("first_update_failed_count", "Not evaluated"))}; all validation failures: {escape(validation.get("failed_count", "Not evaluated"))}.</p>
{board_html}
{failure_html}<img src="loss_curve.svg" alt="Training objective and validation first-update objective, validation descent rate, physical energy, selection metric, physical survivors, and learning rate by epoch">
<section><h2>Within-epoch history</h2>
<p class="muted">Every {escape(config.get("tick_updates", 0))} updates all ranks pause and rank 0 records a tick: the running training loss since the previous tick and a cheap validation on {escape(config.get("tick_scene_count", 4))} held-out scenes ({escape(len(report.get("ticks") or []))} ticks so far). Epoch-end values are drawn in red.</p>
<img src="tick_loss_curve.svg" alt="Training loss per tick and per epoch"><img src="tick_metric_curve.svg" alt="Cheap validation metric per tick and per epoch"></section>
<p class="muted">Lower objective is better. The training objective includes an uphill penalty; validation shows the first update on fixed seeds with the same form. {selection_note} The learning rate is recorded after each epoch's scheduler decision.</p>
<details><summary>How the loss curves are computed</summary>
<p>Mean local training loss averages all queried trajectories and ranks in each completed epoch. Epochs mix solver ages, physical timesteps and curriculum stages.</p>
<p>Per-update loss = asinh(E_after / s) + &lambda; · max((E_after &minus; E_before) / s, 0) with s = max(|E_before|, floor), where E_before is the energy immediately before that update and floor = c · 2<sup>&minus;23</sup> · V · (&lambda;<sub>Lam&eacute;</sub> + 2&mu; + &eta;/dt + &rho;h<sup>2</sup>/dt<sup>2</sup>) is the material-aware float32 energy floor (c = {escape(config.get("energy_floor_scale", 1.0))}). The penalty weight is &lambda; = {escape(config.get("energy_increase_weight", 1.0))}.</p>
<p>Validation reports the same per-update loss for the first update of each fixed seed. Missing measurements leave gaps. Training descent rate is not recorded; the descent panel shows validation only.</p></details>
<section><h2>Latest validation: {validation_iterations} optimizer iterations</h2>
<p class="muted">Epoch {escape(validation_epoch if validation_epoch is not None else "—")}, {escape(validation.get("sample_count", 0))} fixed validation seeds. Each energy value is a per-trajectory physical energy ratio Eᵢ / E₀; the residual is the Euclidean norm of the free-corner position gradient [N]. Mean, median and maximum aggregate the per-trajectory values. Network weights stay fixed during validation.</p>
<div class="legend"><span style="color:#2563eb">● Mean</span><span style="color:#16804a">● Median</span><span style="color:#c63645">● Maximum</span></div>{validation_plot}
{residual_plot}
{relative_residual_plot}
<p class="muted">Relative residual: the mean (median, maximum) free-corner force norm over the validation trajectories at iteration k divided by the same statistic before the first optimizer iteration, on a logarithmic axis; 1 means no reduction. The mean is dominated by the largest trajectory whenever the residuals are heavy-tailed, so it then coincides with the maximum.</p>
{per_trajectory_plot}
<p class="muted">Per-trajectory relative residual: every trajectory is first divided by its own initial residual, then the median, the geometric mean (the mean on the logarithmic axis) and the maximum over trajectories are taken per iteration.</p>
{penetration_plot}
<table><tr><th>At iteration {escape(endpoint.get("iteration", validation_iterations))}</th><th>Mean</th><th>Median</th><th>Maximum</th></tr><tr><td>Relative energy</td><td>{_number(endpoint.get("mean"))}</td><td>{_number(endpoint.get("median"))}</td><td>{_number(endpoint.get("max"))}</td></tr>{residual_table}</table>
<p class="muted">Failed validation trajectories: {escape(validation.get("failed_count", "Not evaluated"))}; physical survivors: {survival_text}; near-zero initial energies: {escape(endpoint.get("near_zero_count", 0))}. Optimizer failures leave gaps from the failed iteration onward; near-zero initial energies are excluded from relative ratios but keep their residuals. Physical-rollout failures are counted separately in the report.<br>
The penetration curve is the deepest penetration of any surface sample into its frozen contact partners, in units of the sample radius r (zero for contact-free scenes). {escape(contact_text)}</p>
<h2>Latest full-horizon validation</h2>
<p class="muted">Held-out seeds run K learned iterations on each of H physical steps {full_budget_text}; every {escape(config.get("validation_full_interval", "—"))} epochs and before curriculum advancement.</p>
{full_html}</section>
<p><a href="report.json">Metrics JSON</a> · <a href="progress.json">Live progress</a> · <a href="epochs.csv">Epoch CSV</a> · <a href="updates.csv">Update CSV</a> · <a href="loss_curve.svg">Training SVG</a> · <a href="tick_loss_curve.svg">Tick loss SVG</a> · <a href="tick_metric_curve.svg">Tick metric SVG</a> · <a href="validation_curve.svg">Validation SVG</a> · <a href="residual_curve.svg">Residual SVG</a> · <a href="relative_residual_curve.svg">Relative residual SVG</a> · <a href="per_trajectory_residual_curve.svg">Per-trajectory residual SVG</a> · <a href="penetration_curve.svg">Penetration SVG</a></p>
<details><summary>Configuration</summary><p>{damping_text}</p><pre>{escape(json.dumps(config, indent=2))}</pre></details>
<p class="muted">This page refreshes every 30 seconds. Curves update after each completed epoch.<br>
Epoch in progress: {escape(progress.get("epoch", "Not started"))}. Training heartbeat: {escape(progress.get("updated_at", "Waiting"))}.<br>
Epoch metrics: {escape(report["updated_at"])}. Page published: {escape(report.get("publication", {}).get("updated_at", "Local report"))}.</p>
</main></body></html>"""
    _atomic_text(output / "report.json", json.dumps(report, indent=2) + "\n")
    if "progress" in report:
        _atomic_text(output / "progress.json", json.dumps(progress, indent=2) + "\n")
    _atomic_text(output / "index.html", page)
