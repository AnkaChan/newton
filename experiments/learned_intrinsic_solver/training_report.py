# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Atomic epoch reports and loss plots for the experimental training campaign."""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path


def _write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_report(output: Path, report: dict):
    """Write a self-contained report, machine-readable history, and epoch plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = report.get("history", [])
    _write(output / "report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = ["epoch", "optimizer_updates", "learning_rate", "epoch_seconds"]
    metrics = ["mean_normalized_loss", "mean_before_joule", "mean_after_joule", "descent_rate", "failed_count"]
    columns += [f"{split}_{key}" for split in ("train", "validation") for key in metrics]
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=columns)
    writer.writeheader()
    for row in history:
        values = {key: row.get(key) for key in columns[:4]}
        values.update(
            {f"{split}_{key}": row.get(split, {}).get(key) for split in ("train", "validation") for key in metrics}
        )
        writer.writerow(values)
    _write(output / "epochs.csv", table.getvalue())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    for split, color in (("train", "#157f94"), ("validation", "#d26a25")):
        values = [row[split] for row in history]
        split_epochs = epochs
        if split == "validation" and report.get("validation_initial"):
            values = [report["validation_initial"], *values]
            split_epochs = [0, *epochs]
        losses = [value.get("mean_normalized_loss") for value in values]
        descent = [value.get("descent_rate", 0) * 100 for value in values]
        axes[0, 0].plot(
            split_epochs,
            [float("nan") if value is None else value * 100 for value in losses],
            label=split,
            color=color,
            marker=".",
        )
        axes[0, 1].plot(split_epochs, descent, label=split, color=color, marker=".")
    axes[0, 0].set_ylabel("Mean normalized energy change (%)")
    axes[0, 0].axhline(0, color="grey", linewidth=0.7)
    axes[0, 0].legend()
    axes[0, 1].set_ylabel("Queries with lower energy (%)")
    axes[0, 1].set_ylim(-2, 102)
    for key, label in (("mean_before_joule", "Before update"), ("mean_after_joule", "After update")):
        values = [row["validation"].get(key) for row in history]
        axes[1, 0].plot(epochs, [float("nan") if value is None else value for value in values], label=label, marker=".")
    axes[1, 0].set_ylabel("Validation mean physical energy (J)")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, [row["learning_rate"] for row in history], color="#7052a3", marker=".")
    axes[1, 1].set_ylabel("Adam learning rate")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xlabel("Completed epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
    fig.suptitle("Learned intrinsic solver — one optimizer iteration per query")
    for extension in ("png", "svg"):
        target = output / f"loss_curve.{extension}"
        temporary = target.with_name(target.name + ".tmp")
        fig.savefig(temporary, format=extension, dpi=150)
        temporary.replace(target)
    plt.close(fig)
    escape = lambda value: html.escape(str(value))  # noqa: E731
    recent = history[-1] if history else None
    validation = recent["validation"] if recent else report.get("validation_initial")
    details = "Waiting for initial validation."
    if validation:
        details = (
            f"Validation energy: {escape(validation.get('mean_before_joule'))} → "
            f"{escape(validation.get('mean_after_joule'))} J. "
            f"Descent: {validation.get('descent_rate', 0):.1%}; invalid queries: "
            f"{escape(validation.get('failed_count'))}."
        )
    config = report.get("config", {})
    failure = report.get("failure")
    failure_html = (
        "" if not failure else f"<h2>Failure diagnostic</h2><pre>{escape(json.dumps(failure, indent=2))}</pre>"
    )
    checkpoints = [
        path
        for path in (output / "checkpoints").glob("*.pt")
        if path.name in ("latest.pt", "best_validation.pt", "final.pt", "failure.pt")
    ]
    checkpoint_html = " · ".join(
        f'<a href="checkpoints/{escape(path.name)}">{escape(path.name)}</a>' for path in sorted(checkpoints)
    )
    if config.get("early_stopping", True):
        stopping_description = (
            f"The run checks for a plateau after at least {config.get('min_epochs', 30)} epochs, "
            f"with a cap of {config.get('max_epochs', 200)}. A plateau counts as convergence only with "
            "five clean validations, at least 95% descent, and lower mean physical energy. "
            "A poor plateau is reported as stalled."
        )
    else:
        stopping_description = (
            f"The target is {config.get('max_epochs', 200)} epochs. Plateau early stopping is disabled; "
            "learning-rate reductions, validation, and checkpoint saving remain enabled. "
            "Reaching the epoch target does not establish convergence."
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<title>Learned intrinsic solver — larger training</title><style>
body{{font:17px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#20323b;background:#f8fafb}}
a{{color:#087c91}} img{{width:100%;background:white;border-radius:12px}} pre{{white-space:pre-wrap;font-size:13px}}
.status{{background:#e6f2f4;padding:18px;border-radius:10px}} h1{{line-height:1.2}}
</style></head><body><p><a href="../index.html">← All solver experiments</a></p>
<h1>Training the one-step optimizer</h1><p class="status"><strong>{escape(report.get("status", "preparing"))}</strong>
 · {report.get("completed_epochs", 0)} completed epochs · {report.get("optimizer_updates", 0)} Adam updates</p>
<p>{report.get("train_count", config.get("train_count", "?")):,} training states and
{report.get("validation_count", config.get("validation_count", "?")):,} fixed validation states.
{report.get("world_size", 1)} GPUs, batch {report.get("batch_size", config.get("batch_size", "?"))} per GPU
(global {report.get("global_batch_size", "?")}). Float32, TF32 and mixed precision disabled.</p>
<p>{details}</p><img src="loss_curve.svg" alt="Training and validation loss, descent rate, raw energy, and learning rate by epoch">
<p>Lower normalized energy change is better. Zero means no improvement. Each epoch visits every training state once,
with fresh candidate perturbations. Validation candidates remain fixed. Invalid validation outputs produce gaps in loss curves
and count against the descent rate.</p><p>{stopping_description}</p>
<p><a href="epochs.csv">Epoch data (CSV)</a> · <a href="report.json">Full report (JSON)</a> ·
<a href="loss_curve.png">Loss plot (PNG)</a> · <a href="loss_curve.svg">Loss plot (SVG)</a></p>
<p>{checkpoint_html}</p>{failure_html}<details><summary>Configuration</summary><pre>{escape(json.dumps(config, indent=2))}</pre></details>
<p>This page refreshes every 60 seconds. Reports are updated after each completed epoch.</p></body></html>"""
    _write(output / "index.html", document)
