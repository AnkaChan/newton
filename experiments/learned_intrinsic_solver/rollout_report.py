# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Combine independent validation trajectories and plot relative physical energy."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

__all__ = ["build_report", "summarize_rollout"]


def summarize_rollout(energies: np.ndarray) -> list[dict]:
    """Normalize each trajectory by its own initial energy, then aggregate."""
    energies = np.asarray(energies, dtype=np.float64)
    if energies.ndim != 2 or min(energies.shape) < 1:
        raise ValueError("energies must be [iteration, validation sample]")
    if not np.all(np.isfinite(energies[0]) & (energies[0] > 0)):
        raise ValueError("initial energy must be positive and finite for every query")
    relative = energies / energies[0]
    rows = []
    for iteration, values in enumerate(relative):
        valid = np.isfinite(values)
        finite = values[valid]
        row = {"iteration": iteration, "valid_count": int(valid.sum()), "failed_count": int((~valid).sum())}
        for name, function in (("max", np.max), ("mean", np.mean), ("median", np.median)):
            value = float(function(finite)) if len(finite) else None
            row[name] = value if valid.all() else None
            row[f"valid_only_{name}"] = value
        rows.append(row)
    return rows


def _write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def build_report(output: Path, *, world_size: int = 4) -> dict:
    """Validate rank coverage, save complete-population statistics, and plot them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    metadata, energies, seeds = [], [], []
    for rank in range(world_size):
        record = json.loads((output / f"rank_{rank}.json").read_text())
        with np.load(output / f"rank_{rank}.npz") as arrays:
            energies.append(arrays["energies"])
            seeds.append(arrays["physical_seeds"])
            np.testing.assert_allclose(
                arrays["relative_energies"],
                arrays["energies"] / arrays["energies"][0],
                rtol=2e-7,
                atol=1e-8,
                equal_nan=True,
            )
        metadata.append(record)
    if len({entry["checkpoint_sha256"] for entry in metadata}) != 1:
        raise ValueError("ranks evaluated different checkpoints")
    if len({entry["iterations"] for entry in metadata}) != 1:
        raise ValueError("ranks used different iteration counts")
    if not all(entry["parameter_state_unchanged"] for entry in metadata):
        raise ValueError("a validation worker modified the checkpoint weights")
    seeds = np.concatenate(seeds).astype(np.int64)
    energy = np.concatenate(energies, axis=1).astype(np.float64)
    order = np.argsort(seeds)
    seeds, energy = seeds[order], energy[:, order]
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("validation shards contain repeated physical seeds")
    expected = np.arange(10000, 10512)
    if world_size == 4 and not np.array_equal(seeds, expected):
        raise ValueError("full campaign report requires exactly the 512 saved validation seeds")
    rows = summarize_rollout(energy)
    ratios = energy / energy[0]
    np.savez_compressed(output / "trajectories.npz", physical_seeds=seeds, energies=energy, relative_energies=ratios)
    step1_loss = (
        float(np.mean((energy[1] - energy[0]) / np.maximum(energy[0], 1.0))) if np.isfinite(energy[1]).all() else None
    )
    expected_step1 = metadata[0].get("checkpoint_validation_mean_normalized_loss")
    step1_error = abs(step1_loss - expected_step1) if step1_loss is not None and expected_step1 is not None else None
    if step1_error is not None and step1_error > 2e-6:
        raise ValueError("first iteration disagrees with the checkpoint's saved validation")
    best_mean = min((row for row in rows if row["mean"] is not None), key=lambda row: row["mean"])
    report = {
        "checkpoint_epoch": metadata[0]["checkpoint_epoch"],
        "checkpoint_sha256": metadata[0]["checkpoint_sha256"],
        "iterations": energy.shape[0] - 1,
        "sample_count": len(seeds),
        "normalization": "Each sample's current implicit Euler energy divided by its own initial energy; no denominator floor.",
        "aggregation": "Maximum, arithmetic mean, and median of per-sample energy ratios at each iteration.",
        "full_population_statistics": "Unavailable after any failed trajectory; valid-only statistics are separately labelled.",
        "initial_energy_range_joule": [float(energy[0].min()), float(energy[0].max())],
        "step1_normalized_change_for_training_comparison": step1_loss,
        "step1_comparison_absolute_error": step1_error,
        "best_mean_iteration": best_mean["iteration"],
        "best_mean_relative_energy": best_mean["mean"],
        "final_states_above_initial_energy": int(np.sum(np.isfinite(ratios[-1]) & (ratios[-1] > 1))),
        "rows": rows,
        "ranks": metadata,
        "failures": [failure for entry in metadata for failure in entry["failures"]],
    }
    _write(output / "report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _write(output / "iterations.csv", table.getvalue())
    raw = io.StringIO()
    writer = csv.writer(raw)
    writer.writerow(["iteration", "physical_seed", "energy_joule", "relative_energy"])
    for iteration in range(len(rows)):
        for column, seed in enumerate(seeds):
            values = energy[iteration, column], ratios[iteration, column]
            writer.writerow([iteration, int(seed), *[float(value) if np.isfinite(value) else "" for value in values]])
    _write(output / "per_sample.csv", raw.getvalue())
    has_failures = any(row["failed_count"] for row in rows)
    x = [row["iteration"] for row in rows]
    colors = {"max": "#c34538", "mean": "#167f9b", "median": "#7d4bb5"}
    for scale in ("linear", "log"):
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for name, color in colors.items():
            axis.plot(x, [row[name] for row in rows], label=name.capitalize(), color=color, linewidth=2)
            if has_failures:
                axis.plot(
                    x,
                    [row[f"valid_only_{name}"] for row in rows],
                    label=f"{name.capitalize()} — valid only",
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                )
        axis.axhline(1, color="#777777", linewidth=1, linestyle=":", label="Initial energy")
        axis.set(xlabel="Learned solver iteration", ylabel="Current energy / initial energy", xlim=(0, x[-1]))
        axis.set_yscale(scale)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2, which="both")
        axis.legend()
        axis.set_title(
            f"{len(seeds)} fixed validation states · epoch {report['checkpoint_epoch']} model\n{report['iterations']} consecutive solver iterations · weights unchanged"
        )
        for extension in ("png", "svg"):
            fig.savefig(output / f"relative_energy_{scale}.{extension}", dpi=160)
        plt.close(fig)
    if has_failures:
        fig, axis = plt.subplots(figsize=(11, 3), constrained_layout=True)
        axis.plot(x, [row["failed_count"] for row in rows], color="#c34538")
        axis.set(xlabel="Learned solver iteration", ylabel="Failed trajectories", xlim=(0, x[-1]))
        axis.grid(alpha=0.2)
        fig.savefig(output / "failures.svg")
        plt.close(fig)
    selected = [iteration for iteration in (0, 1, 2, 3, 5, 10, 20, 50, 100) if iteration < len(rows)]
    cells = []
    for iteration in selected:
        row = rows[iteration]
        values = [f"{row[name]:.6g}" if row[name] is not None else "undefined" for name in ("max", "mean", "median")]
        cells.append(
            f"<tr><td>{iteration}</td><td>{values[0]}</td><td>{values[1]}</td><td>{values[2]}</td><td>{row['valid_count']}</td></tr>"
        )
    failure_note = (
        '<p class="warning">Some trajectories became invalid. Full-set statistics stop at that point. Dashed curves describe only surviving valid trajectories; they must not be interpreted as statistics over all 512 states.</p><img src="failures.svg" alt="Cumulative failed trajectories">'
        if has_failures
        else '<p class="success">All validation trajectories remained valid through every plotted iteration.</p>'
    )
    result_note = ""
    if not has_failures:
        result_note = (
            f"<p>The mean ratio reaches its lowest value, <strong>{best_mean['mean']:.4f}</strong>, "
            f"at iteration <strong>{best_mean['iteration']}</strong>. At iteration {rows[-1]['iteration']}, "
            f"the mean is <strong>{rows[-1]['mean']:.4f}</strong>, and "
            f"{report['final_states_above_initial_energy']} of {len(seeds)} states have more energy than initially.</p>"
        )
    content = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>100 solver iterations · validation energy</title><style>
body{{font:17px/1.55 system-ui,sans-serif;color:#223844;background:#f7fafb;max-width:1120px;margin:30px auto;padding:0 20px}}
h1{{line-height:1.2}} a{{color:#087b91}} img{{width:100%;background:#fff;border-radius:12px}} button{{padding:9px 18px;margin:6px;border:1px solid #bacbd2;background:white;border-radius:6px;cursor:pointer}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{padding:8px 12px;text-align:right;border-bottom:1px solid #dbe3e7}} .success{{background:#e7f3ec;padding:14px}} .warning{{background:#fff0dc;padding:14px}}
</style></head><body><a href="../index.html">← All solver experiments</a><h1>Relative energy over 100 solver iterations</h1>
<p>{len(seeds)} fixed validation states · saved epoch-{report["checkpoint_epoch"]} model · no training updates</p>
<p>For each state, divide its energy at the current iteration by its energy at iteration 0. The curves show the maximum, mean, and median of those individual ratios. They all start at 1. Lower is better.</p>
<div><button onclick="document.getElementById('curve').src='relative_energy_linear.svg'">Linear scale</button><button onclick="document.getElementById('curve').src='relative_energy_log.svg'">Log scale</button></div>
<img id="curve" src="relative_energy_linear.svg" alt="Maximum, mean, and median relative energy over solver iterations">
{result_note}{failure_note}<table><thead><tr><th>Iteration</th><th>Maximum</th><th>Mean</th><th>Median</th><th>Valid states</th></tr></thead><tbody>{"".join(cells)}</tbody></table>
<p>Each iteration predicts local deformation axes, fuses a new global shape, and feeds that shape into the next iteration. Cell frames are recomputed each time. The original inertial target and fixed corners stay unchanged. Physical time does not advance.</p>
<p>The measured quantity is implicit Euler energy: Neo-Hookean elastic energy plus the inertial term. This ratio does not subtract the unknown minimum energy, so zero is not necessarily attainable. No line search or repair is applied to the learned proposals.</p>
<p><a href="iterations.csv">Curve data (CSV)</a> · <a href="per_sample.csv">Every trajectory (CSV)</a> · <a href="trajectories.npz">Trajectory arrays (NPZ)</a> · <a href="report.json">Full report</a></p>
<p>Download plots: <a href="relative_energy_linear.png">PNG</a> · <a href="relative_energy_linear.svg">SVG</a> · <a href="relative_energy_log.svg">Log-scale SVG</a>. <a href="../training-large/index.html">Training curves and saved checkpoints</a>.</p>
<details><summary>Failures</summary><pre>{html.escape(json.dumps(report["failures"], indent=2))}</pre></details></body></html>"""
    _write(output / "index.html", content)
    return report


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    result = build_report(args.output, world_size=args.world_size)
    print(json.dumps({key: result[key] for key in ("checkpoint_epoch", "iterations", "sample_count")}, indent=2))


if __name__ == "__main__":
    _main()
