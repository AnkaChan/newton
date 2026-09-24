# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build plots and an interactive report from completed float32 round trips."""

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import numpy as np

from .data import generate_cuboid
from .round_trip_batch import Float32Decoder, analyze_result, compute_frames_float32

__all__ = ["build_batch_report", "summarize_samples"]


def summarize_samples(samples: list[dict]) -> dict:
    """Summarize completed cases while counting every unsuccessful seed."""
    complete = [sample for sample in samples if sample["status"] == "complete"]
    keys = list(complete[0]["metrics"]) if complete else []
    statistics = {}
    for key in keys:
        values = np.array([sample["metrics"][key] for sample in complete], dtype=np.float64)
        statistics[key] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95, method="linear")),
            "max": float(values.max()),
            "worst_seed": complete[int(values.argmax())]["seed"],
        }
    return {
        "requested_count": len(samples),
        "complete_count": len(complete),
        "failure_count": len(samples) - len(complete),
        "failed_seeds": [sample["seed"] for sample in samples if sample["status"] != "complete"],
        "statistics": statistics,
        "p95_definition": "NumPy linear interpolation of the empirical 95th percentile",
        "scale_counts": {
            str(scale): sum(sample["effective_scale"] == scale for sample in complete)
            for scale in sorted({sample["effective_scale"] for sample in complete})
        },
    }


def _plots(output, samples):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.labelcolor": "#304d60",
            "text.color": "#203b4c",
            "svg.fonttype": "none",
        }
    )
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    rms = np.array([item["metrics"]["corner_rmse_mm"] for item in samples])
    maximum = np.array([item["metrics"]["corner_max_error_mm"] for item in samples])
    deformation = 1000 * np.array([item["metrics"]["original_displacement_rmse_m"] for item in samples])
    scales = np.array([item["effective_scale"] for item in samples])
    colors = dict(zip(sorted(set(scales)), ("#1689a7", "#de8950", "#7973bb"), strict=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for ax, values, label in zip(
        axes, (rms, maximum), ("Corner RMS error (mm)", "Maximum corner error (mm)"), strict=True
    ):
        ax.hist(values, bins=12, color="#2a8fac", edgecolor="white")
        ax.axvline(np.median(values), color="#e68b48", label=f"Median {np.median(values):.3f} mm")
        ax.set(xlabel=label, ylabel="Number of samples")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(plots / "error_histograms.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for scale in sorted(set(scales)):
        mask = scales == scale
        axes[0].scatter(
            deformation[mask],
            rms[mask],
            s=30,
            color=colors[scale],
            alpha=0.85,
            label=f"Scale {scale:g} ({mask.sum()} samples)",
        )
    axes[0].set(xlabel="Original displacement RMS (mm)", ylabel="Corner recovery RMS error (mm)")
    axes[0].legend(frameon=False, fontsize=9)
    for index in np.argsort(rms)[-3:]:
        axes[0].annotate(
            str(samples[index]["seed"]),
            (deformation[index], rms[index]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    groups = [rms[scales == scale] for scale in sorted(set(scales))]
    axes[1].boxplot(
        groups,
        tick_labels=[f"{scale:g}x\n(n={len(group)})" for scale, group in zip(sorted(set(scales)), groups, strict=True)],
        patch_artist=True,
        boxprops={"facecolor": "#c6e5ed"},
        medianprops={"color": "#b35e29"},
    )
    axes[1].set(xlabel="Effective augmentation scale after backtracking", ylabel="Corner recovery RMS error (mm)")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.savefig(plots / "deformation_and_backtracking.svg")
    plt.close(fig)
    null_rms = 1000 * np.array([item["metrics"]["null_error_rmse_m"] for item in samples])
    row_rms = 1000 * np.array([item["metrics"]["rowspace_error_rmse_m"] for item in samples])
    equations = np.array([item["metrics"]["equation_component_rms"] for item in samples])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")
    axes[0].scatter(null_rms, rms, s=24, color="#1689a7", label="Actual float32 solve")
    bounds = [0, 1.07 * max(rms)]
    axes[0].plot(bounds, bounds, color="#8899a6", linestyle="--", linewidth=1, label="Equal RMS")
    axes[0].set(
        xlabel="Error projected onto unresolved modes (mm RMS)",
        ylabel="Total corner error (mm RMS)",
        xlim=bounds,
        ylim=bounds,
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].scatter(equations, row_rms * 1000, s=26, color="#bc7447")
    axes[1].set(xlabel="True normalized equation RMS residual", ylabel="Remaining row-space error (µm RMS)")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.savefig(plots / "numerical_and_null_errors.svg")
    plt.close(fig)


def _selected_viewer(output, settings, selected):
    root = Path(__file__).parent
    destination = output / "selected-cases"
    shutil.copytree(root / "round_trip_web", destination, dirs_exist_ok=True)
    shutil.copytree(root / "inspector_web" / "vendor", destination / "vendor", dirs_exist_ok=True)
    (destination / "data").mkdir(exist_ok=True)
    rest = generate_cuboid(tuple(settings["cell_counts"]), cell_size=settings["cell_size_m"])
    payload = {
        "rest_positions": rest.corner_rest_positions.tolist(),
        "cells": rest.cell_corner_indices.tolist(),
        "fixed_indices": np.flatnonzero(rest.corner_rest_positions[:, 2] == 0).tolist(),
        "cases": [],
    }
    report = {
        "cell_counts": settings["cell_counts"],
        "corner_count": settings["corner_count"],
        "cases": [],
        "small_grid_rank_audit": {
            "centers_axes": {
                "free_scalar_corner_unknowns": 27,
                "rows": 48,
                "rank": 24,
                "nullity_per_world_component": 3,
            }
        },
    }
    for label, entry in selected:
        case_id = f"seed_{entry['seed']:03d}"
        with np.load(output / entry["archive"], allow_pickle=False) as archive:
            payload["cases"].append(
                {
                    "id": case_id,
                    "original": archive["original_positions"].tolist(),
                    "recovered": {"centers_axes": archive["recovered_positions"].tolist()},
                }
            )
        shutil.copy2(output / entry["archive"], destination / "data" / f"{case_id}.npz")
        report["cases"].append(
            {
                "id": case_id,
                "label": f"{label} · seed {entry['seed']}",
                "description": f"{label} by corner RMS among the 100 float32 samples. Effective augmentation scale {entry['effective_scale']:g}; original displacement RMS {1000 * entry['metrics']['original_displacement_rmse_m']:.3f} mm.",
                "polar_F_max_abs_error": entry["metrics"]["polar_F_max_abs_error"],
                "decoders": {"centers_axes": entry["metrics"]},
            }
        )
    payload["report"] = report
    (destination / "data.js").write_text(
        "window.ROUND_TRIP_DATA=" + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n"
    )
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    page = (destination / "index.html").read_text()
    page = page.replace('<option value="axes">R + U + fixed end</option>', "")
    page = page.replace('href="data/multiscale_0.npz"', f'href="data/seed_{selected[0][1]["seed"]:03d}.npz"')
    page = page.replace(
        "LSQR tolerances are 10⁻¹³.",
        "LSQR tolerances are 10⁻⁷; geometry, frames, matrices, iterates and output are float32.",
    )
    page = page.replace('href="source/round_trip.py"', 'href="../source/round_trip_batch.py"')
    page = page.replace('href="reproduce.md"', 'href="../reproduce.md"')
    page = page.replace("The table uses the primary decoder:", "These three cases use the float32 primary decoder:")
    diagnostic_start = page.index('  <details class="method">\n    <summary>Secondary diagnostic:')
    diagnostic_end = page.index("</details>", diagnostic_start) + len("</details>")
    page = page[:diagnostic_start] + page[diagnostic_end:]
    page = page.replace("<main>", '<main><nav><a href="../index.html">← Back to the 100-sample analysis</a></nav>')
    page = page.replace("to floating-point precision", "to float32 accuracy")
    (destination / "index.html").write_text(page)
    script = (destination / "main.js").read_text()
    script = script.replace("q('#rank-audit').textContent=", "if(q('#rank-audit')) q('#rank-audit').textContent=")
    script += "\nconst requestedSeed=new URLSearchParams(location.search).get('seed');\nif(requestedSeed!==null){const requestedId='seed_'+requestedSeed.padStart(3,'0');if(report.cases.some(item=>item.id===requestedId)){caseSelect.value=requestedId;update({fit:true});}}\n"
    (destination / "main.js").write_text(script)


def _tolerance_audit(output, worst, settings):
    path = output / "tolerance_audit.json"
    if path.exists():
        return json.loads(path.read_text())
    rest = generate_cuboid(tuple(settings["cell_counts"]), cell_size=settings["cell_size_m"])
    decoder = Float32Decoder(rest)
    with np.load(output / worst["archive"], allow_pickle=False) as archive:
        original = archive["original_positions"]
    encoded = compute_frames_float32(rest, original)
    recovered, displacement, rhs, diagnostics = decoder.decode(encoded, original[decoder.fixed], tolerance=1e-8)
    result = {
        "seed": worst["seed"],
        "purpose": "One extra actual float32 solve of the worst-RMS seed checks tolerance sensitivity; the 100-case statistics retain 1e-7.",
        "main_tolerance": 1e-7,
        "main_metrics": worst["metrics"],
        "check_tolerance": 1e-8,
        "check_metrics": analyze_result(decoder, original, encoded, recovered, displacement, rhs),
        "check_solver": diagnostics,
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_batch_report(output: Path) -> dict:
    """Analyze persisted results without rerunning the 100 reconstruction solves."""
    output = Path(output)
    batch = json.loads((output / "batch.json").read_text())
    samples = batch["samples"]
    summary = summarize_samples(samples)
    complete = [item for item in samples if item["status"] == "complete"]
    if not complete:
        raise ValueError("No complete cases to plot; inspect batch.json failures")
    ordered = sorted(complete, key=lambda item: item["metrics"]["corner_rmse_m"])
    selected = [
        ("Worst RMS", ordered[-1]),
        ("Median representative", ordered[(len(ordered) - 1) // 2]),
        ("Best RMS", ordered[0]),
    ]
    audit = _tolerance_audit(output, ordered[-1], batch["settings"])
    report = {
        "title": "100 float32 shape round trips",
        "settings": batch["settings"],
        "summary": summary,
        "samples": samples,
        "selected_cases": [{"role": label, "seed": entry["seed"]} for label, entry in selected],
        "median_representative_definition": "The lower middle sample after sorting by corner RMS",
        "batch_wall_seconds": batch["wall_seconds"],
        "dtype_guard_calls": batch["dtype_guard_calls"],
        "tolerance_audit": audit,
        "scope": "Observed results for these 100 seeded initial shapes, not a universal bound or a measure of physical simulation accuracy",
    }
    metric_keys = list(complete[0]["metrics"])
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "seed",
                "status",
                "effective_scale",
                "backtracking_steps",
                "min_tet_volume_ratio",
                "min_sampled_jacobian",
                "wall_seconds",
                *metric_keys,
                "iterations_x",
                "iterations_y",
                "iterations_z",
                "stop_x",
                "stop_y",
                "stop_z",
            ],
        )
        writer.writeheader()
        for entry in samples:
            row = {
                "seed": entry["seed"],
                "status": entry["status"],
                "effective_scale": entry.get("effective_scale"),
                "backtracking_steps": entry.get("backtracking_steps"),
                "wall_seconds": entry["wall_seconds"],
                **entry.get("screen", {}),
                **entry.get("metrics", {}),
            }
            for axis, name in enumerate(("x", "y", "z")):
                if "solver" in entry:
                    row[f"iterations_{name}"] = entry["solver"][axis]["iterations"]
                    row[f"stop_{name}"] = entry["solver"][axis]["stop_code"]
            writer.writerow(row)
    _plots(output, complete)
    _selected_viewer(output, batch["settings"], selected)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "report-data.js").write_text(
        "window.BATCH_REPORT=" + json.dumps(report, separators=(",", ":"), allow_nan=False) + ";\n"
    )
    shutil.copytree(Path(__file__).with_name("batch_report_web"), output, dirs_exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    for name in ("round_trip_batch.py", "round_trip_batch_report.py"):
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    stats = summary["statistics"]
    rows = []
    for label, key, factor in (
        ("Corner RMS (mm)", "corner_rmse_m", 1000),
        ("Maximum corner error (mm)", "corner_max_error_m", 1000),
        ("Original displacement RMS (mm)", "original_displacement_rmse_m", 1000),
        ("Remaining row-space RMS (µm)", "rowspace_error_rmse_m", 1e6),
        ("True equation RMS", "equation_component_rms", 1),
        ("F component RMS", "gradient_component_rmse", 1),
    ):
        values = stats[key]
        cells = "".join(f"<td>{values[stat] * factor:.6g}</td>" for stat in ("mean", "median", "p95", "max"))
        rows.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")
    page = (output / "index.html").read_text().replace("__SUMMARY_ROWS__", "\n".join(rows))
    for token, value in {
        "__COUNT__": str(summary["requested_count"]),
        "__COMPLETE__": str(summary["complete_count"]),
        "__FAILURES__": str(summary["failure_count"]),
        "__MEAN__": f"{stats['corner_rmse_mm']['mean']:.6f}",
        "__MEDIAN__": f"{stats['corner_rmse_mm']['median']:.6f}",
        "__P95__": f"{stats['corner_rmse_mm']['p95']:.6f}",
        "__WORST__": f"{stats['corner_rmse_mm']['max']:.6f}",
        "__WORST_SEED__": str(stats["corner_rmse_mm"]["worst_seed"]),
        "__CORNER_MAX__": f"{stats['corner_max_error_mm']['max']:.6f}",
        "__MEAN_CELL_PCT__": f"{stats['corner_rmse_pct_cell_size']['mean']:.4f}",
        "__MEAN_LENGTH_PCT__": f"{stats['corner_rmse_pct_object_length']['mean']:.5f}",
    }.items():
        page = page.replace(token, value)
    (output / "index.html").write_text(page)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).parent / "generated" / "round_trip_100")
    args = parser.parse_args()
    report = build_batch_report(args.output)
    print(json.dumps(report["summary"], indent=2))
