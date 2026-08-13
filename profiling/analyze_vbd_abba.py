#!/usr/bin/env python3
"""Analyze process-isolated VBD ABBA runtime benchmark results.

The primary estimator is the median of the candidate/baseline FPS ratios for
the nearest-neighbor pairs in each ABBA block. Confidence intervals resample
whole process pairs, never individual environment steps. No observation is
silently discarded; robust outlier diagnostics are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any

PRIMARY_METRIC = "environment_step_fps"


def _phase_from_legacy(data: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for phase in data:
        if phase.get("phase_name") == name:
            return phase
    raise ValueError(f"missing phase {name!r}")


def _legacy_named(entries: list[dict[str, Any]], suffix: str, field: str) -> Any:
    matches = [entry[field] for entry in entries if str(entry.get("name", "")).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one legacy field ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def parse_runtime_result(path: Path) -> dict[str, Any]:
    """Parse current schema, flat Omniperf, or legacy Omniperf JSON."""
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if isinstance(data, dict) and isinstance(data.get("run"), dict) and isinstance(data.get("runtime"), dict):
        run = data["run"]
        runtime = data["runtime"]
        timing = runtime.get("environment_step_timing") or {}
        fps_aggregate = timing.get("environment_step_fps") or runtime.get("collection_fps") or {}
        config = run.get("config") or {}
        return {
            "format": "schema",
            "task": run.get("task"),
            "seed": run.get("seed"),
            "num_envs": run.get("num_envs"),
            "num_steps": runtime.get("iterations_completed"),
            "warmup_steps": timing.get("warmup_steps"),
            "measurement_mode": timing.get("measurement_mode"),
            "presets": list(config.get("presets") or []),
            "reported_physics_backend": config.get("physics_backend"),
            PRIMARY_METRIC: float(fps_aggregate["mean"]),
            "environment_step_fps_std": _optional_float(fps_aggregate.get("std")),
            "iteration_time_ms": 1000.0 * float(runtime["iteration_time_s"]["mean"]),
            "total_wall_time_s": float(runtime["total_wall_time_s"]),
            "status": run.get("status"),
            "hardware": data.get("hardware"),
            "versions": data.get("versions"),
        }

    if isinstance(data, dict) and isinstance(data.get("benchmark_info"), dict):
        info = data["benchmark_info"]
        runtime = data["runtime"]
        fps_key = _flat_fps_key(runtime)
        return {
            "format": "omniperf-flat",
            "task": info.get("task"),
            "seed": None,
            "num_envs": info.get("num_envs"),
            "num_steps": info.get("num_steps"),
            "warmup_steps": info.get("environment_step_warmup_steps"),
            "measurement_mode": info.get("environment_step_measurement_mode"),
            "presets": _split_presets(info.get("presets")),
            "reported_physics_backend": None,
            PRIMARY_METRIC: float(runtime[fps_key]),
            "environment_step_fps_std": _optional_float(runtime.get(fps_key.replace("Mean ", "Std "))),
            "iteration_time_ms": float(runtime["Mean Iteration Time"]),
            "total_wall_time_s": float(runtime["Total Wall Time"]),
            "status": "completed",
            "hardware": data.get("hardware_info"),
            "versions": data.get("version_info"),
        }

    if isinstance(data, list):
        info_phase = _phase_from_legacy(data, "benchmark_info")
        runtime_phase = _phase_from_legacy(data, "runtime")
        info = info_phase.get("metadata") or []
        measurements = runtime_phase.get("measurements") or []
        fps_suffix = _legacy_fps_suffix(measurements)
        return {
            "format": "omniperf-legacy",
            "task": _legacy_named(info, " benchmark_info task", "data"),
            "seed": None,
            "num_envs": _legacy_named(info, " benchmark_info num_envs", "data"),
            "num_steps": _legacy_named(info, " benchmark_info num_steps", "data"),
            "warmup_steps": _legacy_named(info, " benchmark_info environment_step_warmup_steps", "data"),
            "measurement_mode": _legacy_named(info, " benchmark_info environment_step_measurement_mode", "data"),
            "presets": _split_presets(_legacy_named(info, " benchmark_info presets", "data")),
            "reported_physics_backend": None,
            PRIMARY_METRIC: float(_legacy_named(measurements, fps_suffix, "value")),
            "environment_step_fps_std": _optional_float(
                _legacy_named(measurements, fps_suffix.replace(" Mean ", " Std "), "value")
            ),
            "iteration_time_ms": float(_legacy_named(measurements, " runtime Mean Iteration Time", "value")),
            "total_wall_time_s": float(_legacy_named(measurements, " runtime Total Wall Time", "value")),
            "status": "completed",
            "hardware": _optional_legacy_phase(data, "hardware_info"),
            "versions": _optional_legacy_phase(data, "version_info"),
        }

    raise ValueError(f"unsupported benchmark JSON structure in {path}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _split_presets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _flat_fps_key(runtime: dict[str, Any]) -> str:
    preferred = (
        "Mean Environment Step Host-Return FPS",
        "Mean Environment Step Serialized-Synchronized FPS",
        "Mean Collection FPS",
    )
    for key in preferred:
        if key in runtime:
            return key
    raise ValueError("runtime result has no supported mean FPS metric")


def _legacy_fps_suffix(measurements: list[dict[str, Any]]) -> str:
    preferred = (
        " runtime Mean Environment Step Host-Return FPS",
        " runtime Mean Environment Step Serialized-Synchronized FPS",
        " runtime Mean Collection FPS",
    )
    names = [str(item.get("name", "")) for item in measurements]
    for suffix in preferred:
        if sum(name.endswith(suffix) for name in names) == 1:
            return suffix
    raise ValueError("legacy runtime result has no supported mean FPS metric")


def _optional_legacy_phase(data: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    try:
        phase = _phase_from_legacy(data, name)
    except ValueError:
        return None
    return {entry["name"]: entry.get("data") for entry in phase.get("metadata") or []}


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _paired_bootstrap(pairs: list[dict[str, Any]], *, samples: int, seed: int, confidence: float) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(seed)
    ratios = [float(pair["speedup_ratio"]) for pair in pairs]
    ratio_of_medians_samples: list[float] = []
    median_pair_ratio_samples: list[float] = []
    for _ in range(samples):
        selection = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        baselines = [float(pair["baseline_fps"]) for pair in selection]
        candidates = [float(pair["candidate_fps"]) for pair in selection]
        sampled_ratios = [float(pair["speedup_ratio"]) for pair in selection]
        ratio_of_medians_samples.append(statistics.median(candidates) / statistics.median(baselines))
        median_pair_ratio_samples.append(statistics.median(sampled_ratios))
    ratio_of_medians_samples.sort()
    median_pair_ratio_samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "percentile paired process bootstrap",
        "resampling_unit": "nearest-neighbor baseline/candidate process pair",
        "samples": samples,
        "seed": seed,
        "confidence": confidence,
        "median_pair_ratio_ci": [
            _quantile(median_pair_ratio_samples, alpha),
            _quantile(median_pair_ratio_samples, 1.0 - alpha),
        ],
        "ratio_of_medians_ci": [
            _quantile(ratio_of_medians_samples, alpha),
            _quantile(ratio_of_medians_samples, 1.0 - alpha),
        ],
        "observed_pair_ratios": ratios,
    }


def _summary_stats(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "n": len(values),
        "median": median,
        "mean": mean,
        "sample_std": std,
        "coefficient_of_variation": std / mean if mean else None,
        "minimum": min(values),
        "maximum": max(values),
        "median_absolute_deviation": mad,
    }


def _outlier_ordinals(runs: list[dict[str, Any]]) -> list[int]:
    values = [float(run[PRIMARY_METRIC]) for run in runs]
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    if mad == 0.0:
        return []
    return [
        int(run["ordinal"])
        for run, value in zip(runs, values, strict=True)
        if abs(0.6745 * (value - median) / mad) > 3.5
    ]


def _linear_drift_percent(runs: list[dict[str, Any]]) -> float | None:
    if len(runs) < 2:
        return None
    x_values = [float(run["ordinal"]) for run in runs]
    y_values = [float(run[PRIMARY_METRIC]) for run in runs]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)) / denominator
    return 100.0 * slope / y_mean if y_mean else None


def _validate_result(result: dict[str, Any], config: dict[str, Any]) -> None:
    expected = {
        "task": config["task"],
        "num_envs": config["num_envs"],
        "num_steps": config["num_steps"],
        "warmup_steps": config["warmup_steps"],
        "measurement_mode": "host_return",
    }
    for key, expected_value in expected.items():
        actual = result.get(key)
        if actual is not None and actual != expected_value:
            raise ValueError(f"result {key}={actual!r}, expected {expected_value!r}")
    if result.get("seed") is not None and result["seed"] != config["seed"]:
        raise ValueError(f"result seed={result['seed']!r}, expected {config['seed']!r}")
    if result.get("status") not in (None, "completed"):
        raise ValueError(f"result status is {result['status']!r}")
    expected_presets = set(config["expected_reported_presets"])
    missing = expected_presets.difference(result.get("presets") or [])
    if missing:
        raise ValueError(f"result is missing preset/override metadata: {sorted(missing)}")
    fps = float(result[PRIMARY_METRIC])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid FPS value {fps!r}")


def _validate_abba_schedule(completed: list[dict[str, Any]], config: dict[str, Any], sources: dict[str, Any]) -> None:
    ordered = sorted(completed, key=lambda run: int(run["ordinal"]))
    if [int(run["ordinal"]) for run in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("completed-run ordinals are not contiguous")
    for block in range(1, int(config["abba_blocks"]) + 1):
        members = [run for run in ordered if int(run["block"]) == block]
        if [int(run["position"]) for run in members] != [1, 2, 3, 4]:
            raise ValueError(f"block {block} does not contain positions 1-4")
        if [run["variant"] for run in members] != ["baseline", "candidate", "candidate", "baseline"]:
            raise ValueError(f"block {block} is not ordered ABBA")
        expected_pairs = [f"block_{block:02d}_pair_1"] * 2 + [f"block_{block:02d}_pair_2"] * 2
        if [run["pair_id"] for run in members] != expected_pairs:
            raise ValueError(f"block {block} does not use nearest-neighbor ABBA pairs")
    for run in ordered:
        expected_hash = sources[run["variant"]]["source_hash"]
        if run.get("source_hash") != expected_hash:
            raise ValueError(f"run {run['ordinal']} source hash does not match its {run['variant']} source")


def analyze_manifest(
    manifest_path: Path, *, bootstrap_samples: int, bootstrap_seed: int, confidence: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    config = manifest["config"]
    completed: list[dict[str, Any]] = []
    manifest_dir = manifest_path.parent
    for planned in manifest["runs"]:
        if planned.get("status") != "completed":
            continue
        result_path = Path(planned["result_path"])
        if not result_path.is_absolute():
            result_path = manifest_dir / result_path
        result = parse_runtime_result(result_path)
        _validate_result(result, config)
        completed.append({**planned, **result, "result_path": str(result_path.resolve())})

    expected_count = 4 * int(config["abba_blocks"])
    if len(completed) != expected_count:
        raise ValueError(f"manifest has {len(completed)} completed runs; expected {expected_count}")
    _validate_abba_schedule(completed, config, manifest["sources"])

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for run in completed:
        by_pair.setdefault(run["pair_id"], []).append(run)
    pairs: list[dict[str, Any]] = []
    for pair_id in sorted(by_pair):
        members = by_pair[pair_id]
        baselines = [run for run in members if run["variant"] == "baseline"]
        candidates = [run for run in members if run["variant"] == "candidate"]
        if len(baselines) != 1 or len(candidates) != 1:
            raise ValueError(f"pair {pair_id!r} does not contain exactly one process per variant")
        baseline = baselines[0]
        candidate = candidates[0]
        speedup = float(candidate[PRIMARY_METRIC]) / float(baseline[PRIMARY_METRIC])
        pairs.append(
            {
                "pair_id": pair_id,
                "baseline_ordinal": baseline["ordinal"],
                "candidate_ordinal": candidate["ordinal"],
                "baseline_fps": baseline[PRIMARY_METRIC],
                "candidate_fps": candidate[PRIMARY_METRIC],
                "speedup_ratio": speedup,
                "improvement_percent": 100.0 * (speedup - 1.0),
            }
        )

    baseline_runs = sorted((run for run in completed if run["variant"] == "baseline"), key=lambda item: item["ordinal"])
    candidate_runs = sorted(
        (run for run in completed if run["variant"] == "candidate"), key=lambda item: item["ordinal"]
    )
    baseline_values = [float(run[PRIMARY_METRIC]) for run in baseline_runs]
    candidate_values = [float(run[PRIMARY_METRIC]) for run in candidate_runs]
    pair_ratios = [float(pair["speedup_ratio"]) for pair in pairs]
    bootstrap = _paired_bootstrap(pairs, samples=bootstrap_samples, seed=bootstrap_seed, confidence=confidence)
    median_pair_ratio = statistics.median(pair_ratios)
    ratio_of_medians = statistics.median(candidate_values) / statistics.median(baseline_values)
    summary = {
        "schema_version": "vbd-abba-analysis-v1",
        "manifest": str(manifest_path.resolve()),
        "suite_id": manifest.get("suite_id"),
        "task": config["task"],
        "config": config,
        "provenance": {
            "host": manifest.get("host"),
            "isaaclab_source": manifest.get("isaaclab_source"),
            "task_config_evidence": manifest.get("task_config_evidence"),
            "frozen_volume_evidence": manifest.get("frozen_volume_evidence"),
            "additional_immutable_evidence": manifest.get("additional_immutable_evidence"),
        },
        "primary_metric": PRIMARY_METRIC,
        "higher_is_better": True,
        "all_completed_processes_included": True,
        "baseline": {
            **_summary_stats(baseline_values),
            "source": manifest["sources"]["baseline"],
            "outlier_ordinals_modified_z_gt_3_5": _outlier_ordinals(baseline_runs),
            "linear_drift_percent_per_suite_ordinal": _linear_drift_percent(baseline_runs),
        },
        "candidate": {
            **_summary_stats(candidate_values),
            "source": manifest["sources"]["candidate"],
            "outlier_ordinals_modified_z_gt_3_5": _outlier_ordinals(candidate_runs),
            "linear_drift_percent_per_suite_ordinal": _linear_drift_percent(candidate_runs),
        },
        "paired_speedup": {
            "pairing": "nearest chronological neighbors within each ABBA block",
            "pair_count": len(pairs),
            "median_pair_ratio": median_pair_ratio,
            "median_pair_improvement_percent": 100.0 * (median_pair_ratio - 1.0),
            "ratio_of_variant_medians": ratio_of_medians,
            "ratio_of_variant_medians_improvement_percent": 100.0 * (ratio_of_medians - 1.0),
            "bootstrap": bootstrap,
        },
        "pairs": pairs,
        "runs": completed,
    }
    return summary, completed


def _write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "ordinal",
        "block",
        "position",
        "pair_id",
        "variant",
        "environment_step_fps",
        "environment_step_fps_std",
        "iteration_time_ms",
        "total_wall_time_s",
        "source_hash",
        "result_path",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    paired = summary["paired_speedup"]
    bootstrap = paired["bootstrap"]
    low, high = bootstrap["median_pair_ratio_ci"]
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    lines = [
        f"# VBD ABBA benchmark: {summary['task']}",
        "",
        f"- Baseline median: {baseline['median']:,.3f} FPS ({baseline['n']} fresh processes)",
        f"- Candidate median: {candidate['median']:,.3f} FPS ({candidate['n']} fresh processes)",
        f"- Paired median speedup: {paired['median_pair_ratio']:.4f}x "
        f"({paired['median_pair_improvement_percent']:+.2f}%)",
        f"- {100.0 * bootstrap['confidence']:.1f}% paired process-bootstrap CI: [{low:.4f}x, {high:.4f}x]",
        f"- Ratio of variant medians: {paired['ratio_of_variant_medians']:.4f}x",
        "- Outlier policy: all completed processes included; modified-z flags are diagnostics only.",
        "",
        "| Pair | Baseline FPS | Candidate FPS | Speedup |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {pair['pair_id']} | {pair['baseline_fps']:,.3f} | {pair['candidate_fps']:,.3f} | "
        f"{pair['speedup_ratio']:.4f}x |"
        for pair in summary["pairs"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _self_test() -> None:
    root = Path(__file__).resolve().parent
    fixtures = [
        root
        / "cloth_10it_1substep"
        / "benchmark_runtime_Isaac-Lift-Cloth-Franka_2026-08-12_09-40-07-499676_581efc48.json",
        root
        / "optimization"
        / "shape_grouped"
        / "cloth_rep01"
        / "benchmark_runtime_Isaac-Lift-Cloth-Franka_2026-08-12_11-58-21-806775_8ed9b914_schema.json",
        root
        / "optimization"
        / "shape_grouped"
        / "cloth_rep01"
        / "benchmark_runtime_Isaac-Lift-Cloth-Franka_2026-08-12_11-58-21-806775_8ed9b914_omniperf.json",
    ]
    parsed = [parse_runtime_result(path) for path in fixtures]
    assert parsed[0]["format"] == "omniperf-legacy"
    assert parsed[1]["format"] == "schema"
    assert parsed[2]["format"] == "omniperf-flat"
    assert parsed[0]["num_envs"] == parsed[1]["num_envs"] == parsed[2]["num_envs"] == 1024
    assert parsed[1][PRIMARY_METRIC] == parsed[2][PRIMARY_METRIC]
    pairs = [
        {"baseline_fps": 10.0, "candidate_fps": 15.0, "speedup_ratio": 1.5},
        {"baseline_fps": 20.0, "candidate_fps": 30.0, "speedup_ratio": 1.5},
    ]
    bootstrap = _paired_bootstrap(pairs, samples=100, seed=7, confidence=0.95)
    assert bootstrap["median_pair_ratio_ci"] == [1.5, 1.5]
    with tempfile.TemporaryDirectory(prefix="vbd_abba_test_") as temporary:
        temporary_path = Path(temporary)
        manifest_path = temporary_path / "manifest.json"
        source_bundle = json.loads(fixtures[1].read_text(encoding="utf-8"))
        result_paths = []
        for ordinal, fps in enumerate((100.0, 120.0, 110.0, 100.0), start=1):
            bundle = json.loads(json.dumps(source_bundle))
            bundle["runtime"]["environment_step_timing"]["environment_step_fps"]["mean"] = fps
            bundle["runtime"]["collection_fps"]["mean"] = fps
            result_path = temporary_path / f"result_{ordinal}.json"
            result_path.write_text(json.dumps(bundle), encoding="utf-8")
            result_paths.append(result_path)
        manifest = {
            "suite_id": "self-test",
            "config": {
                "task": parsed[1]["task"],
                "num_envs": parsed[1]["num_envs"],
                "num_steps": parsed[1]["num_steps"],
                "warmup_steps": parsed[1]["warmup_steps"],
                "seed": parsed[1]["seed"],
                "abba_blocks": 1,
                "expected_reported_presets": parsed[1]["presets"],
            },
            "sources": {
                "baseline": {"root": "A", "source_hash": "baseline"},
                "candidate": {"root": "B", "source_hash": "candidate"},
            },
            "runs": [
                {
                    "ordinal": ordinal,
                    "block": 1,
                    "position": ordinal,
                    "pair_id": f"block_01_pair_{pair}",
                    "variant": variant,
                    "status": "completed",
                    "source_hash": variant,
                    "result_path": str(result_paths[ordinal - 1]),
                }
                for ordinal, pair, variant in (
                    (1, 1, "baseline"),
                    (2, 1, "candidate"),
                    (3, 2, "candidate"),
                    (4, 2, "baseline"),
                )
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        summary, runs = analyze_manifest(manifest_path, bootstrap_samples=100, bootstrap_seed=9, confidence=0.95)
        assert len(runs) == 4
        assert summary["paired_speedup"]["median_pair_ratio"] == 1.15
        assert summary["paired_speedup"]["ratio_of_variant_medians"] == 1.15
        assert summary["paired_speedup"]["bootstrap"]["median_pair_ratio_ci"] == [1.1, 1.2]
        _write_csv(temporary_path / "runs.csv", runs)
        _write_markdown(temporary_path / "analysis.md", summary)
        assert (temporary_path / "runs.csv").read_text(encoding="utf-8").count("\n") == 5
        assert "1.1500x" in (temporary_path / "analysis.md").read_text(encoding="utf-8")
    print("self-test passed: three result formats, ABBA manifest, paired bootstrap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, help="ABBA suite manifest.json")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.manifest is None:
        parser.error("manifest is required unless --self-test is used")
    summary, runs = analyze_manifest(
        args.manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    output_json = args.output_json or args.manifest.with_name("analysis.json")
    output_csv = args.output_csv or args.manifest.with_name("runs.csv")
    output_markdown = args.output_markdown or args.manifest.with_name("analysis.md")
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_csv, runs)
    _write_markdown(output_markdown, summary)
    print(json.dumps(summary["paired_speedup"], indent=2))
    print(f"wrote {output_json}, {output_csv}, and {output_markdown}")


if __name__ == "__main__":
    main()
