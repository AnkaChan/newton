#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Analyze cloth Franka ABBA timings and Nsight Systems SQLite traces.

The utility never opens an input database for writing. It validates immutable
file evidence and benchmark provenance before calculating results. Raw Nsight
environment metadata is deliberately redacted because reports may contain
credentials and unrelated process state; only an explicit allowlist is shown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "cloth-franka-trace-analysis-v1"
SUITE_SCHEMA = "cloth-franka-abba-analysis-v1"
MANIFEST_SCHEMA = "cloth-franka-abba-suite-v1"
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".json", ".py", ".toml"}
PERFORMANCE_CONFIGURATION_FIELDS = {
    "collision_detection_block_size",
    "self_contact_force_block_dim",
    "self_contact_force_launch_override",
    "self_contact_force_max_blocks",
    "self_contact_force_max_blocks_resolved",
}

PREFIX_COMPONENTS = {
    "force": ("accumulate_self_contact_force_and_hessian_",),
    "vertex_triangle_traversal": ("vertex_triangle_collision_detection_kernel_",),
    "edge_edge_traversal": ("edge_colliding_edges_detection_kernel_",),
    "triangle_aabb": ("compute_tri_aabbs_",),
    "edge_aabb": ("compute_edge_aabbs_",),
    "bvh_group_roots": ("compute_bvh_group_roots_",),
    "planar_truncation": ("apply_planar_truncation_parallel_by_collision_",),
    "truncation_update": ("apply_truncation_ts_",),
}

NATIVE_BVH_REBUILD_NAMES = {
    "DeviceRadixSortExclusiveSumKernel",
    "DeviceRadixSortHistogramKernel",
    "DeviceRadixSortOnesweepKernel",
    "build_hierarchy",
    "build_leaves",
    "compute_key_deltas",
    "compute_morton_codes",
    "compute_total_bounds",
    "compute_total_inv_edges",
    "mark_packed_leaf_nodes",
    "memset_kernel",
}

EXACT_COMPONENTS = {"bvh_refit_kernel": "bvh_refit"}

DETECTOR_PARTS = (
    "vertex_triangle_traversal",
    "edge_edge_traversal",
    "triangle_aabb",
    "edge_aabb",
    "bvh_refit",
    "bvh_rebuild_native",
    "bvh_group_roots",
    "self_contact_buffer_fill",
)
LEGACY_CORE_PARTS = ("vertex_triangle_traversal", "edge_edge_traversal", "force")
FULL_PARTS = (*DETECTOR_PARTS, "force")
AGGREGATE_PARTS = {
    "detector": DETECTOR_PARTS,
    "legacy_traversal_plus_force": LEGACY_CORE_PARTS,
    "detector_plus_force": FULL_PARTS,
    "extended_pipeline": (*FULL_PARTS, "planar_truncation", "truncation_update"),
}

ENVIRONMENT_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "NEWTON_CACHE_PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "UV_CACHE_DIR",
    "UV_NO_SYNC",
    "WARP_CACHE_PATH",
)


@dataclass(frozen=True, slots=True)
class KernelEvent:
    """Store one CUDA kernel event from an Nsight SQLite export."""

    ordinal: int
    start: int
    end: int
    graph_node_id: int | None
    grid_x: int
    block_x: int
    device_id: int
    name: str

    @property
    def duration_ns(self) -> int:
        """Return the kernel duration in nanoseconds."""
        return self.end - self.start


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "n": len(values),
        "mean": mean,
        "median": median,
        "sample_stdev": std,
        "coefficient_of_variation": std / mean if mean else None,
        "minimum": min(values),
        "maximum": max(values),
        "median_absolute_deviation": mad,
    }


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


def _linear_drift_percent(runs: list[dict[str, Any]], field: str) -> float | None:
    if len(runs) < 2:
        return None
    x_values = [float(run["ordinal"]) for run in runs]
    y_values = [float(run[field]) for run in runs]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)) / denominator
    return 100.0 * slope / y_mean if y_mean else None


def _modified_z_outliers(runs: list[dict[str, Any]], field: str) -> list[int]:
    values = [float(run[field]) for run in runs]
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    if mad == 0.0:
        return []
    return [
        int(run["ordinal"])
        for run, value in zip(runs, values, strict=True)
        if abs(0.6745 * (value - median) / mad) > 3.5
    ]


def _sqlite_connection(path: Path) -> sqlite3.Connection:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"SQLite input not found: {path}")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _metadata_rows(connection: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    if table not in _table_names(connection):
        return []
    return [
        (str(name), "" if value is None else str(value))
        for name, value in connection.execute(f"SELECT name, value FROM {table}")
    ]


def _metadata_multimap(rows: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    result: defaultdict[str, list[str]] = defaultdict(list)
    for name, value in rows:
        result[name].append(value)
    return dict(result)


def _capture_provenance(connection: sqlite3.Connection) -> dict[str, Any]:
    capture = _metadata_multimap(_metadata_rows(connection, "META_DATA_CAPTURE"))
    export = _metadata_multimap(_metadata_rows(connection, "META_DATA_EXPORT"))
    environment_values = capture.get("PROCESS_0:ENVIRONMENT_VARIABLE", [])
    environment: dict[str, str] = {}
    for item in environment_values:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key in ENVIRONMENT_ALLOWLIST:
            environment[key] = value.strip('"')

    argument_pairs = []
    for key, values in capture.items():
        prefix = "PROCESS_0:ARGUMENT_"
        if key.startswith(prefix) and key[len(prefix) :].isdigit() and values:
            argument_pairs.append((int(key[len(prefix) :]), values[-1]))
    command = []
    if capture.get("PROCESS_0:COMMAND"):
        command.append(capture["PROCESS_0:COMMAND"][-1])
    command.extend(value for _, value in sorted(argument_pairs))

    session_rows = []
    tables = _table_names(connection)
    if "TARGET_INFO_SESSION_START_TIME" in tables:
        session_rows = connection.execute(
            "SELECT utcEpochNs, utcTime, localTime FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchall()

    gpu_rows = []
    if "TARGET_INFO_GPU" in tables:
        gpu_rows = connection.execute(
            "SELECT id, name, uuid, busLocation, chipName, totalMemory, computeMajor, computeMinor "
            "FROM TARGET_INFO_GPU ORDER BY id"
        ).fetchall()
    selected_ids = []
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible:
        for raw_token in visible.split(","):
            token = raw_token.strip()
            if token.isdigit():
                selected_ids.append(int(token))
    selected_gpus = [row for row in gpu_rows if not selected_ids or int(row[0]) in selected_ids]

    def last(multimap: dict[str, list[str]], key: str) -> str | None:
        values = multimap.get(key)
        return values[-1] if values else None

    return {
        "command": command,
        "working_directory": last(capture, "PROCESS_0:WORKING_DIR"),
        "environment_allowlist": environment,
        "raw_environment_record_count": len(environment_values),
        "raw_environment_redacted": True,
        "session_start": [{"utc_epoch_ns": row[0], "utc_time": row[1], "local_time": row[2]} for row in session_rows],
        "nsight_export": {
            "product": last(export, "EXPORT_PRODUCT_NAME"),
            "version": last(export, "EXPORT_PRODUCT_VERSION"),
            "schema_version": last(export, "EXPORT_SCHEMA_VERSION"),
            "export_time_utc": last(export, "EXPORT_TIME_UTC"),
        },
        "selected_target_gpus": [
            {
                "id": row[0],
                "name": row[1],
                "uuid": row[2],
                "bus_location": row[3],
                "chip_name": row[4],
                "total_memory_bytes": row[5],
                "compute_capability": f"{row[6]}.{row[7]}",
            }
            for row in selected_gpus
        ],
    }


def _read_events(connection: sqlite3.Connection) -> list[KernelEvent]:
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
    missing = required.difference(_table_names(connection))
    if missing:
        raise ValueError(f"Nsight SQLite export is missing tables: {sorted(missing)}")
    rows = connection.execute(
        """
        SELECT k.start, k.end, k.graphNodeId, k.gridX, k.blockX, k.deviceId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.shortName
        ORDER BY k.start, k.end, k.gridId
        """
    ).fetchall()
    events = [
        KernelEvent(
            ordinal=index,
            start=int(row[0]),
            end=int(row[1]),
            graph_node_id=None if row[2] is None else int(row[2]),
            grid_x=int(row[3]),
            block_x=int(row[4]),
            device_id=int(row[5]),
            name=str(row[6]),
        )
        for index, row in enumerate(rows)
    ]
    invalid = [event.ordinal for event in events if event.end < event.start]
    if invalid:
        raise ValueError(f"trace contains negative-duration kernels at ordinals {invalid[:10]}")
    return events


def _prefix_components(name: str) -> set[str]:
    result = set()
    for component, prefixes in PREFIX_COMPONENTS.items():
        if name.startswith(prefixes):
            result.add(component)
    exact = EXACT_COMPONENTS.get(name)
    if exact is not None:
        result.add(exact)
    return result


def _frame_slices(events: list[KernelEvent], expected_frames: int) -> tuple[int, list[list[KernelEvent]], list[str]]:
    graph_events = [event for event in events if event.graph_node_id is not None]
    triangle_events = [event for event in graph_events if event.name.startswith("compute_tri_aabbs_")]
    if not triangle_events:
        raise ValueError("trace contains no graph-captured triangle AABB kernels")
    marker_node = min(int(event.graph_node_id) for event in triangle_events if event.graph_node_id is not None)
    starts = [index for index, event in enumerate(graph_events) if event.graph_node_id == marker_node]
    if len(starts) != expected_frames:
        raise ValueError(f"trace contains {len(starts)} frame markers; expected {expected_frames}")
    warnings = []
    if starts[0] != 0:
        warnings.append(f"ignored {starts[0]} graph kernels before the first frame marker")
    frames = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(graph_events)
        frames.append(graph_events[start:stop])
    return marker_node, frames, warnings


def _structural_components(
    frames: list[list[KernelEvent]], *, expected_substeps: int
) -> tuple[dict[int, set[str]], list[str]]:
    components: defaultdict[int, set[str]] = defaultdict(set)
    warnings = []
    native_counts = []
    fill_counts = []
    for frame_index, frame in enumerate(frames, start=1):
        roots = [index for index, event in enumerate(frame) if event.name.startswith("compute_bvh_group_roots_")]
        if len(roots) != 2:
            raise ValueError(f"frame {frame_index} contains {len(roots)} BVH group-root kernels; expected 2")
        native_ordinals = set()
        for event_index, event in enumerate(frame):
            if event.name not in NATIVE_BVH_REBUILD_NAMES:
                continue
            if event_index > roots[1]:
                raise ValueError(
                    f"frame {frame_index} contains generic native BVH kernel {event.name!r} outside its rebuild window"
                )
            components[event.ordinal].add("bvh_rebuild_native")
            native_ordinals.add(event.ordinal)
        native_counts.append(len(native_ordinals))

        vertex_indices = [
            index
            for index, event in enumerate(frame)
            if event.name.startswith("vertex_triangle_collision_detection_kernel_")
        ]
        edge_indices = [
            index
            for index, event in enumerate(frame)
            if event.name.startswith("edge_colliding_edges_detection_kernel_")
        ]
        if len(vertex_indices) != expected_substeps or len(edge_indices) != expected_substeps:
            raise ValueError(
                f"frame {frame_index} traversal counts are VT={len(vertex_indices)}, EE={len(edge_indices)}; "
                f"expected {expected_substeps} each"
            )
        associated_edges = set()
        frame_fills = set()
        for vertex_index in vertex_indices:
            before = vertex_index - 1
            while before >= 0 and frame[before].name == "memtile_value_kernel":
                components[frame[before].ordinal].add("self_contact_buffer_fill")
                frame_fills.add(frame[before].ordinal)
                before -= 1
            after = vertex_index + 1
            while after < len(frame) and frame[after].name == "memtile_value_kernel":
                components[frame[after].ordinal].add("self_contact_buffer_fill")
                frame_fills.add(frame[after].ordinal)
                after += 1
            if after >= len(frame) or not frame[after].name.startswith("edge_colliding_edges_detection_kernel_"):
                raise ValueError(
                    f"frame {frame_index} VT traversal at graph node {frame[vertex_index].graph_node_id} "
                    "is not followed by its EE traversal"
                )
            associated_edges.add(after)
        if associated_edges != set(edge_indices):
            raise ValueError(f"frame {frame_index} does not pair every VT traversal with exactly one EE traversal")
        fill_counts.append(len(frame_fills))

    if len(set(native_counts)) != 1:
        raise ValueError(f"native BVH rebuild launch count varies by frame: {native_counts}")
    if len(set(fill_counts)) != 1:
        warnings.append(f"self-contact buffer-fill launch count varies by frame: {fill_counts}")
    return dict(components), warnings


def _expected_component_counts(frames: int, substeps: int, iterations: int, color_count: int) -> dict[str, int]:
    return {
        "vertex_triangle_traversal": frames * substeps,
        "edge_edge_traversal": frames * substeps,
        "triangle_aabb": frames * (substeps + 1),
        "edge_aabb": frames * (substeps + 1),
        "bvh_refit": frames * substeps * 2,
        "bvh_group_roots": frames * 2,
        "force": frames * substeps * iterations * color_count,
        "planar_truncation": frames * substeps * (iterations * color_count + 1),
        "truncation_update": frames * substeps * (iterations * color_count + 1),
    }


def summarize_trace(
    path: Path,
    *,
    expected_frames: int = 30,
    expected_substeps: int = 10,
    expected_iterations: int = 5,
    expected_color_count: int = 5,
) -> dict[str, Any]:
    """Read and summarize one Nsight Systems SQLite export."""
    path = path.resolve()
    with _sqlite_connection(path) as connection:
        events = _read_events(connection)
        provenance = _capture_provenance(connection)
    marker_node, frames, warnings = _frame_slices(events, expected_frames)
    structural, structural_warnings = _structural_components(frames, expected_substeps=expected_substeps)
    warnings.extend(structural_warnings)

    totals_ns: defaultdict[str, int] = defaultdict(int)
    counts: defaultdict[str, int] = defaultdict(int)
    selected_rows: defaultdict[tuple[str, int, int, tuple[str, ...]], list[int]] = defaultdict(lambda: [0, 0])
    selected_non_graph = []
    for event in events:
        components = _prefix_components(event.name)
        components.update(structural.get(event.ordinal, ()))
        for component in components:
            totals_ns[component] += event.duration_ns
            counts[component] += 1
        if components:
            key = (event.name, event.grid_x, event.block_x, tuple(sorted(components)))
            selected_rows[key][0] += 1
            selected_rows[key][1] += event.duration_ns
            if event.graph_node_id is None:
                selected_non_graph.append(event.ordinal)
    if selected_non_graph:
        raise ValueError(f"selected self-contact components contain non-graph launches: {selected_non_graph[:10]}")

    for aggregate, parts in AGGREGATE_PARTS.items():
        totals_ns[aggregate] = sum(totals_ns[part] for part in parts)

    expected_counts = _expected_component_counts(
        expected_frames, expected_substeps, expected_iterations, expected_color_count
    )
    mismatches = {
        component: {"actual": counts[component], "expected": expected}
        for component, expected in expected_counts.items()
        if counts[component] != expected
    }
    if mismatches:
        raise ValueError(f"fixed-workload component launch counts do not match: {mismatches}")

    frame_components: dict[str, list[int]] = {key: [] for key in (*AGGREGATE_PARTS, "all_graph_kernels")}
    per_frame_counts: dict[str, list[int]] = {
        component: [] for component in (*expected_counts, "bvh_rebuild_native", "self_contact_buffer_fill")
    }
    for frame in frames:
        frame_totals: defaultdict[str, int] = defaultdict(int)
        frame_counts: defaultdict[str, int] = defaultdict(int)
        for event in frame:
            frame_totals["all_graph_kernels"] += event.duration_ns
            event_components = _prefix_components(event.name)
            event_components.update(structural.get(event.ordinal, ()))
            for component in event_components:
                frame_totals[component] += event.duration_ns
                frame_counts[component] += 1
        for aggregate, parts in AGGREGATE_PARTS.items():
            frame_components[aggregate].append(sum(frame_totals[part] for part in parts))
        frame_components["all_graph_kernels"].append(frame_totals["all_graph_kernels"])
        for component, component_counts in per_frame_counts.items():
            component_counts.append(frame_counts[component])

    device_ids = sorted({event.device_id for event in events})
    if len(device_ids) != 1:
        warnings.append(f"kernel events use multiple CUDA device IDs: {device_ids}")
    return {
        "schema_version": TRACE_SCHEMA,
        "path": str(path),
        "file_size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "provenance": provenance,
        "kernel_device_ids": device_ids,
        "kernel_count": len(events),
        "kernel_time_ns": sum(event.duration_ns for event in events),
        "graph_kernel_count": sum(len(frame) for frame in frames),
        "graph_kernel_time_ns": sum(sum(event.duration_ns for event in frame) for frame in frames),
        "frame_marker_graph_node_id": marker_node,
        "frame_count": len(frames),
        "workload_expectations": {
            "frames": expected_frames,
            "substeps": expected_substeps,
            "iterations": expected_iterations,
            "color_count": expected_color_count,
            "component_counts": expected_counts,
        },
        "component_counts": dict(sorted(counts.items())),
        "component_totals_ns": dict(sorted(totals_ns.items())),
        "aggregate_definitions": {key: list(parts) for key, parts in AGGREGATE_PARTS.items()},
        "per_frame_component_ns": frame_components,
        "per_frame_component_counts": per_frame_counts,
        "selected_kernels": [
            {
                "name": name,
                "grid_x": grid_x,
                "block_x": block_x,
                "components": list(components),
                "count": values[0],
                "total_ns": values[1],
            }
            for (name, grid_x, block_x, components), values in sorted(selected_rows.items())
        ],
        "warnings": warnings,
    }


def compare_traces(traces: dict[str, dict[str, Any]], baseline_label: str) -> dict[str, Any]:
    if baseline_label not in traces:
        raise ValueError(f"baseline trace label {baseline_label!r} not found")
    baseline = traces[baseline_label]
    baseline_totals = baseline["component_totals_ns"]
    comparisons = {}
    for label, trace in traces.items():
        if label == baseline_label:
            continue
        rows = {}
        for component in sorted(set(baseline_totals).union(trace["component_totals_ns"])):
            baseline_ns = int(baseline_totals.get(component, 0))
            candidate_ns = int(trace["component_totals_ns"].get(component, 0))
            rows[component] = {
                "baseline_ns": baseline_ns,
                "candidate_ns": candidate_ns,
                "saved_ns": baseline_ns - candidate_ns,
                "speedup": baseline_ns / candidate_ns if candidate_ns else None,
            }
        comparisons[label] = {
            "baseline_label": baseline_label,
            "candidate_label": label,
            "all_kernel_speedup": baseline["kernel_time_ns"] / trace["kernel_time_ns"],
            "graph_kernel_speedup": baseline["graph_kernel_time_ns"] / trace["graph_kernel_time_ns"],
            "components": rows,
        }
    return comparisons


def _run_capture(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _source_fingerprint(root: Path) -> dict[str, Any]:
    root = root.resolve()
    package = root / "newton"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Newton package not found under {root}")
    files = [path for path in package.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        files.append(pyproject)
    digest = hashlib.sha256()
    byte_count = 0
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
        byte_count += len(contents)
    head = _run_capture(["git", "-C", str(root), "rev-parse", "HEAD"])
    tree = _run_capture(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"])
    status = _run_capture(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"])
    diff = _run_capture(["git", "-C", str(root), "diff", "--no-ext-diff", "--binary", "HEAD", "--"])
    if any(item.returncode != 0 for item in (head, tree, status, diff)):
        raise RuntimeError(f"failed to inspect Git state under {root}")
    return {
        "root": str(root),
        "source_hash": digest.hexdigest(),
        "source_file_count": len(files),
        "source_byte_count": byte_count,
        "git_head": head.stdout.strip(),
        "git_tree": tree.stdout.strip(),
        "git_dirty_tracked": bool(status.stdout.strip()),
        "git_status_tracked": status.stdout.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest(),
    }


def _resolve_manifest_path(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_dir / path).resolve()


def _validate_file_evidence(path: Path, evidence: dict[str, Any], *, field: str) -> str:
    if not path.is_file():
        raise ValueError(f"{field} file not found: {path}")
    actual = _sha256_file(path)
    expected = str(evidence.get("sha256", ""))
    if not expected:
        raise ValueError(f"{field} has no recorded SHA-256")
    if actual != expected:
        raise ValueError(f"{field} SHA-256 is {actual}, expected {expected}")
    if evidence.get("size") is not None and int(evidence["size"]) != path.stat().st_size:
        raise ValueError(f"{field} size is {path.stat().st_size}, expected {evidence['size']}")
    return actual


def _parse_benchmark_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("source", "device", "frames", "elapsed_seconds", "milliseconds_per_frame", "frames_per_second")
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"benchmark result {path} is missing fields: {missing}")
    frames = int(data["frames"])
    elapsed = float(data["elapsed_seconds"])
    milliseconds = float(data["milliseconds_per_frame"])
    fps = float(data["frames_per_second"])
    if frames <= 0 or not all(math.isfinite(value) and value > 0.0 for value in (elapsed, milliseconds, fps)):
        raise ValueError(f"benchmark result {path} contains invalid timing values")
    if not math.isclose(milliseconds, 1000.0 * elapsed / frames, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"benchmark result {path} has inconsistent milliseconds_per_frame")
    if not math.isclose(fps, frames / elapsed, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"benchmark result {path} has inconsistent frames_per_second")
    return data


def _validate_result_provenance(
    result: dict[str, Any], source: dict[str, Any], *, variant: str, strict: bool
) -> list[str]:
    warnings = []
    provenance = result["source"]
    expected_root = Path(source["root"]).resolve()
    imported = Path(provenance["newton_file"]).resolve()
    try:
        imported.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(
            f"{variant} result imported Newton from {imported}, expected it under {expected_root}"
        ) from error
    if provenance.get("git_head") != source.get("git_head"):
        raise ValueError(
            f"{variant} result Git HEAD {provenance.get('git_head')!r} does not match {source.get('git_head')!r}"
        )
    result_hash = provenance.get("source_hash")
    if result_hash is None:
        message = f"{variant} result has no source_hash; Git HEAD alone cannot identify dirty source bytes"
        if strict:
            raise ValueError(message)
        warnings.append(message)
    elif result_hash != source.get("source_hash"):
        raise ValueError(f"{variant} result source_hash does not match its manifest source")
    if provenance.get("git_dirty_tracked") is True:
        raise ValueError(f"{variant} result reports tracked source modifications")
    return warnings


def _validate_overflow(result: dict[str, Any], path: Path) -> None:
    contacts = result.get("contacts_after_final_frame") or {}
    for contact_type in ("vertex_triangle", "edge_edge"):
        value = contacts.get(contact_type) or {}
        if int(value.get("overflow_rows", 0)) != 0:
            raise ValueError(f"benchmark result {path} has {contact_type} contact-buffer overflow")


def _semantic_diagnostics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize nondeterministic state/contact outputs without rejecting valid atomic reordering."""
    state_hash_fields = ("particle_q_sha256", "particle_qd_sha256", "body_q_sha256", "body_qd_sha256")
    state_scalar_fields = ("particle_q_min", "particle_q_max", "particle_qd_abs_max")
    contact_fields = ("raw_count", "stored_count", "nonempty_rows", "overflow_rows")
    state_hashes = {}
    for field in state_hash_fields:
        values = [str((run.get("state") or {}).get(field)) for run in runs if (run.get("state") or {}).get(field)]
        state_hashes[field] = {"unique_count": len(set(values)), "values": sorted(set(values))}
    state_scalars = {}
    for field in state_scalar_fields:
        values = [float((run.get("state") or {})[field]) for run in runs if field in (run.get("state") or {})]
        if values:
            state_scalars[field] = _summary_stats(values)
    contacts = {}
    for contact_type in ("vertex_triangle", "edge_edge"):
        contacts[contact_type] = {}
        for field in contact_fields:
            values = [
                float(((run.get("contacts_after_final_frame") or {}).get(contact_type) or {})[field])
                for run in runs
                if field in ((run.get("contacts_after_final_frame") or {}).get(contact_type) or {})
            ]
            if values:
                contacts[contact_type][field] = _summary_stats(values)
    return {
        "interpretation": (
            "Particle trajectories and contact ordering may vary across fresh CUDA processes because the workload "
            "uses nondeterministic atomics. Hashes are diagnostics, not an equality criterion."
        ),
        "state_hashes": state_hashes,
        "state_scalars": state_scalars,
        "contacts": contacts,
    }


def _bootstrap_blocks(blocks: list[dict[str, Any]], *, samples: int, seed: int, confidence: float) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    log_ratios = [math.log(float(block["speedup_ratio"])) for block in blocks]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [log_ratios[rng.randrange(len(log_ratios))] for _ in log_ratios]
        estimates.append(math.exp(statistics.fmean(selected)))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "percentile bootstrap of balanced four-process blocks",
        "resampling_unit": "one complete ABBA or BAAB block",
        "samples": samples,
        "seed": seed,
        "confidence": confidence,
        "geometric_mean_block_speedup_ci": [
            _quantile(estimates, alpha),
            _quantile(estimates, 1.0 - alpha),
        ],
    }


def analyze_abba_manifest(
    manifest_path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
    strict_provenance: bool,
    verify_live_sources: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and analyze one completed cloth Franka ABBA manifest."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema is {manifest.get('schema_version')!r}; expected {MANIFEST_SCHEMA!r}")
    if manifest.get("status") != "completed":
        raise ValueError(f"manifest status is {manifest.get('status')!r}; expected 'completed'")
    manifest_dir = manifest_path.parent
    config = manifest["config"]
    block_count = int(config["abba_blocks"])
    expected_frames = int(config["frames"])
    sources = manifest["sources"]
    if set(sources) != {"baseline", "candidate"}:
        raise ValueError("manifest sources must contain exactly baseline and candidate")
    if sources["baseline"].get("git_head") != config.get("baseline_git_head"):
        raise ValueError("baseline source Git HEAD does not match config.baseline_git_head")
    if sources["baseline"].get("git_dirty_tracked"):
        raise ValueError("baseline source has tracked modifications")
    if sources["candidate"].get("git_dirty_tracked"):
        raise ValueError("candidate source has tracked modifications")

    validation = {"warnings": [], "immutable_files": [], "live_sources": {}}
    immutable = manifest.get("immutable_files") or []
    if strict_provenance and not immutable:
        raise ValueError("strict provenance requires immutable harness/analyzer file evidence")
    for index, evidence in enumerate(immutable):
        path = _resolve_manifest_path(manifest_dir, evidence["path"])
        actual = _validate_file_evidence(path, evidence, field=f"immutable_files[{index}]")
        validation["immutable_files"].append({"path": str(path), "sha256": actual})

    if verify_live_sources:
        for variant, expected in sources.items():
            actual = _source_fingerprint(Path(expected["root"]))
            for field in ("source_hash", "git_head", "git_tree", "git_diff_sha256"):
                if expected.get(field) != actual.get(field):
                    raise ValueError(f"live {variant} source {field} changed after suite planning")
            if actual["git_dirty_tracked"]:
                raise ValueError(f"live {variant} source has tracked modifications")
            validation["live_sources"][variant] = actual

    planned = sorted(manifest["runs"], key=lambda item: int(item["ordinal"]))
    if len(planned) != 4 * block_count:
        raise ValueError(f"manifest contains {len(planned)} runs; expected {4 * block_count}")
    if [int(run["ordinal"]) for run in planned] != list(range(1, len(planned) + 1)):
        raise ValueError("run ordinals are not contiguous and one-based")
    expected_patterns = {
        1: ["baseline", "candidate", "candidate", "baseline"],
        0: ["candidate", "baseline", "baseline", "candidate"],
    }
    for block in range(1, block_count + 1):
        members = [run for run in planned if int(run["block"]) == block]
        if [int(run["position"]) for run in members] != [1, 2, 3, 4]:
            raise ValueError(f"block {block} does not contain positions 1-4")
        expected = expected_patterns[block % 2]
        actual = [str(run["variant"]) for run in members]
        if actual != expected:
            raise ValueError(f"block {block} order is {actual}; expected {expected}")

    completed = []
    seen_result_hashes: dict[str, Path] = {}
    variant_configurations: dict[str, dict[str, Any]] = {}
    expected_physics_configuration = None
    expected_model = None
    for planned_run in planned:
        if planned_run.get("status") != "completed":
            raise ValueError(f"run {planned_run['ordinal']} status is not completed")
        variant = str(planned_run["variant"])
        path = _resolve_manifest_path(manifest_dir, planned_run["result_path"])
        actual_hash = _sha256_file(path)
        recorded_hash = planned_run.get("result_sha256")
        if strict_provenance and not recorded_hash:
            raise ValueError(f"run {planned_run['ordinal']} has no recorded result_sha256")
        if recorded_hash and recorded_hash != actual_hash:
            raise ValueError(f"run {planned_run['ordinal']} result SHA-256 changed")
        if actual_hash in seen_result_hashes:
            raise ValueError(
                f"run {planned_run['ordinal']} duplicates result bytes from {seen_result_hashes[actual_hash]}"
            )
        seen_result_hashes[actual_hash] = path
        result = _parse_benchmark_result(path)
        if int(result["frames"]) != expected_frames:
            raise ValueError(f"run {planned_run['ordinal']} has {result['frames']} frames; expected {expected_frames}")
        if "--cuda-profiler-api" in (planned_run.get("argv") or []):
            raise ValueError(f"run {planned_run['ordinal']} is an Nsight capture, not a valid end-to-end timing run")
        validation["warnings"].extend(
            _validate_result_provenance(result, sources[variant], variant=variant, strict=strict_provenance)
        )
        _validate_overflow(result, path)
        configuration = result.get("configuration")
        physics_configuration = {
            key: value for key, value in configuration.items() if key not in PERFORMANCE_CONFIGURATION_FIELDS
        }
        model = result.get("model")
        if variant not in variant_configurations:
            variant_configurations[variant] = configuration
        elif configuration != variant_configurations[variant]:
            raise ValueError(f"run {planned_run['ordinal']} configuration differs within {variant}")
        if expected_physics_configuration is None:
            expected_physics_configuration = physics_configuration
            expected_model = model
        if physics_configuration != expected_physics_configuration:
            raise ValueError(f"run {planned_run['ordinal']} physics configuration differs from the suite")
        if model != expected_model:
            raise ValueError(f"run {planned_run['ordinal']} model differs from the suite")
        completed.append(
            {
                **planned_run,
                "result_path": str(path),
                "result_sha256": actual_hash,
                "milliseconds_per_frame": float(result["milliseconds_per_frame"]),
                "frames_per_second": float(result["frames_per_second"]),
                "device": result["device"],
                "source": result["source"],
                "contacts_after_final_frame": result.get("contacts_after_final_frame"),
                "state": result.get("state"),
            }
        )
    devices = {run["device"] for run in completed}
    if len(devices) != 1:
        raise ValueError(f"ABBA runs used multiple devices: {sorted(devices)}")

    blocks = []
    nearest_pairs = []
    for block in range(1, block_count + 1):
        members = [run for run in completed if int(run["block"]) == block]
        baseline_values = [run["milliseconds_per_frame"] for run in members if run["variant"] == "baseline"]
        candidate_values = [run["milliseconds_per_frame"] for run in members if run["variant"] == "candidate"]
        baseline_geomean = math.exp(statistics.fmean(math.log(value) for value in baseline_values))
        candidate_geomean = math.exp(statistics.fmean(math.log(value) for value in candidate_values))
        blocks.append(
            {
                "block": block,
                "order": "ABBA" if block % 2 else "BAAB",
                "baseline_ordinals": [run["ordinal"] for run in members if run["variant"] == "baseline"],
                "candidate_ordinals": [run["ordinal"] for run in members if run["variant"] == "candidate"],
                "baseline_geometric_mean_ms_per_frame": baseline_geomean,
                "candidate_geometric_mean_ms_per_frame": candidate_geomean,
                "speedup_ratio": baseline_geomean / candidate_geomean,
            }
        )
        for pair_number, indexes in enumerate(((0, 1), (2, 3)), start=1):
            pair = [members[index] for index in indexes]
            baseline_run = next(run for run in pair if run["variant"] == "baseline")
            candidate_run = next(run for run in pair if run["variant"] == "candidate")
            nearest_pairs.append(
                {
                    "pair_id": f"block_{block:02d}_pair_{pair_number}",
                    "baseline_ordinal": baseline_run["ordinal"],
                    "candidate_ordinal": candidate_run["ordinal"],
                    "baseline_ms_per_frame": baseline_run["milliseconds_per_frame"],
                    "candidate_ms_per_frame": candidate_run["milliseconds_per_frame"],
                    "speedup_ratio": baseline_run["milliseconds_per_frame"] / candidate_run["milliseconds_per_frame"],
                }
            )

    baseline_runs = [run for run in completed if run["variant"] == "baseline"]
    candidate_runs = [run for run in completed if run["variant"] == "candidate"]
    baseline_values = [run["milliseconds_per_frame"] for run in baseline_runs]
    candidate_values = [run["milliseconds_per_frame"] for run in candidate_runs]
    block_ratios = [float(block["speedup_ratio"]) for block in blocks]
    primary = math.exp(statistics.fmean(math.log(value) for value in block_ratios))
    bootstrap = _bootstrap_blocks(blocks, samples=bootstrap_samples, seed=bootstrap_seed, confidence=confidence)
    summary = {
        "schema_version": SUITE_SCHEMA,
        "manifest": str(manifest_path),
        "suite_id": manifest.get("suite_id"),
        "primary_metric": "milliseconds_per_frame",
        "lower_is_better": True,
        "all_completed_processes_included": True,
        "outlier_policy": "No observations removed; modified-z flags are diagnostics only.",
        "config": config,
        "validation": validation,
        "sources": sources,
        "device": next(iter(devices)),
        "configuration": expected_physics_configuration,
        "variant_configurations": variant_configurations,
        "model": expected_model,
        "baseline": {
            **_summary_stats(baseline_values),
            "linear_drift_percent_per_suite_ordinal": _linear_drift_percent(baseline_runs, "milliseconds_per_frame"),
            "modified_z_outlier_ordinals": _modified_z_outliers(baseline_runs, "milliseconds_per_frame"),
            "semantic_diagnostics": _semantic_diagnostics(baseline_runs),
        },
        "candidate": {
            **_summary_stats(candidate_values),
            "linear_drift_percent_per_suite_ordinal": _linear_drift_percent(candidate_runs, "milliseconds_per_frame"),
            "modified_z_outlier_ordinals": _modified_z_outliers(candidate_runs, "milliseconds_per_frame"),
            "semantic_diagnostics": _semantic_diagnostics(candidate_runs),
        },
        "balanced_block_speedup": {
            "estimator": "geometric mean of complete balanced-block baseline/candidate geometric-mean ratios",
            "speedup_ratio": primary,
            "improvement_percent": 100.0 * (primary - 1.0),
            "median_block_speedup_ratio": statistics.median(block_ratios),
            "ratio_of_variant_medians": statistics.median(baseline_values) / statistics.median(candidate_values),
            "median_nearest_neighbor_speedup_ratio": statistics.median(pair["speedup_ratio"] for pair in nearest_pairs),
            "blocks_faster": sum(value > 1.0 for value in block_ratios),
            "block_count": len(blocks),
            "bootstrap": bootstrap,
        },
        "blocks": blocks,
        "nearest_neighbor_pairs": nearest_pairs,
        "runs": completed,
    }
    return summary, completed


def analyze_suite(
    manifest_path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
    strict_provenance: bool,
    verify_live_sources: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    abba, runs = analyze_abba_manifest(
        manifest_path,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
        strict_provenance=strict_provenance,
        verify_live_sources=verify_live_sources,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.resolve().parent
    trace_summaries = {}
    trace_entries = manifest.get("traces") or {}
    for label, entry in trace_entries.items():
        variant = str(entry.get("variant", label))
        if variant not in manifest["sources"]:
            raise ValueError(f"trace {label!r} has unknown variant {variant!r}")
        if entry.get("source_hash") != manifest["sources"][variant].get("source_hash"):
            raise ValueError(f"trace {label!r} source_hash does not match its {variant} source")
        path = _resolve_manifest_path(manifest_dir, entry["sqlite_path"])
        actual_hash = _validate_file_evidence(path, entry, field=f"traces.{label}")
        trace = summarize_trace(
            path,
            expected_frames=int(manifest["config"]["frames"]),
            expected_substeps=int(abba["configuration"]["sim_substeps"]),
            expected_iterations=int(abba["configuration"]["iterations"]),
            expected_color_count=len(abba["model"]["color_group_sizes"]),
        )
        if trace["sha256"] != actual_hash:
            raise AssertionError("trace SHA validation changed during read")
        command = trace["provenance"]["command"]
        if "--cuda-profiler-api" not in command:
            raise ValueError(f"trace {label!r} did not use the CUDA profiler capture API")
        trace_environment = trace["provenance"]["environment_allowlist"]
        python_path = trace_environment.get("PYTHONPATH")
        expected_root = Path(manifest["sources"][variant]["root"]).resolve()
        if python_path is None:
            if strict_provenance:
                raise ValueError(f"trace {label!r} metadata does not identify PYTHONPATH")
            trace["warnings"].append("trace metadata does not identify PYTHONPATH")
        else:
            imported_root = Path(python_path.split(os.pathsep, 1)[0]).resolve()
            if imported_root != expected_root:
                raise ValueError(f"trace {label!r} PYTHONPATH selects {imported_root}, expected {expected_root}")
        result_value = entry.get("result_path")
        result_hash = entry.get("result_sha256")
        if result_value is None or result_hash is None:
            if strict_provenance:
                raise ValueError(f"trace {label!r} lacks associated result JSON evidence")
            trace["warnings"].append("trace lacks associated result JSON evidence")
        else:
            result_path = _resolve_manifest_path(manifest_dir, result_value)
            _validate_file_evidence(
                result_path,
                {"sha256": result_hash},
                field=f"traces.{label}.result",
            )
            result = _parse_benchmark_result(result_path)
            trace["warnings"].extend(
                _validate_result_provenance(
                    result,
                    manifest["sources"][variant],
                    variant=variant,
                    strict=strict_provenance,
                )
            )
            if int(result["frames"]) != int(manifest["config"]["frames"]):
                raise ValueError(f"trace {label!r} result has the wrong frame count")
            trace["associated_result"] = {
                "path": str(result_path),
                "sha256": result_hash,
                "timing_valid_for_end_to_end": False,
            }
        trace_summaries[label] = trace
    result = {
        "schema_version": "cloth-franka-combined-analysis-v1",
        "abba": abba,
        "traces": trace_summaries,
        "trace_comparisons": compare_traces(trace_summaries, "baseline") if trace_summaries else {},
    }
    return result, runs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_trace_csv(path: Path, traces: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("trace", "component", "count", "total_ns", "total_ms")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, trace in traces.items():
            totals = trace["component_totals_ns"]
            counts = trace["component_counts"]
            for component in sorted(totals):
                total = int(totals[component])
                writer.writerow(
                    {
                        "trace": label,
                        "component": component,
                        "count": counts.get(component, ""),
                        "total_ns": total,
                        "total_ms": total / 1.0e6,
                    }
                )


def _write_runs_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "ordinal",
        "block",
        "position",
        "variant",
        "milliseconds_per_frame",
        "frames_per_second",
        "device",
        "result_sha256",
        "result_path",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def _trace_markdown(analysis: dict[str, Any]) -> str:
    traces = analysis["traces"]
    labels = list(traces)
    lines = [
        "# Cloth Franka self-contact trace analysis",
        "",
        "All durations are sums of CUDA kernel durations in the 30 captured frame graphs.",
        "",
        "| Component | " + " | ".join(f"{label} ms" for label in labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    components = sorted({component for trace in traces.values() for component in trace["component_totals_ns"]})
    for component in components:
        values = [traces[label]["component_totals_ns"].get(component, 0) / 1.0e6 for label in labels]
        lines.append(f"| {component} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    return "\n".join(lines) + "\n"


def _suite_markdown(analysis: dict[str, Any]) -> str:
    abba = analysis["abba"]
    paired = abba["balanced_block_speedup"]
    bootstrap = paired["bootstrap"]
    low, high = bootstrap["geometric_mean_block_speedup_ci"]
    lines = [
        "# Cloth Franka self-contact ABBA analysis",
        "",
        f"- Baseline median: {abba['baseline']['median']:.6f} ms/frame ({abba['baseline']['n']} processes)",
        f"- Candidate median: {abba['candidate']['median']:.6f} ms/frame ({abba['candidate']['n']} processes)",
        f"- Balanced-block speedup: {paired['speedup_ratio']:.5f}x ({paired['improvement_percent']:+.3f}%)",
        f"- {100.0 * bootstrap['confidence']:.1f}% block-bootstrap CI: [{low:.5f}x, {high:.5f}x]",
        f"- Faster blocks: {paired['blocks_faster']}/{paired['block_count']}",
        "- Outlier policy: all completed processes included; flags are diagnostics only.",
        "",
        "| Block | Order | Baseline ms/frame | Candidate ms/frame | Speedup |",
        "|---:|---|---:|---:|---:|",
    ]
    for block in abba["blocks"]:
        lines.append(
            f"| {block['block']} | {block['order']} | "
            f"{block['baseline_geometric_mean_ms_per_frame']:.6f} | "
            f"{block['candidate_geometric_mean_ms_per_frame']:.6f} | {block['speedup_ratio']:.5f}x |"
        )
    if analysis.get("traces"):
        lines.extend(("", _trace_markdown({"traces": analysis["traces"]}).strip()))
    return "\n".join(lines) + "\n"


def _parse_trace_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("trace must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("trace must be LABEL=PATH")
    return label, Path(path)


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="cloth_franka_analysis_") as temporary:
        root = Path(temporary)
        sqlite_path = root / "trace.sqlite"
        connection = sqlite3.connect(sqlite_path)
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, graphNodeId INTEGER, gridX INTEGER, blockX INTEGER,
                deviceId INTEGER, shortName INTEGER, gridId INTEGER
            );
            CREATE TABLE META_DATA_CAPTURE(name TEXT, value TEXT);
            CREATE TABLE META_DATA_EXPORT(name TEXT, value TEXT);
            """
        )
        names = [
            "compute_tri_aabbs_test_cuda_kernel_forward",
            "compute_edge_aabbs_test_cuda_kernel_forward",
            "compute_bvh_group_roots_test_cuda_kernel_forward",
            "compute_total_bounds",
            "bvh_refit_kernel",
            "memtile_value_kernel",
            "vertex_triangle_collision_detection_kernel_test_cuda_kernel_forward",
            "edge_colliding_edges_detection_kernel_test_cuda_kernel_forward",
            "accumulate_self_contact_force_and_hessian_test_cuda_kernel_forward",
            "apply_planar_truncation_parallel_by_collision_test_cuda_kernel_forward",
            "apply_truncation_ts_test_cuda_kernel_forward",
        ]
        name_ids = {name: index + 1 for index, name in enumerate(names)}
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", [(value, key) for key, value in name_ids.items()])
        rows = []
        start = 0
        node = 100
        for _frame in range(2):
            sequence = [
                names[0],
                "compute_total_bounds",
                names[2],
                names[1],
                "compute_total_bounds",
                names[2],
                names[0],
                "bvh_refit_kernel",
                names[1],
                "bvh_refit_kernel",
                "memtile_value_kernel",
                "memtile_value_kernel",
                names[6],
                "memtile_value_kernel",
                names[7],
                names[8],
                names[9],
                names[10],
                names[9],
                names[10],
            ]
            for name in sequence:
                grid = 1
                if name == "memtile_value_kernel":
                    grid = 50
                rows.append((start, start + 100, node, grid, 256, 0, name_ids[name], start + 1))
                start += 200
                node += 1
            node = 100
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
        connection.close()
        trace = summarize_trace(
            sqlite_path, expected_frames=2, expected_substeps=1, expected_iterations=1, expected_color_count=1
        )
        assert trace["component_counts"]["self_contact_buffer_fill"] == 6
        assert trace["component_counts"]["bvh_rebuild_native"] == 4
        assert trace["component_totals_ns"]["detector_plus_force"] == 3200

        evidence_path = root / "harness.py"
        evidence_path.write_text("# immutable\n", encoding="utf-8")
        sources = {}
        for variant in ("baseline", "candidate"):
            variant_root = root / variant
            (variant_root / "newton").mkdir(parents=True)
            (variant_root / "newton" / "__init__.py").write_text("", encoding="utf-8")
            sources[variant] = {
                "root": str(variant_root),
                "git_head": "a" * 40 if variant == "baseline" else "b" * 40,
                "git_tree": "c" * 40,
                "source_hash": f"{variant}-source",
                "git_dirty_tracked": False,
                "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
            }
        patterns = (
            ("baseline", "candidate", "candidate", "baseline"),
            ("candidate", "baseline", "baseline", "candidate"),
        )
        timings = (10.0, 9.0, 9.2, 10.1, 9.1, 10.2, 10.0, 9.0)
        runs = []
        ordinal = 0
        for block, pattern in enumerate(patterns, start=1):
            for position, variant in enumerate(pattern, start=1):
                ordinal += 1
                elapsed = timings[ordinal - 1] * 2 / 1000.0
                result = {
                    "run_id": ordinal,
                    "source": {
                        "newton_file": str(Path(sources[variant]["root"]) / "newton" / "__init__.py"),
                        "git_head": sources[variant]["git_head"],
                        "source_hash": sources[variant]["source_hash"],
                        "git_dirty_tracked": False,
                    },
                    "device": "cuda:0",
                    "frames": 2,
                    "elapsed_seconds": elapsed,
                    "milliseconds_per_frame": timings[ordinal - 1],
                    "frames_per_second": 2 / elapsed,
                    "configuration": {"sim_substeps": 1, "iterations": 1},
                    "model": {"color_group_sizes": [1]},
                    "contacts_after_final_frame": {
                        "vertex_triangle": {"overflow_rows": 0},
                        "edge_edge": {"overflow_rows": 0},
                    },
                    "state": {"hash": variant},
                }
                result_path = root / f"run_{ordinal:02d}.json"
                result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                runs.append(
                    {
                        "ordinal": ordinal,
                        "block": block,
                        "position": position,
                        "variant": variant,
                        "status": "completed",
                        "result_path": result_path.name,
                        "result_sha256": _sha256_file(result_path),
                        "argv": ["python", "harness.py"],
                    }
                )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "suite_id": "self-test",
            "status": "completed",
            "config": {"frames": 2, "abba_blocks": 2, "baseline_git_head": sources["baseline"]["git_head"]},
            "sources": sources,
            "immutable_files": [
                {
                    "path": evidence_path.name,
                    "size": evidence_path.stat().st_size,
                    "sha256": _sha256_file(evidence_path),
                }
            ],
            "runs": runs,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        analysis, parsed_runs = analyze_abba_manifest(
            manifest_path,
            bootstrap_samples=1000,
            bootstrap_seed=7,
            confidence=0.95,
            strict_provenance=True,
            verify_live_sources=False,
        )
        assert len(parsed_runs) == 8
        assert analysis["balanced_block_speedup"]["speedup_ratio"] > 1.0
        _write_runs_csv(root / "runs.csv", parsed_runs)
        assert (root / "runs.csv").read_text(encoding="utf-8").count("\n") == 9
    print("self-test passed: structural trace classification, alternating ABBA, provenance, block bootstrap")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="Analyze one or more LABEL=SQLITE traces")
    trace.add_argument("traces", nargs="+", type=_parse_trace_spec, metavar="LABEL=SQLITE")
    trace.add_argument("--baseline-label", default="baseline")
    trace.add_argument("--frames", type=int, default=30)
    trace.add_argument("--substeps", type=int, default=10)
    trace.add_argument("--iterations", type=int, default=5)
    trace.add_argument("--color-count", type=int, default=5)
    trace.add_argument("--output-json", type=Path)
    trace.add_argument("--output-markdown", type=Path)
    trace.add_argument("--output-csv", type=Path)

    suite = subparsers.add_parser("suite", help="Analyze a completed ABBA suite manifest")
    suite.add_argument("manifest", type=Path)
    suite.add_argument("--bootstrap-samples", type=int, default=100_000)
    suite.add_argument("--bootstrap-seed", type=int, default=20260817)
    suite.add_argument("--confidence", type=float, default=0.95)
    suite.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help="Warn instead of failing when old results lack source/file hashes",
    )
    suite.add_argument(
        "--verify-live-sources",
        action="store_true",
        help="Rehash both worktrees and verify their Git state before reporting",
    )
    suite.add_argument("--output-json", type=Path)
    suite.add_argument("--output-markdown", type=Path)
    suite.add_argument("--output-runs-csv", type=Path)
    suite.add_argument("--output-trace-csv", type=Path)

    subparsers.add_parser("self-test", help="Run synthetic read-only analysis tests")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "self-test":
        _self_test()
        return
    if args.command == "trace":
        labels = [label for label, _ in args.traces]
        if len(labels) != len(set(labels)):
            raise ValueError("trace labels must be unique")
        traces = {
            label: summarize_trace(
                path,
                expected_frames=args.frames,
                expected_substeps=args.substeps,
                expected_iterations=args.iterations,
                expected_color_count=args.color_count,
            )
            for label, path in args.traces
        }
        analysis = {
            "schema_version": "cloth-franka-trace-comparison-v1",
            "traces": traces,
            "comparisons": compare_traces(traces, args.baseline_label) if len(traces) > 1 else {},
        }
        if args.output_json:
            _write_json(args.output_json, analysis)
        if args.output_markdown:
            args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.output_markdown.write_text(_trace_markdown(analysis), encoding="utf-8")
        if args.output_csv:
            _write_trace_csv(args.output_csv, traces)
        print(json.dumps(analysis, indent=2))
        return

    analysis, runs = analyze_suite(
        args.manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
        strict_provenance=not args.allow_legacy_provenance,
        verify_live_sources=args.verify_live_sources,
    )
    if args.output_json:
        _write_json(args.output_json, analysis)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_suite_markdown(analysis), encoding="utf-8")
    if args.output_runs_csv:
        _write_runs_csv(args.output_runs_csv, runs)
    if args.output_trace_csv:
        _write_trace_csv(args.output_trace_csv, analysis["traces"])
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
