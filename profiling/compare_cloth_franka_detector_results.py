#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen cloth Franka detector benchmark results exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
DETECTOR_KINDS = ("vertex_triangle", "edge_edge")
SHA256_LENGTH = 64
WORKLOAD_SCALAR_KEYS = {
    "self_contact_margin",
    "rest_shape_exclusion_radius",
    "edge_edge_parallel_epsilon",
    "record_triangle_contacting_vertices",
    "world_count",
    "world_copies",
    "particle_count",
    "triangle_count",
    "edge_count",
}
WORKLOAD_ARRAY_KEYS = {
    "vertex_positions",
    "rest_positions",
    "tri_indices",
    "edge_indices",
    "particle_world",
    "triangle_bvh_lower_bounds",
    "triangle_bvh_upper_bounds",
    "triangle_bvh_groups",
    "triangle_bvh_group_roots",
    "edge_bvh_lower_bounds",
    "edge_bvh_upper_bounds",
    "edge_bvh_groups",
    "edge_bvh_group_roots",
    "vertex_colliding_triangles_offsets",
    "vertex_colliding_triangles_buffer_sizes",
    "triangle_colliding_vertices_offsets",
    "triangle_colliding_vertices_buffer_sizes",
    "vertex_triangle_filtering_list",
    "vertex_triangle_filtering_list_offsets",
    "edge_colliding_edges_offsets",
    "edge_colliding_edges_buffer_sizes",
    "edge_filtering_list",
    "edge_filtering_list_offsets",
}


class ResultContractError(ValueError):
    """Report a result that cannot support an authoritative comparison."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha(value: Any, lengths: tuple[int, ...] = (SHA256_LENGTH,)) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultContractError(message)


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    _require(isinstance(value, list), f"{path} must be an array")
    return value


def _require_key(mapping: dict[str, Any], key: str, path: str) -> Any:
    _require(key in mapping, f"{path} is missing required field {key!r}")
    return mapping[key]


def _require_bool(mapping: dict[str, Any], key: str, path: str, expected: bool | None = None) -> bool:
    value = _require_key(mapping, key, path)
    _require(type(value) is bool, f"{path}.{key} must be a Boolean")
    if expected is not None:
        _require(value is expected, f"{path}.{key} must be {expected}")
    return value


def _require_int(mapping: dict[str, Any], key: str, path: str, minimum: int | None = None) -> int:
    value = _require_key(mapping, key, path)
    _require(type(value) is int, f"{path}.{key} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{path}.{key} must be at least {minimum}")
    return value


def _require_number(mapping: dict[str, Any], key: str, path: str, positive: bool = False) -> float:
    value = _require_key(mapping, key, path)
    _require(type(value) in (int, float), f"{path}.{key} must be numeric")
    numeric = float(value)
    _require(math.isfinite(numeric), f"{path}.{key} must be finite")
    if positive:
        _require(numeric > 0.0, f"{path}.{key} must be positive")
    return numeric


def _require_sha(mapping: dict[str, Any], key: str, path: str) -> str:
    value = _require_key(mapping, key, path)
    _require(_is_sha(value), f"{path}.{key} must be a hexadecimal SHA-256")
    return value.lower()


def _status_is_clean(value: Any) -> bool:
    return value == "" or value == []


def _validate_code_provenance(run: dict[str, Any], label: str) -> None:
    newton = _require_mapping(_require_key(run, "newton", label), f"{label}.newton")
    for key in ("root", "newton_file", "version"):
        _require(
            isinstance(_require_key(newton, key, f"{label}.newton"), str), f"{label}.newton.{key} must be a string"
        )
    newton_head = _require_key(newton, "git_head", f"{label}.newton")
    newton_tree = _require_key(newton, "git_tree", f"{label}.newton")
    _require(_is_sha(newton_head, (40, 64)), f"{label}.newton.git_head must be a full Git object ID")
    _require(_is_sha(newton_tree, (40, 64)), f"{label}.newton.git_tree must be a full Git object ID")
    _require_bool(newton, "git_dirty_tracked", f"{label}.newton", expected=False)
    newton_status = _require_key(newton, "git_status_tracked", f"{label}.newton")
    _require(_status_is_clean(newton_status), f"{label}.newton.git_status_tracked must be empty")
    _require(
        _require_key(run, "git_head", label) == newton_head,
        f"{label}.git_head disagrees with {label}.newton.git_head",
    )
    _require(
        _require_key(run, "source_root", label) == newton["root"],
        f"{label}.source_root disagrees with {label}.newton.root",
    )

    warp = _require_mapping(_require_key(run, "warp", label), f"{label}.warp")
    for key in ("root", "warp_file", "version"):
        _require(isinstance(_require_key(warp, key, f"{label}.warp"), str), f"{label}.warp.{key} must be a string")
    warp_head = _require_key(warp, "git_head", f"{label}.warp")
    warp_tree = _require_key(warp, "git_tree", f"{label}.warp")
    if warp_head is not None:
        _require(_is_sha(warp_head, (40, 64)), f"{label}.warp.git_head must be null or a full Git object ID")
    if warp_tree is not None:
        _require(_is_sha(warp_tree, (40, 64)), f"{label}.warp.git_tree must be null or a full Git object ID")
    warp_dirty = _require_key(warp, "git_dirty_tracked", f"{label}.warp")
    _require(warp_dirty in (None, False), f"{label}.warp.git_dirty_tracked must be false or null")
    warp_status = _require_key(warp, "git_status_tracked", f"{label}.warp")
    if warp_dirty is False:
        _require(_status_is_clean(warp_status), f"{label}.warp.git_status_tracked must be empty")
        _require(warp_head is not None and warp_tree is not None, f"{label}.warp clean Git provenance is incomplete")
    else:
        _require(
            warp_head is None and warp_tree is None and warp_status is None,
            f"{label}.warp non-Git provenance is inconsistent",
        )

    libraries = _require_mapping(
        _require_key(warp, "native_libraries", f"{label}.warp"), f"{label}.warp.native_libraries"
    )
    for library_name in ("core", "clang"):
        library = _require_mapping(
            _require_key(libraries, library_name, f"{label}.warp.native_libraries"),
            f"{label}.warp.native_libraries.{library_name}",
        )
        _require(
            isinstance(_require_key(library, "path", f"{label}.warp.native_libraries.{library_name}"), str),
            f"{label}.warp.native_libraries.{library_name}.path must be a string",
        )
        _require_int(library, "size", f"{label}.warp.native_libraries.{library_name}", minimum=1)
        _require_sha(library, "sha256", f"{label}.warp.native_libraries.{library_name}")


def _validate_workload(run: dict[str, Any], label: str) -> None:
    state = _require_mapping(_require_key(run, "state", label), f"{label}.state")
    state_kind = _require_key(state, "kind", f"{label}.state")
    _require(state_kind in ("advanced_example", "npz"), f"{label}.state.kind is unsupported")
    if state_kind == "advanced_example":
        _require_int(state, "frames", f"{label}.state", minimum=0)
    else:
        _require(isinstance(_require_key(state, "path", f"{label}.state"), str), f"{label}.state.path must be a string")
        _require_sha(state, "sha256", f"{label}.state")

    _require_sha(run, "frozen_particle_q_sha256", label)
    _require_sha(run, "workload_particle_q_sha256", label)
    replication = _require_mapping(_require_key(run, "replication", label), f"{label}.replication")
    world_copies = _require_int(replication, "world_copies", f"{label}.replication", minimum=1)
    _require_bool(replication, "particle_world_layout_exact", f"{label}.replication", expected=True)
    _require_bool(replication, "rest_positions_tiled_exactly", f"{label}.replication", expected=True)

    for key in ("particle_count", "triangle_count", "edge_count"):
        _require_int(run, key, label, minimum=1)
    _require_int(run, "sm_count", label, minimum=1)
    _require(isinstance(_require_key(run, "device", label), str), f"{label}.device must be a string")

    workload = _require_mapping(_require_key(run, "workload", label), f"{label}.workload")
    workload_sha = _require_sha(workload, "sha256", f"{label}.workload")
    fingerprint_payload = {
        key: _require_key(workload, key, f"{label}.workload") for key in ("scalars", "arrays", "bvh_constructors")
    }
    scalars = _require_mapping(fingerprint_payload["scalars"], f"{label}.workload.scalars")
    arrays = _require_mapping(fingerprint_payload["arrays"], f"{label}.workload.arrays")
    constructors = _require_mapping(fingerprint_payload["bvh_constructors"], f"{label}.workload.bvh_constructors")
    _require(set(scalars) == WORKLOAD_SCALAR_KEYS, f"{label}.workload.scalars has an unexpected schema")
    _require(set(arrays) == WORKLOAD_ARRAY_KEYS, f"{label}.workload.arrays has an unexpected schema")
    _require(set(constructors) == {"triangles", "edges"}, f"{label}.workload.bvh_constructors is invalid")
    for key in ("self_contact_margin", "rest_shape_exclusion_radius", "edge_edge_parallel_epsilon"):
        _require_number(scalars, key, f"{label}.workload.scalars")
    _require_bool(scalars, "record_triangle_contacting_vertices", f"{label}.workload.scalars")
    for key in ("world_count", "world_copies", "particle_count", "triangle_count", "edge_count"):
        _require_int(scalars, key, f"{label}.workload.scalars", minimum=1)
    for name, constructor_value in constructors.items():
        constructor = _require_mapping(constructor_value, f"{label}.workload.bvh_constructors.{name}")
        _require(
            set(constructor) == {"constructor", "leaf_size"},
            f"{label}.workload.bvh_constructors.{name} is invalid",
        )
        _require_int(constructor, "constructor", f"{label}.workload.bvh_constructors.{name}", minimum=0)
        _require_int(constructor, "leaf_size", f"{label}.workload.bvh_constructors.{name}", minimum=1)
    for array_name, evidence_value in arrays.items():
        if evidence_value is None:
            continue
        evidence = _require_mapping(evidence_value, f"{label}.workload.arrays.{array_name}")
        shape = _require_list(
            _require_key(evidence, "shape", f"{label}.workload.arrays.{array_name}"),
            f"{label}.workload.arrays.{array_name}.shape",
        )
        _require(
            all(type(dimension) is int and dimension >= 0 for dimension in shape),
            f"{label}.workload.arrays.{array_name}.shape is invalid",
        )
        _require(
            isinstance(_require_key(evidence, "dtype", f"{label}.workload.arrays.{array_name}"), str),
            f"{label}.workload.arrays.{array_name}.dtype must be a string",
        )
        _require_sha(evidence, "sha256", f"{label}.workload.arrays.{array_name}")
    _require(
        workload_sha == _canonical_sha256(fingerprint_payload),
        f"{label}.workload.sha256 does not match its canonical fingerprint payload",
    )
    scalar_world_copies = scalars["world_copies"]
    _require(scalar_world_copies == world_copies, f"{label}.workload world copy counts disagree")
    _require(scalars["world_count"] == world_copies, f"{label}.workload world_count disagrees with replication")
    for scalar_key, result_key in (
        ("particle_count", "particle_count"),
        ("triangle_count", "triangle_count"),
        ("edge_count", "edge_count"),
    ):
        _require(scalars[scalar_key] == run[result_key], f"{label}.workload {scalar_key} disagrees with result")
    vertex_positions = _require_mapping(arrays["vertex_positions"], f"{label}.workload.arrays.vertex_positions")
    _require(
        vertex_positions["sha256"] == run["workload_particle_q_sha256"],
        f"{label}.workload vertex position hash disagrees with workload_particle_q_sha256",
    )


def _validate_world_checks(checks: Any, world_copies: int, path: str) -> None:
    checks = _require_mapping(checks, path)
    shapes = _require_mapping(_require_key(checks, "shapes", path), f"{path}.shapes")
    _require(shapes, f"{path}.shapes must not be empty")
    for shape_name in shapes:
        _require_bool(shapes, shape_name, f"{path}.shapes", expected=True)
    for key in ("counts", "active_pairs", "owner_min_distances", "targets_stay_in_world", "all"):
        _require_bool(checks, key, path, expected=True)
    for key in ("raw_counts_per_world", "active_pair_counts_per_world"):
        values = _require_list(_require_key(checks, key, path), f"{path}.{key}")
        _require(len(values) == world_copies, f"{path}.{key} must have one entry per world")
        _require(
            all(type(value) is int and value >= 0 for value in values),
            f"{path}.{key} must contain non-negative integers",
        )


def _validate_outputs(outputs: Any, path: str) -> None:
    outputs = _require_mapping(outputs, path)
    for key in ("raw_count", "stored_count", "nonempty_rows", "overflow_rows"):
        _require_int(outputs, key, path, minimum=0)
    for key in (
        "counts_sha256",
        "active_pairs_sha256",
        "owner_min_distances_sha256",
        "resize_flags_sha256",
    ):
        _require_sha(outputs, key, path)
    if "other_min_distances_sha256" in outputs:
        _require_sha(outputs, "other_min_distances_sha256", path)
    flags = _require_list(_require_key(outputs, "resize_flags", path), f"{path}.resize_flags")
    _require(all(type(flag) is int for flag in flags), f"{path}.resize_flags must contain integers")


def _validate_reference_equality(equality: Any, path: str) -> None:
    equality = _require_mapping(equality, path)
    for key in ("counts", "active_pairs", "owner_min_distances", "other_min_distances", "resize_flags", "all"):
        _require_bool(equality, key, path, expected=True)


def _validate_timing(kind_result: dict[str, Any], repeats: int, samples: int, path: str) -> None:
    elapsed = _require_list(_require_key(kind_result, "sample_elapsed_ms", path), f"{path}.sample_elapsed_ms")
    per_launch = _require_list(
        _require_key(kind_result, "sample_microseconds_per_launch", path),
        f"{path}.sample_microseconds_per_launch",
    )
    _require(len(elapsed) == samples and len(per_launch) == samples, f"{path} timing sample counts are invalid")
    for sample, (elapsed_value, launch_value) in enumerate(zip(elapsed, per_launch, strict=True)):
        _require(
            type(elapsed_value) in (int, float) and math.isfinite(elapsed_value) and elapsed_value > 0,
            f"{path}.sample_elapsed_ms[{sample}] must be positive and finite",
        )
        _require(
            type(launch_value) in (int, float) and math.isfinite(launch_value) and launch_value > 0,
            f"{path}.sample_microseconds_per_launch[{sample}] must be positive and finite",
        )
        expected = 1000.0 * float(elapsed_value) / repeats
        _require(
            math.isclose(float(launch_value), expected, rel_tol=1.0e-12, abs_tol=1.0e-12),
            f"{path} per-launch timing disagrees with elapsed time",
        )
    median = _require_number(kind_result, "median_microseconds_per_launch", path, positive=True)
    minimum = _require_number(kind_result, "minimum_microseconds_per_launch", path, positive=True)
    _require(
        math.isclose(median, statistics.median(per_launch), rel_tol=1.0e-12, abs_tol=1.0e-12),
        f"{path} median timing is invalid",
    )
    _require(
        math.isclose(minimum, min(per_launch), rel_tol=1.0e-12, abs_tol=1.0e-12), f"{path} minimum timing is invalid"
    )


def _validate_launch_geometry(
    geometry: Any,
    logical_threads: int,
    block_size: int,
    sm_count: int,
    path: str,
) -> None:
    geometry = _require_mapping(geometry, path)
    blocks = math.ceil(logical_threads / block_size)
    warps_per_block = math.ceil(block_size / 32)
    expected = {
        "logical_threads": logical_threads,
        "blocks": blocks,
        "warps_per_block": warps_per_block,
        "physical_warps": blocks * warps_per_block,
    }
    for key, expected_value in expected.items():
        _require_int(geometry, key, path, minimum=1)
        _require(geometry[key] == expected_value, f"{path}.{key} is inconsistent with block size {block_size}")
    expected_warps_per_sm = expected["physical_warps"] / sm_count
    expected_lane_fraction = block_size / (32 * warps_per_block)
    _require(
        math.isclose(
            _require_number(geometry, "physical_warps_per_sm", path),
            expected_warps_per_sm,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        f"{path}.physical_warps_per_sm is invalid",
    )
    _require(
        math.isclose(
            _require_number(geometry, "full_block_active_lane_fraction", path),
            expected_lane_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        f"{path}.full_block_active_lane_fraction is invalid",
    )


def _validate_schedule(run: dict[str, Any], label: str, block_sizes: list[int]) -> None:
    samples = run["samples"]
    schedule = _require_list(_require_key(run, "timing_schedule", label), f"{label}.timing_schedule")
    expected = Counter(
        (sample, block_size, kind) for sample in range(samples) for block_size in block_sizes for kind in DETECTOR_KINDS
    )
    actual: Counter[tuple[int, int, str]] = Counter()
    for ordinal, entry_value in enumerate(schedule):
        entry = _require_mapping(entry_value, f"{label}.timing_schedule[{ordinal}]")
        sample = _require_int(entry, "sample", f"{label}.timing_schedule[{ordinal}]", minimum=0)
        block_size = _require_int(entry, "block_size", f"{label}.timing_schedule[{ordinal}]", minimum=1)
        kind = _require_key(entry, "kind", f"{label}.timing_schedule[{ordinal}]")
        _require(kind in DETECTOR_KINDS, f"{label}.timing_schedule[{ordinal}].kind is invalid")
        actual[(sample, block_size, kind)] += 1
    _require(actual == expected, f"{label}.timing_schedule does not cover every sample/block/kind exactly once")


def validate_result(run: Any, label: str) -> dict[int, dict[str, Any]]:
    """Validate one benchmark result and index its block records."""
    run = _require_mapping(run, label)
    _require_sha(run, "harness_sha256", label)
    _require(isinstance(_require_key(run, "harness", label), str), f"{label}.harness must be a string")
    _require_list(_require_key(run, "argv", label), f"{label}.argv")
    _validate_code_provenance(run, label)
    _validate_workload(run, label)

    for key, minimum in (("warmups", 0), ("repeats", 1), ("samples", 5)):
        _require_int(run, key, label, minimum=minimum)
    _require_bool(run, "all_outputs_exactly_equal", label, expected=True)
    _require_bool(run, "all_world_copies_exact", label, expected=True)

    block_sizes = _require_list(_require_key(run, "block_sizes", label), f"{label}.block_sizes")
    _require(block_sizes, f"{label}.block_sizes must not be empty")
    _require(
        all(type(size) is int and size > 0 for size in block_sizes),
        f"{label}.block_sizes must contain positive integers",
    )
    _require(len(block_sizes) == len(set(block_sizes)), f"{label}.block_sizes must not contain duplicates")
    reference_size = _require_int(run, "reference_block_size", label, minimum=1)
    _require(reference_size in block_sizes, f"{label}.reference_block_size is not in block_sizes")

    result_records = _require_list(_require_key(run, "results", label), f"{label}.results")
    _require(len(result_records) == len(block_sizes), f"{label}.results length disagrees with block_sizes")
    by_block: dict[int, dict[str, Any]] = {}
    world_copies = run["replication"]["world_copies"]
    logical_threads = {
        "vertex_triangle": run["particle_count"],
        "edge_edge": run["edge_count"],
    }
    for ordinal, record_value in enumerate(result_records):
        path = f"{label}.results[{ordinal}]"
        record = _require_mapping(record_value, path)
        block_size = _require_int(record, "block_size", path, minimum=1)
        _require(block_size not in by_block, f"{label}.results repeats block size {block_size}")
        by_block[block_size] = record
        for kind in DETECTOR_KINDS:
            kind_path = f"{path}.{kind}"
            kind_result = _require_mapping(_require_key(record, kind, path), kind_path)
            _validate_timing(kind_result, run["repeats"], run["samples"], kind_path)
            _validate_launch_geometry(
                _require_key(kind_result, "launch_geometry", kind_path),
                logical_threads[kind],
                block_size,
                run["sm_count"],
                f"{kind_path}.launch_geometry",
            )
            _validate_outputs(_require_key(kind_result, "outputs", kind_path), f"{kind_path}.outputs")
            _validate_reference_equality(
                _require_key(kind_result, "exactly_equal_to_reference", kind_path),
                f"{kind_path}.exactly_equal_to_reference",
            )
            _validate_world_checks(
                _require_key(kind_result, "world_copies_exact", kind_path),
                world_copies,
                f"{kind_path}.world_copies_exact",
            )
    _require(list(by_block) == block_sizes, f"{label}.results order disagrees with block_sizes")
    _validate_schedule(run, label, block_sizes)
    return by_block


def _identity_payloads(run: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    newton = run["newton"]
    newton_identity = {
        key: newton[key] for key in ("version", "git_head", "git_tree", "git_dirty_tracked", "git_status_tracked")
    }
    warp = run["warp"]
    warp_identity = {
        key: warp[key] for key in ("version", "git_head", "git_tree", "git_dirty_tracked", "git_status_tracked")
    }
    warp_identity["native_libraries"] = {
        name: {key: library[key] for key in ("size", "sha256")} for name, library in warp["native_libraries"].items()
    }
    return newton_identity, warp_identity


def _normalized_state(run: dict[str, Any]) -> dict[str, Any]:
    state = dict(run["state"])
    state.pop("path", None)
    return state


def _comparison_check(
    checks: list[dict[str, Any]],
    name: str,
    baseline: Any,
    candidate: Any,
    *,
    include_values: bool = False,
) -> bool:
    exact = _canonical_json(baseline) == _canonical_json(candidate)
    check: dict[str, Any] = {"name": name, "exact": exact}
    if include_values or not exact:
        check["baseline"] = baseline
        check["candidate"] = candidate
    checks.append(check)
    return exact


def _different_paths(baseline: Any, candidate: Any, path: str = "") -> list[str]:
    if type(baseline) is not type(candidate):
        return [path or "$"]
    if isinstance(baseline, dict):
        differences: list[str] = []
        for key in sorted(set(baseline) | set(candidate)):
            child = f"{path}.{key}" if path else key
            if key not in baseline or key not in candidate:
                differences.append(child)
            else:
                differences.extend(_different_paths(baseline[key], candidate[key], child))
        return differences
    if isinstance(baseline, list):
        differences = []
        if len(baseline) != len(candidate):
            differences.append(f"{path}.length")
        for index, (left, right) in enumerate(zip(baseline, candidate, strict=False)):
            differences.extend(_different_paths(left, right, f"{path}[{index}]"))
        return differences
    return [] if baseline == candidate else [path or "$"]


def _timing_record(kind_result: dict[str, Any]) -> dict[str, float]:
    return {
        "median_microseconds_per_launch": float(kind_result["median_microseconds_per_launch"]),
        "minimum_microseconds_per_launch": float(kind_result["minimum_microseconds_per_launch"]),
    }


def _ratio(baseline: float, candidate: float) -> float:
    return baseline / candidate


def _optima(by_block: dict[int, dict[str, Any]], block_sizes: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in DETECTOR_KINDS:
        block = min(block_sizes, key=lambda size: by_block[size][kind]["median_microseconds_per_launch"])
        result[kind] = {
            "block_size": block,
            **_timing_record(by_block[block][kind]),
        }
    combined_by_block = {
        block: sum(float(by_block[block][kind]["median_microseconds_per_launch"]) for kind in DETECTOR_KINDS)
        for block in block_sizes
    }
    combined_block = min(block_sizes, key=combined_by_block.__getitem__)
    result["combined"] = {
        "block_size": combined_block,
        "median_microseconds_per_vt_plus_ee": combined_by_block[combined_block],
    }
    return result


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_baseline_newton_head: str | None = None,
    expected_candidate_newton_head: str | None = None,
    expected_baseline_warp_identity: str | None = None,
    expected_candidate_warp_identity: str | None = None,
) -> dict[str, Any]:
    """Compare two validated benchmark results and return a JSON-ready report."""
    expected_baseline_newton_head = (
        None if expected_baseline_newton_head is None else expected_baseline_newton_head.lower()
    )
    expected_candidate_newton_head = (
        None if expected_candidate_newton_head is None else expected_candidate_newton_head.lower()
    )
    expected_baseline_warp_identity = (
        None if expected_baseline_warp_identity is None else expected_baseline_warp_identity.lower()
    )
    expected_candidate_warp_identity = (
        None if expected_candidate_warp_identity is None else expected_candidate_warp_identity.lower()
    )
    baseline_blocks = validate_result(baseline, "baseline")
    candidate_blocks = validate_result(candidate, "candidate")
    baseline_newton, baseline_warp = _identity_payloads(baseline)
    candidate_newton, candidate_warp = _identity_payloads(candidate)
    baseline_warp_id = _canonical_sha256(baseline_warp)
    candidate_warp_id = _canonical_sha256(candidate_warp)

    errors: list[str] = []
    provenance_checks: list[dict[str, Any]] = []
    newton_same = _canonical_json(baseline_newton) == _canonical_json(candidate_newton)
    if newton_same:
        if expected_baseline_newton_head is not None and baseline_newton["git_head"] != expected_baseline_newton_head:
            errors.append("baseline Newton HEAD does not match --expect-baseline-newton-head")
        if (
            expected_candidate_newton_head is not None
            and candidate_newton["git_head"] != expected_candidate_newton_head
        ):
            errors.append("candidate Newton HEAD does not match --expect-candidate-newton-head")
    else:
        if baseline_newton["git_head"] == candidate_newton["git_head"]:
            errors.append("Newton provenance differs despite identical Git HEADs")
        elif expected_baseline_newton_head is None or expected_candidate_newton_head is None:
            errors.append("Newton provenance differs; both per-side expected Newton HEADs are required")
        elif (
            baseline_newton["git_head"] != expected_baseline_newton_head
            or candidate_newton["git_head"] != expected_candidate_newton_head
        ):
            errors.append("Newton provenance differs from the declared per-side expected HEADs")
    newton_difference_declared = newton_same or (
        expected_baseline_newton_head == baseline_newton["git_head"]
        and expected_candidate_newton_head == candidate_newton["git_head"]
    )
    provenance_checks.append(
        {
            "name": "newton",
            "exact": newton_same,
            "difference_declared": newton_difference_declared,
            "baseline": {**baseline_newton, "root": baseline["newton"]["root"]},
            "candidate": {**candidate_newton, "root": candidate["newton"]["root"]},
        }
    )

    warp_same = baseline_warp_id == candidate_warp_id
    if expected_baseline_warp_identity is not None and baseline_warp_id != expected_baseline_warp_identity:
        errors.append("baseline Warp identity does not match --expect-baseline-warp-identity")
    if expected_candidate_warp_identity is not None and candidate_warp_id != expected_candidate_warp_identity:
        errors.append("candidate Warp identity does not match --expect-candidate-warp-identity")
    if not warp_same and (expected_baseline_warp_identity is None or expected_candidate_warp_identity is None):
        errors.append("Warp provenance differs; both per-side expected Warp identity SHA-256 values are required")
    warp_difference_declared = warp_same or (
        expected_baseline_warp_identity == baseline_warp_id and expected_candidate_warp_identity == candidate_warp_id
    )
    provenance_checks.append(
        {
            "name": "warp",
            "exact": warp_same,
            "difference_declared": warp_difference_declared,
            "baseline": {"identity_sha256": baseline_warp_id, **baseline_warp, "root": baseline["warp"]["root"]},
            "candidate": {"identity_sha256": candidate_warp_id, **candidate_warp, "root": candidate["warp"]["root"]},
        }
    )

    compatibility_checks: list[dict[str, Any]] = []
    comparable_pairs = (
        ("harness_sha256", baseline["harness_sha256"], candidate["harness_sha256"]),
        ("state", _normalized_state(baseline), _normalized_state(candidate)),
        ("frozen_particle_q_sha256", baseline["frozen_particle_q_sha256"], candidate["frozen_particle_q_sha256"]),
        ("workload_particle_q_sha256", baseline["workload_particle_q_sha256"], candidate["workload_particle_q_sha256"]),
        ("workload", baseline["workload"], candidate["workload"]),
        ("replication", baseline["replication"], candidate["replication"]),
        ("particle_count", baseline["particle_count"], candidate["particle_count"]),
        ("triangle_count", baseline["triangle_count"], candidate["triangle_count"]),
        ("edge_count", baseline["edge_count"], candidate["edge_count"]),
        ("device", baseline["device"], candidate["device"]),
        ("sm_count", baseline["sm_count"], candidate["sm_count"]),
        ("warmups", baseline["warmups"], candidate["warmups"]),
        ("repeats", baseline["repeats"], candidate["repeats"]),
        ("samples", baseline["samples"], candidate["samples"]),
    )
    for name, left, right in comparable_pairs:
        if not _comparison_check(compatibility_checks, name, left, right):
            errors.append(f"incompatible {name} provenance")

    baseline_sizes = baseline["block_sizes"]
    candidate_sizes = candidate["block_sizes"]
    common_sizes = sorted(set(baseline_sizes) & set(candidate_sizes))
    if not common_sizes:
        errors.append("the results have no common block size")
    block_summary = {
        "baseline": baseline_sizes,
        "candidate": candidate_sizes,
        "common": common_sizes,
        "baseline_only": sorted(set(baseline_sizes) - set(candidate_sizes)),
        "candidate_only": sorted(set(candidate_sizes) - set(baseline_sizes)),
    }

    output_blocks: list[dict[str, Any]] = []
    timing_blocks: list[dict[str, Any]] = []
    all_outputs_exact = True
    for block_size in common_sizes:
        output_record: dict[str, Any] = {"block_size": block_size}
        timing_record: dict[str, Any] = {"block_size": block_size}
        combined_baseline = 0.0
        combined_candidate = 0.0
        for kind in DETECTOR_KINDS:
            baseline_kind = baseline_blocks[block_size][kind]
            candidate_kind = candidate_blocks[block_size][kind]
            baseline_output = {
                "outputs": baseline_kind["outputs"],
                "world_copies_exact": baseline_kind["world_copies_exact"],
            }
            candidate_output = {
                "outputs": candidate_kind["outputs"],
                "world_copies_exact": candidate_kind["world_copies_exact"],
            }
            differences = _different_paths(baseline_output, candidate_output)
            exact = not differences
            all_outputs_exact = all_outputs_exact and exact
            if not exact:
                errors.append(f"block {block_size} {kind} final outputs differ")
            output_record[kind] = {"exact": exact, "different_fields": differences}
            if not exact:
                output_record[kind]["baseline"] = baseline_output
                output_record[kind]["candidate"] = candidate_output

            baseline_timing = _timing_record(baseline_kind)
            candidate_timing = _timing_record(candidate_kind)
            combined_baseline += baseline_timing["median_microseconds_per_launch"]
            combined_candidate += candidate_timing["median_microseconds_per_launch"]
            timing_record[kind] = {
                "baseline": baseline_timing,
                "candidate": candidate_timing,
                "baseline_over_candidate_median_speedup": _ratio(
                    baseline_timing["median_microseconds_per_launch"],
                    candidate_timing["median_microseconds_per_launch"],
                ),
                "baseline_over_candidate_minimum_speedup": _ratio(
                    baseline_timing["minimum_microseconds_per_launch"],
                    candidate_timing["minimum_microseconds_per_launch"],
                ),
            }
        timing_record["combined"] = {
            "baseline_median_microseconds_per_vt_plus_ee": combined_baseline,
            "candidate_median_microseconds_per_vt_plus_ee": combined_candidate,
            "baseline_over_candidate_median_speedup": _ratio(combined_baseline, combined_candidate),
        }
        output_blocks.append(output_record)
        timing_blocks.append(timing_record)

    timing: dict[str, Any] = {
        "ratio_definition": "baseline time / candidate time; values above 1 mean candidate is faster",
        "per_common_block": timing_blocks,
    }
    if common_sizes:
        baseline_common_optima = _optima(baseline_blocks, common_sizes)
        candidate_common_optima = _optima(candidate_blocks, common_sizes)
        timing["common_block_optima"] = {
            "baseline": baseline_common_optima,
            "candidate": candidate_common_optima,
            "speedups_at_independent_optima": {
                kind: _ratio(
                    baseline_common_optima[kind][
                        "median_microseconds_per_vt_plus_ee" if kind == "combined" else "median_microseconds_per_launch"
                    ],
                    candidate_common_optima[kind][
                        "median_microseconds_per_vt_plus_ee" if kind == "combined" else "median_microseconds_per_launch"
                    ],
                )
                for kind in (*DETECTOR_KINDS, "combined")
            },
        }
    timing["all_tested_block_optima"] = {
        "baseline": _optima(baseline_blocks, baseline_sizes),
        "candidate": _optima(candidate_blocks, candidate_sizes),
    }

    passed = not errors
    return {
        "format_version": FORMAT_VERSION,
        "status": "passed" if passed else "failed",
        "authoritative": passed,
        "provenance": provenance_checks,
        "compatibility": {
            "all_exact": all(check["exact"] for check in compatibility_checks),
            "checks": compatibility_checks,
        },
        "block_sizes": block_summary,
        "final_outputs": {
            "all_common_blocks_exact": all_outputs_exact and bool(common_sizes),
            "per_common_block": output_blocks,
        },
        "timing": timing,
        "errors": errors,
    }


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ResultContractError(f"{path} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ResultContractError(f"{path} contains non-finite JSON number {value}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_nonfinite,
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ResultContractError(f"could not load {path}: {error}") from error
    return _require_mapping(value, str(path))


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)


def _fixture(*, block_sizes: list[int] | None = None, timing_scale: float = 1.0) -> dict[str, Any]:
    block_sizes = block_sizes or [4, 8]
    array_evidence = {"shape": [1], "dtype": "<i4", "sha256": "3" * 64}
    workload_payload = {
        "scalars": {
            "self_contact_margin": 0.01,
            "rest_shape_exclusion_radius": 0.02,
            "edge_edge_parallel_epsilon": 1.0e-5,
            "record_triangle_contacting_vertices": False,
            "world_count": 2,
            "world_copies": 2,
            "particle_count": 8,
            "triangle_count": 6,
            "edge_count": 12,
        },
        "arrays": {key: dict(array_evidence) for key in WORKLOAD_ARRAY_KEYS},
        "bvh_constructors": {
            "triangles": {"constructor": 0, "leaf_size": 1},
            "edges": {"constructor": 0, "leaf_size": 1},
        },
    }
    workload_payload["arrays"]["vertex_positions"] = {
        "shape": [8, 3],
        "dtype": "<f4",
        "sha256": "2" * 64,
    }
    results = []
    schedule = []
    for sample in range(5):
        for block_size in block_sizes:
            for kind in DETECTOR_KINDS:
                schedule.append({"sample": sample, "block_size": block_size, "kind": kind})
    for block_size in block_sizes:
        record: dict[str, Any] = {"block_size": block_size}
        for kind_index, kind in enumerate(DETECTOR_KINDS):
            microseconds = timing_scale * (block_size + kind_index + 1.0)
            elapsed = microseconds * 10 / 1000.0
            logical_threads = 8 if kind == "vertex_triangle" else 12
            blocks = math.ceil(logical_threads / block_size)
            warps_per_block = math.ceil(block_size / 32)
            record[kind] = {
                "sample_elapsed_ms": [elapsed] * 5,
                "sample_microseconds_per_launch": [microseconds] * 5,
                "median_microseconds_per_launch": microseconds,
                "minimum_microseconds_per_launch": microseconds,
                "launch_geometry": {
                    "logical_threads": logical_threads,
                    "blocks": blocks,
                    "warps_per_block": warps_per_block,
                    "physical_warps": blocks * warps_per_block,
                    "physical_warps_per_sm": blocks * warps_per_block / 2,
                    "full_block_active_lane_fraction": block_size / (32 * warps_per_block),
                },
                "outputs": {
                    "raw_count": 4,
                    "stored_count": 4,
                    "nonempty_rows": 2,
                    "overflow_rows": 0,
                    "counts_sha256": "4" * 64,
                    "active_pairs_sha256": "5" * 64,
                    "owner_min_distances_sha256": "6" * 64,
                    "resize_flags": [0, 0],
                    "resize_flags_sha256": "7" * 64,
                },
                "exactly_equal_to_reference": {
                    "counts": True,
                    "active_pairs": True,
                    "owner_min_distances": True,
                    "other_min_distances": True,
                    "resize_flags": True,
                    "all": True,
                },
                "world_copies_exact": {
                    "shapes": {"counts": True, "owner_min_distances": True, "active_pairs": True},
                    "counts": True,
                    "active_pairs": True,
                    "owner_min_distances": True,
                    "targets_stay_in_world": True,
                    "raw_counts_per_world": [2, 2],
                    "active_pair_counts_per_world": [2, 2],
                    "all": True,
                },
            }
        results.append(record)
    return {
        "source_root": "/newton",
        "git_head": "a" * 40,
        "newton": {
            "root": "/newton",
            "newton_file": "/newton/newton/__init__.py",
            "version": "1.0",
            "git_head": "a" * 40,
            "git_tree": "b" * 40,
            "git_dirty_tracked": False,
            "git_status_tracked": [],
        },
        "harness": "/newton/profiling/benchmark.py",
        "harness_sha256": "c" * 64,
        "argv": ["benchmark.py"],
        "warp": {
            "root": "/warp",
            "warp_file": "/warp/warp/__init__.py",
            "version": "1.17.0.dev4",
            "git_head": "d" * 40,
            "git_tree": "e" * 40,
            "git_dirty_tracked": False,
            "git_status_tracked": [],
            "native_libraries": {
                "core": {"path": "/warp/warp/bin/warp.so", "size": 1, "sha256": "f" * 64},
                "clang": {"path": "/warp/warp/bin/warp-clang.so", "size": 1, "sha256": "0" * 64},
            },
        },
        "device": "cuda:0",
        "sm_count": 2,
        "state": {"kind": "npz", "path": "/state.npz", "sha256": "1" * 64},
        "frozen_particle_q_sha256": "1" * 64,
        "workload_particle_q_sha256": "2" * 64,
        "workload": {"sha256": _canonical_sha256(workload_payload), **workload_payload},
        "replication": {
            "mode": "replicated_independent_worlds",
            "world_copies": 2,
            "source_particle_count": 4,
            "source_triangle_count": 3,
            "source_edge_count": 6,
            "particle_world_layout_exact": True,
            "rest_positions_tiled_exactly": True,
        },
        "particle_count": 8,
        "triangle_count": 6,
        "edge_count": 12,
        "warmups": 1,
        "repeats": 10,
        "samples": 5,
        "block_sizes": block_sizes,
        "timing_schedule": schedule,
        "reference_block_size": block_sizes[0],
        "all_outputs_exactly_equal": True,
        "all_world_copies_exact": True,
        "results": results,
    }


class ComparatorSelfTest(unittest.TestCase):
    """Exercise the fail-closed comparison contract without a GPU."""

    def test_accepts_timing_differences_and_reports_speedup(self) -> None:
        """Verify exact outputs pass while timing values differ."""
        baseline = _fixture(timing_scale=2.0)
        candidate = _fixture(timing_scale=1.0)
        report = compare_results(baseline, candidate)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["final_outputs"]["all_common_blocks_exact"])
        self.assertEqual(
            report["timing"]["per_common_block"][0]["combined"]["baseline_over_candidate_median_speedup"],
            2.0,
        )

    def test_rejects_final_output_difference(self) -> None:
        """Verify one changed canonical pair hash fails the comparison."""
        baseline = _fixture()
        candidate = _fixture()
        candidate["results"][1]["edge_edge"]["outputs"]["active_pairs_sha256"] = "8" * 64
        report = compare_results(baseline, candidate)
        self.assertEqual(report["status"], "failed")
        self.assertIn("block 8 edge_edge final outputs differ", report["errors"])

    def test_rejects_missing_or_tampered_provenance(self) -> None:
        """Verify missing Newton provenance and altered workload hashes fail closed."""
        missing = _fixture()
        del missing["newton"]
        with self.assertRaises(ResultContractError):
            validate_result(missing, "missing")

        tampered = _fixture()
        tampered["workload"]["scalars"]["world_copies"] = 3
        with self.assertRaises(ResultContractError):
            validate_result(tampered, "tampered")

    def test_requires_declared_code_provenance_differences(self) -> None:
        """Verify differing Newton and Warp identities require exact declarations."""
        baseline = _fixture()
        candidate = _fixture()
        candidate["newton"]["git_head"] = candidate["git_head"] = "9" * 40
        candidate["newton"]["git_tree"] = "8" * 40
        candidate["warp"]["native_libraries"]["core"]["sha256"] = "7" * 64
        rejected = compare_results(baseline, candidate)
        self.assertEqual(rejected["status"], "failed")

        baseline_warp_id = rejected["provenance"][1]["baseline"]["identity_sha256"]
        candidate_warp_id = rejected["provenance"][1]["candidate"]["identity_sha256"]
        accepted = compare_results(
            baseline,
            candidate,
            expected_baseline_newton_head="a" * 40,
            expected_candidate_newton_head="9" * 40,
            expected_baseline_warp_identity=baseline_warp_id,
            expected_candidate_warp_identity=candidate_warp_id,
        )
        self.assertEqual(accepted["status"], "passed")

    def test_compares_every_common_block(self) -> None:
        """Verify partial block overlap is explicit and fully compared."""
        baseline = _fixture(block_sizes=[4, 8])
        candidate = _fixture(block_sizes=[8, 16])
        report = compare_results(baseline, candidate)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["block_sizes"]["common"], [8])
        self.assertEqual(len(report["final_outputs"]["per_common_block"]), 1)


def _run_self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ComparatorSelfTest)
    with tempfile.TemporaryFile(mode="w+") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        stream.seek(0)
        details = stream.read()
    report = {
        "format_version": FORMAT_VERSION,
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "details": details,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--expect-baseline-newton-head")
    parser.add_argument("--expect-candidate-newton-head")
    parser.add_argument("--expect-baseline-warp-identity")
    parser.add_argument("--expect-candidate-warp-identity")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.baseline is not None or args.candidate is not None or args.output is not None:
            parser.error("--self-test does not accept result paths or --output")
    elif args.baseline is None or args.candidate is None:
        parser.error("baseline and candidate result paths are required")
    for name in (
        "expect_baseline_newton_head",
        "expect_candidate_newton_head",
    ):
        value = getattr(args, name)
        if value is not None and not _is_sha(value, (40, 64)):
            parser.error(f"--{name.replace('_', '-')} must be a full hexadecimal Git object ID")
    for name in (
        "expect_baseline_warp_identity",
        "expect_candidate_warp_identity",
    ):
        value = getattr(args, name)
        if value is not None and not _is_sha(value):
            parser.error(f"--{name.replace('_', '-')} must be a hexadecimal SHA-256")
    return args


def main() -> int:
    args = _parse_args()
    if args.self_test:
        return _run_self_test()
    try:
        baseline = _load_json(args.baseline)
        candidate = _load_json(args.candidate)
        report = compare_results(
            baseline,
            candidate,
            expected_baseline_newton_head=args.expect_baseline_newton_head,
            expected_candidate_newton_head=args.expect_candidate_newton_head,
            expected_baseline_warp_identity=args.expect_baseline_warp_identity,
            expected_candidate_warp_identity=args.expect_candidate_warp_identity,
        )
        report["inputs"] = {
            "baseline": {"path": str(args.baseline.resolve()), "sha256": _file_sha256(args.baseline)},
            "candidate": {"path": str(args.candidate.resolve()), "sha256": _file_sha256(args.candidate)},
        }
    except ResultContractError as error:
        report = {
            "format_version": FORMAT_VERSION,
            "status": "failed",
            "authoritative": False,
            "inputs": {
                "baseline": str(args.baseline.resolve()),
                "candidate": str(args.candidate.resolve()),
            },
            "errors": [str(error)],
        }
    try:
        _write_report(report, args.output)
    except OSError as error:
        print(f"could not write comparison output: {error}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
