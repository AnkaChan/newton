#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Separate cloth Franka EE capsule traversal cost from avoided narrow work.

The diagnostic uses one frozen cloth state and the production world-group
passes. It times count-only AABB and capsule BVH traversal, then replays each
exact candidate set through the production EE filters and closest-point tests
without traversing a BVH. Native BVH node visits are intentionally not claimed:
Warp's public query API exposes yielded primitive indices, not internal nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_common import binary_search

if __package__:
    from . import benchmark_cloth_franka_detector_blocks as frozen_benchmark
    from . import compare_cloth_franka_broad_phase_candidates as broad_phase
else:
    import benchmark_cloth_franka_detector_blocks as frozen_benchmark
    import compare_cloth_franka_broad_phase_candidates as broad_phase


_VERSION = "cloth_franka_ee_capsule_cost_v2"
print(f"[diagnose_cloth_franka_ee_capsule_cost] version: {_VERSION}")

DEFAULT_BLOCK_SIZES = (1, 2, 4, 8, 12)


@wp.kernel
def _replay_edge_edge_narrow_phase(
    max_query_radius: float,
    min_query_radius: float,
    positions: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    candidate_offsets: wp.array[wp.int32],
    candidate_targets: wp.array[wp.int32],
    edge_edge_parallel_epsilon: float,
    edge_filtering_list: wp.array[wp.int32],
    edge_filtering_list_offsets: wp.array[wp.int32],
    reference_positions: wp.array[wp.vec3],
    candidate_counts: wp.array[wp.int32],
    closest_point_counts: wp.array[wp.int32],
    current_hit_counts: wp.array[wp.int32],
    reference_evaluation_counts: wp.array[wp.int32],
    accepted_counts: wp.array[wp.int32],
    minimum_distances: wp.array[float],
    accepted_targets: wp.array[wp.int32],
):
    """Replay production EE filtering and narrow phase over stored candidates."""
    edge_index = wp.tid()
    owner_v0 = edge_indices[edge_index, 2]
    owner_v1 = edge_indices[edge_index, 3]
    owner_p0 = positions[owner_v0]
    owner_p1 = positions[owner_v1]

    use_edge_filter = False
    filter_start = wp.int32(0)
    filter_end = wp.int32(0)
    filter_first = wp.int32(0)
    filter_last = wp.int32(0)
    if edge_filtering_list:
        filter_start = edge_filtering_list_offsets[edge_index]
        filter_end = edge_filtering_list_offsets[edge_index + 1]
        if filter_end > filter_start:
            use_edge_filter = True
            filter_first = edge_filtering_list[filter_start]
            filter_last = edge_filtering_list[filter_end - 1]

    use_reference_filter = False
    owner_ref_p0 = wp.vec3(0.0)
    owner_ref_p1 = wp.vec3(0.0)
    if reference_positions and min_query_radius > 0.0:
        use_reference_filter = True
        owner_ref_p0 = reference_positions[owner_v0]
        owner_ref_p1 = reference_positions[owner_v1]

    visited = wp.int32(0)
    closest_point_evaluations = wp.int32(0)
    current_hits = wp.int32(0)
    reference_evaluations = wp.int32(0)
    accepted = wp.int32(0)
    minimum_distance = max_query_radius

    start = candidate_offsets[edge_index]
    end = candidate_offsets[edge_index + 1]
    for candidate_slot in range(start, end):
        candidate_edge_index = candidate_targets[candidate_slot]
        visited += 1

        candidate_v0 = edge_indices[candidate_edge_index, 2]
        candidate_v1 = edge_indices[candidate_edge_index, 3]
        if owner_v0 == candidate_v0 or owner_v0 == candidate_v1 or owner_v1 == candidate_v0 or owner_v1 == candidate_v1:
            continue

        if use_edge_filter:
            if candidate_edge_index >= filter_first and candidate_edge_index <= filter_last:
                filter_index = binary_search(
                    edge_filtering_list,
                    candidate_edge_index,
                    filter_start,
                    filter_end,
                )
                if filter_index > filter_start and edge_filtering_list[filter_index - 1] == candidate_edge_index:
                    continue

        closest_point_evaluations += 1
        candidate_p0 = positions[candidate_v0]
        candidate_p1 = positions[candidate_v1]
        current_closest = wp.closest_point_edge_edge(
            owner_p0,
            owner_p1,
            candidate_p0,
            candidate_p1,
            edge_edge_parallel_epsilon,
        )
        distance = current_closest[2]
        if distance < max_query_radius:
            current_hits += 1
            if use_reference_filter:
                reference_evaluations += 1
                reference_closest = wp.closest_point_edge_edge(
                    owner_ref_p0,
                    owner_ref_p1,
                    reference_positions[candidate_v0],
                    reference_positions[candidate_v1],
                    edge_edge_parallel_epsilon,
                )
                if reference_closest[2] < min_query_radius:
                    continue

            accepted_targets[start + accepted] = candidate_edge_index
            accepted += 1
            minimum_distance = wp.min(minimum_distance, distance)

    candidate_counts[edge_index] = visited
    closest_point_counts[edge_index] = closest_point_evaluations
    current_hit_counts[edge_index] = current_hits
    reference_evaluation_counts[edge_index] = reference_evaluations
    accepted_counts[edge_index] = accepted
    minimum_distances[edge_index] = minimum_distance


@dataclass(frozen=True)
class ReplayBuffers:
    """Hold one immutable candidate payload and its replay outputs."""

    offsets: wp.array[wp.int32]
    targets: wp.array[wp.int32]
    candidate_counts: wp.array[wp.int32]
    closest_point_counts: wp.array[wp.int32]
    current_hit_counts: wp.array[wp.int32]
    reference_evaluation_counts: wp.array[wp.int32]
    accepted_counts: wp.array[wp.int32]
    minimum_distances: wp.array[float]
    accepted_targets: wp.array[wp.int32]


@dataclass(frozen=True)
class CandidatePayload:
    """Hold canonical evidence and native query-order replay arrays."""

    candidate_set: broad_phase.CandidateSet
    offsets: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class ReplaySnapshot:
    """Hold exact host results from one narrow-phase replay."""

    candidate_counts: np.ndarray
    closest_point_counts: np.ndarray
    current_hit_counts: np.ndarray
    reference_evaluation_counts: np.ndarray
    accepted_counts: np.ndarray
    minimum_distances: np.ndarray
    accepted_pairs: np.ndarray


def _candidate_payload_arrays(candidate_set: broad_phase.CandidateSet) -> tuple[np.ndarray, np.ndarray]:
    """Convert canonical candidate pairs into per-owner CSR arrays for tests."""
    counts = np.ascontiguousarray(candidate_set.counts, dtype=np.int32)
    pairs = np.asarray(candidate_set.pairs)
    if counts.ndim != 1:
        raise ValueError("candidate counts must be one-dimensional")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != np.int32:
        raise ValueError("candidate pairs must have shape (N, 2) and dtype int32")
    if np.any(counts < 0):
        raise ValueError("candidate counts cannot be negative")

    offsets_i64 = np.empty(len(counts) + 1, dtype=np.int64)
    offsets_i64[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets_i64[1:])
    if int(offsets_i64[-1]) != len(pairs):
        raise ValueError("candidate pair length does not match the per-owner counts")
    if int(offsets_i64[-1]) > np.iinfo(np.int32).max:
        raise ValueError("candidate payload exceeds int32 addressing")

    expected_owners = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    if not frozen_benchmark._same_array(pairs[:, 0], expected_owners):
        raise ValueError("candidate pairs are not canonical by owner")
    offsets = np.ascontiguousarray(offsets_i64.astype(np.int32))
    targets = np.ascontiguousarray(pairs[:, 1], dtype=np.int32)
    return offsets, targets


def _enumerate_candidate_payload(
    workload: frozen_benchmark.DetectorWorkload,
    kernel,
    block_size: int,
) -> CandidatePayload:
    """Enumerate EE candidates while retaining each query's native yield order."""
    if kernel is None:
        raise RuntimeError("the imported Warp build does not provide the required radius-query APIs")
    owner_count = workload.model.edge_count
    device = workload.model.device

    count_buffer = wp.empty(owner_count, dtype=wp.int32, device=device)
    broad_phase._launch_candidate_pass(
        workload,
        "edge_edge",
        kernel,
        None,
        count_buffer,
        None,
        block_size,
    )
    counts = np.ascontiguousarray(count_buffer.numpy(), dtype=np.int32)
    if np.any(counts < 0):
        raise RuntimeError("candidate count overflowed int32")

    offsets_i64 = np.empty(owner_count + 1, dtype=np.int64)
    offsets_i64[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets_i64[1:])
    total = int(offsets_i64[-1])
    if total > np.iinfo(np.int32).max:
        raise RuntimeError(f"candidate payload exceeds int32 addressing: {total}")
    offsets = np.ascontiguousarray(offsets_i64.astype(np.int32))

    store_counts_buffer = wp.empty(owner_count, dtype=wp.int32, device=device)
    targets_buffer = wp.empty(total, dtype=wp.int32, device=device)
    broad_phase._launch_candidate_pass(
        workload,
        "edge_edge",
        kernel,
        wp.array(offsets, dtype=wp.int32, device=device),
        store_counts_buffer,
        targets_buffer,
        block_size,
    )
    store_counts = np.ascontiguousarray(store_counts_buffer.numpy(), dtype=np.int32)
    if not frozen_benchmark._same_array(counts, store_counts):
        mismatch = int(np.flatnonzero(counts != store_counts)[0])
        raise RuntimeError(
            f"candidate count changed between passes for owner {mismatch}: "
            f"{counts[mismatch]} != {store_counts[mismatch]}"
        )
    targets = np.ascontiguousarray(targets_buffer.numpy(), dtype=np.int32)
    candidate_set = broad_phase.CandidateSet(
        counts=counts,
        pairs=broad_phase._canonical_pairs(counts, targets),
    )
    return CandidatePayload(candidate_set=candidate_set, offsets=offsets, targets=targets)


def _allocate_replay_buffers(
    payload: CandidatePayload,
    device,
) -> ReplayBuffers:
    """Allocate one device replay payload and fixed-size output rows."""
    owner_count = len(payload.candidate_set.counts)
    return ReplayBuffers(
        offsets=wp.array(payload.offsets, dtype=wp.int32, device=device),
        targets=wp.array(payload.targets, dtype=wp.int32, device=device),
        candidate_counts=wp.empty(owner_count, dtype=wp.int32, device=device),
        closest_point_counts=wp.empty(owner_count, dtype=wp.int32, device=device),
        current_hit_counts=wp.empty(owner_count, dtype=wp.int32, device=device),
        reference_evaluation_counts=wp.empty(owner_count, dtype=wp.int32, device=device),
        accepted_counts=wp.empty(owner_count, dtype=wp.int32, device=device),
        minimum_distances=wp.empty(owner_count, dtype=float, device=device),
        accepted_targets=wp.empty(len(payload.targets), dtype=wp.int32, device=device),
    )


def _launch_traversal_count(
    workload: frozen_benchmark.DetectorWorkload,
    kernel,
    counts: wp.array[wp.int32],
    block_size: int,
) -> None:
    """Launch one public-API EE query without filters or narrow phase."""
    broad_phase._launch_candidate_pass(
        workload,
        "edge_edge",
        kernel,
        None,
        counts,
        None,
        block_size,
    )


def _launch_narrow_replay(
    workload: frozen_benchmark.DetectorWorkload,
    buffers: ReplayBuffers,
    block_size: int,
) -> None:
    """Launch one BVH-free replay of production EE filtering and distance tests."""
    detector = workload.detector
    model = workload.model
    wp.launch(
        kernel=_replay_edge_edge_narrow_phase,
        dim=model.edge_count,
        inputs=[
            workload.self_contact_margin,
            workload.rest_shape_exclusion_radius,
            detector.vertex_positions,
            model.edge_indices,
            buffers.offsets,
            buffers.targets,
            detector.edge_edge_parallel_epsilon,
            detector.edge_filtering_list,
            detector.edge_filtering_list_offsets,
            workload.rest_positions,
        ],
        outputs=[
            buffers.candidate_counts,
            buffers.closest_point_counts,
            buffers.current_hit_counts,
            buffers.reference_evaluation_counts,
            buffers.accepted_counts,
            buffers.minimum_distances,
            buffers.accepted_targets,
        ],
        device=model.device,
        block_dim=block_size,
    )


def _capture_traversal_graph(
    workload: frozen_benchmark.DetectorWorkload,
    kernel,
    counts: wp.array[wp.int32],
    block_size: int,
):
    """Capture one count-only query graph after preloading its block specialization."""
    wp.load_module(broad_phase, device=workload.model.device, block_dim=block_size)
    with wp.ScopedCapture(device=workload.model.device) as capture:
        _launch_traversal_count(workload, kernel, counts, block_size)
    return capture.graph


def _capture_replay_graph(
    workload: frozen_benchmark.DetectorWorkload,
    buffers: ReplayBuffers,
    block_size: int,
):
    """Capture one BVH-free narrow replay graph after module preload."""
    wp.load_module(sys.modules[__name__], device=workload.model.device, block_dim=block_size)
    with wp.ScopedCapture(device=workload.model.device) as capture:
        _launch_narrow_replay(workload, buffers, block_size)
    return capture.graph


def _accepted_pairs(
    payload: CandidatePayload,
    accepted_counts: np.ndarray,
    accepted_targets: np.ndarray,
) -> np.ndarray:
    """Canonicalize active prefixes from the per-owner accepted-target rows."""
    counts = np.asarray(accepted_counts)
    targets = np.asarray(accepted_targets)
    if counts.ndim != 1 or counts.dtype != np.int32:
        raise ValueError("accepted counts must be a one-dimensional int32 array")
    if targets.ndim != 1 or targets.dtype != np.int32:
        raise ValueError("accepted targets must be a one-dimensional int32 array")
    if counts.shape != payload.candidate_set.counts.shape:
        raise ValueError("accepted count shape does not match candidate owners")
    if len(targets) != len(payload.targets):
        raise ValueError("accepted target capacity does not match candidate targets")
    if np.any(counts < 0) or np.any(counts > payload.candidate_set.counts):
        raise ValueError("accepted counts exceed candidate row capacities")

    pairs = np.empty((int(counts.sum(dtype=np.int64)), 2), dtype=np.int32)
    cursor = 0
    for owner, (offset, count) in enumerate(zip(payload.offsets[:-1], counts, strict=True)):
        row_count = int(count)
        if row_count:
            pairs[cursor : cursor + row_count, 0] = owner
            pairs[cursor : cursor + row_count, 1] = targets[int(offset) : int(offset) + row_count]
            cursor += row_count
    if len(pairs):
        pairs = np.ascontiguousarray(pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))])
    return pairs


def _snapshot_replay(buffers: ReplayBuffers, payload: CandidatePayload) -> ReplaySnapshot:
    """Copy one replay result to contiguous host arrays."""
    accepted_counts = np.ascontiguousarray(buffers.accepted_counts.numpy(), dtype=np.int32)
    accepted_targets = np.ascontiguousarray(buffers.accepted_targets.numpy(), dtype=np.int32)
    return ReplaySnapshot(
        candidate_counts=np.ascontiguousarray(buffers.candidate_counts.numpy(), dtype=np.int32),
        closest_point_counts=np.ascontiguousarray(buffers.closest_point_counts.numpy(), dtype=np.int32),
        current_hit_counts=np.ascontiguousarray(buffers.current_hit_counts.numpy(), dtype=np.int32),
        reference_evaluation_counts=np.ascontiguousarray(buffers.reference_evaluation_counts.numpy(), dtype=np.int32),
        accepted_counts=accepted_counts,
        minimum_distances=np.ascontiguousarray(buffers.minimum_distances.numpy(), dtype=np.float32),
        accepted_pairs=_accepted_pairs(payload, accepted_counts, accepted_targets),
    )


def _snapshot_equality(left: ReplaySnapshot, right: ReplaySnapshot) -> dict[str, bool]:
    """Return bitwise field equality for two replay snapshots."""
    equality = {
        field: frozen_benchmark._same_array(getattr(left, field), getattr(right, field))
        for field in ReplaySnapshot.__dataclass_fields__
    }
    equality["all"] = all(equality.values())
    return equality


def _replay_summary(snapshot: ReplaySnapshot) -> dict[str, object]:
    """Summarize the exact candidate-to-contact funnel and output hashes."""
    arrays = {field: getattr(snapshot, field) for field in ReplaySnapshot.__dataclass_fields__}
    return {
        "owner_count": len(snapshot.candidate_counts),
        "total_candidates": int(snapshot.candidate_counts.sum(dtype=np.int64)),
        "closest_point_evaluations": int(snapshot.closest_point_counts.sum(dtype=np.int64)),
        "current_radius_hits": int(snapshot.current_hit_counts.sum(dtype=np.int64)),
        "reference_evaluations": int(snapshot.reference_evaluation_counts.sum(dtype=np.int64)),
        "accepted_contacts": int(snapshot.accepted_counts.sum(dtype=np.int64)),
        "nonempty_contact_rows": int(np.count_nonzero(snapshot.accepted_counts)),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": broad_phase._hash_array(value),
            }
            for name, value in arrays.items()
        },
    }


def _candidate_payload_evidence(payload: CandidatePayload) -> dict[str, object]:
    """Return bitwise hashes for canonical and native-order candidate data."""
    arrays = {
        "counts": payload.candidate_set.counts,
        "canonical_pairs": payload.candidate_set.pairs,
        "offsets": payload.offsets,
        "native_order_targets": payload.targets,
    }
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": broad_phase._hash_array(value),
        }
        for name, value in arrays.items()
    }


def _timing_summary(sample_elapsed_ms: Sequence[float], repeats: int) -> dict[str, object]:
    """Convert repeated-graph elapsed samples into per-launch microseconds."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not sample_elapsed_ms:
        raise ValueError("at least one timing sample is required")
    elapsed = [float(value) for value in sample_elapsed_ms]
    if any(not np.isfinite(value) or value < 0.0 for value in elapsed):
        raise ValueError("timing samples must be finite and non-negative")
    microseconds = [1000.0 * value / repeats for value in elapsed]
    return {
        "sample_elapsed_ms": elapsed,
        "sample_microseconds_per_launch": microseconds,
        "median_microseconds_per_launch": float(np.median(microseconds)),
        "minimum_microseconds_per_launch": min(microseconds),
    }


def _self_test() -> None:
    """Exercise host-only payload, identity, summary, and timing accounting."""
    candidate_set = broad_phase.CandidateSet(
        counts=np.array([2, 0, 1], dtype=np.int32),
        pairs=np.array([[0, 3], [0, 7], [2, 5]], dtype=np.int32),
    )
    offsets, targets = _candidate_payload_arrays(candidate_set)
    if not frozen_benchmark._same_array(offsets, np.array([0, 2, 2, 3], dtype=np.int32)):
        raise AssertionError("candidate CSR offsets are incorrect")
    if not frozen_benchmark._same_array(targets, np.array([3, 7, 5], dtype=np.int32)):
        raise AssertionError("candidate CSR targets are incorrect")

    payload = CandidatePayload(candidate_set=candidate_set, offsets=offsets, targets=targets)
    accepted_pairs = _accepted_pairs(
        payload,
        np.array([1, 0, 1], dtype=np.int32),
        np.array([7, -1, 5], dtype=np.int32),
    )
    expected_accepted_pairs = np.array([[0, 7], [2, 5]], dtype=np.int32)
    if not frozen_benchmark._same_array(accepted_pairs, expected_accepted_pairs):
        raise AssertionError("accepted-pair identities are incorrect")

    snapshot = ReplaySnapshot(
        candidate_counts=np.array([2, 0, 1], dtype=np.int32),
        closest_point_counts=np.array([1, 0, 1], dtype=np.int32),
        current_hit_counts=np.array([1, 0, 0], dtype=np.int32),
        reference_evaluation_counts=np.array([1, 0, 0], dtype=np.int32),
        accepted_counts=np.array([1, 0, 0], dtype=np.int32),
        minimum_distances=np.array([0.1, 0.2, 0.2], dtype=np.float32),
        accepted_pairs=np.array([[0, 7]], dtype=np.int32),
    )
    summary = _replay_summary(snapshot)
    if summary["total_candidates"] != 3 or summary["closest_point_evaluations"] != 2:
        raise AssertionError("replay funnel totals are incorrect")
    if not _snapshot_equality(snapshot, snapshot)["all"]:
        raise AssertionError("identical replay snapshots compare unequal")
    different_bits = ReplaySnapshot(
        **{
            **snapshot.__dict__,
            "minimum_distances": np.array([0.1, 0.2, -0.0], dtype=np.float32),
        }
    )
    positive_zero = ReplaySnapshot(
        **{
            **snapshot.__dict__,
            "minimum_distances": np.array([0.1, 0.2, 0.0], dtype=np.float32),
        }
    )
    if _snapshot_equality(different_bits, positive_zero)["minimum_distances"]:
        raise AssertionError("bitwise equality accepted distinct signed-zero encodings")

    timing = _timing_summary([1.0, 3.0, 2.0], repeats=10)
    if timing["median_microseconds_per_launch"] != 200.0:
        raise AssertionError("timing conversion is incorrect")

    invalid = broad_phase.CandidateSet(
        counts=np.array([1], dtype=np.int32),
        pairs=np.array([[1, 3]], dtype=np.int32),
    )
    try:
        _candidate_payload_arrays(invalid)
    except ValueError as error:
        if "canonical by owner" not in str(error):
            raise
    else:
        raise AssertionError("noncanonical candidate owners were accepted")
    try:
        _accepted_pairs(
            payload,
            np.array([3, 0, 1], dtype=np.int32),
            np.array([3, 7, 5], dtype=np.int32),
        )
    except ValueError as error:
        if "row capacities" not in str(error):
            raise
    else:
        raise AssertionError("oversized accepted row was accepted")

    with tempfile.TemporaryDirectory(prefix="cloth-ee-diagnostic-self-test-") as directory:
        output_path = Path(directory) / "result.json"
        _write_json_exclusive(output_path, '{"complete": true}\n')
        if output_path.read_text(encoding="utf-8") != '{"complete": true}\n':
            raise AssertionError("atomic output contents are incorrect")
        try:
            _write_json_exclusive(output_path, "{}\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic output replaced an existing result")
    print("EE capsule cost diagnostic host-accounting self-test passed")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-input", type=Path, help="Frozen NPZ containing source-world particle_q")
    parser.add_argument("--world-copies", type=int, default=1, help="Replicate the frozen workload this many times")
    parser.add_argument("--block-sizes", nargs="+", type=int, default=list(DEFAULT_BLOCK_SIZES))
    parser.add_argument("--accounting-block-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", help="Warp CUDA device alias; defaults to Warp's preferred device")
    parser.add_argument("--output", type=Path, help="Write JSON exclusively to this path; otherwise print it")
    parser.add_argument("--expected-newton-root", type=Path, help="Required absolute Newton import root")
    parser.add_argument("--expected-newton-git-head", help="Required clean Newton Git HEAD")
    parser.add_argument("--expected-warp-root", type=Path, help="Required absolute Warp import root")
    parser.add_argument(
        "--expected-warp-git-head",
        help=f"Required pinned Warp Git HEAD ({broad_phase.PINNED_WARP_GIT_HEAD})",
    )
    parser.add_argument("--expected-warp-core-sha256", help="Required SHA-256 of warp.so")
    parser.add_argument("--expected-warp-clang-sha256", help="Required SHA-256 of warp-clang.so")
    parser.add_argument("--self-test", action="store_true", help="Run host-only accounting tests and exit")
    args = parser.parse_args(argv)

    if not args.block_sizes or any(not 1 <= value <= 256 for value in args.block_sizes):
        parser.error("--block-sizes must contain values in [1, 256]")
    if len(set(args.block_sizes)) != len(args.block_sizes):
        parser.error("--block-sizes must not contain duplicates")
    if not 1 <= args.accounting_block_size <= 256:
        parser.error("--accounting-block-size must be in [1, 256]")
    if args.world_copies < 1:
        parser.error("--world-copies must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.samples < 5:
        parser.error("--samples must be at least 5")
    return args


def _source_hashes(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Snapshot exact diagnostic source identities."""
    return {name: {"path": str(path), "sha256": frozen_benchmark._file_hash(path)} for name, path in paths.items()}


def _write_json_exclusive(path: Path, serialized: str) -> None:
    """Atomically publish a durable JSON file without replacing an existing path."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the frozen EE traversal and narrow-replay diagnostic."""
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return
    broad_phase._require_authoritative_arguments(args)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.state_input is None or not args.state_input.is_file():
        raise FileNotFoundError(args.state_input)
    if not broad_phase._RADIUS_QUERIES_AVAILABLE:
        raise RuntimeError("the pinned Warp import does not expose bvh_query_capsule")

    source_paths = {
        "harness": Path(__file__).resolve(),
        "candidate_accounting_harness": Path(broad_phase.__file__).resolve(),
        "frozen_workload_producer": Path(frozen_benchmark.__file__).resolve(),
    }
    source_hashes_start = _source_hashes(source_paths)
    argv_start = list(sys.argv if argv is None else [sys.argv[0], *argv])

    newton_provenance = frozen_benchmark._newton_provenance(
        expected_root=args.expected_newton_root,
        expected_git_head=args.expected_newton_git_head,
    )
    warp_provenance = frozen_benchmark._warp_provenance(
        expected_root=args.expected_warp_root,
        expected_git_head=args.expected_warp_git_head,
        expected_core_sha256=args.expected_warp_core_sha256,
        expected_clang_sha256=args.expected_warp_clang_sha256,
    )

    wp.init()
    if args.device:
        wp.set_device(args.device)
    workload, replication, state_provenance = broad_phase._load_workload(
        args.state_input.resolve(),
        args.world_copies,
    )
    model = workload.model
    detector = workload.detector
    if not model.device.is_cuda:
        raise RuntimeError("timing requires a CUDA device")

    candidate_payloads = {
        "padded_aabb": _enumerate_candidate_payload(
            workload,
            broad_phase._count_or_store_edge_edge_aabb_candidates,
            args.accounting_block_size,
        ),
        "capsule": _enumerate_candidate_payload(
            workload,
            broad_phase._count_or_store_edge_edge_capsule_candidates,
            args.accounting_block_size,
        ),
    }
    candidate_sets = {name: payload.candidate_set for name, payload in candidate_payloads.items()}
    candidate_comparison = broad_phase._compare_candidate_sets(
        candidate_sets["padded_aabb"],
        candidate_sets["capsule"],
    )

    traversal_kernels = {
        "padded_aabb": broad_phase._count_or_store_edge_edge_aabb_candidates,
        "capsule": broad_phase._count_or_store_edge_edge_capsule_candidates,
    }
    block_candidate_payloads: dict[tuple[str, int], CandidatePayload] = {}
    block_candidate_checks: dict[tuple[str, int], dict[str, bool]] = {}
    for block_size in args.block_sizes:
        for name, reference_payload in candidate_payloads.items():
            block_payload = (
                reference_payload
                if block_size == args.accounting_block_size
                else _enumerate_candidate_payload(workload, traversal_kernels[name], block_size)
            )
            candidate_identity = {
                "counts": frozen_benchmark._same_array(
                    block_payload.candidate_set.counts,
                    reference_payload.candidate_set.counts,
                ),
                "canonical_pairs": frozen_benchmark._same_array(
                    block_payload.candidate_set.pairs,
                    reference_payload.candidate_set.pairs,
                ),
                "native_query_order": frozen_benchmark._same_array(
                    block_payload.targets,
                    reference_payload.targets,
                ),
            }
            candidate_identity["set_exact"] = candidate_identity["counts"] and candidate_identity["canonical_pairs"]
            if not candidate_identity["set_exact"]:
                raise RuntimeError(f"{name} candidate identities changed at block size {block_size}")
            block_candidate_payloads[(name, block_size)] = block_payload
            block_candidate_checks[(name, block_size)] = candidate_identity

    replay_buffers = {
        (name, block_size): _allocate_replay_buffers(payload, model.device)
        for (name, block_size), payload in block_candidate_payloads.items()
    }
    traversal_counts = {
        name: wp.empty(model.edge_count, dtype=wp.int32, device=model.device) for name in candidate_sets
    }
    graphs: dict[tuple[str, str, int], object] = {}
    for block_size in args.block_sizes:
        for name in candidate_sets:
            graphs[("traversal", name, block_size)] = _capture_traversal_graph(
                workload,
                traversal_kernels[name],
                traversal_counts[name],
                block_size,
            )
            graphs[("narrow_replay", name, block_size)] = _capture_replay_graph(
                workload,
                replay_buffers[(name, block_size)],
                block_size,
            )

    for graph in graphs.values():
        for _ in range(args.warmups):
            wp.capture_launch(graph)
    wp.synchronize_device(model.device)

    timing_samples: dict[tuple[str, str, int], list[float]] = {key: [] for key in graphs}
    timing_schedule: list[dict[str, int | str]] = []
    phases_and_queries = [
        ("traversal", "padded_aabb"),
        ("traversal", "capsule"),
        ("narrow_replay", "padded_aabb"),
        ("narrow_replay", "capsule"),
    ]
    for sample in range(args.samples):
        block_order = args.block_sizes if sample % 2 == 0 else list(reversed(args.block_sizes))
        for block_index, block_size in enumerate(block_order):
            operation_order = phases_and_queries
            if (sample + block_index) % 2:
                operation_order = list(reversed(operation_order))
            for phase, query_name in operation_order:
                key = (phase, query_name, block_size)
                elapsed_ms = frozen_benchmark._time_graph(graphs[key], args.repeats, model.device)
                timing_samples[key].append(elapsed_ms)
                timing_schedule.append(
                    {
                        "sample": sample,
                        "block_size": block_size,
                        "phase": phase,
                        "query": query_name,
                    }
                )

    frozen_benchmark._launch_edge_edge(workload, args.block_sizes[0])
    production_snapshot = frozen_benchmark._snapshot_edge_edge(detector)
    production_counts = np.ascontiguousarray(production_snapshot.counts, dtype=np.int32)
    production_minimum_distances = np.ascontiguousarray(
        production_snapshot.owner_min_distances,
        dtype=np.float32,
    )
    production_pairs = np.ascontiguousarray(production_snapshot.active_pairs, dtype=np.int32)
    production_capacities = np.ascontiguousarray(
        detector.edge_colliding_edges_buffer_sizes.numpy(),
        dtype=np.int32,
    )
    overflow_rows = np.flatnonzero(production_counts > production_capacities)
    if len(overflow_rows):
        raise RuntimeError("production detector collision buffers overflowed; accepted-pair oracle is incomplete")

    reference_snapshots: dict[str, ReplaySnapshot] = {}
    block_checks: dict[int, dict[str, object]] = {}
    for block_size in args.block_sizes:
        query_checks: dict[str, object] = {}
        for name in candidate_sets:
            block_payload = block_candidate_payloads[(name, block_size)]
            wp.capture_launch(graphs[("traversal", name, block_size)])
            traversal_count_values = np.ascontiguousarray(traversal_counts[name].numpy(), dtype=np.int32)
            wp.capture_launch(graphs[("narrow_replay", name, block_size)])
            replay_snapshot = _snapshot_replay(
                replay_buffers[(name, block_size)],
                block_payload,
            )
            if name not in reference_snapshots:
                reference_snapshots[name] = replay_snapshot

            traversal_counts_equal = frozen_benchmark._same_array(
                traversal_count_values,
                block_payload.candidate_set.counts,
            )
            replay_candidates_equal = frozen_benchmark._same_array(
                replay_snapshot.candidate_counts,
                block_payload.candidate_set.counts,
            )
            replay_block_equality = _snapshot_equality(reference_snapshots[name], replay_snapshot)
            production_equality = {
                "accepted_counts": frozen_benchmark._same_array(
                    replay_snapshot.accepted_counts,
                    production_counts,
                ),
                "minimum_distances": frozen_benchmark._same_array(
                    replay_snapshot.minimum_distances,
                    production_minimum_distances,
                ),
                "accepted_pairs": frozen_benchmark._same_array(
                    replay_snapshot.accepted_pairs,
                    production_pairs,
                ),
            }
            production_equality["all"] = all(production_equality.values())
            query_checks[name] = {
                "traversal_counts_equal_enumeration": traversal_counts_equal,
                "enumerated_candidate_identity_equal_accounting_block": block_candidate_checks[(name, block_size)],
                "enumerated_candidate_evidence": _candidate_payload_evidence(block_payload),
                "replay_candidate_counts_equal_enumeration": replay_candidates_equal,
                "replay_exactly_equal_to_reference_block": replay_block_equality,
                "replay_exactly_equal_to_production_detector": production_equality,
            }
            if not traversal_counts_equal or not replay_candidates_equal:
                raise RuntimeError(f"{name} candidate counts changed at block size {block_size}")
            if not replay_block_equality["all"] or not production_equality["all"]:
                raise RuntimeError(f"{name} narrow replay changed detector outputs at block size {block_size}")
        block_checks[block_size] = query_checks

    cross_query_output_equality = {
        "accepted_counts": frozen_benchmark._same_array(
            reference_snapshots["padded_aabb"].accepted_counts,
            reference_snapshots["capsule"].accepted_counts,
        ),
        "minimum_distances": frozen_benchmark._same_array(
            reference_snapshots["padded_aabb"].minimum_distances,
            reference_snapshots["capsule"].minimum_distances,
        ),
        "accepted_pairs": frozen_benchmark._same_array(
            reference_snapshots["padded_aabb"].accepted_pairs,
            reference_snapshots["capsule"].accepted_pairs,
        ),
    }
    cross_query_output_equality["all"] = all(cross_query_output_equality.values())
    if not cross_query_output_equality["all"]:
        raise RuntimeError("AABB and capsule candidate replays produced different accepted contacts")

    replay_summaries = {name: _replay_summary(snapshot) for name, snapshot in reference_snapshots.items()}
    avoided_work = {
        key: replay_summaries["padded_aabb"][key] - replay_summaries["capsule"][key]
        for key in (
            "total_candidates",
            "closest_point_evaluations",
            "current_radius_hits",
            "reference_evaluations",
            "accepted_contacts",
        )
    }

    block_results = []
    for block_size in args.block_sizes:
        traversal = {
            name: _timing_summary(timing_samples[("traversal", name, block_size)], args.repeats)
            for name in candidate_sets
        }
        narrow_replay = {
            name: _timing_summary(timing_samples[("narrow_replay", name, block_size)], args.repeats)
            for name in candidate_sets
        }
        traversal_delta = (
            traversal["capsule"]["median_microseconds_per_launch"]
            - traversal["padded_aabb"]["median_microseconds_per_launch"]
        )
        replay_delta = (
            narrow_replay["capsule"]["median_microseconds_per_launch"]
            - narrow_replay["padded_aabb"]["median_microseconds_per_launch"]
        )
        block_results.append(
            {
                "block_size": block_size,
                "launch_geometry": frozen_benchmark._launch_geometry(
                    model.edge_count,
                    block_size,
                    model.device.sm_count,
                ),
                "traversal_only": traversal,
                "narrow_replay_only": narrow_replay,
                "median_component_deltas_microseconds": {
                    "capsule_minus_aabb_traversal": traversal_delta,
                    "capsule_candidates_minus_aabb_candidates_narrow_replay": replay_delta,
                },
                "checks": block_checks[block_size],
            }
        )

    particle_world = np.ascontiguousarray(model.particle_world.numpy(), dtype=np.int32)
    edge_vertices = np.ascontiguousarray(model.edge_indices.numpy()[:, 2:4], dtype=np.int32)
    edge_owner_world = np.ascontiguousarray(particle_world[edge_vertices[:, 0]], dtype=np.int32)
    edge_groups = np.ascontiguousarray(detector.edge_groups.numpy(), dtype=np.int32)
    source_hashes_end = _source_hashes(source_paths)
    if source_hashes_end != source_hashes_start:
        raise RuntimeError("diagnostic source files changed during measurement")
    newton_provenance_end = frozen_benchmark._newton_provenance(
        expected_root=args.expected_newton_root,
        expected_git_head=args.expected_newton_git_head,
    )
    warp_provenance_end = frozen_benchmark._warp_provenance(
        expected_root=args.expected_warp_root,
        expected_git_head=args.expected_warp_git_head,
        expected_core_sha256=args.expected_warp_core_sha256,
        expected_clang_sha256=args.expected_warp_clang_sha256,
    )
    if newton_provenance_end != newton_provenance:
        raise RuntimeError("Newton provenance changed during measurement")
    if warp_provenance_end != warp_provenance:
        raise RuntimeError("Warp provenance changed during measurement")

    cuda_driver_version = wp.get_cuda_driver_version()
    cuda_toolkit_version = wp.get_cuda_toolkit_version()
    output = {
        "schema_version": "cloth-franka-ee-capsule-cost-v2",
        "harness": {
            **source_hashes_start["harness"],
            "version": _VERSION,
            "argv": argv_start,
        },
        "candidate_accounting_harness": source_hashes_start["candidate_accounting_harness"],
        "frozen_workload_producer": source_hashes_start["frozen_workload_producer"],
        "source_stability": {
            "start": source_hashes_start,
            "end": source_hashes_end,
            "exactly_equal": True,
        },
        "newton": newton_provenance,
        "warp": warp_provenance,
        "required_warp_git_head": broad_phase.PINNED_WARP_GIT_HEAD,
        "state": state_provenance,
        "replication": replication,
        "workload": frozen_benchmark._workload_provenance(workload),
        "device": {
            "alias": str(model.device),
            "name": model.device.name,
            "arch": model.device.arch,
            "sm_count": model.device.sm_count,
            "ordinal": model.device.ordinal,
            "uuid": model.device.uuid,
            "pci_bus_id": model.device.pci_bus_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_driver_version": list(cuda_driver_version) if cuda_driver_version is not None else None,
            "cuda_toolkit_version": list(cuda_toolkit_version) if cuda_toolkit_version is not None else None,
        },
        "configuration": {
            "block_sizes": args.block_sizes,
            "accounting_block_size": args.accounting_block_size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "samples": args.samples,
        },
        "measurement_semantics": {
            "traversal_only": "world-scoped public BVH query plus exact yielded-leaf count; no filters or narrow phase",
            "narrow_replay_only": (
                "native-yield-order stored candidates plus production shared-endpoint/CSR filters and "
                "current/reference closest-point tests; "
                "no BVH traversal or collision-buffer stores; includes diagnostic CSR reads and counters"
            ),
            "interpretation": (
                "traversal and replay are separate controlled microbenchmarks. The replay delta measures the cost "
                "difference between the two stored candidate streams under identical diagnostic instrumentation; "
                "the traversal delta combines query-engine cost, geometric node pruning, and leaf emission"
            ),
            "component_deltas_are_additive": False,
            "non_additivity_reason": (
                "separate graphs omit production fusion/order effects, while replay adds CSR loads "
                "and diagnostic stores"
            ),
        },
        "native_node_visits": {
            "available": False,
            "reason": "Warp's public BVH query API does not expose internal node-visit counters",
        },
        "candidates": {
            "padded_aabb": {
                "summary": broad_phase._candidate_summary(
                    candidate_sets["padded_aabb"],
                    edge_owner_world,
                    edge_groups,
                    model.world_count,
                ),
                "accounting_payload": _candidate_payload_evidence(candidate_payloads["padded_aabb"]),
            },
            "capsule": {
                "summary": broad_phase._candidate_summary(
                    candidate_sets["capsule"],
                    edge_owner_world,
                    edge_groups,
                    model.world_count,
                ),
                "accounting_payload": _candidate_payload_evidence(candidate_payloads["capsule"]),
            },
            "comparison": candidate_comparison,
        },
        "narrow_phase_funnel": {
            **replay_summaries,
            "avoided_by_capsule_candidates": avoided_work,
            "accepted_output_equality": cross_query_output_equality,
        },
        "production_detector_oracle": {
            "counts_sha256": broad_phase._hash_array(production_counts),
            "minimum_distances_sha256": broad_phase._hash_array(production_minimum_distances),
            "accepted_pairs_sha256": broad_phase._hash_array(production_pairs),
            "accepted_pair_count": len(production_pairs),
            "overflow_row_count": len(overflow_rows),
            "all_replays_exact": all(
                checks[name]["replay_exactly_equal_to_production_detector"]["all"]
                for checks in block_checks.values()
                for name in candidate_sets
            ),
        },
        "timing_schedule": timing_schedule,
        "results": block_results,
    }
    serialized = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        _write_json_exclusive(args.output, serialized)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
