#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Benchmark cloth Franka VT and EE blocks on replicated frozen states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry import kernels as geometry_kernels
from newton._src.geometry.kernels import (
    edge_colliding_edges_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton.examples.cloth.example_cloth_franka import Example
from newton.viewer import ViewerNull


@dataclass(frozen=True)
class DetectorSnapshot:
    """Hold exact detector outputs copied to the host."""

    counts: np.ndarray
    active_pairs: np.ndarray
    owner_min_distances: np.ndarray
    other_min_distances: np.ndarray | None
    resize_flags: np.ndarray


@dataclass(frozen=True)
class DetectorWorkload:
    """Hold the arrays and settings needed by the raw traversal launches."""

    model: newton.Model
    detector: TriMeshCollisionDetector
    self_contact_margin: float
    rest_shape_exclusion_radius: float
    rest_positions: wp.array[wp.vec3]


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare shape, dtype, and every output bit, including NaN payloads."""
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _active_pairs(buffer, offsets, counts, capacities) -> np.ndarray:
    """Return stored directed pairs in canonical order, excluding stale tails."""
    buffer_np = buffer.numpy()
    offsets_np = offsets.numpy()
    counts_np = counts.numpy()
    capacities_np = capacities.numpy()
    pair_rows: list[np.ndarray] = []
    for row, (offset, count, capacity) in enumerate(zip(offsets_np[:-1], counts_np, capacities_np, strict=True)):
        stored_count = min(int(count), int(capacity))
        if stored_count == 0:
            continue
        start = 2 * int(offset)
        pairs = buffer_np[start : start + 2 * stored_count].reshape((-1, 2)).copy()
        if not np.all(pairs[:, 0] == row):
            raise RuntimeError(f"collision buffer owner mismatch in row {row}")
        pair_rows.append(pairs)

    if not pair_rows:
        return np.empty((0, 2), dtype=np.int32)
    result = np.concatenate(pair_rows)
    return result[np.lexsort((result[:, 1], result[:, 0]))]


def _snapshot_vertex_triangle(detector) -> DetectorSnapshot:
    info = detector.collision_info
    return DetectorSnapshot(
        counts=info.vertex_colliding_triangles_count.numpy(),
        active_pairs=_active_pairs(
            info.vertex_colliding_triangles,
            info.vertex_colliding_triangles_offsets,
            info.vertex_colliding_triangles_count,
            info.vertex_colliding_triangles_buffer_sizes,
        ),
        owner_min_distances=info.vertex_colliding_triangles_min_dist.numpy(),
        other_min_distances=None,
        resize_flags=detector.resize_flags.numpy(),
    )


def _snapshot_edge_edge(detector) -> DetectorSnapshot:
    info = detector.collision_info
    return DetectorSnapshot(
        counts=info.edge_colliding_edges_count.numpy(),
        active_pairs=_active_pairs(
            info.edge_colliding_edges,
            info.edge_colliding_edges_offsets,
            info.edge_colliding_edges_count,
            info.edge_colliding_edges_buffer_sizes,
        ),
        owner_min_distances=info.edge_colliding_edges_min_dist.numpy(),
        other_min_distances=None,
        resize_flags=detector.resize_flags.numpy(),
    )


def _snapshot_summary(snapshot: DetectorSnapshot, capacities: np.ndarray) -> dict[str, object]:
    summary: dict[str, object] = {
        "raw_count": int(snapshot.counts.sum()),
        "stored_count": int(np.minimum(snapshot.counts, capacities).sum()),
        "nonempty_rows": int(np.count_nonzero(snapshot.counts)),
        "overflow_rows": int(np.count_nonzero(snapshot.counts > capacities)),
        "counts_sha256": _array_hash(snapshot.counts),
        "active_pairs_sha256": _array_hash(snapshot.active_pairs),
        "owner_min_distances_sha256": _array_hash(snapshot.owner_min_distances),
        "resize_flags": snapshot.resize_flags.tolist(),
        "resize_flags_sha256": _array_hash(snapshot.resize_flags),
    }
    if snapshot.other_min_distances is not None:
        summary["other_min_distances_sha256"] = _array_hash(snapshot.other_min_distances)
    return summary


def _snapshot_equality(reference: DetectorSnapshot, candidate: DetectorSnapshot) -> dict[str, bool]:
    equality = {
        "counts": _same_array(reference.counts, candidate.counts),
        "active_pairs": _same_array(reference.active_pairs, candidate.active_pairs),
        "owner_min_distances": _same_array(reference.owner_min_distances, candidate.owner_min_distances),
        "resize_flags": _same_array(reference.resize_flags, candidate.resize_flags),
    }
    if reference.other_min_distances is None or candidate.other_min_distances is None:
        equality["other_min_distances"] = (
            reference.other_min_distances is None and candidate.other_min_distances is None
        )
    else:
        equality["other_min_distances"] = _same_array(reference.other_min_distances, candidate.other_min_distances)
    equality["all"] = all(equality.values())
    return equality


def _world_copy_equality(
    snapshot: DetectorSnapshot,
    world_copies: int,
    owner_count_per_world: int,
    target_count_per_world: int,
) -> dict[str, object]:
    """Verify every world is an index-shifted copy of world zero."""
    expected_owner_count = world_copies * owner_count_per_world
    shapes = {
        "counts": snapshot.counts.shape == (expected_owner_count,),
        "owner_min_distances": snapshot.owner_min_distances.shape == (expected_owner_count,),
        "active_pairs": snapshot.active_pairs.ndim == 2 and snapshot.active_pairs.shape[1:] == (2,),
    }

    counts_equal = False
    owner_min_distances_equal = False
    if shapes["counts"]:
        counts_by_world = snapshot.counts.reshape((world_copies, owner_count_per_world))
        counts_equal = all(_same_array(counts_by_world[0], row) for row in counts_by_world[1:])
        raw_counts_per_world = [int(row.sum()) for row in counts_by_world]
    else:
        raw_counts_per_world = []
    if shapes["owner_min_distances"]:
        distances_by_world = snapshot.owner_min_distances.reshape((world_copies, owner_count_per_world))
        owner_min_distances_equal = all(_same_array(distances_by_world[0], row) for row in distances_by_world[1:])

    pairs_equal = shapes["active_pairs"]
    targets_stay_in_world = shapes["active_pairs"]
    active_pair_counts_per_world: list[int] = []
    normalized_reference: np.ndarray | None = None
    if shapes["active_pairs"]:
        for world in range(world_copies):
            owner_start = world * owner_count_per_world
            owner_end = owner_start + owner_count_per_world
            target_start = world * target_count_per_world
            target_end = target_start + target_count_per_world
            owner_mask = (snapshot.active_pairs[:, 0] >= owner_start) & (snapshot.active_pairs[:, 0] < owner_end)
            world_pairs = snapshot.active_pairs[owner_mask].copy()
            active_pair_counts_per_world.append(int(world_pairs.shape[0]))
            if world_pairs.size:
                targets_stay_in_world = targets_stay_in_world and bool(
                    np.all((world_pairs[:, 1] >= target_start) & (world_pairs[:, 1] < target_end))
                )
                world_pairs[:, 0] -= owner_start
                world_pairs[:, 1] -= target_start
            if normalized_reference is None:
                normalized_reference = world_pairs
            else:
                pairs_equal = pairs_equal and _same_array(normalized_reference, world_pairs)

    equality: dict[str, object] = {
        "shapes": shapes,
        "counts": counts_equal,
        "active_pairs": pairs_equal,
        "owner_min_distances": owner_min_distances_equal,
        "targets_stay_in_world": targets_stay_in_world,
        "raw_counts_per_world": raw_counts_per_world,
        "active_pair_counts_per_world": active_pair_counts_per_world,
    }
    equality["all"] = all(shapes.values()) and all(
        bool(equality[key]) for key in ("counts", "active_pairs", "owner_min_distances", "targets_stay_in_world")
    )
    return equality


def _build_workload(
    example: Example,
    source_frozen_positions: wp.array[wp.vec3],
    world_copies: int,
) -> tuple[DetectorWorkload, dict[str, object]]:
    """Create a native or true multi-world detector workload from one state."""
    source_model = example.model
    source_solver = example.cloth_solver
    if source_solver is None:
        raise RuntimeError("cloth Franka did not create its VBD solver")
    source_detector = source_solver.trimesh_collision_detector

    if world_copies == 1:
        source_detector.rebuild(source_frozen_positions)
        return (
            DetectorWorkload(
                model=source_model,
                detector=source_detector,
                self_contact_margin=source_solver.particle_self_contact_margin,
                rest_shape_exclusion_radius=source_solver.particle_rest_shape_contact_exclusion_radius,
                rest_positions=source_solver.particle_q_rest,
            ),
            {
                "mode": "native_single_world",
                "world_copies": 1,
                "source_particle_count": source_model.particle_count,
                "source_triangle_count": source_model.tri_count,
                "source_edge_count": source_model.edge_count,
                "particle_world_layout_exact": True,
                "rest_positions_tiled_exactly": True,
            },
        )

    source_positions_np = source_frozen_positions.numpy()
    source_rest_np = source_solver.particle_q_rest.numpy()
    replicated_builder = newton.ModelBuilder(gravity=(0.0, 0.0, -981.0))
    replicated_builder.replicate(example.scene, world_copies, spacing=(0.0, 0.0, 0.0))
    model = replicated_builder.finalize(device=source_model.device, requires_grad=False)

    expected_counts = (
        world_copies * source_model.particle_count,
        world_copies * source_model.tri_count,
        world_copies * source_model.edge_count,
    )
    actual_counts = (model.particle_count, model.tri_count, model.edge_count)
    if actual_counts != expected_counts:
        raise RuntimeError(f"replicated model counts {actual_counts} do not match expected {expected_counts}")

    expected_particle_world = np.repeat(np.arange(world_copies, dtype=np.int32), source_model.particle_count)
    particle_world_layout_exact = np.array_equal(model.particle_world.numpy(), expected_particle_world)
    if not particle_world_layout_exact:
        raise RuntimeError("replicated particle worlds are not contiguous independent copies")

    replicated_rest_np = model.particle_q.numpy()
    expected_rest_np = np.tile(source_rest_np, (world_copies, 1))
    rest_positions_tiled_exactly = _same_array(replicated_rest_np, expected_rest_np)
    if not rest_positions_tiled_exactly:
        raise RuntimeError("replicated rest positions differ from exact source copies")

    replicated_positions = wp.array(
        np.tile(source_positions_np, (world_copies, 1)),
        dtype=wp.vec3,
        device=model.device,
    )
    vertex_capacities = source_detector.vertex_colliding_triangles_buffer_sizes.numpy()
    edge_capacities = source_detector.edge_colliding_edges_buffer_sizes.numpy()
    if not np.all(vertex_capacities == vertex_capacities[0]):
        raise RuntimeError("source VT collision capacities are not uniform")
    if not np.all(edge_capacities == edge_capacities[0]):
        raise RuntimeError("source EE collision capacities are not uniform")

    detector = TriMeshCollisionDetector(
        model,
        vertex_positions=replicated_positions,
        vertex_collision_buffer_pre_alloc=int(vertex_capacities[0]),
        edge_collision_buffer_pre_alloc=int(edge_capacities[0]),
        edge_edge_parallel_epsilon=source_detector.edge_edge_parallel_epsilon,
        topological_contact_filter_threshold=source_solver.particle_topological_contact_filter_threshold,
    )
    detector.rebuild(replicated_positions)
    return (
        DetectorWorkload(
            model=model,
            detector=detector,
            self_contact_margin=source_solver.particle_self_contact_margin,
            rest_shape_exclusion_radius=source_solver.particle_rest_shape_contact_exclusion_radius,
            rest_positions=model.particle_q,
        ),
        {
            "mode": "replicated_independent_worlds",
            "world_copies": world_copies,
            "source_particle_count": source_model.particle_count,
            "source_triangle_count": source_model.tri_count,
            "source_edge_count": source_model.edge_count,
            "particle_world_layout_exact": particle_world_layout_exact,
            "rest_positions_tiled_exactly": rest_positions_tiled_exactly,
        },
    )


def _launch_vertex_triangle(workload: DetectorWorkload, block_size: int) -> None:
    """Launch only the VT traversal kernel, excluding block-independent setup."""
    detector = workload.detector
    model = workload.model
    wp.launch(
        kernel=vertex_triangle_collision_detection_kernel,
        inputs=[
            workload.self_contact_margin,
            workload.rest_shape_exclusion_radius,
            detector.bvh_tris.id,
            detector.bvh_tris_group_roots,
            detector.vertex_positions,
            model.tri_indices,
            model.particle_world,
            model.world_count,
            detector.vertex_colliding_triangles_offsets,
            detector.vertex_colliding_triangles_buffer_sizes,
            detector.triangle_colliding_vertices_offsets,
            detector.triangle_colliding_vertices_buffer_sizes,
            detector.vertex_triangle_filtering_list,
            detector.vertex_triangle_filtering_list_offsets,
            workload.rest_positions,
        ],
        outputs=[
            detector.vertex_colliding_triangles,
            detector.vertex_colliding_triangles_count,
            detector.vertex_colliding_triangles_min_dist,
            detector.triangle_colliding_vertices,
            detector.triangle_colliding_vertices_count,
            detector.triangle_colliding_vertices_min_dist,
            detector.resize_flags,
        ],
        dim=model.particle_count,
        device=model.device,
        block_dim=block_size,
    )


def _launch_edge_edge(workload: DetectorWorkload, block_size: int) -> None:
    """Launch only the EE traversal kernel, excluding block-independent setup."""
    detector = workload.detector
    model = workload.model
    wp.launch(
        kernel=edge_colliding_edges_detection_kernel,
        inputs=[
            workload.self_contact_margin,
            workload.rest_shape_exclusion_radius,
            detector.bvh_edges.id,
            detector.bvh_edges_group_roots,
            detector.vertex_positions,
            model.edge_indices,
            model.particle_world,
            model.world_count,
            detector.edge_colliding_edges_offsets,
            detector.edge_colliding_edges_buffer_sizes,
            detector.edge_edge_parallel_epsilon,
            detector.edge_filtering_list,
            detector.edge_filtering_list_offsets,
            workload.rest_positions,
        ],
        outputs=[
            detector.edge_colliding_edges,
            detector.edge_colliding_edges_count,
            detector.edge_colliding_edges_min_dist,
            detector.resize_flags,
        ],
        dim=model.edge_count,
        device=model.device,
        block_dim=block_size,
    )


def _capture_detector_graph(workload: DetectorWorkload, kind: str, block_size: int):
    wp.load_module(geometry_kernels, device=workload.model.device, block_dim=block_size)
    launch = _launch_vertex_triangle if kind == "vertex_triangle" else _launch_edge_edge
    with wp.ScopedCapture() as capture:
        launch(workload, block_size)
    return capture.graph


def _time_graph(graph, repeats: int, device) -> float:
    start = wp.Event(device=device, enable_timing=True)
    end = wp.Event(device=device, enable_timing=True)
    wp.record_event(start)
    for _ in range(repeats):
        wp.capture_launch(graph)
    wp.record_event(end)
    return float(wp.get_event_elapsed_time(start, end))


def _launch_geometry(logical_threads: int, block_size: int, sm_count: int) -> dict[str, int | float]:
    blocks = math.ceil(logical_threads / block_size)
    warps_per_block = math.ceil(block_size / 32)
    physical_warps = blocks * warps_per_block
    return {
        "logical_threads": logical_threads,
        "blocks": blocks,
        "warps_per_block": warps_per_block,
        "physical_warps": physical_warps,
        "physical_warps_per_sm": physical_warps / sm_count,
        "full_block_active_lane_fraction": block_size / (32 * warps_per_block),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[4, 8, 12, 16, 24, 32, 64])
    parser.add_argument("--state-frames", type=int, default=30)
    parser.add_argument("--state-input", type=Path)
    parser.add_argument("--world-copies", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.block_sizes or any(size <= 0 for size in args.block_sizes):
        parser.error("--block-sizes must contain positive integers")
    if len(set(args.block_sizes)) != len(args.block_sizes):
        parser.error("--block-sizes must not contain duplicates")
    if args.state_frames < 0:
        parser.error("--state-frames must be non-negative")
    if args.world_copies <= 0:
        parser.error("--world-copies must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.samples < 5:
        parser.error("--samples must be at least 5")
    return args


def main() -> None:
    args = _parse_args()
    viewer = ViewerNull(num_frames=max(args.state_frames, 1))
    example_args = newton.examples.default_args() if hasattr(newton.examples, "default_args") else None
    example = Example(viewer, example_args)
    if example.cloth_solver is None:
        raise RuntimeError("cloth Franka did not create its VBD solver")

    if args.state_input is None:
        for _ in range(args.state_frames):
            example.step()
        wp.synchronize_device(example.model.device)
        frozen_positions = wp.clone(example.state_0.particle_q)
        state_provenance = {
            "kind": "advanced_example",
            "frames": args.state_frames,
        }
    else:
        state_input = args.state_input.resolve()
        with np.load(state_input, allow_pickle=False) as state:
            if "particle_q" not in state:
                raise ValueError(f"{state_input} does not contain particle_q")
            particle_q = state["particle_q"]
        expected_shape = (example.model.particle_count, 3)
        if particle_q.shape != expected_shape:
            raise ValueError(f"particle_q has shape {particle_q.shape}, expected {expected_shape}")
        frozen_positions = wp.array(particle_q, dtype=wp.vec3, device=example.model.device)
        state_provenance = {
            "kind": "npz",
            "path": str(state_input),
            "sha256": _file_hash(state_input),
        }

    source_model = example.model
    workload, replication = _build_workload(example, frozen_positions, args.world_copies)
    model = workload.model
    detector = workload.detector
    wp.synchronize_device(model.device)

    capacities = {
        "vertex_triangle": detector.collision_info.vertex_colliding_triangles_buffer_sizes.numpy(),
        "edge_edge": detector.collision_info.edge_colliding_edges_buffer_sizes.numpy(),
    }
    snapshots: dict[tuple[str, int], DetectorSnapshot] = {}
    timings: dict[tuple[str, int], list[float]] = {
        (kind, block_size): [] for block_size in args.block_sizes for kind in ("vertex_triangle", "edge_edge")
    }
    graphs: dict[tuple[str, int], object] = {}

    for block_size in args.block_sizes:
        for kind in ("vertex_triangle", "edge_edge"):
            graphs[(kind, block_size)] = _capture_detector_graph(workload, kind, block_size)

    # Warm every specialization before the balanced timed schedule begins.
    for graph in graphs.values():
        for _ in range(args.warmups):
            wp.capture_launch(graph)
    wp.synchronize_device(model.device)

    timing_schedule: list[dict[str, int | str]] = []
    for sample in range(args.samples):
        size_order = args.block_sizes if sample % 2 == 0 else list(reversed(args.block_sizes))
        for size_index, block_size in enumerate(size_order):
            kinds = ("vertex_triangle", "edge_edge")
            if (sample + size_index) % 2:
                kinds = tuple(reversed(kinds))
            for kind in kinds:
                elapsed_ms = _time_graph(graphs[(kind, block_size)], args.repeats, model.device)
                timings[(kind, block_size)].append(elapsed_ms)
                timing_schedule.append({"sample": sample, "block_size": block_size, "kind": kind})

    # Re-run once outside timing so every snapshot belongs unambiguously to its
    # own graph rather than to whichever shared output buffer ran last.
    for block_size in args.block_sizes:
        for kind, snapshot_fn in (
            ("vertex_triangle", _snapshot_vertex_triangle),
            ("edge_edge", _snapshot_edge_edge),
        ):
            detector.resize_flags.zero_()
            wp.capture_launch(graphs[(kind, block_size)])
            wp.synchronize_device(model.device)
            snapshots[(kind, block_size)] = snapshot_fn(detector)

    reference_size = args.block_sizes[0]
    all_outputs_equal = True
    all_world_copies_equal = True
    results: list[dict[str, object]] = []
    for block_size in args.block_sizes:
        kind_results: dict[str, object] = {}
        for kind, logical_threads, owner_count_per_world, target_count_per_world in (
            (
                "vertex_triangle",
                model.particle_count,
                source_model.particle_count,
                source_model.tri_count,
            ),
            ("edge_edge", model.edge_count, source_model.edge_count, source_model.edge_count),
        ):
            snapshot = snapshots[(kind, block_size)]
            equality = _snapshot_equality(snapshots[(kind, reference_size)], snapshot)
            all_outputs_equal = all_outputs_equal and equality["all"]
            world_copy_equality = _world_copy_equality(
                snapshot,
                args.world_copies,
                owner_count_per_world,
                target_count_per_world,
            )
            all_world_copies_equal = all_world_copies_equal and bool(world_copy_equality["all"])
            elapsed_samples_ms = timings[(kind, block_size)]
            microseconds_per_launch = [1000.0 * elapsed_ms / args.repeats for elapsed_ms in elapsed_samples_ms]
            kind_results[kind] = {
                "sample_elapsed_ms": elapsed_samples_ms,
                "sample_microseconds_per_launch": microseconds_per_launch,
                "median_microseconds_per_launch": float(np.median(microseconds_per_launch)),
                "minimum_microseconds_per_launch": min(microseconds_per_launch),
                "launch_geometry": _launch_geometry(logical_threads, block_size, model.device.sm_count),
                "outputs": _snapshot_summary(snapshot, capacities[kind]),
                "exactly_equal_to_reference": equality,
                "world_copies_exact": world_copy_equality,
            }
        results.append({"block_size": block_size, **kind_results})

    source_root = Path(newton.__file__).resolve().parents[1]
    result = {
        "source_root": str(source_root),
        "git_head": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "harness": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "device": str(model.device),
        "sm_count": model.device.sm_count,
        "state": state_provenance,
        "frozen_particle_q_sha256": _array_hash(frozen_positions.numpy()),
        "workload_particle_q_sha256": _array_hash(detector.vertex_positions.numpy()),
        "replication": replication,
        "particle_count": model.particle_count,
        "triangle_count": model.tri_count,
        "edge_count": model.edge_count,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "samples": args.samples,
        "timing_schedule": timing_schedule,
        "reference_block_size": reference_size,
        "all_outputs_exactly_equal": all_outputs_equal,
        "all_world_copies_exact": all_world_copies_equal,
        "results": results,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")

    if not all_outputs_equal:
        raise RuntimeError("detector outputs differed between block sizes")
    if not all_world_copies_equal:
        raise RuntimeError("detector outputs differed between replicated worlds")


if __name__ == "__main__":
    main()
