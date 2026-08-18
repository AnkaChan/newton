# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare exact cloth self-contact broad-phase candidate sets.

This profiling-only tool enumerates the candidates yielded by the legacy AABB
queries and the tighter radius queries on one frozen cloth Franka workload. It
does not run topology filters or narrow-phase contact tests; final detector
output equivalence must be checked separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cloth.example_cloth_franka import Example
from newton.viewer import ViewerNull

if __package__:
    from . import benchmark_cloth_franka_detector_blocks as frozen_benchmark
else:
    import benchmark_cloth_franka_detector_blocks as frozen_benchmark


PINNED_WARP_GIT_HEAD = "7c66e2f604248ce7bc32ebd9b6c77c05f09550d2"
_RADIUS_QUERIES_AVAILABLE = tuple(int(component) for component in wp.config.version.split(".")[:2]) >= (1, 17)


@wp.func
def _resolve_query_root(
    owner_world: int,
    world_count: int,
    group_bvh_roots: wp.array[wp.int32],
    query_pass: int,
):
    """Resolve the production own-world/global-root query passes."""
    run_query = False
    query_root = -1
    if owner_world < 0:
        if query_pass == 0:
            run_query = True
    elif query_pass == 0:
        query_root = group_bvh_roots[owner_world]
        if query_root >= 0:
            run_query = True
    else:
        query_root = group_bvh_roots[world_count]
        if query_root >= 0:
            run_query = True
    return run_query, query_root


@wp.kernel
def _count_or_store_vertex_triangle_aabb_candidates(
    triangle_bvh_id: wp.uint64,
    triangle_bvh_group_roots: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    world_count: int,
    radius: float,
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    targets: wp.array[wp.int32],
):
    vertex_index = wp.tid()
    owner_world = particle_world[vertex_index]
    center = positions[vertex_index]
    extent = wp.vec3(radius)
    lower = center - extent
    upper = center + extent
    candidate_count = wp.int32(0)

    for query_pass in range(2):
        run_query, query_root = _resolve_query_root(owner_world, world_count, triangle_bvh_group_roots, query_pass)
        if run_query:
            query = wp.bvh_query_aabb(triangle_bvh_id, lower, upper, query_root)
            triangle_index = wp.int32(0)
            while wp.bvh_query_next(query, triangle_index):
                if targets:
                    targets[offsets[vertex_index] + candidate_count] = triangle_index
                candidate_count += 1

    counts[vertex_index] = candidate_count


@wp.kernel
def _count_or_store_edge_edge_aabb_candidates(
    edge_bvh_id: wp.uint64,
    edge_bvh_group_roots: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    world_count: int,
    radius: float,
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    targets: wp.array[wp.int32],
):
    edge_index = wp.tid()
    v0 = edge_indices[edge_index, 2]
    v1 = edge_indices[edge_index, 3]
    owner_world = particle_world[v0]
    start = positions[v0]
    end = positions[v1]
    extent = wp.vec3(radius)
    lower = wp.min(start, end) - extent
    upper = wp.max(start, end) + extent
    candidate_count = wp.int32(0)

    for query_pass in range(2):
        run_query, query_root = _resolve_query_root(owner_world, world_count, edge_bvh_group_roots, query_pass)
        if run_query:
            query = wp.bvh_query_aabb(edge_bvh_id, lower, upper, query_root)
            candidate_edge_index = wp.int32(0)
            while wp.bvh_query_next(query, candidate_edge_index):
                if targets:
                    targets[offsets[edge_index] + candidate_count] = candidate_edge_index
                candidate_count += 1

    counts[edge_index] = candidate_count


if _RADIUS_QUERIES_AVAILABLE:

    @wp.kernel
    def _count_or_store_vertex_triangle_sphere_candidates(
        triangle_bvh_id: wp.uint64,
        triangle_bvh_group_roots: wp.array[wp.int32],
        positions: wp.array[wp.vec3],
        particle_world: wp.array[wp.int32],
        world_count: int,
        radius: float,
        offsets: wp.array[wp.int32],
        counts: wp.array[wp.int32],
        targets: wp.array[wp.int32],
    ):
        vertex_index = wp.tid()
        owner_world = particle_world[vertex_index]
        center = positions[vertex_index]
        candidate_count = wp.int32(0)

        for query_pass in range(2):
            run_query, query_root = _resolve_query_root(owner_world, world_count, triangle_bvh_group_roots, query_pass)
            if run_query:
                query = wp.bvh_query_sphere(triangle_bvh_id, center, radius, query_root)
                triangle_index = wp.int32(0)
                while wp.bvh_query_next(query, triangle_index):
                    if targets:
                        targets[offsets[vertex_index] + candidate_count] = triangle_index
                    candidate_count += 1

        counts[vertex_index] = candidate_count

    @wp.kernel
    def _count_or_store_edge_edge_capsule_candidates(
        edge_bvh_id: wp.uint64,
        edge_bvh_group_roots: wp.array[wp.int32],
        positions: wp.array[wp.vec3],
        edge_indices: wp.array2d[wp.int32],
        particle_world: wp.array[wp.int32],
        world_count: int,
        radius: float,
        offsets: wp.array[wp.int32],
        counts: wp.array[wp.int32],
        targets: wp.array[wp.int32],
    ):
        edge_index = wp.tid()
        v0 = edge_indices[edge_index, 2]
        v1 = edge_indices[edge_index, 3]
        owner_world = particle_world[v0]
        start = positions[v0]
        direction = positions[v1] - start
        max_dist = 1.0
        if wp.length_sq(direction) == 0.0:
            direction = wp.vec3(1.0, 0.0, 0.0)
            max_dist = 0.0
        candidate_count = wp.int32(0)

        for query_pass in range(2):
            run_query, query_root = _resolve_query_root(owner_world, world_count, edge_bvh_group_roots, query_pass)
            if run_query:
                query = wp.bvh_query_capsule(edge_bvh_id, start, direction, radius, query_root)
                candidate_edge_index = wp.int32(0)
                while wp.bvh_query_next(query, candidate_edge_index, max_dist):
                    if targets:
                        targets[offsets[edge_index] + candidate_count] = candidate_edge_index
                    candidate_count += 1

        counts[edge_index] = candidate_count

else:
    _count_or_store_vertex_triangle_sphere_candidates = None
    _count_or_store_edge_edge_capsule_candidates = None


@dataclass(frozen=True)
class CandidateSet:
    """One canonical per-owner candidate set."""

    counts: np.ndarray
    pairs: np.ndarray


def _hash_array(array: np.ndarray) -> str:
    """Hash one C-contiguous array without serializing it into JSON."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _count_distribution(counts: np.ndarray) -> list[dict[str, int]]:
    """Return an exact histogram of candidate counts per owner."""
    values, frequencies = np.unique(counts, return_counts=True)
    return [
        {"candidate_count": int(value), "owner_count": int(frequency)}
        for value, frequency in zip(values, frequencies, strict=True)
    ]


def _canonical_pairs(counts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Canonicalize flattened query results into unique ``(owner, target)`` pairs."""
    counts = np.asarray(counts, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int32)
    if counts.ndim != 1 or targets.ndim != 1:
        raise ValueError("counts and targets must be one-dimensional")
    if np.any(counts < 0):
        raise ValueError("candidate counts cannot be negative")
    if int(counts.sum(dtype=np.int64)) != len(targets):
        raise ValueError("flattened candidate length does not match the per-owner counts")

    owners = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    pairs = np.empty((len(targets), 2), dtype=np.int32)
    pairs[:, 0] = owners
    pairs[:, 1] = targets
    if len(pairs):
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = np.ascontiguousarray(pairs[order])
        duplicate = np.all(pairs[1:] == pairs[:-1], axis=1)
        if np.any(duplicate):
            first = pairs[1:][duplicate][0]
            raise RuntimeError(f"duplicate broad-phase candidate pair: ({first[0]}, {first[1]})")
    return pairs


def _pair_keys(pairs: np.ndarray) -> np.ndarray:
    """Encode canonical nonnegative int32 pairs as ordered uint64 keys."""
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != np.int32:
        raise ValueError("pairs must have shape (N, 2) and dtype int32")
    if np.any(pairs < 0):
        raise ValueError("candidate pair indices cannot be negative")
    keys = (pairs[:, 0].astype(np.uint64) << np.uint64(32)) | pairs[:, 1].astype(np.uint32).astype(np.uint64)
    if len(keys) > 1 and np.any(keys[1:] <= keys[:-1]):
        raise RuntimeError("candidate pairs are not strictly canonical")
    return keys


def _pairs_from_keys(keys: np.ndarray) -> np.ndarray:
    """Decode ordered uint64 pair keys into canonical int32 pairs."""
    pairs = np.empty((len(keys), 2), dtype=np.int32)
    pairs[:, 0] = (keys >> np.uint64(32)).astype(np.int32)
    pairs[:, 1] = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32).astype(np.int32)
    return pairs


def _scope_violation_count(
    pairs: np.ndarray,
    owner_worlds: np.ndarray,
    target_groups: np.ndarray,
    world_count: int,
) -> int:
    """Count candidates outside the production own-world/global group scope."""
    if not len(pairs):
        return 0
    if int(pairs[:, 0].max()) >= len(owner_worlds) or int(pairs[:, 1].max()) >= len(target_groups):
        raise RuntimeError("candidate pair index exceeds its owner or target array")
    owner_group = owner_worlds[pairs[:, 0]]
    target_group = target_groups[pairs[:, 1]]
    valid = (owner_group < 0) | (target_group == owner_group) | (target_group == world_count)
    return int(np.count_nonzero(~valid))


def _candidate_summary(
    candidate_set: CandidateSet,
    owner_worlds: np.ndarray,
    target_groups: np.ndarray,
    world_count: int,
) -> dict[str, object]:
    """Summarize an exact candidate set without emitting its full pair payload."""
    counts = candidate_set.counts
    pairs = candidate_set.pairs
    violations = _scope_violation_count(pairs, owner_worlds, target_groups, world_count)
    if violations:
        raise RuntimeError(f"found {violations} candidates outside the expected world/group scope")
    return {
        "owner_count": int(len(counts)),
        "total_candidates": int(len(pairs)),
        "nonempty_owner_count": int(np.count_nonzero(counts)),
        "minimum_per_owner": int(counts.min()) if len(counts) else 0,
        "maximum_per_owner": int(counts.max()) if len(counts) else 0,
        "mean_per_owner": float(counts.mean()) if len(counts) else 0.0,
        "counts_dtype": counts.dtype.str,
        "counts_shape": list(counts.shape),
        "counts_sha256": _hash_array(counts),
        "pairs_dtype": pairs.dtype.str,
        "pairs_shape": list(pairs.shape),
        "pairs_sha256": _hash_array(pairs),
        "count_distribution": _count_distribution(counts),
        "world_scope_violation_count": violations,
    }


def _compare_candidate_sets(old: CandidateSet, new: CandidateSet) -> dict[str, object]:
    """Compare canonical sets and require the radius-query set to be a subset."""
    if len(old.counts) != len(new.counts):
        raise RuntimeError("candidate-set owner counts differ")
    old_keys = _pair_keys(old.pairs)
    new_keys = _pair_keys(new.pairs)
    old_only_keys = np.setdiff1d(old_keys, new_keys, assume_unique=True)
    new_only_keys = np.setdiff1d(new_keys, old_keys, assume_unique=True)
    old_only_pairs = _pairs_from_keys(old_only_keys)
    new_only_pairs = _pairs_from_keys(new_only_keys)

    owner_count = len(old.counts)
    old_only_per_owner = np.bincount(old_only_pairs[:, 0], minlength=owner_count).astype(np.int64, copy=False)
    new_only_per_owner = np.bincount(new_only_pairs[:, 0], minlength=owner_count).astype(np.int64, copy=False)
    expected_delta = old.counts.astype(np.int64) - new.counts.astype(np.int64)
    if not np.array_equal(expected_delta, old_only_per_owner - new_only_per_owner):
        raise RuntimeError("set-difference accounting does not match per-owner query counts")

    result = {
        "old_only_count": int(len(old_only_pairs)),
        "new_only_count": int(len(new_only_pairs)),
        "old_only_pairs_sha256": _hash_array(old_only_pairs),
        "new_only_pairs_sha256": _hash_array(new_only_pairs),
        "old_only_count_distribution": _count_distribution(old_only_per_owner),
        "owners_with_fewer_candidates": int(np.count_nonzero(new.counts < old.counts)),
        "owners_with_equal_candidates": int(np.count_nonzero(new.counts == old.counts)),
        "owners_with_more_candidates": int(np.count_nonzero(new.counts > old.counts)),
        "new_to_old_candidate_ratio": float(len(new.pairs) / len(old.pairs)) if len(old.pairs) else 1.0,
        "candidate_reduction_fraction": float(len(old_only_pairs) / len(old.pairs)) if len(old.pairs) else 0.0,
        "new_is_subset_of_old": len(new_only_pairs) == 0,
    }
    if len(new_only_pairs):
        first = new_only_pairs[0]
        raise RuntimeError(
            f"radius query produced {len(new_only_pairs)} candidates absent from the AABB query; "
            f"first pair=({first[0]}, {first[1]})"
        )
    return result


def _launch_candidate_pass(
    workload: frozen_benchmark.DetectorWorkload,
    primitive: str,
    kernel,
    offsets: wp.array[wp.int32] | None,
    counts: wp.array[wp.int32],
    targets: wp.array[wp.int32] | None,
    block_size: int,
) -> None:
    """Launch one count or exact-store pass for one query implementation."""
    detector = workload.detector
    model = workload.model
    if primitive == "vertex_triangle":
        inputs = [
            detector.bvh_tris.id,
            detector.bvh_tris_group_roots,
            detector.vertex_positions,
            model.particle_world,
            model.world_count,
            workload.self_contact_margin,
            offsets,
        ]
        dim = model.particle_count
    elif primitive == "edge_edge":
        inputs = [
            detector.bvh_edges.id,
            detector.bvh_edges_group_roots,
            detector.vertex_positions,
            model.edge_indices,
            model.particle_world,
            model.world_count,
            workload.self_contact_margin,
            offsets,
        ]
        dim = model.edge_count
    else:
        raise ValueError(f"unknown primitive: {primitive}")
    wp.launch(
        kernel=kernel,
        dim=dim,
        inputs=inputs,
        outputs=[counts, targets],
        device=model.device,
        block_dim=block_size,
    )


def _enumerate_candidates(
    workload: frozen_benchmark.DetectorWorkload,
    primitive: str,
    kernel,
    block_size: int,
) -> CandidateSet:
    """Enumerate exact candidates with separate count and prefix/store passes."""
    if kernel is None:
        raise RuntimeError("the imported Warp build does not provide the required radius-query APIs")
    owner_count = workload.model.particle_count if primitive == "vertex_triangle" else workload.model.edge_count
    device = workload.model.device

    count_buffer = wp.empty(owner_count, dtype=wp.int32, device=device)
    _launch_candidate_pass(workload, primitive, kernel, None, count_buffer, None, block_size)
    counts = np.ascontiguousarray(count_buffer.numpy(), dtype=np.int32)
    if np.any(counts < 0):
        raise RuntimeError("candidate count overflowed int32")

    offsets_i64 = np.empty(owner_count + 1, dtype=np.int64)
    offsets_i64[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets_i64[1:])
    total = int(offsets_i64[-1])
    if total > np.iinfo(np.int32).max:
        raise RuntimeError(f"candidate payload exceeds int32 addressing: {total}")
    offsets = wp.array(offsets_i64.astype(np.int32), dtype=wp.int32, device=device)
    store_count_buffer = wp.empty(owner_count, dtype=wp.int32, device=device)
    target_buffer = wp.empty(total, dtype=wp.int32, device=device)
    _launch_candidate_pass(
        workload,
        primitive,
        kernel,
        offsets,
        store_count_buffer,
        target_buffer,
        block_size,
    )
    store_counts = np.ascontiguousarray(store_count_buffer.numpy(), dtype=np.int32)
    if not np.array_equal(counts, store_counts):
        mismatch = int(np.flatnonzero(counts != store_counts)[0])
        raise RuntimeError(
            f"candidate count changed between passes for owner {mismatch}: "
            f"{counts[mismatch]} != {store_counts[mismatch]}"
        )
    targets = np.ascontiguousarray(target_buffer.numpy(), dtype=np.int32)
    return CandidateSet(counts=counts, pairs=_canonical_pairs(counts, targets))


def _load_workload(
    state_input: Path,
    world_copies: int,
) -> tuple[frozen_benchmark.DetectorWorkload, dict[str, object], dict[str, object]]:
    """Create the producer's exact frozen detector workload from an NPZ state."""
    viewer = ViewerNull(num_frames=1)
    capture_default = Example.capture
    try:
        Example.capture = lambda _example: None
        example = Example(viewer, newton.examples.default_args())
    finally:
        Example.capture = capture_default
    if example.cloth_solver is None:
        raise RuntimeError("example_cloth_franka did not construct its cloth solver")

    with np.load(state_input, allow_pickle=False) as state:
        if "particle_q" not in state:
            raise ValueError(f"{state_input} does not contain particle_q")
        particle_q = np.ascontiguousarray(state["particle_q"])
    expected_shape = (example.model.particle_count, 3)
    if particle_q.shape != expected_shape:
        raise ValueError(f"particle_q shape {particle_q.shape} does not match expected {expected_shape}")
    if particle_q.dtype != np.float32:
        raise ValueError(f"particle_q dtype must be float32, got {particle_q.dtype}")
    if not np.all(np.isfinite(particle_q)):
        raise ValueError("particle_q contains non-finite values")

    frozen_positions = wp.array(particle_q, dtype=wp.vec3, device=example.model.device)
    workload, replication = frozen_benchmark._build_workload(example, frozen_positions, world_copies)
    state_provenance = {
        "path": str(state_input.resolve()),
        "sha256": frozen_benchmark._file_hash(state_input),
        "particle_q_dtype": particle_q.dtype.str,
        "particle_q_shape": list(particle_q.shape),
        "particle_q_sha256": _hash_array(particle_q),
    }
    return workload, replication, state_provenance


def _require_authoritative_arguments(args: argparse.Namespace) -> None:
    """Reject an unpinned measurement before model construction or GPU work."""
    required = {
        "--state-input": args.state_input,
        "--expected-newton-root": args.expected_newton_root,
        "--expected-newton-git-head": args.expected_newton_git_head,
        "--expected-warp-root": args.expected_warp_root,
        "--expected-warp-git-head": args.expected_warp_git_head,
        "--expected-warp-core-sha256": args.expected_warp_core_sha256,
        "--expected-warp-clang-sha256": args.expected_warp_clang_sha256,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(f"authoritative candidate accounting requires: {', '.join(missing)}")
    if args.expected_warp_git_head != PINNED_WARP_GIT_HEAD:
        raise ValueError(f"--expected-warp-git-head must pin {PINNED_WARP_GIT_HEAD}, got {args.expected_warp_git_head}")


def _self_test() -> None:
    """Exercise canonicalization, hashing, distributions, and subset accounting on CPU."""
    old_counts = np.array([3, 1, 0], dtype=np.int32)
    old_targets = np.array([7, 2, 4, 8], dtype=np.int32)
    new_counts = np.array([2, 1, 0], dtype=np.int32)
    new_targets = np.array([7, 2, 8], dtype=np.int32)
    old = CandidateSet(old_counts, _canonical_pairs(old_counts, old_targets))
    new = CandidateSet(new_counts, _canonical_pairs(new_counts, new_targets))
    comparison = _compare_candidate_sets(old, new)
    if comparison["old_only_count"] != 1 or comparison["new_only_count"] != 0:
        raise AssertionError("subset accounting returned unexpected delta counts")
    if comparison["old_only_count_distribution"] != [
        {"candidate_count": 0, "owner_count": 2},
        {"candidate_count": 1, "owner_count": 1},
    ]:
        raise AssertionError("candidate-count distribution is not exact")

    non_subset = CandidateSet(
        np.array([1, 0, 0], dtype=np.int32),
        _canonical_pairs(np.array([1, 0, 0], dtype=np.int32), np.array([99], dtype=np.int32)),
    )
    try:
        _compare_candidate_sets(old, non_subset)
    except RuntimeError as error:
        if "absent from the AABB query" not in str(error):
            raise
    else:
        raise AssertionError("non-subset candidate sets were accepted")

    try:
        _canonical_pairs(np.array([2], dtype=np.int32), np.array([3, 3], dtype=np.int32))
    except RuntimeError as error:
        if "duplicate" not in str(error):
            raise
    else:
        raise AssertionError("duplicate per-owner candidates were accepted")
    print("broad-phase candidate accounting self-test passed")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-input", type=Path, help="Frozen NPZ containing source-world particle_q")
    parser.add_argument(
        "--world-copies", type=int, default=1, help="Replicate the frozen cloth workload this many times"
    )
    parser.add_argument("--block-size", type=int, default=32, help="CUDA launch block size used for accounting kernels")
    parser.add_argument("--device", help="Warp device alias; defaults to Warp's preferred device")
    parser.add_argument("--output", type=Path, help="Write JSON evidence exclusively to this path; otherwise print it")
    parser.add_argument("--expected-newton-root", type=Path, help="Required absolute Newton import root")
    parser.add_argument("--expected-newton-git-head", help="Required clean Newton Git HEAD")
    parser.add_argument("--expected-warp-root", type=Path, help="Required absolute Warp import root")
    parser.add_argument("--expected-warp-git-head", help=f"Required pinned Warp Git HEAD ({PINNED_WARP_GIT_HEAD})")
    parser.add_argument("--expected-warp-core-sha256", help="Required SHA-256 of warp.so")
    parser.add_argument("--expected-warp-clang-sha256", help="Required SHA-256 of warp-clang.so")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only host-accounting tests and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run exact broad-phase candidate accounting."""
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return
    if args.world_copies < 1:
        raise ValueError("--world-copies must be positive")
    if not 1 <= args.block_size <= 256:
        raise ValueError("--block-size must be in [1, 256]")
    _require_authoritative_arguments(args)

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
    if not _RADIUS_QUERIES_AVAILABLE:
        raise RuntimeError("the pinned Warp import does not expose bvh_query_sphere and bvh_query_capsule")
    if args.state_input is None or not args.state_input.is_file():
        raise FileNotFoundError(args.state_input)

    wp.init()
    if args.device:
        wp.set_device(args.device)
    workload, replication, state_provenance = _load_workload(args.state_input, args.world_copies)
    model = workload.model
    detector = workload.detector

    vertex_triangle_aabb = _enumerate_candidates(
        workload,
        "vertex_triangle",
        _count_or_store_vertex_triangle_aabb_candidates,
        args.block_size,
    )
    vertex_triangle_sphere = _enumerate_candidates(
        workload,
        "vertex_triangle",
        _count_or_store_vertex_triangle_sphere_candidates,
        args.block_size,
    )
    edge_edge_aabb = _enumerate_candidates(
        workload,
        "edge_edge",
        _count_or_store_edge_edge_aabb_candidates,
        args.block_size,
    )
    edge_edge_capsule = _enumerate_candidates(
        workload,
        "edge_edge",
        _count_or_store_edge_edge_capsule_candidates,
        args.block_size,
    )

    particle_world = np.ascontiguousarray(model.particle_world.numpy(), dtype=np.int32)
    edge_vertices = np.ascontiguousarray(model.edge_indices.numpy()[:, 2:4], dtype=np.int32)
    edge_owner_world = np.ascontiguousarray(particle_world[edge_vertices[:, 0]], dtype=np.int32)
    triangle_groups = np.ascontiguousarray(detector.tri_groups.numpy(), dtype=np.int32)
    edge_groups = np.ascontiguousarray(detector.edge_groups.numpy(), dtype=np.int32)

    vertex_triangle_comparison = _compare_candidate_sets(vertex_triangle_aabb, vertex_triangle_sphere)
    edge_edge_comparison = _compare_candidate_sets(edge_edge_aabb, edge_edge_capsule)
    workload_provenance = frozen_benchmark._workload_provenance(workload)

    harness_path = Path(__file__).resolve()
    producer_harness_path = Path(frozen_benchmark.__file__).resolve()
    output = {
        "schema_version": "cloth-franka-broad-phase-candidates-v1",
        "harness": {
            "path": str(harness_path),
            "sha256": frozen_benchmark._file_hash(harness_path),
            "argv": sys.argv,
        },
        "frozen_workload_producer": {
            "path": str(producer_harness_path),
            "sha256": frozen_benchmark._file_hash(producer_harness_path),
        },
        "newton": newton_provenance,
        "warp": warp_provenance,
        "required_warp_git_head": PINNED_WARP_GIT_HEAD,
        "state": state_provenance,
        "replication": replication,
        "workload": workload_provenance,
        "device": {
            "alias": str(model.device),
            "name": model.device.name,
            "is_cuda": bool(model.device.is_cuda),
            "arch": model.device.arch,
            "sm_count": model.device.sm_count,
        },
        "block_size": args.block_size,
        "query_semantics": {
            "candidate_level": "raw BVH leaf candidates before topology filters and narrow phase",
            "group_passes": "global owners query the full root once; world owners query own-world then global roots",
            "vertex_triangle_old": "vertex-centered AABB padded by self_contact_margin",
            "vertex_triangle_new": "sphere centered at the vertex with radius self_contact_margin",
            "edge_edge_old": "edge endpoint AABB padded by self_contact_margin",
            "edge_edge_new": "capsule from start along unnormalized end-start, bvh_query_next max_dist=1",
            "zero_length_edge": "direction=(1,0,0), bvh_query_next max_dist=0",
        },
        "vertex_triangle": {
            "aabb": _candidate_summary(
                vertex_triangle_aabb,
                particle_world,
                triangle_groups,
                model.world_count,
            ),
            "sphere": _candidate_summary(
                vertex_triangle_sphere,
                particle_world,
                triangle_groups,
                model.world_count,
            ),
            "comparison": vertex_triangle_comparison,
        },
        "edge_edge": {
            "padded_aabb": _candidate_summary(
                edge_edge_aabb,
                edge_owner_world,
                edge_groups,
                model.world_count,
            ),
            "capsule": _candidate_summary(
                edge_edge_capsule,
                edge_owner_world,
                edge_groups,
                model.world_count,
            ),
            "comparison": edge_edge_comparison,
        },
        "all_radius_candidate_sets_are_subsets": bool(
            vertex_triangle_comparison["new_is_subset_of_old"] and edge_edge_comparison["new_is_subset_of_old"]
        ),
        "contact_output_equivalence_checked": False,
    }
    serialized = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print(f"wrote {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
