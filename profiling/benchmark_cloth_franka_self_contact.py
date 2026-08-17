#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Benchmark and capture the native cloth Franka self-contact workload."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.vbd import particle_vbd_kernels
from newton.examples.cloth.example_cloth_franka import Example
from newton.viewer import ViewerNull

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".toml",
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


def _source_fingerprint(root: Path) -> dict[str, object]:
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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_paths(*paths: Path | None) -> None:
    resolved = [path.resolve() for path in paths if path is not None]
    if len(resolved) != len(set(resolved)):
        raise ValueError("--output and --state-output must be different paths")
    for path in resolved:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _publish_exclusive(path: Path, writer: Callable[[BinaryIO], object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, result: dict[str, object]) -> None:
    payload = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    _publish_exclusive(path, lambda stream: stream.write(payload))


def _write_state(path: Path, arrays: dict[str, np.ndarray]) -> None:
    _publish_exclusive(path, lambda stream: np.savez_compressed(stream, **arrays))


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _active_pairs(buffer, offsets, counts, capacities) -> np.ndarray:
    buffer_np = buffer.numpy()
    offsets_np = offsets.numpy()
    counts_np = counts.numpy()
    capacities_np = capacities.numpy()
    pairs = []
    for row, (offset, count, capacity) in enumerate(zip(offsets_np[:-1], counts_np, capacities_np, strict=True)):
        stored_count = min(int(count), int(capacity))
        for slot in range(stored_count):
            pair_offset = 2 * (int(offset) + slot)
            pairs.append((row, int(buffer_np[pair_offset + 1])))
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    result = np.asarray(pairs, dtype=np.int32)
    return result[np.lexsort((result[:, 1], result[:, 0]))]


def _contact_snapshot(example: Example) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    info = example.cloth_solver.trimesh_collision_detector.collision_info
    vt_counts = info.vertex_colliding_triangles_count.numpy()
    vt_capacities = info.vertex_colliding_triangles_buffer_sizes.numpy()
    ee_counts = info.edge_colliding_edges_count.numpy()
    ee_capacities = info.edge_colliding_edges_buffer_sizes.numpy()
    vt_pairs = _active_pairs(
        info.vertex_colliding_triangles,
        info.vertex_colliding_triangles_offsets,
        info.vertex_colliding_triangles_count,
        info.vertex_colliding_triangles_buffer_sizes,
    )
    ee_pairs = _active_pairs(
        info.edge_colliding_edges,
        info.edge_colliding_edges_offsets,
        info.edge_colliding_edges_count,
        info.edge_colliding_edges_buffer_sizes,
    )
    summary = {
        "vertex_triangle": {
            "raw_count": int(vt_counts.sum()),
            "stored_count": int(np.minimum(vt_counts, vt_capacities).sum()),
            "nonempty_rows": int(np.count_nonzero(vt_counts)),
            "overflow_rows": int(np.count_nonzero(vt_counts > vt_capacities)),
            "pair_sha256": _array_hash(vt_pairs),
        },
        "edge_edge": {
            "raw_count": int(ee_counts.sum()),
            "stored_count": int(np.minimum(ee_counts, ee_capacities).sum()),
            "nonempty_rows": int(np.count_nonzero(ee_counts)),
            "overflow_rows": int(np.count_nonzero(ee_counts > ee_capacities)),
            "pair_sha256": _array_hash(ee_pairs),
        },
    }
    arrays = {
        "vertex_triangle_counts": vt_counts,
        "vertex_triangle_pairs": vt_pairs,
        "edge_edge_counts": ee_counts,
        "edge_edge_pairs": ee_pairs,
    }
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--collision-detection-block-size", type=int)
    parser.add_argument("--self-contact-force-block-dim", type=int, choices=(128, 256))
    parser.add_argument(
        "--self-contact-force-max-blocks",
        choices=("sm", "2sm", "uncapped"),
    )
    parser.add_argument("--expected-source-root", type=Path)
    parser.add_argument("--expected-source-hash")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    _validate_output_paths(args.output, args.state_output)

    source_root = Path(newton.__file__).resolve().parents[1]
    source = _source_fingerprint(source_root)
    if args.expected_source_root is not None and source_root != args.expected_source_root.resolve():
        raise ValueError(f"imported Newton from {source_root}, expected {args.expected_source_root.resolve()}")
    if args.expected_source_hash is not None:
        expected_source_hash = args.expected_source_hash.lower()
        if len(expected_source_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_source_hash
        ):
            raise ValueError("--expected-source-hash must be a 64-character hexadecimal SHA-256")
        if source["source_hash"] != expected_source_hash:
            raise ValueError(f"source SHA-256 {source['source_hash']} does not match expected {expected_source_hash}")
    run_id = args.run_id or uuid.uuid4().hex
    viewer = ViewerNull(num_frames=args.frames)
    example_args = newton.examples.default_args() if hasattr(newton.examples, "default_args") else None
    force_launch_override = (
        args.self_contact_force_block_dim is not None or args.self_contact_force_max_blocks is not None
    )
    defer_capture = args.collision_detection_block_size is not None or force_launch_override
    if args.collision_detection_block_size is not None and args.collision_detection_block_size <= 0:
        raise ValueError("--collision-detection-block-size must be positive")

    capture = None
    if not defer_capture:
        example = Example(viewer, example_args)
    else:
        capture = Example.capture
        try:
            Example.capture = lambda _example: None
            example = Example(viewer, example_args)
        finally:
            Example.capture = capture

    force_block_dim = args.self_contact_force_block_dim or 256
    force_max_blocks_mode = args.self_contact_force_max_blocks or "production"
    sm_count = example.model.device.sm_count
    force_max_blocks = {
        "production": None,
        "sm": sm_count,
        "2sm": 2 * sm_count,
        "uncapped": 0,
    }[force_max_blocks_mode]

    if defer_capture:
        if args.collision_detection_block_size is not None:
            example.cloth_solver.trimesh_collision_detector.collision_detection_block_size = (
                args.collision_detection_block_size
            )

        if force_launch_override:
            wp.load_module(
                particle_vbd_kernels,
                device=example.model.device,
                block_dim=force_block_dim,
            )

            original_launch = wp.launch

            def launch_with_force_settings(*launch_args, **launch_kwargs):
                kernel = launch_kwargs.get("kernel", launch_args[0] if launch_args else None)
                if kernel is particle_vbd_kernels.accumulate_self_contact_force_and_hessian:
                    launch_kwargs["block_dim"] = force_block_dim
                    if force_max_blocks is not None:
                        launch_kwargs["max_blocks"] = force_max_blocks
                return original_launch(*launch_args, **launch_kwargs)

            try:
                wp.launch = launch_with_force_settings
                capture(example)
            finally:
                wp.launch = original_launch
        else:
            capture(example)
    wp.synchronize_device()

    profiler = None
    if args.cuda_profiler_api:
        profiler = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
        if profiler.cudaProfilerStart() != 0:
            raise RuntimeError("cudaProfilerStart() failed")

    start = time.perf_counter()
    try:
        newton.examples.run(example, args=None)
        wp.synchronize_device()
        elapsed = time.perf_counter() - start
    finally:
        if profiler is not None and profiler.cudaProfilerStop() != 0:
            raise RuntimeError("cudaProfilerStop() failed")

    particle_q = example.state_0.particle_q.numpy()
    particle_qd = example.state_0.particle_qd.numpy()
    body_q = example.state_0.body_q.numpy()
    body_qd = example.state_0.body_qd.numpy()
    contact_summary, contact_arrays = _contact_snapshot(example)

    result = {
        "run_id": run_id,
        "source": {
            **source,
            "newton_file": str(Path(newton.__file__).resolve()),
        },
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_hash(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "environment_allowlist": {key: os.environ[key] for key in ENVIRONMENT_ALLOWLIST if key in os.environ},
        "device": str(wp.get_device()),
        "frames": args.frames,
        "elapsed_seconds": elapsed,
        "milliseconds_per_frame": 1000.0 * elapsed / args.frames,
        "frames_per_second": args.frames / elapsed,
        "measurement": {
            "timer": "time.perf_counter",
            "synchronized_before": True,
            "synchronized_after": True,
            "profiler_stop_excluded": True,
            "cuda_profiler_api": args.cuda_profiler_api,
            "valid_for_end_to_end_comparison": not args.cuda_profiler_api,
        },
        "configuration": {
            "sim_substeps": example.sim_substeps,
            "iterations": example.iterations,
            "frame_dt": example.frame_dt,
            "sim_dt": example.sim_dt,
            "particle_self_contact_radius": example.particle_self_contact_radius,
            "particle_self_contact_margin": example.particle_self_contact_margin,
            "particle_collision_detection_interval": example.cloth_solver.particle_collision_detection_interval,
            "collision_detection_block_size": (
                example.cloth_solver.trimesh_collision_detector.collision_detection_block_size
            ),
            "self_contact_force_block_dim": force_block_dim,
            "self_contact_force_max_blocks": force_max_blocks_mode,
            "self_contact_force_max_blocks_resolved": force_max_blocks,
            "self_contact_force_launch_override": force_launch_override,
        },
        "model": {
            "particle_count": example.model.particle_count,
            "triangle_count": example.model.tri_count,
            "edge_count": example.model.edge_count,
            "color_group_sizes": [int(group.size) for group in example.model.particle_color_groups],
        },
        "contacts_after_final_frame": contact_summary,
        "state": {
            "particle_q_sha256": _array_hash(particle_q),
            "particle_qd_sha256": _array_hash(particle_qd),
            "body_q_sha256": _array_hash(body_q),
            "body_qd_sha256": _array_hash(body_qd),
            "particle_q_min": float(particle_q.min()),
            "particle_q_max": float(particle_q.max()),
            "particle_qd_abs_max": float(np.abs(particle_qd).max()),
        },
    }

    if args.output is not None:
        _write_json(args.output, result)
    if args.state_output is not None:
        _write_state(
            args.state_output,
            {
                "particle_q": particle_q,
                "particle_qd": particle_qd,
                "body_q": body_q,
                "body_qd": body_qd,
                **contact_arrays,
            },
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
