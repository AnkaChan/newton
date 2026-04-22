# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Mesh Quality Diagnostic for KFC Bag Sag
#
# Runs example_kfc_bag_sag.py for the requested number of frames and dumps
# mesh-quality metrics on the physics proxy at sampled frames.
#
# Metrics computed per frame (on the bag physics mesh, cm units):
#   1. Triangle aspect ratio (longest_edge / shortest_altitude).
#   2. Triangle angles (all three per triangle).
#   3. Triangle areas (signed vs rest-frame normal + absolute).
#   4. Edge lengths (unique undirected edges).
#   5. Vertex valences (incident triangles per vertex).
#   6. Inversions (signed area negative vs rest-frame normal).
#   7. Overall quality: fraction of triangles with aspect < 2 AND
#      min-angle > 25 deg.
#
# Output:
#   - Per-frame row printed to stdout.
#   - Summary JSON saved to <out_dir>/summary.json
#   - NPZ dumps per sampled frame saved to <out_dir>/data/frame_XXXX.npz
#
# Usage:
#   uv run python examples/mesh_quality_bag_sag.py \
#       --num-frames 300 \
#       --sample-frames 0,50,100,150,200,250,300 \
#       --out-dir /home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/mesh-quality-bag-sag

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton  # noqa: F401  (ensures init)

# Add BagSim directory to sys.path so we can import the Example class
_BAG_SIM_DIR = "/home/horde/Code/Graphics/warp-dev-demos/Newton-Debug/BagSim"
if _BAG_SIM_DIR not in sys.path:
    sys.path.insert(0, _BAG_SIM_DIR)

from example_kfc_bag_sag import Example  # noqa: E402


# ---------------------------------------------------------------------------
# Mesh quality metrics
# ---------------------------------------------------------------------------


def _triangle_metrics(verts: np.ndarray, faces: np.ndarray, ref_normals: np.ndarray | None):
    """Compute per-triangle metrics.

    Args:
        verts: (N, 3) particle positions (cm).
        faces: (F, 3) triangle vertex indices into the bag particle subrange.
        ref_normals: (F, 3) rest-frame per-triangle unit normals, or None if
            ref has not been captured yet (in which case the current normals
            are returned and the caller should seed the reference).

    Returns:
        dict with arrays and scalar stats.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    len01 = np.linalg.norm(e01, axis=1)
    len12 = np.linalg.norm(e12, axis=1)
    len20 = np.linalg.norm(e20, axis=1)

    edge_len_stack = np.stack([len01, len12, len20], axis=1)  # (F, 3)
    longest = edge_len_stack.max(axis=1)
    shortest = edge_len_stack.min(axis=1)

    cross = np.cross(e01, -e20)  # = e01 x e02
    area_vec = 0.5 * cross
    abs_area = np.linalg.norm(area_vec, axis=1)

    # Unit normals (guard against zero area)
    eps = 1e-20
    n_len = np.linalg.norm(cross, axis=1)
    normals = np.where(n_len[:, None] > eps, cross / np.maximum(n_len[:, None], eps), 0.0)

    # Signed area against reference normal (positive if still aligned)
    if ref_normals is not None:
        signed = np.einsum("ij,ij->i", area_vec, ref_normals)
    else:
        signed = abs_area.copy()

    # Shortest altitude h = 2 * area / longest_edge
    safe_longest = np.maximum(longest, eps)
    shortest_altitude = 2.0 * abs_area / safe_longest
    # Aspect ratio: longest edge / shortest altitude
    safe_altitude = np.maximum(shortest_altitude, eps)
    aspect_le_alt = longest / safe_altitude

    # Three angles via law of cosines (a is opposite to v2 side, etc.)
    # angle at v0 is between e01 and -e20 (i.e. e02)
    def _angle(a, b):
        an = np.maximum(np.linalg.norm(a, axis=1), eps)
        bn = np.maximum(np.linalg.norm(b, axis=1), eps)
        c = np.einsum("ij,ij->i", a, b) / (an * bn)
        c = np.clip(c, -1.0, 1.0)
        return np.degrees(np.arccos(c))

    ang0 = _angle(e01, -e20)     # at v0
    ang1 = _angle(-e01, e12)     # at v1
    ang2 = _angle(-e12, e20)     # at v2
    angles = np.stack([ang0, ang1, ang2], axis=1)  # (F, 3)
    min_angle = angles.min(axis=1)
    max_angle = angles.max(axis=1)

    return {
        "abs_area": abs_area,
        "signed_area": signed,
        "longest_edge": longest,
        "shortest_edge": shortest,
        "aspect_le_alt": aspect_le_alt,
        "angles": angles,
        "min_angle": min_angle,
        "max_angle": max_angle,
        "edge_lens": edge_len_stack,
        "normals": normals,
    }


def _edge_lengths(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return lengths of unique undirected edges."""
    e = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)
    e_sorted = np.sort(e, axis=1)
    uniq = np.unique(e_sorted, axis=0)
    d = verts[uniq[:, 0]] - verts[uniq[:, 1]]
    return np.linalg.norm(d, axis=1)


def _vertex_valence(n_verts: int, faces: np.ndarray) -> np.ndarray:
    """Return count of incident triangles per vertex (0..n_verts-1)."""
    v = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    counts = np.bincount(v, minlength=n_verts)
    return counts


def compute_metrics(
    verts: np.ndarray,
    faces: np.ndarray,
    ref_normals: np.ndarray | None,
    particle_radius: float,
) -> dict:
    tm = _triangle_metrics(verts, faces, ref_normals)

    abs_area = tm["abs_area"]
    signed_area = tm["signed_area"]
    aspect = tm["aspect_le_alt"]
    min_ang = tm["min_angle"]
    max_ang = tm["max_angle"]

    # Aspect ratio
    aspect_stats = {
        "min": float(aspect.min()),
        "max": float(aspect.max()),
        "mean": float(aspect.mean()),
        "median": float(np.median(aspect)),
        "count_bad_gt_3": int(np.sum(aspect > 3.0)),
        "count_bad_gt_5": int(np.sum(aspect > 5.0)),
        "count_bad_gt_10": int(np.sum(aspect > 10.0)),
    }

    angle_stats = {
        "min_angle_min": float(min_ang.min()),
        "min_angle_mean": float(min_ang.mean()),
        "max_angle_max": float(max_ang.max()),
        "max_angle_mean": float(max_ang.mean()),
        "count_min_angle_lt_20": int(np.sum(min_ang < 20.0)),
        "count_min_angle_lt_10": int(np.sum(min_ang < 10.0)),
        "count_max_angle_gt_120": int(np.sum(max_ang > 120.0)),
        "count_max_angle_gt_150": int(np.sum(max_ang > 150.0)),
    }

    area_stats = {
        "min": float(abs_area.min()),
        "max": float(abs_area.max()),
        "mean": float(abs_area.mean()),
        "std": float(abs_area.std()),
        "cv": float(abs_area.std() / max(abs_area.mean(), 1e-20)),
        "count_degen_lt_0p01": int(np.sum(abs_area < 0.01)),
        "count_near_degen_lt_0p1": int(np.sum(abs_area < 0.1)),
    }

    edge_lens = _edge_lengths(verts, faces)
    edge_stats = {
        "min": float(edge_lens.min()),
        "max": float(edge_lens.max()),
        "mean": float(edge_lens.mean()),
        "std": float(edge_lens.std()),
        "count_lt_particle_radius": int(np.sum(edge_lens < particle_radius)),
        "count_unique_edges": int(len(edge_lens)),
    }

    valence = _vertex_valence(len(verts), faces)
    # Only care about vertices that actually appear in faces (others are pinned)
    # but for the bag, all particles appear. Still, drop any zero-valence.
    valence_nonzero = valence[valence > 0]
    valence_stats = {
        "min": int(valence_nonzero.min()),
        "max": int(valence_nonzero.max()),
        "mean": float(valence_nonzero.mean()),
        "count_irregular": int(np.sum((valence_nonzero != 5) & (valence_nonzero != 6))),
        "count_valence_3_or_less": int(np.sum(valence_nonzero <= 3)),
        "count_valence_gt_8": int(np.sum(valence_nonzero > 8)),
        "n_vertices": int(len(valence_nonzero)),
    }

    # Inversions: signed_area < 0 vs ref normal
    if ref_normals is not None:
        inv_mask = signed_area < 0.0
        n_inverted = int(inv_mask.sum())
    else:
        n_inverted = 0
    inversion_stats = {
        "count_inverted": n_inverted,
        "count_signed_area_near_zero": int(np.sum(np.abs(signed_area) < 0.01)),
        "fraction_inverted": float(n_inverted) / max(len(faces), 1),
    }

    # Overall: aspect < 2 AND min_angle > 25
    good_mask = (aspect < 2.0) & (min_ang > 25.0)
    overall = {
        "n_faces": int(len(faces)),
        "n_good": int(good_mask.sum()),
        "fraction_good": float(good_mask.mean()),
    }

    return {
        "aspect": aspect_stats,
        "angles": angle_stats,
        "area": area_stats,
        "edges": edge_stats,
        "valence": valence_stats,
        "inversion": inversion_stats,
        "overall": overall,
        "_raw": {
            "abs_area": abs_area,
            "signed_area": signed_area,
            "aspect": aspect,
            "min_angle": min_ang,
            "max_angle": max_ang,
            "edge_lens": edge_lens,
            "valence": valence,
            "normals": tm["normals"],
        },
    }


# ---------------------------------------------------------------------------
# Headless viewer stub
# ---------------------------------------------------------------------------


class _NullViewer:
    """Minimal viewer stub that accepts everything the Example does to it."""

    def __init__(self):
        self.show_collision = False
        self.show_triangles = False

    def set_model(self, model):
        self._model = model

    def set_camera(self, **kwargs):
        pass

    def begin_frame(self, t):
        pass

    def end_frame(self):
        pass

    def log_state(self, state):
        pass

    def log_mesh(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument(
        "--sample-frames", type=str,
        default="0,50,100,150,200,250,300",
        help="Comma-separated frame indices at which to compute metrics",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/mesh-quality-bag-sag",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    sample_frames = sorted({int(x) for x in args.sample_frames.split(",") if x.strip()})
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    # Build Example with null viewer
    viewer = _NullViewer()
    example = Example(viewer=viewer, show_sim_mesh=True, test_mode=False)

    model = example.model
    bag_start = example._bag_particle_start
    bag_end = example._bag_particle_end
    n_bag_verts = bag_end - bag_start

    # Bag faces: physics proxy indices are stored in _proxy_indices_wp as
    # particle indices in the full model (already offset by bag_start).
    # Recover the local-to-bag indices.
    proxy_idx_full = example._proxy_indices_wp.numpy().reshape(-1, 3).astype(np.int64)
    bag_faces_local = proxy_idx_full - bag_start
    # Sanity
    assert bag_faces_local.min() >= 0 and bag_faces_local.max() < n_bag_verts, (
        f"Face index out of bag range: min={bag_faces_local.min()}, max={bag_faces_local.max()}, n_bag_verts={n_bag_verts}"
    )

    particle_radius = 0.80  # matches _PARTICLE_RADIUS

    print(f"[mesh_quality] Bag: {n_bag_verts} verts, {len(bag_faces_local)} triangles")
    print(f"[mesh_quality] Sampling at frames: {sample_frames}")
    print(f"[mesh_quality] Total frames: {args.num_frames}")
    print(f"[mesh_quality] Output: {out_dir}")

    # Reference normals at frame 0 (rest pose) for inversion detection
    init_verts = example.state_0.particle_q.numpy()[bag_start:bag_end].copy()
    ref_tm = _triangle_metrics(init_verts, bag_faces_local, None)
    ref_normals = ref_tm["normals"].copy()

    # Collect summary rows
    summary = {
        "config": {
            "num_frames": args.num_frames,
            "sample_frames": sample_frames,
            "n_bag_verts": int(n_bag_verts),
            "n_bag_faces": int(len(bag_faces_local)),
            "particle_radius": particle_radius,
            "sim_substeps": example.sim_substeps,
            "sim_dt": example.sim_dt,
            "frame_dt": example.frame_dt,
        },
        "frames": [],
    }

    def _sample_and_record(frame_idx: int):
        verts = example.state_0.particle_q.numpy()[bag_start:bag_end]
        metrics = compute_metrics(verts, bag_faces_local, ref_normals, particle_radius)
        raw = metrics.pop("_raw")
        # NPZ dump
        npz_path = data_dir / f"frame_{frame_idx:04d}.npz"
        np.savez_compressed(
            npz_path,
            frame=np.int32(frame_idx),
            verts=verts.astype(np.float32),
            faces=bag_faces_local.astype(np.int32),
            ref_normals=ref_normals.astype(np.float32),
            abs_area=raw["abs_area"].astype(np.float32),
            signed_area=raw["signed_area"].astype(np.float32),
            aspect=raw["aspect"].astype(np.float32),
            min_angle=raw["min_angle"].astype(np.float32),
            max_angle=raw["max_angle"].astype(np.float32),
            edge_lens=raw["edge_lens"].astype(np.float32),
            valence=raw["valence"].astype(np.int32),
        )
        summary["frames"].append({"frame": frame_idx, "metrics": metrics})

        # Pretty-print a row
        a = metrics["aspect"]
        ang = metrics["angles"]
        ar = metrics["area"]
        ed = metrics["edges"]
        inv = metrics["inversion"]
        ov = metrics["overall"]
        print(
            f"[frame {frame_idx:4d}] "
            f"good={ov['fraction_good']*100:5.1f}% | "
            f"aspect mean={a['mean']:.2f} max={a['max']:.2f} bad>3={a['count_bad_gt_3']} | "
            f"minang<20={ang['count_min_angle_lt_20']} maxang>120={ang['count_max_angle_gt_120']} | "
            f"area min={ar['min']:.3f} cv={ar['cv']:.2f} degen={ar['count_degen_lt_0p01']} near={ar['count_near_degen_lt_0p1']} | "
            f"edge min={ed['min']:.2f} max={ed['max']:.2f} <{particle_radius}={ed['count_lt_particle_radius']} | "
            f"inv={inv['count_inverted']}"
        )

    # Frame 0: rest pose, before any step
    if 0 in sample_frames:
        _sample_and_record(0)

    for frame_idx in range(1, args.num_frames + 1):
        example.step()
        if frame_idx in sample_frames:
            _sample_and_record(frame_idx)

    # Save summary JSON
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[mesh_quality] Summary JSON -> {summary_path}")
    print(f"[mesh_quality] Per-frame NPZ -> {data_dir}")


if __name__ == "__main__":
    main()
