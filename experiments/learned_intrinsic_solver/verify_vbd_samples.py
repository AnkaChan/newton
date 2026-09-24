# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the archived states and encoded video files for the 20 VBD samples."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def verify(directory: Path) -> dict:
    """Check video metadata, decoded images, seed diversity, states, and framing."""
    manifest = json.loads((directory / "manifest.json").read_text())
    samples = manifest["samples"]
    assert [sample["seed"] for sample in samples] == list(range(20)), "Expected exactly seeds 0 through 19"
    camera = manifest["simulation"]["camera"]
    camera_position = np.asarray(camera["position"])
    forward = np.asarray(camera["target"]) - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tan_half_fov = np.tan(np.radians(camera["fov_degrees"]) / 2)
    aspect = camera["width"] / camera["height"]
    results, hashes = [], []
    for sample in samples:
        metrics = sample["metrics"]
        assert sample["status"] == "complete"
        assert metrics["finite_all_frames"] and metrics["repeat_seed_exact"]
        assert metrics["clamp_drift_max_m"] == 0
        assert metrics["trajectory_tet_min_volume_ratio"] > 0
        assert metrics["deformation_displacement_scale"] == 1
        assert metrics["frame_count"] == 300 and metrics["simulation_substeps"] == 3000
        assert metrics["physical_duration_seconds"] == 10
        with np.load(directory / sample["initial_state"], allow_pickle=False) as initial:
            assert int(initial["seed"]) == sample["seed"]
            positions = initial["particle_initial_positions_world"]
            velocities = initial["particle_initial_velocities_world"]
            fixed = initial["clamped_particle_indices"]
            rest = initial["particle_rest_positions_world"]
            assert positions.shape == velocities.shape == (4961, 3)
            assert initial["tet_indices_newton"].shape == (20000, 4)
            assert initial["cell_deformation"].shape == (4000, 3, 3)
            assert len(fixed) == 121
            np.testing.assert_array_equal(positions[fixed], rest[fixed])
            np.testing.assert_array_equal(velocities[fixed], 0)
            assert np.isfinite(positions).all() and np.isfinite(velocities).all()
            digest = hashlib.sha256(positions.tobytes() + velocities.tobytes()).hexdigest()
            assert digest == metrics["initialization_sha256"]
            hashes.append(digest)
        with np.load(directory / sample["final_state"], allow_pickle=False) as final:
            assert np.isfinite(final["particle_positions_world"]).all()
            assert np.isfinite(final["particle_velocities_world"]).all()
            np.testing.assert_array_equal(final["particle_positions_world"][fixed], rest[fixed])
            np.testing.assert_array_equal(final["particle_velocities_world"][fixed], 0)
            np.testing.assert_allclose(final["traces"][:, 0], np.arange(1, 301) / 30, atol=0, rtol=0)
        # The entire trajectory AABB lies inside the same fixed camera frustum.
        bounds = np.array(
            list(
                itertools.product(
                    *zip(metrics["trajectory_bounds_min_m"], metrics["trajectory_bounds_max_m"], strict=True)
                )
            )
        )
        camera_points = bounds - camera_position
        depth = camera_points @ forward
        assert depth.min() > 0
        ndc_x = camera_points @ right / (depth * tan_half_fov * aspect)
        ndc_y = camera_points @ up / (depth * tan_half_fov)
        assert np.abs(ndc_x).max() < 1 and np.abs(ndc_y).max() < 1, "Scene escaped fixed-camera framing"
        with imageio.get_reader(directory / sample["video"]) as reader:
            metadata = reader.get_meta_data()
            frame_count = reader.count_frames()
            assert frame_count == 300
            assert metadata["fps"] == 30
            assert abs(metadata["duration"] - 10) < 1e-6
            assert tuple(metadata["size"]) == (1280, 720)
            decoded = [reader.get_data(index) for index in (0, 149, 299)]
            standard_deviations = [float(frame.std()) for frame in decoded]
            assert min(standard_deviations) > 3, "Black/uniform encoded frame"
            motion = float(np.abs(decoded[0].astype(float) - decoded[-1]).mean())
            assert motion > 0.1, "No visible motion in video"
        results.append(
            {
                "seed": sample["seed"],
                "video_frame_count": frame_count,
                "video_duration_seconds": metadata["duration"],
                "video_fps": metadata["fps"],
                "video_size": metadata["size"],
                "decoded_frame_std": standard_deviations,
                "first_last_frame_mean_absolute_difference": motion,
                "trajectory_aabb_max_absolute_ndc": [float(np.abs(ndc_x).max()), float(np.abs(ndc_y).max())],
                "initial_archive_hash_verified": True,
                "passed": True,
            }
        )
        print(
            f"Verified seed {sample['seed']:02d}: 300 frames / 10 seconds / finite / fixed clamp / positive tets",
            flush=True,
        )
    assert len(set(hashes)) == 20, "Initial states were duplicated"
    report = {
        "passed": True,
        "sample_count": 20,
        "seeds": list(range(20)),
        "distinct_initial_state_hashes": 20,
        "total_video_duration_seconds": 200,
        "total_frames": 6000,
        "fixed_camera_all_samples": True,
        "all_trajectory_bounds_inside_camera": True,
        "all_recorded_frames_finite_and_nonblack": True,
        "clamp_drift_max_m": 0.0,
        "trajectory_tet_min_volume_ratio": min(
            sample["metrics"]["trajectory_tet_min_volume_ratio"] for sample in samples
        ),
        "samples": results,
    }
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    verify(args.directory)
