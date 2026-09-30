# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit matched inputs, complete trajectories, and reported ALM floor metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STIFFNESSES = (1000, 100000, 10000000)
FLOORS = {"off": None, "floor9": 9.0, "floor1": 1.0, "floor0p1": 0.1, "floor0p01": 0.01, "inertia": 0.0}
FAMILIES = ("tri_stretch", "tri_area", "bend")


def validate_native_control(directory: Path, native_directory: Path):
    """Compare the experimental 9k floor with native preparation on one trajectory."""
    stem = "ke100000_floor9"
    experimental = json.loads((directory / f"{stem}.json").read_text())
    native = json.loads((native_directory / f"{stem}.json").read_text())
    assert native["native_floor9_without_override"]
    assert not experimental["native_floor9_without_override"]
    for key in ("model_sha256", "initial_state_sha256", "source_sha256", "self_contact_storage_multiplier"):
        assert native[key] == experimental[key], key
    assert native["status"] == experimental["status"] == "complete"
    for first, second in zip(native["rows"][1:], experimental["rows"][1:], strict=True):
        for family in FAMILIES:
            assert first[family] == second[family], (first["frame"], family)
    with (
        np.load(directory / f"{stem}.npz") as first,
        np.load(native_directory / f"{stem}.npz") as second,
    ):
        cloth_delta = first["particle_q"] - second["particle_q"]
        body_delta = first["body_q"][..., :3] - second["body_q"][..., :3]
    first_frame_delta = float(np.max(np.abs(cloth_delta[1])))
    assert first_frame_delta < 1e-5, first_frame_delta
    native_residual = np.array([row["original_rms_N"] for row in native["rows"][61:]])
    experimental_residual = np.array([row["original_rms_N"] for row in experimental["rows"][61:]])
    return {
        "passed": True,
        "case": stem,
        "native_rho_statistics_match_every_frame": True,
        "first_frame_max_abs_particle_position_delta_m": first_frame_delta,
        "first_frame_position_tolerance_m": 1e-5,
        "trajectory_max_abs_particle_position_delta_m": float(np.max(np.abs(cloth_delta))),
        "trajectory_rms_particle_position_delta_m": float(np.sqrt(np.mean(cloth_delta**2))),
        "trajectory_max_abs_body_translation_delta_m": float(np.max(np.abs(body_delta))),
        "native_mean_wiggle_original_residual_N": float(native_residual.mean()),
        "experimental_mean_wiggle_original_residual_N": float(experimental_residual.mean()),
        "mean_wiggle_residual_relative_difference": float(
            abs(experimental_residual.mean() - native_residual.mean()) / native_residual.mean()
        ),
        "note": "Independent CUDA contact trajectories can diverge slightly through floating-point accumulation; full-trajectory differences are reported rather than required to be bitwise zero.",
    }


def validate(directory: Path):
    """Check all eighteen cases, including explicitly held failed trajectories."""
    cases = []
    reference_pose = None
    source_hashes = None
    storage = None
    for stiffness in STIFFNESSES:
        model_hash, state_hash = None, None
        for mode, floor in FLOORS.items():
            stem = f"ke{stiffness}_{mode}"
            data = json.loads((directory / f"{stem}.json").read_text())
            rows = data["rows"]
            assert data["stiffness"] == stiffness and data["mode"] == mode, stem
            assert data["floor_multiplier"] == floor, stem
            assert not data["native_floor9_without_override"], stem
            assert data["params"]["cloth_tri_ke"] == stiffness, stem
            assert data["params"]["cloth_tri_ka"] == 0.2 * stiffness, stem
            assert data["params"]["cloth_edge_ke"] == 200, stem
            assert data["params"]["cloth_tri_kd"] == 0.1, stem
            assert data["params"]["cloth_edge_kd"] == 0.02, stem
            assert data["frames"] == 360 and data["snapshot_count"] == 361, stem
            assert data["substeps_per_frame"] == data["iterations_per_step"] == 10, stem
            assert data["fps"] == 60 and data["dt_s"] == 1.0 / 600.0 and data["rho_scale"] == 1, stem
            assert data["max_pin_error_m"] < 1e-6, stem
            assert data["runaway_cutoff_abs_cloth_coordinate_m"] == 10.0, stem
            case_storage = tuple(
                data[key]
                for key in (
                    "self_contact_storage_multiplier",
                    "self_contact_vt_capacity",
                    "self_contact_ee_capacity",
                    "self_contact_kernel_launch_size",
                )
            )
            if storage is None:
                storage = case_storage
            assert case_storage == storage and case_storage[0] == 16, stem
            assert len(rows) == 361, stem
            if model_hash is None:
                model_hash, state_hash = data["model_sha256"], data["initial_state_sha256"]
            assert data["model_sha256"] == model_hash and data["initial_state_sha256"] == state_hash, stem
            if source_hashes is None:
                source_hashes = data["source_sha256"]
            assert data["source_sha256"] == source_hashes, stem
            failure = data["failure_frame"]
            last_valid = 360 if failure is None else failure - 1
            assert data["last_valid_frame"] == last_valid, stem
            assert data["status"] == ("complete" if failure is None else "failed"), stem
            valid = np.arange(361) <= last_valid
            with np.load(directory / f"{stem}.npz") as trajectory:
                points, bodies = trajectory["particle_q"], trajectory["body_q"]
                assert points.shape == (361, data["particle_count"], 3), stem
                assert bodies.shape == (361, data["body_count"], 7), stem
                assert np.isfinite(points).all() and np.isfinite(bodies).all(), stem
                np.testing.assert_array_equal(trajectory["frame"], np.arange(361), err_msg=stem)
                np.testing.assert_array_equal(trajectory["valid"], valid, err_msg=stem)
                if reference_pose is None:
                    reference_pose = points[0].copy(), bodies[0].copy()
                np.testing.assert_array_equal(points[0], reference_pose[0], err_msg=stem)
                np.testing.assert_array_equal(bodies[0], reference_pose[1], err_msg=stem)
                if failure is not None:
                    np.testing.assert_array_equal(
                        points[failure:], np.broadcast_to(points[last_valid], points[failure:].shape)
                    )
                    np.testing.assert_array_equal(
                        bodies[failure:], np.broadcast_to(bodies[last_valid], bodies[failure:].shape)
                    )
            with (directory / f"{stem}.csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            assert len(csv_rows) == 361, stem
            for frame, row in enumerate(rows):
                assert row["frame"] == frame and row["valid"] == bool(valid[frame]), (stem, frame)
                assert abs(row["time_s"] - frame / 60.0) < 1e-6, (stem, frame)
                assert int(csv_rows[frame]["frame"]) == frame, (stem, frame)
                if frame == 0 or not valid[frame]:
                    assert row["original_rms_N"] is None, (stem, frame)
                    continue
                assert row["original_rms_N"] >= 0 and np.isfinite(row["original_rms_N"]), (stem, frame)
                assert row["self_contact_vt_candidates"] <= data["self_contact_vt_capacity"], (stem, frame)
                assert row["self_contact_ee_candidates"] <= data["self_contact_ee_capacity"], (stem, frame)
                for name in FAMILIES:
                    family = row[name]
                    if floor is None:
                        assert family["active_rows"] == 0 and family["effective_over_k"]["median"] == 1, (
                            stem,
                            frame,
                            name,
                        )
                        continue
                    if family["active_rows"] == 0:
                        continue
                    assert 0 <= family["floor_fraction"] <= 1, (stem, frame, name)
                    for statistic in ("min", "median", "max"):
                        ratio = family["rho_over_k"][statistic]
                        effective = family["effective_over_k"][statistic]
                        assert ratio > 0 and ratio >= floor * (1 - 3e-6), (stem, frame, name)
                        np.testing.assert_allclose(effective, ratio / (1 + ratio), rtol=2e-6, atol=1e-12)
            wiggle = [row for row in rows[61:] if row["valid"]]
            cases.append(
                {
                    "case": stem,
                    "status": data["status"],
                    "failure_frame": failure,
                    "failure_reason": data["failure_reason"],
                    "valid_snapshots": int(valid.sum()),
                    "mean_wiggle_stretch_percent": float(np.mean([100 * row["stretch_score"] for row in wiggle]))
                    if wiggle
                    else None,
                    "mean_wiggle_bend_rad": float(np.mean([row["bend_score"] for row in wiggle])) if wiggle else None,
                    "mean_wiggle_original_residual_N": float(np.mean([row["original_rms_N"] for row in wiggle]))
                    if wiggle
                    else None,
                    "final_valid_Keff_over_k": {name: rows[last_valid][name]["effective_over_k"] for name in FAMILIES},
                }
            )
    return {
        "passed": True,
        "cases": len(cases),
        "requested_snapshots": 18 * 361,
        "source_sha256": source_hashes,
        "results": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "results-rho-video-sweep")
    parser.add_argument("--native-control-directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    if args.native_control_directory:
        control = validate_native_control(args.directory, args.native_control_directory)
        (args.directory / "native-control-validation.json").write_text(json.dumps(control, indent=2) + "\n")
        print(json.dumps(control, indent=2))
