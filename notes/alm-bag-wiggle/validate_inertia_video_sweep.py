# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit native inertia scaling and matched inputs in the three-material sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STIFFNESSES = (1000, 100000, 10000000)
SCALES = {"off": None, "inertia1": 1, "inertia10": 10, "inertia100": 100, "inertia1000": 1000}
FAMILIES = ("tri_stretch", "tri_area", "bend")


def validate(directory):
    """Check all fifteen trajectories and independent initial-pose penalty metrics."""
    results = []
    reference_pose = None
    source_hashes = None
    production_hashes = None
    reference_metrics = {}
    storage = None
    for stiffness in STIFFNESSES:
        model_hash = state_hash = None
        material_metrics = {}
        for mode, scale in SCALES.items():
            stem = f"ke{stiffness}_{mode}"
            data = json.loads((directory / f"{stem}.json").read_text())
            rows = data["rows"]
            assert data["mode"] == mode and data["stiffness"] == stiffness, stem
            assert data["alm"] == (scale is not None), stem
            assert data["floor_multiplier"] is None and data["native_metric_preparation"], stem
            assert not data["experiment_rho_override"], stem
            assert data["initial_metric_pose_sha256"] == data["initial_state_sha256"], stem
            if scale is not None:
                assert data["rho_scale"] == scale, stem
            assert data["params"]["cloth_tri_ke"] == stiffness, stem
            assert data["params"]["cloth_tri_ka"] == 0.2 * stiffness, stem
            assert data["params"]["cloth_edge_ke"] == 200, stem
            assert data["params"]["cloth_tri_kd"] == 0.1 and data["params"]["cloth_edge_kd"] == 0.02, stem
            assert data["frames"] == 360 and data["snapshot_count"] == len(rows) == 361, stem
            assert data["fps"] == 60 and data["dt_s"] == 1 / 600, stem
            assert data["substeps_per_frame"] == data["iterations_per_step"] == 10, stem
            assert data["max_pin_error_m"] < 1e-6, stem
            assert data["runaway_cutoff_abs_cloth_coordinate_m"] == 10, stem
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
            assert case_storage == storage and storage[0] == 16, stem
            if source_hashes is None:
                source_hashes = data["source_sha256"]
            assert data["source_sha256"] == source_hashes, stem
            if production_hashes is None:
                production_hashes = data["production_source_sha256"]
            assert data["production_source_sha256"] == production_hashes, stem
            if model_hash is None:
                model_hash, state_hash = data["model_sha256"], data["initial_state_sha256"]
            assert (data["model_sha256"], data["initial_state_sha256"]) == (model_hash, state_hash), stem
            failure = data["failure_frame"]
            last = 360 if failure is None else failure - 1
            assert data["last_valid_frame"] == last, stem
            assert data["status"] == ("complete" if failure is None else "failed"), stem
            valid = np.arange(361) <= last
            with np.load(directory / f"{stem}.npz") as archive:
                points, bodies = archive["particle_q"], archive["body_q"]
                assert points.shape == (361, data["particle_count"], 3), stem
                assert bodies.shape == (361, data["body_count"], 7), stem
                assert np.isfinite(points).all() and np.isfinite(bodies).all(), stem
                np.testing.assert_array_equal(archive["frame"], np.arange(361), err_msg=stem)
                np.testing.assert_array_equal(archive["valid"], valid, err_msg=stem)
                if reference_pose is None:
                    reference_pose = points[0].copy(), bodies[0].copy()
                np.testing.assert_array_equal(points[0], reference_pose[0], err_msg=stem)
                np.testing.assert_array_equal(bodies[0], reference_pose[1], err_msg=stem)
                if failure is not None:
                    np.testing.assert_array_equal(
                        points[failure:], np.broadcast_to(points[last], points[failure:].shape)
                    )
                    np.testing.assert_array_equal(
                        bodies[failure:], np.broadcast_to(bodies[last], bodies[failure:].shape)
                    )
                if scale is not None:
                    for family in FAMILIES:
                        rho = archive[f"initial_{family}_rho"].astype(np.float64)
                        material_k = archive[f"initial_{family}_k"]
                        assert np.isfinite(rho).all() and (rho >= 0).all() and (rho > 0).any(), (stem, family)
                        normalized = rho / scale
                        if family not in material_metrics:
                            material_metrics[family] = normalized.copy(), material_k.copy()
                        np.testing.assert_allclose(normalized, material_metrics[family][0], rtol=2e-6, atol=0)
                        np.testing.assert_array_equal(material_k, material_metrics[family][1])
                        if family not in reference_metrics:
                            reference_metrics[family] = normalized.copy()
                        np.testing.assert_allclose(normalized, reference_metrics[family], rtol=2e-6, atol=0)
                        active = (rho > 0) & (material_k > 0)
                        ratio = rho[active] / material_k[active].astype(np.float64)
                        initial = data["initial_metrics"][family]
                        assert initial["active_rows"] == int(active.sum()), (stem, family)
                        for name, values in (("rho_over_k", ratio), ("effective_over_k", ratio / (1 + ratio))):
                            for statistic, reducer in (("min", np.min), ("median", np.median), ("max", np.max)):
                                np.testing.assert_allclose(initial[name][statistic], reducer(values), rtol=2e-6)
            with (directory / f"{stem}.csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            assert len(csv_rows) == len(rows), stem
            for frame, row in enumerate(rows):
                assert row["frame"] == int(csv_rows[frame]["frame"]) == frame, stem
                assert row["valid"] == bool(valid[frame]) and abs(row["time_s"] - frame / 60) < 1e-6, stem
                if frame == 0 or not valid[frame]:
                    assert row["original_rms_N"] is None, (stem, frame)
                    continue
                assert np.isfinite(row["original_rms_N"]) and row["original_rms_N"] >= 0, (stem, frame)
                assert row["self_contact_vt_candidates"] <= data["self_contact_vt_capacity"], stem
                assert row["self_contact_ee_candidates"] <= data["self_contact_ee_capacity"], stem
                for family in FAMILIES:
                    stats = row[family]
                    assert stats["floor_fraction"] is None, (stem, frame, family)
                    if scale is None:
                        assert stats["active_rows"] == 0 and stats["effective_over_k"]["median"] == 1, stem
                    elif stats["active_rows"]:
                        for statistic in ("min", "max"):
                            ratio = stats["rho_over_k"][statistic]
                            assert ratio > 0, (stem, frame, family)
                            np.testing.assert_allclose(
                                stats["effective_over_k"][statistic], ratio / (1 + ratio), rtol=2e-6, atol=1e-12
                            )
                        for name in ("rho_over_k", "effective_over_k"):
                            assert stats[name]["min"] <= stats[name]["median"] <= stats[name]["max"], stem
            wiggle = [row for row in rows[61:] if row["valid"]]
            results.append(
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
                }
            )
    return {
        "passed": True,
        "cases": len(results),
        "requested_snapshots": 15 * 361,
        "initial_rho_material_independence": True,
        "initial_rho_scale_linearity": True,
        "source_sha256": source_hashes,
        "production_source_sha256": production_hashes,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent / "results-inertia-video-sweep"
    )
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
