# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit matched trajectories and material-multiplier penalties in the dense floor sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STIFFNESSES = (1000, 100000, 10000000)
FLOORS = (90, 60, 30, 20, 10, 9, 6, 3, 2, 1, 0.6, 0.3, 0.1)
MODES = {"off": None, **{f"floor{value:g}".replace(".", "p"): value for value in FLOORS}}
FAMILIES = ("tri_stretch", "tri_area", "bend")
RESIDUALS = (
    "original_rms_N",
    "alm_rms_N",
    "constitutive_gap_rms_N",
    "elastic_rms_N",
    "damping_rms_N",
    "contact_rms_N",
    "inertia_rms_N",
    "free_vertices",
    "original_max_N",
    "iterate_change_rms_m",
)
ROOT = Path(__file__).resolve().parent


def _check_sources(hashes, root):
    for name, expected in hashes.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, (name, "source changed since simulation")


def _check_stats(stats, floor, context):
    count = stats["active_rows"]
    assert isinstance(count, int) and count >= 0, context
    if floor is None:
        assert count == 0 and stats["rho_over_k"] is None and stats["floor_fraction"] is None, context
        assert stats["effective_over_k"] == dict.fromkeys(("min", "median", "max"), 1.0), context
    elif count == 0:
        assert all(stats[key] is None for key in ("rho_over_k", "effective_over_k", "floor_fraction")), context
    else:
        fraction = stats["floor_fraction"]
        assert fraction is not None and 0 <= fraction <= 1, context
        for name in ("rho_over_k", "effective_over_k"):
            values = stats[name]
            assert all(np.isfinite(value) for value in values.values()), context
            assert 0 < values["min"] <= values["median"] <= values["max"], context
        assert stats["rho_over_k"]["min"] >= floor * (1 - 2e-6), context
        for statistic in ("min", "max"):
            ratio = stats["rho_over_k"][statistic]
            np.testing.assert_allclose(stats["effective_over_k"][statistic], ratio / (1 + ratio), rtol=2e-6)
        if stats["rho_over_k"]["min"] > floor * (1 + 3e-6):
            assert fraction == 0, context
        if abs(stats["rho_over_k"]["max"] - floor) < floor * 1e-6:
            assert fraction == 1, context


def _check_initial_metrics(archive, data, floor, reference_metrics, material_metrics, stem):
    for family in FAMILIES:
        rho = archive[f"initial_{family}_rho"]
        inertia = archive[f"initial_{family}_inertia_rho"]
        stiffness = archive[f"initial_{family}_k"]
        count = data["edge_count"] if family == "bend" else data["triangle_count"]
        assert rho.shape == inertia.shape == stiffness.shape == (count,), (stem, family)
        assert rho.dtype == inertia.dtype == stiffness.dtype == np.float32, (stem, family)
        for values in (rho, inertia, stiffness):
            assert np.isfinite(values).all() and (values >= 0).all(), (stem, family)
        active = inertia > 0
        assert active.any() and (stiffness[active] > 0).all(), (stem, family)
        floor_values = np.float32(floor) * stiffness
        bounds = np.finfo(np.float32)
        expected = np.where(
            active, np.clip(np.maximum(inertia, floor_values), bounds.smallest_subnormal, bounds.max), 0
        )
        np.testing.assert_allclose(rho, expected, rtol=2e-6, atol=0, err_msg=f"{stem}: {family} floor policy")
        np.testing.assert_array_equal(rho > 0, active, err_msg=f"{stem}: {family} retired rows")
        if family not in reference_metrics:
            reference_metrics[family] = inertia.copy(), stiffness.copy()
        np.testing.assert_array_equal(inertia, reference_metrics[family][0], err_msg=f"{stem}: {family} native inertia")
        if family not in material_metrics:
            material_metrics[family] = stiffness.copy()
        np.testing.assert_array_equal(stiffness, material_metrics[family], err_msg=f"{stem}: {family} stiffness")
        if family != "bend":
            expected_k = data["stiffness"] * (1.2 if family == "tri_area" else 1.0)
            np.testing.assert_array_equal(stiffness, np.full(count, expected_k, dtype=np.float32))
        else:
            np.testing.assert_array_equal(
                stiffness, reference_metrics[family][1], err_msg=f"{stem}: fixed bend stiffness"
            )
        ratio = rho[active].astype(np.float64) / stiffness[active].astype(np.float64)
        stats = data["initial_metrics"][family]
        _check_stats(stats, floor, (stem, family, "initial"))
        assert stats["active_rows"] == int(active.sum()), (stem, family)
        expected_fraction = np.mean(np.isclose(rho[active], floor_values[active], rtol=2e-6, atol=0))
        np.testing.assert_allclose(stats["floor_fraction"], expected_fraction, rtol=2e-6, atol=0)
        for name, values in (("rho_over_k", ratio), ("effective_over_k", ratio / (1 + ratio))):
            for statistic, reducer in (("min", np.min), ("median", np.median), ("max", np.max)):
                np.testing.assert_allclose(stats[name][statistic], reducer(values), rtol=2e-6, atol=0)


def _check_csv(row, csv_row, context):
    flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for family in FAMILIES:
        stats = row[family]
        for name in ("active_rows", "floor_fraction"):
            flat[f"{family}_{name}"] = stats[name]
        for name in ("rho_over_k", "effective_over_k"):
            for statistic in ("min", "median", "max"):
                flat[f"{family}_{name}_{statistic}"] = stats[name][statistic] if stats[name] else None
    assert flat.keys() == csv_row.keys(), context
    for name, value in flat.items():
        if value is None:
            assert csv_row[name] == "", (context, name)
        elif isinstance(value, bool | str | int):
            assert csv_row[name] == str(value), (context, name)
        else:
            np.testing.assert_allclose(float(csv_row[name]), value, rtol=2e-6, atol=1e-12, err_msg=f"{context}: {name}")


def validate(directory, *, frames=360, stiffnesses=STIFFNESSES, modes=MODES):
    """Check every requested case, floor application, diagnostic row, and retained trajectory."""
    results = []
    reference_pose = source_hashes = production_hashes = storage = reference_params = None
    reference_metrics = {}
    contact_demand = {"vt": [], "ee": []}
    for stiffness in stiffnesses:
        model_hash = state_hash = None
        material_metrics = {}
        for mode, floor in modes.items():
            stem = f"ke{stiffness}_{mode}"
            data = json.loads((directory / f"{stem}.json").read_text())
            rows = data["rows"]
            assert data["mode"] == mode and data["stiffness"] == stiffness, stem
            assert data["alm"] == (floor is not None) and data["floor_multiplier"] == floor, stem
            assert data["experiment_rho_override"] == (floor is not None) and data["rho_scale"] == 1, stem
            assert data["native_metric_preparation"] == (floor is None) and "Dimensionless" in data["floor_units"], stem
            assert data["initial_metric_pose_sha256"] == data["initial_state_sha256"], stem
            assert "Retained" in data["history_policy"] and "no checkpoint restarts" in data["history_policy"], stem
            params = data["params"].copy()
            assert params.pop("cloth_tri_ke") == stiffness and params.pop("cloth_tri_ka") == 0.2 * stiffness, stem
            assert params["cloth_edge_ke"] == 200, stem
            assert params["cloth_tri_kd"] == 0.1 and params["cloth_edge_kd"] == 0.02, stem
            if reference_params is None:
                reference_params = params
            assert params == reference_params, stem
            assert data["frames"] == frames and data["snapshot_count"] == len(rows) == frames + 1, stem
            assert data["fps"] == 60 and data["dt_s"] == 1 / 600, stem
            assert data["substeps_per_frame"] == data["iterations_per_step"] == 10, stem
            assert data["solver_steps_requested"] == frames * 10, stem
            assert 0 <= data["max_pin_error_m"] < 1e-6, stem
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
            assert case_storage == storage and storage[0] == 16 and all(value > 0 for value in storage), stem
            if source_hashes is None:
                source_hashes, production_hashes = data["source_sha256"], data["production_source_sha256"]
                assert set(source_hashes) == {
                    "run_rho_video_case.py",
                    "cloth_residual.py",
                    "run_case.py",
                    "run_floor_sweep.py",
                    "run_dense_floor_video_case.py",
                }, stem
                assert set(production_hashes) == {
                    "newton/_src/solvers/vbd/particle_alm_kernels.py",
                    "newton/_src/solvers/vbd/particle_vbd_kernels.py",
                    "newton/_src/solvers/vbd/solver_vbd.py",
                }, stem
                _check_sources(source_hashes, ROOT)
                _check_sources(production_hashes, ROOT.parent.parent)
            assert data["source_sha256"] == source_hashes and data["production_source_sha256"] == production_hashes, (
                stem
            )
            assert (
                data["geometry_source_sha256"]
                == hashlib.sha256((ROOT / "sources/bag_parent.py").read_bytes()).hexdigest()
            ), stem
            if model_hash is None:
                model_hash, state_hash = data["model_sha256"], data["initial_state_sha256"]
            assert (data["model_sha256"], data["initial_state_sha256"]) == (model_hash, state_hash), stem
            failure = data["failure_frame"]
            assert failure is None or (isinstance(failure, int) and 1 <= failure <= frames), stem
            last = frames if failure is None else failure - 1
            assert data["last_valid_frame"] == last and data["solver_steps_completed_valid"] == last * 10, stem
            assert data["status"] == ("complete" if failure is None else "failed"), stem
            assert (data["failure_reason"] is None) if failure is None else bool(data["failure_reason"]), stem
            valid = np.arange(frames + 1) <= last
            with np.load(directory / f"{stem}.npz") as archive:
                points, bodies = archive["particle_q"], archive["body_q"]
                assert points.shape == (frames + 1, data["particle_count"], 3), stem
                assert bodies.shape == (frames + 1, data["body_count"], 7), stem
                assert np.isfinite(points).all() and np.isfinite(bodies).all(), stem
                assert np.abs(points).max() <= 10, stem
                np.testing.assert_array_equal(archive["frame"], np.arange(frames + 1), err_msg=stem)
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
                if floor is not None:
                    _check_initial_metrics(archive, data, floor, reference_metrics, material_metrics, stem)
                else:
                    for family in FAMILIES:
                        _check_stats(data["initial_metrics"][family], None, (stem, family, "initial"))
            with (directory / f"{stem}.csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            assert len(csv_rows) == len(rows), stem
            for frame, row in enumerate(rows):
                _check_csv(row, csv_rows[frame], (stem, frame))
                assert row["frame"] == frame and row["valid"] == bool(valid[frame]), stem
                assert abs(row["time_s"] - frame / 60) < 1e-6, stem
                assert row["status"] == ("valid" if valid[frame] else "held_after_failure"), stem
                if not valid[frame]:
                    held = {
                        key: value
                        for key, value in row.items()
                        if key not in (*RESIDUALS, "frame", "time_s", "valid", "status")
                    }
                    assert all(value == rows[last][key] for key, value in held.items()), (stem, frame)
                for name in RESIDUALS:
                    if frame == 0 or not valid[frame]:
                        assert row[name] is None, (stem, frame, name)
                    else:
                        assert np.isfinite(row[name]) and row[name] >= 0, (stem, frame, name)
                assert 0 <= row["pin_error_m"] < 1e-6, (stem, frame)
                assert 0 <= row["self_contact_vt_candidates"] <= data["self_contact_vt_capacity"], stem
                assert 0 <= row["self_contact_ee_candidates"] <= data["self_contact_ee_capacity"], stem
                if valid[frame]:
                    for kind, demand in contact_demand.items():
                        demand.append(row[f"self_contact_{kind}_candidates"])
                np.testing.assert_allclose(
                    row["body_max_abs_translation_m"], np.abs(bodies[frame, :, :3]).max(), rtol=2e-6, atol=1e-12
                )
                if frame > 0:
                    for family in FAMILIES:
                        _check_stats(row[family], floor, (stem, frame, family))
                        count = data["edge_count"] if family == "bend" else data["triangle_count"]
                        assert row[family]["active_rows"] <= count, (stem, frame, family)
            wiggle = [row for row in rows[61:] if row["valid"]]
            results.append(
                {
                    "case": stem,
                    "floor_multiplier": floor,
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
        "requested_snapshots": len(results) * (frames + 1),
        "valid_snapshots": sum(row["valid_snapshots"] for row in results),
        "held_snapshots": sum(frames + 1 - row["valid_snapshots"] for row in results),
        "self_contact_capacity": dict(zip(("vt", "ee"), storage[1:3], strict=True)),
        "self_contact_demand": {
            kind: {"min": min(values), "max": max(values)} for kind, values in contact_demand.items()
        },
        "floor_policy": "rho = max(native inertia rho, floor_multiplier * material row stiffness)",
        "initial_floor_policy_verified": True,
        "initial_native_inertia_material_independence": True,
        "source_sha256": source_hashes,
        "production_source_sha256": production_hashes,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results-dense-floor-video-sweep")
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
