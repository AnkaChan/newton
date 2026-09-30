# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one retained-history bag trajectory with an experiment-local rho floor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from cloth_residual import bend_metrics, triangle_metrics
from metrics import _build_topology, _dihedral_angles, _measure, _triangle_angles
from run_case import ROOT, Bag, drive_pins, pin_motion
from run_floor_sweep import FIELDS, MODES, Measurement, fingerprint

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo, build_tri_mesh_collision_info
from newton._src.solvers.vbd.particle_alm_kernels import (
    create_particle_elasticity_alm_state,
    prepare_particle_elasticity_alm,
)

SELF_CONTACT_STORAGE_MULTIPLIER = 16


@wp.kernel
def record_contact_demand(counters: wp.array[int], peaks: wp.array[int]):
    peaks[0] = wp.max(peaks[0], counters[0])
    peaks[1] = peaks[1] | counters[1]
    peaks[2] = wp.max(peaks[2], counters[2])
    peaks[3] = peaks[3] | counters[3]


class TrajectoryBag(Bag):
    def __init__(self, stiffness, mode, native_floor9=False, *, native_rho_scale=None):
        self.rho_scale = 1.0 if native_rho_scale is None else native_rho_scale
        super().__init__(stiffness, mode != "off", rho_scale=self.rho_scale)
        self.mode = mode
        self.floor = None if native_floor9 or native_rho_scale is not None else MODES[mode]
        self.floor_multiplier = None if native_rho_scale is not None else MODES[mode]
        self.native_floor9 = native_floor9
        self.measure_final_substep = False
        detector = self.solver.trimesh_collision_detector
        detector.vertex_collision_buffer_pre_alloc *= SELF_CONTACT_STORAGE_MULTIPLIER
        detector.edge_collision_buffer_pre_alloc *= SELF_CONTACT_STORAGE_MULTIPLIER
        collision_info = build_tri_mesh_collision_info(
            self.model.particle_count,
            self.model.tri_count,
            self.model.edge_count,
            vertex_collision_buffer_pre_alloc=detector.vertex_collision_buffer_pre_alloc,
            edge_collision_buffer_pre_alloc=detector.edge_collision_buffer_pre_alloc,
            record_triangle_contacting_vertices=detector.record_triangle_contacting_vertices,
            device=self.model.device,
        )
        detector._bind_external_buffers(collision_info)
        self.solver.trimesh_collision_info = wp.array(
            [collision_info], dtype=TriMeshCollisionInfo, device=self.model.device
        )
        # Keep the original launch size: native kernels stride over all actual pairs.
        # Enlarging storage need not create sixteen times as many idle threads.
        self.contact_peaks = wp.zeros(4, dtype=int, device=self.model.device)
        detect = self.solver._collision_detection_penetration_free

        def detect_with_demand(state):
            detect(state)
            wp.launch(
                record_contact_demand,
                1,
                inputs=[detector.collision_info.counters, self.contact_peaks],
                device=self.model.device,
            )

        self.solver._collision_detection_penetration_free = detect_with_demand
        self.measurement = Measurement(self, None)
        self.measurement.rows = wp.zeros((1, len(FIELDS)), dtype=wp.float64, device=self.model.device)
        initialize = self.solver._initialize_particles
        iterate = self.solver._solve_particle_iteration

        def initialize_with_floor(state_in, state_out, dt):
            initialize(state_in, state_out, dt)
            if self.floor is not None:
                m, s = self.model, self.solver
                wp.launch(
                    triangle_metrics,
                    m.tri_count,
                    inputs=[
                        s.particle_q_prev,
                        m.tri_indices,
                        m.tri_poses,
                        m.tri_areas,
                        m.tri_materials,
                        m.particle_inv_mass,
                        m.particle_flags,
                        dt,
                        self.floor,
                        s._particle_elasticity_alm_state,
                    ],
                    device=m.device,
                )
                wp.launch(
                    bend_metrics,
                    m.edge_count,
                    inputs=[
                        s.particle_q_prev,
                        m.edge_indices,
                        m.edge_rest_length,
                        m.edge_bending_properties,
                        m.particle_inv_mass,
                        m.particle_flags,
                        dt,
                        self.floor,
                        s._particle_elasticity_alm_state,
                    ],
                    device=m.device,
                )

        def iterate_with_measurement(state_in, state_out, contacts, dt, iter_num):
            measure = self.measure_final_substep and iter_num == self.solver.iterations - 1
            if measure:
                wp.copy(self.measurement.previous, state_in.particle_q)
            iterate(state_in, state_out, contacts, dt, iter_num)
            if measure:
                self.measurement.rows.zero_()
                self.measurement.record(state_in, 0, dt)

        if self.floor is not None:
            self.solver._initialize_particles = initialize_with_floor
        self.solver._solve_particle_iteration = iterate_with_measurement

    def simulate(self):
        self.contact_peaks.zero_()
        for substep in range(10):
            self.measure_final_substep = substep == 9
            wp.launch(
                drive_pins,
                len(self.pins),
                inputs=[self.pins, self.rest, self.motion, self.state_0.particle_q, self.state_0.particle_qd],
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def residual(self):
        raw = self.measurement.rows.numpy()[0]
        values = raw.copy()
        for col in (*range(7), 9):
            values[col] = np.sqrt(raw[col] / raw[7])
        return dict(zip(FIELDS, map(float, values), strict=True))


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def range_stats(values):
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def rho_stats(sim, prepared, state=None):
    state = sim.solver._particle_elasticity_alm_state if state is None else state
    model = sim.model
    materials = model.tri_materials.numpy()
    families = (
        ("tri_stretch", state.tri_rho_stretch, materials[:, 0]),
        ("tri_area", state.tri_rho_area, materials[:, 0] + materials[:, 1]),
        ("bend", state.bend_rho, model.edge_bending_properties.numpy()[:, 0] * model.edge_rest_length.numpy()),
    )
    result = {}
    for name, array, k in families:
        if sim.mode == "off":
            result[name] = {
                "active_rows": 0,
                "rho_over_k": None,
                "effective_over_k": {"min": 1.0, "median": 1.0, "max": 1.0},
                "floor_fraction": None,
            }
            continue
        if not prepared:
            result[name] = {
                "active_rows": 0,
                "rho_over_k": None,
                "effective_over_k": None,
                "floor_fraction": None,
            }
            continue
        rho = array.numpy()
        active = (rho > 0.0) & (k > 0.0)
        if not np.isfinite(rho).all():
            raise FloatingPointError(f"Nonfinite {name} rho")
        if not active.any():
            result[name] = {
                "active_rows": 0,
                "rho_over_k": None,
                "effective_over_k": None,
                "floor_fraction": None,
            }
            continue
        # Host-side analysis is double; native metrics and any override are float32.
        ratio = rho[active].astype(np.float64) / k[active].astype(np.float64)
        floor = sim.floor_multiplier
        fraction = None
        if floor is not None:
            floor_values = np.float32(floor) * k[active]
            dominated = np.isclose(rho[active], floor_values, rtol=2e-6, atol=0) if floor > 0 else np.zeros_like(ratio)
            if floor > 0:
                assert np.all(ratio >= floor * (1 - 2e-6)), (name, ratio.min(), floor)
            fraction = float(np.mean(dominated))
        result[name] = {
            "active_rows": int(active.sum()),
            "rho_over_k": range_stats(ratio),
            "effective_over_k": range_stats(ratio / (1.0 + ratio)),
            "floor_fraction": fraction,
        }
    return result


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.7g}") if np.isfinite(value) else None
    return value


def flatten(row):
    flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for name in ("tri_stretch", "tri_area", "bend"):
        family = row[name]
        for key in ("active_rows", "floor_fraction"):
            flat[f"{name}_{key}"] = family[key]
        for key in ("rho_over_k", "effective_over_k"):
            for stat in ("min", "median", "max"):
                flat[f"{name}_{key}_{stat}"] = family[key][stat] if family[key] else None
    return flat


def run(args, *, native_rho_scale=None):
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    started = time.monotonic()
    sim = TrajectoryBag(args.stiffness, args.mode, args.native_floor9, native_rho_scale=native_rho_scale)
    model = sim.model
    rest = sim.rest.numpy().astype(np.float64)
    topology = _build_topology(model.tri_indices.numpy())
    rest_lengths = np.linalg.norm(rest[topology.edges[:, 1]] - rest[topology.edges[:, 0]], axis=1)
    rest_angles = _triangle_angles(rest, topology.faces)
    rest_dihedrals = _dihedral_angles(rest, topology.faces, topology.bend_pairs)
    model_hash = digest_arrays(
        [
            model.particle_q,
            model.particle_mass,
            model.particle_flags,
            model.tri_indices,
            model.tri_materials,
            model.edge_bending_properties,
            model.body_q,
            model.body_mass,
        ]
    )
    initial_hash = fingerprint(sim.state_0)
    initial_metrics, initial_arrays = None, {}
    if native_rho_scale is not None:
        initial_state = create_particle_elasticity_alm_state(model, args.mode != "off", False, native_rho_scale)
        retained = sim.solver._particle_elasticity_alm_state
        histories_before = {
            name: getattr(retained, name).numpy()
            for name in ("tri_lambda_stretch", "tri_lambda_area", "bend_lambda", "tri_pending", "bend_pending")
        }
        prepare_particle_elasticity_alm(model, sim.state_0.particle_q, sim.dt, initial_state)
        initial_metrics = rho_stats(sim, True, state=initial_state)
        for name, before in histories_before.items():
            np.testing.assert_array_equal(before, getattr(retained, name).numpy())
        if args.mode != "off":
            materials = model.tri_materials.numpy()
            initial_arrays = {
                "initial_tri_stretch_rho": initial_state.tri_rho_stretch.numpy(),
                "initial_tri_stretch_k": materials[:, 0],
                "initial_tri_area_rho": initial_state.tri_rho_area.numpy(),
                "initial_tri_area_k": materials[:, 0] + materials[:, 1],
                "initial_bend_rho": initial_state.bend_rho.numpy(),
                "initial_bend_k": model.edge_bending_properties.numpy()[:, 0] * model.edge_rest_length.numpy(),
            }
    sim.capture()
    assert fingerprint(sim.state_0) == initial_hash, "Capture changed physical state"
    capture_seconds = time.monotonic() - started
    rows, points_saved, bodies_saved, valid_saved = [], [], [], []
    failure_frame, failure_reason = None, None
    max_pin_error = 0.0
    for frame in range(args.frames + 1):
        if frame > 0 and failure_frame is None:
            try:
                sim.step()
            except Exception as error:
                failure_frame, failure_reason = frame, f"{type(error).__name__}: {error}"
        if failure_frame is None:
            try:
                points = sim.state_0.particle_q.numpy()
                body_q = sim.state_0.body_q.numpy()
                qd = sim.state_0.particle_qd.numpy()
                body_qd = sim.state_0.body_qd.numpy()
                if not all(np.isfinite(array).all() for array in (points, body_q, qd, body_qd)):
                    raise FloatingPointError("Nonfinite position or velocity")
                if np.abs(points).max() > 10.0:
                    raise FloatingPointError("Cloth coordinate exceeded the recorded 10 m runaway cutoff")
                stretch, shear, bend = _measure(
                    points.astype(np.float64), topology, rest_lengths, rest_angles, rest_dihedrals
                )
                if not np.isfinite([stretch, shear, bend]).all():
                    raise FloatingPointError("Nonfinite geometry diagnostic")
                residual = sim.residual() if frame > 0 else dict.fromkeys(FIELDS)
                if frame > 0 and not all(np.isfinite(v) for v in residual.values()):
                    raise FloatingPointError("Nonfinite diagnostic force residual")
                expected = rest[sim.info["top_global_indices"]].copy()
                expected[:, 0] += pin_motion(frame / 60.0)[0]
                pin_error = float(np.abs(points[sim.info["top_global_indices"]] - expected).max())
                max_pin_error = max(max_pin_error, pin_error)
                if pin_error >= 1e-6:
                    raise FloatingPointError(f"Pinned-position error {pin_error} m")
                vt_count, vt_overflow, ee_count, ee_overflow = map(int, sim.contact_peaks.numpy())
                if vt_overflow or ee_overflow:
                    raise RuntimeError(f"Self-contact capacity exceeded: VT={vt_count}, EE={ee_count}")
                row = {
                    "frame": frame,
                    "time_s": frame / 60.0,
                    "valid": True,
                    "status": "valid",
                    "stretch_score": stretch,
                    "shear_score": shear,
                    "bend_score": bend,
                    **residual,
                    "self_contact_vt_candidates": vt_count,
                    "self_contact_ee_candidates": ee_count,
                    "body_max_abs_translation_m": float(np.abs(body_q[:, :3]).max()),
                    "soft_contact_count": int(sim.contacts.soft_contact_count.numpy()[0]),
                    "pin_error_m": pin_error,
                    **rho_stats(sim, frame > 0),
                }
            except Exception as error:
                failure_frame, failure_reason = frame, f"{type(error).__name__}: {error}"
        if failure_frame is not None:
            if not rows:
                raise RuntimeError(f"Initial model invalid: {failure_reason}")
            points, body_q = points_saved[-1], bodies_saved[-1]
            row = {
                **rows[-1],
                "frame": frame,
                "time_s": frame / 60.0,
                "valid": False,
                "status": "held_after_failure",
                **dict.fromkeys(FIELDS),
            }
        rows.append(row)
        points_saved.append(points.copy())
        bodies_saved.append(body_q.copy())
        valid_saved.append(failure_frame is None)
        if frame % 60 == 0 or frame == failure_frame:
            print(
                f"ke={args.stiffness} mode={args.mode} frame={frame} valid={row['valid']} "
                f"stretch={row['stretch_score']:.6g} bend={row['bend_score']:.6g} "
                f"residual={row['original_rms_N']} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            if frame == failure_frame:
                print(failure_reason, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"ke{args.stiffness}_{args.mode}"
    np.savez_compressed(
        args.output / f"{stem}.npz",
        particle_q=np.array(points_saved),
        body_q=np.array(bodies_saved),
        frame=np.arange(args.frames + 1),
        valid=np.array(valid_saved),
        **initial_arrays,
    )
    flat_rows = [flatten(row) for row in rows]
    with (args.output / f"{stem}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    valid_rows = [row for row in rows if row["valid"]]
    metadata = {
        "schema_version": 1,
        "stiffness": args.stiffness,
        "mode": args.mode,
        "alm": args.mode != "off",
        "floor_multiplier": sim.floor_multiplier,
        "native_metric_preparation": native_rho_scale is not None,
        "experiment_rho_override": sim.floor is not None,
        "rho_arithmetic": "float32, using the same bounded product-ratio helper as native preparation",
        "params": sim.params,
        "frames": args.frames,
        "snapshot_count": args.frames + 1,
        "solver_steps_requested": args.frames * 10,
        "solver_steps_completed_valid": (len(valid_rows) - 1) * 10,
        "iterations_per_step": 10,
        "substeps_per_frame": 10,
        "fps": 60,
        "dt_s": sim.dt,
        "rho_scale": sim.rho_scale,
        "history_policy": "Retained ALM stress history across every substep and frame; no checkpoint restarts",
        "status": "complete" if failure_frame is None else "failed",
        "failure_frame": failure_frame,
        "failure_reason": failure_reason,
        "last_valid_frame": valid_rows[-1]["frame"],
        "failure_padding": "Hold last valid finite pose with valid=False; metrics are held except residuals become null",
        "runaway_cutoff_abs_cloth_coordinate_m": 10.0,
        "escaped_rigid_contents_policy": "Finite rigid bodies may escape and fall; their translation has no cutoff",
        "self_contact_storage_multiplier": SELF_CONTACT_STORAGE_MULTIPLIER,
        "self_contact_vt_capacity": sim.solver.trimesh_collision_detector.vt_pairs.shape[0],
        "self_contact_ee_capacity": sim.solver.trimesh_collision_detector.ee_pairs.shape[0],
        "self_contact_capacity_check": "Peak pair demand and any overflow flag across every collision query of all ten substeps each frame",
        "self_contact_kernel_launch_size": sim.solver.particle_self_contact_evaluation_kernel_launch_size,
        "elapsed_seconds": time.monotonic() - started,
        "capture_seconds": capture_seconds,
        "newton_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("run_rho_video_case.py", "cloth_residual.py", "run_case.py", "run_floor_sweep.py")
        },
        "command": sys.argv,
        "device": str(model.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_sha256": model_hash,
        "initial_state_sha256": initial_hash,
        "geometry_source_sha256": hashlib.sha256((ROOT / "sources/bag_parent.py").read_bytes()).hexdigest(),
        "particle_count": model.particle_count,
        "triangle_count": model.tri_count,
        "edge_count": model.edge_count,
        "pin_count": len(sim.pins),
        "body_count": model.body_count,
        "max_pin_error_m": max_pin_error,
        "residual": "Free-particle original elasticity + damping + fresh native contact + inertia RMS, measured after the final particle iteration of the final substep each frame, before velocity finalization. Float64 independent elastic/damping/inertia evaluation at float32 positions; native float32 contact forces. Excludes rigid-body residual and DAT constraint reactions.",
        "rho_statistics": "Active rows at incoming pose of final substep; reported effective_over_k is scalar reduced-row curvature rho/(k+rho), not the entire geometric Hessian. Floor fraction is the fraction within 2e-6 relative tolerance of floor*k.",
        "metrics_precision": "JSON display values use 7 significant figures; CSV preserves full precision",
    }
    if native_rho_scale is None:
        metadata["native_floor9_without_override"] = args.native_floor9
    else:
        metadata["rho_policy"] = (
            "rho = rho_scale * rho_inertia using native float32 preparation, without a material floor"
        )
        metadata["initial_metrics"] = initial_metrics
        metadata["initial_metric_pose_sha256"] = initial_hash
        metadata["initial_metric_preparation"] = (
            "Native preparation on an independent temporary ALM state at the shared initial pose; solver histories and pending flags verified unchanged"
        )
        metadata["rho_statistics"] = (
            "Active rows at incoming pose of final substep; effective_over_k is scalar reduced-row curvature rho/(k+rho), not the entire geometric Hessian. floor_fraction is null because no material floor is used."
        )
        metadata["source_sha256"]["run_inertia_video_case.py"] = hashlib.sha256(
            (ROOT / "run_inertia_video_case.py").read_bytes()
        ).hexdigest()
        repository = ROOT.parent.parent
        metadata["production_source_sha256"] = {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in (
                "newton/_src/solvers/vbd/particle_alm_kernels.py",
                "newton/_src/solvers/vbd/particle_vbd_kernels.py",
                "newton/_src/solvers/vbd/solver_vbd.py",
            )
        }
    metadata["summary"] = {}
    for key in ("stretch_score", "shear_score", "bend_score", "original_rms_N"):
        values = [row[key] for row in valid_rows if row[key] is not None]
        metadata["summary"][key] = {
            "mean": float(np.mean(values)) if values else None,
            "final_valid": valid_rows[-1][key],
            "peak": max(values) if values else None,
        }
    # Compact row formatting keeps each reviewable JSON below the repository's 500 KiB limit.
    header = json.dumps(metadata, indent=2)[:-2]
    text = header + ',\n  "rows": [\n'
    text += ",\n".join("    " + json.dumps(rounded(row), separators=(",", ":")) for row in rows)
    text += "\n  ]\n}\n"
    (args.output / f"{stem}.json").write_text(text)
    print(f"saved {stem}: {len(valid_rows)}/{len(rows)} valid snapshots; {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiffness", type=int, required=True)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--native-floor9", action="store_true", help="Validation only: skip the rho override")
    parser.add_argument("--output", type=Path, default=ROOT / "results-rho-video-sweep")
    arguments = parser.parse_args()
    if arguments.native_floor9 and arguments.mode != "floor9":
        parser.error("--native-floor9 requires --mode floor9")
    run(arguments)
