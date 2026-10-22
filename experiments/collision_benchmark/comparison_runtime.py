"""Solver-mode isolation, substep observation and CUDA event timing."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import warp as wp

import newton.solvers

OriginalSolverVBD = newton.solvers.SolverVBD
MODE = False
ACTIVE = None

CONTACT_OPTIONS = {
    "particle_self_contact_method": "penalty_dat",
    "rigid_compliant_alm": False,
    "rigid_contact_hard": False,
}


def paired_options(mode):
    return {
        **CONTACT_OPTIONS,
        "particle_elasticity_alm": bool(mode),
        "particle_elasticity_alm_deviatoric": True,
        "particle_elasticity_alm_rho_scale": 1.0,
    }


class ObservedSolverVBD(OriginalSolverVBD):
    def __init__(self, model, **kwargs):
        self.comparison_original_options = dict(kwargs)
        kwargs.update(paired_options(MODE))
        self.comparison_options = dict(kwargs)
        super().__init__(model, **kwargs)
        self.comparison_graphs = {}
        self.comparison_timing_ms = []
        self.comparison_compile_s = 0.0
        self.comparison_graph_errors = []
        self.comparison_calls = 0
        if model.device.is_cuda:
            self.comparison_event_start = wp.Event(model.device, enable_timing=True)
            self.comparison_event_end = wp.Event(model.device, enable_timing=True)

    def step(self, state_in, state_out, control, contacts, dt):
        if ACTIVE is None:
            return super().step(state_in, state_out, control, contacts, dt)
        key = (id(state_in), id(state_out), id(control), id(contacts), float(dt))
        if self.model.device.is_cuda:
            if key not in self.comparison_graphs:
                start = time.monotonic()
                with wp.ScopedCapture(device=self.model.device) as capture:
                    super().step(state_in, state_out, control, contacts, dt)
                self.comparison_graphs[key] = capture.graph
                self.comparison_compile_s += time.monotonic() - start
            wp.record_event(self.comparison_event_start)
            wp.capture_launch(self.comparison_graphs[key])
            wp.record_event(self.comparison_event_end)
            elapsed_ms = wp.get_event_elapsed_time(self.comparison_event_start, self.comparison_event_end)
        else:
            start = time.monotonic()
            super().step(state_in, state_out, control, contacts, dt)
            elapsed_ms = (time.monotonic() - start) * 1000
        self.comparison_calls += 1
        # Initial replay has cold caches. Exclude the first two physical steps from timing.
        if self.comparison_calls > 2:
            self.comparison_timing_ms.append(elapsed_ms)
        ACTIVE.after_substep(state_out, dt)

    def timing(self):
        values = np.asarray(self.comparison_timing_ms, dtype=float)
        return {
            "scope": "VBD solver.step GPU time; excludes external collision detection, drivers, observation and rendering",
            "method": "CUDA event pair around captured solver.step; synchronized before observation",
            "count": int(values.size),
            "excluded_initial_substeps": min(2, self.comparison_calls),
            "mean_substep_ms": float(values.mean()) if values.size else None,
            "total_solver_s": float(values.sum() / 1000),
            "graph_capture_s": self.comparison_compile_s,
            "graph_count": len(self.comparison_graphs),
        }


def install_patch():
    newton.solvers.SolverVBD = ObservedSolverVBD


def scene_solver(scene):
    raw = getattr(scene, "raw_scene", scene)
    return raw.example.soft_solver if hasattr(raw, "example") else raw.solver


def replace_solver(scene, mode):
    global MODE  # noqa: PLW0603 -- isolated one-run process mode
    MODE = bool(mode)
    raw = getattr(scene, "raw_scene", scene)
    old = scene_solver(scene)
    new = ObservedSolverVBD(old.model, **old.comparison_options)
    # Constructor allocates a new history for the common task starting state.
    if hasattr(raw, "example"):
        raw.example.soft_solver = new
    else:
        raw.solver = new
    if hasattr(raw, "comparison_start"):
        raw.comparison_start["history_reset_required"] = False
        raw.comparison_start["installed_particle_elasticity_alm"] = bool(mode)
        raw.comparison_start["history_reset_method"] = "fresh solver construction after common state restoration"
    return new


def _array_hash(value):
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    if array.dtype.kind in "OUSV":
        return None
    return hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()).hexdigest()


def input_fingerprint(scene):
    raw = getattr(scene, "raw_scene", scene)
    model = raw.model
    hashes = {}
    for name in (
        "particle_q",
        "particle_qd",
        "particle_mass",
        "particle_inv_mass",
        "particle_radius",
        "particle_flags",
        "gravity",
        "tri_indices",
        "tri_materials",
        "tri_poses",
        "edge_indices",
        "edge_bending_properties",
        "tet_indices",
        "tet_materials",
        "tet_poses",
        "spring_indices",
        "spring_stiffness",
        "shape_body",
        "shape_type",
        "shape_transform",
        "shape_scale",
        "shape_margin",
        "shape_gap",
        "shape_flags",
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_mu",
        "body_mass",
        "body_com",
        "body_inertia",
        "body_inv_inertia",
        "body_flags",
        "body_q",
        "body_qd",
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_target_ke",
        "joint_target_kd",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_limit_ke",
        "joint_limit_kd",
    ):
        value = getattr(model, name, None)
        if value is not None:
            hashes["model." + name] = _array_hash(value)
    for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
        value = getattr(scene.state, name, None)
        if value is not None:
            hashes["state." + name] = _array_hash(value)
    drivers = list(getattr(raw, "drivers", []))
    if hasattr(raw, "driver"):
        drivers.append(raw.driver)
    for i, driver in enumerate(drivers):
        for name, value in vars(driver).items():
            if hasattr(value, "numpy"):
                hashes[f"driver{i}.{name}"] = _array_hash(value)
    for i, mesh in enumerate(getattr(model, "shape_source", [])):
        if mesh is not None:
            for name in ("vertices", "indices"):
                value = getattr(mesh, name, None)
                if value is not None:
                    hashes[f"shape{i}.{name}"] = _array_hash(value)
    if hasattr(raw, "example"):
        for name in ("targets", "robot_key_poses_time", "robot_key_poses", "ik_iters", "ik_joint_q", "target_joint_q"):
            value = getattr(raw.example, name, None)
            if value is not None:
                hashes[f"example.{name}"] = _array_hash(value)
    for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
        hashes[name] = float(getattr(model, name))
    encoded = json.dumps(hashes, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "fields": hashes}
