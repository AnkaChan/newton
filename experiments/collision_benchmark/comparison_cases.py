# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental replay registry and physical-time adapter for VBD comparisons.

The original scenes retain their meshes, material, contacts and motion. A
stationary grasp interval can be extended without slowing physical time.
This module does not choose an elasticity treatment. The comparison runner
installs that treatment after the common baseline preparation is complete.
"""

# ruff: noqa: PLC0415
# Keep registry reads lightweight and defer recipe imports until solver patching.

from __future__ import annotations

import copy
import importlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

__all__ = ["ComparisonScene", "build_case", "load_cases"]

_DEFAULT_REGISTRY = Path(__file__).with_name("cases.json")
_OUTPUT_ENV = ("NEWTON_CLOTH_STATE_DIR", "NEWTON_JACKET_STATE_DIR")


def load_cases(path=None):
    """Read the frozen case registry without importing Newton or Warp."""
    path = Path(path or os.environ.get("NEWTON_COMPARISON_CASES", _DEFAULT_REGISTRY)).resolve()
    data = json.loads(path.read_text())
    cases = {row["case_id"]: row for row in data["cases"]}
    if len(cases) != data["case_count"]:
        raise ValueError("Comparison case IDs must be unique")
    for case in cases.values():
        for name, value in case["env"].items():
            if name.endswith("_RECIPE"):
                case["env"][name] = str((path.parent / value).resolve())
        for record in case["replay"].values():
            record["path"] = str((path.parent / record["path"]).resolve())
    return cases


@contextmanager
def _recipe_environment(values):
    keys = set(values) | set(_OUTPUT_ENV)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _uncaptured_cloth_step(scene):
    """Preserve the original R5 frame update without its unconditional graph."""
    scene.simulate()
    scene.time += scene.frame_dt
    q = scene.state.particle_q.numpy()
    scene.history.append({"time_s": scene.time, "phase": scene.phase, **scene.strain.sample(q)})
    if round(scene.time / scene.frame_dt) % 15 == 0:
        scene.audits.append({"time_s": scene.time, "hands": scene._hand_audits(q)})


def _disable_graphs(scene):
    # Constructors may capture a graph for setup. None is replayed by this
    # adapter, so an observer sees every physical solver.step call.
    for name in ("graph", "_graph"):
        if hasattr(scene, name):
            setattr(scene, name, None)
    if scene.__class__.__name__ == "ClothRobotScene":
        scene.step = MethodType(_uncaptured_cloth_step, scene)
    if hasattr(scene, "example"):
        scene.example.graph = None


def _static_surfaces(scene):
    """Describe actual fixed collision planes/boxes, never visual bounds."""
    import numpy as np
    import warp as wp

    import newton

    model = scene.model
    bodies = model.shape_body.numpy()
    types = model.shape_type.numpy()
    poses = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    flags = model.shape_flags.numpy()
    margins = model.shape_margin.numpy()
    surfaces = []
    for index, body in enumerate(bodies):
        if body != -1 or not int(flags[index]) & int(newton.ShapeFlags.COLLIDE_PARTICLES):
            continue
        pose = poses[index]
        common = {
            "shape_index": index,
            "label": model.shape_label[index],
            "shape_margin_m": float(margins[index]),
            "geometry_source": "model.shape_transform and model.shape_scale",
        }
        if int(types[index]) == int(newton.GeoType.PLANE):
            normal = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0, 0, 1)), dtype=float)
            surfaces.append(
                {**common, "kind": "plane", "normal": normal.tolist(), "offset_m": float(normal @ pose[:3])}
            )
        elif int(types[index]) == int(newton.GeoType.BOX):
            surfaces.append(
                {
                    **common,
                    "kind": "box",
                    "center_m": pose[:3].tolist(),
                    "half_extents_m": scales[index].tolist(),
                    "rotation_xyzw": pose[3:].tolist(),
                }
            )
    return surfaces


class ComparisonScene:
    """Keep elapsed physical time separate from the recipe's motion clock.

    The pause occurs at the middle of an already stationary original interval.
    Each uncaptured source frame runs normally. During the extra hold, its GPU
    clock starts at the same interior point, then advances by one original
    frame. All commands in that interval are constant. Physical solver steps,
    forces, velocities and solver history continue without resets.

    The midpoint has at least a full frame of stationary motion on either side.
    This also makes boundary frames harmless: any subframe timing difference
    lies inside the same constant-command interval, not on an approach or
    release trajectory. Raw smoke histories use motion time; the comparison
    observer must use this adapter's physical time plus its substep offset.
    """

    def __init__(self, raw_scene, case, *, hold_seconds=60.0, surfaces=None):
        self.raw_scene = raw_scene
        self.case = copy.deepcopy(case)
        self.case_id = case["case_id"]
        self.time = 0.0
        self.frame_dt = float(case["frame_dt_s"])
        self.substeps = int(case["substeps"])
        self.dt = self.frame_dt / self.substeps
        self.iterations = int(case["iterations"])
        self._original_duration = float(raw_scene.duration)
        self._hold_start = case["hold_start"]
        self._hold_end = case["hold_end"]
        self._extra_hold = 0.0
        self._pause = None
        if self._hold_start is not None:
            original_hold = self._hold_end - self._hold_start
            if not math.isfinite(hold_seconds) or hold_seconds < original_hold:
                raise ValueError("hold_seconds must be finite and cannot shorten the original hold")
            if original_hold <= 2 * self.frame_dt:
                raise ValueError("Hold plateau must contain more than two original frames")
            self._extra_hold = float(hold_seconds - original_hold)
            self._pause = 0.5 * (self._hold_start + self._hold_end)
        self.duration = self._original_duration + self._extra_hold
        self._frames = 0
        self._clocks = [getattr(raw_scene, name) for name in ("clock", "_clock") if hasattr(raw_scene, name)]
        self.protocol = copy.deepcopy(case)
        self.protocol.update(
            hold_start=self._hold_start,
            hold_end=None if self._hold_end is None else self._hold_end + self._extra_hold,
            duration_s=self.duration,
            dynamic_window=[0.0, self.duration],
            clock_policy="Physical time advances continuously; only the original stationary command clock pauses.",
            extra_hold_s=self._extra_hold,
            raw_metrics_time_basis="original motion time; comparison scores use physical time",
            penetration_surfaces=[] if surfaces is None else surfaces,
        )
        self.protocol["stationary_requested_intervals"] = [
            {"start_s": self.physical_time(a), "end_s": self.physical_time(b), "phase": "declared stationary command"}
            for a, b in case["stationary_windows"]
        ]
        self.protocol["stationary_intervals"] = [
            {
                **interval,
                "start_s": math.ceil(interval["start_s"] / self.dt - 1e-9) * self.dt,
                "end_s": math.floor(interval["end_s"] / self.dt + 1e-9) * self.dt,
                "requested_start_s": interval["start_s"],
                "requested_end_s": interval["end_s"],
            }
            for interval in self.protocol["stationary_requested_intervals"]
        ]
        self.protocol["original_stationary_windows"] = copy.deepcopy(case["stationary_windows"])
        self.protocol["stationary_windows"] = [
            [interval["start_s"], interval["end_s"]] for interval in self.protocol["stationary_intervals"]
        ]
        for source, destination in (
            ("transport_windows", "transport_intervals"),
            ("place_release_windows", "place_release_intervals"),
        ):
            self.protocol[destination] = [
                {"start_s": self.physical_time(a), "end_s": self.physical_time(b)} for a, b in case.get(source, [])
            ]
        if case["applicability"]["analytic_freefall"]:
            self.protocol["gravity_only_interval"] = {
                "start_s": 0.0,
                "end_s": self.duration,
                "gravity_m_s2": [0.0, 0.0, -9.81],
            }
        if hasattr(raw_scene, "cable_bodies"):
            self.protocol["cable_body_indices"] = [int(i) for i in raw_scene.cable_bodies]

    def __getattr__(self, name):
        return getattr(self.raw_scene, name)

    @property
    def motion_time(self):
        return self.motion_time_at(self.time)

    @property
    def phase(self):
        return self.raw_scene.phase

    @property
    def state(self):
        return self.raw_scene.state

    @property
    def solver(self):
        if hasattr(self.raw_scene, "example"):
            return self.raw_scene.example.soft_solver
        return self.raw_scene.solver

    @solver.setter
    def solver(self, value):
        if hasattr(self.raw_scene, "example"):
            self.raw_scene.example.soft_solver = value
        else:
            self.raw_scene.solver = value

    def motion_time_at(self, physical_time):
        if self._pause is None or physical_time <= self._pause:
            return float(physical_time)
        return float(max(self._pause, physical_time - self._extra_hold))

    def physical_time(self, original_time):
        """Map a command boundary after the pause to physical elapsed time."""
        return float(
            original_time + (self._extra_hold if self._pause is not None and original_time > self._pause else 0.0)
        )

    def step(self):
        import numpy as np

        motion_time = self.motion_time
        self.raw_scene.time = motion_time
        for clock in self._clocks:
            clock.assign(np.asarray([motion_time], dtype=np.float32))
        if hasattr(self.raw_scene, "example"):
            self.raw_scene.example.sim_time = motion_time
        self.raw_scene.step()
        self._frames += 1
        self.time = self._frames * self.frame_dt
        self.raw_scene.time = self.motion_time
        if hasattr(self.raw_scene, "example"):
            self.raw_scene.example.sim_time = self.motion_time

    def metrics(self):
        return {"comparison_protocol": self.protocol, "physical_time_s": self.time, "motion_time_s": self.motion_time}


def build_case(case_id, device=None, *, hold_seconds=60.0, solver_kwargs=None):
    """Build a declared case with the runner's baseline SolverVBD factory.

    ``solver_kwargs`` is accepted for contract compatibility. Treatment must
    be installed by the runner after common preparation, not applied here.
    An explicit nonempty value is rejected to avoid silently ignoring it.
    """
    if solver_kwargs:
        raise ValueError("Install solver treatment after build_case; construction uses common baseline preparation")
    cases = load_cases()
    if case_id not in cases:
        raise ValueError(f"Unknown comparison case {case_id!r}")
    case = cases[case_id]
    with _recipe_environment(case["env"]):
        if case["runner"] == "h1":
            from .comparison_h1 import build_h1

            raw = build_h1(case_id, device)
        else:
            module_name = {"soft": "soft_scenes", "cable": "cable_scenes", "manipulation": "manipulation"}[
                case["runner"]
            ]
            module = importlib.import_module(f"experiments.collision_benchmark.{module_name}")
            raw = module.build_scene(case["scene_id"], device=device, use_cuda_graph=False, **case["kwargs"])
    _disable_graphs(raw)
    if not math.isclose(raw.frame_dt, case["frame_dt_s"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Recorded frame interval changed")
    actual_substeps = raw.example.sim_substeps if hasattr(raw, "example") else raw.substeps
    actual_solver = raw.example.soft_solver if hasattr(raw, "example") else raw.solver
    actual_iterations = actual_solver.iterations
    if actual_substeps != case["substeps"] or actual_iterations != case["iterations"]:
        raise ValueError(f"Recorded integration settings changed for {case_id}")
    return ComparisonScene(raw, case, hold_seconds=hold_seconds, surfaces=_static_surfaces(raw))
