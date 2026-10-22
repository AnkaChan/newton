# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original H1 closure from an audited open pose after unconstrained settling.

The parked-to-open robot placement is outside the recorded operation. Both rigid
state buffers start at the same pose. No particle state is prescribed or reset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

from .h1_bag_scenes import H1ClothScene
from .h1_gripper import H1HandDriver

__all__ = ["H1OpenStartScene", "open_start_grasps"]


def open_start_grasps(grasps=None, *, outward_offset_m=0.004):
    """Keep the closed index reference fixed while opening the real thumb.

    The open reference below is a construction frame, not the midpoint of the
    open physical gap. The small outward offset clears the stationary index.
    """
    if grasps is None:
        recipe = os.environ.get("NEWTON_H1_GRASP_RECIPE")
        if not recipe:
            raise ValueError("Provide an explicit audited H1 grasp recipe through grasps or NEWTON_H1_GRASP_RECIPE")
        grasps = json.loads(Path(recipe).read_text())
    result = copy.deepcopy(grasps)
    for grasp in result:
        mode = grasp.get("open_reference_mode", "fixed_closed_index")
        if mode == "explicit_local_reference":
            reference = np.asarray(grasp["open_contact_point_local_m"], dtype=float)
            if reference.shape != (3,) or not np.isfinite(reference).all():
                raise ValueError("Explicit open reference must be a finite local point in metres")
            grasp["open_reference_method"] = (
                "Explicit calibrated local wrist reference supplied by recipe; automatic outward offset is not applied."
            )
            grasp["open_wrist_outward_offset_m"] = None
        elif mode == "fixed_closed_index":
            normal = np.asarray(grasp["contact_normal_local"], dtype=float)
            normal /= np.linalg.norm(normal)
            center = np.asarray(grasp["contact_point_local_m"])
            grasp["open_contact_point_local_m"] = (center - outward_offset_m * normal).tolist()
            grasp["open_reference_method"] = (
                "Closed index contact frame plus declared outward wrist offset; not open-gap midpoint."
            )
            grasp["open_wrist_outward_offset_m"] = float(outward_offset_m)
        else:
            raise ValueError(f"Unknown open reference mode: {mode}")
    return result


class H1OpenStartScene(H1ClothScene):
    """Close original fingers on freely settled cloth from a clear open start."""

    def __init__(self, scene_id, *, grasps=None, outward_offset_m=None, use_cuda_graph=True, **kwargs):
        offset = (
            float(os.environ.get("NEWTON_H1_OPEN_OFFSET", "0.004"))
            if outward_offset_m is None
            else float(outward_offset_m)
        )
        if not 0 <= offset <= 0.020:
            raise ValueError("Open wrist offset must be in [0, 20] mm")
        self._planning_untrimmed = True
        super().__init__(
            scene_id,
            grasps=open_start_grasps(grasps, outward_offset_m=offset),
            use_cuda_graph=False,
            **kwargs,
        )
        self.free_settle_duration = self.settling_time
        self.authored_initial = copy.deepcopy(self.initial)
        self.free_settle_history = []
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            settling_graph = capture.graph
        else:
            settling_graph = None
        for _ in range(round(self.free_settle_duration / self.frame_dt)):
            if settling_graph is None:
                self._simulate()
            else:
                wp.capture_launch(settling_graph)
            self.time += self.frame_dt
            q = self.state.particle_q.numpy()
            self.free_settle_history.append({"time_s": self.time, **self.strain.sample(q)})
        q = self.state.particle_q.numpy()
        if self.state_dir:
            np.savez_compressed(self.state_dir / "free-settled-state.npz", positions=q)
        qd = self.state.particle_qd.numpy().copy()
        self._plan_from_material(q)
        origin = self.times[2]
        times = np.asarray(self.times[2:]) - origin
        drivers = []
        for old, spec, grasp in zip(self.drivers, self.specs, self.grasps, strict=True):
            controls = [grasp["open_controls"]] + [grasp["closed_controls"]] * 7
            controls += [grasp["open_controls"]] * 2 + [grasp.get("release_controls", grasp["open_controls"])] * 2
            drivers.append(
                H1HandDriver(
                    self.model,
                    spec,
                    times=times,
                    positions=old.positions.numpy()[2:],
                    rotations=old.rotations.numpy()[2:],
                    controls=controls,
                )
            )
        self.drivers = drivers
        self.times, self.duration = times.tolist(), float(times[-1])
        self.hold_window = [t - origin for t in self.hold_window]
        self._planning_untrimmed = False
        self.time, self.settling_time = 0.0, 0.0
        self.clock.assign(np.zeros(1, dtype=np.float32))
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1)
        if not np.array_equal(q, self.state.particle_q.numpy()):
            raise AssertionError("Planning or robot initialization changed material positions")
        self.initial = self.strain.sample(q)
        self.recorded_start_audits = [
            d.geometry_audit(self.state, points=q, radii=self.radius, full_surface_contact=True) for d in self.drivers
        ]
        self.near_tip_layer_audits = []
        for target, patch_normal in zip(self.settled_targets, self.patch_normals, strict=True):
            delta = q - target
            normal = np.asarray(patch_normal)
            axial = delta @ normal
            lateral = np.linalg.norm(delta - axial[:, None] * normal, axis=1)
            selected = axial[(lateral < 0.008) & (abs(axial) < 0.025)]
            self.near_tip_layer_audits.append(
                {
                    "tangent_radius_m": 0.008,
                    "normal_search_half_extent_m": 0.025,
                    "vertices": int(len(selected)),
                    "normal_span_m": float(np.ptp(selected)) if len(selected) else None,
                    "normal_min_max_m": [float(selected.min()), float(selected.max())] if len(selected) else None,
                }
            )
        self.planned_path_audit = self._audit_planned_path()
        self.open_start_evidence = {
            "whole_cloth_initial_audits": self.recorded_start_audits,
            "near_tip_layers": self.near_tip_layer_audits,
            "path": self.planned_path_audit,
            "settled_targets_m": [v.tolist() for v in self.settled_targets],
            "recipe": self.grasps,
        }
        if self.state_dir:
            (self.state_dir / "open-start-audit.json").write_text(json.dumps(self.open_start_evidence, indent=2))
        if any(a["minimum_shell_to_robot_distance_m"] < 0.001 - 1e-6 for a in self.recorded_start_audits):
            raise ValueError(f"Audited open start intersects cloth shell or hand margin: {self.recorded_start_audits}")
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1)
        if not np.array_equal(qd, self.state.particle_qd.numpy()):
            raise AssertionError("Planning or robot initialization changed material velocities")
        if not np.array_equal(self.state0.body_q.numpy(), self.state1.body_q.numpy()):
            raise AssertionError("Both rigid state buffers must start at the same open pose")
        self.history, self.audits = [], []
        self.title = f"R7 | H1 pair: {self.name}, audited open start"
        self.description = (
            "Original-size free cloth settles before recording. Both original H1 hands start in a whole-cloth-audited "
            "open pose, then close, draw forward, lift, carry, hold, place, open and withdraw. "
            "The parked-to-open approach is outside this test. All cloth particles remain dynamic; no pins or attachments."
        )
        self.graph = None
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self.graph = capture.graph

    @property
    def phase(self):
        if self._planning_untrimmed:
            return "free settling before recorded operation"
        names = [
            "close original fingers",
            "draw acquired cloth clear of tabletop",
            "lift",
            "raised pause",
            "carry",
            "hold",
            "place",
            "open fingers",
            "extract",
            "withdraw and straighten fingers",
            "released",
        ]
        for end, name in zip(self.times[1:], names, strict=True):
            if self.time < end - 1e-6:
                return name
        return "complete"

    def _audit_planned_path(self):
        original_start = self.settling_time
        if self._planning_untrimmed:
            self.settling_time = self.times[2]
        try:
            return super()._audit_planned_path()
        finally:
            self.settling_time = original_start

    def metrics(self):
        result = super().metrics()
        result.update(
            {
                "operation_change": "Free settling, then real original-finger closure from an audited open start. Parked-to-open approach excluded.",
                "start_mode": "audited open start after free settling",
                "prepared_contact": False,
                "free_settle_duration_s": self.free_settle_duration,
                "free_settle_trajectory": self.free_settle_history,
                "authored_initial": self.authored_initial,
                "recorded_start": self.initial,
                "recorded_start_hand_audits": self.recorded_start_audits,
                "recorded_start_near_tip_layers": self.near_tip_layer_audits,
                "hand_initialization": "Only original hand FK is placed at the open pose in both rigid buffers with zero pose delta. Material positions and velocities are untouched.",
                "open_start_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
        return result
