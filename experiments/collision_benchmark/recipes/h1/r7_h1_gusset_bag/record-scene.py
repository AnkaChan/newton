# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF005, PLW0108
# Preserve the recorded NumPy additions and standalone dispatch verbatim.

"""Prepared original-H1 contact on the original upright gusset body rim.

No acquisition is claimed. The bag starts undeformed on the unchanged table.
Only the original robot joints and wrist poses are prescribed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

from experiments.collision_benchmark import manipulation, run_smoke
from experiments.collision_benchmark.h1_bag_scenes import (
    H1ClothScene,
    _original_placement,
    default_grasps,
)
from experiments.collision_benchmark.h1_gripper import H1HandDriver

OUT = Path(sys.argv[sys.argv.index("--out") + 1])
OUT.mkdir(parents=True, exist_ok=True)


class PreparedBodyRimScene(H1ClothScene):
    def __init__(self, scene_id, **kwargs):
        _, _, _, original, _ = _original_placement("gusset_bag")
        mid = float((original[:, 2].min() + original[:, 2].max()) / 2)
        self.actual_setup = {
            "asset_name": "gusset_bag",
            "rest_grip_positions": [
                [-0.14, 0.395, float(original[:, 2].max())],
                [0.14, 0.395, float(original[:, 2].max())],
            ],
            "initial_rotation": [float(np.sqrt(0.5)), 0, 0, float(np.sqrt(0.5))],
            "initial_translation": [0, 0.30 + mid, 0.255],
            "grip_radius": 0.020,
            "settling_time": 0.0,
            "lift_height": 0.35,
            "placement_height_offset": 0.0,
        }
        grasps = default_grasps()
        for grasp, angle in zip(grasps, (-90, 90), strict=True):
            rotation = wp.quat_from_axis_angle(wp.vec3(0, -1, 0), float(np.deg2rad(angle)))
            grasp["rotation_xyzw"] = list(rotation * wp.quat(*grasp["rotation_xyzw"]))
        (OUT / "setup.json").write_text(json.dumps(self.actual_setup, indent=2))
        (OUT / "grasp-recipe.json").write_text(json.dumps(grasps, indent=2))
        # Replace the required positive parked-hand schedule before any step.
        construction = {**self.actual_setup, "settling_time": 0.1}
        super().__init__(scene_id, grasps=grasps, use_cuda_graph=False, **construction, **kwargs)
        q_before = self.state.particle_q.numpy().copy()
        qd_before = self.state.particle_qd.numpy().copy()
        self.settling_time = 0.0
        self.settled_targets = [
            self.rest[i].astype(float).copy() - np.array([0.0, 0.0, 0.005]) for i in self.patch_anchor_indices
        ]
        # Reuse the exact prior mesh. Move only the original robot contact centers.
        self.patch_anchor_indices = [
            int(np.linalg.norm(self.rest - target, axis=1).argmin()) for target in self.settled_targets
        ]
        self.patch_indices = [
            np.flatnonzero(np.linalg.norm(self.rest - target, axis=1) < 0.020) for target in self.settled_targets
        ]
        self.patch_normals = [[0.0, -1.0, 0.0]] * 2
        self.patch_eigenvalues = [np.linalg.eigvalsh(np.cov(self.rest[p].T)).tolist() for p in self.patch_indices]
        self.patch_plane_spans = [float(np.ptp(self.rest[p, 1])) for p in self.patch_indices]
        self.times = [0.0, 0.5, 2.5, 4.0, 5.5, 7.0, 8.2, 9.4, 12.5, 13.1]
        self.duration = self.times[-1]
        self.hold_window = [4.0, 5.5]
        self.pre_lift_extraction = {
            "distance_m": 0.0,
            "method": "Both original hands start above the tabletop. The complete vertical-lift path is checked; no edge extraction is needed.",
        }
        drivers = []
        self.prepared_normal_shifts = [-0.00011, 0.0]
        for spec, grasp, target, offset in zip(
            self.specs,
            self.grasps,
            self.settled_targets,
            self.prepared_normal_shifts,
            strict=True,
        ):
            rotation = wp.quat(*grasp["rotation_xyzw"])
            root = target - np.asarray(wp.quat_rotate(rotation, wp.vec3(*grasp["contact_point_local_m"])))
            root += np.array([0.0, -offset, 0.0])
            positions = [
                root,
                root,
                root + [0, 0, self.lift_height],
                root + [0.24, 0.09, self.lift_height],
                root + [0.24, 0.09, self.lift_height],
                root + [0.24, 0.09, 0],
                root + [0.24, 0.09, 0],
                root + [0.24, -0.09, 0],
                root + [0.24, -0.09, 0.24],
                root + [0.24, -0.09, 0.24],
            ]
            controls = [grasp["closed_controls"]] * 6 + [grasp["open_controls"]] * 2
            controls += [grasp["release_controls"]] * 2
            drivers.append(
                H1HandDriver(
                    self.model,
                    spec,
                    times=self.times,
                    positions=positions,
                    rotations=tuple(rotation),
                    controls=controls,
                )
            )
        self.drivers = drivers
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1, dt=0.0)
        self.initial_hand_audits = [
            driver.geometry_audit(
                self.state,
                points=self.rest,
                radii=self.radius,
                full_surface_contact=True,
            )
            for driver in self.drivers
        ]
        if any(
            a["vertices_inside_robot_solids"] or a["minimum_vertex_to_robot_distance_m"] < 0.0037
            for a in self.initial_hand_audits
        ):
            raise ValueError(
                f"Prepared body pinch has invalid initial solid/preload geometry: {self.initial_hand_audits}"
            )
        self.prepared_margin_overlap_m = [
            max(0.0, 0.001 - a["minimum_shell_to_robot_distance_m"]) for a in self.initial_hand_audits
        ]
        triangles = self.model.tri_indices.numpy()
        rim_height = 0.255 + self.scale * self.mesh.metadata["bag_body_dimensions_m"][2]
        handle_faces = np.any(self.rest[triangles, 2] > rim_height + 1e-6, axis=1)
        self.handle_vertex_indices = np.unique(triangles[handle_faces])
        self.handle_audits = []
        self.planned_path_audit = self._audit_planned_path()
        if (
            self.planned_path_audit["minimum_conservative_table_separation_m"] < 0
            or self.planned_path_audit["minimum_conservative_interhand_separation_m"] < 0
        ):
            raise ValueError(f"Prepared body pinch path failed: {self.planned_path_audit}")
        state = self.model.state()
        clock = wp.zeros(1, dtype=float, device=self.model.device)
        minimum = (float("inf"), None)
        samples = np.linspace(0, self.duration, int(self.duration * 20) + 1)
        for time in samples:
            clock.assign(np.asarray([time], dtype=np.float32))
            for driver in self.drivers:
                driver.update(clock, state)
                low = min(float(vertices[:, 2].min()) for vertices, _ in driver.world_collision_meshes(state).values())
                minimum = min(minimum, (low - 0.002, float(time)))
        self.additional_ground_audit = {
            "minimum_robot_ground_clearance_including_both_margins_m": minimum[0],
            "time_s": minimum[1],
            "samples": len(samples),
        }
        if minimum[0] < 0:
            raise ValueError(f"Prepared body pinch crosses ground: {self.additional_ground_audit}")
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1, dt=0.0)
        if not np.array_equal(q_before, self.state.particle_q.numpy()) or not np.array_equal(
            qd_before, self.state.particle_qd.numpy()
        ):
            raise AssertionError("Robot preparation modified material state")
        self.camera_pos, self.camera_target = (1.5, -1.7, 1.55), (0.10, 0.40, 0.72)
        self.title = "R7 | H1 pair: deeper prepared upright gusset body-rim hold"
        self.description = (
            "Complete original gusset with both handles, rigidly placed upright on the same table. "
            "Two original H1 hands start in measured contact on single front-panel rim patches, then lift, "
            "carry, hold, place, open and withdraw. No acquisition is claimed. All material is free, "
            "with no pins or attachments. This body-rim hold does not establish handle-loop release."
        )
        self._audit_handles()
        (OUT / "prepared-start-audit.json").write_text(
            json.dumps(
                {
                    "hands": self.initial_hand_audits,
                    "margin_overlap_m": self.prepared_margin_overlap_m,
                    "path": self.planned_path_audit,
                    "ground": self.additional_ground_audit,
                    "handles": self.handle_audits[0],
                    "panel_normal_spans_m": self.patch_plane_spans,
                    "scope": "Prepared contact. No free-start acquisition or settled-collapse acquisition is claimed.",
                },
                indent=2,
            )
        )
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self.graph = capture.graph

    def _audit_handles(self):
        q = self.state.particle_q.numpy()[self.handle_vertex_indices]
        self.handle_audits.append(
            {
                "time_s": self.time,
                "hands": [
                    driver.geometry_audit(
                        self.state,
                        points=q,
                        radii=self.radius,
                        full_surface_contact=True,
                    )
                    for driver in self.drivers
                ],
            }
        )

    def step(self):
        super().step()
        if round(self.time / self.frame_dt) % 15 == 0 or self.time >= self.duration - 1e-6:
            self._audit_handles()

    @property
    def phase(self):
        if not hasattr(self, "prepared_normal_shifts"):
            return "initialize prepared hold"
        names = [
            "prepared body-rim contact",
            "lift",
            "carry",
            "hold",
            "place",
            "open fingers",
            "withdraw from body rim",
            "raise and straighten fingers",
            "settle after withdrawal",
        ]
        for end, name in zip(self.times[1:], names, strict=True):
            if self.time < end - 1e-6:
                return name
        return "complete"

    def metrics(self):
        result = super().metrics()
        result.update(
            {
                "actual_setup": self.actual_setup,
                "operation_change": self.description,
                "prepared_contact": True,
                "prepared_contact_centers_world_m": [p.tolist() for p in self.settled_targets],
                "deeper_engagement_change_m": 0.005,
                "carry_translation_m": [0.24, 0.09, 0.0],
                "acquisition_claimed": False,
                "start_mode": "Prepared original-hand contact on an undeformed upright bag, before any settling",
                "free_settle_duration_s": 0.0,
                "prepared_normal_shifts_m": self.prepared_normal_shifts,
                "prepared_margin_overlap_m": self.prepared_margin_overlap_m,
                "additional_ground_audit": self.additional_ground_audit,
                "handle_vertex_indices": self.handle_vertex_indices.tolist(),
                "handle_contact_audits": self.handle_audits,
                "grip_location_change": "The mesh is identical to the shallow proof. Both original wrist roots move 5 mm down, placing the contact centers about 9.16 mm below the original body rim. Mesh refinement seeds remain at source R5 X +/-0.14 m, Y0.395 m. Both original handle loops remain and have separate live clearance audits.",
            }
        )
        return result


manipulation.build_scene = lambda scene_id, **kwargs: PreparedBodyRimScene(scene_id, **kwargs)
try:
    run_smoke.main()
except Exception as error:
    (OUT / "constructor-or-run-error.json").write_text(json.dumps({"error": str(error)}, indent=2))
    raise
