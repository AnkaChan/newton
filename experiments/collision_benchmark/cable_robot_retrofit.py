# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full original C8 cable arrangements with an original Franka hand.

Original rod builders are reused verbatim. Only the old jaw boxes are omitted.
VBD integrates dynamic rods internally; the real hand uses prescribed FK.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .cable_scenes import _advance_time, _CableScene, _prescribe_bodies
from .robot_gripper import FrankaHandDriver, add_franka_hand

SCENE_IDS = ("c9_pile_pickup", "c9_bundle_squeeze", "c9_bundle_squeeze_withdraw")


class CableRobotScene(_CableScene):
    def __init__(self, scene_id, *, device=None, substeps=10, iterations=24, use_cuda_graph=True):
        if scene_id not in SCENE_IDS:
            raise ValueError(scene_id)
        if int(substeps) != substeps or not 2 <= substeps <= 10 or substeps % 2:
            raise ValueError("Use an even number of substeps in [2, 10]")
        self.scene_id, self.time = scene_id, 0.0
        self.is_pile = scene_id == "c9_pile_pickup"
        self.withdraw = scene_id == "c9_bundle_squeeze_withdraw"
        self.frame_dt, self.duration = 1.0 / 60.0, 9.0
        self.substeps, self.iterations = int(substeps), int(iterations)
        self.sim_dt = self.frame_dt / self.substeps
        self.radius, self._grasp_x = 0.009, 0.10
        self.cable_bodies, self.cables, self.jaw_bodies = [], [], []
        self.fixture_ids, self.fixture_notes = [], []
        self._moving_ids, self._offsets = [], []
        self._times = np.asarray([0.0, 1.5, 2.5, 4.5, 6.5, 7.5, 9.0], dtype=np.float32)
        self.omitted_original_jaws, self.history = [], []
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.default_shape_cfg.ke, builder.default_shape_cfg.kd = 1.0e5, 0.0
        builder.default_shape_cfg.mu = 1.5
        builder.default_shape_cfg.margin, builder.default_shape_cfg.gap = 0.0, 0.002
        self._cfg = newton.ModelBuilder.ShapeConfig(density=500.0, ke=1.0e5, kd=0.0, mu=1.0, margin=0.0, gap=0.002)
        ground_cfg = self._cfg.copy()
        ground_cfg.mu = 0.5
        builder.add_ground_plane(cfg=ground_cfg)
        if self.is_pile:
            self._build_pile(builder)
        else:
            self._build_bundle(builder)
        self.original_asset_ids = list(self.asset_ids)
        self.asset_ids = [a for a in self.asset_ids if not a.startswith("G0_")] + ["cached_franka_fr3_hand"]
        self.title = (
            "C9 | Franka: original loose cable pile pickup"
            if self.is_pile
            else "C9 | Franka: original bundle squeeze and strand pull"
        )
        self.description = (
            "Full original C8 loose pile: 720 mm selected strand, long curved strand, open coil, and three-arm branch. "
            "Original Franka hand approaches the settled selected strand, closes real fingers, lifts 450 mm, holds 2 s, "
            "opens and releases. All cable bodies remain dynamic; no cable attachments."
            if self.is_pile
            else "Full original C8 bundle: three 720 mm cables of radius 9 mm. Original Franka fingers squeeze a local "
            "17.5 mm axial patch; one original declared endpoint fixture pulls the orange strand 160 mm, then fingers open. "
            "No cable-to-hand attachments. The single endpoint boundary condition is retained explicitly."
        )
        if self.withdraw:
            self.title = "C9 | Franka: bundle squeeze, strand pull and withdrawal"
            self.description += " The open hand withdraws 150 mm after 7.5 s to separate fully from strands that spread beyond its 80 mm aperture."
        self.fixture_notes = (
            ["No cable endpoints are pinned. Original hand root and finger joints follow prescribed FK."]
            if self.is_pile
            else [
                "Orange last capsule remains the original explicit mass-zero endpoint pulling fixture.",
                "Original hand root and finger joints use prescribed FK. Cable-to-hand coupling is contact only.",
            ]
        )
        self.pinch = np.asarray((self._grasp_x, 0.0, 0.15) if self.is_pile else (0.0, 0.0, 0.013))
        self.closed_opening = 0.0176 if self.is_pile else 0.050
        self.hand = add_franka_hand(builder, pinch_pos=self.pinch, rotation=(1.0, 0.0, 0.0, 0.0), opening=0.08)
        builder.color()
        self.model = builder.finalize(device=device)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        if self.is_pile:
            times = [0.0, 0.5, 1.5, 2.5, 4.5, 6.5, 7.5, 9.0]
            positions = [(self._grasp_x, 0.0, z) for z in (0.15, 0.15, 0.013, 0.013, 0.463, 0.463, 0.463, 0.463)]
            openings = [0.08] * 3 + [self.closed_opening] * 3 + [0.08] * 2
        else:
            times = self._times.tolist()
            positions = [self.pinch] * len(times)
            openings = [0.08, 0.08, self.closed_opening, self.closed_opening, self.closed_opening, 0.08, 0.08]
            if self.withdraw:
                times = [*self._times[:-1].tolist(), 8.5, 9.0]
                positions = [self.pinch] * 6 + [self.pinch + np.asarray((0.0, 0.0, 0.15))] * 2
                openings = [*openings[:-1], 0.08, 0.08]
        self.driver = FrankaHandDriver(
            self.model, self.hand, times=times, positions=positions, rotations=(1.0, 0.0, 0.0, 0.0), openings=openings
        )
        self.driver.update(self._clock, self.state_0, self.state_1)
        self.pipeline = newton.CollisionPipeline(self.model, contact_matching="sticky")
        self.contacts = self.pipeline.contacts()
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=False,
            rigid_compliant_alm=True,
            rigid_body_contact_buffer_size=128,
            rigid_contact_history=True,
        )
        self._times_wp = wp.array(self._times, dtype=float, device=self.model.device)
        self._moving_ids_wp = wp.array(self._moving_ids, dtype=int, device=self.model.device)
        poses = self.state.body_q.numpy()
        self._base_wp = wp.array(poses[self._moving_ids], dtype=wp.transform, device=self.model.device)
        self._offsets_wp = wp.array(
            np.asarray(self._offsets, dtype=np.float32).reshape((-1, len(self._times), 3)),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.shape_bodies = self.model.shape_body.numpy()
        scales = self.model.shape_scale.numpy()
        self.half_lengths = np.asarray(
            [scales[np.flatnonzero(self.shape_bodies == body)[0], 1] for body in self.cable_bodies]
        )
        self.initial_cable_q = poses[self.cable_bodies].copy()
        self.initial_geometry = self._geometry_audit()
        if self.initial_geometry["minimum_shell_to_robot_distance_m"] < -1e-6:
            raise ValueError("Initial original cable capsules overlap the real open hand")
        if self.initial_geometry["robot_clearance_above_table_plane_m"] < 0.003:
            raise ValueError("Original robot solids intersect or nearly touch the ground")
        self._record_state()
        self._graph = None
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self._graph = capture.graph

    def _jaw(self, builder, position, half_extents, offsets):
        # The inherited geometry builders call this hook. Record, but never add, old boxes.
        self.omitted_original_jaws.append({"position_m": list(position), "half_extents_m": list(half_extents)})
        return -1

    @property
    def phase(self):
        if self.is_pile:
            for end, phase in (
                (0.5, "pile settle"),
                (1.5, "open hand approach"),
                (2.5, "close original fingers"),
                (4.5, "lift"),
                (6.5, "hold"),
                (7.5, "release"),
                (9.1, "fall and settle"),
            ):
                if self.time < end:
                    return phase
        else:
            for end, phase in (
                (1.5, "bundle settle"),
                (2.5, "squeeze"),
                (4.5, "squeeze hold"),
                (6.5, "strand pull"),
                (7.5, "open fingers"),
                (8.5, "open-hand withdrawal" if self.withdraw else "recovery"),
                (9.1, "recovery"),
            ):
                if self.time < end:
                    return phase
        return "complete"

    def _simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            wp.launch(_advance_time, dim=1, inputs=[self._clock, self.sim_dt], device=self.model.device)
            if self._moving_ids:
                wp.launch(
                    _prescribe_bodies,
                    dim=len(self._moving_ids),
                    inputs=[
                        self._clock,
                        self._moving_ids_wp,
                        self._base_wp,
                        self._times_wp,
                        self._offsets_wp,
                        self.state_0.body_q,
                        self.state_1.body_q,
                        self.state_0.body_qd,
                        self.state_1.body_qd,
                    ],
                    device=self.model.device,
                )
            # Internal VBD retains the previous body pose, so command the endpoint in state_in.
            self.driver.update(self._clock, self.state_0, self.state_1)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _centerline(self):
        q = self.state.body_q.numpy()[self.cable_bodies]
        x, y, z, w = (q[:, i] for i in range(3, 7))
        axis = np.column_stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)))
        shift = axis * self.half_lengths[:, None]
        return q[:, :3], q[:, :3] - shift, q[:, :3] + shift

    def _geometry_audit(self):
        center, start, end = self._centerline()
        points = np.concatenate([start, (start + center) / 2, center, (center + end) / 2, end])
        return self.driver.geometry_audit(
            self.state, points=points, radii=self.radius, table_top=0.0, full_surface_contact=False
        )

    def step(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        self.time += self.frame_dt
        self._record_state()

    def _record_state(self):
        center, start, end = self._centerline()
        n = len(self.cables[0])
        vel = self.state.body_qd.numpy()[self.cable_bodies]
        geometry = self._geometry_audit()
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        b0 = self.shape_bodies[self.contacts.rigid_contact_shape0.numpy()[:count]]
        b1 = self.shape_bodies[self.contacts.rigid_contact_shape1.numpy()[:count]]
        cable0, cable1 = np.isin(b0, self.cable_bodies), np.isin(b1, self.cable_bodies)
        hand0, hand1 = np.isin(b0, self.hand.body_ids), np.isin(b1, self.hand.body_ids)
        selected = center[:n]
        local = np.abs(center[:, 0]) < 0.025
        self.history.append(
            {
                "time_s": self.time,
                "phase": self.phase,
                "finite": bool(np.isfinite(center).all() and np.isfinite(vel).all()),
                "selected_center_m": selected.mean(axis=0).tolist(),
                "selected_minimum_capsule_surface_z_m": float(np.minimum(start[:n, 2], end[:n, 2]).min() - self.radius),
                "all_cable_minimum_capsule_surface_z_m": float(np.minimum(start[:, 2], end[:, 2]).min() - self.radius),
                "selected_maximum_neighbor_joint_gap_m": float(np.linalg.norm(end[: n - 1] - start[1:n], axis=1).max()),
                "hand_cable_contact_count": int(np.count_nonzero((cable0 & hand1) | (cable1 & hand0))),
                "cable_self_contact_count": int(np.count_nonzero(cable0 & cable1)),
                "whole_robot_ground_clearance_m": geometry["robot_clearance_above_table_plane_m"],
                "minimum_sampled_capsule_to_robot_separation_m": geometry["minimum_shell_to_robot_distance_m"],
                "physical_tip_aperture_m": geometry["physical_tip_aperture_m"],
                "local_bundle_width_y_m": float(np.ptp(center[local, 1]) + 2 * self.radius) if np.any(local) else None,
                "maximum_cable_linear_speed_m_s": float(np.linalg.norm(vel[:, :3], axis=1).max()),
            }
        )

    def metrics(self):
        hold = [r for r in self.history if 4.5 - 1e-6 <= r["time_s"] <= 6.5 + 1e-6]
        clear = min(r["selected_minimum_capsule_surface_z_m"] for r in hold) if hold else None
        mean_rise = (
            (min(r["selected_center_m"][2] for r in hold) - self.history[0]["selected_center_m"][2]) if hold else None
        )
        lifted = bool(
            self.is_pile
            and hold
            and clear > 0.003
            and mean_rise > 0.10
            and min(r["hand_cable_contact_count"] for r in hold) > 0
        )
        jp, jc = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        attachment = (np.isin(jp, self.cable_bodies) & np.isin(jc, self.hand.body_ids)) | (
            np.isin(jc, self.cable_bodies) & np.isin(jp, self.hand.body_ids)
        )
        return {
            "scene_id": self.scene_id,
            "original_scene_id": "c8_pile_pickup" if self.is_pile else "c8_bundle_squeeze",
            "open_hand_withdrawal_variant": self.withdraw,
            "solver": "VBD",
            "finite": all(r["finite"] for r in self.history),
            "original_rod_geometry_builder": "cable_scenes._CableScene._build_pile/_build_bundle, unchanged",
            "original_cable_geometry_preserved": True,
            "cable_count": len(self.cables),
            "cable_segment_counts": [len(c) for c in self.cables],
            "cable_segment_count": len(self.cable_bodies),
            "cable_mass_kg": float(self.model.body_mass.numpy()[self.cable_bodies].sum()),
            "physics_parameters": {
                "frame_dt_s": self.frame_dt,
                "substeps": self.substeps,
                "substep_dt_s": self.sim_dt,
                "iterations": self.iterations,
                "cable_density_kg_m3": 500.0,
                "cable_capsule_radius_m": self.radius,
                "rod_stretch_stiffness_N_m": 5e4,
                "rod_shear_stiffness_N_m": 5e4,
                "rod_stretch_damping_N_s_m": 0.1,
                "rod_shear_damping_N_s_m": 0.1,
                "rod_bend_stiffness_N_m_rad_by_cable": [0.008, 0.012, 0.008, 0.02] if self.is_pile else [0.02] * 3,
                "rod_twist_stiffness": "same as bend",
                "rod_bend_damping_N_m_s_rad": 0.0005,
                "rod_twist_damping_N_m_s_rad": 0.0005,
                "rigid_contact_ke": 1e5,
                "rigid_contact_kd": 0.0,
                "cable_shape_mu": 1.0,
                "robot_shape_mu": 1.5,
                "ground_shape_mu": 0.5,
                "rigid_shape_margin_m": 0.0,
                "rigid_shape_detection_gap_m": 0.002,
                "rigid_compliant_alm": True,
                "rigid_contact_history": True,
                "contact_matching": "sticky",
                "internal_rigid_integration": True,
                "cable_self_contact": "Enabled; original adjacent rod-joint filters",
                "nominal_closed_finger_opening_m": self.closed_opening,
                "robot_pose_integration": "Hand FK at t+dt in state_in; internal VBD retains prior body poses",
            },
            "robot_provenance": self.hand.provenance,
            "boundary_conditions": self.fixture_notes,
            "fixture_body_ids": self.fixture_ids,
            "cable_zero_inverse_mass_count": int(
                np.count_nonzero(self.model.body_inv_mass.numpy()[self.cable_bodies] == 0.0)
            ),
            "cable_kinematic_body_count": int(
                np.count_nonzero(self.model.body_flags.numpy()[self.cable_bodies] & int(newton.BodyFlags.KINEMATIC))
            ),
            "cable_attachment_joint_count": int(np.count_nonzero(attachment)),
            "omitted_original_cube_jaws": self.omitted_original_jaws,
            "operation_change": (
                "Real 17.5 mm axial fingertip patch replaces 90 mm jaw width. Open robot approaches from TCP Z=150 to 13 mm during 0.5-1.5 s, then retains original 450 mm lift and hold/release times."
                if self.is_pile
                else "Real 17.5 mm axial fingertip patch replaces 240 mm jaw width. Nominal 80 mm opening closes to 50 mm, retaining original closed width, squeeze/strand-pull timing and 160 mm endpoint displacement."
            ),
            "withdrawal_change": "Open hand rises 150 mm during 7.5-8.5 s; original open jaw fixture had 108 mm aperture versus the real hand maximum 80 mm."
            if self.withdraw
            else None,
            "initial_geometry_audit": self.initial_geometry,
            "hold_measurement_window_s": [4.5, 6.5],
            "hold_minimum_selected_capsule_surface_z_m": clear,
            "hold_minimum_hand_cable_contact_count": min(r["hand_cable_contact_count"] for r in hold) if hold else None,
            "hold_maximum_selected_neighbor_joint_gap_m": max(r["selected_maximum_neighbor_joint_gap_m"] for r in hold)
            if hold
            else None,
            "minimum_whole_robot_ground_clearance_m": min(r["whole_robot_ground_clearance_m"] for r in self.history),
            "hold_minimum_selected_mean_rise_m": mean_rise,
            "lift_diagnostic_criterion": "All selected capsule surfaces above 3 mm; mean rise above 100 mm; active hand contact throughout hold. This is a smoke observation, not a benchmark qualification threshold.",
            "lift_observed": lifted,
            "release_observed": bool(
                (lifted or (self.withdraw and self.history[-1]["minimum_sampled_capsule_to_robot_separation_m"] > 0.02))
                and self.time >= 8.9
                and self.history[-1]["hand_cable_contact_count"] == 0
                and self.history[-1]["selected_center_m"][2] < 0.08
            ),
            "squeeze_contact_observed": bool(
                not self.is_pile and hold and max(r["hand_cable_contact_count"] for r in hold) > 0
            ),
            "trajectory": self.history,
            "qualification": "Original full-scene smoke; no dynamic actuator-capacity or long-duration qualification. Robot separation uses sampled capsule-axis points plus radius.",
        }


def build_scene(scene_id, **kwargs):
    return CableRobotScene(scene_id, **kwargs)
