# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Native cached jacket with an original Franka hand and contact-only VBD."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
from experiments.collision_benchmark.jacket_asset import load_native_jacket
from experiments.collision_benchmark.jacket_diagnostics import ClothStrain
from experiments.collision_benchmark.manipulation import tick
from experiments.collision_benchmark.robot_gripper import FrankaHandDriver, add_franka_hand
from newton.solvers import SolverVBD

__all__ = ["FrankaJacketScene"]


class FrankaJacketScene:
    """Prescribe real hand joints; leave every cloth degree of freedom dynamic."""

    def __init__(self, *, device=None, iterations=100, **kwargs):
        self.scene_id = "j2_franka_jacket"
        self.title = "J2 | Native jacket and Franka hand"
        self.description = (
            "Original cached Style3D jacket and original Franka hand collision geometry. "
            "After free settling, the hand closes on a measured raised fold, lifts, carries, holds, places and releases. "
            "Robot root and finger joints are prescribed; all cloth motion is VBD with self-contact and no pins."
        )
        self.asset_ids = ["h1_jacket_native", "G1_franka_original_hand"]
        self.frame_dt, self.substeps, self.iterations = 1 / 30, 10, iterations
        self.dt = self.frame_dt / self.substeps
        self.time, self.duration = 0.0, 10.5
        self.table_top = 0.25
        self.radius = 0.0015
        self.camera_pos = (1.65, -2.15, 1.60)
        self.camera_target = (0.08, 0.28, 0.72)
        self.camera_fov = 48.0
        mesh = load_native_jacket()
        self.mesh = mesh
        lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        self.scale = 0.6 / max(hi - lo)
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.ke, b.default_shape_cfg.kd = 50000.0, 0.1
        b.default_shape_cfg.mu, b.default_shape_cfg.gap = 1.5, 0.005
        b.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=64)
        b.add_cloth_mesh(
            vertices=mesh.vertices.tolist(),
            indices=mesh.triangles.ravel().tolist(),
            pos=(
                -float((lo[0] + hi[0]) * 0.5 * self.scale),
                -float(lo[2] * self.scale),
                0.258 + float(hi[1] * self.scale),
            ),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2),
            vel=(0, 0, 0),
            scale=self.scale,
            density=0.22,
            particle_radius=self.radius,
            tri_ke=1800.0,
            tri_ka=1800.0,
            tri_kd=0.025,
            edge_ke=0.08,
            edge_kd=0.0005,
            color=tuple(mesh.metadata["color_rgb"]),
        )
        table_cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=0.35, ke=50000, kd=0.1, gap=0.005)
        b.add_shape_box(
            -1,
            xform=wp.transform((0.0, 0.3, 0.12), wp.quat_identity()),
            hx=0.75,
            hy=0.70,
            hz=0.13,
            cfg=table_cfg,
            color=(0.48, 0.51, 0.55),
            label="full garment support and placement table",
        )
        # A 15-degree yaw gives the original fingers a clear corridor around
        # the collar fold. Geometry was audited on the saved settled mesh.
        self.rotation = (-float(np.sin(np.pi / 24)), float(np.cos(np.pi / 24)), 0.0, 0.0)
        self.hand = add_franka_hand(b, pinch_pos=(2, 2, 2), rotation=self.rotation, opening=0.08)
        b.add_ground_plane()
        b.color()
        self.model = b.finalize(device=device)
        self.model.soft_contact_ke = 50000.0
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 0.35
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.002,
            particle_self_contact_gap=0.001,
            particle_rest_shape_contact_exclusion_radius=0.005,
            particle_topological_contact_filter_threshold=1,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=64,
            friction_epsilon=0.001,
            integrate_with_external_rigid_solver=True,
            rigid_compliant_alm=True,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.005, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.state0, self.state1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.strain = ClothStrain(self.model)
        self.authored_state = self.strain.sample(self.state.particle_q.numpy())
        self.driver = FrankaHandDriver(
            self.model,
            self.hand,
            times=[0, 20],
            positions=[(2, 2, 2)] * 2,
            rotations=self.rotation,
            openings=[0.08, 0.08],
        )
        self.driver.update(self.clock, self.state0, self.state1)
        self.settle_history = []
        self._capture()
        for frame in range(60):
            wp.capture_launch(self.graph)
            if frame % 3 == 2:
                q = self.state.particle_q.numpy()
                if not np.isfinite(q).all():
                    raise ValueError("Native jacket free settling produced a non-finite state")
                self.settle_history.append({"time_s": (frame + 1) * self.frame_dt, **self.strain.sample(q)})
        q = self.state.particle_q.numpy()
        state_dir = os.environ.get("NEWTON_JACKET_STATE_DIR")
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.state_dir / "settled-state.npz", positions=q, triangles=mesh.triangles)
        highest, target, local_span, yaw, self.rotation = self._select_collar_grasp(q)
        self.target = target
        raised = target + np.asarray([0, 0, 1.05])
        carried = raised + np.asarray([0.24, 0.10, 0])
        placed = carried - np.asarray([0, 0, 1.00])
        withdrawn = placed + np.asarray([0, 0, 0.24])
        self.times = [0, 0.4, 0.7, 2.7, 3.0, 4.5, 6.0, 8.0, 8.7, 9.7, 10.5]
        self.driver = FrankaHandDriver(
            self.model,
            self.hand,
            times=self.times,
            positions=[target, target, target, raised, raised, carried, carried, placed, placed, withdrawn, withdrawn],
            rotations=self.rotation,
            openings=[0.08, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.08, 0.08, 0.08],
        )
        self.clock.zero_()
        self.driver.update(self.clock, self.state0, self.state1)
        self.initial_audit = self.driver.geometry_audit(
            self.state, radii=self.radius, table_top=self.table_top, full_surface_contact=True
        )
        if self.initial_audit["vertices_inside_robot_solids"]:
            raise ValueError("Initial native jacket intersects the original hand collision solids")
        if self.initial_audit["minimum_shell_to_robot_distance_m"] < -1e-6:
            raise ValueError("Initial jacket shell overlaps the original hand collision solids")
        if self.initial_audit["robot_clearance_above_table_plane_m"] < 0.003:
            raise ValueError("Original hand has insufficient initial table clearance")
        self.initial = self.strain.sample(q)
        self.history = []
        self.hand_audits = [{"time_s": 0, **self.initial_audit}]
        self.grasp = {
            "vertex": highest,
            "region": "authored collar: upper 80 mm of source, within 120 mm of centreline",
            "source_vertex_m": mesh.vertices[highest].tolist(),
            "hand_yaw_degrees": yaw,
            "hand_rotation_xyzw": list(self.rotation),
            "tcp_m": target.tolist(),
            "local_material_span_y_m": local_span,
            "lift_displacement_m": 1.05,
            "geometry_candidates_checked": self.candidates_checked,
            "settle_s": 2.0,
            "acquisition_scope": "Open-to-close acquisition starts from the audited open pose after settling. The transfer from the parked hand to that open pose is not an approach-path test.",
        }
        self._capture()

    def _select_collar_grasp(self, q):
        """Choose an open pose by original-hand geometry, without moving cloth."""
        source = self.mesh.vertices
        collar = np.flatnonzero((source[:, 2] > source[:, 2].max() - 0.08) & (np.abs(source[:, 0]) < 0.12))
        choices = []
        for index in collar[np.argsort(q[collar, 2])[::-1]]:
            if all(np.linalg.norm(q[index, :2] - q[other, :2]) > 0.008 for other in choices):
                choices.append(int(index))
            if len(choices) == 16:
                break
        self.candidates_checked = 0
        self.clock.zero_()
        for index in choices:
            for yaw in range(0, 180, 15):
                rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(np.deg2rad(yaw))) * wp.quat(0, 1, 0, 0)
                matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
                for offset in (-0.003, 0.0, 0.003):
                    target = q[index].astype(float)
                    target[2] += offset
                    local = (q - target) @ matrix
                    near = (
                        (np.abs(local[:, 0]) < 0.01025) & (np.abs(local[:, 2]) < 0.01075) & (np.abs(local[:, 1]) < 0.04)
                    )
                    if not near.any() or np.ptp(local[near, 1]) + 2 * self.radius >= 0.08:
                        continue
                    target += matrix[:, 1] * (0.5 * (local[near, 1].min() + local[near, 1].max()))
                    driver = FrankaHandDriver(
                        self.model,
                        self.hand,
                        times=[0, 1],
                        positions=[target, target],
                        rotations=rotation,
                        openings=[0.08, 0.08],
                    )
                    driver.update(self.clock, self.state0, self.state1)
                    audit = driver.geometry_audit(
                        self.state, radii=self.radius, table_top=self.table_top, full_surface_contact=True
                    )
                    self.candidates_checked += 1
                    if (
                        audit["vertices_inside_robot_solids"] == 0
                        and audit["minimum_shell_to_robot_distance_m"] >= 0
                        and audit["robot_clearance_above_table_plane_m"] >= 0.003
                    ):
                        return index, target, float(np.ptp(local[near, 1])), yaw, tuple(rotation)
        raise ValueError("No collar pose passed original-hand shell and table clearance checks")

    @property
    def state(self):
        return self.state0

    @property
    def phase(self):
        for end, name in [
            (0.4, "close"),
            (0.7, "acquire"),
            (2.7, "lift"),
            (3.0, "pause"),
            (4.5, "carry"),
            (6.0, "hold"),
            (8.0, "place"),
            (8.7, "open"),
            (9.7, "withdraw"),
            (20, "released observation"),
        ]:
            if self.time < end:
                return name
        return "done"

    def _capture(self):
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.solver.rebuild_bvh(self.state0)
        for _ in range(self.substeps):
            self.driver.update(self.clock, self.state0, self.state1, dt=self.dt)
            self.state0.clear_forces()
            self.state1.clear_forces()
            self.pipeline.collide(self.state0, self.contacts)
            self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
            self.state0, self.state1 = self.state1, self.state0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        wp.capture_launch(self.graph)
        self.time += self.frame_dt
        self.history.append(
            {"time_s": self.time, "phase": self.phase, **self.strain.sample(self.state.particle_q.numpy())}
        )
        if round(self.time / self.frame_dt) % 15 == 0:
            self.hand_audits.append(
                {
                    "time_s": self.time,
                    **self.driver.geometry_audit(
                        self.state, radii=self.radius, table_top=self.table_top, full_surface_contact=True
                    ),
                }
            )
        if self.state_dir and round(self.time / self.frame_dt) in (21, 81, 135, 180, 240, 261, 315):
            np.savez_compressed(self.state_dir / f"state-{self.time:.2f}s.npz", positions=self.state.particle_q.numpy())

    def metrics(self):
        hold = [row for row in self.history if 4.5 - 1e-9 <= row["time_s"] <= 6.0 + 1e-9]
        return {
            "physics_parameters": {
                "frame_dt_s": self.frame_dt,
                "substeps": self.substeps,
                "substep_dt_s": self.dt,
                "iterations": self.iterations,
                "density_kg_m2": 0.22,
                "membrane_stretch_coefficient_N_m": 1800.0,
                "membrane_area_coefficient_N_m": 1800.0,
                "membrane_damping": 0.025,
                "bending_coefficient_N": 0.08,
                "bending_damping": 0.0005,
                "bending_rest_angles": "authored 3D source geometry",
                "particle_radius_m": self.radius,
                "self_contact_enabled": True,
                "self_contact_margin_m": 0.002,
                "self_contact_gap_m": 0.001,
                "self_contact_rest_exclusion_m": 0.005,
                "self_contact_topological_rings": 1,
                "friction_robot": 1.5,
                "friction_table": 0.35,
                "friction_soft_self": 0.35,
                "contact_stiffness_N_m": 50000.0,
                "contact_damping": 0.1,
                "rigid_soft_detection_gap_m": 0.005,
                "full_surface_rigid_soft_contact": True,
                "palm_sdf_max_resolution": 64,
                "mesh_vertices": len(self.mesh.vertices),
                "mesh_triangles": len(self.mesh.triangles),
                "mesh_scale": self.scale,
            },
            "boundary_conditions": "Only original Franka floating root and finger joints prescribed. Every cloth particle has positive mass; no pins or attachments.",
            "source_asset": self.mesh.metadata,
            "robot_asset": self.hand.provenance,
            "cloth_scale": self.scale,
            "cloth_mass_kg": float(self.strain.mass.sum()),
            "substeps": self.substeps,
            "vbd_iterations": self.iterations,
            "material": {
                "density_kg_m2": 0.22,
                "tri_ke": 1800,
                "tri_ka": 1800,
                "tri_kd": 0.025,
                "edge_ke": 0.08,
                "edge_kd": 0.0005,
                "particle_radius_m": self.radius,
            },
            "self_contact": {
                "enabled": True,
                "interaction_margin_m": 0.002,
                "detection_gap_m": 0.001,
                "rest_neighbour_exclusion_m": 0.005,
                "topological_filter_rings": 1,
            },
            "table_top_m": self.table_top,
            "grasp_recipe": self.grasp,
            "authored_initial_strain": self.authored_state,
            "free_settle_trajectory": self.settle_history,
            "post_settle_initial_strain": self.initial,
            "hold_min_vertex_clearance_m": min((r["min_z_m"] - self.table_top for r in hold), default=None),
            "hold_window_s": [4.5, 6.0],
            "hold_min_shell_clearance_m": min(
                (r["min_z_m"] - self.table_top - self.radius for r in hold), default=None
            ),
            "hold_center_rise_m": float(
                np.mean([r["center_of_mass_m"][2] for r in hold]) - self.initial["center_of_mass_m"][2]
            )
            if hold
            else None,
            "hold_max_principal_stretch": max((r["principal_stretch_max"] for r in hold), default=None),
            "trajectory": self.history,
            "robot_geometry_audits": self.hand_audits,
            "qualification": "Smoke only. Strain ratios and clearance are raw setup diagnostics, not benchmark scores.",
        }
