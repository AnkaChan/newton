# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original pouch held at its front lip by the cached Franka hand's real fingers.

This is a prepared contact hold, not grasp acquisition. No cloth vertices are
pinned. The 2 mm effective shell thickness fits the rest mesh's narrow sides.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .assets import AddedAsset
from .pouch_mesh import build_pouch_mesh
from .robot_gripper import FrankaHandDriver, add_franka_hand

SCENE_IDS = ("m2_franka_pouch",)
__all__ = ["SCENE_IDS", "build_scene"]


@wp.kernel
def _tick(clock: wp.array[float], dt: float):
    clock[0] += dt


class _PouchScene:
    def __init__(self, *, device=None, use_cuda_graph=True, substeps=10, iterations=40):
        self.scene_id = "m2_franka_pouch"
        self.time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.duration = 10.0
        self.substeps = int(substeps)
        self.iterations = int(iterations)
        if self.substeps < 2 or self.substeps % 2:
            raise ValueError("Pouch substeps must be a positive even integer")
        self.dt = self.frame_dt / self.substeps
        self.table_top = 0.18
        self.particle_radius = 0.001
        self.self_contact_margin = 0.002
        self.self_contact_gap = 0.0015
        self.title = "M2 | Franka hand and pouch: prepared rim hold, carry, release"
        self.description = (
            "Original Franka hand collision shapes and finger joints pinch the pouch front lip. "
            "Prepared contact hold; no approach/acquisition claim. The supported pouch is 150 by 200 by 18 mm, "
            "with a 2 mm effective contact thickness and refined front panels. Prescribed robot FK lifts 220 mm, "
            "carries 160 mm in X and 100 mm in Y, holds, places, opens, and withdraws. "
            "VBD computes all pouch motion; every cloth vertex remains free and self-contact stays enabled."
        )
        self.asset_ids = ["pouch_grip_refined_2mm_shell", "cached_franka_fr3_hand"]
        self.camera_pos = (0.72, -0.94, 0.90)
        self.camera_target = (0.055, 0.04, 0.43)
        self.camera_fov = 42.0
        self.history = []
        self.worst_hold_shape = None

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 0.1
        builder.default_shape_cfg.mu = 1.5
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.002
        builder.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=64)
        mesh = build_pouch_mesh()
        particle_start = builder.particle_count
        builder.add_cloth_mesh(
            vertices=mesh.vertices.tolist(),
            indices=mesh.triangles.ravel().tolist(),
            pos=(0.0, 0.0, self.table_top + self.particle_radius + 0.001),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            color=mesh.metadata["color_rgb"],
            density=0.22,
            particle_radius=self.particle_radius,
            tri_ke=3000.0,
            tri_ka=3000.0,
            tri_kd=0.025,
            edge_ke=0.025,
            edge_kd=0.0005,
        )
        self.pouch = AddedAsset(mesh, particle_start, builder.particle_count, mesh.landmarks)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5.0e4, kd=0.1, mu=0.6)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.08, 0.05, self.table_top - 0.03), wp.quat_identity()),
            hx=0.40,
            hy=0.32,
            hz=0.03,
            cfg=cfg,
            color=(0.43, 0.48, 0.54),
            label="fixed_support_and_placement_table",
        )
        builder.add_ground_plane()
        pinch = np.array([0.0, -0.009, self.table_top + self.particle_radius + 0.001 + 0.2 - 0.009])
        rotation = (1.0, 0.0, 0.0, 0.0)
        self.closed_opening = 0.0015
        self.hand = add_franka_hand(
            builder, pinch_pos=pinch, rotation=rotation, opening=self.closed_opening, label="pouch_franka_hand"
        )
        builder.color()
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 1.5
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        times = [0.0, 0.5, 2.5, 3.0, 4.5, 6.0, 7.5, 8.1, 9.0, 10.0]
        offsets = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.22),
            (0.0, 0.0, 0.22),
            (0.16, 0.10, 0.22),
            (0.16, 0.10, 0.22),
            (0.16, 0.10, 0.01),
            (0.16, 0.10, 0.01),
            (0.16, 0.10, 0.16),
            (0.16, 0.10, 0.16),
        ]
        self.driver = FrankaHandDriver(
            self.model,
            self.hand,
            times=times,
            positions=pinch + np.asarray(offsets),
            rotations=rotation,
            openings=[self.closed_opening] * 7 + [0.06] * 3,
        )
        self.driver.update(self._clock, self.state_0, self.state_1)
        self.initial_q = self.state.particle_q.numpy()
        self.initial_geometry = self.driver.geometry_audit(
            self.state,
            points=self.initial_q,
            radii=self.particle_radius,
            table_top=self.table_top,
            full_surface_contact=True,
        )
        if self.initial_geometry["vertices_inside_robot_solids"]:
            raise ValueError("Initial pouch vertices intersect the original Franka collision geometry")
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=self.self_contact_margin,
            particle_self_contact_gap=self.self_contact_gap,
            particle_vertex_contact_buffer_size=128,
            particle_edge_contact_buffer_size=128,
            friction_epsilon=0.001,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.003, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        detector = self.solver.trimesh_collision_detector
        detector.rebuild(self.state.particle_q)
        detector.vertex_triangle_collision_detection(self.self_contact_margin)
        detector.edge_edge_collision_detection(self.self_contact_margin)
        self.initial_self_contact = {
            "interaction_distance_m": self.self_contact_margin,
            "vertex_triangle_pairs_inside_interaction_distance": int(
                detector.vertex_colliding_triangles_count.numpy().sum()
            ),
            "edge_edge_pairs_inside_interaction_distance": int(detector.edge_colliding_edges_count.numpy().sum()),
        }
        self.triangles = self.model.tri_indices.numpy()
        self.edges = np.unique(
            np.sort(
                np.concatenate([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]]),
                axis=1,
            ),
            axis=0,
        )
        self.rest_lengths = np.linalg.norm(self.initial_q[self.edges[:, 1]] - self.initial_q[self.edges[:, 0]], axis=1)
        self._graph = None
        self._record_state()
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self._graph = capture.graph

    @property
    def state(self):
        return self.state_0

    @property
    def phase(self):
        for end, name in (
            (0.5, "prepared front-lip hold"),
            (2.5, "lift"),
            (3.0, "raised pause"),
            (4.5, "carry"),
            (6.0, "hold"),
            (7.5, "place"),
            (8.1, "open fingers"),
            (9.0, "withdraw"),
            (10.1, "released observation"),
        ):
            if self.time < end:
                return name
        return "complete"

    def _simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.driver.update(self._clock, self.state_0, self.state_1, dt=self.dt)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(_tick, dim=1, inputs=[self._clock, self.dt], device=self.model.device)

    def step(self):
        if self._graph is None:
            self._simulate()
        else:
            wp.capture_launch(self._graph)
        self.time += self.frame_dt
        self._record_state()

    def _record_state(self):
        q = self.state.particle_q.numpy()
        v = self.state.particle_qd.numpy()
        ratio = np.linalg.norm(q[self.edges[:, 1]] - q[self.edges[:, 0]], axis=1) / self.rest_lengths
        if 4.5 - 1.0e-6 <= self.time <= 6.0 + 1.0e-6 and (
            self.worst_hold_shape is None or float(ratio.max()) > self.worst_hold_shape["max_edge_ratio"]
        ):
            order = np.argsort(ratio)[-10:][::-1]
            self.worst_hold_shape = {
                "time_s": self.time,
                "max_edge_ratio": float(ratio.max()),
                "rigid_soft_contact_records": int(self.contacts.soft_contact_count.numpy()[0]),
                "edges": [
                    {
                        "vertex_ids": self.edges[i].tolist(),
                        "rest_endpoints_world_m": self.initial_q[self.edges[i]].tolist(),
                        "current_endpoints_world_m": q[self.edges[i]].tolist(),
                        "rest_length_m": float(self.rest_lengths[i]),
                        "length_ratio": float(ratio[i]),
                    }
                    for i in order
                ],
            }
        full_audit = self.driver.geometry_audit(
            self.state, radii=self.particle_radius, table_top=self.table_top, full_surface_contact=True
        )
        audit = {
            key: full_audit[key]
            for key in (
                "physical_tip_aperture_m",
                "minimum_robot_collision_z_m",
                "robot_clearance_above_table_plane_m",
                "vertices_inside_robot_solids",
                "minimum_vertex_to_robot_distance_m",
                "minimum_shell_to_robot_distance_m",
            )
        }
        self.history.append(
            {
                "time_s": self.time,
                "phase": self.phase,
                "finite": bool(np.isfinite(q).all() and np.isfinite(v).all()),
                "center_m": q.mean(axis=0).tolist(),
                "min_surface_z_m": float(q[:, 2].min() - self.particle_radius),
                "max_surface_z_m": float(q[:, 2].max() + self.particle_radius),
                "max_speed_m_s": float(np.linalg.norm(v, axis=1).max()),
                "edge_stretch_max": float(ratio.max()),
                "edge_stretch_p95": float(np.percentile(ratio, 95)),
                "edge_strain_rms": float(np.sqrt(np.mean((ratio - 1.0) ** 2))),
                "robot_geometry": audit,
            }
        )

    def metrics(self):
        hold = [r for r in self.history if 4.5 - 1.0e-6 <= r["time_s"] <= 6.0 + 1.0e-6]
        startup = [r for r in self.history if r["time_s"] <= 0.5]
        initial_center = self.initial_q.mean(axis=0)
        clearance = min(r["min_surface_z_m"] - self.table_top for r in hold) if hold else None
        rise = float(min(r["center_m"][2] - initial_center[2] for r in hold)) if hold else None
        return {
            "scene_id": self.scene_id,
            "solver": "VBD",
            "finite": all(r["finite"] for r in self.history),
            "particle_count": self.model.particle_count,
            "bag_mass_kg": float(self.model.particle_mass.numpy().sum()),
            "pinned_particle_count": int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0.0)),
            "substeps": self.substeps,
            "iterations": self.iterations,
            "dt_s": self.dt,
            "physics_parameters": {
                "frame_dt_s": self.frame_dt,
                "substeps": self.substeps,
                "substep_dt_s": self.dt,
                "iterations": self.iterations,
                "gravity_m_s2": [0.0, 0.0, -9.81],
                "density_kg_m2": 0.22,
                "mass_kg": float(self.model.particle_mass.numpy().sum()),
                "tri_ke": 3000.0,
                "tri_ka": 3000.0,
                "tri_kd": 0.025,
                "edge_ke": 0.025,
                "edge_kd": 0.0005,
                "particle_radius_m": self.particle_radius,
                "effective_shell_thickness_m": 2 * self.particle_radius,
                "self_contact_enabled": True,
                "self_contact_interaction_margin_m": self.self_contact_margin,
                "self_contact_extra_detection_gap_m": self.self_contact_gap,
                "topological_contact_filter_rings": 2,
                "rest_shape_contact_exclusion_radius_m": 0.0,
                "soft_contact_ke": 5.0e4,
                "soft_contact_kd": 0.1,
                "soft_contact_mu": 1.5,
                "robot_shape_mu": 1.5,
                "table_shape_mu": 0.6,
                "rigid_shape_margin_m": 0.0,
                "robot_shape_gap_m": 0.002,
                "ground_shape_gap_m": 0.002,
                "table_shape_gap_m": 0.1,
                "rigid_soft_extra_detection_gap_m": 0.003,
                "rigid_soft_full_surface_contact": True,
                "friction_epsilon": 0.001,
                "mesh_dimensions_xyz_m": [0.15, 0.018, 0.20],
                "mesh_cells_xyz": [24, 4, 24],
                "mesh_vertex_count": self.model.particle_count,
                "mesh_triangle_count": self.model.tri_count,
                "nominal_closed_finger_opening_m": self.closed_opening,
                "robot_pose_integration": "Input FK at t; output FK at t+substep_dt",
            },
            "material": {
                "density_kg_m2": 0.22,
                "tri_ke": 3000.0,
                "tri_ka": 3000.0,
                "tri_kd": 0.025,
                "edge_ke": 0.025,
                "edge_kd": 0.0005,
            },
            "contact": {
                "rigid_particle_radius_m": self.particle_radius,
                "effective_shell_thickness_m": 2 * self.particle_radius,
                "self_contact_interaction_m": self.self_contact_margin,
                "self_contact_detection_extra_m": self.self_contact_gap,
                "soft_contact_ke": 5.0e4,
                "soft_contact_kd": 0.1,
                "soft_contact_mu": 1.5,
            },
            "robot_provenance": self.hand.provenance,
            "boundary_conditions": "Original detached Franka hand root and finger joints use prescribed FK; table fixed; all cloth vertices dynamic.",
            "grasp_protocol": "Prepared contact hold at the front lip; no acquisition claim. Original failed recipe used wider generic jaws.",
            "initial_geometry_audit": self.initial_geometry,
            "initial_self_contact_audit": self.initial_self_contact,
            "table_top_z_m": self.table_top,
            "hold_measurement_window_s": [4.5, 6.0],
            "hold_minimum_bag_surface_clearance_m": clearance,
            "hold_minimum_bag_center_rise_m": rise,
            "hold_center_xy_m": np.mean([r["center_m"][:2] for r in hold], axis=0).tolist() if hold else None,
            "lift_observed": bool(clearance is not None and clearance > 0.05 and rise > 0.10),
            "minimum_whole_robot_table_clearance_m": min(
                r["robot_geometry"]["robot_clearance_above_table_plane_m"] for r in self.history
            ),
            "maximum_vertices_inside_robot_solids": max(
                r["robot_geometry"]["vertices_inside_robot_solids"] for r in self.history
            ),
            "release_observed": bool(
                self.time >= 9.0 and self.history[-1]["robot_geometry"]["minimum_shell_to_robot_distance_m"] > 0.02
            ),
            "startup_edge_stretch_max": max(r["edge_stretch_max"] for r in startup),
            "hold_edge_stretch_max": max(r["edge_stretch_max"] for r in hold) if hold else None,
            "hold_edge_stretch_p95_max": max(r["edge_stretch_p95"] for r in hold) if hold else None,
            "hold_edge_strain_rms_max": max(r["edge_strain_rms"] for r in hold) if hold else None,
            "worst_hold_shape_diagnostic": self.worst_hold_shape,
            "shape_diagnostic_note": "Edge length divided by rest length; strain RMS uses ratio minus one. These are diagnostics, not added benchmark qualification scores.",
            "final_robot_geometry": self.driver.geometry_audit(
                self.state,
                points=self.state.particle_q.numpy(),
                radii=self.particle_radius,
                table_top=self.table_top,
                full_surface_contact=True,
            ),
            "trajectory": self.history,
            "qualification": "Prepared contact hold smoke only; no dynamic robot capacity, calibrated material, or 10/60-second hold qualification.",
        }


def build_scene(scene_id, **kwargs):
    """Build the corrected pouch scene in SI units."""
    if scene_id not in SCENE_IDS:
        raise ValueError(f"Unknown pouch scene {scene_id!r}")
    return _PouchScene(**kwargs)
