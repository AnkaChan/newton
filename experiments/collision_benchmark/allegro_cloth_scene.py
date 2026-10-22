# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental original Allegro hand and an empty, open cloth pouch."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .allegro_gripper import OPEN_FINGERS, AllegroHandDriver, add_allegro_hand
from .allegro_pinched_cloth_scene import PreparedAllegroClothScene
from .jacket_diagnostics import ClothStrain
from .manipulation import tick

__all__ = ["AllegroClothScene", "build_scene"]

_CLOTH_GRASP = (
    0.1030286434,
    0.7176282053,
    1.0497465279,
    0.8962094451,
    0.0,
    0.5993493993,
    0.6753508878,
    1.1366940129,
    -0.1030286434,
    0.7176282060,
    1.0497465260,
    0.8962094479,
    0.8225844374,
    0.9978583720,
    1.2396354384,
    0.5310200419,
)


def _open_pouch():
    """Original rounded pouch surface [m]; one open rim and no interior solid."""
    radius, segments, rings = 0.045, 40, 18
    vertices = [[0.0, 0.0, 0.0025]]
    triangles = []
    for theta in np.linspace(np.pi, 0.45, rings + 1)[1:]:
        for phi in np.arange(segments) * 2 * np.pi / segments:
            vertices.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * (1 + np.cos(theta)) + 0.0025,
                ]
            )
    for j in range(segments):
        triangles.append([0, 1 + (j + 1) % segments, 1 + j])
    for i in range(rings - 1):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            k = (j + 1) % segments
            triangles.extend([[a + j, a + k, b + k], [a + j, b + k, b + j]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


class AllegroClothScene:
    """Acquire an empty cloth pouch through original robot-surface contact."""

    def __init__(self, *, device=None, **_kwargs):
        self.scene_id = "a1_allegro_cloth"
        self.title = "A1 | Allegro and empty cloth pouch"
        self.description = (
            "Original Allegro hand approaches and closes around an empty open cloth pouch on a narrow support, "
            "then lifts, carries, holds and releases. The curved rest shape is authored; the shell has no insert, "
            "tetrahedra, pins or attachments. Original robot joints and floating root are prescribed."
        )
        self.asset_ids = ["G4_Allegro_hand", "original_open_round_cloth_pouch"]
        self.frame_dt, self.substeps, self.iterations = 1 / 30, 10, 30
        self.dt, self.time, self.duration = self.frame_dt / self.substeps, 0.0, 10.0
        self.radius = 0.001
        self.support_height = 0.03
        state_dir = os.environ.get("NEWTON_ALLEGRO_CLOTH_STATE_DIR")
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        self.camera_pos, self.camera_target = (0.58, -0.74, 0.62), (0.04, 0.0, 0.19)
        self.camera_fov = 42.0
        vertices, triangles = _open_pouch()
        self.mesh_hash = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.mu = 0.8
        b.default_shape_cfg.ke, b.default_shape_cfg.kd = 30000.0, 0.1
        b.default_shape_cfg.margin, b.default_shape_cfg.gap = 0.001, 0.005
        rotation = (0, float(np.sqrt(0.5)), 0, float(np.sqrt(0.5)))
        self.hand = add_allegro_hand(b, root_pos=(-0.015, 0, 0.282), rotation=rotation)
        b.add_cloth_mesh(
            pos=(0, 0, self.support_height),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=vertices.tolist(),
            indices=triangles.ravel().tolist(),
            density=0.22,
            particle_radius=self.radius,
            tri_ke=1800.0,
            tri_ka=1800.0,
            tri_kd=0.025,
            edge_ke=0.08,
            edge_kd=0.0005,
            color=(0.25, 0.65, 0.91),
        )
        b.add_shape_box(
            -1,
            xform=wp.transform((0, 0, self.support_height / 2), wp.quat_identity()),
            hx=0.015,
            hy=0.015,
            hz=self.support_height / 2,
            color=(0.72, 0.70, 0.65),
            label="visible narrow pouch support",
        )
        b.add_ground_plane()
        b.color(include_bending=True)
        self.model = b.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 30000.0, 0.1, 0.8
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.0015,
            particle_self_contact_gap=0.001,
            particle_rest_shape_contact_exclusion_radius=0.003,
            particle_topological_contact_filter_threshold=1,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.005, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.state0, self.state1, self.control = self.model.state(), self.model.state(), self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        times = [0, 0.6, 2, 3.5, 4, 5.5, 5.8, 6.6, 7.5, 8.5, 10]
        initial, grasp, raised, carried = (
            (-0.015, 0, 0.282),
            (-0.015, 0, 0.112),
            (-0.015, 0, 0.332),
            (0.065, 0.05, 0.332),
        )
        self.driver = AllegroHandDriver(
            self.model,
            self.hand,
            times=times,
            positions=[initial, initial, grasp, grasp, grasp, raised, raised, carried, carried, carried, carried],
            rotations=rotation,
            finger_positions=[OPEN_FINGERS] * 3 + [_CLOTH_GRASP] * 6 + [OPEN_FINGERS] * 2,
        )
        self.driver.update(self.clock, self.state0, self.state1)
        self.strain = ClothStrain(self.model)
        self.initial = self.strain.sample(self.state.particle_q.numpy())
        self.initial_audit = self.driver.geometry_audit(
            self.state, points=self.state.particle_q.numpy(), radii=self.radius
        )
        if self.initial_audit["minimum_shell_to_hand_distance_m"] < 0:
            raise ValueError("Initial cloth intersects original Allegro collision geometry")
        self.history, self.audits = [], []
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.simulate()
        self.graph = capture.graph

    @property
    def state(self):
        return self.state0

    @property
    def phase(self):
        for end, name in [
            (0.6, "supported settle"),
            (2, "approach"),
            (3.5, "close"),
            (4, "acquire"),
            (5.5, "lift"),
            (5.8, "raised pause"),
            (6.6, "carry"),
            (7.5, "hold"),
            (8.5, "open"),
            (10.1, "released"),
        ]:
            if self.time < end - 1e-6:
                return name
        return "done"

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
        q = self.state.particle_q.numpy()
        self.history.append({"time_s": self.time, "phase": self.phase, **self.strain.sample(q)})
        if self.state_dir and round(self.time / self.frame_dt) in (60, 120, 198, 225, 255, 300):
            np.savez_compressed(self.state_dir / f"state-{self.time:.2f}s.npz", positions=q)
        if round(self.time / self.frame_dt) % 3 == 0:
            self.audits.append(
                {"time_s": self.time, **self.driver.geometry_audit(self.state, points=q, radii=self.radius)}
            )

    def metrics(self):
        hold = [r for r in self.history if 6.6 - 1e-9 <= r["time_s"] <= 7.5 + 1e-9]
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
                "bending_rest_angles": "authored curved pouch surface",
                "particle_radius_m": self.radius,
                "self_contact_enabled": True,
                "self_contact_margin_m": 0.0015,
                "self_contact_gap_m": 0.001,
                "self_contact_rest_exclusion_m": 0.003,
                "self_contact_topological_rings": 1,
                "friction_robot": 0.8,
                "friction_ground": 0.8,
                "friction_soft_self": 0.8,
                "contact_stiffness_N_m": 30000.0,
                "contact_damping": 0.1,
                "rigid_soft_detection_gap_m": 0.005,
                "full_surface_rigid_soft_contact": True,
                "mesh_vertices": self.model.particle_count,
                "mesh_triangles": self.model.tri_count,
                "mesh_scale": 1.0,
            },
            "source_asset": {
                "name": "original_open_round_cloth_pouch",
                "generator_revision": "allegro-cloth-pouch-v1",
                "geometry_sha256": self.mesh_hash,
                "units": "m",
                "license": "Apache-2.0",
                "construction": "original spherical pouch surface with one open rim, no internal solid",
            },
            "robot_asset": self.hand.provenance,
            "grasp_recipe": {
                "closed_joint_positions_rad": list(_CLOTH_GRASP),
                "palm_grasp_position_m": [-0.015, 0, 0.112],
                "palm_rotation_xyzw": [0, float(np.sqrt(0.5)), 0, float(np.sqrt(0.5))],
                "method": "Bounded fingertip inverse kinematics against the measured 2 s cloth state; physical open-to-close acquisition.",
                "closure_audit_samples": 451,
                "minimum_hand_support_separation_including_margin_m": 0.006652772326133057,
                "minimum_hand_ground_clearance_including_margin_m": 0.0041696218516245305,
                "geometry_audit_evidence": "robot-gripper-validation/allegro-cloth-narrow-grasp.json",
            },
            "boundary_conditions": "Only original robot root and joints prescribed; all cloth particles dynamic, no pins or inserts.",
            "initial_strain": self.initial,
            "initial_geometry_audit": self.initial_audit,
            "hold_window_s": [6.6, 7.5],
            "hold_min_shell_clearance_m": min((r["min_z_m"] - self.radius for r in hold), default=None),
            "hold_min_shell_clearance_above_support_plane_m": min(
                (r["min_z_m"] - self.radius - self.support_height for r in hold), default=None
            ),
            "support": {"box_dimensions_m": [0.03, 0.03, self.support_height], "top_m": self.support_height},
            "hold_max_principal_stretch": max((r["principal_stretch_max"] for r in hold), default=None),
            "trajectory": self.history,
            "robot_geometry_audits": self.audits,
            "qualification": "Smoke only; no actuator force or full-arm reach claim.",
        }


def build_scene(scene_id="a1_allegro_cloth", **kwargs):
    if scene_id != "a1_allegro_cloth":
        raise ValueError(scene_id)
    return PreparedAllegroClothScene(**kwargs)
