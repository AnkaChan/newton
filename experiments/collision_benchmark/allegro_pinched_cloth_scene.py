# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental prepared cloth pinch with original Allegro collision geometry."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .allegro_gripper import OPEN_FINGERS, AllegroHandDriver, add_allegro_hand
from .jacket_diagnostics import ClothStrain
from .manipulation import tick

__all__ = ["PreparedAllegroClothScene", "prepared_patch_mesh"]

_INITIAL = (
    0.2388718437723227,
    0.7673740610366316,
    0.9927611726714369,
    0.9035068317272997,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1.0186845328048597,
    0.5912907345882313,
    0.9954048894532371,
    0.4980209511711675,
)
_CLOSED = (
    0.23838780714288266,
    0.7721924491701699,
    0.9943024233582033,
    0.9034811166378188,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1.0289318963389187,
    0.5795015750753556,
    0.9943293793789686,
    0.48806657850715346,
)


def prepared_patch_mesh():
    """Return an original 145 by 40 mm flat cloth panel in palm-local metres."""
    xs, ys = np.linspace(0.060, 0.205, 59), np.linspace(-0.045, -0.005, 17)
    vertices = np.asarray([(x, y, 0.011) for x in xs for y in ys], dtype=np.float32)
    triangles = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            a, b = i * len(ys) + j, (i + 1) * len(ys) + j
            triangles.extend([(a, b, b + 1), (a, b + 1, a + 1)])
    return vertices, np.asarray(triangles, dtype=np.int32)


class PreparedAllegroClothScene:
    """Retain, lift, carry and release an unpinned patch from a prepared pinch."""

    def __init__(self, *, device=None, **_kwargs):
        self.scene_id = "a1_allegro_cloth"
        self.title = "A1 | Allegro prepared cloth pinch"
        self.description = (
            "Original Allegro thumb and index hold a vertical cloth patch from a prepared contact state. "
            "The physical tip gap starts at 3 mm and closes to 2 mm. Initial shell clearance is 0.5 mm, "
            "with 0.5 mm numerical hand-margin contact allowance per tip. All vertices start outside the "
            "actual robot solids. VBD computes all cloth motion; no pins, attachments or inserts. "
            "The prescribed hand lifts, carries, holds, opens and releases. This is not a table-pickup test."
        )
        self.asset_ids = ["G4_Allegro_hand", "original_prepared_cloth_patch"]
        self.frame_dt, self.substeps, self.iterations = 1 / 30, 10, 30
        self.dt, self.time, self.duration = self.frame_dt / self.substeps, 0.0, 8.0
        self.radius = 0.001
        self.camera_pos, self.camera_target = (0.75, -0.95, 0.80), (0.04, 0, 0.32)
        self.camera_fov = 46.0
        state_dir = os.environ.get("NEWTON_ALLEGRO_CLOTH_STATE_DIR")
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        vertices, triangles = prepared_patch_mesh()
        self.mesh_hash = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
        self.rotation = (0, float(np.sqrt(0.5)), 0, float(np.sqrt(0.5)))
        initial, raised, carried, withdrawn = (0, 0, 0.33), (0, 0, 0.53), (0.10, 0.06, 0.53), (0.10, 0.06, 0.63)
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.mu, b.default_shape_cfg.ke, b.default_shape_cfg.kd = 0.8, 30000.0, 0.1
        b.default_shape_cfg.margin, b.default_shape_cfg.gap = 0.001, 0.005
        self.hand = add_allegro_hand(b, root_pos=initial, rotation=self.rotation, finger_positions=_INITIAL)
        b.add_cloth_mesh(
            pos=initial,
            rot=wp.quat(*self.rotation),
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
        self.driver = AllegroHandDriver(
            self.model,
            self.hand,
            times=[0, 0.1, 0.4, 1.9, 2.2, 3, 4.5, 5.5, 6, 8],
            positions=[initial, initial, initial, raised, raised, carried, carried, carried, withdrawn, withdrawn],
            rotations=self.rotation,
            finger_positions=[
                _INITIAL,
                _CLOSED,
                _CLOSED,
                _CLOSED,
                _CLOSED,
                _CLOSED,
                _CLOSED,
                OPEN_FINGERS,
                OPEN_FINGERS,
                OPEN_FINGERS,
            ],
        )
        self.driver.update(self.clock, self.state0, self.state1)
        self.strain = ClothStrain(self.model)
        self.initial = self.strain.sample(self.state.particle_q.numpy())
        self.initial_audit = self.driver.geometry_audit(
            self.state, points=self.state.particle_q.numpy(), radii=self.radius
        )
        if self.initial_audit["vertices_inside_original_hand_solids"]:
            raise ValueError("Prepared patch has vertices inside original Allegro collision solids")
        if self.initial_audit["minimum_shell_to_hand_distance_m"] < 0:
            raise ValueError("Prepared physical cloth shell intersects original Allegro solids")
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
            (0.1, "close prepared pinch"),
            (0.4, "retain"),
            (1.9, "lift"),
            (2.2, "raised pause"),
            (3, "carry"),
            (4.5, "hold"),
            (5.5, "open and release"),
            (6, "withdraw"),
            (8.1, "released"),
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
        if round(self.time / self.frame_dt) % 3 == 0:
            self.audits.append(
                {"time_s": self.time, **self.driver.geometry_audit(self.state, points=q, radii=self.radius)}
            )
        if self.state_dir and round(self.time / self.frame_dt) in (12, 57, 90, 135, 165, 240):
            np.savez_compressed(self.state_dir / f"state-{self.time:.2f}s.npz", positions=q)

    def metrics(self):
        hold = [r for r in self.history if 3 - 1e-9 <= r["time_s"] <= 4.5 + 1e-9]
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
                "bending_rest_angles": "flat source",
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
                "hand_shape_margin_m": 0.001,
                "rigid_soft_detection_gap_m": 0.005,
                "full_surface_rigid_soft_contact": True,
                "mesh_vertices": self.model.particle_count,
                "mesh_triangles": self.model.tri_count,
                "mesh_scale": 1.0,
            },
            "source_asset": {
                "name": "original_prepared_cloth_patch",
                "generator_revision": "allegro-cloth-patch-v1",
                "geometry_sha256": self.mesh_hash,
                "units": "m",
                "license": "Apache-2.0",
                "dimensions_m": [0.145, 0.04, 0],
                "mass_kg": float(self.strain.mass.sum()),
            },
            "robot_asset": self.hand.provenance,
            "grasp_recipe": {
                "initial_joint_positions_rad": list(_INITIAL),
                "closed_joint_positions_rad": list(_CLOSED),
                "palm_initial_position_m": [0, 0, 0.33],
                "palm_rotation_xyzw": list(self.rotation),
                "initial_physical_tip_gap_m": 0.003,
                "closed_physical_tip_gap_m": 0.002,
                "initial_physical_shell_clearance_m": 0.0005,
                "initial_margin_contact_allowance_per_tip_m": 0.0005,
                "closure_duration_s": 0.1,
                "scope": "Prepared contact state; retention, lift, carry and release only. No table pickup or full arm reach claim.",
                "geometry_audit_evidence": "robot-gripper-validation/allegro-prepared-pinch-pair.json",
            },
            "boundary_conditions": "Only original robot root and joints prescribed; all cloth particles dynamic, no pins or inserts.",
            "initial_strain": self.initial,
            "initial_geometry_audit": self.initial_audit,
            "hold_window_s": [3.0, 4.5],
            "hold_min_shell_clearance_m": min((r["min_z_m"] - self.radius for r in hold), default=None),
            "hold_max_principal_stretch": max((r["principal_stretch_max"] for r in hold), default=None),
            "hold_center_rise_m": min(
                (r["center_of_mass_m"][2] - self.initial["center_of_mass_m"][2] for r in hold), default=None
            ),
            "trajectory": self.history,
            "robot_geometry_audits": self.audits,
            "qualification": "Smoke only; no actuator force capacity claim.",
        }
