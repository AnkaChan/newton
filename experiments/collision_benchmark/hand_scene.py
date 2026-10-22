# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allegro hand contact smoke scene with prescribed robot joint motion."""

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .allegro_gripper import GROUND_GRASP_FINGERS, OPEN_FINGERS, AllegroHandDriver, add_allegro_hand
from .assets import add_solid_asset
from .soft_scenes import _advance_clock

SCENE_IDS = ["r3_allegro_sphere", "r3_allegro_ground_sphere"]


class HandScene:
    """Prescribed Allegro joints manipulate an unpinned VBD soft sphere."""

    def __init__(self, scene_id="r3_allegro_sphere", *, device=None, use_cuda_graph=True, **_kwargs):
        if scene_id not in SCENE_IDS:
            raise ValueError(scene_id)
        self.scene_id = "r3_allegro_ground_sphere"
        self.title = "R3 | Allegro hand: palm-down ground pickup, hold, release"
        self.description = "Cached Allegro left-hand URDF approaches a soft sphere resting on the ground with its palm facing down. Original finger joints close, lift, carry, hold, and reopen. VBD computes all sphere motion and contact; no sphere vertices are pinned. Robot root and joints are prescribed within original position and velocity limits; actuator force capacity is not tested."
        self.asset_ids = ["G4_Allegro_hand", "sphere"]
        self.frame_dt = 1.0 / 30.0
        self.substeps = 10
        self.dt = self.frame_dt / self.substeps
        self.time = 0.0
        self.duration = 10.0
        self.camera_pos = (0.55, -0.75, 0.62)
        self.camera_target = (0.035, 0.025, 0.17)
        self.camera_fov = 44.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg.mu = 0.8
        builder.default_shape_cfg.ke = 30000.0
        builder.default_shape_cfg.kd = 0.1
        builder.default_shape_cfg.margin = 0.001
        rotation = tuple(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2))
        self.hand = add_allegro_hand(builder, root_pos=(-0.015, 0, 0.282), rotation=rotation)
        self.added = add_solid_asset(
            builder,
            "sphere",
            resolution="coarse",
            pos=(0.0, 0.0, 0.0025),
            scale=1.8,
            density=180.0,
            particle_radius=0.0015,
            k_mu=8000.0,
            k_lambda=8000.0,
            k_damp=0.5,
            color=(0.94, 0.48, 0.19),
        )
        builder.add_ground_plane()
        builder.color(include_bending=True)
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 30000.0
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 0.8
        self.solver = SolverVBD(
            self.model,
            iterations=15,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.003,
            particle_self_contact_gap=0.005,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, broad_phase="nxn", soft_contact_gap=0.005, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self._s0, self._s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.driver = AllegroHandDriver(
            self.model,
            self.hand,
            times=[0, 0.6, 2, 3.5, 4, 5.5, 5.8, 6.6, 7.5, 8.5, 10],
            positions=[(-0.015, 0, 0.282)] * 2
            + [(-0.015, 0, 0.112)] * 3
            + [(-0.015, 0, 0.332)] * 2
            + [(0.065, 0.05, 0.332)] * 4,
            rotations=rotation,
            finger_positions=[OPEN_FINGERS] * 3 + [GROUND_GRASP_FINGERS] * 6 + [OPEN_FINGERS] * 2,
        )
        self.driver.update(self._clock, self._s0, self._s1)
        self._tets = self.model.tet_indices.numpy()
        self._initial = self._s0.particle_q.numpy()
        self._initial_geometry = self.driver.geometry_audit(self.state, points=self._initial, radii=0.0015)
        if self._initial_geometry["vertices_inside_original_hand_solids"]:
            raise ValueError("Grounded sphere starts inside the open hand")
        if self._initial_geometry["minimum_shell_to_hand_distance_m"] <= 0:
            raise ValueError("Grounded sphere shell overlaps the initial open hand")
        self._min_hand_ground_clearance = self._initial_geometry["hand_ground_clearance_including_margin_m"]
        tets = self._initial[self._tets]
        self._rest_dets = np.linalg.det(tets[:, 1:] - tets[:, :1])
        self._max_center_z = float(self._initial[:, 2].mean())
        self._max_inversions = 0
        self._min_ratio = 1.0
        self._history = []
        self._graph = None
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self._graph = capture.graph

    @property
    def state(self):
        return self._s0

    @property
    def phase(self):
        for end, label in [
            (0.6, "sphere rests on ground"),
            (2.0, "palm-down approach"),
            (3.5, "close fingers"),
            (4.0, "acquire"),
            (5.5, "lift"),
            (5.8, "raised pause"),
            (6.6, "carry"),
            (7.5, "hold after carry"),
            (8.5, "open fingers"),
            (10.0, "released"),
        ]:
            if self.time < end - 1e-6:
                return label
        return "released"

    def _simulate(self):
        for _ in range(self.substeps):
            self.driver.update(self._clock, self._s0, self._s1, dt=self.dt)
            self._s0.clear_forces()
            self.pipeline.collide(self._s0, self.contacts)
            self.solver.step(self._s0, self._s1, self.control, self.contacts, self.dt)
            self._s0, self._s1 = self._s1, self._s0
            wp.launch(_advance_clock, dim=1, inputs=[self._clock, self.dt], device=self.model.device)

    def step(self):
        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        self.time += self.frame_dt
        q = self.state.particle_q.numpy()
        self._max_center_z = max(self._max_center_z, float(q[:, 2].mean()))
        tets = q[self._tets]
        ratios = np.linalg.det(tets[:, 1:] - tets[:, :1]) / self._rest_dets
        self._max_inversions = max(self._max_inversions, int(np.count_nonzero(ratios <= 0)))
        self._min_ratio = min(self._min_ratio, float(ratios.min()))
        geometry = self.driver.geometry_audit(self.state, points=q, radii=0.0015)
        self._min_hand_ground_clearance = min(
            self._min_hand_ground_clearance, geometry["hand_ground_clearance_including_margin_m"]
        )
        self._history.append(
            {
                "time_s": self.time,
                "phase": self.phase,
                "center_m": q.mean(axis=0).tolist(),
                "minimum_surface_z_m": float(q[:, 2].min() - 0.0015),
                "hand_geometry": geometry,
            }
        )

    def metrics(self):
        q = self.state.particle_q.numpy()
        v = self.state.particle_qd.numpy()
        tets = q[self._tets]
        ratios = np.linalg.det(tets[:, 1:] - tets[:, :1]) / self._rest_dets
        return {
            "scene_id": self.scene_id,
            "finite": bool(np.isfinite(q).all() and np.isfinite(v).all()),
            "time_s": self.time,
            "phase": self.phase,
            "particle_count": self.model.particle_count,
            "tet_count": self.model.tet_count,
            "minimum_tet_volume_ratio": float(ratios.min()),
            "inverted_tet_count": int(np.count_nonzero(ratios <= 0)),
            "max_inverted_tets_over_frames": self._max_inversions,
            "min_volume_ratio_over_frames": self._min_ratio,
            "sphere_center_m": q.mean(axis=0).tolist(),
            "sphere_min_z_m": float(q[:, 2].min()),
            "sphere_max_center_z_m": self._max_center_z,
            "initial_sphere_center_z_m": float(self._initial[:, 2].mean()),
            "max_speed_m_s": float(np.linalg.norm(v, axis=1).max()),
            "robot_command": "Prescribed FK root and finger positions; 16 actuated joints plus commanded floating root",
            "rigid_motion_bridge": "Input FK at t; output FK at t+dt; free-joint velocity referenced to root COM",
            "driver_revision": 2,
            "soft_boundary_conditions": "None; every sphere vertex is dynamic",
            "pinned_particle_count": int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0)),
            "initial_hand_geometry": self._initial_geometry,
            "minimum_hand_ground_clearance_over_frames_m": self._min_hand_ground_clearance,
            "urdf_source": self.hand.provenance["source_urdf"],
            "robot_asset": self.hand.provenance,
            "substeps": self.substeps,
            "iterations": 15,
            "contact_friction": 0.8,
            "solid_damping_pa_s": 0.5,
            "trajectory": self._history,
            "hold_minimum_sphere_surface_z_m": min(
                (row["minimum_surface_z_m"] for row in self._history if 6.6 <= row["time_s"] <= 7.5), default=None
            ),
            "physics_parameters": {
                "solver": "VBD",
                "frame_dt_s": self.frame_dt,
                "substeps": self.substeps,
                "substep_dt_s": self.dt,
                "iterations": 15,
                "density_kg_m3": 180.0,
                "tet_shear_modulus_pa": 8000.0,
                "tet_first_lame_parameter_pa": 8000.0,
                "tet_damping_pa_s": 0.5,
                "particle_radius_m": 0.0015,
                "particle_count": self.model.particle_count,
                "tet_count": self.model.tet_count,
                "soft_contact_ke": self.model.soft_contact_ke,
                "soft_contact_kd": self.model.soft_contact_kd,
                "soft_contact_mu": self.model.soft_contact_mu,
                "rigid_shape_margin_m": 0.001,
                "soft_contact_detection_gap_m": 0.005,
                "self_contact_enabled": True,
                "self_contact_interaction_margin_m": 0.003,
                "self_contact_detection_gap_m": 0.005,
                "rigid_soft_full_surface_contact": True,
                "prescribed_robot_fk": True,
                "external_rigid_solver_bridge": True,
            },
        }


def build_scene(scene_id="r3_allegro_sphere", **kwargs):
    return HandScene(scene_id, **kwargs)
