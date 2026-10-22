# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental original-size solid manipulation with original robot hands."""

from __future__ import annotations

import hashlib

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .allegro_gripper import OPEN_FINGERS, AllegroHandDriver, add_allegro_hand
from .assets import add_solid_asset, solid_mesh
from .robot_gripper import FrankaHandDriver, add_franka_hand
from .soft_scenes import _advance_clock

__all__ = ["SCENE_IDS", "SolidRobotScene", "build_scene"]
SCENE_IDS = [f"r6_franka_{name}" for name in ("pad", "capsule", "torus", "branched")] + [
    "r6_allegro_block",
    "r6_allegro_sphere",
]

_BLOCK_JOINTS = (
    -0.14436758591926013,
    1.1592303185075432,
    0.8014652429120367,
    -0.18615144399240321,
    0.0,
    0.0,
    0.0,
    0.0,
    0.14436758591926013,
    1.1592303185075432,
    0.8014652429120367,
    -0.18615144399240321,
    0.6,
    0.0,
    0.0,
    0.0,
)
_BLOCK_OPEN_JOINTS = (
    -0.38739415909684854,
    1.4813021325095121,
    0.14058302544368834,
    -0.09854161661165434,
    0.0,
    0.0,
    0.0,
    0.0,
    0.38739415909684854,
    1.4813021325095121,
    0.14058302544368834,
    -0.09854161661165434,
    0.6,
    0.0,
    0.0,
    0.0,
)
_SPHERE_JOINTS = (
    -0.3589336983076603,
    1.2896538832440176,
    -6.552829793176853e-05,
    0.00011939477650725765,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3589336982409,
    1.2896066709751628,
    3.6905910484831585e-05,
    8.004860365408877e-05,
    1.3386048921280722,
    1.1543242133255607,
    -0.1571794484013016,
    0.7998975932062171,
)


def _separation(center, rotation, half, table_center, table_half):
    """Return a conservative OBB/AABB separation bound [m] using SAT."""
    axes = [*np.eye(3), *rotation.T]
    axes.extend(np.cross(a, b) for a in np.eye(3) for b in rotation.T)
    gaps = []
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-8:
            continue
        unit = axis / length
        gaps.append(abs(unit @ (center - table_center)) - abs(unit) @ table_half - abs(unit @ rotation) @ half)
    return float(max(gaps))


class SolidRobotScene:
    """Replace generated jaw fixtures while retaining the original solid mesh."""

    def __init__(self, scene_id, *, device=None, iterations=None, use_cuda_graph=True, **_kwargs):
        if scene_id not in SCENE_IDS:
            raise ValueError(scene_id)
        self.scene_id = scene_id
        _, self.robot, self.name = scene_id.split("_", 2)
        iterations = (100 if self.name == "capsule" else 30) if iterations is None else int(iterations)
        self.time, self.frame_dt, self.substeps = 0.0, 1 / 30, 10
        self.dt, self.iterations, self.duration = self.frame_dt / self.substeps, iterations, 11.5
        self.table_top, self.radius = 0.25, 0.003
        self.table_center, self.table_half = np.array((0.0, 0.0, 0.12)), np.array((0.6, 0.5, 0.13))
        self.camera_pos, self.camera_target = (1.2, -1.65, 1.45), (0.10, -0.04, 0.59)
        self.camera_fov = 45.0
        mesh = solid_mesh(self.name, resolution="coarse")
        self.scale = 0.16 / np.ptp(mesh.vertices, axis=0).max()
        self.mesh_hash = hashlib.sha256(mesh.vertices.tobytes() + mesh.tetrahedra.tobytes()).hexdigest()
        solid_rotation = wp.quat_identity()
        if self.name == "capsule":
            solid_rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)
        rotated = np.array([wp.quat_rotate(solid_rotation, wp.vec3(*p)) for p in mesh.vertices]) * self.scale
        lo, hi = rotated.min(0), rotated.max(0)
        self.object_dimensions = hi - lo
        pos = np.array((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0.256 - lo[2]))
        if self.name == "pad":
            pos[1] -= 0.48
        center = (lo + hi) / 2 + pos
        self.asset_position, self.asset_rotation = pos, tuple(solid_rotation)
        self.recipe = "Original solid scale and table preserved; actual robot closes on the supported object."
        if self.robot == "franka":
            rotation = (1.0, 0.0, 0.0, 0.0)
            pinch = center.copy()
            if self.name == "pad":
                rotation = tuple(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2))
                pinch[1] = -0.522
                opening, closed = 0.035, 0.012
                self.recipe = "Original pad overhangs the front of the original table; horizontal Franka fingers pinch its 13.3 mm thickness."
            elif self.name == "capsule":
                pinch[2] += 0.010
                opening, closed = 0.08, 0.064
                self.recipe = (
                    "Original 160 mm capsule rests on its long side; Franka pinches the 70 mm middle diameter."
                )
            elif self.name == "torus":
                pinch[1] = -0.060
                opening, closed = 0.065, 0.036
                self.recipe = "Franka grips the 40 mm tube at the front of the original 160 mm torus, with the inner finger in its hole."
            else:
                pinch[0] = 0.057
                opening, closed = 0.06, 0.046
                self.recipe = "Franka grips the original cross's positive-X arm outside the central crossing."
            self.base, self.opening, self.closed = pinch, opening, closed
        else:
            rotation = (0.0, float(np.sqrt(0.5)), 0.0, float(np.sqrt(0.5)))
            self.base = center + np.array((-0.015, 0, 0.11))
            self.closed = _BLOCK_JOINTS if self.name == "block" else _SPHERE_JOINTS
            self.open_fingers = _BLOCK_OPEN_JOINTS if self.name == "block" else OPEN_FINGERS
            self.recipe = "Original Allegro fingers close around the full-size solid using bounded original-URDF fingertip targets."
        self.title = f"R6 | {self.robot.title()} and original {self.name}: pick, carry, release"
        self.description = (
            self.recipe
            + " Every soft vertex is dynamic. Original robot root/joints are prescribed; no pins, attachments, added jaws or actuator-capacity claim."
        )
        self.asset_ids = [self.name, f"original_{self.robot}_hand"]
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.mu, b.default_shape_cfg.ke, b.default_shape_cfg.kd = 1.0, 50000.0, 0.1
        b.default_shape_cfg.margin, b.default_shape_cfg.gap = 0.0, 0.01
        if self.robot == "franka":
            b.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=64)
            self.hand = add_franka_hand(b, pinch_pos=self.base, rotation=rotation, opening=self.opening)
        else:
            self.hand = add_allegro_hand(b, root_pos=self.base, rotation=rotation, finger_positions=self.open_fingers)
        self.added = add_solid_asset(
            b,
            self.name,
            resolution="coarse",
            pos=pos,
            rot=solid_rotation,
            scale=self.scale,
            density=180.0,
            particle_radius=self.radius,
            k_mu=8000.0,
            k_lambda=8000.0,
            k_damp=0.5,
        )
        b.add_shape_box(
            -1,
            xform=wp.transform(tuple(self.table_center), wp.quat_identity()),
            hx=0.6,
            hy=0.5,
            hz=0.13,
            color=(0.48, 0.51, 0.55),
            label="original start and placement table",
        )
        b.add_ground_plane()
        b.color()
        self.model = b.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 50000.0, 0.1, 1.0
        self.solver = SolverVBD(
            self.model,
            iterations=iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.006,
            particle_self_contact_gap=0.006,
            particle_vertex_contact_buffer_size=64,
            friction_epsilon=0.001,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.state0, self.state1, self.control = self.model.state(), self.model.state(), self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.times = [0.0, 0.25, 1.25, 1.5, 3.5, 4.0, 5.5, 7.0, 8.5, 9.5, 11.0, 11.5]
        offsets = np.asarray(
            [(0, 0, 0)] * 4
            + [(0, 0, 0.65)] * 2
            + [(0.24, 0.28, 0.65)] * 2
            + [(0.24, 0.28, 0.08)] * 2
            + [(0.24, 0.28, 0.33)] * 2
        )
        if self.name == "capsule":
            offsets[2:, 2] -= 0.010
            self.recipe += " The hand descends 10 mm during closure to grip the middle diameter."
        if self.name == "pad":
            offsets[8:10, 2] = 0.10
            self.recipe += " Placement remains 20 mm higher to clear the original horizontal palm from the table."
        if self.robot == "franka":
            self.driver = FrankaHandDriver(
                self.model,
                self.hand,
                times=self.times,
                positions=self.base + offsets,
                rotations=rotation,
                openings=[self.opening] * 2 + [self.closed] * 7 + [self.opening] * 3,
            )
        else:
            self.driver = AllegroHandDriver(
                self.model,
                self.hand,
                times=self.times,
                positions=self.base + offsets,
                rotations=rotation,
                finger_positions=[self.open_fingers] * 2 + [self.closed] * 7 + [self.open_fingers] * 3,
            )
        self.description = (
            self.recipe
            + " Every soft vertex is dynamic. Original robot root and joints are prescribed; no pins, attachments, added jaws or actuator-capacity claim."
        )
        if self.name == "block":
            self.description += (
                " The original fingers start open beside the block; approach to this cleared pose is not shown."
            )
        self.driver.update(self.clock, self.state0, self.state1)
        self._shape_body = self.model.shape_body.numpy()
        self._shape_local = self.model.shape_transform.numpy()
        self._shape_margin = self.model.shape_margin.numpy()
        self._proxies = {}
        kinds, scales = self.model.shape_type.numpy(), self.model.shape_scale.numpy()
        for i in self.hand.shape_ids:
            if kinds[i] == newton.GeoType.SPHERE:
                self._proxies[i] = (np.zeros(3), float(scales[i, 0]))
            elif kinds[i] == newton.GeoType.BOX:
                self._proxies[i] = (np.zeros(3), scales[i])
            else:
                vertices = np.asarray(self.model.shape_source[i].vertices) * scales[i]
                self._proxies[i] = ((vertices.max(0) + vertices.min(0)) / 2, (vertices.max(0) - vertices.min(0)) / 2)
        self.initial_q = self.state.particle_q.numpy()
        self.tets = self.model.tet_indices.numpy()
        rest = self.initial_q[self.tets]
        self.rest_matrices = np.swapaxes(rest[:, 1:] - rest[:, :1], 1, 2)
        self.inverse_rest = np.linalg.inv(self.rest_matrices)
        self.initial_geometry = self._geometry()
        if self.initial_geometry["minimum_shell_to_robot_distance_m"] < -1e-6:
            raise ValueError(f"Initial solid overlaps original robot shell: {self.initial_geometry}")
        if self.initial_geometry["robot_table_separation_bound_m"] < 0:
            raise ValueError("Initial original hand intersects the finite table bound")
        self.history = []
        self._record()
        self.graph = None
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self.graph = capture.graph

    @property
    def state(self):
        return self.state0

    @property
    def phase(self):
        for end, label in [
            (0.25, "initial support"),
            (1.25, "close"),
            (1.5, "acquire"),
            (3.5, "lift"),
            (4.0, "pause"),
            (5.5, "carry"),
            (7.0, "hold"),
            (8.5, "place"),
            (9.5, "open"),
            (11.0, "withdraw"),
            (12.0, "released observation"),
        ]:
            if self.time < end - 1e-6:
                return label
        return "done"

    def _geometry(self):
        q = self.state.particle_q.numpy()
        if self.robot == "franka":
            raw = self.driver.geometry_audit(self.state, points=q, radii=self.radius, full_surface_contact=True)
            distance = raw["minimum_shell_to_robot_distance_m"]
            inside = raw["vertices_inside_robot_solids"]
        else:
            raw = self.driver.geometry_audit(self.state, points=q, radii=self.radius)
            distance = raw["minimum_shell_to_hand_distance_m"]
            inside = raw["vertices_inside_original_hand_solids"]
        bodies = self.state.body_q.numpy()
        separation, ground = np.inf, np.inf
        for i, (offset, half) in self._proxies.items():
            tf = wp.transform(*bodies[self._shape_body[i]]) * wp.transform(*self._shape_local[i])
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(tf))).reshape(3, 3)
            center = np.asarray(wp.transform_get_translation(tf)) + rotation @ offset
            if np.isscalar(half):
                delta = abs(center - self.table_center) - self.table_half
                gap = np.linalg.norm(np.maximum(delta, 0)) + min(delta.max(), 0) - half
                low = center[2] - half
            else:
                gap = _separation(center, rotation, half, self.table_center, self.table_half)
                low = center[2] - abs(rotation[2]) @ half
            separation = min(separation, float(gap - self._shape_margin[i]))
            ground = min(ground, float(low - self._shape_margin[i]))
        return {
            "minimum_shell_to_robot_distance_m": distance,
            "vertices_inside_original_robot_solids": inside,
            "robot_table_separation_bound_m": separation,
            "robot_ground_clearance_m": ground,
        }

    def _simulate(self):
        self.solver.rebuild_bvh(self.state0)
        for _ in range(self.substeps):
            self.driver.update(self.clock, self.state0, self.state1, dt=self.dt)
            self.state0.clear_forces()
            self.pipeline.collide(self.state0, self.contacts)
            self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
            self.state0, self.state1 = self.state1, self.state0
            wp.launch(_advance_clock, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def _record(self):
        q, v = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        tet = q[self.tets]
        f = np.swapaxes(tet[:, 1:] - tet[:, :1], 1, 2) @ self.inverse_rest
        ratios = np.linalg.det(f)
        stretches = np.linalg.svd(f, compute_uv=False)
        self.history.append(
            {
                "time_s": self.time,
                "phase": self.phase,
                "finite": bool(np.isfinite(q).all() and np.isfinite(v).all()),
                "center_m": q.mean(0).tolist(),
                "minimum_surface_clearance_m": float(q[:, 2].min() - self.radius - self.table_top),
                "minimum_volume_ratio": float(ratios.min()),
                "inverted_tets": int(np.sum(ratios <= 0)),
                "principal_stretch_max": float(stretches.max()),
                "max_speed_m_s": float(np.linalg.norm(v, axis=1).max()),
                **self._geometry(),
            }
        )

    def step(self):
        if self.graph is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph)
        self.time += self.frame_dt
        self._record()

    def metrics(self):
        hold = [r for r in self.history if 5.5 - 1e-6 <= r["time_s"] <= 7 + 1e-6]
        final = self.history[-1]
        return {
            "scene_id": self.scene_id,
            "finite": all(r["finite"] for r in self.history),
            "physics_parameters": {
                "frame_dt_s": self.frame_dt,
                "substeps": self.substeps,
                "substep_dt_s": self.dt,
                "iterations": self.iterations,
                "density_kg_m3": 180.0,
                "tet_shear_modulus_pa": 8000.0,
                "tet_first_lame_parameter_pa": 8000.0,
                "tet_damping_pa_s": 0.5,
                "particle_radius_m": self.radius,
                "soft_contact_ke": 50000.0,
                "soft_contact_kd": 0.1,
                "soft_contact_mu": 1.0,
                "self_contact_enabled": True,
                "self_contact_margin_m": 0.006,
                "self_contact_gap_m": 0.006,
                "rigid_shape_margin_m": 0.0,
                "rigid_shape_gap_m": 0.01,
                "full_surface_rigid_soft_contact": True,
                "original_palm_mesh_SDF_resolution": 64 if self.robot == "franka" else None,
                "external_rigid_bridge": True,
            },
            "asset": {
                "name": self.name,
                "resolution": "coarse",
                "geometry_sha256": self.mesh_hash,
                "scale": self.scale,
                "dimensions_m": self.object_dimensions.tolist(),
                "position_m": self.asset_position.tolist(),
                "rotation_xyzw": self.asset_rotation,
                "mass_kg": float(self.model.particle_mass.numpy().sum()),
                "particles": self.model.particle_count,
                "tetrahedra": self.model.tet_count,
                "original_max_dimension_m": 0.16,
            },
            "robot_asset": self.hand.provenance,
            "grasp_recipe": self.recipe,
            "robot_trajectory": {
                "times_s": self.times,
                "lift_m": 0.65,
                "carry_xy_m": [0.24, 0.28],
                "place_lowering_m": 0.55 if self.name == "pad" else 0.57,
                "closed_command": list(self.closed) if self.robot == "allegro" else self.closed,
                "open_command": list(self.open_fingers) if self.robot == "allegro" else self.opening,
                "initial_robot_reference_m": self.base.tolist(),
            },
            "pinned_particle_count": int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0)),
            "boundary_conditions": "Only original robot bodies prescribed; every soft vertex dynamic. No attachments or added gripper shapes.",
            "initial_geometry": self.initial_geometry,
            "geometry_method": "Original primitive SAT/sphere bounds; conservative palm-mesh local bounding box; vertex-to-hand signed distance.",
            "hold_window_s": [5.5, 7.0],
            "hold_minimum_surface_clearance_m": min((r["minimum_surface_clearance_m"] for r in hold), default=None),
            "hold_minimum_center_rise_m": min(
                (r["center_m"][2] - self.initial_q[:, 2].mean() for r in hold), default=None
            ),
            "maximum_inverted_tets": max(r["inverted_tets"] for r in self.history),
            "minimum_volume_ratio_over_frames": min(r["minimum_volume_ratio"] for r in self.history),
            "maximum_principal_stretch_over_frames": max(r["principal_stretch_max"] for r in self.history),
            "minimum_robot_table_separation_bound_m": min(r["robot_table_separation_bound_m"] for r in self.history),
            "retention_observed": bool(hold and min(r["minimum_surface_clearance_m"] for r in hold) > 0.03),
            "release_observed": bool(
                self.time > 11
                and final["minimum_shell_to_robot_distance_m"] > 0.05
                and final["minimum_surface_clearance_m"] < 0.02
            ),
            "trajectory": self.history,
            "qualification": "Smoke only; actual retention and material quality are reported separately.",
        }


def build_scene(scene_id, **kwargs):
    return SolidRobotScene(scene_id, **kwargs)
