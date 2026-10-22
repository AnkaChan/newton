# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared contact holds for original bag assets, with optional soft payloads.

These scenes start with jaws around supported handle or rim patches. They test
contact support, lift, carry, and release, not approach or grasp acquisition.
All cloth and payload particles remain dynamic throughout each run.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .assets import add_cloth_asset, add_solid_asset, cloth_mesh

SCENE_IDS = ("m1_tote_empty", "m1_tote_2", "m1_gusset_4", "m1_pouch_empty", "m1_tote_2_spheres")
__all__ = ["SCENE_IDS", "build_scene"]


@wp.func
def _ramp(t: float, start: float, end: float):
    u = wp.clamp((t - start) / (end - start), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@wp.func
def _jaw_offset(t: float, sign: float):
    return wp.vec3(
        0.22 * _ramp(t, 2.5, 4.0),
        0.16 * _ramp(t, 2.5, 4.0) + sign * 0.055 * _ramp(t, 6.8, 7.4),
        0.30 * _ramp(t, 0.5, 2.2) - 0.28 * _ramp(t, 5.5, 6.8) + 0.18 * _ramp(t, 7.4, 8.5),
    )


@wp.kernel
def _drive_jaws(
    clock: wp.array[float],
    ids: wp.array[int],
    bases: wp.array[wp.vec3],
    signs: wp.array[float],
    dt: float,
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    v1: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    offset = _jaw_offset(clock[0], signs[i])
    velocity = (_jaw_offset(clock[0] + dt, signs[i]) - offset) / dt
    pose = wp.transform(bases[i] + offset, wp.quat_identity())
    body = ids[i]
    q0[body] = pose
    q1[body] = pose
    v0[body] = wp.spatial_vector(velocity, wp.vec3(0.0))
    v1[body] = wp.spatial_vector(velocity, wp.vec3(0.0))


@wp.kernel
def _tick(clock: wp.array[float], dt: float):
    clock[0] += dt


class _BagScene:
    def __init__(self, scene_id, *, device=None, use_cuda_graph=True, substeps=16, iterations=12, **_kwargs):
        if scene_id not in SCENE_IDS:
            raise ValueError(f"Unknown bag scene {scene_id!r}; choose {SCENE_IDS}")
        self.scene_id = scene_id
        self.time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.duration = 9.0
        self.substeps = int(substeps)
        self.iterations = int(iterations)
        if self.substeps < 2 or self.substeps % 2:
            raise ValueError("Bag scene substeps must be an even integer >= 2")
        self.dt = self.frame_dt / self.substeps
        self.table_top = 0.18
        self.particle_radius = 0.003
        self.bag_name = "gusset_bag" if "gusset" in scene_id else "pouch" if "pouch" in scene_id else "tote"
        self.payload_count = 4 if scene_id.endswith("_4") else 2 if "_2" in scene_id else 0
        self.payload_name = "sphere" if scene_id.endswith("_spheres") else "block"
        self.title = f"M1 | {self.bag_name.replace('_', ' ')}: {'empty' if not self.payload_count else str(self.payload_count) + ' soft payloads'} | prepared hold"
        self.description = (
            "Prepared contact hold, not approach/acquisition: supported bag starts between prescribed jaws. "
            "Tote and gusset use two jaw pairs on separate handle tops; pouch uses one pair on its front rim. "
            "Jaws lift 0.30 m, carry 0.22 m in X and 0.16 m in Y, hold, place, open, and withdraw. "
            "No bag or payload particle is pinned."
        )
        self.asset_ids = [self.bag_name, "G0_contact_jaw_fixture"] + [self.payload_name] * self.payload_count
        self.camera_pos = (1.28, -1.65, 1.38)
        self.camera_target = (0.08, 0.04, 0.49 if self.bag_name != "pouch" else 0.34)
        self.history = []
        self._finite = True
        self._max_speed = 0.0
        self._min_gap = float("inf")

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 0.1
        bottom = self.table_top + 0.005
        self.bag = add_cloth_asset(
            builder,
            self.bag_name,
            resolution="coarse",
            pos=(0.0, 0.0, bottom),
            density=0.22,
            particle_radius=self.particle_radius,
            tri_ke=3000.0,
            tri_ka=3000.0,
            tri_kd=0.025,
            edge_ke=0.025,
            edge_kd=0.0005,
        )
        self.payloads = []
        payload_positions = (
            [(-0.06, 0.0), (0.06, 0.0)]
            if self.payload_count == 2
            else [(-0.05, -0.026), (0.05, -0.026), (-0.05, 0.026), (0.05, 0.026)]
        )
        colors = [(0.93, 0.33, 0.19), (0.20, 0.55, 0.85), (0.36, 0.72, 0.40), (0.66, 0.39, 0.83)]
        for i in range(self.payload_count):
            x, y = payload_positions[i]
            self.payloads.append(
                add_solid_asset(
                    builder,
                    self.payload_name,
                    resolution="coarse",
                    pos=(x, y, bottom + 0.012),
                    density=180.0,
                    particle_radius=0.0025,
                    k_mu=8000.0,
                    k_lambda=8000.0,
                    k_damp=0.5,
                    color=colors[i],
                )
            )
        table_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5.0e4, kd=0.1, mu=0.6)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.12, 0.07, self.table_top - 0.035), wp.quat_identity()),
            hx=0.57,
            hy=0.44,
            hz=0.035,
            cfg=table_cfg,
            color=(0.43, 0.48, 0.54),
            label="fixed_start_and_placement_table",
        )
        builder.add_ground_plane()

        mesh = cloth_mesh(self.bag_name, resolution="coarse")
        height = float(mesh.vertices[:, 2].max())
        depth = float(mesh.metadata["bag_body_dimensions_m"][1])
        if self.bag_name == "pouch":
            grip_centers = [(0.0, -depth / 2, bottom + height - 0.012)]
            half = (0.058, 0.004, 0.025)
        else:
            grip_centers = [(0.0, -depth / 2, bottom + height - 0.0125), (0.0, depth / 2, bottom + height - 0.0125)]
            half = (0.040, 0.012, 0.030)
        self.jaw_ids = []
        self._jaw_half_height = half[2]
        bases = []
        signs = []
        jaw_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=5.0e4, kd=0.1, mu=1.5)
        for center in grip_centers:
            for sign in (-1.0, 1.0):
                p = np.asarray(center) + np.array([0.0, sign * (half[1] + 0.002), 0.0])
                body = builder.add_body(
                    xform=wp.transform(wp.vec3(*map(float, p)), wp.quat_identity()),
                    is_kinematic=True,
                    label="prepared_contact_jaw",
                )
                builder.add_shape_box(
                    body=body, hx=half[0], hy=half[1], hz=half[2], cfg=jaw_cfg, color=(0.26, 0.37, 0.48)
                )
                self.jaw_ids.append(body)
                bases.append(p)
                signs.append(sign)

        builder.color()
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 1.5
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.005,
            particle_self_contact_gap=0.005,
            particle_vertex_contact_buffer_size=128,
            friction_epsilon=0.001,
            integrate_with_external_rigid_solver=False,
            rigid_compliant_alm=True,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        self._jaw_ids = wp.array(self.jaw_ids, dtype=int, device=self.model.device)
        self._bases = wp.array(bases, dtype=wp.vec3, device=self.model.device)
        self._signs = wp.array(signs, dtype=float, device=self.model.device)
        self.initial_q = self.state.particle_q.numpy()
        signed_distances = []
        for base in bases:
            d = np.abs(self.initial_q - base) - half
            signed_distances.append(np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0))
        jaw_distance = np.min(signed_distances, axis=0)
        radii = self.model.particle_radius.numpy()
        self.initial_geometry = {
            "material_vertices_inside_jaw_solid": int(np.count_nonzero(jaw_distance < -1.0e-6)),
            "minimum_material_vertex_to_jaw_distance_m": float(jaw_distance.min()),
            "maximum_intended_shell_overlap_with_jaw_m": float(np.maximum(radii - jaw_distance, 0.0).max()),
            "minimum_particle_surface_table_gap_m": float((self.initial_q[:, 2] - radii - self.table_top).min()),
            "payload_start_minimum_surface_z_m": [
                float(
                    (
                        self.initial_q[a.particle_start : a.particle_end, 2] - radii[a.particle_start : a.particle_end]
                    ).min()
                )
                for a in self.payloads
            ],
        }
        if self.initial_geometry["material_vertices_inside_jaw_solid"]:
            raise ValueError("Prepared bag pose puts a material vertex inside a jaw solid")
        self.initial_center = self.initial_q[self.bag.particle_start : self.bag.particle_end].mean(axis=0)
        self._particle_masses = self.model.particle_mass.numpy()
        self._graph = None
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self._graph = capture.graph

    @property
    def state(self):
        return self.state_0

    @property
    def phase(self):
        for end, phase in (
            (0.5, "prepared contact hold"),
            (2.2, "lift"),
            (2.5, "raised pause"),
            (4.0, "carry"),
            (5.5, "hold"),
            (6.8, "place"),
            (7.4, "open"),
            (8.5, "withdraw"),
            (10.0, "released observation"),
        ):
            if self.time < end:
                return phase
        return "complete"

    def _simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            wp.launch(
                _drive_jaws,
                dim=len(self.jaw_ids),
                inputs=[
                    self._clock,
                    self._jaw_ids,
                    self._bases,
                    self._signs,
                    self.dt,
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(_tick, dim=1, inputs=[self._clock, self.dt], device=self.model.device)

    def step(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        self.time += self.frame_dt
        q = self.state.particle_q.numpy()
        v = self.state.particle_qd.numpy()
        self._finite = self._finite and bool(np.isfinite(q).all() and np.isfinite(v).all())
        self._max_speed = max(self._max_speed, float(np.linalg.norm(v, axis=1).max()))
        bag_q = q[self.bag.particle_start : self.bag.particle_end]
        self._min_gap = min(self._min_gap, float(bag_q[:, 2].min() - self.particle_radius - self.table_top))
        payloads = []
        for added in self.payloads:
            points = q[added.particle_start : added.particle_end]
            payloads.append(
                {
                    "center": points.mean(axis=0).tolist(),
                    "min_z_m": float(points[:, 2].min()),
                    "max_z_m": float(points[:, 2].max()),
                }
            )
        self.history.append(
            {
                "time_s": self.time,
                "phase": self.phase,
                "bag_center": bag_q.mean(axis=0).tolist(),
                "bag_min_z_m": float(bag_q[:, 2].min()),
                "bag_max_z_m": float(bag_q[:, 2].max()),
                "payloads": payloads,
            }
        )

    def metrics(self):
        hold = [r for r in self.history if 4.1 <= r["time_s"] <= 5.4]
        bag_mass = float(self._particle_masses[self.bag.particle_start : self.bag.particle_end].sum())
        payload_masses = [float(self._particle_masses[a.particle_start : a.particle_end].sum()) for a in self.payloads]
        rise = min(r["bag_center"][2] for r in hold) - float(self.initial_center[2]) if hold else None
        min_clearance = min(r["bag_min_z_m"] for r in hold) - self.particle_radius - self.table_top if hold else None
        payload_hold = []
        for i in range(self.payload_count):
            payload_hold.append(
                {
                    "index": i,
                    "minimum_center_z_m": min(r["payloads"][i]["center"][2] for r in hold) if hold else None,
                    "minimum_vertex_z_m": min(r["payloads"][i]["min_z_m"] for r in hold) if hold else None,
                }
            )
        final_q = self.state.particle_q.numpy()[self.bag.particle_start : self.bag.particle_end]
        jaw_lowest_z = float(self.state.body_q.numpy()[self.jaw_ids, 2].min()) - self._jaw_half_height
        released = self.time >= 8.5 and float(final_q[:, 2].max()) < jaw_lowest_z - 0.03
        return {
            "scene_id": self.scene_id,
            "solver": "VBD",
            "finite": self._finite,
            "time_s": self.time,
            "particle_count": self.model.particle_count,
            "bag_particle_count": self.bag.particle_end - self.bag.particle_start,
            "bag_mass_kg": bag_mass,
            "payload_count": self.payload_count,
            "payload_shape": self.payload_name if self.payload_count else None,
            "payload_mass_kg": payload_masses,
            "substeps": self.substeps,
            "iterations": self.iterations,
            "dt_s": self.dt,
            "grasp_protocol": "prepared contact hold; no approach or acquisition claimed",
            "jaw_pair_count": len(self.jaw_ids) // 2,
            "boundary_conditions": "Only rigid jaw poses are prescribed; all bag and payload particles remain dynamic.",
            "pinned_particle_count": int(np.count_nonzero(self.model.particle_inv_mass.numpy() == 0.0)),
            "initial_geometry_audit": self.initial_geometry,
            "table_top_z_m": self.table_top,
            "minimum_bag_surface_height_relative_table_m": self._min_gap if np.isfinite(self._min_gap) else None,
            "surface_height_metric_note": "Particle Z minus shell radius and table-top Z. This is a vertical height, not a footprint-aware table-penetration test.",
            "initial_bag_center": self.initial_center.tolist(),
            "hold_minimum_bag_center_rise_m": rise,
            "hold_minimum_bag_surface_clearance_m": min_clearance,
            "hold_bag_center_translation_xy_m": np.mean([r["bag_center"] for r in hold], axis=0)[:2].tolist()
            if hold
            else None,
            "payload_hold": payload_hold,
            "lift_observed": bool(min_clearance is not None and min_clearance > 0.05 and rise > 0.10),
            "release_observed": bool(released),
            "max_particle_speed_m_per_s": self._max_speed,
            "final_bag_center": self.state.particle_q.numpy()[self.bag.particle_start : self.bag.particle_end]
            .mean(axis=0)
            .tolist(),
            "trajectory": self.history,
            "qualification": "Setup smoke only. Prepared contact support does not establish grasp acquisition or dynamic robot capacity. No 10 s or 60 s hold qualification.",
        }


def build_scene(scene_id, **kwargs):
    """Build an original bag handling scene, with dimensions in metres."""
    return _BagScene(scene_id, **kwargs)
