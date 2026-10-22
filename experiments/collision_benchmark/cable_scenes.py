# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small VBD cable scenes for collision-benchmark setup checks.

These are smoke recipes, not qualified friction or stability benchmarks. Cable
contact geometry is the actual capsule segments, with radius 9 mm. Kinematic
jaws have prescribed motion and unlimited actuation. Only the explicitly
listed endpoint fixtures constrain cable bodies directly.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

SCENE_IDS = ("c6_crossed_cables", "c7_cylinder_wrap", "c8_bundle_squeeze", "c8_pile_pickup")
__all__ = ["SCENE_IDS", "build_scene"]


@wp.kernel
def _advance_time(clock: wp.array[float], dt: float):
    clock[0] += dt


@wp.kernel
def _prescribe_bodies(
    clock: wp.array[float],
    ids: wp.array[int],
    base: wp.array[wp.transform],
    times: wp.array[float],
    offsets: wp.array2d[wp.vec3],
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    qd0: wp.array[wp.spatial_vector],
    qd1: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    t = clock[0]
    offset = offsets[i, times.shape[0] - 1]
    velocity = wp.vec3(0.0)
    for k in range(times.shape[0] - 1):
        if t >= times[k] and t < times[k + 1]:
            span = times[k + 1] - times[k]
            u = wp.clamp((t - times[k]) / span, 0.0, 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            delta = offsets[i, k + 1] - offsets[i, k]
            offset = offsets[i, k] + blend * delta
            velocity = (6.0 * u * (1.0 - u) / span) * delta
    body = ids[i]
    q = wp.transform(wp.transform_get_translation(base[i]) + offset, wp.transform_get_rotation(base[i]))
    q0[body] = q
    q1[body] = q
    qd0[body] = wp.spatial_vector(velocity, wp.vec3(0.0))
    qd1[body] = wp.spatial_vector(velocity, wp.vec3(0.0))


@wp.kernel
def _load_wrap_ends(clock: wp.array[float], left: int, right: int, body_f: wp.array[wp.spatial_vector]):
    # Known downward forces supplement gravity. The right/left ratio is
    # 1 until 3 s, rises to 2.5 by 5 s, then returns to 1 by 7 s.
    t = clock[0]
    ramp = wp.clamp(t / 1.0, 0.0, 1.0)
    ratio = 1.0 + 1.5 * wp.clamp((t - 3.0) / 2.0, 0.0, 1.0)
    ratio -= 1.5 * wp.clamp((t - 5.0) / 2.0, 0.0, 1.0)
    body_f[left] = wp.spatial_vector(wp.vec3(0.0, 0.0, -0.4 * ramp), wp.vec3(0.0))
    body_f[right] = wp.spatial_vector(wp.vec3(0.0, 0.0, -0.4 * ramp * ratio), wp.vec3(0.0))


def _freeze(builder: newton.ModelBuilder, body: int):
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def _line(start, end, count):
    return np.linspace(np.asarray(start), np.asarray(end), count + 1)


class _CableScene:
    def __init__(self, scene_id, *, device=None, substeps=12, iterations=8, duration=None, **_kwargs):
        if scene_id not in SCENE_IDS:
            raise ValueError(f"Unknown cable scene {scene_id!r}; choose {SCENE_IDS}")
        self.scene_id = scene_id
        self.time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.duration = 9.0 if duration is None else float(duration)
        self.substeps = int(substeps)
        if self.substeps < 2 or self.substeps % 2:
            raise ValueError("Cable scene substeps must be a positive even integer >= 2 for CUDA graph state reuse")
        self.sim_dt = self.frame_dt / self.substeps
        self.iterations = int(iterations)
        self.radius = 0.009
        self.cable_bodies = []
        self.cables = []
        self.jaw_bodies = []
        self._grasp_x = 0.10
        self.fixture_ids = []
        self.fixture_notes = []
        self._moving_ids = []
        self._offsets = []
        self._times = np.asarray([0.0, 1.5, 2.5, 4.5, 6.5, 7.5, 9.0], dtype=np.float32)
        self._max_selected_mean_z = -math.inf
        self._hold_min_z = []
        self._hold_min_surface_z = []
        self._hold_mean_z = []
        self._hold_grip_z = []
        self._finite = True
        self._max_speed = 0.0
        self._max_contact_count = 0
        self._min_z = math.inf
        self._min_surface_z = math.inf
        self._trace = []

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        self._cfg = newton.ModelBuilder.ShapeConfig(density=500.0, ke=1.0e5, kd=0.0, mu=1.0, margin=0.0, gap=0.002)
        ground_cfg = self._cfg.copy()
        ground_cfg.mu = 0.5
        builder.add_ground_plane(cfg=ground_cfg)
        if scene_id == "c6_crossed_cables":
            self._build_crossed(builder)
        elif scene_id == "c7_cylinder_wrap":
            self._build_wrap(builder)
        elif scene_id == "c8_bundle_squeeze":
            self._build_bundle(builder)
        else:
            self._build_pile(builder)

        builder.color()
        self.model = builder.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="sticky")
        self.contacts = self.collision_pipeline.contacts()
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            rigid_compliant_alm=True,
            rigid_body_contact_buffer_size=128,
            rigid_contact_history=True,
        )
        self._clock = wp.zeros(1, dtype=float, device=self.model.device)
        self._times_wp = wp.array(self._times, dtype=float, device=self.model.device)
        self._moving_ids_wp = wp.array(self._moving_ids, dtype=int, device=self.model.device)
        poses = self.state_0.body_q.numpy()
        self._base_wp = wp.array(poses[self._moving_ids], dtype=wp.transform, device=self.model.device)
        self._offsets_wp = wp.array(
            np.asarray(self._offsets, dtype=np.float32).reshape((-1, len(self._times), 3)),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._initial_selected_mean_z = float(poses[self.cables[0], 2].mean())
        self._initial_selected_positions = poses[self.cables[0], :3].copy()
        self._initial_mass = float(self.model.body_mass.numpy()[self.cable_bodies].sum())
        shape_bodies = self.model.shape_body.numpy()
        shape_scales = self.model.shape_scale.numpy()
        self._half_lengths = np.asarray(
            [shape_scales[np.flatnonzero(shape_bodies == body)[0], 1] for body in self.cable_bodies]
        )
        self._graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._simulate()
            self._graph = capture.graph

    @property
    def state(self):
        return self.state_0

    @property
    def phase(self):
        t = self.time
        if self.scene_id == "c7_cylinder_wrap":
            return (
                "preload"
                if t < 1.0
                else "equal_load_hold"
                if t < 3.0
                else "load_ratio_up"
                if t < 5.0
                else "load_ratio_down"
                if t < 7.0
                else "recovery"
            )
        if self.scene_id == "c6_crossed_cables":
            return (
                "preload"
                if t < 1.5
                else "loaded_hold"
                if t < 2.5
                else "upper_cable_slide"
                if t < 4.5
                else "post_slide_hold"
            )
        if self.scene_id == "c8_bundle_squeeze":
            return (
                "settle"
                if t < 1.5
                else "squeeze"
                if t < 2.5
                else "hold"
                if t < 4.5
                else "strand_pull"
                if t < 6.5
                else "release"
                if t < 7.5
                else "recovery"
            )
        return (
            "settle"
            if t < 1.5
            else "close"
            if t < 2.5
            else "lift"
            if t < 4.5
            else "hold"
            if t < 6.5
            else "release"
            if t < 7.5
            else "fall_and_settle"
        )

    def _rod(self, builder, points, label, color, *, edges=None, bend=0.02):
        rod = newton.Rod([wp.vec3(*map(float, p)) for p in points], edges=edges, radius=self.radius)
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=self._cfg,
            stretch_stiffness=5.0e4,
            stretch_damping=0.1,
            bend_stiffness=bend,
            bend_damping=0.0005,
            label=label,
            color=color,
            body_frame_origin="com",
        )
        bodies = [int(b) for b in bodies]
        self.cable_bodies.extend(bodies)
        self.cables.append(bodies)
        return bodies

    def _move(self, builder, body, offsets, *, fixture=False):
        _freeze(builder, body)
        self._moving_ids.append(body)
        self._offsets.append(offsets)
        if fixture:
            self.fixture_ids.append(body)

    def _jaw(self, builder, position, half_extents, offsets):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*position), wp.quat_identity()), label="prescribed_contact_jaw"
        )
        cfg = self._cfg.copy()
        cfg.mu = 1.5
        builder.add_shape_box(
            body=body, hx=half_extents[0], hy=half_extents[1], hz=half_extents[2], cfg=cfg, color=(0.25, 0.33, 0.43)
        )
        self._move(builder, body, offsets)
        self.jaw_bodies.append(body)
        return body

    def _build_crossed(self, builder):
        self.title = "C6 — loaded cable crossing and slide"
        self.description = "Two orthogonal VBD cables. Four endpoint capsules are prescribed fixtures. A 2 mm endpoint stretch loads the crossing; both upper endpoints then move 14 cm in X."
        self.asset_ids = ["C0_straight_flexible", "C0_straight_stiffer"]
        lower = self._rod(
            builder, _line((-0.36, 0.0, 0.10), (0.36, 0.0, 0.10), 28), "lower_cable", (0.18, 0.55, 0.85), bend=0.07
        )
        upper = self._rod(
            builder, _line((0.0, -0.36, 0.12), (0.0, 0.36, 0.12), 28), "upper_cable", (0.95, 0.48, 0.15), bend=0.02
        )
        for ids, axis in ((lower, 0), (upper, 1)):
            for body, sign in ((ids[0], -1.0), (ids[-1], 1.0)):
                offsets = np.zeros((7, 3))
                offsets[1:, axis] = sign * 0.002
                if ids is upper:
                    offsets[3:, 0] = 0.14
                self._move(builder, body, offsets, fixture=True)
                marker_cfg = self._cfg.copy()
                marker_cfg.density = 0.0
                marker_cfg.has_shape_collision = False
                builder.add_shape_box(body=body, hx=0.022, hy=0.022, hz=0.014, cfg=marker_cfg, color=(0.32, 0.35, 0.42))
        self.fixture_notes = [
            "First and last capsule of each cable are mass-zero endpoint fixtures.",
            "The 2 mm endpoint extension is prescribed displacement, not a measured tension.",
        ]
        self.camera_pos = (1.1, -1.15, 0.85)
        self.camera_target = (0.03, 0.0, 0.07)

    def _build_wrap(self, builder):
        self.title = "C7 — cable wrap with known end loads"
        self.description = "Long VBD cable with a 180 degree initial wrap around a fixed cylinder. Downward endpoint forces ramp to 0.4 N per end; the right/left ratio rises to 2.5 and returns to 1. Gravity remains active."
        self.asset_ids = ["C1_long_cable", "fixed_cylinder"]
        center_z, radius = 0.46, 0.09
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, center_z), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2)
            ),
            radius=radius,
            half_height=0.065,
            cfg=self._cfg,
            color=(0.3, 0.39, 0.48),
        )
        r = radius + self.radius + 0.0008
        points = list(_line((-r, 0.0, 0.16), (-r, 0.0, center_z), 12)[:-1])
        points.extend([(r * math.cos(a), 0.0, center_z + r * math.sin(a)) for a in np.linspace(math.pi, 0.0, 18)])
        points.extend(_line((r, 0.0, center_z), (r, 0.0, 0.16), 12)[1:])
        cable = self._rod(builder, points, "wrapped_long_cable", (0.94, 0.48, 0.17), bend=0.006)
        self._wrap_ends = (cable[0], cable[-1])
        self.fixture_notes = [
            "Cylinder is a fixed world shape.",
            "Neither cable end is pinned. Endpoint force directions remain vertical.",
            "Initial wrap angle is 180 degrees; actual contact arc is not yet measured.",
        ]
        self.camera_pos = (0.85, -1.25, 0.85)
        self.camera_target = (0.0, 0.0, 0.31)

    def _build_bundle(self, builder):
        self.title = "C8a — bundle squeeze, strand pull, release"
        self.description = "Three initially parallel cables on a plane. Prescribed jaws squeeze the bundle, hold it, and open. One strand has an explicit endpoint fixture that pulls 16 cm along its axis during the hold."
        self.asset_ids = ["C4_parallel_bundle"]
        colors = [(0.95, 0.47, 0.14), (0.15, 0.56, 0.84), (0.35, 0.74, 0.42)]
        for i, y in enumerate((0.0, -0.02, 0.02)):
            self._rod(
                builder, _line((-0.36, y, 0.012), (0.36, y, 0.012), 28), f"bundle_strand_{i}", colors[i], bend=0.02
            )
        offsets = np.zeros((7, 3))
        offsets[4:, 0] = 0.16
        self._move(builder, self.cables[0][-1], offsets, fixture=True)
        for sign in (-1.0, 1.0):
            offsets = np.zeros((7, 3))
            offsets[2:5, 1] = -sign * 0.029
            self._jaw(builder, (0.0, sign * 0.066, 0.038), (0.12, 0.012, 0.038), offsets)
        self.fixture_notes = [
            "One endpoint capsule of orange strand is an explicit pulling fixture throughout the run.",
            "Jaws are mass-zero boxes with prescribed closure; cable-to-jaw coupling is contact only.",
        ]
        self.camera_pos = (1.02, -1.0, 0.65)
        self.camera_target = (0.04, 0.0, 0.04)

    def _build_pile(self, builder):
        self.title = "C8b — loose cable pile pickup and release"
        self.description = "A loose assembly contains a straight pickup strand, a long curved strand, an open coil, and a branched cable. Prescribed parallel jaws grasp the orange strand through frictional contact, lift, hold for 2 s, and release. No cable body is pinned."
        self.asset_ids = [
            "C0_straight_flexible",
            "C1_long_cable",
            "C3_open_coil",
            "C5_loose_pile",
            "C2_branched_cable",
            "G0_parallel_jaw_fixture",
        ]
        # Add the selected strand first for consistent metrics, above the loose pile.
        self._rod(
            builder, _line((-0.36, 0.0, 0.065), (0.36, 0.0, 0.065), 28), "pickup_strand", (0.97, 0.42, 0.11), bend=0.008
        )
        points = [
            (-0.12 + 0.08 * math.sin(u * 2.0 * math.pi), -0.37 + 0.74 * u, 0.025) for u in np.linspace(0.0, 1.0, 35)
        ]
        self._rod(builder, points, "loose_long_strand", (0.18, 0.57, 0.85), bend=0.012)
        points = [
            (-0.13 + 0.105 * math.cos(a), 0.13 + 0.105 * math.sin(a), 0.014 + 0.001 * a)
            for a in np.linspace(0.0, 1.8 * math.pi, 28)
        ]
        self._rod(builder, points, "open_coil", (0.65, 0.42, 0.83), bend=0.008)
        junction = np.array([0.18, 0.23, 0.012])
        nodes = [junction]
        edges = []
        for angle in (0.0, 2.1, 4.2):
            previous = 0
            for i in range(1, 8):
                nodes.append(junction + 0.021 * i * np.array([math.cos(angle), math.sin(angle), 0.0]))
                edges.append((previous, len(nodes) - 1))
                previous = len(nodes) - 1
        self._rod(builder, nodes, "branched_cable", (0.31, 0.74, 0.43), edges=edges, bend=0.02)
        for sign in (-1.0, 1.0):
            offsets = np.zeros((7, 3))
            offsets[2:5, 1] = -sign * 0.0365
            offsets[3:, 2] = 0.45
            self._jaw(builder, (self._grasp_x, sign * 0.056, 0.043), (0.045, 0.012, 0.043), offsets)
        self.fixture_notes = [
            "No cable endpoint fixtures or cable-to-jaw constraints.",
            "Two mass-zero box jaws follow prescribed poses and surface velocities; grip coupling is collision contact.",
            "Jaws have unlimited actuation. This does not establish a dynamic robot's grasp capacity.",
        ]
        self.camera_pos = (1.08, -1.2, 0.95)
        self.camera_target = (0.0, 0.0, 0.23)

    def _simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
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
            if self.scene_id == "c7_cylinder_wrap":
                wp.launch(
                    _load_wrap_ends,
                    dim=1,
                    inputs=[self._clock, self._wrap_ends[0], self._wrap_ends[1], self.state_0.body_f],
                    device=self.model.device,
                )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(_advance_time, dim=1, inputs=[self._clock, self.sim_dt], device=self.model.device)

    def step(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        self.time += self.frame_dt
        q = self.state.body_q.numpy()
        velocity = self.state.body_qd.numpy()[self.cable_bodies]
        positions = q[self.cable_bodies, :3]
        self._finite = self._finite and bool(np.isfinite(positions).all() and np.isfinite(velocity).all())
        self._max_speed = max(self._max_speed, float(np.linalg.norm(velocity[:, :3], axis=1).max()))
        self._min_z = min(self._min_z, float(positions[:, 2].min()))
        cable_q = q[self.cable_bodies]
        axis_z = 1.0 - 2.0 * (cable_q[:, 3] ** 2 + cable_q[:, 4] ** 2)
        surface_z = cable_q[:, 2] - np.abs(axis_z) * self._half_lengths - self.radius
        self._min_surface_z = min(self._min_surface_z, float(surface_z.min()))
        selected_z = q[self.cables[0], 2]
        self._max_selected_mean_z = max(self._max_selected_mean_z, float(selected_z.mean()))
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        self._max_contact_count = max(self._max_contact_count, count)
        if self.scene_id == "c8_pile_pickup" and self.phase == "hold":
            self._hold_min_z.append(float(selected_z.min()))
            self._hold_min_surface_z.append(float(surface_z[: len(self.cables[0])].min()))
            self._hold_mean_z.append(float(selected_z.mean()))
            local = q[self.cables[0], :3]
            grip = local[np.abs(self._initial_selected_positions[:, 0] - self._grasp_x) < 0.04]
            self._hold_grip_z.append(float(grip[:, 2].mean()))
        if len(self._trace) == 0 or self.time - self._trace[-1]["time_s"] >= 0.099:
            self._trace.append(
                {
                    "time_s": self.time,
                    "phase": self.phase,
                    "selected_mean_z_m": float(selected_z.mean()),
                    "selected_min_z_m": float(selected_z.min()),
                    "contact_count": count,
                }
            )

    def metrics(self):
        q = self.state.body_q.numpy()
        selected = q[self.cables[0], :3]
        result = {
            "scene_id": self.scene_id,
            "solver": "VBD",
            "time_s": self.time,
            "duration_s": self.duration,
            "phase": self.phase,
            "finite": self._finite,
            "body_count": self.model.body_count,
            "cable_segment_count": len(self.cable_bodies),
            "cable_count": len(self.cables),
            "cable_radius_m": self.radius,
            "cable_contact_geometry": "actual capsule segments",
            "cable_mass_kg": self._initial_mass,
            "substeps": self.substeps,
            "iterations": self.iterations,
            "substep_dt_s": self.sim_dt,
            "contact_stiffness_N_per_m": 1.0e5,
            "cable_friction": 1.0,
            "jaw_friction": 1.5,
            "physical_contact_margin_m": 0.0,
            "detection_gap_m": 0.002,
            "max_speed_m_per_s": self._max_speed,
            "minimum_segment_com_z_m": self._min_z if math.isfinite(self._min_z) else None,
            "minimum_capsule_surface_z_m": self._min_surface_z if math.isfinite(self._min_surface_z) else None,
            "max_contact_count": self._max_contact_count,
            "selected_initial_mean_z_m": self._initial_selected_mean_z,
            "selected_final_mean_z_m": float(selected[:, 2].mean()),
            "selected_peak_mean_z_m": self._max_selected_mean_z if math.isfinite(self._max_selected_mean_z) else None,
            "fixture_body_ids": self.fixture_ids,
            "fixture_notes": self.fixture_notes,
            "qualification": "Setup smoke only; no 10 s or 60 s hold qualification, crossing audit, or substep jitter score.",
            "trace": self._trace,
        }
        if self.scene_id == "c8_pile_pickup":
            result.update(
                {
                    "grasp_coupling": "frictional contact only; no cable pins",
                    "hold_duration_s": 2.0,
                    "hold_minimum_segment_com_z_m": min(self._hold_min_z) if self._hold_min_z else None,
                    "hold_minimum_capsule_surface_z_m": min(self._hold_min_surface_z)
                    if self._hold_min_surface_z
                    else None,
                    "hold_minimum_mean_z_m": min(self._hold_mean_z) if self._hold_mean_z else None,
                    "hold_grip_mean_z_range_m": [min(self._hold_grip_z), max(self._hold_grip_z)]
                    if self._hold_grip_z
                    else None,
                    "lift_observed": bool(self._hold_mean_z and min(self._hold_mean_z) > 0.15),
                    "release_observed": bool(
                        self.time >= 8.5 and float(selected[:, 2].mean()) < 0.08 and self._max_selected_mean_z > 0.15
                    ),
                }
            )
        if self.scene_id == "c7_cylinder_wrap":
            result["endpoint_loads_N"] = {
                "left": 0.4,
                "right_peak": 1.0,
                "ratio_range": [1.0, 2.5],
                "direction": "world -Z, in addition to gravity",
            }
        return result


def build_scene(scene_id: str, **kwargs):
    """Build a cable smoke scene with SI units and one-frame ``step()``."""
    return _CableScene(scene_id, **kwargs)
