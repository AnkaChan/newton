# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact-only jaw manipulation smoke scenes. No deformable vertex is pinned."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

CLOTH_NAMES = ["patch", "towel", "sheet", "trousers", "dress", "tshirt", "skirt", "tote", "pouch", "gusset_bag"]
SOLID_NAMES = ["pad", "block", "sphere", "capsule", "torus", "branched"]
SCENE_IDS = (
    ["r1_" + s for s in CLOTH_NAMES] + ["r1_solid_" + s for s in SOLID_NAMES] + ["r2_franka_duck", "r3_allegro_sphere"]
)


@wp.func
def ramp(t: float, a: float, b: float):
    u = wp.clamp((t - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@wp.func
def jaw_pose(t: float, base: wp.vec3, sign: float, half: float, open_gap: float, closed_gap: float, axis: int):
    close = ramp(t, 0.25, 1.25) - ramp(t, 8.5, 9.5)
    gap = open_gap * (1.0 - close) + closed_gap * close
    shift = wp.vec3(
        0.24 * ramp(t, 4.0, 5.5),
        0.28 * ramp(t, 4.0, 5.5),
        0.65 * ramp(t, 1.5, 3.5) - 0.57 * ramp(t, 7.0, 8.5) + 0.25 * ramp(t, 9.5, 11.0),
    )
    if axis == 2:
        shift[2] += sign * (half + gap)
    else:
        shift[1] += sign * (half + gap)
    return wp.transform(base + shift, wp.quat_identity())


@wp.kernel
def drive(
    clock: wp.array[float],
    bodies: wp.array[int],
    base: wp.vec3,
    half: float,
    open_gap: float,
    closed_gap: float,
    axis: int,
    dt: float,
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    v1: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    sign = -1.0 + 2.0 * float(i)
    q = jaw_pose(clock[0], base, sign, half, open_gap, closed_gap, axis)
    qp = jaw_pose(clock[0] + dt, base, sign, half, open_gap, closed_gap, axis)
    velocity = (wp.transform_get_translation(qp) - wp.transform_get_translation(q)) / dt
    b = bodies[i]
    q0[b] = q
    q1[b] = q
    v0[b] = wp.spatial_vector(velocity, wp.vec3(0.0))
    v1[b] = wp.spatial_vector(velocity, wp.vec3(0.0))


@wp.kernel
def tick(clock: wp.array[float], dt: float):
    clock[0] += dt


class JawScene:
    def __init__(self, scene_id, *, device=None, use_cuda_graph=True, cloth_mesh_override=None, **kwargs):
        from experiments.collision_benchmark.assets import (  # noqa: PLC0415
            AddedAsset,
            add_solid_asset,
            cloth_mesh,
            solid_mesh,
        )

        self.scene_id = scene_id
        self.time = 0.0
        self.frame_dt = 1 / 30
        self.substeps = 12
        self.dt = self.frame_dt / self.substeps
        self.duration = 11.5
        self.is_solid = scene_id.startswith("r1_solid_")
        self.initial_recipe = scene_id.endswith("_initial")
        name = scene_id.removesuffix("_initial").removeprefix("r1_solid_" if self.is_solid else "r1_")
        self.title = f"R1 | {'soft solid' if self.is_solid else 'cloth'} {name}: pick, carry, release"
        self.description = "Prescribed rigid jaws close, lift, carry, hold, place, open, and withdraw. Every deformable vertex is free. VBD computes all soft-object motion and contact."
        self.asset_ids = [name, "G0_generated_jaws"]
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.mu = 1.0
        b.default_shape_cfg.ke = 50000
        b.default_shape_cfg.kd = 0.1
        self.axis = 1 if self.is_solid else 2
        self.half = 0.025
        if self.is_solid:
            mesh = solid_mesh(name, resolution="coarse")
            verts = mesh.vertices
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            scale = 0.16 / max(hi - lo)
            pos = (
                -float((lo[0] + hi[0]) * 0.5 * scale),
                -float((lo[1] + hi[1]) * 0.5 * scale),
                0.25 - float(lo[2] * scale) + 0.006,
            )
            self.added = add_solid_asset(
                b,
                name,
                resolution="coarse",
                pos=pos,
                scale=scale,
                density=180.0,
                particle_radius=0.003,
                k_mu=8000.0,
                k_lambda=8000.0,
                k_damp=0.5,
            )
            z = 0.256 + float((hi[2] - lo[2]) * scale) * 0.5
            self.base = (0, 0, z)
            self.closed_gap = float((hi[1] - lo[1]) * scale) * 0.5 - 0.008
            self.open_gap = self.closed_gap + 0.055
            extents = (0.07, self.half, min(0.055, float((hi[2] - lo[2]) * scale) * 0.4))
            table_pos, table_half = (0, 0, 0.12), (0.6, 0.5, 0.13)
            self.camera_pos = (1.15, -1.5, 1.3)
            self.camera_target = (0.12, 0.12, 0.50)
        else:
            mesh = cloth_mesh_override if cloth_mesh_override is not None else cloth_mesh(name, resolution="coarse")
            verts = mesh.vertices
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            scale = 0.6 / max(hi - lo)
            # Lay the x-z garment on a table. Its lower hem faces the jaws.
            rot = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2)
            pos = (
                -float((lo[0] + hi[0]) * 0.5 * scale),
                -float(lo[2] * scale),
                0.258 + (0.0 if self.initial_recipe else float(hi[1] * scale)),
            )
            particle_start = b.particle_count
            b.add_cloth_mesh(
                vertices=mesh.vertices.tolist(),
                indices=mesh.triangles.ravel().tolist(),
                vel=(0.0, 0.0, 0.0),
                color=mesh.metadata.get("color_rgb", (0.35, 0.62, 0.85)),
                pos=pos,
                rot=rot,
                scale=scale,
                density=0.22,
                particle_radius=0.003,
                tri_ke=1800.0,
                tri_ka=1800.0,
                tri_kd=0.025,
                edge_ke=0.08,
                edge_kd=0.0005,
            )
            self.added = AddedAsset(
                mesh,
                particle_start,
                b.particle_count,
                {k: [particle_start + i for i in ids] for k, ids in mesh.landmarks.items()},
            )
            # Choose a real material point near the lower edge, not a garment hole.
            q = np.asarray(b.particle_q)[self.added.particle_start : self.added.particle_end]
            front = q[q[:, 1] < q[:, 1].min() + 0.02]
            anchor = front[np.argmin(abs(front[:, 0]))]
            self.base = (
                float(anchor[0]),
                float(anchor[1]) + 0.015,
                0.258 if self.initial_recipe else float((q[:, 2].min() + q[:, 2].max()) * 0.5),
            )
            self.closed_gap = 0.002
            self.open_gap = 0.035 if self.initial_recipe else max(0.035, float(np.ptp(q[:, 2])) * 0.5 + 0.006)
            extents = (
                (0.10, 0.06, self.half)
                if name in ("dress", "trousers") and not self.initial_recipe
                else (0.06, 0.045, self.half)
            )
            table_pos, table_half = (0, 0.39, 0.12), (0.65, 0.31, 0.13)
            self.camera_pos = (1.1, -1.35, 1.0)
            self.camera_target = (0.08, 0.22, 0.43)
        b.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*table_pos), wp.quat_identity()),
            hx=table_half[0],
            hy=table_half[1],
            hz=table_half[2],
            color=(0.48, 0.51, 0.55),
            label="start and placement table",
        )
        cfg = newton.ModelBuilder.ShapeConfig(density=0, mu=1.0, ke=50000, kd=0.1)
        jaw_ids = []
        for i in range(2):
            p = np.array(self.base)
            p[self.axis] += (-1 + 2 * i) * (self.half + self.open_gap)
            body = b.add_body(
                xform=wp.transform(wp.vec3(*p), wp.quat_identity()), is_kinematic=True, label=f"prescribed jaw {i}"
            )
            b.add_shape_box(
                body,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
                cfg=cfg,
                color=(0.85, 0.43, 0.15) if i else (0.15, 0.43, 0.75),
            )
            jaw_ids.append(body)
        b.add_ground_plane()
        b.color()
        self.model = b.finalize(device=device)
        self.model.soft_contact_ke = 50000.0
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 1.0
        self.solver = SolverVBD(
            self.model,
            iterations=10,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.006,
            particle_self_contact_gap=0.006,
            particle_vertex_contact_buffer_size=64,
            friction_epsilon=0.001,
            integrate_with_external_rigid_solver=False,
            rigid_compliant_alm=True,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.01, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.jaws = wp.array(jaw_ids, dtype=int, device=self.model.device)
        self.initial_q = self.state.particle_q.numpy().copy()
        self.history = []
        self.graph = None
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    @property
    def state(self):
        return self.state0

    @property
    def phase(self):
        for end, name in [
            (0.25, "initial support"),
            (1.25, "close"),
            (1.5, "acquire"),
            (3.5, "lift"),
            (4, "pause"),
            (5.5, "carry"),
            (7, "hold"),
            (8.5, "place"),
            (9.5, "open"),
            (11, "withdraw"),
            (20, "released observation"),
        ]:
            if self.time < end:
                return name
        return "done"

    def simulate(self):
        self.solver.rebuild_bvh(self.state0)
        for _ in range(self.substeps):
            wp.launch(
                drive,
                dim=2,
                inputs=[
                    self.clock,
                    self.jaws,
                    wp.vec3(*self.base),
                    self.half,
                    self.open_gap,
                    self.closed_gap,
                    self.axis,
                    self.dt,
                    self.state0.body_q,
                    self.state1.body_q,
                    self.state0.body_qd,
                    self.state1.body_qd,
                ],
                device=self.model.device,
            )
            self.state0.clear_forces()
            self.state1.clear_forces()
            self.pipeline.collide(self.state0, self.contacts)
            self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
            self.state0, self.state1 = self.state1, self.state0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.time += self.frame_dt
        q = self.state.particle_q.numpy()
        self.history.append(
            {
                "time": self.time,
                "center": q.mean(axis=0).tolist(),
                "min_z": float(q[:, 2].min()),
                "max_z": float(q[:, 2].max()),
            }
        )

    def metrics(self):
        hold = [r for r in self.history if 5.8 <= r["time"] <= 6.9]
        initial = self.initial_q.mean(axis=0)
        return {
            "boundary_conditions": "Only rigid jaws are prescribed. No soft vertex is pinned.",
            "physical_device": self.model.device.name,
            "substeps": self.substeps,
            "vbd_iterations": 10,
            "particle_count": self.model.particle_count,
            "initial_recipe": self.initial_recipe,
            "geometry_sha256": self.added.mesh.metadata["geometry_sha256"],
            "asset_generator_revision": self.added.mesh.metadata["generator_revision"],
            "open_half_gap_m": self.open_gap,
            "closed_half_gap_m": self.closed_gap,
            "contact_ke": self.model.soft_contact_ke,
            "contact_mu": self.model.soft_contact_mu,
            "initial_center": initial.tolist(),
            "hold_center_rise_m": float(np.mean([r["center"][2] for r in hold]) - initial[2]) if hold else None,
            "hold_center_translation_x_m": float(np.mean([r["center"][0] for r in hold]) - initial[0])
            if hold
            else None,
            "hold_min_vertex_height_m": float(min(r["min_z"] for r in hold)) if hold else None,
            "final_center": self.state.particle_q.numpy().mean(axis=0).tolist(),
            "trajectory": self.history,
            "qualification": "Smoke evidence only. No numerical fidelity limits have been accepted.",
        }


class FrankaDuckScene:
    def __init__(self, *, device=None, **kwargs):
        from newton.examples.softbody.example_softbody_franka import Example  # noqa: PLC0415

        class NullViewer:
            def set_model(self, *args, **kwargs):
                pass

            def set_camera(self, *args, **kwargs):
                pass

            def apply_forces(self, *args, **kwargs):
                pass

        with wp.ScopedDevice(device or "cuda:0"):
            self.example = Example(NullViewer())
        self.model = self.example.model
        self.time = 0.0
        self.frame_dt = self.example.frame_dt
        self.duration = 16.2
        self.camera_pos = (-1.0, 0.7, 1.3)
        self.camera_target = (-0.1, -0.5, 0.50)
        self.title = "R2 | Franka and deformable duck"
        self.description = "Newton shipped Franka soft-body recipe. IK commands robot motion. Featherstone provides robot kinematic integration; VBD computes deformable duck response and rigid-soft contact. Duck self-contact is disabled. No duck vertex is pinned."
        self.asset_ids = ["G1_franka", "V5_duck"]
        self.initial = self.state.particle_q.numpy().mean(axis=0)
        self.history = []

    @property
    def state(self):
        return self.example.state_0

    @property
    def phase(self):
        for end, name in [
            (2.5, "approach"),
            (4.5, "descend"),
            (7, "close"),
            (9, "lift"),
            (11, "hold"),
            (13, "place"),
            (14, "open"),
            (17, "withdraw"),
        ]:
            if self.time < end:
                return name
        return "done"

    def step(self):
        self.example.step()
        self.time += self.frame_dt
        q = self.state.particle_q.numpy()
        self.history.append({"time": self.time, "center": q.mean(axis=0).tolist(), "min_z": float(q[:, 2].min())})

    def metrics(self):
        hold = [r for r in self.history if 9.5 <= r["time"] <= 10.9]
        return {
            "robot_motion": "Prescribed IK with Featherstone kinematic integration",
            "soft_solver": "VBD",
            "duck_self_contact": False,
            "physics_parameters": {
                "frame_dt_s": self.example.frame_dt,
                "substeps": self.example.sim_substeps,
                "substep_dt_s": self.example.sim_dt,
                "iterations": self.example.iterations,
                "gravity_m_s2": [0.0, 0.0, -9.81],
                "density_kg_m3": 100.0,
                "tet_shear_modulus_pa": 1.0e6,
                "tet_first_lame_parameter_pa": 1.0e6,
                "tet_damping_pa_s": 1.0,
                "particle_radius_m": self.example.particle_radius,
                "soft_contact_ke": self.model.soft_contact_ke,
                "soft_contact_kd": self.model.soft_contact_kd,
                "soft_contact_mu": self.model.soft_contact_mu,
                "rigid_shape_mu": 1.5,
                "soft_contact_extra_detection_gap_m": self.example.soft_body_contact_margin,
                "self_contact_enabled": False,
                "rigid_soft_full_surface_contact": False,
                "robot_solver": "Featherstone with prescribed IK joint motion",
            },
            "hold_center_rise_m": float(np.mean([r["center"][2] for r in hold]) - self.initial[2]) if hold else None,
            "hold_min_vertex_height_m": float(min(r["min_z"] for r in hold)) if hold else None,
            "table_top_m": 0.2,
            "trajectory": self.history,
            "qualification": "Smoke only; full 60 s hold not run.",
        }


def build_scene(scene_id, **kwargs):
    if scene_id == "j3_h1_jacket":
        from experiments.collision_benchmark.jacket_h1_scene import build_scene as build_h1_jacket  # noqa: PLC0415

        return build_h1_jacket(scene_id, **kwargs)
    if scene_id in ("r7_h1_gusset_bag", "r7_h1_tote", "r7_h1_pouch") or scene_id.startswith("m4_h1_"):
        from experiments.collision_benchmark.h1_bag_scenes import build_scene as build_h1_bag  # noqa: PLC0415

        return build_h1_bag(scene_id, **kwargs)
    if scene_id.startswith("r7_h1_"):
        from experiments.collision_benchmark.cloth_h1_scenes import build_scene as build_h1_cloth  # noqa: PLC0415

        return build_h1_cloth(scene_id, **kwargs)
    if scene_id.startswith("r5_"):
        from experiments.collision_benchmark.cloth_robot_scenes import build_scene as build_cloth_robot  # noqa: PLC0415

        return build_cloth_robot(scene_id, **kwargs)
    if scene_id.startswith("r6_"):
        from experiments.collision_benchmark.solid_robot_scenes import build_scene as build_solid_robot  # noqa: PLC0415

        return build_solid_robot(scene_id, **kwargs)
    if scene_id.startswith("m3_"):
        from experiments.collision_benchmark.bag_robot_scenes import build_scene as build_bag_robot  # noqa: PLC0415

        return build_bag_robot(scene_id, **kwargs)
    if scene_id.startswith("c9_"):
        from experiments.collision_benchmark.cable_robot_retrofit import (  # noqa: PLC0415
            build_scene as build_cable_robot,
        )

        return build_cable_robot(scene_id, **kwargs)
    if scene_id == "a1_allegro_cloth":
        from experiments.collision_benchmark.allegro_cloth_scene import (  # noqa: PLC0415
            build_scene as build_allegro_cloth,
        )

        return build_allegro_cloth(scene_id, **kwargs)
    if scene_id in ("m2_franka_cable", "m2_allegro_cable"):
        from experiments.collision_benchmark.robot_cable_scene import build_scene as build_robot_cable  # noqa: PLC0415

        return build_robot_cable(scene_id, **kwargs)
    if scene_id == "j2_franka_jacket":
        from experiments.collision_benchmark.jacket_scene import build_scene as build_jacket  # noqa: PLC0415

        return build_jacket(scene_id, **kwargs)
    if scene_id == "m2_franka_pouch":
        from experiments.collision_benchmark.pouch_scene import build_scene as build_pouch  # noqa: PLC0415

        return build_pouch(scene_id, **kwargs)
    if scene_id.startswith("j1_"):
        from experiments.collision_benchmark.jacket_scene import build_scene as build_jacket  # noqa: PLC0415

        return build_jacket(scene_id, **kwargs)
    if scene_id in ("r1_solid_capsule_corrected", "r1_solid_branched_corrected"):
        from experiments.collision_benchmark.solid_recipe import build_scene as build_solid  # noqa: PLC0415

        return build_solid(scene_id, **kwargs)
    if scene_id == "r2_franka_duck":
        return FrankaDuckScene(**kwargs)
    if scene_id.startswith("m1_"):
        from experiments.collision_benchmark.bag_scenes import build_scene as build_bag  # noqa: PLC0415

        return build_bag(scene_id, **kwargs)
    if scene_id.startswith("r3_"):
        from experiments.collision_benchmark.hand_scene import build_scene as build_hand  # noqa: PLC0415

        return build_hand(scene_id, **kwargs)
    return JawScene(scene_id, **kwargs)
