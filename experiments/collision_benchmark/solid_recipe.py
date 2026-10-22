# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometry-specific jaw placements for two contact manipulation smoke cases."""

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .assets import add_solid_asset, solid_mesh
from .manipulation import JawScene

SCENE_IDS = ["r1_solid_capsule_corrected", "r1_solid_branched_corrected"]


def recipe_parameters(name):
    """Return placement and jaw parameters in metres for an original solid."""
    mesh = solid_mesh(name, resolution="coarse")
    scale = 0.16 / np.ptp(mesh.vertices, axis=0).max()
    if name == "capsule":
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)
        rotated = np.array([wp.quat_rotate(rot, wp.vec3(*p)) for p in mesh.vertices]) * scale
        lo, hi = rotated.min(axis=0), rotated.max(axis=0)
        pos = np.array((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0.256 - lo[2]))
        center = (lo + hi) / 2 + pos
        return {
            "asset_kwargs": {"pos": tuple(pos), "rot": rot, "scale": scale},
            "base": tuple(float(x) for x in center),
            "closed_gap": float((hi[1] - lo[1]) / 2 - 0.008),
            "extents": (0.07, 0.025, 0.025),
            "note": "Capsule rests on its long side along X. Jaws grip its middle cross-section and clear the table by 16 mm.",
        }
    if name == "branched":
        return {
            "asset_kwargs": {"pos": (0.0, 0.0, 0.256), "scale": scale},
            "base": (0.057, 0.0, 0.272),
            "closed_gap": 0.018,
            "extents": (0.022, 0.025, 0.018),
            "note": "Jaws grip the +X arm between x=35 and 79 mm, outside the central crossing. The other three arms remain free.",
        }
    raise ValueError(f"No specific solid recipe for {name!r}")


class SolidRecipeScene(JawScene):
    """Reuse the common jaw trajectory with a declared asset-specific grasp."""

    def __init__(self, scene_id, *, device=None, use_cuda_graph=True, substeps=48, iterations=20, **_kwargs):
        if scene_id not in SCENE_IDS:
            raise ValueError(scene_id)
        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 2 or substeps % 2:
            raise ValueError("substeps must be an even integer >= 2 for CUDA graph state reuse")
        name = scene_id.removeprefix("r1_solid_").removesuffix("_corrected")
        recipe = recipe_parameters(name)
        self.scene_id = scene_id
        self.time = 0.0
        self.frame_dt = 1 / 30
        self.substeps = substeps
        self.iterations = iterations
        self.dt = self.frame_dt / self.substeps
        self.duration = 11.5
        self.is_solid = True
        self.initial_recipe = False
        self.title = f"R1 | {name}: geometry-specific grasp, carry, release"
        self.description = (
            recipe["note"]
            + " Prescribed rigid jaws; every deformable vertex is dynamic. Material, friction, and jaw trajectory match the initial recipe."
            + f" Refined integration uses {substeps} substeps per display frame and {iterations} VBD iterations."
        )
        self.asset_ids = [name, "G0_generated_jaws"]
        self.camera_pos = (1.15, -1.5, 1.3)
        self.camera_target = (0.12, 0.12, 0.5)
        self.axis = 1
        self.half = 0.025
        self.base = recipe["base"]
        self.closed_gap = recipe["closed_gap"]
        self.open_gap = self.closed_gap + 0.055
        self.recipe = recipe
        b = newton.ModelBuilder(gravity=(0, 0, -9.81))
        b.default_shape_cfg.mu = 1.0
        b.default_shape_cfg.ke = 50000.0
        b.default_shape_cfg.kd = 0.1
        self.added = add_solid_asset(
            b,
            name,
            resolution="coarse",
            **recipe["asset_kwargs"],
            density=180.0,
            particle_radius=0.003,
            k_mu=8000.0,
            k_lambda=8000.0,
            k_damp=0.5,
        )
        b.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0, 0, 0.12), wp.quat_identity()),
            hx=0.6,
            hy=0.5,
            hz=0.13,
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
            hx, hy, hz = recipe["extents"]
            b.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg, color=(0.85, 0.43, 0.15) if i else (0.15, 0.43, 0.75))
            jaw_ids.append(body)
        b.add_ground_plane()
        b.color()
        self.model = b.finalize(device=device)
        self.model.soft_contact_ke = 50000.0
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 1.0
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
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
        self._tets = self.model.tet_indices.numpy()
        rest = self.initial_q[self._tets]
        self._rest_dets = np.linalg.det(rest[:, 1:] - rest[:, :1])
        self._max_inverted = 0
        if use_cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def step(self):
        super().step()
        q = self.state.particle_q.numpy()[self._tets]
        ratios = np.linalg.det(q[:, 1:] - q[:, :1]) / self._rest_dets
        self._max_inverted = max(self._max_inverted, int(np.count_nonzero(ratios <= 0)))

    def metrics(self):
        result = super().metrics()
        result["vbd_iterations"] = self.iterations
        result["grasp_recipe"] = self.recipe["note"]
        result["jaw_center_m"] = list(self.base)
        result["jaw_half_extents_m"] = list(self.recipe["extents"])
        result["max_inverted_tets_over_frames"] = self._max_inverted
        result["table_top_m"] = 0.25
        if result["hold_min_vertex_height_m"] is not None:
            result["hold_min_clearance_above_table_m"] = result["hold_min_vertex_height_m"] - 0.25
        return result


def build_scene(scene_id, **kwargs):
    return SolidRecipeScene(scene_id, **kwargs)
