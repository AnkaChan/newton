# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original-size cloth acquisition with the original left/right H1 hands.

Only robot poses and source finger joints are prescribed. Cloth landmarks are
observed once after settling to plan the approach; no cloth state is changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .assets import cloth_mesh
from .cloth_robot_scenes import _box_separation, _refine, _surface_audit
from .h1_gripper import H1HandDriver, add_h1_hand
from .jacket_diagnostics import ClothStrain

__all__ = ["H1ClothScene", "build_scene", "default_grasps"]


@wp.kernel
def _tick(clock: wp.array[float], dt: float):
    clock[0] += dt


def _original_placement(name):
    mesh = cloth_mesh(name, resolution="coarse")
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    scale = 0.6 / max(hi - lo)
    q = mesh.vertices[:, [0, 2, 1]].astype(float) * scale
    q[:, 0] -= (lo[0] + hi[0]) * 0.5 * scale
    q[:, 1] -= lo[2] * scale
    q[:, 2] = 0.258 + (hi[1] - mesh.vertices[:, 1]) * scale
    if name in ("tote", "gusset_bag"):
        candidates = q[(q[:, 1] > q[:, 1].max() - 0.02) & (q[:, 2] > q[:, 2].max() - 0.01)]
        anchor = candidates[np.argmin(abs(candidates[:, 0]))].copy()
        anchor[1] -= 0.004
        old_grips = [anchor]
    else:
        front = q[q[:, 1] < q[:, 1].min() + 0.02]
        old_grips = []
        for x in np.quantile(front[:, 0], [0.25, 0.75]):
            near = front[abs(front[:, 0] - x) < 0.02]
            if not len(near):
                near = front[np.argsort(abs(front[:, 0] - x))[:2]]
            old_grips.append([x, q[:, 1].min() + 0.012, (near[:, 2].min() + near[:, 2].max()) / 2])
    vertices, triangles = _refine(q, mesh.triangles, old_grips)
    return mesh, float(scale), q, vertices, triangles


def _normal_alignment(source, target):
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot < -1.0 + 1e-8:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        axis /= np.linalg.norm(axis)
        return wp.quat(*axis, 0.0)
    value = np.r_[np.cross(a, b), 1.0 + dot]
    value /= np.linalg.norm(value)
    return wp.quat(*value)


def _triangle_box_gap(vertices, triangles, center, half):
    """Return the minimum triangle/box separating-axis gap, in metres.

    This tests all 13 separating-axis families for each original triangle.
    Positive gaps prove separation; nonpositive gaps indicate triangle/box
    intersection. The box is expanded by both contact margins by the caller.
    This is a sampled geometry test, not continuous collision detection.
    """
    tri = np.asarray(vertices, dtype=float)[triangles] - np.asarray(center)
    half = np.asarray(half, dtype=float)
    gap = np.max(np.maximum(tri.min(axis=1) - half, -half - tri.max(axis=1)), axis=1)
    edges = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], axis=1)
    axes = [np.cross(edges[:, 0], edges[:, 1])]
    axes.extend(np.cross(edges[:, j], axis) for j in range(3) for axis in np.eye(3))
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1)
        valid = norm > 1e-12
        normalized = axis / np.maximum(norm[:, None], 1e-12)
        projections = np.einsum("nvc,nc->nv", tri, normalized)
        radius = abs(normalized) @ half
        separation = np.maximum(projections.min(axis=1) - radius, -projections.max(axis=1) - radius)
        gap = np.maximum(gap, np.where(valid, separation, -np.inf))
    return float(gap.min())


def default_grasps():
    """Return the verified original-mesh index1.0, 30/7.8mm pinch recipe."""
    return [
        {
            "side": "left",
            "rotation_xyzw": [0.9930249466525313, -0.04451669895438087, 0.017717946110308844, -0.10773018716021458],
            "contact_point_local_m": [0.13635849635282835, -0.0729409497713223, 0.02874133930869728],
            "open_contact_point_local_m": [0.13635849635282835, -0.0729409497713223, 0.02874133930869728],
            "contact_normal_local": [0.09222990037490554, 0.9728249394691422, -0.21238004290350523],
            "reference_normal_world": [0, -1, 0],
            "open_controls": [1.25, 0.008686668469377504, 1.0, 0, 0, 0],
            "closed_controls": [1.25, 0.1583178316635836, 1.0, 0, 0, 0],
            "release_controls": [1.25, 0, 0, 0, 0, 0],
            "physical_closed_mesh_gap_m": 0.007799990344614003,
            "physical_open_mesh_gap_m": 0.030000014855481916,
            "nominal_total_shell_margin_preload_m": 0.0002,
            "contact_pair": "original thumb distal and index intermediate collision meshes",
            "unused_finger_pose": "Original middle, ring and pinky joints straight, with source coupling.",
            "geometry_correction": "Reduce original index curl from1.42rad to1.0rad. Refit thumb opening and contact "
            "frame from original meshes. This removes nonadjacent thumb/index solid "
            "intersection.",
        },
        {
            "side": "right",
            "rotation_xyzw": [-0.05337212020859754, 0.9982015008565991, -0.02613958116903868, 0.007867831237327764],
            "contact_point_local_m": [0.13572977662384475, 0.07306576437312409, 0.025749312953539914],
            "open_contact_point_local_m": [0.13572977662384475, 0.07306576437312409, 0.025749312953539914],
            "contact_normal_local": [0.10696359437722766, -0.992936277008495, 0.05134529460830227],
            "reference_normal_world": [0, -1, 0],
            "open_controls": [1.25, 0.009059418357859632, 1.0, 0, 0, 0],
            "closed_controls": [1.25, 0.1607516612533908, 1.0, 0, 0, 0],
            "release_controls": [1.25, 0, 0, 0, 0, 0],
            "physical_closed_mesh_gap_m": 0.007799976276789991,
            "physical_open_mesh_gap_m": 0.02999997695551713,
            "nominal_total_shell_margin_preload_m": 0.0002,
            "contact_pair": "original thumb distal and index intermediate collision meshes",
            "unused_finger_pose": "Original middle, ring and pinky joints straight, with source coupling.",
            "geometry_correction": "Reduce original index curl from1.42rad to1.0rad. Refit thumb opening and contact "
            "frame from original meshes. This removes nonadjacent thumb/index solid "
            "intersection.",
        },
    ]


class H1ClothScene:
    """Preserve the original R5 surface and observe its settled grasp patches.

    ``grasps`` contains one mapping per hand: side, rotation_xyzw,
    contact_point_local_m, open_controls and closed_controls. The contact point
    belongs to the fitted opposed fingertip surfaces in the closed pose; it
    is not an extra collider. Root trajectories are derived from that point.
    """

    def __init__(
        self,
        scene_id,
        *,
        asset_name="gusset_bag",
        grasps=None,
        rest_grip_positions=None,
        grip_radius=0.025,
        settling_time=1.5,
        initial_translation=None,
        initial_rotation=None,
        table_thickness=None,
        lift_height=None,
        placement_height_offset=None,
        pre_lift_extraction_duration=1.0,
        pre_lift_front_clearance=0.020,
        substeps=10,
        iterations=30,
        device=None,
        use_cuda_graph=True,
    ):
        if int(substeps) != substeps or not 2 <= substeps <= 10 or substeps % 2:
            raise ValueError("Use an even number of substeps in [2, 10]")
        if grasps is None:
            recipe = os.environ.get("NEWTON_H1_GRASP_RECIPE")
            grasps = json.loads(Path(recipe).read_text()) if recipe else default_grasps()
        if len(grasps) != 2 or {g["side"] for g in grasps} != {"left", "right"}:
            raise ValueError("Use exactly one original left hand and one original right hand")
        self.scene_id, self.name, self.time = scene_id, asset_name, 0.0
        self.frame_dt, self.substeps, self.iterations = 1 / 30, int(substeps), int(iterations)
        self.dt = self.frame_dt / self.substeps
        self.radius, self.table_top = 0.003, 0.25
        self.initial_translation = np.asarray(
            json.loads(os.environ.get("NEWTON_H1_INITIAL_TRANSLATION", "[0,0,0]"))
            if initial_translation is None
            else initial_translation,
            dtype=float,
        )
        if self.initial_translation.shape != (3,) or not np.isfinite(self.initial_translation).all():
            raise ValueError("initial_translation must be a finite rigid translation in metres")
        self.initial_rotation = np.asarray(
            json.loads(os.environ.get("NEWTON_H1_INITIAL_ROTATION", "[0,0,0,1]"))
            if initial_rotation is None
            else initial_rotation,
            dtype=float,
        )
        if self.initial_rotation.shape != (4,) or not np.isfinite(self.initial_rotation).all():
            raise ValueError("initial_rotation must be a finite xyzw quaternion")
        if not np.isclose(np.linalg.norm(self.initial_rotation), 1.0, atol=1e-5):
            raise ValueError("initial_rotation must be a unit xyzw quaternion")
        self.initial_rotation /= np.linalg.norm(self.initial_rotation)
        rotation_matrix = np.asarray(wp.quat_to_matrix(wp.quat(*self.initial_rotation))).reshape(3, 3)
        self.lift_height = (
            float(os.environ.get("NEWTON_H1_LIFT_HEIGHT", "0.75")) if lift_height is None else float(lift_height)
        )
        self.placement_height_offset = (
            float(os.environ.get("NEWTON_H1_PLACE_OFFSET", "0.16"))
            if placement_height_offset is None
            else float(placement_height_offset)
        )
        self.pre_lift_extraction_duration = float(pre_lift_extraction_duration)
        self.pre_lift_front_clearance = float(pre_lift_front_clearance)
        if self.pre_lift_extraction_duration <= 0 or self.pre_lift_front_clearance < 0:
            raise ValueError("Pre-lift extraction duration must be positive and clearance nonnegative")
        self.pre_lift_extraction = None
        self.table_thickness = (
            float(os.environ.get("NEWTON_H1_TABLE_THICKNESS", "0.025"))
            if table_thickness is None
            else float(table_thickness)
        )
        if not 0 < self.table_thickness <= 0.26:
            raise ValueError("table_thickness must lie in (0,0.26] metres")
        self.table_center = np.array([0.0, 0.39, self.table_top - self.table_thickness / 2])
        self.table_half = np.array([0.65, 0.31, self.table_thickness / 2])
        self.camera_pos, self.camera_target, self.camera_fov = (1.35, -1.70, 1.45), (0.10, 0.25, 0.52), 48.0
        self.title = f"R7 | H1 pair: original-size {asset_name}"
        self.description = (
            "Original R5 cloth shape, scale and material. A thin fixed tabletop keeps the original top height "
            "and footprint, with open space below for the real fingers. The free cloth settles before two original "
            "left/right H1 hands approach measured material patches, close, lift, carry, hold, place, open "
            "and withdraw. Only the robot roots and source finger joints are prescribed. No cloth pins or attachments."
        )
        self.asset_ids = [asset_name, "H1_original_left_hand", "H1_original_right_hand"]
        self.mesh, self.scale, original, vertices, triangles = _original_placement(asset_name)
        self.source_audit = _surface_audit(original, self.mesh.triangles)
        self.previous_r5_audit = _surface_audit(vertices, triangles)
        if rest_grip_positions is None:
            width = float(np.ptp(vertices[:, 0]))
            rest_grip_positions = [
                (-0.36 * width, 0.01, float(vertices[:, 2].max())),
                (0.36 * width, 0.01, float(vertices[:, 2].max())),
            ]
        self.rest_grips = np.asarray(rest_grip_positions, dtype=float)
        if self.rest_grips.shape != (2, 3):
            raise ValueError("rest_grip_positions must contain two original-surface patch centers")
        vertices, triangles = _refine(vertices, triangles, self.rest_grips)
        self.refined_audit = _surface_audit(vertices, triangles)
        if not np.allclose(self.refined_audit["bounds_m"], self.source_audit["bounds_m"], atol=1e-6):
            raise ValueError("Contact refinement changed the original asset bounds")
        if not np.isclose(self.refined_audit["area_m2"], self.source_audit["area_m2"], rtol=1e-5):
            raise ValueError("Contact refinement changed the original surface area")
        if self.refined_audit["euler_number"] != self.source_audit["euler_number"]:
            raise ValueError("Contact refinement changed the original surface topology")
        self.untranslated_geometry_hash = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
        vertices = (vertices.astype(float) @ rotation_matrix.T + self.initial_translation).astype(np.float32)
        self.rest_grips = self.rest_grips @ rotation_matrix.T + self.initial_translation
        self.placed_audit = _surface_audit(vertices, triangles)
        self.inverse_pose_audit = _surface_audit(
            (vertices.astype(float) - self.initial_translation) @ rotation_matrix, triangles
        )
        if not np.allclose(self.inverse_pose_audit["bounds_m"], self.source_audit["bounds_m"], atol=1e-6):
            raise ValueError("Inverse rigid pose does not recover original surface bounds")
        if not np.isclose(self.inverse_pose_audit["area_m2"], self.source_audit["area_m2"], rtol=1e-5):
            raise ValueError("Rigid pose changed the original surface area")
        self.geometry_hash = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
        self.rest = vertices
        self.patch_indices = []
        self.patch_anchor_indices = []
        for center in self.rest_grips:
            indices = np.flatnonzero(np.linalg.norm(vertices - center, axis=1) < grip_radius)
            if len(indices) < 3:
                raise ValueError("Each observed grasp patch must contain at least three material vertices")
            self.patch_indices.append(indices)
            self.patch_anchor_indices.append(
                int(indices[np.argmin(np.linalg.norm(vertices[indices] - center, axis=1))])
            )
        self.grasps = grasps
        self.settling_time = float(settling_time)
        limits = np.array([0.5, 0.5 / 2.4, 0.5, 0.5, 0.5, 0.5])
        self.closing_duration = max(
            1.0,
            max(
                float(np.max(1.5 * abs(np.asarray(g["closed_controls"]) - g["open_controls"]) / limits)) for g in grasps
            )
            + 0.1,
        )
        self.uncurl_duration = max(
            0.8,
            max(
                float(
                    np.max(
                        1.5
                        * abs(np.asarray(g.get("release_controls", g["open_controls"])) - g["open_controls"])
                        / limits
                    )
                )
                for g in grasps
            )
            + 0.1,
        )
        a = self.settling_time
        b = a + 1.5
        c = b + self.closing_duration
        d = c + self.pre_lift_extraction_duration
        self.times = [0.0, a, b, c, d, d + 2.5, d + 2.75, d + 4.25, d + 5.75, d + 7.75]
        self.times += [self.times[-1] + self.closing_duration, self.times[-1] + self.closing_duration + 0.8]
        self.times += [self.times[-1] + self.uncurl_duration, self.times[-1] + self.uncurl_duration + 0.5]
        self.duration = self.times[-1]
        self.hold_window = [self.times[7], self.times[8]]
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        builder.default_shape_cfg.mu, builder.default_shape_cfg.ke, builder.default_shape_cfg.kd = 1.0, 50000.0, 0.1
        builder.default_shape_cfg.margin, builder.default_shape_cfg.gap = 0.001, 0.005
        builder.default_shape_cfg.configure_sdf(force_sdf=True, max_resolution=128)
        builder.add_cloth_mesh(
            pos=(0, 0, 0),
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
            color=tuple(self.mesh.metadata["color_rgb"]),
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(self.table_center, wp.quat_identity()),
            hx=self.table_half[0],
            hy=self.table_half[1],
            hz=self.table_half[2],
            color=(0.48, 0.51, 0.55),
            label="fixed thin tabletop with original top height and footprint",
        )
        self.specs, self.parked_roots = [], []
        for i, grasp in enumerate(grasps):
            offset = np.asarray(
                wp.quat_rotate(wp.quat(*grasp["rotation_xyzw"]), wp.vec3(*grasp["contact_point_local_m"]))
            )
            parked = np.array([self.rest_grips[i, 0] - offset[0], -0.40, 0.45])
            self.parked_roots.append(parked)
            self.specs.append(
                add_h1_hand(
                    builder,
                    side=grasp["side"],
                    root_pos=parked,
                    rotation=grasp["rotation_xyzw"],
                    controls=grasp["open_controls"],
                    label=f"original_h1_{grasp['side']}",
                )
            )
        builder.add_ground_plane()
        builder.color(include_bending=True)
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 50000.0, 0.1, 1.0
        self.solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.002,
            particle_self_contact_gap=0.001,
            particle_rest_shape_contact_exclusion_radius=0.005,
            particle_topological_contact_filter_threshold=1,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=64,
            friction_epsilon=0.001,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, soft_contact_gap=0.005, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        self.state0, self.state1, self.control = self.model.state(), self.model.state(), self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.drivers = []
        for spec, grasp, parked in zip(self.specs, grasps, self.parked_roots, strict=True):
            positions = np.tile(parked, (len(self.times), 1))
            controls = [grasp["open_controls"]] * 3 + [grasp["closed_controls"]] * 7
            controls += [grasp["open_controls"]] * 2 + [grasp.get("release_controls", grasp["open_controls"])] * 2
            self.drivers.append(
                H1HandDriver(
                    self.model,
                    spec,
                    times=self.times,
                    positions=positions,
                    rotations=grasp["rotation_xyzw"],
                    controls=controls,
                )
            )
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1)
        self.strain = ClothStrain(self.model)
        self.initial = self.strain.sample(vertices)
        self.history, self.audits = [], []
        self.settled_targets = None
        self.patch_normals, self.patch_eigenvalues = [], []
        self.patch_plane_spans = []
        self.planned_path_audit = None
        self._collision_bounds = []
        for spec in self.specs:
            self._collision_bounds.append({shape: (v.min(0), v.max(0)) for shape, v in spec.collision_vertices.items()})
        self.initial_hand_audits = [
            d.geometry_audit(self.state, points=vertices, radii=self.radius, full_surface_contact=True)
            for d in self.drivers
        ]
        if any(a["minimum_shell_to_robot_distance_m"] < 0.001 for a in self.initial_hand_audits):
            raise ValueError("Initially parked H1 hand is too close to the original cloth shell")
        self.state_dir = (
            Path(os.environ["NEWTON_CLOTH_STATE_DIR"]) if os.environ.get("NEWTON_CLOTH_STATE_DIR") else None
        )
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.state_dir / "rest-mesh.npz", vertices=vertices, triangles=triangles)
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
        names = [
            "settle free cloth",
            "approach measured patches",
            "close original fingers",
            "draw acquired cloth clear of tabletop",
            "lift",
            "raised pause",
            "carry",
            "hold",
            "place",
            "open fingers",
            "extract",
            "withdraw and straighten fingers",
            "released",
        ]
        for end, name in zip(self.times[1:], names, strict=True):
            if self.time < end - 1e-6:
                return name
        return "complete"

    def _plan_from_material(self, q):
        self.settled_targets = [q[index].astype(float).copy() for index in self.patch_anchor_indices]
        for i, (driver, grasp) in enumerate(zip(self.drivers, self.grasps, strict=True)):
            points = q[self.patch_indices[i]].astype(float)
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
            normal = eigenvectors[:, 0]
            reference = np.asarray(grasp.get("reference_normal_world", (0.0, -1.0, 0.0)))
            if normal @ reference < 0:
                normal *= -1
            self.patch_normals.append(normal.tolist())
            self.patch_eigenvalues.append(eigenvalues.tolist())
            self.patch_plane_spans.append(float(np.ptp(points @ normal)))
            rotation = _normal_alignment(reference, normal) * wp.quat(*grasp["rotation_xyzw"])
            rotated = np.asarray(wp.quat_rotate(rotation, wp.vec3(*grasp["contact_point_local_m"])))
            root = self.settled_targets[i] - rotated
            open_rotated = np.asarray(
                wp.quat_rotate(
                    rotation, wp.vec3(*grasp.get("open_contact_point_local_m", grasp["contact_point_local_m"]))
                )
            )
            open_root = self.settled_targets[i] - open_rotated
            positions = np.asarray(
                [self.parked_roots[i]] * 2
                + [open_root]
                + [root] * 2
                + [np.add(root, (0, 0, self.lift_height))] * 2
                + [np.add(root, (0.24, 0.28, self.lift_height))] * 2
                + [np.add(root, (0.24, 0.28, self.placement_height_offset))]
                + [np.add(open_root, (0.24, 0.28, self.placement_height_offset))]
                + [np.add(open_root, (0.24, 0.10, self.placement_height_offset))]
                + [np.add(open_root, (0.24, 0.10, self.placement_height_offset + 0.24))] * 2,
                dtype=np.float32,
            )
            driver.positions.assign(positions)
            rotations = np.asarray([grasp["rotation_xyzw"]] * 2 + [tuple(rotation)] * (len(self.times) - 2))
            for j in range(1, len(rotations)):
                if rotations[j - 1] @ rotations[j] < 0:
                    rotations[j] *= -1
            driver.rotations.assign(rotations.astype(np.float32))
        closed_state = self.model.state()
        closed_clock = wp.array([self.times[3]], dtype=float, device=self.model.device)
        margins = self.model.shape_margin.numpy()
        closed_hand_max_y = []
        for driver in self.drivers:
            driver.update(closed_clock, closed_state)
            closed_hand_max_y.append(
                max(
                    float(v[:, 1].max()) + float(margins[shape])
                    for shape, (v, _) in driver.world_collision_meshes(closed_state).items()
                )
            )
        table_front = float(self.table_center[1] - self.table_half[1])
        retreat = max(0.0, max(closed_hand_max_y) + 0.001 + self.pre_lift_front_clearance - table_front)
        for driver in self.drivers:
            positions = driver.positions.numpy()
            positions[4:7, 1] -= retreat
            driver.positions.assign(positions)
        self.pre_lift_extraction = {
            "direction_world": [0.0, -1.0, 0.0],
            "distance_m": retreat,
            "duration_s": self.pre_lift_extraction_duration,
            "closed_hand_max_y_including_hand_margin_m": closed_hand_max_y,
            "table_front_y_m": table_front,
            "table_margin_m": 0.001,
            "requested_front_clearance_m": self.pre_lift_front_clearance,
            "method": "Both original hands draw the acquired material forward together until their complete closed collision meshes, including both margins, clear the tabletop front. Vertical lift starts after this translation. No material state is prescribed.",
        }
        self.planned_path_audit = self._audit_planned_path()
        if self.planned_path_audit["minimum_conservative_table_separation_m"] < -1e-5:
            raise ValueError(f"Planned H1/table bounds overlap: {self.planned_path_audit}")
        if self.planned_path_audit["minimum_conservative_interhand_separation_m"] < -1e-5:
            raise ValueError(f"Planned original H1 hand bounds overlap: {self.planned_path_audit}")

    def _pose_bounds(self, state):
        poses = state.body_q.numpy()
        bodies, transforms = self.model.shape_body.numpy(), self.model.shape_transform.numpy()
        margins = self.model.shape_margin.numpy()
        hands, hand_boxes, minimum_table, minimum_coarse, triangle_tests = [], [], float("inf"), float("inf"), 0
        for hand_index, shapes in enumerate(self._collision_bounds):
            world_bounds, boxes = [], []
            for shape, (lo, hi) in shapes.items():
                transform = wp.transform(*poses[bodies[shape]]) * wp.transform(*transforms[shape])
                rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
                center = np.asarray(wp.transform_point(transform, wp.vec3(*((lo + hi) / 2))))
                half = (hi - lo) / 2 + margins[shape]
                boxes.append((shape, center, rotation, half))
                gap = _box_separation(center, rotation, half, self.table_center, self.table_half + 0.001)
                minimum_coarse = min(minimum_coarse, gap)
                if gap <= 0:
                    spec = self.specs[hand_index]
                    world_vertices = spec.collision_vertices[shape] @ rotation.T + np.asarray(
                        wp.transform_get_translation(transform)
                    )
                    gap = _triangle_box_gap(
                        world_vertices,
                        spec.collision_triangles[shape],
                        self.table_center,
                        self.table_half + 0.001 + margins[shape],
                    )
                    triangle_tests += len(spec.collision_triangles[shape])
                minimum_table = min(minimum_table, gap)
                world_half = abs(rotation) @ half
                world_bounds.extend([center - world_half, center + world_half])
            world_bounds = np.asarray(world_bounds)
            hands.append((world_bounds.min(0), world_bounds.max(0)))
            hand_boxes.append(boxes)
        interhand = float(np.max(np.maximum(hands[0][0] - hands[1][1], hands[1][0] - hands[0][1])))
        self._last_interhand_coarse_gap = interhand
        self._last_interhand_pair_tests = 0
        self._last_interhand_unresolved_pair = None
        if interhand <= 0:
            # Express the first OBB in the second OBB's frame. The existing
            # OBB/AABB SAT then tests all 15 axis families for this mesh pair.
            interhand = float("inf")
            for shape_a, center_a, rotation_a, half_a in hand_boxes[0]:
                for shape_b, center_b, rotation_b, half_b in hand_boxes[1]:
                    gap = _box_separation(
                        rotation_b.T @ (center_a - center_b),
                        rotation_b.T @ rotation_a,
                        half_a,
                        np.zeros(3),
                        half_b,
                    )
                    self._last_interhand_pair_tests += 1
                    if gap < interhand:
                        interhand = gap
                        self._last_interhand_unresolved_pair = [int(shape_a), int(shape_b)] if gap <= 0 else None
        return float(minimum_table), interhand, float(minimum_coarse), triangle_tests

    def _audit_planned_path(self):
        state = self.model.state()
        clock = wp.zeros(1, dtype=float, device=self.model.device)
        worst_table, worst_hands = (float("inf"), None), (float("inf"), None)
        worst_coarse, triangle_tests = (float("inf"), None), 0
        worst_coarse_hands, hand_pair_tests, unresolved_hand_pair = (float("inf"), None), 0, None
        times = np.linspace(self.settling_time, self.duration, int((self.duration - self.settling_time) * 20) + 1)
        for time in times:
            clock.assign(np.asarray([time], dtype=np.float32))
            for driver in self.drivers:
                driver.update(clock, state)
            table, hands, coarse, tested = self._pose_bounds(state)
            triangle_tests += tested
            hand_pair_tests += self._last_interhand_pair_tests
            worst_coarse_hands = min(worst_coarse_hands, (self._last_interhand_coarse_gap, float(time)))
            if hands < worst_hands[0]:
                unresolved_hand_pair = self._last_interhand_unresolved_pair
            worst_coarse = min(worst_coarse, (coarse, float(time)))
            worst_table, worst_hands = min(worst_table, (table, float(time))), min(worst_hands, (hands, float(time)))
        for driver in self.drivers:
            driver.update(self.clock, self.state0, self.state1, dt=self.dt)
        return {
            "minimum_conservative_table_separation_m": worst_table[0],
            "minimum_table_separation_time_s": worst_table[1],
            "minimum_conservative_interhand_separation_m": worst_hands[0],
            "minimum_interhand_separation_time_s": worst_hands[1],
            "minimum_coarse_interhand_bound_separation_m": worst_coarse_hands[0],
            "minimum_coarse_interhand_bound_time_s": worst_coarse_hands[1],
            "interhand_original_mesh_obb_pair_tests": hand_pair_tests,
            "unresolved_interhand_shape_pair": unresolved_hand_pair,
            "minimum_coarse_table_bound_separation_m": worst_coarse[0],
            "minimum_coarse_table_bound_time_s": worst_coarse[1],
            "triangle_box_tests": triangle_tests,
            "samples": len(times),
            "method": "Original mesh OBB versus finite table, then exact original triangle/expanded-box SAT for every nonpositive OBB. Both1mm shape margins included. World AABB separation for complete hands; if inconclusive, every original collision-mesh OBB pair is tested by15-axis SAT with both margins. Any unresolved pair remains rejected. Sampled prescribed path, not continuous proof.",
        }

    def _simulate(self):
        self.solver.rebuild_bvh(self.state0)
        for _ in range(self.substeps):
            for driver in self.drivers:
                driver.update(self.clock, self.state0, self.state1, dt=self.dt)
            self.state0.clear_forces()
            self.state1.clear_forces()
            self.pipeline.collide(self.state0, self.contacts)
            self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
            self.state0, self.state1 = self.state1, self.state0
            wp.launch(_tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        if self.graph is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph)
        self.time += self.frame_dt
        q = self.state.particle_q.numpy()
        if self.settled_targets is None and self.time >= self.settling_time - 1e-6:
            self._plan_from_material(q)
        count = int(self.contacts.soft_contact_count.numpy()[0])
        shapes = self.contacts.soft_contact_shape.numpy()[:count]
        table_gap, hand_gap, coarse_table_gap, triangle_tests = self._pose_bounds(self.state)
        self.history.append(
            {
                "time_s": self.time,
                "phase": self.phase,
                "hand_soft_contact_records": [int(np.count_nonzero(np.isin(shapes, s.shape_ids))) for s in self.specs],
                "whole_hand_table_separation_m": table_gap,
                "whole_hand_coarse_table_bounds_separation_m": coarse_table_gap,
                "table_triangle_box_tests": triangle_tests,
                "whole_hand_interhand_bounds_separation_m": hand_gap,
                "whole_hand_coarse_interhand_bounds_separation_m": self._last_interhand_coarse_gap,
                "interhand_original_mesh_obb_pair_tests": self._last_interhand_pair_tests,
                "observed_patch_centers_m": [q[p].mean(0).tolist() for p in self.patch_indices],
                "observed_anchor_positions_m": [q[p].tolist() for p in self.patch_anchor_indices],
                "anchor_motion_since_observation_m": None
                if self.settled_targets is None
                else [
                    float(np.linalg.norm(q[p] - target))
                    for p, target in zip(self.patch_anchor_indices, self.settled_targets, strict=True)
                ],
                **self.strain.sample(q),
            }
        )
        frame = round(self.time / self.frame_dt)
        if frame % 15 == 0 or any(
            abs(self.time - t) <= self.frame_dt / 2 for t in [self.times[2], self.times[3], self.duration]
        ):
            self.audits.append(
                {
                    "time_s": self.time,
                    "hands": [
                        d.geometry_audit(self.state, points=q, radii=self.radius, full_surface_contact=True)
                        for d in self.drivers
                    ],
                }
            )
        if self.state_dir and (frame % 30 == 0 or abs(self.time - self.settling_time) < 1e-6):
            np.savez_compressed(self.state_dir / f"state-{self.time:.2f}s.npz", positions=q)

    def metrics(self):
        hold = [r for r in self.history if self.hold_window[0] - 1e-6 <= r["time_s"] <= self.hold_window[1] + 1e-6]
        return {
            "scene_id": self.scene_id,
            "original_scene": f"r5_franka_{self.name}",
            "operation_change": "Settle the original free cloth, observe accessible original material patches, then acquire with two original H1 hands.",
            "source_asset": self.mesh.metadata,
            "original_scene_scale": self.scale,
            "original_surface_audit": self.source_audit,
            "previous_r5_surface_audit": self.previous_r5_audit,
            "refined_surface_audit": self.refined_audit,
            "refinement": "Conforming midpoint subdivision of the unchanged R5 surface; maximum6mm edges within55mm of new grip patches. No smoothing, shrinkage or surface replacement.",
            "refined_geometry_sha256": self.geometry_hash,
            "untranslated_refined_geometry_sha256": self.untranslated_geometry_hash,
            "rigid_initial_translation_m": self.initial_translation.tolist(),
            "rigid_initial_rotation_xyzw": self.initial_rotation.tolist(),
            "placed_surface_audit": self.placed_audit,
            "inverse_rigid_pose_surface_audit": self.inverse_pose_audit,
            "lift_height_m": self.lift_height,
            "placement_height_offset_m": self.placement_height_offset,
            "pre_lift_extraction": self.pre_lift_extraction,
            "support_setup": {
                "original_table_thickness_m": 0.26,
                "table_thickness_m": self.table_thickness,
                "table_top_m": self.table_top,
                "table_center_m": self.table_center.tolist(),
                "table_half_extents_m": self.table_half.tolist(),
                "change": "Replace the original floor-to-top solid block by a fixed thin tabletop with unchanged top height and XY footprint. This exposes both sides of hanging material for actual finger access.",
            },
            "free_particle_count": int(sum(self.strain.mass > 0)),
            "zero_mass_particle_count": int(sum(self.strain.mass == 0)),
            "cloth_mass_kg": float(self.strain.mass.sum()),
            "robot_assets": [s.provenance for s in self.specs],
            "boundary_conditions": "Table fixed; only original robot roots and finger joints prescribed. All cloth vertices dynamic. No pins, attachments, or cloth-state writes.",
            "prepared_contact": False,
            "rest_grip_positions_m": self.rest_grips.tolist(),
            "observed_patch_particle_indices": [p.tolist() for p in self.patch_indices],
            "observed_anchor_particle_indices": self.patch_anchor_indices,
            "settled_targets_m": None if self.settled_targets is None else [p.tolist() for p in self.settled_targets],
            "settled_patch_normals": self.patch_normals,
            "settled_patch_covariance_eigenvalues_m2": self.patch_eigenvalues,
            "settled_patch_smallest_to_middle_eigenvalue_ratio": [w[0] / w[1] for w in self.patch_eigenvalues],
            "settled_patch_normal_span_m": self.patch_plane_spans,
            "planned_whole_hand_geometry_audit": self.planned_path_audit,
            "initial_hand_audits": self.initial_hand_audits,
            "commanded_hand_times_s": self.times,
            "commanded_hand_positions_m": [d.positions.numpy().tolist() for d in self.drivers],
            "grasp_recipe": self.grasps,
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
                "particle_radius_m": self.radius,
                "self_contact_enabled": True,
                "self_contact_interaction_margin_m": 0.002,
                "self_contact_extra_detection_gap_m": 0.001,
                "self_contact_rest_exclusion_m": 0.005,
                "self_contact_topological_rings": 1,
                "soft_contact_stiffness_N_m": 50000.0,
                "soft_contact_damping": 0.1,
                "soft_contact_friction": 1.0,
                "robot_mesh_sdf_max_resolution": 128,
                "shape_margin_m": 0.001,
                "shape_detection_gap_m": 0.005,
                "rigid_soft_extra_detection_gap_m": 0.005,
                "full_surface_rigid_soft_contact": True,
                "mesh_vertices": self.model.particle_count,
                "mesh_triangles": self.model.tri_count,
                "original_scene_mesh_scale": self.scale,
            },
            "hold_window_s": self.hold_window,
            "hold_min_shell_clearance_above_table_m": min(
                (r["min_z_m"] - self.radius - self.table_top for r in hold), default=None
            ),
            "hold_min_center_rise_m": min(
                (r["center_of_mass_m"][2] - self.initial["center_of_mass_m"][2] for r in hold), default=None
            ),
            "hold_max_principal_stretch": max((r["principal_stretch_max"] for r in hold), default=None),
            "hold_p95_principal_stretch_max": max((r["principal_stretch_p95"] for r in hold), default=None),
            "trajectory": self.history,
            "hand_contact_audits": self.audits,
            "qualification": "Prescribed hand smoke. Finite state does not establish acquisition, low strain, actuator capacity, or full-arm reach.",
        }


def build_scene(scene_id, **kwargs):
    if not scene_id.startswith("r7_h1_"):
        raise ValueError(scene_id)
    return H1ClothScene(scene_id, asset_name=scene_id.removeprefix("r7_h1_"), **kwargs)
