# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Bag Pickup (Two-Way Coupled)
#
# A parallel-jaw gripper on a Cartesian gantry grasps the open top of a
# lunch-bag-sized VBD cloth bag containing rigid bodies, then lifts and
# waves it.  The same grab -> close -> lift -> wave trajectory as the
# Franka pickup examples, but without the robot arm.
#
# The gripper is four prismatic joints:
#
#   World -[Z]-> gantry_z -[X]-> gantry_x -[Y]-> left_finger
#                                          -[Y]-> right_finger
#
# All joints are PD-position-driven.  VBD integrates everything:
# gripper dynamics, cloth, and rigid content bodies
# (integrate_with_external_rigid_solver=False).
#
# Command: python -m newton.examples vbd_bag_franka_pickup_two_way
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples

PARAMS = {
    "shape_names": [
        "mesh",
        "cone",
        "sphere",
        # "box",
        # # "capsule",
        # "cylinder",
    ],
    "shape_size": 0.03,
    "shape_margin": 0.005,
    "shape_clearance_scale": 0.9,
    "soft_contact_creation_margin": 0.012,
    "ground_size": 0.80,
    "ground_thickness": 0.01,
    "bag_size_x": 0.22,
    "bag_size_y": 0.14,
    "bag_size_z": 0.32,
    "bag_res": 18,
    "bag_floor_height": 0.004,
    "vertical_axis": 2,
    "particle_radius": 0.004,
    "fps": 60,
    "settle_frames": 420,
    "sim_substeps": 20,
    "solver_iterations": 5,
    "cloth_density": 0.08,
    "cloth_tri_ke": 5e5,
    "cloth_tri_ka": 1e5,
    "cloth_tri_kd": 1e2,
    "cloth_edge_ke": 200.0,
    "cloth_edge_kd": 2e-1,
    "shape_density": 1000.0,
    "shape_ke": 5.0e5,
    "shape_kd": 5.0e1,
    "shape_mu": 0.5,
    "ground_ke": 1.0e5,
    "ground_kd": 1.0e2,
    "ground_mu": 0.9,
    "soft_contact_ke": 5.0e5,
    "soft_contact_kd": 1.0e1,
    "soft_contact_mu": 1.0,
    "gravity": -9.8,
    "initial_paused": False,
    "seed": 42,
    "draw_wireframe": False,
    "camera_pos": (0.98, -1.38, 0.80),
    "camera_fov": 45.0,
    "camera_pitch": -16.9,
    "camera_yaw": 128.5,
    "body_drop_offset": 0.06,
    "rigid_body_particle_contact_buffer_size": 2048,
    "rigid_body_contact_buffer_size": 512,
    "particle_enable_self_contact": True,
    "particle_self_contact_radius": 0.005,
    "particle_self_contact_margin": 0.01,
    "particle_topological_contact_filter_threshold": 3,
    "rigid_contact_hard": True,
    "collision_broad_phase": "nxn",
    "ground_body": -1,
    "ground_center_xy": (0.0, 0.0),
    "ground_color": (0.45, 0.45, 0.48),
    "ground_label": "ground",
    "cloth_pos": (0.0, 0.0, 0.0),
    "cloth_scale": 1.0,
    "cloth_vel": (0.0, 0.0, 0.0),
    "top_pin_tolerance": 0.001,
    "shape_label_prefix": "bag_contents_",
    "shape_center_spacing_radius_scale": 2.0,
    "capsule_radius_scale": 0.7,
    "cylinder_half_height_scale": 0.5,
    "ground_tolerance_particle_radius_scale": 3.0,
    # ── finger pads ────────────────────────────────────────────────────
    "finger_shape_mu": 1.0,
    "finger_shape_ke": 1.0e6,
    "finger_shape_kd": 1.0e1,
    "finger_pad_density": 1000.0,
    "finger_pad_half_width_scale": 0.5,
    "finger_pad_half_thickness": 0.0075,
    "finger_pad_half_height": 0.026,
    "finger_pad_left_label": "left_finger_bag_pad",
    "finger_pad_right_label": "right_finger_bag_pad",
    "finger_color_left": (0.8, 0.3, 0.3),
    "finger_color_right": (0.3, 0.3, 0.8),
    # ── gripper gap (half-gap per finger) ──────────────────────────────
    "open_half_gap": 0.09,
    "closed_half_gap": 0.001,
    # ── trajectory ─────────────────────────────────────────────────────
    "grasp_xy": (0.0, 0.0),
    "grab_clearance": 0.09,
    "lift_height": 0.62,
    "close_duration": 0.6,
    "pinch_duration": 0.25,
    "lift_duration": 2.8,
    "wave_axis": 0,
    "wave_offset": 0.08,
    "wave_start_duration": 0.5,
    "wave_sweep_duration": 0.7,
    "wave_return_duration": 0.5,
    "hold_duration": 1.0,
    # ── PD drive gains ─────────────────────────────────────────────────
    "gantry_drive_ke": 5.0e6,
    "gantry_drive_kd": 5.0e5,
    "finger_drive_ke": 1.0e6,
    "finger_drive_kd": 1.0e4,
    # ── gantry carrier link mass ───────────────────────────────────────
    "gantry_link_mass": 0.01,
    "finger_z_offset": 0.09,
    # ── cuda graph ─────────────────────────────────────────────────────
    "enable_cuda_graph": True,
}


# ── bag / content geometry helpers (from kinematic example) ─────────────


def _generate_box_bag(half_x, half_y, height, res, z_base):
    """Generate a box-shaped bag (5 faces, open top) as a single merged mesh."""
    cell_x = 2.0 * half_x / res
    cell_y = 2.0 * half_y / res
    cell_z = height / res

    vertex_map = {}
    vertices = []
    faces = []

    def get_or_add_vertex(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([x, y, z])
        return vertex_map[key]

    def add_quad(v00, v10, v01, v11):
        faces.extend([v00, v10, v01])
        faces.extend([v10, v11, v01])

    for i in range(res):
        for j in range(res):
            x0, x1 = -half_x + i * cell_x, -half_x + (i + 1) * cell_x
            y0, y1 = -half_y + j * cell_y, -half_y + (j + 1) * cell_y
            add_quad(
                get_or_add_vertex(x0, y0, z_base),
                get_or_add_vertex(x1, y0, z_base),
                get_or_add_vertex(x0, y1, z_base),
                get_or_add_vertex(x1, y1, z_base),
            )

    sides = [
        lambda i, j: (-half_x + i * cell_x, -half_y, z_base + j * cell_z, cell_x, 0, cell_z, 0),
        lambda i, j: (-half_x + i * cell_x, half_y, z_base + j * cell_z, cell_x, 0, cell_z, 1),
        lambda i, j: (-half_x, -half_y + i * cell_y, z_base + j * cell_z, 0, cell_y, cell_z, 2),
        lambda i, j: (half_x, -half_y + i * cell_y, z_base + j * cell_z, 0, cell_y, cell_z, 3),
    ]
    for side_fn in sides:
        for i in range(res):
            for j in range(res):
                x0, y0, z0, dx, dy, dz, side = side_fn(i, j)
                if side == 0:
                    add_quad(
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0 + dx, y0, z0),
                        get_or_add_vertex(x0, y0, z0 + dz),
                        get_or_add_vertex(x0 + dx, y0, z0 + dz),
                    )
                elif side == 1:
                    add_quad(
                        get_or_add_vertex(x0 + dx, y0, z0),
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0 + dx, y0, z0 + dz),
                        get_or_add_vertex(x0, y0, z0 + dz),
                    )
                elif side == 2:
                    add_quad(
                        get_or_add_vertex(x0, y0 + dy, z0),
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0, y0 + dy, z0 + dz),
                        get_or_add_vertex(x0, y0, z0 + dz),
                    )
                elif side == 3:
                    add_quad(
                        get_or_add_vertex(x0, y0, z0),
                        get_or_add_vertex(x0, y0 + dy, z0),
                        get_or_add_vertex(x0, y0, z0 + dz),
                        get_or_add_vertex(x0, y0 + dy, z0 + dz),
                    )

    return np.array(vertices, dtype=np.float32), faces


def _load_bear_mesh(target_size):
    bear_path = os.path.join(newton.examples.get_asset_directory(), "bear.usd")
    stage = Usd.Stage.Open(bear_path)
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))

    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    points -= center
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    points *= (target_size * 2.0) / extent

    return points, indices.tolist()


def _generate_stacked_positions(count, half_x, half_y, z_bottom, z_top, min_spacing, rng):
    """Generate centered XYZ positions with at least min_spacing between centers."""
    cols = int(np.floor((half_x * 2.0) / min_spacing)) + 1
    rows = int(np.floor((half_y * 2.0) / min_spacing)) + 1
    cols = max(cols, 1)
    rows = max(rows, 1)

    xs = (np.arange(cols, dtype=np.float32) - (cols - 1) * 0.5) * min_spacing
    ys = (np.arange(rows, dtype=np.float32) - (rows - 1) * 0.5) * min_spacing
    layer_xy = [(float(x), float(y)) for y in ys for x in xs]
    per_layer = len(layer_xy)
    layers = int(np.ceil(count / per_layer))

    if z_bottom + (layers - 1) * min_spacing > z_top:
        raise ValueError(f"Bag is too short for {count} shapes with {min_spacing:.3f} m center spacing")

    positions = []
    for layer in range(layers):
        shuffled_xy = list(layer_xy)
        rng.shuffle(shuffled_xy)
        z = z_bottom + layer * min_spacing
        for x, y in shuffled_xy:
            positions.append((x, y, z))
            if len(positions) == count:
                return positions

    return positions


def build_model(builder, params, seed=42):
    rng = np.random.default_rng(seed)

    ground_cfg = newton.ModelBuilder.ShapeConfig()
    ground_cfg.ke = params["ground_ke"]
    ground_cfg.kd = params["ground_kd"]
    ground_cfg.mu = params["ground_mu"]
    builder.add_shape_box(
        params["ground_body"],
        wp.transform(
            wp.vec3(*params["ground_center_xy"], -params["ground_thickness"] / 2.0),
            wp.quat_identity(),
        ),
        hx=params["ground_size"] / 2.0,
        hy=params["ground_size"] / 2.0,
        hz=params["ground_thickness"] / 2.0,
        cfg=ground_cfg,
        color=params["ground_color"],
        label=params["ground_label"],
    )

    bag_verts, bag_faces = _generate_box_bag(
        params["bag_size_x"] / 2,
        params["bag_size_y"] / 2,
        params["bag_size_z"],
        params["bag_res"],
        params["bag_floor_height"],
    )

    pr = params["particle_radius"]
    bag_start_particle = len(builder.particle_q)

    builder.add_cloth_mesh(
        pos=wp.vec3(*params["cloth_pos"]),
        rot=wp.quat_identity(),
        scale=params["cloth_scale"],
        vel=wp.vec3(*params["cloth_vel"]),
        vertices=bag_verts.tolist(),
        indices=bag_faces,
        density=params["cloth_density"],
        tri_ke=params["cloth_tri_ke"],
        tri_ka=params["cloth_tri_ka"],
        tri_kd=params["cloth_tri_kd"],
        edge_ke=params["cloth_edge_ke"],
        edge_kd=params["cloth_edge_kd"],
        particle_radius=pr,
    )

    bag_end_particle = len(builder.particle_q)
    z_top = params["bag_floor_height"] + params["bag_size_z"]
    top_mask = np.abs(bag_verts[:, params["vertical_axis"]] - z_top) < params["top_pin_tolerance"]
    top_global_indices = np.where(top_mask)[0] + bag_start_particle

    r = params["shape_size"]
    margin = params["shape_margin"]
    interior_x = params["bag_size_x"] / 2 - r - margin * 2
    interior_y = params["bag_size_y"] / 2 - r - margin * 2
    z_bottom = params["bag_floor_height"] + params["body_drop_offset"]
    z_top_inside = params["bag_floor_height"] + params["bag_size_z"] - r - margin
    body_indices = []
    shape_indices = []

    shape_names = params["shape_names"]
    if shape_names:
        min_spacing = r * (params["shape_center_spacing_radius_scale"] + params["shape_clearance_scale"])
        positions = _generate_stacked_positions(
            len(shape_names),
            interior_x,
            interior_y,
            z_bottom,
            z_top_inside,
            min_spacing,
            rng,
        )

        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = params["shape_density"]
        cfg.ke = params["shape_ke"]
        cfg.kd = params["shape_kd"]
        cfg.mu = params["shape_mu"]
        cfg.has_particle_collision = True
        cfg.margin = margin

        bear_mesh = None
        for i, name in enumerate(shape_names):
            px, py, pz = positions[i]

            body = builder.add_body(
                xform=wp.transform(wp.vec3(px, py, pz), wp.quat_identity()),
                label=f"{params['shape_label_prefix']}{name}",
            )
            body_indices.append(body)
            shape_idx = len(builder.shape_type)

            if name == "sphere":
                builder.add_shape_sphere(body, radius=r, cfg=cfg)
            elif name == "box":
                builder.add_shape_box(body, hx=r, hy=r, hz=r, cfg=cfg)
            elif name == "capsule":
                builder.add_shape_capsule(body, radius=r * params["capsule_radius_scale"], half_height=r, cfg=cfg)
            elif name == "cylinder":
                builder.add_shape_cylinder(
                    body, radius=r, half_height=r * params["cylinder_half_height_scale"], cfg=cfg
                )
            elif name == "cone":
                builder.add_shape_cone(body, radius=r, half_height=r, cfg=cfg)
            elif name == "mesh":
                if bear_mesh is None:
                    bear_pts, bear_idx = _load_bear_mesh(r)
                    bear_mesh = newton.Mesh(bear_pts, np.array(bear_idx, dtype=np.int32))
                builder.add_shape_mesh(body, mesh=bear_mesh, cfg=cfg)
            shape_indices.append(shape_idx)

    builder.color(include_bending=True)

    return {
        "bag_particle_count": bag_end_particle - bag_start_particle,
        "top_global_indices": top_global_indices,
        "body_indices": body_indices,
        "shape_indices": shape_indices,
        "particle_radius": pr,
    }


# ── helpers ─────────────────────────────────────────────────────────────


def _vec_with_axis_offset(pos: wp.vec3, axis: int, offset: float) -> wp.vec3:
    v = [pos[0], pos[1], pos[2]]
    v[axis] += offset
    return wp.vec3(*v)


# ── example ─────────────────────────────────────────────────────────────


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0
        self._current_waypoint = 0
        self._time_in_waypoint = 0.0
        self._gripper_frac = 0.0

        seed = getattr(args, "seed", self.params["seed"])
        builder = newton.ModelBuilder(gravity=self.params["gravity"])

        self._build_gripper(builder)
        self.info = build_model(builder, self.params, seed=seed)

        self.model = builder.finalize()
        self.model.soft_contact_ke = self.params["soft_contact_ke"]
        self.model.soft_contact_kd = self.params["soft_contact_kd"]
        self.model.soft_contact_mu = self.params["soft_contact_mu"]

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.params["solver_iterations"],
            integrate_with_external_rigid_solver=False,
            rigid_body_particle_contact_buffer_size=self.params["rigid_body_particle_contact_buffer_size"],
            rigid_body_contact_buffer_size=self.params["rigid_body_contact_buffer_size"],
            particle_enable_self_contact=self.params["particle_enable_self_contact"],
            particle_self_contact_radius=self.params["particle_self_contact_radius"],
            particle_self_contact_margin=self.params["particle_self_contact_margin"],
            particle_topological_contact_filter_threshold=self.params["particle_topological_contact_filter_threshold"],
            rigid_contact_hard=self.params["rigid_contact_hard"],
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.params["collision_broad_phase"],
            soft_contact_margin=self.params["soft_contact_creation_margin"],
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)

        self._build_waypoints()
        self._set_targets(self._waypoints[0][0], 0.0)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = self.params["draw_wireframe"]
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = self.params["initial_paused"]
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(*self.params["camera_pos"]),
                self.params["camera_pitch"],
                self.params["camera_yaw"],
            )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

        self._capture_graph()

    # ── model construction ──────────────────────────────────────────────

    def _build_gripper(self, builder):
        p = self.params
        grab_z = p["bag_floor_height"] + p["bag_size_z"] + p["grab_clearance"]

        gantry_z = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, grab_z), wp.quat_identity()),
            mass=p["gantry_link_mass"],
            label="gantry_z",
        )
        gantry_x = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, grab_z), wp.quat_identity()),
            mass=p["gantry_link_mass"],
            label="gantry_x",
        )
        left_finger = builder.add_link(
            xform=wp.transform(
                wp.vec3(0.0, -p["open_half_gap"], grab_z - p["finger_z_offset"]),
                wp.quat_identity(),
            ),
            label="left_finger",
        )
        right_finger = builder.add_link(
            xform=wp.transform(
                wp.vec3(0.0, p["open_half_gap"], grab_z - p["finger_z_offset"]),
                wp.quat_identity(),
            ),
            label="right_finger",
        )

        pad_hx = p["bag_size_x"] * p["finger_pad_half_width_scale"]
        pad_hy = p["finger_pad_half_thickness"]
        pad_hz = p["finger_pad_half_height"]
        finger_cfg = newton.ModelBuilder.ShapeConfig(
            density=p["finger_pad_density"],
            ke=p["finger_shape_ke"],
            kd=p["finger_shape_kd"],
            mu=p["finger_shape_mu"],
            has_particle_collision=True,
        )
        builder.add_shape_box(
            left_finger,
            hx=pad_hx,
            hy=pad_hy,
            hz=pad_hz,
            cfg=finger_cfg,
            color=p["finger_color_left"],
            label=p["finger_pad_left_label"],
        )
        builder.add_shape_box(
            right_finger,
            hx=pad_hx,
            hy=pad_hy,
            hz=pad_hz,
            cfg=finger_cfg,
            color=p["finger_color_right"],
            label=p["finger_pad_right_label"],
        )

        j_z = builder.add_joint_prismatic(
            parent=-1,
            child=gantry_z,
            axis=wp.vec3(0.0, 0.0, 1.0),
            target_ke=p["gantry_drive_ke"],
            target_kd=p["gantry_drive_kd"],
            target_pos=grab_z,
            label="gantry_z_joint",
        )
        j_x = builder.add_joint_prismatic(
            parent=gantry_z,
            child=gantry_x,
            axis=wp.vec3(1.0, 0.0, 0.0),
            target_ke=p["gantry_drive_ke"],
            target_kd=p["gantry_drive_kd"],
            target_pos=0.0,
            label="gantry_x_joint",
        )
        j_left = builder.add_joint_prismatic(
            parent=gantry_x,
            child=left_finger,
            axis=wp.vec3(0.0, -1.0, 0.0),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, p["finger_z_offset"]), wp.quat_identity()),
            target_ke=p["finger_drive_ke"],
            target_kd=p["finger_drive_kd"],
            target_pos=p["open_half_gap"],
            label="left_finger_joint",
        )
        j_right = builder.add_joint_prismatic(
            parent=gantry_x,
            child=right_finger,
            axis=wp.vec3(0.0, 1.0, 0.0),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, p["finger_z_offset"]), wp.quat_identity()),
            target_ke=p["finger_drive_ke"],
            target_kd=p["finger_drive_kd"],
            target_pos=p["open_half_gap"],
            label="right_finger_joint",
        )

        builder.add_articulation([j_z, j_x, j_left, j_right], label="gripper")

        builder.joint_q[0] = grab_z
        builder.joint_q[1] = 0.0
        builder.joint_q[2] = p["open_half_gap"]
        builder.joint_q[3] = p["open_half_gap"]

        self._dof_z = 0
        self._dof_x = 1
        self._dof_left = 2
        self._dof_right = 3

    # ── trajectory ──────────────────────────────────────────────────────

    def _build_waypoints(self):
        p = self.params
        bag_top = p["bag_floor_height"] + p["bag_size_z"]
        grasp_x, grasp_y = p["grasp_xy"]
        grab_pos = wp.vec3(grasp_x, grasp_y, bag_top + p["grab_clearance"])
        lift_pos = wp.vec3(grasp_x, grasp_y, p["lift_height"])
        wave_left = _vec_with_axis_offset(lift_pos, p["wave_axis"], -p["wave_offset"])
        wave_right = _vec_with_axis_offset(lift_pos, p["wave_axis"], p["wave_offset"])
        self._waypoints = [
            (grab_pos, p["close_duration"], 0.0),
            (grab_pos, p["pinch_duration"], 1.0),
            (grab_pos, p["lift_duration"], 1.0),
            (lift_pos, p["wave_start_duration"], 1.0),
            (wave_left, p["wave_sweep_duration"], 1.0),
            (wave_right, p["wave_sweep_duration"], 1.0),
            (wave_left, p["wave_return_duration"], 1.0),
            (lift_pos, p["hold_duration"], 1.0),
        ]

    def _advance_waypoint(self):
        self._time_in_waypoint += self.frame_dt
        cur = self._waypoints[self._current_waypoint]
        nxt = self._waypoints[min(self._current_waypoint + 1, len(self._waypoints) - 1)]
        t = min(self._time_in_waypoint / cur[1], 1.0)

        target_pos = cur[0] * (1.0 - t) + nxt[0] * t
        self._gripper_frac = float(cur[2]) * (1.0 - t) + float(nxt[2]) * t

        if self._time_in_waypoint >= cur[1] and self._current_waypoint < len(self._waypoints) - 1:
            self._current_waypoint += 1
            self._time_in_waypoint = 0.0

        self._set_targets(target_pos, self._gripper_frac)

    def _set_targets(self, pos: wp.vec3, gripper_frac: float):
        p = self.params
        half_gap = p["open_half_gap"] * (1.0 - gripper_frac) + p["closed_half_gap"] * gripper_frac
        targets = np.zeros(self.model.joint_dof_count)
        targets[self._dof_z] = float(pos[2])
        targets[self._dof_x] = float(pos[0])
        targets[self._dof_left] = half_gap
        targets[self._dof_right] = half_gap
        self.control.joint_target_pos.assign(wp.array(targets, dtype=float, device=self.model.device))

    # ── simulation loop ─────────────────────────────────────────────────

    def _capture_graph(self):
        if self.params["enable_cuda_graph"] and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self._advance_waypoint()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Bag particle positions contain non-finite values"
        min_particle_z = float(np.min(particle_q[:, self.params["vertical_axis"]]))
        ground_tolerance = self.info["particle_radius"] * self.params["ground_tolerance_particle_radius_scale"]
        assert min_particle_z > -ground_tolerance, f"Bag penetrated below ground: z={min_particle_z:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=PARAMS["seed"])
        parser.set_defaults(num_frames=PARAMS["settle_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
