# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Bag Franka Pickup (Two-Way Coupled, OBJ Bag)
#
# A Franka FR3 arm grasps the open top of the OBJ/USDA VBD cloth bag
# from example_vbd_bag_franka_pickup_two_way_obj.py, containing rigid
# bodies, then lifts and waves it.
#
# Uses a single VBD solver for everything: robot dynamics, cloth, and
# rigid content.  The robot arm is driven by PD joint drives whose
# targets come from an IK solver each frame.  VBD integrates all
# bodies and particles together.
#
# Derived from example_vbd_bag_franka_pickup_two_way_obj.py by replacing
# the Cartesian gripper with a Franka FR3 robot.
#
# Command: python -m newton.examples vbd_bag_franka_pickup_two_way_obj_franka
#
###########################################################################

from __future__ import annotations

import copy
import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.utils

BAG_MESH_PATH = os.path.join(os.path.dirname(__file__), "sim_mesh_lift_old_default_f1320.usda")

PARAMS = {
    "shape_names": [
        "mesh",
        "cone",
        "sphere",
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
    "sim_substeps": 10,
    "solver_iterations": 10,
    "cloth_density": 0.08,
    "cloth_tri_ke": 1e5,
    "cloth_tri_ka": 2e4,
    "cloth_tri_kd": 1e2,
    "cloth_edge_ke": 200.0,
    "cloth_edge_kd": 0.01,
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
    "initial_sim_time": 0.0,
    "initial_frame": 0,
    "initial_waypoint": 0,
    "initial_waypoint_time": 0.0,
    "initial_gripper_frac": 0.0,
    "seed": 42,
    "draw_wireframe": False,
    "camera_pos": (0.98, -1.38, 0.80),
    "camera_fov": 45.0,
    "camera_pitch": -16.9,
    "camera_yaw": 128.5,
    "body_drop_offset": 0.06,
    "rigid_body_particle_contact_buffer_size": 2048,
    "rigid_body_contact_buffer_size": 512,
    "integrate_with_external_rigid_solver": False,
    "particle_enable_self_contact": False,
    "particle_self_contact_radius": 0.005,
    "particle_self_contact_margin": 0.01,
    "particle_topological_contact_filter_threshold": 3,
    "rigid_contact_hard": True,
    "rigid_joint_linear_ke": 1.0e6,
    "rigid_joint_angular_ke": 1.0e6,
    "rigid_joint_linear_kd": 1.0e3,
    "rigid_joint_angular_kd": 1.0e2,
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
    "finger_shape_mu": 1.0,
    "finger_shape_ke": 1.0e6,
    "finger_shape_kd": 1.0e1,
    "finger_pad_density": 1000.0,
    "finger_pad_half_width_scale": 0.5,
    "finger_pad_half_thickness": 0.0075,
    "finger_pad_half_height": 0.026,
    "finger_pad_local_pos": (0.0, 0.00758, 0.0575),
    "finger_pad_left_label": "left_finger_bag_pad",
    "finger_pad_right_label": "right_finger_bag_pad",
    "gripper_open_gap": 0.08,
    "gripper_closed_gap": 0.001,
    "gripper_open_frac": 0.0,
    "gripper_closed_frac": 1.0,
    "gripper_joint_indices": (7, 8),
    "franka_asset_name": "franka_emika_panda",
    "franka_urdf_path": "urdf/fr3_franka_hand.urdf",
    "franka_base_pos": (-0.50, 0.0, 0.05),
    "franka_floating": False,
    "franka_scale": 1.0,
    "franka_enable_self_collisions": False,
    "franka_parse_visuals_as_colliders": True,
    "franka_zero_mass_body_mass": 0.05,
    "franka_zero_mass_body_inertia": 1.0e-4,
    "franka_init_q": (-3.6802e-03, 2.3902e-02, 3.6804e-03, -2.3683, -1.2919e-04, 2.3922, 7.8549e-01),
    "arm_joint_count": 7,
    "left_finger_body_suffix": "fr3_leftfinger",
    "right_finger_body_suffix": "fr3_rightfinger",
    "hand_body_suffix": "fr3_hand",
    "mesh_approximation_method": "convex_hull",
    "keep_visual_shapes": True,
    "arm_drive_ke": (1.0e6, 1.0e6, 8.0e5, 8.0e5, 6.0e5, 6.0e5, 6.0e5),
    "arm_drive_kd": (1.0e5, 1.0e5, 8.0e4, 8.0e4, 6.0e4, 6.0e4, 6.0e4),
    "gripper_drive_ke": 1.0e6,
    "gripper_drive_kd": 1.0e5,
    "ee_link_offset": (0.0, 0.0, 0.0),
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
    "waypoint_interp_max": 1.0,
    "tool_rotation_axis": (1.0, 0.0, 0.0),
    "tool_rotation_angle": np.pi,
    "ik_n_problems": 1,
    "ik_optimizer": "lbfgs",
    "ik_jacobian_mode": "analytic",
    "ik_lambda_initial": 0.1,
    "ik_iterations": 24,
    "pregrasp_ik_iterations": 48,
    "enable_ik_cuda_graph": True,
    "gripper_gap_test_tolerance": 1.0e-8,
    "hand_lift_test_min_z": 0.53,
    "bag_lift_test_min_delta": 0.08,
    "finger_contact_test_min": 250,
}


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


def _load_usda_bag(mesh_path, half_x, half_y, height, z_base):
    """Load a USDA bag mesh and fit its bounding box to the bag dimensions."""
    stage = Usd.Stage.Open(mesh_path)
    if stage is None:
        raise FileNotFoundError(f"Bag mesh USDA not found: {mesh_path}")

    prim = stage.GetPrimAtPath("/root/bag/bag")
    if not prim.IsValid():
        raise ValueError(f"Bag mesh USDA has no /root/bag/bag mesh: {mesh_path}")

    mesh = UsdGeom.Mesh(prim)
    vertices = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    faces = []

    index_offset = 0
    for count in face_counts:
        face = face_indices[index_offset : index_offset + count]
        index_offset += count
        for i in range(1, count - 1):
            faces.extend([int(face[0]), int(face[i]), int(face[i + 1])])

    if vertices.size == 0:
        raise ValueError(f"Bag mesh USDA has no vertices: {mesh_path}")
    if not faces:
        raise ValueError(f"Bag mesh USDA has no faces: {mesh_path}")

    source_min = vertices.min(axis=0)
    source_extent = vertices.max(axis=0) - source_min
    if np.any(source_extent <= 0.0):
        raise ValueError(f"Bag mesh USDA has non-positive extent: {source_extent}")

    target_min = np.array([-half_x, -half_y, z_base], dtype=np.float32)
    target_extent = np.array([2.0 * half_x, 2.0 * half_y, height], dtype=np.float32)
    vertices = (vertices - source_min) / source_extent * target_extent + target_min

    return vertices.astype(np.float32), faces


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


def build_model(builder, params, seed=PARAMS["seed"]):
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

    bag_verts, bag_faces = _load_usda_bag(
        BAG_MESH_PATH,
        params["bag_size_x"] / 2,
        params["bag_size_y"] / 2,
        params["bag_size_z"],
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
            builder.add_shape_cylinder(body, radius=r, half_height=r * params["cylinder_half_height_scale"], cfg=cfg)
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


def _quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


def _vec_with_axis_offset(pos: wp.vec3, axis: int, offset: float) -> wp.vec3:
    values = [pos[0], pos[1], pos[2]]
    values[axis] += offset
    return wp.vec3(*values)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.sim_time = self.params["initial_sim_time"]
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = self.params["initial_frame"]
        self._current_waypoint = self.params["initial_waypoint"]
        self._time_in_waypoint = self.params["initial_waypoint_time"]
        self._gripper_frac = self.params["initial_gripper_frac"]

        seed = getattr(args, "seed", self.params["seed"])
        builder = newton.ModelBuilder(gravity=self.params["gravity"])

        self._add_robot(builder)
        self.info = build_model(builder, self.params, seed=seed)

        self.model = builder.finalize()
        self.model.soft_contact_ke = self.params["soft_contact_ke"]
        self.model.soft_contact_kd = self.params["soft_contact_kd"]
        self.model.soft_contact_mu = self.params["soft_contact_mu"]

        self._regularize_robot_zero_mass_bodies()
        self._finger_contact_shape_indices = {
            shape_index
            for shape_index, body_index in enumerate(self.model.shape_body.numpy())
            if int(body_index) in self._finger_contact_body_indices
        }
        self._configure_robot_contacts()
        self._configure_joint_drives()

        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.params["collision_broad_phase"],
            soft_contact_margin=self.params["soft_contact_creation_margin"],
        )
        self.contacts = self.pipeline.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.model.body_q, self.state_0.body_q)

        self._state_single = self._model_single.state()
        newton.eval_fk(self._model_single, self._model_single.joint_q, self._model_single.joint_qd, self._state_single)
        self._setup_ik()
        self._initialize_robot_pregrasp()
        self._initial_bag_top_z = float(np.max(self.state_0.particle_q.numpy()[:, self.params["vertical_axis"]]))

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.params["solver_iterations"],
            integrate_with_external_rigid_solver=self.params["integrate_with_external_rigid_solver"],
            rigid_body_particle_contact_buffer_size=self.params["rigid_body_particle_contact_buffer_size"],
            rigid_body_contact_buffer_size=self.params["rigid_body_contact_buffer_size"],
            particle_enable_self_contact=self.params["particle_enable_self_contact"],
            particle_self_contact_radius=self.params["particle_self_contact_radius"],
            particle_self_contact_margin=self.params["particle_self_contact_margin"],
            particle_topological_contact_filter_threshold=self.params["particle_topological_contact_filter_threshold"],
            rigid_contact_hard=self.params["rigid_contact_hard"],
            rigid_joint_linear_ke=self.params["rigid_joint_linear_ke"],
            rigid_joint_angular_ke=self.params["rigid_joint_angular_ke"],
            rigid_joint_linear_kd=self.params["rigid_joint_linear_kd"],
            rigid_joint_angular_kd=self.params["rigid_joint_angular_kd"],
        )

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

    def _regularize_robot_zero_mass_bodies(self):
        body_mass = self.model.body_mass.numpy().copy()
        zero_mass_indices = [
            i
            for i, label in enumerate(self.model.body_label[: self._robot_body_count])
            if body_mass[i] == 0.0 and not label.endswith("/base")
        ]
        if not zero_mass_indices:
            return

        mass = self.params["franka_zero_mass_body_mass"]
        inertia = self.params["franka_zero_mass_body_inertia"]
        body_inv_mass = self.model.body_inv_mass.numpy().copy()
        body_inertia = self.model.body_inertia.numpy().copy()
        body_inv_inertia = self.model.body_inv_inertia.numpy().copy()
        inertia_matrix = np.eye(3, dtype=np.float32) * inertia
        inv_inertia_matrix = np.eye(3, dtype=np.float32) / inertia

        for body_index in zero_mass_indices:
            body_mass[body_index] = mass
            body_inv_mass[body_index] = 1.0 / mass
            body_inertia[body_index] = inertia_matrix
            body_inv_inertia[body_index] = inv_inertia_matrix

        self.model.body_mass = wp.array(body_mass, dtype=float, device=self.model.device)
        self.model.body_inv_mass = wp.array(body_inv_mass, dtype=float, device=self.model.device)
        self.model.body_inertia = wp.array(body_inertia, dtype=wp.mat33, device=self.model.device)
        self.model.body_inv_inertia = wp.array(body_inv_inertia, dtype=wp.mat33, device=self.model.device)

    def _add_robot(self, builder):
        asset_path = newton.utils.download_asset(self.params["franka_asset_name"])
        builder.add_urdf(
            str(asset_path / self.params["franka_urdf_path"]),
            xform=wp.transform(wp.vec3(*self.params["franka_base_pos"]), wp.quat_identity()),
            floating=self.params["franka_floating"],
            scale=self.params["franka_scale"],
            enable_self_collisions=self.params["franka_enable_self_collisions"],
            parse_visuals_as_colliders=self.params["franka_parse_visuals_as_colliders"],
        )

        arm_joint_count = self.params["arm_joint_count"]
        builder.joint_q[:arm_joint_count] = self.params["franka_init_q"]
        open_gripper_value = self._gripper_joint_value(self.params["gripper_open_frac"])
        for joint_index in self.params["gripper_joint_indices"]:
            builder.joint_q[joint_index] = open_gripper_value

        self._left_finger_body = next(
            i for i, label in enumerate(builder.body_label) if label.endswith(self.params["left_finger_body_suffix"])
        )
        self._right_finger_body = next(
            i for i, label in enumerate(builder.body_label) if label.endswith(self.params["right_finger_body_suffix"])
        )
        self._hand_body = next(
            i for i, label in enumerate(builder.body_label) if label.endswith(self.params["hand_body_suffix"])
        )
        self._finger_contact_body_indices = {self._left_finger_body, self._right_finger_body}

        pad_cfg = newton.ModelBuilder.ShapeConfig(
            density=self.params["finger_pad_density"],
            mu=self.params["finger_shape_mu"],
            ke=self.params["finger_shape_ke"],
            kd=self.params["finger_shape_kd"],
        )
        pad_xform = wp.transform(
            wp.vec3(*self.params["finger_pad_local_pos"]),
            wp.quat_identity(),
        )
        pad_hx = self.params["bag_size_x"] * self.params["finger_pad_half_width_scale"]
        pad_hy = self.params["finger_pad_half_thickness"]
        pad_hz = self.params["finger_pad_half_height"]
        builder.add_shape_box(
            body=self._left_finger_body,
            xform=pad_xform,
            hx=pad_hx,
            hy=pad_hy,
            hz=pad_hz,
            cfg=pad_cfg,
            label=self.params["finger_pad_left_label"],
        )
        builder.add_shape_box(
            body=self._right_finger_body,
            xform=pad_xform,
            hx=pad_hx,
            hy=pad_hy,
            hz=pad_hz,
            cfg=pad_cfg,
            label=self.params["finger_pad_right_label"],
        )

        finger_body_set = {self._left_finger_body, self._right_finger_body, self._hand_body}
        non_finger_shape_indices = [s for s, body in enumerate(builder.shape_body) if body not in finger_body_set]
        builder.approximate_meshes(
            method=self.params["mesh_approximation_method"],
            shape_indices=non_finger_shape_indices,
            keep_visual_shapes=self.params["keep_visual_shapes"],
        )

        self._model_single = copy.deepcopy(builder).finalize()
        self._robot_body_count = builder.body_count
        self._ee_body_index = self._hand_body

    def _configure_robot_contacts(self):
        shape_body = self.model.shape_body.numpy()
        shape_flags = self.model.shape_flags.numpy().copy()
        shape_mu = self.model.shape_material_mu.numpy().copy()
        shape_ke = self.model.shape_material_ke.numpy().copy()
        shape_kd = self.model.shape_material_kd.numpy().copy()

        for shape_index, shape_body_index in enumerate(shape_body):
            body_index = int(shape_body_index)
            if 0 <= body_index < self._robot_body_count and body_index not in self._finger_contact_body_indices:
                shape_flags[shape_index] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            elif body_index in self._finger_contact_body_indices:
                shape_mu[shape_index] = self.params["finger_shape_mu"]
                shape_ke[shape_index] = self.params["finger_shape_ke"]
                shape_kd[shape_index] = self.params["finger_shape_kd"]

        self.model.shape_flags = wp.array(
            shape_flags,
            dtype=self.model.shape_flags.dtype,
            device=self.model.device,
        )
        self.model.shape_material_mu = wp.array(shape_mu, dtype=float, device=self.model.device)
        self.model.shape_material_ke = wp.array(shape_ke, dtype=float, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=float, device=self.model.device)

    def _configure_joint_drives(self):
        ke = self.model.joint_target_ke.numpy().copy()
        kd = self.model.joint_target_kd.numpy().copy()
        arm_n = self.params["arm_joint_count"]
        ke[:arm_n] = self.params["arm_drive_ke"][:arm_n]
        kd[:arm_n] = self.params["arm_drive_kd"][:arm_n]
        for ji in self.params["gripper_joint_indices"]:
            ke[ji] = self.params["gripper_drive_ke"]
            kd[ji] = self.params["gripper_drive_kd"]
        self.model.joint_target_ke = wp.array(ke, dtype=float, device=self.model.device)
        self.model.joint_target_kd = wp.array(kd, dtype=float, device=self.model.device)

    def _tool_rotation(self):
        return wp.quat_from_axis_angle(
            wp.vec3(*self.params["tool_rotation_axis"]),
            self.params["tool_rotation_angle"],
        )

    def _setup_ik(self):
        ee_tf = wp.transform(*self._state_single.body_q.numpy()[self._ee_body_index])

        self._pos_obj = ik.IKObjectivePosition(
            link_index=self._ee_body_index,
            link_offset=wp.vec3(*self.params["ee_link_offset"]),
            target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=self._ee_body_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
        )
        self._joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self._model_single.joint_limit_lower,
            joint_limit_upper=self._model_single.joint_limit_upper,
        )
        self._joint_q_ik = wp.array(
            self._model_single.joint_q,
            shape=(self.params["ik_n_problems"], self._model_single.joint_coord_count),
        )
        self._ik_solver = ik.IKSolver(
            model=self._model_single,
            n_problems=self.params["ik_n_problems"],
            objectives=[self._pos_obj, self._rot_obj, self._joint_limits_obj],
            optimizer=self.params["ik_optimizer"],
            lambda_initial=self.params["ik_lambda_initial"],
            jacobian_mode=self.params["ik_jacobian_mode"],
        )

        bag_top = self.params["bag_floor_height"] + self.params["bag_size_z"]
        grasp_x, grasp_y = self.params["grasp_xy"]
        grab_pos = wp.vec3(grasp_x, grasp_y, bag_top + self.params["grab_clearance"])
        lift_pos = wp.vec3(grasp_x, grasp_y, self.params["lift_height"])
        wave_axis = self.params["wave_axis"]
        wave_left_pos = _vec_with_axis_offset(lift_pos, wave_axis, -self.params["wave_offset"])
        wave_right_pos = _vec_with_axis_offset(lift_pos, wave_axis, self.params["wave_offset"])
        self._waypoints = [
            (grab_pos, self.params["close_duration"], self.params["gripper_open_frac"]),
            (grab_pos, self.params["pinch_duration"], self.params["gripper_closed_frac"]),
            (grab_pos, self.params["lift_duration"], self.params["gripper_closed_frac"]),
            (lift_pos, self.params["wave_start_duration"], self.params["gripper_closed_frac"]),
            (wave_left_pos, self.params["wave_sweep_duration"], self.params["gripper_closed_frac"]),
            (wave_right_pos, self.params["wave_sweep_duration"], self.params["gripper_closed_frac"]),
            (wave_left_pos, self.params["wave_return_duration"], self.params["gripper_closed_frac"]),
            (lift_pos, self.params["hold_duration"], self.params["gripper_closed_frac"]),
        ]

        if self.params["enable_ik_cuda_graph"] and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._ik_solver.step(
                    self._joint_q_ik,
                    self._joint_q_ik,
                    iterations=self.params["ik_iterations"],
                )
            self._graph_ik = capture.graph
        else:
            self._graph_ik = None

    def _gripper_joint_value(self, gripper_frac):
        open_value = self.params["gripper_open_gap"]
        closed_value = self.params["gripper_closed_gap"]
        open_frac = self.params["gripper_open_frac"]
        closed_frac = self.params["gripper_closed_frac"]
        frac_range = closed_frac - open_frac
        if frac_range == 0.0:
            return closed_value

        alpha = (gripper_frac - open_frac) / frac_range
        return open_value * (1.0 - alpha) + closed_value * alpha

    def _initialize_robot_pregrasp(self):
        start_pos = self._waypoints[0][0]
        start_frac = float(self._waypoints[0][2])
        start_rot = self._tool_rotation()

        self._pos_obj.set_target_positions(wp.array([start_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_quat_to_vec4(start_rot)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=self.params["pregrasp_ik_iterations"])

        joint_q = self.state_0.joint_q.numpy().copy()
        ik_solution = self._joint_q_ik.numpy()[0]
        arm_joint_count = self.params["arm_joint_count"]
        joint_q[:arm_joint_count] = ik_solution[:arm_joint_count]
        gripper_value = self._gripper_joint_value(start_frac)
        for joint_index in self.params["gripper_joint_indices"]:
            joint_q[joint_index] = gripper_value
        joint_q_wp = wp.array(joint_q, dtype=float, device=self.model.device)

        self._gripper_frac = start_frac
        self.model.joint_q.assign(joint_q_wp)
        self.state_0.joint_q.assign(joint_q_wp)
        self.state_1.joint_q.assign(joint_q_wp)
        self.model.joint_qd.zero_()
        self.state_0.joint_qd.zero_()
        self.state_1.joint_qd.zero_()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.model.body_q, self.state_0.body_q)

        self._update_drive_targets()

    def _set_joint_targets(self):
        self._time_in_waypoint += self.frame_dt
        current = self._waypoints[self._current_waypoint]
        next_waypoint = self._waypoints[min(self._current_waypoint + 1, len(self._waypoints) - 1)]
        t = min(self._time_in_waypoint / current[1], self.params["waypoint_interp_max"])

        target_pos = current[0] * (1.0 - t) + next_waypoint[0] * t
        self._gripper_frac = float(current[2]) * (1.0 - t) + float(next_waypoint[2]) * t

        target_rot = self._tool_rotation()
        self._pos_obj.set_target_positions(wp.array([target_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_quat_to_vec4(target_rot)], dtype=wp.vec4))

        if self._graph_ik is not None:
            wp.capture_launch(self._graph_ik)
        else:
            self._ik_solver.step(
                self._joint_q_ik,
                self._joint_q_ik,
                iterations=self.params["ik_iterations"],
            )

        if self._time_in_waypoint >= current[1] and self._current_waypoint < len(self._waypoints) - 1:
            self._current_waypoint += 1
            self._time_in_waypoint = 0.0

    def _update_drive_targets(self):
        ik_solution = self._joint_q_ik.numpy()[0]
        arm_n = self.params["arm_joint_count"]
        target_pos = np.zeros(self.model.joint_dof_count)
        target_pos[:arm_n] = ik_solution[:arm_n]
        gripper_value = self._gripper_joint_value(self._gripper_frac)
        for ji in self.params["gripper_joint_indices"]:
            target_pos[ji] = gripper_value
        self.control.joint_target_pos.assign(wp.array(target_pos, dtype=float, device=self.model.device))

    def simulate(self):
        self._update_drive_targets()

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self._set_joint_targets()
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
        ground_tolerance = self.params["particle_radius"] * self.params["ground_tolerance_particle_radius_scale"]
        assert min_particle_z > -ground_tolerance, f"Bag penetrated below ground: z={min_particle_z:.4f}"
        open_joint_value = self._gripper_joint_value(self.params["gripper_open_frac"])
        closed_joint_value = self._gripper_joint_value(self.params["gripper_closed_frac"])
        assert abs(open_joint_value - self.params["gripper_open_gap"]) < self.params["gripper_gap_test_tolerance"]
        assert abs(closed_joint_value - self.params["gripper_closed_gap"]) < self.params["gripper_gap_test_tolerance"]
        hand_z = float(self.state_0.body_q.numpy()[self._hand_body][self.params["vertical_axis"]])
        assert hand_z > self.params["hand_lift_test_min_z"], f"Franka hand did not lift: z={hand_z:.4f}"
        bag_top_z = float(np.max(particle_q[:, self.params["vertical_axis"]]))
        bag_lift = bag_top_z - self._initial_bag_top_z
        assert bag_lift > self.params["bag_lift_test_min_delta"], f"Bag did not lift: dz={bag_lift:.4f}"
        self.pipeline.collide(self.state_0, self.contacts)
        contact_count = int(self.contacts.soft_contact_count.numpy()[0])
        contact_shapes = self.contacts.soft_contact_shape.numpy()[:contact_count]
        finger_contact_count = sum(
            1 for shape_index in contact_shapes if int(shape_index) in self._finger_contact_shape_indices
        )
        assert finger_contact_count >= self.params["finger_contact_test_min"], (
            f"Gripper lost pinch contacts: count={finger_contact_count}"
        )

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
