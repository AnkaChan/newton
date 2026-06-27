# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Apple Bag Franka
#
# A Franka FR3 arm slides one open gripper finger through one side handle of
# the supermarket apple bag from example_vbd_apple_bag.py. The released bag
# starts on a table, then hangs from the finger while the arm lifts and wiggles
# the handle in x, y, and z. The apples remain rigid bodies coupled to the VBD
# cloth bag. The Franka uses the original URDF gripper geometry rather than the
# enlarged helper pads used by the full-mouth pickup example.
#
# Commands:
#   # headless validation (no window):
#   python newton/examples/vbd/example_vbd_apple_bag_franka.py --viewer null --test
#   # interactive (OpenGL window):
#   python newton/examples/vbd/example_vbd_apple_bag_franka.py
#
###########################################################################

from __future__ import annotations

import copy
import math
import os

import numpy as np
import warp as wp
from tqdm import tqdm

import newton
import newton.examples
import newton.ik as ik
import newton.utils

# This example was adapted from example_vbd_apple_bag.py.  The bag asset, the
# base simulation parameters, and the cloth-bag + apples builder are inlined
# below (see ``_BAG_PARAMS`` and ``_build_apple_bag``) so this file is fully
# self-contained and does not import that module.
BAG_OBJ = os.path.join(os.path.dirname(__file__), "asset", "supermarket_bag.obj")

_BAG_PARAMS = {
    # --- apples (rigid spheres) ---
    "num_apples": 5,
    "apple_radius": 0.036,
    "apple_margin": 0.005,
    "apple_density": 1000.0,  # ~0.2 kg per apple at r=0.036
    "apple_ke": 5.0e5,
    "apple_kd": 5.0e1,
    "apple_mu": 0.5,
    "apple_drop_offset": 0.045,  # start this far above the rest layer so they drop in
    # --- cloth (plastic bag) ---
    "particle_radius": 0.004,
    "cloth_density": 0.08,
    "cloth_tri_ke": 2.0e5,
    "cloth_tri_ka": 2.0e5,
    "cloth_tri_kd": 1.0e1,
    "cloth_edge_ke": 0.001,  # low bending -> floppy plastic that wrinkles (high tri_ke keeps it from stretching)
    "cloth_edge_kd": 0.001,
    # --- contacts ---
    "soft_contact_ke": 5.0e5,
    "soft_contact_kd": 1.0e0,
    "soft_contact_mu": 0.2,
    "soft_contact_creation_margin": 0.012,
    "rigid_body_particle_contact_buffer_size": 4096,
    "rigid_body_contact_buffer_size": 1024,
    "rigid_contact_hard": True,
    # --- solver / time ---
    "fps": 60,
    "sim_substeps": 10,
    "solver_iterations": 12,
    "gravity": -9.8,
    "vertical_axis": 2,
    # --- pin + wiggle ---
    "pin_band": 0.03,  # pin bag vertices within this many m of the topmost vertex
    "settle_frames": 150,  # let the apples drop and settle before wiggling
    "wiggle_amplitude": 0.085,  # left<->right travel of the pinned handles [m]
    "wiggle_freq": 0.55,  # wiggle frequency [Hz]
    "wiggle_y_amplitude": 0.055,  # front<->back travel of the pinned handles [m]
    "wiggle_y_freq": 0.37,  # y-axis wiggle frequency [Hz]
    "wiggle_y_phase": 0.5 * math.pi,  # phase offset from the x wiggle [rad]
    "wiggle_bob": 0.035,  # vertical bob of the pinned handles [m] (adds bounce/jostle)
    "wiggle_bob_freq": 1.1,  # bob frequency [Hz] (~2x swing -> lively shake)
    "wiggle_ramp": 0.6,  # ease the wiggle in over this many seconds
    "wiggle_axis": 0,  # 0 = x (left<->right)
    # --- view --- 3/4 elevated front view (wireframe shows the apples sloshing inside)
    "camera_pos": (0.22, -1.0, 0.48),
    "camera_target": (0.0, 0.0, 0.14),
    "camera_fov": 45.0,
    "draw_wireframe": True,
    "initial_paused": False,
    "seed": 42,
}

PARAMS = {
    **_BAG_PARAMS,
    # --- scene contents ---
    "include_payload": True,  # include cloth bag + apples; False = robot-only (debug arm motion)
    # --- handle grip ---
    "handle_side": "left",  # hang one side handle of the bag
    "hanger_finger": "right",
    "add_finger_pads": False,  # use the original smaller FR3 gripper
    "franka_grip_offset": (-0.086, 0.0, 0.0),  # hand target offset before choosing the open hanger finger [m]
    "hanger_finger_drop": 0.025,
    "close_gripper": True,  # ease the gripper shut during the lift to secure the handle
    "gripper_close_frames": 45,  # frames to fully close, starting at gripper_close_start_frame (default: lift_start_frame)
    "lift_start_frame": 75,
    "lift_frames": 120,
    "lift_height_delta": 0.20,
    "wiggle_amplitude": 0.045,
    "wiggle_y_amplitude": 0.018,
    "wiggle_bob": 0.020,
    # --- table ---
    "table_size": 0.80,
    "table_thickness": 0.04,
    "table_top_z": 0.0,
    "table_body": -1,
    "table_center_xy": (0.0, 0.0),
    "table_ke": 1.0e5,
    "table_kd": 1.0e2,
    "table_mu": 0.9,
    "table_color": (0.45, 0.45, 0.48),
    "table_label": "table",
    "table_tolerance_particle_radius_scale": 3.0,
    # --- Franka ---
    "franka_asset_name": "franka_emika_panda",
    "franka_urdf_path": "urdf/fr3_franka_hand.urdf",
    "franka_base_pos": (-0.58, 0.0, -0.03),
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
    "gripper_drive_kd": 1.0e4,
    "gripper_open_gap": 0.08,
    "gripper_closed_gap": 0.002,
    "gripper_open_frac": 0.0,
    "gripper_closed_frac": 1.0,
    "gripper_joint_indices": (7, 8),
    "finger_shape_mu": 1.0,
    "finger_shape_ke": 1.0e6,
    "finger_shape_kd": 1.0e1,
    "finger_shape_margin": 0.003,  # outward contact offset on the finger shapes [m]
    "ee_link_offset": (0.0, 0.0, 0.0),
    "tool_rotation_quat": (0.70710678, 0.0, 0.70710678, 0.0),
    "tool_rotation_axis": (1.0, 0.0, 0.0),
    "tool_rotation_angle": np.pi,
    "ik_n_problems": 1,
    "ik_optimizer": "lbfgs",
    "ik_jacobian_mode": "analytic",
    "ik_lambda_initial": 0.1,
    "ik_iterations": 24,
    "pregrasp_ik_iterations": 48,
    "enable_ik_cuda_graph": True,
    "enable_physics_cuda_graph": True,  # replay the VBD + collision substep loop from a CUDA graph
    "waypoint_interp_max": 1.0,
    "integrate_with_external_rigid_solver": False,
    "particle_enable_self_contact": True,
    "particle_self_contact_radius": 0.003,
    "particle_self_contact_margin": 0.005,
    "particle_topological_contact_filter_threshold": 3,
    "rigid_joint_linear_ke": 1.0e6,
    "rigid_joint_angular_ke": 1.0e6,
    "rigid_joint_linear_kd": 1.0e3,
    "rigid_joint_angular_kd": 1.0e3,
    "collision_broad_phase": "nxn",
    "gripper_gap_test_tolerance": 1.0e-8,
    "gripper_open_test_tolerance": 1.0e-5,
    "handle_lift_test_min_delta": 0.06,
    "finger_hang_contact_test_min": 1,
    # --- view ---
    "initial_paused": True,
    "camera_pos": (0.54, -1.12, 0.62),
    "camera_target": (-0.02, 0.0, 0.24),
    "camera_fov": 45.0,
}


def _pitch_yaw(pos, target):
    """Pitch/yaw (deg) to look from pos toward target, Z-up convention (see camera.py)."""
    d = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-9
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(d[2])))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return pitch, yaw


def _load_obj(path):
    """Load a triangle-mesh OBJ as (vertices [V,3] float32, faces flat int list)."""
    vertices = []
    faces = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                # OBJ is 1-indexed; fan-triangulate any polygon into triangles.
                for k in range(1, len(idx) - 1):
                    faces.extend([idx[0] - 1, idx[k] - 1, idx[k + 1] - 1])
    if not vertices or not faces:
        raise ValueError(f"OBJ has no geometry: {path}")
    return np.array(vertices, dtype=np.float32), faces


def _apple_layout(num, r, margin, half_x, half_y, z_floor, drop_offset, rng):
    """Lay apples out in x-rows stacked in layers, centered, with a small drop."""
    half_x_in = max(r, half_x - r - margin)
    spacing = 2.0 * r + 0.022
    per_layer = max(1, int((2.0 * half_x_in) / spacing) + 1)
    layer_gap = 2.0 * r + 0.008
    base_z = z_floor + r + 0.012

    positions = []
    for k in range(num):
        layer = k // per_layer
        col = k % per_layer
        n_here = min(per_layer, num - layer * per_layer)
        xs = (np.arange(n_here) - (n_here - 1) * 0.5) * spacing
        stagger = 0.5 * spacing if (layer % 2 == 1) else 0.0
        x = float(xs[col] + stagger)
        x = float(np.clip(x, -half_x_in, half_x_in))
        y = float(rng.uniform(-0.4, 0.4) * max(0.0, half_y - r - margin))
        z = base_z + layer * layer_gap + drop_offset
        positions.append((x, y, z))
    return positions


def _build_apple_bag(builder, params, seed):
    """Add the welded cloth bag and the rigid apples (inlined from example_vbd_apple_bag.py)."""
    rng = np.random.default_rng(seed)

    bag_verts, bag_faces = _load_obj(BAG_OBJ)
    va = params["vertical_axis"]

    pr = params["particle_radius"]
    bag_start_particle = len(builder.particle_q)

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
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

    # Pin the handle tops: bag vertices within pin_band of the topmost vertex.
    z_top = float(bag_verts[:, va].max())
    z_floor = float(bag_verts[:, va].min())
    pin_mask = bag_verts[:, va] >= z_top - params["pin_band"]
    top_global_indices = np.where(pin_mask)[0] + bag_start_particle

    half_x = 0.5 * float(bag_verts[:, 0].max() - bag_verts[:, 0].min())
    half_y = 0.5 * float(bag_verts[:, 1].max() - bag_verts[:, 1].min())

    # Rigid apples
    r = params["apple_radius"]
    positions = _apple_layout(
        params["num_apples"], r, params["apple_margin"], half_x, half_y, z_floor, params["apple_drop_offset"], rng
    )

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["apple_density"]
    cfg.ke = params["apple_ke"]
    cfg.kd = params["apple_kd"]
    cfg.mu = params["apple_mu"]
    cfg.has_particle_collision = True
    cfg.margin = params["apple_margin"]

    body_indices = []
    shape_indices = []
    for i, (px, py, pz) in enumerate(positions):
        body = builder.add_body(xform=wp.transform(wp.vec3(px, py, pz), wp.quat_identity()), label=f"apple_{i}")
        body_indices.append(body)
        shape_indices.append(len(builder.shape_type))
        builder.add_shape_sphere(body, radius=r, cfg=cfg)

    builder.color(include_bending=True)

    return {
        "bag_particle_count": bag_end_particle - bag_start_particle,
        "top_global_indices": top_global_indices,
        "body_indices": body_indices,
        "shape_indices": shape_indices,
        "particle_radius": pr,
        "z_floor": z_floor,
        "z_top": z_top,
        "half_width": half_x,
        "half_depth": half_y,
    }


def _quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


def _tool_rotation_from_params(params):
    quat = params.get("tool_rotation_quat")
    if quat is not None:
        return wp.normalize(wp.quat(*quat))

    return wp.quat_from_axis_angle(
        wp.vec3(*params["tool_rotation_axis"]),
        params["tool_rotation_angle"],
    )


def _select_handle_top_indices(bag_verts, bag_start_particle, params):
    """Return global particle indices for one side handle top."""
    top_indices = _select_all_handle_top_indices(bag_verts, bag_start_particle, params)
    local_top_indices = top_indices - bag_start_particle

    side = params["handle_side"]
    if side == "left":
        side_mask = bag_verts[local_top_indices, 0] < 0.0
    elif side == "right":
        side_mask = bag_verts[local_top_indices, 0] > 0.0
    else:
        raise ValueError(f"Unsupported handle_side: {side}")

    return top_indices[side_mask]


def _select_all_handle_top_indices(bag_verts, bag_start_particle, params):
    """Return global particle indices for all handle-top vertices."""
    va = params["vertical_axis"]
    z_top = float(bag_verts[:, va].max())
    top_mask = bag_verts[:, va] >= z_top - params["pin_band"]
    return np.where(top_mask)[0].astype(np.int32) + bag_start_particle


def _smoothstep(alpha):
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _lift_offset_array(params, frame):
    lift_start = params["lift_start_frame"]
    lift_frames = params["lift_frames"]
    if frame <= lift_start:
        return np.zeros(3, dtype=np.float32)

    lift_alpha = (frame - lift_start) / lift_frames
    offset = np.zeros(3, dtype=np.float32)
    offset[params["vertical_axis"]] = params["lift_height_delta"] * _smoothstep(lift_alpha)
    return offset


def _wiggle_offset_array(params, frame, frame_dt):
    wiggle_start = params["lift_start_frame"] + params["lift_frames"]
    if frame <= wiggle_start:
        return np.zeros(3, dtype=np.float32)

    t_w = (frame - wiggle_start) * frame_dt
    ramp = min(1.0, t_w / params["wiggle_ramp"])
    dx = params["wiggle_amplitude"] * ramp * math.sin(2.0 * math.pi * params["wiggle_freq"] * t_w)
    dy = (
        params["wiggle_y_amplitude"]
        * ramp
        * math.sin(2.0 * math.pi * params["wiggle_y_freq"] * t_w + params["wiggle_y_phase"])
    )
    dz = params["wiggle_bob"] * ramp * math.sin(2.0 * math.pi * params["wiggle_bob_freq"] * t_w)

    offset = np.zeros(3, dtype=np.float32)
    offset[params["wiggle_axis"]] += dx
    offset[1] += dy
    offset[params["vertical_axis"]] += dz
    return offset


def _franka_hand_offset_for_hanger(params):
    offset = np.asarray(params["franka_grip_offset"], dtype=np.float32).copy()
    hanger_finger = params["hanger_finger"]
    if hanger_finger == "left":
        offset[1] += params["gripper_open_gap"]
    elif hanger_finger == "right":
        offset[1] -= params["gripper_open_gap"]
    else:
        raise ValueError(f"Unsupported hanger_finger: {hanger_finger}")

    offset[params["vertical_axis"]] -= params["hanger_finger_drop"]
    return offset


def _franka_target_from_handle_center(handle_center, params, frame, frame_dt):
    handle_center = np.asarray(handle_center, dtype=np.float32)
    grip_offset = _franka_hand_offset_for_hanger(params)
    return (
        handle_center + grip_offset + _lift_offset_array(params, frame) + _wiggle_offset_array(params, frame, frame_dt)
    )


def _add_table(builder, params):
    table_cfg = newton.ModelBuilder.ShapeConfig()
    table_cfg.ke = params["table_ke"]
    table_cfg.kd = params["table_kd"]
    table_cfg.mu = params["table_mu"]

    shape_index = len(builder.shape_type)
    builder.add_shape_box(
        params["table_body"],
        wp.transform(
            wp.vec3(
                *params["table_center_xy"],
                params["table_top_z"] - params["table_thickness"] / 2.0,
            ),
            wp.quat_identity(),
        ),
        hx=params["table_size"] / 2.0,
        hy=params["table_size"] / 2.0,
        hz=params["table_thickness"] / 2.0,
        cfg=table_cfg,
        color=params["table_color"],
        label=params["table_label"],
    )
    return shape_index


def build_model(builder, params, seed=PARAMS["seed"]):
    bag_verts, _ = _load_obj(BAG_OBJ)
    bag_start_particle = len(builder.particle_q)
    table_shape_index = _add_table(builder, params)
    include_payload = params.get("include_payload", True)

    # The grip target is derived from the rest OBJ, so the gripper follows the
    # same trajectory whether or not the cloth bag is actually instantiated.
    handle_indices = _select_handle_top_indices(bag_verts, bag_start_particle, params)
    if handle_indices.size == 0:
        raise ValueError("No handle particles selected for Franka grip")
    handle_center = bag_verts[handle_indices - bag_start_particle].mean(axis=0).astype(np.float32)

    if include_payload:
        info = _build_apple_bag(builder, params, seed=seed)
    else:
        # Robot-only build: no cloth, no apples. Provide the keys downstream needs.
        va = params["vertical_axis"]
        info = {
            "bag_particle_count": 0,
            "top_global_indices": np.zeros(0, dtype=np.int32),
            "body_indices": [],
            "shape_indices": [],
            "particle_radius": params["particle_radius"],
            "z_floor": float(bag_verts[:, va].min()),
            "z_top": float(bag_verts[:, va].max()),
            "half_width": 0.5 * float(bag_verts[:, 0].max() - bag_verts[:, 0].min()),
            "half_depth": 0.5 * float(bag_verts[:, 1].max() - bag_verts[:, 1].min()),
        }
        handle_indices = np.zeros(0, dtype=np.int32)  # no real particles to track
        builder.color()  # SolverVBD needs body_color_groups even with no cloth particles

    info["handle_global_indices"] = handle_indices
    info["handle_center"] = tuple(float(x) for x in handle_center)
    info["table_shape_index"] = table_shape_index
    info["table_top_z"] = params["table_top_z"]
    info["include_payload"] = include_payload
    return info


def _gripper_frac_for_frame(params, frame):
    if not params["close_gripper"]:
        return params["gripper_open_frac"]

    # Close gradually across the lift stage: stay open until the lift starts,
    # then ease shut over ``gripper_close_frames`` frames.
    close_start = params.get("gripper_close_start_frame", params["lift_start_frame"])
    close_frames = params["gripper_close_frames"]
    if close_frames <= 0:
        return params["gripper_closed_frac"] if frame >= close_start else params["gripper_open_frac"]

    alpha = _smoothstep((frame - close_start) / close_frames)
    return params["gripper_open_frac"] * (1.0 - alpha) + params["gripper_closed_frac"] * alpha


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = dict(PARAMS)
        if getattr(args, "no_payload", False):
            self.params["include_payload"] = False
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0
        self._gripper_frac = self.params["gripper_open_frac"]

        seed = getattr(args, "seed", self.params["seed"])
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self._add_robot(builder)
        self.info = build_model(builder, self.params, seed=seed)
        self._handle_rest_center = np.array(self.info["handle_center"], dtype=np.float32)

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
        if self.info["include_payload"]:
            self._initial_handle_center = self.state_0.particle_q.numpy()[self.info["handle_global_indices"]].mean(
                axis=0
            )
        else:
            self._initial_handle_center = self._handle_rest_center.copy()

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
        self._physics_graph = None  # captured lazily on the first CUDA step()

        print(
            f"[apple_bag_franka] bag particles: {self.info['bag_particle_count']}  "
            f"gripped handle verts: {len(self.info['handle_global_indices'])}  "
            f"apples: {len(self.info['body_indices'])}  "
            f"support: {self.params['table_label']}"
        )

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = self.params["draw_wireframe"]
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = self.params["initial_paused"]
        if hasattr(self.viewer, "set_camera"):
            pitch, yaw = _pitch_yaw(self.params["camera_pos"], self.params["camera_target"])
            self.viewer.set_camera(wp.vec3(*self.params["camera_pos"]), pitch, yaw)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

        self._pbar = tqdm(total=getattr(args, "num_frames", None), desc="frame", unit="f")

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

        if self.params["add_finger_pads"]:
            raise ValueError("This example intentionally uses the original FR3 gripper without helper pads")

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

    def _configure_robot_contacts(self):
        shape_body = self.model.shape_body.numpy()
        shape_flags = self.model.shape_flags.numpy().copy()
        shape_mu = self.model.shape_material_mu.numpy().copy()
        shape_ke = self.model.shape_material_ke.numpy().copy()
        shape_kd = self.model.shape_material_kd.numpy().copy()
        shape_margin = self.model.shape_margin.numpy().copy()

        for shape_index, shape_body_index in enumerate(shape_body):
            body_index = int(shape_body_index)
            if 0 <= body_index < self._robot_body_count and body_index not in self._finger_contact_body_indices:
                shape_flags[shape_index] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            elif body_index in self._finger_contact_body_indices:
                shape_mu[shape_index] = self.params["finger_shape_mu"]
                shape_ke[shape_index] = self.params["finger_shape_ke"]
                shape_kd[shape_index] = self.params["finger_shape_kd"]
                shape_margin[shape_index] = self.params["finger_shape_margin"]

        self.model.shape_flags = wp.array(
            shape_flags,
            dtype=self.model.shape_flags.dtype,
            device=self.model.device,
        )
        self.model.shape_material_mu = wp.array(shape_mu, dtype=float, device=self.model.device)
        self.model.shape_material_ke = wp.array(shape_ke, dtype=float, device=self.model.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=float, device=self.model.device)
        self.model.shape_margin = wp.array(shape_margin, dtype=float, device=self.model.device)

    def _configure_joint_drives(self):
        ke = self.model.joint_target_ke.numpy().copy()
        kd = self.model.joint_target_kd.numpy().copy()
        arm_n = self.params["arm_joint_count"]
        ke[:arm_n] = self.params["arm_drive_ke"][:arm_n]
        kd[:arm_n] = self.params["arm_drive_kd"][:arm_n]
        for joint_index in self.params["gripper_joint_indices"]:
            ke[joint_index] = self.params["gripper_drive_ke"]
            kd[joint_index] = self.params["gripper_drive_kd"]
        self.model.joint_target_ke = wp.array(ke, dtype=float, device=self.model.device)
        self.model.joint_target_kd = wp.array(kd, dtype=float, device=self.model.device)

    def _tool_rotation(self):
        return _tool_rotation_from_params(self.params)

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

    def _gripper_frac_for_frame(self):
        return _gripper_frac_for_frame(self.params, self.frame)

    def _franka_target(self):
        target = _franka_target_from_handle_center(
            self._handle_rest_center,
            self.params,
            self.frame,
            self.frame_dt,
        )
        return wp.vec3(float(target[0]), float(target[1]), float(target[2]))

    def _solve_ik_for_current_target(self):
        target_pos = self._franka_target()
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

    def _initialize_robot_pregrasp(self):
        self._gripper_frac = self.params["gripper_open_frac"]
        self._solve_ik_for_current_target()

        joint_q = self.state_0.joint_q.numpy().copy()
        ik_solution = self._joint_q_ik.numpy()[0]
        arm_joint_count = self.params["arm_joint_count"]
        joint_q[:arm_joint_count] = ik_solution[:arm_joint_count]
        gripper_value = self._gripper_joint_value(self._gripper_frac)
        for joint_index in self.params["gripper_joint_indices"]:
            joint_q[joint_index] = gripper_value
        joint_q_wp = wp.array(joint_q, dtype=float, device=self.model.device)

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
        self._gripper_frac = self._gripper_frac_for_frame()
        self._solve_ik_for_current_target()

    def _update_drive_targets(self):
        ik_solution = self._joint_q_ik.numpy()[0]
        arm_n = self.params["arm_joint_count"]
        target_pos = np.zeros(self.model.joint_dof_count)
        target_pos[:arm_n] = ik_solution[:arm_n]
        gripper_value = self._gripper_joint_value(self._gripper_frac)
        for joint_index in self.params["gripper_joint_indices"]:
            target_pos[joint_index] = gripper_value
        self.control.joint_target_pos.assign(wp.array(target_pos, dtype=float, device=self.model.device))

    def _simulate_substeps(self):
        # Pure device-side substep loop -- safe to run inside a CUDA graph
        # capture.  Per-frame host work (IK readback, drive targets) runs in
        # step() before launch and writes into the fixed control/state buffers
        # these captured kernels read from.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        self._update_drive_targets()
        self._simulate_substeps()

    def _capture_physics_graph(self):
        # The state buffers ping-pong each substep, so the captured graph only
        # chains correctly across frames when the swap count is even -- it then
        # reads from and writes back to the same buffer on every replay.
        if self.sim_substeps % 2 != 0:
            raise ValueError(f"Physics CUDA graph capture requires an even sim_substeps; got {self.sim_substeps}.")
        # Stream capture records the launches without executing them, so this
        # does not advance the sim; the warmup run in step() pre-loads every
        # kernel so none compile mid-capture.
        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self._physics_graph = capture.graph

    def step(self):
        self.frame += 1
        self._set_joint_targets()
        self._update_drive_targets()
        if self._physics_graph is not None:
            wp.capture_launch(self._physics_graph)
        elif self.params["enable_physics_cuda_graph"] and wp.get_device().is_cuda:
            # First CUDA frame: run once eagerly to compile the kernels, then
            # capture the identical loop to replay on every subsequent frame.
            self._simulate_substeps()
            self._capture_physics_graph()
        else:
            self._simulate_substeps()
        self.sim_time += self.frame_dt
        self._pbar.update(1)
        self._pbar.set_postfix_str(f"id={self.frame}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self._log_world_axes()
        self.viewer.end_frame()

    def _log_world_axes(self, length=0.2):
        o = wp.vec3(0.0, 0.0, 0.0)
        starts = wp.array([o, o, o], dtype=wp.vec3)
        ends = wp.array(
            [wp.vec3(length, 0.0, 0.0), wp.vec3(0.0, length, 0.0), wp.vec3(0.0, 0.0, length)],
            dtype=wp.vec3,
        )
        colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],
            dtype=wp.vec3,
        )
        self.viewer.log_arrows("world_axes", starts, ends, colors)

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        hand_pos = body_q[self._hand_body][:3]
        assert np.all(np.isfinite(hand_pos)), "Franka hand position contains non-finite values"

        open_joint_value = self._gripper_joint_value(self.params["gripper_open_frac"])
        closed_joint_value = self._gripper_joint_value(self.params["gripper_closed_frac"])
        assert abs(open_joint_value - self.params["gripper_open_gap"]) < self.params["gripper_gap_test_tolerance"]
        assert abs(closed_joint_value - self.params["gripper_closed_gap"]) < self.params["gripper_gap_test_tolerance"]
        expected_gripper_value = self._gripper_joint_value(self._gripper_frac_for_frame())
        joint_target = self.control.joint_target_pos.numpy()
        for joint_index in self.params["gripper_joint_indices"]:
            assert (
                abs(joint_target[joint_index] - expected_gripper_value) < self.params["gripper_open_test_tolerance"]
            ), (
                f"Gripper target mismatch during hang: "
                f"q_target[{joint_index}]={joint_target[joint_index]:.6f} expected {expected_gripper_value:.6f}"
            )

        if not self.info["include_payload"]:
            return  # robot-only run: no cloth/apples to validate

        pq = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(pq)), "Bag particle positions contain non-finite values"

        body_indices = self.info["body_indices"]
        apple_pos = body_q[body_indices][:, :3]
        assert np.all(np.isfinite(apple_pos)), "Apple positions contain non-finite values"

        va = self.params["vertical_axis"]
        min_particle_z = float(np.min(pq[:, va]))
        table_tolerance = self.params["particle_radius"] * self.params["table_tolerance_particle_radius_scale"]
        assert min_particle_z > self.params["table_top_z"] - table_tolerance, (
            f"Bag penetrated below table: z={min_particle_z:.4f}"
        )

        handle_indices = self.info["handle_global_indices"]
        flags = self.model.particle_flags.numpy()
        handle_active = (flags[handle_indices] & int(newton.ParticleFlags.ACTIVE)) != 0
        assert np.all(handle_active), "Handle particles were pinned instead of hanging from the gripper"

        handle_center_z = float(np.mean(pq[handle_indices, va]))
        initial_handle_center_z = float(self._initial_handle_center[va])
        handle_lift = handle_center_z - initial_handle_center_z
        assert handle_lift > self.params["handle_lift_test_min_delta"], (
            f"Handle did not hang on the gripper and lift: dz={handle_lift:.4f}"
        )

        self.pipeline.collide(self.state_0, self.contacts)
        contact_count = int(self.contacts.soft_contact_count.numpy()[0])
        contact_shapes = self.contacts.soft_contact_shape.numpy()[:contact_count]
        finger_contact_count = sum(
            1 for shape_index in contact_shapes if int(shape_index) in self._finger_contact_shape_indices
        )
        assert finger_contact_count >= self.params["finger_hang_contact_test_min"], (
            f"Handle is not contacting the open hanger finger: count={finger_contact_count}"
        )

        az = apple_pos[:, va]
        assert np.all(az > self.info["z_floor"] - 0.06), f"An apple fell through the bag bottom: min z {az.min():.3f}"
        max_apple_z = self.info["z_top"] + self.params["lift_height_delta"] + self.params["wiggle_bob"] + 0.08
        assert np.all(az < max_apple_z), f"An apple is above the handles: max z {az.max():.3f}"

        motion_limits = [0.0, 0.0, 0.0]
        motion_limits[self.params["wiggle_axis"]] += self.params["wiggle_amplitude"]
        motion_limits[1] += self.params["wiggle_y_amplitude"]
        x_lim = self.info["half_width"] + motion_limits[0] + self.params["apple_radius"] + 0.08
        y_lim = self.info["half_depth"] + motion_limits[1] + self.params["apple_radius"] + 0.06
        assert np.all(np.abs(apple_pos[:, 0]) < x_lim), "An apple escaped the bag in x"
        assert np.all(np.abs(apple_pos[:, 1]) < y_lim), "An apple escaped the bag in y"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=PARAMS["seed"])
        parser.add_argument(
            "--no-payload",
            action="store_true",
            help="Run the robot + table only (skip the cloth bag and apples).",
        )
        parser.set_defaults(num_frames=PARAMS["lift_start_frame"] + PARAMS["lift_frames"] + 270)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
