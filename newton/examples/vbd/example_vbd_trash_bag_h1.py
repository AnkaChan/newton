# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Trash Bag H1 (unified AVBD/VBD)
#
# A fixed-base Unitree H1 lifts a loaded drawstring trash bag out of a
# round trash can that sits on a pedestal.  The bag/rope cloth assets and
# the tunnel-closure layout come from example_vbd_trash_bag.py; the H1
# import, Newton-IK task-space control and AVBD/VBD coupling follow
# example_vbd_tablecloth_h1.py.
#
# Scene (z-up):
#   - a box pedestal, with the thin watertight trash can on top
#   - the drawstring bag lining the can; rigid "trash" spheres drop in
#     while the bag settles
#   - the bag's two drawstring handles (the exposed rope ends that bulge
#     out of the four side holes) are rotated to +y / -y so they face the
#     H1's left / right hands
#
# The handles -- and the bag hem patches around the side holes they exit
# through, which is where a grabbed trash bag actually carries its load --
# are pinned from the start (the bag settles around them, as in the base
# example's pin+cinch mode).  The H1 approaches, hooks its curled fingers
# through both handle loops, and on grasp the pinned vertices are
# re-parented from their world pins to the corresponding hand bodies, so
# the hands kinematically carry the handles.  The arms then pull the
# handles UP and APART -- the nearly inextensible rope loop collapses and
# gathers the bag mouth (the cinch of the base example) -- then lift the
# bag and drag it out over the can rim toward the robot.
#
# The H1 itself is kinematic (zero inverse mass): Newton IK tracks the
# task-space waypoints and eval_fk drives the body transforms directly,
# as in example_vbd_bag_franka_pickup_kinematic.py.
#
# Commands:
#   python newton/examples/vbd/example_vbd_trash_bag_h1.py
#   python newton/examples/vbd/example_vbd_trash_bag_h1.py --viewer gl --headless \
#       --record-video trash_bag_h1.mp4
#   python newton/examples/vbd/example_vbd_trash_bag_h1.py --viewer null --test
###########################################################################

from __future__ import annotations

import atexit
import json
import math
import os
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import ParticleFlags

ASSET = os.path.join(os.path.dirname(__file__), "asset")
BAG_OBJ = os.path.join(ASSET, "trash_bag.obj")
ROPE_OBJ = os.path.join(ASSET, "trash_bag_rope.obj")
BAG_INIT_OBJ = os.path.join(ASSET, "trash_bag_init.obj")
ROPE_INIT_OBJ = os.path.join(ASSET, "trash_bag_rope_init.obj")
LAYOUT_JSON = os.path.join(ASSET, "trash_bag_layout.json")

PARAMS = {
    # --- simulation ---
    "fps": 60,
    "sim_substeps": 16,
    "solver_iterations": 12,
    "gravity": -9.81,
    # --- scene layout ---
    # H1 shoulders sit at z=1.54 with a ~0.6 m comfortable pinch reach, so the
    # pedestal raises the handles (bag base + 0.41) to z=1.21: 0.48 m ahead of
    # and 0.33 m below the shoulders.
    "bag_center_x": -0.22,  # bag/can/pedestal center
    "pedestal_top_z": 0.80,  # bag base height (pedestal surface + can floor)
    "pedestal_half_x": 0.17,
    "pedestal_half_y": 0.17,
    "robot_base_x": -0.70,  # H1 pelvis x (fixed base)
    # --- task-space trajectory (seconds) ---
    "settle_time": 2.0,  # bag expands into the can, trash drops in
    "approach_time": 1.0,  # hands travel from rest to above the handles
    "descend_time": 0.8,  # hands descend so fingers thread the handle loops
    "close_time": 0.6,  # thumbs/indexes curl shut; handles attach at the end
    "cinch_time": 1.8,  # pull up + apart: rope gathers the bag mouth shut
    "lift_time": 1.5,  # raise the bag out of the can
    "carry_time": 1.2,  # carry toward the robot, clear of the can
    "hold_time": 1.0,
    # --- grasp geometry ---
    "hover_height": 0.12,  # hover this far above the handle before descending
    "grasp_z_offset": 0.01,  # pinch point height relative to the handle centroid
    "hole_patch_radius": 0.05,  # bag hem within this range of a handle is held with it
    "cinch_up": 0.26,  # how far the handles rise during the cinch [m]
    "cinch_apart": 0.10,  # how far each handle pulls outward in y [m]
    "lift_up": 0.16,  # additional rise during the lift [m]
    "lift_back": 0.10,  # pull toward the robot during the lift [m]
    "carry_back": 0.08,  # additional pull toward the robot: the bag drags out over the rim [m]
    "carry_drop": 0.0,  # keep height while carrying [m]
    # --- rest pose of the hands ---
    "rest_offset_x": 0.30,  # ahead of the robot base
    "rest_y": 0.32,
    "rest_z": 1.24,
    # --- trash (rigid spheres) ---
    "num_trash": 5,
    "trash_radius": 0.034,
    "trash_margin": 0.005,
    "trash_density": 1000.0,
    "trash_ke": 1.0e5,
    "trash_kd": 1.0e2,
    "trash_mu": 0.5,
    # --- round trash can (as in example_vbd_trash_bag.py, translated) ---
    "can_bottom_radius": 0.12,
    "can_top_radius": 0.14,
    "can_height": 0.31,
    "can_wall_thickness": 0.0025,
    "can_floor_thickness": 0.004,
    "can_ke": 5.0e5,
    "can_kd": 5.0e1,
    "can_mu": 0.4,
    "can_margin": 0.002,
    "can_n_around": 72,
    "can_n_rows": 28,
    # --- bag cloth (floppy plastic) ---
    "bag_rest_scale": 1.1,  # mild excess for wrinkles; large scales drag the anchored drawstring while expanding
    "particle_radius": 0.004,
    "cloth_density": 0.08,
    "cloth_tri_ke": 1.0e5,
    "cloth_tri_ka": 5.0e4,
    "cloth_tri_kd": 1.0e1,
    "cloth_edge_ke": 0.2,
    "cloth_edge_kd": 0.1,
    # --- rope cloth (the tie): stiff so pulling collapses the loop ---
    # The bag rest is oversized 1.5x for the floppy billowed look, but the rope
    # rest stays nearly tight: the bag only expands ~2% into the can, and a
    # slack drawstring would dump long loops out of the tunnels when lifted.
    "rope_rest_scale": 1.1,  # match the bag so the drawstring expands with its tunnels
    "rope_density": 0.08,  # match the bag: a near-massless stiff ribbon stretches badly under fabric drag
    "rope_tri_ke": 2.0e6,  # ribbon triangles are tiny: effective membrane stiffness needs the boost
    "rope_tri_ka": 2.0e6,
    "rope_tri_kd": 1.0e2,
    "rope_edge_ke": 0.05,
    "rope_edge_kd": 0.01,
    # --- tunnel-closure springs ---
    "closure_ke": 2.0e4,
    "closure_kd": 1.0e-3,
    # --- rope-to-tunnel tie springs ---
    # Soft ties bind the drawstring to the bag fabric it threads through: the
    # rope travels with its tunnels (it cannot be dragged out or stretched by
    # fabric friction), and pulling the handles gathers the fabric directly.
    "rope_tie_ke": 1.0e4,
    "rope_tie_kd": 1.0e-1,
    "rope_tie_stride": 4,  # tie every Nth rope vertex
    "rope_tie_max_dist": 0.02,  # only rope running inside/near the tunnels
    # --- contacts ---
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1.0e1,
    "soft_contact_mu": 0.3,
    "soft_contact_creation_margin": 0.012,
    "particle_self_contact_radius": 0.004,
    "particle_self_contact_margin": 0.008,
    "rigid_body_particle_contact_buffer_size": 49152,
    "rigid_body_contact_buffer_size": 1024,
    "enable_water_tight": True,
    "rigid_contact_gap": 0.001,
    "shape_ke": 1.0e3,
    "shape_kd": 1.0e-4,
    "shape_mu": 0.5,
    # --- H1 ---
    "robot_contact_ke": 1.0e3,
    "robot_contact_kd": 1.0e-2,
    "robot_contact_mu": 0.5,
    "robot_contact_margin": 0.002,
    "robot_sdf_padding": 0.012,
    "robot_sdf_max_resolution": 64,
    # AVBD joint drives; Newton IK only generates their targets
    "joint_drive_ke": 5.0e4,
    "joint_drive_kd": 5.0e2,
    "torso_drive_ke": 2.0e5,
    "torso_drive_kd": 2.0e3,
    "finger_drive_ke": 4.0e4,
    "finger_drive_kd": 1.0e2,
    "torso_ik_position_weight": 50.0,
    "torso_ik_rotation_weight": 50.0,
    # finger curl fractions
    "other_finger_fraction": 0.85,
    "index_hook_fraction": 0.55,  # index partially curled while threading the loop
    "index_closed_fraction": 0.95,
    "thumb_closed_fraction": 0.95,
    # --- presentation ---
    "bag_color": (0.16, 0.42, 0.19),  # green plastic
    "rope_color": (0.82, 0.15, 0.12),  # red drawstring
    "camera_pos": (1.05, -1.90, 1.55),
    "camera_target": (-0.30, 0.0, 1.05),
    "camera_fov": 45.0,
    "seed": 42,
}

# Pinch point between thumb and index in the hand-link frame, and the hand
# orientation used while pinching, both from example_vbd_tablecloth_h1.py.
HAND_OFFSETS = (
    (0.146273, -0.068447, 0.028077),
    (0.148808, 0.068652, 0.026675),
)

HAND_ROTATIONS = (
    (-0.09, 0.46, 0.03, 0.88),
    (0.09023, 0.46115, -0.03008, 0.88221),
)

THUMB_CLOSED_VALUES = (
    (1.273907, 0.160957, 0.369535, 0.892908),
    (1.192278, 0.195421, 0.400690, 0.679765),
)

_FINGER_GROUP_LEFT_THUMB = wp.constant(0)
_FINGER_GROUP_RIGHT_THUMB = wp.constant(1)
_FINGER_GROUP_LEFT_INDEX = wp.constant(2)
_FINGER_GROUP_RIGHT_INDEX = wp.constant(3)
_FINGER_GROUP_OTHER = wp.constant(4)


@wp.kernel
def set_finger_targets(
    joint_q: wp.array[float],
    finger_indices: wp.array[wp.int32],
    closed_values: wp.array[float],
    finger_groups: wp.array[wp.int32],
    left_thumb_fraction: float,
    right_thumb_fraction: float,
    left_index_fraction: float,
    right_index_fraction: float,
    other_fraction: float,
):
    i = wp.tid()
    group = finger_groups[i]
    fraction = other_fraction
    if group == _FINGER_GROUP_LEFT_THUMB:
        fraction = left_thumb_fraction
    elif group == _FINGER_GROUP_RIGHT_THUMB:
        fraction = right_thumb_fraction
    elif group == _FINGER_GROUP_LEFT_INDEX:
        fraction = left_index_fraction
    elif group == _FINGER_GROUP_RIGHT_INDEX:
        fraction = right_index_fraction
    joint_q[finger_indices[i]] = fraction * closed_values[i]


@wp.kernel
def drive_handle_vertices(
    handle_indices: wp.array[wp.int32],
    pinned_positions: wp.array[wp.vec3],  # world pins used before the grasp
    local_positions: wp.array[wp.vec3],  # hand-frame offsets used after the grasp
    handle_hand_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    attached: int,
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    tid = wp.tid()
    vi = handle_indices[tid]
    if attached != 0:
        p = wp.transform_point(body_q[handle_hand_body[tid]], local_positions[tid])
    else:
        p = pinned_positions[tid]
    pos_0[vi] = p
    pos_1[vi] = p


def _pitch_yaw(pos, target):
    d = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-9
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(d[2])))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return pitch, yaw


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _normalized_quat(values) -> wp.quat:
    q = np.asarray(values, dtype=np.float32)
    q /= np.linalg.norm(q)
    return wp.quat(*q)


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [i for i, label in enumerate(labels) if label.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def _load_obj(path):
    """Load a triangle-mesh OBJ preserving vertex order (so JSON indices stay valid)."""
    vertices = []
    faces = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.extend([idx[0] - 1, idx[k] - 1, idx[k + 1] - 1])
    return np.array(vertices, dtype=np.float32), faces


def _as_numpy(array):
    if hasattr(array, "numpy"):
        return array.numpy()
    return np.asarray(array)


def _add_filter_entries(filter_map, key, values):
    if not values:
        return
    filter_map.setdefault(int(key), set()).update(int(value) for value in values)


def _triangles_by_vertex(tri_indices):
    tri_indices = np.asarray(tri_indices, dtype=np.int32).reshape(-1, 3)
    vertex_triangles = {}
    for tri_id, tri in enumerate(tri_indices):
        for vertex in tri:
            vertex_triangles.setdefault(int(vertex), set()).add(int(tri_id))
    return vertex_triangles


def _edges_by_vertex(edge_indices):
    edge_indices = np.asarray(edge_indices, dtype=np.int32).reshape(-1, 4)
    vertex_edges = {}
    for edge_id, edge in enumerate(edge_indices):
        for vertex in edge[2:4]:
            if vertex >= 0:
                vertex_edges.setdefault(int(vertex), set()).add(int(edge_id))
    return vertex_edges


def _split_tunnel_sides(tunnel_pairs):
    pairs = [(int(i), int(j)) for i, j in np.asarray(tunnel_pairs, dtype=np.int32).reshape(-1, 2)]
    if len(pairs) < 2:
        return [pairs]
    midpoint = len(pairs) // 2
    return [pairs[:midpoint], pairs[midpoint:]]


def _build_tunnel_seam_contact_filters(model, tunnel_pairs):
    """Build external VBD self-contact filters across tunnel closure seams."""
    vertex_triangles = _triangles_by_vertex(_as_numpy(model.tri_indices))
    vertex_edges = _edges_by_vertex(_as_numpy(model.edge_indices))
    vertex_filter = {}
    edge_filter = {}

    def add_vertex_triangle_filter(vertices, other_vertices):
        other_tris = set()
        for vertex in other_vertices:
            other_tris.update(vertex_triangles.get(int(vertex), ()))
        for vertex in vertices:
            _add_filter_entries(vertex_filter, vertex, other_tris)

    for side_pairs in _split_tunnel_sides(tunnel_pairs):
        for flap_vertex, wall_vertex in side_pairs:
            add_vertex_triangle_filter((flap_vertex,), (wall_vertex,))
            add_vertex_triangle_filter((wall_vertex,), (flap_vertex,))
            flap_edges = vertex_edges.get(int(flap_vertex), ())
            wall_edges = vertex_edges.get(int(wall_vertex), ())
            for edge in flap_edges:
                _add_filter_entries(edge_filter, edge, wall_edges)
            for edge in wall_edges:
                _add_filter_entries(edge_filter, edge, flap_edges)

        for (flap_a, wall_a), (flap_b, wall_b) in pairwise(side_pairs):
            add_vertex_triangle_filter((flap_a, flap_b), (wall_a, wall_b))
            add_vertex_triangle_filter((wall_a, wall_b), (flap_a, flap_b))

    vertex_filter = {key: sorted(values) for key, values in vertex_filter.items()}
    edge_filter = {key: sorted(values) for key, values in edge_filter.items()}
    return vertex_filter, edge_filter


def build_can_mesh(bottom_radius, top_radius, z_bottom, height, wall_thickness, floor_thickness, n_around, n_rows):
    """Thin, watertight truncated-cone bin (open top); see example_vbd_trash_bag.py."""
    z_top = z_bottom + height
    z_floor_top = z_bottom + floor_thickness
    r_bot_out = bottom_radius + wall_thickness
    r_top_out = top_radius + wall_thickness

    profile = [(0.0, z_bottom), (r_bot_out, z_bottom)]
    for j in range(1, n_rows + 1):
        u = j / n_rows
        profile.append((r_bot_out + u * (r_top_out - r_bot_out), z_bottom + u * height))
    profile.append((top_radius, z_top))
    for j in range(1, n_rows + 1):
        u = j / n_rows
        profile.append((top_radius + u * (bottom_radius - top_radius), z_top + u * (z_floor_top - z_top)))
    profile.append((0.0, z_floor_top))

    verts = []
    rings = []
    for r, z in profile:
        if r <= 1e-9:
            rings.append(("center", len(verts)))
            verts.append([0.0, 0.0, float(z)])
        else:
            ring = []
            for k in range(n_around):
                th = 2.0 * math.pi * k / n_around
                ring.append(len(verts))
                verts.append([r * math.cos(th), r * math.sin(th), float(z)])
            rings.append(("ring", ring))

    faces = []
    for i in range(len(profile) - 1):
        a, b = rings[i], rings[i + 1]
        if a[0] == "ring" and b[0] == "ring":
            ra, rb = a[1], b[1]
            for k in range(n_around):
                k2 = (k + 1) % n_around
                faces.append([ra[k], rb[k], rb[k2]])
                faces.append([ra[k], rb[k2], ra[k2]])
        elif a[0] == "center":
            c, rb = a[1], b[1]
            for k in range(n_around):
                faces.append([c, rb[k], rb[(k + 1) % n_around]])
        else:
            ra, c = a[1], b[1]
            for k in range(n_around):
                faces.append([ra[k], c, ra[(k + 1) % n_around]])
    faces = np.array(faces, dtype=np.int32)[:, ::-1]
    return np.array(verts, dtype=np.float32), faces.reshape(-1)


def _bag_world_transform(rope_init_verts, layout, params):
    """Rotation about z aligning the LEFT handle with +y, plus the bag base offset.

    The H1's left hand grabs at +y and its right hand at -y, so the bag (and
    everything pinned to it) is yawed so its two drawstring handles face the
    two hands, then translated onto the pedestal.
    """
    hv = layout["rope"]["handle_vertex_indices"]
    left_centroid = rope_init_verts[hv["left"]].mean(axis=0)
    theta = 0.5 * math.pi - math.atan2(float(left_centroid[1]), float(left_centroid[0]))
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    offset = np.array([params["bag_center_x"], 0.0, params["pedestal_top_z"]], dtype=np.float32)
    return rot, offset


def _add_h1(builder: newton.ModelBuilder, params: dict):
    robot_body_start = builder.body_count
    robot_joint_start = builder.joint_count
    robot_dof_start = builder.joint_dof_count
    robot_shape_start = builder.shape_count
    builder.add_mjcf(
        newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml",
        xform=wp.transform(wp.vec3(params["robot_base_x"], 0.0, 0.0), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        ctrl_direct=False,
        parse_visuals=True,
        parse_sites=True,
        collider_classes=("collision",),
        no_class_as_colliders=True,
    )
    robot_body_end = builder.body_count
    robot_joint_end = builder.joint_count
    robot_dof_end = builder.joint_dof_count
    robot_shape_end = builder.shape_count

    for dof in range(robot_dof_start, robot_dof_end):
        builder.joint_target_ke[dof] = params["joint_drive_ke"]
        builder.joint_target_kd[dof] = params["joint_drive_kd"]
    torso_joint = _find_suffix(builder.joint_label, "torso_joint")
    torso_dof = builder.joint_qd_start[torso_joint]
    builder.joint_target_ke[torso_dof] = params["torso_drive_ke"]
    builder.joint_target_kd[torso_dof] = params["torso_drive_kd"]
    finger_tokens = tuple(
        f"/{side}_{digit}_" for side in ("L", "R") for digit in ("thumb", "index", "middle", "ring", "pinky")
    )
    for joint in range(robot_joint_start, robot_joint_end):
        if any(token in builder.joint_label[joint] for token in finger_tokens):
            dof = builder.joint_qd_start[joint]
            builder.joint_target_ke[dof] = params["finger_drive_ke"]
            builder.joint_target_kd[dof] = params["finger_drive_kd"]

    body_names = {
        "torso": "torso_link",
        "left_hand": "left_hand_link",
        "right_hand": "right_hand_link",
    }
    body_indices = {name: _find_suffix(builder.body_label, suffix) for name, suffix in body_names.items()}
    body_indices["robot_body_count"] = robot_body_end - robot_body_start

    # Rigid colliders stay active for AVBD rigid contact and for cloth contact,
    # EXCEPT the hand/finger colliders: the handles they hold are kinematically
    # attached, so finger-vs-handle contact would only fight the attachment.
    shape_collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_collision_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    collision_mask = shape_collision_flag | particle_collision_flag
    hand_tokens = ("/L_thumb_", "/L_index_", "/L_middle_", "/L_ring_", "/L_pinky_", "/left_hand_link")
    hand_tokens += ("/R_thumb_", "/R_index_", "/R_middle_", "/R_ring_", "/R_pinky_", "/right_hand_link")
    hand_bodies = {
        body
        for body in range(robot_body_start, robot_body_end)
        if any(token in builder.body_label[body] for token in hand_tokens)
    }
    robot_rigid_shapes = []
    for shape in range(robot_shape_start, robot_shape_end):
        original_flags = int(builder.shape_flags[shape])
        is_rigid_collider = bool(original_flags & shape_collision_flag)
        builder.shape_flags[shape] &= ~collision_mask
        if is_rigid_collider:
            builder.shape_flags[shape] |= shape_collision_flag
            if builder.shape_body[shape] not in hand_bodies:
                builder.shape_flags[shape] |= particle_collision_flag
            builder.shape_gap[shape] = params["rigid_contact_gap"]
            builder.shape_material_ke[shape] = params["robot_contact_ke"]
            builder.shape_material_kd[shape] = params["robot_contact_kd"]
            builder.shape_material_mu[shape] = params["robot_contact_mu"]
            builder.shape_margin[shape] = params["robot_contact_margin"]
            builder.shape_sdf_padding[shape] = params["robot_sdf_padding"]
            builder.shape_sdf_max_resolution[shape] = params["robot_sdf_max_resolution"]
            builder.shape_sdf_target_voxel_size[shape] = None
            robot_rigid_shapes.append(shape)

    return body_indices, robot_rigid_shapes


def _add_pedestal_and_can(builder: newton.ModelBuilder, params: dict):
    """Box pedestal with the watertight trash can on top."""
    pedestal_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["shape_mu"],
        gap=params["rigid_contact_gap"],
        has_particle_collision=True,
    )
    # The can's outer floor hangs 0.01 below the bag base; the pedestal top
    # meets it there so the can sits flush.
    pedestal_top = params["pedestal_top_z"] - 0.01
    builder.add_shape_box(
        -1,
        xform=wp.transform(
            wp.vec3(params["bag_center_x"], 0.0, 0.5 * pedestal_top),
            wp.quat_identity(),
        ),
        hx=params["pedestal_half_x"],
        hy=params["pedestal_half_y"],
        hz=0.5 * pedestal_top,
        cfg=pedestal_cfg,
        color=wp.vec3(0.45, 0.33, 0.22),
        label="pedestal",
    )

    can_cfg = newton.ModelBuilder.ShapeConfig()
    can_cfg.ke = params["can_ke"]
    can_cfg.kd = params["can_kd"]
    can_cfg.mu = params["can_mu"]
    can_cfg.has_particle_collision = True
    can_cfg.margin = params["can_margin"]
    can_v, can_f = build_can_mesh(
        params["can_bottom_radius"],
        params["can_top_radius"],
        -0.01,
        params["can_height"],
        params["can_wall_thickness"],
        params["can_floor_thickness"],
        params["can_n_around"],
        params["can_n_rows"],
    )
    can_v = can_v + np.array([params["bag_center_x"], 0.0, params["pedestal_top_z"]], dtype=np.float32)
    builder.add_shape_mesh(
        -1,
        mesh=newton.Mesh(can_v, can_f),
        cfg=can_cfg,
        label="trash_can",
        color=wp.vec3(0.35, 0.37, 0.40),
    )


def _add_bag_and_rope(builder: newton.ModelBuilder, params: dict):
    """Bag + drawstring cloth, tunnel springs, handle pins; yawed and raised."""
    with open(LAYOUT_JSON, encoding="utf-8") as file:
        layout = json.load(file)
    pr = params["particle_radius"]

    bag_verts, bag_faces = _load_obj(BAG_OBJ)
    bag_faces_array = np.array(bag_faces, dtype=np.int32).reshape(-1, 3)
    bag_start = len(builder.particle_q)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=params["bag_rest_scale"],  # oversize the REST shape; initial particle_q is overridden below
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

    rope_verts, rope_faces = _load_obj(ROPE_OBJ)
    rope_faces_array = np.array(rope_faces, dtype=np.int32).reshape(-1, 3)
    rope_start = len(builder.particle_q)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=params["rope_rest_scale"],
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=rope_verts.tolist(),
        indices=rope_faces,
        density=params["rope_density"],
        tri_ke=params["rope_tri_ke"],
        tri_ka=params["rope_tri_ka"],
        tri_kd=params["rope_tri_kd"],
        edge_ke=params["rope_edge_ke"],
        edge_kd=params["rope_edge_kd"],
        particle_radius=pr,
    )

    # Override the INITIAL positions with the round (unflattened) shapes, yawed
    # so the handles face the hands and translated onto the pedestal.  The rest
    # shapes above stay untouched: cloth elasticity is rigid-motion invariant.
    bag_init_verts, _ = _load_obj(BAG_INIT_OBJ)
    rope_init_verts, _ = _load_obj(ROPE_INIT_OBJ)
    assert len(bag_init_verts) == len(bag_verts), "bag init/rest vertex count mismatch"
    assert len(rope_init_verts) == len(rope_verts), "rope init/rest vertex count mismatch"
    rot, offset = _bag_world_transform(rope_init_verts, layout, params)
    bag_init_world = bag_init_verts @ rot.T + offset
    rope_init_world = rope_init_verts @ rot.T + offset
    for i, p in enumerate(bag_init_world):
        builder.particle_q[bag_start + i] = wp.vec3(*map(float, p))
    for i, p in enumerate(rope_init_world):
        builder.particle_q[rope_start + i] = wp.vec3(*map(float, p))

    tunnel_spring_pairs = np.array(
        [[bag_start + i, bag_start + j] for i, j in layout["tunnel_spring_pairs"]], dtype=np.int32
    )
    for i, j in tunnel_spring_pairs:
        builder.add_spring(i, j, params["closure_ke"], params["closure_kd"], 0.0)
        builder.spring_rest_length[-1] = 0.0

    hv = layout["rope"]["handle_vertex_indices"]
    left_idx = np.array([rope_start + i for i in hv["left"]], dtype=np.int32)
    right_idx = np.array([rope_start + i for i in hv["right"]], dtype=np.int32)
    left_centroid = rope_init_world[hv["left"]].mean(axis=0)
    right_centroid = rope_init_world[hv["right"]].mean(axis=0)

    # rope-to-tunnel ties: every Nth non-handle rope vertex binds to its
    # nearest bag vertex when the two run close together (the tunnel walls)
    handle_local = {int(i) for i in hv["left"]} | {int(i) for i in hv["right"]}
    num_tie_springs = 0
    for i in range(0, len(rope_init_world), params["rope_tie_stride"]):
        if i in handle_local:
            continue
        offsets = bag_init_world - rope_init_world[i]
        j = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))
        dist = float(np.linalg.norm(offsets[j]))
        if dist < params["rope_tie_max_dist"]:
            builder.add_spring(rope_start + i, bag_start + j, params["rope_tie_ke"], params["rope_tie_kd"], 0.0)
            builder.spring_rest_length[-1] = dist
            num_tie_springs += 1

    # Bag hem patches around the side holes each handle exits through.  The
    # hands hold these together with the rope handles: a grabbed trash bag
    # carries its weight through the hole rims, not through the free
    # drawstring (whose thin ribbon would otherwise stretch under the full
    # bag load faster than VBD can converge).
    radius = params["hole_patch_radius"]
    left_patch = np.where(np.linalg.norm(bag_init_world - left_centroid, axis=1) < radius)[0]
    right_patch = np.where(np.linalg.norm(bag_init_world - right_centroid, axis=1) < radius)[0]
    left_patch_idx = (left_patch + bag_start).astype(np.int32)
    right_patch_idx = (right_patch + bag_start).astype(np.int32)

    return {
        "bag_start": bag_start,
        "bag_count": len(bag_verts),
        "rope_start": rope_start,
        "rope_count": len(rope_verts),
        "bag_faces": bag_faces_array,
        "rope_faces": rope_faces_array,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "left_patch_idx": left_patch_idx,
        "right_patch_idx": right_patch_idx,
        "left_centroid": left_centroid,
        "right_centroid": right_centroid,
        "tunnel_spring_pairs": tunnel_spring_pairs,
        "num_tie_springs": num_tie_springs,
        "bag_init_world": bag_init_world,
    }


def _add_trash(builder: newton.ModelBuilder, bag_info: dict, params: dict, seed: int):
    rng = np.random.default_rng(seed)
    r = params["trash_radius"]
    bag_init = bag_info["bag_init_world"]
    center = np.array([params["bag_center_x"], 0.0, params["pedestal_top_z"]], dtype=np.float64)
    mid = bag_init[(bag_init[:, 2] > center[2] + 0.05) & (bag_init[:, 2] < center[2] + 0.30)]
    bag_r = float(np.median(np.hypot(mid[:, 0] - center[0], mid[:, 1] - center[1])))
    rad_in = max(0.0, bag_r - r - 0.015)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["trash_density"]
    cfg.ke = params["trash_ke"]
    cfg.kd = params["trash_kd"]
    cfg.mu = params["trash_mu"]
    cfg.has_particle_collision = True
    cfg.margin = params["trash_margin"]

    colors = [
        wp.vec3(0.85, 0.30, 0.20),
        wp.vec3(0.90, 0.70, 0.20),
        wp.vec3(0.30, 0.55, 0.85),
        wp.vec3(0.75, 0.45, 0.75),
        wp.vec3(0.45, 0.75, 0.40),
    ]
    body_indices = []
    n = params["num_trash"]
    for i in range(n):
        ang = i * 2.39996
        rr = rad_in * math.sqrt((i + 0.5) / n)
        px = float(center[0] + rr * math.cos(ang) + rng.uniform(-0.004, 0.004))
        py = float(center[1] + rr * math.sin(ang) + rng.uniform(-0.004, 0.004))
        pz = float(center[2] + r + 0.03 + i * 0.06)
        body = builder.add_body(xform=wp.transform(wp.vec3(px, py, pz), wp.quat_identity()), label=f"trash_{i}")
        body_indices.append(body)
        builder.add_shape_sphere(body, radius=r, cfg=cfg, color=colors[i % len(colors)])
    return body_indices


class Example:
    DEFAULT_PARAMS = PARAMS
    PHASE_NAMES = ("settle", "approach", "descend", "close", "cinch", "lift", "carry", "hold")

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = dict(self.DEFAULT_PARAMS)
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.phase = "settle"
        self.attached = False
        self._ik_debug_state = None
        self._debug = bool(getattr(args, "debug", False))

        seed = getattr(args, "seed", self.params["seed"])
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.robot_bodies, robot_rigid_shapes = _add_h1(builder, self.params)
        self.robot_coord_count = builder.joint_coord_count
        _add_pedestal_and_can(builder, self.params)
        self.bag_info = _add_bag_and_rope(builder, self.params)
        self.trash_bodies = self._build_trash(builder, seed)

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=self.params["shape_ke"],
            kd=self.params["shape_kd"],
            mu=self.params["shape_mu"],
            gap=self.params["rigid_contact_gap"],
        )
        ground_shape = builder.add_ground_plane(cfg=ground_cfg)
        for robot_shape in robot_rigid_shapes:
            builder.add_shape_collision_filter_pair(robot_shape, ground_shape)
        builder.color(include_bending=True)

        if self.params["enable_water_tight"]:
            builder.enable_rigid_mesh_sdfs()
        self.model = builder.finalize()
        self.model.soft_contact_ke = self.params["soft_contact_ke"]
        self.model.soft_contact_kd = self.params["soft_contact_kd"]
        self.model.soft_contact_mu = self.params["soft_contact_mu"]
        self.device = self.model.device

        # The H1 is kinematic: zero inverse mass/inertia, body_q driven directly
        # from the per-frame IK solution via eval_fk (as in
        # example_vbd_bag_franka_pickup_kinematic.py).  The trash spheres stay
        # dynamic; robot-cloth coupling is one-way (robot pushes cloth).
        self.robot_body_count = self.robot_bodies["robot_body_count"]
        inv_mass = self.model.body_inv_mass.numpy()
        inv_inertia = self.model.body_inv_inertia.numpy()
        inv_mass[: self.robot_body_count] = 0.0
        inv_inertia[: self.robot_body_count] = 0.0
        self.model.body_inv_mass = wp.array(inv_mass, dtype=float, device=self.device)
        self.model.body_inv_inertia = wp.array(inv_inertia, dtype=wp.mat33, device=self.device)

        # --- pinned drawstring handles + bag hole patches: inactive particles,
        # kinematically driven (world pins before the grasp, hand frames after) ---
        left_idx = np.concatenate([self.bag_info["left_idx"], self.bag_info["left_patch_idx"]])
        right_idx = np.concatenate([self.bag_info["right_idx"], self.bag_info["right_patch_idx"]])
        handle_idx = np.concatenate([left_idx, right_idx])
        flags = self.model.particle_flags.numpy()
        for vi in handle_idx:
            flags[vi] = flags[vi] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags, dtype=wp.int32)

        self.hand_bodies = [self.robot_bodies["left_hand"], self.robot_bodies["right_hand"]]
        self.hand_offsets = [wp.vec3(*values) for values in HAND_OFFSETS]
        self.hand_rotations = [_normalized_quat(values) for values in HAND_ROTATIONS]

        pq = self.model.particle_q.numpy()
        self.handle_indices = wp.array(handle_idx, dtype=wp.int32, device=self.device)
        self.handle_pinned_positions = wp.array(pq[handle_idx].copy(), dtype=wp.vec3, device=self.device)
        self.handle_local_positions = wp.zeros(len(handle_idx), dtype=wp.vec3, device=self.device)
        handle_hand = np.concatenate(
            [
                np.full(len(left_idx), self.hand_bodies[0], dtype=np.int32),
                np.full(len(right_idx), self.hand_bodies[1], dtype=np.int32),
            ]
        )
        self.handle_hand_body = wp.array(handle_hand, dtype=wp.int32, device=self.device)
        self._handle_idx_np = handle_idx
        self._left_driven_count = len(left_idx)

        vertex_filter, edge_filter = _build_tunnel_seam_contact_filters(
            self.model, self.bag_info["tunnel_spring_pairs"]
        )

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            # AVBD advances the H1 and the trash in the same solve as the VBD cloth.
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=self.params["particle_self_contact_radius"],
            particle_self_contact_margin=self.params["particle_self_contact_margin"],
            particle_external_vertex_contact_filtering_map=vertex_filter,
            particle_external_edge_contact_filtering_map=edge_filter,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=self.params["rigid_body_contact_buffer_size"],
            rigid_body_particle_contact_buffer_size=self.params["rigid_body_particle_contact_buffer_size"],
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
            rigid_joint_linear_kd=1.0e2,
            rigid_joint_angular_kd=1.0e2,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            soft_contact_margin=self.params["soft_contact_creation_margin"],
            enable_water_tight_rigid_soft_contact=self.params["enable_water_tight"],
            contact_matching="latest",
        )

        # --- task-space waypoints, ordered [left hand, right hand] ---
        lc = self.bag_info["left_centroid"]
        rc = self.bag_info["right_centroid"]
        gz = self.params["grasp_z_offset"]
        self.grasp_positions = np.asarray(
            [
                (lc[0], lc[1], lc[2] + gz),
                (rc[0], rc[1], rc[2] + gz),
            ],
            dtype=np.float32,
        )
        hover = self.grasp_positions.copy()
        hover[:, 2] += self.params["hover_height"]
        self.hover_positions = hover
        base_x = self.params["robot_base_x"]
        self.rest_positions = np.asarray(
            [
                (base_x + self.params["rest_offset_x"], self.params["rest_y"], self.params["rest_z"]),
                (base_x + self.params["rest_offset_x"], -self.params["rest_y"], self.params["rest_z"]),
            ],
            dtype=np.float32,
        )
        cinch = self.grasp_positions.copy()
        cinch[:, 2] += self.params["cinch_up"]
        cinch[0, 1] += self.params["cinch_apart"]
        cinch[1, 1] -= self.params["cinch_apart"]
        self.cinch_positions = cinch
        lift = cinch.copy()
        lift[:, 2] += self.params["lift_up"]
        lift[:, 0] -= self.params["lift_back"]
        self.lift_positions = lift
        carry = lift.copy()
        carry[:, 0] -= self.params["carry_back"]
        carry[:, 2] -= self.params["carry_drop"]
        self.carry_positions = carry
        self.target_hand_positions = self.rest_positions.copy()

        self._phase_ends = {}
        t = 0.0
        for name in self.PHASE_NAMES:
            t += self.params[f"{name}_time"]
            self._phase_ends[name] = t
        self.total_time = t

        # Seed the arms in a natural elbows-down pose so the initial IK solve
        # converges to the human-like branch instead of an elbows-up one.
        q = self.model.joint_q.numpy()
        q_starts = self.model.joint_q_start.numpy()
        for side in ("left", "right"):
            for name, value in (("shoulder_pitch_joint", 0.6), ("elbow_joint", 0.9)):
                joint = _find_suffix(self.model.joint_label, f"{side}_{name}")
                q[q_starts[joint]] = value
        self.model.joint_q.assign(q)

        self._setup_ik()
        self._solve_ik(
            self.rest_positions,
            left_thumb_fraction=0.0,
            right_thumb_fraction=0.0,
            left_index_fraction=0.0,
            right_index_fraction=0.0,
            other_fraction=0.0,
            iterations=96,
        )
        self.model.joint_q.assign(self.ik_joint_q_flat)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)

        # kinematic robot driving: FK scratch state + previous body transforms
        self._fk_state = self.model.state()
        self._fk_joint_qd = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.device)
        self._prev_fk_body_q = self.state_0.body_q.numpy()[: self.robot_body_count].copy()
        self._robot_body_com = self.model.body_com.numpy()[: self.robot_body_count].copy()

        # combined mesh index buffers for per-frame colored rendering
        bag_start = self.bag_info["bag_start"]
        rope_start = self.bag_info["rope_start"]
        self._bag_tri_indices = wp.array(
            (self.bag_info["bag_faces"] + bag_start).flatten(), dtype=wp.int32, device=self.device
        )
        self._rope_tri_indices = wp.array(
            (self.bag_info["rope_faces"] + rope_start).flatten(), dtype=wp.int32, device=self.device
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False  # we log the bag and rope separately, colored
        if hasattr(self.viewer, "set_camera"):
            pitch, yaw = _pitch_yaw(self.params["camera_pos"], self.params["camera_target"])
            self.viewer.set_camera(wp.vec3(*self.params["camera_pos"]), pitch, yaw)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

        # optional mp4 recording (GL viewer only)
        self._video_writer = None
        self._record_until = int(getattr(args, "num_frames", 0) or 0)
        record_path = getattr(args, "record_video", None)
        if record_path:
            if not hasattr(self.viewer, "get_frame"):
                raise ValueError("--record-video requires the gl viewer")
            import imageio  # noqa: PLC0415

            record_dir = os.path.dirname(os.path.abspath(record_path))
            os.makedirs(record_dir, exist_ok=True)
            self._video_writer = imageio.get_writer(record_path, fps=self.fps, codec="libx264", quality=8)
            # Backup only: the writer is closed deterministically after the last
            # frame; closing from atexit is unreliable (the ffmpeg pipe may
            # already be torn down during interpreter shutdown).
            atexit.register(self._close_video)
            print(f"[trash_bag_h1] recording video to {record_path}")

        print(
            f"[trash_bag_h1] bag verts {self.bag_info['bag_count']}  rope verts {self.bag_info['rope_count']}  "
            f"trash {len(self.trash_bodies)}  total frames {int(self.total_time * self.fps)}"
        )

    def _close_video(self):
        if self._video_writer is not None:
            writer, self._video_writer = self._video_writer, None
            try:
                writer.close()
                print("[trash_bag_h1] video finalized")
            except OSError as error:
                print(f"[trash_bag_h1] video close failed: {error}")

    # ------------------------------------------------------------------ IK ---
    def _setup_ik(self):
        initial_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, initial_state)
        body_q = initial_state.body_q.numpy()

        self.torso_body = self.robot_bodies["torso"]
        torso_transform = wp.transform(*body_q[self.torso_body])
        self.torso_position_objective = ik.IKObjectivePosition(
            link_index=self.torso_body,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(torso_transform)], dtype=wp.vec3),
            weight=self.params["torso_ik_position_weight"],
        )
        torso_rotation = wp.transform_get_rotation(torso_transform)
        self.torso_rotation_objective = ik.IKObjectiveRotation(
            link_index=self.torso_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*torso_rotation)], dtype=wp.vec4),
            weight=self.params["torso_ik_rotation_weight"],
        )

        self.position_objectives = []
        self.rotation_objectives = []
        for body, offset, rotation in zip(self.hand_bodies, self.hand_offsets, self.hand_rotations, strict=True):
            initial_position = wp.transform_point(wp.transform(*body_q[body]), offset)
            self.position_objectives.append(
                ik.IKObjectivePosition(
                    link_index=body,
                    link_offset=offset,
                    target_positions=wp.array([initial_position], dtype=wp.vec3),
                    weight=5.0,
                )
            )
            self.rotation_objectives.append(
                ik.IKObjectiveRotation(
                    link_index=body,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([wp.vec4(*rotation)], dtype=wp.vec4),
                    weight=0.2,
                )
            )

        joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=1.0,
        )
        self.ik_joint_q = wp.clone(self.model.joint_q).reshape((1, self.model.joint_coord_count))
        self.ik_joint_q_flat = self.ik_joint_q.reshape((-1,))
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[
                *self.position_objectives,
                *self.rotation_objectives,
                self.torso_position_objective,
                self.torso_rotation_objective,
                joint_limits,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        q_starts = self.model.joint_q_start.numpy()
        finger_indices = []
        closed_values = []
        finger_groups = []
        for side_index, side in enumerate(("L", "R")):
            thumb_yaw, thumb_pitch, thumb_intermediate, thumb_distal = THUMB_CLOSED_VALUES[side_index]
            finger_names_and_values = (
                ("thumb_proximal_yaw_joint", thumb_yaw),
                ("thumb_proximal_pitch_joint", thumb_pitch),
                ("thumb_intermediate_joint", thumb_intermediate),
                ("thumb_distal_joint", thumb_distal),
                ("index_proximal_joint", 1.2),
                ("index_intermediate_joint", 1.2),
                ("middle_proximal_joint", 1.0),
                ("middle_intermediate_joint", 1.0),
                ("ring_proximal_joint", 1.0),
                ("ring_intermediate_joint", 1.0),
                ("pinky_proximal_joint", 1.0),
                ("pinky_intermediate_joint", 1.0),
            )
            for suffix, value in finger_names_and_values:
                joint = _find_suffix(self.model.joint_label, f"{side}_{suffix}")
                finger_indices.append(int(q_starts[joint]))
                closed_values.append(value)
                if suffix.startswith("thumb_"):
                    finger_groups.append(_FINGER_GROUP_LEFT_THUMB if side == "L" else _FINGER_GROUP_RIGHT_THUMB)
                elif suffix.startswith("index_"):
                    finger_groups.append(_FINGER_GROUP_LEFT_INDEX if side == "L" else _FINGER_GROUP_RIGHT_INDEX)
                else:
                    finger_groups.append(_FINGER_GROUP_OTHER)
        self.finger_indices = wp.array(finger_indices, dtype=wp.int32, device=self.device)
        self.closed_finger_values = wp.array(closed_values, dtype=float, device=self.device)
        self.finger_groups = wp.array(finger_groups, dtype=wp.int32, device=self.device)

    def _set_ik_positions(self, positions: np.ndarray):
        for objective, position in zip(self.position_objectives, positions, strict=True):
            objective.set_target_position(0, wp.vec3(*position))

    def _solve_ik(
        self,
        positions: np.ndarray,
        left_thumb_fraction: float,
        right_thumb_fraction: float,
        left_index_fraction: float,
        right_index_fraction: float,
        other_fraction: float,
        iterations: int = 24,
    ):
        self._set_ik_positions(positions)
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=iterations)
        wp.launch(
            set_finger_targets,
            dim=self.finger_indices.shape[0],
            inputs=[
                self.ik_joint_q_flat,
                self.finger_indices,
                self.closed_finger_values,
                self.finger_groups,
                left_thumb_fraction,
                right_thumb_fraction,
                left_index_fraction,
                right_index_fraction,
                other_fraction,
            ],
        )

    # ------------------------------------------------------- trajectory ---
    def _update_trajectory(self):
        positions, left_thumb, right_thumb, left_index, right_index, other = self._hand_targets(self.sim_time)
        self.target_hand_positions = np.asarray(positions, dtype=np.float32)
        self._solve_ik(
            self.target_hand_positions,
            left_thumb_fraction=left_thumb,
            right_thumb_fraction=right_thumb,
            left_index_fraction=left_index,
            right_index_fraction=right_index,
            other_fraction=other,
        )
        self._drive_robot_kinematic()

    def _hand_targets(self, t):
        """Task-space targets and finger fractions [left, right] at time t."""
        if t < self._phase_ends["settle"]:
            self.phase = "settle"
            return self.rest_positions, 0.0, 0.0, 0.0, 0.0, 0.0
        return self._bag_grasp_targets(t)

    def _bag_grasp_targets(self, t):
        """The approach -> hold sequence that grabs and lifts the bag."""
        p = self.params
        ends = self._phase_ends

        thumb = 0.0
        index = 0.0
        other = 0.0
        if t < ends["approach"]:
            self.phase = "approach"
            u = _smoothstep((t - ends["settle"]) / p["approach_time"])
            positions = self.rest_positions * (1.0 - u) + self.hover_positions * u
            other = p["other_finger_fraction"] * u
            index = p["index_hook_fraction"] * u
        elif t < ends["descend"]:
            self.phase = "descend"
            u = _smoothstep((t - ends["approach"]) / p["descend_time"])
            positions = self.hover_positions * (1.0 - u) + self.grasp_positions * u
            other = p["other_finger_fraction"]
            index = p["index_hook_fraction"]
        elif t < ends["close"]:
            self.phase = "close"
            u = _smoothstep((t - ends["descend"]) / p["close_time"])
            positions = self.grasp_positions
            other = p["other_finger_fraction"]
            index = p["index_hook_fraction"] + (p["index_closed_fraction"] - p["index_hook_fraction"]) * u
            thumb = p["thumb_closed_fraction"] * u
        elif t < ends["cinch"]:
            self.phase = "cinch"
            if not self.attached:
                self._attach_handles()
            u = _smoothstep((t - ends["close"]) / p["cinch_time"])
            positions = self.grasp_positions * (1.0 - u) + self.cinch_positions * u
            other = p["other_finger_fraction"]
            index = p["index_closed_fraction"]
            thumb = p["thumb_closed_fraction"]
        elif t < ends["lift"]:
            self.phase = "lift"
            u = _smoothstep((t - ends["cinch"]) / p["lift_time"])
            positions = self.cinch_positions * (1.0 - u) + self.lift_positions * u
            other = p["other_finger_fraction"]
            index = p["index_closed_fraction"]
            thumb = p["thumb_closed_fraction"]
        elif t < ends["carry"]:
            self.phase = "carry"
            u = _smoothstep((t - ends["lift"]) / p["carry_time"])
            positions = self.lift_positions * (1.0 - u) + self.carry_positions * u
            other = p["other_finger_fraction"]
            index = p["index_closed_fraction"]
            thumb = p["thumb_closed_fraction"]
        else:
            self.phase = "hold"
            positions = self.carry_positions
            other = p["other_finger_fraction"]
            index = p["index_closed_fraction"]
            thumb = p["thumb_closed_fraction"]

        return positions, thumb, thumb, index, index, other

    def _build_trash(self, builder, seed):
        return _add_trash(builder, self.bag_info, self.params, seed)

    def _drive_robot_kinematic(self):
        """FK the IK joint solution into the robot body transforms and twists.

        Body twists ([v_com, omega], world frame) are finite-differenced from
        the FK transforms: the joint coordinate and DOF counts differ for the
        H1, so per-joint velocity mapping would be more work for no gain.
        """
        newton.eval_fk(self.model, self.ik_joint_q_flat, self._fk_joint_qd, self._fk_state)

        n = self.robot_body_count
        dt = self.frame_dt
        fk_body_q = self._fk_state.body_q.numpy()[:n]
        pos_new, quat_new = fk_body_q[:, :3], fk_body_q[:, 3:7]
        pos_old, quat_old = self._prev_fk_body_q[:, :3], self._prev_fk_body_q[:, 3:7]

        # omega from the relative quaternion q_rel = q_new * conj(q_old)
        x1, y1, z1, w1 = quat_new.T
        x2, y2, z2, w2 = (quat_old * np.array([-1.0, -1.0, -1.0, 1.0])).T
        q_rel = np.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            axis=1,
        )
        omega = (2.0 / dt) * q_rel[:, :3] * np.sign(q_rel[:, 3:4])

        # linear velocity of the COM
        def rotate(quat, vec):
            t = 2.0 * np.cross(quat[:, :3], vec)
            return vec + quat[:, 3:4] * t + np.cross(quat[:, :3], t)

        com = self._robot_body_com
        v_com = ((pos_new + rotate(quat_new, com)) - (pos_old + rotate(quat_old, com))) / dt

        self._prev_fk_body_q = fk_body_q.copy()
        for state in (self.state_0, self.state_1):
            body_q = state.body_q.numpy()
            body_qd = state.body_qd.numpy()
            body_q[:n] = fk_body_q
            body_qd[:n, :3] = v_com
            body_qd[:n, 3:] = omega
            state.body_q.assign(body_q)
            state.body_qd.assign(body_qd)

    def _attach_handles(self):
        """Re-parent both pinned handles from their world pins to the hand bodies."""
        body_q = self.state_0.body_q.numpy()
        pinned = self.handle_pinned_positions.numpy()
        left_count = self._left_driven_count
        locals_np = np.empty_like(pinned)
        for hand_slot, (start, count) in enumerate(((0, left_count), (left_count, len(pinned) - left_count))):
            hand_tf = wp.transform(*body_q[self.hand_bodies[hand_slot]])
            inv_tf = wp.transform_inverse(hand_tf)
            for k in range(start, start + count):
                locals_np[k] = wp.transform_point(inv_tf, wp.vec3(*pinned[k]))
        self.handle_local_positions.assign(locals_np)
        self.attached = True
        errors = []
        for hand_slot, centroid_key in enumerate(("left_centroid", "right_centroid")):
            hand_tf = wp.transform(*body_q[self.hand_bodies[hand_slot]])
            pinch = np.asarray(wp.transform_point(hand_tf, self.hand_offsets[hand_slot]))
            errors.append(float(np.linalg.norm(pinch - self.bag_info[centroid_key])))
        print(
            f"[trash_bag_h1] handles attached at t={self.sim_time:.2f}s  "
            f"pinch-to-handle error L={errors[0]:.3f}m R={errors[1]:.3f}m",
            flush=True,
        )

    # -------------------------------------------------------------- sim ---
    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                drive_handle_vertices,
                dim=self.handle_indices.shape[0],
                inputs=[
                    self.handle_indices,
                    self.handle_pinned_positions,
                    self.handle_local_positions,
                    self.handle_hand_body,
                    self.state_0.body_q,
                    1 if self.attached else 0,
                ],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _debug_rope(self):
        """Print rope stretch statistics and how much rope hangs below the rim."""
        if not hasattr(self, "_rope_edge_pairs"):
            faces = np.asarray(self.bag_info["rope_faces"], dtype=np.int64).reshape(-1, 3)
            edges = np.unique(
                np.sort(np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).reshape(-1, 2)), axis=0
            )
            rest_verts, _ = _load_obj(ROPE_OBJ)
            rest_verts = rest_verts * self.params["rope_rest_scale"]
            self._rope_edge_pairs = edges
            self._rope_edge_rest = np.linalg.norm(rest_verts[edges[:, 0]] - rest_verts[edges[:, 1]], axis=1)
        pq = self.state_0.particle_q.numpy()
        rope = pq[self.bag_info["rope_start"] : self.bag_info["rope_start"] + self.bag_info["rope_count"]]
        lengths = np.linalg.norm(rope[self._rope_edge_pairs[:, 0]] - rope[self._rope_edge_pairs[:, 1]], axis=1)
        strain = lengths / np.maximum(self._rope_edge_rest, 1e-9)
        rim_z = self.params["pedestal_top_z"] + self.params["can_height"]
        below = float(np.count_nonzero(rope[:, 2] < rim_z - 0.05)) / len(rope)
        print(
            f"[trash_bag_h1] rope strain mean={strain.mean():.3f} p99={np.percentile(strain, 99):.2f} "
            f"max={strain.max():.2f}  below-rim frac={below:.3f}",
            flush=True,
        )

    def _debug_tracking(self):
        """Print IK convergence vs AVBD tracking errors at the pinch points."""
        if self._ik_debug_state is None:
            self._ik_debug_state = self.model.state()
        newton.eval_fk(self.model, self.ik_joint_q_flat, self.model.joint_qd, self._ik_debug_state)
        ik_body_q = self._ik_debug_state.body_q.numpy()
        sim_body_q = self.state_0.body_q.numpy()
        rows = []
        for slot, (body, offset) in enumerate(zip(self.hand_bodies, self.hand_offsets, strict=True)):
            target = self.target_hand_positions[slot]
            ik_pinch = np.asarray(wp.transform_point(wp.transform(*ik_body_q[body]), offset))
            sim_pinch = np.asarray(wp.transform_point(wp.transform(*sim_body_q[body]), offset))
            rows.append(
                f"ik_err={np.linalg.norm(ik_pinch - target):.3f} sim_err={np.linalg.norm(sim_pinch - target):.3f}"
            )
        print(f"[trash_bag_h1] frame {self.frame}  phase {self.phase}  L({rows[0]})  R({rows[1]})", flush=True)

    def step(self):
        self._update_trajectory()
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self._debug and self.frame % 30 == 0:
            self._debug_tracking()
            self._debug_rope()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/model/bag",
            self.state_0.particle_q,
            self._bag_tri_indices,
            hidden=False,
            backface_culling=False,
            color=self.params["bag_color"],
            roughness=0.7,
        )
        self.viewer.log_mesh(
            "/model/rope",
            self.state_0.particle_q,
            self._rope_tri_indices,
            hidden=False,
            backface_culling=False,
            color=self.params["rope_color"],
            roughness=0.9,
        )
        self.viewer.end_frame()
        if self._video_writer is not None:
            frame = self.viewer.get_frame().numpy()
            self._video_writer.append_data(frame)
            if self._record_until and self.frame >= self._record_until:
                self._close_video()

    def gui(self, ui):
        ui.text(f"Phase: {self.phase}")
        ui.text(f"Attached: {self.attached}")

    # ------------------------------------------------------------- test ---
    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Cloth state contains non-finite values"
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"
        assert self.attached, "The H1 never grasped the drawstring handles"

        bag = particle_q[self.bag_info["bag_start"] : self.bag_info["bag_start"] + self.bag_info["bag_count"]]
        rim_z = self.params["pedestal_top_z"] + self.params["can_height"]
        # the bag was lifted: its lowest point cleared the can interior
        assert float(bag[:, 2].min()) > self.params["pedestal_top_z"] + 0.05, "Bag was not lifted off the can floor"

        # trash stays inside the lifted bag
        trash = body_q[self.trash_bodies][:, :3]
        bag_centroid = bag.mean(axis=0)
        assert np.all(np.linalg.norm(trash - bag_centroid, axis=1) < 0.6), "Trash escaped the bag"
        assert float(trash[:, 2].min()) > 0.2, "Trash fell to the ground"

        # hands track their targets
        hand_errors = []
        for body, offset, target in zip(self.hand_bodies, self.hand_offsets, self.target_hand_positions, strict=True):
            pinch = np.asarray(wp.transform_point(wp.transform(*body_q[body]), offset))
            hand_errors.append(float(np.linalg.norm(pinch - target)))
        assert max(hand_errors) < 0.15, f"H1 hand tracking error is too large: {max(hand_errors):.3f} m"
        _ = rim_z

    @classmethod
    def create_parser(cls):
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=cls.DEFAULT_PARAMS["seed"])
        parser.add_argument(
            "--record-video",
            type=str,
            default=None,
            help="Record an mp4 of the rendered frames to this path (gl viewer).",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Print hand-tracking and rope-strain statistics every 30 frames.",
        )
        total_time = sum(cls.DEFAULT_PARAMS[f"{name}_time"] for name in cls.PHASE_NAMES)
        parser.set_defaults(num_frames=int(total_time * cls.DEFAULT_PARAMS["fps"]))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
