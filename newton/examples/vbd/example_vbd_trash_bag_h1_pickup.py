# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Trash Bag H1 Pickup
#
# The standalone drawstring trash-bag demo (example_vbd_trash_bag.py)
# moved onto the dish-washing scene's table, with the fixed-base Unitree
# H1 standing in front of it. The bag, drawstring, trash can, and spheres
# are IDENTICAL to the standalone demo — same assets, same rest scale,
# same cloth materials, same contact and solver settings, same solve (10
# substeps x 12 iterations, self-contact, water-tight rigid-soft SDF
# contacts) — rotated so the drawstring handles hang at the robot's left
# and right, and translated onto the tabletop.
#
# The demo starts from a BAKED initial state: `--bake` settles the bag
# (with 2 spheres inside) for 10 simulated seconds and saves the cloth and
# sphere state to asset/trash_bag_pickup_init_state.npz. Normal runs load
# that file as the INITIAL state (the cloth REST shapes are untouched) —
# no long settle at startup — and drop one fresh sphere and one cube into
# the open bag.
#
# Commands:
#   python -m newton.examples vbd_trash_bag_h1_pickup --bake   (once)
#   python -m newton.examples vbd_trash_bag_h1_pickup
#
###########################################################################

from __future__ import annotations

import json
import math
import os
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset")
BAG_OBJ = os.path.join(ASSET, "trash_bag.obj")
ROPE_OBJ = os.path.join(ASSET, "trash_bag_rope.obj")
# Round INITIAL positions (same topology as the flat rest OBJs): the flat bag
# unflattened onto a cylinder so it starts round, lining a round bin, while its
# rest shape stays the flat pressed tube.
BAG_INIT_OBJ = os.path.join(ASSET, "trash_bag_init.obj")
ROPE_INIT_OBJ = os.path.join(ASSET, "trash_bag_rope_init.obj")
LAYOUT_JSON = os.path.join(ASSET, "trash_bag_layout.json")
# baked initial state (written by --bake): settled cloth + sphere poses
STATE_NPZ = os.path.join(ASSET, "trash_bag_pickup_init_state.npz")

PARAMS = {
    # --- apples (rigid spheres), identical to example_vbd_trash_bag.py
    # except the count: only 2 are baked into the settled initial state ---
    "enable_apples": True,
    "num_apples": 2,
    "apple_radius": 0.034,
    "apple_margin": 0.005,
    "apple_density": 1000.0,
    "apple_ke": 5.0e5,
    "apple_kd": 5.0e1,
    "apple_mu": 0.5,
    # --- round trash can, identical to the standalone demo, placed ON the
    # table at (can_x, can_y): as close to the robot as the can fits with
    # its rim inside the tabletop (front edge at x = -0.20)
    "can_x": -0.05,
    "can_y": 0.0,
    "can_bottom_radius": 0.12,
    "can_top_radius": 0.14,
    "can_height": 0.31,
    "can_z_bottom": -0.01,  # outer floor just below the bag bottom, both offset onto the tabletop
    "can_wall_thickness": 0.0025,
    "can_floor_thickness": 0.004,
    "can_ke": 5.0e5,
    "can_kd": 5.0e1,
    "can_mu": 0.4,
    "can_margin": 0.002,
    "can_n_around": 72,
    "can_n_rows": 28,
    # --- bag cloth (floppy plastic), identical to the standalone demo ---
    "bag_rest_scale": 1.5,  # rest shape (bag AND rope) this much bigger than the round init -> expands to fill the can
    "particle_radius": 0.004,
    "cloth_density": 0.08,
    "cloth_tri_ke": 1.0e5,
    "cloth_tri_ka": 5.0e4,
    "cloth_tri_kd": 1.0e1,
    "cloth_edge_ke": 0.2,  # low bending -> floppy, wrinkly plastic
    "cloth_edge_kd": 0.1,
    # --- rope cloth (the tie): stiff/inextensible so pulling collapses the loop ---
    "rope_density": 0.008,
    "rope_tri_ke": 2.0e5,
    "rope_tri_ka": 2.0e5,
    "rope_tri_kd": 1.0e2,
    "rope_edge_ke": 0.05,
    "rope_edge_kd": 0.01,
    # --- springs ---
    "closure_ke": 2.0e4,  # tunnel closure: flap free edge <-> wall
    "closure_kd": 1.0e-3,
    "closure_rest_length": 0.0,
    # --- contacts, identical to the standalone demo ---
    "soft_contact_ke": 2.0e5,
    "soft_contact_kd": 2.0e1,
    "soft_contact_mu": 0.3,
    "soft_contact_creation_margin": 0.012,
    "particle_self_contact_radius": 0.003,
    "particle_self_contact_margin": 0.006,
    "rigid_body_particle_contact_buffer_size": 16384,
    "rigid_body_contact_buffer_size": 1024,
    "rigid_contact_hard": True,
    "enable_water_tight": True,  # water-tight rigid-soft SDF contacts (no tunneling through the thin can)
    # --- objects dropped into the bag when starting from the baked state ---
    "drop_sphere_radius": 0.034,
    "drop_cube_half_extent": 0.030,
    "drop_height": 0.55,  # sphere spawn height above the bag bottom [m, assembly-local]
    "drop_stagger": 0.10,  # the cube spawns this much higher -> they land one after the other
    # --- solver / time, identical to the standalone demo ---
    "fps": 60,
    "sim_substeps": 10,
    "solver_iterations": 12,
    "gravity": -9.8,
    "bake_seconds": 10.0,
    "num_frames": 300,
    # --- table (reused unchanged from the dish-washing scene; the front
    # edge faces the robot at x = -table_half_width) ---
    "table_half_width": 0.20,
    "table_half_depth": 0.36,
    "table_top_z": 1.09,
    "tabletop_half_height": 0.04,
    "table_mu": 0.9,
    "shape_ke": 1.0e3,
    "shape_kd": 1.0e-4,
    "rigid_contact_gap": 0.001,
    # --- H1 (reused unchanged from the dish-washing scene); it stands at
    # the table holding its initial pose through the AVBD joint drives ---
    "robot_base_x": -0.70,
    "robot_contact_ke": 1.0e3,
    "robot_contact_kd": 1.0e-2,
    "robot_contact_mu": 0.5,
    "robot_contact_margin": 0.002,
    "robot_sdf_padding": 0.012,
    "robot_sdf_max_resolution": 64,
    # the palm + index..pinky chains keep particle collision with a finer
    # SDF: the later manipulation steps hook the drawstring with these four
    # fingers (no thumb). Contact ke matches soft_contact_ke so the averaged
    # body-particle contact stays stiff for the cloth.
    "finger_contact_ke": 2.0e5,
    "finger_contact_kd": 2.0e1,
    "finger_contact_mu": 200.0,
    "finger_contact_margin": 0.002,
    "finger_sdf_padding": 0.012,
    "finger_sdf_max_resolution": 128,
    # AVBD joint drives (dish-washing values)
    "joint_drive_ke": 5.0e4,
    "joint_drive_kd": 5.0e2,
    "torso_drive_ke": 2.0e5,
    "torso_drive_kd": 2.0e3,
    "finger_drive_ke": 2.0e4,
    "finger_drive_kd": 1.0e2,
    # --- presentation (the dish-washing scene's camera) ---
    "camera_position": (1.43, 0.42, 1.77),
    "camera_pitch": -15.9,
    "camera_yaw": -160.8,
    "camera_fov": 45.0,
    "enable_cuda_graph": True,
    "seed": 42,
}


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
    # The asset generator emits front tunnel pairs first, then back tunnel pairs.
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
    """Thin, WATERTIGHT truncated-cone bin (open top) the bag sits INSIDE.

    Identical to example_vbd_trash_bag.py: a closed cup cross-section
    revolved around the z-axis, so a bag particle in the cavity stays on the
    cavity side and cannot cross the thin wall.
    """
    z_top = z_bottom + height
    z_floor_top = z_bottom + floor_thickness
    r_bot_out = bottom_radius + wall_thickness
    r_top_out = top_radius + wall_thickness

    # closed cross-section profile in (r, z); r==0 marks an on-axis center vertex.
    profile = [(0.0, z_bottom), (r_bot_out, z_bottom)]
    for j in range(1, n_rows + 1):  # outer wall, bottom -> rim
        u = j / n_rows
        profile.append((r_bot_out + u * (r_top_out - r_bot_out), z_bottom + u * height))
    profile.append((top_radius, z_top))  # inner rim
    for j in range(1, n_rows + 1):  # inner wall, rim -> floor
        u = j / n_rows
        profile.append((top_radius + u * (bottom_radius - top_radius), z_top + u * (z_floor_top - z_top)))
    profile.append((0.0, z_floor_top))  # inner floor center

    verts = []
    rings = []  # ('ring', [indices]) per profile point, or ('center', index)
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
    for i in range(len(profile) - 1):  # no wrap: the two centers bound the solid, no axis face
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
        else:  # ring -> center
            ra, c = a[1], b[1]
            for k in range(n_around):
                faces.append([ra[k], c, ra[(k + 1) % n_around]])
    # reverse winding so the shell's triangles face outward
    faces = np.array(faces, dtype=np.int32)[:, ::-1]
    return np.array(verts, dtype=np.float32), faces.reshape(-1)


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [i for i, label in enumerate(labels) if label.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def _add_h1(builder: newton.ModelBuilder, params: dict) -> tuple[dict[str, int], list[int]]:
    """Add the fixed-base H1 exactly as in the dish-washing demo. The palm
    and index..pinky finger chains keep particle collision (the later steps
    hook the drawstring with these four fingers); everything else is a plain
    rigid collider, filtered against the furniture in build_model."""
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

    shape_collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_collision_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    collision_mask = shape_collision_flag | particle_collision_flag
    hook_tokens = tuple(f"/{side}_{digit}_" for side in ("L", "R") for digit in ("index", "middle", "ring", "pinky"))
    hook_bodies = {
        body
        for body in range(robot_body_start, robot_body_end)
        if any(token in builder.body_label[body] for token in hook_tokens)
        or builder.body_label[body].endswith(("left_hand_link", "right_hand_link"))
    }
    hook_shape_count = 0
    robot_rigid_shapes = []
    for shape in range(robot_shape_start, robot_shape_end):
        original_flags = int(builder.shape_flags[shape])
        is_rigid_collider = bool(original_flags & shape_collision_flag)
        is_hook_collider = is_rigid_collider and builder.shape_body[shape] in hook_bodies
        builder.shape_flags[shape] &= ~collision_mask
        if is_rigid_collider:
            builder.shape_flags[shape] |= shape_collision_flag
            builder.shape_gap[shape] = params["rigid_contact_gap"]
            builder.shape_material_ke[shape] = params["robot_contact_ke"]
            builder.shape_material_kd[shape] = params["robot_contact_kd"]
            builder.shape_material_mu[shape] = params["robot_contact_mu"]
            builder.shape_margin[shape] = params["robot_contact_margin"]
            builder.shape_sdf_padding[shape] = params["robot_sdf_padding"]
            builder.shape_sdf_max_resolution[shape] = params["robot_sdf_max_resolution"]
            builder.shape_sdf_target_voxel_size[shape] = None
            robot_rigid_shapes.append(shape)
        if is_hook_collider:
            builder.shape_flags[shape] |= particle_collision_flag
            builder.shape_material_ke[shape] = params["finger_contact_ke"]
            builder.shape_material_kd[shape] = params["finger_contact_kd"]
            builder.shape_material_mu[shape] = params["finger_contact_mu"]
            builder.shape_margin[shape] = params["finger_contact_margin"]
            builder.shape_sdf_padding[shape] = params["finger_sdf_padding"]
            builder.shape_sdf_max_resolution[shape] = params["finger_sdf_max_resolution"]
            builder.shape_sdf_target_voxel_size[shape] = None
            hook_shape_count += 1

    # 2 palm geoms + 4 two-link finger chains per hand = 10 colliders per side
    if hook_shape_count != 20:
        raise RuntimeError(f"Expected 20 H1 palm/index..pinky colliders, found {hook_shape_count}")
    return body_indices, robot_rigid_shapes


def _add_table(builder: newton.ModelBuilder, params: dict) -> list[int]:
    """The dish-washing demo's table, unchanged."""
    table_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["table_mu"],
        gap=params["rigid_contact_gap"],
        has_particle_collision=True,
    )
    top_z = params["table_top_z"]
    half_height = params["tabletop_half_height"]
    wood = wp.vec3(0.46, 0.24, 0.10)
    shapes = [
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, top_z - half_height), wp.quat_identity()),
            hx=params["table_half_width"],
            hy=params["table_half_depth"],
            hz=half_height,
            cfg=table_cfg,
            color=wood,
        )
    ]

    leg_half_width = 0.03
    leg_half_height = 0.5 * (top_z - 2.0 * half_height)
    for x_sign, y_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        shapes.append(
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    wp.vec3(
                        x_sign * (params["table_half_width"] - 0.05),
                        y_sign * (params["table_half_depth"] - 0.06),
                        leg_half_height,
                    ),
                    wp.quat_identity(),
                ),
                hx=leg_half_width,
                hy=leg_half_width,
                hz=leg_half_height,
                cfg=table_cfg,
                color=wood,
            )
        )
    return shapes


def build_model(builder, params, seed, baked_state=None):
    """The standalone demo's build_model, with everything translated by
    ``offset`` onto the tabletop, plus the table, robot, and ground.

    When ``baked_state`` (a loaded STATE_NPZ) is given, the cloth INITIAL
    positions and the sphere poses come from the baked settle instead of
    the raw assets — the cloth REST shapes are unchanged — and one fresh
    sphere plus one cube are spawned above the open bag to drop in."""
    rng = np.random.default_rng(seed)
    with open(LAYOUT_JSON, encoding="utf-8") as file:
        layout = json.load(file)
    pr = params["particle_radius"]

    # the whole can+bag+apples assembly, translated onto the tabletop: the
    # can's outer floor (local z = can_z_bottom) rests on the table surface
    offset = np.asarray(
        [params["can_x"], params["can_y"], params["table_top_z"] + 0.002 - params["can_z_bottom"]],
        dtype=np.float64,
    )

    # ... and rotated about the can axis so the two drawstring handles exit
    # at the robot's left (+y) and right (-y). The handles leave the asset
    # on a diagonal, so the exact angle comes from the asset itself.
    bag_init_verts, _ = _load_obj(BAG_INIT_OBJ)
    rope_init_verts, _ = _load_obj(ROPE_INIT_OBJ)
    left_handle_centroid = rope_init_verts[
        np.asarray(layout["rope"]["handle_vertex_indices"]["left"], dtype=np.int32)
    ].mean(axis=0)
    theta = -0.5 * math.pi - math.atan2(float(left_handle_centroid[1]), float(left_handle_centroid[0]))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def to_world(x, y, z):
        return wp.vec3(
            float(cos_t * x - sin_t * y + offset[0]),
            float(sin_t * x + cos_t * y + offset[1]),
            float(z + offset[2]),
        )

    # --- robot + table first (rigid scene) ---
    robot_bodies, robot_rigid_shapes = _add_h1(builder, params)
    robot_coord_count = builder.joint_coord_count
    table_shapes = _add_table(builder, params)

    # --- bag shell ---
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
    # Rest state was built from the flat OBJ above; override only the INITIAL
    # positions: the baked settled shape when available, else the round
    # (unflattened) shape so the bag starts in the can.
    assert len(bag_init_verts) == len(bag_verts), "bag init/rest vertex count mismatch"
    if baked_state is not None:
        baked_bag = baked_state["particle_q"][bag_start : bag_start + len(bag_verts)]
        for i, (x, y, z) in enumerate(baked_bag):
            builder.particle_q[bag_start + i] = wp.vec3(float(x), float(y), float(z))
    else:
        for i, (x, y, z) in enumerate(bag_init_verts):
            builder.particle_q[bag_start + i] = to_world(float(x), float(y), float(z))

    # --- drawstring tie (ribbon) ---
    rope_verts, rope_faces = _load_obj(ROPE_OBJ)
    rope_faces_array = np.array(rope_faces, dtype=np.int32).reshape(-1, 3)
    rope_start = len(builder.particle_q)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=params["bag_rest_scale"],  # enlarge the rope REST with the bag so it doesn't choke expansion
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
    # override rope initial positions (baked settled shape when available)
    assert len(rope_init_verts) == len(rope_verts), "rope init/rest vertex count mismatch"
    if baked_state is not None:
        baked_rope = baked_state["particle_q"][rope_start : rope_start + len(rope_verts)]
        for i, (x, y, z) in enumerate(baked_rope):
            builder.particle_q[rope_start + i] = wp.vec3(float(x), float(y), float(z))
    else:
        for i, (x, y, z) in enumerate(rope_init_verts):
            builder.particle_q[rope_start + i] = to_world(float(x), float(y), float(z))

    # --- tunnel-closure springs: flap free edge <-> wall (bag-local indices) ---
    tunnel_spring_pairs = np.array(
        [[bag_start + i, bag_start + j] for i, j in layout["tunnel_spring_pairs"]], dtype=np.int32
    )
    for i, j in tunnel_spring_pairs:
        builder.add_spring(int(i), int(j), params["closure_ke"], params["closure_kd"], 0.0)
        builder.spring_rest_length[-1] = params["closure_rest_length"]

    # --- drawstring handles (the two exposed side ends; free, no pinning) ---
    hv = layout["rope"]["handle_vertex_indices"]
    right_idx = np.array([rope_start + i for i in hv["right"]], dtype=np.int32)
    left_idx = np.array([rope_start + i for i in hv["left"]], dtype=np.int32)

    # --- static round trash can (rigid container) on the tabletop ---
    can_cfg = newton.ModelBuilder.ShapeConfig()
    can_cfg.ke = params["can_ke"]
    can_cfg.kd = params["can_kd"]
    can_cfg.mu = params["can_mu"]
    can_cfg.has_particle_collision = True
    can_cfg.margin = params["can_margin"]
    can_v, can_f = build_can_mesh(
        params["can_bottom_radius"],
        params["can_top_radius"],
        params["can_z_bottom"],
        params["can_height"],
        params["can_wall_thickness"],
        params["can_floor_thickness"],
        params["can_n_around"],
        params["can_n_rows"],
    )
    can_shape = builder.add_shape_mesh(
        -1,
        xform=wp.transform(wp.vec3(*(float(c) for c in offset)), wp.quat_identity()),
        mesh=newton.Mesh(can_v, can_f),
        cfg=can_cfg,
        label="trash_can",
    )

    # --- rigid apples dropped into the round bag (standalone placement + offset) ---
    r = params["apple_radius"]
    z_floor = float(bag_init_verts[:, 2].min())
    mid = bag_init_verts[(bag_init_verts[:, 2] > 0.05) & (bag_init_verts[:, 2] < 0.30)]
    bag_r = float(np.median(np.hypot(mid[:, 0], mid[:, 1])))
    rad_in = max(0.0, bag_r - r - 0.015)  # keep apples inside the round wall
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["apple_density"]
    cfg.ke = params["apple_ke"]
    cfg.kd = params["apple_kd"]
    cfg.mu = params["apple_mu"]
    cfg.has_particle_collision = True
    cfg.margin = params["apple_margin"]

    body_indices = []
    apple_shapes = []
    if baked_state is not None:
        # respawn the settled spheres exactly where the bake left them
        for i, xform in enumerate(np.asarray(baked_state["apple_body_q"], dtype=np.float64)):
            body = builder.add_body(xform=wp.transform(wp.vec3(*xform[:3]), wp.quat(*xform[3:7])), label=f"apple_{i}")
            body_indices.append(body)
            apple_shapes.append(builder.add_shape_sphere(body, radius=r, cfg=cfg))
    else:
        n = params["num_apples"] if params["enable_apples"] else 0
        for i in range(n):
            ang = i * 2.39996  # golden angle -> even spread across the round mouth
            rr = rad_in * math.sqrt((i + 0.5) / n)  # spiral fill within the round radius
            px = float(rr * math.cos(ang) + rng.uniform(-0.004, 0.004))
            py = float(rr * math.sin(ang) + rng.uniform(-0.004, 0.004))
            pz = z_floor + r + 0.05 + i * 0.06  # stacked so they drop in one by one
            body = builder.add_body(xform=wp.transform(to_world(px, py, pz), wp.quat_identity()), label=f"apple_{i}")
            body_indices.append(body)
            apple_shapes.append(builder.add_shape_sphere(body, radius=r, cfg=cfg))

    # --- fresh trash dropped into the settled bag: one sphere + one cube ---
    drop_bodies = []
    drop_shapes = []
    if baked_state is not None:
        drop_r = params["drop_sphere_radius"]
        sphere_body = builder.add_body(
            xform=wp.transform(to_world(0.02, 0.015, z_floor + params["drop_height"]), wp.quat_identity()),
            label="drop_sphere",
        )
        drop_bodies.append(sphere_body)
        drop_shapes.append(builder.add_shape_sphere(sphere_body, radius=drop_r, cfg=cfg))

        he = params["drop_cube_half_extent"]
        # slight tilt so the cube tumbles naturally instead of landing flat
        cube_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 0.3)
        cube_body = builder.add_body(
            xform=wp.transform(
                to_world(-0.02, -0.015, z_floor + params["drop_height"] + params["drop_stagger"]), cube_rot
            ),
            label="drop_cube",
        )
        drop_bodies.append(cube_body)
        drop_shapes.append(builder.add_shape_box(cube_body, hx=he, hy=he, hz=he, cfg=cfg))

    # --- ground the robot stands on ---
    ground_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["table_mu"],
        gap=params["rigid_contact_gap"],
    )
    ground_shape = builder.add_ground_plane(cfg=ground_cfg)

    # the robot only exists alongside the furniture for now: filter every
    # robot collider against the table, can, apples, and ground so a limb
    # brushing static geometry cannot disturb the AVBD solve (the cloth is
    # untouched — particle contact stays on the hook fingers for later steps)
    for robot_shape in robot_rigid_shapes:
        for other in (*table_shapes, can_shape, ground_shape, *apple_shapes, *drop_shapes):
            builder.add_shape_collision_filter_pair(robot_shape, other)

    builder.color(include_bending=True)

    return {
        "bag_start": bag_start,
        "rope_start": rope_start,
        "bag_count": rope_start - bag_start,
        "rope_count": len(rope_verts),
        "bag_faces": bag_faces_array,
        "rope_faces": rope_faces_array,
        "right_idx": right_idx,
        "left_idx": left_idx,
        "body_indices": body_indices,
        "drop_bodies": drop_bodies,
        "z_floor": z_floor + offset[2],
        "num_tunnel_springs": len(layout["tunnel_spring_pairs"]),
        "tunnel_spring_pairs": tunnel_spring_pairs,
        "robot_bodies": robot_bodies,
        "robot_coord_count": robot_coord_count,
        "can_center": np.asarray([params["can_x"], params["can_y"]], dtype=np.float64),
    }


def setup_sim(builder, info, params):
    builder.enable_rigid_mesh_sdfs()
    model = builder.finalize()
    model.soft_contact_ke = params["soft_contact_ke"]
    model.soft_contact_kd = params["soft_contact_kd"]
    model.soft_contact_mu = params["soft_contact_mu"]

    vertex_filter, edge_filter = _build_tunnel_seam_contact_filters(model, info["tunnel_spring_pairs"])

    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=params["solver_iterations"],
        rigid_body_particle_contact_buffer_size=params["rigid_body_particle_contact_buffer_size"],
        rigid_body_contact_buffer_size=params["rigid_body_contact_buffer_size"],
        particle_enable_self_contact=True,
        particle_self_contact_radius=params["particle_self_contact_radius"],
        particle_self_contact_margin=params["particle_self_contact_margin"],
        particle_external_vertex_contact_filtering_map=vertex_filter,
        particle_external_edge_contact_filtering_map=edge_filter,
        rigid_contact_hard=params["rigid_contact_hard"],
        # articulated H1 in the same AVBD solve (dish-washing values)
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e6,
        rigid_joint_linear_kd=1.0e2,
        rigid_joint_angular_kd=1.0e2,
    )
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=params["soft_contact_creation_margin"],
        enable_water_tight_rigid_soft_contact=params["enable_water_tight"],
    )
    return model, solver, pipeline


class Example:
    def __init__(self, viewer, args, params: dict | None = None):
        self.viewer = viewer
        self.params = dict(PARAMS) if params is None else params
        p = self.params
        self.sim_time = 0.0
        self.fps = p["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = p["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0

        seed = getattr(args, "seed", p["seed"]) if args is not None else p["seed"]
        self.bake = bool(getattr(args, "bake", False)) if args is not None else False
        baked_state = None
        if not self.bake:
            if os.path.exists(STATE_NPZ):
                baked_state = np.load(STATE_NPZ)
                print(f"[trash_bag_h1_pickup] starting from baked state {STATE_NPZ}")
            else:
                print(
                    f"[trash_bag_h1_pickup] no baked state at {STATE_NPZ} — starting from the raw "
                    "assets (run once with --bake to settle and save the initial state)"
                )
        builder = newton.ModelBuilder(gravity=p["gravity"])
        self.info = build_model(builder, p, seed=seed, baked_state=baked_state)
        self.model, self.solver, self.pipeline = setup_sim(builder, self.info, p)
        self.device = self.model.device
        self.robot_coord_count = self.info["robot_coord_count"]

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)
        # the H1 holds its initial pose: drive targets = initial joint state
        wp.copy(self.control.joint_target_q, self.model.joint_q, count=self.robot_coord_count)
        self.control.joint_target_qd.zero_()
        self.torso_initial_position = self.state_0.body_q.numpy()[self.info["robot_bodies"]["torso"], :3].copy()

        print(
            f"[trash_bag_h1_pickup] bag verts {self.info['bag_count']}  rope verts {self.info['rope_count']}  "
            f"tunnel springs {self.info['num_tunnel_springs']}  apples {len(self.info['body_indices'])}"
        )

        self._capture_graph()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            wp.vec3(*p["camera_position"]),
            p["camera_pitch"],
            p["camera_yaw"],
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = p["camera_fov"]

    def _capture_graph(self):
        self.graph = None
        if not self.params["enable_cuda_graph"] or not wp.get_device().is_cuda:
            return
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        # even substep count: the state_0/state_1 swap returns to the original
        # buffers, so the captured graph replays against stable pointers
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def save_baked_state(self):
        """Save the settled cloth positions and sphere poses as the demo's
        initial state (the cloth rest shapes are not touched)."""
        particle_q = self.state_0.particle_q.numpy()
        apple_body_q = self.state_0.body_q.numpy()[self.info["body_indices"]]
        np.savez(
            STATE_NPZ,
            particle_q=particle_q,
            apple_body_q=apple_body_q,
            can_x=self.params["can_x"],
            can_y=self.params["can_y"],
            table_top_z=self.params["table_top_z"],
        )
        print(f"[trash_bag_h1_pickup] baked state saved to {STATE_NPZ}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p = self.params
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Cloth positions contain non-finite values"
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"
        # the bag stays in the can on the table
        bag = particle_q[self.info["bag_start"] : self.info["bag_start"] + self.info["bag_count"]]
        assert bag[:, 2].min() > p["table_top_z"] - 0.05, "The bag fell off the table"
        # apples and dropped objects stay roughly within the bag (no escape / explosion)
        trash_bodies = [*self.info["body_indices"], *self.info["drop_bodies"]]
        if trash_bodies:
            trash_pos = body_q[trash_bodies][:, :3]
            assert np.all(np.abs(trash_pos[:, 0] - p["can_x"]) < 0.4), "A trash object escaped in x"
            assert np.all(np.abs(trash_pos[:, 1] - p["can_y"]) < 0.3), "A trash object escaped in y"
        # the dropped objects actually landed inside the can (below the rim)
        rim_z = p["table_top_z"] + 0.002 + p["can_height"]
        for body in self.info["drop_bodies"]:
            assert body_q[body, 2] < rim_z + 0.05, f"A dropped object did not land in the bag: z={body_q[body, 2]:.3f}"
        # the robot held its pose
        torso = body_q[self.info["robot_bodies"]["torso"], :3]
        torso_err = float(np.linalg.norm(torso - self.torso_initial_position))
        assert torso_err < 0.01, f"H1 torso moved {torso_err:.3f} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=PARAMS["seed"])
        parser.add_argument(
            "--bake",
            action="store_true",
            default=False,
            help="Settle the bag (with the baked-in spheres) for bake_seconds and save the initial state.",
        )
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.bake:
        bake_frames = int(PARAMS["bake_seconds"] * example.fps)
        print(f"[trash_bag_h1_pickup] baking: settling for {bake_frames} frames ...")
        for frame in range(bake_frames):
            example.step()
            if (frame + 1) % 120 == 0:
                print(f"[trash_bag_h1_pickup]   bake frame {frame + 1}/{bake_frames}")
        example.save_baked_state()
    else:
        newton.examples.run(example, args)
