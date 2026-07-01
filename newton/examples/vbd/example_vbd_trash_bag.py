# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Trash Bag
#
# Rigid "apples" (spheres) are dropped into a procedural drawstring trash
# bag and the bag is cinched shut by its drawstring.  Two thin-shell cloth
# meshes are loaded from vbd/asset/:
#   - trash_bag.obj      the bag shell, whose top hem folds OUTWARD into two
#                        tunnels (the colored "stripe" band)
#   - trash_bag_rope.obj the drawstring "tie", a single-layer cloth ribbon
#                        that follows the bag contour through both tunnels and
#                        out the four side holes as two handles
# trash_bag_layout.json supplies the 38 tunnel-closure spring pairs (flap edge
# <-> wall, keeping each tunnel shut around the rope while the mesh stays
# manifold) and the rope handle vertex indices.
#
# The two drawstring handles (the exposed left/right rope ends) are pinned.
# After the apples settle, the handles are pulled UP and APART: the nearly
# inextensible rope loop collapses in the perpendicular (depth) direction,
# gathering the mouth shut around the apples -- the "rope to tie".
#
# Contact / cloth parameters follow example_vbd_apple_bag.py.
#
# Commands:
#   python newton/examples/vbd/example_vbd_trash_bag.py --num-frames 360
#   python newton/examples/vbd/example_vbd_trash_bag.py --viewer null --num-frames 360 --test
#   python newton/examples/vbd/example_vbd_trash_bag.py --num-frames 360 --export-ply --ply-output-dir trash_bag_ply
###########################################################################

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

ASSET = os.path.join(os.path.dirname(__file__), "asset")
BAG_OBJ = os.path.join(ASSET, "trash_bag.obj")
ROPE_OBJ = os.path.join(ASSET, "trash_bag_rope.obj")
# Round INITIAL positions (same topology as the flat rest OBJs): the flat bag
# unflattened onto a cylinder so it starts round, lining a round bin, while its
# rest shape stays the flat pressed tube.
BAG_INIT_OBJ = os.path.join(ASSET, "trash_bag_init.obj")
ROPE_INIT_OBJ = os.path.join(ASSET, "trash_bag_rope_init.obj")
LAYOUT_JSON = os.path.join(ASSET, "trash_bag_layout.json")

PARAMS = {
    # --- apples (rigid spheres) ---
    "enable_apples": False,  # set False to drop no rigid spheres (bag/can only)
    "num_apples": 5,
    "apple_radius": 0.034,
    "apple_margin": 0.005,
    "apple_density": 1000.0,
    "apple_ke": 5.0e5,
    "apple_kd": 5.0e1,
    "apple_mu": 0.5,
    # --- round trash can: thin watertight truncated-cone bin the bag sits INSIDE ---
    "can_bottom_radius": 0.12,  # inner radius at the base (just > bag cylinder r ~0.113)
    "can_top_radius": 0.14,  # inner radius at the rim (flared -> truncated cone)
    "can_height": 0.31,  # shorter than the bag (H=0.40) so the bag stands above the rim
    "can_z_bottom": -0.01,  # outer floor just below the bag bottom (z=0)
    "can_wall_thickness": 0.0025,  # thin volumetric wall (closed shell)
    "can_floor_thickness": 0.004,
    "can_ke": 5.0e5,
    "can_kd": 5.0e1,
    "can_mu": 0.4,
    "can_margin": 0.002,  # contact margin for the can
    "can_n_around": 72,
    "can_n_rows": 28,
    # --- bag cloth (floppy plastic) ---
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
    # --- contacts ---
    "soft_contact_ke": 1.0e5,
    "soft_contact_kd": 1.0e1,
    "soft_contact_mu": 0.3,
    "soft_contact_creation_margin": 0.012,
    "particle_self_contact_radius": 0.004,
    "particle_self_contact_margin": 0.008,
    "rigid_body_particle_contact_buffer_size": 16384,
    "rigid_body_contact_buffer_size": 1024,
    "rigid_contact_hard": True,
    "enable_water_tight": True,  # water-tight rigid-soft SDF edge/face contacts (no tunneling through the thin can)
    # --- solver / time ---
    "fps": 60,
    "sim_substeps": 10,
    "solver_iterations": 12,
    "gravity": -9.8,
    "vertical_axis": 2,
    "preroll_frames": 0,
    # --- pin + cinch ---
    "pin_handles": False,  # unpinned for now: handles are free, no cinch drive
    "settle_frames": 150,
    "cinch_frames": 200,
    "cinch_up": 0.26,  # how far the handles rise [m]
    "cinch_together": 0.13,  # how far each handle pulls INWARD in x [m] (gathers the neck shut)
    "cinch_ramp": 1.4,  # ease the cinch in over this many seconds
    # --- view (fixed for the whole clip; framed for full motion z in [0, ~0.62]) ---
    "camera_pos": (0.52, -0.98, 0.58),
    "camera_target": (0.0, 0.0, 0.3),
    "camera_fov": 45.0,
    "draw_wireframe": True,
    "initial_paused": True,
    "enable_cuda_graph": True,
    "seed": 42,
}


def _pitch_yaw(pos, target):
    d = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-9
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(d[2])))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return pitch, yaw


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


def _combined_cloth_mesh(positions, info):
    positions = np.asarray(positions, dtype=np.float32)

    bag_start = int(info["bag_start"])
    bag_count = int(info["bag_count"])
    rope_start = int(info["rope_start"])
    rope_count = int(info["rope_count"])

    bag_vertices = positions[bag_start : bag_start + bag_count]
    rope_vertices = positions[rope_start : rope_start + rope_count]
    vertices = np.vstack((bag_vertices, rope_vertices))

    bag_faces = np.asarray(info["bag_faces"], dtype=np.int32).reshape(-1, 3)
    rope_faces = np.asarray(info["rope_faces"], dtype=np.int32).reshape(-1, 3) + bag_count
    faces = np.vstack((bag_faces, rope_faces))
    return vertices, faces


def _write_ply(path, vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write(f"element face {len(faces)}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")
        for vertex in vertices:
            file.write(f"{float(vertex[0]):.9g} {float(vertex[1]):.9g} {float(vertex[2]):.9g}\n")
        for face in faces:
            file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _export_ply_frame(output_dir, frame, positions, info):
    os.makedirs(output_dir, exist_ok=True)
    vertices, faces = _combined_cloth_mesh(positions, info)
    path = os.path.join(output_dir, f"trash_bag_{frame:06d}.ply")
    _write_ply(path, vertices, faces)
    return path


@wp.kernel
def move_pinned_vertices(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    offset: wp.array[wp.vec3],
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    tid = wp.tid()
    vi = pinned_indices[tid]
    new_p = original_positions[tid] + offset[0]
    pos_0[vi] = new_p
    pos_1[vi] = new_p


def build_can_mesh(bottom_radius, top_radius, z_bottom, height, wall_thickness, floor_thickness, n_around, n_rows):
    """Thin, WATERTIGHT truncated-cone bin (open top) the bag sits INSIDE.

    Revolves a closed cup cross-section (outer floor -> outer wall -> top rim ->
    inner wall -> inner floor) around the z-axis. Because the shell is a closed
    solid with a thin wall, its signed distance is negative only inside the wall
    material; a bag particle in the cavity stays on the cavity side and cannot
    cross the wall -> the bag is contained regardless of triangle facing.
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
    # reverse winding so the shell's triangles face outward (was rendering inverted)
    faces = np.array(faces, dtype=np.int32)[:, ::-1]
    return np.array(verts, dtype=np.float32), faces.reshape(-1)


def build_model(builder, params, seed):
    rng = np.random.default_rng(seed)
    with open(LAYOUT_JSON, encoding="utf-8") as file:
        layout = json.load(file)
    pr = params["particle_radius"]

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
    # positions with the round (unflattened) shape so the bag starts in the can.
    bag_init_verts, _ = _load_obj(BAG_INIT_OBJ)
    assert len(bag_init_verts) == len(bag_verts), "bag init/rest vertex count mismatch"
    for i, (x, y, z) in enumerate(bag_init_verts):
        builder.particle_q[bag_start + i] = wp.vec3(float(x), float(y), float(z))

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
    # override rope initial positions to the matching round (unflattened) shape
    rope_init_verts, _ = _load_obj(ROPE_INIT_OBJ)
    assert len(rope_init_verts) == len(rope_verts), "rope init/rest vertex count mismatch"
    for i, (x, y, z) in enumerate(rope_init_verts):
        builder.particle_q[rope_start + i] = wp.vec3(float(x), float(y), float(z))

    # --- tunnel-closure springs: flap free edge <-> wall (bag-local indices) ---
    tunnel_spring_pairs = np.array(
        [[bag_start + i, bag_start + j] for i, j in layout["tunnel_spring_pairs"]], dtype=np.int32
    )
    for i, j in tunnel_spring_pairs:
        builder.add_spring(i, j, params["closure_ke"], params["closure_kd"], 0.0)
        builder.spring_rest_length[-1] = params["closure_rest_length"]

    # --- pinned drawstring handles (the two exposed side ends) ---
    hv = layout["rope"]["handle_vertex_indices"]
    right_idx = np.array([rope_start + i for i in hv["right"]], dtype=np.int32)
    left_idx = np.array([rope_start + i for i in hv["left"]], dtype=np.int32)

    # --- static round trash can (rigid container) the bag is rounded inside ---
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
    builder.add_shape_mesh(-1, mesh=newton.Mesh(can_v, can_f), cfg=can_cfg, label="trash_can")

    # --- rigid apples dropped into the round bag ---
    r = params["apple_radius"]
    z_floor = float(bag_init_verts[:, 2].min())
    mid = bag_init_verts[(bag_init_verts[:, 2] > 0.05) & (bag_init_verts[:, 2] < 0.30)]
    bag_r = float(np.median(np.hypot(mid[:, 0], mid[:, 1]))) if len(mid) else params["can_radius"]
    rad_in = max(0.0, bag_r - r - 0.015)  # keep apples inside the round wall
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = params["apple_density"]
    cfg.ke = params["apple_ke"]
    cfg.kd = params["apple_kd"]
    cfg.mu = params["apple_mu"]
    cfg.has_particle_collision = True
    cfg.margin = params["apple_margin"]

    body_indices = []
    n = params["num_apples"] if params["enable_apples"] else 0
    for i in range(n):
        ang = i * 2.39996  # golden angle -> even spread across the round mouth
        rr = rad_in * math.sqrt((i + 0.5) / n)  # spiral fill within the round radius
        px = float(rr * math.cos(ang) + rng.uniform(-0.004, 0.004))
        py = float(rr * math.sin(ang) + rng.uniform(-0.004, 0.004))
        pz = z_floor + r + 0.05 + i * 0.06  # stacked so they drop in one by one
        body = builder.add_body(xform=wp.transform(wp.vec3(px, py, pz), wp.quat_identity()), label=f"apple_{i}")
        body_indices.append(body)
        builder.add_shape_sphere(body, radius=r, cfg=cfg)

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
        "z_floor": z_floor,
        "num_tunnel_springs": len(layout["tunnel_spring_pairs"]),
        "tunnel_spring_pairs": tunnel_spring_pairs,
        "num_tethers": 0,
    }


def setup_sim(builder, info, params):
    model = builder.finalize(enable_water_tight_rigid_soft_contact=params["enable_water_tight"])
    model.soft_contact_ke = params["soft_contact_ke"]
    model.soft_contact_kd = params["soft_contact_kd"]
    model.soft_contact_mu = params["soft_contact_mu"]

    # Handle pinning (and the cinch that drives the pinned handles) is optional.
    # When disabled, use empty index sets so the ACTIVE flags stay set and the
    # per-substep move kernel is a no-op -> the handles are fully free.
    if params.get("pin_handles", True):
        right_idx = np.asarray(info["right_idx"], dtype=np.int32)
        left_idx = np.asarray(info["left_idx"], dtype=np.int32)
    else:
        right_idx = np.empty(0, dtype=np.int32)
        left_idx = np.empty(0, dtype=np.int32)

    flags = model.particle_flags.numpy()
    for vi in np.concatenate([right_idx, left_idx]):
        flags[vi] = flags[vi] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags, dtype=wp.int32)

    pq = model.state().particle_q.numpy()
    right = wp.array(right_idx, dtype=wp.int32)
    left = wp.array(left_idx, dtype=wp.int32)
    right_orig = wp.array(pq[right_idx].copy().reshape(-1, 3), dtype=wp.vec3)
    left_orig = wp.array(pq[left_idx].copy().reshape(-1, 3), dtype=wp.vec3)
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
    )
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_margin=params["soft_contact_creation_margin"],
        enable_water_tight_rigid_soft_contact=params["enable_water_tight"],
    )
    return model, solver, pipeline, (right, left, right_orig, left_orig)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = dict(PARAMS)
        self.params["enable_cuda_graph"] = bool(getattr(args, "enable_cuda_graph", self.params["enable_cuda_graph"]))
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0
        self.export_ply = bool(getattr(args, "export_ply", False))
        self.ply_output_dir = getattr(args, "ply_output_dir", os.path.join("outputs", "vbd_trash_bag_ply"))

        seed = getattr(args, "seed", self.params["seed"])
        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.info = build_model(builder, self.params, seed=seed)
        self.model, self.solver, self.pipeline, pins = setup_sim(builder, self.info, self.params)
        self.right, self.left, self.right_orig, self.left_orig = pins
        self.device = self.model.device
        self.right_offset = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.left_offset = wp.zeros(1, dtype=wp.vec3, device=self.device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()

        print(
            f"[trash_bag] bag verts {self.info['bag_count']}  rope verts {self.info['rope_count']}  "
            f"tunnel springs {self.info['num_tunnel_springs']}  tethers {self.info['num_tethers']}  "
            f"apples {len(self.info['body_indices'])}"
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

        self._preroll()
        self._capture_graph()
        if self.export_ply:
            os.makedirs(self.ply_output_dir, exist_ok=True)
            print(f"[trash_bag] exporting PLY frames to {self.ply_output_dir}")

    def _zero_velocities(self):
        if self.state_0.particle_qd is not None:
            self.state_0.particle_qd.zero_()
        if self.state_0.body_qd is not None:
            self.state_0.body_qd.zero_()

    def _preroll(self):
        for _ in range(self.params["preroll_frames"]):
            self.simulate(zero_velocities_each_step=True)

    def _cinch(self):
        """Offsets for the right and left pinned handles."""
        if self.frame <= self.params["settle_frames"]:
            return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)
        t_c = (self.frame - self.params["settle_frames"]) * self.frame_dt
        ramp = min(1.0, t_c / self.params["cinch_ramp"])
        up = self.params["cinch_up"] * ramp
        tg = self.params["cinch_together"] * ramp
        # gather the neck: right handle pulls inward (-x), left inward (+x), both lift up
        return wp.vec3(-tg, 0.0, up), wp.vec3(tg, 0.0, up)

    def _set_cinch_offsets(self):
        off_r, off_l = self._cinch()
        self.right_offset.assign(np.array([[float(off_r[0]), float(off_r[1]), float(off_r[2])]], dtype=np.float32))
        self.left_offset.assign(np.array([[float(off_l[0]), float(off_l[1]), float(off_l[2])]], dtype=np.float32))

    def _capture_graph(self):
        self.graph = None
        if not self.params["enable_cuda_graph"] or not wp.get_device().is_cuda:
            return
        self._set_cinch_offsets()
        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self.graph = capture.graph

    def _simulate_substeps(self, zero_velocities_each_step=False):
        for _ in range(self.sim_substeps):
            wp.launch(
                move_pinned_vertices,
                dim=self.right.shape[0],
                inputs=[self.right, self.right_orig, self.right_offset],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
                device=self.device,
            )
            wp.launch(
                move_pinned_vertices,
                dim=self.left.shape[0],
                inputs=[self.left, self.left_orig, self.left_offset],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            if zero_velocities_each_step:
                self._zero_velocities()

    def simulate(self, zero_velocities_each_step=False):
        self._set_cinch_offsets()
        self._simulate_substeps(zero_velocities_each_step=zero_velocities_each_step)

    def step(self):
        self.frame += 1
        self._set_cinch_offsets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()
        self.sim_time += self.frame_dt
        if self.export_ply:
            _export_ply_frame(self.ply_output_dir, self.frame, self.state_0.particle_q.numpy(), self.info)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pq = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(pq)), "Cloth positions contain non-finite values"
        if self.info["body_indices"]:  # apples optional
            body_q = self.state_0.body_q.numpy()
            apple_pos = body_q[self.info["body_indices"]][:, :3]
            assert np.all(np.isfinite(apple_pos)), "Apple positions contain non-finite values"
            # apples stay roughly within the loaded/lifted bag (no escape / explosion)
            assert np.all(np.abs(apple_pos[:, 0]) < 0.4), "An apple escaped in x"
            assert np.all(np.abs(apple_pos[:, 1]) < 0.3), "An apple escaped in y"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=PARAMS["seed"])
        parser.add_argument(
            "--enable-cuda-graph",
            action=argparse.BooleanOptionalAction,
            default=PARAMS["enable_cuda_graph"],
            help="Capture the simulation step into a CUDA graph when running on CUDA.",
        )
        parser.add_argument(
            "--export-ply",
            action="store_true",
            default=False,
            help="Export one combined bag/rope PLY mesh per simulated frame.",
        )
        parser.add_argument(
            "--ply-output-dir",
            type=str,
            default=os.path.join("outputs", "vbd_trash_bag_ply"),
            help="Directory for per-frame PLY files when --export-ply is set.",
        )
        parser.set_defaults(num_frames=PARAMS["settle_frames"] + PARAMS["cinch_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
