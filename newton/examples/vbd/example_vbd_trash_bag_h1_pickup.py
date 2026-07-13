# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Trash Bag H1 Pickup (unified AVBD/VBD)
#
# The dish-washing scene's fixed-base Unitree H1 stands at the same table,
# but instead of dishes a round trash can sits on the tabletop with the
# procedural drawstring trash bag (vbd/asset/trash_bag*.obj) lining it.
# Rigid "trash" spheres drop into the bag while the two exposed drawstring
# handles fall out of the bag's side holes and dangle at the robot's left
# and right.
#
# Each hand then hooks its handle with the INDEX-TO-LITTLE fingers only —
# the thumb stays open and is never used. The hand approaches in the
# thumb-up "handshake" orientation, sweeps its half-curled fingers past the
# hanging strands so they enter the curl pocket, closes the four fingers
# into a hook, and both hands lift together until the bag leaves the can.
#
# All grasps are physical: no particles are pinned and nothing is scripted
# onto the hands — the hook works by contact and friction alone. The hook
# targets are planned at runtime from where the handles actually settled.
# Newton IK converts task-space hand keyframes into joint targets and one
# SolverVBD instance advances the H1 + trash with AVBD and the bag + rope
# cloth with VBD in the same solve.
#
# Command: python -m newton.examples vbd_trash_bag_h1_pickup
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
import newton.ik as ik
import newton.utils

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset")
BAG_OBJ = os.path.join(ASSET, "trash_bag.obj")
ROPE_OBJ = os.path.join(ASSET, "trash_bag_rope.obj")
# Round INITIAL positions (same topology as the flat rest OBJs): the flat bag
# unflattened onto a cylinder so it starts round, lining the round can, while
# its rest shape stays the flat pressed tube (see example_vbd_trash_bag.py).
BAG_INIT_OBJ = os.path.join(ASSET, "trash_bag_init.obj")
ROPE_INIT_OBJ = os.path.join(ASSET, "trash_bag_rope_init.obj")
LAYOUT_JSON = os.path.join(ASSET, "trash_bag_layout.json")

PARAMS = {
    # simulation
    "fps": 60,
    # note: VBD's penalty friction scales with per-substep displacement, so
    # substep changes shift every contact's effective grip
    "sim_substeps": 10,
    "solver_iterations": 10,
    "enable_cuda_graph": True,
    "gravity": -9.81,
    "num_frames": 1020,
    # presentation (the dish-washing scene's camera)
    "camera_position": (1.43, 0.42, 1.77),
    "camera_pitch": -15.9,
    "camera_yaw": -160.8,
    "camera_fov": 45.0,
    # table (reused unchanged from the dish-washing scene; the front edge
    # faces the robot at x = -table_half_width)
    "table_half_width": 0.20,
    "table_half_depth": 0.36,
    "table_top_z": 1.09,
    "tabletop_half_height": 0.04,
    "table_mu": 0.9,
    "shape_ke": 1.0e3,
    "shape_kd": 1.0e-4,
    "rigid_contact_gap": 0.001,
    # trash can + bag placement. All asset-local dimensions below are scaled
    # by ``bag_scale`` — the floor-standing bag would put the handles beyond
    # the fixed-base H1's comfortable reach once raised onto the table.
    "bag_scale": 0.7,
    # as close to the robot as the can fits on the table (can top outer
    # radius ~0.10 against the front edge at -0.20)
    "can_x": -0.095,
    "can_y": 0.0,
    "can_bottom_radius": 0.12,
    "can_top_radius": 0.14,
    # a squat desk bin, much shorter than the floor demo's 0.31 can: the
    # extraction height is rim + hanging bag length + strand share of the
    # drawstring loop, and every centimetre of rim costs hand height at the
    # lift apex. The low rim also leaves the fallen handles hanging in free
    # air instead of hugging the can wall.
    "can_height": 0.14,
    "can_z_bottom": -0.01,
    "can_wall_thickness": 0.0025,
    "can_floor_thickness": 0.004,
    "can_ke": 5.0e5,
    "can_kd": 5.0e1,
    "can_mu": 0.4,
    "can_margin": 0.002,
    "can_n_around": 72,
    "can_n_rows": 20,
    "can_color": (0.35, 0.38, 0.40),
    # bag cloth (floppy plastic; parameters follow example_vbd_trash_bag.py).
    # Rest-scale is kept MUCH tighter than the standalone demo's 1.5: a bag
    # hanging from its cinched neck extends to its REST height under the
    # trash weight, and the drawstring loop's rest length is exactly the
    # slack the hands must reel in before the bag rises — at 1.5x both blow
    # the lift apex beyond the fixed-base H1's reach envelope. The can is
    # sized at the bag's init radius, so the liner needs no expansion slack.
    # Bag and rope MUST share one rest scale: a rope shorter than the bag's
    # rest mouth chokes the expanding hem, and the resulting tug-of-war
    # whips the drawstring through the tunnels during the settle.
    "bag_rest_scale": 1.05,
    "rope_rest_scale": 1.05,
    "particle_radius": 0.004,
    "cloth_density": 0.08,
    "cloth_tri_ke": 1.0e5,
    "cloth_tri_ka": 5.0e4,
    "cloth_tri_kd": 1.0e1,
    "cloth_edge_ke": 0.2,  # low bending -> floppy, wrinkly plastic
    "cloth_edge_kd": 0.1,
    # rope cloth (the drawstring): stiff/inextensible ribbon
    "rope_density": 0.008,
    "rope_tri_ke": 2.0e5,
    "rope_tri_ka": 2.0e5,
    "rope_tri_kd": 1.0e2,
    "rope_edge_ke": 0.05,
    "rope_edge_kd": 0.01,
    # tunnel closure springs: flap free edge <-> wall
    "closure_ke": 2.0e4,
    "closure_kd": 1.0e-3,
    # rigid trash spheres dropped into the bag (radius in asset-local units)
    "num_trash": 5,
    "trash_radius": 0.034,
    "trash_density": 1000.0,
    "trash_ke": 5.0e5,
    "trash_kd": 5.0e1,
    "trash_mu": 0.5,
    "trash_margin": 0.005,
    "trash_colors": ((0.55, 0.50, 0.45), (0.40, 0.45, 0.50), (0.60, 0.55, 0.40)),
    # cloth contacts (the bag needs the stiff 2e5 of the standalone demo, not
    # the dish demo's sponge-soft 8e2 — a loaded ribbon tunnels through the
    # finger colliders at low contact stiffness)
    "soft_contact_ke": 2.0e5,
    "soft_contact_kd": 1.0e1,
    "soft_contact_mu": 0.3,
    "soft_contact_margin": 0.012,
    "particle_self_contact_radius": 0.003,
    "particle_self_contact_margin": 0.006,
    # water-tight SDF contact keeps the bag from tunneling through the thin
    # can wall. The dish demo's NaN failure was specific to deep volumetric
    # fingertip clamps; the hook never squeezes the cloth against a fingertip.
    "enable_water_tight_rigid_soft_contact": True,
    # H1 (identical placement to the dish-washing scene)
    "robot_base_x": -0.70,
    "robot_contact_ke": 1.0e3,
    "robot_contact_kd": 1.0e-2,
    "robot_contact_mu": 0.5,
    "robot_contact_margin": 0.002,
    "finger_contact_margin": 0.002,
    "robot_sdf_padding": 0.012,
    "robot_sdf_max_resolution": 64,
    "finger_sdf_padding": 0.012,
    "finger_sdf_max_resolution": 128,
    # hook colliders (palm + index..pinky chains). Contact ke matches
    # soft_contact_ke so the averaged body-particle contact stays stiff and
    # the loaded strands cannot ramp-penetrate the fingers; high mu so the
    # caged ribbon does not creep off the curl.
    "hook_finger_contact_ke": 2.0e5,
    "finger_contact_kd": 2.0e1,
    "finger_contact_mu": 200.0,
    # task-space rest poses (dish-washing scene)
    "rest_left": (-0.48, 0.24, 1.24),
    "rest_right": (-0.48, -0.24, 1.24),
    # hook primitive. The hand rides in thumb-up: the horizontal finger-curl
    # plane catches the vertical hanging strands. Targets are planned at
    # runtime from the settled handle geometry; the offsets below are relative
    # to each handle's lowest vertex (the hanging bight).
    "hook_pre_curl": 0.45,  # half-open curl during the sweep (opening faces the strand)
    # closing tighter around a caged ribbon squeezes it out watermelon-seed
    # style — close to a firm hook and keep the fist static afterwards
    "hook_close_index": 0.95,
    "hook_close_other": 0.90,
    "hook_hover_back": 0.10,  # hover this far short of the strand along the reach
    "hook_hover_up": 0.08,
    # catch height above the handle's lowest vertex: mid-strand, clear of
    # whatever the bight rests on but safely below the holes
    "hook_above_bight": 0.05,
    # the curl pocket trails the pinch point, so overshoot the strand along
    # the horizontal reach direction to land the pocket on it
    "hook_overshoot": 0.045,
    # nudge the target toward the bag: the pocket slips between the hanging
    # drape and the bag wall
    "hook_nudge_in": 0.02,
    # lift, three legs:
    #   gather  — move above the holes while the strand is still slack
    #   extract — nearly straight up over the holes (a sideways pull while
    #             the bag is wedged in the can drags the rim and strips the
    #             loops). Pulling the handles up first reels the drawstring
    #             OUT of the tunnels (the loop cinches the mouth), so the
    #             apex is budgeted from the WHOLE loop: neck height (rim +
    #             hanging rest-length of the bag) + each strand's quarter
    #             share of the loop slack left after the cinched-neck wrap.
    #   retreat — a little further up and back toward the robot
    "lift_clearance": 0.03,
    "gather_above_holes": 0.05,
    # estimated rope length still wrapped around the gathered neck when the
    # drawstring is pulled taut (larger = less strand hangs below the hands)
    "neck_wrap": 0.20,
    # extra hanging-bag length under the trash weight (cloth stretch, drape)
    "bag_hang_margin": 0.03,
    # never command the hands above this (IK gets twitchy at full extension)
    "hand_z_max": 1.72,
    # small backward drift during the extract leg keeps the apex inside the
    # comfortable reach envelope
    "extract_drift_back": 0.05,
    "lift_secondary": 0.06,
    "pull_back": 0.15,
    # durations [s]
    "settle_time": 2.5,  # bag expands into the can, trash drops, handles fall
    "approach_time": 1.2,
    "descend_time": 1.0,
    "sweep_time": 1.2,
    "close_time": 1.5,
    "dwell_time": 0.4,
    "gather_time": 2.0,
    "lift_time": 3.5,
    "lift2_time": 1.5,
    "hold_time": 1.5,
    # AVBD joint drives; Newton IK only generates their targets
    "joint_drive_ke": 5.0e4,
    "joint_drive_kd": 5.0e2,
    "torso_drive_ke": 2.0e5,
    "torso_drive_kd": 2.0e3,
    # stiff finger drive on both hands: the hook must hold its curl under the
    # hanging bag load (the fist stays static after the catch, so the
    # squeeze-ejection risk of a stiff drive does not apply)
    "finger_drive_ke": 2.0e4,
    "finger_drive_kd": 1.0e2,
    # per-frame clamp on joint-target motion: smooths IK jumps without the
    # long saturated sweeps a very tight clamp produces
    "joint_target_velocity_limit": 20.0,
    "torso_ik_position_weight": 50.0,
    "torso_ik_rotation_weight": 50.0,
    "seed": 42,
}

# Pinch-point offsets in the hand-link frames, calibrated for the H1 hand
# meshes (shared with the dish-washing demo).
HAND_OFFSETS = (
    (0.146273, -0.068447, 0.028077),
    (0.148808, 0.068652, 0.026675),
)

# Thumb-up "handshake" hook orientation for both hands: identity yaw plus a
# small pitch about y. The horizontal finger-curl plane catches the fallen
# loops' vertical strands; a palm-down pose curls parallel to them and cannot.
_HOOK_PITCH = 0.45
HAND_ROTATIONS = (
    (0.0, math.sin(0.5 * _HOOK_PITCH), 0.0, math.cos(0.5 * _HOOK_PITCH)),
    (0.0, math.sin(0.5 * _HOOK_PITCH), 0.0, math.cos(0.5 * _HOOK_PITCH)),
)

# Side-specific fully-closed thumb angles (the thumb stays open in this demo,
# but the finger-target machinery keys all six groups).
THUMB_CLOSED_VALUES = (
    (1.273907, 0.160957, 0.369535, 0.892908),
    (1.192278, 0.195421, 0.400690, 0.679765),
)

_FINGER_GROUP_LEFT_THUMB = wp.constant(0)
_FINGER_GROUP_RIGHT_THUMB = wp.constant(1)
_FINGER_GROUP_LEFT_INDEX = wp.constant(2)
_FINGER_GROUP_RIGHT_INDEX = wp.constant(3)
_FINGER_GROUP_LEFT_OTHER = wp.constant(4)
_FINGER_GROUP_RIGHT_OTHER = wp.constant(5)


@wp.kernel
def set_finger_targets(
    joint_q: wp.array[float],
    finger_indices: wp.array[wp.int32],
    closed_values: wp.array[float],
    finger_groups: wp.array[wp.int32],
    fractions: wp.array[float],
):
    i = wp.tid()
    joint_q[finger_indices[i]] = fractions[finger_groups[i]] * closed_values[i]


@wp.kernel
def update_control_targets(
    desired_q: wp.array[float],
    previous_q: wp.array[float],
    inv_dt: float,
    velocity_limit: float,
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    i = wp.tid()
    q_prev = previous_q[i]
    max_delta = velocity_limit / inv_dt
    delta = wp.clamp(desired_q[i] - q_prev, -max_delta, max_delta)
    q = q_prev + delta
    qd = delta * inv_dt
    target_q[i] = q
    target_qd[i] = qd
    previous_q[i] = q


def _load_obj(path: str) -> tuple[np.ndarray, list[int]]:
    """Load a triangle-mesh OBJ preserving vertex order (so the layout JSON
    indices stay valid)."""
    vertices = []
    faces: list[int] = []
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


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [i for i, label in enumerate(labels) if label.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def _normalized_quat(values: tuple[float, float, float, float]) -> wp.quat:
    q = np.asarray(values, dtype=np.float32)
    q /= np.linalg.norm(q)
    return wp.quat(*q)


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
    """Build external VBD self-contact filters across the tunnel closure seams."""
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

    See example_vbd_trash_bag.py: a closed cup cross-section revolved around
    the z-axis, so a bag particle in the cavity cannot cross the thin wall.
    """
    z_top = z_bottom + height
    z_floor_top = z_bottom + floor_thickness
    r_bot_out = bottom_radius + wall_thickness
    r_top_out = top_radius + wall_thickness

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
    faces = np.array(faces, dtype=np.int32)[:, ::-1]  # outward winding
    return np.array(verts, dtype=np.float32), faces.reshape(-1)


class _Track:
    """A piecewise keyframe channel with smoothstep or linear easing per segment."""

    def __init__(self, value):
        self.times = [0.0]
        self.values = [np.asarray(value, dtype=np.float64)]
        self.eases = ["smooth"]

    def add(self, time: float, value, ease: str = "smooth"):
        if time < self.times[-1] - 1.0e-9:
            raise ValueError(f"Keyframe time {time} precedes the last key at {self.times[-1]}")
        self.times.append(float(time))
        self.values.append(np.asarray(value, dtype=np.float64))
        self.eases.append(ease)

    def hold_until(self, time: float):
        if time > self.times[-1] + 1.0e-9:
            self.add(time, self.values[-1], "linear")

    def sample(self, t: float) -> np.ndarray:
        times = self.times
        if t >= times[-1]:
            return self.values[-1]
        i = int(np.searchsorted(times, t, side="right"))
        i = max(i, 1)
        t0, t1 = times[i - 1], times[i]
        u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        u = min(max(u, 0.0), 1.0)
        if self.eases[i] == "smooth":
            u = u * u * (3.0 - 2.0 * u)
        return self.values[i - 1] * (1.0 - u) + self.values[i] * u


class _HandCursor:
    """Writes one hand's keyframes (pinch position + finger fractions) on its own clock."""

    def __init__(self, tracks: dict[str, _Track], side: str):
        self._tracks = tracks
        self._side = side
        self.time = 0.0

    def move(
        self,
        duration: float,
        pos=None,
        thumb: float | None = None,
        index: float | None = None,
        other: float | None = None,
        ease: str = "smooth",
    ):
        end = self.time + duration
        channels = {"pos": pos, "thumb": thumb, "index": index, "other": other}
        for name, value in channels.items():
            if value is None:
                continue
            track = self._tracks[f"{self._side}_{name}"]
            track.hold_until(self.time)
            track.add(end, value, ease)
        self.time = end

    def wait(self, duration: float):
        self.time += duration

    def wait_until(self, time: float):
        self.time = max(self.time, time)

    def pos(self) -> np.ndarray:
        return self._tracks[f"{self._side}_pos"].sample(self.time)


def _add_h1(builder: newton.ModelBuilder, params: dict) -> tuple[dict[str, int], list[int], list[int]]:
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

    # The hook colliders — the palm and the index..pinky chains of both hands —
    # keep shape-shape contact AND gain particle contact with a finer
    # texture-backed SDF: they physically catch and carry the cloth ribbon.
    # The thumb is deliberately NOT a particle collider (this demo hooks with
    # index-to-little fingers only; the open thumb stays clear of the strands).
    # Every non-hook collider loses particle contact, and the whole robot is
    # filtered against the furniture in Example.__init__ (a forearm resting on
    # the tabletop otherwise explodes the AVBD contact solve).
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
    hook_shapes = []
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
            builder.shape_material_ke[shape] = params["hook_finger_contact_ke"]
            builder.shape_material_kd[shape] = params["finger_contact_kd"]
            builder.shape_material_mu[shape] = params["finger_contact_mu"]
            builder.shape_margin[shape] = params["finger_contact_margin"]
            builder.shape_sdf_padding[shape] = params["finger_sdf_padding"]
            builder.shape_sdf_max_resolution[shape] = params["finger_sdf_max_resolution"]
            builder.shape_sdf_target_voxel_size[shape] = None
            hook_shapes.append(shape)
            hook_shape_count += 1

    # 2 palm geoms + 4 two-link finger chains per hand = 10 colliders per side
    if hook_shape_count != 20:
        raise RuntimeError(f"Expected 20 H1 palm/index..pinky colliders, found {hook_shape_count}")
    return body_indices, robot_rigid_shapes, hook_shapes


def _add_table(builder: newton.ModelBuilder, params: dict) -> list[int]:
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


def _add_trash_can_and_bag(builder: newton.ModelBuilder, params: dict, seed: int) -> dict:
    """Place the scaled trash can on the tabletop and line it with the bag.

    The whole assembly is rotated about z so the two drawstring handles exit
    at the robot's left (+y) and right (-y), then scaled by ``bag_scale`` and
    translated onto the table at (can_x, can_y).
    """
    rng = np.random.default_rng(seed)
    with open(LAYOUT_JSON, encoding="utf-8") as file:
        layout = json.load(file)
    s = params["bag_scale"]
    pr = params["particle_radius"]

    bag_verts, bag_faces = _load_obj(BAG_OBJ)
    bag_init_verts, _ = _load_obj(BAG_INIT_OBJ)
    rope_verts, rope_faces = _load_obj(ROPE_OBJ)
    rope_init_verts, _ = _load_obj(ROPE_INIT_OBJ)
    assert len(bag_init_verts) == len(bag_verts), "bag init/rest vertex count mismatch"
    assert len(rope_init_verts) == len(rope_verts), "rope init/rest vertex count mismatch"

    # rotate the asset so the handle holes face +/-y (the robot's left/right)
    handle_indices_local = layout["rope"]["handle_vertex_indices"]
    left_centroid = rope_init_verts[np.asarray(handle_indices_local["left"], dtype=np.int32)].mean(axis=0)
    theta = -0.5 * math.pi - math.atan2(float(left_centroid[1]), float(left_centroid[0]))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # can outer floor rests on the tabletop
    origin = np.asarray(
        [params["can_x"], params["can_y"], params["table_top_z"] + 0.002 - s * params["can_z_bottom"]],
        dtype=np.float64,
    )

    def to_world(vertices: np.ndarray) -> np.ndarray:
        v = np.asarray(vertices, dtype=np.float64) * s
        out = np.empty_like(v)
        out[:, 0] = cos_t * v[:, 0] - sin_t * v[:, 1] + origin[0]
        out[:, 1] = sin_t * v[:, 0] + cos_t * v[:, 1] + origin[1]
        out[:, 2] = v[:, 2] + origin[2]
        return out

    # --- bag shell (rest = oversized flat tube, initial = round world pose) ---
    bag_start = len(builder.particle_q)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=params["bag_rest_scale"] * s,
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
    for i, pos in enumerate(to_world(bag_init_verts)):
        builder.particle_q[bag_start + i] = wp.vec3(*(float(c) for c in pos))

    # --- drawstring tie (ribbon) ---
    rope_start = len(builder.particle_q)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=params["rope_rest_scale"] * s,
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
    for i, pos in enumerate(to_world(rope_init_verts)):
        builder.particle_q[rope_start + i] = wp.vec3(*(float(c) for c in pos))

    # --- tunnel-closure springs: flap free edge <-> wall ---
    tunnel_spring_pairs = np.asarray(
        [[bag_start + i, bag_start + j] for i, j in layout["tunnel_spring_pairs"]], dtype=np.int32
    )
    for i, j in tunnel_spring_pairs:
        builder.add_spring(int(i), int(j), params["closure_ke"], params["closure_kd"], 0.0)
        builder.spring_rest_length[-1] = 0.0

    # --- static round trash can on the tabletop ---
    can_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["can_ke"],
        kd=params["can_kd"],
        mu=params["can_mu"],
        gap=params["rigid_contact_gap"],
        has_particle_collision=True,
        margin=params["can_margin"],
    )
    can_v, can_f = build_can_mesh(
        params["can_bottom_radius"] * s,
        params["can_top_radius"] * s,
        params["can_z_bottom"] * s,
        params["can_height"] * s,
        params["can_wall_thickness"],
        params["can_floor_thickness"],
        params["can_n_around"],
        params["can_n_rows"],
    )
    can_shape = builder.add_shape_mesh(
        -1,
        xform=wp.transform(wp.vec3(*(float(c) for c in origin)), wp.quat_identity()),
        mesh=newton.Mesh(can_v, can_f),
        cfg=can_cfg,
        color=wp.vec3(*params["can_color"]),
        label="trash_can",
    )
    # the 2.5 mm wall needs fine SDF voxels for the water-tight containment
    builder.shape_sdf_max_resolution[can_shape] = 256

    # --- rigid trash spheres dropped into the round bag (asset-local spiral) ---
    r_local = params["trash_radius"]
    z_floor_local = float(bag_init_verts[:, 2].min())
    mid = bag_init_verts[(bag_init_verts[:, 2] > 0.05) & (bag_init_verts[:, 2] < 0.30)]
    bag_r_local = float(np.median(np.hypot(mid[:, 0], mid[:, 1])))
    rad_in = max(0.0, bag_r_local - r_local - 0.015)
    trash_cfg = newton.ModelBuilder.ShapeConfig(
        density=params["trash_density"],
        ke=params["trash_ke"],
        kd=params["trash_kd"],
        mu=params["trash_mu"],
        has_particle_collision=True,
        margin=params["trash_margin"],
    )
    trash_bodies = []
    trash_shapes = []
    n = params["num_trash"]
    colors = params["trash_colors"]
    for i in range(n):
        ang = i * 2.39996  # golden angle -> even spread across the round mouth
        rr = rad_in * math.sqrt((i + 0.5) / n)
        px = float(rr * math.cos(ang) + rng.uniform(-0.004, 0.004))
        py = float(rr * math.sin(ang) + rng.uniform(-0.004, 0.004))
        pz = z_floor_local + r_local + 0.05 + i * 0.06  # stacked so they drop in one by one
        pos = to_world(np.asarray([[px, py, pz]]))[0]
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*(float(c) for c in pos)), wp.quat_identity()), label=f"trash_{i}"
        )
        shape = builder.add_shape_sphere(
            body, radius=r_local * s, cfg=trash_cfg, color=wp.vec3(*colors[i % len(colors)])
        )
        trash_bodies.append(body)
        trash_shapes.append(shape)

    rim_z = float(origin[2] + s * (params["can_z_bottom"] + params["can_height"]))
    # lift-budget geometry: the rest length of the closed drawstring loop
    # (the total material the hands can reel out of the tunnels) and the rest
    # height of the bag (how far it hangs below the cinched neck when lifted)
    rope_nodes = rope_verts.reshape(-1, 3, 3).mean(axis=1).astype(np.float64)
    loop_len = float(np.linalg.norm(np.diff(rope_nodes, axis=0), axis=1).sum())
    loop_len += float(np.linalg.norm(rope_nodes[0] - rope_nodes[-1]))
    loop_rest_len = loop_len * params["rope_rest_scale"] * s
    bag_hang_len = float(bag_verts[:, 2].max() - bag_verts[:, 2].min()) * params["bag_rest_scale"] * s
    return {
        "bag_start": bag_start,
        "bag_count": rope_start - bag_start,
        "rope_start": rope_start,
        "rope_count": len(rope_verts),
        "tunnel_spring_pairs": tunnel_spring_pairs,
        "handle_indices": {
            side: np.asarray([rope_start + i for i in handle_indices_local[side]], dtype=np.int32)
            for side in ("left", "right")
        },
        "trash_bodies": trash_bodies,
        "trash_shapes": trash_shapes,
        "can_shape": can_shape,
        "can_center": np.asarray([params["can_x"], params["can_y"]], dtype=np.float64),
        "rim_z": rim_z,
        "bag_bottom_z": float(origin[2]),
        "loop_rest_len": loop_rest_len,
        "bag_hang_len": bag_hang_len,
    }


class Example:
    def __init__(self, viewer, args, params: dict | None = None):
        self.viewer = viewer
        self.params = PARAMS if params is None else params
        p = self.params
        self.fps = p["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = p["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.phase = "settle"
        self._phase_marks: list[tuple[float, str]] = []
        self._hook_planned = False
        self.hook_targets: dict[str, np.ndarray] = {}

        seed = getattr(args, "seed", p["seed"]) if args is not None else p["seed"]

        builder = newton.ModelBuilder(gravity=p["gravity"])
        self.robot_bodies, robot_rigid_shapes, _hook_shapes = _add_h1(builder, p)
        self.robot_coord_count = builder.joint_coord_count
        table_shapes = _add_table(builder, p)
        self.info = _add_trash_can_and_bag(builder, p, seed)
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=p["shape_ke"],
            kd=p["shape_kd"],
            mu=p["table_mu"],
            gap=p["rigid_contact_gap"],
        )
        ground_shape = builder.add_ground_plane(cfg=ground_cfg)
        # The robot only manipulates the CLOTH (particle contact on the hook
        # colliders). Filter every robot rigid shape against the furniture and
        # the trash spheres: a forearm brushing the static can or tabletop
        # otherwise detonates the AVBD contact solve, and the hands never
        # touch the trash directly (it rides inside the bag).
        furniture = [*table_shapes, self.info["can_shape"], ground_shape]
        for robot_shape in robot_rigid_shapes:
            for other in furniture:
                builder.add_shape_collision_filter_pair(robot_shape, other)
            for trash_shape in self.info["trash_shapes"]:
                builder.add_shape_collision_filter_pair(robot_shape, trash_shape)
        builder.color(include_bending=True)

        if p["enable_water_tight_rigid_soft_contact"]:
            builder.enable_rigid_mesh_sdfs()
        self.model = builder.finalize()
        self.model.soft_contact_ke = p["soft_contact_ke"]
        self.model.soft_contact_kd = p["soft_contact_kd"]
        self.model.soft_contact_mu = p["soft_contact_mu"]

        self.hand_bodies = [self.robot_bodies["left_hand"], self.robot_bodies["right_hand"]]
        self.hand_offsets = [wp.vec3(*values) for values in HAND_OFFSETS]
        self.hand_rotations = [_normalized_quat(values) for values in HAND_ROTATIONS]

        self._init_tracks()
        # durations are fixed; only the hook POSITIONS wait for the settled
        # handle geometry (planned in _plan_hook at the end of the settle)
        self.total_time = (
            p["settle_time"]
            + p["approach_time"]
            + p["descend_time"]
            + p["sweep_time"]
            + p["close_time"]
            + p["dwell_time"]
            + p["gather_time"]
            + p["lift_time"]
            + p["lift2_time"]
            + p["hold_time"]
            + 0.6
        )

        self._setup_ik()
        self._solve_ik(
            np.asarray([p["rest_left"], p["rest_right"]], dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            iterations=48,
        )
        self.model.joint_q.assign(self.ik_joint_q_flat)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.torso_initial_transform = self.model.body_q.numpy()[self.torso_body].copy()

        vertex_filter, edge_filter = _build_tunnel_seam_contact_filters(self.model, self.info["tunnel_spring_pairs"])
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            soft_contact_margin=p["soft_contact_margin"],
            enable_water_tight_rigid_soft_contact=p["enable_water_tight_rigid_soft_contact"],
            contact_matching="latest",
        )
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=p["solver_iterations"],
            # AVBD advances the H1 and trash in the same solve as the VBD cloth.
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=p["particle_self_contact_radius"],
            particle_self_contact_margin=p["particle_self_contact_margin"],
            particle_external_vertex_contact_filtering_map=vertex_filter,
            particle_external_edge_contact_filtering_map=edge_filter,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=512,
            rigid_body_particle_contact_buffer_size=16384,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
            rigid_joint_linear_kd=1.0e2,
            rigid_joint_angular_kd=1.0e2,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)
        wp.copy(self.control.joint_target_q, self.model.joint_q, count=self.robot_coord_count)
        self.control.joint_target_qd.zero_()
        self.previous_joint_targets = wp.clone(self.model.joint_q[: self.robot_coord_count])

        print(
            f"[trash_bag_h1_pickup] bag verts {self.info['bag_count']}  rope verts {self.info['rope_count']}  "
            f"tunnel springs {len(self.info['tunnel_spring_pairs'])}  trash {len(self.info['trash_bodies'])}  "
            f"rim z {self.info['rim_z']:.3f}"
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

    # ── choreography ─────────────────────────────────────────────────────

    def _mark(self, time: float, name: str):
        self._phase_marks.append((time, name))

    def _init_tracks(self):
        p = self.params
        tracks: dict[str, _Track] = {}
        for side, rest in (("left", p["rest_left"]), ("right", p["rest_right"])):
            tracks[f"{side}_pos"] = _Track(np.asarray(rest, dtype=np.float64))
            tracks[f"{side}_thumb"] = _Track(0.0)
            tracks[f"{side}_index"] = _Track(0.0)
            tracks[f"{side}_other"] = _Track(0.0)
        self.tracks = tracks

    def _plan_hook(self):
        """Plan both hook grasps from where the handles ACTUALLY settled.

        Each handle hangs from its two holes near the bag top. From the live
        state this reconstructs each handle's centerline polyline and picks a
        catch point mid-strand. The extraction apex is budgeted from the
        WHOLE drawstring loop (see the lift parameter comments): pulling the
        handles up reels the loop out of the tunnels and cinches the mouth,
        so the hands must end at neck height (rim + hanging bag rest length)
        plus each strand's quarter share of the loop slack.
        """
        p = self.params
        particle_q = self.state_0.particle_q.numpy()
        can_xy = self.info["can_center"]

        plans: dict[str, dict] = {}
        for asset_side in ("left", "right"):
            verts = particle_q[self.info["handle_indices"][asset_side]]
            # handle vertices are node-major (56 centerline nodes x 3 width
            # verts, contiguous ids) -> ordered centerline polyline
            nodes = verts.reshape(-1, 3, 3).mean(axis=1).astype(np.float64)
            bight_z = float(nodes[:, 2].min())
            hole_top_z = float(nodes[:, 2].max())
            hook_z = min(bight_z + p["hook_above_bight"], hole_top_z - 0.04)
            # aim where the strand actually crosses the catch height
            band = nodes[np.abs(nodes[:, 2] - hook_z) < 0.03]
            if len(band) == 0:
                band = nodes[[int(np.argmin(np.abs(nodes[:, 2] - hook_z)))]]
            anchor_xy = band[:, :2].mean(axis=0)
            hole_xy = 0.5 * (nodes[0, :2] + nodes[-1, :2])
            hand_side = "left" if float(nodes[:, 1].mean()) > can_xy[1] else "right"
            plans[hand_side] = {
                "asset_side": asset_side,
                "anchor_xy": anchor_xy,
                "hook_z": hook_z,
                "hole_xy": hole_xy,
                "hole_top_z": hole_top_z,
                "bight_z": bight_z,
            }
        if len(plans) != 2:
            raise RuntimeError("Both drawstring handles settled on the same side of the can")

        t0 = self.sim_time
        self._mark(t0, "approach")
        cursors = {}
        for side in ("left", "right"):
            plan = plans[side]
            anchor_xy = plan["anchor_xy"]
            hook_z = plan["hook_z"]
            rest = np.asarray(p[f"rest_{side}"], dtype=np.float64)
            reach = anchor_xy - rest[:2]
            reach /= np.linalg.norm(reach) + 1.0e-9
            toward_bag = can_xy - anchor_xy
            toward_bag /= np.linalg.norm(toward_bag) + 1.0e-9
            hover_xy = anchor_xy - reach * p["hook_hover_back"]
            target_xy = anchor_xy + reach * p["hook_overshoot"] + toward_bag * p["hook_nudge_in"]
            plan["reach"] = reach
            self.hook_targets[side] = np.asarray([target_xy[0], target_xy[1], hook_z], dtype=np.float64)

            cur = _HandCursor(self.tracks, side)
            cur.time = t0
            # pre-curl the four hook fingers on the way in; the thumb is
            # never keyed and stays fully open for the whole demo
            cur.move(
                p["approach_time"],
                pos=(hover_xy[0], hover_xy[1], hook_z + p["hook_hover_up"]),
                index=p["hook_pre_curl"],
                other=p["hook_pre_curl"],
            )
            cur.move(p["descend_time"], pos=(hover_xy[0], hover_xy[1], hook_z))
            if side == "left":
                self._mark(cur.time, "sweep")
            # sweep the half-open curl past the hanging strands
            cur.move(p["sweep_time"], pos=(target_xy[0], target_xy[1], hook_z))
            if side == "left":
                self._mark(cur.time, "close")
            # close into a hook and keep the fist static afterwards (curling
            # tighter around the caged ribbon squeezes it back out)
            cur.move(p["close_time"], index=p["hook_close_index"], other=p["hook_close_other"])
            cur.wait(p["dwell_time"])
            cursors[side] = cur

        # both hands lift TOGETHER: gather over the holes while the strand is
        # slack, extract nearly straight up, then retreat up-and-back into
        # the comfortable part of the reach envelope
        self._mark(cursors["left"].time, "gather")
        strand_h = max(0.05, 0.25 * (self.info["loop_rest_len"] - p["neck_wrap"]))
        neck_z = self.info["rim_z"] + self.info["bag_hang_len"] + p["bag_hang_margin"]
        z_out = min(neck_z + strand_h + p["lift_clearance"], p["hand_z_max"])
        for side in ("left", "right"):
            cur = cursors[side]
            plan = plans[side]
            hole_xy = plan["hole_xy"]
            gather_z = plan["hole_top_z"] + p["gather_above_holes"]
            cur.move(p["gather_time"], pos=(hole_xy[0], hole_xy[1], gather_z))
            if side == "left":
                self._mark(cur.time, "extract")
            out_xy = hole_xy - plan["reach"] * p["extract_drift_back"]
            cur.move(p["lift_time"], pos=(out_xy[0], out_xy[1], z_out))
            back_xy = hole_xy - plan["reach"] * p["pull_back"]
            cur.move(p["lift2_time"], pos=(back_xy[0], back_xy[1], z_out + p["lift_secondary"]))
            cur.wait(p["hold_time"])
        self._mark(cursors["left"].time - p["hold_time"], "clear")
        self._mark(cursors["left"].time, "done")
        self._phase_marks.sort(key=lambda mark: mark[0])

        print(
            f"[trash_bag_h1_pickup] lift budget: loop rest {self.info['loop_rest_len']:.3f} "
            f"strand {strand_h:.3f} neck z {neck_z:.3f} apex z {z_out:.3f}"
        )
        for side in ("left", "right"):
            plan = plans[side]
            print(
                f"[trash_bag_h1_pickup] {side} hand <- {plan['asset_side']} handle: "
                f"anchor ({plan['anchor_xy'][0]:+.3f},{plan['anchor_xy'][1]:+.3f}) hook z {plan['hook_z']:.3f} "
                f"bight z {plan['bight_z']:.3f} hole top z {plan['hole_top_z']:.3f}"
            )
        self._hook_planned = True

    # ── IK ───────────────────────────────────────────────────────────────

    def _setup_ik(self):
        initial_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, initial_state)
        body_q = initial_state.body_q.numpy()

        self.torso_body = self.robot_bodies["torso"]
        torso_transform = wp.transform(*body_q[self.torso_body])
        torso_position = wp.transform_get_translation(torso_transform)
        torso_rotation = wp.transform_get_rotation(torso_transform)
        self.torso_position_objective = ik.IKObjectivePosition(
            link_index=self.torso_body,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([torso_position], dtype=wp.vec3),
            weight=self.params["torso_ik_position_weight"],
        )
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
                    # deliberately weak: a strong rotation constraint makes
                    # the redundant arm flip between IK branches frame to
                    # frame, which reads as arm instability
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
            # heavier LM damping: the per-frame warm-started solve then makes
            # small consistent steps instead of wandering through the arm's
            # redundancy, which shows up as idle-arm jitter
            lambda_initial=1.0,
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
            thumb_group = _FINGER_GROUP_LEFT_THUMB if side == "L" else _FINGER_GROUP_RIGHT_THUMB
            index_group = _FINGER_GROUP_LEFT_INDEX if side == "L" else _FINGER_GROUP_RIGHT_INDEX
            other_group = _FINGER_GROUP_LEFT_OTHER if side == "L" else _FINGER_GROUP_RIGHT_OTHER
            for suffix, value in finger_names_and_values:
                joint = _find_suffix(self.model.joint_label, f"{side}_{suffix}")
                finger_indices.append(int(q_starts[joint]))
                closed_values.append(value)
                if suffix.startswith("thumb_"):
                    finger_groups.append(thumb_group)
                elif suffix.startswith("index_"):
                    finger_groups.append(index_group)
                else:
                    finger_groups.append(other_group)
        self.finger_indices = wp.array(finger_indices, dtype=wp.int32, device=self.model.device)
        self.closed_finger_values = wp.array(closed_values, dtype=float, device=self.model.device)
        self.finger_groups = wp.array(finger_groups, dtype=wp.int32, device=self.model.device)
        self.finger_fractions = wp.zeros(6, dtype=float, device=self.model.device)

    def _solve_ik(self, positions: np.ndarray, fractions: np.ndarray, iterations: int = 24):
        for objective, position in zip(self.position_objectives, positions, strict=True):
            objective.set_target_position(0, wp.vec3(*position))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=iterations)
        self.finger_fractions.assign(np.asarray(fractions, dtype=np.float32))
        wp.launch(
            set_finger_targets,
            dim=self.finger_indices.shape[0],
            inputs=[
                self.ik_joint_q_flat,
                self.finger_indices,
                self.closed_finger_values,
                self.finger_groups,
                self.finger_fractions,
            ],
        )

    def _update_trajectory(self):
        t = self.sim_time
        tr = self.tracks
        positions = np.asarray([tr["left_pos"].sample(t), tr["right_pos"].sample(t)], dtype=np.float32)
        # fraction order matches the _FINGER_GROUP_* constants
        fractions = np.asarray(
            [
                tr["left_thumb"].sample(t),
                tr["right_thumb"].sample(t),
                tr["left_index"].sample(t),
                tr["right_index"].sample(t),
                tr["left_other"].sample(t),
                tr["right_other"].sample(t),
            ],
            dtype=np.float32,
        ).reshape(-1)
        self.target_hand_positions = positions
        for time, name in self._phase_marks:
            if t >= time:
                self.phase = name
        self._solve_ik(positions, fractions)
        wp.launch(
            update_control_targets,
            dim=self.robot_coord_count,
            inputs=[
                self.ik_joint_q_flat,
                self.previous_joint_targets,
                1.0 / self.frame_dt,
                self.params["joint_target_velocity_limit"],
            ],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
        )

    # ── simulation loop ──────────────────────────────────────────────────

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
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if not self._hook_planned and self.sim_time >= self.params["settle_time"] - 0.5 * self.frame_dt:
            self._plan_hook()
        self._update_trajectory()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Phase: {self.phase}")
        ui.text(f"t = {self.sim_time:.2f} / {self.total_time:.2f} s")

    def _hand_pinch_positions(self) -> dict[str, np.ndarray]:
        body_q = self.state_0.body_q.numpy()
        out = {}
        for side, body, offset in zip(("left", "right"), self.hand_bodies, self.hand_offsets, strict=True):
            out[side] = np.asarray(wp.transform_point(wp.transform(*body_q[body]), offset), dtype=np.float64)
        return out

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Cloth state contains non-finite values"
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"
        assert self._hook_planned, "The hook was never planned (run did not reach the settle end)"

        # the bag left the trash can: its lowest particle hangs above the rim
        bag = particle_q[self.info["bag_start"] : self.info["bag_start"] + self.info["bag_count"]]
        bag_min_z = float(bag[:, 2].min())
        assert bag_min_z > self.info["rim_z"] - 0.01, (
            f"Bag did not leave the can: lowest particle z={bag_min_z:.3f}, rim z={self.info['rim_z']:.3f}"
        )

        # both hands still carry their handle
        pinches = self._hand_pinch_positions()
        for asset_side in ("left", "right"):
            verts = particle_q[self.info["handle_indices"][asset_side]]
            dists = {side: float(np.min(np.linalg.norm(verts - pinch, axis=1))) for side, pinch in pinches.items()}
            assert min(dists.values()) < 0.12, f"The {asset_side} handle slipped off both hands (distances {dists})"

        # the trash rides inside the lifted bag
        bag_centroid = bag.mean(axis=0)
        for body in self.info["trash_bodies"]:
            pos = body_q[body, :3]
            assert pos[2] > self.info["rim_z"] - 0.06, f"A trash sphere fell out of the lifted bag: z={pos[2]:.3f}"
            xy_err = float(np.linalg.norm(pos[:2] - bag_centroid[:2]))
            assert xy_err < 0.20, f"A trash sphere escaped the bag: {xy_err:.3f} m from the bag centroid"

        # torso stayed put
        torso_q = body_q[self.torso_body]
        torso_position_error = float(np.linalg.norm(torso_q[:3] - self.torso_initial_transform[:3]))
        assert torso_position_error < 0.01, f"H1 torso translated too far: {torso_position_error:.3f} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--seed", type=int, default=PARAMS["seed"])
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
