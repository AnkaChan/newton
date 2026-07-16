# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Trash Bag H1 Side-Grab Test
#
# SIDE-GRAB EXPERIMENT: only the humanoid and the bagged trash can (with
# the two baked-in spheres). The scene is identical to
# example_vbd_trash_bag_h1_pickup.py and starts from the same baked
# initial state (asset/trash_bag_pickup_init_state.npz, bake it with the
# pickup demo's --bake).
#
# Unlike the hook variant (example_vbd_trash_bag_h1_grab_test.py, which
# slides palm-up hooks UNDER the rope), both hands here grab the hanging
# drawstring handles FROM THE SIDE, following the pickup demo's palmar
# coke grasp: handshake orientation, palm pressed to the strand from
# outside, fingers and thumb close into a fist around the rope. The fist
# pins the rope in every pull direction, then both hands lift the bag out
# of the bucket.
#
# The grab targets are NOT fixed positions: they are computed at runtime
# from the measured position of the rope strands sticking out of the
# bag's holes.
#
# Command: python -m newton.examples vbd_trash_bag_h1_side_grab_test
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
    "rope_edge_kd": 0.05,
    # --- springs ---
    "closure_ke": 2.0e4,  # tunnel closure: flap free edge <-> wall
    "closure_kd": 1.0e-3,
    "closure_rest_length": 0.0,
    # --- contacts, identical to the standalone demo ---
    "soft_contact_ke": 2.0e5,
    "soft_contact_kd": 2.0e1,
    "soft_contact_mu": 0.3,
    "soft_contact_creation_margin": 0.012,
    "particle_self_contact_radius": 0.0025,
    "particle_self_contact_margin": 0.005,
    "rigid_body_particle_contact_buffer_size": 16384,
    "rigid_body_contact_buffer_size": 1024,
    "rigid_contact_hard": True,
    "rigid_contact_max": 4096,
    "enable_water_tight": True,  # water-tight rigid-soft SDF contacts (no tunneling through the thin can)
    # --- the bag-handle SIDE grab (the pickup demo's palmar coke grasp
    # applied to the rope). All positions are computed at runtime from the
    # measured rope strands hanging out of the bag's holes; the offsets
    # below are relative to the measured strand. ---
    # approach standoff outside the strand (along +y for the left handle)
    "grab_standoff_y": 0.10,
    # press the palm this far PAST the strand centre so the closing fist
    # traps the rope against the palm (the strand bight hangs ~5 cm clear
    # of the can wall, so a small press keeps the fingers off the can)
    "grab_press_y": 0.02,
    "grab_dx": 0.0,
    # grab height above the handle's lowest point (the bight): higher on
    # the strand shortens the rope from fist to mouth (raising the lift
    # ceiling), but too high presses the fingers into the can wall
    "grab_above_bight": 0.04,
    # fist closure fractions — the rope is ~6 mm, so close fully (the
    # 58 mm coke can used 0.75)
    "grab_close_index": 1.0,
    "grab_close_other": 1.0,
    "grab_close_thumb": 1.0,
    # both hands lift up AND CONVERGE over the can mouth: the fist pins
    # the rope, so the strand from fist to bag-hole is a fixed-length
    # tether — lifting straight up just swings the hand sideways on that
    # tether (pendulum around the hole). Keeping the strand vertical above
    # its hole sends the pull straight down the rope: the mouth cinches,
    # then the jam creeps while the bag rises. Keep the end target under
    # the ~1.75 IK ceiling mapped in example_vbd_h1_reach_test.py.
    # CINCH FIRST: pull the handles apart horizontally before lifting —
    # the spread reels the drawstring through the tunnels and closes the
    # mouth (how a human shuts a trash bag), so the lift carries a
    # compact closed sack
    "grab_cinch_out": 0.12,
    "grab_cinch_time": 2.0,
    # stop AT the material wall (~mouth over the rim): commanding higher
    # grinds the maxed drives until the arms collapse sideways
    "grab_lift_height": 0.28,
    "grab_lift_time": 6.0,
    # y magnitude the hands converge to (just above the bag holes at the
    # mouth rim, |y| ~0.11)
    "grab_lift_converge_y": 0.10,
    # hand-over-hand: alternate small steps (one hand holds while the
    # other reels) so the jammed loop never becomes a two-hand tug-of-war
    "grab_lift_steps": 3,
    # after the cinch the bag hangs as a compact CLOSED sack, so extract
    # it the human way: raise it straight up as high as the arms safely
    # go, then CARRY it horizontally toward the robot — the hanging bag
    # slides out over the near rim (a closed mouth cannot sock-flow over
    # the rim, so the old sweep/spill-while-low legs do nothing)
    "grab_raise_up": 0.25,
    "grab_raise_time": 5.0,
    "grab_carry_back": 0.40,
    "grab_carry_time": 4.0,
    "grab_hold_time": 3.0,
    # ONE-WAY COUPLING hack: once the fists are closed the robot is
    # switched KINEMATIC — its bodies follow the (velocity-clamped) IK
    # solution exactly via per-frame FK and the rope/bag can push back on
    # nothing. No tug-of-war, no arm sag, no drive-vs-rope explosion; the
    # trade is that rope tension no longer limits what the arms "can" do.
    "one_way_coupling": True,
    # ...and the rope is PINNED to the fists at the same moment: handle
    # particles within this radius of each pinch get zero inverse mass and
    # ride rigidly with the hand — the grip can never slip. The spheres in
    # the bag stay fully simulated.
    "pin_rope": True,
    # the closed fist is ~8 cm across and the strand can sit ~4 cm from
    # the pinch point after the press herds it. If nothing lands inside
    # the radius (the press herds the strand differently run to run), the
    # nearest strand chunk within the hard cap is pinned instead.
    "pin_rope_radius": 0.05,
    "pin_rope_radius_max": 0.09,
    "pin_rope_fallback_count": 21,  # 7 centerline nodes x 3 width verts
    # task-space rest poses (dish-washing scene)
    "rest_left": (-0.48, 0.24, 1.24),
    "rest_right": (-0.48, -0.24, 1.24),
    # durations [s]
    # tiny start buffer only — the baked initial state is already settled
    "settle_time": 0.1,
    "approach_time": 0.5,
    "descend_time": 0.7,  # the slide-in under the rope
    "close_time": 1.6,  # the raise that catches the strand
    "dwell_time": 0.3,
    "lift_time": 0.5,
    "carry_time": 2.4,  # the loaded lift — slow, a fast pull whips the bag
    # per-frame clamp on joint-target motion: tight enough that an IK
    # branch flip (~1.3 rad in one frame) glides over ~0.4 s instead of
    # hammering the stiff articulated solve
    "joint_target_velocity_limit": 3.0,
    # hand targets are clamped to this radius around the shoulder (keeps
    # deep-infeasible commands from grinding the IK at its boundaries)
    "max_hand_reach": 0.57,
    # the IK sees shoulder/elbow limits shrunk by this margin [rad], so its
    # solution can never park a drive target exactly at a mechanical stop
    "ik_limit_margin": 0.10,
    # elbow-shaping target relative to the shoulder (x fwd, y OUTWARD —
    # mirrored per side, z up): biases the redundant arm's elbow out-and-
    # down instead of folding inward
    "elbow_target_offset": (0.05, 0.22, -0.20),
    "torso_ik_position_weight": 50.0,
    "torso_ik_rotation_weight": 50.0,
    # --- solver / time, identical to the standalone demo ---
    "fps": 60,
    "sim_substeps": 10,
    "solver_iterations": 12,
    "gravity": -9.8,
    "bake_seconds": 10.0,
    "num_frames": 480,
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
    # --- H1 (dish-washing scene values). The robot stands CLOSER than the
    # dish demo's -0.70: the handle slide-under points at the can's far
    # sides are ~0.71 m from the shoulders at -0.70 — the arms' reach
    # limit, where the IK saturates and the arms look stuck ---
    "robot_base_x": -0.58,
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
    # AVBD joint drives. Rope tension in the haul equals what the arms can
    # pull — if strands start popping out of the hooks or the solve
    # explodes, this is the first knob to turn back DOWN (5e5 popped
    # strands under the old choreography)
    "joint_drive_ke": 1e6,
    "joint_drive_kd": 2.5e3,
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

# Pinch-point offsets in the hand-link frames, calibrated for the H1 hand
# meshes (shared with the dish-washing demo).
HAND_OFFSETS = (
    (0.146273, -0.068447, 0.028077),
    (0.148808, 0.068652, 0.026675),
)

# Thumb-up "handshake" for both hands (identity yaw + small pitch, the
# pickup demo's palmar-grasp orientation): each palm faces the robot's
# centreline, so a hand approaching its strand from OUTSIDE presses the
# palm flat against the rope.
_HAND_PITCH = 0.45
HAND_ROTATIONS = (
    (0.0, math.sin(0.5 * _HAND_PITCH), 0.0, math.cos(0.5 * _HAND_PITCH)),
    (0.0, math.sin(0.5 * _HAND_PITCH), 0.0, math.cos(0.5 * _HAND_PITCH)),
)

# Side-specific "closed" thumb angles (yaw, pitch, intermediate, distal):
# full yaw + curl — driving the fraction sweeps the thumb across the palm
# and curls it over the closing fingers, locking the rope in the fist.
# Left: the dish-washing tip-opposition calibration; right mirrors it with
# the right hand's yaw calibration.
THUMB_CLOSED_VALUES = (
    (1.273907, 0.160957, 0.369535, 0.892908),
    (1.192278, 0.160957, 0.369535, 0.892908),
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
def drive_pinned_particles(
    body_q: wp.array[wp.transform],
    pin_body: wp.array[wp.int32],
    pin_offset: wp.array[wp.vec3],
    pin_index: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
):
    i = wp.tid()
    particle_q[pin_index[i]] = wp.transform_point(body_q[pin_body[i]], pin_offset[i])


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


def _normalized_quat(values):
    q = np.asarray(values, dtype=np.float32)
    q /= np.linalg.norm(q)
    return wp.quat(*q)


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

    def pos(self) -> np.ndarray:
        return self._tracks[f"{self._side}_pos"].sample(self.time)


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
        "left_elbow": "left_elbow_link",
        "right_elbow": "right_elbow_link",
        "left_shoulder": "left_shoulder_pitch_link",
        "right_shoulder": "right_shoulder_pitch_link",
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
    hook_shapes = []
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
            hook_shapes.append(shape)
            hook_shape_count += 1

    # 2 palm geoms + 4 two-link finger chains per hand = 10 colliders per side
    if hook_shape_count != 20:
        raise RuntimeError(f"Expected 20 H1 palm/index..pinky colliders, found {hook_shape_count}")
    return body_indices, robot_rigid_shapes, hook_shapes


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
    robot_bodies, robot_rigid_shapes, hook_shapes = _add_h1(builder, params)
    robot_coord_count = builder.joint_coord_count
    robot_body_count = builder.body_count
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

    # --- ground the robot stands on ---
    ground_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["table_mu"],
        gap=params["rigid_contact_gap"],
    )
    ground_shape = builder.add_ground_plane(cfg=ground_cfg)

    # filter the robot against the furniture and the loose trash so a limb
    # brushing static geometry cannot disturb the AVBD solve — EXCEPT the
    # palm/finger colliders vs the trash can: the hands work right against
    # the bucket wall while hooking the handles and would otherwise
    # penetrate it visually.
    hook_set = set(hook_shapes)
    for robot_shape in robot_rigid_shapes:
        for other in (*table_shapes, ground_shape, *apple_shapes):
            builder.add_shape_collision_filter_pair(robot_shape, other)
        if robot_shape not in hook_set:
            builder.add_shape_collision_filter_pair(robot_shape, can_shape)

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
        "z_floor": z_floor + offset[2],
        "num_tunnel_springs": len(layout["tunnel_spring_pairs"]),
        "tunnel_spring_pairs": tunnel_spring_pairs,
        "robot_bodies": robot_bodies,
        "robot_coord_count": robot_coord_count,
        "robot_body_count": robot_body_count,
        "can_center": np.asarray([params["can_x"], params["can_y"]], dtype=np.float64),
    }


def finalize_model(builder, params):
    builder.enable_rigid_mesh_sdfs()
    model = builder.finalize()
    model.soft_contact_ke = params["soft_contact_ke"]
    model.soft_contact_kd = params["soft_contact_kd"]
    model.soft_contact_mu = params["soft_contact_mu"]
    return model


def setup_sim(model, info, params):
    """Create the solver and collision pipeline.

    Must be called AFTER the robot's joint_q has been posed (IK rest) and
    eval_fk has updated model.body_q: the solver bakes its structural joint
    data from the model pose at construction, and its stiff
    rigid_joint_angular_ke would otherwise drag the arms back to the pose it
    saw here no matter what the (much softer) drives command."""
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
        # the auto-estimate (~1190) overflows once the kinematic robot
        # presses into the scene
        rigid_contact_max=params["rigid_contact_max"],
    )
    return solver, pipeline


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
                print(f"[trash_bag_h1_side_grab_test] starting from baked state {STATE_NPZ}")
            else:
                print(
                    f"[trash_bag_h1_side_grab_test] no baked state at {STATE_NPZ} — starting from the raw "
                    "assets (run once with --bake to settle and save the initial state)"
                )
        builder = newton.ModelBuilder(gravity=p["gravity"])
        self.info = build_model(builder, p, seed=seed, baked_state=baked_state)
        self.model = finalize_model(builder, p)
        self.device = self.model.device
        self.robot_coord_count = self.info["robot_coord_count"]

        self.hand_bodies = [self.info["robot_bodies"]["left_hand"], self.info["robot_bodies"]["right_hand"]]
        self.hand_offsets = [wp.vec3(*values) for values in HAND_OFFSETS]
        self.hand_rotations = [_normalized_quat(values) for values in HAND_ROTATIONS]

        self.phase = "settle"
        self._phase_marks: list[tuple[float, str]] = []
        self._grab_planned = False
        self._robot_kinematic = False
        self._kinematic_switch_time = float("inf")
        self._pin_count = 0
        self._build_choreography()
        self._setup_ik()
        # start the robot in the task-space rest pose (arms forward at the table)
        self._solve_ik(
            np.asarray([p["rest_left"], p["rest_right"]], dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            iterations=48,
        )
        self.model.joint_q.assign(self.ik_joint_q_flat)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # the solver must be created AFTER the pose assignment above (see
        # the setup_sim docstring)
        self.solver, self.pipeline = setup_sim(self.model, self.info, p)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)
        wp.copy(self.control.joint_target_q, self.model.joint_q, count=self.robot_coord_count)
        self.control.joint_target_qd.zero_()
        self.previous_joint_targets = wp.clone(self.model.joint_q[: self.robot_coord_count])
        self.torso_initial_position = self.state_0.body_q.numpy()[self.info["robot_bodies"]["torso"], :3].copy()

        print(
            f"[trash_bag_h1_side_grab_test] bag verts {self.info['bag_count']}  rope verts {self.info['rope_count']}  "
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

    # ── choreography ─────────────────────────────────────────────────────

    def _mark(self, time: float, name: str):
        self._phase_marks.append((time, name))

    def _build_choreography(self):
        """Initialize the keyframe tracks (hands at rest). The hook grab is
        planned at runtime in _plan_grab, from the measured rope position."""
        p = self.params
        tracks: dict[str, _Track] = {}
        for side, rest in (("left", p["rest_left"]), ("right", p["rest_right"])):
            tracks[f"{side}_pos"] = _Track(np.asarray(rest, dtype=np.float64))
            tracks[f"{side}_thumb"] = _Track(0.0)
            tracks[f"{side}_index"] = _Track(0.0)
            tracks[f"{side}_other"] = _Track(0.0)
        self.tracks = tracks
        # durations are fixed; only the positions wait for the measurement
        self.total_time = (
            p["settle_time"]
            + p["approach_time"]
            + p["descend_time"]
            + p["close_time"]
            + p["dwell_time"]
            + p["grab_cinch_time"]
            + p["grab_lift_time"]
            + p["grab_raise_time"]
            + p["grab_carry_time"]
            + p["grab_hold_time"]
            + 0.8
        )

    def _plan_grab(self):
        """Plan the side grab from the MEASURED positions of the rope
        handles hanging out of the bag's holes (never from fixed
        coordinates).

        BOTH hands in handshake orientation: each approaches its side's
        strand from OUTSIDE, presses the palm onto the rope, closes the
        fingers and thumb into a fist around it, then both lift the bag
        hand-over-hand until it leaves the bucket.
        """
        p = self.params
        particle_q = self.state_0.particle_q.numpy()
        can_xy = self.info["can_center"]
        t0 = self.sim_time
        self._mark(t0, "approach")

        planned_sides = set()
        for idx_key in ("left_idx", "right_idx"):
            verts = particle_q[self.info[idx_key]]
            # centerline polyline (56 nodes x 3 width verts, node-major)
            nodes = verts.reshape(-1, 3, 3).mean(axis=1).astype(np.float64)
            bight = nodes[int(np.argmin(nodes[:, 2]))]
            # which hand: the handle hangs on that side of the can
            hand_side = "left" if float(nodes[:, 1].mean()) > can_xy[1] else "right"
            side_sign = 1.0 if hand_side == "left" else -1.0
            planned_sides.add(hand_side)

            grab_x = float(bight[0]) + p["grab_dx"]
            grab_z = float(bight[2]) + p["grab_above_bight"]
            # grab the strand where it actually hangs at that height: the
            # node nearest the grab height on the hanging handle
            near = nodes[np.argsort(np.abs(nodes[:, 2] - grab_z))[:4]]
            strand_y = float(near[np.argmax(np.abs(near[:, 1]))][1])
            print(
                f"[trash_bag_h1_side_grab_test] {hand_side} rope bight at "
                f"({bight[0]:+.3f}, {bight[1]:+.3f}, {bight[2]:.3f}); grabbing at "
                f"({grab_x:+.3f}, {strand_y:+.3f}, {grab_z:.3f})"
            )

            standoff_y = strand_y + side_sign * p["grab_standoff_y"]
            press_y = strand_y - side_sign * p["grab_press_y"]
            cur = _HandCursor(self.tracks, hand_side)
            cur.time = t0
            # approach from outside at grab height, hand open (handshake)
            cur.move(p["approach_time"], pos=(grab_x, standoff_y, grab_z))
            if hand_side == "right":
                self._mark(cur.time, "press")
            # press the palm onto the strand
            cur.move(p["descend_time"], pos=(grab_x, press_y, grab_z))
            if hand_side == "right":
                self._mark(cur.time, "close")
            # close fingers and thumb into a fist around the rope
            cur.move(
                p["close_time"],
                pos=(grab_x, press_y, grab_z),
                index=p["grab_close_index"],
                other=p["grab_close_other"],
                thumb=p["grab_close_thumb"],
            )
            cur.wait(p["dwell_time"])
            if hand_side == "right":
                self._mark(cur.time, "cinch")
                # both fists are closed by now — from here the robot can go
                # one-way (kinematic)
                self._kinematic_switch_time = cur.time
            # pull the handles APART horizontally first: the spread reels
            # the drawstring through the tunnels and cinches the mouth
            # shut before anything lifts
            cinch_y = press_y + side_sign * p["grab_cinch_out"]
            cur.move(p["grab_cinch_time"], pos=(grab_x, cinch_y, grab_z))
            if hand_side == "right":
                self._mark(cur.time, "lift")
            # lift hand-over-hand: alternating steps so the jammed
            # drawstring never becomes a two-hand tug-of-war. The hands
            # converge over the can mouth so the fist-to-hole strand stays
            # vertical (see grab_lift_converge_y)
            lift_from = np.array((grab_x, cinch_y, grab_z))
            lift_to = np.array(
                (float(can_xy[0]), side_sign * p["grab_lift_converge_y"], grab_z + p["grab_lift_height"])
            )
            n_steps = int(p["grab_lift_steps"])
            step_time = p["grab_lift_time"] / (2 * n_steps)
            for k in range(2 * n_steps):
                if (k % 2 == 0) == (hand_side == "left"):
                    frac = (k // 2 + 1) / n_steps
                    cur.move(step_time, pos=tuple(lift_from + frac * (lift_to - lift_from)))
                else:
                    cur.wait(step_time)
            if hand_side == "right":
                self._mark(cur.time, "raise")
            # raise the closed sack straight up as high as the arms safely
            # go...
            lift_y = float(lift_to[1])
            raise_z = float(lift_to[2]) + p["grab_raise_up"]
            cur.move(p["grab_raise_time"], pos=(float(can_xy[0]), lift_y, raise_z))
            if hand_side == "right":
                self._mark(cur.time, "carry")
            # ...then carry it horizontally toward the robot: the hanging
            # bag slides out over the near rim
            carry_x = float(can_xy[0]) - p["grab_carry_back"]
            cur.move(p["grab_carry_time"], pos=(carry_x, lift_y, raise_z))
            if hand_side == "right":
                self._mark(cur.time, "hold")
            cur.wait(p["grab_hold_time"])
            if hand_side == "right":
                self._mark(cur.time, "done")

        if planned_sides != {"left", "right"}:
            raise RuntimeError(f"Expected one handle per side, got hands {sorted(planned_sides)}")
        self._grab_planned = True

    def _pin_rope_to_fists(self):
        """Zero the inverse mass of the handle particles currently inside
        each fist and record their offsets in the hand frame — from here
        they ride rigidly with the (kinematic) hands. The bag, the rest of
        the rope, and the rigid spheres stay fully simulated."""
        p = self.params
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        can_xy = self.info["can_center"]
        pin_body, pin_index, pin_offset = [], [], []
        for idx_key in ("left_idx", "right_idx"):
            handle_idx = np.asarray(self.info[idx_key], dtype=np.int32)
            hand_side = "left" if float(particle_q[handle_idx][:, 1].mean()) > can_xy[1] else "right"
            hand_i = 0 if hand_side == "left" else 1
            body = self.hand_bodies[hand_i]
            tf = wp.transform(*body_q[body])
            pinch = np.asarray(wp.transform_point(tf, self.hand_offsets[hand_i]))
            dist = np.linalg.norm(particle_q[handle_idx] - pinch, axis=1)
            selected = handle_idx[dist < p["pin_rope_radius"]]
            if len(selected) == 0:
                # the press herds the strand a few cm run to run — fall
                # back to the nearest strand chunk within the hard cap
                order = np.argsort(dist)[: p["pin_rope_fallback_count"]]
                order = order[dist[order] < p["pin_rope_radius_max"]]
                selected = handle_idx[order]
                print(
                    f"[trash_bag_h1_side_grab_test] {hand_side} fist: no strand within "
                    f"{p['pin_rope_radius']:.3f}, falling back to the nearest chunk "
                    f"(min dist {dist.min():.3f})"
                )
            inv_tf = wp.transform_inverse(tf)
            for pi in selected:
                pin_body.append(int(body))
                pin_index.append(int(pi))
                pin_offset.append(np.asarray(wp.transform_point(inv_tf, wp.vec3(*particle_q[pi]))))
            print(f"[trash_bag_h1_side_grab_test] pinned {len(selected)} rope particles to the {hand_side} fist")
            if len(selected) == 0:
                raise RuntimeError(f"No rope particles near the {hand_side} fist to pin — grab missed?")
        inv_mass = self.model.particle_inv_mass.numpy()
        inv_mass[pin_index] = 0.0
        self.model.particle_inv_mass.assign(inv_mass)
        self._pin_body = wp.array(pin_body, dtype=wp.int32, device=self.device)
        self._pin_index = wp.array(pin_index, dtype=wp.int32, device=self.device)
        self._pin_offset = wp.array(np.asarray(pin_offset, dtype=np.float32), dtype=wp.vec3, device=self.device)
        self._pin_count = len(pin_index)

    # ── IK (dish-washing rig) ────────────────────────────────────────────

    def _setup_ik(self):
        initial_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, initial_state)
        body_q = initial_state.body_q.numpy()

        self.torso_body = self.info["robot_bodies"]["torso"]
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

        # weak elbow-shaping objectives: the palm-up orientation on a 4-DOF
        # arm (no wrist) lets the LM solver swing the elbows through weird
        # branches — a gentle pull toward a natural down-and-out elbow pose
        # keeps the redundancy resolved consistently. Keep the target FIXED:
        # re-aiming it at the hand each frame makes the IK flip branches
        # (~1.3 rad single-frame jumps) near high targets.
        self.elbow_objectives = []
        self.shoulder_positions = []
        off = self.params["elbow_target_offset"]
        for side, sign in (("left", 1.0), ("right", -1.0)):
            shoulder_pos = np.asarray(body_q[self.info["robot_bodies"][f"{side}_shoulder"]][:3], dtype=np.float64)
            self.shoulder_positions.append((shoulder_pos, sign))
            elbow_target = shoulder_pos + np.asarray([off[0], sign * off[1], off[2]])
            self.elbow_objectives.append(
                ik.IKObjectivePosition(
                    link_index=self.info["robot_bodies"][f"{side}_elbow"],
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.vec3(*elbow_target)], dtype=wp.vec3),
                    weight=0.3,
                )
            )

        # weight 5.0: at 1.0 the solver happily hyper-extends the elbows a
        # radian past their limits, which reads as very strange arm motion.
        # The arm-joint limits are shrunk by a margin: if the IK parks a
        # joint target exactly AT a mechanical stop (e.g. shoulder pitch at
        # -2.87 reaching overhead), the solver's stiff limit constraint and
        # the drive fight right at the boundary and the solve explodes.
        ik_lo = self.model.joint_limit_lower.numpy().copy()
        ik_hi = self.model.joint_limit_upper.numpy().copy()
        qd_starts = self.model.joint_qd_start.numpy()
        for j, label in enumerate(self.model.joint_label):
            if "shoulder" in label or "elbow" in label:
                d = int(qd_starts[j])
                ik_lo[d] += self.params["ik_limit_margin"]
                ik_hi[d] -= self.params["ik_limit_margin"]
        joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(ik_lo, dtype=float),
            joint_limit_upper=wp.array(ik_hi, dtype=float),
            weight=5.0,
        )
        self.ik_joint_q = wp.clone(self.model.joint_q).reshape((1, self.model.joint_coord_count))
        self.ik_joint_q_flat = self.ik_joint_q.reshape((-1,))
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[
                *self.position_objectives,
                *self.rotation_objectives,
                *self.elbow_objectives,
                self.torso_position_objective,
                self.torso_rotation_objective,
                joint_limits,
            ],
            # heavier LM damping: the per-frame warm-started solve makes
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
        for i, (objective, position) in enumerate(zip(self.position_objectives, positions, strict=True)):
            # clamp the target inside the arm's SAFE envelope: past ~86% of
            # the 0.71 m reach the elbow approaches the straight-arm
            # singularity and the stiff articulated solve explodes (verified
            # robot-only in example_vbd_h1_reach_test.py: z 1.73 tracks with
            # 4 mm error, z 1.80 detonates even with smooth IK targets)
            shoulder, _sign = self.shoulder_positions[i]
            target = np.asarray(position, dtype=np.float64)
            rel = target - shoulder
            dist = float(np.linalg.norm(rel))
            max_reach = self.params["max_hand_reach"]
            if dist > max_reach:
                target = shoulder + rel * (max_reach / dist)
            objective.set_target_position(0, wp.vec3(*target))
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
        if self._robot_kinematic:
            # one-way coupling: the kinematic robot follows the smoothed
            # joint targets exactly — pose its bodies directly via FK.
            # CRITICAL: filter to KINEMATIC bodies only. add_body gives the
            # apples free-joint articulations, so an unfiltered eval_fk
            # teleports them back to their build poses every frame — an
            # energy pump that NaNs them within a second.
            wp.copy(self.model.joint_q, self.previous_joint_targets, count=self.robot_coord_count)
            newton.eval_fk(
                self.model,
                self.model.joint_q,
                self.model.joint_qd,
                self.state_0,
                body_flag_filter=int(newton.BodyFlags.KINEMATIC),
            )
            if self._pin_count:
                # the pinned rope particles ride rigidly with the fists
                wp.launch(
                    drive_pinned_particles,
                    dim=self._pin_count,
                    inputs=[
                        self.state_0.body_q,
                        self._pin_body,
                        self._pin_offset,
                        self._pin_index,
                        self.state_0.particle_q,
                    ],
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
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        # plan the grab from the measured rope position at the settle end
        if not self.bake and not self._grab_planned and self.sim_time >= self.params["settle_time"]:
            self._plan_grab()
        # one-time switch: once both fists are closed, the robot goes
        # KINEMATIC (one-way coupled). The flag refresh updates the
        # solver's effective inverse-mass arrays in place, so the captured
        # graph picks it up on the next replay.
        if (
            self.params["one_way_coupling"]
            and not self._robot_kinematic
            and self._grab_planned
            and self.sim_time >= self._kinematic_switch_time
        ):
            body_flags = self.model.body_flags.numpy()
            body_flags[: self.info["robot_body_count"]] = int(newton.BodyFlags.KINEMATIC)
            self.model.body_flags.assign(body_flags)
            self.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            self._robot_kinematic = True
            print(f"[trash_bag_h1_side_grab_test] t={self.sim_time:.2f}: robot switched to one-way (kinematic)")
            if self.params["pin_rope"]:
                self._pin_rope_to_fists()
        # during --bake the robot just holds its rest pose
        if not self.bake:
            self._update_trajectory()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def gui(self, ui):
        ui.text(f"Phase: {self.phase}")
        ui.text(f"t = {self.sim_time:.2f} / {self.total_time:.2f} s")

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
        print(f"[trash_bag_h1_side_grab_test] baked state saved to {STATE_NPZ}")

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
        # the spheres stay roughly within the bag (no escape / explosion)
        if self.info["body_indices"]:
            trash_pos = body_q[self.info["body_indices"]][:, :3]
            assert np.all(np.abs(trash_pos[:, 0] - p["can_x"]) < 0.4), "A trash object escaped in x"
            assert np.all(np.abs(trash_pos[:, 1] - p["can_y"]) < 0.3), "A trash object escaped in y"
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
        print(f"[trash_bag_h1_side_grab_test] baking: settling for {bake_frames} frames ...")
        for frame in range(bake_frames):
            example.step()
            if (frame + 1) % 120 == 0:
                print(f"[trash_bag_h1_side_grab_test]   bake frame {frame + 1}/{bake_frames}")
        example.save_baked_state()
    else:
        newton.examples.run(example, args)
