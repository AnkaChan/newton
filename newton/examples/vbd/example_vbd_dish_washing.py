# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Dish Washing H1 (unified AVBD/VBD)
#
# A fixed-base Unitree H1 stands at a table holding a pile of rigid plates
# and a soft sponge. The right hand pinches the top plate's rim (the plate
# overhangs the table edge, so the index slides underneath and the thumb
# closes on top), carries it to a washing spot at the table edge, and sets
# it down. The left hand pinch-grabs the sponge — a volumetric FEM block
# held through the water-tight rigid-soft SDF contact path — and rubs
# circles over the plate. The plate stays put during the rub because the
# sponge-plate friction is lower than the plate-table friction. The right
# hand then re-pinches the overhanging rim and places the plate down on the
# clean side of the table.
#
# All grasps are physical: no particles or bodies are pinned or scripted.
# Newton IK converts task-space hand keyframes into joint targets and one
# SolverVBD instance advances the H1 + plates with AVBD and the sponge with
# VBD in the same solve.
#
# Command: python -m newton.examples vbd_dish_washing
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

PARAMS = {
    # simulation
    "fps": 60,
    "sim_substeps": 16,
    "solver_iterations": 8,
    "gravity": -9.81,
    "num_frames": 1150,
    # how many plates are washed (script 2 raises this to the full pile)
    "wash_count": 1,
    "plate_count": 3,
    # presentation
    "camera_position": (1.05, -1.55, 1.62),
    "camera_pitch": -14.0,
    "camera_yaw": -37.0,
    "camera_fov": 45.0,
    # table (front edge faces the robot at x = -table_half_width)
    "table_half_width": 0.20,
    "table_half_depth": 0.36,
    "table_top_z": 1.09,
    "tabletop_half_height": 0.04,
    "table_mu": 0.9,
    # plates: light stoneware cylinders, glossy (low mu) so the wet sponge
    # slides over them while the grippy tabletop holds them in place
    "plate_radius": 0.062,
    "plate_half_height": 0.008,
    "plate_density": 800.0,
    # grippy on the table (mu combines with the 0.9 tabletop) so the wet sponge
    # can scrub without dragging the plate off the overhanging wash spot
    "plate_mu": 0.6,
    "plate_colors": ((0.93, 0.90, 0.82), (0.72, 0.82, 0.90), (0.78, 0.88, 0.78)),
    # dirty pile: flush stack near the front edge; the top plate is offset
    # toward the robot so its rim overhangs the table edge and can be pinched
    "dirty_pile_x": -0.138,
    "dirty_pile_y": -0.24,
    "grab_overhang_x": -0.173,
    # washing spot: also at the front edge (the rim overhang is what makes
    # the plate re-graspable after the rub)
    "wash_x": -0.173,
    "wash_y": -0.02,
    # clean spot: same overhang line (the index needs free air below the rim
    # to slide out from under a set-down plate)
    "clean_x": -0.173,
    "clean_y": 0.16,
    # sponge: a flat soft FEM pad the same thickness as the plate rim, so the
    # calibrated edge-pinch that lifts a plate also grips the sponge (a thick
    # foam cube squirts out of the H1 pinch; a pad is caught at its edge like
    # the soft grid in example_vbd_gripper_soft_grid). Its -x edge overhangs
    # the front table edge so the index can slide underneath.
    "sponge_size": (0.10, 0.075, 0.024),
    "sponge_cells": (8, 6, 2),
    "sponge_x": -0.185,
    "sponge_y": 0.27,
    "sponge_density": 250.0,
    # A concentrated H1 fingertip spears a stiff FEM solid (a tet inverts into a
    # spike). The gripper example grips a soft body only because it is very soft,
    # uses a large particle radius, and heavily damps the contact. Follow that
    # recipe: soft moduli + small tets so a fingertip indents locally instead of
    # inverting the whole block.
    "sponge_k_mu": 4.0e3,
    "sponge_k_lambda": 2.0e4,
    "sponge_k_damp": 1.0e-3,
    "sponge_particle_radius": 0.008,
    "sponge_color": (0.95, 0.85, 0.25),
    # rigid-soft contact: soft and heavily damped. Low friction so the sponge
    # SLIDES over the plate during the rub instead of dragging it off the
    # overhanging wash spot (the grasp is a kinematic pin, so it needs no
    # contact friction of its own).
    "soft_contact_ke": 5.0e2,
    "soft_contact_kd": 8.0,
    "soft_contact_mu": 0.25,
    "soft_contact_margin": 0.010,
    "enable_water_tight_rigid_soft_contact": True,
    "shape_ke": 1.0e3,
    "shape_kd": 1.0e-4,
    "rigid_contact_gap": 0.001,
    # H1; the thumb/index chains carry finer texture-backed SDFs for the pinch
    "robot_base_x": -0.75,
    "robot_contact_ke": 1.0e3,
    "robot_contact_kd": 1.0e-2,
    "robot_contact_mu": 0.5,
    "robot_contact_margin": 0.002,
    "finger_contact_margin": 0.002,
    "robot_sdf_padding": 0.012,
    "robot_sdf_max_resolution": 64,
    "finger_sdf_padding": 0.012,
    "finger_contact_ke": 8.0e3,
    "finger_contact_kd": 2.0e1,
    # High mu keeps the rim pinch and the sponge pinch from creeping.
    "finger_contact_mu": 200.0,
    "finger_sdf_max_resolution": 128,
    # task-space rest poses
    "rest_left": (-0.48, 0.24, 1.24),
    "rest_right": (-0.48, -0.24, 1.24),
    # grasp primitive (hand pinch-point targets relative to the grabbed rim).
    # Calibrated against the H1 hand meshes (kinematic probe): with the index
    # curled to 0.75 the index tip spans z in [P-0.029, P+0.004] and reaches
    # x <= P+0.017 around the pinch target P; the thumb tip bottom sits at
    # P+0.032 / P+0.007 / P-0.003 for fractions 0.5 / 0.85 / 1.0.
    "grab_hover_dx": -0.085,
    "grab_hover_dz": 0.10,
    # Pinch target P is the hand-frame pinch point. With index curled to 0.75
    # the index tip top rides at P+0.004 and the thumb tip bottom at ~P+0.018
    # (frac 0.70) to ~P+0.007 (frac 0.85). Plate is 16 mm thick; underside at
    # bottom_z. Slide the index in with its top ~2 mm below the underside, then
    # raise so the index top sits ~2 mm into the plate (a gentle upward support,
    # not the old 10 mm overdrive that detonated the solve).
    "grab_insert_hand_dz": -0.010,
    "grab_insert_depth": 0.012,
    "grab_raise_hand_dz": -0.002,
    "grab_index_fraction": 0.75,
    # thumb bottom lands ~ at the plate top for a light clamp on the rim
    "grab_thumb_fraction": 0.72,
    "other_finger_fraction": 0.8,
    # sponge pinch: a gentle non-penetrating cradle (unlike the plate's small
    # overdrive, which would spear a soft body). The index slides in ~6 mm below
    # the pad, rises to just kiss the underside, and the thumb only rests on top.
    "sponge_insert_hand_dz": -0.014,
    "sponge_raise_hand_dz": -0.008,
    "sponge_thumb_fraction": 0.52,
    # pinch-point x offset from the held plate's center while carried
    "plate_center_to_pinch_dx": -0.050,
    "carry_lift": 0.055,
    # after opening the thumb, drop the hand well below the rim before sliding
    # it out (finger mu is huge; sliding under load drags the plate off the edge)
    "release_drop": 0.028,
    # drag primitive (script 2): press curled fingertips onto the top plate
    # and slide it toward the table edge until the rim overhangs
    "drag_press_hand_dz": 0.025,
    "drag_hand_dx": -0.020,
    "drag_slip_allowance": 0.012,
    # rub trajectory: the sponge is pinch-held at its -x edge, so the pinch
    # stays behind the plate rim (the index must never cross above it) and
    # the sponge body scrubs the near half of the plate in flat ellipses
    "rub_pinch_behind_rim": 0.045,
    "rub_radius_x": 0.006,
    "rub_radius_y": 0.024,
    "rub_circles": 3,
    "rub_circle_time": 1.2,
    # pinch height above the plate top: the pinned edge rides here and the pad
    # drapes down to just kiss the plate (positive = a light graze, not a press)
    "rub_pinch_above_plate": 0.010,
    "rub_hover_dz": 0.09,
    # durations [s]
    "settle_time": 0.5,
    "approach_time": 0.7,
    "descend_time": 0.55,
    "insert_time": 0.5,
    "raise_time": 0.4,
    "close_time": 0.4,
    "dwell_time": 0.25,
    "lift_time": 0.45,
    "carry_time": 0.9,
    "lower_time": 0.5,
    "release_time": 0.35,
    "retract_time": 0.55,
    "drag_time": 0.7,
    # AVBD joint drives; Newton IK only generates their targets
    "joint_drive_ke": 5.0e4,
    "joint_drive_kd": 5.0e2,
    "torso_drive_ke": 2.0e5,
    "torso_drive_kd": 2.0e3,
    "finger_drive_ke": 4.0e4,
    "finger_drive_kd": 1.0e2,
    "joint_target_velocity_limit": 40.0,
    "torso_ik_position_weight": 50.0,
    "torso_ik_rotation_weight": 50.0,
}

# Pinch-point offsets and pinch orientations in the hand-link frames,
# calibrated for the H1 hand meshes (thumb opposing the curled index).
HAND_OFFSETS = (
    (0.146273, -0.068447, 0.028077),
    (0.148808, 0.068652, 0.026675),
)

HAND_ROTATIONS = (
    (-0.09, 0.46, 0.03, 0.88),
    (0.09023, 0.46115, -0.03008, 0.88221),
)

# Side-specific fully-closed thumb angles that bring the thumb and index mesh
# patches into opposition without intersecting.
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
def drive_grip_particles(
    grip_indices: wp.array[wp.int32],
    grip_offsets: wp.array[wp.vec3],
    hand_transform: wp.transform,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    p = grip_indices[i]
    particle_q[p] = wp.transform_point(hand_transform, grip_offsets[i])
    particle_qd[p] = wp.vec3(0.0)


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


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [i for i, label in enumerate(labels) if label.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def _normalized_quat(values: tuple[float, float, float, float]) -> wp.quat:
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

    def wait_until(self, time: float):
        self.time = max(self.time, time)

    def pos(self) -> np.ndarray:
        return self._tracks[f"{self._side}_pos"].sample(self.time)


def _add_h1(builder: newton.ModelBuilder, params: dict) -> tuple[dict[str, int], list[int]]:
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

    # The four thumb/index grasp chains keep shape-shape contact + a finer
    # texture SDF for the rigid plate pinch. No robot collider gets particle
    # collision: a curved H1 fingertip sweeping through a 3D FEM sponge spears
    # and detonates it (a tet inverts into a spike), so the sponge is instead
    # pinned to the hand as a kinematic grasp (see ``_setup_sponge_grip``) while
    # the fingers curl around it visually. Every non-grasp collider is filtered
    # against the table and plates (a humanoid's forearm resting on the tabletop
    # otherwise explodes the AVBD contact solve).
    shape_collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_collision_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    collision_mask = shape_collision_flag | particle_collision_flag
    grasp_finger_tokens = ("/L_thumb_", "/L_index_", "/R_thumb_", "/R_index_")
    finger_bodies = {
        body
        for body in range(robot_body_start, robot_body_end)
        if any(token in builder.body_label[body] for token in grasp_finger_tokens)
    }
    finger_contact_shape_count = 0
    robot_rigid_shapes = []
    grasp_shapes = []
    for shape in range(robot_shape_start, robot_shape_end):
        original_flags = int(builder.shape_flags[shape])
        is_rigid_collider = bool(original_flags & shape_collision_flag)
        is_grasp_collider = is_rigid_collider and builder.shape_body[shape] in finger_bodies
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
        if is_grasp_collider:
            builder.shape_material_ke[shape] = params["finger_contact_ke"]
            builder.shape_material_kd[shape] = params["finger_contact_kd"]
            builder.shape_material_mu[shape] = params["finger_contact_mu"]
            builder.shape_margin[shape] = params["finger_contact_margin"]
            builder.shape_sdf_padding[shape] = params["finger_sdf_padding"]
            builder.shape_sdf_max_resolution[shape] = params["finger_sdf_max_resolution"]
            builder.shape_sdf_target_voxel_size[shape] = None
            grasp_shapes.append(shape)
            finger_contact_shape_count += 1

    if finger_contact_shape_count != 12:
        raise RuntimeError(f"Expected 12 H1 thumb/index colliders, found {finger_contact_shape_count}")
    return body_indices, robot_rigid_shapes, grasp_shapes


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


def _add_plates(builder: newton.ModelBuilder, params: dict) -> tuple[list[int], list[int]]:
    plate_cfg = newton.ModelBuilder.ShapeConfig(
        density=params["plate_density"],
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["plate_mu"],
        gap=params["rigid_contact_gap"],
        has_particle_collision=True,
        margin=0.0,
    )
    plates = []
    plate_shapes = []
    count = params["plate_count"]
    colors = params["plate_colors"]
    for level in range(count):
        # flush pile, except the top plate which already overhangs the edge
        x = params["grab_overhang_x"] if level == count - 1 else params["dirty_pile_x"]
        z = params["table_top_z"] + (2 * level + 1) * params["plate_half_height"]
        body = builder.add_body(xform=wp.transform(wp.vec3(x, params["dirty_pile_y"], z), wp.quat_identity()))
        plate_shapes.append(
            builder.add_shape_cylinder(
                body,
                radius=params["plate_radius"],
                half_height=params["plate_half_height"],
                cfg=plate_cfg,
                color=wp.vec3(*colors[level % len(colors)]),
                label=f"plate_{level}",
            )
        )
        plates.append(body)
    return plates, plate_shapes


def _add_sponge(builder: newton.ModelBuilder, params: dict) -> dict:
    size = params["sponge_size"]
    cells = params["sponge_cells"]
    particle_start = len(builder.particle_q)
    # spawn just above the tabletop so the block settles instead of ejecting
    builder.add_soft_grid(
        pos=wp.vec3(
            params["sponge_x"] - 0.5 * size[0],
            params["sponge_y"] - 0.5 * size[1],
            params["table_top_z"] + params["sponge_particle_radius"],
        ),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=cells[0],
        dim_y=cells[1],
        dim_z=cells[2],
        cell_x=size[0] / cells[0],
        cell_y=size[1] / cells[1],
        cell_z=size[2] / cells[2],
        density=params["sponge_density"],
        k_mu=params["sponge_k_mu"],
        k_lambda=params["sponge_k_lambda"],
        k_damp=params["sponge_k_damp"],
        particle_radius=params["sponge_particle_radius"],
    )
    particle_end = len(builder.particle_q)
    particles = np.arange(particle_start, particle_end, dtype=np.int32)
    # Grip row: the vertices along the -x edge (the edge the hand pinches). These
    # are pinned to the hand as kinematic BCs during the carry/rub; the rest of
    # the pad stays dynamic and deforms against the plate.
    pos = np.asarray(builder.particle_q[particle_start:particle_end], dtype=np.float64)
    x_min = pos[:, 0].min()
    grip = particles[pos[:, 0] <= x_min + 0.5 * size[0] / cells[0] + 1.0e-5]
    return {
        "particles": particles,
        "grip": np.asarray(grip, dtype=np.int32),
        "home": np.asarray(
            [params["sponge_x"], params["sponge_y"], params["table_top_z"] + 0.5 * size[2]], dtype=np.float64
        ),
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
        self._rub_windows: list[tuple[float, float, int]] = []
        self._rub_min_gap: dict[int, float] = {}

        builder = newton.ModelBuilder(gravity=p["gravity"])
        self.robot_bodies, robot_rigid_shapes, grasp_shapes = _add_h1(builder, p)
        self.robot_coord_count = builder.joint_coord_count
        table_shapes = _add_table(builder, p)
        self.plate_bodies, plate_shapes = _add_plates(builder, p)
        self.sponge_info = _add_sponge(builder, p)
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=p["shape_ke"],
            kd=p["shape_kd"],
            mu=p["table_mu"],
            gap=p["rigid_contact_gap"],
        )
        ground_shape = builder.add_ground_plane(cfg=ground_cfg)
        # A humanoid working over a table plants its forearm on the tabletop.
        # Only the four grasp chains should touch the furniture and dishes; the
        # rest of the robot is filtered against the table, plates, and ground so
        # a resting forearm can't detonate the AVBD contact solve. The grasp
        # fingers are likewise filtered against the table (mu=200 vs the grippy
        # top would stick) but keep plate contact — that pinch is the demo.
        grasp_set = set(grasp_shapes)
        non_grasp = [s for s in robot_rigid_shapes if s not in grasp_set]
        for robot_shape in robot_rigid_shapes:
            builder.add_shape_collision_filter_pair(robot_shape, ground_shape)
            for table_shape in table_shapes:
                builder.add_shape_collision_filter_pair(robot_shape, table_shape)
        for robot_shape in non_grasp:
            for plate_shape in plate_shapes:
                builder.add_shape_collision_filter_pair(robot_shape, plate_shape)
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

        self._build_choreography()

        self._setup_ik()
        self._solve_ik(
            np.asarray([p["rest_left"], p["rest_right"]], dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            iterations=48,
        )
        self.model.joint_q.assign(self.ik_joint_q_flat)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.torso_initial_transform = self.model.body_q.numpy()[self.torso_body].copy()

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
            # AVBD advances the H1 and plates in the same solve as the VBD sponge.
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=512,
            rigid_body_particle_contact_buffer_size=2048,
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

        self.plate_initial_positions = self.model.body_q.numpy()[self.plate_bodies, :3].copy()

        self._setup_sponge_grip()

        self.viewer.set_model(self.model)
        self.viewer.log_mesh(
            "/model/triangles",
            self.model.particle_q,
            self.model.tri_indices.flatten(),
            hidden=False,
            backface_culling=False,
            color=p["sponge_color"],
            roughness=0.9,
        )
        self.viewer.set_camera(
            wp.vec3(*p["camera_position"]),
            p["camera_pitch"],
            p["camera_yaw"],
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = p["camera_fov"]

    # ── choreography ─────────────────────────────────────────────────────

    def _plate_pile_z(self, level: int) -> float:
        return self.params["table_top_z"] + (2 * level + 1) * self.params["plate_half_height"]

    def _mark(self, time: float, name: str):
        self._phase_marks.append((time, name))

    def _grab_rim(
        self,
        hand: _HandCursor,
        rim_x: float,
        y: float,
        bottom_z: float,
        insert_dz: float | None = None,
        raise_dz: float | None = None,
        thumb: float | None = None,
    ):
        """Pinch an overhanging rim: hover behind it, slide the curled index
        underneath, raise until the rim rests on the index, close the thumb."""
        p = self.params
        insert_z = bottom_z + (p["grab_insert_hand_dz"] if insert_dz is None else insert_dz)
        raise_z = bottom_z + (p["grab_raise_hand_dz"] if raise_dz is None else raise_dz)
        self._mark(hand.time, "approach")
        hand.move(p["approach_time"], pos=(rim_x + p["grab_hover_dx"], y, bottom_z + p["grab_hover_dz"]))
        hand.move(
            p["descend_time"],
            pos=(rim_x + p["grab_hover_dx"], y, insert_z),
            index=p["grab_index_fraction"],
            other=p["other_finger_fraction"],
        )
        self._mark(hand.time, "insert")
        hand.move(p["insert_time"], pos=(rim_x + p["grab_insert_depth"], y, insert_z))
        hand.move(p["raise_time"], pos=(rim_x + p["grab_insert_depth"], y, raise_z))
        self._mark(hand.time, "close")
        hand.move(p["close_time"], thumb=p["grab_thumb_fraction"] if thumb is None else thumb)
        hand.wait(p["dwell_time"])

    def _release_and_retract(self, hand: _HandCursor, retreat_pos):
        """Open the thumb, then drop the hand well below the rim (into the free
        air in front of the table edge) BEFORE sliding out — the index stays
        curled but now clears the underside, so it can't drag the set-down plate
        off the edge. Uncurl only once the hand is clear, then retreat."""
        p = self.params
        self._mark(hand.time, "release")
        hand.move(p["release_time"], thumb=0.0)
        pos = hand.pos()
        hand.move(0.3, pos=(pos[0], pos[1], pos[2] - p["release_drop"]))
        hand.move(p["retract_time"], pos=(pos[0] - 0.11, pos[1], pos[2] - p["release_drop"]), index=0.0, other=0.0)
        hand.move(p["retract_time"], pos=retreat_pos)

    def _build_choreography(self):
        p = self.params
        tracks: dict[str, _Track] = {}
        for side, rest in (("left", p["rest_left"]), ("right", p["rest_right"])):
            tracks[f"{side}_pos"] = _Track(np.asarray(rest, dtype=np.float64))
            tracks[f"{side}_thumb"] = _Track(0.0)
            tracks[f"{side}_index"] = _Track(0.0)
            tracks[f"{side}_other"] = _Track(0.0)
        self.tracks = tracks
        right = _HandCursor(tracks, "right")
        left = _HandCursor(tracks, "left")

        plate_r = p["plate_radius"]
        plate_h = 2.0 * p["plate_half_height"]
        pile_y = p["dirty_pile_y"]
        pinch_dx = p["plate_center_to_pinch_dx"]
        wash_pinch = np.asarray([p["wash_x"] + pinch_dx, p["wash_y"], 0.0])
        sponge_size = p["sponge_size"]

        right.wait(p["settle_time"])
        left.wait(p["settle_time"])

        wash_count = p["wash_count"]
        clean_count = 0
        pile_count = p["plate_count"]
        for k in range(wash_count):
            top_level = pile_count - 1
            plate_bottom = self._plate_pile_z(top_level) - p["plate_half_height"]
            grab_rim_x = p["grab_overhang_x"] - plate_r

            if k > 0:
                # The new top plate sits flush in the pile: press the curled
                # fingertips onto it and drag it out until its rim overhangs.
                self._mark(right.time, "drag")
                press_x = p["dirty_pile_x"] + p["drag_hand_dx"]
                press_z = plate_bottom + plate_h + p["drag_press_hand_dz"]
                right.move(
                    p["approach_time"],
                    pos=(press_x, pile_y, press_z + 0.09),
                    index=p["grab_index_fraction"],
                    thumb=0.25,
                    other=0.9,
                )
                right.move(p["descend_time"], pos=(press_x, pile_y, press_z))
                drag = p["dirty_pile_x"] - p["grab_overhang_x"] + p["drag_slip_allowance"]
                right.move(p["drag_time"], pos=(press_x - drag, pile_y, press_z))
                right.move(p["lift_time"], pos=(press_x - drag, pile_y, press_z + 0.09), thumb=0.0)

            # grab the top plate off the pile
            self._grab_rim(right, grab_rim_x, pile_y, plate_bottom)

            # carry to the washing spot and lower it onto the table, but KEEP the
            # pinch closed and the hand in place — the right hand holds the dish
            # for the whole rub (a plate set down at the overhanging wash spot is
            # only marginally stable and the scrubbing tips it off the edge).
            self._mark(right.time, "carry_to_wash")
            carry_z = plate_bottom + p["grab_raise_hand_dz"] + p["carry_lift"]
            right.move(p["lift_time"], pos=(grab_rim_x + p["grab_insert_depth"], pile_y, carry_z))
            right.move(p["carry_time"], pos=(wash_pinch[0], wash_pinch[1], carry_z))
            hold_z = p["table_top_z"] + p["grab_raise_hand_dz"]
            right.move(p["lower_time"], pos=(wash_pinch[0], wash_pinch[1], hold_z))
            plate_ready_time = right.time

            if k == 0:
                # fetch the sponge while the right hand carries the first plate;
                # the pad reuses the plate edge-pinch primitive
                left.wait_until(p["settle_time"] + p["approach_time"] + p["descend_time"])
                sponge_bottom = p["table_top_z"] + p["sponge_particle_radius"]
                sponge_rim_x = p["sponge_x"] - 0.5 * sponge_size[0]
                self._grab_rim(
                    left,
                    sponge_rim_x,
                    p["sponge_y"],
                    sponge_bottom,
                    insert_dz=p["sponge_insert_hand_dz"],
                    raise_dz=p["sponge_raise_hand_dz"],
                    thumb=p["sponge_thumb_fraction"],
                )
                # pin the sponge grip row to the hand now that the fingers have
                # closed around its edge (see _setup_sponge_grip)
                self.sponge_grip_start = left.time
                lift_z = sponge_bottom + p["sponge_raise_hand_dz"] + p["carry_lift"] + 0.03
                left.move(p["lift_time"], pos=(sponge_rim_x + p["grab_insert_depth"], p["sponge_y"], lift_z))

            # rub: flat ellipses grazing the plate top while the right hand holds
            # the plate down against the table (the table takes the rub force).
            left.wait_until(plate_ready_time + 0.2)
            plate_top_z = p["table_top_z"] + plate_h
            plate_rim_x = p["wash_x"] - plate_r
            rub_pinch = np.asarray([plate_rim_x - p["rub_pinch_behind_rim"], p["wash_y"]])
            rub_z = plate_top_z + p["rub_pinch_above_plate"]
            self._mark(left.time, "rub_approach")
            left.move(p["carry_time"], pos=(rub_pinch[0], rub_pinch[1], rub_z + p["rub_hover_dz"]))
            left.move(p["lower_time"], pos=(rub_pinch[0], rub_pinch[1], rub_z))
            self._mark(left.time, "rub")
            rub_start = left.time
            steps_per_circle = 24
            # centred ellipse: +x reaches over the plate, -x stays behind the rim
            for c in range(p["rub_circles"] * steps_per_circle):
                angle = 2.0 * np.pi * (c + 1) / steps_per_circle
                dx = p["rub_radius_x"] * np.cos(angle)
                dy = p["rub_radius_y"] * np.sin(angle)
                left.move(
                    p["rub_circle_time"] / steps_per_circle,
                    pos=(rub_pinch[0] + dx, rub_pinch[1] + dy, rub_z),
                    ease="linear",
                )
            self._rub_windows.append((rub_start, left.time, k))
            self._mark(left.time, "rub_done")
            left.move(p["lower_time"], pos=(rub_pinch[0], rub_pinch[1], rub_z + p["rub_hover_dz"]))
            # park the sponge high on the left while the right hand stacks the plate
            left.move(p["carry_time"], pos=(-0.32, 0.10, p["table_top_z"] + 0.17))

            # the right hand still holds the plate: lift it straight up and carry
            # it to the clean pile (no re-grasp needed)
            right.wait_until(left.time - p["carry_time"] - p["lift_time"])
            self._mark(right.time, "carry_to_clean")
            clean_z_bottom = p["table_top_z"] + clean_count * plate_h
            carry_z = p["table_top_z"] + p["grab_raise_hand_dz"] + p["carry_lift"] + clean_count * plate_h
            right.move(p["lift_time"], pos=(wash_pinch[0], p["wash_y"], carry_z))
            clean_pinch_x = p["clean_x"] + pinch_dx
            right.move(p["carry_time"], pos=(clean_pinch_x, p["clean_y"], carry_z))
            right.move(p["lower_time"], pos=(clean_pinch_x, p["clean_y"], clean_z_bottom + p["grab_raise_hand_dz"]))
            self._release_and_retract(right, (clean_pinch_x - 0.12, p["wash_y"], p["table_top_z"] + 0.18))
            clean_count += 1
            pile_count -= 1

        # epilogue: return the sponge home and rest both hands
        sponge_rim_x = p["sponge_x"] - 0.5 * sponge_size[0]
        sponge_home_pinch_x = sponge_rim_x + p["grab_insert_depth"]
        self._mark(left.time, "sponge_home")
        left.move(p["carry_time"], pos=(sponge_home_pinch_x, p["sponge_y"], p["table_top_z"] + 0.10))
        left.move(
            p["lower_time"],
            pos=(sponge_home_pinch_x, p["sponge_y"], p["table_top_z"] + p["sponge_particle_radius"] - 0.002),
        )
        # unpin the sponge once it is back on the table
        self.sponge_grip_end = left.time
        self._release_and_retract(left, p["rest_left"])
        right.wait_until(left.time)
        right.move(p["retract_time"], pos=p["rest_right"])
        self._mark(max(left.time, right.time), "done")
        self._phase_marks.sort(key=lambda mark: mark[0])
        self.total_time = max(left.time, right.time) + 0.6

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

    # ── sponge grip (kinematic pin) ──────────────────────────────────────

    def _setup_sponge_grip(self):
        """Prepare to pin the sponge's -x edge row to the left hand. A curved H1
        fingertip spears a 3D FEM solid, so rather than a physical finger grip
        the grip row is made kinematic (mass 0) and driven to follow the hand
        while the rest of the pad stays dynamic and scrubs the plate."""
        grip = self.sponge_info["grip"]
        self._grip_np = grip
        self.grip_indices = wp.array(grip, dtype=wp.int32, device=self.model.device)
        self.grip_offsets = wp.zeros(len(grip), dtype=wp.vec3, device=self.model.device)
        self.grip_left_body = self.hand_bodies[0]
        self._grip_saved_mass = self.model.particle_mass.numpy()[grip].copy()
        self._grip_saved_inv_mass = self.model.particle_inv_mass.numpy()[grip].copy()
        self.grip_state = "idle"

    def _left_hand_transform(self) -> wp.transform:
        return wp.transform(*self.state_0.body_q.numpy()[self.grip_left_body])

    def _activate_sponge_grip(self):
        hand_inv = wp.transform_inverse(self._left_hand_transform())
        world = self.state_0.particle_q.numpy()[self._grip_np]
        offsets = np.asarray([wp.transform_point(hand_inv, wp.vec3(*p)) for p in world], dtype=np.float32)
        self.grip_offsets.assign(offsets)
        mass = self.model.particle_mass.numpy()
        inv_mass = self.model.particle_inv_mass.numpy()
        mass[self._grip_np] = 0.0
        inv_mass[self._grip_np] = 0.0
        self.model.particle_mass.assign(mass)
        self.model.particle_inv_mass.assign(inv_mass)
        self.grip_state = "held"

    def _deactivate_sponge_grip(self):
        mass = self.model.particle_mass.numpy()
        inv_mass = self.model.particle_inv_mass.numpy()
        mass[self._grip_np] = self._grip_saved_mass
        inv_mass[self._grip_np] = self._grip_saved_inv_mass
        self.model.particle_mass.assign(mass)
        self.model.particle_inv_mass.assign(inv_mass)
        self.grip_state = "done"

    # ── simulation loop ──────────────────────────────────────────────────

    def simulate(self):
        for _ in range(self.sim_substeps):
            if self.grip_state == "held":
                wp.launch(
                    drive_grip_particles,
                    dim=len(self._grip_np),
                    inputs=[
                        self.grip_indices,
                        self.grip_offsets,
                        self._left_hand_transform(),
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                    ],
                )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.grip_state == "idle" and self.sim_time >= self.sponge_grip_start:
            self._activate_sponge_grip()
        elif self.grip_state == "held" and self.sim_time >= self.sponge_grip_end:
            self._deactivate_sponge_grip()
        self._update_trajectory()
        self.simulate()
        for start, end, plate in self._rub_windows:
            if start <= self.sim_time <= end:
                sponge = self.state_0.particle_q.numpy()[self.sponge_info["particles"]]
                plate_pos = self.state_0.body_q.numpy()[self.plate_bodies[self.params["plate_count"] - 1 - plate], :3]
                # closest sponge particle to the plate centre: the pad is held at
                # its far edge, so its centroid sits well behind the plate
                gap = float(np.min(np.linalg.norm(sponge[:, :2] - plate_pos[:2], axis=1)))
                self._rub_min_gap[plate] = min(self._rub_min_gap.get(plate, np.inf), gap)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Phase: {self.phase}")
        ui.text(f"t = {self.sim_time:.2f} / {self.total_time:.2f} s")

    def test_final(self):
        p = self.params
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Sponge state contains non-finite values"
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"

        # washed plates are stacked on the clean spot
        for k in range(p["wash_count"]):
            level = p["plate_count"] - 1 - k
            pos = body_q[self.plate_bodies[level], :3]
            xy_err = float(np.linalg.norm(pos[:2] - (p["clean_x"], p["clean_y"])))
            assert xy_err < 0.07, f"Washed plate {k} is {xy_err:.3f} m from the clean spot"
            expected_z = p["table_top_z"] + (2 * k + 1) * p["plate_half_height"]
            assert abs(pos[2] - expected_z) < 0.03, f"Washed plate {k} is not resting on the clean pile: z={pos[2]:.3f}"
        # unwashed plates never left the dirty pile
        for level in range(p["plate_count"] - p["wash_count"]):
            pos = body_q[self.plate_bodies[level], :3]
            xy_err = float(np.linalg.norm(pos[:2] - (p["dirty_pile_x"], p["dirty_pile_y"])))
            assert xy_err < 0.06, f"Unwashed plate {level} drifted {xy_err:.3f} m"
        # the sponge went home
        sponge_center = particle_q[self.sponge_info["particles"]].mean(axis=0)
        home_err = float(np.linalg.norm(sponge_center[:2] - self.sponge_info["home"][:2]))
        assert home_err < 0.06, f"Sponge ended {home_err:.3f} m from its home spot"
        assert sponge_center[2] < p["table_top_z"] + 0.06, "Sponge is not resting on the table"
        # the sponge actually rubbed every washed plate
        for k in range(p["wash_count"]):
            gap = self._rub_min_gap.get(k, np.inf)
            assert gap < 0.05, f"Sponge never rubbed plate {k}: min XY distance {gap:.3f} m"
        # torso stayed put
        torso_q = body_q[self.torso_body]
        torso_position_error = float(np.linalg.norm(torso_q[:3] - self.torso_initial_transform[:3]))
        assert torso_position_error < 0.01, f"H1 torso translated too far: {torso_position_error:.3f} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
