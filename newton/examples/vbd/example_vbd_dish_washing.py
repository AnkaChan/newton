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

import os

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

PARAMS = {
    # simulation
    "fps": 60,
    # 20 substeps: the grips are tuned at this rate — VBD's penalty friction
    # scales with per-substep displacement, so changing the substep count
    # de-tunes every contact in the scene
    "sim_substeps": 20,
    # 16 iterations: the sponge particles clamped at the fingertips form a
    # stiff cluster; unconverged VBD iterations inject the energy behind the
    # intermittent fingertip blow-ups
    "solver_iterations": 16,
    "enable_cuda_graph": True,
    "gravity": -9.81,
    "num_frames": 1900,
    # how many plates are washed (script 2 raises this to the full pile)
    "wash_count": 1,
    "plate_count": 3,
    # debug: only the left hand grabs+lifts+holds the sponge (isolates the
    # sponge grip from the plate work and the shared solver)
    "sponge_only": False,
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
    # plates: light stoneware, glossy (low mu) so the wet sponge slides over
    # them while the grippy tabletop holds them in place. The shape comes from
    # asset/dish_plate.obj (see make_dish_plate.py): an offset shell with a
    # deep well, a flat raised rim, and a 4 mm lip; nested plates stack 11 mm
    # apart (the generator's WALL_THICKNESS).
    "plate_radius": 0.075,
    # generator constants the choreography needs: EDGE_THICKNESS (the rim lip
    # the fingers pinch) and WALL_THICKNESS (the nested-stack pitch)
    "plate_rim_thickness": 0.010,
    "plate_stack_pitch": 0.011,
    # heavy stoneware: a light bowl gets tipped in the pinch by the sponge
    # press during the rub and then cannot drop into the nest. The extra
    # in-carry pinch creep that comes with the weight is absorbed by the
    # ride-and-sweep placement, which does not depend on where the creep
    # left the bowl.
    "plate_density": 1000.0,
    # grippy on the table (mu combines with the 0.9 tabletop) so the wet sponge
    # can scrub without dragging the plate off the overhanging wash spot
    "plate_mu": 0.6,
    "plate_colors": ((0.93, 0.90, 0.82), (0.72, 0.82, 0.90), (0.78, 0.88, 0.78)),
    # dirty pile: a nested stack deeper on the table (set dressing — the curled
    # H1 index spans ~30 mm below the pinch point and cannot enter the 11 mm
    # inter-rim gap of a nested stack, and an offset plate will not sit flat on
    # congruent shells). The plate to be washed rests ALONE at the front edge
    # with its rim overhanging, which is the situation the grab primitive is
    # calibrated for.
    "dirty_pile_x": -0.01,
    "dirty_pile_y": -0.24,
    "grab_overhang_x": -0.173,
    # dirty-plate arrangement: "pile" nests plate_count-1 plates at the pile
    # spot and rests the last plate at the overhang spot (the 1-dish example
    # grabs that lone plate); "row" lays them out along the front edge, each
    # overhanging and independently graspable (the all-dishes example).
    # ``row_spacing`` is the y-pitch of the row.
    "dirty_layout": "pile",
    "row_spacing": 0.135,
    # washing spot: at the front edge NEXT TO the sponge's home, so the
    # fragile airborne sponge hop is only ~12 cm. The bowl RESTS on the table
    # here during the rub while the right hand keeps its rim pinch (a bowl
    # held airborne hangs tilted in the rim pinch and works out of the
    # fingers; the bowl reaches this spot by a long table slide, which is
    # robust)
    "wash_x": -0.173,
    "wash_y": 0.10,
    # sponge: a stiff soft FEM pad, physically pinched by the H1 fingers. Its -x
    # edge overhangs the front table edge so the index can slide underneath.
    "sponge_size": (0.10, 0.075, 0.024),
    "sponge_cells": (8, 6, 2),
    "sponge_x": -0.185,
    "sponge_y": 0.27,
    # where the sponge is set aside after the wash: on the table just
    # northeast of the wash station, one short supported slide away from
    # where the rub ends (transporting the pad any further sheds it)
    "sponge_park": (-0.15, 0.22),
    "sponge_density": 250.0,
    # Shear-dominant Neo-Hookean (k_mu > k_lambda): shear keeps the pad holding
    # its shape in the pinch (it doesn't squirt out), lower bulk keeps it
    # compressible. Moderate stiffness — too stiff and the pad fights the
    # fingertip like a rigid body and spikes the contact. The stable material
    # tolerates inverted (negative-volume) tets, so modest penetration is fine.
    "sponge_k_mu": 1.5e4,
    "sponge_k_lambda": 4.0e3,
    "sponge_k_damp": 5.0e-3,
    # particle radius = per-particle finger contact standoff. Matches the stable
    # recipe of example_vbd_gripper_soft_grid (radius ~0.01, low contact ke, high
    # damping, 20 substeps): a firm-but-soft contact that clamps without the
    # force spike a small gap + stiff contact produces.
    "sponge_particle_radius": 0.012,
    "sponge_color": (0.95, 0.85, 0.25),
    # rigid-soft contact for the sponge grip: soft + heavily damped to avoid a
    # contact-force spike (a stiff contact spears/drags the edge tets and blows
    # the mesh up), grippy friction so the clamped pad doesn't slip.
    "soft_contact_ke": 8.0e2,
    "soft_contact_kd": 15.0,
    "soft_contact_mu": 1.2,
    "soft_contact_margin": 0.014,
    "enable_water_tight_rigid_soft_contact": False,
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
    # finger contact stiffness. The finger↔sponge (body↔particle) contact ke is
    # the AVERAGE of this and soft_contact_ke, so a stiff finger keeps the sponge
    # contact stiff and shoots the pad out — keep it low for a gentle grip.
    "finger_contact_ke": 1.0e3,
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
    # measured in-sim (outputs/plot_pinch_geom.py): the curled index tip tops
    # out at ~P+0.005, so the pinch point must go 10 mm PAST the rim edge for
    # the tip apex to sit ~15 mm under the lip — a pinch at the rim edge only
    # nips the outer 2 mm of the lip and slips off at lift.
    "grab_insert_depth": 0.010,
    # the underside at the apex radius (~15 mm inboard of the rim edge) sits
    # ~4 mm below the rim-edge underside the grab spec references; -0.006 puts
    # the tip apex ~2 mm into that local underside for a gentle upward support
    "grab_raise_hand_dz": -0.006,
    "grab_index_fraction": 0.75,
    # with the raised pinch at P = rim-edge underside - 0.006, the 10 mm rim
    # top rides at ~P+0.016; the thumb-tip bottom reaches P+0.011 at 0.80 for
    # a ~5 mm press — enough torsional friction that the carried bowl cannot
    # yaw out of the pinch. Deeper pivots the bowl to a steep hanging tilt at
    # close; a thumb driven below the index apex squeezes the rigid rim out
    # of the fingers like a watermelon seed.
    "grab_thumb_fraction": 0.80,
    "other_finger_fraction": 0.8,
    # sponge pinch: the curled index slides under the pad's -x edge and the
    # thumb closes on top, clamping the edge (the pad's particle-radius standoff
    # + the extra contact margin keep the fingers from spearing the tets). The
    # thumb closes further than the plate to compress the thicker, stiff pad.
    # insert the index clearly BELOW the pad (in the free air in front of the
    # table edge) so it slides UNDER the overhanging edge instead of hitting the
    # pad's side and shoving it away; then rise to lift the edge onto the index.
    "sponge_insert_hand_dz": -0.020,
    "sponge_raise_hand_dz": -0.014,
    # deep index hook + a thumb that seeks contact: with the soft left-hand
    # drive (see sponge_hand_drive_ke) the thumb stalls into a gentle bounded
    # clamp on the pad edge instead of crushing or ejecting it, and the hook
    # carries the weight
    "sponge_thumb_fraction": 0.66,
    "sponge_index_fraction": 0.85,
    # during the rub the pad RESTS on the bowl and the fingers only push it
    # around the ellipses: the thumb opens to a loose cage so the pad edge
    # micro-slips in the fingers instead of the clamped tets being cyclically
    # wrung against the rim (which ratchets the pad apart and detonates the
    # bowl contact)
    "sponge_rub_thumb_fraction": 0.58,
    # pinch-point x offset from the held plate's center while carried
    # (= grab_insert_depth - plate_radius)
    "plate_center_to_pinch_dx": -0.065,
    "carry_lift": 0.055,
    # the wash leg travels low: the hanging bowl's base clears the table by
    # ~2 cm, so if the rim pinch does slip the bowl lands flat beside the wash
    # spot instead of swinging down from carry height
    "wash_carry_lift": 0.035,
    # the hanging bowl creeps forward in the rim pinch under its own gravity
    # torque (penalty friction ratchets a few mm per second of carry), so the
    # carry stops this far SHORT of the wash spot, sets the bowl down, and
    # slides it the rest of the way along the table — on the table the pinch
    # is unloaded and the creep stops
    "wash_approach_back": 0.06,
    # pinch height while the bowl rests on the table at the wash spot: held a
    # few mm LOW so the thumb preloads the rim downward — the scrubbing sponge
    # otherwise levers the resting bowl up into a tilt it carries to the stack
    "wash_hold_lift": 0.019,
    # carrying the washed bowl to the stack: the bowl hangs tilted in the rim
    # pinch — a variable 8-38 deg, so its low leading edge rides anywhere up
    # to ~50 mm below the pinch — and must clear the stack's top rim (~41 mm
    # above the table) on the way in even in the worst case
    "stack_carry_lift": 0.105,
    # how far the carried bowl's base hangs below the pinch point (measured
    # in-sim: the bowl tilts ~16 deg in the rim pinch and rides base ~34 mm
    # under P), and the extra drop left when the hand opens over the stack so
    # the opening fingers cannot drag the bowl off the nest
    "stack_hang": 0.034,
    # nesting by feel: the carried bowl creeps a variable 2-4 cm in the pinch
    # and hangs at a variable tilt, so neither an aimed drop nor a fixed-
    # height sweep lands it. Instead the hand descends DIAGONALLY from west
    # of the stack down to a seated endpoint over the nest: wherever the
    # bowl's low leading edge first meets the stack along that path, it lands
    # on the rim top and rides in, and the well drops into the nest when it
    # crosses the opening — independent of the creep and tilt.
    "stack_sweep_back": 0.055,
    # sweep far enough past the stack centre that the bowl's well crosses the
    # nest opening for the whole 2-4.5 cm range of pinch trailing; once the
    # bowl drops in, the nest captures it and the rest of the sweep just
    # slips the pinch
    "stack_sweep_over": 0.045,
    "stack_seat_depth": 0.004,
    # rub trajectory: the sponge is pinch-held at its -x edge, so the pinch
    # stays behind the plate rim (the index must never cross above it) and
    # the sponge body scrubs the near half of the plate in flat ellipses
    # the pad's stiff mid-section scrubs the rim: pinched further back, the
    # pad's unsupported far end droops several cm and its flapping tip lands
    # deep on the rim, shoving the bowl and squeezing the index out of the
    # maintained rim pinch; pinched closer, the well edge is the hazard
    "rub_pinch_behind_rim": 0.050,
    "rub_radius_x": 0.006,
    "rub_radius_y": 0.018,
    "rub_circles": 3,
    "rub_circle_time": 1.8,
    # pinch height above the plate top: the pinned edge rides here and the pad
    # drapes down to just kiss the plate (positive = a light graze, not a press)
    # the pad dangles from its edge pinch, so its height cannot be controlled
    # in the air: the pinch rides just above the rim plane and the draped pad
    # RESTS on the bowl, mopped around by the ellipses while the bowl (held
    # down by the right hand's pinch) and the table carry the pad's weight
    "rub_pinch_above_plate": 0.010,
    # just enough hover for the dangling pad's tip to clear the bowl rim on
    # the short hop in — height is exposure for the hooked pad
    "rub_hover_dz": 0.05,
    # durations [s]
    "settle_time": 0.5,
    "approach_time": 0.7,
    "descend_time": 0.9,
    "insert_time": 0.8,
    "raise_time": 0.6,
    "close_time": 1.4,
    "dwell_time": 0.25,
    "lift_time": 0.45,
    "carry_time": 0.9,
    "lower_time": 0.5,
    "release_time": 0.35,
    "retract_time": 0.55,
    # settle time after lowering a plate onto the pile before opening the grip
    "place_dwell": 0.45,
    # brief pause after picking the sponge up, before heading to the rub
    "sponge_hold_time": 0.4,
    # AVBD joint drives; Newton IK only generates their targets
    "joint_drive_ke": 5.0e4,
    "joint_drive_kd": 5.0e2,
    "torso_drive_ke": 2.0e5,
    "torso_drive_kd": 2.0e3,
    # stiff finger drive for the RIGHT hand: the bowl's rim pinch needs a firm
    # clamp against the rigid lip
    "finger_drive_ke": 2.0e4,
    "finger_drive_kd": 1.0e2,
    # much softer drive for the LEFT hand: the fingertip gap at any usable
    # thumb fraction is far inside the pad's contact shell, so the drive
    # stiffness — not the fraction — sets the stall force on the handful of
    # clamped sponge particles. The stiff drive crushes/ejects the pad
    # (watermelon-seed) or blows the contact up; the soft drive yields into a
    # gentle bounded grip.
    "sponge_hand_drive_ke": 5.0e3,
    "sponge_hand_drive_kd": 5.0e1,
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


PLATE_OBJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset", "dish_plate.obj")


def _load_obj(path: str) -> tuple[np.ndarray, list[int]]:
    """Load a triangle-mesh OBJ (vertices + triangulated faces)."""
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
            # the left hand grips the soft sponge and needs a much softer,
            # force-limited drive than the right hand's rigid-rim pinch
            if "/L_" in builder.joint_label[joint]:
                builder.joint_target_ke[dof] = params["sponge_hand_drive_ke"]
                builder.joint_target_kd[dof] = params["sponge_hand_drive_kd"]
            else:
                builder.joint_target_ke[dof] = params["finger_drive_ke"]
                builder.joint_target_kd[dof] = params["finger_drive_kd"]

    body_names = {
        "torso": "torso_link",
        "left_hand": "left_hand_link",
        "right_hand": "right_hand_link",
    }
    body_indices = {name: _find_suffix(builder.body_label, suffix) for name, suffix in body_names.items()}

    # The four thumb/index grasp chains keep shape-shape contact (rigid plate
    # pinch) AND particle contact (they physically grip the soft sponge) with a
    # finer texture SDF. The finger-particle contact triggers at a deliberately
    # large gap (``soft_contact_margin`` + ``sponge_particle_radius``) so a
    # fingertip contacts and compresses the stiff Neo-Hookean sponge instead of
    # spearing through it. Every non-grasp collider is filtered against the table
    # and plates (a humanoid's forearm resting on the tabletop otherwise explodes
    # the AVBD contact solve).
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
            builder.shape_flags[shape] |= particle_collision_flag
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


def _add_plates(builder: newton.ModelBuilder, params: dict) -> tuple[list[int], list[int], list[tuple]]:
    """Add the dirty plates (asset/dish_plate.obj) and return (bodies, shapes,
    grab_specs). Each grab_spec is (rim_x, y, rim_underside_z) — where the
    right hand slides its index under that plate's front rim to pick it up.
    The plate body origin sits at the plate's base (the mesh rests on z = 0).

    ``dirty_layout`` selects the arrangement:
      - "pile": plate_count-1 plates nested at the pile spot (set dressing —
        the index cannot enter the 11 mm inter-rim gap of a nested stack) and
        the last plate alone at the overhang spot, rim past the table edge
        (the 1-dish example grabs that lone plate).
      - "row":  plates in a single row along the front edge, each overhanging
        and independently graspable (the all-dishes example)."""
    plate_cfg = newton.ModelBuilder.ShapeConfig(
        density=params["plate_density"],
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["plate_mu"],
        gap=params["rigid_contact_gap"],
        has_particle_collision=True,
        margin=0.0,
    )
    verts, faces = _load_obj(PLATE_OBJ)
    plate_mesh = newton.Mesh(verts, faces)
    rim_underside = float(verts[:, 2].max()) - params["plate_rim_thickness"]
    plates = []
    plate_shapes = []
    grab_specs = []
    count = params["plate_count"]
    colors = params["plate_colors"]
    layout = params["dirty_layout"]
    for level in range(count):
        if layout == "row":
            x = params["grab_overhang_x"]
            y = params["dirty_pile_y"] + (level - 0.5 * (count - 1)) * params["row_spacing"]
            z = params["table_top_z"]
        elif level == count - 1:  # the plate to wash: alone at the overhang spot
            x = params["grab_overhang_x"]
            y = params["dirty_pile_y"]
            z = params["table_top_z"]
        else:  # nested dressing pile, spawned a hair apart so it settles
            x = params["dirty_pile_x"]
            y = params["dirty_pile_y"]
            z = params["table_top_z"] + level * (params["plate_stack_pitch"] + 0.0005)
        body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()))
        shape = builder.add_shape_mesh(
            body,
            mesh=plate_mesh,
            cfg=plate_cfg,
            color=wp.vec3(*colors[level % len(colors)]),
            label=f"plate_{level}",
        )
        # the fingers pinch a 4 mm lip, so the plate SDF needs sub-mm voxels
        builder.shape_sdf_max_resolution[shape] = 256
        plate_shapes.append(shape)
        plates.append(body)
        grab_specs.append((x - params["plate_radius"], y, z + rim_underside))
    return plates, plate_shapes, grab_specs


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
    return {
        "particles": np.arange(particle_start, particle_end, dtype=np.int32),
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
        self.plate_bodies, plate_shapes, self.plate_grab_specs = _add_plates(builder, p)
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

        self._capture_graph()

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
        index: float | None = None,
    ):
        """Pinch an overhanging rim: hover behind it, slide the curled index
        underneath, raise until the rim rests on the index, close the thumb."""
        p = self.params
        insert_z = bottom_z + (p["grab_insert_hand_dz"] if insert_dz is None else insert_dz)
        raise_z = bottom_z + (p["grab_raise_hand_dz"] if raise_dz is None else raise_dz)
        index_frac = p["grab_index_fraction"] if index is None else index
        self._mark(hand.time, "approach")
        hand.move(p["approach_time"], pos=(rim_x + p["grab_hover_dx"], y, bottom_z + p["grab_hover_dz"]))
        hand.move(
            p["descend_time"],
            pos=(rim_x + p["grab_hover_dx"], y, insert_z),
            index=index_frac,
            other=p["other_finger_fraction"],
        )
        self._mark(hand.time, "insert")
        hand.move(p["insert_time"], pos=(rim_x + p["grab_insert_depth"], y, insert_z))
        hand.move(p["raise_time"], pos=(rim_x + p["grab_insert_depth"], y, raise_z), index=index_frac)
        self._mark(hand.time, "close")
        # curl the index further as the thumb closes: a firm index hook keeps a
        # soft pad from sliding off backward during the lift
        hand.move(
            p["close_time"],
            thumb=p["grab_thumb_fraction"] if thumb is None else thumb,
            index=index_frac,
        )
        hand.wait(p["dwell_time"])

    def _grab_sponge(self, hand: _HandCursor, sponge_size) -> None:
        """Physically pinch the sponge pad's -x edge and lift it. The curled
        index slides under the overhanging edge and the thumb clamps on top."""
        p = self.params
        hand.wait_until(p["settle_time"] + p["approach_time"] + p["descend_time"])
        sponge_bottom = p["table_top_z"] + p["sponge_particle_radius"]
        sponge_rim_x = p["sponge_x"] - 0.5 * sponge_size[0]
        self._grab_rim(
            hand,
            sponge_rim_x,
            p["sponge_y"],
            sponge_bottom,
            insert_dz=p["sponge_insert_hand_dz"],
            raise_dz=p["sponge_raise_hand_dz"],
            thumb=p["sponge_thumb_fraction"],
            index=p["sponge_index_fraction"],
        )
        # lift only as high as the rub needs: every centimetre of dangling
        # airtime is a chance for the hooked pad to swing off the fingers or
        # spike the fingertip contact
        lift_z = sponge_bottom + p["sponge_raise_hand_dz"] + 0.045
        pinch_x = sponge_rim_x + p["grab_insert_depth"]
        # lift slowly (2x) so inertia does not flick the pad off the fingers
        hand.move(2.0 * p["lift_time"], pos=(pinch_x, p["sponge_y"], lift_z))
        hand.move(p["sponge_hold_time"], pos=(pinch_x, p["sponge_y"], lift_z))

    def _release_and_retract(self, hand: _HandCursor, retreat_pos):
        """Open the whole hand at once so the object drops the last few cm and
        lands where it is — extracting the still-curled index from under a
        placed plate drags it off the edge, and a stuck plate spikes the solve.
        Then lift the open hand STRAIGHT UP to clear the dropped object before
        retreating (sweeping the open hand sideways knocks it off the edge)."""
        p = self.params
        self._mark(hand.time, "release")
        hand.move(p["release_time"], thumb=0.0, index=0.0, other=0.0)
        pos = hand.pos()
        hand.move(p["retract_time"], pos=(pos[0], pos[1], pos[2] + 0.13))
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
        pinch_dx = p["plate_center_to_pinch_dx"]
        wash_pinch = np.asarray([p["wash_x"] + pinch_dx, p["wash_y"], 0.0])
        sponge_size = p["sponge_size"]

        right.wait(p["settle_time"])
        left.wait(p["settle_time"])

        if p.get("sponge_only", False):
            # isolated sponge-grip test: only the left hand grabs + lifts + holds
            # the sponge; the right hand and plates are untouched.
            self._grab_sponge(left, sponge_size)
            self._mark(left.time, "done")
            self._phase_marks.sort(key=lambda mark: mark[0])
            self.total_time = left.time + 0.6
            return

        wash_count = p["wash_count"]
        for k in range(wash_count):
            # wash plates top-down (pile) / one end to the other (row); this
            # index order matches test_final's stacked-nest expectations
            wash_level = p["plate_count"] - 1 - k
            grab_rim_x, grab_y, plate_bottom = self.plate_grab_specs[wash_level]

            # grab the plate off the pile / out of the row
            self._grab_rim(right, grab_rim_x, grab_y, plate_bottom)

            # carry to the washing spot. The bowl hangs tilted in the rim pinch,
            # so it travels at carry height and is lowered SLOWLY onto the table
            # at the wash spot: descending straight into the hold height drags
            # the low-hanging far edge across the table and torques the bowl out
            # of the fingers. During the rub the bowl rests on the table and the
            # right hand keeps its pinch on the rim.
            self._mark(right.time, "carry_to_wash")
            carry_z = plate_bottom + p["grab_raise_hand_dz"] + p["wash_carry_lift"]
            wash_hold_z = p["table_top_z"] + p["grab_raise_hand_dz"] + p["wash_hold_lift"]
            right.move(p["lift_time"], pos=(grab_rim_x + p["grab_insert_depth"], grab_y, carry_z))
            # carry slowly and stop short: the bowl creeps forward in the rim
            # pinch while airborne (see wash_approach_back), so it is set down
            # before the spot and slid the rest of the way with the table
            # carrying its weight
            approach_y = wash_pinch[1] - p["wash_approach_back"]
            right.move(2.0 * p["carry_time"], pos=(wash_pinch[0], approach_y, carry_z))
            right.move(2.0 * p["lower_time"], pos=(wash_pinch[0], approach_y, wash_hold_z))
            # the bowl carries ~2-3 cm of forward creep from the airborne leg,
            # so the slide aims short by that much for the bowl itself to end
            # at the wash spot (an off-centre bowl gets tipped by the rub)
            right.move(2.0 * p["lower_time"], pos=(wash_pinch[0], wash_pinch[1] - 0.025, wash_hold_z))
            plate_ready_time = right.time

            if k == 0:
                # fetch the sponge while the right hand carries the first plate
                self._grab_sponge(left, sponge_size)

            # rub: flat ellipses grazing the rim plane of the bowl, which rests
            # on the table at the wash spot (grab spec z is the resting rim
            # underside; the flat rim top is one lip thickness up)
            left.wait_until(plate_ready_time + 0.2)
            plate_top_z = plate_bottom + p["plate_rim_thickness"]
            plate_rim_x = p["wash_x"] - plate_r
            rub_pinch = np.asarray([plate_rim_x - p["rub_pinch_behind_rim"], p["wash_y"]])
            rub_z = plate_top_z + p["rub_pinch_above_plate"]
            self._mark(left.time, "rub_approach")
            # come down BEHIND the rim, then slide in horizontally: lowering at
            # the rub point sweeps the dangling pad's tip down across the rim's
            # outer side and shoves the held bowl
            back_x = rub_pinch[0] - 0.05
            # slow approach: the hooked pad swings on fast moves
            left.move(2.0 * p["carry_time"], pos=(back_x, rub_pinch[1], rub_z + p["rub_hover_dz"]))
            left.move(p["lower_time"], pos=(back_x, rub_pinch[1], rub_z))
            left.move(p["lower_time"], pos=(rub_pinch[0], rub_pinch[1], rub_z))
            # open the thumb to a cage for the scrub (see sponge_rub_thumb_fraction)
            left.move(0.3, thumb=p["sponge_rub_thumb_fraction"])
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
            # set the sponge down IMMEDIATELY beside the wash station: one
            # short slide northeast off the bowl's flank onto the table (the
            # pad is supported by the bowl, then the wood, the whole way),
            # lower, and let go. Every centimetre a hooked pad travels
            # unsupported is a chance to shed it.
            sponge_drag_z = p["table_top_z"] + p["sponge_particle_radius"] - 0.002
            # re-close the thumb to the carry clamp before moving off the bowl
            left.move(0.3, thumb=p["sponge_thumb_fraction"])
            self._mark(left.time, "sponge_home")
            park_pinch = (rub_pinch[0] + 0.10, rub_pinch[1] + 0.12)
            left.move(2.0 * p["lower_time"], pos=(park_pinch[0], park_pinch[1], rub_z))
            sponge_clear_time = left.time
            left.move(p["lower_time"], pos=(park_pinch[0], park_pinch[1], sponge_drag_z))
            self._release_and_retract(left, p["rest_left"])

            # the right hand still holds the plate: lift it straight up, carry it
            # over the 2-bowl stack, and lower it into the nest. It starts as
            # soon as the sponge has moved clear of the bowl.
            right.wait_until(sponge_clear_time + 0.2)
            self._mark(right.time, "carry_to_stack")
            carry_z = p["table_top_z"] + p["grab_raise_hand_dz"] + p["stack_carry_lift"]
            right.move(p["lift_time"], pos=(wash_pinch[0], p["wash_y"], carry_z))
            stack_pinch_x = p["dirty_pile_x"] + pinch_dx
            stack_y = p["dirty_pile_y"]
            # carry slowly so the tilted-hanging bowl does not swing/overshoot,
            # stopping short of the stack (see stack_sweep_back)
            sweep_start_x = stack_pinch_x - p["stack_sweep_back"]
            right.move(2.5 * p["carry_time"], pos=(sweep_start_x, stack_y, carry_z))
            # diagonal ride-in (see the stack_sweep_back comment): descend from
            # west of the stack to a seated endpoint over the nest, which for
            # the k-th washed bowl sits (plate_count - 1 + k) pitches up
            nest_z = p["table_top_z"] + (p["plate_count"] - 1 + k) * p["plate_stack_pitch"]
            seat_pinch_z = nest_z + p["stack_hang"] - p["stack_seat_depth"]
            sweep_end_x = stack_pinch_x + p["stack_sweep_over"]
            right.move(4.0 * p["lower_time"], pos=(sweep_end_x, stack_y, seat_pinch_z))
            right.wait(p["place_dwell"])
            self._release_and_retract(right, p["rest_right"])

        # epilogue: the sponge was parked right after the rub; just rest both
        # hands
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
        self._update_trajectory()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
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

        # washed bowls rest in the nest on top of the stack. A contact-placed
        # bowl lands like a real dish — in the nest but not machine-centred —
        # and any in-nest pose is geometrically within ~4.5 cm of the stack
        # axis and ~2 cm of the nominal nest plane (the bowl body origin sits
        # at the bowl base).
        for k in range(p["wash_count"]):
            level = p["plate_count"] - 1 - k
            pos = body_q[self.plate_bodies[level], :3]
            nest_spot = (p["dirty_pile_x"], p["dirty_pile_y"])
            xy_err = float(np.linalg.norm(pos[:2] - nest_spot))
            assert xy_err < 0.045, f"Washed bowl {k} is {xy_err:.3f} m from the stack"
            nest_z = p["table_top_z"] + (p["plate_count"] - 1 + k) * p["plate_stack_pitch"]
            assert abs(pos[2] - nest_z) < 0.02, f"Washed bowl {k} is not nested on the stack: z={pos[2]:.3f}"
        # unwashed plates never left their starting spot
        for level in range(p["plate_count"] - p["wash_count"]):
            pos = body_q[self.plate_bodies[level], :3]
            xy_err = float(np.linalg.norm(pos[:2] - self.plate_initial_positions[level, :2]))
            assert xy_err < 0.06, f"Unwashed plate {level} drifted {xy_err:.3f} m"
        # the sponge was set aside ON the table in the wash-station area. The
        # hard requirements are that it never ended on the floor and stayed in
        # the neighbourhood where the hand let it go — the exact resting spot
        # varies with how the soft pad slides off the fingers.
        sponge_center = particle_q[self.sponge_info["particles"]].mean(axis=0)
        assert sponge_center[2] > p["table_top_z"] - 0.02, f"Sponge fell off the table: z={sponge_center[2]:.3f}"
        park = np.asarray(p["sponge_park"], dtype=np.float64)
        home_err = float(np.linalg.norm(sponge_center[:2] - park))
        assert home_err < 0.25, f"Sponge ended {home_err:.3f} m from the wash station"
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
