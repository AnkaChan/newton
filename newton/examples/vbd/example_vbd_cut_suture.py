# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example VBD Cut Suture

One elastic cuboid hangs in the air between two grippers. A blade, a thin capsule on a
kinematic holder, comes down on the middle of the block. When the contact force on the
blade edge reaches the cut force of the tissue, the solver cuts the tissue along the tet
faces under the edge, one tet row at a time, until the block falls into two halves. The
grippers pull the halves 4 mm apart and the blade leaves. A simulated half-circle needle
with a thread then sews the wound with a running suture, as in ``example_vbd_suture.py``:
the right gripper opens its upper jaw, the holder pulls the thread until the wound edges
meet, the lower jaw withdraws and the right half hangs from the stitches.

The cut edits array contents only (tet corners, spare particles, surface rows), so the
whole frame stays one captured CUDA graph.

With ``--cut-method xfem`` the example shows the XFEM-type cut instead, without the suture.
The blade plane is off the mesh planes, in the middle of a cell (``--blade-yaw-deg`` also
turns it about the vertical axis). The solver cuts through the tets where the blade is: a
cut tet becomes two fragments, its far-side corners become virtual corners, and wound
particles tied to the cut tet edges carry the collision triangles of the two wound faces.
After the cut the grippers pull the halves 4 mm apart, press the wound closed 1.5 mm past
the closed position and open it again. The two wound faces start one start gap (1 mm)
apart, so without self contact that press would take them 0.5 mm through each other; with
it they rest on each other. The needle stays parked above the block.

Limits of the XFEM-type cut scene:

- The self-contact damping is 0 (5.0 for the face cut): with the damping the pressed wound
  shivers at 20 iterations (see ``PARAMS["xfem_self_contact_kd"]``). The wound contact of this
  scene has no dissipation of its own.
- The suture does not know fragments yet, so no stitch is made and the suture script never
  starts. The solver warns about a model with needles; the example keeps the needle parked
  and does not show that warning.
- The blade force has an upper limit with the penalty law of the rigid-soft contact, and the
  scene runs above the value at which the solver warns (cut force 0.5 N, one contact record
  0.25 N). Many records carry the blade here and the cut went through in every run; the
  example does not show that warning. ``rigid_soft_contact_use_log_barrier=True`` removes
  the limit.
- With the blade turned, a wound face is stepped where the blade plane passes near a vertex,
  and some mesh edges cannot hold the full start gap (``cut_report()`` counts them).

Success criteria:
In test mode (``test_final``) the state is finite; exactly the tet faces of the cut plane
are cut and the tissue has two equal parts; the particle mass is unchanged; no cut capacity
overflowed; a copy never crosses its original; the blade penetrates tissue that it has
contact with by less than 0.5 mm; the grippers open the wound to its full width; then the
suture criteria of ``example_vbd_suture.py`` hold (two through tunnels per stitch, bounded
needle penetration and row residual, wound closed before the release, right half hanging
on the stitches).
With ``--cut-method xfem`` the state is finite; exactly the tets that the blade plane
crosses are cut, each once, and the tissue has two parts; the particle mass is unchanged;
no cut capacity and no self-contact list overflowed; no surface row mixes parked and live
particles, names a virtual corner or has an edge short enough to act as a point; the blade
edge never passes a crossed tet that is not cut; the particle speeds and the tie residual
stay below their bounds; the grippers open the wound to its full width; and while the
wound is pressed closed, past the start gap, the two wound faces do not pass through each
other, rest about one self-contact margin apart and are calm.

Runs:
- `python -m newton.examples vbd_cut_suture`
- `python -m newton.examples vbd_cut_suture --stitches 1 --num-frames 1260`
- `python -m newton.examples vbd_cut_suture --cut-arm wound --num-frames 400`
- `python -m newton.examples vbd_cut_suture --cut-method xfem --num-frames 593`
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.exceptions import NewtonWarning
from newton.solvers import SolverVBD

PARAMS = {
    "fps": 60,
    "sim_substeps": 10,
    # 20 iterations: the slack of the 0.2 mg thread capsules shivers at 10 iterations (velocity changes of about
    # 10 mm/s per frame); 20 iterations halve that. Contact damping, joint damping, a heavier thread and a softer
    # thread contact did not (VBD tuning guide, jitter pipeline, last resort).
    "solver_iterations": 20,
    # tissue
    "block_cells": (14, 10, 4),
    "cell": 3.0e-3,
    "gap": 4.0e-3,
    "tissue_density": 1000.0,
    "tissue_youngs_modulus": 1.0e5,
    "tissue_poisson": 0.45,
    # Volumetric damping [Pa s]. The contact values below already follow the tuning guide's stiffness-ratio rule
    # and remove the tissue jitter; a damping of 1e2 to 1e3 adds a start transient (the facing faces bulge 0.2
    # to 0.9 mm in the first 0.2 s, because the damping Hessian k_damp / dt exceeds the elastic stiffness).
    "tissue_damping": 0.0,
    "particle_radius": 5.0e-4,
    # blade: one capsule along y on a kinematic holder with the drive of the needle holder. It comes down on
    # the grid plane x = 0, goes through the block and leaves below it toward the camera side.
    "blade_radius": 4.5e-4,
    "blade_overhang": 2.0e-3,
    "blade_density": 7800.0,
    "blade_mu": 0.2,
    # A thin plate above the edge is the body of the blade. It has no contact; its mass gives the blade body a
    # well-conditioned inertia (one thin capsule alone has almost no inertia about its axis).
    "blade_plate_height": 10.0e-3,
    "blade_plate_thickness": 3.0e-4,
    "blade_clearance": 2.0e-3,
    "blade_depth": 3.0e-3,
    "blade_speed": 4.0e-3,
    "blade_out_dy": -0.060,
    "blade_park_z": -0.060,
    "t_blade_out": 0.75,
    "t_blade_park": 0.75,
    # Tissue cut force [N], per-particle attribute vbd.cut_force: the blade force along the cutting direction
    # that cuts the tissue under the blade edge.
    "cut_force": 0.5,
    # How the cut goes on below the first tet row (a fork of the design; see the task notes):
    # "gate": the blade has no contact with the split vertices (the wound lips), so it goes down to the cut
    #         front and the cut follows the blade edge row by row.
    # "wound": no gate; the blade rests on the lips and every cut update with enough blade force cuts the next
    #         tet row from the cut front.
    # "design": the rule as designed; after the first tet row the blade presses on the lips and later rows
    #         are cut late and unevenly, where the pressed tissue brings the cut front into the contact range.
    "cut_arm": "gate",
    "cut_update_interval": 1,
    # After the cut each gripper moves gap / 2 away from the cut plane, so the wound is `gap` wide for the suture.
    "t_pull_apart": 0.5,
    "t_settle": 0.25,
    # cut capacity: spare particles per color, cut faces, and the parking lattice behind and below the block
    "cut_max_vertices": 32,
    "cut_max_faces": 128,
    "park_position": (-0.0165, 0.060, -0.060),
    "park_spacing": 3.0e-3,
    # ── XFEM-type cut (--cut-method xfem). The face cut reads none of the values of this block. ──
    # The blade plane is off the mesh planes: x = 1.5 mm is the middle of a 3 mm cell. blade_yaw_deg turns the
    # blade about the vertical axis through (blade_x, 0).
    "blade_x": 1.5e-3,
    "blade_yaw_deg": 0.0,
    # Capacity, by the sizing rule of ModelBuilder.add_cut_capacity. The blade plane crosses T = 200 tets (one
    # layer of 10 x 4 cells of 5 tets) with V = 110 vertices, E = 149 mesh edges, F = 56 skin triangles and
    # S = 56 skin edges; the four pools give 110, 96, 104 and 98 spare particles. Turned by 8 degrees:
    # T = 224, V = 120, E = 167, F = 60, S = 60, pools 122, 107, 116 and 109. The values hold the turned scene
    # with a reserve of 14 percent (spare tets), 18 percent (largest pool) and 33 percent (skin rows):
    # max_tets >= T; max_vertices >= largest pool; 2 * max_faces >= 4 * max_tets + 2 * F;
    # max_edges >= 10 * max_tets + S + E.
    "xfem_max_tets": 256,
    "xfem_max_vertices": 144,
    "xfem_max_faces": 592,
    "xfem_max_edges": 2864,
    # The parking lattice has 4624 places (17 per side, 48 mm): behind and below the block, 20 mm above the
    # floor, away from the blade's way out (-y) and from the thread.
    "xfem_park_position": (-0.027, 0.060, -0.080),
    # Self-contact lists per vertex and per edge row (solver defaults 32 and 64): the lists hold the pairs of
    # the two wound faces and, before the solver's cut filter removes them, the pairs of one side with itself.
    # Measured in this scene: at most 36 triangles at one vertex and 65 edges at one edge row (turned by
    # 8 degrees: 47 and 94).
    "xfem_vertex_contact_buffer": 64,
    "xfem_edge_contact_buffer": 128,
    # Self-contact damping of this scene (it replaces self_contact_kd). With 5.0, the value of the face cut,
    # the pressed wound shivers at 20 iterations: wound particles and virtual corners move at 20 to 90 mm/s for
    # the whole hold. It is calm (below 2 mm/s) without the damping, and also with the damping at 40 iterations
    # or without gravity. The damping acts on closing pairs only, with up to 3000 N/m per pair (kd / dt)
    # against a contact stiffness of 500 N/m and a tie stiffness of 1000 N/m. Nothing else is in self contact
    # in this scene.
    "xfem_self_contact_kd": 0.0,
    # After the cut: the grippers pull the halves `gap` apart (t_pull_apart), hold, press the wound closed
    # press_past_closed beyond the closed position (each gripper goes back by gap / 2 plus half of it), hold,
    # open the wound to `gap` again and hold. Every move has the gripper speed of the pull (4 mm/s).
    # The press goes past the start gap of the wound faces (the self-contact margin, 1 mm): without self
    # contact the two wound faces would pass 0.5 mm through each other, so the pass-through checks of
    # test_final can fail.
    "press_past_closed": 1.5e-3,
    "t_hold_open": 1.0,
    "t_hold_pressed": 1.5,
    "t_hold_reopen": 1.0,
    # grippers: every particle farther than grip_offset from the wound is attached to the kinematic upper jaw, so
    # each block is a flat 12 mm cantilever at the wound and does not sag under its own weight. The right lower
    # jaw also collides with the tissue (friction jaw_mu): it carries the block after the upper jaw opens and the
    # block slides along it when the thread pulls. Its clearance equals the contact gap plus the particle radius,
    # so the block rests on it without a drop and without a contact force while it is still gripped.
    "grip_offset": 12.0e-3,
    "jaw_half_thickness": 1.5e-3,
    "jaw_clearance": 1.0e-3,
    "jaw_mu": 0.2,
    "open_dz": 20.0e-3,
    "open_dx": 60.0e-3,
    "floor_top": -0.10,
    # needle: half circle
    "needle_radius": 8.0e-3,
    "wire_radius": 4.0e-4,
    "needle_capsules": 12,
    "needle_density": 7800.0,
    # thread
    # Thicker and heavier than a real 0.15 mm nylon suture (a 2.5 mm capsule weighs 5.5 mg instead of 0.23 mg):
    # the light thread shivered and its slack crept (notes/vbd-suture/decisions.md I23 and I24), and the 0.5 mm
    # thread still reads as a thread next to the 0.8 mm needle wire.
    "thread_radius": 2.5e-4,
    "thread_segment": 2.5e-3,
    "thread_density": 1.0e4,
    "thread_stretch_stiffness": 2.0e3,
    # Bend and twist stiffness per joint [N m/rad] with damping at 1 percent of it, as in the cable examples.
    "thread_bend_stiffness": 5.0e-4,
    "thread_stretch_damping": 0.02,
    "thread_bend_damping": 5.0e-6,
    # Contact material of the thread and needle shapes (averaged with the soft side for rigid-soft contact).
    # The thread's 2e3 N/m joints set its scale: a contact 10 times stiffer than them jitters (tuning guide).
    # Friction 0.3 against tissue holds the slack where it lies; at 0.1 it drifted 8 mm in the last 3 s of the demo.
    "thread_mu": 0.3,
    "thread_contact_ke": 1.0e2,
    "thread_contact_kd": 1.0,
    "needle_mu": 0.2,
    # Free end of the thread: clamped beside the far corner of the left block. The route from the needle tail runs
    # to the far side (+y), down beside the blocks, and back up to the clamp, where neither the needle nor the
    # stitches can catch it. The y values are offsets from the first stitch and keep the loop and the clamp at
    # least 2 mm beyond the jaws (y = 17 mm) for a single stitch. The route below is the one-stitch route
    # (97 mm); its three low points drop by half of thread_slack_per_stitch for every further stitch, since
    # each stitch with its bridge draws about 30 mm, so that 10 to 30 mm of slack remains after the last bite and
    # the pull can tighten the thread within its travel.
    # Rest shape of the thread: False keeps the initial route as the rest shape (its corners travel through the
    # tunnels as rest kinks), True makes the rest shape straight (zero intrinsic bend and twist).
    "thread_rest_straight": True,
    "clamp_position": (-0.046, 0.019, 0.010),
    "thread_route": (
        (0.010, 0.019, 0.026),
        (0.010, 0.021, 0.010),
        (-0.015, 0.023, 0.006),
        (-0.040, 0.021, 0.010),
    ),
    "thread_slack_per_stitch": 0.024,
    # holder drive
    "drive_linear_ke": 1.0e4,
    "drive_linear_kd": 1.0e2,
    "drive_angular_ke": 10.0,
    "drive_angular_kd": 0.1,
    # contact
    "rigid_gap": 1.0e-3,
    "soft_contact_gap": 1.0e-3,
    "full_surface_contact": True,
    # Within 10x of the tissue shear stiffness (34 kPa) and of the thread joints, with 1 percent damping.
    "soft_contact_ke": 5.0e2,
    "soft_contact_kd": 5.0,
    "soft_contact_mu": 0.3,
    # Self-contact between the two wound faces: the solver reads Model.soft_contact_ke/kd for it and averages the
    # same values with the shape material for particle-shape contacts; the shape materials below are set so the
    # particle-shape averages stay at soft_contact_ke/kd (and at the thread values for the thread) whatever the
    # self-contact values are.
    "self_contact_ke": 5.0e2,
    "self_contact_kd": 5.0,
    "self_contact": True,
    "self_contact_margin": 1.0e-3,
    "self_contact_gap": 1.0e-3,
    # 0: detect contacts before and after the inertial step of every substep (default). N > 0: detect before the
    # inertial step and then before every N-th iteration (tuning guide, jitter protocol step 4); the rigid slot
    # follows and the pipeline matches contacts between detections.
    "detection_iterations": 0,
    # A cuttable model needs the live shared-vertex filter only (threshold 1) and real edge-edge contact for
    # millimetre edges (the default epsilon of 1e-5 treats edges of 3.16 mm or less as points).
    "self_contact_filter_threshold": 1,
    "edge_parallel_epsilon": 1.0e-7,
    # suture
    "tunnel_mu": 0.3,
    "grip": 1.0,
    # Tunnel row stiffness [N/m], per-particle attribute vbd.suture_stiffness (infinite = the hard row). The hard
    # rows fight each other where the capsule chain kinks (tunnel entry and exit, capsule joints, the swage) and
    # their multipliers run away to tens or hundreds of newtons, which tears the thread joints apart and yanks
    # the tissue; 1e4 N/m bounds the force at about 10 N per millimetre of offset and keeps the thread within
    # about 1 mm of the tunnel points (notes/vbd-suture/decisions.md I18, I19 and I27).
    "tunnel_stiffness": 1.0e4,
    # Continuous rod tangent for the rows (solver argument suture_smooth_tangent): with the segment direction the
    # transverse plane of a row jumps when its closest place crosses a kinked joint (tunnel entry and exit).
    "suture_smooth_tangent": True,
    "puncture_force": 0.1,
    "puncture_cos": 0.5,
    # holder script [s]
    "stitches": 3,
    "stitch_pitch": 8.0e-3,
    "t_approach": 0.75,
    "t_bite": 5.0,
    "t_hop": 0.75,
    # After the last stitch the right gripper opens its upper jaw (t_open_upper) and the block rests on the lower
    # jaw. The holder then pulls the needle toward pull_target_xz at constant speed (t_pull) while the drive force
    # stays below pull_force_max above its value at the start of the pull: the thread tightens, the slack of the
    # bridges and of the free loop is drawn through the stitches, and the right block slides along its jaw until
    # the wound edges meet, where the force rises to the cap and the pull holds. The lower jaw then withdraws
    # (t_open) and the hanging block tensions the stitches (t_hold).
    "t_open_upper": 0.5,
    "t_pull": 5.0,
    "pull_force_max": 0.15,
    "t_open": 0.5,
    "t_hold": 3.0,
    "bite_angle_deg": 360.0,
    # The hop lifts the arc center from center_bite to center_high; the thread drawn on the way down becomes the
    # slack the next bite consumes (about 25 mm), so a 26 mm lift keeps the bite from pulling thread through the
    # earlier stitches at its end (I27).
    "center_high": 0.040,
    "center_bite": 0.014,
    # Pull: the arc center moves toward this x and z (at the last stitch's y, up and over the left block) at
    # constant speed while the holder drive force stays below pull_force_max, and holds otherwise. The 56 mm of
    # travel exceed the slack to draw (bridges, free loop and the 4 mm gap: 20 to 40 mm).
    "pull_target_xz": (-0.050, 0.040),
    # camera
    "camera_pos": (0.0, -0.21, 0.06),
    "camera_target": (0.0, 0.0, -0.008),
    "camera_fov": 30.0,
}


def _resample_polyline(points: list[wp.vec3], segment_length: float) -> tuple[list[wp.vec3], float]:
    """Resample a polyline at equal arc length. Returns the points and the total length."""
    pts = np.array([[q[0], q[1], q[2]] for q in points], dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg.sum())
    if length <= 0.0:
        raise ValueError("thread route has zero length")
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    segments = max(2, int(round(length / segment_length)))
    result = []
    for k in range(segments + 1):
        s = length * k / segments
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] <= 0.0 else (s - cum[i]) / seg[i]
        result.append(wp.vec3(*(pts[i] + (pts[i + 1] - pts[i]) * min(f, 1.0))))
    return result, length


def _needle_rotation(psi: float) -> wp.quat:
    """Rotation about -y: the tip, at -x in the body frame, descends as ``psi`` grows."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, -1.0, 0.0), float(psi))


def schedule(stitches: int, p: dict | None = None) -> dict[str, float]:
    """Start times of the phases for ``stitches`` stitches [s]."""
    p = PARAMS if p is None else p
    stitch = p["t_approach"] + p["t_bite"] + p["t_hop"]
    open_upper = stitches * stitch - p["t_hop"]
    pull_start = open_upper + p["t_open_upper"]
    release = pull_start + p["t_pull"]
    end = release + p["t_open"] + p["t_hold"]
    return {"stitch": stitch, "open_upper": open_upper, "pull_start": pull_start, "release": release, "end": end}


def cut_schedule(p: dict | None = None) -> dict[str, float]:
    """Start times of the cut phases [s]; the suture schedule starts at ``suture_start``."""
    p = PARAMS if p is None else p
    height = p["block_cells"][2] * p["cell"]
    travel = p["blade_clearance"] + height + p["blade_depth"] + 2.0 * p["blade_radius"]
    cut_end = travel / p["blade_speed"]
    out_end = cut_end + p["t_blade_out"]
    park_end = out_end + p["t_blade_park"]
    suture_start = max(cut_end + p["t_pull_apart"], park_end) + p["t_settle"]
    return {
        "travel": travel,
        "cut_end": cut_end,
        "out_end": out_end,
        "park_end": park_end,
        "suture_start": suture_start,
    }


def xfem_schedule(p: dict | None = None) -> dict[str, float]:
    """Times of the phases after the XFEM-type cut [s]: open the wound, hold, press it closed, hold, open, hold."""
    p = PARAMS if p is None else p
    c = cut_schedule(p)
    # Each gripper moves gap / 2 in t_pull_apart; the press and the second opening keep that speed.
    t_press = p["t_pull_apart"] * (p["gap"] + p["press_past_closed"]) / p["gap"]
    open_end = c["cut_end"] + p["t_pull_apart"]
    press_start = max(open_end + p["t_hold_open"], c["park_end"])
    press_end = press_start + t_press
    reopen_start = press_end + p["t_hold_pressed"]
    reopen_end = reopen_start + t_press
    return {
        "open_start": c["cut_end"],
        "open_end": open_end,
        "press_start": press_start,
        "press_end": press_end,
        "reopen_start": reopen_start,
        "reopen_end": reopen_end,
        "end": reopen_end + p["t_hold_reopen"],
    }


def total_frames(stitches: int, p: dict | None = None, cut_method: str = "faces") -> int:
    p = PARAMS if p is None else p
    if cut_method == "xfem":
        return int(math.ceil(xfem_schedule(p)["end"] * p["fps"])) + 1
    return int(math.ceil((cut_schedule(p)["suture_start"] + schedule(stitches, p)["end"]) * p["fps"])) + 1


def _closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point to ``p`` on the triangle ``(a, b, c)``, for arrays that broadcast to ``[..., 3]``.

    The region tests are those of Ericson, Real-Time Collision Detection, section 5.1.5.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        on_ab = a + ab * (d1 / (d1 - d3))[..., None]
        on_ac = a + ac * (d2 / (d2 - d6))[..., None]
        on_bc = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]
        inside = a + (ab * vb[..., None] + ac * vc[..., None]) / (va + vb + vc)[..., None]
    shape = inside.shape
    regions = [
        (d1 <= 0.0) & (d2 <= 0.0),  # corner a
        (d3 >= 0.0) & (d4 <= d3),  # corner b
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),  # edge ab
        (d6 >= 0.0) & (d5 <= d6),  # corner c
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),  # edge ac
        (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0),  # edge bc
    ]
    places = [np.broadcast_to(place, shape) for place in (a, b, on_ab, c, on_ac, on_bc)]
    return np.select([region[..., None] for region in regions], places, default=inside)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = dict(PARAMS)
        self.params["cut_arm"] = getattr(args, "cut_arm", None) or PARAMS["cut_arm"]
        self.cut_method = getattr(args, "cut_method", None) or "faces"
        self.xfem = self.cut_method == "xfem"
        if self.xfem:
            if self.params["cut_arm"] != "gate":
                raise ValueError("--cut-method xfem runs with --cut-arm gate only")
            yaw = getattr(args, "blade_yaw_deg", None)
            if yaw is not None:
                self.params["blade_yaw_deg"] = float(yaw)
            self.params["self_contact_kd"] = self.params["xfem_self_contact_kd"]
        self.stitches = int(getattr(args, "stitches", None) or PARAMS["stitches"])
        self.times = schedule(self.stitches)
        self.cut_times = cut_schedule(self.params)
        self.t_suture_start = self.cut_times["suture_start"]
        if self.xfem:
            # No stitches with this method yet (the suture tet walk does not know fragments): the suture
            # script never starts, so the needle stays parked above the block.
            self.xfem_times = xfem_schedule(self.params)
            self.t_suture_start = math.inf
        self.frame_dt = 1.0 / self.params["fps"]
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        builder = newton.ModelBuilder()
        # Millimetre-scale scene: the default rigid detection gap would pair every thread capsule with every other.
        builder.rigid_gap = self.params["rigid_gap"]
        SolverVBD.register_custom_attributes(builder)
        self._build_floor(builder)
        self._build_tissue(builder)
        self._build_grippers(builder)
        self._build_needle_and_thread(builder)
        self._build_blade(builder)
        builder.color()
        # Face cut: copies of cut vertices live in a second color group per color, so the two wound faces never
        # share a Gauss-Seidel color pass (the role of the disjoint block colors in example_vbd_suture.py).
        # XFEM-type cut: that does not hold. A wound particle takes a spare of the pool of the vertex of its own
        # side, so the wound particles of both wound faces are in the same copy colors and many of their contact
        # pairs are solved in one color pass.
        p = self.params
        if self.xfem:
            # The XFEM-type cut also needs one spare tet per cut tet; its wound particles and virtual
            # corners come from the same pools, and the wound faces take more surface rows.
            builder.add_cut_capacity(
                max_vertices=p["xfem_max_vertices"],
                max_faces=p["xfem_max_faces"],
                max_edges=p["xfem_max_edges"],
                max_tets=p["xfem_max_tets"],
                park_position=p["xfem_park_position"],
                park_spacing=p["park_spacing"],
                particle_radius=p["particle_radius"],
            )
        else:
            builder.add_cut_capacity(
                max_vertices=p["cut_max_vertices"],
                max_faces=p["cut_max_faces"],
                park_position=p["park_position"],
                park_spacing=p["park_spacing"],
                particle_radius=p["particle_radius"],
            )
        self.spare_particles = np.arange(self.tissue_particle_count, builder.particle_count)
        self.model = builder.finalize()

        cut_force = self.model.vbd.cut_force.numpy()
        cut_force[: self.tissue_particle_count] = p["cut_force"]
        self.model.vbd.cut_force.assign(cut_force)
        self.model.soft_contact_ke = p["self_contact_ke"]
        self.model.soft_contact_kd = p["self_contact_kd"]
        self.model.soft_contact_mu = p["soft_contact_mu"]
        # particle-shape contact uses the average of the model values and the shape material
        self.model.shape_material_ke.fill_(2.0 * p["soft_contact_ke"] - p["self_contact_ke"])
        self.model.shape_material_kd.fill_(2.0 * p["soft_contact_kd"] - p["self_contact_kd"])
        shape_body = self.model.shape_body.numpy()
        thread_shapes = np.flatnonzero(np.isin(shape_body, self.thread_bodies))
        ke = self.model.shape_material_ke.numpy()
        kd = self.model.shape_material_kd.numpy()
        ke[thread_shapes] = p["soft_contact_ke"] + p["thread_contact_ke"] - p["self_contact_ke"]
        kd[thread_shapes] = p["soft_contact_kd"] + p["thread_contact_kd"] - p["self_contact_kd"]
        self.model.shape_material_ke.assign(ke)
        self.model.shape_material_kd.assign(kd)
        self.model.vbd.suture_grip.fill_(p["grip"])
        self.model.vbd.suture_mu.fill_(p["tunnel_mu"])
        self.model.vbd.suture_stiffness.fill_(p["tunnel_stiffness"])
        self.model.vbd.suture_puncture_force.fill_(p["puncture_force"])
        self.model.vbd.suture_puncture_cos.fill_(p["puncture_cos"])

        # Full-surface rigid-soft contact: the 0.3 mm thread lies between the 3 mm particle rows, so per-particle
        # contacts alone let it sink into the tissue; face and edge contacts hold it on the surface.
        schedule_kwargs = {}
        if p["detection_iterations"] > 0:
            slot = newton.solvers.SolverBase.CollisionSlot
            kind = newton.solvers.SolverBase.CollisionFrequencyType
            schedule_kwargs = {
                "collision_frequency_type": {slot.SOFT_SELF_CONTACT: kind.ITERATIONS, slot.RIGID: kind.ITERATIONS},
                "collision_frequency": {
                    slot.SOFT_SELF_CONTACT: p["detection_iterations"],
                    slot.RIGID: p["detection_iterations"],
                },
            }
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_gap=p["soft_contact_gap"],
            enable_rigid_soft_full_surface_contact=p["full_surface_contact"],
            contact_matching="latest" if p["detection_iterations"] > 0 else "disabled",
        )
        cut_kwargs = {}
        if self.xfem:
            cut_kwargs = {
                "cut_method": "xfem",
                "particle_vertex_contact_buffer_size": p["xfem_vertex_contact_buffer"],
                "particle_edge_contact_buffer_size": p["xfem_edge_contact_buffer"],
            }
        with warnings.catch_warnings():
            if self.xfem:
                # Two warnings of the solver that this scene knows and handles (see the module docstring):
                # the needle stays parked, so no tunnel meets cut tissue; and the cut force is above the
                # limit of one contact record of the blade edge, with many records under the long blade.
                warnings.filterwarnings("ignore", message=".*suture does not know fragments.*", category=NewtonWarning)
                warnings.filterwarnings("ignore", message=".*largest vbd.cut_force.*", category=NewtonWarning)
            self.solver = SolverVBD(
                self.model,
                iterations=p["solver_iterations"],
                **schedule_kwargs,
                **cut_kwargs,
                rigid_soft_enable_dat=True,
                particle_enable_self_contact=p["self_contact"],
                particle_self_contact_margin=p["self_contact_margin"],
                particle_self_contact_gap=p["self_contact_gap"],
                particle_topological_contact_filter_threshold=p["self_contact_filter_threshold"],
                particle_edge_parallel_epsilon=p["edge_parallel_epsilon"],
                suture_smooth_tangent=p["suture_smooth_tangent"],
                # The blade edge touches about ten full-surface records per tissue vertex along its whole length.
                rigid_body_particle_contact_buffer_size=1024,
                cut_update_interval=p["cut_update_interval"],
                cut_gate_wound=p["cut_arm"] == "gate",
                cut_press_wound=p["cut_arm"] == "wound",
                collision_pipeline=self.collision_pipeline,
            )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.max_penetration = 0.0
        self.max_blade_penetration = 0.0
        self.min_lip_gap = math.inf
        self.cut_checked = False
        self.particle_mass_0 = float(self.model.particle_mass.numpy().sum())
        self.max_residual = 0.0
        self.tunnels_seen = 0
        self.pull_progress = 0.0
        self.pull_force_base = None
        self.upper_open = False
        self.released = False
        self.gap_before_release = None
        if self.xfem:
            # The rest shape before any cut: the tets that the blade plane crosses are counted against it.
            self._rest_q = self.model.particle_q.numpy()[: self.tissue_particle_count].astype(np.float64)
            self._rest_tets = self.model.tet_indices.numpy().reshape(-1, 4)[: self.tissue_tet_count].copy()
            self.wound_marks = {}  # the wound, read once at the end of the open hold and of the pressed hold
            self.finite = True
            self.max_tie_residual = 0.0
            # Largest speed of each particle kind [m/s]: over the whole run, and while the wound is held closed.
            self.max_speed = {"tissue": 0.0, "wound": 0.0, "virtual": 0.0}
            self.max_speed_pressed = {"tissue": 0.0, "wound": 0.0, "virtual": 0.0}
            self.max_blade_lead = -math.inf
            self.max_contact_list = [0, 0]

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(*p["camera_pos"]), 0.0, 0.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = p["camera_fov"]
        self._look_at(wp.vec3(*p["camera_pos"]), wp.vec3(*p["camera_target"]))
        self.capture()

    # ── model construction ──────────────────────────────────────────────

    def _build_floor(self, builder):
        """A safety floor far below the blocks; a block that falls off the stitches lands on it."""
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, density=0.0)
        z = self.params["floor_top"] - 0.005
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), hx=0.15, hy=0.10, hz=0.005, cfg=cfg
        )

    def _build_tissue(self, builder):
        """One block of 2 x block_cells[0] cells along x; the cut plane is the grid plane x = 0."""
        p = self.params
        nx, ny, nz = p["block_cells"]
        cell = p["cell"]
        E, nu = p["tissue_youngs_modulus"], p["tissue_poisson"]
        k_mu = E / (2.0 * (1.0 + nu))
        k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        width = nx * cell
        depth = ny * cell
        start = builder.particle_count
        builder.add_soft_grid(
            pos=wp.vec3(-width, -0.5 * depth, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2 * nx,
            dim_y=ny,
            dim_z=nz,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            density=p["tissue_density"],
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=p["tissue_damping"],
            particle_radius=p["particle_radius"],
        )
        self.tissue_particle_count = builder.particle_count
        self.tissue_tet_count = builder.tet_count
        q = np.array([[v[0], v[1], v[2]] for v in builder.particle_q[start:]], dtype=np.float64)
        ids = np.arange(start, builder.particle_count)
        # Left half: x <= 0 (the particles of the cut plane stay with it); right half: x > 0, plus the copies
        # that the cut makes of the plane particles.
        self.block_particles = [ids[q[:, 0] <= 1.0e-9], ids[q[:, 0] > 1.0e-9]]
        self.plane_particles = ids[np.abs(q[:, 0]) <= 1.0e-9]
        self.tissue_height = nz * cell
        self.tissue_depth = depth

    def _build_grippers(self, builder):
        """One gripper per block: every particle farther than grip_offset from the wound is tied to a kinematic
        upper jaw by hard attachments; two jaw boxes show the grip. The right lower jaw also collides with the
        tissue: it carries the block after the upper jaw opens and the block slides along it when the thread
        pulls. The other jaws are visual."""
        p = self.params
        self.grippers = []
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        support_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_shape_collision=False, has_particle_collision=True, mu=p["jaw_mu"]
        )
        for block, particles in enumerate(self.block_particles):
            q = np.array([[v[0], v[1], v[2]] for v in (builder.particle_q[i] for i in particles)], dtype=np.float64)
            # distance from the cut plane x = 0 into the half
            depth = -q[:, 0] if block == 0 else q[:, 0]
            slab = np.flatnonzero(depth >= p["grip_offset"] - 1.0e-6)
            cx = float(0.5 * (q[slab, 0].min() + q[slab, 0].max()))
            cy = float(q[:, 1].mean())
            hx = 0.5 * float(q[slab, 0].max() - q[slab, 0].min()) + 1.0e-3
            hy = 0.5 * float(q[:, 1].max() - q[:, 1].min()) + 2.0e-3
            hz = p["jaw_half_thickness"]
            supports = block == 1
            lower_clearance = p["soft_contact_gap"] + p["particle_radius"] if supports else p["jaw_clearance"]
            upper_pos = np.array([cx, cy, float(q[:, 2].max()) + hz + p["jaw_clearance"]])
            lower_pos = np.array([cx, cy, float(q[:, 2].min()) - hz - lower_clearance])
            upper = builder.add_body(
                xform=wp.transform(wp.vec3(*upper_pos), wp.quat_identity()), label=f"gripper_{block}_upper"
            )
            lower = builder.add_body(
                xform=wp.transform(wp.vec3(*lower_pos), wp.quat_identity()), label=f"gripper_{block}_lower"
            )
            builder.add_shape_box(upper, hx=hx, hy=hy, hz=hz, cfg=cfg)
            builder.add_shape_box(lower, hx=hx, hy=hy, hz=hz, cfg=support_cfg if supports else cfg)
            attachments = [
                builder.add_attachment_particle_body(
                    int(particles[i]), upper, body_point=wp.vec3(*(q[i] - upper_pos)), stiffness=math.inf
                )
                for i in slab
            ]
            self.grippers.append(
                {
                    "upper": upper,
                    "lower": lower,
                    "upper_pos": upper_pos,
                    "lower_pos": lower_pos,
                    "attachments": attachments,
                }
            )

    def _stitch_y(self, i: int) -> float:
        """Stitches run from the far side (+y) to the near side (-y), centered on the blocks."""
        return (0.5 * (self.stitches - 1) - i) * self.params["stitch_pitch"]

    def _build_needle_and_thread(self, builder):
        p = self.params
        R = p["needle_radius"]
        r = p["wire_radius"]
        y0 = self._stitch_y(0)
        center = wp.vec3(0.0, y0, p["center_high"])
        rot0 = _needle_rotation(0.0)

        # Holder: a kinematic body at the arc center, driven by the script.
        self.holder = builder.add_body(xform=wp.transform(center, rot0), label="suture_holder")

        # Needle: the upper half circle in the body frame, tip at -x, tail at +x.
        self.needle = builder.add_body(xform=wp.transform(center, rot0), label="suture_needle")
        count = p["needle_capsules"]
        angles = [math.pi - math.pi * k / count for k in range(count + 1)]
        points = [wp.vec3(R * math.cos(a), 0.0, R * math.sin(a)) for a in angles]
        cfg = newton.ModelBuilder.ShapeConfig(density=p["needle_density"], mu=p["needle_mu"])
        self.needle_shapes = []
        for a, b in itertools.pairwise(points):
            mid = 0.5 * (a + b)
            d = b - a
            q = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), wp.normalize(d))
            half = max(0.5 * float(wp.length(d)) - 0.5 * r, 0.25 * r)
            self.needle_shapes.append(
                builder.add_shape_capsule(self.needle, xform=wp.transform(mid, q), radius=r, half_height=half, cfg=cfg)
            )
        tip_shape = self.needle_shapes[0]

        linear = [
            newton.ModelBuilder.JointDofConfig(
                axis=axis, target_ke=p["drive_linear_ke"], target_kd=p["drive_linear_kd"]
            )
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ]
        angular = [
            newton.ModelBuilder.JointDofConfig(
                axis=axis, target_ke=p["drive_angular_ke"], target_kd=p["drive_angular_kd"]
            )
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ]
        builder.add_joint_d6(
            self.holder, self.needle, linear_axes=linear, angular_axes=angular, label="suture_holder_drive"
        )

        # Thread: a rod from the tail to the far side, down below the blocks and back up to the clamp.
        tail_local = points[-1]
        tail_world = wp.transform_point(wp.transform(center, rot0), tail_local)
        shift = wp.vec3(0.0, y0, 0.0)
        clamp_world = wp.vec3(*p["clamp_position"]) + shift
        drop = wp.vec3(0.0, 0.0, -0.5 * p["thread_slack_per_stitch"] * (self.stitches - 1))
        low = [wp.vec3(*q) + shift + drop for q in p["thread_route"][1:]]
        route = [tail_world, wp.vec3(*p["thread_route"][0]) + shift, *low, clamp_world]
        polyline, thread_length = _resample_polyline(route, p["thread_segment"])
        segments = len(polyline) - 1
        self.thread_length = thread_length
        rod = newton.Rod(points=polyline, radius=p["thread_radius"], rest_straight=p["thread_rest_straight"])
        thread_dir = wp.normalize(polyline[1] - polyline[0])
        thread_cfg = newton.ModelBuilder.ShapeConfig(density=p["thread_density"], mu=p["thread_mu"])
        self.thread_bodies, _joints = builder.add_rod(
            rod=rod,
            cfg=thread_cfg,
            stretch_stiffness=p["thread_stretch_stiffness"],
            bend_stiffness=p["thread_bend_stiffness"],
            stretch_damping=p["thread_stretch_damping"],
            bend_damping=p["thread_bend_damping"],
            label="suture_thread",
        )
        half_seg = 0.5 * thread_length / segments
        q_tail = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), wp.quat_rotate_inv(rot0, thread_dir))
        builder.add_joint_rod(
            self.needle,
            self.thread_bodies[0],
            parent_xform=wp.transform(tail_local, q_tail),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -half_seg), wp.quat_identity()),
            stretch_stiffness=p["thread_stretch_stiffness"],
            bend_stiffness=p["thread_bend_stiffness"],
            stretch_damping=p["thread_stretch_damping"],
            bend_damping=p["thread_bend_damping"],
            label="suture_swage",
        )
        clamp = builder.add_body(xform=wp.transform(clamp_world, wp.quat_identity()), label="suture_clamp")
        builder.add_joint_fixed(
            clamp,
            self.thread_bodies[-1],
            child_xform=wp.transform(wp.vec3(0.0, 0.0, half_seg), wp.quat_identity()),
            label="suture_clamp_joint",
        )
        builder.add_suture_needle(
            self.needle,
            tip_shape,
            centerline=points,
            thread_bodies=self.thread_bodies,
            max_tunnels=2 * self.stitches + 2,
        )

    def _build_blade(self, builder):
        """A capsule along y on a kinematic holder; the D6 drive of the needle holder presses it on the tissue."""
        p = self.params
        r = p["blade_radius"]
        self.blade_start = np.array([0.0, 0.0, self.tissue_height + p["blade_clearance"] + r])
        self.blade_rotation = wp.quat_identity()
        if self.xfem:
            # Off the mesh planes: the blade plane goes through (blade_x, 0) and is turned about the vertical axis.
            self.blade_start[0] = p["blade_x"]
            self.blade_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(p["blade_yaw_deg"]))
        xf = wp.transform(wp.vec3(*self.blade_start), self.blade_rotation)
        self.blade_holder = builder.add_body(xform=xf, label="cut_blade_holder")
        self.blade = builder.add_body(xform=xf, label="cut_blade")
        cfg = newton.ModelBuilder.ShapeConfig(density=p["blade_density"], mu=p["blade_mu"], has_shape_collision=False)
        z_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
        self.blade_shape = builder.add_shape_capsule(
            self.blade,
            xform=wp.transform(wp.vec3(0.0), z_to_y),
            radius=r,
            half_height=0.5 * self.tissue_depth + p["blade_overhang"],
            cfg=cfg,
        )
        if p["blade_plate_height"] > 0.0:
            visual = newton.ModelBuilder.ShapeConfig(
                density=p["blade_density"], has_shape_collision=False, has_particle_collision=False
            )
            builder.add_shape_box(
                self.blade,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * p["blade_plate_height"]), wp.quat_identity()),
                hx=0.5 * p["blade_plate_thickness"],
                hy=0.5 * self.tissue_depth + p["blade_overhang"],
                hz=0.5 * p["blade_plate_height"],
                cfg=visual,
            )
        linear = [
            newton.ModelBuilder.JointDofConfig(
                axis=axis, target_ke=p["drive_linear_ke"], target_kd=p["drive_linear_kd"]
            )
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ]
        angular = [
            newton.ModelBuilder.JointDofConfig(
                axis=axis, target_ke=p["drive_angular_ke"], target_kd=p["drive_angular_kd"]
            )
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ]
        builder.add_joint_d6(
            self.blade_holder, self.blade, linear_axes=linear, angular_axes=angular, label="cut_blade_drive"
        )
        builder.add_cut_blade(self.blade, self.blade_shape, cut_direction=(0.0, 0.0, -1.0))

    def _blade_holder_pose(self, t: float) -> wp.transform:
        """Down through the block at blade_speed, out toward the camera side, then down to the parking place."""
        p = self.params
        c = self.cut_times
        pos = self.blade_start.copy()
        pos[2] -= min(t, c["cut_end"]) * p["blade_speed"]
        if t > c["cut_end"]:
            pos[1] += p["blade_out_dy"] * min((t - c["cut_end"]) / p["t_blade_out"], 1.0)
        if t > c["out_end"]:
            s = min((t - c["out_end"]) / p["t_blade_park"], 1.0)
            pos[2] += (p["blade_park_z"] - pos[2]) * s
        return wp.transform(wp.vec3(*pos), self.blade_rotation)

    def _gripper_shift(self, block: int, t: float) -> np.ndarray:
        """After the cut each gripper moves gap / 2 away from the cut plane.

        With the XFEM-type cut the grippers then press the wound closed, press_past_closed beyond the
        closed position, and open it again (:func:`xfem_schedule`).
        """
        p = self.params
        if self.xfem:
            x = self.xfem_times
            times = [
                x["open_start"],
                x["open_end"],
                x["press_start"],
                x["press_end"],
                x["reopen_start"],
                x["reopen_end"],
            ]
            opening = [0.0, p["gap"], p["gap"], -p["press_past_closed"], -p["press_past_closed"], p["gap"]]
            return np.array([(-0.5 if block == 0 else 0.5) * float(np.interp(t, times, opening)), 0.0, 0.0])
        s = min(max((t - self.cut_times["cut_end"]) / p["t_pull_apart"], 0.0), 1.0)
        return np.array([(-0.5 if block == 0 else 0.5) * p["gap"] * s, 0.0, 0.0])

    # ── holder and gripper script ───────────────────────────────────────

    def _holder_pose(self, t: float) -> wp.transform:
        p = self.params
        psi_max = math.radians(p["bite_angle_deg"])
        high, bite = p["center_high"], p["center_bite"]
        if t < self.times["open_upper"]:
            i = min(int(t // self.times["stitch"]), self.stitches - 1)
            tau = t - i * self.times["stitch"]
            y = self._stitch_y(i)
            if tau < p["t_approach"]:
                z = high + (bite - high) * tau / p["t_approach"]
                return wp.transform(wp.vec3(0.0, y, z), _needle_rotation(0.0))
            tau -= p["t_approach"]
            if tau < p["t_bite"]:
                return wp.transform(wp.vec3(0.0, y, bite), _needle_rotation(psi_max * tau / p["t_bite"]))
            # hop: lift and move to the next stitch with the needle back at its start orientation
            s = min((tau - p["t_bite"]) / p["t_hop"], 1.0)
            y_next = self._stitch_y(i + 1)
            return wp.transform(wp.vec3(0.0, y + (y_next - y) * s, bite + (high - bite) * s), _needle_rotation(psi_max))
        # Pull: the center moves toward the target as far as self.pull_progress allows.
        y = self._stitch_y(self.stitches - 1)
        target = np.array([p["pull_target_xz"][0], y, p["pull_target_xz"][1]])
        start = np.array([0.0, y, bite])
        c = start + (target - start) * self.pull_progress
        return wp.transform(wp.vec3(*c), _needle_rotation(psi_max))

    def drive_force(self) -> float:
        """Force of the holder drive on the needle [N], from the drive stiffness and the position lag."""
        body_q = self.state_0.body_q.numpy()
        return float(np.linalg.norm(body_q[self.needle, :3] - body_q[self.holder, :3]) * self.params["drive_linear_ke"])

    def _advance_pull(self):
        """Tighten at constant speed while the thread is slack; hold once the drive force has risen by
        pull_force_max above its value at the start of the phase (the thread is taut)."""
        p = self.params
        t = self.sim_time - self.t_suture_start
        if (
            p["t_pull"] <= 0.0
            or t < self.times["pull_start"]
            or t >= self.times["release"]
            or self.pull_progress >= 1.0
        ):
            return
        force = self.drive_force()
        if self.pull_force_base is None:
            self.pull_force_base = force
        if force - self.pull_force_base < p["pull_force_max"]:
            self.pull_progress = min(1.0, self.pull_progress + self.frame_dt / p["t_pull"])

    def _open_upper(self):
        """Switch the right gripper's attachments off; the block rests on the lower jaw, the upper jaw lifts."""
        enabled = self.model.attachment_particle_body_enabled.numpy()
        enabled[self.grippers[1]["attachments"]] = 0
        self.model.attachment_particle_body_enabled.assign(enabled)
        self.upper_open = True

    def _release(self):
        """Record the gap as the lower jaw starts to withdraw; the block then hangs from the stitches."""
        self.gap_before_release = self.gap_width()
        self.released = True

    def _set_kinematics(self, t_scene: float):
        p = self.params
        t = max(t_scene - self.t_suture_start, 0.0)  # suture time
        body_q = self.state_0.body_q.numpy()
        body_q[self.blade_holder] = self._blade_holder_pose(t_scene)
        body_q[self.holder] = self._holder_pose(t)
        for block, g in enumerate(self.grippers):
            shift = self._gripper_shift(block, t_scene)
            upper = g["upper_pos"] + shift
            lower = g["lower_pos"] + shift
            if block == 1 and t_scene >= self.t_suture_start + self.times["open_upper"]:
                # the right upper jaw lifts and withdraws; the lower jaw carries the block until the release
                if not self.upper_open:
                    self._open_upper()
                su = min((t - self.times["open_upper"]) / p["t_open_upper"], 1.0)
                upper = upper + np.array([p["open_dx"] * su, 0.0, p["open_dz"] * su])
                if t >= self.times["release"]:
                    if not self.released:
                        self._release()
                    so = min((t - self.times["release"]) / p["t_open"], 1.0)
                    lower = lower + np.array([p["open_dx"] * so, 0.0, -p["open_dz"] * so])
            body_q[g["upper"]] = wp.transform(wp.vec3(*upper), wp.quat_identity())
            body_q[g["lower"]] = wp.transform(wp.vec3(*lower), wp.quat_identity())
        self.state_0.body_q.assign(body_q)

    # ── stepping ────────────────────────────────────────────────────────

    def simulate(self):
        # The cut rewrites triangle and edge rows in place; a refit alone lets the tree quality decay.
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.sim_substeps % 2 == 0:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def step(self):
        if not self.cut_checked and self.sim_time >= self.cut_times["cut_end"]:
            # The blade is through: read the cut counters once, outside the captured graph.
            self.cut_checked = True
            report = self.solver.cut_report()
            if self.xfem:
                if sum(report["overflow"]) != 0:
                    wp.utils.warn(
                        f"cut capacity overflow {report['overflow']} (pools, triangle rows, edge rows, spare tets, "
                        "tet records); raise xfem_max_vertices, xfem_max_faces, xfem_max_edges or xfem_max_tets"
                    )
            elif sum(report["overflow"]) != 0 or report["unsafe"] != 0:
                wp.utils.warn(
                    f"cut capacity overflow {report['overflow']} or unsafe start gap {report['unsafe']}; "
                    "raise cut_max_vertices or cut_max_faces"
                )
        if self.xfem:
            # Read the wound once at the end of the open hold and of the pressed hold, outside the captured graph.
            for name, key in (("open", "press_start"), ("pressed", "reopen_start")):
                if name not in self.wound_marks and self.sim_time >= self.xfem_times[key] - 1.0e-9:
                    self.wound_marks[name] = self.wound_state()
        elif not hasattr(self, "_face_a") and self.sim_time >= self.t_suture_start - self.frame_dt:
            self._init_faces()  # the cut is done: the two wound faces are known
        self._advance_pull()
        self._set_kinematics(self.sim_time)
        self.frame += 1
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ── validation ──────────────────────────────────────────────────────

    def _look_at(self, pos, target):
        if not hasattr(self.viewer, "camera"):
            return
        d = target - pos
        dist_xy = math.hypot(d[0], d[1])
        self.viewer.camera.yaw = math.degrees(math.atan2(d[1], d[0]))
        self.viewer.camera.pitch = math.degrees(math.atan2(d[2], dist_xy))

    def _right_particles(self) -> np.ndarray:
        """Particles of the right half: the tissue with x > 0 at rest and every copy the cut made."""
        active = (self.model.particle_flags.numpy()[self.spare_particles] & int(newton.ParticleFlags.ACTIVE)) != 0
        return np.concatenate([self.block_particles[1], self.spare_particles[active]])

    def gap_width(self) -> float:
        """Mean separation of the two wound faces along x: the copies of the cut plane particles against the originals."""
        q = self.state_0.particle_q.numpy()
        return float(q[self._face_b, 0].mean() - q[self._face_a, 0].mean())

    def edge_distance(self) -> float:
        """Distance between the centroids of the two wound faces [m]."""
        q = self.state_0.particle_q.numpy()
        return float(np.linalg.norm(q[self._face_b].mean(axis=0) - q[self._face_a].mean(axis=0)))

    def right_block_lowest(self) -> float:
        """Lowest particle of the right half [m]; it must stay well above the floor after the release."""
        q = self.state_0.particle_q.numpy()
        return float(q[self._right_particles(), 2].min())

    def _init_faces(self):
        """The two wound faces, once the cut is done: the plane particles and their copies."""
        active = (self.model.particle_flags.numpy()[self.spare_particles] & int(newton.ParticleFlags.ACTIVE)) != 0
        self._face_a = self.plane_particles
        self._face_b = self.spare_particles[active]
        if len(self._face_b) == 0:
            self._face_b = self.plane_particles  # the cut made no copy; the gap reads zero
        self._edge_distance_0 = self.edge_distance()
        self._gap_open = self.gap_width()

    def lip_gap(self) -> float:
        """Smallest distance along +x from an original of the cut plane to its copy [m]; infinite before any split.

        A copy starts at the place of its original and moves to the +x side of the wound; a value of
        zero or less means that a wound vertex crossed to the wrong side.
        """
        active = (self.model.particle_flags.numpy()[self.spare_particles] & int(newton.ParticleFlags.ACTIVE)) != 0
        copies = self.spare_particles[active]
        if len(copies) == 0:
            return math.inf
        rest = self.model.particle_q.numpy()
        q = self.state_0.particle_q.numpy()
        plane_rest = rest[self.plane_particles]
        gap = math.inf
        for copy in copies:
            original = self.plane_particles[int(np.argmin(np.linalg.norm(plane_rest - rest[copy], axis=1)))]
            gap = min(gap, float(q[copy, 0] - q[original, 0]))
        return gap

    def tissue_parts(self) -> list[int]:
        """Tet counts of the connected parts of the tissue (connected through shared particles)."""
        tets = self.model.tet_indices.numpy().reshape(-1, 4)[: self.tissue_tet_count]
        parent = np.arange(self.model.particle_count)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for tet in tets:
            root = find(int(tet[0]))
            for v in tet[1:]:
                parent[find(int(v))] = root
        roots = {}
        for tet in tets:
            root = find(int(tet[0]))
            roots[root] = roots.get(root, 0) + 1
        return sorted(roots.values())

    def blade_penetration(self) -> float:
        """Deepest penetration of the blade edge into tissue particles that it has contact with [m].

        With the gate arm the split vertices and their copies have no contact with the blade and are
        left out; the vertices of the cut front stay in.
        """
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        flags = self.model.particle_flags.numpy()
        blade_bits = int(self.model.shape_contact_gate.numpy()[self.blade_shape])
        gated = (self.model.particle_contact_gate.numpy() & blade_bits) != 0
        tissue = np.zeros(len(q), dtype=bool)
        tissue[: self.tissue_particle_count] = True
        tissue[self.spare_particles] = (flags[self.spare_particles] & int(newton.ParticleFlags.ACTIVE)) != 0
        free = tissue & ~gated
        if not free.any():
            return 0.0
        shape = self.blade_shape
        xf = wp.transform_multiply(
            wp.transform(*body_q[self.blade]), wp.transform(*self.model.shape_transform.numpy()[shape])
        )
        scale = self.model.shape_scale.numpy()[shape]
        a = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, -float(scale[1]))))
        b = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, float(scale[1]))))
        d = b - a
        pts = q[free]
        t = np.clip(((pts - a) @ d) / float(np.dot(d, d)), 0.0, 1.0)
        dist = np.linalg.norm(pts - (a + t[:, None] * d), axis=1)
        return max(float(np.max(float(scale[0]) + self.model.particle_radius.numpy()[free] - dist)), 0.0)

    # ── validation of the XFEM-type cut ─────────────────────────────────

    def blade_plane_normal(self) -> np.ndarray:
        """Normal of the nominal blade plane; side B of the cut is the side that it points to."""
        yaw = math.radians(self.params["blade_yaw_deg"])
        return np.array([math.cos(yaw), math.sin(yaw), 0.0])

    def crossed_tets(self) -> tuple[np.ndarray, float]:
        """Tets of the rest shape that the nominal blade plane crosses, and the smallest distance of a vertex from it [m]."""
        origin = np.array([self.params["blade_x"], 0.0, 0.0])
        level = (self._rest_q - origin) @ self.blade_plane_normal()
        side = level[self._rest_tets] >= 0.0
        return np.flatnonzero(side.any(axis=1) & ~side.all(axis=1)), float(np.abs(level).min())

    def tissue_row_parts(self) -> list[int]:
        """Counts of tets and fragment rows in each connected part of the tissue."""
        tets = self.model.tet_indices.numpy().reshape(-1, 4)
        rows = np.flatnonzero(self.solver._xfem["tet_side"].numpy() >= 0)
        parent = np.arange(self.model.particle_count)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            root = find(int(tets[row, 0]))
            for v in tets[row, 1:]:
                parent[find(int(v))] = root
        counts = {}
        for row in rows:
            root = find(int(tets[row, 0]))
            counts[root] = counts.get(root, 0) + 1
        return sorted(counts.values())

    def blade_lead(self) -> float:
        """How far the blade edge is past a crossed tet that is not cut [m]; negative while the cut leads the blade.

        The cut leads the blade edge: a tet that the blade plane crosses is cut before the edge axis
        reaches it. For every crossed tet that is not cut, the value takes the corner that is
        farthest along the cutting direction and measures how far the blade edge axis is past it.
        A positive value means that the blade edge went through tissue that it did not cut.
        """
        nominal, _nearest = self.crossed_tets()
        uncut = nominal[self.solver._xfem["tet_side"].numpy()[nominal] == 0]
        if len(uncut) == 0:
            return -math.inf
        q = self.state_0.particle_q.numpy()
        blade_q = wp.transform(*self.state_0.body_q.numpy()[self.blade])
        axis_point = np.array(wp.transform_get_translation(blade_q))  # the edge capsule is centred on the body
        direction = np.array(wp.quat_rotate(wp.transform_get_rotation(blade_q), wp.vec3(0.0, 0.0, -1.0)))
        corners = q[self.model.tet_indices.numpy().reshape(-1, 4)[uncut]]  # [T, 4, 3]
        ahead = (corners - axis_point) @ direction  # [T, 4]; positive: ahead of the edge axis
        return float((-ahead.max(axis=1)).max())

    def surface_row_errors(self) -> dict[str, int]:
        """Triangle and edge rows that mix parked and live particles, name a virtual corner, or have a degenerate edge.

        An edge of a live row is degenerate when it is shorter than the square root of
        ``edge_parallel_epsilon``: such an edge acts as a point in edge-edge self contact.
        """
        capacity = self.model.cut_capacity
        flags = self.model.particle_flags.numpy()
        q = self.state_0.particle_q.numpy()
        ghost = self.solver._xfem["particle_ghost"].numpy()
        parked = np.zeros(self.model.particle_count, dtype=bool)
        parked[capacity["particle_start"] : capacity["particle_end"]] = True
        parked &= (flags & int(newton.ParticleFlags.ACTIVE)) == 0
        triangles = self.model.tri_indices.numpy().reshape(-1, 3)
        edges = self.model.edge_indices.numpy().reshape(-1, 4)[:, 2:]
        triangle_parked = parked[triangles].sum(axis=1)
        edge_parked = parked[edges].sum(axis=1)
        live_triangles = triangles[triangle_parked == 0]
        live_edges = edges[edge_parked == 0]
        x = q[live_triangles]
        shortest = np.min(
            [np.linalg.norm(x[:, a] - x[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))],
            axis=0,
        )
        edge_length = np.linalg.norm(q[live_edges[:, 0]] - q[live_edges[:, 1]], axis=1)
        point_length = math.sqrt(self.params["edge_parallel_epsilon"])
        return {
            "mixed": int(((triangle_parked > 0) & (triangle_parked < 3)).sum()) + int((edge_parked == 1).sum()),
            "virtual_corner": int(ghost[live_triangles].any(axis=1).sum()) + int(ghost[live_edges].any(axis=1).sum()),
            "degenerate": int((shortest < point_length).sum()) + int((edge_length < point_length).sum()),
            "shortest": float(min(shortest.min(), edge_length.min())),
        }

    def wound_state(self) -> dict[str, float]:
        """The two wound faces, read on the host [m].

        ``gap_mean`` and ``gap_min`` are the mean and the smallest distance along the blade plane normal
        between the two wound particles of one cut mesh edge (side B minus side A). ``face_distance`` is
        the smallest signed distance of a wound particle from the wound face of the other side: the
        distance to the nearest wound triangle of that side, negative when the particle is behind it.
        ``mean_plane_distance`` is the smallest distance along the normal of a wound particle from the
        mean place of the wound particles of the other side. A value of zero or less means that the
        wound faces passed through each other. ``tie_residual`` is the largest distance of a wound
        particle from its tie target. All values are ``nan`` before the first cut advance.
        """
        x = self.solver._xfem
        q = self.state_0.particle_q.numpy().astype(np.float64)
        normal = self.blade_plane_normal()
        side = x["wound_side"].numpy()
        edge_wound = x["edge_wound"].numpy()
        pairs = edge_wound[(edge_wound[:, 0] >= 0) & (edge_wound[:, 1] >= 0)]
        state = dict.fromkeys(("gap_mean", "gap_min", "face_distance", "mean_plane_distance", "tie_residual"), math.nan)
        if len(pairs) == 0:
            return state
        gap = (q[pairs[:, 1]] - q[pairs[:, 0]]) @ normal
        state["gap_mean"] = float(gap.mean())
        state["gap_min"] = float(gap.min())
        level_a = q[side == 0] @ normal
        level_b = q[side == 1] @ normal
        state["mean_plane_distance"] = float(min(level_b.min() - level_a.mean(), level_b.mean() - level_a.max()))
        triangles = self.model.tri_indices.numpy().reshape(-1, 3)
        flags = self.model.particle_flags.numpy()
        wound_rows = (x["tri_kind"].numpy() == 2) & (flags[triangles[:, 0]] != 0)  # wound triangle rows in use
        row_side = x["tri_sub"].numpy() // 2
        smallest = math.inf
        for face_side in (0, 1):
            corners = q[triangles[wound_rows & (row_side == face_side)]]  # [F, 3, 3]
            points = q[side == 1 - face_side]  # [P, 3]
            if len(corners) == 0 or len(points) == 0:
                continue
            a, b, c = corners[None, :, 0], corners[None, :, 1], corners[None, :, 2]
            offset = points[:, None, :] - _closest_points_on_triangles(points[:, None, :], a, b, c)  # [P, F, 3]
            distance = np.linalg.norm(offset, axis=2)
            nearest = np.argmin(distance, axis=1)
            rows = np.arange(len(points))
            face_normal = np.cross(corners[nearest, 1] - corners[nearest, 0], corners[nearest, 2] - corners[nearest, 0])
            front = (offset[rows, nearest] * face_normal).sum(axis=1) >= 0.0
            smallest = min(smallest, float(np.where(front, distance[rows, nearest], -distance[rows, nearest]).min()))
        state["face_distance"] = smallest
        residual = x["tie_residual"].numpy()[x["wound_edge"].numpy() >= 0]
        state["tie_residual"] = float(residual.max()) if len(residual) else 0.0
        return state

    def _track_xfem(self):
        """Per-frame readings of the XFEM-type cut (test mode): finite state, largest speeds, tie residual, list sizes."""
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        self.finite = bool(
            self.finite
            and np.isfinite(q).all()
            and np.isfinite(qd).all()
            and np.isfinite(self.state_0.body_q.numpy()).all()
        )
        x = self.solver._xfem
        active = (self.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        speed = np.where(active, np.linalg.norm(qd, axis=1), 0.0)
        wound = x["wound_edge"].numpy() >= 0
        virtual = x["particle_ghost"].numpy() != 0
        tissue = ~wound & ~virtual
        # The pressed hold, without its first quarter second: the wound settles after the grippers stop.
        times = self.xfem_times
        held_closed = times["press_end"] + 0.25 <= self.sim_time <= times["reopen_start"] + 1.0e-9
        for kind, mask in (("tissue", tissue), ("wound", wound), ("virtual", virtual)):
            if mask.any():
                largest = float(speed[mask].max())
                self.max_speed[kind] = max(self.max_speed[kind], largest)
                if held_closed:
                    self.max_speed_pressed[kind] = max(self.max_speed_pressed[kind], largest)
        if wound.any():
            self.max_tie_residual = max(self.max_tie_residual, float(x["tie_residual"].numpy()[wound].max()))
        info = self.solver.trimesh_collision_info
        self.max_contact_list[0] = max(
            self.max_contact_list[0], int(info.vertex_colliding_triangles_count.numpy().max())
        )
        self.max_contact_list[1] = max(self.max_contact_list[1], int(info.edge_colliding_edges_count.numpy().max()))

    def max_row_residual(self) -> float:
        """Largest transverse distance from an active tunnel point to its rod place [m]."""
        solver = self.solver
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        tets = solver._suture_point_tet.numpy()
        active = solver._suture_point_active.numpy()
        segment = solver._suture_point_segment.numpy()
        if not ((tets >= 0) & (active != 0) & (segment >= 0)).any():
            return 0.0
        bary = solver._suture_point_bary.numpy()
        u = solver._suture_point_u.numpy()
        tangent = solver._suture_point_tangent.numpy()
        tet_indices = self.model.tet_indices.numpy().reshape(-1, 4)
        seg_body = self.model.suture_segment_body.numpy()
        seg_a = self.model.suture_segment_local_a.numpy()
        seg_b = self.model.suture_segment_local_b.numpy()
        worst = 0.0
        for i in np.flatnonzero((tets >= 0) & (active != 0) & (segment >= 0)):
            x = sum(bary[i][k] * q[tet_indices[tets[i]][k]] for k in range(4))
            s = segment[i]
            xf = wp.transform(*body_q[seg_body[s]])
            local = seg_a[s] + u[i] * (seg_b[s] - seg_a[s])
            place = np.array(wp.transform_point(xf, wp.vec3(*local)))
            t = tangent[i]
            d = x - place
            d -= np.dot(d, t) * t
            worst = max(worst, float(np.linalg.norm(d)))
        return worst

    def needle_penetration(self) -> float:
        """Deepest penetration of the needle group's capsules into ungated tissue particles [m]."""
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        gate = self.model.particle_contact_gate.numpy()
        shape_gate = self.model.shape_contact_gate.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_xf = self.model.shape_transform.numpy()
        shape_scale = self.model.shape_scale.numpy()
        radius = self.model.particle_radius.numpy()
        flags = self.model.particle_flags.numpy()
        tissue = np.zeros(len(q), dtype=bool)
        tissue[: self.tissue_particle_count] = True
        tissue[self.spare_particles] = (flags[self.spare_particles] & int(newton.ParticleFlags.ACTIVE)) != 0
        needle_bits = int(np.bitwise_or.reduce(self.model.suture_needle_group.numpy()))
        free = tissue & ((gate & needle_bits) == 0)
        if not free.any():
            return 0.0
        pts = q[free]
        rad = radius[free]
        worst = 0.0
        for shape in np.flatnonzero((shape_gate & needle_bits) != 0):
            body = shape_body[shape]
            xf = wp.transform_multiply(wp.transform(*body_q[body]), wp.transform(*shape_xf[shape]))
            r_cap, half = float(shape_scale[shape][0]), float(shape_scale[shape][1])
            a = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, -half)))
            b = np.array(wp.transform_point(xf, wp.vec3(0.0, 0.0, half)))
            d = b - a
            dd = float(np.dot(d, d))
            t = np.clip(((pts - a) @ d) / dd, 0.0, 1.0) if dd > 0.0 else np.zeros(len(pts))
            closest = a + t[:, None] * d
            dist = np.linalg.norm(pts - closest, axis=1)
            worst = max(worst, float(np.max(r_cap + rad - dist)))
        return max(worst, 0.0)

    def test_post_step(self):
        cutting = self.sim_time < self.cut_times["cut_end"] + self.params["t_pull_apart"]
        if self.xfem:
            if cutting and self.frame % 5 == 0:
                self.max_blade_penetration = max(self.max_blade_penetration, self.blade_penetration())
            if self.sim_time <= self.cut_times["cut_end"]:
                self.max_blade_lead = max(self.max_blade_lead, self.blade_lead())
            self._track_xfem()
            return
        if cutting and self.frame % 5 == 0:
            self.max_blade_penetration = max(self.max_blade_penetration, self.blade_penetration())
            self.min_lip_gap = min(self.min_lip_gap, self.lip_gap())
        if self.frame % 10 == 0:
            self.max_penetration = max(self.max_penetration, self.needle_penetration())
            self.max_residual = max(self.max_residual, self.max_row_residual())
            self.tunnels_seen = max(self.tunnels_seen, int(self.solver._suture_tunnel_count.numpy()[0]))

    def _check_cut_xfem(self, *, blade_penetration_bound: float):
        """The criteria of the XFEM-type cut: the crossed tets are cut once, two parts, mass, no overflow, rows, blade."""
        x = self.solver._xfem
        report = self.solver.cut_report()
        nominal, nearest = self.crossed_tets()
        tet_side = x["tet_side"].numpy()
        cut_rows = np.flatnonzero(tet_side[: self.tissue_tet_count] == 1)
        if not np.array_equal(cut_rows, nominal):
            raise AssertionError(
                f"the blade plane crosses {len(nominal)} tets and {len(cut_rows)} are cut: "
                f"{len(np.setdiff1d(nominal, cut_rows))} crossed tets are not cut and "
                f"{len(np.setdiff1d(cut_rows, nominal))} cut tets are not crossed "
                f"(nearest vertex {nearest * 1e3:.3f} mm from the plane)"
            )
        slots = x["tet_slot"].numpy()[cut_rows]
        twins = x["tet_twin"].numpy()[cut_rows]
        fragments_b = int((tet_side == 2).sum())
        if (
            len(np.unique(slots)) != len(cut_rows)
            or len(np.unique(twins)) != len(cut_rows)
            or not (tet_side[twins] == 2).all()
            or fragments_b != len(cut_rows)
            or report["spare_tets_taken"] != len(cut_rows)
        ):
            raise AssertionError(
                f"a tet is not cut exactly once: {len(cut_rows)} cut tets, {len(np.unique(slots))} cut slots, "
                f"{fragments_b} fragment rows of side B, {report['spare_tets_taken']} spare tets taken"
            )
        parts = self.tissue_row_parts()
        if len(parts) != 2:
            raise AssertionError(f"expected two tissue parts after the cut, found tet and fragment counts {parts}")
        if sum(report["overflow"]) != 0:
            raise AssertionError(
                f"cut capacity overflow {report['overflow']} (pools, triangle rows, edge rows, spare tets, tet records)"
            )
        mass = float(self.model.particle_mass.numpy().sum())
        if abs(mass - self.particle_mass_0) > 1.0e-5 * self.particle_mass_0:
            raise AssertionError(f"particle mass changed from {self.particle_mass_0:.6g} to {mass:.6g} kg")
        rows = self.surface_row_errors()
        if rows["mixed"] or rows["virtual_corner"] or rows["degenerate"]:
            raise AssertionError(
                f"surface rows: {rows['mixed']} mix parked and live particles, {rows['virtual_corner']} name a "
                f"virtual corner, {rows['degenerate']} have an edge that acts as a point (shortest edge "
                f"{rows['shortest'] * 1e3:.3f} mm)"
            )
        if self.max_blade_penetration > blade_penetration_bound:
            raise AssertionError(
                f"blade edge penetrated tissue that it has contact with by {self.max_blade_penetration * 1e3:.3f} mm, "
                f"bound {blade_penetration_bound * 1e3:.2f} mm"
            )
        # With the blade plane in the middle of a cell the bound above cannot trip (the nearest particles with
        # blade contact are the front wound particles, half a start gap off the plane). This one can.
        if self.max_blade_lead > 0.0:
            raise AssertionError(
                f"the blade edge went {self.max_blade_lead * 1e3:.3f} mm past a tet that the blade plane crosses and "
                "that was not cut"
            )

    def check_cut(self, *, blade_penetration_bound: float = 0.5e-3):
        """The cut criteria: the faces of the cut plane are cut, two parts, no overflow, bounded blade penetration."""
        if self.xfem:
            self._check_cut_xfem(blade_penetration_bound=blade_penetration_bound)
            return
        p = self.params
        report = self.solver.cut_report()
        expected_faces = 2 * p["block_cells"][1] * p["block_cells"][2]
        if report["cut_faces"] != expected_faces:
            raise AssertionError(f"expected {expected_faces} cut faces, found {report['cut_faces']}")
        parts = self.tissue_parts()
        if parts != [self.tissue_tet_count // 2] * 2:
            raise AssertionError(f"expected two equal tissue parts after the cut, found tet counts {parts}")
        copies = sum(report["copies_per_pool"])
        if copies != len(self.plane_particles):
            raise AssertionError(f"expected {len(self.plane_particles)} copies, found {copies}")
        if sum(report["overflow"]) != 0 or report["unsafe"] != 0:
            raise AssertionError(f"cut capacity overflow {report['overflow']} or unsafe start gap {report['unsafe']}")
        mass = float(self.model.particle_mass.numpy().sum())
        if abs(mass - self.particle_mass_0) > 1.0e-5 * self.particle_mass_0:
            raise AssertionError(f"particle mass changed from {self.particle_mass_0:.6g} to {mass:.6g} kg")
        if self.min_lip_gap <= 0.0:
            raise AssertionError(
                f"a copy crossed its original during the cut (smallest gap {self.min_lip_gap * 1e3:.3f} mm)"
            )
        if self.max_blade_penetration > blade_penetration_bound:
            raise AssertionError(
                f"blade edge penetrated tissue outside the wound by {self.max_blade_penetration * 1e3:.3f} mm, "
                f"bound {blade_penetration_bound * 1e3:.2f} mm"
            )

    def _check_final(self, *, gap_bound: float, residual_bound: float, penetration_bound: float, hang_bound: float):
        if not hasattr(self, "_face_a"):
            self._init_faces()
        self.check_cut()
        if self._gap_open < 0.75 * self.params["gap"]:
            raise AssertionError(
                f"the grippers opened the wound to {self._gap_open * 1e3:.2f} mm only, expected about "
                f"{self.params['gap'] * 1e3:.1f} mm"
            )
        q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        if not (np.isfinite(q).all() and np.isfinite(body_q).all()):
            raise AssertionError("simulation produced non-finite state")
        # Every stitch leaves one through tunnel per block; a graze (two points, finished, no rows) does not count.
        expected = 2 * self.stitches
        tunnels = int(self.solver._suture_tunnel_count.numpy()[0])
        states = self.solver._suture_tunnel_state.numpy()[:tunnels]
        counts = self.solver._suture_tunnel_point_count.numpy()[:tunnels]
        through = int(sum(1 for s, c in zip(states, counts, strict=True) if int(s) == 2 and int(c) >= 3))
        if through != expected or not all(int(s) in (2, 3, 4) for s in states):
            raise AssertionError(
                f"expected {expected} through tunnels, found {through} among {tunnels} tunnels with states "
                f"{states.tolist()} and point counts {counts.tolist()}"
            )
        if self.max_penetration > penetration_bound:
            raise AssertionError(
                f"needle group penetrated ungated tissue by {self.max_penetration * 1e3:.3f} mm, "
                f"bound {penetration_bound * 1e3:.2f} mm"
            )
        gap = self.gap_before_release if self.released else self.gap_width()
        if abs(gap) > gap_bound:
            raise AssertionError(f"gap between the blocks is {gap * 1e3:.2f} mm, bound {gap_bound * 1e3:.2f} mm")
        if self.max_residual > residual_bound:
            raise AssertionError(
                f"largest tunnel point residual {self.max_residual * 1e3:.3f} mm exceeds {residual_bound * 1e3:.2f} mm"
            )
        if self.released:
            lowest = self.right_block_lowest()
            if lowest < self.params["floor_top"] + 0.02:
                raise AssertionError(f"the released block fell to {lowest * 1e3:.1f} mm; the stitches did not hold it")
            drift = self.edge_distance() - self._edge_distance_0
            if drift > hang_bound:
                raise AssertionError(
                    f"the released block's wound edge moved {drift * 1e3:.1f} mm away from the left edge, "
                    f"bound {hang_bound * 1e3:.1f} mm"
                )

    def _check_final_xfem(
        self,
        *,
        residual_rest_bound: float,
        residual_pressed_bound: float,
        residual_bound: float,
        open_tolerance: float,
        closed_tolerance: float,
        speed_bound: float,
        speed_spare_bound: float,
        speed_pressed_bound: float,
    ):
        """The criteria of the XFEM-type cut and of the wound that the grippers open, press closed and open again."""
        p = self.params
        self.check_cut()
        self._track_xfem()
        if not self.finite:
            raise AssertionError("simulation produced non-finite state")
        missing = [name for name in ("open", "pressed") if name not in self.wound_marks]
        if missing:
            raise AssertionError(
                f"the run ended at {self.sim_time:.2f} s, before the end of the pressed hold at "
                f"{self.xfem_times['reopen_start']:.2f} s; run {total_frames(self.stitches, p, 'xfem')} frames"
            )
        opened, pressed = self.wound_marks["open"], self.wound_marks["pressed"]
        # The open wound: the gripper travel along the blade plane normal plus the start gap.
        start_gap = self.solver._cut_start_gap
        open_width = p["gap"] * math.cos(math.radians(p["blade_yaw_deg"])) + start_gap
        if abs(opened["gap_mean"] - open_width) > open_tolerance * open_width:
            raise AssertionError(
                f"the grippers opened the wound to {opened['gap_mean'] * 1e3:.2f} mm, not within "
                f"{open_tolerance * 100.0:.0f} percent of {open_width * 1e3:.2f} mm (gripper travel plus start gap)"
            )
        # The press goes past the start gap, so without self contact the wound faces pass through each other
        # and every value below is negative.
        if p["press_past_closed"] <= start_gap:
            raise AssertionError(
                f"press_past_closed ({p['press_past_closed'] * 1e3:.2f} mm) is not past the start gap "
                f"({start_gap * 1e3:.2f} mm): the pass-through check cannot fail"
            )
        if min(pressed["face_distance"], pressed["mean_plane_distance"], pressed["gap_min"]) <= 0.0:
            raise AssertionError(
                "the wound faces passed through each other while pressed closed: smallest signed distance of a "
                f"wound particle from the other wound face {pressed['face_distance'] * 1e3:.3f} mm, from the mean "
                f"plane of the other side {pressed['mean_plane_distance'] * 1e3:.3f} mm, smallest gap of an edge "
                f"pair {pressed['gap_min'] * 1e3:.3f} mm"
            )
        # Self contact acts within the margin, so the pressed wound faces rest about one margin apart.
        margin = p["self_contact_margin"]
        if abs(pressed["gap_mean"] - margin) > closed_tolerance * margin:
            raise AssertionError(
                f"the pressed wound is {pressed['gap_mean'] * 1e3:.3f} mm wide, not within "
                f"{closed_tolerance * 100.0:.0f} percent of the self-contact margin ({margin * 1e3:.2f} mm)"
            )
        for label, value, bound in (
            ("at rest with the wound open", opened["tie_residual"], residual_rest_bound),
            ("with the wound pressed closed", pressed["tie_residual"], residual_pressed_bound),
            ("largest of the run", self.max_tie_residual, residual_bound),
        ):
            if value > bound:
                raise AssertionError(f"tie residual {label} is {value * 1e3:.4f} mm, bound {bound * 1e3:.3f} mm")
        for kind, value in self.max_speed.items():
            bound = speed_bound if kind == "tissue" else speed_spare_bound
            if value > bound:
                raise AssertionError(
                    f"largest speed of a {kind} particle is {value * 1e3:.1f} mm/s, bound {bound * 1e3:.0f} mm/s"
                )
        for kind, value in self.max_speed_pressed.items():
            if value > speed_pressed_bound:
                raise AssertionError(
                    f"the pressed wound is not calm: a {kind} particle moves at {value * 1e3:.2f} mm/s during the "
                    f"hold, bound {speed_pressed_bound * 1e3:.1f} mm/s"
                )
        sizes = (p["xfem_vertex_contact_buffer"], p["xfem_edge_contact_buffer"])
        if self.max_contact_list[0] > sizes[0] or self.max_contact_list[1] > sizes[1]:
            raise AssertionError(
                f"a self-contact list overflowed: {self.max_contact_list[0]} triangles at one vertex (size "
                f"{sizes[0]}), {self.max_contact_list[1]} edges at one edge row (size {sizes[1]})"
            )

    def test_final(self):
        """Robust invariants: two through tunnels per stitch, finite state, bounded penetration, wound closed to
        within half a tet edge before the release, and the released block hanging on the stitches. Run-to-run
        variation on the GPU (decisions I19, I21) keeps the design's tighter bounds out of this test; see
        :meth:`check_final_strict`.

        With the XFEM-type cut: the criteria of :meth:`_check_final_xfem`. The tie is a spring, so its residual
        is the force on the wound particle divided by the tie stiffness. The bounds come from the values measured
        with the blade at mid cell and turned by 8 degrees (the larger value is given):

        - Tie residual: 0.005 mm at rest with the wound open (measured 0.0008 mm), 0.2 mm with the wound pressed
          closed (0.086 mm) and 0.5 mm at any time (0.24 mm, on a front wound particle under the blade).
        - Open wound: within 5 percent of the gripper travel along the blade plane normal plus the start gap
          (1.6 percent below it with the blade turned, where some edges hold less than the start gap).
        - Pressed wound: mean width within 10 percent of the self-contact margin (5.3 percent below it).
        - Speeds: 0.5 m/s for a tissue particle and 1 m/s for a wound particle or a virtual corner at any time
          (0.12 m/s; a virtual corner of a thin fragment can be fast for a frame), and 2 mm/s for every
          particle while the wound is held closed (0.5 mm/s; with self-contact damping the wound shivers at
          20 to 90 mm/s).
        """
        if self.xfem:
            self._check_final_xfem(
                residual_rest_bound=0.005e-3,
                residual_pressed_bound=0.2e-3,
                residual_bound=0.5e-3,
                open_tolerance=0.05,
                closed_tolerance=0.1,
                speed_bound=0.5,
                speed_spare_bound=1.0,
                speed_pressed_bound=2.0e-3,
            )
            return
        self._check_final(gap_bound=1.5e-3, residual_bound=1.0e-3, penetration_bound=0.5e-3, hang_bound=10.0e-3)

    def check_final_strict(self):
        """The bounds of the design: gap below 0.5 mm, row residual below 0.3 mm, penetration below 0.1 mm, hanging
        edge within 5 mm of the left edge."""
        self._check_final(gap_bound=0.5e-3, residual_bound=0.3e-3, penetration_bound=1.0e-4, hang_bound=5.0e-3)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--stitches", type=int, default=PARAMS["stitches"], help="Number of stitches.")
        parser.add_argument(
            "--cut-arm",
            choices=("gate", "wound", "design"),
            default=PARAMS["cut_arm"],
            help="How the cut goes on below the first tet row (see PARAMS['cut_arm']).",
        )
        parser.add_argument(
            "--cut-method",
            choices=("faces", "xfem"),
            default="faces",
            help="faces: the cut follows tet faces, then the suture. xfem: the blade is off the mesh planes and "
            "the cut goes through the tets; the grippers open the wound, press it closed and open it again "
            f"(no suture; {total_frames(PARAMS['stitches'], cut_method='xfem')} frames).",
        )
        parser.add_argument(
            "--blade-yaw-deg",
            type=float,
            default=PARAMS["blade_yaw_deg"],
            help="Turn of the blade about the vertical axis [deg]; --cut-method xfem only.",
        )
        parser.set_defaults(num_frames=total_frames(PARAMS["stitches"]))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
