# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example VBD Suture

Two elastic cuboids hang in the air, each held by a gripper 12 mm from the
wound. A simulated half-circle needle with a thread sews them together with a
running suture of several stitches: a kinematic holder rotates the needle about
its arc center so the tip pierces the top of the left block, crosses the gap,
pierces the right block and exits through its top, then hops to the next stitch.
Every bite opens a virtual tunnel of tunnel points in the tetrahedral tissue; the
needle and the thread slide through the tunnels with capped friction. The free end
of the thread is clamped beside the far corner of the left block and its slack
hangs beside the blocks on the far side, out of the needle's way. After the last
stitch the right gripper opens its upper jaw, so the right block rests on the lower
jaw, and the holder pulls the needle away: the thread tightens and draws the right
block along the jaw until the wound edges meet. The lower jaw then withdraws and
the right block hangs from the stitches, which the hanging weight tensions.

Success criteria (test mode, ``test_final``): the state is finite, the needle opens
two through tunnels per stitch, the needle group penetrates ungated tissue by less
than 0.5 mm, the gap before the release is closed to within half a tet edge
(1.5 mm), and after the release the right block hangs on the stitches (it stays
above the floor and its wound edge stays within 10 mm of the left edge).
``check_final_strict`` holds the tighter bounds of the design (gap below 0.5 mm,
largest tunnel row residual below 0.3 mm, penetration below 0.1 mm, hanging edge
within 5 mm); run-to-run variation on the GPU moves these by several tenths of a
millimetre (see notes/vbd-suture/decisions.md I19 and I21), so they are reported,
not asserted.

Runs:
- `python -m newton.examples vbd_suture`
- `python -m newton.examples vbd_suture --stitches 1 --num-frames 886`
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import warp as wp

import newton
import newton.examples
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
    # True: the two blocks get disjoint particle color sets, so a vertex of one wound face is never updated in
    # the same Gauss-Seidel color pass as the face vertex it is in self-contact with.
    "separate_block_colors": True,
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


def total_frames(stitches: int, p: dict | None = None) -> int:
    p = PARAMS if p is None else p
    return int(math.ceil(schedule(stitches, p)["end"] * p["fps"])) + 1


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.stitches = int(getattr(args, "stitches", None) or PARAMS["stitches"])
        self.times = schedule(self.stitches)
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
        builder.color()
        if self.params["separate_block_colors"]:
            (a0, a1), (b0, b1) = self.block_particle_ranges
            groups = [np.asarray(g, dtype=int) for g in builder.particle_color_groups]
            part_a = [g[(g >= a0) & (g < a1)] for g in groups]
            part_b = [g[(g >= b0) & (g < b1)] for g in groups]
            builder.set_coloring([g for g in part_a + part_b if len(g) > 0])
        self.model = builder.finalize()

        p = self.params
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
        self.solver = SolverVBD(
            self.model,
            iterations=p["solver_iterations"],
            **schedule_kwargs,
            rigid_soft_enable_dat=True,
            particle_enable_self_contact=p["self_contact"],
            particle_self_contact_margin=p["self_contact_margin"],
            particle_self_contact_gap=p["self_contact_gap"],
            suture_smooth_tangent=p["suture_smooth_tangent"],
            collision_pipeline=self.collision_pipeline,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.max_penetration = 0.0
        self.max_residual = 0.0
        self.tunnels_seen = 0
        self.pull_progress = 0.0
        self.pull_force_base = None
        self.upper_open = False
        self.released = False
        self.gap_before_release = None
        self._init_faces()

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
        p = self.params
        nx, ny, nz = p["block_cells"]
        cell = p["cell"]
        E, nu = p["tissue_youngs_modulus"], p["tissue_poisson"]
        k_mu = E / (2.0 * (1.0 + nu))
        k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        width = nx * cell
        depth = ny * cell
        self.block_width = width
        self.block_particle_ranges = []
        for sign in (-1.0, 1.0):
            x0 = sign * (0.5 * p["gap"]) + (-width if sign < 0 else 0.0)
            start = builder.particle_count
            builder.add_soft_grid(
                pos=wp.vec3(x0, -0.5 * depth, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=nx,
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
            self.block_particle_ranges.append((start, builder.particle_count))
        self.tissue_height = nz * cell

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
        for block, (start, end) in enumerate(self.block_particle_ranges):
            q = np.array([[v[0], v[1], v[2]] for v in builder.particle_q[start:end]], dtype=np.float64)
            inward = 1.0 if block == 0 else -1.0  # from the facing face into the block
            face_x = q[:, 0].max() if block == 0 else q[:, 0].min()
            depth = (face_x - q[:, 0]) * inward
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
                    int(start + i), upper, body_point=wp.vec3(*(q[i] - upper_pos)), stiffness=math.inf
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
        t = self.sim_time
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

    def _set_kinematics(self, t: float):
        p = self.params
        body_q = self.state_0.body_q.numpy()
        body_q[self.holder] = self._holder_pose(t)
        if t >= self.times["open_upper"]:
            # the right upper jaw lifts and withdraws; the lower jaw carries the block until the release
            g = self.grippers[1]
            if not self.upper_open:
                self._open_upper()
            su = min((t - self.times["open_upper"]) / p["t_open_upper"], 1.0)
            upper = g["upper_pos"] + np.array([p["open_dx"] * su, 0.0, p["open_dz"] * su])
            lower = g["lower_pos"]
            if t >= self.times["release"]:
                if not self.released:
                    self._release()
                so = min((t - self.times["release"]) / p["t_open"], 1.0)
                lower = g["lower_pos"] + np.array([p["open_dx"] * so, 0.0, -p["open_dz"] * so])
            body_q[g["upper"]] = wp.transform(wp.vec3(*upper), wp.quat_identity())
            body_q[g["lower"]] = wp.transform(wp.vec3(*lower), wp.quat_identity())
        self.state_0.body_q.assign(body_q)

    # ── stepping ────────────────────────────────────────────────────────

    def simulate(self):
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

    def gap_width(self) -> float:
        """Mean separation of the facing block faces along x, centroid to centroid, from their initial face particles."""
        q = self.state_0.particle_q.numpy()
        a0, a1 = self.block_particle_ranges[0]
        b0, b1 = self.block_particle_ranges[1]
        return float(q[b0:b1, 0][self._face_b].mean() - q[a0:a1, 0][self._face_a].mean())

    def edge_distance(self) -> float:
        """Distance between the centroids of the two facing faces [m]."""
        q = self.state_0.particle_q.numpy()
        a0, a1 = self.block_particle_ranges[0]
        b0, b1 = self.block_particle_ranges[1]
        return float(np.linalg.norm(q[b0:b1][self._face_b].mean(axis=0) - q[a0:a1][self._face_a].mean(axis=0)))

    def right_block_lowest(self) -> float:
        """Lowest particle of the right block [m]; it must stay well above the floor after the release."""
        q = self.state_0.particle_q.numpy()
        b0, b1 = self.block_particle_ranges[1]
        return float(q[b0:b1, 2].min())

    def _init_faces(self):
        q = self.model.particle_q.numpy()
        a0, a1 = self.block_particle_ranges[0]
        b0, b1 = self.block_particle_ranges[1]
        xa = q[a0:a1, 0]
        xb = q[b0:b1, 0]
        self._face_a = xa > xa.max() - 1.0e-6
        self._face_b = xb < xb.min() + 1.0e-6
        self._edge_distance_0 = self.edge_distance()

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
        tissue = np.zeros(len(q), dtype=bool)
        for start, end in self.block_particle_ranges:
            tissue[start:end] = True
        free = tissue & (gate == 0)
        if not free.any():
            return 0.0
        pts = q[free]
        rad = radius[free]
        worst = 0.0
        for shape in np.flatnonzero(shape_gate != 0):
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
        if not hasattr(self, "_face_a"):
            self._init_faces()
        if self.frame % 10 == 0:
            self.max_penetration = max(self.max_penetration, self.needle_penetration())
            self.max_residual = max(self.max_residual, self.max_row_residual())
            self.tunnels_seen = max(self.tunnels_seen, int(self.solver._suture_tunnel_count.numpy()[0]))

    def _check_final(self, *, gap_bound: float, residual_bound: float, penetration_bound: float, hang_bound: float):
        if not hasattr(self, "_face_a"):
            self._init_faces()
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

    def test_final(self):
        """Robust invariants: two through tunnels per stitch, finite state, bounded penetration, wound closed to
        within half a tet edge before the release, and the released block hanging on the stitches. Run-to-run
        variation on the GPU (decisions I19, I21) keeps the design's tighter bounds out of this test; see
        :meth:`check_final_strict`."""
        self._check_final(gap_bound=1.5e-3, residual_bound=1.0e-3, penetration_bound=0.5e-3, hang_bound=10.0e-3)

    def check_final_strict(self):
        """The bounds of the design: gap below 0.5 mm, row residual below 0.3 mm, penetration below 0.1 mm, hanging
        edge within 5 mm of the left edge."""
        self._check_final(gap_bound=0.5e-3, residual_bound=0.3e-3, penetration_bound=1.0e-4, hang_bound=5.0e-3)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--stitches", type=int, default=PARAMS["stitches"], help="Number of stitches.")
        parser.set_defaults(num_frames=total_frames(PARAMS["stitches"]))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
