# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example KFC Bag Lift — Franka FR3 lifts a bag containing 3 rigid bodies
#
# A Franka Panda FR3 arm grips the top handles of a deformable KFC paper
# bag that is already sitting on the ground with three rigid food items
# inside, then lifts the whole bag to a raised hold height.
#
# Key differences from example_kfc_bag_drop:
#   - Bag starts on the ground (no drop from height).
#   - FR3 robot arm uses a contact-only pinch grasp to lift the bag.
#   - Dynamic objects (sphere, box, capsule) are carried inside the bag.
#   - Robot starts directly at the grasp pose so closing begins quickly.
#
# Physics:
#   - VBD cloth solver + AVBD rigid solver. The FR3 links are treated as
#     kinematic rigid bodies by zeroing their inverse mass/inertia.
#   - broad_phase='nxn' required for particle–body contact detection.
#   - COLLIDE_PARTICLES removed from non-gripper robot shapes to avoid
#     explosive contacts; the two finger links and food-item shapes keep
#     COLLIDE_PARTICLES so the bag can contact them.
#   - All physics in centimeter scale (gravity = -981 cm/s²).
#
# Command: python -m newton.examples bag.example_kfc_bag_lift
#
###########################################################################

from __future__ import annotations

import copy
import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton.examples.bag.mesh import (
    DEFAULT_PROXY_MODE as _DEFAULT_PROXY_MODE,
    add_proxy_mesh_arguments as _add_proxy_mesh_arguments,
    build_bary_map as _build_bary_map_common,
    decimate_mesh as _decimate_mesh_common,
    load_kfc_mesh_zup as _load_kfc_mesh_zup_common,
)
from newton.examples.bag.capture import (
    add_capture_arguments as _add_capture_arguments,
    capture_replay_frame as _capture_replay_frame_common,
    configure_capture as _configure_capture,
    finalize_capture as _finalize_capture,
    finalize_replay_video as _finalize_replay_video_common,
    get_viewer_frame as _get_viewer_frame_common,
    init_video_capture as _init_video_capture_common,
    write_video_frame as _write_video_frame_common,
)
from newton.examples.bag.render import render_bag_meshes as _render_bag_meshes

# ─────────────────────────────────────────────────────────────────────────────
# Simulation scale: centimeters
# ─────────────────────────────────────────────────────────────────────────────
_G_CM = -981.0
_VIZ_SCALE = 0.01  # cm → m for ViewerGL

# ── Bag geometry (cm) ────────────────────────────────────────────────────────
_BAG_H = 27.9        # full bag height
_BAG_X, _BAG_Y = 0.0, 0.0  # bag centre position on the ground

# ── Robot waypoints (cm) ─────────────────────────────────────────────────────
_GRAB_Z  = _BAG_H + 9.0   # grasp height above the handle area
_LIFT_Z  = 65.0           # final lift height above ground

# ── Cloth material (CGS) ─────────────────────────────────────────────────────
_TRI_KE  = 1.0e5
_TRI_KA  = 1.0e5
_TRI_KD  = 1.0e-5
_EDGE_KE = 1.0e4
_EDGE_KD = 0.1
_DENSITY = 0.008  # g/cm²

# ── Collision ────────────────────────────────────────────────────────────────
_PARTICLE_RADIUS = 0.80  # cm
_CONTACT_MARGIN  = 1.00  # cm

# ── Content objects ──────────────────────────────────────────────────────────
_OBJECT_MASS_G = 10.0    # lighter debug payload per internal rigid body [g]

# ── Finger contact tuning ────────────────────────────────────────────────────
_FINGER_SHAPE_MU = 4.0   # higher finger friction for contact-only pinch debugging
_FINGER_PAD_OFFSET_CM = (0.0, 0.758, 5.75)
_FINGER_PAD_HALF_THICKNESS_CM = 0.75
_FINGER_PAD_HALF_HEIGHT_CM = 2.60
_FINGER_PAD_TOP_BAND_CM = 1.0
_SOFT_CONTACT_MU = 2.0   # raise particle-side friction so finger pad mu matters more

# ── Contact stiffness (CGS) ──────────────────────────────────────────────────
# VBD averages particle-side (soft_contact_ke) with shape-side (shape_material_ke):
#   effective ke = 0.5 * (soft_contact_ke + shape_material_ke[shape])
# No drop-impact constraint here, so we can use high ke uniformly.
#   cloth-object: 0.5*(5e3 + 495e3) = 250 000  → stiff enough for 1 kg static load
#   cloth-ground: 0.5*(5e3 + 5e4)   =  27 500  → adequate static support
_CONTACT_KE      = 5.0e3   # soft_contact_ke (particle-side)
_OBJ_SHAPE_KE    = 4.95e5  # shape_material_ke for food-item rigid bodies
_GROUND_SHAPE_KE = 5.0e4   # shape_material_ke for ground plane

# ── Solver ───────────────────────────────────────────────────────────────────
_SIM_SUBSTEPS = 80
_VBD_ITERS    = 40

# ── Decimation ───────────────────────────────────────────────────────────────
_PHYSICS_TARGET_FACES = 1200


# ─────────────────────────────────────────────────────────────────────────────
# Warp kernels
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def _k_scale_pos(
    src: wp.array(dtype=wp.vec3),
    scale: float,
    dst: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def _k_scale_xform(
    src: wp.array(dtype=wp.transform),
    scale: float,
    dst: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    r = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, r)


@wp.kernel
def _k_bary_interp(
    phys_q: wp.array(dtype=wp.vec3),
    vi0: wp.array(dtype=wp.int32),
    vi1: wp.array(dtype=wp.int32),
    vi2: wp.array(dtype=wp.int32),
    bary: wp.array(dtype=wp.vec3),
    disp: wp.array(dtype=wp.vec3),
    scale: float,
    out: wp.array(dtype=wp.vec3),
):
    """Interpolate full-res vertex from physics mesh + precomputed displacement."""
    i = wp.tid()
    b = bary[i]
    p = phys_q[vi0[i]] * b[0] + phys_q[vi1[i]] * b[1] + phys_q[vi2[i]] * b[2]
    out[i] = (p + disp[i]) * scale


# ─────────────────────────────────────────────────────────────────────────────
# Mesh loading & barycentric mapping
# ─────────────────────────────────────────────────────────────────────────────


def _build_bary_map(full_verts, phys_verts, phys_faces):
    return _build_bary_map_common(full_verts, phys_verts, phys_faces)


def _load_kfc_mesh_zup():
    return _load_kfc_mesh_zup_common(_BAG_H)


def _decimate_mesh(verts, faces, target_faces, proxy_mode):
    return _decimate_mesh_common(verts, faces, target_faces, proxy_mode)


def _fit_bag_contents(phys_verts_cm):
    """Return overlap-free start positions for 3 content objects plus capsule quaternion.

    Sphere (R=4) and box (6×6×6) are placed side-by-side in X.
    Capsule (R=3, HL=2) lies horizontal along the bag Y-axis above both.
    All clearances to bag walls and between objects are ≥ 0.5 cm.
    """
    SPHERE_R = 4.0
    BOX_H    = 3.0
    CAP_R, CAP_HL = 3.0, 2.0

    bag_cx = 0.5 * (float(phys_verts_cm[:, 0].min()) + float(phys_verts_cm[:, 0].max()))
    bag_cy = 0.5 * (float(phys_verts_cm[:, 1].min()) + float(phys_verts_cm[:, 1].max()))

    def dist_sphere(ox, oy, oz, r):
        d = np.sqrt((phys_verts_cm[:, 0] - ox) ** 2
                    + (phys_verts_cm[:, 1] - oy) ** 2
                    + (phys_verts_cm[:, 2] - oz) ** 2)
        return float(d.min()) - r

    def dist_box(ox, oy, oz, hx, hy, hz):
        dx = np.maximum(np.abs(phys_verts_cm[:, 0] - ox) - hx, 0.0)
        dy = np.maximum(np.abs(phys_verts_cm[:, 1] - oy) - hy, 0.0)
        dz = np.maximum(np.abs(phys_verts_cm[:, 2] - oz) - hz, 0.0)
        return float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).min())

    def dist_cap_y(ox, oy, oz, r, hl):
        vy = np.clip(phys_verts_cm[:, 1], oy - hl, oy + hl)
        d = np.sqrt((phys_verts_cm[:, 0] - ox) ** 2
                    + (phys_verts_cm[:, 1] - vy) ** 2
                    + (phys_verts_cm[:, 2] - oz) ** 2)
        return float(d.min()) - r

    s_x, s_y, s_z = bag_cx + 2.5, bag_cy, 5.0
    b_x, b_y, b_z = bag_cx - 5.5, bag_cy, 4.0
    c_x, c_y, c_z = bag_cx,       bag_cy, 12.5

    cap_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)

    cs = dist_sphere(s_x, s_y, s_z, SPHERE_R)
    cb = dist_box(b_x, b_y, b_z, BOX_H, BOX_H, BOX_H)
    cc = dist_cap_y(c_x, c_y, c_z, CAP_R, CAP_HL)
    print(
        f"[KFC lift] Object placements (clearance to bag wall):\n"
        f"  sphere  @ ({s_x:.1f}, {s_y:.1f}, {s_z:.1f})  clr={cs:.2f} cm\n"
        f"  box     @ ({b_x:.1f}, {b_y:.1f}, {b_z:.1f})  clr={cb:.2f} cm\n"
        f"  capsule @ ({c_x:.1f}, {c_y:.1f}, {c_z:.1f}) [Y-horiz]  clr={cc:.2f} cm"
    )
    return (s_x, s_y, s_z), (b_x, b_y, b_z), (c_x, c_y, c_z), cap_quat


def _qv4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


# ─────────────────────────────────────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────────────────────────────────────


class Example:
    """Franka FR3 arm picks up a bag containing 3 dynamic rigid bodies.

    The bag starts on the ground with sphere, box, and capsule inside.
    The robot starts at the grasp pose, closes the fingers almost immediately,
    and then lifts the bag. All physics run in centimeters.
    """

    def __init__(
        self,
        viewer,
        save_mp4: str | None = None,
        test_mode: bool = False,
        capture_replay: bool = False,
        capture_frames: int = 300,
        capture_fps: int = 60,
        capture_dir: str = "outputs/replay_capture",
        capture_format: str = "mp4",
        target_faces: int = _PHYSICS_TARGET_FACES,
        mesh_proxy_mode: str = _DEFAULT_PROXY_MODE,
    ):
        self.viewer    = viewer
        self.test_mode = test_mode
        _configure_capture(
            self,
            save_mp4=save_mp4,
            capture_replay=capture_replay,
            capture_frames=capture_frames,
            capture_fps=capture_fps,
            capture_dir=capture_dir,
            capture_format=capture_format,
            capture_background_writes=False,
        )

        self.fps          = 60
        self.frame_dt     = 1.0 / self.fps
        self.sim_time     = 0.0
        self.sim_substeps = _SIM_SUBSTEPS
        self.sim_dt       = self.frame_dt / self.sim_substeps
        self._frame_count = 0
        self._max_bag_top_z_cm = 0.0

        # Gripper command state: 0=open, 1=closed.
        self._gripper_frac = 0.0
        # Saved content body transforms (numpy, shape=(3,7)) from previous VBD step.
        # eval_fk resets body_q for content bodies from their (never-updated) FREE
        # joint_q each frame.  We save after VBD and restore before the substep loop.
        self._content_bq_save: np.ndarray | None = None

        # ── Load meshes ────────────────────────────────────────────────────
        full_verts_cm, full_faces = _load_kfc_mesh_zup()
        phys_verts_cm, phys_faces = _decimate_mesh(
            full_verts_cm,
            full_faces,
            target_faces,
            mesh_proxy_mode,
        )

        print(f"[KFC lift] Full mesh: {len(full_verts_cm)} verts, {len(full_faces)} tris")
        print(f"[KFC lift] Physics mesh: {len(phys_verts_cm)} verts, {len(phys_faces)} tris")

        # ── Barycentric map: full-res → physics mesh ───────────────────────
        print("[KFC lift] Building barycentric map...", end=" ", flush=True)
        self._bary_vi0_np, self._bary_vi1_np, self._bary_vi2_np, bary_w = \
            _build_bary_map(full_verts_cm, phys_verts_cm, phys_faces)
        self._bary_w       = wp.array(bary_w, dtype=wp.vec3)
        self._n_full_verts = len(full_verts_cm)
        self._full_indices_wp = wp.array(
            full_faces.flatten().astype(np.int32), dtype=wp.int32)
        self._viz_full_q = wp.zeros(self._n_full_verts, dtype=wp.vec3)

        bary_proj = (
            phys_verts_cm[self._bary_vi0_np] * bary_w[:, 0:1]
            + phys_verts_cm[self._bary_vi1_np] * bary_w[:, 1:2]
            + phys_verts_cm[self._bary_vi2_np] * bary_w[:, 2:3]
        )
        self._bary_disp = wp.array(
            (full_verts_cm - bary_proj).astype(np.float32), dtype=wp.vec3)
        print("done.")

        # ── Build scene ────────────────────────────────────────────────────
        builder = newton.ModelBuilder(gravity=_G_CM)

        # ── FR3 robot arm (metres → cm via scale=100) ─────────────────────
        # Base at (-50, 0, 5) so the arm can reach the bag at origin.
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, 0.0, 5.0), wp.quat_identity()),
            floating=False,
            scale=100,
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )
        # Seed pose for the startup IK solve; the robot is moved to the first
        # grasp waypoint after initialization.
        init_q = [-3.6802e-03, 2.3902e-02, 3.6804e-03,
                  -2.3683,    -1.2919e-04, 2.3922, 7.8549e-01]
        builder.joint_q[:9] = [*init_q, 4.0, 4.0]

        # ── Identify finger / hand bodies for selective mesh approximation ──
        self._left_finger_body  = next(
            i for i, l in enumerate(builder.body_label)
            if l.endswith("fr3_leftfinger"))
        self._right_finger_body = next(
            i for i, l in enumerate(builder.body_label)
            if l.endswith("fr3_rightfinger"))
        _hand_body = next(
            i for i, l in enumerate(builder.body_label)
            if l.endswith("fr3_hand"))
        _finger_body_set = {self._left_finger_body, self._right_finger_body, _hand_body}
        _gripper_contact_body_set = {self._left_finger_body, self._right_finger_body}

        # Add fingertip pads that span the bag's top edge in the horizontal
        # direction while keeping the pinch thickness unchanged.
        pad_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.001,
            mu=_FINGER_SHAPE_MU,
            ke=_OBJ_SHAPE_KE,
            kd=_OBJ_SHAPE_KE * 1e-4,
        )
        top_band_mask = phys_verts_cm[:, 2] >= (_BAG_H - _FINGER_PAD_TOP_BAND_CM)
        top_band_verts = phys_verts_cm[top_band_mask] if np.any(top_band_mask) else phys_verts_cm
        bag_top_edge_span_x_cm = float(top_band_verts[:, 0].max() - top_band_verts[:, 0].min())
        pad_xform = wp.transform(wp.vec3(*_FINGER_PAD_OFFSET_CM), wp.quat_identity())
        pad_hx = 0.5 * bag_top_edge_span_x_cm
        pad_hy = _FINGER_PAD_HALF_THICKNESS_CM
        pad_hz = _FINGER_PAD_HALF_HEIGHT_CM
        self._bag_top_edge_span_x_cm = bag_top_edge_span_x_cm
        self._finger_pad_half_extents_cm = (pad_hx, pad_hy, pad_hz)
        builder.add_shape_box(
            body=self._left_finger_body,
            xform=pad_xform,
            hx=pad_hx,
            hy=pad_hy,
            hz=pad_hz,
            cfg=pad_cfg,
            label="left_finger_pad",
        )
        builder.add_shape_box(
            body=self._right_finger_body,
            xform=pad_xform,
            hx=pad_hx,
            hy=pad_hy,
            hz=pad_hz,
            cfg=pad_cfg,
            label="right_finger_pad",
        )

        # ── Approximate non-finger shapes as convex hulls (like panda_hydro) ──
        # keep_visual_shapes=True preserves the original detailed mesh for rendering
        # while using the convex hull for physics — gives better robot visuals.
        _non_finger_shape_indices = [
            s for s, b in enumerate(builder.shape_body) if b not in _finger_body_set
        ]
        builder.approximate_meshes(
            method="convex_hull",
            shape_indices=_non_finger_shape_indices,
            keep_visual_shapes=True,
        )

        # ── IK model: robot only (no bag / objects) ───────────────────────
        self._model_single    = copy.deepcopy(builder).finalize()
        self._robot_body_count = builder.body_count

        # ── 3 dynamic content bodies ───────────────────────────────────────
        # Bag starts on the ground so no drop-height offset needed.
        (s_x, s_y, s_z), (b_x, b_y, b_z), (c_x, c_y, c_z), cap_quat = \
            _fit_bag_contents(phys_verts_cm)

        content_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.001, mu=0.5, ke=_OBJ_SHAPE_KE, kd=_OBJ_SHAPE_KE * 1e-4)

        builder.add_body(
            xform=wp.transform(wp.vec3(s_x, s_y, s_z), wp.quat_identity()),
            mass=0.1,
        )
        builder.add_shape_sphere(
            body=self._robot_body_count, radius=4.0, cfg=content_cfg)

        builder.add_body(
            xform=wp.transform(wp.vec3(b_x, b_y, b_z), wp.quat_identity()),
            mass=0.1,
        )
        builder.add_shape_box(
            body=self._robot_body_count + 1, hx=3.0, hy=3.0, hz=3.0, cfg=content_cfg)

        builder.add_body(
            xform=wp.transform(wp.vec3(c_x, c_y, c_z), cap_quat),
            mass=0.1,
        )
        builder.add_shape_capsule(
            body=self._robot_body_count + 2, radius=3.0, half_height=2.0, cfg=content_cfg)

        # ── Bag cloth mesh on the ground ───────────────────────────────────
        self._bag_particle_start = builder.particle_count
        bag_verts_wp = [
            wp.vec3(float(v[0]), float(v[1]), float(v[2]))
            for v in phys_verts_cm
        ]
        builder.add_cloth_mesh(
            pos=wp.vec3(_BAG_X, _BAG_Y, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=bag_verts_wp,
            indices=phys_faces.flatten().tolist(),
            density=_DENSITY,
            tri_ke=_TRI_KE,
            tri_ka=_TRI_KA,
            tri_kd=_TRI_KD,
            edge_ke=_EDGE_KE,
            edge_kd=_EDGE_KD,
            particle_radius=_PARTICLE_RADIUS,
        )
        self._bag_particle_end = builder.particle_count

        # Barycentric indices offset by particle_start
        ps = self._bag_particle_start
        self._bary_vi0 = wp.array(self._bary_vi0_np + ps, dtype=wp.int32)
        self._bary_vi1 = wp.array(self._bary_vi1_np + ps, dtype=wp.int32)
        self._bary_vi2 = wp.array(self._bary_vi2_np + ps, dtype=wp.int32)
        del self._bary_vi0_np, self._bary_vi1_np, self._bary_vi2_np

        self._proxy_indices_wp = wp.array(
            (phys_faces.flatten() + ps).astype(np.int32), dtype=wp.int32)

        # Ground plane
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_GROUND_SHAPE_KE, kd=_GROUND_SHAPE_KE * 1e-5, mu=0.4)
        builder.add_ground_plane(cfg=ground_cfg)

        builder.color(include_bending=True)

        # ── Finalize ──────────────────────────────────────────────────────
        self.model = builder.finalize()

        # soft_contact_ke is particle-side; VBD averages with shape_material_ke.
        self.model.soft_contact_ke = _CONTACT_KE
        self.model.soft_contact_kd = _CONTACT_KE * 1e-4
        self.model.soft_contact_mu = _SOFT_CONTACT_MU

        # Shape materials: per-shape ke so effective contact stiffness is:
        #   cloth–object: 0.5*(5e3 + 495e3) = 250 000  → supports contact-only debug payload
        #   cloth–ground: 0.5*(5e3 + 5e4)   =  27 500  → adequate support
        n_shapes = self.model.shape_material_ke.numpy().shape[0]
        ke_arr = self.model.shape_material_ke.numpy().copy()
        kd_arr = self.model.shape_material_kd.numpy().copy()
        mu_arr = self.model.shape_material_mu.numpy().copy()
        shape_body_np = self.model.shape_body.numpy()
        for s in range(n_shapes - 1):   # all shapes except last (ground)
            ke_arr[s] = _OBJ_SHAPE_KE
            kd_arr[s] = _OBJ_SHAPE_KE * 1e-4
            if int(shape_body_np[s]) in _gripper_contact_body_set:
                mu_arr[s] = _FINGER_SHAPE_MU
        ke_arr[n_shapes - 1] = _GROUND_SHAPE_KE
        kd_arr[n_shapes - 1] = _GROUND_SHAPE_KE * 1e-5
        self.model.shape_material_ke = wp.array(ke_arr, dtype=float)
        self.model.shape_material_kd = wp.array(kd_arr, dtype=float)
        self.model.shape_material_mu = wp.array(mu_arr, dtype=float)

        # Robot bodies are kinematic (driven by IK+FK, not VBD/AVBD)
        inv_m = self.model.body_inv_mass.numpy().copy()
        inv_i = self.model.body_inv_inertia.numpy().copy()
        inv_m[:self._robot_body_count] = 0.0
        inv_i[:self._robot_body_count] = 0.0

        # Content bodies: set mass to _OBJECT_MASS_G and compute inertia from geometry
        # Shapes: sphere (r=4), box (half=3), capsule (r=3, h=2)
        _content_radii   = [4.0, 3.0, 3.0]
        _content_ccoeff  = [0.4, 2.0 / 3.0, 0.5]   # I = c*m*r^2 approximation
        for k in range(3):
            b = self._robot_body_count + k
            inv_m[b] = 1.0 / _OBJECT_MASS_G
            r = _content_radii[k]
            c = _content_ccoeff[k]
            inv_i[b] = 1.0 / (c * _OBJECT_MASS_G * r * r) * np.eye(3)

        self.model.body_inv_mass     = wp.array(inv_m, dtype=float)
        self.model.body_inv_inertia  = wp.array(inv_i, dtype=wp.mat33)

        # Keep particle collision only on the gripper finger links. Other robot
        # links stay off the particle-contact path to avoid excessive cloth-arm
        # contacts away from the grasp region.
        shape_body  = self.model.shape_body.numpy()
        shape_flags = self.model.shape_flags.numpy().copy()
        n_removed = 0
        n_kept = 0
        for s in range(len(shape_flags)):
            b = int(shape_body[s])
            if 0 <= b < self._robot_body_count and b not in _gripper_contact_body_set:
                shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
                n_removed += 1
            elif b in _gripper_contact_body_set:
                n_kept += 1
        self.model.shape_flags = wp.array(
            shape_flags,
            dtype=self.model.shape_flags.dtype,
            device=self.model.device,
        )
        print(
            f"[KFC lift] Removed COLLIDE_PARTICLES from {n_removed} non-finger robot shapes; "
            f"kept {n_kept} finger shapes active for cloth contact"
        )

        # ── Sync model.body_q from initial FK before creating SolverVBD ──────
        # VBD initialises body_q_prev = model.body_q.  If model.body_q hasn't
        # been updated by eval_fk it may not match the actual initial body poses,
        # producing spurious initial velocities for the content bodies.
        _init_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, _init_state)
        wp.copy(self.model.body_q, _init_state.body_q)

        # ── VBD solver ─────────────────────────────────────────────────────
        # integrate_with_external_rigid_solver=False so VBD integrates the 3 content
        # bodies dynamically (sphere, box, capsule physically interact with cloth).
        # The FR3 bodies stay kinematic and the bag is lifted through finger-cloth
        # contact only; no particle pinning or rigid bag advection is applied.
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=_VBD_ITERS,
            integrate_with_external_rigid_solver=False,
            # Enable bag self-contact; finger-vs-cloth still goes through the
            # particle-shape soft-contact path on the active finger shapes.
            particle_enable_self_contact=True,
            particle_self_contact_radius=_PARTICLE_RADIUS,
            particle_self_contact_margin=_CONTACT_MARGIN,
            particle_enable_tile_solve=False,
            rigid_contact_k_start=_OBJ_SHAPE_KE,
        )

        # broad_phase='nxn' required for particle–body contact detection
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        # ── States ─────────────────────────────────────────────────────────
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)

        # ── IK ─────────────────────────────────────────────────────────────
        self._state_single = self._model_single.state()
        newton.eval_fk(
            self._model_single,
            self._model_single.joint_q,
            self._model_single.joint_qd,
            self._state_single,
        )
        self._ee_body_index = 10  # hand link in FR3
        self._setup_ik()
        self._initialize_robot_pregrasp()

        # ── Viewer: scale shape data to metres before set_model() ──────────
        self.viz_state = self.model.state()
        self._sim_shape_transform = self.model.shape_transform
        self._sim_shape_scale     = self.model.shape_scale

        xf_np = self.model.shape_transform.numpy().copy()
        xf_np[:, :3] *= _VIZ_SCALE
        self._viz_shape_transform = wp.array(
            xf_np, dtype=wp.transform, device=self.model.device)

        sc_np = self.model.shape_scale.numpy().copy()
        sc_np *= _VIZ_SCALE
        self._viz_shape_scale = wp.array(
            sc_np, dtype=wp.vec3, device=self.model.device)

        self.model.shape_transform = self._viz_shape_transform
        self.model.shape_scale     = self._viz_shape_scale
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.model.shape_transform = self._sim_shape_transform
        self.model.shape_scale     = self._sim_shape_scale

        if hasattr(self.viewer, "renderer"):
            # Camera: side-on view slightly above ground, back from the bag
            self.viewer.set_camera(
                pos=wp.vec3(1.0, -1.0, 0.8),
                pitch=-10.0,
                yaw=135.0,
            )

        if self.save_mp4:
            self._init_video_capture()

    # ─────────────────────────────────────────────────────────────────────
    # IK setup
    # ─────────────────────────────────────────────────────────────────────

    def _setup_ik(self):
        ee_idx = self._ee_body_index
        ee_tf  = wp.transform(*self._state_single.body_q.numpy()[ee_idx])

        self._pos_obj = ik.IKObjectivePosition(
            link_index=ee_idx,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.array(
                [wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=ee_idx,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(
                [_qv4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
        )
        self._obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self._model_single.joint_limit_lower,
            joint_limit_upper=self._model_single.joint_limit_upper,
        )
        self._joint_q_ik = wp.array(
            self._model_single.joint_q,
            shape=(1, self._model_single.joint_coord_count),
        )
        self._ik_solver = ik.IKSolver(
            model=self._model_single,
            n_problems=1,
            objectives=[self._pos_obj, self._rot_obj, self._obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # Waypoints: (target_pos_cm, duration_s, gripper_frac)
        #
        # Startup is intentionally short for fast iteration:
        #   - wp0 close:   cur_frac=0, nxt_frac=1 at GRAB_Z  → fingers start closing immediately
        #   - wp1 hold:    both cur_frac=1 AND nxt_frac=1; cur_pos=nxt_pos=GRAB_Z  → brief pinch settle
        #   - wp2 lift:    cur_pos=GRAB_Z, nxt_pos=LIFT_Z  → robot rises while staying closed
        bx, by = _BAG_X, _BAG_Y
        self.waypoints = [
            (wp.vec3(bx, by, _GRAB_Z),  0.45, 0.0),  # 0: close while stationary at the grasp pose
            (wp.vec3(bx, by, _GRAB_Z),  0.20, 1.0),  # 1: hold at grab — brief pinch settle
            (wp.vec3(bx, by, _GRAB_Z),  3.0,  1.0),  # 2: lift — interpolates to nxt pos=LIFT_Z
            (wp.vec3(bx, by, _LIFT_Z),  2.5,  1.0),  # 3: hold at lift height (end)
        ]
        self._current_waypoint = 0
        self._time_in_waypoint = 0.0

        # Warm up IK graph
        with wp.ScopedCapture() as cap:
            self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=24)
        self._graph_ik = cap.graph

    def _initialize_robot_pregrasp(self):
        """Place the robot at the first grasp waypoint before simulation starts."""
        start_pos = self.waypoints[0][0]
        start_frac = float(self.waypoints[0][2])
        start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)

        self._pos_obj.set_target_positions(wp.array([start_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_qv4(start_rot)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=48)

        jq = self.state_0.joint_q.numpy().copy()
        ik_sol = self._joint_q_ik.numpy()[0]
        jq[:7] = ik_sol[:7]
        gv = 4.0 * (1.0 - start_frac)
        jq[7], jq[8] = gv, gv
        jq_wp = wp.array(jq, dtype=float)

        self._gripper_frac = start_frac
        self.model.joint_q.assign(jq_wp)
        self.state_0.joint_q.assign(jq_wp)
        self.state_1.joint_q.assign(jq_wp)
        self.model.joint_qd.zero_()
        self.state_0.joint_qd.zero_()
        self.state_1.joint_qd.zero_()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.model.body_q, self.state_0.body_q)
        if getattr(self.solver, "body_q_prev", None) is not None:
            wp.copy(self.solver.body_q_prev, self.state_0.body_q)

    # ─────────────────────────────────────────────────────────────────────
    # Waypoint / grip management
    # ─────────────────────────────────────────────────────────────────────

    def _set_joint_targets(self):
        self._time_in_waypoint += self.frame_dt
        cur = self.waypoints[self._current_waypoint]
        nxt = self.waypoints[min(self._current_waypoint + 1, len(self.waypoints) - 1)]
        t   = min(self._time_in_waypoint / cur[1], 1.0)

        target_pos       = cur[0] * (1.0 - t) + nxt[0] * t
        gripper_frac_new = float(cur[2]) * (1.0 - t) + float(nxt[2]) * t

        r = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi)
        self._pos_obj.set_target_positions(wp.array([target_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_qv4(r)], dtype=wp.vec4))

        if self._graph_ik:
            wp.capture_launch(self._graph_ik)
        else:
            self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=24)

        self._gripper_frac = gripper_frac_new

        # Advance waypoint (stop at last)
        if (self._time_in_waypoint >= cur[1]
                and self._current_waypoint < len(self.waypoints) - 1):
            self._current_waypoint += 1
            self._time_in_waypoint  = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────

    def simulate(self):
        # Update robot FK from IK solution.
        # With integrate_with_external_rigid_solver=False, VBD integrates content
        # body dynamics itself.  The robot bodies (inv_mass=0) are kinematic: we
        # drive them via joint_q → eval_fk each frame.
        #
        # CRITICAL: eval_fk resets content body positions from their initial FREE
        # joint coordinates (which we never update).  Save the VBD-integrated
        # positions beforehand and restore them after eval_fk so VBD continues
        # from the correct state.
        jq = self.state_0.joint_q.numpy().copy()
        ik_sol = self._joint_q_ik.numpy()[0]
        jq[:7] = ik_sol[:7]
        gv = 4.0 * (1.0 - self._gripper_frac)
        jq[7], jq[8] = gv, gv
        self.state_0.joint_q.assign(wp.array(jq, dtype=float))
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Restore VBD-integrated content body positions after eval_fk reset them
        if self._content_bq_save is not None:
            bq_np = self.state_0.body_q.numpy().copy()
            rb = self._robot_body_count
            bq_np[rb:rb + 3] = self._content_bq_save
            self.state_0.body_q.assign(wp.array(bq_np, dtype=wp.transform))

        for _ in range(self.sim_substeps):
            # Copy kinematic robot body poses into state_1 so VBD starts each
            # substep from the correct robot configuration.  Content body poses
            # in state_1 are ignored here — VBD integrates them itself.
            wp.copy(self.state_1.body_q, self.state_0.body_q)

            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control,
                self.contacts, self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Save VBD-integrated content body positions for next frame's restore
        rb = self._robot_body_count
        self._content_bq_save = self.state_0.body_q.numpy()[rb:rb + 3].copy()

    # ─────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────

    def step(self):
        if self.capture_done:
            return

        self._frame_count += 1
        # Always run waypoint tracking — waypoint 0 starts the close-at-grasp phase.
        self._set_joint_targets()
        self.simulate()
        self.sim_time += self.frame_dt

        if self.test_mode:
            pq   = self.state_0.particle_q.numpy()
            topz = pq[self._bag_particle_start:self._bag_particle_end, 2].max()
            self._max_bag_top_z_cm = max(self._max_bag_top_z_cm, topz)

    def _update_render_buffers(self):
        """Refresh cached render buffers for the current simulation frame."""
        wp.launch(
            _k_bary_interp,
            dim=self._n_full_verts,
            inputs=[
                self.state_0.particle_q,
                self._bary_vi0, self._bary_vi1, self._bary_vi2,
                self._bary_w, self._bary_disp, _VIZ_SCALE,
            ],
            outputs=[self._viz_full_q],
        )

        wp.launch(
            _k_scale_pos,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, _VIZ_SCALE],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                _k_scale_xform,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, _VIZ_SCALE],
                outputs=[self.viz_state.body_q],
            )

    def render(self):
        if self.capture_done:
            return

        self._update_render_buffers()

        self.model.shape_transform = self._viz_shape_transform
        self.model.shape_scale     = self._viz_shape_scale

        _render_bag_meshes(
            self.viewer,
            sim_time=self.sim_time,
            viz_state=self.viz_state,
            full_positions=self._viz_full_q,
            full_indices=self._full_indices_wp,
            proxy_positions=self.viz_state.particle_q,
            proxy_indices=self._proxy_indices_wp,
        )

        self.model.shape_transform = self._sim_shape_transform
        self.model.shape_scale     = self._sim_shape_scale

        _write_video_frame_common(self)
        self._capture_replay_frame()

    def test_final(self):
        pq   = self.state_0.particle_q.numpy()
        topz = float(pq[self._bag_particle_start:self._bag_particle_end, 2].max())
        assert topz > 30.0, (
            f"Bag was not lifted: top z = {topz:.1f} cm (expected > 30 cm)"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Video capture
    # ─────────────────────────────────────────────────────────────────────

    def _init_video_capture(self):
        _init_video_capture_common(self)

    def _get_viewer_frame(self, *, render_ui: bool = False):
        return _get_viewer_frame_common(self.viewer, render_ui=render_ui)

    def _capture_replay_frame(self, *, frame_key: int | None = None):
        if frame_key is None:
            frame_key = self._frame_count
        _capture_replay_frame_common(self, frame_key=frame_key)

    def _finalize_replay_video(self):
        _finalize_replay_video_common(self)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=560)   # close immediately, then pinch + lift + final hold
    _add_proxy_mesh_arguments(parser)
    _add_capture_arguments(
        parser,
        replay_help="Capture rendered frames and auto-build replay video",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        save_mp4=getattr(args, "save_mp4", None),
        test_mode=getattr(args, "test", False),
        capture_replay=bool(args.capture_replay),
        capture_frames=int(args.capture_frames),
        capture_fps=int(args.capture_fps),
        capture_dir=str(args.capture_dir),
        capture_format=str(args.capture_format),
        target_faces=int(args.target_faces),
        mesh_proxy_mode=str(args.proxy_mode),
    )
    while viewer.is_running() and not getattr(example, "capture_done", False):
        if example._frame_count >= args.num_frames:
            break
        example.step()
        example.render()

    if args.test:
        example.test_final()

    _finalize_capture(example)
