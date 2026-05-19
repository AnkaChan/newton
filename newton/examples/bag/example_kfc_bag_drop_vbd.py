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
# Example KFC Bag Drop — Baseline VBD cloth bag with 3 rigid bodies
#
# A deformable KFC paper bag containing 3 rigid body "food items"
# drops from a short height and settles on the ground.
#
# - Graphics mesh: full-resolution kfc.usd (24K verts)
# - Physics/cloth mesh: solver-side proxy from shared bag mesh generation
# - Collision proxy: same solver mesh (particle radius provides margin)
# - 3 dynamic rigid bodies inside the bag
# - VBD solver, centimeter scale (gravity = -981 cm/s²)
#
# Command: python -m newton.examples.bag.example_kfc_bag_drop_vbd
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.bag.mesh import (
    DEFAULT_PROXY_MODE as _DEFAULT_PROXY_MODE,
    add_proxy_mesh_arguments as _add_proxy_mesh_arguments,
    build_bary_map_with_logging as _build_bary_map_with_logging,
    decimate_mesh as _decimate_mesh_common,
    load_kfc_mesh_zup as _load_kfc_mesh_zup_common,
    log_mesh_counts as _log_mesh_counts,
)
from newton.examples.bag.lift import log_content_placements_cm as _log_content_placements_cm
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
_LOG_PREFIX = "[KFC drop]"
_G_CM = -981.0  # cm/s²
_VIZ_SCALE = 0.01  # cm → m for ViewerGL

# ── Bag geometry (cm) ────────────────────────────────────────────────────────
_BAG_H = 27.9  # full height
_DROP_HEIGHT = 50.0  # cm above ground — bag hangs here, then drops
_HANG_FRAMES = 60   # frames to hold bag before release (1 second)

# ── Cloth material (CGS: cm, g, s) ──────────────────────────────────────────
_TRI_KE = 1.0e5  # stretch / shear stiffness
_TRI_KA = 1.0e5  # area-preservation stiffness
_TRI_KD = 1.0e-5  # stretch damping
_EDGE_KE = 1.0e4  # bending stiffness
_EDGE_KD = 0.1  # bending damping
_DENSITY = 0.008  # g/cm²  (~80 g/m² kraft paper)

# Softer cloth tuning for a more compliant bag, giving visibly larger sag
# under the heavy content objects while remaining stable at ground contact.
#
# Stability constraint: the VBD implicit GS solve propagates cloth deceleration
# upward at rate ≈ ke/(ke + m/dt²) per vertex per substep.  With 80 substeps
# per frame and m/dt²≈7.4e4 g/s², any tri_ke < ~2.5e4 causes the deceleration
# "cascade" to take > 80 substeps, so the fast-falling upper cloth slams into
# the stopped floor, causing explosion.  tri_ke=3e4 gives cascade time ≈69ss —
# fits within one frame with margin, and gives ~3× more sag than the hard bag.
_SOFT_TRI_KE = 3.0e4
_SOFT_TRI_KA = 3.0e4
_SOFT_TRI_KD = 1.7e-5
_SOFT_EDGE_KE = 5.0e2
_SOFT_EDGE_KD = 0.13
_SOFT_OBJ_SHAPE_KE = 7.5e4
# Two ground planes separate cloth and rigid-body stiffness requirements:
#   • Particle-only (has_shape_collision=False):
#       avg_ke = 0.5*(5e3 + 2e4) = 12500  — enough to support bag weight without
#       excessive floor penetration; c = 0.25×12500/5.2e-4 ≈ 6e6 >> m/dt²=7.4e4
#       so ground contact is strongly anchored (no runaway bounce).
#   • Rigid-only (has_particle_collision=False):
#       avg_ke = 0.5*(7.5e4 + 4.62e10) ≈ 2.31e10 stops rigid bodies quickly.
_SOFT_PARTICLE_GROUND_KE = 2.0e4    # particle-only ground ke for soft bag
_SOFT_PARTICLE_GROUND_KD = 0.5      # shape kd → avg_kd=0.25, prevents cloth bounce
_SOFT_RIGID_GROUND_KE    = 4.62e10  # rigid-only ground ke for soft bag
_SOFT_RIGID_GROUND_KD    = 0.5      # shape kd → avg_kd=0.25 for rigid body
_SOFT_RIGID_K_START      = 5.0e10   # AVBD k_start ≥ avg_ke (starts at cap)

# ── Collision ────────────────────────────────────────────────────────────────
_PARTICLE_RADIUS = 0.80  # cm
_CONTACT_MARGIN = 1.00  # cm

# ── Content objects ──────────────────────────────────────────────────────────
_OBJECT_MASS_G = 1000.0  # mass of each of the 3 food-item rigid bodies [g]

# ── Contact stiffness (CGS) ──────────────────────────────────────────────────
# VBD averages particle-side (soft_contact_ke) with shape-side (shape_material_ke):
#   effective ke = 0.5 * (soft_contact_ke + shape_material_ke[shape])
# Strategy: keep soft_contact_ke LOW for stable ground impact at 50 cm (≈313 cm/s),
# then compensate with high shape_material_ke on the content shapes so that
# cloth-object contact remains stiff enough to support 1 kg objects.
#   cloth-ground: 0.5*(5e3 + 1e3) = 3 000   → absorbs 50 cm impact
#   cloth-object: 0.5*(5e3 + 495e3) = 250 000 → supports 1 kg static load
_CONTACT_KE       = 5.0e3   # soft_contact_ke (particle-side)
_OBJ_SHAPE_KE     = 4.95e5  # shape_material_ke for food-item rigid bodies
_GROUND_SHAPE_KE  = 1.0e3   # shape_material_ke for ground plane

# ── Solver ───────────────────────────────────────────────────────────────────
_SIM_SUBSTEPS = 80
_VBD_ITERS = 40

# ── Decimation target ────────────────────────────────────────────────────────
_PHYSICS_TARGET_FACES = 1200


# ─────────────────────────────────────────────────────────────────────────────
# Warp kernels
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def _k_set_float_scalar(
    arr: wp.array(dtype=float),
    idx: wp.array(dtype=wp.int32),
    val: float,
):
    i = wp.tid()
    arr[idx[i]] = val


@wp.kernel
def _k_set_float_array(
    arr: wp.array(dtype=float),
    idx: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=float),
):
    i = wp.tid()
    arr[idx[i]] = vals[i]


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
    return _build_bary_map_with_logging(
        full_verts,
        phys_verts,
        phys_faces,
    )


def _load_kfc_mesh_zup():
    return _load_kfc_mesh_zup_common(_BAG_H)


def _decimate_mesh(verts, faces, target_faces, proxy_mode):
    return _decimate_mesh_common(verts, faces, target_faces, proxy_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Geometric overlap testing for initial object placement
# ─────────────────────────────────────────────────────────────────────────────


def _fit_bag_contents(phys_verts_cm):
    """Find overlap-free start positions for three content objects inside the bag.

    The bag body is roughly rectangular (≈20 cm wide in X, 14 cm deep in Y).
    Its inner profile narrows sharply above z≈12 cm toward the handles, so all
    objects must stay in the lower half.

    Placement strategy (verified against physics proxy mesh):
      - Sphere (R=4) and box (6×6×6) sit side-by-side in X at the same height.
      - Capsule (R=3, HL=2) is rotated to lie HORIZONTALLY along the bag's Y-axis
        and placed above both at the bag centre, where there is room before
        the bag narrows.

    The fixed offsets below keep the objects clear of the bag wall and each
    other for the default proxy mesh; the exact wall clearances are logged.

    Returns
    -------
    pos_sphere, pos_box, pos_capsule : tuple of (x, y, z) in local bag cm
    cap_quat : wp.quat — rotation that makes the capsule Y-axis aligned
    """
    import math

    SPHERE_R = 4.0
    BOX_H = 3.0
    CAP_R, CAP_HL = 3.0, 2.0
    CLEARANCE = 0.5

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
        """Min distance from bag vertices to a Y-axis horizontal capsule."""
        vy = np.clip(phys_verts_cm[:, 1], oy - hl, oy + hl)
        d = np.sqrt((phys_verts_cm[:, 0] - ox) ** 2
                    + (phys_verts_cm[:, 1] - vy) ** 2
                    + (phys_verts_cm[:, 2] - oz) ** 2)
        return float(d.min()) - r

    # ── Fixed offsets validated against the physics proxy mesh ────────────────
    # Sphere: shifted +2.5 cm in X, z=5 cm (clearance ≥1.0 cm to bag walls)
    # Box:    shifted −5.5 cm in X, z=4 cm (clearance ≥1.0 cm to bag walls)
    # Capsule (Y-axis horizontal): bag centre, z=12.5 cm (clearance ≥0.72 cm)
    # Inter-object gaps: sphere↔box≥1.0, sphere↔capsule≥0.9, box↔capsule≥3.0
    s_x, s_y, s_z = bag_cx + 2.5, bag_cy, 5.0
    b_x, b_y, b_z = bag_cx - 5.5, bag_cy, 4.0
    c_x, c_y, c_z = bag_cx,       bag_cy, 12.5

    # Capsule rotation: 90° around X-axis rotates the default Z-axis capsule
    # to be Y-axis aligned (lying flat across the bag's narrow dimension).
    cap_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)

    cs = dist_sphere(s_x, s_y, s_z, SPHERE_R)
    cb = dist_box(b_x, b_y, b_z, BOX_H, BOX_H, BOX_H)
    cc = dist_cap_y(c_x, c_y, c_z, CAP_R, CAP_HL)
    _log_content_placements_cm(
        sphere_pos_cm=(s_x, s_y, s_z),
        sphere_clearance_cm=cs,
        box_pos_cm=(b_x, b_y, b_z),
        box_clearance_cm=cb,
        capsule_pos_cm=(c_x, c_y, c_z),
        capsule_clearance_cm=cc,
        local_bag_coords=True,
    )
    return (s_x, s_y, s_z), (b_x, b_y, b_z), (c_x, c_y, c_z), cap_quat


# ─────────────────────────────────────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────────────────────────────────────


class Example:
    """KFC bag with 3 dynamic rigid bodies dropping onto the ground.

    VBD solves both cloth (particles) and rigid bodies (AVBD) in a
    unified coupled simulation.  All physics run in centimeters.
    Positions are scaled to metres in render().
    """

    def __init__(
        self,
        viewer,
        save_mp4: str | None = None,
        test_mode: bool = False,
        soft_bag: bool = False,
        capture_replay: bool = False,
        capture_frames: int = 300,
        capture_fps: int = 60,
        capture_dir: str = "outputs/replay_capture",
        capture_format: str = "mp4",
        target_faces: int = _PHYSICS_TARGET_FACES,
        mesh_proxy_mode: str = _DEFAULT_PROXY_MODE,
    ):
        self.viewer = viewer
        self.test_mode = test_mode
        self.soft_bag = bool(soft_bag)
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

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = _SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.soft_bag:
            bag_tri_ke = _SOFT_TRI_KE
            bag_tri_ka = _SOFT_TRI_KA
            bag_tri_kd = _SOFT_TRI_KD
            bag_edge_ke = _SOFT_EDGE_KE
            bag_edge_kd = _SOFT_EDGE_KD
            obj_shape_ke = _SOFT_OBJ_SHAPE_KE
            print(
                f"{_LOG_PREFIX} Using soft bag tuning:"
                f" tri_ke={bag_tri_ke:.3g}, tri_ka={bag_tri_ka:.3g},"
                f" tri_kd={bag_tri_kd:.3g}, edge_ke={bag_edge_ke:.3g},"
                f" edge_kd={bag_edge_kd:.3g},"
                f" obj_contact_ke={0.5 * (_CONTACT_KE + obj_shape_ke):.3g}"
            )
        else:
            bag_tri_ke = _TRI_KE
            bag_tri_ka = _TRI_KA
            bag_tri_kd = _TRI_KD
            bag_edge_ke = _EDGE_KE
            bag_edge_kd = _EDGE_KD
            obj_shape_ke = _OBJ_SHAPE_KE

        # ── Load meshes ────────────────────────────────────────────────────
        full_verts_cm, full_faces = _load_kfc_mesh_zup()
        phys_verts_cm, phys_faces = _decimate_mesh(
            full_verts_cm,
            full_faces,
            target_faces,
            mesh_proxy_mode,
        )

        _log_mesh_counts(
            full_verts=full_verts_cm,
            full_faces=full_faces,
        )

        # ── Barycentric map: full-res → physics mesh ──────────────────────
        # Each full-res vertex is mapped to a physics triangle + barycentrics
        # so the high-res visual mesh tracks the low-res simulation mesh.
        self._bary_vi0_np, self._bary_vi1_np, self._bary_vi2_np, bary_w = \
            _build_bary_map(full_verts_cm, phys_verts_cm, phys_faces)
        self._bary_w = wp.array(bary_w, dtype=wp.vec3)
        self._n_full_verts = len(full_verts_cm)
        self._full_indices_wp = wp.array(
            full_faces.flatten().astype(np.int32), dtype=wp.int32,
        )
        self._viz_full_q = wp.zeros(self._n_full_verts, dtype=wp.vec3)

        # Compute per-vertex displacement: the difference between the actual
        # full-res vertex and its barycentric projection onto the physics mesh.
        # This preserves all geometric detail of the original kfc.usd mesh.
        bary_proj = (
            phys_verts_cm[self._bary_vi0_np] * bary_w[:, 0:1]
            + phys_verts_cm[self._bary_vi1_np] * bary_w[:, 1:2]
            + phys_verts_cm[self._bary_vi2_np] * bary_w[:, 2:3]
        )
        self._bary_disp = wp.array(
            (full_verts_cm - bary_proj).astype(np.float32), dtype=wp.vec3,
        )

        # ── Build scene ────────────────────────────────────────────────────
        builder = newton.ModelBuilder(gravity=_G_CM)

        # 3 dynamic content bodies (food items inside the bag).
        #
        # Key insight for cloth-body coupling:
        #   1. CollisionPipeline must use broad_phase='nxn' — the default
        #      "explicit" broadphase does not test particle-body pairs at all.
        #   2. Objects must be large enough to touch the bag walls
        #      (bag interior ~8 cm from center to wall → radius ≥ 3 cm).
        #   3. Builder masses are placeholders until the post-finalize
        #      _OBJECT_MASS_G inertia override below.
        #
        # Positions are computed geometrically from the physics mesh so that
        # no object overlaps the bag shell at simulation start.
        (s_x, s_y, s_z), (b_x, b_y, b_z), (c_x, c_y, c_z), cap_quat = _fit_bag_contents(phys_verts_cm)

        content_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.001, mu=0.5, ke=_OBJ_SHAPE_KE, kd=_OBJ_SHAPE_KE * 1e-4,
        )

        builder.add_body(
            xform=wp.transform(
                wp.vec3(s_x, s_y, s_z + _DROP_HEIGHT), wp.quat_identity()
            ),
            mass=0.1,
        )
        builder.add_shape_sphere(body=0, radius=4.0, cfg=content_cfg)

        builder.add_body(
            xform=wp.transform(
                wp.vec3(b_x, b_y, b_z + _DROP_HEIGHT), wp.quat_identity()
            ),
            mass=0.1,
        )
        builder.add_shape_box(body=1, hx=3.0, hy=3.0, hz=3.0, cfg=content_cfg)

        builder.add_body(
            xform=wp.transform(
                wp.vec3(c_x, c_y, c_z + _DROP_HEIGHT), cap_quat
            ),
            mass=0.1,
        )
        builder.add_shape_capsule(body=2, radius=3.0, half_height=2.0, cfg=content_cfg)

        # Bag cloth mesh (physics resolution, dropped from _DROP_HEIGHT)
        self._bag_particle_start = builder.particle_count

        bag_verts_wp = [
            wp.vec3(float(v[0]), float(v[1]), float(v[2]))
            for v in phys_verts_cm
        ]
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, _DROP_HEIGHT),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=bag_verts_wp,
            indices=phys_faces.flatten().tolist(),
            density=_DENSITY,
            tri_ke=bag_tri_ke,
            tri_ka=bag_tri_ka,
            tri_kd=bag_tri_kd,
            edge_ke=bag_edge_ke,
            edge_kd=bag_edge_kd,
            particle_radius=_PARTICLE_RADIUS,
        )
        self._bag_particle_end = builder.particle_count

        # Offset barycentric vertex indices by particle_start so they index
        # correctly into the global particle_q array.
        ps = self._bag_particle_start
        self._bary_vi0 = wp.array(self._bary_vi0_np + ps, dtype=wp.int32)
        self._bary_vi1 = wp.array(self._bary_vi1_np + ps, dtype=wp.int32)
        self._bary_vi2 = wp.array(self._bary_vi2_np + ps, dtype=wp.int32)
        del self._bary_vi0_np, self._bary_vi1_np, self._bary_vi2_np

        # Physics proxy mesh indices (offset by particle_start) for collision view
        self._proxy_indices_wp = wp.array(
            (phys_faces.flatten() + ps).astype(np.int32), dtype=wp.int32,
        )

        # Ground plane(s).
        # Soft bag uses two separate planes so cloth and rigid-body stiffness
        # requirements (5 orders of magnitude apart) can be tuned independently.
        if self.soft_bag:
            # Particle-only plane: cloth contacts here.  kd damps first-impact bounce.
            _pgrnd_cfg = newton.ModelBuilder.ShapeConfig(
                ke=_SOFT_PARTICLE_GROUND_KE, kd=_SOFT_PARTICLE_GROUND_KD, mu=0.4,
                has_shape_collision=False)
            builder.add_ground_plane(cfg=_pgrnd_cfg, label="soft_particle_ground")
            # Rigid-only plane: stops 1000 g objects quickly via high avg_ke.
            _rgrnd_cfg = newton.ModelBuilder.ShapeConfig(
                ke=_SOFT_RIGID_GROUND_KE, kd=_SOFT_RIGID_GROUND_KD, mu=0.4,
                has_particle_collision=False)
            builder.add_ground_plane(cfg=_rgrnd_cfg, label="soft_rigid_ground")
            _n_ground_shapes = 2
        else:
            ground_cfg = newton.ModelBuilder.ShapeConfig(ke=5e4, kd=0.5, mu=0.4)
            builder.add_ground_plane(cfg=ground_cfg)
            _n_ground_shapes = 1

        # Color before finalize (required by VBD for parallel solve)
        builder.color(include_bending=True)

        # ── Finalize ──────────────────────────────────────────────────────
        self.model = builder.finalize()

        # soft_contact_ke is the particle-side stiffness; VBD averages it with
        # the shape-side shape_material_ke to get the effective contact stiffness.
        self.model.soft_contact_ke = _CONTACT_KE
        self.model.soft_contact_kd = _CONTACT_KE * 1e-4
        self.model.soft_contact_mu = 2.0

        # Shape materials: set per-shape ke/kd.
        # Content shapes (sphere, box, capsule) are the first n_shapes - _n_ground_shapes.
        # Ground shape(s) ke/kd are already baked in via ShapeConfig above.
        n_shapes = self.model.shape_material_ke.numpy().shape[0]
        ke_arr = self.model.shape_material_ke.numpy().copy()
        kd_arr = self.model.shape_material_kd.numpy().copy()
        mu_arr = self.model.shape_material_mu.numpy().copy()
        n_content = n_shapes - _n_ground_shapes
        for s in range(n_content):  # content shapes (not ground)
            ke_arr[s] = obj_shape_ke
            # kd=0 for soft bag content shapes: preserves sag dynamics.
            # Ground shape kd is set via ShapeConfig (above) and left untouched.
            kd_arr[s] = 0.0 if self.soft_bag else obj_shape_ke * 1e-4
        if not self.soft_bag:
            # Hard bag: single ground plane, override ke/kd.
            ke_arr[n_content] = _GROUND_SHAPE_KE
            kd_arr[n_content] = _GROUND_SHAPE_KE * 1e-4
        # Soft bag ground planes: ke/kd already set via ShapeConfig; leave untouched.
        if self.soft_bag:
            # Zero particle-side damping: cloth-object contact uses no kd so
            # sag is governed purely by cloth stiffness (not damping).
            # Ground damping comes from shape_material_kd on the ground shapes.
            self.model.soft_contact_kd = 0.0
        self.model.shape_material_ke = wp.array(ke_arr, dtype=float)
        self.model.shape_material_kd = wp.array(kd_arr, dtype=float)
        self.model.shape_material_mu = wp.array(mu_arr, dtype=float)

        # Content body mass/inertia override (bodies 0, 1, 2 = sphere, box, capsule)
        inv_m = self.model.body_inv_mass.numpy().copy()
        inv_i = self.model.body_inv_inertia.numpy().copy()
        _content_radii  = [4.0, 3.0, 3.0]
        _content_ccoeff = [0.4, 2.0 / 3.0, 0.5]
        for k in range(3):
            inv_m[k] = 1.0 / _OBJECT_MASS_G
            r = _content_radii[k]
            c = _content_ccoeff[k]
            inv_i[k] = 1.0 / (c * _OBJECT_MASS_G * r * r) * np.eye(3)
        self.model.body_inv_mass    = wp.array(inv_m, dtype=float)
        self.model.body_inv_inertia = wp.array(inv_i, dtype=wp.mat33)

        all_mass = self.model.particle_mass.numpy()
        bag_mass = float(
            all_mass[self._bag_particle_start:self._bag_particle_end].sum()
        )
        object_masses = 1.0 / inv_m[:3]

        print(f"{_LOG_PREFIX} Bodies: {self.model.body_count}, "
              f"Particles: {self.model.particle_count}, "
              f"Shapes: {self.model.shape_count}")
        print(f"{_LOG_PREFIX} Bag mass: {bag_mass:.6f} g")
        print(
            f"{_LOG_PREFIX} Object masses: "
            f"sphere={float(object_masses[0]):.6f} g, "
            f"box={float(object_masses[1]):.6f} g, "
            f"capsule={float(object_masses[2]):.6f} g, "
            f"total={float(object_masses.sum()):.6f} g"
        )

        # ── Grip: pin top 12% of bag particles ───────────────────────────
        # These particles are held fixed during the hang phase, then released.
        z_top_thresh = 0.88 * _BAG_H
        local_z = phys_verts_cm[:, 2]
        handle_rel = np.where(local_z >= z_top_thresh)[0].astype(np.int32)
        if len(handle_rel) < 4:
            z_top_thresh = 0.95 * _BAG_H
            handle_rel = np.where(local_z >= z_top_thresh)[0].astype(np.int32)
        self._handle_global = (handle_rel + self._bag_particle_start).astype(np.int32)
        self._handle_idx_wp = wp.array(self._handle_global, dtype=wp.int32)

        all_inv_m = self.model.particle_inv_mass.numpy()
        self._handle_inv_mass_orig_wp = wp.array(
            all_inv_m[self._handle_global].astype(np.float32), dtype=float)
        self._handle_mass_orig_wp = wp.array(
            all_mass[self._handle_global].astype(np.float32), dtype=float)

        # Pin handles now (will release after _HANG_FRAMES)
        self._is_hanging = True
        self._frame_count = 0
        n_h = len(self._handle_global)
        wp.launch(_k_set_float_scalar, dim=n_h,
                  inputs=[self.model.particle_inv_mass, self._handle_idx_wp, 0.0])
        wp.launch(_k_set_float_scalar, dim=n_h,
                  inputs=[self.model.particle_mass, self._handle_idx_wp, 0.0])

        print(f"{_LOG_PREFIX} Handle particles: {n_h} "
              f"(z >= {z_top_thresh:.1f} cm), hanging for {_HANG_FRAMES} frames")

        # ── VBD solver (solves both cloth + rigid bodies) ─────────────────
        # Soft bag uses a much higher rigid_contact_k_start so that the AVBD
        # penalty is already large enough to stop 1000 g objects on the first
        # contact substep, avoiding the explosive deep-penetration instability.
        _rigid_k = _SOFT_RIGID_K_START if self.soft_bag else _OBJ_SHAPE_KE
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=_VBD_ITERS,
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
            rigid_contact_k_start=_rigid_k,
        )

        # ── States / contacts ─────────────────────────────────────────────
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # FK for initial body transforms
        newton.eval_fk(
            self.model, self.state_0.joint_q,
            self.state_0.joint_qd, self.state_0,
        )

        # CRITICAL: broad_phase='nxn' is required for particle-body contacts.
        # The default "explicit" broadphase only checks registered pairs and
        # completely misses particle-body interactions.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        # ── Viz state (metre scale) ───────────────────────────────────────
        self.viz_state = self.model.state()

        # Store original shape data (cm) and create metre-scale copies
        self._sim_shape_transform = self.model.shape_transform
        self._sim_shape_scale = self.model.shape_scale

        if self.model.shape_count > 0:
            st_np = self.model.shape_transform.numpy().copy()
            for i in range(len(st_np)):
                st_np[i][:3] *= _VIZ_SCALE
            self._viz_shape_transform = wp.array(st_np, dtype=wp.transform)

            ss_np = self.model.shape_scale.numpy().copy()
            ss_np *= _VIZ_SCALE
            self._viz_shape_scale = wp.array(ss_np, dtype=wp.vec3)
        else:
            self._viz_shape_transform = self._sim_shape_transform
            self._viz_shape_scale = self._sim_shape_scale

        # ── Viewer setup ──────────────────────────────────────────────────
        # GL viewer bakes shape dimensions at set_model() time, so scale
        # shape transforms and scales to metre space BEFORE set_model().
        self.model.shape_transform = self._viz_shape_transform
        self.model.shape_scale = self._viz_shape_scale

        self.viewer.set_model(self.model)

        # Suppress the viewer's built-in cloth mesh draw — the shared bag
        # renderer logs either the hi-res or proxy mesh explicitly.
        self.viewer.show_triangles = False

        # Restore cm-scale shape data for simulation
        self.model.shape_transform = self._sim_shape_transform
        self.model.shape_scale = self._sim_shape_scale

        if hasattr(self.viewer, "renderer"):
            # Bag hangs at z=0.5–0.78 m (centre ~0.64 m).  Camera sits at
            # z=0.6 m, ~1.3 m back, looking slightly down so the full hang
            # and the subsequent drop to the ground are both in frame.
            self.viewer.set_camera(
                pos=wp.vec3(1.0, -1.0, 0.6),
                pitch=-8.0,
                yaw=135.0,
            )

        if self.save_mp4:
            self._init_video_capture()

    # ─────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────

    def simulate(self):
        # VBD solves both cloth particles and rigid bodies together
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control,
                self.contacts, self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    # ─────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────

    def step(self):
        if self.capture_done:
            return

        self._frame_count += 1

        # Release the grip after hang phase
        if self._is_hanging and self._frame_count > _HANG_FRAMES:
            self._is_hanging = False
            n_h = len(self._handle_global)
            wp.launch(_k_set_float_array, dim=n_h,
                      inputs=[self.model.particle_mass,
                               self._handle_idx_wp,
                               self._handle_mass_orig_wp])
            wp.launch(_k_set_float_array, dim=n_h,
                      inputs=[self.model.particle_inv_mass,
                               self._handle_idx_wp,
                               self._handle_inv_mass_orig_wp])

        self.simulate()
        self.sim_time += self.frame_dt

    def _update_render_buffers(self):
        """Refresh cached render buffers for the current simulation frame."""
        wp.launch(
            _k_bary_interp,
            dim=self._n_full_verts,
            inputs=[
                self.state_0.particle_q,
                self._bary_vi0,
                self._bary_vi1,
                self._bary_vi2,
                self._bary_w,
                self._bary_disp,
                _VIZ_SCALE,
            ],
            outputs=[self._viz_full_q],
        )

        # Scale particles and body transforms cm -> m
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

        _render_bag_meshes(
            self.viewer,
            sim_time=self.sim_time,
            viz_state=self.viz_state,
            full_positions=self._viz_full_q,
            full_indices=self._full_indices_wp,
            proxy_positions=self.viz_state.particle_q,
            proxy_indices=self._proxy_indices_wp,
        )

        _write_video_frame_common(self)
        self._capture_replay_frame()

    def test_final(self):
        pq = self.state_0.particle_q.numpy()
        bag_z = pq[self._bag_particle_start:self._bag_particle_end, 2]
        # Bag should have settled near the ground
        assert float(bag_z.max()) < _BAG_H + 5.0, (
            f"Bag top too high: {float(bag_z.max()):.1f} cm"
        )
        assert float(bag_z.min()) >= -1.0, (
            f"Bag penetrated ground: {float(bag_z.min()):.1f} cm"
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
    parser.set_defaults(num_frames=300)
    parser.add_argument(
        "--soft-bag",
        action="store_true",
        help="Use a softer bag material tuning",
    )
    _add_proxy_mesh_arguments(parser)
    _add_capture_arguments(
        parser,
        replay_help="Capture rendered frames and build replay video",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        save_mp4=args.save_mp4,
        test_mode=args.test,
        soft_bag=bool(args.soft_bag),
        capture_replay=bool(args.capture_replay),
        capture_frames=int(args.capture_frames),
        capture_fps=int(args.capture_fps),
        capture_dir=str(args.capture_dir),
        capture_format=str(args.capture_format),
        target_faces=int(args.target_faces),
        mesh_proxy_mode=str(args.proxy_mode),
    )

    while viewer.is_running() and not example.capture_done:
        example.step()
        example.render()

    if args.test:
        example.test_final()

    _finalize_capture(example)
