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
# - Physics/cloth mesh: ~1200 tris via pymeshlab quadric decimation
# - Collision proxy: same decimated mesh (particle radius provides margin)
# - 3 kinematic rigid bodies inside the bag
# - VBD solver, centimeter scale (gravity = -981 cm/s²)
#
# Command: python -m newton.examples robot.example_kfc_bag_drop
#
###########################################################################

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import numpy as np
import warp as wp

import newton
import newton.examples

# ─────────────────────────────────────────────────────────────────────────────
# Simulation scale: centimeters
# ─────────────────────────────────────────────────────────────────────────────
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
    """For each full-res vertex, find the closest physics triangle and barycentrics.

    Returns
    -------
    vi0, vi1, vi2 : ndarray (N,) int32 — physics vertex indices per full-res vert
    bary : ndarray (N, 3) float32 — barycentric weights
    """
    from scipy.spatial import cKDTree

    # Build a KD-tree of physics triangle centroids for fast lookup
    v0 = phys_verts[phys_faces[:, 0]]
    v1 = phys_verts[phys_faces[:, 1]]
    v2 = phys_verts[phys_faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    tree = cKDTree(centroids)

    n_full = len(full_verts)
    vi0 = np.zeros(n_full, dtype=np.int32)
    vi1 = np.zeros(n_full, dtype=np.int32)
    vi2 = np.zeros(n_full, dtype=np.int32)
    bary = np.zeros((n_full, 3), dtype=np.float32)

    # Query nearest k centroids and pick the best triangle
    _, nearest = tree.query(full_verts, k=min(5, len(centroids)))
    if nearest.ndim == 1:
        nearest = nearest[:, None]

    for i in range(n_full):
        p = full_verts[i]
        best_dist = 1e30
        best_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        best_t = 0
        for ti in nearest[i]:
            a, b, c = v0[ti], v1[ti], v2[ti]
            # Barycentric coords via projection
            e0 = b - a
            e1 = c - a
            v = p - a
            d00 = e0 @ e0
            d01 = e0 @ e1
            d11 = e1 @ e1
            dv0 = v @ e0
            dv1 = v @ e1
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                continue
            u = (d11 * dv0 - d01 * dv1) / denom
            w = (d00 * dv1 - d01 * dv0) / denom
            t = 1.0 - u - w
            # Clamp to triangle
            t = max(0.0, min(1.0, t))
            u = max(0.0, min(1.0, u))
            w = max(0.0, min(1.0, w))
            s = t + u + w
            if s > 0:
                t /= s
                u /= s
                w /= s
            proj = a * t + b * u + c * w
            dist = float(np.sum((p - proj) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_b = np.array([t, u, w], dtype=np.float32)
                best_t = ti
        vi0[i] = phys_faces[best_t, 0]
        vi1[i] = phys_faces[best_t, 1]
        vi2[i] = phys_faces[best_t, 2]
        bary[i] = best_b

    return vi0, vi1, vi2, bary


def _load_kfc_mesh_zup():
    """Load KFC bag mesh from kfc.usd, convert to Z-up and scale to cm.

    Returns full-resolution vertices (cm) and face indices.
    """
    from pxr import Usd, UsdGeom

    usd_path = str(newton.examples.get_asset("kfc.usd"))
    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetPrimAtPath("/World/material/material_001")
    usd_mesh = UsdGeom.Mesh(prim)

    pts = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
    faces = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)

    # USD is Y-up; rotate to Z-up: (x, y, z) -> (x, -z, y)
    pts_zup = np.column_stack([pts[:, 0], -pts[:, 2], pts[:, 1]])

    # Scale so bag height = _BAG_H cm
    usd_h_m = float(pts_zup[:, 2].max() - pts_zup[:, 2].min())
    scale_m = (_BAG_H / 100.0) / usd_h_m
    pts_zup *= scale_m
    pts_zup[:, 2] -= float(pts_zup[:, 2].min())  # base at z=0

    verts_cm = (pts_zup * 100.0).astype(np.float32)
    return verts_cm, faces


def _decimate_mesh(verts, faces, target_faces):
    """Simplify a mesh to approximately target_faces using pymeshlab.

    Uses isotropic remeshing (not quadric decimation) to produce uniformly
    sized triangles that VBD can simulate stably.  Degenerate triangles at
    open boundaries are filtered out.
    """
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts, faces))

    # Isotropic remeshing: 5% of bounding box diagonal produces ~1500
    # well-shaped triangles from a 50K-tri bag mesh.
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PercentageValue(5.0),
        iterations=10,
    )
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    dm = ms.current_mesh()
    out_v = np.array(dm.vertex_matrix(), dtype=np.float32)
    out_f = np.array(dm.face_matrix(), dtype=np.int32)

    # Remove degenerate triangles (open-boundary artifacts from remeshing)
    areas = np.array([
        0.5 * np.linalg.norm(np.cross(
            out_v[t[1]] - out_v[t[0]], out_v[t[2]] - out_v[t[0]]))
        for t in out_f
    ])
    keep = areas > 0.1  # cm² minimum
    out_f = out_f[keep]
    used = np.unique(out_f)
    remap = np.full(len(out_v), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    out_v = out_v[used]
    out_f = remap[out_f]

    return out_v, out_f


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

    All clearances to the bag wall and between objects are ≥ 0.5 cm.

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
    print(
        f"[KFC drop] Object placements (local bag coords, clearance to bag wall):\n"
        f"  sphere  @ ({s_x:.1f}, {s_y:.1f}, {s_z:.1f})  clr={cs:.2f} cm\n"
        f"  box     @ ({b_x:.1f}, {b_y:.1f}, {b_z:.1f})  clr={cb:.2f} cm\n"
        f"  capsule @ ({c_x:.1f}, {c_y:.1f}, {c_z:.1f}) [Y-horiz]  clr={cc:.2f} cm"
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
    ):
        self.viewer = viewer
        self.test_mode = test_mode
        self.save_mp4 = save_mp4
        self.soft_bag = bool(soft_bag)
        self.capture_replay = bool(capture_replay)
        self.capture_frames = int(capture_frames)
        self.capture_fps = int(capture_fps)
        self.capture_format = str(capture_format)
        self.capture_count = 0
        self.capture_done = False
        self.capture_video_path = None
        self.capture_dir = None

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = _SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        self._video_process = None
        if self.capture_replay and self.capture_frames > 0:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path(capture_dir)
            self.capture_dir = base_dir / f"run_{run_tag}"
            self.capture_dir.mkdir(parents=True, exist_ok=True)

        if self.soft_bag:
            bag_tri_ke = _SOFT_TRI_KE
            bag_tri_ka = _SOFT_TRI_KA
            bag_tri_kd = _SOFT_TRI_KD
            bag_edge_ke = _SOFT_EDGE_KE
            bag_edge_kd = _SOFT_EDGE_KD
            obj_shape_ke = _SOFT_OBJ_SHAPE_KE
            print(
                "[KFC drop] Using soft bag tuning:"
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
            full_verts_cm, full_faces, _PHYSICS_TARGET_FACES
        )

        print(f"[KFC drop] Full mesh: {len(full_verts_cm)} verts, {len(full_faces)} tris")
        print(f"[KFC drop] Physics mesh: {len(phys_verts_cm)} verts, {len(phys_faces)} tris")

        # ── Barycentric map: full-res → physics mesh ──────────────────────
        # Each full-res vertex is mapped to a physics triangle + barycentrics
        # so the high-res visual mesh tracks the low-res simulation mesh.
        print("[KFC drop] Building barycentric map...", end=" ", flush=True)
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
        print("done.")

        # ── Build scene ────────────────────────────────────────────────────
        builder = newton.ModelBuilder(gravity=_G_CM)

        # 3 dynamic content bodies (food items inside the bag).
        #
        # Key insight for cloth-body coupling:
        #   1. CollisionPipeline must use broad_phase='nxn' — the default
        #      "explicit" broadphase does not test particle-body pairs at all.
        #   2. Objects must be large enough to touch the bag walls
        #      (bag interior ~8 cm from center to wall → radius ≥ 3 cm).
        #   3. Very low mass (0.1 g) combined with high ke lets the bag
        #      support them while keeping the simulation stable.
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
        self.model.soft_contact_mu = 0.5

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

        print(f"[KFC drop] Bodies: {self.model.body_count}, "
              f"Particles: {self.model.particle_count}, "
              f"Shapes: {self.model.shape_count}")
        print(f"[KFC drop] Bag mass: {bag_mass:.6f} g")
        print(
            "[KFC drop] Object masses: "
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

        print(f"[KFC drop] Handle particles: {n_h} "
              f"(z >= {z_top_thresh:.1f} cm), hanging for {_HANG_FRAMES} frames")

        # ── VBD solver (solves both cloth + rigid bodies) ─────────────────
        # Soft bag uses a much higher rigid_contact_k_start so that the AVBD
        # penalty is already large enough to stop 1000 g objects on the first
        # contact substep, avoiding the explosive deep-penetration instability.
        _rigid_k = _SOFT_RIGID_K_START if self.soft_bag else 1.0e3
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

        # Suppress the viewer's built-in cloth mesh draw — we render
        # explicitly as either hi-res or proxy mesh via log_mesh().
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

    def render(self):
        if self.capture_done:
            return

        proxy_mode = bool(
            self.viewer.show_collision or self.viewer.show_triangles
        )

        # Interpolate full-res bag mesh from physics particles via barycentrics
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

        # Suppress viewer's built-in cloth draw during log_state
        show_triangles = self.viewer.show_triangles
        self.viewer.show_triangles = False
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.show_triangles = show_triangles

        # Hi-res USD mesh: visible when collision view is OFF
        self.viewer.log_mesh(
            "/bag",
            self._viz_full_q,
            self._full_indices_wp,
            backface_culling=False,
            hidden=proxy_mode,
            alpha=0.5,
        )

        # Physics proxy mesh: visible when collision view is ON
        self.viewer.log_mesh(
            "/bag_proxy",
            self.viz_state.particle_q,
            self._proxy_indices_wp,
            backface_culling=False,
            hidden=not proxy_mode,
        )

        self.viewer.end_frame()

        if self._video_process is not None and hasattr(self.viewer, "get_frame"):
            frame = self._get_viewer_frame()
            self._video_process.stdin.write(frame.numpy().tobytes())
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
        if not hasattr(self.viewer, "get_frame"):
            print("Warning: viewer lacks get_frame(); skipping MP4")
            return
        try:
            w = self.viewer.renderer._screen_width
            h = self.viewer.renderer._screen_height
        except AttributeError:
            print("Warning: cannot determine screen size; skipping MP4")
            return
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(self.fps), "-i", "pipe:0",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            self.save_mp4,
        ]
        try:
            self._video_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            print("Warning: ffmpeg not found; skipping MP4")

    def _get_viewer_frame(self, *, render_ui: bool = False):
        try:
            return self.viewer.get_frame(render_ui=render_ui)
        except TypeError:
            return self.viewer.get_frame()

    def _capture_replay_frame(self):
        if not self.capture_replay or self.capture_done:
            return
        if self.capture_dir is None:
            return
        if self.capture_count >= self.capture_frames:
            self._finalize_replay_video()
            self.capture_done = True
            self.viewer.close()
            return
        if not hasattr(self.viewer, "get_frame"):
            return

        frame_wp = self._get_viewer_frame(render_ui=False)
        frame_np = frame_wp.numpy()
        out_path = self.capture_dir / f"frame_{self.capture_count:05d}.png"
        imageio.imwrite(out_path, frame_np)
        self.capture_count += 1

        if self.capture_count % 20 == 0:
            print(
                f"[replay_capture] saved"
                f" {self.capture_count}/{self.capture_frames} frames"
            )

        if self.capture_count >= self.capture_frames:
            self._finalize_replay_video()
            self.capture_done = True
            self.viewer.close()

    def _finalize_replay_video(self):
        if self.capture_dir is None:
            return
        png_files = sorted(self.capture_dir.glob("frame_*.png"))
        if len(png_files) == 0:
            return

        try:
            if self.capture_format == "gif":
                video_path = self.capture_dir / "replay.gif"
                with imageio.get_writer(
                    video_path,
                    mode="I",
                    duration=1.0 / max(self.capture_fps, 1),
                ) as writer:
                    for path in png_files:
                        writer.append_data(imageio.imread(path))
            else:
                video_path = self.capture_dir / "replay.mp4"
                with imageio.get_writer(
                    video_path,
                    fps=max(self.capture_fps, 1),
                    codec="libx264",
                ) as writer:
                    for path in png_files:
                        writer.append_data(imageio.imread(path))
            self.capture_video_path = video_path
            print(f"[replay_capture] wrote video: {video_path}")
        except Exception as exc:
            fallback = self.capture_dir / "replay.gif"
            with imageio.get_writer(
                fallback,
                mode="I",
                duration=1.0 / max(self.capture_fps, 1),
            ) as writer:
                for path in png_files:
                    writer.append_data(imageio.imread(path))
            self.capture_video_path = fallback
            print(
                f"[replay_capture] mp4 failed ({exc});"
                f" wrote gif: {fallback}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    parser.add_argument(
        "--save-mp4", type=str, default=None,
        help="Save simulation to MP4 file",
    )
    parser.add_argument(
        "--soft-bag",
        action="store_true",
        help="Use a softer bag material tuning",
    )
    parser.add_argument(
        "--capture-replay",
        action="store_true",
        help="Capture rendered frames and build replay video",
    )
    parser.add_argument(
        "--capture-frames",
        type=int,
        default=300,
        help="Number of frames to capture when replay capture is enabled",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=60,
        help="Output replay video FPS",
    )
    parser.add_argument(
        "--capture-dir",
        type=str,
        default="outputs/replay_capture",
        help="Directory to store captured frames and replay video",
    )
    parser.add_argument(
        "--capture-format",
        type=str,
        default="mp4",
        choices=["mp4", "gif"],
        help="Preferred replay output format",
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
    )

    for i in range(args.num_frames):
        if not viewer.is_running() or example.capture_done:
            break
        example.step()
        example.render()

    if args.test:
        example.test_final()
