# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Table Cloth (VBD) — Newton port of the PhysX spread_tablecloth
#
# A folded tablecloth (loaded from ``assets/cloth``) is released above a
# table while a Unitree G1 humanoid holds its initial pose nearby. The
# cloth falls, contacts the table top, and drapes under gravity using the
# Newton VBD cloth solver.
#
# This is a port of the Isaac Lab + PhysX scene from the
# ``i4h-workflows`` ``spread_tablecloth`` task — VBD replaces the PhysX
# FEM-cloth solver and the G1 is kinematic (held at the initial pose by
# zeroing its body inverse masses).
#
# Assets (all loaded from ``newton/examples/assets/cloth/``):
#   - Cloth: ``Cloth_fold10.usd``. The proxy mesh
#     ``/root/Cloth_fold07/Visuals/Cloth_fold06`` (~2.5k verts) drives the
#     physics; the heavier 28k-vert mesh in the same file is skipped.
#   - Robot: ``g1_29dof_with_inspire_rev_1_0.usd`` (G1 + Inspire hands).
#   - Table: ``Table256.usd``. The visual mesh
#     ``/root/Table256/Visuals/Table256`` is added as a non-collidable shape
#     for rendering; cloth contact is handled by a simple box proxy whose
#     top surface sits at the same height as the table top.
#
# Command: python -m newton.examples table_cloth_vbd
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.examples.bag import table_cloth as tc
from newton.examples.bag.capture import (
    add_capture_arguments as _add_capture_arguments,
)
from newton.examples.bag.capture import (
    capture_replay_frame as _capture_replay_frame_common,
)
from newton.examples.bag.capture import (
    configure_capture as _configure_capture,
)
from newton.examples.bag.capture import (
    finalize_capture as _finalize_capture,
)
from newton.examples.bag.capture import (
    finalize_replay_video as _finalize_replay_video_common,
)
from newton.examples.bag.capture import (
    get_viewer_frame as _get_viewer_frame_common,
)
from newton.examples.bag.capture import (
    init_video_capture as _init_video_capture_common,
)
from newton.examples.bag.capture import (
    write_video_frame as _write_video_frame_common,
)

# ─────────────────────────────────────────────────────────────────────────────
# Scene geometry — composed once at import time
# ─────────────────────────────────────────────────────────────────────────────
#
# The cloth + pile world placement is composed from two sources:
#   1. The cloth USD's authored xform on its Cloth_fold07 / Cloth_In002
#      prims — translate + orient about Z. This is what positions the
#      mesh relative to the cloth-USD root.
#   2. IL's env_cfg cloth.init_state.pos / .rot — the world placement
#      the env spawns the cloth-USD root at.
# Table placement is composed from scene04.usd's /World/Table256 xform
# and scene04's own env_cfg placement. See :mod:`table_cloth` for both.
_CLOTH_INIT_POS, _CLOTH_INIT_ROT, _RIGID_POS, _RIGID_QUAT = tc.compose_cloth_and_pile_xforms_from_usd()
_TABLE_CENTER, _TABLE_ROT, _TABLE_SCALE = tc.compose_table_xform_from_scene()

# ─────────────────────────────────────────────────────────────────────────────
# Rigid "cloth pile" (Cloth_In002) — VBD-specific tuning
# ─────────────────────────────────────────────────────────────────────────────
# Total pile mass [kg]. USD authors 0.001 kg with linearDamping = 1.0
# and maxLinearVelocity = 1.0 to keep the body stable; AVBD has no
# equivalent damping, so under the asymmetric initial cloth-rigid
# contacts a 1 g body skitters off the table. 100 g is the experimental
# middle ground: light enough for the cloth's contact pipeline to
# actually slow the pile (instead of the pile blowing through the sheet
# under its own weight), but heavy enough that AVBD keeps the body
# stable without PhysX-style velocity damping. The hand-computed
# inertia tensor below scales linearly with this constant.
_PILE_MASS = 0.1
_PILE_MU = 0.95  # USD material's static/dynamic friction
# Single-hull approximation lives inside the cloth's hollow. The hull is
# centroid-shrunk until ``shrink_pile_hull_clear_of_cloth`` reports no
# overlap with the cloth surface; the helper binary-searches in
# ``[_PILE_SHRINK_MIN, 1.0]`` and uses ppfcs's intersection checker for
# the test so the same hull is also a valid pinned-shell collider for the
# PPFCS variant.
_PILE_SHRINK_MIN = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# Cloth material (SI)
# ─────────────────────────────────────────────────────────────────────────────
_TRI_KE = 1.0e3
_TRI_KA = 1.0e3
_TRI_KD = 1.0e-3
# Bending stiffness paired with the flat-rest override below: every cloth
# edge has its rest dihedral re-zeroed after ``add_cloth_mesh``, so the
# folded USD mesh carries elastic bending energy that drives the cloth to
# relax open under gravity / hand contact (real tablecloth behaviour). With
# rest=0 a stiff edge_ke makes the folds explode open on frame 0; at
# ``_EDGE_KE = 1.0`` the unfolding was vigorous enough that the pile slipped
# out of the cloth's hollow before the surfaces could drape over it. The
# weaker spring + boosted damper here gives the cloth a gentle restoration
# toward flat that lingers in contact with the pile and the hands instead
# of snapping past them.
_EDGE_KE = 0.3
_EDGE_KD = 1.0e-1
# Total cloth mass [kg]. The areal density passed to ``add_cloth_mesh``
# is derived from this at runtime as ``mass / Σ triangle_area`` — the
# folded mesh's triangle-area sum is ~0.54 m², so 0.27 kg gives the same
# 0.5 kg/m² areal density we used to hard-code. Reference: IL/PhysX
# spawned this cloth at ~45 g (240 kg/m^3 * 0.35 mm * 0.54 m^2); 270 g is
# 6x heavier and was tuned to keep the VBD solver well-conditioned
# against the kinematic G1 + dynamic pile.
_CLOTH_MASS = 0.27

# ─────────────────────────────────────────────────────────────────────────────
# Collision (SI)
# ─────────────────────────────────────────────────────────────────────────────
_PARTICLE_RADIUS = 5.0e-3  # 5 mm
# Self-contact radius/margin must stay below the rest-state inter-particle
# spacing (~6 mm for the proxy mesh) or the folded cloth's near-neighbours
# instantly bind together and the sheet refuses to fall.
_SELF_CONTACT_RADIUS = 2.0e-3
_SELF_CONTACT_MARGIN = 2.0e-3
_REST_EXCLUSION_RADIUS = 8.0e-3
_SOFT_CONTACT_MARGIN = 1.0e-2

_SOFT_CONTACT_KE = 1.0e4
_SOFT_CONTACT_KD = 1.0e-1
_SOFT_CONTACT_MU = 0.5

_SHAPE_KE = 1.0e5
_SHAPE_KD = 1.0e0
_SHAPE_MU = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────
# The recorded HDF5 demo uses a 30 Hz control rate (Isaac Lab sim_dt = 1/120 s
# with decimation = 4). We match that so one frame here corresponds to one
# recorded action. Internal substep size is preserved (40 substeps x 1/(30*40)
# = 0.833 ms = same as the previous 20 x 1/(60*20) setup).
_FPS = 30
_SIM_SUBSTEPS = 40
_VBD_ITERS = 10


class Example:
    """Drop a tablecloth onto a table next to a kinematic Unitree G1.

    The G1 is fixed-base and its body inverse masses are zeroed so the
    initial joint pose is held by the VBD coupling. The cloth particles
    are integrated by :class:`newton.solvers.SolverVBD` and contact the
    table box and ground plane through the soft-contact pipeline.
    """

    def __init__(
        self,
        viewer,
        save_mp4: str | None = None,
        capture_replay: bool = False,
        capture_frames: int = 300,
        capture_fps: int = _FPS,
        capture_dir: str = "outputs/replay_capture",
        capture_format: str = "mp4",
        no_pile: bool = False,
        show_record: bool = False,
    ):
        self.viewer = viewer
        self.fps = _FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = _SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self._frame_count = 0
        self._has_pile = not no_pile

        # Replay state ─ populated by ``_load_replay()`` below.
        # Joint motion comes from joint_position: the PD-tracked physical
        # state Isaac Lab rendered (read straight from PhysX in IL).
        self._replay_joint_q = None  # (T, 53)  G1 joint positions per frame (PhysX order)
        self._replay_cloth_pos = None  # (T, 2523, 3)  PhysX cloth particle world positions
        self._replay_total_frames = 0
        self._replay_frame = 0  # latest HDF5 frame index applied to joint_q
        self._replay_started = False  # set True once frame 1's step() runs
        # Last metric values for the HUD.
        self._cloth_delta_rms_mm = float("nan")
        self._cloth_delta_max_mm = float("nan")
        self._cloth_delta_mean_mm = float("nan")
        # UI toggle for the recorded-cloth overlay.
        # UI toggle for the recorded-cloth overlay. Defaults to off; users
        # opt in with ``--show-record`` and can also flip the
        # state during the run via the "Show record" checkbox in the HUD.
        self._show_record = bool(show_record)
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

        builder = newton.ModelBuilder()  # Z-up, gravity = (0, 0, -9.81)

        # ── G1 robot (fixed base, kinematic) ────────────────────────────────
        # Pelvis at world (-0.95, 0, 0.80) — matches the HDF5 recording. With
        # the default identity xform, the G1's feet would be below the ground.
        builder.add_usd(
            newton.examples.get_asset(tc.G1_USD_REL),
            xform=wp.transform(tc.G1_BASE_POS, tc.G1_BASE_ROT),
            floating=False,
        )
        self._robot_body_count = builder.body_count

        # Newton's USD parser builds the 65 G1 bodies + 53 revolute / 12
        # fixed joints from the URDF-style schemas correctly, but the
        # collision *meshes* under ``<link>/collisions`` are skipped — the
        # parser emits them as a generic GPrim subtree it doesn't follow.
        # We walk those meshes ourselves and register one collider per
        # link where the USD authored geometry. The G1 ships
        # ``physics:approximation="convexHull"`` on every populated
        # ``<link>/collisions`` Xform; 24 of the 65 links carry collision
        # geometry (the L_/R_ finger chains, including thumb base). Links
        # without collision in the USD (legs, torso, head, upper arms,
        # hand_base) get no collider, matching the USD's authored intent.
        g1_stage = Usd.Stage.Open(newton.examples.get_asset(tc.G1_USD_REL))
        robot_col_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_SHAPE_KE,
            kd=_SHAPE_KD,
            mu=_SHAPE_MU,
            # COLLIDE_SHAPES enables the viewer's "Show Collision" toggle
            # to render these as collision wireframes (the toggle keys
            # off COLLIDE_SHAPES, not COLLIDE_PARTICLES). The robot is
            # kinematic and IL env_cfg disables self-collisions, so we
            # don't want these to actually participate in shape-shape
            # contact — ``collision_group=0`` short-circuits every pair
            # involving this shape in the broad phase (see
            # ``ModelBuilder._test_group_pair``: any pair with a 0-group
            # endpoint returns False). The COLLIDE_SHAPES flag remains
            # set so the viewer still renders the hull wireframe under
            # "Show Collision".
            has_shape_collision=True,
            has_particle_collision=True,
            collision_group=0,
            # Visual shapes from ``add_usd`` already render the robot;
            # leave the collider invisible so "Show Visual" stays clean.
            is_visible=False,
            # The link's mass/inertia is authored on the body itself via
            # ``PhysicsMassAPI``; the collider should not contribute.
            density=0.0,
        )
        n_robot_colliders = 0
        for b in range(self._robot_body_count):
            link_path = builder.body_label[b]
            geom = tc.collect_link_meshes_in_link_local(g1_stage, link_path, subtree="collisions")
            if geom is None:
                continue
            V_link, F_link = geom
            approx = tc.read_link_collision_approximation(g1_stage, link_path)
            mesh = newton.Mesh(V_link, F_link.flatten(), compute_inertia=False)
            if approx == "convexHull":
                builder.add_shape_convex_hull(body=b, mesh=mesh, cfg=robot_col_cfg)
            else:
                builder.add_shape_mesh(body=b, mesh=mesh, cfg=robot_col_cfg)
            n_robot_colliders += 1
        print(
            f"[table_cloth_vbd] Added {n_robot_colliders} G1 link colliders from USD "
            f"(``physics:approximation`` + ``<link>/collisions`` meshes)"
        )

        # Build the IL-index → Newton-qstart permutation by name-matching the
        # builder's joint labels. ``self._il_to_newton_qs[il_idx]`` is the
        # Newton joint_q coord index for the IL-named joint, or None if the
        # name didn't match (shouldn't happen with the canonical G1 USD).
        self._il_to_newton_qs = tc.build_il_to_newton_qs(builder)
        _missing = [tc.IL_JOINT_NAMES[i] for i, qs in enumerate(self._il_to_newton_qs) if qs is None]
        if _missing:
            print(f"[table_cloth_vbd] WARNING: IL joints not found in Newton model: {_missing}")

        # ── Cloth from assets/cloth/Cloth_fold10.usd ────────────────────────
        # The cloth USD also carries the rigid pile (Cloth_In002), so we open
        # The cloth USD also carries the rigid pile (Cloth_In002), so we
        # open the stage here even if the pile is disabled below.
        cloth_stage = Usd.Stage.Open(newton.examples.get_asset(tc.CLOTH_USD_REL))
        self._cloth_particle_start = builder.particle_count
        cloth_prim = cloth_stage.GetPrimAtPath(tc.CLOTH_PRIM_PATH)
        if not cloth_prim:
            raise RuntimeError(f"Cloth prim not found at {tc.CLOTH_PRIM_PATH}")
        cloth_mesh = newton.usd.get_mesh(cloth_prim)
        vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in cloth_mesh.vertices]
        indices = list(map(int, cloth_mesh.indices))
        self._cloth_indices_np = np.array(indices, dtype=np.int32)
        cloth_edge_start = len(builder.edge_rest_angle)
        # Areal density (kg/m^2) derived from the target total mass and the
        # folded mesh's triangle-area sum. ``add_cloth_mesh`` distributes
        # ``density * area`` to each particle (1/3 per triangle corner),
        # so the cloth's total mass is exactly ``density * sum(area)``.
        v_arr = np.asarray(cloth_mesh.vertices, dtype=np.float64).reshape(-1, 3)
        f_arr = self._cloth_indices_np.reshape(-1, 3)
        v0 = v_arr[f_arr[:, 0]]
        v1 = v_arr[f_arr[:, 1]]
        v2 = v_arr[f_arr[:, 2]]
        cloth_area = 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
        cloth_density = _CLOTH_MASS / cloth_area
        builder.add_cloth_mesh(
            pos=_CLOTH_INIT_POS,
            rot=_CLOTH_INIT_ROT,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=cloth_density,
            tri_ke=_TRI_KE,
            tri_ka=_TRI_KA,
            tri_kd=_TRI_KD,
            edge_ke=_EDGE_KE,
            edge_kd=_EDGE_KD,
            particle_radius=_PARTICLE_RADIUS,
        )
        print(
            f"[table_cloth_vbd] Cloth mass {_CLOTH_MASS * 1000:.1f} g "
            f"(area {cloth_area:.3f} m^2, density {cloth_density:.3f} kg/m^2)"
        )
        # Override the per-edge bending rest angles to 0 (flat). By
        # default ``add_cloth_mesh`` reads the rest dihedral from the
        # input geometry, so the folded USD mesh would have *zero*
        # stored bending energy — the cloth would not want to unfold
        # at all. A real tablecloth's natural rest is flat; using 0
        # everywhere gives the folded shape the elastic energy it
        # needs to relax open under gravity and hand contact, mirroring
        # the ``bend-rest-from-geometry`` removal in the PPFCS variant.
        for e in range(cloth_edge_start, len(builder.edge_rest_angle)):
            builder.edge_rest_angle[e] = 0.0
        self._cloth_particle_end = builder.particle_count

        # ── Rigid "cloth pile" (Cloth_In002) ────────────────────────────────
        # Dynamic rigid body living *inside* the deformable cloth at the
        # USD-authored relative offset. Collision is a *single* convex
        # hull centroid-shrunk until it fits in the cloth's hollow with
        # zero initial overlap. The shrink is computed by
        # ``tc.shrink_pile_hull_clear_of_cloth`` using ppfcs's own
        # intersection checker, so the same hull is also acceptable to
        # the PPFCS variant.
        self._rigid_body_idx = None
        self._rigid_col_shape_idx = -1
        if self._has_pile:
            rigid_col_prim = cloth_stage.GetPrimAtPath(tc.RIGID_COL_PRIM_PATH)
            rigid_vis_prim = cloth_stage.GetPrimAtPath(tc.RIGID_VIS_PRIM_PATH)
            if not rigid_col_prim or not rigid_vis_prim:
                raise RuntimeError("Rigid Cloth_In002 prims not found")
            rigid_col_mesh = newton.usd.get_mesh(rigid_col_prim)
            rigid_vis_mesh = newton.usd.get_mesh(rigid_vis_prim)

            # Diagonal inertia for a ~0.24 x 0.14 x 0.10 m box of the chosen mass.
            I = _PILE_MASS / 12.0
            rigid_inertia = [
                [I * (0.14**2 + 0.10**2), 0.0, 0.0],
                [0.0, I * (0.24**2 + 0.10**2), 0.0],
                [0.0, 0.0, I * (0.24**2 + 0.14**2)],
            ]
            self._rigid_body_idx = builder.add_body(
                xform=wp.transform(_RIGID_POS, _RIGID_QUAT),
                mass=_PILE_MASS,
                inertia=rigid_inertia,
                lock_inertia=True,
            )

            # Compose the pile's world-space vert cloud and the cloth's
            # world-space vert cloud, then ask the helper to find the
            # largest centroid-shrink that keeps the hull clear of the
            # cloth surface. The shrunken verts are then converted back
            # into the rigid body's local frame and shipped to
            # ``add_shape_convex_hull`` — Newton computes the actual hull
            # (capped at 64 verts via ``newton.Mesh.MAX_HULL_VERTICES``).
            R_pile = np.array(
                [[1 - 2 * (_RIGID_QUAT[1] ** 2 + _RIGID_QUAT[2] ** 2),
                  2 * (_RIGID_QUAT[0] * _RIGID_QUAT[1] - _RIGID_QUAT[2] * _RIGID_QUAT[3]),
                  2 * (_RIGID_QUAT[0] * _RIGID_QUAT[2] + _RIGID_QUAT[1] * _RIGID_QUAT[3])],
                 [2 * (_RIGID_QUAT[0] * _RIGID_QUAT[1] + _RIGID_QUAT[2] * _RIGID_QUAT[3]),
                  1 - 2 * (_RIGID_QUAT[0] ** 2 + _RIGID_QUAT[2] ** 2),
                  2 * (_RIGID_QUAT[1] * _RIGID_QUAT[2] - _RIGID_QUAT[0] * _RIGID_QUAT[3])],
                 [2 * (_RIGID_QUAT[0] * _RIGID_QUAT[2] - _RIGID_QUAT[1] * _RIGID_QUAT[3]),
                  2 * (_RIGID_QUAT[1] * _RIGID_QUAT[2] + _RIGID_QUAT[0] * _RIGID_QUAT[3]),
                  1 - 2 * (_RIGID_QUAT[0] ** 2 + _RIGID_QUAT[1] ** 2)]],
                dtype=np.float64,
            )
            T_pile = np.array(
                [float(_RIGID_POS[0]), float(_RIGID_POS[1]), float(_RIGID_POS[2])],
                dtype=np.float64,
            )
            V_pile_world = (
                np.asarray(rigid_col_mesh.vertices, dtype=np.float64).reshape(-1, 3)
                @ R_pile.T
                + T_pile
            )
            R_cloth = np.array(
                [[1 - 2 * (_CLOTH_INIT_ROT[1] ** 2 + _CLOTH_INIT_ROT[2] ** 2),
                  2 * (_CLOTH_INIT_ROT[0] * _CLOTH_INIT_ROT[1] - _CLOTH_INIT_ROT[2] * _CLOTH_INIT_ROT[3]),
                  2 * (_CLOTH_INIT_ROT[0] * _CLOTH_INIT_ROT[2] + _CLOTH_INIT_ROT[1] * _CLOTH_INIT_ROT[3])],
                 [2 * (_CLOTH_INIT_ROT[0] * _CLOTH_INIT_ROT[1] + _CLOTH_INIT_ROT[2] * _CLOTH_INIT_ROT[3]),
                  1 - 2 * (_CLOTH_INIT_ROT[0] ** 2 + _CLOTH_INIT_ROT[2] ** 2),
                  2 * (_CLOTH_INIT_ROT[1] * _CLOTH_INIT_ROT[2] - _CLOTH_INIT_ROT[0] * _CLOTH_INIT_ROT[3])],
                 [2 * (_CLOTH_INIT_ROT[0] * _CLOTH_INIT_ROT[2] - _CLOTH_INIT_ROT[1] * _CLOTH_INIT_ROT[3]),
                  2 * (_CLOTH_INIT_ROT[1] * _CLOTH_INIT_ROT[2] + _CLOTH_INIT_ROT[0] * _CLOTH_INIT_ROT[3]),
                  1 - 2 * (_CLOTH_INIT_ROT[0] ** 2 + _CLOTH_INIT_ROT[1] ** 2)]],
                dtype=np.float64,
            )
            T_cloth = np.array(
                [float(_CLOTH_INIT_POS[0]), float(_CLOTH_INIT_POS[1]), float(_CLOTH_INIT_POS[2])],
                dtype=np.float64,
            )
            V_cloth_world = (
                np.asarray(vertices, dtype=np.float64).reshape(-1, 3) @ R_cloth.T + T_cloth
            )
            cloth_F_np = self._cloth_indices_np.reshape(-1, 3)
            # Default to the SciPy half-space inside-hull test (the
            # helper's no-ppfcs path) — orders of magnitude faster
            # than ppf-contact-solver's tri-tri checker, and the
            # cloth has 2.5 k verts so it's a dense enough sample
            # of the surface. VBD penalty-resolves any sub-mm
            # residual overlap that the conservative test misses.
            V_pile_world_shrunk, _shrink_s = tc.shrink_pile_hull_clear_of_cloth(
                V_pile_world,
                V_cloth_world,
                cloth_F_np,
                s_min=_PILE_SHRINK_MIN,
                ppfcs_dir=None,
            )
            print(f"[table_cloth_vbd] Pile hull shrunk to s={_shrink_s:.3f} of original size")
            # Back to body-local for ``add_shape_convex_hull``.
            V_pile_body = (V_pile_world_shrunk - T_pile) @ R_pile

            # Compact the shrunken point cloud down to the hull-extremal
            # vertices only. Newton's CONVEX_MESH support function
            # iterates EVERY vertex in ``mesh.vertices`` on each GJK
            # support call — shipping the full ~25 k shrunken cloud as
            # the input mesh means O(25 k) per support call, which
            # multiplies out to ~10x per-step slowdown in the narrow
            # phase. Compacting to the hull verts (a few hundred max)
            # restores fps without changing the resulting hull.
            from scipy.spatial import ConvexHull as _ConvexHull  # noqa: PLC0415

            _h = _ConvexHull(V_pile_body)
            _used = np.unique(_h.simplices.flatten())
            _remap = -np.ones(V_pile_body.shape[0], dtype=np.int64)
            _remap[_used] = np.arange(len(_used))
            V_pile_body_hull = V_pile_body[_used]
            F_pile_body_hull = _remap[_h.simplices].astype(np.int32)
            print(
                f"[table_cloth_vbd] Pile hull compacted to {len(V_pile_body_hull)} verts "
                f"(from {len(V_pile_body)} input points)"
            )
            rigid_col_cfg = newton.ModelBuilder.ShapeConfig(
                ke=_SHAPE_KE,
                kd=_SHAPE_KD,
                mu=_PILE_MU,
                density=0.0,
                is_visible=False,
            )
            self._rigid_col_shape_idx = builder.add_shape_convex_hull(
                body=self._rigid_body_idx,
                mesh=newton.Mesh(
                    V_pile_body_hull,
                    F_pile_body_hull.flatten(),
                    compute_inertia=False,
                ),
                cfg=rigid_col_cfg,
            )

            rigid_vis_cfg = newton.ModelBuilder.ShapeConfig(
                has_shape_collision=False,
                has_particle_collision=False,
                density=0.0,
            )
            builder.add_shape_mesh(
                body=self._rigid_body_idx,
                mesh=rigid_vis_mesh,
                cfg=rigid_vis_cfg,
            )

        # ── Table collision: 29 pre-authored convex hulls ───────────────────
        # Table256.usd authors one rigid body whose collider is an offline
        # convex decomposition baked into 29 separate Mesh prims, each tagged
        # ``physics:approximation = convexHull``. We mirror that exactly: 29
        # sibling shapes attached to the world body (the USD's FixedJoint
        # pins the rigid to /root, so static-on-world is the same thing).
        table_stage = Usd.Stage.Open(newton.examples.get_asset(tc.TABLE_USD_REL))
        table_xform = wp.transform(_TABLE_CENTER, _TABLE_ROT)
        table_col_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_SHAPE_KE,
            kd=_SHAPE_KD,
            mu=_SHAPE_MU,
            is_visible=False,
        )
        for i in range(1, tc.TABLE_COL_COUNT + 1):
            col_prim = table_stage.GetPrimAtPath(tc.TABLE_COL_PRIM_FMT.format(i=i))
            if not col_prim:
                raise RuntimeError(f"Table collider not found at {tc.TABLE_COL_PRIM_FMT.format(i=i)}")
            builder.add_shape_convex_hull(
                body=-1,
                xform=table_xform,
                mesh=newton.usd.get_mesh(col_prim),
                scale=_TABLE_SCALE,
                cfg=table_col_cfg,
            )

        # ── Table visual mesh (Table256.usd) ────────────────────────────────
        # Render-only: both collision flags disabled so it never participates
        # in cloth/rigid contact.
        table_vis_prim = table_stage.GetPrimAtPath(tc.TABLE_VIS_PRIM_PATH)
        if not table_vis_prim:
            raise RuntimeError(f"Table visual prim not found at {tc.TABLE_VIS_PRIM_PATH}")
        table_vis_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=False,
            has_particle_collision=False,
            density=0.0,
        )
        builder.add_shape_mesh(
            body=-1,
            xform=table_xform,
            mesh=newton.usd.get_mesh(table_vis_prim),
            scale=_TABLE_SCALE,
            cfg=table_vis_cfg,
        )

        # ── Ground plane ────────────────────────────────────────────────────
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_SHAPE_KE,
            kd=_SHAPE_KD,
            mu=_SHAPE_MU,
        )
        builder.add_ground_plane(cfg=ground_cfg)

        # VBD needs color groups for both particles (cloth) and rigid bodies
        # (the pile). ``include_bending`` adds bending-edge coloring for cloth.
        builder.color(include_bending=True)

        self.model = builder.finalize()

        # ── Make the G1 bodies kinematic (rigid pile stays dynamic) ───────
        inv_m = self.model.body_inv_mass.numpy().copy()
        inv_i = self.model.body_inv_inertia.numpy().copy()
        inv_m[: self._robot_body_count] = 0.0
        inv_i[: self._robot_body_count] = 0.0
        self.model.body_inv_mass = wp.array(inv_m, dtype=float)
        self.model.body_inv_inertia = wp.array(inv_i, dtype=wp.mat33)

        # ── Particle / shape contact materials ──────────────────────────────
        self.model.soft_contact_ke = _SOFT_CONTACT_KE
        self.model.soft_contact_kd = _SOFT_CONTACT_KD
        self.model.soft_contact_mu = _SOFT_CONTACT_MU

        shape_ke = self.model.shape_material_ke.numpy().copy()
        shape_kd = self.model.shape_material_kd.numpy().copy()
        shape_mu = self.model.shape_material_mu.numpy().copy()
        shape_ke[:] = _SHAPE_KE
        shape_kd[:] = _SHAPE_KD
        shape_mu[:] = _SHAPE_MU
        self.model.shape_material_ke = wp.array(shape_ke, dtype=float)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=float)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=float)

        # ── States ──────────────────────────────────────────────────────────
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialise the robot FK once. The robot pose is then frozen because
        # body_inv_mass is zero — VBD will not integrate those bodies.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.model.body_q, self.state_0.body_q)

        # ── VBD solver ──────────────────────────────────────────────────────
        # ``integrate_with_external_rigid_solver=False`` lets AVBD inside VBD
        # integrate the dynamic cloth-pile body. The G1 bodies stay put
        # because their inverse masses were zeroed above (kinematic).
        # ``particle_rest_shape_contact_exclusion_radius`` filters out the
        # self-contact pairs that start within 8 mm in the rest state — without
        # this the folded cloth's overlapping layers freeze the sheet in place.
        #
        # Tuning for the compound (~40-hull) cloth-pile collider:
        #   - ``rigid_body_particle_contact_buffer_size`` is per-body and
        #     defaults to 256; many hulls x cloth particles can blow past
        #     that, silently dropping contacts. We raise it to 2048.
        #   - ``rigid_avbd_beta`` is the penalty ramp rate. AVBD is designed
        #     around a low ``rigid_contact_k_start`` and lets the penalty
        #     adapt. With many sibling-hull contacts all ramping
        #     independently the effective stiffness can explode; we lower
        #     beta to slow that.
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=_VBD_ITERS,
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=_SELF_CONTACT_RADIUS,
            particle_self_contact_margin=_SELF_CONTACT_MARGIN,
            particle_rest_shape_contact_exclusion_radius=_REST_EXCLUSION_RADIUS,
            particle_topological_contact_filter_threshold=1,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=1.0e5,
            rigid_avbd_beta=1.0e3,
            rigid_body_particle_contact_buffer_size=2048,
        )

        # ── Collision pipeline ──────────────────────────────────────────────
        # broad_phase="nxn" is required for particle-rigid-body contact.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=_SOFT_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        # ── Viewer ──────────────────────────────────────────────────────────
        self.viewer.set_model(self.model)
        # Camera framed on the action zone: G1 pelvis at (-0.95, 0, 0.80),
        # table top centered around (-0.50, 0, 0.886), cloth + pile drop in
        # between. We park in the +X / -Y corner and look back-left so the
        # robot is on the right, the table on the left, and the cloth-pile
        # column straight ahead.
        self.viewer.set_camera(wp.vec3(0.8, -1.0, 1.6), -20.0, 140.0)

        # Load the recorded HDF5 once we have the model+state ready.
        # ``step()`` consumes one entry per call starting at frame 1.
        self._load_replay()

        # Frame-0 visible pose: spread_tablecloth custom init (arms
        # relaxed at pitch=-0.3, roll=+-0.5, elbow=-0.5) — matches what
        # Isaac Lab spawns the G1 with before the recording starts.
        if self._il_to_newton_qs is not None:
            self._snap_g1_to_init_pose()

        # Replay slot map: jp_slot → Newton joint_q index, for the full
        # recorded 53-DoF G1 joint state.
        self._replay_slot_qs: list[tuple[int, int]] = []
        if self._replay_joint_q is not None:
            self._build_replay_qmaps()

        self.capture()

        if self.save_mp4:
            self._init_video_capture()

    def capture(self):
        if wp.get_device().is_cuda:
            # ``wp.ScopedCapture`` records *and* executes the kernel
            # launches into the graph, so simulate() inside it advances
            # state by one frame. Snapshot the relevant arrays first
            # and restore them after so frame 0 still shows the
            # USD-loaded scene (no physics step yet).
            snap_particle_q = self.state_0.particle_q.numpy().copy() if self.state_0.particle_q is not None else None
            snap_particle_qd = self.state_0.particle_qd.numpy().copy() if self.state_0.particle_qd is not None else None
            snap_body_q = self.state_0.body_q.numpy().copy() if self.state_0.body_q is not None else None
            snap_body_qd = self.state_0.body_qd.numpy().copy() if self.state_0.body_qd is not None else None
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
            if snap_particle_q is not None:
                self.state_0.particle_q.assign(wp.array(snap_particle_q, dtype=wp.vec3))
            if snap_particle_qd is not None:
                self.state_0.particle_qd.assign(wp.array(snap_particle_qd, dtype=wp.vec3))
            if snap_body_q is not None:
                self.state_0.body_q.assign(wp.array(snap_body_q, dtype=wp.transform))
            if snap_body_qd is not None:
                self.state_0.body_qd.assign(wp.array(snap_body_qd, dtype=wp.spatial_vector))
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Robot stays kinematic — its body_q never changes — but copying
            # into state_1 keeps the contact pipeline consistent.
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.capture_done:
            return

        # Frame 0 is the post-load / pre-simulation snapshot: G1 at the
        # spread_tablecloth init pose, cloth + pile at their authored USD
        # positions. No physics step, no HDF5 playback. Frame 1 onward
        # consumes ``replay_jq[frame_count - 1]`` (so display frame 1
        # corresponds to the recording's first frame).
        if self._frame_count == 0:
            self.sim_time += self.frame_dt
            self.frame_index += 1
            self._frame_count += 1
            return

        if self._replay_joint_q is not None:
            replay_idx = self._frame_count - 1
            self._replay_frame = replay_idx
            self._apply_replay_frame(replay_idx)
            self._update_record_overlay(replay_idx)
            self._replay_started = True

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._frame_count += 1

        # Per-frame cloth-tracking metric vs the recorded PhysX cloth (same
        # 2523 particles). Only meaningful once replay has started.
        if self._replay_started:
            self._compute_cloth_metric(self._replay_frame)

    def render(self):
        if self.capture_done:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Recorded-cloth overlay (Show record toggle). We always update the
        # mesh prototype + one identity-xform instance so the toggle just flips
        # ``hidden`` rather than relying on persistence between frames.
        if self._record_points_wp is not None and self._record_indices_wp is not None:
            self.viewer.log_mesh(
                "/record_cloth",
                points=self._record_points_wp,
                indices=self._record_indices_wp,
                backface_culling=False,
                hidden=not self._show_record,
            )
            self.viewer.log_instances(
                "/record_cloth_inst",
                mesh="/record_cloth",
                xforms=self._record_xforms_wp,
                scales=None,
                colors=self._record_colors_wp,
                materials=None,
                hidden=not self._show_record,
            )
        self.viewer.end_frame()
        _write_video_frame_common(self)
        self._capture_replay_frame()

    # ─────────────────────────────────────────────────────────────────────
    # Video / replay capture helpers
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

    # ─────────────────────────────────────────────────────────────────────
    # HDF5 replay + per-frame cloth-tracking metric
    # ─────────────────────────────────────────────────────────────────────

    def _load_replay(self):
        """Read the recorded G1 joint trajectory + cloth particle trajectory.

        Populates ``self._replay_joint_q``, ``self._replay_cloth_pos`` and a
        few derived warp arrays used by render/metric paths. If the file is
        missing, replay is silently disabled and the G1 stays at its
        spread_tablecloth init pose.
        """
        replay = tc.load_replay()
        if replay is None:
            print("[table_cloth_vbd] G1 will hold its initial pose (no replay).")
            return
        jq = replay["joint_position"]
        cp = replay["nodal_position"]
        self._replay_joint_q = jq
        self._replay_cloth_pos = cp
        self._replay_total_frames = replay["n_frames"]
        print(
            f"[table_cloth_vbd] Loaded {self._replay_total_frames} replay frames "
            f"(G1 dim={jq.shape[1]}, cloth nodes={cp.shape[1]})"
        )
        # Sanity check that the recorded cloth has the same particle count as
        # the proxy mesh we simulate. If not, the per-particle comparison is
        # meaningless.
        ours = self._cloth_particle_end - self._cloth_particle_start
        if ours != cp.shape[1]:
            print(
                f"[table_cloth_vbd] WARNING: our cloth has {ours} particles but the "
                f"recording has {cp.shape[1]}. Per-particle metric will be skipped."
            )
            self._replay_cloth_pos = None

        # Device-side buffer for the "Show record" overlay (one frame at
        # a time).
        if self._replay_cloth_pos is not None and self._cloth_indices_np is not None:
            n_nodes = cp.shape[1]
            self._record_points_wp = wp.zeros(n_nodes, dtype=wp.vec3)
            self._record_indices_wp = wp.array(self._cloth_indices_np, dtype=wp.int32)
            # Seed the overlay with HDF5 frame 0 so it lines up with the
            # display at frame 1 (where the first replay step is applied).
            self._record_points_wp.assign(cp[0])
            # log_instances() needs xforms/colors arrays for the single instance;
            # red-orange so the overlay reads as distinct from our cloth.
            self._record_xforms_wp = wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
                dtype=wp.transform,
            )
            self._record_colors_wp = wp.array([wp.vec3(0.95, 0.35, 0.20)], dtype=wp.vec3)
        else:
            self._record_points_wp = None
            self._record_indices_wp = None
            self._record_xforms_wp = None
            self._record_colors_wp = None

    def _build_replay_qmaps(self) -> None:
        """Build the jp_slot → Newton joint_q index list used by replay.

        At replay time we copy ``joint_position[t, jp_slot]`` to
        ``joint_q[newton_qs]`` for each pair returned here. See
        :func:`table_cloth.jp_slot_to_newton_qs` for the composition.
        """
        self._replay_slot_qs = tc.jp_slot_to_newton_qs(self._il_to_newton_qs)
        print(
            f"[table_cloth_vbd] Replay slot map: {len(self._replay_slot_qs)}/"
            f"{len(tc.JP_SLOT_TO_NAME)} jp slots mapped to Newton joint_q"
        )

    def _apply_replay_frame(self, hdf5_frame: int) -> None:
        """Drive the G1 joints from one recorded frame — full direct replay.

        For every (jp_slot, newton_qs) pair built by
        :meth:`_build_replay_qmaps`, copy ``joint_position[t, jp_slot]``
        into ``state_0.joint_q[newton_qs]``. This mirrors what Isaac
        Sim does in the minimal IL playback (``write_joint_state_to_sim``
        with the full 53-D vector) — every joint we drive comes from the
        same source IL renders from.

        All 53 recorded joints are mapped. Legs and waist move only by the
        small PD drift present in the recording, matching Isaac Lab replay.
        """
        if self._replay_joint_q is None:
            return
        idx = int(min(hdf5_frame, self._replay_total_frames - 1))
        jp_frame = self._replay_joint_q[idx]
        jq = self.state_0.joint_q.numpy().copy()
        for jp_slot, n_qs in self._replay_slot_qs:
            jq[n_qs] = float(jp_frame[jp_slot])
        self.state_0.joint_q.assign(wp.array(jq, dtype=float))

        # eval_fk would teleport the pile (free-joint) body back to its
        # FREE-joint origin coords, so save and restore its body_q.
        body_q_np = self.state_0.body_q.numpy().copy() if self._has_pile else None
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        if self._has_pile and self._rigid_body_idx is not None:
            new_bq = self.state_0.body_q.numpy().copy()
            new_bq[self._rigid_body_idx] = body_q_np[self._rigid_body_idx]
            self.state_0.body_q.assign(wp.array(new_bq, dtype=wp.transform))

    def _snap_g1_to_init_pose(self) -> None:
        """Set Newton's G1 joint_q to the spread_tablecloth custom init pose.

        Matches ``SPREAD_TABLECLOTH_CUSTOM_JOINT_POS`` from
        i4h-workflows' config/robot_config.py — arms held relaxed at
        +-0.5 roll, -0.3 pitch, -0.5 elbow. Used as the frame-0
        pose before HDF5 replay takes over at frame 1.
        """
        jq = self.state_0.joint_q.numpy().copy()
        tc.apply_init_pose(jq, self._il_to_newton_qs)
        self.state_0.joint_q.assign(wp.array(jq, dtype=float))

        body_q_np = self.state_0.body_q.numpy().copy() if self._has_pile else None
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        if self._has_pile and self._rigid_body_idx is not None:
            new_bq = self.state_0.body_q.numpy().copy()
            new_bq[self._rigid_body_idx] = body_q_np[self._rigid_body_idx]
            self.state_0.body_q.assign(wp.array(new_bq, dtype=wp.transform))

    def _update_record_overlay(self, hdf5_frame: int) -> None:
        """Copy the next recorded cloth frame into the overlay buffer."""
        if self._record_points_wp is None or self._replay_cloth_pos is None:
            return
        idx = int(min(hdf5_frame, self._replay_total_frames - 1))
        self._record_points_wp.assign(self._replay_cloth_pos[idx])

    def _compute_cloth_metric(self, hdf5_frame: int) -> None:
        """Compute per-particle L2 deviation between our cloth and the
        recorded cloth at the given HDF5 frame. Stores RMS/max/mean in mm."""
        if (
            self._replay_cloth_pos is None
            or self._cloth_particle_end == self._cloth_particle_start
        ):
            return
        idx = int(min(hdf5_frame, self._replay_total_frames - 1))
        ours = self.state_0.particle_q.numpy()[self._cloth_particle_start : self._cloth_particle_end]
        theirs = self._replay_cloth_pos[idx]
        d = np.linalg.norm(ours - theirs, axis=1)
        self._cloth_delta_rms_mm = float(np.sqrt((d * d).mean()) * 1000.0)
        self._cloth_delta_max_mm = float(d.max() * 1000.0)
        self._cloth_delta_mean_mm = float(d.mean() * 1000.0)

    # ─────────────────────────────────────────────────────────────────────
    # imgui-style HUD (only attached if the viewer supports it)
    # ─────────────────────────────────────────────────────────────────────

    def gui(self, ui):
        ui.text(f"frame: {self._frame_count}")
        ui.text(f"sim time: {self.sim_time:.2f} s")
        if self._replay_joint_q is not None:
            if self._replay_started:
                idx = min(self._replay_frame + 1, self._replay_total_frames)
                ui.text(f"replay: {idx}/{self._replay_total_frames}")
            else:
                ui.text("USD-loaded state (frame 0)")
            if not math.isnan(self._cloth_delta_rms_mm):
                ui.text(f"cloth dRMS:  {self._cloth_delta_rms_mm:.2f} mm")
                ui.text(f"cloth dMean: {self._cloth_delta_mean_mm:.2f} mm")
                ui.text(f"cloth dMax:  {self._cloth_delta_max_mm:.2f} mm")
        if self._record_points_wp is not None:
            _changed, self._show_record = ui.checkbox("Show record", self._show_record)

    def test_final(self):
        return  # temptest
        pq = self.state_0.particle_q.numpy()[self._cloth_particle_start : self._cloth_particle_end]
        qd = self.state_0.particle_qd.numpy()[self._cloth_particle_start : self._cloth_particle_end]
        # The cloth should have dropped from its release height (z = 1.20 m)
        # and settled on or near the table top (z ≈ 0.70 m).
        z_min = float(pq[:, 2].min())
        z_max = float(pq[:, 2].max())
        assert z_min > -0.05, f"Cloth particle below ground: z_min = {z_min:.3f}"
        assert z_max < float(_CLOTH_INIT_POS[2]) - 0.05, (
            f"Cloth has not fallen: z_max = {z_max:.3f} (release was {float(_CLOTH_INIT_POS[2]):.3f})"
        )
        # Cloth should not drift outside the simulated workspace.
        assert float(np.abs(pq[:, 0]).max()) < 2.0, "Cloth drifted too far in X"
        assert float(np.abs(pq[:, 1]).max()) < 1.0, "Cloth drifted too far in Y"
        # Velocities should be finite and bounded.
        assert float(np.abs(qd).max()) < 20.0, f"Cloth velocity too high: |v|max = {np.abs(qd).max():.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    _add_capture_arguments(
        parser,
        replay_help="Capture rendered frames and auto-build a replay video",
        # Default the encoded video FPS to the physics step rate so the
        # captured MP4 plays back at real time (override with --capture-fps).
        capture_fps_default=_FPS,
    )
    parser.add_argument(
        "--no-pile",
        action="store_true",
        help="Skip adding the rigid cloth-pile body and its convex-hull collider",
    )
    parser.add_argument(
        "--show-record",
        action="store_true",
        help=(
            "Render the recorded cloth (red-orange overlay) on top of our "
            "simulated cloth. Off by default; can also be toggled at runtime "
            "via the 'Show record' checkbox in the HUD."
        ),
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        save_mp4=getattr(args, "save_mp4", None),
        capture_replay=bool(args.capture_replay),
        capture_frames=int(args.capture_frames),
        capture_fps=int(args.capture_fps),
        capture_dir=str(args.capture_dir),
        capture_format=str(args.capture_format),
        no_pile=bool(args.no_pile),
        show_record=bool(args.show_record),
    )

    # HUD panel (replay frame + cloth-tracking metric + Show record toggle).
    if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(lambda ui, ex=example: ex.gui(ui), position="side")

    while viewer.is_running() and not getattr(example, "capture_done", False):
        example.step()
        example.render()

    if args.test:
        example.test_final()

    _finalize_capture(example)
