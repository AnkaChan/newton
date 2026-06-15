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
# Example Table Cloth (PPFCS) — Newton port of the PhysX spread_tablecloth
#
# Variant of :mod:`example_table_cloth_vbd` that swaps the cloth solver for
# the ppf-contact-solver (PPFCS) frontend. The physics layout is:
#
#   * Cloth (Cloth_fold10.usd / Cloth_fold06):  dynamic PPFCS tri-shell (soft).
#   * Pile  (Cloth_fold10.usd / Cloth_In002):    single convex hull,
#                                                centroid-shrunk by
#                                                ``tc.shrink_pile_hull_clear_of_cloth``
#                                                so it sits inside the cloth's
#                                                hollow with no IPC overlap at
#                                                frame 0. The combination of
#                                                flat-rest cloth + pile inside
#                                                the fold can drive the solver
#                                                to "failed to advance" once
#                                                the cloth surfaces sweep
#                                                through the pile's IPC gap;
#                                                kept enabled to support
#                                                material-tuning experiments.
#   * Table (Table256.usd):                      pinned PPFCS tri-shell —
#                                                full visual mesh registered
#                                                as-is. The mesh has ~5 k
#                                                authoring self-intersections
#                                                inside the legs, but PPFCS
#                                                skips collider-collider
#                                                intersection pairs at scene
#                                                build, so no cleanup is
#                                                needed for a fully-pinned
#                                                object.
#   * G1 finger links (24 of 65):                one pinned PPFCS tri-shell
#                                                per link, built from the
#                                                convex hull of the USD's
#                                                authored ``<link>/collisions``
#                                                mesh. Every populated
#                                                ``collisions`` Xform carries
#                                                ``physics:approximation =
#                                                "convexHull"`` in the G1 USD;
#                                                we honor that. The 41 other
#                                                links have either no
#                                                collision schema or an empty
#                                                collisions Xform and get no
#                                                PPFCS collider. Per-frame
#                                                kinematics still come from
#                                                ``PinHolder.transform_keyframes``
#                                                fed by the HDF5 joint trajectory.
#
# The Newton-side viewer renders the full G1 articulation and the original
# table mesh (for visual fidelity); the cloth renders from the PPFCS
# solver's per-frame vertex output rather than a Newton simulation. PPFCS
# works in Y-up internally, so the scene gets a Z↔Y swap on the way in and
# back on the way out.
#
# This example is paired with :mod:`example_table_cloth_vbd`. The shared
# constants, joint-name tables and USD xform composition live in
# :mod:`newton.examples.bag.table_cloth`; neither example imports the other.
#
# Command: python -m newton.examples table_cloth_ppfcs
#
###########################################################################

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Switch pyglet to its headless EGL backend BEFORE any newton/pyglet imports
# when:
#   1. ``--headless`` was passed explicitly.
#   2. No DISPLAY env var on non-Windows platforms.
#   3. ``--capture-replay`` was passed on non-Windows platforms.
# Same gating as :mod:`example_kfc_bag_lift_ppfcs` — capture replay is
# offscreen-by-design on Linux and many display setups crash in pyglet's xlib
# backend.
_AUTO_HEADLESS_SUPPORTED = os.name != "nt"
_HEADLESS_REQUESTED = (
    "--headless" in sys.argv
    or (_AUTO_HEADLESS_SUPPORTED and not os.environ.get("DISPLAY"))
    or (_AUTO_HEADLESS_SUPPORTED and "--capture-replay" in sys.argv)
)
if _HEADLESS_REQUESTED:
    import pyglet

    pyglet.options["headless"] = True
    if "--headless" not in sys.argv and "--no-headless" not in sys.argv:
        sys.argv.append("--headless")

import numpy as np
import warp as wp
from pxr import Usd
from scipy.spatial.transform import Rotation

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
    configure_capture as _configure_capture_common,
)
from newton.examples.bag.capture import (
    finalize_capture as _finalize_capture_common,
)
from newton.examples.bag.capture import (
    finalize_replay_video as _finalize_replay_video_common,
)

_LOG_PREFIX = "[table_cloth_ppfcs]"
_FPS = 30  # matches the HDF5 recording's control rate
_FRAME_DT = 1.0 / _FPS

# Number of PPFCS solver substeps per replay frame. Each substep advances
# the solver by FRAME_DT / SUBSTEPS_PER_FRAME. We default to 4 here so
# that ``sim_dt = (1/30) / 4 = 1/120 s`` lines up with the kfc_bag_lift_ppfcs
# example's per-step integration interval (its frame_dt is 1/60 s and it
# defaults to 2 substeps). More substeps = smaller dt = more stable
# contact resolution at the cost of wall-clock time.
_DEFAULT_PPFCS_SUBSTEPS_PER_FRAME = 4

_DEFAULT_JOB_DIR = Path("outputs/ppfcs/table_cloth")


# ─────────────────────────────────────────────────────────────────────────────
# PPFCS material parameters (Y-up, gravity -9.81 in Y)
# ─────────────────────────────────────────────────────────────────────────────
# PPFCS's ``young-mod`` is unit-normalised rather than SI; the values here
# target a disposable non-woven tablecloth: light, low-stretch, more papery
# than woven cotton, and able to keep crease memory instead of behaving like
# a fully elastic sheet.

# Cloth (the actual tablecloth — deformable shell). Non-woven fabric has
# noticeably less in-plane stretch than the previous soft-cloth tuning, but is
# still far more compliant than the paper-bag setup in
# :mod:`example_kfc_bag_lift_ppfcs`.
_CLOTH_YOUNG_MOD = 4500.0
# Lower Poisson's ratio reduces the rubber-like lateral contraction that looks
# wrong for random-fiber non-woven material.
_CLOTH_POISSON = 0.20
# Total cloth mass [kg]. PPFCS's ``density`` for tri-shells is an areal
# density in kg/m^2 (see frontend ``_param_.py``: tri density is "kg/m^2
# areal"), so per-frame mass = density * triangle area sum. We derive
# density at scene-build time from this constant divided by the folded
# mesh's triangle-area sum so the cloth's true mass is exactly
# ``_CLOTH_MASS`` no matter how the mesh changes. 90 g keeps the sheet in a
# plausible non-woven tablecloth range while avoiding the extremely light
# 45 g PhysX reference value that made earlier solver variants poorly
# conditioned against the kinematic hand colliders.
_CLOTH_MASS = 0.09  # 90 g
# Use the folded USD pose as the initial bend rest. Non-woven fabric should
# not spring all the way back to a manufactured flat sheet in a fraction of a
# second; the authored fold is treated as pre-creased material.
_CLOTH_BEND_REST_FROM_GEOMETRY = True
# Moderate elastic bending plus bend plasticity gives partial recovery:
# small bends recover, while larger folds drift the rest angle and retain a
# crease.
_CLOTH_BEND = 8.0
_CLOTH_BEND_PLASTICITY = 1.0
_CLOTH_BEND_PLASTICITY_THRESHOLD = 0.08  # rad
# Cloth friction. PPFCS defaults to ``friction-mode = min`` at the session
# level, so the lower of the two contacting surfaces wins. The table, robot,
# and pile coefficients below are raised with the cloth coefficient so the
# effective contact remains dry and grippy rather than satin-slippery.
_CLOTH_FRICTION = 0.95

# Pile (Cloth_In002 — a stiff, quasi-rigid tri-shell inside the cloth
# fold). The hull is centroid-shrunk by ``tc.shrink_pile_hull_clear_of_cloth``
# until it fits the cloth's hollow with zero IPC overlap at frame 0.
# Material params follow the kfc_bag_lift_ppfcs "interior tet body"
# tuning that stays solver-stable inside a thin shell: a low young-mod
# with a high bend keeps the hull from deforming under cloth contact.
_PILE_YOUNG_MOD = 2000.0
# Total pile mass [kg]. Like ``_CLOTH_MASS``, the PPFCS areal density is
# derived at scene-build time from this constant divided by the hull's
# surface-area sum. Keep the pile at the VBD-tuned 100 g so it remains a
# stable, quasi-rigid support inside the lighter non-woven shell.
_PILE_MASS = 0.1  # 100 g
_PILE_BEND = 5000.0
_PILE_FRICTION = 0.8
_PILE_SHRINK_MIN = 0.30

# Table (pinned tri-shell — friction affects cloth contact, while stiffness
# values would not affect pinned dynamics).
_TABLE_FRICTION = 0.85

# Robot links (pinned tri-shells — friction affects hand/cloth grip).
_ROBOT_FRICTION = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# PPFCS frontend lazy-import + runtime configuration
# ─────────────────────────────────────────────────────────────────────────────
# Duplicated from :mod:`example_kfc_bag_lift_ppfcs` rather than imported to
# keep the two PPFCS examples independent (matches the spirit of the
# table-cloth helper extraction).


def _default_ppfcs_dir() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    submodule = repo_root / "ppf-contact-solver"
    return str(submodule) if submodule.is_dir() else ""


def _ppfcs_binary_path(ppfcs_root: Path) -> Path:
    exe_name = "ppf-contact-solver.exe" if os.name == "nt" else "ppf-contact-solver"
    return ppfcs_root / "target" / "release" / exe_name


def _prepend_env_path(*paths: Path) -> None:
    entries = [str(p) for p in paths if p]
    if entries:
        os.environ["PATH"] = os.pathsep.join([*entries, os.environ.get("PATH", "")])


def _configure_ppfcs_runtime_env(ppfcs_root: Path) -> None:
    """Prepend platform-specific lib/bin dirs to PATH so the solver finds its
    CUDA backend at startup (Windows only — no-op elsewhere)."""
    if os.name != "nt":
        return
    lib_dir = ppfcs_root / "src" / "cpp" / "build" / "lib"
    backend_dll = lib_dir / "libsimbackend_cuda.dll"
    if not backend_dll.exists():
        raise FileNotFoundError(
            f"ppf-contact-solver CUDA backend DLL not found: {backend_dll}\n"
            "Build on Windows with `build-win-native\\warmup.bat /nopause`, then "
            "`build-win-native\\build.bat /nopause`."
        )
    cuda_path_env = os.environ.get("CUDA_PATH")
    cuda_path = Path(cuda_path_env) if cuda_path_env else None
    if cuda_path is None or not (cuda_path / "bin").is_dir():
        local_cuda = ppfcs_root / "build-win-native" / "cuda"
        if (local_cuda / "bin").is_dir():
            cuda_path = local_cuda
            os.environ["CUDA_PATH"] = str(cuda_path)
        else:
            raise FileNotFoundError(
                "CUDA_PATH is not set and the local ppfcs CUDA runtime was not found at "
                f"{local_cuda}.\nRun `build-win-native\\warmup.bat /nopause` inside the "
                "ppfcs repo."
            )
    _prepend_env_path(lib_dir, cuda_path / "bin")


def _require_ppfcs(ppfcs_dir: Path):
    """Import ``App`` from the ppf-contact-solver Python frontend."""
    ppfcs_root = Path(ppfcs_dir).expanduser().resolve()
    if not ppfcs_root.is_dir():
        raise FileNotFoundError(
            f"ppf-contact-solver directory not found: {ppfcs_root}\n"
            "Clone from https://github.com/st-tech/ppf-contact-solver "
            "and build it for your platform."
        )
    binary = _ppfcs_binary_path(ppfcs_root)
    if not binary.exists():
        raise FileNotFoundError(
            f"ppf-contact-solver binary not found: {binary}\n"
            "Build with `cargo build --release` on Linux or the Windows scripts."
        )
    _configure_ppfcs_runtime_env(ppfcs_root)
    if str(ppfcs_root) not in sys.path:
        sys.path.insert(0, str(ppfcs_root))
    try:
        from frontend import App  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"Failed to import ppf-contact-solver frontend from {ppfcs_root}. Original error: {exc}"
        ) from exc
    return App


# ─────────────────────────────────────────────────────────────────────────────
# Z-up ↔ Y-up coordinate conversion (PPFCS is Y-up, gravity -Y)
# ─────────────────────────────────────────────────────────────────────────────

# Permutation P swaps Y and Z components. P is symmetric (P = P^T = P^-1).
_P_ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _zup_to_yup_pts(arr: np.ndarray) -> np.ndarray:
    """``[x, y, z]_zup → [x, z, y]_yup`` for an ``(N, 3)`` vertex array."""
    return arr[:, [0, 2, 1]]


def _yup_to_zup_pts(arr: np.ndarray) -> np.ndarray:
    """``[x, y, z]_yup → [x, z, y]_zup`` (same Y↔Z swap as :func:`_zup_to_yup_pts`)."""
    return arr[:, [0, 2, 1]]


def _quat_xyzw_zup_to_wxyz_yup(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert a Warp/Newton XYZW quaternion in the Z-up frame to a PPFCS
    WXYZ quaternion in the Y-up frame."""
    R_zup = Rotation.from_quat(q_xyzw).as_matrix()
    R_yup = _P_ZUP_TO_YUP @ R_zup @ _P_ZUP_TO_YUP
    q_yup_xyzw = Rotation.from_matrix(R_yup).as_quat()
    return np.array([q_yup_xyzw[3], q_yup_xyzw[0], q_yup_xyzw[1], q_yup_xyzw[2]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Convex-hull helper
# ─────────────────────────────────────────────────────────────────────────────


def _convex_hull_mesh(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(V_hull, F_hull)`` for the 3-D convex hull of a point cloud,
    with every triangle wound so its outward normal points away from the
    hull centroid.

    The G1 USD authors collision shapes as raw triangle meshes tagged
    ``physics:approximation="convexHull"``. PPFCS doesn't compute hulls
    itself, so we run SciPy's quickhull and ship the resulting watertight
    triangulation as a pinned tri-shell. SciPy's ``ConvexHull.simplices``
    does not guarantee a consistent outward winding; PPFCS's IPC barrier
    keys off the per-triangle normal direction, so a few inward-facing
    triangles flip the "inside / outside" interpretation for the
    affected verts and send the cloth flying through the body instead
    of around it. We re-orient every triangle here by comparing the
    cross-product normal to the vector from the hull centroid to the
    triangle centroid.
    """
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    hull = ConvexHull(V)
    used = np.unique(hull.simplices.flatten())
    remap = -np.ones(V.shape[0], dtype=np.int64)
    remap[used] = np.arange(len(used))
    V_out = V[used].astype(np.float64)
    F_out = remap[hull.simplices].astype(np.int32)

    centroid = V_out.mean(axis=0)
    v0 = V_out[F_out[:, 0]]
    v1 = V_out[F_out[:, 1]]
    v2 = V_out[F_out[:, 2]]
    tri_centroids = (v0 + v1 + v2) / 3.0
    normals = np.cross(v1 - v0, v2 - v0)
    outward = tri_centroids - centroid
    # Triangles whose normal points "inward" (against the outward radial)
    # need a winding flip — swap indices 1 and 2.
    flip = np.einsum("ij,ij->i", normals, outward) < 0.0
    F_out[flip] = F_out[flip][:, [0, 2, 1]]
    return V_out, F_out


def _read_cloth_world_zup() -> tuple[np.ndarray, np.ndarray]:
    """Read the cloth visual mesh and transform to Z-up world coordinates."""
    cloth_pos, cloth_rot, _pile_pos, _pile_rot = tc.compose_cloth_and_pile_xforms_from_usd()
    cloth_stage = Usd.Stage.Open(newton.examples.get_asset(tc.CLOTH_USD_REL))
    cloth_prim = cloth_stage.GetPrimAtPath(tc.CLOTH_PRIM_PATH)
    cloth_mesh = newton.usd.get_mesh(cloth_prim)
    V = _apply_pos_rot_to_pts(
        np.asarray(cloth_mesh.vertices, dtype=np.float64).reshape(-1, 3),
        cloth_pos,
        cloth_rot,
    )
    F = np.asarray(cloth_mesh.indices, dtype=np.int32).reshape(-1, 3)
    return V, F


def _read_pile_world_zup() -> tuple[np.ndarray, np.ndarray]:
    """Read the pile (Cloth_In002) collision mesh in Z-up world coords."""
    _cloth_pos, _cloth_rot, pile_pos, pile_rot = tc.compose_cloth_and_pile_xforms_from_usd()
    cloth_stage = Usd.Stage.Open(newton.examples.get_asset(tc.CLOTH_USD_REL))
    pile_prim = cloth_stage.GetPrimAtPath(tc.RIGID_COL_PRIM_PATH)
    pile_mesh = newton.usd.get_mesh(pile_prim)
    V = _apply_pos_rot_to_pts(
        np.asarray(pile_mesh.vertices, dtype=np.float64).reshape(-1, 3),
        pile_pos,
        pile_rot,
    )
    F = np.asarray(pile_mesh.indices, dtype=np.int32).reshape(-1, 3)
    return V, F


def _read_table_world_zup() -> tuple[np.ndarray, np.ndarray]:
    """Read the Table256 visual mesh, scale + place it in the Z-up world,
    and return ``(V, F)`` ready to register as a pinned PPFCS tri-shell.

    The mesh has ~5 k authoring self-intersections among the legs, but
    PPFCS skips collider-collider intersection pairs at scene build (see
    ``_intersection_.py``: a pair is filtered if both triangles are
    pinned), so the table can be registered as-is.
    """
    table_center, table_rot, table_scale = tc.compose_table_xform_from_scene()
    table_stage = Usd.Stage.Open(newton.examples.get_asset(tc.TABLE_USD_REL))
    table_prim = table_stage.GetPrimAtPath(tc.TABLE_VIS_PRIM_PATH)
    table_mesh = newton.usd.get_mesh(table_prim)
    V_local = np.asarray(table_mesh.vertices, dtype=np.float64).reshape(-1, 3) * np.asarray(
        [float(table_scale[0]), float(table_scale[1]), float(table_scale[2])]
    )
    V_world = _apply_pos_rot_to_pts(V_local, table_center, table_rot)
    F = np.asarray(table_mesh.indices, dtype=np.int32).reshape(-1, 3)
    return V_world, F


def _apply_pos_rot_to_pts(V: np.ndarray, pos: wp.vec3, rot_xyzw: wp.quat) -> np.ndarray:
    """Apply a Warp ``(pos, quat_xyzw)`` rigid transform to an ``(N, 3)``
    vertex array. Operates in Z-up world (no coordinate swap)."""
    q = np.array([float(rot_xyzw[0]), float(rot_xyzw[1]), float(rot_xyzw[2]), float(rot_xyzw[3])])
    R = Rotation.from_quat(q).as_matrix()
    t = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
    return V @ R.T + t


# ─────────────────────────────────────────────────────────────────────────────
# Streaming PPFCS frame source — pulls the cloth vertex frame from the
# running solver each replay frame. The table + robot bodies are pinned
# (no useful vertex motion) and Newton renders the real table + robot
# directly, so we don't need to stream those back.
# ─────────────────────────────────────────────────────────────────────────────


class _StreamingFrameSource:
    """Pull per-body vertex frames from a running ppfcs solver.

    Mirrors the streaming source in :mod:`example_kfc_bag_lift_ppfcs` but
    only tracks the cloth (and table, which is pinned and ignored anyway).
    Frames are returned in Z-up coordinates (PPFCS internally works Y-up;
    we swap back here so the rest of Newton stays in Z-up).
    """

    def __init__(
        self,
        fixed_session,
        body_indices: dict[str, np.ndarray],
        num_display_frames: int,
        n_sim_steps: int,
    ):
        self._session = fixed_session
        self._body_indices = body_indices
        self._num_display_frames = num_display_frames
        self._n_sim_steps = n_sim_steps
        self._cache: list[dict[str, np.ndarray]] = []
        self._reported = -1

    @property
    def num_frames(self) -> int:
        return self._num_display_frames

    def fetch(self, display_idx: int) -> dict[str, np.ndarray]:
        if display_idx < len(self._cache):
            return self._cache[display_idx]

        target_sim_idx = display_idx
        # ``vertex_frame_count`` returns the highest existing frame index, or
        # 0 when no frames have been written yet — the two cases are
        # indistinguishable. Don't trust ``done >= target_sim_idx`` for
        # ``target_sim_idx == 0``; instead, poll ``vertex(target_sim_idx)``
        # directly and re-loop while it returns None. For ``target_sim_idx >
        # 0`` the strict ``done >= target_sim_idx`` test is meaningful.
        while True:
            done = self._session.get.vertex_frame_count()
            if done != self._reported and done % max(1, self._n_sim_steps // 20) == 0:
                print(f"{_LOG_PREFIX} {done}/{self._n_sim_steps} sim frames computed...")
                self._reported = done
            result = self._session.get.vertex(target_sim_idx)
            if result is not None:
                break
            if self._session.finished():
                target_sim_idx = max(min(target_sim_idx, done), 0)
                result = self._session.get.vertex(target_sim_idx)
                break
            if not self._session.is_running():
                err_path = Path(self._session.info.path) / "error.log"
                tail = err_path.read_text().strip().splitlines()[-10:] if err_path.exists() else []
                raise RuntimeError(
                    f"ppfcs solver exited before producing frame {target_sim_idx} "
                    f"(only {done}/{self._n_sim_steps} written). "
                    "Last error.log lines:\n  " + "\n  ".join(tail)
                )
            time.sleep(0.05)
        bodies: dict[str, np.ndarray] = {}
        if result is None:
            if self._cache:
                bodies = {k: v.copy() for k, v in self._cache[-1].items()}
            else:
                for name, idx in self._body_indices.items():
                    bodies[name] = np.zeros((len(idx), 3), dtype=np.float32)
        else:
            all_verts_yup, _ = result
            for name, idx in self._body_indices.items():
                bodies[name] = _yup_to_zup_pts(all_verts_yup[idx].astype(np.float32))

        while len(self._cache) < display_idx:
            self._cache.append({k: v.copy() for k, v in bodies.items()})
        self._cache.append(bodies)
        return bodies


# ─────────────────────────────────────────────────────────────────────────────
# Scene assembly
# ─────────────────────────────────────────────────────────────────────────────


def _build_newton_render_model() -> tuple[newton.Model, list[int | None], int]:
    """Build a Newton model with just the G1 robot (+ ground + table visual)
    for rendering. The robot's bodies are made kinematic (body_inv_mass=0) so
    no integration happens; we drive ``joint_q`` from the recording.

    Returns ``(model, il_to_newton_qs, robot_body_count)``.
    """
    builder = newton.ModelBuilder()
    builder.add_usd(
        newton.examples.get_asset(tc.G1_USD_REL),
        xform=wp.transform(tc.G1_BASE_POS, tc.G1_BASE_ROT),
        floating=False,
    )
    robot_body_count = builder.body_count

    # Table visual mesh — render only, no physics. (PPFCS handles the table
    # collision; we draw it via Newton for camera consistency.)
    table_center, table_rot, table_scale = tc.compose_table_xform_from_scene()
    table_stage = Usd.Stage.Open(newton.examples.get_asset(tc.TABLE_USD_REL))
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
        xform=wp.transform(table_center, table_rot),
        mesh=newton.usd.get_mesh(table_vis_prim),
        scale=table_scale,
        cfg=table_vis_cfg,
    )

    builder.add_ground_plane()

    il_to_newton_qs = tc.build_il_to_newton_qs(builder)
    model = builder.finalize()

    # Make every robot body kinematic so no force/integration moves them.
    inv_m = model.body_inv_mass.numpy().copy()
    inv_i = model.body_inv_inertia.numpy().copy()
    inv_m[:robot_body_count] = 0.0
    inv_i[:robot_body_count] = 0.0
    model.body_inv_mass = wp.array(inv_m, dtype=float)
    model.body_inv_inertia = wp.array(inv_i, dtype=wp.mat33)

    return model, il_to_newton_qs, robot_body_count


def _compute_robot_keyframes(
    model: newton.Model,
    il_to_newton_qs: list[int | None],
    replay_jq: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Newton FK across ``n_frames`` PPFCS display frames.

    Returns ``(translations, quaternions_xyzw)`` arrays of shape
    ``(n_frames, body_count, 3)`` and ``(n_frames, body_count, 4)`` in
    Newton's Z-up world frame.

    Display-frame 0 holds the spread_tablecloth init pose — the
    "USD-loaded" snapshot before any HDF5 playback or simulation.
    Display-frame N >= 1 holds ``replay_jq[N - 1]``, so the cloth/pile
    get exactly one ``frame_dt`` of simulation between the init pose
    and the recording's first frame, mirroring the
    ``example_table_cloth_vbd`` semantics.
    """
    state = model.state()
    body_count = model.body_count
    replay_slot_qs = tc.jp_slot_to_newton_qs(il_to_newton_qs)

    T_out = np.zeros((n_frames, body_count, 3), dtype=np.float64)
    Q_out = np.zeros((n_frames, body_count, 4), dtype=np.float64)

    # Frame 0: spread_tablecloth init pose.
    jq_buf = state.joint_q.numpy().copy()
    tc.apply_init_pose(jq_buf, il_to_newton_qs)
    state.joint_q.assign(wp.array(jq_buf, dtype=float))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q = state.body_q.numpy()
    T_out[0] = body_q[:, :3]
    Q_out[0] = body_q[:, 3:7]

    # Frames 1..n_frames-1: replay_jq[t - 1].
    for t in range(1, n_frames):
        jp_frame = replay_jq[min(t - 1, replay_jq.shape[0] - 1)]
        for jp_slot, n_qs in replay_slot_qs:
            jq_buf[n_qs] = float(jp_frame[jp_slot])
        state.joint_q.assign(wp.array(jq_buf, dtype=float))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        body_q = state.body_q.numpy()
        T_out[t] = body_q[:, :3]
        Q_out[t] = body_q[:, 3:7]
    return T_out, Q_out


def _start_ppfcs_streaming(
    ppfcs_dir: Path,
    job_dir: Path,
    robot_link_paths: list[str],
    robot_T_zup: np.ndarray,
    robot_Q_xyzw_zup: np.ndarray,
    cloth_verts_zup: np.ndarray,
    cloth_faces: np.ndarray,
    pile_verts_zup: np.ndarray,
    pile_faces: np.ndarray,
    table_verts_zup: np.ndarray,
    table_faces: np.ndarray,
    n_frames: int,
    frame_dt: float,
    substeps_per_frame: int,
    has_pile: bool = True,
) -> tuple[_StreamingFrameSource, dict[str, np.ndarray]]:
    """Build + start the PPFCS scene and return the streaming source.

    All input coordinates are in Z-up Newton convention. They get swapped to
    Y-up on the way into the solver here.
    """
    App = _require_ppfcs(ppfcs_dir)
    from frontend import Utils as _PpfcsUtils  # noqa: PLC0415

    try:
        _PpfcsUtils.terminate()
    except Exception:
        pass
    time.sleep(0.3)

    n_sim_steps = max(n_frames - 1, 0)
    sim_dt = frame_dt / float(max(substeps_per_frame, 1))
    total_s = n_sim_steps * frame_dt

    print(
        f"{_LOG_PREFIX} PPFCS replay: {n_frames} frames x dt={frame_dt:.5f} s "
        f"= {total_s:.3f} s total (sim_dt={sim_dt:.5f} s, "
        f"{substeps_per_frame} substeps / frame)"
    )

    app = App.create("table_cloth", cache_dir=str(job_dir))

    # ── Assets ───────────────────────────────────────────────────────────
    # Cloth: the proxy mesh is intersection-free out of the box (2 523 verts).
    # Pile (Cloth_In002): EXPERIMENTAL. Single convex hull centroid-shrunk
    # by ``tc.shrink_pile_hull_clear_of_cloth`` so it sits inside the
    # cloth's hollow at frame 0. PPFCS has been seen to panic
    # ``backend.rs::failed to advance`` once the flat-rest cloth unfolds
    # and its surfaces sweep through the pile's IPC contact gap — keep
    # this in mind when tuning. The pile is included anyway so the user
    # can experiment with cloth/pile material values to find a stable
    # combination. To disable the pile temporarily, leave it commented
    # out below.
    # Table: full Table256 visual mesh registered as a pinned tri-shell. The
    # mesh has ~5 k authoring self-intersections in the legs, but PPFCS
    # filters intersection pairs where both triangles are colliders, so a
    # fully-pinned object passes scene validation regardless of its own
    # intra-mesh overlaps.
    cloth_V_yup = _zup_to_yup_pts(cloth_verts_zup).astype(np.float64)
    table_V_yup = _zup_to_yup_pts(table_verts_zup).astype(np.float64)

    # Derive areal densities (kg/m^2) from the target masses and each
    # shell's triangle-area sum. PPFCS's tri-shell mass model is
    # ``per-vertex_mass = (density * triangle_area) / 3`` distributed
    # over the triangle's three corners, so total mass equals exactly
    # ``density * sum(triangle area)`` — same as Newton's add_cloth_mesh.
    def _tri_area_sum(V: np.ndarray, F: np.ndarray) -> float:
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        return 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    cloth_area = _tri_area_sum(cloth_V_yup, cloth_faces.astype(np.int32))
    cloth_density = _CLOTH_MASS / cloth_area
    print(f"{_LOG_PREFIX} Table mesh: V={table_V_yup.shape[0]}, F={table_faces.shape[0]}")
    print(
        f"{_LOG_PREFIX} Cloth mass {_CLOTH_MASS * 1000:.1f} g "
        f"(area {cloth_area:.3f} m^2, density {cloth_density:.3f} kg/m^2)"
    )
    app.asset.add.tri("cloth", cloth_V_yup, cloth_faces.astype(np.int32))
    app.asset.add.tri("table", table_V_yup, table_faces.astype(np.int32))

    pile_density = 0.0  # only used if has_pile
    if has_pile:
        pile_V_zup_shrunk, pile_shrink_s = tc.shrink_pile_hull_clear_of_cloth(
            pile_verts_zup,
            cloth_verts_zup,
            cloth_faces,
            s_min=_PILE_SHRINK_MIN,
            ppfcs_dir=ppfcs_dir,
        )
        pile_hull_V_zup, pile_hull_F = _convex_hull_mesh(pile_V_zup_shrunk)
        pile_V_yup = _zup_to_yup_pts(pile_hull_V_zup).astype(np.float64)
        pile_area = _tri_area_sum(pile_V_yup, pile_hull_F.astype(np.int32))
        pile_density = _PILE_MASS / pile_area
        print(
            f"{_LOG_PREFIX} Pile  mass {_PILE_MASS * 1000:.1f} g "
            f"(area {pile_area:.3f} m^2, density {pile_density:.3f} kg/m^2)"
        )
        print(
            f"{_LOG_PREFIX} Pile hull shrunk to s={pile_shrink_s:.3f}: "
            f"V={pile_V_yup.shape[0]}, F={pile_hull_F.shape[0]}"
        )
        app.asset.add.tri("pile", pile_V_yup, pile_hull_F.astype(np.int32))

    # Robot links: read the USD's authored ``<link>/collisions`` meshes
    # and submit one convex hull per link as a pinned tri-shell. The G1
    # ships ``physics:approximation="convexHull"`` on the populated
    # ``collisions`` Xforms; 24 of the 65 links carry geometry (the L_/R_
    # finger chains, including thumb base), all the others have either
    # no collision schema or an empty collisions Xform. SciPy's
    # quickhull turns each link's collision mesh into a clean watertight
    # triangulation; the raw 540 k-vert input collapses to ~100 hull
    # verts per link.
    #
    # The hull is registered in WORLD COORDS at frame 0 (not link-local
    # coords). PPFCS uses the asset's registered vertices as the rest /
    # initial state for the scene-build intersection check and the
    # initial IPC contact pairing — verts in link-local frame would put
    # the finger hulls near the world origin (small values), which makes
    # PPFCS think the fingers are *below* the cloth and produces a wrong
    # IPC repulsion that throws the cloth upward by 15-30 cm at frame 1.
    # The transform_keyframes operation then drives the link with the
    # *relative* rotation/translation from its frame-0 pose, so the
    # frame-0 keyframe is identity and the verts stay where registered.
    g1_stage = Usd.Stage.Open(newton.examples.get_asset(tc.G1_USD_REL))
    robot_assets: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    skipped: list[int] = []
    total_v_in = 0
    total_v_out = 0
    # Frame-0 link world poses (Z-up) — needed both to seed asset verts at
    # world coords and to compose the per-frame relative transform.
    T0_zup = robot_T_zup[0]
    Q0_zup = robot_Q_xyzw_zup[0]
    R0_zup = np.array([Rotation.from_quat(q).as_matrix() for q in Q0_zup])
    for b, link_path in enumerate(robot_link_paths):
        geom = tc.collect_link_meshes_in_link_local(g1_stage, link_path, subtree="collisions")
        if geom is None:
            skipped.append(b)
            continue
        V_link_local_zup, _F_raw = geom
        total_v_in += V_link_local_zup.shape[0]
        V_hull_local_zup, F_hull = _convex_hull_mesh(V_link_local_zup)
        # Bake the link's frame-0 world pose into the hull verts so
        # the asset registration sees world coords from the start.
        V_hull_world_zup = V_hull_local_zup @ R0_zup[b].T + T0_zup[b]
        total_v_out += V_hull_world_zup.shape[0]
        V_hull_world_yup = _zup_to_yup_pts(V_hull_world_zup).astype(np.float64)
        name = f"robot_{b}"
        app.asset.add.tri(name, V_hull_world_yup, F_hull.astype(np.int32))
        robot_assets.append((b, name, V_hull_world_yup, F_hull))
    print(
        f"{_LOG_PREFIX} Registered {len(robot_assets)} robot link colliders from USD "
        f"({len(skipped)} links have no authored collision geometry)"
    )
    print(
        f"{_LOG_PREFIX} Robot collision hulls: {total_v_in:,} → "
        f"{total_v_out:,} verts (convex hull of <link>/collisions meshes)"
    )

    # ── Scene ────────────────────────────────────────────────────────────
    scene = app.scene.create("table_cloth")

    cloth_obj = scene.add("cloth")
    cloth_obj.param.set("young-mod", _CLOTH_YOUNG_MOD)
    cloth_obj.param.set("poiss-rat", _CLOTH_POISSON)
    cloth_obj.param.set("density", cloth_density)
    cloth_obj.param.set("bend-rest-from-geometry", _CLOTH_BEND_REST_FROM_GEOMETRY)
    cloth_obj.param.set("bend", _CLOTH_BEND)
    cloth_obj.param.set("bend-plasticity", _CLOTH_BEND_PLASTICITY)
    cloth_obj.param.set("bend-plasticity-threshold", _CLOTH_BEND_PLASTICITY_THRESHOLD)
    cloth_obj.param.set("friction", _CLOTH_FRICTION)
    # The folded USD mesh is the initial bend rest for this non-woven variant.
    # Bend plasticity below then lets newly introduced folds become partial
    # creases instead of fully recovering elastically.

    if has_pile:
        pile_obj = scene.add("pile")
        pile_obj.param.set("young-mod", _PILE_YOUNG_MOD)
        pile_obj.param.set("density", pile_density)
        pile_obj.param.set("bend", _PILE_BEND)
        pile_obj.param.set("friction", _PILE_FRICTION)

    table_obj = scene.add("table")
    table_obj.param.set("friction", _TABLE_FRICTION)
    table_obj.pin()  # static — every vertex pinned, no animation operations

    # Robot bodies: pin every vertex and feed full per-frame TRS keyframes.
    # The asset verts were registered at the link's frame-0 world pose
    # above, so each PPFCS keyframe needs to express the link's motion
    # *relative* to that frame-0 pose. For the link's world-frame pose at
    # time t being (R_link(t), T_link(t)) and frame-0 being (R0, T0):
    #
    #   v_world(t) = R_link(t) @ V_link_local + T_link(t)
    #              = R_link(t) @ R0^T @ (V_world_at_frame_0 - T0) + T_link(t)
    #              = R_rel(t) @ V_world_at_frame_0 + (T_link(t) - R_rel(t) @ T0)
    #
    # with R_rel(t) = R_link(t) @ R0^T. At t=0 this collapses to identity
    # rotation and zero translation, so frame 0's keyframe is a no-op and
    # the verts stay where they were registered. ``rest_translation = 0``
    # because the object's own ``_transform`` is left at identity.
    times = [t * frame_dt for t in range(n_frames)]
    segments = [{"interpolation": "LINEAR"} for _ in range(max(n_frames - 1, 0))]
    scales_const = [np.array([1.0, 1.0, 1.0]) for _ in range(n_frames)]

    for b, name, _V_world_yup, _F in robot_assets:
        obj = scene.add(name)
        obj.param.set("friction", _ROBOT_FRICTION)
        holder = obj.pin()
        T0 = T0_zup[b]
        R0 = R0_zup[b]
        translations = []
        quaternions = []
        for t in range(n_frames):
            R_link_zup = Rotation.from_quat(robot_Q_xyzw_zup[t, b]).as_matrix()
            T_link_zup = robot_T_zup[t, b]
            # Relative rotation/translation in Z-up.
            R_rel_zup = R_link_zup @ R0.T
            T_rel_zup = T_link_zup - R_rel_zup @ T0
            # Convert to Y-up via the symmetric Y↔Z permutation.
            R_rel_yup = _P_ZUP_TO_YUP @ R_rel_zup @ _P_ZUP_TO_YUP
            T_rel_yup = _P_ZUP_TO_YUP @ T_rel_zup
            q_xyzw_yup = Rotation.from_matrix(R_rel_yup).as_quat()
            translations.append(T_rel_yup)
            quaternions.append(np.array([q_xyzw_yup[3], q_xyzw_yup[0], q_xyzw_yup[1], q_xyzw_yup[2]], dtype=np.float64))
        holder.transform_keyframes(
            local_vert=_V_world_yup,
            times=times,
            translations=translations,
            quaternions=quaternions,
            scales=scales_const,
            segments=segments,
            rest_translation=np.zeros(3),
        )

    fixed_scene = scene.build()

    # Per-body vertex index slices into the global concatenated vert array.
    # We only stream the cloth back to Newton. Pinned objects (table, robot
    # links) don't show up in the name map because every one of their DOFs
    # is fixed; that's fine — Newton renders the table + robot directly.
    body_indices: dict[str, np.ndarray] = {}
    cloth_idx = fixed_scene._map_by_name.get("cloth")
    if cloth_idx is None:
        raise RuntimeError("PPFCS scene missing vertex map for 'cloth'")
    body_indices["cloth"] = np.array(cloth_idx, dtype=np.int32)
    pile_idx = fixed_scene._map_by_name.get("pile")
    if pile_idx is not None:
        body_indices["pile"] = np.array(pile_idx, dtype=np.int32)

    # ── Session ──────────────────────────────────────────────────────────
    session_name = f"session-{os.getpid()}-{int(time.time() * 1000)}"
    session = app.session.create(fixed_scene, name=session_name)
    output_fps = 1.0 / frame_dt
    session.param.set("frames", n_sim_steps)
    session.param.set("dt", sim_dt)
    session.param.set("fps", output_fps)
    session.param.set("min-newton-steps", 32)
    session.param.set("cg-max-iter", 50000)

    fixed_session = session.build()
    print(f"{_LOG_PREFIX} Starting ppfcs solver (streaming mode)...", flush=True)
    fixed_session.start(force=True, blocking=False)

    # Cleanup hooks so the solver subprocess dies with us.
    import atexit  # noqa: PLC0415
    import signal  # noqa: PLC0415

    _cleaned = {"done": False}

    def _kill_solver(*_args):
        if _cleaned["done"]:
            return
        _cleaned["done"] = True
        try:
            _PpfcsUtils.terminate()
        except Exception:
            pass

    atexit.register(_kill_solver)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev = signal.getsignal(sig)

            def _handler(signum, frame, _prev=prev):
                _kill_solver()
                if callable(_prev):
                    _prev(signum, frame)

            signal.signal(sig, _handler)
        except Exception:
            pass

    streamer = _StreamingFrameSource(
        fixed_session=fixed_session,
        body_indices=body_indices,
        num_display_frames=n_frames,
        n_sim_steps=n_sim_steps,
    )
    return streamer, body_indices


# ─────────────────────────────────────────────────────────────────────────────
# Example class
# ─────────────────────────────────────────────────────────────────────────────


class Example:
    """Drive the PhysX spread_tablecloth scene through PPFCS.

    Newton renders the G1 robot (joint_q replayed from the HDF5 recording)
    and the static table mesh. PPFCS simulates the cloth as a dynamic
    tri-shell while a table box-proxy and every G1 forearm/hand link act
    as pinned colliders. Cloth vertices come back from the streaming
    solver each frame and are rendered as a ``log_mesh`` overlay.
    """

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.description = (
            "Newton port of the PhysX spread_tablecloth scene using PPFCS for "
            "cloth dynamics and pinned tri-shells for the table + G1 links."
        )
        parser.set_defaults(num_frames=300)
        parser.add_argument(
            "--ppfcs-dir",
            type=str,
            default=_default_ppfcs_dir(),
            help=(
                "Path to the compiled ppf-contact-solver repository root. "
                "Defaults to the repo's `ppf-contact-solver` submodule when present."
            ),
        )
        parser.add_argument(
            "--job-dir",
            type=str,
            default=str(_DEFAULT_JOB_DIR),
            help="Directory used for ppfcs solver output.",
        )
        parser.add_argument(
            "--ppfcs-substeps-per-frame",
            type=int,
            default=_DEFAULT_PPFCS_SUBSTEPS_PER_FRAME,
            help="Number of ppfcs solver substeps per replay frame.",
        )
        parser.add_argument(
            "--no-pile",
            action="store_true",
            help=(
                "Skip the dynamic pile (Cloth_In002) tri-shell. With the "
                "flat-rest cloth, the unfolding sheet can push the pile "
                "through the IPC barrier and trigger a solver panic; this "
                "flag turns the pile off so you can iterate on cloth params "
                "without it."
            ),
        )
        _add_capture_arguments(
            parser,
            replay_help="Capture rendered frames and build a replay video.",
            include_save_mp4=False,
            # Default the encoded video FPS to the physics step rate so the
            # captured MP4 plays back at real time (override with --capture-fps).
            capture_fps_default=_FPS,
        )
        return parser

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = _FRAME_DT
        self.sim_time = 0.0
        self._frame_index = -1

        ppfcs_dir_str = str(args.ppfcs_dir).strip()
        if not ppfcs_dir_str:
            raise ValueError(
                "--ppfcs-dir is required when the repo's `ppf-contact-solver` "
                "submodule is absent. Set it to the ppf-contact-solver repo root."
            )
        ppfcs_dir = Path(ppfcs_dir_str)
        job_dir = Path(args.job_dir)

        # ── Load HDF5 recording ──────────────────────────────────────────
        replay = tc.load_replay()
        if replay is None:
            raise RuntimeError(
                "spread_tablecloth HDF5 recording not available; this example requires it to drive the robot."
            )
        n_replay = replay["n_frames"]
        # When ``--capture-replay`` is on, treat ``--capture-frames`` as the
        # authoritative horizon (matches the kfc_bag_lift_ppfcs example):
        # the simulation runs for exactly that many frames and the captured
        # video has the same length. Otherwise the horizon comes from
        # ``--num-frames``. Both are clamped to the recording length.
        if getattr(args, "capture_replay", False):
            requested = int(args.capture_frames)
        else:
            requested = int(args.num_frames)
        n_frames = min(requested, n_replay)
        if n_frames < 2:
            raise ValueError("frame horizon must be at least 2 for keyframe animation.")
        self.n_frames = n_frames
        self.times_s = np.arange(n_frames, dtype=np.float32) * self.frame_dt

        # ── Build Newton render model ────────────────────────────────────
        (
            self.model,
            self._il_to_newton_qs,
            self._robot_body_count,
        ) = _build_newton_render_model()
        # Cache builder.body_label for the link-mesh extraction.
        # We don't keep the builder around, but the labels live in model.body_label.
        self._robot_link_paths = [self.model.body_label[b] for b in range(self._robot_body_count)]

        self.state_0 = self.model.state()
        self.control = self.model.control()
        # The Newton-side rendered G1 is driven straight from the HDF5
        # trajectory in ``step()`` (no init-pose splice at frame 0 — see
        # ``_compute_robot_keyframes`` for why). The state will be
        # re-initialised at frame 0's ``step()`` so we don't need an
        # explicit ``apply_init_pose`` here.
        self._replay_slot_qs = tc.jp_slot_to_newton_qs(self._il_to_newton_qs)
        self._replay_joint_q = replay["joint_position"]

        # ── Pre-compute per-frame robot body keyframes (Z-up) ────────────
        print(f"{_LOG_PREFIX} Running FK across {n_frames} frames to build robot keyframes...")
        robot_T_zup, robot_Q_zup = _compute_robot_keyframes(
            self.model, self._il_to_newton_qs, replay["joint_position"], n_frames
        )

        # ── Read cloth + (optional) pile + table meshes (world Z-up) ────
        self._has_pile = not bool(getattr(args, "no_pile", False))
        cloth_V_zup, cloth_F = _read_cloth_world_zup()
        pile_V_zup, pile_F = (
            _read_pile_world_zup()
            if self._has_pile
            else (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.int32),
            )
        )
        table_V_zup, table_F = _read_table_world_zup()

        # ── Start PPFCS streaming solver ─────────────────────────────────
        self._frame_source, _ = _start_ppfcs_streaming(
            ppfcs_dir=ppfcs_dir,
            job_dir=job_dir,
            robot_link_paths=self._robot_link_paths,
            robot_T_zup=robot_T_zup,
            robot_Q_xyzw_zup=robot_Q_zup,
            cloth_verts_zup=cloth_V_zup,
            cloth_faces=cloth_F,
            pile_verts_zup=pile_V_zup,
            pile_faces=pile_F,
            table_verts_zup=table_V_zup,
            table_faces=table_F,
            n_frames=n_frames,
            frame_dt=self.frame_dt,
            substeps_per_frame=int(args.ppfcs_substeps_per_frame),
            has_pile=self._has_pile,
        )
        self._current_bodies: dict[str, np.ndarray] = {}

        # ── Pre-build wp.arrays for render meshes ────────────────────────
        self._cloth_faces_wp = wp.array(cloth_F.flatten().astype(np.int32), dtype=wp.int32)
        self._cloth_points_wp = wp.zeros(cloth_V_zup.shape[0], dtype=wp.vec3)
        self._cloth_points_wp.assign(cloth_V_zup.astype(np.float32))
        # Single-instance arrays for log_instances (identity xform, fixed colour).
        self._cloth_xforms_wp = wp.array(
            [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
            dtype=wp.transform,
        )
        self._cloth_colors_wp = wp.array([wp.vec3(0.85, 0.85, 0.95)], dtype=wp.vec3)

        # Pile render mesh — sized to match the shrunken hull that's
        # registered with PPFCS, so the per-frame body_indices['pile']
        # slice maps directly. The hull is recomputed here with the
        # same shrink helper + SciPy hull pass so the vertex/triangle
        # counts agree with the asset. Buffers fall back to None when
        # the pile isn't in the scene (the streaming source omits it).
        self._pile_faces_wp = None
        self._pile_points_wp = None
        if self._frame_source._body_indices.get("pile") is not None:
            pile_shrunk_zup, _ = tc.shrink_pile_hull_clear_of_cloth(
                pile_V_zup,
                cloth_V_zup,
                cloth_F,
                s_min=_PILE_SHRINK_MIN,
                ppfcs_dir=ppfcs_dir,
            )
            _ph_V, _ph_F = _convex_hull_mesh(pile_shrunk_zup)
            self._pile_faces_wp = wp.array(_ph_F.flatten().astype(np.int32), dtype=wp.int32)
            self._pile_points_wp = wp.zeros(_ph_V.shape[0], dtype=wp.vec3)
            self._pile_points_wp.assign(_ph_V.astype(np.float32))
            self._pile_xforms_wp = wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
                dtype=wp.transform,
            )
            self._pile_colors_wp = wp.array([wp.vec3(0.95, 0.35, 0.20)], dtype=wp.vec3)

        # ── Capture (PNG / MP4) plumbing ─────────────────────────────────
        _configure_capture_common(
            self,
            capture_replay=bool(getattr(args, "capture_replay", False)),
            capture_frames=n_frames,
            capture_fps=int(getattr(args, "capture_fps", _FPS)),
            capture_dir=str(getattr(args, "capture_dir", "outputs/replay_capture")),
            capture_format=str(getattr(args, "capture_format", "mp4")),
            capture_background_writes=False,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.8, -1.0, 1.6), -20.0, 140.0)

        # Suppress the viewer's built-in particle/cloth draw so we control the
        # cloth mesh rendering ourselves via log_mesh.
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False

    def step(self) -> None:
        """Advance to the next replay frame. Blocks until the PPFCS solver
        has produced the corresponding sim frame."""
        if self._frame_index < self.n_frames - 1:
            self._frame_index += 1
        else:
            self._frame_index = self.n_frames - 1
        self.sim_time = float(self.times_s[self._frame_index])

        # Pull this frame's cloth + pile vertex positions (blocks until ready).
        self._current_bodies = self._frame_source.fetch(self._frame_index)
        if "cloth" in self._current_bodies:
            self._cloth_points_wp.assign(self._current_bodies["cloth"].astype(np.float32))
        if self._pile_points_wp is not None and "pile" in self._current_bodies:
            self._pile_points_wp.assign(self._current_bodies["pile"].astype(np.float32))

        # Drive the rendered Newton G1 to match the keyframed PPFCS robot.
        # Frame 0 = spread_tablecloth init pose; frame N >= 1 = HDF5 entry
        # ``replay_jq[N - 1]``. Mirrors ``_compute_robot_keyframes`` so the
        # rendered robot stays in lockstep with the pinned-shell colliders.
        jq = self.state_0.joint_q.numpy().copy()
        if self._frame_index == 0:
            tc.apply_init_pose(jq, self._il_to_newton_qs)
        else:
            jq_record = self._replay_joint_q[min(self._frame_index - 1, self._replay_joint_q.shape[0] - 1)]
            for jp_slot, n_qs in self._replay_slot_qs:
                jq[n_qs] = float(jq_record[jp_slot])
        self.state_0.joint_q.assign(wp.array(jq, dtype=float))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/cloth",
            points=self._cloth_points_wp,
            indices=self._cloth_faces_wp,
            backface_culling=False,
        )
        self.viewer.log_instances(
            "/cloth_inst",
            mesh="/cloth",
            xforms=self._cloth_xforms_wp,
            scales=None,
            colors=self._cloth_colors_wp,
            materials=None,
        )
        if self._pile_points_wp is not None and self._pile_faces_wp is not None:
            self.viewer.log_mesh(
                "/pile",
                points=self._pile_points_wp,
                indices=self._pile_faces_wp,
                backface_culling=False,
            )
            self.viewer.log_instances(
                "/pile_inst",
                mesh="/pile",
                xforms=self._pile_xforms_wp,
                scales=None,
                colors=self._pile_colors_wp,
                materials=None,
            )
        self.viewer.end_frame()
        self._capture_frame()
        if self._frame_index >= self.n_frames - 1:
            if not getattr(self, "capture_replay", False) or self.capture_done:
                if hasattr(self.viewer, "close"):
                    self.viewer.close()

    def _capture_frame(self) -> None:
        if self._frame_index < 0:
            return
        _capture_replay_frame_common(
            self,
            frame_key=self._frame_index,
            target_frame_count=self.n_frames,
            close_viewer=False,
        )

    def _finalize_video(self) -> None:
        _finalize_replay_video_common(self)

    def cleanup(self) -> None:
        _finalize_capture_common(self)

    def test_final(self) -> None:
        # Basic smoke check: cloth verts finite, not collapsed.
        if not self._current_bodies:
            return
        cloth = self._current_bodies.get("cloth")
        if cloth is None:
            return
        assert np.isfinite(cloth).all(), "cloth vertices contain non-finite values"
        bbox = cloth.max(axis=0) - cloth.min(axis=0)
        assert float(bbox.max()) > 0.05, f"cloth bounding box collapsed: {bbox} (something pinned the cloth?)"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        while viewer.is_running() and not getattr(example, "capture_done", False):
            example.step()
            example.render()
            if example._frame_index >= example.n_frames - 1:
                break
        if args.test:
            example.test_final()
    finally:
        example.cleanup()
