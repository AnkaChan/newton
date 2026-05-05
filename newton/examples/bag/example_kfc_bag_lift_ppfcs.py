# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example KFC Bag Lift via ppf-contact-solver
#
# This example simulates the KFC bag being gripped and lifted using
# ppf-contact-solver (https://github.com/st-tech/ppf-contact-solver) as
# the contact and deformation backend instead of LS-DYNA.
#
# ppf-contact-solver uses Projective Dynamics with Persistent Friction (PPF)
# — a GPU-accelerated, guaranteed penetration-free contact solver.  The bag
# is a deformable triangular shell; two invisible half-space walls represent
# the finger pads and animate through open → close → hold → lift phases.
#
# Flow:
#   1. Load and optionally decimate the KFC bag mesh.
#   2. Build a ppf-contact-solver scene: deformable bag shell, two animated
#      invisible gripping walls, and a static floor wall.
#   3. Run the simulation and collect per-frame vertex positions.
#   4. Replay the bag deformation inside the Newton viewer alongside
#      analytical pad-box visualisations.
#
# Prerequisites:
#   - Build ppf-contact-solver from source:
#       Linux:   cargo build --release
#       Windows: build-win-native\warmup.bat, then build-win-native\build.bat
#   - Set the PPFCS_DIR environment variable or pass --ppfcs-dir.
#
# Runtime knobs:
#   - `--target-faces`     Approximate shell-face count fed to ppfcs.
#   - `--proxy-mode`       Shared bag proxy generation mode.
#   - `--capture-frames`   Limit replay horizon and output frame count.
#   - `--ppfcs-dir`        Path to the compiled ppf-contact-solver repo.
#   - `--ppfcs-substeps-per-frame`
#                          Number of ppfcs solver substeps per replay frame.
#   - `--closed-width-cm`  Final finger-pad gap [cm].
#   - `--job-dir`          Directory for ppfcs output.
#
# Command: python -m newton.examples.bag.example_kfc_bag_lift_ppfcs
###########################################################################

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Switch pyglet to its headless EGL backend BEFORE any newton/pyglet imports
# when:
#   1. `--headless` was passed explicitly.
#   2. No DISPLAY env var on non-Windows platforms (pyglet xlib would fail to
#      connect → segfault).
#   3. `--capture-replay` was passed on non-Windows platforms.  Capture replay
#      is offscreen-by-design there
#      (we write PNGs and stitch a video), and many display setups (DCV,
#      Xvfb, headless servers with a non-functional fake DISPLAY) hang or
#      crash in `ViewerGL.__init__` when pyglet tries to open a window.
#      The capture path is the natural use of this example — match that.
#      On Windows, pyglet's EGL headless backend is often unavailable, so the
#      automatic capture path uses a normal windowed OpenGL context instead.
_AUTO_HEADLESS_SUPPORTED = os.name != "nt"
_HEADLESS_REQUESTED = (
    "--headless" in sys.argv
    or (_AUTO_HEADLESS_SUPPORTED and not os.environ.get("DISPLAY"))
    or (_AUTO_HEADLESS_SUPPORTED and "--capture-replay" in sys.argv)
)
if _HEADLESS_REQUESTED:
    import pyglet  # noqa: PLC0415

    pyglet.options["headless"] = True
    if "--headless" not in sys.argv and "--no-headless" not in sys.argv:
        sys.argv.append("--headless")

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.examples.bag.capture import (
    add_capture_arguments as _add_capture_arguments,
    capture_replay_frame as _capture_replay_frame_common,
    configure_capture as _configure_capture_common,
    finalize_capture as _finalize_capture_common,
    finalize_replay_video as _finalize_replay_video_common,
)
from newton.examples.bag.mesh import (
    add_proxy_mesh_arguments as _add_proxy_mesh_arguments,
    build_bary_map as _shared_build_bary_map,
    decimate_mesh as _shared_decimate_mesh,
    load_kfc_mesh_zup as _shared_load_kfc_mesh_zup,
)
from newton.examples.bag.render import render_bag_meshes as _render_bag_meshes


# ---------------------------------------------------------------------------
# Timing constants (same phases as the LS-DYNA lift example)
# ---------------------------------------------------------------------------
_LOG_PREFIX = "[KFC ppfcs]"
_FPS = 60.0
_FRAME_DT = 1.0 / _FPS
_DEFAULT_PPFCS_SUBSTEPS_PER_FRAME = 2

_PHASE_OPEN_S = 0.45    # waypoint 0 duration: EE static at GRAB while fingers close
_PHASE_CLOSE_S = 0.20   # waypoint 1 duration: brief pinch settle, EE+fingers at GRAB
_PHASE_HOLD_S = 3.00    # waypoint 2 duration: EE rises GRAB→LIFT smoothstep (matches pyansys/radioss/lift)
_PHASE_LIFT_S = 2.50    # waypoint 3 duration: EE+fingers static at LIFT
_TOTAL_DURATION_S = _PHASE_OPEN_S + _PHASE_CLOSE_S + _PHASE_HOLD_S + _PHASE_LIFT_S

# ---------------------------------------------------------------------------
# Geometry defaults
# ---------------------------------------------------------------------------
_BAG_H_M = 0.279                    # nominal bag height [m]
_GRIP_BAND_TOP_FRAC = 0.85          # grippable band = top 15 % of bag
_PAD_HALF_THICKNESS_M = 0.0075      # finger-pad half-thickness [m]
_PAD_HALF_HEIGHT_M = 0.026          # finger-pad half-height [m]
_PAD_OPEN_CLEARANCE_M = 0.045       # extra clearance beyond bag when open [m]
_PAD_LIFT_Z_M = 0.65                # target Z in Z-up for pad centre when lifted [m]

# ---------------------------------------------------------------------------
# FR3 robot rendering (Newton-side replay only — ppfcs walls drive the bag)
# ---------------------------------------------------------------------------
_FR3_BASE_M = (-0.50, 0.0, 0.05)
_FR3_INIT_Q = [-3.6802e-03, 2.3902e-02, 3.6804e-03, -2.3683, -1.2919e-04, 2.3922, 7.8549e-01]
_FINGER_OPEN_Q = 0.04                # FR3 finger joint = 0.04 (open) → 0.0 (closed)
_FINGER_PAD_OFFSET_M = np.array([0.0, 0.00758, 0.0575], dtype=np.float64)
_GRAB_EE_Z_M = _BAG_H_M + 0.09       # EE z for grasp pose (≈0.369 m)
_LIFT_EE_Z_M = _PAD_LIFT_Z_M         # EE z at the end of the lift
# Vertical offset from the FR3 EE (hand link) down to the finger-pad
# centre in world Z when the gripper is in the standard down-pointing pose.
# Derived from the FR3 URDF: finger body sits ~5 cm below the hand link,
# and `_FINGER_PAD_OFFSET_M[2] = 0.0575` puts the pad ~5.75 cm below the
# finger body — total ~11.7 cm.  ppfcs's kinematic pads use this offset so
# their world Z matches the FR3 visual pads attached to the fingers.
_PAD_Z_FROM_EE_M = 0.117

# ---------------------------------------------------------------------------
# ppfcs material parameters (bags are thin engineering plastic)
# ---------------------------------------------------------------------------
# young-mod in ppfcs is `E / volumetric_density` and is unit-normalized — the
# default for tri shells is 100 (see frontend/_param_.py).  Higher values
# (>~1e3) cause the CCD line search to collapse to toi≈0 and the Newton
# solver cannot advance.  Stick close to the default for a paper bag.
_BAG_YOUNG_MOD = 400000.0       # 13× the original milestone's 30000.
                                # masses (15 g bag + 3 kg of interior contents), the bag
                                # walls must be stiff enough to support the interior load
                                # without instantly straining to the strain-limit cap.
                                # The earlier "lower young-mod" reasoning applied when the
                                # bag itself was unrealistically heavy and contributed
                                # inertia spikes; with light bag + heavy contents the
                                # reasoning inverts and we want a stiff cloth.
_BAG_DENSITY_BASE = 0.03        # density at α=1 (≈15 g bag for the KFC mesh).  The actual
                                # via density_kg_m3=240 × thickness=3.5e-4 m → 0.084 kg/m²
                                # surface mass.  ppfcs's `density` for tri shells acts as
                                # an effective surface-mass-density coefficient; this 0.03
                                # value lands the bag in the same ballpark (≈10-20 g for
                                # a ~0.18 m² bag).  Heavier values produce massive
                                # inertia on the gripped rim during smoothstep lift peak,
                                # forcing the strain-limit constraint into geometrically
                                # infeasible configurations and a NaN crash around
                                # frame ~85-92.
_BAG_BEND_BASE = 100000         # Strong bend stiffness (100× milestone's 1000).
                                # ppfcs's official `bend-rest-from-geometry`
                                # flag makes the input mesh the bend rest, so
                                # folded gussets don't receive phantom
                                # flattening stress.
_BAG_STRAIN_LIMIT = 0.12        # 12 % max per-element principal stretch.  At 0.07 the cap
                                # becomes geometrically infeasible during peak lift
                                # acceleration (frame ~84): the constraint pins one row of
                                # elements while inertia from bag-bottom + interior bodies
                                # keeps pulling, and a triangle eventually collapses to a
                                # NaN-producing degenerate config.  0.12 leaves enough
                                # elasticity in reserve to absorb the lift peak.
# NOTE: pads use `.pin()` *without* `.pull()` — that puts pad verts in ppfcs's
# FixPair table (scene.rs:1280-1296) which removes their DOFs from the linear
# system, giving a true kinematic constraint.  `.pull(N)` would make a soft
# constraint that drifts under bag spring-back during the close pinch.
# Friction values driving the grip.  ppfcs combines per-object frictions in
# pair contacts, so both surfaces need to be very high for firm grip on a
# thin shell.  ppfcs accepts μ > 1.0 — useful when the pad-bag normal force
# is small (thin shells deflect rather than compressing) and we need the
# tangential resistance to dominate.
_BAG_FRICTION = 1.5
_PAD_FRICTION = 3.0  # ppfcs combines per-contact friction as max(a, b); pad μ = effective μ
_FLOOR_FRICTION = 0.30

# Interior tet bodies — quasi-rigid via high young-mod + density (cf. ribbon.ipynb)
_INTERIOR_YOUNG_MOD = 2000.0
_INTERIOR_DENSITY_BASE = 4500.0 # density at α=1 (≈1 kg per interior body).  The actual
                                # density used is `_INTERIOR_DENSITY_BASE × α(target_faces)`
                                # — see `_compute_dynamic_alpha` below.

# Dynamic mesh-aware scaling.  Derived from ppfcs source (scene.rs:968,
# energy.cu:135, energy.cu:215) plus uniform-refinement scaling for a
# tri-shell:
#   per-vertex mass     m_v = density × A_v                ∝ density / N_v
#   per-vertex stretch  K_s ∝ mass × young-mod-terms       ∝ density × young-mod
#                                                            (mesh-INVARIANT)
#   per-vertex bend     K_b ∝ bend × edge_length           ∝ bend / √N_v
#
# Calibration: at target_faces=1200 the decimator yields ~765 verts, and
# (α=100, young-mod=400000, bend=100000) was validated as the user's
# preferred "really good stiffness" working point.
#
# Scaling exponents (tunable):
#   density(N_v) = density_base × (N_v / N_v_base)^1
#       — linear: keeps per-vertex mass constant for CG conditioning.
#   young-mod(N_v) = young-mod_base × (N_v / N_v_base)^_DYNAMIC_YOUNG_EXPONENT
#       — positive by default so higher target-face counts keep
#         a similar "loaded" stiffness to the 1200-face baseline instead of
#         stretching noticeably softer during the lift.
#   bend(N_v) = bend_base × (N_v / N_v_base)^_DYNAMIC_BEND_EXPONENT
#       — also scaled upward, but more gently than young-mod.  Fully
#         compensating bend at fine meshes makes frame-1 fold transients worse,
#         so keep the exponent below the stretch exponent.
_DYNAMIC_BASE_N_VERTS = 765       # calibration mesh size (target_faces=1200)
_DYNAMIC_ALPHA_BASE = 100.0       # density multiplier at N_v = base
_DYNAMIC_BEND_EXPONENT = 0.50     # stronger fold-stiffness recovery at fine meshes
_DYNAMIC_YOUNG_EXPONENT = 0.50    # stronger stretch-stiffness recovery at fine meshes


def _compute_dynamic_alpha(n_verts: int) -> float:
    """Density multiplier — keeps per-vertex mass constant under refinement."""
    return _DYNAMIC_ALPHA_BASE * float(n_verts) / float(_DYNAMIC_BASE_N_VERTS)


def _compute_dynamic_bend(n_verts: int) -> float:
    """Bend coefficient — exponent-tunable mesh scaling from the 1200-face baseline."""
    ratio = float(n_verts) / float(_DYNAMIC_BASE_N_VERTS)
    return _BAG_BEND_BASE * (ratio ** _DYNAMIC_BEND_EXPONENT)


def _compute_dynamic_young_mod(n_verts: int) -> float:
    """Young's modulus — exponent-tunable mesh scaling from the 1200-face baseline."""
    ratio = float(n_verts) / float(_DYNAMIC_BASE_N_VERTS)
    return _BAG_YOUNG_MOD * (ratio ** _DYNAMIC_YOUNG_EXPONENT)
_INTERIOR_FRICTION = 0.45

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
_DEFAULT_CLOSED_WIDTH_CM = 0.6
_MAX_GRIPPER_WIDTH_CM = 4.0
_SMALL_PAD_SCALE = math.sqrt(0.20)  # ~0.447 — reduces pad face area to ~20 %
_DEFAULT_JOB_DIR = Path("outputs/ppfcs/kfc_bag_lift")
_PPFCS_DIR_ENV = "PPFCS_DIR"
_DEFAULT_NUM_FRAMES = int(math.ceil(_TOTAL_DURATION_S / _FRAME_DT)) + 1


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _default_ppfcs_dir() -> str:
    env_value = os.environ.get(_PPFCS_DIR_ENV, "").strip()
    if env_value:
        return env_value

    repo_root = Path(__file__).resolve().parents[3]
    submodule = repo_root / "ppf-contact-solver"
    return str(submodule) if submodule.is_dir() else ""


def _ppfcs_binary_path(ppfcs_root: Path) -> Path:
    exe_name = "ppf-contact-solver.exe" if os.name == "nt" else "ppf-contact-solver"
    return ppfcs_root / "target" / "release" / exe_name


def _prepend_env_path(*paths: Path) -> None:
    entries = [str(path) for path in paths if path]
    if entries:
        os.environ["PATH"] = os.pathsep.join(entries + [os.environ.get("PATH", "")])


def _configure_ppfcs_runtime_env(ppfcs_root: Path) -> None:
    """Prepare platform-specific runtime paths before ppfcs starts."""
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
                f"{local_cuda}.\n"
                "Run `build-win-native\\warmup.bat /nopause` inside the ppfcs repo."
            )

    _prepend_env_path(lib_dir, cuda_path / "bin")


def _require_ppfcs(ppfcs_dir: Path):
    """Import App from the ppf-contact-solver Python frontend."""
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
            "Build with `cargo build --release` on Linux, or with "
            "`build-win-native\\warmup.bat /nopause` followed by "
            "`build-win-native\\build.bat /nopause` on Windows."
        )
    _configure_ppfcs_runtime_env(ppfcs_root)
    if str(ppfcs_root) not in sys.path:
        sys.path.insert(0, str(ppfcs_root))
    try:
        from frontend import App  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"Failed to import ppf-contact-solver frontend from {ppfcs_root}. "
            "Ensure the repo is correctly cloned and the frontend/ directory exists. "
            f"Original import error: {exc}"
        ) from exc
    return App


def _require_trimesh():
    try:
        import trimesh  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "This example requires trimesh. "
            "Install with `uv pip install trimesh`."
        ) from exc
    return trimesh


def _ppfcs_default_contact_gap(ppfcs_dir: Path) -> float:
    """Read the ppfcs frontend's default tri contact gap."""
    _require_ppfcs(ppfcs_dir)
    from frontend._param_ import object_param  # noqa: PLC0415

    return float(object_param("tri")["contact-gap"][0])


# ---------------------------------------------------------------------------
# Mesh loading and preparation
# ---------------------------------------------------------------------------

def _load_kfc_mesh_zup() -> tuple[np.ndarray, np.ndarray]:
    """Load the KFC bag mesh from `kfc.usd`, convert to Z-up, scale to metres."""
    verts_cm, faces = _shared_load_kfc_mesh_zup(_BAG_H_M * 100.0)
    return (verts_cm * 0.01).astype(np.float32), faces


def _decimate_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    proxy_mode: str,
    ppfcs_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the shared bag proxy mesh and convert it back to metres."""
    intersection_checker = None
    edge_edge_min_distance = 0.0
    if ppfcs_dir is not None:
        _require_ppfcs(ppfcs_dir)
        from frontend._intersection_ import check_self_intersection  # noqa: PLC0415

        def intersection_checker(check_verts: np.ndarray, check_faces: np.ndarray) -> list[tuple[int, int]]:
            return check_self_intersection(
                check_verts,
                check_faces,
                np.zeros(len(check_faces), dtype=bool),
                verbose=False,
            )

        # ppfcs native initialization rejects edge-edge pairs closer than the
        # summed contact gap.  Read the default from ppfcs so proxy cleanup
        # tracks the solver's own parameter table.
        edge_edge_min_distance = 2.0 * _ppfcs_default_contact_gap(ppfcs_dir) * 100.0

    out_verts_cm, out_faces = _shared_decimate_mesh(
        verts * 100.0,
        faces,
        target_faces,
        proxy_mode,
        make_intersection_free=ppfcs_dir is not None,
        intersection_checker=intersection_checker,
        checker_transform=lambda verts_cm: _zup_to_yup(verts_cm * 0.01),
        edge_edge_min_distance=edge_edge_min_distance,
        intersection_free_min_area=0.1,
        log_prefix=_LOG_PREFIX,
    )
    return (out_verts_cm * 0.01).astype(np.float32), out_faces


def _make_box_mesh_verts_faces(
    half_extents: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a triangulated axis-aligned box mesh."""
    trimesh = _require_trimesh()
    mesh = trimesh.creation.box(extents=(2.0 * half_extents).tolist())
    mesh.apply_translation(center.tolist())
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _build_bary_map(
    full_verts: np.ndarray,
    sim_verts: np.ndarray,
    sim_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map each full-res vertex to a nearby proxy triangle."""
    return _shared_build_bary_map(full_verts, sim_verts, sim_faces)


# ---------------------------------------------------------------------------
# Coordinate conversion (Z-up ↔ Y-up)
# ---------------------------------------------------------------------------

def _zup_to_yup(arr: np.ndarray) -> np.ndarray:
    """[x, y, z]_zup → [x, z, y]_yup  (swap Y and Z)."""
    return arr[:, [0, 2, 1]]


def _yup_to_zup(arr: np.ndarray) -> np.ndarray:
    """[x, y, z]_yup → [x, z, y]_zup  (swap Y and Z, same transform)."""
    return arr[:, [0, 2, 1]]


# ---------------------------------------------------------------------------
# Interior body geometry (sphere/box/capsule fit inside the bag's grippable band)
# ---------------------------------------------------------------------------
# Sizes are deliberately small so all three fit side-by-side in the bag's
# widest cross-section.  ppfcs treats the volumetric stiffness + density as
# the dominant rigidity factor (cf. ribbon.ipynb), so absolute size doesn't
# need to match the pyansys defaults exactly.
# Match pyansys / radioss bag-lift example sizes so the contents look the same
# across all three backends.
_SPHERE_RADIUS_M = 0.04
_BOX_HALF_EXTENT_M = 0.03
_CAPSULE_RADIUS_M = 0.03
_CAPSULE_HALF_HEIGHT_M = 0.02
# Margin around each object to avoid initial self-intersections in ppfcs.
_INTERIOR_CLEARANCE_M = 0.010


def _fit_bag_contents(bag_verts_zup_m: np.ndarray) -> dict:
    """Return Z-up positions and orientations for the 3 rigid interior bodies.

    Layout matches the radioss/pyansys lift examples:
      sphere on the right side, box on the lower-left, capsule stacked on
      top of the box and rotated π/2 around X so it lies horizontally.

    Returned dict values are 7-vectors ``[x, y, z, qx, qy, qz, qw]`` so the
    caller can write them straight into a Newton (Z-up) ``body_q``.

    Z positions are anchored to the bag's actual z_min so the interior bodies
    remain clear of the floor and bag.
    """
    cx = float(0.5 * (bag_verts_zup_m[:, 0].max() + bag_verts_zup_m[:, 0].min()))
    cy = float(0.5 * (bag_verts_zup_m[:, 1].max() + bag_verts_zup_m[:, 1].min()))
    z_floor = float(bag_verts_zup_m[:, 2].min())

    # Original tuned positions (calibrated when bag z_min == 0).
    sphere = np.array([cx + 0.025, cy, z_floor + 0.050], dtype=np.float64)
    box = np.array([cx - 0.055, cy, z_floor + 0.040], dtype=np.float64)
    # ppfcs's tet meshing introduces small surface variations; a 1 mm gap
    # between bodies is too tight (radioss uses rigid bodies and gets away
    # with it).  Use 8 mm clearance to keep frame-0 self-intersection-free.
    capsule_xyz = np.array(
        [box[0], cy, box[2] + _BOX_HALF_EXTENT_M + _CAPSULE_RADIUS_M + 0.008],
        dtype=np.float64,
    )
    capsule_quat = Rotation.from_rotvec(
        np.array([0.5 * math.pi, 0.0, 0.0], dtype=np.float64)
    ).as_quat()

    identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return {
        "sphere":  np.concatenate([sphere,      identity_q]),
        "box":     np.concatenate([box,         identity_q]),
        "capsule": np.concatenate([capsule_xyz, capsule_quat.astype(np.float64)]),
    }


def _fit_rigid_transform(V_ref: np.ndarray, V_curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Procrustes alignment: find rotation R and translation t minimizing
    ||(R @ V_ref + t) − V_curr||.  Both arrays shape (n, 3)."""
    cR = V_ref.mean(axis=0)
    cC = V_curr.mean(axis=0)
    A = V_ref - cR
    B = V_curr - cC
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cC - R @ cR
    return R, t


def _zup_pose_to_yup(pose_zup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Z-up [pos|quat_xyzw] pose to (centre_yup, R_yup_3x3).

    The Z-up→Y-up coordinate change is the permutation P = [[1,0,0],[0,0,1],
    [0,1,0]].  For a rotation R_zup expressed in Z-up, the equivalent Y-up
    rotation is R_yup = P · R_zup · P^T (P is symmetric so P^T = P).
    """
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    centre_yup = P @ np.asarray(pose_zup[:3], dtype=np.float64)
    R_zup = Rotation.from_quat(pose_zup[3:7]).as_matrix()
    R_yup = P @ R_zup @ P
    return centre_yup, R_yup


# ---------------------------------------------------------------------------
# Grip geometry
# ---------------------------------------------------------------------------

def _compute_grip_geometry(
    bag_verts_zup_m: np.ndarray,
    closed_width_cm: float,
) -> dict:
    """Derive finger-pad dimensions and trajectory waypoints from the bag mesh."""
    z_min = float(bag_verts_zup_m[:, 2].min())
    bag_h = float(bag_verts_zup_m[:, 2].max() - z_min)

    # Grippable band: top 15 %
    grip_z_lo = z_min + _GRIP_BAND_TOP_FRAC * bag_h
    mask = bag_verts_zup_m[:, 2] >= grip_z_lo
    gv = bag_verts_zup_m[mask] if mask.any() else bag_verts_zup_m

    bag_cx = float(0.5 * (gv[:, 0].max() + gv[:, 0].min()))
    bag_cy = float(0.5 * (gv[:, 1].max() + gv[:, 1].min()))
    bag_half_x = max(float(0.5 * (gv[:, 0].max() - gv[:, 0].min())), 0.02)
    bag_half_y = float(0.5 * (gv[:, 1].max() - gv[:, 1].min()))

    # ppfcs walls are infinite half-spaces — they constrain the entire bag in
    # Y, not just the grippable band.  Use the bag's full Y extent for the open
    # position so the walls don't intersect the wider bottom of the bag at
    # frame 0 (which produces toi=0 in CCD and stalls the solver).
    full_y_lo = float(bag_verts_zup_m[:, 1].min())
    full_y_hi = float(bag_verts_zup_m[:, 1].max())
    full_half_y = float(0.5 * (full_y_hi - full_y_lo))
    full_cy = float(0.5 * (full_y_hi + full_y_lo))

    # Pad half-extents in Z-up frame (hx=X, hy=Y-close direction, hz=Z-height)
    pad_hx = bag_half_x + 0.01     # slightly wider than bag
    pad_hy = _PAD_HALF_THICKNESS_M
    pad_hz = _PAD_HALF_HEIGHT_M

    # Pad centre height in Z-up.  Anchor it to the FR3 IK trajectory: when
    # the EE is at _GRAB_EE_Z_M the FR3 finger pads (attached via
    # _FINGER_PAD_OFFSET_M) end up ~_PAD_Z_FROM_EE_M below the EE.  Using
    # the same value for the ppfcs kinematic pads keeps the physics pads
    # co-located with the visual pads on the FR3 fingers.
    z_pad_centre = _GRAB_EE_Z_M - _PAD_Z_FROM_EE_M

    # Finger-pad inner-face positions in Z-up Y (close direction).  Use the
    # full bag's Y span so the walls start outside the entire bag.
    closed_half = float(closed_width_cm) * 0.005  # cm → m, half-gap
    y_right_closed = full_cy + closed_half
    y_right_open   = full_cy + full_half_y + pad_hy + _PAD_OPEN_CLEARANCE_M
    y_left_closed  = full_cy - closed_half
    y_left_open    = full_cy - full_half_y - pad_hy - _PAD_OPEN_CLEARANCE_M

    # Lift delta matches the FR3 IK's EE travel (so visual & physics pads
    # both rise by the same amount).
    lift_delta_z = _LIFT_EE_Z_M - _GRAB_EE_Z_M

    return dict(
        bag_cx=bag_cx,
        bag_cy=bag_cy,
        bag_half_x=bag_half_x,
        bag_half_y=bag_half_y,
        pad_hx=pad_hx,
        pad_hy=pad_hy,
        pad_hz=pad_hz,
        z_pad_centre=z_pad_centre,
        y_right_closed=y_right_closed,
        y_right_open=y_right_open,
        y_left_closed=y_left_closed,
        y_left_open=y_left_open,
        lift_delta_z=lift_delta_z,
    )


def _pad_centre_at_time_zup(g: dict, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_centre, right_centre) in Z-up at simulation time t [m]."""
    t0 = _PHASE_OPEN_S
    t1 = t0 + _PHASE_CLOSE_S
    t2 = t1 + _PHASE_HOLD_S
    t3 = t2 + _PHASE_LIFT_S

    # Close fraction
    if t <= t0:
        close_frac = 0.0
    elif t <= t1:
        close_frac = (t - t0) / _PHASE_CLOSE_S
    else:
        close_frac = 1.0

    # Lift fraction
    if t <= t2:
        lift_frac = 0.0
    elif t <= t3:
        lift_frac = (t - t2) / _PHASE_LIFT_S
    else:
        lift_frac = 1.0

    y_right = g["y_right_open"] + close_frac * (g["y_right_closed"] - g["y_right_open"])
    y_left  = g["y_left_open"]  + close_frac * (g["y_left_closed"]  - g["y_left_open"])
    z_pad = g["z_pad_centre"] + lift_frac * g["lift_delta_z"]

    right_centre = np.array([g["bag_cx"], y_right + g["pad_hy"], z_pad], dtype=np.float32)
    left_centre  = np.array([g["bag_cx"], y_left  - g["pad_hy"], z_pad], dtype=np.float32)
    return left_centre, right_centre


# ---------------------------------------------------------------------------
# ppf-contact-solver simulation
# ---------------------------------------------------------------------------


class _StreamingFrameSource:
    """Frame source that pulls per-body vert frames from a running ppfcs solver."""

    def __init__(
        self,
        fixed_session,
        body_indices: dict[str, np.ndarray],
        sim_dt: float,
        frame_dt: float,
        num_display_frames: int,
        n_sim_steps: int,
    ):
        self._session = fixed_session
        # Each entry: name → np.ndarray of vertex indices into the global vert array.
        self._body_indices = body_indices
        self._sim_dt = sim_dt
        self._frame_dt = frame_dt
        self._num_display_frames = num_display_frames
        self._n_sim_steps = n_sim_steps
        self._cache: list[dict[str, np.ndarray]] = []
        self._reported = -1

    @property
    def num_frames(self) -> int:
        return self._num_display_frames

    def fetch(self, display_idx: int) -> dict:
        if display_idx < len(self._cache):
            return self._cache[display_idx]

        # ppfcs writes `vert_N.bin` for *video frame* N (backend.rs:208-244).
        # `_start_ppfcs_streaming()` explicitly sets ppfcs `fps = 1 / frame_dt`,
        # so the Newton display frame index matches the ppfcs video-frame index.
        # That lets us read `vert_N.bin` directly while still letting ppfcs take
        # multiple solver substeps within each replay frame.
        target_sim_idx = display_idx

        while True:
            done = self._session.get.vertex_frame_count()
            if done != self._reported and done % max(1, self._n_sim_steps // 20) == 0:
                print(f"{_LOG_PREFIX} {done}/{self._n_sim_steps} sim frames computed...")
                self._reported = done
            if done >= target_sim_idx:
                break
            if self._session.finished():
                target_sim_idx = max(min(target_sim_idx, done), 0)
                break
            if not self._session.is_running():
                err_path = Path(self._session.info.path) / "error.log"
                tail = (
                    err_path.read_text().strip().splitlines()[-10:]
                    if err_path.exists()
                    else []
                )
                raise RuntimeError(
                    f"ppfcs solver exited before producing frame {target_sim_idx} "
                    f"(only {done}/{self._n_sim_steps} written). "
                    "Last error.log lines:\n  " + "\n  ".join(tail)
                )
            time.sleep(0.1)

        result = self._session.get.vertex(target_sim_idx)
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
                bodies[name] = _yup_to_zup(all_verts_yup[idx].astype(np.float32))

        while len(self._cache) < display_idx:
            self._cache.append({k: v.copy() for k, v in bodies.items()})
        self._cache.append(bodies)
        return bodies

    def all_frames(self) -> np.ndarray:
        """Bag-only stack for test assertions."""
        if not self._cache:
            n = len(self._body_indices.get("bag", []))
            return np.zeros((0, n, 3), dtype=np.float32)
        return np.stack([c["bag"] for c in self._cache], axis=0)


_INTERIOR_TET_EDGE_FAC = 0.15  # coarser tet → fewer verts → less prone to self-intersect

# Quasi-rigid params for the kinematic finger pads (pinned, driven by move_by).
# Stiffer than the bag by 1+ order of magnitude so friction force isn't
# absorbed by pad deformation; the pads stay rigid in shape during contact.
_PAD_YOUNG_MOD = 20000.0
_PAD_DENSITY = 1500.0
_PAD_TET_EDGE_FAC = 0.25  # coarser still — the pad is a slim slab


def _make_tet_box(app, name: str, hx: float, hy: float, hz: float, centre_yup: np.ndarray):
    """Build a tet box centred at `centre_yup` and register it as `name`.

    Returns the surface-triangle array (local indices) for rendering.
    """
    tri = app.mesh.box(width=2.0 * hx, height=2.0 * hy, depth=2.0 * hz)
    tet_mesh = tri.tetrahedralize(edge_length_fac=_INTERIOR_TET_EDGE_FAC)
    V = np.asarray(tet_mesh[0], dtype=np.float64) + centre_yup
    F = np.asarray(tet_mesh[1], dtype=np.int32)
    T = np.asarray(tet_mesh[2], dtype=np.int32)
    app.asset.add.tet(name, V, F, T)
    return F


def _make_tet_sphere(app, name: str, radius: float, centre_yup: np.ndarray, subdiv: int = 1):
    tri = app.mesh.icosphere(r=radius, subdiv_count=subdiv)
    tet_mesh = tri.tetrahedralize(edge_length_fac=_INTERIOR_TET_EDGE_FAC)
    V = np.asarray(tet_mesh[0], dtype=np.float64) + centre_yup
    F = np.asarray(tet_mesh[1], dtype=np.int32)
    T = np.asarray(tet_mesh[2], dtype=np.int32)
    app.asset.add.tet(name, V, F, T)
    return F


def _make_tet_pad(
    app,
    name: str,
    hx: float,
    hy: float,
    hz: float,
    centre_yup: np.ndarray,
):
    """Build a TRI-SHELL box for a finger pad.  The pad is driven kinematically
    via `pin()` (no `.pull()`) so it doesn't need volumetric simulation; only
    the closed surface participates in friction contact with the bag.

    All ppfcs example notebooks that combine `pin()` (no pull) with `move_by`
    use tri-shell colliders (belt rollers, slope, etc.) — using a tet body in
    that role triggers a NaN in CG iter 1 from the system being assembled with
    every pad-vert DOF removed.
    """
    box = app.mesh.box(width=2.0 * hx, height=2.0 * hy, depth=2.0 * hz)
    V_local, F = box[0], box[1]
    V = np.asarray(V_local, dtype=np.float64) + centre_yup
    F = np.asarray(F, dtype=np.int32)
    app.asset.add.tri(name, V, F)
    return F


def _make_tet_box_oriented(
    app, name: str,
    hx: float, hy: float, hz: float,
    centre_yup: np.ndarray, R_yup: np.ndarray,
):
    """Build a tet box rotated by R_yup (3×3) and translated to centre_yup."""
    tri = app.mesh.box(width=2.0 * hx, height=2.0 * hy, depth=2.0 * hz)
    tet_mesh = tri.tetrahedralize(edge_length_fac=_INTERIOR_TET_EDGE_FAC)
    V = np.asarray(tet_mesh[0], dtype=np.float64)
    V = (R_yup @ V.T).T + centre_yup
    F = np.asarray(tet_mesh[1], dtype=np.int32)
    T = np.asarray(tet_mesh[2], dtype=np.int32)
    app.asset.add.tet(name, V, F, T)
    return F


def _make_tet_capsule_oriented(
    app, name: str, radius: float, half_height: float,
    centre_yup: np.ndarray, R_yup: np.ndarray,
):
    """Stretched-icosphere capsule rotated by R_yup, translated to centre_yup."""
    tri = app.mesh.icosphere(r=radius, subdiv_count=1)
    tet_mesh = tri.tetrahedralize(edge_length_fac=_INTERIOR_TET_EDGE_FAC)
    V = np.asarray(tet_mesh[0], dtype=np.float64).copy()
    # Stretch along the local Y-axis (capsule's long axis in object frame).
    V[:, 1] *= 1.0 + half_height / radius
    V = (R_yup @ V.T).T + centre_yup
    F = np.asarray(tet_mesh[1], dtype=np.int32)
    T = np.asarray(tet_mesh[2], dtype=np.int32)
    app.asset.add.tet(name, V, F, T)
    return F


def _make_tet_capsule(app, name: str, radius: float, half_height: float, centre_yup: np.ndarray):
    """Approximate a capsule with a stretched icosphere via ppfcs's tet helper.

    Stretching after `tetrahedralize()` keeps the surface manifold consistent
    (no need for a hand-rolled surface extraction).
    """
    tri = app.mesh.icosphere(r=radius, subdiv_count=1)
    tet_mesh = tri.tetrahedralize(edge_length_fac=_INTERIOR_TET_EDGE_FAC)
    V = np.asarray(tet_mesh[0], dtype=np.float64).copy()
    # Stretch along Y-up Y for capsule-ish elongated profile.
    V[:, 1] *= 1.0 + half_height / radius
    V += centre_yup
    F = np.asarray(tet_mesh[1], dtype=np.int32)
    T = np.asarray(tet_mesh[2], dtype=np.int32)
    app.asset.add.tet(name, V, F, T)
    return F


def _start_ppfcs_streaming(
    bag_verts_zup_m: np.ndarray,
    bag_faces: np.ndarray,
    grip: dict,
    ppfcs_dir: Path,
    job_dir: Path,
    num_frames: int,
    frame_dt: float,
    ppfcs_substeps_per_frame: int,
    *,
    small_pad: bool = False,
) -> tuple[_StreamingFrameSource, dict[str, np.ndarray]]:
    """Start the ppfcs solver and return ``(streamer, body_surface_faces)``.

    `body_surface_faces` maps each body name to its surface triangle array
    (local to the body's vertex slice) so the caller can render via log_mesh.
    """
    App = _require_ppfcs(ppfcs_dir)
    if ppfcs_substeps_per_frame < 1:
        raise ValueError("--ppfcs-substeps-per-frame must be >= 1.")

    # Convert bag mesh to Y-up (ppfcs convention: Y is up, gravity = -9.8 in Y)
    bag_verts_yup = _zup_to_yup(bag_verts_zup_m).astype(np.float64)
    bag_faces_i32 = bag_faces.astype(np.int32)
    output_fps = 1.0 / frame_dt
    sim_dt = frame_dt / float(ppfcs_substeps_per_frame)
    max_output_frame_idx = max(int(num_frames) - 1, 0)
    total_output_s = max_output_frame_idx * frame_dt

    # Phase transition times — matches the canonical IK waypoint convention
    # used by example_kfc_bag_lift_pyansys / _radioss / _lift:
    #   waypoint 0 (0     .. t_close): EE static, gripper closing → pads close laterally here
    #   waypoint 1 (t_close .. t_hold): EE static, brief pinch settle, pads stationary
    #   waypoint 2 (t_hold  .. t_lift): EE rises GRAB→LIFT smoothstep → pads rise vertically here
    #   waypoint 3 (t_lift  .. t_lift + _PHASE_LIFT_S): EE+pads stationary at LIFT
    t_open  = 0.0
    t_close = t_open + _PHASE_OPEN_S
    t_hold  = t_close + _PHASE_CLOSE_S
    t_lift  = t_hold + _PHASE_HOLD_S

    # Wall positions in Y-up: [x, y, z] where y = height, z = close direction
    # Y-up Z  ↔  Z-up Y  (swap via _zup_to_yup)
    bag_cx_yup = float(grip["bag_cx"])           # X unchanged
    y_pad_yup  = float(grip["z_pad_centre"])     # Z-up Z → Y-up Y (height)
    z_r_open   = float(grip["y_right_open"])     # Z-up Y → Y-up Z (right, positive)
    z_r_closed = float(grip["y_right_closed"])
    z_l_open   = float(grip["y_left_open"])      # Z-up Y → Y-up Z (left, negative)
    z_l_closed = float(grip["y_left_closed"])
    lift_dy    = float(grip["lift_delta_z"])     # Z-up ΔZ → Y-up ΔY

    print(f"{_LOG_PREFIX} Bag verts: {len(bag_verts_yup)}, faces: {len(bag_faces_i32)}")
    print(f"{_LOG_PREFIX} Grip height (Y-up Y): {y_pad_yup:.4f} m")
    print(f"{_LOG_PREFIX} Close range (Y-up Z): {z_l_open:.4f} → {z_l_closed:.4f} (left), "
          f"{z_r_open:.4f} → {z_r_closed:.4f} (right)")
    print(f"{_LOG_PREFIX} Lift ΔY (Y-up): {lift_dy:.4f} m")
    print(f"{_LOG_PREFIX} Replay: {num_frames} frames × dt={frame_dt:.5f} s "
          f"= {total_output_s:.3f} s total")
    print(
        f"{_LOG_PREFIX} ppfcs timing: fps={output_fps:.3f}, dt={sim_dt:.5f} s "
        f"({ppfcs_substeps_per_frame} substeps / replay frame), "
        f"frame_limit={max_output_frame_idx}"
    )

    app = App.create("kfc_lift", cache_dir=str(job_dir))

    # Add the bag mesh as a deformable tri-mesh asset
    app.asset.add.tri("bag", bag_verts_yup, bag_faces_i32)

    # Build scene
    scene = app.scene.create("lift")

    # NOTE: ppfcs Wall.move_by() does element-wise addition only when the
    # initial position and delta are numpy arrays — passing Python lists
    # triggers list concatenation and breaks the C++ solver's keyframe count.
    _v3 = lambda *a: np.array(a, dtype=np.float64)  # noqa: E731

    # ---- Floor (Y=0 in Y-up) -----------------------------------------------
    floor = scene.add.invisible.wall(_v3(bag_cx_yup, 0.0, 0.0), _v3(0.0, 1.0, 0.0))
    floor.param.set("friction", _FLOOR_FRICTION)

    # ---- Tet finger pads (pinned, kinematically driven) --------------------
    # Pad dims in Y-up: hx = X width, hy = Y height, hz = Z thickness.
    # grip dict's keys are in Z-up convention: pad_hx (X), pad_hy (Y_thick),
    # pad_hz (Z_height).  In Y-up these map to (X, Y_thick→Z, Z_height→Y).
    # `--small-pad` shrinks the contact face area to ~20% (matching the
    # pyansys/radioss convention) by scaling X width and Z height; thickness
    # stays unchanged.
    vis_scale = _SMALL_PAD_SCALE if small_pad else 1.0
    pad_hx_yup = float(grip["pad_hx"]) * vis_scale
    pad_hy_yup = float(grip["pad_hz"]) * vis_scale  # height (Y in Y-up)
    pad_hz_yup = float(grip["pad_hy"])  # thickness (= _PAD_HALF_THICKNESS_M)

    # Pads start at the open Z position with their inner face at z_*_open;
    # the centre is offset outward by pad_hz_yup so the bag never touches the
    # pad surface at frame 0.
    right_pad_centre_open = _v3(bag_cx_yup, y_pad_yup, z_r_open + pad_hz_yup)
    left_pad_centre_open  = _v3(bag_cx_yup, y_pad_yup, z_l_open  - pad_hz_yup)

    print(f"{_LOG_PREFIX} Building tet finger pads (kinematic, "
          f"hx={pad_hx_yup:.3f} hy={pad_hy_yup:.3f} hz={pad_hz_yup:.4f} m)...")
    right_pad_F = _make_tet_pad(app, "right_pad", pad_hx_yup, pad_hy_yup, pad_hz_yup, right_pad_centre_open)
    left_pad_F  = _make_tet_pad(app, "left_pad",  pad_hx_yup, pad_hy_yup, pad_hz_yup, left_pad_centre_open)

    # ---- Interior tet bodies (sphere / box / capsule) ----------------------
    # These behave quasi-rigidly thanks to high young-mod + density vs the
    # bag's tri-shell material (cf. ribbon.ipynb's heavy ball pattern).
    contents_zup = _fit_bag_contents(bag_verts_zup_m)
    interior_specs = []
    print(f"{_LOG_PREFIX} Building interior tet bodies (sphere/box/capsule)...")

    sphere_centre_yup, sphere_R = _zup_pose_to_yup(contents_zup["sphere"])
    sphere_F = _make_tet_sphere(app, "sphere", _SPHERE_RADIUS_M, sphere_centre_yup)
    interior_specs.append(("sphere", sphere_F))

    box_centre_yup, box_R = _zup_pose_to_yup(contents_zup["box"])
    box_F = _make_tet_box_oriented(
        app, "box",
        _BOX_HALF_EXTENT_M, _BOX_HALF_EXTENT_M, _BOX_HALF_EXTENT_M,
        box_centre_yup, box_R,
    )
    interior_specs.append(("box", box_F))

    cap_centre_yup, cap_R = _zup_pose_to_yup(contents_zup["capsule"])
    cap_F = _make_tet_capsule_oriented(
        app, "capsule", _CAPSULE_RADIUS_M, _CAPSULE_HALF_HEIGHT_M, cap_centre_yup, cap_R
    )
    interior_specs.append(("capsule", cap_F))

    # ---- Bag object ---------------------------------------------------------
    # NOTE: `strain-limit` is intentionally NOT set.  In principle a per-element
    # principal-stretch cap is the right way to make the bag behave like
    # inextensible paper, but in practice ppfcs's Baraff-Witkin + IPC + SL
    # combination becomes geometrically infeasible during the lift acceleration
    # peak when the bag is supporting heavy interior contents (~3 kg) on a
    # light shell — a triangle ends up with a constraint configuration that
    # produces NaN around frame ~50-90 depending on parameters.  We rely on
    # `young-mod` + `bend` alone for stretch resistance.  This means the bag
    # will visibly stretch under the lift load (an "elastic" rather than
    # "plastic" behaviour); to firm that up further bump `_BAG_YOUNG_MOD`.
    # ---- Dynamic mesh-aware scaling ---------------------------------------
    # Scale density from the *actual* decimated vertex count so per-vertex mass
    # stays stable for CG conditioning, and also scale the bag stiffness terms
    # from the same mesh ratio so higher `--target-faces` runs stay closer to
    # the 1200-face baseline's "loaded" feel during the lift.
    n_verts_bag = int(len(bag_verts_zup_m))
    alpha = _compute_dynamic_alpha(n_verts_bag)
    bag_bend = _compute_dynamic_bend(n_verts_bag)
    bag_young = _compute_dynamic_young_mod(n_verts_bag)
    bag_density = _BAG_DENSITY_BASE * alpha
    interior_density = _INTERIOR_DENSITY_BASE * alpha
    stiff_ratio = float(n_verts_bag) / float(_DYNAMIC_BASE_N_VERTS)
    # Empirical SI-mass calibration from earlier sessions: density 0.03 ≈ 15 g
    # bag, density 4500 ≈ 1 kg/body.  Used purely for informative log output.
    bag_mass_g = bag_density / 0.03 * 15.0
    interior_mass_kg = interior_density / 4500.0
    print(
        f"{_LOG_PREFIX} Dynamic scaling: bag N_v={n_verts_bag} (base={_DYNAMIC_BASE_N_VERTS}) "
        f"→ α={alpha:.1f}, ratio={stiff_ratio:.2f}x"
    )
    print(
        f"{_LOG_PREFIX}   young-mod={bag_young:.0f} "
        f"({bag_young / _BAG_YOUNG_MOD:.2f}x base, exp={_DYNAMIC_YOUNG_EXPONENT}), "
        f"bend={bag_bend:.0f} "
        f"({bag_bend / _BAG_BEND_BASE:.2f}x base, exp={_DYNAMIC_BEND_EXPONENT})"
    )
    print(
        f"{_LOG_PREFIX}   bag density={bag_density:.3f} (≈{bag_mass_g:.0f} g) — "
        f"interior density={interior_density:.0f} (≈{interior_mass_kg:.0f} kg/body)"
    )

    bag_obj = scene.add("bag")
    bag_obj.param.set("young-mod", bag_young)
    bag_obj.param.set("density", bag_density)
    bag_obj.param.set("bend", bag_bend)
    bag_obj.param.set("friction", _BAG_FRICTION)
    # Use the input mesh's per-hinge dihedral angles as the bend rest
    # reference. ppfcs computes HingeProp.rest_angle during scene build when
    # this per-object flag is set, so it must be configured before scene.build().
    bag_obj.param.set("bend-rest-from-geometry", True)
    print(f"{_LOG_PREFIX} Bag bend rest = input dihedral angles")

    # Add interior bodies to the scene with high stiffness + density.  Each
    # body uses ppfcs's tet defaults except for stiffness/density/friction.
    for name, _F in interior_specs:
        obj = scene.add(name)
        obj.param.set("young-mod", _INTERIOR_YOUNG_MOD)
        obj.param.set("density", interior_density)
        obj.param.set("friction", _INTERIOR_FRICTION)

    # Add the tet pads, pin every vertex, and animate them through the
    # open→close→hold→lift trajectory via PinHolder.move_by(delta, t0, t1).
    # Use the ribbon.ipynb pattern: `.pin()` makes the body kinematic
    # via a strong constraint without zeroing all DOFs in the matrix.
    right_pad_obj = scene.add("right_pad")
    right_pad_obj.param.set("young-mod", _PAD_YOUNG_MOD)
    right_pad_obj.param.set("density",   _PAD_DENSITY)
    right_pad_obj.param.set("friction",  _PAD_FRICTION)
    right_holder = right_pad_obj.pin()
    right_holder.move_by(_v3(0.0, 0.0, z_r_closed - z_r_open), t_open, t_close)
    # Smooth lift: ramp velocity gradually so the bag isn't peeled off by an
    # initial velocity step — the friction limit must beat inertial force.
    right_holder.interp("smooth").move_by(_v3(0.0, lift_dy, 0.0), t_hold, t_lift)

    left_pad_obj = scene.add("left_pad")
    left_pad_obj.param.set("young-mod", _PAD_YOUNG_MOD)
    left_pad_obj.param.set("density",   _PAD_DENSITY)
    left_pad_obj.param.set("friction",  _PAD_FRICTION)
    left_holder = left_pad_obj.pin()
    left_holder.move_by(_v3(0.0, 0.0, z_l_closed - z_l_open), t_open, t_close)
    left_holder.interp("smooth").move_by(_v3(0.0, lift_dy, 0.0), t_hold, t_lift)

    # Track pad assets so they're rendered alongside the bag and interior bodies.
    interior_specs.append(("right_pad", right_pad_F))
    interior_specs.append(("left_pad",  left_pad_F))

    fixed_scene = scene.build()

    # Collect per-body vertex-index slices from the scene's name map.
    body_indices: dict[str, np.ndarray] = {}
    body_surface_faces: dict[str, np.ndarray] = {}

    bag_vert_indices = fixed_scene._map_by_name.get("bag", None)
    if bag_vert_indices is None:
        bag_vert_indices = list(range(len(bag_verts_yup)))
    body_indices["bag"] = np.array(bag_vert_indices, dtype=np.int32)
    body_surface_faces["bag"] = bag_faces_i32
    print(f"{_LOG_PREFIX} Bag vertex indices count: {len(bag_vert_indices)}")

    for name, F in interior_specs:
        idx = fixed_scene._map_by_name.get(name, None)
        if idx is None:
            print(f"{_LOG_PREFIX} Warning: no vertex index map for '{name}'.")
            continue
        body_indices[name] = np.array(idx, dtype=np.int32)
        body_surface_faces[name] = F
        print(f"{_LOG_PREFIX}   {name}: {len(idx)} verts, {len(F)} surface tris")

    # ---- Session parameters -------------------------------------------------
    # ppfcs advances with `dt` but writes `vert_N.bin` on its video clock set by
    # `fps`.  Keep the output clock aligned to Newton's replay interval while
    # using an integer number of solver substeps within each replay frame.

    session = app.session.create(fixed_scene)
    session.param.set("frames", max_output_frame_idx)
    session.param.set("dt", sim_dt)
    session.param.set("fps", output_fps)
    # Static friction requires multiple Newton steps per frame
    session.param.set("min-newton-steps", 32)
    # Headroom for CG convergence at high mesh densities (default is 10 000;
    # the strain-limit constraint at --target-faces 15000 occasionally needs
    # more during the close-pinch peak).
    session.param.set("cg-max-iter", 50000)

    fixed_session = session.build()
    # Carry sim_dt out for the output sampling step below
    n_sim_steps = max_output_frame_idx
    sim_dt_used = sim_dt

    # ---- Launch solver (non-blocking) ---------------------------------------
    # `blocking=False` is critical: the default is True (outside Jupyter), and
    # would make start() block until the entire simulation finishes — which
    # would defeat per-frame streaming entirely.  With non-blocking, start()
    # spawns the solver subprocess and returns immediately so the Newton
    # viewer can begin pulling per-frame results as the solver writes them.
    print(f"{_LOG_PREFIX} Starting ppfcs solver (streaming mode — frames will "
          "be captured as soon as each becomes available)...", flush=True)
    # Belt-and-suspenders cleanup: SIGTERM any leftover ppfcs binary process
    # (Utils.busy() detects by name and would otherwise refuse to start a
    # fresh solver with `Solver is already running.` from start()).  Pass
    # force=True so the same check inside start() also reaps stale state.
    from frontend import Utils as _PpfcsUtils  # noqa: PLC0415
    try:
        _PpfcsUtils.terminate()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.3)
    fixed_session.start(force=True, blocking=False)

    # Register SIGINT/SIGTERM/atexit cleanup so Ctrl-C kills the orphaned
    # ppf-contact-solver subprocess instead of leaving it running indefinitely.
    # ppfcs starts the solver with `start_new_session=True`, which detaches it
    # from the controlling terminal — without explicit cleanup it survives the
    # parent's death.  Utils.terminate() finds processes by name and SIGTERMs
    # them; we also fall back to the session's own `_process` handle if set.
    import atexit  # noqa: PLC0415
    import signal  # noqa: PLC0415
    from frontend import Utils  # noqa: PLC0415

    _cleaned = {"done": False}

    def _kill_solver(*_args):
        if _cleaned["done"]:
            return
        _cleaned["done"] = True
        proc = getattr(fixed_session, "_process", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        # Belt-and-suspenders: nuke any other ppf-contact-solver process the
        # session might have spawned (e.g. monitor helpers).
        try:
            Utils.terminate()
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_kill_solver)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev = signal.getsignal(_sig)

            def _handler(signum, frame, _prev=prev):
                _kill_solver()
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    _prev(signum, frame)
                else:
                    raise KeyboardInterrupt
            signal.signal(_sig, _handler)
        except (ValueError, OSError):
            # Not running on the main thread — skip (atexit still applies).
            pass

    streamer = _StreamingFrameSource(
        fixed_session=fixed_session,
        body_indices=body_indices,
        sim_dt=sim_dt_used,
        frame_dt=frame_dt,
        num_display_frames=num_frames,
        n_sim_steps=n_sim_steps,
    )
    return streamer, body_surface_faces


def _build_frame_source(
    bag_verts_zup_m: np.ndarray,
    bag_faces: np.ndarray,
    grip: dict,
    ppfcs_dir: Path,
    job_dir: Path,
    num_frames: int,
    frame_dt: float,
    ppfcs_substeps_per_frame: int,
    *,
    small_pad: bool = False,
) -> tuple:
    """Return ``(frame_source, body_surface_faces)``.

    ``body_surface_faces`` maps each ppfcs body name to its surface-tri array
    (local indices) so the caller can render via log_mesh.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    return _start_ppfcs_streaming(
        bag_verts_zup_m,
        bag_faces,
        grip,
        ppfcs_dir,
        job_dir,
        num_frames,
        frame_dt,
        ppfcs_substeps_per_frame,
        small_pad=small_pad,
    )


# ---------------------------------------------------------------------------
# Newton visual model (FR3 robot + finger pads + ground)
# ---------------------------------------------------------------------------


def _quat_to_vec4(quat: wp.quat) -> wp.vec4:
    """Convert a Warp quaternion to a vec4 without changing xyzw ordering."""
    return wp.vec4(quat[0], quat[1], quat[2], quat[3])


def _gripper_frac_from_closed_width_cm(closed_width_cm: float) -> float:
    """Map a target closed gap [cm] to FR3 gripper fraction (0=open, 1=closed)."""
    finger_q_at_close = max(0.5 * 0.01 * float(closed_width_cm), 0.0)
    frac = 1.0 - finger_q_at_close / _FINGER_OPEN_Q
    return float(np.clip(frac, 0.0, 1.0))


def _build_visual_model(
    pad_hx_m: float,
    pad_hy_m: float,
    pad_hz_m: float,
    interior_init_q: dict | None = None,
) -> tuple[newton.Model, int, int, int, dict[str, int]]:
    """Build the Newton replay model: FR3 robot + interior rigid bodies + ground.

    `interior_init_q` maps each interior body name (sphere/box/capsule) to a
    7-vector ``[x, y, z, qx, qy, qz, qw]`` for that body's rest transform.

    Returns:
        (model, ee_body_index, left_finger_body, right_finger_body, interior_body_idx)
    """
    interior_init_q = interior_init_q or {}
    builder = newton.ModelBuilder(gravity=0.0)
    asset_path = newton.utils.download_asset("franka_emika_panda")
    builder.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform(_FR3_BASE_M, wp.quat_identity()),
        floating=False,
        scale=1.0,
        enable_self_collisions=False,
        parse_visuals_as_colliders=True,
    )
    builder.joint_q[:9] = [*_FR3_INIT_Q, _FINGER_OPEN_Q, _FINGER_OPEN_Q]

    left_finger_body = next(
        i for i, label in enumerate(builder.body_label) if label.endswith("fr3_leftfinger")
    )
    right_finger_body = next(
        i for i, label in enumerate(builder.body_label) if label.endswith("fr3_rightfinger")
    )
    ee_body_index = next(
        i for i, label in enumerate(builder.body_label) if label.endswith("fr3_hand")
    )

    # Visual finger pads attached to the FR3 fingers.  These render via
    # log_state (Newton body palette → matches the pyansys/radioss colour
    # scheme).  The ppfcs kinematic tet pad bodies follow the same trajectory
    # below them and are NOT log_mesh'd separately.
    pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.001, mu=0.5, ke=1.0e4, kd=1.0)
    pad_xform = wp.transform(wp.vec3(*_FINGER_PAD_OFFSET_M), wp.quat_identity())
    builder.add_shape_box(
        body=left_finger_body, xform=pad_xform,
        hx=pad_hx_m, hy=pad_hy_m, hz=pad_hz_m,
        cfg=pad_cfg, label="left_finger_pad",
    )
    builder.add_shape_box(
        body=right_finger_body, xform=pad_xform,
        hx=pad_hx_m, hy=pad_hy_m, hz=pad_hz_m,
        cfg=pad_cfg, label="right_finger_pad",
    )

    # Note: do NOT call builder.approximate_meshes("convex_hull") here — that
    # would replace the URDF visual meshes (.dae files) with simplified convex
    # hulls and the robot would render as chunky collision-style geometry.
    # `parse_visuals_as_colliders=True` above already gives us the high-quality
    # visual meshes for both physics and rendering.

    # ---- Interior rigid bodies (sphere / box / capsule) --------------------
    # These are visual-only proxies updated each frame from the corresponding
    # ppfcs tet body's centroid.  Sizes match the radioss/pyansys lift example
    # so the contents look identical across all three backends; default Newton
    # body colours give each shape a distinct hue.
    interior_body_idx: dict[str, int] = {}
    rigid_inertia = np.eye(3, dtype=np.float64) * 1.0e-4
    content_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.001, mu=0.4, ke=1.0e4, kd=10.0
    )

    def _add_interior(name: str, add_shape):
        init_q = interior_init_q.get(
            name, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        body_idx = builder.add_body(
            xform=wp.transform(tuple(init_q[:3]), tuple(init_q[3:7])),
            mass=0.1,
            inertia=rigid_inertia,
            lock_inertia=True,
        )
        add_shape(body_idx)
        interior_body_idx[name] = body_idx

    _add_interior(
        "sphere",
        lambda b: builder.add_shape_sphere(body=b, radius=_SPHERE_RADIUS_M, cfg=content_cfg),
    )
    _add_interior(
        "box",
        lambda b: builder.add_shape_box(
            body=b, hx=_BOX_HALF_EXTENT_M, hy=_BOX_HALF_EXTENT_M, hz=_BOX_HALF_EXTENT_M,
            cfg=content_cfg,
        ),
    )
    _add_interior(
        "capsule",
        lambda b: builder.add_shape_capsule(
            body=b, radius=_CAPSULE_RADIUS_M, half_height=_CAPSULE_HALF_HEIGHT_M,
            cfg=content_cfg,
        ),
    )

    builder.add_ground_plane()
    builder.color()

    return (
        builder.finalize(),
        ee_body_index, left_finger_body, right_finger_body,
        interior_body_idx,
    )


class _Fr3IkPlayer:
    """Drive FR3 joint_q each frame via IK to follow a grip→hold→lift trajectory."""

    _LIFT_WAYPOINT_INDEX = 2

    def __init__(
        self,
        model: newton.Model,
        ee_body_index: int,
        frame_dt_s: float,
        closed_width_cm: float,
    ):
        self.model = model
        self.state = model.state()
        self._ee_body_index = ee_body_index
        self.frame_dt = float(frame_dt_s)
        self._closed_frac = _gripper_frac_from_closed_width_cm(closed_width_cm)
        self._gripper_frac = 0.0

        # Initial FK to read EE pose (used to seed the IK solver target).
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        ee_tf = wp.transform(*self.state.body_q.numpy()[ee_body_index])
        self._pos_obj = ik.IKObjectivePosition(
            link_index=ee_body_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=ee_body_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
        )
        self._joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
        )
        self._joint_q_ik = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
        self._ik_solver = ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=[self._pos_obj, self._rot_obj, self._joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # Phase plan: open hold, close, hold closed, lift.  Each entry is
        # (target EE position, duration in s, gripper fraction at end of phase).
        self._waypoints = [
            (wp.vec3(0.0, 0.0, _GRAB_EE_Z_M), _PHASE_OPEN_S, 0.0),
            (wp.vec3(0.0, 0.0, _GRAB_EE_Z_M), _PHASE_CLOSE_S, self._closed_frac),
            (wp.vec3(0.0, 0.0, _GRAB_EE_Z_M), _PHASE_HOLD_S,  self._closed_frac),
            (wp.vec3(0.0, 0.0, _LIFT_EE_Z_M), _PHASE_LIFT_S,  self._closed_frac),
        ]
        self._waypoint_index = 0
        self._time_in_waypoint = 0.0
        self._initialise()

    @staticmethod
    def _smoothstep(alpha: float) -> float:
        a = float(np.clip(alpha, 0.0, 1.0))
        return a * a * (3.0 - 2.0 * a)

    def _initialise(self):
        start_pos, _, start_frac = self._waypoints[0]
        target_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
        self._pos_obj.set_target_positions(wp.array([start_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_quat_to_vec4(target_quat)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=64)
        self._gripper_frac = float(start_frac)
        self._apply_pose()

    def _apply_pose(self):
        joint_q_np = self.model.joint_q.numpy().copy()
        joint_q_np[:7] = self._joint_q_ik.numpy()[0][:7]
        finger_q = _FINGER_OPEN_Q * (1.0 - float(self._gripper_frac))
        joint_q_np[7] = finger_q
        joint_q_np[8] = finger_q
        joint_q_wp = wp.array(joint_q_np, dtype=float, device=self.model.device)
        self.model.joint_q.assign(joint_q_wp)
        self.state.joint_q.assign(joint_q_wp)
        self.model.joint_qd.zero_()
        self.state.joint_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def advance(self):
        """Step the trajectory by `frame_dt` and update model state via IK + FK.

        Uses the same convention as `example_kfc_bag_lift_pyansys` /
        `_radioss` / `_lift`: each waypoint's *duration* is the time spent
        transitioning from the *current* waypoint's end-state to the
        *next* waypoint's end-state.  In the canonical 4-waypoint plan
        (OPEN, CLOSE, HOLD, LIFT) this means waypoint 0 closes the
        gripper while EE stays at GRAB, waypoint 1 holds at GRAB,
        waypoint 2 (`_LIFT_WAYPOINT_INDEX`) rises GRAB→LIFT smoothstep,
        and waypoint 3 holds at LIFT.
        """
        self._time_in_waypoint += self.frame_dt
        current = self._waypoints[self._waypoint_index]
        nxt = self._waypoints[
            min(self._waypoint_index + 1, len(self._waypoints) - 1)
        ]
        duration = max(float(current[1]), 1.0e-6)
        alpha = min(self._time_in_waypoint / duration, 1.0)
        phase_alpha = (
            self._smoothstep(alpha)
            if self._waypoint_index == self._LIFT_WAYPOINT_INDEX
            else alpha
        )

        target_pos = current[0] * (1.0 - phase_alpha) + nxt[0] * phase_alpha
        self._gripper_frac = float(current[2]) * (1.0 - phase_alpha) + float(nxt[2]) * phase_alpha
        target_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
        self._pos_obj.set_target_positions(wp.array([target_pos], dtype=wp.vec3))
        self._rot_obj.set_target_rotations(wp.array([_quat_to_vec4(target_quat)], dtype=wp.vec4))
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=32)

        if alpha >= 1.0 and self._waypoint_index < len(self._waypoints) - 1:
            self._waypoint_index += 1
            self._time_in_waypoint = 0.0

        self._apply_pose()


# ---------------------------------------------------------------------------
# Capture helpers (same pattern as pyansys example)
# ---------------------------------------------------------------------------

def _duration_from_capture_frames(capture_frames: int, frame_dt: float) -> float:
    if capture_frames < 2:
        raise ValueError("`--capture-frames` must be at least 2.")
    return float(capture_frames - 1) * float(frame_dt)


# ---------------------------------------------------------------------------
# Example class
# ---------------------------------------------------------------------------

class Example:
    """Simulate a KFC bag lift with ppf-contact-solver and replay in Newton."""

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.description = (
            "Simulate a KFC bag lift using ppf-contact-solver (PPF-CTS) as the "
            "contact and deformation backend.  The bag is a deformable triangular "
            "shell; two animated half-space walls represent the finger pads."
        )
        parser.set_defaults(num_frames=_DEFAULT_NUM_FRAMES)
        parser.add_argument(
            "--ppfcs-dir",
            type=str,
            default=_default_ppfcs_dir(),
            help=(
                "Path to the compiled ppf-contact-solver repository root.  "
                f"Defaults to the {_PPFCS_DIR_ENV} environment variable or the "
                "repo's `ppf-contact-solver` submodule when present."
            ),
        )
        parser.add_argument(
            "--job-dir",
            type=str,
            default=str(_DEFAULT_JOB_DIR),
            help="Directory used for ppfcs solver output.",
        )
        _add_proxy_mesh_arguments(parser)
        parser.add_argument(
            "--closed-width-cm",
            type=float,
            default=_DEFAULT_CLOSED_WIDTH_CM,
            help="Final finger-pad gap [cm] when fully closed.",
        )
        _add_capture_arguments(
            parser,
            replay_help="Capture rendered frames and build a replay video or gif.",
            capture_frames_default=_DEFAULT_NUM_FRAMES,
            include_save_mp4=False,
        )
        parser.add_argument(
            "--small-pad",
            action="store_true",
            help=(
                "Use a smaller finger-pad visual (≈20 %% contact area).  "
                "The grip walls are infinite planes so this affects the "
                "visualisation only, not the simulation."
            ),
        )
        parser.add_argument(
            "--ppfcs-substeps-per-frame",
            type=int,
            default=_DEFAULT_PPFCS_SUBSTEPS_PER_FRAME,
            help=(
                "Number of ppfcs solver substeps per replay frame.  "
                "ppfcs dt is set to `_FRAME_DT / this value`."
            ),
        )
        return parser

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self._frame_index = -1
        self.frame_dt = _FRAME_DT
        self.ppfcs_substeps_per_frame = int(args.ppfcs_substeps_per_frame)
        if self.ppfcs_substeps_per_frame < 1:
            raise ValueError("--ppfcs-substeps-per-frame must be >= 1.")
        closed_width_cm = float(args.closed_width_cm)
        job_dir = Path(args.job_dir)

        ppfcs_dir_str = str(args.ppfcs_dir).strip()
        if not ppfcs_dir_str:
            raise ValueError(
                f"--ppfcs-dir (or {_PPFCS_DIR_ENV} env var) is required when running "
                "the solver. Set it to the ppf-contact-solver repo root."
            )
        ppfcs_dir = Path(ppfcs_dir_str) if ppfcs_dir_str else Path(".")

        # Resolve capture-frames vs full simulation length
        req_frames = int(args.capture_frames)
        full_frames = _DEFAULT_NUM_FRAMES
        if req_frames == full_frames:
            self.capture_frames = int(math.ceil(_TOTAL_DURATION_S / self.frame_dt)) + 1
        else:
            self.capture_frames = req_frames
        total_sim_s = _duration_from_capture_frames(self.capture_frames, self.frame_dt)
        self.times_s = np.arange(
            0.0, total_sim_s + 0.5 * self.frame_dt, self.frame_dt, dtype=np.float32
        )
        if len(self.times_s) > self.capture_frames:
            self.times_s = self.times_s[: self.capture_frames]
        self.capture_frames = len(self.times_s)

        _configure_capture_common(
            self,
            capture_replay=bool(getattr(args, "capture_replay", False)),
            capture_frames=self.capture_frames,
            capture_fps=int(getattr(args, "capture_fps", 60)),
            capture_dir=str(getattr(args, "capture_dir", "outputs/replay_capture")),
            capture_format=str(getattr(args, "capture_format", "mp4")),
            # Preserve the previous ppfcs behavior: PNGs are written before the
            # frame is counted, so video finalization sees a complete sequence.
            capture_background_writes=False,
        )
        if self.capture_replay:
            print(f"{_LOG_PREFIX} Capture directory: {self.capture_dir.resolve()}")
            print(f"{_LOG_PREFIX}   PNG frames will be written here as the solver "
                  "produces each frame; replay.mp4 is stitched at the end.")

        # Log resolved paths up-front so it's easy to find solver output and
        # captures in future debugging sessions (the defaults are cwd-relative).
        print(f"{_LOG_PREFIX} Job directory: {job_dir.resolve()}")

        print(f"{_LOG_PREFIX} Loading KFC bag mesh...")
        full_verts_zup_m, full_faces = _load_kfc_mesh_zup()

        # Decimate for simulation
        target_faces = int(args.target_faces)
        proxy_mode = str(args.proxy_mode)
        shell_verts_zup_m, shell_faces = _decimate_mesh(
            full_verts_zup_m,
            full_faces,
            target_faces,
            proxy_mode,
            ppfcs_dir,
        )
        print(
            f"{_LOG_PREFIX} Sim/collision mesh (decimated): "
            f"{len(shell_verts_zup_m)} verts, {len(shell_faces)} faces "
            f"(target={target_faces}, proxy_mode={proxy_mode})"
        )
        print(
            f"{_LOG_PREFIX} Render mesh (full-res): {len(full_verts_zup_m)} verts, "
            f"{len(full_faces)} faces"
        )

        # Barycentric coupling: project each full-res render vertex onto the
        # nearest decimated triangle, plus a fixed displacement so the rest
        # state matches the original mesh exactly.  Per-frame we just remap.
        print(f"{_LOG_PREFIX} Building barycentric render→sim map...", end=" ", flush=True)
        bary_vi0, bary_vi1, bary_vi2, bary_w = _build_bary_map(
            full_verts_zup_m, shell_verts_zup_m, shell_faces
        )
        rest_proj = (
            shell_verts_zup_m[bary_vi0] * bary_w[:, 0:1]
            + shell_verts_zup_m[bary_vi1] * bary_w[:, 1:2]
            + shell_verts_zup_m[bary_vi2] * bary_w[:, 2:3]
        )
        bary_disp = (full_verts_zup_m - rest_proj).astype(np.float32)
        self._bary_vi0 = bary_vi0
        self._bary_vi1 = bary_vi1
        self._bary_vi2 = bary_vi2
        self._bary_w = bary_w.astype(np.float32)
        self._bary_disp = bary_disp
        self._full_faces_wp = wp.array(
            full_faces.flatten().astype(np.int32), dtype=wp.int32
        )
        self._n_full_verts = len(full_verts_zup_m)
        print("done.")

        grip = _compute_grip_geometry(shell_verts_zup_m, closed_width_cm)
        print(
            f"{_LOG_PREFIX} Grip geometry: pad_hx={grip['pad_hx']*100:.1f} cm, "
            f"z_pad_centre={grip['z_pad_centre']*100:.1f} cm, "
            f"y_closed=±{abs(grip['y_right_closed'])*100:.2f} cm"
        )

        # Build the per-frame source. The ppfcs solver starts here and frames
        # are pulled in `step()` as the solver writes them, so the capture
        # directory fills up incrementally rather than only at the end.
        self._frame_source, self._body_faces_np = _build_frame_source(
            shell_verts_zup_m,
            shell_faces,
            grip,
            ppfcs_dir,
            job_dir,
            self.capture_frames,
            self.frame_dt,
            self.ppfcs_substeps_per_frame,
            small_pad=bool(getattr(args, "small_pad", False)),
        )
        self.bag_frames_zup = None
        self.times_s = self.times_s[: self.capture_frames]
        self._current_bodies: dict[str, np.ndarray] = {}

        # Pre-build wp arrays of surface faces for each body (once).
        self._body_faces_wp: dict[str, "wp.array"] = {}
        for name, F in self._body_faces_np.items():
            if name == "bag":
                continue  # bag uses the full-res render mesh + bary mapping
            self._body_faces_wp[name] = wp.array(
                np.asarray(F, dtype=np.int32).flatten(), dtype=wp.int32
            )

        # Synchronise viewer frame count
        if hasattr(viewer, "num_frames"):
            viewer.num_frames = self.capture_frames
        if hasattr(args, "num_frames"):
            args.num_frames = self.capture_frames

        # Pre-compute pad geometry & store for the FR3 player.
        self._grip = grip
        self._shell_faces_wp = wp.array(
            shell_faces.flatten().astype(np.int32), dtype=wp.int32
        )

        # Finger-pad half extents (--small-pad scales x and z, leaves y/thickness).
        small_pad = bool(getattr(args, "small_pad", False))
        vis_scale = _SMALL_PAD_SCALE if small_pad else 1.0
        pad_hx_m = float(grip["pad_hx"]) * vis_scale
        pad_hy_m = _PAD_HALF_THICKNESS_M
        pad_hz_m = _PAD_HALF_HEIGHT_M * vis_scale

        # Initial poses for the interior visual rigid bodies — these render
        # with default Newton body colours (one hue per body) and are updated
        # each frame from the corresponding ppfcs tet body's centroid.
        interior_init_q = _fit_bag_contents(shell_verts_zup_m)

        # Build the Newton replay model: FR3 robot + interior bodies + ground.
        (
            self.model,
            self._ee_body_index,
            self._left_finger_body,
            self._right_finger_body,
            self._interior_body_idx,
        ) = _build_visual_model(pad_hx_m, pad_hy_m, pad_hz_m, interior_init_q)
        self._robot = _Fr3IkPlayer(
            self.model,
            ee_body_index=self._ee_body_index,
            frame_dt_s=self.frame_dt,
            closed_width_cm=closed_width_cm,
        )
        self.viz_state = self._robot.state
        self.state_0 = self.viz_state
        self.viewer.set_model(self.model)
        # Suppress the viewer's built-in cloth draw by default.  The shared
        # bag renderer logs either the hi-res bag or proxy mesh explicitly.
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            # Match the camera pose used by example_kfc_bag_lift_pyansys (and
            # radioss/fenicsx variants) so side-by-side comparisons line up.
            self.viewer.set_camera(
                pos=wp.vec3(1.0, -1.0, 0.8), pitch=-10.0, yaw=135.0
            )

        # Per-interior-body local-frame reference verts (Z-up), populated
        # lazily on the first ppfcs frame and used for Procrustes alignment
        # to recover the rigid (R, t) per frame.
        self._interior_init_q_zup = interior_init_q
        self._interior_rest_local: dict[str, np.ndarray] = {}

    def step(self):
        """Advance to the next frame.  In streaming mode this blocks until the
        ppfcs solver has produced the corresponding sim frame, so the viewer
        — and the per-frame PNG capture — runs in lockstep with the solver."""
        if self._frame_index < len(self.times_s) - 1:
            self._frame_index += 1
        else:
            self._frame_index = len(self.times_s) - 1
        self.sim_time = float(self.times_s[self._frame_index])
        # Pull this frame's body positions (blocks until ready in streaming mode).
        result = self._frame_source.fetch(self._frame_index)
        if isinstance(result, dict):
            self._current_bodies = result
        else:
            self._current_bodies = {"bag": result}
        # Advance the FR3 trajectory so the robot pose tracks the bag motion.
        # Frame 0 starts at the pre-grasp pose; only advance from frame 1 on.
        if self._frame_index > 0:
            self._robot.advance()
        self._log_grip_metric()

    def _log_grip_metric(self):
        """Cheap per-frame slip-detection — prints bag-vs-pad vertical drift
        at the gripper.  If the bag is slipping out of the pads, the bag's
        upper-rim verts (those near the pad band at frame 0) drop relative
        to the pad center.  Log every 5 display frames so the user can
        Ctrl-C immediately when slip starts (instead of waiting 70+ frames).

        Output format:
          [grip] f=NN t=T.TTs phase  bag_top_z=.XXX pad_z=.XXX drift=±.XXX cm
        ``drift`` is the change since grip-close (t=0.65 s); positive means
        the bag is rising relative to the pad (good — being held); negative
        means the bag is falling relative to the pad (slip).
        """
        fi = self._frame_index
        if fi < 0 or fi % 5 != 0 and fi != 0:
            return
        bag = self._current_bodies.get("bag")
        if bag is None:
            return
        # Track the upper-rim verts (those at the bag's top 5 % at frame 0).
        if not hasattr(self, "_grip_metric_top_idx"):
            top_z = bag[:, 2].max()
            band_lo = top_z - 0.02  # top 2 cm of bag
            self._grip_metric_top_idx = np.where(bag[:, 2] >= band_lo)[0]
            self._grip_metric_pad_z0: float | None = None
        top_idx = self._grip_metric_top_idx
        if len(top_idx) == 0:
            return
        bag_top_z = float(bag[top_idx, 2].mean())
        pads = []
        for n in ("left_pad", "right_pad"):
            v = self._current_bodies.get(n)
            if v is not None:
                pads.append(float(v[:, 2].mean()))
        pad_z = float(np.mean(pads)) if pads else float("nan")
        # Anchor at the moment grip closes (frame nearest t=0.65 s).
        if (
            self._grip_metric_pad_z0 is None
            and self.sim_time >= _PHASE_OPEN_S + _PHASE_CLOSE_S
        ):
            self._grip_metric_pad_z0 = pad_z - bag_top_z
        if self._grip_metric_pad_z0 is None:
            phase = "open"
            drift_cm_s = "  (pre-grip)"
        else:
            drift_m = (pad_z - bag_top_z) - self._grip_metric_pad_z0
            # Negative drift_m = pad has fallen below bag (impossible) or
            # bag has risen relative to pad.  We want the bag's vertical
            # offset under the pad to STAY CONSTANT — anything else is slip.
            slip_cm = -drift_m * 100.0  # cm; positive = bag dropped
            # Canonical IK waypoint convention (matches pyansys/radioss/lift):
            # 0..t_pinch      : fingers close, EE at GRAB
            # t_pinch..t_lift0: pinch settle, EE+fingers at GRAB
            # t_lift0..t_post : EE rises GRAB→LIFT smoothstep
            # t_post..        : EE static at LIFT
            t_pinch = _PHASE_OPEN_S
            t_lift0 = t_pinch + _PHASE_CLOSE_S
            t_post = t_lift0 + _PHASE_HOLD_S
            if self.sim_time < t_pinch:
                phase = "close"
            elif self.sim_time < t_lift0:
                phase = "pinch"
            elif self.sim_time < t_post:
                phase = "lift "
            else:
                phase = "post "
            slip_marker = " ⚠ SLIP" if abs(slip_cm) > 0.5 else ""
            drift_cm_s = f"  bag_drop={slip_cm:+.2f} cm{slip_marker}"
        print(
            f"[grip] f={fi:3d} t={self.sim_time:5.2f}s {phase} "
            f"bag_top_z={bag_top_z:.3f}m pad_z={pad_z:.3f}m{drift_cm_s}",
            flush=True,
        )

    def render(self):
        """Render the FR3 robot, finger pads, ground, the bag, the interior
        rigid bodies (as Newton shapes — proper coloring), and the ppfcs tet
        finger pads.  The bag uses the full-res barycentric mapping."""
        fi = max(self._frame_index, 0)

        bodies = self._current_bodies
        if not bodies:
            if self.bag_frames_zup is not None:
                bodies = {"bag": self.bag_frames_zup[fi]}
            else:
                fetched = self._frame_source.fetch(fi)
                bodies = fetched if isinstance(fetched, dict) else {"bag": fetched}
            self._current_bodies = bodies

        sim_bag = bodies["bag"]
        full_bag = (
            sim_bag[self._bary_vi0] * self._bary_w[:, 0:1]
            + sim_bag[self._bary_vi1] * self._bary_w[:, 1:2]
            + sim_bag[self._bary_vi2] * self._bary_w[:, 2:3]
            + self._bary_disp
        ).astype(np.float32)
        bag_pts_wp = wp.array(full_bag, dtype=wp.vec3)
        sim_bag_wp = wp.array(sim_bag.astype(np.float32), dtype=wp.vec3)

        # Update interior rigid-body transforms from the corresponding ppfcs
        # tet body's verts via Procrustes alignment — captures both
        # translation and rotation as the body tumbles inside the bag.
        if self._interior_body_idx:
            body_q_np = self.viz_state.body_q.numpy().copy()
            for name, body_idx in self._interior_body_idx.items():
                verts = bodies.get(name)
                if verts is None:
                    continue
                V_curr = verts.astype(np.float64)
                # Lazily capture body-local reference verts on first frame.
                rest_local = self._interior_rest_local.get(name)
                if rest_local is None:
                    init_q = self._interior_init_q_zup[name]
                    R0 = Rotation.from_quat(init_q[3:7]).as_matrix()
                    t0 = init_q[:3]
                    rest_local = (R0.T @ (V_curr - t0).T).T
                    self._interior_rest_local[name] = rest_local
                R, t = _fit_rigid_transform(rest_local, V_curr)
                body_q_np[body_idx, :3] = t
                body_q_np[body_idx, 3:7] = Rotation.from_matrix(R).as_quat()
            self.viz_state.body_q.assign(
                wp.array(body_q_np, dtype=self.viz_state.body_q.dtype)
            )

        # The ppfcs kinematic tet pads are NOT rendered via log_mesh — the
        # box-shape pads attached to the FR3 fingers (rendered via log_state)
        # serve as their visual stand-ins, matching pyansys's colour scheme.
        # Both are driven by the same trajectory and stay co-located.
        _render_bag_meshes(
            self.viewer,
            sim_time=self.sim_time,
            viz_state=self.viz_state,
            full_positions=bag_pts_wp,
            full_indices=self._full_faces_wp,
            proxy_positions=sim_bag_wp,
            proxy_indices=self._shell_faces_wp,
        )
        self._capture_frame()

        # Close the viewer once we have replayed every captured frame so the
        # example process exits without hanging in the GL event loop.
        if self._frame_index >= len(self.times_s) - 1:
            if not self.capture_replay or self.capture_done:
                if hasattr(self.viewer, "close"):
                    self.viewer.close()

    def _capture_frame(self):
        if self._frame_index < 0:
            return
        _capture_replay_frame_common(
            self,
            frame_key=self._frame_index,
            target_frame_count=self.capture_frames,
            close_viewer=False,
        )

    def _finalize_video(self):
        _finalize_replay_video_common(self)

    def cleanup(self):
        """Release any held resources."""
        _finalize_capture_common(self)

    def test_final(self):
        """Basic lift regression check."""
        if self.bag_frames_zup is None:
            self.bag_frames_zup = self._frame_source.all_frames()
        assert len(self.bag_frames_zup) >= 1, "Simulation produced no frames."
        assert np.isfinite(self.bag_frames_zup).all(), "Simulation contains non-finite positions."

        initial_top_z = float(self.bag_frames_zup[0, :, 2].max())
        final_top_z = float(self.bag_frames_zup[-1, :, 2].max())

        # First frame: bag should be near its natural resting height
        assert initial_top_z > 0.10, (
            f"Bag top at frame 0 unexpectedly low: {initial_top_z:.3f} m "
            "(expected > 0.10 m; check ppfcs material parameters)."
        )
        # Last frame (if full lift): bag top should have risen
        total_sim_s = float(self.times_s[-1]) if len(self.times_s) > 0 else 0.0
        if total_sim_s >= (_PHASE_OPEN_S + _PHASE_CLOSE_S + _PHASE_HOLD_S + 0.5):
            assert final_top_z > initial_top_z - 0.05, (
                f"Bag top fell during simulation: {initial_top_z:.3f} → {final_top_z:.3f} m. "
                "The grip may have failed — check friction and closed-width-cm."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.cleanup()
