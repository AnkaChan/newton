# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Procedural generator for a dinner plate (rigid mesh asset).
#
# Used by the H1 dish-washing examples (example_vbd_dish_washing[_all]) as the
# rigid plate the humanoid pinches off a stack, rubs with the sponge, and sets
# down. The plate is a surface of revolution: a 2D (r, z) cross-section lathed
# about the vertical axis into a closed two-manifold (watertight) mesh, so it
# works with the rigid mesh SDF contact path.
#
# SHAPE — uniform-thickness shell with a thin lip. The plate is defined by ONE
# curve, its top surface (_TOP_SURFACE): a deep central well, a steep inner
# wall, and a flat raised rim. The bottom surface is that same curve offset
# straight DOWN by WALL_THICKNESS over the interior; from TAPER_START outward
# the offset blends down to EDGE_THICKNESS, so the shell thins toward the outer
# edge and the visible side of the plate is a thin lip rather than the full
# wall — it reads as a real plate, not a puck.
#
# STACKING — why the offset shell matters. Two identical plates stack when the
# upper one is shifted up until its bottom surface just meets the lower one's
# top surface. Over the un-tapered interior the two surfaces are congruent (a
# pure vertical offset), so the plates rest there and the pitch equals
# WALL_THICKNESS. So:
#
#     stacking rim-to-rim gap  ==  wall thickness
#
# The wall thickness is therefore the single knob for the finger gap between
# stacked plates: make it ~a finger wide and the top plate can be pinched off
# the stack directly, with no need to slide it out of alignment first. The lip
# taper helps here too: near the rim the shell is thinner than the pitch, so
# stacked rims clear by (WALL_THICKNESS - local thickness) instead of touching —
# an open wedge the index can enter. See stacking_pitch(), which measures the
# pitch from the profile.
#
# Outputs (in this directory):
#   dish_plate.obj    - the watertight plate mesh (radius 0.062 m, wall 0.011 m)
#   dish_profile.png  - (with --render) annotated cross-section + stack diagram
#
# Run:
#   python make_dish_plate.py            # write dish_plate.obj + print stats
#   python make_dish_plate.py --render   # also write the cross-section diagram
#                                        #   (needs matplotlib)
###########################################################################

from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Top surface of the plate: (r_fraction, z[m]) from the well centre out to the
# rim edge — a deep well floor, a steep inner wall, and a flat annular rim that
# stays flat all the way to the outer edge. The bottom surface is this curve
# offset down by the local shell thickness, so this single curve defines the
# whole plate.
_TOP_SURFACE = [
    (0.000, 0.0060),  # well centre (top pole)
    (0.250, 0.0064),
    (0.440, 0.0074),
    (0.540, 0.0090),  # well floor edge
    # steep inner wall up to the rim
    (0.610, 0.0140),
    (0.670, 0.0195),
    (0.720, 0.0235),  # top of inner wall (rim inner edge)
    # flat annular rim out to the edge
    (0.790, 0.0250),
    (0.880, 0.0253),
    (0.945, 0.0253),
    (1.000, 0.0253),  # outer top edge
]

# Wall thickness of the shell interior [m]. Two stacked plates rest exactly
# this far apart (pitch == thickness for a pure vertical offset), so this is
# the finger gap between stacked rims. The plate radius is sized for the H1
# hand span with margin for the pinch.
WALL_THICKNESS = 0.011
PLATE_RADIUS = 0.075

# The shell thins from WALL_THICKNESS at TAPER_START (r fraction, the rim's
# inner edge) to EDGE_THICKNESS at the outer edge, so the annular rim and the
# plate's visible side are a thin lip. The lip is what the H1 pinches, so it
# is kept thick enough for a robust fingertip clamp. The taper never governs
# the stacking pitch (the well keeps the full offset), so the finger gap
# stays == WALL_THICKNESS.
# 10 mm: thick enough that sponge particles pressed onto the lip under scrub
# pressure stay on the near side of its median surface (a thinner lip lets
# the contact SDF flip sign and eject the particle through the shell,
# detonating the soft solve)
EDGE_THICKNESS = 0.010
TAPER_START = 0.72


def _shell_thickness(r_frac: float, thickness: float = WALL_THICKNESS, edge: float = EDGE_THICKNESS) -> float:
    """Local vertical shell thickness: full over the interior, smoothstep-blended
    down to ``edge`` between TAPER_START and the outer edge (the thin lip)."""
    if r_frac <= TAPER_START:
        return thickness
    u = (r_frac - TAPER_START) / (1.0 - TAPER_START)
    u = u * u * (3.0 - 2.0 * u)
    return thickness + (edge - thickness) * u


def dish_profile(radius: float = PLATE_RADIUS, thickness: float = WALL_THICKNESS) -> np.ndarray:
    """(r, z) cross-section from the top-centre pole to the bottom-centre pole.

    The plate is an offset shell: the bottom surface is the top surface
    dropped by the local shell thickness (``thickness`` over the interior,
    tapering to EDGE_THICKNESS at the rim), joined by a short vertical outer
    wall. r runs 0 -> radius over the top, holds at the outer wall, then
    radius -> 0 back under the bottom. The whole profile is shifted so the
    lowest point sits at z = 0 (the plate rests there)."""
    R = radius
    top = [(r * R, z) for r, z in _TOP_SURFACE]
    # bottom: the top surface reversed (r: R -> 0), dropped by the local thickness
    bottom = [(r * R, z - _shell_thickness(r, thickness)) for r, z in reversed(_TOP_SURFACE)]
    pts = np.asarray(top + bottom, dtype=np.float64)
    pts[:, 1] -= pts[:, 1].min()  # rest the lowest point on z = 0
    return pts


def stacking_pitch(profile: np.ndarray) -> tuple[float, float]:
    """Vertical offset at which one plate rests on an identical one below it.

    Splits the (r, z) profile into its top surface (r rising 0 -> R) and bottom
    surface (r falling R -> 0), then finds the smallest upward shift of the upper
    plate so its bottom surface never dips below the lower plate's top surface:
    ``pitch = max_r [ z_top(r) - z_bottom(r) ]``. For the offset shell this is
    just the wall thickness. Returns (pitch, r_governing)."""
    r = profile[:, 0]
    z = profile[:, 1]
    i_max = int(np.argmax(r))  # apex of the outer wall splits top from bottom
    r_top, z_top = r[: i_max + 1], z[: i_max + 1]
    r_bot, z_bot = r[i_max:][::-1], z[i_max:][::-1]  # reverse to ascending r
    rs = np.linspace(0.0, float(r.max()), 400)
    gap = np.interp(rs, r_top, z_top) - np.interp(rs, r_bot, z_bot)
    k = int(np.argmax(gap))
    return float(gap[k]), float(rs[k])


def make_dish_mesh(radius: float = PLATE_RADIUS, segments: int = 64):
    """Return (vertices [N,3], indices [M*3]) for a watertight plate mesh."""
    profile = dish_profile(radius)
    interior = profile[1:-1]  # rings; first/last profile points are the poles
    n_rings = len(interior)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)

    verts = [[0.0, 0.0, profile[0, 1]]]  # top pole (index 0)
    ring0 = len(verts)
    for r, z in interior:
        for s in range(segments):
            verts.append([r * cos[s], r * sin[s], z])
    bot_pole = len(verts)
    verts.append([0.0, 0.0, profile[-1, 1]])
    verts = np.asarray(verts, dtype=np.float32)

    def idx(ring, s):
        return ring0 + ring * segments + (s % segments)

    faces: list[int] = []
    # top pole fan (CCW from outside, normals outward/up)
    for s in range(segments):
        faces += [0, idx(0, s), idx(0, s + 1)]
    # quad bands between successive rings (two triangles each)
    for ri in range(n_rings - 1):
        for s in range(segments):
            a, b = idx(ri, s), idx(ri, s + 1)
            c, d = idx(ri + 1, s), idx(ri + 1, s + 1)
            faces += [a, c, b, b, c, d]
    # bottom pole fan
    for s in range(segments):
        faces += [bot_pole, idx(n_rings - 1, s + 1), idx(n_rings - 1, s)]

    return verts, np.asarray(faces, dtype=np.int32)


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals."""
    tris = faces.reshape(-1, 3)
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    lens = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.clip(lens, 1e-12, None)


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray, header: str = "") -> None:
    """Write a triangle mesh to a Wavefront OBJ (with vertex normals)."""
    normals = _vertex_normals(verts, faces)
    tris = faces.reshape(-1, 3)
    with open(path, "w") as f:
        for line in header.splitlines():
            f.write(f"# {line}\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        for a, b, c in tris + 1:  # OBJ is 1-indexed
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def _mesh_stats(verts, faces):
    tris = faces.reshape(-1, 3)
    v = verts[tris]
    e0 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
    e1 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
    e2 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)

    def angle(a, b, c):  # min interior angle per triangle via law of cosines
        cosv = np.clip((a**2 + b**2 - c**2) / (2 * a * b + 1e-12), -1, 1)
        return np.degrees(np.arccos(cosv))

    ang = np.minimum(np.minimum(angle(e0, e1, e2), angle(e1, e2, e0)), angle(e2, e0, e1))
    return len(verts), len(tris), float(ang.min()), float(np.median(ang))


def _render(profile: np.ndarray) -> None:
    """Write an annotated cross-section diagram beside the obj (matplotlib).

    Left: the plate's (r, z) shell profile mirrored to a full cross-section.
    Right: three plates stacked at the designed pitch, showing the rim gap that
    equals the wall thickness."""
    import matplotlib  # noqa: PLC0415  (render-only dep, kept lazy)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    r, z = profile[:, 0], profile[:, 1]
    full_r = np.concatenate([-r[::-1], r])  # mirror across the axis
    full_z = np.concatenate([z[::-1], z])
    pitch, r_at = stacking_pitch(profile)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0.fill(full_r * 1000, full_z * 1000, facecolor="#e9e4d6", edgecolor="#555", lw=1.2)
    ax0.set_title("plate cross-section (offset shell, rim tapers to a thin lip)")
    ax0.set_xlabel("r [mm]")
    ax0.set_ylabel("z [mm]")
    ax0.set_aspect("equal")
    ax0.grid(True, alpha=0.3)

    for i in range(3):
        ax1.fill(full_r * 1000, (full_z + i * pitch) * 1000, facecolor="#e9e4d6", edgecolor="#555", lw=1.0)
    ax1.annotate(
        f"rim gap = wall = {pitch * 1000:.1f} mm",
        xy=(r_at * 1000, (z.max() + 0.5 * pitch) * 1000),
        xytext=(r_at * 1000 + 20, (z.max() + 1.5 * pitch) * 1000),
        arrowprops={"arrowstyle": "->", "color": "#c0392b"},
        color="#c0392b",
    )
    ax1.set_title("three plates stacked at the designed pitch")
    ax1.set_xlabel("r [mm]")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "dish_profile.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="also write review/stack PNGs")
    args, _ = ap.parse_known_args()

    verts, faces = make_dish_mesh()
    nv, nt, min_ang, med_ang = _mesh_stats(verts, faces)
    pitch, r_at = stacking_pitch(dish_profile())
    print(f"dish mesh: {nv} verts, {nt} tris, min angle {min_ang:.1f} deg, median {med_ang:.1f} deg")
    print(f"radius {PLATE_RADIUS} m, wall {WALL_THICKNESS} m, stacking pitch {pitch * 1000:.1f} mm")

    header = (
        "Dinner plate (surface of revolution): offset shell — a deep well and\n"
        "raised rim, bottom offset straight down by the wall thickness, tapering\n"
        "to a thin lip at the outer edge. Watertight two-manifold. Generated by\n"
        f"make_dish_plate.py. radius {PLATE_RADIUS} m, wall {WALL_THICKNESS} m,\n"
        f"lip {EDGE_THICKNESS} m, 64 segments. Stacks with a rim gap == the wall\n"
        "thickness."
    )
    obj_path = os.path.join(HERE, "dish_plate.obj")
    write_obj(obj_path, verts, faces, header)
    print(f"wrote {obj_path}")

    if args.render:
        _render(dish_profile())
